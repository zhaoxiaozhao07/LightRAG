from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator

from lightrag.api.job_service import JobService
from lightrag.api.kb_service import KnowledgeBaseRecord, KnowledgeBaseService, utc_now_iso
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry
from lightrag.api.llm_json_utils import LLMJsonError, call_llm_json
from lightrag.api.metadata_store import DocumentRecord, JobRecord
from lightrag.constants import DEFAULT_QUERY_PRIORITY
from lightrag.utils import logger
from lightrag.utils_pipeline import strip_lightrag_doc_prefix

PROFILE_SCHEMA_VERSION = 1
DOC_PROFILE_METADATA_KEY = "agent_doc_profile"
KB_AUTO_PROFILE_METADATA_KEY = "agent_auto_profile"
# Dirty markers live under their own metadata key so a completing refresh
# writing the profile payload can never clobber a concurrent dirty flag
# (the flag is only cleared when it predates the refresh snapshot).
KB_AUTO_PROFILE_DIRTY_METADATA_KEY = "agent_auto_profile_dirty"
# Cached per-group digest, stored on the first document (creation order) of
# each aggregation bucket; keyed by a hash of member ids + profile versions.
GROUP_PROFILE_METADATA_KEY = "agent_group_profile"
MANUAL_DESCRIPTION_KEY = "agent_description"
MANUAL_TAGS_KEY = "agent_tags"
MANUAL_PRIORITY_KEY = "agent_priority"
# Manual overrides for the routing-critical auto fields: a wrong generated
# negative_scope can systematically steer the Agent away from the right KB,
# so operators need a direct correction path (non-empty manual wins).
MANUAL_DOMAINS_KEY = "agent_domains"
MANUAL_SAMPLE_QUESTIONS_KEY = "agent_sample_questions"
MANUAL_NEGATIVE_SCOPE_KEY = "agent_negative_scope"
# Per-refresh cap on PROFILE LLM document calls; remaining stale documents are
# reported as pending and picked up by a chained refresh job.
MAX_PROFILE_DOCUMENTS = 24
MAX_DOC_SAMPLE_CHARS = 12000
# KB aggregation modes by profiled-document count N:
# - direct  (N <= MAX_PROFILE_DOCS_FOR_AGGREGATION): every profile verbatim.
# - sampled (N > cap, backfill still pending): stride-sampled profiles across
#   the whole KB — transitional, avoids re-hashing shifting group buckets.
# - grouped (N > cap, backfill converged): map-reduce — documents bucketed in
#   creation order, one cached PROFILE-LLM digest per bucket, final call
#   aggregates the digests. Append-only ingest invalidates only the tail
#   bucket, so steady-state cost per change is O(1) group calls.
MAX_PROFILE_DOCS_FOR_AGGREGATION = 128
AGGREGATION_GROUP_SIZE = MAX_PROFILE_DOCS_FOR_AGGREGATION
MAX_GROUPS_FOR_AGGREGATION = 64
MAX_PROFILE_STAT_ITEMS = 40
MAX_DOCUMENT_PROFILES_IN_KB_METADATA = 8
PROFILE_LLM_ATTEMPTS = 3
_KB_PROFILE_METADATA_BUDGET_BYTES = 12 * 1024
_ACTIVE_JOB_STATUSES = ("queued", "running", "retrying")
# An active agent_profile job whose last update is older than this is treated
# as wedged (e.g. crashed mid-run without orphan recovery) and no longer
# suppresses new refresh jobs.
ACTIVE_PROFILE_JOB_STALE_SECONDS = 3600.0

DOCUMENT_PROFILE_SYSTEM_PROMPT = """
You generate a compact routing profile for one knowledge-base document.
Return strict JSON only. Do not include markdown or chain-of-thought.
The profile is used only to decide which KB should be searched by an Agent.
""".strip()

GROUP_PROFILE_SYSTEM_PROMPT = """
You merge a batch of document routing profiles from ONE knowledge base into a
single compact group digest. Return strict JSON only. Do not include markdown
or chain-of-thought. Preserve topical coverage: mention every distinct theme
present in the batch, not only the dominant one.
""".strip()

KB_PROFILE_SYSTEM_PROMPT = """
You aggregate document routing profiles into one compact knowledge-base Agent profile.
Return strict JSON only. Do not include markdown or chain-of-thought.
Focus on what questions this KB is good for, not a full content summary.
""".strip()


def _clip_text(value: Any, max_chars: int) -> str:
    return str(value if value is not None else "").strip()[:max_chars]


def _clip_text_list(value: Any, *, max_items: int, max_chars: int) -> list[str]:
    """Coerce an LLM-provided value into a bounded list of short strings.

    Truncates instead of rejecting: verbose local models frequently overrun
    schema limits, and a failed profile job is worse than a clipped tag.
    """
    if isinstance(value, str):
        raw: list[Any] = value.split(",")
    elif isinstance(value, list):
        raw = value
    elif value is None:
        raw = []
    else:
        raw = [value]
    cleaned: list[str] = []
    for item in raw:
        text = str(item).strip()[:max_chars]
        if text and text not in cleaned:
            cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    return cleaned


class DocumentAgentProfile(BaseModel):
    type: Literal["document_agent_profile"] = "document_agent_profile"
    summary: str = ""
    tags: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    sample_questions: list[str] = Field(default_factory=list)
    negative_scope: list[str] = Field(default_factory=list)

    @field_validator("summary", mode="before")
    @classmethod
    def _clip_summary(cls, value: Any) -> str:
        return _clip_text(value, 1000)

    @field_validator("tags", "domains", mode="before")
    @classmethod
    def _clip_tags(cls, value: Any) -> list[str]:
        return _clip_text_list(value, max_items=20, max_chars=120)

    @field_validator("sample_questions", "negative_scope", mode="before")
    @classmethod
    def _clip_questions(cls, value: Any) -> list[str]:
        return _clip_text_list(value, max_items=12, max_chars=120)


class DocumentGroupProfile(BaseModel):
    type: Literal["document_group_profile"] = "document_group_profile"
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    sample_questions: list[str] = Field(default_factory=list)
    negative_scope: list[str] = Field(default_factory=list)

    @field_validator("description", mode="before")
    @classmethod
    def _clip_description(cls, value: Any) -> str:
        return _clip_text(value, 1200)

    @field_validator("tags", "domains", mode="before")
    @classmethod
    def _clip_tags(cls, value: Any) -> list[str]:
        return _clip_text_list(value, max_items=24, max_chars=120)

    @field_validator("sample_questions", "negative_scope", mode="before")
    @classmethod
    def _clip_questions(cls, value: Any) -> list[str]:
        return _clip_text_list(value, max_items=8, max_chars=160)


class KBAgentAutoProfile(BaseModel):
    type: Literal["kb_agent_auto_profile"] = "kb_agent_auto_profile"
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    sample_questions: list[str] = Field(default_factory=list)
    negative_scope: list[str] = Field(default_factory=list)

    @field_validator("description", mode="before")
    @classmethod
    def _clip_description(cls, value: Any) -> str:
        return _clip_text(value, 2000)

    @field_validator("tags", "domains", mode="before")
    @classmethod
    def _clip_tags(cls, value: Any) -> list[str]:
        return _clip_text_list(value, max_items=40, max_chars=160)

    @field_validator("sample_questions", "negative_scope", mode="before")
    @classmethod
    def _clip_questions(cls, value: Any) -> list[str]:
        return _clip_text_list(value, max_items=20, max_chars=160)


class AgentProfileUpdateRequest(BaseModel):
    agent_description: str | None = Field(default=None, max_length=2000)
    agent_tags: list[str] | str | None = None
    agent_priority: int | None = None
    agent_domains: list[str] | str | None = None
    agent_sample_questions: list[str] | str | None = None
    agent_negative_scope: list[str] | str | None = None

    @field_validator("agent_description", mode="after")
    @classmethod
    def _strip_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


@dataclass(slots=True)
class AgentProfileRefreshResult:
    kb_id: str
    status: str
    auto_profile: dict[str, Any]
    document_profiles_generated: int
    document_profiles_reused: int
    pending_document_profiles: int = 0
    document_profiles_failed: int = 0
    refresh_started_at: str = ""
    aggregation_mode: str = "direct"
    group_profiles_generated: int = 0
    group_profiles_reused: int = 0


def _coerce_tags(value: Any) -> list[str]:
    return _clip_text_list(value, max_items=64, max_chars=120)


def _coerce_priority(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _auto_profile_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    auto = metadata.get(KB_AUTO_PROFILE_METADATA_KEY)
    return dict(auto) if isinstance(auto, dict) else {}


def _dirty_flag_from_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
    flag = metadata.get(KB_AUTO_PROFILE_DIRTY_METADATA_KEY)
    return dict(flag) if isinstance(flag, dict) else None


def _manual_profile_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "agent_description": str(metadata.get(MANUAL_DESCRIPTION_KEY) or "").strip(),
        "agent_tags": _coerce_tags(metadata.get(MANUAL_TAGS_KEY)),
        "agent_priority": _coerce_priority(metadata.get(MANUAL_PRIORITY_KEY)),
        "agent_domains": _coerce_tags(metadata.get(MANUAL_DOMAINS_KEY)),
        "agent_sample_questions": _coerce_tags(
            metadata.get(MANUAL_SAMPLE_QUESTIONS_KEY)
        ),
        "agent_negative_scope": _coerce_tags(
            metadata.get(MANUAL_NEGATIVE_SCOPE_KEY)
        ),
    }


def effective_agent_profile(record: KnowledgeBaseRecord) -> dict[str, Any]:
    metadata = record.metadata or {}
    manual = _manual_profile_from_metadata(metadata)
    auto = _auto_profile_from_metadata(metadata)
    status = str(auto.get("status") or "missing")
    if _dirty_flag_from_metadata(metadata) is not None and status == "ready":
        status = "dirty"
    return {
        "kb_id": record.id,
        "name": record.name,
        "description": record.description or "",
        "agent_description": manual["agent_description"]
        or str(auto.get("description") or "").strip(),
        "agent_tags": manual["agent_tags"] or _coerce_tags(auto.get("tags")),
        "agent_priority": manual["agent_priority"],
        "agent_auto_profile_status": status,
        "agent_profile_updated_at": auto.get("updated_at"),
        "agent_profile_domains": manual["agent_domains"]
        or _coerce_tags(auto.get("domains")),
        "agent_profile_sample_questions": manual["agent_sample_questions"]
        or _coerce_tags(auto.get("sample_questions")),
        "agent_profile_negative_scope": manual["agent_negative_scope"]
        or _coerce_tags(auto.get("negative_scope")),
    }


def agent_profile_response(record: KnowledgeBaseRecord) -> dict[str, Any]:
    metadata = record.metadata or {}
    return {
        "kb_id": record.id,
        "manual": _manual_profile_from_metadata(metadata),
        "auto": _auto_profile_from_metadata(metadata),
        "dirty": _dirty_flag_from_metadata(metadata),
        "effective": effective_agent_profile(record),
    }


def _sample_text(text: str, max_chars: int = MAX_DOC_SAMPLE_CHARS) -> str:
    value = (text or "").strip()
    if len(value) <= max_chars:
        return value
    head = max_chars * 2 // 3
    tail = max_chars - head
    return f"{value[:head]}\n\n...\n\n{value[-tail:]}"


def _stable_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_payload_size(value: dict[str, Any]) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _seconds_since(value: Any) -> float | None:
    parsed = _parse_iso_datetime(value)
    if parsed is None:
        return None
    return (datetime.now(timezone.utc) - parsed).total_seconds()


def _is_newer(value: Any, reference: Any) -> bool:
    left = _parse_iso_datetime(value)
    right = _parse_iso_datetime(reference)
    if left is None or right is None:
        return False
    return left > right


def _frequency_stats(values: list[str], cap: int = MAX_PROFILE_STAT_ITEMS) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter(v for v in values if v)
    return [
        {"value": value, "count": count}
        for value, count in counter.most_common(cap)
    ]


def _stride_sample(items: list[Any], cap: int) -> list[Any]:
    """Deterministic evenly-spaced sample preserving order (index 0 included).

    Used when a KB has more profiled documents than fit in one aggregation
    prompt: unlike a newest-N cut, the stride covers the whole KB so older
    themes stay represented.
    """
    if len(items) <= cap:
        return list(items)
    step = len(items) / cap
    return [items[min(len(items) - 1, int(i * step))] for i in range(cap)]


def _group_hash(
    documents: list[DocumentRecord], profiles_by_doc: dict[str, dict[str, Any]]
) -> str:
    """Content hash for one aggregation bucket.

    Covers membership AND per-document profile versions (``updated_at``
    changes whenever a profile regenerates), so a cached group digest is
    reused only while every member's profile is unchanged.
    """
    parts = []
    for document in documents:
        profile = profiles_by_doc.get(document.id) or {}
        version = str(
            profile.get("updated_at") or profile.get("source_hash") or ""
        )
        parts.append(f"{document.id}:{version}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _compact_document_profile_entry(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_id": profile.get("document_id"),
        "source_name": profile.get("source_name"),
        "summary": profile.get("summary"),
        "tags": profile.get("tags", []),
        "domains": profile.get("domains", []),
        "sample_questions": (profile.get("sample_questions") or [])[:4],
        "negative_scope": (profile.get("negative_scope") or [])[:4],
    }


def _compact_group_profile_entry(group: dict[str, Any]) -> dict[str, Any]:
    return {
        "group_index": group.get("group_index"),
        "doc_count": group.get("doc_count"),
        "description": group.get("description"),
        "tags": group.get("tags", []),
        "domains": group.get("domains", []),
        "sample_questions": group.get("sample_questions", []),
        "negative_scope": group.get("negative_scope", []),
    }


def _compact_kb_profile_for_metadata(
    profile: KBAgentAutoProfile,
    *,
    job_id: str | None,
    total_ready_documents: int,
    profiled_documents: int,
    pending_documents: int,
    aggregated_documents: list[DocumentRecord],
    profiles_by_doc: dict[str, dict[str, Any]],
    generated: int,
    reused: int,
    aggregation_mode: str = "direct",
    sampled_doc_count: int | None = None,
    group_count: int = 0,
    group_profiles_generated: int = 0,
    group_profiles_reused: int = 0,
    failed_profiles: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    failed_profiles = failed_profiles or []
    source_documents = []
    for document in aggregated_documents[:MAX_DOCUMENT_PROFILES_IN_KB_METADATA]:
        doc_profile = profiles_by_doc.get(document.id) or {}
        source_documents.append(
            {
                "document_id": document.id,
                "source_name": document.source_name[:200],
                "profile_summary": str(doc_profile.get("summary") or "")[:240],
            }
        )
    result = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "type": "kb_agent_auto_profile",
        "status": "ready",
        "description": profile.description[:1600],
        "tags": profile.tags[:32],
        "domains": profile.domains[:32],
        "sample_questions": profile.sample_questions[:12],
        "negative_scope": profile.negative_scope[:12],
        "source_doc_count": total_ready_documents,
        "profiled_doc_count": profiled_documents,
        "pending_document_profiles": pending_documents,
        "aggregation_mode": aggregation_mode,
        "sampled_doc_count": sampled_doc_count
        if sampled_doc_count is not None
        else min(len(aggregated_documents), MAX_PROFILE_DOCS_FOR_AGGREGATION),
        "group_count": group_count,
        "group_profiles_generated": group_profiles_generated,
        "group_profiles_reused": group_profiles_reused,
        "document_profiles_generated": generated,
        "document_profiles_reused": reused,
        "document_profiles_failed": len(failed_profiles),
        "failed_documents": list(failed_profiles[:5]),
        "source_documents": source_documents,
        "updated_at": utc_now_iso(),
        "profile_version": f"agent-profile-v{PROFILE_SCHEMA_VERSION}",
        "job_id": job_id,
        "last_error": None,
    }

    def _over_budget() -> bool:
        return (
            _json_payload_size({KB_AUTO_PROFILE_METADATA_KEY: result})
            > _KB_PROFILE_METADATA_BUDGET_BYTES
        )

    # Shrink progressively: sampled source docs first, then failed-document
    # details, then list fields, then the description itself. The KB metadata
    # store enforces a hard 16KB total budget shared with manual fields, so
    # this must always converge.
    while _over_budget() and result["source_documents"]:
        result["source_documents"].pop()
    while _over_budget() and result["failed_documents"]:
        result["failed_documents"].pop()
    for key in ("sample_questions", "negative_scope", "domains", "tags"):
        while _over_budget() and result[key]:
            result[key].pop()
    for cut in (800, 400, 200):
        if _over_budget():
            result["description"] = result["description"][:cut]
    return result


class AgentProfileService:
    def __init__(
        self,
        *,
        kb_service: KnowledgeBaseService,
        document_service: Any,
        registry: LightRAGInstanceRegistry,
        job_service: JobService | None = None,
        auto_refresh_enabled: bool = True,
        refresh_min_interval_seconds: float = 0.0,
        refresh_doc_delta: int = 1,
    ) -> None:
        self._kb_service = kb_service
        self._document_service = document_service
        self._registry = registry
        self._job_service = job_service
        self._auto_refresh_enabled = auto_refresh_enabled
        self._refresh_min_interval_seconds = max(
            0.0, float(refresh_min_interval_seconds)
        )
        self._refresh_doc_delta = max(1, int(refresh_doc_delta))
        # Set when no durable job worker is running: created jobs are executed
        # in-process via this runner (fire-and-forget task spawner).
        self._job_runner: Callable[[JobRecord], Any] | None = None
        # Serializes refresh execution per KB inside this process so an
        # in-process spawned job and a route-triggered job cannot interleave
        # LLM calls and metadata writes for the same KB.
        self._refresh_locks: dict[str, asyncio.Lock] = {}

    def set_job_runner(self, runner: Callable[[JobRecord], Any] | None) -> None:
        self._job_runner = runner

    def _spawn_job(self, job: JobRecord) -> None:
        if self._job_runner is None:
            return
        try:
            self._job_runner(job)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to spawn agent profile job '%s': %s", job.id, exc)

    def _refresh_lock(self, kb_id: str) -> asyncio.Lock:
        lock = self._refresh_locks.get(kb_id)
        if lock is None:
            lock = asyncio.Lock()
            self._refresh_locks[kb_id] = lock
        return lock

    async def get_profile(self, kb_id: str) -> dict[str, Any]:
        record = await self._kb_service.get(kb_id)
        return agent_profile_response(record)

    async def update_manual_profile(
        self, kb_id: str, body: AgentProfileUpdateRequest
    ) -> dict[str, Any]:
        patch: dict[str, Any] = {}
        fields = body.model_fields_set
        if "agent_description" in fields:
            patch[MANUAL_DESCRIPTION_KEY] = body.agent_description
        if "agent_tags" in fields:
            tags = _clip_text_list(body.agent_tags, max_items=64, max_chars=120)
            patch[MANUAL_TAGS_KEY] = tags or None
        if "agent_priority" in fields:
            patch[MANUAL_PRIORITY_KEY] = body.agent_priority
        if "agent_domains" in fields:
            domains = _clip_text_list(body.agent_domains, max_items=40, max_chars=120)
            patch[MANUAL_DOMAINS_KEY] = domains or None
        if "agent_sample_questions" in fields:
            questions = _clip_text_list(
                body.agent_sample_questions, max_items=20, max_chars=160
            )
            patch[MANUAL_SAMPLE_QUESTIONS_KEY] = questions or None
        if "agent_negative_scope" in fields:
            scope = _clip_text_list(
                body.agent_negative_scope, max_items=20, max_chars=160
            )
            patch[MANUAL_NEGATIVE_SCOPE_KEY] = scope or None
        record = await self._kb_service.update(kb_id, metadata=patch)
        return agent_profile_response(record)

    async def mark_dirty(
        self,
        kb_id: str,
        *,
        reason: str,
        document_id: str | None = None,
        enqueue: bool = True,
    ) -> JobRecord | None:
        record = await self._kb_service.get(kb_id)
        metadata = record.metadata or {}
        previous_flag = _dirty_flag_from_metadata(metadata)
        previous_auto = _auto_profile_from_metadata(metadata)
        flag = {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "dirty_at": utc_now_iso(),
            "reason": reason,
            "document_id": document_id,
        }
        await self._kb_service.update(
            record.id, metadata={KB_AUTO_PROFILE_DIRTY_METADATA_KEY: flag}
        )
        if not enqueue or self._job_service is None:
            return None
        if not await self._should_enqueue_auto_refresh(
            record, previous_flag, previous_auto
        ):
            return None
        job, created = await self.enqueue_refresh(
            record.id,
            force=False,
            reason=f"auto_{reason}",
            document_id=document_id,
        )
        return job if created else None

    async def _should_enqueue_auto_refresh(
        self,
        record: KnowledgeBaseRecord,
        previous_flag: dict[str, Any] | None,
        previous_auto: dict[str, Any],
    ) -> bool:
        if not self._auto_refresh_enabled:
            return False
        if self._refresh_min_interval_seconds > 0:
            recent_marker = None
            if previous_flag is not None:
                recent_marker = previous_flag.get("dirty_at")
            recent_seconds = _seconds_since(
                recent_marker or previous_auto.get("updated_at")
            )
            if (
                recent_seconds is not None
                and recent_seconds < self._refresh_min_interval_seconds
            ):
                return False
        if self._refresh_doc_delta <= 1 or previous_auto.get("status") != "ready":
            return True
        try:
            _documents, ready_total = await self._list_ready_profile_documents(record.id)
            previous_total = int(previous_auto.get("source_doc_count") or 0)
        except (TypeError, ValueError):
            return True
        return abs(ready_total - previous_total) >= self._refresh_doc_delta

    async def _find_active_refresh_jobs(self, kb_id: str) -> list[JobRecord]:
        """Return non-stale queued/running agent_profile jobs (newest first).

        ``list_jobs`` has no job_type filter, so scan a few pages of active
        jobs (bulk ingest keeps this list short because uploads create
        aggregate jobs). Jobs whose last update is older than
        ``ACTIVE_PROFILE_JOB_STALE_SECONDS`` are ignored so a wedged job
        cannot suppress refreshes forever.
        """
        if self._job_service is None:
            return []
        matches: list[JobRecord] = []
        page_size = 200
        for page in range(3):
            jobs, total = await self._job_service.list_jobs(
                kb_id,
                statuses=_ACTIVE_JOB_STATUSES,
                limit=page_size,
                offset=page * page_size,
            )
            for job in jobs:
                if job.job_type != "agent_profile":
                    continue
                age = _seconds_since(job.updated_at)
                if age is not None and age > ACTIVE_PROFILE_JOB_STALE_SECONDS:
                    continue
                matches.append(job)
            if (page + 1) * page_size >= total or not jobs:
                break
        return matches

    async def enqueue_refresh(
        self,
        kb_id: str,
        *,
        force: bool = False,
        reason: str = "manual_refresh",
        document_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> tuple[JobRecord, bool]:
        if self._job_service is None:
            raise HTTPException(status_code=503, detail="Job service is not configured")
        if idempotency_key is None:
            # Coalesce refresh churn: while agent_profile jobs are queued or
            # running, further no-key enqueues return one instead of stacking
            # duplicate full refreshes. A non-force request is covered by ANY
            # active job; a force request is only covered by an active force
            # job (the worker serializes execution; in-process execution is
            # serialized by the per-KB refresh lock).
            active_jobs = await self._find_active_refresh_jobs(kb_id)
            if active_jobs:
                if not force:
                    return active_jobs[0], False
                for active in active_jobs:
                    if bool((active.payload or {}).get("force", False)):
                        return active, False
        payload = {
            "force": force,
            "reason": reason,
            "document_id": document_id,
        }
        job, created = await self._job_service.create_job_once(
            kb_id,
            job_type="agent_profile",
            stage="profiling",
            total_items=1,
            payload=payload,
            idempotency_key=idempotency_key,
        )
        if created:
            self._spawn_job(job)
        return job, created

    async def run_job(self, job: JobRecord) -> None:
        if self._job_service is None:
            raise RuntimeError("Job service is not configured")
        try:
            if job.status in {"queued", "retrying"}:
                await self._job_service.transition_job(
                    job.kb_id,
                    job.id,
                    status="running",
                    stage="profiling",
                    progress=0.05,
                )
            result = await self.refresh_kb_profile(
                job.kb_id,
                force=bool((job.payload or {}).get("force", False)),
                reason=str((job.payload or {}).get("reason") or "manual_refresh"),
                job_id=job.id,
            )
            await self._job_service.transition_job(
                job.kb_id,
                job.id,
                status="succeeded",
                stage="profiled",
                progress=1.0,
                completed_items=1,
                result={
                    "kb_id": result.kb_id,
                    "status": result.status,
                    "auto_profile": result.auto_profile,
                    "document_profiles_generated": result.document_profiles_generated,
                    "document_profiles_reused": result.document_profiles_reused,
                    "document_profiles_failed": result.document_profiles_failed,
                    "pending_document_profiles": result.pending_document_profiles,
                },
            )
            await self._after_refresh(job.kb_id, result)
        except Exception as exc:  # noqa: BLE001
            logger.error("Agent profile job '%s' failed: %s", job.id, exc, exc_info=True)
            await self._mark_profile_failed(job.kb_id, job_id=job.id, error=str(exc))
            await self._job_service.transition_job(
                job.kb_id,
                job.id,
                status="failed",
                stage="profiling_failed",
                progress=1.0,
                failed_items=1,
                error_code="agent_profile_failed",
                error_message=str(exc),
            )

    async def _after_refresh(
        self, kb_id: str, result: AgentProfileRefreshResult
    ) -> None:
        """Post-success bookkeeping: clear the dirty flag it covered, chain a
        follow-up refresh when documents changed mid-run or profile generation
        was capped. Never fails the already-succeeded job."""
        try:
            record = await self._kb_service.get(kb_id)
            flag = _dirty_flag_from_metadata(record.metadata or {})
            dirty_during_run = flag is not None and _is_newer(
                flag.get("dirty_at"), result.refresh_started_at
            )
            if flag is not None and not dirty_during_run:
                await self._kb_service.update(
                    kb_id, metadata={KB_AUTO_PROFILE_DIRTY_METADATA_KEY: None}
                )
            if dirty_during_run or result.pending_document_profiles > 0:
                await self.enqueue_refresh(
                    kb_id,
                    force=False,
                    reason="chained_refresh",
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Agent profile post-refresh bookkeeping failed for KB '%s': %s",
                kb_id,
                exc,
            )

    async def refresh_kb_profile(
        self,
        kb_id: str,
        *,
        force: bool = False,
        reason: str = "manual_refresh",
        job_id: str | None = None,
    ) -> AgentProfileRefreshResult:
        async with self._refresh_lock(kb_id):
            return await self._refresh_kb_profile_locked(
                kb_id, force=force, reason=reason, job_id=job_id
            )

    async def _refresh_kb_profile_locked(
        self,
        kb_id: str,
        *,
        force: bool,
        reason: str,
        job_id: str | None,
    ) -> AgentProfileRefreshResult:
        refresh_started_at = utc_now_iso()
        record = await self._kb_service.get(kb_id)
        rag = await self._registry.get(record.id)
        profile_func = self._get_profile_llm_func(rag)
        documents, total = await self._list_ready_profile_documents(record.id)

        profiles_by_doc: dict[str, dict[str, Any]] = {}
        stale_by_cache: list[DocumentRecord] = []
        for document in documents:
            cached = document.metadata.get(DOC_PROFILE_METADATA_KEY)
            if self._document_profile_cache_valid(document, cached):
                profiles_by_doc[document.id] = dict(cached)
            else:
                stale_by_cache.append(document)

        # force regenerates the newest window even when cached; documents
        # beyond the window keep their cached profiles. Cache-invalid documents
        # beyond the per-run cap are reported as pending and handled by a
        # chained refresh, so large KBs converge over successive runs.
        if force:
            to_generate = documents[:MAX_PROFILE_DOCUMENTS]
        else:
            to_generate = stale_by_cache[:MAX_PROFILE_DOCUMENTS]
        generate_ids = {document.id for document in to_generate}
        pending = sum(
            1 for document in stale_by_cache if document.id not in generate_ids
        )

        generated = 0
        failed_profiles: list[dict[str, str]] = []
        for document in to_generate:
            try:
                doc_profile = await self._generate_document_profile(
                    profile_func=profile_func,
                    rag=rag,
                    record=record,
                    document=document,
                )
            except Exception as exc:  # noqa: BLE001 — one bad document must not
                # block the KB-level profile. The document keeps its invalid
                # cache entry and is retried on the next refresh trigger
                # (document event or manual refresh); failures deliberately do
                # NOT chain an automatic refresh to avoid a retry storm on a
                # permanently-failing document.
                logger.warning(
                    "Agent document profile failed for KB '%s' doc '%s': %s",
                    record.id,
                    document.id,
                    exc,
                )
                failed_profiles.append(
                    {"document_id": document.id, "error": str(exc)[:160]}
                )
                continue
            profiles_by_doc[document.id] = doc_profile
            generated += 1
            await self._document_service.update_document(
                record.id,
                document.id,
                metadata_patch={DOC_PROFILE_METADATA_KEY: doc_profile},
            )
        if failed_profiles and not profiles_by_doc:
            # Nothing cached and nothing generated: keep the loud failure
            # signal instead of writing a contentless "ready" stub profile.
            raise HTTPException(
                status_code=502,
                detail={
                    "error_code": "agent_profile_documents_failed",
                    "message": (
                        f"All {len(failed_profiles)} document profile "
                        "generations failed"
                    ),
                },
            )
        reused = len(profiles_by_doc) - generated

        aggregated_documents = [
            document for document in documents if document.id in profiles_by_doc
        ]
        all_profiles = [profiles_by_doc[document.id] for document in aggregated_documents]
        tag_stats = _frequency_stats(
            [tag for profile in all_profiles for tag in profile.get("tags", [])]
        )
        domain_stats = _frequency_stats(
            [d for profile in all_profiles for d in profile.get("domains", [])]
        )

        # Aggregation strategy (see the constants block for the full rationale):
        # direct for small KBs, stride-sampled while backfill is pending, and
        # cached grouped map-reduce once every document profile exists.
        aggregation_mode = "direct"
        group_entries: list[dict[str, Any]] = []
        groups_generated = 0
        groups_reused = 0
        sampled_documents = aggregated_documents
        if len(aggregated_documents) > MAX_PROFILE_DOCS_FOR_AGGREGATION:
            if pending > 0:
                aggregation_mode = "sampled"
                sampled_documents = _stride_sample(
                    aggregated_documents, MAX_PROFILE_DOCS_FOR_AGGREGATION
                )
            else:
                aggregation_mode = "grouped"
                sampled_documents = []
                (
                    group_entries,
                    groups_generated,
                    groups_reused,
                ) = await self._build_group_profiles(
                    profile_func=profile_func,
                    record=record,
                    aggregated_documents=aggregated_documents,
                    profiles_by_doc=profiles_by_doc,
                )
        kb_profile = await self._generate_kb_profile(
            profile_func=profile_func,
            record=record,
            document_profiles=[
                profiles_by_doc[document.id] for document in sampled_documents
            ],
            group_profiles=_stride_sample(group_entries, MAX_GROUPS_FOR_AGGREGATION),
            aggregation_mode=aggregation_mode,
            total_ready_documents=total,
            profiled_documents=len(aggregated_documents),
            pending_documents=pending,
            tag_stats=tag_stats,
            domain_stats=domain_stats,
            reason=reason,
        )
        sampled_doc_count = (
            sum(int(entry.get("doc_count") or 0) for entry in group_entries)
            if aggregation_mode == "grouped"
            else len(sampled_documents)
        )
        auto_profile = _compact_kb_profile_for_metadata(
            kb_profile,
            job_id=job_id,
            total_ready_documents=total,
            profiled_documents=len(aggregated_documents),
            pending_documents=pending,
            aggregated_documents=aggregated_documents,
            profiles_by_doc=profiles_by_doc,
            generated=generated,
            reused=reused,
            aggregation_mode=aggregation_mode,
            sampled_doc_count=sampled_doc_count,
            group_count=len(group_entries),
            group_profiles_generated=groups_generated,
            group_profiles_reused=groups_reused,
            failed_profiles=failed_profiles,
        )
        await self._kb_service.update(
            record.id, metadata={KB_AUTO_PROFILE_METADATA_KEY: auto_profile}
        )
        return AgentProfileRefreshResult(
            kb_id=record.id,
            status="ready",
            auto_profile=auto_profile,
            document_profiles_generated=generated,
            document_profiles_reused=reused,
            pending_document_profiles=pending,
            document_profiles_failed=len(failed_profiles),
            refresh_started_at=refresh_started_at,
            aggregation_mode=aggregation_mode,
            group_profiles_generated=groups_generated,
            group_profiles_reused=groups_reused,
        )

    async def _build_group_profiles(
        self,
        *,
        profile_func: Any,
        record: KnowledgeBaseRecord,
        aggregated_documents: list[DocumentRecord],
        profiles_by_doc: dict[str, dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int, int]:
        """Map stage of the grouped aggregation: one cached digest per bucket.

        Documents are bucketed in (created_at, id) order so append-mostly KBs
        keep earlier buckets stable; each digest is cached on the bucket's
        first document under ``GROUP_PROFILE_METADATA_KEY`` and reused while
        the bucket's membership/content hash is unchanged.
        """
        ordered = sorted(
            aggregated_documents, key=lambda d: (d.created_at or "", d.id)
        )
        buckets = [
            ordered[i : i + AGGREGATION_GROUP_SIZE]
            for i in range(0, len(ordered), AGGREGATION_GROUP_SIZE)
        ]
        entries: list[dict[str, Any]] = []
        generated = 0
        reused = 0
        for group_index, bucket in enumerate(buckets):
            anchor = bucket[0]
            bucket_hash = _group_hash(bucket, profiles_by_doc)
            cached = anchor.metadata.get(GROUP_PROFILE_METADATA_KEY)
            if (
                isinstance(cached, dict)
                and cached.get("schema_version") == PROFILE_SCHEMA_VERSION
                and cached.get("group_hash") == bucket_hash
                and cached.get("status") == "ready"
            ):
                entry = dict(cached)
                entry["group_index"] = group_index
                entries.append(_compact_group_profile_entry(entry))
                reused += 1
                continue
            parsed = await self._call_profile_json(
                profile_func,
                {
                    "kb": {
                        "kb_id": record.id,
                        "name": record.name,
                        "description": record.description or "",
                    },
                    "group_index": group_index,
                    "doc_count": len(bucket),
                    "document_profiles": [
                        _compact_document_profile_entry(profiles_by_doc[document.id])
                        for document in bucket
                    ],
                    "output_schema": {
                        "type": "document_group_profile",
                        "description": "compact digest covering every theme in this batch",
                        "tags": ["short tag"],
                        "domains": ["domain or business area"],
                        "sample_questions": ["question these documents can answer"],
                        "negative_scope": ["topics these documents should not be used for"],
                    },
                },
                system_prompt=GROUP_PROFILE_SYSTEM_PROMPT,
                model=DocumentGroupProfile,
            )
            assert isinstance(parsed, DocumentGroupProfile)
            stored = parsed.model_dump()
            stored.update(
                {
                    "schema_version": PROFILE_SCHEMA_VERSION,
                    "status": "ready",
                    "group_hash": bucket_hash,
                    "group_index": group_index,
                    "doc_count": len(bucket),
                    "updated_at": utc_now_iso(),
                }
            )
            await self._document_service.update_document(
                record.id,
                anchor.id,
                metadata_patch={GROUP_PROFILE_METADATA_KEY: stored},
            )
            anchor.metadata[GROUP_PROFILE_METADATA_KEY] = stored
            entries.append(_compact_group_profile_entry(stored))
            generated += 1
        if generated:
            logger.info(
                "Agent profile grouped aggregation for KB '%s': %d/%d group digests regenerated",
                record.id,
                generated,
                len(buckets),
            )
        return entries, generated, reused

    def _get_profile_llm_func(self, rag: Any) -> Any:
        global_config = rag._build_global_config()
        profile_func = (global_config.get("role_llm_funcs") or {}).get("profile")
        if profile_func is None:
            raise HTTPException(status_code=500, detail="PROFILE role LLM is unavailable")
        return profile_func

    async def _list_ready_profile_documents(
        self, kb_id: str
    ) -> tuple[list[DocumentRecord], int]:
        """List ALL ready documents (newest first) eligible for profiling.

        Paginates the control-plane listing so KBs beyond one page still get
        full profile coverage; the returned count reflects the filtered
        (enabled, non-archived, indexed) set.
        """
        page_size = 200
        offset = 0
        all_documents: list[DocumentRecord] = []
        while True:
            documents, total = await self._document_service.list_documents(
                kb_id, status="ready", limit=page_size, offset=offset
            )
            all_documents.extend(documents)
            offset += page_size
            if offset >= total or not documents:
                break
        filtered = [
            document
            for document in all_documents
            if document.enabled and not document.archived and document.lightrag_doc_id
        ]
        return filtered, len(filtered)

    def _document_profile_cache_valid(self, document: DocumentRecord, value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        return (
            value.get("schema_version") == PROFILE_SCHEMA_VERSION
            and value.get("source_hash") == document.source_hash
            and value.get("index_hash") == document.index_hash
            and value.get("status") == "ready"
        )

    async def _generate_document_profile(
        self,
        *,
        profile_func: Any,
        rag: Any,
        record: KnowledgeBaseRecord,
        document: DocumentRecord,
    ) -> dict[str, Any]:
        content = await self._document_content_sample(rag, document)
        payload = {
            "kb": {
                "kb_id": record.id,
                "name": record.name,
                "description": record.description or "",
            },
            "document": {
                "document_id": document.id,
                "source_name": document.source_name,
                "content_type": document.content_type,
                "chunks_count": document.chunks_count,
            },
            "content_sample": content,
            "output_schema": {
                "type": "document_agent_profile",
                "summary": "1-4 sentences describing routing-relevant scope",
                "tags": ["short tag"],
                "domains": ["domain or business area"],
                "sample_questions": ["question this document can answer"],
                "negative_scope": ["topics this document should not be used for"],
            },
        }
        parsed = await self._call_profile_json(
            profile_func,
            payload,
            system_prompt=DOCUMENT_PROFILE_SYSTEM_PROMPT,
            model=DocumentAgentProfile,
        )
        assert isinstance(parsed, DocumentAgentProfile)
        profile = parsed.model_dump()
        profile.update(
            {
                "schema_version": PROFILE_SCHEMA_VERSION,
                "status": "ready",
                "document_id": document.id,
                "source_name": document.source_name,
                "source_hash": document.source_hash,
                "index_hash": document.index_hash,
                "content_hash": _stable_text_hash(content),
                "updated_at": utc_now_iso(),
            }
        )
        return profile

    async def _generate_kb_profile(
        self,
        *,
        profile_func: Any,
        record: KnowledgeBaseRecord,
        document_profiles: list[dict[str, Any]],
        group_profiles: list[dict[str, Any]] | None = None,
        aggregation_mode: str = "direct",
        total_ready_documents: int,
        profiled_documents: int,
        pending_documents: int,
        tag_stats: list[dict[str, Any]],
        domain_stats: list[dict[str, Any]],
        reason: str,
    ) -> KBAgentAutoProfile:
        group_profiles = group_profiles or []
        if not document_profiles and not group_profiles:
            return KBAgentAutoProfile(
                description=(record.description or record.name or "").strip(),
                tags=[],
                domains=[],
                sample_questions=[],
                negative_scope=[],
            )
        payload = {
            "kb": {
                "kb_id": record.id,
                "name": record.name,
                "description": record.description or "",
            },
            "reason": reason,
            "total_ready_documents": total_ready_documents,
            "profiled_document_count": profiled_documents,
            "pending_document_profiles": pending_documents,
            # direct: every document profile verbatim; sampled: evenly-spaced
            # subset across the whole KB (backfill in progress); grouped:
            # per-bucket digests covering EVERY profiled document.
            "aggregation_mode": aggregation_mode,
            # Frequency stats cover EVERY cached document profile, so the
            # aggregation still sees the whole KB even when the verbatim
            # profile list below is sampled.
            "tag_stats": tag_stats,
            "domain_stats": domain_stats,
            "document_profiles": [
                _compact_document_profile_entry(item) for item in document_profiles
            ],
            "group_profiles": group_profiles,
            "output_schema": {
                "type": "kb_agent_auto_profile",
                "description": "compact Agent routing description",
                "tags": ["short tag"],
                "domains": ["domain or business area"],
                "sample_questions": ["question this KB can answer"],
                "negative_scope": ["topics this KB should not be used for"],
            },
        }
        parsed = await self._call_profile_json(
            profile_func,
            payload,
            system_prompt=KB_PROFILE_SYSTEM_PROMPT,
            model=KBAgentAutoProfile,
        )
        assert isinstance(parsed, KBAgentAutoProfile)
        return parsed

    async def _document_content_sample(self, rag: Any, document: DocumentRecord) -> str:
        if not document.lightrag_doc_id:
            return ""
        full_docs = getattr(rag, "full_docs", None)
        if full_docs is None:
            return ""
        row = await full_docs.get_by_id(document.lightrag_doc_id)
        if not isinstance(row, dict):
            return ""
        body = strip_lightrag_doc_prefix(row.get("content"), row.get("parse_format"))
        return _sample_text(body)

    async def _call_profile_json(
        self,
        profile_func: Any,
        payload: dict[str, Any],
        *,
        system_prompt: str,
        model: type[DocumentAgentProfile]
        | type[DocumentGroupProfile]
        | type[KBAgentAutoProfile],
    ) -> DocumentAgentProfile | DocumentGroupProfile | KBAgentAutoProfile:
        prompt = json.dumps(payload, ensure_ascii=False)
        try:
            return await call_llm_json(
                profile_func,
                prompt,
                system_prompt=system_prompt,
                priority=DEFAULT_QUERY_PRIORITY,
                parse=model.model_validate,
                attempts=PROFILE_LLM_ATTEMPTS,
                label=f"agent_profile:{model.__name__}",
            )
        except LLMJsonError as exc:
            raise HTTPException(
                status_code=502,
                detail={"error_code": "agent_profile_invalid", "message": str(exc)},
            ) from exc

    async def _mark_profile_failed(self, kb_id: str, *, job_id: str, error: str) -> None:
        try:
            record = await self._kb_service.get(kb_id)
            auto = _auto_profile_from_metadata(record.metadata or {})
            auto.update(
                {
                    "schema_version": PROFILE_SCHEMA_VERSION,
                    "status": "failed",
                    "last_error": error[:1000],
                    "failed_at": utc_now_iso(),
                    "job_id": job_id,
                }
            )
            await self._kb_service.update(
                kb_id, metadata={KB_AUTO_PROFILE_METADATA_KEY: auto}
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to mark Agent profile failure for KB '%s': %s", kb_id, exc
            )
