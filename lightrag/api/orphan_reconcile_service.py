"""Durable plan/apply/resume orphan reconciliation service.

Phase 3.2 Writer O (fix-O).  This service mirrors the
``PipelineArtifactTerminalizationReconciler`` pattern: a thin metadata-driven
reconciler that uses the frozen Phase 3.1-A ``ArtifactMaintenanceRunRecord`` /
``ArtifactMaintenanceItemRecord`` authority plus the frozen
``enqueue_artifact_cleanup_manifest`` API.  It **never** deletes objects
directly -- apply only enqueues cleanup manifests that the accepted
``ArtifactCleanupService`` later drains with verified deletion.

Lifecycle:

* **Plan (default)**: bounded-discover objects under the configured bucket
  prefix via ``list_objects_page`` and classify each one as ``eligible``,
  ``referenced``, ``retained``, ``malformed``, or ``unknown_owner``.  The
  classifications are persisted as maintenance items under a dry-run run.
* **Apply** (``--plan-id`` + ``--apply`` + ``--yes``): claim an apply run and
  for each ``eligible`` item enqueue an ``orphan_reconcile`` cleanup manifest
  with the appropriate ``target_namespace``.  ``retained`` items are only
  released when the caller passes ``release_retained=True``.  ``malformed`` and
  ``unknown_owner`` items are report-only (skipped, never enqueued).
* **Resume**: the frozen maintenance-item state machine is already resumable;
  a crashed apply can be re-driven by calling ``apply_plan`` again.

The classification reuses the same durable authority helpers as the cleanup
service (current source/artifact references, attempt-token lineage, retained
manifests, KB generation, unknown commit outcomes) but only reads them; this
module does not edit stores or ``artifact_lifecycle.py``.

All audit/JSON output is redacted via ``redact_value`` so credentials, DSNs,
scratch paths, and absolute local roots cannot leak.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from urllib.parse import urlparse

from lightrag.api.artifact_lifecycle import (
    ArtifactCleanupManifestRecord,
    ArtifactCleanupTargetKind,
    ArtifactCleanupTargetNamespace,
    ArtifactLifecycleConflictError,
    ArtifactLifecycleError,
    ArtifactLifecycleNotFoundError,
    ArtifactLifecycleStateError,
    ArtifactMaintenanceItemRecord,
    ArtifactMaintenanceItemState,
    ArtifactMaintenanceMetadataBackend,
    ArtifactMaintenanceRunKind,
    ArtifactMaintenanceRunRecord,
    artifact_cleanup_idempotency_key,
    artifact_maintenance_item_key,
    artifact_target_uri_digest,
    normalize_artifact_relative_object_id,
    normalize_artifact_target_uri,
    normalize_artifact_target_uri_authority,
)
from lightrag.api.kb_service import sanitize_workspace, utc_now_iso
from lightrag.api.object_storage import ObjectStorage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAINTENANCE_KIND: ArtifactMaintenanceRunKind = "orphan_reconcile"
_MAINTENANCE_BACKEND_FINGERPRINT = "sha256:orphan-reconcile-service:v1"
_DEFAULT_LEASE_SECONDS = 600.0
_MAX_ITEMS_PER_RUN = 500  # frozen store cap
_MAX_LIST_PAGES = 32  # bounded discovery budget
_DEFAULT_MIN_AGE_HOURS = 24
_DEFAULT_OBJECT_PREFIX = "kb"

OrphanClassification = Literal[
    "eligible", "referenced", "retained", "malformed", "unknown_owner", "too_new"
]
"""Durable classification label recorded against one discovered object.

* ``eligible``       -- orphan, safe to reclaim; apply enqueues a cleanup manifest.
* ``referenced``     -- a live document/artifact/job still owns this object.
* ``retained``       -- a retained cleanup manifest holds this object.
* ``malformed``      -- the object key does not match validated ownership.
* ``unknown_owner``  -- ambiguous ownership (KB lifecycle/workspace mismatch).
* ``too_new``        -- the object is younger than the minimum-age window.
"""

_APPLY_CLASSIFICATIONS: frozenset[OrphanClassification] = frozenset(
    {"eligible", "retained"}
)
_REPORT_ONLY_CLASSIFICATIONS: frozenset[OrphanClassification] = frozenset(
    {"malformed", "unknown_owner", "too_new", "referenced"}
)

_ACTIVE_JOB_STATUSES: tuple[str, ...] = ("queued", "running", "retrying", "cancelling")
_ORIGIN_ATTEMPT_FIELDS = (
    "origin_attempt_token",
    "attempt_token",
    "claim_token",
    "parse_attempt_token",
    "build_attempt_token",
    "pipeline_attempt_token",
    "delete_attempt_token",
    "replace_attempt_token",
)
_CURRENT_ARTIFACT_ID_FIELDS = (
    "current_original_artifact_id",
    "current_sidecar_artifact_id",
    "current_blocks_artifact_id",
    "current_markdown_artifact_id",
)
_CURRENT_ARTIFACT_ID_LIST_FIELDS = (
    "current_artifact_ids",
    "current_raw_artifact_ids",
)
_TERMINAL_ITEM_STATES: frozenset[ArtifactMaintenanceItemState] = frozenset(
    {"verified", "skipped", "blocked", "failed"}
)

# Redaction patterns.  Mirrors the migration-CLI scrubbers so audit/JSON output
# never carries scratch paths, DSNs, credentials, or absolute local roots.
_SCRATCH_RE = re.compile(
    r"(?:file://)?[^\s\"']*\.lightrag-scratch[/\\][^\s\"']*",
    re.IGNORECASE,
)
_DSN_RE = re.compile(
    r"(?:postgres(?:ql)?|mysql|redis|mongodb)(?:\+[\w]+)?://[^\s\"'<>]+",
    re.IGNORECASE,
)
_QUERY_DSN_RE = re.compile(
    r"(?:^|\s)(?:host|hostaddr|port|dbname|database|user|password|sslmode)\s*=",
    re.IGNORECASE,
)
_SECRET_KEY_RE = re.compile(
    r"(?:aws[_-]?access[_-]?key(?:[_-]?id)?|"
    r"aws[_-]?secret[_-]?access[_-]?key|"
    r"access[_-]?key(?:[_-]?id)?|secret[_-]?(?:access[_-]?)?key|"
    r"x-amz-(?:credential|signature|security-token)|"
    r"x-goog-(?:credential|signature)|password|secret|token)"
    r"\s*[=:]\s*[^\s\"'<>]+",
    re.IGNORECASE,
)
_ABSOLUTE_ROOT_RE = re.compile(r"(?:^|[\s\"'<>])(/[^\"'<>\s]+)")
_SCRATCH_TOKEN_RE = re.compile(
    r"(?:\.lightrag-scratch|\.sync-staging|\.replace-staging)",
    re.IGNORECASE,
)


def redact_value(value: object) -> str:
    """Redact scratch paths, DSNs, credentials, and absolute roots from output."""

    text = str(value)
    text = _SCRATCH_RE.sub("<artifact-materialization>", text)
    text = _DSN_RE.sub("<redacted-dsn>", text)
    text = _SECRET_KEY_RE.sub("<redacted-credential>", text)
    text = _QUERY_DSN_RE.sub("<redacted-dsn>", text)
    text = _SCRATCH_TOKEN_RE.sub("<artifact-materialization>", text)
    text = _ABSOLUTE_ROOT_RE.sub(
        lambda match: f"{match.group(1)[:1]}<redacted-root>", text
    )
    return text


def redact_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively redact a nested mapping for audit/JSON output."""

    redacted: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            redacted[key] = redact_mapping(item)
        elif isinstance(item, str):
            redacted[key] = redact_value(item)
        else:
            redacted[key] = item
    return redacted


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class OrphanReconcileError(RuntimeError):
    """Base class for orphan reconciliation errors. Messages are redaction-safe."""


class OrphanReconcilePlanError(OrphanReconcileError):
    """The requested plan is missing, ambiguous, or not yet succeeded."""


class OrphanReconcileApplyGuardError(OrphanReconcileError):
    """A precondition blocked apply (online mutation, stale plan, etc.)."""


# ---------------------------------------------------------------------------
# Parsed object key
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParsedObjectKey:
    """Result of parsing one discovered object key into ownership segments."""

    workspace: str
    document_id: str | None
    namespace: ArtifactCleanupTargetNamespace | None
    source_generation_id: str | None
    artifact_id: str | None
    origin_job_id: str | None
    origin_attempt_token: str | None

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "workspace": self.workspace,
            "document_id": self.document_id,
            "namespace": self.namespace,
            "source_generation_id": self.source_generation_id,
            "artifact_id": self.artifact_id,
            "origin_job_id": self.origin_job_id,
            "origin_attempt_token": self.origin_attempt_token,
        }


# ---------------------------------------------------------------------------
# In-memory classification record
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OrphanCandidateSpec:
    """One discovered object awaiting durable persistence as a maintenance item."""

    target_uri: str
    relative_object_id: str
    classification: OrphanClassification
    parsed: ParsedObjectKey | None
    last_modified: datetime
    size_bytes: int | None
    etag: str | None
    checksum: str | None
    version_id: str | None
    reason_codes: list[str] = field(default_factory=list)
    kb_id: str | None = None
    kb_generation: str | None = None

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "namespace": self.parsed.namespace if self.parsed else None,
            "document_id": self.parsed.document_id if self.parsed else None,
            "size_bytes": self.size_bytes,
            "reason_codes": list(self.reason_codes),
            # Target URIs are object URIs (s3://...) and safe to emit; the
            # redactor still scrubs them if any scratch/credential leaks into
            # the value.  Keys are persisted as relative_object_id below.
            "relative_object_id": self.relative_object_id,
        }


@dataclass(slots=True)
class OrphanReconcilePlanSummary:
    plan_id: str
    item_count: int
    candidates: list[OrphanCandidateSpec]
    apply_run_id: str | None
    metadata_backend: ArtifactMaintenanceMetadataBackend
    classifications: dict[str, int]

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "item_count": self.item_count,
            "metadata_backend": self.metadata_backend,
            "apply_run_id": self.apply_run_id,
            "classifications": dict(self.classifications),
            "candidates": [spec.to_audit_dict() for spec in self.candidates],
        }


@dataclass(slots=True)
class OrphanReconcileApplySummary:
    plan_id: str
    apply_run_id: str
    items_total: int
    items_enqueued: int
    items_skipped: int
    items_blocked: int
    items_failed: int
    counters: dict[str, int]
    issues: list[str] = field(default_factory=list)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "apply_run_id": self.apply_run_id,
            "items_total": self.items_total,
            "items_enqueued": self.items_enqueued,
            "items_skipped": self.items_skipped,
            "items_blocked": self.items_blocked,
            "items_failed": self.items_failed,
            "counters": dict(self.counters),
            "issues": list(self.issues),
        }


# ---------------------------------------------------------------------------
# Provider callbacks (kept narrow so stores stay unedited)
# ---------------------------------------------------------------------------

#: Returns all KB lifecycle rows the reconciler should consider when resolving
#: object ownership.  The default implementation reads ``enterprise_kb_lifecycle``
#: via the SQLite/PostgreSQL store's existing ``list_artifact_maintenance_runs``
#: -style helpers; callers may inject a stub for tests.
KBLifecycleProvider = Callable[[], Awaitable[Sequence[Any]]]


async def _default_kb_lifecycle_provider(
    metadata_store: Any,
) -> list[Any]:
    """Best-effort enumeration of all KB lifecycle rows.

    Reads the lifecycle table directly because the frozen store does not yet
    expose a public ``list_kb_lifecycles`` API.  Falls back gracefully to an
    empty list when the store lacks the expected private hook; the service
    then treats every parsed object as ``unknown_owner`` rather than guessing.
    """

    lister = getattr(metadata_store, "list_kb_lifecycles", None)
    if lister is None:
        return []
    try:
        result = await lister()
    except Exception:
        return []
    if isinstance(result, tuple):
        rows, _total = result
        return list(rows)
    return list(result)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class OrphanReconcileService:
    """Drive the plan/apply/resume orphan-reconciliation state machine.

    The service is deliberately stateless between calls: every phase reads the
    frozen metadata-store authority afresh so a crashed apply can be resumed by
    calling ``apply_plan`` again with the same ``plan_id``.
    """

    def __init__(
        self,
        *,
        metadata_store: Any,
        object_storage: ObjectStorage,
        metadata_backend: ArtifactMaintenanceMetadataBackend,
        bucket: str,
        prefix: str,
        lease_duration_seconds: float = _DEFAULT_LEASE_SECONDS,
        min_age_hours: int = _DEFAULT_MIN_AGE_HOURS,
        now: Callable[[], datetime] | None = None,
        kb_lifecycle_provider: KBLifecycleProvider | None = None,
    ) -> None:
        if not isinstance(bucket, str) or not bucket.strip():
            raise ValueError("bucket must be a non-empty string")
        if not isinstance(prefix, str):
            raise ValueError("prefix must be a string")
        if isinstance(min_age_hours, bool) or not isinstance(min_age_hours, int):
            raise ValueError("min_age_hours must be an integer")
        if min_age_hours < 0:
            raise ValueError("min_age_hours must be non-negative")
        if metadata_backend not in ("sqlite", "postgres"):
            raise ValueError("metadata_backend must be 'sqlite' or 'postgres'")
        self._metadata_store = metadata_store
        self._object_storage = object_storage
        self._metadata_backend: ArtifactMaintenanceMetadataBackend = metadata_backend
        self._bucket = bucket.strip()
        self._prefix = prefix.strip("/") or _DEFAULT_OBJECT_PREFIX
        self._lease_duration_seconds = lease_duration_seconds
        self._min_age_hours = min_age_hours
        self._now = now or _default_now
        self._kb_lifecycle_provider = kb_lifecycle_provider or (
            lambda: _default_kb_lifecycle_provider(metadata_store)
        )

    # ------------------------------------------------------------------
    # Plan
    # ------------------------------------------------------------------

    async def create_plan(
        self,
        *,
        actor_id: str | None = None,
        max_pages: int = _MAX_LIST_PAGES,
    ) -> OrphanReconcilePlanSummary:
        """Discover objects under the configured prefix and classify each.

        The plan persists a ``dry_run`` maintenance run with one item per
        discovered object.  Items never perform side effects.
        """

        if isinstance(max_pages, bool) or not isinstance(max_pages, int):
            raise ValueError("max_pages must be an integer")
        if max_pages <= 0 or max_pages > _MAX_LIST_PAGES:
            raise ValueError(
                f"max_pages must be between 1 and {_MAX_LIST_PAGES} inclusive"
            )
        candidates = await self._discover_and_classify(max_pages=max_pages)

        scope_payload = self._build_scope_payload()
        scope_fingerprint = _fingerprint_json(scope_payload)
        config_payload = {
            "bucket": self._bucket,
            "prefix": self._prefix,
            "min_age_hours": self._min_age_hours,
            "target_uri_authority": self._target_uri_authority(),
        }
        config_fingerprint = _fingerprint_json(config_payload)

        run = ArtifactMaintenanceRunRecord(
            id=self._mint_run_id("plan"),
            kind=_MAINTENANCE_KIND,
            mode="dry_run",
            status="planned",
            metadata_backend=self._metadata_backend,
            parent_plan_id=None,
            backend_fingerprint=_MAINTENANCE_BACKEND_FINGERPRINT,
            scope_fingerprint=scope_fingerprint,
            config_fingerprint=config_fingerprint,
            scope_json=scope_payload,
            actor_id=actor_id,
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
        )
        run = await self._metadata_store.create_artifact_maintenance_run(run)

        maintenance_items = [
            self._spec_to_item(run.id, ordinal, spec)
            for ordinal, spec in enumerate(candidates)
        ]
        for offset in range(0, len(maintenance_items), _MAX_ITEMS_PER_RUN):
            batch = maintenance_items[offset : offset + _MAX_ITEMS_PER_RUN]
            await self._metadata_store.create_artifact_maintenance_items(batch)

        claimed = await self._metadata_store.claim_artifact_maintenance_run(
            run.id,
            lease_owner="orphan-reconcile-plan",
            lease_duration_seconds=self._lease_duration_seconds,
        )
        await self._metadata_store.transition_artifact_maintenance_run(
            run.id,
            expected_status="running",
            new_status="succeeded",
            lease_owner=claimed.lease_owner,
            lease_token=claimed.lease_token,
            counters={
                "total_items": len(candidates),
                "planned_items": len(candidates),
            },
        )

        classifications = _count_classifications(candidates)
        return OrphanReconcilePlanSummary(
            plan_id=run.id,
            item_count=len(candidates),
            candidates=candidates,
            apply_run_id=None,
            metadata_backend=self._metadata_backend,
            classifications=classifications,
        )

    async def _discover_and_classify(
        self, *, max_pages: int
    ) -> list[OrphanCandidateSpec]:
        prefix_uri = self._object_storage.object_prefix_uri_for_key(self._prefix)
        kb_index = await self._build_kb_workspace_index()

        specs: list[OrphanCandidateSpec] = []
        seen_keys: set[str] = set()
        next_token: str | None = None
        for _ in range(max_pages):
            page = await self._object_storage.list_objects_page(
                prefix_uri,
                max_keys=1000,
                continuation_token=next_token,
            )
            for entry in page.entries:
                if entry.key in seen_keys:
                    continue
                seen_keys.add(entry.key)
                spec = await self._classify_entry(entry, kb_index=kb_index)
                specs.append(spec)
            next_token = page.next_token
            if not next_token:
                break

        specs.sort(key=lambda spec: spec.relative_object_id)
        return specs

    async def _classify_entry(
        self, entry: Any, *, kb_index: Mapping[str, Any]
    ) -> OrphanCandidateSpec:
        key = entry.key
        try:
            relative_object_id = normalize_artifact_relative_object_id(key)
        except ValueError:
            return self._malformed_spec(entry, reason="object_key_unnormalized")

        parsed = _parse_object_key(relative_object_id, configured_prefix=self._prefix)
        target_uri = self._safe_target_uri(relative_object_id)

        spec = OrphanCandidateSpec(
            target_uri=target_uri,
            relative_object_id=relative_object_id,
            classification="eligible",
            parsed=parsed,
            last_modified=entry.last_modified,
            size_bytes=getattr(entry, "size", None),
            etag=getattr(entry, "etag", None),
            checksum=getattr(entry, "checksum", None),
            version_id=getattr(entry, "version_id", None),
        )

        if parsed is None:
            spec.classification = "malformed"
            spec.reason_codes.append("object_key_unowned")
            return spec

        # Minimum-age check (UTC).
        if self._is_too_new(entry.last_modified):
            spec.classification = "too_new"
            spec.reason_codes.append("minimum_age_not_met")
            return spec

        kb_info = kb_index.get(parsed.workspace)
        if kb_info is None:
            spec.classification = "unknown_owner"
            spec.reason_codes.append("workspace_kb_missing")
            return spec
        spec.kb_id = kb_info["kb_id"]
        spec.kb_generation = kb_info["generation"]

        if kb_info["state"] in {"deleting", "deleted"}:
            spec.classification = "referenced"
            spec.reason_codes.append(f"kb_{kb_info['state']}")
            return spec

        # Documents and artifacts namespace require a document id.
        document = None
        if parsed.document_id is not None:
            document = await self._metadata_store.get_document_lifecycle(
                kb_info["kb_id"], parsed.document_id
            )
            if document is not None and document.workspace != parsed.workspace:
                spec.classification = "unknown_owner"
                spec.reason_codes.append("workspace_mismatch")
                return spec

        reason_codes: list[str] = []
        if await self._is_live_reference(
            target_uri=target_uri,
            target_kind="object",
            target_namespace=parsed.namespace or "artifact",
            parsed=parsed,
            kb_info=kb_info,
            document=document,
            reason_codes=reason_codes,
        ):
            spec.classification = "referenced"
            spec.reason_codes.extend(reason_codes)
            return spec

        if await self._is_retained_manifest_target(target_uri, reason_codes):
            spec.classification = "retained"
            spec.reason_codes.extend(reason_codes)
            return spec

        if await self._has_pending_manifest(target_uri, reason_codes):
            spec.classification = "referenced"
            spec.reason_codes.extend(reason_codes)
            return spec

        if parsed.document_id is not None and await self._has_active_document_job(
            kb_info["kb_id"], parsed.document_id, reason_codes
        ):
            spec.classification = "referenced"
            spec.reason_codes.extend(reason_codes)
            return spec

        if parsed.document_id is not None and await self._has_unknown_commit_outcome(
            kb_info["kb_id"], parsed.document_id, reason_codes
        ):
            spec.classification = "unknown_owner"
            spec.reason_codes.extend(reason_codes)
            return spec

        if await self._is_in_migration_items(target_uri, reason_codes):
            spec.classification = "referenced"
            spec.reason_codes.extend(reason_codes)
            return spec

        spec.classification = "eligible"
        return spec

    def _is_too_new(self, last_modified: datetime) -> bool:
        if self._min_age_hours <= 0:
            return False
        try:
            normalized = _parse_utc_datetime(last_modified)
        except ValueError:
            return False
        cutoff = self._now() - timedelta(hours=self._min_age_hours)
        return normalized > cutoff

    def _malformed_spec(self, entry: Any, *, reason: str) -> OrphanCandidateSpec:
        return OrphanCandidateSpec(
            target_uri=getattr(entry, "uri", "") or "",
            relative_object_id=getattr(entry, "key", "") or "",
            classification="malformed",
            parsed=None,
            last_modified=entry.last_modified,
            size_bytes=getattr(entry, "size", None),
            etag=getattr(entry, "etag", None),
            checksum=getattr(entry, "checksum", None),
            version_id=getattr(entry, "version_id", None),
            reason_codes=[reason],
        )

    async def _is_live_reference(
        self,
        *,
        target_uri: str,
        target_kind: ArtifactCleanupTargetKind,
        target_namespace: ArtifactCleanupTargetNamespace,
        parsed: ParsedObjectKey,
        kb_info: Mapping[str, Any],
        document: Any,
        reason_codes: list[str],
    ) -> bool:
        if document is None:
            return False

        metadata = document.metadata if isinstance(document.metadata, dict) else {}
        normalized_target = normalize_artifact_target_uri(target_uri)

        if target_namespace in {"source", "legacy_source"}:
            source_uris = _explicit_string_values(
                metadata,
                ("current_source_object_uri", "source_object_uri"),
            )
            if isinstance(document.source_uri, str) and document.source_uri:
                source_uris.add(document.source_uri)
            if any(
                _reference_matches_target(
                    normalized_target,
                    target_kind=target_kind,
                    reference_uri=value,
                )
                for value in source_uris
            ):
                reason_codes.append("current_source_reference")
                return True
            if target_namespace == "source" and parsed.source_generation_id:
                current_generations = _explicit_string_values(
                    metadata,
                    ("current_source_generation_id", "source_generation_id"),
                )
                if parsed.source_generation_id in current_generations:
                    reason_codes.append("current_source_generation")
                    return True

        current_artifact_ids = _current_artifact_ids(metadata)
        if (
            target_namespace == "artifact"
            and parsed.artifact_id in current_artifact_ids
        ):
            reason_codes.append("current_artifact_reference")
            return True

        # Tombstoned documents are not "live" -- cleanup manifests own their
        # objects.  Returning False here lets the reconciler enqueue an
        # orphan_reconcile manifest if no other authority still holds it.
        if document.deleted_at is not None:
            return False

        artifacts, total = await self._metadata_store.list_document_artifacts(
            kb_info["kb_id"],
            parsed.document_id or "",
            limit=200,
            offset=0,
        )
        if total > len(artifacts):
            reason_codes.append("artifact_reference_query_overflow")
            return True
        for artifact in artifacts:
            references = _artifact_reference_uris(artifact)
            for value in references:
                if _reference_matches_target(
                    normalized_target,
                    target_kind=target_kind,
                    reference_uri=value,
                ):
                    reason_codes.append("current_artifact_uri_reference")
                    return True
        return False

    async def _is_retained_manifest_target(
        self, target_uri: str, reason_codes: list[str]
    ) -> bool:
        manifests, total = await self._metadata_store.list_artifact_cleanup_manifests(
            target_uri=target_uri,
            statuses=("retained",),
            limit=50,
        )
        if total > len(manifests):
            reason_codes.append("retained_query_overflow")
            return True
        if manifests:
            reason_codes.append("retained_manifest_holds_target")
            return True
        return False

    async def _has_pending_manifest(
        self, target_uri: str, reason_codes: list[str]
    ) -> bool:
        manifests, total = await self._metadata_store.list_artifact_cleanup_manifests(
            target_uri=target_uri,
            statuses=("pending", "leased", "blocked"),
            limit=50,
        )
        if total > len(manifests):
            reason_codes.append("pending_manifest_query_overflow")
            return True
        if manifests:
            reason_codes.append("pending_manifest_holds_target")
            return True
        return False

    async def _has_active_document_job(
        self,
        kb_id: str,
        document_id: str,
        reason_codes: list[str],
    ) -> bool:
        jobs, total = await self._metadata_store.list_jobs(
            kb_id,
            statuses=_ACTIVE_JOB_STATUSES,
            document_id=document_id,
            limit=50,
            offset=0,
        )
        if total > 0:
            reason_codes.append("active_document_job")
            return True
        return False

    async def _has_unknown_commit_outcome(
        self,
        kb_id: str,
        document_id: str,
        reason_codes: list[str],
    ) -> bool:
        jobs, total = await self._metadata_store.list_jobs(
            kb_id,
            document_id=document_id,
            limit=200,
            offset=0,
        )
        for job in jobs:
            if _job_has_unknown_commit_outcome(job):
                reason_codes.append("metadata_commit_outcome_unknown")
                return True
        return False

    async def _is_in_migration_items(
        self, target_uri: str, reason_codes: list[str]
    ) -> bool:
        digest = artifact_target_uri_digest(target_uri)
        runs, _ = await self._metadata_store.list_artifact_maintenance_runs(
            kind="migration",
            limit=20,
        )
        for run in runs:
            if run.status in {"cancelled", "failed"}:
                continue
            items, _ = await self._metadata_store.list_artifact_maintenance_items(
                run.id,
                target_uri_digest=digest,
                limit=50,
            )
            non_terminal = [
                item for item in items if item.state not in _TERMINAL_ITEM_STATES
            ]
            if non_terminal:
                reason_codes.append("migration_item_active")
                return True
        return False

    async def _build_kb_workspace_index(self) -> dict[str, dict[str, Any]]:
        rows = await self._kb_lifecycle_provider()
        index: dict[str, dict[str, Any]] = {}
        for row in rows:
            kb_id = getattr(row, "kb_id", None)
            generation = getattr(row, "generation", None)
            state = getattr(row, "state", None)
            if not kb_id or not generation or not state:
                continue
            try:
                workspace = sanitize_workspace(kb_id)
            except ValueError:
                continue
            index[workspace] = {
                "kb_id": kb_id,
                "generation": generation,
                "state": state,
            }
        return index

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    async def apply_plan(
        self,
        plan_id: str,
        *,
        release_retained: bool = False,
        resume: bool = False,
        actor_id: str | None = None,
    ) -> OrphanReconcileApplySummary:
        """Apply (or resume) a previously-succeeded dry-run plan.

        Enqueue a cleanup manifest for each ``eligible`` item; release retained
        manifests only when ``release_retained`` is true; mark every other item
        as skipped.  Never deletes objects directly.
        """

        try:
            parent = await self._metadata_store.get_artifact_maintenance_run(plan_id)
        except ArtifactLifecycleNotFoundError as exc:
            raise OrphanReconcilePlanError(
                "Provided --plan-id is not a known orphan reconcile plan"
            ) from exc
        if parent.kind != _MAINTENANCE_KIND or parent.mode != "dry_run":
            raise OrphanReconcilePlanError(
                "Provided --plan-id is not an orphan reconcile dry-run plan"
            )
        if parent.status != "succeeded":
            raise OrphanReconcilePlanError(
                "Dry-run plan must reach 'succeeded' before apply"
            )

        apply_run = await self._ensure_apply_run(parent, actor_id=actor_id)
        if apply_run.status in {"succeeded", "cancelled"}:
            counters = await self._metadata_store.aggregate_artifact_maintenance_items(
                apply_run.id
            )
            return self._apply_summary(plan_id, apply_run.id, counters, issues=[])

        claimed = await self._claim_apply_run(apply_run.id, resume=resume)
        lease_token = claimed.lease_token or ""
        items, _total = await self._metadata_store.list_artifact_maintenance_items(
            apply_run.id, limit=_MAX_ITEMS_PER_RUN
        )

        issues: list[str] = []
        for item in items:
            try:
                await self._advance_item(
                    item=item,
                    run_lease_token=lease_token,
                    release_retained=release_retained,
                )
            except OrphanReconcileApplyGuardError as exc:
                issues.append(f"item {item.item_key} blocked: {redact_value(exc)}")
            except ArtifactLifecycleError as exc:
                issues.append(
                    f"item {item.item_key} lifecycle error: {redact_value(exc)}"
                )

        counters = await self._metadata_store.aggregate_artifact_maintenance_items(
            apply_run.id
        )
        verified = counters.get("verified", 0)
        failed = counters.get("failed", 0) + counters.get("blocked", 0)
        if verified + counters.get("skipped", 0) == counters.get("total", 0) and (
            failed == 0 or counters.get("total", 0) == 0
        ):
            await self._metadata_store.transition_artifact_maintenance_run(
                apply_run.id,
                expected_status="running",
                new_status="succeeded",
                lease_owner=claimed.lease_owner,
                lease_token=claimed.lease_token,
                counters={
                    "total_items": counters.get("total", 0),
                    "verified_items": verified,
                    "planned_items": counters.get("planned", 0),
                    "uploaded_items": counters.get("uploaded", 0),
                    "applied_items": counters.get("applied", 0),
                    "failed_items": counters.get("failed", 0),
                    "blocked_items": counters.get("blocked", 0),
                    "skipped_items": counters.get("skipped", 0),
                },
            )
        else:
            await self._metadata_store.recover_expired_artifact_maintenance_run_leases(
                limit=10
            )

        return self._apply_summary(plan_id, apply_run.id, counters, issues=issues)

    async def _advance_item(
        self,
        *,
        item: ArtifactMaintenanceItemRecord,
        run_lease_token: str,
        release_retained: bool,
    ) -> None:
        if item.state in _TERMINAL_ITEM_STATES:
            return

        payload = _decode_payload(item.payload_json)
        classification = payload.get("classification")
        if classification not in _APPLY_CLASSIFICATIONS:
            # Report-only: never enqueue, never delete.
            await self._skip_item(item, run_lease_token)
            return

        # Re-validate eligibility at apply time.  References may have appeared
        # between plan and apply.
        target_uri = self._safe_target_uri(item.relative_object_id)
        reason_codes: list[str] = []
        revalidated = await self._revalidate_apply_target(
            item=item,
            target_uri=target_uri,
            release_retained=release_retained,
            reason_codes=reason_codes,
        )
        if not revalidated:
            await self._skip_item(item, run_lease_token)
            return

        if classification == "eligible":
            await self._enqueue_cleanup_manifest(
                item=item,
                target_uri=target_uri,
                run_lease_token=run_lease_token,
                reason_codes=reason_codes,
            )
        else:  # retained
            await self._release_retained_manifests(
                item=item,
                target_uri=target_uri,
                run_lease_token=run_lease_token,
                reason_codes=reason_codes,
            )

    async def _revalidate_apply_target(
        self,
        *,
        item: ArtifactMaintenanceItemRecord,
        target_uri: str,
        release_retained: bool,
        reason_codes: list[str],
    ) -> bool:
        """Confirm the planned classification is still valid at apply time.

        Returns ``True`` if the apply action (enqueue or release) may proceed,
        ``False`` if the item must be skipped.  ``reason_codes`` is populated
        with the durable codes for audit output.
        """

        payload = _decode_payload(item.payload_json)
        classification = payload.get("classification")
        kb_index = await self._build_kb_workspace_index()

        workspace = item.workspace or (
            payload.get("workspace") if isinstance(payload, dict) else None
        )
        document_id = item.document_id
        # Re-derive source_generation_id from the durable relative object id;
        # the frozen payload validator forbids persisting it directly.
        parsed_key = _parse_object_key(
            item.relative_object_id, configured_prefix=self._prefix
        )
        source_generation_id = parsed_key.source_generation_id if parsed_key else None
        kb_info = kb_index.get(workspace) if workspace else None

        if kb_info is not None and document_id is not None:
            document = await self._metadata_store.get_document_lifecycle(
                kb_info["kb_id"], document_id
            )
            namespace: ArtifactCleanupTargetNamespace = (
                parsed_key.namespace
                if parsed_key and parsed_key.namespace
                else "artifact"
            )
            parsed = ParsedObjectKey(
                workspace=workspace or "",
                document_id=document_id,
                namespace=namespace,
                source_generation_id=source_generation_id,
                artifact_id=item.artifact_id,
                origin_job_id=None,
                origin_attempt_token=None,
            )
            live_reasons: list[str] = []
            if document is not None and await self._is_live_reference(
                target_uri=target_uri,
                target_kind="object",
                target_namespace=namespace,
                parsed=parsed,
                kb_info=kb_info,
                document=document,
                reason_codes=live_reasons,
            ):
                reason_codes.extend(live_reasons)
                return False
            if await self._has_pending_manifest(target_uri, reason_codes):
                return False
            if await self._has_unknown_commit_outcome(
                kb_info["kb_id"], document_id, reason_codes
            ):
                return False

        if classification == "retained":
            retained_reasons: list[str] = []
            has_retained = await self._is_retained_manifest_target(
                target_uri, retained_reasons
            )
            if not has_retained and not release_retained:
                reason_codes.append("retained_manifest_already_released")
                return False
            if not release_retained:
                # Plan said retained but caller did not opt in to release.
                reason_codes.append("release_retained_not_confirmed")
                return False
            return has_retained

        return True

    async def _enqueue_cleanup_manifest(
        self,
        *,
        item: ArtifactMaintenanceItemRecord,
        target_uri: str,
        run_lease_token: str,
        reason_codes: list[str],
    ) -> None:
        parsed_key = _parse_object_key(
            item.relative_object_id, configured_prefix=self._prefix
        )
        target_namespace: ArtifactCleanupTargetNamespace = (
            parsed_key.namespace if parsed_key and parsed_key.namespace else "artifact"
        )
        if target_namespace == "staging":
            # Staging cleanup requires job+attempt authority which an orphan
            # reclaim never has; treat staging items as skipped at apply time.
            await self._skip_item(item, run_lease_token)
            return
        kb_id = item.kb_id
        kb_generation = item.kb_generation
        workspace = item.workspace
        source_generation_id = parsed_key.source_generation_id if parsed_key else None
        if not kb_id or not kb_generation or not workspace:
            await self._skip_item(item, run_lease_token)
            return
        normalized = normalize_artifact_target_uri(target_uri)
        now = self._now()
        delete_after, cleanup_deadline, audit_retain = self._grace_windows(now)
        idempotency_key = artifact_cleanup_idempotency_key(
            reason="orphan_reconcile",
            kb_id=kb_id,
            kb_generation=kb_generation,
            workspace=workspace,
            document_id=item.document_id,
            artifact_id=item.artifact_id,
            source_generation_id=source_generation_id,
            target_kind="object",
            target_namespace=target_namespace,
            target_uri=normalized,
        )
        manifest_id = _orphan_manifest_id(
            idempotency_key=idempotency_key,
            relative_object_id=item.relative_object_id,
        )
        manifest = ArtifactCleanupManifestRecord(
            id=manifest_id,
            idempotency_key=idempotency_key,
            manifest_group_id=f"orphan-reconcile:{item.run_id}",
            kb_id=kb_id,
            kb_generation=kb_generation,
            workspace=workspace,
            document_id=item.document_id,
            artifact_id=item.artifact_id,
            source_generation_id=source_generation_id,
            origin_job_id=None,
            origin_attempt_token=None,
            reason="orphan_reconcile",
            target_kind="object",
            target_namespace=target_namespace,
            disposition="delete",
            status="pending",
            target_uri=normalized,
            expected_size_bytes=item.expected_size_bytes,
            expected_checksum=item.expected_checksum,
            expected_etag=None,
            expected_version_id=None,
            delete_after=delete_after,
            cleanup_deadline_at=cleanup_deadline,
            audit_retain_until=audit_retain,
            next_attempt_at=delete_after,
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
        )
        try:
            await self._metadata_store.enqueue_artifact_cleanup_manifest(manifest)
        except ArtifactLifecycleConflictError:
            # Idempotent replay: a manifest with the same idempotency key and
            # equivalent payload already exists. Treat as enqueued.
            reason_codes.append("enqueue_idempotent_replay")
        # Transition item: planned -> verified (apply's job is only to enqueue;
        # verified deletion is the cleanup service's responsibility).
        try:
            await self._metadata_store.transition_artifact_maintenance_item(
                item.run_id,
                item.item_key,
                expected_state="planned",
                new_state="verified",
                run_lease_token=run_lease_token,
                increment_attempt=True,
            )
        except ArtifactLifecycleStateError:
            # Concurrent resume advanced the item; nothing more to do.
            pass

    async def _release_retained_manifests(
        self,
        *,
        item: ArtifactMaintenanceItemRecord,
        target_uri: str,
        run_lease_token: str,
        reason_codes: list[str],
    ) -> None:
        kb_id = item.kb_id
        kb_generation = item.kb_generation
        if not kb_id or not kb_generation:
            await self._skip_item(item, run_lease_token)
            return
        retained, total = await self._metadata_store.list_artifact_cleanup_manifests(
            target_uri=target_uri,
            statuses=("retained",),
            limit=50,
        )
        if not retained:
            await self._skip_item(item, run_lease_token)
            return
        for manifest in retained:
            await self._metadata_store.release_retained_artifact_cleanup_manifests(
                kb_id=manifest.kb_id,
                kb_generation=manifest.kb_generation,
                manifest_group_id=manifest.manifest_group_id,
                manifest_ids=[manifest.id],
            )
        try:
            await self._metadata_store.transition_artifact_maintenance_item(
                item.run_id,
                item.item_key,
                expected_state="planned",
                new_state="verified",
                run_lease_token=run_lease_token,
                increment_attempt=True,
            )
        except ArtifactLifecycleStateError:
            pass

    async def _skip_item(
        self,
        item: ArtifactMaintenanceItemRecord,
        run_lease_token: str,
    ) -> None:
        try:
            await self._metadata_store.transition_artifact_maintenance_item(
                item.run_id,
                item.item_key,
                expected_state=item.state,
                new_state="skipped",
                run_lease_token=run_lease_token,
            )
        except ArtifactLifecycleStateError:
            pass

    def _apply_summary(
        self,
        plan_id: str,
        apply_run_id: str,
        counters: Mapping[str, int],
        *,
        issues: list[str],
    ) -> OrphanReconcileApplySummary:
        return OrphanReconcileApplySummary(
            plan_id=plan_id,
            apply_run_id=apply_run_id,
            items_total=counters.get("total", 0),
            items_enqueued=counters.get("verified", 0),
            items_skipped=counters.get("skipped", 0),
            items_blocked=counters.get("blocked", 0),
            items_failed=counters.get("failed", 0),
            counters=dict(counters),
            issues=issues,
        )

    async def _ensure_apply_run(
        self,
        parent: ArtifactMaintenanceRunRecord,
        *,
        actor_id: str | None,
    ) -> ArtifactMaintenanceRunRecord:
        apply_run = ArtifactMaintenanceRunRecord(
            id=self._mint_run_id("apply", parent_id=parent.id),
            kind=_MAINTENANCE_KIND,
            mode="apply",
            status="planned",
            metadata_backend=self._metadata_backend,
            parent_plan_id=parent.id,
            backend_fingerprint=parent.backend_fingerprint,
            scope_fingerprint=parent.scope_fingerprint,
            config_fingerprint=parent.config_fingerprint,
            scope_json=parent.scope_json,
            actor_id=actor_id,
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
        )
        try:
            created = await self._metadata_store.create_artifact_maintenance_run(
                apply_run
            )
        except ArtifactLifecycleConflictError as exc:
            runs, _ = await self._metadata_store.list_artifact_maintenance_runs(
                kind=_MAINTENANCE_KIND,
                mode="apply",
                parent_plan_id=parent.id,
                limit=10,
            )
            for candidate in runs:
                if candidate.idempotency_key == apply_run.idempotency_key:
                    return candidate
            raise OrphanReconcilePlanError(
                "Apply run could not be established for this plan"
            ) from exc

        plan_items, _ = await self._metadata_store.list_artifact_maintenance_items(
            parent.id, limit=_MAX_ITEMS_PER_RUN
        )
        if plan_items:
            apply_items: list[ArtifactMaintenanceItemRecord] = []
            for ordinal, plan_item in enumerate(plan_items):
                apply_items.append(
                    self._clone_item_for_run(created.id, ordinal, plan_item)
                )
            for offset in range(0, len(apply_items), _MAX_ITEMS_PER_RUN):
                batch = apply_items[offset : offset + _MAX_ITEMS_PER_RUN]
                await self._metadata_store.create_artifact_maintenance_items(batch)
        return created

    def _clone_item_for_run(
        self,
        run_id: str,
        ordinal: int,
        plan_item: ArtifactMaintenanceItemRecord,
    ) -> ArtifactMaintenanceItemRecord:
        new_item_key = artifact_maintenance_item_key(
            run_id=run_id,
            subject_kind=plan_item.subject_kind,
            subject_id=plan_item.subject_id,
            kb_id=plan_item.kb_id,
            kb_generation=plan_item.kb_generation,
            workspace=plan_item.workspace,
            document_id=plan_item.document_id,
            artifact_id=plan_item.artifact_id,
            logical_group_id=plan_item.logical_group_id,
            relative_object_id=plan_item.relative_object_id,
            root_label=plan_item.root_label,
            expected_checksum=plan_item.expected_checksum,
            expected_size_bytes=plan_item.expected_size_bytes,
            target_uri_authority=plan_item.target_uri_authority,
            target_uri_digest=plan_item.target_uri_digest,
            payload_json=plan_item.payload_json,
        )
        return ArtifactMaintenanceItemRecord(
            id=f"{run_id}-item-{ordinal:04d}",
            run_id=run_id,
            item_key=new_item_key,
            state="planned",
            ordinal=ordinal,
            subject_kind=plan_item.subject_kind,
            subject_id=plan_item.subject_id,
            kb_id=plan_item.kb_id,
            kb_generation=plan_item.kb_generation,
            workspace=plan_item.workspace,
            document_id=plan_item.document_id,
            artifact_id=plan_item.artifact_id,
            logical_group_id=plan_item.logical_group_id,
            relative_object_id=plan_item.relative_object_id,
            root_label=plan_item.root_label,
            expected_checksum=plan_item.expected_checksum,
            expected_size_bytes=plan_item.expected_size_bytes,
            target_uri_authority=plan_item.target_uri_authority,
            target_uri_digest=plan_item.target_uri_digest,
            payload_json=plan_item.payload_json,
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
        )

    async def _claim_apply_run(
        self, apply_run_id: str, *, resume: bool
    ) -> ArtifactMaintenanceRunRecord:
        if resume:
            await self._metadata_store.recover_expired_artifact_maintenance_run_leases(
                limit=20
            )
        try:
            return await self._metadata_store.claim_artifact_maintenance_run(
                apply_run_id,
                lease_owner="orphan-reconcile-apply",
                lease_duration_seconds=self._lease_duration_seconds,
            )
        except ArtifactLifecycleStateError as exc:
            raise OrphanReconcileApplyGuardError(
                "Apply run is not claimable in its current state; "
                "use --resume after the lease expires"
            ) from exc

    # ------------------------------------------------------------------
    # Record builders
    # ------------------------------------------------------------------

    def _mint_run_id(self, kind: str, *, parent_id: str | None = None) -> str:
        token = secrets.token_urlsafe(9)
        suffix = re.sub(r"[^A-Za-z0-9._-]", "", token)[:12]
        if kind == "plan":
            return f"or-plan-{suffix}"
        if parent_id:
            parent_suffix = parent_id.rsplit("-", 1)[-1]
            return f"or-apply-{parent_suffix}"
        return f"or-apply-{suffix}"

    def _spec_to_item(
        self, run_id: str, ordinal: int, spec: OrphanCandidateSpec
    ) -> ArtifactMaintenanceItemRecord:
        # For malformed objects (parsed is None) the target URI may carry
        # scratch/unowned text that the durable-safe validator rejects.  Use
        # a placeholder authority for those rows; the relative_object_id is
        # still persisted for audit.
        if spec.parsed is not None:
            target_uri = normalize_artifact_target_uri(spec.target_uri)
            target_uri_digest = artifact_target_uri_digest(target_uri)
            target_uri_authority = _uri_authority(target_uri)
            workspace = spec.parsed.workspace
            document_id = spec.parsed.document_id
            artifact_id = spec.parsed.artifact_id
            subject_kind = spec.parsed.namespace or "artifact"
            expected_checksum = spec.checksum
            expected_size_bytes = spec.size_bytes
        else:
            # Derive a safe synthetic authority that survives the durable-safe
            # validator.  The actual object identity is preserved on the item
            # only via ``relative_object_id``; the manifest enqueue path will
            # skip malformed items at apply time, so this authority is never
            # used to enqueue a real cleanup manifest.
            safe_key = spec.relative_object_id or "unowned"
            target_uri = self._object_storage.object_uri_for_key(safe_key)
            try:
                target_uri = normalize_artifact_target_uri(target_uri)
            except ValueError:
                target_uri = normalize_artifact_target_uri(
                    self._object_storage.object_uri_for_key("unowned/malformed")
                )
            target_uri_digest = artifact_target_uri_digest(target_uri)
            target_uri_authority = _uri_authority(target_uri)
            workspace = "unowned"
            document_id = None
            artifact_id = None
            subject_kind = "unowned"
            expected_checksum = None
            expected_size_bytes = None
        subject_id = _subject_id_for(spec)
        logical_group_id = f"orphan:{workspace or 'unowned'}"
        # ``payload_json`` is the only place to carry classification metadata.
        # The frozen maintenance payload validator forbids authoritative /
        # path-bearing keys (size_bytes, checksum, document_id, *_uri, *_token,
        # *_path, etc.) and rejects any string containing ``/``.  Classification
        # and reason codes are short alphanumeric+underscore tokens; size,
        # checksum, namespace, source_generation_id, etag, and version_id all
        # either live on the top-level item record or are re-derived from the
        # durable relative_object_id at apply time.
        payload = {
            "classification": spec.classification,
            "reason_codes": list(spec.reason_codes),
        }
        item_key = artifact_maintenance_item_key(
            run_id=run_id,
            subject_kind=subject_kind,
            subject_id=subject_id,
            kb_id=spec.kb_id,
            kb_generation=spec.kb_generation,
            workspace=workspace,
            document_id=document_id,
            artifact_id=artifact_id,
            logical_group_id=logical_group_id,
            relative_object_id=spec.relative_object_id,
            root_label=None,
            expected_checksum=expected_checksum,
            expected_size_bytes=expected_size_bytes,
            target_uri_authority=target_uri_authority,
            target_uri_digest=target_uri_digest,
            payload_json=payload,
        )
        return ArtifactMaintenanceItemRecord(
            id=f"{run_id}-item-{ordinal:04d}",
            run_id=run_id,
            item_key=item_key,
            state="planned",
            ordinal=ordinal,
            subject_kind=subject_kind,
            subject_id=subject_id,
            kb_id=spec.kb_id,
            kb_generation=spec.kb_generation,
            workspace=workspace,
            document_id=document_id,
            artifact_id=artifact_id,
            logical_group_id=logical_group_id,
            relative_object_id=spec.relative_object_id,
            root_label=None,
            expected_checksum=expected_checksum,
            expected_size_bytes=expected_size_bytes,
            target_uri_authority=target_uri_authority,
            target_uri_digest=target_uri_digest,
            payload_json=payload,
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _grace_windows(self, now: datetime) -> tuple[datetime, datetime, datetime]:
        delete_after = now + timedelta(hours=max(self._min_age_hours, 1))
        cleanup_deadline = delete_after + timedelta(hours=24)
        audit_retain = now + timedelta(days=30)
        return delete_after, cleanup_deadline, audit_retain

    def _safe_target_uri(self, relative_object_id: str) -> str:
        return self._object_storage.object_uri_for_key(relative_object_id)

    def _target_uri_authority(self) -> str:
        probe_uri = self._object_storage.object_uri_for_key(
            self._prefix + "/reconcile-probe"
        )
        return _uri_authority(probe_uri)

    def _build_scope_payload(self) -> dict[str, Any]:
        return {
            "bucket": self._bucket,
            "object_prefix": self._prefix,
            "kind": _MAINTENANCE_KIND,
        }


# ---------------------------------------------------------------------------
# Object key parser
# ---------------------------------------------------------------------------


def _parse_object_key(
    relative_object_id: str,
    *,
    configured_prefix: str,
) -> ParsedObjectKey | None:
    """Parse a validated object key into its ownership segments.

    Returns ``None`` when the key does not match the validated
    workspace/document/source|artifact|staging structure.  This is the same
    shape the S3 ``validate_cleanup_target`` primitive enforces, but expressed
    read-only so this module never edits the store.
    """

    parts = relative_object_id.split("/")
    configured_parts = configured_prefix.split("/") if configured_prefix else []
    workspace_header = [*configured_parts, "workspaces"]
    if parts[: len(workspace_header)] != workspace_header:
        return None
    if len(parts) < len(workspace_header) + 2:
        return None
    workspace = parts[len(workspace_header)]
    if not workspace:
        return None
    if parts[len(workspace_header) + 1] != "documents":
        return None
    after_documents = parts[len(workspace_header) + 2 :]
    if not after_documents or not after_documents[0]:
        return None
    document_id = after_documents[0]
    remainder = after_documents[1:]
    if not remainder:
        # Workspace/document-prefix object (no namespace) is not produced by
        # validated writers; treat as malformed.
        return None
    namespace_segment = remainder[0]
    if namespace_segment == "source":
        if len(remainder) >= 2 and remainder[1] == "generations":
            if len(remainder) < 3 or not remainder[2]:
                return None
            source_generation_id = remainder[2]
            return ParsedObjectKey(
                workspace=workspace,
                document_id=document_id,
                namespace="source",
                source_generation_id=source_generation_id,
                artifact_id=None,
                origin_job_id=None,
                origin_attempt_token=None,
            )
        # legacy source object: documents/<id>/source/<file>
        if len(remainder) != 2 or not remainder[1]:
            return None
        return ParsedObjectKey(
            workspace=workspace,
            document_id=document_id,
            namespace="legacy_source",
            source_generation_id=None,
            artifact_id=None,
            origin_job_id=None,
            origin_attempt_token=None,
        )
    if namespace_segment == "artifacts":
        if len(remainder) < 3 or not remainder[1] or not remainder[2]:
            return None
        artifact_id = remainder[2]
        return ParsedObjectKey(
            workspace=workspace,
            document_id=document_id,
            namespace="artifact",
            source_generation_id=None,
            artifact_id=artifact_id,
            origin_job_id=None,
            origin_attempt_token=None,
        )
    if namespace_segment == "staging":
        if (
            len(remainder) < 4
            or not remainder[1]
            or not remainder[2]
            or not remainder[3]
        ):
            return None
        return ParsedObjectKey(
            workspace=workspace,
            document_id=document_id,
            namespace="staging",
            source_generation_id=None,
            artifact_id=None,
            origin_job_id=remainder[1],
            origin_attempt_token=remainder[2],
        )
    return None


# ---------------------------------------------------------------------------
# Free-standing helpers (mirrors artifact_cleanup_service internals, read-only)
# ---------------------------------------------------------------------------


def _explicit_string_values(
    mapping: Mapping[str, Any], fields: Sequence[str]
) -> set[str]:
    return {
        value
        for field_name in fields
        if isinstance((value := mapping.get(field_name)), str) and value
    }


def _current_artifact_ids(metadata: Mapping[str, Any]) -> set[str]:
    result = _explicit_string_values(metadata, _CURRENT_ARTIFACT_ID_FIELDS)
    for field_name in _CURRENT_ARTIFACT_ID_LIST_FIELDS:
        values = metadata.get(field_name)
        if isinstance(values, list):
            result.update(value for value in values if isinstance(value, str) and value)
    binding = metadata.get("artifact_binding")
    if isinstance(binding, dict):
        result.update(
            _explicit_string_values(
                binding,
                ("sidecar_artifact_id", "blocks_artifact_id"),
            )
        )
        raw_ids = binding.get("raw_artifact_ids")
        if isinstance(raw_ids, list):
            result.update(
                value for value in raw_ids if isinstance(value, str) and value
            )
    return result


def _artifact_reference_uris(artifact: Any) -> set[str]:
    values: set[str] = set()
    if isinstance(artifact.uri, str) and artifact.uri:
        values.add(artifact.uri)
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    values.update(
        _explicit_string_values(metadata, ("object_uri", "object_prefix_uri"))
    )
    return values


def _normalize_reference_uri(value: str) -> str | None:
    try:
        return normalize_artifact_target_uri(value)
    except (TypeError, ValueError):
        return None


def _reference_matches_target(
    target_uri: str,
    *,
    target_kind: str,
    reference_uri: str,
) -> bool:
    normalized_reference = _normalize_reference_uri(reference_uri)
    if normalized_reference is None:
        return False
    if target_kind == "object":
        return normalized_reference == target_uri
    return normalized_reference.startswith(target_uri)


def _job_has_unknown_commit_outcome(job: Any) -> bool:
    if getattr(job, "error_code", None) == "metadata_commit_outcome_unknown":
        return True
    for mapping in (getattr(job, "payload", None), getattr(job, "result", None)):
        if not isinstance(mapping, dict):
            continue
        if mapping.get("metadata_commit_outcome_unknown") is True:
            return True
        for field_name in (
            "error_code",
            "metadata_commit_error_code",
            "commit_outcome",
            "metadata_commit_outcome",
        ):
            value = mapping.get(field_name)
            if isinstance(value, str) and value.strip().lower() in {
                "indeterminate",
                "pending",
                "unresolved",
                "unknown",
                "metadata_commit_outcome_unknown",
            }:
                return True
    return False


def _parse_utc_datetime(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("cleanup time must be a valid UTC timestamp") from exc
    else:
        raise ValueError("cleanup time must be a datetime or ISO timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("cleanup time must include a timezone")
    return parsed.astimezone(timezone.utc)


def _decode_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        import json

        try:
            decoded = json.loads(payload)
        except (TypeError, ValueError):
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _count_classifications(
    specs: Sequence[OrphanCandidateSpec],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for spec in specs:
        counts[spec.classification] = counts.get(spec.classification, 0) + 1
    return counts


def _subject_id_for(spec: OrphanCandidateSpec) -> str:
    """Derive a non-path authority identifier for one candidate."""

    seed = spec.relative_object_id
    if spec.parsed and spec.parsed.document_id:
        seed = f"{spec.parsed.workspace}:{spec.parsed.document_id}:{seed}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"orphan:{digest}"


def _uri_authority(target_uri: str) -> str:
    parsed = urlparse(target_uri)
    return normalize_artifact_target_uri_authority(f"{parsed.scheme}://{parsed.netloc}")


def _orphan_manifest_id(*, idempotency_key: str, relative_object_id: str) -> str:
    seed = f"orphan-reconcile:{idempotency_key}:{relative_object_id}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]
    return f"or-manifest-{digest}"


def _fingerprint_json(payload: Mapping[str, Any]) -> str:
    import json

    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "OrphanCandidateSpec",
    "OrphanClassification",
    "OrphanReconcileApplyGuardError",
    "OrphanReconcileApplySummary",
    "OrphanReconcileError",
    "OrphanReconcilePlanError",
    "OrphanReconcilePlanSummary",
    "OrphanReconcileService",
    "ParsedObjectKey",
    "redact_mapping",
    "redact_value",
]
