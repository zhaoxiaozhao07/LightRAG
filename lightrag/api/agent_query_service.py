from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import partial
from typing import Any, AsyncIterator, Literal, cast
from uuid import uuid4

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from lightrag.api.agent_profile_service import effective_agent_profile
from lightrag.api.bilingual_query_service import (
    bilingual_applies,
    resolve_bilingual_mode,
)
from lightrag.api.chat_memory_routing import ChatMemoryScope, resolve_memory_injection
from lightrag.api.enterprise_auth import (
    agent_max_rounds,
    agent_query_enabled,
    append_enterprise_audit_event,
    enterprise_auth_enabled,
    get_enterprise_authorization_service,
    get_enterprise_user_agent_workflow_prompt_service,
    get_request_principal,
)
from lightrag.api.kb_service import KnowledgeBaseRecord, KnowledgeBaseService
from lightrag.api.llm_json_utils import LLMJsonError, call_llm_json
from lightrag.api.query_tool_service import (
    KBQueryFilters,
    QueryMode,
    QueryToolResult,
    QueryToolService,
)
from lightrag.constants import DEFAULT_QUERY_PRIORITY
from lightrag.prompt import PROMPTS
from lightrag.utils import logger, truncate_list_by_token_size

AGENT_ALLOWED_MODES: set[str] = {"local", "global", "hybrid", "naive", "mix"}

# Bounded retries for the planning call: local models occasionally emit
# invalid JSON; a failed plan is retried before the session fails.
AGENT_PLAN_LLM_ATTEMPTS = 3
_PRIORITY_RANK = {"P0": 0, "P1": 1, "P2": 2}
_PRIORITY_ALIASES = {
    "0": "P0",
    "critical": "P0",
    "high": "P0",
    "highest": "P0",
    "important": "P0",
    "must": "P0",
    "required": "P0",
    "urgent": "P0",
    "关键": "P0",
    "高": "P0",
    "高优先级": "P0",
    "重要": "P0",
    "必要": "P0",
    "1": "P1",
    "default": "P1",
    "medium": "P1",
    "normal": "P1",
    "standard": "P1",
    "一般": "P1",
    "中": "P1",
    "普通": "P1",
    "2": "P2",
    "low": "P2",
    "optional": "P2",
    "低": "P2",
    "低优先级": "P2",
    "可选": "P2",
}
_PLANNING_KB_WARN_THRESHOLD = 50

# One-shot fallback mode per retrieval mode, used when a step succeeds but
# returns zero chunks. The fallback trades the planner's mode choice for any
# evidence at all, so each entry switches retrieval family (graph <-> vector).
EMPTY_RETRY_MODE_FALLBACK: dict[str, str] = {
    "mix": "naive",
    "naive": "hybrid",
    "hybrid": "mix",
    "local": "hybrid",
    "global": "hybrid",
}

DEFAULT_AGENT_WORKFLOW_PROMPT = """
你是 LightRAG Agent 编排器。你只能在服务端提供的 allowed_kbs 中选择 kb_ids，
并为每个检索步骤指定底层检索模式 local/global/hybrid/naive/mix 之一。禁止使用 bypass。
你不直接回答最终问题，只输出严格 JSON。子 query 必须完整自洽，不要依赖“上文”。
法规、禁忌、合规类子问题优先标记 P0。若用户问题缺少关键约束且无法规划检索，输出澄清。
不要输出 markdown，不要输出 chain-of-thought。
""".strip()

# Appended to planner system prompts when bilingual retrieval is on: the
# planner then emits an alternate-language query + keywords per step so the
# executor can retrieve both language halves of a mixed zh/en corpus.
BILINGUAL_PLAN_PROMPT_SUFFIX = """
本次启用双语检索（payload 中 bilingual_retrieval=true）：知识库同时包含中文与英文文档。
为每个检索步骤额外生成 query_alt（该步子问题的另一语言完整版本：中文步骤给英文，英文步骤给中文）、
hl_keywords_alt 与 ll_keywords_alt（与 query_alt 同语言、语义对应的关键词）。
术语使用领域通用译法；型号、代号、化学式、标准号等记号原样保留，不要翻译。
""".strip()


def agent_bilingual_enabled(body: "AgentQueryRequest") -> bool:
    """Agent flows resolve bilingual mode from the request flag plus the
    deployment default only (steps span KBs, so per-KB config is not read)."""
    mode = resolve_bilingual_mode(body.bilingual, None)
    return bilingual_applies(mode, body.query, None)


class AgentPlanStep(BaseModel):
    step_index: int = Field(ge=1)
    title: str = ""
    query: str = Field(min_length=3)
    kb_ids: list[str] = Field(min_length=1)
    mode: Literal["local", "global", "hybrid", "naive", "mix"]
    priority: Literal["P0", "P1", "P2"] = "P1"
    hl_keywords: list[str] = Field(default_factory=list)
    ll_keywords: list[str] = Field(default_factory=list)
    # Bilingual retrieval (optional): alternate-language variant of this
    # step's sub-query plus matching keywords; empty when bilingual is off.
    query_alt: str = ""
    hl_keywords_alt: list[str] = Field(default_factory=list)
    ll_keywords_alt: list[str] = Field(default_factory=list)

    @field_validator("title", mode="before")
    @classmethod
    def _clip_title(cls, value: Any) -> str:
        return str(value if value is not None else "").strip()[:200]

    @field_validator("query", "query_alt", mode="before")
    @classmethod
    def _clip_query(cls, value: Any) -> str:
        return str(value if value is not None else "").strip()[:4096]

    @field_validator("priority", mode="before")
    @classmethod
    def _normalize_priority(cls, value: Any) -> str:
        text = str(value if value is not None else "").strip()
        if not text:
            return "P1"
        upper = text.upper()
        if upper in _PRIORITY_RANK:
            return upper
        normalized = text.lower().replace("_", "-").strip()
        return _PRIORITY_ALIASES.get(normalized, "P1")

    @field_validator(
        "kb_ids",
        "hl_keywords",
        "ll_keywords",
        "hl_keywords_alt",
        "ll_keywords_alt",
        mode="before",
    )
    @classmethod
    def _coerce_str_list(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value]
        if value is None:
            return []
        return [str(value)]


class AgentPlan(BaseModel):
    type: Literal["plan"] = "plan"
    clarification_required: bool = False
    clarification_question: str | None = None
    steps: list[AgentPlanStep] = Field(default_factory=list)
    notes_for_user: str | None = None


class AgentQueryRequest(BaseModel):
    query: str = Field(min_length=3)
    workflow: Literal["plan", "staged"] = "plan"
    candidate_kb_ids: list[str] | None = None
    max_rounds: int | None = Field(default=None, ge=1, le=20)
    response_type: str | None = None
    top_k: int | None = Field(default=None, ge=1)
    chunk_top_k: int | None = Field(default=None, ge=1)
    max_entity_tokens: int | None = Field(default=None, ge=1)
    max_relation_tokens: int | None = Field(default=None, ge=1)
    max_total_tokens: int | None = Field(default=None, ge=1)
    enable_rerank: bool | None = None
    include_references: bool = True
    include_chunk_content: bool = False
    filters: KBQueryFilters | None = None
    user_prompt: str | None = None
    conversation_history: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Prior conversation turns ([{role, content}]) passed to the "
            "planning and synthesis LLM as context (not used for retrieval)."
        ),
    )
    memory: ChatMemoryScope | None = Field(
        default=None,
        description=(
            "Server-side chat memory injection (docs/ChatMemory-zh.md §6). "
            "Provide the chat project_id to prepend that project's memory facts "
            "to the final-answer synthesis. Requires an interactive user owning "
            "the project and LIGHTRAG_CHAT_MEMORY_ENABLED."
        ),
    )
    bilingual: bool | None = Field(
        default=None,
        description=(
            "Explicit dual-path bilingual retrieval override for every "
            "planned step. Omit to follow the deployment default "
            "(BILINGUAL_QUERY_DEFAULT_MODE); requires BILINGUAL_QUERY_ENABLED."
        ),
    )

    @field_validator("query", mode="after")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        return value.strip()

    @field_validator("conversation_history", mode="after")
    @classmethod
    def _validate_history(
        cls, value: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        if value is None:
            return None
        for message in value:
            if "role" not in message:
                raise ValueError("Each message must have a 'role' key.")
            if not isinstance(message["role"], str) or not message["role"].strip():
                raise ValueError("Each message 'role' must be a non-empty string.")
        return value


@dataclass(slots=True)
class AgentEvidenceItem:
    reference_id: str
    kb_id: str
    round_index: int
    step_index: int
    mode: str
    file_path: str
    content: str
    chunk_id: str | None = None
    source_reference_id: str | None = None


@dataclass(slots=True)
class AgentRunResult:
    status: str
    session_id: str
    answer: str = ""
    clarification_question: str | None = None
    references: list[dict[str, Any]] = field(default_factory=list)
    steps_summary: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _json_event(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _dedup_agent_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    result: list[dict[str, Any]] = []
    for chunk in chunks:
        kb_id = chunk.get("kb_id")
        chunk_id = chunk.get("chunk_id")
        if kb_id and chunk_id:
            key = ("chunk", kb_id, chunk_id)
        else:
            key = (
                "content",
                kb_id,
                chunk.get("file_path") or chunk.get("source"),
                hashlib.sha256(str(chunk.get("content", "")).encode("utf-8")).hexdigest(),
            )
        if key in seen:
            continue
        seen.add(key)
        result.append(chunk)
    return result


def _interleave_rounds(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Round-robin merge evidence across retrieval steps.

    Each step's chunks are already relevance-ordered against that step's own
    sub-query (per-step rerank happens inside retrieval). Interleaving keeps
    every step represented near the front so the token-budget truncation trims
    each step's tail instead of dropping whole (late or differently-phrased)
    steps — deliberately NOT re-reranked against the umbrella question.
    """
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    order: list[tuple[int, int]] = []
    for chunk in chunks:
        key = (int(chunk.get("round_index") or 0), int(chunk.get("step_index") or 0))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(chunk)
    result: list[dict[str, Any]] = []
    index = 0
    appended = True
    while appended:
        appended = False
        for key in order:
            bucket = groups[key]
            if index < len(bucket):
                result.append(bucket[index])
                appended = True
        index += 1
    return result


def agent_kb_profile(record: KnowledgeBaseRecord) -> dict[str, Any]:
    return effective_agent_profile(record)


def _agent_plan_response_format(
    effective_records: list[KnowledgeBaseRecord], *, bilingual: bool
) -> dict[str, Any]:
    allowed_kb_ids = [record.id for record in effective_records]
    step_properties: dict[str, Any] = {
        "step_index": {"type": "integer", "minimum": 1},
        "title": {"type": "string"},
        "query": {"type": "string", "minLength": 3},
        "kb_ids": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "enum": allowed_kb_ids},
        },
        "mode": {"type": "string", "enum": sorted(AGENT_ALLOWED_MODES)},
        "priority": {"type": "string", "enum": list(_PRIORITY_RANK)},
        "hl_keywords": {"type": "array", "items": {"type": "string"}},
        "ll_keywords": {"type": "array", "items": {"type": "string"}},
    }
    if bilingual:
        step_properties.update(
            {
                "query_alt": {"type": "string"},
                "hl_keywords_alt": {"type": "array", "items": {"type": "string"}},
                "ll_keywords_alt": {"type": "array", "items": {"type": "string"}},
            }
        )
    schema = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["plan"]},
            "clarification_required": {"type": "boolean"},
            "clarification_question": {"type": ["string", "null"]},
            "steps": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": step_properties,
                    "required": [
                        "step_index",
                        "query",
                        "kb_ids",
                        "mode",
                        "priority",
                    ],
                },
            },
            "notes_for_user": {"type": ["string", "null"]},
        },
        "required": ["type", "clarification_required", "steps"],
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "agent_plan",
            "schema": schema,
        },
    }


class AgentQueryService:
    def __init__(
        self,
        *,
        kb_service: KnowledgeBaseService,
        query_tool_service: QueryToolService,
    ):
        self._kb_service = kb_service
        self._query_tool_service = query_tool_service

    async def run(
        self,
        *,
        request: Request,
        body: AgentQueryRequest,
        stream: bool = False,
    ) -> AgentRunResult:
        result: AgentRunResult | None = None
        async for event in self._run_events(
            request=request, body=body, stream_synthesis=stream
        ):
            candidate = event.get("_result")
            if isinstance(candidate, AgentRunResult):
                result = candidate
        if result is None:  # pragma: no cover — the generator always attaches one
            raise HTTPException(status_code=500, detail="Agent query produced no result")
        return result

    async def stream_events(
        self,
        *,
        request: Request,
        body: AgentQueryRequest,
    ) -> AsyncIterator[str]:
        try:
            async for event in self._run_events(
                request=request, body=body, stream_synthesis=True
            ):
                payload = {
                    key: value
                    for key, value in event.items()
                    if not key.startswith("_")
                }
                yield _json_event(payload)
        except HTTPException as exc:
            yield _json_event(
                {
                    "event": "error",
                    "error_code": "agent_http_error",
                    "status_code": exc.status_code,
                    "message": exc.detail,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Agent stream failed: %s", exc, exc_info=True)
            yield _json_event(
                {"event": "error", "error_code": "agent_error", "message": str(exc)}
            )

    async def _run_events(
        self,
        *,
        request: Request,
        body: AgentQueryRequest,
        stream_synthesis: bool,
    ) -> AsyncIterator[dict[str, Any]]:
        """Drive one Agent session, yielding progress events as they happen.

        Events are emitted live (planning → per-round retrieval → references →
        synthesis deltas) so the NDJSON stream shows progress during long
        sessions; ``run()`` consumes the same generator and only keeps the
        final result attached to the ``done`` event under ``_result``.
        """
        session_id = f"agent_{uuid4().hex}"
        self._require_agent_access(request)
        max_rounds = min(body.max_rounds or agent_max_rounds(), agent_max_rounds())
        effective_records = await self._effective_kbs(request, body.candidate_kb_ids)

        await append_enterprise_audit_event(
            request,
            "agent_session_started",
            target_type="agent_session",
            target_id=session_id,
            metadata={
                "workflow": body.workflow,
                "effective_kb_count": len(effective_records),
                "query_hash": hashlib.sha256(body.query.encode("utf-8")).hexdigest(),
                "max_rounds": max_rounds,
            },
        )
        try:
            yield {
                "event": "session_started",
                "session_id": session_id,
                "metadata": {
                    "workflow": body.workflow,
                    "effective_kb_ids": [record.id for record in effective_records],
                },
            }

            if body.workflow == "staged":
                # Local import: the staged module imports plan-mode helpers from
                # this module, so the dependency must stay one-directional at
                # import time.
                from lightrag.api.agent_staged_service import AgentStagedRunner

                async for event in AgentStagedRunner(self).run_events(
                    request=request,
                    body=body,
                    session_id=session_id,
                    effective_records=effective_records,
                    stream_synthesis=stream_synthesis,
                ):
                    yield event
                return

            plan = await self._plan(
                request=request,
                body=body,
                effective_records=effective_records,
                max_rounds=max_rounds,
            )
            if plan.clarification_required:
                result = AgentRunResult(
                    status="clarification_required",
                    session_id=session_id,
                    clarification_question=plan.clarification_question
                    or "请补充关键约束。",
                    metadata={
                        "workflow": body.workflow,
                        "effective_kb_ids": [record.id for record in effective_records],
                    },
                )
                yield {
                    "event": "clarification_required",
                    "session_id": session_id,
                    "clarification_question": result.clarification_question,
                }
                yield {"event": "done", "session_id": session_id, "_result": result}
                return

            steps, plan_truncated = self._validate_plan(
                plan, effective_records, max_rounds
            )
            yield {
                "event": "plan_created",
                "session_id": session_id,
                "plan_truncated": plan_truncated,
                "notes_for_user": plan.notes_for_user,
                "steps": [
                    {
                        "step_index": step.step_index,
                        "title": step.title or step.query[:80],
                        "query": step.query,
                        "kb_ids": step.kb_ids,
                        "mode": step.mode,
                        "priority": step.priority,
                    }
                    for step in steps
                ],
            }

            evidence_chunks: list[dict[str, Any]] = []
            steps_summary: list[dict[str, Any]] = []
            synth_result: QueryToolResult | None = None

            for round_index, step in enumerate(steps, start=1):
                yield {
                    "event": "round_started",
                    "session_id": session_id,
                    "round": round_index,
                    "step_index": step.step_index,
                    "title": step.title or step.query[:80],
                    "query": step.query,
                    "kb_ids": step.kb_ids,
                    "mode": step.mode,
                    "priority": step.priority,
                }
                summary = {
                    "round": round_index,
                    "step_index": step.step_index,
                    "title": step.title or step.query[:80],
                    "query": step.query,
                    "kb_ids": step.kb_ids,
                    "mode": step.mode,
                    "priority": step.priority,
                    "status": "ok",
                    "chunk_count": 0,
                    "per_kb_chunk_counts": {},
                    "skipped_kbs": [],
                }
                try:
                    tool_result, retried_mode = await self._retrieve_with_empty_retry(
                        http_request=request, body=body, step=step
                    )
                except Exception as exc:  # noqa: BLE001 — tolerate per-step failure
                    # One failed step must not discard evidence accumulated in
                    # earlier rounds; the gap is reported to the user instead.
                    if isinstance(exc, HTTPException):
                        detail = exc.detail
                        error_code = (
                            detail.get("error_code", "agent_step_failed")
                            if isinstance(detail, dict)
                            else "agent_step_failed"
                        )
                    else:
                        error_code = "agent_step_failed"
                        logger.error(
                            "Agent step %d failed: %s", step.step_index, exc,
                            exc_info=True,
                        )
                    summary["status"] = "failed"
                    summary["error_code"] = error_code
                    steps_summary.append(summary)
                    await append_enterprise_audit_event(
                        request,
                        "agent_retrieve_round",
                        target_type="agent_session",
                        target_id=session_id,
                        metadata={
                            "round": round_index,
                            "kb_ids": step.kb_ids,
                            "mode": step.mode,
                            "status": "failed",
                            "error_code": error_code,
                            "query_hash": hashlib.sha256(
                                step.query.encode("utf-8")
                            ).hexdigest(),
                        },
                    )
                    yield {
                        "event": "round_result",
                        "session_id": session_id,
                        **summary,
                    }
                    continue

                if synth_result is None:
                    synth_result = tool_result
                used_mode = retried_mode or step.mode
                if retried_mode:
                    summary["retried_mode"] = retried_mode
                if tool_result.alt_chunk_counts or tool_result.alt_failed_kbs:
                    summary["bilingual"] = True
                    summary["alt_chunk_counts"] = tool_result.alt_chunk_counts
                    if tool_result.alt_failed_kbs:
                        summary["alt_failed_kbs"] = tool_result.alt_failed_kbs
                evidence_chunks.extend(
                    {
                        **chunk,
                        "round_index": round_index,
                        "step_index": step.step_index,
                        "mode": used_mode,
                    }
                    for chunk in tool_result.chunks
                )
                summary.update(
                    {
                        "kb_ids": tool_result.queried_kb_ids,
                        "chunk_count": len(tool_result.chunks),
                        "per_kb_chunk_counts": tool_result.per_kb_chunk_counts,
                        "skipped_kbs": tool_result.skipped_kbs,
                    }
                )
                steps_summary.append(summary)
                audit_metadata = {
                    "round": round_index,
                    "kb_ids": step.kb_ids,
                    "mode": step.mode,
                    "status": "ok",
                    "query_hash": hashlib.sha256(
                        step.query.encode("utf-8")
                    ).hexdigest(),
                    "chunk_count": len(tool_result.chunks),
                    "skipped_kbs": tool_result.skipped_kbs,
                }
                if retried_mode:
                    audit_metadata["retried_mode"] = retried_mode
                await append_enterprise_audit_event(
                    request,
                    "agent_retrieve_round",
                    target_type="agent_session",
                    target_id=session_id,
                    metadata=audit_metadata,
                )
                yield {"event": "round_result", "session_id": session_id, **summary}

            failed_rounds = [s for s in steps_summary if s.get("status") == "failed"]
            if synth_result is None:
                raise HTTPException(
                    status_code=502,
                    detail={
                        "error_code": "agent_all_steps_failed",
                        "message": "All Agent retrieval steps failed",
                        "steps_summary": steps_summary,
                    },
                )

            processed = self._select_evidence(
                body=body,
                evidence_chunks=evidence_chunks,
                synth_result=synth_result,
            )
            references, context_units = self._build_references(
                processed, include_chunk_content=body.include_chunk_content
            )
            if references and body.include_references:
                yield {
                    "event": "references",
                    "session_id": session_id,
                    "references": references,
                }

            answer_parts: list[str] = []
            if not context_units:
                answer = "未检索到可用于回答的证据。"
                yield {"event": "response", "session_id": session_id, "delta": answer}
            else:
                async for delta in self._synthesize_answer(
                    request=request,
                    body=body,
                    synth_result=synth_result,
                    references=references,
                    context_units=context_units,
                    steps_summary=steps_summary,
                    stream=stream_synthesis,
                ):
                    if delta:
                        answer_parts.append(delta)
                        yield {
                            "event": "response",
                            "session_id": session_id,
                            "delta": delta,
                        }
                answer = "".join(answer_parts)

            await append_enterprise_audit_event(
                request,
                "agent_query_completed",
                target_type="agent_session",
                target_id=session_id,
                metadata={
                    "round_count": len(steps_summary),
                    "failed_round_count": len(failed_rounds),
                    "reference_count": len(references),
                    "effective_kb_ids": [record.id for record in effective_records],
                },
            )
            result = AgentRunResult(
                status="success",
                session_id=session_id,
                answer=answer,
                references=references if body.include_references else [],
                steps_summary=steps_summary,
                metadata={
                    "workflow": body.workflow,
                    "effective_kb_ids": [record.id for record in effective_records],
                    "round_count": len(steps_summary),
                    "failed_round_count": len(failed_rounds),
                    "plan_truncated": plan_truncated,
                    "bilingual_retrieval": agent_bilingual_enabled(body),
                    "notes_for_user": plan.notes_for_user,
                },
            )
            yield {"event": "done", "session_id": session_id, "_result": result}
        except Exception as exc:
            try:
                await append_enterprise_audit_event(
                    request,
                    "agent_session_failed",
                    target_type="agent_session",
                    target_id=session_id,
                    metadata={
                        "error": str(
                            exc.detail if isinstance(exc, HTTPException) else exc
                        )[:500],
                        "status_code": exc.status_code
                        if isinstance(exc, HTTPException)
                        else None,
                    },
                )
            except Exception as audit_exc:  # noqa: BLE001
                logger.warning("Agent session-failed audit failed: %s", audit_exc)
            raise

    def _require_agent_access(self, request: Request):
        if not agent_query_enabled():
            raise HTTPException(status_code=403, detail="Agent query is disabled")
        if not enterprise_auth_enabled():
            raise HTTPException(status_code=403, detail="Agent query requires enterprise auth")
        principal = get_request_principal(request)
        get_enterprise_authorization_service(request).require_agent_query(principal)
        return principal

    async def _effective_kbs(
        self, request: Request, candidate_kb_ids: list[str] | None
    ) -> list[KnowledgeBaseRecord]:
        principal = get_request_principal(request)
        all_records = [
            record
            for record in await self._kb_service.list(include_deleted=False)
            if record.status == "active"
        ]
        authorized = await get_enterprise_authorization_service(
            request
        ).filter_kbs_for_principal(principal, all_records)
        authorized_by_id = {record.id: record for record in authorized}
        if candidate_kb_ids:
            selected: list[str] = []
            for kb_id in candidate_kb_ids:
                if kb_id not in selected:
                    selected.append(kb_id)
            unauthorized = [kb_id for kb_id in selected if kb_id not in authorized_by_id]
            if unauthorized:
                raise HTTPException(
                    status_code=403,
                    detail="One or more candidate knowledge bases are not accessible",
                )
            records = [authorized_by_id[kb_id] for kb_id in selected]
        else:
            records = authorized
        if not records:
            raise HTTPException(status_code=403, detail="No accessible knowledge bases for Agent query")
        return records

    async def _agent_llm_context(
        self,
        request: Request,
        body: AgentQueryRequest,
        effective_records: list[KnowledgeBaseRecord],
    ) -> tuple[Any, str]:
        """Resolve the AGENT role LLM plus the merged user workflow prompt.

        Shared by the plan-mode planner and the staged workflow so both use
        identical role resolution and prompt-merging semantics.
        """
        rag = await self._query_tool_service.get_rag(effective_records[0].id)
        global_config = rag._build_global_config()
        agent_func = global_config["role_llm_funcs"].get("agent")
        if agent_func is None:
            raise HTTPException(status_code=500, detail="AGENT role LLM is unavailable")
        if len(effective_records) > _PLANNING_KB_WARN_THRESHOLD:
            logger.warning(
                "Agent planning payload contains %d KB profiles; consider "
                "candidate_kb_ids to narrow the planning context",
                len(effective_records),
            )
        user_prompt = ""
        principal = get_request_principal(request)
        if principal is not None and principal.auth_method == "jwt":
            user_prompt = await get_enterprise_user_agent_workflow_prompt_service(
                request
            ).get_prompt(principal.user_id)
        if body.user_prompt:
            user_prompt = "\n\n".join(
                part for part in [user_prompt, body.user_prompt] if part
            )
        return agent_func, user_prompt

    async def _retrieve_with_empty_retry(
        self,
        *,
        http_request: Request,
        body: AgentQueryRequest,
        step: AgentPlanStep,
    ) -> tuple[QueryToolResult, str | None]:
        """Execute one retrieval step, retrying once on an empty result.

        The retry is best-effort: a fallback-mode failure keeps the original
        empty result instead of failing the step. Returns the result plus the
        fallback mode when the retry produced the returned chunks.
        """
        tool_result = await self._retrieve_for_step(
            http_request=http_request, body=body, step=step, mode=step.mode
        )
        if tool_result.chunks:
            return tool_result, None
        fallback = EMPTY_RETRY_MODE_FALLBACK.get(step.mode)
        if not fallback:
            return tool_result, None
        try:
            retry_result = await self._retrieve_for_step(
                http_request=http_request, body=body, step=step, mode=fallback
            )
        except Exception as exc:  # noqa: BLE001 — retry must not fail the step
            logger.warning(
                "Agent empty-result retry (mode=%s) failed: %s", fallback, exc
            )
            return tool_result, None
        if retry_result.chunks:
            return retry_result, fallback
        return tool_result, None

    async def _retrieve_for_step(
        self,
        *,
        http_request: Request,
        body: AgentQueryRequest,
        step: AgentPlanStep,
        mode: str,
    ) -> QueryToolResult:
        bilingual = agent_bilingual_enabled(body)
        return await self._query_tool_service.retrieve_serial(
            http_request=http_request,
            kb_ids=step.kb_ids,
            query=step.query,
            mode=cast(QueryMode, mode),
            filters=body.filters,
            top_k=body.top_k,
            chunk_top_k=body.chunk_top_k,
            max_entity_tokens=body.max_entity_tokens,
            max_relation_tokens=body.max_relation_tokens,
            max_total_tokens=body.max_total_tokens,
            enable_rerank=body.enable_rerank,
            hl_keywords=step.hl_keywords,
            ll_keywords=step.ll_keywords,
            query_alt=step.query_alt if bilingual else None,
            hl_keywords_alt=step.hl_keywords_alt,
            ll_keywords_alt=step.ll_keywords_alt,
        )

    async def _plan(
        self,
        *,
        request: Request,
        body: AgentQueryRequest,
        effective_records: list[KnowledgeBaseRecord],
        max_rounds: int,
    ) -> AgentPlan:
        agent_func, user_prompt = await self._agent_llm_context(
            request, body, effective_records
        )
        bilingual = agent_bilingual_enabled(body)

        step_schema: dict[str, Any] = {
            "step_index": 1,
            "title": "短标题",
            "query": "完整自洽的检索子问题",
            "kb_ids": [effective_records[0].id],
            "mode": "mix",
            "priority": "P0",
            "hl_keywords": [],
            "ll_keywords": [],
        }
        if bilingual:
            step_schema.update(
                {
                    "query_alt": "该步子问题的另一语言完整版本",
                    "hl_keywords_alt": [],
                    "ll_keywords_alt": [],
                }
            )

        payload = {
            "user_question": body.query,
            "allowed_kbs": [
                agent_kb_profile(record)
                for record in effective_records
            ],
            "max_rounds": max_rounds,
            "bilingual_retrieval": bilingual,
            "default_retrieve_params": {
                "top_k": body.top_k,
                "chunk_top_k": body.chunk_top_k,
                "enable_rerank": body.enable_rerank,
            },
            "user_workflow_prompt": user_prompt,
            "output_schema": {
                "type": "plan",
                "clarification_required": False,
                "clarification_question": None,
                "steps": [step_schema],
                "notes_for_user": "可选一句话",
            },
        }
        prompt = json.dumps(payload, ensure_ascii=False)
        system_prompt = DEFAULT_AGENT_WORKFLOW_PROMPT
        if bilingual:
            system_prompt = f"{system_prompt}\n{BILINGUAL_PLAN_PROMPT_SUFFIX}"

        saw_empty_plan = False

        def _parse_plan(data: Any) -> AgentPlan:
            nonlocal saw_empty_plan
            plan = AgentPlan.model_validate(data)
            if not plan.clarification_required and not plan.steps:
                saw_empty_plan = True
                raise ValueError("plan contains no steps and no clarification")
            return plan

        try:
            return await call_llm_json(
                agent_func,
                prompt,
                system_prompt=system_prompt,
                priority=DEFAULT_QUERY_PRIORITY,
                parse=_parse_plan,
                attempts=AGENT_PLAN_LLM_ATTEMPTS,
                label="agent_plan",
                response_format=_agent_plan_response_format(
                    effective_records, bilingual=bilingual
                ),
            )
        except LLMJsonError as exc:
            if saw_empty_plan:
                # All attempts failed and at least one returned a syntactically
                # valid plan with no steps — a degenerate-but-legal output some
                # JSON-constrained models produce. Degrade to one mix-mode
                # retrieval over the candidate KBs instead of failing the
                # session with 502.
                logger.warning(
                    "Agent planner produced no usable plan after %d attempts; "
                    "using single-step fallback retrieval: %s",
                    AGENT_PLAN_LLM_ATTEMPTS,
                    exc,
                )
                return AgentPlan(
                    type="plan",
                    clarification_required=False,
                    steps=[
                        AgentPlanStep(
                            step_index=1,
                            title="直接检索",
                            query=body.query,
                            kb_ids=[record.id for record in effective_records],
                            mode="mix",
                            priority="P1",
                        )
                    ],
                    notes_for_user=(
                        "规划器未能生成有效检索计划，已改为对候选知识库直接执行单步混合检索。"
                    ),
                )
            raise HTTPException(
                status_code=502,
                detail={"error_code": "agent_plan_invalid", "message": str(exc)},
            ) from exc

    def _validate_plan(
        self,
        plan: AgentPlan,
        effective_records: list[KnowledgeBaseRecord],
        max_rounds: int,
    ) -> tuple[list[AgentPlanStep], bool]:
        allowed_ids = {record.id for record in effective_records}
        # Stable order: P0 before P1/P2, plan order within the same priority.
        # Sub-queries are self-contained by contract, so reordering is safe and
        # guarantees P0 steps survive the max_rounds truncation below.
        ordered = sorted(
            plan.steps,
            key=lambda step: (_PRIORITY_RANK.get(step.priority, 1), step.step_index),
        )
        truncated = len(ordered) > max_rounds
        steps = ordered[:max_rounds]
        if truncated:
            logger.warning(
                "Agent plan truncated from %d to %d steps (max_rounds)",
                len(ordered),
                max_rounds,
            )
        if not steps:
            raise HTTPException(status_code=400, detail="Agent plan contains no executable steps")
        self._ensure_steps_allowed(steps, allowed_ids)
        return steps, truncated

    @staticmethod
    def _ensure_steps_allowed(
        steps: list[AgentPlanStep], allowed_ids: set[str]
    ) -> None:
        """Fail closed on any step using an unsupported mode or an
        inaccessible KB; shared by plan-mode and staged validation."""
        for step in steps:
            if step.mode not in AGENT_ALLOWED_MODES:
                raise HTTPException(status_code=400, detail="Agent plan contains unsupported mode")
            if any(kb_id not in allowed_ids for kb_id in step.kb_ids):
                raise HTTPException(status_code=403, detail="Agent plan selected an inaccessible KB")

    def _select_evidence(
        self,
        *,
        body: AgentQueryRequest,
        evidence_chunks: list[dict[str, Any]],
        synth_result: QueryToolResult,
    ) -> list[dict[str, Any]]:
        """Dedup, interleave across rounds, and token-truncate the evidence.

        No second rerank happens here: chunks were already reranked against
        their own sub-query during retrieval, and re-scoring them against the
        umbrella question systematically drops evidence for sub-questions
        phrased differently (e.g. P0 regulation lookups).
        """
        deduped = _dedup_agent_chunks(evidence_chunks)
        interleaved = _interleave_rounds(deduped)
        rag = synth_result.rag
        param = synth_result.param
        tokenizer = None
        if rag is not None:
            tokenizer = rag._build_global_config().get("tokenizer")
        budget = body.max_total_tokens or (
            getattr(param, "max_total_tokens", None) if param is not None else None
        )
        if tokenizer is not None and budget:
            interleaved = truncate_list_by_token_size(
                interleaved,
                key=lambda chunk: str(chunk.get("content", "")),
                max_token_size=int(budget),
                tokenizer=tokenizer,
            )
        return interleaved

    def _build_references(
        self,
        processed: list[dict[str, Any]],
        *,
        include_chunk_content: bool,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        references: list[dict[str, Any]] = []
        context_units: list[dict[str, str]] = []
        for index, chunk in enumerate(processed, start=1):
            reference_id = f"A{index}"
            content = str(chunk.get("content", ""))
            file_path = str(chunk.get("file_path") or chunk.get("source") or "unknown")
            ref = {
                "reference_id": reference_id,
                "kb_id": str(chunk.get("kb_id", "")),
                "round": int(chunk.get("round_index") or 0),
                "step_index": int(chunk.get("step_index") or 0),
                "mode": str(chunk.get("mode", "")),
                "file_path": file_path,
                "chunk_id": chunk.get("chunk_id"),
                "source_reference_id": chunk.get("reference_id"),
            }
            if include_chunk_content:
                ref["content"] = [content]
            references.append(ref)
            context_units.append({"reference_id": reference_id, "content": content})
        return references, context_units

    async def _synthesize_answer(
        self,
        *,
        request: Request,
        body: AgentQueryRequest,
        synth_result: QueryToolResult,
        references: list[dict[str, Any]],
        context_units: list[dict[str, str]],
        steps_summary: list[dict[str, Any]],
        stream: bool,
        extra_rules: str = "",
    ) -> AsyncIterator[str]:
        global_config = synth_result.rag._build_global_config()
        param = synth_result.param
        if body.response_type:
            param.response_type = body.response_type
        if body.max_total_tokens:
            param.max_total_tokens = body.max_total_tokens
        reference_list_str = "\n".join(
            f"[{ref['reference_id']}] {ref['file_path']} (kb_id={ref['kb_id']}, round={ref['round']})"
            for ref in references
        )
        chunks_str = "\n".join(json.dumps(unit, ensure_ascii=False) for unit in context_units)
        content_data = PROMPTS["naive_query_context"].format(
            text_chunks_str=chunks_str,
            reference_list_str=reference_list_str,
        )
        user_prompt = body.user_prompt or "n/a"
        # Server-side project memory: prepend distilled facts to the answer
        # prompt (fail-open — an unavailable backend just skips injection).
        memory_block, _memory_info = await resolve_memory_injection(
            request, body.memory, body.query
        )
        if memory_block:
            user_prompt = (
                memory_block if user_prompt == "n/a" else f"{memory_block}\n\n{user_prompt}"
            )
        agent_rules = (
            "你只能基于给定证据回答。引用必须使用 [A1]、[A2] 这样的 Agent 级引用编号。"
            "如果证据不足或存在冲突，必须明确说明缺口或冲突；不要编造未在证据中的事实。"
        )
        if extra_rules:
            agent_rules = f"{agent_rules}\n{extra_rules}"
        gap_notes = self._evidence_gap_notes(steps_summary)
        if gap_notes:
            agent_rules = (
                f"{agent_rules}\n已知检索缺口（必须在回答中明确说明对应内容未覆盖）：{gap_notes}"
            )
        sys_prompt = PROMPTS["naive_rag_response"].format(
            response_type=param.response_type or "Multiple Paragraphs",
            user_prompt=f"{agent_rules}\n\n{user_prompt}",
            content_data=content_data,
        )
        query_func = partial(
            global_config["role_llm_funcs"]["query"], _priority=DEFAULT_QUERY_PRIORITY
        )
        response = await query_func(
            body.query,
            system_prompt=sys_prompt,
            history_messages=body.conversation_history or [],
            stream=stream,
            enable_cot=True,
        )
        if hasattr(response, "__aiter__"):
            async for chunk in response:
                if chunk:
                    yield str(chunk)
        else:
            yield str(response)

    @staticmethod
    def _evidence_gap_notes(steps_summary: list[dict[str, Any]]) -> str:
        notes: list[str] = []
        for summary in steps_summary:
            if summary.get("status") == "failed":
                notes.append(
                    f"第{summary['round']}步“{summary['title']}”检索失败"
                )
                continue
            for skipped in summary.get("skipped_kbs") or []:
                notes.append(
                    f"第{summary['round']}步知识库 {skipped.get('kb_id')} 不可用"
                    f"（{skipped.get('reason')}）"
                )
        return "；".join(notes)
