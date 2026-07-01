from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import partial
from typing import Any, AsyncIterator, Literal, cast
from uuid import uuid4

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field, field_validator

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
from lightrag.api.query_tool_service import (
    KBQueryFilters,
    QueryMode,
    QueryToolResult,
    QueryToolService,
)
from lightrag.constants import DEFAULT_QUERY_PRIORITY
from lightrag.prompt import PROMPTS
from lightrag.utils import logger, process_chunks_unified

AGENT_ALLOWED_MODES: set[str] = {"local", "global", "hybrid", "naive", "mix"}

DEFAULT_AGENT_WORKFLOW_PROMPT = """
你是 LightRAG Agent 编排器。你只能在服务端提供的 allowed_kbs 中选择 kb_ids，
并为每个检索步骤指定底层检索模式 local/global/hybrid/naive/mix 之一。禁止使用 bypass。
你不直接回答最终问题，只输出严格 JSON。子 query 必须完整自洽，不要依赖“上文”。
法规、禁忌、合规类子问题优先标记 P0。若用户问题缺少关键约束且无法规划检索，输出澄清。
不要输出 markdown，不要输出 chain-of-thought。
""".strip()


class AgentPlanStep(BaseModel):
    step_index: int = Field(ge=1)
    title: str = Field(default="", max_length=200)
    query: str = Field(min_length=3, max_length=4096)
    kb_ids: list[str] = Field(min_length=1)
    mode: Literal["local", "global", "hybrid", "naive", "mix"]
    priority: Literal["P0", "P1", "P2"] = "P1"
    hl_keywords: list[str] = Field(default_factory=list)
    ll_keywords: list[str] = Field(default_factory=list)

    @field_validator("query", "title", mode="after")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip()


class AgentPlan(BaseModel):
    type: Literal["plan"] = "plan"
    clarification_required: bool = False
    clarification_question: str | None = None
    steps: list[AgentPlanStep] = Field(default_factory=list)
    notes_for_user: str | None = None


class AgentQueryRequest(BaseModel):
    query: str = Field(min_length=3)
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

    @field_validator("query", mode="after")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        return value.strip()


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


def _strip_code_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    return value


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


def _coerce_agent_tags(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _coerce_agent_priority(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def agent_kb_profile(record: KnowledgeBaseRecord) -> dict[str, Any]:
    metadata = record.metadata or {}
    agent_description = str(metadata.get("agent_description") or "").strip()
    return {
        "kb_id": record.id,
        "name": record.name,
        "description": record.description or "",
        "agent_description": agent_description,
        "agent_tags": _coerce_agent_tags(metadata.get("agent_tags")),
        "agent_priority": _coerce_agent_priority(metadata.get("agent_priority")),
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
                "effective_kb_count": len(effective_records),
                "query_hash": hashlib.sha256(body.query.encode("utf-8")).hexdigest(),
                "max_rounds": max_rounds,
            },
        )

        plan = await self._plan(
            request=request,
            body=body,
            effective_records=effective_records,
            max_rounds=max_rounds,
        )
        if plan.clarification_required:
            return AgentRunResult(
                status="clarification_required",
                session_id=session_id,
                clarification_question=plan.clarification_question or "请补充关键约束。",
                metadata={"effective_kb_ids": [record.id for record in effective_records]},
            )

        steps = self._validate_plan(plan, effective_records, max_rounds)
        evidence_chunks: list[dict[str, Any]] = []
        steps_summary: list[dict[str, Any]] = []
        synth_result: QueryToolResult | None = None

        for round_index, step in enumerate(steps, start=1):
            tool_result = await self._query_tool_service.retrieve_serial(
                http_request=request,
                kb_ids=step.kb_ids,
                query=step.query,
                mode=cast(QueryMode, step.mode),
                filters=body.filters,
                top_k=body.top_k,
                chunk_top_k=body.chunk_top_k,
                max_entity_tokens=body.max_entity_tokens,
                max_relation_tokens=body.max_relation_tokens,
                max_total_tokens=body.max_total_tokens,
                enable_rerank=body.enable_rerank,
                hl_keywords=step.hl_keywords,
                ll_keywords=step.ll_keywords,
            )
            if synth_result is None:
                synth_result = tool_result
            evidence_chunks.extend(
                {**chunk, "round_index": round_index, "step_index": step.step_index, "mode": step.mode}
                for chunk in tool_result.chunks
            )
            summary = {
                "round": round_index,
                "step_index": step.step_index,
                "title": step.title or step.query[:80],
                "query": step.query,
                "kb_ids": tool_result.queried_kb_ids,
                "mode": step.mode,
                "priority": step.priority,
                "chunk_count": len(tool_result.chunks),
                "per_kb_chunk_counts": tool_result.per_kb_chunk_counts,
                "skipped_kbs": tool_result.skipped_kbs,
            }
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
                    "query_hash": hashlib.sha256(step.query.encode("utf-8")).hexdigest(),
                    "chunk_count": len(tool_result.chunks),
                    "skipped_kbs": tool_result.skipped_kbs,
                },
            )

        answer, references = await self._synthesize(
            body=body,
            evidence_chunks=evidence_chunks,
            synth_result=synth_result,
            stream=stream,
        )
        await append_enterprise_audit_event(
            request,
            "agent_query_completed",
            target_type="agent_session",
            target_id=session_id,
            metadata={
                "round_count": len(steps_summary),
                "reference_count": len(references),
                "effective_kb_ids": [record.id for record in effective_records],
            },
        )
        return AgentRunResult(
            status="success",
            session_id=session_id,
            answer=answer,
            references=references if body.include_references else [],
            steps_summary=steps_summary,
            metadata={
                "effective_kb_ids": [record.id for record in effective_records],
                "round_count": len(steps_summary),
            },
        )

    async def stream_events(
        self,
        *,
        request: Request,
        body: AgentQueryRequest,
    ) -> AsyncIterator[str]:
        # The first implementation preserves deterministic serial behavior and
        # emits stable Agent envelopes while reusing the non-streaming synthesis
        # path. Streaming token deltas can be added without changing event names.
        try:
            result = await self.run(request=request, body=body, stream=False)
            yield _json_event(
                {
                    "event": "session_started",
                    "session_id": result.session_id,
                    "metadata": result.metadata,
                }
            )
            if result.status == "clarification_required":
                yield _json_event(
                    {
                        "event": "clarification_required",
                        "session_id": result.session_id,
                        "clarification_question": result.clarification_question,
                    }
                )
                yield _json_event({"event": "done", "session_id": result.session_id})
                return
            yield _json_event(
                {
                    "event": "plan_created",
                    "session_id": result.session_id,
                    "steps": result.steps_summary,
                }
            )
            for step in result.steps_summary:
                yield _json_event(
                    {"event": "round_result", "session_id": result.session_id, **step}
                )
            if result.references:
                yield _json_event(
                    {
                        "event": "references",
                        "session_id": result.session_id,
                        "references": result.references,
                    }
                )
            yield _json_event(
                {
                    "event": "response",
                    "session_id": result.session_id,
                    "delta": result.answer,
                }
            )
            yield _json_event({"event": "done", "session_id": result.session_id})
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

    async def _plan(
        self,
        *,
        request: Request,
        body: AgentQueryRequest,
        effective_records: list[KnowledgeBaseRecord],
        max_rounds: int,
    ) -> AgentPlan:
        rag = await self._query_tool_service.get_rag(effective_records[0].id)
        global_config = rag._build_global_config()
        agent_func = global_config["role_llm_funcs"].get("agent")
        if agent_func is None:
            raise HTTPException(status_code=500, detail="AGENT role LLM is unavailable")
        user_prompt = ""
        principal = get_request_principal(request)
        if principal is not None and principal.auth_method == "jwt":
            user_prompt = await get_enterprise_user_agent_workflow_prompt_service(
                request
            ).get_prompt(principal.user_id)
        if body.user_prompt:
            user_prompt = "\n\n".join(part for part in [user_prompt, body.user_prompt] if part)

        payload = {
            "user_question": body.query,
            "allowed_kbs": [
                agent_kb_profile(record)
                for record in effective_records
            ],
            "max_rounds": max_rounds,
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
                "steps": [
                    {
                        "step_index": 1,
                        "title": "短标题",
                        "query": "完整自洽的检索子问题",
                        "kb_ids": [effective_records[0].id],
                        "mode": "mix",
                        "priority": "P0",
                        "hl_keywords": [],
                        "ll_keywords": [],
                    }
                ],
                "notes_for_user": "可选一句话",
            },
        }
        prompt = json.dumps(payload, ensure_ascii=False)
        try:
            raw = await partial(agent_func, _priority=DEFAULT_QUERY_PRIORITY)(
                prompt,
                system_prompt=DEFAULT_AGENT_WORKFLOW_PROMPT,
                stream=False,
                enable_cot=False,
                response_format={"type": "json_object"},
            )
        except TypeError:
            raw = await partial(agent_func, _priority=DEFAULT_QUERY_PRIORITY)(
                prompt,
                system_prompt=DEFAULT_AGENT_WORKFLOW_PROMPT,
                stream=False,
                enable_cot=False,
            )
        try:
            parsed = raw if isinstance(raw, dict) else json.loads(_strip_code_fence(str(raw)))
            return AgentPlan.model_validate(parsed)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=502,
                detail={"error_code": "agent_plan_invalid", "message": str(exc)},
            ) from exc

    def _validate_plan(
        self,
        plan: AgentPlan,
        effective_records: list[KnowledgeBaseRecord],
        max_rounds: int,
    ) -> list[AgentPlanStep]:
        allowed_ids = {record.id for record in effective_records}
        steps = sorted(plan.steps, key=lambda step: step.step_index)[:max_rounds]
        if not steps:
            raise HTTPException(status_code=400, detail="Agent plan contains no executable steps")
        for step in steps:
            if step.mode not in AGENT_ALLOWED_MODES:
                raise HTTPException(status_code=400, detail="Agent plan contains unsupported mode")
            if any(kb_id not in allowed_ids for kb_id in step.kb_ids):
                raise HTTPException(status_code=403, detail="Agent plan selected an inaccessible KB")
        return steps

    async def _synthesize(
        self,
        *,
        body: AgentQueryRequest,
        evidence_chunks: list[dict[str, Any]],
        synth_result: QueryToolResult | None,
        stream: bool,
    ) -> tuple[str, list[dict[str, Any]]]:
        if synth_result is None or synth_result.rag is None or synth_result.param is None:
            return "未检索到可用于回答的证据。", []
        global_config = synth_result.rag._build_global_config()
        param = synth_result.param
        if body.response_type:
            param.response_type = body.response_type
        if body.max_total_tokens:
            param.max_total_tokens = body.max_total_tokens
        processed = await process_chunks_unified(
            query=body.query,
            unique_chunks=_dedup_agent_chunks(evidence_chunks),
            query_param=param,
            global_config=global_config,
            source_type="agent",
            chunk_token_limit=None,
        )
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
            if body.include_chunk_content:
                ref["content"] = [content]
            references.append(ref)
            context_units.append({"reference_id": reference_id, "content": content})

        if not context_units:
            return "未检索到可用于回答的证据。", []
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
        agent_rules = (
            "你只能基于给定证据回答。引用必须使用 [A1]、[A2] 这样的 Agent 级引用编号。"
            "如果证据不足或存在冲突，必须明确说明缺口或冲突；不要编造未在证据中的事实。"
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
            stream=stream,
            enable_cot=True,
        )
        if hasattr(response, "__aiter__"):
            parts: list[str] = []
            async for chunk in response:
                if chunk:
                    parts.append(str(chunk))
            answer = "".join(parts)
        else:
            answer = str(response)
        return answer, references
