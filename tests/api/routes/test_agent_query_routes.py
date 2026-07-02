from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from lightrag.api.agent_query_service import AgentQueryRequest, AgentQueryService
from lightrag.api.enterprise_auth import (
    AuthorizationService,
    Principal,
    SYSTEM_ROLE_SUPER_ADMIN,
    USER_STATUS_ACTIVE,
)
from lightrag.api.kb_service import KnowledgeBaseRecord, utc_now_iso
from lightrag.api.query_tool_service import QueryToolResult
from lightrag.base import QueryParam


pytestmark = pytest.mark.offline


class _KBService:
    def __init__(self):
        now = utc_now_iso()
        self.records = [
            KnowledgeBaseRecord(
                id="kb1",
                name="法规库",
                description="regulations",
                workspace="kb_kb1",
                status="active",
                active_config_version_id=None,
                owner_id=None,
                tenant_id=None,
                visibility="private",
                created_at=now,
                updated_at=now,
                metadata={
                    "agent_description": "优先用于法规、合规、禁忌和限量判断",
                    "agent_tags": ["法规", "合规", "限量"],
                    "agent_priority": 10,
                    "agent_auto_profile": {
                        "status": "ready",
                        "description": "自动生成的法规路由说明",
                        "tags": ["自动法规"],
                        "domains": ["食品合规"],
                        "sample_questions": ["某原料是否允许使用？"],
                        "negative_scope": ["配方工艺细节"],
                    },
                },
            ),
            KnowledgeBaseRecord(
                id="kb2",
                name="配方库",
                description="formulas",
                workspace="kb_kb2",
                status="active",
                active_config_version_id=None,
                owner_id=None,
                tenant_id=None,
                visibility="private",
                created_at=now,
                updated_at=now,
                metadata={
                    "agent_description": "优先用于配方、工艺和原料应用",
                    "agent_tags": "配方, 工艺",
                    "agent_priority": 5,
                },
            ),
        ]

    async def list(self, *, include_deleted=False):
        return self.records


class _FakeRAG:
    def __init__(self, agent_plan, *, raw_plan_responses=None, answer_deltas=None):
        self.agent_plan = agent_plan
        self.raw_plan_responses = list(raw_plan_responses or [])
        self.answer_deltas = answer_deltas
        self.seen_plan_payload = None
        self.plan_calls = 0

    def _build_global_config(self):
        async def agent_func(prompt, **_kwargs):
            self.plan_calls += 1
            self.seen_plan_payload = json.loads(prompt)
            if self.raw_plan_responses:
                return self.raw_plan_responses.pop(0)
            return json.dumps(self.agent_plan, ensure_ascii=False)

        async def query_func(_query, stream=False, **_kwargs):
            if stream and self.answer_deltas is not None:
                async def _gen():
                    for delta in self.answer_deltas:
                        yield delta

                return _gen()
            return "最终答案 [A1]"

        return {
            "role_llm_funcs": {"agent": agent_func, "query": query_func},
            "tokenizer": None,
        }


class _QueryTool:
    def __init__(self, rag, fail_calls: set[int] | None = None):
        self.rag = rag
        self.calls = []
        self._fail_calls = fail_calls or set()

    async def get_rag(self, _kb_id):
        return self.rag

    async def retrieve_serial(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) in self._fail_calls:
            raise HTTPException(
                status_code=502,
                detail={"error_code": "kb_retrieve_failed"},
            )
        return QueryToolResult(
            chunks=[
                {
                    "kb_id": kwargs["kb_ids"][0],
                    "chunk_id": f"chunk-{len(self.calls)}",
                    "file_path": "doc.md",
                    "content": "证据内容",
                }
            ],
            rag=self.rag,
            param=QueryParam(mode=kwargs["mode"]),
            queried_kb_ids=kwargs["kb_ids"],
            per_kb_chunk_counts={kwargs["kb_ids"][0]: 1},
        )


def _request(monkeypatch):
    monkeypatch.setattr("lightrag.api.agent_query_service.enterprise_auth_enabled", lambda: True)
    monkeypatch.setattr("lightrag.api.agent_query_service.agent_query_enabled", lambda: True)
    principal = Principal(
        user_id="admin",
        username="admin",
        system_role=SYSTEM_ROLE_SUPER_ADMIN,
        status=USER_STATUS_ACTIVE,
        tenant_id=None,
        tenant_roles={},
        can_create_kb=True,
        can_use_bypass_query=True,
        token_version=1,
        auth_method="api_key",
        metadata={},
        can_use_agent_query=True,
    )
    state = SimpleNamespace(
        principal=principal,
        enterprise_authorization_service=AuthorizationService(metadata_store=SimpleNamespace()),
    )
    return SimpleNamespace(state=state, app=SimpleNamespace(state=state))


def _audit_recorder(monkeypatch) -> list[str]:
    events: list[str] = []

    async def record(_request, event, **_kwargs):
        events.append(event)

    monkeypatch.setattr(
        "lightrag.api.agent_query_service.append_enterprise_audit_event", record
    )
    return events


def _single_step_plan(kb_id: str = "kb1") -> dict:
    return {
        "type": "plan",
        "clarification_required": False,
        "steps": [
            {
                "step_index": 1,
                "title": "查法规",
                "query": "检索法规限制",
                "kb_ids": [kb_id],
                "mode": "mix",
                "priority": "P0",
            }
        ],
    }


@pytest.mark.asyncio
async def test_agent_query_uses_candidate_kb_subset_and_agent_references(monkeypatch):
    _audit_recorder(monkeypatch)
    rag = _FakeRAG(_single_step_plan())
    service = AgentQueryService(kb_service=_KBService(), query_tool_service=_QueryTool(rag))

    result = await service.run(
        request=_request(monkeypatch),
        body=AgentQueryRequest(query="如何合规推荐配方？", candidate_kb_ids=["kb1"]),
    )

    assert result.status == "success"
    assert rag.seen_plan_payload["allowed_kbs"] == [
        {
            "kb_id": "kb1",
            "name": "法规库",
            "description": "regulations",
            "agent_description": "优先用于法规、合规、禁忌和限量判断",
            "agent_tags": ["法规", "合规", "限量"],
            "agent_priority": 10,
            "agent_auto_profile_status": "ready",
            "agent_profile_updated_at": None,
            "agent_profile_domains": ["食品合规"],
            "agent_profile_sample_questions": ["某原料是否允许使用？"],
            "agent_profile_negative_scope": ["配方工艺细节"],
        }
    ]
    assert result.references[0]["reference_id"] == "A1"
    assert result.references[0]["kb_id"] == "kb1"
    assert result.steps_summary[0]["mode"] == "mix"
    assert result.steps_summary[0]["status"] == "ok"


@pytest.mark.asyncio
async def test_agent_plan_selecting_kb_outside_effective_set_is_rejected(monkeypatch):
    events = _audit_recorder(monkeypatch)
    plan = _single_step_plan(kb_id="kb2")
    service = AgentQueryService(
        kb_service=_KBService(),
        query_tool_service=_QueryTool(_FakeRAG(plan)),
    )

    with pytest.raises(HTTPException) as exc:
        await service.run(
            request=_request(monkeypatch),
            body=AgentQueryRequest(query="只允许 kb1", candidate_kb_ids=["kb1"]),
        )

    assert exc.value.status_code == 403
    assert "agent_session_failed" in events


@pytest.mark.asyncio
async def test_plan_retry_recovers_from_invalid_json(monkeypatch):
    _audit_recorder(monkeypatch)
    rag = _FakeRAG(
        _single_step_plan(),
        raw_plan_responses=["这不是 JSON"],
    )
    service = AgentQueryService(kb_service=_KBService(), query_tool_service=_QueryTool(rag))

    result = await service.run(
        request=_request(monkeypatch),
        body=AgentQueryRequest(query="如何合规推荐配方？", candidate_kb_ids=["kb1"]),
    )

    assert result.status == "success"
    assert rag.plan_calls == 2


@pytest.mark.asyncio
async def test_plan_failure_after_retries_raises_502_and_audits(monkeypatch):
    events = _audit_recorder(monkeypatch)
    rag = _FakeRAG(
        _single_step_plan(),
        raw_plan_responses=["坏输出 1", "坏输出 2", "坏输出 3"],
    )
    service = AgentQueryService(kb_service=_KBService(), query_tool_service=_QueryTool(rag))

    with pytest.raises(HTTPException) as exc:
        await service.run(
            request=_request(monkeypatch),
            body=AgentQueryRequest(query="如何合规推荐配方？", candidate_kb_ids=["kb1"]),
        )

    assert exc.value.status_code == 502
    assert exc.value.detail["error_code"] == "agent_plan_invalid"
    assert rag.plan_calls == 3
    assert "agent_session_failed" in events


@pytest.mark.asyncio
async def test_plan_truncation_keeps_p0_steps(monkeypatch):
    _audit_recorder(monkeypatch)
    plan = {
        "type": "plan",
        "clarification_required": False,
        "steps": [
            {
                "step_index": 1,
                "title": "查配方",
                "query": "检索配方建议",
                "kb_ids": ["kb1"],
                "mode": "mix",
                "priority": "P1",
            },
            {
                "step_index": 2,
                "title": "查法规",
                "query": "检索法规限制",
                "kb_ids": ["kb1"],
                "mode": "hybrid",
                "priority": "P0",
            },
        ],
    }
    tool = _QueryTool(_FakeRAG(plan))
    service = AgentQueryService(kb_service=_KBService(), query_tool_service=tool)

    result = await service.run(
        request=_request(monkeypatch),
        body=AgentQueryRequest(
            query="如何合规推荐配方？", candidate_kb_ids=["kb1"], max_rounds=1
        ),
    )

    assert result.metadata["plan_truncated"] is True
    assert len(result.steps_summary) == 1
    assert result.steps_summary[0]["priority"] == "P0"
    assert result.steps_summary[0]["step_index"] == 2


@pytest.mark.asyncio
async def test_step_failure_is_tolerated_and_reported(monkeypatch):
    _audit_recorder(monkeypatch)
    plan = {
        "type": "plan",
        "clarification_required": False,
        "steps": [
            {
                "step_index": 1,
                "title": "查法规",
                "query": "检索法规限制",
                "kb_ids": ["kb1"],
                "mode": "mix",
                "priority": "P1",
            },
            {
                "step_index": 2,
                "title": "查配方",
                "query": "检索配方建议",
                "kb_ids": ["kb1"],
                "mode": "mix",
                "priority": "P1",
            },
        ],
    }
    tool = _QueryTool(_FakeRAG(plan), fail_calls={1})
    service = AgentQueryService(kb_service=_KBService(), query_tool_service=tool)

    result = await service.run(
        request=_request(monkeypatch),
        body=AgentQueryRequest(query="如何合规推荐配方？", candidate_kb_ids=["kb1"]),
    )

    assert result.status == "success"
    assert result.steps_summary[0]["status"] == "failed"
    assert result.steps_summary[0]["error_code"] == "kb_retrieve_failed"
    assert result.steps_summary[1]["status"] == "ok"
    assert result.metadata["failed_round_count"] == 1
    assert result.references  # evidence from the surviving step


@pytest.mark.asyncio
async def test_all_steps_failed_raises_502(monkeypatch):
    events = _audit_recorder(monkeypatch)
    tool = _QueryTool(_FakeRAG(_single_step_plan()), fail_calls={1})
    service = AgentQueryService(kb_service=_KBService(), query_tool_service=tool)

    with pytest.raises(HTTPException) as exc:
        await service.run(
            request=_request(monkeypatch),
            body=AgentQueryRequest(query="如何合规推荐配方？", candidate_kb_ids=["kb1"]),
        )

    assert exc.value.status_code == 502
    assert exc.value.detail["error_code"] == "agent_all_steps_failed"
    assert "agent_session_failed" in events


@pytest.mark.asyncio
async def test_clarification_plan_short_circuits_without_retrieval(monkeypatch):
    _audit_recorder(monkeypatch)
    plan = {
        "type": "plan",
        "clarification_required": True,
        "clarification_question": "请补充目标场景。",
        "steps": [],
    }
    tool = _QueryTool(_FakeRAG(plan))
    service = AgentQueryService(kb_service=_KBService(), query_tool_service=tool)

    result = await service.run(
        request=_request(monkeypatch),
        body=AgentQueryRequest(query="如何合规推荐配方？", candidate_kb_ids=["kb1"]),
    )

    assert result.status == "clarification_required"
    assert result.clarification_question == "请补充目标场景。"
    assert tool.calls == []


@pytest.mark.asyncio
async def test_stream_events_emit_progressively_with_answer_deltas(monkeypatch):
    _audit_recorder(monkeypatch)
    rag = _FakeRAG(_single_step_plan(), answer_deltas=["部分1", "部分2"])
    service = AgentQueryService(kb_service=_KBService(), query_tool_service=_QueryTool(rag))

    events = []
    async for line in service.stream_events(
        request=_request(monkeypatch),
        body=AgentQueryRequest(query="如何合规推荐配方？", candidate_kb_ids=["kb1"]),
    ):
        events.append(json.loads(line))

    names = [event["event"] for event in events]
    assert names == [
        "session_started",
        "plan_created",
        "round_started",
        "round_result",
        "references",
        "response",
        "response",
        "done",
    ]
    deltas = [event["delta"] for event in events if event["event"] == "response"]
    assert deltas == ["部分1", "部分2"]
    plan_event = events[1]
    assert plan_event["plan_truncated"] is False
    assert plan_event["steps"][0]["priority"] == "P0"


@pytest.mark.asyncio
async def test_stream_emits_error_event_on_plan_failure(monkeypatch):
    _audit_recorder(monkeypatch)
    rag = _FakeRAG(
        _single_step_plan(),
        raw_plan_responses=["坏输出 1", "坏输出 2", "坏输出 3"],
    )
    service = AgentQueryService(kb_service=_KBService(), query_tool_service=_QueryTool(rag))

    events = []
    async for line in service.stream_events(
        request=_request(monkeypatch),
        body=AgentQueryRequest(query="如何合规推荐配方？", candidate_kb_ids=["kb1"]),
    ):
        events.append(json.loads(line))

    assert events[0]["event"] == "session_started"
    assert events[-1]["event"] == "error"
    assert events[-1]["status_code"] == 502
