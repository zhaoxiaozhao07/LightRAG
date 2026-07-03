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


def _kb(kb_id: str, name: str, description: str, metadata: dict | None = None) -> KnowledgeBaseRecord:
    now = utc_now_iso()
    return KnowledgeBaseRecord(
        id=kb_id,
        name=name,
        description=description,
        workspace=f"kb_{kb_id}",
        status="active",
        active_config_version_id=None,
        owner_id=None,
        tenant_id=None,
        visibility="private",
        created_at=now,
        updated_at=now,
        metadata=metadata or {},
    )


class _KBService:
    def __init__(self):
        self.records = [
            _kb("kb_formula", "配方知识库", "历史配方与配比案例"),
            _kb("kb_exp", "实验数据知识库", "实验与测试数据"),
            _kb("kb_paper", "论文知识库", "文献与机理研究"),
            _kb("kb_side", "胎侧知识库", "胎侧应用规范与经验"),
        ]

    async def list(self, *, include_deleted=False):
        return self.records


class _FakeRAG:
    """Queues one JSON response per AGENT-role call, in call order."""

    def __init__(self, agent_responses: list[str], *, answer_deltas=None):
        self.agent_responses = list(agent_responses)
        self.agent_payloads: list[dict] = []
        self.answer_deltas = answer_deltas
        self.query_prompts: list[str] = []

    def _build_global_config(self):
        async def agent_func(prompt, **_kwargs):
            self.agent_payloads.append(json.loads(prompt))
            if not self.agent_responses:
                raise AssertionError("unexpected AGENT LLM call")
            return self.agent_responses.pop(0)

        async def query_func(_query, stream=False, **kwargs):
            self.query_prompts.append(kwargs.get("system_prompt", ""))
            if stream and self.answer_deltas is not None:
                async def _gen():
                    for delta in self.answer_deltas:
                        yield delta

                return _gen()
            return "推荐配比 [A1]"

        return {
            "role_llm_funcs": {"agent": agent_func, "query": query_func},
            "tokenizer": None,
        }


class _QueryTool:
    def __init__(
        self,
        rag,
        *,
        fail_calls: set[int] | None = None,
        empty_calls: set[int] | None = None,
    ):
        self.rag = rag
        self.calls = []
        self._fail_calls = fail_calls or set()
        self._empty_calls = empty_calls or set()

    async def get_rag(self, _kb_id):
        return self.rag

    async def retrieve_serial(self, **kwargs):
        self.calls.append(kwargs)
        call_no = len(self.calls)
        if call_no in self._fail_calls:
            raise HTTPException(
                status_code=502, detail={"error_code": "kb_retrieve_failed"}
            )
        kb_id = kwargs["kb_ids"][0]
        if call_no in self._empty_calls:
            return QueryToolResult(
                chunks=[],
                rag=self.rag,
                param=QueryParam(mode=kwargs["mode"]),
                queried_kb_ids=kwargs["kb_ids"],
                per_kb_chunk_counts={kb_id: 0},
            )
        return QueryToolResult(
            chunks=[
                {
                    "kb_id": kb_id,
                    "chunk_id": f"chunk-{call_no}",
                    "file_path": "doc.md",
                    "content": f"证据内容 {call_no}",
                }
            ],
            rag=self.rag,
            param=QueryParam(mode=kwargs["mode"]),
            queried_kb_ids=kwargs["kb_ids"],
            per_kb_chunk_counts={kb_id: 1},
        )


def _request(monkeypatch):
    monkeypatch.setattr(
        "lightrag.api.agent_query_service.enterprise_auth_enabled", lambda: True
    )
    monkeypatch.setattr(
        "lightrag.api.agent_query_service.agent_query_enabled", lambda: True
    )
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
        enterprise_authorization_service=AuthorizationService(
            metadata_store=SimpleNamespace()
        ),
    )
    return SimpleNamespace(state=state, app=SimpleNamespace(state=state))


def _audit_recorder(monkeypatch) -> list[str]:
    events: list[str] = []

    async def record(_request, event, **_kwargs):
        events.append(event)

    monkeypatch.setattr(
        "lightrag.api.agent_query_service.append_enterprise_audit_event", record
    )
    monkeypatch.setattr(
        "lightrag.api.agent_staged_service.append_enterprise_audit_event", record
    )
    return events


def _requirement_json(properties=None) -> str:
    return json.dumps(
        {
            "type": "requirement",
            "clarification_required": False,
            "application": "胎侧胶料",
            "conditions": ["高寒环境"],
            "target_properties": properties
            or [
                {"name": "低温屈挠性", "why": "低温开裂", "priority": "P0"},
                {"name": "耐臭氧老化", "why": "户外使用", "priority": "P1"},
            ],
            "constraints": [],
        },
        ensure_ascii=False,
    )


def _skeleton_plan_json() -> str:
    return json.dumps(
        {
            "type": "skeleton_plan",
            "kb_roles": {
                "kb_formula": "reference_formula",
                "kb_exp": "experimental",
                "kb_paper": "literature",
                "kb_side": "application_spec",
            },
            "steps": [
                {
                    "step_index": 1,
                    "title": "查参考配方",
                    "query": "高寒环境胎侧胶料参考配方与配比案例",
                    "kb_ids": ["kb_formula", "kb_side"],
                    "mode": "mix",
                    "priority": "P0",
                }
            ],
        },
        ensure_ascii=False,
    )


def _skeleton_extract_json(source_refs=None) -> str:
    return json.dumps(
        {
            "type": "skeleton",
            "components": [
                {
                    "material": "NR/BR 并用",
                    "ratio": "50/50 phr",
                    "function": "低温屈挠性能",
                    "source_refs": source_refs or ["A1"],
                }
            ],
            "open_questions": ["高寒环境下 BR 并用比例对胎侧屈挠性能的影响"],
            "rationale": "最接近的案例",
        },
        ensure_ascii=False,
    )


def _verdicts_json(verdicts) -> str:
    return json.dumps({"type": "verdicts", "verdicts": verdicts}, ensure_ascii=False)


def _happy_path_responses() -> list[str]:
    # Retrieval order: skeleton=A1, factor(open question)=A2,
    # factor(component)=A3, validation rounds=A4/A5.
    return [
        _requirement_json(),
        _skeleton_plan_json(),
        _skeleton_extract_json(),
        _verdicts_json(
            [
                {
                    "property": "低温屈挠性",
                    "verdict": "supported",
                    "evidence_refs": ["A4"],
                    "note": "有实测数据",
                },
                {
                    "property": "耐臭氧老化",
                    "verdict": "supported",
                    "evidence_refs": ["A5"],
                    "note": "有实测数据",
                },
            ]
        ),
    ]


def _staged_body(**overrides) -> AgentQueryRequest:
    payload = {"query": "推荐一种高寒地区使用的胎侧胶料配比", "workflow": "staged"}
    payload.update(overrides)
    return AgentQueryRequest(**payload)


def test_workflow_defaults_to_plan_mode():
    assert AgentQueryRequest(query="任意问题").workflow == "plan"


@pytest.mark.asyncio
async def test_staged_happy_path_stream_event_sequence(monkeypatch):
    _audit_recorder(monkeypatch)
    rag = _FakeRAG(_happy_path_responses(), answer_deltas=["配比", "表 [A1]"])
    service = AgentQueryService(kb_service=_KBService(), query_tool_service=_QueryTool(rag))

    events = []
    async for line in service.stream_events(
        request=_request(monkeypatch), body=_staged_body()
    ):
        events.append(json.loads(line))

    names = [event["event"] for event in events]
    assert names == [
        "session_started",
        "stage_started",  # requirement
        "requirement_parsed",
        "stage_started",  # skeleton
        "kb_roles_assigned",
        "round_started",
        "round_result",
        "skeleton_extracted",
        "stage_started",  # factor_evidence
        "round_started",  # open question
        "round_result",
        "round_started",  # component follow-up
        "round_result",
        "stage_started",  # validation
        "round_started",
        "round_result",
        "round_started",
        "round_result",
        "validation_verdicts",
        "references",
        "response",
        "response",
        "done",
    ]
    assert events[0]["metadata"]["workflow"] == "staged"
    stages = [event["stage"] for event in events if event["event"] == "stage_started"]
    assert stages == ["requirement", "skeleton", "factor_evidence", "validation"]
    assert events[4]["kb_roles"]["kb_formula"] == "reference_formula"
    skeleton_event = events[7]
    assert skeleton_event["components"][0]["source_refs"] == ["A1"]
    assert skeleton_event["dropped_components"] == 0
    factor_round = events[9]
    assert factor_round["kb_ids"] == ["kb_exp", "kb_paper", "kb_side"]
    validation_round = events[14]
    assert validation_round["kb_ids"] == ["kb_exp"]
    assert validation_round["priority"] == "P0"
    verdict_event = events[18]
    assert verdict_event["after_repair"] is False
    assert {v["verdict"] for v in verdict_event["verdicts"]} == {"supported"}
    reference_ids = [ref["reference_id"] for ref in events[19]["references"]]
    assert reference_ids == ["A1", "A2", "A3", "A4", "A5"]
    assert events[19]["references"][0]["stage"] == "skeleton"
    assert events[19]["references"][0]["evidence_role"] == "reference_formula"


@pytest.mark.asyncio
async def test_staged_happy_path_result_metadata_and_verdict_payload(monkeypatch):
    _audit_recorder(monkeypatch)
    monkeypatch.setattr(
        "lightrag.api.agent_staged_service.agent_staged_max_retrievals", lambda: 24
    )
    rag = _FakeRAG(_happy_path_responses())
    tool = _QueryTool(rag)
    service = AgentQueryService(kb_service=_KBService(), query_tool_service=tool)

    result = await service.run(request=_request(monkeypatch), body=_staged_body())

    assert result.status == "success"
    assert result.answer == "推荐配比 [A1]"
    metadata = result.metadata
    assert metadata["workflow"] == "staged"
    assert metadata["requirement"]["application"] == "胎侧胶料"
    assert metadata["skeleton_component_count"] == 1
    assert metadata["retrieval_budget"] == {"max": 24, "used": 5}
    assert [v["verdict"] for v in metadata["property_verdicts"]] == [
        "supported",
        "supported",
    ]
    assert metadata["kb_roles"]["kb_exp"] == "experimental"
    # Verdict prompt maps each property to the chunks of its own round.
    verdict_payload = rag.agent_payloads[3]
    by_property = {
        entry["property"]: entry for entry in verdict_payload["evidence_by_property"]
    }
    assert by_property["低温屈挠性"]["chunks"][0]["reference_id"] == "A4"
    assert by_property["耐臭氧老化"]["chunks"][0]["reference_id"] == "A5"
    # Structured summary is injected into the synthesis system prompt.
    assert "property_verdicts" in rag.query_prompts[0]
    assert len(tool.calls) == 5


@pytest.mark.asyncio
async def test_staged_clarification_short_circuits(monkeypatch):
    _audit_recorder(monkeypatch)
    rag = _FakeRAG(
        [
            json.dumps(
                {
                    "type": "requirement",
                    "clarification_required": True,
                    "clarification_question": "请补充目标应用与环境。",
                    "target_properties": [],
                },
                ensure_ascii=False,
            )
        ]
    )
    tool = _QueryTool(rag)
    service = AgentQueryService(kb_service=_KBService(), query_tool_service=tool)

    result = await service.run(request=_request(monkeypatch), body=_staged_body())

    assert result.status == "clarification_required"
    assert result.clarification_question == "请补充目标应用与环境。"
    assert result.metadata["workflow"] == "staged"
    assert tool.calls == []
    assert len(rag.agent_payloads) == 1


@pytest.mark.asyncio
async def test_staged_requirement_invalid_after_retries(monkeypatch):
    events = _audit_recorder(monkeypatch)
    rag = _FakeRAG(["坏输出 1", "坏输出 2", "坏输出 3"])
    service = AgentQueryService(kb_service=_KBService(), query_tool_service=_QueryTool(rag))

    with pytest.raises(HTTPException) as exc:
        await service.run(request=_request(monkeypatch), body=_staged_body())

    assert exc.value.status_code == 502
    assert exc.value.detail["error_code"] == "agent_requirement_invalid"
    assert "agent_session_failed" in events


@pytest.mark.asyncio
async def test_staged_skeleton_component_without_valid_refs_is_dropped(monkeypatch):
    _audit_recorder(monkeypatch)
    responses = [
        _requirement_json(),
        _skeleton_plan_json(),
        _skeleton_extract_json(source_refs=["A99"]),
        _verdicts_json(
            [
                {
                    "property": "低温屈挠性",
                    "verdict": "supported",
                    "evidence_refs": ["A3"],
                },
                {
                    "property": "耐臭氧老化",
                    "verdict": "supported",
                    "evidence_refs": ["A4"],
                },
            ]
        ),
    ]
    rag = _FakeRAG(responses)
    service = AgentQueryService(kb_service=_KBService(), query_tool_service=_QueryTool(rag))

    result = await service.run(request=_request(monkeypatch), body=_staged_body())

    assert result.status == "success"
    assert result.metadata["skeleton_component_count"] == 0
    assert result.metadata["dropped_component_count"] == 1
    assert any("骨架" in note for note in result.metadata["clipped"])


@pytest.mark.asyncio
async def test_staged_verdicts_fail_closed(monkeypatch):
    _audit_recorder(monkeypatch)
    responses = [
        _requirement_json(),
        _skeleton_plan_json(),
        _skeleton_extract_json(),
        # "低温屈挠性" claims support without valid refs; "耐臭氧老化" missing.
        _verdicts_json(
            [
                {
                    "property": "低温屈挠性",
                    "verdict": "supported",
                    "evidence_refs": ["A999"],
                }
            ]
        ),
        # Gap repair proposes nothing.
        json.dumps({"type": "repair_plan", "steps": []}, ensure_ascii=False),
    ]
    rag = _FakeRAG(responses)
    service = AgentQueryService(kb_service=_KBService(), query_tool_service=_QueryTool(rag))

    result = await service.run(request=_request(monkeypatch), body=_staged_body())

    verdicts = {v["property"]: v for v in result.metadata["property_verdicts"]}
    assert verdicts["低温屈挠性"]["verdict"] == "no_data"
    assert "降级" in verdicts["低温屈挠性"]["note"]
    assert verdicts["耐臭氧老化"]["verdict"] == "no_data"
    assert len(rag.agent_payloads) == 5


@pytest.mark.asyncio
async def test_staged_gap_repair_updates_verdicts(monkeypatch):
    _audit_recorder(monkeypatch)
    responses = [
        _requirement_json(),
        _skeleton_plan_json(),
        _skeleton_extract_json(),
        _verdicts_json(
            [
                {
                    "property": "低温屈挠性",
                    "verdict": "supported",
                    "evidence_refs": ["A3"],
                },
                {"property": "耐臭氧老化", "verdict": "no_data", "evidence_refs": []},
            ]
        ),
        json.dumps(
            {
                "type": "repair_plan",
                "steps": [
                    {
                        "step_index": 1,
                        "title": "补查臭氧老化",
                        "query": "胎侧胶料耐臭氧老化实验数据",
                        "kb_ids": ["kb_exp"],
                        "mode": "naive",
                        "priority": "P0",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        _verdicts_json(
            [
                {
                    "property": "耐臭氧老化",
                    "verdict": "supported",
                    "evidence_refs": ["A5"],
                    "note": "补查到实测",
                }
            ]
        ),
    ]
    rag = _FakeRAG(responses, answer_deltas=["答案 [A5]"])
    tool = _QueryTool(rag)
    service = AgentQueryService(kb_service=_KBService(), query_tool_service=tool)

    events = []
    async for line in service.stream_events(
        request=_request(monkeypatch), body=_staged_body()
    ):
        events.append(json.loads(line))

    verdict_events = [e for e in events if e["event"] == "validation_verdicts"]
    assert len(verdict_events) == 2
    assert verdict_events[0]["after_repair"] is False
    assert verdict_events[1]["after_repair"] is True
    updated = {v["property"]: v for v in verdict_events[1]["verdicts"]}
    assert updated["耐臭氧老化"]["verdict"] == "supported"
    assert updated["耐臭氧老化"]["evidence_refs"] == ["A5"]
    assert updated["低温屈挠性"]["verdict"] == "supported"
    stages = [e["stage"] for e in events if e["event"] == "stage_started"]
    assert stages[-1] == "gap_repair"
    repair_rounds = [
        e for e in events if e["event"] == "round_result" and e.get("stage") == "gap_repair"
    ]
    assert len(repair_rounds) == 1
    assert repair_rounds[0]["mode"] == "naive"


@pytest.mark.asyncio
async def test_staged_budget_cap_skips_and_reports(monkeypatch):
    _audit_recorder(monkeypatch)
    monkeypatch.setattr(
        "lightrag.api.agent_staged_service.agent_staged_max_retrievals", lambda: 1
    )
    responses = [
        _requirement_json(),
        _skeleton_plan_json(),
        _skeleton_extract_json(),
        _verdicts_json([]),
    ]
    rag = _FakeRAG(responses)
    tool = _QueryTool(rag)
    service = AgentQueryService(kb_service=_KBService(), query_tool_service=tool)

    result = await service.run(request=_request(monkeypatch), body=_staged_body())

    assert result.status == "success"
    assert len(tool.calls) == 1  # only the skeleton step ran
    assert result.metadata["retrieval_budget"] == {"max": 1, "used": 1}
    skipped_notes = [
        note for note in result.metadata["clipped"] if "预算不足" in note
    ]
    assert len(skipped_notes) == 2
    assert all(
        v["verdict"] == "no_data" for v in result.metadata["property_verdicts"]
    )
    # Budget exhausted: no repair planning call was made.
    assert len(rag.agent_payloads) == 4


@pytest.mark.asyncio
async def test_staged_skeleton_plan_outside_effective_set_is_rejected(monkeypatch):
    events = _audit_recorder(monkeypatch)
    plan = json.dumps(
        {
            "type": "skeleton_plan",
            "kb_roles": {},
            "steps": [
                {
                    "step_index": 1,
                    "title": "越权",
                    "query": "越权检索参考配方",
                    "kb_ids": ["kb_paper"],
                    "mode": "mix",
                    "priority": "P0",
                }
            ],
        },
        ensure_ascii=False,
    )
    rag = _FakeRAG([_requirement_json(), plan])
    service = AgentQueryService(kb_service=_KBService(), query_tool_service=_QueryTool(rag))

    with pytest.raises(HTTPException) as exc:
        await service.run(
            request=_request(monkeypatch),
            body=_staged_body(candidate_kb_ids=["kb_formula"]),
        )

    assert exc.value.status_code == 403
    assert "agent_session_failed" in events


@pytest.mark.asyncio
async def test_staged_caps_kbs_per_step_by_priority(monkeypatch):
    """Unknown-size KB fleets: per-step KB count is capped, preferring
    higher manual agent_priority, and the narrowing is reported."""
    _audit_recorder(monkeypatch)
    monkeypatch.setattr(
        "lightrag.api.agent_staged_service.agent_staged_max_retrievals", lambda: 24
    )
    monkeypatch.setattr(
        "lightrag.api.agent_staged_service.agent_staged_max_kbs_per_step", lambda: 4
    )

    kb_service = _KBService()
    kb_service.records = [
        _kb("k1", "库一", "未知内容一"),
        _kb("k2", "库二", "未知内容二"),
        _kb("k3", "库三", "未知内容三"),
        _kb("k4", "库四", "未知内容四"),
        _kb("k5", "库五", "核心库", metadata={"agent_priority": 10}),
        _kb("k6", "库六", "次级核心库", metadata={"agent_priority": 5}),
    ]
    responses = [
        _requirement_json(),
        json.dumps(
            {
                "type": "skeleton_plan",
                "kb_roles": {},  # nothing classified -> stages fall back to all KBs
                "steps": [
                    {
                        "step_index": 1,
                        "title": "查参考配方",
                        "query": "高寒环境参考配方与配比案例",
                        "kb_ids": ["k1", "k2", "k3", "k4", "k5"],
                        "mode": "mix",
                        "priority": "P0",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        _skeleton_extract_json(),
        _verdicts_json([]),
        json.dumps({"type": "repair_plan", "steps": []}, ensure_ascii=False),
    ]
    rag = _FakeRAG(responses)
    tool = _QueryTool(rag)
    service = AgentQueryService(kb_service=kb_service, query_tool_service=tool)

    result = await service.run(request=_request(monkeypatch), body=_staged_body())

    assert result.status == "success"
    # Skeleton step narrowed from 5 KBs to 4: priority KB first, then plan order.
    assert tool.calls[0]["kb_ids"] == ["k5", "k1", "k2", "k3"]
    # Factor/validation fall back to "all KBs" (roles unassigned) and are
    # capped to the top-priority subset.
    assert tool.calls[1]["kb_ids"] == ["k5", "k6", "k1", "k2"]
    assert tool.calls[2]["kb_ids"] == ["k5", "k6", "k1", "k2"]
    assert any("裁剪到 4 个" in note for note in result.metadata["clipped"])
    assert any("优先级最高的 4 个" in note for note in result.metadata["clipped"])


@pytest.mark.asyncio
async def test_plan_mode_empty_result_retries_with_fallback_mode(monkeypatch):
    _audit_recorder(monkeypatch)
    plan = json.dumps(
        {
            "type": "plan",
            "clarification_required": False,
            "steps": [
                {
                    "step_index": 1,
                    "title": "查配方",
                    "query": "检索胎侧配方",
                    "kb_ids": ["kb_formula"],
                    "mode": "mix",
                    "priority": "P0",
                }
            ],
        },
        ensure_ascii=False,
    )
    rag = _FakeRAG([plan])
    tool = _QueryTool(rag, empty_calls={1})
    service = AgentQueryService(kb_service=_KBService(), query_tool_service=tool)

    result = await service.run(
        request=_request(monkeypatch),
        body=AgentQueryRequest(query="推荐胎侧配方", candidate_kb_ids=["kb_formula"]),
    )

    assert result.status == "success"
    assert len(tool.calls) == 2
    assert tool.calls[0]["mode"] == "mix"
    assert tool.calls[1]["mode"] == "naive"
    assert result.steps_summary[0]["retried_mode"] == "naive"
    assert result.steps_summary[0]["chunk_count"] == 1
    assert result.metadata["workflow"] == "plan"
    assert result.references
