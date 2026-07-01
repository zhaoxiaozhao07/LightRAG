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
    def __init__(self, agent_plan):
        self.agent_plan = agent_plan
        self.seen_plan_payload = None

    def _build_global_config(self):
        async def agent_func(prompt, **_kwargs):
            self.seen_plan_payload = json.loads(prompt)
            return json.dumps(self.agent_plan, ensure_ascii=False)

        async def query_func(_query, **_kwargs):
            return "最终答案 [A1]"

        return {
            "role_llm_funcs": {"agent": agent_func, "query": query_func},
            "tokenizer": None,
        }


class _QueryTool:
    def __init__(self, rag):
        self.rag = rag
        self.calls = []

    async def get_rag(self, _kb_id):
        return self.rag

    async def retrieve_serial(self, **kwargs):
        self.calls.append(kwargs)
        return QueryToolResult(
            chunks=[
                {
                    "kb_id": kwargs["kb_ids"][0],
                    "chunk_id": "chunk-1",
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


@pytest.mark.asyncio
async def test_agent_query_uses_candidate_kb_subset_and_agent_references(monkeypatch):
    async def noop_audit(*_args, **_kwargs):
        return None

    monkeypatch.setattr("lightrag.api.agent_query_service.append_enterprise_audit_event", noop_audit)
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
                "priority": "P0",
            }
        ],
    }
    rag = _FakeRAG(plan)
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
        }
    ]
    assert result.references[0]["reference_id"] == "A1"
    assert result.references[0]["kb_id"] == "kb1"
    assert result.steps_summary[0]["mode"] == "mix"


@pytest.mark.asyncio
async def test_agent_plan_selecting_kb_outside_effective_set_is_rejected(monkeypatch):
    async def noop_audit(*_args, **_kwargs):
        return None

    monkeypatch.setattr("lightrag.api.agent_query_service.append_enterprise_audit_event", noop_audit)
    plan = {
        "type": "plan",
        "clarification_required": False,
        "steps": [
            {
                "step_index": 1,
                "title": "越权",
                "query": "查询未授权库",
                "kb_ids": ["kb2"],
                "mode": "mix",
                "priority": "P0",
            }
        ],
    }
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
