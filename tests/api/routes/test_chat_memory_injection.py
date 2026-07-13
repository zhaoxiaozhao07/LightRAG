"""Server-side chat-memory injection on the query and agent endpoints.

Drives the real kb_query routes with a fake ``LightRAG`` (recording the
``QueryParam`` it receives) and a fake memory service on ``app.state`` so we
can assert the memory fact block reaches ``param.user_prompt`` — i.e. the
front end no longer needs to search + stitch. Covers the opt-in field,
ownership/enablement gating, fail-open, and the audit metadata surface.
"""

from __future__ import annotations

# ruff: noqa: E402 - LightRAG API modules parse sys.argv at import time.

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from lightrag.api.auth import auth_handler
from lightrag.api.enterprise_auth import (
    AuditService,
    AuthorizationService,
    ChatConversationService,
    SystemSettingsService,
    UserService,
)
from lightrag.api.kb_service import KnowledgeBaseService
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry
from lightrag.api.metadata_store import SQLiteMetadataStore
from lightrag.api.routers.chat_routes import create_chat_routes
from lightrag.api.routers.kb_query_routes import create_kb_query_routes
from lightrag.api.utils_api import get_combined_auth_dependency
sys.argv = _original_argv

pytestmark = pytest.mark.offline

_API_KEY = "test-key"


class FakeRAG:
    def __init__(self, workspace: str):
        self.workspace = workspace
        self.query_params: list = []
        self.kb_active_query_config: dict = {}

    async def aquery_llm(self, query: str, *, param):
        self.query_params.append(param)
        return {
            "llm_response": {
                "content": f"answer: {query}",
                "is_streaming": False,
            },
            "data": {"references": [], "chunks": []},
        }


class BuilderProbe:
    def __init__(self):
        self.instances: dict[str, FakeRAG] = {}

    async def build(self, record) -> FakeRAG:
        rag = FakeRAG(record.workspace)
        self.instances[record.id] = rag
        return rag

    async def finalize(self, rag) -> None:
        return None


class FakeMemoryService:
    def __init__(self):
        self.block: str | None = "[项目记忆] 以下是该项目历史对话沉淀的事实：\n- 采用 NR/BR 并用"
        self.info: dict = {"enabled": True, "project_id": "", "fact_count": 1}
        self.raise_unavailable = False
        self.calls: list[dict] = []

    async def build_memory_block(self, *, user_id, project_id, query, limit=None):
        self.calls.append(
            {
                "user_id": user_id,
                "project_id": project_id,
                "query": query,
                "limit": limit,
            }
        )
        # Mirror the real service: build_memory_block is fail-open and never
        # raises ChatMemoryUnavailableError to the caller.
        if self.raise_unavailable:
            return None, {"enabled": False, "reason": "unavailable"}
        info = dict(self.info)
        info["project_id"] = project_id
        return self.block, info


def _enterprise_args(**overrides):
    values = {
        "enterprise_auth_enabled": True,
        "enterprise_legacy_api_key_superadmin": False,
        "enterprise_disable_global_routes": True,
        "enterprise_rate_limit_enabled": False,
        "enterprise_rate_limit_requests": 60,
        "enterprise_rate_limit_window_seconds": 60.0,
        "enterprise_tenant_rate_limit_requests": 0,
        "enterprise_tenant_rate_limit_window_seconds": 60.0,
        "enterprise_quota_requests": 0,
        "enterprise_quota_window_seconds": 86400.0,
        "enterprise_tenant_quota_requests": 0,
        "enterprise_tenant_quota_window_seconds": 86400.0,
        "enterprise_mask_storage_uris": True,
        "chat_session_default_context_rounds": 1,
        "token_auto_renew": False,
        "token_renew_threshold": 0.5,
        "bilingual_query_enabled": False,
        "bilingual_query_default_mode": "off",
        "top_k": 40,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _patch_enterprise_args(monkeypatch, args) -> None:
    from lightrag.api import config as api_config
    import lightrag.api.utils_api as utils_api

    monkeypatch.setattr(api_config, "global_args", args)
    monkeypatch.setattr(utils_api, "global_args", args)
    dependency_functions = [get_combined_auth_dependency]
    for factory in (create_chat_routes, create_kb_query_routes):
        dependency = factory.__globals__["get_combined_auth_dependency"]
        if dependency not in dependency_functions:
            dependency_functions.append(dependency)
    for dependency in dependency_functions:
        monkeypatch.setitem(dependency.__globals__, "global_args", args)


def _token(user_service, user) -> str:
    return auth_handler.create_token(
        username=user.username,
        role=user.system_role,
        metadata=user_service.token_metadata_for_user(user),
    )


def _build(monkeypatch, tmp_path: Path, *, with_memory=True):
    args = _enterprise_args()
    _patch_enterprise_args(monkeypatch, args)

    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kb.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    from lightrag.api.document_lifecycle_service import DocumentLifecycleService

    document_service = DocumentLifecycleService(
        kb_service, metadata_store, tmp_path / "inputs"
    )
    audit_service = AuditService(metadata_store)
    user_service = UserService(metadata_store, audit_service)
    settings_service = SystemSettingsService(metadata_store)
    chat_service = ChatConversationService(metadata_store, audit_service)
    authz_service = AuthorizationService(
        metadata_store, audit_service, kb_service=kb_service
    )
    probe = BuilderProbe()
    registry = LightRAGInstanceRegistry(kb_service, probe.build, probe.finalize)
    memory_service = FakeMemoryService() if with_memory else None

    async def seed():
        await kb_service.initialize()
        await metadata_store.initialize()
        await settings_service.initialize_registration_setting(False)
        admin = await user_service.bootstrap_super_admin(
            username="admin", password="admin-pass", password_hash=None
        )
        alice = await user_service.create_user(
            username="alice", password="alice-pass", can_create_kb=True
        )
        bob = await user_service.create_user(username="bob", password="bob-pass")
        kb = await kb_service.create(name="KB", kb_id="kb_mem")
        await authz_service.grant_kb_role(
            kb.id, alice.id, "kb_viewer", granted_by=admin.id
        )
        return alice, bob

    alice, bob = asyncio.run(seed())

    app = FastAPI()
    app.state.enterprise_enabled = True
    app.state.metadata_store = metadata_store
    app.state.enterprise_user_service = user_service
    app.state.enterprise_settings_service = settings_service
    app.state.enterprise_chat_conversation_service = chat_service
    app.state.enterprise_authorization_service = authz_service
    app.state.enterprise_audit_service = audit_service
    if memory_service is not None:
        app.state.enterprise_chat_memory_service = memory_service
    app.include_router(create_chat_routes(api_key=_API_KEY))
    app.include_router(
        create_kb_query_routes(document_service, registry, api_key=_API_KEY)
    )
    return TestClient(app), user_service, alice, bob, probe, memory_service


def _project(client, headers) -> str:
    resp = client.post("/chat/projects", json={"name": "记忆项目"}, headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


def test_kb_query_injects_memory_into_user_prompt(monkeypatch, tmp_path):
    client, user_service, alice, _bob, probe, memory = _build(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    project_id = _project(client, headers)

    resp = client.post(
        "/kbs/kb_mem/query",
        json={
            "query": "低温性能怎么做？",
            "mode": "mix",
            "user_prompt": "请用中文回答",
            "memory": {"project_id": project_id, "limit": 5},
        },
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Response metadata advertises the injection.
    assert body["metadata"]["memory"]["enabled"] is True
    assert body["metadata"]["memory"]["fact_count"] == 1
    # The memory service was called with the owning user + project + query.
    assert memory.calls == [
        {
            "user_id": alice.id,
            "project_id": project_id,
            "query": "低温性能怎么做？",
            "limit": 5,
        }
    ]
    # The fact block is prepended to the effective user_prompt reaching the RAG.
    param = probe.instances["kb_mem"].query_params[-1]
    assert "[项目记忆]" in param.user_prompt
    assert param.user_prompt.strip().endswith("请用中文回答")


def test_kb_query_without_memory_field_is_untouched(monkeypatch, tmp_path):
    client, user_service, alice, _bob, probe, memory = _build(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    resp = client.post(
        "/kbs/kb_mem/query",
        json={"query": "无记忆注入", "mode": "mix"},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    assert "memory" not in resp.json()["metadata"]
    assert memory.calls == []
    param = probe.instances["kb_mem"].query_params[-1]
    assert param.user_prompt in (None, "", "n/a")


def test_kb_query_memory_foreign_project_is_404(monkeypatch, tmp_path):
    client, user_service, alice, _bob, _probe, _memory = _build(monkeypatch, tmp_path)
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    _project(client, alice_headers)
    # A project id alice does not own is an ownership 404 (no existence leak),
    # not a memory-search error.
    resp = client.post(
        "/kbs/kb_mem/query",
        json={"query": "abc", "mode": "mix", "memory": {"project_id": "proj_ghost"}},
        headers=alice_headers,
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Chat project not found"


def test_kb_query_memory_disabled_returns_503(monkeypatch, tmp_path):
    client, user_service, alice, _bob, _probe, _memory = _build(
        monkeypatch, tmp_path, with_memory=False
    )
    headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    project_id = _project(client, headers)
    resp = client.post(
        "/kbs/kb_mem/query",
        json={"query": "abc", "mode": "mix", "memory": {"project_id": project_id}},
        headers=headers,
    )
    assert resp.status_code == 503
    assert resp.json()["detail"] == "Chat memory is not enabled"


def test_kb_query_memory_unavailable_fails_open(monkeypatch, tmp_path):
    client, user_service, alice, _bob, probe, memory = _build(monkeypatch, tmp_path)
    memory.raise_unavailable = True
    headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    project_id = _project(client, headers)
    resp = client.post(
        "/kbs/kb_mem/query",
        json={
            "query": "后端挂了也要能答",
            "mode": "mix",
            "memory": {"project_id": project_id},
        },
        headers=headers,
    )
    # Fail-open: the query still succeeds, memory just reports unavailable.
    assert resp.status_code == 200, resp.text
    assert resp.json()["metadata"]["memory"] == {"enabled": False, "reason": "unavailable"}
    param = probe.instances["kb_mem"].query_params[-1]
    assert param.user_prompt in (None, "", "n/a")
