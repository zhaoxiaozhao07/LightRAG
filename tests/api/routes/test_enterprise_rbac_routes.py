from __future__ import annotations

# ruff: noqa: E402 - LightRAG API modules parse sys.argv at import time.

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from lightrag.api.auth import auth_handler
from lightrag.api.document_lifecycle_service import DocumentLifecycleService
from lightrag.api.enterprise_auth import (
    AuditService,
    AuthorizationService,
    SystemSettingsService,
    UserService,
)
from lightrag.api.job_service import JobService
from lightrag.api.kb_service import KnowledgeBaseService
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry, LightRAGLike
from lightrag.api.metadata_store import SQLiteMetadataStore
from lightrag.api.routers.enterprise_routes import create_enterprise_routes
from lightrag.api.routers.kb_query_routes import create_kb_query_routes
from lightrag.api.routers.kb_routes import create_kb_routes
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.base import QueryParam
sys.argv = _original_argv

pytestmark = pytest.mark.offline

_API_KEY = "test-key"


class FakeRAG:
    def __init__(self, workspace: str):
        self.workspace = workspace
        self.query_params: list[QueryParam] = []
        self.kb_active_query_config: dict[str, object] = {}

    async def finalize_storages(self) -> None:
        return None

    async def adrop_all_storages(self) -> dict:
        return {"dropped": 0, "failed": 0, "errors": []}

    async def aquery_llm(self, query: str, *, param: QueryParam):
        self.query_params.append(param)
        return {
            "llm_response": {
                "content": f"answer from {self.workspace}: {query}",
                "is_streaming": False,
            },
            "data": {"references": [], "chunks": []},
        }

    async def aquery_data(self, query: str, *, param: QueryParam):
        self.query_params.append(param)
        return {
            "status": "success",
            "message": "ok",
            "data": {},
            "metadata": {"query_mode": param.mode},
        }


class BuilderProbe:
    def __init__(self):
        self.instances: dict[str, FakeRAG] = {}
        self.active_query_config: dict[str, object] = {}

    async def build(self, record) -> FakeRAG:
        rag = FakeRAG(record.workspace)
        if self.active_query_config:
            rag.kb_active_query_config = dict(self.active_query_config)
        self.instances[record.id] = rag
        return rag

    async def finalize(self, rag: LightRAGLike) -> None:
        return None


def _enterprise_args(**overrides):
    values = {
        "enterprise_auth_enabled": True,
        "enterprise_legacy_api_key_superadmin": False,
        "enterprise_disable_global_routes": True,
        "token_auto_renew": False,
        "token_renew_threshold": 0.5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _patch_enterprise_args(monkeypatch, args, *, whitelist_patterns=None) -> None:
    from lightrag.api import config as api_config
    import lightrag.api.utils_api as utils_api

    monkeypatch.setattr(api_config, "global_args", args)
    monkeypatch.setattr(utils_api, "global_args", args)
    if whitelist_patterns is not None:
        monkeypatch.setattr(utils_api, "whitelist_patterns", whitelist_patterns)

    dependency_functions = [get_combined_auth_dependency]
    for factory in (create_kb_routes, create_kb_query_routes, create_enterprise_routes):
        dependency = factory.__globals__["get_combined_auth_dependency"]
        if dependency not in dependency_functions:
            dependency_functions.append(dependency)

    for dependency in dependency_functions:
        monkeypatch.setitem(dependency.__globals__, "global_args", args)
        if whitelist_patterns is not None:
            monkeypatch.setitem(
                dependency.__globals__, "whitelist_patterns", whitelist_patterns
            )


def _token(user_service: UserService, user) -> str:
    return auth_handler.create_token(
        username=user.username,
        role=user.system_role,
        metadata=user_service.token_metadata_for_user(user),
    )


def _build_enterprise_client(monkeypatch, tmp_path: Path):
    args = _enterprise_args()
    _patch_enterprise_args(monkeypatch, args)

    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kb.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    document_service = DocumentLifecycleService(
        kb_service, metadata_store, tmp_path / "inputs"
    )
    job_service = JobService(kb_service, metadata_store)
    audit_service = AuditService(metadata_store)
    user_service = UserService(metadata_store, audit_service)
    settings_service = SystemSettingsService(metadata_store)
    authz_service = AuthorizationService(metadata_store, audit_service)
    probe = BuilderProbe()
    registry = LightRAGInstanceRegistry(kb_service, probe.build, probe.finalize)

    async def seed():
        await kb_service.initialize()
        await metadata_store.initialize()
        await settings_service.initialize_registration_setting(False)
        admin = await user_service.bootstrap_super_admin(
            username="admin",
            password="admin-pass",
            password_hash=None,
        )
        alice = await user_service.create_user(
            username="alice",
            password="alice-pass",
            can_create_kb=True,
        )
        bob = await user_service.create_user(
            username="bob",
            password="bob-pass",
        )
        return admin, alice, bob

    admin, alice, bob = asyncio.run(seed())

    app = FastAPI()
    app.state.enterprise_enabled = True
    app.state.enterprise_user_service = user_service
    app.state.enterprise_settings_service = settings_service
    app.state.enterprise_authorization_service = authz_service
    app.state.enterprise_audit_service = audit_service
    app.include_router(
        create_kb_routes(kb_service, registry, api_key=_API_KEY, job_service=job_service)
    )
    app.include_router(create_kb_query_routes(document_service, registry, api_key=_API_KEY))
    app.include_router(create_enterprise_routes(api_key=_API_KEY))
    return (
        TestClient(app),
        user_service,
        authz_service,
        admin,
        alice,
        bob,
        probe,
    )


def test_enterprise_kb_create_list_acl_delete_and_bypass(monkeypatch, tmp_path):
    client, user_service, _authz, admin, alice, bob, probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}

    denied_create = client.post(
        "/kbs", json={"id": "kb_denied", "name": "Denied"}, headers=bob_headers
    )
    assert denied_create.status_code == 403

    created = client.post(
        "/kbs",
        json={
            "id": "kb_alpha",
            "name": "Alpha",
            "owner_id": "spoofed-owner",
            "tenant_id": "spoofed-tenant",
        },
        headers=alice_headers,
    )
    assert created.status_code == 200, created.text
    created_body = created.json()
    assert created_body["owner_id"] == alice.id
    assert created_body["tenant_id"] is None

    alice_list = client.get("/kbs", headers=alice_headers)
    assert alice_list.status_code == 200
    assert [item["id"] for item in alice_list.json()["knowledge_bases"]] == ["kb_alpha"]

    bob_list = client.get("/kbs", headers=bob_headers)
    assert bob_list.status_code == 200
    assert bob_list.json()["knowledge_bases"] == []

    bob_query_denied = client.post(
        "/kbs/kb_alpha/query",
        json={"query": "what is alpha", "mode": "mix"},
        headers=bob_headers,
    )
    assert bob_query_denied.status_code == 403

    grant = client.put(
        "/admin/kbs/kb_alpha/acl",
        json={"user_id": bob.id, "role": "kb_viewer"},
        headers=admin_headers,
    )
    assert grant.status_code == 200, grant.text
    assert grant.json()["role"] == "kb_viewer"

    bob_query = client.post(
        "/kbs/kb_alpha/query",
        json={"query": "what is alpha", "mode": "mix"},
        headers=bob_headers,
    )
    assert bob_query.status_code == 200, bob_query.text
    assert "answer from" in bob_query.json()["response"]
    assert probe.instances["kb_alpha"].query_params[-1].mode == "mix"

    bypass_denied = client.post(
        "/kbs/kb_alpha/query",
        json={"query": "raw model", "mode": "bypass"},
        headers=bob_headers,
    )
    assert bypass_denied.status_code == 403

    spoof_update = client.patch(
        "/kbs/kb_alpha",
        json={"owner_id": "evil-owner", "tenant_id": "evil-tenant", "name": "Safe"},
        headers=alice_headers,
    )
    assert spoof_update.status_code == 200, spoof_update.text
    assert spoof_update.json()["name"] == "Safe"
    assert spoof_update.json()["owner_id"] == alice.id
    assert spoof_update.json()["tenant_id"] is None

    owner_delete_denied = client.delete("/kbs/kb_alpha", headers=alice_headers)
    assert owner_delete_denied.status_code == 403

    admin_delete = client.delete("/kbs/kb_alpha", headers=admin_headers)
    assert admin_delete.status_code == 200, admin_delete.text
    assert admin_delete.json()["status"] == "deleted"

    audit_events = client.get("/admin/audit-events", headers=admin_headers)
    assert audit_events.status_code == 200
    assert any(event["event_type"] == "kb_deleted" for event in audit_events.json())


def test_enterprise_bypass_query_default_requires_capability(monkeypatch, tmp_path):
    client, user_service, _authz, admin, alice, bob, probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}

    created = client.post(
        "/kbs",
        json={"id": "kb_default_bypass", "name": "Default Bypass"},
        headers=alice_headers,
    )
    assert created.status_code == 200, created.text
    grant = client.put(
        "/admin/kbs/kb_default_bypass/acl",
        json={"user_id": bob.id, "role": "kb_viewer"},
        headers=admin_headers,
    )
    assert grant.status_code == 200, grant.text

    probe.active_query_config = {"mode": "bypass"}
    denied = client.post(
        "/kbs/kb_default_bypass/query",
        json={"query": "uses active bypass default"},
        headers=bob_headers,
    )
    assert denied.status_code == 403
    assert denied.json()["detail"] == "Bypass-query permission required"

    updated = client.patch(
        f"/admin/users/{bob.id}",
        json={"can_use_bypass_query": True},
        headers=admin_headers,
    )
    assert updated.status_code == 200, updated.text
    refreshed_bob = asyncio.run(user_service.get_user_or_404(bob.id))
    allowed = client.post(
        "/kbs/kb_default_bypass/query",
        json={"query": "uses active bypass default"},
        headers={"Authorization": f"Bearer {_token(user_service, refreshed_bob)}"},
    )
    assert allowed.status_code == 200, allowed.text
    assert probe.instances["kb_default_bypass"].query_params[-1].mode == "bypass"


def test_enterprise_registration_patch_and_disabled_token_rejection(monkeypatch, tmp_path):
    client, user_service, _authz, admin, _alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}

    closed = client.post(
        "/auth/register", json={"username": "carol", "password": "carol-pass"}
    )
    assert closed.status_code == 403

    toggle = client.patch(
        "/admin/settings/registration",
        json={"enabled": True},
        headers=admin_headers,
    )
    assert toggle.status_code == 200, toggle.text
    assert toggle.json() == {"enabled": True}

    registered = client.post(
        "/auth/register", json={"username": "carol", "password": "carol-pass"}
    )
    assert registered.status_code == 200, registered.text
    body = registered.json()
    assert body["auth_mode"] == "enterprise"
    assert body["user"]["can_create_kb"] is False

    me = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert me.status_code == 200
    assert me.json()["user"]["username"] == "carol"

    disabled = client.patch(
        f"/admin/users/{bob.id}", json={"status": "disabled"}, headers=admin_headers
    )
    assert disabled.status_code == 200, disabled.text
    assert disabled.json()["status"] == "disabled"

    stale_token = client.get("/auth/me", headers=bob_headers)
    assert stale_token.status_code == 401


def test_enterprise_api_key_and_protected_whitelist_do_not_bypass(monkeypatch, tmp_path):
    args = _enterprise_args()
    _patch_enterprise_args(monkeypatch, args, whitelist_patterns=[("/api", True)])

    app = FastAPI()
    app.state.enterprise_enabled = True
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata.sqlite3")
    audit_service = AuditService(metadata_store)
    app.state.enterprise_user_service = UserService(metadata_store, audit_service)
    app.state.enterprise_settings_service = SystemSettingsService(metadata_store)
    app.state.enterprise_authorization_service = AuthorizationService(
        metadata_store, audit_service
    )
    app.state.enterprise_audit_service = audit_service
    combined_auth = get_combined_auth_dependency(api_key=_API_KEY)

    @app.get("/api/tags", dependencies=[Depends(combined_auth)])
    async def tags():
        return {"models": []}

    @app.post("/query", dependencies=[Depends(combined_auth)])
    async def legacy_query():
        return {"response": "legacy"}

    client = TestClient(app)
    response = client.get("/api/tags", headers={"X-API-Key": _API_KEY})
    assert response.status_code == 403
    assert response.json()["detail"] == "API key is disabled in enterprise mode"

    async def seed_admin():
        await metadata_store.initialize()
        user_service = app.state.enterprise_user_service
        return await user_service.bootstrap_super_admin(
            username="admin", password="admin-pass", password_hash=None
        )

    admin = asyncio.run(seed_admin())
    admin_headers = {
        "Authorization": "Bearer " + _token(app.state.enterprise_user_service, admin)
    }
    legacy = client.post("/query", headers=admin_headers)
    assert legacy.status_code == 403
    assert legacy.json()["detail"] == "Legacy global route disabled in enterprise mode"
