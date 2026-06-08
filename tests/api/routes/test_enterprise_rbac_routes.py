from __future__ import annotations

# ruff: noqa: E402 - LightRAG API modules parse sys.argv at import time.

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from lightrag.api.auth import auth_handler
from lightrag.api.config_version_service import ConfigVersionService
from lightrag.api.document_lifecycle_service import DocumentLifecycleService
from lightrag.api.enterprise_auth import (
    AuditService,
    AuthorizationService,
    EnterpriseLimitService,
    InvitationService,
    ServiceAPIKeyService,
    SystemSettingsService,
    UserService,
    service_api_key_is_expired,
)
from lightrag.api.job_service import JobService
from lightrag.api.kb_service import KnowledgeBaseService
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry, LightRAGLike
from lightrag.api.metadata_store import SQLiteMetadataStore
from lightrag.api.routers.enterprise_routes import create_enterprise_routes
from lightrag.api.routers.kb_document_routes import create_kb_document_routes
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

    async def parse_native(self, doc_id: str, file_path: str, content_data):
        return await self._parse(doc_id, file_path, content_data)

    async def parse_mineru(self, doc_id: str, file_path: str, content_data):
        return await self._parse(doc_id, file_path, content_data)

    async def parse_docling(self, doc_id: str, file_path: str, content_data):
        return await self._parse(doc_id, file_path, content_data)

    async def _parse(self, doc_id: str, file_path: str, content_data):
        source_path = Path(file_path)
        parsed_dir = source_path.parent / "__parsed__" / f"{source_path.name}.parsed"
        parsed_dir.mkdir(parents=True, exist_ok=True)
        blocks_path = parsed_dir / f"{source_path.stem}.blocks.jsonl"
        blocks_path.write_text(
            '{"type":"content","text":"parsed"}\n', encoding="utf-8"
        )
        (parsed_dir / "full.md").write_text("# parsed", encoding="utf-8")
        if content_data.get("archive_source_after_parse", True):
            source_path.unlink()
        return {
            "doc_id": doc_id,
            "file_path": file_path,
            "parse_format": "lightrag",
            "content": "parsed",
            "blocks_path": str(blocks_path),
            "parse_stage_skipped": False,
        }

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
        "enterprise_rate_limit_enabled": False,
        "enterprise_rate_limit_requests": 60,
        "enterprise_rate_limit_window_seconds": 60.0,
        "enterprise_tenant_rate_limit_requests": 0,
        "enterprise_tenant_rate_limit_window_seconds": 60.0,
        "enterprise_quota_requests": 0,
        "enterprise_quota_window_seconds": 86400.0,
        "enterprise_tenant_quota_requests": 0,
        "enterprise_tenant_quota_window_seconds": 86400.0,
        "enterprise_artifact_download_min_role": "kb_viewer",
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
    for factory in (
        create_kb_routes,
        create_kb_document_routes,
        create_kb_query_routes,
        create_enterprise_routes,
    ):
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


def _build_enterprise_client(
    monkeypatch,
    tmp_path: Path,
    *,
    api_key: str | None = _API_KEY,
    args: SimpleNamespace | None = None,
):
    args = args or _enterprise_args()
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
    api_key_service = ServiceAPIKeyService(metadata_store, audit_service)
    invitation_service = InvitationService(metadata_store, audit_service)
    limit_service = EnterpriseLimitService(audit_service)
    authz_service = AuthorizationService(metadata_store, audit_service)
    probe = BuilderProbe()
    registry = LightRAGInstanceRegistry(kb_service, probe.build, probe.finalize)
    config_service = ConfigVersionService(kb_service, metadata_store, registry)

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
    app.state.enterprise_api_key_service = api_key_service
    app.state.enterprise_invitation_service = invitation_service
    app.state.enterprise_limit_service = limit_service
    app.state.enterprise_authorization_service = authz_service
    app.state.enterprise_audit_service = audit_service
    app.include_router(
        create_kb_routes(
            kb_service,
            registry,
            api_key=api_key,
            job_service=job_service,
            config_service=config_service,
        )
    )
    app.include_router(
        create_kb_document_routes(
            document_service,
            job_service,
            api_key=api_key,
            registry=registry,
        )
    )
    app.include_router(create_kb_query_routes(document_service, registry, api_key=api_key))
    app.include_router(create_enterprise_routes(api_key=api_key, kb_service=kb_service))
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


def test_enterprise_tenant_kb_acl_authorizes_members_only(monkeypatch, tmp_path):
    client, user_service, _authz, admin, alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}

    async def create_carol():
        return await user_service.create_user(
            username="carol",
            password="carol-pass",
            created_by=admin.id,
        )

    carol = asyncio.run(create_carol())
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}
    carol_headers = {"Authorization": f"Bearer {_token(user_service, carol)}"}

    created = client.post(
        "/kbs",
        json={"id": "kb_tenant_acl", "name": "Tenant ACL"},
        headers=alice_headers,
    )
    assert created.status_code == 200, created.text

    bob_membership = client.put(
        f"/admin/tenants/tenant-a/members/{bob.id}",
        json={"role": "tenant_member"},
        headers=admin_headers,
    )
    assert bob_membership.status_code == 200, bob_membership.text
    assert bob_membership.json()["role"] == "tenant_member"
    carol_membership = client.put(
        f"/admin/tenants/tenant-b/members/{carol.id}",
        json={"role": "tenant_member"},
        headers=admin_headers,
    )
    assert carol_membership.status_code == 200, carol_membership.text

    members = client.get("/admin/tenants/tenant-a/members", headers=admin_headers)
    assert members.status_code == 200, members.text
    assert [item["user_id"] for item in members.json()] == [bob.id]

    tenant_grant = client.put(
        "/admin/kbs/kb_tenant_acl/acl",
        json={"tenant_id": "tenant-a", "role": "kb_viewer"},
        headers=admin_headers,
    )
    assert tenant_grant.status_code == 200, tenant_grant.text
    assert tenant_grant.json()["principal_type"] == "tenant"
    assert tenant_grant.json()["tenant_id"] == "tenant-a"

    bob_list = client.get("/kbs", headers=bob_headers)
    assert bob_list.status_code == 200, bob_list.text
    assert [item["id"] for item in bob_list.json()["knowledge_bases"]] == [
        "kb_tenant_acl"
    ]
    bob_query = client.post(
        "/kbs/kb_tenant_acl/query",
        json={"query": "tenant member", "mode": "mix"},
        headers=bob_headers,
    )
    assert bob_query.status_code == 200, bob_query.text

    carol_query_denied = client.post(
        "/kbs/kb_tenant_acl/query",
        json={"query": "cross tenant", "mode": "mix"},
        headers=carol_headers,
    )
    assert carol_query_denied.status_code == 403

    direct_grant = client.put(
        "/admin/kbs/kb_tenant_acl/acl",
        json={"user_id": carol.id, "role": "kb_viewer"},
        headers=admin_headers,
    )
    assert direct_grant.status_code == 200, direct_grant.text
    assert direct_grant.json()["principal_type"] == "user"
    carol_query_allowed = client.post(
        "/kbs/kb_tenant_acl/query",
        json={"query": "direct grant", "mode": "mix"},
        headers=carol_headers,
    )
    assert carol_query_allowed.status_code == 200, carol_query_allowed.text

    acl = client.get("/admin/kbs/kb_tenant_acl/acl", headers=admin_headers)
    assert acl.status_code == 200, acl.text
    assert {(item["principal_type"], item.get("user_id"), item.get("tenant_id")) for item in acl.json()} == {
        ("user", alice.id, None),
        ("user", carol.id, None),
        ("tenant", None, "tenant-a"),
    }

    revoke_tenant = client.delete(
        "/admin/kbs/kb_tenant_acl/acl/tenants/tenant-a",
        headers=admin_headers,
    )
    assert revoke_tenant.status_code == 200, revoke_tenant.text
    assert revoke_tenant.json() == {"deleted": True}

    audit_events = client.get("/admin/audit-events", headers=admin_headers)
    assert audit_events.status_code == 200, audit_events.text
    event_types = {event["event_type"] for event in audit_events.json()}
    assert {
        "tenant_membership_granted",
        "tenant_kb_acl_granted",
        "tenant_kb_acl_revoked",
    }.issubset(event_types)


def test_enterprise_service_key_tenant_id_does_not_inherit_tenant_acl(
    monkeypatch, tmp_path
):
    client, user_service, _authz, admin, alice, bob, _probe = _build_enterprise_client(
        monkeypatch,
        tmp_path,
        api_key=None,
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}

    created = client.post(
        "/kbs",
        json={"id": "kb_service_tenant_acl", "name": "Service Tenant ACL"},
        headers=alice_headers,
    )
    assert created.status_code == 200, created.text
    membership = client.put(
        f"/admin/tenants/tenant-service/members/{bob.id}",
        json={"role": "tenant_member"},
        headers=admin_headers,
    )
    assert membership.status_code == 200, membership.text
    tenant_grant = client.put(
        "/admin/kbs/kb_service_tenant_acl/acl",
        json={"tenant_id": "tenant-service", "role": "kb_viewer"},
        headers=admin_headers,
    )
    assert tenant_grant.status_code == 200, tenant_grant.text

    user_allowed = client.post(
        "/kbs/kb_service_tenant_acl/query",
        json={"query": "tenant user", "mode": "mix"},
        headers=bob_headers,
    )
    assert user_allowed.status_code == 200, user_allowed.text

    inherited_key = client.post(
        "/admin/service-api-keys",
        json={"name": "tenant-only", "tenant_id": "tenant-service"},
        headers=admin_headers,
    )
    assert inherited_key.status_code == 200, inherited_key.text
    inherited_headers = {"X-API-Key": inherited_key.json()["api_key"]}
    service_list = client.get("/kbs", headers=inherited_headers)
    assert service_list.status_code == 200, service_list.text
    assert service_list.json()["knowledge_bases"] == []
    inherited_query = client.post(
        "/kbs/kb_service_tenant_acl/query",
        json={"query": "must not inherit", "mode": "mix"},
        headers=inherited_headers,
    )
    assert inherited_query.status_code == 403

    explicit_key = client.post(
        "/admin/service-api-keys",
        json={
            "name": "explicit-reader",
            "tenant_id": "tenant-service",
            "kb_roles": {"kb_service_tenant_acl": "kb_viewer"},
        },
        headers=admin_headers,
    )
    assert explicit_key.status_code == 200, explicit_key.text
    explicit_query = client.post(
        "/kbs/kb_service_tenant_acl/query",
        json={"query": "explicit", "mode": "mix"},
        headers={"X-API-Key": explicit_key.json()["api_key"]},
    )
    assert explicit_query.status_code == 200, explicit_query.text


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


def test_enterprise_audit_events_cover_kb_config_query_document_and_artifact(
    monkeypatch, tmp_path
):
    client, user_service, _authz, admin, alice, _bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}

    created = client.post(
        "/kbs",
        json={"id": "kb_audit", "name": "Audit"},
        headers=alice_headers,
    )
    assert created.status_code == 200, created.text

    config_secret = "raw-config-secret"
    config = client.post(
        "/kbs/kb_audit/configs",
        json={"config": {"query": {"system_prompt": config_secret}}},
        headers=alice_headers,
    )
    assert config.status_code == 200, config.text
    activated = client.post(
        f"/kbs/kb_audit/configs/{config.json()['id']}:activate",
        headers=alice_headers,
    )
    assert activated.status_code == 200, activated.text

    upload_secret = b"raw upload secret body"
    uploaded = client.post(
        "/kbs/kb_audit/documents:upload",
        files=[("files", ("audit.pdf", upload_secret, "application/pdf"))],
        headers=alice_headers,
    )
    assert uploaded.status_code == 200, uploaded.text
    document_id = uploaded.json()["documents"][0]["id"]

    process_options = "iF"
    parsed = client.post(
        f"/kbs/kb_audit/documents/{document_id}:parse",
        json={"engine": "mineru", "process_options": process_options},
        headers=alice_headers,
    )
    assert parsed.status_code == 200, parsed.text

    artifacts = client.get(
        f"/kbs/kb_audit/documents/{document_id}/artifacts",
        headers=alice_headers,
    )
    assert artifacts.status_code == 200, artifacts.text
    artifact_items = artifacts.json()["artifacts"]
    assert artifact_items
    downloaded = client.get(
        f"/kbs/kb_audit/documents/{document_id}/artifacts/{artifact_items[0]['id']}:download",
        headers=alice_headers,
    )
    assert downloaded.status_code == 200, downloaded.text

    query_secret = "raw audit secret query"
    queried = client.post(
        "/kbs/kb_audit/query",
        json={"query": query_secret, "mode": "mix"},
        headers=alice_headers,
    )
    assert queried.status_code == 200, queried.text

    audit_response = client.get("/admin/audit-events", headers=admin_headers)
    assert audit_response.status_code == 200, audit_response.text
    events = audit_response.json()
    event_types = {event["event_type"] for event in events}
    assert {
        "kb_created",
        "kb_config_activated",
        "documents_uploaded",
        "document_parse_queued",
        "artifact_downloaded",
        "query_executed",
    }.issubset(event_types)

    query_events = [event for event in events if event["event_type"] == "query_executed"]
    assert query_events
    assert "query_hash" in query_events[0]["metadata"]
    assert "query" not in query_events[0]["metadata"]

    audit_json = json.dumps(events, ensure_ascii=False)
    assert upload_secret.decode("utf-8") not in audit_json
    assert query_secret not in audit_json
    assert config_secret not in audit_json
    assert "source_uri" not in audit_json
    assert "presigned" not in audit_json


def test_enterprise_artifact_download_can_require_stronger_role(monkeypatch, tmp_path):
    args = _enterprise_args(enterprise_artifact_download_min_role="kb_editor")
    client, user_service, _authz, admin, alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path, args=args
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}

    created = client.post(
        "/kbs",
        json={"id": "kb_artifact_policy", "name": "Artifact Policy"},
        headers=alice_headers,
    )
    assert created.status_code == 200, created.text
    grant_viewer = client.put(
        "/admin/kbs/kb_artifact_policy/acl",
        json={"user_id": bob.id, "role": "kb_viewer"},
        headers=admin_headers,
    )
    assert grant_viewer.status_code == 200, grant_viewer.text

    uploaded = client.post(
        "/kbs/kb_artifact_policy/documents:upload",
        files=[("files", ("artifact.pdf", b"pdf", "application/pdf"))],
        headers=alice_headers,
    )
    assert uploaded.status_code == 200, uploaded.text
    document_id = uploaded.json()["documents"][0]["id"]
    parsed = client.post(
        f"/kbs/kb_artifact_policy/documents/{document_id}:parse",
        json={"engine": "mineru", "process_options": "iF"},
        headers=alice_headers,
    )
    assert parsed.status_code == 200, parsed.text

    artifacts = client.get(
        f"/kbs/kb_artifact_policy/documents/{document_id}/artifacts",
        headers=bob_headers,
    )
    assert artifacts.status_code == 200, artifacts.text
    artifact_items = artifacts.json()["artifacts"]
    previewable = next(
        item
        for item in artifact_items
        if item["artifact_type"] in {"blocks", "markdown"}
    )
    artifact_id = previewable["id"]

    detail = client.get(
        f"/kbs/kb_artifact_policy/documents/{document_id}/artifacts/{artifact_id}",
        headers=bob_headers,
    )
    assert detail.status_code == 200, detail.text
    preview = client.get(
        f"/kbs/kb_artifact_policy/documents/{document_id}/artifacts/{artifact_id}:preview",
        headers=bob_headers,
    )
    assert preview.status_code == 200, preview.text

    denied_download = client.get(
        f"/kbs/kb_artifact_policy/documents/{document_id}/artifacts/{artifact_id}:download",
        headers=bob_headers,
    )
    assert denied_download.status_code == 403
    denied_presign = client.get(
        f"/kbs/kb_artifact_policy/documents/{document_id}/artifacts/{artifact_id}:download-url",
        headers=bob_headers,
    )
    assert denied_presign.status_code == 403

    grant_editor = client.put(
        "/admin/kbs/kb_artifact_policy/acl",
        json={"user_id": bob.id, "role": "kb_editor"},
        headers=admin_headers,
    )
    assert grant_editor.status_code == 200, grant_editor.text
    allowed_download = client.get(
        f"/kbs/kb_artifact_policy/documents/{document_id}/artifacts/{artifact_id}:download",
        headers=bob_headers,
    )
    assert allowed_download.status_code == 200, allowed_download.text


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
    assert toggle.json() == {"enabled": True, "mode": "open"}

    setting = client.get("/admin/settings/registration", headers=admin_headers)
    assert setting.status_code == 200, setting.text
    assert setting.json() == {"enabled": True, "mode": "open"}

    invite_only = client.patch(
        "/admin/settings/registration",
        json={"mode": "invite_only"},
        headers=admin_headers,
    )
    assert invite_only.status_code == 200, invite_only.text
    assert invite_only.json() == {"enabled": False, "mode": "invite_only"}

    invite_only_denied = client.post(
        "/auth/register", json={"username": "invite", "password": "invite-pass"}
    )
    assert invite_only_denied.status_code == 403

    reopened = client.patch(
        "/admin/settings/registration",
        json={"mode": "open"},
        headers=admin_headers,
    )
    assert reopened.status_code == 200, reopened.text
    assert reopened.json() == {"enabled": True, "mode": "open"}

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


def test_enterprise_admin_user_lifecycle_and_acl_batch(monkeypatch, tmp_path):
    client, user_service, _authz, admin, alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}

    detail = client.get(f"/admin/users/{bob.id}", headers=admin_headers)
    assert detail.status_code == 200, detail.text
    assert detail.json()["id"] == bob.id

    invalid_status = client.patch(
        f"/admin/users/{bob.id}", json={"status": "locked"}, headers=admin_headers
    )
    assert invalid_status.status_code == 400

    disabled = client.post(f"/admin/users/{bob.id}:disable", headers=admin_headers)
    assert disabled.status_code == 200, disabled.text
    assert disabled.json()["status"] == "disabled"
    assert client.get("/auth/me", headers=bob_headers).status_code == 401

    enabled = client.post(f"/admin/users/{bob.id}:enable", headers=admin_headers)
    assert enabled.status_code == 200, enabled.text
    assert enabled.json()["status"] == "active"

    reset = client.post(
        f"/admin/users/{bob.id}:reset-password",
        json={"password": "bob-new-pass"},
        headers=admin_headers,
    )
    assert reset.status_code == 200, reset.text
    assert reset.json()["token_version"] > bob.token_version
    assert asyncio.run(user_service.authenticate("bob", "bob-pass")) is None
    assert asyncio.run(user_service.authenticate("bob", "bob-new-pass")) is not None

    missing_acl = client.put(
        "/admin/kbs/missing/acl",
        json={"user_id": bob.id, "role": "kb_viewer"},
        headers=admin_headers,
    )
    assert missing_acl.status_code == 404

    created = client.post(
        "/kbs",
        json={"id": "kb_acl_batch", "name": "ACL Batch"},
        headers=alice_headers,
    )
    assert created.status_code == 200, created.text

    batch = client.post(
        "/admin/kbs/kb_acl_batch/acl:batch-set",
        json={
            "entries": [
                {"user_id": bob.id, "role": "kb_editor"},
                {"tenant_id": "tenant-batch", "role": "kb_viewer"},
                {"user_id": alice.id, "action": "revoke"},
            ]
        },
        headers=admin_headers,
    )
    assert batch.status_code == 200, batch.text
    body = batch.json()
    assert [(item["principal_type"], item.get("user_id"), item.get("tenant_id")) for item in body["granted"]] == [
        ("user", bob.id, None),
        ("tenant", None, "tenant-batch"),
    ]
    assert body["granted"][0]["role"] == "kb_editor"
    assert body["granted"][1]["role"] == "kb_viewer"
    assert body["revoked"] == [alice.id]

    acl = client.get("/admin/kbs/kb_acl_batch/acl", headers=admin_headers)
    assert acl.status_code == 200, acl.text
    assert [(item["principal_type"], item.get("user_id"), item.get("tenant_id"), item["role"]) for item in acl.json()] == [
        ("user", bob.id, None, "kb_editor"),
        ("tenant", None, "tenant-batch", "kb_viewer"),
    ]


def test_enterprise_service_api_keys_are_scoped_and_revocable(monkeypatch, tmp_path):
    client, user_service, _authz, admin, alice, _bob, probe = _build_enterprise_client(
        monkeypatch,
        tmp_path,
        api_key=None,
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}

    alpha = client.post(
        "/kbs",
        json={"id": "kb_service_alpha", "name": "Service Alpha"},
        headers=alice_headers,
    )
    beta = client.post(
        "/kbs",
        json={"id": "kb_service_beta", "name": "Service Beta"},
        headers=alice_headers,
    )
    assert alpha.status_code == 200, alpha.text
    assert beta.status_code == 200, beta.text

    non_admin = client.get("/admin/service-api-keys", headers=alice_headers)
    assert non_admin.status_code == 403
    assert non_admin.json()["detail"] == "Super admin permission required"

    created = client.post(
        "/admin/service-api-keys",
        json={
            "name": "alpha-reader",
            "kb_roles": {"kb_service_alpha": "kb_viewer"},
        },
        headers=admin_headers,
    )
    assert created.status_code == 200, created.text
    created_body = created.json()
    raw_key = created_body["api_key"]
    key_record = created_body["key"]
    assert raw_key.startswith("lrsk_svc_key_")
    assert key_record["key_preview"] == raw_key[-6:]
    assert key_record["status"] == "active"
    assert "key_hash" not in key_record

    listed = client.get("/admin/service-api-keys", headers=admin_headers)
    assert listed.status_code == 200, listed.text
    assert listed.json()[0]["id"] == key_record["id"]
    assert "api_key" not in listed.json()[0]
    assert "key_hash" not in listed.json()[0]

    service_headers = {"X-API-Key": raw_key}
    service_list = client.get("/kbs", headers=service_headers)
    assert service_list.status_code == 200, service_list.text
    assert [item["id"] for item in service_list.json()["knowledge_bases"]] == [
        "kb_service_alpha"
    ]

    create_denied = client.post(
        "/kbs",
        json={"id": "kb_service_new", "name": "Denied"},
        headers=service_headers,
    )
    assert create_denied.status_code == 403
    assert create_denied.json()["detail"] == "Create-KB permission required"

    update_denied = client.patch(
        "/kbs/kb_service_alpha",
        json={"name": "Denied"},
        headers=service_headers,
    )
    assert update_denied.status_code == 403

    alpha_query = client.post(
        "/kbs/kb_service_alpha/query",
        json={"query": "what is alpha", "mode": "mix"},
        headers=service_headers,
    )
    assert alpha_query.status_code == 200, alpha_query.text
    assert probe.instances["kb_service_alpha"].query_params[-1].mode == "mix"

    beta_query = client.post(
        "/kbs/kb_service_beta/query",
        json={"query": "what is beta", "mode": "mix"},
        headers=service_headers,
    )
    assert beta_query.status_code == 403

    bypass_denied = client.post(
        "/kbs/kb_service_alpha/query",
        json={"query": "raw model", "mode": "bypass"},
        headers=service_headers,
    )
    assert bypass_denied.status_code == 403
    assert bypass_denied.json()["detail"] == "Bypass-query permission required"

    bypass_key = client.post(
        "/admin/service-api-keys",
        json={
            "name": "alpha-bypass",
            "kb_roles": {"kb_service_alpha": "kb_viewer"},
            "can_use_bypass_query": True,
        },
        headers=admin_headers,
    )
    assert bypass_key.status_code == 200, bypass_key.text
    bypass_allowed = client.post(
        "/kbs/kb_service_alpha/query",
        json={"query": "raw model", "mode": "bypass"},
        headers={"X-API-Key": bypass_key.json()["api_key"]},
    )
    assert bypass_allowed.status_code == 200, bypass_allowed.text
    assert probe.instances["kb_service_alpha"].query_params[-1].mode == "bypass"

    bad_bearer_wins = client.post(
        "/kbs/kb_service_alpha/query",
        json={"query": "bad bearer", "mode": "mix"},
        headers={"Authorization": "Bearer invalid", "X-API-Key": raw_key},
    )
    assert bad_bearer_wins.status_code == 401

    revoked = client.post(
        f"/admin/service-api-keys/{key_record['id']}:revoke",
        headers=admin_headers,
    )
    assert revoked.status_code == 200, revoked.text
    assert revoked.json()["status"] == "revoked"
    assert "key_hash" not in revoked.json()

    after_revoke = client.post(
        "/kbs/kb_service_alpha/query",
        json={"query": "after revoke", "mode": "mix"},
        headers=service_headers,
    )
    assert after_revoke.status_code == 401
    assert after_revoke.json()["detail"] == "Enterprise login required"

    audit_events = client.get("/admin/audit-events", headers=admin_headers)
    assert audit_events.status_code == 200
    event_types = [event["event_type"] for event in audit_events.json()]
    assert "service_api_key_created" in event_types
    assert "service_api_key_revoked" in event_types


def test_service_api_key_expiry_rejected_at_auth(tmp_path):
    """An expired service API key is rejected at authentication; a non-expired
    one still resolves to a principal. Built against the real SQLite store +
    ServiceAPIKeyService so it never depends on wall-clock waiting."""
    from datetime import datetime, timedelta, timezone

    store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    service = ServiceAPIKeyService(store, AuditService(store))

    async def scenario():
        await store.initialize()
        past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        expired_record, expired_raw = await service.create_key(
            name="expired", expires_at=past
        )
        valid_record, valid_raw = await service.create_key(
            name="valid", expires_at=future
        )
        return (
            expired_record,
            valid_record,
            await service.principal_from_api_key(expired_raw),
            await service.principal_from_api_key(valid_raw),
        )

    expired_record, valid_record, expired_principal, valid_principal = asyncio.run(
        scenario()
    )

    assert expired_record.expires_at is not None
    assert expired_principal is None
    assert valid_principal is not None
    assert valid_principal.metadata["service_api_key_id"] == valid_record.id
    assert service_api_key_is_expired(expired_record.expires_at) is True
    assert service_api_key_is_expired(valid_record.expires_at) is False
    assert service_api_key_is_expired(None) is False


def test_create_service_api_key_with_expiry_round_trips(monkeypatch, tmp_path):
    """POST /admin/service-api-keys with expires_in_seconds returns expires_at;
    a far-future key still authenticates and stays scoped to its KB; a
    non-positive expires_in_seconds is rejected at the request boundary."""
    client, user_service, _authz, admin, alice, _bob, _probe = _build_enterprise_client(
        monkeypatch,
        tmp_path,
        api_key=None,
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}

    created_kb = client.post(
        "/kbs", json={"id": "kb_expiry", "name": "Expiry"}, headers=alice_headers
    )
    assert created_kb.status_code == 200, created_kb.text

    created = client.post(
        "/admin/service-api-keys",
        json={
            "name": "expiring-reader",
            "kb_roles": {"kb_expiry": "kb_viewer"},
            "expires_in_seconds": 3600,
        },
        headers=admin_headers,
    )
    assert created.status_code == 200, created.text
    key_record = created.json()["key"]
    assert key_record["expires_at"] is not None

    listed = client.get("/admin/service-api-keys", headers=admin_headers)
    assert listed.status_code == 200, listed.text
    assert listed.json()[0]["expires_at"] == key_record["expires_at"]

    service_headers = {"X-API-Key": created.json()["api_key"]}
    service_list = client.get("/kbs", headers=service_headers)
    assert service_list.status_code == 200, service_list.text
    assert [item["id"] for item in service_list.json()["knowledge_bases"]] == [
        "kb_expiry"
    ]

    invalid = client.post(
        "/admin/service-api-keys",
        json={"name": "bad", "expires_in_seconds": 0},
        headers=admin_headers,
    )
    assert invalid.status_code == 422


def test_enterprise_user_rate_limit_returns_429_and_audits(monkeypatch, tmp_path):
    args = _enterprise_args()
    client, user_service, _authz, admin, alice, bob, _probe = _build_enterprise_client(
        monkeypatch,
        tmp_path,
        args=args,
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}

    created = client.post(
        "/kbs",
        json={"id": "kb_user_limit", "name": "User Limit"},
        headers=alice_headers,
    )
    assert created.status_code == 200, created.text
    grant = client.put(
        "/admin/kbs/kb_user_limit/acl",
        json={"user_id": bob.id, "role": "kb_viewer"},
        headers=admin_headers,
    )
    assert grant.status_code == 200, grant.text

    args.enterprise_rate_limit_enabled = True
    args.enterprise_rate_limit_requests = 2
    args.enterprise_rate_limit_window_seconds = 60.0

    for idx in range(2):
        response = client.post(
            "/kbs/kb_user_limit/query",
            json={"query": f"allowed {idx}", "mode": "mix"},
            headers=bob_headers,
        )
        assert response.status_code == 200, response.text

    limited = client.post(
        "/kbs/kb_user_limit/query",
        json={"query": "blocked", "mode": "mix"},
        headers=bob_headers,
    )
    assert limited.status_code == 429
    assert limited.json()["detail"] == "Enterprise rate limit exceeded"
    assert int(limited.headers["Retry-After"]) > 0

    args.enterprise_rate_limit_enabled = False
    audit_events = client.get("/admin/audit-events", headers=admin_headers)
    assert audit_events.status_code == 200, audit_events.text
    limited_events = [
        event for event in audit_events.json() if event["event_type"] == "rate_limited"
    ]
    assert limited_events
    assert limited_events[0]["target_type"] == "user"
    assert limited_events[0]["target_id"] == bob.id
    assert limited_events[0]["metadata"]["path"] == "/kbs/kb_user_limit/query"
    assert "blocked" not in str(limited_events[0]["metadata"])


def test_enterprise_tenant_rate_limit_shares_bucket_across_users(
    monkeypatch, tmp_path
):
    args = _enterprise_args()
    client, user_service, _authz, admin, alice, bob, _probe = _build_enterprise_client(
        monkeypatch,
        tmp_path,
        args=args,
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}

    created = client.post(
        "/kbs",
        json={"id": "kb_tenant_limit", "name": "Tenant Limit"},
        headers=alice_headers,
    )
    assert created.status_code == 200, created.text

    async def seed_tenant_users():
        updated_bob = await user_service.update_user(
            bob.id,
            tenant_id="tenant_shared",
            actor_user_id=admin.id,
        )
        carol = await user_service.create_user(
            username="carol",
            password="carol-pass",
            tenant_id="tenant_shared",
            created_by=admin.id,
        )
        return updated_bob, carol

    bob, carol = asyncio.run(seed_tenant_users())
    for user in (bob, carol):
        grant = client.put(
            "/admin/kbs/kb_tenant_limit/acl",
            json={"user_id": user.id, "role": "kb_viewer"},
            headers=admin_headers,
        )
        assert grant.status_code == 200, grant.text

    args.enterprise_rate_limit_enabled = True
    args.enterprise_rate_limit_requests = 100
    args.enterprise_tenant_rate_limit_requests = 2
    args.enterprise_tenant_rate_limit_window_seconds = 60.0

    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}
    carol_headers = {"Authorization": f"Bearer {_token(user_service, carol)}"}
    first = client.post(
        "/kbs/kb_tenant_limit/query",
        json={"query": "tenant first", "mode": "mix"},
        headers=bob_headers,
    )
    assert first.status_code == 200, first.text
    second = client.post(
        "/kbs/kb_tenant_limit/query",
        json={"query": "tenant second", "mode": "mix"},
        headers=carol_headers,
    )
    assert second.status_code == 200, second.text
    limited = client.post(
        "/kbs/kb_tenant_limit/query",
        json={"query": "tenant blocked", "mode": "mix"},
        headers=bob_headers,
    )
    assert limited.status_code == 429
    assert limited.json()["detail"] == "Enterprise rate limit exceeded"

    args.enterprise_rate_limit_enabled = False
    audit_events = client.get("/admin/audit-events", headers=admin_headers)
    assert audit_events.status_code == 200, audit_events.text
    limited_events = [
        event for event in audit_events.json() if event["event_type"] == "rate_limited"
    ]
    assert any(
        event["target_type"] == "tenant"
        and event["target_id"] == "tenant_shared"
        and event["metadata"]["limit_name"] == "tenant_rate"
        for event in limited_events
    )


def test_enterprise_service_key_quota_returns_429_and_audits(monkeypatch, tmp_path):
    args = _enterprise_args()
    client, user_service, _authz, admin, alice, _bob, _probe = _build_enterprise_client(
        monkeypatch,
        tmp_path,
        api_key=None,
        args=args,
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}

    created_kb = client.post(
        "/kbs",
        json={"id": "kb_quota", "name": "Quota"},
        headers=alice_headers,
    )
    assert created_kb.status_code == 200, created_kb.text
    created_key = client.post(
        "/admin/service-api-keys",
        json={"name": "quota-reader", "kb_roles": {"kb_quota": "kb_viewer"}},
        headers=admin_headers,
    )
    assert created_key.status_code == 200, created_key.text
    raw_key = created_key.json()["api_key"]
    key_id = created_key.json()["key"]["id"]

    args.enterprise_rate_limit_enabled = True
    args.enterprise_rate_limit_requests = 100
    args.enterprise_quota_requests = 1
    args.enterprise_quota_window_seconds = 86400.0

    service_headers = {"X-API-Key": raw_key}
    first = client.post(
        "/kbs/kb_quota/query",
        json={"query": "quota first", "mode": "mix"},
        headers=service_headers,
    )
    assert first.status_code == 200, first.text
    limited = client.post(
        "/kbs/kb_quota/query",
        json={"query": "quota blocked", "mode": "mix"},
        headers=service_headers,
    )
    assert limited.status_code == 429
    assert limited.json()["detail"] == "Enterprise quota exceeded"

    args.enterprise_rate_limit_enabled = False
    audit_events = client.get("/admin/audit-events", headers=admin_headers)
    assert audit_events.status_code == 200, audit_events.text
    limited_events = [
        event for event in audit_events.json() if event["event_type"] == "quota_exceeded"
    ]
    assert limited_events
    assert limited_events[0]["target_type"] == "service_api_key"
    assert limited_events[0]["target_id"] == key_id
    assert limited_events[0]["metadata"]["auth_method"] == "service_api_key"
    assert "quota blocked" not in str(limited_events[0]["metadata"])


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


def test_registration_admin_approval_creates_pending_user_then_approves(
    monkeypatch, tmp_path
):
    client, user_service, _authz, admin, _alice, _bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}

    set_mode = client.patch(
        "/admin/settings/registration",
        json={"mode": "admin_approval"},
        headers=admin_headers,
    )
    assert set_mode.status_code == 200, set_mode.text
    assert set_mode.json() == {"enabled": False, "mode": "admin_approval"}

    reg = client.post(
        "/auth/register", json={"username": "carol", "password": "carol-pass"}
    )
    assert reg.status_code == 200, reg.text
    body = reg.json()
    assert body["status"] == "pending"
    assert "access_token" not in body
    assert body["user"]["status"] == "pending"
    user_id = body["user"]["id"]

    # A pending user cannot authenticate yet.
    assert asyncio.run(user_service.authenticate("carol", "carol-pass")) is None

    approved = client.post(f"/admin/users/{user_id}:enable", headers=admin_headers)
    assert approved.status_code == 200, approved.text
    assert approved.json()["status"] == "active"

    # After approval the user can authenticate.
    assert asyncio.run(user_service.authenticate("carol", "carol-pass")) is not None


def test_registration_invite_only_requires_valid_single_use_token(
    monkeypatch, tmp_path
):
    client, user_service, _authz, admin, _alice, _bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}

    set_mode = client.patch(
        "/admin/settings/registration",
        json={"mode": "invite_only"},
        headers=admin_headers,
    )
    assert set_mode.status_code == 200, set_mode.text

    # No token / bad token are both rejected.
    assert (
        client.post(
            "/auth/register", json={"username": "dave", "password": "dave-pass"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/auth/register",
            json={
                "username": "dave",
                "password": "dave-pass",
                "invitation_token": "lrinv_not_a_real_token",
            },
        ).status_code
        == 403
    )

    minted = client.post("/admin/invitations", json={}, headers=admin_headers)
    assert minted.status_code == 200, minted.text
    raw_token = minted.json()["invitation_token"]
    assert raw_token.startswith("lrinv_")
    assert "token_hash" not in minted.json()["invitation"]

    ok = client.post(
        "/auth/register",
        json={
            "username": "dave",
            "password": "dave-pass",
            "invitation_token": raw_token,
        },
    )
    assert ok.status_code == 200, ok.text
    assert "access_token" in ok.json()
    assert ok.json()["user"]["status"] == "active"

    # Single-use: the same token cannot be reused.
    reused = client.post(
        "/auth/register",
        json={
            "username": "erin",
            "password": "erin-pass",
            "invitation_token": raw_token,
        },
    )
    assert reused.status_code == 403

    listed = client.get("/admin/invitations", headers=admin_headers)
    assert listed.status_code == 200, listed.text
    assert listed.json()[0]["status"] == "used"

    # A revoked invitation is also rejected.
    minted2 = client.post("/admin/invitations", json={}, headers=admin_headers)
    token2 = minted2.json()["invitation_token"]
    inv2_id = minted2.json()["invitation"]["id"]
    revoke = client.post(
        f"/admin/invitations/{inv2_id}:revoke", headers=admin_headers
    )
    assert revoke.status_code == 200, revoke.text
    assert revoke.json()["status"] == "revoked"
    after_revoke = client.post(
        "/auth/register",
        json={
            "username": "frank",
            "password": "frank-pass",
            "invitation_token": token2,
        },
    )
    assert after_revoke.status_code == 403


def test_invitation_expiry_rejected_at_consume(tmp_path):
    """An expired invitation is rejected by consume_invitation (403). Built
    against the real SQLite store + InvitationService without waiting."""
    from datetime import datetime, timedelta, timezone

    store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    service = InvitationService(store, AuditService(store))

    async def scenario():
        await store.initialize()
        past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        record, raw_token = await service.create_invitation(expires_at=past)
        return record, raw_token

    record, raw_token = asyncio.run(scenario())
    assert record.status == "active"
    with pytest.raises(HTTPException) as exc:
        asyncio.run(service.consume_invitation(raw_token, used_by="someone"))
    assert exc.value.status_code == 403


def test_concurrent_job_quota_blocks_excess_jobs(monkeypatch, tmp_path):
    """JobService rejects (429) a new job once the principal already holds the
    configured number of in-flight jobs; a different principal is unaffected.
    Driven directly against JobService + the real SQLite store."""
    from lightrag.api import config as api_config
    from lightrag.api.enterprise_auth import (
        Principal,
        SYSTEM_ROLE_USER,
        USER_STATUS_ACTIVE,
        set_current_principal,
    )

    args = _enterprise_args(enterprise_max_concurrent_jobs=1)
    monkeypatch.setattr(api_config, "global_args", args)

    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kb.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    job_service = JobService(kb_service, metadata_store)

    def _principal(user_id: str) -> Principal:
        return Principal(
            user_id=user_id,
            username=user_id,
            system_role=SYSTEM_ROLE_USER,
            status=USER_STATUS_ACTIVE,
            tenant_id=None,
            tenant_roles={},
            can_create_kb=True,
            can_use_bypass_query=False,
            token_version=1,
            auth_method="jwt",
            metadata={},
        )

    async def setup():
        await kb_service.initialize()
        await metadata_store.initialize()
        await kb_service.create(name="Quota KB", kb_id="kb_quota")

    asyncio.run(setup())

    async def create_as(principal: Principal):
        set_current_principal(principal)
        try:
            return await job_service.create_job("kb_quota", job_type="parse")
        finally:
            set_current_principal(None)

    first = asyncio.run(create_as(_principal("usr_alice")))
    assert first.status == "queued"
    assert first.payload["_principal"]["subject_id"] == "usr_alice"

    with pytest.raises(HTTPException) as exc:
        asyncio.run(create_as(_principal("usr_alice")))
    assert exc.value.status_code == 429

    # A different principal is unaffected by alice's in-flight jobs.
    bob_job = asyncio.run(create_as(_principal("usr_bob")))
    assert bob_job.status == "queued"
