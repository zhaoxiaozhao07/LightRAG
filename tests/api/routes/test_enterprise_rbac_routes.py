from __future__ import annotations

# ruff: noqa: E402 - LightRAG API modules parse sys.argv at import time.

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

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
    LoginAttemptTracker,
    ServiceAPIKeyService,
    SystemSettingsService,
    UserKBQuerySettingsService,
    UserService,
    service_api_key_is_expired,
)
from lightrag.api.job_service import JobService
from lightrag.api.kb_operation_fence import KBWriteAdmissionMiddleware
from lightrag.api.kb_deletion_service import KBDeletionService
from lightrag.api.kb_service import is_tenant_owned_kb, KnowledgeBaseService
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry, LightRAGLike
from lightrag.api.metadata_store import KBLifecycleConflictError, SQLiteMetadataStore
from lightrag.api.routers.enterprise_routes import create_enterprise_routes
from lightrag.api.routers.kb_document_routes import create_kb_document_routes
from lightrag.api.routers.kb_graph_routes import create_kb_graph_routes
from lightrag.api.routers.kb_query_routes import create_kb_query_routes
from lightrag.api.routers.kb_routes import create_kb_routes
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.base import QueryParam
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
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

    async def parse_legacy(self, doc_id: str, file_path: str, content_data):
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

    async def get_graph_labels(self) -> list[str]:
        return []

    async def aedit_entity(
        self,
        entity_name: str,
        updated_data: dict,
        allow_rename: bool = True,
        allow_merge: bool = False,
    ) -> dict:
        return {"entity_name": updated_data.get("entity_name", entity_name)}

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
        "enterprise_artifact_download_policy": "",
        "enterprise_artifact_action_policy": "",
        "enterprise_mask_storage_uris": True,
        "enterprise_registration_max_attempts": 10,
        "enterprise_registration_window_seconds": 300.0,
        "enterprise_registration_lockout_seconds": 900.0,
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


def _grant_file_download(
    user_service: UserService, user, *, actor_user_id: str
):
    return asyncio.run(
        user_service.update_user(
            user.id,
            can_download_files=True,
            actor_user_id=actor_user_id,
        )
    )


def _build_enterprise_client(
    monkeypatch,
    tmp_path: Path,
    *,
    api_key: str | None = _API_KEY,
    args: SimpleNamespace | None = None,
    inject_enterprise_router_kb_service: bool = True,
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
    user_kb_query_settings_service = UserKBQuerySettingsService(
        metadata_store, audit_service
    )
    api_key_service = ServiceAPIKeyService(
        metadata_store, audit_service, kb_service=kb_service
    )
    invitation_service = InvitationService(metadata_store, audit_service)
    limit_service = EnterpriseLimitService(audit_service)
    authz_service = AuthorizationService(
        metadata_store, audit_service, kb_service=kb_service
    )
    probe = BuilderProbe()
    registry = LightRAGInstanceRegistry(kb_service, probe.build, probe.finalize)
    config_service = ConfigVersionService(kb_service, metadata_store, registry)
    deletion_service = KBDeletionService(
        kb_service,
        metadata_store,
        registry,
        input_root=tmp_path / "inputs",
        working_dir=tmp_path / "working",
    )

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
    app.state.kb_service = kb_service
    app.state.metadata_store = metadata_store
    app.state.enterprise_user_service = user_service
    app.state.enterprise_settings_service = settings_service
    app.state.enterprise_user_kb_query_settings_service = user_kb_query_settings_service
    app.state.enterprise_api_key_service = api_key_service
    app.state.enterprise_invitation_service = invitation_service
    app.state.enterprise_limit_service = limit_service
    app.state.enterprise_authorization_service = authz_service
    app.state.enterprise_audit_service = audit_service
    app.add_middleware(
        KBWriteAdmissionMiddleware,
        kb_service=kb_service,
        metadata_store=metadata_store,
    )
    app.include_router(
        create_kb_routes(
            kb_service,
            registry,
            api_key=api_key,
            job_service=job_service,
            config_service=config_service,
            deletion_service=deletion_service,
            metadata_store=metadata_store,
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
    app.include_router(create_kb_graph_routes(registry, api_key=api_key))
    app.include_router(
        create_enterprise_routes(
            api_key=api_key,
            kb_service=kb_service
            if inject_enterprise_router_kb_service
            else None,
        )
    )
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
            "metadata": {
                "tenant_managed": True,
                "tenant_tag": "tenant:spoofed-tenant",
                "tags": ["tenant:spoofed-tenant"],
            },
        },
        headers=alice_headers,
    )
    assert created.status_code == 200, created.text
    created_body = created.json()
    assert created_body["owner_id"] == alice.id
    assert created_body["tenant_id"] is None
    assert created_body["origin"] == "platform"
    assert created_body["metadata"] == {"tags": ["tenant:spoofed-tenant"]}

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
        json={
            "owner_id": "evil-owner",
            "tenant_id": "evil-tenant",
            "origin": "tenant",
            "name": "Safe",
        },
        headers=alice_headers,
    )
    assert spoof_update.status_code == 200, spoof_update.text
    assert spoof_update.json()["name"] == "Safe"
    assert spoof_update.json()["owner_id"] == alice.id
    assert spoof_update.json()["tenant_id"] is None
    assert spoof_update.json()["origin"] == "platform"

    owner_delete_denied = client.delete("/kbs/kb_alpha", headers=alice_headers)
    assert owner_delete_denied.status_code == 403

    admin_delete = client.delete("/kbs/kb_alpha", headers=admin_headers)
    assert admin_delete.status_code == 200, admin_delete.text
    assert admin_delete.json()["status"] == "deleted"

    audit_events = client.get("/admin/audit-events", headers=admin_headers)
    assert audit_events.status_code == 200
    assert any(event["event_type"] == "kb_deleted" for event in audit_events.json())


def test_enterprise_tenant_admin_create_policy_and_generation(monkeypatch, tmp_path):
    from lightrag.api.enterprise_auth import Principal

    client, user_service, authz, admin, alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    asyncio.run(
        authz.grant_tenant_membership(
            "tenant-a",
            alice.id,
            "tenant_admin",
            granted_by=admin.id,
        )
    )
    asyncio.run(
        authz.grant_tenant_membership(
            "tenant-a",
            bob.id,
            "tenant_member",
            granted_by=admin.id,
        )
    )
    alice = asyncio.run(
        user_service.update_user(
            alice.id,
            can_create_kb=False,
            actor_user_id=admin.id,
        )
    )
    assert alice.can_create_kb is False
    bob = asyncio.run(user_service.get_user_or_404(bob.id))
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}

    created = client.post(
        "/kbs",
        json={
            "id": "kb_tenant_created",
            "name": "Tenant Created",
            "owner_id": bob.id,
            "tenant_id": "spoofed-tenant",
            "metadata": {
                "tags": [
                    "custom",
                    "tenant:tenant-a",
                    "custom",
                    "tenant:tenant-a",
                    "tenant:foreign",
                    "other",
                    "other",
                ],
                "team": "legal",
                "platform_provisioned": True,
            },
        },
        headers=alice_headers,
    )
    assert created.status_code == 200, created.text
    payload = created.json()
    assert payload["owner_id"] == alice.id
    assert payload["tenant_id"] == "tenant-a"
    assert payload["visibility"] == "private"
    assert payload["origin"] == "tenant"
    assert payload["metadata"] == {
        "tags": ["custom", "other", "tenant:tenant-a"],
        "team": "legal",
    }

    store = client.app.state.metadata_store
    lifecycle = asyncio.run(store.get_kb_lifecycle("kb_tenant_created"))
    assert lifecycle is not None
    assert lifecycle.state == "active"
    owner_acl = asyncio.run(store.list_kb_acl("kb_tenant_created"))
    assert [(item.user_id, item.role) for item in owner_acl] == [
        (alice.id, "kb_owner")
    ]

    non_private = client.post(
        "/kbs",
        json={
            "id": "kb_tenant_internal",
            "name": "Not Private",
            "visibility": "internal",
        },
        headers=alice_headers,
    )
    assert non_private.status_code == 400
    invalid_tags = client.post(
        "/kbs",
        json={
            "id": "kb_bad_tags",
            "name": "Bad Tags",
            "metadata": {"tags": "not-a-list"},
        },
        headers=alice_headers,
    )
    assert invalid_tags.status_code == 400

    for visibility in ("private", "public", "internal"):
        blocked_patch = client.patch(
            "/kbs/kb_tenant_created",
            json={"visibility": visibility},
            headers=alice_headers,
        )
        assert blocked_patch.status_code == 403
    safe_patch = client.patch(
        "/kbs/kb_tenant_created",
        json={"name": "Tenant Created Renamed"},
        headers=alice_headers,
    )
    assert safe_patch.status_code == 200, safe_patch.text
    assert safe_patch.json()["visibility"] == "private"
    assert safe_patch.json()["origin"] == "tenant"
    for reserved_key in (
        "platform_provisioned",
        "tenant_managed",
        "tenant_tag",
    ):
        reserved_patch = client.patch(
            "/kbs/kb_tenant_created",
            json={"metadata": {reserved_key: True}},
            headers=alice_headers,
        )
        assert reserved_patch.status_code == 400

    member_denied = client.post(
        "/kbs",
        json={"id": "kb_member_denied", "name": "Member Denied"},
        headers=bob_headers,
    )
    assert member_denied.status_code == 403

    provisioned = client.post(
        "/kbs",
        json={
            "id": "kb_platform_tenant",
            "name": "Platform Tenant",
            "owner_id": bob.id,
            "tenant_id": "tenant-a",
            "visibility": "internal",
            "metadata": {"team": "platform"},
        },
        headers=admin_headers,
    )
    assert provisioned.status_code == 200, provisioned.text
    provisioned_payload = provisioned.json()
    assert provisioned_payload["owner_id"] == bob.id
    assert provisioned_payload["tenant_id"] == "tenant-a"
    assert provisioned_payload["visibility"] == "internal"
    assert provisioned_payload["origin"] == "platform"
    assert provisioned_payload["metadata"] == {"team": "platform"}
    platform_acl = asyncio.run(store.list_kb_acl("kb_platform_tenant"))
    assert [(item.user_id, item.role) for item in platform_acl] == [
        (bob.id, "kb_owner")
    ]

    tenant_admin_without_primary = Principal(
        user_id="orphan-admin",
        username="orphan-admin",
        system_role="user",
        status="active",
        tenant_id=None,
        tenant_roles={"tenant-a": "tenant_admin"},
        can_create_kb=True,
        can_use_bypass_query=False,
        token_version=1,
        auth_method="jwt",
        metadata={},
    )
    with pytest.raises(HTTPException) as exc:
        authz.require_create_kb(tenant_admin_without_primary)
    assert exc.value.status_code == 403


def test_enterprise_kb_lifecycle_uses_tenant_provenance_not_acl(
    monkeypatch, tmp_path
):
    client, user_service, authz, admin, alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path, api_key=None
    )
    asyncio.run(
        authz.grant_tenant_membership(
            "tenant-a",
            alice.id,
            "tenant_admin",
            granted_by=admin.id,
        )
    )
    asyncio.run(
        authz.grant_tenant_membership(
            "tenant-b",
            bob.id,
            "tenant_admin",
            granted_by=admin.id,
        )
    )
    alice = asyncio.run(user_service.get_user_or_404(alice.id))
    bob = asyncio.run(user_service.get_user_or_404(bob.id))
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}

    def create_as(headers, kb_id: str, **extra):
        body = {"id": kb_id, "name": kb_id}
        body.update(extra)
        response = client.post("/kbs", json=body, headers=headers)
        assert response.status_code == 200, response.text
        return response

    tenant_soft = create_as(
        alice_headers,
        "kb_tenant_soft",
        metadata={"platform_provisioned": True},
    )
    assert tenant_soft.json()["origin"] == "tenant"
    assert "platform_provisioned" not in tenant_soft.json()["metadata"]
    cross_tenant = client.delete("/kbs/kb_tenant_soft", headers=bob_headers)
    assert cross_tenant.status_code == 403
    soft_deleted = client.delete("/kbs/kb_tenant_soft", headers=alice_headers)
    assert soft_deleted.status_code == 200, soft_deleted.text
    restored = client.post("/kbs/kb_tenant_soft:restore", headers=alice_headers)
    assert restored.status_code == 200, restored.text
    assert restored.json()["status"] == "active"

    create_as(alice_headers, "kb_tenant_hard")
    hard_deleted = client.delete(
        "/kbs/kb_tenant_hard?hard=true",
        headers=alice_headers,
    )
    assert hard_deleted.status_code == 200, hard_deleted.text
    assert hard_deleted.json()["status"] == "deleted"

    create_as(
        admin_headers,
        "kb_platform_provisioned",
        tenant_id="tenant-a",
        metadata={
            "tenant_managed": True,
            "tenant_tag": "tenant:tenant-a",
            "tags": ["tenant:tenant-a"],
        },
    )
    platform_record = asyncio.run(
        client.app.state.kb_service.get("kb_platform_provisioned")
    )
    assert platform_record.origin == "platform"
    assert not is_tenant_owned_kb(platform_record, "tenant-a")
    assert (
        client.delete("/kbs/kb_platform_provisioned", headers=alice_headers).status_code
        == 403
    )
    platform_deleted = client.delete(
        "/kbs/kb_platform_provisioned", headers=admin_headers
    )
    assert platform_deleted.status_code == 200, platform_deleted.text
    assert (
        client.post(
            "/kbs/kb_platform_provisioned:restore", headers=alice_headers
        ).status_code
        == 403
    )
    platform_restored = client.post(
        "/kbs/kb_platform_provisioned:restore", headers=admin_headers
    )
    assert platform_restored.status_code == 200, platform_restored.text

    legacy = create_as(
        admin_headers,
        "kb_legacy_spoofed_origin",
        tenant_id="tenant-a",
    )
    catalog_path = client.app.state.kb_service.metadata_path
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    legacy_row = catalog["knowledge_bases"][legacy.json()["id"]]
    legacy_row.pop("origin")
    legacy_row["metadata"] = {
        "tenant_managed": True,
        "platform_provisioned": True,
        "tenant_tag": "tenant:tenant-a",
    }
    catalog_path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    loaded_legacy = asyncio.run(
        client.app.state.kb_service.get("kb_legacy_spoofed_origin")
    )
    assert loaded_legacy.origin == "platform"
    assert loaded_legacy.metadata["tenant_managed"] is True
    assert (
        client.delete(
            "/kbs/kb_legacy_spoofed_origin",
            headers=alice_headers,
        ).status_code
        == 403
    )

    create_as(admin_headers, "kb_tenant_acl_only")
    tenant_acl = client.put(
        "/admin/kbs/kb_tenant_acl_only/acl",
        json={"tenant_id": "tenant-a", "role": "kb_owner"},
        headers=admin_headers,
    )
    assert tenant_acl.status_code == 200, tenant_acl.text
    assert (
        client.delete("/kbs/kb_tenant_acl_only", headers=alice_headers).status_code
        == 403
    )

    create_as(
        admin_headers,
        "kb_direct_owner_only",
        owner_id=alice.id,
    )
    direct_acl = asyncio.run(
        client.app.state.metadata_store.get_kb_acl_role(
            "kb_direct_owner_only", alice.id
        )
    )
    assert direct_acl == "kb_owner"
    assert (
        client.delete("/kbs/kb_direct_owner_only", headers=alice_headers).status_code
        == 403
    )

    create_as(
        admin_headers,
        "kb_public_only",
        visibility="public",
    )
    assert client.get("/kbs/kb_public_only", headers=alice_headers).status_code == 200
    assert (
        client.delete("/kbs/kb_public_only", headers=alice_headers).status_code
        == 403
    )
    super_delete = client.delete("/kbs/kb_public_only", headers=admin_headers)
    assert super_delete.status_code == 200, super_delete.text

    service_key = client.post(
        "/admin/service-api-keys",
        json={
            "name": "lifecycle-denied",
            "kb_roles": {"kb_direct_owner_only": "kb_owner"},
        },
        headers=admin_headers,
    )
    assert service_key.status_code == 200, service_key.text
    service_headers = {"X-API-Key": service_key.json()["api_key"]}
    service_denied = client.delete(
        "/kbs/kb_direct_owner_only",
        headers=service_headers,
    )
    assert service_denied.status_code == 403

    lifecycle_events = asyncio.run(
        client.app.state.metadata_store.list_audit_events(
            target_type="kb",
            target_id="kb_tenant_soft",
        )
    )
    successful_lifecycle_events = [
        event
        for event in lifecycle_events
        if event.event_type in {"kb_deleted", "kb_restored"}
    ]
    assert {event.event_type for event in successful_lifecycle_events} == {
        "kb_deleted",
        "kb_restored",
    }
    assert all(
        event.actor_tenant_id == "tenant-a"
        for event in successful_lifecycle_events
    )


def test_enterprise_user_kb_query_settings_are_per_user_and_used_by_query(
    monkeypatch, tmp_path
):
    client, user_service, _authz, admin, alice, bob, probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}
    probe.active_query_config = {"user_prompt": "kb default prompt"}

    created = client.post(
        "/kbs",
        json={"id": "kb_prompt", "name": "Prompt KB"},
        headers=alice_headers,
    )
    assert created.status_code == 200, created.text

    default_settings = client.get(
        "/auth/me/kbs/kb_prompt/query-settings",
        headers=alice_headers,
    )
    assert default_settings.status_code == 200, default_settings.text
    assert default_settings.json() == {
        "user_id": alice.id,
        "kb_id": "kb_prompt",
        "user_prompt": "",
    }

    bob_settings_denied = client.put(
        "/auth/me/kbs/kb_prompt/query-settings",
        json={"user_prompt": "bob prompt"},
        headers=bob_headers,
    )
    assert bob_settings_denied.status_code == 403

    grant = client.put(
        "/admin/kbs/kb_prompt/acl",
        json={"user_id": bob.id, "role": "kb_viewer"},
        headers=admin_headers,
    )
    assert grant.status_code == 200, grant.text

    alice_settings = client.put(
        "/auth/me/kbs/kb_prompt/query-settings",
        json={"user_prompt": "alice persistent prompt"},
        headers=alice_headers,
    )
    assert alice_settings.status_code == 200, alice_settings.text
    assert alice_settings.json()["user_prompt"] == "alice persistent prompt"

    bob_settings = client.put(
        "/auth/me/kbs/kb_prompt/query-settings",
        json={"user_prompt": "bob persistent prompt"},
        headers=bob_headers,
    )
    assert bob_settings.status_code == 200, bob_settings.text
    assert bob_settings.json()["user_prompt"] == "bob persistent prompt"

    alice_query = client.post(
        "/kbs/kb_prompt/query",
        json={"query": "what prompt applies", "mode": "mix"},
        headers=alice_headers,
    )
    assert alice_query.status_code == 200, alice_query.text
    assert probe.instances["kb_prompt"].query_params[-1].user_prompt == (
        "alice persistent prompt"
    )

    bob_query = client.post(
        "/kbs/kb_prompt/query",
        json={"query": "what prompt applies", "mode": "mix"},
        headers=bob_headers,
    )
    assert bob_query.status_code == 200, bob_query.text
    assert probe.instances["kb_prompt"].query_params[-1].user_prompt == (
        "bob persistent prompt"
    )

    explicit_query = client.post(
        "/kbs/kb_prompt/query",
        json={
            "query": "what prompt applies",
            "mode": "mix",
            "user_prompt": "request prompt wins",
        },
        headers=bob_headers,
    )
    assert explicit_query.status_code == 200, explicit_query.text
    assert probe.instances["kb_prompt"].query_params[-1].user_prompt == (
        "request prompt wins"
    )

    cleared = client.put(
        "/auth/me/kbs/kb_prompt/query-settings",
        json={"user_prompt": ""},
        headers=bob_headers,
    )
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["user_prompt"] == ""

    fallback_query = client.post(
        "/kbs/kb_prompt/query",
        json={"query": "what prompt applies", "mode": "mix"},
        headers=bob_headers,
    )
    assert fallback_query.status_code == 200, fallback_query.text
    assert probe.instances["kb_prompt"].query_params[-1].user_prompt == (
        "kb default prompt"
    )


def test_enterprise_query_settings_writes_run_inside_kb_guard(
    monkeypatch, tmp_path
):
    client, user_service, _authz, _admin, alice, _bob, _probe = (
        _build_enterprise_client(monkeypatch, tmp_path)
    )
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    created = client.post(
        "/kbs",
        json={"id": "kb_query_settings_guard", "name": "Settings Guard"},
        headers=alice_headers,
    )
    assert created.status_code == 200, created.text

    store = client.app.state.metadata_store
    guard_depth = 0
    original_guard = store.kb_write_guard

    @asynccontextmanager
    async def tracking_guard(kb_id, expected_generation):
        nonlocal guard_depth
        async with original_guard(kb_id, expected_generation) as lifecycle:
            guard_depth += 1
            try:
                yield lifecycle
            finally:
                guard_depth -= 1

    monkeypatch.setattr(store, "kb_write_guard", tracking_guard)
    original_upsert = store.upsert_enterprise_user_kb_query_settings
    original_delete = store.delete_enterprise_user_kb_query_settings

    async def guarded_upsert(record):
        # Pure-ASGI admission plus the service-owned direct-call fence.
        assert guard_depth == 2
        return await original_upsert(record)

    async def guarded_delete(user_id, kb_id):
        assert guard_depth == 2
        return await original_delete(user_id, kb_id)

    monkeypatch.setattr(
        store,
        "upsert_enterprise_user_kb_query_settings",
        guarded_upsert,
    )
    monkeypatch.setattr(
        store,
        "delete_enterprise_user_kb_query_settings",
        guarded_delete,
    )
    saved = client.put(
        "/auth/me/kbs/kb_query_settings_guard/query-settings",
        json={"user_prompt": "guarded prompt"},
        headers=alice_headers,
    )
    assert saved.status_code == 200, saved.text
    assert guard_depth == 0
    cleared = client.put(
        "/auth/me/kbs/kb_query_settings_guard/query-settings",
        json={"user_prompt": ""},
        headers=alice_headers,
    )
    assert cleared.status_code == 200, cleared.text
    assert guard_depth == 0


def test_enterprise_query_settings_write_is_blocked_while_kb_deleting(
    monkeypatch, tmp_path
):
    client, user_service, _authz, _admin, alice, _bob, _probe = (
        _build_enterprise_client(monkeypatch, tmp_path)
    )
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    created = client.post(
        "/kbs",
        json={"id": "kb_query_settings_deleting", "name": "Deleting"},
        headers=alice_headers,
    )
    assert created.status_code == 200, created.text

    store = client.app.state.metadata_store
    kb_service = client.app.state.kb_service

    async def mark_deleting() -> None:
        record = await kb_service.get("kb_query_settings_deleting")
        async with store.kb_deletion_guard(
            record.id,
            record.generation,
            "job_delete_query_settings_guard",
        ):
            pass

    asyncio.run(mark_deleting())
    response = client.put(
        "/auth/me/kbs/kb_query_settings_deleting/query-settings",
        json={"user_prompt": "must not persist"},
        headers=alice_headers,
    )
    assert response.status_code == 409
    settings = asyncio.run(
        store.get_enterprise_user_kb_query_settings(
            alice.id,
            "kb_query_settings_deleting",
        )
    )
    assert settings is None


def test_query_settings_service_direct_set_and_clear_reject_deleting(
    monkeypatch, tmp_path
):
    client, user_service, _authz, _admin, alice, _bob, _probe = (
        _build_enterprise_client(monkeypatch, tmp_path)
    )
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    created = client.post(
        "/kbs",
        json={"id": "kb_query_settings_direct", "name": "Direct Settings"},
        headers=alice_headers,
    )
    assert created.status_code == 200, created.text

    store = client.app.state.metadata_store
    kb_service = client.app.state.kb_service
    service = client.app.state.enterprise_user_kb_query_settings_service

    async def exercise_direct_calls() -> None:
        record = await kb_service.get("kb_query_settings_direct")
        saved = await service.set_user_prompt(
            user_id=alice.id,
            kb_id=record.id,
            user_prompt="preserve me",
        )
        assert saved.user_prompt == "preserve me"
        async with store.kb_deletion_guard(
            record.id,
            record.generation,
            "job_delete_query_settings_direct",
        ):
            pass
        with pytest.raises(KBLifecycleConflictError):
            await service.set_user_prompt(
                user_id=alice.id,
                kb_id=record.id,
                user_prompt="must not persist",
            )
        with pytest.raises(KBLifecycleConflictError):
            await service.clear_user_prompt(
                user_id=alice.id,
                kb_id=record.id,
            )

    asyncio.run(exercise_direct_calls())
    settings = asyncio.run(
        store.get_enterprise_user_kb_query_settings(
            alice.id,
            "kb_query_settings_direct",
        )
    )
    assert settings is not None
    assert settings.user_prompt == "preserve me"


def test_query_settings_service_keeps_legacy_no_lifecycle_compatibility(tmp_path):
    store = SQLiteMetadataStore(tmp_path / "legacy-query-settings.sqlite3")
    service = UserKBQuerySettingsService(store)

    async def exercise_legacy_settings() -> None:
        await store.initialize()
        user = await UserService(store).create_user(
            username="legacy-settings-user",
            password="legacy-settings-password",
        )
        assert await store.get_kb_lifecycle("kb_legacy_settings") is None
        saved = await service.set_user_prompt(
            user_id=user.id,
            kb_id="kb_legacy_settings",
            user_prompt="legacy prompt",
        )
        assert saved.user_prompt == "legacy prompt"
        assert await service.clear_user_prompt(
            user_id=user.id,
            kb_id="kb_legacy_settings",
        )

    asyncio.run(exercise_legacy_settings())


def test_enterprise_user_kb_query_settings_reject_legacy_api_key_principal(
    monkeypatch, tmp_path
):
    args = _enterprise_args(enterprise_legacy_api_key_superadmin=True)
    client, user_service, _authz, _admin, alice, _bob, _probe = _build_enterprise_client(
        monkeypatch,
        tmp_path,
        args=args,
    )
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    legacy_api_key_headers = {"X-API-Key": _API_KEY}

    created = client.post(
        "/kbs",
        json={"id": "kb_api_key_settings", "name": "API Key Settings"},
        headers=alice_headers,
    )
    assert created.status_code == 200, created.text

    settings_read = client.get(
        "/auth/me/kbs/kb_api_key_settings/query-settings",
        headers=legacy_api_key_headers,
    )
    assert settings_read.status_code == 403
    settings_write = client.put(
        "/auth/me/kbs/kb_api_key_settings/query-settings",
        json={"user_prompt": "api key prompt"},
        headers=legacy_api_key_headers,
    )
    assert settings_write.status_code == 403


def test_enterprise_tenant_kb_acl_authorizes_members_only(monkeypatch, tmp_path):
    client, user_service, _authz, admin, alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}

    async def create_carol_and_dave():
        carol = await user_service.create_user(
            username="carol",
            password="carol-pass",
            created_by=admin.id,
        )
        dave = await user_service.create_user(
            username="dave",
            password="dave-pass",
            created_by=admin.id,
        )
        return carol, dave

    carol, dave = asyncio.run(create_carol_and_dave())
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}
    carol_headers = {"Authorization": f"Bearer {_token(user_service, carol)}"}
    dave_headers = {"Authorization": f"Bearer {_token(user_service, dave)}"}

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
    dave_membership = client.put(
        f"/admin/tenants/tenant-a/members/{dave.id}",
        json={"role": "tenant_admin"},
        headers=admin_headers,
    )
    assert dave_membership.status_code == 200, dave_membership.text

    members = client.get("/admin/tenants/tenant-a/members", headers=admin_headers)
    assert members.status_code == 200, members.text
    assert {item["user_id"] for item in members.json()} == {bob.id, dave.id}
    # Membership listings resolve user names so a client never has to join
    # against the (super-admin-only) /admin/users endpoints.
    admin_members_by_user = {item["user_id"]: item for item in members.json()}
    assert admin_members_by_user[bob.id]["username"] == "bob"
    assert admin_members_by_user[dave.id]["username"] == "dave"

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

    tenant_admin_members = client.get(
        "/tenants/tenant-a/members",
        headers=dave_headers,
    )
    assert tenant_admin_members.status_code == 200, tenant_admin_members.text
    assert {item["user_id"] for item in tenant_admin_members.json()} == {bob.id, dave.id}
    # Tenant admins cannot call /admin/users, so the self-service listing
    # itself must carry usernames (and display_name once the user sets one).
    self_service_by_user = {
        item["user_id"]: item for item in tenant_admin_members.json()
    }
    assert self_service_by_user[bob.id]["username"] == "bob"
    assert self_service_by_user[bob.id]["display_name"] is None
    profile_update = client.patch(
        "/auth/me",
        json={"display_name": "Bob B."},
        headers=bob_headers,
    )
    assert profile_update.status_code == 200, profile_update.text
    refreshed_members = client.get("/tenants/tenant-a/members", headers=dave_headers)
    assert refreshed_members.status_code == 200, refreshed_members.text
    refreshed_by_user = {item["user_id"]: item for item in refreshed_members.json()}
    assert refreshed_by_user[bob.id]["username"] == "bob"
    assert refreshed_by_user[bob.id]["display_name"] == "Bob B."
    tenant_admin_add_member = client.put(
        f"/tenants/tenant-a/members/{carol.id}",
        json={"role": "tenant_member"},
        headers=dave_headers,
    )
    # A tenant admin cannot move a known user out of another primary tenant.
    assert tenant_admin_add_member.status_code == 404
    super_admin_reassignment = client.put(
        f"/admin/tenants/tenant-a/members/{carol.id}",
        json={"role": "tenant_member"},
        headers=admin_headers,
    )
    assert super_admin_reassignment.status_code == 200, super_admin_reassignment.text
    assert super_admin_reassignment.json()["username"] == "carol"
    tenant_admin_promote_denied = client.put(
        f"/tenants/tenant-a/members/{carol.id}",
        json={"role": "tenant_admin"},
        headers=dave_headers,
    )
    assert tenant_admin_promote_denied.status_code == 404
    tenant_admin_self_revoke_denied = client.delete(
        f"/tenants/tenant-a/members/{dave.id}",
        headers=dave_headers,
    )
    assert tenant_admin_self_revoke_denied.status_code == 409
    cross_tenant_admin_denied = client.get(
        "/tenants/tenant-b/members",
        headers=dave_headers,
    )
    assert cross_tenant_admin_denied.status_code == 403
    carol_query_via_self_service = client.post(
        "/kbs/kb_tenant_acl/query",
        json={"query": "tenant admin added member", "mode": "mix"},
        headers=carol_headers,
    )
    assert carol_query_via_self_service.status_code == 200, carol_query_via_self_service.text

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


def test_enterprise_service_key_tenant_id_inherits_tenant_acl_only_when_requested(
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
        json={"query": "must not inherit by default", "mode": "mix"},
        headers=inherited_headers,
    )
    assert inherited_query.status_code == 403

    opt_in_key = client.post(
        "/admin/service-api-keys",
        json={
            "name": "tenant-inherited-reader",
            "tenant_id": "tenant-service",
            "inherit_tenant_kb_acl": True,
        },
        headers=admin_headers,
    )
    assert opt_in_key.status_code == 200, opt_in_key.text
    assert opt_in_key.json()["key"]["scopes"]["inherit_tenant_kb_acl"] is True
    opt_in_headers = {"X-API-Key": opt_in_key.json()["api_key"]}
    opt_in_list = client.get("/kbs", headers=opt_in_headers)
    assert opt_in_list.status_code == 200, opt_in_list.text
    assert [item["id"] for item in opt_in_list.json()["knowledge_bases"]] == [
        "kb_service_tenant_acl"
    ]
    opt_in_query = client.post(
        "/kbs/kb_service_tenant_acl/query",
        json={"query": "inherits when requested", "mode": "mix"},
        headers=opt_in_headers,
    )
    assert opt_in_query.status_code == 200, opt_in_query.text

    missing_tenant = client.post(
        "/admin/service-api-keys",
        json={"name": "bad-inherit", "inherit_tenant_kb_acl": True},
        headers=admin_headers,
    )
    assert missing_tenant.status_code == 400

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
    alice = _grant_file_download(user_service, alice, actor_user_id=admin.id)
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
    bob = _grant_file_download(user_service, bob, actor_user_id=admin.id)
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
        item for item in artifact_items if item["artifact_type"] == "original"
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


def test_enterprise_file_download_capability_guards_exports_and_original_preview(
    monkeypatch, tmp_path
):
    client, user_service, _authz, admin, alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}
    assert bob.can_download_files is False

    created = client.post(
        "/kbs",
        json={"id": "kb_file_capability", "name": "File Capability"},
        headers=alice_headers,
    )
    assert created.status_code == 200, created.text
    granted = client.put(
        "/admin/kbs/kb_file_capability/acl",
        json={"user_id": bob.id, "role": "kb_viewer"},
        headers=admin_headers,
    )
    assert granted.status_code == 200, granted.text
    uploaded = client.post(
        "/kbs/kb_file_capability/documents:upload",
        files=[("files", ("source.pdf", b"source-bytes", "application/pdf"))],
        headers=alice_headers,
    )
    assert uploaded.status_code == 200, uploaded.text
    document_id = uploaded.json()["documents"][0]["id"]
    parsed = client.post(
        f"/kbs/kb_file_capability/documents/{document_id}:parse",
        json={"engine": "mineru", "process_options": "iF"},
        headers=alice_headers,
    )
    assert parsed.status_code == 200, parsed.text
    artifacts = client.get(
        f"/kbs/kb_file_capability/documents/{document_id}/artifacts",
        headers=bob_headers,
    ).json()["artifacts"]
    original = next(item for item in artifacts if item["artifact_type"] == "original")
    blocks = next(item for item in artifacts if item["artifact_type"] == "blocks")

    for suffix in (":download", ":download-url", ":preview"):
        denied = client.get(
            f"/kbs/kb_file_capability/documents/{document_id}/artifacts/"
            f"{original['id']}{suffix}",
            headers=bob_headers,
        )
        assert denied.status_code == 403
        assert denied.json()["detail"] == "File download permission required"

    # Derived previews remain a viewer surface, while exporting even a derived
    # artifact requires the user-global capability.
    derived_preview = client.get(
        f"/kbs/kb_file_capability/documents/{document_id}/artifacts/"
        f"{blocks['id']}:preview",
        headers=bob_headers,
    )
    assert derived_preview.status_code == 200, derived_preview.text
    derived_download = client.get(
        f"/kbs/kb_file_capability/documents/{document_id}/artifacts/"
        f"{blocks['id']}:download",
        headers=bob_headers,
    )
    assert derived_download.status_code == 403

    bob = _grant_file_download(user_service, bob, actor_user_id=admin.id)
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}
    original_download = client.get(
        f"/kbs/kb_file_capability/documents/{document_id}/artifacts/"
        f"{original['id']}:download",
        headers=bob_headers,
    )
    assert original_download.status_code == 200, original_download.text
    assert original_download.content == b"source-bytes"
    original_preview = client.get(
        f"/kbs/kb_file_capability/documents/{document_id}/artifacts/"
        f"{original['id']}:preview",
        headers=bob_headers,
    )
    assert original_preview.status_code == 200, original_preview.text


def test_enterprise_artifact_per_type_policy_and_storage_uri_masking(
    monkeypatch, tmp_path
):
    args = _enterprise_args(
        enterprise_artifact_download_policy=json.dumps({"original": "kb_editor"}),
        enterprise_mask_storage_uris=True,
    )
    client, user_service, _authz, admin, alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path, args=args
    )
    bob = _grant_file_download(user_service, bob, actor_user_id=admin.id)
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}

    created = client.post(
        "/kbs",
        json={"id": "kb_artifact_type_policy", "name": "Artifact Type Policy"},
        headers=alice_headers,
    )
    assert created.status_code == 200, created.text
    grant_viewer = client.put(
        "/admin/kbs/kb_artifact_type_policy/acl",
        json={"user_id": bob.id, "role": "kb_viewer"},
        headers=admin_headers,
    )
    assert grant_viewer.status_code == 200, grant_viewer.text

    uploaded = client.post(
        "/kbs/kb_artifact_type_policy/documents:upload",
        files=[("files", ("artifact.pdf", b"pdf", "application/pdf"))],
        headers=alice_headers,
    )
    assert uploaded.status_code == 200, uploaded.text
    document_id = uploaded.json()["documents"][0]["id"]
    parsed = client.post(
        f"/kbs/kb_artifact_type_policy/documents/{document_id}:parse",
        json={"engine": "mineru", "process_options": "iF"},
        headers=alice_headers,
    )
    assert parsed.status_code == 200, parsed.text

    document_detail = client.get(
        f"/kbs/kb_artifact_type_policy/documents/{document_id}",
        headers=bob_headers,
    )
    assert document_detail.status_code == 200, document_detail.text
    assert document_detail.json()["source_uri"] == "<masked>"

    artifacts = client.get(
        f"/kbs/kb_artifact_type_policy/documents/{document_id}/artifacts",
        headers=bob_headers,
    )
    assert artifacts.status_code == 200, artifacts.text
    artifact_items = artifacts.json()["artifacts"]
    original = next(item for item in artifact_items if item["artifact_type"] == "original")
    blocks = next(item for item in artifact_items if item["artifact_type"] == "blocks")
    assert original["uri"] == "<masked>"
    assert "object_uri" not in original["metadata"]
    jobs = client.get("/kbs/kb_artifact_type_policy/jobs", headers=bob_headers)
    assert jobs.status_code == 200, jobs.text
    jobs_json = json.dumps(jobs.json(), ensure_ascii=False)
    assert "source_uri" not in jobs_json
    assert "blocks_path" not in jobs_json
    assert str(tmp_path) not in jobs_json

    viewer_original_download = client.get(
        f"/kbs/kb_artifact_type_policy/documents/{document_id}/artifacts/{original['id']}:download",
        headers=bob_headers,
    )
    assert viewer_original_download.status_code == 403
    viewer_original_preview = client.get(
        f"/kbs/kb_artifact_type_policy/documents/{document_id}/artifacts/{original['id']}:preview",
        headers=bob_headers,
    )
    assert viewer_original_preview.status_code == 403

    viewer_blocks_download = client.get(
        f"/kbs/kb_artifact_type_policy/documents/{document_id}/artifacts/{blocks['id']}:download",
        headers=bob_headers,
    )
    assert viewer_blocks_download.status_code == 200, viewer_blocks_download.text

    grant_editor = client.put(
        "/admin/kbs/kb_artifact_type_policy/acl",
        json={"user_id": bob.id, "role": "kb_editor"},
        headers=admin_headers,
    )
    assert grant_editor.status_code == 200, grant_editor.text
    editor_original_download = client.get(
        f"/kbs/kb_artifact_type_policy/documents/{document_id}/artifacts/{original['id']}:download",
        headers=bob_headers,
    )
    assert editor_original_download.status_code == 200, editor_original_download.text


def test_enterprise_artifact_action_policy_overrides_per_action(
    monkeypatch, tmp_path
):
    args = _enterprise_args(
        enterprise_artifact_action_policy=json.dumps(
            {
                "preview": {"*": "kb_editor"},
                "download": {"blocks": "kb_admin"},
                "download-url": {"original": "kb_admin"},
            }
        )
    )
    client, user_service, _authz, admin, alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path, args=args
    )
    bob = _grant_file_download(user_service, bob, actor_user_id=admin.id)
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}

    created = client.post(
        "/kbs",
        json={"id": "kb_artifact_action_policy", "name": "Artifact Action Policy"},
        headers=alice_headers,
    )
    assert created.status_code == 200, created.text
    grant_viewer = client.put(
        "/admin/kbs/kb_artifact_action_policy/acl",
        json={"user_id": bob.id, "role": "kb_viewer"},
        headers=admin_headers,
    )
    assert grant_viewer.status_code == 200, grant_viewer.text

    uploaded = client.post(
        "/kbs/kb_artifact_action_policy/documents:upload",
        files=[("files", ("artifact.pdf", b"pdf", "application/pdf"))],
        headers=alice_headers,
    )
    assert uploaded.status_code == 200, uploaded.text
    document_id = uploaded.json()["documents"][0]["id"]
    parsed = client.post(
        f"/kbs/kb_artifact_action_policy/documents/{document_id}:parse",
        json={"engine": "mineru", "process_options": "iF"},
        headers=alice_headers,
    )
    assert parsed.status_code == 200, parsed.text
    artifact_items = client.get(
        f"/kbs/kb_artifact_action_policy/documents/{document_id}/artifacts",
        headers=bob_headers,
    ).json()["artifacts"]
    original = next(item for item in artifact_items if item["artifact_type"] == "original")
    blocks = next(item for item in artifact_items if item["artifact_type"] == "blocks")

    viewer_preview = client.get(
        f"/kbs/kb_artifact_action_policy/documents/{document_id}/artifacts/{original['id']}:preview",
        headers=bob_headers,
    )
    assert viewer_preview.status_code == 403
    viewer_blocks_download = client.get(
        f"/kbs/kb_artifact_action_policy/documents/{document_id}/artifacts/{blocks['id']}:download",
        headers=bob_headers,
    )
    assert viewer_blocks_download.status_code == 403

    grant_editor = client.put(
        "/admin/kbs/kb_artifact_action_policy/acl",
        json={"user_id": bob.id, "role": "kb_editor"},
        headers=admin_headers,
    )
    assert grant_editor.status_code == 200, grant_editor.text
    editor_preview = client.get(
        f"/kbs/kb_artifact_action_policy/documents/{document_id}/artifacts/{original['id']}:preview",
        headers=bob_headers,
    )
    assert editor_preview.status_code == 200, editor_preview.text
    editor_blocks_download = client.get(
        f"/kbs/kb_artifact_action_policy/documents/{document_id}/artifacts/{blocks['id']}:download",
        headers=bob_headers,
    )
    assert editor_blocks_download.status_code == 403
    editor_original_presign = client.get(
        f"/kbs/kb_artifact_action_policy/documents/{document_id}/artifacts/{original['id']}:download-url",
        headers=bob_headers,
    )
    assert editor_original_presign.status_code == 403

    grant_admin = client.put(
        "/admin/kbs/kb_artifact_action_policy/acl",
        json={"user_id": bob.id, "role": "kb_admin"},
        headers=admin_headers,
    )
    assert grant_admin.status_code == 200, grant_admin.text
    admin_blocks_download = client.get(
        f"/kbs/kb_artifact_action_policy/documents/{document_id}/artifacts/{blocks['id']}:download",
        headers=bob_headers,
    )
    assert admin_blocks_download.status_code == 200, admin_blocks_download.text


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


def test_enterprise_registration_failed_attempts_are_limited_and_audited(
    monkeypatch, tmp_path
):
    client, user_service, _authz, admin, _alice, _bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    client.app.state.enterprise_registration_tracker = LoginAttemptTracker(
        max_attempts=2,
        window_seconds=60.0,
        lockout_seconds=120.0,
        time_func=lambda: 1000.0,
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}

    opened = client.patch(
        "/admin/settings/registration",
        json={"mode": "open"},
        headers=admin_headers,
    )
    assert opened.status_code == 200, opened.text

    duplicate_payload = {"username": "alice", "password": "not-used"}
    first = client.post("/auth/register", json=duplicate_payload)
    second = client.post("/auth/register", json=duplicate_payload)
    locked = client.post("/auth/register", json=duplicate_payload)

    assert first.status_code == 409
    assert second.status_code == 409
    assert locked.status_code == 429
    assert locked.headers["Retry-After"] == "120"

    audit_events = client.get("/admin/audit-events", headers=admin_headers)
    assert audit_events.status_code == 200, audit_events.text
    event_types = [event["event_type"] for event in audit_events.json()]
    assert "registration_failed" in event_types
    assert "registration_locked" in event_types


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
    refreshed_bob = asyncio.run(user_service.authenticate("bob", "bob-new-pass"))
    assert refreshed_bob is not None
    bob_headers = {"Authorization": f"Bearer {_token(user_service, refreshed_bob)}"}

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

    secondary = client.post(
        "/kbs",
        json={"id": "kb_user_batch_secondary", "name": "User Batch Secondary"},
        headers=alice_headers,
    )
    assert secondary.status_code == 200, secondary.text
    user_batch = client.post(
        f"/admin/users/{bob.id}/kb-access:batch-set",
        json={
            "entries": [
                {"kb_id": "kb_acl_batch", "action": "revoke"},
                {"kb_id": "kb_user_batch_secondary", "role": "kb_viewer"},
            ]
        },
        headers=admin_headers,
    )
    assert user_batch.status_code == 200, user_batch.text
    user_batch_body = user_batch.json()
    assert user_batch_body["revoked"] == ["kb_acl_batch"]
    assert [(item["kb_id"], item["user_id"], item["role"]) for item in user_batch_body["granted"]] == [
        ("kb_user_batch_secondary", bob.id, "kb_viewer")
    ]
    bob_kbs = client.get("/kbs", headers=bob_headers)
    assert bob_kbs.status_code == 200, bob_kbs.text
    assert [item["id"] for item in bob_kbs.json()["knowledge_bases"]] == [
        "kb_user_batch_secondary"
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

    service_me = client.get("/auth/me", headers=service_headers)
    assert service_me.status_code == 200, service_me.text
    assert service_me.json()["user"] is None
    assert service_me.json()["principal"]["auth_method"] == "service_api_key"
    assert service_me.json()["principal"]["user_id"] == f"service-key:{key_record['id']}"
    assert service_me.json()["principal"]["username"] == "alpha-reader"

    service_settings = client.get(
        "/auth/me/kbs/kb_service_alpha/query-settings",
        headers=service_headers,
    )
    assert service_settings.status_code == 403
    service_settings_write = client.put(
        "/auth/me/kbs/kb_service_alpha/query-settings",
        json={"user_prompt": "service prompt"},
        headers=service_headers,
    )
    assert service_settings_write.status_code == 403

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


def test_rotate_service_api_key_returns_new_raw_key_and_revokes_old(
    monkeypatch, tmp_path
):
    client, user_service, _authz, admin, alice, _bob, _probe = _build_enterprise_client(
        monkeypatch,
        tmp_path,
        api_key=None,
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}

    created_kb = client.post(
        "/kbs", json={"id": "kb_rotate", "name": "Rotate"}, headers=alice_headers
    )
    assert created_kb.status_code == 200, created_kb.text
    created = client.post(
        "/admin/service-api-keys",
        json={"name": "rotating", "kb_roles": {"kb_rotate": "kb_viewer"}},
        headers=admin_headers,
    )
    assert created.status_code == 200, created.text
    old_raw = created.json()["api_key"]
    old_key = created.json()["key"]

    rotated = client.post(
        f"/admin/service-api-keys/{old_key['id']}:rotate",
        json={"expires_in_seconds": 7200},
        headers=admin_headers,
    )
    assert rotated.status_code == 200, rotated.text
    new_raw = rotated.json()["api_key"]
    new_key = rotated.json()["key"]
    assert new_raw != old_raw
    assert new_key["id"] != old_key["id"]
    assert new_key["metadata"]["rotated_from"] == old_key["id"]
    assert new_key["expires_at"] is not None

    assert client.get("/kbs", headers={"X-API-Key": old_raw}).status_code == 401
    new_list = client.get("/kbs", headers={"X-API-Key": new_raw})
    assert new_list.status_code == 200, new_list.text
    assert [item["id"] for item in new_list.json()["knowledge_bases"]] == [
        "kb_rotate"
    ]

    listed = client.get("/admin/service-api-keys", headers=admin_headers)
    by_id = {item["id"]: item for item in listed.json()}
    assert by_id[old_key["id"]]["status"] == "revoked"
    assert by_id[new_key["id"]]["status"] == "active"
    assert "api_key" not in by_id[new_key["id"]]
    assert "key_hash" not in by_id[new_key["id"]]

    events = client.get("/admin/audit-events", headers=admin_headers).json()
    event_types = {event["event_type"] for event in events}
    assert "service_api_key_rotated" in event_types


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


# ---------------------------------------------------------------------------
# Ownership-aware document deletion + can_delete_documents capability
# ---------------------------------------------------------------------------


def _dd_create_kb(client, headers, kb_id, name="KB"):
    resp = client.post("/kbs", json={"id": kb_id, "name": name}, headers=headers)
    assert resp.status_code == 200, resp.text


def _dd_grant(client, admin_headers, kb_id, user_id, role):
    resp = client.put(
        f"/admin/kbs/{kb_id}/acl",
        json={"user_id": user_id, "role": role},
        headers=admin_headers,
    )
    assert resp.status_code == 200, resp.text


def _dd_upload(client, headers, kb_id, source_name="note.md"):
    resp = client.post(
        f"/kbs/{kb_id}/documents:texts",
        json={"documents": [{"text": "hello world", "source_name": source_name}]},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["documents"][0]


def _dd_audit(client, admin_headers):
    resp = client.get("/admin/audit-events", headers=admin_headers)
    assert resp.status_code == 200
    return resp.json()


def test_enterprise_kb_editor_can_self_delete_own_document(monkeypatch, tmp_path):
    client, user_service, _authz, admin, _alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}

    _dd_create_kb(client, admin_headers, "kb_self")
    _dd_grant(client, admin_headers, "kb_self", bob.id, "kb_editor")
    doc = _dd_upload(client, bob_headers, "kb_self")
    # created_by is principal-derived and surfaced on the document record.
    assert doc["metadata"]["created_by"] == bob.id

    deleted = client.delete(
        f"/kbs/kb_self/documents/{doc['id']}", headers=bob_headers
    )
    assert deleted.status_code == 200, deleted.text

    events = _dd_audit(client, admin_headers)
    delete_events = [e for e in events if e["event_type"] == "document_delete_queued"]
    assert delete_events, "expected a document_delete_queued audit event"
    meta = delete_events[0]["metadata"]
    assert meta["delete_scope"] == "self"
    assert meta["document_owner"] == bob.id
    assert delete_events[0]["actor_user_id"] == bob.id


def test_enterprise_kb_editor_cannot_delete_others_document(monkeypatch, tmp_path):
    client, user_service, _authz, admin, _alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    carol = asyncio.run(
        user_service.create_user(username="carol", password="carol-pass")
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}
    carol_headers = {"Authorization": f"Bearer {_token(user_service, carol)}"}

    _dd_create_kb(client, admin_headers, "kb_shared")
    _dd_grant(client, admin_headers, "kb_shared", bob.id, "kb_editor")
    _dd_grant(client, admin_headers, "kb_shared", carol.id, "kb_editor")
    doc = _dd_upload(client, carol_headers, "kb_shared")  # carol owns it

    denied = client.delete(
        f"/kbs/kb_shared/documents/{doc['id']}", headers=bob_headers
    )
    assert denied.status_code == 403
    assert denied.json()["detail"] == "Document delete denied"

    events = _dd_audit(client, admin_headers)
    assert any(
        e["event_type"] == "permission_denied"
        and e["actor_user_id"] == bob.id
        and e.get("metadata", {}).get("minimum_role") == "kb_admin"
        for e in events
    )
    # The denied document must not have produced a delete job / queued event.
    assert not any(
        e["event_type"] == "document_delete_queued"
        and doc["id"] in (e.get("metadata", {}).get("document_ids") or [])
        for e in events
    )


def test_enterprise_can_delete_documents_capability_allows_deleting_others(
    monkeypatch, tmp_path
):
    client, user_service, _authz, admin, _alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    carol = asyncio.run(
        user_service.create_user(username="carol", password="carol-pass")
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    carol_headers = {"Authorization": f"Bearer {_token(user_service, carol)}"}

    _dd_create_kb(client, admin_headers, "kb_cap")
    _dd_grant(client, admin_headers, "kb_cap", bob.id, "kb_editor")
    _dd_grant(client, admin_headers, "kb_cap", carol.id, "kb_editor")
    doc = _dd_upload(client, carol_headers, "kb_cap")  # carol owns it

    # Without the capability bob (editor) cannot delete carol's document.
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}
    pre = client.delete(f"/kbs/kb_cap/documents/{doc['id']}", headers=bob_headers)
    assert pre.status_code == 403

    granted = client.patch(
        f"/admin/users/{bob.id}",
        json={"can_delete_documents": True},
        headers=admin_headers,
    )
    assert granted.status_code == 200, granted.text
    assert granted.json()["can_delete_documents"] is True

    # The capability toggle bumps token_version, so re-mint bob's token.
    bob = asyncio.run(user_service.get_user_or_404(bob.id))
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}

    me = client.get("/auth/me", headers=bob_headers)
    assert me.status_code == 200, me.text
    assert me.json()["principal"]["can_delete_documents"] is True

    deleted = client.delete(f"/kbs/kb_cap/documents/{doc['id']}", headers=bob_headers)
    assert deleted.status_code == 200, deleted.text
    events = _dd_audit(client, admin_headers)
    delete_events = [e for e in events if e["event_type"] == "document_delete_queued"]
    assert delete_events[0]["metadata"]["delete_scope"] == "privileged"


def test_enterprise_kb_admin_and_super_admin_delete_any_document(monkeypatch, tmp_path):
    client, user_service, _authz, admin, _alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    carol = asyncio.run(
        user_service.create_user(username="carol", password="carol-pass")
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}
    carol_headers = {"Authorization": f"Bearer {_token(user_service, carol)}"}

    _dd_create_kb(client, admin_headers, "kb_admindel")
    _dd_grant(client, admin_headers, "kb_admindel", bob.id, "kb_admin")
    _dd_grant(client, admin_headers, "kb_admindel", carol.id, "kb_editor")
    doc1 = _dd_upload(client, carol_headers, "kb_admindel", source_name="one.md")
    doc2 = _dd_upload(client, carol_headers, "kb_admindel", source_name="two.md")

    admin_del = client.delete(
        f"/kbs/kb_admindel/documents/{doc1['id']}", headers=bob_headers
    )
    assert admin_del.status_code == 200, admin_del.text

    super_del = client.delete(
        f"/kbs/kb_admindel/documents/{doc2['id']}", headers=admin_headers
    )
    assert super_del.status_code == 200, super_del.text

    events = _dd_audit(client, admin_headers)
    scopes = [
        e["metadata"]["delete_scope"]
        for e in events
        if e["event_type"] == "document_delete_queued"
    ]
    assert scopes == ["privileged", "privileged"]


def test_enterprise_batch_delete_mixes_self_and_permission_denied(monkeypatch, tmp_path):
    client, user_service, _authz, admin, _alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    carol = asyncio.run(
        user_service.create_user(username="carol", password="carol-pass")
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}
    carol_headers = {"Authorization": f"Bearer {_token(user_service, carol)}"}

    _dd_create_kb(client, admin_headers, "kb_batch")
    _dd_grant(client, admin_headers, "kb_batch", bob.id, "kb_editor")
    _dd_grant(client, admin_headers, "kb_batch", carol.id, "kb_editor")
    doc_own = _dd_upload(client, bob_headers, "kb_batch", source_name="own.md")
    doc_other = _dd_upload(client, carol_headers, "kb_batch", source_name="other.md")

    resp = client.post(
        "/kbs/kb_batch/documents:batch-delete",
        json={"document_ids": [doc_own["id"], doc_other["id"]]},
        headers=bob_headers,
    )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["id"]

    job = client.get(f"/kbs/kb_batch/jobs/{job_id}", headers=bob_headers).json()
    items = {item["document_id"]: item for item in job["result"]["items"]}
    assert items[doc_own["id"]]["status"] == "succeeded"
    assert items[doc_other["id"]]["error_code"] == "permission_denied"

    # The unauthorized document was never claimed and remains intact.
    other = client.get(
        f"/kbs/kb_batch/documents/{doc_other['id']}", headers=bob_headers
    )
    assert other.status_code == 200
    assert other.json()["status"] == "uploaded"

    events = _dd_audit(client, admin_headers)
    batch_events = [
        e for e in events if e["event_type"] == "documents_batch_delete_queued"
    ]
    assert batch_events[0]["metadata"]["permission_denied_count"] == 1
    assert batch_events[0]["metadata"]["delete_scopes"][doc_own["id"]] == "self"


def test_enterprise_document_created_by_is_unspoofable(monkeypatch, tmp_path):
    client, user_service, _authz, admin, _alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}

    _dd_create_kb(client, admin_headers, "kb_spoof")
    _dd_grant(client, admin_headers, "kb_spoof", bob.id, "kb_editor")

    # created_by is a reserved metadata key — rejected on :texts upload...
    spoof_texts = client.post(
        "/kbs/kb_spoof/documents:texts",
        json={
            "documents": [
                {"text": "x", "source_name": "n.md", "metadata": {"created_by": "evil"}}
            ]
        },
        headers=bob_headers,
    )
    assert spoof_texts.status_code == 422

    doc = _dd_upload(client, bob_headers, "kb_spoof")
    assert doc["metadata"]["created_by"] == bob.id

    # ...and rejected on PATCH metadata.
    spoof_patch = client.patch(
        f"/kbs/kb_spoof/documents/{doc['id']}",
        json={"metadata": {"created_by": "evil"}},
        headers=bob_headers,
    )
    assert spoof_patch.status_code == 422


def test_enterprise_service_key_delete_is_owner_scoped(monkeypatch, tmp_path):
    client, user_service, _authz, admin, _alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}

    _dd_create_kb(client, admin_headers, "kb_svc")
    _dd_grant(client, admin_headers, "kb_svc", bob.id, "kb_editor")

    created = client.post(
        "/admin/service-api-keys",
        json={"name": "svc-editor", "kb_roles": {"kb_svc": "kb_editor"}},
        headers=admin_headers,
    )
    assert created.status_code == 200, created.text
    svc_headers = {"X-API-Key": created.json()["api_key"]}

    # The service key may delete a document it uploaded itself (scope "self").
    own = _dd_upload(client, svc_headers, "kb_svc", source_name="svc-own.md")
    own_del = client.delete(
        f"/kbs/kb_svc/documents/{own['id']}", headers=svc_headers
    )
    assert own_del.status_code == 200, own_del.text

    # ...but not a document uploaded by another principal (no capability).
    other = _dd_upload(client, bob_headers, "kb_svc", source_name="bob-own.md")
    denied = client.delete(
        f"/kbs/kb_svc/documents/{other['id']}", headers=svc_headers
    )
    assert denied.status_code == 403


def test_authorize_document_delete_decision_matrix(tmp_path):
    from lightrag.api.enterprise_auth import (
        AuditService,
        AuthorizationService,
        Principal,
        SERVICE_API_KEY_AUTH_METHOD,
        SYSTEM_ROLE_SUPER_ADMIN,
        SYSTEM_ROLE_USER,
        USER_STATUS_ACTIVE,
    )
    from lightrag.api.kb_service import utc_now_iso
    from lightrag.api.metadata_store import EnterpriseUserRecord, KBACLRecord

    store = SQLiteMetadataStore(tmp_path / "metadata" / "matrix.sqlite3")
    authz = AuthorizationService(store, AuditService(store))

    def principal(user_id, *, role=SYSTEM_ROLE_USER, can_delete=False, auth="jwt"):
        return Principal(
            user_id=user_id,
            username=user_id,
            system_role=role,
            status=USER_STATUS_ACTIVE,
            tenant_id=None,
            tenant_roles={},
            can_create_kb=False,
            can_use_bypass_query=False,
            token_version=1,
            auth_method=auth,
            metadata={},
            can_delete_documents=can_delete,
        )

    async def run():
        await store.initialize()
        now = utc_now_iso()
        # enterprise_kb_acl.user_id has a FK to enterprise_users(id).
        for uid in ("usr_editor", "usr_admin_role"):
            await store.upsert_enterprise_user(
                EnterpriseUserRecord(
                    id=uid,
                    username=uid,
                    password_hash="x",
                    system_role=SYSTEM_ROLE_USER,
                    status=USER_STATUS_ACTIVE,
                    tenant_id=None,
                    can_create_kb=False,
                    can_use_bypass_query=False,
                    token_version=1,
                    metadata={},
                    created_at=now,
                    updated_at=now,
                )
            )
        await store.upsert_kb_acl(
            KBACLRecord(
                kb_id="kb1",
                user_id="usr_editor",
                role="kb_editor",
                granted_by="admin",
                created_at=now,
                updated_at=now,
            )
        )
        await store.upsert_kb_acl(
            KBACLRecord(
                kb_id="kb1",
                user_id="usr_admin_role",
                role="kb_admin",
                granted_by="admin",
                created_at=now,
                updated_at=now,
            )
        )

        # super admin and kb_admin and the capability all delete any document.
        assert (
            await authz.authorize_document_delete(
                principal("x", role=SYSTEM_ROLE_SUPER_ADMIN), "kb1", document_owner_id=None
            )
            == "privileged"
        )
        assert (
            await authz.authorize_document_delete(
                principal("usr_admin_role"), "kb1", document_owner_id="someone"
            )
            == "privileged"
        )
        assert (
            await authz.authorize_document_delete(
                principal("usr_cap", can_delete=True), "kb1", document_owner_id="someone"
            )
            == "privileged"
        )

        # editor deletes only its own uploads.
        assert (
            await authz.authorize_document_delete(
                principal("usr_editor"), "kb1", document_owner_id="usr_editor"
            )
            == "self"
        )
        with pytest.raises(HTTPException) as denied_other:
            await authz.authorize_document_delete(
                principal("usr_editor"), "kb1", document_owner_id="someone_else"
            )
        assert denied_other.value.status_code == 403

        # legacy documents without created_by are only deletable by privileged actors.
        with pytest.raises(HTTPException) as denied_legacy:
            await authz.authorize_document_delete(
                principal("usr_editor"), "kb1", document_owner_id=None
            )
        assert denied_legacy.value.status_code == 403

        # the capability is ignored for service-key principals.
        with pytest.raises(HTTPException) as denied_service:
            await authz.authorize_document_delete(
                principal("svc", can_delete=True, auth=SERVICE_API_KEY_AUTH_METHOD),
                "kb1",
                document_owner_id="someone",
            )
        assert denied_service.value.status_code == 403

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Multi-KB query ACL (handler self-enforces viewer on every target KB)
# ---------------------------------------------------------------------------


def test_multi_kb_query_requires_viewer_on_every_kb(monkeypatch, tmp_path):
    client, user_service, _authz, admin, _alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}
    _dd_create_kb(client, admin_headers, "kb_m1")
    _dd_create_kb(client, admin_headers, "kb_m2")
    _dd_grant(client, admin_headers, "kb_m1", bob.id, "kb_viewer")

    # bob has viewer on kb_m1 but NOT kb_m2 → fail closed (the central
    # middleware does not cover /kbs:query, so the handler must enforce this).
    denied = client.post(
        "/kbs:query",
        json={"kb_ids": ["kb_m1", "kb_m2"], "query": "cross kb question"},
        headers=bob_headers,
    )
    assert denied.status_code == 403

    _dd_grant(client, admin_headers, "kb_m2", bob.id, "kb_viewer")
    ok = client.post(
        "/kbs:query",
        json={"kb_ids": ["kb_m1", "kb_m2"], "query": "cross kb question"},
        headers=bob_headers,
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["kb_ids"] == ["kb_m1", "kb_m2"]


def test_multi_kb_query_bypass_denied(monkeypatch, tmp_path):
    client, user_service, _authz, admin, _alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}
    _dd_create_kb(client, admin_headers, "kb_bp")
    _dd_grant(client, admin_headers, "kb_bp", bob.id, "kb_viewer")
    resp = client.post(
        "/kbs:query",
        json={"kb_ids": ["kb_bp"], "query": "raw model please", "mode": "bypass"},
        headers=bob_headers,
    )
    assert resp.status_code == 400


def test_multi_kb_query_writes_audit(monkeypatch, tmp_path):
    client, user_service, _authz, admin, _alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}
    _dd_create_kb(client, admin_headers, "kb_au")
    _dd_grant(client, admin_headers, "kb_au", bob.id, "kb_viewer")
    resp = client.post(
        "/kbs:query",
        json={"kb_ids": ["kb_au"], "query": "please audit this question"},
        headers=bob_headers,
    )
    assert resp.status_code == 200, resp.text
    events = _dd_audit(client, admin_headers)
    audit = [e for e in events if e["event_type"] == "multi_kb_query_executed"]
    assert audit, "expected a multi_kb_query_executed audit event"
    meta = audit[0]["metadata"]
    assert meta["kb_ids"] == ["kb_au"]
    assert "query_hash" in meta
    # The raw query text must never be logged.
    assert "please audit this question" not in json.dumps(meta)


def test_admin_audit_events_filtering_and_pagination(monkeypatch, tmp_path):
    client, user_service, _authz, admin, _alice, _bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    _dd_create_kb(client, admin_headers, "kb_x")
    _dd_create_kb(client, admin_headers, "kb_y")
    created = client.post(
        "/admin/users",
        json={"username": "carol", "password": "carol-pass"},
        headers=admin_headers,
    )
    assert created.status_code == 200, created.text

    # Filter by event_type — exact match only.
    kb_created = client.get(
        "/admin/audit-events", params={"event_type": "kb_created"}, headers=admin_headers
    )
    assert kb_created.status_code == 200
    assert {e["event_type"] for e in kb_created.json()} == {"kb_created"}
    assert len(kb_created.json()) >= 2

    # Filter by target_type.
    user_events = client.get(
        "/admin/audit-events", params={"target_type": "user"}, headers=admin_headers
    ).json()
    assert user_events and all(e["target_type"] == "user" for e in user_events)

    # Pagination (newest-first) yields distinct rows.
    page1 = client.get(
        "/admin/audit-events",
        params={"event_type": "kb_created", "limit": 1},
        headers=admin_headers,
    ).json()
    page2 = client.get(
        "/admin/audit-events",
        params={"event_type": "kb_created", "limit": 1, "offset": 1},
        headers=admin_headers,
    ).json()
    assert len(page1) == 1 and len(page2) == 1
    assert page1[0]["id"] != page2[0]["id"]

    # Filter by actor.
    by_actor = client.get(
        "/admin/audit-events", params={"actor_user_id": admin.id}, headers=admin_headers
    ).json()
    assert by_actor and all(e["actor_user_id"] == admin.id for e in by_actor)


def test_admin_tenant_crud_and_overview(monkeypatch, tmp_path):
    client, user_service, _authz, admin, _alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}

    # Tenant management is super-admin only (enforced by the /admin middleware).
    denied = client.post("/admin/tenants", json={"name": "X"}, headers=bob_headers)
    assert denied.status_code == 403

    created = client.post(
        "/admin/tenants",
        json={"name": "Acme", "tenant_id": "tenant-acme"},
        headers=admin_headers,
    )
    assert created.status_code == 200, created.text
    assert created.json()["id"] == "tenant-acme"
    assert created.json()["status"] == "active"

    dup = client.post(
        "/admin/tenants",
        json={"name": "Acme2", "tenant_id": "tenant-acme"},
        headers=admin_headers,
    )
    assert dup.status_code == 409

    listed = client.get("/admin/tenants", headers=admin_headers)
    assert listed.status_code == 200
    assert "tenant-acme" in [t["id"] for t in listed.json()]

    # A user in the tenant creates a KB; tenant overview should reflect it.
    dave = asyncio.run(
        user_service.create_user(
            username="dave",
            password="dave-pass",
            tenant_id="tenant-acme",
            can_create_kb=True,
        )
    )
    dave_headers = {"Authorization": f"Bearer {_token(user_service, dave)}"}
    grant = client.put(
        f"/admin/tenants/tenant-acme/members/{dave.id}",
        json={"role": "tenant_member"},
        headers=admin_headers,
    )
    assert grant.status_code == 200, grant.text
    kb = client.post(
        "/kbs", json={"id": "kb_acme", "name": "Acme KB"}, headers=dave_headers
    )
    assert kb.status_code == 200, kb.text

    detail = client.get("/admin/tenants/tenant-acme", headers=admin_headers)
    assert detail.status_code == 200, detail.text
    assert detail.json()["member_count"] == 1
    assert detail.json()["kb_count"] == 1

    kbs = client.get("/admin/tenants/tenant-acme/kbs", headers=admin_headers)
    assert kbs.status_code == 200
    assert [k["id"] for k in kbs.json()] == ["kb_acme"]

    updated = client.patch(
        "/admin/tenants/tenant-acme",
        json={"status": "disabled", "name": "Acme Inc"},
        headers=admin_headers,
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["status"] == "disabled"
    assert updated.json()["name"] == "Acme Inc"

    assert (
        client.get("/admin/tenants/ghost", headers=admin_headers).status_code == 404
    )


def test_admin_users_list_filtering(monkeypatch, tmp_path):
    client, user_service, _authz, admin, _alice, _bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    asyncio.run(user_service.create_user(username="carol", password="p", tenant_id="t1"))
    asyncio.run(user_service.create_user(username="dave", password="p", tenant_id="t1"))
    asyncio.run(user_service.create_user(username="erin", password="p", tenant_id="t2"))

    by_tenant = client.get(
        "/admin/users", params={"tenant_id": "t1"}, headers=admin_headers
    ).json()
    assert {u["username"] for u in by_tenant} == {"carol", "dave"}

    by_q = client.get("/admin/users", params={"q": "ar"}, headers=admin_headers).json()
    assert {u["username"] for u in by_q} == {"carol"}

    by_status = client.get(
        "/admin/users", params={"status": "active"}, headers=admin_headers
    ).json()
    assert by_status and all(u["status"] == "active" for u in by_status)

    page = client.get(
        "/admin/users", params={"limit": 2, "offset": 0}, headers=admin_headers
    ).json()
    assert len(page) == 2


def test_admin_tenant_delete_requires_empty(monkeypatch, tmp_path):
    client, user_service, _authz, admin, _alice, _bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}

    # Empty tenant is deletable.
    client.post(
        "/admin/tenants",
        json={"name": "Empty", "tenant_id": "t-empty"},
        headers=admin_headers,
    )
    deleted = client.delete("/admin/tenants/t-empty", headers=admin_headers)
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["deleted"] is True
    assert (
        client.get("/admin/tenants/t-empty", headers=admin_headers).status_code == 404
    )

    # A tenant with a member + user + KB cannot be deleted (no cascade).
    client.post(
        "/admin/tenants",
        json={"name": "Full", "tenant_id": "t-full"},
        headers=admin_headers,
    )
    dave = asyncio.run(
        user_service.create_user(
            username="dave", password="p", tenant_id="t-full", can_create_kb=True
        )
    )
    client.put(
        f"/admin/tenants/t-full/members/{dave.id}",
        json={"role": "tenant_member"},
        headers=admin_headers,
    )
    dave_headers = {"Authorization": f"Bearer {_token(user_service, dave)}"}
    kb = client.post(
        "/kbs", json={"id": "kb_full", "name": "KB"}, headers=dave_headers
    )
    assert kb.status_code == 200, kb.text

    blocked = client.delete("/admin/tenants/t-full", headers=admin_headers)
    assert blocked.status_code == 409
    detail = blocked.json()["detail"]
    assert detail["error_code"] == "tenant_not_empty"
    assert detail["member_count"] == 1
    assert detail["kb_count"] == 1
    assert detail["user_count"] == 1

    assert client.delete("/admin/tenants/ghost", headers=admin_headers).status_code == 404


def test_admin_user_access_view(monkeypatch, tmp_path):
    client, user_service, _authz, admin, alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}

    created = client.post(
        "/kbs", json={"id": "kb_acc", "name": "Acc"}, headers=alice_headers
    )
    assert created.status_code == 200, created.text
    _dd_grant(client, admin_headers, "kb_acc", bob.id, "kb_editor")
    grant_t = client.put(
        f"/admin/tenants/t-acc/members/{bob.id}",
        json={"role": "tenant_member"},
        headers=admin_headers,
    )
    assert grant_t.status_code == 200, grant_t.text

    resp = client.get(f"/admin/users/{bob.id}/access", headers=admin_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["user_id"] == bob.id
    assert body["can_delete_documents"] is False
    assert {m["tenant_id"]: m["role"] for m in body["tenant_memberships"]} == {
        "t-acc": "tenant_member"
    }
    assert {a["kb_id"]: a["role"] for a in body["kb_acls"]} == {"kb_acc": "kb_editor"}

    assert (
        client.get("/admin/users/usr_ghost/access", headers=admin_headers).status_code
        == 404
    )


def test_enterprise_kb_visibility_public_grants_viewer_read_only(monkeypatch, tmp_path):
    client, user_service, authz, admin, alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path, api_key=None
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}

    created = client.post(
        "/kbs",
        json={"id": "kb_pub", "name": "Public KB", "visibility": "private"},
        headers=alice_headers,
    )
    assert created.status_code == 200, created.text

    # Private baseline: no ACL means no access and no listing for bob.
    assert client.get("/kbs/kb_pub", headers=bob_headers).status_code == 403
    assert client.get("/kbs", headers=bob_headers).json()["knowledge_bases"] == []

    same_value_denied = client.patch(
        "/kbs/kb_pub", json={"visibility": "private"}, headers=alice_headers
    )
    assert same_value_denied.status_code == 403
    direct_admin_denied = client.patch(
        "/kbs/kb_pub", json={"visibility": "public"}, headers=alice_headers
    )
    assert direct_admin_denied.status_code == 403

    asyncio.run(
        authz.grant_tenant_membership(
            "tenant-visibility",
            bob.id,
            "tenant_admin",
            granted_by=admin.id,
        )
    )
    bob = asyncio.run(user_service.get_user_or_404(bob.id))
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}
    tenant_acl = client.put(
        "/admin/kbs/kb_pub/acl",
        json={"tenant_id": "tenant-visibility", "role": "kb_admin"},
        headers=admin_headers,
    )
    assert tenant_acl.status_code == 200, tenant_acl.text
    tenant_admin_denied = client.patch(
        "/kbs/kb_pub", json={"visibility": "internal"}, headers=bob_headers
    )
    assert tenant_admin_denied.status_code == 403
    revoked_tenant_acl = client.delete(
        "/admin/kbs/kb_pub/acl/tenants/tenant-visibility",
        headers=admin_headers,
    )
    assert revoked_tenant_acl.status_code == 200, revoked_tenant_acl.text

    patched = client.patch(
        "/kbs/kb_pub", json={"visibility": "public"}, headers=admin_headers
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["visibility"] == "public"

    # Public implies kb_viewer for any authenticated interactive user.
    assert client.get("/kbs/kb_pub", headers=bob_headers).status_code == 200
    bob_list = client.get("/kbs", headers=bob_headers)
    assert [item["id"] for item in bob_list.json()["knowledge_bases"]] == ["kb_pub"]
    bob_query = client.post(
        "/kbs/kb_pub/query",
        json={"query": "what is public", "mode": "mix"},
        headers=bob_headers,
    )
    assert bob_query.status_code == 200, bob_query.text

    # Viewer only: writes and KB config stay denied without explicit ACL.
    assert (
        client.patch(
            "/kbs/kb_pub", json={"name": "Nope"}, headers=bob_headers
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/kbs/kb_pub/documents:texts",
            json={"documents": [{"text": "hello", "source_name": "x.md"}]},
            headers=bob_headers,
        ).status_code
        == 403
    )

    # Service keys keep explicit-scope-only semantics: public grants nothing.
    key_created = client.post(
        "/admin/service-api-keys",
        json={"name": "no-scope", "kb_roles": {}},
        headers=admin_headers,
    )
    assert key_created.status_code == 200, key_created.text
    service_headers = {"X-API-Key": key_created.json()["api_key"]}
    assert client.get("/kbs/kb_pub", headers=service_headers).status_code == 403
    service_list = client.get("/kbs", headers=service_headers)
    assert service_list.status_code == 200
    assert service_list.json()["knowledge_bases"] == []


def test_enterprise_kb_visibility_internal_same_tenant_only(monkeypatch, tmp_path):
    client, user_service, _authz, admin, alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}

    carol_created = client.post(
        "/admin/users",
        json={
            "username": "carol",
            "password": "carol-pass",
            "can_create_kb": True,
            "tenant_id": "tenant-a",
        },
        headers=admin_headers,
    )
    assert carol_created.status_code == 200, carol_created.text
    carol = asyncio.run(user_service.get_user_or_404(carol_created.json()["id"]))

    created = client.post(
        "/kbs",
        json={
            "id": "kb_internal",
            "name": "Internal KB",
            "owner_id": carol.id,
            "tenant_id": "tenant-a",
            "visibility": "internal",
        },
        headers=admin_headers,
    )
    assert created.status_code == 200, created.text
    assert created.json()["tenant_id"] == "tenant-a"
    assert created.json()["origin"] == "platform"
    assert created.json()["metadata"] == {}

    # No tenant and a different tenant both stay denied.
    assert client.get("/kbs/kb_internal", headers=bob_headers).status_code == 403
    other_tenant = client.patch(
        f"/admin/users/{bob.id}", json={"tenant_id": "tenant-b"}, headers=admin_headers
    )
    assert other_tenant.status_code == 200, other_tenant.text
    bob_b = asyncio.run(user_service.get_user_or_404(bob.id))
    bob_b_headers = {"Authorization": f"Bearer {_token(user_service, bob_b)}"}
    assert client.get("/kbs/kb_internal", headers=bob_b_headers).status_code == 403

    # Direct user.tenant_id assignment to the KB tenant implies kb_viewer.
    same_tenant = client.patch(
        f"/admin/users/{bob.id}", json={"tenant_id": "tenant-a"}, headers=admin_headers
    )
    assert same_tenant.status_code == 200, same_tenant.text
    bob_a = asyncio.run(user_service.get_user_or_404(bob.id))
    bob_a_headers = {"Authorization": f"Bearer {_token(user_service, bob_a)}"}
    assert client.get("/kbs/kb_internal", headers=bob_a_headers).status_code == 200
    bob_list = client.get("/kbs", headers=bob_a_headers)
    assert [item["id"] for item in bob_list.json()["knowledge_bases"]] == [
        "kb_internal"
    ]
    assert (
        client.patch(
            "/kbs/kb_internal", json={"name": "Nope"}, headers=bob_a_headers
        ).status_code
        == 403
    )

    # Tenant membership (without user.tenant_id) implies kb_viewer as well.
    membership = client.put(
        f"/admin/tenants/tenant-a/members/{alice.id}",
        json={"role": "tenant_member"},
        headers=admin_headers,
    )
    assert membership.status_code == 200, membership.text
    assert client.get("/kbs/kb_internal", headers=alice_headers).status_code == 200

    # Clearing the tenant assignment revokes the implied access.
    cleared = client.patch(
        f"/admin/users/{bob.id}", json={"tenant_id": None}, headers=admin_headers
    )
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["tenant_id"] is None
    bob_cleared = asyncio.run(user_service.get_user_or_404(bob.id))
    bob_cleared_headers = {"Authorization": f"Bearer {_token(user_service, bob_cleared)}"}
    assert client.get("/kbs/kb_internal", headers=bob_cleared_headers).status_code == 403
    assert client.get("/kbs", headers=bob_cleared_headers).json()["knowledge_bases"] == []


def test_admin_patch_user_tenant_id_null_clears_and_empty_rejected(
    monkeypatch, tmp_path
):
    client, user_service, _authz, admin, _alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}

    assigned = client.patch(
        f"/admin/users/{bob.id}", json={"tenant_id": "tenant-x"}, headers=admin_headers
    )
    assert assigned.status_code == 200, assigned.text
    assert assigned.json()["tenant_id"] == "tenant-x"

    # Omitted tenant_id leaves the assignment unchanged.
    unrelated = client.patch(
        f"/admin/users/{bob.id}", json={"can_create_kb": True}, headers=admin_headers
    )
    assert unrelated.status_code == 200, unrelated.text
    assert unrelated.json()["tenant_id"] == "tenant-x"

    pre_clear = asyncio.run(user_service.get_user_or_404(bob.id))
    stale_headers = {"Authorization": f"Bearer {_token(user_service, pre_clear)}"}
    assert client.get("/auth/me", headers=stale_headers).status_code == 200

    # Explicit null clears the tenant and invalidates outstanding tokens.
    cleared = client.patch(
        f"/admin/users/{bob.id}", json={"tenant_id": None}, headers=admin_headers
    )
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["tenant_id"] is None
    assert cleared.json()["token_version"] == pre_clear.token_version + 1
    assert client.get("/auth/me", headers=stale_headers).status_code == 401

    # Empty/whitespace strings are rejected instead of storing a bogus tenant.
    assert (
        client.patch(
            f"/admin/users/{bob.id}", json={"tenant_id": ""}, headers=admin_headers
        ).status_code
        == 400
    )
    assert (
        client.patch(
            f"/admin/users/{bob.id}", json={"tenant_id": "   "}, headers=admin_headers
        ).status_code
        == 400
    )


def test_enterprise_stale_user_mutation_returns_conflict(monkeypatch, tmp_path):
    client, user_service, _authz, admin, _alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    stale_snapshot = asyncio.run(user_service.get_user_or_404(bob.id))

    async def writer_b():
        await user_service.update_user(
            bob.id,
            tenant_id="tenant-writer-b",
            can_create_kb=True,
            can_delete_documents=True,
            actor_user_id=admin.id,
        )
        return await user_service.change_password(
            bob.id, "writer-b-password", actor_user_id=admin.id
        )

    writer_b_state = asyncio.run(writer_b())
    original_get_user = user_service.get_user_or_404

    async def stale_get_user(user_id: str):
        if user_id == bob.id:
            return stale_snapshot
        return await original_get_user(user_id)

    monkeypatch.setattr(user_service, "get_user_or_404", stale_get_user)
    response = client.patch(
        f"/admin/users/{bob.id}",
        json={"can_use_bypass_query": True},
        headers=admin_headers,
    )
    assert response.status_code == 409, response.text
    assert "modified concurrently" in response.json()["detail"]

    current = asyncio.run(original_get_user(bob.id))
    assert current == writer_b_state
    assert current.tenant_id == "tenant-writer-b"
    assert current.can_create_kb is True
    assert current.can_delete_documents is True
    assert current.can_use_bypass_query is False
    assert asyncio.run(user_service.authenticate("bob", "bob-pass")) is None
    assert asyncio.run(user_service.authenticate("bob", "writer-b-password")) is not None


def test_enterprise_kb_create_is_hidden_until_final_generation_cas(
    monkeypatch, tmp_path
):
    from lightrag.api.enterprise_auth import KB_ROLE_VIEWER, principal_from_user

    client, user_service, authz, _admin, alice, _bob, _probe = (
        _build_enterprise_client(monkeypatch, tmp_path)
    )
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    principal = principal_from_user(
        alice,
        auth_method="jwt",
        memberships=[],
    )
    kb_service = client.app.state.kb_service
    store = client.app.state.metadata_store
    original_update = kb_service.update
    observed: dict[str, object] = {}

    async def update_after_provisioning(kb_id: str, **kwargs):
        if kwargs.get("status") == "active":
            current = await kb_service.get(kb_id)
            lifecycle = await store.get_kb_lifecycle(kb_id)
            owner_acl = await store.list_kb_acl(kb_id)
            events = await store.list_audit_events(
                target_type="kb",
                target_id=kb_id,
            )
            decision = await authz.resolve_kb_access(principal, current)
            observed.update(
                {
                    "status_before_final": current.status,
                    "expected_generation": kwargs.get("expected_generation"),
                    "generation": current.generation,
                    "lifecycle_state": lifecycle.state if lifecycle else None,
                    "owner_acl": [(item.user_id, item.role) for item in owner_acl],
                    "audit_complete": any(
                        event.event_type == "kb_created" for event in events
                    ),
                    "decision_role": decision.effective_role,
                    "filtered": await authz.filter_kbs_for_principal(
                        principal, [current]
                    ),
                }
            )
            with pytest.raises(HTTPException) as denied:
                await authz.require_kb_role(principal, kb_id, KB_ROLE_VIEWER)
            observed["query_denied"] = denied.value.status_code
        return await original_update(kb_id, **kwargs)

    monkeypatch.setattr(kb_service, "update", update_after_provisioning)
    response = client.post(
        "/kbs",
        json={"id": "kb_create_hidden", "name": "Create Hidden"},
        headers=alice_headers,
    )
    assert response.status_code == 200, response.text
    assert observed == {
        "status_before_final": "creating",
        "expected_generation": observed["generation"],
        "generation": observed["generation"],
        "lifecycle_state": "active",
        "owner_acl": [(alice.id, "kb_owner")],
        "audit_complete": True,
        "decision_role": None,
        "filtered": [],
        "query_denied": 403,
    }
    assert response.json()["status"] == "active"
    assert response.json()["origin"] == "platform"
    assert asyncio.run(kb_service.get("kb_create_hidden")).status == "active"


def test_enterprise_kb_create_failure_rolls_back_catalog_acl_and_tombstones(
    monkeypatch, tmp_path
):
    client, user_service, authz, admin, alice, _bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    store = client.app.state.metadata_store
    original_grant = authz.grant_kb_role
    captured_generations: list[str | None] = []

    async def grant_then_fail(*args, **kwargs):
        captured_generations.append(kwargs.get("expected_generation"))
        await original_grant(*args, **kwargs)
        current = await client.app.state.kb_service.get("kb_init_rollback")
        assert current.status == "creating"
        raise HTTPException(status_code=409, detail="simulated owner grant race")

    monkeypatch.setattr(authz, "grant_kb_role", grant_then_fail)
    failed = client.post(
        "/kbs",
        json={"id": "kb_init_rollback", "name": "Rollback"},
        headers=alice_headers,
    )
    assert failed.status_code == 409, failed.text
    assert failed.json()["detail"] == "simulated owner grant race"
    assert len(captured_generations) == 1
    failed_generation = captured_generations[0]
    assert failed_generation
    assert asyncio.run(store.list_kb_acl("kb_init_rollback")) == []
    tombstone = asyncio.run(store.get_kb_lifecycle("kb_init_rollback"))
    assert tombstone is not None
    assert tombstone.generation == failed_generation
    assert tombstone.state == "deleted"
    listed = client.get("/kbs?include_deleted=true", headers=admin_headers)
    assert listed.status_code == 200, listed.text
    assert all(
        item["id"] != "kb_init_rollback"
        for item in listed.json()["knowledge_bases"]
    )

    monkeypatch.setattr(authz, "grant_kb_role", original_grant)
    retried = client.post(
        "/kbs",
        json={"id": "kb_init_rollback", "name": "Retry"},
        headers=alice_headers,
    )
    assert retried.status_code == 200, retried.text
    active = asyncio.run(store.get_kb_lifecycle("kb_init_rollback"))
    assert active is not None
    assert active.state == "active"
    assert active.generation != failed_generation


def test_enterprise_acl_generation_is_captured_before_other_awaits(
    monkeypatch, tmp_path
):
    from lightrag.api.enterprise_auth import AuditService

    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "race-kbs.json")
    store = SQLiteMetadataStore(tmp_path / "metadata" / "race.sqlite3")
    audit = AuditService(store)
    users = UserService(store, audit)
    authz = AuthorizationService(store, audit, kb_service=kb_service)

    async def scenario():
        await kb_service.initialize()
        await store.initialize()
        user = await users.create_user(username="race-user", password="race-pass")
        record = await kb_service.create(kb_id="kb_acl_race", name="ACL Race")
        await store.activate_kb_generation(record.id, record.generation)
        original_get_user = store.get_enterprise_user_by_id
        replacement_generation = "replacement-generation"
        raced = False

        async def race_after_generation_capture(user_id: str):
            nonlocal raced
            if not raced:
                raced = True
                await store.purge_kb_metadata(
                    record.id,
                    generation=record.generation,
                )
                await store.activate_kb_generation(
                    record.id,
                    replacement_generation,
                )
            return await original_get_user(user_id)

        monkeypatch.setattr(
            store,
            "get_enterprise_user_by_id",
            race_after_generation_capture,
        )
        with pytest.raises(HTTPException) as exc:
            await authz.grant_kb_role(record.id, user.id, "kb_owner")
        assert exc.value.status_code == 409
        assert raced is True
        assert await store.list_kb_acl(record.id) == []
        lifecycle = await store.get_kb_lifecycle(record.id)
        assert lifecycle is not None
        assert lifecycle.generation == replacement_generation

        async def catalog_must_not_be_read(*_args, **_kwargs):
            raise AssertionError("catalog was re-read for an explicit generation")

        monkeypatch.setattr(kb_service, "get", catalog_must_not_be_read)
        with pytest.raises(HTTPException) as stale:
            await authz.grant_kb_role(
                record.id,
                user.id,
                "kb_owner",
                expected_generation=record.generation,
            )
        assert stale.value.status_code == 409

    asyncio.run(scenario())


def test_source_aware_kb_decision_lifecycle_and_generation_guard(tmp_path):
    from lightrag.api.enterprise_auth import (
        AuditService,
        AuthorizationService,
        KB_ROLE_ADMIN,
        KB_ROLE_EDITOR,
        KB_ROLE_VIEWER,
        UserService,
        principal_from_user,
    )
    from lightrag.api.kb_service import KnowledgeBaseService
    from lightrag.api.metadata_store import SQLiteMetadataStore

    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "decision-kbs.json")
    store = SQLiteMetadataStore(tmp_path / "metadata" / "decision.sqlite3")
    audit = AuditService(store)
    users = UserService(store, audit)
    authz = AuthorizationService(store, audit, kb_service=kb_service)

    async def scenario():
        await kb_service.initialize()
        await store.initialize()
        member = await users.create_user(username="member", password="member-pass")
        await authz.grant_tenant_membership(
            "tenant-a",
            member.id,
            "tenant_admin",
            granted_by="platform-admin",
        )
        member = await users.get_user_or_404(member.id)
        memberships = await store.list_user_tenant_memberships(member.id)
        principal = principal_from_user(
            member,
            auth_method="jwt",
            memberships=memberships,
        )

        provisioned = await kb_service.create(
            kb_id="kb_provisioned",
            name="Provisioned",
            tenant_id="tenant-a",
            origin="platform",
            metadata={
                "tenant_managed": True,
                "tenant_tag": "tenant:tenant-a",
                "tags": ["tenant:tenant-a"],
            },
        )
        await store.activate_kb_generation(
            provisioned.id,
            provisioned.generation,
        )
        assert not is_tenant_owned_kb(provisioned, "tenant-a")
        without_acl = await authz.resolve_kb_access(principal, provisioned)
        assert without_acl.effective_role is None
        assert without_acl.tenant_owned is False

        # These calls succeed only when AuthorizationService passes the loaded
        # catalog generation through to the transactional ACL methods.
        await authz.grant_tenant_kb_role(
            provisioned.id,
            "tenant-a",
            KB_ROLE_ADMIN,
            granted_by="platform-admin",
        )
        inherited = await authz.resolve_kb_access(principal, provisioned)
        assert inherited.effective_role == KB_ROLE_ADMIN
        assert inherited.tenant_role == KB_ROLE_ADMIN
        assert inherited.sources == ("tenant_acl",)

        await authz.grant_tenant_user_kb_override(
            provisioned.id,
            "tenant-a",
            member.id,
            KB_ROLE_VIEWER,
            granted_by="tenant-admin",
            expected_generation=provisioned.generation,
        )
        capped = await authz.resolve_kb_access(principal, provisioned)
        assert capped.effective_role == KB_ROLE_VIEWER
        assert capped.tenant_override_effect == "allow"
        assert capped.sources == ("tenant_override_capped",)

        assert await authz.revoke_tenant_kb_role(
            provisioned.id,
            "tenant-a",
            expected_generation=provisioned.generation,
        )
        orphaned_allow = await authz.resolve_kb_access(principal, provisioned)
        assert orphaned_allow.effective_role is None
        assert orphaned_allow.tenant_acl_role is None
        assert orphaned_allow.tenant_override_effect == "allow"
        assert orphaned_allow.tenant_owned is False
        await authz.grant_tenant_kb_role(
            provisioned.id,
            "tenant-a",
            KB_ROLE_ADMIN,
            granted_by="platform-admin",
            expected_generation=provisioned.generation,
        )

        await authz.revoke_tenant_user_kb_override(
            provisioned.id,
            "tenant-a",
            member.id,
            granted_by="tenant-admin",
            expected_generation=provisioned.generation,
        )
        denied = await authz.resolve_kb_access(principal, provisioned)
        assert denied.effective_role is None
        assert denied.tenant_override_effect == "deny"

        await authz.grant_kb_role(
            provisioned.id,
            member.id,
            KB_ROLE_EDITOR,
            granted_by="platform-admin",
        )
        direct_survives_deny = await authz.resolve_kb_access(principal, provisioned)
        assert direct_survives_deny.platform_role == KB_ROLE_EDITOR
        assert direct_survives_deny.effective_role == KB_ROLE_EDITOR
        assert direct_survives_deny.sources == ("direct",)
        assert await authz.revoke_kb_role(provisioned.id, member.id)

        # Tenant-owned KBs do not inherit the tenant ACL. Members need an allow
        # override; public visibility remains an independent platform source.
        owned = await kb_service.create(
            kb_id="kb_tenant_owned",
            name="Tenant Owned",
            tenant_id="tenant-a",
            origin="tenant",
        )
        await store.activate_kb_generation(owned.id, owned.generation)
        await authz.grant_tenant_kb_role(
            owned.id,
            "tenant-a",
            KB_ROLE_ADMIN,
            granted_by="platform-admin",
        )
        no_owned_inheritance = await authz.resolve_kb_access(principal, owned)
        assert is_tenant_owned_kb(owned, "tenant-a")
        assert no_owned_inheritance.tenant_owned is True
        assert no_owned_inheritance.effective_role is None
        await authz.grant_tenant_user_kb_override(
            owned.id,
            "tenant-a",
            member.id,
            KB_ROLE_EDITOR,
            granted_by="tenant-admin",
            expected_generation=owned.generation,
        )
        owned_allowed = await authz.resolve_kb_access(principal, owned)
        assert owned_allowed.effective_role == KB_ROLE_EDITOR
        assert owned_allowed.sources == ("tenant_owned_override",)

        assert await authz.authorize_kb_lifecycle(principal, owned, "hard-delete")
        with pytest.raises(HTTPException) as exc:
            await authz.authorize_kb_lifecycle(principal, provisioned, "delete")
        assert exc.value.status_code == 403

        assert await authz.revoke_tenant_kb_role(owned.id, "tenant-a")

    asyncio.run(scenario())


def test_enterprise_kb_restore_tenant_admin_and_rebuild_editor_allowed(
    monkeypatch, tmp_path
):
    client, user_service, authz, admin, alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    asyncio.run(
        authz.grant_tenant_membership(
            "tenant-restore",
            alice.id,
            "tenant_admin",
            granted_by=admin.id,
        )
    )
    alice = asyncio.run(user_service.get_user_or_404(alice.id))
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}

    created = client.post(
        "/kbs", json={"id": "kb_restorable", "name": "Restorable"}, headers=alice_headers
    )
    assert created.status_code == 200, created.text

    # Middleware regression: the ":action" suffix used to leak into the
    # extracted kb_id ("kb_restorable:rebuild"), making :rebuild 403 for every
    # non-super-admin. An editor must reach the route (empty KB -> no-op 200).
    grant = client.put(
        "/admin/kbs/kb_restorable/acl",
        json={"user_id": bob.id, "role": "kb_editor"},
        headers=admin_headers,
    )
    assert grant.status_code == 200, grant.text
    rebuild = client.post("/kbs/kb_restorable:rebuild", json={}, headers=bob_headers)
    assert rebuild.status_code == 200, rebuild.text
    assert rebuild.json()["documents"] == []

    deleted = client.delete("/kbs/kb_restorable", headers=alice_headers)
    assert deleted.status_code == 200, deleted.text

    # A canonical tenant administrator can restore a genuinely tenant-created
    # KB; this no longer relies on a middleware super-admin shortcut.
    restored = client.post("/kbs/kb_restorable:restore", headers=alice_headers)
    assert restored.status_code == 200, restored.text
    assert restored.json()["status"] == "active"
    assert client.get("/kbs/kb_restorable", headers=alice_headers).status_code == 200

    events = client.get("/admin/audit-events", headers=admin_headers)
    assert any(event["event_type"] == "kb_restored" for event in events.json())


def test_auth_me_profile_patch_and_logout(monkeypatch, tmp_path):
    client, user_service, _authz, admin, _alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path, api_key=None
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}

    # Profile fields default to null and PATCH sets them without invalidating
    # the current token (token_version untouched).
    me_before = client.get("/auth/me", headers=bob_headers)
    assert me_before.status_code == 200
    assert me_before.json()["user"]["display_name"] is None
    version_before = me_before.json()["user"]["token_version"]

    patched = client.patch(
        "/auth/me",
        json={"display_name": "  Bob B.  ", "email": "bob@corp.local"},
        headers=bob_headers,
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["user"]["display_name"] == "Bob B."
    assert patched.json()["user"]["email"] == "bob@corp.local"
    assert patched.json()["user"]["token_version"] == version_before
    assert client.get("/auth/me", headers=bob_headers).status_code == 200

    # Omitted fields stay; explicit null clears; bad values are rejected.
    cleared = client.patch("/auth/me", json={"email": None}, headers=bob_headers)
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["user"]["email"] is None
    assert cleared.json()["user"]["display_name"] == "Bob B."
    assert (
        client.patch("/auth/me", json={"email": "not-an-email"}, headers=bob_headers).status_code
        == 400
    )
    assert (
        client.patch("/auth/me", json={"display_name": "   "}, headers=bob_headers).status_code
        == 400
    )

    # Logout bumps token_version: the very token used is rejected afterwards.
    logged_out = client.post("/auth/logout", headers=bob_headers)
    assert logged_out.status_code == 200, logged_out.text
    assert logged_out.json()["status"] == "logged_out"
    assert client.get("/auth/me", headers=bob_headers).status_code == 401

    bob_after = asyncio.run(user_service.get_user_or_404(bob.id))
    fresh_headers = {"Authorization": f"Bearer {_token(user_service, bob_after)}"}
    assert client.get("/auth/me", headers=fresh_headers).status_code == 200

    events = client.get("/admin/audit-events", headers=admin_headers)
    event_types = [event["event_type"] for event in events.json()]
    assert "user_profile_updated" in event_types
    assert "user_logged_out" in event_types

    # Service keys have no interactive session: both endpoints refuse them.
    key_created = client.post(
        "/admin/service-api-keys",
        json={"name": "svc", "kb_roles": {}},
        headers=admin_headers,
    )
    assert key_created.status_code == 200, key_created.text
    service_headers = {"X-API-Key": key_created.json()["api_key"]}
    assert client.post("/auth/logout", headers=service_headers).status_code == 403
    assert (
        client.patch(
            "/auth/me", json={"display_name": "svc"}, headers=service_headers
        ).status_code
        == 403
    )


def test_admin_overview_aggregates_platform(monkeypatch, tmp_path):
    client, user_service, _authz, admin, alice, _bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}

    created = client.post(
        "/kbs", json={"id": "kb_ov", "name": "Overview"}, headers=alice_headers
    )
    assert created.status_code == 200, created.text
    texts = client.post(
        "/kbs/kb_ov/documents:texts",
        json={"documents": [{"text": "hello overview", "source_name": "n.md"}]},
        headers=alice_headers,
    )
    assert texts.status_code == 200, texts.text

    overview = client.get("/admin/overview", headers=admin_headers)
    assert overview.status_code == 200, overview.text
    body = overview.json()
    assert body["kbs"]["by_status"].get("active", 0) >= 1
    assert body["kbs"]["total"] >= 1
    assert body["documents"]["total"] >= 1
    assert body["jobs"]["total"] >= 1
    assert body["counters"] == {"chunks": 0, "entities": 0, "relations": 0}
    # admin + alice + bob from the harness seed.
    assert body["enterprise"]["users_by_status"].get("active", 0) >= 3
    assert body["enterprise"]["audit_events"] >= 1

    # /admin prefix stays super-admin gated.
    assert client.get("/admin/overview", headers=alice_headers).status_code == 403


def test_enterprise_kb_graph_write_requires_admin(monkeypatch, tmp_path):
    # The graph-write busy-guard reads shared pipeline state; bootstrap it
    # like the real server lifespan does.
    initialize_share_data()
    try:
        _run_enterprise_kb_graph_write_requires_admin(monkeypatch, tmp_path)
    finally:
        finalize_share_data()


def _run_enterprise_kb_graph_write_requires_admin(monkeypatch, tmp_path):
    client, user_service, _authz, admin, alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}

    created = client.post(
        "/kbs", json={"id": "kb_graph_rbac", "name": "Graph RBAC"}, headers=alice_headers
    )
    assert created.status_code == 200, created.text

    edit_payload = {
        "entity_name": "Tesla",
        "updated_data": {"description": "fixed description"},
    }

    # Graph reads stay viewer-level; graph writes escalate to kb_admin.
    grant_viewer = client.put(
        "/admin/kbs/kb_graph_rbac/acl",
        json={"user_id": bob.id, "role": "kb_viewer"},
        headers=admin_headers,
    )
    assert grant_viewer.status_code == 200, grant_viewer.text
    assert (
        client.get("/kbs/kb_graph_rbac/graph/entities", headers=bob_headers).status_code
        == 200
    )
    assert (
        client.post(
            "/kbs/kb_graph_rbac/graph/entity:edit",
            json=edit_payload,
            headers=bob_headers,
        ).status_code
        == 403
    )

    grant_editor = client.put(
        "/admin/kbs/kb_graph_rbac/acl",
        json={"user_id": bob.id, "role": "kb_editor"},
        headers=admin_headers,
    )
    assert grant_editor.status_code == 200, grant_editor.text
    assert (
        client.post(
            "/kbs/kb_graph_rbac/graph/entity:edit",
            json=edit_payload,
            headers=bob_headers,
        ).status_code
        == 403
    )

    # The kb_owner (rank above kb_admin) may curate the graph; audited.
    edited = client.post(
        "/kbs/kb_graph_rbac/graph/entity:edit",
        json=edit_payload,
        headers=alice_headers,
    )
    assert edited.status_code == 200, edited.text
    assert edited.json()["data"]["entity_name"] == "Tesla"

    events = client.get("/admin/audit-events", headers=admin_headers)
    assert any(
        event["event_type"] == "kb_graph_entity_edited" for event in events.json()
    )


def test_tenant_overview_active_union_fallback_and_exact_admin_detail(
    monkeypatch, tmp_path
):
    client, user_service, authz, admin, alice, bob, _probe = _build_enterprise_client(
        monkeypatch,
        tmp_path,
        api_key=None,
        inject_enterprise_router_kb_service=False,
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    for tenant_id in ("tenant-a", "tenant-b"):
        response = client.post(
            "/admin/tenants",
            json={"name": tenant_id, "tenant_id": tenant_id},
            headers=admin_headers,
        )
        assert response.status_code == 200, response.text
    asyncio.run(
        authz.grant_tenant_membership(
            "tenant-a", alice.id, "tenant_admin", granted_by=admin.id
        )
    )
    asyncio.run(
        authz.grant_tenant_membership(
            "tenant-b", bob.id, "tenant_admin", granted_by=admin.id
        )
    )
    alice = asyncio.run(user_service.get_user_or_404(alice.id))
    bob = asyncio.run(user_service.get_user_or_404(bob.id))
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}

    owned = client.post(
        "/kbs",
        json={"id": "kb_tenant_owned", "name": "Owned"},
        headers=alice_headers,
    )
    assert owned.status_code == 200, owned.text
    provisioned = client.post(
        "/kbs",
        json={"id": "kb_tenant_assigned", "name": "Assigned"},
        headers=admin_headers,
    )
    assert provisioned.status_code == 200, provisioned.text
    deleted = client.post(
        "/kbs",
        json={"id": "kb_tenant_deleted", "name": "Deleted"},
        headers=admin_headers,
    )
    assert deleted.status_code == 200, deleted.text
    for kb_id in ("kb_tenant_owned", "kb_tenant_assigned", "kb_tenant_deleted"):
        grant = client.put(
            f"/admin/kbs/{kb_id}/acl",
            json={"tenant_id": "tenant-a", "role": "kb_viewer"},
            headers=admin_headers,
        )
        assert grant.status_code == 200, grant.text
    removed = client.delete("/kbs/kb_tenant_deleted", headers=admin_headers)
    assert removed.status_code == 200, removed.text

    # The router closure intentionally has no KB service. Production-style
    # app.state fallback must still count owned + assigned exactly once.
    scoped = client.get("/tenants/tenant-a", headers=alice_headers)
    assert scoped.status_code == 200, scoped.text
    assert scoped.json()["kb_count"] == 2
    scoped_kbs = client.get("/tenants/tenant-a/kbs", headers=alice_headers)
    assert scoped_kbs.status_code == 200, scoped_kbs.text
    assert {item["id"] for item in scoped_kbs.json()} == {
        "kb_tenant_owned",
        "kb_tenant_assigned",
    }

    exact_admin = client.get("/admin/tenants/tenant-a", headers=alice_headers)
    assert exact_admin.status_code == 200, exact_admin.text
    assert exact_admin.json()["kb_count"] == 2
    assert client.get("/admin/tenants/tenant-b", headers=alice_headers).status_code == 403
    assert client.get("/admin/tenants/tenant-a", headers=bob_headers).status_code == 403

    # No prefix/subpath broadening: only the exact detail route is deferred.
    for path in (
        "/admin/tenants/tenant-a/kbs",
        "/admin/tenants/tenant-a/members",
        "/admin/tenants",
    ):
        assert client.get(path, headers=alice_headers).status_code == 403
    for path in (
        "/admin/tenants/tenant-a/query",
        "/admin/tenants/tenant-a%2Fkbs",
    ):
        assert client.get(path, headers=alice_headers).status_code in {403, 404}


def test_tenant_scoped_user_lifecycle_capabilities_and_boundaries(
    monkeypatch, tmp_path
):
    client, user_service, authz, admin, alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path, api_key=None
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    for tenant_id in ("tenant-users", "tenant-other"):
        created_tenant = client.post(
            "/admin/tenants",
            json={"name": tenant_id, "tenant_id": tenant_id},
            headers=admin_headers,
        )
        assert created_tenant.status_code == 200, created_tenant.text
    asyncio.run(
        authz.grant_tenant_membership(
            "tenant-users", alice.id, "tenant_admin", granted_by=admin.id
        )
    )
    asyncio.run(
        authz.grant_tenant_membership(
            "tenant-other", bob.id, "tenant_member", granted_by=admin.id
        )
    )
    alice = asyncio.run(user_service.get_user_or_404(alice.id))
    bob = asyncio.run(user_service.get_user_or_404(bob.id))
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}

    wrong_tenant = client.post(
        "/tenants/tenant-users/users",
        json={
            "username": "wrong",
            "password": "pass",
            "tenant_id": "tenant-other",
        },
        headers=alice_headers,
    )
    assert wrong_tenant.status_code == 400

    created = client.post(
        "/tenants/tenant-users/users",
        json={
            "username": "tenant-user",
            "password": "initial-pass",
            "can_create_kb": True,
            "can_use_bypass_query": True,
            "can_use_agent_query": True,
            "can_delete_documents": True,
            "can_download_files": True,
        },
        headers=alice_headers,
    )
    assert created.status_code == 200, created.text
    payload = created.json()
    user_id = payload["id"]
    assert payload["system_role"] == "user"
    assert payload["tenant_id"] == "tenant-users"
    for capability in (
        "can_create_kb",
        "can_use_bypass_query",
        "can_use_agent_query",
        "can_delete_documents",
        "can_download_files",
    ):
        assert payload[capability] is True

    target = asyncio.run(user_service.get_user_or_404(user_id))
    stale_headers = {"Authorization": f"Bearer {_token(user_service, target)}"}
    listed = client.get("/tenants/tenant-users/users", headers=alice_headers)
    assert listed.status_code == 200, listed.text
    assert user_id in {item["id"] for item in listed.json()}
    detail = client.get(
        f"/tenants/tenant-users/users/{user_id}", headers=alice_headers
    )
    assert detail.status_code == 200, detail.text

    updated = client.patch(
        f"/tenants/tenant-users/users/{user_id}",
        json={
            "can_create_kb": False,
            "can_use_bypass_query": False,
            "can_use_agent_query": False,
            "can_delete_documents": False,
            "can_download_files": False,
        },
        headers=alice_headers,
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["token_version"] == target.token_version + 1
    assert client.get("/auth/me", headers=stale_headers).status_code == 401
    assert all(
        updated.json()[field] is False
        for field in (
            "can_create_kb",
            "can_use_bypass_query",
            "can_use_agent_query",
            "can_delete_documents",
            "can_download_files",
        )
    )
    forbidden_patch = client.patch(
        f"/tenants/tenant-users/users/{user_id}",
        json={"tenant_id": "tenant-other"},
        headers=alice_headers,
    )
    assert forbidden_patch.status_code == 400

    current_target = asyncio.run(user_service.get_user_or_404(user_id))
    pre_disable_headers = {
        "Authorization": f"Bearer {_token(user_service, current_target)}"
    }
    disabled = client.post(
        f"/tenants/tenant-users/users/{user_id}:disable",
        headers=alice_headers,
    )
    assert disabled.status_code == 200, disabled.text
    assert disabled.json()["status"] == "disabled"
    assert client.get("/auth/me", headers=pre_disable_headers).status_code == 401
    enabled = client.post(
        f"/tenants/tenant-users/users/{user_id}:enable",
        headers=alice_headers,
    )
    assert enabled.status_code == 200, enabled.text
    assert enabled.json()["status"] == "active"
    reset = client.post(
        f"/tenants/tenant-users/users/{user_id}:reset-password",
        json={"password": "reset-pass"},
        headers=alice_headers,
    )
    assert reset.status_code == 200, reset.text
    assert asyncio.run(user_service.authenticate("tenant-user", "initial-pass")) is None
    assert asyncio.run(user_service.authenticate("tenant-user", "reset-pass")) is not None

    # Missing, cross-tenant, and super-admin targets are indistinguishable.
    for hidden_id in ("usr_missing", bob.id, admin.id):
        hidden = client.get(
            f"/tenants/tenant-users/users/{hidden_id}", headers=alice_headers
        )
        assert hidden.status_code == 404
    assert (
        client.post(
            f"/tenants/tenant-users/users/{alice.id}:disable",
            headers=alice_headers,
        ).status_code
        == 409
    )
    assert (
        client.delete(
            f"/tenants/tenant-users/users/{alice.id}", headers=alice_headers
        ).status_code
        == 409
    )

    service_key = client.post(
        "/admin/service-api-keys",
        json={"name": "tenant-admin-denied"},
        headers=admin_headers,
    )
    assert service_key.status_code == 200, service_key.text
    assert (
        client.get(
            "/tenants/tenant-users/users",
            headers={"X-API-Key": service_key.json()["api_key"]},
        ).status_code
        == 403
    )

    deleted = client.delete(
        f"/tenants/tenant-users/users/{user_id}", headers=alice_headers
    )
    assert deleted.status_code == 200, deleted.text
    assert deleted.json() == {"deleted": True}
    scoped_audit = client.get(
        "/tenants/tenant-users/audit-events",
        params={"event_type": "user_deleted", "target_id": user_id},
        headers=alice_headers,
    )
    assert scoped_audit.status_code == 200, scoped_audit.text
    assert len(scoped_audit.json()) == 1
    assert scoped_audit.json()[0]["actor_user_id"] == alice.id
    assert (
        client.get(
            f"/tenants/tenant-users/users/{user_id}", headers=alice_headers
        ).status_code
        == 404
    )

    # Existing super-admin lifecycle remains compatible with the fifth flag.
    admin_created = client.post(
        "/admin/users",
        json={
            "username": "platform-user",
            "password": "platform-pass",
            "can_download_files": True,
        },
        headers=admin_headers,
    )
    assert admin_created.status_code == 200, admin_created.text
    assert admin_created.json()["can_download_files"] is True


def test_tenant_scoped_mutations_hide_protected_targets_and_block_self(
    monkeypatch, tmp_path
):
    client, user_service, authz, admin, alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path, api_key=None
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    for tenant_id in ("tenant-protected", "tenant-cross"):
        response = client.post(
            "/admin/tenants",
            json={"name": tenant_id, "tenant_id": tenant_id},
            headers=admin_headers,
        )
        assert response.status_code == 200, response.text

    async def seed_targets():
        tenant_admin = await user_service.create_user(
            username="protected-admin", password="protected-admin-pass"
        )
        tenant_owner = await user_service.create_user(
            username="protected-owner", password="protected-owner-pass"
        )
        await authz.grant_tenant_membership(
            "tenant-protected", alice.id, "tenant_admin", granted_by=admin.id
        )
        await authz.grant_tenant_membership(
            "tenant-protected",
            tenant_admin.id,
            "tenant_admin",
            granted_by=admin.id,
        )
        await authz.grant_tenant_membership(
            "tenant-protected",
            tenant_owner.id,
            "tenant_owner",
            granted_by=admin.id,
        )
        await authz.grant_tenant_membership(
            "tenant-cross", bob.id, "tenant_member", granted_by=admin.id
        )
        return (
            await user_service.get_user_or_404(alice.id),
            await user_service.get_user_or_404(tenant_admin.id),
            await user_service.get_user_or_404(tenant_owner.id),
            await user_service.get_user_or_404(bob.id),
        )

    alice, tenant_admin, tenant_owner, bob = asyncio.run(seed_targets())
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    created_kb = client.post(
        "/kbs",
        json={"id": "kb_protected_targets", "name": "Protected targets"},
        headers=alice_headers,
    )
    assert created_kb.status_code == 200, created_kb.text
    kb_member_path = (
        "/tenants/tenant-protected/kbs/kb_protected_targets/members"
    )

    self_before = asyncio.run(user_service.get_user_or_404(alice.id))
    self_patch = client.patch(
        f"/tenants/tenant-protected/users/{alice.id}",
        json={
            "can_create_kb": not self_before.can_create_kb,
            "can_use_bypass_query": not self_before.can_use_bypass_query,
            "can_use_agent_query": not self_before.can_use_agent_query,
            "can_delete_documents": not self_before.can_delete_documents,
            "can_download_files": not self_before.can_download_files,
        },
        headers=alice_headers,
    )
    assert self_patch.status_code == 409
    assert asyncio.run(user_service.get_user_or_404(alice.id)) == self_before
    assert (
        client.post(
            f"/tenants/tenant-protected/users/{alice.id}:reset-password",
            json={"password": "must-not-change"},
            headers=alice_headers,
        ).status_code
        == 409
    )
    assert asyncio.run(user_service.authenticate("alice", "alice-pass")) is not None
    assert (
        client.put(
            f"{kb_member_path}/{alice.id}",
            json={"role": "viewer"},
            headers=alice_headers,
        ).status_code
        == 409
    )
    assert (
        client.delete(
            f"{kb_member_path}/{alice.id}", headers=alice_headers
        ).status_code
        == 409
    )
    assert (
        client.put(
            f"/tenants/tenant-protected/members/{alice.id}",
            json={"role": "tenant_member"},
            headers=alice_headers,
        ).status_code
        == 409
    )

    listed = client.get(
        "/tenants/tenant-protected/users", headers=alice_headers
    )
    assert listed.status_code == 200, listed.text
    assert {tenant_admin.id, tenant_owner.id}.issubset(
        {item["id"] for item in listed.json()}
    )

    passwords = {
        tenant_admin.id: ("protected-admin", "protected-admin-pass"),
        tenant_owner.id: ("protected-owner", "protected-owner-pass"),
    }
    for target in (tenant_admin, tenant_owner):
        detail = client.get(
            f"/tenants/tenant-protected/users/{target.id}", headers=alice_headers
        )
        assert detail.status_code == 200, detail.text
        before = asyncio.run(user_service.get_user_or_404(target.id))
        responses = [
            client.patch(
                f"/tenants/tenant-protected/users/{target.id}",
                json={"can_create_kb": True},
                headers=alice_headers,
            ),
            client.post(
                f"/tenants/tenant-protected/users/{target.id}:reset-password",
                json={"password": "must-not-change"},
                headers=alice_headers,
            ),
            client.post(
                f"/tenants/tenant-protected/users/{target.id}:disable",
                headers=alice_headers,
            ),
            client.post(
                f"/tenants/tenant-protected/users/{target.id}:enable",
                headers=alice_headers,
            ),
            client.delete(
                f"/tenants/tenant-protected/users/{target.id}",
                headers=alice_headers,
            ),
            client.put(
                f"{kb_member_path}/{target.id}",
                json={"role": "viewer"},
                headers=alice_headers,
            ),
            client.delete(
                f"{kb_member_path}/{target.id}", headers=alice_headers
            ),
            client.put(
                f"/tenants/tenant-protected/members/{target.id}",
                json={"role": "tenant_member"},
                headers=alice_headers,
            ),
            client.delete(
                f"/tenants/tenant-protected/members/{target.id}",
                headers=alice_headers,
            ),
        ]
        assert {response.status_code for response in responses} == {404}
        assert asyncio.run(user_service.get_user_or_404(target.id)) == before
        username, password = passwords[target.id]
        assert asyncio.run(user_service.authenticate(username, password)) is not None
        assert (
            asyncio.run(
                client.app.state.metadata_store.get_tenant_user_kb_override(
                    "tenant-protected", "kb_protected_targets", target.id
                )
            )
            is None
        )

    for hidden_id in ("usr_missing", bob.id, admin.id):
        hidden_responses = [
            client.patch(
                f"/tenants/tenant-protected/users/{hidden_id}",
                json={"can_download_files": True},
                headers=alice_headers,
            ),
            client.post(
                f"/tenants/tenant-protected/users/{hidden_id}:reset-password",
                json={"password": "hidden"},
                headers=alice_headers,
            ),
            client.post(
                f"/tenants/tenant-protected/users/{hidden_id}:disable",
                headers=alice_headers,
            ),
            client.delete(
                f"/tenants/tenant-protected/users/{hidden_id}",
                headers=alice_headers,
            ),
            client.put(
                f"{kb_member_path}/{hidden_id}",
                json={"role": "viewer"},
                headers=alice_headers,
            ),
            client.put(
                f"/tenants/tenant-protected/members/{hidden_id}",
                json={"role": "tenant_member"},
                headers=alice_headers,
            ),
            client.delete(
                f"/tenants/tenant-protected/members/{hidden_id}",
                headers=alice_headers,
            ),
        ]
        assert {response.status_code for response in hidden_responses} == {404}


def test_tenant_scoped_user_and_membership_cas_reject_post_validation_races(
    monkeypatch, tmp_path
):
    client, user_service, authz, admin, alice, _bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path, api_key=None
    )

    async def seed():
        await authz.grant_tenant_membership(
            "tenant-race-a", alice.id, "tenant_admin", granted_by=admin.id
        )
        update_target = await user_service.create_user(
            username="race-update",
            password="race-update-pass",
            tenant_id="tenant-race-a",
        )
        password_target = await user_service.create_user(
            username="race-password",
            password="race-password-pass",
            tenant_id="tenant-race-a",
        )
        delete_target = await user_service.create_user(
            username="race-delete",
            password="race-delete-pass",
            tenant_id="tenant-race-a",
        )
        grant_target = await user_service.create_user(
            username="race-grant", password="race-grant-pass"
        )
        revoke_target = await user_service.create_user(
            username="race-revoke",
            password="race-revoke-pass",
            tenant_id="tenant-race-a",
        )
        return (
            await user_service.get_user_or_404(alice.id),
            update_target,
            password_target,
            delete_target,
            grant_target,
            revoke_target,
        )

    (
        alice,
        update_target,
        password_target,
        delete_target,
        grant_target,
        revoke_target,
    ) = asyncio.run(seed())
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}

    original_update = user_service.update_user

    async def update_after_move(user_id, **kwargs):
        await authz.grant_tenant_membership(
            "tenant-race-b", user_id, "tenant_member", granted_by=admin.id
        )
        return await original_update(user_id, **kwargs)

    monkeypatch.setattr(user_service, "update_user", update_after_move)
    stale_update = client.patch(
        f"/tenants/tenant-race-a/users/{update_target.id}",
        json={"can_create_kb": True},
        headers=alice_headers,
    )
    assert stale_update.status_code == 409, stale_update.text
    moved = asyncio.run(user_service.get_user_or_404(update_target.id))
    assert moved.tenant_id == "tenant-race-b"
    assert moved.can_create_kb is False
    monkeypatch.setattr(user_service, "update_user", original_update)

    original_change_password = user_service.change_password

    async def password_after_promotion(user_id, password, **kwargs):
        await authz.grant_tenant_membership(
            "tenant-race-a", user_id, "tenant_admin", granted_by=admin.id
        )
        return await original_change_password(user_id, password, **kwargs)

    monkeypatch.setattr(
        user_service, "change_password", password_after_promotion
    )
    stale_password = client.post(
        f"/tenants/tenant-race-a/users/{password_target.id}:reset-password",
        json={"password": "stale-new-password"},
        headers=alice_headers,
    )
    assert stale_password.status_code == 409, stale_password.text
    assert (
        asyncio.run(user_service.authenticate("race-password", "race-password-pass"))
        is not None
    )
    assert (
        asyncio.run(user_service.authenticate("race-password", "stale-new-password"))
        is None
    )
    promoted = asyncio.run(
        authz.get_tenant_membership("tenant-race-a", password_target.id)
    )
    assert promoted is not None and promoted.role == "tenant_admin"
    monkeypatch.setattr(
        user_service, "change_password", original_change_password
    )

    original_delete = user_service.delete_user

    async def delete_after_revision(user_id, **kwargs):
        await original_update(
            user_id, can_create_kb=True, actor_user_id=admin.id
        )
        return await original_delete(user_id, **kwargs)

    monkeypatch.setattr(user_service, "delete_user", delete_after_revision)
    stale_delete = client.delete(
        f"/tenants/tenant-race-a/users/{delete_target.id}",
        headers=alice_headers,
    )
    assert stale_delete.status_code == 409, stale_delete.text
    retained = asyncio.run(user_service.get_user_or_404(delete_target.id))
    assert retained.can_create_kb is True
    monkeypatch.setattr(user_service, "delete_user", original_delete)

    original_grant = authz.grant_tenant_membership

    async def grant_after_move(tenant_id, user_id, role, **kwargs):
        await original_grant(
            "tenant-race-b", user_id, "tenant_member", granted_by=admin.id
        )
        return await original_grant(tenant_id, user_id, role, **kwargs)

    monkeypatch.setattr(authz, "grant_tenant_membership", grant_after_move)
    stale_grant = client.put(
        f"/tenants/tenant-race-a/members/{grant_target.id}",
        json={"role": "tenant_member"},
        headers=alice_headers,
    )
    assert stale_grant.status_code == 409, stale_grant.text
    grant_current = asyncio.run(user_service.get_user_or_404(grant_target.id))
    assert grant_current.tenant_id == "tenant-race-b"
    monkeypatch.setattr(authz, "grant_tenant_membership", original_grant)

    original_revoke = authz.revoke_tenant_membership

    async def revoke_after_promotion(tenant_id, user_id, **kwargs):
        await original_grant(
            tenant_id, user_id, "tenant_admin", granted_by=admin.id
        )
        return await original_revoke(tenant_id, user_id, **kwargs)

    monkeypatch.setattr(authz, "revoke_tenant_membership", revoke_after_promotion)
    stale_revoke = client.delete(
        f"/tenants/tenant-race-a/members/{revoke_target.id}",
        headers=alice_headers,
    )
    assert stale_revoke.status_code == 409, stale_revoke.text
    revoke_current = asyncio.run(user_service.get_user_or_404(revoke_target.id))
    revoke_membership = asyncio.run(
        authz.get_tenant_membership("tenant-race-a", revoke_target.id)
    )
    assert revoke_current.tenant_id == "tenant-race-a"
    assert revoke_membership is not None and revoke_membership.role == "tenant_admin"


def test_tenant_kb_override_target_cas_rejects_route_races_without_enumeration(
    monkeypatch, tmp_path
):
    client, user_service, authz, admin, alice, _bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path, api_key=None
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    for tenant_id in ("tenant-override-race-a", "tenant-override-race-b"):
        created = client.post(
            "/admin/tenants",
            json={"name": tenant_id, "tenant_id": tenant_id},
            headers=admin_headers,
        )
        assert created.status_code == 200, created.text

    async def seed():
        await authz.grant_tenant_membership(
            "tenant-override-race-a",
            alice.id,
            "tenant_admin",
            granted_by=admin.id,
        )
        promoted_target = await user_service.create_user(
            username="override-race-promote",
            password="pass",
            tenant_id="tenant-override-race-a",
        )
        moved_target = await user_service.create_user(
            username="override-race-move",
            password="pass",
            tenant_id="tenant-override-race-a",
        )
        revised_target = await user_service.create_user(
            username="override-race-revision",
            password="pass",
            tenant_id="tenant-override-race-a",
        )
        return (
            await user_service.get_user_or_404(alice.id),
            promoted_target,
            moved_target,
            revised_target,
        )

    alice, promoted_target, moved_target, revised_target = asyncio.run(seed())
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    created_kb = client.post(
        "/kbs",
        json={"id": "kb_override_target_race", "name": "Override race"},
        headers=alice_headers,
    )
    assert created_kb.status_code == 200, created_kb.text
    member_path = (
        "/tenants/tenant-override-race-a/kbs/"
        "kb_override_target_race/members"
    )
    store = client.app.state.metadata_store

    original_grant_override = authz.grant_tenant_user_kb_override

    async def grant_after_promotion(kb_id, tenant_id, user_id, role, **kwargs):
        await authz.grant_tenant_membership(
            tenant_id,
            user_id,
            "tenant_admin",
            granted_by=admin.id,
        )
        return await original_grant_override(
            kb_id, tenant_id, user_id, role, **kwargs
        )

    monkeypatch.setattr(
        authz, "grant_tenant_user_kb_override", grant_after_promotion
    )
    promoted = client.put(
        f"{member_path}/{promoted_target.id}",
        json={"role": "viewer"},
        headers=alice_headers,
    )
    assert promoted.status_code == 404, promoted.text
    assert promoted.json() == {"detail": "User not found"}
    assert "tenant-override-race-b" not in promoted.text
    assert (
        asyncio.run(
            store.get_tenant_user_kb_override(
                "tenant-override-race-a",
                "kb_override_target_race",
                promoted_target.id,
            )
        )
        is None
    )
    monkeypatch.setattr(
        authz, "grant_tenant_user_kb_override", original_grant_override
    )

    original_revoke_override = authz.revoke_tenant_user_kb_override

    async def revoke_after_move(kb_id, tenant_id, user_id, **kwargs):
        await authz.grant_tenant_membership(
            "tenant-override-race-b",
            user_id,
            "tenant_member",
            granted_by=admin.id,
        )
        return await original_revoke_override(kb_id, tenant_id, user_id, **kwargs)

    monkeypatch.setattr(
        authz, "revoke_tenant_user_kb_override", revoke_after_move
    )
    moved = client.delete(
        f"{member_path}/{moved_target.id}", headers=alice_headers
    )
    assert moved.status_code == 404, moved.text
    assert moved.json() == {"detail": "User not found"}
    assert "tenant-override-race-b" not in moved.text
    assert (
        asyncio.run(
            store.get_tenant_user_kb_override(
                "tenant-override-race-a",
                "kb_override_target_race",
                moved_target.id,
            )
        )
        is None
    )
    monkeypatch.setattr(
        authz, "revoke_tenant_user_kb_override", original_revoke_override
    )

    baseline = asyncio.run(
        authz.grant_tenant_user_kb_override(
            "kb_override_target_race",
            "tenant-override-race-a",
            revised_target.id,
            "kb_viewer",
            granted_by=admin.id,
        )
    )
    original_reset_override = authz.reset_tenant_user_kb_override

    async def reset_after_revision(kb_id, tenant_id, user_id, **kwargs):
        await user_service.update_user(
            user_id,
            can_create_kb=True,
            actor_user_id=admin.id,
        )
        return await original_reset_override(kb_id, tenant_id, user_id, **kwargs)

    monkeypatch.setattr(
        authz, "reset_tenant_user_kb_override", reset_after_revision
    )
    revised = client.delete(
        f"{member_path}/{revised_target.id}",
        params={"reset": True},
        headers=alice_headers,
    )
    assert revised.status_code == 409, revised.text
    assert revised.json() == {
        "detail": "User was modified concurrently; retry the request"
    }
    assert "tenant_id" not in revised.text and "token_version" not in revised.text
    assert (
        asyncio.run(
            store.get_tenant_user_kb_override(
                "tenant-override-race-a",
                "kb_override_target_race",
                revised_target.id,
            )
        )
        == baseline
    )


def test_tenant_mutation_audits_keep_request_principal_tenant_after_actor_move(
    monkeypatch, tmp_path
):
    client, user_service, authz, admin, alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path, api_key=None
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    tenant_a = "tenant-event-snapshot-a"
    tenant_b = "tenant-event-snapshot-b"
    for tenant_id in (tenant_a, tenant_b):
        created = client.post(
            "/admin/tenants",
            json={"name": tenant_id, "tenant_id": tenant_id},
            headers=admin_headers,
        )
        assert created.status_code == 200, created.text

    async def seed():
        actor_user = await user_service.create_user(
            username="audit-actor-user", password="pass"
        )
        actor_membership = await user_service.create_user(
            username="audit-actor-membership", password="pass"
        )
        actor_override = await user_service.create_user(
            username="audit-actor-override", password="pass"
        )
        for actor in (alice, actor_user, actor_membership, actor_override):
            await authz.grant_tenant_membership(
                tenant_a,
                actor.id,
                "tenant_admin",
                granted_by=admin.id,
            )
        await authz.grant_tenant_membership(
            tenant_b,
            bob.id,
            "tenant_admin",
            granted_by=admin.id,
        )
        update_target = await user_service.create_user(
            username="audit-update-target",
            password="pass",
            tenant_id=tenant_a,
        )
        membership_target = await user_service.create_user(
            username="audit-membership-target",
            password="pass",
        )
        override_target = await user_service.create_user(
            username="audit-override-target",
            password="pass",
            tenant_id=tenant_a,
        )
        return (
            await user_service.get_user_or_404(alice.id),
            await user_service.get_user_or_404(bob.id),
            await user_service.get_user_or_404(actor_user.id),
            await user_service.get_user_or_404(actor_membership.id),
            await user_service.get_user_or_404(actor_override.id),
            update_target,
            membership_target,
            override_target,
        )

    (
        observer_a,
        observer_b,
        actor_user,
        actor_membership,
        actor_override,
        update_target,
        membership_target,
        override_target,
    ) = asyncio.run(seed())
    observer_a_headers = {
        "Authorization": f"Bearer {_token(user_service, observer_a)}"
    }
    observer_b_headers = {
        "Authorization": f"Bearer {_token(user_service, observer_b)}"
    }
    actor_headers = {
        actor_user.id: {"Authorization": f"Bearer {_token(user_service, actor_user)}"},
        actor_membership.id: {
            "Authorization": f"Bearer {_token(user_service, actor_membership)}"
        },
        actor_override.id: {
            "Authorization": f"Bearer {_token(user_service, actor_override)}"
        },
    }
    created_kb = client.post(
        "/kbs",
        json={"id": "kb_event_snapshot", "name": "Audit event snapshot"},
        headers=observer_a_headers,
    )
    assert created_kb.status_code == 200, created_kb.text

    store = client.app.state.metadata_store
    original_append = store.append_audit_event
    move_on_event = {
        ("user_updated", actor_user.id),
        ("tenant_membership_granted", actor_membership.id),
        ("tenant_user_kb_override_set", actor_override.id),
    }
    moved: set[tuple[str, str]] = set()

    async def move_actor_after_event_construction(event):
        key = (event.event_type, event.actor_user_id)
        if key in move_on_event and key not in moved:
            moved.add(key)
            assert event.actor_tenant_id == tenant_a
            await authz.grant_tenant_membership(
                tenant_b,
                event.actor_user_id,
                "tenant_admin",
                granted_by=admin.id,
            )
        return await original_append(event)

    monkeypatch.setattr(
        store, "append_audit_event", move_actor_after_event_construction
    )

    updated = client.patch(
        f"/tenants/{tenant_a}/users/{update_target.id}",
        json={"can_create_kb": True},
        headers=actor_headers[actor_user.id],
    )
    assert updated.status_code == 200, updated.text
    membership = client.put(
        f"/tenants/{tenant_a}/members/{membership_target.id}",
        json={"role": "tenant_member"},
        headers=actor_headers[actor_membership.id],
    )
    assert membership.status_code == 200, membership.text
    override = client.put(
        f"/tenants/{tenant_a}/kbs/kb_event_snapshot/members/{override_target.id}",
        json={"role": "viewer"},
        headers=actor_headers[actor_override.id],
    )
    assert override.status_code == 200, override.text
    assert moved == move_on_event

    events_a = client.get(
        f"/tenants/{tenant_a}/audit-events",
        headers=observer_a_headers,
    )
    assert events_a.status_code == 200, events_a.text
    relevant = {
        (event["event_type"], event["actor_user_id"]): event
        for event in events_a.json()
        if (event["event_type"], event["actor_user_id"]) in move_on_event
    }
    assert set(relevant) == move_on_event
    assert all(event["actor_tenant_id"] == tenant_a for event in relevant.values())

    events_b = client.get(
        f"/tenants/{tenant_b}/audit-events",
        headers=observer_b_headers,
    )
    assert events_b.status_code == 200, events_b.text
    assert not any(
        (event["event_type"], event["actor_user_id"]) in move_on_event
        for event in events_b.json()
    )


def test_tenant_audit_uses_event_time_tenant_and_reuses_filters(
    monkeypatch, tmp_path
):
    client, user_service, authz, admin, alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path, api_key=None
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    for tenant_id in ("tenant-audit", "tenant-later"):
        response = client.post(
            "/admin/tenants",
            json={"name": tenant_id, "tenant_id": tenant_id},
            headers=admin_headers,
        )
        assert response.status_code == 200, response.text
    asyncio.run(
        authz.grant_tenant_membership(
            "tenant-audit", alice.id, "tenant_admin", granted_by=admin.id
        )
    )
    asyncio.run(
        authz.grant_tenant_membership(
            "tenant-audit", bob.id, "tenant_member", granted_by=admin.id
        )
    )
    alice = asyncio.run(user_service.get_user_or_404(alice.id))
    bob = asyncio.run(user_service.get_user_or_404(bob.id))
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    audit_service = client.app.state.enterprise_audit_service

    visible_first = asyncio.run(
        audit_service.append(
            "tenant_visible",
            actor_user_id=bob.id,
            actor_tenant_id="tenant-audit",
            target_type="user",
            target_id=bob.id,
            metadata={"page": 1},
        )
    )
    visible_second = asyncio.run(
        audit_service.append(
            "tenant_visible",
            actor_user_id=bob.id,
            actor_tenant_id="tenant-audit",
            target_type="kb",
            target_id="missing-kb",
            metadata={"page": 2},
        )
    )
    asyncio.run(
        audit_service.append(
            "legacy_null",
            actor_user_id=None,
            actor_tenant_id=None,
        )
    )
    asyncio.run(
        audit_service.append(
            "other_tenant",
            actor_user_id=bob.id,
            actor_tenant_id="tenant-later",
        )
    )

    # A later primary-tenant move cannot rewrite historical attribution.
    asyncio.run(
        user_service.update_user(
            bob.id,
            tenant_id="tenant-later",
            actor_user_id=admin.id,
        )
    )
    events = client.get(
        "/tenants/tenant-audit/audit-events",
        # An attempted caller override is ignored; SQL remains pinned to path.
        params={"actor_tenant_id": "tenant-later"},
        headers=alice_headers,
    )
    assert events.status_code == 200, events.text
    ids = {event["id"] for event in events.json()}
    assert visible_first.id in ids
    assert visible_second.id in ids
    assert all(
        event["actor_tenant_id"] == "tenant-audit" for event in events.json()
    )
    assert all(event["event_type"] != "legacy_null" for event in events.json())
    assert all(event["event_type"] != "other_tenant" for event in events.json())
    assert {
        event["actor_username"]
        for event in events.json()
        if event["event_type"] == "tenant_visible"
    } == {"bob"}

    filtered = client.get(
        "/tenants/tenant-audit/audit-events",
        params={"event_type": "tenant_visible", "target_type": "user"},
        headers=alice_headers,
    )
    assert filtered.status_code == 200, filtered.text
    assert [event["id"] for event in filtered.json()] == [visible_first.id]
    page_one = client.get(
        "/tenants/tenant-audit/audit-events",
        params={"event_type": "tenant_visible", "limit": 1},
        headers=alice_headers,
    ).json()
    page_two = client.get(
        "/tenants/tenant-audit/audit-events",
        params={"event_type": "tenant_visible", "limit": 1, "offset": 1},
        headers=alice_headers,
    ).json()
    assert len(page_one) == len(page_two) == 1
    assert page_one[0]["id"] != page_two[0]["id"]


def test_tenant_kb_member_allow_deny_reset_caps_sources_and_generation(
    monkeypatch, tmp_path
):
    client, user_service, authz, admin, alice, bob, _probe = _build_enterprise_client(
        monkeypatch, tmp_path, api_key=None
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    for tenant_id in ("tenant-members", "tenant-cross"):
        response = client.post(
            "/admin/tenants",
            json={"name": tenant_id, "tenant_id": tenant_id},
            headers=admin_headers,
        )
        assert response.status_code == 200, response.text
    carol = asyncio.run(
        user_service.create_user(username="carol-member", password="pass")
    )
    asyncio.run(
        authz.grant_tenant_membership(
            "tenant-members", alice.id, "tenant_admin", granted_by=admin.id
        )
    )
    asyncio.run(
        authz.grant_tenant_membership(
            "tenant-members", bob.id, "tenant_member", granted_by=admin.id
        )
    )
    asyncio.run(
        authz.grant_tenant_membership(
            "tenant-cross", carol.id, "tenant_admin", granted_by=admin.id
        )
    )
    alice = asyncio.run(user_service.get_user_or_404(alice.id))
    bob = asyncio.run(user_service.get_user_or_404(bob.id))
    carol = asyncio.run(user_service.get_user_or_404(carol.id))
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}
    carol_headers = {"Authorization": f"Bearer {_token(user_service, carol)}"}

    owned = client.post(
        "/kbs",
        json={"id": "kb_member_owned", "name": "Member Owned"},
        headers=alice_headers,
    )
    assert owned.status_code == 200, owned.text
    members_path = "/tenants/tenant-members/kbs/kb_member_owned/members"
    initial = client.get(members_path, headers=alice_headers)
    assert initial.status_code == 200, initial.text
    initial_by_user = {item["user_id"]: item for item in initial.json()}
    assert {alice.id, bob.id}.issubset(initial_by_user)
    assert initial_by_user[bob.id]["effective_role"] is None

    allowed = client.put(
        f"{members_path}/{bob.id}",
        json={"role": "viewer"},
        headers=alice_headers,
    )
    assert allowed.status_code == 200, allowed.text
    assert allowed.json()["override_effect"] == "allow"
    assert allowed.json()["override_role"] == "kb_viewer"
    assert allowed.json()["effective_role"] == "kb_viewer"
    assert "tenant_owned_override" in allowed.json()["sources"]
    assert "kb_member_owned" in {
        item["id"]
        for item in client.get("/kbs", headers=bob_headers).json()[
            "knowledge_bases"
        ]
    }
    query_allowed = client.post(
        "/kbs/kb_member_owned/query",
        json={"query": "allowed", "mode": "mix"},
        headers=bob_headers,
    )
    assert query_allowed.status_code == 200, query_allowed.text

    denied = client.delete(f"{members_path}/{bob.id}", headers=alice_headers)
    assert denied.status_code == 200, denied.text
    assert denied.json()["override_effect"] == "deny"
    assert denied.json()["effective_role"] is None
    assert "kb_member_owned" not in {
        item["id"]
        for item in client.get("/kbs", headers=bob_headers).json()[
            "knowledge_bases"
        ]
    }
    assert (
        client.post(
            "/kbs/kb_member_owned/query",
            json={"query": "denied", "mode": "mix"},
            headers=bob_headers,
        ).status_code
        == 403
    )
    reset = client.delete(
        f"{members_path}/{bob.id}",
        params={"reset": True},
        headers=alice_headers,
    )
    assert reset.status_code == 200, reset.text
    assert reset.json()["user_id"] == bob.id
    assert reset.json()["override_effect"] is None

    # A tenant deny never suppresses a direct platform grant.
    direct = client.put(
        "/admin/kbs/kb_member_owned/acl",
        json={"user_id": bob.id, "role": "kb_viewer"},
        headers=admin_headers,
    )
    assert direct.status_code == 200, direct.text
    direct_remaining = client.delete(
        f"{members_path}/{bob.id}", headers=alice_headers
    )
    assert direct_remaining.status_code == 200, direct_remaining.text
    assert direct_remaining.json()["effective_role"] == "kb_viewer"
    assert direct_remaining.json()["platform_role"] == "kb_viewer"
    assert "direct" in direct_remaining.json()["sources"]

    provisioned = client.post(
        "/kbs",
        json={
            "id": "kb_member_provisioned",
            "name": "Provisioned",
            "tenant_id": "tenant-members",
            "metadata": {
                "tenant_managed": True,
                "tenant_tag": "tenant:tenant-members",
            },
        },
        headers=admin_headers,
    )
    assert provisioned.status_code == 200, provisioned.text
    assert provisioned.json()["origin"] == "platform"
    provisioned_path = (
        "/tenants/tenant-members/kbs/kb_member_provisioned/members"
    )
    assert client.get(provisioned_path, headers=alice_headers).status_code == 403
    tenant_acl = client.put(
        "/admin/kbs/kb_member_provisioned/acl",
        json={"tenant_id": "tenant-members", "role": "kb_editor"},
        headers=admin_headers,
    )
    assert tenant_acl.status_code == 200, tenant_acl.text
    too_high = client.put(
        f"{provisioned_path}/{bob.id}",
        json={"role": "admin"},
        headers=alice_headers,
    )
    assert too_high.status_code == 400
    capped = client.put(
        f"{provisioned_path}/{bob.id}",
        json={"role": "viewer"},
        headers=alice_headers,
    )
    assert capped.status_code == 200, capped.text
    assert capped.json()["effective_role"] == "kb_viewer"
    assert capped.json()["tenant_acl_role"] == "kb_editor"
    assert "tenant_override_capped" in capped.json()["sources"]

    downgraded = client.put(
        "/admin/kbs/kb_member_provisioned/acl",
        json={"tenant_id": "tenant-members", "role": "kb_viewer"},
        headers=admin_headers,
    )
    assert downgraded.status_code == 200, downgraded.text
    after_downgrade = client.get(provisioned_path, headers=alice_headers)
    assert after_downgrade.status_code == 200, after_downgrade.text
    bob_after_downgrade = next(
        item for item in after_downgrade.json() if item["user_id"] == bob.id
    )
    assert bob_after_downgrade["effective_role"] == "kb_viewer"
    assert bob_after_downgrade["tenant_acl_role"] == "kb_viewer"
    revoked_acl = client.delete(
        "/admin/kbs/kb_member_provisioned/acl/tenants/tenant-members",
        headers=admin_headers,
    )
    assert revoked_acl.status_code == 200, revoked_acl.text
    bob_kbs_after_revoke = {
        item["id"]
        for item in client.get("/kbs", headers=bob_headers).json()[
            "knowledge_bases"
        ]
    }
    assert "kb_member_provisioned" not in bob_kbs_after_revoke
    assert client.get(provisioned_path, headers=alice_headers).status_code == 403

    # Visibility survives a tenant-scoped deny and is disclosed as a source.
    public = client.post(
        "/kbs",
        json={
            "id": "kb_member_public",
            "name": "Public Provisioned",
            "visibility": "public",
        },
        headers=admin_headers,
    )
    assert public.status_code == 200, public.text
    public_acl = client.put(
        "/admin/kbs/kb_member_public/acl",
        json={"tenant_id": "tenant-members", "role": "kb_viewer"},
        headers=admin_headers,
    )
    assert public_acl.status_code == 200, public_acl.text
    public_path = "/tenants/tenant-members/kbs/kb_member_public/members"
    visibility_remaining = client.delete(
        f"{public_path}/{bob.id}", headers=alice_headers
    )
    assert visibility_remaining.status_code == 200, visibility_remaining.text
    assert visibility_remaining.json()["effective_role"] == "kb_viewer"
    assert visibility_remaining.json()["platform_role"] == "kb_viewer"
    assert "visibility" in visibility_remaining.json()["sources"]
    visibility_reset = client.delete(
        f"{public_path}/{bob.id}",
        params={"reset": True},
        headers=alice_headers,
    )
    assert visibility_reset.status_code == 200, visibility_reset.text
    assert visibility_reset.json()["user_id"] == bob.id

    public_only = client.post(
        "/kbs",
        json={
            "id": "kb_member_public_only",
            "name": "Public Only",
            "visibility": "public",
        },
        headers=admin_headers,
    )
    assert public_only.status_code == 200, public_only.text
    assert (
        client.get(
            "/tenants/tenant-members/kbs/kb_member_public_only/members",
            headers=alice_headers,
        ).status_code
        == 403
    )
    assert client.get(members_path, headers=carol_headers).status_code == 403
    assert (
        client.put(
            f"{members_path}/{carol.id}",
            json={"role": "viewer"},
            headers=alice_headers,
        ).status_code
        == 404
    )
    service_key = client.post(
        "/admin/service-api-keys",
        json={"name": "member-admin-denied"},
        headers=admin_headers,
    )
    assert service_key.status_code == 200, service_key.text
    assert (
        client.get(
            members_path,
            headers={"X-API-Key": service_key.json()["api_key"]},
        ).status_code
        == 403
    )

    # Capture the original catalog generation exactly once. If lifecycle state
    # switches before the write, the old generation is rejected with 409 and
    # the handler does not re-read/upgrade to the replacement identity.
    kb_service = client.app.state.kb_service
    metadata_store = client.app.state.metadata_store
    original_get = kb_service.get
    get_calls = 0

    async def counted_get(*args, **kwargs):
        nonlocal get_calls
        get_calls += 1
        return await original_get(*args, **kwargs)

    original_grant_override = authz.grant_tenant_user_kb_override

    async def race_grant_override(kb_id, tenant_id, user_id, role, **kwargs):
        expected_generation = kwargs["expected_generation"]
        await metadata_store.purge_kb_metadata(kb_id, expected_generation)
        await metadata_store.activate_kb_generation(kb_id, str(uuid4()))
        return await original_grant_override(
            kb_id, tenant_id, user_id, role, **kwargs
        )

    monkeypatch.setattr(kb_service, "get", counted_get)
    monkeypatch.setattr(
        authz, "grant_tenant_user_kb_override", race_grant_override
    )
    stale_write = client.put(
        f"{members_path}/{bob.id}",
        json={"role": "viewer"},
        headers=alice_headers,
    )
    assert stale_write.status_code == 409, stale_write.text
    # One read belongs to the outer write-admission middleware and one to the
    # authorization service's own generation capture; neither re-reads after
    # the injected replacement.
    assert get_calls == 2


def test_service_api_key_create_and_rotate_use_catalog_generation_map(
    monkeypatch, tmp_path
):
    client, user_service, _authz, admin, _alice, _bob, _probe = (
        _build_enterprise_client(monkeypatch, tmp_path, api_key=None)
    )
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    created_kb = client.post(
        "/kbs",
        json={"id": "kb_service_generation", "name": "Generation"},
        headers=admin_headers,
    )
    assert created_kb.status_code == 200, created_kb.text
    generation = asyncio.run(
        client.app.state.kb_service.get("kb_service_generation")
    ).generation

    store = client.app.state.metadata_store
    original_create = store.create_enterprise_api_key
    generation_maps: list[dict[str, str]] = []

    async def capture_create(record, *, expected_kb_generations=None):
        generation_maps.append(dict(expected_kb_generations or {}))
        return await original_create(
            record, expected_kb_generations=expected_kb_generations
        )

    monkeypatch.setattr(store, "create_enterprise_api_key", capture_create)
    created_key = client.post(
        "/admin/service-api-keys",
        json={
            "name": "generation-reader",
            "kb_roles": {"kb_service_generation": "kb_viewer"},
        },
        headers=admin_headers,
    )
    assert created_key.status_code == 200, created_key.text
    rotated = client.post(
        f"/admin/service-api-keys/{created_key.json()['key']['id']}:rotate",
        headers=admin_headers,
    )
    assert rotated.status_code == 200, rotated.text
    assert generation_maps == [
        {"kb_service_generation": generation},
        {"kb_service_generation": generation},
    ]
