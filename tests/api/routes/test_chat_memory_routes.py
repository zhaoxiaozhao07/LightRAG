"""Route contract tests for Chat Memory reads and durable admin controls.

The memory service is replaced by a recording fake on ``app.state`` so these
tests exercise routing, auth, ownership and verify CRUD routes do not call the
deprecated fire-and-forget hooks.
"""

from __future__ import annotations

# ruff: noqa: E402 - LightRAG API modules parse sys.argv at import time.

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from lightrag.api.auth import auth_handler
from lightrag.api.chat_memory_service import (
    ChatMemoryEventNotFoundError,
    ChatMemoryRetryConflictError,
    ChatMemoryUnavailableError,
)
from lightrag.api.enterprise_auth import (
    AuditService,
    AuthorizationService,
    ChatConversationService,
    EnterpriseLimitService,
    ServiceAPIKeyService,
    SystemSettingsService,
    UserService,
)
from lightrag.api.kb_service import KnowledgeBaseService
from lightrag.api.metadata_store import SQLiteMetadataStore
from lightrag.api.routers.chat_routes import create_chat_routes
from lightrag.api.routers.enterprise_routes import create_enterprise_routes
from lightrag.api.utils_api import get_combined_auth_dependency
sys.argv = _original_argv

pytestmark = pytest.mark.offline

_API_KEY = "test-key"


class FakeChatMemoryService:
    """Duck-typed stand-in recording every hook/endpoint interaction."""

    def __init__(self):
        self.ingest_calls: list[dict] = []
        self.purge_calls: list[tuple[str, list[str]]] = []
        self.forget_message_calls: list[dict] = []
        self.forget_session_calls: list[dict] = []
        self.search_calls: list[dict] = []
        self.search_results: list[dict] = []
        self.search_error: Exception | None = None
        self.backlog_scan_calls: list[int] = []
        self.enqueue_purge_calls: list[dict] = []
        self.outbox_stats_calls = 0
        self.retry_calls: list[dict] = []
        self.retry_event: Any = None
        self.retry_error: Exception | None = None
        self.group_create_calls: list[tuple[str, str]] = []

    def schedule_ingest(self, *, user_id, project_id, session_id, messages):
        self.ingest_calls.append(
            {
                "user_id": user_id,
                "project_id": project_id,
                "session_id": session_id,
                "messages": list(messages),
            }
        )
        return None

    def schedule_purge(self, user_id, project_ids):
        self.purge_calls.append((user_id, list(project_ids)))
        return None

    def schedule_forget_message(self, *, user_id, project_id, session_id, seq):
        self.forget_message_calls.append(
            {
                "user_id": user_id,
                "project_id": project_id,
                "session_id": session_id,
                "seq": seq,
            }
        )
        return None

    def schedule_forget_session(self, *, user_id, project_id, session_id):
        self.forget_session_calls.append(
            {
                "user_id": user_id,
                "project_id": project_id,
                "session_id": session_id,
            }
        )
        return None

    async def search(self, *, user_id, project_id, query, limit=None):
        if self.search_error is not None:
            raise self.search_error
        self.search_calls.append(
            {
                "user_id": user_id,
                "project_id": project_id,
                "query": query,
                "limit": limit,
            }
        )
        return self.search_results

    async def project_overview(self, user_id, project_id):
        return {
            "project_id": project_id,
            "enabled": True,
            "available": True,
            "episode_count": 3,
            "last_ingested_at": "2026-07-11T09:00:00+00:00",
        }

    async def global_stats(self):
        return {
            "enabled": True,
            "available": True,
            "pending_tasks": 0,
            "episode_count": 7,
            "user_count": 2,
            "project_count": 4,
        }

    async def purge_projects(self, user_id, project_ids):
        raise AssertionError("admin purge must not clear Graphiti directly")

    async def enqueue_purge_projects(
        self,
        user_id,
        project_ids,
        *,
        actor_user_id,
        actor_tenant_id=None,
    ):
        call = {
            "user_id": user_id,
            "project_ids": list(project_ids),
            "actor_user_id": actor_user_id,
            "actor_tenant_id": actor_tenant_id,
        }
        self.enqueue_purge_calls.append(call)
        return {"queued": len(call["project_ids"]), "noop": 0}

    async def outbox_stats(self):
        self.outbox_stats_calls += 1
        return {
            "pending": 3,
            "running": 1,
            "retry_wait": 2,
            "dead_letter": 0,
            "oldest_available_at": "2026-07-16T00:00:00+00:00",
            "oldest_lag_seconds": 4.5,
        }

    async def retry_purge_event(
        self,
        event_id,
        *,
        actor_user_id,
        actor_tenant_id=None,
    ):
        self.retry_calls.append(
            {
                "event_id": event_id,
                "actor_user_id": actor_user_id,
                "actor_tenant_id": actor_tenant_id,
            }
        )
        if self.retry_error is not None:
            raise self.retry_error
        if self.retry_event is None:
            raise ChatMemoryEventNotFoundError(event_id)
        return self.retry_event

    async def run_backlog_scan(self, *, limit=100):
        self.backlog_scan_calls.append(limit)
        raise AssertionError("legacy watermark backlog scan must not run")


class FakeChatMemoryWorker:
    def __init__(self):
        self.recover_calls: list[int] = []
        self.nudge_calls = 0

    async def recover_once(self, *, limit=100):
        self.recover_calls.append(limit)
        return 5

    def nudge(self):
        self.nudge_calls += 1


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
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _patch_enterprise_args(monkeypatch, args) -> None:
    from lightrag.api import config as api_config
    import lightrag.api.utils_api as utils_api

    monkeypatch.setattr(api_config, "global_args", args)
    monkeypatch.setattr(utils_api, "global_args", args)

    dependency_functions = [get_combined_auth_dependency]
    for factory in (create_chat_routes, create_enterprise_routes):
        dependency = factory.__globals__["get_combined_auth_dependency"]
        if dependency not in dependency_functions:
            dependency_functions.append(dependency)

    for dependency in dependency_functions:
        monkeypatch.setitem(dependency.__globals__, "global_args", args)


def _token(user_service: UserService, user) -> str:
    return auth_handler.create_token(
        username=user.username,
        role=user.system_role,
        metadata=user_service.token_metadata_for_user(user),
    )


def _build_client(
    monkeypatch,
    tmp_path: Path,
    *,
    memory_service: FakeChatMemoryService | None = None,
    with_admin_routes: bool = False,
    api_key: str | None = _API_KEY,
    args: SimpleNamespace | None = None,
):
    args = args or _enterprise_args()
    _patch_enterprise_args(monkeypatch, args)

    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kb.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    audit_service = AuditService(metadata_store)
    user_service = UserService(metadata_store, audit_service)
    settings_service = SystemSettingsService(metadata_store)
    chat_service = ChatConversationService(metadata_store, audit_service)
    api_key_service = ServiceAPIKeyService(metadata_store, audit_service)
    limit_service = EnterpriseLimitService(audit_service)
    authz_service = AuthorizationService(
        metadata_store, audit_service, kb_service=kb_service
    )

    async def seed():
        await kb_service.initialize()
        await metadata_store.initialize()
        await settings_service.initialize_registration_setting(False)
        alice = await user_service.create_user(
            username="alice",
            password="alice-pass",
        )
        bob = await user_service.create_user(
            username="bob",
            password="bob-pass",
        )
        return alice, bob

    alice, bob = asyncio.run(seed())

    app = FastAPI()
    app.state.enterprise_enabled = True
    app.state.metadata_store = metadata_store
    app.state.enterprise_user_service = user_service
    app.state.enterprise_settings_service = settings_service
    app.state.enterprise_chat_conversation_service = chat_service
    app.state.enterprise_api_key_service = api_key_service
    app.state.enterprise_limit_service = limit_service
    app.state.enterprise_authorization_service = authz_service
    app.state.enterprise_audit_service = audit_service
    if memory_service is not None:
        app.state.enterprise_chat_memory_service = memory_service
        app.state.enterprise_chat_memory_maintenance_service = memory_service
        app.state.enterprise_chat_memory_worker = FakeChatMemoryWorker()
    app.include_router(create_chat_routes(api_key=api_key))
    if with_admin_routes:
        app.include_router(
            create_enterprise_routes(api_key=api_key, kb_service=kb_service)
        )
    return TestClient(app), user_service, alice, bob


def _create_project(client, headers, name="记忆项目") -> str:
    response = client.post("/chat/projects", json={"name": name}, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()["id"]


def test_memory_search_happy_path(monkeypatch, tmp_path):
    memory = FakeChatMemoryService()
    memory.search_results = [
        {
            "uuid": "edge-1",
            "name": "USES",
            "fact": "项目采用 NR/BR 并用",
            "valid_at": "2026-07-10T08:00:05+00:00",
            "invalid_at": None,
            "created_at": "2026-07-10T08:00:41+00:00",
            "expired_at": None,
        }
    ]
    client, user_service, alice, _bob = _build_client(
        monkeypatch, tmp_path, memory_service=memory
    )
    headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    project_id = _create_project(client, headers)

    response = client.post(
        f"/chat/projects/{project_id}/memory:search",
        json={"query": "之前对低温性能有什么结论？", "limit": 5},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["project_id"] == project_id
    assert body["total"] == 1
    assert body["facts"][0]["fact"] == "项目采用 NR/BR 并用"
    assert body["facts"][0]["invalid_at"] is None
    assert memory.search_calls == [
        {
            "user_id": alice.id,
            "project_id": project_id,
            "query": "之前对低温性能有什么结论？",
            "limit": 5,
        }
    ]


def test_memory_overview_endpoint(monkeypatch, tmp_path):
    memory = FakeChatMemoryService()
    client, user_service, alice, bob = _build_client(
        monkeypatch, tmp_path, memory_service=memory
    )
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}
    project_id = _create_project(client, alice_headers)

    resp = client.get(f"/chat/projects/{project_id}/memory", headers=alice_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["project_id"] == project_id
    assert body["episode_count"] == 3
    assert body["available"] is True
    assert body["last_ingested_at"] == "2026-07-11T09:00:00+00:00"

    # Foreign/missing project => 404 (no existence leak).
    assert (
        client.get(f"/chat/projects/{project_id}/memory", headers=bob_headers)
        .status_code
        == 404
    )
    assert (
        client.get("/chat/projects/proj_ghost/memory", headers=alice_headers)
        .status_code
        == 404
    )


def test_memory_overview_disabled_returns_503(monkeypatch, tmp_path):
    client, user_service, alice, _bob = _build_client(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    project_id = _create_project(client, headers)
    resp = client.get(f"/chat/projects/{project_id}/memory", headers=headers)
    assert resp.status_code == 503


def test_memory_search_validation_and_ownership(monkeypatch, tmp_path):
    memory = FakeChatMemoryService()
    client, user_service, alice, bob = _build_client(
        monkeypatch, tmp_path, memory_service=memory
    )
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}
    project_id = _create_project(client, alice_headers)
    url = f"/chat/projects/{project_id}/memory:search"

    # Foreign or missing projects are indistinguishable 404s.
    assert (
        client.post(url, json={"query": "hax"}, headers=bob_headers).status_code == 404
    )
    assert (
        client.post(
            "/chat/projects/proj_missing/memory:search",
            json={"query": "x"},
            headers=alice_headers,
        ).status_code
        == 404
    )
    # Validation: empty query -> 422; limit out of bounds -> 422.
    assert (
        client.post(url, json={"query": ""}, headers=alice_headers).status_code == 422
    )
    assert (
        client.post(
            url, json={"query": "x", "limit": 0}, headers=alice_headers
        ).status_code
        == 422
    )
    assert (
        client.post(
            url, json={"query": "x", "limit": 51}, headers=alice_headers
        ).status_code
        == 422
    )
    assert memory.search_calls == []


def test_memory_search_disabled_and_unavailable(monkeypatch, tmp_path):
    # No service on app.state -> feature disabled.
    client, user_service, alice, _bob = _build_client(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    project_id = _create_project(client, headers)
    disabled = client.post(
        f"/chat/projects/{project_id}/memory:search",
        json={"query": "x"},
        headers=headers,
    )
    assert disabled.status_code == 503
    assert disabled.json()["detail"] == "Chat memory is not enabled"

    # Service present but backend down -> temporary 503.
    memory = FakeChatMemoryService()
    memory.search_error = ChatMemoryUnavailableError("neo4j down")
    client2, user_service2, alice2, _bob2 = _build_client(
        monkeypatch, tmp_path / "u", memory_service=memory
    )
    headers2 = {"Authorization": f"Bearer {_token(user_service2, alice2)}"}
    project2 = _create_project(client2, headers2)
    unavailable = client2.post(
        f"/chat/projects/{project2}/memory:search",
        json={"query": "x"},
        headers=headers2,
    )
    assert unavailable.status_code == 503
    assert unavailable.json()["detail"] == "Chat memory is temporarily unavailable"


def test_memory_search_rejects_api_key_principal(monkeypatch, tmp_path):
    memory = FakeChatMemoryService()
    args = _enterprise_args(enterprise_legacy_api_key_superadmin=True)
    client, _user_service, _alice, _bob = _build_client(
        monkeypatch, tmp_path, memory_service=memory, args=args
    )
    response = client.post(
        "/chat/projects/proj_x/memory:search",
        json={"query": "x"},
        headers={"X-API-Key": _API_KEY},
    )
    assert response.status_code == 403
    assert client.post(
        "/chat/projects/proj_x/memory:search", json={"query": "x"}
    ).status_code in (401, 403)


def test_append_messages_does_not_schedule_legacy_memory_ingest(monkeypatch, tmp_path):
    memory = FakeChatMemoryService()
    client, user_service, alice, _bob = _build_client(
        monkeypatch, tmp_path, memory_service=memory
    )
    headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    project_id = _create_project(client, headers)
    session_id = client.post(
        f"/chat/projects/{project_id}/sessions", headers=headers
    ).json()["id"]

    appended = client.post(
        f"/chat/projects/{project_id}/sessions/{session_id}/messages",
        json={
            "messages": [
                {"role": "user", "content": "低温屈挠性怎么提升？"},
                {"role": "assistant", "content": "建议 NR/BR 并用… [A1]"},
            ]
        },
        headers=headers,
    )
    assert appended.status_code == 200, appended.text
    assert memory.ingest_calls == []

    # A failed append (foreign session) must not ingest.
    foreign = client.post(
        f"/chat/projects/{project_id}/sessions/sess_missing/messages",
        json={"messages": [{"role": "user", "content": "x"}]},
        headers=headers,
    )
    assert foreign.status_code == 404
    assert memory.ingest_calls == []


def test_append_messages_works_without_memory_service(monkeypatch, tmp_path):
    client, user_service, alice, _bob = _build_client(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    project_id = _create_project(client, headers)
    session_id = client.post(
        f"/chat/projects/{project_id}/sessions", headers=headers
    ).json()["id"]
    appended = client.post(
        f"/chat/projects/{project_id}/sessions/{session_id}/messages",
        json={"messages": [{"role": "user", "content": "无记忆服务也能落库"}]},
        headers=headers,
    )
    assert appended.status_code == 200, appended.text


def test_delete_message_and_session_do_not_schedule_legacy_forget(monkeypatch, tmp_path):
    memory = FakeChatMemoryService()
    client, user_service, alice, _bob = _build_client(
        monkeypatch, tmp_path, memory_service=memory
    )
    headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    project_id = _create_project(client, headers)
    session_id = client.post(
        f"/chat/projects/{project_id}/sessions", headers=headers
    ).json()["id"]
    appended = client.post(
        f"/chat/projects/{project_id}/sessions/{session_id}/messages",
        json={
            "messages": [
                {"role": "user", "content": "问题一"},
                {"role": "assistant", "content": "回答一"},
            ]
        },
        headers=headers,
    ).json()
    message_id = appended["messages"][0]["id"]

    deleted = client.delete(
        f"/chat/projects/{project_id}/sessions/{session_id}/messages/{message_id}",
        headers=headers,
    )
    assert deleted.status_code == 200, deleted.text
    assert memory.forget_message_calls == []

    # A no-op delete (missing message) does not schedule a forget.
    missing = client.delete(
        f"/chat/projects/{project_id}/sessions/{session_id}/messages/msg_ghost",
        headers=headers,
    )
    assert missing.status_code == 404
    assert memory.forget_message_calls == []

    session_deleted = client.delete(
        f"/chat/projects/{project_id}/sessions/{session_id}", headers=headers
    )
    assert session_deleted.status_code == 200, session_deleted.text
    assert memory.forget_session_calls == []


def test_delete_project_does_not_schedule_legacy_purge(monkeypatch, tmp_path):
    memory = FakeChatMemoryService()
    client, user_service, alice, bob = _build_client(
        monkeypatch, tmp_path, memory_service=memory
    )
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}
    project_id = _create_project(client, alice_headers)

    # A foreign delete 404s and must not purge.
    assert (
        client.delete(f"/chat/projects/{project_id}", headers=bob_headers).status_code
        == 404
    )
    assert memory.purge_calls == []

    deleted = client.delete(f"/chat/projects/{project_id}", headers=alice_headers)
    assert deleted.status_code == 200, deleted.text
    assert memory.purge_calls == []


def test_admin_delete_user_does_not_enumerate_or_schedule_legacy_purge(
    monkeypatch, tmp_path
):
    memory = FakeChatMemoryService()
    client, user_service, alice, bob = _build_client(
        monkeypatch, tmp_path, memory_service=memory, with_admin_routes=True
    )

    async def make_admin():
        return await user_service.bootstrap_super_admin(
            username="root", password="root-pass", password_hash=None
        )

    admin = asyncio.run(make_admin())
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}
    _create_project(client, alice_headers, name="项目A")
    _create_project(client, alice_headers, name="项目B")
    # Another user's project must not be swept up.
    _create_project(client, bob_headers, name="bob的项目")

    deleted = client.delete(f"/admin/users/{alice.id}", headers=admin_headers)
    assert deleted.status_code == 200, deleted.text
    assert deleted.json() == {"deleted": True}

    assert memory.purge_calls == []
    assert memory.enqueue_purge_calls == []

    # Deleting a user without chat projects schedules nothing.
    missing = client.delete("/admin/users/usr_ghost", headers=admin_headers)
    assert missing.status_code == 404
    assert memory.purge_calls == []


def test_admin_purge_and_backlog_endpoints(monkeypatch, tmp_path):
    memory = FakeChatMemoryService()
    client, user_service, alice, bob = _build_client(
        monkeypatch, tmp_path, memory_service=memory, with_admin_routes=True
    )

    async def make_admin():
        return await user_service.bootstrap_super_admin(
            username="root", password="root-pass", password_hash=None
        )

    admin = asyncio.run(make_admin())
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}
    project_a = _create_project(client, alice_headers, name="项目A")
    project_b = _create_project(client, alice_headers, name="项目B")
    bob_project = _create_project(client, bob_headers, name="bob项目")

    # Explicit project list purge.
    resp = client.post(
        f"/admin/users/{alice.id}/chat-memory:purge",
        json={"project_ids": [project_a]},
        headers=admin_headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"queued": 1, "noop": 0, "project_ids": [project_a]}
    assert memory.enqueue_purge_calls[-1] == {
        "user_id": alice.id,
        "project_ids": [project_a],
        "actor_user_id": admin.id,
        "actor_tenant_id": None,
    }
    assert memory.purge_calls == []

    foreign = client.post(
        f"/admin/users/{alice.id}/chat-memory:purge",
        json={"project_ids": [bob_project]},
        headers=admin_headers,
    )
    assert foreign.status_code == 404
    assert len(memory.enqueue_purge_calls) == 1

    # Omitted list => purge every project of the user.
    resp = client.post(
        f"/admin/users/{alice.id}/chat-memory:purge",
        json={},
        headers=admin_headers,
    )
    assert resp.status_code == 200, resp.text
    assert set(resp.json()["project_ids"]) == {project_a, project_b}
    assert resp.json()["queued"] == 2
    assert resp.json()["noop"] == 0

    # Purge for an unknown user => 404.
    assert (
        client.post(
            "/admin/users/usr_ghost/chat-memory:purge", json={}, headers=admin_headers
        ).status_code
        == 404
    )

    # Manual backlog scan.
    scan = client.post(
        "/admin/chat-memory:backlog-scan", json={"limit": 50}, headers=admin_headers
    )
    assert scan.status_code == 200, scan.text
    assert scan.json() == {
        "recovered_events": 5,
        "outbox": {
            "pending": 3,
            "running": 1,
            "retry_wait": 2,
            "dead_letter": 0,
            "oldest_available_at": "2026-07-16T00:00:00+00:00",
            "oldest_lag_seconds": 4.5,
        },
    }
    assert memory.backlog_scan_calls == []
    assert memory.outbox_stats_calls == 1
    assert memory.retry_calls == []
    worker = cast(FastAPI, client.app).state.enterprise_chat_memory_worker
    assert worker.recover_calls == [50]
    assert worker.nudge_calls == 1


def test_admin_retry_purge_event_by_durable_id_after_source_deletion(
    monkeypatch, tmp_path
):
    memory = FakeChatMemoryService()
    memory.retry_event = SimpleNamespace(
        event_id="evt-deleted-target",
        status="retry_wait",
        user_id="usr-already-deleted",
        project_id="proj-already-deleted",
        event_type="purge",
    )
    client, user_service, alice, _bob = _build_client(
        monkeypatch, tmp_path, memory_service=memory, with_admin_routes=True
    )

    async def make_admin():
        return await user_service.bootstrap_super_admin(
            username="root", password="root-pass", password_hash=None
        )

    admin = asyncio.run(make_admin())
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}

    response = client.post(
        "/admin/chat-memory/events/evt-deleted-target:retry",
        headers=admin_headers,
    )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "event_id": "evt-deleted-target",
        "status": "retry_wait",
        "user_id": "usr-already-deleted",
        "project_id": "proj-already-deleted",
        "event_type": "purge",
    }
    assert memory.retry_calls == [
        {
            "event_id": "evt-deleted-target",
            "actor_user_id": admin.id,
            "actor_tenant_id": None,
        }
    ]
    assert memory.group_create_calls == []

    denied = client.post(
        "/admin/chat-memory/events/evt-deleted-target:retry",
        headers=alice_headers,
    )
    assert denied.status_code == 403
    assert len(memory.retry_calls) == 1


def test_admin_retry_purge_event_maps_missing_and_conflict_errors(
    monkeypatch, tmp_path
):
    memory = FakeChatMemoryService()
    client, user_service, _alice, _bob = _build_client(
        monkeypatch, tmp_path, memory_service=memory, with_admin_routes=True
    )

    async def make_admin():
        return await user_service.bootstrap_super_admin(
            username="root", password="root-pass", password_hash=None
        )

    admin = asyncio.run(make_admin())
    headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}

    memory.retry_error = ChatMemoryEventNotFoundError("evt-forged")
    missing = client.post(
        "/admin/chat-memory/events/evt-forged:retry",
        headers=headers,
    )
    assert missing.status_code == 404
    assert missing.json() == {"detail": "Chat memory event not found"}
    assert memory.group_create_calls == []

    memory.retry_error = ChatMemoryRetryConflictError(
        "chat_memory_old_graph_store_required",
        "Restore the original MEMORY_NEO4J_DEPLOYMENT_ID or Neo4j backend, then retry",
    )
    conflict = client.post(
        "/admin/chat-memory/events/evt-old-graph:retry",
        headers=headers,
    )
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == {
        "error_code": "chat_memory_old_graph_store_required",
        "message": (
            "Restore the original MEMORY_NEO4J_DEPLOYMENT_ID or Neo4j backend, "
            "then retry"
        ),
    }


def test_admin_memory_endpoints_503_when_disabled(monkeypatch, tmp_path):
    client, user_service, alice, _bob = _build_client(
        monkeypatch, tmp_path, with_admin_routes=True
    )

    async def make_admin():
        return await user_service.bootstrap_super_admin(
            username="root", password="root-pass", password_hash=None
        )

    admin = asyncio.run(make_admin())
    admin_headers = {"Authorization": f"Bearer {_token(user_service, admin)}"}
    assert (
        client.post(
            f"/admin/users/{alice.id}/chat-memory:purge", json={}, headers=admin_headers
        ).status_code
        == 503
    )
    assert (
        client.post(
            "/admin/chat-memory:backlog-scan", json={}, headers=admin_headers
        ).status_code
        == 503
    )
    assert (
        client.post(
            "/admin/chat-memory/events/evt-missing:retry",
            headers=admin_headers,
        ).status_code
        == 503
    )
