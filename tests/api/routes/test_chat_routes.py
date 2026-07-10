from __future__ import annotations

# ruff: noqa: E402 - LightRAG API modules parse sys.argv at import time.

import asyncio
import re
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
    EnterpriseLimitService,
    ServiceAPIKeyService,
    SystemSettingsService,
    UserService,
)
from lightrag.api.kb_service import KnowledgeBaseService
from lightrag.api.metadata_store import SQLiteMetadataStore
from lightrag.api.routers.chat_routes import create_chat_routes
from lightrag.api.utils_api import get_combined_auth_dependency
sys.argv = _original_argv

pytestmark = pytest.mark.offline

_API_KEY = "test-key"
_DEFAULT_SESSION_NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


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
    dependency = create_chat_routes.__globals__["get_combined_auth_dependency"]
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


def _build_chat_client(
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
    app.include_router(create_chat_routes(api_key=api_key))
    return TestClient(app), user_service, metadata_store, alice, bob


def test_chat_project_lifecycle(monkeypatch, tmp_path):
    client, user_service, _store, alice, _bob = _build_chat_client(
        monkeypatch, tmp_path
    )
    headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}

    created = client.post(
        "/chat/projects", json={"name": "  胎侧配方调研  "}, headers=headers
    )
    assert created.status_code == 200, created.text
    project = created.json()
    assert project["name"] == "胎侧配方调研"
    assert project["user_id"] == alice.id
    assert project["id"].startswith("proj_")

    # Blank names are rejected: whitespace-only -> 400, empty -> 422 (pydantic).
    assert (
        client.post("/chat/projects", json={"name": "   "}, headers=headers).status_code
        == 400
    )
    assert (
        client.post("/chat/projects", json={"name": ""}, headers=headers).status_code
        == 422
    )

    listed = client.get("/chat/projects", headers=headers)
    assert listed.status_code == 200
    body = listed.json()
    assert body["total"] == 1
    assert [item["id"] for item in body["projects"]] == [project["id"]]

    detail = client.get(f"/chat/projects/{project['id']}", headers=headers)
    assert detail.status_code == 200
    assert detail.json()["name"] == "胎侧配方调研"

    renamed = client.patch(
        f"/chat/projects/{project['id']}", json={"name": "新名字"}, headers=headers
    )
    assert renamed.status_code == 200
    assert renamed.json()["name"] == "新名字"
    assert renamed.json()["updated_at"] >= project["updated_at"]

    deleted = client.delete(f"/chat/projects/{project['id']}", headers=headers)
    assert deleted.status_code == 200
    assert deleted.json() == {
        "id": project["id"],
        "deleted": True,
        "deleted_sessions": 0,
        "deleted_messages": 0,
    }
    assert (
        client.get(f"/chat/projects/{project['id']}", headers=headers).status_code
        == 404
    )
    assert (
        client.delete(f"/chat/projects/{project['id']}", headers=headers).status_code
        == 404
    )
    assert (
        client.patch(
            f"/chat/projects/{project['id']}", json={"name": "x"}, headers=headers
        ).status_code
        == 404
    )


def test_chat_session_lifecycle_and_default_time_name(monkeypatch, tmp_path):
    client, user_service, _store, alice, _bob = _build_chat_client(
        monkeypatch, tmp_path
    )
    headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}

    project_id = client.post(
        "/chat/projects", json={"name": "项目A"}, headers=headers
    ).json()["id"]

    # Omitted body -> session named after creation time.
    default_named = client.post(
        f"/chat/projects/{project_id}/sessions", headers=headers
    )
    assert default_named.status_code == 200, default_named.text
    default_session = default_named.json()
    assert _DEFAULT_SESSION_NAME_RE.match(default_session["name"])
    assert default_session["project_id"] == project_id
    assert default_session["user_id"] == alice.id
    assert default_session["id"].startswith("sess_")
    assert default_session["context_rounds"] == 1

    # Blank name also falls back to the time-based default.
    blank_named = client.post(
        f"/chat/projects/{project_id}/sessions", json={"name": "   "}, headers=headers
    )
    assert blank_named.status_code == 200
    assert _DEFAULT_SESSION_NAME_RE.match(blank_named.json()["name"])

    custom = client.post(
        f"/chat/projects/{project_id}/sessions",
        json={"name": "低温屈挠专题"},
        headers=headers,
    )
    assert custom.status_code == 200
    custom_session = custom.json()
    assert custom_session["name"] == "低温屈挠专题"

    listed = client.get(f"/chat/projects/{project_id}/sessions", headers=headers)
    assert listed.status_code == 200
    body = listed.json()
    assert body["total"] == 3
    assert {item["id"] for item in body["sessions"]} == {
        default_session["id"],
        blank_named.json()["id"],
        custom_session["id"],
    }

    detail = client.get(
        f"/chat/projects/{project_id}/sessions/{custom_session['id']}",
        headers=headers,
    )
    assert detail.status_code == 200

    renamed = client.patch(
        f"/chat/projects/{project_id}/sessions/{custom_session['id']}",
        json={"name": "改名会话"},
        headers=headers,
    )
    assert renamed.status_code == 200
    assert renamed.json()["name"] == "改名会话"

    deleted = client.delete(
        f"/chat/projects/{project_id}/sessions/{custom_session['id']}",
        headers=headers,
    )
    assert deleted.status_code == 200
    assert deleted.json() == {
        "id": custom_session["id"],
        "project_id": project_id,
        "deleted": True,
        "deleted_messages": 0,
    }
    assert (
        client.get(
            f"/chat/projects/{project_id}/sessions/{custom_session['id']}",
            headers=headers,
        ).status_code
        == 404
    )

    # Session endpoints under a missing project are 404s.
    assert (
        client.post("/chat/projects/proj_missing/sessions", headers=headers).status_code
        == 404
    )
    assert (
        client.get("/chat/projects/proj_missing/sessions", headers=headers).status_code
        == 404
    )

    # Deleting the project cascades the remaining sessions.
    project_deleted = client.delete(f"/chat/projects/{project_id}", headers=headers)
    assert project_deleted.status_code == 200
    assert project_deleted.json()["deleted_sessions"] == 2
    assert (
        client.get(
            f"/chat/projects/{project_id}/sessions/{default_session['id']}",
            headers=headers,
        ).status_code
        == 404
    )


def test_chat_session_context_rounds(monkeypatch, tmp_path):
    # Deployment default comes from CHAT_SESSION_DEFAULT_CONTEXT_ROUNDS.
    args = _enterprise_args(chat_session_default_context_rounds=3)
    client, user_service, _store, alice, _bob = _build_chat_client(
        monkeypatch, tmp_path, args=args
    )
    headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    project_id = client.post(
        "/chat/projects", json={"name": "轮次项目"}, headers=headers
    ).json()["id"]
    sessions_url = f"/chat/projects/{project_id}/sessions"

    # Omitted -> env-configured default.
    from_env = client.post(sessions_url, headers=headers)
    assert from_env.status_code == 200, from_env.text
    assert from_env.json()["context_rounds"] == 3

    # Explicit values win over the default; -1 means "send full history".
    explicit = client.post(
        sessions_url, json={"context_rounds": 10}, headers=headers
    )
    assert explicit.status_code == 200
    session = explicit.json()
    assert session["context_rounds"] == 10
    unlimited = client.post(
        sessions_url, json={"context_rounds": -1}, headers=headers
    )
    assert unlimited.status_code == 200
    assert unlimited.json()["context_rounds"] == -1

    # Invalid values are rejected on create and update.
    assert (
        client.post(sessions_url, json={"context_rounds": 0}, headers=headers)
        .status_code
        == 400
    )
    assert (
        client.post(sessions_url, json={"context_rounds": -2}, headers=headers)
        .status_code
        == 400
    )
    session_url = f"{sessions_url}/{session['id']}"
    assert (
        client.patch(session_url, json={"context_rounds": 0}, headers=headers)
        .status_code
        == 400
    )

    # Empty PATCH body is rejected.
    assert client.patch(session_url, json={}, headers=headers).status_code == 400

    # Rounds-only update keeps the name; name-only update keeps the rounds.
    rounds_only = client.patch(
        session_url, json={"context_rounds": 5}, headers=headers
    )
    assert rounds_only.status_code == 200
    assert rounds_only.json()["context_rounds"] == 5
    assert rounds_only.json()["name"] == session["name"]
    name_only = client.patch(session_url, json={"name": "改名"}, headers=headers)
    assert name_only.status_code == 200
    assert name_only.json()["name"] == "改名"
    assert name_only.json()["context_rounds"] == 5
    both = client.patch(
        session_url, json={"name": "再改", "context_rounds": -1}, headers=headers
    )
    assert both.status_code == 200
    assert both.json()["name"] == "再改"
    assert both.json()["context_rounds"] == -1


def test_chat_message_sync_lifecycle(monkeypatch, tmp_path):
    client, user_service, _store, alice, bob = _build_chat_client(
        monkeypatch, tmp_path
    )
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}

    project_id = client.post(
        "/chat/projects", json={"name": "同步项目"}, headers=alice_headers
    ).json()["id"]
    session_id = client.post(
        f"/chat/projects/{project_id}/sessions",
        json={"name": "会话A"},
        headers=alice_headers,
    ).json()["id"]
    messages_url = f"/chat/projects/{project_id}/sessions/{session_id}/messages"

    references = [{"reference_id": "A1", "kb_id": "kb_x", "file_path": "a.pdf"}]
    appended = client.post(
        messages_url,
        json={
            "messages": [
                {"role": "user", "content": "低温屈挠性怎么提升？"},
                {
                    "role": "assistant",
                    "content": "建议 NR/BR 并用… [A1]",
                    "metadata": {"references": references, "mode": "mix"},
                },
            ]
        },
        headers=alice_headers,
    )
    assert appended.status_code == 200, appended.text
    batch = appended.json()
    assert batch["session_id"] == session_id
    assert [m["seq"] for m in batch["messages"]] == [1, 2]
    assert batch["messages"][0]["id"].startswith("msg_")
    assert batch["messages"][1]["metadata"]["references"] == references

    second = client.post(
        messages_url,
        json={"messages": [{"role": "user", "content": "换成 EPDM 呢？"}]},
        headers=alice_headers,
    )
    assert second.status_code == 200
    assert [m["seq"] for m in second.json()["messages"]] == [3]

    # Another browser/device replays the full history from the server.
    listed = client.get(messages_url, headers=alice_headers)
    assert listed.status_code == 200
    body = listed.json()
    assert body["total"] == 3
    assert [m["seq"] for m in body["messages"]] == [1, 2, 3]
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assert body["messages"][1]["metadata"]["mode"] == "mix"
    page = client.get(
        f"{messages_url}?limit=2&offset=1", headers=alice_headers
    ).json()
    assert page["total"] == 3
    assert [m["seq"] for m in page["messages"]] == [2, 3]

    # Validation: unknown role / empty content / oversized batch or metadata.
    assert (
        client.post(
            messages_url,
            json={"messages": [{"role": "tool", "content": "x"}]},
            headers=alice_headers,
        ).status_code
        == 422
    )
    assert (
        client.post(
            messages_url,
            json={"messages": [{"role": "user", "content": ""}]},
            headers=alice_headers,
        ).status_code
        == 422
    )
    assert (
        client.post(
            messages_url,
            json={"messages": [{"role": "user", "content": "x"}] * 21},
            headers=alice_headers,
        ).status_code
        == 422
    )
    assert (
        client.post(
            messages_url,
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "x",
                        "metadata": {"blob": "y" * (64 * 1024 + 1)},
                    }
                ]
            },
            headers=alice_headers,
        ).status_code
        == 400
    )

    # Per-user isolation: bob cannot read, append or delete alice's messages.
    assert client.get(messages_url, headers=bob_headers).status_code == 404
    assert (
        client.post(
            messages_url,
            json={"messages": [{"role": "user", "content": "hax"}]},
            headers=bob_headers,
        ).status_code
        == 404
    )
    first_message_id = body["messages"][0]["id"]
    assert (
        client.delete(
            f"{messages_url}/{first_message_id}", headers=bob_headers
        ).status_code
        == 404
    )

    # Appending bumps the session's recency: a newer-but-idle session sorts
    # below the active one after new messages arrive.
    other_session_id = client.post(
        f"/chat/projects/{project_id}/sessions",
        json={"name": "会话B"},
        headers=alice_headers,
    ).json()["id"]
    client.post(
        messages_url,
        json={"messages": [{"role": "user", "content": "再问一句"}]},
        headers=alice_headers,
    )
    session_list = client.get(
        f"/chat/projects/{project_id}/sessions", headers=alice_headers
    ).json()
    assert [s["id"] for s in session_list["sessions"]][0] == session_id
    assert {s["id"] for s in session_list["sessions"]} == {
        session_id,
        other_session_id,
    }

    # Delete one message, then the messages under a missing session are 404s.
    deleted = client.delete(
        f"{messages_url}/{first_message_id}", headers=alice_headers
    )
    assert deleted.status_code == 200
    assert deleted.json() == {
        "id": first_message_id,
        "session_id": session_id,
        "project_id": project_id,
        "deleted": True,
    }
    assert (
        client.delete(
            f"{messages_url}/{first_message_id}", headers=alice_headers
        ).status_code
        == 404
    )
    assert client.get(messages_url, headers=alice_headers).json()["total"] == 3

    # Session delete reports how many messages went with it.
    session_deleted = client.delete(
        f"/chat/projects/{project_id}/sessions/{session_id}", headers=alice_headers
    )
    assert session_deleted.status_code == 200
    assert session_deleted.json()["deleted_messages"] == 3
    assert (
        client.get(messages_url, headers=alice_headers).status_code == 404
    )


def test_chat_records_are_isolated_per_user(monkeypatch, tmp_path):
    client, user_service, _store, alice, bob = _build_chat_client(
        monkeypatch, tmp_path
    )
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    bob_headers = {"Authorization": f"Bearer {_token(user_service, bob)}"}

    project_id = client.post(
        "/chat/projects", json={"name": "alice-only"}, headers=alice_headers
    ).json()["id"]
    session_id = client.post(
        f"/chat/projects/{project_id}/sessions", headers=alice_headers
    ).json()["id"]

    assert client.get("/chat/projects", headers=bob_headers).json()["total"] == 0
    assert (
        client.get(f"/chat/projects/{project_id}", headers=bob_headers).status_code
        == 404
    )
    assert (
        client.patch(
            f"/chat/projects/{project_id}", json={"name": "hax"}, headers=bob_headers
        ).status_code
        == 404
    )
    assert (
        client.delete(f"/chat/projects/{project_id}", headers=bob_headers).status_code
        == 404
    )
    assert (
        client.post(
            f"/chat/projects/{project_id}/sessions", headers=bob_headers
        ).status_code
        == 404
    )
    assert (
        client.get(
            f"/chat/projects/{project_id}/sessions/{session_id}", headers=bob_headers
        ).status_code
        == 404
    )
    assert (
        client.delete(
            f"/chat/projects/{project_id}/sessions/{session_id}", headers=bob_headers
        ).status_code
        == 404
    )

    # Alice still sees her records untouched.
    assert client.get("/chat/projects", headers=alice_headers).json()["total"] == 1
    assert (
        client.get(
            f"/chat/projects/{project_id}/sessions/{session_id}",
            headers=alice_headers,
        ).status_code
        == 200
    )


def test_chat_rejects_api_key_principal(monkeypatch, tmp_path):
    args = _enterprise_args(enterprise_legacy_api_key_superadmin=True)
    client, _user_service, _store, _alice, _bob = _build_chat_client(
        monkeypatch, tmp_path, args=args
    )
    legacy_api_key_headers = {"X-API-Key": _API_KEY}

    assert (
        client.get("/chat/projects", headers=legacy_api_key_headers).status_code == 403
    )
    assert (
        client.post(
            "/chat/projects", json={"name": "x"}, headers=legacy_api_key_headers
        ).status_code
        == 403
    )


def test_chat_requires_authentication(monkeypatch, tmp_path):
    client, _user_service, _store, _alice, _bob = _build_chat_client(
        monkeypatch, tmp_path
    )
    response = client.get("/chat/projects")
    assert response.status_code in (401, 403)
