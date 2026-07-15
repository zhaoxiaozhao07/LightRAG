from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import APIRouter, HTTPException


pytestmark = pytest.mark.offline


def test_create_app_injects_kb_service_into_enterprise_control_plane(
    tmp_path, monkeypatch
):
    isolated_env = (
        "LIGHTRAG_KB_METADATA_BACKEND",
        "LIGHTRAG_OBJECT_STORAGE",
        "LIGHTRAG_KB_JOB_WORKER",
        "LIGHTRAG_CHAT_MEMORY_ENABLED",
        "LLM_BINDING",
        "LLM_BINDING_HOST",
        "LLM_BINDING_API_KEY",
        "LLM_MODEL",
        "EMBEDDING_BINDING",
        "EMBEDDING_BINDING_HOST",
        "EMBEDDING_BINDING_API_KEY",
        "EMBEDDING_MODEL",
        "EMBEDDING_DIM",
        "RERANK_BINDING",
        "LIGHTRAG_ENTERPRISE_AUTH_ENABLED",
    )
    for name in isolated_env:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("LLM_BINDING_HOST", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_BINDING_API_KEY", "test-llm-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("EMBEDDING_BINDING", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING_HOST", "https://api.openai.com/v1")
    monkeypatch.setenv("EMBEDDING_BINDING_API_KEY", "test-embedding-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_DIM", "1536")
    monkeypatch.setenv("RERANK_BINDING", "null")
    monkeypatch.setenv("WORKING_DIR", str(tmp_path / "rag_storage"))
    monkeypatch.setenv("INPUT_DIR", str(tmp_path / "inputs"))
    monkeypatch.setenv("LIGHTRAG_ENTERPRISE_AUTH_ENABLED", "true")
    monkeypatch.setenv("TOKEN_SECRET", "enterprise-wiring-secret")
    monkeypatch.setenv("LIGHTRAG_SUPER_ADMIN_USERNAME", "admin")
    monkeypatch.setenv("LIGHTRAG_SUPER_ADMIN_PASSWORD", "admin-pass")

    from lightrag.api.config import parse_args
    from lightrag.api import lightrag_server

    captured: dict[str, object] = {}

    def enterprise_router_probe(*, api_key=None, kb_service=None):
        captured["api_key"] = api_key
        captured["kb_service"] = kb_service
        return APIRouter()

    monkeypatch.setattr(
        lightrag_server, "create_enterprise_routes", enterprise_router_probe
    )
    original_argv = sys.argv.copy()
    sys.argv = ["lightrag-server"]
    try:
        args = parse_args()
        # config may already be imported by a broader test run; pin paths on
        # the parsed object as well as through the environment.
        args.working_dir = str(tmp_path / "rag_storage")
        args.input_dir = str(tmp_path / "inputs")
        with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
            fake_rag = MagicMock()
            fake_rag.initialize_storages = AsyncMock()
            fake_rag.check_and_migrate_data = AsyncMock()
            fake_rag.finalize_storages = AsyncMock()
            mock_rag.return_value = fake_rag
            app = lightrag_server.create_app(args)
    finally:
        sys.argv = original_argv

    assert captured["kb_service"] is app.state.kb_service
    assert app.state.enterprise_api_key_service._kb_service is app.state.kb_service

    login_endpoint = next(
        route.endpoint for route in app.routes if getattr(route, "path", None) == "/login"
    )
    username = f"tenant-login-{uuid4().hex}"

    async def reset_and_login():
        await app.state.metadata_store.initialize()
        user = await app.state.enterprise_user_service.create_user(
            username=username,
            password="initial-pass",
            can_create_kb=True,
            can_use_bypass_query=True,
            can_use_agent_query=True,
            can_delete_documents=True,
            can_download_files=True,
        )
        await app.state.enterprise_user_service.change_password(
            user.id, "reset-pass", actor_user_id=user.id
        )
        with pytest.raises(HTTPException) as exc_info:
            await login_endpoint(
                SimpleNamespace(username=username, password="initial-pass")
            )
        assert exc_info.value.status_code == 401
        return await login_endpoint(
            SimpleNamespace(username=username, password="reset-pass")
        )

    login_response = asyncio.run(reset_and_login())
    assert all(
        login_response["user"][field] is True
        for field in (
            "can_create_kb",
            "can_use_bypass_query",
            "can_use_agent_query",
            "can_delete_documents",
            "can_download_files",
        )
    )
