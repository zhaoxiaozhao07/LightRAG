"""Server-integration wiring for the graphiti-backed chat memory service.

``tests/api/test_chat_memory_service.py`` covers the service runtime behavior
(with injected fakes) and ``tests/api/routes/test_chat_memory_routes.py``
covers the endpoint contract against a fake service. This file closes the
remaining gap: ``create_app`` builds ``app.state.enterprise_chat_memory_service``
only for enterprise deployments with ``LIGHTRAG_CHAT_MEMORY_ENABLED=true``,
resolving the documented config fallback chain — and omits it otherwise.

The lifespan initialize/finalize calls are one-line delegations to methods
exercised by the unit tests; entering the lifespan here would dial the
configured Neo4j, so these tests stay at construction level.
"""

from __future__ import annotations

import sys

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

pytestmark = pytest.mark.offline

# Env that the project's .env may populate at config import time; clear so the
# test is hermetic (notably the memory flag and the MEMORY_*/NEO4J_* fallbacks).
_ENV_TO_ISOLATE = (
    "LLM_BINDING",
    "EMBEDDING_BINDING",
    "LLM_BINDING_HOST",
    "LLM_BINDING_API_KEY",
    "LLM_MODEL",
    "EMBEDDING_BINDING_HOST",
    "EMBEDDING_BINDING_API_KEY",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "RERANK_BINDING",
    "LIGHTRAG_KV_STORAGE",
    "LIGHTRAG_VECTOR_STORAGE",
    "LIGHTRAG_GRAPH_STORAGE",
    "LIGHTRAG_DOC_STATUS_STORAGE",
    "LIGHTRAG_KB_METADATA_BACKEND",
    "LIGHTRAG_OBJECT_STORAGE",
    "LIGHTRAG_KB_JOB_WORKER",
    "LIGHTRAG_ENTERPRISE_AUTH_ENABLED",
    "LIGHTRAG_CHAT_MEMORY_ENABLED",
    "QUERY_LLM_BINDING",
    "QUERY_LLM_MODEL",
    "QUERY_LLM_BINDING_HOST",
    "QUERY_LLM_BINDING_API_KEY",
    "MEMORY_LLM_BINDING_HOST",
    "MEMORY_LLM_BINDING_API_KEY",
    "MEMORY_LLM_MODEL",
    "MEMORY_LLM_SMALL_MODEL",
    "MEMORY_OPENAI_LLM_EXTRA_BODY",
    "MEMORY_EMBEDDING_BINDING_HOST",
    "MEMORY_EMBEDDING_BINDING_API_KEY",
    "MEMORY_EMBEDDING_MODEL",
    "MEMORY_EMBEDDING_DIM",
    "MEMORY_NEO4J_URI",
    "MEMORY_NEO4J_USERNAME",
    "MEMORY_NEO4J_PASSWORD",
    "MEMORY_NEO4J_DATABASE",
    "NEO4J_URI",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
    "NEO4J_DATABASE",
)


def _make_app(tmp_path, monkeypatch, *, enterprise: bool, memory_enabled: bool):
    for var in _ENV_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("LLM_BINDING_HOST", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_BINDING_API_KEY", "base-llm-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("EMBEDDING_BINDING", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING_HOST", "https://api.openai.com/v1")
    monkeypatch.setenv("EMBEDDING_BINDING_API_KEY", "embed-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_DIM", "1536")
    monkeypatch.setenv("RERANK_BINDING", "null")
    monkeypatch.setenv("WORKING_DIR", str(tmp_path / "rag_storage"))
    # Query-role override that the memory config must inherit.
    monkeypatch.setenv("QUERY_LLM_MODEL", "query-model")
    # Point the memory graph store at a closed local port; construction never
    # dials it (only lifespan initialize would), this just pins the fallback.
    monkeypatch.setenv("NEO4J_URI", "bolt://127.0.0.1:9")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "test-password")
    if enterprise:
        monkeypatch.setenv("LIGHTRAG_ENTERPRISE_AUTH_ENABLED", "true")
        monkeypatch.setenv("TOKEN_SECRET", "wiring-test-secret-not-default")
        monkeypatch.setenv("LIGHTRAG_SUPER_ADMIN_USERNAME", "admin")
        monkeypatch.setenv("LIGHTRAG_SUPER_ADMIN_PASSWORD", "admin-pass")
    if memory_enabled:
        monkeypatch.setenv("LIGHTRAG_CHAT_MEMORY_ENABLED", "true")

    from lightrag.api.config import parse_args

    original_argv = sys.argv.copy()
    sys.argv = ["lightrag-server"]
    try:
        args = parse_args()
        with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
            fake_rag = MagicMock()
            fake_rag.initialize_storages = AsyncMock()
            fake_rag.check_and_migrate_data = AsyncMock()
            fake_rag.finalize_storages = AsyncMock()
            mock_rag.return_value = fake_rag
            from lightrag.api.lightrag_server import create_app

            return create_app(args)
    finally:
        sys.argv = original_argv


def test_memory_service_wired_with_resolved_config(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch, enterprise=True, memory_enabled=True)
    service = app.state.enterprise_chat_memory_service
    assert service is not None, "memory service must be wired when enabled"

    config = service.config
    assert config.enabled is True
    # LLM chain: MEMORY_* unset -> QUERY_* model -> base host/key.
    assert config.llm_model == "query-model"
    assert config.llm_small_model == "query-model"
    assert config.llm_base_url == "https://api.openai.com/v1"
    assert config.llm_api_key == "base-llm-key"
    # Embedding chain inherits the deployment settings.
    assert config.embedding_model == "text-embedding-3-small"
    assert config.embedding_dim == 1536
    assert config.embedding_api_key == "embed-key"
    # Graph store chain inherits NEO4J_*.
    assert config.neo4j_uri == "bolt://127.0.0.1:9"
    assert config.neo4j_username == "neo4j"
    assert config.neo4j_database == "neo4j"
    # Construction must not have touched the backend yet (lazy init).
    assert service.available is False


def test_memory_service_absent_when_flag_off(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch, enterprise=True, memory_enabled=False)
    assert app.state.enterprise_chat_memory_service is None


def test_memory_service_absent_outside_enterprise_mode(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch, enterprise=False, memory_enabled=True)
    assert getattr(app.state, "enterprise_chat_memory_service", None) is None
