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
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
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
    "LIGHTRAG_CHAT_MEMORY_MAINTENANCE_ENABLED",
    "LIGHTRAG_CHAT_MEMORY_WORKER_POLL_SECONDS",
    "LIGHTRAG_CHAT_MEMORY_WORKER_RECOVERY_INTERVAL_SECONDS",
    "LIGHTRAG_CHAT_MEMORY_WORKER_SIDE_EFFECT_TIMEOUT_SECONDS",
    "LIGHTRAG_CHAT_MEMORY_WORKER_SHUTDOWN_TIMEOUT_SECONDS",
    "LIGHTRAG_CHAT_MEMORY_OPERATION_TIMEOUT_SECONDS",
    "LIGHTRAG_CHAT_MEMORY_REBUILD_MAX_MESSAGES",
    "LIGHTRAG_CHAT_MEMORY_REBUILD_MAX_BYTES",
    "LIGHTRAG_CHAT_MEMORY_STORE_RAW_EPISODE_CONTENT",
    "LIGHTRAG_CHAT_MEMORY_PROMPT_MAX_TOKENS",
    "LIGHTRAG_CHAT_MEMORY_PROMPT_MAX_CHARS",
    "LIGHTRAG_CHAT_MEMORY_ALLOW_CROSS_PROVIDER_QUERY_EGRESS",
    "MEMORY_MAINTENANCE_ENABLED",
    "MEMORY_WORKER_POLL_SECONDS",
    "MEMORY_WORKER_RECOVERY_INTERVAL_SECONDS",
    "MEMORY_WORKER_SIDE_EFFECT_TIMEOUT_SECONDS",
    "MEMORY_WORKER_SHUTDOWN_TIMEOUT_SECONDS",
    "MEMORY_SIDE_EFFECT_TIMEOUT_SECONDS",
    "MEMORY_OPERATION_TIMEOUT_SECONDS",
    "MEMORY_GRAPHITI_TIMEOUT_SECONDS",
    "MEMORY_REBUILD_MAX_MESSAGES",
    "MEMORY_REBUILD_MAX_BYTES",
    "MEMORY_STORE_RAW_EPISODE_CONTENT",
    "MEMORY_PROMPT_MAX_TOKENS",
    "MEMORY_PROMPT_MAX_CHARS",
    "MEMORY_ALLOW_CROSS_PROVIDER_QUERY_EGRESS",
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
    "MEMORY_NEO4J_DEPLOYMENT_ID",
    "NEO4J_URI",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
    "NEO4J_DATABASE",
)


def _make_app(
    tmp_path,
    monkeypatch,
    *,
    enterprise: bool,
    memory_enabled: bool,
    metadata_backend: str | None = None,
    memory_env: dict[str, str] | None = None,
    clear_binding_hosts_before_create: bool = False,
):
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
    monkeypatch.setenv(
        "LIGHTRAG_KB_METADATA_BACKEND",
        metadata_backend or ("postgres" if memory_enabled else "local"),
    )
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
    for key, value in (memory_env or {}).items():
        monkeypatch.setenv(key, value)

    from lightrag.api.config import parse_args

    original_argv = sys.argv.copy()
    sys.argv = ["lightrag-server"]
    try:
        args = parse_args()
        if clear_binding_hosts_before_create:
            args.llm_binding_host = None
            args.embedding_binding_host = None
        with (
            patch("lightrag.api.lightrag_server.LightRAG") as mock_rag,
            patch(
                "lightrag.api.lightrag_server.PostgresKnowledgeBaseService.from_env",
                return_value=MagicMock(),
            ),
            patch(
                "lightrag.api.lightrag_server.PostgresMetadataStore.from_env",
                return_value=MagicMock(),
            ),
        ):
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
    app = _make_app(
        tmp_path,
        monkeypatch,
        enterprise=True,
        memory_enabled=True,
        memory_env={"MEMORY_NEO4J_DEPLOYMENT_ID": "memory-cluster-a"},
    )
    service = app.state.enterprise_chat_memory_service
    assert service is not None, "memory service must be wired when enabled"

    config = service.config
    assert config.enabled is True
    assert config.maintenance_enabled is True
    assert config.store_raw_episode_content is False
    assert config.prompt_max_tokens == 1024
    assert config.prompt_max_chars == 8192
    assert config.allow_cross_provider_query_egress is False
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
    assert config.neo4j_deployment_id == "memory-cluster-a"
    assert config.worker_shutdown_timeout_seconds == 10.0
    # Construction must not have touched the backend yet (lazy init).
    assert service.available is False
    worker = app.state.enterprise_chat_memory_worker
    assert worker is not None
    assert worker.event_types == ("ingest", "rebuild", "purge")
    assert worker.runtime_fingerprint == config.runtime_fingerprint()
    assert worker.extraction_fingerprint == config.extraction_fingerprint()
    assert worker.graph_store_fingerprint == config.graph_store_fingerprint()
    assert app.state.enterprise_chat_memory_runtime_fingerprint == (
        config.extraction_fingerprint()
    )
    assert app.state.enterprise_chat_memory_graph_store_fingerprint == (
        config.graph_store_fingerprint()
    )
    conversation_service = app.state.enterprise_chat_conversation_service
    user_service = app.state.enterprise_user_service
    for wired_service in (conversation_service, user_service):
        assert wired_service.memory_admission_enabled is True
        assert (
            wired_service.memory_extraction_fingerprint
            == config.extraction_fingerprint()
        )
        assert (
            wired_service.memory_graph_store_fingerprint
            == config.graph_store_fingerprint()
        )
        assert wired_service.memory_maintenance_configured is True
        assert wired_service._post_commit_nudge.__self__ is worker
    assert service._post_commit_nudge.__self__ is worker


def test_memory_read_render_and_egress_env_settings(tmp_path, monkeypatch):
    app = _make_app(
        tmp_path,
        monkeypatch,
        enterprise=True,
        memory_enabled=True,
        memory_env={
            "LIGHTRAG_CHAT_MEMORY_PROMPT_MAX_TOKENS": "2048",
            "LIGHTRAG_CHAT_MEMORY_PROMPT_MAX_CHARS": "16384",
            "LIGHTRAG_CHAT_MEMORY_ALLOW_CROSS_PROVIDER_QUERY_EGRESS": "true",
        },
    )
    config = app.state.enterprise_chat_memory_service.config
    assert config.prompt_max_tokens == 2048
    assert config.prompt_max_chars == 16384
    assert config.allow_cross_provider_query_egress is True


def test_programmatic_create_app_resolves_default_hosts_before_memory_config(
    tmp_path, monkeypatch
):
    app = _make_app(
        tmp_path,
        monkeypatch,
        enterprise=True,
        memory_enabled=True,
        clear_binding_hosts_before_create=True,
    )

    config = app.state.enterprise_chat_memory_config
    service = app.state.enterprise_chat_memory_service
    assert service is not None
    assert service.config is config
    assert config.llm_base_url == "https://api.openai.com/v1"
    assert config.embedding_base_url == "https://api.openai.com/v1"


def test_memory_service_absent_when_flag_off(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch, enterprise=True, memory_enabled=False)
    assert app.state.enterprise_chat_memory_service is None
    assert app.state.enterprise_chat_memory_config.enabled is False
    assert app.state.enterprise_chat_memory_config.maintenance_enabled is True
    assert app.state.enterprise_chat_memory_worker is None


def test_feature_off_resolves_durable_config_without_starting_worker(
    tmp_path, monkeypatch
):
    app = _make_app(
        tmp_path,
        monkeypatch,
        enterprise=True,
        memory_enabled=False,
        memory_env={
            "LIGHTRAG_CHAT_MEMORY_MAINTENANCE_ENABLED": "false",
            "LIGHTRAG_CHAT_MEMORY_WORKER_POLL_SECONDS": "0",
            "LIGHTRAG_CHAT_MEMORY_WORKER_RECOVERY_INTERVAL_SECONDS": "99999",
            "LIGHTRAG_CHAT_MEMORY_OPERATION_TIMEOUT_SECONDS": "12.5",
            "LIGHTRAG_CHAT_MEMORY_WORKER_SHUTDOWN_TIMEOUT_SECONDS": "99999",
            "LIGHTRAG_CHAT_MEMORY_REBUILD_MAX_MESSAGES": "2500",
            "LIGHTRAG_CHAT_MEMORY_REBUILD_MAX_BYTES": "1048576",
            "LIGHTRAG_CHAT_MEMORY_STORE_RAW_EPISODE_CONTENT": "true",
        },
    )

    config = app.state.enterprise_chat_memory_config
    assert config.enabled is False
    assert config.maintenance_enabled is False
    assert config.worker_poll_interval_seconds == 0.05
    assert config.worker_recovery_interval_seconds == 3600.0
    assert config.worker_side_effect_timeout_seconds == 12.5
    assert config.worker_shutdown_timeout_seconds == 300.0
    assert config.rebuild_max_messages == 2500
    assert config.rebuild_max_bytes == 1_048_576
    assert config.store_raw_episode_content is True
    assert app.state.enterprise_chat_memory_service is None
    assert app.state.enterprise_chat_memory_worker is None
    assert (
        app.state.enterprise_chat_conversation_service.memory_admission_enabled
        is False
    )
    assert (
        app.state.enterprise_chat_conversation_service.memory_maintenance_configured
        is False
    )
    assert app.state.enterprise_user_service.memory_maintenance_configured is False


def test_maintenance_only_postgres_wires_lazy_service_and_worker(
    tmp_path, monkeypatch
):
    app = _make_app(
        tmp_path,
        monkeypatch,
        enterprise=True,
        memory_enabled=False,
        metadata_backend="postgres",
    )

    assert app.state.enterprise_chat_memory_service is None
    service = app.state.enterprise_chat_memory_maintenance_service
    worker = app.state.enterprise_chat_memory_worker
    assert service is not None
    assert service.available is False
    assert worker is not None
    assert worker.event_types == ("rebuild", "purge")
    conversation_service = app.state.enterprise_chat_conversation_service
    user_service = app.state.enterprise_user_service
    for wired_service in (conversation_service, user_service):
        assert wired_service.memory_admission_enabled is False
        assert (
            wired_service.memory_extraction_fingerprint
            == service.config.extraction_fingerprint()
        )
        assert (
            wired_service.memory_graph_store_fingerprint
            == service.config.graph_store_fingerprint()
        )
        assert wired_service.memory_maintenance_configured is True
        assert wired_service._post_commit_nudge.__self__ is worker


def test_maintenance_worker_lifespan_is_lazy_and_stops_before_service(
    tmp_path, monkeypatch
):
    app = _make_app(
        tmp_path,
        monkeypatch,
        enterprise=True,
        memory_enabled=False,
        metadata_backend="postgres",
    )
    service = app.state.enterprise_chat_memory_maintenance_service
    worker = app.state.enterprise_chat_memory_worker
    assert service is not None and worker is not None
    shutdown_order: list[str] = []

    service.initialize = AsyncMock()
    service.finalize = AsyncMock(
        side_effect=lambda: shutdown_order.append("service.finalize")
    )
    worker.start = MagicMock()

    async def stop_worker():
        shutdown_order.append("worker.stop")

    worker.stop = AsyncMock(side_effect=stop_worker)
    app.state.kb_service.initialize = AsyncMock()
    app.state.kb_service.close = AsyncMock()
    app.state.metadata_store.initialize = AsyncMock()
    app.state.metadata_store.close = AsyncMock()
    app.state.enterprise_settings_service.initialize_registration_setting = AsyncMock()
    app.state.enterprise_user_service.bootstrap_super_admin = AsyncMock()
    app.state.job_service.recover_orphan_jobs = AsyncMock(return_value=[])
    app.state.lightrag_registry.shutdown = AsyncMock()

    with patch("lightrag.api.lightrag_server.finalize_share_data"):
        with TestClient(app) as client:
            assert client.portal is not None
            worker.start.assert_called_once_with()
            service.initialize.assert_not_awaited()

    assert shutdown_order[:2] == ["worker.stop", "service.finalize"]


def test_maintenance_shutdown_continues_after_worker_and_service_failures(
    tmp_path, monkeypatch
):
    app = _make_app(
        tmp_path,
        monkeypatch,
        enterprise=True,
        memory_enabled=False,
        metadata_backend="postgres",
    )
    service = app.state.enterprise_chat_memory_maintenance_service
    worker = app.state.enterprise_chat_memory_worker
    assert service is not None and worker is not None

    service.initialize = AsyncMock()
    service.finalize = AsyncMock(side_effect=RuntimeError("service close failed"))
    worker.start = MagicMock()
    worker.stop = AsyncMock(side_effect=RuntimeError("worker stop failed"))
    app.state.kb_service.initialize = AsyncMock()
    app.state.kb_service.close = AsyncMock()
    app.state.metadata_store.initialize = AsyncMock()
    app.state.metadata_store.close = AsyncMock()
    app.state.enterprise_settings_service.initialize_registration_setting = AsyncMock()
    app.state.enterprise_user_service.bootstrap_super_admin = AsyncMock()
    app.state.job_service.recover_orphan_jobs = AsyncMock(return_value=[])
    app.state.lightrag_registry.shutdown = AsyncMock()

    with patch("lightrag.api.lightrag_server.finalize_share_data") as finalize_shared:
        with TestClient(app):
            service.initialize.assert_not_awaited()

    worker.stop.assert_awaited_once_with()
    service.finalize.assert_awaited_once_with()
    app.state.lightrag_registry.shutdown.assert_awaited_once_with()
    app.state.metadata_store.close.assert_awaited_once_with()
    app.state.kb_service.close.assert_awaited_once_with()
    finalize_shared.assert_called_once_with()


def test_memory_enabled_rejects_non_enterprise_mode(tmp_path, monkeypatch):
    with pytest.raises(
        ValueError, match="LIGHTRAG_ENTERPRISE_AUTH_ENABLED=true"
    ):
        _make_app(tmp_path, monkeypatch, enterprise=False, memory_enabled=True)


def test_memory_enabled_rejects_local_metadata_backend(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="LIGHTRAG_KB_METADATA_BACKEND=postgres"):
        _make_app(
            tmp_path,
            monkeypatch,
            enterprise=True,
            memory_enabled=True,
            metadata_backend="local",
        )


def test_create_app_revalidates_programmatic_memory_configuration():
    from lightrag.api.lightrag_server import create_app

    with pytest.raises(
        ValueError, match="LIGHTRAG_ENTERPRISE_AUTH_ENABLED=true"
    ):
        create_app(
            SimpleNamespace(
                chat_memory_enabled=True,
                enterprise_auth_enabled=False,
                kb_metadata_backend="postgres",
            )
        )

    with pytest.raises(ValueError, match="LIGHTRAG_KB_METADATA_BACKEND=postgres"):
        create_app(
            SimpleNamespace(
                chat_memory_enabled=True,
                enterprise_auth_enabled=True,
                kb_metadata_backend="local",
            )
        )
