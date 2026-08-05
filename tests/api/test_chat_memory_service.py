"""Unit tests for the graphiti-backed chat memory service.

All tests run offline: graphiti is replaced by an injected fake via the
``graphiti_factory`` / ``clear_data_fn`` constructor hooks, so neither
``graphiti-core`` nor a live Neo4j/LLM is required.
"""

from __future__ import annotations

# ruff: noqa: E402 - LightRAG API modules parse sys.argv at import time.

import asyncio
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from types import ModuleType, SimpleNamespace

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from lightrag.api.chat_memory_service import (
    MEMORY_SEARCH_MAX_LIMIT,
    ChatMemoryConfig,
    ChatMemoryEventNotFoundError,
    ChatMemoryRetryConflictError,
    ChatMemoryService,
    ChatMemoryUnavailableError,
    _ExtraBodyAsyncOpenAI,
    _RerankFnCrossEncoder,
    _default_graphiti_factory,
)
from lightrag.api.kb_service import utc_now_iso
from lightrag.api.metadata_store import (
    ChatMessageRecord,
    ChatMemoryReadToken,
    ChatProjectRecord,
    ChatSessionRecord,
    EnterpriseUserRecord,
    SQLiteMetadataStore,
)
sys.argv = _original_argv

pytestmark = pytest.mark.offline


def test_module_import_is_lazy_without_graphiti():
    script = """
import builtins

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "graphiti_core" or name.startswith("graphiti_core."):
        raise RuntimeError("graphiti_core must not be imported at module import time")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import lightrag.api.chat_memory_service  # noqa: F401
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


class FakeGraphiti:
    def __init__(self):
        self.build_calls = 0
        self.episodes: list[dict] = []
        self.search_calls: list[dict] = []
        self.search_recipe_calls: list[dict] = []
        self.search_results: list = []
        self.markers: list[tuple[str, str]] = []
        self.add_episode_delay = 0.0
        self.add_episode_error: Exception | None = None
        self.removed_episodes: list[str] = []
        self.closed = False
        self.driver = object()

    async def build_indices_and_constraints(self):
        self.build_calls += 1

    async def add_episode(self, **kwargs):
        name = kwargs.get("name", "")
        self.markers.append(("start", name))
        if self.add_episode_delay:
            await asyncio.sleep(self.add_episode_delay)
        if self.add_episode_error is not None:
            self.markers.append(("end", name))
            raise self.add_episode_error
        self.episodes.append(kwargs)
        self.markers.append(("end", name))
        return SimpleNamespace(episode=SimpleNamespace(uuid=f"ep-{len(self.episodes)}"))

    async def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return self.search_results

    async def search_(self, **kwargs):
        self.search_recipe_calls.append(kwargs)
        return SimpleNamespace(edges=self.search_results)

    async def remove_episode(self, episode_uuid):
        self.removed_episodes.append(episode_uuid)

    async def close(self):
        self.closed = True


class FakeAuditService:
    def __init__(self, order: list[str] | None = None):
        self.events: list[tuple[str, dict]] = []
        self.order = order

    async def append(self, event_type, **kwargs):
        if self.order is not None:
            self.order.append("audit")
        self.events.append((event_type, kwargs))

    def of_type(self, event_type):
        return [event for event in self.events if event[0] == event_type]


class FakeClearData:
    def __init__(self):
        self.calls: list[tuple[object, list[str]]] = []

    async def __call__(self, graphiti, group_ids):
        assert group_ids, "clear_data must never receive an empty/None group list"
        self.calls.append((graphiti, list(group_ids)))


class FakeReadTokenStore:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.calls: list[tuple[str, str]] = []

    async def get_chat_memory_read_token(self, user_id, project_id):
        self.calls.append((user_id, project_id))
        if len(self.tokens) > 1:
            return self.tokens.pop(0)
        return self.tokens[0] if self.tokens else None


class FakeDurableAdminStore:
    def __init__(self):
        self.purge_calls: list[dict] = []

    async def enqueue_chat_memory_purge(
        self,
        user_id,
        project_id,
        config_fingerprint,
        **kwargs,
    ):
        self.purge_calls.append(
            {
                "user_id": user_id,
                "project_id": project_id,
                "config_fingerprint": config_fingerprint,
                **kwargs,
            }
        )
        if project_id == "proj-noop":
            return None
        return SimpleNamespace(event_id=f"event-{project_id}")

    async def get_chat_memory_outbox_stats(self):
        return SimpleNamespace(
            pending=2,
            running=1,
            retry_wait=3,
            dead_letter=4,
            oldest_available_at="2026-07-16T00:00:00+00:00",
            oldest_lag_seconds=5.5,
        )


class FakeRetryStore:
    def __init__(self, event=None, *, order: list[str] | None = None):
        self.events = {event.event_id: event} if event is not None else {}
        self.order = order if order is not None else []
        self.get_calls: list[str] = []
        self.requeue_calls: list[dict] = []
        self.group_create_calls: list[tuple[str, str]] = []

    async def get_chat_memory_event(self, event_id):
        self.get_calls.append(event_id)
        self.order.append("get")
        return self.events.get(event_id)

    async def requeue_chat_memory_purge(
        self,
        event_id,
        runtime_fingerprint,
        *,
        runtime_graph_store_fingerprint=None,
        retry_delay_seconds=5.0,
    ):
        self.requeue_calls.append(
            {
                "event_id": event_id,
                "runtime_fingerprint": runtime_fingerprint,
                "runtime_graph_store_fingerprint": (
                    runtime_graph_store_fingerprint
                ),
                "retry_delay_seconds": retry_delay_seconds,
            }
        )
        current = self.events[event_id]
        updated = SimpleNamespace(**vars(current))
        updated.status = "retry_wait"
        self.events[event_id] = updated
        self.order.append("requeue_commit")
        return updated


def _retry_event(
    config: ChatMemoryConfig,
    *,
    event_id: str = "evt-purge-1",
    event_type: str = "purge",
    status: str = "dead_letter",
    graph_store_fingerprint: str | None = None,
):
    return SimpleNamespace(
        event_id=event_id,
        user_id="usr-deleted",
        project_id="proj-deleted",
        event_type=event_type,
        status=status,
        graph_store_fingerprint=(
            graph_store_fingerprint or config.graph_store_fingerprint()
        ),
        last_error_message="private source content must not be audited",
    )


def _message(role: str, content: str, seq: int, created_at: str | None = None):
    return SimpleNamespace(
        role=role,
        content=content,
        seq=seq,
        created_at=created_at or "2026-07-10T08:00:05.000000+00:00",
    )


def _service(
    config: ChatMemoryConfig | None = None,
    fake: FakeGraphiti | None = None,
    audit: FakeAuditService | None = None,
    clear: FakeClearData | None = None,
    metadata_store=None,
):
    fake = fake or FakeGraphiti()
    clear = clear or FakeClearData()
    service = ChatMemoryService(
        config or ChatMemoryConfig(enabled=True),
        audit_service=audit,
        graphiti_factory=lambda _config: fake,
        clear_data_fn=clear,
        metadata_store=metadata_store,
    )
    return service, fake, clear


def _read_token(
    config: ChatMemoryConfig,
    *,
    state: str = "active",
    state_version: int = 1,
    generation: int | None = 1,
    graph_group_id: str | None = "physical-g1",
    generation_state: str | None = "active",
    extraction_fingerprint: str | None = None,
    graph_store_fingerprint: str | None = None,
) -> ChatMemoryReadToken:
    return ChatMemoryReadToken(
        user_id="usr_a",
        project_id="proj_b",
        state=state,  # type: ignore[arg-type]
        state_version=state_version,
        active_generation=generation,
        active_config_fingerprint=(
            extraction_fingerprint or config.extraction_fingerprint()
        ),
        active_graph_store_fingerprint=(
            graph_store_fingerprint or config.graph_store_fingerprint()
        ),
        graph_group_id=graph_group_id,
        generation_state=generation_state,  # type: ignore[arg-type]
    )


def _assert_current_fact_filter(search_call: dict) -> None:
    search_filter = search_call["search_filter"]
    assert search_filter.invalid_at[0][0].comparison_operator.value == "IS NULL"
    assert search_filter.expired_at[0][0].comparison_operator.value == "IS NULL"


# --------------------------------------------------------------------- config


def test_config_fallback_chain_inherits_query_then_base():
    args = SimpleNamespace(
        chat_memory_enabled=True,
        llm_binding_host="http://base:8000/v1",
        llm_binding_api_key="base-key",
        llm_model="base-model",
        query_llm_binding_host=None,
        query_llm_binding_api_key="query-key",
        query_llm_model="query-model",
        memory_llm_model="memory-model",
        embedding_binding_host="http://embed:8002/v1",
        embedding_binding_api_key="embed-key",
        embedding_model="embed-model",
        embedding_dim=4096,
    )
    config = ChatMemoryConfig.from_args(args)
    assert config.enabled is True
    # MEMORY_* wins, then QUERY_*, then base LLM_*.
    assert config.llm_model == "memory-model"
    assert config.llm_api_key == "query-key"
    assert config.llm_base_url == "http://base:8000/v1"
    # small model defaults to the resolved model.
    assert config.llm_small_model == "memory-model"
    # Embedding inherits the deployment settings.
    assert config.embedding_base_url == "http://embed:8002/v1"
    assert config.embedding_api_key == "embed-key"
    assert config.embedding_model == "embed-model"
    assert config.embedding_dim == 4096


def test_config_neo4j_falls_back_to_deployment_env(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://fallback:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    monkeypatch.delenv("NEO4J_DATABASE", raising=False)
    config = ChatMemoryConfig.from_args(
        SimpleNamespace(
            chat_memory_enabled=True,
            memory_neo4j_username="override",
            memory_neo4j_deployment_id="cluster-a",
        )
    )
    assert config.neo4j_uri == "bolt://fallback:7687"
    assert config.neo4j_username == "override"
    assert config.neo4j_password == "secret"
    assert config.neo4j_database == "neo4j"
    assert config.neo4j_deployment_id == "cluster-a"


def test_config_sanitizes_extra_body_mode_and_clamps():
    config = ChatMemoryConfig.from_args(
        SimpleNamespace(
            chat_memory_enabled=True,
            memory_openai_llm_extra_body="not-json",
            memory_structured_output_mode="yaml",
            memory_search_limit=999,
            memory_ingest_concurrency=0,
            memory_ingest_max_chars=1,
        )
    )
    assert config.llm_extra_body is None
    assert config.structured_output_mode == "json_schema"
    assert config.search_limit == MEMORY_SEARCH_MAX_LIMIT
    assert config.ingest_concurrency == 1
    assert config.ingest_max_chars == 200

    parsed = ChatMemoryConfig.from_args(
        SimpleNamespace(
            chat_memory_enabled=True,
            memory_openai_llm_extra_body=(
                '{"chat_template_kwargs": {"enable_thinking": false}}'
            ),
            memory_structured_output_mode="json_object",
        )
    )
    assert parsed.llm_extra_body == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert parsed.structured_output_mode == "json_object"


def test_config_separates_read_ingest_from_maintenance_and_raw_storage():
    config = ChatMemoryConfig.from_args(
        SimpleNamespace(chat_memory_enabled=False)
    )

    assert config.enabled is False
    assert config.read_ingest_enabled is False
    assert config.maintenance_enabled is True
    assert config.worker_poll_interval_seconds == 1.0
    assert config.worker_recovery_interval_seconds == 30.0
    assert config.worker_side_effect_timeout_seconds == 900.0
    assert config.worker_shutdown_timeout_seconds == 10.0
    assert config.rebuild_max_messages == 10_000
    assert config.rebuild_max_bytes == 64 * 1024 * 1024
    assert config.store_raw_episode_content is False


def test_config_clamps_durable_worker_and_rebuild_limits():
    config = ChatMemoryConfig.from_args(
        SimpleNamespace(
            chat_memory_enabled="false",
            chat_memory_maintenance_enabled="false",
            memory_worker_poll_seconds=0,
            memory_worker_recovery_interval_seconds=99_999,
            memory_worker_side_effect_timeout_seconds=-10,
            memory_worker_shutdown_timeout_seconds=99_999,
            memory_rebuild_max_messages=0,
            memory_rebuild_max_bytes=10**20,
            memory_store_raw_episode_content="true",
        )
    )

    assert config.enabled is False
    assert config.maintenance_enabled is False
    assert config.worker_poll_interval_seconds == 0.05
    assert config.worker_recovery_interval_seconds == 3600.0
    assert config.worker_side_effect_timeout_seconds == 1.0
    assert config.worker_shutdown_timeout_seconds == 300.0
    assert config.rebuild_max_messages == 1
    assert config.rebuild_max_bytes == 4 * 1024 * 1024 * 1024
    assert config.store_raw_episode_content is True


def test_extraction_fingerprint_is_canonical_versioned_and_excludes_tuning(
    monkeypatch,
):
    config = ChatMemoryConfig(
        enabled=True,
        maintenance_enabled=True,
        llm_base_url="https://llm/v1",
        llm_api_key="secret-a",
        llm_model="extract-model",
        llm_small_model="small-model",
        llm_temperature=0.2,
        llm_max_tokens=4096,
        llm_extra_body={"z": 1, "a": {"b": False}},
        structured_output_mode="json_schema",
        embedding_base_url="https://embed/v1",
        embedding_api_key="embed-secret-a",
        embedding_model="embed-model",
        embedding_dim=1536,
        ingest_max_chars=1234,
        store_raw_episode_content=False,
    )
    fingerprint = config.extraction_fingerprint()
    assert fingerprint.startswith("chat-memory-extraction:v1:sha256:")
    assert len(fingerprint.rsplit(":", 1)[-1]) == 64
    assert config.runtime_fingerprint() == fingerprint

    tuning_only = replace(
        config,
        enabled=False,
        maintenance_enabled=False,
        llm_api_key="secret-b",
        embedding_api_key="embed-secret-b",
        worker_poll_interval_seconds=60,
        worker_recovery_interval_seconds=3600,
        worker_side_effect_timeout_seconds=1,
        rebuild_max_messages=1,
        rebuild_max_bytes=1,
        ingest_concurrency=64,
        llm_extra_body={"a": {"b": False}, "z": 1},
    )
    assert tuning_only.extraction_fingerprint() == fingerprint

    for changed in (
        replace(config, llm_model="other"),
        replace(config, llm_base_url="https://other-llm/v1"),
        replace(config, llm_temperature=0.3),
        replace(config, structured_output_mode="json_object"),
        replace(config, llm_extra_body={"different": True}),
        replace(config, embedding_model="other-embed"),
        replace(config, embedding_base_url="https://other-embed/v1"),
        replace(config, embedding_dim=3072),
        replace(config, ingest_max_chars=1235),
        replace(config, store_raw_episode_content=True),
    ):
        assert changed.extraction_fingerprint() != fingerprint

    import lightrag.api.chat_memory_service as memory_module

    monkeypatch.setattr(memory_module, "GRAPHITI_PINNED_VERSION", "0.29.3")
    assert config.extraction_fingerprint() != fingerprint
    monkeypatch.setattr(memory_module, "GRAPHITI_PINNED_VERSION", "0.29.2")
    monkeypatch.setattr(memory_module, "_CHAT_MEMORY_ADMISSION_POLICY_VERSION", 2)
    assert config.extraction_fingerprint() != fingerprint
    monkeypatch.setattr(memory_module, "_CHAT_MEMORY_ADMISSION_POLICY_VERSION", 1)
    monkeypatch.setattr(memory_module, "CHAT_MEMORY_SNAPSHOT_DIGEST_VERSION", 2)
    assert config.extraction_fingerprint() != fingerprint


def test_graph_store_fingerprint_is_canonical_and_credential_free():
    config = ChatMemoryConfig(
        neo4j_uri="bolt://alice:secret@NEO4J.EXAMPLE:7687/",
        neo4j_username="alice",
        neo4j_password="secret",
        neo4j_database="memory",
    )
    fingerprint = config.graph_store_fingerprint()
    assert fingerprint.startswith("chat-memory-graph-store:v1:sha256:")
    assert len(fingerprint.rsplit(":", 1)[-1]) == 64

    assert replace(
        config,
        neo4j_uri="bolt://bob:other@neo4j.example:7687",
        neo4j_username="bob",
        neo4j_password="other",
    ).graph_store_fingerprint() == fingerprint
    assert replace(config, neo4j_database="other").graph_store_fingerprint() != fingerprint
    assert replace(
        config, neo4j_uri="bolt://other.example:7687"
    ).graph_store_fingerprint() != fingerprint

    pinned = replace(config, neo4j_deployment_id="cluster-a")
    assert replace(
        pinned, neo4j_uri="neo4j+s://renamed.internal:7687"
    ).graph_store_fingerprint() == pinned.graph_store_fingerprint()
    assert replace(
        pinned, neo4j_deployment_id="cluster-b"
    ).graph_store_fingerprint() != pinned.graph_store_fingerprint()


async def test_default_factory_passes_policy_and_closes_owned_openai_clients(
    monkeypatch,
):
    captured: dict[str, object] = {}
    clients = []

    def module(name: str, **attrs):
        value = ModuleType(name)
        value.__dict__.update(attrs)
        if "." not in name or name.rsplit(".", 1)[-1] not in {
            "client",
            "neo4j_driver",
            "openai",
            "config",
            "openai_generic_client",
        }:
            value.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, name, value)
        return value

    class FakeCrossEncoderClient:
        pass

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeGraphiti:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.close_calls = 0

        async def close(self):
            self.close_calls += 1

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.close_calls = 0
            clients.append(self)

        async def close(self):
            self.close_calls += 1

    class FakeEmbedder:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.client = FakeAsyncOpenAI(kind="embedder")

    module("graphiti_core", Graphiti=FakeGraphiti)
    module("graphiti_core.cross_encoder")
    module(
        "graphiti_core.cross_encoder.client",
        CrossEncoderClient=FakeCrossEncoderClient,
    )
    module("graphiti_core.driver")
    module("graphiti_core.driver.neo4j_driver", Neo4jDriver=lambda **kwargs: kwargs)
    module("graphiti_core.embedder")
    module(
        "graphiti_core.embedder.openai",
        OpenAIEmbedder=FakeEmbedder,
        OpenAIEmbedderConfig=FakeConfig,
    )
    module("graphiti_core.llm_client")
    module("graphiti_core.llm_client.config", LLMConfig=FakeConfig)
    module(
        "graphiti_core.llm_client.openai_generic_client",
        OpenAIGenericClient=lambda **kwargs: kwargs,
    )
    module("openai", AsyncOpenAI=FakeAsyncOpenAI)

    first = _default_graphiti_factory(
        ChatMemoryConfig(
            llm_base_url="https://llm/v1",
            llm_model="model",
            embedding_base_url="https://embed/v1",
            neo4j_uri="bolt://neo4j:7687",
            store_raw_episode_content=False,
        )
    )
    assert captured["store_raw_episode_content"] is False
    first_owned = list(first._lightrag_owned_clients)
    assert len(first_owned) == 2
    # Deliberately duplicate one client to verify identity-based close dedupe.
    first._lightrag_owned_clients = (*first._lightrag_owned_clients, first_owned[0])
    service = ChatMemoryService(ChatMemoryConfig(enabled=True))
    await service._close_backend_instance(first)
    assert first.close_calls == 1
    assert [client.close_calls for client in first_owned] == [1, 1]

    second = _default_graphiti_factory(
        ChatMemoryConfig(
            llm_base_url="https://llm/v1",
            llm_model="model",
            embedding_base_url="https://embed/v1",
            neo4j_uri="bolt://neo4j:7687",
            store_raw_episode_content=True,
        )
    )
    assert captured["store_raw_episode_content"] is True
    await service._close_backend_instance(second)


def test_server_mode_disables_deprecated_schedule_paths():
    service, fake, _clear = _service()
    service._legacy_scheduling_enabled = False

    with pytest.warns(DeprecationWarning):
        assert (
            service.schedule_ingest(
                user_id="usr_a",
                project_id="proj_b",
                session_id="sess_c",
                messages=[_message("user", "q", 1)],
            )
            is None
        )
    with pytest.warns(DeprecationWarning):
        assert service.schedule_purge("usr_a", ["proj_b"]) is None
    assert fake.episodes == []


async def test_durable_admin_purge_and_stats_do_not_touch_graphiti():
    store = FakeDurableAdminStore()
    config = ChatMemoryConfig(
        enabled=False,
        neo4j_uri="bolt://neo4j:7687",
        neo4j_database="memory",
    )
    nudges: list[str] = []
    audit = FakeAuditService()
    service, fake, clear = _service(
        config=config,
        metadata_store=store,
        audit=audit,
    )
    service.set_post_commit_nudge_callback(lambda: nudges.append("nudge"))

    result = await service.enqueue_purge_projects(
        "usr-target",
        ["proj-a", "proj-noop", "proj-a"],
        actor_user_id="usr-admin",
        actor_tenant_id="tenant-admin",
    )

    assert result == {"queued": 1, "noop": 1}
    assert [call["project_id"] for call in store.purge_calls] == [
        "proj-a",
        "proj-noop",
    ]
    for call in store.purge_calls:
        assert call["config_fingerprint"] == config.extraction_fingerprint()
        assert call["graph_store_fingerprint"] == config.graph_store_fingerprint()
        assert call["actor_user_id"] == "usr-admin"
        assert call["actor_tenant_id"] == "tenant-admin"
    assert nudges == ["nudge"]
    assert fake.build_calls == 0
    assert clear.calls == []
    queued_audits = audit.of_type("chat_memory_purge_queued")
    assert [item[1]["actor_user_id"] for item in queued_audits] == [
        "usr-admin",
        "usr-admin",
    ]
    assert [item[1]["actor_tenant_id"] for item in queued_audits] == [
        "tenant-admin",
        "tenant-admin",
    ]
    assert [item[1]["target_id"] for item in queued_audits] == [
        "proj-a",
        "proj-noop",
    ]
    assert all(item[1]["target_type"] == "chat_project" for item in queued_audits)

    assert await service.outbox_stats() == {
        "pending": 2,
        "running": 1,
        "retry_wait": 3,
        "dead_letter": 4,
        "oldest_available_at": "2026-07-16T00:00:00+00:00",
        "oldest_lag_seconds": 5.5,
    }
    assert fake.build_calls == 0


async def test_retry_dead_letter_purge_uses_durable_deleted_target_and_audits_actor():
    config = ChatMemoryConfig(
        enabled=False,
        neo4j_uri="bolt://neo4j:7687",
        neo4j_database="memory",
        neo4j_deployment_id="memory-primary",
    )
    order: list[str] = []
    durable_event = _retry_event(config)
    store = FakeRetryStore(durable_event, order=order)
    audit = FakeAuditService(order)
    service, fake, clear = _service(
        config=config,
        metadata_store=store,
        audit=audit,
    )
    service.set_post_commit_nudge_callback(lambda: order.append("nudge"))

    retried = await service.retry_purge_event(
        durable_event.event_id,
        actor_user_id="usr-real-admin",
        actor_tenant_id="tenant-admin",
    )

    assert retried.event_id == durable_event.event_id
    assert retried.status == "retry_wait"
    assert retried.user_id == "usr-deleted"
    assert retried.project_id == "proj-deleted"
    assert store.requeue_calls == [
        {
            "event_id": durable_event.event_id,
            "runtime_fingerprint": config.extraction_fingerprint(),
            "runtime_graph_store_fingerprint": config.graph_store_fingerprint(),
            "retry_delay_seconds": 0,
        }
    ]
    assert order == ["get", "requeue_commit", "audit", "nudge"]
    assert store.group_create_calls == []
    assert fake.build_calls == 0
    assert clear.calls == []

    retried_audits = audit.of_type("chat_memory_purge_retry_queued")
    assert len(retried_audits) == 1
    payload = retried_audits[0][1]
    assert payload["actor_user_id"] == "usr-real-admin"
    assert payload["actor_tenant_id"] == "tenant-admin"
    assert payload["target_type"] == "chat_memory_event"
    assert payload["target_id"] == durable_event.event_id
    assert payload["metadata"] == {
        "event_id": durable_event.event_id,
        "user_id": "usr-deleted",
        "project_id": "proj-deleted",
        "event_type": "purge",
        "status": "retry_wait",
        "previous_status": "dead_letter",
        "requeued": True,
    }
    assert "private source content" not in str(payload)


@pytest.mark.parametrize("status", ["pending", "retry_wait"])
async def test_retry_pending_purge_is_idempotent(status):
    config = ChatMemoryConfig(neo4j_deployment_id="memory-primary")
    event = _retry_event(config, status=status)
    store = FakeRetryStore(event)
    audit = FakeAuditService()
    nudges: list[str] = []
    service, _fake, _clear = _service(
        config=config,
        metadata_store=store,
        audit=audit,
    )
    service.set_post_commit_nudge_callback(lambda: nudges.append("nudge"))

    current = await service.retry_purge_event(
        event.event_id,
        actor_user_id="usr-admin",
    )

    assert current is event
    assert current.status == status
    assert store.requeue_calls == []
    assert nudges == ["nudge"]
    assert audit.of_type("chat_memory_purge_retry_queued")[0][1]["metadata"][
        "requeued"
    ] is False


async def test_retry_missing_event_does_not_create_forged_group():
    store = FakeRetryStore()
    audit = FakeAuditService()
    nudges: list[str] = []
    service, _fake, _clear = _service(metadata_store=store, audit=audit)
    service.set_post_commit_nudge_callback(lambda: nudges.append("nudge"))

    with pytest.raises(ChatMemoryEventNotFoundError):
        await service.retry_purge_event(
            "evt-forged",
            actor_user_id="usr-admin",
        )

    assert store.get_calls == ["evt-forged"]
    assert store.requeue_calls == []
    assert store.group_create_calls == []
    assert audit.events == []
    assert nudges == []


async def test_retry_rejects_non_purge_event():
    config = ChatMemoryConfig(neo4j_deployment_id="memory-primary")
    event = _retry_event(config, event_type="rebuild")
    store = FakeRetryStore(event)
    service, _fake, _clear = _service(config=config, metadata_store=store)

    with pytest.raises(ChatMemoryRetryConflictError) as exc_info:
        await service.retry_purge_event(event.event_id, actor_user_id="usr-admin")

    assert exc_info.value.error_code == "chat_memory_retry_purge_only"
    assert event.status == "dead_letter"
    assert store.requeue_calls == []


@pytest.mark.parametrize("status", ["running", "succeeded", "superseded"])
async def test_retry_rejects_non_retryable_purge_status(status):
    config = ChatMemoryConfig(neo4j_deployment_id="memory-primary")
    event = _retry_event(config, status=status)
    store = FakeRetryStore(event)
    service, _fake, _clear = _service(config=config, metadata_store=store)

    with pytest.raises(ChatMemoryRetryConflictError) as exc_info:
        await service.retry_purge_event(event.event_id, actor_user_id="usr-admin")

    assert exc_info.value.error_code == "chat_memory_event_not_retryable"
    assert status in exc_info.value.message
    assert event.status == status
    assert store.requeue_calls == []


async def test_retry_wrong_graph_requires_original_backend_without_state_change():
    config = ChatMemoryConfig(neo4j_deployment_id="memory-current")
    event = _retry_event(
        config,
        graph_store_fingerprint=ChatMemoryConfig(
            neo4j_deployment_id="memory-original"
        ).graph_store_fingerprint(),
    )
    store = FakeRetryStore(event)
    service, _fake, _clear = _service(config=config, metadata_store=store)

    with pytest.raises(ChatMemoryRetryConflictError) as exc_info:
        await service.retry_purge_event(event.event_id, actor_user_id="usr-admin")

    assert exc_info.value.error_code == "chat_memory_old_graph_store_required"
    assert "MEMORY_NEO4J_DEPLOYMENT_ID" in exc_info.value.message
    assert "backend" in exc_info.value.message
    assert event.status == "dead_letter"
    assert store.requeue_calls == []


async def test_retry_nudge_failure_does_not_change_committed_result():
    config = ChatMemoryConfig(neo4j_deployment_id="memory-primary")
    order: list[str] = []
    event = _retry_event(config)
    store = FakeRetryStore(event, order=order)
    service, _fake, _clear = _service(config=config, metadata_store=store)

    def broken_nudge():
        order.append("nudge")
        raise RuntimeError("worker offline")

    service.set_post_commit_nudge_callback(broken_nudge)

    retried = await service.retry_purge_event(
        event.event_id,
        actor_user_id="usr-admin",
    )

    assert retried.status == "retry_wait"
    assert store.events[event.event_id].status == "retry_wait"
    assert order == ["get", "requeue_commit", "nudge"]


# ------------------------------------------------------------------- group id


def test_group_id_composition_and_validation():
    assert (
        ChatMemoryService.build_group_id("usr_ab12", "proj_cd34")
        == "usr_ab12--proj_cd34"
    )
    for user_id, project_id in [
        ("", "proj_x"),
        ("usr_x", ""),
        ("usr x", "proj_x"),
        ("usr_x", "proj:x"),
        ("usr_x", "proj/../x"),
    ]:
        with pytest.raises(ValueError):
            ChatMemoryService.build_group_id(user_id, project_id)


# --------------------------------------------------------------------- ingest


async def test_ingest_formats_episode_and_audits():
    audit = FakeAuditService()
    service, fake, _clear = _service(audit=audit)
    task = service.schedule_ingest(
        user_id="usr_a",
        project_id="proj_b",
        session_id="sess_c",
        messages=[
            _message("user", "低温屈挠性怎么提升？", 1),
            _message("assistant", "建议 NR/BR 并用… [A1]", 2),
        ],
    )
    assert task is not None
    await service.wait_for_background_tasks()

    assert len(fake.episodes) == 1
    episode = fake.episodes[0]
    assert episode["group_id"] == "usr_a--proj_b"
    assert episode["name"] == "sess_c:1-2"
    assert episode["episode_body"] == (
        "user: 低温屈挠性怎么提升？\nassistant: 建议 NR/BR 并用… [A1]"
    )
    assert episode["source_description"] == "enterprise chat"
    reference_time = episode["reference_time"]
    assert isinstance(reference_time, datetime) and reference_time.tzinfo is not None
    assert reference_time == datetime(2026, 7, 10, 8, 0, 5, tzinfo=timezone.utc)

    ingested = audit.of_type("chat_memory_ingested")
    assert len(ingested) == 1
    metadata = ingested[0][1]["metadata"]
    assert metadata["message_count"] == 2
    assert metadata["project_id"] == "proj_b"
    assert metadata["episode_uuid"] == "ep-1"
    # Message bodies never reach the audit trail.
    assert "低温" not in str(ingested[0][1])


async def test_ingest_truncates_long_content_and_skips_empty():
    config = ChatMemoryConfig(enabled=True, ingest_max_chars=200)
    service, fake, _clear = _service(config=config)
    long_content = "x" * 500
    service.schedule_ingest(
        user_id="usr_a",
        project_id="proj_b",
        session_id="sess_c",
        messages=[
            _message("user", long_content, 1),
            _message("assistant", "   ", 2),  # blank -> filtered out
        ],
    )
    await service.wait_for_background_tasks()
    assert len(fake.episodes) == 1
    body = fake.episodes[0]["episode_body"]
    assert body.startswith("user: " + "x" * 200)
    assert body.endswith("…[truncated]")
    assert "assistant:" not in body

    # Nothing ingestible -> no task at all.
    assert (
        service.schedule_ingest(
            user_id="usr_a",
            project_id="proj_b",
            session_id="sess_c",
            messages=[_message("assistant", "", 3)],
        )
        is None
    )


async def test_ingest_serializes_same_group_and_swallows_errors():
    fake = FakeGraphiti()
    fake.add_episode_delay = 0.02
    service, _fake, _clear = _service(
        config=ChatMemoryConfig(enabled=True, ingest_concurrency=4), fake=fake
    )
    for seq in (1, 2):
        service.schedule_ingest(
            user_id="usr_a",
            project_id="proj_b",
            session_id="sess_c",
            messages=[_message("user", f"q{seq}", seq)],
        )
    await service.wait_for_background_tasks()
    # Same group must never interleave: start/end pairs are strictly ordered.
    assert [kind for kind, _ in fake.markers] == ["start", "end", "start", "end"]

    # An add_episode failure is logged, not raised, and does not audit.
    audit = FakeAuditService()
    failing = FakeGraphiti()
    failing.add_episode_error = RuntimeError("llm exploded")
    service2, _f, _c = _service(fake=failing, audit=audit)
    task = service2.schedule_ingest(
        user_id="usr_a",
        project_id="proj_b",
        session_id="sess_c",
        messages=[_message("user", "boom", 1)],
    )
    await task
    assert audit.of_type("chat_memory_ingested") == []


async def test_ingest_cross_group_concurrency_capped():
    fake = FakeGraphiti()
    fake.add_episode_delay = 0.02
    service, _fake, _clear = _service(
        config=ChatMemoryConfig(enabled=True, ingest_concurrency=1), fake=fake
    )
    for project in ("proj_1", "proj_2"):
        service.schedule_ingest(
            user_id="usr_a",
            project_id=project,
            session_id="sess_c",
            messages=[_message("user", "q", 1)],
        )
    await service.wait_for_background_tasks()
    # With a global concurrency of 1, distinct groups still run one at a time.
    assert [kind for kind, _ in fake.markers] == ["start", "end", "start", "end"]


async def test_ingest_skipped_when_backend_unavailable():
    def failing_factory(_config):
        raise ChatMemoryUnavailableError("neo4j down")

    service = ChatMemoryService(
        ChatMemoryConfig(enabled=True), graphiti_factory=failing_factory
    )
    assert await service.initialize() is False
    task = service.schedule_ingest(
        user_id="usr_a",
        project_id="proj_b",
        session_id="sess_c",
        messages=[_message("user", "q", 1)],
    )
    await task  # must not raise
    assert service.available is False


# --------------------------------------------------------------------- search


async def test_legacy_search_forces_logical_group_and_current_fact_filter():
    audit = FakeAuditService()
    service, fake, _clear = _service(audit=audit)
    fake.search_results = [
        SimpleNamespace(
            uuid="edge-1",
            name="USES",
            fact="项目采用 NR/BR 并用",
            valid_at=datetime(2026, 7, 10, 8, 0, 5, tzinfo=timezone.utc),
            invalid_at=None,
            created_at=datetime(2026, 7, 10, 8, 0, 41, tzinfo=timezone.utc),
            expired_at=None,
        )
    ]
    facts = await service.search(
        user_id="usr_a", project_id="proj_b", query="低温性能结论？", limit=5
    )
    assert len(fake.search_calls) == 1
    search_call = fake.search_calls[0]
    assert search_call["query"] == "低温性能结论？"
    assert search_call["group_ids"] == ["usr_a--proj_b"]
    assert search_call["num_results"] == 5
    _assert_current_fact_filter(search_call)
    assert facts == [
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
    searched = audit.of_type("chat_memory_searched")
    assert len(searched) == 1
    metadata = searched[0][1]["metadata"]
    assert metadata["fact_count"] == 1
    assert len(metadata["query_hash"]) == 64
    # The raw query never reaches the audit trail.
    assert "低温性能" not in str(searched[0][1])


async def test_store_backed_search_uses_active_physical_group_and_fences_read():
    config = ChatMemoryConfig(enabled=True, neo4j_uri="bolt://neo4j:7687")
    token = _read_token(config, graph_group_id="physical-generation-7")
    store = FakeReadTokenStore([token, token])
    service, fake, _clear = _service(config=config, metadata_store=store)

    await service.search(user_id="usr_a", project_id="proj_b", query="q")

    assert store.calls == [("usr_a", "proj_b"), ("usr_a", "proj_b")]
    assert fake.search_calls[0]["group_ids"] == ["physical-generation-7"]
    _assert_current_fact_filter(fake.search_calls[0])


async def test_search_discards_raced_generation_and_retries_once():
    config = ChatMemoryConfig(enabled=True, neo4j_uri="bolt://neo4j:7687")
    first = _read_token(config, generation=1, graph_group_id="physical-g1")
    second = _read_token(
        config,
        generation=2,
        state_version=2,
        graph_group_id="physical-g2",
    )
    store = FakeReadTokenStore([first, second, second, second])
    service, fake, _clear = _service(config=config, metadata_store=store)

    async def raced_search(**kwargs):
        fake.search_calls.append(kwargs)
        group_id = kwargs["group_ids"][0]
        return [
            SimpleNamespace(
                uuid=group_id,
                name="N",
                fact=f"fact from {group_id}",
                valid_at=None,
                invalid_at=None,
                created_at=None,
                expired_at=None,
            )
        ]

    fake.search = raced_search
    facts = await service.search(user_id="usr_a", project_id="proj_b", query="q")

    assert [call["group_ids"] for call in fake.search_calls] == [
        ["physical-g1"],
        ["physical-g2"],
    ]
    assert facts[0]["fact"] == "fact from physical-g2"


async def test_search_raises_when_active_generation_changes_twice():
    config = ChatMemoryConfig(enabled=True, neo4j_uri="bolt://neo4j:7687")
    tokens = [
        _read_token(config, generation=1, state_version=1, graph_group_id="g1"),
        _read_token(config, generation=2, state_version=2, graph_group_id="g2"),
        _read_token(config, generation=2, state_version=2, graph_group_id="g2"),
        _read_token(config, generation=3, state_version=3, graph_group_id="g3"),
    ]
    store = FakeReadTokenStore(tokens)
    service, fake, _clear = _service(config=config, metadata_store=store)

    with pytest.raises(ChatMemoryUnavailableError, match="changed during search"):
        await service.search(user_id="usr_a", project_id="proj_b", query="q")
    assert len(fake.search_calls) == 2


@pytest.mark.parametrize(
    ("token_kwargs"),
    [
        {"state": "rebuilding"},
        {"state": "deleting"},
        {"state": "deleted"},
        {"state": "failed"},
        {"generation": None, "graph_group_id": None, "generation_state": None},
        {"generation_state": "building"},
    ],
)
async def test_inactive_read_token_never_initializes_or_searches_backend(token_kwargs):
    config = ChatMemoryConfig(enabled=True, neo4j_uri="bolt://neo4j:7687")
    store = FakeReadTokenStore([_read_token(config, **token_kwargs)])
    service, fake, _clear = _service(config=config, metadata_store=store)

    assert await service.search(user_id="usr_a", project_id="proj_b", query="q") == []
    assert fake.build_calls == 0
    assert fake.search_calls == []
    assert service.available is False


@pytest.mark.parametrize("mismatch", ["extraction", "graph"])
async def test_wrong_active_fingerprint_fails_open_without_graphiti(mismatch):
    config = ChatMemoryConfig(enabled=True, neo4j_uri="bolt://neo4j:7687")
    kwargs = {
        "extraction_fingerprint": "chat-memory-extraction:v1:sha256:" + "0" * 64
    }
    if mismatch == "graph":
        kwargs = {
            "graph_store_fingerprint": (
                "chat-memory-graph-store:v1:sha256:" + "0" * 64
            )
        }
    store = FakeReadTokenStore([_read_token(config, **kwargs)])
    service, fake, _clear = _service(config=config, metadata_store=store)

    assert await service.search(user_id="usr_a", project_id="proj_b", query="q") == []
    assert fake.search_calls == []
    block, info = await service.build_memory_block(
        user_id="usr_a", project_id="proj_b", query="q"
    )
    assert block is None
    assert info == {"enabled": True, "project_id": "proj_b", "fact_count": 0}


async def test_search_limit_defaults_and_clamps():
    service, fake, _clear = _service(
        config=ChatMemoryConfig(enabled=True, search_limit=7)
    )
    await service.search(user_id="usr_a", project_id="proj_b", query="q")
    assert fake.search_calls[-1]["num_results"] == 7
    await service.search(user_id="usr_a", project_id="proj_b", query="q", limit=999)
    assert fake.search_calls[-1]["num_results"] == MEMORY_SEARCH_MAX_LIMIT


async def test_search_raises_unavailable_and_recovers_after_retry():
    attempts = {"count": 0}
    fake = FakeGraphiti()

    def flaky_factory(_config):
        attempts["count"] += 1
        if attempts["count"] <= 2:
            raise RuntimeError("neo4j still starting")
        return fake

    service = ChatMemoryService(
        ChatMemoryConfig(enabled=True), graphiti_factory=flaky_factory
    )
    # Startup failure is soft (attempt 1)…
    assert await service.initialize() is False
    # …the next use surfaces a typed error (attempt 2, wrapped)…
    with pytest.raises(ChatMemoryUnavailableError):
        await service.search(user_id="usr_a", project_id="proj_b", query="q")
    # …and the service recovers once the backend comes back (attempt 3).
    facts = await service.search(user_id="usr_a", project_id="proj_b", query="q")
    assert facts == []
    assert service.available is True
    assert fake.build_calls == 1


async def test_search_runtime_failure_resets_and_closes_backend():
    first = FakeGraphiti()
    second = FakeGraphiti()

    async def failing_search(**_kwargs):
        raise RuntimeError("neo4j connection lost")

    first.search = failing_search
    instances = iter((first, second))
    service = ChatMemoryService(
        ChatMemoryConfig(enabled=True, worker_side_effect_timeout_seconds=1),
        graphiti_factory=lambda _config: next(instances),
    )

    with pytest.raises(ChatMemoryUnavailableError):
        await service.search(user_id="usr_a", project_id="proj_b", query="q")
    assert first.closed is True
    assert service.available is False

    assert (
        await service.search(user_id="usr_a", project_id="proj_b", query="q")
        == []
    )
    assert second.build_calls == 1
    assert service.available is True


async def test_search_cancellation_does_not_invalidate_backend():
    fake = FakeGraphiti()
    started = asyncio.Event()

    async def blocking_search(**kwargs):
        fake.search_calls.append(kwargs)
        started.set()
        await asyncio.Event().wait()

    fake.search = blocking_search
    service, _fake, _clear = _service(fake=fake)
    task = asyncio.create_task(
        service.search(user_id="usr_a", project_id="proj_b", query="q")
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert service.available is True
    assert fake.closed is False
    await service.finalize()
    assert fake.closed is True


async def test_failed_group_retires_shared_slot_without_closing_active_peer():
    first = FakeGraphiti()
    second = FakeGraphiti()
    peer_started = asyncio.Event()
    release_peer = asyncio.Event()

    async def shared_search(**kwargs):
        first.search_calls.append(kwargs)
        if kwargs["query"] == "peer":
            peer_started.set()
            await release_peer.wait()
            return []
        raise RuntimeError("group A driver failure")

    first.search = shared_search
    instances = iter((first, second))
    service = ChatMemoryService(
        ChatMemoryConfig(enabled=True, worker_side_effect_timeout_seconds=1),
        graphiti_factory=lambda _config: next(instances),
    )
    peer_task = asyncio.create_task(
        service.search(user_id="usr_b", project_id="proj_b", query="peer")
    )
    await peer_started.wait()

    with pytest.raises(ChatMemoryUnavailableError):
        await service.search(user_id="usr_a", project_id="proj_a", query="fail")
    assert service.available is False
    assert first.closed is False

    release_peer.set()
    assert await peer_task == []
    assert first.closed is True

    assert await service.search(
        user_id="usr_c", project_id="proj_c", query="new"
    ) == []
    assert service.available is True
    assert second.build_calls == 1


async def test_initialization_timeout_closes_unpublished_candidate():
    candidate = FakeGraphiti()

    async def slow_build():
        await asyncio.sleep(1)

    candidate.build_indices_and_constraints = slow_build
    service = ChatMemoryService(
        ChatMemoryConfig(enabled=True, worker_side_effect_timeout_seconds=0.01),
        graphiti_factory=lambda _config: candidate,
    )

    with pytest.raises(ChatMemoryUnavailableError, match="backend unavailable"):
        await service.ensure_backend()
    assert candidate.closed is True
    assert service.available is False


async def test_initialization_cancellation_closes_unpublished_candidate():
    candidate = FakeGraphiti()
    build_started = asyncio.Event()

    async def blocked_build():
        build_started.set()
        await asyncio.Event().wait()

    candidate.build_indices_and_constraints = blocked_build
    service = ChatMemoryService(
        ChatMemoryConfig(enabled=True, worker_side_effect_timeout_seconds=1),
        graphiti_factory=lambda _config: candidate,
    )
    task = asyncio.create_task(service.ensure_backend())
    await build_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert candidate.closed is True
    assert service.available is False


# ---------------------------------------------------------------------- purge


async def test_purge_passes_explicit_group_lists():
    audit = FakeAuditService()
    service, fake, clear = _service(audit=audit)
    task = service.schedule_purge("usr_a", ["proj_1", "proj_2"])
    assert task is not None
    await service.wait_for_background_tasks()
    assert clear.calls == [(fake, ["usr_a--proj_1", "usr_a--proj_2"])]
    purged = audit.of_type("chat_memory_purged")
    assert len(purged) == 1
    assert purged[0][1]["metadata"]["project_count"] == 2

    # Empty input never reaches clear_data.
    assert service.schedule_purge("usr_a", []) is None
    assert await service.purge_projects("usr_a", []) == 0
    assert len(clear.calls) == 1


async def test_finalize_blocks_new_work_and_closes_backend():
    service, fake, _clear = _service()
    await service.search(user_id="usr_a", project_id="proj_b", query="warm up")
    await service.finalize()
    assert fake.closed is True
    assert (
        service.schedule_ingest(
            user_id="usr_a",
            project_id="proj_b",
            session_id="sess_c",
            messages=[_message("user", "q", 1)],
        )
        is None
    )
    assert service.schedule_purge("usr_a", ["proj_b"]) is None
    with pytest.raises(ChatMemoryUnavailableError):
        await service.search(user_id="usr_a", project_id="proj_b", query="q")


# ------------------------------------------------------------- openai wrapper


async def test_extra_body_wrapper_merges_and_delegates():
    calls: list[dict] = []

    class FakeCompletions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return "ok"

    inner = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions()),
        api_key="k",
    )
    wrapper = _ExtraBodyAsyncOpenAI(
        inner, {"chat_template_kwargs": {"enable_thinking": False}}
    )
    result = await wrapper.chat.completions.create(
        model="qwen", messages=[], extra_body={"top_k": 5}
    )
    assert result == "ok"
    assert calls == [
        {
            "model": "qwen",
            "messages": [],
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
                "top_k": 5,
            },
        }
    ]
    # Unwrapped attributes delegate to the inner client.
    assert wrapper.api_key == "k"


# ------------------------------------------------------------- rerank adapter


async def test_rerank_cross_encoder_sorts_and_falls_back():
    async def rerank_fn(*, query, documents, top_n=None):
        # Return out-of-order relevance to prove the adapter re-sorts.
        return [
            {"index": 0, "relevance_score": 0.1},
            {"index": 1, "relevance_score": 0.9},
        ]

    encoder = _RerankFnCrossEncoder(rerank_fn)
    ranked = await encoder.rank("q", ["low", "high"])
    assert [passage for passage, _ in ranked] == ["high", "low"]
    assert await encoder.rank("q", []) == []

    async def boom(**_kwargs):
        raise RuntimeError("rerank down")

    fallback = _RerankFnCrossEncoder(boom)
    kept = await fallback.rank("q", ["a", "b"])
    # Failure keeps original order rather than dropping availability.
    assert [passage for passage, _ in kept] == ["a", "b"]


async def test_search_uses_cross_encoder_recipe_when_enabled():
    fake = FakeGraphiti()
    fake.search_results = [
        SimpleNamespace(
            uuid="e1", name="N", fact="f", valid_at=None, invalid_at=None,
            created_at=None, expired_at=None,
        )
    ]

    async def rerank_fn(*, query, documents, top_n=None):
        return [{"index": 0, "relevance_score": 1.0}]

    service = ChatMemoryService(
        ChatMemoryConfig(enabled=True, rerank_enabled=True),
        graphiti_factory=lambda _cfg: fake,
        rerank_fn=rerank_fn,
    )
    facts = await service.search(user_id="usr_a", project_id="proj_b", query="q")
    # The cross-encoder recipe path (search_) was used, not plain search().
    assert len(fake.search_recipe_calls) == 1
    assert fake.search_recipe_calls[0]["group_ids"] == ["usr_a--proj_b"]
    _assert_current_fact_filter(fake.search_recipe_calls[0])
    assert fake.search_calls == []
    assert facts[0]["fact"] == "f"


# ----------------------------------------------------- store-backed durability


def _seed_store_service(tmp_path, fake=None, clear=None, config=None, audit=None):
    """Build a service backed by a real SQLite metadata store + seeded chat rows."""
    fake = fake or FakeGraphiti()
    clear = clear or FakeClearData()
    store = SQLiteMetadataStore(tmp_path / "metadata.sqlite3")

    async def seed():
        await store.initialize()
        now = utc_now_iso()
        await store.upsert_enterprise_user(
            EnterpriseUserRecord(
                id="usr_a",
                username="alice",
                password_hash="{bcrypt}$2b$12$placeholderplaceholderplaceholderplaceholderpl",
                system_role="user",
                status="active",
                tenant_id=None,
                can_create_kb=False,
                can_use_bypass_query=False,
                token_version=1,
                metadata={},
                created_at=now,
                updated_at=now,
            )
        )
        await store.create_chat_project(
            ChatProjectRecord(
                id="proj_b", user_id="usr_a", name="P", created_at=now, updated_at=now
            )
        )
        await store.create_chat_session(
            ChatSessionRecord(
                id="sess_c",
                project_id="proj_b",
                user_id="usr_a",
                name="S",
                created_at=now,
                updated_at=now,
                context_rounds=1,
            )
        )

    service = ChatMemoryService(
        config or ChatMemoryConfig(enabled=True),
        audit_service=audit,
        graphiti_factory=lambda _cfg: fake,
        clear_data_fn=clear,
        metadata_store=store,
    )
    return service, fake, clear, store, seed


async def _append(store, contents):
    from uuid import uuid4

    records = [
        ChatMessageRecord(
            id=f"msg_{uuid4().hex[:12]}",
            session_id="sess_c",
            project_id="proj_b",
            user_id="usr_a",
            role=role,
            content=content,
            metadata={},
            seq=0,
            created_at=utc_now_iso(),
        )
        for role, content in contents
    ]
    return await store.append_chat_messages(records)


async def test_real_store_retry_survives_user_and_project_source_deletion(tmp_path):
    config = ChatMemoryConfig(
        enabled=False,
        neo4j_deployment_id="memory-primary",
    )
    audit = FakeAuditService()
    service, fake, _clear, store, seed = _seed_store_service(
        tmp_path,
        config=config,
        audit=audit,
    )
    await seed()
    await store.append_chat_messages_with_memory(
        [
            ChatMessageRecord(
                id="msg_durable_retry",
                session_id="sess_c",
                project_id="proj_b",
                user_id="usr_a",
                role="user",
                content="private durable source body",
                metadata={},
                seq=0,
                created_at=utc_now_iso(),
            )
        ],
        config_fingerprint=config.extraction_fingerprint(),
        graph_store_fingerprint=config.graph_store_fingerprint(),
    )
    assert await store.delete_enterprise_user_with_memory(
        "usr_a",
        config_fingerprint=config.extraction_fingerprint(),
        graph_store_fingerprint=config.graph_store_fingerprint(),
        actor_user_id="usr-delete-admin",
    )
    assert await store.get_enterprise_user_by_id("usr_a") is None
    assert await store.get_chat_project("usr_a", "proj_b") is None

    claimed = await store.claim_next_chat_memory_event(
        config.extraction_fingerprint(),
        runtime_graph_store_fingerprint=config.graph_store_fingerprint(),
        event_types=["purge"],
    )
    assert claimed is not None and claimed.claim_token is not None
    dead = await store.fail_chat_memory_purge_before_side_effect(
        claimed.event_id,
        claimed.claim_token,
        config.extraction_fingerprint(),
        runtime_graph_store_fingerprint=config.graph_store_fingerprint(),
        error_code="clear_unavailable",
        error_message="private durable source body",
        retry_delay_seconds=None,
        max_attempts=1,
    )
    assert dead.status == "dead_letter"

    retried = await service.retry_purge_event(
        dead.event_id,
        actor_user_id="usr-retry-admin",
        actor_tenant_id="tenant-ops",
    )

    assert retried.event_id == dead.event_id
    assert retried.generation == dead.generation
    assert retried.status == "retry_wait"
    assert retried.user_id == "usr_a"
    assert retried.project_id == "proj_b"
    assert fake.build_calls == 0
    payload = audit.of_type("chat_memory_purge_retry_queued")[0][1]
    assert payload["actor_user_id"] == "usr-retry-admin"
    assert payload["actor_tenant_id"] == "tenant-ops"
    assert "private durable source body" not in str(payload)


async def test_ingest_is_idempotent_via_watermark(tmp_path):
    service, fake, _clear, store, seed = _seed_store_service(tmp_path)
    await seed()
    saved = await _append(store, [("user", "q1"), ("assistant", "a1")])
    await service._ingest(
        user_id="usr_a", project_id="proj_b", session_id="sess_c", messages=saved
    )
    assert len(fake.episodes) == 1
    assert await store.get_chat_memory_watermark("usr_a", "proj_b", "sess_c") == 2

    # Re-ingesting the same records is a no-op (watermark already covers them).
    await service._ingest(
        user_id="usr_a", project_id="proj_b", session_id="sess_c", messages=saved
    )
    assert len(fake.episodes) == 1

    # A new turn advances past the watermark and is ingested.
    saved2 = await _append(store, [("user", "q2")])
    await service._ingest(
        user_id="usr_a", project_id="proj_b", session_id="sess_c", messages=saved2
    )
    assert len(fake.episodes) == 2
    assert await store.get_chat_memory_watermark("usr_a", "proj_b", "sess_c") == 3


async def test_blank_range_advances_watermark_without_episode(tmp_path):
    service, fake, _clear, store, seed = _seed_store_service(tmp_path)
    await seed()
    # Persist a blank assistant turn (the store allows any content); memory
    # skips it but must still advance the watermark to avoid a retry loop.
    saved = await _append(store, [("user", "   ")])
    # Force ingest of the blank range directly.
    await service._ingest(
        user_id="usr_a",
        project_id="proj_b",
        session_id="sess_c",
        messages=saved,
        force=True,
    )
    assert fake.episodes == []
    assert await store.get_chat_memory_watermark("usr_a", "proj_b", "sess_c") == 1


async def test_backlog_scan_reingests_lost_work(tmp_path):
    service, fake, _clear, store, seed = _seed_store_service(tmp_path)
    await seed()
    # Messages exist but were never ingested (fire-and-forget lost to a crash).
    await _append(store, [("user", "q1"), ("assistant", "a1")])
    backlog = await store.list_chat_memory_backlog()
    assert any(item.session_id == "sess_c" for item in backlog)

    ingested = await service.run_backlog_scan()
    assert ingested == 1
    assert len(fake.episodes) == 1
    assert await store.get_chat_memory_watermark("usr_a", "proj_b", "sess_c") == 2
    # Converged: nothing left to scan.
    assert await service.run_backlog_scan() == 0


async def test_forget_message_removes_episode_and_reingests_survivors(tmp_path):
    service, fake, _clear, store, seed = _seed_store_service(tmp_path)
    await seed()
    saved = await _append(
        store, [("user", "q1"), ("assistant", "a1"), ("user", "q2")]
    )
    await service._ingest(
        user_id="usr_a", project_id="proj_b", session_id="sess_c", messages=saved
    )
    assert len(fake.episodes) == 1
    episode_uuid = "ep-1"

    # Delete the middle message (seq 2), then forget it.
    await store.delete_chat_message("usr_a", "proj_b", "sess_c", saved[1].id)
    await service._forget_message(
        user_id="usr_a", project_id="proj_b", session_id="sess_c", seq=2
    )
    # Original episode removed from graphiti; survivors (seq 1,3) re-ingested.
    assert episode_uuid in fake.removed_episodes
    assert len(fake.episodes) == 2
    survivor_body = fake.episodes[1]["episode_body"]
    assert "q1" in survivor_body and "q2" in survivor_body
    assert "a1" not in survivor_body


async def test_forget_session_removes_all_episodes(tmp_path):
    service, fake, _clear, store, seed = _seed_store_service(tmp_path)
    await seed()
    saved = await _append(store, [("user", "q1")])
    await service._ingest(
        user_id="usr_a", project_id="proj_b", session_id="sess_c", messages=saved
    )
    await service._forget_session(
        user_id="usr_a", project_id="proj_b", session_id="sess_c"
    )
    assert fake.removed_episodes == ["ep-1"]
    assert (
        await store.list_chat_memory_episodes_for_session("usr_a", "proj_b", "sess_c")
        == []
    )


async def test_purge_projects_clears_episode_rows(tmp_path):
    service, fake, clear, store, seed = _seed_store_service(tmp_path)
    await seed()
    saved = await _append(store, [("user", "q1")])
    await service._ingest(
        user_id="usr_a", project_id="proj_b", session_id="sess_c", messages=saved
    )
    assert await store.get_chat_memory_watermark("usr_a", "proj_b", "sess_c") == 1
    purged = await service.purge_projects("usr_a", ["proj_b"])
    assert purged == 1
    assert clear.calls == [(fake, ["usr_a--proj_b"])]
    # Episode mapping rows are cleared alongside the graph partition.
    assert await store.get_chat_memory_watermark("usr_a", "proj_b", "sess_c") == 0


async def test_debounced_ingest_buffers_then_flushes(tmp_path):
    service, fake, _clear, store, seed = _seed_store_service(
        tmp_path,
        config=ChatMemoryConfig(
            enabled=True, ingest_mode="debounced", ingest_debounce_seconds=0.05
        ),
    )
    await seed()
    saved1 = await _append(store, [("user", "q1")])
    saved2 = await _append(store, [("assistant", "a1")])
    # Two rapid batches within the debounce window coalesce into one episode.
    service.schedule_ingest(
        user_id="usr_a", project_id="proj_b", session_id="sess_c", messages=saved1
    )
    service.schedule_ingest(
        user_id="usr_a", project_id="proj_b", session_id="sess_c", messages=saved2
    )
    await asyncio.sleep(0.2)
    await service.wait_for_background_tasks()
    assert len(fake.episodes) == 1
    body = fake.episodes[0]["episode_body"]
    assert "q1" in body and "a1" in body


# ------------------------------------------------------------- prompt blocks


def test_format_memory_block_marks_validity():
    block = ChatMemoryService.format_memory_block(
        [
            {"fact": "采用 NR/BR 并用", "valid_at": "2026-07-10T08:00:00+00:00",
             "invalid_at": None},
            {"fact": "旧结论", "valid_at": "2026-07-01T00:00:00+00:00",
             "invalid_at": "2026-07-10T00:00:00+00:00"},
            {"fact": "  ", "valid_at": None, "invalid_at": None},
        ]
    )
    assert "采用 NR/BR 并用（自 2026-07-10 起）" in block
    assert "旧结论（已失效于 2026-07-10" in block
    assert block.startswith("[项目记忆]")
    assert ChatMemoryService.format_memory_block([]) == ""


async def test_build_memory_block_fails_open_when_unavailable():
    def failing_factory(_config):
        raise ChatMemoryUnavailableError("neo4j down")

    service = ChatMemoryService(
        ChatMemoryConfig(enabled=True), graphiti_factory=failing_factory
    )
    block, info = await service.build_memory_block(
        user_id="usr_a", project_id="proj_b", query="q"
    )
    assert block is None
    assert info == {"enabled": False, "reason": "unavailable"}


# ---------------------------------------------------------------- overview


async def test_project_overview_and_global_stats(tmp_path):
    service, fake, _clear, store, seed = _seed_store_service(tmp_path)
    await seed()
    # Empty project (overview reads the mapping table; it does not force
    # graphiti init, so available stays False until the first ingest/search).
    overview = await service.project_overview("usr_a", "proj_b")
    assert overview["enabled"] is True
    assert overview["available"] is False
    assert overview["episode_count"] == 0
    assert overview["last_ingested_at"] is None

    saved = await _append(store, [("user", "q1"), ("assistant", "a1")])
    await service._ingest(
        user_id="usr_a", project_id="proj_b", session_id="sess_c", messages=saved
    )
    overview = await service.project_overview("usr_a", "proj_b")
    assert overview["episode_count"] == 1
    assert overview["last_ingested_at"] is not None

    stats = await service.global_stats()
    assert stats["enabled"] is True
    assert stats["episode_count"] == 1
    assert stats["user_count"] == 1
    assert stats["project_count"] == 1


async def test_project_overview_without_store_returns_zero():
    service, _fake, _clear = _service()  # no metadata_store
    overview = await service.project_overview("usr_a", "proj_b")
    assert overview["episode_count"] == 0
    assert overview["last_ingested_at"] is None


# --------------------------------------------------------- per-user fairness


async def test_ingest_per_user_inflight_cap_defers_excess():
    fake = FakeGraphiti()
    fake.add_episode_delay = 0.05
    service = ChatMemoryService(
        ChatMemoryConfig(enabled=True, ingest_concurrency=8, max_inflight_per_user=2),
        graphiti_factory=lambda _cfg: fake,
    )
    # Three rapid batches from the same user; the 3rd exceeds the cap (2) and
    # is deferred (returns None) rather than piling onto the shared LLM.
    tasks = [
        service.schedule_ingest(
            user_id="usr_a",
            project_id=f"proj_{i}",
            session_id="sess_c",
            messages=[_message("user", f"q{i}", 1)],
        )
        for i in range(3)
    ]
    assert tasks[0] is not None and tasks[1] is not None
    assert tasks[2] is None  # deferred by the per-user cap
    await service.wait_for_background_tasks()
    # After the first two drain, the counter is released and new work flows.
    again = service.schedule_ingest(
        user_id="usr_a",
        project_id="proj_x",
        session_id="sess_c",
        messages=[_message("user", "q", 1)],
    )
    assert again is not None
    await service.wait_for_background_tasks()


