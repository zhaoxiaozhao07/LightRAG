"""Unit tests for the graphiti-backed chat memory service.

All tests run offline: graphiti is replaced by an injected fake via the
``graphiti_factory`` / ``clear_data_fn`` constructor hooks, so neither
``graphiti-core`` nor a live Neo4j/LLM is required.
"""

from __future__ import annotations

# ruff: noqa: E402 - LightRAG API modules parse sys.argv at import time.

import asyncio
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from lightrag.api.chat_memory_service import (
    MEMORY_SEARCH_MAX_LIMIT,
    ChatMemoryConfig,
    ChatMemoryService,
    ChatMemoryUnavailableError,
    _ExtraBodyAsyncOpenAI,
    _RerankFnCrossEncoder,
)
from lightrag.api.kb_service import utc_now_iso
from lightrag.api.metadata_store import (
    ChatMessageRecord,
    ChatProjectRecord,
    ChatSessionRecord,
    EnterpriseUserRecord,
    SQLiteMetadataStore,
)
sys.argv = _original_argv

pytestmark = pytest.mark.offline


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
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def append(self, event_type, **kwargs):
        self.events.append((event_type, kwargs))

    def of_type(self, event_type):
        return [event for event in self.events if event[0] == event_type]


class FakeClearData:
    def __init__(self):
        self.calls: list[tuple[object, list[str]]] = []

    async def __call__(self, graphiti, group_ids):
        assert group_ids, "clear_data must never receive an empty/None group list"
        self.calls.append((graphiti, list(group_ids)))


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
):
    fake = fake or FakeGraphiti()
    clear = clear or FakeClearData()
    service = ChatMemoryService(
        config or ChatMemoryConfig(enabled=True),
        audit_service=audit,
        graphiti_factory=lambda _config: fake,
        clear_data_fn=clear,
    )
    return service, fake, clear


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
        SimpleNamespace(chat_memory_enabled=True, memory_neo4j_username="override")
    )
    assert config.neo4j_uri == "bolt://fallback:7687"
    assert config.neo4j_username == "override"
    assert config.neo4j_password == "secret"
    assert config.neo4j_database == "neo4j"


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


async def test_search_forces_group_ids_and_maps_fields():
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
    assert fake.search_calls == [
        {
            "query": "低温性能结论？",
            "group_ids": ["usr_a--proj_b"],
            "num_results": 5,
        }
    ]
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
    assert fake.search_calls == []
    assert facts[0]["fact"] == "f"


# ----------------------------------------------------- store-backed durability


def _seed_store_service(tmp_path, fake=None, clear=None, config=None):
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


