"""Phase 1 contract tests for durable enterprise Chat Memory store state."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import uuid
from datetime import datetime
from typing import Any

import pytest

from lightrag.api.kb_service import utc_now_iso
from lightrag.api.metadata_store import (
    ChatMemoryEpisodeRecord,
    ChatMessageRecord,
    ChatProjectRecord,
    ChatSessionRecord,
    EnterpriseUserRecord,
    SQLiteMetadataStore,
    chat_memory_graph_group_id,
)

pytestmark = pytest.mark.offline

_POSTGRES_DSN = os.getenv("LIGHTRAG_KB_POSTGRES_TEST_DSN") or os.getenv(
    "POSTGRES_TEST_DSN"
)
_FINGERPRINT = "chat-memory-config:v1:test"
_GRAPH_FINGERPRINT = "chat-memory-graph-store:v1:test"


async def _make_store(backend: str, tmp_path) -> Any:
    if backend == "sqlite":
        store = SQLiteMetadataStore(tmp_path / "chat-memory-metadata.sqlite3")
    else:
        from lightrag.api.postgres_metadata_store import PostgresMetadataStore

        store = PostgresMetadataStore(
            dsn=_POSTGRES_DSN,
            min_size=1,
            max_size=2,
            operation_lock_pool_max_size=2,
        )
    await store.initialize()
    store._test_chat_memory_user_ids = []  # type: ignore[attr-defined]
    return store


@pytest.fixture(params=["sqlite", "postgres"])
async def store(request, tmp_path):
    backend = request.param
    if backend == "postgres" and not _POSTGRES_DSN:
        pytest.skip(
            "live PostgreSQL Chat Memory contract test skipped: set "
            "LIGHTRAG_KB_POSTGRES_TEST_DSN to enable"
        )
    instance = await _make_store(backend, tmp_path)
    try:
        yield instance
    finally:
        if backend == "postgres":
            user_ids = instance._test_chat_memory_user_ids  # type: ignore[attr-defined]
            if user_ids:
                async with instance._pool_or_raise().acquire() as conn:
                    async with conn.transaction():
                        for table in (
                            "enterprise_chat_memory_outbox",
                            "enterprise_chat_memory_generations",
                            "enterprise_chat_memory_groups",
                            "enterprise_chat_memory_episodes",
                            "enterprise_chat_messages",
                            "enterprise_chat_sessions",
                            "enterprise_chat_projects",
                            "enterprise_tenant_user_kb_overrides",
                            "enterprise_tenant_memberships",
                            "enterprise_kb_acl",
                            "enterprise_user_kb_query_settings",
                        ):
                            await conn.execute(
                                f"DELETE FROM {table} WHERE user_id = ANY($1::text[])",
                                user_ids,
                            )
                        await conn.execute(
                            "DELETE FROM enterprise_users WHERE id = ANY($1::text[])",
                            user_ids,
                        )
        await instance.close()


def _user() -> EnterpriseUserRecord:
    now = utc_now_iso()
    token = uuid.uuid4().hex[:12]
    return EnterpriseUserRecord(
        id=f"usr_cm_{token}",
        username=f"cm_{token}",
        password_hash="{bcrypt}$2b$12$placeholderplaceholderplaceholderplaceholderplace",
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


async def _create_chat(store, *, project_name: str = "memory"):
    user = await store.upsert_enterprise_user(_user())
    store._test_chat_memory_user_ids.append(user.id)  # type: ignore[attr-defined]
    now = utc_now_iso()
    project = await store.create_chat_project(
        ChatProjectRecord(
            id=f"proj_cm_{uuid.uuid4().hex[:12]}",
            user_id=user.id,
            name=project_name,
            created_at=now,
            updated_at=now,
        )
    )
    session = await store.create_chat_session(
        ChatSessionRecord(
            id=f"sess_cm_{uuid.uuid4().hex[:12]}",
            project_id=project.id,
            user_id=user.id,
            name="session",
            created_at=now,
            updated_at=now,
        )
    )
    return user, project, session


def _message(user_id: str, project_id: str, session_id: str, content: str):
    return ChatMessageRecord(
        id=f"msg_cm_{uuid.uuid4().hex[:16]}",
        session_id=session_id,
        project_id=project_id,
        user_id=user_id,
        role="user",
        content=content,
        metadata={},
        seq=0,
        created_at=utc_now_iso(),
    )


async def test_memory_append_is_atomic_monotonic_and_replay_admitted_only(store):
    user, project, session = await _create_chat(store)

    feature_off = await store.append_chat_messages(
        [_message(user.id, project.id, session.id, "feature off")]
    )
    assert feature_off[0].append_batch_id is None
    assert feature_off[0].project_event_seq is None
    assert feature_off[0].memory_reference_time is None

    first = await store.append_chat_messages_with_memory(
        [
            _message(user.id, project.id, session.id, "one"),
            _message(user.id, project.id, session.id, "two"),
        ],
        config_fingerprint=_FINGERPRINT,
    )
    second = await store.append_chat_messages_with_memory(
        [_message(user.id, project.id, session.id, "three")],
        config_fingerprint=_FINGERPRINT,
    )

    assert [item.project_event_seq for item in first] == [1, 1]
    assert [item.project_event_seq for item in second] == [2]
    assert first[0].append_batch_id == first[1].append_batch_id
    assert first[0].append_batch_id != second[0].append_batch_id
    first_reference = datetime.fromisoformat(first[0].memory_reference_time or "")
    second_reference = datetime.fromisoformat(second[0].memory_reference_time or "")
    assert second_reference > first_reference

    group = await store.get_chat_memory_group(user.id, project.id)
    assert group is not None
    assert group.active_generation is None
    assert group.desired_generation == 1
    assert group.next_event_seq == 3
    assert group.last_reference_time == second[0].memory_reference_time

    generation = await store.get_chat_memory_generation(user.id, project.id, 1)
    assert generation is not None
    assert generation.state == "building"
    assert generation.graph_group_id == chat_memory_graph_group_id(
        user.id, project.id, 1
    )

    events = await store.list_chat_memory_events(
        user_id=user.id, project_id=project.id
    )
    assert [(event.event_seq, event.event_type, event.status) for event in events] == [
        (1, "ingest", "pending"),
        (2, "ingest", "pending"),
    ]
    assert len({event.deterministic_key for event in events}) == 2
    assert await store.get_chat_memory_event(events[0].event_id) == events[0]
    assert (
        await store.get_chat_memory_event_by_sequence(user.id, project.id, 2)
        == events[1]
    )
    await store.record_chat_memory_episode(
        ChatMemoryEpisodeRecord(
            episode_uuid=f"ep_cm_{uuid.uuid4().hex[:12]}",
            session_id=session.id,
            project_id=project.id,
            user_id=user.id,
            first_seq=first[0].seq,
            last_seq=first[-1].seq,
            created_at=utc_now_iso(),
            event_id=events[0].event_id,
            generation=events[0].generation,
            graph_group_id=events[0].graph_group_id,
            append_batch_id=events[0].append_batch_id,
            project_event_seq=events[0].event_seq,
        )
    )
    mapped = await store.list_chat_memory_episodes_for_session(
        user.id, project.id, session.id
    )
    assert mapped[0].event_id == events[0].event_id
    assert mapped[0].generation == 1
    assert mapped[0].graph_group_id == events[0].graph_group_id
    assert mapped[0].append_batch_id == events[0].append_batch_id
    assert mapped[0].project_event_seq == events[0].event_seq

    replay = await store.list_admitted_chat_memory_replay_batches(
        user.id, project.id, through_event_seq=2
    )
    assert [batch.project_event_seq for batch in replay] == [1, 2]
    assert [[message.content for message in batch.messages] for batch in replay] == [
        ["one", "two"],
        ["three"],
    ]

    before_failure = await store.get_chat_memory_group(user.id, project.id)
    assert before_failure is not None
    duplicate = _message(user.id, project.id, session.id, "duplicate")
    duplicate.id = first[0].id
    with pytest.raises(Exception):
        await store.append_chat_messages_with_memory(
            [duplicate], config_fingerprint=_FINGERPRINT
        )
    after_failure = await store.get_chat_memory_group(user.id, project.id)
    assert after_failure is not None
    assert after_failure.next_event_seq == before_failure.next_event_seq
    assert after_failure.last_reference_time == before_failure.last_reference_time
    assert len(
        await store.list_chat_memory_events(user_id=user.id, project_id=project.id)
    ) == 2


async def test_chat_memory_read_token_is_atomic_for_inactive_and_active_group(store):
    user, project, session = await _create_chat(store)
    assert await store.get_chat_memory_read_token(user.id, project.id) is None

    await store.append_chat_messages_with_memory(
        [_message(user.id, project.id, session.id, "token")],
        config_fingerprint=_FINGERPRINT,
        graph_store_fingerprint=_GRAPH_FINGERPRINT,
    )
    inactive = await store.get_chat_memory_read_token(user.id, project.id)
    assert inactive is not None
    assert inactive.state == "rebuilding"
    assert inactive.state_version == 1
    assert inactive.active_generation is None
    assert inactive.active_config_fingerprint is None
    assert inactive.active_graph_store_fingerprint is None
    assert inactive.graph_group_id is None
    assert inactive.generation_state is None

    event = await store.claim_next_chat_memory_event(
        _FINGERPRINT,
        runtime_graph_store_fingerprint=_GRAPH_FINGERPRINT,
        event_types=["ingest"],
    )
    assert event is not None and event.claim_token
    assert event.graph_store_fingerprint == _GRAPH_FINGERPRINT
    await store.mark_chat_memory_event_side_effect_started(
        event.event_id,
        event.claim_token,
        _FINGERPRINT,
        runtime_graph_store_fingerprint=_GRAPH_FINGERPRINT,
    )
    await store.finalize_chat_memory_ingest(
        event.event_id,
        event.claim_token,
        _FINGERPRINT,
        episode_uuid=f"read-token-{uuid.uuid4().hex}",
        runtime_graph_store_fingerprint=_GRAPH_FINGERPRINT,
    )

    active = await store.get_chat_memory_read_token(user.id, project.id)
    assert active is not None
    assert active.state == "active"
    assert active.state_version == 2
    assert active.active_generation == 1
    assert active.active_config_fingerprint == _FINGERPRINT
    assert active.active_graph_store_fingerprint == _GRAPH_FINGERPRINT
    assert active.graph_group_id == chat_memory_graph_group_id(
        user.id, project.id, 1
    )
    assert active.generation_state == "active"


async def test_episode_mapping_is_unique_per_generation_and_append_batch(store):
    user, project, session = await _create_chat(store)
    saved = await store.append_chat_messages_with_memory(
        [_message(user.id, project.id, session.id, "mapped")],
        config_fingerprint=_FINGERPRINT,
    )
    event = (
        await store.list_chat_memory_events(user_id=user.id, project_id=project.id)
    )[0]
    assert event.append_batch_id == saved[0].append_batch_id

    def mapping(episode_uuid: str, generation: int) -> ChatMemoryEpisodeRecord:
        return ChatMemoryEpisodeRecord(
            episode_uuid=episode_uuid,
            session_id=session.id,
            project_id=project.id,
            user_id=user.id,
            first_seq=saved[0].seq,
            last_seq=saved[0].seq,
            created_at=utc_now_iso(),
            event_id=event.event_id,
            generation=generation,
            graph_group_id=chat_memory_graph_group_id(
                user.id, project.id, generation
            ),
            append_batch_id=event.append_batch_id,
            project_event_seq=event.event_seq,
        )

    await store.record_chat_memory_episode(
        mapping(f"ep_cm_{uuid.uuid4().hex[:12]}", 1)
    )
    await store.record_chat_memory_episode(
        mapping(f"ep_cm_{uuid.uuid4().hex[:12]}", 2)
    )
    with pytest.raises(Exception):
        await store.record_chat_memory_episode(
            mapping(f"ep_cm_{uuid.uuid4().hex[:12]}", 1)
        )

    rows = await store.list_chat_memory_episodes_for_session(
        user.id, project.id, session.id
    )
    assert sorted((row.generation, row.event_id) for row in rows) == [
        (1, event.event_id),
        (2, event.event_id),
    ]


async def test_episode_identity_rejects_every_partial_combination(store):
    user, project, session = await _create_chat(store)
    saved = await store.append_chat_messages_with_memory(
        [_message(user.id, project.id, session.id, "identity")],
        config_fingerprint=_FINGERPRINT,
    )
    event = (
        await store.list_chat_memory_events(user_id=user.id, project_id=project.id)
    )[0]

    def identity_record(mask: int) -> ChatMemoryEpisodeRecord:
        return ChatMemoryEpisodeRecord(
            episode_uuid=f"ep_identity_{mask}_{uuid.uuid4().hex[:10]}",
            session_id=session.id,
            project_id=project.id,
            user_id=user.id,
            first_seq=saved[0].seq,
            last_seq=saved[0].seq,
            created_at=utc_now_iso(),
            event_id=event.event_id if mask & 0b00001 else None,
            generation=1 if mask & 0b00010 else None,
            graph_group_id=(
                chat_memory_graph_group_id(user.id, project.id, 1)
                if mask & 0b00100
                else None
            ),
            append_batch_id=event.append_batch_id if mask & 0b01000 else None,
            project_event_seq=event.event_seq if mask & 0b10000 else None,
        )

    valid_masks = {0b00000, 0b00111, 0b11111}
    for mask in range(0b100000):
        if mask in valid_masks:
            continue
        with pytest.raises(Exception):
            await store.record_chat_memory_episode(identity_record(mask))

    for mask in sorted(valid_masks):
        await store.record_chat_memory_episode(identity_record(mask))
    rows = await store.list_chat_memory_episodes_for_session(
        user.id, project.id, session.id
    )
    assert len(rows) == 3
    assert {
        (
            row.event_id is not None,
            row.generation is not None,
            row.graph_group_id is not None,
            row.append_batch_id is not None,
            row.project_event_seq is not None,
        )
        for row in rows
    } == {
        (False, False, False, False, False),
        (True, True, True, False, False),
        (True, True, True, True, True),
    }


async def test_delete_advances_generation_and_preserves_shared_event_domain(store):
    user, project, session = await _create_chat(store)
    first = await store.append_chat_messages_with_memory(
        [_message(user.id, project.id, session.id, "one")],
        config_fingerprint=_FINGERPRINT,
    )
    assert await store.delete_chat_message_with_memory(
        user.id,
        project.id,
        session.id,
        first[0].id,
        config_fingerprint=_FINGERPRINT,
    )
    second = await store.append_chat_messages_with_memory(
        [_message(user.id, project.id, session.id, "after delete")],
        config_fingerprint=_FINGERPRINT,
    )
    assert second[0].project_event_seq == 3
    assert datetime.fromisoformat(
        second[0].memory_reference_time or ""
    ) > datetime.fromisoformat(first[0].memory_reference_time or "")

    group = await store.get_chat_memory_group(user.id, project.id)
    assert group is not None
    assert group.desired_generation == 2
    assert group.next_event_seq == 4
    assert group.state == "rebuilding"
    inventory = await store.list_chat_memory_generations(user.id, project.id)
    assert [(item.generation, item.state) for item in inventory] == [
        (1, "abandoned"),
        (2, "building"),
    ]
    events = await store.list_chat_memory_events(
        user_id=user.id, project_id=project.id
    )
    assert [(event.event_seq, event.event_type, event.status) for event in events] == [
        (1, "ingest", "superseded"),
        (2, "rebuild", "pending"),
        (3, "ingest", "pending"),
    ]
    replay = await store.list_admitted_chat_memory_replay_batches(
        user.id, project.id, through_event_seq=group.next_event_seq - 1
    )
    assert [batch.project_event_seq for batch in replay] == [3]


@pytest.mark.parametrize("delete_scope", ["message", "session"])
async def test_message_and_session_delete_bind_existing_graph_store(
    store,
    delete_scope,
):
    old_extraction = "chat-memory-extraction:v1:old"
    new_extraction = "chat-memory-extraction:v2:new"
    new_graph = "chat-memory-graph-store:v2:new"
    user, project, session = await _create_chat(store)
    saved = await store.append_chat_messages_with_memory(
        [_message(user.id, project.id, session.id, "remember")],
        config_fingerprint=old_extraction,
        graph_store_fingerprint=_GRAPH_FINGERPRINT,
    )

    if delete_scope == "message":
        assert await store.delete_chat_message_with_memory(
            user.id,
            project.id,
            session.id,
            saved[0].id,
            config_fingerprint=new_extraction,
            graph_store_fingerprint=new_graph,
        )
    else:
        assert await store.delete_chat_session_with_memory(
            user.id,
            project.id,
            session.id,
            config_fingerprint=new_extraction,
            graph_store_fingerprint=new_graph,
        ) == (True, 1)

    group = await store.get_chat_memory_group(user.id, project.id)
    assert group is not None
    assert group.desired_generation == 2
    assert group.desired_graph_store_fingerprint == _GRAPH_FINGERPRINT
    inventory = await store.list_chat_memory_generations(user.id, project.id)
    assert {item.graph_store_fingerprint for item in inventory} == {
        _GRAPH_FINGERPRINT
    }
    events = await store.list_chat_memory_events(
        user_id=user.id, project_id=project.id
    )
    assert [(event.event_type, event.status) for event in events] == [
        ("ingest", "superseded"),
        ("rebuild", "pending"),
    ]
    assert events[-1].graph_store_fingerprint == _GRAPH_FINGERPRINT
    assert (
        await store.claim_next_chat_memory_event(
            new_extraction,
            runtime_graph_store_fingerprint=new_graph,
            event_types=["rebuild"],
        )
        is None
    )
    claimed = await store.claim_next_chat_memory_event(
        new_extraction,
        runtime_graph_store_fingerprint=_GRAPH_FINGERPRINT,
        event_types=["rebuild"],
    )
    assert claimed is not None and claimed.event_id == events[-1].event_id


async def test_concurrent_appends_share_strict_project_order(store):
    user, project, session = await _create_chat(store)
    if isinstance(store, SQLiteMetadataStore):
        peer = SQLiteMetadataStore(store.db_path)
    else:
        from lightrag.api.postgres_metadata_store import PostgresMetadataStore

        peer = PostgresMetadataStore(
            dsn=_POSTGRES_DSN,
            min_size=1,
            max_size=1,
            operation_lock_pool_max_size=1,
        )
    await peer.initialize()
    try:
        results = await asyncio.gather(
            store.append_chat_messages_with_memory(
                [_message(user.id, project.id, session.id, "left")],
                config_fingerprint=_FINGERPRINT,
            ),
            peer.append_chat_messages_with_memory(
                [_message(user.id, project.id, session.id, "right")],
                config_fingerprint=_FINGERPRINT,
            ),
        )
    finally:
        await peer.close()

    saved = [batch[0] for batch in results]
    assert {item.project_event_seq for item in saved} == {1, 2}
    references = sorted(
        datetime.fromisoformat(item.memory_reference_time or "") for item in saved
    )
    assert references[1] > references[0]
    events = await store.list_chat_memory_events(
        user_id=user.id, project_id=project.id
    )
    assert [event.event_seq for event in events] == [1, 2]
    messages, total = await store.list_chat_messages(
        user.id, project.id, session.id
    )
    assert total == 2
    assert [message.seq for message in messages] == [1, 2]


async def test_postgres_outbox_control_times_use_database_clock(store, monkeypatch):
    if isinstance(store, SQLiteMetadataStore):
        pytest.skip("PostgreSQL database-clock contract")

    import lightrag.api.postgres_metadata_store as postgres_metadata_store

    user, project, session = await _create_chat(store)
    monkeypatch.setattr(
        postgres_metadata_store,
        "utc_now_iso",
        lambda: "2099-01-01T00:00:00+00:00",
    )
    await store.append_chat_messages_with_memory(
        [_message(user.id, project.id, session.id, "database clock")],
        config_fingerprint=_FINGERPRINT,
    )
    event = (
        await store.list_chat_memory_events(user_id=user.id, project_id=project.id)
    )[0]
    group = await store.get_chat_memory_group(user.id, project.id)
    generation = await store.get_chat_memory_generation(user.id, project.id, 1)
    assert group is not None and group.last_reference_time is not None
    assert generation is not None
    async with store._pool_or_raise().acquire() as conn:
        database_now = await conn.fetchval("SELECT clock_timestamp()")

    available_at = datetime.fromisoformat(event.available_at)
    assert event.available_at == event.created_at == event.updated_at
    assert available_at.year != 2099
    assert generation.created_at == generation.updated_at
    assert datetime.fromisoformat(generation.created_at) <= datetime.fromisoformat(
        group.last_reference_time
    )
    assert group.updated_at == group.last_reference_time
    assert datetime.fromisoformat(group.last_reference_time) <= available_at
    assert available_at <= database_now


async def test_project_purge_tombstone_survives_source_delete(store):
    old_extraction = "chat-memory-extraction:v1:old"
    new_extraction = "chat-memory-extraction:v2:new"
    new_graph = "chat-memory-graph-store:v2:new"
    user, project, session = await _create_chat(store)
    await store.append_chat_messages_with_memory(
        [_message(user.id, project.id, session.id, "remember")],
        config_fingerprint=old_extraction,
        graph_store_fingerprint=_GRAPH_FINGERPRINT,
    )
    assert await store.delete_chat_project_with_memory(
        user.id,
        project.id,
        config_fingerprint=new_extraction,
        graph_store_fingerprint=new_graph,
    ) == (True, 1, 1)
    assert await store.get_chat_project(user.id, project.id) is None

    group = await store.get_chat_memory_group(user.id, project.id)
    assert group is not None and group.state == "deleting"
    assert group.desired_graph_store_fingerprint == _GRAPH_FINGERPRINT
    events = await store.list_chat_memory_events(
        user_id=user.id, project_id=project.id
    )
    assert [(event.event_type, event.status) for event in events] == [
        ("ingest", "superseded"),
        ("purge", "pending"),
    ]
    assert events[-1].graph_store_fingerprint == _GRAPH_FINGERPRINT
    assert all(
        item.state == "purge_pending"
        for item in await store.list_chat_memory_generations(user.id, project.id)
    )
    assert (
        await store.claim_next_chat_memory_event(
            new_extraction,
            runtime_graph_store_fingerprint=new_graph,
            event_types=["purge"],
        )
        is None
    )
    claimed = await store.claim_next_chat_memory_event(
        new_extraction,
        runtime_graph_store_fingerprint=_GRAPH_FINGERPRINT,
        event_types=["purge"],
    )
    assert claimed is not None and claimed.event_id == events[-1].event_id


async def test_user_delete_creates_sorted_project_purges_that_survive(store):
    old_extraction = "chat-memory-extraction:v1:old"
    new_extraction = "chat-memory-extraction:v2:new"
    new_graph = "chat-memory-graph-store:v2:new"
    user, first_project, first_session = await _create_chat(
        store, project_name="first"
    )
    now = utc_now_iso()
    second_project = await store.create_chat_project(
        ChatProjectRecord(
            id=f"proj_cm_{uuid.uuid4().hex[:12]}",
            user_id=user.id,
            name="second",
            created_at=now,
            updated_at=now,
        )
    )
    second_session = await store.create_chat_session(
        ChatSessionRecord(
            id=f"sess_cm_{uuid.uuid4().hex[:12]}",
            project_id=second_project.id,
            user_id=user.id,
            name="second",
            created_at=now,
            updated_at=now,
        )
    )
    for project, session in (
        (first_project, first_session),
        (second_project, second_session),
    ):
        await store.append_chat_messages_with_memory(
            [_message(user.id, project.id, session.id, "remember")],
            config_fingerprint=old_extraction,
            graph_store_fingerprint=_GRAPH_FINGERPRINT,
        )

    assert await store.delete_enterprise_user_with_memory(
        user.id,
        config_fingerprint=new_extraction,
        graph_store_fingerprint=new_graph,
        actor_user_id="usr_admin",
    )
    assert await store.get_enterprise_user_by_id(user.id) is None
    for project in (first_project, second_project):
        group = await store.get_chat_memory_group(user.id, project.id)
        assert group is not None and group.state == "deleting"
        assert group.desired_graph_store_fingerprint == _GRAPH_FINGERPRINT
        events = await store.list_chat_memory_events(
            user_id=user.id, project_id=project.id
        )
        assert events[-1].event_type == "purge"
        assert events[-1].status == "pending"
        assert events[-1].graph_store_fingerprint == _GRAPH_FINGERPRINT
        assert events[-1].actor_user_id == "usr_admin"

    assert (
        await store.claim_next_chat_memory_event(
            new_extraction,
            runtime_graph_store_fingerprint=new_graph,
            event_types=["purge"],
        )
        is None
    )
    claimed = [
        await store.claim_next_chat_memory_event(
            new_extraction,
            runtime_graph_store_fingerprint=_GRAPH_FINGERPRINT,
            event_types=["purge"],
        )
        for _ in range(2)
    ]
    assert all(event is not None for event in claimed)
    assert {event.project_id for event in claimed if event is not None} == {
        first_project.id,
        second_project.id,
    }


async def test_chat_memory_schema_initialization_is_idempotent(store):
    if isinstance(store, SQLiteMetadataStore):
        def sqlite_schema_snapshot():
            with sqlite3.connect(store.db_path) as conn:
                return conn.execute(
                    """
                    SELECT type, name, sql FROM sqlite_master
                    WHERE name LIKE 'enterprise_chat_memory_%'
                       OR name LIKE '%chat_memory_episode_generation_batch%'
                    ORDER BY type, name
                    """
                ).fetchall()

        before = sqlite_schema_snapshot()
        await store.initialize()
        after = sqlite_schema_snapshot()
        assert after == before
        assert any(
            row[1] == "uq_enterprise_chat_memory_episode_generation_batch"
            for row in after
        )
        with sqlite3.connect(store.db_path) as conn:
            group_columns = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(enterprise_chat_memory_groups)"
                )
            }
            generation_columns = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(enterprise_chat_memory_generations)"
                )
            }
            outbox_columns = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(enterprise_chat_memory_outbox)"
                )
            }
            versions = {
                row[0] for row in conn.execute("SELECT version FROM metadata_schema")
            }
        assert {
            "active_graph_store_fingerprint",
            "desired_graph_store_fingerprint",
        } <= group_columns
        assert "graph_store_fingerprint" in generation_columns
        assert "graph_store_fingerprint" in outbox_columns
        assert 11 in versions
    else:
        async def postgres_schema_snapshot():
            async with store._pool_or_raise().acquire() as conn:
                constraints = await conn.fetch(
                    """
                    SELECT oid::text AS oid, conrelid::regclass::text AS table_name,
                           conname, convalidated, pg_get_constraintdef(oid) AS ddl
                    FROM pg_constraint
                    WHERE conrelid IN (
                        'enterprise_chat_messages'::regclass,
                        'enterprise_chat_memory_episodes'::regclass
                    ) AND contype = 'c'
                    ORDER BY table_name, conname
                    """
                )
                indexes = await conn.fetch(
                    """
                    SELECT c.oid::text AS oid, c.relname,
                           pg_get_indexdef(c.oid) AS ddl
                    FROM pg_class c
                    WHERE c.relname IN (
                        'uq_enterprise_chat_memory_episode_generation_batch',
                        'idx_enterprise_chat_messages_memory_replay',
                        'idx_enterprise_chat_memory_episodes_generation'
                    )
                    ORDER BY c.relname
                    """
                )
                version = await conn.fetchrow(
                    """
                    SELECT version, applied_at FROM kb_metadata_schema
                    WHERE version = 2
                    """
                )
                graph_columns = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM information_schema.columns
                    WHERE table_schema = current_schema()
                      AND (
                          (table_name = 'enterprise_chat_memory_groups'
                           AND column_name IN (
                               'active_graph_store_fingerprint',
                               'desired_graph_store_fingerprint'
                           ))
                          OR
                          (table_name IN (
                               'enterprise_chat_memory_generations',
                               'enterprise_chat_memory_outbox'
                           ) AND column_name = 'graph_store_fingerprint')
                      )
                    """
                )
                schema_v4 = await conn.fetchval(
                    "SELECT 1 FROM kb_metadata_schema WHERE version = 4"
                )
            return (
                [tuple(row.values()) for row in constraints],
                [tuple(row.values()) for row in indexes],
                tuple(version.values()) if version is not None else None,
                int(graph_columns or 0),
                schema_v4,
            )

        before = await postgres_schema_snapshot()
        await store.initialize()
        after = await postgres_schema_snapshot()
        assert after == before
        constraint_names = {row[2] for row in after[0]}
        assert "enterprise_chat_messages_admission_v2_check" in constraint_names
        assert (
            "enterprise_chat_memory_episode_generation_v2_check"
            in constraint_names
        )
        assert "enterprise_chat_memory_episode_identity_v2_check" in constraint_names
        assert "enterprise_chat_messages_admission_v1_check" not in constraint_names
        assert (
            "enterprise_chat_memory_episode_generation_v1_check"
            not in constraint_names
        )
        assert (
            "enterprise_chat_memory_episode_mapping_v2_check"
            not in constraint_names
        )
        assert after[3] == 4
        assert after[4] == 1
    assert await store.count_chat_memory_events() >= 0


async def test_sqlite_episode_identity_migration_downgrades_invalid_rows(tmp_path):
    db_path = tmp_path / "partial-episode-identity.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE enterprise_chat_memory_episodes (
                episode_uuid TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                first_seq INTEGER NOT NULL,
                last_seq INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                event_id TEXT,
                generation INTEGER,
                graph_group_id TEXT,
                append_batch_id TEXT,
                project_event_seq INTEGER
            );
            INSERT INTO enterprise_chat_memory_episodes VALUES (
                'partial_triplet', 'sess_partial', 'proj_partial', 'usr_partial',
                1, 1, '2026-01-01T00:00:00+00:00',
                'event_partial', NULL, 'graph_partial', NULL, NULL
            );
            INSERT INTO enterprise_chat_memory_episodes VALUES (
                'partial_pair', 'sess_partial', 'proj_partial', 'usr_partial',
                2, 2, '2026-01-01T00:00:00+00:00',
                'event_pair', 1, 'graph_pair', 'batch_pair', NULL
            );
            INSERT INTO enterprise_chat_memory_episodes VALUES (
                'duplicate_a', 'sess_partial', 'proj_partial', 'usr_partial',
                3, 3, '2026-01-01T00:00:00+00:00',
                'event_a', 1, 'graph_1', 'batch_duplicate', 3
            );
            INSERT INTO enterprise_chat_memory_episodes VALUES (
                'duplicate_b', 'sess_partial', 'proj_partial', 'usr_partial',
                3, 3, '2026-01-01T00:00:00+00:00',
                'event_b', 1, 'graph_1', 'batch_duplicate', 3
            );
            """
        )

    store = SQLiteMetadataStore(db_path)
    await store.initialize()
    rows = await store.list_chat_memory_episodes_for_session(
        "usr_partial", "proj_partial", "sess_partial"
    )
    by_id = {row.episode_uuid: row for row in rows}
    for episode_id in ("partial_triplet", "partial_pair", "duplicate_b"):
        row = by_id[episode_id]
        assert (
            row.event_id,
            row.generation,
            row.graph_group_id,
            row.append_batch_id,
            row.project_event_seq,
        ) == (None, None, None, None, None)
    assert (
        by_id["duplicate_a"].event_id,
        by_id["duplicate_a"].generation,
        by_id["duplicate_a"].graph_group_id,
        by_id["duplicate_a"].append_batch_id,
        by_id["duplicate_a"].project_event_seq,
    ) == ("event_a", 1, "graph_1", "batch_duplicate", 3)
    with sqlite3.connect(db_path) as conn:
        table_sql = conn.execute(
            """
            SELECT sql FROM sqlite_master
            WHERE type = 'table' AND name = 'enterprise_chat_memory_episodes'
            """
        ).fetchone()[0]
    assert "enterprise_chat_memory_episode_identity_v2_check" in table_sql


async def test_sqlite_chat_memory_migration_is_idempotent_and_legacy_safe(tmp_path):
    db_path = tmp_path / "legacy-chat-memory.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE enterprise_chat_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                seq INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE enterprise_chat_memory_episodes (
                episode_uuid TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                first_seq INTEGER NOT NULL,
                last_seq INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            INSERT INTO enterprise_chat_messages VALUES (
                'msg_legacy', 'sess_legacy', 'proj_legacy', 'usr_legacy',
                'user', 'legacy', '{}', 1, '2026-01-01T00:00:00+00:00'
            );
            INSERT INTO enterprise_chat_memory_episodes VALUES (
                'ep_legacy', 'sess_legacy', 'proj_legacy', 'usr_legacy',
                1, 1, '2026-01-01T00:00:00+00:00'
            );
            """
        )

    store = SQLiteMetadataStore(db_path)
    await store.initialize()
    with sqlite3.connect(db_path) as conn:
        before_schema = conn.execute(
            """
            SELECT type, name, sql FROM sqlite_master
            WHERE name LIKE 'enterprise_chat_memory_%'
               OR name LIKE '%chat_memory_episode_generation_batch%'
            ORDER BY type, name
            """
        ).fetchall()
    await store.initialize()
    with sqlite3.connect(db_path) as conn:
        after_schema = conn.execute(
            """
            SELECT type, name, sql FROM sqlite_master
            WHERE name LIKE 'enterprise_chat_memory_%'
               OR name LIKE '%chat_memory_episode_generation_batch%'
            ORDER BY type, name
            """
        ).fetchall()
    assert after_schema == before_schema
    messages, total = await store.list_chat_messages(
        "usr_legacy", "proj_legacy", "sess_legacy"
    )
    assert total == 1
    assert messages[0].append_batch_id is None
    assert messages[0].project_event_seq is None
    assert messages[0].memory_reference_time is None
    episodes = await store.list_chat_memory_episodes_for_session(
        "usr_legacy", "proj_legacy", "sess_legacy"
    )
    assert episodes[0].event_id is None
    assert episodes[0].generation is None
    assert episodes[0].graph_group_id is None
    assert episodes[0].append_batch_id is None
    assert episodes[0].project_event_seq is None

    with sqlite3.connect(db_path) as conn:
        message_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(enterprise_chat_messages)")
        }
        episode_columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(enterprise_chat_memory_episodes)"
            )
        }
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        indexes = {
            row[0]
            for row in conn.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'index'
                  AND tbl_name = 'enterprise_chat_memory_episodes'
                """
            )
        }
    assert {
        "append_batch_id",
        "project_event_seq",
        "memory_reference_time",
    } <= message_columns
    assert {
        "event_id",
        "generation",
        "graph_group_id",
        "append_batch_id",
        "project_event_seq",
    } <= episode_columns
    assert {
        "enterprise_chat_memory_groups",
        "enterprise_chat_memory_generations",
        "enterprise_chat_memory_outbox",
    } <= tables
    assert "uq_enterprise_chat_memory_episode_generation_batch" in indexes
    assert "uq_enterprise_chat_memory_episodes_event" not in indexes


async def test_sqlite_graph_store_fingerprint_migration_backfills_idempotently(
    tmp_path,
):
    db_path = tmp_path / "legacy-chat-memory-graph-fingerprint.sqlite3"
    initial = SQLiteMetadataStore(db_path)
    await initial.initialize()
    initial._test_chat_memory_user_ids = []  # type: ignore[attr-defined]
    user, project, session = await _create_chat(initial)
    await initial.append_chat_messages_with_memory(
        [_message(user.id, project.id, session.id, "legacy identity")],
        config_fingerprint=_FINGERPRINT,
    )
    await initial.close()

    graph_columns = {
        "active_graph_store_fingerprint",
        "desired_graph_store_fingerprint",
        "graph_store_fingerprint",
    }
    with sqlite3.connect(db_path) as conn:
        for table in (
            "enterprise_chat_memory_groups",
            "enterprise_chat_memory_generations",
            "enterprise_chat_memory_outbox",
        ):
            legacy_columns = [
                row[1]
                for row in conn.execute(f"PRAGMA table_info({table})")
                if row[1] not in graph_columns
            ]
            projection = ", ".join(legacy_columns)
            conn.execute(
                f"CREATE TABLE {table}_legacy AS SELECT {projection} FROM {table}"
            )
            conn.execute(f"DROP TABLE {table}")
            conn.execute(f"ALTER TABLE {table}_legacy RENAME TO {table}")
        conn.execute("DELETE FROM metadata_schema WHERE version = 11")

    migrated = SQLiteMetadataStore(db_path)
    await migrated.initialize()
    await migrated.initialize()
    try:
        group = await migrated.get_chat_memory_group(user.id, project.id)
        generation = await migrated.get_chat_memory_generation(
            user.id, project.id, 1
        )
        events = await migrated.list_chat_memory_events(
            user_id=user.id, project_id=project.id
        )
        assert group is not None
        assert generation is not None
        assert group.active_graph_store_fingerprint is None
        assert group.desired_graph_store_fingerprint == _FINGERPRINT
        assert generation.graph_store_fingerprint == _FINGERPRINT
        assert len(events) == 1
        assert events[0].graph_store_fingerprint == _FINGERPRINT
        with sqlite3.connect(db_path) as conn:
            versions = {
                row[0] for row in conn.execute("SELECT version FROM metadata_schema")
            }
        assert 11 in versions
    finally:
        await migrated.close()
