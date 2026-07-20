"""Phase 2a execution primitive contracts for enterprise Chat Memory."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from lightrag.api.metadata_store import SQLiteMetadataStore
from tests.api.test_chat_memory_store_phase1 import (
    _FINGERPRINT,
    _GRAPH_FINGERPRINT,
    _POSTGRES_DSN,
    _create_chat,
    _make_store,
    _message,
)

pytestmark = pytest.mark.offline


@pytest.fixture(params=["sqlite", "postgres"])
async def execution_store(request, tmp_path):
    backend = request.param
    if backend == "postgres" and not _POSTGRES_DSN:
        pytest.skip(
            "live PostgreSQL Chat Memory execution contract skipped: set "
            "LIGHTRAG_KB_POSTGRES_TEST_DSN to enable"
        )
    store = await _make_store(backend, tmp_path)
    try:
        yield store
    finally:
        if backend == "postgres":
            user_ids = store._test_chat_memory_user_ids  # type: ignore[attr-defined]
            if user_ids:
                async with store._pool_or_raise().acquire() as conn:
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
        await store.close()


async def _append_one(store, content: str = "memory"):
    user, project, session = await _create_chat(store)
    saved = await store.append_chat_messages_with_memory(
        [_message(user.id, project.id, session.id, content)],
        config_fingerprint=_FINGERPRINT,
    )
    return user, project, session, saved


async def _force_terminal_purge(store, user_id: str, project_id: str) -> None:
    if isinstance(store, SQLiteMetadataStore):
        def sqlite_write(conn):
            now = "2026-07-15T00:00:00+00:00"
            conn.execute(
                """
                UPDATE enterprise_chat_memory_groups
                SET state = 'deleted', deleted_at = ?, updated_at = ?,
                    active_generation = NULL,
                    active_config_fingerprint = NULL
                WHERE user_id = ? AND project_id = ?
                """,
                (now, now, user_id, project_id),
            )
            conn.execute(
                """
                UPDATE enterprise_chat_memory_generations
                SET state = 'purged', cleared_at = ?, updated_at = ?
                WHERE user_id = ? AND project_id = ?
                """,
                (now, now, user_id, project_id),
            )
            conn.execute(
                """
                UPDATE enterprise_chat_memory_outbox
                SET status = 'succeeded', completed_at = ?, updated_at = ?,
                    claim_token = NULL, claimed_by = NULL, claimed_at = NULL,
                    side_effect_started_at = NULL,
                    side_effect_state_version = NULL
                WHERE user_id = ? AND project_id = ? AND event_type = 'purge'
                """,
                (now, now, user_id, project_id),
            )
            conn.execute(
                """
                DELETE FROM enterprise_chat_memory_episodes
                WHERE user_id = ? AND project_id = ?
                """,
                (user_id, project_id),
            )

        await store._write(sqlite_write)
        return

    async def postgres_write(conn):
        await conn.execute(
            """
            UPDATE enterprise_chat_memory_groups
            SET state = 'deleted', deleted_at = clock_timestamp(),
                updated_at = clock_timestamp(), active_generation = NULL,
                active_config_fingerprint = NULL
            WHERE user_id = $1 AND project_id = $2
            """,
            user_id,
            project_id,
        )
        await conn.execute(
            """
            UPDATE enterprise_chat_memory_generations
            SET state = 'purged', cleared_at = clock_timestamp(),
                updated_at = clock_timestamp()
            WHERE user_id = $1 AND project_id = $2
            """,
            user_id,
            project_id,
        )
        await conn.execute(
            """
            UPDATE enterprise_chat_memory_outbox
            SET status = 'succeeded', completed_at = clock_timestamp(),
                updated_at = clock_timestamp(), claim_token = NULL,
                claimed_by = NULL, claimed_at = NULL,
                side_effect_started_at = NULL,
                side_effect_state_version = NULL
            WHERE user_id = $1 AND project_id = $2 AND event_type = 'purge'
            """,
            user_id,
            project_id,
        )
        await conn.execute(
            """
            DELETE FROM enterprise_chat_memory_episodes
            WHERE user_id = $1 AND project_id = $2
            """,
            user_id,
            project_id,
        )

    await store._write(postgres_write)


async def _defensive_enqueue_purge(store, user_id: str, project_id: str):
    if isinstance(store, SQLiteMetadataStore):
        return await store._write(
            lambda conn: store._enqueue_sqlite_chat_memory_purge(
                conn,
                user_id,
                project_id,
                _FINGERPRINT,
                actor_user_id=None,
                actor_tenant_id=None,
            )
        )

    async def write(conn):
        return await store._enqueue_postgres_chat_memory_purge(
            conn,
            user_id,
            project_id,
            _FINGERPRINT,
            actor_user_id=None,
            actor_tenant_id=None,
        )

    return await store._write(write)


async def test_claim_enforces_same_group_head_of_line_and_fingerprint(execution_store):
    store = execution_store
    user, project, session, _saved = await _append_one(store, "first")
    await store.append_chat_messages_with_memory(
        [_message(user.id, project.id, session.id, "second")],
        config_fingerprint=_FINGERPRINT,
    )

    assert await store.claim_next_chat_memory_event("wrong-fingerprint") is None
    first = await store.claim_next_chat_memory_event(
        _FINGERPRINT, worker_id="worker-a"
    )
    assert first is not None
    assert first.event_seq == 1
    assert first.status == "running"
    assert first.attempt_no == 1
    assert first.claim_token
    assert first.claimed_by == "worker-a"
    execution_state = await store.get_chat_memory_execution_state(first.event_id)
    assert execution_state is not None
    assert execution_state.event.claim_token == first.claim_token
    assert execution_state.group.desired_generation == first.generation
    assert execution_state.generation.graph_group_id == first.graph_group_id
    assert await store.claim_next_chat_memory_event(_FINGERPRINT) is None

    await store.mark_chat_memory_event_side_effect_started(
        first.event_id, first.claim_token, _FINGERPRINT
    )
    await store.finalize_chat_memory_ingest(
        first.event_id,
        first.claim_token,
        _FINGERPRINT,
        episode_uuid=f"episode_{uuid.uuid4().hex}",
    )
    second = await store.claim_next_chat_memory_event(_FINGERPRINT)
    assert second is not None and second.event_seq == 2


async def test_purge_claim_ignores_extraction_upgrade_but_requires_graph_identity(
    execution_store,
):
    store = execution_store
    old_extraction = "chat-memory-extraction:v1:old"
    new_extraction = "chat-memory-extraction:v2:new"
    wrong_graph = "chat-memory-graph-store:v1:wrong"
    user, project, session = await _create_chat(store)
    await store.append_chat_messages_with_memory(
        [_message(user.id, project.id, session.id, "purge")],
        config_fingerprint=old_extraction,
        graph_store_fingerprint=_GRAPH_FINGERPRINT,
    )
    queued = await store.enqueue_chat_memory_purge(
        user.id,
        project.id,
        new_extraction,
        graph_store_fingerprint=wrong_graph,
    )
    assert queued is not None
    assert queued.config_fingerprint == new_extraction
    assert queued.graph_store_fingerprint == _GRAPH_FINGERPRINT

    assert (
        await store.claim_next_chat_memory_event(
            new_extraction,
            runtime_graph_store_fingerprint=wrong_graph,
            event_types=["purge"],
        )
        is None
    )
    purge = await store.claim_next_chat_memory_event(
        new_extraction,
        runtime_graph_store_fingerprint=_GRAPH_FINGERPRINT,
        event_types=["purge"],
    )
    assert purge is not None and purge.claim_token
    assert purge.config_fingerprint == new_extraction
    assert purge.graph_store_fingerprint == _GRAPH_FINGERPRINT
    started = await store.mark_chat_memory_event_side_effect_started(
        purge.event_id,
        purge.claim_token,
        new_extraction,
        runtime_graph_store_fingerprint=_GRAPH_FINGERPRINT,
    )
    assert started.side_effect_started_at is not None
    retried = await store.recover_stale_chat_memory_event(
        purge.event_id,
        purge.claim_token,
        new_extraction,
        runtime_graph_store_fingerprint=_GRAPH_FINGERPRINT,
        retry_delay_seconds=0,
    )
    assert retried is not None and retried.status == "retry_wait"
    second = await store.claim_next_chat_memory_event(
        new_extraction,
        runtime_graph_store_fingerprint=_GRAPH_FINGERPRINT,
        event_types=["purge"],
    )
    assert second is not None and second.claim_token
    dead = await store.fail_chat_memory_purge_before_side_effect(
        second.event_id,
        second.claim_token,
        new_extraction,
        runtime_graph_store_fingerprint=_GRAPH_FINGERPRINT,
        error_code="purge_retry_exhausted",
        error_message="dead letter for requeue",
        retry_delay_seconds=None,
        max_attempts=2,
    )
    assert dead.status == "dead_letter"
    requeued = await store.requeue_chat_memory_purge(
        dead.event_id,
        new_extraction,
        runtime_graph_store_fingerprint=_GRAPH_FINGERPRINT,
        retry_delay_seconds=0,
    )
    assert requeued.status == "retry_wait"
    assert requeued.config_fingerprint == new_extraction
    assert requeued.graph_store_fingerprint == _GRAPH_FINGERPRINT


async def test_old_extraction_worker_cannot_claim_new_ingest(execution_store):
    store = execution_store
    old_extraction = "chat-memory-extraction:v1:old"
    new_extraction = "chat-memory-extraction:v2:new"
    user, project, session = await _create_chat(store)
    await store.append_chat_messages_with_memory(
        [_message(user.id, project.id, session.id, "new extraction")],
        config_fingerprint=new_extraction,
        graph_store_fingerprint=_GRAPH_FINGERPRINT,
    )
    assert (
        await store.claim_next_chat_memory_event(
            old_extraction,
            runtime_graph_store_fingerprint=_GRAPH_FINGERPRINT,
            event_types=["ingest"],
        )
        is None
    )
    claimed = await store.claim_next_chat_memory_event(
        new_extraction,
        runtime_graph_store_fingerprint=_GRAPH_FINGERPRINT,
        event_types=["ingest"],
    )
    assert claimed is not None
    assert claimed.config_fingerprint == new_extraction
    assert claimed.graph_store_fingerprint == _GRAPH_FINGERPRINT


def test_postgres_claim_sql_contract_is_event_type_aware():
    import inspect

    from lightrag.api.postgres_metadata_store import PostgresMetadataStore

    source = inspect.getsource(PostgresMetadataStore.claim_next_chat_memory_event)
    assert "event.event_type = 'purge'" in source
    assert "event.graph_store_fingerprint = $2" in source
    assert "event.event_type IN ('ingest', 'rebuild')" in source
    assert "event.config_fingerprint = $1" in source


async def test_claim_rejects_invalid_event_type(execution_store):
    with pytest.raises(ValueError):
        await execution_store.claim_next_chat_memory_event(
            _FINGERPRINT, event_types=["not-a-real-event"]
        )


async def test_claims_different_groups_concurrently(execution_store):
    store = execution_store
    first_user, first_project, _session, _saved = await _append_one(store, "one")
    second_user, second_project, _session, _saved = await _append_one(store, "two")

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
        claimed = await asyncio.gather(
            store.claim_next_chat_memory_event(_FINGERPRINT, worker_id="left"),
            peer.claim_next_chat_memory_event(_FINGERPRINT, worker_id="right"),
        )
    finally:
        await peer.close()

    assert all(event is not None for event in claimed)
    assert {
        (event.user_id, event.project_id) for event in claimed if event is not None
    } == {
        (first_user.id, first_project.id),
        (second_user.id, second_project.id),
    }


async def test_concurrent_claims_do_not_skip_same_group_head(execution_store):
    store = execution_store
    user, project, session, _saved = await _append_one(store, "head")
    await store.append_chat_messages_with_memory(
        [_message(user.id, project.id, session.id, "tail")],
        config_fingerprint=_FINGERPRINT,
    )
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
        claims = await asyncio.gather(
            store.claim_next_chat_memory_event(_FINGERPRINT),
            peer.claim_next_chat_memory_event(_FINGERPRINT),
        )
    finally:
        await peer.close()
    claimed = [event for event in claims if event is not None]
    assert len(claimed) == 1
    assert claimed[0].event_seq == 1


async def test_atomic_ingest_finalize_activates_first_generation(execution_store):
    store = execution_store
    user, project, session, saved = await _append_one(store)
    claimed = await store.claim_next_chat_memory_event(_FINGERPRINT)
    assert claimed is not None and claimed.claim_token
    await store.mark_chat_memory_event_side_effect_started(
        claimed.event_id, claimed.claim_token, _FINGERPRINT
    )
    with pytest.raises(Exception):
        await store.finalize_chat_memory_ingest(
            claimed.event_id,
            claimed.claim_token,
            "wrong-fingerprint",
            episode_uuid="wrong-fingerprint-episode",
        )
    final = await store.finalize_chat_memory_ingest(
        claimed.event_id,
        claimed.claim_token,
        _FINGERPRINT,
        episode_uuid="episode-first-activation",
    )
    assert final.event.status == "succeeded"
    assert final.group.state == "active"
    assert final.group.active_generation == 1
    assert final.group.active_config_fingerprint == _FINGERPRINT
    assert final.generation.state == "active"
    mappings = await store.list_chat_memory_episodes_for_session(
        user.id, project.id, session.id
    )
    assert len(mappings) == 1
    mapping = mappings[0]
    assert mapping.episode_uuid == "episode-first-activation"
    assert mapping.event_id == claimed.event_id
    assert mapping.append_batch_id == saved[0].append_batch_id
    assert mapping.project_event_seq == claimed.event_seq
    with pytest.raises(Exception):
        await store.finalize_chat_memory_ingest(
            claimed.event_id,
            "stale-token",
            _FINGERPRINT,
            episode_uuid="different-episode",
        )
    assert len(
        await store.list_chat_memory_episodes_for_session(
            user.id, project.id, session.id
        )
    ) == 1


async def test_begin_rejects_runtime_mismatch_without_side_effect(execution_store):
    store = execution_store
    _user, _project, _session, _saved = await _append_one(store)
    claimed = await store.claim_next_chat_memory_event(_FINGERPRINT)
    assert claimed is not None and claimed.claim_token
    mismatched = await store.mark_chat_memory_event_side_effect_started(
        claimed.event_id,
        claimed.claim_token,
        "wrong-fingerprint",
        fingerprint_retry_delay_seconds=0,
    )
    assert mismatched.status == "retry_wait"
    assert mismatched.side_effect_started_at is None
    assert mismatched.side_effect_state_version is None
    reclaimed = await store.claim_next_chat_memory_event(_FINGERPRINT)
    assert reclaimed is not None and reclaimed.claim_token
    started = await store.mark_chat_memory_event_side_effect_started(
        reclaimed.event_id, reclaimed.claim_token, _FINGERPRINT
    )
    state = await store.get_chat_memory_execution_state(started.event_id)
    assert state is not None
    assert started.status == "running"
    assert started.side_effect_started_at is not None
    assert started.side_effect_state_version == state.group.state_version


async def test_claim_then_delete_supersedes_before_side_effect(execution_store):
    store = execution_store
    user, project, session, saved = await _append_one(store)
    claimed = await store.claim_next_chat_memory_event(_FINGERPRINT)
    assert claimed is not None and claimed.claim_token
    assert await store.delete_chat_message_with_memory(
        user.id,
        project.id,
        session.id,
        saved[0].id,
        config_fingerprint=_FINGERPRINT,
    )
    resolved = await store.mark_chat_memory_event_side_effect_started(
        claimed.event_id, claimed.claim_token, _FINGERPRINT
    )
    assert resolved.status == "superseded"
    assert resolved.side_effect_started_at is None
    group = await store.get_chat_memory_group(user.id, project.id)
    assert group is not None
    assert group.desired_generation == 2
    assert group.state == "rebuilding"


async def test_begin_then_delete_prevents_finalize_and_extra_escalation(execution_store):
    store = execution_store
    user, project, session, saved = await _append_one(store)
    claimed = await store.claim_next_chat_memory_event(_FINGERPRINT)
    assert claimed is not None and claimed.claim_token
    started = await store.mark_chat_memory_event_side_effect_started(
        claimed.event_id, claimed.claim_token, _FINGERPRINT
    )
    assert started.side_effect_state_version is not None
    assert await store.delete_chat_message_with_memory(
        user.id,
        project.id,
        session.id,
        saved[0].id,
        config_fingerprint=_FINGERPRINT,
    )
    finalized = await store.finalize_chat_memory_ingest(
        claimed.event_id,
        claimed.claim_token,
        _FINGERPRINT,
        episode_uuid="must-not-be-recorded",
    )
    assert finalized is None
    event = await store.get_chat_memory_event(claimed.event_id)
    assert event is not None and event.status == "superseded"
    assert (
        await store.list_chat_memory_episodes_for_session(
            user.id, project.id, session.id
        )
        == []
    )
    group = await store.get_chat_memory_group(user.id, project.id)
    assert group is not None and group.desired_generation == 2


async def test_begin_then_project_delete_recovery_preserves_deleting(execution_store):
    store = execution_store
    user, project, _session, _saved = await _append_one(store)
    claimed = await store.claim_next_chat_memory_event(_FINGERPRINT)
    assert claimed is not None and claimed.claim_token
    await store.mark_chat_memory_event_side_effect_started(
        claimed.event_id, claimed.claim_token, _FINGERPRINT
    )
    deleted, _sessions, _messages = await store.delete_chat_project_with_memory(
        user.id, project.id, config_fingerprint=_FINGERPRINT
    )
    assert deleted
    recovered = await store.recover_stale_chat_memory_event(
        claimed.event_id,
        claimed.claim_token,
        _FINGERPRINT,
        retry_delay_seconds=0,
    )
    assert recovered is not None and recovered.status == "superseded"
    group = await store.get_chat_memory_group(user.id, project.id)
    assert group is not None
    assert group.state == "deleting"
    assert group.desired_generation == 1
    inventory = await store.list_chat_memory_generations(user.id, project.id)
    assert [item.generation for item in inventory] == [1]
    assert inventory[0].state == "purge_pending"


async def test_known_failure_retries_then_dead_letters_and_blocks(execution_store):
    store = execution_store
    user, project, session, _saved = await _append_one(store, "head")
    await store.append_chat_messages_with_memory(
        [_message(user.id, project.id, session.id, "behind")],
        config_fingerprint=_FINGERPRINT,
    )
    first = await store.claim_next_chat_memory_event(_FINGERPRINT)
    assert first is not None and first.claim_token
    retry = await store.fail_chat_memory_event_before_side_effect(
        first.event_id,
        first.claim_token,
        _FINGERPRINT,
        error_code="temporary",
        error_message="retry",
        retry_delay_seconds=0,
        max_attempts=2,
    )
    assert retry.status == "retry_wait"
    second_attempt = await store.claim_next_chat_memory_event(_FINGERPRINT)
    assert second_attempt is not None
    assert second_attempt.event_id == first.event_id
    assert second_attempt.attempt_no == 2
    assert second_attempt.claim_token != first.claim_token
    with pytest.raises(Exception):
        await store.mark_chat_memory_event_side_effect_started(
            first.event_id, first.claim_token, _FINGERPRINT
        )
    dead = await store.fail_chat_memory_event_before_side_effect(
        second_attempt.event_id,
        second_attempt.claim_token,
        _FINGERPRINT,
        error_code="exhausted",
        error_message="dead",
        retry_delay_seconds=0,
        max_attempts=2,
    )
    assert dead.status == "dead_letter"
    group = await store.get_chat_memory_group(user.id, project.id)
    assert group is not None and group.state == "failed"
    assert await store.claim_next_chat_memory_event(_FINGERPRINT) is None
    stats = await store.get_chat_memory_outbox_stats()
    assert stats.dead_letter == 1
    assert stats.pending == 1

    rebuild = await store.supersede_chat_memory_dead_letter_with_rebuild(
        dead.event_id, _FINGERPRINT
    )
    assert rebuild.event_type == "rebuild"
    assert rebuild.generation == 2
    events = await store.list_chat_memory_events(
        user_id=user.id, project_id=project.id
    )
    assert [event.status for event in events] == [
        "superseded",
        "superseded",
        "pending",
    ]
    claimed_rebuild = await store.claim_next_chat_memory_event(_FINGERPRINT)
    assert claimed_rebuild is not None
    assert claimed_rebuild.event_id == rebuild.event_id


async def test_unknown_outcome_abandons_generation_and_enqueues_rebuild(execution_store):
    store = execution_store
    user, project, _session, _saved = await _append_one(store)
    claimed = await store.claim_next_chat_memory_event(_FINGERPRINT)
    assert claimed is not None and claimed.claim_token
    await store.mark_chat_memory_event_side_effect_started(
        claimed.event_id, claimed.claim_token, _FINGERPRINT
    )
    rebuild = await store.escalate_chat_memory_event_unknown(
        claimed.event_id,
        claimed.claim_token,
        _FINGERPRINT,
        error_message="connection lost after write",
    )
    assert rebuild.event_type == "rebuild"
    assert rebuild.generation == 2
    assert rebuild.graph_group_id != claimed.graph_group_id
    original = await store.get_chat_memory_event(claimed.event_id)
    assert original is not None
    assert original.status == "superseded"
    assert original.superseded_by_event_id == rebuild.event_id
    first_generation = await store.get_chat_memory_generation(user.id, project.id, 1)
    second_generation = await store.get_chat_memory_generation(user.id, project.id, 2)
    assert first_generation is not None and first_generation.state == "abandoned"
    assert second_generation is not None and second_generation.state == "building"
    group = await store.get_chat_memory_group(user.id, project.id)
    assert group is not None
    assert group.desired_generation == 2
    assert group.active_generation is None
    assert group.state == "rebuilding"
    assert group.active_rebuild_event_id == rebuild.event_id
    with pytest.raises(Exception):
        await store.finalize_chat_memory_ingest(
            claimed.event_id,
            claimed.claim_token,
            _FINGERPRINT,
            episode_uuid="stale-finalizer",
        )


async def test_stale_recovery_chooses_retry_or_unknown_escalation(execution_store):
    store = execution_store
    _user, _project, _session, _saved = await _append_one(store)
    first = await store.claim_next_chat_memory_event(_FINGERPRINT)
    assert first is not None and first.claim_token
    stale = await store.list_stale_chat_memory_running_events(
        stale_after_seconds=0
    )
    assert [event.event_id for event in stale] == [first.event_id]
    retried = await store.recover_stale_chat_memory_event(
        first.event_id,
        first.claim_token,
        _FINGERPRINT,
        retry_delay_seconds=0,
    )
    assert retried.status == "retry_wait"
    second = await store.claim_next_chat_memory_event(_FINGERPRINT)
    assert second is not None and second.claim_token
    await store.mark_chat_memory_event_side_effect_started(
        second.event_id, second.claim_token, _FINGERPRINT
    )
    escalated = await store.recover_stale_chat_memory_event(
        second.event_id,
        second.claim_token,
        _FINGERPRINT,
    )
    assert escalated.event_type == "rebuild"
    assert escalated.generation == 2


async def test_stale_recovery_requires_group_guard(execution_store):
    store = execution_store
    user, project, _session, _saved = await _append_one(store)
    claimed = await store.claim_next_chat_memory_event(_FINGERPRINT)
    assert claimed is not None and claimed.claim_token
    logical_group_id = (
        await store.get_chat_memory_group(user.id, project.id)
    ).logical_group_id
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
        async with peer.chat_memory_group_execution_guard(logical_group_id) as held:
            assert held
            assert (
                await store.recover_stale_chat_memory_event(
                    claimed.event_id,
                    claimed.claim_token,
                    _FINGERPRINT,
                    retry_delay_seconds=0,
                )
                is None
            )
            unchanged = await store.get_chat_memory_event(claimed.event_id)
            assert unchanged is not None and unchanged.status == "running"
        recovered = await store.recover_stale_chat_memory_event(
            claimed.event_id,
            claimed.claim_token,
            _FINGERPRINT,
            retry_delay_seconds=0,
        )
        assert recovered is not None and recovered.status == "retry_wait"
    finally:
        await peer.close()


async def test_purge_claim_failure_unknown_and_explicit_requeue(execution_store):
    store = execution_store
    user, project, _session, _saved = await _append_one(store)
    deleted, _sessions, _messages = await store.delete_chat_project_with_memory(
        user.id, project.id, config_fingerprint=_FINGERPRINT
    )
    assert deleted
    assert (
        await store.claim_next_chat_memory_event(
            _FINGERPRINT, event_types=["ingest", "rebuild"]
        )
        is None
    )
    purge = await store.claim_next_chat_memory_event(
        _FINGERPRINT, event_types=["purge"]
    )
    assert purge is not None and purge.event_type == "purge" and purge.claim_token
    started = await store.mark_chat_memory_event_side_effect_started(
        purge.event_id, purge.claim_token, _FINGERPRINT
    )
    assert started.side_effect_state_version is not None
    unknown = await store.recover_stale_chat_memory_event(
        purge.event_id,
        purge.claim_token,
        _FINGERPRINT,
        retry_delay_seconds=0,
    )
    assert unknown is not None and unknown.status == "retry_wait"
    assert unknown.side_effect_started_at is None
    assert unknown.side_effect_state_version is None
    group = await store.get_chat_memory_group(user.id, project.id)
    assert group is not None and group.state == "deleting"
    assert group.desired_generation == purge.generation
    generation = await store.get_chat_memory_generation(
        user.id, project.id, purge.generation
    )
    assert generation is not None and generation.state == "purge_pending"

    second = await store.claim_next_chat_memory_event(
        _FINGERPRINT, event_types=["purge"]
    )
    assert second is not None and second.event_id == purge.event_id
    dead = await store.fail_chat_memory_purge_before_side_effect(
        second.event_id,
        second.claim_token,
        _FINGERPRINT,
        error_code="clear_unavailable",
        error_message="clear failed before request",
        retry_delay_seconds=0,
        max_attempts=2,
    )
    assert dead.status == "dead_letter"
    group = await store.get_chat_memory_group(user.id, project.id)
    assert group is not None and group.state == "deleting"
    requeued = await store.requeue_chat_memory_purge(
        dead.event_id, _FINGERPRINT, retry_delay_seconds=0
    )
    assert requeued.status == "retry_wait"
    assert requeued.generation == purge.generation
    third = await store.claim_next_chat_memory_event(
        _FINGERPRINT, event_types=["purge"]
    )
    assert third is not None and third.event_id == purge.event_id
    assert third.generation == purge.generation


async def test_terminal_purge_history_does_not_repurge_on_user_delete(execution_store):
    store = execution_store
    user, project, _session, _saved = await _append_one(store)
    deleted, _sessions, _messages = await store.delete_chat_project_with_memory(
        user.id, project.id, config_fingerprint=_FINGERPRINT
    )
    assert deleted
    await _force_terminal_purge(store, user.id, project.id)
    before = await store.list_chat_memory_events(
        user_id=user.id, project_id=project.id
    )
    assert [event.status for event in before] == ["superseded", "succeeded"]

    assert await _defensive_enqueue_purge(store, user.id, project.id) is None
    after_defensive = await store.list_chat_memory_events(
        user_id=user.id, project_id=project.id
    )
    assert [event.event_id for event in after_defensive] == [
        event.event_id for event in before
    ]
    group = await store.get_chat_memory_group(user.id, project.id)
    assert group is not None and group.state == "deleted"

    assert await store.delete_enterprise_user_with_memory(
        user.id, config_fingerprint=_FINGERPRINT
    )
    after_user_delete = await store.list_chat_memory_events(
        user_id=user.id, project_id=project.id
    )
    assert [event.event_id for event in after_user_delete] == [
        event.event_id for event in before
    ]
    group = await store.get_chat_memory_group(user.id, project.id)
    assert group is not None and group.state == "deleted"
    generations = await store.list_chat_memory_generations(user.id, project.id)
    assert generations and all(item.state == "purged" for item in generations)


async def test_group_execution_guard_is_reentrant_and_mutually_exclusive(
    execution_store,
):
    store = execution_store
    logical_group_id = f"cm_guard_{uuid.uuid4().hex}"
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
        async with store.chat_memory_group_execution_guard(
            logical_group_id
        ) as acquired:
            assert acquired
            async with store.chat_memory_group_execution_guard(
                logical_group_id
            ) as nested:
                assert nested
            async with peer.chat_memory_group_execution_guard(
                logical_group_id, wait=False
            ) as competing:
                assert not competing
        async with peer.chat_memory_group_execution_guard(
            logical_group_id, wait=False
        ) as acquired_after_release:
            assert acquired_after_release
    finally:
        await peer.close()
