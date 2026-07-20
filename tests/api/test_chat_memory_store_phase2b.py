"""Phase 2b rebuild/purge store contracts for enterprise Chat Memory."""

from __future__ import annotations

import asyncio
import copy
import sqlite3
import uuid

import pytest

from lightrag.api.kb_service import utc_now_iso
from lightrag.api.metadata_store import (
    ChatMemoryEpisodeRecord,
    ChatMemoryReplayMappingInput,
    MetadataConflictError,
    SQLiteMetadataStore,
    chat_memory_graph_group_id,
    chat_memory_legacy_graph_group_id,
)
from tests.api.test_chat_memory_store_phase1 import (
    _FINGERPRINT,
    _GRAPH_FINGERPRINT,
    _POSTGRES_DSN,
    _create_chat,
    _make_store,
    _message,
)
from tests.api.test_chat_memory_store_phase2a import _append_one

pytestmark = pytest.mark.offline


@pytest.fixture(params=["sqlite", "postgres"])
async def phase2b_store(request, tmp_path):
    backend = request.param
    if backend == "postgres" and not _POSTGRES_DSN:
        pytest.skip(
            "live PostgreSQL Chat Memory phase 2b contract skipped: set "
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


def _snapshot_mappings(snapshot, prefix: str):
    mappings: list[ChatMemoryReplayMappingInput] = []
    for index, batch in enumerate(snapshot.replay_batches):
        mappings.append(
            ChatMemoryReplayMappingInput(
                append_batch_id=batch.append_batch_id,
                project_event_seq=batch.project_event_seq,
                session_id=batch.session_id,
                first_seq=min(message.seq for message in batch.messages),
                last_seq=max(message.seq for message in batch.messages),
                episode_uuid=f"{prefix}_{index}_{uuid.uuid4().hex[:12]}",
            )
        )
    return mappings


async def _activate_first(store, content: str = "first"):
    user, project, session, saved = await _append_one(store, content)
    event = await store.claim_next_chat_memory_event(
        _FINGERPRINT, event_types=["ingest"]
    )
    assert event is not None and event.claim_token
    await store.mark_chat_memory_event_side_effect_started(
        event.event_id, event.claim_token, _FINGERPRINT
    )
    final = await store.finalize_chat_memory_ingest(
        event.event_id,
        event.claim_token,
        _FINGERPRINT,
        episode_uuid=f"phase2b-first-{uuid.uuid4().hex}",
    )
    assert final is not None
    return user, project, session, saved, final


async def _prepare_rebuild_targets(store, event):
    assert event.claim_token
    targets = await store.prepare_chat_memory_rebuild_targets(
        event.event_id,
        event.claim_token,
        _FINGERPRINT,
    )
    assert targets is not None
    return targets


def _memory_message(
    user_id: str,
    project_id: str,
    session_id: str,
    content: str,
    *,
    role: str = "user",
    memory_eligible: bool | None = None,
):
    message = _message(user_id, project_id, session_id, content)
    message.role = role
    if memory_eligible is not None:
        message.metadata["memory_eligible"] = memory_eligible
    return message


async def _inject_defensive_purge_inventory(
    store,
    user_id: str,
    project_id: str,
) -> tuple[str, str]:
    mapping_group = f"orphan-mapping-{uuid.uuid4().hex}"
    outbox_group = f"orphan-outbox-{uuid.uuid4().hex}"
    if isinstance(store, SQLiteMetadataStore):

        def sqlite_write(conn):
            conn.execute(
                """
                UPDATE enterprise_chat_memory_generations
                SET state = 'purged'
                WHERE user_id = ? AND project_id = ? AND generation = 1
                """,
                (user_id, project_id),
            )
            conn.execute(
                """
                UPDATE enterprise_chat_memory_episodes
                SET graph_group_id = ?
                WHERE episode_uuid = (
                    SELECT episode_uuid FROM enterprise_chat_memory_episodes
                    WHERE user_id = ? AND project_id = ?
                    ORDER BY created_at, episode_uuid LIMIT 1
                )
                """,
                (mapping_group, user_id, project_id),
            )
            conn.execute(
                """
                UPDATE enterprise_chat_memory_outbox
                SET graph_group_id = ?
                WHERE event_id = (
                    SELECT event_id FROM enterprise_chat_memory_outbox
                    WHERE user_id = ? AND project_id = ?
                      AND event_type <> 'purge'
                    ORDER BY event_seq LIMIT 1
                )
                """,
                (outbox_group, user_id, project_id),
            )

        await store._write(sqlite_write)
        return mapping_group, outbox_group

    async def postgres_write(conn):
        await conn.execute(
            """
            UPDATE enterprise_chat_memory_generations
            SET state = 'purged'
            WHERE user_id = $1 AND project_id = $2 AND generation = 1
            """,
            user_id,
            project_id,
        )
        await conn.execute(
            """
            UPDATE enterprise_chat_memory_episodes
            SET graph_group_id = $1
            WHERE episode_uuid = (
                SELECT episode_uuid FROM enterprise_chat_memory_episodes
                WHERE user_id = $2 AND project_id = $3
                ORDER BY created_at, episode_uuid LIMIT 1
            )
            """,
            mapping_group,
            user_id,
            project_id,
        )
        await conn.execute(
            """
            UPDATE enterprise_chat_memory_outbox
            SET graph_group_id = $1
            WHERE event_id = (
                SELECT event_id FROM enterprise_chat_memory_outbox
                WHERE user_id = $2 AND project_id = $3
                  AND event_type <> 'purge'
                ORDER BY event_seq LIMIT 1
            )
            """,
            outbox_group,
            user_id,
            project_id,
        )

    await store._write(postgres_write)
    return mapping_group, outbox_group


async def _inject_mixed_graph_store_inventory(
    store,
    user_id: str,
    project_id: str,
    generation: int,
    graph_store_fingerprint: str,
) -> None:
    if isinstance(store, SQLiteMetadataStore):

        def sqlite_write(conn):
            conn.execute(
                """
                UPDATE enterprise_chat_memory_generations
                SET graph_store_fingerprint = ?
                WHERE user_id = ? AND project_id = ? AND generation = ?
                """,
                (
                    graph_store_fingerprint,
                    user_id,
                    project_id,
                    generation,
                ),
            )

        await store._write(sqlite_write)
        return

    async def postgres_write(conn):
        await conn.execute(
            """
            UPDATE enterprise_chat_memory_generations
            SET graph_store_fingerprint = $1
            WHERE user_id = $2 AND project_id = $3 AND generation = $4
            """,
            graph_store_fingerprint,
            user_id,
            project_id,
            generation,
        )

    await store._write(postgres_write)


async def _set_generation_state(
    store,
    user_id: str,
    project_id: str,
    generation: int,
    state: str,
) -> None:
    if isinstance(store, SQLiteMetadataStore):

        def sqlite_write(conn):
            conn.execute(
                """
                UPDATE enterprise_chat_memory_generations
                SET state = ?, updated_at = ?
                WHERE user_id = ? AND project_id = ? AND generation = ?
                """,
                (
                    state,
                    utc_now_iso(),
                    user_id,
                    project_id,
                    generation,
                ),
            )

        await store._write(sqlite_write)
        return

    async def postgres_write(conn):
        await conn.execute(
            """
            UPDATE enterprise_chat_memory_generations
            SET state = $1, updated_at = clock_timestamp()
            WHERE user_id = $2 AND project_id = $3 AND generation = $4
            """,
            state,
            user_id,
            project_id,
            generation,
        )

    await store._write(postgres_write)


async def _complete_admin_rebuild(store, user_id: str, project_id: str, prefix: str):
    queued = await store.enqueue_chat_memory_rebuild(
        user_id, project_id, _FINGERPRINT
    )
    assert queued is not None
    claimed = await store.claim_next_chat_memory_event(
        _FINGERPRINT, event_types=["rebuild"]
    )
    assert claimed is not None and claimed.event_id == queued.event_id
    assert claimed.claim_token
    snapshot = await store.prepare_chat_memory_rebuild_snapshot(
        claimed.event_id,
        claimed.claim_token,
        _FINGERPRINT,
        max_messages=1000,
        max_bytes=1_000_000,
    )
    assert snapshot is not None
    targets = await _prepare_rebuild_targets(store, claimed)
    await store.mark_chat_memory_event_side_effect_started(
        claimed.event_id, claimed.claim_token, _FINGERPRINT
    )
    final = await store.finalize_chat_memory_rebuild(
        claimed.event_id,
        claimed.claim_token,
        _FINGERPRINT,
        snapshot,
        _snapshot_mappings(snapshot, prefix),
        targets,
        targets.group_ids,
    )
    assert final is not None
    return claimed, snapshot, final


async def test_noop_ingest_atomically_activates_first_generation(phase2b_store):
    store = phase2b_store
    user, project, session = await _create_chat(store)
    saved = await store.append_chat_messages_with_memory(
        [_memory_message(user.id, project.id, session.id, "   ")],
        config_fingerprint=_FINGERPRINT,
    )
    event = await store.claim_next_chat_memory_event(
        _FINGERPRINT, event_types=["ingest"]
    )
    assert event is not None and event.claim_token
    final = await store.finalize_chat_memory_ingest_noop(
        event.event_id, event.claim_token, _FINGERPRINT
    )
    assert final is not None

    assert final.event.status == "succeeded"
    assert final.event.side_effect_started_at is None
    assert final.event.side_effect_state_version is None
    assert final.group.state == "active"
    assert final.group.active_generation == 1
    assert final.generation.state == "active"
    mappings = await store.list_chat_memory_episodes_for_session(
        user.id, project.id, session.id
    )
    assert len(mappings) == 1
    assert mappings[0].episode_uuid.startswith("noop_")
    assert mappings[0].generation == 1
    assert mappings[0].append_batch_id == saved[0].append_batch_id


@pytest.mark.parametrize(
    ("role", "content", "memory_eligible"),
    [
        ("user", "   ", None),
        ("system", "unsupported", None),
        ("assistant", "not explicitly eligible", None),
        ("assistant", "explicitly false", False),
        ("assistant", "   ", True),
    ],
)
async def test_noop_ingest_accepts_only_empty_canonical_payload(
    phase2b_store,
    role,
    content,
    memory_eligible,
):
    store = phase2b_store
    user, project, session = await _create_chat(store)
    saved = await store.append_chat_messages_with_memory(
        [
            _memory_message(
                user.id,
                project.id,
                session.id,
                content,
                role=role,
                memory_eligible=memory_eligible,
            )
        ],
        config_fingerprint=_FINGERPRINT,
    )
    event = await store.claim_next_chat_memory_event(
        _FINGERPRINT, event_types=["ingest"]
    )
    assert event is not None and event.claim_token
    final = await store.finalize_chat_memory_ingest_noop(
        event.event_id, event.claim_token, _FINGERPRINT
    )
    assert final is not None and final.event.status == "succeeded"
    mappings = await store.list_chat_memory_episodes_for_session(
        user.id, project.id, session.id
    )
    assert len(mappings) == 1
    assert mappings[0].append_batch_id == saved[0].append_batch_id
    assert mappings[0].episode_uuid.startswith("noop_")


@pytest.mark.parametrize(
    ("role", "memory_eligible"),
    [("user", None), ("assistant", True)],
)
async def test_noop_ingest_rejects_eligible_payload_and_zero_facts_finalize_normally(
    phase2b_store,
    role,
    memory_eligible,
):
    store = phase2b_store
    user, project, session = await _create_chat(store)
    await store.append_chat_messages_with_memory(
        [
            _memory_message(
                user.id,
                project.id,
                session.id,
                "eligible payload",
                role=role,
                memory_eligible=memory_eligible,
            )
        ],
        config_fingerprint=_FINGERPRINT,
    )
    event = await store.claim_next_chat_memory_event(
        _FINGERPRINT, event_types=["ingest"]
    )
    assert event is not None and event.claim_token
    with pytest.raises(MetadataConflictError):
        await store.finalize_chat_memory_ingest_noop(
            event.event_id, event.claim_token, _FINGERPRINT
        )
    assert (
        await store.list_chat_memory_episodes_for_session(
            user.id, project.id, session.id
        )
        == []
    )

    await store.mark_chat_memory_event_side_effect_started(
        event.event_id, event.claim_token, _FINGERPRINT
    )
    final = await store.finalize_chat_memory_ingest(
        event.event_id,
        event.claim_token,
        _FINGERPRINT,
        episode_uuid=f"zero-facts-{uuid.uuid4().hex}",
    )
    assert final is not None and final.event.status == "succeeded"


async def test_noop_ingest_stale_and_mapping_conflict_fail_closed(phase2b_store):
    store = phase2b_store

    stale_user, stale_project, stale_session = await _create_chat(store)
    stale_saved = await store.append_chat_messages_with_memory(
        [_memory_message(stale_user.id, stale_project.id, stale_session.id, " ")],
        config_fingerprint=_FINGERPRINT,
    )
    stale_event = await store.claim_next_chat_memory_event(
        _FINGERPRINT, event_types=["ingest"]
    )
    assert stale_event is not None and stale_event.claim_token
    assert await store.delete_chat_message_with_memory(
        stale_user.id,
        stale_project.id,
        stale_session.id,
        stale_saved[0].id,
        config_fingerprint=_FINGERPRINT,
    )
    assert (
        await store.finalize_chat_memory_ingest_noop(
            stale_event.event_id, stale_event.claim_token, _FINGERPRINT
        )
        is None
    )
    stale_after = await store.get_chat_memory_event(stale_event.event_id)
    stale_group = await store.get_chat_memory_group(
        stale_user.id, stale_project.id
    )
    assert stale_after is not None and stale_after.status == "superseded"
    assert stale_group is not None and stale_group.active_generation is None

    user, project, session = await _create_chat(store)
    saved = await store.append_chat_messages_with_memory(
        [_memory_message(user.id, project.id, session.id, " ")],
        config_fingerprint=_FINGERPRINT,
    )
    event = await store.claim_next_chat_memory_event(
        _FINGERPRINT, event_types=["ingest"]
    )
    assert event is not None and event.claim_token
    await store.record_chat_memory_episode(
        ChatMemoryEpisodeRecord(
            episode_uuid=f"noop-conflict-{uuid.uuid4().hex}",
            session_id=session.id,
            project_id=project.id,
            user_id=user.id,
            first_seq=saved[0].seq,
            last_seq=saved[0].seq,
            created_at=utc_now_iso(),
            event_id=event.event_id,
            generation=event.generation,
            graph_group_id=event.graph_group_id,
            append_batch_id=event.append_batch_id,
            project_event_seq=event.event_seq,
        )
    )
    with pytest.raises(MetadataConflictError):
        await store.finalize_chat_memory_ingest_noop(
            event.event_id, event.claim_token, _FINGERPRINT
        )
    unchanged_event = await store.get_chat_memory_event(event.event_id)
    unchanged_group = await store.get_chat_memory_group(user.id, project.id)
    unchanged_generation = await store.get_chat_memory_generation(
        user.id, project.id, event.generation
    )
    assert unchanged_event is not None and unchanged_event.status == "running"
    assert unchanged_group is not None and unchanged_group.active_generation is None
    assert unchanged_generation is not None
    assert unchanged_generation.state == "building"


async def test_rebuild_snapshot_has_fixed_cutoff_and_continuous_append_survives(
    phase2b_store,
):
    store = phase2b_store
    user, project, session, _saved, _final = await _activate_first(store, "é")
    queued = await store.enqueue_chat_memory_rebuild(
        user.id, project.id, _FINGERPRINT
    )
    assert queued is not None
    claimed = await store.claim_next_chat_memory_event(
        _FINGERPRINT, event_types=["rebuild"]
    )
    assert claimed is not None and claimed.claim_token
    appended_before_cutoff = await store.append_chat_messages_with_memory(
        [_message(user.id, project.id, session.id, "before H")],
        config_fingerprint=_FINGERPRINT,
    )
    snapshot = await store.prepare_chat_memory_rebuild_snapshot(
        claimed.event_id,
        claimed.claim_token,
        _FINGERPRINT,
        max_messages=100,
        max_bytes=1000,
    )
    assert snapshot is not None
    assert snapshot.snapshot_cutoff == appended_before_cutoff[0].project_event_seq
    assert [batch.project_event_seq for batch in snapshot.replay_batches] == [
        1,
        appended_before_cutoff[0].project_event_seq,
    ]
    assert snapshot.message_count == 2
    assert snapshot.byte_count == len("ébefore H".encode())
    persisted_event = await store.get_chat_memory_event(claimed.event_id)
    persisted_generation = await store.get_chat_memory_generation(
        user.id, project.id, claimed.generation
    )
    assert persisted_event is not None
    assert snapshot.snapshot_digest is not None
    assert snapshot.snapshot_digest.startswith("chat-memory-snapshot:v1:sha256:")
    assert (
        persisted_event.snapshot_cutoff,
        persisted_event.snapshot_batch_count,
        persisted_event.snapshot_message_count,
        persisted_event.snapshot_byte_count,
        persisted_event.snapshot_digest,
    ) == (
        snapshot.snapshot_cutoff,
        snapshot.batch_count,
        snapshot.message_count,
        snapshot.byte_count,
        snapshot.snapshot_digest,
    )
    assert persisted_generation is not None
    assert (
        persisted_generation.snapshot_cutoff,
        persisted_generation.replay_batch_count,
        persisted_generation.replay_message_count,
        persisted_generation.replay_byte_count,
        persisted_generation.snapshot_digest,
    ) == (
        snapshot.snapshot_cutoff,
        snapshot.batch_count,
        snapshot.message_count,
        snapshot.byte_count,
        snapshot.snapshot_digest,
    )

    appended_after_cutoff = await store.append_chat_messages_with_memory(
        [_message(user.id, project.id, session.id, "after H")],
        config_fingerprint=_FINGERPRINT,
    )
    assert appended_after_cutoff[0].project_event_seq == snapshot.snapshot_cutoff + 1
    retried_snapshot = await store.prepare_chat_memory_rebuild_snapshot(
        claimed.event_id,
        claimed.claim_token,
        _FINGERPRINT,
        max_messages=100,
        max_bytes=1000,
    )
    assert retried_snapshot is not None
    assert retried_snapshot.snapshot_cutoff == snapshot.snapshot_cutoff
    assert retried_snapshot.snapshot_digest == snapshot.snapshot_digest
    assert [
        batch.project_event_seq for batch in retried_snapshot.replay_batches
    ] == [batch.project_event_seq for batch in snapshot.replay_batches]
    assert appended_after_cutoff[0].project_event_seq not in {
        batch.project_event_seq for batch in retried_snapshot.replay_batches
    }
    targets = await _prepare_rebuild_targets(store, claimed)
    await store.mark_chat_memory_event_side_effect_started(
        claimed.event_id, claimed.claim_token, _FINGERPRINT
    )
    final = await store.finalize_chat_memory_rebuild(
        claimed.event_id,
        claimed.claim_token,
        _FINGERPRINT,
        snapshot,
        _snapshot_mappings(snapshot, "fixed_h"),
        targets,
        targets.group_ids,
    )
    assert final is not None
    assert final.group.active_generation == 2
    assert final.generation.state == "active"
    inventory = await store.list_chat_memory_generations(user.id, project.id)
    assert [(item.generation, item.state) for item in inventory] == [
        (1, "purged"),
        (2, "active"),
    ]
    assert inventory[0].cleared_at is not None
    events = await store.list_chat_memory_events(
        user_id=user.id, project_id=project.id
    )
    before_event = next(
        event
        for event in events
        if event.event_seq == appended_before_cutoff[0].project_event_seq
    )
    after_event = next(
        event
        for event in events
        if event.event_seq == appended_after_cutoff[0].project_event_seq
    )
    assert before_event.status == "superseded"
    assert before_event.superseded_by_event_id == claimed.event_id
    assert after_event.status == "pending"
    assert after_event.event_seq > snapshot.snapshot_cutoff


async def test_first_rebuild_targets_target_and_legacy_without_old_generation(
    phase2b_store,
):
    store = phase2b_store
    user, project, _session = await _create_chat(store)
    queued = await store.enqueue_chat_memory_rebuild(
        user.id, project.id, _FINGERPRINT
    )
    assert queued is not None and queued.generation == 1
    claimed = await store.claim_next_chat_memory_event(
        _FINGERPRINT, event_types=["rebuild"]
    )
    assert claimed is not None and claimed.claim_token
    snapshot = await store.prepare_chat_memory_rebuild_snapshot(
        claimed.event_id,
        claimed.claim_token,
        _FINGERPRINT,
        max_messages=100,
        max_bytes=1000,
    )
    assert snapshot is not None and snapshot.replay_batches == []
    targets = await _prepare_rebuild_targets(store, claimed)
    assert targets.event_id == claimed.event_id
    assert targets.user_id == user.id
    assert targets.project_id == project.id
    assert targets.group_ids == tuple(
        sorted(
            {
                claimed.graph_group_id,
                chat_memory_legacy_graph_group_id(user.id, project.id),
            }
        )
    )

    await store.mark_chat_memory_event_side_effect_started(
        claimed.event_id, claimed.claim_token, _FINGERPRINT
    )
    final = await store.finalize_chat_memory_rebuild(
        claimed.event_id,
        claimed.claim_token,
        _FINGERPRINT,
        snapshot,
        [],
        targets,
        targets.group_ids,
    )
    assert final is not None
    assert final.group.active_generation == 1
    assert final.generation.state == "active"


@pytest.mark.parametrize("delete_scope", ["message", "session"])
async def test_delete_rebuild_purges_old_active_generation(
    phase2b_store,
    delete_scope,
):
    store = phase2b_store
    user, project, session, saved, _final = await _activate_first(store)
    old_group_id = chat_memory_graph_group_id(user.id, project.id, 1)
    if delete_scope == "message":
        assert await store.delete_chat_message_with_memory(
            user.id,
            project.id,
            session.id,
            saved[0].id,
            config_fingerprint=_FINGERPRINT,
        )
    else:
        assert await store.delete_chat_session_with_memory(
            user.id,
            project.id,
            session.id,
            config_fingerprint=_FINGERPRINT,
        ) == (True, 1)

    claimed = await store.claim_next_chat_memory_event(
        _FINGERPRINT, event_types=["rebuild"]
    )
    assert claimed is not None and claimed.claim_token
    snapshot = await store.prepare_chat_memory_rebuild_snapshot(
        claimed.event_id,
        claimed.claim_token,
        _FINGERPRINT,
        max_messages=100,
        max_bytes=1000,
    )
    assert snapshot is not None
    targets = await _prepare_rebuild_targets(store, claimed)
    assert {
        old_group_id,
        claimed.graph_group_id,
        chat_memory_legacy_graph_group_id(user.id, project.id),
    }.issubset(targets.group_ids)
    await store.mark_chat_memory_event_side_effect_started(
        claimed.event_id, claimed.claim_token, _FINGERPRINT
    )
    final = await store.finalize_chat_memory_rebuild(
        claimed.event_id,
        claimed.claim_token,
        _FINGERPRINT,
        snapshot,
        _snapshot_mappings(snapshot, f"delete-{delete_scope}"),
        targets,
        targets.group_ids,
    )
    assert final is not None and final.group.active_generation == 2

    inventory = await store.list_chat_memory_generations(user.id, project.id)
    assert [(item.generation, item.state) for item in inventory] == [
        (1, "purged"),
        (2, "active"),
    ]
    assert inventory[0].cleared_at is not None


async def test_rebuild_finalize_missing_old_clear_is_fully_atomic(phase2b_store):
    store = phase2b_store
    user, project, session, _saved, _final = await _activate_first(store)
    queued = await store.enqueue_chat_memory_rebuild(
        user.id, project.id, _FINGERPRINT
    )
    assert queued is not None
    claimed = await store.claim_next_chat_memory_event(
        _FINGERPRINT, event_types=["rebuild"]
    )
    assert claimed is not None and claimed.claim_token
    snapshot = await store.prepare_chat_memory_rebuild_snapshot(
        claimed.event_id,
        claimed.claim_token,
        _FINGERPRINT,
        max_messages=100,
        max_bytes=1000,
    )
    assert snapshot is not None
    targets = await _prepare_rebuild_targets(store, claimed)
    old_group_id = chat_memory_graph_group_id(user.id, project.id, 1)
    assert old_group_id in targets.group_ids
    await store.mark_chat_memory_event_side_effect_started(
        claimed.event_id, claimed.claim_token, _FINGERPRINT
    )

    baseline_group = await store.get_chat_memory_group(user.id, project.id)
    baseline_generations = await store.list_chat_memory_generations(
        user.id, project.id
    )
    baseline_event = await store.get_chat_memory_event(claimed.event_id)
    baseline_mappings = await store.list_chat_memory_episodes_for_session(
        user.id, project.id, session.id
    )
    wrong_identity = copy.deepcopy(targets)
    wrong_identity.logical_group_id = f"{targets.logical_group_id}-tampered"
    with pytest.raises(MetadataConflictError) as identity_conflict:
        await store.finalize_chat_memory_rebuild(
            claimed.event_id,
            claimed.claim_token,
            _FINGERPRINT,
            snapshot,
            _snapshot_mappings(snapshot, "wrong-target-identity"),
            wrong_identity,
            targets.group_ids,
        )
    assert identity_conflict.value.entity_type == "chat_memory_rebuild_targets"
    assert await store.get_chat_memory_group(user.id, project.id) == baseline_group
    assert (
        await store.list_chat_memory_generations(user.id, project.id)
        == baseline_generations
    )
    assert await store.get_chat_memory_event(claimed.event_id) == baseline_event
    assert (
        await store.list_chat_memory_episodes_for_session(
            user.id, project.id, session.id
        )
        == baseline_mappings
    )
    definitely_cleared = tuple(
        group_id for group_id in targets.group_ids if group_id != old_group_id
    )

    with pytest.raises(MetadataConflictError) as conflict:
        await store.finalize_chat_memory_rebuild(
            claimed.event_id,
            claimed.claim_token,
            _FINGERPRINT,
            snapshot,
            _snapshot_mappings(snapshot, "missing-old-clear"),
            targets,
            definitely_cleared,
        )
    assert conflict.value.entity_type == "chat_memory_rebuild_clear"
    assert old_group_id in conflict.value.current["missing"]
    assert await store.get_chat_memory_group(user.id, project.id) == baseline_group
    assert (
        await store.list_chat_memory_generations(user.id, project.id)
        == baseline_generations
    )
    assert await store.get_chat_memory_event(claimed.event_id) == baseline_event
    assert (
        await store.list_chat_memory_episodes_for_session(
            user.id, project.id, session.id
        )
        == baseline_mappings
    )


async def test_rebuild_targets_sweep_all_generation_states_orphans_and_legacy(
    phase2b_store,
):
    store = phase2b_store
    user, project, _session, _saved, _final = await _activate_first(store)
    queued = await store.enqueue_chat_memory_rebuild(
        user.id, project.id, _FINGERPRINT
    )
    assert queued is not None and queued.generation == 2

    for expected_generation in (2, 3, 4):
        uncertain = await store.claim_next_chat_memory_event(
            _FINGERPRINT, event_types=["rebuild"]
        )
        assert uncertain is not None and uncertain.claim_token
        assert uncertain.generation == expected_generation
        snapshot = await store.prepare_chat_memory_rebuild_snapshot(
            uncertain.event_id,
            uncertain.claim_token,
            _FINGERPRINT,
            max_messages=100,
            max_bytes=1000,
        )
        assert snapshot is not None
        await _prepare_rebuild_targets(store, uncertain)
        await store.mark_chat_memory_event_side_effect_started(
            uncertain.event_id, uncertain.claim_token, _FINGERPRINT
        )
        replacement = await store.escalate_chat_memory_event_unknown(
            uncertain.event_id,
            uncertain.claim_token,
            _FINGERPRINT,
            error_message="unknown clear outcome",
        )
        assert replacement.generation == expected_generation + 1

    await _set_generation_state(store, user.id, project.id, 2, "retired")
    await _set_generation_state(store, user.id, project.id, 3, "purged")
    target = await store.claim_next_chat_memory_event(
        _FINGERPRINT, event_types=["rebuild"]
    )
    assert target is not None and target.claim_token and target.generation == 5
    snapshot = await store.prepare_chat_memory_rebuild_snapshot(
        target.event_id,
        target.claim_token,
        _FINGERPRINT,
        max_messages=100,
        max_bytes=1000,
    )
    assert snapshot is not None
    mapping_group, outbox_group = await _inject_defensive_purge_inventory(
        store, user.id, project.id
    )
    await _set_generation_state(store, user.id, project.id, 1, "active")
    targets = await _prepare_rebuild_targets(store, target)
    retried_snapshot = await store.prepare_chat_memory_rebuild_snapshot(
        target.event_id,
        target.claim_token,
        _FINGERPRINT,
        max_messages=100,
        max_bytes=1000,
    )
    assert retried_snapshot is not None
    assert retried_snapshot.snapshot_digest == snapshot.snapshot_digest

    inventory = await store.list_chat_memory_generations(user.id, project.id)
    assert [(item.generation, item.state) for item in inventory] == [
        (1, "active"),
        (2, "retired"),
        (3, "purged"),
        (4, "abandoned"),
        (5, "building"),
    ]
    expected_physical = {
        chat_memory_graph_group_id(user.id, project.id, generation)
        for generation in range(1, 6)
    }
    assert set(targets.group_ids) == expected_physical | {
        mapping_group,
        outbox_group,
        chat_memory_legacy_graph_group_id(user.id, project.id),
    }
    assert chat_memory_graph_group_id(user.id, project.id, 4) in targets.group_ids
    assert targets.group_ids == tuple(sorted(set(targets.group_ids)))


async def test_mixed_graph_inventory_prevents_rebuild_target_preparation(
    phase2b_store,
):
    store = phase2b_store
    user, project, _session, _saved, _final = await _activate_first(store)
    queued = await store.enqueue_chat_memory_rebuild(
        user.id, project.id, _FINGERPRINT
    )
    assert queued is not None
    claimed = await store.claim_next_chat_memory_event(
        _FINGERPRINT, event_types=["rebuild"]
    )
    assert claimed is not None and claimed.claim_token
    snapshot = await store.prepare_chat_memory_rebuild_snapshot(
        claimed.event_id,
        claimed.claim_token,
        _FINGERPRINT,
        max_messages=100,
        max_bytes=1000,
    )
    assert snapshot is not None
    mixed_graph = "chat-memory-graph-store:v1:mixed-rebuild-inventory"
    await _inject_mixed_graph_store_inventory(
        store,
        user.id,
        project.id,
        1,
        mixed_graph,
    )

    with pytest.raises(MetadataConflictError) as conflict:
        await _prepare_rebuild_targets(store, claimed)
    assert (
        conflict.value.current["error_code"]
        == "graph_store_migration_required"
    )
    assert set(conflict.value.current["graph_store_fingerprints"]) == {
        _FINGERPRINT,
        mixed_graph,
    }
    event = await store.get_chat_memory_event(claimed.event_id)
    assert event is not None and event.status == "running"
    assert event.side_effect_started_at is None


async def test_append_and_public_rebuild_wrong_graph_roll_back_atomically(
    phase2b_store,
):
    store = phase2b_store
    old_extraction = "chat-memory-extraction:v1:old"
    new_extraction = "chat-memory-extraction:v2:new"
    new_graph = "chat-memory-graph-store:v2:new"
    user, project, session = await _create_chat(store)
    await store.append_chat_messages_with_memory(
        [_message(user.id, project.id, session.id, "active old")],
        config_fingerprint=old_extraction,
        graph_store_fingerprint=_GRAPH_FINGERPRINT,
    )
    first = await store.claim_next_chat_memory_event(
        old_extraction,
        runtime_graph_store_fingerprint=_GRAPH_FINGERPRINT,
        event_types=["ingest"],
    )
    assert first is not None and first.claim_token
    await store.mark_chat_memory_event_side_effect_started(
        first.event_id,
        first.claim_token,
        old_extraction,
        runtime_graph_store_fingerprint=_GRAPH_FINGERPRINT,
    )
    await store.finalize_chat_memory_ingest(
        first.event_id,
        first.claim_token,
        old_extraction,
        episode_uuid=f"old-active-{uuid.uuid4().hex}",
        runtime_graph_store_fingerprint=_GRAPH_FINGERPRINT,
    )
    await store.append_chat_messages_with_memory(
        [_message(user.id, project.id, session.id, "old pending")],
        config_fingerprint=old_extraction,
        graph_store_fingerprint=_GRAPH_FINGERPRINT,
    )

    async def snapshot():
        group = await store.get_chat_memory_group(user.id, project.id)
        assert group is not None
        messages, total = await store.list_chat_messages(
            user.id, project.id, session.id
        )
        return (
            group.to_dict(),
            [
                item.to_dict()
                for item in await store.list_chat_memory_generations(
                    user.id, project.id
                )
            ],
            [
                item.to_dict()
                for item in await store.list_chat_memory_events(
                    user_id=user.id, project_id=project.id
                )
            ],
            [item.to_dict() for item in messages],
            total,
        )

    before = await snapshot()
    rejected = _message(user.id, project.id, session.id, "wrong graph")
    with pytest.raises(MetadataConflictError) as append_error:
        await store.append_chat_messages_with_memory(
            [rejected],
            config_fingerprint=new_extraction,
            graph_store_fingerprint=new_graph,
        )
    assert (
        append_error.value.current["error_code"]
        == "graph_store_migration_required"
    )
    assert rejected.seq == 0
    assert rejected.append_batch_id is None
    assert rejected.project_event_seq is None
    assert await snapshot() == before

    with pytest.raises(MetadataConflictError) as rebuild_error:
        await store.enqueue_chat_memory_rebuild(
            user.id,
            project.id,
            new_extraction,
            graph_store_fingerprint=new_graph,
        )
    assert (
        rebuild_error.value.current["error_code"]
        == "graph_store_migration_required"
    )
    assert await snapshot() == before


async def test_public_rebuild_allows_extraction_upgrade_on_same_graph(
    phase2b_store,
):
    store = phase2b_store
    old_extraction = "chat-memory-extraction:v1:old"
    new_extraction = "chat-memory-extraction:v2:new"
    user, project, session = await _create_chat(store)
    await store.append_chat_messages_with_memory(
        [_message(user.id, project.id, session.id, "old source")],
        config_fingerprint=old_extraction,
        graph_store_fingerprint=_GRAPH_FINGERPRINT,
    )

    queued = await store.enqueue_chat_memory_rebuild(
        user.id,
        project.id,
        new_extraction,
        graph_store_fingerprint=_GRAPH_FINGERPRINT,
    )
    assert queued is not None and queued.generation == 2
    appended = await store.append_chat_messages_with_memory(
        [_message(user.id, project.id, session.id, "new source")],
        config_fingerprint=new_extraction,
        graph_store_fingerprint=_GRAPH_FINGERPRINT,
    )

    group = await store.get_chat_memory_group(user.id, project.id)
    assert group is not None
    assert group.desired_generation == 2
    assert group.desired_config_fingerprint == new_extraction
    assert group.desired_graph_store_fingerprint == _GRAPH_FINGERPRINT
    assert appended[0].project_event_seq is not None
    inventory = await store.list_chat_memory_generations(user.id, project.id)
    assert [(item.generation, item.config_fingerprint) for item in inventory] == [
        (1, old_extraction),
        (2, new_extraction),
    ]
    assert {
        item.graph_store_fingerprint for item in inventory
    } == {_GRAPH_FINGERPRINT}
    events = await store.list_chat_memory_events(
        user_id=user.id, project_id=project.id
    )
    assert [(event.event_type, event.generation) for event in events] == [
        ("ingest", 1),
        ("rebuild", 2),
        ("ingest", 2),
    ]
    assert all(
        event.graph_store_fingerprint == _GRAPH_FINGERPRINT for event in events
    )


async def test_extraction_change_on_same_graph_fences_generation_and_orders_rebuild(
    phase2b_store,
):
    store = phase2b_store
    old_extraction = "chat-memory-extraction:v1:old"
    new_extraction = "chat-memory-extraction:v2:new"
    user, project, session = await _create_chat(store)

    await store.append_chat_messages_with_memory(
        [_message(user.id, project.id, session.id, "active old")],
        config_fingerprint=old_extraction,
        graph_store_fingerprint=_GRAPH_FINGERPRINT,
    )
    first = await store.claim_next_chat_memory_event(
        old_extraction,
        runtime_graph_store_fingerprint=_GRAPH_FINGERPRINT,
        event_types=["ingest"],
    )
    assert first is not None and first.claim_token
    await store.mark_chat_memory_event_side_effect_started(
        first.event_id,
        first.claim_token,
        old_extraction,
        runtime_graph_store_fingerprint=_GRAPH_FINGERPRINT,
    )
    await store.finalize_chat_memory_ingest(
        first.event_id,
        first.claim_token,
        old_extraction,
        episode_uuid=f"old-active-{uuid.uuid4().hex}",
        runtime_graph_store_fingerprint=_GRAPH_FINGERPRINT,
    )

    await store.append_chat_messages_with_memory(
        [_message(user.id, project.id, session.id, "old pending")],
        config_fingerprint=old_extraction,
        graph_store_fingerprint=_GRAPH_FINGERPRINT,
    )
    changed = await store.append_chat_messages_with_memory(
        [_message(user.id, project.id, session.id, "new target")],
        config_fingerprint=new_extraction,
        graph_store_fingerprint=_GRAPH_FINGERPRINT,
    )

    group = await store.get_chat_memory_group(user.id, project.id)
    assert group is not None
    assert group.active_generation == 1
    assert group.active_config_fingerprint == old_extraction
    assert group.active_graph_store_fingerprint == _GRAPH_FINGERPRINT
    assert group.desired_generation == 2
    assert group.desired_config_fingerprint == new_extraction
    assert group.desired_graph_store_fingerprint == _GRAPH_FINGERPRINT

    inventory = await store.list_chat_memory_generations(user.id, project.id)
    assert [item.generation for item in inventory] == [1, 2]
    assert inventory[0].state == "active"
    assert inventory[0].config_fingerprint == old_extraction
    assert inventory[0].graph_store_fingerprint == _GRAPH_FINGERPRINT
    assert inventory[1].state == "building"
    assert inventory[1].config_fingerprint == new_extraction
    assert inventory[1].graph_store_fingerprint == _GRAPH_FINGERPRINT

    events = await store.list_chat_memory_events(
        user_id=user.id, project_id=project.id
    )
    assert [(event.event_type, event.status, event.generation) for event in events] == [
        ("ingest", "succeeded", 1),
        ("ingest", "superseded", 1),
        ("rebuild", "pending", 2),
        ("ingest", "pending", 2),
    ]
    assert events[2].event_seq < changed[0].project_event_seq
    assert events[2].config_fingerprint == new_extraction
    assert events[2].graph_store_fingerprint == _GRAPH_FINGERPRINT
    assert events[3].config_fingerprint == new_extraction
    assert events[3].graph_store_fingerprint == _GRAPH_FINGERPRINT

    token = await store.get_chat_memory_read_token(user.id, project.id)
    assert token is not None
    assert token.state == "rebuilding"
    assert token.active_generation == 1
    assert token.active_config_fingerprint == old_extraction
    assert token.active_graph_store_fingerprint == _GRAPH_FINGERPRINT
    assert token.graph_group_id == chat_memory_graph_group_id(
        user.id, project.id, 1
    )
    assert token.generation_state == "active"

    assert (
        await store.claim_next_chat_memory_event(
            old_extraction,
            runtime_graph_store_fingerprint=_GRAPH_FINGERPRINT,
            event_types=["ingest", "rebuild"],
        )
        is None
    )
    rebuild = await store.claim_next_chat_memory_event(
        new_extraction,
        runtime_graph_store_fingerprint=_GRAPH_FINGERPRINT,
        event_types=["rebuild"],
    )
    assert rebuild is not None and rebuild.event_id == events[2].event_id


async def test_rebuild_snapshot_digest_rejects_tamper_matrix_atomically(
    phase2b_store,
):
    store = phase2b_store
    user, project, session, _saved, _final = await _activate_first(store, "seed")
    queued = await store.enqueue_chat_memory_rebuild(
        user.id, project.id, _FINGERPRINT
    )
    assert queued is not None
    claimed = await store.claim_next_chat_memory_event(
        _FINGERPRINT, event_types=["rebuild"]
    )
    assert claimed is not None and claimed.claim_token
    appended = await store.append_chat_messages_with_memory(
        [
            _memory_message(user.id, project.id, session.id, "alpha"),
            _memory_message(
                user.id,
                project.id,
                session.id,
                "assistant memory",
                role="assistant",
                memory_eligible=True,
            ),
        ],
        config_fingerprint=_FINGERPRINT,
    )
    snapshot = await store.prepare_chat_memory_rebuild_snapshot(
        claimed.event_id,
        claimed.claim_token,
        _FINGERPRINT,
        max_messages=100,
        max_bytes=100_000,
        ingest_max_chars=5,
    )
    assert snapshot is not None and snapshot.snapshot_digest
    assert len(snapshot.replay_batches[-1].messages) == 2
    targets = await _prepare_rebuild_targets(store, claimed)
    await store.mark_chat_memory_event_side_effect_started(
        claimed.event_id, claimed.claim_token, _FINGERPRINT
    )
    mappings = _snapshot_mappings(snapshot, "digest-valid")

    for tamper in (
        "same-byte-content",
        "reference-time",
        "role",
        "message-order",
        "message-id-substitution",
        "message-omission",
        "admission-metadata",
        "ingest-max-chars",
    ):
        candidate = copy.deepcopy(snapshot)
        batch = candidate.replay_batches[-1]
        if tamper == "same-byte-content":
            assert len(batch.messages[0].content.encode()) == len("omega".encode())
            batch.messages[0].content = "omega"
        elif tamper == "reference-time":
            batch.memory_reference_time += "-tampered"
        elif tamper == "role":
            batch.messages[0].role = "assistant"
        elif tamper == "message-order":
            batch.messages.reverse()
        elif tamper == "message-id-substitution":
            batch.messages[0].id = f"substituted-{uuid.uuid4().hex}"
        elif tamper == "message-omission":
            batch.messages.pop()
        elif tamper == "admission-metadata":
            batch.messages[1].metadata["memory_eligible"] = False
        else:
            candidate.ingest_max_chars += 1

        with pytest.raises(MetadataConflictError) as conflict:
            await store.finalize_chat_memory_rebuild(
                claimed.event_id,
                claimed.claim_token,
                _FINGERPRINT,
                candidate,
                mappings,
                targets,
                targets.group_ids,
            )
        assert conflict.value.entity_type == "chat_memory_rebuild_snapshot"

        group = await store.get_chat_memory_group(user.id, project.id)
        generation = await store.get_chat_memory_generation(
            user.id, project.id, claimed.generation
        )
        event = await store.get_chat_memory_event(claimed.event_id)
        pending_ingest = await store.get_chat_memory_event_by_sequence(
            user.id,
            project.id,
            appended[0].project_event_seq,
        )
        generation_mappings = [
            mapping
            for mapping in await store.list_chat_memory_episodes_for_session(
                user.id, project.id, session.id
            )
            if mapping.generation == claimed.generation
        ]
        assert group is not None and group.active_generation == 1
        assert generation is not None and generation.state == "building"
        assert event is not None and event.status == "running"
        assert pending_ingest is not None and pending_ingest.status == "pending"
        assert generation_mappings == []

    final = await store.finalize_chat_memory_rebuild(
        claimed.event_id,
        claimed.claim_token,
        _FINGERPRINT,
        snapshot,
        mappings,
        targets,
        targets.group_ids,
    )
    assert final is not None and final.group.active_generation == 2


async def test_rebuild_finalize_uses_snapshot_ingest_max_chars_only(
    phase2b_store,
):
    store = phase2b_store
    user, project, session, _saved, _final = await _activate_first(store, "seed")
    queued = await store.enqueue_chat_memory_rebuild(
        user.id, project.id, _FINGERPRINT
    )
    assert queued is not None
    claimed = await store.claim_next_chat_memory_event(
        _FINGERPRINT, event_types=["rebuild"]
    )
    assert claimed is not None and claimed.claim_token
    snapshot = await store.prepare_chat_memory_rebuild_snapshot(
        claimed.event_id,
        claimed.claim_token,
        _FINGERPRINT,
        max_messages=100,
        max_bytes=100_000,
        ingest_max_chars=5,
    )
    assert snapshot is not None and snapshot.ingest_max_chars == 5
    tampered_snapshot = copy.deepcopy(snapshot)
    tampered_snapshot.ingest_max_chars = 6
    mappings = _snapshot_mappings(tampered_snapshot, "single-max-chars-source")
    targets = await _prepare_rebuild_targets(store, claimed)
    await store.mark_chat_memory_event_side_effect_started(
        claimed.event_id, claimed.claim_token, _FINGERPRINT
    )

    baseline_group = await store.get_chat_memory_group(user.id, project.id)
    baseline_generation = await store.get_chat_memory_generation(
        user.id, project.id, claimed.generation
    )
    baseline_event = await store.get_chat_memory_event(claimed.event_id)
    baseline_mappings = await store.list_chat_memory_episodes_for_session(
        user.id, project.id, session.id
    )
    assert baseline_group is not None
    assert baseline_generation is not None
    assert baseline_event is not None

    async def assert_finalize_state_unchanged() -> None:
        assert (
            await store.get_chat_memory_group(user.id, project.id)
            == baseline_group
        )
        assert (
            await store.get_chat_memory_generation(
                user.id, project.id, claimed.generation
            )
            == baseline_generation
        )
        assert await store.get_chat_memory_event(claimed.event_id) == baseline_event
        assert (
            await store.list_chat_memory_episodes_for_session(
                user.id, project.id, session.id
            )
            == baseline_mappings
        )

    with pytest.raises(TypeError):
        await store.finalize_chat_memory_rebuild(
            claimed.event_id,
            claimed.claim_token,
            _FINGERPRINT,
            tampered_snapshot,
            mappings,
            targets,
            targets.group_ids,
            ingest_max_chars=5,
        )
    with pytest.raises(TypeError):
        await store.finalize_chat_memory_rebuild(
            claimed.event_id,
            claimed.claim_token,
            _FINGERPRINT,
            tampered_snapshot,
            mappings,
            targets,
            targets.group_ids,
            5,
        )
    await assert_finalize_state_unchanged()

    with pytest.raises(MetadataConflictError) as conflict:
        await store.finalize_chat_memory_rebuild(
            claimed.event_id,
            claimed.claim_token,
            _FINGERPRINT,
            tampered_snapshot,
            mappings,
            targets,
            targets.group_ids,
        )
    assert conflict.value.entity_type == "chat_memory_rebuild_snapshot"
    await assert_finalize_state_unchanged()


@pytest.mark.parametrize(
    ("max_messages", "max_bytes"),
    [(1, 1_000_000), (100, 1)],
)
async def test_rebuild_hard_cap_dead_letters_without_partial_activation(
    phase2b_store,
    monkeypatch,
    max_messages,
    max_bytes,
):
    store = phase2b_store
    user, project, session, _saved, _final = await _activate_first(store)
    await store.append_chat_messages_with_memory(
        [_message(user.id, project.id, session.id, "too large")],
        config_fingerprint=_FINGERPRINT,
    )
    queued = await store.enqueue_chat_memory_rebuild(
        user.id, project.id, _FINGERPRINT
    )
    assert queued is not None
    claimed = await store.claim_next_chat_memory_event(
        _FINGERPRINT, event_types=["rebuild"]
    )
    assert claimed is not None and claimed.claim_token

    def materialize_must_not_run(*_args, **_kwargs):
        raise AssertionError("hard-cap preflight materialized full message rows")

    helper_name = (
        "_materialize_sqlite_chat_memory_rebuild_batches"
        if isinstance(store, SQLiteMetadataStore)
        else "_materialize_postgres_chat_memory_rebuild_batches"
    )
    monkeypatch.setattr(store, helper_name, materialize_must_not_run)
    assert (
        await store.prepare_chat_memory_rebuild_snapshot(
            claimed.event_id,
            claimed.claim_token,
            _FINGERPRINT,
            max_messages=max_messages,
            max_bytes=max_bytes,
        )
        is None
    )

    event = await store.get_chat_memory_event(claimed.event_id)
    group = await store.get_chat_memory_group(user.id, project.id)
    generation = await store.get_chat_memory_generation(user.id, project.id, 2)
    assert event is not None and event.status == "dead_letter"
    assert event.snapshot_message_count == 2
    assert event.snapshot_digest is None
    assert event.side_effect_started_at is None
    assert group is not None and group.state == "failed"
    assert group.active_generation == 1
    assert generation is not None and generation.state == "building"
    assert generation.replay_message_count == 2
    assert generation.snapshot_digest is None
    mappings = await store.list_chat_memory_episodes_for_session(
        user.id, project.id, session.id
    )
    assert {mapping.generation for mapping in mappings} == {1}


async def test_stale_rebuild_never_activates_or_writes_mappings(phase2b_store):
    store = phase2b_store
    user, project, session, saved, _final = await _activate_first(store)
    queued = await store.enqueue_chat_memory_rebuild(
        user.id, project.id, _FINGERPRINT
    )
    assert queued is not None
    claimed = await store.claim_next_chat_memory_event(
        _FINGERPRINT, event_types=["rebuild"]
    )
    assert claimed is not None and claimed.claim_token
    snapshot = await store.prepare_chat_memory_rebuild_snapshot(
        claimed.event_id,
        claimed.claim_token,
        _FINGERPRINT,
        max_messages=100,
        max_bytes=1000,
    )
    assert snapshot is not None
    targets = await _prepare_rebuild_targets(store, claimed)
    await store.mark_chat_memory_event_side_effect_started(
        claimed.event_id, claimed.claim_token, _FINGERPRINT
    )
    assert await store.delete_chat_message_with_memory(
        user.id,
        project.id,
        session.id,
        saved[0].id,
        config_fingerprint=_FINGERPRINT,
    )

    assert (
        await store.finalize_chat_memory_rebuild(
            claimed.event_id,
            claimed.claim_token,
            _FINGERPRINT,
            snapshot,
            _snapshot_mappings(snapshot, "stale"),
            targets,
            targets.group_ids,
        )
        is None
    )
    event = await store.get_chat_memory_event(claimed.event_id)
    group = await store.get_chat_memory_group(user.id, project.id)
    generation = await store.get_chat_memory_generation(user.id, project.id, 2)
    assert event is not None and event.status == "superseded"
    assert group is not None
    assert group.desired_generation == 3
    assert group.active_generation == 1
    assert generation is not None and generation.state == "abandoned"
    mappings = await store.list_chat_memory_episodes_for_session(
        user.id, project.id, session.id
    )
    assert all(mapping.generation != 2 for mapping in mappings)


async def test_rebuild_mapping_conflict_rolls_back_entire_finalize(phase2b_store):
    store = phase2b_store
    user, project, session, _saved, _final = await _activate_first(store)
    second = await store.append_chat_messages_with_memory(
        [_message(user.id, project.id, session.id, "second")],
        config_fingerprint=_FINGERPRINT,
    )
    second_event = await store.claim_next_chat_memory_event(
        _FINGERPRINT, event_types=["ingest"]
    )
    assert second_event is not None and second_event.claim_token
    await store.mark_chat_memory_event_side_effect_started(
        second_event.event_id, second_event.claim_token, _FINGERPRINT
    )
    await store.finalize_chat_memory_ingest(
        second_event.event_id,
        second_event.claim_token,
        _FINGERPRINT,
        episode_uuid=f"phase2b-second-{uuid.uuid4().hex}",
    )

    queued = await store.enqueue_chat_memory_rebuild(
        user.id, project.id, _FINGERPRINT
    )
    assert queued is not None
    claimed = await store.claim_next_chat_memory_event(
        _FINGERPRINT, event_types=["rebuild"]
    )
    assert claimed is not None and claimed.claim_token
    snapshot = await store.prepare_chat_memory_rebuild_snapshot(
        claimed.event_id,
        claimed.claim_token,
        _FINGERPRINT,
        max_messages=100,
        max_bytes=1000,
    )
    assert snapshot is not None and len(snapshot.replay_batches) == 2
    targets = await _prepare_rebuild_targets(store, claimed)
    conflicting_batch = snapshot.replay_batches[-1]
    await store.record_chat_memory_episode(
        ChatMemoryEpisodeRecord(
            episode_uuid="phase2b-existing-conflict",
            session_id=conflicting_batch.session_id,
            project_id=project.id,
            user_id=user.id,
            first_seq=second[0].seq,
            last_seq=second[0].seq,
            created_at=utc_now_iso(),
            event_id=claimed.event_id,
            generation=claimed.generation,
            graph_group_id=claimed.graph_group_id,
            append_batch_id=conflicting_batch.append_batch_id,
            project_event_seq=conflicting_batch.project_event_seq,
        )
    )
    await store.mark_chat_memory_event_side_effect_started(
        claimed.event_id, claimed.claim_token, _FINGERPRINT
    )
    with pytest.raises(MetadataConflictError):
        await store.finalize_chat_memory_rebuild(
            claimed.event_id,
            claimed.claim_token,
            _FINGERPRINT,
            snapshot,
            _snapshot_mappings(snapshot, "conflict"),
            targets,
            targets.group_ids,
        )

    group = await store.get_chat_memory_group(user.id, project.id)
    generation = await store.get_chat_memory_generation(user.id, project.id, 2)
    event = await store.get_chat_memory_event(claimed.event_id)
    mappings = await store.list_chat_memory_episodes_for_session(
        user.id, project.id, session.id
    )
    generation_two = [mapping for mapping in mappings if mapping.generation == 2]
    assert group is not None and group.active_generation == 1
    assert generation is not None and generation.state == "building"
    assert event is not None and event.status == "running"
    assert [mapping.episode_uuid for mapping in generation_two] == [
        "phase2b-existing-conflict"
    ]


async def test_purge_inventory_missing_target_atomic_completion_and_terminal_noop(
    phase2b_store,
):
    store = phase2b_store
    user, project, session, _saved, _final = await _activate_first(store)
    await _complete_admin_rebuild(store, user.id, project.id, "purge_active")

    queued = await store.enqueue_chat_memory_rebuild(
        user.id, project.id, _FINGERPRINT
    )
    assert queued is not None and queued.generation == 3
    uncertain = await store.claim_next_chat_memory_event(
        _FINGERPRINT, event_types=["rebuild"]
    )
    assert uncertain is not None and uncertain.claim_token
    snapshot = await store.prepare_chat_memory_rebuild_snapshot(
        uncertain.event_id,
        uncertain.claim_token,
        _FINGERPRINT,
        max_messages=100,
        max_bytes=1000,
    )
    assert snapshot is not None
    await store.mark_chat_memory_event_side_effect_started(
        uncertain.event_id, uncertain.claim_token, _FINGERPRINT
    )
    replacement = await store.escalate_chat_memory_event_unknown(
        uncertain.event_id,
        uncertain.claim_token,
        _FINGERPRINT,
        error_message="unknown rebuild outcome",
    )
    assert replacement.generation == 4

    deleted, _sessions, _messages = await store.delete_chat_project_with_memory(
        user.id, project.id, config_fingerprint=_FINGERPRINT
    )
    assert deleted
    assert await store.get_chat_project(user.id, project.id) is None
    purge = await store.claim_next_chat_memory_event(
        _FINGERPRINT, event_types=["purge"]
    )
    assert purge is not None and purge.claim_token
    mapping_group, outbox_group = await _inject_defensive_purge_inventory(
        store, user.id, project.id
    )
    targets = await store.get_chat_memory_purge_targets(
        purge.event_id, purge.claim_token, _FINGERPRINT
    )
    assert targets is not None
    expected_physical = {
        chat_memory_graph_group_id(user.id, project.id, generation)
        for generation in range(1, 5)
    }
    assert set(targets.group_ids) == expected_physical | {
        chat_memory_legacy_graph_group_id(user.id, project.id),
        mapping_group,
        outbox_group,
    }
    assert targets.group_ids == tuple(sorted(set(targets.group_ids)))

    await store.mark_chat_memory_event_side_effect_started(
        purge.event_id, purge.claim_token, _FINGERPRINT
    )
    with pytest.raises(MetadataConflictError):
        await store.finalize_chat_memory_purge(
            purge.event_id,
            purge.claim_token,
            _FINGERPRINT,
            targets,
            targets.group_ids[:-1],
        )
    unchanged = await store.get_chat_memory_event(purge.event_id)
    group = await store.get_chat_memory_group(user.id, project.id)
    assert unchanged is not None and unchanged.status == "running"
    assert group is not None and group.state == "deleting"
    before_finalize_inventory = await store.list_chat_memory_generations(
        user.id, project.id
    )
    assert before_finalize_inventory[0].state == "purged"
    assert all(
        generation.state == "purge_pending"
        for generation in before_finalize_inventory[1:]
    )

    before_event_count = len(
        await store.list_chat_memory_events(user_id=user.id, project_id=project.id)
    )
    final = await store.finalize_chat_memory_purge(
        purge.event_id,
        purge.claim_token,
        _FINGERPRINT,
        targets,
        targets.group_ids,
    )
    assert final is not None
    assert final.event.status == "succeeded"
    assert final.group.state == "deleted"
    assert final.group.active_generation is None
    assert all(
        generation.state == "purged"
        for generation in await store.list_chat_memory_generations(
            user.id, project.id
        )
    )
    assert (
        await store.list_chat_memory_episodes_for_session(
            user.id, project.id, session.id
        )
        == []
    )

    assert (
        await store.enqueue_chat_memory_purge(
            user.id, project.id, _FINGERPRINT
        )
        is None
    )
    assert (
        await store.enqueue_chat_memory_rebuild(
            user.id, project.id, _FINGERPRINT
        )
        is None
    )
    after_events = await store.list_chat_memory_events(
        user_id=user.id, project_id=project.id
    )
    assert len(after_events) == before_event_count
    assert any(event.event_type == "purge" for event in after_events)


async def test_mixed_graph_inventory_prevents_purge_terminalization(phase2b_store):
    store = phase2b_store
    user, project, session, _saved, _final = await _activate_first(store)
    await _complete_admin_rebuild(store, user.id, project.id, "mixed_graph")
    queued = await store.enqueue_chat_memory_purge(
        user.id, project.id, _FINGERPRINT
    )
    assert queued is not None and queued.generation == 2
    purge = await store.claim_next_chat_memory_event(
        _FINGERPRINT, event_types=["purge"]
    )
    assert purge is not None and purge.claim_token
    targets = await store.prepare_chat_memory_purge_targets(
        purge.event_id, purge.claim_token, _FINGERPRINT
    )
    assert targets is not None
    await store.mark_chat_memory_event_side_effect_started(
        purge.event_id, purge.claim_token, _FINGERPRINT
    )
    mixed_graph = "chat-memory-graph-store:v1:mixed-inventory"
    await _inject_mixed_graph_store_inventory(
        store,
        user.id,
        project.id,
        1,
        mixed_graph,
    )

    with pytest.raises(MetadataConflictError) as conflict:
        await store.finalize_chat_memory_purge(
            purge.event_id,
            purge.claim_token,
            _FINGERPRINT,
            targets,
            targets.group_ids,
        )
    assert (
        conflict.value.current["error_code"]
        == "graph_store_migration_required"
    )
    assert set(conflict.value.current["graph_store_fingerprints"]) == {
        _FINGERPRINT,
        mixed_graph,
    }

    event = await store.get_chat_memory_event(purge.event_id)
    group = await store.get_chat_memory_group(user.id, project.id)
    inventory = await store.list_chat_memory_generations(user.id, project.id)
    mappings = await store.list_chat_memory_episodes_for_session(
        user.id, project.id, session.id
    )
    assert event is not None and event.status == "running"
    assert group is not None and group.state == "deleting"
    assert [(item.generation, item.state) for item in inventory] == [
        (1, "purged"),
        (2, "purge_pending"),
    ]
    assert any(item.graph_store_fingerprint == mixed_graph for item in inventory)
    assert mappings


async def test_project_delete_without_sql_memory_evidence_still_enqueues_legacy_sweep(
    phase2b_store,
):
    store = phase2b_store
    user, project, _session = await _create_chat(store)
    assert await store.get_chat_memory_group(user.id, project.id) is None

    deleted, session_count, message_count = (
        await store.delete_chat_project_with_memory(
            user.id,
            project.id,
            config_fingerprint=_FINGERPRINT,
        )
    )
    assert (deleted, session_count, message_count) == (True, 1, 0)
    purge = await store.claim_next_chat_memory_event(
        _FINGERPRINT, event_types=["purge"]
    )
    assert purge is not None and purge.claim_token
    targets = await store.get_chat_memory_purge_targets(
        purge.event_id, purge.claim_token, _FINGERPRINT
    )
    assert targets is not None
    assert chat_memory_legacy_graph_group_id(user.id, project.id) in (
        targets.group_ids
    )


async def test_user_delete_unconditionally_purges_source_and_durable_orphan_projects(
    phase2b_store,
):
    store = phase2b_store
    user, project, _session = await _create_chat(store)
    orphan_project_id = f"orphan-project-{uuid.uuid4().hex}"
    orphan_rebuild = await store.enqueue_chat_memory_rebuild(
        user.id, orphan_project_id, _FINGERPRINT
    )
    assert orphan_rebuild is not None

    assert await store.delete_enterprise_user_with_memory(
        user.id,
        config_fingerprint=_FINGERPRINT,
        actor_user_id="usr_admin",
    )
    for project_id in (project.id, orphan_project_id):
        group = await store.get_chat_memory_group(user.id, project_id)
        events = await store.list_chat_memory_events(
            user_id=user.id, project_id=project_id
        )
        assert group is not None and group.state == "deleting"
        assert events[-1].event_type == "purge"
        assert events[-1].status == "pending"
        assert events[-1].actor_user_id == "usr_admin"


async def test_terminal_public_purge_does_not_reopen_when_source_is_retained(
    phase2b_store,
):
    store = phase2b_store
    user, project, session, _saved, _final = await _activate_first(
        store, "retained source"
    )
    queued = await store.enqueue_chat_memory_purge(
        user.id, project.id, _FINGERPRINT
    )
    assert queued is not None
    purge = await store.claim_next_chat_memory_event(
        _FINGERPRINT, event_types=["purge"]
    )
    assert purge is not None and purge.claim_token
    targets = await store.get_chat_memory_purge_targets(
        purge.event_id, purge.claim_token, _FINGERPRINT
    )
    assert targets is not None
    await store.mark_chat_memory_event_side_effect_started(
        purge.event_id, purge.claim_token, _FINGERPRINT
    )
    final = await store.finalize_chat_memory_purge(
        purge.event_id,
        purge.claim_token,
        _FINGERPRINT,
        targets,
        targets.group_ids,
    )
    assert final is not None and final.group.state == "deleted"
    retained, retained_count = await store.list_chat_messages(
        user.id, project.id, session.id
    )
    assert retained_count == 1 and retained[0].content == "retained source"

    before_group = await store.get_chat_memory_group(user.id, project.id)
    before_events = await store.list_chat_memory_events(
        user_id=user.id, project_id=project.id
    )
    assert (
        await store.enqueue_chat_memory_purge(
            user.id, project.id, _FINGERPRINT
        )
        is None
    )
    assert (
        await store.enqueue_chat_memory_rebuild(
            user.id, project.id, _FINGERPRINT
        )
        is None
    )
    after_group = await store.get_chat_memory_group(user.id, project.id)
    after_events = await store.list_chat_memory_events(
        user_id=user.id, project_id=project.id
    )
    assert after_group == before_group
    assert [event.event_id for event in after_events] == [
        event.event_id for event in before_events
    ]


@pytest.mark.parametrize(
    ("method_name", "event_type"),
    [
        ("enqueue_chat_memory_rebuild", "rebuild"),
        ("enqueue_chat_memory_purge", "purge"),
    ],
)
async def test_first_public_enqueue_is_cross_store_idempotent(
    phase2b_store,
    method_name,
    event_type,
):
    store = phase2b_store
    user, project, _session = await _create_chat(store)
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
        first, second = await asyncio.gather(
            getattr(store, method_name)(user.id, project.id, _FINGERPRINT),
            getattr(peer, method_name)(user.id, project.id, _FINGERPRINT),
        )
    finally:
        await peer.close()

    assert first is not None and second is not None
    assert first.event_id == second.event_id
    assert first.deterministic_key == second.deterministic_key
    events = await store.list_chat_memory_events(
        user_id=user.id, project_id=project.id
    )
    assert [(event.event_type, event.event_id) for event in events] == [
        (event_type, first.event_id)
    ]


async def test_sqlite_snapshot_digest_schema_migrates_idempotently(tmp_path):
    db_path = tmp_path / "phase2b-snapshot-digest-migration.sqlite3"
    store = SQLiteMetadataStore(db_path)
    await store.initialize()
    await store.close()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "ALTER TABLE enterprise_chat_memory_generations "
            "DROP COLUMN snapshot_digest"
        )
        conn.execute(
            "ALTER TABLE enterprise_chat_memory_outbox "
            "DROP COLUMN snapshot_digest"
        )
        conn.execute("DELETE FROM metadata_schema WHERE version = 11")
        conn.execute(
            "INSERT OR IGNORE INTO metadata_schema(version, applied_at) "
            "VALUES (9, '2026-01-01T00:00:00+00:00')"
        )

    migrated = SQLiteMetadataStore(db_path)
    await migrated.initialize()
    await migrated.initialize()
    try:
        with sqlite3.connect(db_path) as conn:
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
        assert "snapshot_digest" in generation_columns
        assert "snapshot_digest" in outbox_columns
        assert 11 in versions
    finally:
        await migrated.close()


async def test_phase2b_public_contract_exists_on_both_stores(phase2b_store):
    store = phase2b_store
    for method_name in (
        "enqueue_chat_memory_rebuild",
        "enqueue_chat_memory_purge",
        "prepare_chat_memory_rebuild_snapshot",
        "prepare_chat_memory_rebuild_targets",
        "finalize_chat_memory_ingest_noop",
        "finalize_chat_memory_rebuild",
        "get_chat_memory_purge_targets",
        "prepare_chat_memory_purge_targets",
        "finalize_chat_memory_purge",
    ):
        assert callable(getattr(store, method_name))
    if isinstance(store, SQLiteMetadataStore):
        return
    assert store.__class__.__name__ == "PostgresMetadataStore"
    async with store._pool_or_raise().acquire() as conn:
        digest_columns = await conn.fetchval(
            """
            SELECT COUNT(*) FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND column_name = 'snapshot_digest'
              AND table_name IN (
                  'enterprise_chat_memory_generations',
                  'enterprise_chat_memory_outbox'
              )
            """
        )
        schema_v3 = await conn.fetchval(
            "SELECT 1 FROM kb_metadata_schema WHERE version = 3"
        )
    assert digest_columns == 2
    assert schema_v3 == 1
