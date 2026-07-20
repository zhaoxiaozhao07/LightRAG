"""Focused runtime contracts for the durable Chat Memory worker."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from lightrag.api.chat_memory_service import ChatMemoryConfig
from lightrag.api.chat_memory_worker import ChatMemoryWorker
from lightrag.api.metadata_store import (
    ChatMemoryOutboxEventRecord,
    ChatMemoryPurgeTargetSet,
    ChatMemoryRebuildSnapshot,
    ChatMemoryRebuildTargetSet,
    ChatMemoryReplayBatch,
    ChatMessageRecord,
    chat_memory_legacy_graph_group_id,
    chat_memory_logical_group_id,
)

pytestmark = pytest.mark.offline


def _config(**kwargs: Any) -> ChatMemoryConfig:
    defaults = {
        "llm_base_url": "https://llm/v1",
        "llm_model": "memory-model",
        "embedding_base_url": "https://embed/v1",
        "embedding_model": "embed-model",
        "embedding_dim": 1536,
        "neo4j_uri": "bolt://neo4j:7687",
        "worker_side_effect_timeout_seconds": 1.0,
        "worker_shutdown_timeout_seconds": 0.1,
        "ingest_concurrency": 2,
    }
    defaults.update(kwargs)
    return ChatMemoryConfig(**defaults)


def _event(
    config: ChatMemoryConfig,
    *,
    event_id: str,
    event_type: str = "ingest",
    user_id: str = "usr_a",
    project_id: str = "proj_a",
    event_seq: int = 1,
    generation: int = 1,
    fingerprint: str | None = None,
) -> ChatMemoryOutboxEventRecord:
    append_batch_id = f"batch-{event_id}" if event_type == "ingest" else None
    return ChatMemoryOutboxEventRecord(
        event_id=event_id,
        deterministic_key=f"key-{event_id}",
        user_id=user_id,
        project_id=project_id,
        event_seq=event_seq,
        generation=generation,
        graph_group_id=f"group-{user_id}-{project_id}-g{generation}",
        config_fingerprint=fingerprint or config.extraction_fingerprint(),
        event_type=event_type,  # type: ignore[arg-type]
        status="pending",
        available_at="2026-07-15T00:00:00+00:00",
        attempt_no=0,
        created_at="2026-07-15T00:00:00+00:00",
        updated_at="2026-07-15T00:00:00+00:00",
        source_session_id="sess_a" if event_type == "ingest" else None,
        append_batch_id=append_batch_id,
        first_seq=event_seq if event_type == "ingest" else None,
        last_seq=event_seq if event_type == "ingest" else None,
        graph_store_fingerprint=config.graph_store_fingerprint(),
    )


def _message(
    *,
    content: str,
    seq: int,
    event_seq: int,
    append_batch_id: str,
    role: str = "user",
    memory_eligible: bool | None = None,
) -> ChatMessageRecord:
    metadata: dict[str, Any] = {}
    if memory_eligible is not None:
        metadata["memory_eligible"] = memory_eligible
    return ChatMessageRecord(
        id=f"msg-{event_seq}-{seq}-{role}",
        session_id="sess_a",
        project_id="proj_a",
        user_id="usr_a",
        role=role,
        content=content,
        metadata=metadata,
        seq=seq,
        created_at="2026-07-15T00:00:00+00:00",
        append_batch_id=append_batch_id,
        project_event_seq=event_seq,
        memory_reference_time="2026-07-15T00:00:00+00:00",
    )


def _batch(
    event: ChatMemoryOutboxEventRecord,
    messages: list[ChatMessageRecord],
) -> ChatMemoryReplayBatch:
    assert event.append_batch_id is not None
    return ChatMemoryReplayBatch(
        append_batch_id=event.append_batch_id,
        project_event_seq=event.event_seq,
        memory_reference_time="2026-07-15T00:00:00+00:00",
        session_id="sess_a",
        messages=messages,
    )


def _rebuild_targets(
    event: ChatMemoryOutboxEventRecord, *additional_group_ids: str
) -> ChatMemoryRebuildTargetSet:
    return ChatMemoryRebuildTargetSet(
        event_id=event.event_id,
        user_id=event.user_id,
        project_id=event.project_id,
        logical_group_id=chat_memory_logical_group_id(
            event.user_id, event.project_id
        ),
        group_ids=tuple(
            sorted(
                {
                    event.graph_group_id,
                    chat_memory_legacy_graph_group_id(
                        event.user_id, event.project_id
                    ),
                    *additional_group_ids,
                }
            )
        ),
    )


class FakeStore:
    def __init__(
        self,
        events: list[ChatMemoryOutboxEventRecord] | None = None,
        *,
        timeline: list[str] | None = None,
    ) -> None:
        self.events = list(events or [])
        self.timeline = timeline if timeline is not None else []
        self.claim_calls: list[dict[str, Any]] = []
        self.claim_count = 0
        self.claimed_event = asyncio.Event()
        self.guard_calls: list[str] = []
        self._guard_locks: dict[str, asyncio.Lock] = {}
        self.batches: dict[str, list[ChatMemoryReplayBatch]] = {}
        self.snapshots: dict[str, ChatMemoryRebuildSnapshot | None] = {}
        self.rebuild_targets: dict[str, ChatMemoryRebuildTargetSet | None] = {}
        self.targets: dict[str, ChatMemoryPurgeTargetSet | None] = {}
        self.marker_statuses: list[str] = []
        self.mark_calls: list[str] = []
        self.ingest_finalized: list[tuple[str, str]] = []
        self.noop_finalized: list[str] = []
        self.rebuild_finalized: list[tuple[str, list[Any]]] = []
        self.rebuild_finalize_targets: list[
            tuple[ChatMemoryRebuildTargetSet, tuple[str, ...]]
        ] = []
        self.purge_finalized: list[str] = []
        self.known_failures: list[tuple[str, str]] = []
        self.unknown_escalations: list[str] = []
        self.purge_unknown_retries: list[str] = []
        self.stale_events: list[ChatMemoryOutboxEventRecord] = []
        self.recovery_calls: list[str] = []
        self.ignore_claim_fingerprint = False
        self.finalizer_errors: dict[str, Exception] = {}
        self.store_identity_calls: list[tuple[str, str, str]] = []
        self.unknown_transition_gate: asyncio.Event | None = None
        self.unknown_transition_started = asyncio.Event()

    async def claim_next_chat_memory_event(
        self,
        runtime_fingerprint,
        *,
        runtime_graph_store_fingerprint=None,
        worker_id=None,
        event_types=None,
    ):
        self.claim_count += 1
        self.claimed_event.set()
        self.claim_calls.append(
            {
                "runtime_fingerprint": runtime_fingerprint,
                "runtime_graph_store_fingerprint": runtime_graph_store_fingerprint,
                "worker_id": worker_id,
                "event_types": tuple(event_types or ()),
            }
        )
        for index, event in enumerate(self.events):
            if event.event_type not in tuple(event_types or ()):
                continue
            if (
                not self.ignore_claim_fingerprint
                and (
                    event.graph_store_fingerprint
                    != runtime_graph_store_fingerprint
                    or (
                        event.event_type != "purge"
                        and event.config_fingerprint != runtime_fingerprint
                    )
                )
            ):
                continue
            self.events.pop(index)
            event.status = "running"
            event.attempt_no += 1
            event.claim_token = f"claim-{event.event_id}-{event.attempt_no}"
            event.claimed_by = worker_id
            event.claimed_at = "2026-07-15T00:00:00+00:00"
            return event
        return None

    @asynccontextmanager
    async def chat_memory_group_execution_guard(self, logical_group_id, *, wait=True):
        self.guard_calls.append(logical_group_id)
        lock = self._guard_locks.setdefault(logical_group_id, asyncio.Lock())
        await lock.acquire()
        try:
            yield True
        finally:
            lock.release()

    async def get_chat_memory_execution_state(self, event_id):
        candidates = [*self.events, *self.stale_events]
        event = next((item for item in candidates if item.event_id == event_id), None)
        if event is None:
            # Claimed events are removed from the queue; tests retain them through
            # the claimed event references recorded in claim_calls_by_id.
            event = getattr(self, f"claimed_{event_id}", None)
        return SimpleNamespace(event=event) if event is not None else None

    async def list_admitted_chat_memory_replay_batches(
        self,
        user_id,
        project_id,
        *,
        through_event_seq,
        after_event_seq=0,
        limit=100,
    ):
        matching = [
            batch
            for batches in self.batches.values()
            for batch in batches
            if after_event_seq < batch.project_event_seq <= through_event_seq
        ]
        return sorted(matching, key=lambda item: item.project_event_seq)[:limit]

    async def mark_chat_memory_event_side_effect_started(
        self,
        event_id,
        claim_token,
        runtime_fingerprint,
        *,
        runtime_graph_store_fingerprint=None,
        fingerprint_retry_delay_seconds=1.0,
    ):
        event = getattr(self, f"claimed_{event_id}")
        self.timeline.append("mark")
        self.mark_calls.append(event_id)
        status = self.marker_statuses.pop(0) if self.marker_statuses else "running"
        event.status = status
        if status == "running":
            event.side_effect_started_at = "2026-07-15T00:00:01+00:00"
            event.side_effect_state_version = 1
        return event

    async def finalize_chat_memory_ingest(
        self,
        event_id,
        claim_token,
        runtime_fingerprint,
        *,
        episode_uuid,
        runtime_graph_store_fingerprint=None,
    ):
        self.ingest_finalized.append((event_id, episode_uuid))
        if error := self.finalizer_errors.get("ingest"):
            raise error

    async def finalize_chat_memory_ingest_noop(
        self,
        event_id,
        claim_token,
        runtime_fingerprint,
        *,
        runtime_graph_store_fingerprint=None,
    ):
        self.noop_finalized.append(event_id)

    async def prepare_chat_memory_rebuild_snapshot(
        self,
        event_id,
        claim_token,
        runtime_fingerprint,
        max_messages,
        max_bytes,
        ingest_max_chars,
        *,
        runtime_graph_store_fingerprint=None,
    ):
        self.timeline.append("prepare-rebuild")
        return self.snapshots[event_id]

    async def prepare_chat_memory_rebuild_targets(
        self,
        event_id,
        claim_token,
        runtime_fingerprint,
        *,
        runtime_graph_store_fingerprint=None,
    ):
        self.timeline.append("prepare-targets")
        if event_id in self.rebuild_targets:
            return self.rebuild_targets[event_id]
        return _rebuild_targets(getattr(self, f"claimed_{event_id}"))

    async def finalize_chat_memory_rebuild(
        self,
        event_id,
        claim_token,
        runtime_fingerprint,
        snapshot,
        mappings,
        targets,
        definitely_cleared_group_ids,
        *,
        runtime_graph_store_fingerprint=None,
    ):
        self.rebuild_finalized.append((event_id, list(mappings)))
        self.rebuild_finalize_targets.append(
            (targets, tuple(definitely_cleared_group_ids))
        )
        if error := self.finalizer_errors.get("rebuild"):
            raise error

    async def prepare_chat_memory_purge_targets(
        self,
        event_id,
        claim_token,
        runtime_fingerprint,
        *,
        runtime_graph_store_fingerprint=None,
    ):
        return self.targets[event_id]

    async def finalize_chat_memory_purge(
        self,
        event_id,
        claim_token,
        runtime_fingerprint,
        targets,
        definitely_cleared_group_ids,
        *,
        runtime_graph_store_fingerprint=None,
    ):
        self.purge_finalized.append(event_id)
        if error := self.finalizer_errors.get("purge"):
            raise error

    async def fail_chat_memory_event_before_side_effect(
        self,
        event_id,
        claim_token,
        runtime_fingerprint,
        *,
        runtime_graph_store_fingerprint=None,
        error_code,
        error_message,
        retry_delay_seconds,
        max_attempts,
    ):
        self.known_failures.append((event_id, error_code))

    async def fail_chat_memory_purge_before_side_effect(self, *args, **kwargs):
        self.known_failures.append((args[0], kwargs["error_code"]))

    async def escalate_chat_memory_event_unknown(self, event_id, *args, **kwargs):
        self.unknown_transition_started.set()
        if self.unknown_transition_gate is not None:
            await self.unknown_transition_gate.wait()
        self.unknown_escalations.append(event_id)

    async def retry_chat_memory_purge_after_unknown_clear(
        self, event_id, *args, **kwargs
    ):
        self.unknown_transition_started.set()
        if self.unknown_transition_gate is not None:
            await self.unknown_transition_gate.wait()
        self.purge_unknown_retries.append(event_id)

    async def list_stale_chat_memory_running_events(
        self, *, stale_after_seconds, limit=100
    ):
        return self.stale_events[:limit]

    async def recover_stale_chat_memory_event(self, event_id, *args, **kwargs):
        self.recovery_calls.append(event_id)
        return SimpleNamespace(event_id=event_id)

    def retain_claimed(self, event: ChatMemoryOutboxEventRecord) -> None:
        setattr(self, f"claimed_{event.event_id}", event)


class FakeService:
    def __init__(self, *, timeline: list[str] | None = None) -> None:
        self.timeline = timeline if timeline is not None else []
        self.graphiti = object()
        self.ensure_calls = 0
        self.add_calls: list[dict[str, Any]] = []
        self.clear_calls: list[list[str]] = []
        self.invalidated: list[Any] = []
        self.add_error: Exception | None = None
        self.clear_error: Exception | None = None
        self.add_delay = 0.0
        self.clear_delay = 0.0
        self.lease_error: Exception | None = None
        self.active_leases = 0

    async def ensure_backend(self):
        self.timeline.append("ensure")
        self.ensure_calls += 1
        return self.graphiti

    @asynccontextmanager
    async def backend_lease(self):
        self.timeline.append("ensure")
        self.ensure_calls += 1
        if self.lease_error is not None:
            raise self.lease_error
        self.active_leases += 1
        try:
            yield self.graphiti
        finally:
            self.active_leases -= 1

    async def add_episode(self, graphiti, **kwargs):
        self.timeline.append("add")
        self.add_calls.append(kwargs)
        if self.add_delay:
            await asyncio.sleep(self.add_delay)
        if self.add_error is not None:
            raise self.add_error
        return SimpleNamespace(
            episode=SimpleNamespace(uuid=f"episode-{len(self.add_calls)}")
        )

    async def clear_graph_groups(self, graphiti, group_ids):
        self.timeline.append("clear")
        self.clear_calls.append(list(group_ids))
        if self.clear_delay:
            await asyncio.sleep(self.clear_delay)
        if self.clear_error is not None:
            raise self.clear_error

    async def invalidate_backend(self, graphiti=None):
        self.invalidated.append(graphiti)


def _worker(
    store: FakeStore,
    service: FakeService,
    config: ChatMemoryConfig,
    **kwargs: Any,
) -> ChatMemoryWorker:
    worker = ChatMemoryWorker(
        store,
        service,  # type: ignore[arg-type]
        config,
        worker_id="worker-test",
        retry_delay_seconds=0,
        **kwargs,
    )
    for event in store.events:
        store.retain_claimed(event)
    return worker


async def test_poll_once_claims_whitelist_and_ingests_exact_batch():
    timeline: list[str] = []
    config = _config(ingest_max_chars=5)
    event = _event(config, event_id="ingest-1")
    assert event.append_batch_id is not None
    messages = [
        _message(
            content="123456789",
            seq=1,
            event_seq=1,
            append_batch_id=event.append_batch_id,
        ),
        _message(
            content="assistant excluded",
            seq=2,
            event_seq=1,
            append_batch_id=event.append_batch_id,
            role="assistant",
        ),
    ]
    store = FakeStore([event], timeline=timeline)
    store.batches[event.event_id] = [_batch(event, messages)]
    service = FakeService(timeline=timeline)
    worker = _worker(store, service, config, event_types=["ingest"])

    assert await worker.poll_once() is event
    assert (
        store.claim_calls[0]["runtime_fingerprint"]
        == config.extraction_fingerprint()
    )
    assert (
        store.claim_calls[0]["runtime_graph_store_fingerprint"]
        == config.graph_store_fingerprint()
    )
    assert store.claim_calls[0]["worker_id"] == "worker-test"
    assert store.claim_calls[0]["event_types"] == ("ingest",)
    assert timeline.index("ensure") < timeline.index("mark") < timeline.index("add")
    assert service.add_calls[0]["episode_body"] == "user: 12345…[truncated]"
    assert "uuid" not in service.add_calls[0]
    assert store.ingest_finalized == [(event.event_id, "episode-1")]


async def test_ingest_empty_canonical_payload_finalizes_noop_without_backend():
    config = _config()
    event = _event(config, event_id="noop-1")
    assert event.append_batch_id is not None
    message = _message(
        content="assistant not admitted",
        seq=1,
        event_seq=1,
        append_batch_id=event.append_batch_id,
        role="assistant",
    )
    store = FakeStore([event])
    store.batches[event.event_id] = [_batch(event, [message])]
    service = FakeService()
    worker = _worker(store, service, config, event_types=["ingest"])

    await worker.poll_once()
    assert store.noop_finalized == [event.event_id]
    assert store.mark_calls == []
    assert service.ensure_calls == 0
    assert service.add_calls == []


async def test_rebuild_uses_snapshot_only_and_maps_noop_batches():
    timeline: list[str] = []
    config = _config(ingest_max_chars=4)
    event = _event(config, event_id="rebuild-1", event_type="rebuild", generation=2)
    first = ChatMemoryReplayBatch(
        append_batch_id="batch-real",
        project_event_seq=1,
        memory_reference_time="2026-07-15T00:00:00+00:00",
        session_id="sess_a",
        messages=[
            _message(
                content="abcdef",
                seq=1,
                event_seq=1,
                append_batch_id="batch-real",
            )
        ],
    )
    second = ChatMemoryReplayBatch(
        append_batch_id="batch-noop",
        project_event_seq=2,
        memory_reference_time="2026-07-15T00:00:01+00:00",
        session_id="sess_a",
        messages=[
            _message(
                content="not admitted",
                seq=2,
                event_seq=2,
                append_batch_id="batch-noop",
                role="assistant",
            )
        ],
    )
    snapshot = ChatMemoryRebuildSnapshot(
        event_id=event.event_id,
        user_id=event.user_id,
        project_id=event.project_id,
        generation=event.generation,
        graph_group_id=event.graph_group_id,
        config_fingerprint=config.extraction_fingerprint(),
        group_state_version=1,
        snapshot_cutoff=2,
        replay_batches=[first, second],
        batch_count=2,
        message_count=2,
        byte_count=17,
        snapshot_digest="digest",
        ingest_max_chars=4,
        graph_store_fingerprint=config.graph_store_fingerprint(),
    )
    store = FakeStore([event], timeline=timeline)
    store.snapshots[event.event_id] = snapshot
    targets = _rebuild_targets(
        event,
        "group-active",
        "group-retired",
        "group-abandoned",
    )
    store.rebuild_targets[event.event_id] = targets
    service = FakeService(timeline=timeline)
    worker = _worker(store, service, config, event_types=["rebuild"])

    await worker.poll_once()
    assert service.clear_calls == [list(targets.group_ids)]
    assert {
        event.graph_group_id,
        "group-active",
        "group-retired",
        "group-abandoned",
        chat_memory_legacy_graph_group_id(event.user_id, event.project_id),
    } == set(service.clear_calls[0])
    assert timeline.index("prepare-rebuild") < timeline.index("prepare-targets")
    assert timeline.index("prepare-targets") < timeline.index("mark")
    assert timeline.index("mark") < timeline.index("clear") < timeline.index("add")
    assert len(service.add_calls) == 1
    assert service.add_calls[0]["episode_body"] == "user: abcd…[truncated]"
    mappings = store.rebuild_finalized[0][1]
    assert [mapping.append_batch_id for mapping in mappings] == [
        "batch-real",
        "batch-noop",
    ]
    assert mappings[0].episode_uuid == "episode-1"
    assert mappings[1].episode_uuid.startswith("noop_")
    assert store.rebuild_finalize_targets == [(targets, targets.group_ids)]


async def test_first_rebuild_without_old_generation_clears_target_and_legacy():
    config = _config()
    event = _event(
        config,
        event_id="rebuild-first",
        event_type="rebuild",
        generation=1,
    )
    store = FakeStore([event])
    store.snapshots[event.event_id] = ChatMemoryRebuildSnapshot(
        event_id=event.event_id,
        user_id=event.user_id,
        project_id=event.project_id,
        generation=event.generation,
        graph_group_id=event.graph_group_id,
        config_fingerprint=config.extraction_fingerprint(),
        group_state_version=1,
        snapshot_cutoff=0,
        replay_batches=[],
        batch_count=0,
        message_count=0,
        byte_count=0,
        snapshot_digest="digest",
        graph_store_fingerprint=config.graph_store_fingerprint(),
    )
    targets = _rebuild_targets(event)
    store.rebuild_targets[event.event_id] = targets
    service = FakeService()
    worker = _worker(store, service, config, event_types=["rebuild"])

    await worker.poll_once()

    assert service.clear_calls == [list(targets.group_ids)]
    assert set(targets.group_ids) == {
        event.graph_group_id,
        chat_memory_legacy_graph_group_id(event.user_id, event.project_id),
    }
    assert store.rebuild_finalize_targets == [(targets, targets.group_ids)]


@pytest.mark.parametrize("event_type", ["ingest", "rebuild"])
async def test_unknown_ingest_or_rebuild_escalates_generation(event_type):
    config = _config()
    event = _event(config, event_id=f"unknown-{event_type}", event_type=event_type)
    store = FakeStore([event])
    service = FakeService()
    if event_type == "ingest":
        assert event.append_batch_id is not None
        store.batches[event.event_id] = [
            _batch(
                event,
                [
                    _message(
                        content="payload",
                        seq=1,
                        event_seq=1,
                        append_batch_id=event.append_batch_id,
                    )
                ],
            )
        ]
        service.add_error = RuntimeError("connection lost")
    else:
        store.snapshots[event.event_id] = ChatMemoryRebuildSnapshot(
            event_id=event.event_id,
            user_id=event.user_id,
            project_id=event.project_id,
            generation=event.generation,
            graph_group_id=event.graph_group_id,
            config_fingerprint=config.extraction_fingerprint(),
            group_state_version=1,
            snapshot_cutoff=0,
            replay_batches=[],
            batch_count=0,
            message_count=0,
            byte_count=0,
            snapshot_digest="digest",
            graph_store_fingerprint=config.graph_store_fingerprint(),
        )
        service.clear_error = RuntimeError("clear outcome unknown")
    worker = _worker(store, service, config, event_types=[event_type])

    await worker.poll_once()
    assert store.unknown_escalations == [event.event_id]
    assert service.invalidated == [service.graphiti]


async def test_purge_unknown_clear_retries_same_event_final_sweep():
    config = _config()
    event = _event(config, event_id="purge-1", event_type="purge")
    store = FakeStore([event])
    store.targets[event.event_id] = ChatMemoryPurgeTargetSet(
        event_id=event.event_id,
        user_id=event.user_id,
        project_id=event.project_id,
        logical_group_id="logical",
        group_ids=("physical-a", "legacy"),
    )
    service = FakeService()
    service.clear_error = RuntimeError("timeout after clear")
    worker = _worker(store, service, config, event_types=["purge"])

    await worker.poll_once()
    assert service.clear_calls == [["physical-a", "legacy"]]
    assert store.purge_unknown_retries == [event.event_id]
    assert store.unknown_escalations == []


async def test_purge_success_clears_explicit_targets_and_finalizes_atomically():
    config = _config()
    event = _event(config, event_id="purge-success", event_type="purge")
    store = FakeStore([event])
    targets = ChatMemoryPurgeTargetSet(
        event_id=event.event_id,
        user_id=event.user_id,
        project_id=event.project_id,
        logical_group_id="logical",
        group_ids=("active", "abandoned", "legacy"),
    )
    store.targets[event.event_id] = targets
    service = FakeService()
    worker = _worker(store, service, config, event_types=["purge"])

    await worker.poll_once()
    assert service.clear_calls == [list(targets.group_ids)]
    assert store.purge_finalized == [event.event_id]
    assert store.purge_unknown_retries == []


async def test_purge_claim_allows_old_extraction_on_same_graph_store():
    config = _config()
    event = _event(
        config,
        event_id="purge-old-extraction",
        event_type="purge",
        fingerprint="chat-memory-extraction:v0:sha256:" + "0" * 64,
    )
    store = FakeStore([event])
    store.targets[event.event_id] = ChatMemoryPurgeTargetSet(
        event_id=event.event_id,
        user_id=event.user_id,
        project_id=event.project_id,
        logical_group_id="logical",
        group_ids=("physical-old", "legacy"),
    )
    service = FakeService()
    worker = _worker(store, service, config, event_types=["purge"])

    assert await worker.poll_once() is event
    assert store.purge_finalized == [event.event_id]
    assert service.clear_calls == [["physical-old", "legacy"]]
    assert (
        store.claim_calls[0]["runtime_graph_store_fingerprint"]
        == config.graph_store_fingerprint()
    )


@pytest.mark.parametrize("event_type", ["ingest", "rebuild", "purge"])
async def test_graph_success_finalizer_failure_immediately_transitions_unknown(
    event_type,
):
    config = _config()
    event = _event(
        config,
        event_id=f"finalizer-{event_type}",
        event_type=event_type,
    )
    store = FakeStore([event])
    service = FakeService()
    store.finalizer_errors[event_type] = RuntimeError("database commit response lost")
    if event_type == "ingest":
        assert event.append_batch_id is not None
        store.batches[event.event_id] = [
            _batch(
                event,
                [
                    _message(
                        content="payload",
                        seq=1,
                        event_seq=1,
                        append_batch_id=event.append_batch_id,
                    )
                ],
            )
        ]
    elif event_type == "rebuild":
        store.snapshots[event.event_id] = ChatMemoryRebuildSnapshot(
            event_id=event.event_id,
            user_id=event.user_id,
            project_id=event.project_id,
            generation=event.generation,
            graph_group_id=event.graph_group_id,
            config_fingerprint=config.extraction_fingerprint(),
            group_state_version=1,
            snapshot_cutoff=0,
            replay_batches=[],
            batch_count=0,
            message_count=0,
            byte_count=0,
            snapshot_digest="digest",
            graph_store_fingerprint=config.graph_store_fingerprint(),
        )
    else:
        store.targets[event.event_id] = ChatMemoryPurgeTargetSet(
            event_id=event.event_id,
            user_id=event.user_id,
            project_id=event.project_id,
            logical_group_id="logical",
            group_ids=("physical",),
        )
    worker = _worker(store, service, config, event_types=[event_type])

    await worker.poll_once()

    assert service.invalidated == [service.graphiti]
    if event_type == "purge":
        assert store.purge_unknown_retries == [event.event_id]
        assert store.unknown_escalations == []
    else:
        assert store.unknown_escalations == [event.event_id]


async def test_rebuild_timeout_is_one_budget_for_clear_and_all_adds():
    config = _config(worker_side_effect_timeout_seconds=0.12)
    event = _event(config, event_id="rebuild-total-timeout", event_type="rebuild")
    batches = []
    for event_seq in (1, 2):
        batch_id = f"batch-{event_seq}"
        batches.append(
            ChatMemoryReplayBatch(
                append_batch_id=batch_id,
                project_event_seq=event_seq,
                memory_reference_time="2026-07-15T00:00:00+00:00",
                session_id="sess_a",
                messages=[
                    _message(
                        content=f"payload-{event_seq}",
                        seq=event_seq,
                        event_seq=event_seq,
                        append_batch_id=batch_id,
                    )
                ],
            )
        )
    store = FakeStore([event])
    store.snapshots[event.event_id] = ChatMemoryRebuildSnapshot(
        event_id=event.event_id,
        user_id=event.user_id,
        project_id=event.project_id,
        generation=event.generation,
        graph_group_id=event.graph_group_id,
        config_fingerprint=config.extraction_fingerprint(),
        group_state_version=1,
        snapshot_cutoff=2,
        replay_batches=batches,
        batch_count=2,
        message_count=2,
        byte_count=18,
        snapshot_digest="digest",
        graph_store_fingerprint=config.graph_store_fingerprint(),
    )
    service = FakeService()
    service.add_delay = 0.08
    worker = _worker(store, service, config, event_types=["rebuild"])

    await worker.poll_once()

    assert len(service.add_calls) == 2
    assert store.rebuild_finalized == []
    assert store.unknown_escalations == [event.event_id]
    assert service.invalidated == [service.graphiti]


async def test_external_timeout_invalidates_backend_and_escalates():
    config = _config(worker_side_effect_timeout_seconds=0.01)
    event = _event(config, event_id="timeout-1")
    assert event.append_batch_id is not None
    store = FakeStore([event])
    store.batches[event.event_id] = [
        _batch(
            event,
            [
                _message(
                    content="payload",
                    seq=1,
                    event_seq=1,
                    append_batch_id=event.append_batch_id,
                )
            ],
        )
    ]
    service = FakeService()
    service.add_delay = 0.1
    worker = _worker(store, service, config, event_types=["ingest"])

    await worker.poll_once()
    assert store.unknown_escalations == [event.event_id]
    assert service.invalidated == [service.graphiti]


async def test_cancelled_side_effect_bounds_shielded_unknown_transition():
    config = _config(worker_shutdown_timeout_seconds=0.02)
    event = _event(config, event_id="cancel-bounded")
    assert event.append_batch_id is not None
    store = FakeStore([event])
    store.batches[event.event_id] = [
        _batch(
            event,
            [
                _message(
                    content="payload",
                    seq=1,
                    event_seq=1,
                    append_batch_id=event.append_batch_id,
                )
            ],
        )
    ]
    store.unknown_transition_gate = asyncio.Event()
    service = FakeService()
    service.add_delay = 10
    worker = _worker(store, service, config, event_types=["ingest"])
    task = asyncio.create_task(worker.poll_once())
    while not service.add_calls:
        await asyncio.sleep(0)
    started = asyncio.get_running_loop().time()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    elapsed = asyncio.get_running_loop().time() - started
    assert elapsed < 0.2
    assert store.unknown_transition_started.is_set()
    assert service.invalidated == [service.graphiti]


async def test_runtime_fingerprint_mismatch_never_touches_backend():
    config = _config()
    event = _event(
        config,
        event_id="mismatch-1",
        fingerprint="chat-memory-extraction:v1:sha256:" + "0" * 64,
    )
    store = FakeStore([event])
    store.ignore_claim_fingerprint = True
    service = FakeService()
    worker = _worker(store, service, config, event_types=["ingest"])

    await worker.poll_once()
    assert store.known_failures == [
        (event.event_id, "runtime_fingerprint_mismatch")
    ]
    assert service.ensure_calls == 0
    assert store.mark_calls == []


async def test_rebuild_stale_fence_stops_before_second_episode_and_activation():
    config = _config()
    event = _event(config, event_id="stale-1", event_type="rebuild", generation=2)
    batches = []
    for event_seq in (1, 2):
        batch_id = f"batch-{event_seq}"
        batches.append(
            ChatMemoryReplayBatch(
                append_batch_id=batch_id,
                project_event_seq=event_seq,
                memory_reference_time="2026-07-15T00:00:00+00:00",
                session_id="sess_a",
                messages=[
                    _message(
                        content=f"payload-{event_seq}",
                        seq=event_seq,
                        event_seq=event_seq,
                        append_batch_id=batch_id,
                    )
                ],
            )
        )
    store = FakeStore([event])
    store.marker_statuses = ["running", "running", "superseded"]
    store.snapshots[event.event_id] = ChatMemoryRebuildSnapshot(
        event_id=event.event_id,
        user_id=event.user_id,
        project_id=event.project_id,
        generation=event.generation,
        graph_group_id=event.graph_group_id,
        config_fingerprint=config.extraction_fingerprint(),
        group_state_version=1,
        snapshot_cutoff=2,
        replay_batches=batches,
        batch_count=2,
        message_count=2,
        byte_count=18,
        snapshot_digest="digest",
        graph_store_fingerprint=config.graph_store_fingerprint(),
    )
    service = FakeService()
    worker = _worker(store, service, config, event_types=["rebuild"])

    await worker.poll_once()
    assert len(service.add_calls) == 1
    assert store.rebuild_finalized == []
    assert store.unknown_escalations == []


async def test_store_group_guard_serializes_same_group_but_allows_other_groups():
    config = _config()
    same_events = [
        _event(config, event_id=f"same-{index}", event_seq=index)
        for index in (1, 2)
    ]
    same_store = FakeStore(same_events)
    same_service = FakeService()
    same_active = 0
    same_max = 0

    async def same_handler(_event):
        nonlocal same_active, same_max
        same_active += 1
        same_max = max(same_max, same_active)
        await asyncio.sleep(0.02)
        same_active -= 1

    same_worker = _worker(
        same_store,
        same_service,
        config,
        handlers={"ingest": same_handler},
    )
    await asyncio.gather(same_worker.poll_once(), same_worker.poll_once())
    assert same_max == 1
    assert len(same_store.guard_calls) == 2

    other_events = [
        _event(
            config,
            event_id=f"other-{index}",
            user_id=f"usr_{index}",
            project_id=f"proj_{index}",
        )
        for index in (1, 2)
    ]
    other_store = FakeStore(other_events)
    other_active = 0
    other_max = 0

    async def other_handler(_event):
        nonlocal other_active, other_max
        other_active += 1
        other_max = max(other_max, other_active)
        await asyncio.sleep(0.02)
        other_active -= 1

    other_worker = _worker(
        other_store,
        FakeService(),
        config,
        handlers={"ingest": other_handler},
    )
    await asyncio.gather(other_worker.poll_once(), other_worker.poll_once())
    assert other_max == 2


async def test_recovery_delegates_owner_guard_to_store():
    config = _config()
    event = _event(config, event_id="stale-recovery")
    event.status = "running"
    event.claim_token = "stale-token"
    store = FakeStore()
    store.stale_events = [event]
    worker = _worker(store, FakeService(), config, event_types=["ingest"])

    assert await worker.recover_once() == 1
    assert store.recovery_calls == [event.event_id]
    assert store.guard_calls == []


async def test_start_stop_and_nudge_wake_bounded_consumers():
    config = _config(
        ingest_concurrency=2,
        worker_poll_interval_seconds=10,
        worker_recovery_interval_seconds=10,
    )
    store = FakeStore()
    worker = _worker(store, FakeService(), config, event_types=["ingest"])

    worker.start()
    for _ in range(50):
        if store.claim_count >= 2:
            break
        await asyncio.sleep(0.01)
    baseline = store.claim_count
    assert worker.running is True
    assert baseline >= 2

    worker.nudge()
    for _ in range(50):
        if store.claim_count > baseline:
            break
        await asyncio.sleep(0.01)
    assert store.claim_count > baseline

    await worker.stop()
    assert worker.running is False
    assert worker._consumer_tasks == []
    assert worker._recovery_task is None


async def test_stop_returns_after_shutdown_budget_when_task_ignores_cancel():
    config = _config(worker_shutdown_timeout_seconds=0.02)
    worker = _worker(FakeStore(), FakeService(), config, event_types=["ingest"])
    release = asyncio.Event()

    async def stubborn_consumer():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()

    task = asyncio.create_task(stubborn_consumer())
    await asyncio.sleep(0)
    worker._consumer_tasks = [task]
    started = asyncio.get_running_loop().time()

    await worker.stop()

    elapsed = asyncio.get_running_loop().time() - started
    assert elapsed < 0.2
    assert task.done() is False
    assert worker.running is False
    release.set()
    await asyncio.wait_for(task, timeout=0.2)


async def test_worker_ingest_composes_with_real_sqlite_store(tmp_path):
    from lightrag.api.metadata_store import SQLiteMetadataStore
    from tests.api.test_chat_memory_store_phase1 import _create_chat, _message

    config = _config()
    store = SQLiteMetadataStore(tmp_path / "worker-integration.sqlite3")
    await store.initialize()
    store._test_chat_memory_user_ids = []  # type: ignore[attr-defined]
    try:
        user, project, session = await _create_chat(store)
        await store.append_chat_messages_with_memory(
            [_message(user.id, project.id, session.id, "durable payload")],
            config_fingerprint=config.extraction_fingerprint(),
            graph_store_fingerprint=config.graph_store_fingerprint(),
        )
        service = FakeService()
        worker = ChatMemoryWorker(
            store,
            service,  # type: ignore[arg-type]
            config,
            worker_id="sqlite-worker",
            event_types=["ingest"],
            retry_delay_seconds=0,
        )

        claimed = await worker.poll_once()
        assert claimed is not None
        persisted = await store.get_chat_memory_event(claimed.event_id)
        assert persisted is not None and persisted.status == "succeeded"
        mappings = await store.list_chat_memory_episodes_for_session(
            user.id, project.id, session.id
        )
        assert [mapping.episode_uuid for mapping in mappings] == ["episode-1"]
    finally:
        await store.close()
