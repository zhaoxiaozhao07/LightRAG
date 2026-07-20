"""Durable FIFO worker for enterprise Chat Memory outbox events."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from lightrag.api.chat_memory_service import ChatMemoryConfig, ChatMemoryService
from lightrag.api.metadata_store import (
    ChatMemoryOutboxEventRecord,
    ChatMemoryReplayMappingInput,
    _chat_memory_canonical_episode_payload,
    _chat_memory_noop_episode_uuid,
    chat_memory_logical_group_id,
)
from lightrag.utils import logger

ChatMemoryEventHandler = Callable[[ChatMemoryOutboxEventRecord], Awaitable[None]]
_SUPPORTED_EVENT_TYPES = ("ingest", "rebuild", "purge")


def _reference_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class ChatMemoryWorker:
    """Claim and execute durable Chat Memory events with bounded concurrency."""

    def __init__(
        self,
        metadata_store: Any,
        service: ChatMemoryService,
        config: ChatMemoryConfig,
        *,
        worker_id: str | None = None,
        event_types: Sequence[str] | None = None,
        handlers: Mapping[str, ChatMemoryEventHandler] | None = None,
        concurrency: int | None = None,
        retry_delay_seconds: float = 1.0,
        max_attempts: int = 3,
        stale_after_seconds: float | None = None,
    ) -> None:
        self._metadata_store = metadata_store
        self._service = service
        self._config = config
        self._runtime_extraction_fingerprint = config.extraction_fingerprint()
        self._runtime_graph_store_fingerprint = config.graph_store_fingerprint()
        self._worker_id = worker_id or (
            f"cmw_{os.getpid()}_{uuid4().hex[:12]}"
        )
        default_handlers: dict[str, ChatMemoryEventHandler] = {
            "ingest": self._handle_ingest,
            "rebuild": self._handle_rebuild,
            "purge": self._handle_purge,
        }
        if handlers is not None:
            selected = dict(handlers)
        else:
            selected_types = tuple(event_types or _SUPPORTED_EVENT_TYPES)
            selected = {
                event_type: default_handlers[event_type]
                for event_type in selected_types
                if event_type in default_handlers
            }
            unknown = set(selected_types).difference(default_handlers)
            if unknown:
                raise ValueError(
                    "Unsupported Chat Memory worker event types: "
                    + ", ".join(sorted(unknown))
                )
        if not selected:
            raise ValueError("Chat Memory worker requires at least one event handler")
        unknown_handlers = set(selected).difference(_SUPPORTED_EVENT_TYPES)
        if unknown_handlers:
            raise ValueError(
                "Unsupported Chat Memory worker handlers: "
                + ", ".join(sorted(unknown_handlers))
            )
        self._handlers = selected
        self._event_types = tuple(selected)
        self._concurrency = max(
            1,
            min(
                64,
                int(
                    config.ingest_concurrency
                    if concurrency is None
                    else concurrency
                ),
            ),
        )
        self._poll_interval = max(
            0.05, float(config.worker_poll_interval_seconds)
        )
        self._recovery_interval = max(
            0.0, float(config.worker_recovery_interval_seconds)
        )
        self._side_effect_timeout = max(
            0.01, float(config.worker_side_effect_timeout_seconds)
        )
        self._shutdown_timeout = max(
            0.01, float(config.worker_shutdown_timeout_seconds)
        )
        self._retry_delay = max(0.0, float(retry_delay_seconds))
        self._max_attempts = max(1, int(max_attempts))
        self._stale_after = max(
            0.0,
            float(
                self._side_effect_timeout
                if stale_after_seconds is None
                else stale_after_seconds
            ),
        )
        self._stop_event = asyncio.Event()
        self._nudge_event = asyncio.Event()
        self._consumer_tasks: list[asyncio.Task[None]] = []
        self._recovery_task: asyncio.Task[None] | None = None

    @property
    def runtime_fingerprint(self) -> str:
        """Compatibility alias for the extraction fingerprint."""

        return self._runtime_extraction_fingerprint

    @property
    def extraction_fingerprint(self) -> str:
        return self._runtime_extraction_fingerprint

    @property
    def graph_store_fingerprint(self) -> str:
        return self._runtime_graph_store_fingerprint

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def event_types(self) -> tuple[str, ...]:
        return self._event_types

    @property
    def running(self) -> bool:
        return any(not task.done() for task in self._consumer_tasks)

    async def poll_once(self) -> ChatMemoryOutboxEventRecord | None:
        """Claim and fully execute one event; deterministic test entry point."""

        try:
            event = await self._metadata_store.claim_next_chat_memory_event(
                self._runtime_extraction_fingerprint,
                runtime_graph_store_fingerprint=(
                    self._runtime_graph_store_fingerprint
                ),
                worker_id=self._worker_id,
                event_types=self._event_types,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - polling must remain alive
            logger.error("ChatMemoryWorker claim failed: %s", exc)
            return None
        if event is None:
            return None
        try:
            await self._execute_claimed(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - one event must not stop consumers
            logger.error(
                "ChatMemoryWorker event %s execution failed: %s",
                event.event_id,
                exc,
            )
        return event

    async def _execute_claimed(self, claimed: ChatMemoryOutboxEventRecord) -> None:
        if not claimed.claim_token:
            logger.error("Claimed Chat Memory event %s has no claim token", claimed.event_id)
            return
        logical_group_id = chat_memory_logical_group_id(
            claimed.user_id, claimed.project_id
        )
        async with self._metadata_store.chat_memory_group_execution_guard(
            logical_group_id
        ) as acquired:
            if not acquired:
                return
            state = await self._metadata_store.get_chat_memory_execution_state(
                claimed.event_id
            )
            if state is None:
                return
            event = state.event
            if (
                event.status != "running"
                or event.claim_token != claimed.claim_token
                or event.claimed_by != claimed.claimed_by
            ):
                return
            if event.graph_store_fingerprint != self._runtime_graph_store_fingerprint:
                await self._fail_before_side_effect(
                    event,
                    error_code="runtime_graph_store_fingerprint_mismatch",
                    error_message=(
                        "Worker graph-store fingerprint does not match claimed event"
                    ),
                )
                return
            if (
                event.event_type != "purge"
                and event.config_fingerprint
                != self._runtime_extraction_fingerprint
            ):
                await self._fail_before_side_effect(
                    event,
                    error_code="runtime_fingerprint_mismatch",
                    error_message=(
                        "Worker extraction fingerprint does not match claimed event"
                    ),
                )
                return
            handler = self._handlers.get(event.event_type)
            if handler is None:
                await self._fail_before_side_effect(
                    event,
                    error_code="unsupported_event_type",
                    error_message=f"No handler for event type {event.event_type}",
                )
                return
            await handler(event)

    async def _fail_before_side_effect(
        self,
        event: ChatMemoryOutboxEventRecord,
        *,
        error_code: str,
        error_message: str,
    ) -> None:
        if not event.claim_token:
            return
        try:
            if event.event_type == "purge":
                await self._metadata_store.fail_chat_memory_purge_before_side_effect(
                    event.event_id,
                    event.claim_token,
                    self._runtime_extraction_fingerprint,
                    runtime_graph_store_fingerprint=(
                        self._runtime_graph_store_fingerprint
                    ),
                    error_code=error_code,
                    error_message=error_message,
                    retry_delay_seconds=self._retry_delay,
                    max_attempts=self._max_attempts,
                )
            else:
                await self._metadata_store.fail_chat_memory_event_before_side_effect(
                    event.event_id,
                    event.claim_token,
                    self._runtime_extraction_fingerprint,
                    runtime_graph_store_fingerprint=(
                        self._runtime_graph_store_fingerprint
                    ),
                    error_code=error_code,
                    error_message=error_message,
                    retry_delay_seconds=self._retry_delay,
                    max_attempts=self._max_attempts,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - best-effort transition
            logger.error(
                "ChatMemoryWorker could not record known failure for %s: %s",
                event.event_id,
                exc,
            )

    async def _invalidate_backend_quietly(self, graphiti: Any) -> None:
        try:
            await self._service.invalidate_backend(graphiti)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chat Memory backend invalidation failed: %s", exc)

    async def _handle_unknown_graph_outcome(
        self,
        event: ChatMemoryOutboxEventRecord,
        graphiti: Any,
        exc: BaseException,
    ) -> None:
        if not event.claim_token:
            return
        await self._invalidate_backend_quietly(graphiti)
        message = f"{type(exc).__name__}: {exc}"
        try:
            if event.event_type == "purge":
                await self._metadata_store.retry_chat_memory_purge_after_unknown_clear(
                    event.event_id,
                    event.claim_token,
                    self._runtime_extraction_fingerprint,
                    runtime_graph_store_fingerprint=(
                        self._runtime_graph_store_fingerprint
                    ),
                    retry_delay_seconds=self._retry_delay,
                    error_code="purge_clear_outcome_unknown",
                    error_message=message,
                )
            else:
                await self._metadata_store.escalate_chat_memory_event_unknown(
                    event.event_id,
                    event.claim_token,
                    self._runtime_extraction_fingerprint,
                    runtime_graph_store_fingerprint=(
                        self._runtime_graph_store_fingerprint
                    ),
                    error_code="side_effect_outcome_unknown",
                    error_message=message,
                    actor_user_id=event.actor_user_id,
                    actor_tenant_id=event.actor_tenant_id,
                )
        except asyncio.CancelledError:
            raise
        except Exception as transition_exc:  # noqa: BLE001
            logger.error(
                "ChatMemoryWorker could not transition unknown event %s: %s",
                event.event_id,
                transition_exc,
            )

    async def _handle_unknown_shielded(
        self,
        event: ChatMemoryOutboxEventRecord,
        graphiti: Any,
        exc: BaseException,
    ) -> None:
        task = asyncio.create_task(
            self._handle_unknown_graph_outcome(event, graphiti, exc)
        )
        try:
            await asyncio.wait_for(
                asyncio.shield(task), timeout=self._shutdown_timeout
            )
        except asyncio.TimeoutError:
            logger.error(
                "ChatMemoryWorker unknown transition for %s exceeded "
                "shutdown budget %.3fs; durable recovery will continue it",
                event.event_id,
                self._shutdown_timeout,
            )
            task.cancel()
            task.add_done_callback(self._consume_task_result)
        except asyncio.CancelledError:
            task.cancel()
            task.add_done_callback(self._consume_task_result)

    @staticmethod
    def _consume_task_result(task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            return
        try:
            task.exception()
        except asyncio.CancelledError:
            pass

    async def _mark_side_effect(
        self,
        event: ChatMemoryOutboxEventRecord,
        *,
        suppress_errors: bool = True,
    ) -> ChatMemoryOutboxEventRecord | None:
        if not event.claim_token:
            return None
        try:
            started = (
                await self._metadata_store.mark_chat_memory_event_side_effect_started(
                    event.event_id,
                    event.claim_token,
                    self._runtime_extraction_fingerprint,
                    runtime_graph_store_fingerprint=(
                        self._runtime_graph_store_fingerprint
                    ),
                    fingerprint_retry_delay_seconds=self._retry_delay,
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not suppress_errors:
                raise
            logger.error(
                "ChatMemoryWorker could not mark side effect for %s: %s",
                event.event_id,
                exc,
            )
            return None
        if (
            started.status != "running"
            or started.claim_token != event.claim_token
            or started.side_effect_started_at is None
        ):
            return None
        return started

    async def _handle_ingest(self, event: ChatMemoryOutboxEventRecord) -> None:
        assert event.claim_token is not None
        try:
            batches = (
                await self._metadata_store.list_admitted_chat_memory_replay_batches(
                    event.user_id,
                    event.project_id,
                    through_event_seq=event.event_seq,
                    after_event_seq=event.event_seq - 1,
                    limit=2,
                )
            )
            if (
                len(batches) != 1
                or batches[0].project_event_seq != event.event_seq
                or batches[0].append_batch_id != event.append_batch_id
            ):
                raise RuntimeError("Ingest event source batch is missing or inconsistent")
            batch = batches[0]
            payload = _chat_memory_canonical_episode_payload(
                batch.messages,
                ingest_max_chars=self._config.ingest_max_chars,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail_before_side_effect(
                event,
                error_code="ingest_preparation_failed",
                error_message=f"{type(exc).__name__}: {exc}",
            )
            return

        if not payload["messages"]:
            try:
                await self._metadata_store.finalize_chat_memory_ingest_noop(
                    event.event_id,
                    event.claim_token,
                    self._runtime_extraction_fingerprint,
                    runtime_graph_store_fingerprint=(
                        self._runtime_graph_store_fingerprint
                    ),
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Chat Memory no-op finalize failed: %s", exc)
            return

        first_seq = min(message.seq for message in batch.messages)
        last_seq = max(message.seq for message in batch.messages)
        try:
            async with self._service.backend_lease() as graphiti:
                if await self._mark_side_effect(event) is None:
                    return
                try:
                    async with asyncio.timeout(self._side_effect_timeout):
                        result = await self._service.add_episode(
                            graphiti,
                            name=f"{batch.session_id}:{first_seq}-{last_seq}",
                            episode_body=payload["episode_body"],
                            source_description="enterprise chat durable outbox",
                            reference_time=_reference_time(
                                batch.memory_reference_time
                            ),
                            group_id=event.graph_group_id,
                        )
                    episode_uuid = getattr(
                        getattr(result, "episode", None), "uuid", None
                    )
                    if not episode_uuid:
                        raise RuntimeError(
                            "Graphiti add_episode returned no episode UUID"
                        )
                    await self._metadata_store.finalize_chat_memory_ingest(
                        event.event_id,
                        event.claim_token,
                        self._runtime_extraction_fingerprint,
                        episode_uuid=str(episode_uuid),
                        runtime_graph_store_fingerprint=(
                            self._runtime_graph_store_fingerprint
                        ),
                    )
                except asyncio.CancelledError as exc:
                    await self._handle_unknown_shielded(event, graphiti, exc)
                    raise
                except Exception as exc:
                    await self._handle_unknown_graph_outcome(event, graphiti, exc)
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail_before_side_effect(
                event,
                error_code="backend_unavailable_before_side_effect",
                error_message=f"{type(exc).__name__}: {exc}",
            )

    async def _recheck_side_effect_fence(
        self, event: ChatMemoryOutboxEventRecord
    ) -> bool:
        return (
            await self._mark_side_effect(event, suppress_errors=False) is not None
        )

    async def _handle_rebuild(self, event: ChatMemoryOutboxEventRecord) -> None:
        assert event.claim_token is not None
        try:
            snapshot = await self._metadata_store.prepare_chat_memory_rebuild_snapshot(
                event.event_id,
                event.claim_token,
                self._runtime_extraction_fingerprint,
                max_messages=self._config.rebuild_max_messages,
                max_bytes=self._config.rebuild_max_bytes,
                ingest_max_chars=self._config.ingest_max_chars,
                runtime_graph_store_fingerprint=(
                    self._runtime_graph_store_fingerprint
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail_before_side_effect(
                event,
                error_code="rebuild_snapshot_preparation_failed",
                error_message=f"{type(exc).__name__}: {exc}",
            )
            return
        if snapshot is None:
            return
        try:
            targets = await self._metadata_store.prepare_chat_memory_rebuild_targets(
                event.event_id,
                event.claim_token,
                self._runtime_extraction_fingerprint,
                runtime_graph_store_fingerprint=(
                    self._runtime_graph_store_fingerprint
                ),
            )
            if targets is not None and not targets.group_ids:
                raise RuntimeError("Rebuild target inventory is empty")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail_before_side_effect(
                event,
                error_code="rebuild_target_preparation_failed",
                error_message=f"{type(exc).__name__}: {exc}",
            )
            return
        if targets is None:
            return
        try:
            async with self._service.backend_lease() as graphiti:
                if await self._mark_side_effect(event) is None:
                    return
                try:
                    async with asyncio.timeout(self._side_effect_timeout):
                        await self._service.clear_graph_groups(
                            graphiti, list(targets.group_ids)
                        )
                        mappings: list[ChatMemoryReplayMappingInput] = []
                        for batch in snapshot.replay_batches:
                            if not await self._recheck_side_effect_fence(event):
                                return
                            payload = _chat_memory_canonical_episode_payload(
                                batch.messages,
                                ingest_max_chars=snapshot.ingest_max_chars,
                            )
                            first_seq = min(
                                message.seq for message in batch.messages
                            )
                            last_seq = max(message.seq for message in batch.messages)
                            if not payload["messages"]:
                                episode_uuid = _chat_memory_noop_episode_uuid(
                                    event_id=event.event_id,
                                    generation=event.generation,
                                    append_batch_id=batch.append_batch_id,
                                )
                            else:
                                result = await self._service.add_episode(
                                    graphiti,
                                    name=(
                                        f"{batch.session_id}:{first_seq}-{last_seq}"
                                    ),
                                    episode_body=payload["episode_body"],
                                    source_description=(
                                        "enterprise chat rebuild snapshot"
                                    ),
                                    reference_time=_reference_time(
                                        batch.memory_reference_time
                                    ),
                                    group_id=snapshot.graph_group_id,
                                )
                                episode_uuid = getattr(
                                    getattr(result, "episode", None), "uuid", None
                                )
                                if not episode_uuid:
                                    raise RuntimeError(
                                        "Graphiti add_episode returned no episode UUID"
                                    )
                            mappings.append(
                                ChatMemoryReplayMappingInput(
                                    append_batch_id=batch.append_batch_id,
                                    project_event_seq=batch.project_event_seq,
                                    session_id=batch.session_id,
                                    first_seq=first_seq,
                                    last_seq=last_seq,
                                    episode_uuid=str(episode_uuid),
                                )
                            )
                        if not await self._recheck_side_effect_fence(event):
                            return
                    await self._metadata_store.finalize_chat_memory_rebuild(
                        event.event_id,
                        event.claim_token,
                        self._runtime_extraction_fingerprint,
                        snapshot,
                        mappings,
                        targets,
                        targets.group_ids,
                        runtime_graph_store_fingerprint=(
                            self._runtime_graph_store_fingerprint
                        ),
                    )
                except asyncio.CancelledError as exc:
                    await self._handle_unknown_shielded(event, graphiti, exc)
                    raise
                except Exception as exc:
                    await self._handle_unknown_graph_outcome(event, graphiti, exc)
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail_before_side_effect(
                event,
                error_code="backend_unavailable_before_side_effect",
                error_message=f"{type(exc).__name__}: {exc}",
            )

    async def _handle_purge(self, event: ChatMemoryOutboxEventRecord) -> None:
        assert event.claim_token is not None
        try:
            targets = await self._metadata_store.prepare_chat_memory_purge_targets(
                event.event_id,
                event.claim_token,
                self._runtime_extraction_fingerprint,
                runtime_graph_store_fingerprint=(
                    self._runtime_graph_store_fingerprint
                ),
            )
            if targets is not None and not targets.group_ids:
                raise RuntimeError("Purge target inventory is empty")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail_before_side_effect(
                event,
                error_code="purge_target_preparation_failed",
                error_message=f"{type(exc).__name__}: {exc}",
            )
            return
        if targets is None:
            return
        try:
            async with self._service.backend_lease() as graphiti:
                if await self._mark_side_effect(event) is None:
                    return
                try:
                    async with asyncio.timeout(self._side_effect_timeout):
                        await self._service.clear_graph_groups(
                            graphiti, list(targets.group_ids)
                        )
                    await self._metadata_store.finalize_chat_memory_purge(
                        event.event_id,
                        event.claim_token,
                        self._runtime_extraction_fingerprint,
                        targets,
                        targets.group_ids,
                        runtime_graph_store_fingerprint=(
                            self._runtime_graph_store_fingerprint
                        ),
                    )
                except asyncio.CancelledError as exc:
                    await self._handle_unknown_shielded(event, graphiti, exc)
                    raise
                except Exception as exc:
                    await self._handle_unknown_graph_outcome(event, graphiti, exc)
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail_before_side_effect(
                event,
                error_code="backend_unavailable_before_side_effect",
                error_message=f"{type(exc).__name__}: {exc}",
            )

    async def recover_once(self, *, limit: int = 100) -> int:
        """Delegate stale-claim recovery to the store's owner-guard primitive."""

        try:
            stale_events = (
                await self._metadata_store.list_stale_chat_memory_running_events(
                    stale_after_seconds=self._stale_after,
                    limit=limit,
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("ChatMemoryWorker stale-event listing failed: %s", exc)
            return 0
        recovered = 0
        for event in stale_events:
            if not event.claim_token:
                continue
            try:
                result = await self._metadata_store.recover_stale_chat_memory_event(
                    event.event_id,
                    event.claim_token,
                    self._runtime_extraction_fingerprint,
                    runtime_graph_store_fingerprint=(
                        self._runtime_graph_store_fingerprint
                    ),
                    retry_delay_seconds=self._retry_delay,
                    max_attempts=self._max_attempts,
                )
                if result is not None:
                    recovered += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "ChatMemoryWorker stale recovery failed for %s: %s",
                    event.event_id,
                    exc,
                )
        return recovered

    async def _consumer_loop(self, index: int) -> None:
        logger.info("ChatMemoryWorker consumer %d started", index)
        try:
            while not self._stop_event.is_set():
                event = await self.poll_once()
                if event is not None:
                    continue
                try:
                    await asyncio.wait_for(
                        self._nudge_event.wait(), timeout=self._poll_interval
                    )
                except asyncio.TimeoutError:
                    pass
                self._nudge_event.clear()
        finally:
            logger.info("ChatMemoryWorker consumer %d stopped", index)

    async def _recovery_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self._recovery_interval
                    )
                except asyncio.TimeoutError:
                    await self.recover_once()
        finally:
            logger.info("ChatMemoryWorker stale recovery stopped")

    def start(self) -> None:
        """Start bounded consumers and independent stale recovery (idempotent)."""

        if self.running:
            return
        self._stop_event.clear()
        self._nudge_event.clear()
        self._consumer_tasks = [
            asyncio.create_task(
                self._consumer_loop(index),
                name=f"chat-memory-worker-{index}",
            )
            for index in range(self._concurrency)
        ]
        if self._recovery_interval > 0:
            self._recovery_task = asyncio.create_task(
                self._recovery_loop(), name="chat-memory-worker-recovery"
            )

    def nudge(self) -> None:
        """Wake sleeping consumers after a request enqueues durable work."""

        self._nudge_event.set()

    async def stop(self) -> None:
        """Cancel consumers within a bounded durable-recovery shutdown budget."""

        self._stop_event.set()
        self._nudge_event.set()
        tasks = [*self._consumer_tasks]
        if self._recovery_task is not None:
            tasks.append(self._recovery_task)
        for task in tasks:
            task.cancel()
        if tasks:
            done, pending = await asyncio.wait(
                tasks, timeout=self._shutdown_timeout
            )
            for task in done:
                self._consume_task_result(task)
            if pending:
                logger.error(
                    "ChatMemoryWorker shutdown exceeded %.3fs with %d task(s) "
                    "still running; relying on durable stale-event recovery",
                    self._shutdown_timeout,
                    len(pending),
                )
                for task in pending:
                    task.add_done_callback(self._consume_task_result)
        self._consumer_tasks = []
        self._recovery_task = None
