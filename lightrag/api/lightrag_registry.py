from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol

from lightrag.utils import logger
from lightrag.api.kb_service import KnowledgeBaseRecord, KnowledgeBaseService


class LightRAGLike(Protocol):
    workspace: str

    async def finalize_storages(self) -> None: ...

@dataclass(slots=True)
class RegistryEntry:
    kb_id: str
    workspace: str
    rag: LightRAGLike
    config_version_id: str | None
    last_used: float


LightRAGBuilder = Callable[[KnowledgeBaseRecord], Awaitable[LightRAGLike]]
LightRAGFinalizer = Callable[[LightRAGLike], Awaitable[None]]


class DestructiveLockBusyError(RuntimeError):
    """Raised when a destructive job is already in flight for the KB."""


class LightRAGInstanceRegistry:
    """Per-KB LightRAG instance cache.

    The registry guarantees:

    - Single-flight initialization for the same ``kb_id``.
    - LRU eviction once ``max_entries`` is exceeded.
    - Idle TTL eviction (``idle_ttl_seconds``); ``None`` disables time-based
      reclaim (default for tests / low-traffic deployments).
    - Destructive job protection: while a destructive operation
      (``clear`` / hard delete / rebuild) holds the destructive lock for a
      KB, the registry refuses to evict, finalize, or rebuild that KB's
      instance.
    """

    def __init__(
        self,
        kb_service: KnowledgeBaseService,
        builder: LightRAGBuilder,
        finalizer: LightRAGFinalizer,
        *,
        max_entries: int | None = None,
        idle_ttl_seconds: float | None = None,
    ):
        self._kb_service = kb_service
        self._builder = builder
        self._finalizer = finalizer
        self._max_entries = max_entries if max_entries and max_entries > 0 else None
        self._idle_ttl_seconds = (
            idle_ttl_seconds if idle_ttl_seconds and idle_ttl_seconds > 0 else None
        )
        self._entries: OrderedDict[str, RegistryEntry] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}
        self._destructive_locks: dict[str, asyncio.Lock] = {}
        self._destructive_held: set[str] = set()
        self._locks_guard = asyncio.Lock()

    async def get(self, kb_id: str) -> LightRAGLike:
        entry = await self.get_entry(kb_id)
        return entry.rag

    async def get_entry(self, kb_id: str) -> RegistryEntry:
        record = await self._kb_service.get(kb_id)
        lock = await self._get_lock(record.id)
        async with lock:
            entry = self._entries.get(record.id)
            now = time.monotonic()
            if (
                entry is not None
                and entry.workspace == record.workspace
                and entry.config_version_id == record.active_config_version_id
                and not self._is_expired(entry, now)
            ):
                entry.last_used = now
                self._entries.move_to_end(record.id)
                return entry

            if entry is not None:
                if record.id in self._destructive_held:
                    # Refuse to rebuild while destructive job holds the lock —
                    # the destructive worker is responsible for the existing
                    # instance's lifecycle.
                    raise DestructiveLockBusyError(
                        f"Destructive job in flight for KB '{record.id}'"
                    )
                await self._safe_finalize(entry.rag)
                self._entries.pop(record.id, None)

            rag = await self._builder(record)
            entry = RegistryEntry(
                kb_id=record.id,
                workspace=record.workspace,
                rag=rag,
                config_version_id=record.active_config_version_id,
                last_used=now,
            )
            self._entries[record.id] = entry
            await self._enforce_capacity(protect_kb_id=record.id)
            return entry

    async def discard(self, kb_id: str) -> bool:
        if kb_id in self._destructive_held:
            raise DestructiveLockBusyError(
                f"Destructive job in flight for KB '{kb_id}'"
            )
        lock = await self._get_lock(kb_id)
        async with lock:
            entry = self._entries.pop(kb_id, None)
            if entry is None:
                return False
            await self._safe_finalize(entry.rag)
            return True

    async def reap_idle(self) -> int:
        """Evict entries past the idle TTL. Returns count evicted."""
        if self._idle_ttl_seconds is None:
            return 0
        now = time.monotonic()
        candidates: list[str] = []
        for kb_id, entry in self._entries.items():
            if kb_id in self._destructive_held:
                continue
            if self._is_expired(entry, now):
                candidates.append(kb_id)
        evicted = 0
        for kb_id in candidates:
            lock = await self._get_lock(kb_id)
            async with lock:
                entry = self._entries.get(kb_id)
                if entry is None or not self._is_expired(entry, time.monotonic()):
                    continue
                if kb_id in self._destructive_held:
                    continue
                await self._safe_finalize(entry.rag)
                self._entries.pop(kb_id, None)
                evicted += 1
        return evicted

    @asynccontextmanager
    async def destructive_lock(self, kb_id: str):
        """Acquire the destructive lock for a KB.

        While held, ``discard`` and ``get_entry``-driven reload are blocked;
        callers must finalize / rebuild the instance themselves. The lock is
        released even on exception.
        """
        lock = await self._get_destructive_lock(kb_id)
        async with lock:
            self._destructive_held.add(kb_id)
            try:
                yield
            finally:
                self._destructive_held.discard(kb_id)

    async def force_evict(self, kb_id: str) -> bool:
        """Evict an entry even if a destructive lock is held by the caller.

        Intended for the destructive worker itself — it must already hold
        the destructive lock via :meth:`destructive_lock` before calling.
        """
        if kb_id not in self._destructive_held:
            raise DestructiveLockBusyError(
                f"force_evict requires destructive lock on KB '{kb_id}'"
            )
        lock = await self._get_lock(kb_id)
        async with lock:
            entry = self._entries.pop(kb_id, None)
            if entry is None:
                return False
            await self._safe_finalize(entry.rag)
            return True

    async def shutdown(self) -> None:
        async with self._locks_guard:
            kb_ids = list(self._entries.keys())

        for kb_id in kb_ids:
            try:
                await self.discard(kb_id)
            except DestructiveLockBusyError:
                logger.warning(
                    "Skipping shutdown discard for KB '%s' — destructive lock held",
                    kb_id,
                )

    def is_loaded(self, kb_id: str) -> bool:
        return kb_id in self._entries

    def loaded_workspaces(self) -> dict[str, str]:
        return {kb_id: entry.workspace for kb_id, entry in self._entries.items()}

    def is_destructive_held(self, kb_id: str) -> bool:
        return kb_id in self._destructive_held

    def _is_expired(self, entry: RegistryEntry, now: float) -> bool:
        if self._idle_ttl_seconds is None:
            return False
        return (now - entry.last_used) >= self._idle_ttl_seconds

    async def _enforce_capacity(self, *, protect_kb_id: str | None = None) -> None:
        if self._max_entries is None:
            return
        while len(self._entries) > self._max_entries:
            evict_id: str | None = None
            for kb_id in self._entries:
                if kb_id == protect_kb_id:
                    continue
                if kb_id in self._destructive_held:
                    continue
                evict_id = kb_id
                break
            if evict_id is None:
                # Cache full of protected entries — give up to avoid deadlock.
                return
            entry = self._entries.pop(evict_id, None)
            if entry is not None:
                await self._safe_finalize(entry.rag)

    async def _get_lock(self, kb_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._locks.get(kb_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[kb_id] = lock
            return lock

    async def _get_destructive_lock(self, kb_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._destructive_locks.get(kb_id)
            if lock is None:
                lock = asyncio.Lock()
                self._destructive_locks[kb_id] = lock
            return lock

    async def _safe_finalize(self, rag: LightRAGLike) -> None:
        try:
            await self._finalizer(rag)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to finalize KB LightRAG instance: %s", exc)
            raise
