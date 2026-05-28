from __future__ import annotations

import asyncio
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


LightRAGBuilder = Callable[[KnowledgeBaseRecord], Awaitable[LightRAGLike]]
LightRAGFinalizer = Callable[[LightRAGLike], Awaitable[None]]


class LightRAGInstanceRegistry:
    def __init__(
        self,
        kb_service: KnowledgeBaseService,
        builder: LightRAGBuilder,
        finalizer: LightRAGFinalizer,
    ):
        self._kb_service = kb_service
        self._builder = builder
        self._finalizer = finalizer
        self._entries: dict[str, RegistryEntry] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    async def get(self, kb_id: str) -> LightRAGLike:
        entry = await self.get_entry(kb_id)
        return entry.rag

    async def get_entry(self, kb_id: str) -> RegistryEntry:
        record = await self._kb_service.get(kb_id)
        lock = await self._get_lock(record.id)
        async with lock:
            entry = self._entries.get(record.id)
            if (
                entry is not None
                and entry.workspace == record.workspace
                and entry.config_version_id == record.active_config_version_id
            ):
                return entry

            if entry is not None:
                await self._safe_finalize(entry.rag)
                self._entries.pop(record.id, None)

            rag = await self._builder(record)
            entry = RegistryEntry(
                kb_id=record.id,
                workspace=record.workspace,
                rag=rag,
                config_version_id=record.active_config_version_id,
            )
            self._entries[record.id] = entry
            return entry

    async def discard(self, kb_id: str) -> bool:
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
            await self.discard(kb_id)

    def is_loaded(self, kb_id: str) -> bool:
        return kb_id in self._entries

    def loaded_workspaces(self) -> dict[str, str]:
        return {kb_id: entry.workspace for kb_id, entry in self._entries.items()}

    async def _get_lock(self, kb_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._locks.get(kb_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[kb_id] = lock
            return lock

    async def _safe_finalize(self, rag: LightRAGLike) -> None:
        try:
            await self._finalizer(rag)
        except Exception as exc:
            logger.error("Failed to finalize KB LightRAG instance: %s", exc)
            raise
