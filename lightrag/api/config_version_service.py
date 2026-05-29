from __future__ import annotations

import hashlib
import json
from typing import Any

from lightrag.api.kb_service import KnowledgeBaseService, utc_now_iso
from lightrag.api.lightrag_registry import (
    DestructiveLockBusyError,
    LightRAGInstanceRegistry,
)
from lightrag.api.metadata_store import (
    ConfigVersionRecord,
    SQLiteMetadataStore,
)
from lightrag.utils import generate_track_id


def _section_hash(section_name: str, payload: dict[str, Any] | None) -> str:
    blob = {
        "schema": f"kb-{section_name}-hash-v1",
        "payload": payload or {},
    }
    encoded = json.dumps(blob, ensure_ascii=False, sort_keys=True, default=str).encode(
        "utf-8"
    )
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


class ConfigVersionService:
    """Persist and activate KB-scoped configuration versions.

    A config version is a frozen snapshot of parser / chunk / embedding /
    extraction / query / storage settings, plus three derived hashes
    (``parser_hash``, ``index_hash``, ``query_hash``). Activation simply
    points the KB record at the new ``active_config_version_id`` and
    discards the cached LightRAG instance so the next request rebuilds it.
    """

    def __init__(
        self,
        kb_service: KnowledgeBaseService,
        metadata_store: SQLiteMetadataStore,
        registry: LightRAGInstanceRegistry,
    ):
        self._kb_service = kb_service
        self._metadata_store = metadata_store
        self._registry = registry

    async def create(
        self,
        kb_id: str,
        *,
        config: dict[str, Any],
        created_by: str | None = None,
    ) -> ConfigVersionRecord:
        record = await self._kb_service.get(kb_id)
        derived = self._derive_hashes(config)
        version_record = ConfigVersionRecord(
            id=generate_track_id("cfg"),
            kb_id=record.id,
            workspace=record.workspace,
            version=0,
            config=config,
            parser_hash=derived["parser_hash"],
            index_hash=derived["index_hash"],
            query_hash=derived["query_hash"],
            created_at=utc_now_iso(),
            activated_at=None,
            created_by=created_by,
        )
        return await self._metadata_store.create_config_version(version_record)

    async def list(
        self, kb_id: str, *, limit: int = 50, offset: int = 0
    ) -> tuple[list[ConfigVersionRecord], int]:
        record = await self._kb_service.get(kb_id)
        return await self._metadata_store.list_config_versions(
            record.id, limit=limit, offset=offset
        )

    async def get(self, kb_id: str, version_id: str) -> ConfigVersionRecord:
        record = await self._kb_service.get(kb_id)
        return await self._metadata_store.get_config_version(record.id, version_id)

    async def activate(self, kb_id: str, version_id: str) -> ConfigVersionRecord:
        record = await self._kb_service.get(kb_id)
        version = await self._metadata_store.get_config_version(record.id, version_id)
        await self._kb_service.update(
            record.id, active_config_version_id=version.id
        )
        marked = await self._metadata_store.mark_config_version_activated(
            record.id, version.id
        )
        # Drop cached instance so the next request rebuilds with the new
        # config version. If a destructive job is in flight, leave it alone.
        try:
            await self._registry.discard(record.id)
        except DestructiveLockBusyError:
            pass
        return marked

    async def diff(
        self, kb_id: str, version_id: str
    ) -> dict[str, Any]:
        """Compare the target version against the currently active one.

        Returns rebuild requirements (``requires_reparse`` /
        ``requires_reindex`` / ``requires_vector_rebuild``) plus a list of
        changed config sections — enough for clients to preview the impact
        before activation.
        """
        record = await self._kb_service.get(kb_id)
        target = await self._metadata_store.get_config_version(record.id, version_id)
        if not record.active_config_version_id:
            return {
                "target_version_id": target.id,
                "active_version_id": None,
                "requires_reparse": True,
                "requires_reindex": True,
                "requires_vector_rebuild": True,
                "reasons": ["no_active_version"],
            }
        active = await self._metadata_store.get_config_version(
            record.id, record.active_config_version_id
        )
        reasons: list[str] = []
        requires_reparse = active.parser_hash != target.parser_hash
        requires_reindex = (
            requires_reparse or active.index_hash != target.index_hash
        )
        requires_vector_rebuild = self._embedding_changed(
            active.config, target.config
        )
        if requires_reparse:
            reasons.append("parser_hash_changed")
        if active.index_hash != target.index_hash:
            reasons.append("index_hash_changed")
        if requires_vector_rebuild:
            reasons.append("embedding_changed")
        if active.query_hash != target.query_hash:
            reasons.append("query_hash_changed")
        return {
            "target_version_id": target.id,
            "active_version_id": active.id,
            "requires_reparse": requires_reparse,
            "requires_reindex": requires_reindex,
            "requires_vector_rebuild": requires_vector_rebuild,
            "reasons": reasons,
        }

    @staticmethod
    def _derive_hashes(config: dict[str, Any]) -> dict[str, str]:
        return {
            "parser_hash": _section_hash("parser", config.get("parser_config")),
            "index_hash": _section_hash(
                "index",
                {
                    "chunk": config.get("chunk_config"),
                    "embedding": config.get("embedding_config"),
                    "extraction": config.get("llm_role_config"),
                },
            ),
            "query_hash": _section_hash("query", config.get("query_config")),
        }

    @staticmethod
    def _embedding_changed(
        active: dict[str, Any], target: dict[str, Any]
    ) -> bool:
        active_embed = active.get("embedding_config") or {}
        target_embed = target.get("embedding_config") or {}
        for key in ("model", "dim"):
            if active_embed.get(key) != target_embed.get(key):
                return True
        return False
