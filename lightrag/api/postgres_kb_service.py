from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from typing import Any

from lightrag.api.kb_service import (
    KB_STATUS_VALUES,
    VISIBILITY_VALUES,
    _UNSET,
    _assert_expected_generation,
    _assert_expected_status,
    _merge_kb_metadata,
    _new_kb_generation,
    _optional_string,
    _require_string,
    _validate_kb_metadata,
    KnowledgeBaseConflictError,
    KnowledgeBaseNotFoundError,
    KnowledgeBaseOrigin,
    KnowledgeBaseRecord,
    KnowledgeBaseService,
    KnowledgeBaseStatus,
    KnowledgeBaseVisibility,
    UpdateField,
    sanitize_workspace,
    utc_now_iso,
    validate_kb_id,
)


def _load_asyncpg() -> Any:
    try:
        import asyncpg
    except ImportError as exc:  # pragma: no cover - exercised only without optional deps
        raise RuntimeError(
            "PostgreSQL KB catalog backend requires asyncpg. "
            "Install LightRAG with the api/offline-storage extras or install asyncpg."
        ) from exc
    return asyncpg


def _loads_json_object(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return dict(value)
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise RuntimeError("Knowledge base JSON must be an object")
    return loaded


def _record_from_row(row: Any) -> KnowledgeBaseRecord:
    return KnowledgeBaseRecord.from_dict(_loads_json_object(row["data_json"]))


def _record_json(record: KnowledgeBaseRecord) -> str:
    return json.dumps(asdict(record), ensure_ascii=False, sort_keys=True)


class PostgresKnowledgeBaseService(KnowledgeBaseService):
    """PostgreSQL-backed catalog for knowledge base records."""

    def __init__(
        self,
        dsn: str | None = None,
        *,
        host: str | None = None,
        port: int | str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        min_size: int = 1,
        max_size: int = 5,
    ):
        self._dsn = dsn
        self._connect_kwargs = {
            "host": host,
            "port": int(port) if port is not None else None,
            "user": user,
            "password": password,
            "database": database,
        }
        self._connect_kwargs = {
            key: value for key, value in self._connect_kwargs.items() if value is not None
        }
        self._min_size = min_size
        self._max_size = max_size
        self._pool: Any | None = None
        self._lock = asyncio.Lock()
        self._initialized = False

    @classmethod
    def from_env(cls) -> "PostgresKnowledgeBaseService":
        dsn = os.getenv("LIGHTRAG_KB_POSTGRES_DSN") or os.getenv("POSTGRES_DSN")
        return cls(
            dsn=dsn,
            host=os.getenv("LIGHTRAG_KB_POSTGRES_HOST") or os.getenv("POSTGRES_HOST"),
            port=os.getenv("LIGHTRAG_KB_POSTGRES_PORT") or os.getenv("POSTGRES_PORT"),
            user=os.getenv("LIGHTRAG_KB_POSTGRES_USER") or os.getenv("POSTGRES_USER"),
            password=os.getenv("LIGHTRAG_KB_POSTGRES_PASSWORD")
            or os.getenv("POSTGRES_PASSWORD"),
            database=os.getenv("LIGHTRAG_KB_POSTGRES_DATABASE")
            or os.getenv("POSTGRES_DATABASE"),
            min_size=int(os.getenv("LIGHTRAG_KB_POSTGRES_POOL_MIN_SIZE", "1")),
            max_size=int(os.getenv("LIGHTRAG_KB_POSTGRES_POOL_MAX_SIZE", "10")),
        )

    async def initialize(self) -> None:
        async with self._lock:
            if self._pool is None:
                asyncpg = _load_asyncpg()
                if self._dsn:
                    self._pool = await asyncpg.create_pool(
                        dsn=self._dsn,
                        min_size=self._min_size,
                        max_size=self._max_size,
                    )
                else:
                    self._pool = await asyncpg.create_pool(
                        min_size=self._min_size,
                        max_size=self._max_size,
                        **self._connect_kwargs,
                    )
            async with self._pool_or_raise().acquire() as conn:
                await self._initialize_schema(conn)
            self._initialized = True

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._initialized = False

    async def create(
        self,
        *,
        name: str,
        kb_id: str | None = None,
        description: str | None = None,
        owner_id: str | None = None,
        tenant_id: str | None = None,
        visibility: KnowledgeBaseVisibility = "private",
        metadata: dict[str, Any] | None = None,
        origin: KnowledgeBaseOrigin = "platform",
        initial_status: KnowledgeBaseStatus = "active",
    ) -> KnowledgeBaseRecord:
        await self._ensure_initialized()
        import uuid

        normalized_id = validate_kb_id(kb_id) if kb_id is not None else f"kb_{uuid.uuid4().hex[:12]}"
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("Knowledge base name cannot be empty")
        if visibility not in VISIBILITY_VALUES:
            raise ValueError("Invalid knowledge base visibility")
        if origin not in {"tenant", "platform"}:
            raise ValueError("Invalid knowledge base origin")
        if initial_status not in KB_STATUS_VALUES or initial_status == "deleted":
            raise ValueError("Invalid initial knowledge base status")
        normalized_metadata = (
            _validate_kb_metadata(dict(metadata)) if metadata else {}
        )
        now = utc_now_iso()
        record = KnowledgeBaseRecord(
            id=normalized_id,
            name=normalized_name,
            description=description,
            workspace=sanitize_workspace(normalized_id),
            status=initial_status,
            active_config_version_id=None,
            owner_id=owner_id,
            tenant_id=tenant_id,
            visibility=visibility,
            created_at=now,
            updated_at=now,
            deleted_at=None,
            metadata=normalized_metadata,
            origin=origin,
            generation=_new_kb_generation(),
        )
        async with self._pool_or_raise().acquire() as conn:
            async with conn.transaction():
                existing = await conn.fetchrow(
                    "SELECT id FROM kb_catalog WHERE id = $1", normalized_id
                )
                if existing is not None:
                    raise KnowledgeBaseConflictError(
                        f"Knowledge base '{normalized_id}' already exists"
                    )
                await self._insert_record(conn, record)
        return record

    async def list(self, *, include_deleted: bool = False) -> list[KnowledgeBaseRecord]:
        await self._ensure_initialized()
        where = "" if include_deleted else "WHERE status <> 'deleted'"
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT data_json FROM kb_catalog
                {where}
                ORDER BY created_at ASC, id ASC
                """
            )
        return [_record_from_row(row) for row in rows]

    async def get(
        self, kb_id: str, *, include_deleted: bool = False
    ) -> KnowledgeBaseRecord:
        await self._ensure_initialized()
        normalized_id = validate_kb_id(kb_id)
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data_json FROM kb_catalog WHERE id = $1", normalized_id
            )
        if row is None:
            raise KnowledgeBaseNotFoundError(f"Knowledge base '{normalized_id}' not found")
        record = _record_from_row(row)
        if record.status == "deleted" and not include_deleted:
            raise KnowledgeBaseNotFoundError(f"Knowledge base '{normalized_id}' not found")
        return record

    async def update(
        self,
        kb_id: str,
        *,
        name: UpdateField = _UNSET,
        description: UpdateField = _UNSET,
        status: UpdateField = _UNSET,
        owner_id: UpdateField = _UNSET,
        tenant_id: UpdateField = _UNSET,
        visibility: UpdateField = _UNSET,
        active_config_version_id: UpdateField = _UNSET,
        metadata: Any = _UNSET,
        expected_generation: str | None = None,
    ) -> KnowledgeBaseRecord:
        await self._ensure_initialized()
        normalized_id = validate_kb_id(kb_id)
        async with self._pool_or_raise().acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT data_json FROM kb_catalog WHERE id = $1 FOR UPDATE",
                    normalized_id,
                )
                if row is None:
                    raise KnowledgeBaseNotFoundError(
                        f"Knowledge base '{normalized_id}' not found"
                    )
                record = _record_from_row(row)
                if record.status == "deleted":
                    raise KnowledgeBaseNotFoundError(
                        f"Knowledge base '{normalized_id}' not found"
                    )
                _assert_expected_generation(record, expected_generation)
                updated = record.to_dict()
                if name is not _UNSET:
                    normalized_name = _require_string(name, "Knowledge base name").strip()
                    if not normalized_name:
                        raise ValueError("Knowledge base name cannot be empty")
                    updated["name"] = normalized_name
                if description is not _UNSET:
                    updated["description"] = _optional_string(description, "Description")
                if status is not _UNSET:
                    status_value = _require_string(status, "Knowledge base status")
                    if status_value not in KB_STATUS_VALUES or status_value == "deleted":
                        raise ValueError("Invalid knowledge base status")
                    updated["status"] = status_value
                if owner_id is not _UNSET:
                    updated["owner_id"] = _optional_string(owner_id, "Owner id")
                if tenant_id is not _UNSET:
                    updated["tenant_id"] = _optional_string(tenant_id, "Tenant id")
                if visibility is not _UNSET:
                    visibility_value = _require_string(
                        visibility, "Knowledge base visibility"
                    )
                    if visibility_value not in VISIBILITY_VALUES:
                        raise ValueError("Invalid knowledge base visibility")
                    updated["visibility"] = visibility_value
                if active_config_version_id is not _UNSET:
                    updated["active_config_version_id"] = _optional_string(
                        active_config_version_id, "Active config version id"
                    )
                if metadata is not _UNSET:
                    updated["metadata"] = _merge_kb_metadata(record.metadata, metadata)
                updated["updated_at"] = utc_now_iso()
                next_record = KnowledgeBaseRecord.from_dict(updated)
                await self._save_record(conn, next_record)
                return next_record

    async def delete(
        self, kb_id: str, *, expected_generation: str | None = None
    ) -> KnowledgeBaseRecord:
        await self._ensure_initialized()
        normalized_id = validate_kb_id(kb_id)
        async with self._pool_or_raise().acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT data_json FROM kb_catalog WHERE id = $1 FOR UPDATE",
                    normalized_id,
                )
                if row is None:
                    raise KnowledgeBaseNotFoundError(
                        f"Knowledge base '{normalized_id}' not found"
                    )
                record = _record_from_row(row)
                if record.status == "deleted":
                    raise KnowledgeBaseNotFoundError(
                        f"Knowledge base '{normalized_id}' not found"
                    )
                _assert_expected_generation(record, expected_generation)
                now = utc_now_iso()
                updated = record.to_dict()
                updated["status"] = "deleted"
                updated["updated_at"] = now
                updated["deleted_at"] = now
                deleted_record = KnowledgeBaseRecord.from_dict(updated)
                await self._save_record(conn, deleted_record)
                return deleted_record

    async def restore(
        self, kb_id: str, *, expected_generation: str | None = None
    ) -> KnowledgeBaseRecord:
        """Restore a soft-deleted knowledge base back to ``active``.

        Raises ``KnowledgeBaseNotFoundError`` for an unknown id and
        ``ValueError`` when the record is not currently soft-deleted (the
        caller maps that to HTTP 409).
        """
        await self._ensure_initialized()
        normalized_id = validate_kb_id(kb_id)
        async with self._pool_or_raise().acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT data_json FROM kb_catalog WHERE id = $1 FOR UPDATE",
                    normalized_id,
                )
                if row is None:
                    raise KnowledgeBaseNotFoundError(
                        f"Knowledge base '{normalized_id}' not found"
                    )
                record = _record_from_row(row)
                _assert_expected_generation(record, expected_generation)
                if record.status != "deleted":
                    raise ValueError(
                        f"Knowledge base '{normalized_id}' is not deleted"
                    )
                updated = record.to_dict()
                updated["status"] = "active"
                updated["updated_at"] = utc_now_iso()
                updated["deleted_at"] = None
                restored_record = KnowledgeBaseRecord.from_dict(updated)
                await self._save_record(conn, restored_record)
                return restored_record

    async def purge(
        self,
        kb_id: str,
        *,
        expected_generation: str | None = None,
        expected_status: KnowledgeBaseStatus | None = "deleted",
    ) -> bool:
        """Hard-remove the kb_catalog row so the kb_id (and its workspace)
        become reusable. Idempotent: returns False when the row is absent.

        The default status CAS requires ``deleted`` in addition to the pinned
        generation, preventing a delayed cleanup from purging a restored row.

        Called from :meth:`KBDeletionService._execute_clear` at the end of a
        ``hard=true`` flow. The shared catalog row would otherwise keep the
        id and the UNIQUE workspace locked in ``status='deleted'`` forever,
        and the next ``POST /kbs`` with the same id would 409.
        """
        await self._ensure_initialized()
        normalized_id = validate_kb_id(kb_id)
        async with self._pool_or_raise().acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT data_json FROM kb_catalog WHERE id = $1 FOR UPDATE",
                    normalized_id,
                )
                if row is None:
                    return False
                record = _record_from_row(row)
                _assert_expected_generation(record, expected_generation)
                _assert_expected_status(record, expected_status)
                status = await conn.execute(
                    "DELETE FROM kb_catalog WHERE id = $1", normalized_id
                )
        try:
            return int(status.rsplit(" ", 1)[-1]) > 0
        except (ValueError, AttributeError):
            return False

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    def _pool_or_raise(self) -> Any:
        if self._pool is None:
            raise RuntimeError("PostgreSQL KB catalog service is not initialized")
        return self._pool

    async def _initialize_schema(self, conn: Any) -> None:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kb_catalog_schema (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS kb_catalog (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                workspace TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                owner_id TEXT,
                tenant_id TEXT,
                visibility TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted_at TEXT,
                data_json JSONB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_kb_catalog_status
                ON kb_catalog (status);
            CREATE INDEX IF NOT EXISTS idx_kb_catalog_tenant
                ON kb_catalog (tenant_id);
            """
        )
        await conn.execute(
            """
            INSERT INTO kb_catalog_schema(version, applied_at)
            VALUES (1, $1)
            ON CONFLICT (version) DO NOTHING
            """,
            utc_now_iso(),
        )

    async def _insert_record(self, conn: Any, record: KnowledgeBaseRecord) -> None:
        await conn.execute(
            """
            INSERT INTO kb_catalog (
                id, name, workspace, status, owner_id, tenant_id, visibility,
                created_at, updated_at, deleted_at, data_json
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb)
            """,
            record.id,
            record.name,
            record.workspace,
            record.status,
            record.owner_id,
            record.tenant_id,
            record.visibility,
            record.created_at,
            record.updated_at,
            record.deleted_at,
            _record_json(record),
        )

    async def _save_record(self, conn: Any, record: KnowledgeBaseRecord) -> None:
        await conn.execute(
            """
            UPDATE kb_catalog
            SET name = $1, status = $2, owner_id = $3, tenant_id = $4,
                visibility = $5, updated_at = $6, deleted_at = $7, data_json = $8::jsonb
            WHERE id = $9
            """,
            record.name,
            record.status,
            record.owner_id,
            record.tenant_id,
            record.visibility,
            record.updated_at,
            record.deleted_at,
            _record_json(record),
            record.id,
        )
