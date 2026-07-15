from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Sequence, TypeVar

from lightrag.api.kb_service import utc_now_iso
from lightrag.api.metadata_store import (
    _AGGREGATE_RESUMABLE_JOB_TYPES,
    _LEGACY_KB_TOMBSTONE_PREFIX,
    _ORPHANED_DOCUMENT_STATUS_TARGETS,
    _ORPHANED_JOB_STATUSES,
    _REPLACE_DERIVED_METADATA_KEYS,
    _allowed_next_job_statuses,
    _assert_enterprise_user_membership_precondition,
    _assert_enterprise_user_write_preconditions,
    _assert_tenant_user_kb_override_target_preconditions,
    _escape_like,
    _EXPECTATION_UNSET,
    _kb_lifecycle_conflict,
    _missing_kb_lifecycle_conflict,
    _metadata_source_key,
    _document_job_ids,
    _job_recovery_document_ids,
    _orphan_recovery_cutoff,
    _same_job_execution_identity,
    _should_requeue_orphaned_clear_job,
    _TENANT_MEMBERSHIP_ROLES,
    _validate_job_execution_id,
    _validate_delete_job_id,
    _validate_kb_lifecycle_identity,
    _validate_tenant_user_kb_override,
    _wait_for_kb_guard_borrowers,
    ActiveDocumentBuildJobError,
    ActiveDocumentDeleteJobError,
    ActiveDocumentParseJobError,
    ActiveDocumentReplaceJobError,
    ArtifactRecord,
    AuditEventRecord,
    ChatMemoryBacklogItem,
    ChatMemoryEpisodeRecord,
    ChatMessageRecord,
    ChatProjectRecord,
    ChatSessionRecord,
    ConfigVersionRecord,
    DocumentNotParsedError,
    DocumentRecord,
    DuplicateDocumentSourceKeyError,
    EnterpriseUserRecord,
    EnterpriseUserKBQuerySettingsRecord,
    EnterpriseAPIKeyRecord,
    EnterpriseInvitationRecord,
    IdempotencyKeyConflictError,
    InvalidJobTransitionError,
    JobRecord,
    KBACLRecord,
    KBLifecycleRecord,
    EnterpriseTenantKBACLRecord,
    EnterpriseTenantMembershipRecord,
    EnterpriseTenantRecord,
    EnterpriseTenantUserKBOverrideRecord,
    MetadataJobStatus,
    MetadataConflictError,
    MetadataStoreError,
    MetadataRecordNotFoundError,
    InvalidTenantUserKBOverrideError,
)

_T = TypeVar("_T")


@dataclass(slots=True)
class _OperationSessionState:
    store: Any
    owner_task: asyncio.Task[Any] | None
    connection: Any
    depth: int = 1
    kb_write_depths: dict[tuple[str, str | None], int] = field(default_factory=dict)
    kb_write_idle_events: dict[tuple[str, str | None], asyncio.Event] = field(
        default_factory=dict
    )


_OPERATION_SESSION_STATES: ContextVar[dict[int, _OperationSessionState] | None] = (
    ContextVar("postgres_metadata_operation_sessions", default=None)
)


def _load_asyncpg() -> Any:
    try:
        import asyncpg
    except ImportError as exc:  # pragma: no cover - exercised only without optional deps
        raise RuntimeError(
            "PostgreSQL KB metadata backend requires asyncpg. "
            "Install LightRAG with the api/offline-storage extras or install asyncpg."
        ) from exc
    return asyncpg


def _loads_json_object(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        loaded = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise MetadataStoreError("Metadata JSON must be an object") from exc
    if not isinstance(loaded, dict):
        raise MetadataStoreError("Metadata JSON must be an object")
    return loaded


def _dumps_json(value: dict[str, Any] | None) -> str:
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True)


def _document_from_row(row: Any) -> DocumentRecord:
    data = _loads_json_object(row["data_json"])
    return DocumentRecord(**data)


def _job_from_row(row: Any) -> JobRecord:
    data = _loads_json_object(row["data_json"])
    return JobRecord(**data)


def _artifact_from_row(row: Any) -> ArtifactRecord:
    data = _loads_json_object(row["data_json"])
    return ArtifactRecord(**data)


def _config_from_row(row: Any) -> ConfigVersionRecord:
    data = _loads_json_object(row["data_json"])
    return ConfigVersionRecord(**data)


def _kb_lifecycle_from_row(row: Any) -> KBLifecycleRecord:
    try:
        projection_keys = set(row.keys())
    except (AttributeError, TypeError):
        projection_keys = set(row) if isinstance(row, dict) else set()
    return KBLifecycleRecord(
        kb_id=str(row["kb_id"]),
        generation=str(row["generation"]),
        state=str(row["state"]),  # type: ignore[arg-type]
        activated_at=str(row["activated_at"]),
        deleted_at=row["deleted_at"],
        updated_at=str(row["updated_at"]),
        delete_job_id=(
            row["delete_job_id"] if "delete_job_id" in projection_keys else None
        ),
    )


def _enterprise_user_from_row(row: Any) -> EnterpriseUserRecord:
    data = _loads_json_object(row["data_json"])
    # Legacy JSONB rows predate can_delete_documents; default it so the
    # dataclass deserializes without raising on the missing key.
    data.setdefault("can_delete_documents", False)
    data.setdefault("can_use_agent_query", False)
    # PostgreSQL's historical records are JSONB-only. Missing means the user
    # predates download governance and therefore retains access.
    data.setdefault("can_download_files", True)
    try:
        projection_keys = set(row.keys())
    except (AttributeError, TypeError):
        projection_keys = set(row) if isinstance(row, dict) else set()
    for key in (
        "id",
        "username",
        "system_role",
        "status",
        "tenant_id",
        "token_version",
        "created_at",
        "updated_at",
    ):
        if key not in projection_keys:
            continue
        value: Any = row[key]
        if key == "tenant_id":
            data[key] = None if value is None else str(value)
        elif key == "token_version":
            data[key] = int(value)
        else:
            data[key] = str(value)
    return EnterpriseUserRecord(**data)


def _enterprise_user_kb_query_settings_from_row(
    row: Any,
) -> EnterpriseUserKBQuerySettingsRecord:
    data = _loads_json_object(row["data_json"])
    return EnterpriseUserKBQuerySettingsRecord(**data)


def _chat_project_from_row(row: Any) -> ChatProjectRecord:
    data = _loads_json_object(row["data_json"])
    return ChatProjectRecord(**data)


def _chat_session_from_row(row: Any) -> ChatSessionRecord:
    data = _loads_json_object(row["data_json"])
    # Legacy JSONB rows predate context_rounds; default it so the dataclass
    # deserializes without raising on the missing key.
    data.setdefault("context_rounds", 1)
    return ChatSessionRecord(**data)


def _chat_message_from_row(row: Any) -> ChatMessageRecord:
    data = _loads_json_object(row["data_json"])
    return ChatMessageRecord(**data)


def _chat_memory_episode_from_row(row: Any) -> ChatMemoryEpisodeRecord:
    return ChatMemoryEpisodeRecord(
        episode_uuid=str(row["episode_uuid"]),
        session_id=str(row["session_id"]),
        project_id=str(row["project_id"]),
        user_id=str(row["user_id"]),
        first_seq=int(row["first_seq"]),
        last_seq=int(row["last_seq"]),
        created_at=str(row["created_at"]),
    )


def _enterprise_api_key_from_row(row: Any) -> EnterpriseAPIKeyRecord:
    data = _loads_json_object(row["data_json"])
    return EnterpriseAPIKeyRecord(**data)


def _enterprise_invitation_from_row(row: Any) -> EnterpriseInvitationRecord:
    data = _loads_json_object(row["data_json"])
    return EnterpriseInvitationRecord(**data)


def _kb_acl_from_row(row: Any) -> KBACLRecord:
    data = _loads_json_object(row["data_json"])
    return KBACLRecord(**data)


def _tenant_membership_from_row(row: Any) -> EnterpriseTenantMembershipRecord:
    data = _loads_json_object(row["data_json"])
    try:
        projection = {
            "tenant_id": str(row["tenant_id"]),
            "user_id": str(row["user_id"]),
            "role": str(row["role"]),
            "granted_by": row["granted_by"],
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }
    except (KeyError, IndexError):
        # Compatibility for old unit-level callers that only supplied JSONB.
        return EnterpriseTenantMembershipRecord(**data)

    # Projection columns are canonical. In particular, never let stale JSONB
    # substitute a foreign tenant or a more privileged role on principal reads.
    return EnterpriseTenantMembershipRecord(**projection)


def _enterprise_tenant_from_row(row: Any) -> EnterpriseTenantRecord:
    return EnterpriseTenantRecord(**_loads_json_object(row["data_json"]))


def _tenant_kb_acl_from_row(row: Any) -> EnterpriseTenantKBACLRecord:
    data = _loads_json_object(row["data_json"])
    return EnterpriseTenantKBACLRecord(**data)


def _tenant_user_kb_override_from_row(
    row: Any,
) -> EnterpriseTenantUserKBOverrideRecord:
    data = _loads_json_object(row["data_json"])
    return EnterpriseTenantUserKBOverrideRecord(**data)


def _audit_event_from_row(row: Any) -> AuditEventRecord:
    data = _loads_json_object(row["data_json"])
    data.setdefault("actor_tenant_id", None)
    return AuditEventRecord(**data)


def _record_json(record: Any) -> str:
    return json.dumps(asdict(record), ensure_ascii=False, sort_keys=True)


def _active_parse_job_id(document: DocumentRecord) -> str:
    if document.status == "parse_queued":
        job_id = document.metadata.get("pending_parse_job_id")
        return str(job_id) if job_id else "unknown"
    if document.status == "parsing":
        job_id = document.metadata.get("current_parse_job_id")
        return str(job_id) if job_id else "unknown"
    return "unknown"


def _active_build_job_id(document: DocumentRecord) -> str:
    if document.status == "build_queued":
        job_id = document.metadata.get("pending_build_job_id")
        return str(job_id) if job_id else "unknown"
    if document.status == "building":
        job_id = document.metadata.get("current_build_job_id")
        return str(job_id) if job_id else "unknown"
    return "unknown"


def _active_delete_job_id(document: DocumentRecord) -> str:
    job_id = document.metadata.get("pending_delete_job_id") or document.metadata.get(
        "current_delete_job_id"
    )
    return str(job_id) if job_id else "unknown"


def _active_replace_job_id(document: DocumentRecord) -> str:
    job_id = document.metadata.get("pending_replace_job_id") or document.metadata.get(
        "current_replace_job_id"
    )
    return str(job_id) if job_id else "unknown"


class PostgresMetadataStore:
    """PostgreSQL-backed KB control-plane metadata store.

    The public methods intentionally mirror :class:`SQLiteMetadataStore` so the
    API services can switch backends without changing route logic. Records are
    persisted as JSONB plus indexed projection columns that are needed for list,
    idempotency, source-key, and worker-claim queries.
    """

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
        max_size: int = 10,
        operation_lock_pool_max_size: int = 10,
    ):
        if operation_lock_pool_max_size < 1:
            raise ValueError("operation_lock_pool_max_size must be at least 1")
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
        self._operation_lock_pool_max_size = operation_lock_pool_max_size
        self._pool: Any | None = None
        self._operation_lock_pool: Any | None = None
        self._lock = asyncio.Lock()
        self._operation_lock_pool_init_lock = asyncio.Lock()
        self._initialized = False

    @classmethod
    def from_env(cls) -> "PostgresMetadataStore":
        import os

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
            operation_lock_pool_max_size=int(
                os.getenv(
                    "LIGHTRAG_KB_POSTGRES_OPERATION_LOCK_POOL_MAX_SIZE",
                    "10",
                )
            ),
        )

    async def initialize(self) -> None:
        async with self._lock:
            if self._pool is None:
                self._pool = await self._create_pool(
                    min_size=self._min_size,
                    max_size=self._max_size,
                )
            async with self._pool_or_raise().acquire() as conn:
                # Schema repair + the single-membership unique index must be
                # serialized across server workers and committed atomically.
                async with conn.transaction():
                    await conn.execute(
                        "SELECT pg_advisory_xact_lock($1)", 4767034634417628498
                    )
                    await self._initialize_schema(conn)
            await self._ensure_operation_lock_pool()
            self._initialized = True

    async def close(self) -> None:
        operation_lock_pool = self._operation_lock_pool
        pool = self._pool
        self._operation_lock_pool = None
        self._pool = None
        self._initialized = False
        try:
            if operation_lock_pool is not None:
                await operation_lock_pool.close()
        finally:
            if pool is not None and pool is not operation_lock_pool:
                await pool.close()

    async def create_documents_and_job(
        self, documents: Sequence[DocumentRecord], job: JobRecord
    ) -> tuple[list[DocumentRecord], JobRecord, bool]:
        await self._ensure_initialized()

        async def write(conn: Any) -> tuple[list[DocumentRecord], JobRecord, bool]:
            existing = await self._get_job_by_idempotency_key(
                conn, job.kb_id, job.idempotency_key, job_type=job.job_type
            )
            if existing is not None:
                self._validate_idempotent_job(existing, job)
                return await self._documents_for_job(conn, existing), existing, False
            for document in documents:
                await self._insert_document(conn, document)
            await self._insert_job(conn, job)
            return list(documents), job, True

        return await self._write(write)

    async def list_documents(
        self,
        kb_id: str,
        *,
        status: str | None = None,
        source_name: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[DocumentRecord], int]:
        await self._ensure_initialized()
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        clauses = ["kb_id = $1", "deleted_at IS NULL"]
        params: list[Any] = [kb_id]
        if status is not None:
            params.append(status)
            clauses.append(f"status = ${len(params)}")
        if source_name is not None:
            params.append(f"%{_escape_like(source_name)}%")
            # ESCAPE must be a single character. ``_escape_like`` escapes with a
            # single backslash, so the SQL escape string is one backslash too:
            # the Python literal ``"\\"`` is exactly one backslash at runtime.
            # (Using ``"\\\\"`` here sends a two-char string to Postgres, which
            # rejects it with InvalidEscapeSequenceError.)
            clauses.append(f"source_name ILIKE ${len(params)} ESCAPE '\\'")
        where = " AND ".join(clauses)
        async with self._pool_or_raise().acquire() as conn:
            total = await conn.fetchval(f"SELECT COUNT(*) FROM kb_documents WHERE {where}", *params)
            rows = await conn.fetch(
                f"""
                SELECT data_json FROM kb_documents
                WHERE {where}
                ORDER BY created_at DESC, id DESC
                LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
                """,
                *params,
                limit,
                offset,
            )
        return [_document_from_row(row) for row in rows], int(total or 0)

    async def get_document(self, kb_id: str, document_id: str) -> DocumentRecord:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            document = await self._get_document(conn, kb_id, document_id)
        return document

    async def get_documents_by_ids(
        self, kb_id: str, document_ids: Sequence[str]
    ) -> list[DocumentRecord]:
        await self._ensure_initialized()
        ordered_ids = list(dict.fromkeys(document_ids))
        if not ordered_ids:
            return []
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM kb_documents
                WHERE kb_id = $1 AND id = ANY($2::text[]) AND deleted_at IS NULL
                """,
                kb_id,
                ordered_ids,
            )
        by_id = {row["data_json"]["id"] if isinstance(row["data_json"], dict) else _loads_json_object(row["data_json"])["id"]: _document_from_row(row) for row in rows}
        return [by_id[document_id] for document_id in document_ids if document_id in by_id]

    async def get_documents_by_source_keys(
        self, kb_id: str, source_keys: Sequence[str]
    ) -> dict[str, DocumentRecord]:
        await self._ensure_initialized()
        ordered_keys = list(dict.fromkeys(source_keys))
        if not ordered_keys:
            return {}
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT source_key, data_json FROM kb_documents
                WHERE kb_id = $1 AND source_key = ANY($2::text[]) AND deleted_at IS NULL
                ORDER BY updated_at DESC, created_at DESC, id DESC
                """,
                kb_id,
                ordered_keys,
            )
        documents: dict[str, DocumentRecord] = {}
        wanted = set(ordered_keys)
        for row in rows:
            source_key = str(row["source_key"])
            if source_key in wanted and source_key not in documents:
                documents[source_key] = _document_from_row(row)
        return documents

    async def list_documents_by_batch_id(
        self, kb_id: str, batch_id: str
    ) -> list[DocumentRecord]:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM kb_documents
                WHERE kb_id = $1 AND batch_id = $2 AND deleted_at IS NULL
                ORDER BY created_at ASC, id ASC
                """,
                kb_id,
                batch_id,
            )
        return [_document_from_row(row) for row in rows]

    async def update_document(
        self,
        kb_id: str,
        document_id: str,
        *,
        metadata_patch: dict[str, Any] | None = None,
        enabled: bool | None = None,
        archived: bool | None = None,
    ) -> DocumentRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> DocumentRecord:
            document = await self._get_document(conn, kb_id, document_id, for_update=True)
            metadata = dict(document.metadata)
            if metadata_patch:
                metadata.update(metadata_patch)
            document.metadata = metadata
            if enabled is not None:
                document.enabled = enabled
            if archived is not None:
                document.archived = archived
            document.updated_at = utc_now_iso()
            await self._save_document(conn, document)
            return document

        return await self._write(write)

    async def mark_document_parse_queued(
        self, kb_id: str, document_id: str, *, metadata_patch: dict[str, Any]
    ) -> DocumentRecord:
        await self._ensure_initialized()
        return await self._write(
            lambda conn: self._claim_document_parse_queued(
                conn,
                kb_id,
                document_id,
                metadata_patch=metadata_patch,
                raise_on_active=True,
            )
        )

    async def claim_documents_parse_queued(
        self, kb_id: str, claims: Sequence[tuple[str, dict[str, Any]]]
    ) -> tuple[list[DocumentRecord], list[dict[str, Any]]]:
        await self._ensure_initialized()

        async def write(conn: Any) -> tuple[list[DocumentRecord], list[dict[str, Any]]]:
            documents: list[DocumentRecord] = []
            failures: list[dict[str, Any]] = []
            for document_id, metadata_patch in claims:
                try:
                    documents.append(
                        await self._claim_document_parse_queued(
                            conn,
                            kb_id,
                            document_id,
                            metadata_patch=metadata_patch,
                            raise_on_active=True,
                        )
                    )
                except ActiveDocumentParseJobError as exc:
                    failures.append(_active_failure(document_id, "parse_job_active", exc))
                except ActiveDocumentBuildJobError as exc:
                    failures.append(_active_failure(document_id, "build_job_active", exc))
                except ActiveDocumentDeleteJobError as exc:
                    failures.append(_active_failure(document_id, "delete_job_active", exc))
                except ActiveDocumentReplaceJobError as exc:
                    failures.append(_active_failure(document_id, "replace_job_active", exc))
                except MetadataRecordNotFoundError as exc:
                    failures.append(
                        {
                            "document_id": document_id,
                            "status": "failed",
                            "error_code": "document_not_found",
                            "error_message": str(exc),
                        }
                    )
            return documents, failures

        return await self._write(write)

    async def mark_document_parsing(
        self, kb_id: str, document_id: str, *, metadata_patch: dict[str, Any]
    ) -> DocumentRecord:
        await self._ensure_initialized()
        return await self._write(
            lambda conn: self._update_document_parse_state(
                conn,
                kb_id,
                document_id,
                status="parsing",
                metadata_patch=metadata_patch,
                clear_error=True,
            )
        )

    async def complete_document_parse(
        self,
        kb_id: str,
        document_id: str,
        *,
        parser_hash: str,
        lightrag_doc_id: str,
        metadata_patch: dict[str, Any],
        artifacts: Sequence[ArtifactRecord],
    ) -> tuple[DocumentRecord, list[ArtifactRecord]]:
        await self._ensure_initialized()

        async def write(conn: Any) -> tuple[DocumentRecord, list[ArtifactRecord]]:
            document = await self._update_document_parse_state(
                conn,
                kb_id,
                document_id,
                status="parsed",
                metadata_patch=metadata_patch,
                parser_hash=parser_hash,
                lightrag_doc_id=lightrag_doc_id,
                clear_error=True,
            )
            await conn.execute(
                "DELETE FROM kb_document_artifacts WHERE kb_id = $1 AND document_id = $2",
                kb_id,
                document_id,
            )
            for artifact in artifacts:
                await self._insert_artifact(conn, artifact)
            return document, list(artifacts)

        return await self._write(write)

    async def fail_document_parse(
        self,
        kb_id: str,
        document_id: str,
        *,
        error_code: str,
        error_message: str,
        metadata_patch: dict[str, Any],
    ) -> DocumentRecord:
        await self._ensure_initialized()
        return await self._write(
            lambda conn: self._update_document_parse_state(
                conn,
                kb_id,
                document_id,
                status="parse_failed",
                metadata_patch=metadata_patch,
                error_code=error_code,
                error_message=error_message,
            )
        )

    async def claim_document_build_queued(
        self,
        kb_id: str,
        document_id: str,
        *,
        metadata_patch: dict[str, Any],
        require_parsed: bool = True,
    ) -> DocumentRecord:
        await self._ensure_initialized()
        return await self._write(
            lambda conn: self._claim_document_build_queued(
                conn,
                kb_id,
                document_id,
                metadata_patch=metadata_patch,
                require_parsed=require_parsed,
            )
        )

    async def claim_documents_build_queued(
        self,
        kb_id: str,
        claims: Sequence[tuple[str, dict[str, Any]]],
        *,
        require_parsed: bool = True,
    ) -> tuple[list[DocumentRecord], list[dict[str, Any]]]:
        await self._ensure_initialized()

        async def write(conn: Any) -> tuple[list[DocumentRecord], list[dict[str, Any]]]:
            documents: list[DocumentRecord] = []
            failures: list[dict[str, Any]] = []
            for document_id, metadata_patch in claims:
                try:
                    documents.append(
                        await self._claim_document_build_queued(
                            conn,
                            kb_id,
                            document_id,
                            metadata_patch=metadata_patch,
                            require_parsed=require_parsed,
                        )
                    )
                except ActiveDocumentBuildJobError as exc:
                    failures.append(_active_failure(document_id, "build_job_active", exc))
                except ActiveDocumentDeleteJobError as exc:
                    failures.append(_active_failure(document_id, "delete_job_active", exc))
                except ActiveDocumentReplaceJobError as exc:
                    failures.append(_active_failure(document_id, "replace_job_active", exc))
                except DocumentNotParsedError as exc:
                    failures.append(
                        {
                            "document_id": document_id,
                            "status": "failed",
                            "error_code": "document_not_parsed",
                            "error_message": str(exc),
                            "current_status": exc.current_status,
                        }
                    )
                except MetadataRecordNotFoundError as exc:
                    failures.append(
                        {
                            "document_id": document_id,
                            "status": "failed",
                            "error_code": "document_not_found",
                            "error_message": str(exc),
                        }
                    )
            return documents, failures

        return await self._write(write)

    async def mark_document_building(
        self, kb_id: str, document_id: str, *, metadata_patch: dict[str, Any]
    ) -> DocumentRecord:
        await self._ensure_initialized()
        return await self._write(
            lambda conn: self._update_document_parse_state(
                conn,
                kb_id,
                document_id,
                status="building",
                metadata_patch=metadata_patch,
                clear_error=True,
            )
        )

    async def complete_document_build(
        self,
        kb_id: str,
        document_id: str,
        *,
        index_hash: str,
        chunks_count: int | None = None,
        entity_count: int | None = None,
        relation_count: int | None = None,
        metadata_patch: dict[str, Any],
    ) -> DocumentRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> DocumentRecord:
            document = await self._get_document(conn, kb_id, document_id, for_update=True)
            document.metadata.update(metadata_patch)
            document.status = "ready"
            document.index_hash = index_hash
            if chunks_count is not None:
                document.chunks_count = chunks_count
            if entity_count is not None:
                document.entity_count = entity_count
            if relation_count is not None:
                document.relation_count = relation_count
            document.error_code = None
            document.error_message = None
            document.updated_at = utc_now_iso()
            await self._save_document(conn, document)
            return document

        return await self._write(write)

    async def fail_document_build(
        self,
        kb_id: str,
        document_id: str,
        *,
        error_code: str,
        error_message: str,
        metadata_patch: dict[str, Any],
    ) -> DocumentRecord:
        await self._ensure_initialized()
        return await self._write(
            lambda conn: self._update_document_parse_state(
                conn,
                kb_id,
                document_id,
                status="build_failed",
                metadata_patch=metadata_patch,
                error_code=error_code,
                error_message=error_message,
            )
        )

    async def claim_document_deleting(
        self, kb_id: str, document_id: str, *, metadata_patch: dict[str, Any]
    ) -> DocumentRecord:
        await self._ensure_initialized()
        return await self._write(
            lambda conn: self._claim_document_deleting(
                conn, kb_id, document_id, metadata_patch=metadata_patch
            )
        )

    async def claim_documents_deleting(
        self, kb_id: str, claims: Sequence[tuple[str, dict[str, Any]]]
    ) -> tuple[list[DocumentRecord], list[dict[str, Any]]]:
        await self._ensure_initialized()

        async def write(conn: Any) -> tuple[list[DocumentRecord], list[dict[str, Any]]]:
            documents: list[DocumentRecord] = []
            failures: list[dict[str, Any]] = []
            for document_id, metadata_patch in claims:
                try:
                    documents.append(
                        await self._claim_document_deleting(
                            conn, kb_id, document_id, metadata_patch=metadata_patch
                        )
                    )
                except ActiveDocumentParseJobError as exc:
                    failures.append(_active_failure(document_id, "parse_job_active", exc))
                except ActiveDocumentBuildJobError as exc:
                    failures.append(_active_failure(document_id, "build_job_active", exc))
                except ActiveDocumentDeleteJobError as exc:
                    failures.append(_active_failure(document_id, "delete_job_active", exc))
                except ActiveDocumentReplaceJobError as exc:
                    failures.append(_active_failure(document_id, "replace_job_active", exc))
                except MetadataRecordNotFoundError as exc:
                    failures.append(
                        {
                            "document_id": document_id,
                            "status": "failed",
                            "error_code": "document_not_found",
                            "error_message": str(exc),
                        }
                    )
            return documents, failures

        return await self._write(write)

    async def complete_document_delete(
        self, kb_id: str, document_id: str, *, metadata_patch: dict[str, Any]
    ) -> DocumentRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> DocumentRecord:
            document = await self._get_document(conn, kb_id, document_id, for_update=True)
            document.metadata.update(metadata_patch)
            now = utc_now_iso()
            await conn.execute(
                "DELETE FROM kb_document_artifacts WHERE kb_id = $1 AND document_id = $2",
                kb_id,
                document_id,
            )
            document.status = "deleted"
            document.enabled = False
            document.archived = True
            document.error_code = None
            document.error_message = None
            document.updated_at = now
            document.deleted_at = now
            await self._save_document(conn, document)
            return document

        return await self._write(write)

    async def fail_document_delete(
        self,
        kb_id: str,
        document_id: str,
        *,
        error_code: str,
        error_message: str,
        metadata_patch: dict[str, Any],
    ) -> DocumentRecord:
        await self._ensure_initialized()
        return await self._write(
            lambda conn: self._update_document_parse_state(
                conn,
                kb_id,
                document_id,
                status="delete_failed",
                metadata_patch=metadata_patch,
                error_code=error_code,
                error_message=error_message,
            )
        )

    async def claim_document_replacing(
        self, kb_id: str, document_id: str, *, metadata_patch: dict[str, Any]
    ) -> DocumentRecord:
        await self._ensure_initialized()
        return await self._write(
            lambda conn: self._claim_document_replacing(
                conn, kb_id, document_id, metadata_patch=metadata_patch
            )
        )

    async def complete_document_replace(
        self,
        kb_id: str,
        document_id: str,
        *,
        source_name: str,
        source_uri: str,
        source_type: str,
        source_hash: str,
        content_type: str | None,
        size_bytes: int,
        metadata_patch: dict[str, Any],
    ) -> DocumentRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> DocumentRecord:
            document = await self._get_document(conn, kb_id, document_id, for_update=True)
            metadata = dict(document.metadata)
            for key in _REPLACE_DERIVED_METADATA_KEYS:
                metadata.pop(key, None)
            metadata.update(metadata_patch)
            document.source_type = source_type
            document.source_name = source_name
            document.source_uri = source_uri
            document.source_hash = source_hash
            document.content_type = content_type
            document.size_bytes = size_bytes
            document.lightrag_doc_id = None
            document.parser_hash = None
            document.index_hash = None
            document.status = "uploaded"
            document.chunks_count = None
            document.entity_count = None
            document.relation_count = None
            document.error_code = None
            document.error_message = None
            document.metadata = metadata
            document.updated_at = utc_now_iso()
            await conn.execute(
                "DELETE FROM kb_document_artifacts WHERE kb_id = $1 AND document_id = $2",
                kb_id,
                document_id,
            )
            await self._save_document(conn, document)
            return document

        return await self._write(write)

    async def fail_document_replace(
        self,
        kb_id: str,
        document_id: str,
        *,
        error_code: str,
        error_message: str,
        metadata_patch: dict[str, Any],
        clear_index_metadata: bool = False,
    ) -> DocumentRecord:
        await self._ensure_initialized()
        return await self._write(
            lambda conn: self._update_document_parse_state(
                conn,
                kb_id,
                document_id,
                status="replace_failed",
                metadata_patch=metadata_patch,
                error_code=error_code,
                error_message=error_message,
                clear_lightrag_doc_id=clear_index_metadata,
                clear_index_state=clear_index_metadata,
            )
        )

    async def list_document_artifacts(
        self,
        kb_id: str,
        document_id: str,
        *,
        artifact_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[ArtifactRecord], int]:
        await self._ensure_initialized()
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        params: list[Any] = [kb_id, document_id]
        clauses = ["kb_id = $1", "document_id = $2"]
        if artifact_type is not None:
            params.append(artifact_type)
            clauses.append(f"artifact_type = ${len(params)}")
        where = " AND ".join(clauses)
        async with self._pool_or_raise().acquire() as conn:
            total = await conn.fetchval(
                f"SELECT COUNT(*) FROM kb_document_artifacts WHERE {where}", *params
            )
            rows = await conn.fetch(
                f"""
                SELECT data_json FROM kb_document_artifacts
                WHERE {where}
                ORDER BY created_at DESC, id DESC
                LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
                """,
                *params,
                limit,
                offset,
            )
        return [_artifact_from_row(row) for row in rows], int(total or 0)

    async def get_document_artifact(
        self, kb_id: str, document_id: str, artifact_id: str
    ) -> ArtifactRecord:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT data_json FROM kb_document_artifacts
                WHERE kb_id = $1 AND document_id = $2 AND id = $3
                """,
                kb_id,
                document_id,
                artifact_id,
            )
        if row is None:
            raise MetadataRecordNotFoundError(f"Artifact '{artifact_id}' not found")
        return _artifact_from_row(row)

    async def aggregate_control_plane_stats(
        self, kb_id: str | None = None
    ) -> dict[str, Any]:
        """Control-plane aggregates for the stats endpoints.

        Mirrors ``SQLiteMetadataStore.aggregate_control_plane_stats``; document
        counters are not projected columns in PostgreSQL so they are summed
        from ``data_json``.
        """
        await self._ensure_initialized()
        where = "" if kb_id is None else " WHERE kb_id = $1"
        params: list[Any] = [] if kb_id is None else [kb_id]
        async with self._pool_or_raise().acquire() as conn:
            documents_by_status = {
                str(row["status"]): int(row["count"])
                for row in await conn.fetch(
                    "SELECT status, COUNT(*) AS count FROM kb_documents"
                    f"{where} GROUP BY status",
                    *params,
                )
            }
            counter_row = await conn.fetchrow(
                "SELECT "
                "COALESCE(SUM((data_json->>'chunks_count')::bigint), 0) AS chunks, "
                "COALESCE(SUM((data_json->>'entity_count')::bigint), 0) AS entities, "
                "COALESCE(SUM((data_json->>'relation_count')::bigint), 0) AS relations "
                f"FROM kb_documents{where}",
                *params,
            )
            jobs_by_status = {
                str(row["status"]): int(row["count"])
                for row in await conn.fetch(
                    f"SELECT status, COUNT(*) AS count FROM kb_jobs{where} GROUP BY status",
                    *params,
                )
            }
            dead_letter_where = (
                " WHERE kb_id = $1 AND " if kb_id is not None else " WHERE "
            )
            dead_letter = await conn.fetchval(
                f"SELECT COUNT(*) FROM kb_jobs{dead_letter_where}"
                "status = 'failed' AND retry_count >= max_retries",
                *params,
            )
            artifacts = await conn.fetchval(
                f"SELECT COUNT(*) FROM kb_document_artifacts{where}", *params
            )
        return {
            "documents_by_status": documents_by_status,
            "document_counters": {
                "chunks": int(counter_row["chunks"]),
                "entities": int(counter_row["entities"]),
                "relations": int(counter_row["relations"]),
            },
            "jobs_by_status": jobs_by_status,
            "dead_letter_jobs": int(dead_letter or 0),
            "artifacts": int(artifacts or 0),
        }

    async def aggregate_enterprise_stats(self) -> dict[str, Any]:
        """Platform-wide enterprise aggregates for ``GET /admin/overview``."""
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            users_by_status = {
                str(row["status"]): int(row["count"])
                for row in await conn.fetch(
                    "SELECT status, COUNT(*) AS count FROM enterprise_users GROUP BY status"
                )
            }
            tenants = await conn.fetchval("SELECT COUNT(*) FROM enterprise_tenants")
            api_keys_by_status = {
                str(row["status"]): int(row["count"])
                for row in await conn.fetch(
                    "SELECT status, COUNT(*) AS count FROM enterprise_api_keys GROUP BY status"
                )
            }
            audit_events = await conn.fetchval(
                "SELECT COUNT(*) FROM enterprise_audit_events"
            )
        return {
            "users_by_status": users_by_status,
            "tenants": int(tenants or 0),
            "api_keys_by_status": api_keys_by_status,
            "audit_events": int(audit_events or 0),
        }

    async def count_active_jobs_for_principal(self, subject_id: str) -> int:
        """Count in-flight jobs (queued/running/retrying/cancelling) across all
        KBs attributed to a principal via ``payload._principal.subject_id``."""
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            value = await conn.fetchval(
                """
                SELECT COUNT(*) FROM kb_jobs
                WHERE status = ANY($1::text[])
                  AND data_json->'payload'->'_principal'->>'subject_id' = $2
                """,
                ["queued", "running", "retrying", "cancelling"],
                subject_id,
            )
        return int(value or 0)

    async def count_active_jobs_for_tenant(self, tenant_id: str) -> int:
        """Count in-flight jobs across all KBs attributed to a tenant via
        ``payload._principal.tenant_id``."""
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            value = await conn.fetchval(
                """
                SELECT COUNT(*) FROM kb_jobs
                WHERE status = ANY($1::text[])
                  AND data_json->'payload'->'_principal'->>'tenant_id' = $2
                """,
                ["queued", "running", "retrying", "cancelling"],
                tenant_id,
            )
        return int(value or 0)

    async def create_job(self, job: JobRecord) -> JobRecord:
        created_job, _created = await self.create_job_once(job)
        return created_job

    async def create_job_once(self, job: JobRecord) -> tuple[JobRecord, bool]:
        await self._ensure_initialized()

        async def write(conn: Any) -> tuple[JobRecord, bool]:
            existing = await self._get_job_by_idempotency_key(
                conn, job.kb_id, job.idempotency_key, job_type=job.job_type
            )
            if existing is not None:
                self._validate_idempotent_job(existing, job)
                return existing, False
            await self._insert_job(conn, job)
            return job, True

        return await self._write(write)

    async def get_job_by_idempotency_key(
        self, kb_id: str, idempotency_key: str, *, job_type: str | None = None
    ) -> JobRecord | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            return await self._get_job_by_idempotency_key(
                conn, kb_id, idempotency_key, job_type=job_type
            )

    async def list_jobs(
        self,
        kb_id: str,
        *,
        statuses: Sequence[str] | None = None,
        document_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[JobRecord], int]:
        await self._ensure_initialized()
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        params: list[Any] = [kb_id]
        clauses = ["kb_id = $1"]
        if document_id is not None:
            params.append(document_id)
            clauses.append(f"document_id = ${len(params)}")
        if statuses:
            params.append(list(statuses))
            clauses.append(f"status = ANY(${len(params)}::text[])")
        where = " AND ".join(clauses)
        async with self._pool_or_raise().acquire() as conn:
            total = await conn.fetchval(f"SELECT COUNT(*) FROM kb_jobs WHERE {where}", *params)
            rows = await conn.fetch(
                f"""
                SELECT data_json FROM kb_jobs
                WHERE {where}
                ORDER BY created_at DESC, id DESC
                LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
                """,
                *params,
                limit,
                offset,
            )
        return [_job_from_row(row) for row in rows], int(total or 0)

    async def list_dead_letter_jobs(
        self, kb_id: str, *, limit: int = 50, offset: int = 0
    ) -> tuple[list[JobRecord], int]:
        await self._ensure_initialized()
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        async with self._pool_or_raise().acquire() as conn:
            total = await conn.fetchval(
                """
                SELECT COUNT(*) FROM kb_jobs
                WHERE kb_id = $1 AND status = 'failed' AND retry_count >= max_retries
                """,
                kb_id,
            )
            rows = await conn.fetch(
                """
                SELECT data_json FROM kb_jobs
                WHERE kb_id = $1 AND status = 'failed' AND retry_count >= max_retries
                ORDER BY updated_at DESC, id DESC
                LIMIT $2 OFFSET $3
                """,
                kb_id,
                limit,
                offset,
            )
        return [_job_from_row(row) for row in rows], int(total or 0)

    async def get_job(self, kb_id: str, job_id: str) -> JobRecord:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            job = await self._get_job(conn, kb_id, job_id)
        return job

    async def transition_job(
        self,
        kb_id: str,
        job_id: str,
        *,
        status: MetadataJobStatus,
        stage: str | None = None,
        progress: float | None = None,
        completed_items: int | None = None,
        failed_items: int | None = None,
        result: dict[str, Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> JobRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> JobRecord:
            current = await self._get_job(conn, kb_id, job_id, for_update=True)
            if status not in _allowed_next_job_statuses(current.status):
                raise InvalidJobTransitionError(
                    f"Cannot transition job '{job_id}' from {current.status} to {status}"
                )
            now = utc_now_iso()
            current.status = status
            if stage is not None:
                current.stage = stage
            if progress is not None:
                current.progress = progress
            if completed_items is not None:
                current.completed_items = completed_items
            if failed_items is not None:
                current.failed_items = failed_items
            if result is not None:
                current.result = result
            current.error_code = error_code
            current.error_message = error_message
            current.updated_at = now
            if status == "running" and current.started_at is None:
                current.started_at = now
            if status in {"succeeded", "failed"} and current.finished_at is None:
                current.finished_at = now
            if status == "cancelled" and current.cancelled_at is None:
                current.cancelled_at = now
            await self._save_job(conn, current)
            return current

        return await self._write(write)

    async def update_job_payload_patch(
        self,
        kb_id: str,
        job_id: str,
        *,
        payload_patch: dict[str, Any],
    ) -> JobRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> JobRecord:
            current = await self._get_job(conn, kb_id, job_id, for_update=True)
            current.payload = {**current.payload, **payload_patch}
            current.updated_at = utc_now_iso()
            await self._save_job(conn, current)
            return current

        return await self._write(write)

    async def update_job_progress(
        self,
        kb_id: str,
        job_id: str,
        *,
        progress: float | None = None,
        completed_items: int | None = None,
        stage: str | None = None,
        result_patch: dict[str, Any] | None = None,
    ) -> JobRecord:
        """Patch live progress on a running job WITHOUT changing its status.

        Unlike :meth:`transition_job` this never touches ``status`` (so it
        bypasses the status state-machine and can be called repeatedly while a
        job runs) and never touches ``error_*`` / timestamps. ``result_patch``
        is shallow-merged into the existing ``result``. Patches are silently
        ignored once the job has left an active state, so a late progress poll
        cannot resurrect / overwrite a finished job's terminal snapshot.
        """
        await self._ensure_initialized()

        async def write(conn: Any) -> JobRecord:
            current = await self._get_job(conn, kb_id, job_id, for_update=True)
            if current.status not in {"running", "retrying", "cancelling"}:
                return current
            if progress is not None:
                current.progress = progress
            if completed_items is not None:
                current.completed_items = completed_items
            if stage is not None:
                current.stage = stage
            if result_patch is not None:
                current.result = {**(current.result or {}), **result_patch}
            current.updated_at = utc_now_iso()
            await self._save_job(conn, current)
            return current

        return await self._write(write)

    async def reset_job_for_retry(
        self, kb_id: str, job_id: str, *, new_idempotency_key: str | None
    ) -> JobRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> JobRecord:
            current = await self._get_job(conn, kb_id, job_id, for_update=True)
            if current.status not in {"failed", "cancelled"}:
                raise InvalidJobTransitionError(
                    f"Cannot retry job '{job_id}' from {current.status}"
                )
            if current.retry_count >= current.max_retries:
                raise InvalidJobTransitionError(
                    f"Job '{job_id}' has reached max_retries={current.max_retries}"
                )
            now = utc_now_iso()
            current.status = "queued"
            current.progress = 0.0
            current.completed_items = 0
            current.failed_items = 0
            current.result = None
            current.error_code = None
            current.error_message = None
            current.retry_count += 1
            if new_idempotency_key is not None:
                current.idempotency_key = new_idempotency_key
            current.updated_at = now
            current.queued_at = now
            current.started_at = None
            current.finished_at = None
            current.cancelled_at = None
            await self._save_job(conn, current)
            return current

        return await self._write(write)

    async def create_config_version(
        self, record: ConfigVersionRecord
    ) -> ConfigVersionRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> ConfigVersionRecord:
            next_version = (
                await conn.fetchval(
                    "SELECT COALESCE(MAX(version), 0) + 1 FROM kb_config_versions WHERE kb_id = $1",
                    record.kb_id,
                )
                or 1
            )
            persisted = ConfigVersionRecord(
                id=record.id,
                kb_id=record.kb_id,
                workspace=record.workspace,
                version=int(next_version),
                config=record.config,
                parser_hash=record.parser_hash,
                index_hash=record.index_hash,
                query_hash=record.query_hash,
                created_at=record.created_at,
                activated_at=None,
                created_by=record.created_by,
            )
            await self._insert_config_version(conn, persisted)
            return persisted

        return await self._write(write)

    async def list_config_versions(
        self, kb_id: str, *, limit: int = 50, offset: int = 0
    ) -> tuple[list[ConfigVersionRecord], int]:
        await self._ensure_initialized()
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        async with self._pool_or_raise().acquire() as conn:
            total = await conn.fetchval(
                "SELECT COUNT(*) FROM kb_config_versions WHERE kb_id = $1", kb_id
            )
            rows = await conn.fetch(
                """
                SELECT data_json FROM kb_config_versions
                WHERE kb_id = $1
                ORDER BY version DESC
                LIMIT $2 OFFSET $3
                """,
                kb_id,
                limit,
                offset,
            )
        return [_config_from_row(row) for row in rows], int(total or 0)

    async def get_config_version(
        self, kb_id: str, version_id: str
    ) -> ConfigVersionRecord:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT data_json FROM kb_config_versions
                WHERE kb_id = $1 AND id = $2
                """,
                kb_id,
                version_id,
            )
        if row is None:
            raise MetadataRecordNotFoundError(
                f"Config version '{version_id}' not found"
            )
        return _config_from_row(row)

    async def mark_config_version_activated(
        self, kb_id: str, version_id: str
    ) -> ConfigVersionRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> ConfigVersionRecord:
            row = await conn.fetchrow(
                """
                SELECT data_json FROM kb_config_versions
                WHERE kb_id = $1 AND id = $2
                FOR UPDATE
                """,
                kb_id,
                version_id,
            )
            if row is None:
                raise MetadataRecordNotFoundError(
                    f"Config version '{version_id}' not found"
                )
            record = _config_from_row(row)
            record.activated_at = utc_now_iso()
            await self._save_config_version(conn, record)
            return record

        return await self._write(write)

    async def get_enterprise_user_by_username(
        self, username: str
    ) -> EnterpriseUserRecord | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, username, status, tenant_id, created_at, updated_at,
                       data_json
                FROM enterprise_users WHERE username = $1
                """,
                username,
            )
        return _enterprise_user_from_row(row) if row is not None else None

    async def get_enterprise_user_by_id(
        self, user_id: str
    ) -> EnterpriseUserRecord | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, username, status, tenant_id, created_at, updated_at,
                       data_json
                FROM enterprise_users WHERE id = $1
                """,
                user_id,
            )
        return _enterprise_user_from_row(row) if row is not None else None

    async def list_enterprise_users(self) -> list[EnterpriseUserRecord]:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, username, status, tenant_id, created_at, updated_at,
                       data_json
                FROM enterprise_users
                ORDER BY created_at ASC, id ASC
                """
            )
        return [_enterprise_user_from_row(row) for row in rows]

    async def upsert_enterprise_user(
        self,
        user: EnterpriseUserRecord,
        *,
        expected_updated_at: Any = _EXPECTATION_UNSET,
        expected_token_version: Any = _EXPECTATION_UNSET,
        expected_tenant_id: Any = _EXPECTATION_UNSET,
    ) -> EnterpriseUserRecord:
        await self._ensure_initialized()
        saved, _membership = await self._write(
            lambda conn: self._upsert_enterprise_user_with_membership(
                conn,
                user,
                membership=None,
                expected_updated_at=expected_updated_at,
                expected_token_version=expected_token_version,
                expected_tenant_id=expected_tenant_id,
                allow_tenant_change=False,
            )
        )
        return saved

    async def update_enterprise_user_cas(
        self,
        user: EnterpriseUserRecord,
        *,
        expected_updated_at: str,
        expected_token_version: int,
        expected_tenant_id: str | None,
    ) -> EnterpriseUserRecord:
        return await self.upsert_enterprise_user(
            user,
            expected_updated_at=expected_updated_at,
            expected_token_version=expected_token_version,
            expected_tenant_id=expected_tenant_id,
        )

    async def upsert_enterprise_user_with_membership(
        self,
        user: EnterpriseUserRecord,
        membership: EnterpriseTenantMembershipRecord | None,
        *,
        expected_updated_at: Any = _EXPECTATION_UNSET,
        expected_token_version: Any = _EXPECTATION_UNSET,
        expected_tenant_id: Any = _EXPECTATION_UNSET,
        expected_membership: Any = _EXPECTATION_UNSET,
    ) -> tuple[EnterpriseUserRecord, EnterpriseTenantMembershipRecord | None]:
        """Atomically save a user and its zero-or-one canonical membership."""
        await self._ensure_initialized()
        return await self._write(
            lambda conn: self._upsert_enterprise_user_with_membership(
                conn,
                user,
                membership=membership,
                expected_updated_at=expected_updated_at,
                expected_token_version=expected_token_version,
                expected_tenant_id=expected_tenant_id,
                expected_membership=expected_membership,
                allow_tenant_change=True,
            )
        )

    async def delete_enterprise_user(
        self,
        user_id: str,
        *,
        expected_updated_at: Any = _EXPECTATION_UNSET,
        expected_token_version: Any = _EXPECTATION_UNSET,
        expected_tenant_id: Any = _EXPECTATION_UNSET,
        expected_membership: Any = _EXPECTATION_UNSET,
    ) -> bool:
        """Delete a user and cascade-remove related tenant memberships, KB ACLs,
        per-user query settings and chat projects/sessions.  Returns ``True`` if
        the user existed."""
        await self._ensure_initialized()

        async def write(conn: Any) -> bool:
            current_row = await conn.fetchrow(
                """
                SELECT id, username, status, tenant_id, created_at, updated_at,
                       data_json
                FROM enterprise_users WHERE id = $1 FOR UPDATE
                """,
                user_id,
            )
            conditional = any(
                value is not _EXPECTATION_UNSET
                for value in (
                    expected_updated_at,
                    expected_token_version,
                    expected_tenant_id,
                    expected_membership,
                )
            )
            if current_row is None:
                if conditional:
                    raise MetadataConflictError(
                        "enterprise_user",
                        user_id,
                        expected={"exists": True},
                        current={"exists": False},
                    )
                return False
            current_user = _enterprise_user_from_row(current_row)
            if any(
                value is not _EXPECTATION_UNSET
                for value in (
                    expected_updated_at,
                    expected_token_version,
                    expected_tenant_id,
                )
            ):
                _assert_enterprise_user_write_preconditions(
                    current_user,
                    current_user,
                    expected_updated_at=expected_updated_at,
                    expected_token_version=expected_token_version,
                    expected_tenant_id=expected_tenant_id,
                    allow_tenant_change=False,
                )
            if expected_membership is not _EXPECTATION_UNSET:
                membership_rows = await conn.fetch(
                    """
                    SELECT tenant_id, user_id, role, granted_by, created_at,
                           updated_at, data_json
                    FROM enterprise_tenant_memberships
                    WHERE user_id = $1 ORDER BY tenant_id ASC
                    FOR UPDATE
                    """,
                    user_id,
                )
                _assert_enterprise_user_membership_precondition(
                    user_id,
                    [_tenant_membership_from_row(row) for row in membership_rows],
                    expected_membership=expected_membership,
                )
            # Cascade: remove related records first.
            await conn.execute(
                "DELETE FROM enterprise_tenant_memberships WHERE user_id = $1",
                user_id,
            )
            await conn.execute(
                "DELETE FROM enterprise_tenant_user_kb_overrides WHERE user_id = $1",
                user_id,
            )
            await conn.execute(
                "DELETE FROM enterprise_kb_acl WHERE user_id = $1",
                user_id,
            )
            await conn.execute(
                "DELETE FROM enterprise_user_kb_query_settings WHERE user_id = $1",
                user_id,
            )
            await conn.execute(
                "DELETE FROM enterprise_chat_messages WHERE user_id = $1",
                user_id,
            )
            await conn.execute(
                "DELETE FROM enterprise_chat_sessions WHERE user_id = $1",
                user_id,
            )
            await conn.execute(
                "DELETE FROM enterprise_chat_projects WHERE user_id = $1",
                user_id,
            )
            status = await conn.execute(
                "DELETE FROM enterprise_users WHERE id = $1",
                user_id,
            )
            return _rowcount(status) > 0

        return await self._write(write)

    async def get_enterprise_user_kb_query_settings(
        self, user_id: str, kb_id: str
    ) -> EnterpriseUserKBQuerySettingsRecord | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT data_json FROM enterprise_user_kb_query_settings
                WHERE user_id = $1 AND kb_id = $2
                """,
                user_id,
                kb_id,
            )
        return (
            _enterprise_user_kb_query_settings_from_row(row)
            if row is not None
            else None
        )

    async def upsert_enterprise_user_kb_query_settings(
        self, record: EnterpriseUserKBQuerySettingsRecord
    ) -> EnterpriseUserKBQuerySettingsRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> EnterpriseUserKBQuerySettingsRecord:
            await conn.execute(
                """
                INSERT INTO enterprise_user_kb_query_settings (
                    user_id, kb_id, user_prompt, created_at, updated_at, data_json
                ) VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                ON CONFLICT (user_id, kb_id) DO UPDATE SET
                    user_prompt = excluded.user_prompt,
                    updated_at = excluded.updated_at,
                    data_json = jsonb_set(
                        excluded.data_json,
                        '{created_at}',
                        to_jsonb(enterprise_user_kb_query_settings.created_at)
                    )
                """,
                record.user_id,
                record.kb_id,
                record.user_prompt,
                record.created_at,
                record.updated_at,
                _record_json(record),
            )
            row = await conn.fetchrow(
                """
                SELECT data_json FROM enterprise_user_kb_query_settings
                WHERE user_id = $1 AND kb_id = $2
                """,
                record.user_id,
                record.kb_id,
            )
            if row is None:
                raise MetadataRecordNotFoundError("User KB query settings not found")
            return _enterprise_user_kb_query_settings_from_row(row)

        return await self._write(write)

    async def delete_enterprise_user_kb_query_settings(
        self, user_id: str, kb_id: str
    ) -> bool:
        await self._ensure_initialized()

        async def write(conn: Any) -> bool:
            status = await conn.execute(
                """
                DELETE FROM enterprise_user_kb_query_settings
                WHERE user_id = $1 AND kb_id = $2
                """,
                user_id,
                kb_id,
            )
            return _rowcount(status) > 0

        return await self._write(write)

    async def create_chat_project(self, record: ChatProjectRecord) -> ChatProjectRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> ChatProjectRecord:
            await conn.execute(
                """
                INSERT INTO enterprise_chat_projects (
                    id, user_id, name, created_at, updated_at, data_json
                ) VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                """,
                record.id,
                record.user_id,
                record.name,
                record.created_at,
                record.updated_at,
                _record_json(record),
            )
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_chat_projects WHERE id = $1",
                record.id,
            )
            if row is None:
                raise MetadataRecordNotFoundError("Chat project not found")
            return _chat_project_from_row(row)

        return await self._write(write)

    async def get_chat_project(
        self, user_id: str, project_id: str
    ) -> ChatProjectRecord | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT data_json FROM enterprise_chat_projects
                WHERE id = $1 AND user_id = $2
                """,
                project_id,
                user_id,
            )
        return _chat_project_from_row(row) if row is not None else None

    async def list_chat_projects(
        self, user_id: str, *, limit: int = 50, offset: int = 0
    ) -> tuple[list[ChatProjectRecord], int]:
        await self._ensure_initialized()
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        async with self._pool_or_raise().acquire() as conn:
            total = await conn.fetchval(
                "SELECT COUNT(*) FROM enterprise_chat_projects WHERE user_id = $1",
                user_id,
            )
            rows = await conn.fetch(
                """
                SELECT data_json FROM enterprise_chat_projects
                WHERE user_id = $1
                ORDER BY updated_at DESC, id DESC
                LIMIT $2 OFFSET $3
                """,
                user_id,
                limit,
                offset,
            )
        return [_chat_project_from_row(row) for row in rows], int(total)

    async def rename_chat_project(
        self, user_id: str, project_id: str, *, name: str
    ) -> ChatProjectRecord | None:
        await self._ensure_initialized()

        async def write(conn: Any) -> ChatProjectRecord | None:
            row = await conn.fetchrow(
                """
                SELECT data_json FROM enterprise_chat_projects
                WHERE id = $1 AND user_id = $2
                FOR UPDATE
                """,
                project_id,
                user_id,
            )
            if row is None:
                return None
            record = _chat_project_from_row(row)
            record.name = name
            record.updated_at = utc_now_iso()
            await conn.execute(
                """
                UPDATE enterprise_chat_projects
                SET name = $3, updated_at = $4, data_json = $5::jsonb
                WHERE id = $1 AND user_id = $2
                """,
                project_id,
                user_id,
                record.name,
                record.updated_at,
                _record_json(record),
            )
            return record

        return await self._write(write)

    async def delete_chat_project(
        self, user_id: str, project_id: str
    ) -> tuple[bool, int, int]:
        """Delete a chat project owned by ``user_id`` and cascade-delete its
        sessions and messages. Returns ``(deleted, deleted_sessions,
        deleted_messages)``."""
        await self._ensure_initialized()

        async def write(conn: Any) -> tuple[bool, int, int]:
            owner = await conn.fetchval(
                """
                SELECT id FROM enterprise_chat_projects
                WHERE id = $1 AND user_id = $2
                """,
                project_id,
                user_id,
            )
            if owner is None:
                return False, 0, 0
            messages_status = await conn.execute(
                "DELETE FROM enterprise_chat_messages WHERE project_id = $1",
                project_id,
            )
            sessions_status = await conn.execute(
                "DELETE FROM enterprise_chat_sessions WHERE project_id = $1",
                project_id,
            )
            await conn.execute(
                "DELETE FROM enterprise_chat_projects WHERE id = $1",
                project_id,
            )
            return True, _rowcount(sessions_status), _rowcount(messages_status)

        return await self._write(write)

    async def create_chat_session(self, record: ChatSessionRecord) -> ChatSessionRecord:
        """Insert a chat session. Raises :class:`MetadataRecordNotFoundError`
        when the parent project does not exist or is not owned by the user."""
        await self._ensure_initialized()

        async def write(conn: Any) -> ChatSessionRecord:
            project = await conn.fetchval(
                """
                SELECT id FROM enterprise_chat_projects
                WHERE id = $1 AND user_id = $2
                """,
                record.project_id,
                record.user_id,
            )
            if project is None:
                raise MetadataRecordNotFoundError(
                    f"Chat project '{record.project_id}' not found"
                )
            await conn.execute(
                """
                INSERT INTO enterprise_chat_sessions (
                    id, project_id, user_id, name, created_at, updated_at, data_json
                ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                """,
                record.id,
                record.project_id,
                record.user_id,
                record.name,
                record.created_at,
                record.updated_at,
                _record_json(record),
            )
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_chat_sessions WHERE id = $1",
                record.id,
            )
            if row is None:
                raise MetadataRecordNotFoundError("Chat session not found")
            return _chat_session_from_row(row)

        return await self._write(write)

    async def get_chat_session(
        self, user_id: str, project_id: str, session_id: str
    ) -> ChatSessionRecord | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT data_json FROM enterprise_chat_sessions
                WHERE id = $1 AND project_id = $2 AND user_id = $3
                """,
                session_id,
                project_id,
                user_id,
            )
        return _chat_session_from_row(row) if row is not None else None

    async def list_chat_sessions(
        self, user_id: str, project_id: str, *, limit: int = 50, offset: int = 0
    ) -> tuple[list[ChatSessionRecord], int]:
        await self._ensure_initialized()
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        async with self._pool_or_raise().acquire() as conn:
            total = await conn.fetchval(
                """
                SELECT COUNT(*) FROM enterprise_chat_sessions
                WHERE project_id = $1 AND user_id = $2
                """,
                project_id,
                user_id,
            )
            rows = await conn.fetch(
                """
                SELECT data_json FROM enterprise_chat_sessions
                WHERE project_id = $1 AND user_id = $2
                ORDER BY updated_at DESC, id DESC
                LIMIT $3 OFFSET $4
                """,
                project_id,
                user_id,
                limit,
                offset,
            )
        return [_chat_session_from_row(row) for row in rows], int(total)

    async def update_chat_session(
        self,
        user_id: str,
        project_id: str,
        session_id: str,
        *,
        name: str | None = None,
        context_rounds: int | None = None,
    ) -> ChatSessionRecord | None:
        """Update the provided fields of an owned session; ``None`` leaves a
        field unchanged. Returns ``None`` when the session is not found."""
        await self._ensure_initialized()

        async def write(conn: Any) -> ChatSessionRecord | None:
            row = await conn.fetchrow(
                """
                SELECT data_json FROM enterprise_chat_sessions
                WHERE id = $1 AND project_id = $2 AND user_id = $3
                FOR UPDATE
                """,
                session_id,
                project_id,
                user_id,
            )
            if row is None:
                return None
            record = _chat_session_from_row(row)
            if name is not None:
                record.name = name
            if context_rounds is not None:
                record.context_rounds = context_rounds
            record.updated_at = utc_now_iso()
            await conn.execute(
                """
                UPDATE enterprise_chat_sessions
                SET name = $4, updated_at = $5, data_json = $6::jsonb
                WHERE id = $1 AND project_id = $2 AND user_id = $3
                """,
                session_id,
                project_id,
                user_id,
                record.name,
                record.updated_at,
                _record_json(record),
            )
            return record

        return await self._write(write)

    async def delete_chat_session(
        self, user_id: str, project_id: str, session_id: str
    ) -> tuple[bool, int]:
        """Delete an owned session and cascade-delete its messages. Returns
        ``(deleted, deleted_messages)``."""
        await self._ensure_initialized()

        async def write(conn: Any) -> tuple[bool, int]:
            owner = await conn.fetchval(
                """
                SELECT id FROM enterprise_chat_sessions
                WHERE id = $1 AND project_id = $2 AND user_id = $3
                """,
                session_id,
                project_id,
                user_id,
            )
            if owner is None:
                return False, 0
            messages_status = await conn.execute(
                "DELETE FROM enterprise_chat_messages WHERE session_id = $1",
                session_id,
            )
            await conn.execute(
                "DELETE FROM enterprise_chat_sessions WHERE id = $1",
                session_id,
            )
            return True, _rowcount(messages_status)

        return await self._write(write)

    async def append_chat_messages(
        self, records: Sequence[ChatMessageRecord]
    ) -> list[ChatMessageRecord]:
        """Atomically append messages to a single owned session.

        Assigns consecutive per-session ``seq`` values and bumps the session's
        ``updated_at`` in the same transaction. All records must target the
        same ``(user_id, project_id, session_id)``. Raises
        :class:`MetadataRecordNotFoundError` when the session does not exist
        or is not owned by the user."""
        if not records:
            return []
        head = records[0]
        if any(
            record.session_id != head.session_id
            or record.project_id != head.project_id
            or record.user_id != head.user_id
            for record in records
        ):
            raise ValueError("All appended messages must target the same session")
        await self._ensure_initialized()

        async def write(conn: Any) -> list[ChatMessageRecord]:
            session_row = await conn.fetchrow(
                """
                SELECT data_json FROM enterprise_chat_sessions
                WHERE id = $1 AND project_id = $2 AND user_id = $3
                FOR UPDATE
                """,
                head.session_id,
                head.project_id,
                head.user_id,
            )
            if session_row is None:
                raise MetadataRecordNotFoundError(
                    f"Chat session '{head.session_id}' not found"
                )
            next_seq = (
                int(
                    await conn.fetchval(
                        """
                        SELECT COALESCE(MAX(seq), 0) FROM enterprise_chat_messages
                        WHERE session_id = $1
                        """,
                        head.session_id,
                    )
                )
                + 1
            )
            for index, record in enumerate(records):
                record.seq = next_seq + index
                await conn.execute(
                    """
                    INSERT INTO enterprise_chat_messages (
                        id, session_id, project_id, user_id, seq, created_at,
                        data_json
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                    """,
                    record.id,
                    record.session_id,
                    record.project_id,
                    record.user_id,
                    record.seq,
                    record.created_at,
                    _record_json(record),
                )
            session = _chat_session_from_row(session_row)
            session.updated_at = utc_now_iso()
            await conn.execute(
                """
                UPDATE enterprise_chat_sessions
                SET updated_at = $2, data_json = $3::jsonb
                WHERE id = $1
                """,
                head.session_id,
                session.updated_at,
                _record_json(session),
            )
            rows = await conn.fetch(
                """
                SELECT data_json FROM enterprise_chat_messages
                WHERE session_id = $1 AND seq >= $2
                ORDER BY seq ASC, id ASC
                """,
                head.session_id,
                next_seq,
            )
            return [_chat_message_from_row(row) for row in rows]

        return await self._write(write)

    async def list_chat_messages(
        self,
        user_id: str,
        project_id: str,
        session_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[ChatMessageRecord], int]:
        await self._ensure_initialized()
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        async with self._pool_or_raise().acquire() as conn:
            total = await conn.fetchval(
                """
                SELECT COUNT(*) FROM enterprise_chat_messages
                WHERE session_id = $1 AND project_id = $2 AND user_id = $3
                """,
                session_id,
                project_id,
                user_id,
            )
            rows = await conn.fetch(
                """
                SELECT data_json FROM enterprise_chat_messages
                WHERE session_id = $1 AND project_id = $2 AND user_id = $3
                ORDER BY seq ASC, id ASC
                LIMIT $4 OFFSET $5
                """,
                session_id,
                project_id,
                user_id,
                limit,
                offset,
            )
        return [_chat_message_from_row(row) for row in rows], int(total)

    async def delete_chat_message(
        self, user_id: str, project_id: str, session_id: str, message_id: str
    ) -> bool:
        await self._ensure_initialized()

        async def write(conn: Any) -> bool:
            status = await conn.execute(
                """
                DELETE FROM enterprise_chat_messages
                WHERE id = $1 AND session_id = $2 AND project_id = $3
                    AND user_id = $4
                """,
                message_id,
                session_id,
                project_id,
                user_id,
            )
            return _rowcount(status) > 0

        return await self._write(write)

    async def get_chat_message(
        self, user_id: str, project_id: str, session_id: str, message_id: str
    ) -> ChatMessageRecord | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT data_json FROM enterprise_chat_messages
                WHERE id = $1 AND session_id = $2 AND project_id = $3
                    AND user_id = $4
                """,
                message_id,
                session_id,
                project_id,
                user_id,
            )
        return _chat_message_from_row(row) if row is not None else None

    async def list_chat_messages_after_seq(
        self,
        user_id: str,
        project_id: str,
        session_id: str,
        *,
        after_seq: int,
        limit: int = 200,
    ) -> list[ChatMessageRecord]:
        """Messages of an owned session with ``seq > after_seq``, ascending."""
        await self._ensure_initialized()
        limit = max(1, min(int(limit), 1000))
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM enterprise_chat_messages
                WHERE session_id = $1 AND project_id = $2 AND user_id = $3
                    AND seq > $4
                ORDER BY seq ASC, id ASC
                LIMIT $5
                """,
                session_id,
                project_id,
                user_id,
                int(after_seq),
                limit,
            )
        return [_chat_message_from_row(row) for row in rows]

    async def record_chat_memory_episode(
        self, record: ChatMemoryEpisodeRecord
    ) -> None:
        await self._ensure_initialized()

        async def write(conn: Any) -> None:
            await conn.execute(
                """
                INSERT INTO enterprise_chat_memory_episodes (
                    episode_uuid, session_id, project_id, user_id,
                    first_seq, last_seq, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (episode_uuid) DO UPDATE SET
                    session_id = excluded.session_id,
                    project_id = excluded.project_id,
                    user_id = excluded.user_id,
                    first_seq = excluded.first_seq,
                    last_seq = excluded.last_seq,
                    created_at = excluded.created_at
                """,
                record.episode_uuid,
                record.session_id,
                record.project_id,
                record.user_id,
                record.first_seq,
                record.last_seq,
                record.created_at,
            )

        await self._write(write)

    async def get_chat_memory_watermark(
        self, user_id: str, project_id: str, session_id: str
    ) -> int:
        """Highest ingested ``last_seq`` for a session (0 when none)."""
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            value = await conn.fetchval(
                """
                SELECT COALESCE(MAX(last_seq), 0)
                FROM enterprise_chat_memory_episodes
                WHERE session_id = $1 AND project_id = $2 AND user_id = $3
                """,
                session_id,
                project_id,
                user_id,
            )
        return int(value or 0)

    async def find_chat_memory_episodes_covering(
        self, user_id: str, project_id: str, session_id: str, seq: int
    ) -> list[ChatMemoryEpisodeRecord]:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM enterprise_chat_memory_episodes
                WHERE session_id = $1 AND project_id = $2 AND user_id = $3
                    AND first_seq <= $4 AND last_seq >= $4
                ORDER BY first_seq ASC
                """,
                session_id,
                project_id,
                user_id,
                int(seq),
            )
        return [_chat_memory_episode_from_row(row) for row in rows]

    async def list_chat_memory_episodes_for_session(
        self, user_id: str, project_id: str, session_id: str
    ) -> list[ChatMemoryEpisodeRecord]:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM enterprise_chat_memory_episodes
                WHERE session_id = $1 AND project_id = $2 AND user_id = $3
                ORDER BY first_seq ASC
                """,
                session_id,
                project_id,
                user_id,
            )
        return [_chat_memory_episode_from_row(row) for row in rows]

    async def delete_chat_memory_episodes(
        self, episode_uuids: Sequence[str]
    ) -> int:
        ids = [uuid for uuid in episode_uuids if uuid]
        if not ids:
            return 0
        await self._ensure_initialized()

        async def write(conn: Any) -> int:
            status = await conn.execute(
                """
                DELETE FROM enterprise_chat_memory_episodes
                WHERE episode_uuid = ANY($1::text[])
                """,
                ids,
            )
            return _rowcount(status)

        return await self._write(write)

    async def delete_chat_memory_episodes_for_project(self, project_id: str) -> int:
        await self._ensure_initialized()

        async def write(conn: Any) -> int:
            status = await conn.execute(
                "DELETE FROM enterprise_chat_memory_episodes WHERE project_id = $1",
                project_id,
            )
            return _rowcount(status)

        return await self._write(write)

    async def delete_chat_memory_episodes_for_user(self, user_id: str) -> int:
        await self._ensure_initialized()

        async def write(conn: Any) -> int:
            status = await conn.execute(
                "DELETE FROM enterprise_chat_memory_episodes WHERE user_id = $1",
                user_id,
            )
            return _rowcount(status)

        return await self._write(write)

    async def list_chat_memory_backlog(
        self, *, limit: int = 100
    ) -> list[ChatMemoryBacklogItem]:
        """Sessions whose max message ``seq`` exceeds the ingestion watermark."""
        await self._ensure_initialized()
        limit = max(1, min(int(limit), 1000))
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT m.user_id AS user_id, m.project_id AS project_id,
                       m.session_id AS session_id,
                       COALESCE(e.max_last, 0) AS ingested_seq,
                       MAX(m.seq) AS max_seq
                FROM enterprise_chat_messages m
                LEFT JOIN (
                    SELECT session_id, MAX(last_seq) AS max_last
                    FROM enterprise_chat_memory_episodes
                    GROUP BY session_id
                ) e ON e.session_id = m.session_id
                GROUP BY m.session_id, m.project_id, m.user_id, e.max_last
                HAVING MAX(m.seq) > COALESCE(e.max_last, 0)
                ORDER BY m.session_id
                LIMIT $1
                """,
                limit,
            )
        return [
            ChatMemoryBacklogItem(
                user_id=str(row["user_id"]),
                project_id=str(row["project_id"]),
                session_id=str(row["session_id"]),
                ingested_seq=int(row["ingested_seq"]),
                max_seq=int(row["max_seq"]),
            )
            for row in rows
        ]

    async def count_chat_memory_episodes_for_project(
        self, user_id: str, project_id: str
    ) -> tuple[int, str | None]:
        """Return ``(episode_count, last_ingested_at)`` for a project (noop
        placeholder rows excluded)."""
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT COUNT(*) AS c, MAX(created_at) AS last_at
                FROM enterprise_chat_memory_episodes
                WHERE user_id = $1 AND project_id = $2
                    AND episode_uuid NOT LIKE 'noop\\_%'
                """,
                user_id,
                project_id,
            )
        if row is None:
            return 0, None
        return int(row["c"] or 0), (row["last_at"] if row["last_at"] else None)

    async def count_chat_memory_episodes(self) -> tuple[int, int, int]:
        """Global ``(episode_count, distinct_users, distinct_projects)`` for
        admin observability (noop rows excluded)."""
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT COUNT(*) AS c,
                       COUNT(DISTINCT user_id) AS u,
                       COUNT(DISTINCT project_id) AS p
                FROM enterprise_chat_memory_episodes
                WHERE episode_uuid NOT LIKE 'noop\\_%'
                """
            )
        if row is None:
            return 0, 0, 0
        return int(row["c"] or 0), int(row["u"] or 0), int(row["p"] or 0)

    async def set_enterprise_system_setting(
        self, key: str, value: str, *, updated_by: str | None = None
    ) -> None:
        await self._ensure_initialized()

        async def write(conn: Any) -> None:
            now = utc_now_iso()
            await conn.execute(
                """
                INSERT INTO enterprise_system_settings (
                    key, value, updated_by, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (key) DO UPDATE SET
                    value = excluded.value,
                    updated_by = excluded.updated_by,
                    updated_at = excluded.updated_at
                """,
                key,
                value,
                updated_by,
                now,
                now,
            )

        await self._write(write)

    async def get_enterprise_system_setting(
        self, key: str, default: str | None = None
    ) -> str | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            value = await conn.fetchval(
                "SELECT value FROM enterprise_system_settings WHERE key = $1",
                key,
            )
        return default if value is None else str(value)

    async def get_kb_lifecycle(self, kb_id: str) -> KBLifecycleRecord | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT kb_id, generation, state, activated_at, deleted_at, updated_at,
                       delete_job_id
                FROM enterprise_kb_lifecycle
                WHERE kb_id = $1
                """,
                kb_id,
            )
        return _kb_lifecycle_from_row(row) if row is not None else None

    async def assert_kb_not_deleting(
        self, kb_id: str, expected_generation: str | None = None
    ) -> KBLifecycleRecord | None:
        if expected_generation is not None:
            _validate_kb_lifecycle_identity(kb_id, expected_generation)
        current = await self.get_kb_lifecycle(kb_id)
        if current is not None and (
            current.state == "deleting"
            or (
                expected_generation is not None
                and current.generation != expected_generation
            )
        ):
            raise _kb_lifecycle_conflict(kb_id, expected_generation, current)
        return current

    async def activate_kb_generation(
        self,
        kb_id: str,
        generation: str,
        *,
        activated_at: str | None = None,
    ) -> KBLifecycleRecord:
        _validate_kb_lifecycle_identity(kb_id, generation)
        await self._ensure_initialized()

        async def write(conn: Any) -> KBLifecycleRecord:
            return await self._activate_kb_generation(
                conn,
                kb_id,
                generation,
                activated_at=activated_at or utc_now_iso(),
            )

        return await self._write(write)

    async def register_kb_generation(
        self,
        kb_id: str,
        generation: str,
        *,
        activated_at: str | None = None,
    ) -> KBLifecycleRecord:
        return await self.activate_kb_generation(
            kb_id, generation, activated_at=activated_at
        )

    async def assert_kb_generation(
        self, kb_id: str, expected_generation: str | None
    ) -> KBLifecycleRecord | None:
        await self._ensure_initialized()

        async def write(conn: Any) -> KBLifecycleRecord | None:
            return await self._assert_kb_generation(
                conn, kb_id, expected_generation
            )

        return await self._write(write)

    async def assert_current_kb_generation(
        self, kb_id: str, expected_generation: str | None
    ) -> KBLifecycleRecord | None:
        return await self.assert_kb_generation(kb_id, expected_generation)

    @asynccontextmanager
    async def job_execution_guard(
        self, job_id: str, *, wait: bool = True
    ) -> AsyncIterator[bool]:
        """Own one durable job with a session advisory lock.

        The job namespace (1263295563) is stable and independent from the KB
        shared/exclusive namespace. Session death provides crash-stop cleanup;
        this deliberately does not emulate a lease or add a run-token column.
        ``wait=False`` uses ``pg_try_advisory_lock``. Re-entry by the same task
        uses the context-bound operation session and PostgreSQL's balanced,
        re-entrant advisory-lock accounting.
        """

        _validate_job_execution_id(job_id)
        await self._ensure_initialized()
        await self._ensure_operation_lock_pool()
        async with self._operation_session() as conn:
            locked = False
            try:
                if wait:
                    await conn.execute(
                        "SELECT pg_advisory_lock("
                        "hashtextextended($1, 1263295563))",
                        job_id,
                    )
                    locked = True
                else:
                    locked = bool(
                        await conn.fetchval(
                            "SELECT pg_try_advisory_lock("
                            "hashtextextended($1, 1263295563))",
                            job_id,
                        )
                    )
                yield locked
            finally:
                if locked:
                    await self._unlock_operation_guard(
                        conn,
                        "SELECT pg_advisory_unlock("
                        "hashtextextended($1, 1263295563))",
                        job_id,
                    )

    @asynccontextmanager
    async def kb_write_guard(
        self, kb_id: str, expected_generation: str | None
    ) -> AsyncIterator[KBLifecycleRecord | None]:
        """Hold a session-level shared advisory lock for the full write.

        Exact same-task/store/KB/generation re-entry reuses the operation
        session and the already-held advisory lock instead of incrementing
        PostgreSQL's advisory-lock count a second time.
        """

        await self._ensure_initialized()
        # Reject a deletion that has already committed without queueing behind
        # its exclusive session lock; assert again after acquiring to close the
        # preflight race.
        preflight_current = await self.assert_kb_generation(
            kb_id, expected_generation
        )
        guard_key = (kb_id, expected_generation)
        inherited_states = _OPERATION_SESSION_STATES.get() or {}
        inherited_state = inherited_states.get(id(self))
        if (
            inherited_state is not None
            and inherited_state.store is self
            and inherited_state.owner_task is not asyncio.current_task()
            and inherited_state.kb_write_depths.get(guard_key, 0)
        ):
            # A parent operation session owns the advisory lock while awaiting
            # this child task. Borrow the fence without concurrently using the
            # parent's asyncpg connection; the owner waits before unlocking.
            inherited_state.kb_write_depths[guard_key] += 1
            idle_event = inherited_state.kb_write_idle_events[guard_key]
            idle_event.clear()
            try:
                yield preflight_current
            finally:
                remaining = inherited_state.kb_write_depths[guard_key] - 1
                if remaining:
                    inherited_state.kb_write_depths[guard_key] = remaining
                else:
                    inherited_state.kb_write_depths.pop(guard_key, None)
                    idle_event.set()
            return

        await self._ensure_operation_lock_pool()
        async with self._operation_session() as conn:
            states = _OPERATION_SESSION_STATES.get() or {}
            state = states.get(id(self))
            if (
                state is not None
                and state.store is self
                and state.owner_task is asyncio.current_task()
                and state.kb_write_depths.get(guard_key, 0)
            ):
                state.kb_write_depths[guard_key] += 1
                idle_event = state.kb_write_idle_events[guard_key]
                idle_event.clear()
                try:
                    async with conn.transaction():
                        current = await self._assert_kb_generation(
                            conn, kb_id, expected_generation
                        )
                    yield current
                finally:
                    remaining = state.kb_write_depths[guard_key] - 1
                    if remaining:
                        state.kb_write_depths[guard_key] = remaining
                    else:
                        state.kb_write_depths.pop(guard_key, None)
                        idle_event.set()
                return

            locked = False
            try:
                await conn.execute(
                    "SELECT pg_advisory_lock_shared("
                    "hashtextextended($1, 1263295562))",
                    kb_id,
                )
                locked = True
                async with conn.transaction():
                    current = await self._assert_kb_generation(
                        conn, kb_id, expected_generation
                    )
                idle_event: asyncio.Event | None = None
                if state is not None:
                    idle_event = asyncio.Event()
                    state.kb_write_idle_events[guard_key] = idle_event
                    state.kb_write_depths[guard_key] = 1
                try:
                    yield current
                finally:
                    if state is not None:
                        assert idle_event is not None
                        remaining = state.kb_write_depths[guard_key] - 1
                        if remaining:
                            state.kb_write_depths[guard_key] = remaining
                            await _wait_for_kb_guard_borrowers(idle_event)
                        else:
                            state.kb_write_depths.pop(guard_key, None)
                            idle_event.set()
                        state.kb_write_idle_events.pop(guard_key, None)
            finally:
                if locked:
                    await self._unlock_operation_guard(
                        conn,
                        "SELECT pg_advisory_unlock_shared("
                        "hashtextextended($1, 1263295562))",
                        kb_id,
                    )

    @asynccontextmanager
    async def kb_exclusive_operation_guard(
        self,
        kb_id: str,
    ) -> AsyncIterator[None]:
        """Hold only the session-level exclusive KB advisory lock."""

        await self._ensure_initialized()
        await self._ensure_operation_lock_pool()
        async with self._operation_session() as conn:
            locked = False
            try:
                await conn.execute(
                    "SELECT pg_advisory_lock("
                    "hashtextextended($1, 1263295562))",
                    kb_id,
                )
                locked = True
                yield
            finally:
                if locked:
                    await self._unlock_operation_guard(
                        conn,
                        "SELECT pg_advisory_unlock("
                        "hashtextextended($1, 1263295562))",
                        kb_id,
                    )

    @asynccontextmanager
    async def kb_deletion_guard(
        self,
        kb_id: str,
        expected_generation: str | None = None,
        delete_job_id: str | None = None,
    ) -> AsyncIterator[KBLifecycleRecord | None]:
        """Compatibility wrapper around the split delete lock/state APIs."""

        if (expected_generation is None) != (delete_job_id is None):
            raise MetadataStoreError(
                "KB deletion guard requires both generation and delete_job_id"
            )
        async with self.kb_exclusive_operation_guard(kb_id):
            if expected_generation is None or delete_job_id is None:
                yield None
                return
            _validate_kb_lifecycle_identity(kb_id, expected_generation)
            _validate_delete_job_id(delete_job_id)
            # Preserve the historical wrapper's same-session transaction while
            # the new public begin method remains an independent main-pool write.
            async with self._operation_session() as conn:
                async with conn.transaction():
                    current = await self._begin_kb_deletion(
                        conn,
                        kb_id,
                        expected_generation,
                        delete_job_id,
                    )
                yield current

    async def begin_kb_deletion(
        self,
        kb_id: str,
        generation: str,
        delete_job_id: str,
    ) -> KBLifecycleRecord:
        """Atomically bind ``active -> deleting``; exact retries are idempotent."""

        _validate_kb_lifecycle_identity(kb_id, generation)
        _validate_delete_job_id(delete_job_id)
        await self._ensure_initialized()

        async def write(conn: Any) -> KBLifecycleRecord:
            return await self._begin_kb_deletion(
                conn,
                kb_id,
                generation,
                delete_job_id,
            )

        return await self._write(write)

    async def complete_kb_deletion(
        self, kb_id: str, generation: str, delete_job_id: str
    ) -> KBLifecycleRecord:
        _validate_kb_lifecycle_identity(kb_id, generation)
        _validate_delete_job_id(delete_job_id)
        await self._ensure_initialized()

        async def write(conn: Any) -> KBLifecycleRecord:
            return await self._complete_kb_deletion(
                conn, kb_id, generation, delete_job_id
            )

        return await self._write(write)

    async def create_enterprise_api_key(
        self,
        record: EnterpriseAPIKeyRecord,
        *,
        expected_kb_generations: dict[str, str] | None = None,
    ) -> EnterpriseAPIKeyRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> EnterpriseAPIKeyRecord:
            if not isinstance(record.scopes, dict):
                raise MetadataStoreError("Service API key scopes must be an object")
            kb_roles = record.scopes.get("kb_roles", {})
            if not isinstance(kb_roles, dict):
                raise MetadataStoreError("Service API key kb_roles must be an object")
            for kb_id in kb_roles:
                if not isinstance(kb_id, str) or not kb_id:
                    raise MetadataStoreError("Service API key KB id must be non-empty")
            for kb_id in sorted(kb_roles):
                await self._assert_kb_generation(
                    conn,
                    kb_id,
                    (expected_kb_generations or {}).get(kb_id),
                )
            await conn.execute(
                """
                INSERT INTO enterprise_api_keys (
                    id, name, key_hash, key_preview, status, created_by, tenant_id,
                    created_at, updated_at, last_used_at, revoked_at, revoked_by,
                    data_json
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13::jsonb)
                """,
                record.id,
                record.name,
                record.key_hash,
                record.key_preview,
                record.status,
                record.created_by,
                record.tenant_id,
                record.created_at,
                record.updated_at,
                record.last_used_at,
                record.revoked_at,
                record.revoked_by,
                _record_json(record),
            )
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_api_keys WHERE id = $1", record.id
            )
            if row is None:
                raise MetadataRecordNotFoundError(f"API key '{record.id}' not found")
            return _enterprise_api_key_from_row(row)

        return await self._write(write)

    async def get_enterprise_api_key_by_hash(
        self, key_hash: str
    ) -> EnterpriseAPIKeyRecord | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_api_keys WHERE key_hash = $1",
                key_hash,
            )
        return _enterprise_api_key_from_row(row) if row is not None else None

    async def get_enterprise_api_key_by_id(
        self, key_id: str
    ) -> EnterpriseAPIKeyRecord | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_api_keys WHERE id = $1",
                key_id,
            )
        return _enterprise_api_key_from_row(row) if row is not None else None

    async def list_enterprise_api_keys(self) -> list[EnterpriseAPIKeyRecord]:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM enterprise_api_keys
                ORDER BY created_at DESC, id DESC
                """
            )
        return [_enterprise_api_key_from_row(row) for row in rows]

    async def revoke_enterprise_api_key(
        self,
        key_id: str,
        *,
        revoked_by: str | None = None,
        revoked_at: str | None = None,
    ) -> EnterpriseAPIKeyRecord | None:
        await self._ensure_initialized()

        async def write(conn: Any) -> EnterpriseAPIKeyRecord | None:
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_api_keys WHERE id = $1 FOR UPDATE",
                key_id,
            )
            if row is None:
                return None
            current = _enterprise_api_key_from_row(row)
            now = revoked_at or utc_now_iso()
            updated = EnterpriseAPIKeyRecord(
                **{
                    **current.to_dict(),
                    "status": "revoked",
                    "updated_at": now,
                    "revoked_at": now,
                    "revoked_by": revoked_by,
                }
            )
            await conn.execute(
                """
                UPDATE enterprise_api_keys
                SET status = $2, updated_at = $3, revoked_at = $4, revoked_by = $5,
                    data_json = $6::jsonb
                WHERE id = $1
                """,
                key_id,
                updated.status,
                updated.updated_at,
                updated.revoked_at,
                updated.revoked_by,
                _record_json(updated),
            )
            return updated

        return await self._write(write)

    async def mark_enterprise_api_key_used(
        self, key_id: str, *, last_used_at: str | None = None
    ) -> EnterpriseAPIKeyRecord | None:
        await self._ensure_initialized()

        async def write(conn: Any) -> EnterpriseAPIKeyRecord | None:
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_api_keys WHERE id = $1 FOR UPDATE",
                key_id,
            )
            if row is None:
                return None
            current = _enterprise_api_key_from_row(row)
            now = last_used_at or utc_now_iso()
            updated = EnterpriseAPIKeyRecord(
                **{
                    **current.to_dict(),
                    "last_used_at": now,
                    "updated_at": now,
                }
            )
            await conn.execute(
                """
                UPDATE enterprise_api_keys
                SET updated_at = $2, last_used_at = $3, data_json = $4::jsonb
                WHERE id = $1
                """,
                key_id,
                updated.updated_at,
                updated.last_used_at,
                _record_json(updated),
            )
            return updated

        return await self._write(write)

    async def create_enterprise_invitation(
        self, record: EnterpriseInvitationRecord
    ) -> EnterpriseInvitationRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> EnterpriseInvitationRecord:
            await conn.execute(
                """
                INSERT INTO enterprise_invitations (
                    id, token_hash, status, created_by, expires_at, used_by,
                    used_at, created_at, updated_at, data_json
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)
                """,
                record.id,
                record.token_hash,
                record.status,
                record.created_by,
                record.expires_at,
                record.used_by,
                record.used_at,
                record.created_at,
                record.updated_at,
                _record_json(record),
            )
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_invitations WHERE id = $1", record.id
            )
            if row is None:
                raise MetadataRecordNotFoundError(
                    f"Invitation '{record.id}' not found"
                )
            return _enterprise_invitation_from_row(row)

        return await self._write(write)

    async def get_enterprise_invitation_by_token_hash(
        self, token_hash: str
    ) -> EnterpriseInvitationRecord | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_invitations WHERE token_hash = $1",
                token_hash,
            )
        return _enterprise_invitation_from_row(row) if row is not None else None

    async def list_enterprise_invitations(self) -> list[EnterpriseInvitationRecord]:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM enterprise_invitations
                ORDER BY created_at DESC, id DESC
                """
            )
        return [_enterprise_invitation_from_row(row) for row in rows]

    async def consume_enterprise_invitation(
        self, token_hash: str, *, used_by: str | None, used_at: str | None = None
    ) -> EnterpriseInvitationRecord | None:
        await self._ensure_initialized()

        async def write(conn: Any) -> EnterpriseInvitationRecord | None:
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_invitations "
                "WHERE token_hash = $1 FOR UPDATE",
                token_hash,
            )
            if row is None:
                return None
            current = _enterprise_invitation_from_row(row)
            if current.status != "active":
                return None
            now = used_at or utc_now_iso()
            updated = EnterpriseInvitationRecord(
                **{
                    **current.to_dict(),
                    "status": "used",
                    "used_by": used_by,
                    "used_at": now,
                    "updated_at": now,
                }
            )
            await conn.execute(
                """
                UPDATE enterprise_invitations
                SET status = $2, used_by = $3, used_at = $4, updated_at = $5,
                    data_json = $6::jsonb
                WHERE id = $1
                """,
                current.id,
                updated.status,
                updated.used_by,
                updated.used_at,
                updated.updated_at,
                _record_json(updated),
            )
            return updated

        return await self._write(write)

    async def revoke_enterprise_invitation(
        self, invitation_id: str, *, revoked_at: str | None = None
    ) -> EnterpriseInvitationRecord | None:
        await self._ensure_initialized()

        async def write(conn: Any) -> EnterpriseInvitationRecord | None:
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_invitations WHERE id = $1 FOR UPDATE",
                invitation_id,
            )
            if row is None:
                return None
            current = _enterprise_invitation_from_row(row)
            if current.status != "active":
                return current
            now = revoked_at or utc_now_iso()
            updated = EnterpriseInvitationRecord(
                **{
                    **current.to_dict(),
                    "status": "revoked",
                    "updated_at": now,
                }
            )
            await conn.execute(
                """
                UPDATE enterprise_invitations
                SET status = $2, updated_at = $3, data_json = $4::jsonb
                WHERE id = $1
                """,
                current.id,
                updated.status,
                updated.updated_at,
                _record_json(updated),
            )
            return updated

        return await self._write(write)

    async def upsert_kb_acl(
        self, acl: KBACLRecord, *, expected_generation: str | None = None
    ) -> KBACLRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> KBACLRecord:
            await self._assert_kb_generation(conn, acl.kb_id, expected_generation)
            await conn.execute(
                """
                INSERT INTO enterprise_kb_acl (
                    kb_id, user_id, role, granted_by, created_at, updated_at, data_json
                ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                ON CONFLICT (kb_id, user_id) DO UPDATE SET
                    role = excluded.role,
                    granted_by = excluded.granted_by,
                    updated_at = excluded.updated_at,
                    data_json = excluded.data_json
                """,
                acl.kb_id,
                acl.user_id,
                acl.role,
                acl.granted_by,
                acl.created_at,
                acl.updated_at,
                _record_json(acl),
            )
            row = await conn.fetchrow(
                """
                SELECT data_json FROM enterprise_kb_acl
                WHERE kb_id = $1 AND user_id = $2
                """,
                acl.kb_id,
                acl.user_id,
            )
            if row is None:
                raise MetadataRecordNotFoundError("KB ACL grant not found")
            return _kb_acl_from_row(row)

        return await self._write(write)

    async def delete_kb_acl(
        self,
        kb_id: str,
        user_id: str,
        *,
        expected_generation: str | None = None,
    ) -> bool:
        await self._ensure_initialized()

        async def write(conn: Any) -> bool:
            await self._assert_kb_generation(conn, kb_id, expected_generation)
            status = await conn.execute(
                "DELETE FROM enterprise_kb_acl WHERE kb_id = $1 AND user_id = $2",
                kb_id,
                user_id,
            )
            return _rowcount(status) > 0

        return await self._write(write)

    async def list_kb_acl(self, kb_id: str) -> list[KBACLRecord]:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM enterprise_kb_acl
                WHERE kb_id = $1
                ORDER BY created_at ASC, user_id ASC
                """,
                kb_id,
            )
        return [_kb_acl_from_row(row) for row in rows]

    async def get_kb_acl_role(self, kb_id: str, user_id: str) -> str | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            role = await conn.fetchval(
                """
                SELECT role FROM enterprise_kb_acl
                WHERE kb_id = $1 AND user_id = $2
                """,
                kb_id,
                user_id,
            )
        return None if role is None else str(role)

    async def list_kb_ids_for_user(self, user_id: str) -> list[str]:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT kb_id FROM enterprise_kb_acl
                WHERE user_id = $1
                ORDER BY kb_id ASC
                """,
                user_id,
            )
        return [str(row["kb_id"]) for row in rows]

    async def upsert_enterprise_tenant(
        self, tenant: EnterpriseTenantRecord
    ) -> EnterpriseTenantRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> EnterpriseTenantRecord:
            await conn.execute(
                """
                INSERT INTO enterprise_tenants (id, name, status, created_at, data_json)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    name = excluded.name,
                    status = excluded.status,
                    data_json = excluded.data_json
                """,
                tenant.id,
                tenant.name,
                tenant.status,
                tenant.created_at,
                _record_json(tenant),
            )
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_tenants WHERE id = $1", tenant.id
            )
            if row is None:
                raise MetadataRecordNotFoundError(f"Tenant '{tenant.id}' not found")
            return _enterprise_tenant_from_row(row)

        return await self._write(write)

    async def get_enterprise_tenant_by_id(
        self, tenant_id: str
    ) -> EnterpriseTenantRecord | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_tenants WHERE id = $1", tenant_id
            )
        return _enterprise_tenant_from_row(row) if row is not None else None

    async def list_enterprise_tenants(self) -> list[EnterpriseTenantRecord]:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM enterprise_tenants
                ORDER BY created_at ASC, id ASC
                """
            )
        return [_enterprise_tenant_from_row(row) for row in rows]

    async def delete_enterprise_tenant(self, tenant_id: str) -> bool:
        await self._ensure_initialized()

        async def write(conn: Any) -> bool:
            exists = await conn.fetchval(
                "SELECT 1 FROM enterprise_tenants WHERE id = $1 FOR UPDATE", tenant_id
            )
            if exists is None:
                return False
            now = utc_now_iso()
            await conn.execute(
                """
                UPDATE enterprise_users
                SET tenant_id = NULL,
                    updated_at = $2,
                    data_json = jsonb_set(
                        jsonb_set(data_json, '{tenant_id}', 'null'::jsonb, true),
                        '{updated_at}', to_jsonb($2::text), true
                    )
                WHERE tenant_id = $1
                """,
                tenant_id,
                now,
            )
            await conn.execute(
                "DELETE FROM enterprise_tenant_memberships WHERE tenant_id = $1",
                tenant_id,
            )
            await conn.execute(
                "DELETE FROM enterprise_tenant_kb_acl WHERE tenant_id = $1",
                tenant_id,
            )
            await conn.execute(
                "DELETE FROM enterprise_tenant_user_kb_overrides WHERE tenant_id = $1",
                tenant_id,
            )
            status = await conn.execute(
                "DELETE FROM enterprise_tenants WHERE id = $1", tenant_id
            )
            return _rowcount(status) > 0

        return await self._write(write)

    async def upsert_tenant_membership(
        self, membership: EnterpriseTenantMembershipRecord
    ) -> EnterpriseTenantMembershipRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> EnterpriseTenantMembershipRecord:
            row = await conn.fetchrow(
                """
                SELECT id, username, status, tenant_id, created_at, updated_at,
                       data_json
                FROM enterprise_users WHERE id = $1 FOR UPDATE
                """,
                membership.user_id,
            )
            if row is None:
                raise MetadataRecordNotFoundError(
                    f"User '{membership.user_id}' not found"
                )
            user = _enterprise_user_from_row(row)
            updated_user = EnterpriseUserRecord(
                **{
                    **user.to_dict(),
                    "tenant_id": membership.tenant_id,
                    "updated_at": membership.updated_at,
                }
            )
            _saved_user, saved_membership = (
                await self._upsert_enterprise_user_with_membership(
                    conn,
                    updated_user,
                    membership=membership,
                    expected_updated_at=user.updated_at,
                    expected_token_version=user.token_version,
                    expected_tenant_id=user.tenant_id,
                    allow_tenant_change=True,
                )
            )
            if saved_membership is None:
                raise MetadataStoreError("Canonical membership was not saved")
            return saved_membership

        return await self._write(write)

    async def delete_tenant_membership(self, tenant_id: str, user_id: str) -> bool:
        await self._ensure_initialized()

        async def write(conn: Any) -> bool:
            user_row = await conn.fetchrow(
                """
                SELECT id, username, status, tenant_id, created_at, updated_at,
                       data_json
                FROM enterprise_users WHERE id = $1 FOR UPDATE
                """,
                user_id,
            )
            membership_row = await conn.fetchrow(
                """
                SELECT data_json FROM enterprise_tenant_memberships
                WHERE tenant_id = $1 AND user_id = $2
                FOR UPDATE
                """,
                tenant_id,
                user_id,
            )
            if membership_row is None:
                return False
            if user_row is not None:
                user = _enterprise_user_from_row(user_row)
            else:
                user = None
            if user is not None and user.tenant_id == tenant_id:
                cleared_user = EnterpriseUserRecord(
                    **{
                        **user.to_dict(),
                        "tenant_id": None,
                        "updated_at": utc_now_iso(),
                    }
                )
                await self._upsert_enterprise_user_with_membership(
                    conn,
                    cleared_user,
                    membership=None,
                    expected_updated_at=user.updated_at,
                    expected_token_version=user.token_version,
                    expected_tenant_id=user.tenant_id,
                    allow_tenant_change=True,
                )
            else:
                await conn.execute(
                    """
                    DELETE FROM enterprise_tenant_memberships
                    WHERE tenant_id = $1 AND user_id = $2
                    """,
                    tenant_id,
                    user_id,
                )
                await conn.execute(
                    """
                    DELETE FROM enterprise_tenant_user_kb_overrides
                    WHERE tenant_id = $1 AND user_id = $2
                    """,
                    tenant_id,
                    user_id,
                )
            return True

        return await self._write(write)

    async def list_tenant_memberships(
        self, tenant_id: str
    ) -> list[EnterpriseTenantMembershipRecord]:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT tenant_id, user_id, role, granted_by, created_at,
                       updated_at, data_json
                FROM enterprise_tenant_memberships
                WHERE tenant_id = $1
                ORDER BY created_at ASC, user_id ASC
                """,
                tenant_id,
            )
        return [_tenant_membership_from_row(row) for row in rows]

    async def list_user_tenant_memberships(
        self, user_id: str
    ) -> list[EnterpriseTenantMembershipRecord]:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT tenant_id, user_id, role, granted_by, created_at,
                       updated_at, data_json
                FROM enterprise_tenant_memberships
                WHERE user_id = $1
                ORDER BY tenant_id ASC
                """,
                user_id,
            )
        return [_tenant_membership_from_row(row) for row in rows]

    async def get_tenant_membership(
        self, tenant_id: str, user_id: str
    ) -> EnterpriseTenantMembershipRecord | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT tenant_id, user_id, role, granted_by, created_at,
                       updated_at, data_json
                FROM enterprise_tenant_memberships
                WHERE tenant_id = $1 AND user_id = $2
                """,
                tenant_id,
                user_id,
            )
        return _tenant_membership_from_row(row) if row is not None else None

    async def upsert_tenant_kb_acl(
        self,
        acl: EnterpriseTenantKBACLRecord,
        *,
        expected_generation: str | None = None,
    ) -> EnterpriseTenantKBACLRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> EnterpriseTenantKBACLRecord:
            await self._assert_kb_generation(conn, acl.kb_id, expected_generation)
            await conn.execute(
                """
                INSERT INTO enterprise_tenant_kb_acl (
                    tenant_id, kb_id, role, granted_by, created_at, updated_at, data_json
                ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                ON CONFLICT (tenant_id, kb_id) DO UPDATE SET
                    role = excluded.role,
                    granted_by = excluded.granted_by,
                    updated_at = excluded.updated_at,
                    data_json = excluded.data_json
                """,
                acl.tenant_id,
                acl.kb_id,
                acl.role,
                acl.granted_by,
                acl.created_at,
                acl.updated_at,
                _record_json(acl),
            )
            row = await conn.fetchrow(
                """
                SELECT data_json FROM enterprise_tenant_kb_acl
                WHERE tenant_id = $1 AND kb_id = $2
                """,
                acl.tenant_id,
                acl.kb_id,
            )
            if row is None:
                raise MetadataRecordNotFoundError("Tenant KB ACL grant not found")
            return _tenant_kb_acl_from_row(row)

        return await self._write(write)

    async def delete_tenant_kb_acl(
        self,
        tenant_id: str,
        kb_id: str,
        *,
        expected_generation: str | None = None,
    ) -> bool:
        await self._ensure_initialized()

        async def write(conn: Any) -> bool:
            await self._assert_kb_generation(conn, kb_id, expected_generation)
            status = await conn.execute(
                """
                DELETE FROM enterprise_tenant_kb_acl
                WHERE tenant_id = $1 AND kb_id = $2
                """,
                tenant_id,
                kb_id,
            )
            return _rowcount(status) > 0

        return await self._write(write)

    async def list_kb_tenant_acl(
        self, kb_id: str
    ) -> list[EnterpriseTenantKBACLRecord]:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM enterprise_tenant_kb_acl
                WHERE kb_id = $1
                ORDER BY created_at ASC, tenant_id ASC
                """,
                kb_id,
            )
        return [_tenant_kb_acl_from_row(row) for row in rows]

    async def get_tenant_kb_acl_role(self, tenant_id: str, kb_id: str) -> str | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            role = await conn.fetchval(
                """
                SELECT role FROM enterprise_tenant_kb_acl
                WHERE tenant_id = $1 AND kb_id = $2
                """,
                tenant_id,
                kb_id,
            )
        return None if role is None else str(role)

    async def list_kb_ids_for_tenants(self, tenant_ids: Sequence[str]) -> list[str]:
        await self._ensure_initialized()
        normalized_ids = sorted({tenant_id for tenant_id in tenant_ids if tenant_id})
        if not normalized_ids:
            return []
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT kb_id FROM enterprise_tenant_kb_acl
                WHERE tenant_id = ANY($1::text[])
                ORDER BY kb_id ASC
                """,
                normalized_ids,
            )
        return [str(row["kb_id"]) for row in rows]

    async def get_tenant_user_kb_override(
        self, tenant_id: str, kb_id: str, user_id: str
    ) -> EnterpriseTenantUserKBOverrideRecord | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT data_json FROM enterprise_tenant_user_kb_overrides
                WHERE tenant_id = $1 AND kb_id = $2 AND user_id = $3
                """,
                tenant_id,
                kb_id,
                user_id,
            )
        return _tenant_user_kb_override_from_row(row) if row is not None else None

    async def list_tenant_user_kb_overrides(
        self, tenant_id: str, kb_id: str
    ) -> list[EnterpriseTenantUserKBOverrideRecord]:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM enterprise_tenant_user_kb_overrides
                WHERE tenant_id = $1 AND kb_id = $2
                ORDER BY created_at ASC, user_id ASC
                """,
                tenant_id,
                kb_id,
            )
        return [_tenant_user_kb_override_from_row(row) for row in rows]

    async def list_user_tenant_kb_overrides(
        self,
        user_id: str,
        *,
        tenant_ids: Sequence[str] | None = None,
        kb_id: str | None = None,
    ) -> list[EnterpriseTenantUserKBOverrideRecord]:
        await self._ensure_initialized()
        clauses = ["user_id = $1"]
        params: list[Any] = [user_id]
        if tenant_ids is not None:
            normalized_ids = sorted({item for item in tenant_ids if item})
            if not normalized_ids:
                return []
            params.append(normalized_ids)
            clauses.append(f"tenant_id = ANY(${len(params)}::text[])")
        if kb_id is not None:
            params.append(kb_id)
            clauses.append(f"kb_id = ${len(params)}")
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT data_json FROM enterprise_tenant_user_kb_overrides
                WHERE {" AND ".join(clauses)}
                ORDER BY tenant_id ASC, kb_id ASC
                """,
                *params,
            )
        return [_tenant_user_kb_override_from_row(row) for row in rows]

    async def list_tenant_user_kb_overrides_for_user(
        self,
        user_id: str,
        *,
        tenant_ids: Sequence[str] | None = None,
        kb_id: str | None = None,
    ) -> list[EnterpriseTenantUserKBOverrideRecord]:
        return await self.list_user_tenant_kb_overrides(
            user_id, tenant_ids=tenant_ids, kb_id=kb_id
        )

    async def upsert_tenant_user_kb_override(
        self,
        record: EnterpriseTenantUserKBOverrideRecord,
        *,
        expected_generation: str | None = None,
        expected_user: Any = _EXPECTATION_UNSET,
        expected_membership: Any = _EXPECTATION_UNSET,
    ) -> EnterpriseTenantUserKBOverrideRecord:
        _validate_tenant_user_kb_override(record)
        await self._ensure_initialized()

        async def write(conn: Any) -> EnterpriseTenantUserKBOverrideRecord:
            await self._assert_kb_generation(
                conn, record.kb_id, expected_generation
            )
            user_row = await conn.fetchrow(
                """
                SELECT id, username, status, tenant_id, created_at, updated_at,
                       data_json
                FROM enterprise_users
                WHERE id = $1
                FOR UPDATE
                """,
                record.user_id,
            )
            membership_row = await conn.fetchrow(
                """
                SELECT tenant_id, user_id, role, granted_by, created_at,
                       updated_at, data_json
                FROM enterprise_tenant_memberships
                WHERE tenant_id = $1 AND user_id = $2
                FOR UPDATE
                """,
                record.tenant_id,
                record.user_id,
            )
            current_user = (
                _enterprise_user_from_row(user_row) if user_row is not None else None
            )
            current_membership = (
                _tenant_membership_from_row(membership_row)
                if membership_row is not None
                else None
            )
            _assert_tenant_user_kb_override_target_preconditions(
                record.tenant_id,
                record.user_id,
                current_user,
                current_membership,
                expected_user=expected_user,
                expected_membership=expected_membership,
            )
            if (
                current_user is None
                or current_user.tenant_id != record.tenant_id
                or current_membership is None
            ):
                raise InvalidTenantUserKBOverrideError(
                    "Override user must have a matching canonical tenant membership"
                )
            await conn.execute(
                """
                INSERT INTO enterprise_tenant_user_kb_overrides (
                    tenant_id, kb_id, user_id, effect, role, granted_by,
                    created_at, updated_at, data_json
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
                ON CONFLICT (tenant_id, kb_id, user_id) DO UPDATE SET
                    effect = excluded.effect,
                    role = excluded.role,
                    granted_by = excluded.granted_by,
                    updated_at = excluded.updated_at,
                    data_json = jsonb_set(
                        excluded.data_json,
                        '{created_at}',
                        to_jsonb(enterprise_tenant_user_kb_overrides.created_at)
                    )
                """,
                record.tenant_id,
                record.kb_id,
                record.user_id,
                record.effect,
                record.role,
                record.granted_by,
                record.created_at,
                record.updated_at,
                _record_json(record),
            )
            row = await conn.fetchrow(
                """
                SELECT data_json FROM enterprise_tenant_user_kb_overrides
                WHERE tenant_id = $1 AND kb_id = $2 AND user_id = $3
                """,
                record.tenant_id,
                record.kb_id,
                record.user_id,
            )
            if row is None:
                raise MetadataStoreError("Tenant user KB override was not saved")
            return _tenant_user_kb_override_from_row(row)

        return await self._write(write)

    async def delete_tenant_user_kb_override(
        self,
        tenant_id: str,
        kb_id: str,
        user_id: str,
        *,
        granted_by: str | None = None,
        updated_at: str | None = None,
        expected_generation: str | None = None,
        expected_user: Any = _EXPECTATION_UNSET,
        expected_membership: Any = _EXPECTATION_UNSET,
    ) -> EnterpriseTenantUserKBOverrideRecord:
        now = updated_at or utc_now_iso()
        return await self.upsert_tenant_user_kb_override(
            EnterpriseTenantUserKBOverrideRecord(
                tenant_id=tenant_id,
                kb_id=kb_id,
                user_id=user_id,
                effect="deny",
                role=None,
                granted_by=granted_by,
                created_at=now,
                updated_at=now,
            ),
            expected_generation=expected_generation,
            expected_user=expected_user,
            expected_membership=expected_membership,
        )

    async def reset_tenant_user_kb_override(
        self,
        tenant_id: str,
        kb_id: str,
        user_id: str,
        *,
        expected_generation: str | None = None,
        expected_user: Any = _EXPECTATION_UNSET,
        expected_membership: Any = _EXPECTATION_UNSET,
    ) -> bool:
        await self._ensure_initialized()

        async def write(conn: Any) -> bool:
            await self._assert_kb_generation(conn, kb_id, expected_generation)
            if (
                expected_user is not _EXPECTATION_UNSET
                or expected_membership is not _EXPECTATION_UNSET
            ):
                user_row = await conn.fetchrow(
                    """
                    SELECT id, username, status, tenant_id, created_at, updated_at,
                           data_json
                    FROM enterprise_users
                    WHERE id = $1
                    FOR UPDATE
                    """,
                    user_id,
                )
                membership_row = await conn.fetchrow(
                    """
                    SELECT tenant_id, user_id, role, granted_by, created_at,
                           updated_at, data_json
                    FROM enterprise_tenant_memberships
                    WHERE tenant_id = $1 AND user_id = $2
                    FOR UPDATE
                    """,
                    tenant_id,
                    user_id,
                )
                _assert_tenant_user_kb_override_target_preconditions(
                    tenant_id,
                    user_id,
                    _enterprise_user_from_row(user_row)
                    if user_row is not None
                    else None,
                    _tenant_membership_from_row(membership_row)
                    if membership_row is not None
                    else None,
                    expected_user=expected_user,
                    expected_membership=expected_membership,
                )
            status = await conn.execute(
                """
                DELETE FROM enterprise_tenant_user_kb_overrides
                WHERE tenant_id = $1 AND kb_id = $2 AND user_id = $3
                """,
                tenant_id,
                kb_id,
                user_id,
            )
            return _rowcount(status) > 0

        return await self._write(write)

    async def append_audit_event(
        self, event: AuditEventRecord
    ) -> AuditEventRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> AuditEventRecord:
            await conn.execute(
                """
                INSERT INTO enterprise_audit_events (
                    id, event_type, actor_user_id, actor_tenant_id, target_type,
                    target_id, created_at, data_json
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
                """,
                event.id,
                event.event_type,
                event.actor_user_id,
                event.actor_tenant_id,
                event.target_type,
                event.target_id,
                event.created_at,
                _record_json(event),
            )
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_audit_events WHERE id = $1",
                event.id,
            )
            if row is None:
                raise MetadataRecordNotFoundError(
                    f"Audit event '{event.id}' not found"
                )
            return _audit_event_from_row(row)

        return await self._write(write)

    async def list_audit_events(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        event_type: str | None = None,
        actor_user_id: str | None = None,
        actor_tenant_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
    ) -> list[AuditEventRecord]:
        await self._ensure_initialized()
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        clauses: list[str] = []
        params: list[Any] = []

        def _add(expr_tmpl: str, value: Any) -> None:
            params.append(value)
            clauses.append(expr_tmpl.format(idx=len(params)))

        for column, value in (
            ("event_type", event_type),
            ("actor_user_id", actor_user_id),
            ("actor_tenant_id", actor_tenant_id),
            ("target_type", target_type),
            ("target_id", target_id),
        ):
            if value:
                _add(f"{column} = ${{idx}}", value)
        if created_after:
            _add("created_at >= ${idx}", created_after)
        if created_before:
            _add("created_at <= ${idx}", created_before)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        limit_idx = len(params)
        params.append(offset)
        offset_idx = len(params)
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT data_json FROM enterprise_audit_events
                {where}
                ORDER BY created_at DESC, id DESC
                LIMIT ${limit_idx} OFFSET ${offset_idx}
                """,
                *params,
            )
        return [_audit_event_from_row(row) for row in rows]

    async def purge_kb_metadata(
        self,
        kb_id: str,
        generation: str | None = None,
        *,
        delete_job_id: str | None = None,
    ) -> dict[str, int]:
        await self._ensure_initialized()

        async def write(conn: Any) -> dict[str, int]:
            if generation is not None:
                _validate_kb_lifecycle_identity(kb_id, generation)
            strict_generation = generation
            if delete_job_id is not None:
                _validate_delete_job_id(delete_job_id)
                if strict_generation is None:
                    raise MetadataStoreError(
                        "Strict KB metadata purge requires a generation"
                    )
            now = utc_now_iso()
            lifecycle = await self._lock_kb_lifecycle(conn, kb_id)
            if delete_job_id is not None:
                assert strict_generation is not None
                if lifecycle is None:
                    raise _missing_kb_lifecycle_conflict(
                        kb_id,
                        strict_generation,
                        expected_state="deleting",
                        expected_delete_job_id=delete_job_id,
                    )
                if (
                    lifecycle.state != "deleting"
                    or lifecycle.generation != strict_generation
                    or lifecycle.delete_job_id != delete_job_id
                ):
                    raise _kb_lifecycle_conflict(
                        kb_id,
                        strict_generation,
                        lifecycle,
                        expected_state="deleting",
                        expected_delete_job_id=delete_job_id,
                    )
            elif lifecycle is None:
                tombstone_generation = generation or (
                    f"{_LEGACY_KB_TOMBSTONE_PREFIX}{kb_id}"
                )
                _validate_kb_lifecycle_identity(kb_id, tombstone_generation)
                await conn.execute(
                    """
                    INSERT INTO enterprise_kb_lifecycle (
                        kb_id, generation, state, activated_at, deleted_at, updated_at,
                        delete_job_id
                    ) VALUES ($1, $2, 'deleted', $3, $3, $3, NULL)
                    """,
                    kb_id,
                    tombstone_generation,
                    now,
                )
            else:
                legacy_retry = (
                    generation is None
                    and lifecycle.state == "deleted"
                    and lifecycle.generation
                    == f"{_LEGACY_KB_TOMBSTONE_PREFIX}{kb_id}"
                )
                if not legacy_retry and generation != lifecycle.generation:
                    raise _kb_lifecycle_conflict(kb_id, generation, lifecycle)
                if lifecycle.state == "deleting":
                    raise _kb_lifecycle_conflict(kb_id, generation, lifecycle)
                if lifecycle.state == "active":
                    status = await conn.execute(
                        """
                        UPDATE enterprise_kb_lifecycle
                        SET state = 'deleted', deleted_at = $3, updated_at = $3,
                            delete_job_id = NULL
                        WHERE kb_id = $1 AND generation = $2 AND state = 'active'
                        """,
                        kb_id,
                        lifecycle.generation,
                        now,
                    )
                    if _rowcount(status) != 1:
                        refreshed = await self._lock_kb_lifecycle(conn, kb_id)
                        if refreshed is None:
                            raise _missing_kb_lifecycle_conflict(
                                kb_id,
                                lifecycle.generation,
                                expected_state="active",
                            )
                        raise _kb_lifecycle_conflict(
                            kb_id, lifecycle.generation, refreshed
                        )

            source_key_count = await conn.fetchval(
                """
                SELECT COUNT(*) FROM kb_documents
                WHERE kb_id = $1 AND source_key IS NOT NULL
                """,
                kb_id,
            )
            counts = {"document_source_keys": int(source_key_count or 0)}
            updated_keys = 0
            key_rows = await conn.fetch(
                "SELECT id, data_json FROM enterprise_api_keys FOR UPDATE"
            )
            for key_row in key_rows:
                key_data = _loads_json_object(key_row["data_json"])
                scopes = key_data.get("scopes", {})
                if not isinstance(scopes, dict):
                    raise MetadataStoreError(
                        "Service API key scopes must be an object"
                    )
                kb_roles = scopes.get("kb_roles", {})
                if not isinstance(kb_roles, dict):
                    raise MetadataStoreError(
                        "Service API key kb_roles must be an object"
                    )
                if kb_id not in kb_roles:
                    continue
                updated_scopes = dict(scopes)
                updated_scopes["kb_roles"] = {
                    role_kb_id: role
                    for role_kb_id, role in kb_roles.items()
                    if role_kb_id != kb_id
                }
                updated_data = dict(key_data)
                updated_data["scopes"] = updated_scopes
                updated_data["updated_at"] = now
                await conn.execute(
                    """
                    UPDATE enterprise_api_keys
                    SET updated_at = $2, data_json = $3::jsonb
                    WHERE id = $1
                    """,
                    key_row["id"],
                    now,
                    _dumps_json(updated_data),
                )
                updated_keys += 1
            counts["enterprise_api_keys"] = updated_keys
            for table, label in (
                ("kb_document_artifacts", "document_artifacts"),
                ("enterprise_kb_acl", "enterprise_kb_acl"),
                ("enterprise_tenant_kb_acl", "enterprise_tenant_kb_acl"),
                (
                    "enterprise_tenant_user_kb_overrides",
                    "enterprise_tenant_user_kb_overrides",
                ),
                ("enterprise_user_kb_query_settings", "enterprise_user_kb_query_settings"),
                ("kb_config_versions", "kb_config_versions"),
            ):
                status = await conn.execute(f"DELETE FROM {table} WHERE kb_id = $1", kb_id)
                counts[label] = _rowcount(status)
            if delete_job_id is None:
                jobs_status = await conn.execute(
                    "DELETE FROM kb_jobs WHERE kb_id = $1", kb_id
                )
            else:
                jobs_status = await conn.execute(
                    "DELETE FROM kb_jobs WHERE kb_id = $1 AND id <> $2",
                    kb_id,
                    delete_job_id,
                )
            counts["jobs"] = _rowcount(jobs_status)
            documents_status = await conn.execute(
                "DELETE FROM kb_documents WHERE kb_id = $1", kb_id
            )
            counts["documents"] = _rowcount(documents_status)
            return counts

        return await self._write(write)

    async def recover_orphan_jobs(
        self,
        *,
        error_code: str = "worker_orphaned",
        error_message: str = "Job worker crashed before completion; please retry",
        resumable_job_types: set[str] | None = None,
        grace_seconds: float = 0.0,
    ) -> list[JobRecord]:
        """PostgreSQL parity for crash-stop, owner-aware orphan recovery.

        Resumable orphaned ``clear_kb`` jobs are requeued in place; other
        transient jobs retain the existing failed-recovery behavior.
        """

        await self._ensure_initialized()
        resumable = set(resumable_job_types or set())
        cutoff = _orphan_recovery_cutoff(grace_seconds)

        async def recover_queued(conn: Any) -> list[JobRecord]:
            rows = await conn.fetch(
                """
                SELECT data_json FROM kb_jobs
                WHERE status = 'queued'
                ORDER BY created_at ASC, id ASC
                FOR UPDATE
                """
            )
            now = utc_now_iso()
            updated: list[JobRecord] = []
            for row in rows:
                job = _job_from_row(row)
                if (
                    job.job_type in resumable
                    and (
                        job.document_id is not None
                        or job.job_type in _AGGREGATE_RESUMABLE_JOB_TYPES
                    )
                ):
                    continue
                job.status = "failed"
                job.error_code = error_code
                job.error_message = error_message
                job.updated_at = now
                if job.finished_at is None:
                    job.finished_at = now
                await self._save_job(conn, job)
                await self._recover_documents_for_job(
                    conn,
                    job,
                    error_code=error_code,
                    error_message=error_message,
                    now=now,
                )
                updated.append(job)
            return updated

        updated = await self._write(recover_queued)
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM kb_jobs
                WHERE status = ANY($1::text[]) AND updated_at <= $2
                ORDER BY updated_at ASC, id ASC
                """,
                sorted(_ORPHANED_JOB_STATUSES),
                cutoff,
            )
        candidates = [_job_from_row(row) for row in rows]

        for candidate in candidates:
            async with self.job_execution_guard(
                candidate.id, wait=False
            ) as acquired:
                if not acquired:
                    continue

                async def recover_candidate(conn: Any) -> JobRecord | None:
                    try:
                        current = await self._get_job(
                            conn,
                            candidate.kb_id,
                            candidate.id,
                            for_update=True,
                        )
                    except MetadataRecordNotFoundError:
                        return None
                    if (
                        current.status not in _ORPHANED_JOB_STATUSES
                        or current.updated_at > cutoff
                        or current.updated_at != candidate.updated_at
                        or not _same_job_execution_identity(candidate, current)
                    ):
                        return None
                    now = utc_now_iso()
                    requeue_clear = _should_requeue_orphaned_clear_job(
                        current,
                        resumable,
                    )
                    if requeue_clear:
                        current.status = "queued"
                        current.error_code = None
                        current.error_message = None
                        current.updated_at = now
                        current.queued_at = now
                        current.started_at = None
                        current.finished_at = None
                        current.cancelled_at = None
                        current.retry_count = min(
                            current.retry_count + 1,
                            current.max_retries,
                        )
                    else:
                        current.status = "failed"
                        current.error_code = error_code
                        current.error_message = error_message
                        current.updated_at = now
                        if current.finished_at is None:
                            current.finished_at = now
                    await self._save_job(conn, current)
                    if not requeue_clear:
                        await self._recover_documents_for_job(
                            conn,
                            current,
                            error_code=error_code,
                            error_message=error_message,
                            now=now,
                        )
                    return current

                recovered = await self._write(recover_candidate)
                if recovered is not None:
                    updated.append(recovered)
        return updated

    async def _recover_documents_for_job(
        self,
        conn: Any,
        job: JobRecord,
        *,
        error_code: str,
        error_message: str,
        now: str,
    ) -> None:
        document_ids = _job_recovery_document_ids(job)
        rows = await conn.fetch(
            """
            SELECT data_json FROM kb_documents
            WHERE kb_id = $1 AND deleted_at IS NULL
                AND status = ANY($2::text[])
            FOR UPDATE
            """,
            job.kb_id,
            sorted(_ORPHANED_DOCUMENT_STATUS_TARGETS),
        )
        for row in rows:
            document = _document_from_row(row)
            active_job_ids = _document_job_ids(document)
            belongs_to_job = (
                job.id in active_job_ids
                if active_job_ids
                else document.id in document_ids
            )
            if not belongs_to_job:
                continue
            target_status = _ORPHANED_DOCUMENT_STATUS_TARGETS.get(document.status)
            if target_status is None:
                continue
            document.status = target_status
            document.error_code = error_code
            document.error_message = error_message
            document.updated_at = now
            await self._save_document(conn, document)

    async def claim_next_worker_job(
        self, *, job_types: Sequence[str], max_queued_at: str | None = None
    ) -> JobRecord | None:
        await self._ensure_initialized()
        if not job_types:
            return None

        async def write(conn: Any) -> JobRecord | None:
            params: list[Any] = [list(job_types), sorted(_AGGREGATE_RESUMABLE_JOB_TYPES)]
            max_filter = ""
            if max_queued_at is not None:
                params.append(max_queued_at)
                max_filter = f" AND queued_at <= ${len(params)}"
            row = await conn.fetchrow(
                f"""
                SELECT data_json FROM kb_jobs
                WHERE status = 'queued'
                  AND job_type = ANY($1::text[])
                  AND (document_id IS NOT NULL OR job_type = ANY($2::text[]))
                  {max_filter}
                ORDER BY queued_at ASC, id ASC
                LIMIT 1
                FOR UPDATE SKIP LOCKED
                """,
                *params,
            )
            if row is None:
                return None
            job = _job_from_row(row)
            now = utc_now_iso()
            job.status = "running"
            job.progress = max(float(job.progress or 0.0), 0.1)
            job.updated_at = now
            if job.started_at is None:
                job.started_at = now
            await self._save_job(conn, job)
            return job

        return await self._write(write)

    async def _upsert_enterprise_user_with_membership(
        self,
        conn: Any,
        user: EnterpriseUserRecord,
        *,
        membership: EnterpriseTenantMembershipRecord | None,
        expected_updated_at: Any,
        expected_token_version: Any,
        expected_tenant_id: Any,
        expected_membership: Any = _EXPECTATION_UNSET,
        allow_tenant_change: bool,
    ) -> tuple[EnterpriseUserRecord, EnterpriseTenantMembershipRecord | None]:
        canonical_tenant = user.tenant_id
        if canonical_tenant is not None and (
            not isinstance(canonical_tenant, str)
            or not canonical_tenant.strip()
            or canonical_tenant != canonical_tenant.strip()
        ):
            raise MetadataStoreError(
                "Enterprise user tenant_id must be null or a normalized non-empty string"
            )
        if membership is not None:
            if membership.user_id != user.id:
                raise MetadataStoreError("Membership user_id must match the saved user")
            if canonical_tenant is None or membership.tenant_id != canonical_tenant:
                raise MetadataStoreError(
                    "Membership tenant_id must match the user's canonical tenant_id"
                )
            if membership.role not in _TENANT_MEMBERSHIP_ROLES:
                raise MetadataStoreError("Membership role is not recognized")

        # Serialize and validate against the snapshot that produced the whole
        # record. The candidate's new updated_at is deliberately not the CAS
        # expectation.
        current_row = await conn.fetchrow(
            """
            SELECT id, username, status, tenant_id, created_at, updated_at,
                   data_json
            FROM enterprise_users WHERE id = $1 FOR UPDATE
            """,
            user.id,
        )
        current_user = (
            _enterprise_user_from_row(current_row)
            if current_row is not None
            else None
        )
        _assert_enterprise_user_write_preconditions(
            user,
            current_user,
            expected_updated_at=expected_updated_at,
            expected_token_version=expected_token_version,
            expected_tenant_id=expected_tenant_id,
            allow_tenant_change=allow_tenant_change,
        )
        if expected_membership is not _EXPECTATION_UNSET:
            membership_rows = await conn.fetch(
                """
                SELECT tenant_id, user_id, role, granted_by, created_at,
                       updated_at, data_json
                FROM enterprise_tenant_memberships
                WHERE user_id = $1 ORDER BY tenant_id ASC
                FOR UPDATE
                """,
                user.id,
            )
            _assert_enterprise_user_membership_precondition(
                user.id,
                [_tenant_membership_from_row(row) for row in membership_rows],
                expected_membership=expected_membership,
            )
        existing_membership: EnterpriseTenantMembershipRecord | None = None
        if canonical_tenant is not None:
            row = await conn.fetchrow(
                """
                SELECT tenant_id, user_id, role, granted_by, created_at,
                       updated_at, data_json
                FROM enterprise_tenant_memberships
                WHERE tenant_id = $1 AND user_id = $2
                """,
                canonical_tenant,
                user.id,
            )
            if row is not None:
                existing_membership = _tenant_membership_from_row(row)

        selected_membership = membership
        if canonical_tenant is not None and selected_membership is None:
            selected_membership = existing_membership or EnterpriseTenantMembershipRecord(
                tenant_id=canonical_tenant,
                user_id=user.id,
                role="tenant_member",
                granted_by=None,
                created_at=user.updated_at,
                updated_at=user.updated_at,
            )
        elif selected_membership is not None and existing_membership is not None:
            selected_membership = EnterpriseTenantMembershipRecord(
                **{
                    **selected_membership.to_dict(),
                    "created_at": existing_membership.created_at,
                }
            )

        await conn.execute(
            """
            INSERT INTO enterprise_users (
                id, username, status, tenant_id, created_at, updated_at, data_json
            ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
            ON CONFLICT (id) DO UPDATE SET
                username = excluded.username,
                status = excluded.status,
                tenant_id = excluded.tenant_id,
                updated_at = excluded.updated_at,
                data_json = excluded.data_json
            """,
            user.id,
            user.username,
            user.status,
            canonical_tenant,
            user.created_at,
            user.updated_at,
            _record_json(user),
        )
        if canonical_tenant is None:
            await conn.execute(
                "DELETE FROM enterprise_tenant_memberships WHERE user_id = $1",
                user.id,
            )
        else:
            await conn.execute(
                """
                DELETE FROM enterprise_tenant_memberships
                WHERE user_id = $1 AND tenant_id <> $2
                """,
                user.id,
                canonical_tenant,
            )
        persisted_membership: EnterpriseTenantMembershipRecord | None = None
        if selected_membership is not None:
            await conn.execute(
                """
                INSERT INTO enterprise_tenant_memberships (
                    tenant_id, user_id, role, granted_by, created_at, updated_at,
                    data_json
                ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                ON CONFLICT (tenant_id, user_id) DO UPDATE SET
                    role = excluded.role,
                    granted_by = excluded.granted_by,
                    updated_at = excluded.updated_at,
                    data_json = jsonb_set(
                        excluded.data_json,
                        '{created_at}',
                        to_jsonb(enterprise_tenant_memberships.created_at)
                    )
                """,
                selected_membership.tenant_id,
                selected_membership.user_id,
                selected_membership.role,
                selected_membership.granted_by,
                selected_membership.created_at,
                selected_membership.updated_at,
                _record_json(selected_membership),
            )
            persisted_membership = selected_membership

        if canonical_tenant is None:
            await conn.execute(
                "DELETE FROM enterprise_tenant_user_kb_overrides WHERE user_id = $1",
                user.id,
            )
        else:
            await conn.execute(
                """
                DELETE FROM enterprise_tenant_user_kb_overrides
                WHERE user_id = $1 AND tenant_id <> $2
                """,
                user.id,
                canonical_tenant,
            )
        row = await conn.fetchrow(
            """
            SELECT id, username, status, tenant_id, created_at, updated_at,
                   data_json
            FROM enterprise_users WHERE id = $1
            """,
            user.id,
        )
        if row is None:
            raise MetadataRecordNotFoundError(f"User '{user.id}' not found")
        return _enterprise_user_from_row(row), persisted_membership

    async def _lock_kb_lifecycle(
        self, conn: Any, kb_id: str
    ) -> KBLifecycleRecord | None:
        # The advisory lock also serializes the no-row case, where FOR UPDATE
        # alone has nothing to lock. The fixed namespace is implicit in the
        # hash seed and all lifecycle writers use this helper.
        await conn.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 1263295561))",
            kb_id,
        )
        row = await conn.fetchrow(
            """
            SELECT kb_id, generation, state, activated_at, deleted_at, updated_at,
                   delete_job_id
            FROM enterprise_kb_lifecycle
            WHERE kb_id = $1
            FOR UPDATE
            """,
            kb_id,
        )
        return _kb_lifecycle_from_row(row) if row is not None else None

    async def _assert_kb_generation(
        self,
        conn: Any,
        kb_id: str,
        expected_generation: str | None,
    ) -> KBLifecycleRecord | None:
        if expected_generation is not None:
            _validate_kb_lifecycle_identity(kb_id, expected_generation)
        current = await self._lock_kb_lifecycle(conn, kb_id)
        if current is None:
            # KBs that predate lifecycle registration remain writable.
            return None
        if (
            current.state != "active"
            or expected_generation is None
            or expected_generation != current.generation
        ):
            raise _kb_lifecycle_conflict(kb_id, expected_generation, current)
        return current

    async def _activate_kb_generation(
        self,
        conn: Any,
        kb_id: str,
        generation: str,
        *,
        activated_at: str,
    ) -> KBLifecycleRecord:
        _validate_kb_lifecycle_identity(kb_id, generation)
        current = await self._lock_kb_lifecycle(conn, kb_id)
        if current is None:
            await conn.execute(
                """
                INSERT INTO enterprise_kb_lifecycle (
                    kb_id, generation, state, activated_at, deleted_at, updated_at,
                    delete_job_id
                ) VALUES ($1, $2, 'active', $3, NULL, $3, NULL)
                """,
                kb_id,
                generation,
                activated_at,
            )
            return KBLifecycleRecord(
                kb_id=kb_id,
                generation=generation,
                state="active",
                activated_at=activated_at,
                deleted_at=None,
                updated_at=activated_at,
                delete_job_id=None,
            )

        if current.state == "active":
            if current.generation == generation:
                return current
            raise _kb_lifecycle_conflict(kb_id, generation, current)
        if current.state == "deleting":
            raise _kb_lifecycle_conflict(kb_id, generation, current)
        if current.generation == generation:
            raise _kb_lifecycle_conflict(kb_id, generation, current)

        status = await conn.execute(
            """
            UPDATE enterprise_kb_lifecycle
            SET generation = $2, state = 'active', activated_at = $3,
                deleted_at = NULL, updated_at = $3, delete_job_id = NULL
            WHERE kb_id = $1 AND generation = $4 AND state = 'deleted'
            """,
            kb_id,
            generation,
            activated_at,
            current.generation,
        )
        if _rowcount(status) != 1:
            refreshed = await self._lock_kb_lifecycle(conn, kb_id)
            if refreshed is None:
                raise _missing_kb_lifecycle_conflict(
                    kb_id, generation, expected_state="deleted"
                )
            raise _kb_lifecycle_conflict(kb_id, generation, refreshed)
        return KBLifecycleRecord(
            kb_id=kb_id,
            generation=generation,
            state="active",
            activated_at=activated_at,
            deleted_at=None,
            updated_at=activated_at,
            delete_job_id=None,
        )

    async def _begin_kb_deletion(
        self,
        conn: Any,
        kb_id: str,
        generation: str,
        delete_job_id: str,
    ) -> KBLifecycleRecord:
        current = await self._lock_kb_lifecycle(conn, kb_id)
        if current is None:
            raise _missing_kb_lifecycle_conflict(
                kb_id,
                generation,
                expected_state="active",
                expected_delete_job_id=delete_job_id,
            )
        if current.state == "active":
            if current.generation != generation:
                raise _kb_lifecycle_conflict(kb_id, generation, current)
            now = utc_now_iso()
            status = await conn.execute(
                """
                UPDATE enterprise_kb_lifecycle
                SET state = 'deleting', delete_job_id = $3, deleted_at = NULL,
                    updated_at = $4
                WHERE kb_id = $1 AND generation = $2 AND state = 'active'
                    AND delete_job_id IS NULL
                """,
                kb_id,
                generation,
                delete_job_id,
                now,
            )
            if _rowcount(status) == 1:
                return KBLifecycleRecord(
                    kb_id=kb_id,
                    generation=generation,
                    state="deleting",
                    activated_at=current.activated_at,
                    deleted_at=None,
                    updated_at=now,
                    delete_job_id=delete_job_id,
                )
            refreshed = await self._lock_kb_lifecycle(conn, kb_id)
            if refreshed is None:
                raise _missing_kb_lifecycle_conflict(
                    kb_id,
                    generation,
                    expected_state="active",
                    expected_delete_job_id=delete_job_id,
                )
            current = refreshed

        if (
            current.state in {"deleting", "deleted"}
            and current.generation == generation
            and current.delete_job_id == delete_job_id
        ):
            return current
        raise _kb_lifecycle_conflict(
            kb_id,
            generation,
            current,
            expected_state="deleting",
            expected_delete_job_id=delete_job_id,
        )

    async def _complete_kb_deletion(
        self,
        conn: Any,
        kb_id: str,
        generation: str,
        delete_job_id: str,
    ) -> KBLifecycleRecord:
        current = await self._lock_kb_lifecycle(conn, kb_id)
        if current is None:
            raise _missing_kb_lifecycle_conflict(
                kb_id,
                generation,
                expected_state="deleting",
                expected_delete_job_id=delete_job_id,
            )
        if (
            current.state == "deleted"
            and current.generation == generation
            and current.delete_job_id == delete_job_id
        ):
            return current
        if not (
            current.state == "deleting"
            and current.generation == generation
            and current.delete_job_id == delete_job_id
        ):
            raise _kb_lifecycle_conflict(
                kb_id,
                generation,
                current,
                expected_state="deleting",
                expected_delete_job_id=delete_job_id,
            )
        now = utc_now_iso()
        status = await conn.execute(
            """
            UPDATE enterprise_kb_lifecycle
            SET state = 'deleted', deleted_at = $4, updated_at = $4
            WHERE kb_id = $1 AND generation = $2 AND state = 'deleting'
                AND delete_job_id = $3
            """,
            kb_id,
            generation,
            delete_job_id,
            now,
        )
        if _rowcount(status) != 1:
            refreshed = await self._lock_kb_lifecycle(conn, kb_id)
            if refreshed is None:
                raise _missing_kb_lifecycle_conflict(
                    kb_id,
                    generation,
                    expected_state="deleting",
                    expected_delete_job_id=delete_job_id,
                )
            if (
                refreshed.state == "deleted"
                and refreshed.generation == generation
                and refreshed.delete_job_id == delete_job_id
            ):
                return refreshed
            raise _kb_lifecycle_conflict(
                kb_id,
                generation,
                refreshed,
                expected_state="deleting",
                expected_delete_job_id=delete_job_id,
            )
        return KBLifecycleRecord(
            kb_id=kb_id,
            generation=generation,
            state="deleted",
            activated_at=current.activated_at,
            deleted_at=now,
            updated_at=now,
            delete_job_id=delete_job_id,
        )

    def _pool_or_raise(self) -> Any:
        if self._pool is None:
            raise RuntimeError("PostgreSQL metadata store is not initialized")
        return self._pool

    async def _create_pool(self, *, min_size: int, max_size: int) -> Any:
        asyncpg = _load_asyncpg()
        if self._dsn:
            return await asyncpg.create_pool(
                dsn=self._dsn,
                min_size=min_size,
                max_size=max_size,
            )
        return await asyncpg.create_pool(
            min_size=min_size,
            max_size=max_size,
            **self._connect_kwargs,
        )

    async def _ensure_operation_lock_pool(self) -> Any:
        pool = self._operation_lock_pool
        if pool is not None:
            return pool
        async with self._operation_lock_pool_init_lock:
            if self._operation_lock_pool is None:
                self._operation_lock_pool = await self._create_pool(
                    min_size=0,
                    max_size=self._operation_lock_pool_max_size,
                )
            return self._operation_lock_pool

    @asynccontextmanager
    async def _operation_session(self) -> AsyncIterator[Any]:
        """Reuse one operation-lock connection within the current task/store.

        Job ownership and nested KB shared/exclusive guards must be on the same
        PostgreSQL session. Otherwise an operation pool with ``max_size=1``
        self-deadlocks while the nested guard waits for the connection already
        held by its outer job guard. ContextVar state is task-checked because
        child tasks inherit context values but must still own independent
        sessions.
        """

        task = asyncio.current_task()
        states = _OPERATION_SESSION_STATES.get() or {}
        state = states.get(id(self))
        if state is not None and state.store is self and state.owner_task is task:
            state.depth += 1
            try:
                yield state.connection
            finally:
                state.depth -= 1
            return

        pool = await self._ensure_operation_lock_pool()
        acquire_context = pool.acquire()
        connection = await acquire_context.__aenter__()
        state = _OperationSessionState(
            store=self,
            owner_task=task,
            connection=connection,
        )
        next_states = dict(states)
        next_states[id(self)] = state
        token = _OPERATION_SESSION_STATES.set(next_states)
        try:
            yield connection
        finally:
            _OPERATION_SESSION_STATES.reset(token)
            release_task = asyncio.create_task(
                acquire_context.__aexit__(None, None, None)
            )
            try:
                await asyncio.shield(release_task)
            except asyncio.CancelledError:
                await asyncio.gather(release_task, return_exceptions=True)
                raise

    @staticmethod
    async def _unlock_operation_guard(
        conn: Any,
        statement: str,
        lock_id: str,
    ) -> None:
        unlock_task = asyncio.create_task(conn.execute(statement, lock_id))
        try:
            await asyncio.shield(unlock_task)
        except asyncio.CancelledError:
            # Do not return the session to the pool while its explicit unlock
            # is still running. Preserve cancellation after cleanup completes.
            await asyncio.gather(unlock_task, return_exceptions=True)
            raise

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def _write(self, callback: Callable[[Any], Awaitable[_T]]) -> _T:
        async with self._lock:
            async with self._pool_or_raise().acquire() as conn:
                async with conn.transaction():
                    return await callback(conn)

    async def _initialize_schema(self, conn: Any) -> None:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kb_metadata_schema (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS kb_documents (
                id TEXT PRIMARY KEY,
                kb_id TEXT NOT NULL,
                workspace TEXT NOT NULL,
                status TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                source_key TEXT,
                batch_id TEXT,
                deleted_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                data_json JSONB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_kb_documents_kb_status
                ON kb_documents (kb_id, status);
            CREATE INDEX IF NOT EXISTS idx_kb_documents_kb_source_hash
                ON kb_documents (kb_id, source_hash);
            CREATE INDEX IF NOT EXISTS idx_kb_documents_workspace
                ON kb_documents (workspace);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_kb_documents_source_key
                ON kb_documents (kb_id, source_key)
                WHERE source_key IS NOT NULL AND deleted_at IS NULL;

            CREATE TABLE IF NOT EXISTS kb_jobs (
                id TEXT PRIMARY KEY,
                kb_id TEXT NOT NULL,
                workspace TEXT NOT NULL,
                batch_id TEXT,
                document_id TEXT,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL,
                idempotency_key TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0,
                max_retries INTEGER NOT NULL DEFAULT 3,
                queued_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                data_json JSONB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_kb_jobs_kb_status
                ON kb_jobs (kb_id, status);
            CREATE INDEX IF NOT EXISTS idx_kb_jobs_kb_document
                ON kb_jobs (kb_id, document_id);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_kb_jobs_kb_type_idempotency
                ON kb_jobs (kb_id, job_type, idempotency_key)
                WHERE idempotency_key IS NOT NULL;

            CREATE TABLE IF NOT EXISTS kb_document_artifacts (
                id TEXT PRIMARY KEY,
                kb_id TEXT NOT NULL,
                workspace TEXT NOT NULL,
                document_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                data_json JSONB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_kb_artifacts_kb_document
                ON kb_document_artifacts (kb_id, document_id);
            CREATE INDEX IF NOT EXISTS idx_kb_artifacts_workspace_type
                ON kb_document_artifacts (workspace, artifact_type);

            CREATE TABLE IF NOT EXISTS kb_config_versions (
                id TEXT PRIMARY KEY,
                kb_id TEXT NOT NULL,
                workspace TEXT NOT NULL,
                version INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                data_json JSONB NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_kb_config_versions_kb_version_unique
                ON kb_config_versions (kb_id, version);
            CREATE INDEX IF NOT EXISTS idx_kb_config_versions_workspace
                ON kb_config_versions (workspace);

            CREATE TABLE IF NOT EXISTS enterprise_users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                tenant_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                data_json JSONB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_enterprise_users_status
                ON enterprise_users (status);
            CREATE INDEX IF NOT EXISTS idx_enterprise_users_tenant
                ON enterprise_users (tenant_id);

            CREATE TABLE IF NOT EXISTS enterprise_system_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_by TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS enterprise_kb_lifecycle (
                kb_id TEXT PRIMARY KEY,
                generation TEXT NOT NULL,
                state TEXT NOT NULL CHECK (
                    state IN ('active', 'deleting', 'deleted')
                ),
                activated_at TEXT NOT NULL,
                deleted_at TEXT,
                updated_at TEXT NOT NULL,
                delete_job_id TEXT,
                CHECK (kb_id <> '' AND kb_id = btrim(kb_id)),
                CHECK (generation <> '' AND generation = btrim(generation)),
                CHECK (delete_job_id IS NULL OR (
                    delete_job_id <> '' AND delete_job_id = btrim(delete_job_id)
                )),
                CHECK (
                    (state = 'active' AND deleted_at IS NULL AND delete_job_id IS NULL)
                    OR (
                        state = 'deleting' AND deleted_at IS NULL
                        AND delete_job_id IS NOT NULL
                    )
                    OR (state = 'deleted' AND deleted_at IS NOT NULL)
                )
            );

            CREATE TABLE IF NOT EXISTS enterprise_user_kb_query_settings (
                user_id TEXT NOT NULL,
                kb_id TEXT NOT NULL,
                user_prompt TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                data_json JSONB NOT NULL,
                PRIMARY KEY (user_id, kb_id),
                FOREIGN KEY (user_id) REFERENCES enterprise_users(id)
            );
            CREATE INDEX IF NOT EXISTS idx_enterprise_user_kb_query_settings_kb
                ON enterprise_user_kb_query_settings (kb_id, user_id);

            CREATE TABLE IF NOT EXISTS enterprise_chat_projects (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                data_json JSONB NOT NULL,
                FOREIGN KEY (user_id) REFERENCES enterprise_users(id)
            );
            CREATE INDEX IF NOT EXISTS idx_enterprise_chat_projects_user
                ON enterprise_chat_projects (user_id, updated_at DESC);

            CREATE TABLE IF NOT EXISTS enterprise_chat_sessions (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                data_json JSONB NOT NULL,
                FOREIGN KEY (project_id) REFERENCES enterprise_chat_projects(id),
                FOREIGN KEY (user_id) REFERENCES enterprise_users(id)
            );
            CREATE INDEX IF NOT EXISTS idx_enterprise_chat_sessions_project
                ON enterprise_chat_sessions (project_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_enterprise_chat_sessions_user
                ON enterprise_chat_sessions (user_id, project_id);

            CREATE TABLE IF NOT EXISTS enterprise_chat_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                seq BIGINT NOT NULL,
                created_at TEXT NOT NULL,
                data_json JSONB NOT NULL,
                FOREIGN KEY (session_id) REFERENCES enterprise_chat_sessions(id),
                FOREIGN KEY (user_id) REFERENCES enterprise_users(id)
            );
            CREATE INDEX IF NOT EXISTS idx_enterprise_chat_messages_session
                ON enterprise_chat_messages (session_id, seq);
            CREATE INDEX IF NOT EXISTS idx_enterprise_chat_messages_project
                ON enterprise_chat_messages (project_id);
            CREATE INDEX IF NOT EXISTS idx_enterprise_chat_messages_user
                ON enterprise_chat_messages (user_id);

            CREATE TABLE IF NOT EXISTS enterprise_chat_memory_episodes (
                episode_uuid TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                first_seq BIGINT NOT NULL,
                last_seq BIGINT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_enterprise_chat_memory_episodes_session
                ON enterprise_chat_memory_episodes (session_id, last_seq);
            CREATE INDEX IF NOT EXISTS idx_enterprise_chat_memory_episodes_project
                ON enterprise_chat_memory_episodes (project_id);
            CREATE INDEX IF NOT EXISTS idx_enterprise_chat_memory_episodes_user
                ON enterprise_chat_memory_episodes (user_id);

            CREATE TABLE IF NOT EXISTS enterprise_api_keys (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                key_hash TEXT NOT NULL UNIQUE,
                key_preview TEXT NOT NULL,
                status TEXT NOT NULL,
                created_by TEXT,
                tenant_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_used_at TEXT,
                revoked_at TEXT,
                revoked_by TEXT,
                data_json JSONB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_enterprise_api_keys_status
                ON enterprise_api_keys (status);
            CREATE INDEX IF NOT EXISTS idx_enterprise_api_keys_created_by
                ON enterprise_api_keys (created_by, status);
            CREATE INDEX IF NOT EXISTS idx_enterprise_api_keys_tenant
                ON enterprise_api_keys (tenant_id, status);

            CREATE TABLE IF NOT EXISTS enterprise_kb_acl (
                kb_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                granted_by TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                data_json JSONB NOT NULL,
                PRIMARY KEY (kb_id, user_id),
                FOREIGN KEY (user_id) REFERENCES enterprise_users(id)
            );
            CREATE INDEX IF NOT EXISTS idx_enterprise_kb_acl_user
                ON enterprise_kb_acl (user_id, kb_id);

            CREATE TABLE IF NOT EXISTS enterprise_tenants (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                data_json JSONB NOT NULL
            );

            CREATE TABLE IF NOT EXISTS enterprise_tenant_memberships (
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                granted_by TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                data_json JSONB NOT NULL,
                PRIMARY KEY (tenant_id, user_id),
                FOREIGN KEY (user_id) REFERENCES enterprise_users(id)
            );
            CREATE INDEX IF NOT EXISTS idx_enterprise_tenant_memberships_user
                ON enterprise_tenant_memberships (user_id, tenant_id);

            CREATE TABLE IF NOT EXISTS enterprise_tenant_kb_acl (
                tenant_id TEXT NOT NULL,
                kb_id TEXT NOT NULL,
                role TEXT NOT NULL,
                granted_by TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                data_json JSONB NOT NULL,
                PRIMARY KEY (tenant_id, kb_id)
            );
            CREATE INDEX IF NOT EXISTS idx_enterprise_tenant_kb_acl_kb
                ON enterprise_tenant_kb_acl (kb_id, tenant_id);

            CREATE TABLE IF NOT EXISTS enterprise_tenant_user_kb_overrides (
                tenant_id TEXT NOT NULL,
                kb_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                effect TEXT NOT NULL CHECK (effect IN ('allow', 'deny')),
                role TEXT,
                granted_by TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                data_json JSONB NOT NULL,
                PRIMARY KEY (tenant_id, kb_id, user_id),
                FOREIGN KEY (tenant_id, user_id)
                    REFERENCES enterprise_tenant_memberships(tenant_id, user_id)
                    ON DELETE CASCADE,
                CHECK (tenant_id <> '' AND tenant_id = btrim(tenant_id)),
                CHECK (kb_id <> '' AND kb_id = btrim(kb_id)),
                CHECK (user_id <> '' AND user_id = btrim(user_id)),
                CHECK (created_at <> '' AND updated_at <> ''),
                CHECK (granted_by IS NULL OR (
                    granted_by <> '' AND granted_by = btrim(granted_by)
                )),
                CHECK (
                    (effect = 'deny' AND role IS NULL)
                    OR
                    (effect = 'allow' AND role IN (
                        'kb_viewer', 'kb_editor', 'kb_admin', 'kb_owner'
                    ))
                )
            );
            CREATE INDEX IF NOT EXISTS idx_enterprise_tenant_user_kb_overrides_user
                ON enterprise_tenant_user_kb_overrides (user_id, tenant_id, kb_id);
            CREATE INDEX IF NOT EXISTS idx_enterprise_tenant_user_kb_overrides_kb
                ON enterprise_tenant_user_kb_overrides (tenant_id, kb_id, user_id);

            CREATE TABLE IF NOT EXISTS enterprise_audit_events (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                actor_user_id TEXT,
                actor_tenant_id TEXT,
                target_type TEXT,
                target_id TEXT,
                created_at TEXT NOT NULL,
                data_json JSONB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_enterprise_audit_events_created
                ON enterprise_audit_events (created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_enterprise_audit_events_actor
                ON enterprise_audit_events (actor_user_id);

            CREATE TABLE IF NOT EXISTS enterprise_invitations (
                id TEXT PRIMARY KEY,
                token_hash TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                created_by TEXT,
                expires_at TEXT,
                used_by TEXT,
                used_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                data_json JSONB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_enterprise_invitations_status
                ON enterprise_invitations (status);
            """
        )
        await conn.execute(
            """
            ALTER TABLE enterprise_kb_lifecycle
            ADD COLUMN IF NOT EXISTS delete_job_id TEXT
            """
        )
        # Existing installations have unnamed CHECK constraints that only
        # permit active/deleted. Drop every state-bearing legacy CHECK by
        # definition, then install stable named constraints idempotently.
        await conn.execute(
            """
            DO $$
            DECLARE lifecycle_constraint TEXT;
            BEGIN
                FOR lifecycle_constraint IN
                    SELECT c.conname
                    FROM pg_constraint c
                    WHERE c.conrelid = 'enterprise_kb_lifecycle'::regclass
                      AND c.contype = 'c'
                      AND pg_get_constraintdef(c.oid) ILIKE '%state%'
                LOOP
                    EXECUTE format(
                        'ALTER TABLE enterprise_kb_lifecycle DROP CONSTRAINT %I',
                        lifecycle_constraint
                    );
                END LOOP;
            END $$
            """
        )
        await conn.execute(
            """
            ALTER TABLE enterprise_kb_lifecycle
                DROP CONSTRAINT IF EXISTS enterprise_kb_lifecycle_delete_job_v2_check;
            ALTER TABLE enterprise_kb_lifecycle
                ADD CONSTRAINT enterprise_kb_lifecycle_state_v2_check
                CHECK (state IN ('active', 'deleting', 'deleted'));
            ALTER TABLE enterprise_kb_lifecycle
                ADD CONSTRAINT enterprise_kb_lifecycle_delete_job_v2_check
                CHECK (delete_job_id IS NULL OR (
                    delete_job_id <> '' AND delete_job_id = btrim(delete_job_id)
                ));
            ALTER TABLE enterprise_kb_lifecycle
                ADD CONSTRAINT enterprise_kb_lifecycle_state_payload_v2_check
                CHECK (
                    (state = 'active' AND deleted_at IS NULL AND delete_job_id IS NULL)
                    OR (
                        state = 'deleting' AND deleted_at IS NULL
                        AND delete_job_id IS NOT NULL
                    )
                    OR (state = 'deleted' AND deleted_at IS NOT NULL)
                )
            """
        )
        await conn.execute(
            """
            ALTER TABLE enterprise_audit_events
            ADD COLUMN IF NOT EXISTS actor_tenant_id TEXT
            """
        )
        await conn.execute(
            """
            UPDATE enterprise_audit_events
            SET actor_tenant_id = data_json->>'actor_tenant_id'
            WHERE actor_tenant_id IS NULL
              AND data_json ? 'actor_tenant_id'
              AND data_json->>'actor_tenant_id' IS NOT NULL
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_enterprise_audit_events_actor_tenant
            ON enterprise_audit_events (actor_tenant_id, created_at DESC, id)
            """
        )

        # enterprise_users.tenant_id is canonical. Repair legacy duplicate or
        # mismatched rows before adding the cross-tenant uniqueness invariant.
        await conn.execute(
            """
            UPDATE enterprise_users
            SET data_json = jsonb_set(
                data_json,
                '{tenant_id}',
                COALESCE(to_jsonb(tenant_id), 'null'::jsonb),
                true
            )
            WHERE data_json->>'tenant_id' IS DISTINCT FROM tenant_id
            """
        )
        await conn.execute(
            """
            DELETE FROM enterprise_tenant_memberships m
            WHERE NOT EXISTS (
                SELECT 1 FROM enterprise_users u
                WHERE u.id = m.user_id
                  AND u.tenant_id IS NOT NULL
                  AND u.tenant_id = m.tenant_id
            )
            """
        )
        repair_timestamp = utc_now_iso()
        await conn.execute(
            """
            INSERT INTO enterprise_tenant_memberships (
                tenant_id, user_id, role, granted_by, created_at, updated_at,
                data_json
            )
            SELECT u.tenant_id, u.id, 'tenant_member', NULL, $1, $1,
                   jsonb_build_object(
                       'tenant_id', u.tenant_id,
                       'user_id', u.id,
                       'role', 'tenant_member',
                       'granted_by', NULL,
                       'created_at', $1::text,
                       'updated_at', $1::text
                   )
            FROM enterprise_users u
            WHERE u.tenant_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM enterprise_tenant_memberships m
                  WHERE m.user_id = u.id AND m.tenant_id = u.tenant_id
              )
            """,
            repair_timestamp,
        )
        await conn.execute(
            """
            UPDATE enterprise_tenant_memberships
            SET data_json = jsonb_build_object(
                'tenant_id', tenant_id,
                'user_id', user_id,
                'role', role,
                'granted_by', granted_by,
                'created_at', created_at,
                'updated_at', updated_at
            )
            WHERE data_json IS DISTINCT FROM jsonb_build_object(
                'tenant_id', tenant_id,
                'user_id', user_id,
                'role', role,
                'granted_by', granted_by,
                'created_at', created_at,
                'updated_at', updated_at
            )
            """
        )
        await conn.execute(
            """
            DELETE FROM enterprise_tenant_user_kb_overrides o
            WHERE NOT EXISTS (
                SELECT 1
                FROM enterprise_users u
                JOIN enterprise_tenant_memberships m
                  ON m.user_id = u.id AND m.tenant_id = u.tenant_id
                WHERE u.id = o.user_id AND u.tenant_id = o.tenant_id
            )
            """
        )
        await conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_enterprise_tenant_memberships_user
            ON enterprise_tenant_memberships (user_id)
            """
        )
        await conn.execute(
            """
            INSERT INTO kb_metadata_schema(version, applied_at)
            VALUES (1, $1)
            ON CONFLICT (version) DO NOTHING
            """,
            utc_now_iso(),
        )

    async def _get_document(
        self, conn: Any, kb_id: str, document_id: str, *, for_update: bool = False
    ) -> DocumentRecord:
        suffix = " FOR UPDATE" if for_update else ""
        row = await conn.fetchrow(
            f"""
            SELECT data_json FROM kb_documents
            WHERE kb_id = $1 AND id = $2 AND deleted_at IS NULL
            {suffix}
            """,
            kb_id,
            document_id,
        )
        if row is None:
            raise MetadataRecordNotFoundError(f"Document '{document_id}' not found")
        return _document_from_row(row)

    async def _insert_document(self, conn: Any, document: DocumentRecord) -> None:
        await self._check_source_key_available(conn, document)
        await conn.execute(
            """
            INSERT INTO kb_documents (
                id, kb_id, workspace, status, source_name, source_hash, source_key,
                batch_id, deleted_at, created_at, updated_at, data_json
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::jsonb)
            """,
            document.id,
            document.kb_id,
            document.workspace,
            document.status,
            document.source_name,
            document.source_hash,
            _metadata_source_key(document.metadata),
            _batch_id(document.metadata),
            document.deleted_at,
            document.created_at,
            document.updated_at,
            _record_json(document),
        )

    async def _save_document(self, conn: Any, document: DocumentRecord) -> None:
        await self._check_source_key_available(conn, document)
        status = await conn.execute(
            """
            UPDATE kb_documents
            SET status = $1, source_name = $2, source_hash = $3, source_key = $4,
                batch_id = $5, deleted_at = $6, updated_at = $7, data_json = $8::jsonb
            WHERE kb_id = $9 AND id = $10
            """,
            document.status,
            document.source_name,
            document.source_hash,
            _metadata_source_key(document.metadata) if document.deleted_at is None else None,
            _batch_id(document.metadata),
            document.deleted_at,
            document.updated_at,
            _record_json(document),
            document.kb_id,
            document.id,
        )
        if _rowcount(status) == 0:
            raise MetadataRecordNotFoundError(f"Document '{document.id}' not found")

    async def _check_source_key_available(self, conn: Any, document: DocumentRecord) -> None:
        source_key = _metadata_source_key(document.metadata)
        if source_key is None or document.deleted_at is not None:
            return
        existing_id = await conn.fetchval(
            """
            SELECT id FROM kb_documents
            WHERE kb_id = $1 AND source_key = $2 AND deleted_at IS NULL AND id <> $3
            LIMIT 1
            """,
            document.kb_id,
            source_key,
            document.id,
        )
        if existing_id is not None:
            raise DuplicateDocumentSourceKeyError(document.kb_id, source_key, str(existing_id))

    async def _insert_artifact(self, conn: Any, artifact: ArtifactRecord) -> None:
        await conn.execute(
            """
            INSERT INTO kb_document_artifacts (
                id, kb_id, workspace, document_id, artifact_type, created_at, data_json
            ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
            """,
            artifact.id,
            artifact.kb_id,
            artifact.workspace,
            artifact.document_id,
            artifact.artifact_type,
            artifact.created_at,
            _record_json(artifact),
        )

    async def _get_job(
        self, conn: Any, kb_id: str, job_id: str, *, for_update: bool = False
    ) -> JobRecord:
        suffix = " FOR UPDATE" if for_update else ""
        row = await conn.fetchrow(
            f"SELECT data_json FROM kb_jobs WHERE kb_id = $1 AND id = $2{suffix}",
            kb_id,
            job_id,
        )
        if row is None:
            raise MetadataRecordNotFoundError(f"Job '{job_id}' not found")
        return _job_from_row(row)

    async def _insert_job(self, conn: Any, job: JobRecord) -> None:
        await conn.execute(
            """
            INSERT INTO kb_jobs (
                id, kb_id, workspace, batch_id, document_id, job_type, status,
                idempotency_key, retry_count, max_retries, queued_at, created_at,
                updated_at, data_json
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::jsonb)
            """,
            job.id,
            job.kb_id,
            job.workspace,
            job.batch_id,
            job.document_id,
            job.job_type,
            job.status,
            job.idempotency_key,
            job.retry_count,
            job.max_retries,
            job.queued_at,
            job.created_at,
            job.updated_at,
            _record_json(job),
        )

    async def _save_job(self, conn: Any, job: JobRecord) -> None:
        status = await conn.execute(
            """
            UPDATE kb_jobs
            SET status = $1, idempotency_key = $2, retry_count = $3,
                max_retries = $4, queued_at = $5, updated_at = $6, data_json = $7::jsonb
            WHERE kb_id = $8 AND id = $9
            """,
            job.status,
            job.idempotency_key,
            job.retry_count,
            job.max_retries,
            job.queued_at,
            job.updated_at,
            _record_json(job),
            job.kb_id,
            job.id,
        )
        if _rowcount(status) == 0:
            raise MetadataRecordNotFoundError(f"Job '{job.id}' not found")

    async def _get_job_by_idempotency_key(
        self,
        conn: Any,
        kb_id: str,
        idempotency_key: str | None,
        *,
        job_type: str | None = None,
    ) -> JobRecord | None:
        if not idempotency_key:
            return None
        params: list[Any] = [kb_id, idempotency_key]
        clause = "kb_id = $1 AND idempotency_key = $2"
        if job_type is not None:
            params.append(job_type)
            clause += f" AND job_type = ${len(params)}"
        row = await conn.fetchrow(
            f"""
            SELECT data_json FROM kb_jobs
            WHERE {clause}
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            *params,
        )
        return _job_from_row(row) if row is not None else None

    def _validate_idempotent_job(self, existing: JobRecord, candidate: JobRecord) -> None:
        if existing.payload.get("idempotency_fingerprint") != candidate.payload.get(
            "idempotency_fingerprint"
        ):
            raise IdempotencyKeyConflictError(candidate.idempotency_key or "")

    async def _documents_for_job(self, conn: Any, job: JobRecord) -> list[DocumentRecord]:
        document_ids = job.payload.get("document_ids")
        if isinstance(document_ids, list) and all(
            isinstance(document_id, str) for document_id in document_ids
        ):
            if not document_ids:
                return []
            rows = await conn.fetch(
                """
                SELECT data_json FROM kb_documents
                WHERE kb_id = $1 AND id = ANY($2::text[]) AND deleted_at IS NULL
                """,
                job.kb_id,
                document_ids,
            )
            by_id = {_loads_json_object(row["data_json"])["id"]: _document_from_row(row) for row in rows}
            return [by_id[document_id] for document_id in document_ids if document_id in by_id]
        if not job.batch_id:
            return []
        rows = await conn.fetch(
            """
            SELECT data_json FROM kb_documents
            WHERE kb_id = $1 AND batch_id = $2 AND deleted_at IS NULL
            ORDER BY created_at ASC, id ASC
            """,
            job.kb_id,
            job.batch_id,
        )
        return [_document_from_row(row) for row in rows]

    async def _claim_document_parse_queued(
        self,
        conn: Any,
        kb_id: str,
        document_id: str,
        *,
        metadata_patch: dict[str, Any],
        raise_on_active: bool,
    ) -> DocumentRecord:
        document = await self._get_document(conn, kb_id, document_id, for_update=True)
        if raise_on_active and document.status in {"parse_queued", "parsing"}:
            raise ActiveDocumentParseJobError(document_id, _active_parse_job_id(document))
        if document.status in {"build_queued", "building"}:
            raise ActiveDocumentBuildJobError(document_id, _active_build_job_id(document))
        if document.status == "deleting":
            raise ActiveDocumentDeleteJobError(document_id, _active_delete_job_id(document))
        if document.status == "replacing":
            raise ActiveDocumentReplaceJobError(document_id, _active_replace_job_id(document))
        return await self._update_document_parse_state(
            conn,
            kb_id,
            document_id,
            status="parse_queued",
            metadata_patch=metadata_patch,
            clear_error=True,
        )

    async def _claim_document_build_queued(
        self,
        conn: Any,
        kb_id: str,
        document_id: str,
        *,
        metadata_patch: dict[str, Any],
        require_parsed: bool,
    ) -> DocumentRecord:
        document = await self._get_document(conn, kb_id, document_id, for_update=True)
        if document.status in {"build_queued", "building"}:
            raise ActiveDocumentBuildJobError(document_id, _active_build_job_id(document))
        if document.status == "deleting":
            raise ActiveDocumentDeleteJobError(document_id, _active_delete_job_id(document))
        if document.status == "replacing":
            raise ActiveDocumentReplaceJobError(document_id, _active_replace_job_id(document))
        if require_parsed and document.status not in {"parsed", "ready", "build_failed"}:
            raise DocumentNotParsedError(document_id, str(document.status))
        return await self._update_document_parse_state(
            conn,
            kb_id,
            document_id,
            status="build_queued",
            metadata_patch=metadata_patch,
            clear_error=True,
        )

    async def _claim_document_deleting(
        self, conn: Any, kb_id: str, document_id: str, *, metadata_patch: dict[str, Any]
    ) -> DocumentRecord:
        document = await self._get_document(conn, kb_id, document_id, for_update=True)
        if document.status in {"parse_queued", "parsing"}:
            raise ActiveDocumentParseJobError(document_id, _active_parse_job_id(document))
        if document.status in {"build_queued", "building"}:
            raise ActiveDocumentBuildJobError(document_id, _active_build_job_id(document))
        if document.status == "deleting":
            existing_job_id = _active_delete_job_id(document)
            requested_job_id = metadata_patch.get("pending_delete_job_id") or metadata_patch.get(
                "current_delete_job_id"
            )
            if requested_job_id is not None and str(requested_job_id) == existing_job_id:
                return await self._update_document_parse_state(
                    conn,
                    kb_id,
                    document_id,
                    status="deleting",
                    metadata_patch=metadata_patch,
                    clear_error=True,
                )
            raise ActiveDocumentDeleteJobError(document_id, existing_job_id)
        if document.status == "replacing":
            raise ActiveDocumentReplaceJobError(document_id, _active_replace_job_id(document))
        return await self._update_document_parse_state(
            conn,
            kb_id,
            document_id,
            status="deleting",
            metadata_patch=metadata_patch,
            clear_error=True,
        )

    async def _claim_document_replacing(
        self, conn: Any, kb_id: str, document_id: str, *, metadata_patch: dict[str, Any]
    ) -> DocumentRecord:
        document = await self._get_document(conn, kb_id, document_id, for_update=True)
        if document.status in {"parse_queued", "parsing"}:
            raise ActiveDocumentParseJobError(document_id, _active_parse_job_id(document))
        if document.status in {"build_queued", "building"}:
            raise ActiveDocumentBuildJobError(document_id, _active_build_job_id(document))
        if document.status == "deleting":
            raise ActiveDocumentDeleteJobError(document_id, _active_delete_job_id(document))
        if document.status == "replacing":
            raise ActiveDocumentReplaceJobError(document_id, _active_replace_job_id(document))
        return await self._update_document_parse_state(
            conn,
            kb_id,
            document_id,
            status="replacing",
            metadata_patch=metadata_patch,
            clear_error=True,
        )

    async def _update_document_parse_state(
        self,
        conn: Any,
        kb_id: str,
        document_id: str,
        *,
        status: str,
        metadata_patch: dict[str, Any],
        parser_hash: str | None = None,
        lightrag_doc_id: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        clear_error: bool = False,
        clear_lightrag_doc_id: bool = False,
        clear_index_state: bool = False,
    ) -> DocumentRecord:
        document = await self._get_document(conn, kb_id, document_id, for_update=True)
        document.metadata.update(metadata_patch)
        document.status = status
        if parser_hash is not None:
            document.parser_hash = parser_hash
        if clear_lightrag_doc_id:
            document.lightrag_doc_id = None
        elif lightrag_doc_id is not None:
            document.lightrag_doc_id = lightrag_doc_id
        if clear_index_state:
            document.index_hash = None
            document.chunks_count = None
            document.entity_count = None
            document.relation_count = None
        if clear_error:
            document.error_code = None
            document.error_message = None
        else:
            document.error_code = error_code
            document.error_message = error_message
        document.updated_at = utc_now_iso()
        await self._save_document(conn, document)
        return document

    async def _insert_config_version(self, conn: Any, record: ConfigVersionRecord) -> None:
        await conn.execute(
            """
            INSERT INTO kb_config_versions (
                id, kb_id, workspace, version, created_at, data_json
            ) VALUES ($1, $2, $3, $4, $5, $6::jsonb)
            """,
            record.id,
            record.kb_id,
            record.workspace,
            record.version,
            record.created_at,
            _record_json(record),
        )

    async def _save_config_version(self, conn: Any, record: ConfigVersionRecord) -> None:
        await conn.execute(
            """
            UPDATE kb_config_versions
            SET data_json = $1::jsonb
            WHERE kb_id = $2 AND id = $3
            """,
            _record_json(record),
            record.kb_id,
            record.id,
        )


def _batch_id(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("batch_id")
    return value if isinstance(value, str) and value else None


def _rowcount(status: str) -> int:
    try:
        return int(status.rsplit(" ", 1)[-1])
    except (ValueError, IndexError):
        return 0


def _active_failure(document_id: str, error_code: str, exc: Exception) -> dict[str, Any]:
    detail = {
        "document_id": document_id,
        "status": "failed",
        "error_code": error_code,
        "error_message": str(exc),
    }
    existing_job_id = getattr(exc, "existing_job_id", None)
    if existing_job_id is not None:
        detail["existing_job_id"] = existing_job_id
    return detail
