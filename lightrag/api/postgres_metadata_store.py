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
    _chat_memory_append_batch_id,
    _chat_memory_canonical_episode_payload,
    _chat_memory_existing_graph_store_fingerprint,
    _chat_memory_event_identity,
    _chat_memory_graph_store_migration_conflict,
    _chat_memory_noop_episode_uuid,
    _chat_memory_replay_batches_from_messages,
    _chat_memory_replay_snapshot_metrics,
    _chat_memory_snapshot_digest,
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
    _new_chat_memory_claim_token,
    _normalize_chat_memory_group_ids,
    _normalize_chat_memory_event_types,
    _orphan_recovery_cutoff,
    _resolve_chat_memory_graph_store_fingerprint,
    _same_job_execution_identity,
    _should_requeue_orphaned_clear_job,
    _TENANT_MEMBERSHIP_ROLES,
    _validate_job_execution_id,
    _validate_delete_job_id,
    _validate_kb_lifecycle_identity,
    _validate_chat_memory_fingerprint,
    _validate_chat_memory_ingest_max_chars,
    _validate_chat_memory_ingest_source_batch,
    _validate_chat_memory_worker_id,
    _validate_tenant_user_kb_override,
    _wait_for_kb_guard_borrowers,
    ActiveDocumentBuildJobError,
    ActiveDocumentDeleteJobError,
    ActiveDocumentParseJobError,
    ActiveDocumentReplaceJobError,
    ArtifactRecord,
    AuditEventRecord,
    CHAT_MEMORY_DEFAULT_INGEST_MAX_CHARS,
    CHAT_MEMORY_RECORD_VERSION,
    ChatMemoryBacklogItem,
    ChatMemoryEpisodeRecord,
    ChatMemoryEventStatus,
    ChatMemoryEventType,
    ChatMemoryExecutionState,
    ChatMemoryGenerationRecord,
    ChatMemoryGenerationState,
    ChatMemoryGroupRecord,
    ChatMemoryOutboxEventRecord,
    ChatMemoryOutboxStats,
    ChatMemoryPurgeTargetSet,
    ChatMemoryRebuildSnapshot,
    ChatMemoryRebuildTargetSet,
    ChatMemoryReadToken,
    ChatMemoryReplayMappingInput,
    ChatMemoryReplayBatch,
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
    chat_memory_graph_group_id,
    chat_memory_legacy_graph_group_id,
    chat_memory_logical_group_id,
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


def _iso_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    return str(value)


def _projection_keys(row: Any) -> set[str]:
    try:
        return set(row.keys())
    except (AttributeError, TypeError):
        return set(row) if isinstance(row, dict) else set()


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
    data.setdefault("append_batch_id", None)
    data.setdefault("project_event_seq", None)
    data.setdefault("memory_reference_time", None)
    projection_keys = _projection_keys(row)
    if "append_batch_id" in projection_keys:
        data["append_batch_id"] = row["append_batch_id"]
    if "project_event_seq" in projection_keys:
        data["project_event_seq"] = (
            int(row["project_event_seq"])
            if row["project_event_seq"] is not None
            else None
        )
    if "memory_reference_time" in projection_keys:
        data["memory_reference_time"] = _iso_timestamp(
            row["memory_reference_time"]
        )
    return ChatMessageRecord(**data)


def _chat_memory_episode_from_row(row: Any) -> ChatMemoryEpisodeRecord:
    projection_keys = _projection_keys(row)
    return ChatMemoryEpisodeRecord(
        episode_uuid=str(row["episode_uuid"]),
        session_id=str(row["session_id"]),
        project_id=str(row["project_id"]),
        user_id=str(row["user_id"]),
        first_seq=int(row["first_seq"]),
        last_seq=int(row["last_seq"]),
        created_at=str(row["created_at"]),
        event_id=(row["event_id"] if "event_id" in projection_keys else None),
        generation=(
            int(row["generation"])
            if "generation" in projection_keys and row["generation"] is not None
            else None
        ),
        graph_group_id=(
            row["graph_group_id"]
            if "graph_group_id" in projection_keys
            else None
        ),
        append_batch_id=(
            row["append_batch_id"]
            if "append_batch_id" in projection_keys
            else None
        ),
        project_event_seq=(
            int(row["project_event_seq"])
            if "project_event_seq" in projection_keys
            and row["project_event_seq"] is not None
            else None
        ),
    )


def _chat_memory_group_from_row(row: Any) -> ChatMemoryGroupRecord:
    projection_keys = _projection_keys(row)
    return ChatMemoryGroupRecord(
        user_id=str(row["user_id"]),
        project_id=str(row["project_id"]),
        logical_group_id=str(row["logical_group_id"]),
        active_generation=(
            int(row["active_generation"])
            if row["active_generation"] is not None
            else None
        ),
        desired_generation=int(row["desired_generation"]),
        next_event_seq=int(row["next_event_seq"]),
        last_reference_time=_iso_timestamp(row["last_reference_time"]),
        state=str(row["state"]),  # type: ignore[arg-type]
        state_version=int(row["state_version"]),
        active_config_fingerprint=row["active_config_fingerprint"],
        desired_config_fingerprint=str(row["desired_config_fingerprint"]),
        active_rebuild_event_id=row["active_rebuild_event_id"],
        last_success_at=_iso_timestamp(row["last_success_at"]),
        last_error_code=row["last_error_code"],
        last_error_message=row["last_error_message"],
        last_error_at=_iso_timestamp(row["last_error_at"]),
        created_at=str(_iso_timestamp(row["created_at"])),
        updated_at=str(_iso_timestamp(row["updated_at"])),
        deleted_at=_iso_timestamp(row["deleted_at"]),
        record_version=int(row["record_version"]),
        active_graph_store_fingerprint=(
            row["active_graph_store_fingerprint"]
            if "active_graph_store_fingerprint" in projection_keys
            else row["active_config_fingerprint"]
        ),
        desired_graph_store_fingerprint=(
            str(row["desired_graph_store_fingerprint"])
            if "desired_graph_store_fingerprint" in projection_keys
            else str(row["desired_config_fingerprint"])
        ),
    )


def _chat_memory_generation_from_row(row: Any) -> ChatMemoryGenerationRecord:
    projection_keys = _projection_keys(row)
    return ChatMemoryGenerationRecord(
        user_id=str(row["user_id"]),
        project_id=str(row["project_id"]),
        generation=int(row["generation"]),
        graph_group_id=str(row["graph_group_id"]),
        config_fingerprint=str(row["config_fingerprint"]),
        state=str(row["state"]),  # type: ignore[arg-type]
        snapshot_cutoff=(
            int(row["snapshot_cutoff"])
            if row["snapshot_cutoff"] is not None
            else None
        ),
        replay_batch_count=(
            int(row["replay_batch_count"])
            if row["replay_batch_count"] is not None
            else None
        ),
        replay_message_count=(
            int(row["replay_message_count"])
            if row["replay_message_count"] is not None
            else None
        ),
        replay_byte_count=(
            int(row["replay_byte_count"])
            if row["replay_byte_count"] is not None
            else None
        ),
        clear_attempt_no=int(row["clear_attempt_no"]),
        clear_started_at=_iso_timestamp(row["clear_started_at"]),
        created_at=str(_iso_timestamp(row["created_at"])),
        updated_at=str(_iso_timestamp(row["updated_at"])),
        activated_at=_iso_timestamp(row["activated_at"]),
        cleared_at=_iso_timestamp(row["cleared_at"]),
        last_error_code=row["last_error_code"],
        last_error_message=row["last_error_message"],
        last_error_at=_iso_timestamp(row["last_error_at"]),
        snapshot_digest=(
            row["snapshot_digest"]
            if "snapshot_digest" in projection_keys
            else None
        ),
        record_version=int(row["record_version"]),
        graph_store_fingerprint=(
            str(row["graph_store_fingerprint"])
            if "graph_store_fingerprint" in projection_keys
            else str(row["config_fingerprint"])
        ),
    )


def _chat_memory_event_from_row(row: Any) -> ChatMemoryOutboxEventRecord:
    projection_keys = _projection_keys(row)
    return ChatMemoryOutboxEventRecord(
        event_id=str(row["event_id"]),
        deterministic_key=str(row["deterministic_key"]),
        user_id=str(row["user_id"]),
        project_id=str(row["project_id"]),
        event_seq=int(row["event_seq"]),
        generation=int(row["generation"]),
        graph_group_id=str(row["graph_group_id"]),
        config_fingerprint=str(row["config_fingerprint"]),
        event_type=str(row["event_type"]),  # type: ignore[arg-type]
        status=str(row["status"]),  # type: ignore[arg-type]
        available_at=str(_iso_timestamp(row["available_at"])),
        attempt_no=int(row["attempt_no"]),
        created_at=str(_iso_timestamp(row["created_at"])),
        updated_at=str(_iso_timestamp(row["updated_at"])),
        source_session_id=row["source_session_id"],
        append_batch_id=row["append_batch_id"],
        first_seq=(int(row["first_seq"]) if row["first_seq"] is not None else None),
        last_seq=(int(row["last_seq"]) if row["last_seq"] is not None else None),
        snapshot_cutoff=(
            int(row["snapshot_cutoff"])
            if row["snapshot_cutoff"] is not None
            else None
        ),
        snapshot_batch_count=(
            int(row["snapshot_batch_count"])
            if row["snapshot_batch_count"] is not None
            else None
        ),
        snapshot_message_count=(
            int(row["snapshot_message_count"])
            if row["snapshot_message_count"] is not None
            else None
        ),
        snapshot_byte_count=(
            int(row["snapshot_byte_count"])
            if row["snapshot_byte_count"] is not None
            else None
        ),
        snapshot_digest=(
            row["snapshot_digest"]
            if "snapshot_digest" in projection_keys
            else None
        ),
        claim_token=row["claim_token"],
        claimed_by=row["claimed_by"],
        claimed_at=_iso_timestamp(row["claimed_at"]),
        side_effect_started_at=_iso_timestamp(row["side_effect_started_at"]),
        side_effect_state_version=(
            int(row["side_effect_state_version"])
            if row["side_effect_state_version"] is not None
            else None
        ),
        completed_at=_iso_timestamp(row["completed_at"]),
        superseded_by_event_id=row["superseded_by_event_id"],
        last_error_code=row["last_error_code"],
        last_error_message=row["last_error_message"],
        last_error_at=_iso_timestamp(row["last_error_at"]),
        actor_user_id=row["actor_user_id"],
        actor_tenant_id=row["actor_tenant_id"],
        target_user_id=row["target_user_id"],
        target_project_id=row["target_project_id"],
        target_session_id=row["target_session_id"],
        target_message_id=row["target_message_id"],
        record_version=int(row["record_version"]),
        graph_store_fingerprint=(
            str(row["graph_store_fingerprint"])
            if "graph_store_fingerprint" in projection_keys
            else str(row["config_fingerprint"])
        ),
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

    async def delete_chat_message_with_memory(
        self,
        user_id: str,
        project_id: str,
        session_id: str,
        message_id: str,
        *,
        config_fingerprint: str,
        graph_store_fingerprint: str | None = None,
        actor_user_id: str | None = None,
        actor_tenant_id: str | None = None,
    ) -> bool:
        """Delete one source message and enqueue a fenced rebuild if admitted."""

        fingerprint = _validate_chat_memory_fingerprint(config_fingerprint)
        graph_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            fingerprint, graph_store_fingerprint
        )
        await self._ensure_initialized()

        async def write(conn: Any) -> bool:
            if (
                await conn.fetchval(
                    "SELECT id FROM enterprise_users WHERE id = $1 FOR UPDATE",
                    user_id,
                )
                is None
            ):
                return False
            if (
                await conn.fetchval(
                    """
                    SELECT id FROM enterprise_chat_projects
                    WHERE id = $1 AND user_id = $2 FOR UPDATE
                    """,
                    project_id,
                    user_id,
                )
                is None
            ):
                return False
            if (
                await conn.fetchval(
                    """
                    SELECT id FROM enterprise_chat_sessions
                    WHERE id = $1 AND project_id = $2 AND user_id = $3
                    FOR UPDATE
                    """,
                    session_id,
                    project_id,
                    user_id,
                )
                is None
            ):
                return False
            message_row = await conn.fetchrow(
                """
                SELECT id, seq, project_event_seq
                FROM enterprise_chat_messages
                WHERE id = $1 AND session_id = $2 AND project_id = $3
                  AND user_id = $4
                FOR UPDATE
                """,
                message_id,
                session_id,
                project_id,
                user_id,
            )
            if message_row is None:
                return False
            memory_affected = message_row["project_event_seq"] is not None
            if not memory_affected:
                memory_affected = (
                    await conn.fetchval(
                        """
                        SELECT 1 FROM enterprise_chat_memory_episodes
                        WHERE session_id = $1 AND project_id = $2 AND user_id = $3
                          AND first_seq <= $4 AND last_seq >= $4
                        LIMIT 1
                        """,
                        session_id,
                        project_id,
                        user_id,
                        int(message_row["seq"]),
                    )
                    is not None
                )
            await conn.execute(
                "DELETE FROM enterprise_chat_messages WHERE id = $1", message_id
            )
            if memory_affected:
                group, _ = await self._ensure_postgres_chat_memory_group(
                    conn,
                    user_id,
                    project_id,
                    fingerprint,
                    graph_fingerprint,
                    generation_state="building",
                )
                bound_graph_fingerprint = (
                    _chat_memory_existing_graph_store_fingerprint(
                        group, graph_fingerprint
                    )
                )
                await self._enqueue_postgres_chat_memory_rebuild(
                    conn,
                    group,
                    fingerprint,
                    bound_graph_fingerprint,
                    actor_user_id=actor_user_id or user_id,
                    actor_tenant_id=actor_tenant_id,
                    target_session_id=session_id,
                    target_message_id=message_id,
                )
            return True

        return await self._write(write)

    async def delete_chat_session_with_memory(
        self,
        user_id: str,
        project_id: str,
        session_id: str,
        *,
        config_fingerprint: str,
        graph_store_fingerprint: str | None = None,
        actor_user_id: str | None = None,
        actor_tenant_id: str | None = None,
    ) -> tuple[bool, int]:
        """Delete an owned session and atomically enqueue any required rebuild."""

        fingerprint = _validate_chat_memory_fingerprint(config_fingerprint)
        graph_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            fingerprint, graph_store_fingerprint
        )
        await self._ensure_initialized()

        async def write(conn: Any) -> tuple[bool, int]:
            if (
                await conn.fetchval(
                    "SELECT id FROM enterprise_users WHERE id = $1 FOR UPDATE",
                    user_id,
                )
                is None
            ):
                return False, 0
            if (
                await conn.fetchval(
                    """
                    SELECT id FROM enterprise_chat_projects
                    WHERE id = $1 AND user_id = $2 FOR UPDATE
                    """,
                    project_id,
                    user_id,
                )
                is None
            ):
                return False, 0
            if (
                await conn.fetchval(
                    """
                    SELECT id FROM enterprise_chat_sessions
                    WHERE id = $1 AND project_id = $2 AND user_id = $3
                    FOR UPDATE
                    """,
                    session_id,
                    project_id,
                    user_id,
                )
                is None
            ):
                return False, 0
            message_rows = await conn.fetch(
                """
                SELECT id, project_event_seq FROM enterprise_chat_messages
                WHERE session_id = $1 ORDER BY seq, id FOR UPDATE
                """,
                session_id,
            )
            mapped = await conn.fetchval(
                """
                SELECT 1 FROM enterprise_chat_memory_episodes
                WHERE session_id = $1 AND project_id = $2 AND user_id = $3 LIMIT 1
                """,
                session_id,
                project_id,
                user_id,
            )
            memory_affected = mapped is not None or any(
                row["project_event_seq"] is not None for row in message_rows
            )
            messages_status = await conn.execute(
                "DELETE FROM enterprise_chat_messages WHERE session_id = $1",
                session_id,
            )
            await conn.execute(
                "DELETE FROM enterprise_chat_sessions WHERE id = $1", session_id
            )
            if memory_affected:
                group, _ = await self._ensure_postgres_chat_memory_group(
                    conn,
                    user_id,
                    project_id,
                    fingerprint,
                    graph_fingerprint,
                    generation_state="building",
                )
                bound_graph_fingerprint = (
                    _chat_memory_existing_graph_store_fingerprint(
                        group, graph_fingerprint
                    )
                )
                await self._enqueue_postgres_chat_memory_rebuild(
                    conn,
                    group,
                    fingerprint,
                    bound_graph_fingerprint,
                    actor_user_id=actor_user_id or user_id,
                    actor_tenant_id=actor_tenant_id,
                    target_session_id=session_id,
                    target_message_id=None,
                )
            return True, _rowcount(messages_status)

        return await self._write(write)

    async def delete_chat_project_with_memory(
        self,
        user_id: str,
        project_id: str,
        *,
        config_fingerprint: str,
        graph_store_fingerprint: str | None = None,
        actor_user_id: str | None = None,
        actor_tenant_id: str | None = None,
    ) -> tuple[bool, int, int]:
        """Persist a purge tombstone before deleting an owned source project."""

        fingerprint = _validate_chat_memory_fingerprint(config_fingerprint)
        graph_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            fingerprint, graph_store_fingerprint
        )
        await self._ensure_initialized()

        async def write(conn: Any) -> tuple[bool, int, int]:
            if (
                await conn.fetchval(
                    "SELECT id FROM enterprise_users WHERE id = $1 FOR UPDATE",
                    user_id,
                )
                is None
            ):
                return False, 0, 0
            if (
                await conn.fetchval(
                    """
                    SELECT id FROM enterprise_chat_projects
                    WHERE id = $1 AND user_id = $2 FOR UPDATE
                    """,
                    project_id,
                    user_id,
                )
                is None
            ):
                return False, 0, 0
            await conn.fetch(
                """
                SELECT id FROM enterprise_chat_sessions
                WHERE project_id = $1 AND user_id = $2
                ORDER BY id FOR UPDATE
                """,
                project_id,
                user_id,
            )
            await conn.fetch(
                """
                SELECT id FROM enterprise_chat_messages
                WHERE project_id = $1 AND user_id = $2
                ORDER BY session_id, seq, id FOR UPDATE
                """,
                project_id,
                user_id,
            )
            await self._enqueue_postgres_chat_memory_purge(
                conn,
                user_id,
                project_id,
                fingerprint,
                graph_fingerprint,
                actor_user_id=actor_user_id or user_id,
                actor_tenant_id=actor_tenant_id,
            )
            messages_status = await conn.execute(
                "DELETE FROM enterprise_chat_messages WHERE project_id = $1",
                project_id,
            )
            sessions_status = await conn.execute(
                "DELETE FROM enterprise_chat_sessions WHERE project_id = $1",
                project_id,
            )
            await conn.execute(
                "DELETE FROM enterprise_chat_projects WHERE id = $1", project_id
            )
            return True, _rowcount(sessions_status), _rowcount(messages_status)

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
                record.append_batch_id = None
                record.project_event_seq = None
                record.memory_reference_time = None
                await conn.execute(
                    """
                    INSERT INTO enterprise_chat_messages (
                        id, session_id, project_id, user_id, seq, created_at,
                        append_batch_id, project_event_seq, memory_reference_time,
                        data_json
                    ) VALUES ($1, $2, $3, $4, $5, $6, NULL, NULL, NULL,
                              $7::jsonb)
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
                SELECT append_batch_id, project_event_seq, memory_reference_time,
                       data_json
                FROM enterprise_chat_messages
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
                SELECT append_batch_id, project_event_seq, memory_reference_time,
                       data_json
                FROM enterprise_chat_messages
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
                SELECT append_batch_id, project_event_seq, memory_reference_time,
                       data_json
                FROM enterprise_chat_messages
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
                SELECT append_batch_id, project_event_seq, memory_reference_time,
                       data_json
                FROM enterprise_chat_messages
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
                    first_seq, last_seq, created_at, event_id, generation,
                    graph_group_id, append_batch_id, project_event_seq
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                          $11, $12)
                ON CONFLICT (episode_uuid) DO UPDATE SET
                    session_id = excluded.session_id,
                    project_id = excluded.project_id,
                    user_id = excluded.user_id,
                    first_seq = excluded.first_seq,
                    last_seq = excluded.last_seq,
                    created_at = excluded.created_at,
                    event_id = excluded.event_id,
                    generation = excluded.generation,
                    graph_group_id = excluded.graph_group_id,
                    append_batch_id = excluded.append_batch_id,
                    project_event_seq = excluded.project_event_seq
                """,
                record.episode_uuid,
                record.session_id,
                record.project_id,
                record.user_id,
                record.first_seq,
                record.last_seq,
                record.created_at,
                record.event_id,
                record.generation,
                record.graph_group_id,
                record.append_batch_id,
                record.project_event_seq,
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

    async def append_chat_messages_with_memory(
        self,
        records: Sequence[ChatMessageRecord],
        *,
        config_fingerprint: str,
        graph_store_fingerprint: str | None = None,
        actor_user_id: str | None = None,
        actor_tenant_id: str | None = None,
    ) -> list[ChatMessageRecord]:
        """Atomically admit one append batch and enqueue its ingest event."""

        if not records:
            return []
        fingerprint = _validate_chat_memory_fingerprint(config_fingerprint)
        graph_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            fingerprint, graph_store_fingerprint
        )
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
            # Stable row-lock order: user -> project -> session -> group.
            if (
                await conn.fetchval(
                    "SELECT id FROM enterprise_users WHERE id = $1 FOR UPDATE",
                    head.user_id,
                )
                is None
            ):
                raise MetadataRecordNotFoundError(
                    f"Chat session '{head.session_id}' not found"
                )
            if (
                await conn.fetchval(
                    """
                    SELECT id FROM enterprise_chat_projects
                    WHERE id = $1 AND user_id = $2 FOR UPDATE
                    """,
                    head.project_id,
                    head.user_id,
                )
                is None
            ):
                raise MetadataRecordNotFoundError(
                    f"Chat session '{head.session_id}' not found"
                )
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
            group, _created = await self._ensure_postgres_chat_memory_group(
                conn,
                head.user_id,
                head.project_id,
                fingerprint,
                graph_fingerprint,
                generation_state="building",
            )
            await self._assert_postgres_chat_memory_graph_store_invariant(
                conn, group, graph_fingerprint
            )
            if group.state in {"deleting", "deleted"}:
                raise MetadataConflictError(
                    "chat_memory_group",
                    f"{head.user_id}:{head.project_id}",
                    expected={"state": "writable"},
                    current={"state": group.state},
                )
            if group.desired_config_fingerprint != fingerprint:
                await self._enqueue_postgres_chat_memory_rebuild(
                    conn,
                    group,
                    fingerprint,
                    graph_fingerprint,
                    actor_user_id=actor_user_id or head.user_id,
                    actor_tenant_id=actor_tenant_id,
                    target_session_id=head.session_id,
                    target_message_id=None,
                )
                group = await self._get_postgres_chat_memory_group(
                    conn, head.user_id, head.project_id, for_update=True
                )
                assert group is not None

            event_seq, reference_value = (
                await self._allocate_postgres_chat_memory_event_seq(
                    conn,
                    head.user_id,
                    head.project_id,
                    allocate_reference_time=True,
                )
            )
            assert reference_value is not None
            reference_time = str(_iso_timestamp(reference_value))
            append_batch_id = _chat_memory_append_batch_id(
                user_id=head.user_id,
                project_id=head.project_id,
                session_id=head.session_id,
                event_seq=event_seq,
                message_ids=[record.id for record in records],
            )
            next_seq = (
                int(
                    await conn.fetchval(
                        """
                        SELECT COALESCE(MAX(seq), 0)
                        FROM enterprise_chat_messages WHERE session_id = $1
                        """,
                        head.session_id,
                    )
                )
                + 1
            )
            for index, record in enumerate(records):
                record.seq = next_seq + index
                record.append_batch_id = append_batch_id
                record.project_event_seq = event_seq
                record.memory_reference_time = reference_time
                await conn.execute(
                    """
                    INSERT INTO enterprise_chat_messages (
                        id, session_id, project_id, user_id, seq, created_at,
                        append_batch_id, project_event_seq, memory_reference_time,
                        data_json
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9,
                              $10::jsonb)
                    """,
                    record.id,
                    record.session_id,
                    record.project_id,
                    record.user_id,
                    record.seq,
                    record.created_at,
                    append_batch_id,
                    event_seq,
                    reference_value,
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
            event_id, deterministic_key = _chat_memory_event_identity(
                event_type="ingest",
                user_id=head.user_id,
                project_id=head.project_id,
                event_seq=event_seq,
                generation=group.desired_generation,
                append_batch_id=append_batch_id,
                target_session_id=head.session_id,
            )
            now = str(_iso_timestamp(await conn.fetchval("SELECT clock_timestamp()")))
            await self._insert_postgres_chat_memory_event(
                conn,
                ChatMemoryOutboxEventRecord(
                    event_id=event_id,
                    deterministic_key=deterministic_key,
                    user_id=head.user_id,
                    project_id=head.project_id,
                    event_seq=event_seq,
                    generation=group.desired_generation,
                    graph_group_id=chat_memory_graph_group_id(
                        head.user_id, head.project_id, group.desired_generation
                    ),
                    config_fingerprint=group.desired_config_fingerprint,
                    graph_store_fingerprint=group.desired_graph_store_fingerprint,
                    event_type="ingest",
                    status="pending",
                    available_at=now,
                    attempt_no=0,
                    created_at=now,
                    updated_at=now,
                    source_session_id=head.session_id,
                    append_batch_id=append_batch_id,
                    first_seq=next_seq,
                    last_seq=next_seq + len(records) - 1,
                    actor_user_id=actor_user_id or head.user_id,
                    actor_tenant_id=actor_tenant_id,
                    target_user_id=head.user_id,
                    target_project_id=head.project_id,
                    target_session_id=head.session_id,
                ),
            )
            rows = await conn.fetch(
                """
                SELECT append_batch_id, project_event_seq, memory_reference_time,
                       data_json
                FROM enterprise_chat_messages
                WHERE append_batch_id = $1
                ORDER BY seq ASC, id ASC
                """,
                append_batch_id,
            )
            return [_chat_message_from_row(row) for row in rows]

        return await self._write(write)

    async def delete_enterprise_user_with_memory(
        self,
        user_id: str,
        *,
        config_fingerprint: str,
        graph_store_fingerprint: str | None = None,
        actor_user_id: str | None = None,
        actor_tenant_id: str | None = None,
        expected_updated_at: Any = _EXPECTATION_UNSET,
        expected_token_version: Any = _EXPECTATION_UNSET,
        expected_tenant_id: Any = _EXPECTATION_UNSET,
        expected_membership: Any = _EXPECTATION_UNSET,
    ) -> bool:
        """Create sorted per-project purge work before deleting a user."""

        fingerprint = _validate_chat_memory_fingerprint(config_fingerprint)
        graph_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            fingerprint, graph_store_fingerprint
        )
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
                    WHERE user_id = $1 ORDER BY tenant_id ASC FOR UPDATE
                    """,
                    user_id,
                )
                _assert_enterprise_user_membership_precondition(
                    user_id,
                    [_tenant_membership_from_row(row) for row in membership_rows],
                    expected_membership=expected_membership,
                )

            project_rows = await conn.fetch(
                """
                SELECT project_id FROM (
                    SELECT id AS project_id FROM enterprise_chat_projects
                    WHERE user_id = $1
                    UNION
                    SELECT project_id FROM enterprise_chat_memory_groups
                    WHERE user_id = $1
                    UNION
                    SELECT project_id FROM enterprise_chat_memory_generations
                    WHERE user_id = $1
                    UNION
                    SELECT project_id FROM enterprise_chat_memory_episodes
                    WHERE user_id = $1
                    UNION
                    SELECT project_id FROM enterprise_chat_memory_outbox
                    WHERE user_id = $1
                ) projects ORDER BY project_id ASC
                """,
                user_id,
            )
            for project_row in project_rows:
                project_id = str(project_row["project_id"])
                await conn.fetchval(
                    """
                    SELECT id FROM enterprise_chat_projects
                    WHERE id = $1 AND user_id = $2 FOR UPDATE
                    """,
                    project_id,
                    user_id,
                )
                await conn.fetch(
                    """
                    SELECT id FROM enterprise_chat_sessions
                    WHERE project_id = $1 AND user_id = $2
                    ORDER BY id FOR UPDATE
                    """,
                    project_id,
                    user_id,
                )
                await conn.fetch(
                    """
                    SELECT id FROM enterprise_chat_messages
                    WHERE project_id = $1 AND user_id = $2
                    ORDER BY session_id, seq, id FOR UPDATE
                    """,
                    project_id,
                    user_id,
                )
                await self._enqueue_postgres_chat_memory_purge(
                    conn,
                    user_id,
                    project_id,
                    fingerprint,
                    graph_fingerprint,
                    actor_user_id=actor_user_id or user_id,
                    actor_tenant_id=actor_tenant_id,
                )

            await conn.execute(
                "DELETE FROM enterprise_tenant_memberships WHERE user_id = $1",
                user_id,
            )
            await conn.execute(
                "DELETE FROM enterprise_tenant_user_kb_overrides WHERE user_id = $1",
                user_id,
            )
            await conn.execute(
                "DELETE FROM enterprise_kb_acl WHERE user_id = $1", user_id
            )
            await conn.execute(
                "DELETE FROM enterprise_user_kb_query_settings WHERE user_id = $1",
                user_id,
            )
            await conn.execute(
                "DELETE FROM enterprise_chat_messages WHERE user_id = $1", user_id
            )
            await conn.execute(
                "DELETE FROM enterprise_chat_sessions WHERE user_id = $1", user_id
            )
            await conn.execute(
                "DELETE FROM enterprise_chat_projects WHERE user_id = $1", user_id
            )
            status = await conn.execute(
                "DELETE FROM enterprise_users WHERE id = $1", user_id
            )
            return _rowcount(status) > 0

        return await self._write(write)

    async def get_chat_memory_group(
        self, user_id: str, project_id: str
    ) -> ChatMemoryGroupRecord | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            return await self._get_postgres_chat_memory_group(
                conn, user_id, project_id, for_update=False
            )

    async def get_chat_memory_read_token(
        self, user_id: str, project_id: str
    ) -> ChatMemoryReadToken | None:
        """Read one complete logical/active-generation identity in one SQL."""

        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT groups.state, groups.state_version,
                       groups.active_generation,
                       groups.active_config_fingerprint,
                       groups.active_graph_store_fingerprint,
                       generation.graph_group_id,
                       generation.state AS generation_state
                FROM enterprise_chat_memory_groups AS groups
                LEFT JOIN enterprise_chat_memory_generations AS generation
                  ON generation.user_id = groups.user_id
                 AND generation.project_id = groups.project_id
                 AND generation.generation = groups.active_generation
                WHERE groups.user_id = $1 AND groups.project_id = $2
                """,
                user_id,
                project_id,
            )
        if row is None:
            return None
        return ChatMemoryReadToken(
            user_id=user_id,
            project_id=project_id,
            state=str(row["state"]),  # type: ignore[arg-type]
            state_version=int(row["state_version"]),
            active_generation=(
                int(row["active_generation"])
                if row["active_generation"] is not None
                else None
            ),
            active_config_fingerprint=row["active_config_fingerprint"],
            active_graph_store_fingerprint=row[
                "active_graph_store_fingerprint"
            ],
            graph_group_id=row["graph_group_id"],
            generation_state=(
                str(row["generation_state"])  # type: ignore[arg-type]
                if row["generation_state"] is not None
                else None
            ),
        )

    async def get_chat_memory_generation(
        self, user_id: str, project_id: str, generation: int
    ) -> ChatMemoryGenerationRecord | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM enterprise_chat_memory_generations
                WHERE user_id = $1 AND project_id = $2 AND generation = $3
                """,
                user_id,
                project_id,
                int(generation),
            )
        return _chat_memory_generation_from_row(row) if row is not None else None

    async def list_chat_memory_generations(
        self, user_id: str, project_id: str
    ) -> list[ChatMemoryGenerationRecord]:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM enterprise_chat_memory_generations
                WHERE user_id = $1 AND project_id = $2
                ORDER BY generation ASC
                """,
                user_id,
                project_id,
            )
        return [_chat_memory_generation_from_row(row) for row in rows]

    async def get_chat_memory_event(
        self, event_id: str
    ) -> ChatMemoryOutboxEventRecord | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM enterprise_chat_memory_outbox WHERE event_id = $1",
                event_id,
            )
        return _chat_memory_event_from_row(row) if row is not None else None

    async def get_chat_memory_event_by_sequence(
        self, user_id: str, project_id: str, event_seq: int
    ) -> ChatMemoryOutboxEventRecord | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM enterprise_chat_memory_outbox
                WHERE user_id = $1 AND project_id = $2 AND event_seq = $3
                """,
                user_id,
                project_id,
                int(event_seq),
            )
        return _chat_memory_event_from_row(row) if row is not None else None

    async def list_chat_memory_events(
        self,
        *,
        user_id: str | None = None,
        project_id: str | None = None,
        status: ChatMemoryEventStatus | None = None,
        event_type: ChatMemoryEventType | None = None,
        limit: int = 100,
    ) -> list[ChatMemoryOutboxEventRecord]:
        await self._ensure_initialized()
        limit = max(1, min(int(limit), 1000))
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("user_id", user_id),
            ("project_id", project_id),
            ("status", status),
            ("event_type", event_type),
        ):
            if value is not None:
                params.append(value)
                clauses.append(f"{column} = ${len(params)}")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT * FROM enterprise_chat_memory_outbox
                {where}
                ORDER BY user_id, project_id, event_seq
                LIMIT ${len(params)}
                """,
                *params,
            )
        return [_chat_memory_event_from_row(row) for row in rows]

    async def count_chat_memory_events(
        self,
        *,
        status: ChatMemoryEventStatus | None = None,
        event_type: ChatMemoryEventType | None = None,
    ) -> int:
        await self._ensure_initialized()
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            params.append(status)
            clauses.append(f"status = ${len(params)}")
        if event_type is not None:
            params.append(event_type)
            clauses.append(f"event_type = ${len(params)}")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        async with self._pool_or_raise().acquire() as conn:
            value = await conn.fetchval(
                f"SELECT COUNT(*) FROM enterprise_chat_memory_outbox {where}",
                *params,
            )
        return int(value or 0)

    async def list_admitted_chat_memory_replay_batches(
        self,
        user_id: str,
        project_id: str,
        *,
        through_event_seq: int,
        after_event_seq: int = 0,
        limit: int = 100,
    ) -> list[ChatMemoryReplayBatch]:
        """Page complete surviving admitted batches through a fixed cutoff."""

        await self._ensure_initialized()
        cutoff = max(0, int(through_event_seq))
        after = max(0, int(after_event_seq))
        limit = max(1, min(int(limit), 1000))
        async with self._pool_or_raise().acquire() as conn:
            batch_rows = await conn.fetch(
                """
                SELECT project_event_seq, append_batch_id
                FROM enterprise_chat_messages
                WHERE user_id = $1 AND project_id = $2
                  AND project_event_seq IS NOT NULL
                  AND append_batch_id IS NOT NULL
                  AND project_event_seq > $3 AND project_event_seq <= $4
                GROUP BY project_event_seq, append_batch_id
                ORDER BY project_event_seq ASC
                LIMIT $5
                """,
                user_id,
                project_id,
                after,
                cutoff,
                limit,
            )
            if not batch_rows:
                return []
            event_seqs = [int(row["project_event_seq"]) for row in batch_rows]
            message_rows = await conn.fetch(
                """
                SELECT append_batch_id, project_event_seq, memory_reference_time,
                       data_json
                FROM enterprise_chat_messages
                WHERE user_id = $1 AND project_id = $2
                  AND project_event_seq = ANY($3::bigint[])
                ORDER BY project_event_seq ASC, session_id ASC, seq ASC, id ASC
                """,
                user_id,
                project_id,
                event_seqs,
            )
        by_event: dict[int, list[ChatMessageRecord]] = {
            event_seq: [] for event_seq in event_seqs
        }
        for row in message_rows:
            record = _chat_message_from_row(row)
            assert record.project_event_seq is not None
            by_event[record.project_event_seq].append(record)
        batches: list[ChatMemoryReplayBatch] = []
        for batch_row in batch_rows:
            event_seq = int(batch_row["project_event_seq"])
            messages = by_event[event_seq]
            if not messages:
                continue
            first = messages[0]
            assert first.append_batch_id is not None
            assert first.memory_reference_time is not None
            batches.append(
                ChatMemoryReplayBatch(
                    append_batch_id=first.append_batch_id,
                    project_event_seq=event_seq,
                    memory_reference_time=first.memory_reference_time,
                    session_id=first.session_id,
                    messages=messages,
                )
            )
        return batches

    async def enqueue_chat_memory_rebuild(
        self,
        user_id: str,
        project_id: str,
        config_fingerprint: str,
        *,
        graph_store_fingerprint: str | None = None,
        actor_user_id: str | None = None,
        actor_tenant_id: str | None = None,
    ) -> ChatMemoryOutboxEventRecord | None:
        """Durably enqueue an administrative rebuild, coalescing current work."""

        fingerprint = _validate_chat_memory_fingerprint(config_fingerprint)
        graph_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            fingerprint, graph_store_fingerprint
        )
        await self._ensure_initialized()

        async def write(conn: Any) -> ChatMemoryOutboxEventRecord | None:
            await self._lock_postgres_chat_memory_logical_group_transaction(
                conn, user_id, project_id
            )
            group = await self._get_postgres_chat_memory_group(
                conn, user_id, project_id, for_update=True
            )
            if group is not None:
                await self._assert_postgres_chat_memory_graph_store_invariant(
                    conn, group, graph_fingerprint
                )
                if group.state in {"deleting", "deleted"}:
                    return None
            else:
                group, _ = await self._ensure_postgres_chat_memory_group(
                    conn,
                    user_id,
                    project_id,
                    fingerprint,
                    graph_fingerprint,
                    generation_state="building",
                )
            existing = await conn.fetchrow(
                """
                SELECT * FROM enterprise_chat_memory_outbox
                WHERE user_id = $1 AND project_id = $2
                  AND event_type = 'rebuild' AND generation = $3
                  AND config_fingerprint = $4
                  AND graph_store_fingerprint = $5
                  AND status IN ('pending', 'running', 'retry_wait')
                ORDER BY event_seq DESC LIMIT 1 FOR UPDATE
                """,
                user_id,
                project_id,
                group.desired_generation,
                fingerprint,
                graph_fingerprint,
            )
            if existing is not None:
                return _chat_memory_event_from_row(existing)
            return await self._enqueue_postgres_chat_memory_rebuild(
                conn,
                group,
                fingerprint,
                graph_fingerprint,
                actor_user_id=actor_user_id,
                actor_tenant_id=actor_tenant_id,
                target_session_id=None,
                target_message_id=None,
            )

        return await self._write(write)

    async def enqueue_chat_memory_purge(
        self,
        user_id: str,
        project_id: str,
        config_fingerprint: str,
        *,
        graph_store_fingerprint: str | None = None,
        actor_user_id: str | None = None,
        actor_tenant_id: str | None = None,
    ) -> ChatMemoryOutboxEventRecord | None:
        """Durably enqueue an administrative purge without reviving terminal work."""

        fingerprint = _validate_chat_memory_fingerprint(config_fingerprint)
        graph_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            fingerprint, graph_store_fingerprint
        )
        await self._ensure_initialized()

        async def write(conn: Any) -> ChatMemoryOutboxEventRecord | None:
            await self._lock_postgres_chat_memory_logical_group_transaction(
                conn, user_id, project_id
            )
            group = await self._get_postgres_chat_memory_group(
                conn, user_id, project_id, for_update=True
            )
            if group is not None and group.state == "deleted":
                return None
            existing = await conn.fetchrow(
                """
                SELECT * FROM enterprise_chat_memory_outbox
                WHERE user_id = $1 AND project_id = $2 AND event_type = 'purge'
                  AND status IN ('pending', 'running', 'retry_wait')
                ORDER BY event_seq DESC LIMIT 1 FOR UPDATE
                """,
                user_id,
                project_id,
            )
            if existing is not None:
                return _chat_memory_event_from_row(existing)
            return await self._enqueue_postgres_chat_memory_purge(
                conn,
                user_id,
                project_id,
                fingerprint,
                graph_fingerprint,
                actor_user_id=actor_user_id,
                actor_tenant_id=actor_tenant_id,
            )

        return await self._write(write)

    async def _materialize_postgres_chat_memory_rebuild_batches(
        self,
        conn: Any,
        user_id: str,
        project_id: str,
        cutoff: int,
    ) -> list[ChatMemoryReplayBatch]:
        """Fetch full replay JSON only after aggregate cap preflight succeeds."""

        rows = await conn.fetch(
            """
            SELECT append_batch_id, project_event_seq, memory_reference_time,
                   data_json
            FROM enterprise_chat_messages
            WHERE user_id = $1 AND project_id = $2
              AND project_event_seq IS NOT NULL
              AND append_batch_id IS NOT NULL
              AND project_event_seq <= $3
            ORDER BY project_event_seq ASC, session_id ASC, seq ASC, id ASC
            """,
            user_id,
            project_id,
            int(cutoff),
        )
        return _chat_memory_replay_batches_from_messages(
            [_chat_message_from_row(row) for row in rows]
        )

    async def prepare_chat_memory_rebuild_snapshot(
        self,
        event_id: str,
        claim_token: str,
        runtime_fingerprint: str,
        max_messages: int,
        max_bytes: int,
        ingest_max_chars: int = CHAT_MEMORY_DEFAULT_INGEST_MAX_CHARS,
        *,
        runtime_graph_store_fingerprint: str | None = None,
    ) -> ChatMemoryRebuildSnapshot | None:
        """Capture and persist one complete fixed-cutoff rebuild snapshot."""

        fingerprint = _validate_chat_memory_fingerprint(runtime_fingerprint)
        graph_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            fingerprint, runtime_graph_store_fingerprint
        )
        message_cap = int(max_messages)
        byte_cap = int(max_bytes)
        episode_max_chars = _validate_chat_memory_ingest_max_chars(ingest_max_chars)
        if message_cap < 0 or byte_cap < 0:
            raise ValueError("Chat Memory rebuild caps must be non-negative")
        await self._ensure_initialized()

        async def write(conn: Any) -> ChatMemoryRebuildSnapshot | None:
            state = await self._require_postgres_chat_memory_running_claim(
                conn, event_id, claim_token
            )
            stale = await self._resolve_postgres_stale_chat_memory_execution(
                conn, state, claim_token
            )
            if stale is not None:
                return None
            if not self._postgres_chat_memory_runtime_identity_matches(
                state.event, fingerprint, graph_fingerprint
            ):
                await self._retry_postgres_chat_memory_runtime_mismatch(
                    conn,
                    state,
                    claim_token,
                    fingerprint,
                    graph_fingerprint,
                    retry_delay_seconds=1.0,
                )
                return None
            self._validate_postgres_chat_memory_execution_fence(
                state, fingerprint, graph_fingerprint
            )
            event = state.event
            if event.event_type != "rebuild":
                raise MetadataConflictError(
                    "chat_memory_event",
                    event_id,
                    expected={"event_type": "rebuild"},
                    current={"event_type": event.event_type},
                )
            if state.group.state != "rebuilding" or state.generation.state != "building":
                raise MetadataConflictError(
                    "chat_memory_rebuild",
                    event_id,
                    expected={
                        "group_state": "rebuilding",
                        "generation_state": "building",
                    },
                    current={
                        "group_state": state.group.state,
                        "generation_state": state.generation.state,
                    },
                )
            if state.group.active_rebuild_event_id != event_id:
                raise MetadataConflictError(
                    "chat_memory_rebuild",
                    event_id,
                    expected={"active_rebuild_event_id": event_id},
                    current={
                        "active_rebuild_event_id": state.group.active_rebuild_event_id
                    },
                )
            if event.side_effect_started_at is not None:
                raise MetadataConflictError(
                    "chat_memory_event",
                    event_id,
                    expected={"side_effect_started_at": None},
                    current={
                        "side_effect_started_at": event.side_effect_started_at
                    },
                )

            cutoff = (
                event.snapshot_cutoff
                if event.snapshot_cutoff is not None
                else state.group.next_event_seq - 1
            )
            metrics = await conn.fetchrow(
                """
                SELECT COUNT(*)::bigint AS message_count,
                       COALESCE(SUM(octet_length(COALESCE(
                           data_json->>'content', ''
                       ))), 0)::bigint AS byte_count,
                       COUNT(DISTINCT (
                           project_event_seq, append_batch_id
                       ))::bigint AS batch_count
                FROM enterprise_chat_messages
                WHERE user_id = $1 AND project_id = $2
                  AND project_event_seq IS NOT NULL
                  AND append_batch_id IS NOT NULL
                  AND project_event_seq <= $3
                """,
                event.user_id,
                event.project_id,
                cutoff,
            )
            assert metrics is not None
            batch_count = int(metrics["batch_count"] or 0)
            message_count = int(metrics["message_count"] or 0)
            byte_count = int(metrics["byte_count"] or 0)
            control_time = await conn.fetchval("SELECT clock_timestamp()")
            if message_count > message_cap or byte_count > byte_cap:
                error_message = (
                    "Rebuild snapshot exceeds hard cap: "
                    f"messages={message_count}/{message_cap}, "
                    f"bytes={byte_count}/{byte_cap}"
                )
                await conn.execute(
                    """
                    UPDATE enterprise_chat_memory_generations
                    SET snapshot_cutoff = $1, replay_batch_count = $2,
                        replay_message_count = $3, replay_byte_count = $4,
                        snapshot_digest = NULL,
                        last_error_code = 'rebuild_snapshot_hard_cap_exceeded',
                        last_error_message = $5, last_error_at = $6,
                        updated_at = $6
                    WHERE user_id = $7 AND project_id = $8 AND generation = $9
                      AND state = 'building'
                    """,
                    cutoff,
                    batch_count,
                    message_count,
                    byte_count,
                    error_message,
                    control_time,
                    event.user_id,
                    event.project_id,
                    event.generation,
                )
                dead = await conn.fetchrow(
                    """
                    UPDATE enterprise_chat_memory_outbox
                    SET status = 'dead_letter', snapshot_cutoff = $1,
                        snapshot_batch_count = $2, snapshot_message_count = $3,
                        snapshot_byte_count = $4, snapshot_digest = NULL,
                        claim_token = NULL,
                        claimed_by = NULL, claimed_at = NULL,
                        side_effect_started_at = NULL,
                        side_effect_state_version = NULL, completed_at = $5,
                        last_error_code = 'rebuild_snapshot_hard_cap_exceeded',
                        last_error_message = $6, last_error_at = $5,
                        updated_at = $5
                    WHERE event_id = $7 AND status = 'running' AND claim_token = $8
                      AND side_effect_started_at IS NULL
                    RETURNING *
                    """,
                    cutoff,
                    batch_count,
                    message_count,
                    byte_count,
                    control_time,
                    error_message,
                    event_id,
                    claim_token,
                )
                if dead is None:
                    raise MetadataConflictError(
                        "chat_memory_event",
                        event_id,
                        expected={"status": "running", "claim_token": claim_token},
                        current={"status": event.status, "claim_token": event.claim_token},
                    )
                await conn.execute(
                    """
                    UPDATE enterprise_chat_memory_groups
                    SET state = 'failed', state_version = state_version + 1,
                        last_error_code = 'rebuild_snapshot_hard_cap_exceeded',
                        last_error_message = $1, last_error_at = $2,
                        updated_at = $2
                    WHERE user_id = $3 AND project_id = $4
                    """,
                    error_message,
                    control_time,
                    event.user_id,
                    event.project_id,
                )
                return None

            replay_batches = (
                await self._materialize_postgres_chat_memory_rebuild_batches(
                    conn,
                    event.user_id,
                    event.project_id,
                    cutoff,
                )
            )
            materialized_metrics = _chat_memory_replay_snapshot_metrics(replay_batches)
            if materialized_metrics != (batch_count, message_count, byte_count):
                raise MetadataConflictError(
                    "chat_memory_rebuild_snapshot",
                    event_id,
                    expected={
                        "aggregate_metrics": (batch_count, message_count, byte_count)
                    },
                    current={"materialized_metrics": materialized_metrics},
                )
            snapshot_digest = _chat_memory_snapshot_digest(
                replay_batches,
                ingest_max_chars=episode_max_chars,
            )
            persisted_digests = (
                event.snapshot_digest,
                state.generation.snapshot_digest,
            )
            if persisted_digests not in {
                (None, None),
                (snapshot_digest, snapshot_digest),
            }:
                raise MetadataConflictError(
                    "chat_memory_rebuild_snapshot",
                    event_id,
                    expected={"snapshot_digest": persisted_digests},
                    current={"snapshot_digest": snapshot_digest},
                )

            generation_row = await conn.fetchrow(
                """
                UPDATE enterprise_chat_memory_generations
                SET snapshot_cutoff = $1, replay_batch_count = $2,
                    replay_message_count = $3, replay_byte_count = $4,
                    snapshot_digest = $5, last_error_code = NULL,
                    last_error_message = NULL, last_error_at = NULL,
                    updated_at = $6
                WHERE user_id = $7 AND project_id = $8 AND generation = $9
                  AND state = 'building' AND config_fingerprint = $10
                  AND graph_store_fingerprint = $11
                  AND graph_group_id = $12
                RETURNING *
                """,
                cutoff,
                batch_count,
                message_count,
                byte_count,
                snapshot_digest,
                control_time,
                event.user_id,
                event.project_id,
                event.generation,
                fingerprint,
                graph_fingerprint,
                event.graph_group_id,
            )
            event_row = await conn.fetchrow(
                """
                UPDATE enterprise_chat_memory_outbox
                SET snapshot_cutoff = $1, snapshot_batch_count = $2,
                    snapshot_message_count = $3, snapshot_byte_count = $4,
                    snapshot_digest = $5, updated_at = $6
                WHERE event_id = $7 AND status = 'running' AND claim_token = $8
                  AND side_effect_started_at IS NULL
                RETURNING *
                """,
                cutoff,
                batch_count,
                message_count,
                byte_count,
                snapshot_digest,
                control_time,
                event_id,
                claim_token,
            )
            if generation_row is None or event_row is None:
                raise MetadataConflictError(
                    "chat_memory_rebuild_snapshot",
                    event_id,
                    expected={
                        "event_status": "running",
                        "claim_token": claim_token,
                        "generation_state": "building",
                    },
                    current={
                        "event_status": event.status,
                        "claim_token": event.claim_token,
                        "generation_state": state.generation.state,
                    },
                )
            return ChatMemoryRebuildSnapshot(
                event_id=event.event_id,
                user_id=event.user_id,
                project_id=event.project_id,
                generation=event.generation,
                graph_group_id=event.graph_group_id,
                config_fingerprint=fingerprint,
                graph_store_fingerprint=graph_fingerprint,
                group_state_version=state.group.state_version,
                snapshot_cutoff=cutoff,
                replay_batches=replay_batches,
                batch_count=batch_count,
                message_count=message_count,
                byte_count=byte_count,
                snapshot_digest=snapshot_digest,
                ingest_max_chars=episode_max_chars,
            )

        return await self._write(write)

    async def prepare_chat_memory_rebuild_targets(
        self,
        event_id: str,
        claim_token: str,
        runtime_fingerprint: str,
        *,
        runtime_graph_store_fingerprint: str | None = None,
    ) -> ChatMemoryRebuildTargetSet | None:
        """Capture the complete graph-group universe a rebuild must clear."""

        fingerprint = _validate_chat_memory_fingerprint(runtime_fingerprint)
        graph_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            fingerprint, runtime_graph_store_fingerprint
        )
        await self._ensure_initialized()

        async def write(conn: Any) -> ChatMemoryRebuildTargetSet | None:
            state = await self._require_postgres_chat_memory_running_claim(
                conn, event_id, claim_token
            )
            stale = await self._resolve_postgres_stale_chat_memory_execution(
                conn, state, claim_token
            )
            if stale is not None:
                return None
            if not self._postgres_chat_memory_runtime_identity_matches(
                state.event, fingerprint, graph_fingerprint
            ):
                await self._retry_postgres_chat_memory_runtime_mismatch(
                    conn,
                    state,
                    claim_token,
                    fingerprint,
                    graph_fingerprint,
                    retry_delay_seconds=1.0,
                )
                return None
            self._validate_postgres_chat_memory_execution_fence(
                state, fingerprint, graph_fingerprint
            )
            event = state.event
            if (
                event.event_type != "rebuild"
                or state.group.state != "rebuilding"
                or state.generation.state != "building"
                or state.group.active_rebuild_event_id != event_id
            ):
                raise MetadataConflictError(
                    "chat_memory_rebuild",
                    event_id,
                    expected={
                        "event_type": "rebuild",
                        "group_state": "rebuilding",
                        "generation_state": "building",
                        "active_rebuild_event_id": event_id,
                    },
                    current={
                        "event_type": event.event_type,
                        "group_state": state.group.state,
                        "generation_state": state.generation.state,
                        "active_rebuild_event_id": (
                            state.group.active_rebuild_event_id
                        ),
                    },
                )
            if event.side_effect_started_at is not None:
                raise MetadataConflictError(
                    "chat_memory_event",
                    event_id,
                    expected={"side_effect_started_at": None},
                    current={
                        "side_effect_started_at": event.side_effect_started_at
                    },
                )
            await self._assert_postgres_chat_memory_graph_store_invariant(
                conn, state.group, graph_fingerprint
            )
            return ChatMemoryRebuildTargetSet(
                event_id=event.event_id,
                user_id=event.user_id,
                project_id=event.project_id,
                logical_group_id=state.group.logical_group_id,
                group_ids=await self._postgres_chat_memory_rebuild_group_ids(
                    conn, event
                ),
            )

        return await self._write(write)

    async def claim_next_chat_memory_event(
        self,
        runtime_fingerprint: str,
        *,
        runtime_graph_store_fingerprint: str | None = None,
        worker_id: str | None = None,
        event_types: Sequence[ChatMemoryEventType] | None = None,
    ) -> ChatMemoryOutboxEventRecord | None:
        """Claim one globally ordered eligible per-group FIFO head."""

        fingerprint = _validate_chat_memory_fingerprint(runtime_fingerprint)
        graph_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            fingerprint, runtime_graph_store_fingerprint
        )
        claimed_by = _validate_chat_memory_worker_id(worker_id)
        claimable_types = _normalize_chat_memory_event_types(event_types)
        if not claimable_types:
            return None
        claim_token = _new_chat_memory_claim_token()
        await self._ensure_initialized()

        async def write(conn: Any) -> ChatMemoryOutboxEventRecord | None:
            row = await conn.fetchrow(
                """
                WITH candidate AS (
                    SELECT event.event_id
                    FROM enterprise_chat_memory_outbox AS event
                    WHERE event.status IN ('pending', 'retry_wait')
                      AND event.available_at <= clock_timestamp()
                      AND (
                          (event.event_type = 'purge'
                           AND event.graph_store_fingerprint = $2)
                          OR
                          (event.event_type IN ('ingest', 'rebuild')
                           AND event.config_fingerprint = $1
                           AND event.graph_store_fingerprint = $2)
                      )
                      AND event.event_type = ANY($5::text[])
                      AND NOT EXISTS (
                          SELECT 1
                          FROM enterprise_chat_memory_outbox AS blocker
                          WHERE blocker.user_id = event.user_id
                            AND blocker.project_id = event.project_id
                            AND blocker.event_seq < event.event_seq
                            AND blocker.status IN (
                                'pending', 'running', 'retry_wait', 'dead_letter'
                            )
                      )
                    ORDER BY event.available_at ASC,
                             event.event_seq ASC,
                             event.user_id ASC,
                             event.project_id ASC,
                             event.event_id ASC
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                ), control AS (
                    SELECT clock_timestamp() AS control_time
                )
                UPDATE enterprise_chat_memory_outbox AS event
                SET status = 'running', attempt_no = event.attempt_no + 1,
                    claim_token = $3, claimed_by = $4,
                    claimed_at = control.control_time,
                    side_effect_started_at = NULL,
                    side_effect_state_version = NULL, completed_at = NULL,
                    last_error_code = NULL, last_error_message = NULL,
                    last_error_at = NULL, updated_at = control.control_time
                FROM candidate, control
                WHERE event.event_id = candidate.event_id
                RETURNING event.*
                """,
                fingerprint,
                graph_fingerprint,
                claim_token,
                claimed_by,
                list(claimable_types),
            )
            return _chat_memory_event_from_row(row) if row is not None else None

        return await self._write(write)

    @asynccontextmanager
    async def chat_memory_group_execution_guard(
        self, logical_group_id: str, *, wait: bool = True
    ) -> AsyncIterator[bool]:
        """Own one logical Chat Memory group with a session advisory lock."""

        if not isinstance(logical_group_id, str) or not logical_group_id.strip():
            raise ValueError("Chat Memory logical_group_id must be non-empty")
        logical_group_id = logical_group_id.strip()
        await self._ensure_initialized()
        await self._ensure_operation_lock_pool()
        async with self._operation_session() as conn:
            locked = False
            try:
                if wait:
                    await conn.execute(
                        "SELECT pg_advisory_lock("
                        "hashtextextended($1, 1263295564))",
                        logical_group_id,
                    )
                    locked = True
                else:
                    locked = bool(
                        await conn.fetchval(
                            "SELECT pg_try_advisory_lock("
                            "hashtextextended($1, 1263295564))",
                            logical_group_id,
                        )
                    )
                yield locked
            finally:
                if locked:
                    await self._unlock_operation_guard(
                        conn,
                        "SELECT pg_advisory_unlock("
                        "hashtextextended($1, 1263295564))",
                        logical_group_id,
                    )

    async def get_chat_memory_execution_state(
        self, event_id: str
    ) -> ChatMemoryExecutionState | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            return await self._get_postgres_chat_memory_execution_state(
                conn, event_id, for_update=False
            )

    async def mark_chat_memory_event_side_effect_started(
        self,
        event_id: str,
        claim_token: str,
        runtime_fingerprint: str,
        *,
        runtime_graph_store_fingerprint: str | None = None,
        fingerprint_retry_delay_seconds: float = 1.0,
    ) -> ChatMemoryOutboxEventRecord:
        """Start only a current fenced side effect; status reports safe deferral."""

        fingerprint = _validate_chat_memory_fingerprint(runtime_fingerprint)
        graph_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            fingerprint, runtime_graph_store_fingerprint
        )
        await self._ensure_initialized()

        async def write(conn: Any) -> ChatMemoryOutboxEventRecord:
            state = await self._require_postgres_chat_memory_running_claim(
                conn, event_id, claim_token
            )
            stale = await self._resolve_postgres_stale_chat_memory_execution(
                conn, state, claim_token
            )
            if stale is not None:
                return stale
            if not self._postgres_chat_memory_runtime_identity_matches(
                state.event, fingerprint, graph_fingerprint
            ):
                return await self._retry_postgres_chat_memory_runtime_mismatch(
                    conn,
                    state,
                    claim_token,
                    fingerprint,
                    graph_fingerprint,
                    retry_delay_seconds=fingerprint_retry_delay_seconds,
                )
            if state.event.side_effect_started_at is None:
                row = await conn.fetchrow(
                    """
                    WITH control AS (
                        SELECT clock_timestamp() AS control_time
                    )
                    UPDATE enterprise_chat_memory_outbox
                    SET side_effect_started_at = control.control_time,
                        side_effect_state_version = $3,
                        updated_at = control.control_time
                    FROM control
                    WHERE event_id = $1 AND status = 'running'
                      AND claim_token = $2
                    RETURNING enterprise_chat_memory_outbox.*
                    """,
                    event_id,
                    claim_token,
                    state.group.state_version,
                )
                if row is None:
                    raise MetadataConflictError(
                        "chat_memory_event",
                        event_id,
                        expected={"status": "running", "claim_token": claim_token},
                        current={
                            "status": state.event.status,
                            "claim_token": state.event.claim_token,
                        },
                    )
                return _chat_memory_event_from_row(row)
            return state.event

        return await self._write(write)

    async def finalize_chat_memory_ingest(
        self,
        event_id: str,
        claim_token: str,
        runtime_fingerprint: str,
        *,
        episode_uuid: str,
        runtime_graph_store_fingerprint: str | None = None,
    ) -> ChatMemoryExecutionState | None:
        """Commit an ingest atomically; return None after stale supersession."""

        fingerprint = _validate_chat_memory_fingerprint(runtime_fingerprint)
        graph_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            fingerprint, runtime_graph_store_fingerprint
        )
        if not episode_uuid:
            raise ValueError("episode_uuid must be non-empty")
        await self._ensure_initialized()

        async def write(conn: Any) -> ChatMemoryExecutionState | None:
            state = await self._require_postgres_chat_memory_running_claim(
                conn, event_id, claim_token
            )
            stale = await self._resolve_postgres_stale_chat_memory_execution(
                conn, state, claim_token
            )
            if stale is not None:
                return None
            self._validate_postgres_chat_memory_ingest_execution(
                state, fingerprint, graph_fingerprint
            )
            event = state.event
            if event.side_effect_started_at is None:
                raise MetadataConflictError(
                    "chat_memory_event",
                    event_id,
                    expected={"side_effect_started_at": "set"},
                    current={"side_effect_started_at": None},
                )
            if (
                event.append_batch_id is None
                or event.source_session_id is None
                or event.first_seq is None
                or event.last_seq is None
            ):
                raise MetadataStoreError("Ingest event is missing source batch identity")
            control_time = await conn.fetchval("SELECT clock_timestamp()")
            expected_mapping = ChatMemoryEpisodeRecord(
                episode_uuid=episode_uuid,
                session_id=event.source_session_id,
                project_id=event.project_id,
                user_id=event.user_id,
                first_seq=event.first_seq,
                last_seq=event.last_seq,
                created_at=str(_iso_timestamp(control_time)),
                event_id=event.event_id,
                generation=event.generation,
                graph_group_id=event.graph_group_id,
                append_batch_id=event.append_batch_id,
                project_event_seq=event.event_seq,
            )
            await self._insert_postgres_chat_memory_historical_mapping(
                conn, expected_mapping
            )
            activate_first = (
                state.group.active_generation is None
                and state.group.state == "rebuilding"
                and state.generation.state == "building"
            )
            if activate_first:
                await conn.execute(
                    """
                    UPDATE enterprise_chat_memory_generations
                    SET state = 'active', activated_at = $1, updated_at = $1,
                        last_error_code = NULL, last_error_message = NULL,
                        last_error_at = NULL
                    WHERE user_id = $2 AND project_id = $3 AND generation = $4
                    """,
                    control_time,
                    event.user_id,
                    event.project_id,
                    event.generation,
                )
                await conn.execute(
                    """
                    UPDATE enterprise_chat_memory_groups
                    SET active_generation = $1, active_config_fingerprint = $2,
                        active_graph_store_fingerprint = $3,
                        state = 'active', state_version = state_version + 1,
                        active_rebuild_event_id = NULL, last_success_at = $4,
                        last_error_code = NULL, last_error_message = NULL,
                        last_error_at = NULL, updated_at = $4
                    WHERE user_id = $5 AND project_id = $6
                    """,
                    event.generation,
                    fingerprint,
                    graph_fingerprint,
                    control_time,
                    event.user_id,
                    event.project_id,
                )
            else:
                await conn.execute(
                    """
                    UPDATE enterprise_chat_memory_groups
                    SET last_success_at = $1, last_error_code = NULL,
                        last_error_message = NULL, last_error_at = NULL,
                        updated_at = $1
                    WHERE user_id = $2 AND project_id = $3
                    """,
                    control_time,
                    event.user_id,
                    event.project_id,
                )
            row = await conn.fetchrow(
                """
                UPDATE enterprise_chat_memory_outbox
                SET status = 'succeeded', completed_at = $1, updated_at = $1,
                    last_error_code = NULL, last_error_message = NULL,
                    last_error_at = NULL
                WHERE event_id = $2 AND status = 'running' AND claim_token = $3
                RETURNING *
                """,
                control_time,
                event_id,
                claim_token,
            )
            if row is None:
                raise MetadataConflictError(
                    "chat_memory_event",
                    event_id,
                    expected={"status": "running", "claim_token": claim_token},
                    current={"status": event.status, "claim_token": event.claim_token},
                )
            final = await self._get_postgres_chat_memory_execution_state(
                conn, event_id, for_update=False
            )
            assert final is not None
            return final

        return await self._write(write)

    async def finalize_chat_memory_ingest_noop(
        self,
        event_id: str,
        claim_token: str,
        runtime_fingerprint: str,
        *,
        runtime_graph_store_fingerprint: str | None = None,
    ) -> ChatMemoryExecutionState | None:
        """Atomically complete an ingest that required no Graphiti side effect."""

        fingerprint = _validate_chat_memory_fingerprint(runtime_fingerprint)
        graph_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            fingerprint, runtime_graph_store_fingerprint
        )
        await self._ensure_initialized()

        async def write(conn: Any) -> ChatMemoryExecutionState | None:
            state = await self._require_postgres_chat_memory_running_claim(
                conn, event_id, claim_token
            )
            stale = await self._resolve_postgres_stale_chat_memory_execution(
                conn, state, claim_token
            )
            if stale is not None:
                return None
            self._validate_postgres_chat_memory_ingest_execution(
                state, fingerprint, graph_fingerprint
            )
            event = state.event
            if (
                event.side_effect_started_at is not None
                or event.side_effect_state_version is not None
            ):
                raise MetadataConflictError(
                    "chat_memory_event",
                    event_id,
                    expected={
                        "side_effect_started_at": None,
                        "side_effect_state_version": None,
                    },
                    current={
                        "side_effect_started_at": event.side_effect_started_at,
                        "side_effect_state_version": event.side_effect_state_version,
                    },
                )
            if (
                event.append_batch_id is None
                or event.source_session_id is None
                or event.first_seq is None
                or event.last_seq is None
            ):
                raise MetadataStoreError("Ingest event is missing source batch identity")

            source_rows = await conn.fetch(
                """
                SELECT append_batch_id, project_event_seq, memory_reference_time,
                       data_json
                FROM enterprise_chat_messages
                WHERE user_id = $1 AND project_id = $2
                  AND project_event_seq = $3
                ORDER BY seq ASC, id ASC
                """,
                event.user_id,
                event.project_id,
                event.event_seq,
            )
            source_messages = [
                _chat_message_from_row(row) for row in source_rows
            ]
            _validate_chat_memory_ingest_source_batch(event, source_messages)
            eligible_payload = _chat_memory_canonical_episode_payload(
                source_messages,
                ingest_max_chars=CHAT_MEMORY_DEFAULT_INGEST_MAX_CHARS,
            )
            if eligible_payload["messages"]:
                raise MetadataConflictError(
                    "chat_memory_ingest_noop",
                    event_id,
                    expected={"eligible_payload": "empty"},
                    current={
                        "eligible_message_ids": tuple(
                            item["id"] for item in eligible_payload["messages"]
                        )
                    },
                )

            control_time = await conn.fetchval("SELECT clock_timestamp()")
            episode_uuid = _chat_memory_noop_episode_uuid(
                event_id=event.event_id,
                generation=event.generation,
                append_batch_id=event.append_batch_id,
            )
            await self._insert_postgres_chat_memory_historical_mapping(
                conn,
                ChatMemoryEpisodeRecord(
                    episode_uuid=episode_uuid,
                    session_id=event.source_session_id,
                    project_id=event.project_id,
                    user_id=event.user_id,
                    first_seq=event.first_seq,
                    last_seq=event.last_seq,
                    created_at=str(_iso_timestamp(control_time)),
                    event_id=event.event_id,
                    generation=event.generation,
                    graph_group_id=event.graph_group_id,
                    append_batch_id=event.append_batch_id,
                    project_event_seq=event.event_seq,
                ),
            )

            activate_first = (
                state.group.active_generation is None
                and state.group.state == "rebuilding"
                and state.generation.state == "building"
            )
            if activate_first:
                generation_row = await conn.fetchrow(
                    """
                    UPDATE enterprise_chat_memory_generations
                    SET state = 'active', activated_at = $1, updated_at = $1,
                        last_error_code = NULL, last_error_message = NULL,
                        last_error_at = NULL
                    WHERE user_id = $2 AND project_id = $3 AND generation = $4
                      AND state = 'building'
                    RETURNING *
                    """,
                    control_time,
                    event.user_id,
                    event.project_id,
                    event.generation,
                )
                group_row = await conn.fetchrow(
                    """
                    UPDATE enterprise_chat_memory_groups
                    SET active_generation = $1, active_config_fingerprint = $2,
                        active_graph_store_fingerprint = $3,
                        state = 'active', state_version = state_version + 1,
                        active_rebuild_event_id = NULL, last_success_at = $4,
                        last_error_code = NULL, last_error_message = NULL,
                        last_error_at = NULL, updated_at = $4
                    WHERE user_id = $5 AND project_id = $6
                      AND desired_generation = $1 AND state_version = $7
                      AND state = 'rebuilding'
                    RETURNING *
                    """,
                    event.generation,
                    fingerprint,
                    graph_fingerprint,
                    control_time,
                    event.user_id,
                    event.project_id,
                    state.group.state_version,
                )
                if generation_row is None or group_row is None:
                    raise MetadataConflictError(
                        "chat_memory_ingest_noop",
                        event_id,
                        expected={"generation_state": "building", "group": "current"},
                        current={
                            "generation_state": state.generation.state,
                            "group_state": state.group.state,
                        },
                    )
            else:
                await conn.execute(
                    """
                    UPDATE enterprise_chat_memory_groups
                    SET last_success_at = $1, last_error_code = NULL,
                        last_error_message = NULL, last_error_at = NULL,
                        updated_at = $1
                    WHERE user_id = $2 AND project_id = $3
                    """,
                    control_time,
                    event.user_id,
                    event.project_id,
                )

            event_row = await conn.fetchrow(
                """
                UPDATE enterprise_chat_memory_outbox
                SET status = 'succeeded', completed_at = $1, updated_at = $1,
                    last_error_code = NULL, last_error_message = NULL,
                    last_error_at = NULL
                WHERE event_id = $2 AND status = 'running' AND claim_token = $3
                  AND side_effect_started_at IS NULL
                  AND side_effect_state_version IS NULL
                RETURNING *
                """,
                control_time,
                event_id,
                claim_token,
            )
            if event_row is None:
                raise MetadataConflictError(
                    "chat_memory_event",
                    event_id,
                    expected={"status": "running", "claim_token": claim_token},
                    current={"status": event.status, "claim_token": event.claim_token},
                )
            final = await self._get_postgres_chat_memory_execution_state(
                conn, event_id, for_update=False
            )
            assert final is not None
            return final

        return await self._write(write)

    async def finalize_chat_memory_rebuild(
        self,
        event_id: str,
        claim_token: str,
        runtime_fingerprint: str,
        snapshot: ChatMemoryRebuildSnapshot,
        mappings: Sequence[ChatMemoryReplayMappingInput],
        targets: ChatMemoryRebuildTargetSet,
        definitely_cleared_group_ids: Sequence[str],
        *,
        runtime_graph_store_fingerprint: str | None = None,
    ) -> ChatMemoryExecutionState | None:
        """Atomically install a complete replay and activate its generation."""

        fingerprint = _validate_chat_memory_fingerprint(runtime_fingerprint)
        graph_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            fingerprint, runtime_graph_store_fingerprint
        )
        if not isinstance(targets, ChatMemoryRebuildTargetSet):
            raise TypeError("targets must be a ChatMemoryRebuildTargetSet")
        replay_mappings = list(mappings)
        normalized_targets = _normalize_chat_memory_group_ids(targets.group_ids)
        normalized_cleared = _normalize_chat_memory_group_ids(
            definitely_cleared_group_ids
        )
        await self._ensure_initialized()

        async def write(conn: Any) -> ChatMemoryExecutionState | None:
            state = await self._require_postgres_chat_memory_running_claim(
                conn, event_id, claim_token
            )
            stale = await self._resolve_postgres_stale_chat_memory_execution(
                conn, state, claim_token
            )
            if stale is not None:
                return None
            self._validate_postgres_chat_memory_execution_fence(
                state, fingerprint, graph_fingerprint
            )
            event = state.event
            if event.event_type != "rebuild":
                raise MetadataConflictError(
                    "chat_memory_event",
                    event_id,
                    expected={"event_type": "rebuild"},
                    current={"event_type": event.event_type},
                )
            if state.group.active_rebuild_event_id != event_id:
                raise MetadataConflictError(
                    "chat_memory_rebuild",
                    event_id,
                    expected={"active_rebuild_event_id": event_id},
                    current={
                        "active_rebuild_event_id": state.group.active_rebuild_event_id
                    },
                )
            if event.side_effect_started_at is None:
                raise MetadataConflictError(
                    "chat_memory_event",
                    event_id,
                    expected={"side_effect_started_at": "set"},
                    current={"side_effect_started_at": None},
                )

            await self._assert_postgres_chat_memory_graph_store_invariant(
                conn, state.group, graph_fingerprint
            )
            expected_target_identity = (
                event.event_id,
                event.user_id,
                event.project_id,
                state.group.logical_group_id,
            )
            current_target_identity = (
                targets.event_id,
                targets.user_id,
                targets.project_id,
                targets.logical_group_id,
            )
            if current_target_identity != expected_target_identity:
                raise MetadataConflictError(
                    "chat_memory_rebuild_targets",
                    event_id,
                    expected={"identity": expected_target_identity},
                    current={"identity": current_target_identity},
                )
            authoritative_targets = (
                await self._postgres_chat_memory_rebuild_group_ids(conn, event)
            )
            if normalized_targets != authoritative_targets:
                raise MetadataConflictError(
                    "chat_memory_rebuild_targets",
                    event_id,
                    expected={"group_ids": authoritative_targets},
                    current={"group_ids": normalized_targets},
                )
            missing_clears = tuple(
                sorted(set(authoritative_targets).difference(normalized_cleared))
            )
            if missing_clears:
                raise MetadataConflictError(
                    "chat_memory_rebuild_clear",
                    event_id,
                    expected={"definitely_cleared": authoritative_targets},
                    current={
                        "definitely_cleared": normalized_cleared,
                        "missing": missing_clears,
                    },
                )

            identity_expected = {
                "event_id": event.event_id,
                "user_id": event.user_id,
                "project_id": event.project_id,
                "generation": event.generation,
                "graph_group_id": event.graph_group_id,
                "config_fingerprint": fingerprint,
                "graph_store_fingerprint": graph_fingerprint,
                "group_state_version": state.group.state_version,
            }
            identity_current = {
                "event_id": snapshot.event_id,
                "user_id": snapshot.user_id,
                "project_id": snapshot.project_id,
                "generation": snapshot.generation,
                "graph_group_id": snapshot.graph_group_id,
                "config_fingerprint": snapshot.config_fingerprint,
                "graph_store_fingerprint": snapshot.graph_store_fingerprint,
                "group_state_version": snapshot.group_state_version,
            }
            if identity_current != identity_expected:
                raise MetadataConflictError(
                    "chat_memory_rebuild_snapshot",
                    event_id,
                    expected=identity_expected,
                    current=identity_current,
                )
            if event.side_effect_state_version != snapshot.group_state_version:
                raise MetadataConflictError(
                    "chat_memory_rebuild_snapshot",
                    event_id,
                    expected={
                        "side_effect_state_version": snapshot.group_state_version
                    },
                    current={
                        "side_effect_state_version": event.side_effect_state_version
                    },
                )

            invocation_digest = _chat_memory_snapshot_digest(
                snapshot.replay_batches,
                ingest_max_chars=snapshot.ingest_max_chars,
            )
            persisted_digests = (
                event.snapshot_digest,
                state.generation.snapshot_digest,
            )
            if (
                snapshot.snapshot_digest != invocation_digest
                or persisted_digests != (invocation_digest, invocation_digest)
            ):
                raise MetadataConflictError(
                    "chat_memory_rebuild_snapshot",
                    event_id,
                    expected={"snapshot_digest": persisted_digests},
                    current={
                        "snapshot_digest": snapshot.snapshot_digest,
                        "recomputed_snapshot_digest": invocation_digest,
                    },
                )

            batch_count, message_count, byte_count = (
                _chat_memory_replay_snapshot_metrics(snapshot.replay_batches)
            )
            event_seqs = [
                batch.project_event_seq for batch in snapshot.replay_batches
            ]
            if event_seqs != sorted(set(event_seqs)) or any(
                event_seq > snapshot.snapshot_cutoff for event_seq in event_seqs
            ):
                raise MetadataStoreError(
                    "Chat Memory rebuild snapshot batches must be unique and ordered"
                )
            if (
                snapshot.batch_count,
                snapshot.message_count,
                snapshot.byte_count,
            ) != (batch_count, message_count, byte_count):
                raise MetadataStoreError(
                    "Chat Memory rebuild snapshot counts do not match its replay batches"
                )
            persisted_counts = (
                event.snapshot_cutoff,
                event.snapshot_batch_count,
                event.snapshot_message_count,
                event.snapshot_byte_count,
                state.generation.snapshot_cutoff,
                state.generation.replay_batch_count,
                state.generation.replay_message_count,
                state.generation.replay_byte_count,
            )
            expected_counts = (
                snapshot.snapshot_cutoff,
                batch_count,
                message_count,
                byte_count,
                snapshot.snapshot_cutoff,
                batch_count,
                message_count,
                byte_count,
            )
            if persisted_counts != expected_counts:
                raise MetadataConflictError(
                    "chat_memory_rebuild_snapshot",
                    event_id,
                    expected={"persisted_counts": expected_counts},
                    current={"persisted_counts": persisted_counts},
                )
            batches_by_key: dict[tuple[str, int], ChatMemoryReplayBatch] = {}
            for batch in snapshot.replay_batches:
                if not batch.messages:
                    raise MetadataStoreError("Chat Memory replay batches cannot be empty")
                key = (batch.append_batch_id, batch.project_event_seq)
                if key in batches_by_key:
                    raise MetadataStoreError("Duplicate Chat Memory replay batch identity")
                batches_by_key[key] = batch
            mappings_by_key: dict[
                tuple[str, int], ChatMemoryReplayMappingInput
            ] = {}
            episode_uuids: set[str] = set()
            for mapping in replay_mappings:
                key = (mapping.append_batch_id, int(mapping.project_event_seq))
                if key in mappings_by_key or not mapping.episode_uuid:
                    raise MetadataStoreError(
                        "Chat Memory rebuild mappings must have unique identities"
                    )
                if mapping.episode_uuid in episode_uuids:
                    raise MetadataStoreError(
                        "Chat Memory rebuild episode UUIDs must be unique"
                    )
                mappings_by_key[key] = mapping
                episode_uuids.add(mapping.episode_uuid)
            if set(mappings_by_key) != set(batches_by_key):
                raise MetadataStoreError(
                    "Chat Memory rebuild mappings must cover every replay batch exactly once"
                )

            control_time = await conn.fetchval("SELECT clock_timestamp()")
            created_at = str(_iso_timestamp(control_time))
            for batch in snapshot.replay_batches:
                key = (batch.append_batch_id, batch.project_event_seq)
                mapping = mappings_by_key[key]
                session_ids = {message.session_id for message in batch.messages}
                first_seq = min(message.seq for message in batch.messages)
                last_seq = max(message.seq for message in batch.messages)
                if (
                    session_ids != {batch.session_id}
                    or mapping.session_id != batch.session_id
                    or mapping.first_seq != first_seq
                    or mapping.last_seq != last_seq
                ):
                    raise MetadataStoreError(
                        "Chat Memory rebuild mapping does not match its replay batch"
                    )
                await self._insert_postgres_chat_memory_historical_mapping(
                    conn,
                    ChatMemoryEpisodeRecord(
                        episode_uuid=mapping.episode_uuid,
                        session_id=mapping.session_id,
                        project_id=event.project_id,
                        user_id=event.user_id,
                        first_seq=mapping.first_seq,
                        last_seq=mapping.last_seq,
                        created_at=created_at,
                        event_id=event.event_id,
                        generation=event.generation,
                        graph_group_id=event.graph_group_id,
                        append_batch_id=mapping.append_batch_id,
                        project_event_seq=mapping.project_event_seq,
                    ),
                )

            await conn.execute(
                """
                UPDATE enterprise_chat_memory_generations
                SET state = 'purged', cleared_at = COALESCE(cleared_at, $1),
                    updated_at = $1, last_error_code = NULL,
                    last_error_message = NULL, last_error_at = NULL
                WHERE user_id = $2 AND project_id = $3 AND generation <> $4
                """,
                control_time,
                event.user_id,
                event.project_id,
                event.generation,
            )

            activated = await conn.fetchrow(
                """
                UPDATE enterprise_chat_memory_generations
                SET state = 'active', activated_at = $1, updated_at = $1,
                    last_error_code = NULL, last_error_message = NULL,
                    last_error_at = NULL
                WHERE user_id = $2 AND project_id = $3 AND generation = $4
                  AND state = 'building' AND snapshot_cutoff = $5
                  AND replay_batch_count = $6 AND replay_message_count = $7
                  AND replay_byte_count = $8 AND snapshot_digest = $9
                RETURNING *
                """,
                control_time,
                event.user_id,
                event.project_id,
                event.generation,
                snapshot.snapshot_cutoff,
                batch_count,
                message_count,
                byte_count,
                invocation_digest,
            )
            if activated is None:
                raise MetadataConflictError(
                    "chat_memory_generation",
                    event.graph_group_id,
                    expected={"state": "building", "snapshot": expected_counts[4:]},
                    current={
                        "state": state.generation.state,
                        "snapshot": persisted_counts[4:],
                    },
                )

            for batch in snapshot.replay_batches:
                await conn.execute(
                    """
                    UPDATE enterprise_chat_memory_outbox
                    SET status = 'superseded', superseded_by_event_id = $1,
                        completed_at = $2, updated_at = $2
                    WHERE user_id = $3 AND project_id = $4
                      AND event_type = 'ingest'
                      AND status IN ('pending', 'retry_wait', 'dead_letter')
                      AND event_seq = $5 AND event_seq <= $6
                      AND append_batch_id = $7
                    """,
                    event.event_id,
                    control_time,
                    event.user_id,
                    event.project_id,
                    batch.project_event_seq,
                    snapshot.snapshot_cutoff,
                    batch.append_batch_id,
                )

            group_row = await conn.fetchrow(
                """
                UPDATE enterprise_chat_memory_groups
                SET active_generation = $1, active_config_fingerprint = $2,
                    active_graph_store_fingerprint = $3,
                    state = 'active', state_version = state_version + 1,
                    active_rebuild_event_id = NULL, last_success_at = $4,
                    last_error_code = NULL, last_error_message = NULL,
                    last_error_at = NULL, updated_at = $4
                WHERE user_id = $5 AND project_id = $6
                  AND desired_generation = $1 AND desired_config_fingerprint = $2
                  AND desired_graph_store_fingerprint = $3
                  AND state = 'rebuilding' AND state_version = $7
                  AND active_rebuild_event_id = $8
                RETURNING *
                """,
                event.generation,
                fingerprint,
                graph_fingerprint,
                control_time,
                event.user_id,
                event.project_id,
                snapshot.group_state_version,
                event.event_id,
            )
            event_row = await conn.fetchrow(
                """
                UPDATE enterprise_chat_memory_outbox
                SET status = 'succeeded', completed_at = $1, updated_at = $1,
                    last_error_code = NULL, last_error_message = NULL,
                    last_error_at = NULL
                WHERE event_id = $2 AND status = 'running' AND claim_token = $3
                  AND side_effect_started_at IS NOT NULL
                  AND side_effect_state_version = $4 AND snapshot_cutoff = $5
                  AND snapshot_batch_count = $6 AND snapshot_message_count = $7
                  AND snapshot_byte_count = $8 AND snapshot_digest = $9
                RETURNING *
                """,
                control_time,
                event_id,
                claim_token,
                snapshot.group_state_version,
                snapshot.snapshot_cutoff,
                batch_count,
                message_count,
                byte_count,
                invocation_digest,
            )
            if group_row is None or event_row is None:
                raise MetadataConflictError(
                    "chat_memory_rebuild_finalize",
                    event_id,
                    expected={"group": "current", "event": "running/current"},
                    current={
                        "group_state": state.group.state,
                        "event_status": event.status,
                    },
                )
            final = await self._get_postgres_chat_memory_execution_state(
                conn, event_id, for_update=False
            )
            assert final is not None
            return final

        return await self._write(write)

    async def get_chat_memory_purge_targets(
        self,
        event_id: str,
        claim_token: str,
        runtime_fingerprint: str,
        *,
        runtime_graph_store_fingerprint: str | None = None,
    ) -> ChatMemoryPurgeTargetSet | None:
        """Return every durable physical target plus the legacy graph group."""

        fingerprint = _validate_chat_memory_fingerprint(runtime_fingerprint)
        graph_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            fingerprint, runtime_graph_store_fingerprint
        )
        await self._ensure_initialized()

        async def write(conn: Any) -> ChatMemoryPurgeTargetSet | None:
            state = await self._require_postgres_chat_memory_running_claim(
                conn, event_id, claim_token
            )
            await self._assert_postgres_chat_memory_graph_store_invariant(
                conn,
                state.group,
                _resolve_chat_memory_graph_store_fingerprint(
                    state.event.config_fingerprint,
                    state.event.graph_store_fingerprint,
                ),
            )
            stale = await self._resolve_postgres_stale_chat_memory_execution(
                conn, state, claim_token
            )
            if stale is not None:
                return None
            if not self._postgres_chat_memory_runtime_identity_matches(
                state.event, fingerprint, graph_fingerprint
            ):
                await self._retry_postgres_chat_memory_runtime_mismatch(
                    conn,
                    state,
                    claim_token,
                    fingerprint,
                    graph_fingerprint,
                    retry_delay_seconds=1.0,
                )
                return None
            self._validate_postgres_chat_memory_execution_fence(
                state, fingerprint, graph_fingerprint
            )
            event = state.event
            if (
                event.event_type != "purge"
                or state.group.state != "deleting"
                or state.generation.state != "purge_pending"
            ):
                raise MetadataConflictError(
                    "chat_memory_purge",
                    event_id,
                    expected={
                        "event_type": "purge",
                        "group_state": "deleting",
                        "generation_state": "purge_pending",
                    },
                    current={
                        "event_type": event.event_type,
                        "group_state": state.group.state,
                        "generation_state": state.generation.state,
                    },
                )
            group_ids = await self._postgres_chat_memory_purge_group_ids(
                conn,
                event.user_id,
                event.project_id,
            )
            return ChatMemoryPurgeTargetSet(
                event_id=event.event_id,
                user_id=event.user_id,
                project_id=event.project_id,
                logical_group_id=state.group.logical_group_id,
                group_ids=group_ids,
            )

        return await self._write(write)

    async def prepare_chat_memory_purge_targets(
        self,
        event_id: str,
        claim_token: str,
        runtime_fingerprint: str,
        *,
        runtime_graph_store_fingerprint: str | None = None,
    ) -> ChatMemoryPurgeTargetSet | None:
        """Alias emphasizing that target enumeration precedes clear side effects."""

        return await self.get_chat_memory_purge_targets(
            event_id,
            claim_token,
            runtime_fingerprint,
            runtime_graph_store_fingerprint=runtime_graph_store_fingerprint,
        )

    async def finalize_chat_memory_purge(
        self,
        event_id: str,
        claim_token: str,
        runtime_fingerprint: str,
        targets: ChatMemoryPurgeTargetSet | Sequence[str] | None = None,
        definitely_cleared_group_ids: Sequence[str] | None = None,
        *,
        expected_group_ids: Sequence[str] | None = None,
        cleared_group_ids: Sequence[str] | None = None,
        runtime_graph_store_fingerprint: str | None = None,
    ) -> ChatMemoryExecutionState | None:
        """Terminalize a purge only after every expected target definitely cleared."""

        fingerprint = _validate_chat_memory_fingerprint(runtime_fingerprint)
        graph_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            fingerprint, runtime_graph_store_fingerprint
        )
        target_record = targets if isinstance(targets, ChatMemoryPurgeTargetSet) else None
        positional_expected: Sequence[str] | None = (
            targets
            if targets is not None
            and not isinstance(targets, ChatMemoryPurgeTargetSet)
            else None
        )
        if positional_expected is not None and expected_group_ids is not None:
            raise ValueError("Specify purge expected group ids only once")
        if (
            definitely_cleared_group_ids is not None
            and cleared_group_ids is not None
        ):
            raise ValueError("Specify definitely cleared group ids only once")
        caller_expected = (
            target_record.group_ids
            if target_record is not None
            else (
                positional_expected
                if positional_expected is not None
                else expected_group_ids
            )
        )
        normalized_expected = (
            _normalize_chat_memory_group_ids(caller_expected)
            if caller_expected is not None
            else None
        )
        normalized_cleared = _normalize_chat_memory_group_ids(
            definitely_cleared_group_ids
            if definitely_cleared_group_ids is not None
            else (cleared_group_ids or ())
        )
        await self._ensure_initialized()

        async def write(conn: Any) -> ChatMemoryExecutionState | None:
            state = await self._require_postgres_chat_memory_running_claim(
                conn, event_id, claim_token
            )
            await self._assert_postgres_chat_memory_graph_store_invariant(
                conn,
                state.group,
                _resolve_chat_memory_graph_store_fingerprint(
                    state.event.config_fingerprint,
                    state.event.graph_store_fingerprint,
                ),
            )
            stale = await self._resolve_postgres_stale_chat_memory_execution(
                conn, state, claim_token
            )
            if stale is not None:
                return None
            self._validate_postgres_chat_memory_execution_fence(
                state, fingerprint, graph_fingerprint
            )
            event = state.event
            if (
                event.event_type != "purge"
                or state.group.state != "deleting"
                or state.generation.state != "purge_pending"
            ):
                raise MetadataConflictError(
                    "chat_memory_purge",
                    event_id,
                    expected={
                        "event_type": "purge",
                        "group_state": "deleting",
                        "generation_state": "purge_pending",
                    },
                    current={
                        "event_type": event.event_type,
                        "group_state": state.group.state,
                        "generation_state": state.generation.state,
                    },
                )
            if (
                event.side_effect_started_at is None
                or event.side_effect_state_version != state.group.state_version
            ):
                raise MetadataConflictError(
                    "chat_memory_purge",
                    event_id,
                    expected={
                        "side_effect_started_at": "set",
                        "side_effect_state_version": state.group.state_version,
                    },
                    current={
                        "side_effect_started_at": event.side_effect_started_at,
                        "side_effect_state_version": event.side_effect_state_version,
                    },
                )
            if target_record is not None:
                expected_identity = (
                    event.event_id,
                    event.user_id,
                    event.project_id,
                    state.group.logical_group_id,
                )
                current_identity = (
                    target_record.event_id,
                    target_record.user_id,
                    target_record.project_id,
                    target_record.logical_group_id,
                )
                if current_identity != expected_identity:
                    raise MetadataConflictError(
                        "chat_memory_purge_targets",
                        event_id,
                        expected={"identity": expected_identity},
                        current={"identity": current_identity},
                    )

            authoritative_expected = (
                await self._postgres_chat_memory_purge_group_ids(
                    conn,
                    event.user_id,
                    event.project_id,
                )
            )
            if (
                normalized_expected is not None
                and normalized_expected != authoritative_expected
            ):
                raise MetadataConflictError(
                    "chat_memory_purge_targets",
                    event_id,
                    expected={"group_ids": authoritative_expected},
                    current={"group_ids": normalized_expected},
                )
            missing = sorted(
                set(authoritative_expected).difference(normalized_cleared)
            )
            if missing:
                raise MetadataConflictError(
                    "chat_memory_purge_clear",
                    event_id,
                    expected={"definitely_cleared": authoritative_expected},
                    current={
                        "definitely_cleared": normalized_cleared,
                        "missing": tuple(missing),
                    },
                )

            control_time = await conn.fetchval("SELECT clock_timestamp()")
            await conn.execute(
                """
                UPDATE enterprise_chat_memory_generations
                SET state = 'purged', cleared_at = COALESCE(cleared_at, $1),
                    updated_at = $1, last_error_code = NULL,
                    last_error_message = NULL, last_error_at = NULL
                WHERE user_id = $2 AND project_id = $3
                """,
                control_time,
                event.user_id,
                event.project_id,
            )
            await conn.execute(
                """
                DELETE FROM enterprise_chat_memory_episodes
                WHERE user_id = $1 AND project_id = $2
                """,
                event.user_id,
                event.project_id,
            )
            group_row = await conn.fetchrow(
                """
                UPDATE enterprise_chat_memory_groups
                SET state = 'deleted', active_generation = NULL,
                    active_config_fingerprint = NULL,
                    active_graph_store_fingerprint = NULL,
                    active_rebuild_event_id = NULL,
                    state_version = state_version + 1, deleted_at = $1,
                    last_success_at = $1, last_error_code = NULL,
                    last_error_message = NULL, last_error_at = NULL,
                    updated_at = $1
                WHERE user_id = $2 AND project_id = $3 AND state = 'deleting'
                  AND desired_generation = $4
                  AND desired_graph_store_fingerprint = $5 AND state_version = $6
                RETURNING *
                """,
                control_time,
                event.user_id,
                event.project_id,
                event.generation,
                graph_fingerprint,
                event.side_effect_state_version,
            )
            event_row = await conn.fetchrow(
                """
                UPDATE enterprise_chat_memory_outbox
                SET status = 'succeeded', completed_at = $1, updated_at = $1,
                    last_error_code = NULL, last_error_message = NULL,
                    last_error_at = NULL
                WHERE event_id = $2 AND status = 'running' AND claim_token = $3
                  AND side_effect_started_at IS NOT NULL
                  AND side_effect_state_version = $4
                RETURNING *
                """,
                control_time,
                event_id,
                claim_token,
                event.side_effect_state_version,
            )
            if group_row is None or event_row is None:
                raise MetadataConflictError(
                    "chat_memory_purge_finalize",
                    event_id,
                    expected={"group": "current", "event": "running/current"},
                    current={
                        "group_state": state.group.state,
                        "event_status": event.status,
                    },
                )
            final = await self._get_postgres_chat_memory_execution_state(
                conn, event_id, for_update=False
            )
            assert final is not None
            return final

        return await self._write(write)

    async def fail_chat_memory_event_before_side_effect(
        self,
        event_id: str,
        claim_token: str,
        runtime_fingerprint: str,
        *,
        runtime_graph_store_fingerprint: str | None = None,
        error_code: str,
        error_message: str,
        retry_delay_seconds: float | None = 1.0,
        max_attempts: int = 3,
    ) -> ChatMemoryOutboxEventRecord:
        """Retry or dead-letter purge work without leaving deleting state."""

        fingerprint = _validate_chat_memory_fingerprint(runtime_fingerprint)
        graph_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            fingerprint, runtime_graph_store_fingerprint
        )
        await self._ensure_initialized()
        max_attempts = max(1, int(max_attempts))

        async def write(conn: Any) -> ChatMemoryOutboxEventRecord:
            state = await self._require_postgres_chat_memory_running_claim(
                conn, event_id, claim_token
            )
            stale = await self._resolve_postgres_stale_chat_memory_execution(
                conn, state, claim_token
            )
            if stale is not None:
                return stale
            if not self._postgres_chat_memory_runtime_identity_matches(
                state.event, fingerprint, graph_fingerprint
            ):
                return await self._retry_postgres_chat_memory_runtime_mismatch(
                    conn,
                    state,
                    claim_token,
                    fingerprint,
                    graph_fingerprint,
                    retry_delay_seconds=float(retry_delay_seconds or 1.0),
                )
            if state.event.event_type == "purge":
                raise MetadataConflictError(
                    "chat_memory_event",
                    event_id,
                    expected={"event_type": "ingest/rebuild"},
                    current={"event_type": "purge"},
                )
            if state.event.side_effect_started_at is not None:
                raise MetadataConflictError(
                    "chat_memory_event",
                    event_id,
                    expected={"side_effect_started_at": None},
                    current={
                        "side_effect_started_at": state.event.side_effect_started_at
                    },
                )
            dead_letter = (
                retry_delay_seconds is None
                or state.event.attempt_no >= max_attempts
            )
            delay = max(0.0, float(retry_delay_seconds or 0.0))
            status: ChatMemoryEventStatus = (
                "dead_letter" if dead_letter else "retry_wait"
            )
            row = await conn.fetchrow(
                """
                WITH control AS (
                    SELECT clock_timestamp() AS control_time
                )
                UPDATE enterprise_chat_memory_outbox
                SET status = $1,
                    available_at = CASE WHEN $1 = 'dead_letter'
                        THEN control.control_time
                        ELSE control.control_time
                             + ($2::double precision * INTERVAL '1 second') END,
                    claim_token = NULL, claimed_by = NULL, claimed_at = NULL,
                    side_effect_state_version = NULL,
                    completed_at = CASE WHEN $1 = 'dead_letter'
                        THEN control.control_time ELSE NULL END,
                    last_error_code = $3, last_error_message = $4,
                    last_error_at = control.control_time,
                    updated_at = control.control_time
                FROM control
                WHERE event_id = $5 AND status = 'running' AND claim_token = $6
                RETURNING enterprise_chat_memory_outbox.*
                """,
                status,
                delay,
                error_code,
                error_message,
                event_id,
                claim_token,
            )
            if row is None:
                raise MetadataConflictError(
                    "chat_memory_event",
                    event_id,
                    expected={"status": "running", "claim_token": claim_token},
                    current={
                        "status": state.event.status,
                        "claim_token": state.event.claim_token,
                    },
                )
            if dead_letter:
                await conn.execute(
                    """
                    UPDATE enterprise_chat_memory_groups
                    SET state = 'failed', state_version = state_version + 1,
                        last_error_code = $1, last_error_message = $2,
                        last_error_at = clock_timestamp(),
                        updated_at = clock_timestamp()
                    WHERE user_id = $3 AND project_id = $4
                    """,
                    error_code,
                    error_message,
                    state.event.user_id,
                    state.event.project_id,
                )
            return _chat_memory_event_from_row(row)

        return await self._write(write)

    async def supersede_chat_memory_dead_letter_with_rebuild(
        self,
        event_id: str,
        runtime_fingerprint: str,
        *,
        runtime_graph_store_fingerprint: str | None = None,
        actor_user_id: str | None = None,
        actor_tenant_id: str | None = None,
    ) -> ChatMemoryOutboxEventRecord:
        """Persist an unknown clear as same-generation delayed final-sweep work."""

        fingerprint = _validate_chat_memory_fingerprint(runtime_fingerprint)
        graph_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            fingerprint, runtime_graph_store_fingerprint
        )
        await self._ensure_initialized()

        async def write(conn: Any) -> ChatMemoryOutboxEventRecord:
            state = await self._get_postgres_chat_memory_execution_state(
                conn, event_id, for_update=True
            )
            if state is None:
                raise MetadataRecordNotFoundError(
                    f"Chat Memory event '{event_id}' not found"
                )
            if state.event.status != "dead_letter" or state.event.event_type == "purge":
                raise MetadataConflictError(
                    "chat_memory_event",
                    event_id,
                    expected={"status": "dead_letter", "event_type": "ingest/rebuild"},
                    current={
                        "status": state.event.status,
                        "event_type": state.event.event_type,
                    },
                )
            return await self._enqueue_postgres_chat_memory_rebuild(
                conn,
                state.group,
                fingerprint,
                graph_fingerprint,
                actor_user_id=actor_user_id,
                actor_tenant_id=actor_tenant_id,
                target_session_id=state.event.target_session_id,
                target_message_id=state.event.target_message_id,
            )

        return await self._write(write)

    async def fail_chat_memory_purge_before_side_effect(
        self,
        event_id: str,
        claim_token: str,
        runtime_fingerprint: str,
        *,
        runtime_graph_store_fingerprint: str | None = None,
        error_code: str,
        error_message: str,
        retry_delay_seconds: float | None = 5.0,
        max_attempts: int = 3,
    ) -> ChatMemoryOutboxEventRecord:
        fingerprint = _validate_chat_memory_fingerprint(runtime_fingerprint)
        graph_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            fingerprint, runtime_graph_store_fingerprint
        )
        max_attempts = max(1, int(max_attempts))
        await self._ensure_initialized()

        async def write(conn: Any) -> ChatMemoryOutboxEventRecord:
            state = await self._require_postgres_chat_memory_running_claim(
                conn, event_id, claim_token
            )
            stale = await self._resolve_postgres_stale_chat_memory_execution(
                conn, state, claim_token
            )
            if stale is not None:
                return stale
            if state.event.event_type != "purge":
                raise MetadataConflictError(
                    "chat_memory_event",
                    event_id,
                    expected={"event_type": "purge"},
                    current={"event_type": state.event.event_type},
                )
            if not self._postgres_chat_memory_runtime_identity_matches(
                state.event, fingerprint, graph_fingerprint
            ):
                return await self._retry_postgres_chat_memory_runtime_mismatch(
                    conn,
                    state,
                    claim_token,
                    fingerprint,
                    graph_fingerprint,
                    retry_delay_seconds=float(retry_delay_seconds or 5.0),
                )
            if state.event.side_effect_started_at is not None:
                raise MetadataConflictError(
                    "chat_memory_event",
                    event_id,
                    expected={"side_effect_started_at": None},
                    current={
                        "side_effect_started_at": state.event.side_effect_started_at
                    },
                )
            dead_letter = (
                retry_delay_seconds is None
                or state.event.attempt_no >= max_attempts
            )
            status: ChatMemoryEventStatus = (
                "dead_letter" if dead_letter else "retry_wait"
            )
            row = await conn.fetchrow(
                """
                WITH control AS (
                    SELECT clock_timestamp() AS control_time
                )
                UPDATE enterprise_chat_memory_outbox
                SET status = $1,
                    available_at = CASE WHEN $1 = 'dead_letter'
                        THEN control.control_time
                        ELSE control.control_time
                             + ($2::double precision * INTERVAL '1 second') END,
                    claim_token = NULL, claimed_by = NULL, claimed_at = NULL,
                    side_effect_started_at = NULL,
                    side_effect_state_version = NULL,
                    completed_at = CASE WHEN $1 = 'dead_letter'
                        THEN control.control_time ELSE NULL END,
                    last_error_code = $3, last_error_message = $4,
                    last_error_at = control.control_time,
                    updated_at = control.control_time
                FROM control
                WHERE event_id = $5 AND status = 'running' AND claim_token = $6
                RETURNING enterprise_chat_memory_outbox.*
                """,
                status,
                max(0.0, float(retry_delay_seconds or 0.0)),
                error_code,
                error_message,
                event_id,
                claim_token,
            )
            if row is None:
                raise MetadataConflictError(
                    "chat_memory_event",
                    event_id,
                    expected={"status": "running", "claim_token": claim_token},
                    current={
                        "status": state.event.status,
                        "claim_token": state.event.claim_token,
                    },
                )
            await conn.execute(
                """
                UPDATE enterprise_chat_memory_groups
                SET last_error_code = $1, last_error_message = $2,
                    last_error_at = clock_timestamp(),
                    updated_at = clock_timestamp()
                WHERE user_id = $3 AND project_id = $4 AND state = 'deleting'
                """,
                error_code,
                error_message,
                state.event.user_id,
                state.event.project_id,
            )
            return _chat_memory_event_from_row(row)

        return await self._write(write)

    async def retry_chat_memory_purge_after_unknown_clear(
        self,
        event_id: str,
        claim_token: str,
        runtime_fingerprint: str,
        *,
        runtime_graph_store_fingerprint: str | None = None,
        retry_delay_seconds: float = 5.0,
        error_code: str = "purge_clear_outcome_unknown",
        error_message: str = "Purge clear outcome is unknown; final sweep required",
    ) -> ChatMemoryOutboxEventRecord:
        fingerprint = _validate_chat_memory_fingerprint(runtime_fingerprint)
        graph_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            fingerprint, runtime_graph_store_fingerprint
        )
        await self._ensure_initialized()

        async def write(conn: Any) -> ChatMemoryOutboxEventRecord:
            state = await self._require_postgres_chat_memory_running_claim(
                conn, event_id, claim_token
            )
            stale = await self._resolve_postgres_stale_chat_memory_execution(
                conn, state, claim_token
            )
            if stale is not None:
                return stale
            event = state.event
            if event.event_type != "purge":
                raise MetadataConflictError(
                    "chat_memory_event",
                    event_id,
                    expected={"event_type": "purge"},
                    current={"event_type": event.event_type},
                )
            if not self._postgres_chat_memory_runtime_identity_matches(
                event, fingerprint, graph_fingerprint
            ):
                raise MetadataConflictError(
                    "chat_memory_runtime_fingerprint",
                    event_id,
                    expected={
                        "runtime_graph_store_fingerprint": (
                            event.graph_store_fingerprint
                        )
                    },
                    current={
                        "runtime_graph_store_fingerprint": graph_fingerprint
                    },
                )
            if event.side_effect_started_at is None:
                raise MetadataConflictError(
                    "chat_memory_event",
                    event_id,
                    expected={"side_effect_started_at": "set"},
                    current={"side_effect_started_at": None},
                )
            await conn.execute(
                """
                UPDATE enterprise_chat_memory_generations
                SET state = 'purge_pending', cleared_at = NULL,
                    updated_at = clock_timestamp(), last_error_code = $1,
                    last_error_message = $2, last_error_at = clock_timestamp()
                WHERE user_id = $3 AND project_id = $4 AND generation = $5
                """,
                error_code,
                error_message,
                event.user_id,
                event.project_id,
                event.generation,
            )
            row = await conn.fetchrow(
                """
                WITH control AS (
                    SELECT clock_timestamp() AS control_time
                )
                UPDATE enterprise_chat_memory_outbox
                SET status = 'retry_wait',
                    available_at = control.control_time
                        + ($1::double precision * INTERVAL '1 second'),
                    claim_token = NULL, claimed_by = NULL, claimed_at = NULL,
                    side_effect_started_at = NULL,
                    side_effect_state_version = NULL, completed_at = NULL,
                    last_error_code = $2, last_error_message = $3,
                    last_error_at = control.control_time,
                    updated_at = control.control_time
                FROM control
                WHERE event_id = $4 AND status = 'running' AND claim_token = $5
                RETURNING enterprise_chat_memory_outbox.*
                """,
                max(0.0, float(retry_delay_seconds)),
                error_code,
                error_message,
                event_id,
                claim_token,
            )
            if row is None:
                raise MetadataConflictError(
                    "chat_memory_event",
                    event_id,
                    expected={"status": "running", "claim_token": claim_token},
                    current={"status": event.status, "claim_token": event.claim_token},
                )
            await conn.execute(
                """
                UPDATE enterprise_chat_memory_groups
                SET last_error_code = $1, last_error_message = $2,
                    last_error_at = clock_timestamp(),
                    updated_at = clock_timestamp()
                WHERE user_id = $3 AND project_id = $4 AND state = 'deleting'
                """,
                error_code,
                error_message,
                event.user_id,
                event.project_id,
            )
            return _chat_memory_event_from_row(row)

        return await self._write(write)

    async def requeue_chat_memory_purge(
        self,
        event_id: str,
        runtime_fingerprint: str,
        *,
        runtime_graph_store_fingerprint: str | None = None,
        retry_delay_seconds: float = 5.0,
    ) -> ChatMemoryOutboxEventRecord:
        fingerprint = _validate_chat_memory_fingerprint(runtime_fingerprint)
        graph_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            fingerprint, runtime_graph_store_fingerprint
        )
        await self._ensure_initialized()

        async def write(conn: Any) -> ChatMemoryOutboxEventRecord:
            state = await self._get_postgres_chat_memory_execution_state(
                conn, event_id, for_update=True
            )
            if state is None:
                raise MetadataRecordNotFoundError(
                    f"Chat Memory event '{event_id}' not found"
                )
            event = state.event
            if (
                event.event_type != "purge"
                or event.status != "dead_letter"
                or state.group.state != "deleting"
                or state.generation.state != "purge_pending"
                or state.group.desired_generation != event.generation
                or state.group.desired_graph_store_fingerprint != graph_fingerprint
                or state.generation.graph_store_fingerprint != graph_fingerprint
                or event.graph_store_fingerprint != graph_fingerprint
            ):
                raise MetadataConflictError(
                    "chat_memory_purge_requeue",
                    event_id,
                    expected={
                        "event_type": "purge",
                        "status": "dead_letter",
                        "group_state": "deleting",
                        "generation_state": "purge_pending",
                        "runtime_graph_store_fingerprint": graph_fingerprint,
                    },
                    current={
                        "event_type": event.event_type,
                        "status": event.status,
                        "group_state": state.group.state,
                        "generation_state": state.generation.state,
                        "event_graph_store_fingerprint": (
                            event.graph_store_fingerprint
                        ),
                    },
                )
            row = await conn.fetchrow(
                """
                WITH control AS (
                    SELECT clock_timestamp() AS control_time
                )
                UPDATE enterprise_chat_memory_outbox
                SET status = 'retry_wait',
                    available_at = control.control_time
                        + ($1::double precision * INTERVAL '1 second'),
                    claim_token = NULL, claimed_by = NULL, claimed_at = NULL,
                    side_effect_started_at = NULL,
                    side_effect_state_version = NULL, completed_at = NULL,
                    last_error_code = NULL, last_error_message = NULL,
                    last_error_at = NULL, updated_at = control.control_time
                FROM control
                WHERE event_id = $2 AND status = 'dead_letter'
                RETURNING enterprise_chat_memory_outbox.*
                """,
                max(0.0, float(retry_delay_seconds)),
                event_id,
            )
            if row is None:
                raise MetadataConflictError(
                    "chat_memory_purge_requeue",
                    event_id,
                    expected={"status": "dead_letter"},
                    current={"status": event.status},
                )
            return _chat_memory_event_from_row(row)

        return await self._write(write)

    async def escalate_chat_memory_event_unknown(
        self,
        event_id: str,
        claim_token: str,
        runtime_fingerprint: str,
        *,
        runtime_graph_store_fingerprint: str | None = None,
        error_code: str = "side_effect_outcome_unknown",
        error_message: str = "Graph side-effect outcome is unknown",
        actor_user_id: str | None = None,
        actor_tenant_id: str | None = None,
    ) -> ChatMemoryOutboxEventRecord:
        fingerprint = _validate_chat_memory_fingerprint(runtime_fingerprint)
        graph_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            fingerprint, runtime_graph_store_fingerprint
        )
        await self._ensure_initialized()

        async def write(conn: Any) -> ChatMemoryOutboxEventRecord:
            state = await self._require_postgres_chat_memory_running_claim(
                conn, event_id, claim_token
            )
            stale = await self._resolve_postgres_stale_chat_memory_execution(
                conn, state, claim_token
            )
            if stale is not None:
                return stale
            event = state.event
            if event.side_effect_started_at is None:
                raise MetadataConflictError(
                    "chat_memory_event",
                    event_id,
                    expected={"side_effect_started_at": "set"},
                    current={"side_effect_started_at": None},
                )
            if event.event_type == "purge":
                raise MetadataConflictError(
                    "chat_memory_event",
                    event_id,
                    expected={"event_type": "ingest/rebuild"},
                    current={"event_type": event.event_type},
                )
            self._validate_postgres_chat_memory_execution_fence(
                state, fingerprint, graph_fingerprint
            )
            control_time = await conn.fetchval("SELECT clock_timestamp()")
            await conn.execute(
                """
                UPDATE enterprise_chat_memory_generations
                SET state = 'abandoned', updated_at = $1,
                    last_error_code = $2, last_error_message = $3,
                    last_error_at = $1
                WHERE user_id = $4 AND project_id = $5 AND generation = $6
                """,
                control_time,
                error_code,
                error_message,
                event.user_id,
                event.project_id,
                event.generation,
            )
            if state.group.active_generation == event.generation:
                await conn.execute(
                    """
                    UPDATE enterprise_chat_memory_groups
                    SET active_generation = NULL,
                        active_config_fingerprint = NULL,
                        active_graph_store_fingerprint = NULL
                    WHERE user_id = $1 AND project_id = $2
                    """,
                    event.user_id,
                    event.project_id,
                )
            rebuild = await self._enqueue_postgres_chat_memory_rebuild(
                conn,
                state.group,
                fingerprint,
                graph_fingerprint,
                actor_user_id=actor_user_id,
                actor_tenant_id=actor_tenant_id,
                target_session_id=event.target_session_id,
                target_message_id=event.target_message_id,
            )
            row = await conn.fetchrow(
                """
                UPDATE enterprise_chat_memory_outbox
                SET status = 'superseded', superseded_by_event_id = $1,
                    completed_at = $2, last_error_code = $3,
                    last_error_message = $4, last_error_at = $2,
                    updated_at = $2
                WHERE event_id = $5 AND status = 'running' AND claim_token = $6
                RETURNING *
                """,
                rebuild.event_id,
                control_time,
                error_code,
                error_message,
                event_id,
                claim_token,
            )
            if row is None:
                raise MetadataConflictError(
                    "chat_memory_event",
                    event_id,
                    expected={"status": "running", "claim_token": claim_token},
                    current={"status": event.status, "claim_token": event.claim_token},
                )
            return rebuild

        return await self._write(write)

    async def list_stale_chat_memory_running_events(
        self, *, stale_after_seconds: float, limit: int = 100
    ) -> list[ChatMemoryOutboxEventRecord]:
        await self._ensure_initialized()
        limit = max(1, min(int(limit), 1000))
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM enterprise_chat_memory_outbox
                WHERE status = 'running' AND claimed_at IS NOT NULL
                  AND claimed_at <= clock_timestamp()
                      - ($1::double precision * INTERVAL '1 second')
                ORDER BY claimed_at, user_id, project_id, event_seq
                LIMIT $2
                """,
                max(0.0, float(stale_after_seconds)),
                limit,
            )
        return [_chat_memory_event_from_row(row) for row in rows]

    async def recover_stale_chat_memory_event(
        self,
        event_id: str,
        claim_token: str,
        runtime_fingerprint: str,
        *,
        runtime_graph_store_fingerprint: str | None = None,
        retry_delay_seconds: float = 1.0,
        max_attempts: int = 3,
        error_code: str = "stale_worker_claim",
        error_message: str = "Worker ownership ended before finalization",
    ) -> ChatMemoryOutboxEventRecord | None:
        """Try group ownership once; return None without mutation if unavailable."""

        event = await self.get_chat_memory_event(event_id)
        if event is None:
            raise MetadataRecordNotFoundError(
                f"Chat Memory event '{event_id}' not found"
            )
        logical_group_id = chat_memory_logical_group_id(
            event.user_id, event.project_id
        )
        async with self.chat_memory_group_execution_guard(
            logical_group_id, wait=False
        ) as acquired:
            if not acquired:
                return None
            event = await self.get_chat_memory_event(event_id)
            if event is None:
                raise MetadataRecordNotFoundError(
                    f"Chat Memory event '{event_id}' not found"
                )
            if event.event_type == "purge":
                if event.side_effect_started_at is None:
                    return await self.fail_chat_memory_purge_before_side_effect(
                        event_id,
                        claim_token,
                        runtime_fingerprint,
                        runtime_graph_store_fingerprint=(
                            runtime_graph_store_fingerprint
                        ),
                        error_code=error_code,
                        error_message=error_message,
                        retry_delay_seconds=retry_delay_seconds,
                        max_attempts=max_attempts,
                    )
                return await self.retry_chat_memory_purge_after_unknown_clear(
                    event_id,
                    claim_token,
                    runtime_fingerprint,
                    runtime_graph_store_fingerprint=(
                        runtime_graph_store_fingerprint
                    ),
                    retry_delay_seconds=retry_delay_seconds,
                    error_code=error_code,
                    error_message=error_message,
                )
            if event.side_effect_started_at is None:
                return await self.fail_chat_memory_event_before_side_effect(
                    event_id,
                    claim_token,
                    runtime_fingerprint,
                    runtime_graph_store_fingerprint=(
                        runtime_graph_store_fingerprint
                    ),
                    error_code=error_code,
                    error_message=error_message,
                    retry_delay_seconds=retry_delay_seconds,
                    max_attempts=max_attempts,
                )
            return await self.escalate_chat_memory_event_unknown(
                event_id,
                claim_token,
                runtime_fingerprint,
                runtime_graph_store_fingerprint=runtime_graph_store_fingerprint,
                error_code=error_code,
                error_message=error_message,
            )

    async def get_chat_memory_outbox_stats(self) -> ChatMemoryOutboxStats:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) FILTER (WHERE status = 'pending') AS pending,
                    COUNT(*) FILTER (WHERE status = 'running') AS running,
                    COUNT(*) FILTER (WHERE status = 'retry_wait') AS retry_wait,
                    COUNT(*) FILTER (WHERE status = 'dead_letter') AS dead_letter,
                    MIN(available_at) FILTER (WHERE status IN (
                        'pending', 'running', 'retry_wait', 'dead_letter'
                    )) AS oldest_available_at,
                    COALESCE(GREATEST(0, EXTRACT(EPOCH FROM (
                        clock_timestamp() - MIN(available_at) FILTER (
                            WHERE status IN (
                                'pending', 'running', 'retry_wait', 'dead_letter'
                            )
                        )
                    ))), 0) AS oldest_lag_seconds
                FROM enterprise_chat_memory_outbox
                """
            )
        assert row is not None
        return ChatMemoryOutboxStats(
            pending=int(row["pending"] or 0),
            running=int(row["running"] or 0),
            retry_wait=int(row["retry_wait"] or 0),
            dead_letter=int(row["dead_letter"] or 0),
            oldest_available_at=_iso_timestamp(row["oldest_available_at"]),
            oldest_lag_seconds=float(row["oldest_lag_seconds"] or 0.0),
        )

    async def _get_postgres_chat_memory_execution_state(
        self, conn: Any, event_id: str, *, for_update: bool
    ) -> ChatMemoryExecutionState | None:
        hint = await conn.fetchrow(
            """
            SELECT user_id, project_id FROM enterprise_chat_memory_outbox
            WHERE event_id = $1
            """,
            event_id,
        )
        if hint is None:
            return None
        group = await self._get_postgres_chat_memory_group(
            conn,
            str(hint["user_id"]),
            str(hint["project_id"]),
            for_update=for_update,
        )
        event_suffix = " FOR UPDATE" if for_update else ""
        event_row = await conn.fetchrow(
            f"""
            SELECT * FROM enterprise_chat_memory_outbox
            WHERE event_id = $1{event_suffix}
            """,
            event_id,
        )
        if event_row is None:
            return None
        event = _chat_memory_event_from_row(event_row)
        generation_suffix = " FOR UPDATE" if for_update else ""
        generation_row = await conn.fetchrow(
            f"""
            SELECT * FROM enterprise_chat_memory_generations
            WHERE user_id = $1 AND project_id = $2 AND generation = $3
            {generation_suffix}
            """,
            event.user_id,
            event.project_id,
            event.generation,
        )
        if group is None or generation_row is None:
            raise MetadataStoreError(
                f"Chat Memory execution inventory missing for event '{event_id}'"
            )
        return ChatMemoryExecutionState(
            group=group,
            event=event,
            generation=_chat_memory_generation_from_row(generation_row),
        )

    async def _require_postgres_chat_memory_running_claim(
        self, conn: Any, event_id: str, claim_token: str
    ) -> ChatMemoryExecutionState:
        state = await self._get_postgres_chat_memory_execution_state(
            conn, event_id, for_update=True
        )
        if state is None:
            raise MetadataRecordNotFoundError(
                f"Chat Memory event '{event_id}' not found"
            )
        if state.event.status != "running" or state.event.claim_token != claim_token:
            raise MetadataConflictError(
                "chat_memory_event",
                event_id,
                expected={"status": "running", "claim_token": claim_token},
                current={
                    "status": state.event.status,
                    "claim_token": state.event.claim_token,
                },
            )
        return state

    def _postgres_chat_memory_stale_execution_reason(
        self, state: ChatMemoryExecutionState
    ) -> str | None:
        event = state.event
        group = state.group
        generation = state.generation
        if group.desired_generation != event.generation:
            return "desired_generation_advanced"
        if generation.graph_group_id != event.graph_group_id:
            return "physical_generation_changed"
        if event.graph_store_fingerprint != group.desired_graph_store_fingerprint:
            return "desired_graph_store_fingerprint_changed"
        if event.graph_store_fingerprint != generation.graph_store_fingerprint:
            return "generation_graph_store_fingerprint_changed"
        if (
            event.event_type != "purge"
            and event.config_fingerprint != group.desired_config_fingerprint
        ):
            return "desired_fingerprint_changed"
        if (
            event.event_type != "purge"
            and event.config_fingerprint != generation.config_fingerprint
        ):
            return "generation_fingerprint_changed"
        allowed = {
            "ingest": (
                {"active", "rebuilding"},
                {"active", "building"},
            ),
            "rebuild": ({"rebuilding"}, {"building"}),
            "purge": ({"deleting"}, {"purge_pending"}),
        }
        group_states, generation_states = allowed[event.event_type]
        if group.state not in group_states:
            return f"group_state_{group.state}"
        if generation.state not in generation_states:
            return f"generation_state_{generation.state}"
        if event.side_effect_started_at is not None and (
            event.side_effect_state_version is None
            or event.side_effect_state_version != group.state_version
        ):
            return "side_effect_state_version_changed"
        return None

    @staticmethod
    def _postgres_chat_memory_runtime_identity_matches(
        event: ChatMemoryOutboxEventRecord,
        runtime_fingerprint: str,
        runtime_graph_store_fingerprint: str,
    ) -> bool:
        return (
            event.graph_store_fingerprint == runtime_graph_store_fingerprint
            and (
                event.event_type == "purge"
                or event.config_fingerprint == runtime_fingerprint
            )
        )

    async def _resolve_postgres_stale_chat_memory_execution(
        self,
        conn: Any,
        state: ChatMemoryExecutionState,
        claim_token: str,
    ) -> ChatMemoryOutboxEventRecord | None:
        reason = self._postgres_chat_memory_stale_execution_reason(state)
        if reason is None:
            return None
        takeover_id = await conn.fetchval(
            """
            SELECT event_id FROM enterprise_chat_memory_outbox
            WHERE user_id = $1 AND project_id = $2 AND event_seq > $3
              AND event_type IN ('rebuild', 'purge')
              AND status <> 'superseded'
            ORDER BY event_seq ASC LIMIT 1
            """,
            state.event.user_id,
            state.event.project_id,
            state.event.event_seq,
        )
        row = await conn.fetchrow(
            """
            WITH control AS (
                SELECT clock_timestamp() AS control_time
            )
            UPDATE enterprise_chat_memory_outbox
            SET status = 'superseded', superseded_by_event_id = $1,
                completed_at = control.control_time,
                last_error_code = 'stale_execution_fence',
                last_error_message = $2, last_error_at = control.control_time,
                updated_at = control.control_time
            FROM control
            WHERE event_id = $3 AND status = 'running' AND claim_token = $4
            RETURNING enterprise_chat_memory_outbox.*
            """,
            takeover_id,
            reason,
            state.event.event_id,
            claim_token,
        )
        if row is None:
            raise MetadataConflictError(
                "chat_memory_event",
                state.event.event_id,
                expected={"status": "running", "claim_token": claim_token},
                current={
                    "status": state.event.status,
                    "claim_token": state.event.claim_token,
                },
            )
        return _chat_memory_event_from_row(row)

    async def _retry_postgres_chat_memory_runtime_mismatch(
        self,
        conn: Any,
        state: ChatMemoryExecutionState,
        claim_token: str,
        runtime_fingerprint: str,
        runtime_graph_store_fingerprint: str,
        *,
        retry_delay_seconds: float,
    ) -> ChatMemoryOutboxEventRecord:
        if state.event.side_effect_started_at is not None:
            raise MetadataConflictError(
                "chat_memory_runtime_fingerprint",
                state.event.event_id,
                expected={
                    "runtime_fingerprint": state.event.config_fingerprint,
                    "runtime_graph_store_fingerprint": (
                        state.event.graph_store_fingerprint
                    ),
                },
                current={
                    "runtime_fingerprint": runtime_fingerprint,
                    "runtime_graph_store_fingerprint": (
                        runtime_graph_store_fingerprint
                    ),
                },
            )
        row = await conn.fetchrow(
            """
            WITH control AS (
                SELECT clock_timestamp() AS control_time
            )
            UPDATE enterprise_chat_memory_outbox
            SET status = 'retry_wait',
                available_at = control.control_time
                    + ($1::double precision * INTERVAL '1 second'),
                claim_token = NULL, claimed_by = NULL, claimed_at = NULL,
                side_effect_started_at = NULL,
                side_effect_state_version = NULL,
                last_error_code = 'runtime_fingerprint_mismatch',
                last_error_message = $2,
                last_error_at = control.control_time,
                updated_at = control.control_time
            FROM control
            WHERE event_id = $3 AND status = 'running' AND claim_token = $4
            RETURNING enterprise_chat_memory_outbox.*
            """,
            max(0.0, float(retry_delay_seconds)),
            (
                "expected runtime identity "
                f"({state.event.config_fingerprint}, "
                f"{state.event.graph_store_fingerprint}), got "
                f"({runtime_fingerprint}, {runtime_graph_store_fingerprint})"
            ),
            state.event.event_id,
            claim_token,
        )
        if row is None:
            raise MetadataConflictError(
                "chat_memory_event",
                state.event.event_id,
                expected={"status": "running", "claim_token": claim_token},
                current={
                    "status": state.event.status,
                    "claim_token": state.event.claim_token,
                },
            )
        return _chat_memory_event_from_row(row)

    def _validate_postgres_chat_memory_execution_fence(
        self,
        state: ChatMemoryExecutionState,
        runtime_fingerprint: str,
        runtime_graph_store_fingerprint: str,
    ) -> None:
        event = state.event
        group = state.group
        generation = state.generation
        expected = {
            "desired_generation": event.generation,
            "event_fingerprint": runtime_fingerprint,
            "generation_fingerprint": runtime_fingerprint,
            "desired_fingerprint": runtime_fingerprint,
            "event_graph_store_fingerprint": runtime_graph_store_fingerprint,
            "generation_graph_store_fingerprint": (
                runtime_graph_store_fingerprint
            ),
            "desired_graph_store_fingerprint": runtime_graph_store_fingerprint,
            "graph_group_id": event.graph_group_id,
        }
        current = {
            "desired_generation": group.desired_generation,
            "event_fingerprint": event.config_fingerprint,
            "generation_fingerprint": generation.config_fingerprint,
            "desired_fingerprint": group.desired_config_fingerprint,
            "event_graph_store_fingerprint": event.graph_store_fingerprint,
            "generation_graph_store_fingerprint": (
                generation.graph_store_fingerprint
            ),
            "desired_graph_store_fingerprint": (
                group.desired_graph_store_fingerprint
            ),
            "graph_group_id": generation.graph_group_id,
        }
        extraction_fingerprints_match = (
            event.event_type == "purge"
            or (
                event.config_fingerprint == runtime_fingerprint
                and generation.config_fingerprint == runtime_fingerprint
                and group.desired_config_fingerprint == runtime_fingerprint
            )
        )
        if (
            group.desired_generation != event.generation
            or not extraction_fingerprints_match
            or event.graph_store_fingerprint != runtime_graph_store_fingerprint
            or generation.graph_store_fingerprint
            != runtime_graph_store_fingerprint
            or group.desired_graph_store_fingerprint
            != runtime_graph_store_fingerprint
            or generation.graph_group_id != event.graph_group_id
        ):
            raise MetadataConflictError(
                "chat_memory_execution_fence",
                event.event_id,
                expected=expected,
                current=current,
            )

    def _validate_postgres_chat_memory_ingest_execution(
        self,
        state: ChatMemoryExecutionState,
        runtime_fingerprint: str,
        runtime_graph_store_fingerprint: str,
    ) -> None:
        self._validate_postgres_chat_memory_execution_fence(
            state, runtime_fingerprint, runtime_graph_store_fingerprint
        )
        if state.event.event_type != "ingest":
            raise MetadataConflictError(
                "chat_memory_event",
                state.event.event_id,
                expected={"event_type": "ingest"},
                current={"event_type": state.event.event_type},
            )
        if state.group.state in {"deleting", "deleted", "failed"}:
            raise MetadataConflictError(
                "chat_memory_group",
                state.group.logical_group_id,
                expected={"state": "active/rebuilding"},
                current={"state": state.group.state},
            )
        if state.generation.state not in {"building", "active"}:
            raise MetadataConflictError(
                "chat_memory_generation",
                state.generation.graph_group_id,
                expected={"state": "building/active"},
                current={"state": state.generation.state},
            )

    async def _insert_postgres_chat_memory_historical_mapping(
        self, conn: Any, expected: ChatMemoryEpisodeRecord
    ) -> None:
        def same_payload(row: Any) -> bool:
            current = _chat_memory_episode_from_row(row)
            return all(
                getattr(current, field) == getattr(expected, field)
                for field in (
                    "episode_uuid",
                    "session_id",
                    "project_id",
                    "user_id",
                    "first_seq",
                    "last_seq",
                    "event_id",
                    "generation",
                    "graph_group_id",
                    "append_batch_id",
                    "project_event_seq",
                )
            )

        by_uuid = await conn.fetchrow(
            """
            SELECT * FROM enterprise_chat_memory_episodes
            WHERE episode_uuid = $1 FOR UPDATE
            """,
            expected.episode_uuid,
        )
        by_identity = await conn.fetchrow(
            """
            SELECT * FROM enterprise_chat_memory_episodes
            WHERE user_id = $1 AND project_id = $2 AND generation = $3
              AND append_batch_id = $4
            FOR UPDATE
            """,
            expected.user_id,
            expected.project_id,
            expected.generation,
            expected.append_batch_id,
        )
        for row in (by_uuid, by_identity):
            if row is not None and not same_payload(row):
                raise MetadataConflictError(
                    "chat_memory_episode_mapping",
                    expected.episode_uuid,
                    expected=expected.to_dict(),
                    current=_chat_memory_episode_from_row(row).to_dict(),
                )
        if by_uuid is not None or by_identity is not None:
            return
        await conn.execute(
            """
            INSERT INTO enterprise_chat_memory_episodes (
                episode_uuid, session_id, project_id, user_id, first_seq,
                last_seq, created_at, event_id, generation, graph_group_id,
                append_batch_id, project_event_seq
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """,
            expected.episode_uuid,
            expected.session_id,
            expected.project_id,
            expected.user_id,
            expected.first_seq,
            expected.last_seq,
            expected.created_at,
            expected.event_id,
            expected.generation,
            expected.graph_group_id,
            expected.append_batch_id,
            expected.project_event_seq,
        )

    async def _postgres_chat_memory_graph_store_fingerprints(
        self,
        conn: Any,
        user_id: str,
        project_id: str,
    ) -> tuple[str, ...]:
        """Return every non-empty graph-store identity attached to a group."""

        rows = await conn.fetch(
            """
            SELECT graph_store_fingerprint FROM (
                SELECT active_graph_store_fingerprint AS graph_store_fingerprint
                FROM enterprise_chat_memory_groups
                WHERE user_id = $1 AND project_id = $2
                UNION ALL
                SELECT desired_graph_store_fingerprint
                FROM enterprise_chat_memory_groups
                WHERE user_id = $1 AND project_id = $2
                UNION ALL
                SELECT graph_store_fingerprint
                FROM enterprise_chat_memory_generations
                WHERE user_id = $1 AND project_id = $2
                UNION ALL
                SELECT graph_store_fingerprint
                FROM enterprise_chat_memory_outbox
                WHERE user_id = $1 AND project_id = $2
                UNION ALL
                SELECT generation.graph_store_fingerprint
                FROM enterprise_chat_memory_episodes AS mapping
                JOIN enterprise_chat_memory_generations AS generation
                  ON generation.user_id = mapping.user_id
                 AND generation.project_id = mapping.project_id
                 AND generation.generation = mapping.generation
                WHERE mapping.user_id = $1 AND mapping.project_id = $2
                UNION ALL
                SELECT event.graph_store_fingerprint
                FROM enterprise_chat_memory_episodes AS mapping
                JOIN enterprise_chat_memory_outbox AS event
                  ON event.event_id = mapping.event_id
                WHERE mapping.user_id = $1 AND mapping.project_id = $2
            ) AS identities
            WHERE graph_store_fingerprint IS NOT NULL
              AND btrim(graph_store_fingerprint) <> ''
            """,
            user_id,
            project_id,
        )
        return tuple(
            sorted({str(row["graph_store_fingerprint"]) for row in rows})
        )

    async def _assert_postgres_chat_memory_graph_store_invariant(
        self,
        conn: Any,
        group: ChatMemoryGroupRecord,
        required_graph_store_fingerprint: str,
    ) -> None:
        required = _validate_chat_memory_fingerprint(
            required_graph_store_fingerprint
        )
        observed = await self._postgres_chat_memory_graph_store_fingerprints(
            conn, group.user_id, group.project_id
        )
        if observed != (required,):
            raise _chat_memory_graph_store_migration_conflict(
                group.logical_group_id,
                required,
                observed,
            )

    async def _get_postgres_chat_memory_group(
        self,
        conn: Any,
        user_id: str,
        project_id: str,
        *,
        for_update: bool,
    ) -> ChatMemoryGroupRecord | None:
        suffix = " FOR UPDATE" if for_update else ""
        row = await conn.fetchrow(
            f"""
            SELECT * FROM enterprise_chat_memory_groups
            WHERE user_id = $1 AND project_id = $2{suffix}
            """,
            user_id,
            project_id,
        )
        return _chat_memory_group_from_row(row) if row is not None else None

    async def _lock_postgres_chat_memory_logical_group_transaction(
        self,
        conn: Any,
        user_id: str,
        project_id: str,
    ) -> None:
        """Serialize first public enqueue before a logical-group row exists."""

        await conn.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 1263295565))",
            chat_memory_logical_group_id(user_id, project_id),
        )

    async def _ensure_postgres_chat_memory_group(
        self,
        conn: Any,
        user_id: str,
        project_id: str,
        config_fingerprint: str,
        graph_store_fingerprint: str,
        *,
        generation_state: ChatMemoryGenerationState,
    ) -> tuple[ChatMemoryGroupRecord, bool]:
        await self._lock_postgres_chat_memory_logical_group_transaction(
            conn, user_id, project_id
        )
        group = await self._get_postgres_chat_memory_group(
            conn, user_id, project_id, for_update=True
        )
        if group is not None:
            return group, False
        await conn.execute(
            """
            INSERT INTO enterprise_chat_memory_groups (
                user_id, project_id, logical_group_id, active_generation,
                desired_generation, next_event_seq, last_reference_time, state,
                state_version, active_config_fingerprint,
                desired_config_fingerprint, active_graph_store_fingerprint,
                desired_graph_store_fingerprint, active_rebuild_event_id,
                last_success_at, last_error_code, last_error_message,
                last_error_at, created_at, updated_at, deleted_at, record_version
            )
            SELECT $1, $2, $3, NULL, 1, 1, NULL, $4, 1, NULL, $5, NULL, $6,
                   NULL, NULL, NULL, NULL, NULL, control.control_time,
                   control.control_time, NULL, $7
            FROM (SELECT clock_timestamp() AS control_time) AS control
            """,
            user_id,
            project_id,
            chat_memory_logical_group_id(user_id, project_id),
            "deleting" if generation_state == "purge_pending" else "rebuilding",
            config_fingerprint,
            graph_store_fingerprint,
            CHAT_MEMORY_RECORD_VERSION,
        )
        await self._insert_postgres_chat_memory_generation(
            conn,
            user_id=user_id,
            project_id=project_id,
            generation=1,
            config_fingerprint=config_fingerprint,
            graph_store_fingerprint=graph_store_fingerprint,
            state=generation_state,
        )
        group = await self._get_postgres_chat_memory_group(
            conn, user_id, project_id, for_update=True
        )
        assert group is not None
        return group, True

    async def _insert_postgres_chat_memory_generation(
        self,
        conn: Any,
        *,
        user_id: str,
        project_id: str,
        generation: int,
        config_fingerprint: str,
        graph_store_fingerprint: str,
        state: ChatMemoryGenerationState,
    ) -> None:
        await conn.execute(
            """
            INSERT INTO enterprise_chat_memory_generations (
                user_id, project_id, generation, graph_group_id,
                config_fingerprint, graph_store_fingerprint, state, snapshot_cutoff,
                replay_batch_count, replay_message_count, replay_byte_count,
                snapshot_digest, clear_attempt_no, clear_started_at, created_at,
                updated_at, activated_at, cleared_at, last_error_code,
                last_error_message, last_error_at, record_version
            )
            SELECT $1, $2, $3, $4, $5, $6, $7, NULL, NULL, NULL, NULL, NULL, 0, NULL,
                   control.control_time, control.control_time, NULL, NULL,
                   NULL, NULL, NULL, $8
            FROM (SELECT clock_timestamp() AS control_time) AS control
            """,
            user_id,
            project_id,
            int(generation),
            chat_memory_graph_group_id(user_id, project_id, generation),
            config_fingerprint,
            graph_store_fingerprint,
            state,
            CHAT_MEMORY_RECORD_VERSION,
        )

    async def _allocate_postgres_chat_memory_event_seq(
        self,
        conn: Any,
        user_id: str,
        project_id: str,
        *,
        allocate_reference_time: bool,
    ) -> tuple[int, Any | None]:
        if allocate_reference_time:
            row = await conn.fetchrow(
                """
                WITH control AS (
                    SELECT clock_timestamp() AS control_time
                )
                UPDATE enterprise_chat_memory_groups
                SET next_event_seq = next_event_seq + 1,
                    last_reference_time = CASE
                        WHEN last_reference_time IS NULL
                            THEN control.control_time
                        ELSE GREATEST(
                            control.control_time,
                            last_reference_time + INTERVAL '1 microsecond'
                        )
                    END,
                    updated_at = CASE
                        WHEN last_reference_time IS NULL
                            THEN control.control_time
                        ELSE GREATEST(
                            control.control_time,
                            last_reference_time + INTERVAL '1 microsecond'
                        )
                    END
                FROM control
                WHERE user_id = $1 AND project_id = $2
                RETURNING next_event_seq - 1 AS event_seq, last_reference_time
                """,
                user_id,
                project_id,
            )
        else:
            row = await conn.fetchrow(
                """
                WITH control AS (
                    SELECT clock_timestamp() AS control_time
                )
                UPDATE enterprise_chat_memory_groups
                SET next_event_seq = next_event_seq + 1,
                    updated_at = control.control_time
                FROM control
                WHERE user_id = $1 AND project_id = $2
                RETURNING next_event_seq - 1 AS event_seq, NULL::timestamptz
                    AS last_reference_time
                """,
                user_id,
                project_id,
            )
        if row is None:
            raise MetadataRecordNotFoundError("Chat Memory group not found")
        return int(row["event_seq"]), row["last_reference_time"]

    async def _insert_postgres_chat_memory_event(
        self, conn: Any, event: ChatMemoryOutboxEventRecord
    ) -> ChatMemoryOutboxEventRecord:
        row = await conn.fetchrow(
            """
            INSERT INTO enterprise_chat_memory_outbox (
                event_id, deterministic_key, user_id, project_id, event_seq,
                generation, graph_group_id, config_fingerprint,
                graph_store_fingerprint, event_type,
                status, available_at, attempt_no, source_session_id,
                append_batch_id, first_seq, last_seq, snapshot_cutoff,
                snapshot_batch_count, snapshot_message_count,
                snapshot_byte_count, snapshot_digest, claim_token, claimed_by,
                claimed_at, side_effect_started_at, side_effect_state_version, completed_at,
                superseded_by_event_id, last_error_code, last_error_message,
                last_error_at, actor_user_id, actor_tenant_id, target_user_id,
                target_project_id, target_session_id, target_message_id,
                created_at, updated_at, record_version
            )
            SELECT $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                   control.control_time, $12, $13, $14, $15, $16, $17,
                   $18, $19, $20, $21, $22, $23,
                   ($24::text)::timestamptz, ($25::text)::timestamptz, $26,
                   ($27::text)::timestamptz, $28, $29, $30,
                   ($31::text)::timestamptz, $32, $33, $34, $35, $36, $37,
                   control.control_time, control.control_time, $38
            FROM (SELECT clock_timestamp() AS control_time) AS control
            RETURNING available_at, created_at, updated_at
            """,
            event.event_id,
            event.deterministic_key,
            event.user_id,
            event.project_id,
            event.event_seq,
            event.generation,
            event.graph_group_id,
            event.config_fingerprint,
            event.graph_store_fingerprint,
            event.event_type,
            event.status,
            event.attempt_no,
            event.source_session_id,
            event.append_batch_id,
            event.first_seq,
            event.last_seq,
            event.snapshot_cutoff,
            event.snapshot_batch_count,
            event.snapshot_message_count,
            event.snapshot_byte_count,
            event.snapshot_digest,
            event.claim_token,
            event.claimed_by,
            event.claimed_at,
            event.side_effect_started_at,
            event.side_effect_state_version,
            event.completed_at,
            event.superseded_by_event_id,
            event.last_error_code,
            event.last_error_message,
            event.last_error_at,
            event.actor_user_id,
            event.actor_tenant_id,
            event.target_user_id,
            event.target_project_id,
            event.target_session_id,
            event.target_message_id,
            event.record_version,
        )
        assert row is not None
        event.available_at = str(_iso_timestamp(row["available_at"]))
        event.created_at = str(_iso_timestamp(row["created_at"]))
        event.updated_at = str(_iso_timestamp(row["updated_at"]))
        return event

    async def _enqueue_postgres_chat_memory_rebuild(
        self,
        conn: Any,
        group: ChatMemoryGroupRecord,
        config_fingerprint: str,
        graph_store_fingerprint: str | None = None,
        *,
        actor_user_id: str | None,
        actor_tenant_id: str | None,
        target_session_id: str | None,
        target_message_id: str | None,
    ) -> ChatMemoryOutboxEventRecord:
        graph_store_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            config_fingerprint, graph_store_fingerprint
        )
        await self._assert_postgres_chat_memory_graph_store_invariant(
            conn, group, graph_store_fingerprint
        )
        generation_row = await conn.fetchrow(
            """
            SELECT generation FROM enterprise_chat_memory_generations
            WHERE user_id = $1 AND project_id = $2 AND generation = $3
            FOR UPDATE
            """,
            group.user_id,
            group.project_id,
            group.desired_generation,
        )
        is_new_group = group.next_event_seq == 1 and generation_row is not None
        if is_new_group and group.active_generation is None:
            generation = group.desired_generation
        else:
            generation = group.desired_generation + 1
            await conn.execute(
                """
                WITH control AS (
                    SELECT clock_timestamp() AS control_time
                )
                UPDATE enterprise_chat_memory_generations
                SET state = 'abandoned', updated_at = control.control_time,
                    last_error_code = 'source_changed',
                    last_error_message = 'Superseded by a newer source snapshot',
                    last_error_at = control.control_time
                FROM control
                WHERE user_id = $1 AND project_id = $2 AND state = 'building'
                """,
                group.user_id,
                group.project_id,
            )
            await self._insert_postgres_chat_memory_generation(
                conn,
                user_id=group.user_id,
                project_id=group.project_id,
                generation=generation,
                config_fingerprint=config_fingerprint,
                graph_store_fingerprint=graph_store_fingerprint,
                state="building",
            )
            await conn.execute(
                """
                UPDATE enterprise_chat_memory_groups
                SET desired_generation = $3,
                    desired_config_fingerprint = $4,
                    desired_graph_store_fingerprint = $5,
                    state = 'rebuilding', state_version = state_version + 1,
                    active_rebuild_event_id = NULL, last_error_code = NULL,
                    last_error_message = NULL, last_error_at = NULL,
                    updated_at = clock_timestamp(),
                    deleted_at = NULL
                WHERE user_id = $1 AND project_id = $2
                """,
                group.user_id,
                group.project_id,
                generation,
                config_fingerprint,
                graph_store_fingerprint,
            )

        event_seq, _ = await self._allocate_postgres_chat_memory_event_seq(
            conn,
            group.user_id,
            group.project_id,
            allocate_reference_time=False,
        )
        event_id, deterministic_key = _chat_memory_event_identity(
            event_type="rebuild",
            user_id=group.user_id,
            project_id=group.project_id,
            event_seq=event_seq,
            generation=generation,
            target_session_id=target_session_id,
            target_message_id=target_message_id,
        )
        await conn.execute(
            """
            WITH control AS (
                SELECT clock_timestamp() AS control_time
            )
            UPDATE enterprise_chat_memory_outbox
            SET status = 'superseded', superseded_by_event_id = $1,
                completed_at = control.control_time,
                updated_at = control.control_time
            FROM control
            WHERE user_id = $2 AND project_id = $3 AND event_seq < $4
              AND event_type IN ('ingest', 'rebuild')
              AND status IN ('pending', 'retry_wait', 'dead_letter')
            """,
            event_id,
            group.user_id,
            group.project_id,
            event_seq,
        )
        now = str(_iso_timestamp(await conn.fetchval("SELECT clock_timestamp()")))
        event = ChatMemoryOutboxEventRecord(
            event_id=event_id,
            deterministic_key=deterministic_key,
            user_id=group.user_id,
            project_id=group.project_id,
            event_seq=event_seq,
            generation=generation,
            graph_group_id=chat_memory_graph_group_id(
                group.user_id, group.project_id, generation
            ),
            config_fingerprint=config_fingerprint,
            graph_store_fingerprint=graph_store_fingerprint,
            event_type="rebuild",
            status="pending",
            available_at=now,
            attempt_no=0,
            created_at=now,
            updated_at=now,
            actor_user_id=actor_user_id,
            actor_tenant_id=actor_tenant_id,
            target_user_id=group.user_id,
            target_project_id=group.project_id,
            target_session_id=target_session_id,
            target_message_id=target_message_id,
        )
        await self._insert_postgres_chat_memory_event(conn, event)
        await conn.execute(
            """
            UPDATE enterprise_chat_memory_groups
            SET state = 'rebuilding', desired_config_fingerprint = $3,
                desired_graph_store_fingerprint = $4,
                active_rebuild_event_id = $5, updated_at = clock_timestamp()
            WHERE user_id = $1 AND project_id = $2
            """,
            group.user_id,
            group.project_id,
            config_fingerprint,
            graph_store_fingerprint,
            event_id,
        )
        return event

    async def _postgres_chat_memory_purge_group_ids(
        self,
        conn: Any,
        user_id: str,
        project_id: str,
    ) -> tuple[str, ...]:
        """Return all inventory, mapping, outbox, and legacy purge targets."""

        rows = await conn.fetch(
            """
            SELECT graph_group_id FROM (
                SELECT graph_group_id
                FROM enterprise_chat_memory_generations
                WHERE user_id = $1 AND project_id = $2
                UNION
                SELECT graph_group_id
                FROM enterprise_chat_memory_episodes
                WHERE user_id = $1 AND project_id = $2
                UNION
                SELECT graph_group_id
                FROM enterprise_chat_memory_outbox
                WHERE user_id = $1 AND project_id = $2
            ) AS targets
            WHERE graph_group_id IS NOT NULL
              AND btrim(graph_group_id) <> ''
            ORDER BY graph_group_id ASC
            """,
            user_id,
            project_id,
        )
        return _normalize_chat_memory_group_ids(
            [
                *(str(row["graph_group_id"]) for row in rows),
                chat_memory_legacy_graph_group_id(user_id, project_id),
            ]
        )

    async def _postgres_chat_memory_rebuild_group_ids(
        self,
        conn: Any,
        event: ChatMemoryOutboxEventRecord,
    ) -> tuple[str, ...]:
        """Return every old, orphan, legacy, and target rebuild group id."""

        return _normalize_chat_memory_group_ids(
            [
                *(
                    await self._postgres_chat_memory_purge_group_ids(
                        conn, event.user_id, event.project_id
                    )
                ),
                event.graph_group_id,
            ]
        )

    async def _enqueue_postgres_chat_memory_purge(
        self,
        conn: Any,
        user_id: str,
        project_id: str,
        config_fingerprint: str,
        graph_store_fingerprint: str | None = None,
        *,
        actor_user_id: str | None,
        actor_tenant_id: str | None,
    ) -> ChatMemoryOutboxEventRecord | None:
        graph_store_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            config_fingerprint, graph_store_fingerprint
        )
        group = await self._get_postgres_chat_memory_group(
            conn, user_id, project_id, for_update=True
        )
        if group is not None and group.state == "deleted":
            return None
        existing = await conn.fetchrow(
            """
            SELECT * FROM enterprise_chat_memory_outbox
            WHERE user_id = $1 AND project_id = $2 AND event_type = 'purge'
              AND status IN ('pending', 'running', 'retry_wait')
            ORDER BY event_seq DESC LIMIT 1 FOR UPDATE
            """,
            user_id,
            project_id,
        )
        if existing is not None:
            return _chat_memory_event_from_row(existing)
        if group is None:
            group, _ = await self._ensure_postgres_chat_memory_group(
                conn,
                user_id,
                project_id,
                config_fingerprint,
                graph_store_fingerprint,
                generation_state="purge_pending",
            )
        else:
            graph_store_fingerprint = (
                _chat_memory_existing_graph_store_fingerprint(
                    group, graph_store_fingerprint
                )
            )
        generation_rows = await conn.fetch(
            """
            SELECT generation FROM enterprise_chat_memory_generations
            WHERE user_id = $1 AND project_id = $2
            ORDER BY generation FOR UPDATE
            """,
            user_id,
            project_id,
        )
        if not any(
            int(row["generation"]) == group.desired_generation
            for row in generation_rows
        ):
            await self._insert_postgres_chat_memory_generation(
                conn,
                user_id=user_id,
                project_id=project_id,
                generation=group.desired_generation,
                config_fingerprint=group.desired_config_fingerprint,
                graph_store_fingerprint=(
                    group.desired_graph_store_fingerprint
                    or group.desired_config_fingerprint
                ),
                state="purge_pending",
            )
        await conn.execute(
            """
            UPDATE enterprise_chat_memory_generations
            SET state = 'purge_pending', updated_at = clock_timestamp(),
                cleared_at = NULL
            WHERE user_id = $1 AND project_id = $2 AND state <> 'purged'
            """,
            user_id,
            project_id,
        )
        await conn.execute(
            """
            UPDATE enterprise_chat_memory_groups
            SET state = 'deleting', state_version = state_version + 1,
                desired_config_fingerprint = $3,
                active_rebuild_event_id = NULL, last_error_code = NULL,
                active_generation = NULL, active_config_fingerprint = NULL,
                active_graph_store_fingerprint = NULL,
                last_error_message = NULL, last_error_at = NULL,
                updated_at = clock_timestamp(),
                deleted_at = NULL
            WHERE user_id = $1 AND project_id = $2
            """,
            user_id,
            project_id,
            config_fingerprint,
        )
        event_seq, _ = await self._allocate_postgres_chat_memory_event_seq(
            conn,
            user_id,
            project_id,
            allocate_reference_time=False,
        )
        event_id, deterministic_key = _chat_memory_event_identity(
            event_type="purge",
            user_id=user_id,
            project_id=project_id,
            event_seq=event_seq,
            generation=group.desired_generation,
        )
        await conn.execute(
            """
            WITH control AS (
                SELECT clock_timestamp() AS control_time
            )
            UPDATE enterprise_chat_memory_outbox
            SET status = 'superseded', superseded_by_event_id = $1,
                completed_at = control.control_time,
                updated_at = control.control_time
            FROM control
            WHERE user_id = $2 AND project_id = $3 AND event_seq < $4
              AND status IN ('pending', 'retry_wait', 'dead_letter')
            """,
            event_id,
            user_id,
            project_id,
            event_seq,
        )
        now = str(_iso_timestamp(await conn.fetchval("SELECT clock_timestamp()")))
        event = ChatMemoryOutboxEventRecord(
            event_id=event_id,
            deterministic_key=deterministic_key,
            user_id=user_id,
            project_id=project_id,
            event_seq=event_seq,
            generation=group.desired_generation,
            graph_group_id=chat_memory_graph_group_id(
                user_id, project_id, group.desired_generation
            ),
            config_fingerprint=config_fingerprint,
            graph_store_fingerprint=graph_store_fingerprint,
            event_type="purge",
            status="pending",
            available_at=now,
            attempt_no=0,
            created_at=now,
            updated_at=now,
            actor_user_id=actor_user_id,
            actor_tenant_id=actor_tenant_id,
            target_user_id=user_id,
            target_project_id=project_id,
        )
        await self._insert_postgres_chat_memory_event(conn, event)
        return event

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
            )
            """
        )
        schema_version_rows = await conn.fetch(
            "SELECT version FROM kb_metadata_schema ORDER BY version FOR UPDATE"
        )
        schema_versions = {int(row["version"]) for row in schema_version_rows}
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
                append_batch_id TEXT,
                project_event_seq BIGINT,
                memory_reference_time TIMESTAMPTZ,
                data_json JSONB NOT NULL,
                FOREIGN KEY (session_id) REFERENCES enterprise_chat_sessions(id),
                FOREIGN KEY (user_id) REFERENCES enterprise_users(id),
                CONSTRAINT enterprise_chat_messages_admission_v2_check CHECK (
                    (append_batch_id IS NULL AND project_event_seq IS NULL
                     AND memory_reference_time IS NULL)
                    OR
                    (append_batch_id IS NOT NULL AND project_event_seq > 0
                     AND memory_reference_time IS NOT NULL)
                )
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
                created_at TEXT NOT NULL,
                event_id TEXT,
                generation BIGINT,
                graph_group_id TEXT,
                append_batch_id TEXT,
                project_event_seq BIGINT,
                CONSTRAINT enterprise_chat_memory_episode_generation_v2_check
                    CHECK (generation IS NULL OR generation > 0),
                CONSTRAINT enterprise_chat_memory_episode_identity_v2_check CHECK ((
                    (event_id IS NULL AND generation IS NULL
                     AND graph_group_id IS NULL AND append_batch_id IS NULL
                     AND project_event_seq IS NULL)
                    OR
                    (event_id IS NOT NULL AND generation > 0
                     AND graph_group_id IS NOT NULL
                     AND append_batch_id IS NULL
                     AND project_event_seq IS NULL)
                    OR
                    (event_id IS NOT NULL AND generation > 0
                     AND graph_group_id IS NOT NULL
                     AND append_batch_id IS NOT NULL
                     AND project_event_seq > 0)
                ) IS TRUE)
            );
            CREATE INDEX IF NOT EXISTS idx_enterprise_chat_memory_episodes_session
                ON enterprise_chat_memory_episodes (session_id, last_seq);
            CREATE INDEX IF NOT EXISTS idx_enterprise_chat_memory_episodes_project
                ON enterprise_chat_memory_episodes (project_id);
            CREATE INDEX IF NOT EXISTS idx_enterprise_chat_memory_episodes_user
                ON enterprise_chat_memory_episodes (user_id);

            CREATE TABLE IF NOT EXISTS enterprise_chat_memory_groups (
                user_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                logical_group_id TEXT NOT NULL UNIQUE,
                active_generation BIGINT,
                desired_generation BIGINT NOT NULL,
                next_event_seq BIGINT NOT NULL DEFAULT 1,
                last_reference_time TIMESTAMPTZ,
                state TEXT NOT NULL CHECK (
                    state IN ('active', 'rebuilding', 'deleting', 'failed', 'deleted')
                ),
                state_version BIGINT NOT NULL DEFAULT 1,
                active_config_fingerprint TEXT,
                desired_config_fingerprint TEXT NOT NULL,
                active_graph_store_fingerprint TEXT,
                desired_graph_store_fingerprint TEXT NOT NULL,
                active_rebuild_event_id TEXT,
                last_success_at TIMESTAMPTZ,
                last_error_code TEXT,
                last_error_message TEXT,
                last_error_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                deleted_at TIMESTAMPTZ,
                record_version INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (user_id, project_id),
                CHECK (desired_generation >= 1),
                CHECK (next_event_seq >= 1),
                CHECK (state_version >= 1),
                CHECK (
                    active_generation IS NULL OR (
                        active_generation >= 1
                        AND active_generation <= desired_generation
                    )
                ),
                CONSTRAINT enterprise_chat_memory_group_active_identity_v4_check CHECK (
                    (active_generation IS NULL
                     AND active_config_fingerprint IS NULL
                     AND active_graph_store_fingerprint IS NULL)
                    OR
                    (active_generation IS NOT NULL
                     AND active_config_fingerprint IS NOT NULL
                     AND active_graph_store_fingerprint IS NOT NULL)
                ),
                CONSTRAINT enterprise_chat_memory_group_desired_graph_v4_check CHECK (
                    desired_graph_store_fingerprint <> ''
                    AND desired_graph_store_fingerprint =
                        btrim(desired_graph_store_fingerprint)
                    AND (
                        active_graph_store_fingerprint IS NULL
                        OR (
                            active_graph_store_fingerprint <> ''
                            AND active_graph_store_fingerprint =
                                btrim(active_graph_store_fingerprint)
                        )
                    )
                ),
                CHECK (state <> 'active' OR active_generation IS NOT NULL),
                CHECK (record_version = 1)
            );
            CREATE INDEX IF NOT EXISTS idx_enterprise_chat_memory_groups_state
                ON enterprise_chat_memory_groups (state, updated_at, project_id);

            CREATE TABLE IF NOT EXISTS enterprise_chat_memory_generations (
                user_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                generation BIGINT NOT NULL,
                graph_group_id TEXT NOT NULL UNIQUE,
                config_fingerprint TEXT NOT NULL,
                graph_store_fingerprint TEXT NOT NULL,
                state TEXT NOT NULL CHECK (
                    state IN (
                        'building', 'active', 'retired', 'abandoned',
                        'purge_pending', 'purged'
                    )
                ),
                snapshot_cutoff BIGINT,
                replay_batch_count BIGINT,
                replay_message_count BIGINT,
                replay_byte_count BIGINT,
                snapshot_digest TEXT,
                clear_attempt_no INTEGER NOT NULL DEFAULT 0,
                clear_started_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                activated_at TIMESTAMPTZ,
                cleared_at TIMESTAMPTZ,
                last_error_code TEXT,
                last_error_message TEXT,
                last_error_at TIMESTAMPTZ,
                record_version INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (user_id, project_id, generation),
                CHECK (generation >= 1),
                CHECK (snapshot_cutoff IS NULL OR snapshot_cutoff >= 0),
                CHECK (replay_batch_count IS NULL OR replay_batch_count >= 0),
                CHECK (replay_message_count IS NULL OR replay_message_count >= 0),
                CHECK (replay_byte_count IS NULL OR replay_byte_count >= 0),
                CHECK (clear_attempt_no >= 0),
                CONSTRAINT enterprise_chat_memory_generation_graph_v4_check CHECK (
                    graph_store_fingerprint <> ''
                    AND graph_store_fingerprint = btrim(graph_store_fingerprint)
                ),
                CHECK (record_version = 1)
            );
            CREATE INDEX IF NOT EXISTS idx_enterprise_chat_memory_generations_group
                ON enterprise_chat_memory_generations (
                    user_id, project_id, generation, state
                );

            CREATE TABLE IF NOT EXISTS enterprise_chat_memory_outbox (
                event_id TEXT PRIMARY KEY,
                deterministic_key TEXT NOT NULL UNIQUE,
                user_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                event_seq BIGINT NOT NULL,
                generation BIGINT NOT NULL,
                graph_group_id TEXT NOT NULL,
                config_fingerprint TEXT NOT NULL,
                graph_store_fingerprint TEXT NOT NULL,
                event_type TEXT NOT NULL CHECK (
                    event_type IN ('ingest', 'rebuild', 'purge')
                ),
                status TEXT NOT NULL CHECK (
                    status IN (
                        'pending', 'running', 'retry_wait', 'succeeded',
                        'superseded', 'dead_letter'
                    )
                ),
                available_at TIMESTAMPTZ NOT NULL,
                attempt_no INTEGER NOT NULL DEFAULT 0,
                source_session_id TEXT,
                append_batch_id TEXT,
                first_seq BIGINT,
                last_seq BIGINT,
                snapshot_cutoff BIGINT,
                snapshot_batch_count BIGINT,
                snapshot_message_count BIGINT,
                snapshot_byte_count BIGINT,
                snapshot_digest TEXT,
                claim_token TEXT,
                claimed_by TEXT,
                claimed_at TIMESTAMPTZ,
                side_effect_started_at TIMESTAMPTZ,
                side_effect_state_version BIGINT,
                completed_at TIMESTAMPTZ,
                superseded_by_event_id TEXT,
                last_error_code TEXT,
                last_error_message TEXT,
                last_error_at TIMESTAMPTZ,
                actor_user_id TEXT,
                actor_tenant_id TEXT,
                target_user_id TEXT,
                target_project_id TEXT,
                target_session_id TEXT,
                target_message_id TEXT,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                record_version INTEGER NOT NULL DEFAULT 1,
                UNIQUE (user_id, project_id, event_seq),
                CHECK (event_seq >= 1),
                CHECK (generation >= 1),
                CHECK (attempt_no >= 0),
                CHECK (snapshot_cutoff IS NULL OR snapshot_cutoff >= 0),
                CHECK (snapshot_batch_count IS NULL OR snapshot_batch_count >= 0),
                CHECK (snapshot_message_count IS NULL OR snapshot_message_count >= 0),
                CHECK (snapshot_byte_count IS NULL OR snapshot_byte_count >= 0),
                CHECK (
                    (first_seq IS NULL AND last_seq IS NULL)
                    OR (first_seq >= 1 AND last_seq >= first_seq)
                ),
                CONSTRAINT enterprise_chat_memory_outbox_graph_v4_check CHECK (
                    graph_store_fingerprint <> ''
                    AND graph_store_fingerprint = btrim(graph_store_fingerprint)
                ),
                CHECK (record_version = 1)
            );
            CREATE INDEX IF NOT EXISTS idx_enterprise_chat_memory_outbox_claim
                ON enterprise_chat_memory_outbox (
                    status, available_at, user_id, project_id, event_seq
                );
            CREATE INDEX IF NOT EXISTS idx_enterprise_chat_memory_outbox_head
                ON enterprise_chat_memory_outbox (
                    user_id, project_id, event_seq, status
                )
                WHERE status IN ('pending', 'running', 'retry_wait', 'dead_letter');
            CREATE INDEX IF NOT EXISTS idx_enterprise_chat_memory_outbox_generation
                ON enterprise_chat_memory_outbox (
                    user_id, project_id, generation, event_seq
                );
            CREATE UNIQUE INDEX IF NOT EXISTS uq_enterprise_chat_memory_rebuild_target
                ON enterprise_chat_memory_outbox (user_id, project_id, generation)
                WHERE event_type = 'rebuild'
                  AND status IN ('pending', 'running', 'retry_wait', 'dead_letter');

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
        chat_memory_v2_complete = await self._chat_memory_schema_v2_complete(conn)
        chat_memory_v2_migration_needed = (
            2 not in schema_versions or not chat_memory_v2_complete
        )
        if chat_memory_v2_migration_needed:
            await self._migrate_chat_memory_schema_v2(conn)
            if not await self._chat_memory_schema_v2_complete(conn):
                raise RuntimeError("Chat Memory metadata schema v2 migration incomplete")
        chat_memory_v3_complete = await self._chat_memory_schema_v3_complete(conn)
        chat_memory_v3_migration_needed = (
            3 not in schema_versions or not chat_memory_v3_complete
        )
        if chat_memory_v3_migration_needed:
            await self._migrate_chat_memory_schema_v3(conn)
            if not await self._chat_memory_schema_v3_complete(conn):
                raise RuntimeError("Chat Memory metadata schema v3 migration incomplete")
        chat_memory_v4_complete = await self._chat_memory_schema_v4_complete(conn)
        chat_memory_v4_migration_needed = (
            4 not in schema_versions or not chat_memory_v4_complete
        )
        if chat_memory_v4_migration_needed:
            await self._migrate_chat_memory_schema_v4(conn)
            if not await self._chat_memory_schema_v4_complete(conn):
                raise RuntimeError("Chat Memory metadata schema v4 migration incomplete")
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
            VALUES (1, clock_timestamp()::text)
            ON CONFLICT (version) DO NOTHING
            """
        )
        if chat_memory_v2_migration_needed:
            await conn.execute(
                """
                INSERT INTO kb_metadata_schema(version, applied_at)
                VALUES (2, clock_timestamp()::text)
                ON CONFLICT (version) DO UPDATE
                SET applied_at = excluded.applied_at
                """
            )
        if chat_memory_v3_migration_needed:
            await conn.execute(
                """
                INSERT INTO kb_metadata_schema(version, applied_at)
                VALUES (3, clock_timestamp()::text)
                ON CONFLICT (version) DO UPDATE
                SET applied_at = excluded.applied_at
                """
            )
        if chat_memory_v4_migration_needed:
            await conn.execute(
                """
                INSERT INTO kb_metadata_schema(version, applied_at)
                VALUES (4, clock_timestamp()::text)
                ON CONFLICT (version) DO UPDATE
                SET applied_at = excluded.applied_at
                """
            )

    async def _chat_memory_schema_v2_complete(self, conn: Any) -> bool:
        return bool(
            await conn.fetchval(
                """
                SELECT
                    (
                        SELECT COUNT(*) = 3
                        FROM information_schema.columns
                        WHERE table_schema = current_schema()
                          AND table_name = 'enterprise_chat_messages'
                          AND column_name IN (
                              'append_batch_id', 'project_event_seq',
                              'memory_reference_time'
                          )
                    )
                    AND (
                        SELECT COUNT(*) = 5
                        FROM information_schema.columns
                        WHERE table_schema = current_schema()
                          AND table_name = 'enterprise_chat_memory_episodes'
                          AND column_name IN (
                              'event_id', 'generation', 'graph_group_id',
                              'append_batch_id', 'project_event_seq'
                          )
                    )
                    AND EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_schema = current_schema()
                          AND table_name = 'enterprise_chat_memory_outbox'
                          AND column_name = 'claimed_by'
                    )
                    AND EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_schema = current_schema()
                          AND table_name = 'enterprise_chat_memory_outbox'
                          AND column_name = 'side_effect_state_version'
                    )
                    AND EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_schema = current_schema()
                          AND table_name = 'enterprise_chat_memory_generations'
                          AND column_name = 'replay_byte_count'
                    )
                    AND (
                        SELECT COUNT(*) = 3
                        FROM information_schema.columns
                        WHERE table_schema = current_schema()
                          AND table_name = 'enterprise_chat_memory_outbox'
                          AND column_name IN (
                              'snapshot_batch_count', 'snapshot_message_count',
                              'snapshot_byte_count'
                          )
                    )
                    AND EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conrelid = 'enterprise_chat_messages'::regclass
                          AND conname =
                              'enterprise_chat_messages_admission_v2_check'
                          AND contype = 'c' AND convalidated
                    )
                    AND 1 = (
                        SELECT COUNT(*) FROM pg_constraint
                        WHERE conrelid = 'enterprise_chat_messages'::regclass
                          AND contype = 'c'
                          AND pg_get_constraintdef(oid) ILIKE
                              '%append_batch_id%memory_reference_time%'
                    )
                    AND EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conrelid =
                              'enterprise_chat_memory_episodes'::regclass
                          AND conname =
                              'enterprise_chat_memory_episode_generation_v2_check'
                          AND contype = 'c' AND convalidated
                    )
                    AND EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conrelid =
                              'enterprise_chat_memory_episodes'::regclass
                          AND conname =
                              'enterprise_chat_memory_episode_identity_v2_check'
                          AND contype = 'c' AND convalidated
                          AND pg_get_constraintdef(oid) ILIKE '%IS TRUE%'
                    )
                    AND 1 = (
                        SELECT COUNT(*) FROM pg_constraint
                        WHERE conrelid =
                              'enterprise_chat_memory_episodes'::regclass
                          AND contype = 'c'
                          AND pg_get_constraintdef(oid) ILIKE '%event_id%'
                          AND pg_get_constraintdef(oid) ILIKE '%graph_group_id%'
                    )
                    AND EXISTS (
                        SELECT 1 FROM pg_index
                        WHERE indexrelid = to_regclass(
                            'uq_enterprise_chat_memory_episode_generation_batch'
                        )
                          AND indisunique AND indisvalid
                          AND pg_get_indexdef(indexrelid) ILIKE
                              '%user_id, project_id, generation, append_batch_id%'
                    )
                    AND to_regclass(
                        'uq_enterprise_chat_memory_episodes_event'
                    ) IS NULL
                    AND to_regclass(
                        'idx_enterprise_chat_messages_memory_replay'
                    ) IS NOT NULL
                    AND to_regclass(
                        'idx_enterprise_chat_memory_episodes_generation'
                    ) IS NOT NULL
                """
            )
        )

    async def _migrate_chat_memory_schema_v2(self, conn: Any) -> None:
        """Install Chat Memory v2 once without recurring constraint churn."""

        await conn.execute(
            """
            ALTER TABLE enterprise_chat_messages
                ADD COLUMN IF NOT EXISTS append_batch_id TEXT;
            ALTER TABLE enterprise_chat_messages
                ADD COLUMN IF NOT EXISTS project_event_seq BIGINT;
            ALTER TABLE enterprise_chat_messages
                ADD COLUMN IF NOT EXISTS memory_reference_time TIMESTAMPTZ;
            ALTER TABLE enterprise_chat_memory_episodes
                ADD COLUMN IF NOT EXISTS event_id TEXT;
            ALTER TABLE enterprise_chat_memory_episodes
                ADD COLUMN IF NOT EXISTS generation BIGINT;
            ALTER TABLE enterprise_chat_memory_episodes
                ADD COLUMN IF NOT EXISTS graph_group_id TEXT;
            ALTER TABLE enterprise_chat_memory_episodes
                ADD COLUMN IF NOT EXISTS append_batch_id TEXT;
            ALTER TABLE enterprise_chat_memory_episodes
                ADD COLUMN IF NOT EXISTS project_event_seq BIGINT;
            ALTER TABLE enterprise_chat_memory_groups
                ADD COLUMN IF NOT EXISTS last_error_at TIMESTAMPTZ;
            ALTER TABLE enterprise_chat_memory_generations
                ADD COLUMN IF NOT EXISTS last_error_at TIMESTAMPTZ;
            ALTER TABLE enterprise_chat_memory_outbox
                ADD COLUMN IF NOT EXISTS last_error_at TIMESTAMPTZ;
            ALTER TABLE enterprise_chat_memory_outbox
                ADD COLUMN IF NOT EXISTS claimed_by TEXT;
            ALTER TABLE enterprise_chat_memory_outbox
                ADD COLUMN IF NOT EXISTS side_effect_state_version BIGINT;
            ALTER TABLE enterprise_chat_memory_generations
                ADD COLUMN IF NOT EXISTS replay_byte_count BIGINT;
            ALTER TABLE enterprise_chat_memory_outbox
                ADD COLUMN IF NOT EXISTS snapshot_batch_count BIGINT;
            ALTER TABLE enterprise_chat_memory_outbox
                ADD COLUMN IF NOT EXISTS snapshot_message_count BIGINT;
            ALTER TABLE enterprise_chat_memory_outbox
                ADD COLUMN IF NOT EXISTS snapshot_byte_count BIGINT;

            DROP INDEX IF EXISTS uq_enterprise_chat_memory_episodes_event;
            DROP INDEX IF EXISTS
                uq_enterprise_chat_memory_episode_generation_batch;
            """
        )

        await conn.execute(
            """
            UPDATE enterprise_chat_messages
            SET append_batch_id = NULL,
                project_event_seq = NULL,
                memory_reference_time = NULL
            WHERE (
                (append_batch_id IS NULL AND project_event_seq IS NULL
                 AND memory_reference_time IS NULL)
                OR
                (append_batch_id IS NOT NULL AND project_event_seq > 0
                 AND memory_reference_time IS NOT NULL)
            ) IS NOT TRUE
            """
        )
        await conn.execute(
            """
            UPDATE enterprise_chat_memory_episodes AS episode
            SET append_batch_id = outbox.append_batch_id,
                project_event_seq = outbox.event_seq
            FROM enterprise_chat_memory_outbox AS outbox
            WHERE episode.event_id = outbox.event_id
              AND episode.append_batch_id IS NULL
              AND episode.project_event_seq IS NULL
              AND episode.generation IS NOT NULL
              AND episode.graph_group_id IS NOT NULL
              AND outbox.append_batch_id IS NOT NULL
            """
        )
        await conn.execute(
            """
            UPDATE enterprise_chat_memory_episodes
            SET event_id = NULL,
                generation = NULL,
                graph_group_id = NULL,
                append_batch_id = NULL,
                project_event_seq = NULL
            WHERE (
                (event_id IS NULL AND generation IS NULL
                 AND graph_group_id IS NULL AND append_batch_id IS NULL
                 AND project_event_seq IS NULL)
                OR
                (event_id IS NOT NULL AND generation > 0
                 AND graph_group_id IS NOT NULL AND append_batch_id IS NULL
                 AND project_event_seq IS NULL)
                OR
                (event_id IS NOT NULL AND generation > 0
                 AND graph_group_id IS NOT NULL
                 AND append_batch_id IS NOT NULL
                 AND project_event_seq > 0)
            ) IS NOT TRUE
            """
        )
        await conn.execute(
            """
            WITH ranked AS (
                SELECT episode_uuid,
                       ROW_NUMBER() OVER (
                           PARTITION BY user_id, project_id, generation,
                                        append_batch_id
                           ORDER BY created_at, episode_uuid
                       ) AS duplicate_rank
                FROM enterprise_chat_memory_episodes
                WHERE generation IS NOT NULL AND append_batch_id IS NOT NULL
            )
            UPDATE enterprise_chat_memory_episodes AS episode
            SET event_id = NULL,
                generation = NULL,
                graph_group_id = NULL,
                append_batch_id = NULL,
                project_event_seq = NULL
            FROM ranked
            WHERE episode.episode_uuid = ranked.episode_uuid
              AND ranked.duplicate_rank > 1
            """
        )
        await conn.execute(
            """
            DO $$
            DECLARE constraint_name TEXT;
            BEGIN
                FOR constraint_name IN
                    SELECT conname FROM pg_constraint
                    WHERE conrelid = 'enterprise_chat_messages'::regclass
                      AND contype = 'c'
                      AND conname <>
                          'enterprise_chat_messages_admission_v2_check'
                      AND pg_get_constraintdef(oid) ILIKE '%append_batch_id%'
                      AND pg_get_constraintdef(oid) ILIKE
                          '%memory_reference_time%'
                LOOP
                    EXECUTE format(
                        'ALTER TABLE enterprise_chat_messages '
                        'DROP CONSTRAINT %I', constraint_name
                    );
                END LOOP;
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conrelid = 'enterprise_chat_messages'::regclass
                      AND conname =
                          'enterprise_chat_messages_admission_v2_check'
                ) THEN
                    ALTER TABLE enterprise_chat_messages
                    ADD CONSTRAINT enterprise_chat_messages_admission_v2_check
                    CHECK (
                        (append_batch_id IS NULL AND project_event_seq IS NULL
                         AND memory_reference_time IS NULL)
                        OR
                        (append_batch_id IS NOT NULL AND project_event_seq > 0
                         AND memory_reference_time IS NOT NULL)
                    ) NOT VALID;
                END IF;
            END $$;
            ALTER TABLE enterprise_chat_messages VALIDATE CONSTRAINT
                enterprise_chat_messages_admission_v2_check;
            """
        )
        await conn.execute(
            """
            DO $$
            DECLARE constraint_name TEXT;
            BEGIN
                FOR constraint_name IN
                    SELECT conname FROM pg_constraint
                    WHERE conrelid =
                          'enterprise_chat_memory_episodes'::regclass
                      AND contype = 'c'
                      AND conname <>
                          'enterprise_chat_memory_episode_generation_v2_check'
                      AND (
                          conname <>
                              'enterprise_chat_memory_episode_identity_v2_check'
                          OR pg_get_constraintdef(oid) NOT ILIKE '%IS TRUE%'
                      )
                      AND (
                          pg_get_constraintdef(oid) ILIKE '%generation%'
                          OR pg_get_constraintdef(oid) ILIKE '%event_id%'
                          OR pg_get_constraintdef(oid) ILIKE '%append_batch_id%'
                      )
                LOOP
                    EXECUTE format(
                        'ALTER TABLE enterprise_chat_memory_episodes '
                        'DROP CONSTRAINT %I', constraint_name
                    );
                END LOOP;
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conrelid =
                          'enterprise_chat_memory_episodes'::regclass
                      AND conname =
                          'enterprise_chat_memory_episode_generation_v2_check'
                ) THEN
                    ALTER TABLE enterprise_chat_memory_episodes
                    ADD CONSTRAINT
                        enterprise_chat_memory_episode_generation_v2_check
                    CHECK (generation IS NULL OR generation > 0) NOT VALID;
                END IF;
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conrelid =
                          'enterprise_chat_memory_episodes'::regclass
                      AND conname =
                          'enterprise_chat_memory_episode_identity_v2_check'
                ) THEN
                    ALTER TABLE enterprise_chat_memory_episodes
                    ADD CONSTRAINT
                        enterprise_chat_memory_episode_identity_v2_check
                    CHECK ((
                        (event_id IS NULL AND generation IS NULL
                         AND graph_group_id IS NULL
                         AND append_batch_id IS NULL
                         AND project_event_seq IS NULL)
                        OR
                        (event_id IS NOT NULL AND generation > 0
                         AND graph_group_id IS NOT NULL
                         AND append_batch_id IS NULL
                         AND project_event_seq IS NULL)
                        OR
                        (event_id IS NOT NULL AND generation > 0
                         AND graph_group_id IS NOT NULL
                         AND append_batch_id IS NOT NULL
                         AND project_event_seq > 0)
                    ) IS TRUE) NOT VALID;
                END IF;
            END $$;
            ALTER TABLE enterprise_chat_memory_episodes VALIDATE CONSTRAINT
                enterprise_chat_memory_episode_generation_v2_check;
            ALTER TABLE enterprise_chat_memory_episodes VALIDATE CONSTRAINT
                enterprise_chat_memory_episode_identity_v2_check;
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_enterprise_chat_messages_memory_replay
                ON enterprise_chat_messages (
                    user_id, project_id, project_event_seq, session_id, seq
                )
                WHERE project_event_seq IS NOT NULL;
            CREATE UNIQUE INDEX
                uq_enterprise_chat_memory_episode_generation_batch
                ON enterprise_chat_memory_episodes (
                    user_id, project_id, generation, append_batch_id
                )
                WHERE generation IS NOT NULL AND append_batch_id IS NOT NULL;
            CREATE INDEX IF NOT EXISTS
                idx_enterprise_chat_memory_episodes_generation
                ON enterprise_chat_memory_episodes (
                    user_id, project_id, generation, graph_group_id
                );
            """
        )

    async def _chat_memory_schema_v3_complete(self, conn: Any) -> bool:
        return bool(
            await conn.fetchval(
                """
                SELECT (
                    SELECT COUNT(*) = 2
                    FROM information_schema.columns
                    WHERE table_schema = current_schema()
                      AND column_name = 'snapshot_digest'
                      AND table_name IN (
                          'enterprise_chat_memory_generations',
                          'enterprise_chat_memory_outbox'
                      )
                )
                """
            )
        )

    async def _migrate_chat_memory_schema_v3(self, conn: Any) -> None:
        """Add the versioned rebuild snapshot attestation columns."""

        await conn.execute(
            """
            ALTER TABLE enterprise_chat_memory_generations
                ADD COLUMN IF NOT EXISTS snapshot_digest TEXT;
            ALTER TABLE enterprise_chat_memory_outbox
                ADD COLUMN IF NOT EXISTS snapshot_digest TEXT;
            """
        )

    async def _chat_memory_schema_v4_complete(self, conn: Any) -> bool:
        columns_complete = bool(
            await conn.fetchval(
                """
                SELECT COUNT(*) = 4
                FROM information_schema.columns
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
        )
        if not columns_complete:
            return False
        return bool(
            await conn.fetchval(
                """
                SELECT NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = current_schema()
                      AND (
                          (table_name = 'enterprise_chat_memory_groups'
                           AND column_name = 'desired_graph_store_fingerprint')
                          OR
                          (table_name IN (
                               'enterprise_chat_memory_generations',
                               'enterprise_chat_memory_outbox'
                           ) AND column_name = 'graph_store_fingerprint')
                      )
                      AND is_nullable <> 'NO'
                )
                AND EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conrelid = 'enterprise_chat_memory_groups'::regclass
                      AND conname =
                          'enterprise_chat_memory_group_active_identity_v4_check'
                      AND contype = 'c' AND convalidated
                )
                AND EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conrelid = 'enterprise_chat_memory_groups'::regclass
                      AND conname =
                          'enterprise_chat_memory_group_desired_graph_v4_check'
                      AND contype = 'c' AND convalidated
                      AND pg_get_constraintdef(oid) ILIKE
                          '%btrim(active_graph_store_fingerprint)%'
                )
                AND EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conrelid =
                          'enterprise_chat_memory_generations'::regclass
                      AND conname =
                          'enterprise_chat_memory_generation_graph_v4_check'
                      AND contype = 'c' AND convalidated
                )
                AND EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conrelid = 'enterprise_chat_memory_outbox'::regclass
                      AND conname =
                          'enterprise_chat_memory_outbox_graph_v4_check'
                      AND contype = 'c' AND convalidated
                )
                AND NOT EXISTS (
                    SELECT 1 FROM enterprise_chat_memory_groups
                    WHERE desired_graph_store_fingerprint IS NULL
                       OR (active_generation IS NULL) IS DISTINCT FROM
                          (active_graph_store_fingerprint IS NULL)
                )
                AND NOT EXISTS (
                    SELECT 1 FROM enterprise_chat_memory_generations
                    WHERE graph_store_fingerprint IS NULL
                )
                AND NOT EXISTS (
                    SELECT 1 FROM enterprise_chat_memory_outbox
                    WHERE graph_store_fingerprint IS NULL
                )
                """
            )
        )

    async def _migrate_chat_memory_schema_v4(self, conn: Any) -> None:
        """Split extraction/runtime identity from physical graph-store identity."""

        await conn.execute(
            """
            ALTER TABLE enterprise_chat_memory_groups
                ADD COLUMN IF NOT EXISTS active_graph_store_fingerprint TEXT;
            ALTER TABLE enterprise_chat_memory_groups
                ADD COLUMN IF NOT EXISTS desired_graph_store_fingerprint TEXT;
            ALTER TABLE enterprise_chat_memory_generations
                ADD COLUMN IF NOT EXISTS graph_store_fingerprint TEXT;
            ALTER TABLE enterprise_chat_memory_outbox
                ADD COLUMN IF NOT EXISTS graph_store_fingerprint TEXT;

            UPDATE enterprise_chat_memory_groups
            SET active_graph_store_fingerprint = active_config_fingerprint
            WHERE active_graph_store_fingerprint IS NULL
              AND active_config_fingerprint IS NOT NULL;
            UPDATE enterprise_chat_memory_groups
            SET desired_graph_store_fingerprint = desired_config_fingerprint
            WHERE desired_graph_store_fingerprint IS NULL;
            UPDATE enterprise_chat_memory_generations
            SET graph_store_fingerprint = config_fingerprint
            WHERE graph_store_fingerprint IS NULL;
            UPDATE enterprise_chat_memory_outbox
            SET graph_store_fingerprint = config_fingerprint
            WHERE graph_store_fingerprint IS NULL;

            ALTER TABLE enterprise_chat_memory_groups
                ALTER COLUMN desired_graph_store_fingerprint SET NOT NULL;
            ALTER TABLE enterprise_chat_memory_generations
                ALTER COLUMN graph_store_fingerprint SET NOT NULL;
            ALTER TABLE enterprise_chat_memory_outbox
                ALTER COLUMN graph_store_fingerprint SET NOT NULL;

            ALTER TABLE enterprise_chat_memory_groups DROP CONSTRAINT IF EXISTS
                enterprise_chat_memory_group_active_identity_v4_check;
            ALTER TABLE enterprise_chat_memory_groups DROP CONSTRAINT IF EXISTS
                enterprise_chat_memory_group_desired_graph_v4_check;
            ALTER TABLE enterprise_chat_memory_generations DROP CONSTRAINT IF EXISTS
                enterprise_chat_memory_generation_graph_v4_check;
            ALTER TABLE enterprise_chat_memory_outbox DROP CONSTRAINT IF EXISTS
                enterprise_chat_memory_outbox_graph_v4_check;
            """
        )
        await conn.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conrelid = 'enterprise_chat_memory_groups'::regclass
                      AND conname =
                          'enterprise_chat_memory_group_active_identity_v4_check'
                ) THEN
                    ALTER TABLE enterprise_chat_memory_groups
                    ADD CONSTRAINT
                        enterprise_chat_memory_group_active_identity_v4_check
                    CHECK (
                        (active_generation IS NULL
                         AND active_config_fingerprint IS NULL
                         AND active_graph_store_fingerprint IS NULL)
                        OR
                        (active_generation IS NOT NULL
                         AND active_config_fingerprint IS NOT NULL
                         AND active_graph_store_fingerprint IS NOT NULL)
                    ) NOT VALID;
                END IF;
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conrelid = 'enterprise_chat_memory_groups'::regclass
                      AND conname =
                          'enterprise_chat_memory_group_desired_graph_v4_check'
                ) THEN
                    ALTER TABLE enterprise_chat_memory_groups
                    ADD CONSTRAINT
                        enterprise_chat_memory_group_desired_graph_v4_check
                    CHECK (
                        desired_graph_store_fingerprint <> ''
                        AND desired_graph_store_fingerprint =
                            btrim(desired_graph_store_fingerprint)
                        AND (
                            active_graph_store_fingerprint IS NULL
                            OR (
                                active_graph_store_fingerprint <> ''
                                AND active_graph_store_fingerprint =
                                    btrim(active_graph_store_fingerprint)
                            )
                        )
                    ) NOT VALID;
                END IF;
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conrelid =
                          'enterprise_chat_memory_generations'::regclass
                      AND conname =
                          'enterprise_chat_memory_generation_graph_v4_check'
                ) THEN
                    ALTER TABLE enterprise_chat_memory_generations
                    ADD CONSTRAINT
                        enterprise_chat_memory_generation_graph_v4_check
                    CHECK (
                        graph_store_fingerprint <> ''
                        AND graph_store_fingerprint =
                            btrim(graph_store_fingerprint)
                    ) NOT VALID;
                END IF;
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conrelid = 'enterprise_chat_memory_outbox'::regclass
                      AND conname =
                          'enterprise_chat_memory_outbox_graph_v4_check'
                ) THEN
                    ALTER TABLE enterprise_chat_memory_outbox
                    ADD CONSTRAINT enterprise_chat_memory_outbox_graph_v4_check
                    CHECK (
                        graph_store_fingerprint <> ''
                        AND graph_store_fingerprint =
                            btrim(graph_store_fingerprint)
                    ) NOT VALID;
                END IF;
            END $$;
            ALTER TABLE enterprise_chat_memory_groups VALIDATE CONSTRAINT
                enterprise_chat_memory_group_active_identity_v4_check;
            ALTER TABLE enterprise_chat_memory_groups VALIDATE CONSTRAINT
                enterprise_chat_memory_group_desired_graph_v4_check;
            ALTER TABLE enterprise_chat_memory_generations VALIDATE CONSTRAINT
                enterprise_chat_memory_generation_graph_v4_check;
            ALTER TABLE enterprise_chat_memory_outbox VALIDATE CONSTRAINT
                enterprise_chat_memory_outbox_graph_v4_check;
            """
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
