from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import sqlite3
import threading
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, TypeVar

from lightrag.api.kb_service import _MetadataFileLock, utc_now_iso

MetadataJobStatus = Literal[
    "queued", "running", "succeeded", "failed", "cancelling", "cancelled", "retrying"
]

TenantUserKBOverrideEffect = Literal["allow", "deny"]

_TENANT_USER_KB_OVERRIDE_ROLES = frozenset(
    {"kb_viewer", "kb_editor", "kb_admin", "kb_owner"}
)
_TENANT_MEMBERSHIP_ROLES = frozenset(
    {"tenant_member", "tenant_admin", "tenant_owner"}
)
_LEGACY_KB_TOMBSTONE_PREFIX = "legacy-tombstone:"

DocumentStatus = Literal[
    "uploaded",
    "parse_queued",
    "parsing",
    "parsed",
    "parse_failed",
    "build_queued",
    "building",
    "ready",
    "build_failed",
    "deleting",
    "delete_failed",
    "replacing",
    "replace_failed",
    "deleted",
]

_SCHEMA_VERSION = 11
_T = TypeVar("_T")
_EXPECTATION_UNSET: Any = object()
_KB_OPERATION_LOCK_POLL_SECONDS = 0.05
_JOB_EXECUTION_LOCK_POLL_SECONDS = 0.05
_ORPHANED_JOB_STATUSES = frozenset({"running", "retrying", "cancelling"})
_ORPHANED_DOCUMENT_STATUS_TARGETS = {
    "parse_queued": "parse_failed",
    "parsing": "parse_failed",
    "build_queued": "build_failed",
    "building": "build_failed",
    "deleting": "delete_failed",
    "replacing": "replace_failed",
}

CHAT_MEMORY_RECORD_VERSION = 1
CHAT_MEMORY_SNAPSHOT_DIGEST_VERSION = 1
CHAT_MEMORY_DEFAULT_INGEST_MAX_CHARS = 6000
_CHAT_MEMORY_ADMISSION_POLICY_VERSION = 1
_CHAT_MEMORY_TRUNCATION_MARKER = "…[truncated]"
_CHAT_MEMORY_GRAPH_STORE_MIGRATION_REQUIRED = "graph_store_migration_required"
ChatMemoryGroupState = Literal[
    "active", "rebuilding", "deleting", "failed", "deleted"
]
ChatMemoryGenerationState = Literal[
    "building", "active", "retired", "abandoned", "purge_pending", "purged"
]
ChatMemoryEventType = Literal["ingest", "rebuild", "purge"]
ChatMemoryEventStatus = Literal[
    "pending", "running", "retry_wait", "succeeded", "superseded", "dead_letter"
]
_CHAT_MEMORY_BLOCKING_EVENT_STATUSES = (
    "pending",
    "running",
    "retry_wait",
    "dead_letter",
)


def chat_memory_logical_group_id(user_id: str, project_id: str) -> str:
    """Return the stable logical Chat Memory group id for one project."""

    digest = hashlib.sha256(f"{user_id}\0{project_id}".encode("utf-8")).hexdigest()
    return f"cm_{digest[:24]}"


def chat_memory_graph_group_id(
    user_id: str, project_id: str, generation: int
) -> str:
    """Return the generation-fenced physical Graphiti group id."""

    if int(generation) < 1:
        raise ValueError("Chat Memory generation must be at least 1")
    return f"{chat_memory_logical_group_id(user_id, project_id)}_g{int(generation)}"


def chat_memory_legacy_graph_group_id(user_id: str, project_id: str) -> str:
    """Return the pre-generation Graphiti group id used by legacy workers."""

    return f"{user_id}--{project_id}"


def _validate_chat_memory_fingerprint(config_fingerprint: str) -> str:
    if (
        not isinstance(config_fingerprint, str)
        or not config_fingerprint.strip()
        or config_fingerprint != config_fingerprint.strip()
    ):
        raise ValueError(
            "Chat Memory config_fingerprint must be a normalized non-empty string"
        )
    return config_fingerprint


def _resolve_chat_memory_graph_store_fingerprint(
    config_fingerprint: str,
    graph_store_fingerprint: str | None,
) -> str:
    """Resolve the graph-store identity for legacy-compatible store calls.

    Falling back to the extraction/runtime fingerprint is compatibility-only
    for existing tests and callers. Production callers must pass the graph
    store fingerprint explicitly so extraction changes cannot alias a physical
    graph store identity.
    """

    return _validate_chat_memory_fingerprint(
        config_fingerprint
        if graph_store_fingerprint is None
        else graph_store_fingerprint
    )


def _new_chat_memory_claim_token() -> str:
    return f"cmc_{secrets.token_urlsafe(24)}"


def _validate_chat_memory_worker_id(worker_id: str | None) -> str | None:
    if worker_id is None:
        return None
    if not isinstance(worker_id, str) or not worker_id.strip():
        raise ValueError("Chat Memory worker_id must be a non-empty string")
    return worker_id.strip()


def _normalize_chat_memory_event_types(
    event_types: Sequence[ChatMemoryEventType] | None,
) -> tuple[ChatMemoryEventType, ...]:
    if event_types is None:
        return ("ingest", "rebuild", "purge")
    normalized: list[ChatMemoryEventType] = []
    for event_type in event_types:
        if event_type not in {"ingest", "rebuild", "purge"}:
            raise ValueError(f"Unsupported Chat Memory event type: {event_type}")
        if event_type not in normalized:
            normalized.append(event_type)
    return tuple(normalized)


def _chat_memory_event_identity(
    *,
    event_type: ChatMemoryEventType,
    user_id: str,
    project_id: str,
    event_seq: int,
    generation: int,
    append_batch_id: str | None = None,
    target_session_id: str | None = None,
    target_message_id: str | None = None,
) -> tuple[str, str]:
    """Build a deterministic SQL event id and stable versioned key."""

    canonical = json.dumps(
        {
            "append_batch_id": append_batch_id,
            "event_seq": int(event_seq),
            "event_type": event_type,
            "generation": int(generation),
            "project_id": project_id,
            "record_version": CHAT_MEMORY_RECORD_VERSION,
            "target_message_id": target_message_id,
            "target_session_id": target_session_id,
            "user_id": user_id,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"cme_{digest[:32]}", f"chat-memory-event:v1:{digest}"


def _chat_memory_append_batch_id(
    *,
    user_id: str,
    project_id: str,
    session_id: str,
    event_seq: int,
    message_ids: Sequence[str],
) -> str:
    canonical = json.dumps(
        {
            "event_seq": int(event_seq),
            "message_ids": list(message_ids),
            "project_id": project_id,
            "record_version": CHAT_MEMORY_RECORD_VERSION,
            "session_id": session_id,
            "user_id": user_id,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"cmb_{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:32]}"


def _next_sqlite_chat_memory_reference_time(last_value: str | None) -> str:
    now = datetime.now(timezone.utc)
    if last_value:
        last = datetime.fromisoformat(last_value)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        now = max(now, last + timedelta(microseconds=1))
    return now.isoformat()


class _AsyncKBOperationLock:
    """Process-local writer-preferring shared/exclusive KB operation lock."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    @asynccontextmanager
    async def shared(self) -> AsyncIterator[None]:
        async with self._condition:
            await self._condition.wait_for(
                lambda: not self._writer and self._waiting_writers == 0
            )
            self._readers += 1
        try:
            yield
        finally:
            async with self._condition:
                self._readers -= 1
                if self._readers == 0:
                    self._condition.notify_all()

    @asynccontextmanager
    async def exclusive(self) -> AsyncIterator[None]:
        async with self._condition:
            self._waiting_writers += 1
            acquired = False
            try:
                await self._condition.wait_for(
                    lambda: not self._writer and self._readers == 0
                )
                self._writer = True
                acquired = True
            finally:
                self._waiting_writers -= 1
                if not acquired:
                    self._condition.notify_all()
        try:
            yield
        finally:
            async with self._condition:
                self._writer = False
                self._condition.notify_all()


_PROCESS_KB_OPERATION_LOCKS: dict[
    tuple[asyncio.AbstractEventLoop, str], _AsyncKBOperationLock
] = {}
_PROCESS_KB_OPERATION_LOCKS_GUARD = threading.Lock()


def _process_kb_operation_lock(db_path: Path, kb_id: str) -> _AsyncKBOperationLock:
    lock_name = (
        f"{db_path.resolve()}:"
        f"{hashlib.sha256(kb_id.encode('utf-8')).hexdigest()}"
    )
    key = (asyncio.get_running_loop(), lock_name)
    with _PROCESS_KB_OPERATION_LOCKS_GUARD:
        lock = _PROCESS_KB_OPERATION_LOCKS.get(key)
        if lock is None:
            lock = _AsyncKBOperationLock()
            _PROCESS_KB_OPERATION_LOCKS[key] = lock
        return lock


class _KBOperationFileLock:
    """Cross-process shared/exclusive byte-range lock for one KB operation."""

    def __init__(self, lock_path: Path, *, shared: bool) -> None:
        self.lock_path = lock_path
        self.shared = shared
        self._file: Any | None = None

    def try_acquire(self) -> bool:
        if self._file is not None:
            return True
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        file = self.lock_path.open("a+b")
        try:
            if file.seek(0, os.SEEK_END) == 0:
                file.write(b"0")
                file.flush()
                os.fsync(file.fileno())
            file.seek(0)
            if os.name == "nt":
                import msvcrt

                mode = msvcrt.LK_NBRLCK if self.shared else msvcrt.LK_NBLCK
                msvcrt.locking(file.fileno(), mode, 1)
            else:
                import fcntl

                mode = fcntl.LOCK_SH if self.shared else fcntl.LOCK_EX
                fcntl.flock(file.fileno(), mode | fcntl.LOCK_NB)
        except OSError:
            file.close()
            return False
        except BaseException:
            file.close()
            raise
        self._file = file
        return True

    def release(self) -> None:
        if self._file is None:
            return
        try:
            self._file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None


async def _acquire_kb_operation_file_lock(lock: _KBOperationFileLock) -> None:
    while not lock.try_acquire():
        await asyncio.sleep(_KB_OPERATION_LOCK_POLL_SECONDS)


class _AsyncJobExecutionLock:
    """Process-local exclusive lock with a real non-blocking acquire mode."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._locked = False

    async def acquire(self, *, wait: bool) -> bool:
        async with self._condition:
            if not wait:
                if self._locked:
                    return False
                self._locked = True
                return True
            await self._condition.wait_for(lambda: not self._locked)
            self._locked = True
            return True

    async def release(self) -> None:
        async with self._condition:
            self._locked = False
            self._condition.notify(1)


_PROCESS_JOB_EXECUTION_LOCKS: dict[
    tuple[asyncio.AbstractEventLoop, str], _AsyncJobExecutionLock
] = {}
_PROCESS_JOB_EXECUTION_LOCKS_GUARD = threading.Lock()


def _process_job_execution_lock(
    db_path: Path, job_id: str
) -> _AsyncJobExecutionLock:
    digest = hashlib.sha256(job_id.encode("utf-8")).hexdigest()
    lock_name = f"{db_path.resolve()}:{digest}"
    key = (asyncio.get_running_loop(), lock_name)
    with _PROCESS_JOB_EXECUTION_LOCKS_GUARD:
        lock = _PROCESS_JOB_EXECUTION_LOCKS.get(key)
        if lock is None:
            lock = _AsyncJobExecutionLock()
            _PROCESS_JOB_EXECUTION_LOCKS[key] = lock
        return lock


@dataclass(slots=True)
class _SQLiteJobGuardTaskState:
    owner_task: asyncio.Task[Any] | None
    depths: dict[str, int]


@dataclass(slots=True)
class _SQLiteKBWriteGuardTaskState:
    owner_task: asyncio.Task[Any] | None
    depths: dict[tuple[str, str | None], int]
    idle_events: dict[tuple[str, str | None], asyncio.Event]


@dataclass(slots=True)
class _SQLiteChatMemoryGuardTaskState:
    owner_task: asyncio.Task[Any] | None
    depths: dict[str, int]


_PROCESS_CHAT_MEMORY_GROUP_LOCKS: dict[
    tuple[asyncio.AbstractEventLoop, str], _AsyncJobExecutionLock
] = {}
_PROCESS_CHAT_MEMORY_GROUP_LOCKS_GUARD = threading.Lock()


def _process_chat_memory_group_lock(
    db_path: Path, logical_group_id: str
) -> _AsyncJobExecutionLock:
    lock_name = f"{db_path.resolve()}:{logical_group_id}"
    key = (asyncio.get_running_loop(), lock_name)
    with _PROCESS_CHAT_MEMORY_GROUP_LOCKS_GUARD:
        lock = _PROCESS_CHAT_MEMORY_GROUP_LOCKS.get(key)
        if lock is None:
            lock = _AsyncJobExecutionLock()
            _PROCESS_CHAT_MEMORY_GROUP_LOCKS[key] = lock
        return lock


async def _wait_for_kb_guard_borrowers(event: asyncio.Event) -> None:
    wait_task = asyncio.create_task(event.wait())
    try:
        await asyncio.shield(wait_task)
    except asyncio.CancelledError:
        await asyncio.gather(wait_task, return_exceptions=True)
        raise


def _validate_job_execution_id(job_id: str) -> None:
    if not isinstance(job_id, str) or not job_id.strip() or job_id != job_id.strip():
        raise MetadataStoreError(
            "Job execution lock id must be a normalized non-empty string"
        )


def _orphan_recovery_cutoff(grace_seconds: float) -> str:
    grace = max(0.0, float(grace_seconds))
    return (datetime.now(timezone.utc) - timedelta(seconds=grace)).isoformat()


def _same_job_execution_identity(left: JobRecord, right: JobRecord) -> bool:
    """Compare one durable claim incarnation without adding a run-token column."""

    return (
        left.id,
        left.kb_id,
        left.workspace,
        left.batch_id,
        left.document_id,
        left.job_type,
        left.created_at,
        left.queued_at,
        left.started_at,
        left.retry_count,
    ) == (
        right.id,
        right.kb_id,
        right.workspace,
        right.batch_id,
        right.document_id,
        right.job_type,
        right.created_at,
        right.queued_at,
        right.started_at,
        right.retry_count,
    )


def _job_recovery_document_ids(job: JobRecord) -> set[str]:
    document_ids: set[str] = set()
    if job.document_id:
        document_ids.add(job.document_id)
    payload = job.payload or {}
    raw_document_ids = payload.get("document_ids")
    if isinstance(raw_document_ids, list):
        document_ids.update(
            item for item in raw_document_ids if isinstance(item, str) and item
        )
    raw_items = payload.get("items")
    if isinstance(raw_items, list):
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            document_id = item.get("document_id")
            if isinstance(document_id, str) and document_id:
                document_ids.add(document_id)
    return document_ids


def _document_job_ids(document: DocumentRecord) -> set[str]:
    metadata = document.metadata or {}
    return {
        value
        for key in (
            "pending_parse_job_id",
            "current_parse_job_id",
            "pending_build_job_id",
            "current_build_job_id",
            "pending_delete_job_id",
            "current_delete_job_id",
            "pending_replace_job_id",
            "current_replace_job_id",
        )
        if isinstance((value := metadata.get(key)), str) and value
    }

# Aggregate jobs (``document_id IS NULL``) that a durable worker can still
# re-drive after a restart because everything they need is persisted:
# - ``delete``: ``documents:batch-delete`` carries ``document_ids`` + options;
# - ``parse`` / ``build_kg`` / ``reindex``: ``batch-*`` and multi-file
#   ``upload`` / ``texts`` auto_parse carry ``document_ids`` and the source
#   files are already on disk before the job runs;
# - ``sync``: ``documents:sync`` stages upload bytes under the batch id before
#   the queued aggregate job is created and persists per-item source keys,
#   source names, hashes, content types, and sync options in the job payload;
# - ``clear_kb``: carries ``kb_id`` / ``workspace``; the destructive clear is
#   idempotent so a queued job can be re-driven after restart.
# - ``agent_profile``: carries ``kb_id`` plus force/reason flags; profile
#   generation is idempotent and writes compact control-plane metadata.
# Single-document ``replace`` matches the ``document_id IS NOT NULL`` arm and is
# now worker-resumable: its uploaded bytes are staged to disk at claim time
# (``stage_replacement_bytes``) and a ``replace`` executor is registered, so a
# queued/retried replace job can be re-driven from disk.
_AGGREGATE_RESUMABLE_JOB_TYPES: frozenset[str] = frozenset(
    {"delete", "parse", "build_kg", "reindex", "sync", "clear_kb", "agent_profile"}
)


def _should_requeue_orphaned_clear_job(
    job: JobRecord,
    resumable_job_types: set[str],
) -> bool:
    """Return whether crash recovery should directly requeue this clear job."""

    return job.job_type == "clear_kb" and "clear_kb" in resumable_job_types


def _worker_eligibility_sql(job_types: Sequence[str]) -> tuple[str, list[Any]]:
    """SQL fragment + params selecting worker-claimable jobs.

    A job is eligible when its ``job_type`` is one the worker handles AND it is
    either single-document (``document_id`` set) or an aggregate type that is
    safely re-drivable from persisted state.
    """
    type_placeholders = ",".join("?" for _ in job_types)
    agg_types = sorted(_AGGREGATE_RESUMABLE_JOB_TYPES)
    agg_placeholders = ",".join("?" for _ in agg_types)
    fragment = (
        f"status = 'queued' AND job_type IN ({type_placeholders}) "
        f"AND (document_id IS NOT NULL OR job_type IN ({agg_placeholders}))"
    )
    params: list[Any] = [*job_types, *agg_types]
    return fragment, params


class MetadataStoreError(RuntimeError):
    pass


class MetadataConflictError(MetadataStoreError):
    """An optimistic concurrency or lifecycle precondition did not match."""

    def __init__(
        self,
        entity_type: str,
        entity_id: str,
        *,
        expected: dict[str, Any],
        current: dict[str, Any],
    ):
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.expected = expected
        self.current = current
        super().__init__(
            f"{entity_type} metadata conflict for '{entity_id}': "
            f"expected {expected}, current {current}"
        )


class KBLifecycleConflictError(MetadataConflictError):
    pass


class MetadataRecordNotFoundError(MetadataStoreError):
    pass


class InvalidJobTransitionError(MetadataStoreError):
    pass


class ActiveDocumentParseJobError(MetadataStoreError):
    def __init__(self, document_id: str, existing_job_id: str):
        self.document_id = document_id
        self.existing_job_id = existing_job_id
        super().__init__(f"Document '{document_id}' already has an active parse job")


class ActiveDocumentBuildJobError(MetadataStoreError):
    def __init__(self, document_id: str, existing_job_id: str):
        self.document_id = document_id
        self.existing_job_id = existing_job_id
        super().__init__(f"Document '{document_id}' already has an active build job")


class ActiveDocumentDeleteJobError(MetadataStoreError):
    def __init__(self, document_id: str, existing_job_id: str):
        self.document_id = document_id
        self.existing_job_id = existing_job_id
        super().__init__(f"Document '{document_id}' already has an active delete job")


class ActiveDocumentReplaceJobError(MetadataStoreError):
    def __init__(self, document_id: str, existing_job_id: str):
        self.document_id = document_id
        self.existing_job_id = existing_job_id
        super().__init__(f"Document '{document_id}' already has an active replace job")


class DocumentNotParsedError(MetadataStoreError):
    def __init__(self, document_id: str, current_status: str):
        self.document_id = document_id
        self.current_status = current_status
        super().__init__(
            f"Document '{document_id}' must be parsed before build (current status: {current_status})"
        )


class IdempotencyKeyConflictError(MetadataStoreError):
    def __init__(self, idempotency_key: str):
        self.idempotency_key = idempotency_key
        super().__init__(
            f"Idempotency key '{idempotency_key}' is already used for a different request"
        )


class DuplicateDocumentSourceKeyError(MetadataStoreError):
    def __init__(self, kb_id: str, source_key: str, existing_document_id: str):
        self.kb_id = kb_id
        self.source_key = source_key
        self.existing_document_id = existing_document_id
        super().__init__(
            f"Source key '{source_key}' already belongs to document '{existing_document_id}'"
        )


class InvalidTenantUserKBOverrideError(MetadataStoreError, ValueError):
    pass


@dataclass(slots=True)
class DocumentRecord:
    id: str
    kb_id: str
    workspace: str
    lightrag_doc_id: str | None
    source_type: str
    source_name: str
    source_uri: str
    source_hash: str
    content_type: str | None
    size_bytes: int
    parser_hash: str | None
    index_hash: str | None
    status: str
    enabled: bool
    archived: bool
    chunks_count: int | None
    entity_count: int | None
    relation_count: int | None
    error_code: str | None
    error_message: str | None
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    deleted_at: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "DocumentRecord":
        return cls(
            id=str(row["id"]),
            kb_id=str(row["kb_id"]),
            workspace=str(row["workspace"]),
            lightrag_doc_id=row["lightrag_doc_id"],
            source_type=str(row["source_type"]),
            source_name=str(row["source_name"]),
            source_uri=str(row["source_uri"]),
            source_hash=str(row["source_hash"]),
            content_type=row["content_type"],
            size_bytes=int(row["size_bytes"]),
            parser_hash=row["parser_hash"],
            index_hash=row["index_hash"],
            status=str(row["status"]),
            enabled=bool(row["enabled"]),
            archived=bool(row["archived"]),
            chunks_count=row["chunks_count"],
            entity_count=row["entity_count"],
            relation_count=row["relation_count"],
            error_code=row["error_code"],
            error_message=row["error_message"],
            metadata=_loads_json_object(row["metadata_json"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            deleted_at=row["deleted_at"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class JobRecord:
    id: str
    kb_id: str
    workspace: str
    batch_id: str | None
    document_id: str | None
    job_type: str
    status: str
    stage: str | None
    progress: float
    total_items: int
    completed_items: int
    failed_items: int
    idempotency_key: str | None
    config_version_id: str | None
    config_hash: str | None
    retry_count: int
    max_retries: int
    payload: dict[str, Any]
    result: dict[str, Any] | None
    error_code: str | None
    error_message: str | None
    created_at: str
    updated_at: str
    queued_at: str | None
    started_at: str | None
    finished_at: str | None
    cancelled_at: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "JobRecord":
        result_json = row["result_json"]
        return cls(
            id=str(row["id"]),
            kb_id=str(row["kb_id"]),
            workspace=str(row["workspace"]),
            batch_id=row["batch_id"],
            document_id=row["document_id"],
            job_type=str(row["job_type"]),
            status=str(row["status"]),
            stage=row["stage"],
            progress=float(row["progress"]),
            total_items=int(row["total_items"]),
            completed_items=int(row["completed_items"]),
            failed_items=int(row["failed_items"]),
            idempotency_key=row["idempotency_key"],
            config_version_id=row["config_version_id"],
            config_hash=row["config_hash"],
            retry_count=int(row["retry_count"]),
            max_retries=int(row["max_retries"]),
            payload=_loads_json_object(row["payload_json"]),
            result=_loads_optional_json_object(result_json),
            error_code=row["error_code"],
            error_message=row["error_message"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            queued_at=row["queued_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            cancelled_at=row["cancelled_at"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ArtifactRecord:
    id: str
    kb_id: str
    workspace: str
    document_id: str
    artifact_type: str
    uri: str
    checksum: str | None
    size_bytes: int | None
    metadata: dict[str, Any]
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ArtifactRecord":
        return cls(
            id=str(row["id"]),
            kb_id=str(row["kb_id"]),
            workspace=str(row["workspace"]),
            document_id=str(row["document_id"]),
            artifact_type=str(row["artifact_type"]),
            uri=str(row["uri"]),
            checksum=row["checksum"],
            size_bytes=row["size_bytes"],
            metadata=_loads_json_object(row["metadata_json"]),
            created_at=str(row["created_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ConfigVersionRecord:
    id: str
    kb_id: str
    workspace: str
    version: int
    config: dict[str, Any]
    parser_hash: str | None
    index_hash: str | None
    query_hash: str | None
    created_at: str
    activated_at: str | None
    created_by: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ConfigVersionRecord":
        return cls(
            id=str(row["id"]),
            kb_id=str(row["kb_id"]),
            workspace=str(row["workspace"]),
            version=int(row["version"]),
            config=_loads_json_object(row["config_json"]),
            parser_hash=row["parser_hash"],
            index_hash=row["index_hash"],
            query_hash=row["query_hash"],
            created_at=str(row["created_at"]),
            activated_at=row["activated_at"],
            created_by=row["created_by"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EnterpriseUserRecord:
    id: str
    username: str
    password_hash: str
    system_role: str
    status: str
    tenant_id: str | None
    can_create_kb: bool
    can_use_bypass_query: bool
    token_version: int
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    # Capability to delete documents uploaded by other users. Declared last with
    # a default so old PostgreSQL JSONB rows (predating the column) deserialize
    # and existing keyword constructions keep working.
    can_delete_documents: bool = False
    # Capability to use the high-cost server-side Agent query mode.
    can_use_agent_query: bool = False
    # New users are denied original-file downloads until explicitly granted.
    # Backend migrations preserve access for records created before this field.
    can_download_files: bool = False

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "EnterpriseUserRecord":
        return cls(
            id=str(row["id"]),
            username=str(row["username"]),
            password_hash=str(row["password_hash"]),
            system_role=str(row["system_role"]),
            status=str(row["status"]),
            tenant_id=row["tenant_id"],
            can_create_kb=bool(row["can_create_kb"]),
            can_use_bypass_query=bool(row["can_use_bypass_query"]),
            token_version=int(row["token_version"]),
            metadata=_loads_json_object(row["metadata_json"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            can_delete_documents=bool(row["can_delete_documents"]),
            can_use_agent_query=bool(row["can_use_agent_query"]),
            can_download_files=bool(row["can_download_files"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _assert_enterprise_user_write_preconditions(
    user: EnterpriseUserRecord,
    current_user: EnterpriseUserRecord | None,
    *,
    expected_updated_at: Any,
    expected_token_version: Any,
    expected_tenant_id: Any,
    allow_tenant_change: bool,
) -> None:
    """Validate the revision used by an enterprise-user whole-record write.

    Creation intentionally has no expected revision. Every existing-user write
    must provide all three values from the snapshot that produced ``user``.
    Keeping the expected revision separate from the candidate's new
    ``updated_at`` prevents a stale whole-record replay from restoring security
    fields changed by another request.
    """

    expectation_values = {
        "updated_at": expected_updated_at,
        "token_version": expected_token_version,
        "tenant_id": expected_tenant_id,
    }
    supplied = {
        key: value is not _EXPECTATION_UNSET
        for key, value in expectation_values.items()
    }
    display_expected = {
        key: value if supplied[key] else "<missing>"
        for key, value in expectation_values.items()
    }

    if current_user is None:
        if any(supplied.values()):
            raise MetadataConflictError(
                "enterprise_user",
                user.id,
                expected=display_expected,
                current={"exists": False},
            )
        if not allow_tenant_change and user.tenant_id is not None:
            raise MetadataConflictError(
                "enterprise_user",
                user.id,
                expected={"tenant_id": None},
                current={"candidate_tenant_id": user.tenant_id},
            )
        return

    current = {
        "updated_at": current_user.updated_at,
        "token_version": current_user.token_version,
        "tenant_id": current_user.tenant_id,
    }
    if not all(supplied.values()) or expectation_values != current:
        raise MetadataConflictError(
            "enterprise_user",
            user.id,
            expected=display_expected,
            current=current,
        )
    if not allow_tenant_change and user.tenant_id != current_user.tenant_id:
        raise MetadataConflictError(
            "enterprise_user",
            user.id,
            expected={"tenant_id": current_user.tenant_id},
            current={"candidate_tenant_id": user.tenant_id},
        )
    if user.token_version < current_user.token_version:
        raise MetadataConflictError(
            "enterprise_user",
            user.id,
            expected={"minimum_token_version": current_user.token_version},
            current={"candidate_token_version": user.token_version},
        )


@dataclass(slots=True)
class KBLifecycleRecord:
    kb_id: str
    generation: str
    state: Literal["active", "deleting", "deleted"]
    activated_at: str
    deleted_at: str | None
    updated_at: str
    delete_job_id: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "KBLifecycleRecord":
        columns = set(row.keys())
        return cls(
            kb_id=str(row["kb_id"]),
            generation=str(row["generation"]),
            state=str(row["state"]),  # type: ignore[arg-type]
            activated_at=str(row["activated_at"]),
            deleted_at=row["deleted_at"],
            updated_at=str(row["updated_at"]),
            delete_job_id=(
                row["delete_job_id"] if "delete_job_id" in columns else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EnterpriseUserKBQuerySettingsRecord:
    user_id: str
    kb_id: str
    user_prompt: str
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "EnterpriseUserKBQuerySettingsRecord":
        return cls(
            user_id=str(row["user_id"]),
            kb_id=str(row["kb_id"]),
            user_prompt=str(row["user_prompt"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChatProjectRecord:
    """Per-user chat project (top-level container of chat sessions)."""

    id: str
    user_id: str
    name: str
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ChatProjectRecord":
        return cls(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            name=str(row["name"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChatSessionRecord:
    """Chat session under a per-user chat project."""

    id: str
    project_id: str
    user_id: str
    name: str
    created_at: str
    updated_at: str
    # Conversation rounds sent to the LLM per query (-1 = full history).
    # Declared last with a default so legacy PostgreSQL JSONB rows written
    # before this field existed still deserialize.
    context_rounds: int = 1

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ChatSessionRecord":
        return cls(
            id=str(row["id"]),
            project_id=str(row["project_id"]),
            user_id=str(row["user_id"]),
            name=str(row["name"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            context_rounds=int(row["context_rounds"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChatMessageRecord:
    """Persisted chat message inside a session (server-side history sync).

    ``seq`` is a per-session monotonic sequence assigned by the store at
    insert time; message ordering is ``(seq, id)`` ascending.
    """

    id: str
    session_id: str
    project_id: str
    user_id: str
    role: str
    content: str
    metadata: dict[str, Any]
    seq: int
    created_at: str
    # Admission fields are nullable by design. Legacy and feature-off messages
    # remain source data but are not silently imported into memory rebuilds.
    append_batch_id: str | None = None
    project_event_seq: int | None = None
    memory_reference_time: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ChatMessageRecord":
        columns = set(row.keys())
        return cls(
            id=str(row["id"]),
            session_id=str(row["session_id"]),
            project_id=str(row["project_id"]),
            user_id=str(row["user_id"]),
            role=str(row["role"]),
            content=str(row["content"]),
            metadata=_loads_json_object(row["metadata_json"]),
            seq=int(row["seq"]),
            created_at=str(row["created_at"]),
            append_batch_id=(
                row["append_batch_id"] if "append_batch_id" in columns else None
            ),
            project_event_seq=(
                int(row["project_event_seq"])
                if "project_event_seq" in columns
                and row["project_event_seq"] is not None
                else None
            ),
            memory_reference_time=(
                row["memory_reference_time"]
                if "memory_reference_time" in columns
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChatMemoryEpisodeRecord:
    """Mapping between a graphiti memory episode and the chat messages it
    distilled (docs/ChatMemory-zh.md).

    ``first_seq``/``last_seq`` retain compatibility with the legacy session
    watermark. Generation-aware producers also persist the admitted append
    identity. The same producer event/batch may be replayed into multiple
    generations, while each generation admits at most one logical mapping for
    that append batch. ``noop_*`` uuids remain valid legacy mappings.
    """

    episode_uuid: str
    session_id: str
    project_id: str
    user_id: str
    first_seq: int
    last_seq: int
    created_at: str
    event_id: str | None = None
    generation: int | None = None
    graph_group_id: str | None = None
    append_batch_id: str | None = None
    project_event_seq: int | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ChatMemoryEpisodeRecord":
        columns = set(row.keys())
        return cls(
            episode_uuid=str(row["episode_uuid"]),
            session_id=str(row["session_id"]),
            project_id=str(row["project_id"]),
            user_id=str(row["user_id"]),
            first_seq=int(row["first_seq"]),
            last_seq=int(row["last_seq"]),
            created_at=str(row["created_at"]),
            event_id=row["event_id"] if "event_id" in columns else None,
            generation=(
                int(row["generation"])
                if "generation" in columns and row["generation"] is not None
                else None
            ),
            graph_group_id=(
                row["graph_group_id"] if "graph_group_id" in columns else None
            ),
            append_batch_id=(
                row["append_batch_id"] if "append_batch_id" in columns else None
            ),
            project_event_seq=(
                int(row["project_event_seq"])
                if "project_event_seq" in columns
                and row["project_event_seq"] is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChatMemoryGroupRecord:
    """Durable logical Chat Memory group state for one user/project."""

    user_id: str
    project_id: str
    logical_group_id: str
    active_generation: int | None
    desired_generation: int
    next_event_seq: int
    last_reference_time: str | None
    state: ChatMemoryGroupState
    state_version: int
    active_config_fingerprint: str | None
    desired_config_fingerprint: str
    active_rebuild_event_id: str | None
    last_success_at: str | None
    last_error_code: str | None
    last_error_message: str | None
    last_error_at: str | None
    created_at: str
    updated_at: str
    deleted_at: str | None
    record_version: int = CHAT_MEMORY_RECORD_VERSION
    active_graph_store_fingerprint: str | None = None
    desired_graph_store_fingerprint: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ChatMemoryGroupRecord":
        columns = set(row.keys())
        return cls(
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
            last_reference_time=row["last_reference_time"],
            state=str(row["state"]),  # type: ignore[arg-type]
            state_version=int(row["state_version"]),
            active_config_fingerprint=row["active_config_fingerprint"],
            desired_config_fingerprint=str(row["desired_config_fingerprint"]),
            active_rebuild_event_id=row["active_rebuild_event_id"],
            last_success_at=row["last_success_at"],
            last_error_code=row["last_error_code"],
            last_error_message=row["last_error_message"],
            last_error_at=row["last_error_at"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            deleted_at=row["deleted_at"],
            record_version=int(row["record_version"]),
            active_graph_store_fingerprint=(
                row["active_graph_store_fingerprint"]
                if "active_graph_store_fingerprint" in columns
                else row["active_config_fingerprint"]
            ),
            desired_graph_store_fingerprint=(
                str(row["desired_graph_store_fingerprint"])
                if "desired_graph_store_fingerprint" in columns
                else str(row["desired_config_fingerprint"])
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _chat_memory_existing_graph_store_fingerprint(
    group: ChatMemoryGroupRecord,
    fallback_graph_store_fingerprint: str,
) -> str:
    """Bind destructive work to an existing logical group's graph store."""

    return _validate_chat_memory_fingerprint(
        group.desired_graph_store_fingerprint
        or group.active_graph_store_fingerprint
        or fallback_graph_store_fingerprint
    )


def _chat_memory_graph_store_migration_conflict(
    logical_group_id: str,
    required_graph_store_fingerprint: str,
    observed_graph_store_fingerprints: Sequence[str],
) -> MetadataConflictError:
    """Report that changing or reconciling graph stores needs explicit migration."""

    return MetadataConflictError(
        "chat_memory_graph_store",
        logical_group_id,
        expected={
            "graph_store_fingerprints": (required_graph_store_fingerprint,),
        },
        current={
            "error_code": _CHAT_MEMORY_GRAPH_STORE_MIGRATION_REQUIRED,
            "graph_store_fingerprints": tuple(
                sorted(set(observed_graph_store_fingerprints))
            ),
        },
    )


@dataclass(slots=True)
class ChatMemoryGenerationRecord:
    """Inventory row for one physical generation of a logical group."""

    user_id: str
    project_id: str
    generation: int
    graph_group_id: str
    config_fingerprint: str
    state: ChatMemoryGenerationState
    snapshot_cutoff: int | None
    replay_batch_count: int | None
    replay_message_count: int | None
    replay_byte_count: int | None
    clear_attempt_no: int
    clear_started_at: str | None
    created_at: str
    updated_at: str
    activated_at: str | None
    cleared_at: str | None
    last_error_code: str | None
    last_error_message: str | None
    last_error_at: str | None
    snapshot_digest: str | None = None
    record_version: int = CHAT_MEMORY_RECORD_VERSION
    graph_store_fingerprint: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ChatMemoryGenerationRecord":
        columns = set(row.keys())
        return cls(
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
            clear_started_at=row["clear_started_at"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            activated_at=row["activated_at"],
            cleared_at=row["cleared_at"],
            last_error_code=row["last_error_code"],
            last_error_message=row["last_error_message"],
            last_error_at=row["last_error_at"],
            snapshot_digest=(
                row["snapshot_digest"] if "snapshot_digest" in columns else None
            ),
            record_version=int(row["record_version"]),
            graph_store_fingerprint=(
                str(row["graph_store_fingerprint"])
                if "graph_store_fingerprint" in columns
                else str(row["config_fingerprint"])
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChatMemoryOutboxEventRecord:
    """Durable, generation-fenced Chat Memory mutation event."""

    event_id: str
    deterministic_key: str
    user_id: str
    project_id: str
    event_seq: int
    generation: int
    graph_group_id: str
    config_fingerprint: str
    event_type: ChatMemoryEventType
    status: ChatMemoryEventStatus
    available_at: str
    attempt_no: int
    created_at: str
    updated_at: str
    source_session_id: str | None = None
    append_batch_id: str | None = None
    first_seq: int | None = None
    last_seq: int | None = None
    snapshot_cutoff: int | None = None
    snapshot_batch_count: int | None = None
    snapshot_message_count: int | None = None
    snapshot_byte_count: int | None = None
    snapshot_digest: str | None = None
    claim_token: str | None = None
    claimed_by: str | None = None
    claimed_at: str | None = None
    side_effect_started_at: str | None = None
    side_effect_state_version: int | None = None
    completed_at: str | None = None
    superseded_by_event_id: str | None = None
    last_error_code: str | None = None
    last_error_message: str | None = None
    last_error_at: str | None = None
    actor_user_id: str | None = None
    actor_tenant_id: str | None = None
    target_user_id: str | None = None
    target_project_id: str | None = None
    target_session_id: str | None = None
    target_message_id: str | None = None
    record_version: int = CHAT_MEMORY_RECORD_VERSION
    graph_store_fingerprint: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ChatMemoryOutboxEventRecord":
        columns = set(row.keys())
        return cls(
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
            available_at=str(row["available_at"]),
            attempt_no=int(row["attempt_no"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
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
                row["snapshot_digest"] if "snapshot_digest" in columns else None
            ),
            claim_token=row["claim_token"],
            claimed_by=row["claimed_by"],
            claimed_at=row["claimed_at"],
            side_effect_started_at=row["side_effect_started_at"],
            side_effect_state_version=(
                int(row["side_effect_state_version"])
                if row["side_effect_state_version"] is not None
                else None
            ),
            completed_at=row["completed_at"],
            superseded_by_event_id=row["superseded_by_event_id"],
            last_error_code=row["last_error_code"],
            last_error_message=row["last_error_message"],
            last_error_at=row["last_error_at"],
            actor_user_id=row["actor_user_id"],
            actor_tenant_id=row["actor_tenant_id"],
            target_user_id=row["target_user_id"],
            target_project_id=row["target_project_id"],
            target_session_id=row["target_session_id"],
            target_message_id=row["target_message_id"],
            record_version=int(row["record_version"]),
            graph_store_fingerprint=(
                str(row["graph_store_fingerprint"])
                if "graph_store_fingerprint" in columns
                else str(row["config_fingerprint"])
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChatMemoryExecutionState:
    group: ChatMemoryGroupRecord
    event: ChatMemoryOutboxEventRecord
    generation: ChatMemoryGenerationRecord


@dataclass(slots=True, frozen=True)
class ChatMemoryReadToken:
    """Atomic logical/physical identity used to fence Chat Memory reads."""

    user_id: str
    project_id: str
    state: ChatMemoryGroupState
    state_version: int
    active_generation: int | None
    active_config_fingerprint: str | None
    active_graph_store_fingerprint: str | None
    graph_group_id: str | None
    generation_state: ChatMemoryGenerationState | None


@dataclass(slots=True)
class ChatMemoryOutboxStats:
    pending: int
    running: int
    retry_wait: int
    dead_letter: int
    oldest_available_at: str | None
    oldest_lag_seconds: float


@dataclass(slots=True)
class ChatMemoryRebuildSnapshot:
    event_id: str
    user_id: str
    project_id: str
    generation: int
    graph_group_id: str
    config_fingerprint: str
    group_state_version: int
    snapshot_cutoff: int
    replay_batches: list[ChatMemoryReplayBatch]
    batch_count: int
    message_count: int
    byte_count: int
    snapshot_digest: str | None = None
    ingest_max_chars: int = CHAT_MEMORY_DEFAULT_INGEST_MAX_CHARS
    graph_store_fingerprint: str | None = None

    @property
    def state_version(self) -> int:
        """Compatibility alias for the captured logical-group state version."""

        return self.group_state_version


@dataclass(slots=True)
class ChatMemoryReplayMappingInput:
    append_batch_id: str
    project_event_seq: int
    session_id: str
    first_seq: int
    last_seq: int
    episode_uuid: str


@dataclass(slots=True)
class ChatMemoryRebuildTargetSet:
    event_id: str
    user_id: str
    project_id: str
    logical_group_id: str
    group_ids: tuple[str, ...]


@dataclass(slots=True)
class ChatMemoryPurgeTargetSet:
    event_id: str
    user_id: str
    project_id: str
    logical_group_id: str
    group_ids: tuple[str, ...]


@dataclass(slots=True)
class ChatMemoryReplayBatch:
    """One admitted append boundary returned for generation replay."""

    append_batch_id: str
    project_event_seq: int
    memory_reference_time: str
    session_id: str
    messages: list[ChatMessageRecord]


def _validate_chat_memory_ingest_max_chars(value: int) -> int:
    if isinstance(value, bool):
        raise ValueError("Chat Memory ingest_max_chars must be a non-negative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError("Chat Memory ingest_max_chars must be non-negative")
    return normalized


def _chat_memory_admitted_message_content(
    message: ChatMessageRecord,
) -> str | None:
    """Apply the fixed, shared Chat Memory admission policy.

    Non-empty user messages are admitted. Assistant messages are admitted only
    when their JSON metadata contains the literal boolean
    ``memory_eligible=true``. Other roles and blank content are not admitted.
    """

    content = message.content.strip()
    if not content:
        return None
    if message.role == "user":
        return content
    if (
        message.role == "assistant"
        and message.metadata.get("memory_eligible") is True
    ):
        return content
    return None


def _chat_memory_canonical_episode_payload(
    messages: Sequence[ChatMessageRecord],
    *,
    ingest_max_chars: int,
) -> dict[str, Any]:
    """Return the normalized final episode payload for one admitted batch."""

    max_chars = _validate_chat_memory_ingest_max_chars(ingest_max_chars)
    admitted: list[dict[str, Any]] = []
    lines: list[str] = []
    for message in messages:
        content = _chat_memory_admitted_message_content(message)
        if content is None:
            continue
        normalized_content = content
        if len(normalized_content) > max_chars:
            normalized_content = (
                normalized_content[:max_chars] + _CHAT_MEMORY_TRUNCATION_MARKER
            )
        admitted.append(
            {
                "id": message.id,
                "seq": int(message.seq),
                "role": message.role,
                "content": normalized_content,
            }
        )
        lines.append(f"{message.role}: {normalized_content}")
    return {
        "admission_policy_version": _CHAT_MEMORY_ADMISSION_POLICY_VERSION,
        "ingest_max_chars": max_chars,
        "messages": admitted,
        "episode_body": "\n".join(lines),
    }


def _chat_memory_canonical_snapshot_manifest(
    replay_batches: Sequence[ChatMemoryReplayBatch],
    *,
    ingest_max_chars: int,
) -> dict[str, Any]:
    """Build the ordered, versioned manifest attested by rebuild snapshots."""

    max_chars = _validate_chat_memory_ingest_max_chars(ingest_max_chars)
    batches: list[dict[str, Any]] = []
    for batch in replay_batches:
        message_manifest: list[dict[str, Any]] = []
        for message in batch.messages:
            metadata = message.metadata
            if not isinstance(metadata, dict):
                raise MetadataStoreError("Chat Memory message metadata must be an object")
            message_manifest.append(
                {
                    "id": message.id,
                    "seq": int(message.seq),
                    "role": message.role,
                    "content": message.content,
                    "admission_metadata": {
                        "memory_eligible_present": "memory_eligible" in metadata,
                        "memory_eligible": metadata.get("memory_eligible"),
                    },
                }
            )
        batches.append(
            {
                "project_event_seq": int(batch.project_event_seq),
                "append_batch_id": batch.append_batch_id,
                "session_id": batch.session_id,
                "memory_reference_time": batch.memory_reference_time,
                "messages": message_manifest,
                "episode_payload": _chat_memory_canonical_episode_payload(
                    batch.messages,
                    ingest_max_chars=max_chars,
                ),
            }
        )
    return {
        "snapshot_manifest_version": CHAT_MEMORY_SNAPSHOT_DIGEST_VERSION,
        "ingest_max_chars": max_chars,
        "batches": batches,
    }


def _chat_memory_snapshot_digest(
    replay_batches: Sequence[ChatMemoryReplayBatch],
    *,
    ingest_max_chars: int,
) -> str:
    """Return a versioned SHA-256 digest for an ordered replay manifest."""

    manifest = _chat_memory_canonical_snapshot_manifest(
        replay_batches,
        ingest_max_chars=ingest_max_chars,
    )
    try:
        canonical = json.dumps(
            manifest,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise MetadataStoreError(
            "Chat Memory rebuild snapshot manifest is not JSON-canonicalizable"
        ) from exc
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"chat-memory-snapshot:v{CHAT_MEMORY_SNAPSHOT_DIGEST_VERSION}:sha256:{digest}"


def _chat_memory_replay_batches_from_messages(
    messages: Sequence[ChatMessageRecord],
) -> list[ChatMemoryReplayBatch]:
    """Materialize validated replay batches from database-ordered messages."""

    grouped: dict[tuple[int, str], list[ChatMessageRecord]] = {}
    event_batch_ids: dict[int, str] = {}
    for message in messages:
        if message.project_event_seq is None or message.append_batch_id is None:
            raise MetadataStoreError(
                "Chat Memory replay message is missing admission identity"
            )
        previous_batch_id = event_batch_ids.setdefault(
            message.project_event_seq, message.append_batch_id
        )
        if previous_batch_id != message.append_batch_id:
            raise MetadataStoreError(
                "One Chat Memory project event sequence maps to multiple batches"
            )
        grouped.setdefault(
            (message.project_event_seq, message.append_batch_id), []
        ).append(message)

    replay_batches: list[ChatMemoryReplayBatch] = []
    for (project_event_seq, append_batch_id), batch_messages in grouped.items():
        session_ids = {message.session_id for message in batch_messages}
        reference_times = {
            message.memory_reference_time for message in batch_messages
        }
        if (
            len(session_ids) != 1
            or len(reference_times) != 1
            or None in reference_times
        ):
            raise MetadataStoreError(
                "Chat Memory replay batch identity is internally inconsistent"
            )
        replay_batches.append(
            ChatMemoryReplayBatch(
                append_batch_id=append_batch_id,
                project_event_seq=project_event_seq,
                memory_reference_time=str(next(iter(reference_times))),
                session_id=next(iter(session_ids)),
                messages=list(batch_messages),
            )
        )
    return replay_batches


def _validate_chat_memory_ingest_source_batch(
    event: ChatMemoryOutboxEventRecord,
    messages: Sequence[ChatMessageRecord],
) -> None:
    """Fail closed unless rows still exactly represent the ingest event batch."""

    if (
        event.append_batch_id is None
        or event.source_session_id is None
        or event.first_seq is None
        or event.last_seq is None
    ):
        raise MetadataStoreError("Ingest event is missing source batch identity")
    expected_seqs = list(range(event.first_seq, event.last_seq + 1))
    current_seqs = [message.seq for message in messages]
    reference_times = {
        message.memory_reference_time for message in messages
    }
    recomputed_batch_id = _chat_memory_append_batch_id(
        user_id=event.user_id,
        project_id=event.project_id,
        session_id=event.source_session_id,
        event_seq=event.event_seq,
        message_ids=[message.id for message in messages],
    )
    identity_matches = all(
        message.user_id == event.user_id
        and message.project_id == event.project_id
        and message.session_id == event.source_session_id
        and message.append_batch_id == event.append_batch_id
        and message.project_event_seq == event.event_seq
        and message.memory_reference_time is not None
        for message in messages
    ) and len(reference_times) == 1
    identity_matches = (
        identity_matches and recomputed_batch_id == event.append_batch_id
    )
    if current_seqs != expected_seqs or not identity_matches:
        raise MetadataConflictError(
            "chat_memory_ingest_source_batch",
            event.event_id,
            expected={
                "session_id": event.source_session_id,
                "append_batch_id": event.append_batch_id,
                "project_event_seq": event.event_seq,
                "seqs": expected_seqs,
            },
            current={
                "message_count": len(messages),
                "seqs": current_seqs,
                "identity_matches": identity_matches,
                "recomputed_append_batch_id": recomputed_batch_id,
            },
        )


def _chat_memory_replay_snapshot_metrics(
    replay_batches: Sequence[ChatMemoryReplayBatch],
) -> tuple[int, int, int]:
    """Return batch/message/content UTF-8 byte counts for a complete replay."""

    message_count = sum(len(batch.messages) for batch in replay_batches)
    byte_count = sum(
        len(message.content.encode("utf-8"))
        for batch in replay_batches
        for message in batch.messages
    )
    return len(replay_batches), message_count, byte_count


def _chat_memory_noop_episode_uuid(
    *, event_id: str, generation: int, append_batch_id: str
) -> str:
    """Build a deterministic generation-aware mapping id for a no-op ingest."""

    digest = hashlib.sha256(
        f"{event_id}\0{int(generation)}\0{append_batch_id}".encode("utf-8")
    ).hexdigest()
    return f"noop_{digest[:32]}"


def _normalize_chat_memory_group_ids(group_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(group_ids, str):
        raise ValueError("Chat Memory graph group ids must be a sequence")
    normalized: set[str] = set()
    for group_id in group_ids:
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError("Chat Memory graph group ids must be non-empty strings")
        normalized.add(group_id.strip())
    return tuple(sorted(normalized))


@dataclass(slots=True)
class ChatMemoryBacklogItem:
    """One session whose persisted messages run ahead of the memory watermark."""

    user_id: str
    project_id: str
    session_id: str
    ingested_seq: int
    max_seq: int


@dataclass(slots=True)
class EnterpriseAPIKeyRecord:
    id: str
    name: str
    key_hash: str
    key_preview: str
    status: str
    created_by: str | None
    tenant_id: str | None
    scopes: dict[str, Any]
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    last_used_at: str | None
    revoked_at: str | None
    revoked_by: str | None
    expires_at: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "EnterpriseAPIKeyRecord":
        columns = set(row.keys())
        return cls(
            id=str(row["id"]),
            name=str(row["name"]),
            key_hash=str(row["key_hash"]),
            key_preview=str(row["key_preview"]),
            status=str(row["status"]),
            created_by=row["created_by"],
            tenant_id=row["tenant_id"],
            scopes=_loads_json_object(row["scopes_json"]),
            metadata=_loads_json_object(row["metadata_json"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            last_used_at=row["last_used_at"],
            revoked_at=row["revoked_at"],
            revoked_by=row["revoked_by"],
            expires_at=row["expires_at"] if "expires_at" in columns else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EnterpriseInvitationRecord:
    id: str
    token_hash: str
    token_preview: str
    status: str
    created_by: str | None
    expires_at: str | None
    used_by: str | None
    used_at: str | None
    metadata: dict[str, Any]
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "EnterpriseInvitationRecord":
        return cls(
            id=str(row["id"]),
            token_hash=str(row["token_hash"]),
            token_preview=str(row["token_preview"]),
            status=str(row["status"]),
            created_by=row["created_by"],
            expires_at=row["expires_at"],
            used_by=row["used_by"],
            used_at=row["used_at"],
            metadata=_loads_json_object(row["metadata_json"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class KBACLRecord:
    kb_id: str
    user_id: str
    role: str
    granted_by: str | None
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "KBACLRecord":
        return cls(
            kb_id=str(row["kb_id"]),
            user_id=str(row["user_id"]),
            role=str(row["role"]),
            granted_by=row["granted_by"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EnterpriseTenantMembershipRecord:
    tenant_id: str
    user_id: str
    role: str
    granted_by: str | None
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "EnterpriseTenantMembershipRecord":
        return cls(
            tenant_id=str(row["tenant_id"]),
            user_id=str(row["user_id"]),
            role=str(row["role"]),
            granted_by=row["granted_by"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _assert_enterprise_user_membership_precondition(
    user_id: str,
    current_memberships: Sequence[EnterpriseTenantMembershipRecord],
    *,
    expected_membership: Any,
) -> None:
    """Compare the unique canonical membership with a captured snapshot."""

    if expected_membership is _EXPECTATION_UNSET:
        return
    if expected_membership is not None and not isinstance(
        expected_membership, EnterpriseTenantMembershipRecord
    ):
        raise MetadataStoreError(
            "Expected tenant membership must be a membership record or null"
        )
    expected = (
        [] if expected_membership is None else [expected_membership.to_dict()]
    )
    current = [membership.to_dict() for membership in current_memberships]
    if current != expected:
        raise MetadataConflictError(
            "enterprise_user_membership",
            user_id,
            expected={"memberships": expected},
            current={"memberships": current},
        )


@dataclass(slots=True)
class EnterpriseTenantKBACLRecord:
    tenant_id: str
    kb_id: str
    role: str
    granted_by: str | None
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "EnterpriseTenantKBACLRecord":
        return cls(
            tenant_id=str(row["tenant_id"]),
            kb_id=str(row["kb_id"]),
            role=str(row["role"]),
            granted_by=row["granted_by"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EnterpriseTenantUserKBOverrideRecord:
    tenant_id: str
    kb_id: str
    user_id: str
    effect: TenantUserKBOverrideEffect
    role: str | None
    granted_by: str | None
    created_at: str
    updated_at: str

    @classmethod
    def from_row(
        cls, row: sqlite3.Row
    ) -> "EnterpriseTenantUserKBOverrideRecord":
        return cls(
            tenant_id=str(row["tenant_id"]),
            kb_id=str(row["kb_id"]),
            user_id=str(row["user_id"]),
            effect=str(row["effect"]),  # type: ignore[arg-type]
            role=row["role"],
            granted_by=row["granted_by"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EnterpriseTenantRecord:
    id: str
    name: str
    description: str | None
    status: str
    metadata: dict[str, Any]
    created_by: str | None
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "EnterpriseTenantRecord":
        return cls(
            id=str(row["id"]),
            name=str(row["name"]),
            description=row["description"],
            status=str(row["status"]),
            metadata=_loads_json_object(row["metadata_json"]),
            created_by=row["created_by"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AuditEventRecord:
    id: str
    event_type: str
    actor_user_id: str | None
    target_type: str | None
    target_id: str | None
    metadata: dict[str, Any]
    created_at: str
    # Event-time snapshot. Legacy events remain unscoped (None).
    actor_tenant_id: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "AuditEventRecord":
        return cls(
            id=str(row["id"]),
            event_type=str(row["event_type"]),
            actor_user_id=row["actor_user_id"],
            target_type=row["target_type"],
            target_id=row["target_id"],
            metadata=_loads_json_object(row["metadata_json"]),
            created_at=str(row["created_at"]),
            actor_tenant_id=row["actor_tenant_id"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EnterprisePersonRecord:
    """A natural-person identity that can be linked to multiple accounts."""

    id: str
    status: str
    auth_epoch: int
    metadata: dict[str, Any]
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "EnterprisePersonRecord":
        return cls(
            id=str(row["id"]),
            status=str(row["status"]),
            auth_epoch=int(row["auth_epoch"]),
            metadata=_loads_json_object(row["metadata_json"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EnterprisePersonCredentialRecord:
    """Strict bcrypt credential for a person (no legacy plaintext fallback)."""

    id: str
    person_id: str
    credential_type: str
    algorithm: str
    password_hash: str
    status: str
    failed_count: int
    locked_until: str | None
    last_used_at: str | None
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "EnterprisePersonCredentialRecord":
        return cls(
            id=str(row["id"]),
            person_id=str(row["person_id"]),
            credential_type=str(row["credential_type"]),
            algorithm=str(row["algorithm"]),
            password_hash=str(row["password_hash"]),
            status=str(row["status"]),
            failed_count=int(row["failed_count"]),
            locked_until=row["locked_until"],
            last_used_at=row["last_used_at"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EnterprisePersonEnrollmentGrantRecord:
    """One-time enrollment grant signed by a super admin."""

    id: str
    account_id: str
    token_hash: str
    status: str
    created_by: str | None
    consumed_by_person: str | None
    expires_at: str
    created_at: str
    updated_at: str
    consumed_at: str | None

    @classmethod
    def from_row(
        cls, row: sqlite3.Row
    ) -> "EnterprisePersonEnrollmentGrantRecord":
        return cls(
            id=str(row["id"]),
            account_id=str(row["account_id"]),
            token_hash=str(row["token_hash"]),
            status=str(row["status"]),
            created_by=row["created_by"],
            consumed_by_person=row["consumed_by_person"],
            expires_at=str(row["expires_at"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            consumed_at=row["consumed_at"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EnterprisePersonAccountLinkRecord:
    """A binding between a person and an enterprise account."""

    id: str
    person_id: str
    account_id: str
    status: str
    bound_by: str | None
    bound_at: str | None
    confirmed_by_person_at: str | None
    revoked_by: str | None
    revoked_at: str | None
    reason: str | None
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "EnterprisePersonAccountLinkRecord":
        return cls(
            id=str(row["id"]),
            person_id=str(row["person_id"]),
            account_id=str(row["account_id"]),
            status=str(row["status"]),
            bound_by=row["bound_by"],
            bound_at=row["bound_at"],
            confirmed_by_person_at=row["confirmed_by_person_at"],
            revoked_by=row["revoked_by"],
            revoked_at=row["revoked_at"],
            reason=row["reason"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EnterprisePersonLoginSessionRecord:
    """A person login session; identity proven via sid + session_epoch."""

    id: str
    person_id: str
    active_account_id: str | None
    status: str
    person_epoch: int
    session_epoch: int
    absolute_expires_at: str
    created_at: str
    last_seen_at: str | None
    revoked_at: str | None
    # Snapshot of the active account's token_version at issue/switch time.
    # account-access validation compares this against the live account row so
    # that a password reset (which bumps token_version without changing status)
    # invalidates outstanding v2 account-access tokens. session-control skips
    # this check (doc 6.4: reset still allows list/switch/logout).
    account_token_version: int = 0

    @classmethod
    def from_row(
        cls, row: sqlite3.Row
    ) -> "EnterprisePersonLoginSessionRecord":
        return cls(
            id=str(row["id"]),
            person_id=str(row["person_id"]),
            active_account_id=row["active_account_id"],
            status=str(row["status"]),
            person_epoch=int(row["person_epoch"]),
            session_epoch=int(row["session_epoch"]),
            absolute_expires_at=str(row["absolute_expires_at"]),
            created_at=str(row["created_at"]),
            last_seen_at=row["last_seen_at"],
            revoked_at=row["revoked_at"],
            account_token_version=int(row["account_token_version"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EnterprisePersonKBShareRecord:
    """A personal KB shared to another department account of the SAME person.

    Zero-copy sharing: the KB entity (id/workspace/build artifacts) stays
    single-owner; the share materializes as a direct ``enterprise_kb_acl`` row
    for the target account (written in the same transaction) plus a
    department-admin oversight signal keyed by ``target_tenant_id``. Sharing
    into a department implies that department's tenant admins gain the
    configured oversight floor role on the KB.
    """

    id: str
    kb_id: str
    person_id: str
    owner_account_id: str
    target_account_id: str
    target_tenant_id: str | None
    role: str
    status: str
    created_by: str | None
    revoked_by: str | None
    reason: str | None
    created_at: str
    updated_at: str
    revoked_at: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "EnterprisePersonKBShareRecord":
        return cls(
            id=str(row["id"]),
            kb_id=str(row["kb_id"]),
            person_id=str(row["person_id"]),
            owner_account_id=str(row["owner_account_id"]),
            target_account_id=str(row["target_account_id"]),
            target_tenant_id=row["target_tenant_id"],
            role=str(row["role"]),
            status=str(row["status"]),
            created_by=row["created_by"],
            revoked_by=row["revoked_by"],
            reason=row["reason"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            revoked_at=row["revoked_at"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dumps_json(value: dict[str, Any] | None) -> str:
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True)


def _loads_json_object(value: str | bytes | None) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise MetadataStoreError("Metadata JSON must be an object")
    return loaded


def _new_audit_id() -> str:
    return f"audit_{secrets.token_hex(12)}"


def _insert_audit_event(conn: sqlite3.Connection, event: AuditEventRecord) -> None:
    """Insert an audit row inside an in-flight _write() transaction.

    Person identity events are platform-level: callers pass
    ``actor_tenant_id=None``. Use this instead of the public
    ``AuditService.append`` to avoid opening a nested write transaction.
    """

    conn.execute(
        """
        INSERT INTO enterprise_audit_events (
            id, event_type, actor_user_id, actor_tenant_id, target_type,
            target_id, metadata_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event.id,
            event.event_type,
            event.actor_user_id,
            event.actor_tenant_id,
            event.target_type,
            event.target_id,
            _dumps_json(event.metadata),
            event.created_at,
        ),
    )


def _loads_optional_json_object(value: str | bytes | None) -> dict[str, Any] | None:
    if value in (None, ""):
        return None
    return _loads_json_object(value)


def _metadata_source_key(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("source_key")
    if not isinstance(value, str):
        return None
    source_key = value.strip()
    return source_key or None


def _validate_tenant_user_kb_override(
    record: EnterpriseTenantUserKBOverrideRecord,
) -> None:
    for field_name in ("tenant_id", "kb_id", "user_id"):
        value = getattr(record, field_name)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise InvalidTenantUserKBOverrideError(
                f"Override {field_name} must be a non-empty normalized string"
            )
    for field_name in ("created_at", "updated_at"):
        value = getattr(record, field_name)
        if not isinstance(value, str) or not value.strip():
            raise InvalidTenantUserKBOverrideError(
                f"Override {field_name} must be a non-empty string"
            )
    if record.granted_by is not None and (
        not isinstance(record.granted_by, str)
        or not record.granted_by.strip()
        or record.granted_by != record.granted_by.strip()
    ):
        raise InvalidTenantUserKBOverrideError(
            "Override granted_by must be null or a normalized non-empty string"
        )
    if record.effect == "allow":
        if record.role not in _TENANT_USER_KB_OVERRIDE_ROLES:
            raise InvalidTenantUserKBOverrideError(
                "Allow override role must be a canonical KB role"
            )
    elif record.effect == "deny":
        if record.role is not None:
            raise InvalidTenantUserKBOverrideError(
                "Deny override must not include a role"
            )
    else:
        raise InvalidTenantUserKBOverrideError(
            "Override effect must be either 'allow' or 'deny'"
        )


def _tenant_user_kb_override_target_snapshot(
    tenant_id: str,
    user_id: str,
    user: EnterpriseUserRecord | None,
    membership: EnterpriseTenantMembershipRecord | None,
) -> dict[str, Any]:
    """Return the security-relevant target revision for an override write."""

    user_snapshot = (
        None
        if user is None
        else {
            "id": user.id,
            "tenant_id": user.tenant_id,
            "system_role": user.system_role,
            "token_version": user.token_version,
            "updated_at": user.updated_at,
        }
    )
    membership_snapshot = (
        None
        if membership is None
        else {
            "tenant_id": membership.tenant_id,
            "user_id": membership.user_id,
            "role": membership.role,
            "updated_at": membership.updated_at,
        }
    )
    eligible = bool(
        user is not None
        and user.id == user_id
        and user.tenant_id == tenant_id
        and user.system_role != "super_admin"
        and membership is not None
        and membership.tenant_id == tenant_id
        and membership.user_id == user_id
        and membership.role == "tenant_member"
    )
    return {
        "eligible": eligible,
        "user": user_snapshot,
        "membership": membership_snapshot,
    }


def _assert_tenant_user_kb_override_target_preconditions(
    tenant_id: str,
    user_id: str,
    current_user: EnterpriseUserRecord | None,
    current_membership: EnterpriseTenantMembershipRecord | None,
    *,
    expected_user: Any = _EXPECTATION_UNSET,
    expected_membership: Any = _EXPECTATION_UNSET,
) -> None:
    """CAS-check an override target's user and membership in the write txn."""

    if (
        expected_user is _EXPECTATION_UNSET
        and expected_membership is _EXPECTATION_UNSET
    ):
        return
    if not isinstance(expected_user, EnterpriseUserRecord) or not isinstance(
        expected_membership, EnterpriseTenantMembershipRecord
    ):
        raise MetadataStoreError(
            "Override target CAS requires both user and membership snapshots"
        )

    expected = _tenant_user_kb_override_target_snapshot(
        tenant_id,
        user_id,
        expected_user,
        expected_membership,
    )
    current = _tenant_user_kb_override_target_snapshot(
        tenant_id,
        user_id,
        current_user,
        current_membership,
    )
    if not expected["eligible"] or not current["eligible"] or current != expected:
        raise MetadataConflictError(
            "tenant_user_kb_override_target",
            f"{tenant_id}:{user_id}",
            expected=expected,
            current=current,
        )


def _validate_kb_lifecycle_identity(kb_id: str, generation: str) -> None:
    for name, value in (("kb_id", kb_id), ("generation", generation)):
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise MetadataStoreError(
                f"KB lifecycle {name} must be a normalized non-empty string"
            )


def _validate_delete_job_id(delete_job_id: str) -> None:
    if (
        not isinstance(delete_job_id, str)
        or not delete_job_id.strip()
        or delete_job_id != delete_job_id.strip()
    ):
        raise MetadataStoreError(
            "KB lifecycle delete_job_id must be a normalized non-empty string"
        )


def _kb_lifecycle_conflict(
    kb_id: str,
    expected_generation: str | None,
    current: KBLifecycleRecord,
    *,
    expected_state: str = "active",
    expected_delete_job_id: str | None = None,
) -> KBLifecycleConflictError:
    expected: dict[str, Any] = {
        "generation": expected_generation,
        "state": expected_state,
    }
    if expected_delete_job_id is not None:
        expected["delete_job_id"] = expected_delete_job_id
    return KBLifecycleConflictError(
        "kb_lifecycle",
        kb_id,
        expected=expected,
        current={
            "generation": current.generation,
            "state": current.state,
            "delete_job_id": current.delete_job_id,
        },
    )


def _missing_kb_lifecycle_conflict(
    kb_id: str,
    expected_generation: str,
    *,
    expected_state: str,
    expected_delete_job_id: str | None = None,
) -> KBLifecycleConflictError:
    expected: dict[str, Any] = {
        "generation": expected_generation,
        "state": expected_state,
    }
    if expected_delete_job_id is not None:
        expected["delete_job_id"] = expected_delete_job_id
    return KBLifecycleConflictError(
        "kb_lifecycle",
        kb_id,
        expected=expected,
        current={"exists": False},
    )


class SQLiteMetadataStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.lock_path = Path(f"{self.db_path}.lock")
        self._lock = asyncio.Lock()
        self._job_guard_state: ContextVar[_SQLiteJobGuardTaskState | None] = (
            ContextVar(f"sqlite_job_guard_state_{id(self)}", default=None)
        )
        self._kb_write_guard_state: ContextVar[
            _SQLiteKBWriteGuardTaskState | None
        ] = ContextVar(f"sqlite_kb_write_guard_state_{id(self)}", default=None)
        self._chat_memory_guard_state: ContextVar[
            _SQLiteChatMemoryGuardTaskState | None
        ] = ContextVar(f"sqlite_chat_memory_guard_state_{id(self)}", default=None)
        self._initialized = False

    async def initialize(self) -> None:
        async with self._lock:
            with _MetadataFileLock(self.lock_path):
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                with self._connect() as conn:
                    self._initialize_schema(conn)
                self._initialized = True

    async def close(self) -> None:
        return None

    async def create_documents_and_job(
        self,
        documents: Sequence[DocumentRecord],
        job: JobRecord,
    ) -> tuple[list[DocumentRecord], JobRecord, bool]:
        await self._ensure_initialized()

        def write(
            conn: sqlite3.Connection,
        ) -> tuple[list[DocumentRecord], JobRecord, bool]:
            existing = self._get_job_by_idempotency_key(
                conn, job.kb_id, job.idempotency_key, job_type=job.job_type
            )
            if existing is not None:
                self._validate_idempotent_job(existing, job)
                return self._documents_for_job(conn, existing), existing, False
            for document in documents:
                self._insert_document(conn, document)
            self._insert_job(conn, job)
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
        with self._connect() as conn:
            where = "kb_id = ? AND deleted_at IS NULL"
            params: list[Any] = [kb_id]
            if status is not None:
                where += " AND status = ?"
                params.append(status)
            if source_name is not None:
                where += " AND source_name LIKE ? ESCAPE '\\' COLLATE NOCASE"
                params.append(f"%{_escape_like(source_name)}%")
            total = conn.execute(
                f"SELECT COUNT(*) FROM documents WHERE {where}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"""
                SELECT * FROM documents
                WHERE {where}
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()
        return [DocumentRecord.from_row(row) for row in rows], int(total)

    async def get_document(self, kb_id: str, document_id: str) -> DocumentRecord:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM documents
                WHERE kb_id = ? AND id = ? AND deleted_at IS NULL
                """,
                (kb_id, document_id),
            ).fetchone()
        if row is None:
            raise MetadataRecordNotFoundError(f"Document '{document_id}' not found")
        return DocumentRecord.from_row(row)

    async def get_documents_by_ids(
        self, kb_id: str, document_ids: Sequence[str]
    ) -> list[DocumentRecord]:
        await self._ensure_initialized()
        if not document_ids:
            return []
        placeholders = ", ".join("?" for _ in document_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM documents
                WHERE kb_id = ? AND id IN ({placeholders}) AND deleted_at IS NULL
                """,
                [kb_id, *document_ids],
            ).fetchall()
        records_by_id = {row["id"]: DocumentRecord.from_row(row) for row in rows}
        return [
            records_by_id[document_id]
            for document_id in document_ids
            if document_id in records_by_id
        ]

    async def get_documents_by_source_keys(
        self, kb_id: str, source_keys: Sequence[str]
    ) -> dict[str, DocumentRecord]:
        await self._ensure_initialized()
        ordered_keys = list(dict.fromkeys(source_keys))
        if not ordered_keys:
            return {}
        placeholders = ", ".join("?" for _ in ordered_keys)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT d.*, sk.source_key AS mapped_source_key
                FROM document_source_keys sk
                JOIN documents d ON d.kb_id = sk.kb_id AND d.id = sk.document_id
                WHERE sk.kb_id = ? AND sk.source_key IN ({placeholders})
                    AND d.deleted_at IS NULL
                ORDER BY d.updated_at DESC, d.created_at DESC, d.id DESC
                """,
                [kb_id, *ordered_keys],
            ).fetchall()
        documents: dict[str, DocumentRecord] = {}
        wanted = set(ordered_keys)
        for row in rows:
            source_key = str(row["mapped_source_key"])
            if source_key in wanted and source_key not in documents:
                documents[source_key] = DocumentRecord.from_row(row)
        return documents

    async def list_documents_by_batch_id(
        self, kb_id: str, batch_id: str
    ) -> list[DocumentRecord]:
        await self._ensure_initialized()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM documents
                WHERE kb_id = ? AND deleted_at IS NULL
                ORDER BY created_at ASC, id ASC
                """,
                (kb_id,),
            ).fetchall()
        documents = [DocumentRecord.from_row(row) for row in rows]
        return [
            document
            for document in documents
            if document.metadata.get("batch_id") == batch_id
        ]

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

        def write(conn: sqlite3.Connection) -> DocumentRecord:
            current_row = conn.execute(
                """
                SELECT * FROM documents
                WHERE kb_id = ? AND id = ? AND deleted_at IS NULL
                """,
                (kb_id, document_id),
            ).fetchone()
            if current_row is None:
                raise MetadataRecordNotFoundError(f"Document '{document_id}' not found")
            metadata = _loads_json_object(current_row["metadata_json"])
            if metadata_patch:
                metadata.update(metadata_patch)
            now = utc_now_iso()
            self._sync_document_source_key(
                conn,
                kb_id=kb_id,
                document_id=document_id,
                source_key=_metadata_source_key(metadata),
                timestamp=now,
            )
            conn.execute(
                """
                UPDATE documents
                SET enabled = ?, archived = ?, metadata_json = ?, updated_at = ?
                WHERE kb_id = ? AND id = ?
                """,
                (
                    int(enabled) if enabled is not None else current_row["enabled"],
                    int(archived) if archived is not None else current_row["archived"],
                    _dumps_json(metadata),
                    now,
                    kb_id,
                    document_id,
                ),
            )
            row = conn.execute(
                "SELECT * FROM documents WHERE kb_id = ? AND id = ?",
                (kb_id, document_id),
            ).fetchone()
            if row is None:
                raise MetadataRecordNotFoundError(f"Document '{document_id}' not found")
            return DocumentRecord.from_row(row)

        return await self._write(write)

    async def mark_document_parse_queued(
        self,
        kb_id: str,
        document_id: str,
        *,
        metadata_patch: dict[str, Any],
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
        self,
        kb_id: str,
        claims: Sequence[tuple[str, dict[str, Any]]],
    ) -> tuple[list[DocumentRecord], list[dict[str, Any]]]:
        await self._ensure_initialized()

        def write(
            conn: sqlite3.Connection,
        ) -> tuple[list[DocumentRecord], list[dict[str, Any]]]:
            documents: list[DocumentRecord] = []
            failures: list[dict[str, Any]] = []
            for document_id, metadata_patch in claims:
                try:
                    documents.append(
                        self._claim_document_parse_queued(
                            conn,
                            kb_id,
                            document_id,
                            metadata_patch=metadata_patch,
                            raise_on_active=True,
                        )
                    )
                except ActiveDocumentParseJobError as exc:
                    failures.append(
                        {
                            "document_id": document_id,
                            "status": "failed",
                            "error_code": "parse_job_active",
                            "error_message": str(exc),
                            "existing_job_id": exc.existing_job_id,
                        }
                    )
                except ActiveDocumentBuildJobError as exc:
                    failures.append(
                        {
                            "document_id": document_id,
                            "status": "failed",
                            "error_code": "build_job_active",
                            "error_message": str(exc),
                            "existing_job_id": exc.existing_job_id,
                        }
                    )
                except ActiveDocumentDeleteJobError as exc:
                    failures.append(
                        {
                            "document_id": document_id,
                            "status": "failed",
                            "error_code": "delete_job_active",
                            "error_message": str(exc),
                            "existing_job_id": exc.existing_job_id,
                        }
                    )
                except ActiveDocumentReplaceJobError as exc:
                    failures.append(
                        {
                            "document_id": document_id,
                            "status": "failed",
                            "error_code": "replace_job_active",
                            "error_message": str(exc),
                            "existing_job_id": exc.existing_job_id,
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

    async def mark_document_parsing(
        self,
        kb_id: str,
        document_id: str,
        *,
        metadata_patch: dict[str, Any],
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

        def write(
            conn: sqlite3.Connection,
        ) -> tuple[DocumentRecord, list[ArtifactRecord]]:
            document = self._update_document_parse_state(
                conn,
                kb_id,
                document_id,
                status="parsed",
                metadata_patch=metadata_patch,
                parser_hash=parser_hash,
                lightrag_doc_id=lightrag_doc_id,
                clear_error=True,
            )
            conn.execute(
                "DELETE FROM document_artifacts WHERE kb_id = ? AND document_id = ?",
                (kb_id, document_id),
            )
            for artifact in artifacts:
                self._insert_artifact(conn, artifact)
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

        def write(
            conn: sqlite3.Connection,
        ) -> tuple[list[DocumentRecord], list[dict[str, Any]]]:
            documents: list[DocumentRecord] = []
            failures: list[dict[str, Any]] = []
            for document_id, metadata_patch in claims:
                try:
                    documents.append(
                        self._claim_document_build_queued(
                            conn,
                            kb_id,
                            document_id,
                            metadata_patch=metadata_patch,
                            require_parsed=require_parsed,
                        )
                    )
                except ActiveDocumentBuildJobError as exc:
                    failures.append(
                        {
                            "document_id": document_id,
                            "status": "failed",
                            "error_code": "build_job_active",
                            "error_message": str(exc),
                            "existing_job_id": exc.existing_job_id,
                        }
                    )
                except ActiveDocumentDeleteJobError as exc:
                    failures.append(
                        {
                            "document_id": document_id,
                            "status": "failed",
                            "error_code": "delete_job_active",
                            "error_message": str(exc),
                            "existing_job_id": exc.existing_job_id,
                        }
                    )
                except ActiveDocumentReplaceJobError as exc:
                    failures.append(
                        {
                            "document_id": document_id,
                            "status": "failed",
                            "error_code": "replace_job_active",
                            "error_message": str(exc),
                            "existing_job_id": exc.existing_job_id,
                        }
                    )
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
        self,
        kb_id: str,
        document_id: str,
        *,
        metadata_patch: dict[str, Any],
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

        def write(conn: sqlite3.Connection) -> DocumentRecord:
            current_row = conn.execute(
                """
                SELECT * FROM documents
                WHERE kb_id = ? AND id = ? AND deleted_at IS NULL
                """,
                (kb_id, document_id),
            ).fetchone()
            if current_row is None:
                raise MetadataRecordNotFoundError(f"Document '{document_id}' not found")
            metadata = _loads_json_object(current_row["metadata_json"])
            metadata.update(metadata_patch)
            now = utc_now_iso()
            conn.execute(
                """
                UPDATE documents
                SET status = ?, index_hash = ?, chunks_count = ?, entity_count = ?,
                    relation_count = ?, error_code = NULL, error_message = NULL,
                    metadata_json = ?, updated_at = ?
                WHERE kb_id = ? AND id = ?
                """,
                (
                    "ready",
                    index_hash,
                    chunks_count
                    if chunks_count is not None
                    else current_row["chunks_count"],
                    entity_count
                    if entity_count is not None
                    else current_row["entity_count"],
                    relation_count
                    if relation_count is not None
                    else current_row["relation_count"],
                    _dumps_json(metadata),
                    now,
                    kb_id,
                    document_id,
                ),
            )
            row = conn.execute(
                "SELECT * FROM documents WHERE kb_id = ? AND id = ?",
                (kb_id, document_id),
            ).fetchone()
            if row is None:
                raise MetadataRecordNotFoundError(f"Document '{document_id}' not found")
            return DocumentRecord.from_row(row)

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
        self,
        kb_id: str,
        document_id: str,
        *,
        metadata_patch: dict[str, Any],
    ) -> DocumentRecord:
        await self._ensure_initialized()
        return await self._write(
            lambda conn: self._claim_document_deleting(
                conn,
                kb_id,
                document_id,
                metadata_patch=metadata_patch,
            )
        )

    async def claim_documents_deleting(
        self,
        kb_id: str,
        claims: Sequence[tuple[str, dict[str, Any]]],
    ) -> tuple[list[DocumentRecord], list[dict[str, Any]]]:
        await self._ensure_initialized()

        def write(
            conn: sqlite3.Connection,
        ) -> tuple[list[DocumentRecord], list[dict[str, Any]]]:
            documents: list[DocumentRecord] = []
            failures: list[dict[str, Any]] = []
            for document_id, metadata_patch in claims:
                try:
                    documents.append(
                        self._claim_document_deleting(
                            conn,
                            kb_id,
                            document_id,
                            metadata_patch=metadata_patch,
                        )
                    )
                except ActiveDocumentParseJobError as exc:
                    failures.append(
                        {
                            "document_id": document_id,
                            "status": "failed",
                            "error_code": "parse_job_active",
                            "error_message": str(exc),
                            "existing_job_id": exc.existing_job_id,
                        }
                    )
                except ActiveDocumentBuildJobError as exc:
                    failures.append(
                        {
                            "document_id": document_id,
                            "status": "failed",
                            "error_code": "build_job_active",
                            "error_message": str(exc),
                            "existing_job_id": exc.existing_job_id,
                        }
                    )
                except ActiveDocumentDeleteJobError as exc:
                    failures.append(
                        {
                            "document_id": document_id,
                            "status": "failed",
                            "error_code": "delete_job_active",
                            "error_message": str(exc),
                            "existing_job_id": exc.existing_job_id,
                        }
                    )
                except ActiveDocumentReplaceJobError as exc:
                    failures.append(
                        {
                            "document_id": document_id,
                            "status": "failed",
                            "error_code": "replace_job_active",
                            "error_message": str(exc),
                            "existing_job_id": exc.existing_job_id,
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

    async def complete_document_delete(
        self,
        kb_id: str,
        document_id: str,
        *,
        metadata_patch: dict[str, Any],
    ) -> DocumentRecord:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> DocumentRecord:
            current_row = conn.execute(
                """
                SELECT * FROM documents
                WHERE kb_id = ? AND id = ? AND deleted_at IS NULL
                """,
                (kb_id, document_id),
            ).fetchone()
            if current_row is None:
                raise MetadataRecordNotFoundError(f"Document '{document_id}' not found")
            metadata = _loads_json_object(current_row["metadata_json"])
            metadata.update(metadata_patch)
            now = utc_now_iso()
            conn.execute(
                "DELETE FROM document_source_keys WHERE kb_id = ? AND document_id = ?",
                (kb_id, document_id),
            )
            conn.execute(
                "DELETE FROM document_artifacts WHERE kb_id = ? AND document_id = ?",
                (kb_id, document_id),
            )
            conn.execute(
                """
                UPDATE documents
                SET status = 'deleted', enabled = 0, archived = 1,
                    error_code = NULL, error_message = NULL, metadata_json = ?,
                    updated_at = ?, deleted_at = ?
                WHERE kb_id = ? AND id = ?
                """,
                (_dumps_json(metadata), now, now, kb_id, document_id),
            )
            row = conn.execute(
                "SELECT * FROM documents WHERE kb_id = ? AND id = ?",
                (kb_id, document_id),
            ).fetchone()
            if row is None:
                raise MetadataRecordNotFoundError(f"Document '{document_id}' not found")
            return DocumentRecord.from_row(row)

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
        self,
        kb_id: str,
        document_id: str,
        *,
        metadata_patch: dict[str, Any],
    ) -> DocumentRecord:
        await self._ensure_initialized()
        return await self._write(
            lambda conn: self._claim_document_replacing(
                conn,
                kb_id,
                document_id,
                metadata_patch=metadata_patch,
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

        def write(conn: sqlite3.Connection) -> DocumentRecord:
            current_row = conn.execute(
                """
                SELECT * FROM documents
                WHERE kb_id = ? AND id = ? AND deleted_at IS NULL
                """,
                (kb_id, document_id),
            ).fetchone()
            if current_row is None:
                raise MetadataRecordNotFoundError(f"Document '{document_id}' not found")
            metadata = _loads_json_object(current_row["metadata_json"])
            for key in _REPLACE_DERIVED_METADATA_KEYS:
                metadata.pop(key, None)
            metadata.update(metadata_patch)
            now = utc_now_iso()
            self._sync_document_source_key(
                conn,
                kb_id=kb_id,
                document_id=document_id,
                source_key=_metadata_source_key(metadata),
                timestamp=now,
            )
            conn.execute(
                "DELETE FROM document_artifacts WHERE kb_id = ? AND document_id = ?",
                (kb_id, document_id),
            )
            conn.execute(
                """
                UPDATE documents
                SET source_type = ?, source_name = ?, source_uri = ?,
                    source_hash = ?, content_type = ?, size_bytes = ?,
                    lightrag_doc_id = NULL, parser_hash = NULL, index_hash = NULL,
                    status = 'uploaded', chunks_count = NULL, entity_count = NULL,
                    relation_count = NULL, error_code = NULL, error_message = NULL,
                    metadata_json = ?, updated_at = ?
                WHERE kb_id = ? AND id = ?
                """,
                (
                    source_type,
                    source_name,
                    source_uri,
                    source_hash,
                    content_type,
                    size_bytes,
                    _dumps_json(metadata),
                    now,
                    kb_id,
                    document_id,
                ),
            )
            row = conn.execute(
                "SELECT * FROM documents WHERE kb_id = ? AND id = ?",
                (kb_id, document_id),
            ).fetchone()
            if row is None:
                raise MetadataRecordNotFoundError(f"Document '{document_id}' not found")
            return DocumentRecord.from_row(row)

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
        where = "kb_id = ? AND document_id = ?"
        params: list[Any] = [kb_id, document_id]
        if artifact_type is not None:
            where += " AND artifact_type = ?"
            params.append(artifact_type)
        with self._connect() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM document_artifacts WHERE {where}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"""
                SELECT * FROM document_artifacts
                WHERE {where}
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()
        return [ArtifactRecord.from_row(row) for row in rows], int(total)

    async def get_document_artifact(
        self, kb_id: str, document_id: str, artifact_id: str
    ) -> ArtifactRecord:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM document_artifacts
                WHERE kb_id = ? AND document_id = ? AND id = ?
                """,
                (kb_id, document_id, artifact_id),
            ).fetchone()
        if row is None:
            raise MetadataRecordNotFoundError(f"Artifact '{artifact_id}' not found")
        return ArtifactRecord.from_row(row)

    async def create_job(self, job: JobRecord) -> JobRecord:
        await self._ensure_initialized()

        created_job, _created = await self.create_job_once(job)
        return created_job

    async def create_job_once(self, job: JobRecord) -> tuple[JobRecord, bool]:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> tuple[JobRecord, bool]:
            existing = self._get_job_by_idempotency_key(
                conn, job.kb_id, job.idempotency_key, job_type=job.job_type
            )
            if existing is not None:
                self._validate_idempotent_job(existing, job)
                return existing, False
            return self._insert_job(conn, job), True

        return await self._write(write)

    async def get_job_by_idempotency_key(
        self, kb_id: str, idempotency_key: str, *, job_type: str | None = None
    ) -> JobRecord | None:
        await self._ensure_initialized()
        where = "kb_id = ? AND idempotency_key = ?"
        params: list[Any] = [kb_id, idempotency_key]
        if job_type is not None:
            where += " AND job_type = ?"
            params.append(job_type)
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT * FROM jobs
                WHERE {where}
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                params,
            ).fetchone()
        return JobRecord.from_row(row) if row is not None else None

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
        where = "kb_id = ?"
        params: list[Any] = [kb_id]
        if document_id is not None:
            where += " AND document_id = ?"
            params.append(document_id)
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            where += f" AND status IN ({placeholders})"
            params.extend(statuses)
        with self._connect() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM jobs WHERE {where}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"""
                SELECT * FROM jobs
                WHERE {where}
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()
        return [JobRecord.from_row(row) for row in rows], int(total)

    async def aggregate_control_plane_stats(
        self, kb_id: str | None = None
    ) -> dict[str, Any]:
        """Control-plane aggregates for the stats endpoints.

        Single GROUP-BY round trips on projected columns; ``kb_id=None``
        aggregates across every knowledge base. Soft-deleted documents keep
        their ``deleted`` status bucket; their counters were reset on delete
        so the counter sums only reflect live index state.
        """
        await self._ensure_initialized()
        where = "" if kb_id is None else " WHERE kb_id = ?"
        params: list[Any] = [] if kb_id is None else [kb_id]
        with self._connect() as conn:
            documents_by_status = {
                str(row[0]): int(row[1])
                for row in conn.execute(
                    f"SELECT status, COUNT(*) FROM documents{where} GROUP BY status",
                    params,
                ).fetchall()
            }
            counter_row = conn.execute(
                "SELECT COALESCE(SUM(chunks_count), 0), "
                "COALESCE(SUM(entity_count), 0), "
                f"COALESCE(SUM(relation_count), 0) FROM documents{where}",
                params,
            ).fetchone()
            jobs_by_status = {
                str(row[0]): int(row[1])
                for row in conn.execute(
                    f"SELECT status, COUNT(*) FROM jobs{where} GROUP BY status",
                    params,
                ).fetchall()
            }
            dead_letter_where = (
                " WHERE kb_id = ? AND " if kb_id is not None else " WHERE "
            )
            dead_letter = conn.execute(
                f"SELECT COUNT(*) FROM jobs{dead_letter_where}"
                "status = 'failed' AND retry_count >= max_retries",
                params,
            ).fetchone()[0]
            artifacts = conn.execute(
                f"SELECT COUNT(*) FROM document_artifacts{where}", params
            ).fetchone()[0]
        return {
            "documents_by_status": documents_by_status,
            "document_counters": {
                "chunks": int(counter_row[0]),
                "entities": int(counter_row[1]),
                "relations": int(counter_row[2]),
            },
            "jobs_by_status": jobs_by_status,
            "dead_letter_jobs": int(dead_letter),
            "artifacts": int(artifacts),
        }

    async def aggregate_enterprise_stats(self) -> dict[str, Any]:
        """Platform-wide enterprise aggregates for ``GET /admin/overview``."""
        await self._ensure_initialized()
        with self._connect() as conn:
            users_by_status = {
                str(row[0]): int(row[1])
                for row in conn.execute(
                    "SELECT status, COUNT(*) FROM enterprise_users GROUP BY status"
                ).fetchall()
            }
            tenants = conn.execute(
                "SELECT COUNT(*) FROM enterprise_tenants"
            ).fetchone()[0]
            api_keys_by_status = {
                str(row[0]): int(row[1])
                for row in conn.execute(
                    "SELECT status, COUNT(*) FROM enterprise_api_keys GROUP BY status"
                ).fetchall()
            }
            audit_events = conn.execute(
                "SELECT COUNT(*) FROM enterprise_audit_events"
            ).fetchone()[0]
        return {
            "users_by_status": users_by_status,
            "tenants": int(tenants),
            "api_keys_by_status": api_keys_by_status,
            "audit_events": int(audit_events),
        }

    async def count_active_jobs_for_principal(self, subject_id: str) -> int:
        """Count in-flight jobs (queued/running/retrying/cancelling) across all
        KBs attributed to a principal via ``payload._principal.subject_id``."""
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) FROM jobs
                WHERE status IN ('queued', 'running', 'retrying', 'cancelling')
                  AND json_extract(payload_json, '$._principal.subject_id') = ?
                """,
                (subject_id,),
            ).fetchone()
        return int(row[0])

    async def count_active_jobs_for_tenant(self, tenant_id: str) -> int:
        """Count in-flight jobs across all KBs attributed to a tenant via
        ``payload._principal.tenant_id``."""
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) FROM jobs
                WHERE status IN ('queued', 'running', 'retrying', 'cancelling')
                  AND json_extract(payload_json, '$._principal.tenant_id') = ?
                """,
                (tenant_id,),
            ).fetchone()
        return int(row[0])

    async def list_dead_letter_jobs(
        self,
        kb_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[JobRecord], int]:
        """List dead-lettered jobs: ``failed`` AND retries exhausted.

        A job is dead-lettered once it is ``failed`` and ``retry_count >=
        max_retries`` — :meth:`reset_job_for_retry` refuses to retry it, so it
        will never run again without operator intervention. Surfacing these
        separately lets operators triage terminal failures instead of scanning
        every failed job (some of which are still retryable). ``cancelled`` jobs
        are excluded: they were stopped deliberately, not exhausted.
        """
        await self._ensure_initialized()
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        where = "kb_id = ? AND status = 'failed' AND retry_count >= max_retries"
        params: list[Any] = [kb_id]
        with self._connect() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM jobs WHERE {where}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"""
                SELECT * FROM jobs
                WHERE {where}
                ORDER BY updated_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()
        return [JobRecord.from_row(row) for row in rows], int(total)

    async def get_job(self, kb_id: str, job_id: str) -> JobRecord:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE kb_id = ? AND id = ?",
                (kb_id, job_id),
            ).fetchone()
        if row is None:
            raise MetadataRecordNotFoundError(f"Job '{job_id}' not found")
        return JobRecord.from_row(row)

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

        def write(conn: sqlite3.Connection) -> JobRecord:
            current_row = conn.execute(
                "SELECT * FROM jobs WHERE kb_id = ? AND id = ?",
                (kb_id, job_id),
            ).fetchone()
            if current_row is None:
                raise MetadataRecordNotFoundError(f"Job '{job_id}' not found")
            current = JobRecord.from_row(current_row)
            if status not in _allowed_next_job_statuses(current.status):
                raise InvalidJobTransitionError(
                    f"Cannot transition job '{job_id}' from {current.status} to {status}"
                )

            now = utc_now_iso()
            started_at = current.started_at
            finished_at = current.finished_at
            cancelled_at = current.cancelled_at
            if status == "running" and started_at is None:
                started_at = now
            if status in {"succeeded", "failed"} and finished_at is None:
                finished_at = now
            if status == "cancelled" and cancelled_at is None:
                cancelled_at = now

            conn.execute(
                """
                UPDATE jobs
                SET status = ?, stage = ?, progress = ?, completed_items = ?,
                    failed_items = ?, result_json = ?, error_code = ?,
                    error_message = ?, updated_at = ?,
                    started_at = ?, finished_at = ?, cancelled_at = ?
                WHERE kb_id = ? AND id = ?
                """,
                (
                    status,
                    stage if stage is not None else current.stage,
                    progress if progress is not None else current.progress,
                    completed_items
                    if completed_items is not None
                    else current.completed_items,
                    failed_items if failed_items is not None else current.failed_items,
                    _dumps_json(result)
                    if result is not None
                    else current_row["result_json"],
                    error_code,
                    error_message,
                    now,
                    started_at,
                    finished_at,
                    cancelled_at,
                    kb_id,
                    job_id,
                ),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE kb_id = ? AND id = ?",
                (kb_id, job_id),
            ).fetchone()
            if row is None:
                raise MetadataRecordNotFoundError(f"Job '{job_id}' not found")
            return JobRecord.from_row(row)

        return await self._write(write)

    async def update_job_payload_patch(
        self,
        kb_id: str,
        job_id: str,
        *,
        payload_patch: dict[str, Any],
    ) -> JobRecord:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> JobRecord:
            current_row = conn.execute(
                "SELECT * FROM jobs WHERE kb_id = ? AND id = ?",
                (kb_id, job_id),
            ).fetchone()
            if current_row is None:
                raise MetadataRecordNotFoundError(f"Job '{job_id}' not found")
            current = JobRecord.from_row(current_row)
            payload = dict(current.payload)
            payload.update(payload_patch)
            now = utc_now_iso()
            conn.execute(
                """
                UPDATE jobs
                SET payload_json = ?, updated_at = ?
                WHERE kb_id = ? AND id = ?
                """,
                (_dumps_json(payload), now, kb_id, job_id),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE kb_id = ? AND id = ?",
                (kb_id, job_id),
            ).fetchone()
            if row is None:
                raise MetadataRecordNotFoundError(f"Job '{job_id}' not found")
            return JobRecord.from_row(row)

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

        def write(conn: sqlite3.Connection) -> JobRecord:
            current_row = conn.execute(
                "SELECT * FROM jobs WHERE kb_id = ? AND id = ?",
                (kb_id, job_id),
            ).fetchone()
            if current_row is None:
                raise MetadataRecordNotFoundError(f"Job '{job_id}' not found")
            current = JobRecord.from_row(current_row)
            if current.status not in {"running", "retrying", "cancelling"}:
                return current
            merged_result = current.result
            if result_patch is not None:
                merged_result = {**(current.result or {}), **result_patch}
            conn.execute(
                """
                UPDATE jobs
                SET progress = ?, completed_items = ?, stage = ?,
                    result_json = ?, updated_at = ?
                WHERE kb_id = ? AND id = ?
                """,
                (
                    progress if progress is not None else current.progress,
                    completed_items
                    if completed_items is not None
                    else current.completed_items,
                    stage if stage is not None else current.stage,
                    _dumps_json(merged_result)
                    if merged_result is not None
                    else current_row["result_json"],
                    utc_now_iso(),
                    kb_id,
                    job_id,
                ),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE kb_id = ? AND id = ?",
                (kb_id, job_id),
            ).fetchone()
            if row is None:
                raise MetadataRecordNotFoundError(f"Job '{job_id}' not found")
            return JobRecord.from_row(row)

        return await self._write(write)

    async def reset_job_for_retry(
        self,
        kb_id: str,
        job_id: str,
        *,
        new_idempotency_key: str | None,
    ) -> JobRecord:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> JobRecord:
            current_row = conn.execute(
                "SELECT * FROM jobs WHERE kb_id = ? AND id = ?",
                (kb_id, job_id),
            ).fetchone()
            if current_row is None:
                raise MetadataRecordNotFoundError(f"Job '{job_id}' not found")
            current = JobRecord.from_row(current_row)
            if current.status not in {"failed", "cancelled"}:
                raise InvalidJobTransitionError(
                    f"Cannot retry job '{job_id}' from {current.status}"
                )
            if current.retry_count >= current.max_retries:
                raise InvalidJobTransitionError(
                    f"Job '{job_id}' has reached max_retries={current.max_retries}"
                )
            now = utc_now_iso()
            # Preserve the existing idempotency key when the caller does not
            # supply a new one; only an explicit non-null value replaces it.
            # (Passing ``None`` through verbatim would wipe the original key,
            # breaking get-or-create lookups for the retried job.)
            preserved_idempotency_key = (
                new_idempotency_key
                if new_idempotency_key is not None
                else current.idempotency_key
            )
            conn.execute(
                """
                UPDATE jobs
                SET status = 'queued', stage = ?, progress = 0.0,
                    completed_items = 0, failed_items = 0,
                    result_json = NULL, error_code = NULL, error_message = NULL,
                    retry_count = retry_count + 1,
                    idempotency_key = ?,
                    updated_at = ?, queued_at = ?,
                    started_at = NULL, finished_at = NULL, cancelled_at = NULL
                WHERE kb_id = ? AND id = ?
                """,
                (
                    current.stage,
                    preserved_idempotency_key,
                    now,
                    now,
                    kb_id,
                    job_id,
                ),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE kb_id = ? AND id = ?",
                (kb_id, job_id),
            ).fetchone()
            if row is None:
                raise MetadataRecordNotFoundError(f"Job '{job_id}' not found")
            return JobRecord.from_row(row)

        return await self._write(write)

    async def create_config_version(
        self, record: ConfigVersionRecord
    ) -> ConfigVersionRecord:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> ConfigVersionRecord:
            row = conn.execute(
                "SELECT MAX(version) FROM kb_config_versions WHERE kb_id = ?",
                (record.kb_id,),
            ).fetchone()
            next_version = (row[0] or 0) + 1 if row[0] is not None else 1
            persisted = ConfigVersionRecord(
                id=record.id,
                kb_id=record.kb_id,
                workspace=record.workspace,
                version=next_version,
                config=record.config,
                parser_hash=record.parser_hash,
                index_hash=record.index_hash,
                query_hash=record.query_hash,
                created_at=record.created_at,
                activated_at=None,
                created_by=record.created_by,
            )
            conn.execute(
                """
                INSERT INTO kb_config_versions (
                    id, kb_id, workspace, version, config_json, parser_hash,
                    index_hash, query_hash, created_at, activated_at, created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    persisted.id,
                    persisted.kb_id,
                    persisted.workspace,
                    persisted.version,
                    _dumps_json(persisted.config),
                    persisted.parser_hash,
                    persisted.index_hash,
                    persisted.query_hash,
                    persisted.created_at,
                    persisted.activated_at,
                    persisted.created_by,
                ),
            )
            return persisted

        return await self._write(write)

    async def list_config_versions(
        self, kb_id: str, *, limit: int = 50, offset: int = 0
    ) -> tuple[list[ConfigVersionRecord], int]:
        await self._ensure_initialized()
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM kb_config_versions WHERE kb_id = ?", (kb_id,)
            ).fetchone()[0]
            rows = conn.execute(
                """
                SELECT * FROM kb_config_versions
                WHERE kb_id = ?
                ORDER BY version DESC
                LIMIT ? OFFSET ?
                """,
                (kb_id, limit, offset),
            ).fetchall()
        return [ConfigVersionRecord.from_row(row) for row in rows], int(total)

    async def get_config_version(
        self, kb_id: str, version_id: str
    ) -> ConfigVersionRecord:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM kb_config_versions
                WHERE kb_id = ? AND id = ?
                """,
                (kb_id, version_id),
            ).fetchone()
        if row is None:
            raise MetadataRecordNotFoundError(
                f"Config version '{version_id}' not found"
            )
        return ConfigVersionRecord.from_row(row)

    async def mark_config_version_activated(
        self, kb_id: str, version_id: str
    ) -> ConfigVersionRecord:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> ConfigVersionRecord:
            row = conn.execute(
                """
                SELECT * FROM kb_config_versions
                WHERE kb_id = ? AND id = ?
                """,
                (kb_id, version_id),
            ).fetchone()
            if row is None:
                raise MetadataRecordNotFoundError(
                    f"Config version '{version_id}' not found"
                )
            conn.execute(
                """
                UPDATE kb_config_versions
                SET activated_at = ?
                WHERE kb_id = ? AND id = ?
                """,
                (utc_now_iso(), kb_id, version_id),
            )
            refreshed = conn.execute(
                """
                SELECT * FROM kb_config_versions
                WHERE kb_id = ? AND id = ?
                """,
                (kb_id, version_id),
            ).fetchone()
            assert refreshed is not None
            return ConfigVersionRecord.from_row(refreshed)

        return await self._write(write)

    async def get_enterprise_user_by_username(
        self, username: str
    ) -> EnterpriseUserRecord | None:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM enterprise_users WHERE username = ?",
                (username,),
            ).fetchone()
        return EnterpriseUserRecord.from_row(row) if row is not None else None

    async def get_enterprise_user_by_id(
        self, user_id: str
    ) -> EnterpriseUserRecord | None:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM enterprise_users WHERE id = ?",
                (user_id,),
            ).fetchone()
        return EnterpriseUserRecord.from_row(row) if row is not None else None

    async def list_enterprise_users(self) -> list[EnterpriseUserRecord]:
        await self._ensure_initialized()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM enterprise_users ORDER BY created_at ASC, id ASC"
            ).fetchall()
        return [EnterpriseUserRecord.from_row(row) for row in rows]

    async def upsert_enterprise_user(
        self,
        user: EnterpriseUserRecord,
        *,
        expected_updated_at: Any = _EXPECTATION_UNSET,
        expected_token_version: Any = _EXPECTATION_UNSET,
        expected_tenant_id: Any = _EXPECTATION_UNSET,
    ) -> EnterpriseUserRecord:
        """Create a tenant-less user or CAS-update an existing user."""
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
        """Atomically save a user and synchronize its canonical membership.

        ``user.tenant_id`` is authoritative. A null tenant produces no
        membership. For a non-null tenant, an explicitly supplied matching
        membership is used; otherwise the existing canonical membership is
        preserved or a ``tenant_member`` row is created. Overrides belonging to
        every tenant the user leaves are removed in the same transaction.
        """
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

        def write(conn: sqlite3.Connection) -> bool:
            current_row = conn.execute(
                "SELECT * FROM enterprise_users WHERE id = ?", (user_id,)
            ).fetchone()
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
            current_user = EnterpriseUserRecord.from_row(current_row)
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
                membership_rows = conn.execute(
                    """
                    SELECT * FROM enterprise_tenant_memberships
                    WHERE user_id = ? ORDER BY tenant_id ASC
                    """,
                    (user_id,),
                ).fetchall()
                _assert_enterprise_user_membership_precondition(
                    user_id,
                    [
                        EnterpriseTenantMembershipRecord.from_row(row)
                        for row in membership_rows
                    ],
                    expected_membership=expected_membership,
                )
            # Person identity lifecycle: revoke person login sessions pointing
            # at this account and remove person-account links BEFORE deleting
            # the account row. Links would otherwise CASCADE-delete (SQLite
            # honors ON DELETE CASCADE with PRAGMA foreign_keys=ON) but the
            # explicit delete is clearer and safer; sessions are revoked so no
            # active session can survive pointing at a deleted account. Both
            # the revoke and the audit row stay inside this transaction.
            now = utc_now_iso()
            self._sqlite_revoke_person_sessions_locked(
                conn,
                None,
                account_id=user_id,
                actor_user_id=None,
                now=now,
                audit_event_type="person_session_revoked_by_account_change",
            )
            self._revoke_person_kb_shares_locked(
                conn,
                either_side_account_id=user_id,
                actor_user_id=None,
                now=now,
                audit_event_type="person_kb_share_revoked_by_account_change",
                reason="account_deleted",
            )
            conn.execute(
                "DELETE FROM enterprise_person_account_links WHERE account_id = ?",
                (user_id,),
            )
            # Cascade: remove related records first.
            conn.execute(
                "DELETE FROM enterprise_tenant_memberships WHERE user_id = ?",
                (user_id,),
            )
            conn.execute(
                "DELETE FROM enterprise_tenant_user_kb_overrides WHERE user_id = ?",
                (user_id,),
            )
            conn.execute(
                "DELETE FROM enterprise_kb_acl WHERE user_id = ?",
                (user_id,),
            )
            conn.execute(
                "DELETE FROM enterprise_user_kb_query_settings WHERE user_id = ?",
                (user_id,),
            )
            conn.execute(
                "DELETE FROM enterprise_chat_messages WHERE user_id = ?",
                (user_id,),
            )
            conn.execute(
                "DELETE FROM enterprise_chat_sessions WHERE user_id = ?",
                (user_id,),
            )
            conn.execute(
                "DELETE FROM enterprise_chat_projects WHERE user_id = ?",
                (user_id,),
            )
            cursor = conn.execute(
                "DELETE FROM enterprise_users WHERE id = ?",
                (user_id,),
            )
            return bool(cursor.rowcount)

        return await self._write(write)

    async def get_enterprise_user_kb_query_settings(
        self, user_id: str, kb_id: str
    ) -> EnterpriseUserKBQuerySettingsRecord | None:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM enterprise_user_kb_query_settings
                WHERE user_id = ? AND kb_id = ?
                """,
                (user_id, kb_id),
            ).fetchone()
        return (
            EnterpriseUserKBQuerySettingsRecord.from_row(row)
            if row is not None
            else None
        )

    async def upsert_enterprise_user_kb_query_settings(
        self, record: EnterpriseUserKBQuerySettingsRecord
    ) -> EnterpriseUserKBQuerySettingsRecord:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> EnterpriseUserKBQuerySettingsRecord:
            conn.execute(
                """
                INSERT INTO enterprise_user_kb_query_settings (
                    user_id, kb_id, user_prompt, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, kb_id) DO UPDATE SET
                    user_prompt = excluded.user_prompt,
                    updated_at = excluded.updated_at
                """,
                (
                    record.user_id,
                    record.kb_id,
                    record.user_prompt,
                    record.created_at,
                    record.updated_at,
                ),
            )
            row = conn.execute(
                """
                SELECT * FROM enterprise_user_kb_query_settings
                WHERE user_id = ? AND kb_id = ?
                """,
                (record.user_id, record.kb_id),
            ).fetchone()
            assert row is not None
            return EnterpriseUserKBQuerySettingsRecord.from_row(row)

        return await self._write(write)

    async def delete_enterprise_user_kb_query_settings(
        self, user_id: str, kb_id: str
    ) -> bool:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute(
                """
                DELETE FROM enterprise_user_kb_query_settings
                WHERE user_id = ? AND kb_id = ?
                """,
                (user_id, kb_id),
            )
            return bool(cursor.rowcount)

        return await self._write(write)

    async def create_chat_project(self, record: ChatProjectRecord) -> ChatProjectRecord:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> ChatProjectRecord:
            conn.execute(
                """
                INSERT INTO enterprise_chat_projects (
                    id, user_id, name, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.user_id,
                    record.name,
                    record.created_at,
                    record.updated_at,
                ),
            )
            row = conn.execute(
                "SELECT * FROM enterprise_chat_projects WHERE id = ?", (record.id,)
            ).fetchone()
            assert row is not None
            return ChatProjectRecord.from_row(row)

        return await self._write(write)

    async def get_chat_project(
        self, user_id: str, project_id: str
    ) -> ChatProjectRecord | None:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM enterprise_chat_projects
                WHERE id = ? AND user_id = ?
                """,
                (project_id, user_id),
            ).fetchone()
        return ChatProjectRecord.from_row(row) if row is not None else None

    async def list_chat_projects(
        self, user_id: str, *, limit: int = 50, offset: int = 0
    ) -> tuple[list[ChatProjectRecord], int]:
        await self._ensure_initialized()
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM enterprise_chat_projects WHERE user_id = ?",
                (user_id,),
            ).fetchone()[0]
            rows = conn.execute(
                """
                SELECT * FROM enterprise_chat_projects
                WHERE user_id = ?
                ORDER BY updated_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (user_id, limit, offset),
            ).fetchall()
        return [ChatProjectRecord.from_row(row) for row in rows], int(total)

    async def rename_chat_project(
        self, user_id: str, project_id: str, *, name: str
    ) -> ChatProjectRecord | None:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> ChatProjectRecord | None:
            cursor = conn.execute(
                """
                UPDATE enterprise_chat_projects
                SET name = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (name, utc_now_iso(), project_id, user_id),
            )
            if not cursor.rowcount:
                return None
            row = conn.execute(
                "SELECT * FROM enterprise_chat_projects WHERE id = ?", (project_id,)
            ).fetchone()
            assert row is not None
            return ChatProjectRecord.from_row(row)

        return await self._write(write)

    async def delete_chat_project(
        self, user_id: str, project_id: str
    ) -> tuple[bool, int, int]:
        """Delete a chat project owned by ``user_id`` and cascade-delete its
        sessions and messages. Returns ``(deleted, deleted_sessions,
        deleted_messages)``."""
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> tuple[bool, int, int]:
            owner_row = conn.execute(
                """
                SELECT id FROM enterprise_chat_projects
                WHERE id = ? AND user_id = ?
                """,
                (project_id, user_id),
            ).fetchone()
            if owner_row is None:
                return False, 0, 0
            messages_cursor = conn.execute(
                "DELETE FROM enterprise_chat_messages WHERE project_id = ?",
                (project_id,),
            )
            sessions_cursor = conn.execute(
                "DELETE FROM enterprise_chat_sessions WHERE project_id = ?",
                (project_id,),
            )
            conn.execute(
                "DELETE FROM enterprise_chat_projects WHERE id = ?",
                (project_id,),
            )
            return True, int(sessions_cursor.rowcount), int(messages_cursor.rowcount)

        return await self._write(write)

    async def create_chat_session(self, record: ChatSessionRecord) -> ChatSessionRecord:
        """Insert a chat session. Raises :class:`MetadataRecordNotFoundError`
        when the parent project does not exist or is not owned by the user."""
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> ChatSessionRecord:
            project_row = conn.execute(
                """
                SELECT id FROM enterprise_chat_projects
                WHERE id = ? AND user_id = ?
                """,
                (record.project_id, record.user_id),
            ).fetchone()
            if project_row is None:
                raise MetadataRecordNotFoundError(
                    f"Chat project '{record.project_id}' not found"
                )
            conn.execute(
                """
                INSERT INTO enterprise_chat_sessions (
                    id, project_id, user_id, name, context_rounds,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.project_id,
                    record.user_id,
                    record.name,
                    record.context_rounds,
                    record.created_at,
                    record.updated_at,
                ),
            )
            row = conn.execute(
                "SELECT * FROM enterprise_chat_sessions WHERE id = ?", (record.id,)
            ).fetchone()
            assert row is not None
            return ChatSessionRecord.from_row(row)

        return await self._write(write)

    async def get_chat_session(
        self, user_id: str, project_id: str, session_id: str
    ) -> ChatSessionRecord | None:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM enterprise_chat_sessions
                WHERE id = ? AND project_id = ? AND user_id = ?
                """,
                (session_id, project_id, user_id),
            ).fetchone()
        return ChatSessionRecord.from_row(row) if row is not None else None

    async def list_chat_sessions(
        self, user_id: str, project_id: str, *, limit: int = 50, offset: int = 0
    ) -> tuple[list[ChatSessionRecord], int]:
        await self._ensure_initialized()
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        with self._connect() as conn:
            total = conn.execute(
                """
                SELECT COUNT(*) FROM enterprise_chat_sessions
                WHERE project_id = ? AND user_id = ?
                """,
                (project_id, user_id),
            ).fetchone()[0]
            rows = conn.execute(
                """
                SELECT * FROM enterprise_chat_sessions
                WHERE project_id = ? AND user_id = ?
                ORDER BY updated_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (project_id, user_id, limit, offset),
            ).fetchall()
        return [ChatSessionRecord.from_row(row) for row in rows], int(total)

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

        def write(conn: sqlite3.Connection) -> ChatSessionRecord | None:
            sets = ["updated_at = ?"]
            params: list[Any] = [utc_now_iso()]
            if name is not None:
                sets.append("name = ?")
                params.append(name)
            if context_rounds is not None:
                sets.append("context_rounds = ?")
                params.append(context_rounds)
            cursor = conn.execute(
                f"""
                UPDATE enterprise_chat_sessions
                SET {", ".join(sets)}
                WHERE id = ? AND project_id = ? AND user_id = ?
                """,
                (*params, session_id, project_id, user_id),
            )
            if not cursor.rowcount:
                return None
            row = conn.execute(
                "SELECT * FROM enterprise_chat_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            assert row is not None
            return ChatSessionRecord.from_row(row)

        return await self._write(write)

    async def delete_chat_session(
        self, user_id: str, project_id: str, session_id: str
    ) -> tuple[bool, int]:
        """Delete an owned session and cascade-delete its messages. Returns
        ``(deleted, deleted_messages)``."""
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> tuple[bool, int]:
            owner_row = conn.execute(
                """
                SELECT id FROM enterprise_chat_sessions
                WHERE id = ? AND project_id = ? AND user_id = ?
                """,
                (session_id, project_id, user_id),
            ).fetchone()
            if owner_row is None:
                return False, 0
            messages_cursor = conn.execute(
                "DELETE FROM enterprise_chat_messages WHERE session_id = ?",
                (session_id,),
            )
            conn.execute(
                "DELETE FROM enterprise_chat_sessions WHERE id = ?",
                (session_id,),
            )
            return True, int(messages_cursor.rowcount)

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

        def write(conn: sqlite3.Connection) -> list[ChatMessageRecord]:
            session_row = conn.execute(
                """
                SELECT id FROM enterprise_chat_sessions
                WHERE id = ? AND project_id = ? AND user_id = ?
                """,
                (head.session_id, head.project_id, head.user_id),
            ).fetchone()
            if session_row is None:
                raise MetadataRecordNotFoundError(
                    f"Chat session '{head.session_id}' not found"
                )
            next_seq = (
                int(
                    conn.execute(
                        """
                        SELECT COALESCE(MAX(seq), 0) FROM enterprise_chat_messages
                        WHERE session_id = ?
                        """,
                        (head.session_id,),
                    ).fetchone()[0]
                )
                + 1
            )
            for index, record in enumerate(records):
                record.seq = next_seq + index
                record.append_batch_id = None
                record.project_event_seq = None
                record.memory_reference_time = None
                conn.execute(
                    """
                    INSERT INTO enterprise_chat_messages (
                        id, session_id, project_id, user_id, role, content,
                        metadata_json, seq, created_at, append_batch_id,
                        project_event_seq, memory_reference_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.id,
                        record.session_id,
                        record.project_id,
                        record.user_id,
                        record.role,
                        record.content,
                        _dumps_json(record.metadata),
                        record.seq,
                        record.created_at,
                        None,
                        None,
                        None,
                    ),
                )
            conn.execute(
                "UPDATE enterprise_chat_sessions SET updated_at = ? WHERE id = ?",
                (utc_now_iso(), head.session_id),
            )
            rows = conn.execute(
                """
                SELECT * FROM enterprise_chat_messages
                WHERE session_id = ? AND seq >= ?
                ORDER BY seq ASC, id ASC
                """,
                (head.session_id, next_seq),
            ).fetchall()
            return [ChatMessageRecord.from_row(row) for row in rows]

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
        with self._connect() as conn:
            total = conn.execute(
                """
                SELECT COUNT(*) FROM enterprise_chat_messages
                WHERE session_id = ? AND project_id = ? AND user_id = ?
                """,
                (session_id, project_id, user_id),
            ).fetchone()[0]
            rows = conn.execute(
                """
                SELECT * FROM enterprise_chat_messages
                WHERE session_id = ? AND project_id = ? AND user_id = ?
                ORDER BY seq ASC, id ASC
                LIMIT ? OFFSET ?
                """,
                (session_id, project_id, user_id, limit, offset),
            ).fetchall()
        return [ChatMessageRecord.from_row(row) for row in rows], int(total)

    async def delete_chat_message(
        self, user_id: str, project_id: str, session_id: str, message_id: str
    ) -> bool:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute(
                """
                DELETE FROM enterprise_chat_messages
                WHERE id = ? AND session_id = ? AND project_id = ? AND user_id = ?
                """,
                (message_id, session_id, project_id, user_id),
            )
            return bool(cursor.rowcount)

        return await self._write(write)

    async def get_chat_message(
        self, user_id: str, project_id: str, session_id: str, message_id: str
    ) -> ChatMessageRecord | None:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM enterprise_chat_messages
                WHERE id = ? AND session_id = ? AND project_id = ? AND user_id = ?
                """,
                (message_id, session_id, project_id, user_id),
            ).fetchone()
        return ChatMessageRecord.from_row(row) if row is not None else None

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
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM enterprise_chat_messages
                WHERE session_id = ? AND project_id = ? AND user_id = ? AND seq > ?
                ORDER BY seq ASC, id ASC
                LIMIT ?
                """,
                (session_id, project_id, user_id, int(after_seq), limit),
            ).fetchall()
        return [ChatMessageRecord.from_row(row) for row in rows]

    async def record_chat_memory_episode(
        self, record: ChatMemoryEpisodeRecord
    ) -> None:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> None:
            conn.execute(
                """
                INSERT INTO enterprise_chat_memory_episodes (
                    episode_uuid, session_id, project_id, user_id,
                    first_seq, last_seq, created_at, event_id, generation,
                    graph_group_id, append_batch_id, project_event_seq
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(episode_uuid) DO UPDATE SET
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
                (
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
                ),
            )

        await self._write(write)

    async def get_chat_memory_watermark(
        self, user_id: str, project_id: str, session_id: str
    ) -> int:
        """Highest ingested ``last_seq`` for a session (0 when none)."""
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(MAX(last_seq), 0) FROM enterprise_chat_memory_episodes
                WHERE session_id = ? AND project_id = ? AND user_id = ?
                """,
                (session_id, project_id, user_id),
            ).fetchone()
        return int(row[0]) if row is not None else 0

    async def find_chat_memory_episodes_covering(
        self, user_id: str, project_id: str, session_id: str, seq: int
    ) -> list[ChatMemoryEpisodeRecord]:
        await self._ensure_initialized()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM enterprise_chat_memory_episodes
                WHERE session_id = ? AND project_id = ? AND user_id = ?
                    AND first_seq <= ? AND last_seq >= ?
                ORDER BY first_seq ASC
                """,
                (session_id, project_id, user_id, int(seq), int(seq)),
            ).fetchall()
        return [ChatMemoryEpisodeRecord.from_row(row) for row in rows]

    async def list_chat_memory_episodes_for_session(
        self, user_id: str, project_id: str, session_id: str
    ) -> list[ChatMemoryEpisodeRecord]:
        await self._ensure_initialized()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM enterprise_chat_memory_episodes
                WHERE session_id = ? AND project_id = ? AND user_id = ?
                ORDER BY first_seq ASC
                """,
                (session_id, project_id, user_id),
            ).fetchall()
        return [ChatMemoryEpisodeRecord.from_row(row) for row in rows]

    async def delete_chat_memory_episodes(
        self, episode_uuids: Sequence[str]
    ) -> int:
        ids = [uuid for uuid in episode_uuids if uuid]
        if not ids:
            return 0
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> int:
            placeholders = ",".join("?" for _ in ids)
            cursor = conn.execute(
                "DELETE FROM enterprise_chat_memory_episodes "
                f"WHERE episode_uuid IN ({placeholders})",
                tuple(ids),
            )
            return int(cursor.rowcount)

        return await self._write(write)

    async def delete_chat_memory_episodes_for_project(self, project_id: str) -> int:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(
                "DELETE FROM enterprise_chat_memory_episodes WHERE project_id = ?",
                (project_id,),
            )
            return int(cursor.rowcount)

        return await self._write(write)

    async def delete_chat_memory_episodes_for_user(self, user_id: str) -> int:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(
                "DELETE FROM enterprise_chat_memory_episodes WHERE user_id = ?",
                (user_id,),
            )
            return int(cursor.rowcount)

        return await self._write(write)

    async def list_chat_memory_backlog(
        self, *, limit: int = 100
    ) -> list[ChatMemoryBacklogItem]:
        """Sessions whose max message ``seq`` exceeds the ingestion watermark."""
        await self._ensure_initialized()
        limit = max(1, min(int(limit), 1000))
        with self._connect() as conn:
            rows = conn.execute(
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
                GROUP BY m.session_id, m.project_id, m.user_id
                HAVING MAX(m.seq) > COALESCE(e.max_last, 0)
                ORDER BY m.session_id
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
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
        """Return ``(episode_count, last_ingested_at)`` for a project.

        Excludes ``noop_`` placeholder rows (blank ranges that only advance the
        watermark) so the count reflects real distilled episodes. Used by the
        project memory-overview endpoint.
        """
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS c, MAX(created_at) AS last_at
                FROM enterprise_chat_memory_episodes
                WHERE user_id = ? AND project_id = ?
                    AND episode_uuid NOT LIKE 'noop\\_%' ESCAPE '\\'
                """,
                (user_id, project_id),
            ).fetchone()
        if row is None:
            return 0, None
        return int(row["c"] or 0), (row["last_at"] if row["last_at"] else None)

    async def count_chat_memory_episodes(self) -> tuple[int, int, int]:
        """Global counts for admin observability:
        ``(episode_count, distinct_users, distinct_projects)`` — noop rows
        excluded."""
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS c,
                       COUNT(DISTINCT user_id) AS u,
                       COUNT(DISTINCT project_id) AS p
                FROM enterprise_chat_memory_episodes
                WHERE episode_uuid NOT LIKE 'noop\\_%' ESCAPE '\\'
                """
            ).fetchone()
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

        def write(conn: sqlite3.Connection) -> list[ChatMessageRecord]:
            # Stable source lock order: user -> project -> session -> group.
            if (
                conn.execute(
                    "SELECT id FROM enterprise_users WHERE id = ?", (head.user_id,)
                ).fetchone()
                is None
            ):
                raise MetadataRecordNotFoundError(
                    f"Chat session '{head.session_id}' not found"
                )
            if (
                conn.execute(
                    """
                    SELECT id FROM enterprise_chat_projects
                    WHERE id = ? AND user_id = ?
                    """,
                    (head.project_id, head.user_id),
                ).fetchone()
                is None
            ):
                raise MetadataRecordNotFoundError(
                    f"Chat session '{head.session_id}' not found"
                )
            session_row = conn.execute(
                """
                SELECT * FROM enterprise_chat_sessions
                WHERE id = ? AND project_id = ? AND user_id = ?
                """,
                (head.session_id, head.project_id, head.user_id),
            ).fetchone()
            if session_row is None:
                raise MetadataRecordNotFoundError(
                    f"Chat session '{head.session_id}' not found"
                )

            group, _created = self._ensure_sqlite_chat_memory_group(
                conn,
                head.user_id,
                head.project_id,
                fingerprint,
                graph_fingerprint,
                generation_state="building",
            )
            self._assert_sqlite_chat_memory_graph_store_invariant(
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
                self._enqueue_sqlite_chat_memory_rebuild(
                    conn,
                    group,
                    fingerprint,
                    graph_fingerprint,
                    actor_user_id=actor_user_id or head.user_id,
                    actor_tenant_id=actor_tenant_id,
                    target_session_id=head.session_id,
                    target_message_id=None,
                )
                group = self._get_sqlite_chat_memory_group(
                    conn, head.user_id, head.project_id
                )
                assert group is not None

            event_seq, reference_time = self._allocate_sqlite_chat_memory_event_seq(
                conn,
                head.user_id,
                head.project_id,
                allocate_reference_time=True,
            )
            assert reference_time is not None
            append_batch_id = _chat_memory_append_batch_id(
                user_id=head.user_id,
                project_id=head.project_id,
                session_id=head.session_id,
                event_seq=event_seq,
                message_ids=[record.id for record in records],
            )
            next_seq = (
                int(
                    conn.execute(
                        """
                        SELECT COALESCE(MAX(seq), 0)
                        FROM enterprise_chat_messages WHERE session_id = ?
                        """,
                        (head.session_id,),
                    ).fetchone()[0]
                )
                + 1
            )
            for index, record in enumerate(records):
                record.seq = next_seq + index
                record.append_batch_id = append_batch_id
                record.project_event_seq = event_seq
                record.memory_reference_time = reference_time
                conn.execute(
                    """
                    INSERT INTO enterprise_chat_messages (
                        id, session_id, project_id, user_id, role, content,
                        metadata_json, seq, created_at, append_batch_id,
                        project_event_seq, memory_reference_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.id,
                        record.session_id,
                        record.project_id,
                        record.user_id,
                        record.role,
                        record.content,
                        _dumps_json(record.metadata),
                        record.seq,
                        record.created_at,
                        append_batch_id,
                        event_seq,
                        reference_time,
                    ),
                )

            now = utc_now_iso()
            conn.execute(
                "UPDATE enterprise_chat_sessions SET updated_at = ? WHERE id = ?",
                (now, head.session_id),
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
            self._insert_sqlite_chat_memory_event(
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
            rows = conn.execute(
                """
                SELECT * FROM enterprise_chat_messages
                WHERE append_batch_id = ?
                ORDER BY seq ASC, id ASC
                """,
                (append_batch_id,),
            ).fetchall()
            return [ChatMessageRecord.from_row(row) for row in rows]

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

        def write(conn: sqlite3.Connection) -> bool:
            # Source order mirrors PostgreSQL even though SQLite serializes writes.
            if conn.execute(
                "SELECT id FROM enterprise_users WHERE id = ?", (user_id,)
            ).fetchone() is None:
                return False
            if conn.execute(
                """
                SELECT id FROM enterprise_chat_projects
                WHERE id = ? AND user_id = ?
                """,
                (project_id, user_id),
            ).fetchone() is None:
                return False
            if conn.execute(
                """
                SELECT id FROM enterprise_chat_sessions
                WHERE id = ? AND project_id = ? AND user_id = ?
                """,
                (session_id, project_id, user_id),
            ).fetchone() is None:
                return False
            message_row = conn.execute(
                """
                SELECT * FROM enterprise_chat_messages
                WHERE id = ? AND session_id = ? AND project_id = ? AND user_id = ?
                """,
                (message_id, session_id, project_id, user_id),
            ).fetchone()
            if message_row is None:
                return False
            memory_affected = message_row["project_event_seq"] is not None
            if not memory_affected:
                memory_affected = (
                    conn.execute(
                        """
                        SELECT 1 FROM enterprise_chat_memory_episodes
                        WHERE session_id = ? AND project_id = ? AND user_id = ?
                          AND first_seq <= ? AND last_seq >= ?
                        LIMIT 1
                        """,
                        (
                            session_id,
                            project_id,
                            user_id,
                            int(message_row["seq"]),
                            int(message_row["seq"]),
                        ),
                    ).fetchone()
                    is not None
                )
            conn.execute(
                "DELETE FROM enterprise_chat_messages WHERE id = ?", (message_id,)
            )
            if memory_affected:
                group, _ = self._ensure_sqlite_chat_memory_group(
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
                self._enqueue_sqlite_chat_memory_rebuild(
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

        def write(conn: sqlite3.Connection) -> tuple[bool, int]:
            if conn.execute(
                "SELECT id FROM enterprise_users WHERE id = ?", (user_id,)
            ).fetchone() is None:
                return False, 0
            if conn.execute(
                """
                SELECT id FROM enterprise_chat_projects
                WHERE id = ? AND user_id = ?
                """,
                (project_id, user_id),
            ).fetchone() is None:
                return False, 0
            session_row = conn.execute(
                """
                SELECT id FROM enterprise_chat_sessions
                WHERE id = ? AND project_id = ? AND user_id = ?
                """,
                (session_id, project_id, user_id),
            ).fetchone()
            if session_row is None:
                return False, 0
            admitted = conn.execute(
                """
                SELECT 1 FROM enterprise_chat_messages
                WHERE session_id = ? AND project_event_seq IS NOT NULL LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            mapped = conn.execute(
                """
                SELECT 1 FROM enterprise_chat_memory_episodes
                WHERE session_id = ? AND project_id = ? AND user_id = ? LIMIT 1
                """,
                (session_id, project_id, user_id),
            ).fetchone()
            messages_cursor = conn.execute(
                "DELETE FROM enterprise_chat_messages WHERE session_id = ?",
                (session_id,),
            )
            conn.execute(
                "DELETE FROM enterprise_chat_sessions WHERE id = ?", (session_id,)
            )
            if admitted is not None or mapped is not None:
                group, _ = self._ensure_sqlite_chat_memory_group(
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
                self._enqueue_sqlite_chat_memory_rebuild(
                    conn,
                    group,
                    fingerprint,
                    bound_graph_fingerprint,
                    actor_user_id=actor_user_id or user_id,
                    actor_tenant_id=actor_tenant_id,
                    target_session_id=session_id,
                    target_message_id=None,
                )
            return True, int(messages_cursor.rowcount)

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

        def write(conn: sqlite3.Connection) -> tuple[bool, int, int]:
            if conn.execute(
                "SELECT id FROM enterprise_users WHERE id = ?", (user_id,)
            ).fetchone() is None:
                return False, 0, 0
            project_row = conn.execute(
                """
                SELECT id FROM enterprise_chat_projects
                WHERE id = ? AND user_id = ?
                """,
                (project_id, user_id),
            ).fetchone()
            if project_row is None:
                return False, 0, 0
            # Materialize child rows before the durable group/outbox rows.
            conn.execute(
                """
                SELECT id FROM enterprise_chat_sessions
                WHERE project_id = ? AND user_id = ? ORDER BY id
                """,
                (project_id, user_id),
            ).fetchall()
            conn.execute(
                """
                SELECT id FROM enterprise_chat_messages
                WHERE project_id = ? AND user_id = ? ORDER BY session_id, seq, id
                """,
                (project_id, user_id),
            ).fetchall()
            self._enqueue_sqlite_chat_memory_purge(
                conn,
                user_id,
                project_id,
                fingerprint,
                graph_fingerprint,
                actor_user_id=actor_user_id or user_id,
                actor_tenant_id=actor_tenant_id,
            )
            messages_cursor = conn.execute(
                "DELETE FROM enterprise_chat_messages WHERE project_id = ?",
                (project_id,),
            )
            sessions_cursor = conn.execute(
                "DELETE FROM enterprise_chat_sessions WHERE project_id = ?",
                (project_id,),
            )
            conn.execute(
                "DELETE FROM enterprise_chat_projects WHERE id = ?", (project_id,)
            )
            return True, int(sessions_cursor.rowcount), int(messages_cursor.rowcount)

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

        def write(conn: sqlite3.Connection) -> bool:
            current_row = conn.execute(
                "SELECT * FROM enterprise_users WHERE id = ?", (user_id,)
            ).fetchone()
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
            current_user = EnterpriseUserRecord.from_row(current_row)
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
                membership_rows = conn.execute(
                    """
                    SELECT * FROM enterprise_tenant_memberships
                    WHERE user_id = ? ORDER BY tenant_id ASC
                    """,
                    (user_id,),
                ).fetchall()
                _assert_enterprise_user_membership_precondition(
                    user_id,
                    [
                        EnterpriseTenantMembershipRecord.from_row(row)
                        for row in membership_rows
                    ],
                    expected_membership=expected_membership,
                )

            project_rows = conn.execute(
                """
                SELECT project_id FROM (
                    SELECT id AS project_id FROM enterprise_chat_projects
                    WHERE user_id = ?
                    UNION
                    SELECT project_id FROM enterprise_chat_memory_groups
                    WHERE user_id = ?
                    UNION
                    SELECT project_id FROM enterprise_chat_memory_generations
                    WHERE user_id = ?
                    UNION
                    SELECT project_id FROM enterprise_chat_memory_episodes
                    WHERE user_id = ?
                    UNION
                    SELECT project_id FROM enterprise_chat_memory_outbox
                    WHERE user_id = ?
                ) ORDER BY project_id ASC
                """,
                (user_id, user_id, user_id, user_id, user_id),
            ).fetchall()
            for project_row in project_rows:
                project_id = str(project_row["project_id"])
                conn.execute(
                    """
                    SELECT id FROM enterprise_chat_sessions
                    WHERE project_id = ? AND user_id = ? ORDER BY id
                    """,
                    (project_id, user_id),
                ).fetchall()
                conn.execute(
                    """
                    SELECT id FROM enterprise_chat_messages
                    WHERE project_id = ? AND user_id = ?
                    ORDER BY session_id, seq, id
                    """,
                    (project_id, user_id),
                ).fetchall()
                self._enqueue_sqlite_chat_memory_purge(
                    conn,
                    user_id,
                    project_id,
                    fingerprint,
                    graph_fingerprint,
                    actor_user_id=actor_user_id or user_id,
                    actor_tenant_id=actor_tenant_id,
                )

            # Person identity lifecycle: revoke person login sessions pointing
            # at this account and remove person-account links BEFORE deleting
            # the account row (see delete_enterprise_user for rationale).
            now = utc_now_iso()
            self._sqlite_revoke_person_sessions_locked(
                conn,
                None,
                account_id=user_id,
                actor_user_id=actor_user_id,
                now=now,
                audit_event_type="person_session_revoked_by_account_change",
            )
            self._revoke_person_kb_shares_locked(
                conn,
                either_side_account_id=user_id,
                actor_user_id=actor_user_id,
                now=now,
                audit_event_type="person_kb_share_revoked_by_account_change",
                reason="account_deleted",
            )
            conn.execute(
                "DELETE FROM enterprise_person_account_links WHERE account_id = ?",
                (user_id,),
            )

            conn.execute(
                "DELETE FROM enterprise_tenant_memberships WHERE user_id = ?",
                (user_id,),
            )
            conn.execute(
                "DELETE FROM enterprise_tenant_user_kb_overrides WHERE user_id = ?",
                (user_id,),
            )
            conn.execute(
                "DELETE FROM enterprise_kb_acl WHERE user_id = ?", (user_id,)
            )
            conn.execute(
                "DELETE FROM enterprise_user_kb_query_settings WHERE user_id = ?",
                (user_id,),
            )
            conn.execute(
                "DELETE FROM enterprise_chat_messages WHERE user_id = ?", (user_id,)
            )
            conn.execute(
                "DELETE FROM enterprise_chat_sessions WHERE user_id = ?", (user_id,)
            )
            conn.execute(
                "DELETE FROM enterprise_chat_projects WHERE user_id = ?", (user_id,)
            )
            cursor = conn.execute(
                "DELETE FROM enterprise_users WHERE id = ?", (user_id,)
            )
            return bool(cursor.rowcount)

        return await self._write(write)

    async def get_chat_memory_group(
        self, user_id: str, project_id: str
    ) -> ChatMemoryGroupRecord | None:
        await self._ensure_initialized()
        with self._connect() as conn:
            return self._get_sqlite_chat_memory_group(conn, user_id, project_id)

    async def get_chat_memory_read_token(
        self, user_id: str, project_id: str
    ) -> ChatMemoryReadToken | None:
        """Read one complete logical/active-generation identity atomically."""

        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
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
                WHERE groups.user_id = ? AND groups.project_id = ?
                """,
                (user_id, project_id),
            ).fetchone()
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
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM enterprise_chat_memory_generations
                WHERE user_id = ? AND project_id = ? AND generation = ?
                """,
                (user_id, project_id, int(generation)),
            ).fetchone()
        return ChatMemoryGenerationRecord.from_row(row) if row is not None else None

    async def list_chat_memory_generations(
        self, user_id: str, project_id: str
    ) -> list[ChatMemoryGenerationRecord]:
        await self._ensure_initialized()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM enterprise_chat_memory_generations
                WHERE user_id = ? AND project_id = ?
                ORDER BY generation ASC
                """,
                (user_id, project_id),
            ).fetchall()
        return [ChatMemoryGenerationRecord.from_row(row) for row in rows]

    async def get_chat_memory_event(
        self, event_id: str
    ) -> ChatMemoryOutboxEventRecord | None:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM enterprise_chat_memory_outbox WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return ChatMemoryOutboxEventRecord.from_row(row) if row is not None else None

    async def get_chat_memory_event_by_sequence(
        self, user_id: str, project_id: str, event_seq: int
    ) -> ChatMemoryOutboxEventRecord | None:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM enterprise_chat_memory_outbox
                WHERE user_id = ? AND project_id = ? AND event_seq = ?
                """,
                (user_id, project_id, int(event_seq)),
            ).fetchone()
        return ChatMemoryOutboxEventRecord.from_row(row) if row is not None else None

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
                clauses.append(f"{column} = ?")
                params.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM enterprise_chat_memory_outbox
                {where}
                ORDER BY user_id, project_id, event_seq
                LIMIT ?
                """,
                (*params, limit),
            ).fetchall()
        return [ChatMemoryOutboxEventRecord.from_row(row) for row in rows]

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
            clauses.append("status = ?")
            params.append(status)
        if event_type is not None:
            clauses.append("event_type = ?")
            params.append(event_type)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            value = conn.execute(
                f"SELECT COUNT(*) FROM enterprise_chat_memory_outbox {where}",
                params,
            ).fetchone()[0]
        return int(value)

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
        with self._connect() as conn:
            batch_rows = conn.execute(
                """
                SELECT project_event_seq, append_batch_id
                FROM enterprise_chat_messages
                WHERE user_id = ? AND project_id = ?
                  AND project_event_seq IS NOT NULL
                  AND append_batch_id IS NOT NULL
                  AND project_event_seq > ? AND project_event_seq <= ?
                GROUP BY project_event_seq, append_batch_id
                ORDER BY project_event_seq ASC
                LIMIT ?
                """,
                (user_id, project_id, after, cutoff, limit),
            ).fetchall()
            if not batch_rows:
                return []
            event_seqs = [int(row["project_event_seq"]) for row in batch_rows]
            placeholders = ",".join("?" for _ in event_seqs)
            message_rows = conn.execute(
                f"""
                SELECT * FROM enterprise_chat_messages
                WHERE user_id = ? AND project_id = ?
                  AND project_event_seq IN ({placeholders})
                ORDER BY project_event_seq ASC, session_id ASC, seq ASC, id ASC
                """,
                (user_id, project_id, *event_seqs),
            ).fetchall()
        by_event: dict[int, list[ChatMessageRecord]] = {
            event_seq: [] for event_seq in event_seqs
        }
        for row in message_rows:
            record = ChatMessageRecord.from_row(row)
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

        def write(conn: sqlite3.Connection) -> ChatMemoryOutboxEventRecord | None:
            group = self._get_sqlite_chat_memory_group(conn, user_id, project_id)
            if group is not None:
                self._assert_sqlite_chat_memory_graph_store_invariant(
                    conn, group, graph_fingerprint
                )
                if group.state in {"deleting", "deleted"}:
                    return None
            else:
                group, _ = self._ensure_sqlite_chat_memory_group(
                    conn,
                    user_id,
                    project_id,
                    fingerprint,
                    graph_fingerprint,
                    generation_state="building",
                )
            existing = conn.execute(
                """
                SELECT * FROM enterprise_chat_memory_outbox
                WHERE user_id = ? AND project_id = ? AND event_type = 'rebuild'
                  AND generation = ? AND config_fingerprint = ?
                  AND graph_store_fingerprint = ?
                  AND status IN ('pending', 'running', 'retry_wait')
                ORDER BY event_seq DESC LIMIT 1
                """,
                (
                    user_id,
                    project_id,
                    group.desired_generation,
                    fingerprint,
                    graph_fingerprint,
                ),
            ).fetchone()
            if existing is not None:
                return ChatMemoryOutboxEventRecord.from_row(existing)
            return self._enqueue_sqlite_chat_memory_rebuild(
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

        def write(conn: sqlite3.Connection) -> ChatMemoryOutboxEventRecord | None:
            group = self._get_sqlite_chat_memory_group(conn, user_id, project_id)
            if group is not None and group.state == "deleted":
                return None
            existing = conn.execute(
                """
                SELECT * FROM enterprise_chat_memory_outbox
                WHERE user_id = ? AND project_id = ? AND event_type = 'purge'
                  AND status IN ('pending', 'running', 'retry_wait')
                ORDER BY event_seq DESC LIMIT 1
                """,
                (user_id, project_id),
            ).fetchone()
            if existing is not None:
                return ChatMemoryOutboxEventRecord.from_row(existing)
            return self._enqueue_sqlite_chat_memory_purge(
                conn,
                user_id,
                project_id,
                fingerprint,
                graph_fingerprint,
                actor_user_id=actor_user_id,
                actor_tenant_id=actor_tenant_id,
            )

        return await self._write(write)

    def _materialize_sqlite_chat_memory_rebuild_batches(
        self,
        conn: sqlite3.Connection,
        user_id: str,
        project_id: str,
        cutoff: int,
    ) -> list[ChatMemoryReplayBatch]:
        """Fetch full replay rows only after the aggregate cap preflight passes."""

        rows = conn.execute(
            """
            SELECT * FROM enterprise_chat_messages
            WHERE user_id = ? AND project_id = ?
              AND project_event_seq IS NOT NULL
              AND append_batch_id IS NOT NULL
              AND project_event_seq <= ?
            ORDER BY project_event_seq ASC, session_id ASC, seq ASC, id ASC
            """,
            (user_id, project_id, int(cutoff)),
        ).fetchall()
        return _chat_memory_replay_batches_from_messages(
            [ChatMessageRecord.from_row(row) for row in rows]
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

        def write(conn: sqlite3.Connection) -> ChatMemoryRebuildSnapshot | None:
            state = self._require_sqlite_chat_memory_running_claim(
                conn, event_id, claim_token
            )
            stale = self._resolve_sqlite_stale_chat_memory_execution(
                conn, state, claim_token
            )
            if stale is not None:
                return None
            if not self._chat_memory_runtime_identity_matches(
                state.event, fingerprint, graph_fingerprint
            ):
                self._retry_sqlite_chat_memory_runtime_mismatch(
                    conn,
                    state,
                    claim_token,
                    fingerprint,
                    graph_fingerprint,
                    retry_delay_seconds=1.0,
                )
                return None
            self._validate_chat_memory_execution_fence(
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
            metrics = conn.execute(
                """
                WITH source AS (
                    SELECT project_event_seq, append_batch_id, content
                    FROM enterprise_chat_messages
                    WHERE user_id = ? AND project_id = ?
                      AND project_event_seq IS NOT NULL
                      AND append_batch_id IS NOT NULL
                      AND project_event_seq <= ?
                ), batch_metrics AS (
                    SELECT COUNT(*) AS batch_count FROM (
                        SELECT project_event_seq, append_batch_id
                        FROM source
                        GROUP BY project_event_seq, append_batch_id
                    )
                )
                SELECT COUNT(*) AS message_count,
                       COALESCE(SUM(length(CAST(content AS BLOB))), 0) AS byte_count,
                       (SELECT batch_count FROM batch_metrics) AS batch_count
                FROM source
                """,
                (event.user_id, event.project_id, cutoff),
            ).fetchone()
            assert metrics is not None
            batch_count = int(metrics["batch_count"] or 0)
            message_count = int(metrics["message_count"] or 0)
            byte_count = int(metrics["byte_count"] or 0)
            now = utc_now_iso()
            if message_count > message_cap or byte_count > byte_cap:
                error_message = (
                    "Rebuild snapshot exceeds hard cap: "
                    f"messages={message_count}/{message_cap}, "
                    f"bytes={byte_count}/{byte_cap}"
                )
                conn.execute(
                    """
                    UPDATE enterprise_chat_memory_generations
                    SET snapshot_cutoff = ?, replay_batch_count = ?,
                        replay_message_count = ?, replay_byte_count = ?,
                        snapshot_digest = NULL,
                        last_error_code = 'rebuild_snapshot_hard_cap_exceeded',
                        last_error_message = ?, last_error_at = ?, updated_at = ?
                    WHERE user_id = ? AND project_id = ? AND generation = ?
                      AND state = 'building'
                    """,
                    (
                        cutoff,
                        batch_count,
                        message_count,
                        byte_count,
                        error_message,
                        now,
                        now,
                        event.user_id,
                        event.project_id,
                        event.generation,
                    ),
                )
                cursor = conn.execute(
                    """
                    UPDATE enterprise_chat_memory_outbox
                    SET status = 'dead_letter', snapshot_cutoff = ?,
                        snapshot_batch_count = ?, snapshot_message_count = ?,
                        snapshot_byte_count = ?, snapshot_digest = NULL,
                        claim_token = NULL,
                        claimed_by = NULL, claimed_at = NULL,
                        side_effect_started_at = NULL,
                        side_effect_state_version = NULL, completed_at = ?,
                        last_error_code = 'rebuild_snapshot_hard_cap_exceeded',
                        last_error_message = ?, last_error_at = ?, updated_at = ?
                    WHERE event_id = ? AND status = 'running' AND claim_token = ?
                      AND side_effect_started_at IS NULL
                    """,
                    (
                        cutoff,
                        batch_count,
                        message_count,
                        byte_count,
                        now,
                        error_message,
                        now,
                        now,
                        event_id,
                        claim_token,
                    ),
                )
                if cursor.rowcount != 1:
                    raise MetadataConflictError(
                        "chat_memory_event",
                        event_id,
                        expected={"status": "running", "claim_token": claim_token},
                        current={"status": event.status, "claim_token": event.claim_token},
                    )
                conn.execute(
                    """
                    UPDATE enterprise_chat_memory_groups
                    SET state = 'failed', state_version = state_version + 1,
                        last_error_code = 'rebuild_snapshot_hard_cap_exceeded',
                        last_error_message = ?, last_error_at = ?, updated_at = ?
                    WHERE user_id = ? AND project_id = ?
                    """,
                    (
                        error_message,
                        now,
                        now,
                        event.user_id,
                        event.project_id,
                    ),
                )
                return None

            replay_batches = self._materialize_sqlite_chat_memory_rebuild_batches(
                conn,
                event.user_id,
                event.project_id,
                cutoff,
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

            generation_cursor = conn.execute(
                """
                UPDATE enterprise_chat_memory_generations
                SET snapshot_cutoff = ?, replay_batch_count = ?,
                    replay_message_count = ?, replay_byte_count = ?,
                    snapshot_digest = ?,
                    last_error_code = NULL, last_error_message = NULL,
                    last_error_at = NULL, updated_at = ?
                WHERE user_id = ? AND project_id = ? AND generation = ?
                  AND state = 'building' AND config_fingerprint = ?
                  AND graph_store_fingerprint = ?
                  AND graph_group_id = ?
                """,
                (
                    cutoff,
                    batch_count,
                    message_count,
                    byte_count,
                    snapshot_digest,
                    now,
                    event.user_id,
                    event.project_id,
                    event.generation,
                    fingerprint,
                    graph_fingerprint,
                    event.graph_group_id,
                ),
            )
            event_cursor = conn.execute(
                """
                UPDATE enterprise_chat_memory_outbox
                SET snapshot_cutoff = ?, snapshot_batch_count = ?,
                    snapshot_message_count = ?, snapshot_byte_count = ?,
                    snapshot_digest = ?,
                    updated_at = ?
                WHERE event_id = ? AND status = 'running' AND claim_token = ?
                  AND side_effect_started_at IS NULL
                """,
                (
                    cutoff,
                    batch_count,
                    message_count,
                    byte_count,
                    snapshot_digest,
                    now,
                    event_id,
                    claim_token,
                ),
            )
            if generation_cursor.rowcount != 1 or event_cursor.rowcount != 1:
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

        def write(conn: sqlite3.Connection) -> ChatMemoryRebuildTargetSet | None:
            state = self._require_sqlite_chat_memory_running_claim(
                conn, event_id, claim_token
            )
            stale = self._resolve_sqlite_stale_chat_memory_execution(
                conn, state, claim_token
            )
            if stale is not None:
                return None
            if not self._chat_memory_runtime_identity_matches(
                state.event, fingerprint, graph_fingerprint
            ):
                self._retry_sqlite_chat_memory_runtime_mismatch(
                    conn,
                    state,
                    claim_token,
                    fingerprint,
                    graph_fingerprint,
                    retry_delay_seconds=1.0,
                )
                return None
            self._validate_chat_memory_execution_fence(
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
            self._assert_sqlite_chat_memory_graph_store_invariant(
                conn, state.group, graph_fingerprint
            )
            return ChatMemoryRebuildTargetSet(
                event_id=event.event_id,
                user_id=event.user_id,
                project_id=event.project_id,
                logical_group_id=state.group.logical_group_id,
                group_ids=self._sqlite_chat_memory_rebuild_group_ids(conn, event),
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
        """Claim one eligible FIFO head event for SQLite test parity."""

        fingerprint = _validate_chat_memory_fingerprint(runtime_fingerprint)
        graph_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            fingerprint, runtime_graph_store_fingerprint
        )
        claimed_by = _validate_chat_memory_worker_id(worker_id)
        claimable_types = _normalize_chat_memory_event_types(event_types)
        if not claimable_types:
            return None
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> ChatMemoryOutboxEventRecord | None:
            now = utc_now_iso()
            type_placeholders = ",".join("?" for _ in claimable_types)
            row = conn.execute(
                f"""
                SELECT candidate.*
                FROM enterprise_chat_memory_outbox AS candidate
                WHERE candidate.status IN ('pending', 'retry_wait')
                  AND candidate.available_at <= ?
                  AND (
                      (candidate.event_type = 'purge'
                       AND candidate.graph_store_fingerprint = ?)
                      OR
                      (candidate.event_type IN ('ingest', 'rebuild')
                       AND candidate.config_fingerprint = ?
                       AND candidate.graph_store_fingerprint = ?)
                  )
                  AND candidate.event_type IN ({type_placeholders})
                  AND NOT EXISTS (
                      SELECT 1
                      FROM enterprise_chat_memory_outbox AS blocker
                      WHERE blocker.user_id = candidate.user_id
                        AND blocker.project_id = candidate.project_id
                        AND blocker.event_seq < candidate.event_seq
                        AND blocker.status IN (
                            'pending', 'running', 'retry_wait', 'dead_letter'
                        )
                  )
                ORDER BY candidate.available_at ASC,
                         candidate.event_seq ASC,
                         candidate.user_id ASC,
                         candidate.project_id ASC,
                         candidate.event_id ASC
                LIMIT 1
                """,
                (
                    now,
                    graph_fingerprint,
                    fingerprint,
                    graph_fingerprint,
                    *claimable_types,
                ),
            ).fetchone()
            if row is None:
                return None
            claim_token = _new_chat_memory_claim_token()
            cursor = conn.execute(
                """
                UPDATE enterprise_chat_memory_outbox
                SET status = 'running', attempt_no = attempt_no + 1,
                    claim_token = ?, claimed_by = ?, claimed_at = ?,
                    side_effect_started_at = NULL,
                    side_effect_state_version = NULL, completed_at = NULL,
                    last_error_code = NULL, last_error_message = NULL,
                    last_error_at = NULL, updated_at = ?
                WHERE event_id = ? AND status IN ('pending', 'retry_wait')
                  AND available_at <= ?
                  AND (
                      (event_type = 'purge' AND graph_store_fingerprint = ?)
                      OR
                      (event_type IN ('ingest', 'rebuild')
                       AND config_fingerprint = ?
                       AND graph_store_fingerprint = ?)
                  )
                """,
                (
                    claim_token,
                    claimed_by,
                    now,
                    now,
                    row["event_id"],
                    now,
                    graph_fingerprint,
                    fingerprint,
                    graph_fingerprint,
                ),
            )
            if cursor.rowcount != 1:
                return None
            claimed = conn.execute(
                "SELECT * FROM enterprise_chat_memory_outbox WHERE event_id = ?",
                (row["event_id"],),
            ).fetchone()
            assert claimed is not None
            return ChatMemoryOutboxEventRecord.from_row(claimed)

        return await self._write(write)

    @asynccontextmanager
    async def chat_memory_group_execution_guard(
        self, logical_group_id: str, *, wait: bool = True
    ) -> AsyncIterator[bool]:
        """Process-local SQLite parity guard for one logical memory group."""

        if not isinstance(logical_group_id, str) or not logical_group_id.strip():
            raise ValueError("Chat Memory logical_group_id must be non-empty")
        logical_group_id = logical_group_id.strip()
        await self._ensure_initialized()
        task = asyncio.current_task()
        state = self._chat_memory_guard_state.get()
        token: Any | None = None
        if state is None or state.owner_task is not task:
            state = _SQLiteChatMemoryGuardTaskState(owner_task=task, depths={})
            token = self._chat_memory_guard_state.set(state)
        depth = state.depths.get(logical_group_id, 0)
        if depth:
            state.depths[logical_group_id] = depth + 1
            try:
                yield True
            finally:
                remaining = state.depths[logical_group_id] - 1
                if remaining:
                    state.depths[logical_group_id] = remaining
                else:
                    state.depths.pop(logical_group_id, None)
                if token is not None:
                    self._chat_memory_guard_state.reset(token)
            return

        lock = _process_chat_memory_group_lock(self.db_path, logical_group_id)
        acquired = False
        try:
            acquired = await lock.acquire(wait=wait)
            if not acquired:
                yield False
                return
            state.depths[logical_group_id] = 1
            try:
                yield True
            finally:
                state.depths.pop(logical_group_id, None)
        finally:
            if acquired:
                await lock.release()
            if token is not None:
                self._chat_memory_guard_state.reset(token)

    async def get_chat_memory_execution_state(
        self, event_id: str
    ) -> ChatMemoryExecutionState | None:
        await self._ensure_initialized()
        with self._connect() as conn:
            return self._get_sqlite_chat_memory_execution_state(conn, event_id)

    async def mark_chat_memory_event_side_effect_started(
        self,
        event_id: str,
        claim_token: str,
        runtime_fingerprint: str,
        *,
        runtime_graph_store_fingerprint: str | None = None,
        fingerprint_retry_delay_seconds: float = 1.0,
    ) -> ChatMemoryOutboxEventRecord:
        """Begin a side effect only while the claimed execution fence is current.

        A stale execution is atomically superseded. A worker running the wrong
        runtime fingerprint releases the claim to retry_wait without setting a
        side-effect marker.
        """

        fingerprint = _validate_chat_memory_fingerprint(runtime_fingerprint)
        graph_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            fingerprint, runtime_graph_store_fingerprint
        )
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> ChatMemoryOutboxEventRecord:
            state = self._require_sqlite_chat_memory_running_claim(
                conn, event_id, claim_token
            )
            stale = self._resolve_sqlite_stale_chat_memory_execution(
                conn, state, claim_token
            )
            if stale is not None:
                return stale
            if not self._chat_memory_runtime_identity_matches(
                state.event, fingerprint, graph_fingerprint
            ):
                return self._retry_sqlite_chat_memory_runtime_mismatch(
                    conn,
                    state,
                    claim_token,
                    fingerprint,
                    graph_fingerprint,
                    retry_delay_seconds=fingerprint_retry_delay_seconds,
                )
            if state.event.side_effect_started_at is None:
                now = utc_now_iso()
                conn.execute(
                    """
                    UPDATE enterprise_chat_memory_outbox
                    SET side_effect_started_at = ?,
                        side_effect_state_version = ?, updated_at = ?
                    WHERE event_id = ? AND status = 'running'
                      AND claim_token = ?
                    """,
                    (
                        now,
                        state.group.state_version,
                        now,
                        event_id,
                        claim_token,
                    ),
                )
            row = conn.execute(
                "SELECT * FROM enterprise_chat_memory_outbox WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            assert row is not None
            return ChatMemoryOutboxEventRecord.from_row(row)

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

        def write(conn: sqlite3.Connection) -> ChatMemoryExecutionState | None:
            state = self._require_sqlite_chat_memory_running_claim(
                conn, event_id, claim_token
            )
            stale = self._resolve_sqlite_stale_chat_memory_execution(
                conn, state, claim_token
            )
            if stale is not None:
                return None
            self._validate_chat_memory_ingest_execution(
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
            now = utc_now_iso()
            expected_mapping = ChatMemoryEpisodeRecord(
                episode_uuid=episode_uuid,
                session_id=event.source_session_id,
                project_id=event.project_id,
                user_id=event.user_id,
                first_seq=event.first_seq,
                last_seq=event.last_seq,
                created_at=now,
                event_id=event.event_id,
                generation=event.generation,
                graph_group_id=event.graph_group_id,
                append_batch_id=event.append_batch_id,
                project_event_seq=event.event_seq,
            )
            self._insert_sqlite_chat_memory_historical_mapping(
                conn, expected_mapping
            )

            activate_first = (
                state.group.active_generation is None
                and state.group.state == "rebuilding"
                and state.generation.state == "building"
            )
            if activate_first:
                conn.execute(
                    """
                    UPDATE enterprise_chat_memory_generations
                    SET state = 'active', activated_at = ?, updated_at = ?,
                        last_error_code = NULL, last_error_message = NULL,
                        last_error_at = NULL
                    WHERE user_id = ? AND project_id = ? AND generation = ?
                    """,
                    (
                        now,
                        now,
                        event.user_id,
                        event.project_id,
                        event.generation,
                    ),
                )
                conn.execute(
                    """
                    UPDATE enterprise_chat_memory_groups
                    SET active_generation = ?, active_config_fingerprint = ?,
                        active_graph_store_fingerprint = ?,
                        state = 'active', state_version = state_version + 1,
                        active_rebuild_event_id = NULL, last_success_at = ?,
                        last_error_code = NULL, last_error_message = NULL,
                        last_error_at = NULL, updated_at = ?
                    WHERE user_id = ? AND project_id = ?
                    """,
                    (
                        event.generation,
                        fingerprint,
                        graph_fingerprint,
                        now,
                        now,
                        event.user_id,
                        event.project_id,
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE enterprise_chat_memory_groups
                    SET last_success_at = ?, last_error_code = NULL,
                        last_error_message = NULL, last_error_at = NULL,
                        updated_at = ?
                    WHERE user_id = ? AND project_id = ?
                    """,
                    (now, now, event.user_id, event.project_id),
                )
            cursor = conn.execute(
                """
                UPDATE enterprise_chat_memory_outbox
                SET status = 'succeeded', completed_at = ?, updated_at = ?,
                    last_error_code = NULL, last_error_message = NULL,
                    last_error_at = NULL
                WHERE event_id = ? AND status = 'running' AND claim_token = ?
                """,
                (now, now, event_id, claim_token),
            )
            if cursor.rowcount != 1:
                raise MetadataConflictError(
                    "chat_memory_event",
                    event_id,
                    expected={"status": "running", "claim_token": claim_token},
                    current={"status": event.status, "claim_token": event.claim_token},
                )
            final = self._get_sqlite_chat_memory_execution_state(conn, event_id)
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

        def write(conn: sqlite3.Connection) -> ChatMemoryExecutionState | None:
            state = self._require_sqlite_chat_memory_running_claim(
                conn, event_id, claim_token
            )
            stale = self._resolve_sqlite_stale_chat_memory_execution(
                conn, state, claim_token
            )
            if stale is not None:
                return None
            self._validate_chat_memory_ingest_execution(
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

            source_rows = conn.execute(
                """
                SELECT * FROM enterprise_chat_messages
                WHERE user_id = ? AND project_id = ? AND project_event_seq = ?
                ORDER BY seq ASC, id ASC
                """,
                (
                    event.user_id,
                    event.project_id,
                    event.event_seq,
                ),
            ).fetchall()
            source_messages = [
                ChatMessageRecord.from_row(row) for row in source_rows
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

            now = utc_now_iso()
            episode_uuid = _chat_memory_noop_episode_uuid(
                event_id=event.event_id,
                generation=event.generation,
                append_batch_id=event.append_batch_id,
            )
            self._insert_sqlite_chat_memory_historical_mapping(
                conn,
                ChatMemoryEpisodeRecord(
                    episode_uuid=episode_uuid,
                    session_id=event.source_session_id,
                    project_id=event.project_id,
                    user_id=event.user_id,
                    first_seq=event.first_seq,
                    last_seq=event.last_seq,
                    created_at=now,
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
                generation_cursor = conn.execute(
                    """
                    UPDATE enterprise_chat_memory_generations
                    SET state = 'active', activated_at = ?, updated_at = ?,
                        last_error_code = NULL, last_error_message = NULL,
                        last_error_at = NULL
                    WHERE user_id = ? AND project_id = ? AND generation = ?
                      AND state = 'building'
                    """,
                    (
                        now,
                        now,
                        event.user_id,
                        event.project_id,
                        event.generation,
                    ),
                )
                group_cursor = conn.execute(
                    """
                    UPDATE enterprise_chat_memory_groups
                    SET active_generation = ?, active_config_fingerprint = ?,
                        active_graph_store_fingerprint = ?,
                        state = 'active', state_version = state_version + 1,
                        active_rebuild_event_id = NULL, last_success_at = ?,
                        last_error_code = NULL, last_error_message = NULL,
                        last_error_at = NULL, updated_at = ?
                    WHERE user_id = ? AND project_id = ?
                      AND desired_generation = ? AND state_version = ?
                      AND state = 'rebuilding'
                    """,
                    (
                        event.generation,
                        fingerprint,
                        graph_fingerprint,
                        now,
                        now,
                        event.user_id,
                        event.project_id,
                        event.generation,
                        state.group.state_version,
                    ),
                )
                if generation_cursor.rowcount != 1 or group_cursor.rowcount != 1:
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
                conn.execute(
                    """
                    UPDATE enterprise_chat_memory_groups
                    SET last_success_at = ?, last_error_code = NULL,
                        last_error_message = NULL, last_error_at = NULL,
                        updated_at = ?
                    WHERE user_id = ? AND project_id = ?
                    """,
                    (now, now, event.user_id, event.project_id),
                )

            cursor = conn.execute(
                """
                UPDATE enterprise_chat_memory_outbox
                SET status = 'succeeded', completed_at = ?, updated_at = ?,
                    last_error_code = NULL, last_error_message = NULL,
                    last_error_at = NULL
                WHERE event_id = ? AND status = 'running' AND claim_token = ?
                  AND side_effect_started_at IS NULL
                  AND side_effect_state_version IS NULL
                """,
                (now, now, event_id, claim_token),
            )
            if cursor.rowcount != 1:
                raise MetadataConflictError(
                    "chat_memory_event",
                    event_id,
                    expected={"status": "running", "claim_token": claim_token},
                    current={"status": event.status, "claim_token": event.claim_token},
                )
            final = self._get_sqlite_chat_memory_execution_state(conn, event_id)
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

        def write(conn: sqlite3.Connection) -> ChatMemoryExecutionState | None:
            state = self._require_sqlite_chat_memory_running_claim(
                conn, event_id, claim_token
            )
            stale = self._resolve_sqlite_stale_chat_memory_execution(
                conn, state, claim_token
            )
            if stale is not None:
                return None
            self._validate_chat_memory_execution_fence(
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

            self._assert_sqlite_chat_memory_graph_store_invariant(
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
            authoritative_targets = self._sqlite_chat_memory_rebuild_group_ids(
                conn, event
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

            now = utc_now_iso()
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
                self._insert_sqlite_chat_memory_historical_mapping(
                    conn,
                    ChatMemoryEpisodeRecord(
                        episode_uuid=mapping.episode_uuid,
                        session_id=mapping.session_id,
                        project_id=event.project_id,
                        user_id=event.user_id,
                        first_seq=mapping.first_seq,
                        last_seq=mapping.last_seq,
                        created_at=now,
                        event_id=event.event_id,
                        generation=event.generation,
                        graph_group_id=event.graph_group_id,
                        append_batch_id=mapping.append_batch_id,
                        project_event_seq=mapping.project_event_seq,
                    ),
                )

            conn.execute(
                """
                UPDATE enterprise_chat_memory_generations
                SET state = 'purged', cleared_at = COALESCE(cleared_at, ?),
                    updated_at = ?, last_error_code = NULL,
                    last_error_message = NULL, last_error_at = NULL
                WHERE user_id = ? AND project_id = ? AND generation <> ?
                """,
                (
                    now,
                    now,
                    event.user_id,
                    event.project_id,
                    event.generation,
                ),
            )

            activated = conn.execute(
                """
                UPDATE enterprise_chat_memory_generations
                SET state = 'active', activated_at = ?, updated_at = ?,
                    last_error_code = NULL, last_error_message = NULL,
                    last_error_at = NULL
                WHERE user_id = ? AND project_id = ? AND generation = ?
                  AND state = 'building' AND snapshot_cutoff = ?
                  AND replay_batch_count = ? AND replay_message_count = ?
                  AND replay_byte_count = ? AND snapshot_digest = ?
                """,
                (
                    now,
                    now,
                    event.user_id,
                    event.project_id,
                    event.generation,
                    snapshot.snapshot_cutoff,
                    batch_count,
                    message_count,
                    byte_count,
                    invocation_digest,
                ),
            )
            if activated.rowcount != 1:
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
                conn.execute(
                    """
                    UPDATE enterprise_chat_memory_outbox
                    SET status = 'superseded', superseded_by_event_id = ?,
                        completed_at = ?, updated_at = ?
                    WHERE user_id = ? AND project_id = ?
                      AND event_type = 'ingest'
                      AND status IN ('pending', 'retry_wait', 'dead_letter')
                      AND event_seq = ? AND event_seq <= ?
                      AND append_batch_id = ?
                    """,
                    (
                        event.event_id,
                        now,
                        now,
                        event.user_id,
                        event.project_id,
                        batch.project_event_seq,
                        snapshot.snapshot_cutoff,
                        batch.append_batch_id,
                    ),
                )

            group_cursor = conn.execute(
                """
                UPDATE enterprise_chat_memory_groups
                SET active_generation = ?, active_config_fingerprint = ?,
                    active_graph_store_fingerprint = ?,
                    state = 'active', state_version = state_version + 1,
                    active_rebuild_event_id = NULL, last_success_at = ?,
                    last_error_code = NULL, last_error_message = NULL,
                    last_error_at = NULL, updated_at = ?
                WHERE user_id = ? AND project_id = ?
                  AND desired_generation = ? AND desired_config_fingerprint = ?
                  AND desired_graph_store_fingerprint = ?
                  AND state = 'rebuilding' AND state_version = ?
                  AND active_rebuild_event_id = ?
                """,
                (
                    event.generation,
                    fingerprint,
                    graph_fingerprint,
                    now,
                    now,
                    event.user_id,
                    event.project_id,
                    event.generation,
                    fingerprint,
                    graph_fingerprint,
                    snapshot.group_state_version,
                    event.event_id,
                ),
            )
            event_cursor = conn.execute(
                """
                UPDATE enterprise_chat_memory_outbox
                SET status = 'succeeded', completed_at = ?, updated_at = ?,
                    last_error_code = NULL, last_error_message = NULL,
                    last_error_at = NULL
                WHERE event_id = ? AND status = 'running' AND claim_token = ?
                  AND side_effect_started_at IS NOT NULL
                  AND side_effect_state_version = ? AND snapshot_cutoff = ?
                  AND snapshot_batch_count = ? AND snapshot_message_count = ?
                  AND snapshot_byte_count = ? AND snapshot_digest = ?
                """,
                (
                    now,
                    now,
                    event_id,
                    claim_token,
                    snapshot.group_state_version,
                    snapshot.snapshot_cutoff,
                    batch_count,
                    message_count,
                    byte_count,
                    invocation_digest,
                ),
            )
            if group_cursor.rowcount != 1 or event_cursor.rowcount != 1:
                raise MetadataConflictError(
                    "chat_memory_rebuild_finalize",
                    event_id,
                    expected={"group": "current", "event": "running/current"},
                    current={
                        "group_state": state.group.state,
                        "event_status": event.status,
                    },
                )
            final = self._get_sqlite_chat_memory_execution_state(conn, event_id)
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

        def write(conn: sqlite3.Connection) -> ChatMemoryPurgeTargetSet | None:
            state = self._require_sqlite_chat_memory_running_claim(
                conn, event_id, claim_token
            )
            self._assert_sqlite_chat_memory_graph_store_invariant(
                conn,
                state.group,
                _resolve_chat_memory_graph_store_fingerprint(
                    state.event.config_fingerprint,
                    state.event.graph_store_fingerprint,
                ),
            )
            stale = self._resolve_sqlite_stale_chat_memory_execution(
                conn, state, claim_token
            )
            if stale is not None:
                return None
            if not self._chat_memory_runtime_identity_matches(
                state.event, fingerprint, graph_fingerprint
            ):
                self._retry_sqlite_chat_memory_runtime_mismatch(
                    conn,
                    state,
                    claim_token,
                    fingerprint,
                    graph_fingerprint,
                    retry_delay_seconds=1.0,
                )
                return None
            self._validate_chat_memory_execution_fence(
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
            group_ids = self._sqlite_chat_memory_purge_group_ids(
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

        def write(conn: sqlite3.Connection) -> ChatMemoryExecutionState | None:
            state = self._require_sqlite_chat_memory_running_claim(
                conn, event_id, claim_token
            )
            self._assert_sqlite_chat_memory_graph_store_invariant(
                conn,
                state.group,
                _resolve_chat_memory_graph_store_fingerprint(
                    state.event.config_fingerprint,
                    state.event.graph_store_fingerprint,
                ),
            )
            stale = self._resolve_sqlite_stale_chat_memory_execution(
                conn, state, claim_token
            )
            if stale is not None:
                return None
            self._validate_chat_memory_execution_fence(
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

            authoritative_expected = self._sqlite_chat_memory_purge_group_ids(
                conn,
                event.user_id,
                event.project_id,
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

            now = utc_now_iso()
            conn.execute(
                """
                UPDATE enterprise_chat_memory_generations
                SET state = 'purged', cleared_at = COALESCE(cleared_at, ?),
                    updated_at = ?, last_error_code = NULL,
                    last_error_message = NULL, last_error_at = NULL
                WHERE user_id = ? AND project_id = ?
                """,
                (now, now, event.user_id, event.project_id),
            )
            conn.execute(
                """
                DELETE FROM enterprise_chat_memory_episodes
                WHERE user_id = ? AND project_id = ?
                """,
                (event.user_id, event.project_id),
            )
            group_cursor = conn.execute(
                """
                UPDATE enterprise_chat_memory_groups
                SET state = 'deleted', active_generation = NULL,
                    active_config_fingerprint = NULL,
                    active_graph_store_fingerprint = NULL,
                    active_rebuild_event_id = NULL,
                    state_version = state_version + 1, deleted_at = ?,
                    last_success_at = ?, last_error_code = NULL,
                    last_error_message = NULL, last_error_at = NULL,
                    updated_at = ?
                WHERE user_id = ? AND project_id = ? AND state = 'deleting'
                  AND desired_generation = ?
                  AND desired_graph_store_fingerprint = ? AND state_version = ?
                """,
                (
                    now,
                    now,
                    now,
                    event.user_id,
                    event.project_id,
                    event.generation,
                    graph_fingerprint,
                    event.side_effect_state_version,
                ),
            )
            event_cursor = conn.execute(
                """
                UPDATE enterprise_chat_memory_outbox
                SET status = 'succeeded', completed_at = ?, updated_at = ?,
                    last_error_code = NULL, last_error_message = NULL,
                    last_error_at = NULL
                WHERE event_id = ? AND status = 'running' AND claim_token = ?
                  AND side_effect_started_at IS NOT NULL
                  AND side_effect_state_version = ?
                """,
                (
                    now,
                    now,
                    event_id,
                    claim_token,
                    event.side_effect_state_version,
                ),
            )
            if group_cursor.rowcount != 1 or event_cursor.rowcount != 1:
                raise MetadataConflictError(
                    "chat_memory_purge_finalize",
                    event_id,
                    expected={"group": "current", "event": "running/current"},
                    current={
                        "group_state": state.group.state,
                        "event_status": event.status,
                    },
                )
            final = self._get_sqlite_chat_memory_execution_state(conn, event_id)
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
        """CAS a known pre-side-effect failure to retry_wait or dead_letter."""

        fingerprint = _validate_chat_memory_fingerprint(runtime_fingerprint)
        graph_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            fingerprint, runtime_graph_store_fingerprint
        )
        await self._ensure_initialized()
        max_attempts = max(1, int(max_attempts))

        def write(conn: sqlite3.Connection) -> ChatMemoryOutboxEventRecord:
            state = self._require_sqlite_chat_memory_running_claim(
                conn, event_id, claim_token
            )
            stale = self._resolve_sqlite_stale_chat_memory_execution(
                conn, state, claim_token
            )
            if stale is not None:
                return stale
            if not self._chat_memory_runtime_identity_matches(
                state.event, fingerprint, graph_fingerprint
            ):
                return self._retry_sqlite_chat_memory_runtime_mismatch(
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
            now_dt = datetime.now(timezone.utc)
            now = now_dt.isoformat()
            dead_letter = (
                retry_delay_seconds is None
                or state.event.attempt_no >= max_attempts
            )
            status: ChatMemoryEventStatus = (
                "dead_letter" if dead_letter else "retry_wait"
            )
            available_at = (
                now
                if dead_letter
                else (
                    now_dt
                    + timedelta(seconds=max(0.0, float(retry_delay_seconds or 0)))
                ).isoformat()
            )
            conn.execute(
                """
                UPDATE enterprise_chat_memory_outbox
                SET status = ?, available_at = ?, claim_token = NULL,
                    claimed_by = NULL, claimed_at = NULL,
                    side_effect_state_version = NULL,
                    completed_at = CASE WHEN ? = 'dead_letter' THEN ? ELSE NULL END,
                    last_error_code = ?, last_error_message = ?,
                    last_error_at = ?, updated_at = ?
                WHERE event_id = ? AND status = 'running' AND claim_token = ?
                """,
                (
                    status,
                    available_at,
                    status,
                    now,
                    error_code,
                    error_message,
                    now,
                    now,
                    event_id,
                    claim_token,
                ),
            )
            if dead_letter:
                conn.execute(
                    """
                    UPDATE enterprise_chat_memory_groups
                    SET state = 'failed', state_version = state_version + 1,
                        last_error_code = ?, last_error_message = ?,
                        last_error_at = ?, updated_at = ?
                    WHERE user_id = ? AND project_id = ?
                    """,
                    (
                        error_code,
                        error_message,
                        now,
                        now,
                        state.event.user_id,
                        state.event.project_id,
                    ),
                )
            row = conn.execute(
                "SELECT * FROM enterprise_chat_memory_outbox WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            assert row is not None
            return ChatMemoryOutboxEventRecord.from_row(row)

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
        fingerprint = _validate_chat_memory_fingerprint(runtime_fingerprint)
        graph_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            fingerprint, runtime_graph_store_fingerprint
        )
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> ChatMemoryOutboxEventRecord:
            state = self._get_sqlite_chat_memory_execution_state(conn, event_id)
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
            return self._enqueue_sqlite_chat_memory_rebuild(
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
        """Retry or dead-letter a purge failure known to precede any clear."""

        fingerprint = _validate_chat_memory_fingerprint(runtime_fingerprint)
        graph_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            fingerprint, runtime_graph_store_fingerprint
        )
        max_attempts = max(1, int(max_attempts))
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> ChatMemoryOutboxEventRecord:
            state = self._require_sqlite_chat_memory_running_claim(
                conn, event_id, claim_token
            )
            stale = self._resolve_sqlite_stale_chat_memory_execution(
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
            if not self._chat_memory_runtime_identity_matches(
                state.event, fingerprint, graph_fingerprint
            ):
                return self._retry_sqlite_chat_memory_runtime_mismatch(
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
            now_dt = datetime.now(timezone.utc)
            now = now_dt.isoformat()
            dead_letter = (
                retry_delay_seconds is None
                or state.event.attempt_no >= max_attempts
            )
            status: ChatMemoryEventStatus = (
                "dead_letter" if dead_letter else "retry_wait"
            )
            available = (
                now
                if dead_letter
                else (
                    now_dt
                    + timedelta(seconds=max(0.0, float(retry_delay_seconds or 0)))
                ).isoformat()
            )
            conn.execute(
                """
                UPDATE enterprise_chat_memory_outbox
                SET status = ?, available_at = ?, claim_token = NULL,
                    claimed_by = NULL, claimed_at = NULL,
                    side_effect_started_at = NULL,
                    side_effect_state_version = NULL,
                    completed_at = CASE WHEN ? = 'dead_letter' THEN ? ELSE NULL END,
                    last_error_code = ?, last_error_message = ?,
                    last_error_at = ?, updated_at = ?
                WHERE event_id = ? AND status = 'running' AND claim_token = ?
                """,
                (
                    status,
                    available,
                    status,
                    now,
                    error_code,
                    error_message,
                    now,
                    now,
                    event_id,
                    claim_token,
                ),
            )
            conn.execute(
                """
                UPDATE enterprise_chat_memory_groups
                SET last_error_code = ?, last_error_message = ?,
                    last_error_at = ?, updated_at = ?
                WHERE user_id = ? AND project_id = ? AND state = 'deleting'
                """,
                (
                    error_code,
                    error_message,
                    now,
                    now,
                    state.event.user_id,
                    state.event.project_id,
                ),
            )
            row = conn.execute(
                "SELECT * FROM enterprise_chat_memory_outbox WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            assert row is not None
            return ChatMemoryOutboxEventRecord.from_row(row)

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
        """Schedule a same-generation purge final sweep after an unknown clear."""

        fingerprint = _validate_chat_memory_fingerprint(runtime_fingerprint)
        graph_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            fingerprint, runtime_graph_store_fingerprint
        )
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> ChatMemoryOutboxEventRecord:
            state = self._require_sqlite_chat_memory_running_claim(
                conn, event_id, claim_token
            )
            stale = self._resolve_sqlite_stale_chat_memory_execution(
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
            if not self._chat_memory_runtime_identity_matches(
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
            now_dt = datetime.now(timezone.utc)
            now = now_dt.isoformat()
            available = (
                now_dt + timedelta(seconds=max(0.0, float(retry_delay_seconds)))
            ).isoformat()
            conn.execute(
                """
                UPDATE enterprise_chat_memory_generations
                SET state = 'purge_pending', cleared_at = NULL, updated_at = ?,
                    last_error_code = ?, last_error_message = ?,
                    last_error_at = ?
                WHERE user_id = ? AND project_id = ? AND generation = ?
                """,
                (
                    now,
                    error_code,
                    error_message,
                    now,
                    event.user_id,
                    event.project_id,
                    event.generation,
                ),
            )
            conn.execute(
                """
                UPDATE enterprise_chat_memory_outbox
                SET status = 'retry_wait', available_at = ?,
                    claim_token = NULL, claimed_by = NULL, claimed_at = NULL,
                    side_effect_started_at = NULL,
                    side_effect_state_version = NULL, completed_at = NULL,
                    last_error_code = ?, last_error_message = ?,
                    last_error_at = ?, updated_at = ?
                WHERE event_id = ? AND status = 'running' AND claim_token = ?
                """,
                (
                    available,
                    error_code,
                    error_message,
                    now,
                    now,
                    event_id,
                    claim_token,
                ),
            )
            conn.execute(
                """
                UPDATE enterprise_chat_memory_groups
                SET last_error_code = ?, last_error_message = ?,
                    last_error_at = ?, updated_at = ?
                WHERE user_id = ? AND project_id = ? AND state = 'deleting'
                """,
                (
                    error_code,
                    error_message,
                    now,
                    now,
                    event.user_id,
                    event.project_id,
                ),
            )
            row = conn.execute(
                "SELECT * FROM enterprise_chat_memory_outbox WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            assert row is not None
            return ChatMemoryOutboxEventRecord.from_row(row)

        return await self._write(write)

    async def requeue_chat_memory_purge(
        self,
        event_id: str,
        runtime_fingerprint: str,
        *,
        runtime_graph_store_fingerprint: str | None = None,
        retry_delay_seconds: float = 5.0,
    ) -> ChatMemoryOutboxEventRecord:
        """Explicitly requeue the same dead-letter purge without generation churn."""

        fingerprint = _validate_chat_memory_fingerprint(runtime_fingerprint)
        graph_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            fingerprint, runtime_graph_store_fingerprint
        )
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> ChatMemoryOutboxEventRecord:
            state = self._get_sqlite_chat_memory_execution_state(conn, event_id)
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
            now_dt = datetime.now(timezone.utc)
            now = now_dt.isoformat()
            available = (
                now_dt + timedelta(seconds=max(0.0, float(retry_delay_seconds)))
            ).isoformat()
            conn.execute(
                """
                UPDATE enterprise_chat_memory_outbox
                SET status = 'retry_wait', available_at = ?,
                    claim_token = NULL, claimed_by = NULL, claimed_at = NULL,
                    side_effect_started_at = NULL,
                    side_effect_state_version = NULL, completed_at = NULL,
                    last_error_code = NULL, last_error_message = NULL,
                    last_error_at = NULL, updated_at = ?
                WHERE event_id = ? AND status = 'dead_letter'
                """,
                (available, now, event_id),
            )
            row = conn.execute(
                "SELECT * FROM enterprise_chat_memory_outbox WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            assert row is not None
            return ChatMemoryOutboxEventRecord.from_row(row)

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
        """Fence an uncertain generation and enqueue a fresh rebuild atomically."""

        fingerprint = _validate_chat_memory_fingerprint(runtime_fingerprint)
        graph_fingerprint = _resolve_chat_memory_graph_store_fingerprint(
            fingerprint, runtime_graph_store_fingerprint
        )
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> ChatMemoryOutboxEventRecord:
            state = self._require_sqlite_chat_memory_running_claim(
                conn, event_id, claim_token
            )
            stale = self._resolve_sqlite_stale_chat_memory_execution(
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
            self._validate_chat_memory_execution_fence(
                state, fingerprint, graph_fingerprint
            )
            now = utc_now_iso()
            conn.execute(
                """
                UPDATE enterprise_chat_memory_generations
                SET state = 'abandoned', updated_at = ?,
                    last_error_code = ?, last_error_message = ?,
                    last_error_at = ?
                WHERE user_id = ? AND project_id = ? AND generation = ?
                """,
                (
                    now,
                    error_code,
                    error_message,
                    now,
                    event.user_id,
                    event.project_id,
                    event.generation,
                ),
            )
            if state.group.active_generation == event.generation:
                conn.execute(
                    """
                    UPDATE enterprise_chat_memory_groups
                    SET active_generation = NULL,
                        active_config_fingerprint = NULL
                        , active_graph_store_fingerprint = NULL
                    WHERE user_id = ? AND project_id = ?
                    """,
                    (event.user_id, event.project_id),
                )
            rebuild = self._enqueue_sqlite_chat_memory_rebuild(
                conn,
                state.group,
                fingerprint,
                graph_fingerprint,
                actor_user_id=actor_user_id,
                actor_tenant_id=actor_tenant_id,
                target_session_id=event.target_session_id,
                target_message_id=event.target_message_id,
            )
            cursor = conn.execute(
                """
                UPDATE enterprise_chat_memory_outbox
                SET status = 'superseded', superseded_by_event_id = ?,
                    completed_at = ?, last_error_code = ?,
                    last_error_message = ?, last_error_at = ?, updated_at = ?
                WHERE event_id = ? AND status = 'running' AND claim_token = ?
                """,
                (
                    rebuild.event_id,
                    now,
                    error_code,
                    error_message,
                    now,
                    now,
                    event_id,
                    claim_token,
                ),
            )
            if cursor.rowcount != 1:
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
        cutoff = (
            datetime.now(timezone.utc)
            - timedelta(seconds=max(0.0, float(stale_after_seconds)))
        ).isoformat()
        limit = max(1, min(int(limit), 1000))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM enterprise_chat_memory_outbox
                WHERE status = 'running' AND claimed_at IS NOT NULL
                  AND claimed_at <= ?
                ORDER BY claimed_at, user_id, project_id, event_seq
                LIMIT ?
                """,
                (cutoff, limit),
            ).fetchall()
        return [ChatMemoryOutboxEventRecord.from_row(row) for row in rows]

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
        now = datetime.now(timezone.utc)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending,
                    SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) AS running,
                    SUM(CASE WHEN status = 'retry_wait' THEN 1 ELSE 0 END) AS retry_wait,
                    SUM(CASE WHEN status = 'dead_letter' THEN 1 ELSE 0 END) AS dead_letter,
                    MIN(CASE WHEN status IN (
                        'pending', 'running', 'retry_wait', 'dead_letter'
                    ) THEN available_at END) AS oldest_available_at
                FROM enterprise_chat_memory_outbox
                """
            ).fetchone()
        oldest = row["oldest_available_at"] if row is not None else None
        lag = 0.0
        if oldest:
            oldest_dt = datetime.fromisoformat(str(oldest))
            if oldest_dt.tzinfo is None:
                oldest_dt = oldest_dt.replace(tzinfo=timezone.utc)
            lag = max(0.0, (now - oldest_dt).total_seconds())
        return ChatMemoryOutboxStats(
            pending=int(row["pending"] or 0) if row is not None else 0,
            running=int(row["running"] or 0) if row is not None else 0,
            retry_wait=int(row["retry_wait"] or 0) if row is not None else 0,
            dead_letter=int(row["dead_letter"] or 0) if row is not None else 0,
            oldest_available_at=str(oldest) if oldest else None,
            oldest_lag_seconds=lag,
        )

    def _get_sqlite_chat_memory_execution_state(
        self, conn: sqlite3.Connection, event_id: str
    ) -> ChatMemoryExecutionState | None:
        event_row = conn.execute(
            "SELECT * FROM enterprise_chat_memory_outbox WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        if event_row is None:
            return None
        event = ChatMemoryOutboxEventRecord.from_row(event_row)
        group = self._get_sqlite_chat_memory_group(
            conn, event.user_id, event.project_id
        )
        generation_row = conn.execute(
            """
            SELECT * FROM enterprise_chat_memory_generations
            WHERE user_id = ? AND project_id = ? AND generation = ?
            """,
            (event.user_id, event.project_id, event.generation),
        ).fetchone()
        if group is None or generation_row is None:
            raise MetadataStoreError(
                f"Chat Memory execution inventory missing for event '{event_id}'"
            )
        return ChatMemoryExecutionState(
            group=group,
            event=event,
            generation=ChatMemoryGenerationRecord.from_row(generation_row),
        )

    def _require_sqlite_chat_memory_running_claim(
        self, conn: sqlite3.Connection, event_id: str, claim_token: str
    ) -> ChatMemoryExecutionState:
        state = self._get_sqlite_chat_memory_execution_state(conn, event_id)
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

    def _chat_memory_stale_execution_reason(
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
    def _chat_memory_runtime_identity_matches(
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

    def _resolve_sqlite_stale_chat_memory_execution(
        self,
        conn: sqlite3.Connection,
        state: ChatMemoryExecutionState,
        claim_token: str,
    ) -> ChatMemoryOutboxEventRecord | None:
        reason = self._chat_memory_stale_execution_reason(state)
        if reason is None:
            return None
        takeover = conn.execute(
            """
            SELECT event_id FROM enterprise_chat_memory_outbox
            WHERE user_id = ? AND project_id = ? AND event_seq > ?
              AND event_type IN ('rebuild', 'purge')
              AND status <> 'superseded'
            ORDER BY event_seq ASC LIMIT 1
            """,
            (state.event.user_id, state.event.project_id, state.event.event_seq),
        ).fetchone()
        superseded_by = takeover["event_id"] if takeover is not None else None
        now = utc_now_iso()
        cursor = conn.execute(
            """
            UPDATE enterprise_chat_memory_outbox
            SET status = 'superseded', superseded_by_event_id = ?,
                completed_at = ?, last_error_code = 'stale_execution_fence',
                last_error_message = ?, last_error_at = ?, updated_at = ?
            WHERE event_id = ? AND status = 'running' AND claim_token = ?
            """,
            (
                superseded_by,
                now,
                reason,
                now,
                now,
                state.event.event_id,
                claim_token,
            ),
        )
        if cursor.rowcount != 1:
            raise MetadataConflictError(
                "chat_memory_event",
                state.event.event_id,
                expected={"status": "running", "claim_token": claim_token},
                current={
                    "status": state.event.status,
                    "claim_token": state.event.claim_token,
                },
            )
        row = conn.execute(
            "SELECT * FROM enterprise_chat_memory_outbox WHERE event_id = ?",
            (state.event.event_id,),
        ).fetchone()
        assert row is not None
        return ChatMemoryOutboxEventRecord.from_row(row)

    def _retry_sqlite_chat_memory_runtime_mismatch(
        self,
        conn: sqlite3.Connection,
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
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat()
        available = (
            now_dt + timedelta(seconds=max(0.0, float(retry_delay_seconds)))
        ).isoformat()
        conn.execute(
            """
            UPDATE enterprise_chat_memory_outbox
            SET status = 'retry_wait', available_at = ?, claim_token = NULL,
                claimed_by = NULL, claimed_at = NULL,
                side_effect_started_at = NULL,
                side_effect_state_version = NULL,
                last_error_code = 'runtime_fingerprint_mismatch',
                last_error_message = ?, last_error_at = ?, updated_at = ?
            WHERE event_id = ? AND status = 'running' AND claim_token = ?
            """,
            (
                available,
                (
                    "expected runtime identity "
                    f"({state.event.config_fingerprint}, "
                    f"{state.event.graph_store_fingerprint}), got "
                    f"({runtime_fingerprint}, {runtime_graph_store_fingerprint})"
                ),
                now,
                now,
                state.event.event_id,
                claim_token,
            ),
        )
        row = conn.execute(
            "SELECT * FROM enterprise_chat_memory_outbox WHERE event_id = ?",
            (state.event.event_id,),
        ).fetchone()
        assert row is not None
        return ChatMemoryOutboxEventRecord.from_row(row)

    def _validate_chat_memory_execution_fence(
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

    def _validate_chat_memory_ingest_execution(
        self,
        state: ChatMemoryExecutionState,
        runtime_fingerprint: str,
        runtime_graph_store_fingerprint: str,
    ) -> None:
        self._validate_chat_memory_execution_fence(
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

    def _insert_sqlite_chat_memory_historical_mapping(
        self, conn: sqlite3.Connection, expected: ChatMemoryEpisodeRecord
    ) -> None:
        def same_payload(row: sqlite3.Row) -> bool:
            current = ChatMemoryEpisodeRecord.from_row(row)
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

        by_uuid = conn.execute(
            """
            SELECT * FROM enterprise_chat_memory_episodes
            WHERE episode_uuid = ?
            """,
            (expected.episode_uuid,),
        ).fetchone()
        by_identity = conn.execute(
            """
            SELECT * FROM enterprise_chat_memory_episodes
            WHERE user_id = ? AND project_id = ? AND generation = ?
              AND append_batch_id = ?
            """,
            (
                expected.user_id,
                expected.project_id,
                expected.generation,
                expected.append_batch_id,
            ),
        ).fetchone()
        for row in (by_uuid, by_identity):
            if row is not None and not same_payload(row):
                raise MetadataConflictError(
                    "chat_memory_episode_mapping",
                    expected.episode_uuid,
                    expected=expected.to_dict(),
                    current=ChatMemoryEpisodeRecord.from_row(row).to_dict(),
                )
        if by_uuid is not None or by_identity is not None:
            return
        conn.execute(
            """
            INSERT INTO enterprise_chat_memory_episodes (
                episode_uuid, session_id, project_id, user_id, first_seq,
                last_seq, created_at, event_id, generation, graph_group_id,
                append_batch_id, project_event_seq
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
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
            ),
        )

    def _sqlite_chat_memory_graph_store_fingerprints(
        self,
        conn: sqlite3.Connection,
        user_id: str,
        project_id: str,
    ) -> tuple[str, ...]:
        """Return every non-empty graph-store identity attached to a group."""

        rows = conn.execute(
            """
            SELECT graph_store_fingerprint FROM (
                SELECT active_graph_store_fingerprint AS graph_store_fingerprint
                FROM enterprise_chat_memory_groups
                WHERE user_id = ? AND project_id = ?
                UNION ALL
                SELECT desired_graph_store_fingerprint
                FROM enterprise_chat_memory_groups
                WHERE user_id = ? AND project_id = ?
                UNION ALL
                SELECT graph_store_fingerprint
                FROM enterprise_chat_memory_generations
                WHERE user_id = ? AND project_id = ?
                UNION ALL
                SELECT graph_store_fingerprint
                FROM enterprise_chat_memory_outbox
                WHERE user_id = ? AND project_id = ?
                UNION ALL
                SELECT generation.graph_store_fingerprint
                FROM enterprise_chat_memory_episodes AS mapping
                JOIN enterprise_chat_memory_generations AS generation
                  ON generation.user_id = mapping.user_id
                 AND generation.project_id = mapping.project_id
                 AND generation.generation = mapping.generation
                WHERE mapping.user_id = ? AND mapping.project_id = ?
                UNION ALL
                SELECT event.graph_store_fingerprint
                FROM enterprise_chat_memory_episodes AS mapping
                JOIN enterprise_chat_memory_outbox AS event
                  ON event.event_id = mapping.event_id
                WHERE mapping.user_id = ? AND mapping.project_id = ?
            )
            WHERE graph_store_fingerprint IS NOT NULL
              AND trim(graph_store_fingerprint) <> ''
            """,
            (
                user_id,
                project_id,
                user_id,
                project_id,
                user_id,
                project_id,
                user_id,
                project_id,
                user_id,
                project_id,
                user_id,
                project_id,
            ),
        ).fetchall()
        return tuple(
            sorted({str(row["graph_store_fingerprint"]) for row in rows})
        )

    def _assert_sqlite_chat_memory_graph_store_invariant(
        self,
        conn: sqlite3.Connection,
        group: ChatMemoryGroupRecord,
        required_graph_store_fingerprint: str,
    ) -> None:
        required = _validate_chat_memory_fingerprint(
            required_graph_store_fingerprint
        )
        observed = self._sqlite_chat_memory_graph_store_fingerprints(
            conn, group.user_id, group.project_id
        )
        if observed != (required,):
            raise _chat_memory_graph_store_migration_conflict(
                group.logical_group_id,
                required,
                observed,
            )

    def _get_sqlite_chat_memory_group(
        self, conn: sqlite3.Connection, user_id: str, project_id: str
    ) -> ChatMemoryGroupRecord | None:
        row = conn.execute(
            """
            SELECT * FROM enterprise_chat_memory_groups
            WHERE user_id = ? AND project_id = ?
            """,
            (user_id, project_id),
        ).fetchone()
        return ChatMemoryGroupRecord.from_row(row) if row is not None else None

    def _ensure_sqlite_chat_memory_group(
        self,
        conn: sqlite3.Connection,
        user_id: str,
        project_id: str,
        config_fingerprint: str,
        graph_store_fingerprint: str,
        *,
        generation_state: ChatMemoryGenerationState,
    ) -> tuple[ChatMemoryGroupRecord, bool]:
        group = self._get_sqlite_chat_memory_group(conn, user_id, project_id)
        if group is not None:
            return group, False
        now = utc_now_iso()
        logical_group_id = chat_memory_logical_group_id(user_id, project_id)
        conn.execute(
            """
            INSERT INTO enterprise_chat_memory_groups (
                user_id, project_id, logical_group_id, active_generation,
                desired_generation, next_event_seq, last_reference_time, state,
                state_version, active_config_fingerprint,
                desired_config_fingerprint, active_graph_store_fingerprint,
                desired_graph_store_fingerprint, active_rebuild_event_id,
                last_success_at, last_error_code, last_error_message,
                last_error_at, created_at, updated_at, deleted_at, record_version
            ) VALUES (?, ?, ?, NULL, 1, 1, NULL, ?, 1, NULL, ?, NULL, ?, NULL,
                      NULL, NULL, NULL, NULL, ?, ?, NULL, ?)
            """,
            (
                user_id,
                project_id,
                logical_group_id,
                "deleting" if generation_state == "purge_pending" else "rebuilding",
                config_fingerprint,
                graph_store_fingerprint,
                now,
                now,
                CHAT_MEMORY_RECORD_VERSION,
            ),
        )
        self._insert_sqlite_chat_memory_generation(
            conn,
            user_id=user_id,
            project_id=project_id,
            generation=1,
            config_fingerprint=config_fingerprint,
            graph_store_fingerprint=graph_store_fingerprint,
            state=generation_state,
            now=now,
        )
        group = self._get_sqlite_chat_memory_group(conn, user_id, project_id)
        assert group is not None
        return group, True

    def _insert_sqlite_chat_memory_generation(
        self,
        conn: sqlite3.Connection,
        *,
        user_id: str,
        project_id: str,
        generation: int,
        config_fingerprint: str,
        graph_store_fingerprint: str,
        state: ChatMemoryGenerationState,
        now: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO enterprise_chat_memory_generations (
                user_id, project_id, generation, graph_group_id,
                config_fingerprint, graph_store_fingerprint, state, snapshot_cutoff,
                replay_batch_count, replay_message_count, replay_byte_count,
                snapshot_digest, clear_attempt_no, clear_started_at, created_at,
                updated_at, activated_at, cleared_at, last_error_code,
                last_error_message, last_error_at, record_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, 0, NULL,
                      ?, ?, NULL, NULL, NULL, NULL, NULL, ?)
            """,
            (
                user_id,
                project_id,
                int(generation),
                chat_memory_graph_group_id(user_id, project_id, generation),
                config_fingerprint,
                graph_store_fingerprint,
                state,
                now,
                now,
                CHAT_MEMORY_RECORD_VERSION,
            ),
        )

    def _allocate_sqlite_chat_memory_event_seq(
        self,
        conn: sqlite3.Connection,
        user_id: str,
        project_id: str,
        *,
        allocate_reference_time: bool,
    ) -> tuple[int, str | None]:
        row = conn.execute(
            """
            SELECT next_event_seq, last_reference_time
            FROM enterprise_chat_memory_groups
            WHERE user_id = ? AND project_id = ?
            """,
            (user_id, project_id),
        ).fetchone()
        if row is None:
            raise MetadataRecordNotFoundError("Chat Memory group not found")
        event_seq = int(row["next_event_seq"])
        reference_time = (
            _next_sqlite_chat_memory_reference_time(row["last_reference_time"])
            if allocate_reference_time
            else None
        )
        now = utc_now_iso()
        conn.execute(
            """
            UPDATE enterprise_chat_memory_groups
            SET next_event_seq = ?,
                last_reference_time = CASE WHEN ? IS NULL
                    THEN last_reference_time ELSE ? END,
                updated_at = ?
            WHERE user_id = ? AND project_id = ?
            """,
            (
                event_seq + 1,
                reference_time,
                reference_time,
                now,
                user_id,
                project_id,
            ),
        )
        return event_seq, reference_time

    def _insert_sqlite_chat_memory_event(
        self, conn: sqlite3.Connection, event: ChatMemoryOutboxEventRecord
    ) -> None:
        conn.execute(
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
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
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
                event.available_at,
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
                event.created_at,
                event.updated_at,
                event.record_version,
            ),
        )

    def _enqueue_sqlite_chat_memory_rebuild(
        self,
        conn: sqlite3.Connection,
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
        self._assert_sqlite_chat_memory_graph_store_invariant(
            conn, group, graph_store_fingerprint
        )
        existing_generation = conn.execute(
            """
            SELECT 1 FROM enterprise_chat_memory_generations
            WHERE user_id = ? AND project_id = ? AND generation = ?
            """,
            (group.user_id, group.project_id, group.desired_generation),
        ).fetchone()
        is_new_group = group.next_event_seq == 1 and existing_generation is not None
        if is_new_group and group.active_generation is None:
            generation = group.desired_generation
        else:
            generation = group.desired_generation + 1
            now = utc_now_iso()
            conn.execute(
                """
                UPDATE enterprise_chat_memory_generations
                SET state = 'abandoned', updated_at = ?,
                    last_error_code = 'source_changed',
                    last_error_message = 'Superseded by a newer source snapshot',
                    last_error_at = ?
                WHERE user_id = ? AND project_id = ? AND state = 'building'
                """,
                (now, now, group.user_id, group.project_id),
            )
            self._insert_sqlite_chat_memory_generation(
                conn,
                user_id=group.user_id,
                project_id=group.project_id,
                generation=generation,
                config_fingerprint=config_fingerprint,
                graph_store_fingerprint=graph_store_fingerprint,
                state="building",
                now=now,
            )
            conn.execute(
                """
                UPDATE enterprise_chat_memory_groups
                SET desired_generation = ?, desired_config_fingerprint = ?,
                    desired_graph_store_fingerprint = ?,
                    state = 'rebuilding', state_version = state_version + 1,
                    active_rebuild_event_id = NULL, last_error_code = NULL,
                    last_error_message = NULL, last_error_at = NULL,
                    updated_at = ?, deleted_at = NULL
                WHERE user_id = ? AND project_id = ?
                """,
                (
                    generation,
                    config_fingerprint,
                    graph_store_fingerprint,
                    now,
                    group.user_id,
                    group.project_id,
                ),
            )

        event_seq, _ = self._allocate_sqlite_chat_memory_event_seq(
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
        now = utc_now_iso()
        conn.execute(
            """
            UPDATE enterprise_chat_memory_outbox
            SET status = 'superseded', superseded_by_event_id = ?,
                completed_at = ?, updated_at = ?
            WHERE user_id = ? AND project_id = ? AND event_seq < ?
              AND event_type IN ('ingest', 'rebuild')
              AND status IN ('pending', 'retry_wait', 'dead_letter')
            """,
            (
                event_id,
                now,
                now,
                group.user_id,
                group.project_id,
                event_seq,
            ),
        )
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
        self._insert_sqlite_chat_memory_event(conn, event)
        conn.execute(
            """
            UPDATE enterprise_chat_memory_groups
            SET state = 'rebuilding', desired_config_fingerprint = ?,
                desired_graph_store_fingerprint = ?,
                active_rebuild_event_id = ?, updated_at = ?
            WHERE user_id = ? AND project_id = ?
            """,
            (
                config_fingerprint,
                graph_store_fingerprint,
                event_id,
                now,
                group.user_id,
                group.project_id,
            ),
        )
        return event

    def _sqlite_chat_memory_purge_group_ids(
        self,
        conn: sqlite3.Connection,
        user_id: str,
        project_id: str,
    ) -> tuple[str, ...]:
        """Return the complete defensive graph-group universe for a purge."""

        rows = conn.execute(
            """
            SELECT graph_group_id FROM (
                SELECT graph_group_id
                FROM enterprise_chat_memory_generations
                WHERE user_id = ? AND project_id = ?
                UNION
                SELECT graph_group_id
                FROM enterprise_chat_memory_episodes
                WHERE user_id = ? AND project_id = ?
                UNION
                SELECT graph_group_id
                FROM enterprise_chat_memory_outbox
                WHERE user_id = ? AND project_id = ?
            )
            WHERE graph_group_id IS NOT NULL AND trim(graph_group_id) <> ''
            ORDER BY graph_group_id ASC
            """,
            (
                user_id,
                project_id,
                user_id,
                project_id,
                user_id,
                project_id,
            ),
        ).fetchall()
        return _normalize_chat_memory_group_ids(
            [
                *(str(row["graph_group_id"]) for row in rows),
                chat_memory_legacy_graph_group_id(user_id, project_id),
            ]
        )

    def _sqlite_chat_memory_rebuild_group_ids(
        self,
        conn: sqlite3.Connection,
        event: ChatMemoryOutboxEventRecord,
    ) -> tuple[str, ...]:
        """Return every old, orphan, legacy, and target rebuild group id."""

        return _normalize_chat_memory_group_ids(
            [
                *self._sqlite_chat_memory_purge_group_ids(
                    conn, event.user_id, event.project_id
                ),
                event.graph_group_id,
            ]
        )

    def _enqueue_sqlite_chat_memory_purge(
        self,
        conn: sqlite3.Connection,
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
        group = self._get_sqlite_chat_memory_group(conn, user_id, project_id)
        if group is not None and group.state == "deleted":
            return None
        existing = conn.execute(
            """
            SELECT * FROM enterprise_chat_memory_outbox
            WHERE user_id = ? AND project_id = ? AND event_type = 'purge'
              AND status IN ('pending', 'running', 'retry_wait')
            ORDER BY event_seq DESC LIMIT 1
            """,
            (user_id, project_id),
        ).fetchone()
        if existing is not None:
            return ChatMemoryOutboxEventRecord.from_row(existing)
        if group is None:
            group, _ = self._ensure_sqlite_chat_memory_group(
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
        generation_row = conn.execute(
            """
            SELECT graph_group_id FROM enterprise_chat_memory_generations
            WHERE user_id = ? AND project_id = ? AND generation = ?
            """,
            (user_id, project_id, group.desired_generation),
        ).fetchone()
        if generation_row is None:
            self._insert_sqlite_chat_memory_generation(
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
                now=utc_now_iso(),
            )
        now = utc_now_iso()
        conn.execute(
            """
            UPDATE enterprise_chat_memory_generations
            SET state = 'purge_pending', updated_at = ?, cleared_at = NULL
            WHERE user_id = ? AND project_id = ? AND state <> 'purged'
            """,
            (now, user_id, project_id),
        )
        conn.execute(
            """
            UPDATE enterprise_chat_memory_groups
            SET state = 'deleting', state_version = state_version + 1,
                desired_config_fingerprint = ?, active_rebuild_event_id = NULL,
                active_generation = NULL, active_config_fingerprint = NULL,
                active_graph_store_fingerprint = NULL,
                last_error_code = NULL, last_error_message = NULL,
                last_error_at = NULL, updated_at = ?, deleted_at = NULL
            WHERE user_id = ? AND project_id = ?
            """,
            (config_fingerprint, now, user_id, project_id),
        )
        event_seq, _ = self._allocate_sqlite_chat_memory_event_seq(
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
        conn.execute(
            """
            UPDATE enterprise_chat_memory_outbox
            SET status = 'superseded', superseded_by_event_id = ?,
                completed_at = ?, updated_at = ?
            WHERE user_id = ? AND project_id = ? AND event_seq < ?
              AND status IN ('pending', 'retry_wait', 'dead_letter')
            """,
            (event_id, now, now, user_id, project_id, event_seq),
        )
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
        self._insert_sqlite_chat_memory_event(conn, event)
        return event

    async def set_enterprise_system_setting(
        self, key: str, value: str, *, updated_by: str | None = None
    ) -> None:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> None:
            now = utc_now_iso()
            conn.execute(
                """
                INSERT INTO enterprise_system_settings (
                    key, value, updated_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_by = excluded.updated_by,
                    updated_at = excluded.updated_at
                """,
                (key, value, updated_by, now, now),
            )

        await self._write(write)

    async def get_enterprise_system_setting(
        self, key: str, default: str | None = None
    ) -> str | None:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM enterprise_system_settings WHERE key = ?", (key,)
            ).fetchone()
        return default if row is None else str(row["value"])

    async def get_kb_lifecycle(self, kb_id: str) -> KBLifecycleRecord | None:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM enterprise_kb_lifecycle WHERE kb_id = ?", (kb_id,)
            ).fetchone()
        return KBLifecycleRecord.from_row(row) if row is not None else None

    async def assert_kb_not_deleting(
        self, kb_id: str, expected_generation: str | None = None
    ) -> KBLifecycleRecord | None:
        """Return lifecycle state unless an in-progress hard delete owns the KB."""

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
        return await self._write(
            lambda conn: self._activate_kb_generation(
                conn,
                kb_id,
                generation,
                activated_at=activated_at or utc_now_iso(),
            )
        )

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
        return await self._write(
            lambda conn: self._assert_kb_generation(
                conn, kb_id, expected_generation
            )
        )

    async def assert_current_kb_generation(
        self, kb_id: str, expected_generation: str | None
    ) -> KBLifecycleRecord | None:
        return await self.assert_kb_generation(kb_id, expected_generation)

    @asynccontextmanager
    async def job_execution_guard(
        self, job_id: str, *, wait: bool = True
    ) -> AsyncIterator[bool]:
        """Own one durable job across processes for the complete context.

        The lock is a stable job-id hash in a sibling lock directory, combined
        with a process-local lock shared by store instances using the same DB.
        Standard crash-stop cleanup closes the file descriptor and releases the
        OS lock. This is intentionally session ownership, not a durable lease or
        run token. ``wait=False`` is a single non-blocking ownership attempt.
        The same asyncio task may re-enter the same job on this store safely.
        """

        _validate_job_execution_id(job_id)
        await self._ensure_initialized()
        task = asyncio.current_task()
        state = self._job_guard_state.get()
        state_token: Any | None = None
        if state is None or state.owner_task is not task:
            state = _SQLiteJobGuardTaskState(owner_task=task, depths={})
            state_token = self._job_guard_state.set(state)

        current_depth = state.depths.get(job_id, 0)
        if current_depth:
            state.depths[job_id] = current_depth + 1
            try:
                yield True
            finally:
                remaining = state.depths[job_id] - 1
                if remaining:
                    state.depths[job_id] = remaining
                else:
                    state.depths.pop(job_id, None)
                if state_token is not None:
                    self._job_guard_state.reset(state_token)
            return

        process_lock = _process_job_execution_lock(self.db_path, job_id)
        process_acquired = False
        file_lock: _KBOperationFileLock | None = None
        file_acquired = False
        try:
            process_acquired = await process_lock.acquire(wait=wait)
            if not process_acquired:
                yield False
                return
            file_lock = _KBOperationFileLock(
                self._job_execution_lock_path(job_id), shared=False
            )
            if wait:
                while not file_lock.try_acquire():
                    await asyncio.sleep(_JOB_EXECUTION_LOCK_POLL_SECONDS)
                file_acquired = True
            else:
                file_acquired = file_lock.try_acquire()
            if not file_acquired:
                yield False
                return
            state.depths[job_id] = 1
            try:
                yield True
            finally:
                state.depths.pop(job_id, None)
        finally:
            try:
                if file_acquired and file_lock is not None:
                    file_lock.release()
            finally:
                try:
                    if process_acquired:
                        release_task = asyncio.create_task(process_lock.release())
                        try:
                            await asyncio.shield(release_task)
                        except asyncio.CancelledError:
                            await asyncio.gather(release_task, return_exceptions=True)
                            raise
                finally:
                    if state_token is not None:
                        self._job_guard_state.reset(state_token)

    @asynccontextmanager
    async def kb_write_guard(
        self, kb_id: str, expected_generation: str | None
    ) -> AsyncIterator[KBLifecycleRecord | None]:
        """Hold the cross-process shared KB operation fence for a write.

        The same task may re-enter the same store/KB/generation without taking
        a second process or file lock. This is required when request middleware,
        a service operation, and a worker executor all enforce the same fence.
        """

        await self._ensure_initialized()
        # Fast rejection avoids waiting behind a deletion guard whose lifecycle
        # transition is already committed. The guarded assertion below closes
        # the race between this preflight and lock acquisition.
        await self.assert_kb_generation(kb_id, expected_generation)
        task = asyncio.current_task()
        guard_key = (kb_id, expected_generation)
        state = self._kb_write_guard_state.get()
        if (
            state is not None
            and state.owner_task is not task
            and state.depths.get(guard_key, 0)
        ):
            # Child tasks inherit ContextVar values. Borrow the active parent
            # fence while the parent awaits this task; the owner will not
            # release its process/file lock until every borrower exits.
            state.depths[guard_key] += 1
            idle_event = state.idle_events[guard_key]
            idle_event.clear()
            try:
                yield await self.assert_kb_generation(kb_id, expected_generation)
            finally:
                remaining = state.depths[guard_key] - 1
                if remaining:
                    state.depths[guard_key] = remaining
                else:
                    state.depths.pop(guard_key, None)
                    idle_event.set()
            return

        state_token: Any | None = None
        if state is None or state.owner_task is not task:
            state = _SQLiteKBWriteGuardTaskState(
                owner_task=task,
                depths={},
                idle_events={},
            )
            state_token = self._kb_write_guard_state.set(state)

        current_depth = state.depths.get(guard_key, 0)
        if current_depth:
            state.depths[guard_key] = current_depth + 1
            try:
                yield await self.assert_kb_generation(kb_id, expected_generation)
            finally:
                remaining = state.depths[guard_key] - 1
                if remaining:
                    state.depths[guard_key] = remaining
                else:
                    state.depths.pop(guard_key, None)
                    state.idle_events[guard_key].set()
                if state_token is not None:
                    self._kb_write_guard_state.reset(state_token)
            return

        process_lock = _process_kb_operation_lock(self.db_path, kb_id)
        try:
            async with process_lock.shared():
                file_lock = _KBOperationFileLock(
                    self._kb_operation_lock_path(kb_id), shared=True
                )
                await _acquire_kb_operation_file_lock(file_lock)
                try:
                    current = await self.assert_kb_generation(
                        kb_id, expected_generation
                    )
                    idle_event = asyncio.Event()
                    state.idle_events[guard_key] = idle_event
                    state.depths[guard_key] = 1
                    try:
                        yield current
                    finally:
                        remaining = state.depths[guard_key] - 1
                        if remaining:
                            state.depths[guard_key] = remaining
                            await _wait_for_kb_guard_borrowers(idle_event)
                        else:
                            state.depths.pop(guard_key, None)
                            idle_event.set()
                        state.idle_events.pop(guard_key, None)
                finally:
                    file_lock.release()
        finally:
            if state_token is not None:
                self._kb_write_guard_state.reset(state_token)

    @asynccontextmanager
    async def kb_exclusive_operation_guard(
        self,
        kb_id: str,
    ) -> AsyncIterator[None]:
        """Hold only the cross-process exclusive KB operation fence.

        This context deliberately does not mutate lifecycle state. Hard-delete
        orchestration can therefore re-read the catalog while exclusion is held
        and call :meth:`begin_kb_deletion` only after that catalog precondition
        still matches.
        """

        await self._ensure_initialized()
        process_lock = _process_kb_operation_lock(self.db_path, kb_id)
        async with process_lock.exclusive():
            file_lock = _KBOperationFileLock(
                self._kb_operation_lock_path(kb_id), shared=False
            )
            await _acquire_kb_operation_file_lock(file_lock)
            try:
                yield
            finally:
                file_lock.release()

    @asynccontextmanager
    async def kb_deletion_guard(
        self,
        kb_id: str,
        expected_generation: str | None = None,
        delete_job_id: str | None = None,
    ) -> AsyncIterator[KBLifecycleRecord | None]:
        """Compatibility wrapper around the split delete lock/state APIs.

        Calling this with only ``kb_id`` is the new pure-exclusive form. Older
        callers may still pass generation/job; that form begins deletion after
        acquiring the same fence and retains the historical durable binding.
        """

        if (expected_generation is None) != (delete_job_id is None):
            raise MetadataStoreError(
                "KB deletion guard requires both generation and delete_job_id"
            )
        async with self.kb_exclusive_operation_guard(kb_id):
            if expected_generation is None or delete_job_id is None:
                yield None
                return
            yield await self.begin_kb_deletion(
                kb_id,
                expected_generation,
                delete_job_id,
            )

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
        return await self._write(
            lambda conn: self._begin_kb_deletion(
                conn,
                kb_id,
                generation,
                delete_job_id,
            )
        )

    async def complete_kb_deletion(
        self, kb_id: str, generation: str, delete_job_id: str
    ) -> KBLifecycleRecord:
        """Atomically finish ``deleting`` -> ``deleted``; exact retries are safe."""

        _validate_kb_lifecycle_identity(kb_id, generation)
        _validate_delete_job_id(delete_job_id)
        await self._ensure_initialized()
        return await self._write(
            lambda conn: self._complete_kb_deletion(
                conn, kb_id, generation, delete_job_id
            )
        )

    async def create_enterprise_api_key(
        self,
        record: EnterpriseAPIKeyRecord,
        *,
        expected_kb_generations: dict[str, str] | None = None,
    ) -> EnterpriseAPIKeyRecord:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> EnterpriseAPIKeyRecord:
            if not isinstance(record.scopes, dict):
                raise MetadataStoreError("Service API key scopes must be an object")
            kb_roles = record.scopes.get("kb_roles", {})
            if not isinstance(kb_roles, dict):
                raise MetadataStoreError("Service API key kb_roles must be an object")
            for kb_id in kb_roles:
                if not isinstance(kb_id, str) or not kb_id:
                    raise MetadataStoreError("Service API key KB id must be non-empty")
            for kb_id in sorted(kb_roles):
                self._assert_kb_generation(
                    conn,
                    kb_id,
                    (expected_kb_generations or {}).get(kb_id),
                )
            conn.execute(
                """
                INSERT INTO enterprise_api_keys (
                    id, name, key_hash, key_preview, status, created_by, tenant_id,
                    scopes_json, metadata_json, created_at, updated_at, last_used_at,
                    revoked_at, revoked_by, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.name,
                    record.key_hash,
                    record.key_preview,
                    record.status,
                    record.created_by,
                    record.tenant_id,
                    _dumps_json(record.scopes),
                    _dumps_json(record.metadata),
                    record.created_at,
                    record.updated_at,
                    record.last_used_at,
                    record.revoked_at,
                    record.revoked_by,
                    record.expires_at,
                ),
            )
            row = conn.execute(
                "SELECT * FROM enterprise_api_keys WHERE id = ?", (record.id,)
            ).fetchone()
            assert row is not None
            return EnterpriseAPIKeyRecord.from_row(row)

        return await self._write(write)

    async def get_enterprise_api_key_by_hash(
        self, key_hash: str
    ) -> EnterpriseAPIKeyRecord | None:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM enterprise_api_keys WHERE key_hash = ?",
                (key_hash,),
            ).fetchone()
        return EnterpriseAPIKeyRecord.from_row(row) if row is not None else None

    async def get_enterprise_api_key_by_id(
        self, key_id: str
    ) -> EnterpriseAPIKeyRecord | None:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM enterprise_api_keys WHERE id = ?",
                (key_id,),
            ).fetchone()
        return EnterpriseAPIKeyRecord.from_row(row) if row is not None else None

    async def list_enterprise_api_keys(self) -> list[EnterpriseAPIKeyRecord]:
        await self._ensure_initialized()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM enterprise_api_keys
                ORDER BY created_at DESC, id DESC
                """
            ).fetchall()
        return [EnterpriseAPIKeyRecord.from_row(row) for row in rows]

    async def revoke_enterprise_api_key(
        self,
        key_id: str,
        *,
        revoked_by: str | None = None,
        revoked_at: str | None = None,
    ) -> EnterpriseAPIKeyRecord | None:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> EnterpriseAPIKeyRecord | None:
            now = revoked_at or utc_now_iso()
            cursor = conn.execute(
                """
                UPDATE enterprise_api_keys
                SET status = ?, revoked_at = ?, revoked_by = ?, updated_at = ?
                WHERE id = ?
                """,
                ("revoked", now, revoked_by, now, key_id),
            )
            if not cursor.rowcount:
                return None
            row = conn.execute(
                "SELECT * FROM enterprise_api_keys WHERE id = ?", (key_id,)
            ).fetchone()
            assert row is not None
            return EnterpriseAPIKeyRecord.from_row(row)

        return await self._write(write)

    async def mark_enterprise_api_key_used(
        self, key_id: str, *, last_used_at: str | None = None
    ) -> EnterpriseAPIKeyRecord | None:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> EnterpriseAPIKeyRecord | None:
            now = last_used_at or utc_now_iso()
            cursor = conn.execute(
                """
                UPDATE enterprise_api_keys
                SET last_used_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (now, now, key_id),
            )
            if not cursor.rowcount:
                return None
            row = conn.execute(
                "SELECT * FROM enterprise_api_keys WHERE id = ?", (key_id,)
            ).fetchone()
            assert row is not None
            return EnterpriseAPIKeyRecord.from_row(row)

        return await self._write(write)

    async def create_enterprise_invitation(
        self, record: EnterpriseInvitationRecord
    ) -> EnterpriseInvitationRecord:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> EnterpriseInvitationRecord:
            conn.execute(
                """
                INSERT INTO enterprise_invitations (
                    id, token_hash, token_preview, status, created_by, expires_at,
                    used_by, used_at, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.token_hash,
                    record.token_preview,
                    record.status,
                    record.created_by,
                    record.expires_at,
                    record.used_by,
                    record.used_at,
                    _dumps_json(record.metadata),
                    record.created_at,
                    record.updated_at,
                ),
            )
            row = conn.execute(
                "SELECT * FROM enterprise_invitations WHERE id = ?", (record.id,)
            ).fetchone()
            assert row is not None
            return EnterpriseInvitationRecord.from_row(row)

        return await self._write(write)

    async def get_enterprise_invitation_by_token_hash(
        self, token_hash: str
    ) -> EnterpriseInvitationRecord | None:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM enterprise_invitations WHERE token_hash = ?",
                (token_hash,),
            ).fetchone()
        return EnterpriseInvitationRecord.from_row(row) if row is not None else None

    async def list_enterprise_invitations(self) -> list[EnterpriseInvitationRecord]:
        await self._ensure_initialized()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM enterprise_invitations
                ORDER BY created_at DESC, id DESC
                """
            ).fetchall()
        return [EnterpriseInvitationRecord.from_row(row) for row in rows]

    async def consume_enterprise_invitation(
        self, token_hash: str, *, used_by: str | None, used_at: str | None = None
    ) -> EnterpriseInvitationRecord | None:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> EnterpriseInvitationRecord | None:
            now = used_at or utc_now_iso()
            cursor = conn.execute(
                """
                UPDATE enterprise_invitations
                SET status = 'used', used_by = ?, used_at = ?, updated_at = ?
                WHERE token_hash = ? AND status = 'active'
                """,
                (used_by, now, now, token_hash),
            )
            if not cursor.rowcount:
                return None
            row = conn.execute(
                "SELECT * FROM enterprise_invitations WHERE token_hash = ?",
                (token_hash,),
            ).fetchone()
            assert row is not None
            return EnterpriseInvitationRecord.from_row(row)

        return await self._write(write)

    async def revoke_enterprise_invitation(
        self, invitation_id: str, *, revoked_at: str | None = None
    ) -> EnterpriseInvitationRecord | None:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> EnterpriseInvitationRecord | None:
            now = revoked_at or utc_now_iso()
            conn.execute(
                """
                UPDATE enterprise_invitations
                SET status = 'revoked', updated_at = ?
                WHERE id = ? AND status = 'active'
                """,
                (now, invitation_id),
            )
            row = conn.execute(
                "SELECT * FROM enterprise_invitations WHERE id = ?", (invitation_id,)
            ).fetchone()
            return EnterpriseInvitationRecord.from_row(row) if row is not None else None

        return await self._write(write)

    # ------------------------------------------------------------------
    # Multi-account person identity store (SQLite)
    #
    # Simple reads/writes plus aggregate atomic methods. The atomic methods
    # perform state reads, CAS, multi-table writes and audit-row inserts
    # inside a single ``_write()`` transaction. They never call the public
    # ``AuditService.append`` (which opens its own write transaction) nor
    # nest another ``_write()``. See docs/多账号身份关联与切换执行文档.md 7.2.
    # ------------------------------------------------------------------

    async def get_person_by_id(
        self, person_id: str
    ) -> EnterprisePersonRecord | None:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM enterprise_persons WHERE id = ?", (person_id,)
            ).fetchone()
        return EnterprisePersonRecord.from_row(row) if row is not None else None

    async def list_person_account_links(
        self, person_id: str, *, only_active: bool = False
    ) -> list[EnterprisePersonAccountLinkRecord]:
        await self._ensure_initialized()
        if only_active:
            sql = (
                "SELECT * FROM enterprise_person_account_links "
                "WHERE person_id = ? AND status = 'active' "
                "ORDER BY bound_at ASC, id ASC"
            )
        else:
            sql = (
                "SELECT * FROM enterprise_person_account_links "
                "WHERE person_id = ? ORDER BY status ASC, id ASC"
            )
        with self._connect() as conn:
            rows = conn.execute(sql, (person_id,)).fetchall()
        return [EnterprisePersonAccountLinkRecord.from_row(row) for row in rows]

    async def get_person_account_link(
        self, person_id: str, account_id: str
    ) -> EnterprisePersonAccountLinkRecord | None:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM enterprise_person_account_links "
                "WHERE person_id = ? AND account_id = ?",
                (person_id, account_id),
            ).fetchone()
        return (
            EnterprisePersonAccountLinkRecord.from_row(row)
            if row is not None
            else None
        )

    async def get_active_person_link_for_account(
        self, account_id: str
    ) -> EnterprisePersonAccountLinkRecord | None:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM enterprise_person_account_links "
                "WHERE account_id = ? AND status = 'active'",
                (account_id,),
            ).fetchone()
        return (
            EnterprisePersonAccountLinkRecord.from_row(row)
            if row is not None
            else None
        )

    async def get_person_credential(
        self, person_id: str
    ) -> EnterprisePersonCredentialRecord | None:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM enterprise_person_credentials "
                "WHERE person_id = ? AND credential_type = 'password' "
                "AND status = 'active'",
                (person_id,),
            ).fetchone()
        return (
            EnterprisePersonCredentialRecord.from_row(row)
            if row is not None
            else None
        )

    async def record_person_credential_failure_atomic(
        self,
        credential_id: str,
        *,
        max_attempts: int,
        lockout_seconds: float,
        now: str | None = None,
    ) -> EnterprisePersonCredentialRecord:
        """Atomically count a failed person-password attempt.

        The increment happens in SQL (``failed_count = failed_count + 1``) so
        concurrent failures never lose counts. When the new count reaches
        ``max_attempts`` the credential is locked for ``lockout_seconds`` and a
        ``person_login_locked`` audit row is written; every failure writes a
        ``person_login_failed`` row. Failure events carry no actor (doc 7.3:
        failures must not fabricate an actor).
        """

        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> EnterprisePersonCredentialRecord:
            timestamp = now or utc_now_iso()
            cursor = conn.execute(
                "UPDATE enterprise_person_credentials "
                "SET failed_count = failed_count + 1, updated_at = ? "
                "WHERE id = ?",
                (timestamp, credential_id),
            )
            if not cursor.rowcount:
                raise MetadataRecordNotFoundError(
                    f"Person credential '{credential_id}' not found"
                )
            row = conn.execute(
                "SELECT * FROM enterprise_person_credentials WHERE id = ?",
                (credential_id,),
            ).fetchone()
            assert row is not None
            current = EnterprisePersonCredentialRecord.from_row(row)
            locked = False
            if max_attempts > 0 and current.failed_count >= max_attempts:
                locked_until = (
                    datetime.fromisoformat(timestamp)
                    + timedelta(seconds=float(lockout_seconds))
                ).isoformat()
                conn.execute(
                    "UPDATE enterprise_person_credentials "
                    "SET locked_until = ? WHERE id = ?",
                    (locked_until, credential_id),
                )
                current = replace(current, locked_until=locked_until)
                locked = True
            _insert_audit_event(
                conn,
                AuditEventRecord(
                    id=_new_audit_id(),
                    event_type="person_login_failed",
                    actor_user_id=None,
                    actor_tenant_id=None,
                    target_type="person_credential",
                    target_id=credential_id,
                    metadata={
                        "person_id": current.person_id,
                        "failed_count": current.failed_count,
                    },
                    created_at=timestamp,
                ),
            )
            if locked:
                _insert_audit_event(
                    conn,
                    AuditEventRecord(
                        id=_new_audit_id(),
                        event_type="person_login_locked",
                        actor_user_id=None,
                        actor_tenant_id=None,
                        target_type="person_credential",
                        target_id=credential_id,
                        metadata={
                            "person_id": current.person_id,
                            "failed_count": current.failed_count,
                            "locked_until": current.locked_until,
                        },
                        created_at=timestamp,
                    ),
                )
            return current

        return await self._write(write)

    async def reset_person_credential_failures_atomic(
        self,
        credential_id: str,
        *,
        now: str | None = None,
    ) -> None:
        """Clear failure counters after a successful person authentication."""

        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> None:
            timestamp = now or utc_now_iso()
            conn.execute(
                "UPDATE enterprise_person_credentials "
                "SET failed_count = 0, locked_until = NULL, last_used_at = ?, "
                "updated_at = ? WHERE id = ?",
                (timestamp, timestamp, credential_id),
            )

        await self._write(write)

    async def get_person_login_session(
        self, session_id: str
    ) -> EnterprisePersonLoginSessionRecord | None:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM enterprise_person_login_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return (
            EnterprisePersonLoginSessionRecord.from_row(row)
            if row is not None
            else None
        )

    async def list_person_login_sessions(
        self, person_id: str, *, only_active: bool = False
    ) -> list[EnterprisePersonLoginSessionRecord]:
        await self._ensure_initialized()
        if only_active:
            sql = (
                "SELECT * FROM enterprise_person_login_sessions "
                "WHERE person_id = ? AND status = 'active' "
                "ORDER BY created_at ASC, id ASC"
            )
        else:
            sql = (
                "SELECT * FROM enterprise_person_login_sessions "
                "WHERE person_id = ? ORDER BY created_at DESC, id DESC"
            )
        with self._connect() as conn:
            rows = conn.execute(sql, (person_id,)).fetchall()
        return [EnterprisePersonLoginSessionRecord.from_row(row) for row in rows]

    async def get_person_enrollment_grant_by_token_hash(
        self, token_hash: str
    ) -> EnterprisePersonEnrollmentGrantRecord | None:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM enterprise_person_enrollment_grants "
                "WHERE token_hash = ?",
                (token_hash,),
            ).fetchone()
        return (
            EnterprisePersonEnrollmentGrantRecord.from_row(row)
            if row is not None
            else None
        )

    async def get_person_enrollment_grant(
        self, grant_id: str
    ) -> EnterprisePersonEnrollmentGrantRecord | None:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM enterprise_person_enrollment_grants WHERE id = ?",
                (grant_id,),
            ).fetchone()
        return (
            EnterprisePersonEnrollmentGrantRecord.from_row(row)
            if row is not None
            else None
        )

    def _sqlite_revoke_person_sessions_locked(
        self,
        conn: sqlite3.Connection,
        person_id: str | None,
        *,
        account_id: str | None,
        actor_user_id: str | None,
        now: str,
        audit_event_type: str,
    ) -> int:
        """Revoke matching active person sessions and emit one audit row each.

        Exactly one of ``person_id``/``account_id`` scopes the match. Returns
        the number of sessions revoked. Called inside a _write() transaction.
        """

        if person_id is not None:
            match_clause = "person_id = ? AND status = 'active'"
            params: tuple[Any, ...] = (person_id,)
        else:
            assert account_id is not None
            match_clause = "active_account_id = ? AND status = 'active'"
            params = (account_id,)
        rows = conn.execute(
            f"SELECT id, person_id FROM enterprise_person_login_sessions "
            f"WHERE {match_clause}",
            params,
        ).fetchall()
        for srow in rows:
            sid = str(srow["id"])
            sperson = srow["person_id"]
            conn.execute(
                "UPDATE enterprise_person_login_sessions "
                "SET status = 'revoked', revoked_at = ?, last_seen_at = ? "
                "WHERE id = ? AND status = 'active'",
                (now, now, sid),
            )
            _insert_audit_event(
                conn,
                AuditEventRecord(
                    id=_new_audit_id(),
                    event_type=audit_event_type,
                    actor_user_id=actor_user_id,
                    actor_tenant_id=None,
                    target_type="person_login_session",
                    target_id=sid,
                    metadata={
                        "person_id": sperson,
                        "account_id": account_id,
                    },
                    created_at=now,
                ),
            )
        return len(rows)

    async def create_person_enrollment_grant_atomic(
        self,
        grant: EnterprisePersonEnrollmentGrantRecord,
        *,
        actor_user_id: str | None = None,
    ) -> EnterprisePersonEnrollmentGrantRecord:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> EnterprisePersonEnrollmentGrantRecord:
            try:
                conn.execute(
                    """
                    INSERT INTO enterprise_person_enrollment_grants (
                        id, account_id, token_hash, status, created_by,
                        consumed_by_person, expires_at, created_at, updated_at,
                        consumed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        grant.id,
                        grant.account_id,
                        grant.token_hash,
                        grant.status,
                        grant.created_by,
                        grant.consumed_by_person,
                        grant.expires_at,
                        grant.created_at,
                        grant.updated_at,
                        grant.consumed_at,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise MetadataConflictError(
                    "person_enrollment_grant_active",
                    grant.account_id,
                    expected={"status": "no active grant"},
                    current={"error": str(exc)},
                ) from exc
            _insert_audit_event(
                conn,
                AuditEventRecord(
                    id=_new_audit_id(),
                    event_type="person_enrollment_grant_created",
                    actor_user_id=actor_user_id,
                    actor_tenant_id=None,
                    target_type="person_enrollment_grant",
                    target_id=grant.id,
                    metadata={"account_id": grant.account_id},
                    created_at=grant.created_at,
                ),
            )
            row = conn.execute(
                "SELECT * FROM enterprise_person_enrollment_grants WHERE id = ?",
                (grant.id,),
            ).fetchone()
            assert row is not None
            return EnterprisePersonEnrollmentGrantRecord.from_row(row)

        return await self._write(write)

    async def revoke_person_enrollment_grant_atomic(
        self,
        grant_id: str,
        *,
        actor_user_id: str | None = None,
        revoked_at: str | None = None,
        reason: str | None = None,
    ) -> EnterprisePersonEnrollmentGrantRecord | None:
        await self._ensure_initialized()

        def write(
            conn: sqlite3.Connection,
        ) -> EnterprisePersonEnrollmentGrantRecord | None:
            row = conn.execute(
                "SELECT * FROM enterprise_person_enrollment_grants WHERE id = ?",
                (grant_id,),
            ).fetchone()
            if row is None:
                return None
            current = EnterprisePersonEnrollmentGrantRecord.from_row(row)
            if current.status == "active":
                now = revoked_at or utc_now_iso()
                conn.execute(
                    """
                    UPDATE enterprise_person_enrollment_grants
                    SET status = 'revoked', updated_at = ?
                    WHERE id = ? AND status = 'active'
                    """,
                    (now, grant_id),
                )
                current = replace(
                    current, status="revoked", updated_at=now
                )
                _insert_audit_event(
                    conn,
                    AuditEventRecord(
                        id=_new_audit_id(),
                        event_type="person_enrollment_grant_revoked",
                        actor_user_id=actor_user_id,
                        actor_tenant_id=None,
                        target_type="person_enrollment_grant",
                        target_id=grant_id,
                        metadata={
                            "account_id": current.account_id,
                            "reason": reason,
                        },
                        created_at=now,
                    ),
                )
            return current

        return await self._write(write)

    async def consume_enrollment_grant_atomic(
        self,
        token_hash: str,
        *,
        person_id: str,
        actor_user_id: str | None = None,
        consumed_at: str | None = None,
    ) -> EnterprisePersonEnrollmentGrantRecord:
        """Atomically consume (active->consumed) an enrollment grant.

        Raises ``MetadataRecordNotFoundError`` if absent and
        ``MetadataConflictError`` if not active or already expired. Used as a
        standalone CAS primitive; ``enroll_person_atomic`` inlines the same
        consumption so the full enroll stays in one transaction.
        """

        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> EnterprisePersonEnrollmentGrantRecord:
            row = conn.execute(
                "SELECT * FROM enterprise_person_enrollment_grants "
                "WHERE token_hash = ?",
                (token_hash,),
            ).fetchone()
            if row is None:
                raise MetadataRecordNotFoundError(
                    "Enrollment grant for token hash not found"
                )
            current = EnterprisePersonEnrollmentGrantRecord.from_row(row)
            now = consumed_at or utc_now_iso()
            if current.status != "active":
                raise MetadataConflictError(
                    "person_enrollment_grant",
                    current.id,
                    expected={"status": "active"},
                    current={"status": current.status},
                )
            if current.expires_at <= now:
                conn.execute(
                    """
                    UPDATE enterprise_person_enrollment_grants
                    SET status = 'expired', updated_at = ?
                    WHERE id = ? AND status = 'active'
                    """,
                    (now, current.id),
                )
                raise MetadataConflictError(
                    "person_enrollment_grant",
                    current.id,
                    expected={"status": "active", "not_expired": True},
                    current={"status": "expired"},
                )
            conn.execute(
                """
                UPDATE enterprise_person_enrollment_grants
                SET status = 'consumed', consumed_by_person = ?,
                    consumed_at = ?, updated_at = ?
                WHERE id = ? AND status = 'active'
                """,
                (person_id, now, now, current.id),
            )
            row = conn.execute(
                "SELECT * FROM enterprise_person_enrollment_grants WHERE id = ?",
                (current.id,),
            ).fetchone()
            assert row is not None
            return EnterprisePersonEnrollmentGrantRecord.from_row(row)

        return await self._write(write)

    async def enroll_person_atomic(
        self,
        *,
        grant_token_hash: str,
        person: EnterprisePersonRecord,
        credential: EnterprisePersonCredentialRecord,
        link: EnterprisePersonAccountLinkRecord,
        session: EnterprisePersonLoginSessionRecord,
        actor_user_id: str | None = None,
    ) -> tuple[
        EnterprisePersonRecord,
        EnterprisePersonCredentialRecord,
        EnterprisePersonAccountLinkRecord,
        EnterprisePersonLoginSessionRecord,
    ]:
        await self._ensure_initialized()

        def write(
            conn: sqlite3.Connection,
        ) -> tuple[
            EnterprisePersonRecord,
            EnterprisePersonCredentialRecord,
            EnterprisePersonAccountLinkRecord,
            EnterprisePersonLoginSessionRecord,
        ]:
            now = utc_now_iso()
            grow = conn.execute(
                "SELECT * FROM enterprise_person_enrollment_grants "
                "WHERE token_hash = ?",
                (grant_token_hash,),
            ).fetchone()
            if grow is None:
                raise MetadataRecordNotFoundError(
                    "Enrollment grant for token hash not found"
                )
            grant_rec = EnterprisePersonEnrollmentGrantRecord.from_row(grow)
            if grant_rec.status != "active":
                raise MetadataConflictError(
                    "person_enrollment_grant",
                    grant_rec.id,
                    expected={"status": "active"},
                    current={"status": grant_rec.status},
                )
            if grant_rec.expires_at <= now:
                raise MetadataConflictError(
                    "person_enrollment_grant",
                    grant_rec.id,
                    expected={"status": "active", "not_expired": True},
                    current={"status": "expired"},
                )
            # Proactive active-link conflict check (clear error in the common
            # case); the partial unique index is the final concurrency arbiter.
            clash = conn.execute(
                "SELECT id FROM enterprise_person_account_links "
                "WHERE account_id = ? AND status = 'active'",
                (link.account_id,),
            ).fetchone()
            if clash is not None:
                raise MetadataConflictError(
                    "person_account_link_active",
                    link.account_id,
                    expected={"status": "no active link"},
                    current={"status": "already_linked"},
                )
            # Consume grant first.
            conn.execute(
                """
                UPDATE enterprise_person_enrollment_grants
                SET status = 'consumed', consumed_by_person = ?,
                    consumed_at = ?, updated_at = ?
                WHERE id = ? AND status = 'active'
                """,
                (person.id, now, now, grant_rec.id),
            )
            conn.execute(
                """
                INSERT INTO enterprise_persons (
                    id, status, auth_epoch, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    person.id,
                    person.status,
                    person.auth_epoch,
                    _dumps_json(person.metadata),
                    person.created_at,
                    person.updated_at,
                ),
            )
            conn.execute(
                """
                INSERT INTO enterprise_person_credentials (
                    id, person_id, credential_type, algorithm, password_hash,
                    status, failed_count, locked_until, last_used_at,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    credential.id,
                    credential.person_id,
                    credential.credential_type,
                    credential.algorithm,
                    credential.password_hash,
                    credential.status,
                    credential.failed_count,
                    credential.locked_until,
                    credential.last_used_at,
                    credential.created_at,
                    credential.updated_at,
                ),
            )
            try:
                conn.execute(
                    """
                    INSERT INTO enterprise_person_account_links (
                        id, person_id, account_id, status, bound_by, bound_at,
                        confirmed_by_person_at, revoked_by, revoked_at, reason,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        link.id,
                        link.person_id,
                        link.account_id,
                        link.status,
                        link.bound_by,
                        link.bound_at,
                        link.confirmed_by_person_at,
                        link.revoked_by,
                        link.revoked_at,
                        link.reason,
                        link.created_at,
                        link.updated_at,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise MetadataConflictError(
                    "person_account_link_active",
                    link.account_id,
                    expected={"status": "no active link"},
                    current={"error": str(exc)},
                ) from exc
            # Snapshot the active account's token_version so v2 account-access
            # validation can detect a password reset (doc 4.5/6.4).
            acct_row = conn.execute(
                "SELECT token_version FROM enterprise_users WHERE id = ?",
                (link.account_id,),
            ).fetchone()
            account_token_version = (
                int(acct_row["token_version"]) if acct_row is not None else 0
            )
            conn.execute(
                """
                INSERT INTO enterprise_person_login_sessions (
                    id, person_id, active_account_id, status, person_epoch,
                    session_epoch, absolute_expires_at, created_at,
                    last_seen_at, revoked_at, account_token_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    session.person_id,
                    session.active_account_id,
                    session.status,
                    session.person_epoch,
                    session.session_epoch,
                    session.absolute_expires_at,
                    session.created_at,
                    session.last_seen_at,
                    session.revoked_at,
                    account_token_version,
                ),
            )
            _insert_audit_event(
                conn,
                AuditEventRecord(
                    id=_new_audit_id(),
                    event_type="person_enrolled",
                    actor_user_id=actor_user_id,
                    actor_tenant_id=None,
                    target_type="person",
                    target_id=person.id,
                    metadata={
                        "person_id": person.id,
                        "account_id": link.account_id,
                        "grant_id": grant_rec.id,
                        "session_id": session.id,
                    },
                    created_at=now,
                ),
            )
            prow = conn.execute(
                "SELECT * FROM enterprise_persons WHERE id = ?", (person.id,)
            ).fetchone()
            crow = conn.execute(
                "SELECT * FROM enterprise_person_credentials WHERE id = ?",
                (credential.id,),
            ).fetchone()
            lrow = conn.execute(
                "SELECT * FROM enterprise_person_account_links WHERE id = ?",
                (link.id,),
            ).fetchone()
            srow = conn.execute(
                "SELECT * FROM enterprise_person_login_sessions WHERE id = ?",
                (session.id,),
            ).fetchone()
            assert prow is not None and crow is not None
            assert lrow is not None and srow is not None
            return (
                EnterprisePersonRecord.from_row(prow),
                EnterprisePersonCredentialRecord.from_row(crow),
                EnterprisePersonAccountLinkRecord.from_row(lrow),
                EnterprisePersonLoginSessionRecord.from_row(srow),
            )

        return await self._write(write)

    async def create_person_session_atomic(
        self,
        session: EnterprisePersonLoginSessionRecord,
        *,
        expected_person_epoch: int,
        actor_user_id: str | None = None,
    ) -> EnterprisePersonLoginSessionRecord:
        await self._ensure_initialized()

        def write(
            conn: sqlite3.Connection,
        ) -> EnterprisePersonLoginSessionRecord:
            now = session.created_at
            prow = conn.execute(
                "SELECT * FROM enterprise_persons WHERE id = ?",
                (session.person_id,),
            ).fetchone()
            if prow is None:
                raise MetadataRecordNotFoundError(
                    f"Person '{session.person_id}' not found"
                )
            current_person = EnterprisePersonRecord.from_row(prow)
            if current_person.status != "active":
                raise MetadataConflictError(
                    "person",
                    session.person_id,
                    expected={"status": "active"},
                    current={"status": current_person.status},
                )
            if current_person.auth_epoch != expected_person_epoch:
                raise MetadataConflictError(
                    "person",
                    session.person_id,
                    expected={"auth_epoch": expected_person_epoch},
                    current={"auth_epoch": current_person.auth_epoch},
                )
            # Snapshot the active account's token_version (doc 4.5/6.4).
            account_token_version = 0
            if session.active_account_id:
                acct_row = conn.execute(
                    "SELECT token_version FROM enterprise_users WHERE id = ?",
                    (session.active_account_id,),
                ).fetchone()
                if acct_row is not None:
                    account_token_version = int(acct_row["token_version"])
            conn.execute(
                """
                INSERT INTO enterprise_person_login_sessions (
                    id, person_id, active_account_id, status, person_epoch,
                    session_epoch, absolute_expires_at, created_at,
                    last_seen_at, revoked_at, account_token_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    session.person_id,
                    session.active_account_id,
                    session.status,
                    session.person_epoch,
                    session.session_epoch,
                    session.absolute_expires_at,
                    session.created_at,
                    session.last_seen_at,
                    session.revoked_at,
                    account_token_version,
                ),
            )
            _insert_audit_event(
                conn,
                AuditEventRecord(
                    id=_new_audit_id(),
                    event_type="person_login_succeeded",
                    actor_user_id=actor_user_id,
                    actor_tenant_id=None,
                    target_type="person_login_session",
                    target_id=session.id,
                    metadata={
                        "person_id": session.person_id,
                        "account_id": session.active_account_id,
                    },
                    created_at=now,
                ),
            )
            srow = conn.execute(
                "SELECT * FROM enterprise_person_login_sessions WHERE id = ?",
                (session.id,),
            ).fetchone()
            assert srow is not None
            return EnterprisePersonLoginSessionRecord.from_row(srow)

        return await self._write(write)

    async def switch_person_session_atomic(
        self,
        *,
        session_id: str,
        expected_session_epoch: int,
        target_account_id: str,
        actor_user_id: str | None = None,
        switched_at: str | None = None,
    ) -> EnterprisePersonLoginSessionRecord:
        await self._ensure_initialized()

        def write(
            conn: sqlite3.Connection,
        ) -> EnterprisePersonLoginSessionRecord:
            now = switched_at or utc_now_iso()
            srow = conn.execute(
                "SELECT * FROM enterprise_person_login_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if srow is None:
                raise MetadataRecordNotFoundError(
                    f"Person login session '{session_id}' not found"
                )
            current = EnterprisePersonLoginSessionRecord.from_row(srow)
            if current.status != "active":
                raise MetadataConflictError(
                    "person_login_session",
                    session_id,
                    expected={"status": "active"},
                    current={"status": current.status},
                )
            if current.session_epoch != expected_session_epoch:
                raise MetadataConflictError(
                    "person_login_session",
                    session_id,
                    expected={"session_epoch": expected_session_epoch},
                    current={"session_epoch": current.session_epoch},
                )
            if current.absolute_expires_at <= now:
                raise MetadataConflictError(
                    "person_login_session",
                    session_id,
                    expected={"not_expired": True},
                    current={"status": "expired"},
                )
            link_row = conn.execute(
                "SELECT * FROM enterprise_person_account_links "
                "WHERE person_id = ? AND account_id = ? AND status = 'active'",
                (current.person_id, target_account_id),
            ).fetchone()
            if link_row is None:
                raise MetadataRecordNotFoundError(
                    "Target account is not an active person link"
                )
            source_account = current.active_account_id
            # Snapshot the target account's token_version on switch (doc 4.5).
            target_acct_row = conn.execute(
                "SELECT token_version FROM enterprise_users WHERE id = ?",
                (target_account_id,),
            ).fetchone()
            target_token_version = (
                int(target_acct_row["token_version"])
                if target_acct_row is not None
                else 0
            )
            conn.execute(
                """
                UPDATE enterprise_person_login_sessions
                SET active_account_id = ?, session_epoch = ?, last_seen_at = ?,
                    account_token_version = ?
                WHERE id = ?
                """,
                (
                    target_account_id,
                    current.session_epoch + 1,
                    now,
                    target_token_version,
                    session_id,
                ),
            )
            _insert_audit_event(
                conn,
                AuditEventRecord(
                    id=_new_audit_id(),
                    event_type="person_account_switched",
                    actor_user_id=actor_user_id,
                    actor_tenant_id=None,
                    target_type="person_login_session",
                    target_id=session_id,
                    metadata={
                        "person_id": current.person_id,
                        "source_account_id": source_account,
                        "target_account_id": target_account_id,
                    },
                    created_at=now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM enterprise_person_login_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            assert row is not None
            return EnterprisePersonLoginSessionRecord.from_row(row)

        return await self._write(write)

    async def rotate_person_credential_atomic(
        self,
        *,
        person_id: str,
        new_credential: EnterprisePersonCredentialRecord,
        actor_user_id: str | None = None,
    ) -> tuple[EnterprisePersonRecord, EnterprisePersonCredentialRecord]:
        await self._ensure_initialized()

        def write(
            conn: sqlite3.Connection,
        ) -> tuple[EnterprisePersonRecord, EnterprisePersonCredentialRecord]:
            now = new_credential.updated_at
            prow = conn.execute(
                "SELECT * FROM enterprise_persons WHERE id = ?", (person_id,)
            ).fetchone()
            if prow is None:
                raise MetadataRecordNotFoundError(
                    f"Person '{person_id}' not found"
                )
            current_person = EnterprisePersonRecord.from_row(prow)
            if current_person.status != "active":
                raise MetadataConflictError(
                    "person",
                    person_id,
                    expected={"status": "active"},
                    current={"status": current_person.status},
                )
            # Per doc 4.2 the UNIQUE(person_id, credential_type) constraint
            # means rotation UPDATES the existing active row in place (new
            # bcrypt hash, reset failure counters) rather than inserting a new
            # row and revoking the old one.
            existing_cred_row = conn.execute(
                "SELECT * FROM enterprise_person_credentials "
                "WHERE person_id = ? AND credential_type = 'password' "
                "AND status = 'active'",
                (person_id,),
            ).fetchone()
            if existing_cred_row is None:
                raise MetadataRecordNotFoundError(
                    f"Active password credential for person '{person_id}' not found"
                )
            existing_cred = EnterprisePersonCredentialRecord.from_row(
                existing_cred_row
            )
            conn.execute(
                """
                UPDATE enterprise_person_credentials
                SET algorithm = ?, password_hash = ?, failed_count = 0,
                    locked_until = NULL, last_used_at = NULL, updated_at = ?
                WHERE id = ?
                """,
                (
                    new_credential.algorithm,
                    new_credential.password_hash,
                    now,
                    existing_cred.id,
                ),
            )
            new_epoch = current_person.auth_epoch + 1
            conn.execute(
                "UPDATE enterprise_persons SET auth_epoch = ?, updated_at = ? "
                "WHERE id = ?",
                (new_epoch, now, person_id),
            )
            self._sqlite_revoke_person_sessions_locked(
                conn,
                person_id,
                account_id=None,
                actor_user_id=actor_user_id,
                now=now,
                audit_event_type="person_session_revoked_by_credential_rotation",
            )
            _insert_audit_event(
                conn,
                AuditEventRecord(
                    id=_new_audit_id(),
                    event_type="person_credential_rotated",
                    actor_user_id=actor_user_id,
                    actor_tenant_id=None,
                    target_type="person",
                    target_id=person_id,
                    metadata={
                        "person_id": person_id,
                        "credential_id": existing_cred.id,
                        "auth_epoch": new_epoch,
                    },
                    created_at=now,
                ),
            )
            prow = conn.execute(
                "SELECT * FROM enterprise_persons WHERE id = ?", (person_id,)
            ).fetchone()
            crow = conn.execute(
                "SELECT * FROM enterprise_person_credentials WHERE id = ?",
                (existing_cred.id,),
            ).fetchone()
            assert prow is not None and crow is not None
            return (
                EnterprisePersonRecord.from_row(prow),
                EnterprisePersonCredentialRecord.from_row(crow),
            )

        return await self._write(write)

    async def disable_person_atomic(
        self,
        *,
        person_id: str,
        actor_user_id: str | None = None,
        reason: str | None = None,
        disabled_at: str | None = None,
    ) -> EnterprisePersonRecord:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> EnterprisePersonRecord:
            now = disabled_at or utc_now_iso()
            prow = conn.execute(
                "SELECT * FROM enterprise_persons WHERE id = ?", (person_id,)
            ).fetchone()
            if prow is None:
                raise MetadataRecordNotFoundError(
                    f"Person '{person_id}' not found"
                )
            current_person = EnterprisePersonRecord.from_row(prow)
            new_epoch = current_person.auth_epoch + 1
            conn.execute(
                "UPDATE enterprise_persons SET status = 'disabled', "
                "auth_epoch = ?, updated_at = ? WHERE id = ?",
                (new_epoch, now, person_id),
            )
            self._sqlite_revoke_person_sessions_locked(
                conn,
                person_id,
                account_id=None,
                actor_user_id=actor_user_id,
                now=now,
                audit_event_type="person_session_revoked_by_person_disable",
            )
            _insert_audit_event(
                conn,
                AuditEventRecord(
                    id=_new_audit_id(),
                    event_type="person_disabled",
                    actor_user_id=actor_user_id,
                    actor_tenant_id=None,
                    target_type="person",
                    target_id=person_id,
                    metadata={
                        "person_id": person_id,
                        "reason": reason,
                        "auth_epoch": new_epoch,
                    },
                    created_at=now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM enterprise_persons WHERE id = ?", (person_id,)
            ).fetchone()
            assert row is not None
            return EnterprisePersonRecord.from_row(row)

        return await self._write(write)

    async def enable_person_atomic(
        self,
        *,
        person_id: str,
        actor_user_id: str | None = None,
        enabled_at: str | None = None,
    ) -> EnterprisePersonRecord:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> EnterprisePersonRecord:
            now = enabled_at or utc_now_iso()
            prow = conn.execute(
                "SELECT * FROM enterprise_persons WHERE id = ?", (person_id,)
            ).fetchone()
            if prow is None:
                raise MetadataRecordNotFoundError(
                    f"Person '{person_id}' not found"
                )
            conn.execute(
                "UPDATE enterprise_persons SET status = 'active', "
                "updated_at = ? WHERE id = ?",
                (now, person_id),
            )
            _insert_audit_event(
                conn,
                AuditEventRecord(
                    id=_new_audit_id(),
                    event_type="person_enabled",
                    actor_user_id=actor_user_id,
                    actor_tenant_id=None,
                    target_type="person",
                    target_id=person_id,
                    metadata={"person_id": person_id},
                    created_at=now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM enterprise_persons WHERE id = ?", (person_id,)
            ).fetchone()
            assert row is not None
            return EnterprisePersonRecord.from_row(row)

        return await self._write(write)

    async def propose_person_account_link_atomic(
        self,
        link: EnterprisePersonAccountLinkRecord,
        *,
        actor_user_id: str | None = None,
    ) -> EnterprisePersonAccountLinkRecord:
        await self._ensure_initialized()

        def write(
            conn: sqlite3.Connection,
        ) -> EnterprisePersonAccountLinkRecord:
            existing_row = conn.execute(
                "SELECT * FROM enterprise_person_account_links "
                "WHERE person_id = ? AND account_id = ?",
                (link.person_id, link.account_id),
            ).fetchone()
            now = link.updated_at
            if existing_row is not None:
                existing = EnterprisePersonAccountLinkRecord.from_row(existing_row)
                if existing.status == "pending":
                    return existing
                if existing.status == "active":
                    raise MetadataConflictError(
                        "person_account_link",
                        f"{link.person_id}:{link.account_id}",
                        expected={"status": "not active"},
                        current={"status": "active"},
                    )
                # Revoked -> re-propose as pending.
                conn.execute(
                    """
                    UPDATE enterprise_person_account_links
                    SET status = 'pending', bound_by = ?, bound_at = ?,
                        confirmed_by_person_at = ?, revoked_by = ?,
                        revoked_at = ?, reason = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        link.bound_by,
                        link.bound_at,
                        None,
                        None,
                        None,
                        link.reason,
                        now,
                        existing.id,
                    ),
                )
                link_id = existing.id
            else:
                conn.execute(
                    """
                    INSERT INTO enterprise_person_account_links (
                        id, person_id, account_id, status, bound_by, bound_at,
                        confirmed_by_person_at, revoked_by, revoked_at, reason,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        link.id,
                        link.person_id,
                        link.account_id,
                        "pending",
                        link.bound_by,
                        link.bound_at,
                        link.confirmed_by_person_at,
                        link.revoked_by,
                        link.revoked_at,
                        link.reason,
                        link.created_at,
                        link.updated_at,
                    ),
                )
                link_id = link.id
            _insert_audit_event(
                conn,
                AuditEventRecord(
                    id=_new_audit_id(),
                    event_type="person_account_link_proposed",
                    actor_user_id=actor_user_id,
                    actor_tenant_id=None,
                    target_type="person_account_link",
                    target_id=link_id,
                    metadata={
                        "person_id": link.person_id,
                        "account_id": link.account_id,
                    },
                    created_at=now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM enterprise_person_account_links WHERE id = ?",
                (link_id,),
            ).fetchone()
            assert row is not None
            return EnterprisePersonAccountLinkRecord.from_row(row)

        return await self._write(write)

    async def confirm_person_account_link_atomic(
        self,
        *,
        person_id: str,
        account_id: str,
        actor_user_id: str | None = None,
        confirmed_at: str | None = None,
    ) -> tuple[EnterprisePersonRecord, EnterprisePersonAccountLinkRecord]:
        await self._ensure_initialized()

        def write(
            conn: sqlite3.Connection,
        ) -> tuple[EnterprisePersonRecord, EnterprisePersonAccountLinkRecord]:
            now = confirmed_at or utc_now_iso()
            prow = conn.execute(
                "SELECT * FROM enterprise_persons WHERE id = ?", (person_id,)
            ).fetchone()
            if prow is None:
                raise MetadataRecordNotFoundError(
                    f"Person '{person_id}' not found"
                )
            current_person = EnterprisePersonRecord.from_row(prow)
            if current_person.status != "active":
                raise MetadataConflictError(
                    "person",
                    person_id,
                    expected={"status": "active"},
                    current={"status": current_person.status},
                )
            clash = conn.execute(
                "SELECT id FROM enterprise_person_account_links "
                "WHERE account_id = ? AND status = 'active'",
                (account_id,),
            ).fetchone()
            if clash is not None:
                raise MetadataConflictError(
                    "person_account_link_active",
                    account_id,
                    expected={"status": "no active link"},
                    current={"status": "already_linked"},
                )
            lrow = conn.execute(
                "SELECT * FROM enterprise_person_account_links "
                "WHERE person_id = ? AND account_id = ?",
                (person_id, account_id),
            ).fetchone()
            if lrow is None:
                raise MetadataRecordNotFoundError(
                    "Pending person-account link not found"
                )
            current_link = EnterprisePersonAccountLinkRecord.from_row(lrow)
            if current_link.status != "pending":
                raise MetadataConflictError(
                    "person_account_link",
                    f"{person_id}:{account_id}",
                    expected={"status": "pending"},
                    current={"status": current_link.status},
                )
            try:
                conn.execute(
                    """
                    UPDATE enterprise_person_account_links
                    SET status = 'active', confirmed_by_person_at = ?,
                        updated_at = ?
                    WHERE id = ? AND status = 'pending'
                    """,
                    (now, now, current_link.id),
                )
            except sqlite3.IntegrityError as exc:
                raise MetadataConflictError(
                    "person_account_link_active",
                    account_id,
                    expected={"status": "no active link"},
                    current={"error": str(exc)},
                ) from exc
            new_epoch = current_person.auth_epoch + 1
            conn.execute(
                "UPDATE enterprise_persons SET auth_epoch = ?, updated_at = ? "
                "WHERE id = ?",
                (new_epoch, now, person_id),
            )
            self._sqlite_revoke_person_sessions_locked(
                conn,
                person_id,
                account_id=None,
                actor_user_id=actor_user_id,
                now=now,
                audit_event_type="person_session_revoked_by_link_activation",
            )
            _insert_audit_event(
                conn,
                AuditEventRecord(
                    id=_new_audit_id(),
                    event_type="person_account_link_confirmed",
                    actor_user_id=actor_user_id,
                    actor_tenant_id=None,
                    target_type="person_account_link",
                    target_id=current_link.id,
                    metadata={
                        "person_id": person_id,
                        "account_id": account_id,
                        "auth_epoch": new_epoch,
                    },
                    created_at=now,
                ),
            )
            prow = conn.execute(
                "SELECT * FROM enterprise_persons WHERE id = ?", (person_id,)
            ).fetchone()
            nrow = conn.execute(
                "SELECT * FROM enterprise_person_account_links WHERE id = ?",
                (current_link.id,),
            ).fetchone()
            assert prow is not None and nrow is not None
            return (
                EnterprisePersonRecord.from_row(prow),
                EnterprisePersonAccountLinkRecord.from_row(nrow),
            )

        return await self._write(write)

    async def revoke_person_account_link_atomic(
        self,
        *,
        person_id: str,
        account_id: str,
        actor_user_id: str | None = None,
        revoked_at: str | None = None,
        reason: str | None = None,
    ) -> tuple[EnterprisePersonAccountLinkRecord, int]:
        await self._ensure_initialized()

        def write(
            conn: sqlite3.Connection,
        ) -> tuple[EnterprisePersonAccountLinkRecord, int]:
            now = revoked_at or utc_now_iso()
            lrow = conn.execute(
                "SELECT * FROM enterprise_person_account_links "
                "WHERE person_id = ? AND account_id = ?",
                (person_id, account_id),
            ).fetchone()
            if lrow is None:
                raise MetadataRecordNotFoundError(
                    "Person-account link not found"
                )
            current_link = EnterprisePersonAccountLinkRecord.from_row(lrow)
            if current_link.status != "active":
                revoked_sessions = 0
            else:
                revoked_sessions = self._sqlite_revoke_person_sessions_locked(
                    conn,
                    None,
                    account_id=account_id,
                    actor_user_id=actor_user_id,
                    now=now,
                    audit_event_type="person_session_revoked_by_account_change",
                )
                # The account no longer belongs to this person: person-KB
                # shares involving it (either side) lose their basis.
                self._revoke_person_kb_shares_locked(
                    conn,
                    either_side_account_id=account_id,
                    actor_user_id=actor_user_id,
                    now=now,
                    audit_event_type="person_kb_share_revoked_by_account_change",
                    reason="person_link_revoked",
                )
                conn.execute(
                    """
                    UPDATE enterprise_person_account_links
                    SET status = 'revoked', revoked_by = ?, revoked_at = ?,
                        reason = ?, updated_at = ?
                    WHERE id = ? AND status = 'active'
                    """,
                    (actor_user_id, now, reason, now, current_link.id),
                )
            _insert_audit_event(
                conn,
                AuditEventRecord(
                    id=_new_audit_id(),
                    event_type="person_account_unbound",
                    actor_user_id=actor_user_id,
                    actor_tenant_id=None,
                    target_type="person_account_link",
                    target_id=current_link.id,
                    metadata={
                        "person_id": person_id,
                        "account_id": account_id,
                        "reason": reason,
                        "revoked_sessions": revoked_sessions,
                    },
                    created_at=now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM enterprise_person_account_links WHERE id = ?",
                (current_link.id,),
            ).fetchone()
            assert row is not None
            return EnterprisePersonAccountLinkRecord.from_row(row), revoked_sessions

        return await self._write(write)

    async def revoke_person_session_atomic(
        self,
        session_id: str,
        *,
        actor_user_id: str | None = None,
        revoked_at: str | None = None,
    ) -> EnterprisePersonLoginSessionRecord | None:
        await self._ensure_initialized()

        def write(
            conn: sqlite3.Connection,
        ) -> EnterprisePersonLoginSessionRecord | None:
            now = revoked_at or utc_now_iso()
            cursor = conn.execute(
                "UPDATE enterprise_person_login_sessions "
                "SET status = 'revoked', revoked_at = ?, last_seen_at = ? "
                "WHERE id = ? AND status = 'active'",
                (now, now, session_id),
            )
            if not cursor.rowcount:
                row = conn.execute(
                    "SELECT * FROM enterprise_person_login_sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
                return (
                    EnterprisePersonLoginSessionRecord.from_row(row)
                    if row is not None
                    else None
                )
            _insert_audit_event(
                conn,
                AuditEventRecord(
                    id=_new_audit_id(),
                    event_type="person_session_logout",
                    actor_user_id=actor_user_id,
                    actor_tenant_id=None,
                    target_type="person_login_session",
                    target_id=session_id,
                    metadata={"session_id": session_id},
                    created_at=now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM enterprise_person_login_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            assert row is not None
            return EnterprisePersonLoginSessionRecord.from_row(row)

        return await self._write(write)

    async def revoke_all_person_sessions_atomic(
        self,
        person_id: str,
        *,
        actor_user_id: str | None = None,
        revoked_at: str | None = None,
    ) -> tuple[EnterprisePersonRecord, int]:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> tuple[EnterprisePersonRecord, int]:
            now = revoked_at or utc_now_iso()
            prow = conn.execute(
                "SELECT * FROM enterprise_persons WHERE id = ?", (person_id,)
            ).fetchone()
            if prow is None:
                raise MetadataRecordNotFoundError(
                    f"Person '{person_id}' not found"
                )
            current_person = EnterprisePersonRecord.from_row(prow)
            new_epoch = current_person.auth_epoch + 1
            conn.execute(
                "UPDATE enterprise_persons SET auth_epoch = ?, updated_at = ? "
                "WHERE id = ?",
                (new_epoch, now, person_id),
            )
            revoked = self._sqlite_revoke_person_sessions_locked(
                conn,
                person_id,
                account_id=None,
                actor_user_id=actor_user_id,
                now=now,
                audit_event_type="person_session_revoked_by_logout_all",
            )
            _insert_audit_event(
                conn,
                AuditEventRecord(
                    id=_new_audit_id(),
                    event_type="person_sessions_logout_all",
                    actor_user_id=actor_user_id,
                    actor_tenant_id=None,
                    target_type="person",
                    target_id=person_id,
                    metadata={
                        "person_id": person_id,
                        "revoked_sessions": revoked,
                        "auth_epoch": new_epoch,
                    },
                    created_at=now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM enterprise_persons WHERE id = ?", (person_id,)
            ).fetchone()
            assert row is not None
            return EnterprisePersonRecord.from_row(row), revoked

        return await self._write(write)

    # ------------------------------------------------------------------
    # Person KB shares: one natural person exposing a personal KB to their
    # OWN account in another department. Zero-copy: the share materializes a
    # direct kb_acl row for the target account (same transaction) and flags
    # the target department for tenant-admin oversight. While a share is
    # active it owns the (kb_id, target_account_id) ACL row.
    # ------------------------------------------------------------------

    def _revoke_person_kb_shares_locked(
        self,
        conn: sqlite3.Connection,
        *,
        kb_id: str | None = None,
        target_account_id: str | None = None,
        either_side_account_id: str | None = None,
        target_only_account_id: str | None = None,
        target_tenant_id: str | None = None,
        actor_user_id: str | None,
        now: str,
        audit_event_type: str,
        reason: str | None,
    ) -> int:
        """Revoke matching ACTIVE shares plus their materialized kb_acl rows.

        Exactly one filter shape: (kb_id + target_account_id) for a single
        share, ``either_side_account_id`` for unlink/account-delete (the
        account may be owner or target), ``target_only_account_id`` for a
        canonical-tenant move (department-scoped target grants die; shares the
        account OWNS are not departmental and survive), or ``target_tenant_id``
        when the department itself is deleted. One audit row per revoked
        share, same transaction.
        """

        if kb_id is not None and target_account_id is not None:
            where = "kb_id = ? AND target_account_id = ? AND status = 'active'"
            params: tuple[Any, ...] = (kb_id, target_account_id)
        elif either_side_account_id is not None:
            where = (
                "(owner_account_id = ? OR target_account_id = ?) "
                "AND status = 'active'"
            )
            params = (either_side_account_id, either_side_account_id)
        elif target_tenant_id is not None:
            where = "target_tenant_id = ? AND status = 'active'"
            params = (target_tenant_id,)
        else:
            assert target_only_account_id is not None
            where = "target_account_id = ? AND status = 'active'"
            params = (target_only_account_id,)
        rows = conn.execute(
            f"SELECT * FROM enterprise_person_kb_shares WHERE {where}", params
        ).fetchall()
        for row in rows:
            share = EnterprisePersonKBShareRecord.from_row(row)
            conn.execute(
                """
                UPDATE enterprise_person_kb_shares
                SET status = 'revoked', revoked_by = ?, revoked_at = ?,
                    reason = COALESCE(?, reason), updated_at = ?
                WHERE id = ? AND status = 'active'
                """,
                (actor_user_id, now, reason, now, share.id),
            )
            conn.execute(
                "DELETE FROM enterprise_kb_acl WHERE kb_id = ? AND user_id = ?",
                (share.kb_id, share.target_account_id),
            )
            _insert_audit_event(
                conn,
                AuditEventRecord(
                    id=_new_audit_id(),
                    event_type=audit_event_type,
                    actor_user_id=actor_user_id,
                    actor_tenant_id=None,
                    target_type="person_kb_share",
                    target_id=share.id,
                    metadata={
                        "kb_id": share.kb_id,
                        "person_id": share.person_id,
                        "owner_account_id": share.owner_account_id,
                        "target_account_id": share.target_account_id,
                        "target_tenant_id": share.target_tenant_id,
                        "reason": reason,
                    },
                    created_at=now,
                ),
            )
        return len(rows)

    async def create_person_kb_share_atomic(
        self,
        share: EnterprisePersonKBShareRecord,
        *,
        expected_generation: str | None = None,
        actor_user_id: str | None = None,
    ) -> EnterprisePersonKBShareRecord:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> EnterprisePersonKBShareRecord:
            self._assert_kb_generation(conn, share.kb_id, expected_generation)
            conn.execute(
                """
                INSERT INTO enterprise_person_kb_shares (
                    id, kb_id, person_id, owner_account_id, target_account_id,
                    target_tenant_id, role, status, created_by, revoked_by,
                    reason, created_at, updated_at, revoked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, NULL, ?, ?, ?, NULL)
                ON CONFLICT(kb_id, target_account_id) DO UPDATE SET
                    person_id = excluded.person_id,
                    owner_account_id = excluded.owner_account_id,
                    target_tenant_id = excluded.target_tenant_id,
                    role = excluded.role,
                    status = 'active',
                    created_by = excluded.created_by,
                    revoked_by = NULL,
                    reason = excluded.reason,
                    updated_at = excluded.updated_at,
                    revoked_at = NULL
                """,
                (
                    share.id,
                    share.kb_id,
                    share.person_id,
                    share.owner_account_id,
                    share.target_account_id,
                    share.target_tenant_id,
                    share.role,
                    share.created_by,
                    share.reason,
                    share.created_at,
                    share.updated_at,
                ),
            )
            # Materialize the grant as a direct ACL so resolve_kb_access needs
            # no extra lookups for the target account.
            conn.execute(
                """
                INSERT INTO enterprise_kb_acl (
                    kb_id, user_id, role, granted_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(kb_id, user_id) DO UPDATE SET
                    role = excluded.role,
                    granted_by = excluded.granted_by,
                    updated_at = excluded.updated_at
                """,
                (
                    share.kb_id,
                    share.target_account_id,
                    share.role,
                    share.created_by,
                    share.created_at,
                    share.updated_at,
                ),
            )
            _insert_audit_event(
                conn,
                AuditEventRecord(
                    id=_new_audit_id(),
                    event_type="person_kb_share_created",
                    actor_user_id=actor_user_id,
                    actor_tenant_id=None,
                    target_type="person_kb_share",
                    target_id=share.id,
                    metadata={
                        "kb_id": share.kb_id,
                        "person_id": share.person_id,
                        "owner_account_id": share.owner_account_id,
                        "target_account_id": share.target_account_id,
                        "target_tenant_id": share.target_tenant_id,
                        "role": share.role,
                    },
                    created_at=share.created_at,
                ),
            )
            row = conn.execute(
                "SELECT * FROM enterprise_person_kb_shares "
                "WHERE kb_id = ? AND target_account_id = ?",
                (share.kb_id, share.target_account_id),
            ).fetchone()
            assert row is not None
            return EnterprisePersonKBShareRecord.from_row(row)

        return await self._write(write)

    async def revoke_person_kb_share_atomic(
        self,
        kb_id: str,
        target_account_id: str,
        *,
        revoked_by: str | None = None,
        reason: str | None = None,
        revoked_at: str | None = None,
    ) -> tuple[EnterprisePersonKBShareRecord | None, int]:
        await self._ensure_initialized()

        def write(
            conn: sqlite3.Connection,
        ) -> tuple[EnterprisePersonKBShareRecord | None, int]:
            now = revoked_at or utc_now_iso()
            revoked = self._revoke_person_kb_shares_locked(
                conn,
                kb_id=kb_id,
                target_account_id=target_account_id,
                actor_user_id=revoked_by,
                now=now,
                audit_event_type="person_kb_share_revoked",
                reason=reason,
            )
            row = conn.execute(
                "SELECT * FROM enterprise_person_kb_shares "
                "WHERE kb_id = ? AND target_account_id = ?",
                (kb_id, target_account_id),
            ).fetchone()
            return (
                EnterprisePersonKBShareRecord.from_row(row)
                if row is not None
                else None
            ), revoked

        return await self._write(write)

    async def get_person_kb_share(
        self, kb_id: str, target_account_id: str
    ) -> EnterprisePersonKBShareRecord | None:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM enterprise_person_kb_shares "
                "WHERE kb_id = ? AND target_account_id = ?",
                (kb_id, target_account_id),
            ).fetchone()
        return (
            EnterprisePersonKBShareRecord.from_row(row) if row is not None else None
        )

    async def list_person_kb_shares(
        self,
        *,
        kb_id: str | None = None,
        person_id: str | None = None,
        target_account_id: str | None = None,
        only_active: bool = False,
    ) -> list[EnterprisePersonKBShareRecord]:
        await self._ensure_initialized()
        clauses: list[str] = []
        params: list[Any] = []
        if kb_id is not None:
            clauses.append("kb_id = ?")
            params.append(kb_id)
        if person_id is not None:
            clauses.append("person_id = ?")
            params.append(person_id)
        if target_account_id is not None:
            clauses.append("target_account_id = ?")
            params.append(target_account_id)
        if only_active:
            clauses.append("status = 'active'")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM enterprise_person_kb_shares "
                f"{where} ORDER BY created_at ASC, id ASC",
                tuple(params),
            ).fetchall()
        return [EnterprisePersonKBShareRecord.from_row(row) for row in rows]

    async def kb_has_active_person_share_for_tenant(
        self, kb_id: str, tenant_id: str
    ) -> bool:
        """Oversight predicate for the target department's tenant admins.

        The JOIN re-checks the target account's CURRENT canonical tenant so a
        share goes dark for the old department the moment the account moves
        (the tenant-move hook also revokes it; this is the fail-closed read).
        """

        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM enterprise_person_kb_shares s
                JOIN enterprise_users u ON u.id = s.target_account_id
                WHERE s.kb_id = ? AND s.status = 'active'
                    AND s.target_tenant_id = ? AND u.tenant_id = ?
                LIMIT 1
                """,
                (kb_id, tenant_id, tenant_id),
            ).fetchone()
        return row is not None

    async def upsert_kb_acl(
        self, acl: KBACLRecord, *, expected_generation: str | None = None
    ) -> KBACLRecord:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> KBACLRecord:
            self._assert_kb_generation(conn, acl.kb_id, expected_generation)
            conn.execute(
                """
                INSERT INTO enterprise_kb_acl (
                    kb_id, user_id, role, granted_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(kb_id, user_id) DO UPDATE SET
                    role = excluded.role,
                    granted_by = excluded.granted_by,
                    updated_at = excluded.updated_at
                """,
                (
                    acl.kb_id,
                    acl.user_id,
                    acl.role,
                    acl.granted_by,
                    acl.created_at,
                    acl.updated_at,
                ),
            )
            row = conn.execute(
                """
                SELECT * FROM enterprise_kb_acl
                WHERE kb_id = ? AND user_id = ?
                """,
                (acl.kb_id, acl.user_id),
            ).fetchone()
            assert row is not None
            return KBACLRecord.from_row(row)

        return await self._write(write)

    async def delete_kb_acl(
        self,
        kb_id: str,
        user_id: str,
        *,
        expected_generation: str | None = None,
    ) -> bool:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> bool:
            self._assert_kb_generation(conn, kb_id, expected_generation)
            cursor = conn.execute(
                "DELETE FROM enterprise_kb_acl WHERE kb_id = ? AND user_id = ?",
                (kb_id, user_id),
            )
            return bool(cursor.rowcount)

        return await self._write(write)

    async def list_kb_acl(self, kb_id: str) -> list[KBACLRecord]:
        await self._ensure_initialized()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM enterprise_kb_acl
                WHERE kb_id = ?
                ORDER BY created_at ASC, user_id ASC
                """,
                (kb_id,),
            ).fetchall()
        return [KBACLRecord.from_row(row) for row in rows]

    async def get_kb_acl_role(self, kb_id: str, user_id: str) -> str | None:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT role FROM enterprise_kb_acl
                WHERE kb_id = ? AND user_id = ?
                """,
                (kb_id, user_id),
            ).fetchone()
        return None if row is None else str(row["role"])

    async def list_kb_ids_for_user(self, user_id: str) -> list[str]:
        await self._ensure_initialized()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT kb_id FROM enterprise_kb_acl
                WHERE user_id = ?
                ORDER BY kb_id ASC
                """,
                (user_id,),
            ).fetchall()
        return [str(row["kb_id"]) for row in rows]

    async def upsert_enterprise_tenant(
        self, tenant: EnterpriseTenantRecord
    ) -> EnterpriseTenantRecord:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> EnterpriseTenantRecord:
            conn.execute(
                """
                INSERT INTO enterprise_tenants (
                    id, name, description, status, metadata_json,
                    created_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    description = excluded.description,
                    status = excluded.status,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    tenant.id,
                    tenant.name,
                    tenant.description,
                    tenant.status,
                    _dumps_json(tenant.metadata),
                    tenant.created_by,
                    tenant.created_at,
                    tenant.updated_at,
                ),
            )
            row = conn.execute(
                "SELECT * FROM enterprise_tenants WHERE id = ?", (tenant.id,)
            ).fetchone()
            assert row is not None
            return EnterpriseTenantRecord.from_row(row)

        return await self._write(write)

    async def get_enterprise_tenant_by_id(
        self, tenant_id: str
    ) -> EnterpriseTenantRecord | None:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM enterprise_tenants WHERE id = ?", (tenant_id,)
            ).fetchone()
        return EnterpriseTenantRecord.from_row(row) if row is not None else None

    async def list_enterprise_tenants(self) -> list[EnterpriseTenantRecord]:
        await self._ensure_initialized()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM enterprise_tenants
                ORDER BY created_at ASC, id ASC
                """
            ).fetchall()
        return [EnterpriseTenantRecord.from_row(row) for row in rows]

    async def delete_enterprise_tenant(self, tenant_id: str) -> bool:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> bool:
            if (
                conn.execute(
                    "SELECT 1 FROM enterprise_tenants WHERE id = ?", (tenant_id,)
                ).fetchone()
                is None
            ):
                return False
            now = utc_now_iso()
            # Departments are deleted only once structurally empty, so any
            # share still targeting this tenant is a straggler (its target
            # account was detached through a path that bypassed the move
            # hook). Revoke defensively before dropping the grant tables.
            self._revoke_person_kb_shares_locked(
                conn,
                target_tenant_id=tenant_id,
                actor_user_id=None,
                now=now,
                audit_event_type="person_kb_share_revoked_by_account_change",
                reason="tenant_deleted",
            )
            conn.execute(
                """
                UPDATE enterprise_users
                SET tenant_id = NULL, updated_at = ?
                WHERE tenant_id = ?
                """,
                (now, tenant_id),
            )
            conn.execute(
                "DELETE FROM enterprise_tenant_memberships WHERE tenant_id = ?",
                (tenant_id,),
            )
            conn.execute(
                "DELETE FROM enterprise_tenant_kb_acl WHERE tenant_id = ?",
                (tenant_id,),
            )
            conn.execute(
                "DELETE FROM enterprise_tenant_user_kb_overrides WHERE tenant_id = ?",
                (tenant_id,),
            )
            cursor = conn.execute(
                "DELETE FROM enterprise_tenants WHERE id = ?", (tenant_id,)
            )
            return bool(cursor.rowcount)

        return await self._write(write)

    async def upsert_tenant_membership(
        self, membership: EnterpriseTenantMembershipRecord
    ) -> EnterpriseTenantMembershipRecord:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> EnterpriseTenantMembershipRecord:
            user_row = conn.execute(
                "SELECT * FROM enterprise_users WHERE id = ?", (membership.user_id,)
            ).fetchone()
            if user_row is None:
                raise MetadataRecordNotFoundError(
                    f"User '{membership.user_id}' not found"
                )
            user = EnterpriseUserRecord.from_row(user_row)
            updated_user = EnterpriseUserRecord(
                **{
                    **user.to_dict(),
                    "tenant_id": membership.tenant_id,
                    "updated_at": membership.updated_at,
                }
            )
            _saved_user, saved_membership = (
                self._upsert_enterprise_user_with_membership(
                    conn,
                    updated_user,
                    membership=membership,
                    expected_updated_at=user.updated_at,
                    expected_token_version=user.token_version,
                    expected_tenant_id=user.tenant_id,
                    allow_tenant_change=True,
                )
            )
            assert saved_membership is not None
            return saved_membership

        return await self._write(write)

    async def delete_tenant_membership(self, tenant_id: str, user_id: str) -> bool:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> bool:
            membership_row = conn.execute(
                """
                SELECT * FROM enterprise_tenant_memberships
                WHERE tenant_id = ? AND user_id = ?
                """,
                (tenant_id, user_id),
            ).fetchone()
            if membership_row is None:
                return False
            user_row = conn.execute(
                "SELECT * FROM enterprise_users WHERE id = ?", (user_id,)
            ).fetchone()
            if user_row is not None and user_row["tenant_id"] == tenant_id:
                user = EnterpriseUserRecord.from_row(user_row)
                cleared_user = EnterpriseUserRecord(
                    **{
                        **user.to_dict(),
                        "tenant_id": None,
                        "updated_at": utc_now_iso(),
                    }
                )
                self._upsert_enterprise_user_with_membership(
                    conn,
                    cleared_user,
                    membership=None,
                    expected_updated_at=user.updated_at,
                    expected_token_version=user.token_version,
                    expected_tenant_id=user.tenant_id,
                    allow_tenant_change=True,
                )
            else:
                conn.execute(
                    """
                    DELETE FROM enterprise_tenant_memberships
                    WHERE tenant_id = ? AND user_id = ?
                    """,
                    (tenant_id, user_id),
                )
                conn.execute(
                    """
                    DELETE FROM enterprise_tenant_user_kb_overrides
                    WHERE tenant_id = ? AND user_id = ?
                    """,
                    (tenant_id, user_id),
                )
            return True

        return await self._write(write)

    async def list_tenant_memberships(
        self, tenant_id: str
    ) -> list[EnterpriseTenantMembershipRecord]:
        await self._ensure_initialized()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM enterprise_tenant_memberships
                WHERE tenant_id = ?
                ORDER BY created_at ASC, user_id ASC
                """,
                (tenant_id,),
            ).fetchall()
        return [EnterpriseTenantMembershipRecord.from_row(row) for row in rows]

    async def list_user_tenant_memberships(
        self, user_id: str
    ) -> list[EnterpriseTenantMembershipRecord]:
        await self._ensure_initialized()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM enterprise_tenant_memberships
                WHERE user_id = ?
                ORDER BY tenant_id ASC
                """,
                (user_id,),
            ).fetchall()
        return [EnterpriseTenantMembershipRecord.from_row(row) for row in rows]

    async def get_tenant_membership(
        self, tenant_id: str, user_id: str
    ) -> EnterpriseTenantMembershipRecord | None:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM enterprise_tenant_memberships
                WHERE tenant_id = ? AND user_id = ?
                """,
                (tenant_id, user_id),
            ).fetchone()
        return None if row is None else EnterpriseTenantMembershipRecord.from_row(row)

    async def upsert_tenant_kb_acl(
        self,
        acl: EnterpriseTenantKBACLRecord,
        *,
        expected_generation: str | None = None,
    ) -> EnterpriseTenantKBACLRecord:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> EnterpriseTenantKBACLRecord:
            self._assert_kb_generation(conn, acl.kb_id, expected_generation)
            conn.execute(
                """
                INSERT INTO enterprise_tenant_kb_acl (
                    tenant_id, kb_id, role, granted_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, kb_id) DO UPDATE SET
                    role = excluded.role,
                    granted_by = excluded.granted_by,
                    updated_at = excluded.updated_at
                """,
                (
                    acl.tenant_id,
                    acl.kb_id,
                    acl.role,
                    acl.granted_by,
                    acl.created_at,
                    acl.updated_at,
                ),
            )
            row = conn.execute(
                """
                SELECT * FROM enterprise_tenant_kb_acl
                WHERE tenant_id = ? AND kb_id = ?
                """,
                (acl.tenant_id, acl.kb_id),
            ).fetchone()
            assert row is not None
            return EnterpriseTenantKBACLRecord.from_row(row)

        return await self._write(write)

    async def delete_tenant_kb_acl(
        self,
        tenant_id: str,
        kb_id: str,
        *,
        expected_generation: str | None = None,
    ) -> bool:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> bool:
            self._assert_kb_generation(conn, kb_id, expected_generation)
            cursor = conn.execute(
                """
                DELETE FROM enterprise_tenant_kb_acl
                WHERE tenant_id = ? AND kb_id = ?
                """,
                (tenant_id, kb_id),
            )
            return bool(cursor.rowcount)

        return await self._write(write)

    async def list_kb_tenant_acl(
        self, kb_id: str
    ) -> list[EnterpriseTenantKBACLRecord]:
        await self._ensure_initialized()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM enterprise_tenant_kb_acl
                WHERE kb_id = ?
                ORDER BY created_at ASC, tenant_id ASC
                """,
                (kb_id,),
            ).fetchall()
        return [EnterpriseTenantKBACLRecord.from_row(row) for row in rows]

    async def get_tenant_kb_acl_role(self, tenant_id: str, kb_id: str) -> str | None:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT role FROM enterprise_tenant_kb_acl
                WHERE tenant_id = ? AND kb_id = ?
                """,
                (tenant_id, kb_id),
            ).fetchone()
        return None if row is None else str(row["role"])

    async def list_kb_ids_for_tenants(self, tenant_ids: Sequence[str]) -> list[str]:
        await self._ensure_initialized()
        normalized_ids = sorted({tenant_id for tenant_id in tenant_ids if tenant_id})
        if not normalized_ids:
            return []
        placeholders = ",".join("?" for _ in normalized_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT DISTINCT kb_id FROM enterprise_tenant_kb_acl
                WHERE tenant_id IN ({placeholders})
                ORDER BY kb_id ASC
                """,
                normalized_ids,
            ).fetchall()
        return [str(row["kb_id"]) for row in rows]

    async def get_tenant_user_kb_override(
        self, tenant_id: str, kb_id: str, user_id: str
    ) -> EnterpriseTenantUserKBOverrideRecord | None:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM enterprise_tenant_user_kb_overrides
                WHERE tenant_id = ? AND kb_id = ? AND user_id = ?
                """,
                (tenant_id, kb_id, user_id),
            ).fetchone()
        return (
            EnterpriseTenantUserKBOverrideRecord.from_row(row)
            if row is not None
            else None
        )

    async def list_tenant_user_kb_overrides(
        self, tenant_id: str, kb_id: str
    ) -> list[EnterpriseTenantUserKBOverrideRecord]:
        await self._ensure_initialized()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM enterprise_tenant_user_kb_overrides
                WHERE tenant_id = ? AND kb_id = ?
                ORDER BY created_at ASC, user_id ASC
                """,
                (tenant_id, kb_id),
            ).fetchall()
        return [EnterpriseTenantUserKBOverrideRecord.from_row(row) for row in rows]

    async def list_user_tenant_kb_overrides(
        self,
        user_id: str,
        *,
        tenant_ids: Sequence[str] | None = None,
        kb_id: str | None = None,
    ) -> list[EnterpriseTenantUserKBOverrideRecord]:
        """List a user's overrides, optionally constrained to memberships/KB."""
        await self._ensure_initialized()
        clauses = ["user_id = ?"]
        params: list[Any] = [user_id]
        if tenant_ids is not None:
            normalized_ids = sorted({item for item in tenant_ids if item})
            if not normalized_ids:
                return []
            placeholders = ",".join("?" for _ in normalized_ids)
            clauses.append(f"tenant_id IN ({placeholders})")
            params.extend(normalized_ids)
        if kb_id is not None:
            clauses.append("kb_id = ?")
            params.append(kb_id)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM enterprise_tenant_user_kb_overrides
                WHERE {" AND ".join(clauses)}
                ORDER BY tenant_id ASC, kb_id ASC
                """,
                params,
            ).fetchall()
        return [EnterpriseTenantUserKBOverrideRecord.from_row(row) for row in rows]

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

        def write(conn: sqlite3.Connection) -> EnterpriseTenantUserKBOverrideRecord:
            self._assert_kb_generation(conn, record.kb_id, expected_generation)
            user_row = conn.execute(
                "SELECT * FROM enterprise_users WHERE id = ?",
                (record.user_id,),
            ).fetchone()
            membership_row = conn.execute(
                """
                SELECT * FROM enterprise_tenant_memberships
                WHERE tenant_id = ? AND user_id = ?
                """,
                (record.tenant_id, record.user_id),
            ).fetchone()
            current_user = (
                EnterpriseUserRecord.from_row(user_row)
                if user_row is not None
                else None
            )
            current_membership = (
                EnterpriseTenantMembershipRecord.from_row(membership_row)
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
            conn.execute(
                """
                INSERT INTO enterprise_tenant_user_kb_overrides (
                    tenant_id, kb_id, user_id, effect, role, granted_by,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, kb_id, user_id) DO UPDATE SET
                    effect = excluded.effect,
                    role = excluded.role,
                    granted_by = excluded.granted_by,
                    updated_at = excluded.updated_at
                """,
                (
                    record.tenant_id,
                    record.kb_id,
                    record.user_id,
                    record.effect,
                    record.role,
                    record.granted_by,
                    record.created_at,
                    record.updated_at,
                ),
            )
            row = conn.execute(
                """
                SELECT * FROM enterprise_tenant_user_kb_overrides
                WHERE tenant_id = ? AND kb_id = ? AND user_id = ?
                """,
                (record.tenant_id, record.kb_id, record.user_id),
            ).fetchone()
            assert row is not None
            return EnterpriseTenantUserKBOverrideRecord.from_row(row)

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
        """Write a tenant-scoped deny; reset physically removes the row."""
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

        def write(conn: sqlite3.Connection) -> bool:
            self._assert_kb_generation(conn, kb_id, expected_generation)
            if (
                expected_user is not _EXPECTATION_UNSET
                or expected_membership is not _EXPECTATION_UNSET
            ):
                user_row = conn.execute(
                    "SELECT * FROM enterprise_users WHERE id = ?",
                    (user_id,),
                ).fetchone()
                membership_row = conn.execute(
                    """
                    SELECT * FROM enterprise_tenant_memberships
                    WHERE tenant_id = ? AND user_id = ?
                    """,
                    (tenant_id, user_id),
                ).fetchone()
                _assert_tenant_user_kb_override_target_preconditions(
                    tenant_id,
                    user_id,
                    EnterpriseUserRecord.from_row(user_row)
                    if user_row is not None
                    else None,
                    EnterpriseTenantMembershipRecord.from_row(membership_row)
                    if membership_row is not None
                    else None,
                    expected_user=expected_user,
                    expected_membership=expected_membership,
                )
            cursor = conn.execute(
                """
                DELETE FROM enterprise_tenant_user_kb_overrides
                WHERE tenant_id = ? AND kb_id = ? AND user_id = ?
                """,
                (tenant_id, kb_id, user_id),
            )
            return bool(cursor.rowcount)

        return await self._write(write)

    async def append_audit_event(
        self, event: AuditEventRecord
    ) -> AuditEventRecord:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> AuditEventRecord:
            conn.execute(
                """
                INSERT INTO enterprise_audit_events (
                    id, event_type, actor_user_id, actor_tenant_id, target_type,
                    target_id, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.event_type,
                    event.actor_user_id,
                    event.actor_tenant_id,
                    event.target_type,
                    event.target_id,
                    _dumps_json(event.metadata),
                    event.created_at,
                ),
            )
            row = conn.execute(
                "SELECT * FROM enterprise_audit_events WHERE id = ?", (event.id,)
            ).fetchone()
            assert row is not None
            return AuditEventRecord.from_row(row)

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
        for column, value in (
            ("event_type", event_type),
            ("actor_user_id", actor_user_id),
            ("actor_tenant_id", actor_tenant_id),
            ("target_type", target_type),
            ("target_id", target_id),
        ):
            if value:
                clauses.append(f"{column} = ?")
                params.append(value)
        if created_after:
            clauses.append("created_at >= ?")
            params.append(created_after)
        if created_before:
            clauses.append("created_at <= ?")
            params.append(created_before)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.extend([limit, offset])
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM enterprise_audit_events
                {where}
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                tuple(params),
            ).fetchall()
        return [AuditEventRecord.from_row(row) for row in rows]

    async def purge_kb_metadata(
        self,
        kb_id: str,
        generation: str | None = None,
        *,
        delete_job_id: str | None = None,
    ) -> dict[str, int]:
        """Hard delete all SQLite control-plane state for a KB.

        Returns the row count removed per table for audit purposes. Does
        NOT touch on-disk artifacts / inputs / vector / graph storage —
        callers must orchestrate those separately. Supplying ``delete_job_id``
        enables the strict hard-delete path: lifecycle must already be bound to
        that deleting generation/job, the clear job row is retained, and the
        lifecycle remains ``deleting`` until :meth:`complete_kb_deletion`.
        """
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> dict[str, int]:
            counts: dict[str, int] = {}
            updated_keys = 0
            now = utc_now_iso()
            if generation is not None:
                _validate_kb_lifecycle_identity(kb_id, generation)
            strict_generation = generation
            if delete_job_id is not None:
                _validate_delete_job_id(delete_job_id)
                if strict_generation is None:
                    raise MetadataStoreError(
                        "Strict KB metadata purge requires a generation"
                    )
            lifecycle_row = conn.execute(
                "SELECT * FROM enterprise_kb_lifecycle WHERE kb_id = ?", (kb_id,)
            ).fetchone()
            if delete_job_id is not None:
                assert strict_generation is not None
                if lifecycle_row is None:
                    raise _missing_kb_lifecycle_conflict(
                        kb_id,
                        strict_generation,
                        expected_state="deleting",
                        expected_delete_job_id=delete_job_id,
                    )
                lifecycle = KBLifecycleRecord.from_row(lifecycle_row)
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
            elif lifecycle_row is None:
                tombstone_generation = generation or (
                    f"{_LEGACY_KB_TOMBSTONE_PREFIX}{kb_id}"
                )
                _validate_kb_lifecycle_identity(kb_id, tombstone_generation)
                conn.execute(
                    """
                    INSERT INTO enterprise_kb_lifecycle (
                        kb_id, generation, state, activated_at, deleted_at,
                        updated_at, delete_job_id
                    ) VALUES (?, ?, 'deleted', ?, ?, ?, NULL)
                    """,
                    (kb_id, tombstone_generation, now, now, now),
                )
            else:
                lifecycle = KBLifecycleRecord.from_row(lifecycle_row)
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
                    cursor = conn.execute(
                        """
                        UPDATE enterprise_kb_lifecycle
                        SET state = 'deleted', deleted_at = ?, updated_at = ?,
                            delete_job_id = NULL
                        WHERE kb_id = ? AND generation = ? AND state = 'active'
                        """,
                        (now, now, kb_id, lifecycle.generation),
                    )
                    if cursor.rowcount != 1:
                        refreshed_row = conn.execute(
                            "SELECT * FROM enterprise_kb_lifecycle WHERE kb_id = ?",
                            (kb_id,),
                        ).fetchone()
                        if refreshed_row is None:
                            raise _missing_kb_lifecycle_conflict(
                                kb_id,
                                lifecycle.generation,
                                expected_state="active",
                            )
                        raise _kb_lifecycle_conflict(
                            kb_id,
                            lifecycle.generation,
                            KBLifecycleRecord.from_row(refreshed_row),
                        )
            key_rows = conn.execute(
                "SELECT id, scopes_json FROM enterprise_api_keys"
            ).fetchall()
            for key_row in key_rows:
                scopes = _loads_json_object(key_row["scopes_json"])
                kb_roles = scopes.get("kb_roles", {})
                if not isinstance(kb_roles, dict):
                    raise MetadataStoreError(
                        "Service API key kb_roles must be an object"
                    )
                if kb_id not in kb_roles:
                    continue
                scopes["kb_roles"] = {
                    role_kb_id: role
                    for role_kb_id, role in kb_roles.items()
                    if role_kb_id != kb_id
                }
                conn.execute(
                    """
                    UPDATE enterprise_api_keys
                    SET scopes_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (_dumps_json(scopes), now, key_row["id"]),
                )
                updated_keys += 1
            counts["enterprise_api_keys"] = updated_keys
            for table in (
                "document_artifacts",
                "document_source_keys",
                "enterprise_kb_acl",
                "enterprise_person_kb_shares",
                "enterprise_tenant_kb_acl",
                "enterprise_tenant_user_kb_overrides",
                "enterprise_user_kb_query_settings",
                "kb_config_versions",
            ):
                cursor = conn.execute(f"DELETE FROM {table} WHERE kb_id = ?", (kb_id,))
                counts[table] = cursor.rowcount or 0
            if delete_job_id is None:
                jobs_cursor = conn.execute("DELETE FROM jobs WHERE kb_id = ?", (kb_id,))
            else:
                jobs_cursor = conn.execute(
                    "DELETE FROM jobs WHERE kb_id = ? AND id <> ?",
                    (kb_id, delete_job_id),
                )
            counts["jobs"] = jobs_cursor.rowcount or 0
            documents_cursor = conn.execute(
                "DELETE FROM documents WHERE kb_id = ?", (kb_id,)
            )
            counts["documents"] = documents_cursor.rowcount or 0
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
        """Recover transient jobs without stealing one from a live owner.

        Queued semantics are unchanged: resumable queued jobs remain queued and
        every other queued job is failed immediately. Running/retrying/
        cancelling rows are recovered only when their ``updated_at`` is older
        than ``grace_seconds`` and a non-blocking :meth:`job_execution_guard`
        succeeds. Status, timestamp, and claim identity are checked again while
        ownership is held. Resumable ``clear_kb`` rows are requeued in place so
        their post-catalog-purge tail can finish; other orphan rows and their
        documents retain the existing failed-recovery behavior.

        This is a standard crash-stop boundary, not a lease: a live owner holds
        the OS/session lock, while process or database-session death releases it
        for exactly one recovery owner. Retry and claim continue on the same
        durable job row; no run-token column is required.

        Direct store calls default to zero grace for backwards compatibility.
        Production startup and :class:`~lightrag.api.job_worker.JobWorker` pass
        a safe grace at least as large as the worker claim gap.
        """
        await self._ensure_initialized()
        resumable = set(resumable_job_types or set())
        cutoff = _orphan_recovery_cutoff(grace_seconds)

        def recover_queued(conn: sqlite3.Connection) -> list[JobRecord]:
            rows = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'queued'
                ORDER BY created_at ASC, id ASC
                """
            ).fetchall()
            updated: list[JobRecord] = []
            now = utc_now_iso()
            for row in rows:
                if (
                    row["job_type"] in resumable
                    and (
                        row["document_id"] is not None
                        or row["job_type"] in _AGGREGATE_RESUMABLE_JOB_TYPES
                    )
                ):
                    continue
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'failed', error_code = ?, error_message = ?,
                        updated_at = ?, finished_at = COALESCE(finished_at, ?)
                    WHERE id = ?
                    """,
                    (error_code, error_message, now, now, row["id"]),
                )
                refreshed = conn.execute(
                    "SELECT * FROM jobs WHERE id = ?", (row["id"],)
                ).fetchone()
                if refreshed is not None:
                    recovered = JobRecord.from_row(refreshed)
                    self._recover_documents_for_job(
                        conn,
                        recovered,
                        error_code=error_code,
                        error_message=error_message,
                        now=now,
                    )
                    updated.append(recovered)
            return updated

        updated = await self._write(recover_queued)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status IN ('running', 'retrying', 'cancelling')
                    AND updated_at <= ?
                ORDER BY updated_at ASC, id ASC
                """,
                (cutoff,),
            ).fetchall()
        candidates = [JobRecord.from_row(row) for row in rows]

        for candidate in candidates:
            async with self.job_execution_guard(
                candidate.id, wait=False
            ) as acquired:
                if not acquired:
                    continue

                def recover_candidate(
                    conn: sqlite3.Connection,
                ) -> JobRecord | None:
                    row = conn.execute(
                        "SELECT * FROM jobs WHERE id = ?", (candidate.id,)
                    ).fetchone()
                    if row is None:
                        return None
                    current = JobRecord.from_row(row)
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
                        conn.execute(
                            """
                            UPDATE jobs
                            SET status = 'queued', error_code = NULL,
                                error_message = NULL, updated_at = ?, queued_at = ?,
                                started_at = NULL, finished_at = NULL,
                                cancelled_at = NULL,
                                retry_count = MIN(retry_count + 1, max_retries)
                            WHERE id = ?
                            """,
                            (now, now, current.id),
                        )
                    else:
                        conn.execute(
                            """
                            UPDATE jobs
                            SET status = 'failed', error_code = ?, error_message = ?,
                                updated_at = ?, finished_at = COALESCE(finished_at, ?)
                            WHERE id = ?
                            """,
                            (error_code, error_message, now, now, current.id),
                        )
                    refreshed = conn.execute(
                        "SELECT * FROM jobs WHERE id = ?", (current.id,)
                    ).fetchone()
                    if refreshed is None:
                        return None
                    recovered = JobRecord.from_row(refreshed)
                    if not requeue_clear:
                        self._recover_documents_for_job(
                            conn,
                            recovered,
                            error_code=error_code,
                            error_message=error_message,
                            now=now,
                        )
                    return recovered

                recovered = await self._write(recover_candidate)
                if recovered is not None:
                    updated.append(recovered)
        return updated

    def _recover_documents_for_job(
        self,
        conn: sqlite3.Connection,
        job: JobRecord,
        *,
        error_code: str,
        error_message: str,
        now: str,
    ) -> None:
        document_ids = _job_recovery_document_ids(job)
        rows = conn.execute(
            """
            SELECT * FROM documents
            WHERE kb_id = ? AND deleted_at IS NULL
                AND status IN (
                    'parse_queued', 'parsing', 'build_queued', 'building',
                    'deleting', 'replacing'
                )
            """,
            (job.kb_id,),
        ).fetchall()
        for row in rows:
            document = DocumentRecord.from_row(row)
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
            conn.execute(
                """
                UPDATE documents
                SET status = ?, error_code = ?, error_message = ?, updated_at = ?
                WHERE kb_id = ? AND id = ?
                """,
                (
                    target_status,
                    error_code,
                    error_message,
                    now,
                    document.kb_id,
                    document.id,
                ),
            )

    async def claim_next_worker_job(
        self,
        *,
        job_types: Sequence[str],
        max_queued_at: str | None = None,
    ) -> JobRecord | None:
        """Atomically claim the oldest eligible ``queued`` job for a worker.

        Selects the oldest ``queued`` job whose ``job_type`` is in
        ``job_types`` and (when ``max_queued_at`` is given) whose ``queued_at``
        is at or before that timestamp, then transitions it to ``running`` in
        the SAME serialized write transaction. Because :meth:`_write`
        serializes all writers, exactly one caller can win the
        ``queued → running`` transition for a given job — a concurrent
        in-process background task attempting the same transition will then
        see ``running`` and abort. Returns the claimed :class:`JobRecord`, or
        ``None`` when no eligible job exists.

        ``max_queued_at`` implements a grace window: callers pass "now minus
        a few seconds" so freshly-created jobs (which their in-process task
        flips to ``running`` within milliseconds) are not stolen mid-creation;
        only jobs that have sat ``queued`` beyond the window — retried jobs and
        jobs whose owner crashed/restarted — are claimed.

        Single-document jobs are eligible, plus the aggregate types listed in
        :data:`_AGGREGATE_RESUMABLE_JOB_TYPES` whose payloads carry everything
        needed to re-drive them: ``documents:batch-delete`` (document ids +
        options), ``batch-parse`` / ``batch-build`` / ``batch-reindex`` and
        multi-file ``upload`` / ``texts`` auto_parse (document ids; sources
        already on disk), and ``clear_kb`` (kb_id/workspace; idempotent clear).
        The worker only actually claims a type if an executor is registered for
        it, so enabling a new (aggregate or single-document) type is gated by
        both this predicate and executor registration — e.g. single-document
        ``replace`` matches the predicate but has no durable executor, so it is
        never claimed.
        """
        await self._ensure_initialized()
        if not job_types:
            return None
        where, params = _worker_eligibility_sql(job_types)
        if max_queued_at is not None:
            where += " AND queued_at <= ?"
            params.append(max_queued_at)

        def write(conn: sqlite3.Connection) -> JobRecord | None:
            row = conn.execute(
                f"""
                SELECT * FROM jobs
                WHERE {where}
                ORDER BY queued_at ASC, id ASC
                LIMIT 1
                """,
                params,
            ).fetchone()
            if row is None:
                return None
            now = utc_now_iso()
            conn.execute(
                """
                UPDATE jobs
                SET status = 'running', progress = ?, updated_at = ?,
                    started_at = COALESCE(started_at, ?)
                WHERE id = ? AND status = 'queued'
                """,
                (max(float(row["progress"] or 0.0), 0.1), now, now, row["id"]),
            )
            refreshed = conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (row["id"],)
            ).fetchone()
            if refreshed is None:
                return None
            return JobRecord.from_row(refreshed)

        return await self._write(write)

    def _upsert_enterprise_user_with_membership(
        self,
        conn: sqlite3.Connection,
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

        current_row = conn.execute(
            "SELECT * FROM enterprise_users WHERE id = ?", (user.id,)
        ).fetchone()
        current_user = (
            EnterpriseUserRecord.from_row(current_row)
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
            membership_rows = conn.execute(
                """
                SELECT * FROM enterprise_tenant_memberships
                WHERE user_id = ? ORDER BY tenant_id ASC
                """,
                (user.id,),
            ).fetchall()
            _assert_enterprise_user_membership_precondition(
                user.id,
                [
                    EnterpriseTenantMembershipRecord.from_row(row)
                    for row in membership_rows
                ],
                expected_membership=expected_membership,
            )

        existing_membership: EnterpriseTenantMembershipRecord | None = None
        if canonical_tenant is not None:
            row = conn.execute(
                """
                SELECT * FROM enterprise_tenant_memberships
                WHERE tenant_id = ? AND user_id = ?
                """,
                (canonical_tenant, user.id),
            ).fetchone()
            if row is not None:
                existing_membership = EnterpriseTenantMembershipRecord.from_row(row)

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

        conn.execute(
            """
            INSERT INTO enterprise_users (
                id, username, password_hash, system_role, status, tenant_id,
                can_create_kb, can_use_bypass_query, can_delete_documents,
                can_use_agent_query, can_download_files, token_version,
                metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                username = excluded.username,
                password_hash = excluded.password_hash,
                system_role = excluded.system_role,
                status = excluded.status,
                tenant_id = excluded.tenant_id,
                can_create_kb = excluded.can_create_kb,
                can_use_bypass_query = excluded.can_use_bypass_query,
                can_delete_documents = excluded.can_delete_documents,
                can_use_agent_query = excluded.can_use_agent_query,
                can_download_files = excluded.can_download_files,
                token_version = excluded.token_version,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                user.id,
                user.username,
                user.password_hash,
                user.system_role,
                user.status,
                canonical_tenant,
                int(user.can_create_kb),
                int(user.can_use_bypass_query),
                int(user.can_delete_documents),
                int(user.can_use_agent_query),
                int(user.can_download_files),
                user.token_version,
                _dumps_json(user.metadata),
                user.created_at,
                user.updated_at,
            ),
        )
        if current_user is not None and current_user.tenant_id != canonical_tenant:
            # Canonical tenant changed: person-KB shares that granted this
            # account access were scoped to the OLD department (the department
            # admins' oversight was tied to it). They lose their basis; the
            # person can re-share into the new department explicitly. Shares
            # this account OWNS are not departmental and survive.
            self._revoke_person_kb_shares_locked(
                conn,
                target_only_account_id=user.id,
                actor_user_id=None,
                now=user.updated_at,
                audit_event_type="person_kb_share_revoked_by_account_change",
                reason="tenant_changed",
            )
        if canonical_tenant is None:
            conn.execute(
                "DELETE FROM enterprise_tenant_memberships WHERE user_id = ?",
                (user.id,),
            )
        else:
            conn.execute(
                """
                DELETE FROM enterprise_tenant_memberships
                WHERE user_id = ? AND tenant_id <> ?
                """,
                (user.id, canonical_tenant),
            )
        persisted_membership: EnterpriseTenantMembershipRecord | None = None
        if selected_membership is not None:
            conn.execute(
                """
                INSERT INTO enterprise_tenant_memberships (
                    tenant_id, user_id, role, granted_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, user_id) DO UPDATE SET
                    role = excluded.role,
                    granted_by = excluded.granted_by,
                    updated_at = excluded.updated_at
                """,
                (
                    selected_membership.tenant_id,
                    selected_membership.user_id,
                    selected_membership.role,
                    selected_membership.granted_by,
                    selected_membership.created_at,
                    selected_membership.updated_at,
                ),
            )
            persisted_membership = selected_membership

        if canonical_tenant is None:
            conn.execute(
                "DELETE FROM enterprise_tenant_user_kb_overrides WHERE user_id = ?",
                (user.id,),
            )
        else:
            conn.execute(
                """
                DELETE FROM enterprise_tenant_user_kb_overrides
                WHERE user_id = ? AND tenant_id <> ?
                """,
                (user.id, canonical_tenant),
            )

        row = conn.execute(
            "SELECT * FROM enterprise_users WHERE id = ?", (user.id,)
        ).fetchone()
        assert row is not None
        return EnterpriseUserRecord.from_row(row), persisted_membership

    def _assert_kb_generation(
        self,
        conn: sqlite3.Connection,
        kb_id: str,
        expected_generation: str | None,
    ) -> KBLifecycleRecord | None:
        if expected_generation is not None:
            _validate_kb_lifecycle_identity(kb_id, expected_generation)
        row = conn.execute(
            "SELECT * FROM enterprise_kb_lifecycle WHERE kb_id = ?", (kb_id,)
        ).fetchone()
        if row is None:
            # KBs that predate lifecycle registration remain writable.
            return None
        current = KBLifecycleRecord.from_row(row)
        if (
            current.state != "active"
            or expected_generation is None
            or expected_generation != current.generation
        ):
            raise _kb_lifecycle_conflict(kb_id, expected_generation, current)
        return current

    def _activate_kb_generation(
        self,
        conn: sqlite3.Connection,
        kb_id: str,
        generation: str,
        *,
        activated_at: str,
    ) -> KBLifecycleRecord:
        _validate_kb_lifecycle_identity(kb_id, generation)
        row = conn.execute(
            "SELECT * FROM enterprise_kb_lifecycle WHERE kb_id = ?", (kb_id,)
        ).fetchone()
        if row is None:
            conn.execute(
                """
                INSERT INTO enterprise_kb_lifecycle (
                    kb_id, generation, state, activated_at, deleted_at, updated_at,
                    delete_job_id
                ) VALUES (?, ?, 'active', ?, NULL, ?, NULL)
                """,
                (kb_id, generation, activated_at, activated_at),
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

        current = KBLifecycleRecord.from_row(row)
        if current.state == "active":
            if current.generation == generation:
                return current
            raise _kb_lifecycle_conflict(kb_id, generation, current)
        if current.state == "deleting":
            # No generation, including a fresh one, may replace an identity
            # while its destructive cleanup is incomplete.
            raise _kb_lifecycle_conflict(kb_id, generation, current)
        if current.generation == generation:
            # A tombstoned identity cannot be resurrected; recreations need a
            # fresh generation so delayed grants from the old KB stay invalid.
            raise _kb_lifecycle_conflict(kb_id, generation, current)

        cursor = conn.execute(
            """
            UPDATE enterprise_kb_lifecycle
            SET generation = ?, state = 'active', activated_at = ?,
                deleted_at = NULL, updated_at = ?, delete_job_id = NULL
            WHERE kb_id = ? AND generation = ? AND state = 'deleted'
            """,
            (
                generation,
                activated_at,
                activated_at,
                kb_id,
                current.generation,
            ),
        )
        if cursor.rowcount != 1:
            refreshed_row = conn.execute(
                "SELECT * FROM enterprise_kb_lifecycle WHERE kb_id = ?", (kb_id,)
            ).fetchone()
            if refreshed_row is None:
                raise _missing_kb_lifecycle_conflict(
                    kb_id, generation, expected_state="deleted"
                )
            raise _kb_lifecycle_conflict(
                kb_id, generation, KBLifecycleRecord.from_row(refreshed_row)
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

    def _begin_kb_deletion(
        self,
        conn: sqlite3.Connection,
        kb_id: str,
        generation: str,
        delete_job_id: str,
    ) -> KBLifecycleRecord:
        row = conn.execute(
            "SELECT * FROM enterprise_kb_lifecycle WHERE kb_id = ?", (kb_id,)
        ).fetchone()
        if row is None:
            raise _missing_kb_lifecycle_conflict(
                kb_id,
                generation,
                expected_state="active",
                expected_delete_job_id=delete_job_id,
            )
        current = KBLifecycleRecord.from_row(row)
        if current.state == "active":
            if current.generation != generation:
                raise _kb_lifecycle_conflict(kb_id, generation, current)
            now = utc_now_iso()
            cursor = conn.execute(
                """
                UPDATE enterprise_kb_lifecycle
                SET state = 'deleting', delete_job_id = ?, deleted_at = NULL,
                    updated_at = ?
                WHERE kb_id = ? AND generation = ? AND state = 'active'
                    AND delete_job_id IS NULL
                """,
                (delete_job_id, now, kb_id, generation),
            )
            if cursor.rowcount == 1:
                return KBLifecycleRecord(
                    kb_id=kb_id,
                    generation=generation,
                    state="deleting",
                    activated_at=current.activated_at,
                    deleted_at=None,
                    updated_at=now,
                    delete_job_id=delete_job_id,
                )
            refreshed_row = conn.execute(
                "SELECT * FROM enterprise_kb_lifecycle WHERE kb_id = ?", (kb_id,)
            ).fetchone()
            if refreshed_row is None:
                raise _missing_kb_lifecycle_conflict(
                    kb_id,
                    generation,
                    expected_state="active",
                    expected_delete_job_id=delete_job_id,
                )
            current = KBLifecycleRecord.from_row(refreshed_row)

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

    def _complete_kb_deletion(
        self,
        conn: sqlite3.Connection,
        kb_id: str,
        generation: str,
        delete_job_id: str,
    ) -> KBLifecycleRecord:
        row = conn.execute(
            "SELECT * FROM enterprise_kb_lifecycle WHERE kb_id = ?", (kb_id,)
        ).fetchone()
        if row is None:
            raise _missing_kb_lifecycle_conflict(
                kb_id,
                generation,
                expected_state="deleting",
                expected_delete_job_id=delete_job_id,
            )
        current = KBLifecycleRecord.from_row(row)
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
        cursor = conn.execute(
            """
            UPDATE enterprise_kb_lifecycle
            SET state = 'deleted', deleted_at = ?, updated_at = ?
            WHERE kb_id = ? AND generation = ? AND state = 'deleting'
                AND delete_job_id = ?
            """,
            (now, now, kb_id, generation, delete_job_id),
        )
        if cursor.rowcount != 1:
            refreshed_row = conn.execute(
                "SELECT * FROM enterprise_kb_lifecycle WHERE kb_id = ?", (kb_id,)
            ).fetchone()
            if refreshed_row is None:
                raise _missing_kb_lifecycle_conflict(
                    kb_id,
                    generation,
                    expected_state="deleting",
                    expected_delete_job_id=delete_job_id,
                )
            refreshed = KBLifecycleRecord.from_row(refreshed_row)
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

    def _kb_operation_lock_path(self, kb_id: str) -> Path:
        digest = hashlib.sha256(kb_id.encode("utf-8")).hexdigest()
        return Path(f"{self.db_path}.kb-operation-locks") / f"{digest}.lock"

    def _job_execution_lock_path(self, job_id: str) -> Path:
        digest = hashlib.sha256(job_id.encode("utf-8")).hexdigest()
        return Path(f"{self.db_path}.job-execution-locks") / f"{digest}.lock"

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def _write(self, callback: Callable[[sqlite3.Connection], _T]) -> _T:
        async with self._lock:
            with _MetadataFileLock(self.lock_path):
                with self._connect() as conn:
                    try:
                        conn.execute("BEGIN IMMEDIATE")
                        result = callback(conn)
                        conn.commit()
                        return result
                    except Exception:
                        conn.rollback()
                        raise

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _initialize_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            BEGIN IMMEDIATE;

            CREATE TABLE IF NOT EXISTS metadata_schema (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                kb_id TEXT NOT NULL,
                workspace TEXT NOT NULL,
                lightrag_doc_id TEXT,
                source_type TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_uri TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                content_type TEXT,
                size_bytes INTEGER NOT NULL DEFAULT 0,
                parser_hash TEXT,
                index_hash TEXT,
                status TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                archived INTEGER NOT NULL DEFAULT 0,
                chunks_count INTEGER,
                entity_count INTEGER,
                relation_count INTEGER,
                error_code TEXT,
                error_message TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_documents_kb_status
                ON documents (kb_id, status);
            CREATE INDEX IF NOT EXISTS idx_documents_kb_source_hash
                ON documents (kb_id, source_hash);
            CREATE INDEX IF NOT EXISTS idx_documents_workspace
                ON documents (workspace);

            CREATE TABLE IF NOT EXISTS document_source_keys (
                kb_id TEXT NOT NULL,
                source_key TEXT NOT NULL,
                document_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (kb_id, source_key),
                FOREIGN KEY (document_id) REFERENCES documents(id)
            );

            CREATE INDEX IF NOT EXISTS idx_document_source_keys_document
                ON document_source_keys (kb_id, document_id);

            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                kb_id TEXT NOT NULL,
                workspace TEXT NOT NULL,
                batch_id TEXT,
                document_id TEXT,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL,
                stage TEXT,
                progress REAL NOT NULL DEFAULT 0,
                total_items INTEGER NOT NULL DEFAULT 0,
                completed_items INTEGER NOT NULL DEFAULT 0,
                failed_items INTEGER NOT NULL DEFAULT 0,
                idempotency_key TEXT,
                config_version_id TEXT,
                config_hash TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0,
                max_retries INTEGER NOT NULL DEFAULT 3,
                payload_json TEXT NOT NULL DEFAULT '{}',
                result_json TEXT,
                error_code TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                queued_at TEXT,
                started_at TEXT,
                finished_at TEXT,
                cancelled_at TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            );

            CREATE INDEX IF NOT EXISTS idx_jobs_kb_status
                ON jobs (kb_id, status);
            CREATE INDEX IF NOT EXISTS idx_jobs_kb_document
                ON jobs (kb_id, document_id);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_kb_type_idempotency
                ON jobs (kb_id, job_type, idempotency_key)
                WHERE idempotency_key IS NOT NULL;

            CREATE TABLE IF NOT EXISTS document_artifacts (
                id TEXT PRIMARY KEY,
                kb_id TEXT NOT NULL,
                workspace TEXT NOT NULL,
                document_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                uri TEXT NOT NULL,
                checksum TEXT,
                size_bytes INTEGER,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            );

            CREATE INDEX IF NOT EXISTS idx_artifacts_kb_document
                ON document_artifacts (kb_id, document_id);
            CREATE INDEX IF NOT EXISTS idx_artifacts_workspace_type
                ON document_artifacts (workspace, artifact_type);

            CREATE TABLE IF NOT EXISTS kb_config_versions (
                id TEXT PRIMARY KEY,
                kb_id TEXT NOT NULL,
                workspace TEXT NOT NULL,
                version INTEGER NOT NULL,
                config_json TEXT NOT NULL,
                parser_hash TEXT,
                index_hash TEXT,
                query_hash TEXT,
                created_at TEXT NOT NULL,
                activated_at TEXT,
                created_by TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_config_versions_kb_version
                ON kb_config_versions (kb_id, version);
            CREATE INDEX IF NOT EXISTS idx_config_versions_workspace
                ON kb_config_versions (workspace);

            CREATE TABLE IF NOT EXISTS enterprise_users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                system_role TEXT NOT NULL,
                status TEXT NOT NULL,
                tenant_id TEXT,
                can_create_kb INTEGER NOT NULL DEFAULT 0,
                can_use_bypass_query INTEGER NOT NULL DEFAULT 0,
                can_delete_documents INTEGER NOT NULL DEFAULT 0,
                can_use_agent_query INTEGER NOT NULL DEFAULT 0,
                can_download_files INTEGER NOT NULL DEFAULT 0,
                token_version INTEGER NOT NULL DEFAULT 1,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
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
                state TEXT NOT NULL CHECK (state IN ('active', 'deleting', 'deleted')),
                activated_at TEXT NOT NULL,
                deleted_at TEXT,
                updated_at TEXT NOT NULL,
                delete_job_id TEXT,
                CHECK (kb_id <> '' AND kb_id = trim(kb_id)),
                CHECK (generation <> '' AND generation = trim(generation)),
                CHECK (delete_job_id IS NULL OR (
                    delete_job_id <> '' AND delete_job_id = trim(delete_job_id)
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
                FOREIGN KEY (user_id) REFERENCES enterprise_users(id)
            );

            CREATE INDEX IF NOT EXISTS idx_enterprise_chat_projects_user
                ON enterprise_chat_projects (user_id, updated_at DESC);

            CREATE TABLE IF NOT EXISTS enterprise_chat_sessions (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                context_rounds INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
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
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                seq INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                append_batch_id TEXT,
                project_event_seq INTEGER,
                memory_reference_time TEXT,
                FOREIGN KEY (session_id) REFERENCES enterprise_chat_sessions(id),
                FOREIGN KEY (user_id) REFERENCES enterprise_users(id),
                CHECK (
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
                first_seq INTEGER NOT NULL,
                last_seq INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                event_id TEXT,
                generation INTEGER,
                graph_group_id TEXT,
                append_batch_id TEXT,
                project_event_seq INTEGER,
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
                active_generation INTEGER,
                desired_generation INTEGER NOT NULL,
                next_event_seq INTEGER NOT NULL DEFAULT 1,
                last_reference_time TEXT,
                state TEXT NOT NULL CHECK (
                    state IN ('active', 'rebuilding', 'deleting', 'failed', 'deleted')
                ),
                state_version INTEGER NOT NULL DEFAULT 1,
                active_config_fingerprint TEXT,
                desired_config_fingerprint TEXT NOT NULL,
                active_graph_store_fingerprint TEXT,
                desired_graph_store_fingerprint TEXT NOT NULL,
                active_rebuild_event_id TEXT,
                last_success_at TEXT,
                last_error_code TEXT,
                last_error_message TEXT,
                last_error_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted_at TEXT,
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
                        trim(desired_graph_store_fingerprint)
                    AND (
                        active_graph_store_fingerprint IS NULL
                        OR (
                            active_graph_store_fingerprint <> ''
                            AND active_graph_store_fingerprint =
                                trim(active_graph_store_fingerprint)
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
                generation INTEGER NOT NULL,
                graph_group_id TEXT NOT NULL UNIQUE,
                config_fingerprint TEXT NOT NULL,
                graph_store_fingerprint TEXT NOT NULL,
                state TEXT NOT NULL CHECK (
                    state IN (
                        'building', 'active', 'retired', 'abandoned',
                        'purge_pending', 'purged'
                    )
                ),
                snapshot_cutoff INTEGER,
                replay_batch_count INTEGER,
                replay_message_count INTEGER,
                replay_byte_count INTEGER,
                snapshot_digest TEXT,
                clear_attempt_no INTEGER NOT NULL DEFAULT 0,
                clear_started_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                activated_at TEXT,
                cleared_at TEXT,
                last_error_code TEXT,
                last_error_message TEXT,
                last_error_at TEXT,
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
                    AND graph_store_fingerprint = trim(graph_store_fingerprint)
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
                event_seq INTEGER NOT NULL,
                generation INTEGER NOT NULL,
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
                available_at TEXT NOT NULL,
                attempt_no INTEGER NOT NULL DEFAULT 0,
                source_session_id TEXT,
                append_batch_id TEXT,
                first_seq INTEGER,
                last_seq INTEGER,
                snapshot_cutoff INTEGER,
                snapshot_batch_count INTEGER,
                snapshot_message_count INTEGER,
                snapshot_byte_count INTEGER,
                snapshot_digest TEXT,
                claim_token TEXT,
                claimed_by TEXT,
                claimed_at TEXT,
                side_effect_started_at TEXT,
                side_effect_state_version INTEGER,
                completed_at TEXT,
                superseded_by_event_id TEXT,
                last_error_code TEXT,
                last_error_message TEXT,
                last_error_at TEXT,
                actor_user_id TEXT,
                actor_tenant_id TEXT,
                target_user_id TEXT,
                target_project_id TEXT,
                target_session_id TEXT,
                target_message_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
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
                    AND graph_store_fingerprint = trim(graph_store_fingerprint)
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
                scopes_json TEXT NOT NULL DEFAULT '{}',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_used_at TEXT,
                revoked_at TEXT,
                revoked_by TEXT,
                expires_at TEXT
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
                PRIMARY KEY (kb_id, user_id),
                FOREIGN KEY (user_id) REFERENCES enterprise_users(id)
            );

            CREATE INDEX IF NOT EXISTS idx_enterprise_kb_acl_user
                ON enterprise_kb_acl (user_id, kb_id);

            CREATE TABLE IF NOT EXISTS enterprise_tenants (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_by TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS enterprise_tenant_memberships (
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                granted_by TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
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
                PRIMARY KEY (tenant_id, kb_id, user_id),
                FOREIGN KEY (tenant_id, user_id)
                    REFERENCES enterprise_tenant_memberships(tenant_id, user_id)
                    ON DELETE CASCADE,
                CHECK (tenant_id <> '' AND tenant_id = trim(tenant_id)),
                CHECK (kb_id <> '' AND kb_id = trim(kb_id)),
                CHECK (user_id <> '' AND user_id = trim(user_id)),
                CHECK (created_at <> '' AND updated_at <> ''),
                CHECK (granted_by IS NULL OR (
                    granted_by <> '' AND granted_by = trim(granted_by)
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
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_enterprise_audit_events_created
                ON enterprise_audit_events (created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_enterprise_audit_events_actor
                ON enterprise_audit_events (actor_user_id);

            CREATE TABLE IF NOT EXISTS enterprise_invitations (
                id TEXT PRIMARY KEY,
                token_hash TEXT NOT NULL UNIQUE,
                token_preview TEXT NOT NULL,
                status TEXT NOT NULL,
                created_by TEXT,
                expires_at TEXT,
                used_by TEXT,
                used_at TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_enterprise_invitations_status
                ON enterprise_invitations (status);
            """
        )
        # Multi-account person identity tables. Idempotent: fresh and existing
        # databases both converge without altering legacy rows. Created inside
        # the same schema-initialization transaction so a half-migrated state
        # cannot persist. See docs/多账号身份关联与切换执行文档.md section 4.
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS enterprise_persons (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                auth_epoch INTEGER NOT NULL DEFAULT 1,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                CHECK (status IN ('active', 'disabled'))
            );

            CREATE INDEX IF NOT EXISTS idx_enterprise_persons_status
                ON enterprise_persons (status);

            CREATE TABLE IF NOT EXISTS enterprise_person_credentials (
                id TEXT PRIMARY KEY,
                person_id TEXT NOT NULL,
                credential_type TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                failed_count INTEGER NOT NULL DEFAULT 0,
                locked_until TEXT,
                last_used_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                CHECK (status IN ('active', 'revoked')),
                CHECK (credential_type IN ('password')),
                FOREIGN KEY (person_id) REFERENCES enterprise_persons(id),
                UNIQUE (person_id, credential_type)
            );

            CREATE INDEX IF NOT EXISTS idx_enterprise_person_credentials_person
                ON enterprise_person_credentials (person_id);

            CREATE TABLE IF NOT EXISTS enterprise_person_enrollment_grants (
                id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                created_by TEXT,
                consumed_by_person TEXT,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                consumed_at TEXT,
                CHECK (status IN ('active', 'consumed', 'revoked', 'expired'))
            );

            CREATE UNIQUE INDEX IF NOT EXISTS uq_person_enrollment_grant_active
                ON enterprise_person_enrollment_grants (account_id)
                WHERE status = 'active';
            CREATE INDEX IF NOT EXISTS idx_enterprise_person_enrollment_grants_account
                ON enterprise_person_enrollment_grants (account_id);

            CREATE TABLE IF NOT EXISTS enterprise_person_account_links (
                id TEXT PRIMARY KEY,
                person_id TEXT NOT NULL,
                account_id TEXT NOT NULL,
                status TEXT NOT NULL,
                bound_by TEXT,
                bound_at TEXT,
                confirmed_by_person_at TEXT,
                revoked_by TEXT,
                revoked_at TEXT,
                reason TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                CHECK (status IN ('pending', 'active', 'revoked')),
                FOREIGN KEY (person_id) REFERENCES enterprise_persons(id),
                FOREIGN KEY (account_id) REFERENCES enterprise_users(id)
                    ON DELETE CASCADE,
                UNIQUE (person_id, account_id)
            );

            CREATE UNIQUE INDEX IF NOT EXISTS uq_person_account_active
                ON enterprise_person_account_links (account_id)
                WHERE status = 'active';
            CREATE INDEX IF NOT EXISTS idx_enterprise_person_account_links_person_status
                ON enterprise_person_account_links (person_id, status);
            CREATE INDEX IF NOT EXISTS idx_enterprise_person_account_links_account_status
                ON enterprise_person_account_links (account_id, status);

            CREATE TABLE IF NOT EXISTS enterprise_person_login_sessions (
                id TEXT PRIMARY KEY,
                person_id TEXT NOT NULL,
                active_account_id TEXT,
                status TEXT NOT NULL,
                person_epoch INTEGER NOT NULL,
                session_epoch INTEGER NOT NULL,
                absolute_expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_seen_at TEXT,
                revoked_at TEXT,
                account_token_version INTEGER NOT NULL DEFAULT 0,
                CHECK (status IN ('active', 'revoked', 'expired')),
                FOREIGN KEY (person_id) REFERENCES enterprise_persons(id),
                FOREIGN KEY (active_account_id) REFERENCES enterprise_users(id)
                    ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_enterprise_person_login_sessions_person_status
                ON enterprise_person_login_sessions (person_id, status);
            CREATE INDEX IF NOT EXISTS idx_enterprise_person_login_sessions_account_status
                ON enterprise_person_login_sessions (active_account_id, status);

            CREATE TABLE IF NOT EXISTS enterprise_person_kb_shares (
                id TEXT PRIMARY KEY,
                kb_id TEXT NOT NULL,
                person_id TEXT NOT NULL,
                owner_account_id TEXT NOT NULL,
                target_account_id TEXT NOT NULL,
                target_tenant_id TEXT,
                role TEXT NOT NULL,
                status TEXT NOT NULL,
                created_by TEXT,
                revoked_by TEXT,
                reason TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                revoked_at TEXT,
                CHECK (status IN ('active', 'revoked')),
                CHECK (role IN ('kb_viewer', 'kb_editor', 'kb_admin')),
                FOREIGN KEY (person_id) REFERENCES enterprise_persons(id),
                FOREIGN KEY (target_account_id) REFERENCES enterprise_users(id)
                    ON DELETE CASCADE,
                UNIQUE (kb_id, target_account_id)
            );

            CREATE INDEX IF NOT EXISTS idx_enterprise_person_kb_shares_kb_status
                ON enterprise_person_kb_shares (kb_id, status);
            CREATE INDEX IF NOT EXISTS idx_enterprise_person_kb_shares_person_status
                ON enterprise_person_kb_shares (person_id, status);
            CREATE INDEX IF NOT EXISTS idx_enterprise_person_kb_shares_target_status
                ON enterprise_person_kb_shares (target_account_id, status);
            CREATE INDEX IF NOT EXISTS idx_enterprise_person_kb_shares_tenant_status
                ON enterprise_person_kb_shares (target_tenant_id, status);
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO metadata_schema(version, applied_at) VALUES (?, ?)",
            (_SCHEMA_VERSION, utc_now_iso()),
        )
        self._ensure_added_columns(conn)
        self._backfill_document_source_keys(conn)
        conn.commit()

    def _ensure_added_columns(self, conn: sqlite3.Connection) -> None:
        """Idempotently add columns introduced after the initial schema.

        ``CREATE TABLE IF NOT EXISTS`` never alters an already-created table, so
        an existing ``metadata.sqlite3`` would miss columns added later. This
        migrates those tables forward; fresh databases already have the column
        from the DDL and skip the ALTER.
        """
        self._migrate_kb_lifecycle_schema(conn)
        additions: dict[str, dict[str, str]] = {
            "enterprise_api_keys": {"expires_at": "TEXT"},
            "enterprise_users": {
                "can_delete_documents": "INTEGER NOT NULL DEFAULT 0",
                "can_use_agent_query": "INTEGER NOT NULL DEFAULT 0",
                # Existing users retain historical download access. Fresh
                # tables declare DEFAULT 0 and explicit records write a value.
                "can_download_files": "INTEGER NOT NULL DEFAULT 1",
            },
            "enterprise_audit_events": {"actor_tenant_id": "TEXT"},
            # account_token_version: snapshot for v2 account-access validation.
            # Sentinel 0 for pre-existing rows; new sessions write the live
            # account token_version. See docs/多账号身份关联与切换执行文档.md 4.5.
            "enterprise_person_login_sessions": {
                "account_token_version": "INTEGER NOT NULL DEFAULT 0",
            },
            "enterprise_chat_sessions": {
                "context_rounds": "INTEGER NOT NULL DEFAULT 1",
            },
            "enterprise_chat_messages": {
                "append_batch_id": "TEXT",
                "project_event_seq": "INTEGER",
                "memory_reference_time": "TEXT",
            },
            "enterprise_chat_memory_episodes": {
                "event_id": "TEXT",
                "generation": "INTEGER",
                "graph_group_id": "TEXT",
                "append_batch_id": "TEXT",
                "project_event_seq": "INTEGER",
            },
            "enterprise_chat_memory_groups": {
                "last_error_at": "TEXT",
                "active_graph_store_fingerprint": "TEXT",
                "desired_graph_store_fingerprint": "TEXT",
            },
            "enterprise_chat_memory_generations": {
                "last_error_at": "TEXT",
                "replay_byte_count": "INTEGER",
                "snapshot_digest": "TEXT",
                "graph_store_fingerprint": "TEXT",
            },
            "enterprise_chat_memory_outbox": {
                "last_error_at": "TEXT",
                "claimed_by": "TEXT",
                "side_effect_state_version": "INTEGER",
                "snapshot_batch_count": "INTEGER",
                "snapshot_message_count": "INTEGER",
                "snapshot_byte_count": "INTEGER",
                "snapshot_digest": "TEXT",
                "graph_store_fingerprint": "TEXT",
            },
        }
        for table, columns in additions.items():
            existing = {
                str(info["name"])
                for info in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for column, ddl in columns.items():
                if column not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

        # Legacy rows used config_fingerprint for both extraction/runtime and
        # physical graph-store identity. Preserve that recoverability while new
        # production writes persist the two identities independently.
        conn.execute(
            """
            UPDATE enterprise_chat_memory_groups
            SET active_graph_store_fingerprint = active_config_fingerprint
            WHERE active_graph_store_fingerprint IS NULL
              AND active_config_fingerprint IS NOT NULL
            """
        )
        conn.execute(
            """
            UPDATE enterprise_chat_memory_groups
            SET desired_graph_store_fingerprint = desired_config_fingerprint
            WHERE desired_graph_store_fingerprint IS NULL
            """
        )
        conn.execute(
            """
            UPDATE enterprise_chat_memory_generations
            SET graph_store_fingerprint = config_fingerprint
            WHERE graph_store_fingerprint IS NULL
            """
        )
        conn.execute(
            """
            UPDATE enterprise_chat_memory_outbox
            SET graph_store_fingerprint = config_fingerprint
            WHERE graph_store_fingerprint IS NULL
            """
        )
        incomplete_graph_identity = conn.execute(
            """
            SELECT 1
            FROM enterprise_chat_memory_groups
            WHERE desired_graph_store_fingerprint IS NULL
               OR (active_generation IS NULL) <>
                  (active_graph_store_fingerprint IS NULL)
            UNION ALL
            SELECT 1 FROM enterprise_chat_memory_generations
            WHERE graph_store_fingerprint IS NULL
            UNION ALL
            SELECT 1 FROM enterprise_chat_memory_outbox
            WHERE graph_store_fingerprint IS NULL
            LIMIT 1
            """
        ).fetchone()
        if incomplete_graph_identity is not None:
            raise RuntimeError(
                "Chat Memory graph-store fingerprint migration incomplete"
            )

        self._repair_enterprise_tenant_memberships(conn)
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_enterprise_tenant_memberships_user
            ON enterprise_tenant_memberships (user_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_enterprise_audit_events_actor_tenant
            ON enterprise_audit_events (actor_tenant_id, created_at DESC, id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_enterprise_chat_messages_memory_replay
            ON enterprise_chat_messages (
                user_id, project_id, project_event_seq, session_id, seq
            )
            WHERE project_event_seq IS NOT NULL
            """
        )
        conn.execute(
            """
            UPDATE enterprise_chat_memory_episodes AS episode
            SET append_batch_id = (
                    SELECT outbox.append_batch_id
                    FROM enterprise_chat_memory_outbox AS outbox
                    WHERE outbox.event_id = episode.event_id
                ),
                project_event_seq = (
                    SELECT outbox.event_seq
                    FROM enterprise_chat_memory_outbox AS outbox
                    WHERE outbox.event_id = episode.event_id
                )
            WHERE episode.append_batch_id IS NULL
              AND episode.project_event_seq IS NULL
              AND EXISTS (
                  SELECT 1 FROM enterprise_chat_memory_outbox AS outbox
                  WHERE outbox.event_id = episode.event_id
                    AND outbox.append_batch_id IS NOT NULL
              )
            """
        )
        conn.execute(
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
        conn.execute(
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
            UPDATE enterprise_chat_memory_episodes
            SET event_id = NULL,
                generation = NULL,
                graph_group_id = NULL,
                append_batch_id = NULL,
                project_event_seq = NULL
            WHERE episode_uuid IN (
                SELECT episode_uuid FROM ranked WHERE duplicate_rank > 1
            )
            """
        )
        self._migrate_chat_memory_episode_identity_schema(conn)
        conn.execute("DROP INDEX IF EXISTS uq_enterprise_chat_memory_episodes_event")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_enterprise_chat_memory_episodes_session
            ON enterprise_chat_memory_episodes (session_id, last_seq)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_enterprise_chat_memory_episodes_project
            ON enterprise_chat_memory_episodes (project_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_enterprise_chat_memory_episodes_user
            ON enterprise_chat_memory_episodes (user_id)
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS
                uq_enterprise_chat_memory_episode_generation_batch
            ON enterprise_chat_memory_episodes (
                user_id, project_id, generation, append_batch_id
            )
            WHERE generation IS NOT NULL AND append_batch_id IS NOT NULL
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_enterprise_chat_memory_episodes_generation
            ON enterprise_chat_memory_episodes (
                user_id, project_id, generation, graph_group_id
            )
            """
        )

    def _migrate_chat_memory_episode_identity_schema(
        self, conn: sqlite3.Connection
    ) -> None:
        table_row = conn.execute(
            """
            SELECT sql FROM sqlite_master
            WHERE type = 'table' AND name = 'enterprise_chat_memory_episodes'
            """
        ).fetchone()
        if table_row is None:
            return
        table_sql = str(table_row["sql"] or "").lower()
        if (
            "enterprise_chat_memory_episode_identity_v2_check" in table_sql
            and "is true" in table_sql
        ):
            return

        conn.execute(
            "DROP TABLE IF EXISTS enterprise_chat_memory_episodes_identity_migrated"
        )
        conn.execute(
            """
            CREATE TABLE enterprise_chat_memory_episodes_identity_migrated (
                episode_uuid TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                first_seq INTEGER NOT NULL,
                last_seq INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                event_id TEXT,
                generation INTEGER,
                graph_group_id TEXT,
                append_batch_id TEXT,
                project_event_seq INTEGER,
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
            )
            """
        )
        conn.execute(
            """
            INSERT INTO enterprise_chat_memory_episodes_identity_migrated (
                episode_uuid, session_id, project_id, user_id, first_seq,
                last_seq, created_at, event_id, generation, graph_group_id,
                append_batch_id, project_event_seq
            )
            SELECT episode_uuid, session_id, project_id, user_id, first_seq,
                   last_seq, created_at, event_id, generation, graph_group_id,
                   append_batch_id, project_event_seq
            FROM enterprise_chat_memory_episodes
            """
        )
        conn.execute("DROP TABLE enterprise_chat_memory_episodes")
        conn.execute(
            """
            ALTER TABLE enterprise_chat_memory_episodes_identity_migrated
            RENAME TO enterprise_chat_memory_episodes
            """
        )

    def _migrate_kb_lifecycle_schema(self, conn: sqlite3.Connection) -> None:
        """Rebuild the lifecycle table when the old two-state CHECK is present."""

        table_row = conn.execute(
            """
            SELECT sql FROM sqlite_master
            WHERE type = 'table' AND name = 'enterprise_kb_lifecycle'
            """
        ).fetchone()
        if table_row is None:
            return
        table_sql = str(table_row["sql"] or "").lower()
        columns = {
            str(info["name"])
            for info in conn.execute(
                "PRAGMA table_info(enterprise_kb_lifecycle)"
            ).fetchall()
        }
        if "delete_job_id" in columns and "'deleting'" in table_sql:
            return

        delete_job_projection = (
            "delete_job_id" if "delete_job_id" in columns else "NULL"
        )
        conn.execute("DROP TABLE IF EXISTS enterprise_kb_lifecycle_migrated")
        conn.execute(
            """
            CREATE TABLE enterprise_kb_lifecycle_migrated (
                kb_id TEXT PRIMARY KEY,
                generation TEXT NOT NULL,
                state TEXT NOT NULL CHECK (
                    state IN ('active', 'deleting', 'deleted')
                ),
                activated_at TEXT NOT NULL,
                deleted_at TEXT,
                updated_at TEXT NOT NULL,
                delete_job_id TEXT,
                CHECK (kb_id <> '' AND kb_id = trim(kb_id)),
                CHECK (generation <> '' AND generation = trim(generation)),
                CHECK (delete_job_id IS NULL OR (
                    delete_job_id <> '' AND delete_job_id = trim(delete_job_id)
                )),
                CHECK (
                    (state = 'active' AND deleted_at IS NULL AND delete_job_id IS NULL)
                    OR (
                        state = 'deleting' AND deleted_at IS NULL
                        AND delete_job_id IS NOT NULL
                    )
                    OR (state = 'deleted' AND deleted_at IS NOT NULL)
                )
            )
            """
        )
        conn.execute(
            f"""
            INSERT INTO enterprise_kb_lifecycle_migrated (
                kb_id, generation, state, activated_at, deleted_at, updated_at,
                delete_job_id
            )
            SELECT kb_id, generation, state, activated_at, deleted_at, updated_at,
                   {delete_job_projection}
            FROM enterprise_kb_lifecycle
            """
        )
        conn.execute("DROP TABLE enterprise_kb_lifecycle")
        conn.execute(
            "ALTER TABLE enterprise_kb_lifecycle_migrated "
            "RENAME TO enterprise_kb_lifecycle"
        )

    def _repair_enterprise_tenant_memberships(
        self, conn: sqlite3.Connection
    ) -> None:
        """Reconcile legacy memberships to enterprise_users.tenant_id."""
        conn.execute(
            """
            DELETE FROM enterprise_tenant_memberships
            WHERE NOT EXISTS (
                SELECT 1 FROM enterprise_users u
                WHERE u.id = enterprise_tenant_memberships.user_id
                  AND u.tenant_id IS NOT NULL
                  AND u.tenant_id = enterprise_tenant_memberships.tenant_id
            )
            """
        )
        conn.execute(
            """
            INSERT INTO enterprise_tenant_memberships (
                tenant_id, user_id, role, granted_by, created_at, updated_at
            )
            SELECT u.tenant_id, u.id, 'tenant_member', NULL,
                   u.updated_at, u.updated_at
            FROM enterprise_users u
            WHERE u.tenant_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM enterprise_tenant_memberships m
                  WHERE m.user_id = u.id AND m.tenant_id = u.tenant_id
              )
            """
        )
        conn.execute(
            """
            DELETE FROM enterprise_tenant_user_kb_overrides
            WHERE NOT EXISTS (
                SELECT 1
                FROM enterprise_users u
                JOIN enterprise_tenant_memberships m
                  ON m.user_id = u.id AND m.tenant_id = u.tenant_id
                WHERE u.id = enterprise_tenant_user_kb_overrides.user_id
                  AND u.tenant_id = enterprise_tenant_user_kb_overrides.tenant_id
            )
            """
        )

    def _backfill_document_source_keys(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            """
            SELECT id, kb_id, metadata_json, created_at, updated_at
            FROM documents
            WHERE deleted_at IS NULL
            ORDER BY updated_at DESC, created_at DESC, id DESC
            """
        ).fetchall()
        for row in rows:
            source_key = _metadata_source_key(_loads_json_object(row["metadata_json"]))
            if source_key is None:
                continue
            conn.execute(
                """
                INSERT OR IGNORE INTO document_source_keys (
                    kb_id, source_key, document_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    row["kb_id"],
                    source_key,
                    row["id"],
                    row["created_at"],
                    row["updated_at"],
                ),
            )

    def _sync_document_source_key(
        self,
        conn: sqlite3.Connection,
        *,
        kb_id: str,
        document_id: str,
        source_key: str | None,
        timestamp: str,
    ) -> None:
        if source_key is None:
            conn.execute(
                "DELETE FROM document_source_keys WHERE kb_id = ? AND document_id = ?",
                (kb_id, document_id),
            )
            return
        existing = conn.execute(
            """
            SELECT document_id
            FROM document_source_keys
            WHERE kb_id = ? AND source_key = ?
            """,
            (kb_id, source_key),
        ).fetchone()
        if existing is not None and existing["document_id"] != document_id:
            raise DuplicateDocumentSourceKeyError(
                kb_id, source_key, str(existing["document_id"])
            )
        conn.execute(
            """
            DELETE FROM document_source_keys
            WHERE kb_id = ? AND document_id = ? AND source_key <> ?
            """,
            (kb_id, document_id, source_key),
        )
        conn.execute(
            """
            INSERT INTO document_source_keys (
                kb_id, source_key, document_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(kb_id, source_key) DO UPDATE SET updated_at = excluded.updated_at
            """,
            (kb_id, source_key, document_id, timestamp, timestamp),
        )

    def _insert_document(
        self, conn: sqlite3.Connection, document: DocumentRecord
    ) -> DocumentRecord:
        conn.execute(
            """
            INSERT INTO documents (
                id, kb_id, workspace, lightrag_doc_id, source_type, source_name,
                source_uri, source_hash, content_type, size_bytes, parser_hash,
                index_hash, status, enabled, archived, chunks_count, entity_count,
                relation_count, error_code, error_message, metadata_json,
                created_at, updated_at, deleted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document.id,
                document.kb_id,
                document.workspace,
                document.lightrag_doc_id,
                document.source_type,
                document.source_name,
                document.source_uri,
                document.source_hash,
                document.content_type,
                document.size_bytes,
                document.parser_hash,
                document.index_hash,
                document.status,
                int(document.enabled),
                int(document.archived),
                document.chunks_count,
                document.entity_count,
                document.relation_count,
                document.error_code,
                document.error_message,
                _dumps_json(document.metadata),
                document.created_at,
                document.updated_at,
                document.deleted_at,
            ),
        )
        source_key = _metadata_source_key(document.metadata)
        self._sync_document_source_key(
            conn,
            kb_id=document.kb_id,
            document_id=document.id,
            source_key=source_key,
            timestamp=document.created_at,
        )
        return document

    def _insert_job(self, conn: sqlite3.Connection, job: JobRecord) -> JobRecord:
        conn.execute(
            """
            INSERT INTO jobs (
                id, kb_id, workspace, batch_id, document_id, job_type, status,
                stage, progress, total_items, completed_items, failed_items,
                idempotency_key, config_version_id, config_hash, retry_count,
                max_retries, payload_json, result_json, error_code, error_message,
                created_at, updated_at, queued_at, started_at, finished_at,
                cancelled_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job.id,
                job.kb_id,
                job.workspace,
                job.batch_id,
                job.document_id,
                job.job_type,
                job.status,
                job.stage,
                job.progress,
                job.total_items,
                job.completed_items,
                job.failed_items,
                job.idempotency_key,
                job.config_version_id,
                job.config_hash,
                job.retry_count,
                job.max_retries,
                _dumps_json(job.payload),
                _dumps_json(job.result) if job.result is not None else None,
                job.error_code,
                job.error_message,
                job.created_at,
                job.updated_at,
                job.queued_at,
                job.started_at,
                job.finished_at,
                job.cancelled_at,
            ),
        )
        return job

    def _insert_artifact(
        self, conn: sqlite3.Connection, artifact: ArtifactRecord
    ) -> ArtifactRecord:
        conn.execute(
            """
            INSERT INTO document_artifacts (
                id, kb_id, workspace, document_id, artifact_type, uri, checksum,
                size_bytes, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact.id,
                artifact.kb_id,
                artifact.workspace,
                artifact.document_id,
                artifact.artifact_type,
                artifact.uri,
                artifact.checksum,
                artifact.size_bytes,
                _dumps_json(artifact.metadata),
                artifact.created_at,
            ),
        )
        return artifact

    def _get_job_by_idempotency_key(
        self,
        conn: sqlite3.Connection,
        kb_id: str,
        idempotency_key: str | None,
        *,
        job_type: str | None = None,
    ) -> JobRecord | None:
        if not idempotency_key:
            return None
        where = "kb_id = ? AND idempotency_key = ?"
        params: list[Any] = [kb_id, idempotency_key]
        if job_type is not None:
            where += " AND job_type = ?"
            params.append(job_type)
        row = conn.execute(
            f"""
            SELECT * FROM jobs
            WHERE {where}
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            params,
        ).fetchone()
        return JobRecord.from_row(row) if row is not None else None

    def _validate_idempotent_job(
        self, existing: JobRecord, candidate: JobRecord
    ) -> None:
        existing_fingerprint = existing.payload.get("idempotency_fingerprint")
        candidate_fingerprint = candidate.payload.get("idempotency_fingerprint")
        if existing_fingerprint != candidate_fingerprint:
            raise IdempotencyKeyConflictError(candidate.idempotency_key or "")

    def _documents_for_job(
        self, conn: sqlite3.Connection, job: JobRecord
    ) -> list[DocumentRecord]:
        document_ids = job.payload.get("document_ids")
        if not isinstance(document_ids, list) or not all(
            isinstance(document_id, str) for document_id in document_ids
        ):
            if not job.batch_id:
                return []
            rows = conn.execute(
                """
                SELECT * FROM documents
                WHERE kb_id = ? AND deleted_at IS NULL
                ORDER BY created_at ASC, id ASC
                """,
                (job.kb_id,),
            ).fetchall()
            return [
                DocumentRecord.from_row(row)
                for row in rows
                if _loads_json_object(row["metadata_json"]).get("batch_id")
                == job.batch_id
            ]
        if not document_ids:
            return []
        placeholders = ", ".join("?" for _ in document_ids)
        rows = conn.execute(
            f"""
            SELECT * FROM documents
            WHERE kb_id = ? AND id IN ({placeholders}) AND deleted_at IS NULL
            """,
            [job.kb_id, *document_ids],
        ).fetchall()
        documents_by_id = {row["id"]: DocumentRecord.from_row(row) for row in rows}
        return [
            documents_by_id[document_id]
            for document_id in document_ids
            if document_id in documents_by_id
        ]

    def _claim_document_parse_queued(
        self,
        conn: sqlite3.Connection,
        kb_id: str,
        document_id: str,
        *,
        metadata_patch: dict[str, Any],
        raise_on_active: bool,
    ) -> DocumentRecord:
        current_row = conn.execute(
            """
            SELECT * FROM documents
            WHERE kb_id = ? AND id = ? AND deleted_at IS NULL
            """,
            (kb_id, document_id),
        ).fetchone()
        if current_row is None:
            raise MetadataRecordNotFoundError(f"Document '{document_id}' not found")
        if raise_on_active and current_row["status"] in {"parse_queued", "parsing"}:
            raise ActiveDocumentParseJobError(
                document_id,
                _active_parse_job_id_from_row(current_row),
            )
        if current_row["status"] in {"build_queued", "building"}:
            raise ActiveDocumentBuildJobError(
                document_id,
                _active_build_job_id_from_row(current_row),
            )
        if current_row["status"] == "deleting":
            raise ActiveDocumentDeleteJobError(
                document_id,
                _active_delete_job_id_from_row(current_row),
            )
        if current_row["status"] == "replacing":
            raise ActiveDocumentReplaceJobError(
                document_id,
                _active_replace_job_id_from_row(current_row),
            )
        return self._update_document_parse_state(
            conn,
            kb_id,
            document_id,
            status="parse_queued",
            metadata_patch=metadata_patch,
            clear_error=True,
        )

    def _claim_document_build_queued(
        self,
        conn: sqlite3.Connection,
        kb_id: str,
        document_id: str,
        *,
        metadata_patch: dict[str, Any],
        require_parsed: bool,
    ) -> DocumentRecord:
        current_row = conn.execute(
            """
            SELECT * FROM documents
            WHERE kb_id = ? AND id = ? AND deleted_at IS NULL
            """,
            (kb_id, document_id),
        ).fetchone()
        if current_row is None:
            raise MetadataRecordNotFoundError(f"Document '{document_id}' not found")
        status = current_row["status"]
        if status in {"build_queued", "building"}:
            raise ActiveDocumentBuildJobError(
                document_id,
                _active_build_job_id_from_row(current_row),
            )
        if status == "deleting":
            raise ActiveDocumentDeleteJobError(
                document_id,
                _active_delete_job_id_from_row(current_row),
            )
        if status == "replacing":
            raise ActiveDocumentReplaceJobError(
                document_id,
                _active_replace_job_id_from_row(current_row),
            )
        if require_parsed and status not in {"parsed", "ready", "build_failed"}:
            raise DocumentNotParsedError(document_id, str(status))
        return self._update_document_parse_state(
            conn,
            kb_id,
            document_id,
            status="build_queued",
            metadata_patch=metadata_patch,
            clear_error=True,
        )

    def _claim_document_deleting(
        self,
        conn: sqlite3.Connection,
        kb_id: str,
        document_id: str,
        *,
        metadata_patch: dict[str, Any],
    ) -> DocumentRecord:
        current_row = conn.execute(
            """
            SELECT * FROM documents
            WHERE kb_id = ? AND id = ? AND deleted_at IS NULL
            """,
            (kb_id, document_id),
        ).fetchone()
        if current_row is None:
            raise MetadataRecordNotFoundError(f"Document '{document_id}' not found")
        status = str(current_row["status"])
        if status in {"parse_queued", "parsing"}:
            raise ActiveDocumentParseJobError(
                document_id,
                _active_parse_job_id_from_row(current_row),
            )
        if status in {"build_queued", "building"}:
            raise ActiveDocumentBuildJobError(
                document_id,
                _active_build_job_id_from_row(current_row),
            )
        if status == "deleting":
            existing_job_id = _active_delete_job_id_from_row(current_row)
            requested_job_id = metadata_patch.get("pending_delete_job_id") or metadata_patch.get(
                "current_delete_job_id"
            )
            if requested_job_id is not None and str(requested_job_id) == existing_job_id:
                return self._update_document_parse_state(
                    conn,
                    kb_id,
                    document_id,
                    status="deleting",
                    metadata_patch=metadata_patch,
                    clear_error=True,
                )
            raise ActiveDocumentDeleteJobError(
                document_id,
                existing_job_id,
            )
        if status == "replacing":
            raise ActiveDocumentReplaceJobError(
                document_id,
                _active_replace_job_id_from_row(current_row),
            )
        return self._update_document_parse_state(
            conn,
            kb_id,
            document_id,
            status="deleting",
            metadata_patch=metadata_patch,
            clear_error=True,
        )

    def _claim_document_replacing(
        self,
        conn: sqlite3.Connection,
        kb_id: str,
        document_id: str,
        *,
        metadata_patch: dict[str, Any],
    ) -> DocumentRecord:
        current_row = conn.execute(
            """
            SELECT * FROM documents
            WHERE kb_id = ? AND id = ? AND deleted_at IS NULL
            """,
            (kb_id, document_id),
        ).fetchone()
        if current_row is None:
            raise MetadataRecordNotFoundError(f"Document '{document_id}' not found")
        status = str(current_row["status"])
        if status in {"parse_queued", "parsing"}:
            raise ActiveDocumentParseJobError(
                document_id,
                _active_parse_job_id_from_row(current_row),
            )
        if status in {"build_queued", "building"}:
            raise ActiveDocumentBuildJobError(
                document_id,
                _active_build_job_id_from_row(current_row),
            )
        if status == "deleting":
            raise ActiveDocumentDeleteJobError(
                document_id,
                _active_delete_job_id_from_row(current_row),
            )
        if status == "replacing":
            raise ActiveDocumentReplaceJobError(
                document_id,
                _active_replace_job_id_from_row(current_row),
            )
        return self._update_document_parse_state(
            conn,
            kb_id,
            document_id,
            status="replacing",
            metadata_patch=metadata_patch,
            clear_error=True,
        )

    def _update_document_parse_state(
        self,
        conn: sqlite3.Connection,
        kb_id: str,
        document_id: str,
        *,
        status: DocumentStatus,
        metadata_patch: dict[str, Any],
        parser_hash: str | None = None,
        lightrag_doc_id: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        clear_error: bool = False,
        clear_lightrag_doc_id: bool = False,
        clear_index_state: bool = False,
    ) -> DocumentRecord:
        current_row = conn.execute(
            """
            SELECT * FROM documents
            WHERE kb_id = ? AND id = ? AND deleted_at IS NULL
            """,
            (kb_id, document_id),
        ).fetchone()
        if current_row is None:
            raise MetadataRecordNotFoundError(f"Document '{document_id}' not found")

        metadata = _loads_json_object(current_row["metadata_json"])
        metadata.update(metadata_patch)
        now = utc_now_iso()
        if "source_key" in metadata_patch:
            self._sync_document_source_key(
                conn,
                kb_id=kb_id,
                document_id=document_id,
                source_key=_metadata_source_key(metadata),
                timestamp=now,
            )
        next_parser_hash = (
            parser_hash if parser_hash is not None else current_row["parser_hash"]
        )
        next_lightrag_doc_id = (
            None
            if clear_lightrag_doc_id
            else lightrag_doc_id
            if lightrag_doc_id is not None
            else current_row["lightrag_doc_id"]
        )
        next_index_hash = None if clear_index_state else current_row["index_hash"]
        next_chunks_count = None if clear_index_state else current_row["chunks_count"]
        next_entity_count = None if clear_index_state else current_row["entity_count"]
        next_relation_count = (
            None if clear_index_state else current_row["relation_count"]
        )
        next_error_code = None if clear_error else error_code
        next_error_message = None if clear_error else error_message
        conn.execute(
            """
            UPDATE documents
            SET status = ?, parser_hash = ?, lightrag_doc_id = ?, index_hash = ?,
                chunks_count = ?, entity_count = ?, relation_count = ?, error_code = ?,
                error_message = ?, metadata_json = ?, updated_at = ?
            WHERE kb_id = ? AND id = ?
            """,
            (
                status,
                next_parser_hash,
                next_lightrag_doc_id,
                next_index_hash,
                next_chunks_count,
                next_entity_count,
                next_relation_count,
                next_error_code,
                next_error_message,
                _dumps_json(metadata),
                now,
                kb_id,
                document_id,
            ),
        )
        row = conn.execute(
            "SELECT * FROM documents WHERE kb_id = ? AND id = ?",
            (kb_id, document_id),
        ).fetchone()
        if row is None:
            raise MetadataRecordNotFoundError(f"Document '{document_id}' not found")
        return DocumentRecord.from_row(row)


def _allowed_next_job_statuses(current: str) -> set[str]:
    transitions = {
        "queued": {"running", "cancelling", "cancelled", "failed"},
        "running": {"succeeded", "failed", "cancelling"},
        "cancelling": {"cancelled", "failed"},
        "retrying": {"queued", "running", "failed"},
        "succeeded": set(),
        "failed": {"retrying", "queued"},
        "cancelled": {"retrying", "queued"},
    }
    return transitions.get(current, set())


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


_REPLACE_DERIVED_METADATA_KEYS = {
    "artifact_count",
    "auto_index",
    "auto_parse",
    "blocks_path",
    "build_skipped",
    "build_skip_reason",
    "build_started_at",
    "current_build_job_id",
    "current_parse_job_id",
    "current_replace_job_id",
    "force_embedding",
    "force_extract",
    "force_rechunk",
    "force_reparse",
    "last_built_at",
    "last_build_job_id",
    "last_failed_build_job_id",
    "last_failed_parse_job_id",
    "last_failed_parser_hash",
    "last_parse_job_id",
    "last_parsed_at",
    "parse_engine",
    "parse_format",
    "parse_stage_skipped",
    "parse_started_at",
    "parser_engine",
    "pending_build_job_id",
    "pending_index_hash",
    "pending_lightrag_doc_id",
    "pending_parse_batch_id",
    "pending_parse_job_id",
    "pending_parser_hash",
    "pending_replace_job_id",
    "process_options",
}


def _active_parse_job_id_from_row(row: sqlite3.Row) -> str:
    metadata = _loads_json_object(row["metadata_json"])
    if row["status"] == "parse_queued":
        job_id = metadata.get("pending_parse_job_id")
        return str(job_id) if job_id else "unknown"
    if row["status"] == "parsing":
        job_id = metadata.get("current_parse_job_id")
        return str(job_id) if job_id else "unknown"
    return "unknown"


def _active_build_job_id_from_row(row: sqlite3.Row) -> str:
    metadata = _loads_json_object(row["metadata_json"])
    if row["status"] == "build_queued":
        job_id = metadata.get("pending_build_job_id")
        return str(job_id) if job_id else "unknown"
    if row["status"] == "building":
        job_id = metadata.get("current_build_job_id")
        return str(job_id) if job_id else "unknown"
    return "unknown"


def _active_delete_job_id_from_row(row: sqlite3.Row) -> str:
    metadata = _loads_json_object(row["metadata_json"])
    job_id = metadata.get("pending_delete_job_id") or metadata.get(
        "current_delete_job_id"
    )
    return str(job_id) if job_id else "unknown"


def _active_replace_job_id_from_row(row: sqlite3.Row) -> str:
    metadata = _loads_json_object(row["metadata_json"])
    job_id = metadata.get("pending_replace_job_id") or metadata.get(
        "current_replace_job_id"
    )
    return str(job_id) if job_id else "unknown"
