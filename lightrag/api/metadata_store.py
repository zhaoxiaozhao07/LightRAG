from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import threading
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
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

_SCHEMA_VERSION = 3
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

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ChatMessageRecord":
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
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChatMemoryEpisodeRecord:
    """Mapping between a graphiti memory episode and the chat messages it
    distilled (docs/ChatMemory-zh.md).

    ``first_seq``/``last_seq`` bound the per-session message range covered by
    the episode; ``MAX(last_seq)`` per session is the ingestion watermark used
    for idempotent ingestion and restart compensation. ``noop_*`` uuids mark
    ranges that produced no graphiti episode (e.g. blank content) but still
    advance the watermark.
    """

    episode_uuid: str
    session_id: str
    project_id: str
    user_id: str
    first_seq: int
    last_seq: int
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ChatMemoryEpisodeRecord":
        return cls(
            episode_uuid=str(row["episode_uuid"]),
            session_id=str(row["session_id"]),
            project_id=str(row["project_id"]),
            user_id=str(row["user_id"]),
            first_seq=int(row["first_seq"]),
            last_seq=int(row["last_seq"]),
            created_at=str(row["created_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def _dumps_json(value: dict[str, Any] | None) -> str:
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True)


def _loads_json_object(value: str | bytes | None) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise MetadataStoreError("Metadata JSON must be an object")
    return loaded


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
                conn.execute(
                    """
                    INSERT INTO enterprise_chat_messages (
                        id, session_id, project_id, user_id, role, content,
                        metadata_json, seq, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                INSERT OR REPLACE INTO enterprise_chat_memory_episodes (
                    episode_uuid, session_id, project_id, user_id,
                    first_seq, last_seq, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.episode_uuid,
                    record.session_id,
                    record.project_id,
                    record.user_id,
                    record.first_seq,
                    record.last_seq,
                    record.created_at,
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
                first_seq INTEGER NOT NULL,
                last_seq INTEGER NOT NULL,
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
            "enterprise_chat_sessions": {
                "context_rounds": "INTEGER NOT NULL DEFAULT 1",
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
