from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from lightrag.api.enterprise_auth import (
    enforce_concurrent_job_quota,
    enterprise_auth_enabled,
    get_current_principal,
    principal_job_subject,
)
from lightrag.api.kb_service import (
    KnowledgeBaseConflictError,
    KnowledgeBaseRecord,
    KnowledgeBaseService,
    utc_now_iso,
)
from lightrag.api.metadata_store import (
    JobRecord,
    MetadataJobStatus,
    SQLiteMetadataStore,
)
from lightrag.api.postgres_metadata_store import PostgresMetadataStore
from lightrag.utils import generate_track_id

_RUNNING_JOB_STATUSES = ("queued", "running", "retrying", "cancelling")
MetadataStore = SQLiteMetadataStore | PostgresMetadataStore

# Every production executor except ``clear_kb`` mutates KB-scoped state and
# must run under the shared generation fence. ``clear_kb`` deliberately takes
# the exclusive fence inside KBDeletionService instead.
RESUMABLE_KB_MUTATION_JOB_TYPES: frozenset[str] = frozenset(
    {"parse", "build_kg", "reindex", "delete", "replace", "sync", "agent_profile"}
)
_GENERATION_STAMP_EXEMPT_JOB_TYPES: frozenset[str] = frozenset({"clear_kb"})


def _assert_active_catalog_generation(
    record: KnowledgeBaseRecord,
    expected_generation: str,
) -> None:
    if record.generation != expected_generation:
        raise KnowledgeBaseConflictError(
            f"Knowledge base '{record.id}' changed generation"
        )
    if record.status != "active":
        raise KnowledgeBaseConflictError(
            f"Knowledge base '{record.id}' is not active"
        )


async def assert_active_kb_generation(
    kb_service: KnowledgeBaseService,
    metadata_store: MetadataStore,
    kb_id: str,
    expected_generation: str,
) -> KnowledgeBaseRecord:
    """Assert one catalog generation is still active in both control planes."""

    record = await kb_service.get(kb_id, include_deleted=True)
    # Check lifecycle first so an in-progress hard delete surfaces the metadata
    # conflict that owns admission, rather than being flattened into a catalog
    # status error.
    await metadata_store.assert_kb_generation(record.id, expected_generation)
    _assert_active_catalog_generation(record, expected_generation)
    return record


async def prepare_kb_job_payload(
    kb_service: KnowledgeBaseService,
    metadata_store: MetadataStore,
    kb_id: str,
    *,
    job_type: str,
    payload: dict[str, Any] | None,
) -> tuple[KnowledgeBaseRecord, dict[str, Any]]:
    """Capture and stamp the catalog generation for a newly-created KB job."""

    job_payload = dict(payload or {})
    if job_type in _GENERATION_STAMP_EXEMPT_JOB_TYPES:
        return await kb_service.get(kb_id, include_deleted=True), job_payload

    record = await kb_service.get(kb_id, include_deleted=True)
    supplied_generation = job_payload.get("kb_generation")
    if supplied_generation is not None and supplied_generation != record.generation:
        raise KnowledgeBaseConflictError(
            f"Knowledge base '{record.id}' changed generation"
        )
    record = await assert_active_kb_generation(
        kb_service,
        metadata_store,
        record.id,
        record.generation,
    )
    job_payload["kb_generation"] = record.generation
    return record, job_payload


class JobService:
    def __init__(
        self,
        kb_service: KnowledgeBaseService,
        metadata_store: MetadataStore,
    ):
        self._kb_service = kb_service
        self._metadata_store = metadata_store

    async def _persist_job(self, job: JobRecord) -> JobRecord:
        await self._apply_enterprise_job_controls(job)
        async with self._job_persistence_guard(job):
            return await self._metadata_store.create_job(job)

    async def _persist_job_once(self, job: JobRecord) -> tuple[JobRecord, bool]:
        await self._apply_enterprise_job_controls(job)
        async with self._job_persistence_guard(job):
            return await self._metadata_store.create_job_once(job)

    @asynccontextmanager
    async def _job_persistence_guard(
        self,
        job: JobRecord,
    ) -> AsyncIterator[None]:
        """Fence the final persistence of every non-clear KB mutation job."""

        if job.job_type in _GENERATION_STAMP_EXEMPT_JOB_TYPES:
            yield
            return

        generation = job.payload.get("kb_generation")
        if not isinstance(generation, str) or not generation.strip():
            raise KnowledgeBaseConflictError(
                f"Job '{job.id}' is missing its KB generation stamp"
            )
        async with self.kb_write_guard(job.kb_id, generation) as record:
            self._assert_guarded_job_write_admitted(job, record, generation)
            yield

    async def _prepare_new_job(
        self,
        kb_id: str,
        *,
        job_type: str,
        payload: dict[str, Any] | None,
    ) -> tuple[KnowledgeBaseRecord, dict[str, Any]]:
        return await prepare_kb_job_payload(
            self._kb_service,
            self._metadata_store,
            kb_id,
            job_type=job_type,
            payload=payload,
        )

    @staticmethod
    def _assert_guarded_job_write_admitted(
        job: JobRecord,
        record: KnowledgeBaseRecord,
        generation: str,
    ) -> None:
        if record.id != job.kb_id:
            raise KnowledgeBaseConflictError(
                f"Knowledge base '{job.kb_id}' changed identity"
            )
        _assert_active_catalog_generation(record, generation)
        if record.workspace != job.workspace:
            raise KnowledgeBaseConflictError(
                f"Knowledge base '{record.id}' changed workspace"
            )

    @asynccontextmanager
    async def job_execution_guard(
        self,
        job_id: str,
        *,
        wait: bool = True,
    ) -> AsyncIterator[bool]:
        """Hold cross-process execution ownership for one durable job."""

        async with self._metadata_store.job_execution_guard(
            job_id, wait=wait
        ) as acquired:
            yield acquired

    @asynccontextmanager
    async def kb_write_guard(
        self,
        kb_id: str,
        expected_generation: str,
    ) -> AsyncIterator[KnowledgeBaseRecord]:
        """Expose the stable shared fence used by durable/background executors.

        The underlying SQLite file lock or PostgreSQL session advisory lock is
        held for the complete caller context. It relies on standard crash-stop
        cleanup by the OS/database; it is not a persistent run-token lease.
        """

        async with self._metadata_store.kb_write_guard(kb_id, expected_generation):
            # The metadata guard has already asserted lifecycle state while
            # holding its own connection/lock. Repeating that assertion here
            # could deadlock a PostgreSQL pool configured with one connection.
            record = await self._kb_service.get(kb_id, include_deleted=True)
            _assert_active_catalog_generation(record, expected_generation)
            yield record

    async def _apply_enterprise_job_controls(self, job: JobRecord) -> None:
        """Enforce the per-principal/tenant concurrent-job quota and stamp the
        creating principal into the job payload so in-flight jobs can be
        attributed and counted. No-op outside enterprise mode or when no
        principal is bound to the current async context (e.g. durable worker
        re-runs of already-counted jobs)."""
        if not enterprise_auth_enabled():
            return
        principal = get_current_principal()
        if principal is None:
            return
        await enforce_concurrent_job_quota(self._metadata_store, principal)
        job.payload["_principal"] = principal_job_subject(principal)

    async def create_job(
        self,
        kb_id: str,
        *,
        job_type: str,
        document_id: str | None = None,
        batch_id: str | None = None,
        stage: str | None = None,
        total_items: int = 1,
        payload: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> JobRecord:
        record, job_payload = await self._prepare_new_job(
            kb_id,
            job_type=job_type,
            payload=payload,
        )
        now = utc_now_iso()
        job = JobRecord(
            id=generate_track_id(f"job_{job_type}"),
            kb_id=record.id,
            workspace=record.workspace,
            batch_id=batch_id,
            document_id=document_id,
            job_type=job_type,
            status="queued",
            stage=stage,
            progress=0.0,
            total_items=total_items,
            completed_items=0,
            failed_items=0,
            idempotency_key=idempotency_key,
            config_version_id=record.active_config_version_id,
            config_hash=None,
            retry_count=0,
            max_retries=3,
            payload=job_payload,
            result=None,
            error_code=None,
            error_message=None,
            created_at=now,
            updated_at=now,
            queued_at=now,
            started_at=None,
            finished_at=None,
            cancelled_at=None,
        )
        return await self._persist_job(job)

    async def create_job_once(
        self,
        kb_id: str,
        *,
        job_type: str,
        document_id: str | None = None,
        batch_id: str | None = None,
        stage: str | None = None,
        total_items: int = 1,
        payload: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> tuple[JobRecord, bool]:
        record, job_payload = await self._prepare_new_job(
            kb_id,
            job_type=job_type,
            payload=payload,
        )
        now = utc_now_iso()
        job = JobRecord(
            id=generate_track_id(f"job_{job_type}"),
            kb_id=record.id,
            workspace=record.workspace,
            batch_id=batch_id,
            document_id=document_id,
            job_type=job_type,
            status="queued",
            stage=stage,
            progress=0.0,
            total_items=total_items,
            completed_items=0,
            failed_items=0,
            idempotency_key=idempotency_key,
            config_version_id=record.active_config_version_id,
            config_hash=None,
            retry_count=0,
            max_retries=3,
            payload=job_payload,
            result=None,
            error_code=None,
            error_message=None,
            created_at=now,
            updated_at=now,
            queued_at=now,
            started_at=None,
            finished_at=None,
            cancelled_at=None,
        )
        return await self._persist_job_once(job)

    async def create_parse_job(
        self,
        kb_id: str,
        *,
        document_id: str,
        parser_hash: str,
        lightrag_doc_id: str,
        parser_engine: str,
        process_options: str,
        source_hash: str,
        source_name: str | None = None,
        source_object_uri: str | None = None,
        raw_object_refs: Sequence[dict[str, Any]] | None = None,
        source_uri: str | None = None,
        force_reparse: bool = False,
        auto_index: bool = False,
        idempotency_key: str | None = None,
    ) -> JobRecord:
        resolved_source_name = source_name or (
            Path(source_uri).name if source_uri else None
        )
        payload = {
            "document_id": document_id,
            "source_name": resolved_source_name,
            "source_object_uri": source_object_uri,
            "raw_object_refs": list(raw_object_refs or ()),
            "source_hash": source_hash,
            "parser_engine": parser_engine,
            "process_options": process_options,
            "parser_hash": parser_hash,
            "lightrag_doc_id": lightrag_doc_id,
            "force_reparse": force_reparse,
            "auto_index": auto_index,
        }
        payload["idempotency_fingerprint"] = _idempotency_fingerprint(payload)
        record, payload = await self._prepare_new_job(
            kb_id,
            job_type="parse",
            payload=payload,
        )
        now = utc_now_iso()
        job = JobRecord(
            id=generate_track_id("job_parse"),
            kb_id=record.id,
            workspace=record.workspace,
            batch_id=None,
            document_id=document_id,
            job_type="parse",
            status="queued",
            stage="parsing",
            progress=0.0,
            total_items=1,
            completed_items=0,
            failed_items=0,
            idempotency_key=idempotency_key,
            config_version_id=record.active_config_version_id,
            config_hash=parser_hash,
            retry_count=0,
            max_retries=3,
            payload=payload,
            result=None,
            error_code=None,
            error_message=None,
            created_at=now,
            updated_at=now,
            queued_at=now,
            started_at=None,
            finished_at=None,
            cancelled_at=None,
        )
        return await self._persist_job(job)

    async def create_parse_job_once(
        self,
        kb_id: str,
        *,
        document_id: str,
        parser_hash: str,
        lightrag_doc_id: str,
        parser_engine: str,
        process_options: str,
        source_hash: str,
        source_name: str | None = None,
        source_object_uri: str | None = None,
        raw_object_refs: Sequence[dict[str, Any]] | None = None,
        source_uri: str | None = None,
        force_reparse: bool = False,
        auto_index: bool = False,
        idempotency_key: str | None = None,
    ) -> tuple[JobRecord, bool]:
        resolved_source_name = source_name or (
            Path(source_uri).name if source_uri else None
        )
        payload = {
            "document_id": document_id,
            "source_name": resolved_source_name,
            "source_object_uri": source_object_uri,
            "raw_object_refs": list(raw_object_refs or ()),
            "source_hash": source_hash,
            "parser_engine": parser_engine,
            "process_options": process_options,
            "parser_hash": parser_hash,
            "lightrag_doc_id": lightrag_doc_id,
            "force_reparse": force_reparse,
            "auto_index": auto_index,
        }
        payload["idempotency_fingerprint"] = _idempotency_fingerprint(payload)
        record, payload = await self._prepare_new_job(
            kb_id,
            job_type="parse",
            payload=payload,
        )
        now = utc_now_iso()
        job = JobRecord(
            id=generate_track_id("job_parse"),
            kb_id=record.id,
            workspace=record.workspace,
            batch_id=None,
            document_id=document_id,
            job_type="parse",
            status="queued",
            stage="parsing",
            progress=0.0,
            total_items=1,
            completed_items=0,
            failed_items=0,
            idempotency_key=idempotency_key,
            config_version_id=record.active_config_version_id,
            config_hash=parser_hash,
            retry_count=0,
            max_retries=3,
            payload=payload,
            result=None,
            error_code=None,
            error_message=None,
            created_at=now,
            updated_at=now,
            queued_at=now,
            started_at=None,
            finished_at=None,
            cancelled_at=None,
        )
        return await self._persist_job_once(job)

    async def create_batch_parse_job(
        self,
        kb_id: str,
        *,
        batch_id: str,
        document_ids: Sequence[str],
        total_items: int,
        plan_items: Sequence[dict[str, Any]],
        planning_failures: Sequence[dict[str, Any]],
        force_reparse: bool = False,
        auto_index: bool = False,
        idempotency_key: str | None = None,
    ) -> JobRecord:
        payload = {
            "document_ids": list(document_ids),
            "items": list(plan_items),
            "planning_failures": list(planning_failures),
            "force_reparse": force_reparse,
            "auto_index": auto_index,
        }
        payload["idempotency_fingerprint"] = _idempotency_fingerprint(payload)
        return await self.create_job(
            kb_id,
            job_type="parse",
            batch_id=batch_id,
            document_id=None,
            stage="parsing",
            total_items=total_items,
            payload=payload,
            idempotency_key=idempotency_key,
        )

    async def create_batch_parse_job_once(
        self,
        kb_id: str,
        *,
        batch_id: str,
        document_ids: Sequence[str],
        total_items: int,
        plan_items: Sequence[dict[str, Any]],
        planning_failures: Sequence[dict[str, Any]],
        force_reparse: bool = False,
        auto_index: bool = False,
        idempotency_key: str | None = None,
    ) -> tuple[JobRecord, bool]:
        payload = {
            "document_ids": list(document_ids),
            "items": list(plan_items),
            "planning_failures": list(planning_failures),
            "force_reparse": force_reparse,
            "auto_index": auto_index,
        }
        payload["idempotency_fingerprint"] = _idempotency_fingerprint(payload)
        return await self.create_job_once(
            kb_id,
            job_type="parse",
            batch_id=batch_id,
            document_id=None,
            stage="parsing",
            total_items=total_items,
            payload=payload,
            idempotency_key=idempotency_key,
        )

    async def list_document_ids_for_config_followup(
        self, kb_id: str, *, statuses: Sequence[str]
    ) -> list[str]:
        """Return enabled, non-archived document ids matching config follow-up statuses."""
        record = await self._kb_service.get(kb_id)
        document_ids: list[str] = []
        seen: set[str] = set()
        for status in statuses:
            offset = 0
            while True:
                documents, total = await self._metadata_store.list_documents(
                    record.id,
                    status=status,
                    limit=200,
                    offset=offset,
                )
                for document in documents:
                    if not document.enabled or document.archived:
                        continue
                    if document.id in seen:
                        continue
                    seen.add(document.id)
                    document_ids.append(document.id)
                if not documents or offset + len(documents) >= total:
                    break
                offset += len(documents)
        return document_ids

    async def list_jobs(
        self,
        kb_id: str,
        *,
        statuses: Sequence[str] | None = None,
        document_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
        include_deleted: bool = False,
    ) -> tuple[list[JobRecord], int]:
        record = await self._kb_service.get(kb_id, include_deleted=include_deleted)
        return await self._metadata_store.list_jobs(
            record.id,
            statuses=statuses,
            document_id=document_id,
            limit=limit,
            offset=offset,
        )

    async def list_running_jobs(
        self, kb_id: str, *, limit: int = 20, include_deleted: bool = False
    ) -> list[JobRecord]:
        jobs, _total = await self.list_jobs(
            kb_id,
            statuses=_RUNNING_JOB_STATUSES,
            limit=limit,
            offset=0,
            include_deleted=include_deleted,
        )
        return jobs

    async def list_dead_letter_jobs(
        self, kb_id: str, *, limit: int = 50, offset: int = 0, include_deleted: bool = False
    ) -> tuple[list[JobRecord], int]:
        """List dead-lettered jobs (failed + retries exhausted) for the KB."""
        record = await self._kb_service.get(kb_id, include_deleted=include_deleted)
        return await self._metadata_store.list_dead_letter_jobs(
            record.id, limit=limit, offset=offset
        )

    async def get_job(
        self, kb_id: str, job_id: str, *, include_deleted: bool = False
    ) -> JobRecord:
        record = await self._kb_service.get(kb_id, include_deleted=include_deleted)
        return await self._metadata_store.get_job(record.id, job_id)

    async def get_job_by_idempotency_key(
        self, kb_id: str, idempotency_key: str, *, job_type: str | None = None
    ) -> JobRecord | None:
        record = await self._kb_service.get(kb_id)
        return await self._metadata_store.get_job_by_idempotency_key(
            record.id, idempotency_key, job_type=job_type
        )

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
        # Terminal bookkeeping must remain possible after the catalog has been
        # soft-deleted. In particular, a worker that claims just before the
        # lifecycle switches to ``deleting`` must fail the row closed.
        record = await self._kb_service.get(kb_id, include_deleted=True)
        return await self._metadata_store.transition_job(
            record.id,
            job_id,
            status=status,
            stage=stage,
            progress=progress,
            completed_items=completed_items,
            failed_items=failed_items,
            result=result,
            error_code=error_code,
            error_message=error_message,
        )

    async def update_job_payload_patch(
        self,
        kb_id: str,
        job_id: str,
        *,
        payload_patch: dict[str, Any],
    ) -> JobRecord:
        record = await self._kb_service.get(kb_id)
        return await self._metadata_store.update_job_payload_patch(
            record.id,
            job_id,
            payload_patch=payload_patch,
        )

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
        """Patch live progress on a running job without a status transition.

        Thin pass-through to the active metadata store's ``update_job_progress``
        (see its docstring). Used by long-running job handlers to publish
        incremental progress and the current pipeline activity message so
        clients polling the job see live movement instead of a frozen 0%.
        """
        record = await self._kb_service.get(kb_id)
        return await self._metadata_store.update_job_progress(
            record.id,
            job_id,
            progress=progress,
            completed_items=completed_items,
            stage=stage,
            result_patch=result_patch,
        )

    async def recover_orphan_jobs(
        self,
        *,
        resumable_job_types: set[str] | None = None,
        grace_seconds: float = 0.0,
    ) -> list[JobRecord]:
        """Recover orphan jobs while respecting live execution ownership.

        When ``resumable_job_types`` is given (durable worker enabled), queued
        jobs of those types are left in place for the worker to consume.
        """
        return await self._metadata_store.recover_orphan_jobs(
            resumable_job_types=resumable_job_types,
            grace_seconds=grace_seconds,
        )

    async def claim_next_worker_job(
        self,
        *,
        job_types: Sequence[str],
        max_queued_at: str | None = None,
    ) -> JobRecord | None:
        """Atomically claim the oldest eligible queued job for a durable worker."""
        return await self._metadata_store.claim_next_worker_job(
            job_types=job_types,
            max_queued_at=max_queued_at,
        )

    async def fail_claimed_worker_job(
        self,
        job: JobRecord,
        *,
        error_code: str,
        error_message: str,
    ) -> JobRecord:
        """Fail a claimed row by its persisted identity, even without catalog.

        Fence rejection commonly means the catalog is deleting, replaced, or
        already absent. Re-reading it must not prevent the required
        ``running -> failed`` terminal transition.
        """

        return await self._metadata_store.transition_job(
            job.kb_id,
            job.id,
            status="failed",
            progress=1.0,
            failed_items=1,
            error_code=error_code,
            error_message=error_message,
        )

    async def get_persisted_job(self, job: JobRecord) -> JobRecord:
        """Re-read a claimed row without requiring a live catalog record."""

        return await self._metadata_store.get_job(job.kb_id, job.id)

    async def cancel_job(
        self, kb_id: str, job_id: str, *, include_deleted: bool = False
    ) -> tuple[JobRecord, bool]:
        record = await self._kb_service.get(kb_id, include_deleted=include_deleted)
        existing = await self._metadata_store.get_job(record.id, job_id)
        if existing.status in {"succeeded", "cancelled"}:
            return existing, False
        if existing.status == "queued":
            updated = await self._metadata_store.transition_job(
                record.id,
                job_id,
                status="cancelled",
                error_code="cancelled_by_user",
                error_message="Job cancelled before execution",
            )
            return updated, True
        if existing.status in {"running", "retrying"}:
            updated = await self._metadata_store.transition_job(
                record.id, job_id, status="cancelling"
            )
            return updated, True
        if existing.status == "cancelling":
            return existing, False
        if existing.status == "failed":
            return existing, False
        return existing, False

    async def retry_job(
        self,
        kb_id: str,
        job_id: str,
        *,
        new_idempotency_key: str | None = None,
        include_deleted: bool = False,
    ) -> JobRecord:
        record = await self._kb_service.get(kb_id, include_deleted=include_deleted)
        async with self._metadata_store.job_execution_guard(job_id) as acquired:
            if not acquired:  # wait=True always acquires; retained for API parity.
                raise RuntimeError(f"Could not acquire execution guard for '{job_id}'")
            return await self._metadata_store.reset_job_for_retry(
                record.id, job_id, new_idempotency_key=new_idempotency_key
            )

    async def create_build_job_once(
        self,
        kb_id: str,
        *,
        document_id: str,
        parser_hash: str,
        index_hash: str,
        source_hash: str,
        lightrag_doc_id: str,
        sidecar_artifact_id: str | None = None,
        blocks_artifact_id: str | None = None,
        sidecar_uri: str | None = None,
        blocks_path: str | None = None,
        process_options: str | None = None,
        force_rechunk: bool = False,
        force_extract: bool = False,
        force_embedding: bool = False,
        job_type: str = "build_kg",
        idempotency_key: str | None = None,
    ) -> tuple[JobRecord, bool]:
        del sidecar_uri, blocks_path, process_options
        payload = {
            "document_id": document_id,
            "parser_hash": parser_hash,
            "index_hash": index_hash,
            "source_hash": source_hash,
            "lightrag_doc_id": lightrag_doc_id,
            "sidecar_artifact_id": sidecar_artifact_id,
            "blocks_artifact_id": blocks_artifact_id,
            "force_rechunk": force_rechunk,
            "force_extract": force_extract,
            "force_embedding": force_embedding,
        }
        payload["idempotency_fingerprint"] = _idempotency_fingerprint(payload)
        record, payload = await self._prepare_new_job(
            kb_id,
            job_type=job_type,
            payload=payload,
        )
        now = utc_now_iso()
        job = JobRecord(
            id=generate_track_id("job_build"),
            kb_id=record.id,
            workspace=record.workspace,
            batch_id=None,
            document_id=document_id,
            job_type=job_type,
            status="queued",
            stage="building",
            progress=0.0,
            total_items=1,
            completed_items=0,
            failed_items=0,
            idempotency_key=idempotency_key,
            config_version_id=record.active_config_version_id,
            config_hash=index_hash,
            retry_count=0,
            max_retries=3,
            payload=payload,
            result=None,
            error_code=None,
            error_message=None,
            created_at=now,
            updated_at=now,
            queued_at=now,
            started_at=None,
            finished_at=None,
            cancelled_at=None,
        )
        return await self._persist_job_once(job)

    async def create_batch_build_job_once(
        self,
        kb_id: str,
        *,
        batch_id: str,
        document_ids: Sequence[str],
        total_items: int,
        plan_items: Sequence[dict[str, Any]],
        planning_failures: Sequence[dict[str, Any]],
        force_rechunk: bool = False,
        force_extract: bool = False,
        force_embedding: bool = False,
        job_type: str = "build_kg",
        idempotency_key: str | None = None,
    ) -> tuple[JobRecord, bool]:
        payload = {
            "document_ids": list(document_ids),
            "items": list(plan_items),
            "planning_failures": list(planning_failures),
            "force_rechunk": force_rechunk,
            "force_extract": force_extract,
            "force_embedding": force_embedding,
        }
        payload["idempotency_fingerprint"] = _idempotency_fingerprint(payload)
        return await self.create_job_once(
            kb_id,
            job_type=job_type,
            batch_id=batch_id,
            document_id=None,
            stage="building",
            total_items=total_items,
            payload=payload,
            idempotency_key=idempotency_key,
        )

    async def create_delete_job_once(
        self,
        kb_id: str,
        *,
        document_id: str,
        lightrag_doc_id: str | None,
        delete_source_file: bool = False,
        delete_artifacts: bool = False,
        delete_llm_cache: bool = False,
        delete_graph_orphans: bool = True,
        strategy: str = "safe",
        idempotency_key: str | None = None,
    ) -> tuple[JobRecord, bool]:
        payload = {
            "document_id": document_id,
            "lightrag_doc_id": lightrag_doc_id,
            "delete_source_file": delete_source_file,
            "delete_artifacts": delete_artifacts,
            "delete_llm_cache": delete_llm_cache,
            "delete_graph_orphans": delete_graph_orphans,
            "strategy": strategy,
            # Per-document COW attempt-token map ({document_id: token}).
            # Populated by route/worker after the B1 state machine claims or
            # rotates a token; reused on durable worker resume via Store A
            # claim idempotency. Each document's token is independent.
            "attempt_tokens": {},
        }
        payload["idempotency_fingerprint"] = _idempotency_fingerprint(payload)
        return await self.create_job_once(
            kb_id,
            job_type="delete",
            document_id=document_id,
            stage="deleting",
            total_items=1,
            payload=payload,
            idempotency_key=idempotency_key,
        )

    async def create_replace_job_once(
        self,
        kb_id: str,
        *,
        document_id: str,
        previous_lightrag_doc_id: str | None,
        source_name: str,
        source_type: str,
        source_hash: str,
        content_type: str | None,
        size_bytes: int,
        delete_source_file: bool = True,
        delete_artifacts: bool = True,
        delete_llm_cache: bool = False,
        auto_parse: bool = False,
        auto_index: bool = False,
        parser_engine: str | None = None,
        process_options: str | None = None,
        force_reparse: bool = False,
        idempotency_key: str | None = None,
        staging_object_uri: str | None = None,
    ) -> tuple[JobRecord, bool]:
        fingerprint_payload = {
            "document_id": document_id,
            "source_name": source_name,
            "source_type": source_type,
            "source_hash": source_hash,
            "content_type": content_type,
            "size_bytes": size_bytes,
            "delete_source_file": delete_source_file,
            "delete_artifacts": delete_artifacts,
            "delete_llm_cache": delete_llm_cache,
            "auto_parse": auto_parse,
            "auto_index": auto_index,
            "parser_engine": parser_engine,
            "process_options": process_options,
            "force_reparse": force_reparse,
        }
        payload = {
            **fingerprint_payload,
            "previous_lightrag_doc_id": previous_lightrag_doc_id,
            # Per-document COW attempt-token map ({document_id: token}).
            # Populated by route/worker after the B1 state machine claims or
            # rotates a token; reused on durable worker resume via Store A
            # claim idempotancy.
            "attempt_tokens": {},
        }
        if staging_object_uri is not None:
            # Phase 3.2 object-backed staging: the immutable candidate object
            # URI that carries the replacement bytes across request-process
            # death and moved-root worker resume. Metadata-only — never a local
            # path. May also be patched in after staging by the route.
            payload["staging_object_uri"] = staging_object_uri
        payload["idempotency_fingerprint"] = _idempotency_fingerprint(
            fingerprint_payload
        )
        return await self.create_job_once(
            kb_id,
            job_type="replace",
            document_id=document_id,
            stage="replacing",
            total_items=1,
            payload=payload,
            idempotency_key=idempotency_key,
        )

    async def create_batch_delete_job_once(
        self,
        kb_id: str,
        *,
        batch_id: str,
        document_ids: Sequence[str],
        delete_source_file: bool = False,
        delete_artifacts: bool = False,
        delete_llm_cache: bool = False,
        delete_graph_orphans: bool = True,
        strategy: str = "safe",
        idempotency_key: str | None = None,
    ) -> tuple[JobRecord, bool]:
        payload = {
            "document_ids": list(document_ids),
            "delete_source_file": delete_source_file,
            "delete_artifacts": delete_artifacts,
            "delete_llm_cache": delete_llm_cache,
            "delete_graph_orphans": delete_graph_orphans,
            "strategy": strategy,
            # Per-document COW attempt-token map ({document_id: token}).
            # Each document in the batch gets its own independent token;
            # partial failures use per-document tokens with the existing
            # partial_* transition shapes.
            "attempt_tokens": {},
        }
        payload["idempotency_fingerprint"] = _idempotency_fingerprint(payload)
        return await self.create_job_once(
            kb_id,
            job_type="delete",
            batch_id=batch_id,
            document_id=None,
            stage="deleting",
            total_items=len(document_ids),
            payload=payload,
            idempotency_key=idempotency_key,
        )


def _idempotency_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
