from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from lightrag.api.kb_service import KnowledgeBaseService, utc_now_iso
from lightrag.api.metadata_store import (
    JobRecord,
    MetadataJobStatus,
    SQLiteMetadataStore,
)
from lightrag.utils import generate_track_id

_RUNNING_JOB_STATUSES = ("queued", "running", "retrying", "cancelling")


class JobService:
    def __init__(
        self,
        kb_service: KnowledgeBaseService,
        metadata_store: SQLiteMetadataStore,
    ):
        self._kb_service = kb_service
        self._metadata_store = metadata_store

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
        record = await self._kb_service.get(kb_id)
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
            payload=payload or {},
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
        return await self._metadata_store.create_job(job)

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
        record = await self._kb_service.get(kb_id)
        now = utc_now_iso()
        job_payload = payload or {}
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
        return await self._metadata_store.create_job_once(job)

    async def create_parse_job(
        self,
        kb_id: str,
        *,
        document_id: str,
        parser_hash: str,
        lightrag_doc_id: str,
        parser_engine: str,
        process_options: str,
        source_uri: str,
        source_hash: str,
        force_reparse: bool = False,
        auto_index: bool = False,
        idempotency_key: str | None = None,
    ) -> JobRecord:
        record = await self._kb_service.get(kb_id)
        now = utc_now_iso()
        payload = {
            "document_id": document_id,
            "source_uri": source_uri,
            "source_hash": source_hash,
            "parser_engine": parser_engine,
            "process_options": process_options,
            "parser_hash": parser_hash,
            "lightrag_doc_id": lightrag_doc_id,
            "force_reparse": force_reparse,
            "auto_index": auto_index,
        }
        payload["idempotency_fingerprint"] = _idempotency_fingerprint(payload)
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
        return await self._metadata_store.create_job(job)

    async def create_parse_job_once(
        self,
        kb_id: str,
        *,
        document_id: str,
        parser_hash: str,
        lightrag_doc_id: str,
        parser_engine: str,
        process_options: str,
        source_uri: str,
        source_hash: str,
        force_reparse: bool = False,
        auto_index: bool = False,
        idempotency_key: str | None = None,
    ) -> tuple[JobRecord, bool]:
        record = await self._kb_service.get(kb_id)
        now = utc_now_iso()
        payload = {
            "document_id": document_id,
            "source_uri": source_uri,
            "source_hash": source_hash,
            "parser_engine": parser_engine,
            "process_options": process_options,
            "parser_hash": parser_hash,
            "lightrag_doc_id": lightrag_doc_id,
            "force_reparse": force_reparse,
            "auto_index": auto_index,
        }
        payload["idempotency_fingerprint"] = _idempotency_fingerprint(payload)
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
        return await self._metadata_store.create_job_once(job)

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

    async def list_jobs(
        self,
        kb_id: str,
        *,
        statuses: Sequence[str] | None = None,
        document_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[JobRecord], int]:
        record = await self._kb_service.get(kb_id)
        return await self._metadata_store.list_jobs(
            record.id,
            statuses=statuses,
            document_id=document_id,
            limit=limit,
            offset=offset,
        )

    async def list_running_jobs(
        self, kb_id: str, *, limit: int = 20
    ) -> list[JobRecord]:
        jobs, _total = await self.list_jobs(
            kb_id, statuses=_RUNNING_JOB_STATUSES, limit=limit, offset=0
        )
        return jobs

    async def list_dead_letter_jobs(
        self, kb_id: str, *, limit: int = 50, offset: int = 0
    ) -> tuple[list[JobRecord], int]:
        """List dead-lettered jobs (failed + retries exhausted) for the KB."""
        record = await self._kb_service.get(kb_id)
        return await self._metadata_store.list_dead_letter_jobs(
            record.id, limit=limit, offset=offset
        )

    async def get_job(self, kb_id: str, job_id: str) -> JobRecord:
        record = await self._kb_service.get(kb_id)
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
        record = await self._kb_service.get(kb_id)
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

    async def recover_orphan_jobs(
        self, *, resumable_job_types: set[str] | None = None
    ) -> list[JobRecord]:
        """Mark queued/running/cancelling/retrying jobs as failed at startup.

        When ``resumable_job_types`` is given (durable worker enabled), queued
        jobs of those types are left in place for the worker to consume.
        """
        return await self._metadata_store.recover_orphan_jobs(
            resumable_job_types=resumable_job_types
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

    async def cancel_job(
        self, kb_id: str, job_id: str
    ) -> tuple[JobRecord, bool]:
        record = await self._kb_service.get(kb_id)
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
    ) -> JobRecord:
        record = await self._kb_service.get(kb_id)
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
        sidecar_uri: str | None,
        blocks_path: str | None,
        process_options: str,
        force_rechunk: bool = False,
        force_extract: bool = False,
        force_embedding: bool = False,
        job_type: str = "build_kg",
        idempotency_key: str | None = None,
    ) -> tuple[JobRecord, bool]:
        record = await self._kb_service.get(kb_id)
        now = utc_now_iso()
        payload = {
            "document_id": document_id,
            "parser_hash": parser_hash,
            "index_hash": index_hash,
            "source_hash": source_hash,
            "lightrag_doc_id": lightrag_doc_id,
            "sidecar_uri": sidecar_uri,
            "blocks_path": blocks_path,
            "process_options": process_options,
            "force_rechunk": force_rechunk,
            "force_extract": force_extract,
            "force_embedding": force_embedding,
        }
        payload["idempotency_fingerprint"] = _idempotency_fingerprint(payload)
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
        return await self._metadata_store.create_job_once(job)

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
    ) -> tuple[JobRecord, bool]:
        fingerprint_payload = {
            "document_id": document_id,
            "source_name": source_name,
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
        }
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
