"""Durable job worker for KB long-running operations.

The KB API layer creates persistent ``jobs`` rows and (on the happy path)
runs them inside FastAPI ``BackgroundTasks``. That in-process execution has
two gaps the audit called out:

1. ``POST /jobs/{job_id}:retry`` resets a job back to ``queued`` but nothing
   consumes it — the client had to re-trigger the original business action.
2. After a process restart, ``queued`` jobs cannot resume; orphan recovery
   simply fails them.

:class:`JobWorker` closes both gaps for job types that are *re-drivable from
persisted state* (single-document ``parse`` / ``build_kg`` / ``reindex``
plus single- and batch-document ``delete`` jobs).
It polls the metadata store for eligible ``queued`` jobs, atomically claims
each one (``queued → running`` single-winner CAS via
:meth:`SQLiteMetadataStore.claim_next_worker_job`), and dispatches to a
registered executor that rebuilds the plan and runs it to a terminal state.

Coordination with the in-process happy path is handled by a *grace window*:
freshly-created jobs are flipped to ``running`` by their own background task
within milliseconds, so the worker — which only claims jobs that have sat
``queued`` longer than ``claim_grace_seconds`` — never races them. Retried
jobs (``:retry`` refreshes ``queued_at``) and restart-orphaned ``queued`` jobs
age past the window and get picked up.

The worker is **opt-in** (``LIGHTRAG_KB_JOB_WORKER=true``). When disabled, the
system behaves exactly as before and orphan recovery fails every transient
job. Executors that need request-scoped inputs that are not persisted (upload
bytes for ``replace`` / batch ``sync`` / aggregate parse-and-build upload
jobs) are intentionally NOT registered as resumable; those job types still
fail on restart and require a fresh request.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from lightrag.api.job_service import JobService
from lightrag.api.metadata_store import JobRecord
from lightrag.utils import logger

# Executor contract: given a freshly-claimed (already ``running``) job, drive
# it to a terminal state (``succeeded`` / ``failed``) and return None. Raising
# is allowed — the worker will mark the job ``failed`` as a backstop.
JobExecutor = Callable[[JobRecord], Awaitable[None]]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JobWorker:
    """Polls for queued jobs and dispatches them to registered executors."""

    def __init__(
        self,
        job_service: JobService,
        *,
        executors: dict[str, JobExecutor],
        poll_interval_seconds: float = 1.0,
        claim_grace_seconds: float = 5.0,
    ) -> None:
        self._job_service = job_service
        self._executors = dict(executors)
        self._poll_interval = max(0.05, float(poll_interval_seconds))
        self._claim_grace_seconds = max(0.0, float(claim_grace_seconds))
        self._job_types = tuple(self._executors.keys())
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    @property
    def resumable_job_types(self) -> set[str]:
        """Job types this worker can re-drive from persisted state."""
        return set(self._job_types)

    def _grace_cutoff(self) -> str | None:
        if self._claim_grace_seconds <= 0:
            return None
        cutoff = _utc_now() - timedelta(seconds=self._claim_grace_seconds)
        return cutoff.isoformat()

    async def poll_once(self) -> JobRecord | None:
        """Claim and run a single eligible job. Returns the job, or None.

        Deterministic entry point used by tests and by the polling loop. Any
        exception escaping the executor is caught and converted into a
        ``failed`` terminal transition so a single bad job cannot wedge the
        loop.
        """
        if not self._job_types:
            return None
        try:
            job = await self._job_service.claim_next_worker_job(
                job_types=self._job_types,
                max_queued_at=self._grace_cutoff(),
            )
        except Exception as exc:  # noqa: BLE001 — never let polling crash
            logger.error("JobWorker claim failed: %s", exc)
            return None
        if job is None:
            return None

        executor = self._executors.get(job.job_type)
        if executor is None:  # pragma: no cover — job_types is derived from keys
            return job
        try:
            await executor(job)
        except Exception as exc:  # noqa: BLE001 — backstop terminal failure
            logger.error(
                "JobWorker executor for job '%s' (type=%s) raised: %s",
                job.id,
                job.job_type,
                exc,
            )
            await self._fail_job_quietly(job, str(exc))
        return job

    async def _fail_job_quietly(self, job: JobRecord, message: str) -> None:
        try:
            await self._job_service.transition_job(
                job.kb_id,
                job.id,
                status="failed",
                progress=1.0,
                failed_items=1,
                error_code="worker_executor_error",
                error_message=message,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "JobWorker could not mark job '%s' failed: %s", job.id, exc
            )

    async def _run_loop(self) -> None:
        logger.info(
            "JobWorker started (types=%s, poll=%.2fs, grace=%.2fs)",
            ",".join(self._job_types) or "<none>",
            self._poll_interval,
            self._claim_grace_seconds,
        )
        while not self._stop_event.is_set():
            try:
                # Drain all currently-eligible jobs before sleeping.
                while not self._stop_event.is_set():
                    claimed = await self.poll_once()
                    if claimed is None:
                        break
            except Exception as exc:  # noqa: BLE001 — keep the loop alive
                logger.error("JobWorker loop iteration failed: %s", exc)
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self._poll_interval
                )
            except asyncio.TimeoutError:
                pass
        logger.info("JobWorker stopped")

    def start(self) -> None:
        """Start the background polling loop (idempotent)."""
        if self._task is not None and not self._task.done():
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Signal the loop to stop and await its completion."""
        self._stop_event.set()
        if self._task is not None:
            try:
                await self._task
            except asyncio.CancelledError:  # pragma: no cover
                pass
            self._task = None


def build_parse_executor(
    *,
    document_service: Any,
    registry: Any,
    job_service: JobService,
) -> JobExecutor:
    """Executor that re-drives a single-document ``parse`` job.

    Rebuilds the parse plan from the document's persisted parser directives,
    re-claims the document into ``parse_queued`` (allowed from
    ``parse_failed`` / ``uploaded`` / ``parsed`` — only active states block),
    and reuses the same ``_execute_parse_plan`` helper the route uses, then
    applies the terminal job transition.
    """
    # Lazy import to avoid a router <-> worker import cycle.
    from lightrag.api.routers.kb_document_routes import _execute_parse_plan

    async def _run(job: JobRecord) -> None:
        kb_id = job.kb_id
        document_id = job.document_id or job.payload.get("document_id")
        if not document_id:
            await job_service.transition_job(
                kb_id,
                job.id,
                status="failed",
                progress=1.0,
                failed_items=1,
                error_code="worker_invalid_payload",
                error_message="parse job has no document_id",
            )
            return
        payload = job.payload or {}
        plan = await document_service.create_parse_plan(
            kb_id,
            document_id,
            parser_engine=payload.get("parser_engine"),
            process_options=payload.get("process_options"),
            force_reparse=bool(payload.get("force_reparse", False)),
            auto_index=bool(payload.get("auto_index", False)),
        )
        await document_service.mark_parse_queued(kb_id, document_id, job=job, plan=plan)
        rag = await registry.get(kb_id)
        item = await _execute_parse_plan(
            document_service=document_service,
            kb_id=kb_id,
            job_id=job.id,
            plan=plan,
            rag=rag,
            job_service=job_service,
        )
        if item["status"] == "succeeded":
            await job_service.transition_job(
                kb_id,
                job.id,
                status="succeeded",
                progress=1.0,
                completed_items=1,
                result={
                    "document_id": item["document_id"],
                    "parser_hash": item["parser_hash"],
                    "lightrag_doc_id": item["lightrag_doc_id"],
                    "artifact_count": item["artifact_count"],
                    "resumed_by_worker": True,
                },
            )
        elif item["status"] == "cancelled":
            await job_service.transition_job(
                kb_id,
                job.id,
                status="cancelled",
                progress=1.0,
                error_code="cancelled_by_user",
                error_message=item.get("error_message"),
            )
        else:
            await job_service.transition_job(
                kb_id,
                job.id,
                status="failed",
                progress=1.0,
                failed_items=1,
                error_code=item["error_code"],
                error_message=item["error_message"],
            )

    return _run


def build_build_kg_executor(
    *,
    document_service: Any,
    index_service: Any,
    registry: Any,
    job_service: JobService,
) -> JobExecutor:
    """Executor that re-drives a single-document ``build_kg`` / ``reindex`` job."""
    from lightrag.api.routers.kb_document_routes import _execute_build_plan

    async def _run(job: JobRecord) -> None:
        kb_id = job.kb_id
        document_id = job.document_id or job.payload.get("document_id")
        if not document_id:
            await job_service.transition_job(
                kb_id,
                job.id,
                status="failed",
                progress=1.0,
                failed_items=1,
                error_code="worker_invalid_payload",
                error_message="build job has no document_id",
            )
            return
        payload = job.payload or {}
        rag = await registry.get(kb_id)
        plan = await index_service.create_build_plan(
            kb_id,
            document_id,
            rag=rag,
            force_rechunk=bool(payload.get("force_rechunk", False)),
            force_extract=bool(payload.get("force_extract", False)),
            force_embedding=bool(payload.get("force_embedding", False)),
        )
        if not plan.skipped:
            await index_service.claim_build_queued(kb_id, job_id=job.id, plan=plan)
        item = await _execute_build_plan(
            index_service=index_service,
            kb_id=kb_id,
            job_id=job.id,
            plan=plan,
            rag=rag,
            job_service=job_service,
        )
        if item["status"] == "succeeded":
            await job_service.transition_job(
                kb_id,
                job.id,
                status="succeeded",
                progress=1.0,
                completed_items=1,
                result={
                    "document_id": item["document_id"],
                    "skipped": item["skipped"],
                    "skip_reason": item.get("skip_reason"),
                    "index_hash": item["index_hash"],
                    "chunks_count": item.get("chunks_count"),
                    "entity_count": item.get("entity_count"),
                    "relation_count": item.get("relation_count"),
                    "resumed_by_worker": True,
                },
            )
        elif item["status"] == "cancelled":
            await job_service.transition_job(
                kb_id,
                job.id,
                status="cancelled",
                progress=1.0,
                error_code="cancelled_by_user",
                error_message=item.get("error_message"),
            )
        else:
            await job_service.transition_job(
                kb_id,
                job.id,
                status="failed",
                progress=1.0,
                failed_items=1,
                error_code=item["error_code"],
                error_message=item["error_message"],
            )

    return _run


def build_delete_executor(
    *,
    document_service: Any,
    registry: Any,
    job_service: JobService,
    index_service: Any | None = None,
) -> JobExecutor:
    """Executor that re-drives persisted ``delete`` jobs.

    Delete jobs need only persisted document ids + delete options, so both
    single-document and ``documents:batch-delete`` jobs can resume after a crash.
    Replace/sync jobs still need request-uploaded bytes and remain non-resumable.
    """
    from lightrag.api.routers.kb_document_routes import (
        _delete_failure_message,
        _delete_job_result,
        _execute_delete_document_impl,
        _run_conservative_kb_rebuild,
    )

    async def _run_single(job: JobRecord, payload: dict[str, Any]) -> None:
        kb_id = job.kb_id
        document_id = job.document_id or payload.get("document_id")
        if not document_id:
            await job_service.transition_job(
                kb_id,
                job.id,
                status="failed",
                progress=1.0,
                failed_items=1,
                error_code="worker_invalid_payload",
                error_message="delete job has no document_id",
            )
            return
        delete_source_file = bool(payload.get("delete_source_file", False))
        delete_artifacts = bool(payload.get("delete_artifacts", False))
        delete_llm_cache = bool(payload.get("delete_llm_cache", False))
        document = await document_service.claim_delete(
            kb_id,
            str(document_id),
            job=job,
            delete_source_file=delete_source_file,
            delete_artifacts=delete_artifacts,
        )
        item = await _execute_delete_document_impl(
            document_service=document_service,
            kb_id=kb_id,
            job_id=job.id,
            document=document,
            active_registry=registry,
            delete_source_file=delete_source_file,
            delete_artifacts=delete_artifacts,
            delete_llm_cache=delete_llm_cache,
        )
        if item["status"] == "succeeded":
            result: dict[str, Any] = {
                "document_id": item["document_id"],
                "lightrag_doc_id": item.get("lightrag_doc_id"),
                "resumed_by_worker": True,
            }
            if payload.get("strategy") == "rebuild_kb" and index_service is not None:
                result["rebuild"] = await _run_conservative_kb_rebuild(
                    document_service=document_service,
                    index_service=index_service,
                    registry=registry,
                    kb_id=kb_id,
                )
            await job_service.transition_job(
                kb_id,
                job.id,
                status="succeeded",
                progress=1.0,
                completed_items=1,
                result=result,
            )
        else:
            await job_service.transition_job(
                kb_id,
                job.id,
                status="failed",
                progress=1.0,
                failed_items=1,
                error_code=item["error_code"],
                error_message=item["error_message"],
            )

    async def _run_batch(job: JobRecord, payload: dict[str, Any]) -> None:
        kb_id = job.kb_id
        raw_document_ids = payload.get("document_ids")
        if not isinstance(raw_document_ids, list) or not all(
            isinstance(item, str) and item for item in raw_document_ids
        ):
            await job_service.transition_job(
                kb_id,
                job.id,
                status="failed",
                progress=1.0,
                failed_items=1,
                error_code="worker_invalid_payload",
                error_message="batch delete job has no valid document_ids payload",
            )
            return
        document_ids = list(dict.fromkeys(raw_document_ids))
        delete_source_file = bool(payload.get("delete_source_file", False))
        delete_artifacts = bool(payload.get("delete_artifacts", False))
        delete_llm_cache = bool(payload.get("delete_llm_cache", False))
        documents, claim_failures = await document_service.claim_batch_delete(
            kb_id,
            document_ids,
            job=job,
            delete_source_file=delete_source_file,
            delete_artifacts=delete_artifacts,
        )
        item_results = [*claim_failures]
        completed_items = 0
        failed_items = len(item_results)
        for document in documents:
            item = await _execute_delete_document_impl(
                document_service=document_service,
                kb_id=kb_id,
                job_id=job.id,
                document=document,
                active_registry=registry,
                delete_source_file=delete_source_file,
                delete_artifacts=delete_artifacts,
                delete_llm_cache=delete_llm_cache,
            )
            item_results.append(item)
            if item["status"] == "succeeded":
                completed_items += 1
            else:
                failed_items += 1
        final_result = _delete_job_result(
            batch_id=job.batch_id,
            total_items=len(document_ids),
            completed_items=completed_items,
            failed_items=failed_items,
            items=item_results,
        )
        final_result["resumed_by_worker"] = True
        if payload.get("strategy") == "rebuild_kb" and completed_items > 0 and index_service is not None:
            final_result["rebuild"] = await _run_conservative_kb_rebuild(
                document_service=document_service,
                index_service=index_service,
                registry=registry,
                kb_id=kb_id,
            )
        await job_service.transition_job(
            kb_id,
            job.id,
            status="succeeded" if failed_items == 0 else "failed",
            progress=1.0,
            completed_items=completed_items,
            failed_items=failed_items,
            result=final_result,
            error_code=None if failed_items == 0 else "partial_delete_failed",
            error_message=None
            if failed_items == 0
            else _delete_failure_message(failed_items, len(document_ids)),
        )

    async def _run(job: JobRecord) -> None:
        payload = job.payload or {}
        if job.document_id is None and isinstance(payload.get("document_ids"), list):
            await _run_batch(job, payload)
            return
        await _run_single(job, payload)

    return _run
