"""Durable job worker for KB long-running operations.

The KB API layer creates persistent ``jobs`` rows and (on the happy path)
runs them inside FastAPI ``BackgroundTasks``. That in-process execution has
two gaps the audit called out:

1. ``POST /jobs/{job_id}:retry`` resets a job back to ``queued`` but nothing
   consumes it — the client had to re-trigger the original business action.
2. After a process restart, ``queued`` jobs cannot resume; orphan recovery
   simply fails them.

:class:`JobWorker` closes both gaps for job types that are *re-drivable from
persisted state* (single-document ``parse`` / ``build_kg`` / ``reindex``,
single- and batch-document ``delete`` jobs, KB hard-delete, and Agent profile
refresh jobs).
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
bytes for aggregate parse-and-build upload jobs) are intentionally NOT
registered as resumable; those job types still fail on restart and require a
fresh request. Replace and aggregate sync jobs stage request bytes before they
are queued, so the worker can resume them from persisted files.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from lightrag.api.job_service import JobService
from lightrag.api.metadata_store import JobRecord, MetadataRecordNotFoundError
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
    index_service: Any | None = None,
) -> JobExecutor:
    """Executor that re-drives ``parse`` jobs (single-document or aggregate).

    Single-document jobs rebuild the parse plan from the document's persisted
    parser directives, re-claim the document into ``parse_queued`` (allowed
    from ``parse_failed`` / ``uploaded`` / ``parsed`` — only active states
    block), and reuse the same ``_execute_parse_plan`` helper the route uses.

    Aggregate jobs (``document_id`` is ``None``, payload carries
    ``document_ids``) are produced by multi-file ``upload`` / ``texts``
    ``auto_parse`` and by ``documents:batch-parse``. The source files are
    persisted before the job runs, so the executor re-plans each document,
    re-claims it (orphan recovery resets in-flight docs to ``parse_failed``),
    parses it, and — when ``auto_index`` is set and an index service is wired —
    chains the KG build to ``ready``. Per-item results aggregate into the
    single job, mirroring the in-process ``_run_auto_parse_batch``.
    """
    # Lazy import to avoid a router <-> worker import cycle.
    from lightrag.api.routers.kb_document_routes import (
        _batch_parse_failure_message,
        _batch_parse_job_result,
        _execute_build_plan_batch,
        _execute_parse_plan,
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
                error_message="parse job has no document_id",
            )
            return
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

    async def _run_aggregate(job: JobRecord, payload: dict[str, Any]) -> None:
        kb_id = job.kb_id
        raw_ids = payload.get("document_ids")
        if not isinstance(raw_ids, list) or not all(
            isinstance(item, str) and item for item in raw_ids
        ):
            await job_service.transition_job(
                kb_id,
                job.id,
                status="failed",
                progress=1.0,
                failed_items=1,
                error_code="worker_invalid_payload",
                error_message="aggregate parse job has no valid document_ids payload",
            )
            return
        document_ids = list(dict.fromkeys(raw_ids))
        auto_index = bool(payload.get("auto_index", False))
        completed_items = 0
        failed_items = 0
        rag = await registry.get(kb_id) if document_ids else None
        item_by_id: dict[str, dict[str, Any]] = {}

        # ── Phase 1: concurrent parse (bounded by MAX_PARALLEL_PARSE_MINERU) ──
        parse_concurrency = max(
            1, int(getattr(rag, "max_parallel_parse_mineru", 1) or 1)
        )
        parse_sem = asyncio.Semaphore(parse_concurrency)

        async def _do_one_parse(
            document_id: str,
        ) -> tuple[str, Any, dict[str, Any]]:
            async with parse_sem:
                try:
                    plan = await document_service.create_parse_plan(
                        kb_id,
                        document_id,
                        parser_engine=payload.get("parser_engine"),
                        process_options=payload.get("process_options"),
                        force_reparse=bool(payload.get("force_reparse", False)),
                        auto_index=auto_index,
                    )
                    await document_service.mark_parse_queued(
                        kb_id, document_id, job=job, plan=plan
                    )
                except Exception as exc:  # noqa: BLE001 — plan/claim failure
                    return (
                        document_id,
                        None,
                        {
                            "document_id": document_id,
                            "status": "failed",
                            "error_code": "parse_failed",
                            "error_message": str(exc),
                        },
                    )
                item = await _execute_parse_plan(
                    document_service=document_service,
                    kb_id=kb_id,
                    job_id=job.id,
                    plan=plan,
                    rag=rag,
                    job_service=job_service,
                )
                return document_id, plan, item

        parse_outcomes: list[tuple[str, Any, dict[str, Any]]] = []
        if rag is not None and document_ids:
            raw_outcomes = await asyncio.gather(
                *[_do_one_parse(d) for d in document_ids],
                return_exceptions=True,
            )
            # return_exceptions=True keeps Phase 2 reachable (releasing any
            # build_queued claim) even on an unexpected BaseException; map any
            # exception back to its doc as a failed item via positional zip.
            for doc_id, outcome in zip(document_ids, raw_outcomes):
                if isinstance(outcome, BaseException):
                    parse_outcomes.append(
                        (
                            doc_id,
                            None,
                            {
                                "document_id": doc_id,
                                "status": "failed",
                                "error_code": "parse_failed",
                                "error_message": str(outcome),
                            },
                        )
                    )
                else:
                    parse_outcomes.append(outcome)
        for doc_id, _plan, item in parse_outcomes:
            item_by_id[doc_id] = item

        # ── Phase 2: bulk auto_index build through a single pipeline drain ──
        if auto_index and index_service is not None and rag is not None:
            build_plans: list[Any] = []
            build_plan_to_item: dict[str, dict[str, Any]] = {}
            for doc_id, _plan, item in parse_outcomes:
                if item["status"] != "succeeded":
                    continue
                try:
                    build_plan = await index_service.create_build_plan(
                        kb_id, doc_id, rag=rag
                    )
                    if not build_plan.skipped:
                        await index_service.claim_build_queued(
                            kb_id, job_id=job.id, plan=build_plan
                        )
                except Exception as exc:  # noqa: BLE001 — per-item plan failure
                    item["status"] = "failed"
                    item["error_code"] = "build_failed"
                    item["error_message"] = str(exc)
                    continue
                build_plans.append(build_plan)
                build_plan_to_item[build_plan.document.id] = item

            if build_plans:
                build_results = await _execute_build_plan_batch(
                    index_service=index_service,
                    kb_id=kb_id,
                    job_id=job.id,
                    rag=rag,
                    plans=build_plans,
                    job_service=job_service,
                )
                for build_plan in build_plans:
                    item = build_plan_to_item[build_plan.document.id]
                    build_item = build_results.get(build_plan.document.id)
                    if build_item is None:
                        item["status"] = "failed"
                        item["error_code"] = "build_failed"
                        item["error_message"] = "Build result missing from batch"
                        continue
                    item["build_result"] = build_item
                    if build_item["status"] not in {"succeeded", "cancelled"}:
                        item["status"] = "failed"
                        item["error_code"] = build_item.get("error_code")
                        item["error_message"] = build_item.get("error_message")

        # ── Phase 3: finalize (preserve input order for stable reporting) ──
        item_results: list[dict[str, Any]] = []
        for document_id in document_ids:
            item = item_by_id.get(document_id)
            if item is None:  # pragma: no cover — rag None / empty list path
                continue
            item_results.append(item)
            if item["status"] == "succeeded":
                completed_items += 1
            else:
                failed_items += 1
        final_result = _batch_parse_job_result(
            batch_id=job.batch_id or "",
            total_items=len(document_ids),
            completed_items=completed_items,
            failed_items=failed_items,
            items=item_results,
        )
        final_result["resumed_by_worker"] = True
        await job_service.transition_job(
            kb_id,
            job.id,
            status="succeeded" if failed_items == 0 else "failed",
            progress=1.0,
            completed_items=completed_items,
            failed_items=failed_items,
            result=final_result,
            error_code=None if failed_items == 0 else "partial_parse_failed",
            error_message=None
            if failed_items == 0
            else _batch_parse_failure_message(failed_items, len(document_ids)),
        )

    async def _run(job: JobRecord) -> None:
        payload = job.payload or {}
        if job.document_id is None and isinstance(payload.get("document_ids"), list):
            await _run_aggregate(job, payload)
            return
        await _run_single(job, payload)

    return _run


def build_build_kg_executor(
    *,
    document_service: Any,
    index_service: Any,
    registry: Any,
    job_service: JobService,
) -> JobExecutor:
    """Executor that re-drives ``build_kg`` / ``reindex`` jobs.

    Single-document jobs rebuild the plan and run it. Aggregate jobs
    (``document_id`` is ``None``, payload carries ``document_ids``) are produced
    by ``documents:batch-build-kg`` / ``:batch-reindex`` / ``kb:rebuild``; the
    parse artifacts are already persisted, so the executor re-plans each
    document from the persisted ``force_*`` directives and runs them, mirroring
    the in-process batch build loop.
    """
    from lightrag.api.routers.kb_document_routes import (
        _batch_build_failure_message,
        _batch_build_job_result,
        _execute_build_plan,
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
                error_message="build job has no document_id",
            )
            return
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

    async def _run_aggregate(job: JobRecord, payload: dict[str, Any]) -> None:
        kb_id = job.kb_id
        raw_ids = payload.get("document_ids")
        if not isinstance(raw_ids, list) or not all(
            isinstance(item, str) and item for item in raw_ids
        ):
            await job_service.transition_job(
                kb_id,
                job.id,
                status="failed",
                progress=1.0,
                failed_items=1,
                error_code="worker_invalid_payload",
                error_message="aggregate build job has no valid document_ids payload",
            )
            return
        document_ids = list(dict.fromkeys(raw_ids))
        rag = await registry.get(kb_id) if document_ids else None
        batch_plan = await index_service.create_batch_build_plan(
            kb_id,
            document_ids,
            rag=rag,
            force_rechunk=bool(payload.get("force_rechunk", False)),
            force_extract=bool(payload.get("force_extract", False)),
            force_embedding=bool(payload.get("force_embedding", False)),
        )
        item_results: list[dict[str, Any]] = [*batch_plan.failures]
        completed_items = 0
        failed_items = len(item_results)
        for plan in batch_plan.plans:
            if not plan.skipped:
                try:
                    await index_service.claim_build_queued(
                        kb_id, job_id=job.id, plan=plan
                    )
                except Exception as exc:  # noqa: BLE001 — per-item claim failure
                    item_results.append(
                        {
                            "document_id": plan.document.id,
                            "status": "failed",
                            "error_code": "build_failed",
                            "error_message": str(exc),
                        }
                    )
                    failed_items += 1
                    continue
            item = await _execute_build_plan(
                index_service=index_service,
                kb_id=kb_id,
                job_id=job.id,
                plan=plan,
                rag=rag,
                job_service=job_service,
            )
            item_results.append(item)
            if item["status"] == "succeeded":
                completed_items += 1
            else:
                failed_items += 1
        final_result = _batch_build_job_result(
            batch_id=job.batch_id or batch_plan.batch_id,
            total_items=len(document_ids),
            completed_items=completed_items,
            failed_items=failed_items,
            items=item_results,
        )
        final_result["resumed_by_worker"] = True
        await job_service.transition_job(
            kb_id,
            job.id,
            status="succeeded" if failed_items == 0 else "failed",
            progress=1.0,
            completed_items=completed_items,
            failed_items=failed_items,
            result=final_result,
            error_code=None if failed_items == 0 else "partial_build_failed",
            error_message=None
            if failed_items == 0
            else _batch_build_failure_message(failed_items, len(document_ids)),
        )

    async def _run(job: JobRecord) -> None:
        payload = job.payload or {}
        if job.document_id is None and isinstance(payload.get("document_ids"), list):
            await _run_aggregate(job, payload)
            return
        await _run_single(job, payload)

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
    """
    from lightrag.api.routers.kb_document_routes import (
        _capture_graph_footprint,
        _delete_failure_message,
        _delete_job_result,
        _deserialize_graph_footprint,
        _execute_delete_document_impl,
        _merge_footprints,
        _run_conservative_kb_rebuild,
        _run_subgraph_rebuild,
        _serialize_graph_footprint,
    )

    footprint_payload_key = "rebuild_subgraph_footprints"

    def _serialized_footprints(payload: dict[str, Any]) -> list[dict[str, Any]]:
        raw_footprints = payload.get(footprint_payload_key)
        if not isinstance(raw_footprints, list):
            return []
        return [item for item in raw_footprints if isinstance(item, dict)]

    def _footprints_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            _deserialize_graph_footprint(item)
            for item in _serialized_footprints(payload)
        ]

    async def _persist_footprints(
        *, kb_id: str, job_id: str, footprints: list[dict[str, Any]]
    ) -> dict[str, Any]:
        updated_job = await job_service.update_job_payload_patch(
            kb_id,
            job_id,
            payload_patch={footprint_payload_key: footprints},
        )
        return updated_job.payload or {footprint_payload_key: footprints}

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
        strategy = payload.get("strategy")
        persisted_footprints = _serialized_footprints(payload)
        pre_delete_footprint = _merge_footprints(_footprints_from_payload(payload))
        document: Any | None = None
        item: dict[str, Any] | None = None
        try:
            document = await document_service.claim_delete(
                kb_id,
                str(document_id),
                job=job,
                delete_source_file=delete_source_file,
                delete_artifacts=delete_artifacts,
            )
        except MetadataRecordNotFoundError as exc:
            if (
                strategy != "rebuild_subgraph"
                or index_service is None
                or not persisted_footprints
            ):
                await job_service.transition_job(
                    kb_id,
                    job.id,
                    status="failed",
                    progress=1.0,
                    failed_items=1,
                    error_code="document_not_found",
                    error_message=str(exc),
                )
                return
            item = {
                "document_id": str(document_id),
                "status": "succeeded",
                "lightrag_doc_id": persisted_footprints[0].get("lightrag_doc_id")
                or payload.get("lightrag_doc_id"),
                "already_deleted_by_previous_attempt": True,
            }
        if document is not None:
            if strategy == "rebuild_subgraph" and index_service is not None:
                if not persisted_footprints:
                    footprint_rag = await registry.get(kb_id)
                    pre_delete_footprint = await _capture_graph_footprint(
                        rag=footprint_rag,
                        lightrag_doc_id=document.lightrag_doc_id,
                    )
                    persisted_footprints = [
                        _serialize_graph_footprint(
                            pre_delete_footprint,
                            document_id=document.id,
                            lightrag_doc_id=document.lightrag_doc_id,
                        )
                    ]
                    payload = await _persist_footprints(
                        kb_id=kb_id,
                        job_id=job.id,
                        footprints=persisted_footprints,
                    )
                else:
                    pre_delete_footprint = _merge_footprints(
                        _footprints_from_payload(payload)
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
        if item is None:
            await job_service.transition_job(
                kb_id,
                job.id,
                status="failed",
                progress=1.0,
                failed_items=1,
                error_code="worker_invalid_state",
                error_message="delete worker produced no item result",
            )
            return
        if item["status"] == "succeeded":
            result: dict[str, Any] = {
                "document_id": item["document_id"],
                "lightrag_doc_id": item.get("lightrag_doc_id"),
                "resumed_by_worker": True,
            }
            if strategy == "rebuild_kb" and index_service is not None:
                result["rebuild"] = await _run_conservative_kb_rebuild(
                    document_service=document_service,
                    index_service=index_service,
                    registry=registry,
                    kb_id=kb_id,
                )
            elif strategy == "rebuild_subgraph" and index_service is not None:
                result["rebuild"] = await _run_subgraph_rebuild(
                    document_service=document_service,
                    index_service=index_service,
                    registry=registry,
                    kb_id=kb_id,
                    footprint=pre_delete_footprint,
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
        strategy = payload.get("strategy")
        persisted_footprints = _serialized_footprints(payload)
        persisted_doc_ids = {
            str(item["document_id"])
            for item in persisted_footprints
            if isinstance(item.get("document_id"), str)
        }
        documents, claim_failures = await document_service.claim_batch_delete(
            kb_id,
            document_ids,
            job=job,
            delete_source_file=delete_source_file,
            delete_artifacts=delete_artifacts,
        )
        pre_delete_footprints: list[dict[str, Any]] = _footprints_from_payload(payload)
        if strategy == "rebuild_subgraph" and index_service is not None:
            footprint_rag = await registry.get(kb_id)
            next_persisted = list(persisted_footprints)
            for document in documents:
                if document.id in persisted_doc_ids:
                    continue
                footprint = await _capture_graph_footprint(
                    rag=footprint_rag,
                    lightrag_doc_id=document.lightrag_doc_id,
                )
                pre_delete_footprints.append(footprint)
                next_persisted.append(
                    _serialize_graph_footprint(
                        footprint,
                        document_id=document.id,
                        lightrag_doc_id=document.lightrag_doc_id,
                    )
                )
            if next_persisted != persisted_footprints:
                payload = await _persist_footprints(
                    kb_id=kb_id,
                    job_id=job.id,
                    footprints=next_persisted,
                )
                persisted_footprints = _serialized_footprints(payload)
                persisted_doc_ids = {
                    str(item["document_id"])
                    for item in persisted_footprints
                    if isinstance(item.get("document_id"), str)
                }
        item_results: list[dict[str, Any]] = []
        completed_items = 0
        failed_items = 0
        for failure in claim_failures:
            failed_document_id = failure.get("document_id")
            if (
                strategy == "rebuild_subgraph"
                and index_service is not None
                and failure.get("error_code") == "document_not_found"
                and isinstance(failed_document_id, str)
                and failed_document_id in persisted_doc_ids
            ):
                item_results.append(
                    {
                        "document_id": failed_document_id,
                        "status": "succeeded",
                        "already_deleted_by_previous_attempt": True,
                    }
                )
                completed_items += 1
                continue
            item_results.append(failure)
            failed_items += 1
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
        if strategy == "rebuild_kb" and completed_items > 0 and index_service is not None:
            final_result["rebuild"] = await _run_conservative_kb_rebuild(
                document_service=document_service,
                index_service=index_service,
                registry=registry,
                kb_id=kb_id,
            )
        elif strategy == "rebuild_subgraph" and completed_items > 0 and index_service is not None:
            final_result["rebuild"] = await _run_subgraph_rebuild(
                document_service=document_service,
                index_service=index_service,
                registry=registry,
                kb_id=kb_id,
                footprint=_merge_footprints(pre_delete_footprints),
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


def build_replace_executor(
    *,
    document_service: Any,
    registry: Any,
    job_service: JobService,
    index_service: Any | None = None,
) -> JobExecutor:
    """Executor that re-drives a queued single-document ``replace`` job.

    Replace jobs become worker-resumable because the replacement bytes are
    staged to disk at claim time (``stage_replacement_bytes``). After a crash,
    orphan recovery fails the in-flight job and resets the document to
    ``replace_failed``; a ``:retry`` puts the job back to ``queued`` and this
    executor rebuilds the ``DocumentReplacementSource`` from the staged file,
    re-claims the document into ``replacing`` and replays the same
    ``_execute_replace_document`` the route uses (delete old index → swap source
    → optional auto_parse/auto_index). If the staged bytes are missing (an older
    job that never staged), it fails cleanly with a clear error rather than
    guessing.
    """
    from lightrag.api.routers.kb_document_routes import _execute_replace_document
    from lightrag.api.document_lifecycle_service import SOURCE_TYPES

    allowed_source_types = set(SOURCE_TYPES)

    async def _run(job: JobRecord) -> None:
        kb_id = job.kb_id
        payload = job.payload or {}
        document_id = job.document_id or payload.get("document_id")
        if not document_id:
            await job_service.transition_job(
                kb_id,
                job.id,
                status="failed",
                progress=1.0,
                failed_items=1,
                error_code="worker_invalid_payload",
                error_message="replace job has no document_id",
            )
            return
        document_id = str(document_id)
        source_type = payload.get("source_type")
        if source_type not in allowed_source_types:
            await job_service.transition_job(
                kb_id,
                job.id,
                status="failed",
                progress=1.0,
                failed_items=1,
                error_code="worker_invalid_payload",
                error_message="replace job has invalid source_type",
            )
            return
        try:
            replacement = await document_service.load_staged_replacement(
                kb_id,
                document_id,
                job_id=job.id,
                source_name=str(payload.get("source_name") or "replacement"),
                source_hash=str(payload.get("source_hash") or ""),
                content_type=payload.get("content_type"),
                size_bytes=int(payload.get("size_bytes") or 0),
                source_type=source_type,
            )
        except ValueError as exc:
            # Staged bytes are present but corrupted (content hash no longer
            # matches the payload source_hash). Fail cleanly as not-resumable
            # rather than replaying the replace with wrong bytes, and drop the
            # unusable staging file. Mirrors the sync executor's guard.
            await job_service.transition_job(
                kb_id,
                job.id,
                status="failed",
                progress=1.0,
                failed_items=1,
                error_code="replace_not_resumable",
                error_message=str(exc),
            )
            await document_service.clear_staged_replacement(
                kb_id, document_id, job_id=job.id
            )
            return
        if replacement is None:
            await job_service.transition_job(
                kb_id,
                job.id,
                status="failed",
                progress=1.0,
                failed_items=1,
                error_code="replace_not_resumable",
                error_message=(
                    "replacement bytes were not staged for this job; "
                    "re-submit the replace request"
                ),
            )
            return
        document = await document_service.claim_replace(
            kb_id,
            document_id,
            job=job,
            replacement=replacement,
            delete_source_file=bool(payload.get("delete_source_file", True)),
            delete_artifacts=bool(payload.get("delete_artifacts", True)),
            delete_llm_cache=bool(payload.get("delete_llm_cache", False)),
            auto_parse=bool(payload.get("auto_parse", False)),
            auto_index=bool(payload.get("auto_index", False)),
            parser_engine=payload.get("parser_engine"),
            process_options=payload.get("process_options"),
            force_reparse=bool(payload.get("force_reparse", False)),
        )
        item = await _execute_replace_document(
            document_service=document_service,
            kb_id=kb_id,
            job=job,
            document=document,
            replacement=replacement,
            active_registry=registry,
            active_index_service=index_service,
            delete_source_file=bool(payload.get("delete_source_file", True)),
            delete_artifacts=bool(payload.get("delete_artifacts", True)),
            delete_llm_cache=bool(payload.get("delete_llm_cache", False)),
            auto_parse=bool(payload.get("auto_parse", False)),
            auto_index=bool(payload.get("auto_index", False)),
            parser_engine=payload.get("parser_engine"),
            process_options=payload.get("process_options"),
            force_reparse=bool(payload.get("force_reparse", False)),
        )
        if item["status"] == "succeeded":
            item["resumed_by_worker"] = True
            await job_service.transition_job(
                kb_id,
                job.id,
                status="succeeded",
                progress=1.0,
                completed_items=1,
                result=item,
            )
        else:
            await job_service.transition_job(
                kb_id,
                job.id,
                status="failed",
                progress=1.0,
                failed_items=1,
                error_code=item.get("error_code", "replace_failed"),
                error_message=item.get("error_message", "replace failed"),
                result=item,
            )
        await document_service.clear_staged_replacement(
            kb_id, document_id, job_id=job.id
        )

    return _run


def build_sync_executor(
    *,
    document_service: Any,
    registry: Any,
    job_service: JobService,
    index_service: Any | None = None,
) -> JobExecutor:
    """Executor that re-drives queued aggregate ``documents:sync`` jobs.

    Sync is resumable only because the route stages every upload to disk before
    creating the queued aggregate job, and the job payload persists each staged
    item's source key/name/hash/content type/options. The worker reconstructs
    ``DocumentSourceInput`` values from those files and then calls the same
    per-item sync helper the route uses. Missing staged files fail clearly as
    ``sync_not_resumable`` instead of guessing from incomplete state.
    """
    from lightrag.api.routers.kb_document_routes import (
        _execute_build_plan_batch,
        _execute_sync_item,
        _sync_failure_message,
        _sync_job_result,
    )
    from lightrag.api.document_lifecycle_service import SOURCE_TYPES

    allowed_source_types = set(SOURCE_TYPES)

    def _invalid_payload_message(message: str) -> str:
        return f"sync job has invalid resumable payload: {message}"

    async def _clear_staged_sync_if_terminal(
        kb_id: str, batch_id: Any | None
    ) -> None:
        if isinstance(batch_id, str) and batch_id:
            await document_service.clear_staged_sync_sources(kb_id, batch_id=batch_id)

    async def _fail_invalid_payload(
        job: JobRecord, message: str, *, batch_id: Any | None = None
    ) -> None:
        await job_service.transition_job(
            job.kb_id,
            job.id,
            status="failed",
            progress=1.0,
            failed_items=1,
            error_code="worker_invalid_payload",
            error_message=_invalid_payload_message(message),
        )
        await _clear_staged_sync_if_terminal(job.kb_id, batch_id)

    async def _run(job: JobRecord) -> None:
        kb_id = job.kb_id
        payload = job.payload or {}
        raw_items = payload.get("items")
        batch_id = job.batch_id or payload.get("batch_id")
        if job.document_id is not None:
            await _fail_invalid_payload(
                job, "sync jobs must be aggregate jobs", batch_id=batch_id
            )
            return
        if not isinstance(batch_id, str) or not batch_id:
            await _fail_invalid_payload(job, "missing batch_id")
            return
        if not isinstance(raw_items, list) or not raw_items:
            await _fail_invalid_payload(job, "missing items", batch_id=batch_id)
            return

        prepared_sources: list[dict[str, Any]] = []
        for index, raw_item in enumerate(raw_items):
            if not isinstance(raw_item, dict):
                await _fail_invalid_payload(
                    job, "item is not an object", batch_id=batch_id
                )
                return
            source_key = raw_item.get("source_key")
            source_name = raw_item.get("source_name")
            source_hash = raw_item.get("source_hash")
            if not isinstance(source_key, str) or not source_key:
                await _fail_invalid_payload(
                    job, "item missing source_key", batch_id=batch_id
                )
                return
            if not isinstance(source_name, str) or not source_name:
                await _fail_invalid_payload(
                    job, "item missing source_name", batch_id=batch_id
                )
                return
            if not isinstance(source_hash, str) or not source_hash:
                await _fail_invalid_payload(
                    job, "item missing source_hash", batch_id=batch_id
                )
                return
            content_type = raw_item.get("content_type")
            if content_type is not None and not isinstance(content_type, str):
                await _fail_invalid_payload(
                    job, "item content_type must be a string", batch_id=batch_id
                )
                return
            source_type = raw_item.get("source_type")
            if source_type not in allowed_source_types:
                await _fail_invalid_payload(
                    job, "item source_type is not supported", batch_id=batch_id
                )
                return
            try:
                source = await document_service.load_staged_sync_source(
                    kb_id,
                    batch_id=batch_id,
                    item_index=index,
                    source_name=source_name,
                    content_type=content_type,
                    metadata={"source_key": source_key},
                    expected_hash=source_hash,
                    source_type=source_type,
                )
            except ValueError as exc:
                await job_service.transition_job(
                    kb_id,
                    job.id,
                    status="failed",
                    progress=1.0,
                    failed_items=1,
                    error_code="sync_not_resumable",
                    error_message=str(exc),
                )
                await _clear_staged_sync_if_terminal(kb_id, batch_id)
                return
            if source is None:
                await job_service.transition_job(
                    kb_id,
                    job.id,
                    status="failed",
                    progress=1.0,
                    failed_items=1,
                    error_code="sync_not_resumable",
                    error_message=(
                        "sync source bytes were not staged for this job; "
                        "re-submit the sync request"
                    ),
                )
                await _clear_staged_sync_if_terminal(kb_id, batch_id)
                return
            prepared_sources.append(
                {
                    "source_key": source_key,
                    "source": source,
                    "source_hash": source_hash,
                    "content_type": content_type,
                    "size_bytes": int(raw_item.get("size_bytes") or len(source.content)),
                }
            )

        item_results: list[dict[str, Any]] = []
        completed_items = 0
        failed_items = 0
        skipped_items = 0
        existing_by_source_key = await document_service.get_documents_by_source_keys(
            kb_id, [str(item["source_key"]) for item in prepared_sources]
        )

        # Phase 1: per-item sync runs concurrently (bounded by
        # MAX_PARALLEL_PARSE_MINERU); auto_index builds are deferred so the
        # whole batch drains the pipeline once (overlapping analyze / extract
        # / merge across documents) in Phase 2.
        rag = await registry.get(kb_id) if prepared_sources else None
        parse_concurrency = max(
            1, int(getattr(rag, "max_parallel_parse_mineru", 1) or 1)
        )
        parse_sem = asyncio.Semaphore(parse_concurrency)

        async def _do_one_sync_item(prepared: dict[str, Any]) -> dict[str, Any]:
            async with parse_sem:
                item, _ = await _execute_sync_item(
                    document_service=document_service,
                    kb_id=kb_id,
                    job=job,
                    prepared=prepared,
                    existing_by_source_key=existing_by_source_key,
                    active_registry=registry,
                    active_index_service=index_service,
                    rag=rag,
                    auto_parse=bool(payload.get("auto_parse", False)),
                    auto_index=bool(payload.get("auto_index", False)),
                    parser_engine=payload.get("parser_engine"),
                    process_options=payload.get("process_options"),
                    force_reparse=bool(payload.get("force_reparse", False)),
                    delete_source_file=bool(payload.get("delete_source_file", True)),
                    delete_artifacts=bool(payload.get("delete_artifacts", True)),
                    delete_llm_cache=bool(payload.get("delete_llm_cache", False)),
                    defer_build=True,
                )
            return item

        sync_items: list[dict[str, Any]] = []
        if prepared_sources:
            raw_sync = await asyncio.gather(
                *[_do_one_sync_item(prepared) for prepared in prepared_sources],
                return_exceptions=True,
            )
            # return_exceptions=True keeps Phase 2 reachable (releasing any
            # build_queued claim) even on an unexpected BaseException; map any
            # exception back to a failed item via positional zip.
            for prepared, outcome in zip(prepared_sources, raw_sync):
                if isinstance(outcome, BaseException):
                    sync_items.append(
                        {
                            "source_key": str(prepared["source_key"]),
                            "source_name": getattr(
                                prepared.get("source"), "source_name", ""
                            ),
                            "source_hash": str(prepared.get("source_hash") or ""),
                            "action": "unknown",
                            "status": "failed",
                            "error_code": "sync_item_failed",
                            "error_message": str(outcome),
                        }
                    )
                else:
                    sync_items.append(outcome)

        # Phase 2: batch-build any deferred auto_index plans in one drain.
        if index_service is not None and rag is not None:
            deferred_pairs: list[tuple[Any, dict[str, Any]]] = []
            for item in sync_items:
                build_plan = item.pop("_deferred_build_plan", None)
                if build_plan is not None:
                    deferred_pairs.append((build_plan, item))
            if deferred_pairs:
                batch_results = await _execute_build_plan_batch(
                    index_service=index_service,
                    kb_id=kb_id,
                    job_id=job.id,
                    rag=rag,
                    plans=[bp for bp, _ in deferred_pairs],
                    job_service=job_service,
                )
                for build_plan, item in deferred_pairs:
                    build_item = batch_results.get(build_plan.document.id)
                    if build_item is None:
                        item.update(
                            {
                                "status": "failed",
                                "error_code": "build_failed",
                                "error_message": "Build result missing from batch",
                            }
                        )
                        continue
                    item["build_result"] = build_item
                    if build_item["status"] not in {"succeeded", "cancelled"}:
                        item.update(
                            {
                                "status": "failed",
                                "error_code": build_item.get(
                                    "error_code", "build_failed"
                                ),
                                "error_message": build_item.get(
                                    "error_message", "Document sync build failed"
                                ),
                            }
                        )

        for item in sync_items:
            item_results.append(item)
            if item["status"] == "failed":
                failed_items += 1
            else:
                completed_items += 1
                if item["status"] == "skipped":
                    skipped_items += 1

        final_result = _sync_job_result(
            batch_id=batch_id,
            total_items=len(prepared_sources),
            completed_items=completed_items,
            failed_items=failed_items,
            skipped_items=skipped_items,
            items=item_results,
        )
        final_result["resumed_by_worker"] = True
        await job_service.transition_job(
            kb_id,
            job.id,
            status="succeeded" if failed_items == 0 else "failed",
            progress=1.0,
            completed_items=completed_items,
            failed_items=failed_items,
            result=final_result,
            error_code=None if failed_items == 0 else "partial_sync_failed",
            error_message=None
            if failed_items == 0
            else _sync_failure_message(failed_items, len(prepared_sources)),
        )
        await document_service.clear_staged_sync_sources(kb_id, batch_id=batch_id)

    return _run


def build_clear_kb_executor(*, deletion_service: Any) -> JobExecutor:
    """Executor that re-drives a queued ``clear_kb`` (KB hard-delete) job.

    The job carries only ``kb_id`` / ``workspace`` and the destructive clear
    (evict instance, drop working/input dirs, purge control-plane rows) is
    idempotent, so a queued ``clear_kb`` job left by restart recovery can be
    safely re-driven to completion. ``resume_hard_delete`` itself owns the
    terminal job transition (succeeded/failed); the worker's backstop only
    fires if it raises.
    """

    async def _run(job: JobRecord) -> None:
        await deletion_service.resume_hard_delete(job)

    return _run


def build_agent_profile_executor(*, profile_service: Any) -> JobExecutor:
    """Executor that re-drives queued KB Agent profile generation jobs."""

    async def _run(job: JobRecord) -> None:
        await profile_service.run_job(job)

    return _run
