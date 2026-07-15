from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from lightrag.api.job_service import (
    RESUMABLE_KB_MUTATION_JOB_TYPES,
    JobService,
)
from lightrag.api.job_worker import JobWorker
from lightrag.api.kb_service import (
    KnowledgeBaseService,
    utc_now_iso,
)
from lightrag.api.metadata_store import (
    DocumentRecord,
    JobRecord,
    KBLifecycleConflictError,
    SQLiteMetadataStore,
)

pytestmark = pytest.mark.offline


_UNSET = object()


def _job(
    kb_id: str,
    job_id: str,
    *,
    job_type: str = "parse",
    status: str = "queued",
    document_id: object = _UNSET,
    queued_at: str | None = None,
    generation: object = _UNSET,
) -> JobRecord:
    now = utc_now_iso()
    # Real single-document jobs usually carry a document_id; default to the job
    # id so fixtures match production. Pass document_id=None explicitly for
    # aggregate-job fixtures such as batch-delete.
    resolved_document_id: str | None
    if document_id is _UNSET:
        resolved_document_id = f"doc_{job_id}"
    elif document_id is None or isinstance(document_id, str):
        resolved_document_id = document_id
    else:
        resolved_document_id = str(document_id)
    payload: dict[str, Any] = (
        {"document_id": resolved_document_id} if resolved_document_id else {}
    )
    if generation is not _UNSET:
        payload["kb_generation"] = generation
    return JobRecord(
        id=job_id,
        kb_id=kb_id,
        workspace=f"kb_{kb_id}",
        batch_id=None,
        document_id=resolved_document_id,
        job_type=job_type,
        status=status,
        stage=None,
        progress=0.0,
        total_items=1,
        completed_items=0,
        failed_items=0,
        idempotency_key=None,
        config_version_id=None,
        config_hash=None,
        retry_count=0,
        max_retries=3,
        payload=payload,
        result=None,
        error_code=None,
        error_message=None,
        created_at=now,
        updated_at=now,
        queued_at=queued_at or now,
        started_at=None,
        finished_at=None,
        cancelled_at=None,
    )


def _document(kb_id: str, document_id: str) -> DocumentRecord:
    now = utc_now_iso()
    return DocumentRecord(
        id=document_id,
        kb_id=kb_id,
        workspace=f"kb_{kb_id}",
        lightrag_doc_id=None,
        source_type="upload",
        source_name=f"{document_id}.pdf",
        source_uri=f"/tmp/{document_id}.pdf",
        source_hash="sha256:seed",
        content_type="application/pdf",
        size_bytes=1,
        parser_hash=None,
        index_hash=None,
        status="uploaded",
        enabled=True,
        archived=False,
        chunks_count=None,
        entity_count=None,
        relation_count=None,
        error_code=None,
        error_message=None,
        metadata={},
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )


async def _create_job(store: SQLiteMetadataStore, job: JobRecord) -> JobRecord:
    """Create a job, first seeding the referenced document row to satisfy the
    jobs.document_id -> documents(id) foreign key (single-doc jobs only)."""
    if job.document_id is not None:
        document = _document(job.kb_id, job.document_id)

        def _seed(conn):
            try:
                store._insert_document(conn, document)
            except Exception:
                # Already seeded by an earlier job referencing the same doc.
                pass

        await store._write(_seed)
    return await store.create_job(job)


async def _make_store(tmp_path: Path) -> SQLiteMetadataStore:
    store = SQLiteMetadataStore(tmp_path / "metadata.sqlite3")
    await store.initialize()
    return store


@pytest.mark.asyncio
async def test_claim_next_worker_job_is_single_winner(tmp_path: Path):
    store = await _make_store(tmp_path)
    await _create_job(store, _job("kb_a", "job_1", job_type="parse"))

    first = await store.claim_next_worker_job(job_types=["parse"], max_queued_at=None)
    assert first is not None
    assert first.id == "job_1"
    assert first.status == "running"

    # A second claim finds nothing — the job is no longer queued.
    second = await store.claim_next_worker_job(
        job_types=["parse"], max_queued_at=None
    )
    assert second is None


@pytest.mark.asyncio
async def test_claim_respects_job_type_filter(tmp_path: Path):
    store = await _make_store(tmp_path)
    await _create_job(store, _job("kb_a", "job_build", job_type="build_kg"))

    # Worker only handles "parse" — build job is not claimed.
    claimed = await store.claim_next_worker_job(
        job_types=["parse"], max_queued_at=None
    )
    assert claimed is None

    claimed_build = await store.claim_next_worker_job(
        job_types=["build_kg"], max_queued_at=None
    )
    assert claimed_build is not None
    assert claimed_build.id == "job_build"


@pytest.mark.asyncio
async def test_claim_allows_aggregate_resumable_jobs(tmp_path: Path):
    """Aggregate jobs whose document_ids/options live entirely in the persisted
    payload (multi-file upload auto_parse, batch-parse/build/reindex, sync) ARE
    claimable by the durable worker: the source files / artifacts are persisted
    before the job runs, so the worker can re-drive them from document_ids.
    """
    store = await _make_store(tmp_path)
    # Aggregate parse job (oldest): document_id=None, payload carries doc ids.
    await store.create_job(
        _job(
            "kb_a",
            "agg_parse",
            job_type="parse",
            document_id=None,
            queued_at="2026-05-29T09:00:00+00:00",
        )
    )
    # A single-document parse job created later.
    single = _job(
        "kb_a",
        "single_parse",
        job_type="parse",
        queued_at="2026-05-29T10:00:00+00:00",
    )
    await _create_job(store, single)

    # Oldest-first ordering claims the aggregate job before the single one.
    first = await store.claim_next_worker_job(
        job_types=["parse"], max_queued_at=None
    )
    assert first is not None
    assert first.id == "agg_parse"
    assert first.status == "running"

    second = await store.claim_next_worker_job(
        job_types=["parse"], max_queued_at=None
    )
    assert second is not None
    assert second.id == "single_parse"


@pytest.mark.asyncio
async def test_claim_skips_non_resumable_aggregate_jobs(tmp_path: Path):
    """Aggregate types NOT re-drivable from persisted state (multi-file
    ``upload`` aggregate, ``replace`` request bytes) must NOT be claimed. The
    multi-file ``upload`` (no auto_parse) carries no parse work and ``replace``
    needs request-uploaded bytes, so neither is worker-resumable as an
    aggregate document_id=None job.
    """
    store = await _make_store(tmp_path)
    await store.create_job(
        _job("kb_a", "agg_upload", job_type="upload", document_id=None)
    )

    claimed = await store.claim_next_worker_job(
        job_types=["upload"], max_queued_at=None
    )
    assert claimed is None
    agg = await store.get_job("kb_a", "agg_upload")
    assert agg.status == "queued"


@pytest.mark.asyncio
async def test_claim_allows_aggregate_delete_jobs(tmp_path: Path):
    """Batch delete has a persisted document_ids payload, so it is worker-safe."""
    store = await _make_store(tmp_path)
    await store.create_job(
        _job("kb_a", "agg_delete", job_type="delete", document_id=None)
    )

    claimed = await store.claim_next_worker_job(
        job_types=["delete"], max_queued_at=None
    )

    assert claimed is not None
    assert claimed.id == "agg_delete"
    assert claimed.status == "running"


@pytest.mark.asyncio
async def test_claim_allows_aggregate_sync_jobs(tmp_path: Path):
    """Batch sync has staged source bytes plus source keys/options in payload."""
    store = await _make_store(tmp_path)
    await store.create_job(
        _job("kb_a", "agg_sync", job_type="sync", document_id=None)
    )

    claimed = await store.claim_next_worker_job(
        job_types=["sync"], max_queued_at=None
    )

    assert claimed is not None
    assert claimed.id == "agg_sync"
    assert claimed.status == "running"


@pytest.mark.asyncio
async def test_claim_grace_window_excludes_fresh_jobs(tmp_path: Path):
    store = await _make_store(tmp_path)
    # Job queued "now"; a cutoff in the past must not claim it.
    await _create_job(store, _job("kb_a", "fresh", job_type="parse"))
    past_cutoff = "2000-01-01T00:00:00+00:00"
    assert (
        await store.claim_next_worker_job(
            job_types=["parse"], max_queued_at=past_cutoff
        )
        is None
    )
    # A future cutoff (grace elapsed) does claim it.
    future_cutoff = "2999-01-01T00:00:00+00:00"
    claimed = await store.claim_next_worker_job(
        job_types=["parse"], max_queued_at=future_cutoff
    )
    assert claimed is not None
    assert claimed.id == "fresh"


@pytest.mark.asyncio
async def test_claim_orders_oldest_first(tmp_path: Path):
    store = await _make_store(tmp_path)
    await _create_job(
        store, _job("kb_a", "newer", queued_at="2026-05-29T10:00:00+00:00")
    )
    await _create_job(
        store, _job("kb_a", "older", queued_at="2026-05-29T09:00:00+00:00")
    )
    claimed = await store.claim_next_worker_job(
        job_types=["parse"], max_queued_at=None
    )
    assert claimed is not None
    assert claimed.id == "older"


@pytest.mark.asyncio
async def test_worker_poll_once_dispatches_to_executor(tmp_path: Path):
    store = await _make_store(tmp_path)
    kb_service = KnowledgeBaseService(tmp_path / "kb.json")
    await kb_service.initialize()
    await kb_service.create(kb_id="kb_worker", name="Worker")
    record = await kb_service.get("kb_worker")
    job_service = JobService(kb_service, store)
    await _create_job(
        store,
        _job(
            record.id,
            "job_exec",
            job_type="parse",
            generation=record.generation,
        ),
    )

    executed: list[str] = []

    async def fake_parse_executor(job: JobRecord) -> None:
        executed.append(job.id)
        assert job.status == "running"  # already claimed before dispatch
        await job_service.transition_job(
            job.kb_id, job.id, status="succeeded", progress=1.0, completed_items=1
        )

    worker = JobWorker(
        job_service,
        executors={"parse": fake_parse_executor},
        claim_grace_seconds=0.0,
    )
    claimed = await worker.poll_once()
    assert claimed is not None and claimed.id == "job_exec"
    assert executed == ["job_exec"]
    refreshed = await job_service.get_job("kb_worker", "job_exec")
    assert refreshed.status == "succeeded"

    # Nothing left to claim.
    assert await worker.poll_once() is None


@pytest.mark.asyncio
async def test_worker_consumes_retried_job(tmp_path: Path):
    """:retry resets a failed job to queued; the worker then auto-consumes it."""
    store = await _make_store(tmp_path)
    kb_service = KnowledgeBaseService(tmp_path / "kb.json")
    await kb_service.initialize()
    await kb_service.create(kb_id="kb_retry", name="Retry")
    record = await kb_service.get("kb_retry")
    job_service = JobService(kb_service, store)

    await _create_job(
        store,
        _job(
            record.id,
            "job_retry",
            job_type="parse",
            generation=record.generation,
        ),
    )
    await store.transition_job(record.id, "job_retry", status="running")
    await store.transition_job(
        record.id, "job_retry", status="failed", error_code="boom"
    )
    # Simulate the :retry API resetting the job back to queued.
    await store.reset_job_for_retry(record.id, "job_retry", new_idempotency_key=None)
    refreshed = await job_service.get_job("kb_retry", "job_retry")
    assert refreshed.status == "queued"
    assert refreshed.retry_count == 1

    runs: list[str] = []

    async def fake_executor(job: JobRecord) -> None:
        runs.append(job.id)
        await job_service.transition_job(
            job.kb_id, job.id, status="succeeded", progress=1.0
        )

    worker = JobWorker(
        job_service,
        executors={"parse": fake_executor},
        claim_grace_seconds=0.0,
    )
    assert (await worker.poll_once()) is not None
    assert runs == ["job_retry"]
    final = await job_service.get_job("kb_retry", "job_retry")
    assert final.status == "succeeded"


@pytest.mark.asyncio
async def test_worker_executor_error_marks_job_failed(tmp_path: Path):
    store = await _make_store(tmp_path)
    kb_service = KnowledgeBaseService(tmp_path / "kb.json")
    await kb_service.initialize()
    await kb_service.create(kb_id="kb_err", name="Err")
    record = await kb_service.get("kb_err")
    job_service = JobService(kb_service, store)
    await _create_job(
        store,
        _job(
            record.id,
            "job_bad",
            job_type="parse",
            generation=record.generation,
        ),
    )

    async def boom(job: JobRecord) -> None:
        raise RuntimeError("executor exploded")

    worker = JobWorker(
        job_service,
        executors={"parse": boom},
        claim_grace_seconds=0.0,
    )
    await worker.poll_once()
    refreshed = await job_service.get_job("kb_err", "job_bad")
    assert refreshed.status == "failed"
    assert refreshed.error_code == "worker_executor_error"


@pytest.mark.asyncio
async def test_recovery_leaves_resumable_queued_jobs(tmp_path: Path):
    """With the worker enabled, queued resumable jobs survive restart recovery."""
    store = await _make_store(tmp_path)
    # A queued parse job (resumable) + a queued delete job (not in the
    # resumable set for this run) +
    # a running parse job (mid-flight, cannot resume).
    await _create_job(store, _job("kb_r", "queued_parse", job_type="parse"))
    await _create_job(store, _job("kb_r", "queued_delete", job_type="delete"))
    await _create_job(store, _job("kb_r", "running_parse", job_type="parse"))
    await store.transition_job("kb_r", "running_parse", status="running")

    recovered = await store.recover_orphan_jobs(resumable_job_types={"parse"})
    recovered_ids = {job.id for job in recovered}
    # queued_parse is left for the worker; the other two are failed.
    assert "queued_parse" not in recovered_ids
    assert recovered_ids == {"queued_delete", "running_parse"}

    survivor = await store.get_job("kb_r", "queued_parse")
    assert survivor.status == "queued"
    failed_running = await store.get_job("kb_r", "running_parse")
    assert failed_running.status == "failed"


@pytest.mark.asyncio
async def test_recovery_leaves_resumable_delete_queued(tmp_path: Path):
    """When 'delete' is a resumable type, queued single-doc and batch delete
    jobs survive restart recovery because their payload is enough to re-drive
    them."""
    store = await _make_store(tmp_path)
    await _create_job(store, _job("kb_d", "queued_delete", job_type="delete"))
    await store.create_job(
        _job("kb_d", "queued_batch_delete", job_type="delete", document_id=None)
    )
    await store.create_job(
        _job("kb_d", "queued_sync", job_type="sync", document_id=None)
    )
    await _create_job(store, _job("kb_d", "queued_upload", job_type="upload"))

    recovered = await store.recover_orphan_jobs(
        resumable_job_types={"parse", "build_kg", "reindex", "delete", "sync"}
    )
    recovered_ids = {job.id for job in recovered}
    # Delete/sync are kept queued for the worker; upload (needs request bytes) fails.
    assert "queued_delete" not in recovered_ids
    assert "queued_batch_delete" not in recovered_ids
    assert "queued_sync" not in recovered_ids
    assert "queued_upload" in recovered_ids

    single = await store.get_job("kb_d", "queued_delete")
    batch = await store.get_job("kb_d", "queued_batch_delete")
    sync = await store.get_job("kb_d", "queued_sync")
    assert single.status == "queued"
    assert batch.status == "queued"
    assert sync.status == "queued"


@pytest.mark.asyncio
async def test_recovery_leaves_aggregate_delete_when_delete_resumable(tmp_path: Path):
    """An aggregate delete job is now recoverable because the delete executor
    can replay its persisted document_ids/options payload."""
    store = await _make_store(tmp_path)
    await store.create_job(
        _job("kb_b", "batch_delete", job_type="delete", document_id=None)
    )
    await _create_job(store, _job("kb_b", "single_delete", job_type="delete"))

    recovered = await store.recover_orphan_jobs(
        resumable_job_types={"parse", "build_kg", "reindex", "delete"}
    )
    recovered_ids = {job.id for job in recovered}
    assert "batch_delete" not in recovered_ids
    assert "single_delete" not in recovered_ids
    assert (await store.get_job("kb_b", "batch_delete")).status == "queued"
    assert (await store.get_job("kb_b", "single_delete")).status == "queued"


@pytest.mark.asyncio
async def test_recovery_skips_live_owner_then_recovers_stale_job_once(
    tmp_path: Path,
):
    db_path = tmp_path / "owner-aware-recovery.sqlite3"
    owner_store = SQLiteMetadataStore(db_path)
    recovery_store = SQLiteMetadataStore(db_path)
    await owner_store.initialize()
    await recovery_store.initialize()
    document = _document("kb_owner", "doc_owner")
    document.status = "parsing"
    await owner_store.create_documents_and_job(
        [document],
        _job(
            "kb_owner",
            "job_owner",
            job_type="parse",
            document_id="doc_owner",
        ),
    )
    await owner_store.transition_job("kb_owner", "job_owner", status="running")

    async with owner_store.job_execution_guard("job_owner") as acquired:
        assert acquired is True
        recovered = await recovery_store.recover_orphan_jobs(grace_seconds=0)
        assert recovered == []
        assert (await recovery_store.get_job("kb_owner", "job_owner")).status == (
            "running"
        )
        assert (await recovery_store.get_document("kb_owner", "doc_owner")).status == (
            "parsing"
        )

    first, second = await asyncio.gather(
        owner_store.recover_orphan_jobs(grace_seconds=0),
        recovery_store.recover_orphan_jobs(grace_seconds=0),
    )
    assert sum(len(items) for items in (first, second)) == 1
    assert (await owner_store.get_job("kb_owner", "job_owner")).status == "failed"
    assert (
        await owner_store.get_document("kb_owner", "doc_owner")
    ).status == "parse_failed"


@pytest.mark.asyncio
async def test_recovery_grace_protects_claim_to_owner_lock_window(tmp_path: Path):
    store = await _make_store(tmp_path)
    await _create_job(store, _job("kb_gap", "job_gap", job_type="parse"))
    claimed = await store.claim_next_worker_job(
        job_types=["parse"], max_queued_at=None
    )
    assert claimed is not None and claimed.status == "running"

    assert await store.recover_orphan_jobs(grace_seconds=30) == []
    assert (await store.get_job("kb_gap", "job_gap")).status == "running"
    recovered = await store.recover_orphan_jobs(grace_seconds=0)
    assert [job.id for job in recovered] == ["job_gap"]


@pytest.mark.asyncio
async def test_two_store_workers_execute_claimed_job_at_most_once(tmp_path: Path):
    db_path = tmp_path / "two-worker-owner.sqlite3"
    first_store = SQLiteMetadataStore(db_path)
    second_store = SQLiteMetadataStore(db_path)
    await first_store.initialize()
    await second_store.initialize()
    kb_service = KnowledgeBaseService(tmp_path / "two-worker-owner-kb.json")
    await kb_service.initialize()
    record = await kb_service.create(kb_id="kb_two_workers", name="Two Workers")
    await first_store.activate_kb_generation(record.id, record.generation)
    first_service = JobService(kb_service, first_store)
    second_service = JobService(kb_service, second_store)
    await _create_job(
        first_store,
        _job(
            record.id,
            "job_two_workers",
            job_type="parse",
            generation=record.generation,
        ),
    )
    executor_calls = 0

    async def executor(job: JobRecord) -> None:
        nonlocal executor_calls
        executor_calls += 1
        await first_store.transition_job(job.kb_id, job.id, status="succeeded")

    first_worker = JobWorker(
        first_service,
        executors={"parse": executor},
        claim_grace_seconds=0,
    )
    second_worker = JobWorker(
        second_service,
        executors={"parse": executor},
        claim_grace_seconds=0,
    )
    await asyncio.gather(first_worker.poll_once(), second_worker.poll_once())

    assert executor_calls == 1
    assert (
        await first_store.get_job(record.id, "job_two_workers")
    ).status == "succeeded"


@pytest.mark.asyncio
async def test_worker_rechecks_claim_identity_after_waiting_for_job_owner(
    tmp_path: Path,
):
    db_path = tmp_path / "claim-identity.sqlite3"
    worker_store = SQLiteMetadataStore(db_path)
    owner_store = SQLiteMetadataStore(db_path)
    await worker_store.initialize()
    await owner_store.initialize()
    kb_service = KnowledgeBaseService(tmp_path / "claim-identity-kb.json")
    await kb_service.initialize()
    record = await kb_service.create(kb_id="kb_claim_identity", name="Identity")
    await worker_store.activate_kb_generation(record.id, record.generation)
    job_service = JobService(kb_service, worker_store)
    await _create_job(
        worker_store,
        _job(
            record.id,
            "job_claim_identity",
            job_type="parse",
            generation=record.generation,
        ),
    )
    executor_calls = 0

    async def executor(_job_record: JobRecord) -> None:
        nonlocal executor_calls
        executor_calls += 1

    worker = JobWorker(
        job_service,
        executors={"parse": executor},
        claim_grace_seconds=0,
    )

    async with owner_store.job_execution_guard("job_claim_identity"):
        worker_task = asyncio.create_task(worker.poll_once())
        for _ in range(50):
            first_claim = await owner_store.get_job(
                record.id, "job_claim_identity"
            )
            if first_claim.status == "running":
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("worker did not reach the claim-to-owner-lock window")

        await owner_store.transition_job(
            record.id, "job_claim_identity", status="failed"
        )
        await owner_store.reset_job_for_retry(
            record.id,
            "job_claim_identity",
            new_idempotency_key=None,
        )
        second_claim = await owner_store.claim_next_worker_job(
            job_types=["parse"], max_queued_at=None
        )
        assert second_claim is not None
        assert second_claim.retry_count == 1

    await asyncio.wait_for(worker_task, timeout=2)
    assert executor_calls == 0
    assert (
        await owner_store.get_job(record.id, "job_claim_identity")
    ).status == "running"


@pytest.mark.asyncio
async def test_cancelled_executor_releases_job_owner_for_recovery(tmp_path: Path):
    db_path = tmp_path / "cancelled-owner.sqlite3"
    worker_store = SQLiteMetadataStore(db_path)
    recovery_store = SQLiteMetadataStore(db_path)
    await worker_store.initialize()
    await recovery_store.initialize()
    kb_service = KnowledgeBaseService(tmp_path / "cancelled-owner-kb.json")
    await kb_service.initialize()
    record = await kb_service.create(kb_id="kb_cancel_owner", name="Cancel Owner")
    await worker_store.activate_kb_generation(record.id, record.generation)
    job_service = JobService(kb_service, worker_store)
    await _create_job(
        worker_store,
        _job(
            record.id,
            "job_cancel_owner",
            job_type="parse",
            generation=record.generation,
        ),
    )
    entered = asyncio.Event()
    never_release = asyncio.Event()

    async def executor(_job_record: JobRecord) -> None:
        entered.set()
        await never_release.wait()

    worker = JobWorker(
        job_service,
        executors={"parse": executor},
        claim_grace_seconds=0,
    )
    task = asyncio.create_task(worker.poll_once())
    await asyncio.wait_for(entered.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    async with recovery_store.job_execution_guard(
        "job_cancel_owner", wait=False
    ) as acquired:
        assert acquired is True
    recovered = await recovery_store.recover_orphan_jobs(grace_seconds=0)
    assert [job.id for job in recovered] == ["job_cancel_owner"]


@pytest.mark.asyncio
async def test_running_worker_periodically_recovers_newly_stale_jobs(tmp_path: Path):
    store = await _make_store(tmp_path)
    kb_service = KnowledgeBaseService(tmp_path / "periodic-recovery-kb.json")
    await kb_service.initialize()
    record = await kb_service.create(kb_id="kb_periodic", name="Periodic")
    await store.activate_kb_generation(record.id, record.generation)
    job_service = JobService(kb_service, store)
    await _create_job(
        store,
        _job(
            record.id,
            "job_periodic",
            job_type="parse",
            generation=record.generation,
        ),
    )
    await store.transition_job(record.id, "job_periodic", status="running")
    executor_calls = 0

    async def executor(_job_record: JobRecord) -> None:
        nonlocal executor_calls
        executor_calls += 1

    worker = JobWorker(
        job_service,
        executors={"parse": executor},
        poll_interval_seconds=0.05,
        claim_grace_seconds=0,
        recovery_interval_seconds=0.05,
        recovery_grace_seconds=0,
    )
    worker.start()
    try:
        for _ in range(50):
            if (await store.get_job(record.id, "job_periodic")).status == "failed":
                break
            await asyncio.sleep(0.02)
        else:
            pytest.fail("periodic recovery did not fail the orphaned running job")
    finally:
        await worker.stop()

    assert executor_calls == 0


@pytest.mark.asyncio
async def test_worker_run_loop_consumes_then_stops(tmp_path: Path):
    """Drive the real background loop: start() schedules _run_loop, which drains
    a queued job to completion, then stop() signals the loop and awaits it."""
    store = await _make_store(tmp_path)
    kb_service = KnowledgeBaseService(tmp_path / "kb.json")
    await kb_service.initialize()
    await kb_service.create(kb_id="kb_loop", name="Loop")
    record = await kb_service.get("kb_loop")
    job_service = JobService(kb_service, store)
    await _create_job(
        store,
        _job(
            record.id,
            "loop_job",
            job_type="parse",
            generation=record.generation,
        ),
    )

    done = asyncio.Event()

    async def fake_executor(job: JobRecord) -> None:
        await job_service.transition_job(
            job.kb_id, job.id, status="succeeded", progress=1.0, completed_items=1
        )
        done.set()

    worker = JobWorker(
        job_service,
        executors={"parse": fake_executor},
        poll_interval_seconds=0.05,
        claim_grace_seconds=0.0,
    )
    worker.start()
    # start() is idempotent — a second call must not spawn a second loop.
    worker.start()
    try:
        await asyncio.wait_for(done.wait(), timeout=5.0)
    finally:
        await worker.stop()

    final = await job_service.get_job("kb_loop", "loop_job")
    assert final.status == "succeeded"
    # After stop(), the loop task is cleared and nothing else is claimable.
    assert await worker.poll_once() is None


@pytest.mark.asyncio
async def test_job_service_stamps_every_resumable_kb_mutation_type(tmp_path: Path):
    store = await _make_store(tmp_path)
    kb_service = KnowledgeBaseService(tmp_path / "stamp-kb.json")
    await kb_service.initialize()
    record = await kb_service.create(kb_id="kb_stamp", name="Stamp")
    await store.activate_kb_generation(record.id, record.generation)
    job_service = JobService(kb_service, store)

    created = []
    for job_type in sorted(RESUMABLE_KB_MUTATION_JOB_TYPES):
        created.append(
            await job_service.create_job(
                record.id,
                job_type=job_type,
                payload={"marker": job_type},
            )
        )

    assert {job.job_type for job in created} == RESUMABLE_KB_MUTATION_JOB_TYPES
    assert {job.payload["kb_generation"] for job in created} == {
        record.generation
    }


@pytest.mark.asyncio
async def test_job_service_rejects_new_job_after_lifecycle_is_deleting(
    tmp_path: Path,
):
    store = await _make_store(tmp_path)
    kb_service = KnowledgeBaseService(tmp_path / "create-deleting-kb.json")
    await kb_service.initialize()
    record = await kb_service.create(kb_id="kb_create_deleting", name="Deleting")
    await store.activate_kb_generation(record.id, record.generation)
    job_service = JobService(kb_service, store)

    async with store.kb_deletion_guard(
        record.id,
        record.generation,
        "job_delete_create_guard",
    ):
        pass

    with pytest.raises(KBLifecycleConflictError):
        await job_service.create_job(record.id, job_type="parse")
    jobs, total = await store.list_jobs(record.id)
    assert jobs == []
    assert total == 0


@pytest.mark.asyncio
async def test_job_service_final_persist_guard_blocks_exclusive_delete(
    tmp_path: Path,
    monkeypatch,
):
    db_path = tmp_path / "job-final-persist.sqlite3"
    writer_store = SQLiteMetadataStore(db_path)
    delete_store = SQLiteMetadataStore(db_path)
    await writer_store.initialize()
    await delete_store.initialize()
    kb_service = KnowledgeBaseService(tmp_path / "job-final-persist-kb.json")
    await kb_service.initialize()
    record = await kb_service.create(kb_id="kb_job_final_persist", name="Persist")
    await writer_store.activate_kb_generation(record.id, record.generation)
    job_service = JobService(kb_service, writer_store)

    persist_entered = asyncio.Event()
    release_persist = asyncio.Event()
    exclusive_entered = asyncio.Event()
    original_create_job = writer_store.create_job

    async def blocked_create_job(job: JobRecord) -> JobRecord:
        # JobService has completed its guarded catalog/lifecycle/workspace
        # checks before it calls the store persistence primitive.
        persist_entered.set()
        await release_persist.wait()
        return await original_create_job(job)

    monkeypatch.setattr(writer_store, "create_job", blocked_create_job)

    async def delete_attempt() -> None:
        async with delete_store.kb_deletion_guard(
            record.id,
            record.generation,
            "job_delete_after_final_check",
        ):
            exclusive_entered.set()

    create_task = asyncio.create_task(
        job_service.create_job(record.id, job_type="parse")
    )
    await asyncio.wait_for(persist_entered.wait(), timeout=2)
    delete_task = asyncio.create_task(delete_attempt())
    await asyncio.sleep(0.1)
    assert not exclusive_entered.is_set()

    release_persist.set()
    created = await asyncio.wait_for(create_task, timeout=2)
    await asyncio.wait_for(exclusive_entered.wait(), timeout=2)
    await asyncio.wait_for(delete_task, timeout=2)

    assert created.payload["kb_generation"] == record.generation
    assert (await writer_store.get_job(record.id, created.id)).id == created.id
    with pytest.raises(KBLifecycleConflictError):
        await job_service.create_job(record.id, job_type="parse")


@pytest.mark.parametrize(
    ("generation", "expected_error_code"),
    [
        (_UNSET, "worker_kb_generation_missing"),
        ("wrong-generation", "worker_kb_generation_conflict"),
    ],
)
@pytest.mark.asyncio
async def test_worker_fails_closed_for_legacy_or_mismatched_generation(
    tmp_path: Path,
    generation: object,
    expected_error_code: str,
):
    store = await _make_store(tmp_path)
    kb_service = KnowledgeBaseService(tmp_path / f"fence-{expected_error_code}.json")
    await kb_service.initialize()
    record = await kb_service.create(
        kb_id=f"kb_{expected_error_code}",
        name=expected_error_code,
    )
    await store.activate_kb_generation(record.id, record.generation)
    job_service = JobService(kb_service, store)
    await store.create_job(
        _job(
            record.id,
            f"job_{expected_error_code}",
            job_type="parse",
            document_id=None,
            generation=generation,
        )
    )
    executor_calls = 0

    async def executor(_job_record: JobRecord) -> None:
        nonlocal executor_calls
        executor_calls += 1

    worker = JobWorker(
        job_service,
        executors={"parse": executor},
        claim_grace_seconds=0.0,
    )
    claimed = await worker.poll_once()

    assert claimed is not None
    assert executor_calls == 0
    failed = await store.get_job(record.id, claimed.id)
    assert failed.status == "failed"
    assert failed.error_code == expected_error_code


@pytest.mark.asyncio
async def test_worker_catalog_missing_fails_claimed_row_without_executor(
    tmp_path: Path,
):
    store = await _make_store(tmp_path)
    kb_id = "kb_worker_catalog_missing"
    generation = "missing-catalog-generation"
    await store.activate_kb_generation(kb_id, generation)
    await store.create_job(
        _job(
            kb_id,
            "job_catalog_missing",
            job_type="parse",
            document_id=None,
            generation=generation,
        )
    )
    kb_service = KnowledgeBaseService(tmp_path / "missing-catalog.json")
    await kb_service.initialize()
    job_service = JobService(kb_service, store)
    executor_calls = 0

    async def executor(_job_record: JobRecord) -> None:
        nonlocal executor_calls
        executor_calls += 1

    worker = JobWorker(
        job_service,
        executors={"parse": executor},
        claim_grace_seconds=0.0,
    )
    claimed = await worker.poll_once()

    assert claimed is not None
    assert executor_calls == 0
    failed = await store.get_job(kb_id, claimed.id)
    assert failed.status == "failed"
    assert failed.error_code == "worker_kb_generation_conflict"


@pytest.mark.asyncio
async def test_worker_claim_during_deleting_has_no_executor_side_effect(
    tmp_path: Path,
):
    store = await _make_store(tmp_path)
    kb_service = KnowledgeBaseService(tmp_path / "execute-deleting-kb.json")
    await kb_service.initialize()
    record = await kb_service.create(kb_id="kb_execute_deleting", name="Deleting")
    await store.activate_kb_generation(record.id, record.generation)
    job_service = JobService(kb_service, store)
    job = await job_service.create_job(record.id, job_type="parse")
    async with store.kb_deletion_guard(
        record.id,
        record.generation,
        "job_delete_execute_guard",
    ):
        pass

    executor_calls = 0

    async def executor(_job_record: JobRecord) -> None:
        nonlocal executor_calls
        executor_calls += 1

    worker = JobWorker(
        job_service,
        executors={"parse": executor},
        claim_grace_seconds=0.0,
    )
    claimed = await worker.poll_once()

    assert claimed is not None and claimed.id == job.id
    assert executor_calls == 0
    failed = await store.get_job(record.id, job.id)
    assert failed.status == "failed"
    assert failed.error_code == "worker_kb_generation_conflict"


@pytest.mark.asyncio
async def test_worker_shared_guard_blocks_two_store_exclusive_delete(
    tmp_path: Path,
):
    db_path = tmp_path / "two-store-worker.sqlite3"
    worker_store = SQLiteMetadataStore(db_path)
    delete_store = SQLiteMetadataStore(db_path)
    await worker_store.initialize()
    await delete_store.initialize()
    kb_service = KnowledgeBaseService(tmp_path / "two-store-kb.json")
    await kb_service.initialize()
    record = await kb_service.create(kb_id="kb_two_store_worker", name="Two Store")
    await worker_store.activate_kb_generation(record.id, record.generation)
    job_service = JobService(kb_service, worker_store)
    job = await job_service.create_job(record.id, job_type="parse")

    executor_entered = asyncio.Event()
    release_executor = asyncio.Event()
    exclusive_entered = asyncio.Event()

    async def executor(claimed: JobRecord) -> None:
        executor_entered.set()
        await release_executor.wait()
        await job_service.transition_job(
            claimed.kb_id,
            claimed.id,
            status="succeeded",
            progress=1.0,
            completed_items=1,
        )

    async def delete_attempt() -> None:
        async with delete_store.kb_deletion_guard(
            record.id,
            record.generation,
            "job_delete_two_store_guard",
        ):
            exclusive_entered.set()

    worker = JobWorker(
        job_service,
        executors={"parse": executor},
        claim_grace_seconds=0.0,
    )
    worker_task = asyncio.create_task(worker.poll_once())
    await asyncio.wait_for(executor_entered.wait(), timeout=2.0)
    delete_task = asyncio.create_task(delete_attempt())
    await asyncio.sleep(0.1)
    assert not exclusive_entered.is_set()

    release_executor.set()
    claimed = await asyncio.wait_for(worker_task, timeout=2.0)
    await asyncio.wait_for(exclusive_entered.wait(), timeout=2.0)
    await asyncio.wait_for(delete_task, timeout=2.0)

    assert claimed is not None and claimed.id == job.id
    assert (await worker_store.get_job(record.id, job.id)).status == "succeeded"


@pytest.mark.asyncio
async def test_clear_job_does_not_take_shared_guard_before_exclusive_executor(
    tmp_path: Path,
):
    store = await _make_store(tmp_path)
    kb_service = KnowledgeBaseService(tmp_path / "clear-no-shared-kb.json")
    await kb_service.initialize()
    record = await kb_service.create(kb_id="kb_clear_no_shared", name="Clear")
    await store.activate_kb_generation(record.id, record.generation)
    job_service = JobService(kb_service, store)
    clear_job = _job(
        record.id,
        "job_clear_no_shared",
        job_type="clear_kb",
        document_id=None,
    )
    clear_job.payload = {
        "kb_generation": record.generation,
        "workspace": record.workspace,
    }
    await store.create_job(clear_job)

    async def clear_executor(claimed: JobRecord) -> None:
        async with store.kb_deletion_guard(
            claimed.kb_id,
            record.generation,
            claimed.id,
        ):
            await store.transition_job(
                claimed.kb_id,
                claimed.id,
                status="succeeded",
                progress=1.0,
                completed_items=1,
            )

    worker = JobWorker(
        job_service,
        executors={"clear_kb": clear_executor},
        claim_grace_seconds=0.0,
    )
    claimed = await asyncio.wait_for(worker.poll_once(), timeout=2.0)

    assert claimed is not None and claimed.id == clear_job.id
    assert (await store.get_job(record.id, clear_job.id)).status == "succeeded"
