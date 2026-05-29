from __future__ import annotations

from pathlib import Path

import pytest

from lightrag.api.job_service import JobService
from lightrag.api.job_worker import JobWorker
from lightrag.api.kb_service import KnowledgeBaseService, utc_now_iso
from lightrag.api.metadata_store import (
    DocumentRecord,
    JobRecord,
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
) -> JobRecord:
    now = utc_now_iso()
    # Real single-document jobs always carry a document_id; default to the job
    # id so fixtures match production (the worker only claims document_id IS NOT
    # NULL). Pass document_id=None explicitly for aggregate-job fixtures.
    resolved_document_id = (
        f"doc_{job_id}" if document_id is _UNSET else document_id
    )
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
        payload={"document_id": resolved_document_id}
        if resolved_document_id
        else {},
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
async def test_claim_skips_aggregate_jobs_without_document_id(tmp_path: Path):
    """Aggregate jobs (document_id NULL — e.g. multi-file upload auto_parse,
    batch-parse/build/delete) must NOT be claimed by the single-document
    worker: it cannot re-drive them and would mis-mark worker_invalid_payload.
    """
    store = await _make_store(tmp_path)
    # Aggregate parse job: document_id=None, payload carries document_ids list.
    await store.create_job(
        _job("kb_a", "agg_parse", job_type="parse", document_id=None)
    )
    # A real single-document parse job alongside it.
    await _create_job(store, _job("kb_a", "single_parse", job_type="parse"))

    claimed = await store.claim_next_worker_job(
        job_types=["parse"], max_queued_at=None
    )
    # The single-doc job is claimed; the aggregate is skipped entirely.
    assert claimed is not None
    assert claimed.id == "single_parse"

    # Nothing else is claimable — the aggregate job is never picked up.
    assert (
        await store.claim_next_worker_job(job_types=["parse"], max_queued_at=None)
        is None
    )
    agg = await store.get_job("kb_a", "agg_parse")
    assert agg.status == "queued"  # left untouched for its owner task


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
    await _create_job(store, _job(record.id, "job_exec", job_type="parse"))

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

    await _create_job(store, _job(record.id, "job_retry", job_type="parse"))
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
    await _create_job(store, _job(record.id, "job_bad", job_type="parse"))

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
    # A queued parse job (resumable) + a queued delete job (not resumable) +
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
    """When 'delete' is a resumable type (single-doc delete needs only the
    persisted payload), a queued delete job survives restart recovery."""
    store = await _make_store(tmp_path)
    await _create_job(store, _job("kb_d", "queued_delete", job_type="delete"))
    await _create_job(store, _job("kb_d", "queued_upload", job_type="upload"))

    recovered = await store.recover_orphan_jobs(
        resumable_job_types={"parse", "build_kg", "reindex", "delete"}
    )
    recovered_ids = {job.id for job in recovered}
    # delete is kept queued for the worker; upload (needs request bytes) fails.
    assert "queued_delete" not in recovered_ids
    assert "queued_upload" in recovered_ids

    survivor = await store.get_job("kb_d", "queued_delete")
    assert survivor.status == "queued"


@pytest.mark.asyncio
async def test_recovery_fails_aggregate_delete_even_when_delete_resumable(tmp_path: Path):
    """An AGGREGATE delete job (document_id NULL, e.g. batch-delete) must be
    FAILED on restart even though 'delete' is resumable: the worker only claims
    document_id IS NOT NULL, so keeping it queued would leak a zombie that is
    never picked up. Single-doc delete is still kept."""
    store = await _make_store(tmp_path)
    await store.create_job(
        _job("kb_b", "batch_delete", job_type="delete", document_id=None)
    )
    await _create_job(store, _job("kb_b", "single_delete", job_type="delete"))

    recovered = await store.recover_orphan_jobs(
        resumable_job_types={"parse", "build_kg", "reindex", "delete"}
    )
    recovered_ids = {job.id for job in recovered}
    # Aggregate delete is failed; single-doc delete is kept for the worker.
    assert "batch_delete" in recovered_ids
    assert "single_delete" not in recovered_ids
    assert (await store.get_job("kb_b", "batch_delete")).status == "failed"
    assert (await store.get_job("kb_b", "single_delete")).status == "queued"


@pytest.mark.asyncio
async def test_worker_run_loop_consumes_then_stops(tmp_path: Path):
    """Drive the real background loop: start() schedules _run_loop, which drains
    a queued job to completion, then stop() signals the loop and awaits it."""
    import asyncio

    store = await _make_store(tmp_path)
    kb_service = KnowledgeBaseService(tmp_path / "kb.json")
    await kb_service.initialize()
    await kb_service.create(kb_id="kb_loop", name="Loop")
    record = await kb_service.get("kb_loop")
    job_service = JobService(kb_service, store)
    await _create_job(store, _job(record.id, "loop_job", job_type="parse"))

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


