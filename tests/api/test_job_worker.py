from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import patch

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

# Phase 3.1-C Integration Writer B2: object-mode worker resume test imports.
import hashlib as _hashlib_b2w
from datetime import datetime as _dt_b2w, timezone as _tz_b2w
from uuid import uuid4 as _uuid4_b2w

from lightrag.api.artifact_materialization import (
    ArtifactMaterializer as _AM_b2w,
    MaterializationLimits as _ML_b2w,
)
from lightrag.api.config import ArtifactCleanupConfig as _ACC_b2w
from lightrag.api.document_lifecycle_service import (
    DocumentLifecycleService as _DLS_b2w,
)
from lightrag.api.kb_service import utc_now_iso as _now_b2w
from lightrag.api.metadata_store import (
    ArtifactRecord as _AR_b2w,
    DocumentRecord as _DR_b2w,
)
from lightrag.api.object_storage import (
    ObjectReadback as _ORB_b2w,
    ObjectStat as _OS_b2w,
    ObjectStorage as _OS_b2w_base,
    ObjectStorageError as _OSE_b2w,
    ObjectStorageNotFoundError as _OSNF_b2w,
)
from lightrag.utils_pipeline import (
    reset_canonical_input_root_for_tests as _rr_b2w,
    set_canonical_input_root as _sr_b2w,
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


# ---------------------------------------------------------------------------
# Phase 3.1-C Integration Writer B2: object-mode worker resume tests.
#
# These tests exercise the object-mode branches in the durable worker
# executors (build_delete_executor, build_replace_executor) via direct calls
# with a real SQLite store + fake object storage. They verify:
# - Worker resume at engine_cleanup_pending re-calls B1 with persisted token.
# - Token rotation after precommit failure.
# - Object-mode delete/batch-delete worker resume.
# ---------------------------------------------------------------------------

_B2W_NOW = _dt_b2w(2026, 8, 3, 12, 0, 0, tzinfo=_tz_b2w.utc)
_B2W_BUCKET = "b2w-bucket"


class _B2WFakeObjectStorage(_OS_b2w_base):
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.upload_proof_calls: list[tuple[str, str | None]] = []
        self.deleted_uris: list[str] = []
        self.deleted_prefixes: list[str] = []

    async def initialize(self):
        return None

    async def close(self):
        return None

    async def upload_file(self, local_path: Path, *, key: str, content_type=None):
        uri = self.object_uri_for_key(key)
        self.files[uri] = local_path.read_bytes()
        return uri

    async def upload_file_if_absent(
        self, local_path: Path, *, key: str, content_type=None, expected_sha256=None
    ):
        del content_type
        uri = self.object_uri_for_key(key)
        self.upload_proof_calls.append((uri, expected_sha256))
        if uri in self.files:
            return uri, False
        self.files[uri] = local_path.read_bytes()
        return uri, True

    def object_uri_for_key(self, key: str):
        return f"s3://{_B2W_BUCKET}/{key.lstrip('/')}"

    def object_prefix_uri_for_key(self, prefix: str):
        return f"s3://{_B2W_BUCKET}/{prefix.strip('/')}/"

    async def stat_object(self, object_uri: str):
        rb = await self.inspect_object(object_uri)
        if not rb.present or rb.stat is None:
            raise _OSE_b2w(f"Missing: {object_uri}")
        return rb.stat

    async def inspect_object(self, object_uri: str, *, version_id=None):
        if object_uri not in self.files:
            return _ORB_b2w(present=False)
        data = self.files[object_uri]
        return _ORB_b2w(
            present=True,
            stat=_OS_b2w(
                size=len(data),
                etag=f'"etag-{len(data)}"',
                last_modified=_B2W_NOW,
                checksum=f"sha256:{_hashlib_b2w.sha256(data).hexdigest()}",
            ),
        )

    async def download_file(self, object_uri: str, local_path: Path):
        if object_uri not in self.files:
            raise _OSNF_b2w()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(self.files[object_uri])

    async def delete_uri(self, object_uri: str):
        self.deleted_uris.append(object_uri)
        return self.files.pop(object_uri, None) is not None

    async def delete_prefix(self, prefix_uri: str):
        self.deleted_prefixes.append(prefix_uri)
        count = 0
        for uri in list(self.files):
            if uri.startswith(prefix_uri):
                self.files.pop(uri)
                count += 1
        return count

    async def delete_workspace(self, workspace: str):
        return 0

    def validate_document_file_uri(self, *args, **kwargs):
        return None

    def validate_document_prefix_uri(self, *args, **kwargs):
        return None


class _B2WFakeRAG:
    def __init__(self):
        self.deleted: list[tuple[str, bool]] = []

    async def adelete_by_doc_id(self, doc_id: str, delete_llm_cache: bool = False):
        self.deleted.append((doc_id, delete_llm_cache))
        from types import SimpleNamespace

        return SimpleNamespace(
            status="success", doc_id=doc_id, message="deleted",
            status_code=200, file_path="",
        )

    async def finalize_storages(self):
        return None

    async def adrop_all_storages(self):
        return {"dropped": 0, "failed": 0, "errors": []}


class _B2WFakeRegistry:
    def __init__(self, rag):
        self._rag = rag

    async def get(self, kb_id: str):
        return self._rag

    async def acquire(self, kb_id: str):
        return self._rag


def _b2w_sha256(data: bytes) -> str:
    return _hashlib_b2w.sha256(data).hexdigest()


def _b2w_limits():
    return _ML_b2w(max_objects=1000, max_total_bytes=64 * 1024 * 1024, stale_ttl_seconds=1)


def _b2w_document(kb_id, document_id, *, workspace, artifact_id: str | None = "artifact-b2w"):  # type: ignore[no-untyped-def]
    now = _now_b2w()
    source_uri = (
        f"s3://{_B2W_BUCKET}/workspaces/{workspace}/documents/{document_id}/source/"
        f"generations/srcg-b2w-old/source.pdf"
    )
    metadata: dict = {
        "source_object_uri": source_uri,
        "source_generation_id": "srcg-b2w-old",
    }
    if artifact_id:
        metadata.update(
            {"current_sidecar_artifact_id": artifact_id, "current_artifact_ids": [artifact_id]}
        )
    return _DR_b2w(
        id=document_id, kb_id=kb_id, workspace=workspace,
        lightrag_doc_id=f"engine-{document_id}",
        source_type="upload", source_name="source.pdf", source_uri=source_uri,
        source_hash="sha256:" + "0" * 64, content_type="application/pdf",
        size_bytes=4, parser_hash="p", index_hash="i",
        status="ready", enabled=True, archived=False, chunks_count=1,
        entity_count=0, relation_count=0, error_code=None, error_message=None,
        metadata=metadata, created_at=now, updated_at=now, deleted_at=None,
    )


def _b2w_artifact(document, artifact_id="artifact-b2w"):
    now = _now_b2w()
    uri = (
        f"s3://{_B2W_BUCKET}/workspaces/{document.workspace}/documents/{document.id}/"
        f"artifacts/raw/{artifact_id}/sidecar.json"
    )
    return _AR_b2w(
        id=artifact_id, kb_id=document.kb_id, workspace=document.workspace,
        document_id=document.id, artifact_type="sidecar", uri=uri,
        checksum="sha256:" + "a" * 64, size_bytes=9,
        metadata={"object_uri": uri}, created_at=now,
    )


async def _b2w_put_artifact(store, artifact):
    def write(conn):
        store._insert_artifact(conn, artifact)
    await store._write(write)


@pytest.fixture
def b2w_setup(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir(parents=True, exist_ok=True)
    _rr_b2w()
    _sr_b2w(root)

    async def _build():
        store = SQLiteMetadataStore(tmp_path / "b2w.sqlite3")
        await store.initialize()
        kb_service = KnowledgeBaseService(tmp_path / "b2w_kbs.json")
        await kb_service.initialize()
        kb_id = f"kb_b2w_{_uuid4_b2w().hex[:10]}"
        record = await kb_service.create(name=kb_id, kb_id=kb_id)
        workspace = record.workspace
        generation = record.generation
        await store.activate_kb_generation(kb_id, generation)
        storage = _B2WFakeObjectStorage()
        materializer = _AM_b2w(storage, input_root=root, limits=_b2w_limits())
        service = _DLS_b2w(
            kb_service, store, root,
            object_storage=storage, artifact_storage_mode="object",
            materializer=materializer, artifact_cleanup_config=_ACC_b2w(),
            clock=lambda: _B2W_NOW,
        )
        return service, store, storage, kb_id, workspace, generation, kb_service

    return _build


async def test_b2w_worker_resume_replace_engine_cleanup_pending(b2w_setup):
    """Worker resume at engine_cleanup_pending re-calls B1 with persisted token."""
    service, store, storage, kb_id, workspace, generation, kb_service = await b2w_setup()
    document = _b2w_document(kb_id, "doc-r1", workspace=workspace)
    artifact = _b2w_artifact(document)
    storage.files[document.metadata["source_object_uri"]] = b"old"
    storage.files[artifact.uri] = b"art"
    now = _now_b2w()

    # Phase 1: run the COW replace with a failing engine to create the
    # committed engine_cleanup_pending state (manifests + pointer committed,
    # engine not yet cleaned up). This uses a real replace job row.
    job_id = "job-r1"
    source_hash = "sha256:" + _b2w_sha256(b"new-content")
    from lightrag.api.metadata_store import JobRecord as _JR

    seed_job = _JR(
        id=job_id, kb_id=kb_id, workspace=workspace, batch_id=None,
        document_id="doc-r1", job_type="replace", status="running",
        stage="replacing", progress=0.1, total_items=1, completed_items=0,
        failed_items=0, idempotency_key="idem-r1", config_version_id=None,
        config_hash=None, retry_count=0, max_retries=3,
        payload={"idempotency_fingerprint": "sha256:r1", "attempt_tokens": {}},
        result=None, error_code=None, error_message=None,
        created_at=now, updated_at=now, queued_at=now, started_at=now,
        finished_at=None, cancelled_at=None,
    )
    await store.create_documents_and_job([document], seed_job)
    await _b2w_put_artifact(store, artifact)

    class _FailOnceRAG:
        def __init__(self):
            self.deleted: list = []

        async def adelete_by_doc_id(self, doc_id, delete_llm_cache=False):
            raise RuntimeError("engine crashed mid-cleanup")

        async def finalize_storages(self):
            return None

        async def adrop_all_storages(self):
            return {"dropped": 0, "failed": 0, "errors": []}

    async def _fail_engine(kb, doc, prev_id, identity):
        raise RuntimeError("engine crashed")

    # First call: engine fails, leaves engine_cleanup_pending.
    from lightrag.api.document_lifecycle_service import DocumentCowEngineDeleteError

    with pytest.raises(DocumentCowEngineDeleteError):
        await service.execute_document_replace_cow(
            kb_id, "doc-r1", job_id=job_id, kb_generation=generation,
            new_source_type="upload", new_source_name="source.pdf",
            new_source_uri="", new_source_hash=source_hash,
            new_content_type="application/pdf", new_size_bytes=11,
            replacement_content=b"new-content",
            engine_delete=_fail_engine,
            claim_token="attempt-resume-1",
        )
    # Verify the document is in engine_cleanup_pending.
    pending = await store.get_document(kb_id, "doc-r1")
    assert pending.metadata.get("replace_phase") == "engine_cleanup_pending"

    # Phase 2: the worker re-drives the job from queued. Simulate the
    # crash-recovery retry flow: running → failed (orphan recovery) → queued.
    await store.transition_job(
        kb_id, job_id, status="failed", progress=1.0, failed_items=1,
        error_code="replace_engine_cleanup_pending",
    )
    await store.transition_job(kb_id, job_id, status="queued")
    queued_job = await store.get_job(kb_id, job_id)
    # Persist the claim token in the payload (route would do this).
    await store.update_job_payload_patch(
        kb_id, job_id, payload_patch={
            "source_name": "source.pdf", "source_type": "upload",
            "source_hash": source_hash, "content_type": "application/pdf",
            "size_bytes": 11, "delete_source_file": True,
            "delete_artifacts": True, "delete_llm_cache": False,
            "auto_parse": False, "auto_index": False,
            "previous_lightrag_doc_id": document.lightrag_doc_id,
            "attempt_tokens": {"doc-r1": "attempt-resume-1"},
        },
    )
    queued_job = await store.get_job(kb_id, job_id)

    # Phase 3: the worker executor resumes and finishes engine cleanup.
    rag = _B2WFakeRAG()
    registry = _B2WFakeRegistry(rag)
    job_service = JobService(kb_service, store)
    from lightrag.api.job_worker import build_replace_executor

    executor = build_replace_executor(
        document_service=service, registry=registry,
        job_service=job_service, index_service=None,
    )
    await store.transition_job(kb_id, job_id, status="running", progress=0.1)
    await executor(queued_job)
    final_job = await store.get_job(kb_id, job_id)
    assert final_job.status == "succeeded"
    # Engine delete happened on resume.
    assert rag.deleted == [(document.lightrag_doc_id, False)]
    result = final_job.result or {}
    assert result.get("resumed_by_worker") is True
    # Document is finalized.
    doc = await store.get_document(kb_id, "doc-r1")
    assert doc.metadata.get("replace_phase") == "completed"


async def test_b2w_worker_resume_delete_object_mode(b2w_setup):
    """Object-mode delete worker resume via B1 with per-document token."""
    service, store, storage, kb_id, workspace, generation, kb_service = await b2w_setup()
    document = _b2w_document(kb_id, "doc-d1", workspace=workspace)
    artifact = _b2w_artifact(document)
    storage.files[document.metadata["source_object_uri"]] = b"old"
    storage.files[artifact.uri] = b"art"
    now = _now_b2w()
    job = JobRecord(
        id="job-d1", kb_id=kb_id, workspace=workspace, batch_id=None,
        document_id="doc-d1", job_type="delete", status="queued",
        stage="deleting", progress=0.0, total_items=1, completed_items=0,
        failed_items=0, idempotency_key="idem-d1", config_version_id=None,
        config_hash=None, retry_count=0, max_retries=3,
        payload={
            "document_id": "doc-d1", "delete_source_file": True,
            "delete_artifacts": True, "delete_llm_cache": False,
            "delete_graph_orphans": True, "strategy": "safe",
            "attempt_tokens": {},
            "idempotency_fingerprint": "sha256:d1",
        },
        result=None, error_code=None, error_message=None,
        created_at=now, updated_at=now, queued_at=now, started_at=None,
        finished_at=None, cancelled_at=None,
    )
    await store.create_documents_and_job([document], job)
    await _b2w_put_artifact(store, artifact)

    rag = _B2WFakeRAG()
    registry = _B2WFakeRegistry(rag)
    job_service = JobService(kb_service, store)
    from lightrag.api.job_worker import build_delete_executor

    executor = build_delete_executor(
        document_service=service, registry=registry,
        job_service=job_service, index_service=None,
    )
    await store.transition_job(kb_id, "job-d1", status="running", progress=0.1)
    await executor(job)
    final_job = await store.get_job(kb_id, "job-d1")
    assert final_job.status == "succeeded"
    assert rag.deleted == [(document.lightrag_doc_id, False)]
    result = final_job.result or {}
    assert result.get("resumed_by_worker") is True
    # Tombstone committed.
    tomb = await store.get_document_lifecycle(kb_id, "doc-d1")
    assert tomb.deleted_at is not None
    # Token persisted.
    assert "doc-d1" in (final_job.payload or {}).get("attempt_tokens", {})


async def test_b2w_worker_resume_batch_delete_object_mode(b2w_setup):
    """Object-mode batch delete worker resume with per-document tokens."""
    service, store, storage, kb_id, workspace, generation, kb_service = await b2w_setup()
    doc1 = _b2w_document(kb_id, "doc-bd1", workspace=workspace)
    doc2 = _b2w_document(kb_id, "doc-bd2", workspace=workspace, artifact_id=None)
    storage.files[doc1.metadata["source_object_uri"]] = b"old1"
    storage.files[doc2.metadata["source_object_uri"]] = b"old2"
    art1 = _b2w_artifact(doc1)
    now = _now_b2w()
    job = JobRecord(
        id="job-bd", kb_id=kb_id, workspace=workspace, batch_id="batch-bd",
        document_id=None, job_type="delete", status="queued",
        stage="deleting", progress=0.0, total_items=2, completed_items=0,
        failed_items=0, idempotency_key="idem-bd", config_version_id=None,
        config_hash=None, retry_count=0, max_retries=3,
        payload={
            "document_ids": ["doc-bd1", "doc-bd2"],
            "delete_source_file": True, "delete_artifacts": True,
            "delete_llm_cache": False, "delete_graph_orphans": True,
            "strategy": "safe", "attempt_tokens": {},
            "idempotency_fingerprint": "sha256:bd",
        },
        result=None, error_code=None, error_message=None,
        created_at=now, updated_at=now, queued_at=now, started_at=None,
        finished_at=None, cancelled_at=None,
    )
    await store.create_documents_and_job([doc1, doc2], job)
    await _b2w_put_artifact(store, art1)

    rag = _B2WFakeRAG()
    registry = _B2WFakeRegistry(rag)
    job_service = JobService(kb_service, store)
    from lightrag.api.job_worker import build_delete_executor

    executor = build_delete_executor(
        document_service=service, registry=registry,
        job_service=job_service, index_service=None,
    )
    await store.transition_job(kb_id, "job-bd", status="running", progress=0.0)
    await executor(job)
    final_job = await store.get_job(kb_id, "job-bd")
    assert final_job.status == "succeeded"
    result = final_job.result or {}
    assert result.get("resumed_by_worker") is True
    items = result.get("items", [])
    succeeded = [i for i in items if i.get("status") == "succeeded"]
    assert len(succeeded) == 2
    # Each document tombstoned.
    for doc_id in ("doc-bd1", "doc-bd2"):
        tomb = await store.get_document_lifecycle(kb_id, doc_id)
        assert tomb.deleted_at is not None


async def test_b2w_worker_replace_not_resumable_without_engine_cleanup_pending(b2w_setup):
    """Object-mode replace worker fails cleanly when not in engine_cleanup_pending."""
    service, store, storage, kb_id, workspace, generation, kb_service = await b2w_setup()
    document = _b2w_document(kb_id, "doc-nr", workspace=workspace)
    now = _now_b2w()
    job = JobRecord(
        id="job-nr", kb_id=kb_id, workspace=workspace, batch_id=None,
        document_id="doc-nr", job_type="replace", status="queued",
        stage="replacing", progress=0.0, total_items=1, completed_items=0,
        failed_items=0, idempotency_key="idem-nr", config_version_id=None,
        config_hash=None, retry_count=0, max_retries=3,
        payload={
            "document_id": "doc-nr", "source_name": "source.pdf",
            "source_type": "upload", "source_hash": "sha256:x",
            "content_type": "application/pdf", "size_bytes": 1,
            "delete_source_file": True, "delete_artifacts": True,
            "delete_llm_cache": False, "auto_parse": False, "auto_index": False,
            "previous_lightrag_doc_id": document.lightrag_doc_id,
            "attempt_tokens": {},
            "idempotency_fingerprint": "sha256:nr",
        },
        result=None, error_code=None, error_message=None,
        created_at=now, updated_at=now, queued_at=now, started_at=None,
        finished_at=None, cancelled_at=None,
    )
    await store.create_documents_and_job([document], job)

    rag = _B2WFakeRAG()
    registry = _B2WFakeRegistry(rag)
    job_service = JobService(kb_service, store)
    from lightrag.api.job_worker import build_replace_executor

    executor = build_replace_executor(
        document_service=service, registry=registry,
        job_service=job_service, index_service=None,
    )
    await store.transition_job(kb_id, "job-nr", status="running", progress=0.1)
    await executor(job)
    final_job = await store.get_job(kb_id, "job-nr")
    assert final_job.status == "failed"
    assert final_job.error_code == "replace_not_resumable"
    # No engine side effect.
    assert rag.deleted == []


# ---------------------------------------------------------------------------
# Phase 3.2 Gate 2 non-blocking hardening: object-mode sync resume fallback.
#
# The sync executor's object branch previously constructed an empty-bytes
# ``DocumentSourceInput`` whenever ``staging_object_uris`` lacked an entry,
# WITHOUT verifying the document was in a post-commit state. These cases pin
# the new explicit guard: post-commit docs proceed; anything else fails
# cleanly as ``sync_not_resumable``; the staging-URI happy path and the
# local-mode branch are unchanged regressions.
# ---------------------------------------------------------------------------


def _b2w_sync_document(
    kb_id: str,
    document_id: str,
    *,
    workspace: str,
    source_key: str,
    source_hash: str,
    replace_phase: str | None = None,
    artifact_id: str | None = None,
) -> _DR_b2w:
    """A b2w document seeded with a ``source_key`` (and optional replace_phase).

    ``source_key`` lives in metadata so ``get_documents_by_source_keys`` can
    resolve the row back from the store during sync resume.
    """
    doc = _b2w_document(kb_id, document_id, workspace=workspace, artifact_id=artifact_id)
    doc.source_hash = source_hash
    doc.metadata["source_key"] = source_key
    if replace_phase is not None:
        doc.metadata["replace_phase"] = replace_phase
    return doc


def _b2w_sync_job(
    kb_id: str,
    workspace: str,
    *,
    job_id: str,
    batch_id: str,
    items: list[dict[str, Any]],
    staging_object_uris: dict[str, str] | None = None,
) -> JobRecord:
    """Aggregate ``sync`` job row with the minimal resumable payload."""
    now = _now_b2w()
    payload: dict[str, Any] = {
        "batch_id": batch_id,
        "items": items,
        "auto_parse": False,
        "auto_index": False,
        "delete_source_file": True,
        "delete_artifacts": True,
        "delete_llm_cache": False,
        "force_reparse": False,
    }
    if staging_object_uris is not None:
        payload["staging_object_uris"] = staging_object_uris
    return JobRecord(
        id=job_id, kb_id=kb_id, workspace=workspace, batch_id=batch_id,
        document_id=None, job_type="sync", status="queued",
        stage="syncing", progress=0.0, total_items=len(items),
        completed_items=0, failed_items=0, idempotency_key=f"idem-{job_id}",
        config_version_id=None, config_hash=None, retry_count=0, max_retries=3,
        payload=payload, result=None, error_code=None, error_message=None,
        created_at=now, updated_at=now, queued_at=now, started_at=None,
        finished_at=None, cancelled_at=None,
    )


def _b2w_sync_item(source_key: str, source_name: str, *, source_hash: str,
                   size_bytes: int, source_type: str = "upload") -> dict[str, Any]:
    return {
        "source_key": source_key,
        "source_name": source_name,
        "source_type": source_type,
        "source_hash": source_hash,
        "content_type": "application/pdf",
        "size_bytes": size_bytes,
    }


async def test_b2w_sync_resume_with_staging_object_uri_succeeds(b2w_setup):
    """Regression: a persisted staging_object_uri loads bytes and proceeds."""
    service, store, storage, kb_id, workspace, generation, kb_service = await b2w_setup()
    content = b"sync-staged-bytes"
    source_hash = "sha256:" + _b2w_sha256(content)
    source_key = "manual/sync-staged.pdf"
    source_name = "sync-staged.pdf"
    # Existing doc carries the SAME hash so the sync item skips (light path);
    # the staging-URI branch still downloads + integrity-checks the object.
    doc = _b2w_sync_document(
        kb_id, "doc-sync-staged", workspace=workspace,
        source_key=source_key, source_hash=source_hash,
    )
    staging_uri = storage.object_uri_for_key(
        f"workspaces/{workspace}/.sync-staging/batch-sync-staged/staged"
    )
    storage.files[staging_uri] = content
    job = _b2w_sync_job(
        kb_id, workspace, job_id="job-sync-staged", batch_id="batch-sync-staged",
        items=[_b2w_sync_item(source_key, source_name, source_hash=source_hash,
                              size_bytes=len(content))],
        staging_object_uris={source_key: staging_uri},
    )
    await store.create_documents_and_job([doc], job)

    rag = _B2WFakeRAG()
    registry = _B2WFakeRegistry(rag)
    job_service = JobService(kb_service, store)
    from lightrag.api.job_worker import build_sync_executor

    executor = build_sync_executor(
        document_service=service, registry=registry,
        job_service=job_service, index_service=None,
    )
    await store.transition_job(kb_id, job.id, status="running", progress=0.0)
    await executor(job)
    final = await store.get_job(kb_id, job.id)
    assert final.status == "succeeded"
    result = final.result or {}
    assert result.get("resumed_by_worker") is True
    item = result["items"][0]
    assert item["status"] == "skipped"  # hash match — no re-upload


async def test_b2w_sync_resume_without_staging_uri_post_commit_proceeds(b2w_setup):
    """No staging URI but document is post-commit: empty-bytes fallback is safe."""
    service, store, storage, kb_id, workspace, generation, kb_service = await b2w_setup()
    content = b"sync-postcommit-bytes"
    source_hash = "sha256:" + _b2w_sha256(content)
    source_key = "manual/sync-postcommit.pdf"
    source_name = "sync-postcommit.pdf"
    # Document is in engine_cleanup_pending (COW commit already landed) AND
    # carries the same hash, so the metadata-only path is safe (item skipped).
    doc = _b2w_sync_document(
        kb_id, "doc-sync-postcommit", workspace=workspace,
        source_key=source_key, source_hash=source_hash,
        replace_phase="engine_cleanup_pending",
    )
    job = _b2w_sync_job(
        kb_id, workspace, job_id="job-sync-postcommit",
        batch_id="batch-sync-postcommit",
        items=[_b2w_sync_item(source_key, source_name, source_hash=source_hash,
                              size_bytes=len(content))],
        # Deliberately NO staging_object_uris — exercise the fallback.
    )
    await store.create_documents_and_job([doc], job)

    rag = _B2WFakeRAG()
    registry = _B2WFakeRegistry(rag)
    job_service = JobService(kb_service, store)
    from lightrag.api.job_worker import build_sync_executor

    executor = build_sync_executor(
        document_service=service, registry=registry,
        job_service=job_service, index_service=None,
    )
    await store.transition_job(kb_id, job.id, status="running", progress=0.0)
    await executor(job)
    final = await store.get_job(kb_id, job.id)
    assert final.status == "succeeded"
    item = (final.result or {})["items"][0]
    assert item["status"] == "skipped"  # post-commit + hash match
    # No integrity-checked upload of empty bytes was attempted.
    assert storage.upload_proof_calls == []


async def test_b2w_sync_resume_without_staging_uri_not_resumable(b2w_setup):
    """No staging URI and no post-commit document: fails as sync_not_resumable."""
    service, store, storage, kb_id, workspace, generation, kb_service = await b2w_setup()
    content = b"sync-new-bytes"
    source_hash = "sha256:" + _b2w_sha256(content)
    source_key = "manual/sync-new.pdf"
    source_name = "sync-new.pdf"
    # No existing document seeded and no staging_object_uri: models a
    # pre-commit crash where the request died before persisting any
    # object-backed staging. The guard must fail cleanly.
    job = _b2w_sync_job(
        kb_id, workspace, job_id="job-sync-new", batch_id="batch-sync-new",
        items=[_b2w_sync_item(source_key, source_name, source_hash=source_hash,
                              size_bytes=len(content))],
    )
    await store.create_job(job)

    rag = _B2WFakeRAG()
    registry = _B2WFakeRegistry(rag)
    job_service = JobService(kb_service, store)
    from lightrag.api.job_worker import build_sync_executor

    executor = build_sync_executor(
        document_service=service, registry=registry,
        job_service=job_service, index_service=None,
    )
    await store.transition_job(kb_id, job.id, status="running", progress=0.0)
    await executor(job)
    final = await store.get_job(kb_id, job.id)
    assert final.status == "failed"
    assert final.error_code == "sync_not_resumable"
    assert "post-commit" in (final.error_message or "")
    # No engine side effect and no integrity-checked upload attempted.
    assert rag.deleted == []
    assert storage.upload_proof_calls == []


async def test_local_mode_sync_resume_from_staged_bytes_unchanged(tmp_path: Path):
    """Regression: local-mode sync resume still loads staged bytes from disk."""
    _rr_b2w()
    local_root = tmp_path / "local-source"
    local_root.mkdir(parents=True, exist_ok=True)
    _sr_b2w(local_root)
    store = SQLiteMetadataStore(tmp_path / "local.sqlite3")
    await store.initialize()
    kb_service = KnowledgeBaseService(tmp_path / "local_kbs.json")
    await kb_service.initialize()
    kb_id = f"kb_local_sync_{_uuid4_b2w().hex[:10]}"
    record = await kb_service.create(name=kb_id, kb_id=kb_id)
    workspace = record.workspace
    await store.activate_kb_generation(kb_id, record.generation)
    service = _DLS_b2w(kb_service, store, local_root)
    assert service.object_authoritative is False

    content = b"local-sync-bytes"
    source_key = "manual/local-sync.pdf"
    source_name = "local-sync.pdf"
    from lightrag.api.document_lifecycle_service import DocumentSourceInput

    source_input = DocumentSourceInput(
        source_name=source_name, content=content, source_type="scan",
        content_type="application/pdf", metadata={"source_key": source_key},
    )
    # Local mode computes source_hash as bare SHA-256 hex (no ``sha256:``
    # prefix) via ``_content_hash``; mirror the route's own computation so the
    # staged-file integrity check and the document skip-hash both match.
    source_hash = service.prepare_replacement_source(source_input).source_hash
    batch_id = "batch-local-sync"
    staged_path = await service.stage_sync_source_bytes(
        kb_id, batch_id=batch_id, item_index=0, source=source_input,
    )
    assert Path(staged_path).is_file()

    # Existing doc with matching hash → sync skips (light path).
    doc = _b2w_sync_document(
        kb_id, "doc-local-sync", workspace=workspace,
        source_key=source_key, source_hash=source_hash,
    )
    job = _b2w_sync_job(
        kb_id, workspace, job_id="job-local-sync", batch_id=batch_id,
        items=[_b2w_sync_item(source_key, source_name, source_hash=source_hash,
                              size_bytes=len(content), source_type="scan")],
    )
    await store.create_documents_and_job([doc], job)

    rag = _B2WFakeRAG()
    registry = _B2WFakeRegistry(rag)
    job_service = JobService(kb_service, store)
    from lightrag.api.job_worker import build_sync_executor

    executor = build_sync_executor(
        document_service=service, registry=registry,
        job_service=job_service, index_service=None,
    )
    await store.transition_job(kb_id, job.id, status="running", progress=0.0)
    await executor(job)
    final = await store.get_job(kb_id, job.id)
    assert final.status == "succeeded"
    item = (final.result or {})["items"][0]
    assert item["status"] == "skipped"  # hash match
    # Local-mode cleanup removed the staged file on terminal transition.
    assert not Path(staged_path).exists()


# ---------------------------------------------------------------------------
# Phase 3.1-D Writer D: artifact_cleanup_callback wiring on JobWorker.
# Mirrors the proven artifact_recovery_callback pattern (additive optional
# constructor param, invoked inside the shared recovery cadence with failure
# isolation). These cases exercise the contract directly so the server-level
# wiring test in test_artifact_cleanup_cadence.py can stay focused on
# construction/injection only.
# ---------------------------------------------------------------------------


class _RecoveryCycleJobService:
    """Minimal job_service double for direct _run_recovery_cycle tests.

    ``recover_orphan_jobs`` is the only method the cycle calls on job_service
    when there are no queued executors to drain. It records call order so
    tests can assert recovery-before-cleanup sequencing.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def claim_next_worker_job(self, **_kwargs: Any) -> None:
        return None

    async def recover_orphan_jobs(self, **_kwargs: Any) -> list[Any]:
        self.calls.append("orphan_recovery")
        return []


@pytest.mark.asyncio
async def test_recovery_cycle_runs_cleanup_after_recovery_callback() -> None:
    """Cleanup callback runs AFTER the artifact recovery callback in the cycle.

    Asserts the documented ordering: recovery first (terminalize any
    half-committed pipeline artifacts), then cleanup of drained artifacts.
    """

    job_service = _RecoveryCycleJobService()
    order: list[str] = []

    async def recovery_callback() -> None:
        order.append("artifact_recovery")

    async def cleanup_callback() -> None:
        order.append("artifact_cleanup")

    worker = JobWorker(
        job_service,  # type: ignore[arg-type]
        executors={},
        poll_interval_seconds=1.0,
        recovery_interval_seconds=1.0,
        artifact_recovery_callback=recovery_callback,
        artifact_cleanup_callback=cleanup_callback,
    )
    await worker._run_recovery_cycle()

    assert order == ["artifact_recovery", "artifact_cleanup"]
    assert job_service.calls == ["orphan_recovery"]


@pytest.mark.asyncio
async def test_cleanup_callback_failure_does_not_break_cycle_or_recovery() -> None:
    """A raising cleanup callback must not crash the recovery cycle nor block
    the artifact_recovery_callback on subsequent cycles."""

    job_service = _RecoveryCycleJobService()
    recovery_calls = 0
    cleanup_calls = 0
    recovery_done = asyncio.Event()

    async def recovery_callback() -> None:
        nonlocal recovery_calls
        recovery_calls += 1

    async def cleanup_callback() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        if cleanup_calls == 1:
            raise RuntimeError(
                "s3://cleanup:secret@bucket/.lightrag-scratch/private-cleanup"
            )
        recovery_done.set()

    worker = JobWorker(
        job_service,  # type: ignore[arg-type]
        executors={},
        poll_interval_seconds=0.05,
        recovery_interval_seconds=0.01,
        artifact_recovery_callback=recovery_callback,
        artifact_cleanup_callback=cleanup_callback,
    )
    with patch("lightrag.api.job_worker.logger.warning") as log_warning:
        worker.start()
        try:
            await asyncio.wait_for(recovery_done.wait(), timeout=2.0)
        finally:
            await worker.stop()

    # Both callbacks invoked at least twice across cycles: failure on the
    # first cleanup call must not stop the cadence.
    assert recovery_calls >= 2
    assert cleanup_calls >= 2
    # Failure was logged via the quiet helper; the redacted exception text
    # never leaks into log arguments.
    logged = repr(log_warning.call_args_list)
    assert "cleanup:secret" not in logged
    assert ".lightrag-scratch" not in logged
    assert "private-cleanup" not in logged


@pytest.mark.asyncio
async def test_cleanup_callback_none_default_is_backward_compatible() -> None:
    """Without the new param, JobWorker behaves exactly as before."""

    job_service = _RecoveryCycleJobService()
    worker = JobWorker(
        job_service,  # type: ignore[arg-type]
        executors={},
        recovery_interval_seconds=0,
    )
    assert worker._artifact_cleanup_callback is None
    assert worker._artifact_recovery_callback is None
    # Running the cycle must not raise even though no cleanup callback is set.
    await worker._run_recovery_cycle()
    assert job_service.calls == ["orphan_recovery"]


@pytest.mark.asyncio
async def test_cleanup_callback_skipped_when_recovery_callback_raises() -> None:
    """Even when recovery itself raises inside its quiet helper, the cleanup
    callback still gets a chance to run (full isolation between the two)."""

    job_service = _RecoveryCycleJobService()
    cleanup_calls = 0

    async def recovery_callback() -> None:
        raise RuntimeError(
            "s3://recovery:secret@bucket/.lightrag-scratch/private-recovery"
        )

    async def cleanup_callback() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1

    worker = JobWorker(
        job_service,  # type: ignore[arg-type]
        executors={},
        recovery_interval_seconds=1.0,
        artifact_recovery_callback=recovery_callback,
        artifact_cleanup_callback=cleanup_callback,
    )
    with (
        patch("lightrag.api.job_worker.logger.error") as log_error,
        patch("lightrag.api.job_worker.logger.warning") as log_warning,
    ):
        await worker._run_recovery_cycle()

    # Cleanup ran exactly once even though recovery raised.
    assert cleanup_calls == 1
    logged = repr((log_error.call_args_list, log_warning.call_args_list))
    assert "recovery:secret" not in logged
    assert ".lightrag-scratch" not in logged
    assert "private-recovery" not in logged


@pytest.mark.asyncio
async def test_job_worker_running_property_reflects_started_state() -> None:
    """The ``running`` property is the non-blocking signal /health uses."""

    job_service = _RecoveryCycleJobService()
    worker = JobWorker(
        job_service,  # type: ignore[arg-type]
        executors={},
        poll_interval_seconds=0.05,
        recovery_interval_seconds=0,  # disable the recovery timer; not under test
    )
    try:
        assert worker.running is False
        worker.start()
        # Started: main polling task exists and is not done.
        assert worker._task is not None
        assert worker.running is True
    finally:
        await worker.stop()
    # After stop, the task slot is cleared and running reports False again.
    assert worker._task is None
    assert worker.running is False


@pytest.mark.asyncio
async def test_cleanup_and_recovery_callbacks_share_one_recovery_timer() -> None:
    """Both callbacks ride the existing recovery timer — no duplicate tasks."""

    job_service = _RecoveryCycleJobService()

    async def recovery_callback() -> None:
        return None

    async def cleanup_callback() -> None:
        return None

    worker = JobWorker(
        job_service,  # type: ignore[arg-type]
        executors={},
        poll_interval_seconds=0.05,
        recovery_interval_seconds=0.05,
        artifact_recovery_callback=recovery_callback,
        artifact_cleanup_callback=cleanup_callback,
    )
    worker.start()
    try:
        # Exactly one polling task and one recovery task — never one per
        # callback. ``_artifact_cleanup_task`` must not exist.
        assert worker._task is not None
        assert worker._recovery_task is not None
        assert not hasattr(worker, "_artifact_cleanup_task")
        assert not hasattr(worker, "_artifact_recovery_task")
        # start() is idempotent: a second call must not spawn duplicate tasks.
        first_polling = worker._task
        first_recovery = worker._recovery_task
        worker.start()
        assert worker._task is first_polling
        assert worker._recovery_task is first_recovery
    finally:
        await worker.stop()
    assert worker._task is None
    assert worker._recovery_task is None
