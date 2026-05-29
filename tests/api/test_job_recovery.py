from __future__ import annotations

from pathlib import Path

import pytest

from lightrag.api.kb_service import KnowledgeBaseService, utc_now_iso
from lightrag.api.metadata_store import (
    DocumentRecord,
    JobRecord,
    SQLiteMetadataStore,
)

pytestmark = pytest.mark.offline


def _doc(kb_id: str, doc_id: str, *, status: str) -> DocumentRecord:
    now = utc_now_iso()
    return DocumentRecord(
        id=doc_id,
        kb_id=kb_id,
        workspace=f"kb_{kb_id}",
        lightrag_doc_id=None,
        source_type="upload",
        source_name=f"{doc_id}.pdf",
        source_uri=f"/tmp/{doc_id}.pdf",
        source_hash="sha256:abc",
        content_type="application/pdf",
        size_bytes=10,
        parser_hash="sha256:p",
        index_hash=None,
        status=status,
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


def _job(
    kb_id: str,
    job_id: str,
    *,
    job_type: str,
    status: str,
    document_id: str | None = None,
) -> JobRecord:
    now = utc_now_iso()
    return JobRecord(
        id=job_id,
        kb_id=kb_id,
        workspace=f"kb_{kb_id}",
        batch_id=None,
        document_id=document_id,
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
        payload={},
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


@pytest.mark.asyncio
async def test_recover_orphan_jobs_marks_running_jobs_failed(tmp_path: Path):
    store = SQLiteMetadataStore(tmp_path / "metadata.sqlite3")
    await store.initialize()

    # Seed three jobs and a document in mid-states
    doc = _doc("kb_recover", "doc_orphan", status="parsing")
    parse_job = _job(
        "kb_recover",
        "job_running",
        job_type="parse",
        status="queued",
        document_id="doc_orphan",
    )
    await store.create_documents_and_job([doc], parse_job)
    await store.transition_job(
        "kb_recover", "job_running", status="running", progress=0.5
    )
    await store.create_job(
        _job("kb_recover", "job_queued", job_type="parse", status="queued")
    )
    await store.create_job(
        _job(
            "kb_recover",
            "job_done",
            job_type="parse",
            status="queued",
            document_id=None,
        )
    )
    await store.transition_job(
        "kb_recover", "job_done", status="running"
    )
    await store.transition_job(
        "kb_recover", "job_done", status="succeeded", progress=1.0
    )

    recovered = await store.recover_orphan_jobs()
    recovered_ids = {job.id for job in recovered}
    assert recovered_ids == {"job_running", "job_queued"}
    for job in recovered:
        assert job.status == "failed"
        assert job.error_code == "worker_orphaned"

    # Already-succeeded job is untouched
    refreshed = await store.get_job("kb_recover", "job_done")
    assert refreshed.status == "succeeded"

    # Document stuck in parsing was reset to parse_failed
    refreshed_doc = await store.get_document("kb_recover", "doc_orphan")
    assert refreshed_doc.status == "parse_failed"
    assert refreshed_doc.error_code == "worker_orphaned"


@pytest.mark.asyncio
async def test_recover_orphan_jobs_resets_build_states(tmp_path: Path):
    store = SQLiteMetadataStore(tmp_path / "metadata.sqlite3")
    await store.initialize()

    doc = _doc("kb_build_recover", "doc_build", status="building")
    job = _job(
        "kb_build_recover",
        "job_build",
        job_type="build_kg",
        status="queued",
        document_id="doc_build",
    )
    await store.create_documents_and_job([doc], job)
    await store.transition_job(
        "kb_build_recover", "job_build", status="running"
    )

    await store.recover_orphan_jobs()
    refreshed = await store.get_document("kb_build_recover", "doc_build")
    assert refreshed.status == "build_failed"


@pytest.mark.asyncio
async def test_recover_orphan_jobs_resets_replace_states(tmp_path: Path):
    store = SQLiteMetadataStore(tmp_path / "metadata.sqlite3")
    await store.initialize()

    doc = _doc("kb_replace_recover", "doc_replace", status="replacing")
    job = _job(
        "kb_replace_recover",
        "job_replace",
        job_type="replace",
        status="queued",
        document_id="doc_replace",
    )
    await store.create_documents_and_job([doc], job)
    await store.transition_job(
        "kb_replace_recover", "job_replace", status="running"
    )

    await store.recover_orphan_jobs()
    refreshed = await store.get_document("kb_replace_recover", "doc_replace")
    assert refreshed.status == "replace_failed"
    assert refreshed.error_code == "worker_orphaned"


@pytest.mark.asyncio
async def test_recover_orphan_jobs_idempotent(tmp_path: Path):
    store = SQLiteMetadataStore(tmp_path / "metadata.sqlite3")
    await store.initialize()
    first = await store.recover_orphan_jobs()
    second = await store.recover_orphan_jobs()
    assert first == []
    assert second == []


@pytest.mark.asyncio
async def test_recover_orphan_jobs_via_job_service(tmp_path: Path):
    from lightrag.api.job_service import JobService

    kb_service = KnowledgeBaseService(tmp_path / "kb.json")
    await kb_service.initialize()
    await kb_service.create(kb_id="kb_recover_svc", name="Recover")
    store = SQLiteMetadataStore(tmp_path / "metadata.sqlite3")
    await store.initialize()
    job_service = JobService(kb_service, store)

    record = await kb_service.get("kb_recover_svc")
    job = _job(record.id, "job_to_recover", job_type="parse", status="queued")
    await store.create_job(job)
    await store.transition_job(record.id, job.id, status="running")

    recovered = await job_service.recover_orphan_jobs()
    assert {item.id for item in recovered} == {"job_to_recover"}
    refreshed = await job_service.get_job("kb_recover_svc", "job_to_recover")
    assert refreshed.status == "failed"
