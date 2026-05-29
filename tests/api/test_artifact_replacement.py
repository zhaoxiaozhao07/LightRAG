from __future__ import annotations

from pathlib import Path

import pytest

from lightrag.api.document_lifecycle_service import DocumentLifecycleService
from lightrag.api.kb_service import KnowledgeBaseService, utc_now_iso
from lightrag.api.metadata_store import (
    ArtifactRecord,
    DocumentRecord,
    JobRecord,
    SQLiteMetadataStore,
)

pytestmark = pytest.mark.offline


def _doc(record_id: str, *, workspace: str, status: str = "uploaded") -> DocumentRecord:
    now = utc_now_iso()
    return DocumentRecord(
        id="doc_replace",
        kb_id=record_id,
        workspace=workspace,
        lightrag_doc_id="doc-rt",
        source_type="upload",
        source_name="paper.pdf",
        source_uri="/tmp/paper.pdf",
        source_hash="sha256:src",
        content_type="application/pdf",
        size_bytes=10,
        parser_hash=None,
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


def _seed_job(record_id: str, workspace: str) -> JobRecord:
    now = utc_now_iso()
    return JobRecord(
        id="job_seed",
        kb_id=record_id,
        workspace=workspace,
        batch_id=None,
        document_id="doc_replace",
        job_type="parse",
        status="queued",
        stage="parsing",
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
async def test_complete_document_parse_replaces_existing_artifacts(tmp_path: Path):
    """Re-running parse (e.g. after a retry) must replace the prior artifacts
    rather than append, so a stale ``raw_dir`` from the previous attempt
    cannot leak into the post-retry artifact list."""
    kb_service = KnowledgeBaseService(tmp_path / "kb.json")
    await kb_service.initialize()
    record = await kb_service.create(kb_id="kb_replace", name="Replace")

    store = SQLiteMetadataStore(tmp_path / "metadata.sqlite3")
    await store.initialize()
    document = _doc(record.id, workspace=record.workspace)
    job = _seed_job(record.id, record.workspace)
    await store.create_documents_and_job([document], job)

    initial_artifacts = [
        ArtifactRecord(
            id="art_orig_old",
            kb_id=record.id,
            workspace=record.workspace,
            document_id=document.id,
            artifact_type="original",
            uri="/tmp/old-original",
            checksum=None,
            size_bytes=10,
            metadata={"old": True},
            created_at=utc_now_iso(),
        ),
        ArtifactRecord(
            id="art_blocks_old",
            kb_id=record.id,
            workspace=record.workspace,
            document_id=document.id,
            artifact_type="blocks",
            uri="/tmp/old-blocks.jsonl",
            checksum=None,
            size_bytes=20,
            metadata={"old": True},
            created_at=utc_now_iso(),
        ),
        ArtifactRecord(
            id="art_raw_old",
            kb_id=record.id,
            workspace=record.workspace,
            document_id=document.id,
            artifact_type="raw_dir",
            uri="/tmp/old-raw-dir",
            checksum=None,
            size_bytes=None,
            metadata={"is_directory": True, "old": True},
            created_at=utc_now_iso(),
        ),
    ]
    await store.complete_document_parse(
        record.id,
        document.id,
        parser_hash="sha256:p1",
        lightrag_doc_id="doc-rt",
        artifacts=initial_artifacts,
        metadata_patch={"last_parse_job_id": job.id},
    )
    pre_listed, _ = await store.list_document_artifacts(record.id, document.id)
    assert {item.id for item in pre_listed} == {
        "art_orig_old",
        "art_blocks_old",
        "art_raw_old",
    }

    # Simulate a retry: same parser_hash, fewer artifacts (no raw_dir)
    new_artifacts = [
        ArtifactRecord(
            id="art_orig_new",
            kb_id=record.id,
            workspace=record.workspace,
            document_id=document.id,
            artifact_type="original",
            uri="/tmp/new-original",
            checksum=None,
            size_bytes=10,
            metadata={"old": False},
            created_at=utc_now_iso(),
        ),
        ArtifactRecord(
            id="art_blocks_new",
            kb_id=record.id,
            workspace=record.workspace,
            document_id=document.id,
            artifact_type="blocks",
            uri="/tmp/new-blocks.jsonl",
            checksum=None,
            size_bytes=20,
            metadata={"old": False},
            created_at=utc_now_iso(),
        ),
    ]
    await store.complete_document_parse(
        record.id,
        document.id,
        parser_hash="sha256:p2",
        lightrag_doc_id="doc-rt",
        artifacts=new_artifacts,
        metadata_patch={"last_parse_job_id": "job_retry"},
    )
    post_listed, _ = await store.list_document_artifacts(record.id, document.id)
    ids = {item.id for item in post_listed}
    assert ids == {"art_orig_new", "art_blocks_new"}
    refreshed = await store.get_document(record.id, document.id)
    assert refreshed.parser_hash == "sha256:p2"
    assert refreshed.metadata.get("last_parse_job_id") == "job_retry"


@pytest.mark.asyncio
async def test_document_lifecycle_complete_parse_replaces_artifacts(tmp_path: Path):
    """End-to-end coverage via DocumentLifecycleService."""
    kb_service = KnowledgeBaseService(tmp_path / "kb.json")
    await kb_service.initialize()
    record = await kb_service.create(kb_id="kb_replace_e2e", name="Replace")

    store = SQLiteMetadataStore(tmp_path / "metadata.sqlite3")
    await store.initialize()
    service = DocumentLifecycleService(kb_service, store, tmp_path / "inputs")

    document = _doc(record.id, workspace=record.workspace, status="uploaded")
    job = _seed_job(record.id, record.workspace)
    await store.create_documents_and_job([document], job)

    first = [
        ArtifactRecord(
            id="art_first",
            kb_id=record.id,
            workspace=record.workspace,
            document_id=document.id,
            artifact_type="blocks",
            uri="/tmp/first.jsonl",
            checksum=None,
            size_bytes=8,
            metadata={},
            created_at=utc_now_iso(),
        )
    ]
    await store.complete_document_parse(
        record.id,
        document.id,
        parser_hash="sha256:p1",
        lightrag_doc_id="doc-rt",
        artifacts=first,
        metadata_patch={},
    )

    second = [
        ArtifactRecord(
            id="art_second",
            kb_id=record.id,
            workspace=record.workspace,
            document_id=document.id,
            artifact_type="blocks",
            uri="/tmp/second.jsonl",
            checksum=None,
            size_bytes=8,
            metadata={},
            created_at=utc_now_iso(),
        )
    ]
    await store.complete_document_parse(
        record.id,
        document.id,
        parser_hash="sha256:p1-retry",
        lightrag_doc_id="doc-rt",
        artifacts=second,
        metadata_patch={},
    )
    artifacts, total = await service.list_document_artifacts(
        record.id, document.id
    )
    assert total == 1
    assert {a.id for a in artifacts} == {"art_second"}
