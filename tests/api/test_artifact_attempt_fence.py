from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from lightrag.api.document_lifecycle_service import (
    DocumentLifecycleService,
    DocumentSourceInput,
)
from lightrag.api.index_build_service import IndexBuildService
from lightrag.api.kb_service import KnowledgeBaseService, utc_now_iso
from lightrag.api.metadata_store import (
    ArtifactPointerConflictError,
    DocumentAttemptOwnershipError,
    DocumentRecord,
    DocumentSnapshotConflictError,
    JobRecord,
    SQLiteMetadataStore,
    document_state_snapshot,
)
from lightrag.api.postgres_metadata_store import PostgresMetadataStore
from tests.api.test_artifact_storage_phase2a import _ParserRAG

pytestmark = pytest.mark.offline


class _FenceRAG:
    kb_active_index_hash = "sha256:index"

    def __init__(self) -> None:
        self.delete_calls: list[str] = []

    async def adelete_by_doc_id(self, document_id: str):
        self.delete_calls.append(document_id)
        return type("DeleteResult", (), {"status": "success", "message": "ok"})()


def _document(
    kb_id: str,
    workspace: str,
    *,
    status: str = "ready",
    parser_hash: str | None = "sha256:parser-s1",
    index_hash: str | None = "sha256:index",
) -> DocumentRecord:
    now = utc_now_iso()
    return DocumentRecord(
        id="doc_fenced",
        kb_id=kb_id,
        workspace=workspace,
        lightrag_doc_id="doc-lightrag-fenced",
        source_type="upload",
        source_name="fenced.pdf",
        source_uri="/inputs/fenced.pdf",
        source_hash="sha256:source-s1",
        content_type="application/pdf",
        size_bytes=10,
        parser_hash=parser_hash,
        index_hash=index_hash,
        status=status,
        enabled=True,
        archived=False,
        chunks_count=3,
        entity_count=2,
        relation_count=1,
        error_code=None,
        error_message=None,
        metadata={
            "current_parse_generation_id": "parse-generation-s1",
            "process_options": "",
        },
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )


def _seed_job(kb_id: str, workspace: str) -> JobRecord:
    now = utc_now_iso()
    return JobRecord(
        id="job_seed_fence",
        kb_id=kb_id,
        workspace=workspace,
        batch_id=None,
        document_id="doc_fenced",
        job_type="upload",
        status="succeeded",
        stage="uploading",
        progress=1.0,
        total_items=1,
        completed_items=1,
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
        queued_at=None,
        started_at=now,
        finished_at=now,
        cancelled_at=None,
    )


async def _build_services(
    tmp_path: Path, *, kb_id: str
) -> tuple[SQLiteMetadataStore, DocumentLifecycleService, IndexBuildService, _FenceRAG]:
    kb_service = KnowledgeBaseService(tmp_path / "kbs.json")
    record = await kb_service.create(kb_id=kb_id, name=kb_id)
    store = SQLiteMetadataStore(tmp_path / "metadata.sqlite3")
    await store.create_documents_and_job(
        [_document(record.id, record.workspace)],
        _seed_job(record.id, record.workspace),
    )
    document_service = DocumentLifecycleService(kb_service, store, tmp_path / "inputs")
    return store, document_service, IndexBuildService(document_service), _FenceRAG()


async def _simulate_reparse(
    store: SQLiteMetadataStore, kb_id: str, document_id: str
) -> DocumentRecord:
    document, _artifacts = await store.complete_document_parse(
        kb_id,
        document_id,
        parser_hash="sha256:parser-s2",
        lightrag_doc_id="doc-lightrag-fenced",
        artifacts=[],
        metadata_patch={
            "current_parse_generation_id": "parse-generation-s2",
            "current_build_generation_id": None,
            "current_sidecar_artifact_id": None,
            "current_blocks_artifact_id": None,
        },
    )
    return document


async def test_stale_skipped_build_claim_conflicts_before_engine_side_effects(
    tmp_path: Path,
):
    store, _document_service, index_service, rag = await _build_services(
        tmp_path, kb_id="kb_stale_skipped"
    )
    plan = await index_service.create_build_plan(
        "kb_stale_skipped", "doc_fenced", rag=rag
    )
    assert plan.skipped is True

    reparsed = await _simulate_reparse(store, "kb_stale_skipped", "doc_fenced")
    before_conflict = asdict(reparsed)
    with pytest.raises(DocumentSnapshotConflictError):
        await index_service.claim_build_queued(
            "kb_stale_skipped", job_id="job-stale-skipped", plan=plan
        )

    after_conflict = await store.get_document("kb_stale_skipped", "doc_fenced")
    assert asdict(after_conflict) == before_conflict
    assert plan.claim_token is None
    assert rag.delete_calls == []


async def test_stale_non_skipped_build_claim_conflicts_before_engine_delete(
    tmp_path: Path,
):
    store, _document_service, index_service, rag = await _build_services(
        tmp_path, kb_id="kb_stale_active"
    )
    plan = await index_service.create_build_plan(
        "kb_stale_active",
        "doc_fenced",
        rag=rag,
        force_rechunk=True,
    )
    assert plan.skipped is False

    reparsed = await _simulate_reparse(store, "kb_stale_active", "doc_fenced")
    before_conflict = asdict(reparsed)
    with pytest.raises(DocumentSnapshotConflictError):
        await index_service.claim_build_queued(
            "kb_stale_active", job_id="job-stale-active", plan=plan
        )

    after_conflict = await store.get_document("kb_stale_active", "doc_fenced")
    assert asdict(after_conflict) == before_conflict
    assert plan.claim_token is None
    assert rag.delete_calls == []


async def test_same_build_job_retry_rotates_token_and_fences_late_attempt(
    tmp_path: Path,
):
    store, _document_service, index_service, rag = await _build_services(
        tmp_path, kb_id="kb_build_retry_token"
    )
    old_plan = await index_service.create_build_plan(
        "kb_build_retry_token",
        "doc_fenced",
        rag=rag,
        force_rechunk=True,
    )
    await index_service.claim_build_queued(
        "kb_build_retry_token", job_id="job-retry", plan=old_plan
    )
    await index_service.mark_building(
        "kb_build_retry_token",
        "doc_fenced",
        job_id="job-retry",
        claim_token=old_plan.claim_token,
        plan=old_plan,
    )
    old_token = old_plan.claim_token
    assert old_token
    await index_service.release_build_if_owned(
        "kb_build_retry_token",
        "doc_fenced",
        job_id="job-retry",
        plan=old_plan,
        error_code="retryable",
        error_message="retry",
    )

    new_plan = await index_service.create_build_plan(
        "kb_build_retry_token",
        "doc_fenced",
        rag=rag,
        force_rechunk=True,
    )
    await index_service.claim_build_queued(
        "kb_build_retry_token", job_id="job-retry", plan=new_plan
    )
    await index_service.mark_building(
        "kb_build_retry_token",
        "doc_fenced",
        job_id="job-retry",
        claim_token=new_plan.claim_token,
        plan=new_plan,
    )
    new_token = new_plan.claim_token
    assert new_token and new_token != old_token

    with pytest.raises(DocumentAttemptOwnershipError):
        await index_service.complete_build(
            "kb_build_retry_token",
            "doc_fenced",
            job_id="job-retry",
            plan=old_plan,
            execution=None,
            run_result={
                "chunks_count": 99,
                "entity_count": 99,
                "relation_count": 99,
            },
        )
    with pytest.raises(DocumentAttemptOwnershipError):
        await index_service.fail_build(
            "kb_build_retry_token",
            "doc_fenced",
            job_id="job-retry",
            plan=old_plan,
            error_code="late_failure",
            error_message="late",
        )
    await index_service.release_build_if_owned(
        "kb_build_retry_token",
        "doc_fenced",
        job_id="job-retry",
        plan=old_plan,
        error_code="late_release",
        error_message="late",
    )

    current = await store.get_document("kb_build_retry_token", "doc_fenced")
    assert current.status == "building"
    assert current.metadata["current_build_job_id"] == "job-retry"
    assert current.metadata["current_build_claim_token"] == new_token
    assert current.error_code is None
    assert current.chunks_count == 3


async def test_local_direct_parse_lazily_attaches_execution_and_cleans_on_complete(
    tmp_path: Path,
):
    kb_service = KnowledgeBaseService(tmp_path / "kbs.json")
    await kb_service.create(kb_id="kb_local_direct", name="local")
    store = SQLiteMetadataStore(tmp_path / "metadata.sqlite3")
    service = DocumentLifecycleService(kb_service, store, tmp_path / "inputs")
    created = await service.create_source_batch(
        "kb_local_direct",
        [
            DocumentSourceInput(
                source_name="local.pdf",
                content=b"pdf-bytes",
                source_type="upload",
                content_type="application/pdf",
            )
        ],
    )
    document = created.documents[0]
    plan = await service.create_parse_plan(
        "kb_local_direct", document.id, parser_engine="mineru"
    )
    assert plan.claim_token is None
    assert plan.execution is None

    parsed_data = await service.run_parse(_ParserRAG(), plan)
    attached_execution = plan.execution
    assert attached_execution is not None
    assert attached_execution.source_path.is_file()
    result = await service.complete_parse(
        "kb_local_direct",
        document.id,
        job_id="job-local-direct",
        plan=plan,
        parsed_data=parsed_data,
    )

    assert plan.execution is None
    assert plan.claim_token
    assert result.document.status == "parsed"
    assert result.document.metadata["current_parse_generation_id"] == plan.claim_token


async def test_local_claim_build_without_plan_derives_snapshot_and_attempt(
    tmp_path: Path,
):
    store, _document_service, index_service, _rag = await _build_services(
        tmp_path, kb_id="kb_local_build_compat"
    )
    claimed = await index_service.claim_build_queued(
        "kb_local_build_compat",
        "doc_fenced",
        job_id="job-local-compat",
        plan=None,
    )
    token = claimed.metadata["pending_build_claim_token"]
    assert token

    building = await index_service.mark_building(
        "kb_local_build_compat",
        "doc_fenced",
        job_id="job-local-compat",
    )
    assert building.status == "building"
    assert building.metadata["current_build_claim_token"] == token


async def test_local_complete_build_derives_current_attempt_token(tmp_path: Path):
    store, _document_service, index_service, rag = await _build_services(
        tmp_path, kb_id="kb_local_build_complete"
    )
    plan = await index_service.create_build_plan(
        "kb_local_build_complete", "doc_fenced", rag=rag
    )
    await index_service.claim_build_queued(
        "kb_local_build_complete", job_id="job-local-build", plan=plan
    )
    token = plan.claim_token
    assert token
    await index_service.mark_building(
        "kb_local_build_complete",
        "doc_fenced",
        job_id="job-local-build",
        claim_token=token,
        plan=plan,
    )

    plan.claim_token = None
    completed = await index_service.complete_build(
        "kb_local_build_complete",
        "doc_fenced",
        job_id="job-local-build",
        plan=plan,
        execution=None,
        run_result={"skipped": True},
    )

    assert plan.claim_token == token
    assert completed.metadata["current_build_generation_id"] == token
    assert completed.status == "ready"
    await store.close()


async def test_local_fail_build_derives_current_attempt_token(tmp_path: Path):
    store, _document_service, index_service, rag = await _build_services(
        tmp_path, kb_id="kb_local_build_fail"
    )
    plan = await index_service.create_build_plan(
        "kb_local_build_fail", "doc_fenced", rag=rag, force_rechunk=True
    )
    await index_service.claim_build_queued(
        "kb_local_build_fail", job_id="job-local-build-fail", plan=plan
    )
    token = plan.claim_token
    assert token
    await index_service.mark_building(
        "kb_local_build_fail",
        "doc_fenced",
        job_id="job-local-build-fail",
        claim_token=token,
        plan=plan,
    )

    plan.claim_token = None
    failed = await index_service.fail_build(
        "kb_local_build_fail",
        "doc_fenced",
        job_id="job-local-build-fail",
        plan=plan,
        error_code="build_failed",
        error_message="failed",
    )

    assert plan.claim_token == token
    assert failed.status == "build_failed"
    assert failed.metadata["current_build_claim_token"] is None
    await store.close()


async def test_local_fail_parse_derives_current_attempt_token(tmp_path: Path):
    kb_service = KnowledgeBaseService(tmp_path / "kbs.json")
    await kb_service.create(kb_id="kb_local_parse_fail", name="local")
    store = SQLiteMetadataStore(tmp_path / "metadata.sqlite3")
    service = DocumentLifecycleService(kb_service, store, tmp_path / "inputs")
    created = await service.create_source_batch(
        "kb_local_parse_fail",
        [
            DocumentSourceInput(
                source_name="local.pdf",
                content=b"pdf-bytes",
                source_type="upload",
                content_type="application/pdf",
            )
        ],
    )
    document = created.documents[0]
    plan = await service.create_parse_plan(
        "kb_local_parse_fail", document.id, parser_engine="mineru"
    )
    await service.mark_parse_queued(
        "kb_local_parse_fail", document.id, job=created.job, plan=plan
    )
    await service.mark_parse_running(
        "kb_local_parse_fail",
        document.id,
        job_id=created.job.id,
        claim_token=plan.claim_token,
        plan=plan,
    )
    token = plan.claim_token
    assert token

    plan.claim_token = None
    failed = await service.fail_parse(
        "kb_local_parse_fail",
        document.id,
        job_id=created.job.id,
        plan=plan,
        error_code="parse_failed",
        error_message="failed",
    )

    assert plan.claim_token == token
    assert failed.status == "parse_failed"
    assert failed.metadata["current_parse_claim_token"] is None
    await store.close()


async def test_same_parse_job_retry_rotates_token_and_fences_late_attempt(
    tmp_path: Path,
):
    store = SQLiteMetadataStore(tmp_path / "metadata.sqlite3")
    document = _document(
        "kb_parse_retry_token",
        "workspace",
        status="uploaded",
        parser_hash=None,
        index_hash=None,
    )
    await store.create_documents_and_job(
        [document], _seed_job("kb_parse_retry_token", "workspace")
    )

    first = await store.mark_document_parse_queued(
        "kb_parse_retry_token",
        document.id,
        metadata_patch={"pending_parse_job_id": "job-retry"},
        expected_snapshot=document_state_snapshot(document),
    )
    old_token = first.metadata["pending_parse_claim_token"]
    await store.mark_document_parsing(
        "kb_parse_retry_token",
        document.id,
        metadata_patch={"current_parse_job_id": "job-retry"},
        job_id="job-retry",
        claim_token=old_token,
    )
    await store.release_document_parse_if_owned(
        "kb_parse_retry_token",
        document.id,
        job_id="job-retry",
        claim_token=old_token,
        error_code="retryable",
        error_message="retry",
    )

    retry_snapshot = document_state_snapshot(
        await store.get_document("kb_parse_retry_token", document.id)
    )
    second = await store.mark_document_parse_queued(
        "kb_parse_retry_token",
        document.id,
        metadata_patch={"pending_parse_job_id": "job-retry"},
        expected_snapshot=retry_snapshot,
    )
    new_token = second.metadata["pending_parse_claim_token"]
    assert new_token != old_token
    await store.mark_document_parsing(
        "kb_parse_retry_token",
        document.id,
        metadata_patch={"current_parse_job_id": "job-retry"},
        job_id="job-retry",
        claim_token=new_token,
    )

    with pytest.raises(DocumentAttemptOwnershipError):
        await store.complete_document_parse(
            "kb_parse_retry_token",
            document.id,
            parser_hash="sha256:late",
            lightrag_doc_id="doc-late",
            artifacts=[],
            metadata_patch={},
            job_id="job-retry",
            claim_token=old_token,
        )
    with pytest.raises(DocumentAttemptOwnershipError):
        await store.fail_document_parse(
            "kb_parse_retry_token",
            document.id,
            error_code="late_failure",
            error_message="late",
            metadata_patch={},
            job_id="job-retry",
            claim_token=old_token,
        )
    await store.release_document_parse_if_owned(
        "kb_parse_retry_token",
        document.id,
        job_id="job-retry",
        claim_token=old_token,
        error_code="late_release",
        error_message="late",
    )

    current = await store.get_document("kb_parse_retry_token", document.id)
    assert current.status == "parsing"
    assert current.metadata["current_parse_claim_token"] == new_token
    assert current.error_code is None


async def test_pointer_loser_release_is_owned_then_noops_after_takeover(
    tmp_path: Path,
):
    store, _document_service, _index_service, _rag = await _build_services(
        tmp_path, kb_id="kb_pointer_release"
    )
    document = await store.get_document("kb_pointer_release", "doc_fenced")
    expected_snapshot = document_state_snapshot(document)
    claimed = await store.claim_document_build_queued(
        "kb_pointer_release",
        document.id,
        metadata_patch={"pending_build_job_id": "job-loser"},
        expected_snapshot=expected_snapshot,
    )
    loser_token = claimed.metadata["pending_build_claim_token"]
    await store.mark_document_building(
        "kb_pointer_release",
        document.id,
        metadata_patch={"current_build_job_id": "job-loser"},
        job_id="job-loser",
        claim_token=loser_token,
    )
    await store.update_document(
        "kb_pointer_release",
        document.id,
        metadata_patch={"current_sidecar_artifact_id": "winner-sidecar"},
    )

    with pytest.raises(ArtifactPointerConflictError):
        await store.complete_document_build_with_artifact_promotion(
            "kb_pointer_release",
            document.id,
            index_hash="sha256:new-index",
            expected_current_sidecar_artifact_id=None,
            expected_current_blocks_artifact_id=None,
            current_sidecar_artifact_id="loser-sidecar",
            current_blocks_artifact_id=None,
            artifacts=[],
            metadata_patch={},
            job_id="job-loser",
            claim_token=loser_token,
            expected_snapshot=expected_snapshot,
        )
    released = await store.release_document_build_if_owned(
        "kb_pointer_release",
        document.id,
        job_id="job-loser",
        claim_token=loser_token,
        error_code="artifact_pointer_conflict",
        error_message="lost CAS",
    )
    assert released.status == "build_failed"
    assert released.metadata["current_build_job_id"] is None
    assert released.metadata["current_build_claim_token"] is None
    assert released.metadata["current_sidecar_artifact_id"] == "winner-sidecar"

    takeover_snapshot = document_state_snapshot(released)
    claimed_again = await store.claim_document_build_queued(
        "kb_pointer_release",
        document.id,
        metadata_patch={"pending_build_job_id": "job-loser-2"},
        expected_snapshot=takeover_snapshot,
    )
    old_token = claimed_again.metadata["pending_build_claim_token"]
    await store.mark_document_building(
        "kb_pointer_release",
        document.id,
        metadata_patch={"current_build_job_id": "job-loser-2"},
        job_id="job-loser-2",
        claim_token=old_token,
    )
    await store.update_document(
        "kb_pointer_release",
        document.id,
        metadata_patch={
            "current_build_job_id": "job-winner",
            "current_build_claim_token": "winner-token",
            "current_sidecar_artifact_id": "winner-sidecar-2",
        },
    )
    before_old_release = await store.get_document("kb_pointer_release", document.id)
    after_old_release = await store.release_document_build_if_owned(
        "kb_pointer_release",
        document.id,
        job_id="job-loser-2",
        claim_token=old_token,
        error_code="artifact_pointer_conflict",
        error_message="late loser",
    )
    assert asdict(after_old_release) == asdict(before_old_release)
    assert after_old_release.status == "building"
    assert after_old_release.metadata["current_build_job_id"] == "job-winner"
    assert after_old_release.metadata["current_build_claim_token"] == "winner-token"


async def test_postgres_build_claim_contract_locks_compares_and_rotates_token(
    monkeypatch: pytest.MonkeyPatch,
):
    store = PostgresMetadataStore(dsn="postgresql://unused")
    current = _document("kb-pg", "workspace", status="parsed", index_hash=None)
    lock_calls: list[bool] = []
    updates: list[dict[str, Any]] = []

    async def ensure() -> None:
        return None

    async def write(callback):
        return await callback(object())

    async def get_document(_conn, _kb_id, _document_id, *, for_update=False):
        lock_calls.append(for_update)
        return deepcopy(current)

    async def update_document_state(
        _conn,
        _kb_id,
        _document_id,
        *,
        status,
        metadata_patch,
        **_kwargs,
    ):
        updates.append(dict(metadata_patch))
        result = deepcopy(current)
        result.status = status
        result.metadata.update(metadata_patch)
        return result

    monkeypatch.setattr(store, "_ensure_initialized", ensure)
    monkeypatch.setattr(store, "_write", write)
    monkeypatch.setattr(store, "_get_document", get_document)
    monkeypatch.setattr(store, "_update_document_parse_state", update_document_state)

    stale_snapshot = document_state_snapshot(current)
    stale_snapshot["parser_hash"] = "sha256:stale"
    with pytest.raises(DocumentSnapshotConflictError):
        await store.claim_document_build_queued(
            "kb-pg",
            current.id,
            metadata_patch={"pending_build_job_id": "job-pg"},
            expected_snapshot=stale_snapshot,
        )
    assert updates == []

    first = await store.claim_document_build_queued(
        "kb-pg",
        current.id,
        metadata_patch={"pending_build_job_id": "job-pg"},
        expected_snapshot=document_state_snapshot(current),
    )
    second = await store.claim_document_build_queued(
        "kb-pg",
        current.id,
        metadata_patch={"pending_build_job_id": "job-pg"},
        expected_snapshot=document_state_snapshot(current),
    )
    first_token = first.metadata["pending_build_claim_token"]
    second_token = second.metadata["pending_build_claim_token"]
    assert first_token != second_token
    assert all(lock_calls)
    assert updates[0]["pending_build_job_id"] == "job-pg"
    assert updates[0]["current_build_job_id"] is None
