"""Phase 3.2 Writer S: object-backed replace/sync staging.

Exercises the additive object-mode staging helpers on
:class:`DocumentLifecycleService` against a real SQLite Store A plus a
deterministic in-memory object storage that supports both immutable upload
proof and download. These tests prove:

* the request-time staging upload targets the *same* deterministic COW
  candidate key the frozen Core Writer B1 state machine uploads to, so the
  COW commit's ``upload_file_if_absent`` is idempotent (``created=False``);
* a durable worker can resume a replace job after request-process death by
  downloading the staged candidate object and re-driving
  ``execute_document_replace_cow`` (pre-commit crash, no local bytes);
* a durable worker can resume an aggregate ``sync`` job from per-item staging
  object URIs;
* the persisted job payload is metadata-only (an ``s3://`` URI, never a local
  path);
* a failed staging upload leaves no partial local scratch state;
* staging is object-backed, so worker resume has no local-path dependency
  (moved-root safe).

No live endpoint, no ``.env``, no repository state. The HTTP admission gate
and ``OBJECT_AUTHORITATIVE_LIFECYCLE_IMPLEMENTED`` remain closed/false; these
tests exercise the service/worker directly with fake object storage.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from lightrag.api.artifact_materialization import (
    ArtifactMaterializer,
    MaterializationLimits,
)
from lightrag.api.config import ArtifactCleanupConfig
from lightrag.api.document_lifecycle_service import (
    DocumentLifecycleService,
    DocumentReplacementSource,
    DocumentSourceInput,
)
from lightrag.api.job_service import JobService
from lightrag.api.kb_service import KnowledgeBaseService, utc_now_iso
from lightrag.api.metadata_store import (
    ArtifactRecord,
    DocumentRecord,
    JobRecord,
    SQLiteMetadataStore,
    document_source_generation_id,
)
from lightrag.api.object_storage import (
    ObjectStat,
    ObjectStorage,
    ObjectStorageError,
    ObjectStorageNotFoundError,
    ObjectReadback,
)
from lightrag.utils_pipeline import (
    reset_canonical_input_root_for_tests,
    set_canonical_input_root,
)

pytestmark = pytest.mark.offline

_NOW = datetime(2026, 8, 4, 12, 0, 0, tzinfo=timezone.utc)
_BUCKET = "stage-bucket"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _StagingFakeStorage(ObjectStorage):
    """Deterministic object storage with immutable upload proof + download."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.upload_proof_calls: list[tuple[str, str | None]] = []
        self.download_calls: list[str] = []
        self.upload_errors: dict[str, Exception] = {}

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def upload_file(
        self, local_path: Path, *, key: str, content_type: str | None = None
    ) -> str:
        uri = self.object_uri_for_key(key)
        self.files[uri] = local_path.read_bytes()
        return uri

    async def upload_file_if_absent(
        self,
        local_path: Path,
        *,
        key: str,
        content_type: str | None = None,
        expected_sha256: str | None = None,
    ) -> tuple[str, bool]:
        del content_type  # unused by the proof path
        uri = self.object_uri_for_key(key)
        if uri in self.upload_errors:
            raise self.upload_errors[uri]
        self.upload_proof_calls.append((uri, expected_sha256))
        if uri in self.files:
            return uri, False
        self.files[uri] = local_path.read_bytes()
        return uri, True

    async def download_file(self, object_uri: str, local_path: Path) -> None:
        self.download_calls.append(object_uri)
        if object_uri not in self.files:
            raise ObjectStorageNotFoundError()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(self.files[object_uri])

    def object_uri_for_key(self, key: str) -> str:
        return f"s3://{_BUCKET}/{key.lstrip('/')}"

    def object_prefix_uri_for_key(self, prefix: str) -> str:
        return f"s3://{_BUCKET}/{prefix.strip('/')}/"

    async def stat_object(self, object_uri: str) -> ObjectStat:
        rb = await self.inspect_object(object_uri)
        if not rb.present or rb.stat is None:
            raise ObjectStorageError(f"Missing fake object: {object_uri}")
        return rb.stat

    async def inspect_object(
        self, object_uri: str, *, version_id: str | None = None
    ) -> ObjectReadback:
        if object_uri not in self.files:
            return ObjectReadback(present=False)
        data = self.files[object_uri]
        return ObjectReadback(
            present=True,
            stat=ObjectStat(
                size=len(data),
                etag=f'"etag-{len(data)}"',
                last_modified=_NOW,
                checksum=f"sha256:{hashlib.sha256(data).hexdigest()}",
            ),
        )

    async def delete_uri(self, object_uri: str) -> bool:
        return self.files.pop(object_uri, None) is not None

    async def delete_prefix(self, prefix_uri: str) -> int:
        count = 0
        for uri in list(self.files):
            if uri.startswith(prefix_uri):
                self.files.pop(uri)
                count += 1
        return count

    async def delete_workspace(self, workspace: str) -> int:
        return 0

    def validate_document_file_uri(self, *args: Any, **kwargs: Any) -> None:
        return None

    def validate_document_prefix_uri(self, *args: Any, **kwargs: Any) -> None:
        return None


def _limits() -> MaterializationLimits:
    return MaterializationLimits(
        max_objects=1_000,
        max_total_bytes=64 * 1024 * 1024,
        stale_ttl_seconds=1,
    )


def _document(
    kb_id: str,
    document_id: str,
    *,
    workspace: str,
    source_generation_id: str = "srcg-old",
    artifact_id: str | None = "artifact-old",
) -> DocumentRecord:
    now = utc_now_iso()
    source_uri = (
        f"s3://{_BUCKET}/workspaces/{workspace}/documents/{document_id}/source/"
        f"generations/{source_generation_id}/source.pdf"
    )
    metadata: dict[str, Any] = {
        "source_object_uri": source_uri,
        "source_generation_id": source_generation_id,
    }
    if artifact_id is not None:
        metadata.update(
            {
                "current_sidecar_artifact_id": artifact_id,
                "current_artifact_ids": [artifact_id],
            }
        )
    return DocumentRecord(
        id=document_id,
        kb_id=kb_id,
        workspace=workspace,
        lightrag_doc_id=f"engine-{document_id}",
        source_type="upload",
        source_name="source.pdf",
        source_uri=source_uri,
        source_hash="sha256:" + "0" * 64,
        content_type="application/pdf",
        size_bytes=4,
        parser_hash="parser-old",
        index_hash="index-old",
        status="ready",
        enabled=True,
        archived=False,
        chunks_count=1,
        entity_count=0,
        relation_count=0,
        error_code=None,
        error_message=None,
        metadata=metadata,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )


def _artifact(
    document: DocumentRecord, artifact_id: str = "artifact-old"
) -> ArtifactRecord:
    now = utc_now_iso()
    object_uri = (
        f"s3://{_BUCKET}/workspaces/{document.workspace}/documents/{document.id}/"
        f"artifacts/raw/{artifact_id}/sidecar.json"
    )
    return ArtifactRecord(
        id=artifact_id,
        kb_id=document.kb_id,
        workspace=document.workspace,
        document_id=document.id,
        artifact_type="sidecar",
        uri=object_uri,
        checksum="sha256:" + "a" * 64,
        size_bytes=9,
        metadata={"object_uri": object_uri},
        created_at=now,
    )


def _job_record(
    kb_id: str,
    workspace: str,
    document_id: str,
    job_id: str,
    *,
    operation: str = "replace",
    payload: dict[str, Any] | None = None,
    status: str = "running",
) -> JobRecord:
    now = utc_now_iso()
    return JobRecord(
        id=job_id,
        kb_id=kb_id,
        workspace=workspace,
        batch_id=None,
        document_id=document_id,
        job_type=operation,
        status=status,
        stage="replacing" if operation == "replace" else "syncing",
        progress=0.1,
        total_items=1,
        completed_items=0,
        failed_items=0,
        idempotency_key=f"idem-{job_id}",
        config_version_id=None,
        config_hash=None,
        retry_count=0,
        max_retries=3,
        payload=payload or {"idempotency_fingerprint": "sha256:stage"},
        result=None,
        error_code=None,
        error_message=None,
        created_at=now,
        updated_at=now,
        queued_at=now,
        started_at=now,
        finished_at=None,
        cancelled_at=None,
    )


async def _put_artifact(store: SQLiteMetadataStore, artifact: ArtifactRecord) -> None:
    def write(conn):
        store._insert_artifact(conn, artifact)

    await store._write(write)


class _FakeRAG:
    """Records idempotent engine deletes."""

    def __init__(self) -> None:
        self.deleted: list[str | None] = []

    async def adelete_by_doc_id(
        self, doc_id: str, delete_llm_cache: bool = False
    ) -> Any:
        self.deleted.append(doc_id)
        from types import SimpleNamespace

        return SimpleNamespace(
            status="success",
            doc_id=doc_id,
            message="deleted",
            status_code=200,
            file_path="",
        )

    async def finalize_storages(self) -> None:
        return None

    async def adrop_all_storages(self) -> dict[str, Any]:
        return {"dropped": 0, "failed": 0, "errors": []}


class _FakeRegistry:
    def __init__(self, rag: Any) -> None:
        self._rag = rag
        self.max_parallel_parse_mineru = 1

    async def get(self, kb_id: str) -> Any:
        return self._rag

    async def acquire(self, kb_id: str) -> Any:
        return self._rag


@pytest.fixture
def stage_setup(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir(parents=True, exist_ok=True)
    reset_canonical_input_root_for_tests()
    set_canonical_input_root(root)

    async def _build():
        store = SQLiteMetadataStore(tmp_path / "stage.sqlite3")
        await store.initialize()
        kb_service = KnowledgeBaseService(tmp_path / "kbs.json")
        kb_id = f"kb_stage_{uuid4().hex[:10]}"
        kb_record = await kb_service.create(name=kb_id, kb_id=kb_id)
        workspace = kb_record.workspace
        generation = kb_record.generation
        await store.activate_kb_generation(kb_id, generation)
        storage = _StagingFakeStorage()
        materializer = ArtifactMaterializer(storage, input_root=root, limits=_limits())
        service = DocumentLifecycleService(
            kb_service,
            store,
            root,
            object_storage=storage,
            artifact_storage_mode="object",
            materializer=materializer,
            artifact_cleanup_config=ArtifactCleanupConfig(),
            clock=lambda: _NOW,
        )
        return service, store, storage, kb_id, workspace, generation, kb_service, root

    return _build


async def _seed_replaceable(
    stage_setup, *, document_id: str = "doc-1", job_id: str = "job-1"
):
    (
        service,
        store,
        storage,
        kb_id,
        workspace,
        generation,
        kb_service,
        root,
    ) = await stage_setup()
    document = _document(kb_id, document_id, workspace=workspace)
    artifact = _artifact(document)
    job = _job_record(kb_id, workspace, document_id, job_id)
    await store.create_documents_and_job([document], job)
    await _put_artifact(store, artifact)
    storage.files[document.metadata["source_object_uri"]] = b"old-bytes"
    storage.files[artifact.uri] = b"artifact"
    return (
        service,
        store,
        storage,
        kb_id,
        workspace,
        generation,
        document,
        artifact,
        job,
        kb_service,
    )


def _engine_callback(rag: _FakeRAG):
    async def _callback(kb_id, document_id, previous_lightrag_doc_id, engine_identity):
        await rag.adelete_by_doc_id(previous_lightrag_doc_id or "")
        # Return a JSON-serializable dict (the COW executor persists this in
        # document metadata as ``lightrag_delete_result``).
        return {
            "status": "success",
            "doc_id": previous_lightrag_doc_id,
            "deleted": previous_lightrag_doc_id,
        }

    return _callback


async def test_replace_staging_upload_is_idempotent_with_cow_candidate(stage_setup):
    """Staging upload + COW commit upload hit the same key; COW upload is a no-op."""
    (
        service,
        store,
        storage,
        kb_id,
        workspace,
        generation,
        document,
        artifact,
        job,
        kb_service,
    ) = await _seed_replaceable(stage_setup)

    payload = b"new-content"
    source_hash = "sha256:" + _sha256(payload)
    replacement = DocumentReplacementSource(
        source_name="source.pdf",
        content=payload,
        source_type="upload",
        source_hash=source_hash,
        content_type="application/pdf",
        size_bytes=len(payload),
    )

    # Issue the staging identity (token + deterministic generation id).
    attempt_token, source_generation_id = await service.prepare_object_replace_staging(
        kb_id,
        document.id,
        job_id=job.id,
        source_hash=replacement.source_hash,
    )
    # The generation id is the same Store A computes from the token.
    assert source_generation_id == document_source_generation_id(
        kb_id=kb_id,
        kb_generation=generation,
        document_id=document.id,
        job_id=job.id,
        attempt_token=attempt_token,
        source_hash=replacement.source_hash,
    )

    staging_uri = await service.stage_replacement_object(
        kb_id,
        document.id,
        job_id=job.id,
        source_generation_id=source_generation_id,
        replacement=replacement,
    )
    # The staging URI is the deterministic COW candidate URI.
    expected_key = (
        f"workspaces/{workspace}/documents/{document.id}/source/generations/"
        f"{source_generation_id}/source.pdf"
    )
    assert staging_uri == f"s3://{_BUCKET}/{expected_key}"
    assert staging_uri in storage.files
    # One upload proof call so far (the staging upload), created=True.
    assert storage.upload_proof_calls[-1] == (staging_uri, _sha256(payload))

    rag = _FakeRAG()
    # Re-drive the COW commit with the SAME token -> same candidate key ->
    # upload_file_if_absent returns created=False.
    result = await service.execute_document_replace_cow(
        kb_id,
        document.id,
        job_id=job.id,
        kb_generation=generation,
        new_source_type="upload",
        new_source_name="source.pdf",
        new_source_uri="",
        new_source_hash=source_hash,
        new_content_type="application/pdf",
        new_size_bytes=len(payload),
        replacement_content=payload,
        engine_delete=_engine_callback(rag),
        claim_token=attempt_token,
    )
    assert result.phase == "completed"

    # The committed source pointer equals the staged candidate URI.
    final = await store.get_document(kb_id, document.id)
    assert final.metadata["source_object_uri"] == staging_uri
    # Two upload proof calls to the same URI; the second was a no-op
    # (created=False) and the bytes are unchanged.
    candidate_calls = [c for c in storage.upload_proof_calls if c[0] == staging_uri]
    assert len(candidate_calls) == 2
    assert storage.files[staging_uri] == payload


async def test_worker_resume_replace_pre_commit_from_staging_object_uri(stage_setup):
    """Request-process death after staging; worker resumes from staging_object_uri."""
    (
        service,
        store,
        storage,
        kb_id,
        workspace,
        generation,
        document,
        artifact,
        job,
        kb_service,
    ) = await _seed_replaceable(stage_setup)

    payload = b"replacement-bytes-v2"
    source_hash = "sha256:" + _sha256(payload)
    replacement = DocumentReplacementSource(
        source_name="source.pdf",
        content=payload,
        source_type="upload",
        source_hash=source_hash,
        content_type="application/pdf",
        size_bytes=len(payload),
    )

    # Request time: stage the object and persist the URI + token.
    attempt_token, source_generation_id = await service.prepare_object_replace_staging(
        kb_id,
        document.id,
        job_id=job.id,
        source_hash=replacement.source_hash,
    )
    staging_uri = await service.stage_replacement_object(
        kb_id,
        document.id,
        job_id=job.id,
        source_generation_id=source_generation_id,
        replacement=replacement,
    )
    await store.update_job_payload_patch(
        kb_id,
        job.id,
        payload_patch={
            "staging_object_uri": staging_uri,
            "attempt_tokens": {document.id: attempt_token},
            "source_name": "source.pdf",
            "source_type": "upload",
            "source_hash": source_hash,
            "content_type": "application/pdf",
            "size_bytes": len(payload),
            "delete_source_file": True,
            "delete_artifacts": True,
            "delete_llm_cache": False,
            "auto_parse": False,
            "auto_index": False,
            "previous_lightrag_doc_id": document.lightrag_doc_id,
        },
    )

    # Simulate request-process death: the job stays queued (never run in
    # process). Model the orphan-recovery retry: queued -> claimed by worker.
    queued_job = await store.get_job(kb_id, job.id)
    assert queued_job.payload["staging_object_uri"] == staging_uri
    assert queued_job.payload["staging_object_uri"].startswith("s3://")
    # No local path leaked into the payload.
    for value in queued_job.payload.values():
        if isinstance(value, str):
            assert (
                "source" not in value or value.startswith("s3://") or "/" not in value
            )

    # Worker resume: download staged bytes and re-drive the COW commit.
    loaded = await service.load_staged_replacement_object(
        kb_id,
        document.id,
        job_id=job.id,
        staging_object_uri=staging_uri,
        source_name="source.pdf",
        source_hash=source_hash,
        content_type="application/pdf",
        size_bytes=len(payload),
        source_type="upload",
    )
    assert loaded is not None
    assert loaded.content == payload
    # The download materialized the bytes from the object URI.
    assert staging_uri in storage.download_calls

    rag = _FakeRAG()
    result = await service.execute_document_replace_cow(
        kb_id,
        document.id,
        job_id=job.id,
        kb_generation=generation,
        new_source_type="upload",
        new_source_name="source.pdf",
        new_source_uri="",
        new_source_hash=source_hash,
        new_content_type="application/pdf",
        new_size_bytes=len(payload),
        replacement_content=loaded.content,
        engine_delete=_engine_callback(rag),
        claim_token=queued_job.payload["attempt_tokens"][document.id],
    )
    assert result.phase == "completed"
    final = await store.get_document(kb_id, document.id)
    assert final.metadata["source_object_uri"] == staging_uri
    assert final.metadata["replace_phase"] == "completed"
    # Engine delete happened during resume.
    assert rag.deleted == [document.lightrag_doc_id]


async def test_worker_resume_replace_via_executor_pre_commit(stage_setup):
    """The durable replace executor resumes a pre-commit crash from staging_object_uri."""
    (
        service,
        store,
        storage,
        kb_id,
        workspace,
        generation,
        document,
        artifact,
        job,
        kb_service,
    ) = await _seed_replaceable(stage_setup, document_id="doc-exec", job_id="job-exec")

    payload = b"exec-replacement"
    source_hash = "sha256:" + _sha256(payload)

    # Request time: stage and persist.
    attempt_token, source_generation_id = await service.prepare_object_replace_staging(
        kb_id,
        "doc-exec",
        job_id="job-exec",
        source_hash=source_hash,
    )
    staging_uri = await service.stage_replacement_object(
        kb_id,
        "doc-exec",
        job_id="job-exec",
        source_generation_id=source_generation_id,
        replacement=DocumentReplacementSource(
            source_name="source.pdf",
            content=payload,
            source_type="upload",
            source_hash=source_hash,
            content_type="application/pdf",
            size_bytes=len(payload),
        ),
    )

    # Model the post-crash retry flow: the in-process task never ran (process
    # died right after staging). Orphan recovery fails the running job, then a
    # :retry puts it back to queued. Persist the staging URI + token so the
    # worker can resume.
    await store.transition_job(
        kb_id,
        "job-exec",
        status="failed",
        progress=1.0,
        failed_items=1,
        error_code="replace_failed",
    )
    await store.transition_job(kb_id, "job-exec", status="queued")
    await store.update_job_payload_patch(
        kb_id,
        "job-exec",
        payload_patch={
            "document_id": "doc-exec",
            "source_name": "source.pdf",
            "source_type": "upload",
            "source_hash": source_hash,
            "content_type": "application/pdf",
            "size_bytes": len(payload),
            "delete_source_file": True,
            "delete_artifacts": True,
            "delete_llm_cache": False,
            "auto_parse": False,
            "auto_index": False,
            "previous_lightrag_doc_id": document.lightrag_doc_id,
            "attempt_tokens": {"doc-exec": attempt_token},
            "staging_object_uri": staging_uri,
        },
    )
    queued_job = await store.get_job(kb_id, "job-exec")
    assert queued_job.payload["staging_object_uri"] == staging_uri

    job_service = JobService(kb_service, store)
    rag = _FakeRAG()
    registry = _FakeRegistry(rag)
    from lightrag.api.job_worker import build_replace_executor

    executor = build_replace_executor(
        document_service=service,
        registry=registry,
        job_service=job_service,
        index_service=None,
    )
    await store.transition_job(kb_id, "job-exec", status="running", progress=0.1)
    await executor(queued_job)

    final_job = await store.get_job(kb_id, "job-exec")
    assert final_job.status == "succeeded", final_job
    assert (final_job.result or {}).get("resumed_by_worker") is True
    # Document finalized; source pointer equals the staged candidate.
    doc = await store.get_document(kb_id, "doc-exec")
    assert doc.metadata["replace_phase"] == "completed"
    assert doc.metadata["source_object_uri"] == staging_uri
    assert rag.deleted == [document.lightrag_doc_id]


async def test_worker_resume_replace_without_staging_still_fails_cleanly(stage_setup):
    """A pre-commit crash WITHOUT staging_object_uri still fails as not_resumable."""
    (
        service,
        store,
        storage,
        kb_id,
        workspace,
        generation,
        document,
        artifact,
        job,
        kb_service,
    ) = await _seed_replaceable(
        stage_setup, document_id="doc-nostage", job_id="job-nostage"
    )

    # Model the post-crash retry: running -> failed -> queued, with NO
    # staging_object_uri persisted (the request never staged).
    await store.transition_job(
        kb_id,
        "job-nostage",
        status="failed",
        progress=1.0,
        failed_items=1,
        error_code="replace_failed",
    )
    await store.transition_job(kb_id, "job-nostage", status="queued")
    await store.update_job_payload_patch(
        kb_id,
        "job-nostage",
        payload_patch={
            "document_id": "doc-nostage",
            "source_name": "source.pdf",
            "source_type": "upload",
            "source_hash": "sha256:" + "x" * 64,
            "content_type": "application/pdf",
            "size_bytes": 1,
            "delete_source_file": True,
            "delete_artifacts": True,
            "delete_llm_cache": False,
            "auto_parse": False,
            "auto_index": False,
            "previous_lightrag_doc_id": document.lightrag_doc_id,
        },
    )
    queued_job = await store.get_job(kb_id, "job-nostage")

    job_service = JobService(kb_service, store)
    registry = _FakeRegistry(_FakeRAG())
    from lightrag.api.job_worker import build_replace_executor

    executor = build_replace_executor(
        document_service=service,
        registry=registry,
        job_service=job_service,
        index_service=None,
    )
    await store.transition_job(kb_id, "job-nostage", status="running", progress=0.1)
    await executor(queued_job)

    final_job = await store.get_job(kb_id, "job-nostage")
    assert final_job.status == "failed"
    assert final_job.error_code == "replace_not_resumable"


async def test_sync_staging_object_resume_loads_bytes(stage_setup):
    """Object-backed sync staging: worker loads per-item bytes from staging URIs."""
    (
        service,
        store,
        storage,
        kb_id,
        workspace,
        generation,
        kb_service,
        root,
    ) = await stage_setup()

    content = b"sync-source-bytes"
    source = DocumentSourceInput(
        source_name="synced.pdf",
        content=content,
        source_type="upload",
        content_type="application/pdf",
        metadata={"source_key": "key-1"},
    )
    staging_uri = await service.stage_sync_source_object(
        kb_id,
        batch_id="batch-sync",
        item_index=0,
        source=source,
    )
    assert staging_uri.startswith("s3://")
    assert staging_uri in storage.files
    assert storage.files[staging_uri] == content

    # Worker resume: load from the staging URI.
    loaded = await service.load_staged_sync_source_object(
        kb_id,
        staging_object_uri=staging_uri,
        source_name="synced.pdf",
        content_type="application/pdf",
        metadata={"source_key": "key-1"},
        expected_hash=_sha256(content),
        source_type="upload",
    )
    assert loaded is not None
    assert loaded.content == content
    assert loaded.metadata == {"source_key": "key-1"}

    # Absent staging object -> None (cleanly not resumable).
    missing = await service.load_staged_sync_source_object(
        kb_id,
        staging_object_uri="s3://stage-bucket/workspaces/ws/missing",
        source_name="synced.pdf",
        content_type="application/pdf",
        metadata={},
        expected_hash=_sha256(content),
        source_type="upload",
    )
    assert missing is None


async def test_sync_staging_checksum_mismatch_raises(stage_setup):
    """A staged sync object whose checksum no longer matches fails cleanly."""
    (
        service,
        store,
        storage,
        kb_id,
        workspace,
        generation,
        kb_service,
        root,
    ) = await stage_setup()
    content = b"will-be-corrupted"
    source = DocumentSourceInput(
        source_name="synced.pdf",
        content=content,
        source_type="upload",
        content_type="application/pdf",
        metadata={"source_key": "key-1"},
    )
    staging_uri = await service.stage_sync_source_object(
        kb_id,
        batch_id="batch-bad",
        item_index=0,
        source=source,
    )
    # Corrupt the staged object in place.
    storage.files[staging_uri] = b"different-bytes"
    with pytest.raises(ValueError, match="hash mismatch"):
        await service.load_staged_sync_source_object(
            kb_id,
            staging_object_uri=staging_uri,
            source_name="synced.pdf",
            content_type="application/pdf",
            metadata={},
            expected_hash=_sha256(content),
            source_type="upload",
        )


async def test_staging_upload_failure_leaves_no_partial_scratch(stage_setup):
    """A failed staging upload cleans the operation-scoped scratch file."""
    (
        service,
        store,
        storage,
        kb_id,
        workspace,
        generation,
        document,
        artifact,
        job,
        kb_service,
    ) = await _seed_replaceable(stage_setup, document_id="doc-fail", job_id="job-fail")
    payload = b"fail-content"
    source_hash = "sha256:" + _sha256(payload)
    attempt_token, source_generation_id = await service.prepare_object_replace_staging(
        kb_id,
        "doc-fail",
        job_id="job-fail",
        source_hash=source_hash,
    )
    expected_key = (
        f"workspaces/{workspace}/documents/doc-fail/source/generations/"
        f"{source_generation_id}/source.pdf"
    )
    expected_uri = f"s3://{_BUCKET}/{expected_key}"
    # Force the staging upload to fail.
    storage.upload_errors[expected_uri] = ObjectStorageError("upload blew up")

    replacement = DocumentReplacementSource(
        source_name="source.pdf",
        content=payload,
        source_type="upload",
        source_hash=source_hash,
        content_type="application/pdf",
        size_bytes=len(payload),
    )
    with pytest.raises(ObjectStorageError):
        await service.stage_replacement_object(
            kb_id,
            "doc-fail",
            job_id="job-fail",
            source_generation_id=source_generation_id,
            replacement=replacement,
        )
    # No partial state: the object was never created and no scratch file remains.
    assert expected_uri not in storage.files
    scratch_files = list(service._source_root.rglob(".replace-staging-*.tmp"))  # type: ignore[attr-defined]
    assert scratch_files == []


async def test_load_staged_replacement_absent_returns_none(stage_setup):
    """An absent staging object returns None (not resumable), not an error."""
    (
        service,
        store,
        storage,
        kb_id,
        workspace,
        generation,
        document,
        artifact,
        job,
        kb_service,
    ) = await _seed_replaceable(stage_setup, document_id="doc-abs", job_id="job-abs")
    loaded = await service.load_staged_replacement_object(
        kb_id,
        "doc-abs",
        job_id="job-abs",
        staging_object_uri="s3://stage-bucket/workspaces/ws/absent",
        source_name="source.pdf",
        source_hash="sha256:" + "0" * 64,
        content_type="application/pdf",
        size_bytes=4,
        source_type="upload",
    )
    assert loaded is None


async def test_replace_staging_is_moved_root_safe(stage_setup, tmp_path: Path):
    """Staging is object-backed: resume works across a moved input root.

    The staged bytes live at a deterministic object key; the local root only
    holds an operation-scoped scratch file that is removed after upload. A
    resuming process under a different root downloads the bytes from the object
    URI with no local-path dependency.
    """
    (
        service_a,
        store,
        storage,
        kb_id,
        workspace,
        generation,
        document,
        artifact,
        job,
        kb_service,
    ) = await _seed_replaceable(stage_setup, document_id="doc-mv", job_id="job-mv")

    payload = b"moved-root-bytes"
    source_hash = "sha256:" + _sha256(payload)
    (
        attempt_token,
        source_generation_id,
    ) = await service_a.prepare_object_replace_staging(
        kb_id,
        "doc-mv",
        job_id="job-mv",
        source_hash=source_hash,
    )
    staging_uri = await service_a.stage_replacement_object(
        kb_id,
        "doc-mv",
        job_id="job-mv",
        source_generation_id=source_generation_id,
        replacement=DocumentReplacementSource(
            source_name="source.pdf",
            content=payload,
            source_type="upload",
            source_hash=source_hash,
            content_type="application/pdf",
            size_bytes=len(payload),
        ),
    )
    # After staging, no local scratch file survives under root A.
    root_a = service_a._source_root  # type: ignore[attr-defined]
    assert list(root_a.rglob(".replace-staging-*.tmp")) == []

    # Construct a second service under a DIFFERENT root that shares the same
    # metadata store + object storage (modeling a moved checkout / new worker).
    root_b = tmp_path / "source-moved"
    root_b.mkdir(parents=True, exist_ok=True)
    reset_canonical_input_root_for_tests()
    set_canonical_input_root(root_b)
    materializer_b = ArtifactMaterializer(storage, input_root=root_b, limits=_limits())
    service_b = DocumentLifecycleService(
        kb_service,
        store,
        root_b,
        object_storage=storage,
        artifact_storage_mode="object",
        materializer=materializer_b,
        artifact_cleanup_config=ArtifactCleanupConfig(),
        clock=lambda: _NOW,
    )
    loaded = await service_b.load_staged_replacement_object(
        kb_id,
        "doc-mv",
        job_id="job-mv",
        staging_object_uri=staging_uri,
        source_name="source.pdf",
        source_hash=source_hash,
        content_type="application/pdf",
        size_bytes=len(payload),
        source_type="upload",
    )
    assert loaded is not None
    assert loaded.content == payload
    # The bytes came from the object URI, not from root A's filesystem.
    assert staging_uri in storage.download_calls


async def test_clear_staged_replacement_object_is_noop(stage_setup):
    """Object staging clear is an intentional no-op (immutable candidate retained)."""
    (
        service,
        store,
        storage,
        kb_id,
        workspace,
        generation,
        document,
        artifact,
        job,
        kb_service,
    ) = await _seed_replaceable(stage_setup, document_id="doc-clr", job_id="job-clr")
    payload = b"clear-bytes"
    source_hash = "sha256:" + _sha256(payload)
    attempt_token, source_generation_id = await service.prepare_object_replace_staging(
        kb_id,
        "doc-clr",
        job_id="job-clr",
        source_hash=source_hash,
    )
    staging_uri = await service.stage_replacement_object(
        kb_id,
        "doc-clr",
        job_id="job-clr",
        source_generation_id=source_generation_id,
        replacement=DocumentReplacementSource(
            source_name="source.pdf",
            content=payload,
            source_type="upload",
            source_hash=source_hash,
            content_type="application/pdf",
            size_bytes=len(payload),
        ),
    )
    assert staging_uri in storage.files
    # The no-op clear must NOT delete the staged candidate.
    await service.clear_staged_replacement_object(
        kb_id, "doc-clr", job_id="job-clr", staging_object_uri=staging_uri
    )
    assert staging_uri in storage.files
    await service.clear_staged_sync_sources_object(kb_id, batch_id="batch-x")
    assert storage.files  # unchanged
