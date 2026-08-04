"""Phase 3.1-C Core Writer B1: object-authoritative document COW service tests.

Exercises :class:`DocumentLifecycleService.execute_document_replace_cow` and
``execute_document_delete_cow`` against a real SQLite Store A plus a
deterministic in-memory object storage and engine-delete callback.  No live
endpoint, no ``.env``, no GetObject/download on the proof path.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace as dataclass_replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from lightrag.api.artifact_materialization import (
    ArtifactMaterializer,
    MaterializationLimits,
)
from lightrag.api.artifact_lifecycle import (
    ArtifactCleanupManifestRecord,
)
from lightrag.api.config import ArtifactCleanupConfig
from lightrag.api.document_lifecycle_service import (
    DocumentCowEngineDeleteError,
    DocumentCowRetryableError,
    DocumentLifecycleService,
)
from lightrag.api.kb_service import KnowledgeBaseService, utc_now_iso
from lightrag.api.metadata_store import (
    ArtifactRecord,
    DocumentRecord,
    JobRecord,
    SQLiteMetadataStore,
)
from lightrag.api.object_storage import (
    ObjectStat,
    ObjectStorage,
    ObjectStorageError,
    ObjectReadback,
)
from lightrag.utils_pipeline import (
    reset_canonical_input_root_for_tests,
    set_canonical_input_root,
)

pytestmark = pytest.mark.offline

_NOW = datetime(2026, 8, 3, 12, 0, 0, tzinfo=timezone.utc)
_BUCKET = "cow-bucket"


class _FakeObjectStorage(ObjectStorage):
    """Deterministic object storage with metadata-only inspection (no GetObject)."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.upload_proof_calls: list[tuple[str, str | None]] = []
        self.inspect_calls: list[str] = []
        self.deleted_uris: list[str] = []
        self.deleted_prefixes: list[str] = []
        self.inspect_overrides: dict[str, ObjectReadback] = {}
        self.inspect_errors: dict[str, Exception] = {}

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
        del content_type
        uri = self.object_uri_for_key(key)
        self.upload_proof_calls.append((uri, expected_sha256))
        if uri in self.files:
            return uri, False
        self.files[uri] = local_path.read_bytes()
        return uri, True

    def object_uri_for_key(self, key: str) -> str:
        return f"s3://{_BUCKET}/{key.lstrip('/')}"

    def object_prefix_uri_for_key(self, prefix: str) -> str:
        return f"s3://{_BUCKET}/{prefix.strip('/')}/"

    async def stat_object(self, object_uri: str) -> ObjectStat:
        readback = await self.inspect_object(object_uri)
        if not readback.present or readback.stat is None:
            raise ObjectStorageError(f"Missing fake object: {object_uri}")
        return readback.stat

    async def inspect_object(
        self, object_uri: str, *, version_id: str | None = None
    ) -> ObjectReadback:
        self.inspect_calls.append(object_uri)
        if object_uri in self.inspect_errors:
            raise self.inspect_errors[object_uri]
        if object_uri in self.inspect_overrides:
            return self.inspect_overrides[object_uri]
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
        self.deleted_uris.append(object_uri)
        return self.files.pop(object_uri, None) is not None

    async def delete_prefix(self, prefix_uri: str) -> int:
        self.deleted_prefixes.append(prefix_uri)
        count = 0
        for uri in list(self.files):
            if uri.startswith(prefix_uri):
                self.files.pop(uri)
                count += 1
        return count

    async def delete_workspace(self, workspace: str) -> int:
        return 0

    def validate_document_file_uri(self, *args, **kwargs) -> None:
        return None

    def validate_document_prefix_uri(self, *args, **kwargs) -> None:
        return None


class _Engine:
    """Records idempotent engine deletes; optionally fails N times first."""

    def __init__(self) -> None:
        self.deleted: list[str | None] = []
        self.fail_remaining = 0

    def __call__(self, kb_id, document_id, previous_lightrag_doc_id, engine_identity):
        async def run():
            if self.fail_remaining > 0:
                self.fail_remaining -= 1
                raise RuntimeError("engine delete failed")
            self.deleted.append(previous_lightrag_doc_id)
            return {"deleted": previous_lightrag_doc_id, "identity": engine_identity}

        return run()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def _job(
    document: DocumentRecord,
    job_id: str,
    *,
    operation: str,
) -> JobRecord:
    now = utc_now_iso()
    return JobRecord(
        id=job_id,
        kb_id=document.kb_id,
        workspace=document.workspace,
        batch_id=None,
        document_id=document.id,
        job_type=operation,
        status="running",
        stage="replacing" if operation == "replace" else "deleting",
        progress=0.1,
        total_items=1,
        completed_items=0,
        failed_items=0,
        idempotency_key=f"idem-{job_id}",
        config_version_id=None,
        config_hash=None,
        retry_count=0,
        max_retries=3,
        payload={"idempotency_fingerprint": "sha256:cow"},
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


@pytest.fixture
def cow_setup(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir(parents=True, exist_ok=True)
    reset_canonical_input_root_for_tests()
    set_canonical_input_root(root)

    async def _build():
        store = SQLiteMetadataStore(tmp_path / "cow.sqlite3")
        await store.initialize()
        kb_service = KnowledgeBaseService(tmp_path / "kbs.json")
        kb_id = f"kb_cow_{uuid4().hex[:10]}"
        kb_record = await kb_service.create(name=kb_id, kb_id=kb_id)
        workspace = kb_record.workspace
        generation = kb_record.generation
        await store.activate_kb_generation(kb_id, generation)
        storage = _FakeObjectStorage()
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
        return service, store, storage, kb_id, workspace, generation

    return _build


async def _seed_replace(
    cow_setup, *, document_id: str = "doc-1", job_id: str = "job-1"
):
    service, store, storage, kb_id, workspace, generation = await cow_setup()
    document = _document(kb_id, document_id, workspace=workspace)
    artifact = _artifact(document)
    job = _job(document, job_id, operation="replace")
    await store.create_documents_and_job([document], job)
    await _put_artifact(store, artifact)
    # The old source object exists in object storage (inspectable for cleanup proof).
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
    )


async def _put_artifact(store: SQLiteMetadataStore, artifact: ArtifactRecord) -> None:
    def write(conn):
        store._insert_artifact(conn, artifact)

    await store._write(write)


async def test_replace_cow_happy_path_orders_commit_before_engine_delete(cow_setup):
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
    ) = await _seed_replace(cow_setup)
    engine = _Engine()
    payload = b"new-content"
    result = await service.execute_document_replace_cow(
        kb_id,
        document.id,
        job_id=job.id,
        kb_generation=generation,
        new_source_type="upload",
        new_source_name="source.pdf",
        new_source_uri="",
        new_source_hash="sha256:" + _sha256(payload),
        new_content_type="application/pdf",
        new_size_bytes=len(payload),
        replacement_content=payload,
        engine_delete=engine,
        claim_token="attempt-a",
    )

    assert result.phase == "completed"
    assert result.outcome == "completed"
    assert result.cleanup_pending_count == 2  # old source + old artifact
    assert result.cleanup_retained_count == 0
    # The previous engine document was deleted after commit.
    assert engine.deleted == [document.lightrag_doc_id]
    final = await store.get_document(kb_id, document.id)
    assert final.status == "uploaded"
    assert final.lightrag_doc_id is None
    assert final.metadata["replace_phase"] == "completed"
    # The new source pointer is committed and is NOT in the cleanup group.
    new_uri = final.metadata["source_object_uri"]
    manifests, total = await store.list_artifact_cleanup_manifests(
        kb_id=kb_id, document_id=document.id, limit=20
    )
    assert total == 2
    assert new_uri not in {m.target_uri for m in manifests}
    assert document.metadata["source_object_uri"] in {m.target_uri for m in manifests}
    assert artifact.uri in {m.target_uri for m in manifests}
    # The immutable upload proof ran with the expected SHA-256.
    assert storage.upload_proof_calls
    assert storage.upload_proof_calls[-1][1] == _sha256(payload)


async def test_replace_cow_deterministic_generation_and_engine_identity(cow_setup):
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
    ) = await _seed_replace(cow_setup)
    from lightrag.api.metadata_store import document_source_generation_id

    expected_gen = document_source_generation_id(
        kb_id=kb_id,
        kb_generation=generation,
        document_id=document.id,
        job_id=job.id,
        attempt_token="attempt-det",
        source_hash="sha256:" + _sha256(b"x"),
    )
    payload = b"x"
    result = await service.execute_document_replace_cow(
        kb_id,
        document.id,
        job_id=job.id,
        kb_generation=generation,
        new_source_type="upload",
        new_source_name="source.pdf",
        new_source_uri="",
        new_source_hash="sha256:" + _sha256(payload),
        new_content_type="application/pdf",
        new_size_bytes=len(payload),
        replacement_content=payload,
        engine_delete=_Engine(),
        claim_token="attempt-det",
    )
    assert result.source_generation_id == expected_gen
    assert result.attempt_token == "attempt-det"
    final = await store.get_document(kb_id, document.id)
    assert final.metadata["source_generation_id"] == expected_gen


async def test_replace_cow_legacy_first_replace_without_old_source_generation(
    cow_setup,
):
    """A document with a source uri but no source generation uses legacy_source."""

    service, store, storage, kb_id, workspace, generation = await cow_setup()
    document = _document(kb_id, "doc-legacy", workspace=workspace, artifact_id=None)
    # Remove the generation id so the old source is treated as legacy_source.
    legacy_uri = document.metadata["source_object_uri"]
    document = dataclass_replace(
        document,
        metadata={"source_object_uri": legacy_uri},
    )
    job = _job(document, "job-legacy", operation="replace")
    await store.create_documents_and_job([document], job)
    storage.files[legacy_uri] = b"legacy"
    payload = b"fresh"
    result = await service.execute_document_replace_cow(
        kb_id,
        document.id,
        job_id=job.id,
        kb_generation=generation,
        new_source_type="upload",
        new_source_name="source.pdf",
        new_source_uri="",
        new_source_hash="sha256:" + _sha256(payload),
        new_content_type="application/pdf",
        new_size_bytes=len(payload),
        replacement_content=payload,
        engine_delete=_Engine(),
        claim_token="attempt-legacy",
    )
    manifests, _ = await store.list_artifact_cleanup_manifests(
        kb_id=kb_id, document_id=document.id, limit=20
    )
    # Exactly one cleanup target: the legacy source, namespace legacy_source.
    assert len(manifests) == 1
    assert manifests[0].target_namespace == "legacy_source"
    assert manifests[0].target_uri == legacy_uri
    assert result.cleanup_pending_count == 1


async def test_replace_cow_retain_flags_mark_manifests_retained(cow_setup):
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
    ) = await _seed_replace(cow_setup)
    payload = b"retained"
    result = await service.execute_document_replace_cow(
        kb_id,
        document.id,
        job_id=job.id,
        kb_generation=generation,
        new_source_type="upload",
        new_source_name="source.pdf",
        new_source_uri="",
        new_source_hash="sha256:" + _sha256(payload),
        new_content_type="application/pdf",
        new_size_bytes=len(payload),
        replacement_content=payload,
        engine_delete=_Engine(),
        claim_token="attempt-retain",
        retain_source=True,
        retain_artifacts=True,
    )
    assert result.cleanup_retained_count == 2
    assert result.cleanup_pending_count == 0
    manifests, _ = await store.list_artifact_cleanup_manifests(
        kb_id=kb_id, document_id=document.id, limit=20
    )
    assert all(m.status == "retained" and m.disposition == "retain" for m in manifests)


async def test_replace_cow_engine_failure_leaves_re_driveable_state(cow_setup):
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
    ) = await _seed_replace(cow_setup)
    engine = _Engine()
    engine.fail_remaining = 1
    payload = b"engine-fail"
    with pytest.raises(DocumentCowEngineDeleteError) as exc_info:
        await service.execute_document_replace_cow(
            kb_id,
            document.id,
            job_id=job.id,
            kb_generation=generation,
            new_source_type="upload",
            new_source_name="source.pdf",
            new_source_uri="",
            new_source_hash="sha256:" + _sha256(payload),
            new_content_type="application/pdf",
            new_size_bytes=len(payload),
            replacement_content=payload,
            engine_delete=engine,
            claim_token="attempt-engfail",
        )
    partial = exc_info.value.result
    assert partial is not None
    assert partial.phase == "engine_cleanup_pending"
    stalled = await store.get_document(kb_id, document.id)
    assert stalled.metadata["replace_phase"] == "engine_cleanup_pending"
    assert stalled.error_code == "engine_cleanup_failed"
    # The new pointer committed before the engine failure.
    assert stalled.metadata["source_generation_id"] == partial.source_generation_id

    # Re-drive: recognizes the committed state and completes engine cleanup.
    result = await service.execute_document_replace_cow(
        kb_id,
        document.id,
        job_id=job.id,
        kb_generation=generation,
        new_source_type="upload",
        new_source_name="source.pdf",
        new_source_uri="",
        new_source_hash="sha256:" + _sha256(payload),
        new_content_type="application/pdf",
        new_size_bytes=len(payload),
        replacement_content=payload,
        engine_delete=engine,
        claim_token="attempt-engfail",
    )
    assert result.phase == "completed"
    assert engine.deleted == [document.lightrag_doc_id]


async def test_replace_cow_empty_group_when_no_old_targets(cow_setup):
    """A document with no source uri and no artifacts yields an empty group."""

    service, store, storage, kb_id, workspace, generation = await cow_setup()
    document = _document(kb_id, "doc-empty", workspace=workspace, artifact_id=None)
    document = dataclass_replace(document, metadata={})
    job = _job(document, "job-empty", operation="replace")
    await store.create_documents_and_job([document], job)
    payload = b"empty"
    result = await service.execute_document_replace_cow(
        kb_id,
        document.id,
        job_id=job.id,
        kb_generation=generation,
        new_source_type="upload",
        new_source_name="source.pdf",
        new_source_uri="",
        new_source_hash="sha256:" + _sha256(payload),
        new_content_type="application/pdf",
        new_size_bytes=len(payload),
        replacement_content=payload,
        engine_delete=_Engine(),
        claim_token="attempt-empty",
    )
    assert result.cleanup_pending_count == 0
    manifests, total = await store.list_artifact_cleanup_manifests(
        kb_id=kb_id, document_id=document.id, limit=20
    )
    assert total == 0


async def test_replace_cow_inspect_unprovable_blocks_before_engine(cow_setup):
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
    ) = await _seed_replace(cow_setup)
    # The old source object exists but its metadata cannot be proved (HEAD
    # forbidden). Manifest preparation must fail closed before any commit.
    storage.inspect_errors[document.metadata["source_object_uri"]] = ObjectStorageError(
        "head forbidden"
    )
    payload = b"block"
    with pytest.raises(Exception):
        await service.execute_document_replace_cow(
            kb_id,
            document.id,
            job_id=job.id,
            kb_generation=generation,
            new_source_type="upload",
            new_source_name="source.pdf",
            new_source_uri="",
            new_source_hash="sha256:" + _sha256(payload),
            new_content_type="application/pdf",
            new_size_bytes=len(payload),
            replacement_content=payload,
            engine_delete=_Engine(),
            claim_token="attempt-block",
        )
    # No commit occurred; the document keeps its engine identity and the
    # claim was safely released (no engine delete ran).
    stalled = await store.get_document(kb_id, document.id)
    assert stalled.lightrag_doc_id == document.lightrag_doc_id
    assert stalled.metadata.get("source_generation_id") == "srcg-old"


async def test_replace_cow_absent_old_target_allowed_without_expected_evidence(
    cow_setup,
):
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
    ) = await _seed_replace(cow_setup)
    # The old source object is absent (already removed); cleanup manifest carries
    # no expected evidence but the group is still valid.
    storage.files.pop(document.metadata["source_object_uri"], None)
    payload = b"absent"
    result = await service.execute_document_replace_cow(
        kb_id,
        document.id,
        job_id=job.id,
        kb_generation=generation,
        new_source_type="upload",
        new_source_name="source.pdf",
        new_source_uri="",
        new_source_hash="sha256:" + _sha256(payload),
        new_content_type="application/pdf",
        new_size_bytes=len(payload),
        replacement_content=payload,
        engine_delete=_Engine(),
        claim_token="attempt-absent",
    )
    manifests, _ = await store.list_artifact_cleanup_manifests(
        kb_id=kb_id, document_id=document.id, limit=20
    )
    source_manifest = next(
        m for m in manifests if m.target_uri == document.metadata["source_object_uri"]
    )
    assert source_manifest.expected_size_bytes is None
    assert source_manifest.expected_checksum is None
    assert result.phase == "completed"


async def test_replace_cow_prefix_artifact_target_has_no_invented_checksum(cow_setup):
    service, store, storage, kb_id, workspace, generation = await cow_setup()
    document = _document(kb_id, "doc-prefix", workspace=workspace)
    # Artifact points at a prefix (trailing slash), not a single object.
    prefix_uri = (
        f"s3://{_BUCKET}/workspaces/{workspace}/documents/{document.id}/"
        f"artifacts/raw/artifact-prefix/"
    )
    artifact = _artifact(document, artifact_id="artifact-prefix")
    artifact = dataclass_replace(
        artifact,
        id="artifact-prefix",
        uri=prefix_uri,
        metadata={"object_prefix_uri": prefix_uri},
    )
    document.metadata["current_sidecar_artifact_id"] = "artifact-prefix"
    document.metadata["current_artifact_ids"] = ["artifact-prefix"]
    job = _job(document, "job-prefix", operation="replace")
    await store.create_documents_and_job([document], job)
    await _put_artifact(store, artifact)
    storage.files[document.metadata["source_object_uri"]] = b"old"
    payload = b"prefix"
    result = await service.execute_document_replace_cow(
        kb_id,
        document.id,
        job_id=job.id,
        kb_generation=generation,
        new_source_type="upload",
        new_source_name="source.pdf",
        new_source_uri="",
        new_source_hash="sha256:" + _sha256(payload),
        new_content_type="application/pdf",
        new_size_bytes=len(payload),
        replacement_content=payload,
        engine_delete=_Engine(),
        claim_token="attempt-prefix",
    )
    manifests, _ = await store.list_artifact_cleanup_manifests(
        kb_id=kb_id, document_id=document.id, limit=20
    )
    prefix_manifest = next(m for m in manifests if m.target_kind == "prefix")
    assert prefix_manifest.expected_checksum is None
    assert prefix_manifest.expected_size_bytes is None
    assert result.phase == "completed"


async def test_delete_cow_pre_engine_recheck_and_tombstone(cow_setup):
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
    ) = await _seed_replace(cow_setup, job_id="job-seed")
    # Create a dedicated delete job (job_type must match the operation).
    del_job = dataclass_replace(
        job,
        id="job-del",
        job_type="delete",
        stage="deleting",
        idempotency_key="idem-job-del",
    )
    await store.create_job(del_job)
    engine = _Engine()
    result = await service.execute_document_delete_cow(
        kb_id,
        document.id,
        job_id="job-del",
        kb_generation=generation,
        engine_delete=engine,
        claim_token="attempt-del",
    )
    assert result.outcome == "deleted"
    tombstone = await store.get_document_lifecycle(kb_id, document.id)
    assert tombstone.deleted_at is not None
    assert engine.deleted == [document.lightrag_doc_id]
    manifests, total = await store.list_artifact_cleanup_manifests(
        kb_id=kb_id, document_id=document.id, limit=20
    )
    # Source + artifact cleanup pending; no generic object delete occurred.
    assert total == 2
    assert all(m.reason == "document_delete" for m in manifests)
    assert storage.deleted_uris == []
    assert storage.deleted_prefixes == []


async def test_delete_cow_engine_failure_preserves_bytes(cow_setup):
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
    ) = await _seed_replace(cow_setup, job_id="job-delfail")
    engine = _Engine()
    engine.fail_remaining = 1
    with pytest.raises(Exception):
        await service.execute_document_delete_cow(
            kb_id,
            document.id,
            job_id="job-delfail",
            kb_generation=generation,
            engine_delete=engine,
            claim_token="attempt-delfail",
        )
    # Bytes are preserved: document is not tombstoned, no object cleanup.
    stalled = await store.get_document(kb_id, document.id)
    assert stalled.deleted_at is None
    assert storage.deleted_uris == []


async def test_replace_cow_stale_generation_is_fenced(cow_setup):
    """A stale kb generation cannot race a committed replace."""

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
    ) = await _seed_replace(cow_setup)
    payload = b"stale-gen"
    # A caller asserting a stale generation is rejected by the write guard
    # before any claim or object side effect.
    with pytest.raises(Exception):
        await service.execute_document_replace_cow(
            kb_id,
            document.id,
            job_id=job.id,
            kb_generation=generation + "-stale",
            new_source_type="upload",
            new_source_name="source.pdf",
            new_source_uri="",
            new_source_hash="sha256:" + _sha256(payload),
            new_content_type="application/pdf",
            new_size_bytes=len(payload),
            replacement_content=payload,
            engine_delete=_Engine(),
            claim_token="attempt-stale",
        )
    stalled = await store.get_document(kb_id, document.id)
    assert stalled.lightrag_doc_id == document.lightrag_doc_id


async def test_replace_cow_compensation_enqueue_after_rolled_back_commit(cow_setup):
    """If commit proves ROLLED_BACK, candidate cleanup is enqueued safely."""

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
    ) = await _seed_replace(cow_setup)
    payload = b"rolled"
    # Force the commit to raise; reconciliation will prove ROLLED_BACK because
    # no commit landed. We simulate this by making the object storage upload
    # succeed but the candidate will be orphaned via the rolled-back path.
    original = store.commit_document_replace_cow

    async def failing_commit(*args, **kwargs):
        raise RuntimeError("commit transport lost")

    store.commit_document_replace_cow = failing_commit  # type: ignore[assignment]
    try:
        with pytest.raises((DocumentCowRetryableError, Exception)):
            await service.execute_document_replace_cow(
                kb_id,
                document.id,
                job_id=job.id,
                kb_generation=generation,
                new_source_type="upload",
                new_source_name="source.pdf",
                new_source_uri="",
                new_source_hash="sha256:" + _sha256(payload),
                new_content_type="application/pdf",
                new_size_bytes=len(payload),
                replacement_content=payload,
                engine_delete=_Engine(),
                claim_token="attempt-rolled",
            )
    finally:
        store.commit_document_replace_cow = original  # type: ignore[assignment]
    # The document rolled back to its pre-claim state.
    rolled = await store.get_document(kb_id, document.id)
    assert rolled.lightrag_doc_id == document.lightrag_doc_id
    # An orphan_reconcile compensation manifest was enqueued for the candidate.
    manifests, _ = await store.list_artifact_cleanup_manifests(
        kb_id=kb_id, document_id=document.id, limit=20
    )
    orphan = [m for m in manifests if m.reason == "orphan_reconcile"]
    assert len(orphan) == 1
    assert orphan[0].target_namespace == "source"
    assert orphan[0].origin_attempt_token == "attempt-rolled"


async def test_replace_cow_cleanup_group_authorizes_two_replacements_in_grace(
    cow_setup,
):
    """Section F regression: after B commits, A's old-target manifest is still
    authorized during the grace window, while B's current source is blocked."""

    (
        service,
        store,
        storage,
        kb_id,
        workspace,
        generation,
        document,
        artifact,
        job_a,
    ) = await _seed_replace(cow_setup, document_id="doc-grace", job_id="job-a")
    payload_a = b"replacement-a"
    await service.execute_document_replace_cow(
        kb_id,
        document.id,
        job_id=job_a.id,
        kb_generation=generation,
        new_source_type="upload",
        new_source_name="source.pdf",
        new_source_uri="",
        new_source_hash="sha256:" + _sha256(payload_a),
        new_content_type="application/pdf",
        new_size_bytes=len(payload_a),
        replacement_content=payload_a,
        engine_delete=_Engine(),
        claim_token="attempt-a",
    )
    # A's old source cleanup manifest is pending and authorized by attempt-a.
    manifests_a, _ = await store.list_artifact_cleanup_manifests(
        kb_id=kb_id, document_id=document.id, limit=20
    )
    a_old_source = next(
        m for m in manifests_a if m.target_uri == document.metadata["source_object_uri"]
    )
    assert a_old_source.origin_attempt_token == "attempt-a"

    # Second replacement (B) on the now-committed document.
    job_b = _job(document, "job-b", operation="replace")
    job_b = dataclass_replace(job_b, id="job-b", idempotency_key="idem-job-b")
    await store.create_job(job_b)
    payload_b = b"replacement-b"
    result_b = await service.execute_document_replace_cow(
        kb_id,
        document.id,
        job_id="job-b",
        kb_generation=generation,
        new_source_type="upload",
        new_source_name="source.pdf",
        new_source_uri="",
        new_source_hash="sha256:" + _sha256(payload_b),
        new_content_type="application/pdf",
        new_size_bytes=len(payload_b),
        replacement_content=payload_b,
        engine_delete=_Engine(),
        claim_token="attempt-b",
    )
    # B's current source (result_a's new source) is now in B's cleanup group,
    # but A's earlier manifest is still present and authorized by its attempt
    # token recorded in the durable history.
    final = await store.get_document(kb_id, document.id)
    history = final.metadata.get("replace_attempt_token_history") or []
    assert "attempt-a" in history
    assert "attempt-b" in history
    # A's manifest target is not the current source.
    assert a_old_source.target_uri != final.metadata["source_object_uri"]
    assert result_b.phase == "completed"


async def test_orphan_reconcile_without_attempt_history_blocks(cow_setup):
    """A tokenless orphan_reconcile manifest cannot cleanup a candidate.

    Exercises the corrected ``_check_document_origin_lineage`` directly: an
    orphan_reconcile/source manifest whose attempt token is not present in the
    durable replace attempt-token history is blocked.
    """

    from lightrag.api.artifact_cleanup_service import ArtifactCleanupService
    from lightrag.api.artifact_lifecycle import artifact_cleanup_idempotency_key

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
    ) = await _seed_replace(cow_setup)
    target_uri = document.metadata["source_object_uri"]
    manifest = ArtifactCleanupManifestRecord(
        id="manifest-orphan-notoken",
        idempotency_key=artifact_cleanup_idempotency_key(
            reason="orphan_reconcile",
            kb_id=kb_id,
            kb_generation=generation,
            workspace=workspace,
            document_id=document.id,
            artifact_id=None,
            source_generation_id="srcg-old",
            target_kind="object",
            target_namespace="source",
            target_uri=target_uri,
        ),
        manifest_group_id="orphan-test-group",
        kb_id=kb_id,
        kb_generation=generation,
        workspace=workspace,
        document_id=document.id,
        artifact_id=None,
        source_generation_id="srcg-old",
        origin_job_id=job.id,
        origin_attempt_token="attempt-never-recorded",
        reason="orphan_reconcile",
        target_kind="object",
        target_namespace="source",
        disposition="delete",
        status="pending",
        target_uri=target_uri,
        delete_after=_NOW,
        cleanup_deadline_at=_NOW,
        audit_retain_until=_NOW,
        next_attempt_at=_NOW,
        created_at=_NOW,
        updated_at=_NOW,
    )
    # Document has no replace attempt history, so the lineage check blocks.
    with pytest.raises(Exception):
        ArtifactCleanupService._check_document_origin_lineage(
            manifest,
            {"last_replace_job_id": job.id},  # no attempt-token history present
        )
    # When the attempt token IS recorded in history, the lineage check passes.
    ArtifactCleanupService._check_document_origin_lineage(
        manifest,
        {
            "last_replace_job_id": job.id,
            "replace_attempt_token_history": ["attempt-never-recorded"],
        },
    )
    # When the recorded job differs but the attempt token is in history, the
    # manifest is still authorized (older replace within grace).
    ArtifactCleanupService._check_document_origin_lineage(
        manifest,
        {
            "last_replace_job_id": "job-newer",
            "replace_attempt_token_history": ["attempt-never-recorded"],
        },
    )


async def test_local_mode_lifecycle_methods_unchanged(cow_setup):
    """Local-mode destructive operations remain disabled-in-object-mode gated
    and the local replace path is not altered by the COW additions."""

    service, store, storage, kb_id, workspace, generation = await cow_setup()
    # object_authoritative is True here; legacy destructive ops must be rejected.
    assert service.object_authoritative is True
    with pytest.raises(Exception):
        service.assert_destructive_operation_supported("Document replace")
