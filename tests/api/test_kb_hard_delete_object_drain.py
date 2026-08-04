"""Phase 3.1-D Writer C: KB hard-delete object-mode drain (manifest-driven).

These tests exercise the object-authoritative workspace drain directly against
the service. After the B-2 fix, ``assert_hard_delete_supported`` is coupled to
the ``OBJECT_AUTHORITATIVE_LIFECYCLE_IMPLEMENTED`` capability constant: in
production the constant is still ``False`` so the route boundary returns HTTP
503; these tests open the gate via the ``_hard_delete_capability_enabled``
indirection (autouse fixture below) so the drain path is reachable, mirroring
the post-Gate-3 state.

Drain contract (parent-frozen):

* Replace the legacy ``object_storage.delete_workspace`` bulk deletion with a
  manifest-driven drain driven by one workspace-prefix
  ``reason="kb_delete"`` manifest whose ``origin_job_id`` equals the
  hard-delete job id (and therefore ``lifecycle.delete_job_id``).
* Drain uses checkpoint ``stage="draining"`` + ``object_cleanup_pending``
  result field; the exclusive fence is released between attempts and
  re-acquired per resume.
* Empty proof: zero pending manifests for the KB generation followed by one
  ``list_objects_page(prefix_uri, max_keys=1000)`` returning zero entries.
* After verified-empty, before ``purge_kb_metadata``, call the additive
  ``store.delete_artifact_recovery_cursor`` (catch+log if unavailable).
* Local mode is unchanged: ``delete_workspace`` still runs there.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from lightrag.api.config import ArtifactCleanupConfig
from lightrag.api import kb_deletion_service
from lightrag.api.kb_deletion_service import (
    KBDeletionService,
    KBHardDeleteUnsupportedError,
    _CLEAR_DRAINING_STAGE,
    _CLEAR_FINALIZING_STAGE,
)
from lightrag.api.kb_service import (
    KnowledgeBaseNotFoundError,
    KnowledgeBaseRecord,
    KnowledgeBaseService,
    utc_now_iso,
)
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry
from lightrag.api.metadata_store import (
    DocumentRecord,
    JobRecord,
    SQLiteMetadataStore,
)
from lightrag.api.object_storage import ObjectStorage

# Real S3 backend wired against the in-memory fake aioboto3 client. Using the
# production S3ObjectStorage here exercises the real list_objects_page /
# object_prefix_uri_for_key / verified_delete_cleanup_target paths that the
# cleanup service will exercise in production.
from tests.api.test_object_storage_s3 import _FakeS3State, _make_storage

pytestmark = pytest.mark.offline

_BUCKET = "lightrag-kb"
_PREFIX = "kb"


@pytest.fixture(autouse=True)
def _enable_hard_delete_capability(monkeypatch: pytest.MonkeyPatch):
    """Open the hard-delete gate for object-drain tests (B-2 fix).

    The frozen ``OBJECT_AUTHORITATIVE_LIFECYCLE_IMPLEMENTED`` constant stays
    ``False``; these tests inject ``True`` via the
    ``_hard_delete_capability_enabled`` indirection so the real
    ``assert_hard_delete_supported`` gate opens and the manifest-driven drain
    runs through the production code path. This mirrors the post-Gate-3 state
    without touching the capability constant. The gate-closed regression test
    overrides this by re-patching the helper to return ``False``.
    """

    monkeypatch.setattr(
        kb_deletion_service,
        "_hard_delete_capability_enabled",
        lambda: True,
    )


# ---------------------------------------------------------------------------
# Fakes mirroring tests/api/test_kb_hard_delete.py (kept inline so this file
# stays inside Writer-C exclusive ownership).
# ---------------------------------------------------------------------------


class FakeRAG:
    def __init__(self, record: KnowledgeBaseRecord, probe: "BuilderProbe"):
        self.workspace = record.workspace
        self.record = record
        self.probe = probe
        self.finalized = False
        self.dropped = False

    async def finalize_storages(self) -> None:
        self.finalized = True

    async def adrop_all_storages(self) -> dict[str, Any]:
        self.dropped = True
        self.probe.drop_calls += 1
        return {"dropped": 9, "failed": 0, "errors": []}


class BuilderProbe:
    def __init__(self) -> None:
        self.instances: list[FakeRAG] = []
        self.records: list[KnowledgeBaseRecord] = []
        self.finalized: list[FakeRAG] = []
        self.drop_calls = 0

    async def build(self, record: KnowledgeBaseRecord) -> FakeRAG:
        self.records.append(record)
        rag = FakeRAG(record, self)
        self.instances.append(rag)
        return rag

    async def finalize(self, rag: Any) -> None:
        assert isinstance(rag, FakeRAG)
        await rag.finalize_storages()
        self.finalized.append(rag)


class CountingObjectStorage(ObjectStorage):
    """Object storage proxy that records drain-relevant calls.

    For object-mode regression coverage we keep using the production
    S3ObjectStorage (so list_objects_page behaves exactly as in prod) but
    wrap it to observe ``delete_workspace`` invocations.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.delete_workspace_calls: list[str] = []
        self.list_objects_page_calls: list[tuple[str, int]] = []

    async def initialize(self) -> None:
        return await self._inner.initialize()

    async def close(self) -> None:
        return await self._inner.close()

    def object_prefix_uri_for_key(self, prefix: str) -> str:
        return self._inner.object_prefix_uri_for_key(prefix)

    def object_uri_for_key(self, key: str) -> str:
        return self._inner.object_uri_for_key(key)

    async def list_objects_page(
        self, prefix_uri, *, max_keys=1000, continuation_token=None
    ):
        self.list_objects_page_calls.append((prefix_uri, max_keys))
        return await self._inner.list_objects_page(
            prefix_uri,
            max_keys=max_keys,
            continuation_token=continuation_token,
        )

    async def delete_workspace(self, workspace: str) -> int:
        self.delete_workspace_calls.append(workspace)
        return await self._inner.delete_workspace(workspace)

    async def delete_uri(self, object_uri: str) -> bool:
        return await self._inner.delete_uri(object_uri)

    async def delete_prefix(self, prefix_uri: str) -> int:
        return await self._inner.delete_prefix(prefix_uri)

    def validate_cleanup_target(self, target_uri, **kwargs):
        return self._inner.validate_cleanup_target(target_uri, **kwargs)

    async def verified_delete_cleanup_target(self, target, **kwargs):
        return await self._inner.verified_delete_cleanup_target(target, **kwargs)


# ---------------------------------------------------------------------------
# Environment builder.
# ---------------------------------------------------------------------------


def _doc(kb_id: str, doc_id: str, *, workspace: str) -> DocumentRecord:
    now = utc_now_iso()
    return DocumentRecord(
        id=doc_id,
        kb_id=kb_id,
        workspace=workspace,
        lightrag_doc_id=f"lr-{doc_id}",
        source_type="upload",
        source_name=f"{doc_id}.bin",
        source_uri=f"/tmp/{doc_id}.bin",
        source_hash="sha256:" + "1" * 64,
        content_type="application/octet-stream",
        size_bytes=4,
        parser_hash="parser-" + doc_id,
        index_hash="index-" + doc_id,
        status="ready",
        enabled=True,
        archived=False,
        chunks_count=1,
        entity_count=0,
        relation_count=0,
        error_code=None,
        error_message=None,
        metadata={},
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )


def _seed_job(kb_id: str, workspace: str, *, job_id: str) -> JobRecord:
    now = utc_now_iso()
    return JobRecord(
        id=job_id,
        kb_id=kb_id,
        workspace=workspace,
        batch_id=None,
        document_id=None,
        job_type="upload",
        status="succeeded",
        stage=None,
        progress=1.0,
        total_items=1,
        completed_items=1,
        failed_items=0,
        idempotency_key=f"idem-{job_id}",
        config_version_id=None,
        config_hash=None,
        retry_count=0,
        max_retries=3,
        payload={},
        result={"documents_created": 1},
        error_code=None,
        error_message=None,
        created_at=now,
        updated_at=now,
        queued_at=now,
        started_at=now,
        finished_at=now,
        cancelled_at=None,
    )


async def _build_object_env(
    tmp_path: Path,
    *,
    kb_id: str,
    create_files: bool = True,
) -> tuple[
    KnowledgeBaseService,
    SQLiteMetadataStore,
    LightRAGInstanceRegistry,
    KBDeletionService,
    BuilderProbe,
    KnowledgeBaseRecord,
    Path,
    Path,
    CountingObjectStorage,
    _FakeS3State,
]:
    """Build an object-authoritative hard-delete environment.

    A real SQLiteMetadataStore + KnowledgeBaseService + LightRAGInstanceRegistry
    is used; only the S3 transport is faked so list_objects_page and the cleanup
    service's verified-delete primitives behave exactly as in production.
    """

    kb_service = KnowledgeBaseService(tmp_path / "kb.json")
    await kb_service.initialize()
    record = await kb_service.create(kb_id=kb_id, name=kb_id)

    store = SQLiteMetadataStore(tmp_path / "metadata.sqlite3")
    await store.initialize()
    document = _doc(record.id, f"doc_{kb_id}", workspace=record.workspace)
    seed = _seed_job(record.id, record.workspace, job_id=f"seed_{kb_id}")
    await store.create_documents_and_job([document], seed)

    input_root = tmp_path / "inputs"
    input_workspace = input_root / record.workspace
    working_root = tmp_path / "working"
    working_workspace = working_root / record.workspace
    if create_files:
        (input_workspace / document.id).mkdir(parents=True)
        (input_workspace / document.id / "source.bin").write_bytes(b"raw")
        working_workspace.mkdir(parents=True)
        (working_workspace / "graph.json").write_text("{}", encoding="utf-8")

    s3_storage, s3_state, _session = _make_storage(
        bucket=_BUCKET, prefix=_PREFIX, page_size=1000
    )
    await s3_storage.initialize()
    counting_storage = CountingObjectStorage(s3_storage)

    probe = BuilderProbe()
    registry = LightRAGInstanceRegistry(kb_service, probe.build, probe.finalize)
    deletion_service = KBDeletionService(
        kb_service,
        store,
        registry,
        input_root=input_root,
        working_dir=working_root,
        object_storage=counting_storage,
        artifact_storage_mode="object",
        artifact_cleanup_config=ArtifactCleanupConfig(),
    )
    # The ``assert_hard_delete_supported`` gate is now coupled to the
    # ``OBJECT_AUTHORITATIVE_LIFECYCLE_IMPLEMENTED`` capability constant (B-2
    # fix). The autouse ``_enable_hard_delete_capability`` fixture below flips
    # the ``_hard_delete_capability_enabled`` indirection to True for every
    # object-drain test, so the real gate opens and the drain path is
    # exercised through the production code path rather than bypassed.
    return (
        kb_service,
        store,
        registry,
        deletion_service,
        probe,
        record,
        input_workspace,
        working_workspace,
        counting_storage,
        s3_state,
    )


async def _soft_delete(
    kb_service: KnowledgeBaseService, record: KnowledgeBaseRecord
) -> KnowledgeBaseRecord:
    return await kb_service.delete(
        record.id,
        expected_generation=record.generation,
    )


def _workspace_prefix_key(workspace: str) -> str:
    return f"{_PREFIX}/workspaces/{workspace}/"


def _put_workspace_object(state: _FakeS3State, workspace: str, name: str) -> None:
    state.objects[(_BUCKET, f"{_PREFIX}/workspaces/{workspace}/{name}")] = b"x"


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_object_drain_enqueues_workspace_prefix_manifest(tmp_path: Path):
    (
        kb_service,
        store,
        _registry,
        deletion_service,
        probe,
        record,
        _input_ws,
        _working_ws,
        counting_storage,
        _s3_state,
    ) = await _build_object_env(tmp_path, kb_id="kb_drain_enqueue")

    await _soft_delete(kb_service, record)
    result = await deletion_service.hard_delete(
        record.id, expected_generation=record.generation
    )

    # First call must checkpoint to draining because the just-enqueued manifest
    # is still pending.
    assert result.object_cleanup_pending is True
    assert result.job.status == "running"
    assert result.job.stage == _CLEAR_DRAINING_STAGE
    assert result.errors == []

    manifests, total = await store.list_artifact_cleanup_manifests(
        kb_id=record.id, kb_generation=record.generation
    )
    assert total == 1
    assert len(manifests) == 1
    manifest = manifests[0]
    assert manifest.reason == "kb_delete"
    assert manifest.target_kind == "prefix"
    assert manifest.target_namespace == "workspace"
    assert manifest.disposition == "delete"
    assert manifest.status == "pending"
    assert manifest.origin_job_id == result.job.id
    assert manifest.kb_id == record.id
    assert manifest.kb_generation == record.generation
    assert manifest.workspace == record.workspace
    assert manifest.document_id is None
    assert manifest.artifact_id is None
    assert manifest.source_generation_id is None
    expected_target = counting_storage.object_prefix_uri_for_key(
        f"workspaces/{record.workspace}"
    )
    assert manifest.target_uri == expected_target
    assert manifest.target_uri.endswith("/")

    # Engine/local-compat cleanup must still run; object deletion must NOT.
    assert probe.drop_calls == 1
    assert counting_storage.delete_workspace_calls == []


@pytest.mark.asyncio
async def test_object_drain_polls_until_empty_and_completes_full_ordering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (
        kb_service,
        store,
        _registry,
        deletion_service,
        probe,
        record,
        input_workspace,
        working_workspace,
        counting_storage,
        s3_state,
    ) = await _build_object_env(tmp_path, kb_id="kb_drain_complete")
    await _soft_delete(kb_service, record)
    # Seed two workspace objects so the empty proof is meaningful.
    _put_workspace_object(s3_state, record.workspace, "doc1/source.bin")
    _put_workspace_object(s3_state, record.workspace, "doc2/artifact.json")

    first = await deletion_service.hard_delete(
        record.id, expected_generation=record.generation
    )
    assert first.object_cleanup_pending is True
    assert first.job.stage == _CLEAR_DRAINING_STAGE
    assert counting_storage.list_objects_page_calls == []
    # The just-enqueued manifest is pending, so the empty proof was NOT run.

    # Simulate the cleanup service: claim + succeed the manifest, then remove
    # the workspace objects so the verified-empty listing returns zero entries.
    await _drive_cleanup_service_to_success(store, record)
    for key in list(s3_state.objects):
        if key[0] == _BUCKET and key[1].startswith(
            _workspace_prefix_key(record.workspace)
        ):
            s3_state.objects.pop(key, None)

    # Instrument the tail to verify the exact ordering.
    order: list[str] = []
    real_purge_meta = store.purge_kb_metadata
    real_complete = store.complete_kb_deletion
    real_catalog_purge = kb_service.purge
    real_delete_cursor = getattr(store, "delete_artifact_recovery_cursor", None)

    async def observed_purge(kb_id, generation=None, *, delete_job_id=None):
        order.append("metadata_purge")
        return await real_purge_meta(
            kb_id, generation=generation, delete_job_id=delete_job_id
        )

    async def observed_complete(kb_id, generation, delete_job_id):
        order.append("complete_lifecycle")
        return await real_complete(kb_id, generation, delete_job_id)

    async def observed_catalog(kb_id, **kwargs):
        order.append("catalog_purge")
        return await real_catalog_purge(kb_id, **kwargs)

    monkeypatch.setattr(store, "purge_kb_metadata", observed_purge)
    monkeypatch.setattr(store, "complete_kb_deletion", observed_complete)
    monkeypatch.setattr(kb_service, "purge", observed_catalog)
    cursor_calls: list[tuple[str, str]] = []

    async def observed_cursor(kb_id, kb_generation):
        cursor_calls.append((kb_id, kb_generation))
        order.append("cursor_removal")
        if real_delete_cursor is None:
            return True
        return await real_delete_cursor(kb_id, kb_generation)

    if real_delete_cursor is not None:
        monkeypatch.setattr(store, "delete_artifact_recovery_cursor", observed_cursor)
    else:
        monkeypatch.setattr(
            store, "delete_artifact_recovery_cursor", observed_cursor, raising=False
        )

    # Second resume: manifests succeeded -> list_objects_page returns 0 ->
    # verified empty -> cursor removal -> purge -> complete -> catalog purge.
    resumed = await deletion_service.resume_hard_delete(first.job)

    assert resumed.errors == []
    assert resumed.object_cleanup_pending is False
    assert resumed.job.status == "succeeded"
    assert resumed.job.stage == _CLEAR_FINALIZING_STAGE
    assert resumed.purged_catalog is True
    assert resumed.cleared_object_storage is True
    assert resumed.job.result is not None
    assert resumed.job.result["object_cleanup_pending"] is False
    assert resumed.job.result["cleared_object_storage"] is True

    # One list_objects_page call proving the empty workspace.
    assert counting_storage.list_objects_page_calls == [
        (
            counting_storage.object_prefix_uri_for_key(
                f"workspaces/{record.workspace}"
            ),
            1000,
        )
    ]
    # Cursor removed BEFORE metadata purge (Phase 3.1-D ordering requirement).
    if real_delete_cursor is not None:
        assert order[:1] == ["cursor_removal"]
        assert cursor_calls == [(record.id, record.generation)]
    assert "metadata_purge" in order
    assert "complete_lifecycle" in order
    assert "catalog_purge" in order
    cursor_idx = order.index("cursor_removal") if "cursor_removal" in order else -1
    purge_idx = order.index("metadata_purge")
    complete_idx = order.index("complete_lifecycle")
    catalog_idx = order.index("catalog_purge")
    if cursor_idx >= 0:
        assert cursor_idx < purge_idx
    assert purge_idx < complete_idx < catalog_idx

    # Engine/local-compat cleanup ran exactly once (physical stage); it must
    # not be repeated on the second resume.
    assert probe.drop_calls == 1
    assert counting_storage.delete_workspace_calls == []

    with pytest.raises(KnowledgeBaseNotFoundError):
        await kb_service.get(record.id, include_deleted=True)


@pytest.mark.asyncio
async def test_object_drain_partial_prefix_delete_retries_until_empty(
    tmp_path: Path,
):
    """Objects remaining after cleanup's first pass -> retry -> then empty."""

    (
        kb_service,
        store,
        _registry,
        deletion_service,
        _probe,
        record,
        _input_ws,
        _working_ws,
        counting_storage,
        s3_state,
    ) = await _build_object_env(tmp_path, kb_id="kb_drain_partial")
    await _soft_delete(kb_service, record)
    _put_workspace_object(s3_state, record.workspace, "leftover.bin")

    first = await deletion_service.hard_delete(
        record.id, expected_generation=record.generation
    )
    assert first.object_cleanup_pending is True
    # Manifest just enqueued -> drain not empty.

    # Simulate partial: mark the manifest succeeded but leave the object in
    # place. The drain must observe the leftover object and stay in draining.
    await _drive_cleanup_service_to_success(store, record)
    second = await deletion_service.resume_hard_delete(first.job)
    assert second.object_cleanup_pending is True
    assert second.job.stage == _CLEAR_DRAINING_STAGE
    assert counting_storage.list_objects_page_calls  # proof was attempted

    # Now remove the leftover and resume once more -> verified empty.
    for key in list(s3_state.objects):
        if key == (_BUCKET, f"{_PREFIX}/workspaces/{record.workspace}/leftover.bin"):
            s3_state.objects.pop(key, None)
    third = await deletion_service.resume_hard_delete(first.job)
    assert third.errors == []
    assert third.object_cleanup_pending is False
    assert third.job.status == "succeeded"


@pytest.mark.asyncio
async def test_object_drain_checkpoints_pending_and_resumes_via_resume_hard_delete(
    tmp_path: Path,
):
    """object_cleanup_pending checkpoint + resume via resume_hard_delete."""

    (
        kb_service,
        store,
        _registry,
        deletion_service,
        _probe,
        record,
        _input_ws,
        _working_ws,
        counting_storage,
        s3_state,
    ) = await _build_object_env(tmp_path, kb_id="kb_drain_checkpoint")
    await _soft_delete(kb_service, record)
    _put_workspace_object(s3_state, record.workspace, "a.bin")

    first = await deletion_service.hard_delete(
        record.id, expected_generation=record.generation
    )
    assert first.job.status == "running"
    assert first.job.stage == _CLEAR_DRAINING_STAGE
    assert first.job.result is not None
    assert first.job.result["object_cleanup_pending"] is True

    persisted = await store.get_job(record.id, first.job.id)
    assert persisted.status == "running"
    assert persisted.stage == _CLEAR_DRAINING_STAGE

    # The next resume re-acquires the fence and re-checks without redoing
    # physical cleanup. Still pending.
    again = await deletion_service.resume_hard_delete(first.job)
    assert again.object_cleanup_pending is True
    assert again.job.stage == _CLEAR_DRAINING_STAGE

    # Drive the cleanup service to success, remove the object, resume once.
    await _drive_cleanup_service_to_success(store, record)
    s3_state.objects.clear()
    final = await deletion_service.resume_hard_delete(first.job)
    assert final.errors == []
    assert final.job.status == "succeeded"
    assert final.job.stage == _CLEAR_FINALIZING_STAGE


@pytest.mark.asyncio
async def test_object_drain_blocked_manifest_fails_closed(tmp_path: Path):
    """A blocked manifest for the generation surfaces a safe error."""

    (
        kb_service,
        store,
        _registry,
        deletion_service,
        _probe,
        record,
        _input_ws,
        _working_ws,
        counting_storage,
        _s3_state,
    ) = await _build_object_env(tmp_path, kb_id="kb_drain_blocked")
    await _soft_delete(kb_service, record)
    first = await deletion_service.hard_delete(
        record.id, expected_generation=record.generation
    )
    assert first.object_cleanup_pending is True

    # Force the manifest into 'blocked' to simulate cleanup-service permanent
    # failure (e.g. ownership conflict that the cleanup service escalates).
    await _force_manifest_to_status(store, record, status="blocked")

    second = await deletion_service.resume_hard_delete(first.job)
    assert second.job.status == "failed"
    assert any("blocked" in err for err in second.errors)

    # Catalog and lifecycle fence must be preserved for retry.
    catalog = await kb_service.get(record.id, include_deleted=True)
    assert catalog.status == "deleted"
    lifecycle = await store.get_kb_lifecycle(record.id)
    assert lifecycle is not None and lifecycle.state == "deleting"


@pytest.mark.asyncio
async def test_object_drain_stale_generation_fence_fails_closed(tmp_path: Path):
    """A drain job whose pinned generation no longer matches lifecycle fails."""

    (
        kb_service,
        store,
        _registry,
        deletion_service,
        _probe,
        record,
        _input_ws,
        _working_ws,
        counting_storage,
        _s3_state,
    ) = await _build_object_env(tmp_path, kb_id="kb_drain_stale_gen")
    await _soft_delete(kb_service, record)
    first = await deletion_service.hard_delete(
        record.id, expected_generation=record.generation
    )
    assert first.object_cleanup_pending is True

    # Simulate the lifecycle moving on (e.g. an operator-driven generation
    # reset) while the drain job is checkpointed. The fence re-acquired by the
    # next resume must observe this and fail closed before any purge runs.
    await store.complete_kb_deletion(record.id, record.generation, first.job.id)
    await kb_service.purge(
        record.id,
        expected_generation=record.generation,
        expected_status="deleted",
    )
    # A new generation is created and activated, leaving the old generation's
    # drain job with a stale pinned payload.
    new_record = await kb_service.create(kb_id=record.id, name="Recreated")
    await store.activate_kb_generation(new_record.id, new_record.generation)

    resumed = await deletion_service.resume_hard_delete(first.job)

    assert resumed.job.status == "failed"
    assert any(
        "generation changed" in err or "catalog status is not deleted" in err
        for err in resumed.errors
    )


@pytest.mark.asyncio
async def test_object_drain_releases_existing_retained_manifests(tmp_path: Path):
    """Retained manifests for the KB generation are released during enqueue."""

    (
        kb_service,
        store,
        _registry,
        deletion_service,
        _probe,
        record,
        _input_ws,
        _working_ws,
        counting_storage,
        _s3_state,
    ) = await _build_object_env(tmp_path, kb_id="kb_drain_release")
    # Seed a retained manifest for the same KB generation (representing the
    # kind of artifact-cleanup reservation an earlier document mutation might
    # have left behind). The drain enqueue must release it.
    from lightrag.api.artifact_lifecycle import (
        ArtifactCleanupManifestRecord,
        artifact_cleanup_idempotency_key,
    )

    now = datetime.now(timezone.utc)
    retained_target_uri = (
        f"s3://{_BUCKET}/{_PREFIX}/workspaces/{record.workspace}/"
        f"documents/legacy-doc/source/legacy.bin"
    )
    retained_idempotency = artifact_cleanup_idempotency_key(
        reason="replace",
        kb_id=record.id,
        kb_generation=record.generation,
        workspace=record.workspace,
        document_id="legacy-doc",
        artifact_id=None,
        source_generation_id="srcg-legacy",
        target_kind="object",
        target_namespace="source",
        target_uri=retained_target_uri,
    )
    retained_manifest = ArtifactCleanupManifestRecord(
        id=f"retained-{uuid4().hex[:12]}",
        idempotency_key=retained_idempotency,
        manifest_group_id="dmg_legacy",
        kb_id=record.id,
        kb_generation=record.generation,
        workspace=record.workspace,
        document_id="legacy-doc",
        artifact_id=None,
        source_generation_id="srcg-legacy",
        origin_job_id="legacy-replace",
        origin_attempt_token="attempt-legacy",
        reason="replace",
        target_kind="object",
        target_namespace="source",
        disposition="retain",
        status="retained",
        target_uri=retained_target_uri,
        delete_after=now,
        cleanup_deadline_at=now,
        audit_retain_until=now,
        next_attempt_at=now,
        created_at=now,
        updated_at=now,
    )
    await store.enqueue_artifact_cleanup_manifest(retained_manifest)

    await _soft_delete(kb_service, record)
    first = await deletion_service.hard_delete(
        record.id, expected_generation=record.generation
    )
    assert first.object_cleanup_pending is True

    # The previously retained manifest must have been released (disposition
    # flipped to 'delete' and status to 'pending') so the cleanup service may
    # drain it under the now-deleting lifecycle.
    released_rows, total = await store.list_artifact_cleanup_manifests(
        kb_id=record.id,
        kb_generation=record.generation,
    )
    assert total == 2
    retained_now = next(row for row in released_rows if row.id == retained_manifest.id)
    assert retained_now.disposition == "delete"
    assert retained_now.status == "pending"


@pytest.mark.asyncio
async def test_object_drain_runs_engine_and_local_compat_only(tmp_path: Path):
    """force_evict + drop_kb_data + rmtree must run; delete_workspace must not."""

    (
        kb_service,
        _store,
        registry,
        deletion_service,
        probe,
        record,
        input_workspace,
        working_workspace,
        counting_storage,
        _s3_state,
    ) = await _build_object_env(tmp_path, kb_id="kb_drain_compat")
    cached = await registry.get(record.id)
    await _soft_delete(kb_service, record)

    first = await deletion_service.hard_delete(
        record.id, expected_generation=record.generation
    )
    assert first.object_cleanup_pending is True

    # force_evict finalizes the cached instance; drop_kb_data builds a fresh
    # FakeRAG and drops it (the registry never re-reads the catalog). Probe
    # observes every build/drop so we can assert the engine compat cleanup
    # ran without holding a stale reference.
    assert cached.finalized is True
    assert probe.drop_calls == 1
    assert any(instance.dropped for instance in probe.instances)
    assert counting_storage.delete_workspace_calls == []
    # Local artifacts are still torn down by the engine compat path.
    assert not input_workspace.exists()
    assert not working_workspace.exists()


@pytest.mark.asyncio
async def test_object_drain_recovers_when_cursor_method_temporarily_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A store that does not expose delete_artifact_recovery_cursor is harmless.

    The drain code reads the additive method via ``getattr(..., None)`` so a
    slightly older store that predates Writer A's new method simply skips the
    cursor removal. The hard delete still completes; the cursor becomes
    harmless residue that ``purge_kb_metadata`` does not touch.
    """

    (
        kb_service,
        store,
        _registry,
        deletion_service,
        _probe,
        record,
        _input_ws,
        _working_ws,
        counting_storage,
        s3_state,
    ) = await _build_object_env(tmp_path, kb_id="kb_drain_no_cursor")
    await _soft_delete(kb_service, record)
    first = await deletion_service.hard_delete(
        record.id, expected_generation=record.generation
    )
    await _drive_cleanup_service_to_success(store, record)
    s3_state.objects.clear()

    # Model the additive method's absence by removing it from the class so
    # ``getattr(instance, ...)`` falls back to the ``None`` default. This
    # mirrors what a rolling-deployed older store would look like to the
    # service code.
    if hasattr(type(store), "delete_artifact_recovery_cursor"):
        monkeypatch.delattr(type(store), "delete_artifact_recovery_cursor")
    assert getattr(store, "delete_artifact_recovery_cursor", None) is None

    resumed = await deletion_service.resume_hard_delete(first.job)
    assert resumed.errors == []
    assert resumed.job.status == "succeeded"


@pytest.mark.asyncio
async def test_object_drain_recovers_when_cursor_delete_raises(tmp_path: Path):
    """A failing cursor delete must not fail the hard-delete."""

    (
        kb_service,
        store,
        _registry,
        deletion_service,
        _probe,
        record,
        _input_ws,
        _working_ws,
        counting_storage,
        s3_state,
    ) = await _build_object_env(tmp_path, kb_id="kb_drain_cursor_fail")
    await _soft_delete(kb_service, record)
    first = await deletion_service.hard_delete(
        record.id, expected_generation=record.generation
    )
    await _drive_cleanup_service_to_success(store, record)
    s3_state.objects.clear()

    real_delete_cursor = getattr(store, "delete_artifact_recovery_cursor", None)
    if real_delete_cursor is not None:

        async def failing_cursor(_kb_id, _kb_generation):
            raise RuntimeError("cursor store temporarily unavailable")

        store.delete_artifact_recovery_cursor = failing_cursor  # type: ignore[method-assign]

    resumed = await deletion_service.resume_hard_delete(first.job)
    assert resumed.errors == []
    assert resumed.job.status == "succeeded"


# ---------------------------------------------------------------------------
# Local-mode regression: delete_workspace path is unchanged by Phase 3.1-D.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_mode_unchanged_runs_delete_workspace(tmp_path: Path):
    """In local mode, _run_physical_cleanup still calls delete_workspace."""

    from tests.api.test_kb_hard_delete import (
        FakeObjectStorage,
        _build_environment,
        _soft_delete as _local_soft_delete,
    )

    object_storage = FakeObjectStorage(deleted_count=7)
    (
        kb_service,
        _store,
        _registry,
        deletion_service,
        _probe,
        record,
        _input_ws,
        _working_ws,
    ) = await _build_environment(
        tmp_path, kb_id="kb_local_regression", object_storage=object_storage
    )
    assert deletion_service._artifact_storage_mode == "local"
    assert deletion_service._object_authoritative() is False

    await _local_soft_delete(kb_service, record)
    result = await deletion_service.hard_delete(
        record.id, expected_generation=record.generation
    )

    assert result.errors == []
    assert result.job.status == "succeeded"
    assert result.cleared_object_storage is True
    assert result.deleted_objects == 7
    assert object_storage.deleted_workspaces == [record.workspace]
    assert result.job.stage == _CLEAR_FINALIZING_STAGE
    # No draining checkpoint in local mode.
    assert result.object_cleanup_pending is False


# ---------------------------------------------------------------------------
# B-2 regression: ``assert_hard_delete_supported`` consults the capability
# constant instead of firing purely on object mode.
# ---------------------------------------------------------------------------


def test_hard_delete_gate_opens_when_capability_enabled(tmp_path: Path):
    """B-2: object mode + capability True -> gate does NOT raise, drain admitted.

    The autouse ``_enable_hard_delete_capability`` fixture flips the
    ``_hard_delete_capability_enabled`` indirection to True. With the gate now
    coupled to that constant (B-2 fix), object-mode hard delete is admitted and
    the manifest-driven drain path is reachable instead of returning HTTP 503.
    The full drain behaviour is exercised by the tests above; this is the
    focused gate-opening regression.
    """

    deletion_service = KBDeletionService(
        object(),  # type: ignore[arg-type]  # kb_service unused by the gate
        object(),  # type: ignore[arg-type]  # metadata_store unused by the gate
        object(),  # type: ignore[arg-type]  # registry unused by the gate
        input_root=tmp_path / "inputs",
        artifact_storage_mode="object",
    )
    assert deletion_service._artifact_storage_mode == "object"
    # Autouse fixture opened the gate via the capability indirection.
    assert kb_deletion_service._hard_delete_capability_enabled() is True
    # The gate must NOT raise: object-mode hard delete is admitted.
    deletion_service.assert_hard_delete_supported()


def test_hard_delete_gate_still_raises_when_capability_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """B-2: object mode + capability False -> gate raises (behavior preserved).

    Overrides the autouse fixture to restore the real production state (the
    ``OBJECT_AUTHORITATIVE_LIFECYCLE_IMPLEMENTED`` constant is ``False``),
    preserving the existing HTTP 503 admission contract until Gate 3 flips the
    constant.
    """

    monkeypatch.setattr(
        kb_deletion_service,
        "_hard_delete_capability_enabled",
        lambda: False,
    )

    deletion_service = KBDeletionService(
        object(),  # type: ignore[arg-type]  # kb_service unused by the gate
        object(),  # type: ignore[arg-type]  # metadata_store unused by the gate
        object(),  # type: ignore[arg-type]  # registry unused by the gate
        input_root=tmp_path / "inputs",
        artifact_storage_mode="object",
    )
    assert deletion_service._artifact_storage_mode == "object"
    assert kb_deletion_service._hard_delete_capability_enabled() is False
    with pytest.raises(KBHardDeleteUnsupportedError):
        deletion_service.assert_hard_delete_supported()


# ---------------------------------------------------------------------------
# Helpers for simulating cleanup-service progress between drain polls.
# ---------------------------------------------------------------------------


async def _drive_cleanup_service_to_success(
    store: SQLiteMetadataStore,
    record: KnowledgeBaseRecord,
) -> None:
    """Mark all pending kb_delete manifests for this KB as succeeded.

    Uses the same low-level claim/succeed path the cleanup service uses, so the
    drain code observes a realistic manifest state transition.
    """

    pending, total = await store.list_artifact_cleanup_manifests(
        kb_id=record.id,
        kb_generation=record.generation,
        statuses=["pending"],
        limit=100,
    )
    assert total == len(pending)
    if not pending:
        return
    lease_owner = "drain-test-worker"
    claimed = await store.claim_due_artifact_cleanup_manifests(
        lease_owner=lease_owner,
        lease_duration_seconds=60.0,
        limit=100,
        now=datetime.now(timezone.utc),
    )
    claimed_ids = {manifest.id for manifest in claimed}
    for manifest in pending:
        if manifest.id not in claimed_ids:
            # The cleanup service may have already leased it on a prior tick;
            # re-fetch and use the live lease credentials.
            current = await store.get_artifact_cleanup_manifest(manifest.id)
            if current.status != "leased":
                continue
            await store.succeed_artifact_cleanup_manifest(
                manifest.id,
                lease_owner=current.lease_owner or lease_owner,
                lease_token=current.lease_token or "drain-test-token",
            )
            continue
        live = next(m for m in claimed if m.id == manifest.id)
        await store.succeed_artifact_cleanup_manifest(
            manifest.id,
            lease_owner=live.lease_owner or lease_owner,
            lease_token=live.lease_token or "drain-test-token",
        )


async def _force_manifest_to_status(
    store: SQLiteMetadataStore,
    record: KnowledgeBaseRecord,
    *,
    status: str,
) -> None:
    """Forcibly transition a kb_delete manifest to a terminal state.

    Used to simulate cleanup-service permanent failure (blocked) without
    needing to drive the full claim/block dance. Goes through the public
    claim API so lease ownership is consistent with the store's invariants.
    """

    claimed = await store.claim_due_artifact_cleanup_manifests(
        lease_owner="drain-test-blocker",
        lease_duration_seconds=60.0,
        limit=100,
        now=datetime.now(timezone.utc),
    )
    for manifest in claimed:
        if status == "blocked":
            await store.block_artifact_cleanup_manifest(
                manifest.id,
                lease_owner=manifest.lease_owner or "drain-test-blocker",
                lease_token=manifest.lease_token or "drain-test-token",
                error_code="object_ownership_conflict",
            )
        elif status == "succeeded":
            await store.succeed_artifact_cleanup_manifest(
                manifest.id,
                lease_owner=manifest.lease_owner or "drain-test-blocker",
                lease_token=manifest.lease_token or "drain-test-token",
            )
