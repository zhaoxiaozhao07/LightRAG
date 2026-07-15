from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Literal, cast

import pytest

from lightrag.api import kb_deletion_service
from lightrag.api.job_service import JobService
from lightrag.api.kb_deletion_service import (
    KBDeletionService,
    KBHardDeleteInProgressError,
)
from lightrag.api.kb_service import (
    KnowledgeBaseConflictError,
    KnowledgeBaseNotFoundError,
    KnowledgeBaseRecord,
    KnowledgeBaseService,
    KnowledgeBaseStatus,
    utc_now_iso,
)
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry
from lightrag.api.metadata_store import (
    DocumentRecord,
    JobRecord,
    SQLiteMetadataStore,
)
from lightrag.api.object_storage import ObjectStorage

pytestmark = pytest.mark.offline


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
        if self.probe.drop_failures > 0:
            self.probe.drop_failures -= 1
            return {
                "dropped": 0,
                "failed": 1,
                "errors": ["entities_vdb: connection refused"],
            }
        return {"dropped": 12, "failed": 0, "errors": []}


class BuilderProbe:
    def __init__(self, *, drop_failures: int = 0):
        self.instances: list[FakeRAG] = []
        self.records: list[KnowledgeBaseRecord] = []
        self.finalized: list[FakeRAG] = []
        self.drop_calls = 0
        self.drop_failures = drop_failures

    async def build(self, record: KnowledgeBaseRecord) -> FakeRAG:
        self.records.append(record)
        rag = FakeRAG(record, self)
        self.instances.append(rag)
        return rag

    async def finalize(self, rag: Any) -> None:
        assert isinstance(rag, FakeRAG)
        await rag.finalize_storages()
        self.finalized.append(rag)


class FakeObjectStorage(ObjectStorage):
    def __init__(self, deleted_count: int = 3, *, failures: int = 0):
        self.deleted_count = deleted_count
        self.failures = failures
        self.deleted_workspaces: list[str] = []

    async def delete_workspace(self, workspace: str) -> int:
        self.deleted_workspaces.append(workspace)
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("object cleanup denied")
        return self.deleted_count


def _doc(kb_id: str, doc_id: str, *, workspace: str) -> DocumentRecord:
    now = utc_now_iso()
    return DocumentRecord(
        id=doc_id,
        kb_id=kb_id,
        workspace=workspace,
        lightrag_doc_id="doc-123",
        source_type="upload",
        source_name=f"{doc_id}.pdf",
        source_uri="/tmp/x.pdf",
        source_hash="sha256:x",
        content_type="application/pdf",
        size_bytes=1,
        parser_hash="sha256:p",
        index_hash=None,
        status="parsed",
        enabled=True,
        archived=False,
        chunks_count=None,
        entity_count=None,
        relation_count=None,
        error_code=None,
        error_message=None,
        metadata={"source_key": f"manual/{doc_id}.pdf"},
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )


def _job(
    kb_id: str,
    workspace: str,
    *,
    job_id: str,
    job_type: str = "upload",
    status: str = "succeeded",
    idempotency_key: str | None = None,
    payload: dict[str, Any] | None = None,
) -> JobRecord:
    now = utc_now_iso()
    terminal = status in {"succeeded", "failed"}
    return JobRecord(
        id=job_id,
        kb_id=kb_id,
        workspace=workspace,
        batch_id=None,
        document_id=None,
        job_type=job_type,
        status=status,
        stage="deleting" if job_type == "clear_kb" else None,
        progress=1.0 if terminal else 0.1,
        total_items=1,
        completed_items=1 if status == "succeeded" else 0,
        failed_items=1 if status == "failed" else 0,
        idempotency_key=idempotency_key,
        config_version_id=None,
        config_hash=None,
        retry_count=0,
        max_retries=3,
        payload=dict(payload or {}),
        result={"documents_created": 1} if job_type != "clear_kb" else None,
        error_code=None,
        error_message=None,
        created_at=now,
        updated_at=now,
        queued_at=now,
        started_at=now if status == "running" else None,
        finished_at=now if terminal else None,
        cancelled_at=now if status == "cancelled" else None,
    )


async def _build_environment(
    tmp_path: Path,
    *,
    kb_id: str,
    probe: BuilderProbe | None = None,
    object_storage: FakeObjectStorage | None = None,
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
]:
    kb_service = KnowledgeBaseService(tmp_path / "kb.json")
    await kb_service.initialize()
    record = await kb_service.create(kb_id=kb_id, name=kb_id)

    store = SQLiteMetadataStore(tmp_path / "metadata.sqlite3")
    await store.initialize()
    document = _doc(record.id, f"doc_{kb_id}", workspace=record.workspace)
    seed = _job(record.id, record.workspace, job_id=f"seed_{kb_id}")
    await store.create_documents_and_job([document], seed)

    input_root = tmp_path / "inputs"
    input_workspace = input_root / record.workspace
    working_root = tmp_path / "working"
    working_workspace = working_root / record.workspace
    if create_files:
        (input_workspace / document.id).mkdir(parents=True)
        (input_workspace / document.id / "source.pdf").write_bytes(b"raw")
        working_workspace.mkdir(parents=True)
        (working_workspace / "graph.json").write_text("{}", encoding="utf-8")

    active_probe = probe or BuilderProbe()
    registry = LightRAGInstanceRegistry(
        kb_service, active_probe.build, active_probe.finalize
    )
    deletion_service = KBDeletionService(
        kb_service,
        store,
        registry,
        input_root=input_root,
        working_dir=working_root,
        object_storage=object_storage,
    )
    return (
        kb_service,
        store,
        registry,
        deletion_service,
        active_probe,
        record,
        input_workspace,
        working_workspace,
    )


async def _soft_delete(
    kb_service: KnowledgeBaseService, record: KnowledgeBaseRecord
) -> KnowledgeBaseRecord:
    return await kb_service.delete(
        record.id,
        expected_generation=record.generation,
    )


@pytest.mark.asyncio
async def test_hard_delete_uses_generation_payload_and_strict_purge_retains_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    object_storage = FakeObjectStorage(deleted_count=5)
    (
        kb_service,
        store,
        registry,
        deletion_service,
        probe,
        record,
        input_workspace,
        working_workspace,
    ) = await _build_environment(
        tmp_path,
        kb_id="kb_success",
        object_storage=object_storage,
    )
    cached = cast(FakeRAG, await registry.get(record.id))
    await _soft_delete(kb_service, record)

    commit_order: list[str] = []
    purge_calls: list[tuple[str, str | None, str | None]] = []
    original_cleanup = deletion_service._run_physical_cleanup
    original_metadata_purge = store.purge_kb_metadata
    original_complete = store.complete_kb_deletion
    original_catalog_purge = kb_service.purge
    original_transition = store.transition_job

    async def observed_cleanup(
        cleanup_record: KnowledgeBaseRecord,
        cleanup_result: kb_deletion_service.KBHardDeleteResult,
    ) -> None:
        commit_order.append("physical_cleanup")
        await original_cleanup(cleanup_record, cleanup_result)

    async def strict_purge(
        kb_id: str,
        generation: str | None = None,
        *,
        delete_job_id: str | None = None,
    ) -> dict[str, int]:
        commit_order.append("metadata_purge")
        purge_calls.append((kb_id, generation, delete_job_id))
        return await original_metadata_purge(
            kb_id,
            generation=generation,
            delete_job_id=delete_job_id,
        )

    async def observed_complete(
        kb_id: str, generation: str, delete_job_id: str
    ):
        lifecycle = await store.get_kb_lifecycle(kb_id)
        assert lifecycle is not None and lifecycle.state == "deleting"
        assert (await kb_service.get(kb_id, include_deleted=True)).status == "deleted"
        commit_order.append("complete_lifecycle")
        completed = await original_complete(kb_id, generation, delete_job_id)
        assert completed.state == "deleted"
        assert (await kb_service.get(kb_id, include_deleted=True)).status == "deleted"
        return completed

    async def observed_catalog_purge(
        kb_id: str,
        *,
        expected_generation: str | None = None,
        expected_status: KnowledgeBaseStatus | None = "deleted",
    ) -> bool:
        lifecycle = await store.get_kb_lifecycle(kb_id)
        assert lifecycle is not None and lifecycle.state == "deleted"
        commit_order.append("catalog_purge")
        return await original_catalog_purge(
            kb_id,
            expected_generation=expected_generation,
            expected_status=expected_status,
        )

    async def observed_transition(kb_id: str, job_id: str, **kwargs: Any):
        if kwargs.get("status") == "succeeded":
            commit_order.append("job_succeeded")
        return await original_transition(kb_id, job_id, **kwargs)

    monkeypatch.setattr(deletion_service, "_run_physical_cleanup", observed_cleanup)
    monkeypatch.setattr(store, "purge_kb_metadata", strict_purge)
    monkeypatch.setattr(store, "complete_kb_deletion", observed_complete)
    monkeypatch.setattr(kb_service, "purge", observed_catalog_purge)
    monkeypatch.setattr(store, "transition_job", observed_transition)
    result = await deletion_service.hard_delete(
        record.id,
        expected_generation=record.generation,
    )

    assert result.errors == []
    assert result.job.status == "succeeded"
    assert result.job.idempotency_key == (
        f"clear_kb:{record.id}:{record.generation}"
    )
    assert set(result.job.payload) == {
        "kb_generation",
        "workspace",
        "idempotency_fingerprint",
    }
    assert result.job.payload["kb_generation"] == record.generation
    assert result.job.payload["workspace"] == record.workspace
    assert purge_calls == [(record.id, record.generation, result.job.id)]
    assert commit_order == [
        "physical_cleanup",
        "metadata_purge",
        "complete_lifecycle",
        "catalog_purge",
        "job_succeeded",
    ]
    assert result.purged_rows["document_source_keys"] == 1
    assert result.purged_rows["documents"] == 1
    assert result.purged_rows["jobs"] == 1  # seed only; clear job is retained
    assert result.cleared_input_dir is True
    assert result.cleared_object_storage is True
    assert result.deleted_objects == 5
    assert result.dropped_storages == 12
    assert result.finalized_storages is True
    assert result.purged_catalog is True
    assert cached.finalized is True
    assert not input_workspace.exists()
    assert not working_workspace.exists()
    assert object_storage.deleted_workspaces == [record.workspace]
    assert probe.drop_calls == 1

    with pytest.raises(KnowledgeBaseNotFoundError):
        await kb_service.get(record.id, include_deleted=True)
    lifecycle = await store.get_kb_lifecycle(record.id)
    assert lifecycle is not None
    assert lifecycle.state == "deleted"
    assert lifecycle.generation == record.generation
    assert lifecycle.delete_job_id == result.job.id
    jobs, total = await store.list_jobs(record.id)
    assert total == 1
    assert jobs[0].id == result.job.id
    assert jobs[0].status == "succeeded"


@pytest.mark.asyncio
async def test_same_generation_concurrent_enqueue_reuses_one_job_row(tmp_path: Path):
    (
        kb_service,
        store,
        _registry,
        deletion_service,
        _probe,
        record,
        _input_workspace,
        _working_workspace,
    ) = await _build_environment(tmp_path, kb_id="kb_concurrent_enqueue")
    await _soft_delete(kb_service, record)

    jobs = await asyncio.gather(
        *(
            deletion_service.enqueue_hard_delete(
                record.id,
                expected_generation=record.generation,
            )
            for _ in range(20)
        )
    )

    assert len({job.id for job in jobs}) == 1
    assert {job.status for job in jobs} == {"queued"}
    persisted, total = await store.list_jobs(record.id)
    clear_jobs = [job for job in persisted if job.job_type == "clear_kb"]
    assert total == 2  # seed + exactly one clear job
    assert len(clear_jobs) == 1
    assert clear_jobs[0].idempotency_key == (
        f"clear_kb:{record.id}:{record.generation}"
    )


@pytest.mark.parametrize("terminal_status", ["failed", "cancelled"])
@pytest.mark.asyncio
async def test_concurrent_enqueue_resets_terminal_clear_job_once_in_place(
    tmp_path: Path,
    terminal_status: Literal["failed", "cancelled"],
):
    (
        kb_service,
        store,
        _registry,
        deletion_service,
        _probe,
        record,
        _input_workspace,
        _working_workspace,
    ) = await _build_environment(
        tmp_path,
        kb_id=f"kb_reset_{terminal_status}",
    )
    await _soft_delete(kb_service, record)
    queued = await deletion_service.enqueue_hard_delete(
        record.id,
        expected_generation=record.generation,
    )
    terminal = await store.transition_job(
        record.id,
        queued.id,
        status=terminal_status,
        error_code="test_terminal",
        error_message="retry me",
    )
    original_key = terminal.idempotency_key

    enqueued = await asyncio.gather(
        *(
            deletion_service.enqueue_hard_delete(
                record.id,
                expected_generation=record.generation,
            )
            for _ in range(20)
        )
    )

    assert {job.id for job in enqueued} == {queued.id}
    assert {job.status for job in enqueued} == {"queued"}
    persisted = await store.get_job(record.id, queued.id)
    assert persisted.status == "queued"
    assert persisted.retry_count == 1
    assert persisted.idempotency_key == original_key
    jobs, total = await store.list_jobs(record.id)
    assert total == 2
    assert len([job for job in jobs if job.job_type == "clear_kb"]) == 1


@pytest.mark.asyncio
async def test_enqueue_returns_same_failed_job_and_sync_retry_resets_same_id(
    tmp_path: Path
):
    probe = BuilderProbe(drop_failures=1)
    (
        kb_service,
        _store,
        _registry,
        deletion_service,
        _probe,
        record,
        _input_workspace,
        _working_workspace,
    ) = await _build_environment(
        tmp_path,
        kb_id="kb_retry_same_job",
        probe=probe,
    )
    await _soft_delete(kb_service, record)

    first = await deletion_service.hard_delete(
        record.id,
        expected_generation=record.generation,
    )
    assert first.job.status == "failed"
    failed_key = first.job.idempotency_key

    enqueued = await deletion_service.enqueue_hard_delete(
        record.id,
        expected_generation=record.generation,
    )
    assert enqueued.id == first.job.id
    assert enqueued.status == "queued"
    assert enqueued.idempotency_key == failed_key
    assert enqueued.retry_count == 1

    retried = await deletion_service.hard_delete(
        record.id,
        expected_generation=record.generation,
    )
    assert retried.errors == []
    assert retried.job.status == "succeeded"
    assert retried.job.id == first.job.id
    assert retried.job.idempotency_key == failed_key
    assert retried.job.retry_count == 1


@pytest.mark.asyncio
async def test_sync_hard_delete_refuses_running_job_with_live_owner(tmp_path: Path):
    (
        kb_service,
        store,
        _registry,
        deletion_service,
        _probe,
        record,
        _input_workspace,
        _working_workspace,
    ) = await _build_environment(tmp_path, kb_id="kb_running")
    await _soft_delete(kb_service, record)
    queued = await deletion_service.enqueue_hard_delete(
        record.id,
        expected_generation=record.generation,
    )
    running = await store.transition_job(
        record.id,
        queued.id,
        status="running",
    )
    owner_store = SQLiteMetadataStore(store.db_path)
    await owner_store.initialize()

    try:
        async with owner_store.job_execution_guard(running.id) as acquired:
            assert acquired is True
            with pytest.raises(KBHardDeleteInProgressError) as exc_info:
                await deletion_service.hard_delete(
                    record.id,
                    expected_generation=record.generation,
                )
            assert exc_info.value.job.id == running.id
    finally:
        await owner_store.close()


@pytest.mark.asyncio
async def test_sync_hard_delete_resumes_running_job_after_cancelled_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (
        kb_service,
        store,
        _registry,
        deletion_service,
        _probe,
        record,
        _input_workspace,
        _working_workspace,
    ) = await _build_environment(tmp_path, kb_id="kb_cancelled_sync_owner")
    await _soft_delete(kb_service, record)
    entered = asyncio.Event()
    never_release = asyncio.Event()
    original_cleanup = deletion_service._run_physical_cleanup

    async def blocked_cleanup(cleanup_record, result):
        entered.set()
        await never_release.wait()
        return await original_cleanup(cleanup_record, result)

    monkeypatch.setattr(deletion_service, "_run_physical_cleanup", blocked_cleanup)
    first_task = asyncio.create_task(
        deletion_service.hard_delete(
            record.id,
            expected_generation=record.generation,
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=2)
    running_jobs, _total = await store.list_jobs(
        record.id,
        statuses=["running"],
    )
    running = next(job for job in running_jobs if job.job_type == "clear_kb")
    original_key = running.idempotency_key
    lifecycle = await store.get_kb_lifecycle(record.id)
    assert lifecycle is not None and lifecycle.state == "deleting"
    assert lifecycle.delete_job_id == running.id

    first_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_task
    async with store.job_execution_guard(running.id, wait=False) as acquired:
        assert acquired is True
    assert (await store.get_job(record.id, running.id)).status == "running"

    monkeypatch.setattr(deletion_service, "_run_physical_cleanup", original_cleanup)
    resumed = await deletion_service.hard_delete(
        record.id,
        expected_generation=record.generation,
    )

    assert resumed.errors == []
    assert resumed.job.status == "succeeded"
    assert resumed.job.id == running.id
    assert resumed.job.idempotency_key == original_key


@pytest.mark.asyncio
async def test_sync_hard_delete_owner_blocks_cross_store_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (
        kb_service,
        store,
        _registry,
        deletion_service,
        _probe,
        record,
        _input_workspace,
        _working_workspace,
    ) = await _build_environment(tmp_path, kb_id="kb_sync_owner")
    peer = SQLiteMetadataStore(store.db_path)
    await peer.initialize()
    await _soft_delete(kb_service, record)
    entered = asyncio.Event()
    release = asyncio.Event()
    original_cleanup = deletion_service._run_physical_cleanup

    async def blocked_cleanup(cleanup_record, result):
        entered.set()
        await release.wait()
        await original_cleanup(cleanup_record, result)

    monkeypatch.setattr(deletion_service, "_run_physical_cleanup", blocked_cleanup)
    task = asyncio.create_task(
        deletion_service.hard_delete(
            record.id,
            expected_generation=record.generation,
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        jobs, _total = await peer.list_jobs(record.id, statuses=["running"])
        clear_job = next(job for job in jobs if job.job_type == "clear_kb")

        assert await peer.recover_orphan_jobs(
            resumable_job_types={"clear_kb"}, grace_seconds=0
        ) == []
        assert (await peer.get_job(record.id, clear_job.id)).status == "running"

        release.set()
        result = await asyncio.wait_for(task, timeout=5)
        assert result.job.status == "succeeded"
    finally:
        release.set()
        if not task.done():
            await asyncio.wait_for(task, timeout=5)
        await peer.close()


@pytest.mark.parametrize(
    "failure_point",
    ["force_evict", "drop", "working_dir", "input_dir", "object_storage"],
)
@pytest.mark.asyncio
async def test_physical_cleanup_failure_preserves_metadata_catalog_and_fence_then_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
):
    probe = BuilderProbe(drop_failures=1 if failure_point == "drop" else 0)
    object_storage = FakeObjectStorage(
        failures=1 if failure_point == "object_storage" else 0
    )
    (
        kb_service,
        store,
        registry,
        deletion_service,
        _probe,
        record,
        input_workspace,
        working_workspace,
    ) = await _build_environment(
        tmp_path,
        kb_id=f"kb_fail_{failure_point}",
        probe=probe,
        object_storage=object_storage,
    )
    await registry.get(record.id)
    await _soft_delete(kb_service, record)

    if failure_point == "force_evict":
        original_force_evict = registry.force_evict
        failures = 1

        async def fail_force_evict_once(kb_id: str) -> bool:
            nonlocal failures
            if failures:
                failures -= 1
                raise RuntimeError("force evict denied")
            return await original_force_evict(kb_id)

        monkeypatch.setattr(registry, "force_evict", fail_force_evict_once)
    elif failure_point in {"working_dir", "input_dir"}:
        original_rmtree = kb_deletion_service.shutil.rmtree
        failed_path = (
            working_workspace if failure_point == "working_dir" else input_workspace
        ).resolve()
        failures = 1

        def fail_rmtree_once(path: str | Path, *args: Any, **kwargs: Any) -> None:
            nonlocal failures
            if Path(path).resolve() == failed_path and failures:
                failures -= 1
                raise OSError("rmtree denied")
            original_rmtree(path, *args, **kwargs)

        monkeypatch.setattr(kb_deletion_service.shutil, "rmtree", fail_rmtree_once)

    first = await deletion_service.hard_delete(
        record.id,
        expected_generation=record.generation,
    )

    assert first.job.status == "failed"
    assert first.errors
    assert first.purged_rows == {}
    assert first.purged_catalog is False
    catalog = await kb_service.get(record.id, include_deleted=True)
    assert catalog.generation == record.generation
    assert catalog.status == "deleted"
    documents, document_total = await store.list_documents(record.id)
    assert document_total == 1
    assert documents[0].id.startswith("doc_")
    jobs, job_total = await store.list_jobs(record.id)
    assert job_total == 2
    assert {job.id for job in jobs} >= {first.job.id}
    lifecycle = await store.get_kb_lifecycle(record.id)
    assert lifecycle is not None
    assert lifecycle.state == "deleting"
    assert lifecycle.generation == record.generation
    assert lifecycle.delete_job_id == first.job.id
    with pytest.raises(KnowledgeBaseConflictError):
        await kb_service.create(kb_id=record.id, name="Must Stay Reserved")

    second = await deletion_service.hard_delete(
        record.id,
        expected_generation=record.generation,
    )

    assert second.errors == []
    assert second.job.status == "succeeded"
    assert second.job.id == first.job.id
    assert second.job.retry_count == 1
    with pytest.raises(KnowledgeBaseNotFoundError):
        await kb_service.get(record.id, include_deleted=True)
    documents, document_total = await store.list_documents(record.id)
    assert documents == []
    assert document_total == 0
    lifecycle = await store.get_kb_lifecycle(record.id)
    assert lifecycle is not None and lifecycle.state == "deleted"


@pytest.mark.asyncio
async def test_catalog_purge_failure_keeps_fence_and_retry_uses_same_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    object_storage = FakeObjectStorage()
    (
        kb_service,
        store,
        _registry,
        deletion_service,
        probe,
        record,
        _input_workspace,
        _working_workspace,
    ) = await _build_environment(
        tmp_path,
        kb_id="kb_catalog_retry",
        object_storage=object_storage,
    )
    await _soft_delete(kb_service, record)
    original_catalog_purge = kb_service.purge
    original_metadata_purge = store.purge_kb_metadata
    catalog_attempts = 0
    metadata_attempts = 0

    async def count_metadata_purge(
        kb_id: str,
        generation: str | None = None,
        *,
        delete_job_id: str | None = None,
    ) -> dict[str, int]:
        nonlocal metadata_attempts
        metadata_attempts += 1
        return await original_metadata_purge(
            kb_id,
            generation=generation,
            delete_job_id=delete_job_id,
        )

    async def fail_once(
        kb_id: str,
        *,
        expected_generation: str | None = None,
        expected_status: KnowledgeBaseStatus | None = "deleted",
    ) -> bool:
        nonlocal catalog_attempts
        catalog_attempts += 1
        if catalog_attempts == 1:
            raise RuntimeError("catalog temporarily unavailable")
        return await original_catalog_purge(
            kb_id,
            expected_generation=expected_generation,
            expected_status=expected_status,
        )

    monkeypatch.setattr(store, "purge_kb_metadata", count_metadata_purge)
    monkeypatch.setattr(kb_service, "purge", fail_once)
    first = await deletion_service.hard_delete(
        record.id,
        expected_generation=record.generation,
    )

    assert first.job.status == "failed"
    assert any("kb_catalog_purge" in error for error in first.errors)
    assert (await kb_service.get(record.id, include_deleted=True)).status == "deleted"
    lifecycle = await store.get_kb_lifecycle(record.id)
    assert lifecycle is not None and lifecycle.state == "deleted"
    jobs, total = await store.list_jobs(record.id)
    assert total == 1
    assert jobs[0].id == first.job.id  # strict metadata purge retained it
    assert metadata_attempts == 1
    physical_snapshot = (
        probe.drop_calls,
        len(probe.instances),
        len(object_storage.deleted_workspaces),
    )

    retried = await JobService(kb_service, store).retry_job(
        record.id,
        first.job.id,
        include_deleted=True,
    )
    assert retried.id == first.job.id
    assert retried.status == "queued"
    assert retried.idempotency_key == first.job.idempotency_key
    second = await deletion_service.resume_hard_delete(retried)
    assert second.errors == []
    assert second.job.id == first.job.id
    assert second.job.status == "succeeded"
    assert catalog_attempts == 2
    assert metadata_attempts == 1
    assert (
        probe.drop_calls,
        len(probe.instances),
        len(object_storage.deleted_workspaces),
    ) == physical_snapshot
    with pytest.raises(KnowledgeBaseNotFoundError):
        await kb_service.get(record.id, include_deleted=True)


@pytest.mark.asyncio
async def test_resume_after_catalog_purge_crash_only_completes_running_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    object_storage = FakeObjectStorage()
    (
        kb_service,
        store,
        _registry,
        deletion_service,
        probe,
        record,
        input_workspace,
        working_workspace,
    ) = await _build_environment(
        tmp_path,
        kb_id="kb_catalog_crash",
        object_storage=object_storage,
    )
    await _soft_delete(kb_service, record)
    queued = await deletion_service.enqueue_hard_delete(
        record.id,
        expected_generation=record.generation,
    )
    running = await store.transition_job(record.id, queued.id, status="running")

    async with store.kb_deletion_guard(
        record.id,
        record.generation,
        running.id,
    ):
        await store.purge_kb_metadata(
            record.id,
            generation=record.generation,
            delete_job_id=running.id,
        )
        await store.complete_kb_deletion(
            record.id,
            record.generation,
            running.id,
        )
        await kb_service.purge(
            record.id,
            expected_generation=record.generation,
            expected_status="deleted",
        )

    async def forbidden(*_args: Any, **_kwargs: Any):
        raise AssertionError("deleted tail must not repeat destructive work")

    monkeypatch.setattr(deletion_service, "_run_physical_cleanup", forbidden)
    monkeypatch.setattr(store, "purge_kb_metadata", forbidden)
    monkeypatch.setattr(store, "complete_kb_deletion", forbidden)
    monkeypatch.setattr(kb_service, "purge", forbidden)

    recovered = await store.recover_orphan_jobs(
        resumable_job_types={"clear_kb"},
        grace_seconds=0,
    )
    assert [job.id for job in recovered] == [running.id]
    queued_after_recovery = await store.get_job(record.id, running.id)
    assert queued_after_recovery.status == "queued"
    assert queued_after_recovery.idempotency_key == running.idempotency_key
    assert queued_after_recovery.payload == running.payload

    from lightrag.api.job_worker import JobWorker, build_clear_kb_executor

    worker = JobWorker(
        JobService(kb_service, store),
        executors={
            "clear_kb": build_clear_kb_executor(
                deletion_service=deletion_service,
            )
        },
        claim_grace_seconds=0,
    )
    claimed = await worker.poll_once()
    assert claimed is not None and claimed.id == running.id
    resumed_job = await store.get_job(record.id, running.id)

    assert resumed_job.status == "succeeded"
    assert resumed_job.id == running.id
    assert resumed_job.result is not None
    assert resumed_job.result["purged_catalog"] is True
    assert probe.drop_calls == 0
    assert object_storage.deleted_workspaces == []
    assert input_workspace.exists()
    assert working_workspace.exists()
    lifecycle = await store.get_kb_lifecycle(record.id)
    assert lifecycle is not None
    assert lifecycle.state == "deleted"
    assert lifecycle.delete_job_id == running.id


@pytest.mark.asyncio
async def test_complete_failure_retries_tail_without_retouching_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    object_storage = FakeObjectStorage()
    (
        kb_service,
        store,
        _registry,
        deletion_service,
        probe,
        record,
        _input_workspace,
        _working_workspace,
    ) = await _build_environment(
        tmp_path,
        kb_id="kb_complete_retry",
        object_storage=object_storage,
    )
    await _soft_delete(kb_service, record)
    original_complete = store.complete_kb_deletion
    attempts = 0

    async def fail_once(kb_id: str, generation: str, delete_job_id: str):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("completion write unavailable")
        return await original_complete(kb_id, generation, delete_job_id)

    monkeypatch.setattr(store, "complete_kb_deletion", fail_once)
    first = await deletion_service.hard_delete(
        record.id,
        expected_generation=record.generation,
    )

    assert first.job.status == "failed"
    assert any("complete_kb_deletion" in error for error in first.errors)
    catalog = await kb_service.get(record.id, include_deleted=True)
    assert catalog.generation == record.generation
    assert catalog.status == "deleted"
    lifecycle = await store.get_kb_lifecycle(record.id)
    assert lifecycle is not None and lifecycle.state == "deleting"
    assert first.job.stage == "finalizing"
    documents, document_total = await store.list_documents(record.id)
    assert documents == []
    assert document_total == 0
    physical_snapshot = (
        probe.drop_calls,
        len(probe.instances),
        len(object_storage.deleted_workspaces),
    )

    with pytest.raises(KnowledgeBaseConflictError):
        await kb_service.create(kb_id=record.id, name="Must not replace")

    second = await deletion_service.hard_delete(
        record.id,
        expected_generation=record.generation,
    )
    assert second.errors == []
    assert second.job.id == first.job.id
    assert second.job.status == "succeeded"
    assert attempts == 2
    assert (
        probe.drop_calls,
        len(probe.instances),
        len(object_storage.deleted_workspaces),
    ) == physical_snapshot
    lifecycle = await store.get_kb_lifecycle(record.id)
    assert lifecycle is not None and lifecycle.state == "deleted"
    with pytest.raises(KnowledgeBaseNotFoundError):
        await kb_service.get(record.id, include_deleted=True)


@pytest.mark.parametrize("missing_field", ["kb_generation", "workspace"])
@pytest.mark.asyncio
async def test_legacy_clear_job_missing_pinned_identity_fails_closed(
    tmp_path: Path,
    missing_field: str,
):
    object_storage = FakeObjectStorage()
    (
        kb_service,
        store,
        registry,
        deletion_service,
        probe,
        record,
        input_workspace,
        working_workspace,
    ) = await _build_environment(
        tmp_path,
        kb_id=f"kb_legacy_{missing_field}",
        object_storage=object_storage,
    )
    await registry.get(record.id)
    await _soft_delete(kb_service, record)
    await store.activate_kb_generation(record.id, record.generation)
    payload: dict[str, Any] = {
        "kb_generation": record.generation,
        "workspace": record.workspace,
        "idempotency_fingerprint": "legacy",
    }
    payload.pop(missing_field)
    legacy = _job(
        record.id,
        record.workspace,
        job_id=f"legacy_{missing_field}",
        job_type="clear_kb",
        status="running",
        payload=payload,
    )
    await store.create_job(legacy)

    result = await deletion_service.resume_hard_delete(legacy)

    assert result.job.status == "failed"
    assert any("invalid_clear_payload" in error for error in result.errors)
    assert probe.drop_calls == 0
    assert probe.finalized == []
    assert object_storage.deleted_workspaces == []
    assert input_workspace.exists()
    assert working_workspace.exists()
    assert (await kb_service.get(record.id, include_deleted=True)).status == "deleted"
    _documents, total = await store.list_documents(record.id)
    assert total == 1
    lifecycle = await store.get_kb_lifecycle(record.id)
    assert lifecycle is not None and lifecycle.state == "active"
    duplicate = await deletion_service.enqueue_hard_delete(
        record.id,
        expected_generation=record.generation,
    )
    assert duplicate.id == legacy.id
    jobs, _total = await store.list_jobs(record.id)
    assert len([job for job in jobs if job.job_type == "clear_kb"]) == 1


@pytest.mark.asyncio
async def test_restored_catalog_status_is_checked_before_destroy(tmp_path: Path):
    (
        kb_service,
        store,
        _registry,
        deletion_service,
        probe,
        record,
        input_workspace,
        working_workspace,
    ) = await _build_environment(tmp_path, kb_id="kb_restore_race")
    await _soft_delete(kb_service, record)
    queued = await deletion_service.enqueue_hard_delete(
        record.id,
        expected_generation=record.generation,
    )
    running = await store.transition_job(record.id, queued.id, status="running")
    await kb_service.restore(
        record.id,
        expected_generation=record.generation,
    )

    result = await deletion_service.resume_hard_delete(running)

    assert result.job.status == "failed"
    assert any("status is not deleted" in error for error in result.errors)
    assert probe.drop_calls == 0
    assert input_workspace.exists()
    assert working_workspace.exists()
    assert (await kb_service.get(record.id)).status == "active"
    lifecycle = await store.get_kb_lifecycle(record.id)
    assert lifecycle is not None and lifecycle.state == "active"


@pytest.mark.asyncio
async def test_catalog_is_rechecked_inside_deletion_guard_before_destroy(
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
    ) = await _build_environment(tmp_path, kb_id="kb_inner_cas")
    await _soft_delete(kb_service, record)
    queued = await deletion_service.enqueue_hard_delete(
        record.id,
        expected_generation=record.generation,
    )
    running = await store.transition_job(record.id, queued.id, status="running")
    original_get = kb_service.get
    calls = 0

    async def race_get(kb_id: str, *, include_deleted: bool = False):
        nonlocal calls
        calls += 1
        if calls == 2:
            await kb_service.restore(
                record.id,
                expected_generation=record.generation,
            )
        return await original_get(kb_id, include_deleted=include_deleted)

    monkeypatch.setattr(kb_service, "get", race_get)
    result = await deletion_service.resume_hard_delete(running)

    assert calls == 2
    assert result.job.status == "failed"
    assert probe.drop_calls == 0
    assert input_workspace.exists()
    assert working_workspace.exists()
    assert (await original_get(record.id)).status == "active"
    lifecycle = await store.get_kb_lifecycle(record.id)
    assert lifecycle is not None and lifecycle.state == "active"
    assert lifecycle.delete_job_id is None


@pytest.mark.asyncio
async def test_old_generation_job_cannot_touch_recreated_generation(tmp_path: Path):
    (
        kb_service,
        store,
        _registry,
        deletion_service,
        probe,
        old_record,
        _input_workspace,
        _working_workspace,
    ) = await _build_environment(tmp_path, kb_id="kb_new_generation")
    await _soft_delete(kb_service, old_record)
    old_job = await deletion_service.enqueue_hard_delete(
        old_record.id,
        expected_generation=old_record.generation,
    )
    async with store.kb_deletion_guard(
        old_record.id,
        old_record.generation,
        old_job.id,
    ):
        await store.complete_kb_deletion(
            old_record.id,
            old_record.generation,
            old_job.id,
        )
        await kb_service.purge(
            old_record.id,
            expected_generation=old_record.generation,
            expected_status="deleted",
        )
    new_record = await kb_service.create(
        kb_id=old_record.id,
        name="New generation",
    )
    await store.activate_kb_generation(new_record.id, new_record.generation)
    running = await store.transition_job(
        old_record.id,
        old_job.id,
        status="running",
    )

    result = await deletion_service.resume_hard_delete(running)

    assert result.job.status == "failed"
    assert any("generation changed" in error for error in result.errors)
    assert probe.drop_calls == 0
    current = await kb_service.get(new_record.id)
    assert current.generation == new_record.generation
    assert current.status == "active"


@pytest.mark.asyncio
async def test_registry_drop_uses_pinned_record_without_catalog_reread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    kb_service = KnowledgeBaseService(tmp_path / "kb.json")
    record = await kb_service.create(kb_id="kb_pinned_registry", name="Pinned")
    probe = BuilderProbe()
    registry = LightRAGInstanceRegistry(kb_service, probe.build, probe.finalize)

    async def forbidden_get(*_args: Any, **_kwargs: Any):
        raise AssertionError("drop_kb_data must not re-read the catalog")

    monkeypatch.setattr(kb_service, "get", forbidden_get)
    async with registry.destructive_lock(record.id):
        summary = await registry.drop_kb_data(record)

    assert summary["dropped"] == 12
    assert probe.records == [record]
    assert probe.records[0] is record


@pytest.mark.asyncio
async def test_success_allows_new_generation_and_old_succeeded_job_is_noop(
    tmp_path: Path
):
    (
        kb_service,
        store,
        _registry,
        deletion_service,
        probe,
        old_record,
        _input_workspace,
        _working_workspace,
    ) = await _build_environment(tmp_path, kb_id="kb_reusable")
    await _soft_delete(kb_service, old_record)
    completed = await deletion_service.hard_delete(
        old_record.id,
        expected_generation=old_record.generation,
    )
    assert completed.job.status == "succeeded"

    new_record = await kb_service.create(kb_id=old_record.id, name="Recreated")
    assert new_record.generation != old_record.generation
    await store.activate_kb_generation(new_record.id, new_record.generation)
    side_effect_snapshot = (probe.drop_calls, len(probe.instances), len(probe.finalized))

    resumed = await deletion_service.resume_hard_delete(completed.job)

    assert resumed.job.status == "succeeded"
    assert (probe.drop_calls, len(probe.instances), len(probe.finalized)) == (
        side_effect_snapshot
    )
    current = await kb_service.get(new_record.id)
    assert current.generation == new_record.generation
    assert current.status == "active"


@pytest.mark.asyncio
async def test_generic_retry_preserves_clear_idempotency_key(tmp_path: Path):
    (
        kb_service,
        store,
        _registry,
        deletion_service,
        _probe,
        record,
        _input_workspace,
        _working_workspace,
    ) = await _build_environment(tmp_path, kb_id="kb_generic_retry")
    await _soft_delete(kb_service, record)
    queued = await deletion_service.enqueue_hard_delete(
        record.id,
        expected_generation=record.generation,
    )
    failed = await store.transition_job(
        record.id,
        queued.id,
        status="failed",
        error_code="test",
        error_message="test",
    )
    reset = await store.reset_job_for_retry(
        record.id,
        failed.id,
        new_idempotency_key=None,
    )

    assert reset.id == queued.id
    assert reset.status == "queued"
    assert reset.idempotency_key == queued.idempotency_key


@pytest.mark.asyncio
async def test_resume_executor_drives_claimed_job(tmp_path: Path):
    from lightrag.api.job_worker import build_clear_kb_executor

    (
        kb_service,
        store,
        _registry,
        deletion_service,
        _probe,
        record,
        _input_workspace,
        _working_workspace,
    ) = await _build_environment(tmp_path, kb_id="kb_worker_resume")
    await _soft_delete(kb_service, record)
    queued = await deletion_service.enqueue_hard_delete(
        record.id,
        expected_generation=record.generation,
    )
    claimed = await store.claim_next_worker_job(
        job_types=["clear_kb"],
        max_queued_at=None,
    )
    assert claimed is not None and claimed.id == queued.id

    executor = build_clear_kb_executor(deletion_service=deletion_service)
    await executor(claimed)

    final = await store.get_job(record.id, queued.id)
    assert final.status == "succeeded"


@pytest.mark.asyncio
async def test_worker_job_guard_reenters_clear_service_without_deadlock(
    tmp_path: Path,
):
    from lightrag.api.job_worker import JobWorker, build_clear_kb_executor

    (
        kb_service,
        store,
        _registry,
        deletion_service,
        _probe,
        record,
        _input_workspace,
        _working_workspace,
    ) = await _build_environment(tmp_path, kb_id="kb_worker_nested_guard")
    await _soft_delete(kb_service, record)
    queued = await deletion_service.enqueue_hard_delete(
        record.id,
        expected_generation=record.generation,
    )
    worker = JobWorker(
        JobService(kb_service, store),
        executors={
            "clear_kb": build_clear_kb_executor(
                deletion_service=deletion_service
            )
        },
        claim_grace_seconds=0,
    )

    claimed = await asyncio.wait_for(worker.poll_once(), timeout=5)

    assert claimed is not None and claimed.id == queued.id
    assert (await store.get_job(record.id, queued.id)).status == "succeeded"
