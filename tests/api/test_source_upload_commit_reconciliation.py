from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from lightrag.api import document_lifecycle_service as lifecycle_module
from lightrag.api.commit_reconciliation import (
    MetadataCommitOutcomeUnknownError,
    await_cancellation_safe_reconciliation,
)
from lightrag.api.document_lifecycle_service import DocumentSourceInput
from lightrag.api.kb_service import KnowledgeBaseService
from lightrag.api.metadata_store import DocumentRecord, JobRecord, SQLiteMetadataStore
from tests.api.test_artifact_storage_phase2a import (
    _FakeObjectStorage,
    _build_object_service,
)


class _CommitFaultMetadataStore:
    def __init__(
        self,
        delegate: SQLiteMetadataStore,
        *,
        mode: str,
        read_release: asyncio.Event | None = None,
        commit_release: asyncio.Event | None = None,
    ) -> None:
        self.delegate = delegate
        self.mode = mode
        self.read_release = read_release
        self.commit_release = commit_release
        self.commit_saved = asyncio.Event()
        self.read_started = asyncio.Event()
        self.read_completed = asyncio.Event()
        self.candidate_documents: list[DocumentRecord] = []
        self.candidate_job: JobRecord | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    async def create_documents_and_job(
        self,
        documents: list[DocumentRecord],
        job: JobRecord,
    ) -> tuple[list[DocumentRecord], JobRecord, bool]:
        self.candidate_documents = list(documents)
        self.candidate_job = job
        if self.mode in {
            "committed",
            "committed_blocked_read",
            "committed_blocked_commit",
        }:
            await self.delegate.create_documents_and_job(documents, job)
            self.commit_saved.set()
            if self.mode == "committed_blocked_commit":
                assert self.commit_release is not None
                await self.commit_release.wait()
        elif self.mode == "partial":
            await self.delegate.create_documents_and_job(documents[:1], job)
        elif self.mode not in {"rollback", "read_failure"}:
            raise AssertionError(f"Unsupported commit fault mode: {self.mode}")
        raise RuntimeError("metadata connection lost after transaction boundary")

    async def get_documents_and_job_by_ids(
        self,
        kb_id: str,
        document_ids: list[str],
        job_id: str,
    ) -> tuple[list[DocumentRecord], JobRecord | None]:
        self.read_started.set()
        if self.read_release is not None:
            await self.read_release.wait()
        if self.mode == "read_failure":
            raise RuntimeError("metadata read-back unavailable")
        result = await self.delegate.get_documents_and_job_by_ids(
            kb_id,
            document_ids,
            job_id,
        )
        self.read_completed.set()
        return result


async def _build_faulting_source_service(
    tmp_path: Path,
    *,
    mode: str,
    read_release=None,
    commit_release=None,
):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kbs.json")
    await kb_service.create(kb_id="kb_commit", name="commit")
    delegate = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    metadata_store = _CommitFaultMetadataStore(
        delegate,
        mode=mode,
        read_release=read_release,
        commit_release=commit_release,
    )
    storage = _FakeObjectStorage()
    service, _materializer = _build_object_service(
        root=tmp_path / "inputs",
        kb_service=kb_service,
        metadata_store=metadata_store,  # type: ignore[arg-type]
        storage=storage,
    )
    return service, metadata_store, delegate, storage


def _sources(count: int = 1) -> list[DocumentSourceInput]:
    return [
        DocumentSourceInput(
            source_name=f"source-{index}.txt",
            content=f"source-content-{index}".encode(),
            source_type="upload",
            content_type="text/plain",
        )
        for index in range(count)
    ]


def _candidate_object_uris(store: _CommitFaultMetadataStore) -> list[str]:
    return [
        str(document.metadata["source_object_uri"])
        for document in store.candidate_documents
    ]


def _candidate_local_paths(store: _CommitFaultMetadataStore) -> list[Path]:
    return [Path(document.source_uri) for document in store.candidate_documents]


async def test_committed_readback_returns_result_without_deleting_objects(tmp_path):
    service, store, delegate, storage = await _build_faulting_source_service(
        tmp_path,
        mode="committed",
    )

    result = await service.create_source_batch("kb_commit", _sources())

    candidate_ids = [document.id for document in store.candidate_documents]
    assert [document.id for document in result.documents] == candidate_ids
    assert result.job.id == store.candidate_job.id
    assert result.created is True
    assert all(uri in storage.files for uri in _candidate_object_uris(store))
    assert storage.deleted_files == []
    assert all(not path.exists() for path in _candidate_local_paths(store))
    persisted_documents, persisted_job = await delegate.get_documents_and_job_by_ids(
        "kb_commit",
        candidate_ids,
        result.job.id,
    )
    assert [document.id for document in persisted_documents] == candidate_ids
    assert persisted_job is not None and persisted_job.job_type == "upload"


async def test_rolled_back_readback_deletes_candidate_objects_and_rethrows(tmp_path):
    service, store, delegate, storage = await _build_faulting_source_service(
        tmp_path,
        mode="rollback",
    )

    with pytest.raises(RuntimeError, match="metadata connection lost"):
        await service.create_source_batch("kb_commit", _sources())

    object_uris = _candidate_object_uris(store)
    assert all(uri not in storage.files for uri in object_uris)
    assert set(storage.deleted_files) == set(object_uris)
    assert all(not path.exists() for path in _candidate_local_paths(store))
    documents, job = await delegate.get_documents_and_job_by_ids(
        "kb_commit",
        [document.id for document in store.candidate_documents],
        store.candidate_job.id,
    )
    assert documents == []
    assert job is None


async def test_readback_failure_is_unknown_and_preserves_objects_and_local_sources(
    tmp_path,
    monkeypatch,
):
    warnings: list[str] = []

    def capture_warning(message: str, *args: object, **_kwargs: object) -> None:
        warnings.append(message % args if args else message)

    monkeypatch.setattr(lifecycle_module.logger, "warning", capture_warning)
    service, store, _delegate, storage = await _build_faulting_source_service(
        tmp_path,
        mode="read_failure",
    )

    with pytest.raises(MetadataCommitOutcomeUnknownError) as error:
        await service.create_source_batch("kb_commit", _sources())

    candidate_ids = [document.id for document in store.candidate_documents]
    assert error.value.candidate_document_ids == tuple(candidate_ids)
    assert all(uri in storage.files for uri in _candidate_object_uris(store))
    assert storage.deleted_files == []
    assert all(path.is_file() for path in _candidate_local_paths(store))
    warning_output = "\n".join(warnings)
    assert "candidate_document_ids" in warning_output
    assert all(uri not in warning_output for uri in _candidate_object_uris(store))


async def test_multi_document_partial_presence_is_unknown_and_preserves_all_objects(
    tmp_path,
):
    service, store, delegate, storage = await _build_faulting_source_service(
        tmp_path,
        mode="partial",
    )

    with pytest.raises(MetadataCommitOutcomeUnknownError):
        await service.create_source_batch("kb_commit", _sources(2))

    candidate_ids = [document.id for document in store.candidate_documents]
    persisted_documents, persisted_job = await delegate.get_documents_and_job_by_ids(
        "kb_commit",
        candidate_ids,
        store.candidate_job.id,
    )
    assert len(persisted_documents) == 1
    assert persisted_job is not None
    assert all(uri in storage.files for uri in _candidate_object_uris(store))
    assert storage.deleted_files == []
    assert all(path.is_file() for path in _candidate_local_paths(store))


async def test_cancellation_during_readback_waits_for_reconciliation_and_keeps_commit(
    tmp_path,
):
    read_release = asyncio.Event()
    service, store, delegate, storage = await _build_faulting_source_service(
        tmp_path,
        mode="committed_blocked_read",
        read_release=read_release,
    )
    task = asyncio.create_task(service.create_source_batch("kb_commit", _sources()))
    await asyncio.wait_for(store.read_started.wait(), timeout=5)

    task.cancel()
    await asyncio.sleep(0)
    assert task.done() is False
    assert all(uri in storage.files for uri in _candidate_object_uris(store))
    assert storage.deleted_files == []

    read_release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
    assert store.read_completed.is_set()
    assert all(uri in storage.files for uri in _candidate_object_uris(store))
    assert storage.deleted_files == []
    assert all(not path.exists() for path in _candidate_local_paths(store))
    documents, job = await delegate.get_documents_and_job_by_ids(
        "kb_commit",
        [document.id for document in store.candidate_documents],
        store.candidate_job.id,
    )
    assert len(documents) == 1
    assert job is not None


async def test_cancelled_metadata_call_reconciles_committed_transaction(tmp_path):
    commit_release = asyncio.Event()
    service, store, delegate, storage = await _build_faulting_source_service(
        tmp_path,
        mode="committed_blocked_commit",
        commit_release=commit_release,
    )
    task = asyncio.create_task(service.create_source_batch("kb_commit", _sources()))
    await asyncio.wait_for(store.commit_saved.wait(), timeout=5)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)

    assert store.read_completed.is_set()
    assert all(uri in storage.files for uri in _candidate_object_uris(store))
    assert storage.deleted_files == []
    assert all(not path.exists() for path in _candidate_local_paths(store))
    documents, job = await delegate.get_documents_and_job_by_ids(
        "kb_commit",
        [document.id for document in store.candidate_documents],
        store.candidate_job.id,
    )
    assert len(documents) == 1
    assert job is not None


async def test_shared_reconciliation_propagates_unrelated_read_exception():
    async def fail_readback() -> object:
        raise ValueError("classifier bug")

    with pytest.raises(ValueError, match="classifier bug"):
        await await_cancellation_safe_reconciliation(fail_readback)


async def test_shared_reconciliation_timeout_does_not_cancel_read_task():
    release = asyncio.Event()
    completed = asyncio.Event()

    async def slow_readback() -> str:
        await release.wait()
        completed.set()
        return "committed"

    with pytest.raises(TimeoutError):
        await await_cancellation_safe_reconciliation(
            slow_readback,
            timeout=0.01,
        )
    release.set()
    await asyncio.wait_for(completed.wait(), timeout=5)
