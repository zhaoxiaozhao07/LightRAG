from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from lightrag.api.index_build_service import IndexBuildService
from lightrag.api.job_service import JobService
from lightrag.api.kb_service import KnowledgeBaseService
from lightrag.api.metadata_store import (
    ArtifactPointerConflictError,
    ArtifactRecord,
    DocumentRecord,
    SQLiteMetadataStore,
)
from tests.api.test_artifact_storage_phase2a import (
    _FakeObjectStorage,
    _ParserRAG,
    _build_object_service,
    _create_document,
    _execute_one_parse,
)
from tests.api.test_artifact_storage_phase2b import (
    _attach_processing_owner,
    _build_object_service as _build_phase2b_object_service,
    _execute_build,
    _setup_parsed_object_document,
)


class _ArtifactCommitFaultStore:
    def __init__(
        self,
        delegate: SQLiteMetadataStore,
        *,
        target: str,
        mode: str,
        read_release: asyncio.Event | None = None,
    ) -> None:
        self.delegate = delegate
        self.target = target
        self.mode = mode
        self.read_release = read_release
        self.read_started = asyncio.Event()
        self.read_completed = asyncio.Event()
        self.candidate_artifacts: list[ArtifactRecord] = []
        self.document_id: str | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    async def complete_document_parse(
        self,
        kb_id: str,
        document_id: str,
        **kwargs: Any,
    ) -> tuple[DocumentRecord, list[ArtifactRecord]]:
        if self.target != "parse":
            return await self.delegate.complete_document_parse(
                kb_id, document_id, **kwargs
            )
        self.document_id = document_id
        self.candidate_artifacts = list(kwargs["artifacts"])
        if self.mode in {"committed", "committed_blocked_read"}:
            await self.delegate.complete_document_parse(kb_id, document_id, **kwargs)
        raise RuntimeError("metadata connection lost after parse commit boundary")

    async def complete_document_build_with_artifact_promotion(
        self,
        kb_id: str,
        document_id: str,
        **kwargs: Any,
    ) -> tuple[DocumentRecord, list[ArtifactRecord]]:
        if self.target != "build":
            return await self.delegate.complete_document_build_with_artifact_promotion(
                kb_id, document_id, **kwargs
            )
        self.document_id = document_id
        self.candidate_artifacts = list(kwargs["artifacts"])
        if self.mode == "committed":
            await self.delegate.complete_document_build_with_artifact_promotion(
                kb_id, document_id, **kwargs
            )
        elif self.mode == "pointer_loser":
            raise ArtifactPointerConflictError(
                "document_artifact_pointer",
                document_id,
                expected={"sidecar": "expected"},
                current={"sidecar": "winner"},
            )
        raise RuntimeError("metadata connection lost after build promotion boundary")

    async def get_document_and_artifacts_by_ids(
        self,
        kb_id: str,
        document_id: str,
        artifact_ids: list[str],
    ) -> tuple[DocumentRecord | None, dict[str, ArtifactRecord]]:
        if self.target == "build" and not self.candidate_artifacts:
            return await self.delegate.get_document_and_artifacts_by_ids(
                kb_id,
                document_id,
                artifact_ids,
            )
        self.read_started.set()
        if self.read_release is not None:
            await self.read_release.wait()
        if self.mode == "read_failure":
            if self.target == "build":
                self.read_completed.set()
            raise RuntimeError("metadata artifact read-back unavailable")
        document, artifacts = await self.delegate.get_document_and_artifacts_by_ids(
            kb_id,
            document_id,
            artifact_ids,
        )
        if self.mode == "partial" and self.candidate_artifacts:
            candidate = self.candidate_artifacts[0]
            artifacts = {candidate.id: candidate}
        self.read_completed.set()
        return document, artifacts


def _artifact_object_refs(artifacts: list[ArtifactRecord]) -> set[str]:
    refs: set[str] = set()
    for artifact in artifacts:
        for key in ("object_uri", "object_prefix_uri"):
            value = artifact.metadata.get(key)
            if isinstance(value, str) and value:
                refs.add(value)
    return refs


def _new_candidate_refs(
    storage: _FakeObjectStorage,
    artifacts: list[ArtifactRecord],
    *,
    existing_files: set[str],
    existing_prefixes: set[str],
) -> set[str]:
    return {
        ref
        for ref in _artifact_object_refs(artifacts)
        if ref not in existing_files and ref not in existing_prefixes
    }


def _object_ref_contains_file_uri(ref: str, file_uri: str) -> bool:
    return file_uri == ref or (ref.endswith("/") and file_uri.startswith(ref))


def _object_ref_present(storage: _FakeObjectStorage, ref: str) -> bool:
    return bool(
        ref in storage.files
        or ref in storage.prefixes
        or any(
            _object_ref_contains_file_uri(ref, file_uri) for file_uri in storage.files
        )
    )


def _object_ref_absent(storage: _FakeObjectStorage, ref: str) -> bool:
    return bool(
        ref not in storage.files
        and ref not in storage.prefixes
        and not any(
            _object_ref_contains_file_uri(ref, file_uri) for file_uri in storage.files
        )
    )


def _exact_file_bytes_for_refs(
    storage: _FakeObjectStorage,
    refs: set[str],
) -> dict[str, bytes]:
    return {
        file_uri: payload
        for file_uri, payload in storage.files.items()
        if any(_object_ref_contains_file_uri(ref, file_uri) for ref in refs)
    }


async def _parse_fault_setup(tmp_path: Path, *, mode: str, read_release=None):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kbs.json")
    await kb_service.create(kb_id="kb_parse_reconcile", name="parse")
    delegate = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    store = _ArtifactCommitFaultStore(
        delegate,
        target="parse",
        mode=mode,
        read_release=read_release,
    )
    storage = _FakeObjectStorage()
    service, materializer = _build_object_service(
        root=tmp_path / "inputs",
        kb_service=kb_service,
        metadata_store=store,  # type: ignore[arg-type]
        storage=storage,
    )
    document, _job = await _create_document(service, "kb_parse_reconcile")
    job_service = JobService(kb_service, store)  # type: ignore[arg-type]
    return service, store, delegate, storage, materializer, document, job_service


async def _build_fault_setup(tmp_path: Path, *, mode: str):
    (
        kb_service,
        delegate,
        storage,
        document,
        _parse_materializer,
        job_service,
        rag,
    ) = await _setup_parsed_object_document(tmp_path, kb_id="kb_build_reconcile")
    store = _ArtifactCommitFaultStore(delegate, target="build", mode=mode)
    service, materializer = _build_phase2b_object_service(
        root=tmp_path / "build-root" / "inputs",
        kb_service=kb_service,
        metadata_store=store,  # type: ignore[arg-type]
        storage=storage,
    )
    index_service = IndexBuildService(service)
    rag.mutate = True
    await _attach_processing_owner(
        rag,
        kb_id="kb_build_reconcile",
        kb_service=kb_service,
        document_service=service,
        index_service=index_service,
        materializer=materializer,
    )
    return (
        index_service,
        store,
        delegate,
        storage,
        materializer,
        document,
        job_service,
        rag,
    )


async def test_parse_commit_ack_loss_recovers_without_object_compensation(tmp_path):
    (
        service,
        store,
        delegate,
        storage,
        materializer,
        document,
        job_service,
    ) = await _parse_fault_setup(tmp_path, mode="committed")
    existing_files = set(storage.files)
    existing_prefixes = set(storage.prefixes)

    _plan, _job, item = await _execute_one_parse(
        service,
        job_service,
        kb_id="kb_parse_reconcile",
        document_id=document.id,
        rag=_ParserRAG(),
    )

    assert item["status"] == "succeeded"
    candidate_refs = _new_candidate_refs(
        storage,
        store.candidate_artifacts,
        existing_files=existing_files,
        existing_prefixes=existing_prefixes,
    )
    assert candidate_refs
    assert all(
        ref in storage.files or ref in storage.prefixes for ref in candidate_refs
    )
    assert not candidate_refs.intersection(storage.deleted_files)
    assert not candidate_refs.intersection(storage.deleted_prefixes)
    persisted = await delegate.get_document("kb_parse_reconcile", document.id)
    assert persisted.status == "parsed"
    assert persisted.metadata["current_parse_generation_id"]
    assert not list(materializer.scratch_root.iterdir())


async def test_parse_rollback_deletes_only_new_candidate_objects(tmp_path):
    (
        service,
        store,
        delegate,
        storage,
        _materializer,
        document,
        job_service,
    ) = await _parse_fault_setup(tmp_path, mode="rollback")
    existing_files = set(storage.files)
    existing_prefixes = set(storage.prefixes)

    _plan, _job, item = await _execute_one_parse(
        service,
        job_service,
        kb_id="kb_parse_reconcile",
        document_id=document.id,
        rag=_ParserRAG(),
    )

    assert item["status"] == "failed"
    candidate_refs = _new_candidate_refs(
        storage,
        store.candidate_artifacts,
        existing_files=existing_files,
        existing_prefixes=existing_prefixes,
    )
    assert candidate_refs
    assert all(
        ref not in storage.files and ref not in storage.prefixes
        for ref in candidate_refs
    )
    _, persisted_artifacts = await delegate.get_document_and_artifacts_by_ids(
        "kb_parse_reconcile",
        document.id,
        [artifact.id for artifact in store.candidate_artifacts],
    )
    assert persisted_artifacts == {}
    assert existing_files <= set(storage.files)


@pytest.mark.parametrize("mode", ["partial", "read_failure"])
async def test_parse_partial_or_read_failure_is_unknown_and_preserves_objects(
    tmp_path,
    mode,
):
    (
        service,
        store,
        _delegate,
        storage,
        _materializer,
        document,
        job_service,
    ) = await _parse_fault_setup(tmp_path, mode=mode)
    existing_files = set(storage.files)
    existing_prefixes = set(storage.prefixes)

    _plan, _job, item = await _execute_one_parse(
        service,
        job_service,
        kb_id="kb_parse_reconcile",
        document_id=document.id,
        rag=_ParserRAG(),
    )

    assert item["error_code"] == "metadata_commit_outcome_unknown"
    candidate_refs = _new_candidate_refs(
        storage,
        store.candidate_artifacts,
        existing_files=existing_files,
        existing_prefixes=existing_prefixes,
    )
    assert candidate_refs
    assert all(
        ref in storage.files or ref in storage.prefixes for ref in candidate_refs
    )
    assert not candidate_refs.intersection(storage.deleted_files)
    assert not candidate_refs.intersection(storage.deleted_prefixes)


async def test_parse_cancellation_waits_for_committed_readback_without_delete(tmp_path):
    read_release = asyncio.Event()
    (
        service,
        store,
        delegate,
        storage,
        _materializer,
        document,
        job_service,
    ) = await _parse_fault_setup(
        tmp_path,
        mode="committed_blocked_read",
        read_release=read_release,
    )
    plan = await service.create_parse_plan(
        "kb_parse_reconcile", document.id, parser_engine="mineru"
    )
    job, _created = await job_service.create_parse_job_once(
        "kb_parse_reconcile",
        document_id=document.id,
        parser_hash=plan.parser_hash,
        lightrag_doc_id=plan.lightrag_doc_id,
        parser_engine=plan.parser_engine,
        process_options=plan.process_options,
        source_hash=plan.document.source_hash,
        source_name=plan.source_name,
        source_object_uri=plan.source_object_uri,
    )
    await service.mark_parse_queued(
        "kb_parse_reconcile", document.id, job=job, plan=plan
    )
    await service.mark_parse_running(
        "kb_parse_reconcile",
        document.id,
        job_id=job.id,
        claim_token=plan.claim_token,
        plan=plan,
    )
    execution = await service.materialize_parse_execution(plan)
    rag = _ParserRAG()
    parsed_data = await service.run_parse(rag, plan, execution)
    await service.finalize_parse_runtime_references(rag, plan, execution, parsed_data)
    task = asyncio.create_task(
        service.complete_parse(
            "kb_parse_reconcile",
            document.id,
            job_id=job.id,
            plan=plan,
            execution=execution,
            parsed_data=parsed_data,
        )
    )
    await asyncio.wait_for(store.read_started.wait(), timeout=5)
    task.cancel()
    await asyncio.sleep(0)
    assert task.done() is False
    read_release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
    execution.cleanup()
    assert store.read_completed.is_set()
    assert storage.deleted_files == []
    assert storage.deleted_prefixes == []
    persisted = await delegate.get_document("kb_parse_reconcile", document.id)
    assert persisted.status == "parsed"


async def test_build_promotion_ack_loss_recovers_ready_without_deleting_objects(
    tmp_path,
):
    (
        index_service,
        store,
        delegate,
        storage,
        materializer,
        document,
        job_service,
        rag,
    ) = await _build_fault_setup(tmp_path, mode="committed")
    existing_files = dict(storage.files)
    existing_prefixes = {
        prefix_uri: dict(payload) for prefix_uri, payload in storage.prefixes.items()
    }
    upload_start = len(storage.file_uploads)
    deleted_file_start = len(storage.deleted_files)
    deleted_prefix_start = len(storage.deleted_prefixes)

    plan, job, item = await _execute_build(
        index_service,
        job_service,
        kb_id="kb_build_reconcile",
        document_id=document.id,
        rag=rag,
        force=True,
    )

    assert item["status"] == "succeeded"
    candidate_refs = _new_candidate_refs(
        storage,
        store.candidate_artifacts,
        existing_files=set(existing_files),
        existing_prefixes=set(existing_prefixes),
    )
    assert candidate_refs
    assert all(_object_ref_present(storage, ref) for ref in candidate_refs)
    candidate_file_bytes = _exact_file_bytes_for_refs(storage, candidate_refs)
    invocation_uploads = set(storage.file_uploads[upload_start:])
    assert candidate_file_bytes
    assert set(candidate_file_bytes) == invocation_uploads
    assert storage.deleted_files[deleted_file_start:] == []
    assert storage.deleted_prefixes[deleted_prefix_start:] == []
    assert set(existing_files) <= set(storage.files)
    assert all(storage.files[uri] == payload for uri, payload in existing_files.items())
    assert set(existing_prefixes) <= set(storage.prefixes)
    assert all(
        storage.prefixes[uri] == payload for uri, payload in existing_prefixes.items()
    )
    persisted = await delegate.get_document("kb_build_reconcile", document.id)
    assert persisted.status == "ready"
    assert persisted.index_hash == plan.index_hash
    assert persisted.metadata["current_build_generation_id"] == plan.claim_token
    assert persisted.metadata["last_build_job_id"] == job.id
    assert persisted.metadata["current_build_job_id"] is None
    assert persisted.metadata["current_build_claim_token"] is None
    artifacts_by_type = {
        artifact.artifact_type: artifact for artifact in store.candidate_artifacts
    }
    assert set(artifacts_by_type) == {"sidecar", "blocks"}
    assert (
        persisted.metadata["current_sidecar_artifact_id"]
        == artifacts_by_type["sidecar"].id
    )
    assert (
        persisted.metadata["current_blocks_artifact_id"]
        == artifacts_by_type["blocks"].id
    )
    assert not list(materializer.scratch_root.iterdir())


@pytest.mark.parametrize("mode", ["rollback", "pointer_loser"])
async def test_build_rollback_or_pointer_loser_deletes_candidate_objects(
    tmp_path,
    mode,
):
    (
        index_service,
        store,
        delegate,
        storage,
        _materializer,
        document,
        job_service,
        rag,
    ) = await _build_fault_setup(tmp_path, mode=mode)
    existing_files = dict(storage.files)
    existing_prefixes = {
        prefix_uri: dict(payload) for prefix_uri, payload in storage.prefixes.items()
    }
    upload_start = len(storage.file_uploads)
    deleted_file_start = len(storage.deleted_files)
    deleted_prefix_start = len(storage.deleted_prefixes)

    _plan, _job, item = await _execute_build(
        index_service,
        job_service,
        kb_id="kb_build_reconcile",
        document_id=document.id,
        rag=rag,
        force=True,
    )

    assert item["status"] == "failed"
    assert item["error_code"] == "build_failed"
    if mode == "pointer_loser":
        assert (
            "artifact_binding_stale: attempt authority changed before terminalization"
            in item["error_message"]
        )
    else:
        assert (
            "metadata connection lost after build promotion boundary"
            in item["error_message"]
        )
    candidate_refs = _new_candidate_refs(
        storage,
        store.candidate_artifacts,
        existing_files=set(existing_files),
        existing_prefixes=set(existing_prefixes),
    )
    assert candidate_refs
    assert all(_object_ref_absent(storage, ref) for ref in candidate_refs)
    invocation_uploads = set(storage.file_uploads[upload_start:])
    assert invocation_uploads
    assert all(
        any(_object_ref_contains_file_uri(ref, file_uri) for ref in candidate_refs)
        for file_uri in invocation_uploads
    )
    assert set(storage.deleted_files[deleted_file_start:]) == invocation_uploads
    assert storage.deleted_prefixes[deleted_prefix_start:] == []
    _, persisted_artifacts = await delegate.get_document_and_artifacts_by_ids(
        "kb_build_reconcile",
        document.id,
        [artifact.id for artifact in store.candidate_artifacts],
    )
    assert persisted_artifacts == {}
    assert set(existing_files) <= set(storage.files)
    assert all(storage.files[uri] == payload for uri, payload in existing_files.items())
    assert set(existing_prefixes) <= set(storage.prefixes)
    assert all(
        storage.prefixes[uri] == payload for uri, payload in existing_prefixes.items()
    )


@pytest.mark.parametrize("mode", ["partial", "read_failure"])
async def test_build_partial_or_read_failure_is_unknown_and_preserves_objects(
    tmp_path,
    mode,
):
    (
        index_service,
        store,
        delegate,
        storage,
        _materializer,
        document,
        job_service,
        rag,
    ) = await _build_fault_setup(tmp_path, mode=mode)
    existing_files = dict(storage.files)
    existing_prefixes = {
        prefix_uri: dict(payload) for prefix_uri, payload in storage.prefixes.items()
    }
    upload_start = len(storage.file_uploads)
    deleted_file_start = len(storage.deleted_files)
    deleted_prefix_start = len(storage.deleted_prefixes)

    task = asyncio.create_task(
        _execute_build(
            index_service,
            job_service,
            kb_id="kb_build_reconcile",
            document_id=document.id,
            rag=rag,
            force=True,
        )
    )
    try:
        await asyncio.wait_for(store.read_started.wait(), timeout=5)
        assert store.candidate_artifacts
        await asyncio.wait_for(store.read_completed.wait(), timeout=5)
        assert rag.runtime_sidecars
        runtime_sidecar = rag.runtime_sidecars[-1]

        async def wait_for_processing_session_close() -> None:
            while runtime_sidecar.exists():
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_for_processing_session_close(), timeout=5)
        assert task.done() is False

        candidate_refs = _new_candidate_refs(
            storage,
            store.candidate_artifacts,
            existing_files=set(existing_files),
            existing_prefixes=set(existing_prefixes),
        )
        assert candidate_refs
        assert all(_object_ref_present(storage, ref) for ref in candidate_refs)
        candidate_file_bytes = _exact_file_bytes_for_refs(storage, candidate_refs)
        invocation_uploads = set(storage.file_uploads[upload_start:])
        assert candidate_file_bytes
        assert set(candidate_file_bytes) == invocation_uploads
        assert storage.deleted_files[deleted_file_start:] == []
        assert storage.deleted_prefixes[deleted_prefix_start:] == []

        running_jobs, total = await job_service.list_jobs(
            "kb_build_reconcile",
            statuses=["running"],
            document_id=document.id,
        )
        assert total == 1
        assert len(running_jobs) == 1
        owner_job = running_jobs[0]
        assert owner_job.job_type == "build_kg"
        claim_tokens = {
            artifact.metadata.get("build_generation_id")
            for artifact in store.candidate_artifacts
        }
        assert len(claim_tokens) == 1
        owner_claim_token = next(iter(claim_tokens))
        assert isinstance(owner_claim_token, str) and owner_claim_token

        in_flight = await delegate.get_document("kb_build_reconcile", document.id)
        assert in_flight.status == "building"
        assert in_flight.metadata["pending_build_job_id"] is None
        assert in_flight.metadata["pending_build_claim_token"] is None
        assert in_flight.metadata["current_build_job_id"] == owner_job.id
        assert in_flight.metadata["current_build_claim_token"] == owner_claim_token

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)

        assert (
            _exact_file_bytes_for_refs(storage, candidate_refs) == candidate_file_bytes
        )
        assert all(_object_ref_present(storage, ref) for ref in candidate_refs)
        assert storage.deleted_files[deleted_file_start:] == []
        assert storage.deleted_prefixes[deleted_prefix_start:] == []
        assert set(existing_files) <= set(storage.files)
        assert all(
            storage.files[uri] == payload for uri, payload in existing_files.items()
        )
        assert set(existing_prefixes) <= set(storage.prefixes)
        assert all(
            storage.prefixes[uri] == payload
            for uri, payload in existing_prefixes.items()
        )

        after_cancel = await delegate.get_document("kb_build_reconcile", document.id)
        assert after_cancel.status == "building"
        assert after_cancel.metadata["current_build_job_id"] == owner_job.id
        assert after_cancel.metadata["current_build_claim_token"] == owner_claim_token
        persisted_job = await job_service.get_job("kb_build_reconcile", owner_job.id)
        assert persisted_job.status == "running"
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
