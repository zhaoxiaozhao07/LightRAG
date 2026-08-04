from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

import pytest

from lightrag.api.job_service import JobService
from lightrag.api.metadata_store import SQLiteMetadataStore
from lightrag.artifact_runtime import (
    PipelineArtifactBinding,
    PipelineAttemptCommitStaleError,
    PipelineAttemptRowKind,
    extract_pipeline_attempt_token,
)
from lightrag.pipeline import _PipelineMixin
from tests.api.test_artifact_storage_phase2a import (
    _FakeObjectStorage,
    _build_object_service,
    _create_document,
)


pytestmark = pytest.mark.offline


def _assert_binding_safe(value: Any) -> None:
    encoded = json.dumps(value, ensure_ascii=False, default=str)
    assert ".lightrag-scratch" not in encoded
    assert "sidecar_location" not in encoded
    assert "blocks_path" not in encoded


def _assert_no_scratch(value: Any) -> None:
    encoded = json.dumps(value, ensure_ascii=False, default=str)
    assert ".lightrag-scratch" not in encoded


class _RejectingJobMetadataStore(SQLiteMetadataStore):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.job_writes: list[dict[str, Any]] = []

    async def create_job(self, job):
        payload = asdict(job)
        _assert_no_scratch(payload)
        self.job_writes.append(payload)
        return await super().create_job(job)

    async def create_job_once(self, job):
        payload = asdict(job)
        _assert_no_scratch(payload)
        self.job_writes.append(payload)
        return await super().create_job_once(job)


class _RejectingDurableStore:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.writes: list[dict[str, dict[str, Any]]] = []
        self.fail_next_upsert = False
        self.fail_on_upsert_number: int | None = None
        self.upsert_attempts = 0
        self.lock = asyncio.Lock()
        self.pause_on_upsert_number: int | None = None
        self.cas_entered = asyncio.Event()
        self.cas_release = asyncio.Event()

    async def get_by_id(self, key: str) -> dict[str, Any] | None:
        async with self.lock:
            value = self.records.get(key)
            return deepcopy(value) if value is not None else None

    async def upsert(self, values: dict[str, dict[str, Any]]) -> None:
        _assert_binding_safe(values)
        self.upsert_attempts += 1
        if self.fail_next_upsert or self.upsert_attempts == self.fail_on_upsert_number:
            self.fail_next_upsert = False
            raise RuntimeError("injected durable upsert failure")
        copied = deepcopy(values)
        async with self.lock:
            self.writes.append(copied)
            self.records.update(copied)

    async def replace_attempt(self, key: str, value: Mapping[str, Any]) -> None:
        _assert_binding_safe(value)
        async with self.lock:
            self.records[key] = deepcopy(dict(value))

    async def compare_and_commit_pipeline_attempt(
        self,
        key: str,
        payload: Mapping[str, Any],
        *,
        expected_attempt_token: str,
        row_kind: PipelineAttemptRowKind,
    ) -> bool:
        _assert_binding_safe(payload)
        self.upsert_attempts += 1
        attempt_number = self.upsert_attempts
        if attempt_number == self.pause_on_upsert_number:
            self.cas_entered.set()
            await self.cas_release.wait()
        if self.fail_next_upsert or attempt_number == self.fail_on_upsert_number:
            self.fail_next_upsert = False
            raise RuntimeError("injected durable upsert failure")
        async with self.lock:
            current = self.records.get(key)
            if (
                extract_pipeline_attempt_token(current, row_kind=row_kind)
                != expected_attempt_token
            ):
                return False
            copied = deepcopy(dict(payload))
            self.records[key] = copied
            self.writes.append({key: deepcopy(copied)})
            return True

    async def index_done_callback(self) -> None:
        return None


class _LegacyBindingParseRAG(_PipelineMixin):
    def __init__(self, workspace: str) -> None:
        self.workspace = workspace
        self.addon_params: dict[str, Any] = {}
        self.full_docs = _RejectingDurableStore()
        self.doc_status = _RejectingDurableStore()

    def _resolve_source_file_for_parser(
        self,
        file_path: str,
        *,
        source_file: str | None = None,
        parser_engine: str | None = None,
    ) -> str:
        del source_file, parser_engine
        return file_path


async def _prepare_parse_context(tmp_path: Path):
    from lightrag.api.kb_service import KnowledgeBaseService

    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kbs.json")
    metadata_store = _RejectingJobMetadataStore(
        tmp_path / "metadata" / "metadata.sqlite3"
    )
    record = await kb_service.create(kb_id="kb_binding", name="binding")
    storage = _FakeObjectStorage()
    service, materializer = _build_object_service(
        root=tmp_path / "inputs",
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
    )
    document, _upload_job = await _create_document(
        service,
        record.id,
        source_name="note.txt",
        content=b"durable parse binding",
    )
    plan = await service.create_parse_plan(
        record.id,
        document.id,
        parser_engine="legacy",
    )
    job_service = JobService(kb_service, metadata_store)
    job, _created = await job_service.create_parse_job_once(
        record.id,
        document_id=document.id,
        parser_hash=plan.parser_hash,
        lightrag_doc_id=plan.lightrag_doc_id,
        parser_engine=plan.parser_engine,
        process_options=plan.process_options,
        source_hash=plan.document.source_hash,
        source_name=plan.source_name,
        source_object_uri=plan.source_object_uri,
        raw_object_refs=[
            {
                "artifact_id": ref.artifact_id,
                "object_prefix_uri": ref.object_prefix_uri,
                "directory_name": ref.directory_name,
                "checksum": ref.checksum,
            }
            for ref in plan.raw_object_refs
        ],
    )
    await service.mark_parse_queued(record.id, document.id, job=job, plan=plan)
    assert metadata_store.job_writes
    rag = _LegacyBindingParseRAG(record.workspace)
    return service, materializer, metadata_store, plan, job, rag, job_service


async def _prepare_claimed_parse(tmp_path: Path):
    (
        service,
        materializer,
        metadata_store,
        plan,
        job,
        rag,
        _job_service,
    ) = await _prepare_parse_context(tmp_path)
    await service.mark_parse_running(
        plan.document.kb_id,
        plan.document.id,
        job_id=job.id,
        claim_token=plan.claim_token,
        plan=plan,
    )
    execution = await service.materialize_parse_execution(plan)
    return service, materializer, metadata_store, plan, job, rag, execution


@pytest.mark.asyncio
async def test_object_parse_first_write_is_claimed_binding_then_committed(
    tmp_path,
) -> None:
    (
        service,
        materializer,
        metadata_store,
        plan,
        job,
        rag,
        execution,
    ) = await _prepare_claimed_parse(tmp_path)
    try:
        parsed_data = await service.run_parse(rag, plan, execution)

        first_write = rag.full_docs.writes[0][plan.lightrag_doc_id]
        _assert_binding_safe(first_write)
        assert first_write["file_path"] == "note.txt"
        assert "sidecar_location" not in first_write
        assert "blocks_path" not in first_write
        claimed = PipelineArtifactBinding.from_mapping(
            first_write["artifact_binding"],
            expected_workspace=plan.document.workspace,
        )
        assert claimed.state == "claimed"
        assert claimed.operation == "parse"
        assert claimed.kb_generation == plan.kb_generation
        assert claimed.document_id == plan.document.id
        assert claimed.job_id == job.id
        assert claimed.claim_token == plan.claim_token
        assert Path(parsed_data["blocks_path"]).is_relative_to(
            materializer.scratch_root
        )

        await service.finalize_parse_runtime_references(
            rag, plan, execution, parsed_data
        )
        finalized = rag.full_docs.records[plan.lightrag_doc_id]
        _assert_binding_safe(finalized)
        assert "sidecar_location" not in finalized
        assert "blocks_path" not in finalized

        result = await service.complete_parse(
            plan.document.kb_id,
            plan.document.id,
            job_id=job.id,
            plan=plan,
            execution=execution,
            parsed_data=parsed_data,
        )
        await service.commit_parse_artifact_binding(rag, plan, result)

        committed_row = rag.full_docs.records[plan.lightrag_doc_id]
        _assert_binding_safe(committed_row)
        committed = PipelineArtifactBinding.from_mapping(
            committed_row["artifact_binding"],
            expected_workspace=plan.document.workspace,
        )
        assert committed.state == "committed"
        assert committed.parse_generation_id == plan.claim_token
        assert committed.sidecar_artifact_id == result.document.metadata.get(
            "current_sidecar_artifact_id"
        )
        assert committed.blocks_artifact_id == result.document.metadata.get(
            "current_blocks_artifact_id"
        )
        assert committed.raw_artifact_ids == tuple(
            artifact.id
            for artifact in result.artifacts
            if artifact.artifact_type == "raw_dir"
        )
        persisted = await metadata_store.get_document(
            plan.document.kb_id, plan.document.id
        )
        assert persisted.metadata["current_parse_generation_id"] == plan.claim_token
    finally:
        execution.cleanup()


@pytest.mark.asyncio
async def test_parse_finalizer_fault_is_explicit_and_first_write_remains_safe(
    tmp_path,
) -> None:
    (
        service,
        _materializer,
        _store,
        plan,
        _job,
        rag,
        execution,
    ) = await _prepare_claimed_parse(tmp_path)
    try:
        parsed_data = await service.run_parse(rag, plan, execution)
        first_write = dict(rag.full_docs.records[plan.lightrag_doc_id])
        _assert_binding_safe(first_write)
        assert (
            PipelineArtifactBinding.from_mapping(first_write["artifact_binding"]).state
            == "claimed"
        )

        rag.full_docs.fail_next_upsert = True
        with pytest.raises(RuntimeError, match="injected durable upsert failure"):
            await service.finalize_parse_runtime_references(
                rag, plan, execution, parsed_data
            )

        retained = rag.full_docs.records[plan.lightrag_doc_id]
        assert retained == first_write
        _assert_binding_safe(retained)
    finally:
        execution.cleanup()


@pytest.mark.asyncio
async def test_committed_binding_patch_failure_is_explicit_and_retains_safe_claimed_row(
    tmp_path,
) -> None:
    (
        service,
        _materializer,
        store,
        plan,
        job,
        rag,
        execution,
    ) = await _prepare_claimed_parse(tmp_path)
    try:
        parsed_data = await service.run_parse(rag, plan, execution)
        await service.finalize_parse_runtime_references(
            rag, plan, execution, parsed_data
        )
        result = await service.complete_parse(
            plan.document.kb_id,
            plan.document.id,
            job_id=job.id,
            plan=plan,
            execution=execution,
            parsed_data=parsed_data,
        )
        rag.full_docs.fail_next_upsert = True
        with pytest.raises(RuntimeError, match="injected durable upsert failure"):
            await service.commit_parse_artifact_binding(rag, plan, result)

        retained = rag.full_docs.records[plan.lightrag_doc_id]
        _assert_binding_safe(retained)
        assert (
            PipelineArtifactBinding.from_mapping(retained["artifact_binding"]).state
            == "claimed"
        )
        committed_document = await store.get_document(
            plan.document.kb_id, plan.document.id
        )
        assert (
            committed_document.metadata["current_parse_generation_id"]
            == plan.claim_token
        )
    finally:
        execution.cleanup()


@pytest.mark.asyncio
async def test_parse_executor_reports_commit_patch_failure_before_job_terminal(
    tmp_path,
) -> None:
    from lightrag.api.routers.kb_document_routes import _execute_parse_plan

    (
        service,
        materializer,
        store,
        plan,
        job,
        rag,
        job_service,
    ) = await _prepare_parse_context(tmp_path)
    rag.full_docs.fail_on_upsert_number = 3

    item = await _execute_parse_plan(
        document_service=service,
        kb_id=plan.document.kb_id,
        job_id=job.id,
        plan=plan,
        rag=rag,
        job_service=job_service,
    )

    assert item["status"] == "failed"
    assert item["error_code"] == "artifact_binding_commit_failed"
    retained = rag.full_docs.records[plan.lightrag_doc_id]
    _assert_binding_safe(retained)
    assert (
        PipelineArtifactBinding.from_mapping(retained["artifact_binding"]).state
        == "claimed"
    )
    committed_document = await store.get_document(plan.document.kb_id, plan.document.id)
    assert (
        committed_document.metadata["current_parse_generation_id"] == plan.claim_token
    )
    durable_job = await job_service.get_job(plan.document.kb_id, job.id)
    assert durable_job.status not in {"succeeded", "failed", "cancelled"}
    assert not list(materializer.scratch_root.iterdir())


@pytest.mark.asyncio
async def test_parse_commit_takeover_between_read_and_cas_preserves_new_claim(
    tmp_path,
) -> None:
    (
        service,
        _materializer,
        _store,
        plan,
        job,
        rag,
        execution,
    ) = await _prepare_claimed_parse(tmp_path)
    try:
        parsed_data = await service.run_parse(rag, plan, execution)
        await service.finalize_parse_runtime_references(
            rag,
            plan,
            execution,
            parsed_data,
        )
        result = await service.complete_parse(
            plan.document.kb_id,
            plan.document.id,
            job_id=job.id,
            plan=plan,
            execution=execution,
            parsed_data=parsed_data,
        )

        # run_parse origin write is attempt 1; finalizer CAS is attempt 2.
        rag.full_docs.pause_on_upsert_number = 3
        old_commit = asyncio.create_task(
            service.commit_parse_artifact_binding(rag, plan, result)
        )
        await asyncio.wait_for(rag.full_docs.cas_entered.wait(), timeout=2)

        current = await rag.full_docs.get_by_id(plan.lightrag_doc_id)
        assert current is not None
        old_binding = PipelineArtifactBinding.from_mapping(
            current["artifact_binding"],
            expected_workspace=plan.document.workspace,
        )
        newer_token = "parse-attempt-newer"
        newer_binding = replace(
            old_binding,
            job_id="job-parse-newer",
            claim_token=newer_token,
            parse_generation_id=newer_token,
        )
        newer_row = dict(current)
        newer_row["artifact_binding"] = newer_binding.to_dict()
        await rag.full_docs.replace_attempt(plan.lightrag_doc_id, newer_row)
        rag.full_docs.cas_release.set()

        with pytest.raises(PipelineAttemptCommitStaleError):
            await asyncio.wait_for(old_commit, timeout=2)

        retained = rag.full_docs.records[plan.lightrag_doc_id]
        assert (
            PipelineArtifactBinding.from_mapping(
                retained["artifact_binding"],
                expected_workspace=plan.document.workspace,
            )
            == newer_binding
        )
        _assert_binding_safe(retained)
    finally:
        execution.cleanup()
