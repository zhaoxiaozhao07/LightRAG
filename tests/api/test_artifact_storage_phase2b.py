from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lightrag.artifact_runtime import (
    PipelineArtifactBinding,
    PipelineArtifactCommitOutcome,
    PipelineTerminalOutcome,
)
from lightrag.api.artifact_materialization import ArtifactMaterializer
from lightrag.api.index_build_service import IndexBuildService
from lightrag.api.job_service import JobService
from lightrag.api.kb_service import KnowledgeBaseService, utc_now_iso
from lightrag.api.metadata_store import (
    ActiveDocumentBuildJobError,
    ArtifactPointerConflictError,
    ArtifactRecord,
    SQLiteMetadataStore,
    _dumps_json,
    _loads_json_object,
)
from lightrag.api.pipeline_artifact_coordinator import PipelineArtifactCoordinator
from lightrag.api.postgres_metadata_store import PostgresMetadataStore
from tests.api.test_artifact_storage_phase2a import (
    _FakeObjectStorage,
    _ParserRAG,
    _build_object_service,
    _create_document,
    _execute_one_parse,
)
from lightrag.api.routers import kb_document_routes

pytestmark = pytest.mark.offline


class _DocStatus:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any] | None]:
        return [dict(self.rows[item]) if item in self.rows else None for item in ids]

    async def get_by_id(self, item: str) -> dict[str, Any] | None:
        value = self.rows.get(item)
        return dict(value) if value is not None else None


class _BuildRAG(_ParserRAG):
    def __init__(self, *, drain_mode: str = "self", mutate: bool = False) -> None:
        super().__init__()
        self.kb_active_index_hash = "sha256:phase2b-index"
        self.doc_status = _DocStatus()
        self.drain_mode = drain_mode
        self.mutate = mutate
        self.delete_calls: list[str] = []
        self.runtime_sidecars: list[Path] = []
        self.pipeline_artifact_materializer: (
            Callable[[PipelineArtifactBinding], Awaitable[Any]] | None
        ) = None
        self.observed_scratch_root: Path | None = None
        self.scratch_entries_at_enqueue: list[tuple[Path, ...]] = []
        self.enqueue_payloads: list[dict[str, Any]] = []
        self.drain_returned = asyncio.Event()
        self.release_owner = asyncio.Event()
        self.cancel_processing_owner = False
        self._enqueued: dict[str, tuple[PipelineArtifactBinding, str]] = {}
        self._owner_tasks: list[asyncio.Task[None]] = []

    async def adelete_by_doc_id(self, doc_id: str, **_kwargs):
        self.delete_calls.append(doc_id)
        self.doc_status.rows.pop(doc_id, None)
        self.full_docs.records.pop(doc_id, None)
        return SimpleNamespace(status="success", message="deleted")

    async def apipeline_enqueue_documents(
        self,
        *,
        input: list[str],
        ids: list[str],
        file_paths: list[str],
        lightrag_document_paths: list[str] | None = None,
        artifact_bindings: list[PipelineArtifactBinding | dict[str, Any]] | None = None,
        **_kwargs,
    ) -> None:
        assert len(input) == len(ids) == len(file_paths)
        assert lightrag_document_paths is None
        assert artifact_bindings is not None
        assert len(artifact_bindings) == len(ids)
        if self.observed_scratch_root is not None:
            self.scratch_entries_at_enqueue.append(
                tuple(self.observed_scratch_root.iterdir())
            )
        durable_bindings: list[dict[str, Any]] = []
        for doc_id, file_path, raw_binding in zip(
            ids, file_paths, artifact_bindings, strict=True
        ):
            binding = (
                raw_binding
                if isinstance(raw_binding, PipelineArtifactBinding)
                else PipelineArtifactBinding.from_mapping(raw_binding)
            )
            assert binding.operation == "build"
            assert binding.lightrag_doc_id == doc_id
            durable_binding = binding.to_dict()
            durable_bindings.append(durable_binding)
            self._enqueued[doc_id] = (binding, file_path)
            self.doc_status.rows[doc_id] = {
                "status": "pending",
                "chunks_count": None,
                "entity_count": None,
                "relation_count": None,
                "metadata": {
                    "pipeline_attempt_token": binding.claim_token,
                    "artifact_binding": durable_binding,
                },
            }
            existing = await self.full_docs.get_by_id(doc_id) or {}
            await self.full_docs.upsert(
                {
                    doc_id: {
                        **existing,
                        "file_path": file_path,
                        "parse_format": "lightrag",
                        "artifact_binding": durable_binding,
                    }
                }
            )
        self.enqueue_payloads.append(
            {
                "input": list(input),
                "ids": list(ids),
                "file_paths": list(file_paths),
                "artifact_bindings": durable_bindings,
                "lightrag_document_paths": lightrag_document_paths,
            }
        )

    async def apipeline_process_enqueue_documents(self) -> None:
        if self.drain_mode == "raise":
            self.drain_returned.set()
            raise RuntimeError("injected pipeline drain failure")
        sessions = await self._open_processing_sessions()
        if self.drain_mode in {"owner", "timeout"}:
            self.drain_returned.set()

            async def finish() -> None:
                await self.release_owner.wait()
                await self._finish_enqueued(sessions)

            self._owner_tasks.append(asyncio.create_task(finish()))
            return
        await self._finish_enqueued(sessions)
        self.drain_returned.set()

    async def _open_processing_sessions(self) -> dict[str, Any]:
        materialize = self.pipeline_artifact_materializer
        assert materialize is not None
        sessions: dict[str, Any] = {}
        for doc_id, (binding, _file_path) in self._enqueued.items():
            session = await materialize(binding)
            sidecar = session.sidecar_dir
            assert sidecar is not None and sidecar.is_dir()
            assert session.blocks_path is not None and session.blocks_path.is_file()
            sessions[doc_id] = session
            self.runtime_sidecars.append(sidecar)
            if self.mutate:
                blocks = session.blocks_path
                with blocks.open("a", encoding="utf-8") as file:
                    file.write('{"type":"content","text":"build mutation"}\n')
                (sidecar / "tables.json").write_text(
                    json.dumps(
                        {
                            "tables": [
                                {
                                    "id": "table-1",
                                    "surrounding": {"leading": "before"},
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
        return sessions

    async def _finish_enqueued(self, sessions: dict[str, Any]) -> None:
        for doc_id, session in sessions.items():
            binding, file_path = self._enqueued[doc_id]
            if self.cancel_processing_owner:
                await session.finish(PipelineTerminalOutcome.CANCELLED)
                await session.aclose()
                self.doc_status.rows[doc_id] = {
                    "status": "failed",
                    "chunks_count": None,
                    "entity_count": None,
                    "relation_count": None,
                    "error_msg": "User cancelled",
                    "metadata": {
                        "pipeline_attempt_token": binding.claim_token,
                        "artifact_binding": binding.to_dict(),
                    },
                }
                continue
            try:
                finalization = await session.handoff_success(
                    parsed_data={"entity_count": 3, "relation_count": 2},
                    chunks_count=4,
                )
            except Exception as exc:  # noqa: BLE001 - emulate pipeline terminal path
                error_message = session.redact(exc)
                with contextlib.suppress(Exception):
                    await session.finish(PipelineTerminalOutcome.FAILED)
                await session.aclose()
                self.doc_status.rows[doc_id] = {
                    "status": "failed",
                    "chunks_count": None,
                    "entity_count": None,
                    "relation_count": None,
                    "error_msg": error_message,
                    "metadata": {
                        "pipeline_attempt_token": binding.claim_token,
                        "artifact_binding": binding.to_dict(),
                    },
                }
                continue
            if finalization.outcome is PipelineArtifactCommitOutcome.UNKNOWN:
                await session.aclose()
                continue
            committed_binding = finalization.committed_binding
            assert committed_binding is not None
            existing = await self.full_docs.get_by_id(doc_id) or {}
            await self.full_docs.upsert(
                {
                    doc_id: {
                        "content": existing.get("content", ""),
                        "file_path": file_path,
                        "parse_format": existing.get("parse_format", "lightrag"),
                        "artifact_binding": committed_binding.to_dict(),
                    }
                }
            )
            await session.aclose()
            self.doc_status.rows[doc_id] = {
                "status": "processed",
                "chunks_count": finalization.chunks_count,
                "entity_count": finalization.entity_count,
                "relation_count": finalization.relation_count,
                "metadata": {
                    "pipeline_attempt_token": binding.claim_token,
                    "artifact_binding": committed_binding.to_dict(),
                },
            }

    async def wait_for_processing_owner(self) -> None:
        if self._owner_tasks:
            await asyncio.gather(*self._owner_tasks)


async def _attach_processing_owner(
    rag: _BuildRAG,
    *,
    kb_id: str,
    kb_service: KnowledgeBaseService,
    document_service: Any,
    index_service: IndexBuildService,
    materializer: ArtifactMaterializer,
) -> None:
    coordinator = PipelineArtifactCoordinator(
        kb_service, document_service, index_service
    )
    rag.pipeline_artifact_materializer = coordinator.materializer_for(
        await kb_service.get(kb_id)
    )
    rag.observed_scratch_root = materializer.scratch_root


async def _setup_parsed_object_document(
    tmp_path: Path, *, kb_id: str
) -> tuple[
    KnowledgeBaseService,
    SQLiteMetadataStore,
    _FakeObjectStorage,
    Any,
    ArtifactMaterializer,
    JobService,
    _BuildRAG,
]:
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kbs.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    await kb_service.create(name=kb_id, kb_id=kb_id)
    storage = _FakeObjectStorage()
    service, materializer = _build_object_service(
        root=tmp_path / "parse-root" / "inputs",
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
    )
    document, _upload_job = await _create_document(service, kb_id)
    job_service = JobService(kb_service, metadata_store)
    rag = _BuildRAG()
    _plan, _job, item = await _execute_one_parse(
        service,
        job_service,
        kb_id=kb_id,
        document_id=document.id,
        rag=rag,
    )
    assert item["status"] == "succeeded"
    return (
        kb_service,
        metadata_store,
        storage,
        document,
        materializer,
        job_service,
        rag,
    )


async def _execute_build(
    index_service: IndexBuildService,
    job_service: JobService,
    *,
    kb_id: str,
    document_id: str,
    rag: _BuildRAG,
    force: bool = False,
):
    plan = await index_service.create_build_plan(
        kb_id,
        document_id,
        rag=rag,
        force_rechunk=force,
        force_extract=force,
        force_embedding=force,
    )
    job, _created = await job_service.create_build_job_once(
        kb_id,
        document_id=document_id,
        parser_hash=plan.parser_hash,
        index_hash=plan.index_hash,
        source_hash=plan.document.source_hash,
        lightrag_doc_id=plan.document.lightrag_doc_id or "",
        sidecar_artifact_id=(
            plan.sidecar_artifact.id if plan.sidecar_artifact is not None else None
        ),
        blocks_artifact_id=(
            plan.blocks_artifact.id if plan.blocks_artifact is not None else None
        ),
        force_rechunk=force,
        force_extract=force,
        force_embedding=force,
    )
    await index_service.claim_build_queued(kb_id, job_id=job.id, plan=plan)
    await job_service.transition_job(kb_id, job.id, status="running", progress=0.1)
    item = await kb_document_routes._execute_build_plan(
        index_service=index_service,
        kb_id=kb_id,
        job_id=job.id,
        plan=plan,
        rag=rag,
        job_service=job_service,
    )
    if item["status"] == "succeeded":
        await job_service.transition_job(
            kb_id,
            job.id,
            status="succeeded",
            progress=1.0,
            completed_items=1,
            result={"items": [item]},
        )
    elif item["status"] == "failed":
        await job_service.transition_job(
            kb_id,
            job.id,
            status="failed",
            progress=1.0,
            failed_items=1,
            result={"items": [item]},
            error_code=item["error_code"],
            error_message=item["error_message"],
        )
    return plan, job, item


def _assert_no_scratch(value: Any) -> None:
    assert ".lightrag-scratch" not in json.dumps(value, default=str)


async def test_moved_root_build_is_metadata_only_and_materializes_before_delete(
    tmp_path, monkeypatch
):
    (
        kb_service,
        metadata_store,
        storage,
        document,
        _parse_materializer,
        job_service,
        rag,
    ) = await _setup_parsed_object_document(tmp_path, kb_id="kb_build_move")
    parsed_document = await metadata_store.get_document("kb_build_move", document.id)
    old_sidecar_id = parsed_document.metadata["current_sidecar_artifact_id"]
    old_blocks_id = parsed_document.metadata["current_blocks_artifact_id"]
    old_sidecar = await metadata_store.get_document_artifact(
        "kb_build_move", document.id, old_sidecar_id
    )
    old_blocks = await metadata_store.get_document_artifact(
        "kb_build_move", document.id, old_blocks_id
    )
    storage.prefixes[old_sidecar.metadata["object_prefix_uri"]].pop(
        Path(old_blocks.uri).name
    )
    file_download_count = len(storage.file_downloads)

    moved_service, materializer = _build_object_service(
        root=tmp_path / "moved-root" / "inputs",
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
    )
    index_service = IndexBuildService(moved_service)
    await _attach_processing_owner(
        rag,
        kb_id="kb_build_move",
        kb_service=kb_service,
        document_service=moved_service,
        index_service=index_service,
        materializer=materializer,
    )
    durable_full_docs_before_preflight = deepcopy(rag.full_docs.records)
    durable_doc_status_before_preflight = deepcopy(rag.doc_status.rows)
    materialize_preflight = index_service.materialize_build_preflight
    api_preflight_sidecars: list[Path] = []
    complete_build = index_service.complete_build
    complete_build_calls = 0

    async def observe_preflight(plan):
        preflight = await materialize_preflight(plan)
        assert preflight.runtime_sidecar_dir.is_dir()
        api_preflight_sidecars.append(preflight.runtime_sidecar_dir)
        assert rag.full_docs.records == durable_full_docs_before_preflight
        assert rag.doc_status.rows == durable_doc_status_before_preflight
        return preflight

    async def observe_complete_build(*args, **kwargs):
        nonlocal complete_build_calls
        complete_build_calls += 1
        return await complete_build(*args, **kwargs)

    monkeypatch.setattr(index_service, "materialize_build_preflight", observe_preflight)
    monkeypatch.setattr(index_service, "complete_build", observe_complete_build)
    file_upload_count = len(storage.file_uploads)
    prefix_upload_count = len(storage.prefix_uploads)
    plan, build_job, item = await _execute_build(
        index_service,
        job_service,
        kb_id="kb_build_move",
        document_id=document.id,
        rag=rag,
    )
    assert item["status"] == "succeeded"
    assert complete_build_calls == 1  # processing owner only; API confirms read-only
    assert plan.sidecar_artifact is not None
    assert plan.blocks_artifact is not None
    assert plan.sidecar_artifact.id == old_sidecar_id
    assert plan.blocks_artifact.id == old_blocks_id
    after_build = await metadata_store.get_document("kb_build_move", document.id)
    assert plan.claim_token
    assert plan.claim_token != build_job.id
    assert after_build.metadata["current_build_generation_id"] == plan.claim_token
    assert after_build.metadata["current_sidecar_artifact_id"] == old_sidecar_id
    assert after_build.metadata["current_blocks_artifact_id"] == old_blocks_id
    assert len(storage.file_uploads) == file_upload_count
    assert len(storage.prefix_uploads) == prefix_upload_count
    assert len(storage.file_downloads) > file_download_count
    assert rag.scratch_entries_at_enqueue == [()]
    assert api_preflight_sidecars
    assert all(not sidecar.exists() for sidecar in api_preflight_sidecars)
    payload = kb_document_routes._build_plan_payload(plan)
    persisted_build_job = await job_service.get_job("kb_build_move", build_job.id)
    for durable_payload in (
        payload,
        persisted_build_job.payload,
        persisted_build_job.result,
    ):
        serialized = json.dumps(durable_payload, default=str)
        assert "sidecar_uri" not in serialized
        assert "blocks_path" not in serialized
        assert "file://" not in serialized
        assert ".lightrag-scratch" not in serialized
    runtime_sidecar = rag.runtime_sidecars[-1]
    assert runtime_sidecar not in api_preflight_sidecars
    assert runtime_sidecar.is_relative_to(materializer.scratch_root)
    assert not runtime_sidecar.exists()
    assert not list(materializer.scratch_root.iterdir())
    _assert_no_scratch(rag.full_docs.records)
    _assert_no_scratch(rag.doc_status.rows)
    _assert_no_scratch(rag.enqueue_payloads)
    assert plan.document.lightrag_doc_id is not None
    full_doc = rag.full_docs.records[plan.document.lightrag_doc_id]
    assert "sidecar_location" not in full_doc
    committed_binding = PipelineArtifactBinding.from_mapping(
        full_doc["artifact_binding"]
    )
    assert committed_binding.state == "committed"
    assert committed_binding.claim_token == plan.claim_token
    _assert_no_scratch(asdict(plan))

    downloads_before_skip = (
        len(storage.file_downloads),
        len(storage.prefix_downloads),
    )
    skipped_plan, _skipped_job, skipped_item = await _execute_build(
        index_service,
        job_service,
        kb_id="kb_build_move",
        document_id=document.id,
        rag=rag,
    )
    assert skipped_plan.skipped is True
    assert skipped_item["status"] == "succeeded"
    assert (
        complete_build_calls == 2
    )  # skipped object plan is metadata-only API complete
    assert (
        len(storage.file_downloads),
        len(storage.prefix_downloads),
    ) == downloads_before_skip
    assert not list(materializer.scratch_root.iterdir())

    sidecar_prefix = plan.sidecar_artifact.object_prefix_uri
    assert sidecar_prefix is not None
    storage.prefixes.pop(sidecar_prefix)
    deletes_before = list(rag.delete_calls)
    _failed_plan, _failed_job, failed_item = await _execute_build(
        index_service,
        job_service,
        kb_id="kb_build_move",
        document_id=document.id,
        rag=rag,
        force=True,
    )
    assert failed_item["status"] == "failed"
    assert rag.delete_calls == deletes_before
    _assert_no_scratch(failed_item)


async def test_busy_drain_holds_lease_until_terminal_and_scrubs_full_docs(tmp_path):
    (
        kb_service,
        metadata_store,
        storage,
        document,
        _parse_materializer,
        job_service,
        rag,
    ) = await _setup_parsed_object_document(tmp_path, kb_id="kb_busy")
    object_service, build_materializer = _build_object_service(
        root=tmp_path / "busy-build" / "inputs",
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
    )
    index_service = IndexBuildService(object_service)
    await _attach_processing_owner(
        rag,
        kb_id="kb_busy",
        kb_service=kb_service,
        document_service=object_service,
        index_service=index_service,
        materializer=build_materializer,
    )
    rag.drain_mode = "owner"
    task = asyncio.create_task(
        _execute_build(
            index_service,
            job_service,
            kb_id="kb_busy",
            document_id=document.id,
            rag=rag,
        )
    )
    await asyncio.wait_for(rag.drain_returned.wait(), timeout=5)
    runtime_sidecar = rag.runtime_sidecars[-1]
    assert runtime_sidecar.is_dir()
    assert runtime_sidecar.is_relative_to(build_materializer.scratch_root)
    assert rag.scratch_entries_at_enqueue == [()]
    claimed_full_doc = rag.full_docs.records[next(iter(rag._enqueued))]
    assert "sidecar_location" not in claimed_full_doc
    assert (
        PipelineArtifactBinding.from_mapping(claimed_full_doc["artifact_binding"]).state
        == "claimed"
    )
    rag.release_owner.set()
    _plan, _job, item = await asyncio.wait_for(task, timeout=5)
    assert item["status"] == "succeeded"
    assert not runtime_sidecar.exists()
    assert not list(build_materializer.scratch_root.iterdir())
    _assert_no_scratch(rag.full_docs.records)
    _assert_no_scratch(rag.doc_status.rows)


async def test_batch_busy_drain_has_no_api_execution_map_or_preflight_lease(
    tmp_path, monkeypatch
):
    (
        kb_service,
        metadata_store,
        storage,
        document,
        _parse_materializer,
        job_service,
        rag,
    ) = await _setup_parsed_object_document(tmp_path, kb_id="kb_batch_busy")
    service, materializer = _build_object_service(
        root=tmp_path / "batch-busy-build" / "inputs",
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
    )
    index_service = IndexBuildService(service)
    await _attach_processing_owner(
        rag,
        kb_id="kb_batch_busy",
        kb_service=kb_service,
        document_service=service,
        index_service=index_service,
        materializer=materializer,
    )
    seen_execution_maps: list[dict[str, Any]] = []
    run_build_batch = index_service.run_build_batch

    async def observe_execution_map(rag_arg, plans_arg, executions_arg, *, job_id=None):
        seen_execution_maps.append(dict(executions_arg))
        return await run_build_batch(rag_arg, plans_arg, executions_arg, job_id=job_id)

    monkeypatch.setattr(index_service, "run_build_batch", observe_execution_map)
    index_service._build_drain_timeout = 5.0
    index_service._build_drain_poll = 0.01
    second_document, _upload_job = await _create_document(
        service,
        "kb_batch_busy",
        source_name="report-2.pdf",
        content=b"second-pdf-bytes",
    )
    _parse_plan, _parse_job, parse_item = await _execute_one_parse(
        service,
        job_service,
        kb_id="kb_batch_busy",
        document_id=second_document.id,
        rag=rag,
    )
    assert parse_item["status"] == "succeeded"
    plans = [
        await index_service.create_build_plan("kb_batch_busy", item.id, rag=rag)
        for item in (document, second_document)
    ]
    batch_job, _created = await job_service.create_batch_build_job_once(
        "kb_batch_busy",
        batch_id="batch-busy",
        document_ids=[plan.document.id for plan in plans],
        total_items=len(plans),
        plan_items=[kb_document_routes._build_plan_payload(plan) for plan in plans],
        planning_failures=[],
    )
    claimed, failures = await index_service.claim_batch_build_queued(
        "kb_batch_busy", job_id=batch_job.id, plans=plans
    )
    assert len(claimed) == len(plans) and failures == []
    await job_service.transition_job(
        "kb_batch_busy", batch_job.id, status="running", progress=0.1
    )
    rag.drain_mode = "owner"
    task = asyncio.create_task(
        kb_document_routes._execute_build_plan_batch(
            index_service=index_service,
            kb_id="kb_batch_busy",
            job_id=batch_job.id,
            rag=rag,
            plans=plans,
            job_service=job_service,
        )
    )
    await asyncio.wait_for(rag.drain_returned.wait(), timeout=5)
    runtime_sidecars = rag.runtime_sidecars[-len(plans) :]
    assert all(sidecar.is_dir() for sidecar in runtime_sidecars)
    assert all(
        sidecar.is_relative_to(materializer.scratch_root)
        for sidecar in runtime_sidecars
    )
    assert task.done() is False
    assert seen_execution_maps == [{}]
    assert rag.scratch_entries_at_enqueue == [()]
    assert all(plan.object_preflight_complete for plan in plans)
    assert len(materializer._deferred_leases) == 0
    assert len(list(materializer.scratch_root.iterdir())) == len(plans)
    for plan in plans:
        in_flight = await metadata_store.get_document("kb_batch_busy", plan.document.id)
        assert in_flight.status == "building"
        assert in_flight.metadata["current_build_claim_token"] == plan.claim_token
    rag.release_owner.set()
    results = await asyncio.wait_for(task, timeout=5)
    assert set(results) == {plan.document.id for plan in plans}
    assert all(item["status"] == "succeeded" for item in results.values())
    await job_service.transition_job(
        "kb_batch_busy",
        batch_job.id,
        status="succeeded",
        progress=1.0,
        completed_items=len(plans),
        result={"items": list(results.values())},
    )
    persisted_batch_job = await job_service.get_job("kb_batch_busy", batch_job.id)
    for durable_payload in (
        persisted_batch_job.payload,
        persisted_batch_job.result,
    ):
        serialized = json.dumps(durable_payload, default=str)
        assert "sidecar_uri" not in serialized
        assert "blocks_path" not in serialized
        assert "file://" not in serialized
        assert ".lightrag-scratch" not in serialized
    assert all(not sidecar.exists() for sidecar in runtime_sidecars)
    assert not list(materializer.scratch_root.iterdir())
    _assert_no_scratch(rag.full_docs.records)
    _assert_no_scratch(rag.doc_status.rows)
    _assert_no_scratch(rag.enqueue_payloads)


async def test_object_build_busy_timeout_keeps_owner_until_terminal_and_allows_retry(
    tmp_path,
):
    (
        kb_service,
        metadata_store,
        storage,
        document,
        _parse_materializer,
        job_service,
        rag,
    ) = await _setup_parsed_object_document(tmp_path, kb_id="kb_timeout")
    service, materializer = _build_object_service(
        root=tmp_path / "timeout-build" / "inputs",
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
    )
    index_service = IndexBuildService(service)
    await _attach_processing_owner(
        rag,
        kb_id="kb_timeout",
        kb_service=kb_service,
        document_service=service,
        index_service=index_service,
        materializer=materializer,
    )
    index_service._build_drain_timeout = 0.05
    index_service._build_drain_poll = 0.01
    plan = await index_service.create_build_plan("kb_timeout", document.id, rag=rag)
    job, _created = await job_service.create_build_job_once(
        "kb_timeout",
        document_id=document.id,
        parser_hash=plan.parser_hash,
        index_hash=plan.index_hash,
        source_hash=plan.document.source_hash,
        lightrag_doc_id=plan.document.lightrag_doc_id or "",
        sidecar_artifact_id=plan.sidecar_artifact.id if plan.sidecar_artifact else None,
        blocks_artifact_id=plan.blocks_artifact.id if plan.blocks_artifact else None,
    )
    await index_service.claim_build_queued("kb_timeout", job_id=job.id, plan=plan)
    first_token = plan.claim_token
    assert first_token
    await job_service.transition_job(
        "kb_timeout", job.id, status="running", progress=0.1
    )
    rag.drain_mode = "timeout"
    task = asyncio.create_task(
        kb_document_routes._execute_build_plan(
            index_service=index_service,
            kb_id="kb_timeout",
            job_id=job.id,
            plan=plan,
            rag=rag,
            job_service=job_service,
        )
    )
    await asyncio.wait_for(rag.drain_returned.wait(), timeout=5)
    runtime_sidecar = rag.runtime_sidecars[-1]
    await asyncio.sleep(0.1)
    assert task.done() is False
    assert runtime_sidecar.is_dir()
    assert rag.scratch_entries_at_enqueue == [()]
    assert len(materializer._deferred_leases) == 0
    assert len(list(materializer.scratch_root.iterdir())) == 1
    in_flight = await metadata_store.get_document("kb_timeout", document.id)
    assert in_flight.status == "building"
    assert in_flight.metadata["current_build_job_id"] == job.id
    assert in_flight.metadata["current_build_claim_token"] == first_token
    retry_plan = await index_service.create_build_plan(
        "kb_timeout",
        document.id,
        rag=rag,
        force_rechunk=True,
        force_extract=True,
        force_embedding=True,
    )
    with pytest.raises(ActiveDocumentBuildJobError):
        await index_service.claim_build_queued(
            "kb_timeout", job_id="job-retry-too-early", plan=retry_plan
        )

    rag.release_owner.set()
    await asyncio.wait_for(rag.wait_for_processing_owner(), timeout=5)
    item = await asyncio.wait_for(task, timeout=5)
    assert item["status"] == "succeeded"
    await job_service.transition_job(
        "kb_timeout",
        job.id,
        status="succeeded",
        progress=1.0,
        completed_items=1,
        result={"items": [item]},
    )
    assert not runtime_sidecar.exists()
    assert len(materializer._deferred_leases) == 0
    assert not list(materializer.scratch_root.iterdir())
    _assert_no_scratch(rag.full_docs.records)
    _assert_no_scratch(rag.doc_status.rows)

    rag.drain_mode = "self"
    rag.drain_returned = asyncio.Event()
    rag.kb_active_index_hash = "sha256:phase2b-index-retry"
    post_terminal_plan = await index_service.create_build_plan(
        "kb_timeout",
        document.id,
        rag=rag,
        force_rechunk=True,
        force_extract=True,
        force_embedding=True,
    )
    await index_service.claim_build_queued(
        "kb_timeout", job_id="job-after-terminal", plan=post_terminal_plan
    )
    assert post_terminal_plan.claim_token
    assert post_terminal_plan.claim_token != first_token
    await index_service.release_build_if_owned(
        "kb_timeout",
        document.id,
        job_id="job-after-terminal",
        plan=post_terminal_plan,
        error_code="test_cleanup",
        error_message="test cleanup",
    )


async def test_object_build_cancel_waits_for_busy_owner_before_release(
    tmp_path, monkeypatch
):
    (
        kb_service,
        metadata_store,
        storage,
        document,
        _parse_materializer,
        job_service,
        rag,
    ) = await _setup_parsed_object_document(tmp_path, kb_id="kb_build_cancel")
    service, materializer = _build_object_service(
        root=tmp_path / "cancel-build" / "inputs",
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
    )
    index_service = IndexBuildService(service)
    await _attach_processing_owner(
        rag,
        kb_id="kb_build_cancel",
        kb_service=kb_service,
        document_service=service,
        index_service=index_service,
        materializer=materializer,
    )
    api_release_calls = 0
    release_build_if_owned = index_service.release_build_if_owned

    async def observe_api_release(*args, **kwargs):
        nonlocal api_release_calls
        api_release_calls += 1
        return await release_build_if_owned(*args, **kwargs)

    monkeypatch.setattr(index_service, "release_build_if_owned", observe_api_release)
    index_service._build_drain_timeout = 5.0
    index_service._build_drain_poll = 0.01
    plan = await index_service.create_build_plan(
        "kb_build_cancel", document.id, rag=rag
    )
    job, _created = await job_service.create_build_job_once(
        "kb_build_cancel",
        document_id=document.id,
        parser_hash=plan.parser_hash,
        index_hash=plan.index_hash,
        source_hash=plan.document.source_hash,
        lightrag_doc_id=plan.document.lightrag_doc_id or "",
        sidecar_artifact_id=plan.sidecar_artifact.id if plan.sidecar_artifact else None,
        blocks_artifact_id=plan.blocks_artifact.id if plan.blocks_artifact else None,
    )
    await index_service.claim_build_queued("kb_build_cancel", job_id=job.id, plan=plan)
    await job_service.transition_job(
        "kb_build_cancel", job.id, status="running", progress=0.1
    )
    rag.drain_mode = "owner"
    task = asyncio.create_task(
        kb_document_routes._execute_build_plan(
            index_service=index_service,
            kb_id="kb_build_cancel",
            job_id=job.id,
            plan=plan,
            rag=rag,
            job_service=job_service,
        )
    )
    await asyncio.wait_for(rag.drain_returned.wait(), timeout=5)
    runtime_sidecar = rag.runtime_sidecars[-1]
    await job_service.transition_job("kb_build_cancel", job.id, status="cancelling")
    await asyncio.sleep(0.1)
    assert task.done() is False
    assert runtime_sidecar.is_dir()
    assert rag.scratch_entries_at_enqueue == [()]
    assert len(materializer._deferred_leases) == 0
    in_flight = await metadata_store.get_document("kb_build_cancel", document.id)
    assert in_flight.status == "building"
    assert in_flight.metadata["current_build_claim_token"] == plan.claim_token

    rag.cancel_processing_owner = True
    rag.release_owner.set()
    item = await asyncio.wait_for(task, timeout=5)
    assert item["status"] == "cancelled"
    await job_service.transition_job(
        "kb_build_cancel", job.id, status="cancelled", progress=1.0
    )
    cancelled_document = await metadata_store.get_document(
        "kb_build_cancel", document.id
    )
    assert cancelled_document.status == "build_failed"
    assert cancelled_document.metadata.get("pending_build_job_id") is None
    assert cancelled_document.metadata.get("pending_build_claim_token") is None
    assert cancelled_document.metadata.get("current_build_job_id") is None
    assert cancelled_document.metadata.get("current_build_claim_token") is None
    assert api_release_calls == 0
    assert not runtime_sidecar.exists()
    assert len(materializer._deferred_leases) == 0
    assert not list(materializer.scratch_root.iterdir())
    _assert_no_scratch(item)
    _assert_no_scratch(rag.doc_status.rows)


async def test_multimodal_mutation_promotes_immutable_generation_and_compensates_commit_failure(
    tmp_path, monkeypatch
):
    (
        kb_service,
        metadata_store,
        storage,
        document,
        _parse_materializer,
        job_service,
        rag,
    ) = await _setup_parsed_object_document(tmp_path, kb_id="kb_promote")
    service, materializer = _build_object_service(
        root=tmp_path / "promote-build" / "inputs",
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
    )
    index_service = IndexBuildService(service)
    await _attach_processing_owner(
        rag,
        kb_id="kb_promote",
        kb_service=kb_service,
        document_service=service,
        index_service=index_service,
        materializer=materializer,
    )
    before_document = await metadata_store.get_document("kb_promote", document.id)
    old_sidecar_id = before_document.metadata["current_sidecar_artifact_id"]
    old_blocks_id = before_document.metadata["current_blocks_artifact_id"]
    old_files = set(storage.files)
    old_prefixes = set(storage.prefixes)
    rag.mutate = True
    _plan, _job, item = await _execute_build(
        index_service,
        job_service,
        kb_id="kb_promote",
        document_id=document.id,
        rag=rag,
    )
    assert item["status"] == "succeeded"
    promoted_document = await metadata_store.get_document("kb_promote", document.id)
    new_sidecar_id = promoted_document.metadata["current_sidecar_artifact_id"]
    new_blocks_id = promoted_document.metadata["current_blocks_artifact_id"]
    assert new_sidecar_id != old_sidecar_id
    assert new_blocks_id != old_blocks_id
    assert old_files <= set(storage.files)
    assert old_prefixes <= set(storage.prefixes)
    new_sidecar = await metadata_store.get_document_artifact(
        "kb_promote", document.id, new_sidecar_id
    )
    new_blocks = await metadata_store.get_document_artifact(
        "kb_promote", document.id, new_blocks_id
    )
    _assert_no_scratch(asdict(new_sidecar))
    _assert_no_scratch(asdict(new_blocks))
    prefix_uri = new_sidecar.metadata["object_prefix_uri"]
    object_uri = new_blocks.metadata["object_uri"]
    blocks_name = Path(new_blocks.uri).name
    promoted_sidecar_objects = {
        uri: payload
        for uri, payload in storage.files.items()
        if uri.startswith(prefix_uri)
    }
    assert promoted_sidecar_objects
    assert (
        promoted_sidecar_objects[f"{prefix_uri}{blocks_name}"]
        == storage.files[object_uri]
    )
    assert not list(materializer.scratch_root.iterdir())

    pointers_before_failure = (
        new_sidecar_id,
        new_blocks_id,
    )
    committed_files = set(storage.files)
    committed_prefixes = set(storage.prefixes)
    upload_file_index = len(storage.file_uploads)
    upload_prefix_index = len(storage.prefix_uploads)

    async def fail_commit(*_args, **_kwargs):
        raise RuntimeError("injected build promotion commit failure")

    api_fail_calls = 0
    fail_build = index_service.fail_build

    async def observe_api_fail(*args, **kwargs):
        nonlocal api_fail_calls
        api_fail_calls += 1
        return await fail_build(*args, **kwargs)

    monkeypatch.setattr(
        metadata_store,
        "complete_document_build_with_artifact_promotion",
        fail_commit,
    )
    monkeypatch.setattr(index_service, "fail_build", observe_api_fail)
    rag.kb_active_index_hash = "sha256:phase2b-index-failure"
    _plan2, _job2, failed_item = await _execute_build(
        index_service,
        job_service,
        kb_id="kb_promote",
        document_id=document.id,
        rag=rag,
        force=True,
    )
    assert failed_item["status"] == "failed"
    assert api_fail_calls == 0
    after_failure = await metadata_store.get_document("kb_promote", document.id)
    assert (
        after_failure.metadata["current_sidecar_artifact_id"],
        after_failure.metadata["current_blocks_artifact_id"],
    ) == pointers_before_failure
    new_file_uploads = storage.file_uploads[upload_file_index:]
    new_prefix_uploads = storage.prefix_uploads[upload_prefix_index:]
    assert new_file_uploads
    assert new_prefix_uploads == []
    assert all(uri not in storage.files for uri in new_file_uploads)
    assert all(uri not in storage.prefixes for uri in new_prefix_uploads)
    assert committed_files <= set(storage.files)
    assert committed_prefixes <= set(storage.prefixes)
    _assert_no_scratch(rag.full_docs.records)
    _assert_no_scratch(rag.doc_status.rows)
    _assert_no_scratch(asdict(after_failure))


async def test_artifact_pointer_cas_loser_compensates_only_loser_objects(
    tmp_path, monkeypatch
):
    (
        kb_service,
        metadata_store,
        storage,
        document,
        _parse_materializer,
        job_service,
        _rag,
    ) = await _setup_parsed_object_document(tmp_path, kb_id="kb_cas")
    service, materializer = _build_object_service(
        root=tmp_path / "cas-build" / "inputs",
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
    )
    index_service = IndexBuildService(service)
    rag = _BuildRAG(mutate=True)
    await _attach_processing_owner(
        rag,
        kb_id="kb_cas",
        kb_service=kb_service,
        document_service=service,
        index_service=index_service,
        materializer=materializer,
    )
    winner_sidecar_id = "artifact_sidecar-winner"
    winner_blocks_id = "artifact_blocks-winner"
    winner_prefix = (
        f"s3://phase2a/workspaces/{document.workspace}/documents/"
        f"{document.id}/artifacts/sidecar/{winner_sidecar_id}/winner.parsed/"
    )
    winner_object = (
        f"s3://phase2a/workspaces/{document.workspace}/documents/"
        f"{document.id}/artifacts/blocks/{winner_blocks_id}/winner.blocks.jsonl"
    )
    storage.prefixes[winner_prefix] = {"winner.blocks.jsonl": b"winner"}
    storage.files[winner_object] = b"winner"
    original_complete = metadata_store.complete_document_build_with_artifact_promotion
    winner_installed = False

    async def install_winner_before_loser_cas(*args, **kwargs):
        nonlocal winner_installed
        if not winner_installed:
            candidate_artifacts = kwargs["artifacts"]
            candidate_sidecar = next(
                artifact
                for artifact in candidate_artifacts
                if artifact.artifact_type == "sidecar"
            )
            candidate_blocks = next(
                artifact
                for artifact in candidate_artifacts
                if artifact.artifact_type == "blocks"
            )
            now = utc_now_iso()
            winner_artifacts = [
                ArtifactRecord(
                    id=winner_sidecar_id,
                    kb_id="kb_cas",
                    workspace=document.workspace,
                    document_id=document.id,
                    artifact_type="sidecar",
                    uri=candidate_sidecar.uri,
                    checksum="sha256:winner-sidecar",
                    size_bytes=None,
                    metadata={
                        "is_directory": True,
                        "object_prefix_uri": winner_prefix,
                        "blocks_path": candidate_blocks.uri,
                    },
                    created_at=now,
                ),
                ArtifactRecord(
                    id=winner_blocks_id,
                    kb_id="kb_cas",
                    workspace=document.workspace,
                    document_id=document.id,
                    artifact_type="blocks",
                    uri=candidate_blocks.uri,
                    checksum="sha256:winner-blocks",
                    size_bytes=6,
                    metadata={"object_uri": winner_object},
                    created_at=now,
                ),
            ]

            def install_winner_pointers(conn) -> None:
                for artifact in winner_artifacts:
                    metadata_store._insert_artifact(conn, artifact)
                row = conn.execute(
                    "SELECT metadata_json FROM documents WHERE kb_id = ? AND id = ?",
                    ("kb_cas", document.id),
                ).fetchone()
                metadata = _loads_json_object(row["metadata_json"])
                metadata["current_sidecar_artifact_id"] = winner_sidecar_id
                metadata["current_blocks_artifact_id"] = winner_blocks_id
                conn.execute(
                    "UPDATE documents SET metadata_json = ? WHERE kb_id = ? AND id = ?",
                    (_dumps_json(metadata), "kb_cas", document.id),
                )

            await metadata_store._write(install_winner_pointers)
            winner_installed = True
        return await original_complete(*args, **kwargs)

    monkeypatch.setattr(
        metadata_store,
        "complete_document_build_with_artifact_promotion",
        install_winner_before_loser_cas,
    )
    upload_file_index = len(storage.file_uploads)
    upload_prefix_index = len(storage.prefix_uploads)
    plan, _job, item = await _execute_build(
        index_service,
        job_service,
        kb_id="kb_cas",
        document_id=document.id,
        rag=rag,
        force=True,
    )
    assert item["status"] == "failed"
    assert winner_installed is True
    loser_files = storage.file_uploads[upload_file_index:]
    loser_prefixes = storage.prefix_uploads[upload_prefix_index:]
    assert loser_files
    assert loser_prefixes == []
    assert all(uri not in storage.files for uri in loser_files)
    assert all(uri not in storage.prefixes for uri in loser_prefixes)
    assert winner_object in storage.files
    assert winner_prefix in storage.prefixes
    winner_document = await metadata_store.get_document("kb_cas", document.id)
    assert winner_document.metadata["current_sidecar_artifact_id"] == winner_sidecar_id
    assert winner_document.metadata["current_blocks_artifact_id"] == winner_blocks_id
    assert winner_document.metadata.get("current_build_job_id") is None
    assert winner_document.metadata.get("current_build_claim_token") is None
    assert plan.claim_token
    _assert_no_scratch(asdict(winner_document))
    _assert_no_scratch(rag.full_docs.records)
    _assert_no_scratch(rag.doc_status.rows)
    assert not list(materializer.scratch_root.iterdir())


async def test_response_lifetime_presign_manifest_and_old_generation_ignore_local_uri(
    tmp_path, monkeypatch
):
    (
        kb_service,
        metadata_store,
        storage,
        document,
        _parse_materializer,
        job_service,
        rag,
    ) = await _setup_parsed_object_document(tmp_path, kb_id="kb_response")
    old_previews, _total = await metadata_store.list_document_artifacts(
        "kb_response", document.id, artifact_type="preview_text", limit=1
    )
    old_preview = old_previews[0]
    await _execute_one_parse(
        _build_object_service(
            root=tmp_path / "reparse-root" / "inputs",
            kb_service=kb_service,
            metadata_store=metadata_store,
            storage=storage,
        )[0],
        job_service,
        kb_id="kb_response",
        document_id=document.id,
        rag=rag,
        force_reparse=True,
    )
    service, materializer = _build_object_service(
        root=tmp_path / "response-root" / "inputs",
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
    )
    manifest = await service.get_document_preview_manifest("kb_response", document.id)
    latest_previews, _total = await metadata_store.list_document_artifacts(
        "kb_response", document.id, artifact_type="preview_text", limit=1
    )
    assert manifest["preferred"]["artifact_id"] == latest_previews[0].id
    url_result = await service.get_document_artifact_download_url(
        "kb_response", document.id, old_preview.id
    )
    assert url_result.url.startswith("https://objects.invalid/")

    artifact_file = await service.get_document_artifact_file(
        "kb_response", document.id, old_preview.id
    )
    response = kb_document_routes._artifact_preview_response(artifact_file)
    response_path = artifact_file.path
    assert response_path.is_file()
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await response(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/artifact",
            "raw_path": b"/artifact",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1),
            "server": ("test", 80),
        },
        receive,
        send,
    )
    assert sent
    assert not response_path.exists()

    download_file = await service.get_document_artifact_file(
        "kb_response", document.id, old_preview.id
    )
    download_path = download_file.path
    download_response = kb_document_routes._artifact_download_file_response(
        download_file
    )
    assert download_path.is_file()
    await download_response(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/artifact-download",
            "raw_path": b"/artifact-download",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1),
            "server": ("test", 80),
        },
        receive,
        send,
    )
    assert not download_path.exists()

    parsed_document = await metadata_store.get_document("kb_response", document.id)
    sidecar = await metadata_store.get_document_artifact(
        "kb_response",
        document.id,
        parsed_document.metadata["current_sidecar_artifact_id"],
    )
    directory_file = await service.get_document_artifact_file(
        "kb_response", document.id, sidecar.id
    )
    directory_path = directory_file.path
    zip_response = kb_document_routes._stream_directory_as_zip(directory_file)
    assert zip_response.status_code == 200
    assert not directory_path.exists()

    monkeypatch.setattr(kb_document_routes, "_MAX_DIRECTORY_ARTIFACT_BYTES", 1)
    oversized = await service.get_document_artifact_file(
        "kb_response", document.id, sidecar.id
    )
    oversized_path = oversized.path
    with pytest.raises(Exception) as too_large:
        kb_document_routes._stream_directory_as_zip(oversized)
    assert getattr(too_large.value, "status_code", None) == 413
    assert not oversized_path.exists()

    monkeypatch.setattr(
        kb_document_routes,
        "_MAX_DIRECTORY_ARTIFACT_BYTES",
        512 * 1024 * 1024,
    )
    broken = await service.get_document_artifact_file(
        "kb_response", document.id, sidecar.id
    )
    broken_path = broken.path

    def fail_zip_write(*_args, **_kwargs):
        raise RuntimeError("injected zip failure")

    monkeypatch.setattr(kb_document_routes.zipfile.ZipFile, "write", fail_zip_write)
    with pytest.raises(RuntimeError, match="injected zip failure"):
        kb_document_routes._stream_directory_as_zip(broken)
    assert not broken_path.exists()
    assert not list(materializer.scratch_root.iterdir())


async def test_current_pointer_exact_then_legacy_newest_per_type_over_200_rows(
    tmp_path,
):
    (
        kb_service,
        metadata_store,
        storage,
        document,
        _materializer,
        _job_service,
        rag,
    ) = await _setup_parsed_object_document(tmp_path, kb_id="kb_history")
    current = await metadata_store.get_document("kb_history", document.id)
    pointed_sidecar = current.metadata["current_sidecar_artifact_id"]
    pointed_blocks = current.metadata["current_blocks_artifact_id"]
    newer_sidecar_id = "artifact_sidecar-newest-legacy"
    newer_blocks_id = "artifact_blocks-newest-legacy"
    prefix_uri = (
        f"s3://phase2a/workspaces/{current.workspace}/documents/{document.id}/"
        f"artifacts/sidecar/{newer_sidecar_id}/newest.parsed/"
    )
    object_uri = (
        f"s3://phase2a/workspaces/{current.workspace}/documents/{document.id}/"
        f"artifacts/blocks/{newer_blocks_id}/newest.blocks.jsonl"
    )
    storage.prefixes[prefix_uri] = {"newest.blocks.jsonl": b"newest"}
    storage.files[object_uri] = b"newest"

    def seed(conn) -> None:
        metadata_store._insert_artifact(
            conn,
            ArtifactRecord(
                id=newer_sidecar_id,
                kb_id="kb_history",
                workspace=current.workspace,
                document_id=document.id,
                artifact_type="sidecar",
                uri="/obsolete/checkout/newest.parsed",
                checksum=None,
                size_bytes=None,
                metadata={
                    "is_directory": True,
                    "object_prefix_uri": prefix_uri,
                    "blocks_path": "/obsolete/checkout/newest.blocks.jsonl",
                },
                created_at="2027-01-02T00:00:00+00:00",
            ),
        )
        metadata_store._insert_artifact(
            conn,
            ArtifactRecord(
                id=newer_blocks_id,
                kb_id="kb_history",
                workspace=current.workspace,
                document_id=document.id,
                artifact_type="blocks",
                uri="/obsolete/checkout/newest.blocks.jsonl",
                checksum=None,
                size_bytes=6,
                metadata={"object_uri": object_uri},
                created_at="2027-01-02T00:00:00+00:00",
            ),
        )
        for index in range(205):
            metadata_store._insert_artifact(
                conn,
                ArtifactRecord(
                    id=f"artifact_history-{index:03d}",
                    kb_id="kb_history",
                    workspace=current.workspace,
                    document_id=document.id,
                    artifact_type=f"history_{index:03d}",
                    uri=f"/obsolete/history/{index}",
                    checksum=None,
                    size_bytes=None,
                    metadata={},
                    created_at=f"2028-01-01T00:{index // 60:02d}:{index % 60:02d}+00:00",
                ),
            )

    await metadata_store._write(seed)
    service, _build_materializer = _build_object_service(
        root=tmp_path / "history-build" / "inputs",
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
    )
    index_service = IndexBuildService(service)
    pointer_plan = await index_service.create_build_plan(
        "kb_history", document.id, rag=rag
    )
    assert pointer_plan.sidecar_artifact is not None
    assert pointer_plan.blocks_artifact is not None
    assert pointer_plan.sidecar_artifact.id == pointed_sidecar
    assert pointer_plan.blocks_artifact.id == pointed_blocks

    def clear_pointers(conn) -> None:
        row = conn.execute(
            "SELECT metadata_json FROM documents WHERE kb_id = ? AND id = ?",
            ("kb_history", document.id),
        ).fetchone()
        metadata = _loads_json_object(row["metadata_json"])
        metadata.pop("current_sidecar_artifact_id", None)
        metadata.pop("current_blocks_artifact_id", None)
        conn.execute(
            "UPDATE documents SET metadata_json = ? WHERE kb_id = ? AND id = ?",
            (_dumps_json(metadata), "kb_history", document.id),
        )

    await metadata_store._write(clear_pointers)
    legacy_plan = await index_service.create_build_plan(
        "kb_history", document.id, rag=rag
    )
    assert legacy_plan.sidecar_artifact is not None
    assert legacy_plan.blocks_artifact is not None
    assert legacy_plan.sidecar_artifact.id == newer_sidecar_id
    assert legacy_plan.blocks_artifact.id == newer_blocks_id


async def test_postgres_pointer_promotion_contract_uses_cas_without_live_backend(
    monkeypatch,
):
    store = PostgresMetadataStore(dsn="postgresql://unused")
    document = SimpleNamespace(
        id="doc-pg",
        metadata={
            "current_sidecar_artifact_id": "sidecar-old",
            "current_blocks_artifact_id": "blocks-old",
        },
        status="building",
        index_hash=None,
        chunks_count=None,
        entity_count=None,
        relation_count=None,
        error_code=None,
        error_message=None,
        updated_at="",
    )
    inserted: list[str] = []
    saved: list[Any] = []

    async def ensure() -> None:
        return None

    async def write(callback):
        return await callback(object())

    async def get_document(_conn, _kb_id, _document_id, *, for_update=False):
        assert for_update is True
        return deepcopy(document)

    async def insert_artifact(_conn, artifact):
        inserted.append(artifact.id)

    async def save_document(_conn, value):
        saved.append(value)

    monkeypatch.setattr(store, "_ensure_initialized", ensure)
    monkeypatch.setattr(store, "_write", write)
    monkeypatch.setattr(store, "_get_document", get_document)
    monkeypatch.setattr(store, "_insert_artifact", insert_artifact)
    monkeypatch.setattr(store, "_save_document", save_document)
    artifact = ArtifactRecord(
        id="sidecar-new",
        kb_id="kb-pg",
        workspace="workspace",
        document_id="doc-pg",
        artifact_type="sidecar",
        uri="/canonical/sidecar",
        checksum="sha256:new",
        size_bytes=None,
        metadata={},
        created_at=utc_now_iso(),
    )
    result, created = await store.complete_document_build_with_artifact_promotion(
        "kb-pg",
        "doc-pg",
        index_hash="sha256:index",
        expected_current_sidecar_artifact_id="sidecar-old",
        expected_current_blocks_artifact_id="blocks-old",
        current_sidecar_artifact_id="sidecar-new",
        current_blocks_artifact_id="blocks-old",
        artifacts=[artifact],
        metadata_patch={"current_build_generation_id": "job-pg"},
    )
    assert created == [artifact]
    assert inserted == [artifact.id]
    assert saved
    assert result.metadata["current_sidecar_artifact_id"] == "sidecar-new"

    async def fail_save(_conn, _value):
        raise RuntimeError("injected postgres commit failure")

    inserted.clear()
    saved.clear()
    monkeypatch.setattr(store, "_save_document", fail_save)
    with pytest.raises(RuntimeError, match="injected postgres commit failure"):
        await store.complete_document_build_with_artifact_promotion(
            "kb-pg",
            "doc-pg",
            index_hash="sha256:index",
            expected_current_sidecar_artifact_id="sidecar-old",
            expected_current_blocks_artifact_id="blocks-old",
            current_sidecar_artifact_id="sidecar-new",
            current_blocks_artifact_id="blocks-old",
            artifacts=[artifact],
            metadata_patch={},
        )
    assert document.metadata["current_sidecar_artifact_id"] == "sidecar-old"

    monkeypatch.setattr(store, "_save_document", save_document)
    document.metadata["current_sidecar_artifact_id"] = "winner"
    inserted.clear()
    saved.clear()
    with pytest.raises(ArtifactPointerConflictError):
        await store.complete_document_build_with_artifact_promotion(
            "kb-pg",
            "doc-pg",
            index_hash="sha256:index",
            expected_current_sidecar_artifact_id="sidecar-old",
            expected_current_blocks_artifact_id="blocks-old",
            current_sidecar_artifact_id="loser",
            current_blocks_artifact_id="blocks-old",
            artifacts=[artifact],
            metadata_patch={},
        )
    assert inserted == []
    assert saved == []
