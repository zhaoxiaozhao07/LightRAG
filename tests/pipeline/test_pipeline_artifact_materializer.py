from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import MISSING
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import pytest

import lightrag.pipeline as pipeline_module
from lightrag import LightRAG
from lightrag.artifact_runtime import (
    PipelineArtifactCommitOutcome,
    PipelineArtifactBinding,
    PipelineArtifactFinalizationResult,
    PipelineTerminalOutcome,
)
from lightrag.base import DocProcessingStatus, DocStatus
from lightrag.constants import (
    FULL_DOCS_FORMAT_LIGHTRAG,
    FULL_DOCS_FORMAT_PENDING_PARSE,
)
from lightrag.exceptions import PipelineCancelledException
from lightrag.pipeline import _BatchRunContext, _PipelineMixin
from lightrag.utils import EmbeddingFunc


pytestmark = pytest.mark.offline


def _status_value(value: Any) -> str:
    return value.value if isinstance(value, DocStatus) else str(value)


class _DurableStore:
    def __init__(self, events: list[tuple[Any, ...]]) -> None:
        self.events = events
        self.records: dict[str, dict[str, Any]] = {}
        self.writes: list[dict[str, dict[str, Any]]] = []

    async def filter_keys(self, keys: set[str]) -> set[str]:
        return set(keys) - set(self.records)

    async def get_by_id(self, key: str) -> dict[str, Any] | None:
        value = self.records.get(key)
        return deepcopy(value) if value is not None else None

    async def get_doc_by_file_basename(
        self, basename: str
    ) -> tuple[str, dict[str, Any]] | None:
        for key, value in self.records.items():
            if value.get("file_path") == basename:
                return key, deepcopy(value)
        return None

    async def get_doc_by_content_hash(
        self, content_hash: str
    ) -> tuple[str, dict[str, Any]] | None:
        for key, value in self.records.items():
            if value.get("content_hash") == content_hash:
                return key, deepcopy(value)
        return None

    async def upsert(self, values: dict[str, dict[str, Any]]) -> None:
        encoded = json.dumps(values, ensure_ascii=False, default=str)
        assert ".lightrag-scratch" not in encoded
        assert "sidecar_location" not in encoded
        assert "blocks_path" not in encoded
        copied = deepcopy(values)
        self.writes.append(copied)
        self.records.update(copied)
        for doc_id, value in copied.items():
            status = _status_value(value.get("status", ""))
            if status in {DocStatus.FAILED.value, DocStatus.PROCESSED.value}:
                self.events.append(("terminal", doc_id, status))

    async def index_done_callback(self) -> None:
        return None


class _RecordingQueue(asyncio.Queue):
    def __init__(self, events: list[tuple[Any, ...]], label: str) -> None:
        super().__init__()
        self.events = events
        self.label = label

    async def put(self, item: Any) -> None:
        await super().put(item)
        self.events.append(("put", self.label))


class _FailingPutQueue(asyncio.Queue):
    def __init__(self, events: list[tuple[Any, ...]]) -> None:
        super().__init__()
        self.events = events

    async def put(self, item: Any) -> None:
        del item
        self.events.append(("put_failed", "analyze"))
        raise RuntimeError("queue handoff failed")


class _RuntimeSession:
    def __init__(
        self,
        *,
        label: str,
        binding: PipelineArtifactBinding,
        events: list[tuple[Any, ...]],
        source_path: Path | None = None,
        sidecar_dir: Path | None = None,
        blocks_path: Path | None = None,
    ) -> None:
        self.label = label
        self.binding = binding
        self.source_path = source_path
        self.sidecar_dir = sidecar_dir
        self.blocks_path = blocks_path
        self.events = events
        self.producer_active = False

    def redact(self, error: object) -> str:
        text = str(error)
        for path in (self.source_path, self.blocks_path, self.sidecar_dir):
            if path is not None:
                text = text.replace(str(path), "<artifact-runtime>")
        return text.replace(".lightrag-scratch", "artifact-runtime")

    def defer_cleanup(self) -> None:
        self.events.append(("defer", self.label))

    async def finish(self, outcome: PipelineTerminalOutcome) -> None:
        self.events.append(("finish", self.label, outcome.value))

    async def handoff_success(
        self,
        *,
        parsed_data: Mapping[str, Any] | None = None,
        chunks_count: int | None = None,
    ) -> PipelineArtifactFinalizationResult:
        self.events.append(("handoff_success", self.label))
        entity_count = (
            parsed_data.get("entity_count")
            if isinstance(parsed_data, Mapping)
            else None
        )
        relation_count = (
            parsed_data.get("relation_count")
            if isinstance(parsed_data, Mapping)
            else None
        )
        return PipelineArtifactFinalizationResult(
            outcome=PipelineArtifactCommitOutcome.COMMITTED,
            committed_binding=self.binding.committed(
                parse_generation_id=(
                    self.binding.claim_token
                    if self.binding.operation == "parse"
                    else self.binding.parse_generation_id
                ),
                index_hash=self.binding.index_hash,
                sidecar_artifact_id=self.binding.sidecar_artifact_id,
                blocks_artifact_id=self.binding.blocks_artifact_id,
                raw_artifact_ids=(
                    self.binding.raw_artifact_ids
                    if self.binding.operation == "parse"
                    else ()
                ),
            ),
            chunks_count=chunks_count,
            entity_count=entity_count,
            relation_count=relation_count,
        )

    async def aclose(self) -> None:
        self.events.append(("close", self.label))


class _TestPipeline(_PipelineMixin):
    def __init__(
        self,
        *,
        workspace: str,
        events: list[tuple[Any, ...]],
    ) -> None:
        self.workspace = workspace
        self.events = events
        self.addon_params: dict[str, Any] = {}
        self.full_docs = _DurableStore(events)
        self.doc_status = _DurableStore(events)
        self.pipeline_artifact_materializer: Any = None
        self.llm_response_cache = None
        self.analyze_error: BaseException | None = None
        self.resolver_calls: list[tuple[str, str | None]] = []

    def _resolve_source_file_for_parser(
        self,
        file_path: str,
        *,
        source_file: str | None = None,
        parser_engine: str | None = None,
    ) -> str:
        del parser_engine
        self.resolver_calls.append((file_path, source_file))
        return file_path

    async def analyze_multimodal(
        self,
        doc_id: str,
        file_path: str,
        parsed_data: dict[str, Any],
        *,
        process_options: str | None = None,
        pipeline_status: dict | None = None,
        pipeline_status_lock: Any | None = None,
    ) -> dict[str, Any]:
        del file_path, process_options, pipeline_status, pipeline_status_lock
        self.events.append(("analyze", doc_id))
        if self.analyze_error is not None:
            self.events.append(("producer_stop", "analyze"))
            raise self.analyze_error
        return {
            **parsed_data,
            "analyzing_stage_skipped": True,
        }


def _binding(
    doc_id: str,
    *,
    operation: Literal["parse", "build"] = "build",
    workspace: str = "workspace-b",
) -> PipelineArtifactBinding:
    return PipelineArtifactBinding(
        version=1,
        authority="kb_metadata",
        state="claimed",
        operation=operation,
        kb_id="kb-a",
        kb_generation="generation-a",
        workspace=workspace,
        document_id=f"metadata-{doc_id}",
        lightrag_doc_id=doc_id,
        job_id=f"job-{doc_id}",
        claim_token=f"claim-{doc_id}",
        source_hash="sha256:source",
        parser_hash="sha256:parser",
        parse_generation_id="parse-generation-a",
        index_hash="sha256:index",
        sidecar_artifact_id="sidecar-a",
        blocks_artifact_id="blocks-a",
        expected_current_sidecar_artifact_id="sidecar-a",
        expected_current_blocks_artifact_id="blocks-a",
        raw_artifact_ids=("raw-a",) if operation == "parse" else (),
    )


def _status_doc(binding: PipelineArtifactBinding, file_path: str) -> DocProcessingStatus:
    now = datetime.now(timezone.utc).isoformat()
    return DocProcessingStatus(
        content_summary="",
        content_length=0,
        file_path=file_path,
        status=DocStatus.PENDING,
        created_at=now,
        updated_at=now,
        track_id="track-a",
        metadata={"pipeline_attempt_token": binding.claim_token},
    )


def _build_full_doc(binding: PipelineArtifactBinding, file_path: str) -> dict[str, Any]:
    return {
        "content": "",
        "file_path": file_path,
        "parse_format": FULL_DOCS_FORMAT_LIGHTRAG,
        "parse_engine": "native",
        "process_options": "!",
        "chunk_options": {
            "chunk_token_size": 128,
            "fixed_token": {"chunk_overlap_token_size": 0},
        },
        "artifact_binding": binding.to_dict(),
    }


def _pending_full_doc(
    binding: PipelineArtifactBinding, file_path: str
) -> dict[str, Any]:
    return {
        "content": "",
        "file_path": file_path,
        "parse_format": FULL_DOCS_FORMAT_PENDING_PARSE,
        "parse_engine": "legacy",
        "process_options": "!",
        "chunk_options": {
            "chunk_token_size": 128,
            "fixed_token": {"chunk_overlap_token_size": 0},
        },
        "artifact_binding": binding.to_dict(),
    }


def _write_sidecar(root: Path, text: str) -> tuple[Path, Path]:
    sidecar = root / ".lightrag-scratch" / "op-runtime" / "doc.parsed"
    sidecar.mkdir(parents=True)
    blocks = sidecar / "doc.blocks.jsonl"
    blocks.write_text(
        "\n".join(
            (
                json.dumps({"type": "meta", "doc_id": "doc-a"}),
                json.dumps({"type": "content", "content": text}),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return sidecar, blocks


def _ctx(events: list[tuple[Any, ...]]) -> _BatchRunContext:
    del events
    return _BatchRunContext(
        pipeline_status={
            "busy": True,
            "history_messages": [],
            "latest_message": "",
            "cancellation_requested": False,
            "cancellation_reason": None,
            "cancellation_detail": None,
        },
        pipeline_status_lock=asyncio.Lock(),
        semaphore=asyncio.Semaphore(2),
        total_files=1,
        q_native=asyncio.Queue(),
        q_mineru=asyncio.Queue(),
        q_docling=asyncio.Queue(),
        q_analyze=asyncio.Queue(),
        q_process=asyncio.Queue(),
    )


async def _run_worker(worker_coro: Any, queue: asyncio.Queue) -> None:
    worker = asyncio.create_task(worker_coro)
    try:
        await asyncio.wait_for(queue.join(), timeout=3)
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


async def _seed_build(
    pipeline: _TestPipeline,
    binding: PipelineArtifactBinding,
    *,
    file_path: str = "doc.pdf",
) -> DocProcessingStatus:
    status_doc = _status_doc(binding, file_path)
    pipeline.full_docs.records[binding.lightrag_doc_id] = _build_full_doc(
        binding, file_path
    )
    pipeline.doc_status.records[binding.lightrag_doc_id] = {
        "status": DocStatus.PENDING,
        "content_summary": "",
        "content_length": 0,
        "file_path": file_path,
        "created_at": status_doc.created_at,
        "updated_at": status_doc.updated_at,
        "track_id": status_doc.track_id,
        "metadata": deepcopy(status_doc.metadata),
    }
    return status_doc


def test_lightrag_materializer_callback_defaults_to_none() -> None:
    dataclass_field = LightRAG.__dataclass_fields__["pipeline_artifact_materializer"]
    assert dataclass_field.default is None
    assert dataclass_field.default_factory is MISSING


def test_runtime_callback_is_not_deepcopied_into_global_config(tmp_path: Path) -> None:
    class NonCopyableMaterializer:
        def __deepcopy__(self, memo: Any) -> Any:
            del memo
            raise AssertionError("runtime callback must not be deep-copied")

        async def __call__(self, binding: PipelineArtifactBinding) -> Any:
            del binding
            raise AssertionError("runtime callback must not be invoked")

    async def embedding(texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 4))

    async def llm(prompt: str, **kwargs: Any) -> str:
        del prompt, kwargs
        return ""

    callback = NonCopyableMaterializer()
    rag = LightRAG(
        working_dir=str(tmp_path / "rag"),
        pipeline_artifact_materializer=callback,
        llm_model_func=llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=4,
            max_token_size=128,
            func=embedding,
        ),
    )

    assert rag.pipeline_artifact_materializer is callback
    assert "pipeline_artifact_materializer" not in rag._build_global_config()


@pytest.mark.asyncio
async def test_callback_missing_fails_closed_without_local_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[Any, ...]] = []
    pipeline = _TestPipeline(workspace="workspace-b", events=events)
    binding = _binding("doc-missing")
    status_doc = await _seed_build(pipeline, binding)
    local_reads: list[str] = []

    async def forbidden_blocks_loader(path: Any) -> tuple[str, str]:
        local_reads.append(str(path))
        raise AssertionError("binding must not read a historical local sidecar")

    async def forbidden_legacy(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        local_reads.append("legacy")
        raise AssertionError("binding must not fall back to the local parser path")

    monkeypatch.setattr(
        pipeline_module,
        "load_lightrag_document_content_from_blocks_path",
        forbidden_blocks_loader,
    )
    monkeypatch.setattr(pipeline, "parse_legacy", forbidden_legacy)
    ctx = _ctx(events)
    await ctx.q_native.put((binding.lightrag_doc_id, status_doc, None))
    await _run_worker(pipeline._parse_worker("legacy", ctx.q_native, ctx), ctx.q_native)

    assert local_reads == []
    terminal = pipeline.doc_status.records[binding.lightrag_doc_id]
    assert _status_value(terminal["status"]) == DocStatus.FAILED.value
    assert "artifact_materializer_required" in terminal["error_msg"]
    assert ctx.active_sessions == {}
    assert not tmp_path.joinpath("unused").exists()


@pytest.mark.asyncio
async def test_materializer_failure_is_redacted_before_terminal(
    tmp_path: Path,
) -> None:
    events: list[tuple[Any, ...]] = []
    pipeline = _TestPipeline(workspace="workspace-b", events=events)
    binding = _binding("doc-materialize-error")
    status_doc = await _seed_build(pipeline, binding)
    leaked_path = tmp_path / ".lightrag-scratch" / "secret-operation" / "source"

    async def materialize(received: PipelineArtifactBinding) -> _RuntimeSession:
        del received
        raise RuntimeError(f"download failed at {leaked_path}")

    pipeline.pipeline_artifact_materializer = materialize
    ctx = _ctx(events)
    await ctx.q_native.put((binding.lightrag_doc_id, status_doc, None))
    await _run_worker(pipeline._parse_worker("legacy", ctx.q_native, ctx), ctx.q_native)

    terminal = pipeline.doc_status.records[binding.lightrag_doc_id]
    assert "artifact_materialization_failed" in terminal["error_msg"]
    assert str(leaked_path) not in terminal["error_msg"]
    assert ".lightrag-scratch" not in terminal["error_msg"]


@pytest.mark.asyncio
async def test_cross_root_drain_uses_only_owner_runtime_and_same_session_all_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[Any, ...]] = []
    workspace = "workspace-b"
    binding = _binding("doc-cross-root", workspace=workspace)
    pipeline_a = _TestPipeline(workspace=workspace, events=[])
    pipeline_b = _TestPipeline(workspace=workspace, events=events)

    namespace_status = {
        "busy": False,
        "scanning_exclusive": False,
        "destructive_busy": False,
    }
    namespace_lock = asyncio.Lock()

    async def get_namespace_data(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return namespace_status

    def get_namespace_lock(*args: Any, **kwargs: Any) -> asyncio.Lock:
        del args, kwargs
        return namespace_lock

    monkeypatch.setattr(pipeline_module, "get_namespace_data", get_namespace_data)
    monkeypatch.setattr(pipeline_module, "get_namespace_lock", get_namespace_lock)

    root_a = tmp_path / "root-a"
    sidecar_a, blocks_a = _write_sidecar(root_a, "poison-from-root-a")
    await pipeline_a.apipeline_enqueue_documents(
        input=[""],
        ids=[binding.lightrag_doc_id],
        file_paths=["doc.pdf"],
        docs_format=FULL_DOCS_FORMAT_LIGHTRAG,
        process_options="!",
        artifact_bindings=[binding],
    )
    durable_a = pipeline_a.full_docs.records[binding.lightrag_doc_id]
    assert str(root_a) not in json.dumps(durable_a, default=str)
    assert str(sidecar_a) not in json.dumps(durable_a, default=str)
    assert str(blocks_a) not in json.dumps(durable_a, default=str)

    pipeline_b.full_docs.records = deepcopy(pipeline_a.full_docs.records)
    pipeline_b.doc_status.records = deepcopy(pipeline_a.doc_status.records)
    status_doc = _status_doc(binding, "doc.pdf")

    root_b = tmp_path / "root-b"
    sidecar_b, blocks_b = _write_sidecar(root_b, "content-from-root-b")
    session = _RuntimeSession(
        label="cross-root",
        binding=binding,
        events=events,
        sidecar_dir=sidecar_b,
        blocks_path=blocks_b,
    )

    async def materialize(received: PipelineArtifactBinding) -> _RuntimeSession:
        events.append(("open", "cross-root"))
        assert received == binding
        return session

    pipeline_b.pipeline_artifact_materializer = materialize
    owner_reads: list[Path] = []
    original_loader = pipeline_module.load_lightrag_document_content_from_blocks_path

    async def owner_loader(path: Any) -> tuple[str, str]:
        owner_reads.append(Path(path).resolve())
        return await original_loader(path)

    monkeypatch.setattr(
        pipeline_module,
        "load_lightrag_document_content_from_blocks_path",
        owner_loader,
    )
    ctx = _ctx(events)
    ctx.q_analyze = _RecordingQueue(events, "analyze")
    ctx.q_process = _RecordingQueue(events, "process")
    original_handoff = pipeline_b._handoff_pipeline_artifact_session

    def recording_handoff(**kwargs: Any) -> None:
        events.append(("handoff", kwargs["from_stage"], kwargs["to_stage"]))
        original_handoff(**kwargs)

    monkeypatch.setattr(
        pipeline_b,
        "_handoff_pipeline_artifact_session",
        recording_handoff,
    )
    await ctx.q_native.put((binding.lightrag_doc_id, status_doc, None))
    await _run_worker(
        pipeline_b._parse_worker("legacy", ctx.q_native, ctx), ctx.q_native
    )

    assert owner_reads == [blocks_b.resolve()]
    assert blocks_a.resolve() not in owner_reads
    persisted = pipeline_b.full_docs.records[binding.lightrag_doc_id]
    assert persisted["content"] == "{{LRdoc}}content-from-root-b"
    assert "sidecar_location" not in persisted
    assert "blocks_path" not in persisted
    assert str(root_b) not in json.dumps(persisted, default=str)

    analyze_item = ctx.q_analyze.get_nowait()
    ctx.q_analyze.task_done()
    assert analyze_item[3] is session
    assert Path(analyze_item[2]["blocks_path"]).resolve() == blocks_b.resolve()
    await ctx.q_analyze.put(analyze_item)
    await _run_worker(pipeline_b._analyze_worker(ctx), ctx.q_analyze)

    process_item = ctx.q_process.get_nowait()
    ctx.q_process.task_done()
    assert process_item[3] is session
    await ctx.q_process.put(process_item)
    seen_process_sessions: list[Any] = []

    async def finish_process(**kwargs: Any) -> None:
        seen_process_sessions.append(kwargs["artifact_session"])
        state = kwargs["ctx"].active_sessions[kwargs["doc_id"]]
        state.producer_active = False
        await pipeline_b._handoff_pipeline_artifact_success(
            doc_id=kwargs["doc_id"],
            session=kwargs["artifact_session"],
            parsed_data={
                **kwargs["parsed_data"],
                "entity_count": 0,
                "relation_count": 0,
            },
            chunks_count=1,
            status_doc=kwargs["status_doc"],
            file_path=kwargs["status_doc"].file_path,
            terminal_fields={"chunks_count": 1, "chunks_list": ["chunk-a"]},
            terminal_metadata={"process_start_time": 1, "process_end_time": 2},
            ctx=kwargs["ctx"],
        )

    monkeypatch.setattr(pipeline_b, "process_single_document", finish_process)
    await _run_worker(pipeline_b._process_worker(ctx), ctx.q_process)

    assert seen_process_sessions == [session]
    assert events.count(("handoff_success", "cross-root")) == 1
    assert events.index(("put", "analyze")) < events.index(
        ("handoff", "parse", "analyze")
    )
    assert events.index(("put", "process")) < events.index(
        ("handoff", "analyze", "process")
    )
    assert not any(event[:2] == ("finish", "cross-root") for event in events)
    assert events.count(("close", "cross-root")) == 1
    assert (
        "terminal",
        binding.lightrag_doc_id,
        DocStatus.PROCESSED.value,
    ) in events
    committed = PipelineArtifactBinding.from_mapping(
        pipeline_b.full_docs.records[binding.lightrag_doc_id]["artifact_binding"]
    )
    assert committed.state == "committed"
    assert (
        events.index(("close", "cross-root"))
        < events.index(
            ("terminal", binding.lightrag_doc_id, DocStatus.PROCESSED.value)
        )
    )
    assert ctx.active_sessions == {}


@pytest.mark.asyncio
async def test_pending_parse_uses_session_source_path_only(tmp_path: Path) -> None:
    events: list[tuple[Any, ...]] = []
    pipeline = _TestPipeline(workspace="workspace-b", events=events)
    binding = _binding("doc-pending", operation="parse")
    source = (
        tmp_path
        / "root-b"
        / ".lightrag-scratch"
        / "op-source"
        / "source"
        / "note.txt"
    )
    source.parent.mkdir(parents=True)
    source.write_text("pending source from owner B", encoding="utf-8")
    session = _RuntimeSession(
        label="pending",
        binding=binding,
        events=events,
        source_path=source,
    )

    async def materialize(received: PipelineArtifactBinding) -> _RuntimeSession:
        assert received == binding
        return session

    pipeline.pipeline_artifact_materializer = materialize
    pipeline.full_docs.records[binding.lightrag_doc_id] = _pending_full_doc(
        binding, "note.txt"
    )
    status_doc = _status_doc(binding, "note.txt")
    pipeline.doc_status.records[binding.lightrag_doc_id] = {
        "status": DocStatus.PENDING,
        "content_summary": "",
        "content_length": 0,
        "file_path": "note.txt",
        "created_at": status_doc.created_at,
        "updated_at": status_doc.updated_at,
        "track_id": status_doc.track_id,
        "metadata": deepcopy(status_doc.metadata),
    }
    ctx = _ctx(events)
    await ctx.q_native.put((binding.lightrag_doc_id, status_doc, None))
    await _run_worker(
        pipeline._parse_worker("legacy", ctx.q_native, ctx), ctx.q_native
    )

    assert pipeline.resolver_calls == [(str(source.resolve()), None)]
    persisted = pipeline.full_docs.records[binding.lightrag_doc_id]
    assert persisted["file_path"] == "note.txt"
    assert persisted["content"] == "{{LRdoc}}pending source from owner B"
    assert str(source) not in json.dumps(persisted, default=str)
    assert "sidecar_location" not in persisted
    analyze_item = ctx.q_analyze.get_nowait()
    ctx.q_analyze.task_done()
    assert analyze_item[3] is session
    assert Path(analyze_item[2]["blocks_path"]).is_relative_to(source.parent)
    state = ctx.active_sessions[binding.lightrag_doc_id]
    state.producer_active = False
    await pipeline._finish_pipeline_artifact_session(
        doc_id=binding.lightrag_doc_id,
        session=session,
        outcome=PipelineTerminalOutcome.FAILED,
        ctx=ctx,
    )


@pytest.mark.asyncio
async def test_queue_put_failure_cleans_owner_before_any_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[Any, ...]] = []
    pipeline = _TestPipeline(workspace="workspace-b", events=events)
    binding = _binding("doc-put-failure")
    status_doc = await _seed_build(pipeline, binding)
    sidecar, blocks = _write_sidecar(tmp_path, "runtime content")
    session = _RuntimeSession(
        label="put-failure",
        binding=binding,
        events=events,
        sidecar_dir=sidecar,
        blocks_path=blocks,
    )

    async def materialize(received: PipelineArtifactBinding) -> _RuntimeSession:
        assert received == binding
        return session

    pipeline.pipeline_artifact_materializer = materialize
    original_handoff = pipeline._handoff_pipeline_artifact_session

    def recording_handoff(**kwargs: Any) -> None:
        events.append(("handoff", kwargs["from_stage"], kwargs["to_stage"]))
        original_handoff(**kwargs)

    monkeypatch.setattr(
        pipeline,
        "_handoff_pipeline_artifact_session",
        recording_handoff,
    )
    ctx = _ctx(events)
    ctx.q_analyze = _FailingPutQueue(events)
    await ctx.q_native.put((binding.lightrag_doc_id, status_doc, None))
    await _run_worker(pipeline._parse_worker("native", ctx.q_native, ctx), ctx.q_native)

    assert ("put_failed", "analyze") in events
    assert not any(event[0] == "handoff" for event in events)
    assert events.count(
        ("finish", "put-failure", PipelineTerminalOutcome.FAILED.value)
    ) == 1
    assert events.count(("close", "put-failure")) == 1
    terminal_index = events.index(
        ("terminal", binding.lightrag_doc_id, DocStatus.FAILED.value)
    )
    assert events.index(("close", "put-failure")) < terminal_index
    assert ctx.active_sessions == {}


@pytest.mark.parametrize("stage", ["parse", "analyze", "process"])
@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.asyncio
async def test_stage_failure_or_cancel_finishes_and_closes_before_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    cancelled: bool,
) -> None:
    events: list[tuple[Any, ...]] = []
    pipeline = _TestPipeline(workspace="workspace-b", events=events)
    binding = _binding(f"doc-{stage}-{cancelled}")
    status_doc = await _seed_build(pipeline, binding)
    sidecar, blocks = _write_sidecar(tmp_path / stage, "runtime content")
    session = _RuntimeSession(
        label=stage,
        binding=binding,
        events=events,
        sidecar_dir=sidecar,
        blocks_path=blocks,
    )

    async def materialize(received: PipelineArtifactBinding) -> _RuntimeSession:
        assert received == binding
        return session

    pipeline.pipeline_artifact_materializer = materialize
    error: BaseException
    if cancelled:
        error = PipelineCancelledException(
            f"User cancelled during {stage}: {blocks}"
        )
        expected_outcome = PipelineTerminalOutcome.CANCELLED.value
    else:
        error = RuntimeError(f"{stage} failed at {blocks}")
        expected_outcome = PipelineTerminalOutcome.FAILED.value
    ctx = _ctx(events)

    if stage == "parse":

        async def stop_parse(**kwargs: Any) -> dict[str, Any]:
            del kwargs
            events.append(("producer_stop", stage))
            raise error

        monkeypatch.setattr(
            pipeline,
            "_parse_with_pipeline_artifact_session",
            stop_parse,
        )
        await ctx.q_native.put((binding.lightrag_doc_id, status_doc, None))
        await _run_worker(
            pipeline._parse_worker("legacy", ctx.q_native, ctx), ctx.q_native
        )
    elif stage == "analyze":
        await pipeline._open_pipeline_artifact_session(
            binding=binding,
            doc_id=binding.lightrag_doc_id,
            status_doc=status_doc,
            file_path=status_doc.file_path,
            ctx=ctx,
        )
        pipeline._handoff_pipeline_artifact_session(
            doc_id=binding.lightrag_doc_id,
            session=session,
            from_stage="parse",
            to_stage="analyze",
            ctx=ctx,
        )
        pipeline.analyze_error = error
        await ctx.q_analyze.put(
            (
                binding.lightrag_doc_id,
                status_doc,
                {"blocks_path": str(blocks)},
                session,
            )
        )
        await _run_worker(pipeline._analyze_worker(ctx), ctx.q_analyze)
    else:
        await pipeline._open_pipeline_artifact_session(
            binding=binding,
            doc_id=binding.lightrag_doc_id,
            status_doc=status_doc,
            file_path=status_doc.file_path,
            ctx=ctx,
        )
        state = ctx.active_sessions[binding.lightrag_doc_id]
        pipeline._handoff_pipeline_artifact_session(
            doc_id=binding.lightrag_doc_id,
            session=session,
            from_stage="parse",
            to_stage="analyze",
            ctx=ctx,
        )
        pipeline._handoff_pipeline_artifact_session(
            doc_id=binding.lightrag_doc_id,
            session=session,
            from_stage="analyze",
            to_stage="process",
            ctx=ctx,
        )
        state.producer_active = True
        producer_ready = asyncio.Event()

        async def active_producer() -> None:
            producer_ready.set()
            try:
                await asyncio.Event().wait()
            finally:
                events.append(("producer_stop", stage))

        producer = asyncio.create_task(active_producer())
        await producer_ready.wait()
        await pipeline._finalize_doc_failure(
            doc_id=binding.lightrag_doc_id,
            status_doc=status_doc,
            file_path=status_doc.file_path,
            error=error,
            stage_label="extract",
            current_file_number=1,
            total_files=1,
            failed_chunks_snapshot=([], 0),
            pending_tasks=[producer],
            metadata_extra={},
            pipeline_status=ctx.pipeline_status,
            pipeline_status_lock=ctx.pipeline_status_lock,
            artifact_session=session,
            artifact_binding=binding,
            ctx=ctx,
        )

    producer_index = events.index(("producer_stop", stage))
    finish_event = ("finish", stage, expected_outcome)
    finish_index = events.index(finish_event)
    close_index = events.index(("close", stage))
    terminal_index = events.index(
        ("terminal", binding.lightrag_doc_id, DocStatus.FAILED.value)
    )
    assert producer_index < finish_index < close_index < terminal_index
    assert events.count(finish_event) == 1
    assert events.count(("close", stage)) == 1
    terminal = pipeline.doc_status.records[binding.lightrag_doc_id]
    assert ".lightrag-scratch" not in terminal["error_msg"]
    assert str(blocks) not in terminal["error_msg"]
    assert ctx.active_sessions == {}


@pytest.mark.asyncio
async def test_batch_residual_cleanup_closes_inactive_and_defers_active_producer(
    tmp_path: Path,
) -> None:
    events: list[tuple[Any, ...]] = []
    pipeline = _TestPipeline(workspace="workspace-b", events=events)
    ctx = _ctx(events)

    inactive_binding = _binding("doc-residual-inactive")
    active_binding = _binding("doc-residual-active")
    inactive_status = await _seed_build(pipeline, inactive_binding)
    active_status = await _seed_build(pipeline, active_binding)
    inactive_sidecar, inactive_blocks = _write_sidecar(
        tmp_path / "inactive", "inactive"
    )
    active_sidecar, active_blocks = _write_sidecar(tmp_path / "active", "active")
    inactive = _RuntimeSession(
        label="inactive",
        binding=inactive_binding,
        events=events,
        sidecar_dir=inactive_sidecar,
        blocks_path=inactive_blocks,
    )
    active = _RuntimeSession(
        label="active",
        binding=active_binding,
        events=events,
        sidecar_dir=active_sidecar,
        blocks_path=active_blocks,
    )
    sessions = {
        inactive_binding.lightrag_doc_id: inactive,
        active_binding.lightrag_doc_id: active,
    }

    async def materialize(binding: PipelineArtifactBinding) -> _RuntimeSession:
        return sessions[binding.lightrag_doc_id]

    pipeline.pipeline_artifact_materializer = materialize
    for binding, status_doc, session, blocks in (
        (inactive_binding, inactive_status, inactive, inactive_blocks),
        (active_binding, active_status, active, active_blocks),
    ):
        await pipeline._open_pipeline_artifact_session(
            binding=binding,
            doc_id=binding.lightrag_doc_id,
            status_doc=status_doc,
            file_path=status_doc.file_path,
            ctx=ctx,
        )
        state = ctx.active_sessions[binding.lightrag_doc_id]
        pipeline._handoff_pipeline_artifact_session(
            doc_id=binding.lightrag_doc_id,
            session=session,
            from_stage="parse",
            to_stage="analyze",
            ctx=ctx,
        )
        state.producer_active = session is active

    await pipeline._cleanup_residual_pipeline_artifact_sessions(ctx)

    assert ("finish", "inactive", PipelineTerminalOutcome.FAILED.value) in events
    assert ("close", "inactive") in events
    assert (
        "terminal",
        inactive_binding.lightrag_doc_id,
        DocStatus.FAILED.value,
    ) in events
    assert ("defer", "active") in events
    assert not any(event[:2] == ("finish", "active") for event in events)
    assert ("close", "active") not in events
    assert not any(
        event[:2] == ("terminal", active_binding.lightrag_doc_id)
        for event in events
    )
    assert ctx.active_sessions == {}
