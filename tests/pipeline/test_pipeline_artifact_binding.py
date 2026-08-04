from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Literal

import pytest

import lightrag.pipeline as pipeline_module
from lightrag.artifact_runtime import PipelineArtifactBinding
from lightrag.constants import (
    FULL_DOCS_FORMAT_LIGHTRAG,
    FULL_DOCS_FORMAT_PENDING_PARSE,
)
from lightrag.pipeline import _PipelineMixin
from lightrag.utils_pipeline import (
    doc_status_reset_metadata,
    doc_status_transition_metadata,
)


pytestmark = pytest.mark.offline


def _assert_durable_write(value: Any) -> None:
    encoded = json.dumps(value, ensure_ascii=False, default=str)
    assert ".lightrag-scratch" not in encoded
    assert "sidecar_location" not in encoded
    assert "blocks_path" not in encoded


class _GuardedMemoryStore:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.writes: list[dict[str, dict[str, Any]]] = []

    async def filter_keys(self, keys: set[str]) -> set[str]:
        return set(keys) - set(self.records)

    async def get_by_id(self, key: str) -> dict[str, Any] | None:
        value = self.records.get(key)
        return dict(value) if value is not None else None

    async def get_doc_by_file_basename(
        self, basename: str
    ) -> tuple[str, dict[str, Any]] | None:
        for key, value in self.records.items():
            if value.get("file_path") == basename:
                return key, dict(value)
        return None

    async def get_doc_by_content_hash(
        self, content_hash: str
    ) -> tuple[str, dict[str, Any]] | None:
        for key, value in self.records.items():
            if value.get("content_hash") == content_hash:
                return key, dict(value)
        return None

    async def upsert(self, values: dict[str, dict[str, Any]]) -> None:
        _assert_durable_write(values)
        copied = {key: dict(value) for key, value in values.items()}
        self.writes.append(copied)
        self.records.update(copied)

    async def index_done_callback(self) -> None:
        return None


class _BindingPipeline(_PipelineMixin):
    def __init__(self, workspace: str = "workspace-a") -> None:
        self.workspace = workspace
        self.addon_params: dict[str, Any] = {}
        self.full_docs = _GuardedMemoryStore()
        self.doc_status = _GuardedMemoryStore()


class _LocalMemoryStore(_GuardedMemoryStore):
    async def upsert(self, values: dict[str, dict[str, Any]]) -> None:
        copied = {key: dict(value) for key, value in values.items()}
        self.writes.append(copied)
        self.records.update(copied)


class _LocalParsePipeline(_PipelineMixin):
    def __init__(self) -> None:
        self.workspace = "local-workspace"
        self.addon_params: dict[str, Any] = {}
        self.full_docs = _LocalMemoryStore()
        self.doc_status = _LocalMemoryStore()

    def _resolve_source_file_for_parser(
        self,
        file_path: str,
        *,
        source_file: str | None = None,
        parser_engine: str | None = None,
    ) -> str:
        del source_file, parser_engine
        return file_path


@pytest.fixture
def pipeline_runtime(monkeypatch):
    status: dict[str, Any] = {
        "busy": False,
        "scanning_exclusive": False,
        "destructive_busy": False,
    }
    lock = asyncio.Lock()

    async def get_namespace_data(*args, **kwargs):
        del args, kwargs
        return status

    def get_namespace_lock(*args, **kwargs):
        del args, kwargs
        return lock

    monkeypatch.setattr(pipeline_module, "get_namespace_data", get_namespace_data)
    monkeypatch.setattr(pipeline_module, "get_namespace_lock", get_namespace_lock)
    return status


def _binding(
    doc_id: str,
    *,
    workspace: str = "workspace-a",
    operation: Literal["parse", "build"] = "build",
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
        job_id="job-a",
        claim_token=f"claim-{doc_id}",
        source_hash="sha256:source",
        parser_hash="sha256:parser",
        parse_generation_id="parse-generation-a",
        index_hash="sha256:index",
        sidecar_artifact_id="sidecar-a",
        blocks_artifact_id="blocks-a",
        expected_current_sidecar_artifact_id="sidecar-a",
        expected_current_blocks_artifact_id="blocks-a",
        raw_artifact_ids=("raw-a",),
    )


def test_binding_schema_roundtrip_and_strict_rejection() -> None:
    binding = _binding("doc-a")
    encoded = binding.to_dict()
    assert PipelineArtifactBinding.from_mapping(
        encoded, expected_workspace="workspace-a"
    ) == binding

    invalid_mappings = []
    for patch in (
        {"version": 2},
        {"version": True},
        {"version": 1.0},
        {"authority": None},
        {"state": "running"},
        {"operation": "replace"},
        {"claim_token": ""},
        {"job_id": "/tmp/runtime-job"},
        {"job_id": "relative/runtime-job"},
        {"job_id": "s3://bucket/runtime-job"},
        {"job_id": " job-a"},
        {"job_id": "run/.lightrag-scratch/job"},
        {"raw_artifact_ids": ["raw-a", "raw-a"]},
    ):
        invalid_mappings.append({**encoded, **patch})
    invalid_mappings.append({**encoded, "object_uri": "s3://bucket/key"})
    invalid_mappings.append({**encoded, "presigned_url": "https://secret"})
    invalid_mappings.append({**encoded, "runtime_source_path": "/tmp/source"})
    invalid_mappings.append({**encoded, "scratch_lease_id": "lease-a"})
    invalid_mappings.append({**encoded, "sidecar_location": "file:///tmp/parsed"})
    invalid_mappings.append({**encoded, "blocks_path": "/tmp/blocks.jsonl"})

    for invalid in invalid_mappings:
        with pytest.raises((TypeError, ValueError)):
            PipelineArtifactBinding.from_mapping(invalid)

    with pytest.raises(ValueError, match="workspace mismatch"):
        PipelineArtifactBinding.from_mapping(
            encoded, expected_workspace="workspace-b"
        )
    with pytest.raises(TypeError, match="raw_artifact_ids"):
        PipelineArtifactBinding(**encoded)


@pytest.mark.asyncio
async def test_binding_enqueue_uses_explicit_ids_without_content_or_basename_dedupe(
    pipeline_runtime, monkeypatch
) -> None:
    del pipeline_runtime
    pipeline = _BindingPipeline()
    resolver_calls: list[str] = []

    def forbidden_resolver(value):
        resolver_calls.append(str(value))
        raise AssertionError("binding enqueue must not resolve a sidecar")

    async def forbidden_loader(value):
        resolver_calls.append(str(value))
        raise AssertionError("binding enqueue must not read a sidecar")

    monkeypatch.setattr(pipeline_module, "resolve_sidecar_uri", forbidden_resolver)
    monkeypatch.setattr(
        pipeline_module, "load_lightrag_document_content", forbidden_loader
    )

    bindings = [_binding("doc-a"), _binding("doc-b")]
    await pipeline.apipeline_enqueue_documents(
        input=["", ""],
        ids=["doc-a", "doc-b"],
        file_paths=["same.pdf", "same.pdf"],
        docs_format=FULL_DOCS_FORMAT_LIGHTRAG,
        artifact_bindings=bindings,
    )

    assert resolver_calls == []
    assert set(pipeline.full_docs.records) == {"doc-a", "doc-b"}
    for binding in bindings:
        full_doc = pipeline.full_docs.records[binding.lightrag_doc_id]
        assert full_doc["content"] == ""
        assert full_doc["file_path"] == "same.pdf"
        assert full_doc["artifact_binding"] == binding.to_dict()
        assert "content_hash" not in full_doc
        assert "sidecar_location" not in full_doc
        assert "blocks_path" not in full_doc

        status = pipeline.doc_status.records[binding.lightrag_doc_id]
        assert status["file_path"] == "same.pdf"
        assert status["metadata"]["pipeline_attempt_token"] == binding.claim_token
        assert doc_status_transition_metadata(status)["pipeline_attempt_token"] == (
            binding.claim_token
        )
        assert doc_status_reset_metadata(status)["pipeline_attempt_token"] == (
            binding.claim_token
        )
        _assert_durable_write(full_doc)
        _assert_durable_write(status)


@pytest.mark.asyncio
async def test_binding_enqueue_rejects_alignment_identity_workspace_and_runtime_inputs(
    pipeline_runtime,
) -> None:
    del pipeline_runtime
    pipeline = _BindingPipeline()
    binding = _binding("doc-a")

    invalid_calls = (
        {
            "input": [""],
            "ids": None,
            "file_paths": ["a.pdf"],
            "artifact_bindings": [binding],
        },
        {
            "input": [""],
            "ids": ["doc-a"],
            "file_paths": None,
            "artifact_bindings": [binding],
        },
        {
            "input": [""],
            "ids": ["doc-a"],
            "file_paths": ["a.pdf"],
            "artifact_bindings": [],
        },
        {
            "input": [""],
            "ids": ["doc-other"],
            "file_paths": ["a.pdf"],
            "artifact_bindings": [binding],
        },
        {
            "input": [""],
            "ids": ["doc-a"],
            "file_paths": ["a.pdf"],
            "artifact_bindings": [_binding("doc-a", workspace="workspace-b")],
        },
        {
            "input": [""],
            "ids": ["doc-a"],
            "file_paths": ["/tmp/.lightrag-scratch/a.pdf"],
            "artifact_bindings": [binding],
        },
        {
            "input": [""],
            "ids": ["doc-a"],
            "file_paths": ["a.pdf"],
            "lightrag_document_paths": ["/tmp/.lightrag-scratch/a.parsed"],
            "artifact_bindings": [binding],
        },
    )
    for kwargs in invalid_calls:
        with pytest.raises((TypeError, ValueError)):
            await pipeline.apipeline_enqueue_documents(
                docs_format=FULL_DOCS_FORMAT_LIGHTRAG,
                **kwargs,
            )
    with pytest.raises(ValueError, match="requires lightrag"):
        await pipeline.apipeline_enqueue_documents(
            input=[""],
            ids=["doc-a"],
            file_paths=["a.pdf"],
            artifact_bindings=[binding],
        )
    assert pipeline.full_docs.records == {}
    assert pipeline.doc_status.records == {}


@pytest.mark.asyncio
async def test_binding_durable_guard_rejects_scratch_before_first_store_write(
    pipeline_runtime,
) -> None:
    del pipeline_runtime
    pipeline = _BindingPipeline()
    with pytest.raises(ValueError, match="scratch runtime reference"):
        await pipeline.apipeline_enqueue_documents(
            input=["transient .lightrag-scratch/source content"],
            ids=["doc-a"],
            file_paths=["a.pdf"],
            docs_format=FULL_DOCS_FORMAT_LIGHTRAG,
            artifact_bindings=[_binding("doc-a")],
        )
    assert pipeline.full_docs.writes == []
    assert pipeline.doc_status.writes == []


@pytest.mark.asyncio
async def test_legacy_enqueue_without_binding_keeps_existing_behavior(
    pipeline_runtime,
) -> None:
    del pipeline_runtime
    pipeline = _BindingPipeline()
    await pipeline.apipeline_enqueue_documents(
        input=["legacy content"],
        ids=["legacy-doc"],
        file_paths=["/tmp/legacy.txt"],
    )
    full_doc = pipeline.full_docs.records["legacy-doc"]
    assert full_doc["file_path"] == "legacy.txt"
    assert full_doc["content"] == "legacy content"
    assert "content_hash" in full_doc
    assert "artifact_binding" not in full_doc


@pytest.mark.asyncio
async def test_local_lightrag_enqueue_still_rejects_sidecars_outside_input_root(
    pipeline_runtime, monkeypatch, tmp_path
) -> None:
    del pipeline_runtime
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    outside_blocks = tmp_path / "outside.blocks.jsonl"
    outside_blocks.write_text('{"type":"meta"}\n', encoding="utf-8")
    monkeypatch.setattr(pipeline_module, "input_dir_path", lambda: input_root)

    pipeline = _BindingPipeline()
    with pytest.raises(ValueError, match="stay under INPUT_DIR"):
        await pipeline.apipeline_enqueue_documents(
            input=[""],
            ids=["local-doc"],
            file_paths=["outside.pdf"],
            docs_format=FULL_DOCS_FORMAT_LIGHTRAG,
            lightrag_document_paths=[str(outside_blocks)],
        )


@pytest.mark.asyncio
async def test_local_pending_parse_keeps_runtime_compatibility_fields(tmp_path) -> None:
    source = tmp_path / "local-note.txt"
    source.write_text("local pending parse", encoding="utf-8")
    pipeline = _LocalParsePipeline()

    parsed = await pipeline.parse_legacy(
        "local-doc",
        str(source),
        {
            "parse_format": FULL_DOCS_FORMAT_PENDING_PARSE,
            "archive_source_after_parse": False,
        },
    )

    persisted = pipeline.full_docs.records["local-doc"]
    assert persisted["file_path"] == str(source)
    assert persisted["sidecar_location"].startswith("file://")
    assert "artifact_binding" not in persisted
    assert Path(parsed["blocks_path"]).is_file()
    assert source.is_file()
