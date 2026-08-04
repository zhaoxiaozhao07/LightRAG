from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import pytest

from lightrag import LightRAG
from lightrag.artifact_runtime import (
    PipelineArtifactBinding,
    PipelineArtifactCommitOutcome,
    PipelineArtifactFinalizationResult,
    PipelineAttemptCommitOutcomeUnknownError,
    PipelineAttemptRowKind,
    PipelineTerminalOutcome,
    assert_no_runtime_artifact_payload,
    extract_pipeline_attempt_token,
)
from lightrag.base import DocProcessingStatus, DocStatus
from lightrag.constants import FULL_DOCS_FORMAT_LIGHTRAG
from lightrag.pipeline import _ActivePipelineArtifactSession, _BatchRunContext


pytestmark = pytest.mark.offline

_DOC_ID = "doc-h2c2"
_FILE_PATH = "artifact.pdf"
_ATTEMPT_TOKEN = "attempt-h2c2"
_NEWER_ATTEMPT_TOKEN = "attempt-newer"
_RUNTIME_PARSED_DATA = {
    "entity_count": 99,
    "blocks_path": "/tmp/.lightrag-scratch/op/doc.blocks.jsonl",
    "runtime_payload": {"scratch_lease_id": "lease-secret"},
}


class _AuthorityStaleError(RuntimeError):
    error_code = "artifact_binding_stale"


class _Store:
    def __init__(
        self,
        label: str,
        record: dict[str, Any],
        events: list[str],
        *,
        fail_upsert: bool = False,
    ) -> None:
        self.label = label
        self.records = {_DOC_ID: deepcopy(record)}
        self.events = events
        self.fail_upsert = fail_upsert
        self.lock = asyncio.Lock()
        self.cas_entered: asyncio.Event | None = None
        self.cas_release: asyncio.Event | None = None

    async def get_by_id(self, key: str) -> dict[str, Any] | None:
        async with self.lock:
            value = self.records.get(key)
            return deepcopy(value) if value is not None else None

    async def upsert(self, values: dict[str, dict[str, Any]]) -> None:
        assert_no_runtime_artifact_payload(values, context=f"test {self.label} write")
        copied = deepcopy(values)
        async with self.lock:
            self.records.update(copied)

    async def replace_attempt(self, value: dict[str, Any]) -> None:
        assert_no_runtime_artifact_payload(value, context=f"test {self.label} takeover")
        async with self.lock:
            self.records[_DOC_ID] = deepcopy(value)

    async def compare_and_commit_pipeline_attempt(
        self,
        key: str,
        payload: Mapping[str, Any],
        *,
        expected_attempt_token: str,
        row_kind: PipelineAttemptRowKind,
    ) -> bool:
        assert key == _DOC_ID
        assert row_kind == self.label
        assert_no_runtime_artifact_payload(payload, context=f"test {self.label} CAS")
        if self.cas_entered is not None:
            self.cas_entered.set()
        if self.cas_release is not None:
            await self.cas_release.wait()
        if self.fail_upsert:
            self.events.append(f"{self.label}.write_failed")
            raise RuntimeError(f"{self.label} write failed")
        async with self.lock:
            current = self.records.get(key)
            if (
                extract_pipeline_attempt_token(current, row_kind=row_kind)
                != expected_attempt_token
            ):
                self.events.append(f"{self.label}.cas_stale")
                return False
            value = deepcopy(dict(payload))
            self.records[key] = value
        if self.label == "full_docs":
            binding = PipelineArtifactBinding.from_mapping(value["artifact_binding"])
            assert binding.state == "committed"
            self.events.append("full_docs.committed_binding")
        else:
            assert value["status"] is DocStatus.PROCESSED
            self.events.append("doc_status.processed")
        return True

    async def index_done_callback(self) -> None:
        assert self.label == "full_docs"
        self.events.append("full_docs.callback")


class _Session:
    source_path = None
    sidecar_dir = None
    blocks_path = None
    producer_active = False

    def __init__(
        self,
        binding: PipelineArtifactBinding,
        events: list[str],
        *,
        result: PipelineArtifactFinalizationResult | None,
        handoff_error: BaseException | None = None,
        close_fails: bool = False,
    ) -> None:
        self.binding = binding
        self.events = events
        self.result = result
        self.handoff_error = handoff_error
        self.close_fails = close_fails
        self.received: tuple[Mapping[str, Any] | None, int | None] | None = None

    def redact(self, error: object) -> str:
        return str(error)

    def defer_cleanup(self) -> None:
        self.events.append("scratch.defer")

    async def finish(self, outcome: PipelineTerminalOutcome) -> None:
        del outcome
        self.events.append("owner.release")

    async def handoff_success(
        self,
        *,
        parsed_data: Mapping[str, Any] | None = None,
        chunks_count: int | None = None,
    ) -> PipelineArtifactFinalizationResult:
        self.received = (deepcopy(parsed_data), chunks_count)
        self.events.append("session.handoff")
        if self.handoff_error is not None:
            self.events.append("pg.authority_stale")
            raise self.handoff_error
        assert self.result is not None
        self.events.append(f"pg.{self.result.outcome.value}")
        return self.result

    async def aclose(self) -> None:
        self.events.append("scratch.aclose")
        if self.close_fails:
            raise RuntimeError("scratch close failed")


@dataclass
class _Harness:
    pipeline: LightRAG
    full_docs: _Store
    doc_status: _Store
    committed_binding: PipelineArtifactBinding
    status_doc: DocProcessingStatus
    session: _Session
    ctx: _BatchRunContext
    events: list[str]


def _binding(token: str = _ATTEMPT_TOKEN) -> PipelineArtifactBinding:
    return PipelineArtifactBinding(
        version=1,
        authority="kb_metadata",
        state="claimed",
        operation="build",
        kb_id="kb-h2c2",
        kb_generation="generation-h2c2",
        workspace="workspace-h2c2",
        document_id="metadata-h2c2",
        lightrag_doc_id=_DOC_ID,
        job_id=f"job-{token}",
        claim_token=token,
        source_hash="sha256:source",
        parser_hash="sha256:parser",
        parse_generation_id="parse-generation-h2c2",
        index_hash="sha256:index",
        sidecar_artifact_id="sidecar-claimed",
        blocks_artifact_id="blocks-claimed",
        expected_current_sidecar_artifact_id="sidecar-claimed",
        expected_current_blocks_artifact_id="blocks-claimed",
    )


def _committed(binding: PipelineArtifactBinding) -> PipelineArtifactBinding:
    return binding.committed(
        parse_generation_id=binding.parse_generation_id,
        index_hash=binding.index_hash,
        sidecar_artifact_id="sidecar-committed",
        blocks_artifact_id="blocks-committed",
        raw_artifact_ids=(),
    )


def _full_doc(binding: PipelineArtifactBinding) -> dict[str, Any]:
    return {
        "content": "{{LRdoc}}durable content",
        "file_path": _FILE_PATH,
        "parse_format": FULL_DOCS_FORMAT_LIGHTRAG,
        "parse_engine": "native",
        "process_options": "!",
        "chunk_options": {"chunk_token_size": 128},
        "artifact_binding": binding.to_dict(),
    }


def _status_doc(binding: PipelineArtifactBinding) -> DocProcessingStatus:
    return DocProcessingStatus(
        content_summary="durable",
        content_length=15,
        file_path=_FILE_PATH,
        status=DocStatus.PROCESSING,
        created_at="2026-08-02T00:00:00+00:00",
        updated_at="2026-08-02T00:00:01+00:00",
        track_id="track-h2c2",
        metadata={
            "process_options": "!",
            "pipeline_attempt_token": binding.claim_token,
        },
    )


def _context(
    binding: PipelineArtifactBinding,
    session: _Session,
    status_doc: DocProcessingStatus,
) -> _BatchRunContext:
    ctx = _BatchRunContext(
        pipeline_status={},
        pipeline_status_lock=asyncio.Lock(),
        semaphore=asyncio.Semaphore(1),
        total_files=1,
        q_native=asyncio.Queue(),
        q_mineru=asyncio.Queue(),
        q_docling=asyncio.Queue(),
        q_analyze=asyncio.Queue(),
        q_process=asyncio.Queue(),
    )
    ctx.active_sessions[_DOC_ID] = _ActivePipelineArtifactSession(
        binding=binding,
        session=session,
        status_doc=status_doc,
        file_path=_FILE_PATH,
        stage="process",
    )
    return ctx


def _harness(fault: str | None = None) -> _Harness:
    events: list[str] = []
    binding = _binding()
    committed_binding = _committed(binding)
    result = PipelineArtifactFinalizationResult(
        outcome=(
            PipelineArtifactCommitOutcome.UNKNOWN
            if fault == "unknown"
            else PipelineArtifactCommitOutcome.COMMITTED
        ),
        committed_binding=None if fault == "unknown" else committed_binding,
        chunks_count=None if fault == "unknown" else 6,
        entity_count=None if fault == "unknown" else 2,
        relation_count=None if fault == "unknown" else 1,
        reason="commit outcome unknown" if fault == "unknown" else None,
    )
    session = _Session(
        binding,
        events,
        result=result,
        handoff_error=(
            _AuthorityStaleError("newer authority owns the document")
            if fault == "stale"
            else None
        ),
        close_fails=fault == "close",
    )
    durable_binding = (
        _committed(_binding(_NEWER_ATTEMPT_TOKEN)) if fault == "stale" else binding
    )
    durable_status_binding = (
        _binding(_NEWER_ATTEMPT_TOKEN) if fault == "stale" else binding
    )

    pipeline = object.__new__(LightRAG)
    full_docs = _Store(
        "full_docs",
        _full_doc(durable_binding),
        events,
        fail_upsert=fault == "binding_patch",
    )
    doc_status = _Store(
        "doc_status",
        asdict(_status_doc(durable_status_binding)),
        events,
        fail_upsert=fault == "doc_status",
    )
    setattr(pipeline, "full_docs", full_docs)
    setattr(pipeline, "doc_status", doc_status)
    status_doc = _status_doc(binding)
    return _Harness(
        pipeline=pipeline,
        full_docs=full_docs,
        doc_status=doc_status,
        committed_binding=committed_binding,
        status_doc=status_doc,
        session=session,
        ctx=_context(binding, session, status_doc),
        events=events,
    )


async def _handoff(harness: _Harness):
    return await harness.pipeline._handoff_pipeline_artifact_success(
        doc_id=_DOC_ID,
        session=harness.session,
        parsed_data=_RUNTIME_PARSED_DATA,
        chunks_count=4,
        status_doc=harness.status_doc,
        file_path=_FILE_PATH,
        terminal_fields={"chunks_count": 999, "chunks_list": ["chunk-h2c2"]},
        terminal_metadata={"process_start_time": 10, "process_end_time": 20},
        ctx=harness.ctx,
    )


@pytest.mark.asyncio
async def test_committed_handoff_orders_patch_close_then_processed() -> None:
    harness = _harness()

    disposition = await _handoff(harness)

    assert harness.events == [
        "session.handoff",
        "pg.committed",
        "full_docs.committed_binding",
        "scratch.aclose",
        "doc_status.processed",
    ]
    assert disposition.result is harness.session.result
    assert disposition.status_published is True
    assert disposition.recovery_stage is None
    assert harness.ctx.active_sessions == {}
    assert harness.session.received == (_RUNTIME_PARSED_DATA, 4)

    full_doc = harness.full_docs.records[_DOC_ID]
    assert (
        PipelineArtifactBinding.from_mapping(full_doc["artifact_binding"])
        == harness.committed_binding
    )
    terminal = harness.doc_status.records[_DOC_ID]
    assert terminal["status"] is DocStatus.PROCESSED
    assert terminal["metadata"]["pipeline_attempt_token"] == _ATTEMPT_TOKEN
    assert terminal["chunks_count"] == 6
    assert terminal["entity_count"] == 2
    assert terminal["relation_count"] == 1
    assert terminal["chunks_list"] == ["chunk-h2c2"]
    assert_no_runtime_artifact_payload(harness.full_docs.records)
    assert_no_runtime_artifact_payload(harness.doc_status.records)


def _fault_events(fault: str) -> list[str]:
    if fault == "unknown":
        return ["session.handoff", "pg.unknown", "scratch.aclose"]
    if fault == "stale":
        return ["session.handoff", "pg.authority_stale", "scratch.aclose"]
    events = ["session.handoff", "pg.committed"]
    if fault == "binding_patch":
        return [*events, "full_docs.write_failed", "scratch.aclose"]
    events.extend(["full_docs.committed_binding", "scratch.aclose"])
    if fault == "doc_status":
        events.append("doc_status.write_failed")
    return events


@pytest.mark.parametrize(
    ("fault", "recovery_stage", "binding_patched"),
    [
        pytest.param("unknown", "metadata_commit_unknown", False, id="unknown"),
        pytest.param("binding_patch", "binding_patch", False, id="binding-patch"),
        pytest.param("close", "close", True, id="close"),
        pytest.param("doc_status", "doc_status", True, id="doc-status"),
        pytest.param("stale", "authority_stale", False, id="authority-stale"),
    ],
)
@pytest.mark.asyncio
async def test_handoff_fault_boundaries_never_publish_false_success(
    fault: str,
    recovery_stage: str,
    binding_patched: bool,
) -> None:
    harness = _harness(fault)
    full_docs_before = deepcopy(harness.full_docs.records)
    doc_status_before = deepcopy(harness.doc_status.records)

    disposition = await _handoff(harness)

    assert harness.events == _fault_events(fault)
    assert disposition.recovery_stage == recovery_stage
    assert disposition.status_published is False
    assert disposition.result is (None if fault == "stale" else harness.session.result)
    assert harness.ctx.active_sessions == {}
    assert "owner.release" not in harness.events
    assert "scratch.defer" not in harness.events

    if binding_patched:
        full_doc = harness.full_docs.records[_DOC_ID]
        assert (
            PipelineArtifactBinding.from_mapping(full_doc["artifact_binding"])
            == harness.committed_binding
        )
    else:
        assert harness.full_docs.records == full_docs_before

    assert harness.doc_status.records == doc_status_before
    assert harness.doc_status.records[_DOC_ID]["status"] is not DocStatus.PROCESSED
    if fault == "stale":
        newer_binding = PipelineArtifactBinding.from_mapping(
            harness.full_docs.records[_DOC_ID]["artifact_binding"]
        )
        assert newer_binding.claim_token == _NEWER_ATTEMPT_TOKEN
        assert (
            harness.doc_status.records[_DOC_ID]["metadata"]["pipeline_attempt_token"]
            == _NEWER_ATTEMPT_TOKEN
        )
    assert_no_runtime_artifact_payload(harness.full_docs.records)
    assert_no_runtime_artifact_payload(harness.doc_status.records)


def _pending_status(binding: PipelineArtifactBinding) -> dict[str, Any]:
    row = asdict(_status_doc(binding))
    row["status"] = DocStatus.PENDING
    return row


def test_unknown_cas_error_reason_is_durable_safe() -> None:
    error = PipelineAttemptCommitOutcomeUnknownError(
        _DOC_ID,
        row_kind="full_docs",
        reason=(
            "write_json: OSError: failed at "
            "/tmp/.lightrag-scratch/private with access:secret"
        ),
    )

    assert error.reason == "write_json:OSError"
    assert ".lightrag-scratch" not in str(error)
    assert "access:secret" not in str(error)


@pytest.mark.asyncio
async def test_old_terminalization_takeover_before_full_docs_cas_is_stale() -> None:
    harness = _harness()
    harness.full_docs.cas_entered = asyncio.Event()
    harness.full_docs.cas_release = asyncio.Event()

    old_terminalization = asyncio.create_task(_handoff(harness))
    await asyncio.wait_for(harness.full_docs.cas_entered.wait(), timeout=2)

    newer = _binding(_NEWER_ATTEMPT_TOKEN)
    await harness.full_docs.replace_attempt(_full_doc(newer))
    await harness.doc_status.replace_attempt(_pending_status(newer))
    harness.full_docs.cas_release.set()

    disposition = await asyncio.wait_for(old_terminalization, timeout=2)

    assert disposition.status_published is False
    assert disposition.recovery_stage == "authority_stale"
    assert "full_docs.cas_stale" in harness.events
    assert "scratch.aclose" in harness.events
    assert "doc_status.processed" not in harness.events
    current_full = PipelineArtifactBinding.from_mapping(
        harness.full_docs.records[_DOC_ID]["artifact_binding"]
    )
    assert current_full == newer
    current_status = harness.doc_status.records[_DOC_ID]
    assert current_status["status"] is DocStatus.PENDING
    assert current_status["metadata"]["pipeline_attempt_token"] == _NEWER_ATTEMPT_TOKEN


@pytest.mark.asyncio
async def test_newer_attempt_between_binding_and_status_cas_is_not_overwritten() -> (
    None
):
    harness = _harness()
    harness.doc_status.cas_entered = asyncio.Event()
    harness.doc_status.cas_release = asyncio.Event()

    old_terminalization = asyncio.create_task(_handoff(harness))
    await asyncio.wait_for(harness.doc_status.cas_entered.wait(), timeout=2)
    assert "full_docs.committed_binding" in harness.events
    assert "scratch.aclose" in harness.events

    newer = _binding(_NEWER_ATTEMPT_TOKEN)
    await harness.full_docs.replace_attempt(_full_doc(newer))
    await harness.doc_status.replace_attempt(_pending_status(newer))
    harness.doc_status.cas_release.set()

    disposition = await asyncio.wait_for(old_terminalization, timeout=2)

    assert disposition.status_published is False
    assert disposition.recovery_stage == "doc_status_stale"
    assert "doc_status.cas_stale" in harness.events
    assert "doc_status.processed" not in harness.events
    assert (
        PipelineArtifactBinding.from_mapping(
            harness.full_docs.records[_DOC_ID]["artifact_binding"]
        )
        == newer
    )
    current_status = harness.doc_status.records[_DOC_ID]
    assert current_status["status"] is DocStatus.PENDING
    assert current_status["metadata"]["pipeline_attempt_token"] == _NEWER_ATTEMPT_TOKEN
