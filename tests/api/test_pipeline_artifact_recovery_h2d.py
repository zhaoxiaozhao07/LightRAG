from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, cast

import pytest

from lightrag.artifact_runtime import (
    PipelineArtifactBinding,
    PipelineArtifactCommitOutcome,
    PipelineAttemptCommitOutcomeUnknownError,
    PipelineAttemptRowKind,
    extract_pipeline_attempt_token,
)
from lightrag.api.document_lifecycle_service import DocumentLifecycleService
from lightrag.api.index_build_service import IndexBuildService
from lightrag.api.job_service import JobService
from lightrag.api.kb_service import KnowledgeBaseService
from lightrag.api.artifact_lifecycle import ArtifactRecoveryGenerationError
from lightrag.api.metadata_store import (
    ArtifactRecoveryCursorRecord,
    DocumentRecord,
    SQLiteMetadataStore,
    _dumps_json,
    _loads_json_object,
)
from lightrag.api.pipeline_artifact_coordinator import PipelineArtifactCoordinator
from lightrag.api.pipeline_artifact_recovery import (
    PipelineArtifactTerminalizationReconciler,
)
from lightrag.utils_pipeline import reset_canonical_input_root_for_tests
from tests.api.test_artifact_storage_phase2a import (
    _FakeObjectStorage,
    _ParserRAG,
    _build_object_service,
    _create_document,
)


pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _reset_input_root() -> Any:
    reset_canonical_input_root_for_tests()
    yield
    reset_canonical_input_root_for_tests()


@dataclass(slots=True)
class _SharedKVState:
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    writes: list[dict[str, dict[str, Any]]] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class _ProcessKV:
    def __init__(
        self,
        state: _SharedKVState,
        *,
        row_kind: PipelineAttemptRowKind,
    ) -> None:
        self.state = state
        self.row_kind = row_kind
        self.fail_next_upsert: BaseException | None = None
        self.fail_next_callback: BaseException | None = None
        self.get_attempts = 0
        self.upsert_attempts = 0
        self.callback_attempts = 0
        self.cas_entered: asyncio.Event | None = None
        self.cas_release: asyncio.Event | None = None

    async def get_by_id(self, key: str) -> dict[str, Any] | None:
        self.get_attempts += 1
        async with self.state.lock:
            value = self.state.rows.get(key)
            return deepcopy(value) if value is not None else None

    async def upsert(self, values: dict[str, dict[str, Any]]) -> None:
        self.upsert_attempts += 1
        if self.fail_next_upsert is not None:
            error = self.fail_next_upsert
            self.fail_next_upsert = None
            raise error
        copied = deepcopy(values)
        async with self.state.lock:
            self.state.rows.update(copied)
            self.state.writes.append(copied)

    async def compare_and_commit_pipeline_attempt(
        self,
        key: str,
        payload: Mapping[str, Any],
        *,
        expected_attempt_token: str,
        row_kind: PipelineAttemptRowKind,
    ) -> bool:
        assert row_kind == self.row_kind
        self.upsert_attempts += 1
        if self.cas_entered is not None:
            self.cas_entered.set()
        if self.cas_release is not None:
            await self.cas_release.wait()
        if self.fail_next_upsert is not None:
            error = self.fail_next_upsert
            self.fail_next_upsert = None
            raise error
        copied = deepcopy(dict(payload))
        async with self.state.lock:
            current = self.state.rows.get(key)
            if (
                extract_pipeline_attempt_token(current, row_kind=row_kind)
                != expected_attempt_token
            ):
                return False
            self.state.rows[key] = copied
            self.state.writes.append({key: deepcopy(copied)})
            return True

    async def index_done_callback(self) -> None:
        self.callback_attempts += 1
        if self.fail_next_callback is not None:
            error = self.fail_next_callback
            self.fail_next_callback = None
            raise error


class _ProducerParseRAG(_ParserRAG):
    def __init__(self, full_docs: _ProcessKV) -> None:
        super().__init__()
        self.full_docs = full_docs
        self.kb_active_index_hash = "sha256:h2d-build-index"
        self.addon_params: dict[str, Any] = {}


class _RecoveryRAG:
    def __init__(
        self,
        *,
        full_docs: _ProcessKV,
        doc_status: _ProcessKV,
        process_root: Path,
    ) -> None:
        self.full_docs = full_docs
        self.doc_status = doc_status
        self.process_root = process_root
        self.materializer_calls = 0

    async def pipeline_artifact_materializer(self, binding: Any) -> None:
        del binding
        self.materializer_calls += 1
        raise AssertionError("recovery must not materialize artifacts")


@dataclass(slots=True)
class _ParsedBase:
    kb_id: str
    kb_service: KnowledgeBaseService
    metadata_store: SQLiteMetadataStore
    storage: _FakeObjectStorage
    document_service: DocumentLifecycleService
    producer_materializer: Any
    producer_rag: _ProducerParseRAG
    job_service: JobService
    plan: Any
    job: Any
    result: Any
    claimed_binding: PipelineArtifactBinding
    full_state: _SharedKVState
    status_state: _SharedKVState


@dataclass(slots=True)
class _CrashState:
    base: _ParsedBase
    operation: str
    binding: PipelineArtifactBinding
    committed_binding: PipelineArtifactBinding
    document: Any

    @property
    def lightrag_doc_id(self) -> str:
        return self.binding.lightrag_doc_id


@dataclass(slots=True)
class _RecoveryProcess:
    document_service: DocumentLifecycleService
    materializer: Any
    rag: _RecoveryRAG
    reconciler: PipelineArtifactTerminalizationReconciler
    full_docs: _ProcessKV
    doc_status: _ProcessKV
    lease_calls: list[None]


@dataclass(frozen=True, slots=True)
class _ReservationCall:
    """One durable reservation call recorded for test assertions.

    ``status_after``/``offset_after`` describe the in-memory cursor position
    after the reservation completed (the next page starts there). They are the
    durable-cursor analogue of the legacy per-status offset tracking.
    """

    kb_generation: str
    limit: int
    returned: int
    status_after: str
    offset_after: int
    sweep_after: int


class _RecoveryMetadataStore:
    """In-memory durable recovery-cursor store for reconciler tests.

    Mirrors the production store's keyset reservation contract: atomic
    version-CAS inside an ``asyncio.Lock`` (so two concurrent reservations
    never observe the same page), sweep-wrap semantics across ``parsed`` and
    ``ready``, and a persisted cursor row that survives across instances
    sharing the same state object.
    """

    def __init__(
        self,
        delegate: Any,
        *,
        rows_by_status: Mapping[str, list[DocumentRecord]],
        fail_next_reserve: BaseException | None = None,
        stale_generation: bool = False,
        block_first_reserve: bool = False,
    ) -> None:
        self._delegate = delegate
        self._rows_by_status = {
            status: list(rows) for status, rows in rows_by_status.items()
        }
        self._lock = asyncio.Lock()
        self._cursor_status = "parsed"
        self._cursor_offset = 0
        self._cursor_sweep = 0
        self._cursor_version = 1
        self._cursor_exists = False
        self._fail_next_reserve = fail_next_reserve
        self._stale_generation = stale_generation
        self._block_first_reserve = block_first_reserve
        self._did_block = False
        self.block_entered = asyncio.Event()
        self.block_release = asyncio.Event()
        self.calls: list[_ReservationCall] = []
        self.delete_calls: list[tuple[str, str]] = []
        self.get_cursor_calls: list[tuple[str, str]] = []

    async def reserve_pipeline_artifact_recovery_page(
        self,
        kb_id: str,
        kb_generation: str,
        limit: int,
    ) -> list[DocumentRecord]:
        del kb_id
        async with self._lock:
            if self._stale_generation:
                raise ArtifactRecoveryGenerationError()
            if self._fail_next_reserve is not None:
                error = self._fail_next_reserve
                self._fail_next_reserve = None
                raise error
            if self._block_first_reserve and not self._did_block:
                self._did_block = True
                # The cursor must NOT advance while the reservation is blocked:
                # the production store advances atomically inside its
                # transaction, so a cancelled reservation leaves no progress.
                self.block_entered.set()
                await self.block_release.wait()

            # The cursor row is created (or confirmed present) before paging,
            # matching the production store's INSERT ... ON CONFLICT DO NOTHING.
            self._cursor_exists = True
            self._cursor_version += 1

            selected: list[DocumentRecord] = []
            status = self._cursor_status
            offset = self._cursor_offset
            sweep = self._cursor_sweep
            remaining = limit
            while remaining > 0:
                rows = self._rows_by_status.get(status, [])
                page = list(rows[offset : offset + remaining])
                page_end = offset + len(page)
                total = len(rows)
                inspected = len(page)
                selected.extend(page)
                remaining -= inspected
                exhausted = inspected == 0 or page_end >= total
                if not exhausted:
                    offset = page_end
                    break
                if status == "ready":
                    status = "parsed"
                    offset = 0
                    sweep += 1
                    break
                status = "ready"
                offset = 0

            self._cursor_status = status
            self._cursor_offset = offset
            self._cursor_sweep = sweep
            self.calls.append(
                _ReservationCall(
                    kb_generation=kb_generation,
                    limit=limit,
                    returned=len(selected),
                    status_after=status,
                    offset_after=offset,
                    sweep_after=sweep,
                )
            )
            await asyncio.sleep(0)
            return selected

    async def get_artifact_recovery_cursor(
        self,
        kb_id: str,
        kb_generation: str,
    ) -> ArtifactRecoveryCursorRecord | None:
        self.get_cursor_calls.append((kb_id, kb_generation))
        if not self._cursor_exists:
            return None
        return ArtifactRecoveryCursorRecord(
            kb_id=kb_id,
            kb_generation=kb_generation,
            status=cast(Any, self._cursor_status),
            last_created_at=None,
            last_document_id=None,
            sweep=self._cursor_sweep,
            version=self._cursor_version,
            updated_at="2026-08-04T00:00:00+00:00",
        )

    async def delete_artifact_recovery_cursor(
        self,
        kb_id: str,
        kb_generation: str,
    ) -> bool:
        self.delete_calls.append((kb_id, kb_generation))
        if not self._cursor_exists:
            return False
        self._cursor_exists = False
        # Reset to the initial sweep position so a later reservation on the
        # same fixture starts a fresh sweep, matching the production store
        # which would re-create the cursor at parsed/offset-0/sweep-0.
        self._cursor_status = "parsed"
        self._cursor_offset = 0
        self._cursor_sweep = 0
        return True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class _CursorDocumentService:
    object_authoritative = True
    object_storage = object()

    def __init__(self, metadata_store: _RecoveryMetadataStore) -> None:
        self.metadata_store = metadata_store


class _CursorRAG:
    def __init__(self, full_docs: _ProcessKV, doc_status: _ProcessKV) -> None:
        self.full_docs = full_docs
        self.doc_status = doc_status


class _RecordingRecoveryReconciler(PipelineArtifactTerminalizationReconciler):
    def __init__(self, document_service: Any) -> None:
        super().__init__(document_service)
        self.examined_document_ids: list[str] = []

    async def _reconcile_candidate(
        self,
        kb_id: str,
        candidate: Any,
        *,
        full_docs: Any,
        doc_status: Any,
    ) -> None:
        del kb_id, full_docs, doc_status
        await asyncio.sleep(0)
        self.examined_document_ids.append(candidate.document_id)


def _synthetic_document(
    *,
    kb_id: str,
    status: str,
    index: int,
) -> DocumentRecord:
    document_id = f"document-{status}-{index:04d}"
    return DocumentRecord(
        id=document_id,
        kb_id=kb_id,
        workspace=f"workspace-{kb_id}",
        lightrag_doc_id=f"doc-{status}-{index:04d}",
        source_type="upload",
        source_name=f"{document_id}.pdf",
        source_uri=f"upload://{document_id}",
        source_hash=f"sha256:{index:064x}",
        content_type="application/pdf",
        size_bytes=1,
        parser_hash="sha256:" + "a" * 64,
        index_hash="sha256:" + "b" * 64 if status == "ready" else None,
        status=status,
        enabled=True,
        archived=False,
        chunks_count=1 if status == "ready" else None,
        entity_count=1 if status == "ready" else None,
        relation_count=1 if status == "ready" else None,
        error_code=None,
        error_message=None,
        metadata={},
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        deleted_at=None,
    )


def _synthetic_claimed_binding(document: DocumentRecord) -> PipelineArtifactBinding:
    operation = "parse" if document.status == "parsed" else "build"
    claim_token = f"claim-{document.id}"
    sidecar_id = f"sidecar-{document.id}" if operation == "build" else None
    blocks_id = f"blocks-{document.id}" if operation == "build" else None
    return PipelineArtifactBinding(
        version=1,
        authority="kb_metadata",
        state="claimed",
        operation=operation,
        kb_id=document.kb_id,
        kb_generation="kb-generation-cursor",
        workspace=document.workspace,
        document_id=document.id,
        lightrag_doc_id=document.lightrag_doc_id or "",
        job_id=f"job-{document.id}",
        claim_token=claim_token,
        source_hash=document.source_hash,
        parser_hash=document.parser_hash,
        parse_generation_id=(
            claim_token if operation == "parse" else "parse-generation-cursor"
        ),
        index_hash=document.index_hash,
        sidecar_artifact_id=sidecar_id,
        blocks_artifact_id=blocks_id,
        expected_current_sidecar_artifact_id=sidecar_id,
        expected_current_blocks_artifact_id=blocks_id,
        raw_artifact_ids=(),
    )


async def _prepare_parsed_base(tmp_path: Path, *, kb_id: str) -> _ParsedBase:
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kbs.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    kb_record = await kb_service.create(kb_id=kb_id, name=kb_id)
    # Register the lifecycle row so the reconciler can resolve the KB
    # generation for the durable recovery cursor reservation.
    await metadata_store.activate_kb_generation(kb_id, kb_record.generation)
    storage = _FakeObjectStorage()
    document_service, producer_materializer = _build_object_service(
        root=tmp_path / "producer" / "inputs",
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
    )
    document, _upload_job = await _create_document(
        document_service,
        kb_id,
        source_name="recovery.pdf",
        content=b"pipeline artifact recovery",
    )
    plan = await document_service.create_parse_plan(
        kb_id,
        document.id,
        parser_engine="mineru",
    )
    job_service = JobService(kb_service, metadata_store)
    job, _created = await job_service.create_parse_job_once(
        kb_id,
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
        force_reparse=False,
    )
    await document_service.mark_parse_queued(
        kb_id,
        document.id,
        job=job,
        plan=plan,
    )
    await document_service.mark_parse_running(
        kb_id,
        document.id,
        job_id=job.id,
        claim_token=plan.claim_token,
        plan=plan,
    )

    full_state = _SharedKVState()
    status_state = _SharedKVState()
    producer_rag = _ProducerParseRAG(_ProcessKV(full_state, row_kind="full_docs"))
    execution = await document_service.materialize_parse_execution(plan)
    try:
        parsed_data = await document_service.run_parse(producer_rag, plan, execution)
        await document_service.finalize_parse_runtime_references(
            producer_rag,
            plan,
            execution,
            parsed_data,
        )
        result = await document_service.complete_parse(
            kb_id,
            document.id,
            job_id=job.id,
            plan=plan,
            execution=execution,
            parsed_data=parsed_data,
        )
    finally:
        execution.cleanup()

    full_doc = full_state.rows[plan.lightrag_doc_id]
    claimed_binding = PipelineArtifactBinding.from_mapping(
        full_doc["artifact_binding"],
        expected_workspace=plan.document.workspace,
    )
    assert claimed_binding.state == "claimed"
    assert result.document.status == "parsed"
    assert not list(producer_materializer.scratch_root.iterdir())
    return _ParsedBase(
        kb_id=kb_id,
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
        document_service=document_service,
        producer_materializer=producer_materializer,
        producer_rag=producer_rag,
        job_service=job_service,
        plan=plan,
        job=job,
        result=result,
        claimed_binding=claimed_binding,
        full_state=full_state,
        status_state=status_state,
    )


def _seed_status(state: _CrashState) -> None:
    full_doc = state.base.full_state.rows[state.lightrag_doc_id]
    state.base.status_state.rows[state.lightrag_doc_id] = {
        "status": "processing",
        "content_summary": "parsed",
        "content_length": 6,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "file_path": full_doc["file_path"],
        "track_id": state.binding.job_id,
        "chunks_list": ["chunk-existing"],
        "metadata": {
            "pipeline_attempt_token": state.binding.claim_token,
            "process_options": full_doc.get("process_options", ""),
        },
    }


async def _prepare_parse_crash(tmp_path: Path, *, kb_id: str) -> _CrashState:
    base = await _prepare_parsed_base(tmp_path, kb_id=kb_id)
    document = base.result.document
    committed = base.claimed_binding.committed(
        parse_generation_id=base.claimed_binding.claim_token,
        index_hash=document.index_hash,
        sidecar_artifact_id=document.metadata.get("current_sidecar_artifact_id"),
        blocks_artifact_id=document.metadata.get("current_blocks_artifact_id"),
        raw_artifact_ids=tuple(
            sorted(
                artifact.id
                for artifact in base.result.artifacts
                if artifact.artifact_type == "raw_dir"
            )
        ),
    )
    state = _CrashState(
        base=base,
        operation="parse",
        binding=base.claimed_binding,
        committed_binding=committed,
        document=document,
    )
    _seed_status(state)
    return state


async def _prepare_build_crash(
    tmp_path: Path,
    *,
    kb_id: str,
    mutate_artifacts: bool = False,
) -> _CrashState:
    base = await _prepare_parsed_base(tmp_path, kb_id=kb_id)
    await base.document_service.commit_parse_artifact_binding(
        base.producer_rag,
        base.plan,
        base.result,
    )
    index_service = IndexBuildService(base.document_service)
    plan = await index_service.create_build_plan(
        kb_id,
        base.plan.document.id,
        rag=base.producer_rag,
        force_rechunk=True,
    )
    job, _created = await base.job_service.create_build_job_once(
        kb_id,
        document_id=plan.document.id,
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
        force_rechunk=True,
        force_extract=False,
        force_embedding=False,
    )
    await index_service.claim_build_queued(kb_id, job_id=job.id, plan=plan)
    await index_service.mark_building(
        kb_id,
        plan.document.id,
        job_id=job.id,
        claim_token=plan.claim_token,
        plan=plan,
    )
    binding = index_service.build_artifact_binding(plan)
    existing = base.full_state.rows[binding.lightrag_doc_id]
    base.full_state.rows[binding.lightrag_doc_id] = {
        "content": existing["content"],
        "file_path": f"{plan.document.id}__{plan.document.source_name}",
        "parse_format": "lightrag",
        "parse_engine": existing.get("parse_engine", "mineru"),
        "process_options": existing.get("process_options", ""),
        "chunk_options": {"chunk_token_size": 1200},
        "artifact_binding": binding.to_dict(),
    }

    coordinator = PipelineArtifactCoordinator(
        base.kb_service,
        base.document_service,
        index_service,
    )
    session = await coordinator.open(binding)
    try:
        if mutate_artifacts:
            assert session.blocks_path is not None
            with session.blocks_path.open("a", encoding="utf-8") as blocks_file:
                blocks_file.write('{"type":"content","text":"h2d mutation"}\n')
        finalization = await session.handoff_success(
            parsed_data={"entity_count": 5, "relation_count": 4},
            chunks_count=7,
        )
    finally:
        await session.aclose()
    assert finalization.outcome is PipelineArtifactCommitOutcome.COMMITTED
    assert finalization.committed_binding is not None
    document = await base.metadata_store.get_document(kb_id, binding.document_id)
    assert document.status == "ready"
    assert document.chunks_count == 7
    assert document.entity_count == 5
    assert document.relation_count == 4
    assert not list(base.producer_materializer.scratch_root.iterdir())
    state = _CrashState(
        base=base,
        operation="build",
        binding=binding,
        committed_binding=finalization.committed_binding,
        document=document,
    )
    _seed_status(state)
    return state


def _new_recovery_process(
    state: _CrashState,
    tmp_path: Path,
    *,
    name: str,
) -> _RecoveryProcess:
    root = tmp_path / name / "inputs"
    document_service, materializer = _build_object_service(
        root=root,
        kb_service=state.base.kb_service,
        metadata_store=state.base.metadata_store,
        storage=state.base.storage,
    )
    full_docs = _ProcessKV(state.base.full_state, row_kind="full_docs")
    doc_status = _ProcessKV(state.base.status_state, row_kind="doc_status")
    rag = _RecoveryRAG(
        full_docs=full_docs,
        doc_status=doc_status,
        process_root=root,
    )
    lease_calls: list[None] = []

    def forbidden_create_lease() -> None:
        lease_calls.append(None)
        raise AssertionError("recovery must not create a scratch lease")

    materializer_for_test: Any = materializer
    materializer_for_test.create_lease = forbidden_create_lease
    return _RecoveryProcess(
        document_service=document_service,
        materializer=materializer,
        rag=rag,
        reconciler=PipelineArtifactTerminalizationReconciler(document_service),
        full_docs=full_docs,
        doc_status=doc_status,
        lease_calls=lease_calls,
    )


def _binding_from_full_docs(state: _CrashState) -> PipelineArtifactBinding:
    return PipelineArtifactBinding.from_mapping(
        state.base.full_state.rows[state.lightrag_doc_id]["artifact_binding"],
        expected_workspace=state.binding.workspace,
    )


def _status_binding(state: _CrashState) -> PipelineArtifactBinding:
    metadata = state.base.status_state.rows[state.lightrag_doc_id]["metadata"]
    return PipelineArtifactBinding.from_mapping(
        metadata["artifact_binding"],
        expected_workspace=state.binding.workspace,
    )


def _assert_no_runtime_locators(value: Any) -> None:
    encoded = json.dumps(value, ensure_ascii=False, default=str)
    for forbidden in (
        ".lightrag-scratch",
        "sidecar_location",
        "blocks_path",
        "runtime_source_path",
        "object_uri",
        "object_prefix_uri",
        "presigned_url",
        "file://",
    ):
        assert forbidden not in encoded


async def _patch_document(
    state: _CrashState,
    *,
    status: str | None = None,
    metadata_patch: dict[str, Any] | None = None,
) -> None:
    def mutate(conn: Any) -> None:
        row = conn.execute(
            "SELECT metadata_json FROM documents WHERE kb_id = ? AND id = ?",
            (state.binding.kb_id, state.binding.document_id),
        ).fetchone()
        metadata = _loads_json_object(row["metadata_json"])
        metadata.update(metadata_patch or {})
        if status is None:
            conn.execute(
                "UPDATE documents SET metadata_json = ? WHERE kb_id = ? AND id = ?",
                (
                    _dumps_json(metadata),
                    state.binding.kb_id,
                    state.binding.document_id,
                ),
            )
        else:
            conn.execute(
                "UPDATE documents SET status = ?, metadata_json = ? "
                "WHERE kb_id = ? AND id = ?",
                (
                    status,
                    _dumps_json(metadata),
                    state.binding.kb_id,
                    state.binding.document_id,
                ),
            )

    await state.base.metadata_store._write(mutate)


async def test_ready_cursor_eventually_reaches_late_claimed_build_and_wraps(
    tmp_path: Path,
) -> None:
    state = await _prepare_build_crash(tmp_path, kb_id="kb_h2d_ready_cursor")
    assert state.binding.state == "claimed"
    assert state.binding.operation == "build"
    process = _new_recovery_process(state, tmp_path, name="recovery-ready-cursor")

    ready_rows = [
        replace(
            state.document,
            id=f"ready-dummy-{index:04d}",
            lightrag_doc_id=f"doc-ready-dummy-{index:04d}",
            source_name=f"ready-dummy-{index:04d}.pdf",
        )
        for index in range(451)
    ]
    target_index = 425
    ready_rows[target_index] = state.document
    metadata = _RecoveryMetadataStore(
        state.base.metadata_store,
        rows_by_status={"parsed": [], "ready": ready_rows},
    )
    process.reconciler._metadata_store = metadata  # type: ignore[assignment]
    kb_generation = state.binding.kb_generation

    summaries = []
    for _ in range(6):
        summary = await process.reconciler.reconcile_kb(
            state.base.kb_id,
            process.rag,
            limit=100,
            kb_generation=kb_generation,
        )
        summaries.append(summary)
        assert (
            summary.finalized + summary.skipped + summary.error_count
            == summary.discovered
        )
        assert summary.discovered <= 100

    assert sum(summary.finalized for summary in summaries) == 1
    assert state.base.status_state.rows[state.lightrag_doc_id]["status"] == "processed"
    # The durable cursor advances through the ready keyset and wraps to parsed
    # after exhausting ready (call 5), then starts a fresh sweep (call 6).
    status_after = [call.status_after for call in metadata.calls]
    offset_after = [call.offset_after for call in metadata.calls]
    assert status_after == ["ready", "ready", "ready", "ready", "parsed", "ready"]
    assert offset_after == [100, 200, 300, 400, 0, 100]
    assert metadata.calls[4].sweep_after == 1
    assert metadata.calls[5].sweep_after == 1
    assert target_index > 400
    # Cursor row is NOT deleted: the reservation still returns rows each sweep
    # (dummy documents remain in ready), so the KB is not drained.
    assert all(call.returned > 0 for call in metadata.calls)
    assert metadata.delete_calls == []


async def test_balanced_status_cursor_eventually_examines_both_late_classes() -> None:
    kb_id = "kb_h2d_balanced_cursor"
    parsed_rows = [
        _synthetic_document(kb_id=kb_id, status="parsed", index=index)
        for index in range(230)
    ]
    ready_rows = [
        _synthetic_document(kb_id=kb_id, status="ready", index=index)
        for index in range(230)
    ]
    late_parsed = parsed_rows[220]
    late_ready = ready_rows[220]
    metadata = _RecoveryMetadataStore(
        None,
        rows_by_status={"parsed": parsed_rows, "ready": ready_rows},
    )
    reconciler = _RecordingRecoveryReconciler(_CursorDocumentService(metadata))
    full_state = _SharedKVState(
        rows={
            late_parsed.lightrag_doc_id or "": {
                "artifact_binding": _synthetic_claimed_binding(late_parsed).to_dict()
            },
            late_ready.lightrag_doc_id or "": {
                "artifact_binding": _synthetic_claimed_binding(late_ready).to_dict()
            },
        }
    )
    rag = _CursorRAG(
        _ProcessKV(full_state, row_kind="full_docs"),
        _ProcessKV(_SharedKVState(), row_kind="doc_status"),
    )
    kb_generation = "kb-generation-cursor"

    summaries = []
    for _ in range(5):
        summary = await reconciler.reconcile_kb(
            kb_id, rag, limit=100, kb_generation=kb_generation
        )
        summaries.append(summary)
        assert summary.discovered <= 100
        assert (
            summary.finalized + summary.skipped + summary.error_count
            == summary.discovered
        )

    assert reconciler.examined_document_ids == [late_parsed.id, late_ready.id]
    assert sum(summary.finalized for summary in summaries) == 2
    # Durable cursor walks parsed first (3 pages: 0-100, 100-200, 200-230),
    # then wraps into ready for the remaining budget of each sweep.
    # Sweep 1: parsed[0:100] (100 parsed).        status_after=parsed, offset=100
    # Sweep 2: parsed[100:200] (100 parsed).      status_after=parsed, offset=200
    # Sweep 3: parsed[200:230] (30) -> ready[0:70] (70). status_after=ready, offset=70
    # Sweep 4: ready[70:170] (100 ready).         status_after=ready, offset=170
    # Sweep 5: ready[170:230] (60) -> wrap.        status_after=parsed, offset=0, sweep=1
    assert [(call.status_after, call.offset_after) for call in metadata.calls] == [
        ("parsed", 100),
        ("parsed", 200),
        ("ready", 70),
        ("ready", 170),
        ("parsed", 0),
    ]
    assert metadata.calls[4].sweep_after == 1
    assert all(0 < call.limit <= 100 for call in metadata.calls)
    assert metadata.delete_calls == []


async def test_concurrent_cursor_windows_dedupe_via_store_cas(tmp_path: Path) -> None:
    state = await _prepare_build_crash(tmp_path, kb_id="kb_h2d_cursor_concurrent")
    process = _new_recovery_process(
        state,
        tmp_path,
        name="recovery-cursor-concurrent",
    )
    ready_rows = [
        replace(
            state.document,
            id=f"concurrent-dummy-{index:04d}",
            lightrag_doc_id=f"doc-concurrent-dummy-{index:04d}",
            source_name=f"concurrent-dummy-{index:04d}.pdf",
        )
        for index in range(250)
    ]
    # The target appears twice in the seeded rows; the durable store CAS must
    # still hand each reconciler a disjoint page so the candidate is examined
    # at most once.
    ready_rows[50] = state.document
    ready_rows[75] = state.document
    metadata = _RecoveryMetadataStore(
        state.base.metadata_store,
        rows_by_status={"parsed": [], "ready": ready_rows},
    )
    process.reconciler._metadata_store = metadata  # type: ignore[assignment]
    kb_generation = state.binding.kb_generation
    full_writes_before = len(state.base.full_state.writes)
    status_writes_before = len(state.base.status_state.writes)

    first, second = await asyncio.gather(
        process.reconciler.reconcile_kb(
            state.base.kb_id,
            process.rag,
            limit=100,
            kb_generation=kb_generation,
        ),
        process.reconciler.reconcile_kb(
            state.base.kb_id,
            process.rag,
            limit=100,
            kb_generation=kb_generation,
        ),
    )

    # The store-level CAS guarantees the two reservations are serial: each
    # returns its own 100-row page, disjoint from the other. The reconciler
    # then deduplicates the repeated target id within the page that contains
    # both copies, so the discovered totals sum to 199 (100 + 100 - 1 dup).
    assert first.discovered + second.discovered == 199
    assert {first.discovered, second.discovered} == {99, 100}
    assert first.error_count == second.error_count == 0
    assert first.finalized + second.finalized == 1
    assert first.skipped + second.skipped == 198
    assert [call.offset_after for call in metadata.calls] == [100, 200]
    assert all(call.limit == 100 for call in metadata.calls)
    assert len(state.base.full_state.writes) == full_writes_before + 1
    assert len(state.base.status_state.writes) == status_writes_before + 1
    assert state.base.status_state.rows[state.lightrag_doc_id]["status"] == "processed"


async def test_cancelled_metadata_reservation_leaves_durable_cursor_unchanged() -> None:
    kb_id = "kb_h2d_cursor_cancelled"
    parsed_rows = [
        _synthetic_document(kb_id=kb_id, status="parsed", index=index)
        for index in range(250)
    ]
    metadata = _RecoveryMetadataStore(
        None,
        rows_by_status={"parsed": parsed_rows, "ready": []},
        block_first_reserve=True,
    )
    reconciler = _RecordingRecoveryReconciler(_CursorDocumentService(metadata))
    rag = _CursorRAG(
        _ProcessKV(_SharedKVState(), row_kind="full_docs"),
        _ProcessKV(_SharedKVState(), row_kind="doc_status"),
    )
    kb_generation = "kb-generation-cursor"

    cancelled = asyncio.create_task(
        reconciler.reconcile_kb(kb_id, rag, limit=100, kb_generation=kb_generation)
    )
    await asyncio.wait_for(metadata.block_entered.wait(), timeout=2)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled

    # The durable reservation is atomic: a cancelled call does NOT advance the
    # cursor row. The next call must therefore resume from the initial position
    # rather than skipping the blocked page.
    cursor_after_cancel = await metadata.get_artifact_recovery_cursor(
        kb_id, kb_generation
    )
    assert cursor_after_cancel is None
    assert metadata.calls == []

    summary = await reconciler.reconcile_kb(
        kb_id, rag, limit=100, kb_generation=kb_generation
    )

    assert summary.discovered == summary.skipped == 100
    assert summary.finalized == summary.error_count == 0
    assert [(call.status_after, call.offset_after) for call in metadata.calls] == [
        ("parsed", 100),
    ]
    cursor_after_resume = await metadata.get_artifact_recovery_cursor(
        kb_id, kb_generation
    )
    assert cursor_after_resume is not None
    assert cursor_after_resume.status == "parsed"
    assert cursor_after_resume.sweep == 0


async def test_restart_resume_reads_durable_cursor_not_offset_zero() -> None:
    """A fresh reconciler instance must resume from the persisted cursor row."""

    kb_id = "kb_h2d_cursor_restart"
    parsed_rows = [
        _synthetic_document(kb_id=kb_id, status="parsed", index=index)
        for index in range(350)
    ]
    metadata = _RecoveryMetadataStore(
        None,
        rows_by_status={"parsed": parsed_rows, "ready": []},
    )
    # First "process" reserves one page and advances the durable cursor.
    first_reconciler = _RecordingRecoveryReconciler(_CursorDocumentService(metadata))
    first_rag = _CursorRAG(
        _ProcessKV(_SharedKVState(), row_kind="full_docs"),
        _ProcessKV(_SharedKVState(), row_kind="doc_status"),
    )
    kb_generation = "kb-generation-cursor"
    first_summary = await first_reconciler.reconcile_kb(
        kb_id, first_rag, limit=100, kb_generation=kb_generation
    )
    assert first_summary.discovered == 100
    assert metadata.calls[0].offset_after == 100

    # A brand-new reconciler instance (simulating process restart) reads the
    # SAME durable cursor row from the store and continues at offset 100, not
    # offset 0. The process-local dict is gone; the store is the single source
    # of truth.
    second_reconciler = _RecordingRecoveryReconciler(_CursorDocumentService(metadata))
    second_rag = _CursorRAG(
        _ProcessKV(_SharedKVState(), row_kind="full_docs"),
        _ProcessKV(_SharedKVState(), row_kind="doc_status"),
    )
    second_summary = await second_reconciler.reconcile_kb(
        kb_id, second_rag, limit=100, kb_generation=kb_generation
    )
    assert second_summary.discovered == 100
    assert len(metadata.calls) == 2
    assert metadata.calls[1].offset_after == 200
    assert metadata.calls[1].status_after == "parsed"
    # The cursor row survived the instance swap.
    cursor = await metadata.get_artifact_recovery_cursor(kb_id, kb_generation)
    assert cursor is not None
    assert cursor.status == "parsed"
    assert cursor.sweep == 0


async def test_stale_generation_skips_kb_without_crashing_sweep() -> None:
    """ArtifactRecoveryGenerationError is caught and the KB is skipped."""

    kb_id = "kb_h2d_cursor_stale_generation"
    parsed_rows = [
        _synthetic_document(kb_id=kb_id, status="parsed", index=index)
        for index in range(10)
    ]
    metadata = _RecoveryMetadataStore(
        None,
        rows_by_status={"parsed": parsed_rows, "ready": []},
        stale_generation=True,
    )
    reconciler = _RecordingRecoveryReconciler(_CursorDocumentService(metadata))
    rag = _CursorRAG(
        _ProcessKV(_SharedKVState(), row_kind="full_docs"),
        _ProcessKV(_SharedKVState(), row_kind="doc_status"),
    )

    summary = await reconciler.reconcile_kb(
        kb_id, rag, limit=100, kb_generation="stale-generation"
    )

    assert summary.discovered == 0
    assert summary.finalized == 0
    assert summary.skipped == 0
    assert summary.error_count == 0
    # The store raised before any page was returned; no terminalization ran.
    assert reconciler.examined_document_ids == []
    assert metadata.delete_calls == []


async def test_durable_cursor_deleted_after_kb_generation_drains() -> None:
    """An empty reservation triggers cursor row cleanup."""

    kb_id = "kb_h2d_cursor_drained"
    metadata = _RecoveryMetadataStore(
        None,
        rows_by_status={"parsed": [], "ready": []},
    )
    reconciler = _RecordingRecoveryReconciler(_CursorDocumentService(metadata))
    rag = _CursorRAG(
        _ProcessKV(_SharedKVState(), row_kind="full_docs"),
        _ProcessKV(_SharedKVState(), row_kind="doc_status"),
    )
    kb_generation = "kb-generation-cursor"

    summary = await reconciler.reconcile_kb(
        kb_id, rag, limit=100, kb_generation=kb_generation
    )

    assert summary.discovered == 0
    assert summary.finalized == 0
    assert summary.skipped == 0
    assert summary.error_count == 0
    # The reservation returned zero rows -> the reconciler deletes the cursor.
    assert metadata.delete_calls == [(kb_id, kb_generation)]
    cursor_after = await metadata.get_artifact_recovery_cursor(kb_id, kb_generation)
    assert cursor_after is None


async def test_concurrent_two_store_instances_never_overlap_pages() -> None:
    """Two independent store instances (processes) get disjoint pages via CAS."""

    kb_id = "kb_h2d_cursor_cross_process"
    ready_rows = [
        _synthetic_document(kb_id=kb_id, status="ready", index=index)
        for index in range(460)
    ]
    # Both instances share the same in-memory cursor state, mirroring two
    # processes sharing the same durable cursor table.
    shared_metadata = _RecoveryMetadataStore(
        None,
        rows_by_status={"parsed": [], "ready": ready_rows},
    )

    class _TwoDocService:
        object_authoritative = True
        object_storage = object()

        def __init__(self, store: Any) -> None:
            self.metadata_store = store

    reconciler_a = _RecordingRecoveryReconciler(_TwoDocService(shared_metadata))
    reconciler_b = _RecordingRecoveryReconciler(_TwoDocService(shared_metadata))
    rag_a = _CursorRAG(
        _ProcessKV(_SharedKVState(), row_kind="full_docs"),
        _ProcessKV(_SharedKVState(), row_kind="doc_status"),
    )
    rag_b = _CursorRAG(
        _ProcessKV(_SharedKVState(), row_kind="full_docs"),
        _ProcessKV(_SharedKVState(), row_kind="doc_status"),
    )
    kb_generation = "kb-generation-cursor"

    summary_a, summary_b = await asyncio.gather(
        reconciler_a.reconcile_kb(kb_id, rag_a, limit=120, kb_generation=kb_generation),
        reconciler_b.reconcile_kb(kb_id, rag_b, limit=120, kb_generation=kb_generation),
    )

    assert summary_a.discovered == 120
    assert summary_b.discovered == 120
    # The two reservations advanced the same shared cursor by 240 rows total,
    # proving they did not observe the same page.
    assert len(shared_metadata.calls) == 2
    assert {
        shared_metadata.calls[0].offset_after,
        shared_metadata.calls[1].offset_after,
    } == {
        120,
        240,
    }
    assert all(call.status_after == "ready" for call in shared_metadata.calls)


async def test_missing_doc_status_token_is_not_created_by_recovery_cas(
    tmp_path: Path,
) -> None:
    state = await _prepare_parse_crash(tmp_path, kb_id="kb_h2d_parse")
    state.base.status_state.rows.pop(state.lightrag_doc_id)
    process = _new_recovery_process(state, tmp_path, name="recovery-parse")
    downloads_before = (
        len(state.base.storage.file_downloads),
        len(state.base.storage.prefix_downloads),
    )
    deletes_before = (
        list(state.base.storage.deleted_files),
        list(state.base.storage.deleted_prefixes),
    )
    scratch_before = list(process.materializer.scratch_root.iterdir())

    summary = await process.reconciler.reconcile_kb(state.base.kb_id, process.rag)

    assert summary.discovered == 1
    assert summary.finalized == 0
    assert summary.skipped == 1
    assert summary.error_count == 0
    assert _binding_from_full_docs(state) == state.committed_binding
    assert state.lightrag_doc_id not in state.base.status_state.rows
    assert process.full_docs.callback_attempts == 0
    assert process.rag.materializer_calls == 0
    assert process.lease_calls == []
    assert (
        len(state.base.storage.file_downloads),
        len(state.base.storage.prefix_downloads),
    ) == downloads_before
    assert (
        state.base.storage.deleted_files,
        state.base.storage.deleted_prefixes,
    ) == deletes_before
    assert list(process.materializer.scratch_root.iterdir()) == scratch_before == []
    _assert_no_runtime_locators(state.base.full_state.rows)
    _assert_no_runtime_locators(state.base.status_state.rows)


@pytest.mark.parametrize("mutate_artifacts", [False, True])
async def test_committed_build_claimed_binding_uses_durable_counts(
    tmp_path: Path,
    mutate_artifacts: bool,
) -> None:
    state = await _prepare_build_crash(
        tmp_path,
        kb_id=f"kb_h2d_build_{mutate_artifacts}",
        mutate_artifacts=mutate_artifacts,
    )
    process = _new_recovery_process(state, tmp_path, name="recovery-build")
    downloads_before = (
        len(state.base.storage.file_downloads),
        len(state.base.storage.prefix_downloads),
    )

    summary = await process.reconciler.reconcile_kb(state.base.kb_id, process.rag)

    assert summary.finalized == 1
    assert summary.error_count == 0
    assert _binding_from_full_docs(state) == state.committed_binding
    status = state.base.status_state.rows[state.lightrag_doc_id]
    assert status["status"] == "processed"
    assert status["chunks_count"] == state.document.chunks_count == 7
    assert status["entity_count"] == state.document.entity_count == 5
    assert status["relation_count"] == state.document.relation_count == 4
    assert status["metadata"]["chunks_count"] == 7
    assert status["metadata"]["entity_count"] == 5
    assert status["metadata"]["relation_count"] == 4
    assert _status_binding(state) == state.committed_binding
    assert process.full_docs.callback_attempts == 0
    if mutate_artifacts:
        assert (
            state.committed_binding.sidecar_artifact_id
            != state.binding.expected_current_sidecar_artifact_id
        )
        assert (
            state.committed_binding.blocks_artifact_id
            != state.binding.expected_current_blocks_artifact_id
        )
    else:
        assert (
            state.committed_binding.sidecar_artifact_id
            == state.binding.expected_current_sidecar_artifact_id
        )
        assert (
            state.committed_binding.blocks_artifact_id
            == state.binding.expected_current_blocks_artifact_id
        )
    assert (
        len(state.base.storage.file_downloads),
        len(state.base.storage.prefix_downloads),
    ) == downloads_before
    assert not list(process.materializer.scratch_root.iterdir())
    _assert_no_runtime_locators(state.base.full_state.rows)
    _assert_no_runtime_locators(state.base.status_state.rows)


async def test_status_failure_after_binding_patch_retries_successfully(
    tmp_path: Path,
) -> None:
    state = await _prepare_parse_crash(tmp_path, kb_id="kb_h2d_status_retry")
    process = _new_recovery_process(state, tmp_path, name="recovery-status-retry")
    process.doc_status.fail_next_upsert = RuntimeError(
        "failed at /tmp/.lightrag-scratch/secret using "
        "s3://access:secret@example.invalid/private"
    )

    first = await process.reconciler.reconcile_kb(state.base.kb_id, process.rag)

    assert first.finalized == 0
    assert first.error_count == 1
    assert first.errors[0].stage == "doc_status_write"
    assert ".lightrag-scratch" not in first.errors[0].message
    assert "access:secret" not in first.errors[0].message
    assert _binding_from_full_docs(state) == state.committed_binding
    assert state.base.status_state.rows[state.lightrag_doc_id]["status"] == "processing"
    full_docs_writes = process.full_docs.upsert_attempts
    process.full_docs.fail_next_upsert = AssertionError(
        "already-committed full_docs must not be rewritten"
    )

    second = await process.reconciler.reconcile_kb(state.base.kb_id, process.rag)

    assert second.finalized == 1
    assert second.error_count == 0
    assert process.full_docs.upsert_attempts == full_docs_writes
    assert state.base.status_state.rows[state.lightrag_doc_id]["status"] == "processed"
    assert _status_binding(state) == state.committed_binding


async def test_unknown_full_docs_cas_is_an_error_and_is_not_retried(
    tmp_path: Path,
) -> None:
    state = await _prepare_parse_crash(tmp_path, kb_id="kb_h2d_unknown_cas")
    process = _new_recovery_process(state, tmp_path, name="recovery-unknown-cas")
    process.full_docs.fail_next_upsert = PipelineAttemptCommitOutcomeUnknownError(
        state.lightrag_doc_id,
        row_kind="full_docs",
    )

    summary = await process.reconciler.reconcile_kb(state.base.kb_id, process.rag)

    assert summary.finalized == 0
    assert summary.skipped == 0
    assert summary.error_count == 1
    assert summary.errors[0].stage == "full_docs_write"
    assert summary.errors[0].error_code == "committed_binding_write_outcome_unknown"
    assert process.full_docs.upsert_attempts == 1
    assert _binding_from_full_docs(state).state == "claimed"


async def test_new_owner_inside_fence_skips_stale_binding(tmp_path: Path) -> None:
    state = await _prepare_parse_crash(tmp_path, kb_id="kb_h2d_new_owner")
    await _patch_document(
        state,
        metadata_patch={
            "current_parse_job_id": "job-new-owner",
            "current_parse_claim_token": "claim-new-owner",
        },
    )
    process = _new_recovery_process(state, tmp_path, name="recovery-new-owner")

    summary = await process.reconciler.reconcile_kb(state.base.kb_id, process.rag)

    assert summary.finalized == 0
    assert summary.skipped == 1
    assert summary.error_count == 0
    assert _binding_from_full_docs(state).state == "claimed"
    assert process.doc_status.upsert_attempts == 0


async def test_kb_generation_fence_skips_stale_binding(tmp_path: Path) -> None:
    state = await _prepare_parse_crash(tmp_path, kb_id="kb_h2d_generation")
    stale = replace(state.binding, kb_generation="stale-kb-generation")
    state.base.full_state.rows[state.lightrag_doc_id]["artifact_binding"] = (
        stale.to_dict()
    )
    state.base.status_state.rows[state.lightrag_doc_id]["metadata"][
        "pipeline_attempt_token"
    ] = stale.claim_token
    process = _new_recovery_process(state, tmp_path, name="recovery-generation")

    summary = await process.reconciler.reconcile_kb(state.base.kb_id, process.rag)

    assert summary.finalized == 0
    assert summary.skipped == 1
    assert summary.error_count == 0
    assert _binding_from_full_docs(state).state == "claimed"
    assert process.full_docs.upsert_attempts == 0
    assert process.doc_status.upsert_attempts == 0


@pytest.mark.parametrize(
    "damage",
    ["missing_blocks", "missing_raw", "invalid_checksum"],
)
async def test_partial_or_invalid_artifact_authority_is_skipped(
    tmp_path: Path,
    damage: str,
) -> None:
    state = await _prepare_parse_crash(
        tmp_path,
        kb_id=f"kb_h2d_partial_{damage}",
    )
    blocks_id = state.document.metadata["current_blocks_artifact_id"]
    if damage in {"missing_blocks", "missing_raw"}:
        artifact_id = blocks_id
        if damage == "missing_raw":
            artifact_id = next(
                artifact.id
                for artifact in state.base.result.artifacts
                if artifact.artifact_type == "raw_dir"
            )
        await state.base.metadata_store._write(
            lambda conn: conn.execute(
                "DELETE FROM document_artifacts WHERE kb_id = ? AND id = ?",
                (state.binding.kb_id, artifact_id),
            )
        )
    else:
        await state.base.metadata_store._write(
            lambda conn: conn.execute(
                "UPDATE document_artifacts SET checksum = ? WHERE kb_id = ? AND id = ?",
                ("unknown", state.binding.kb_id, blocks_id),
            )
        )
    process = _new_recovery_process(state, tmp_path, name=f"recovery-{damage}")

    summary = await process.reconciler.reconcile_kb(state.base.kb_id, process.rag)

    assert summary.finalized == 0
    assert summary.skipped == 1
    assert summary.error_count == 0
    assert _binding_from_full_docs(state).state == "claimed"
    assert process.full_docs.upsert_attempts == 0
    assert process.doc_status.upsert_attempts == 0


async def test_double_run_is_idempotent(tmp_path: Path) -> None:
    state = await _prepare_parse_crash(tmp_path, kb_id="kb_h2d_double")
    process = _new_recovery_process(state, tmp_path, name="recovery-double")

    first = await process.reconciler.reconcile_kb(state.base.kb_id, process.rag)
    writes_after_first = (
        len(state.base.full_state.writes),
        len(state.base.status_state.writes),
    )
    second = await process.reconciler.reconcile_kb(state.base.kb_id, process.rag)

    assert first.finalized == 1
    assert second.finalized == 0
    assert second.skipped == 1
    assert second.error_count == 0
    assert (
        len(state.base.full_state.writes),
        len(state.base.status_state.writes),
    ) == writes_after_first
    assert _binding_from_full_docs(state) == state.committed_binding
    assert _status_binding(state) == state.committed_binding


async def test_two_process_like_reconcilers_are_concurrent_safe(
    tmp_path: Path,
) -> None:
    state = await _prepare_build_crash(tmp_path, kb_id="kb_h2d_concurrent")
    process_a = _new_recovery_process(state, tmp_path, name="recovery-concurrent-a")
    process_b = _new_recovery_process(state, tmp_path, name="recovery-concurrent-b")

    first, second = await asyncio.gather(
        process_a.reconciler.reconcile_kb(state.base.kb_id, process_a.rag),
        process_b.reconciler.reconcile_kb(state.base.kb_id, process_b.rag),
    )

    assert first.error_count == second.error_count == 0
    assert first.finalized + second.finalized >= 1
    assert _binding_from_full_docs(state) == state.committed_binding
    assert _status_binding(state) == state.committed_binding
    status = state.base.status_state.rows[state.lightrag_doc_id]
    assert status["chunks_count"] == 7
    assert status["entity_count"] == 5
    assert status["relation_count"] == 4
    assert process_a.rag.materializer_calls == 0
    assert process_b.rag.materializer_calls == 0
    assert not list(process_a.materializer.scratch_root.iterdir())
    assert not list(process_b.materializer.scratch_root.iterdir())


async def test_revalidation_to_full_docs_cas_race_skips_newer_attempt(
    tmp_path: Path,
) -> None:
    state = await _prepare_parse_crash(tmp_path, kb_id="kb_h2d_cas_race")
    process = _new_recovery_process(state, tmp_path, name="recovery-cas-race")
    process.full_docs.cas_entered = asyncio.Event()
    process.full_docs.cas_release = asyncio.Event()

    reconcile = asyncio.create_task(
        process.reconciler.reconcile_kb(state.base.kb_id, process.rag)
    )
    await asyncio.wait_for(process.full_docs.cas_entered.wait(), timeout=2)

    newer_token = "h2d-attempt-newer"
    newer_binding = replace(
        state.binding,
        job_id="job-h2d-newer",
        claim_token=newer_token,
        parse_generation_id=newer_token,
    )
    async with state.base.full_state.lock:
        newer_full_doc = deepcopy(state.base.full_state.rows[state.lightrag_doc_id])
        newer_full_doc["artifact_binding"] = newer_binding.to_dict()
        state.base.full_state.rows[state.lightrag_doc_id] = newer_full_doc
    async with state.base.status_state.lock:
        newer_status = deepcopy(state.base.status_state.rows[state.lightrag_doc_id])
        newer_status["status"] = "pending"
        newer_status["metadata"]["pipeline_attempt_token"] = newer_token
        newer_status["metadata"].pop("artifact_binding", None)
        state.base.status_state.rows[state.lightrag_doc_id] = newer_status
    process.full_docs.cas_release.set()

    summary = await asyncio.wait_for(reconcile, timeout=2)

    assert summary.finalized == 0
    assert summary.skipped == 1
    assert summary.error_count == 0
    assert _binding_from_full_docs(state) == newer_binding
    status = state.base.status_state.rows[state.lightrag_doc_id]
    assert status["status"] == "pending"
    assert status["metadata"]["pipeline_attempt_token"] == newer_token


async def test_revalidation_to_doc_status_cas_race_skips_newer_attempt(
    tmp_path: Path,
) -> None:
    state = await _prepare_parse_crash(tmp_path, kb_id="kb_h2d_status_cas_race")
    process = _new_recovery_process(
        state,
        tmp_path,
        name="recovery-status-cas-race",
    )
    process.doc_status.cas_entered = asyncio.Event()
    process.doc_status.cas_release = asyncio.Event()

    reconcile = asyncio.create_task(
        process.reconciler.reconcile_kb(state.base.kb_id, process.rag)
    )
    await asyncio.wait_for(process.doc_status.cas_entered.wait(), timeout=2)
    assert _binding_from_full_docs(state) == state.committed_binding

    newer_token = "h2d-status-attempt-newer"
    newer_binding = replace(
        state.binding,
        job_id="job-h2d-status-newer",
        claim_token=newer_token,
        parse_generation_id=newer_token,
    )
    async with state.base.full_state.lock:
        newer_full_doc = deepcopy(state.base.full_state.rows[state.lightrag_doc_id])
        newer_full_doc["artifact_binding"] = newer_binding.to_dict()
        state.base.full_state.rows[state.lightrag_doc_id] = newer_full_doc
    async with state.base.status_state.lock:
        newer_status = deepcopy(state.base.status_state.rows[state.lightrag_doc_id])
        newer_status["status"] = "pending"
        newer_status["metadata"]["pipeline_attempt_token"] = newer_token
        newer_status["metadata"].pop("artifact_binding", None)
        state.base.status_state.rows[state.lightrag_doc_id] = newer_status
    process.doc_status.cas_release.set()

    summary = await asyncio.wait_for(reconcile, timeout=2)

    assert summary.finalized == 0
    assert summary.skipped == 1
    assert summary.error_count == 0
    assert _binding_from_full_docs(state) == newer_binding
    status = state.base.status_state.rows[state.lightrag_doc_id]
    assert status["status"] == "pending"
    assert status["metadata"]["pipeline_attempt_token"] == newer_token


@pytest.mark.parametrize(
    ("operation", "failed_status"),
    [("parse", "parse_failed"), ("build", "build_failed")],
)
async def test_failed_a_state_with_stale_claimed_binding_is_not_terminalized(
    tmp_path: Path,
    operation: str,
    failed_status: str,
) -> None:
    if operation == "parse":
        state = await _prepare_parse_crash(tmp_path, kb_id="kb_h2d_parse_failed")
    else:
        state = await _prepare_build_crash(tmp_path, kb_id="kb_h2d_build_failed")
    await _patch_document(state, status=failed_status)
    process = _new_recovery_process(
        state,
        tmp_path,
        name=f"recovery-{failed_status}",
    )

    summary = await process.reconciler.reconcile_kb(state.base.kb_id, process.rag)

    assert summary.finalized == 0
    assert summary.error_count == 0
    assert _binding_from_full_docs(state).state == "claimed"
    assert state.base.status_state.rows[state.lightrag_doc_id]["status"] == "processing"
    assert process.full_docs.get_attempts == 0
    assert process.full_docs.upsert_attempts == 0
    assert process.doc_status.upsert_attempts == 0


class _MissingCASStorage:
    async def get_by_id(self, key: str) -> None:
        del key
        return None

    async def upsert(self, values: Any) -> None:
        del values


class _InitializationRAG:
    def __init__(self, *, full_docs: Any, doc_status: Any) -> None:
        self.full_docs = full_docs
        self.doc_status = doc_status
        self.pipeline_artifact_materializer = None
        self.migrated = False

    async def initialize_storages(self) -> None:
        return None

    async def check_and_migrate_data(self) -> None:
        self.migrated = True


class _InitializationCoordinator:
    def materializer_for(self, record: Any) -> Any:
        del record

        async def materialize(binding: Any) -> None:
            del binding

        return materialize


async def test_object_initialization_missing_cas_fails_closed_local_unchanged(
    tmp_path: Path,
) -> None:
    from lightrag.api.lightrag_server import _initialize_kb_lightrag_instance

    state = await _prepare_parse_crash(tmp_path, kb_id="kb_h2d_init_capability")
    record = await state.base.kb_service.get(state.base.kb_id)
    object_rag = _InitializationRAG(
        full_docs=_MissingCASStorage(),
        doc_status=_ProcessKV(_SharedKVState(), row_kind="doc_status"),
    )

    with pytest.raises(
        RuntimeError,
        match=r"full_docs\.compare_and_commit_pipeline_attempt",
    ):
        await _initialize_kb_lightrag_instance(
            object_rag,
            record,
            artifact_storage_mode="object",
            coordinator=_InitializationCoordinator(),  # type: ignore[arg-type]
        )
    assert object_rag.migrated is False

    local_rag = _InitializationRAG(
        full_docs=_MissingCASStorage(),
        doc_status=_MissingCASStorage(),
    )
    returned = await _initialize_kb_lightrag_instance(
        local_rag,
        record,
        artifact_storage_mode="local",
        coordinator=None,
    )
    assert returned is local_rag
    assert local_rag.migrated is True
