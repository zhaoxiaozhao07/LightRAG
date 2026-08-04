from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import pytest

from lightrag.api.artifact_materialization import (
    ArtifactMaterializationLease,
    ArtifactMaterializer,
    MaterializationLimits,
)
from lightrag.api.document_lifecycle_service import (
    UploadedArtifactObject,
    _directory_checksum,
    _file_checksum,
)
from lightrag.api.index_build_service import (
    ImmutableArtifactObjectConflictError,
    IndexBuildService,
    _upload_immutable_artifact_directory,
    _upload_immutable_artifact_file,
)
from lightrag.api.kb_service import utc_now_iso
from lightrag.api.metadata_store import ArtifactRecord, DocumentRecord
from lightrag.artifact_runtime import (
    PipelineArtifactBinding,
    PipelineArtifactCommitOutcome,
    PipelineArtifactFinalizationResult,
)
from tests.api.test_artifact_storage_phase2a import _FakeObjectStorage

pytestmark = pytest.mark.offline

_KB_ID = "kb_h2c2"
_WORKSPACE = "workspace-h2c2"
_DOCUMENT_ID = "doc-h2c2"
_LIGHTRAG_DOCUMENT_ID = "lightrag-doc-h2c2"
_OLD_SIDECAR_ID = "artifact_sidecar_old"
_OLD_BLOCKS_ID = "artifact_blocks_old"
_PARSER_HASH = "sha256:parser-h2c2"
_INDEX_HASH = "sha256:index-h2c2"
_PARSE_GENERATION_ID = "parse-generation-h2c2"

_CommitMode = Literal["commit", "ack_loss", "unknown", "rollback"]


class _TrackingObjectStorage(_FakeObjectStorage):
    def __init__(self) -> None:
        super().__init__()
        self.file_upload_attempts: list[str] = []

    async def upload_file_if_absent(
        self,
        local_path: Path,
        *,
        key: str,
        content_type: str | None = None,
    ) -> tuple[str, bool]:
        self.file_upload_attempts.append(self.object_uri_for_key(key))
        return await super().upload_file_if_absent(
            local_path,
            key=key,
            content_type=content_type,
        )


class _MemoryBuildMetadataStore:
    """In-memory build store with PostgreSQL-style commit fault boundaries."""

    def __init__(
        self,
        document: DocumentRecord,
        artifacts: Sequence[ArtifactRecord],
        *,
        mode: _CommitMode,
    ) -> None:
        self.document = deepcopy(document)
        self.artifacts = {artifact.id: deepcopy(artifact) for artifact in artifacts}
        self.mode = mode
        self.commit_calls = 0
        self.candidate_artifacts: list[ArtifactRecord] = []

    async def complete_document_build_with_artifact_promotion(
        self,
        kb_id: str,
        document_id: str,
        *,
        index_hash: str,
        expected_current_sidecar_artifact_id: str | None,
        expected_current_blocks_artifact_id: str | None,
        current_sidecar_artifact_id: str,
        current_blocks_artifact_id: str | None,
        artifacts: Sequence[ArtifactRecord],
        chunks_count: int | None = None,
        entity_count: int | None = None,
        relation_count: int | None = None,
        metadata_patch: dict[str, Any],
        job_id: str | None = None,
        claim_token: str | None = None,
        expected_snapshot: dict[str, Any] | None = None,
    ) -> tuple[DocumentRecord, list[ArtifactRecord]]:
        del expected_snapshot
        assert kb_id == self.document.kb_id
        assert document_id == self.document.id
        assert job_id
        assert claim_token
        assert self.document.metadata.get("current_sidecar_artifact_id") == (
            expected_current_sidecar_artifact_id
        )
        assert self.document.metadata.get("current_blocks_artifact_id") == (
            expected_current_blocks_artifact_id
        )

        self.commit_calls += 1
        self.candidate_artifacts = deepcopy(list(artifacts))
        if self.mode in {"commit", "ack_loss"}:
            self._apply_commit(
                index_hash=index_hash,
                current_sidecar_artifact_id=current_sidecar_artifact_id,
                current_blocks_artifact_id=current_blocks_artifact_id,
                chunks_count=chunks_count,
                entity_count=entity_count,
                relation_count=relation_count,
                metadata_patch=metadata_patch,
            )
            if self.mode == "commit":
                return deepcopy(self.document), deepcopy(self.candidate_artifacts)
            raise RuntimeError("postgres commit ACK lost after durable commit")

        if self.mode == "unknown":
            raise RuntimeError("postgres connection lost at commit boundary")
        raise RuntimeError("postgres transaction confirmed rolled back")

    async def get_document_and_artifacts_by_ids(
        self,
        kb_id: str,
        document_id: str,
        artifact_ids: Sequence[str],
    ) -> tuple[DocumentRecord | None, dict[str, ArtifactRecord]]:
        assert kb_id == self.document.kb_id
        assert document_id == self.document.id
        if self.mode == "unknown" and self.candidate_artifacts:
            partial = self.candidate_artifacts[0]
            return deepcopy(self.document), {partial.id: deepcopy(partial)}
        return deepcopy(self.document), {
            artifact_id: deepcopy(self.artifacts[artifact_id])
            for artifact_id in artifact_ids
            if artifact_id in self.artifacts
        }

    def _apply_commit(
        self,
        *,
        index_hash: str,
        current_sidecar_artifact_id: str,
        current_blocks_artifact_id: str | None,
        chunks_count: int | None,
        entity_count: int | None,
        relation_count: int | None,
        metadata_patch: dict[str, Any],
    ) -> None:
        for artifact in self.candidate_artifacts:
            self.artifacts[artifact.id] = deepcopy(artifact)
        metadata = deepcopy(self.document.metadata)
        metadata.update(deepcopy(metadata_patch))
        metadata.update(
            {
                "current_sidecar_artifact_id": current_sidecar_artifact_id,
                "current_blocks_artifact_id": current_blocks_artifact_id,
                "pending_build_job_id": None,
                "pending_build_claim_token": None,
                "current_build_job_id": None,
                "current_build_claim_token": None,
            }
        )
        self.document = replace(
            self.document,
            status="ready",
            index_hash=index_hash,
            chunks_count=(
                chunks_count if chunks_count is not None else self.document.chunks_count
            ),
            entity_count=(
                entity_count if entity_count is not None else self.document.entity_count
            ),
            relation_count=(
                relation_count
                if relation_count is not None
                else self.document.relation_count
            ),
            error_code=None,
            error_message=None,
            metadata=metadata,
            updated_at=utc_now_iso(),
        )


class _MemoryDocumentService:
    object_authoritative = True

    def __init__(
        self,
        metadata_store: _MemoryBuildMetadataStore,
        object_storage: _TrackingObjectStorage,
        *,
        canonical_root: Path,
    ) -> None:
        self.metadata_store = metadata_store
        self.object_storage = object_storage
        self._canonical_root = canonical_root
        self.compensated: list[UploadedArtifactObject] = []

    async def get_document(self, kb_id: str, document_id: str) -> DocumentRecord:
        assert kb_id == self.metadata_store.document.kb_id
        assert document_id == self.metadata_store.document.id
        return deepcopy(self.metadata_store.document)

    def canonical_document_root(self, document: DocumentRecord) -> Path:
        assert document.id == self.metadata_store.document.id
        return self._canonical_root

    async def compensate_uploaded_artifact_objects(
        self, uploaded: list[UploadedArtifactObject]
    ) -> None:
        self.compensated.extend(uploaded)
        for uploaded_object in reversed(uploaded):
            if uploaded_object.is_prefix:
                await self.object_storage.delete_prefix(uploaded_object.uri)
            else:
                await self.object_storage.delete_uri(uploaded_object.uri)


@dataclass(slots=True)
class _BuildHarness:
    document: DocumentRecord
    binding: PipelineArtifactBinding
    sidecar_artifact: ArtifactRecord
    blocks_artifact: ArtifactRecord
    lease: ArtifactMaterializationLease
    runtime_sidecar_dir: Path
    runtime_blocks_path: Path
    storage: _TrackingObjectStorage
    store: _MemoryBuildMetadataStore
    document_service: _MemoryDocumentService
    index_service: IndexBuildService

    def mutate_outputs(self, marker: str = "changed") -> None:
        self.runtime_blocks_path.write_text(
            f'{{"text":"{marker}"}}\n', encoding="utf-8"
        )
        (self.runtime_sidecar_dir / "tables.json").write_text(
            f'{{"table":"{marker}"}}\n', encoding="utf-8"
        )


def _document(*, claim_token: str, job_id: str) -> DocumentRecord:
    now = utc_now_iso()
    return DocumentRecord(
        id=_DOCUMENT_ID,
        kb_id=_KB_ID,
        workspace=_WORKSPACE,
        lightrag_doc_id=_LIGHTRAG_DOCUMENT_ID,
        source_type="upload",
        source_name="report.pdf",
        source_uri="/canonical/report.pdf",
        source_hash="sha256:source-h2c2",
        content_type="application/pdf",
        size_bytes=100,
        parser_hash=_PARSER_HASH,
        index_hash="sha256:old-index-h2c2",
        status="building",
        enabled=True,
        archived=False,
        chunks_count=None,
        entity_count=None,
        relation_count=None,
        error_code=None,
        error_message=None,
        metadata={
            "parse_engine": "mineru",
            "process_options": "",
            "current_parse_generation_id": _PARSE_GENERATION_ID,
            "current_sidecar_artifact_id": _OLD_SIDECAR_ID,
            "current_blocks_artifact_id": _OLD_BLOCKS_ID,
            "pending_build_job_id": None,
            "pending_build_claim_token": None,
            "current_build_job_id": job_id,
            "current_build_claim_token": claim_token,
        },
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )


def _make_harness(
    root: Path,
    *,
    claim_token: str,
    mode: _CommitMode,
    storage: _TrackingObjectStorage,
) -> _BuildHarness:
    job_id = f"job-{claim_token}"
    document = _document(claim_token=claim_token, job_id=job_id)
    materializer = ArtifactMaterializer(
        storage,
        input_root=root / "inputs",
        limits=MaterializationLimits(
            max_objects=100,
            max_total_bytes=1024 * 1024,
            stale_ttl_seconds=0,
        ),
    )
    lease = materializer.create_lease()
    tree = lease.create_document_tree(document.source_name)
    runtime_sidecar_dir = tree.parsed_root / "report.pdf.parsed"
    runtime_sidecar_dir.mkdir(mode=0o700)
    runtime_blocks_path = runtime_sidecar_dir / "report.blocks.jsonl"
    runtime_blocks_path.write_text('{"text":"initial"}\n', encoding="utf-8")
    (runtime_sidecar_dir / "summary.json").write_text(
        '{"summary":"initial"}\n', encoding="utf-8"
    )

    canonical_root = root / "canonical" / _DOCUMENT_ID
    canonical_sidecar = canonical_root / "__parsed__" / runtime_sidecar_dir.name
    canonical_blocks = canonical_sidecar / runtime_blocks_path.name
    old_sidecar_prefix = storage.object_prefix_uri_for_key(
        "/".join(
            (
                "workspaces",
                document.workspace,
                "documents",
                document.id,
                "artifacts",
                "sidecar",
                _OLD_SIDECAR_ID,
                runtime_sidecar_dir.name,
            )
        )
    )
    old_blocks_uri = storage.object_uri_for_key(
        "/".join(
            (
                "workspaces",
                document.workspace,
                "documents",
                document.id,
                "artifacts",
                "blocks",
                _OLD_BLOCKS_ID,
                runtime_blocks_path.name,
            )
        )
    )
    sidecar_artifact = ArtifactRecord(
        id=_OLD_SIDECAR_ID,
        kb_id=document.kb_id,
        workspace=document.workspace,
        document_id=document.id,
        artifact_type="sidecar",
        uri=str(canonical_sidecar),
        checksum=_directory_checksum(runtime_sidecar_dir),
        size_bytes=None,
        metadata={
            "is_directory": True,
            "object_prefix_uri": old_sidecar_prefix,
            "blocks_path": str(canonical_blocks),
        },
        created_at=utc_now_iso(),
    )
    blocks_artifact = ArtifactRecord(
        id=_OLD_BLOCKS_ID,
        kb_id=document.kb_id,
        workspace=document.workspace,
        document_id=document.id,
        artifact_type="blocks",
        uri=str(canonical_blocks),
        checksum=_file_checksum(runtime_blocks_path),
        size_bytes=runtime_blocks_path.stat().st_size,
        metadata={"object_uri": old_blocks_uri},
        created_at=utc_now_iso(),
    )
    store = _MemoryBuildMetadataStore(
        document,
        [sidecar_artifact, blocks_artifact],
        mode=mode,
    )
    document_service = _MemoryDocumentService(
        store,
        storage,
        canonical_root=canonical_root,
    )
    binding = PipelineArtifactBinding(
        version=1,
        authority="kb_metadata",
        state="claimed",
        operation="build",
        kb_id=document.kb_id,
        kb_generation="kb-generation-h2c2",
        workspace=document.workspace,
        document_id=document.id,
        lightrag_doc_id=document.lightrag_doc_id or "",
        job_id=job_id,
        claim_token=claim_token,
        source_hash=document.source_hash,
        parser_hash=document.parser_hash,
        parse_generation_id=_PARSE_GENERATION_ID,
        index_hash=_INDEX_HASH,
        sidecar_artifact_id=sidecar_artifact.id,
        blocks_artifact_id=blocks_artifact.id,
        expected_current_sidecar_artifact_id=sidecar_artifact.id,
        expected_current_blocks_artifact_id=blocks_artifact.id,
        raw_artifact_ids=(),
    )
    return _BuildHarness(
        document=document,
        binding=binding,
        sidecar_artifact=sidecar_artifact,
        blocks_artifact=blocks_artifact,
        lease=lease,
        runtime_sidecar_dir=runtime_sidecar_dir,
        runtime_blocks_path=runtime_blocks_path,
        storage=storage,
        store=store,
        document_service=document_service,
        index_service=IndexBuildService(document_service),  # type: ignore[arg-type]
    )


@pytest.fixture
def build_harness_factory(
    tmp_path: Path,
) -> Iterator[Callable[..., _BuildHarness]]:
    harnesses: list[_BuildHarness] = []

    def create(
        *,
        claim_token: str = "build-claim-a",
        mode: _CommitMode = "commit",
        storage: _TrackingObjectStorage | None = None,
    ) -> _BuildHarness:
        harness = _make_harness(
            tmp_path / f"harness-{len(harnesses)}",
            claim_token=claim_token,
            mode=mode,
            storage=storage or _TrackingObjectStorage(),
        )
        harnesses.append(harness)
        return harness

    yield create

    for harness in reversed(harnesses):
        harness.lease.cleanup()


async def _finalize(
    harness: _BuildHarness,
) -> PipelineArtifactFinalizationResult:
    return await harness.index_service.complete_pipeline_artifact_success(
        harness.binding,
        document=harness.document,
        sidecar_artifact=harness.sidecar_artifact,
        blocks_artifact=harness.blocks_artifact,
        lease=harness.lease,
        runtime_sidecar_dir=harness.runtime_sidecar_dir,
        runtime_blocks_path=harness.runtime_blocks_path,
        chunks_count=7,
        entity_count=5,
        relation_count=3,
    )


def _expected_artifact_id(
    document: DocumentRecord,
    artifact_type: Literal["sidecar", "blocks"],
    claim_token: str,
) -> str:
    payload = "\0".join(
        (
            "pipeline-build-artifact-v1",
            document.kb_id,
            document.workspace,
            document.id,
            document.lightrag_doc_id or "",
            artifact_type,
            claim_token,
        )
    ).encode()
    return f"artifact_{artifact_type}_{hashlib.sha256(payload).hexdigest()}"


def _candidate_sidecar_prefix(harness: _BuildHarness) -> str:
    sidecar_id = _expected_artifact_id(
        harness.document,
        "sidecar",
        harness.binding.claim_token,
    )
    return "/".join(
        (
            "workspaces",
            harness.document.workspace,
            "documents",
            harness.document.id,
            "artifacts",
            "sidecar",
            sidecar_id,
            harness.runtime_sidecar_dir.name,
        )
    )


def _candidate_sidecar_uri(harness: _BuildHarness, relative_path: str) -> str:
    return harness.storage.object_uri_for_key(
        f"{_candidate_sidecar_prefix(harness)}/{relative_path}"
    )


def _candidate_blocks_uri(harness: _BuildHarness) -> str:
    blocks_id = _expected_artifact_id(
        harness.document,
        "blocks",
        harness.binding.claim_token,
    )
    return harness.storage.object_uri_for_key(
        "/".join(
            (
                "workspaces",
                harness.document.workspace,
                "documents",
                harness.document.id,
                "artifacts",
                "blocks",
                blocks_id,
                harness.runtime_blocks_path.name,
            )
        )
    )


def _assert_promotion_did_not_commit(harness: _BuildHarness) -> None:
    assert harness.store.commit_calls == 0
    assert harness.store.candidate_artifacts == []
    assert set(harness.store.artifacts) == {_OLD_SIDECAR_ID, _OLD_BLOCKS_ID}
    assert harness.store.document.status == "building"
    assert harness.store.document.metadata["current_sidecar_artifact_id"] == (
        _OLD_SIDECAR_ID
    )
    assert harness.store.document.metadata["current_blocks_artifact_id"] == (
        _OLD_BLOCKS_ID
    )


def _assert_exact_committed_artifacts(
    harness: _BuildHarness,
    result: PipelineArtifactFinalizationResult,
) -> dict[str, ArtifactRecord]:
    assert result.outcome is PipelineArtifactCommitOutcome.COMMITTED
    assert result.committed_binding is not None
    candidates = {
        artifact.artifact_type: artifact
        for artifact in harness.store.candidate_artifacts
    }
    assert set(candidates) == {"sidecar", "blocks"}
    assert result.committed_binding.sidecar_artifact_id == candidates["sidecar"].id
    assert result.committed_binding.blocks_artifact_id == candidates["blocks"].id
    assert harness.store.document.metadata["current_sidecar_artifact_id"] == (
        candidates["sidecar"].id
    )
    assert harness.store.document.metadata["current_blocks_artifact_id"] == (
        candidates["blocks"].id
    )
    assert harness.store.artifacts[candidates["sidecar"].id].metadata == (
        candidates["sidecar"].metadata
    )
    assert harness.store.artifacts[candidates["blocks"].id].metadata == (
        candidates["blocks"].metadata
    )
    return candidates


async def test_unchanged_outputs_commit_old_binding_without_creating_objects(
    build_harness_factory: Callable[..., _BuildHarness],
) -> None:
    harness = build_harness_factory()
    old_artifact_ids = set(harness.store.artifacts)

    result = await _finalize(harness)

    assert result.outcome is PipelineArtifactCommitOutcome.COMMITTED
    assert result.committed_binding is not None
    assert result.committed_binding.sidecar_artifact_id == _OLD_SIDECAR_ID
    assert result.committed_binding.blocks_artifact_id == _OLD_BLOCKS_ID
    assert harness.store.document.metadata["current_sidecar_artifact_id"] == (
        _OLD_SIDECAR_ID
    )
    assert harness.store.document.metadata["current_blocks_artifact_id"] == (
        _OLD_BLOCKS_ID
    )
    assert harness.store.candidate_artifacts == []
    assert set(harness.store.artifacts) == old_artifact_ids
    assert harness.storage.file_upload_attempts == []
    assert harness.storage.file_uploads == []
    assert harness.storage.prefix_uploads == []


async def test_changed_outputs_have_attempt_deterministic_and_isolated_ids(
    build_harness_factory: Callable[..., _BuildHarness],
) -> None:
    storage = _TrackingObjectStorage()
    first = build_harness_factory(storage=storage, claim_token="build-claim-a")
    first.mutate_outputs()

    first_result = await _finalize(first)
    first_candidates = _assert_exact_committed_artifacts(first, first_result)
    first_ids = {artifact.id for artifact in first_candidates.values()}
    assert first_candidates["sidecar"].id == _expected_artifact_id(
        first.document, "sidecar", first.binding.claim_token
    )
    assert first_candidates["blocks"].id == _expected_artifact_id(
        first.document, "blocks", first.binding.claim_token
    )
    first_create_count = len(storage.file_uploads)
    assert first_create_count > 0

    same_attempt_retry = build_harness_factory(
        storage=storage,
        claim_token="build-claim-a",
    )
    same_attempt_retry.mutate_outputs()
    retry_result = await _finalize(same_attempt_retry)
    retry_candidates = _assert_exact_committed_artifacts(
        same_attempt_retry, retry_result
    )

    assert {artifact.id for artifact in retry_candidates.values()} == first_ids
    assert len(storage.file_uploads) == first_create_count
    assert len(storage.file_upload_attempts) > first_create_count

    isolated_attempt = build_harness_factory(
        storage=storage,
        claim_token="build-claim-b",
    )
    isolated_attempt.mutate_outputs()
    isolated_result = await _finalize(isolated_attempt)
    isolated_candidates = _assert_exact_committed_artifacts(
        isolated_attempt, isolated_result
    )
    isolated_ids = {artifact.id for artifact in isolated_candidates.values()}

    assert first_ids.isdisjoint(isolated_ids)
    assert len(storage.file_uploads) > first_create_count


async def test_immutable_upload_helpers_verify_retries_without_overwrite(
    build_harness_factory: Callable[..., _BuildHarness],
) -> None:
    harness = build_harness_factory()
    prefix = "immutable/helper/sidecar/"
    blocks_key = "immutable/helper/blocks/report.blocks.jsonl"
    first_uploaded_objects: list[UploadedArtifactObject] = []

    prefix_uri, first_directory_objects = await _upload_immutable_artifact_directory(
        harness.storage,
        harness.runtime_sidecar_dir,
        prefix=prefix,
        verification_root=harness.runtime_sidecar_dir.parent,
        uploaded_objects=first_uploaded_objects,
    )
    blocks_uri, first_blocks_created = await _upload_immutable_artifact_file(
        harness.storage,
        harness.runtime_blocks_path,
        key=blocks_key,
        verification_root=harness.runtime_sidecar_dir.parent,
        uploaded_objects=first_uploaded_objects,
    )
    stored_before_retry = dict(harness.storage.files)
    create_count = len(harness.storage.file_uploads)
    retry_uploaded_objects: list[UploadedArtifactObject] = []

    (
        retry_prefix_uri,
        retry_directory_objects,
    ) = await _upload_immutable_artifact_directory(
        harness.storage,
        harness.runtime_sidecar_dir,
        prefix=prefix,
        verification_root=harness.runtime_sidecar_dir.parent,
        uploaded_objects=retry_uploaded_objects,
    )
    retry_blocks_uri, retry_blocks_created = await _upload_immutable_artifact_file(
        harness.storage,
        harness.runtime_blocks_path,
        key=blocks_key,
        verification_root=harness.runtime_sidecar_dir.parent,
        uploaded_objects=retry_uploaded_objects,
    )

    assert first_directory_objects
    assert first_blocks_created is True
    assert {item.uri for item in first_uploaded_objects} == set(
        harness.storage.file_uploads
    )
    assert retry_prefix_uri == prefix_uri
    assert retry_directory_objects == []
    assert retry_blocks_uri == blocks_uri
    assert retry_blocks_created is False
    assert retry_uploaded_objects == []
    assert len(harness.storage.file_uploads) == create_count
    assert harness.storage.files == stored_before_retry

    harness.runtime_blocks_path.write_text('{"text":"different"}\n', encoding="utf-8")
    conflict_uploaded_objects: list[UploadedArtifactObject] = []
    with pytest.raises(ImmutableArtifactObjectConflictError):
        await _upload_immutable_artifact_directory(
            harness.storage,
            harness.runtime_sidecar_dir,
            prefix=prefix,
            verification_root=harness.runtime_sidecar_dir.parent,
            uploaded_objects=conflict_uploaded_objects,
        )
    with pytest.raises(ImmutableArtifactObjectConflictError):
        await _upload_immutable_artifact_file(
            harness.storage,
            harness.runtime_blocks_path,
            key=blocks_key,
            verification_root=harness.runtime_sidecar_dir.parent,
            uploaded_objects=conflict_uploaded_objects,
        )
    assert conflict_uploaded_objects == []
    assert harness.storage.files == stored_before_retry


async def test_sidecar_conflict_compensates_earlier_created_child(
    build_harness_factory: Callable[..., _BuildHarness],
) -> None:
    harness = build_harness_factory()
    harness.mutate_outputs("sidecar-conflict")
    ordered_members = sorted(
        path.relative_to(harness.runtime_sidecar_dir).as_posix()
        for path in harness.runtime_sidecar_dir.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    assert ordered_members[:2] == ["report.blocks.jsonl", "summary.json"]
    first_created_uri = _candidate_sidecar_uri(harness, ordered_members[0])
    conflicting_uri = _candidate_sidecar_uri(harness, ordered_members[1])
    conflicting_payload = b'{"summary":"preexisting-conflict"}\n'
    harness.storage.files[conflicting_uri] = conflicting_payload

    with pytest.raises(
        ImmutableArtifactObjectConflictError,
        match="already contains different bytes",
    ):
        await _finalize(harness)

    assert harness.storage.file_upload_attempts == [
        first_created_uri,
        conflicting_uri,
    ]
    assert harness.storage.file_uploads == [first_created_uri]
    assert harness.storage.deleted_files == [first_created_uri]
    assert harness.storage.deleted_prefixes == []
    assert first_created_uri not in harness.storage.files
    assert harness.storage.files[conflicting_uri] == conflicting_payload
    assert conflicting_uri not in harness.storage.deleted_files
    assert harness.document_service.compensated == [
        UploadedArtifactObject(uri=first_created_uri, is_prefix=False)
    ]
    _assert_promotion_did_not_commit(harness)


async def test_sidecar_conflict_retains_already_identical_first_child(
    build_harness_factory: Callable[..., _BuildHarness],
) -> None:
    harness = build_harness_factory()
    harness.mutate_outputs("sidecar-identical-first")
    identical_uri = _candidate_sidecar_uri(
        harness,
        harness.runtime_blocks_path.name,
    )
    conflicting_uri = _candidate_sidecar_uri(harness, "summary.json")
    identical_payload = harness.runtime_blocks_path.read_bytes()
    conflicting_payload = b'{"summary":"preexisting-conflict"}\n'
    harness.storage.files[identical_uri] = identical_payload
    harness.storage.files[conflicting_uri] = conflicting_payload
    preexisting = dict(harness.storage.files)

    with pytest.raises(
        ImmutableArtifactObjectConflictError,
        match="already contains different bytes",
    ):
        await _finalize(harness)

    assert harness.storage.file_upload_attempts == [identical_uri, conflicting_uri]
    assert harness.storage.file_uploads == []
    assert harness.storage.deleted_files == []
    assert harness.storage.deleted_prefixes == []
    assert harness.document_service.compensated == []
    assert harness.storage.files == preexisting
    _assert_promotion_did_not_commit(harness)


async def test_blocks_conflict_compensates_all_created_sidecar_children(
    build_harness_factory: Callable[..., _BuildHarness],
) -> None:
    harness = build_harness_factory()
    harness.mutate_outputs("blocks-conflict")
    sidecar_member_uris = {
        _candidate_sidecar_uri(
            harness,
            path.relative_to(harness.runtime_sidecar_dir).as_posix(),
        )
        for path in harness.runtime_sidecar_dir.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    assert len(sidecar_member_uris) >= 2
    conflicting_blocks_uri = _candidate_blocks_uri(harness)
    conflicting_blocks_payload = b'{"text":"preexisting-blocks-conflict"}\n'
    harness.storage.files[conflicting_blocks_uri] = conflicting_blocks_payload

    with pytest.raises(
        ImmutableArtifactObjectConflictError,
        match="already contains different bytes",
    ):
        await _finalize(harness)

    assert set(harness.storage.file_uploads) == sidecar_member_uris
    assert harness.storage.file_upload_attempts[-1] == conflicting_blocks_uri
    assert set(harness.storage.deleted_files) == sidecar_member_uris
    assert harness.storage.deleted_prefixes == []
    assert sidecar_member_uris.isdisjoint(harness.storage.files)
    assert harness.storage.files[conflicting_blocks_uri] == conflicting_blocks_payload
    assert conflicting_blocks_uri not in harness.storage.deleted_files
    assert {item.uri for item in harness.document_service.compensated} == (
        sidecar_member_uris
    )
    assert all(not item.is_prefix for item in harness.document_service.compensated)
    _assert_promotion_did_not_commit(harness)


async def test_postgres_commit_ack_loss_readback_returns_committed_binding(
    build_harness_factory: Callable[..., _BuildHarness],
) -> None:
    harness = build_harness_factory(mode="ack_loss")
    harness.mutate_outputs()

    result = await _finalize(harness)
    _assert_exact_committed_artifacts(harness, result)

    created_objects = set(harness.storage.file_uploads)
    assert created_objects
    assert created_objects <= set(harness.storage.files)
    assert harness.storage.deleted_files == []
    assert harness.storage.deleted_prefixes == []
    assert harness.store.document.status == "ready"
    assert harness.store.commit_calls == 1


async def test_unknown_commit_outcome_retains_candidates_without_binding(
    build_harness_factory: Callable[..., _BuildHarness],
) -> None:
    harness = build_harness_factory(mode="unknown")
    harness.mutate_outputs()

    result = await _finalize(harness)

    assert result.outcome is PipelineArtifactCommitOutcome.UNKNOWN
    assert result.committed_binding is None
    assert result.reason == "candidate_artifacts_partial_or_mismatched"
    candidate_objects = set(harness.storage.file_uploads)
    assert candidate_objects
    assert candidate_objects <= set(harness.storage.files)
    assert harness.storage.deleted_files == []
    assert harness.storage.deleted_prefixes == []
    assert harness.store.document.status == "building"
    assert harness.store.document.metadata["current_sidecar_artifact_id"] == (
        _OLD_SIDECAR_ID
    )
    assert harness.store.document.metadata["current_blocks_artifact_id"] == (
        _OLD_BLOCKS_ID
    )


async def test_confirmed_rollback_compensates_only_objects_created_by_invocation(
    build_harness_factory: Callable[..., _BuildHarness],
) -> None:
    harness = build_harness_factory(mode="rollback")
    harness.mutate_outputs("rollback")
    sidecar_id = _expected_artifact_id(
        harness.document, "sidecar", harness.binding.claim_token
    )
    blocks_id = _expected_artifact_id(
        harness.document, "blocks", harness.binding.claim_token
    )
    sidecar_prefix = "/".join(
        (
            "workspaces",
            harness.document.workspace,
            "documents",
            harness.document.id,
            "artifacts",
            "sidecar",
            sidecar_id,
            harness.runtime_sidecar_dir.name,
        )
    )
    preexisting_sidecar_uri = harness.storage.object_uri_for_key(
        f"{sidecar_prefix}/summary.json"
    )
    preexisting_blocks_uri = harness.storage.object_uri_for_key(
        "/".join(
            (
                "workspaces",
                harness.document.workspace,
                "documents",
                harness.document.id,
                "artifacts",
                "blocks",
                blocks_id,
                harness.runtime_blocks_path.name,
            )
        )
    )
    unrelated_uri = harness.storage.object_uri_for_key("retained/unrelated-object")
    harness.storage.files[preexisting_sidecar_uri] = (
        harness.runtime_sidecar_dir / "summary.json"
    ).read_bytes()
    harness.storage.files[preexisting_blocks_uri] = (
        harness.runtime_blocks_path.read_bytes()
    )
    harness.storage.files[unrelated_uri] = b"unrelated"
    preexisting = dict(harness.storage.files)

    with pytest.raises(RuntimeError, match="confirmed rolled back"):
        await _finalize(harness)

    created_by_invocation = set(harness.storage.file_uploads)
    assert created_by_invocation
    assert created_by_invocation.isdisjoint(preexisting)
    assert created_by_invocation.isdisjoint(harness.storage.files)
    assert set(harness.storage.deleted_files) == created_by_invocation
    assert {item.uri for item in harness.document_service.compensated} == (
        created_by_invocation
    )
    for uri, payload in preexisting.items():
        assert harness.storage.files[uri] == payload
        assert uri not in harness.storage.deleted_files
    assert harness.store.document.metadata["current_sidecar_artifact_id"] == (
        _OLD_SIDECAR_ID
    )
    assert harness.store.document.metadata["current_blocks_artifact_id"] == (
        _OLD_BLOCKS_ID
    )
