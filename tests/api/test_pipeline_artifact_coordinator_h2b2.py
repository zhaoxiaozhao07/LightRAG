from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from lightrag.artifact_runtime import (
    PipelineArtifactCommitOutcome,
    PipelineArtifactBinding,
    PipelineTerminalOutcome,
)
from lightrag.api.document_lifecycle_service import DocumentLifecycleService
from lightrag.api.index_build_service import IndexBuildPlan, IndexBuildService
from lightrag.api.kb_service import KnowledgeBaseRecord
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry
from lightrag.api.metadata_store import _dumps_json, _loads_json_object
from lightrag.api.pipeline_artifact_coordinator import (
    ArtifactChecksumMismatchError,
    ArtifactMigrationRequiredError,
    CoordinatedPipelineArtifactSession,
    PipelineArtifactBindingStaleError,
    PipelineArtifactCoordinator,
)
from lightrag.utils_pipeline import reset_canonical_input_root_for_tests
from tests.api.test_artifact_storage_phase2a import _build_object_service
from tests.api.test_artifact_storage_phase2b import (
    _BuildRAG,
    _setup_parsed_object_document,
)


pytestmark = pytest.mark.offline


@dataclass(slots=True)
class _BuildAuthority:
    coordinator: PipelineArtifactCoordinator
    binding: PipelineArtifactBinding
    plan: IndexBuildPlan
    job_id: str
    kb_service: Any
    metadata_store: Any
    storage: Any
    materializer: Any
    document_service: DocumentLifecycleService


async def _claimed_build_authority(tmp_path: Path) -> _BuildAuthority:
    (
        kb_service,
        metadata_store,
        storage,
        document,
        _enqueue_materializer,
        job_service,
        _rag,
    ) = await _setup_parsed_object_document(tmp_path, kb_id="kb_h2b2_build")
    document_service, materializer = _build_object_service(
        root=tmp_path / "drain-owner" / "inputs",
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
    )
    index_service = IndexBuildService(document_service)
    plan = await index_service.create_build_plan(
        "kb_h2b2_build",
        document.id,
        rag=_BuildRAG(),
        force_rechunk=True,
    )
    job, _created = await job_service.create_build_job_once(
        "kb_h2b2_build",
        document_id=document.id,
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
    await index_service.claim_build_queued("kb_h2b2_build", job_id=job.id, plan=plan)
    await index_service.mark_building(
        "kb_h2b2_build",
        document.id,
        job_id=job.id,
        claim_token=plan.claim_token,
        plan=plan,
    )
    record = await kb_service.get("kb_h2b2_build")
    assert plan.claim_token
    assert plan.sidecar_artifact is not None
    assert plan.blocks_artifact is not None
    binding = PipelineArtifactBinding(
        version=1,
        authority="kb_metadata",
        state="claimed",
        operation="build",
        kb_id=record.id,
        kb_generation=record.generation,
        workspace=record.workspace,
        document_id=plan.document.id,
        lightrag_doc_id=plan.document.lightrag_doc_id or "",
        job_id=job.id,
        claim_token=plan.claim_token,
        source_hash=plan.document.source_hash,
        parser_hash=plan.parser_hash,
        parse_generation_id=plan.expected_current_parse_generation_id,
        index_hash=plan.index_hash,
        sidecar_artifact_id=plan.sidecar_artifact.id,
        blocks_artifact_id=plan.blocks_artifact.id,
        expected_current_sidecar_artifact_id=(
            plan.expected_current_sidecar_artifact_id
        ),
        expected_current_blocks_artifact_id=(plan.expected_current_blocks_artifact_id),
        raw_artifact_ids=(),
    )
    return _BuildAuthority(
        coordinator=PipelineArtifactCoordinator(
            kb_service, document_service, index_service
        ),
        binding=binding,
        plan=plan,
        job_id=job.id,
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
        materializer=materializer,
        document_service=document_service,
    )


@dataclass(slots=True)
class _ParseAuthority:
    coordinator: PipelineArtifactCoordinator
    binding: PipelineArtifactBinding
    plan: Any
    job_id: str
    kb_service: Any
    metadata_store: Any
    storage: Any
    materializer: Any
    document_service: DocumentLifecycleService


async def _claimed_parse_authority(tmp_path: Path) -> _ParseAuthority:
    (
        kb_service,
        metadata_store,
        storage,
        document,
        _enqueue_materializer,
        job_service,
        _rag,
    ) = await _setup_parsed_object_document(tmp_path, kb_id="kb_h2b2_parse")
    document_service, materializer = _build_object_service(
        root=tmp_path / "parse-drain-owner" / "inputs",
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
    )
    index_service = IndexBuildService(document_service)
    plan = await document_service.create_parse_plan(
        "kb_h2b2_parse",
        document.id,
        parser_engine="mineru",
    )
    job, _created = await job_service.create_parse_job_once(
        "kb_h2b2_parse",
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
        "kb_h2b2_parse", document.id, job=job, plan=plan
    )
    await document_service.mark_parse_running(
        "kb_h2b2_parse",
        document.id,
        job_id=job.id,
        claim_token=plan.claim_token,
        plan=plan,
    )
    record = await kb_service.get("kb_h2b2_parse")
    assert plan.claim_token
    binding = PipelineArtifactBinding(
        version=1,
        authority="kb_metadata",
        state="claimed",
        operation="parse",
        kb_id=record.id,
        kb_generation=record.generation,
        workspace=record.workspace,
        document_id=plan.document.id,
        lightrag_doc_id=plan.lightrag_doc_id,
        job_id=job.id,
        claim_token=plan.claim_token,
        source_hash=plan.document.source_hash,
        parser_hash=plan.parser_hash,
        parse_generation_id=plan.claim_token,
        index_hash=plan.document.index_hash,
        sidecar_artifact_id=plan.expected_current_sidecar_artifact_id,
        blocks_artifact_id=plan.expected_current_blocks_artifact_id,
        expected_current_sidecar_artifact_id=(
            plan.expected_current_sidecar_artifact_id
        ),
        expected_current_blocks_artifact_id=(plan.expected_current_blocks_artifact_id),
        raw_artifact_ids=tuple(ref.artifact_id for ref in plan.raw_object_refs),
    )
    return _ParseAuthority(
        coordinator=PipelineArtifactCoordinator(
            kb_service, document_service, index_service
        ),
        binding=binding,
        plan=plan,
        job_id=job.id,
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
        materializer=materializer,
        document_service=document_service,
    )


async def test_build_materializes_exact_artifacts_only_in_drain_owner_root(
    tmp_path: Path,
) -> None:
    authority = await _claimed_build_authority(tmp_path)
    enqueue_root = tmp_path / "parse-root" / "inputs"
    poison = enqueue_root / "poison-sidecar"
    poison.mkdir(parents=True)
    (poison / "poison.blocks.jsonl").write_text("enqueue-owner", encoding="utf-8")

    callback = authority.coordinator.materializer_for(
        await authority.kb_service.get(authority.binding.kb_id)
    )
    session = await callback(authority.binding)
    assert isinstance(session, CoordinatedPipelineArtifactSession)
    assert session.source_path is None
    assert session.sidecar_dir is not None and session.sidecar_dir.is_dir()
    assert session.blocks_path is not None and session.blocks_path.is_file()
    assert session.sidecar_dir.is_relative_to(authority.materializer.scratch_root)
    assert not session.sidecar_dir.is_relative_to(enqueue_root)
    assert "enqueue-owner" not in session.blocks_path.read_text(encoding="utf-8")
    assert authority.storage.prefix_downloads
    assert authority.storage.file_downloads

    lease_path = session.sidecar_dir.parents[2]
    await session.aclose()
    assert not lease_path.exists()


async def test_parse_materializes_source_and_exact_raw_cache_in_drain_owner_root(
    tmp_path: Path,
) -> None:
    authority = await _claimed_parse_authority(tmp_path)
    callback = authority.coordinator.materializer_for(
        await authority.kb_service.get(authority.binding.kb_id)
    )
    session = await callback(authority.binding)

    assert session.source_path is not None and session.source_path.is_file()
    assert session.source_path.is_relative_to(authority.materializer.scratch_root)
    assert session.sidecar_dir is None
    assert session.blocks_path is None
    for ref in authority.plan.raw_object_refs:
        raw_dir = session.source_path.parent / "__parsed__" / ref.directory_name
        assert raw_dir.is_dir()
    assert authority.storage.file_downloads
    assert authority.storage.prefix_downloads

    lease_path = session.source_path.parents[1]
    await session.aclose()
    assert not lease_path.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kb_generation", "wrong-generation"),
        ("workspace", "wrong-workspace"),
        ("lightrag_doc_id", "wrong-lightrag-id"),
        ("job_id", "wrong-job"),
        ("claim_token", "wrong-token"),
        ("source_hash", "sha256:" + "0" * 64),
        ("parser_hash", "sha256:" + "1" * 64),
        ("parse_generation_id", "wrong-parse-generation"),
        ("index_hash", "sha256:" + "f" * 64),
        ("expected_current_sidecar_artifact_id", "wrong-sidecar-pointer"),
        ("expected_current_blocks_artifact_id", "wrong-blocks-pointer"),
    ],
)
async def test_binding_mismatch_fails_before_any_download(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    authority = await _claimed_build_authority(tmp_path)
    mutated = replace(authority.binding, **{field: value})
    downloads_before = (
        len(authority.storage.file_downloads),
        len(authority.storage.prefix_downloads),
    )

    with pytest.raises(PipelineArtifactBindingStaleError):
        await authority.coordinator.open(mutated)

    assert (
        len(authority.storage.file_downloads),
        len(authority.storage.prefix_downloads),
    ) == downloads_before
    assert not list(authority.materializer.scratch_root.iterdir())


async def test_artifact_row_ownership_and_object_scope_fail_before_download(
    tmp_path: Path,
) -> None:
    authority = await _claimed_build_authority(tmp_path)
    sidecar_id = authority.binding.sidecar_artifact_id
    assert sidecar_id
    downloads_before = (
        len(authority.storage.file_downloads),
        len(authority.storage.prefix_downloads),
    )

    await authority.metadata_store._write(
        lambda conn: conn.execute(
            "UPDATE document_artifacts SET workspace = ? WHERE id = ?",
            ("wrong-workspace", sidecar_id),
        )
    )
    with pytest.raises(PipelineArtifactBindingStaleError):
        await authority.coordinator.open(authority.binding)
    assert (
        len(authority.storage.file_downloads),
        len(authority.storage.prefix_downloads),
    ) == downloads_before


async def test_artifact_object_uri_ownership_fails_before_download(
    tmp_path: Path,
) -> None:
    authority = await _claimed_build_authority(tmp_path)
    sidecar_id = authority.binding.sidecar_artifact_id
    assert sidecar_id
    downloads_before = (
        len(authority.storage.file_downloads),
        len(authority.storage.prefix_downloads),
    )

    def replace_prefix(conn) -> None:
        row = conn.execute(
            "SELECT metadata_json FROM document_artifacts WHERE id = ?",
            (sidecar_id,),
        ).fetchone()
        metadata = _loads_json_object(row["metadata_json"])
        metadata["object_prefix_uri"] = (
            "s3://phase2a/workspaces/wrong/documents/wrong/"
            f"artifacts/sidecar/{sidecar_id}/bundle/"
        )
        conn.execute(
            "UPDATE document_artifacts SET metadata_json = ? WHERE id = ?",
            (_dumps_json(metadata), sidecar_id),
        )

    await authority.metadata_store._write(replace_prefix)
    with pytest.raises(PipelineArtifactBindingStaleError):
        await authority.coordinator.open(authority.binding)
    assert (
        len(authority.storage.file_downloads),
        len(authority.storage.prefix_downloads),
    ) == downloads_before


async def test_invalid_checksum_metadata_fails_before_download(tmp_path: Path) -> None:
    authority = await _claimed_build_authority(tmp_path)
    blocks_id = authority.binding.blocks_artifact_id
    assert blocks_id
    downloads_before = (
        len(authority.storage.file_downloads),
        len(authority.storage.prefix_downloads),
    )
    await authority.metadata_store._write(
        lambda conn: conn.execute(
            "UPDATE document_artifacts SET checksum = ? WHERE id = ?",
            ("legacy-unverified", blocks_id),
        )
    )

    with pytest.raises(ArtifactMigrationRequiredError):
        await authority.coordinator.open(authority.binding)
    assert (
        len(authority.storage.file_downloads),
        len(authority.storage.prefix_downloads),
    ) == downloads_before


async def test_tampered_download_fails_checksum_before_pipeline_mutation(
    tmp_path: Path,
) -> None:
    authority = await _claimed_build_authority(tmp_path)
    assert authority.plan.blocks_artifact is not None
    blocks_uri = authority.plan.blocks_artifact.object_uri
    assert blocks_uri
    authority.storage.files[blocks_uri] = b"tampered-blocks"

    with pytest.raises(ArtifactChecksumMismatchError):
        await authority.coordinator.open(authority.binding)

    assert not list(authority.materializer.scratch_root.iterdir())


async def test_authority_change_during_download_fails_before_session_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = await _claimed_build_authority(tmp_path)
    original_download_prefix = authority.storage.download_prefix

    async def mutate_owner_after_download(*args, **kwargs):
        downloaded = await original_download_prefix(*args, **kwargs)

        def mutate_owner(conn) -> None:
            row = conn.execute(
                "SELECT metadata_json FROM documents WHERE kb_id = ? AND id = ?",
                (authority.binding.kb_id, authority.binding.document_id),
            ).fetchone()
            metadata = _loads_json_object(row["metadata_json"])
            metadata["current_build_claim_token"] = "winner-token"
            conn.execute(
                "UPDATE documents SET metadata_json = ? WHERE kb_id = ? AND id = ?",
                (
                    _dumps_json(metadata),
                    authority.binding.kb_id,
                    authority.binding.document_id,
                ),
            )

        await authority.metadata_store._write(mutate_owner)
        return downloaded

    monkeypatch.setattr(
        authority.storage,
        "download_prefix",
        mutate_owner_after_download,
    )
    with pytest.raises(PipelineArtifactBindingStaleError):
        await authority.coordinator.open(authority.binding)
    assert not list(authority.materializer.scratch_root.iterdir())


async def test_cancelled_materialization_defers_lease_for_safe_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = await _claimed_build_authority(tmp_path)

    async def cancel_download(*args, **kwargs):
        del args, kwargs
        raise asyncio.CancelledError()

    monkeypatch.setattr(authority.storage, "download_prefix", cancel_download)
    with pytest.raises(asyncio.CancelledError):
        await authority.coordinator.open(authority.binding)

    deferred = list(authority.materializer._deferred_leases)
    assert len(deferred) == 1
    assert deferred[0].cleanup_deferred is True
    assert deferred[0].path.is_dir()
    deferred[0].cleanup()
    assert not list(authority.materializer.scratch_root.iterdir())


async def test_missing_build_object_reference_requires_migration_without_fallback(
    tmp_path: Path,
) -> None:
    authority = await _claimed_build_authority(tmp_path)
    sidecar_id = authority.binding.sidecar_artifact_id
    assert sidecar_id

    def remove_object_ref(conn) -> None:
        row = conn.execute(
            "SELECT metadata_json FROM document_artifacts WHERE id = ?",
            (sidecar_id,),
        ).fetchone()
        metadata = _loads_json_object(row["metadata_json"])
        metadata.pop("object_prefix_uri", None)
        conn.execute(
            "UPDATE document_artifacts SET metadata_json = ? WHERE id = ?",
            (_dumps_json(metadata), sidecar_id),
        )

    await authority.metadata_store._write(remove_object_ref)
    with pytest.raises(ArtifactMigrationRequiredError) as exc_info:
        await authority.coordinator.open(authority.binding)
    assert "artifact_migration_required" in str(exc_info.value)
    assert not list(authority.materializer.scratch_root.iterdir())


async def test_missing_parse_source_reference_requires_migration_without_local_fallback(
    tmp_path: Path,
) -> None:
    authority = await _claimed_parse_authority(tmp_path)
    await authority.metadata_store.update_document(
        authority.binding.kb_id,
        authority.binding.document_id,
        metadata_patch={"source_object_uri": None},
    )
    downloads_before = (
        len(authority.storage.file_downloads),
        len(authority.storage.prefix_downloads),
    )

    with pytest.raises(ArtifactMigrationRequiredError) as exc_info:
        await authority.coordinator.open(authority.binding)
    assert "artifact_migration_required" in str(exc_info.value)
    assert (
        len(authority.storage.file_downloads),
        len(authority.storage.prefix_downloads),
    ) == downloads_before
    assert not list(authority.materializer.scratch_root.iterdir())


async def test_parse_source_checksum_mismatch_fails_without_local_fallback(
    tmp_path: Path,
) -> None:
    authority = await _claimed_parse_authority(tmp_path)
    source_uri = authority.plan.source_object_uri
    assert source_uri
    authority.storage.files[source_uri] = b"tampered-source"

    with pytest.raises(ArtifactChecksumMismatchError):
        await authority.coordinator.open(authority.binding)
    assert not list(authority.materializer.scratch_root.iterdir())


async def test_missing_parse_raw_object_reference_requires_migration(
    tmp_path: Path,
) -> None:
    authority = await _claimed_parse_authority(tmp_path)
    assert authority.binding.raw_artifact_ids
    raw_id = authority.binding.raw_artifact_ids[0]
    downloads_before = (
        len(authority.storage.file_downloads),
        len(authority.storage.prefix_downloads),
    )

    def remove_raw_prefix(conn) -> None:
        row = conn.execute(
            "SELECT metadata_json FROM document_artifacts WHERE id = ?",
            (raw_id,),
        ).fetchone()
        metadata = _loads_json_object(row["metadata_json"])
        metadata.pop("object_prefix_uri", None)
        conn.execute(
            "UPDATE document_artifacts SET metadata_json = ? WHERE id = ?",
            (_dumps_json(metadata), raw_id),
        )

    await authority.metadata_store._write(remove_raw_prefix)
    with pytest.raises(ArtifactMigrationRequiredError):
        await authority.coordinator.open(authority.binding)
    assert (
        len(authority.storage.file_downloads),
        len(authority.storage.prefix_downloads),
    ) == downloads_before
    assert not list(authority.materializer.scratch_root.iterdir())


async def test_failure_cancel_release_is_owner_aware_and_cleans_lease(
    tmp_path: Path,
) -> None:
    authority = await _claimed_build_authority(tmp_path)
    session = await authority.coordinator.open(authority.binding)
    assert session.sidecar_dir is not None
    lease_path = session.sidecar_dir.parents[2]

    await session.finish(PipelineTerminalOutcome.CANCELLED)
    await session.finish(PipelineTerminalOutcome.CANCELLED)
    await session.aclose()
    await session.aclose()

    document = await authority.metadata_store.get_document(
        authority.binding.kb_id, authority.binding.document_id
    )
    assert document.status == "build_failed"
    assert document.metadata.get("current_build_claim_token") is None
    assert document.error_code == "pipeline_artifact_cancelled"
    assert not lease_path.exists()


async def test_parse_failure_releases_only_claimed_owner_and_cleans_lease(
    tmp_path: Path,
) -> None:
    authority = await _claimed_parse_authority(tmp_path)
    session = await authority.coordinator.open(authority.binding)
    assert session.source_path is not None
    lease_path = session.source_path.parents[1]

    await session.finish(PipelineTerminalOutcome.FAILED)
    await session.aclose()

    document = await authority.metadata_store.get_document(
        authority.binding.kb_id, authority.binding.document_id
    )
    assert document.status == "parse_failed"
    assert document.metadata.get("current_parse_claim_token") is None
    assert document.error_code == "pipeline_artifact_failed"
    assert not lease_path.exists()


async def test_late_failure_does_not_overwrite_new_attempt_winner(
    tmp_path: Path,
) -> None:
    authority = await _claimed_build_authority(tmp_path)
    old_session = await authority.coordinator.open(authority.binding)
    assert old_session.sidecar_dir is not None

    await authority.document_service.metadata_store.release_document_build_if_owned(
        authority.binding.kb_id,
        authority.binding.document_id,
        job_id=authority.binding.job_id,
        claim_token=authority.binding.claim_token,
        error_code="retry",
        error_message="retry",
    )
    index_service = IndexBuildService(authority.document_service)
    new_plan = await index_service.create_build_plan(
        authority.binding.kb_id,
        authority.binding.document_id,
        rag=_BuildRAG(),
        force_rechunk=True,
    )
    await index_service.claim_build_queued(
        authority.binding.kb_id,
        job_id="job-new-winner",
        plan=new_plan,
    )
    await index_service.mark_building(
        authority.binding.kb_id,
        authority.binding.document_id,
        job_id="job-new-winner",
        claim_token=new_plan.claim_token,
        plan=new_plan,
    )
    winner_token = new_plan.claim_token

    await old_session.finish(PipelineTerminalOutcome.FAILED)
    await old_session.aclose()

    document = await authority.metadata_store.get_document(
        authority.binding.kb_id, authority.binding.document_id
    )
    assert document.status == "building"
    assert document.metadata["current_build_job_id"] == "job-new-winner"
    assert document.metadata["current_build_claim_token"] == winner_token


async def test_success_handoff_commits_once_then_close_removes_runtime_without_path_leak(
    tmp_path: Path,
) -> None:
    authority = await _claimed_build_authority(tmp_path)
    session = await authority.coordinator.open(authority.binding)
    assert isinstance(session, CoordinatedPipelineArtifactSession)
    assert session.sidecar_dir is not None
    lease_path = session.sidecar_dir.parents[2]

    first = await session.handoff_success(
        parsed_data={"entity_count": 2, "relation_count": 1},
        chunks_count=3,
    )
    second = await session.handoff_success(
        parsed_data={"entity_count": 999, "relation_count": 999},
        chunks_count=999,
    )
    assert second is first
    assert first.outcome is PipelineArtifactCommitOutcome.COMMITTED
    assert first.committed_binding is not None
    assert first.committed_binding.state == "committed"
    assert first.chunks_count == 3
    assert first.entity_count == 2
    assert first.relation_count == 1
    assert session.awaiting_owner_terminalization is True
    assert session.lifecycle_state == "awaiting_h2c_owner_terminalization"
    assert lease_path.is_dir()
    assert session._lease.cleanup_deferred is False
    document = await authority.metadata_store.get_document(
        authority.binding.kb_id, authority.binding.document_id
    )
    assert document.status == "ready"
    assert document.metadata.get("current_build_claim_token") is None
    assert document.chunks_count == 3
    assert document.entity_count == 2
    assert document.relation_count == 1
    await session.aclose()
    await session.aclose()
    assert not lease_path.exists()
    assert ".lightrag-scratch" not in str(document.to_dict())

    redacted = session.redact(
        RuntimeError(
            f"failed at {session.blocks_path} using "
            "s3://access:secret@example.invalid/private"
        )
    )
    assert ".lightrag-scratch" not in redacted
    assert "access:secret" not in redacted
    assert str(session.blocks_path) not in redacted


class _FakeCoordinator:
    def __init__(self) -> None:
        self.generations: list[str] = []

    def materializer_for(self, record: KnowledgeBaseRecord):
        generation = record.generation
        self.generations.append(generation)

        async def callback(binding):
            del binding
            return generation

        setattr(callback, "expected_generation", generation)
        return callback


class _ServerRAG:
    def __init__(self, workspace: str) -> None:
        self.workspace = workspace
        self.pipeline_artifact_materializer = None
        self.events: list[tuple[str, Any]] = []
        self.finalized = False

    async def initialize_storages(self) -> None:
        self.events.append(("initialize", self.pipeline_artifact_materializer))

    async def check_and_migrate_data(self) -> None:
        self.events.append(("migrate", self.pipeline_artifact_materializer))

    async def finalize_storages(self) -> None:
        self.finalized = True

    async def adrop_all_storages(self) -> dict[str, Any]:
        return {"dropped": 0, "failed": 0, "errors": []}


class _MutableKBService:
    def __init__(self, record: KnowledgeBaseRecord) -> None:
        self.record = record

    async def get(self, kb_id: str) -> KnowledgeBaseRecord:
        assert kb_id == self.record.id
        return self.record


def _kb_record(*, generation: str) -> KnowledgeBaseRecord:
    return KnowledgeBaseRecord(
        id="kb_server_h2b2",
        name="server",
        description=None,
        workspace="workspace-server-h2b2",
        status="active",
        active_config_version_id=None,
        owner_id=None,
        tenant_id=None,
        visibility="private",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        generation=generation,
    )


async def test_server_registry_injects_before_initialize_for_every_generation_and_transient(
    tmp_path: Path,
) -> None:
    del tmp_path
    from lightrag.api.lightrag_server import _initialize_kb_lightrag_instance

    coordinator = _FakeCoordinator()
    kb_service = _MutableKBService(_kb_record(generation="generation-a"))
    built: list[_ServerRAG] = []

    async def builder(record: KnowledgeBaseRecord) -> _ServerRAG:
        rag = _ServerRAG(record.workspace)
        built.append(rag)
        return await _initialize_kb_lightrag_instance(
            rag,
            record,
            artifact_storage_mode="object",
            coordinator=coordinator,  # type: ignore[arg-type]
        )

    async def finalizer(rag: _ServerRAG) -> None:
        await rag.finalize_storages()

    registry = LightRAGInstanceRegistry(
        kb_service,  # type: ignore[arg-type]
        builder,
        finalizer,
    )
    first = await registry.get(kb_service.record.id)
    first_callback = first.pipeline_artifact_materializer
    assert callable(first_callback)
    assert first.events[0] == ("initialize", first_callback)

    kb_service.record = replace(kb_service.record, generation="generation-b")
    second = await registry.get(kb_service.record.id)
    second_callback = second.pipeline_artifact_materializer
    assert second is not first
    assert callable(second_callback)
    assert second_callback is not first_callback
    assert second.events[0] == ("initialize", second_callback)

    async with registry.destructive_lock(kb_service.record.id):
        await registry.drop_kb_data(kb_service.record)
    assert coordinator.generations == [
        "generation-a",
        "generation-b",
        "generation-b",
    ]

    local = _ServerRAG("local")
    await _initialize_kb_lightrag_instance(
        local,
        kb_service.record,
        artifact_storage_mode="local",
        coordinator=None,
    )
    assert local.pipeline_artifact_materializer is None
    assert local.events[0] == ("initialize", None)


async def test_create_app_exposes_coordinator_and_injects_registry_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightrag.api import lightrag_server
    from tests.api.test_artifact_storage_foundation import _complete_server_args
    from tests.api.test_artifact_storage_phase2a import _FakeObjectStorage

    reset_canonical_input_root_for_tests()
    args = _complete_server_args(tmp_path, monkeypatch)
    args.artifact_storage_mode = "object"
    storage = _FakeObjectStorage()
    built: list[_ServerRAG] = []

    class CapturedLightRAG(_ServerRAG):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(kwargs["workspace"])
            self.kwargs = kwargs
            self.role_llm_builder = None
            self.ollama_server_infos = kwargs["ollama_server_infos"]

        def register_role_llm_builder(self, builder) -> None:
            self.role_llm_builder = builder

        def get_llm_role_config(self) -> dict[str, Any]:
            return {}

    def build_captured(**kwargs: Any) -> CapturedLightRAG:
        rag = CapturedLightRAG(**kwargs)
        built.append(rag)
        return rag

    monkeypatch.setattr(lightrag_server, "LightRAG", build_captured)
    monkeypatch.setattr(
        lightrag_server, "create_object_storage", lambda config: storage
    )
    monkeypatch.setattr(
        lightrag_server,
        "validate_artifact_storage_configuration",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        lightrag_server,
        "validate_artifact_storage_server_admission",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(lightrag_server, "check_frontend_build", lambda: (True, False))

    app = lightrag_server.create_app(args)
    assert isinstance(
        app.state.pipeline_artifact_coordinator,
        PipelineArtifactCoordinator,
    )
    assert built[0].pipeline_artifact_materializer is None

    await app.state.kb_service.initialize()
    await app.state.metadata_store.initialize()
    record = await app.state.kb_service.create(
        kb_id="kb_app_h2b2",
        name="app-h2b2",
    )
    kb_rag = await app.state.lightrag_registry.get(record.id)
    assert callable(kb_rag.pipeline_artifact_materializer)
    assert kb_rag.events[0] == (
        "initialize",
        kb_rag.pipeline_artifact_materializer,
    )
    await app.state.lightrag_registry.shutdown()
    reset_canonical_input_root_for_tests()


async def test_materializer_callback_captures_only_expected_identity_and_services(
    tmp_path: Path,
) -> None:
    authority = await _claimed_build_authority(tmp_path)
    record = await authority.kb_service.get(authority.binding.kb_id)
    callback = authority.coordinator.materializer_for(record)
    closure = inspect.getclosurevars(callback)

    assert set(closure.nonlocals) == {"self"}
    assert callback.__kwdefaults__ == {
        "_kb_id": record.id,
        "_generation": record.generation,
        "_workspace": record.workspace,
    }
    captured_text = repr((closure.nonlocals, callback.__kwdefaults__))
    for forbidden in ("request", "plan", "execution", "lease"):
        assert forbidden not in captured_text.lower()

    reset_canonical_input_root_for_tests()
