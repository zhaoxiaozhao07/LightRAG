from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Mapping, cast
from urllib.parse import urlsplit

from lightrag.artifact_runtime import (
    PipelineArtifactCommitOutcome,
    PipelineArtifactBinding,
    PipelineArtifactFinalizationResult,
    PipelineArtifactSession,
    canonicalize_pipeline_logical_filename,
)
from lightrag.api.artifact_materialization import ArtifactMaterializationLease
from lightrag.api.commit_reconciliation import (
    MetadataCommitOutcome,
    MetadataCommitOutcomeUnknownError,
    MetadataCommitReconciliation,
    await_cancellation_safe_reconciliation,
)
from lightrag.api.document_lifecycle_service import (
    DocumentLifecycleError,
    DocumentLifecycleService,
    UploadedArtifactObject,
    _active_document_job_error,
    _active_document_job_error_code,
    _artifact_commit_candidate_matches,
    _directory_checksum,
    _file_checksum,
    _resolve_service_attempt_owner,
)
from lightrag.api.kb_service import utc_now_iso
from lightrag.api.metadata_store import (
    ArtifactRecord,
    DocumentNotParsedError,
    DocumentRecord,
    DocumentSnapshotConflictError,
    MetadataRecordNotFoundError,
    document_state_snapshot,
)
from lightrag.utils import generate_track_id, logger
from lightrag.utils_pipeline import (
    resolve_sidecar_uri,
    sidecar_blocks_path,
    sidecar_uri_for,
)

# doc_status status values that mean "still being processed by the pipeline".
# A document showing one of these has NOT reached a terminal build state yet —
# the read-back must wait it out rather than treating it as a failure (a
# concurrent same-KB pipeline drain may be the one that will finish it; see
# the concurrency notes on ``run_build_batch``).
_INFLIGHT_BUILD_STATUSES = frozenset(
    {"pending", "parsing", "analyzing", "processing", "preprocessed"}
)
# Substring the pipeline writes into a doc's ``error_msg`` when a build is
# cancelled mid-drain (see pipeline.py ``_mark_doc_cancelled_in_stage`` /
# merge cancellation). Used to reclassify such docs as cancelled rather than
# a hard build failure.
_CANCEL_ERROR_MARKER = "User cancelled"
# How long the post-drain read-back will wait for each enqueued doc to reach a
# terminal doc_status (processed/failed) before giving up. Generous because a
# concurrent drain that owns the pipeline ``busy`` flag may be processing a
# large merge for many minutes; the common single-flow case returns on the
# first poll (our own drain already finished) so this never sleeps.
DEFAULT_BUILD_DRAIN_TIMEOUT_SECONDS = float(
    os.getenv("KB_BUILD_DRAIN_TIMEOUT_SECONDS", "3600")
)
DEFAULT_BUILD_DRAIN_POLL_SECONDS = float(
    os.getenv("KB_BUILD_DRAIN_POLL_SECONDS", "1.0")
)

AgentProfileDirtyCallback = Callable[[str, str], Awaitable[None]]


@dataclass(slots=True)
class BuildArtifactReference:
    id: str
    artifact_type: str
    checksum: str | None
    size_bytes: int | None
    object_uri: str | None
    object_prefix_uri: str | None
    compatibility_locator: str | None
    blocks_locator: str | None = None


@dataclass(slots=True)
class IndexBuildPlan:
    document: DocumentRecord
    sidecar_artifact: BuildArtifactReference | None
    blocks_artifact: BuildArtifactReference | None
    expected_current_sidecar_artifact_id: str | None
    expected_current_blocks_artifact_id: str | None
    parser_hash: str
    index_hash: str
    process_options: str
    force_rechunk: bool
    force_extract: bool
    force_embedding: bool
    skipped: bool = False
    skip_reason: str | None = None
    expected_status: str | None = None
    expected_source_hash: str | None = None
    expected_parser_hash: str | None = None
    expected_current_parse_generation_id: str | None = None
    expected_index_hash: str | None = None
    claim_token: str | None = None
    kb_generation: str = ""
    job_id: str | None = None
    object_preflight_complete: bool = False

    def __post_init__(self) -> None:
        if self.expected_status is None:
            self.expected_status = self.document.status
            self.expected_source_hash = self.document.source_hash
            self.expected_parser_hash = self.document.parser_hash
            self.expected_current_parse_generation_id = self.document.metadata.get(
                "current_parse_generation_id"
            )
            self.expected_current_sidecar_artifact_id = self.document.metadata.get(
                "current_sidecar_artifact_id"
            )
            self.expected_current_blocks_artifact_id = self.document.metadata.get(
                "current_blocks_artifact_id"
            )
            self.expected_index_hash = self.document.index_hash
        if self.claim_token is None:
            token_key = (
                "pending_build_claim_token"
                if self.document.status == "build_queued"
                else "current_build_claim_token"
                if self.document.status == "building"
                else None
            )
            token = self.document.metadata.get(token_key) if token_key else None
            if isinstance(token, str) and token:
                self.claim_token = token

    @property
    def expected_snapshot(self) -> dict[str, Any]:
        return {
            "status": self.expected_status,
            "source_hash": self.expected_source_hash,
            "parser_hash": self.expected_parser_hash,
            "current_parse_generation_id": (self.expected_current_parse_generation_id),
            "current_sidecar_artifact_id": (self.expected_current_sidecar_artifact_id),
            "current_blocks_artifact_id": self.expected_current_blocks_artifact_id,
            "index_hash": self.expected_index_hash,
        }

    @property
    def force(self) -> bool:
        """Whether any force flag requires bypassing incremental reuse."""
        return self.force_rechunk or self.force_extract or self.force_embedding


@dataclass(slots=True)
class IndexBuildExecution:
    lease: ArtifactMaterializationLease | None
    runtime_sidecar_dir: Path
    runtime_sidecar_uri: str
    runtime_blocks_path: Path
    canonical_sidecar_locator: Path
    canonical_blocks_locator: Path
    expected_current_sidecar_artifact_id: str | None
    expected_current_blocks_artifact_id: str | None
    initial_sidecar_checksum: str
    initial_blocks_checksum: str
    pipeline_started: bool = False
    terminal: bool = False

    def defer_cleanup(self) -> None:
        if self.lease is not None and not self.lease.cleanup_deferred:
            self.lease.defer_cleanup()

    def cleanup(self) -> None:
        if self.lease is not None and not self.lease.cleanup_deferred:
            self.lease.cleanup()

    def durable_error_message(self, error: object) -> str:
        message = str(error)
        if self.lease is None:
            return message
        replacements = (
            (
                self.runtime_sidecar_dir.resolve(strict=False).as_uri(),
                self.canonical_sidecar_locator.resolve(strict=False).as_uri(),
            ),
            (str(self.runtime_sidecar_dir), str(self.canonical_sidecar_locator)),
            (
                self.lease.path.resolve(strict=False).as_uri(),
                "materialization://redacted",
            ),
            (str(self.lease.path), "<artifact-materialization>"),
        )
        for runtime_value, durable_value in replacements:
            message = message.replace(runtime_value, durable_value)
        return message.replace(".lightrag-scratch", "artifact-materialization")


@dataclass(slots=True)
class PipelineBuildPreflight:
    """Short object-input validation lease, always closed before engine clear."""

    binding: PipelineArtifactBinding
    session: PipelineArtifactSession
    plan: IndexBuildPlan
    cleaned: bool = False

    @property
    def runtime_sidecar_dir(self) -> Path:
        sidecar_dir = self.session.sidecar_dir
        if sidecar_dir is None:
            raise DocumentLifecycleError("Build preflight has no runtime sidecar")
        return sidecar_dir

    @property
    def runtime_blocks_path(self) -> Path:
        blocks_path = self.session.blocks_path
        if blocks_path is None:
            raise DocumentLifecycleError("Build preflight has no runtime blocks")
        return blocks_path

    def defer_cleanup(self) -> None:
        if not self.cleaned:
            self.session.defer_cleanup()

    async def cleanup(self) -> None:
        if self.cleaned:
            return
        close_task = asyncio.create_task(self.session.aclose())
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError:
            await close_task
            self.cleaned = True
            self.plan.object_preflight_complete = True
            raise
        self.cleaned = True
        self.plan.object_preflight_complete = True

    async def aclose(self) -> None:
        await self.cleanup()

    def durable_error_message(self, error: object) -> str:
        return self.session.redact(error)


@dataclass(slots=True)
class BatchIndexBuildPlan:
    batch_id: str
    plans: list[IndexBuildPlan]
    failures: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class IndexBuildResult:
    document: DocumentRecord
    skipped: bool
    chunks_count: int | None
    entity_count: int | None
    relation_count: int | None
    index_hash: str


class ImmutableArtifactObjectConflictError(DocumentLifecycleError):
    """A deterministic immutable object key already contains other bytes."""


class IndexBuildService:
    """Drive KG / index construction on top of parsed artifacts.

    Reuses LightRAG's pipeline (``apipeline_enqueue_documents`` +
    ``apipeline_process_enqueue_documents``) for chunking, entity/relation
    extraction, embedding, and KG merge. The service is responsible for
    incremental ingestion semantics: hash-based skip when source_hash +
    parser_hash + index_hash all match, else feed the document into the
    pipeline and stamp the new index_hash on success.
    """

    def __init__(
        self,
        document_service: DocumentLifecycleService,
        agent_profile_dirty_callback: AgentProfileDirtyCallback | None = None,
    ):
        self._document_service = document_service
        self._build_drain_timeout = DEFAULT_BUILD_DRAIN_TIMEOUT_SECONDS
        self._build_drain_poll = DEFAULT_BUILD_DRAIN_POLL_SECONDS
        self._agent_profile_dirty_callback = agent_profile_dirty_callback

    def set_agent_profile_dirty_callback(
        self, callback: AgentProfileDirtyCallback | None
    ) -> None:
        self._agent_profile_dirty_callback = callback

    @property
    def object_authoritative(self) -> bool:
        return bool(getattr(self._document_service, "object_authoritative", False))

    async def create_build_plan(
        self,
        kb_id: str,
        document_id: str,
        *,
        rag: Any,
        force_rechunk: bool = False,
        force_extract: bool = False,
        force_embedding: bool = False,
    ) -> IndexBuildPlan:
        kb_record = await self._document_service.kb_service.get(kb_id)
        document = await self._document_service.get_document(kb_id, document_id)
        if document.workspace != kb_record.workspace:
            raise DocumentLifecycleError(
                "Build document workspace does not match the current KB generation"
            )
        if document.status not in {
            "parsed",
            "ready",
            "build_failed",
            "build_queued",
            "building",
        }:
            raise DocumentNotParsedError(document_id, document.status)
        if not document.parser_hash:
            raise DocumentNotParsedError(document_id, document.status)

        (
            sidecar_artifact,
            blocks_artifact,
            expected_sidecar_id,
            expected_blocks_id,
        ) = await self._resolve_artifacts(kb_id, document)
        index_hash = compute_index_hash(rag)
        process_options_value = document.metadata.get("process_options")
        process_options = (
            str(process_options_value) if process_options_value is not None else ""
        )
        force = force_rechunk or force_extract or force_embedding
        skipped = False
        skip_reason: str | None = None
        if (
            not force
            and document.status == "ready"
            and document.index_hash == index_hash
        ):
            skipped = True
            skip_reason = "index_hash_match"
        return IndexBuildPlan(
            document=document,
            sidecar_artifact=sidecar_artifact,
            blocks_artifact=blocks_artifact,
            expected_current_sidecar_artifact_id=expected_sidecar_id,
            expected_current_blocks_artifact_id=expected_blocks_id,
            expected_status=document.status,
            expected_source_hash=document.source_hash,
            expected_parser_hash=document.parser_hash,
            expected_current_parse_generation_id=document.metadata.get(
                "current_parse_generation_id"
            ),
            expected_index_hash=document.index_hash,
            parser_hash=document.parser_hash,
            index_hash=index_hash,
            process_options=process_options,
            force_rechunk=force_rechunk,
            force_extract=force_extract,
            force_embedding=force_embedding,
            skipped=skipped,
            skip_reason=skip_reason,
            kb_generation=kb_record.generation,
        )

    async def create_batch_build_plan(
        self,
        kb_id: str,
        document_ids: list[str],
        *,
        rag: Any,
        force_rechunk: bool = False,
        force_extract: bool = False,
        force_embedding: bool = False,
    ) -> BatchIndexBuildPlan:
        plans: list[IndexBuildPlan] = []
        failures: list[dict[str, Any]] = []
        for document_id in document_ids:
            try:
                plan = await self.create_build_plan(
                    kb_id,
                    document_id,
                    rag=rag,
                    force_rechunk=force_rechunk,
                    force_extract=force_extract,
                    force_embedding=force_embedding,
                )
                plans.append(plan)
            except MetadataRecordNotFoundError as exc:
                failures.append(
                    _build_failure_item(
                        document_id,
                        error_code="document_not_found",
                        error_message=str(exc),
                    )
                )
            except DocumentNotParsedError as exc:
                failures.append(
                    _build_failure_item(
                        document_id,
                        error_code="document_not_parsed",
                        error_message=str(exc),
                        current_status=exc.current_status,
                    )
                )
            except FileNotFoundError as exc:
                failures.append(
                    _build_failure_item(
                        document_id,
                        error_code="parse_artifact_missing",
                        error_message=str(exc),
                    )
                )
        return BatchIndexBuildPlan(
            batch_id=generate_track_id("batch"),
            plans=plans,
            failures=failures,
        )

    async def claim_build_queued(
        self,
        kb_id: str,
        document_id: str | None = None,
        *,
        job_id: str,
        plan: IndexBuildPlan | None = None,
    ) -> DocumentRecord:
        if plan is None:
            if self._document_service.object_authoritative:
                raise DocumentLifecycleError(
                    "Object build claims require an explicit metadata-only plan"
                )
            if not document_id:
                raise ValueError(
                    "document_id is required when a local compatibility claim has no plan"
                )
            document = await self._document_service.get_document(kb_id, document_id)
            expected_snapshot = document_state_snapshot(document)
            index_hash = document.index_hash or ""
            force_rechunk = bool(document.metadata.get("force_rechunk", False))
            force_extract = bool(document.metadata.get("force_extract", False))
            force_embedding = bool(document.metadata.get("force_embedding", False))
        else:
            if document_id is not None and document_id != plan.document.id:
                raise ValueError("Build plan document does not match document_id")
            document_id = plan.document.id
            expected_snapshot = plan.expected_snapshot
            index_hash = plan.index_hash
            force_rechunk = plan.force_rechunk
            force_extract = plan.force_extract
            force_embedding = plan.force_embedding

        try:
            document = (
                await self._document_service.metadata_store.claim_document_build_queued(
                    kb_id,
                    document_id,
                    metadata_patch={
                        "pending_build_job_id": job_id,
                        "pending_index_hash": index_hash,
                        "force_rechunk": force_rechunk,
                        "force_extract": force_extract,
                        "force_embedding": force_embedding,
                    },
                    expected_snapshot=expected_snapshot,
                    claim_token=plan.claim_token if plan is not None else None,
                )
            )
        except DocumentSnapshotConflictError:
            current = await self._document_service.get_document(kb_id, document_id)
            active_error = _active_document_job_error(current)
            if active_error is not None:
                raise active_error
            raise
        token = document.metadata.get("pending_build_claim_token")
        if not isinstance(token, str) or not token:
            raise DocumentLifecycleError("Build claim did not persist an attempt token")
        if plan is not None:
            plan.claim_token = token
            plan.job_id = job_id
        return document

    async def claim_batch_build_queued(
        self, kb_id: str, *, job_id: str, plans: list[IndexBuildPlan]
    ) -> tuple[list[DocumentRecord], list[dict[str, Any]]]:
        claims = [
            (
                plan.document.id,
                {
                    "pending_build_job_id": job_id,
                    "pending_index_hash": plan.index_hash,
                    "force_rechunk": plan.force_rechunk,
                    "force_extract": plan.force_extract,
                    "force_embedding": plan.force_embedding,
                },
                plan.expected_snapshot,
                plan.claim_token,
            )
            for plan in plans
        ]
        (
            documents,
            failures,
        ) = await self._document_service.metadata_store.claim_documents_build_queued(
            kb_id, claims
        )
        normalized_failures: list[dict[str, Any]] = []
        for failure in failures:
            normalized = failure
            if failure.get("error_code") == "document_snapshot_conflict":
                document_id = failure.get("document_id")
                if isinstance(document_id, str):
                    current = await self._document_service.get_document(
                        kb_id, document_id
                    )
                    active_error = _active_document_job_error(current)
                    if active_error is not None:
                        normalized = {
                            **failure,
                            "error_code": _active_document_job_error_code(active_error),
                            "error_message": str(active_error),
                            "existing_job_id": active_error.existing_job_id,
                        }
            normalized_failures.append(normalized)
        plans_by_id = {plan.document.id: plan for plan in plans}
        for document in documents:
            token = document.metadata.get("pending_build_claim_token")
            if not isinstance(token, str) or not token:
                raise DocumentLifecycleError(
                    "Build claim did not persist an attempt token"
                )
            plans_by_id[document.id].claim_token = token
            plans_by_id[document.id].job_id = job_id
        return documents, normalized_failures

    async def mark_building(
        self,
        kb_id: str,
        document_id: str,
        *,
        job_id: str,
        claim_token: str | None = None,
        plan: IndexBuildPlan | None = None,
    ) -> DocumentRecord:
        if plan is not None:
            if plan.job_id is not None and plan.job_id != job_id:
                raise DocumentLifecycleError(
                    "Build plan job identity does not match the running owner"
                )
            plan.job_id = job_id
        token = claim_token or (plan.claim_token if plan is not None else None)
        if self._document_service.object_authoritative and token is None:
            raise DocumentLifecycleError(
                "Object build execution requires an explicit claim token"
            )
        document = await self._document_service.metadata_store.mark_document_building(
            kb_id,
            document_id,
            metadata_patch={
                "current_build_job_id": job_id,
                "build_started_at": utc_now_iso(),
            },
            job_id=job_id if token is not None else None,
            claim_token=token,
        )
        resolved_token = document.metadata.get("current_build_claim_token")
        if plan is not None and isinstance(resolved_token, str) and resolved_token:
            plan.claim_token = resolved_token
        return document

    def build_artifact_binding(self, plan: IndexBuildPlan) -> PipelineArtifactBinding:
        return _claimed_build_artifact_binding(plan)

    async def materialize_build_preflight(
        self, plan: IndexBuildPlan
    ) -> PipelineBuildPreflight:
        """Validate exact object inputs in a short processing-independent lease."""

        if not self.object_authoritative:
            raise DocumentLifecycleError(
                "Object build preflight is unavailable in local artifact mode"
            )
        if plan.skipped:
            raise DocumentLifecycleError("Skipped build plans are not materialized")
        binding = self.build_artifact_binding(plan)
        # Import locally to avoid the coordinator -> IndexBuildService type cycle.
        from lightrag.api.pipeline_artifact_coordinator import (
            PipelineArtifactCoordinator,
        )

        coordinator = PipelineArtifactCoordinator(
            self._document_service.kb_service,
            self._document_service,
            self,
        )
        session = await coordinator.open(binding)
        if session.sidecar_dir is None or session.blocks_path is None:
            await session.aclose()
            raise DocumentLifecycleError(
                "Object build preflight returned no exact sidecar/blocks runtime"
            )
        return PipelineBuildPreflight(binding=binding, session=session, plan=plan)

    async def materialize_build_execution(
        self, plan: IndexBuildPlan
    ) -> IndexBuildExecution:
        """Resolve local-mode parsed artifacts into runtime inputs."""

        if self.object_authoritative:
            raise DocumentLifecycleError(
                "Object builds require materialize_build_preflight(), not a "
                "terminal-lifetime IndexBuildExecution"
            )
        if plan.skipped:
            raise DocumentLifecycleError("Skipped build plans are not materialized")
        sidecar = plan.sidecar_artifact
        if sidecar is None:
            raise FileNotFoundError(
                f"Document {plan.document.id!r} has no sidecar artifact for build"
            )
        sidecar_path = _local_sidecar_path(
            self._document_service.source_root,
            plan.document,
            sidecar,
        )
        blocks_path = _local_blocks_path(sidecar_path, plan.blocks_artifact, sidecar)
        return IndexBuildExecution(
            lease=None,
            runtime_sidecar_dir=sidecar_path,
            runtime_sidecar_uri=sidecar_uri_for(sidecar_path),
            runtime_blocks_path=blocks_path,
            canonical_sidecar_locator=sidecar_path,
            canonical_blocks_locator=blocks_path,
            expected_current_sidecar_artifact_id=(
                plan.expected_current_sidecar_artifact_id
            ),
            expected_current_blocks_artifact_id=(
                plan.expected_current_blocks_artifact_id
            ),
            initial_sidecar_checksum=_directory_checksum(sidecar_path),
            initial_blocks_checksum=_file_checksum(blocks_path),
        )

    async def run_build(
        self,
        rag: Any,
        plan: IndexBuildPlan,
        execution: IndexBuildExecution | None = None,
    ) -> dict[str, Any]:
        """Enqueue one local build or one durable object binding."""

        if plan.skipped:
            return {
                "skipped": True,
                "skip_reason": plan.skip_reason,
                "chunks_count": plan.document.chunks_count,
                "entity_count": plan.document.entity_count,
                "relation_count": plan.document.relation_count,
            }
        if not plan.document.lightrag_doc_id:
            raise DocumentLifecycleError("Document has no LightRAG document identity")

        binding: PipelineArtifactBinding | None = None
        if self.object_authoritative:
            if execution is not None:
                raise DocumentLifecycleError(
                    "Object build enqueue must not retain a materialized execution"
                )
            if not plan.object_preflight_complete:
                raise DocumentLifecycleError(
                    "Object build preflight must finish before engine mutation"
                )
            binding = self.build_artifact_binding(plan)
            logical_filename = canonicalize_pipeline_logical_filename(
                plan.document.source_name
            )
        else:
            if execution is None:
                execution = await self.materialize_build_execution(plan)
            logical_filename = _kb_unique_basename(plan)

        if _build_needs_engine_clear(plan):
            deletion_result = await rag.adelete_by_doc_id(plan.document.lightrag_doc_id)
            status = getattr(deletion_result, "status", None)
            if status not in {"success", "not_found"}:
                raise RuntimeError(
                    getattr(deletion_result, "message", None)
                    or "Build could not clear existing LightRAG doc "
                    f"{plan.document.lightrag_doc_id!r} before re-enqueue "
                    f"(status={status})"
                )

        track_id = generate_track_id(f"build_{plan.document.id}")
        try:
            if self.object_authoritative:
                assert binding is not None
                await rag.apipeline_enqueue_documents(
                    input=[""],
                    ids=[plan.document.lightrag_doc_id],
                    file_paths=[logical_filename],
                    track_id=track_id,
                    docs_format="lightrag",
                    parse_engine=plan.document.metadata.get("parse_engine"),
                    process_options=plan.process_options or None,
                    artifact_bindings=[binding],
                )
            else:
                assert execution is not None
                execution.pipeline_started = True
                await rag.apipeline_enqueue_documents(
                    input=[""],
                    ids=[plan.document.lightrag_doc_id],
                    file_paths=[logical_filename],
                    track_id=track_id,
                    docs_format="lightrag",
                    lightrag_document_paths=[execution.runtime_sidecar_uri],
                    parse_engine=plan.document.metadata.get("parse_engine"),
                    process_options=plan.process_options or None,
                )
            await rag.apipeline_process_enqueue_documents()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.object_authoritative:
                return {
                    "error_code": "build_outcome_unknown",
                    "outcome_unknown": True,
                    "error_message": (
                        "Object build enqueue/drain outcome is unknown: "
                        f"{type(exc).__name__}"
                    ),
                }
            raise
        return await self._resolve_build_result(rag, plan, execution=execution)

    async def run_build_batch(
        self,
        rag: Any,
        plans: list[IndexBuildPlan],
        executions: dict[str, IndexBuildExecution],
        *,
        job_id: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Bulk enqueue local executions or durable object bindings."""

        results: dict[str, dict[str, Any]] = {}
        runnable: list[IndexBuildPlan] = []
        bindings: dict[str, PipelineArtifactBinding] = {}
        for plan in plans:
            if plan.skipped:
                results[plan.document.id] = {
                    "skipped": True,
                    "skip_reason": plan.skip_reason,
                    "chunks_count": plan.document.chunks_count,
                    "entity_count": plan.document.entity_count,
                    "relation_count": plan.document.relation_count,
                }
                continue
            if not plan.document.lightrag_doc_id:
                results[plan.document.id] = {
                    "error_code": "build_failed",
                    "error_message": "Document has no LightRAG document identity",
                }
                continue
            if self.object_authoritative:
                if not plan.object_preflight_complete:
                    results[plan.document.id] = {
                        "error_code": "build_failed",
                        "error_message": (
                            "Object build preflight must finish before engine mutation"
                        ),
                    }
                    continue
                try:
                    bindings[plan.document.id] = self.build_artifact_binding(plan)
                except Exception as exc:
                    results[plan.document.id] = {
                        "error_code": "build_failed",
                        "error_message": str(exc),
                    }
                    continue
            elif plan.document.id not in executions:
                results[plan.document.id] = {
                    "error_code": "build_failed",
                    "error_message": (
                        f"Document {plan.document.id!r} build was not materialized"
                    ),
                }
                continue
            runnable.append(plan)

        for plan in runnable:
            if not _build_needs_engine_clear(plan):
                continue
            try:
                deletion_result = await rag.adelete_by_doc_id(
                    plan.document.lightrag_doc_id
                )
                status = getattr(deletion_result, "status", None)
                if status not in {"success", "not_found"}:
                    raise RuntimeError(
                        getattr(deletion_result, "message", None)
                        or "Build could not clear existing LightRAG doc "
                        f"{plan.document.lightrag_doc_id!r} before re-enqueue "
                        f"(status={status})"
                    )
            except Exception as exc:
                results[plan.document.id] = {
                    "error_code": "build_failed",
                    "error_message": str(exc),
                }
        runnable = [plan for plan in runnable if plan.document.id not in results]
        if not runnable:
            return results

        batch_track_id = generate_track_id(
            f"build_batch_{job_id or runnable[0].document.id}"
        )
        ids = [plan.document.lightrag_doc_id for plan in runnable]
        parse_engines = [
            plan.document.metadata.get("parse_engine") or "" for plan in runnable
        ]
        process_options = [plan.process_options or "" for plan in runnable]
        try:
            if self.object_authoritative:
                await rag.apipeline_enqueue_documents(
                    input=[""] * len(runnable),
                    ids=ids,
                    file_paths=[
                        canonicalize_pipeline_logical_filename(
                            plan.document.source_name
                        )
                        for plan in runnable
                    ],
                    track_id=batch_track_id,
                    docs_format="lightrag",
                    parse_engine=parse_engines,
                    process_options=process_options,
                    artifact_bindings=[bindings[plan.document.id] for plan in runnable],
                )
            else:
                for plan in runnable:
                    executions[plan.document.id].pipeline_started = True
                await rag.apipeline_enqueue_documents(
                    input=[""] * len(runnable),
                    ids=ids,
                    file_paths=[_kb_unique_basename(plan) for plan in runnable],
                    track_id=batch_track_id,
                    docs_format="lightrag",
                    lightrag_document_paths=[
                        executions[plan.document.id].runtime_sidecar_uri
                        for plan in runnable
                    ],
                    parse_engine=parse_engines,
                    process_options=process_options,
                )
            await rag.apipeline_process_enqueue_documents()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self.object_authoritative:
                raise
            for plan in runnable:
                results[plan.document.id] = {
                    "error_code": "build_outcome_unknown",
                    "outcome_unknown": True,
                    "error_message": (
                        "Object build enqueue/drain outcome is unknown: "
                        f"{type(exc).__name__}"
                    ),
                }
            return results

        for plan in runnable:
            results[plan.document.id] = await self._resolve_build_result(
                rag,
                plan,
                execution=executions.get(plan.document.id),
            )
        return results

    async def _resolve_build_result(
        self,
        rag: Any,
        plan: IndexBuildPlan,
        *,
        execution: IndexBuildExecution | None,
    ) -> dict[str, Any]:
        doc_status_storage = getattr(rag, "doc_status", None)
        if doc_status_storage is None:
            if self.object_authoritative:
                await asyncio.Future()
            if execution is not None:
                execution.terminal = True
            return {
                "skipped": False,
                "chunks_count": None,
                "entity_count": None,
                "relation_count": None,
            }
        if not plan.document.lightrag_doc_id:
            raise DocumentLifecycleError("Document has no LightRAG document identity")
        row = await self._await_doc_terminal(rag, plan.document.lightrag_doc_id)
        if row is None:
            return {
                "error_code": "build_failed",
                "outcome_unknown": True,
                "error_message": (
                    f"Document {plan.document.id!r} build did not create a "
                    "doc_status row"
                ),
            }
        status = row.get("status")
        if status == "processed":
            if self.object_authoritative:
                try:
                    confirmed = await self.confirm_pipeline_build_completion(
                        rag, plan, terminal_row=row
                    )
                except Exception as exc:
                    return {
                        "error_code": "build_confirmation_failed",
                        "outcome_unknown": True,
                        "error_message": str(exc),
                    }
                await self._notify_agent_profile_dirty(
                    plan.document.kb_id, plan.document.id
                )
                return {
                    "skipped": False,
                    "chunks_count": row.get("chunks_count"),
                    "entity_count": row.get("entity_count"),
                    "relation_count": row.get("relation_count"),
                    "_confirmed_document": confirmed,
                }
            if execution is not None:
                execution.terminal = True
            return {
                "skipped": False,
                "chunks_count": row.get("chunks_count"),
                "entity_count": row.get("entity_count"),
                "relation_count": row.get("relation_count"),
            }

        error_msg = str(row.get("error_msg") or "")
        if status in _INFLIGHT_BUILD_STATUSES:
            return {
                "error_code": "build_failed",
                "outcome_unknown": True,
                "error_message": (
                    f"Document {plan.document.id!r} build timed out waiting for "
                    f"the pipeline drain (status={status})"
                ),
            }
        if self.object_authoritative:
            if not await self._object_terminal_release_is_visible(plan, row):
                return {
                    "error_code": "build_confirmation_failed",
                    "outcome_unknown": True,
                    "error_message": (
                        "Object build terminal row is not fenced to the claimed attempt"
                    ),
                }
            if _CANCEL_ERROR_MARKER in error_msg:
                return {
                    "cancelled": True,
                    "owner_terminalized": True,
                    "error_message": error_msg,
                }
            return {
                "error_code": "build_failed",
                "owner_terminalized": True,
                "error_message": (
                    f"Document {plan.document.id!r} build did not reach processed "
                    f"(status={status}: {error_msg})"
                ),
            }
        if execution is not None:
            execution.terminal = True
        if _CANCEL_ERROR_MARKER in error_msg:
            return {"cancelled": True, "error_message": error_msg}
        return {
            "error_code": "build_failed",
            "error_message": (
                f"Document {plan.document.id!r} build did not reach processed "
                f"(status={status}: {error_msg})"
            ),
        }

    async def confirm_pipeline_build_completion(
        self,
        rag: Any,
        plan: IndexBuildPlan,
        *,
        terminal_row: Mapping[str, Any],
    ) -> DocumentRecord:
        """Read-only verification of processing-owner C2 completion."""

        if terminal_row.get("status") != "processed":
            raise DocumentLifecycleError("Pipeline build terminal is not processed")
        terminal_metadata = terminal_row.get("metadata")
        if not isinstance(terminal_metadata, Mapping):
            raise DocumentLifecycleError("Pipeline build terminal metadata is missing")
        if terminal_metadata.get("pipeline_attempt_token") != plan.claim_token:
            raise DocumentLifecycleError(
                "Pipeline build terminal attempt token does not match the claim"
            )

        full_docs = getattr(rag, "full_docs", None)
        get_by_id = getattr(full_docs, "get_by_id", None)
        if not callable(get_by_id) or not plan.document.lightrag_doc_id:
            raise DocumentLifecycleError(
                "Pipeline build completion has no durable full_docs authority"
            )
        get_record = cast(Callable[[str], Awaitable[Any]], get_by_id)
        full_doc = await get_record(plan.document.lightrag_doc_id)
        if not isinstance(full_doc, Mapping):
            raise DocumentLifecycleError("Committed pipeline build binding is missing")
        raw_binding = full_doc.get("artifact_binding")
        if not isinstance(raw_binding, Mapping):
            raise DocumentLifecycleError("Committed pipeline build binding is missing")
        binding = PipelineArtifactBinding.from_mapping(
            raw_binding, expected_workspace=plan.document.workspace
        )
        _validate_committed_build_artifact_binding(binding, plan)

        kb_record = await self._document_service.kb_service.get(plan.document.kb_id)
        if kb_record.generation != plan.kb_generation:
            raise DocumentLifecycleError(
                "Pipeline build completed against a replaced KB generation"
            )
        document = await self._document_service.get_document(
            plan.document.kb_id, plan.document.id
        )
        metadata = document.metadata
        mismatches: list[str] = []
        if document.status != "ready":
            mismatches.append("status")
        if document.index_hash != plan.index_hash:
            mismatches.append("index_hash")
        if metadata.get("current_build_generation_id") != plan.claim_token:
            mismatches.append("current_build_generation_id")
        if metadata.get("last_build_job_id") != plan.job_id:
            mismatches.append("last_build_job_id")
        if metadata.get("current_sidecar_artifact_id") != binding.sidecar_artifact_id:
            mismatches.append("current_sidecar_artifact_id")
        if metadata.get("current_blocks_artifact_id") != binding.blocks_artifact_id:
            mismatches.append("current_blocks_artifact_id")
        for owner_key in (
            "pending_build_job_id",
            "pending_build_claim_token",
            "current_build_job_id",
            "current_build_claim_token",
        ):
            if metadata.get(owner_key) is not None:
                mismatches.append(owner_key)
        for field_name in ("chunks_count", "entity_count", "relation_count"):
            if terminal_row.get(field_name) != getattr(document, field_name):
                mismatches.append(field_name)
        if mismatches:
            raise DocumentLifecycleError(
                "Pipeline build completion confirmation mismatch: "
                + ", ".join(sorted(set(mismatches)))
            )
        return document

    async def _object_terminal_release_is_visible(
        self, plan: IndexBuildPlan, terminal_row: Mapping[str, Any]
    ) -> bool:
        metadata = terminal_row.get("metadata")
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("pipeline_attempt_token") != plan.claim_token
        ):
            return False
        document = await self._document_service.get_document(
            plan.document.kb_id, plan.document.id
        )
        current_token = document.metadata.get("current_build_claim_token")
        current_job = document.metadata.get("current_build_job_id")
        if current_token == plan.claim_token and current_job == plan.job_id:
            return False
        if document.status == "build_failed":
            return True
        return bool(current_token is not None and current_token != plan.claim_token)

    async def _await_doc_terminal(
        self, rag: Any, lightrag_doc_id: str
    ) -> dict[str, Any] | None:
        """Poll a doc's doc_status row until it leaves the inflight set.

        Returns the final row (processed/failed), the last inflight row on
        timeout, or None when the row is absent. Never sleeps when the row is
        already terminal on the first read (the single-flow fast path).
        """
        return await _await_doc_status_terminal(
            rag,
            lightrag_doc_id,
            timeout=self._build_drain_timeout,
            poll_interval=self._build_drain_poll,
            wait_until_terminal=self.object_authoritative,
        )

    async def collect_doc_status(
        self, rag: Any, plan: IndexBuildPlan
    ) -> dict[str, Any]:
        """Public read-back of LightRAG ``doc_status`` for a single plan.

        Exposed for callers that drive their own pipeline (e.g., the batch
        aggregate flow) and need to fetch the final ``chunks/entities/
        relations`` counts after their own drain.
        """
        return await _collect_doc_status(rag, plan)

    async def finalize_build_runtime_references(
        self,
        rag: Any,
        plan: IndexBuildPlan,
        execution: IndexBuildExecution,
    ) -> None:
        """Replace terminal pipeline scratch references before lease cleanup."""

        if not self._document_service.object_authoritative:
            return
        full_docs = getattr(rag, "full_docs", None)
        get_by_id = getattr(full_docs, "get_by_id", None)
        upsert = getattr(full_docs, "upsert", None)
        if not callable(get_by_id) or not callable(upsert):
            return
        if not plan.document.lightrag_doc_id:
            return
        get_record = cast(Callable[[str], Awaitable[Any]], get_by_id)
        upsert_records = cast(Callable[[dict[str, Any]], Awaitable[Any]], upsert)
        existing = await get_record(plan.document.lightrag_doc_id)
        if not isinstance(existing, dict):
            return
        payload = dict(existing)
        payload["file_path"] = _kb_unique_basename(plan)
        payload["sidecar_location"] = sidecar_uri_for(
            execution.canonical_sidecar_locator
        )
        await upsert_records({plan.document.lightrag_doc_id: payload})
        callback = getattr(full_docs, "index_done_callback", None)
        if callable(callback):
            done_callback = cast(Callable[[], Awaitable[Any]], callback)
            await done_callback()

    async def complete_build(
        self,
        kb_id: str,
        document_id: str,
        *,
        job_id: str,
        plan: IndexBuildPlan,
        run_result: dict[str, Any],
        execution: IndexBuildExecution | None = None,
        propagate_cancellation_after_commit: bool = True,
    ) -> DocumentRecord:
        current_document = await self._document_service.get_document(kb_id, document_id)
        resolved_token, phase = _resolve_service_attempt_owner(
            current_document,
            operation="build",
            job_id=job_id,
            claim_token=plan.claim_token,
            strict=True,
        )
        if resolved_token is not None:
            plan.claim_token = resolved_token
        if phase != "current":
            await self.mark_building(
                kb_id,
                document_id,
                job_id=job_id,
                claim_token=resolved_token,
                plan=plan,
            )
        elif resolved_token is None:
            raise DocumentLifecycleError(
                "Build completion could not attach a token to a legacy running attempt"
            )
        generation_id = plan.claim_token
        if not isinstance(generation_id, str) or not generation_id:
            raise DocumentLifecycleError(
                "Build completion could not resolve an attempt token"
            )
        current_sidecar_id = (
            plan.sidecar_artifact.id if plan.sidecar_artifact is not None else None
        )
        current_blocks_id = (
            plan.blocks_artifact.id if plan.blocks_artifact is not None else None
        )
        pointer_patch = {
            "current_sidecar_artifact_id": current_sidecar_id,
            "current_blocks_artifact_id": current_blocks_id,
        }
        if plan.skipped or run_result.get("skipped"):
            metadata_patch = {
                "last_build_job_id": job_id,
                "last_built_at": utc_now_iso(),
                "build_skipped": True,
                "build_skip_reason": plan.skip_reason or "index_hash_match",
                "current_build_generation_id": generation_id,
                "pending_build_job_id": None,
                "current_build_job_id": None,
                "pending_index_hash": None,
                **pointer_patch,
            }
            return await self._document_service.metadata_store.complete_document_build(
                kb_id,
                document_id,
                index_hash=plan.index_hash,
                metadata_patch=metadata_patch,
                job_id=job_id,
                claim_token=generation_id,
                expected_snapshot=plan.expected_snapshot,
            )
        chunks_count = run_result.get("chunks_count")
        entity_count = run_result.get("entity_count")
        relation_count = run_result.get("relation_count")
        metadata_patch = {
            "last_build_job_id": job_id,
            "last_built_at": utc_now_iso(),
            "build_skipped": False,
            "current_build_generation_id": generation_id,
            "pending_build_job_id": None,
            "current_build_job_id": None,
            "pending_index_hash": None,
            **pointer_patch,
        }
        if self._document_service.object_authoritative and execution is not None:
            metadata_patch["blocks_path"] = str(execution.canonical_blocks_locator)
        promoted_artifacts: list[ArtifactRecord] = []
        uploaded: list[UploadedArtifactObject] = []
        if self._document_service.object_authoritative:
            if execution is None:
                raise DocumentLifecycleError(
                    "Object build completion requires a materialized execution"
                )
            promoted_artifacts, uploaded = await self._promote_changed_artifacts(
                plan, execution, generation_id=generation_id
            )
        if self._document_service.object_authoritative:
            promoted_sidecar = next(
                (
                    artifact
                    for artifact in promoted_artifacts
                    if artifact.artifact_type == "sidecar"
                ),
                None,
            )
            promoted_blocks = next(
                (
                    artifact
                    for artifact in promoted_artifacts
                    if artifact.artifact_type == "blocks"
                ),
                None,
            )
            final_sidecar_id = (
                promoted_sidecar.id
                if promoted_sidecar is not None
                else current_sidecar_id
            )
            if final_sidecar_id is None:
                raise DocumentLifecycleError(
                    "Object build completion requires a current sidecar artifact"
                )
            final_blocks_id = (
                promoted_blocks.id if promoted_blocks is not None else current_blocks_id
            )
            metadata_patch["current_sidecar_artifact_id"] = final_sidecar_id
            metadata_patch["current_blocks_artifact_id"] = final_blocks_id
            try:
                (
                    document,
                    _created,
                ) = await self._document_service.metadata_store.complete_document_build_with_artifact_promotion(
                    kb_id,
                    document_id,
                    index_hash=plan.index_hash,
                    expected_current_sidecar_artifact_id=(
                        plan.expected_current_sidecar_artifact_id
                    ),
                    expected_current_blocks_artifact_id=(
                        plan.expected_current_blocks_artifact_id
                    ),
                    current_sidecar_artifact_id=final_sidecar_id,
                    current_blocks_artifact_id=final_blocks_id,
                    artifacts=promoted_artifacts,
                    chunks_count=chunks_count,
                    entity_count=entity_count,
                    relation_count=relation_count,
                    metadata_patch=metadata_patch,
                    job_id=job_id,
                    claim_token=generation_id,
                    expected_snapshot=plan.expected_snapshot,
                )
            except (Exception, asyncio.CancelledError) as commit_error:
                document = await self._reconcile_build_promotion_commit_exception(
                    kb_id=kb_id,
                    document_id=document_id,
                    artifacts=promoted_artifacts,
                    uploaded=uploaded,
                    job_id=job_id,
                    claim_token=generation_id,
                    index_hash=plan.index_hash,
                    current_sidecar_artifact_id=final_sidecar_id,
                    current_blocks_artifact_id=final_blocks_id,
                    commit_error=commit_error,
                    propagate_cancellation=propagate_cancellation_after_commit,
                )
        else:
            document = (
                await self._document_service.metadata_store.complete_document_build(
                    kb_id,
                    document_id,
                    index_hash=plan.index_hash,
                    chunks_count=chunks_count,
                    entity_count=entity_count,
                    relation_count=relation_count,
                    metadata_patch=metadata_patch,
                    job_id=job_id,
                    claim_token=generation_id,
                    expected_snapshot=plan.expected_snapshot,
                )
            )
        await self._notify_agent_profile_dirty(kb_id, document_id)
        return document

    async def complete_pipeline_artifact_success(
        self,
        binding: PipelineArtifactBinding,
        *,
        document: DocumentRecord,
        sidecar_artifact: ArtifactRecord,
        blocks_artifact: ArtifactRecord,
        lease: ArtifactMaterializationLease,
        runtime_sidecar_dir: Path,
        runtime_blocks_path: Path,
        chunks_count: int | None,
        entity_count: int | None = None,
        relation_count: int | None = None,
    ) -> PipelineArtifactFinalizationResult:
        """Complete one processing-owner build from exact revalidated authority."""

        if binding.operation != "build":
            raise DocumentLifecycleError(
                "Pipeline build finalization requires a build binding"
            )
        if not binding.parser_hash or not binding.index_hash:
            raise DocumentLifecycleError(
                "Pipeline build finalization requires parser and index hashes"
            )
        runtime_sidecar_dir = runtime_sidecar_dir.resolve(strict=True)
        runtime_blocks_path = runtime_blocks_path.resolve(strict=True)
        if not runtime_sidecar_dir.is_dir():
            raise DocumentLifecycleError("Pipeline build sidecar is not a directory")
        if not runtime_blocks_path.is_file() or not runtime_blocks_path.is_relative_to(
            runtime_sidecar_dir
        ):
            raise DocumentLifecycleError(
                "Pipeline build blocks file is outside the sidecar directory"
            )

        canonical_sidecar = (
            self._document_service.canonical_document_root(document)
            / "__parsed__"
            / runtime_sidecar_dir.name
        ).resolve(strict=False)
        canonical_blocks = (canonical_sidecar / runtime_blocks_path.name).resolve(
            strict=False
        )
        plan = IndexBuildPlan(
            document=document,
            sidecar_artifact=_build_reference_from_record(sidecar_artifact),
            blocks_artifact=_build_reference_from_record(blocks_artifact),
            expected_current_sidecar_artifact_id=(
                binding.expected_current_sidecar_artifact_id
            ),
            expected_current_blocks_artifact_id=(
                binding.expected_current_blocks_artifact_id
            ),
            parser_hash=binding.parser_hash,
            index_hash=binding.index_hash,
            process_options=str(document.metadata.get("process_options") or ""),
            force_rechunk=False,
            force_extract=False,
            force_embedding=False,
            expected_status=document.status,
            expected_source_hash=binding.source_hash,
            expected_parser_hash=binding.parser_hash,
            expected_current_parse_generation_id=binding.parse_generation_id,
            expected_index_hash=document.index_hash,
            claim_token=binding.claim_token,
            kb_generation=binding.kb_generation,
            job_id=binding.job_id,
            object_preflight_complete=True,
        )
        execution = IndexBuildExecution(
            lease=lease,
            runtime_sidecar_dir=runtime_sidecar_dir,
            runtime_sidecar_uri=sidecar_uri_for(runtime_sidecar_dir),
            runtime_blocks_path=runtime_blocks_path,
            canonical_sidecar_locator=canonical_sidecar,
            canonical_blocks_locator=canonical_blocks,
            expected_current_sidecar_artifact_id=(
                binding.expected_current_sidecar_artifact_id
            ),
            expected_current_blocks_artifact_id=(
                binding.expected_current_blocks_artifact_id
            ),
            initial_sidecar_checksum=_required_build_checksum(sidecar_artifact),
            initial_blocks_checksum=_required_build_checksum(blocks_artifact),
            pipeline_started=True,
        )
        try:
            committed_document = await self.complete_build(
                binding.kb_id,
                binding.document_id,
                job_id=binding.job_id,
                plan=plan,
                run_result={
                    "chunks_count": chunks_count,
                    "entity_count": entity_count,
                    "relation_count": relation_count,
                },
                execution=execution,
                propagate_cancellation_after_commit=False,
            )
        except MetadataCommitOutcomeUnknownError as exc:
            return PipelineArtifactFinalizationResult(
                outcome=PipelineArtifactCommitOutcome.UNKNOWN,
                reason=exc.reason or "build_metadata_commit_unknown",
            )

        committed_binding = binding.committed(
            parse_generation_id=committed_document.metadata.get(
                "current_parse_generation_id"
            ),
            index_hash=committed_document.index_hash,
            sidecar_artifact_id=committed_document.metadata.get(
                "current_sidecar_artifact_id"
            ),
            blocks_artifact_id=committed_document.metadata.get(
                "current_blocks_artifact_id"
            ),
            raw_artifact_ids=(),
        )
        return PipelineArtifactFinalizationResult(
            outcome=PipelineArtifactCommitOutcome.COMMITTED,
            committed_binding=committed_binding,
            chunks_count=committed_document.chunks_count,
            entity_count=committed_document.entity_count,
            relation_count=committed_document.relation_count,
        )

    async def _read_build_promotion_commit_outcome(
        self,
        *,
        kb_id: str,
        document_id: str,
        artifacts: list[ArtifactRecord],
        job_id: str,
        claim_token: str,
        index_hash: str,
        current_sidecar_artifact_id: str,
        current_blocks_artifact_id: str | None,
    ) -> MetadataCommitReconciliation[DocumentRecord]:
        artifact_ids = [artifact.id for artifact in artifacts]
        (
            document,
            persisted_artifacts,
        ) = await self._document_service.metadata_store.get_document_and_artifacts_by_ids(
            kb_id,
            document_id,
            artifact_ids,
        )
        if document is None:
            return MetadataCommitReconciliation(
                outcome=MetadataCommitOutcome.UNKNOWN,
                reason="candidate_document_missing",
            )

        artifacts_complete = bool(
            len(persisted_artifacts) == len(artifacts)
            and set(persisted_artifacts) == set(artifact_ids)
            and all(
                _artifact_commit_candidate_matches(
                    artifact,
                    persisted_artifacts[artifact.id],
                )
                for artifact in artifacts
            )
        )
        metadata = document.metadata
        document_committed = bool(
            document.status == "ready"
            and document.index_hash == index_hash
            and metadata.get("current_build_generation_id") == claim_token
            and metadata.get("last_build_job_id") == job_id
            and metadata.get("current_sidecar_artifact_id")
            == current_sidecar_artifact_id
            and metadata.get("current_blocks_artifact_id") == current_blocks_artifact_id
            and metadata.get("pending_build_job_id") is None
            and metadata.get("pending_build_claim_token") is None
            and metadata.get("current_build_job_id") is None
            and metadata.get("current_build_claim_token") is None
        )
        if artifacts_complete and document_committed:
            return MetadataCommitReconciliation(
                outcome=MetadataCommitOutcome.COMMITTED,
                value=document,
                reason="candidate_document_and_artifacts_match",
            )

        candidate_artifact_ids = set(artifact_ids)
        candidate_pointer_visible = bool(
            metadata.get("current_build_generation_id") == claim_token
            or metadata.get("current_sidecar_artifact_id") in candidate_artifact_ids
            or metadata.get("current_blocks_artifact_id") in candidate_artifact_ids
        )
        if not persisted_artifacts and not candidate_pointer_visible:
            return MetadataCommitReconciliation(
                outcome=MetadataCommitOutcome.ROLLED_BACK,
                reason="candidate_rows_and_pointers_absent",
            )
        return MetadataCommitReconciliation(
            outcome=MetadataCommitOutcome.UNKNOWN,
            reason=(
                "candidate_artifacts_partial_or_mismatched"
                if persisted_artifacts
                else "candidate_document_pointer_mismatch"
            ),
        )

    async def _reconcile_build_promotion_commit_exception(
        self,
        *,
        kb_id: str,
        document_id: str,
        artifacts: list[ArtifactRecord],
        uploaded: list[UploadedArtifactObject],
        job_id: str,
        claim_token: str,
        index_hash: str,
        current_sidecar_artifact_id: str,
        current_blocks_artifact_id: str | None,
        commit_error: BaseException,
        propagate_cancellation: bool = True,
    ) -> DocumentRecord:
        async def read_back() -> MetadataCommitReconciliation[DocumentRecord]:
            return await self._read_build_promotion_commit_outcome(
                kb_id=kb_id,
                document_id=document_id,
                artifacts=artifacts,
                job_id=job_id,
                claim_token=claim_token,
                index_hash=index_hash,
                current_sidecar_artifact_id=current_sidecar_artifact_id,
                current_blocks_artifact_id=current_blocks_artifact_id,
            )

        caller_cancelled = isinstance(commit_error, asyncio.CancelledError)
        readback_error: BaseException | None = None
        try:
            safe_result = await await_cancellation_safe_reconciliation(read_back)
            reconciliation = safe_result.value
            caller_cancelled = caller_cancelled or safe_result.caller_cancelled
        except asyncio.CancelledError as exc:
            readback_error = exc
            reconciliation = MetadataCommitReconciliation[DocumentRecord](
                outcome=MetadataCommitOutcome.UNKNOWN,
                reason="readback_cancelled",
            )
        except Exception as exc:  # noqa: BLE001 - read failure means unknown
            readback_error = exc
            reconciliation = MetadataCommitReconciliation[DocumentRecord](
                outcome=MetadataCommitOutcome.UNKNOWN,
                reason="readback_failed",
            )

        if (
            reconciliation.outcome is MetadataCommitOutcome.COMMITTED
            and reconciliation.value is not None
        ):
            if caller_cancelled and propagate_cancellation:
                if isinstance(commit_error, asyncio.CancelledError):
                    raise commit_error
                raise asyncio.CancelledError() from commit_error
            return reconciliation.value

        if reconciliation.outcome is MetadataCommitOutcome.ROLLED_BACK:
            await self._document_service.compensate_uploaded_artifact_objects(uploaded)
            if caller_cancelled and propagate_cancellation:
                if isinstance(commit_error, asyncio.CancelledError):
                    raise commit_error
                raise asyncio.CancelledError() from commit_error
            raise commit_error

        artifact_ids = [artifact.id for artifact in artifacts]
        logger.warning(
            "metadata_commit_reconciliation outcome=unknown operation=%s "
            "document_id=%s candidate_artifact_ids=%s candidate_artifact_types=%s "
            "reason=%s commit_error_type=%s readback_error_type=%s",
            "build_artifact_promotion",
            document_id,
            artifact_ids,
            [artifact.artifact_type for artifact in artifacts],
            reconciliation.reason or "unknown",
            type(commit_error).__name__,
            type(readback_error).__name__ if readback_error is not None else None,
        )
        unknown_error = MetadataCommitOutcomeUnknownError(
            "build_artifact_promotion",
            candidate_document_ids=[document_id],
            candidate_job_id=job_id,
            candidate_artifact_ids=artifact_ids,
            candidate_artifact_types=[artifact.artifact_type for artifact in artifacts],
            reason=reconciliation.reason,
        )
        if caller_cancelled and propagate_cancellation:
            if isinstance(commit_error, asyncio.CancelledError):
                raise commit_error from unknown_error
            raise asyncio.CancelledError() from unknown_error
        raise unknown_error from (readback_error or commit_error)

    async def _promote_changed_artifacts(
        self,
        plan: IndexBuildPlan,
        execution: IndexBuildExecution,
        *,
        generation_id: str,
    ) -> tuple[list[ArtifactRecord], list[UploadedArtifactObject]]:
        current_sidecar_checksum = _directory_checksum(execution.runtime_sidecar_dir)
        current_blocks_checksum = _file_checksum(execution.runtime_blocks_path)
        sidecar_changed = current_sidecar_checksum != execution.initial_sidecar_checksum
        blocks_changed = current_blocks_checksum != execution.initial_blocks_checksum
        if not sidecar_changed and not blocks_changed:
            return [], []

        object_storage = self._document_service.object_storage
        if object_storage is None:
            raise DocumentLifecycleError("Object storage is not enabled")
        now = utc_now_iso()
        sidecar_artifact_id = _deterministic_build_artifact_id(
            plan.document,
            artifact_type="sidecar",
            generation_id=generation_id,
        )
        sidecar_artifact = ArtifactRecord(
            id=sidecar_artifact_id,
            kb_id=plan.document.kb_id,
            workspace=plan.document.workspace,
            document_id=plan.document.id,
            artifact_type="sidecar",
            uri=str(execution.canonical_sidecar_locator),
            checksum=current_sidecar_checksum,
            size_bytes=None,
            metadata={
                "is_directory": True,
                "filename": execution.runtime_sidecar_dir.name,
                "blocks_path": str(execution.canonical_blocks_locator),
                "parse_engine": plan.document.metadata.get("parse_engine"),
                "parser_hash": plan.parser_hash,
                "build_generation_id": generation_id,
            },
            created_at=now,
        )
        promoted: list[ArtifactRecord] = [sidecar_artifact]
        if blocks_changed:
            blocks_artifact_id = _deterministic_build_artifact_id(
                plan.document,
                artifact_type="blocks",
                generation_id=generation_id,
            )
            promoted.append(
                ArtifactRecord(
                    id=blocks_artifact_id,
                    kb_id=plan.document.kb_id,
                    workspace=plan.document.workspace,
                    document_id=plan.document.id,
                    artifact_type="blocks",
                    uri=str(execution.canonical_blocks_locator),
                    checksum=current_blocks_checksum,
                    size_bytes=execution.runtime_blocks_path.stat().st_size,
                    metadata={
                        "parse_engine": plan.document.metadata.get("parse_engine"),
                        "parser_hash": plan.parser_hash,
                        "build_generation_id": generation_id,
                        "filename": execution.runtime_blocks_path.name,
                    },
                    created_at=now,
                )
            )

        uploaded_objects: list[UploadedArtifactObject] = []
        try:
            (
                prefix_uri,
                _created_sidecar_objects,
            ) = await _upload_immutable_artifact_directory(
                object_storage,
                execution.runtime_sidecar_dir,
                prefix=_build_artifact_object_prefix(
                    plan.document,
                    sidecar_artifact,
                    execution.runtime_sidecar_dir,
                ),
                verification_root=execution.runtime_sidecar_dir.parent,
                uploaded_objects=uploaded_objects,
            )
            object_storage.validate_document_prefix_uri(
                prefix_uri,
                workspace=plan.document.workspace,
                document_id=plan.document.id,
                namespace="artifacts",
                artifact_id=sidecar_artifact.id,
            )
            sidecar_artifact.metadata["object_prefix_uri"] = prefix_uri
            blocks_artifact = next(
                (
                    artifact
                    for artifact in promoted
                    if artifact.artifact_type == "blocks"
                ),
                None,
            )
            if blocks_artifact is not None:
                object_uri, _created = await _upload_immutable_artifact_file(
                    object_storage,
                    execution.runtime_blocks_path,
                    key=_build_artifact_object_key(
                        plan.document,
                        blocks_artifact,
                        execution.runtime_blocks_path,
                    ),
                    verification_root=execution.runtime_sidecar_dir.parent,
                    uploaded_objects=uploaded_objects,
                )
                object_storage.validate_document_file_uri(
                    object_uri,
                    workspace=plan.document.workspace,
                    document_id=plan.document.id,
                    namespace="artifacts",
                    artifact_id=blocks_artifact.id,
                )
                blocks_artifact.metadata["object_uri"] = object_uri
            return promoted, uploaded_objects
        except BaseException:
            await self._document_service.compensate_uploaded_artifact_objects(
                uploaded_objects
            )
            raise

    async def fail_build(
        self,
        kb_id: str,
        document_id: str,
        *,
        job_id: str,
        error_code: str,
        error_message: str,
        plan: IndexBuildPlan | None = None,
        claim_token: str | None = None,
    ) -> DocumentRecord:
        token = claim_token or (plan.claim_token if plan is not None else None)
        phase: Literal["pending", "current"] | None = None
        if token is None:
            current_document = await self._document_service.get_document(
                kb_id, document_id
            )
            token, phase = _resolve_service_attempt_owner(
                current_document,
                operation="build",
                job_id=job_id,
                claim_token=None,
                strict=True,
            )
            if plan is not None and token is not None:
                plan.claim_token = token
        if self._document_service.object_authoritative and token is None:
            raise DocumentLifecycleError(
                "Object build failure requires an explicit claim token"
            )
        metadata_patch = {
            "last_failed_build_job_id": job_id,
            "pending_build_job_id": None,
            "current_build_job_id": None,
        }
        if phase == "pending":
            return await self._document_service.metadata_store.release_document_build_if_owned(
                kb_id,
                document_id,
                job_id=job_id,
                claim_token=token,
                error_code=error_code,
                error_message=error_message,
                metadata_patch=metadata_patch,
            )
        return await self._document_service.metadata_store.fail_document_build(
            kb_id,
            document_id,
            error_code=error_code,
            error_message=error_message,
            job_id=job_id if token is not None or phase == "current" else None,
            claim_token=token,
            metadata_patch=metadata_patch,
        )

    async def release_build_if_owned(
        self,
        kb_id: str,
        document_id: str,
        *,
        job_id: str,
        plan: IndexBuildPlan,
        error_code: str,
        error_message: str,
    ) -> DocumentRecord:
        token = plan.claim_token
        if token is None:
            current_document = await self._document_service.get_document(
                kb_id, document_id
            )
            token, _phase = _resolve_service_attempt_owner(
                current_document,
                operation="build",
                job_id=job_id,
                claim_token=None,
                strict=False,
            )
            if token is not None:
                plan.claim_token = token
        if self._document_service.object_authoritative and token is None:
            raise DocumentLifecycleError(
                "Object build release requires an explicit claim token"
            )
        return (
            await self._document_service.metadata_store.release_document_build_if_owned(
                kb_id,
                document_id,
                job_id=job_id,
                claim_token=token,
                error_code=error_code,
                error_message=error_message,
                metadata_patch={
                    "last_failed_build_job_id": job_id,
                },
            )
        )

    async def _resolve_artifacts(
        self, kb_id: str, document: DocumentRecord
    ) -> tuple[
        BuildArtifactReference | None,
        BuildArtifactReference | None,
        str | None,
        str | None,
    ]:
        sidecar, expected_sidecar_id = await self._resolve_current_artifact(
            kb_id,
            document,
            artifact_type="sidecar",
            pointer_key="current_sidecar_artifact_id",
        )
        blocks, expected_blocks_id = await self._resolve_current_artifact(
            kb_id,
            document,
            artifact_type="blocks",
            pointer_key="current_blocks_artifact_id",
        )
        if self._document_service.object_authoritative and sidecar is not None:
            sidecar_name = _artifact_locator_name(
                sidecar.compatibility_locator,
                fallback=f"{document.source_name}.parsed",
            )
            canonical_sidecar = (
                self._document_service.canonical_document_root(document)
                / "__parsed__"
                / sidecar_name
            ).resolve(strict=False)
            blocks_name = _artifact_locator_name(
                blocks.compatibility_locator if blocks is not None else None,
                fallback=_artifact_locator_name(
                    sidecar.blocks_locator,
                    fallback=f"{Path(document.source_name).stem}.blocks.jsonl",
                ),
            )
            sidecar.compatibility_locator = str(canonical_sidecar)
            sidecar.blocks_locator = str(canonical_sidecar / blocks_name)
            if blocks is not None:
                blocks.compatibility_locator = str(canonical_sidecar / blocks_name)
        return sidecar, blocks, expected_sidecar_id, expected_blocks_id

    async def _resolve_current_artifact(
        self,
        kb_id: str,
        document: DocumentRecord,
        *,
        artifact_type: str,
        pointer_key: str,
    ) -> tuple[BuildArtifactReference | None, str | None]:
        raw_pointer = document.metadata.get(pointer_key)
        expected_pointer = (
            raw_pointer if isinstance(raw_pointer, str) and raw_pointer else None
        )
        artifact: ArtifactRecord | None = None
        if expected_pointer is not None:
            artifact = await self._document_service.get_document_artifact(
                kb_id, document.id, expected_pointer
            )
            if (
                artifact.document_id != document.id
                or artifact.artifact_type != artifact_type
            ):
                raise DocumentLifecycleError(
                    f"Document '{document.id}' current {artifact_type} pointer is invalid"
                )
        else:
            artifacts, _total = await self._document_service.list_document_artifacts(
                kb_id,
                document.id,
                artifact_type=artifact_type,
                limit=1,
                offset=0,
            )
            artifact = artifacts[0] if artifacts else None
        if artifact is None:
            return None, expected_pointer
        object_uri = artifact.metadata.get("object_uri")
        object_prefix_uri = artifact.metadata.get("object_prefix_uri")
        object_uri = object_uri if isinstance(object_uri, str) and object_uri else None
        object_prefix_uri = (
            object_prefix_uri
            if isinstance(object_prefix_uri, str) and object_prefix_uri
            else None
        )
        object_storage = self._document_service.object_storage
        if self._document_service.object_authoritative:
            if object_storage is None:
                raise DocumentLifecycleError("Object storage is not enabled")
            if object_uri is not None:
                object_storage.validate_document_file_uri(
                    object_uri,
                    workspace=document.workspace,
                    document_id=document.id,
                    namespace="artifacts",
                    artifact_id=artifact.id,
                )
            if object_prefix_uri is not None:
                object_storage.validate_document_prefix_uri(
                    object_prefix_uri,
                    workspace=document.workspace,
                    document_id=document.id,
                    namespace="artifacts",
                    artifact_id=artifact.id,
                )
        blocks_locator = artifact.metadata.get("blocks_path")
        return (
            BuildArtifactReference(
                id=artifact.id,
                artifact_type=artifact.artifact_type,
                checksum=artifact.checksum,
                size_bytes=artifact.size_bytes,
                object_uri=object_uri,
                object_prefix_uri=object_prefix_uri,
                compatibility_locator=artifact.uri or None,
                blocks_locator=(
                    blocks_locator
                    if isinstance(blocks_locator, str) and blocks_locator
                    else None
                ),
            ),
            expected_pointer,
        )

    async def _notify_agent_profile_dirty(self, kb_id: str, document_id: str) -> None:
        if self._agent_profile_dirty_callback is None:
            return
        try:
            await self._agent_profile_dirty_callback(kb_id, document_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Agent profile dirty callback failed for KB '%s' doc '%s': %s",
                kb_id,
                document_id,
                exc,
            )


def _artifact_locator_name(locator: str | None, *, fallback: str) -> str:
    candidate = locator or fallback
    parsed = urlsplit(candidate).path if "://" in candidate else candidate
    name = Path(parsed.rstrip("/")).name or Path(fallback).name
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or any(ord(char) < 0x20 or ord(char) == 0x7F for char in name)
    ):
        raise DocumentLifecycleError("Artifact locator has an unsafe basename")
    return name


def _build_reference_from_record(artifact: ArtifactRecord) -> BuildArtifactReference:
    object_uri = artifact.metadata.get("object_uri")
    object_prefix_uri = artifact.metadata.get("object_prefix_uri")
    blocks_locator = artifact.metadata.get("blocks_path")
    return BuildArtifactReference(
        id=artifact.id,
        artifact_type=artifact.artifact_type,
        checksum=artifact.checksum,
        size_bytes=artifact.size_bytes,
        object_uri=object_uri if isinstance(object_uri, str) and object_uri else None,
        object_prefix_uri=(
            object_prefix_uri
            if isinstance(object_prefix_uri, str) and object_prefix_uri
            else None
        ),
        compatibility_locator=artifact.uri or None,
        blocks_locator=(
            blocks_locator
            if isinstance(blocks_locator, str) and blocks_locator
            else None
        ),
    )


def _required_build_checksum(artifact: ArtifactRecord) -> str:
    checksum = artifact.checksum
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        raise DocumentLifecycleError(
            f"Build {artifact.artifact_type} artifact has no verifiable checksum"
        )
    return checksum


def _local_sidecar_path(
    source_root: Path,
    document: DocumentRecord,
    sidecar: BuildArtifactReference,
) -> Path:
    locator = sidecar.compatibility_locator
    if not locator:
        raise FileNotFoundError(
            f"Document '{document.id}' has no local sidecar locator"
        )
    resolved = resolve_sidecar_uri(locator)
    path = resolved if resolved is not None else Path(locator)
    if not path.is_absolute():
        path = source_root / path
    path = path.resolve(strict=False)
    if path.is_file() and path.name.endswith(".blocks.jsonl"):
        path = path.parent
    allowed_root = source_root.expanduser().resolve(strict=False)
    if not path.is_relative_to(allowed_root):
        raise ValueError("LightRAG document sidecar path must stay under INPUT_DIR")
    if not path.is_dir():
        raise FileNotFoundError(f"Build sidecar directory not found: {path}")
    return path


def _local_blocks_path(
    sidecar_path: Path,
    blocks: BuildArtifactReference | None,
    sidecar: BuildArtifactReference,
) -> Path:
    locator = (
        blocks.compatibility_locator
        if blocks is not None and blocks.compatibility_locator
        else sidecar.blocks_locator
    )
    candidate: Path | None = None
    if locator:
        resolved = resolve_sidecar_uri(locator)
        candidate = resolved if resolved is not None else Path(locator)
        if not candidate.is_absolute():
            candidate = sidecar_path / candidate.name
    if candidate is None:
        inferred = sidecar_blocks_path(sidecar_uri_for(sidecar_path))
        candidate = Path(inferred) if inferred else None
    if candidate is None or not candidate.is_file():
        matches = sorted(sidecar_path.glob("*.blocks.jsonl"))
        candidate = matches[0] if len(matches) == 1 else candidate
    if candidate is None:
        raise FileNotFoundError("Build sidecar has no blocks file")
    candidate = candidate.resolve(strict=False)
    if not candidate.is_relative_to(sidecar_path.resolve(strict=False)):
        raise ValueError("Build blocks path escapes the sidecar directory")
    if not candidate.is_file():
        raise FileNotFoundError(f"Build blocks file not found: {candidate}")
    return candidate


def _build_artifact_object_key(
    document: DocumentRecord, artifact: ArtifactRecord, path: Path
) -> str:
    return "/".join(
        [
            "workspaces",
            document.workspace,
            "documents",
            document.id,
            "artifacts",
            artifact.artifact_type,
            artifact.id,
            path.name,
        ]
    )


def _build_artifact_object_prefix(
    document: DocumentRecord, artifact: ArtifactRecord, path: Path
) -> str:
    return _build_artifact_object_key(document, artifact, path).rstrip("/") + "/"


def _deterministic_build_artifact_id(
    document: DocumentRecord,
    *,
    artifact_type: Literal["sidecar", "blocks"],
    generation_id: str,
) -> str:
    """Derive one immutable generation identity from stable attempt authority."""

    payload = "\0".join(
        (
            "pipeline-build-artifact-v1",
            document.kb_id,
            document.workspace,
            document.id,
            document.lightrag_doc_id or "",
            artifact_type,
            generation_id,
        )
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return f"artifact_{artifact_type}_{digest}"


async def _upload_immutable_artifact_directory(
    object_storage: Any,
    local_dir: Path,
    *,
    prefix: str,
    verification_root: Path,
    uploaded_objects: list[UploadedArtifactObject],
) -> tuple[str, list[UploadedArtifactObject]]:
    """Upload members and immediately collect every newly created object."""

    root = local_dir.resolve(strict=True)
    created_start = len(uploaded_objects)
    regular_files = [
        path
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file() and not path.is_symlink()
    ]
    if not regular_files:
        raise DocumentLifecycleError("Build sidecar directory contains no files")
    for path in regular_files:
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise DocumentLifecycleError("Build sidecar entry escapes its directory")
        relative = resolved.relative_to(root).as_posix()
        await _upload_immutable_artifact_file(
            object_storage,
            resolved,
            key=f"{prefix.rstrip('/')}/{relative}",
            verification_root=verification_root,
            uploaded_objects=uploaded_objects,
        )
    return (
        object_storage.object_prefix_uri_for_key(prefix),
        uploaded_objects[created_start:],
    )


async def _upload_immutable_artifact_file(
    object_storage: Any,
    local_path: Path,
    *,
    key: str,
    verification_root: Path,
    uploaded_objects: list[UploadedArtifactObject],
) -> tuple[str, bool]:
    """Conditionally create a key and immediately record invocation ownership."""

    expected_checksum = _file_checksum(local_path)
    object_uri, created = await object_storage.upload_file_if_absent(
        local_path,
        key=key,
    )
    if created:
        uploaded_objects.append(UploadedArtifactObject(uri=object_uri, is_prefix=False))
    expected_uri = object_storage.object_uri_for_key(key)
    if object_uri != expected_uri:
        raise DocumentLifecycleError(
            "Object storage returned an unexpected immutable key"
        )
    if created:
        return object_uri, True

    verification_dir = (
        verification_root.resolve(strict=True) / ".lightrag-object-verification"
    )
    verification_dir.mkdir(mode=0o700, exist_ok=True)
    verification_path = (
        verification_dir / hashlib.sha256(object_uri.encode("utf-8")).hexdigest()
    )
    try:
        await object_storage.download_file(object_uri, verification_path)
        if _file_checksum(verification_path) != expected_checksum:
            raise ImmutableArtifactObjectConflictError(
                "Deterministic artifact object key already contains different bytes"
            )
    finally:
        verification_path.unlink(missing_ok=True)
        try:
            verification_dir.rmdir()
        except OSError:
            pass
    return object_uri, False


def compute_index_hash(rag: Any) -> str:
    """Build a hash that captures chunk/embedding/extraction config.

    Anything that, when changed, would invalidate previously-built chunks,
    vectors, or KG content for a document. Query-time-only knobs (top_k etc.)
    are intentionally excluded.
    """
    active_index_hash = getattr(rag, "kb_active_index_hash", None)
    if active_index_hash:
        return str(active_index_hash)

    addon = getattr(rag, "addon_params", {}) or {}
    payload = {
        "schema": "kb-index-hash-v1",
        "embedding_func": getattr(
            getattr(rag, "embedding_func", None), "func_name", None
        )
        or getattr(getattr(rag, "embedding_func", None), "__name__", None),
        "embedding_dim": getattr(rag, "embedding_dim", None),
        "chunk_token_size": getattr(rag, "chunk_token_size", None),
        "chunk_overlap_token_size": getattr(rag, "chunk_overlap_token_size", None),
        "tiktoken_model_name": getattr(rag, "tiktoken_model_name", None),
        "summary_max_tokens": getattr(rag, "summary_max_tokens", None),
        "force_llm_summary_on_merge": getattr(rag, "force_llm_summary_on_merge", None),
        "addon_chunker": addon.get("chunker"),
        "addon_entity_types": addon.get("entity_types"),
        "addon_language": addon.get("language"),
        "addon_extraction": addon.get("extraction"),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, default=str
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _claimed_build_artifact_binding(
    plan: IndexBuildPlan,
) -> PipelineArtifactBinding:
    if not plan.kb_generation:
        raise DocumentLifecycleError("Build artifact binding requires a KB generation")
    if not plan.job_id:
        raise DocumentLifecycleError(
            "Build artifact binding requires a claimed job identity"
        )
    if not plan.claim_token:
        raise DocumentLifecycleError(
            "Build artifact binding requires a claimed attempt token"
        )
    if not plan.document.lightrag_doc_id:
        raise DocumentLifecycleError("Document has no LightRAG document identity")
    if not plan.document.source_hash:
        raise DocumentLifecycleError(
            "Build artifact binding requires a source snapshot"
        )
    if not plan.expected_current_parse_generation_id:
        raise DocumentLifecycleError(
            "Build artifact binding requires a parse generation snapshot"
        )
    if plan.sidecar_artifact is None or plan.blocks_artifact is None:
        raise DocumentLifecycleError(
            "Build artifact binding requires exact sidecar and blocks artifacts"
        )
    if (
        plan.sidecar_artifact.id != plan.expected_current_sidecar_artifact_id
        or plan.blocks_artifact.id != plan.expected_current_blocks_artifact_id
    ):
        raise DocumentLifecycleError(
            "Build artifact binding does not match current artifact pointers"
        )
    return PipelineArtifactBinding(
        version=1,
        authority="kb_metadata",
        state="claimed",
        operation="build",
        kb_id=plan.document.kb_id,
        kb_generation=plan.kb_generation,
        workspace=plan.document.workspace,
        document_id=plan.document.id,
        lightrag_doc_id=plan.document.lightrag_doc_id,
        job_id=plan.job_id,
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


def _validate_committed_build_artifact_binding(
    binding: PipelineArtifactBinding,
    plan: IndexBuildPlan,
) -> None:
    expected = {
        "state": "committed",
        "operation": "build",
        "kb_id": plan.document.kb_id,
        "kb_generation": plan.kb_generation,
        "workspace": plan.document.workspace,
        "document_id": plan.document.id,
        "lightrag_doc_id": plan.document.lightrag_doc_id,
        "job_id": plan.job_id,
        "claim_token": plan.claim_token,
        "source_hash": plan.document.source_hash,
        "parser_hash": plan.parser_hash,
        "parse_generation_id": plan.expected_current_parse_generation_id,
        "index_hash": plan.index_hash,
        "expected_current_sidecar_artifact_id": (
            plan.expected_current_sidecar_artifact_id
        ),
        "expected_current_blocks_artifact_id": (
            plan.expected_current_blocks_artifact_id
        ),
    }
    mismatches = [
        field_name
        for field_name, expected_value in expected.items()
        if getattr(binding, field_name) != expected_value
    ]
    if binding.sidecar_artifact_id is None:
        mismatches.append("sidecar_artifact_id")
    if binding.blocks_artifact_id is None:
        mismatches.append("blocks_artifact_id")
    if binding.raw_artifact_ids:
        mismatches.append("raw_artifact_ids")
    if mismatches:
        raise DocumentLifecycleError(
            "Committed build artifact binding mismatch: "
            + ", ".join(sorted(set(mismatches)))
        )


def _build_needs_engine_clear(plan: IndexBuildPlan) -> bool:
    """Whether the old LightRAG row must be deleted before a (re-)enqueue.

    LightRAG's enqueue de-duplicates a document whose id is already present
    (``full_docs`` / ``doc_status`` ``filter_keys``). A KB document keeps the
    SAME ``lightrag_doc_id`` across rebuilds, so re-enqueuing an already-built
    doc without deleting it first is a silent no-op that re-reports the stale
    doc_status counts as success. We clear the old row whenever the document
    already has a live engine row:

    * ``plan.force`` — an explicit ``:reindex`` / ``force_*`` on ``:build-kg``
      (unchanged behavior; a never-built id resolves to a cheap ``not_found``).
    * ``index_hash is not None`` — the document was successfully indexed before
      and has not since been reset. ``complete_document_build`` stamps
      ``index_hash``; a ``replace`` clears it (and deletes the engine row
      itself), so a post-replace first build sees ``None`` and skips the
      delete; a ``reparse`` preserves it, so a post-reparse rebuild correctly
      clears the stale row. This is what makes a NON-forced ``:build-kg`` of an
      already-built document actually rebuild instead of re-reporting stale
      counts.

    A genuine first build (``index_hash is None``, not forced) has no engine
    row to clear and is left untouched.
    """
    if not plan.document.lightrag_doc_id:
        return False
    return plan.force or plan.document.index_hash is not None


def _kb_unique_basename(plan: IndexBuildPlan) -> str:
    """Build a basename that is globally unique inside the KB workspace.

    LightRAG's filename-based dedup keys off the basename of the supplied
    ``file_path``. KB-layer source names can collide (e.g. two CJK PDFs
    that both sanitise to ``_.pdf``); prefixing with the KB document id
    keeps each entry distinct without losing the original suffix used for
    filetype detection downstream.
    """
    raw_name = (plan.document.source_name or "").strip() or "document"
    safe_name = raw_name.replace("/", "_").replace("\\", "_")
    return f"{plan.document.id}__{safe_name}"


async def _await_doc_status_terminal(
    rag: Any,
    lightrag_doc_id: str,
    *,
    timeout: float,
    poll_interval: float,
    wait_until_terminal: bool = False,
) -> dict[str, Any] | None:
    """Poll ``rag.doc_status`` for ``lightrag_doc_id`` until it leaves the
    inflight set (pending/parsing/analyzing/processing/preprocessed).

    Local mode returns the last observed state when its timeout expires. Object
    mode uses the timeout only as an observability-warning threshold and keeps
    polling: the current attempt owns both its metadata claim and scratch lease
    until the actual pipeline producer reaches a terminal status.

    The wait is what makes the read-back safe under concurrent same-KB drains:
    ``apipeline_process_enqueue_documents`` returns immediately (setting
    ``request_pending``) when another flow holds the pipeline ``busy`` flag, so
    our freshly-enqueued docs may still be inflight on return; the owning loop
    will drive them to a terminal state shortly.
    """
    doc_status_storage = getattr(rag, "doc_status", None)
    if doc_status_storage is None:
        return None
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0.0, timeout)
    consecutive_missing = 0
    warning_emitted = False
    while True:
        try:
            rows = await doc_status_storage.get_by_ids([lightrag_doc_id])
        except Exception as exc:  # noqa: BLE001
            if not wait_until_terminal:
                logger.warning(
                    "Failed to read doc_status while awaiting build of '%s': %s",
                    lightrag_doc_id,
                    exc,
                )
                return None
            if not warning_emitted:
                logger.warning(
                    "Object build '%s' remains owner-inflight after doc_status "
                    "probe failure; continuing until producer terminal: %s",
                    lightrag_doc_id,
                    exc,
                )
                warning_emitted = True
            await asyncio.sleep(max(0.01, poll_interval))
            continue
        row = rows[0] if rows else None
        if row is None:
            consecutive_missing += 1
            if not wait_until_terminal and (
                consecutive_missing >= 2 or loop.time() >= deadline
            ):
                return None
            if wait_until_terminal and loop.time() >= deadline and not warning_emitted:
                logger.warning(
                    "Object build '%s' has no doc_status row after %.3fs; "
                    "retaining attempt owner and continuing to poll",
                    lightrag_doc_id,
                    max(0.0, timeout),
                )
                warning_emitted = True
            await asyncio.sleep(max(0.01, poll_interval))
            continue
        consecutive_missing = 0
        status = row.get("status")
        if status not in _INFLIGHT_BUILD_STATUSES:
            # processed / failed → terminal enough to classify.
            return row
        if loop.time() >= deadline:
            if not wait_until_terminal:
                return row
            if not warning_emitted:
                logger.warning(
                    "Object build '%s' remains %s after %.3fs; retaining "
                    "attempt owner and lease until producer terminal",
                    lightrag_doc_id,
                    status,
                    max(0.0, timeout),
                )
                warning_emitted = True
        await asyncio.sleep(max(0.01, poll_interval))


async def _collect_doc_status(
    rag: Any,
    plan: IndexBuildPlan,
    *,
    timeout: float = DEFAULT_BUILD_DRAIN_TIMEOUT_SECONDS,
    poll_interval: float = DEFAULT_BUILD_DRAIN_POLL_SECONDS,
) -> dict[str, Any]:
    doc_status_storage = getattr(rag, "doc_status", None)
    if doc_status_storage is None:
        return {
            "skipped": False,
            "chunks_count": None,
            "entity_count": None,
            "relation_count": None,
        }
    if not plan.document.lightrag_doc_id:
        raise DocumentLifecycleError("Document has no LightRAG document identity")
    # Wait out a concurrent drain rather than reading once and mis-reporting a
    # still-inflight doc as a failure (see _await_doc_status_terminal).
    row = await _await_doc_status_terminal(
        rag, plan.document.lightrag_doc_id, timeout=timeout, poll_interval=poll_interval
    )
    if row is None:
        raise RuntimeError(
            f"Document '{plan.document.id}' build did not create doc_status row "
            f"for LightRAG doc '{plan.document.lightrag_doc_id}'"
        )
    if row.get("status") != "processed":
        raise RuntimeError(
            f"Document '{plan.document.id}' build did not reach processed (status={row.get('status')}: {row.get('error_msg')})"
        )
    return {
        "skipped": False,
        "chunks_count": row.get("chunks_count"),
        "entity_count": row.get("entity_count"),
        "relation_count": row.get("relation_count"),
    }


def _to_sidecar_uri(directory: str) -> str:
    return sidecar_uri_for(directory)


def _build_failure_item(
    document_id: str,
    *,
    error_code: str,
    error_message: str,
    **extra: Any,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "document_id": document_id,
        "status": "failed",
        "error_code": error_code,
        "error_message": error_message,
    }
    item.update({key: value for key, value in extra.items() if value is not None})
    return item
