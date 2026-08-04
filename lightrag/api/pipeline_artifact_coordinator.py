"""Processing-owner materialization for durable pipeline artifact bindings.

The coordinator is process-scoped.  Its callbacks capture only immutable KB
identity, while every invocation re-reads catalog and metadata authority before
allocating or downloading into this process's own materialization root.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Awaitable, Callable, Mapping

from lightrag.artifact_runtime import (
    PipelineArtifactCommitOutcome,
    PipelineArtifactBinding,
    PipelineArtifactFinalizationResult,
    PipelineArtifactMaterializer,
    PipelineArtifactRuntimeError,
    PipelineArtifactSession,
    PipelineTerminalOutcome,
    canonicalize_pipeline_logical_filename,
)
from lightrag.api.artifact_materialization import (
    ArtifactMaterializationLease,
    ArtifactMaterializer,
)
from lightrag.api.document_lifecycle_service import (
    DocumentLifecycleService,
    directory_artifact_checksum,
    file_artifact_checksum,
    normalize_artifact_checksum,
)
from lightrag.api.index_build_service import IndexBuildService
from lightrag.api.kb_service import (
    KnowledgeBaseConflictError,
    KnowledgeBaseNotFoundError,
    KnowledgeBaseRecord,
)
from lightrag.api.metadata_store import (
    ArtifactPointerConflictError,
    ArtifactRecord,
    DocumentAttemptOwnershipError,
    DocumentRecord,
    DocumentSnapshotConflictError,
    MetadataRecordNotFoundError,
)


_URI_WITH_AUTHORITY = re.compile(
    r"[A-Za-z][A-Za-z0-9+.-]*://[^\s'\"<>]+",
    re.IGNORECASE,
)

_SuccessFinalizer = Callable[
    [Mapping[str, Any] | None, int | None],
    Awaitable[PipelineArtifactFinalizationResult],
]


class PipelineArtifactCoordinatorError(PipelineArtifactRuntimeError):
    """Base durable-safe coordinator failure."""

    error_code = "pipeline_artifact_coordinator_error"

    def __init__(self, reason: str) -> None:
        super().__init__(f"{self.error_code}: {reason}")


class PipelineArtifactBindingStaleError(PipelineArtifactCoordinatorError):
    """The durable binding no longer matches current metadata authority."""

    error_code = "artifact_binding_stale"


class ArtifactMigrationRequiredError(PipelineArtifactCoordinatorError):
    """An object-authoritative binding references a legacy local-only row."""

    error_code = "artifact_migration_required"


class ArtifactChecksumMismatchError(PipelineArtifactCoordinatorError):
    """Downloaded bytes do not match the exact metadata artifact row."""

    error_code = "artifact_checksum_mismatch"


@dataclass(frozen=True, slots=True)
class _ValidatedAuthority:
    binding: PipelineArtifactBinding
    document: DocumentRecord
    sidecar: ArtifactRecord | None
    blocks: ArtifactRecord | None
    raw_artifacts: tuple[ArtifactRecord, ...]
    source_object_uri: str | None
    sidecar_object_prefix_uri: str | None
    blocks_object_uri: str | None
    raw_object_prefix_uris: tuple[str, ...]
    blocks_filename: str | None
    raw_directory_names: tuple[str, ...]

    @property
    def sensitive_object_references(self) -> tuple[str, ...]:
        return tuple(
            value
            for value in (
                self.source_object_uri,
                self.sidecar_object_prefix_uri,
                self.blocks_object_uri,
                *self.raw_object_prefix_uris,
            )
            if value
        )


class PipelineArtifactCoordinator:
    """Open exact processing-owner sessions from shared KB/object authority."""

    def __init__(
        self,
        kb_service: Any,
        document_service: DocumentLifecycleService,
        index_service: IndexBuildService,
    ) -> None:
        if not document_service.object_authoritative:
            raise PipelineArtifactCoordinatorError(
                "processing-owner coordinator requires object artifact mode"
            )
        materializer = document_service.materializer
        object_storage = document_service.object_storage
        if not isinstance(materializer, ArtifactMaterializer) or object_storage is None:
            raise PipelineArtifactCoordinatorError(
                "processing-owner coordinator requires object storage and materializer"
            )
        self._kb_service = kb_service
        self._document_service = document_service
        self._index_service = index_service
        self._metadata_store = document_service.metadata_store
        self._object_storage = object_storage
        self._materializer = materializer

    @property
    def materializer(self) -> ArtifactMaterializer:
        return self._materializer

    def materializer_for(
        self, record: KnowledgeBaseRecord
    ) -> PipelineArtifactMaterializer:
        """Return a callback capturing only immutable expected KB identity."""

        expected_kb_id = str(record.id)
        expected_generation = str(record.generation)
        expected_workspace = str(record.workspace)

        async def materialize(
            binding: PipelineArtifactBinding,
            *,
            _kb_id: str = expected_kb_id,
            _generation: str = expected_generation,
            _workspace: str = expected_workspace,
        ) -> PipelineArtifactSession:
            return await self.open(
                binding,
                expected_kb_id=_kb_id,
                expected_generation=_generation,
                expected_workspace=_workspace,
            )

        return materialize

    async def open(
        self,
        binding: PipelineArtifactBinding,
        *,
        expected_kb_id: str | None = None,
        expected_generation: str | None = None,
        expected_workspace: str | None = None,
    ) -> PipelineArtifactSession:
        """Re-read authority, validate the complete binding, then download."""

        if not isinstance(binding, PipelineArtifactBinding):
            raise PipelineArtifactBindingStaleError("binding type mismatch")
        binding.__post_init__()
        if binding.state != "claimed":
            raise PipelineArtifactBindingStaleError("binding is not claimed")
        self._validate_callback_identity(
            binding,
            expected_kb_id=expected_kb_id,
            expected_generation=expected_generation,
            expected_workspace=expected_workspace,
        )
        authority = await self._read_and_validate_authority(binding)
        return await self._materialize_authority(authority)

    @staticmethod
    def _validate_callback_identity(
        binding: PipelineArtifactBinding,
        *,
        expected_kb_id: str | None,
        expected_generation: str | None,
        expected_workspace: str | None,
    ) -> None:
        expected = {
            "kb_id": expected_kb_id,
            "kb_generation": expected_generation,
            "workspace": expected_workspace,
        }
        mismatches = [
            field_name
            for field_name, value in expected.items()
            if value is not None and getattr(binding, field_name) != value
        ]
        if mismatches:
            raise PipelineArtifactBindingStaleError(
                "callback identity mismatch: " + ", ".join(sorted(mismatches))
            )

    async def _read_and_validate_authority(
        self, binding: PipelineArtifactBinding
    ) -> _ValidatedAuthority:
        try:
            record = await self._kb_service.get(binding.kb_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise PipelineArtifactBindingStaleError(
                "knowledge-base authority is unavailable"
            ) from exc
        mismatches = []
        if record.id != binding.kb_id:
            mismatches.append("kb_id")
        if record.generation != binding.kb_generation:
            mismatches.append("kb_generation")
        if record.workspace != binding.workspace:
            mismatches.append("workspace")
        if mismatches:
            raise PipelineArtifactBindingStaleError(
                "knowledge-base identity mismatch: " + ", ".join(sorted(mismatches))
            )

        artifact_ids = self._binding_artifact_ids(binding)
        try:
            (
                document,
                artifacts,
            ) = await self._metadata_store.get_document_and_artifacts_by_ids(
                binding.kb_id,
                binding.document_id,
                artifact_ids,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise PipelineArtifactCoordinatorError(
                "metadata authority read failed"
            ) from exc
        if document is None:
            raise PipelineArtifactBindingStaleError("document authority is missing")
        missing_ids = [
            artifact_id for artifact_id in artifact_ids if artifact_id not in artifacts
        ]
        if missing_ids:
            raise ArtifactMigrationRequiredError(
                "one or more exact artifact rows are missing"
            )

        self._validate_document_snapshot(binding, document)
        sidecar = (
            artifacts[binding.sidecar_artifact_id]
            if binding.sidecar_artifact_id is not None
            else None
        )
        blocks = (
            artifacts[binding.blocks_artifact_id]
            if binding.blocks_artifact_id is not None
            else None
        )
        raw_artifacts = tuple(artifacts[item] for item in binding.raw_artifact_ids)
        self._validate_artifact_rows(
            binding,
            document,
            sidecar=sidecar,
            blocks=blocks,
            raw_artifacts=raw_artifacts,
        )

        if binding.operation == "build":
            return self._validate_build_object_references(
                binding,
                document,
                sidecar=sidecar,
                blocks=blocks,
                raw_artifacts=raw_artifacts,
            )
        return self._validate_parse_object_references(
            binding,
            document,
            sidecar=sidecar,
            blocks=blocks,
            raw_artifacts=raw_artifacts,
        )

    @staticmethod
    def _binding_artifact_ids(binding: PipelineArtifactBinding) -> list[str]:
        values = [
            value
            for value in (
                binding.sidecar_artifact_id,
                binding.blocks_artifact_id,
                *binding.raw_artifact_ids,
            )
            if value is not None
        ]
        if len(values) != len(set(values)):
            raise PipelineArtifactBindingStaleError(
                "binding artifact identities overlap"
            )
        return values

    @staticmethod
    def _metadata_identity(metadata: dict[str, Any], key: str) -> str | None:
        value = metadata.get(key)
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise PipelineArtifactBindingStaleError(
                f"document metadata field {key!r} is invalid"
            )
        return value

    def _validate_document_snapshot(
        self,
        binding: PipelineArtifactBinding,
        document: DocumentRecord,
    ) -> None:
        mismatches: list[str] = []
        for field_name, actual, expected in (
            ("kb_id", document.kb_id, binding.kb_id),
            ("workspace", document.workspace, binding.workspace),
            ("document_id", document.id, binding.document_id),
            ("source_hash", document.source_hash or None, binding.source_hash),
        ):
            if actual != expected:
                mismatches.append(field_name)

        metadata = document.metadata
        operation = binding.operation
        if document.status != ("building" if operation == "build" else "parsing"):
            mismatches.append("status")
        if (
            self._metadata_identity(metadata, f"current_{operation}_job_id")
            != binding.job_id
        ):
            mismatches.append("job_id")
        if (
            self._metadata_identity(metadata, f"current_{operation}_claim_token")
            != binding.claim_token
        ):
            mismatches.append("claim_token")

        current_sidecar = self._metadata_identity(
            metadata, "current_sidecar_artifact_id"
        )
        current_blocks = self._metadata_identity(metadata, "current_blocks_artifact_id")
        if current_sidecar != binding.expected_current_sidecar_artifact_id:
            mismatches.append("expected_current_sidecar_artifact_id")
        if current_blocks != binding.expected_current_blocks_artifact_id:
            mismatches.append("expected_current_blocks_artifact_id")
        if binding.sidecar_artifact_id != binding.expected_current_sidecar_artifact_id:
            mismatches.append("sidecar_artifact_id")
        if binding.blocks_artifact_id != binding.expected_current_blocks_artifact_id:
            mismatches.append("blocks_artifact_id")

        if operation == "build":
            if document.lightrag_doc_id != binding.lightrag_doc_id:
                mismatches.append("lightrag_doc_id")
            if document.parser_hash != binding.parser_hash:
                mismatches.append("parser_hash")
            if (
                self._metadata_identity(metadata, "current_parse_generation_id")
                != binding.parse_generation_id
            ):
                mismatches.append("parse_generation_id")
            if (
                self._metadata_identity(metadata, "pending_index_hash")
                != binding.index_hash
            ):
                mismatches.append("index_hash")
            if binding.raw_artifact_ids:
                mismatches.append("raw_artifact_ids")
        else:
            pending_lightrag_id = self._metadata_identity(
                metadata, "pending_lightrag_doc_id"
            )
            if pending_lightrag_id != binding.lightrag_doc_id:
                mismatches.append("lightrag_doc_id")
            if (
                document.lightrag_doc_id is not None
                and document.lightrag_doc_id != binding.lightrag_doc_id
            ):
                mismatches.append("current_lightrag_doc_id")
            if (
                self._metadata_identity(metadata, "pending_parser_hash")
                != binding.parser_hash
            ):
                mismatches.append("parser_hash")
            if binding.parse_generation_id != binding.claim_token:
                mismatches.append("parse_generation_id")
            if document.index_hash != binding.index_hash:
                mismatches.append("index_hash")

        if mismatches:
            raise PipelineArtifactBindingStaleError(
                "document snapshot mismatch: " + ", ".join(sorted(set(mismatches)))
            )

    @staticmethod
    def _validate_artifact_rows(
        binding: PipelineArtifactBinding,
        document: DocumentRecord,
        *,
        sidecar: ArtifactRecord | None,
        blocks: ArtifactRecord | None,
        raw_artifacts: tuple[ArtifactRecord, ...],
    ) -> None:
        expected_rows: list[tuple[ArtifactRecord, str]] = []
        if sidecar is not None:
            expected_rows.append((sidecar, "sidecar"))
        if blocks is not None:
            expected_rows.append((blocks, "blocks"))
        expected_rows.extend((artifact, "raw_dir") for artifact in raw_artifacts)
        for artifact, artifact_type in expected_rows:
            mismatches = []
            if artifact.kb_id != binding.kb_id:
                mismatches.append("kb_id")
            if artifact.workspace != binding.workspace:
                mismatches.append("workspace")
            if artifact.document_id != document.id:
                mismatches.append("document_id")
            if artifact.artifact_type != artifact_type:
                mismatches.append("artifact_type")
            if mismatches:
                raise PipelineArtifactBindingStaleError(
                    "artifact row ownership mismatch: " + ", ".join(sorted(mismatches))
                )

    def _validate_build_object_references(
        self,
        binding: PipelineArtifactBinding,
        document: DocumentRecord,
        *,
        sidecar: ArtifactRecord | None,
        blocks: ArtifactRecord | None,
        raw_artifacts: tuple[ArtifactRecord, ...],
    ) -> _ValidatedAuthority:
        if sidecar is None or blocks is None:
            raise ArtifactMigrationRequiredError(
                "build binding requires exact sidecar and blocks artifacts"
            )
        self._require_checksum(sidecar)
        self._require_checksum(blocks)
        sidecar_prefix = self._required_artifact_reference(
            sidecar,
            metadata_key="object_prefix_uri",
            is_prefix=True,
        )
        blocks_uri = self._required_artifact_reference(
            blocks,
            metadata_key="object_uri",
            is_prefix=False,
        )
        blocks_filename = self._safe_metadata_filename(blocks, "filename")
        return _ValidatedAuthority(
            binding=binding,
            document=document,
            sidecar=sidecar,
            blocks=blocks,
            raw_artifacts=raw_artifacts,
            source_object_uri=None,
            sidecar_object_prefix_uri=sidecar_prefix,
            blocks_object_uri=blocks_uri,
            raw_object_prefix_uris=(),
            blocks_filename=blocks_filename,
            raw_directory_names=(),
        )

    def _validate_parse_object_references(
        self,
        binding: PipelineArtifactBinding,
        document: DocumentRecord,
        *,
        sidecar: ArtifactRecord | None,
        blocks: ArtifactRecord | None,
        raw_artifacts: tuple[ArtifactRecord, ...],
    ) -> _ValidatedAuthority:
        source_uri = document.metadata.get("source_object_uri")
        if not isinstance(source_uri, str) or not source_uri:
            raise ArtifactMigrationRequiredError(
                "document source has no object reference"
            )
        self._validate_file_reference(
            source_uri,
            workspace=binding.workspace,
            document_id=binding.document_id,
            namespace="source",
            artifact_id=None,
        )
        if normalize_artifact_checksum(binding.source_hash) is None:
            raise ArtifactMigrationRequiredError(
                "document source has no verifiable checksum"
            )

        parser_engine = document.metadata.get("parser_engine")
        raw_prefixes: list[str] = []
        raw_names: list[str] = []
        seen_names: set[str] = set()
        for artifact in raw_artifacts:
            if artifact.metadata.get("parse_engine") != parser_engine:
                raise PipelineArtifactBindingStaleError(
                    "raw artifact parser snapshot mismatch"
                )
            self._require_checksum(artifact)
            raw_prefixes.append(
                self._required_artifact_reference(
                    artifact,
                    metadata_key="object_prefix_uri",
                    is_prefix=True,
                )
            )
            raw_name = self._safe_metadata_filename(artifact, "raw_directory_name")
            if raw_name in seen_names:
                raise PipelineArtifactBindingStaleError(
                    "raw artifact directory identities overlap"
                )
            seen_names.add(raw_name)
            raw_names.append(raw_name)
        return _ValidatedAuthority(
            binding=binding,
            document=document,
            sidecar=sidecar,
            blocks=blocks,
            raw_artifacts=raw_artifacts,
            source_object_uri=source_uri,
            sidecar_object_prefix_uri=None,
            blocks_object_uri=None,
            raw_object_prefix_uris=tuple(raw_prefixes),
            blocks_filename=None,
            raw_directory_names=tuple(raw_names),
        )

    @staticmethod
    def _require_checksum(artifact: ArtifactRecord) -> str:
        checksum = normalize_artifact_checksum(artifact.checksum)
        if checksum is None:
            raise ArtifactMigrationRequiredError(
                f"{artifact.artifact_type} artifact has no verifiable checksum"
            )
        return checksum

    @staticmethod
    def _safe_metadata_filename(artifact: ArtifactRecord, key: str) -> str:
        value = artifact.metadata.get(key)
        if not isinstance(value, str) or not value:
            raise ArtifactMigrationRequiredError(
                f"{artifact.artifact_type} artifact has no object filename metadata"
            )
        if value in {".", ".."} or PurePath(value).name != value:
            raise PipelineArtifactBindingStaleError(
                f"{artifact.artifact_type} artifact filename is invalid"
            )
        if "/" in value or "\\" in value or any(ord(char) < 32 for char in value):
            raise PipelineArtifactBindingStaleError(
                f"{artifact.artifact_type} artifact filename is invalid"
            )
        return value

    def _required_artifact_reference(
        self,
        artifact: ArtifactRecord,
        *,
        metadata_key: str,
        is_prefix: bool,
    ) -> str:
        value = artifact.metadata.get(metadata_key)
        if not isinstance(value, str) or not value:
            raise ArtifactMigrationRequiredError(
                f"{artifact.artifact_type} artifact has no object reference"
            )
        if is_prefix:
            self._validate_prefix_reference(
                value,
                workspace=artifact.workspace,
                document_id=artifact.document_id,
                namespace="artifacts",
                artifact_id=artifact.id,
            )
        else:
            self._validate_file_reference(
                value,
                workspace=artifact.workspace,
                document_id=artifact.document_id,
                namespace="artifacts",
                artifact_id=artifact.id,
            )
        return value

    def _validate_file_reference(
        self,
        value: str,
        *,
        workspace: str,
        document_id: str,
        namespace: str,
        artifact_id: str | None,
    ) -> None:
        try:
            self._object_storage.validate_document_file_uri(
                value,
                workspace=workspace,
                document_id=document_id,
                namespace=namespace,
                artifact_id=artifact_id,
            )
        except Exception as exc:
            raise PipelineArtifactBindingStaleError(
                "artifact object ownership mismatch"
            ) from exc

    def _validate_prefix_reference(
        self,
        value: str,
        *,
        workspace: str,
        document_id: str,
        namespace: str,
        artifact_id: str,
    ) -> None:
        try:
            self._object_storage.validate_document_prefix_uri(
                value,
                workspace=workspace,
                document_id=document_id,
                namespace=namespace,
                artifact_id=artifact_id,
            )
        except Exception as exc:
            raise PipelineArtifactBindingStaleError(
                "artifact object ownership mismatch"
            ) from exc

    async def _materialize_authority(
        self, authority: _ValidatedAuthority
    ) -> PipelineArtifactSession:
        lease = self._materializer.create_lease()
        try:
            if authority.binding.operation == "build":
                source_path, sidecar_dir, blocks_path = await self._materialize_build(
                    authority, lease
                )
            else:
                source_path, sidecar_dir, blocks_path = await self._materialize_parse(
                    authority, lease
                )
            refreshed_authority = await self._read_and_validate_authority(
                authority.binding
            )
            if refreshed_authority != authority:
                raise PipelineArtifactBindingStaleError(
                    "metadata authority changed during materialization"
                )

            async def finalize_success(
                parsed_data: Mapping[str, Any] | None,
                chunks_count: int | None,
            ) -> PipelineArtifactFinalizationResult:
                return await self._finalize_success(
                    authority.binding,
                    lease=lease,
                    runtime_sidecar_dir=sidecar_dir,
                    runtime_blocks_path=blocks_path,
                    parsed_data=parsed_data,
                    chunks_count=chunks_count,
                )

            return CoordinatedPipelineArtifactSession(
                authority.binding,
                document_service=self._document_service,
                success_finalizer=finalize_success,
                lease=lease,
                source_path=source_path,
                sidecar_dir=sidecar_dir,
                blocks_path=blocks_path,
                sensitive_values=authority.sensitive_object_references,
            )
        except asyncio.CancelledError:
            if not lease.cleanup_deferred:
                lease.defer_cleanup()
            raise
        except PipelineArtifactCoordinatorError:
            lease.cleanup()
            raise
        except Exception as exc:
            lease.cleanup()
            raise PipelineArtifactCoordinatorError(
                "artifact materialization failed"
            ) from exc
        except BaseException:
            lease.cleanup()
            raise

    async def _finalize_success(
        self,
        binding: PipelineArtifactBinding,
        *,
        lease: ArtifactMaterializationLease,
        runtime_sidecar_dir: Path | None,
        runtime_blocks_path: Path | None,
        parsed_data: Mapping[str, Any] | None,
        chunks_count: int | None,
    ) -> PipelineArtifactFinalizationResult:
        """Re-read terminal authority and perform only the binding's transition."""

        chunks_count = self._optional_count(chunks_count, "chunks_count")
        entity_count = self._count_from_parsed_data(parsed_data, "entity_count")
        relation_count = self._count_from_parsed_data(parsed_data, "relation_count")
        try:
            async with self._document_service.kb_write_guard(
                binding.kb_id,
                expected_generation=binding.kb_generation,
            ) as record:
                if record.workspace != binding.workspace:
                    raise PipelineArtifactBindingStaleError(
                        "knowledge-base workspace changed before terminalization"
                    )
                if binding.operation == "parse":
                    return await self._confirm_committed_parse_success(
                        binding,
                        chunks_count=chunks_count,
                        entity_count=entity_count,
                        relation_count=relation_count,
                    )

                authority = await self._read_and_validate_authority(binding)
                if authority.sidecar is None or authority.blocks is None:
                    raise ArtifactMigrationRequiredError(
                        "build terminalization requires exact sidecar and blocks rows"
                    )
                if runtime_sidecar_dir is None or runtime_blocks_path is None:
                    raise PipelineArtifactCoordinatorError(
                        "build terminalization has no processing-owner runtime"
                    )
                return await self._index_service.complete_pipeline_artifact_success(
                    binding,
                    document=authority.document,
                    sidecar_artifact=authority.sidecar,
                    blocks_artifact=authority.blocks,
                    lease=lease,
                    runtime_sidecar_dir=runtime_sidecar_dir,
                    runtime_blocks_path=runtime_blocks_path,
                    chunks_count=chunks_count,
                    entity_count=entity_count,
                    relation_count=relation_count,
                )
        except PipelineArtifactBindingStaleError:
            raise
        except (
            ArtifactPointerConflictError,
            DocumentAttemptOwnershipError,
            DocumentSnapshotConflictError,
            KnowledgeBaseConflictError,
            KnowledgeBaseNotFoundError,
            MetadataRecordNotFoundError,
        ) as exc:
            raise PipelineArtifactBindingStaleError(
                "attempt authority changed before terminalization"
            ) from exc

    async def _confirm_committed_parse_success(
        self,
        binding: PipelineArtifactBinding,
        *,
        chunks_count: int | None,
        entity_count: int | None,
        relation_count: int | None,
    ) -> PipelineArtifactFinalizationResult:
        """Confirm the parse metadata path already committed this exact attempt."""

        raw_artifacts = await self._list_parse_generation_raw_artifacts(binding)
        document = await self._metadata_store.get_document(
            binding.kb_id, binding.document_id
        )
        metadata = document.metadata
        sidecar_id = self._metadata_identity(metadata, "current_sidecar_artifact_id")
        blocks_id = self._metadata_identity(metadata, "current_blocks_artifact_id")
        if (sidecar_id is None) != (blocks_id is None):
            raise PipelineArtifactBindingStaleError(
                "committed parse artifact pointers are incomplete"
            )

        artifact_ids = [
            value
            for value in (sidecar_id, blocks_id, *(item.id for item in raw_artifacts))
            if value is not None
        ]
        document, artifacts = (
            await self._metadata_store.get_document_and_artifacts_by_ids(
                binding.kb_id,
                binding.document_id,
                artifact_ids,
            )
        )
        if document is None or len(artifacts) != len(artifact_ids):
            raise PipelineArtifactBindingStaleError(
                "committed parse artifact authority is incomplete"
            )
        self._validate_committed_parse_document(binding, document)

        sidecar = artifacts.get(sidecar_id) if sidecar_id is not None else None
        blocks = artifacts.get(blocks_id) if blocks_id is not None else None
        exact_raw = tuple(artifacts[item.id] for item in raw_artifacts)
        self._validate_artifact_rows(
            binding,
            document,
            sidecar=sidecar,
            blocks=blocks,
            raw_artifacts=exact_raw,
        )
        for artifact in tuple(
            item for item in (sidecar, blocks, *exact_raw) if item is not None
        ):
            if artifact.metadata.get("parse_generation_id") != binding.claim_token:
                raise PipelineArtifactBindingStaleError(
                    "committed parse artifact generation mismatch"
                )
        if sidecar is not None and blocks is not None:
            self._validate_build_object_references(
                binding,
                document,
                sidecar=sidecar,
                blocks=blocks,
                raw_artifacts=(),
            )
        for artifact in exact_raw:
            self._require_checksum(artifact)
            self._required_artifact_reference(
                artifact,
                metadata_key="object_prefix_uri",
                is_prefix=True,
            )

        committed = binding.committed(
            parse_generation_id=binding.claim_token,
            index_hash=document.index_hash,
            sidecar_artifact_id=sidecar_id,
            blocks_artifact_id=blocks_id,
            raw_artifact_ids=tuple(sorted(item.id for item in exact_raw)),
        )
        return PipelineArtifactFinalizationResult(
            outcome=PipelineArtifactCommitOutcome.COMMITTED,
            committed_binding=committed,
            chunks_count=chunks_count,
            entity_count=entity_count,
            relation_count=relation_count,
        )

    async def _list_parse_generation_raw_artifacts(
        self, binding: PipelineArtifactBinding
    ) -> tuple[ArtifactRecord, ...]:
        matches: dict[str, ArtifactRecord] = {}
        offset = 0
        while True:
            artifacts, total = await self._metadata_store.list_document_artifacts(
                binding.kb_id,
                binding.document_id,
                artifact_type="raw_dir",
                limit=200,
                offset=offset,
            )
            for artifact in artifacts:
                if artifact.metadata.get("parse_generation_id") == binding.claim_token:
                    matches[artifact.id] = artifact
            offset += len(artifacts)
            if not artifacts or offset >= total:
                break
        return tuple(matches[key] for key in sorted(matches))

    def _validate_committed_parse_document(
        self,
        binding: PipelineArtifactBinding,
        document: DocumentRecord,
    ) -> None:
        metadata = document.metadata
        mismatches: list[str] = []
        for field_name, actual, expected in (
            ("kb_id", document.kb_id, binding.kb_id),
            ("workspace", document.workspace, binding.workspace),
            ("document_id", document.id, binding.document_id),
            ("source_hash", document.source_hash or None, binding.source_hash),
            ("parser_hash", document.parser_hash, binding.parser_hash),
            ("lightrag_doc_id", document.lightrag_doc_id, binding.lightrag_doc_id),
            ("index_hash", document.index_hash, binding.index_hash),
            ("status", document.status, "parsed"),
            (
                "current_parse_generation_id",
                self._metadata_identity(metadata, "current_parse_generation_id"),
                binding.claim_token,
            ),
            (
                "last_parse_job_id",
                self._metadata_identity(metadata, "last_parse_job_id"),
                binding.job_id,
            ),
        ):
            if actual != expected:
                mismatches.append(field_name)
        for owner_key in (
            "pending_parse_job_id",
            "pending_parse_claim_token",
            "current_parse_job_id",
            "current_parse_claim_token",
        ):
            if metadata.get(owner_key) is not None:
                mismatches.append(owner_key)
        if mismatches:
            raise PipelineArtifactBindingStaleError(
                "committed parse authority mismatch: "
                + ", ".join(sorted(set(mismatches)))
            )

    @classmethod
    def _count_from_parsed_data(
        cls,
        parsed_data: Mapping[str, Any] | None,
        field_name: str,
    ) -> int | None:
        if parsed_data is None:
            return None
        return cls._optional_count(parsed_data.get(field_name), field_name)

    @staticmethod
    def _optional_count(value: object, field_name: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise PipelineArtifactCoordinatorError(
                f"{field_name} must be a non-negative integer"
            )
        return value

    async def _materialize_build(
        self,
        authority: _ValidatedAuthority,
        lease: ArtifactMaterializationLease,
    ) -> tuple[None, Path, Path]:
        assert authority.sidecar is not None
        assert authority.blocks is not None
        assert authority.sidecar_object_prefix_uri is not None
        assert authority.blocks_object_uri is not None
        assert authority.blocks_filename is not None
        tree = lease.create_document_tree("document.bin")
        runtime_sidecar = await lease.materialize_document_prefix(
            authority.sidecar_object_prefix_uri,
            workspace=authority.binding.workspace,
            document_id=authority.binding.document_id,
            artifact_id=authority.sidecar.id,
            tree=tree,
            directory_name="sidecar",
        )
        runtime_blocks = runtime_sidecar / authority.blocks_filename
        if runtime_blocks.exists():
            if runtime_blocks.is_symlink() or not runtime_blocks.is_file():
                raise PipelineArtifactBindingStaleError(
                    "materialized sidecar blocks entry is invalid"
                )
            runtime_blocks.unlink()
        runtime_blocks = await lease.materialize_document_artifact_file(
            authority.blocks_object_uri,
            workspace=authority.binding.workspace,
            document_id=authority.binding.document_id,
            artifact_id=authority.blocks.id,
            tree=tree,
            directory=runtime_sidecar,
            filename=authority.blocks_filename,
        )
        self._assert_file_checksum(runtime_blocks, authority.blocks)
        self._assert_directory_checksum(runtime_sidecar, authority.sidecar)
        return None, runtime_sidecar, runtime_blocks

    async def _materialize_parse(
        self,
        authority: _ValidatedAuthority,
        lease: ArtifactMaterializationLease,
    ) -> tuple[Path, None, None]:
        assert authority.source_object_uri is not None
        source_name = canonicalize_pipeline_logical_filename(
            authority.document.source_name
        )
        tree = await lease.materialize_document_source(
            authority.source_object_uri,
            workspace=authority.binding.workspace,
            document_id=authority.binding.document_id,
            source_name=source_name,
        )
        source_checksum = file_artifact_checksum(tree.source_path)
        expected_source_checksum = normalize_artifact_checksum(
            authority.binding.source_hash
        )
        if source_checksum != expected_source_checksum:
            raise ArtifactChecksumMismatchError(
                "materialized document source checksum mismatch"
            )
        for artifact, prefix_uri, directory_name in zip(
            authority.raw_artifacts,
            authority.raw_object_prefix_uris,
            authority.raw_directory_names,
            strict=True,
        ):
            raw_dir = await lease.materialize_document_prefix(
                prefix_uri,
                workspace=authority.binding.workspace,
                document_id=authority.binding.document_id,
                artifact_id=artifact.id,
                tree=tree,
                directory_name=directory_name,
            )
            self._assert_directory_checksum(raw_dir, artifact)
        return tree.source_path, None, None

    @staticmethod
    def _assert_file_checksum(path: Path, artifact: ArtifactRecord) -> None:
        expected = normalize_artifact_checksum(artifact.checksum)
        if expected is None or file_artifact_checksum(path) != expected:
            raise ArtifactChecksumMismatchError(
                f"materialized {artifact.artifact_type} checksum mismatch"
            )

    @staticmethod
    def _assert_directory_checksum(path: Path, artifact: ArtifactRecord) -> None:
        expected = normalize_artifact_checksum(artifact.checksum)
        if expected is None or directory_artifact_checksum(path) != expected:
            raise ArtifactChecksumMismatchError(
                f"materialized {artifact.artifact_type} checksum mismatch"
            )


class CoordinatedPipelineArtifactSession:
    """Concrete processing-owner session backed by one scratch lease."""

    def __init__(
        self,
        binding: PipelineArtifactBinding,
        *,
        document_service: DocumentLifecycleService,
        success_finalizer: _SuccessFinalizer,
        lease: ArtifactMaterializationLease,
        source_path: Path | None,
        sidecar_dir: Path | None,
        blocks_path: Path | None,
        sensitive_values: tuple[str, ...],
    ) -> None:
        self._binding = binding
        self._document_service = document_service
        self._success_finalizer = success_finalizer
        self._lease = lease
        self._source_path = source_path
        self._sidecar_dir = sidecar_dir
        self._blocks_path = blocks_path
        self._sensitive_values = sensitive_values
        self._lock = asyncio.Lock()
        self._producer_active = False
        self._finish_called = False
        self._close_called = False
        self._success_handed_off = False
        self._success_result: PipelineArtifactFinalizationResult | None = None
        self._success_error: BaseException | None = None
        self._terminal_outcome: PipelineTerminalOutcome | None = None

    @property
    def binding(self) -> PipelineArtifactBinding:
        return self._binding

    @property
    def source_path(self) -> Path | None:
        return self._source_path

    @property
    def sidecar_dir(self) -> Path | None:
        return self._sidecar_dir

    @property
    def blocks_path(self) -> Path | None:
        return self._blocks_path

    @property
    def producer_active(self) -> bool:
        return self._producer_active

    @property
    def awaiting_owner_terminalization(self) -> bool:
        return self._success_handed_off and not self._close_called

    @property
    def lifecycle_state(self) -> str:
        if self.awaiting_owner_terminalization:
            return "awaiting_h2c_owner_terminalization"
        if self._terminal_outcome is not None:
            return self._terminal_outcome.value
        if self._close_called:
            return "closed"
        if self._lease.cleanup_deferred:
            return "cleanup_deferred"
        return "open"

    @property
    def terminal_outcome(self) -> PipelineTerminalOutcome | None:
        return self._terminal_outcome

    def redact(self, error: object) -> str:
        message = str(error)
        replacements = [
            *self._sensitive_values,
            str(self._lease.path),
            self._lease.path.as_uri(),
        ]
        for path in (self._source_path, self._sidecar_dir, self._blocks_path):
            if path is not None:
                replacements.extend((str(path), path.as_uri()))
        for value in sorted(set(replacements), key=len, reverse=True):
            if value:
                message = message.replace(value, "<artifact-runtime>")
        message = _URI_WITH_AUTHORITY.sub("<artifact-object>", message)
        return message.replace(".lightrag-scratch", "artifact-runtime")

    def defer_cleanup(self) -> None:
        if self._close_called:
            return
        if not self._lease.cleanup_deferred:
            self._lease.defer_cleanup()

    async def finish(self, outcome: PipelineTerminalOutcome) -> None:
        if outcome is PipelineTerminalOutcome.SUCCEEDED:
            await self.handoff_success()
            return
        if outcome not in {
            PipelineTerminalOutcome.FAILED,
            PipelineTerminalOutcome.CANCELLED,
        }:
            raise PipelineArtifactCoordinatorError(
                "session finish requires a terminal failed/cancelled outcome"
            )
        async with self._lock:
            if self._finish_called:
                return
            if self._success_handed_off:
                raise PipelineArtifactCoordinatorError(
                    "successful session is awaiting H2-C owner terminalization"
                )
            self._finish_called = True
            self._terminal_outcome = outcome
            await self._document_service.release_pipeline_artifact_attempt_if_owned(
                self._binding,
                outcome,
            )

    async def handoff_success(
        self,
        *,
        parsed_data: Mapping[str, Any] | None = None,
        chunks_count: int | None = None,
    ) -> PipelineArtifactFinalizationResult:
        async with self._lock:
            if self._success_result is not None:
                return self._success_result
            if self._success_error is not None:
                raise self._success_error
            if self._finish_called or self._close_called:
                raise PipelineArtifactCoordinatorError(
                    "closed session cannot accept successful owner handoff"
                )
            try:
                result = await self._success_finalizer(parsed_data, chunks_count)
            except BaseException as exc:
                self._success_error = exc
                raise
            if not isinstance(result, PipelineArtifactFinalizationResult):
                error = PipelineArtifactCoordinatorError(
                    "success finalizer returned an invalid durable result"
                )
                self._success_error = error
                raise error
            result.__post_init__()
            self._success_result = result
            self._success_handed_off = True
            return result

    async def aclose(self) -> None:
        async with self._lock:
            if self._close_called:
                return
            if self._producer_active:
                self.defer_cleanup()
                return
            self._lease.cleanup()
            self._close_called = True
