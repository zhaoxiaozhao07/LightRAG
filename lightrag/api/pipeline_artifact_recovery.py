"""Crash-stop recovery for committed pipeline artifact terminalization.

This reconciler repairs only the narrow H2-D crash window where KB metadata
already proves that a parse/build attempt committed, while LightRAG's durable
``full_docs`` binding and/or ``doc_status`` terminal row were not published.
It never materializes artifacts, reads object bytes, releases owners, deletes
objects, or performs compensation.

The metadata page reservation is delegated to the durable store keyset API
(``reserve_pipeline_artifact_recovery_page``). The store owns the recovery
cursor row, advances it atomically with version-CAS, and survives process
restart. A stale KB generation is reported via ``ArtifactRecoveryGenerationError``
and skipped without crashing the recovery cycle. When a reservation returns no
candidates the reconciler deletes the durable cursor row; if deletion fails the
cursor row remains as harmless residue.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, cast

from lightrag.artifact_runtime import (
    PipelineArtifactBinding,
    PipelineAttemptCommitOutcomeUnknownError,
    PipelineAttemptCommitStaleError,
    PipelineAttemptCompareAndCommitStorage,
    assert_no_runtime_artifact_payload,
    canonicalize_pipeline_logical_filename,
    commit_pipeline_attempt_if_current,
)
from lightrag.api.document_lifecycle_service import (
    DocumentLifecycleService,
    normalize_artifact_checksum,
)
from lightrag.api.kb_service import (
    KnowledgeBaseConflictError,
    KnowledgeBaseNotFoundError,
)
from lightrag.api.metadata_store import (
    ArtifactRecoveryGenerationError,
    ArtifactRecord,
    DocumentRecord,
    MetadataRecordNotFoundError,
)
from lightrag.base import DocStatus
from lightrag.constants import FULL_DOCS_FORMAT_LIGHTRAG
from lightrag.utils import get_content_summary
from lightrag.utils_pipeline import (
    doc_status_transition_metadata,
    strip_lightrag_doc_prefix,
)

if TYPE_CHECKING:
    from lightrag.api.index_build_service import IndexBuildService
    from lightrag.api.pipeline_artifact_coordinator import (
        PipelineArtifactCoordinator,
    )


_logger = logging.getLogger(__name__)


RecoveryOperation = Literal["parse", "build"]
RecoveryDocumentStatus = Literal["parsed", "ready"]


class PipelineArtifactRecoveryConfigurationError(RuntimeError):
    """The reconciler was constructed outside object-authoritative mode."""


@dataclass(frozen=True, slots=True)
class PipelineArtifactRecoveryError:
    """One durable-safe recovery failure.

    Messages intentionally contain neither exception text nor artifact/object
    locators.  The exception type is retained only as a non-sensitive diagnostic
    category.
    """

    stage: str
    error_code: str
    message: str
    document_id: str | None = None
    lightrag_doc_id: str | None = None


@dataclass(frozen=True, slots=True)
class PipelineArtifactRecoverySummary:
    """Immutable bounded reconciliation result."""

    discovered: int
    finalized: int
    skipped: int
    errors: tuple[PipelineArtifactRecoveryError, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("discovered", "finalized", "skipped"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.finalized + self.skipped + len(self.errors) != self.discovered:
            raise ValueError("Recovery summary counts do not match discovered items")

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def candidate_count(self) -> int:
        return self.discovered

    @property
    def finalized_count(self) -> int:
        return self.finalized

    @property
    def skipped_count(self) -> int:
        return self.skipped


@dataclass(frozen=True, slots=True)
class _Candidate:
    document_id: str
    lightrag_doc_id: str
    binding: PipelineArtifactBinding


@dataclass(frozen=True, slots=True)
class _CommittedAuthority:
    document: DocumentRecord
    committed_binding: PipelineArtifactBinding


@dataclass(frozen=True, slots=True)
class _ValidatedCandidate:
    authority: _CommittedAuthority
    full_doc: dict[str, Any]
    status_row: dict[str, Any] | None


class _RecoverySkip(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class _RecoveryFailure(Exception):
    def __init__(self, stage: str, error_code: str, cause: BaseException) -> None:
        self.stage = stage
        self.error_code = error_code
        self.cause = cause
        super().__init__(error_code)


class PipelineArtifactTerminalizationReconciler:
    """Publish terminal LightRAG rows from exact committed KB authority.

    ``artifact_coordinator`` and ``index_service`` are accepted for explicit
    startup wiring symmetry, but recovery deliberately uses only metadata and
    object-reference validators.  Their materialization/finalization methods are
    never called.
    """

    def __init__(
        self,
        document_service: DocumentLifecycleService,
        artifact_coordinator: PipelineArtifactCoordinator | None = None,
        index_service: IndexBuildService | None = None,
    ) -> None:
        if not document_service.object_authoritative:
            raise PipelineArtifactRecoveryConfigurationError(
                "Pipeline artifact recovery requires object artifact mode"
            )
        object_storage = document_service.object_storage
        if object_storage is None:
            raise PipelineArtifactRecoveryConfigurationError(
                "Pipeline artifact recovery requires object storage"
            )
        self._document_service = document_service
        self._metadata_store = document_service.metadata_store
        self._object_storage = object_storage
        # Retain explicit dependencies for later wiring/introspection without
        # invoking any runtime-owner behavior from this metadata-only reconciler.
        self._artifact_coordinator = artifact_coordinator
        self._index_service = index_service
        # The durable store keyset cursor is the single source of truth across
        # restarts and processes. This lock only serializes same-process
        # reservation calls; cross-process CAS is owned by the store.
        self._recovery_cursor_lock = asyncio.Lock()

    async def reconcile_kb(
        self,
        kb_id: str,
        rag: Any,
        *,
        limit: int = 200,
        kb_generation: str | None = None,
    ) -> PipelineArtifactRecoverySummary:
        """Reconcile at most ``limit`` parsed/ready metadata candidates.

        ``kb_generation`` is the KB generation that owns the durable recovery
        cursor row. When it is omitted, the reconciler resolves the current
        active generation from the metadata store lifecycle table; if no active
        lifecycle row exists the call returns an empty summary. Callers that
        already hold the KB record (e.g. the sweep callback) should pass the
        generation directly to avoid the extra lookup.
        """

        if not isinstance(kb_id, str) or not kb_id.strip():
            raise ValueError("kb_id must be a non-empty string")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        if kb_generation is not None and (
            not isinstance(kb_generation, str) or not kb_generation.strip()
        ):
            raise ValueError("kb_generation must be a non-empty string when provided")
        effective_limit = min(limit, 200)
        full_docs, doc_status = self._rag_storages(rag)

        resolved_generation = await self._resolve_kb_generation(kb_id, kb_generation)
        if resolved_generation is None:
            # No active lifecycle row and no explicit generation: nothing to
            # reserve. The durable cursor requires a KB generation identity.
            return PipelineArtifactRecoverySummary(
                discovered=0,
                finalized=0,
                skipped=0,
            )

        try:
            candidates, skipped, errors, discovered = await self._discover_candidates(
                kb_id,
                resolved_generation,
                full_docs,
                limit=effective_limit,
            )
        except ArtifactRecoveryGenerationError:
            # Stale KB generation: the lifecycle advanced past the cursor's
            # generation. Skip this KB without crashing the recovery sweep.
            _logger.info(
                "Pipeline artifact recovery skipped stale generation for KB %s",
                kb_id,
            )
            return PipelineArtifactRecoverySummary(
                discovered=0,
                finalized=0,
                skipped=0,
            )

        finalized = 0
        for candidate in candidates:
            try:
                await self._reconcile_candidate(
                    kb_id,
                    candidate,
                    full_docs=full_docs,
                    doc_status=doc_status,
                )
            except asyncio.CancelledError:
                raise
            except _RecoverySkip:
                skipped += 1
            except _RecoveryFailure as exc:
                errors.append(self._safe_error(candidate, exc))
            except Exception as exc:  # noqa: BLE001 - isolate each candidate
                errors.append(
                    self._safe_error(
                        candidate,
                        _RecoveryFailure(
                            "candidate",
                            "pipeline_artifact_recovery_failed",
                            exc,
                        ),
                    )
                )
            else:
                finalized += 1

        # When the durable reservation returned no rows, both ``parsed`` and
        # ``ready`` are drained for this KB generation. Remove the cursor row
        # so the table only tracks KBs with outstanding terminalization work.
        # Cursor rows are harmless residue if the deletion fails.
        if discovered == 0:
            await self._delete_recovery_cursor_safely(kb_id, resolved_generation)

        return PipelineArtifactRecoverySummary(
            discovered=discovered,
            finalized=finalized,
            skipped=skipped,
            errors=tuple(errors),
        )

    @staticmethod
    def _rag_storages(rag: Any) -> tuple[Any, Any]:
        full_docs = getattr(rag, "full_docs", None)
        doc_status = getattr(rag, "doc_status", None)
        for name, storage in (("full_docs", full_docs), ("doc_status", doc_status)):
            if storage is None:
                raise PipelineArtifactRecoveryConfigurationError(
                    f"Recovery RAG has no {name} storage"
                )
            has_reader = callable(getattr(storage, "get_by_id", None)) or callable(
                getattr(storage, "get_by_ids", None)
            )
            if not has_reader:
                raise PipelineArtifactRecoveryConfigurationError(
                    f"Recovery RAG {name} storage lacks point-read"
                )
            if not isinstance(storage, PipelineAttemptCompareAndCommitStorage):
                raise PipelineArtifactRecoveryConfigurationError(
                    f"Recovery RAG {name} storage lacks "
                    "compare_and_commit_pipeline_attempt"
                )
        return full_docs, doc_status

    async def _discover_candidates(
        self,
        kb_id: str,
        kb_generation: str,
        full_docs: Any,
        *,
        limit: int,
    ) -> tuple[
        list[_Candidate],
        int,
        list[PipelineArtifactRecoveryError],
        int,
    ]:
        """List only parsed/ready documents, then read exact full_docs IDs."""

        selected, errors = await self._reserve_metadata_rows(
            kb_id,
            kb_generation,
            limit=limit,
        )

        candidates: list[_Candidate] = []
        skipped = 0
        for document in selected:
            lightrag_doc_id = document.lightrag_doc_id
            if not isinstance(lightrag_doc_id, str) or not lightrag_doc_id:
                skipped += 1
                continue
            try:
                full_doc = await self._storage_get_by_id(full_docs, lightrag_doc_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    self._safe_discovery_error(
                        stage="full_docs_read",
                        code="full_docs_candidate_read_failed",
                        cause=exc,
                        document_id=document.id,
                        lightrag_doc_id=lightrag_doc_id,
                    )
                )
                continue
            if not isinstance(full_doc, Mapping):
                skipped += 1
                continue
            raw_binding = full_doc.get("artifact_binding")
            if not isinstance(raw_binding, Mapping):
                skipped += 1
                continue
            try:
                binding = PipelineArtifactBinding.from_mapping(
                    raw_binding,
                    expected_workspace=document.workspace,
                )
            except (TypeError, ValueError) as exc:
                errors.append(
                    self._safe_discovery_error(
                        stage="binding_decode",
                        code="artifact_binding_invalid",
                        cause=exc,
                        document_id=document.id,
                        lightrag_doc_id=lightrag_doc_id,
                    )
                )
                continue
            expected_operation = "parse" if document.status == "parsed" else "build"
            if (
                binding.state not in {"claimed", "committed"}
                or binding.operation != expected_operation
                or binding.kb_id != kb_id
                or binding.document_id != document.id
                or binding.lightrag_doc_id != lightrag_doc_id
            ):
                skipped += 1
                continue
            candidates.append(
                _Candidate(
                    document_id=document.id,
                    lightrag_doc_id=lightrag_doc_id,
                    binding=binding,
                )
            )

        # Every selected metadata row is exactly one finalized/skipped/error
        # outcome. Status-list errors are additional discovered error items.
        discovered_count = len(selected) + sum(
            1 for error in errors if error.document_id is None
        )
        return candidates, skipped, errors, discovered_count

    async def _reserve_metadata_rows(
        self,
        kb_id: str,
        kb_generation: str,
        *,
        limit: int,
    ) -> tuple[list[DocumentRecord], list[PipelineArtifactRecoveryError]]:
        """Reserve one bounded metadata page via the durable store keyset API.

        The store owns the recovery cursor row, advances it atomically with a
        version-CAS inside its own transaction, and guards the KB generation.
        Cancellation of this call therefore leaves the cursor in its pre-call
        state; the store's CAS also guarantees that two concurrent calls
        (in-process or cross-process) never observe the same page.

        ``ArtifactRecoveryGenerationError`` propagates to the caller so the
        sweep can skip the stale KB without crashing. Any other failure is
        reported as a discovery error and the page is treated as empty for
        this call.
        """

        reserve_attr = getattr(
            self._metadata_store,
            "reserve_pipeline_artifact_recovery_page",
            None,
        )
        if not callable(reserve_attr):
            return [], [
                self._safe_discovery_error(
                    stage="reserve_recovery_page",
                    code="metadata_recovery_reserve_unavailable",
                    cause=RuntimeError(
                        "metadata store does not implement "
                        "reserve_pipeline_artifact_recovery_page"
                    ),
                )
            ]

        reserve = cast(
            Callable[[str, str, int], Awaitable[list[DocumentRecord]]],
            reserve_attr,
        )
        async with self._recovery_cursor_lock:
            try:
                reserved = await reserve(kb_id, kb_generation, limit)
            except ArtifactRecoveryGenerationError:
                raise
            except asyncio.CancelledError:
                raise

        selected: list[DocumentRecord] = []
        seen_ids: set[str] = set()
        for document in reserved:
            if not isinstance(document, DocumentRecord) or document.id in seen_ids:
                continue
            seen_ids.add(document.id)
            selected.append(document)
        return selected, []

    async def _resolve_kb_generation(
        self,
        kb_id: str,
        kb_generation: str | None,
    ) -> str | None:
        """Resolve the active KB generation for the durable cursor identity.

        When the caller supplies a generation it is used verbatim. Otherwise
        the reconciler asks the metadata store for the active lifecycle row.
        KBs without a lifecycle row cannot own a durable cursor identity and
        are skipped until one is registered.
        """

        if kb_generation is not None:
            return kb_generation
        get_lifecycle_attr = getattr(self._metadata_store, "get_kb_lifecycle", None)
        if not callable(get_lifecycle_attr):
            return None
        get_kb_lifecycle = cast(
            Callable[[str], Awaitable[Any]],
            get_lifecycle_attr,
        )
        try:
            lifecycle = await get_kb_lifecycle(kb_id)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - lifecycle lookup is best-effort here
            return None
        if lifecycle is None or lifecycle.state != "active":
            return None
        generation = getattr(lifecycle, "generation", None)
        if not isinstance(generation, str) or not generation.strip():
            return None
        return generation

    async def _delete_recovery_cursor_safely(
        self,
        kb_id: str,
        kb_generation: str,
    ) -> None:
        """Best-effort cleanup of the durable cursor row once a KB drains.

        Cursor rows are harmless residue: a later reservation simply re-uses
        or recreates them. If the additive ``delete_artifact_recovery_cursor``
        method is temporarily unavailable (e.g. during parallel writer
        integration) or raises, the warning is logged and the sweep continues.
        """

        delete_cursor_attr = getattr(
            self._metadata_store,
            "delete_artifact_recovery_cursor",
            None,
        )
        if not callable(delete_cursor_attr):
            return
        delete_cursor = cast(
            Callable[[str, str], Awaitable[bool]],
            delete_cursor_attr,
        )
        try:
            await delete_cursor(kb_id, kb_generation)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - residue is harmless
            _logger.warning(
                "Pipeline artifact recovery cursor cleanup failed for KB %s "
                "(error_type=%s)",
                kb_id,
                type(exc).__name__,
            )

    async def _reconcile_candidate(
        self,
        kb_id: str,
        candidate: _Candidate,
        *,
        full_docs: Any,
        doc_status: Any,
    ) -> None:
        try:
            async with self._document_service.kb_write_guard(
                kb_id,
                expected_generation=candidate.binding.kb_generation,
            ) as record:
                if record.workspace != candidate.binding.workspace:
                    raise _RecoverySkip("kb_workspace_changed")

                validated = await self._read_and_validate_candidate(
                    candidate,
                    full_docs=full_docs,
                    doc_status=doc_status,
                )
                committed = validated.authority.committed_binding
                if self._status_is_exact(
                    validated.status_row,
                    committed,
                    validated.authority.document,
                ):
                    raise _RecoverySkip("already_finalized")

                write_full_doc, write_status = await self._read_durable_rows(
                    candidate,
                    full_docs=full_docs,
                    doc_status=doc_status,
                    stage="full_docs_write_revalidation",
                )
                write_binding = self._binding_from_full_doc(
                    write_full_doc,
                    expected_workspace=committed.workspace,
                )
                if not self._binding_is_allowed(candidate, write_binding, committed):
                    raise _RecoverySkip("binding_changed_before_full_docs_write")
                self._assert_status_attempt_is_current(write_status, write_binding)
                full_doc_payload = self._committed_full_doc_payload(
                    write_full_doc,
                    committed,
                )
                assert_no_runtime_artifact_payload(
                    {candidate.lightrag_doc_id: full_doc_payload},
                    context="pipeline artifact recovery full_docs write",
                )
                if write_full_doc != full_doc_payload:
                    try:
                        await commit_pipeline_attempt_if_current(
                            full_docs,
                            candidate.lightrag_doc_id,
                            full_doc_payload,
                            expected_attempt_token=write_binding.claim_token,
                            row_kind="full_docs",
                        )
                    except asyncio.CancelledError:
                        raise
                    except PipelineAttemptCommitStaleError as exc:
                        raise _RecoverySkip(
                            "binding_changed_during_full_docs_cas"
                        ) from exc
                    except PipelineAttemptCommitOutcomeUnknownError as exc:
                        raise _RecoveryFailure(
                            "full_docs_write",
                            "committed_binding_write_outcome_unknown",
                            exc,
                        ) from exc
                    except Exception as exc:  # noqa: BLE001
                        raise _RecoveryFailure(
                            "full_docs_write",
                            "committed_binding_write_failed",
                            exc,
                        ) from exc

                # Re-read all three durable authorities after the binding CAS
                # and immediately before the status write. This is the required
                # write-time stale-owner/generation revalidation.
                terminal = await self._read_and_validate_candidate(
                    candidate,
                    full_docs=full_docs,
                    doc_status=doc_status,
                )
                if terminal.authority.committed_binding != committed:
                    raise _RecoverySkip("committed_authority_changed")
                current_binding = self._binding_from_full_doc(
                    terminal.full_doc,
                    expected_workspace=committed.workspace,
                )
                if current_binding != committed:
                    raise _RecoverySkip("committed_binding_not_visible")
                if self._status_is_exact(
                    terminal.status_row,
                    committed,
                    terminal.authority.document,
                ):
                    return

                write_full_doc, write_status = await self._read_durable_rows(
                    candidate,
                    full_docs=full_docs,
                    doc_status=doc_status,
                    stage="doc_status_write_revalidation",
                )
                write_binding = self._binding_from_full_doc(
                    write_full_doc,
                    expected_workspace=committed.workspace,
                )
                if write_binding != committed:
                    raise _RecoverySkip("binding_changed_before_status_write")
                self._assert_status_attempt_is_current(write_status, committed)
                if self._status_is_exact(
                    write_status,
                    committed,
                    terminal.authority.document,
                ):
                    return
                status_payload = self._processed_status_payload(
                    write_full_doc,
                    write_status,
                    committed,
                    terminal.authority.document,
                )
                assert_no_runtime_artifact_payload(
                    {candidate.lightrag_doc_id: status_payload},
                    context="pipeline artifact recovery doc_status write",
                )
                try:
                    await commit_pipeline_attempt_if_current(
                        doc_status,
                        candidate.lightrag_doc_id,
                        status_payload,
                        expected_attempt_token=committed.claim_token,
                        row_kind="doc_status",
                    )
                except asyncio.CancelledError:
                    raise
                except PipelineAttemptCommitStaleError as exc:
                    raise _RecoverySkip("status_changed_during_doc_status_cas") from exc
                except PipelineAttemptCommitOutcomeUnknownError as exc:
                    raise _RecoveryFailure(
                        "doc_status_write",
                        "processed_status_write_outcome_unknown",
                        exc,
                    ) from exc
                except Exception as exc:  # noqa: BLE001
                    raise _RecoveryFailure(
                        "doc_status_write",
                        "processed_status_write_failed",
                        exc,
                    ) from exc

        except asyncio.CancelledError:
            raise
        except _RecoverySkip:
            raise
        except (KnowledgeBaseConflictError, KnowledgeBaseNotFoundError):
            raise _RecoverySkip("kb_generation_changed") from None
        except _RecoveryFailure:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _RecoveryFailure(
                "kb_guard",
                "kb_recovery_guard_failed",
                exc,
            ) from exc

    async def _read_and_validate_candidate(
        self,
        candidate: _Candidate,
        *,
        full_docs: Any,
        doc_status: Any,
    ) -> _ValidatedCandidate:
        full_doc, status_row = await self._read_durable_rows(
            candidate,
            full_docs=full_docs,
            doc_status=doc_status,
            stage="durable_read",
        )

        current_binding = self._binding_from_full_doc(
            full_doc,
            expected_workspace=candidate.binding.workspace,
        )
        if not self._same_attempt(candidate.binding, current_binding):
            raise _RecoverySkip("binding_attempt_changed")
        if candidate.binding.state == "claimed":
            self._validate_claimed_binding_shape(candidate.binding)
        if current_binding.state == "claimed":
            self._validate_claimed_binding_shape(current_binding)
        self._assert_status_attempt_is_current(status_row, current_binding)

        authority = await self._validate_committed_authority(current_binding)
        if not self._binding_is_allowed(
            candidate,
            current_binding,
            authority.committed_binding,
        ):
            raise _RecoverySkip("binding_state_changed")
        return _ValidatedCandidate(
            authority=authority,
            full_doc=full_doc,
            status_row=status_row,
        )

    async def _read_durable_rows(
        self,
        candidate: _Candidate,
        *,
        full_docs: Any,
        doc_status: Any,
        stage: str,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        try:
            full_doc_value, status_value = await asyncio.gather(
                self._storage_get_by_id(full_docs, candidate.lightrag_doc_id),
                self._storage_get_by_id(doc_status, candidate.lightrag_doc_id),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _RecoveryFailure(
                stage,
                "pipeline_durable_rows_read_failed",
                exc,
            ) from exc
        if not isinstance(full_doc_value, Mapping):
            raise _RecoverySkip("full_docs_row_missing")
        full_doc = dict(full_doc_value)
        if status_value is not None and not isinstance(status_value, Mapping):
            raise _RecoverySkip("doc_status_row_invalid")
        status_row = dict(status_value) if isinstance(status_value, Mapping) else None
        return full_doc, status_row

    @staticmethod
    async def _storage_get_by_id(storage: Any, key: str) -> Any:
        get_by_id = getattr(storage, "get_by_id", None)
        if callable(get_by_id):
            reader = cast(Callable[[str], Awaitable[Any]], get_by_id)
            return await reader(key)
        get_by_ids = getattr(storage, "get_by_ids", None)
        if not callable(get_by_ids):  # pragma: no cover - constructor guards it
            raise PipelineArtifactRecoveryConfigurationError(
                "Recovery storage has no point-read operation"
            )
        batch_reader = cast(Callable[[list[str]], Awaitable[Any]], get_by_ids)
        rows = await batch_reader([key])
        if not isinstance(rows, (list, tuple)) or not rows:
            return None
        return rows[0]

    @staticmethod
    def _binding_from_full_doc(
        full_doc: Mapping[str, Any],
        *,
        expected_workspace: str,
    ) -> PipelineArtifactBinding:
        raw_binding = full_doc.get("artifact_binding")
        if not isinstance(raw_binding, Mapping):
            raise _RecoverySkip("artifact_binding_missing")
        try:
            return PipelineArtifactBinding.from_mapping(
                raw_binding,
                expected_workspace=expected_workspace,
            )
        except (TypeError, ValueError) as exc:
            raise _RecoverySkip("artifact_binding_invalid") from exc

    @staticmethod
    def _same_attempt(
        discovered: PipelineArtifactBinding,
        current: PipelineArtifactBinding,
    ) -> bool:
        fixed_fields = (
            "version",
            "authority",
            "operation",
            "kb_id",
            "kb_generation",
            "workspace",
            "document_id",
            "lightrag_doc_id",
            "job_id",
            "claim_token",
            "source_hash",
            "parser_hash",
            "expected_current_sidecar_artifact_id",
            "expected_current_blocks_artifact_id",
        )
        if any(
            getattr(discovered, field_name) != getattr(current, field_name)
            for field_name in fixed_fields
        ):
            return False
        if discovered.operation == "build":
            return bool(
                discovered.parse_generation_id == current.parse_generation_id
                and discovered.index_hash == current.index_hash
                and not discovered.raw_artifact_ids
                and not current.raw_artifact_ids
            )
        return bool(
            discovered.parse_generation_id == discovered.claim_token
            and current.parse_generation_id == current.claim_token
            and discovered.index_hash == current.index_hash
        )

    @staticmethod
    def _binding_is_allowed(
        candidate: _Candidate,
        current: PipelineArtifactBinding,
        committed: PipelineArtifactBinding,
    ) -> bool:
        if current == committed:
            return True
        return candidate.binding.state == "claimed" and current == candidate.binding

    @staticmethod
    def _validate_claimed_binding_shape(binding: PipelineArtifactBinding) -> None:
        if binding.state != "claimed":
            return
        if binding.operation == "build":
            if (
                binding.sidecar_artifact_id is None
                or binding.blocks_artifact_id is None
                or binding.sidecar_artifact_id
                != binding.expected_current_sidecar_artifact_id
                or binding.blocks_artifact_id
                != binding.expected_current_blocks_artifact_id
                or binding.raw_artifact_ids
            ):
                raise _RecoverySkip("claimed_build_binding_invalid")
            return

        if (
            binding.parse_generation_id != binding.claim_token
            or binding.sidecar_artifact_id
            != binding.expected_current_sidecar_artifact_id
            or binding.blocks_artifact_id != binding.expected_current_blocks_artifact_id
            or (
                binding.expected_current_sidecar_artifact_id is None
                and binding.expected_current_blocks_artifact_id is not None
            )
            or (
                binding.expected_current_sidecar_artifact_id is not None
                and binding.expected_current_blocks_artifact_id is None
            )
        ):
            raise _RecoverySkip("claimed_parse_binding_invalid")
        artifact_ids = [
            value
            for value in (
                binding.sidecar_artifact_id,
                binding.blocks_artifact_id,
                *binding.raw_artifact_ids,
            )
            if value is not None
        ]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise _RecoverySkip("claimed_parse_binding_artifact_overlap")

    @staticmethod
    def _assert_status_attempt_is_current(
        status_row: Mapping[str, Any] | None,
        binding: PipelineArtifactBinding,
    ) -> None:
        if status_row is None:
            return
        metadata = status_row.get("metadata")
        token = (
            metadata.get("pipeline_attempt_token")
            if isinstance(metadata, Mapping)
            else None
        )
        if token != binding.claim_token:
            raise _RecoverySkip("doc_status_attempt_changed")

    async def _validate_committed_authority(
        self,
        binding: PipelineArtifactBinding,
    ) -> _CommittedAuthority:
        if binding.operation == "parse":
            return await self._validate_committed_parse(binding)
        return await self._validate_committed_build(binding)

    async def _validate_committed_parse(
        self,
        binding: PipelineArtifactBinding,
    ) -> _CommittedAuthority:
        try:
            document = await self._metadata_store.get_document(
                binding.kb_id,
                binding.document_id,
            )
        except asyncio.CancelledError:
            raise
        except MetadataRecordNotFoundError as exc:
            raise _RecoverySkip("document_missing") from exc
        except Exception as exc:  # noqa: BLE001
            raise _RecoveryFailure(
                "parse_authority_read",
                "parse_metadata_read_failed",
                exc,
            ) from exc
        self._validate_parse_document(binding, document)

        sidecar_id = self._optional_metadata_identity(
            document.metadata,
            "current_sidecar_artifact_id",
        )
        blocks_id = self._optional_metadata_identity(
            document.metadata,
            "current_blocks_artifact_id",
        )
        if (sidecar_id is None) != (blocks_id is None):
            raise _RecoverySkip("parse_artifact_pointers_partial")
        generation_artifacts = await self._list_parse_generation_artifacts(binding)
        expected_artifact_count = document.metadata.get("artifact_count")
        if (
            isinstance(expected_artifact_count, bool)
            or not isinstance(expected_artifact_count, int)
            or expected_artifact_count < 0
            or expected_artifact_count != len(generation_artifacts)
        ):
            raise _RecoverySkip("parse_artifact_generation_partial")
        generation_ids = {artifact.id for artifact in generation_artifacts}
        if len(generation_ids) != len(generation_artifacts):
            raise _RecoverySkip("parse_artifact_id_overlap")
        if sidecar_id is not None and sidecar_id not in generation_ids:
            raise _RecoverySkip("parse_sidecar_generation_missing")
        if blocks_id is not None and blocks_id not in generation_ids:
            raise _RecoverySkip("parse_blocks_generation_missing")
        raw_artifacts = tuple(
            artifact
            for artifact in generation_artifacts
            if artifact.artifact_type == "raw_dir"
        )
        artifact_ids = sorted(generation_ids)
        document, artifacts = await self._read_document_artifacts(
            binding,
            artifact_ids,
            stage="parse_artifacts_read",
        )
        self._validate_parse_document(binding, document)
        if len(artifacts) != len(artifact_ids):
            raise _RecoverySkip("parse_artifact_rows_partial")

        sidecar = artifacts.get(sidecar_id) if sidecar_id is not None else None
        blocks = artifacts.get(blocks_id) if blocks_id is not None else None
        exact_raw = tuple(artifacts[artifact.id] for artifact in raw_artifacts)
        terminal_artifacts: list[tuple[ArtifactRecord, str]] = []
        if sidecar is not None:
            terminal_artifacts.append((sidecar, "sidecar"))
        if blocks is not None:
            terminal_artifacts.append((blocks, "blocks"))
        terminal_artifacts.extend((artifact, "raw_dir") for artifact in exact_raw)
        for artifact, artifact_type in terminal_artifacts:
            self._validate_artifact_row(binding, artifact, artifact_type)
            if artifact.metadata.get("parse_generation_id") != binding.claim_token:
                raise _RecoverySkip("parse_artifact_generation_mismatch")
            self._validate_artifact_object_reference(artifact)
            self._validate_parser_snapshot(artifact, document, binding)

        self._validate_parse_source_reference(binding, document)
        committed = binding.committed(
            parse_generation_id=binding.claim_token,
            index_hash=document.index_hash,
            sidecar_artifact_id=sidecar_id,
            blocks_artifact_id=blocks_id,
            raw_artifact_ids=tuple(sorted(artifact.id for artifact in exact_raw)),
        )
        return _CommittedAuthority(document=document, committed_binding=committed)

    async def _validate_committed_build(
        self,
        binding: PipelineArtifactBinding,
    ) -> _CommittedAuthority:
        try:
            document = await self._metadata_store.get_document(
                binding.kb_id,
                binding.document_id,
            )
        except asyncio.CancelledError:
            raise
        except MetadataRecordNotFoundError as exc:
            raise _RecoverySkip("document_missing") from exc
        except Exception as exc:  # noqa: BLE001
            raise _RecoveryFailure(
                "build_authority_read",
                "build_metadata_read_failed",
                exc,
            ) from exc
        self._validate_build_document(binding, document)

        sidecar_id = self._required_metadata_identity(
            document.metadata,
            "current_sidecar_artifact_id",
        )
        blocks_id = self._required_metadata_identity(
            document.metadata,
            "current_blocks_artifact_id",
        )
        if sidecar_id == blocks_id:
            raise _RecoverySkip("build_artifact_id_overlap")
        document, artifacts = await self._read_document_artifacts(
            binding,
            [sidecar_id, blocks_id],
            stage="build_artifacts_read",
        )
        self._validate_build_document(binding, document)
        if len(artifacts) != 2:
            raise _RecoverySkip("build_artifact_rows_partial")
        sidecar = artifacts[sidecar_id]
        blocks = artifacts[blocks_id]
        self._validate_artifact_row(binding, sidecar, "sidecar")
        self._validate_artifact_row(binding, blocks, "blocks")
        self._validate_artifact_object_reference(sidecar)
        self._validate_artifact_object_reference(blocks)
        self._validate_parser_snapshot(sidecar, document, binding)
        self._validate_parser_snapshot(blocks, document, binding)
        self._validate_build_pointer_generation(
            sidecar,
            expected_artifact_id=binding.expected_current_sidecar_artifact_id,
            claim_token=binding.claim_token,
        )
        self._validate_build_pointer_generation(
            blocks,
            expected_artifact_id=binding.expected_current_blocks_artifact_id,
            claim_token=binding.claim_token,
        )
        self._validate_document_counts(document)

        committed = binding.committed(
            parse_generation_id=binding.parse_generation_id,
            index_hash=document.index_hash,
            sidecar_artifact_id=sidecar_id,
            blocks_artifact_id=blocks_id,
            raw_artifact_ids=(),
        )
        return _CommittedAuthority(document=document, committed_binding=committed)

    def _validate_parse_document(
        self,
        binding: PipelineArtifactBinding,
        document: DocumentRecord,
    ) -> None:
        mismatches = self._document_identity_mismatches(binding, document)
        expected = (
            ("status", document.status, "parsed"),
            ("parser_hash", document.parser_hash, binding.parser_hash),
            ("index_hash", document.index_hash, binding.index_hash),
            (
                "current_parse_generation_id",
                document.metadata.get("current_parse_generation_id"),
                binding.claim_token,
            ),
            (
                "last_parse_job_id",
                document.metadata.get("last_parse_job_id"),
                binding.job_id,
            ),
        )
        mismatches.extend(name for name, actual, wanted in expected if actual != wanted)
        if binding.parse_generation_id != binding.claim_token:
            mismatches.append("parse_generation_id")
        if (binding.expected_current_sidecar_artifact_id is None) != (
            binding.expected_current_blocks_artifact_id is None
        ):
            mismatches.append("expected_artifact_pointers")
        self._append_active_owner_mismatches(document, "parse", mismatches)
        parser_engine = document.metadata.get("parse_engine")
        if not isinstance(parser_engine, str) or not parser_engine:
            mismatches.append("parse_engine")
        if mismatches:
            raise _RecoverySkip("committed_parse_authority_mismatch")

    def _validate_build_document(
        self,
        binding: PipelineArtifactBinding,
        document: DocumentRecord,
    ) -> None:
        mismatches = self._document_identity_mismatches(binding, document)
        expected = (
            ("status", document.status, "ready"),
            ("parser_hash", document.parser_hash, binding.parser_hash),
            ("index_hash", document.index_hash, binding.index_hash),
            (
                "current_parse_generation_id",
                document.metadata.get("current_parse_generation_id"),
                binding.parse_generation_id,
            ),
            (
                "current_build_generation_id",
                document.metadata.get("current_build_generation_id"),
                binding.claim_token,
            ),
            (
                "last_build_job_id",
                document.metadata.get("last_build_job_id"),
                binding.job_id,
            ),
        )
        mismatches.extend(name for name, actual, wanted in expected if actual != wanted)
        if not binding.parse_generation_id or not binding.index_hash:
            mismatches.append("build_generation_snapshot")
        if normalize_artifact_checksum(binding.source_hash) is None:
            mismatches.append("source_hash")
        if (
            binding.expected_current_sidecar_artifact_id is None
            or binding.expected_current_blocks_artifact_id is None
        ):
            mismatches.append("expected_artifact_pointers")
        if binding.raw_artifact_ids:
            mismatches.append("raw_artifact_ids")
        self._append_active_owner_mismatches(document, "build", mismatches)
        if mismatches:
            raise _RecoverySkip("committed_build_authority_mismatch")

    @staticmethod
    def _document_identity_mismatches(
        binding: PipelineArtifactBinding,
        document: DocumentRecord,
    ) -> list[str]:
        expected = (
            ("kb_id", document.kb_id, binding.kb_id),
            ("workspace", document.workspace, binding.workspace),
            ("document_id", document.id, binding.document_id),
            ("lightrag_doc_id", document.lightrag_doc_id, binding.lightrag_doc_id),
            ("source_hash", document.source_hash or None, binding.source_hash),
        )
        return [name for name, actual, wanted in expected if actual != wanted]

    @staticmethod
    def _append_active_owner_mismatches(
        document: DocumentRecord,
        operation: RecoveryOperation,
        mismatches: list[str],
    ) -> None:
        for key in (
            f"pending_{operation}_job_id",
            f"pending_{operation}_claim_token",
            f"current_{operation}_job_id",
            f"current_{operation}_claim_token",
        ):
            if document.metadata.get(key) is not None:
                mismatches.append(key)

    async def _list_parse_generation_artifacts(
        self,
        binding: PipelineArtifactBinding,
    ) -> tuple[ArtifactRecord, ...]:
        matches: dict[str, ArtifactRecord] = {}
        offset = 0
        while True:
            try:
                artifacts, total = await self._metadata_store.list_document_artifacts(
                    binding.kb_id,
                    binding.document_id,
                    limit=200,
                    offset=offset,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise _RecoveryFailure(
                    "parse_generation_artifacts_read",
                    "parse_generation_artifacts_read_failed",
                    exc,
                ) from exc
            for artifact in artifacts:
                if artifact.metadata.get("parse_generation_id") == binding.claim_token:
                    matches[artifact.id] = artifact
            offset += len(artifacts)
            if not artifacts or offset >= total:
                break
        return tuple(matches[artifact_id] for artifact_id in sorted(matches))

    async def _read_document_artifacts(
        self,
        binding: PipelineArtifactBinding,
        artifact_ids: list[str],
        *,
        stage: str,
    ) -> tuple[DocumentRecord, dict[str, ArtifactRecord]]:
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
        except Exception as exc:  # noqa: BLE001
            raise _RecoveryFailure(
                stage,
                "artifact_authority_read_failed",
                exc,
            ) from exc
        if document is None:
            raise _RecoverySkip("document_missing")
        return document, artifacts

    @staticmethod
    def _optional_metadata_identity(
        metadata: Mapping[str, Any],
        key: str,
    ) -> str | None:
        value = metadata.get(key)
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise _RecoverySkip("metadata_identity_invalid")
        return value

    @classmethod
    def _required_metadata_identity(
        cls,
        metadata: Mapping[str, Any],
        key: str,
    ) -> str:
        value = cls._optional_metadata_identity(metadata, key)
        if value is None:
            raise _RecoverySkip("metadata_identity_missing")
        return value

    @staticmethod
    def _validate_artifact_row(
        binding: PipelineArtifactBinding,
        artifact: ArtifactRecord,
        artifact_type: str,
    ) -> None:
        if (
            artifact.kb_id != binding.kb_id
            or artifact.workspace != binding.workspace
            or artifact.document_id != binding.document_id
            or artifact.artifact_type != artifact_type
        ):
            raise _RecoverySkip("artifact_row_ownership_mismatch")
        if normalize_artifact_checksum(artifact.checksum) is None:
            raise _RecoverySkip("artifact_checksum_invalid")

    def _validate_artifact_object_reference(self, artifact: ArtifactRecord) -> None:
        if artifact.artifact_type in {"sidecar", "raw_dir"}:
            key = "object_prefix_uri"
            value = artifact.metadata.get(key)
            if not isinstance(value, str) or not value:
                raise _RecoverySkip("artifact_object_reference_missing")
            try:
                self._object_storage.validate_document_prefix_uri(
                    value,
                    workspace=artifact.workspace,
                    document_id=artifact.document_id,
                    namespace="artifacts",
                    artifact_id=artifact.id,
                )
            except Exception as exc:
                raise _RecoverySkip("artifact_object_ownership_mismatch") from exc
        elif artifact.artifact_type == "blocks":
            key = "object_uri"
            value = artifact.metadata.get(key)
            if not isinstance(value, str) or not value:
                raise _RecoverySkip("artifact_object_reference_missing")
            try:
                self._object_storage.validate_document_file_uri(
                    value,
                    workspace=artifact.workspace,
                    document_id=artifact.document_id,
                    namespace="artifacts",
                    artifact_id=artifact.id,
                )
            except Exception as exc:
                raise _RecoverySkip("artifact_object_ownership_mismatch") from exc
        else:  # pragma: no cover - guarded by callers
            raise _RecoverySkip("artifact_type_not_terminalizable")

        filename_key = (
            "raw_directory_name" if artifact.artifact_type == "raw_dir" else "filename"
        )
        filename = artifact.metadata.get(filename_key)
        try:
            canonicalize_pipeline_logical_filename(filename)
        except (TypeError, ValueError) as exc:
            raise _RecoverySkip("artifact_filename_invalid") from exc

    @staticmethod
    def _validate_parser_snapshot(
        artifact: ArtifactRecord,
        document: DocumentRecord,
        binding: PipelineArtifactBinding,
    ) -> None:
        parser_engine = document.metadata.get("parse_engine")
        artifact_engine = artifact.metadata.get("parse_engine")
        if artifact_engine != parser_engine:
            raise _RecoverySkip("artifact_parser_engine_mismatch")
        artifact_parser_hash = artifact.metadata.get("parser_hash")
        if (
            artifact_parser_hash is not None
            and artifact_parser_hash != binding.parser_hash
        ):
            raise _RecoverySkip("artifact_parser_hash_mismatch")

    def _validate_parse_source_reference(
        self,
        binding: PipelineArtifactBinding,
        document: DocumentRecord,
    ) -> None:
        if normalize_artifact_checksum(binding.source_hash) is None:
            raise _RecoverySkip("source_checksum_invalid")
        source_object_uri = document.metadata.get("source_object_uri")
        if not isinstance(source_object_uri, str) or not source_object_uri:
            raise _RecoverySkip("source_object_reference_missing")
        try:
            self._object_storage.validate_document_file_uri(
                source_object_uri,
                workspace=binding.workspace,
                document_id=binding.document_id,
                namespace="source",
                artifact_id=None,
            )
        except Exception as exc:
            raise _RecoverySkip("source_object_ownership_mismatch") from exc

    @staticmethod
    def _validate_build_pointer_generation(
        artifact: ArtifactRecord,
        *,
        expected_artifact_id: str | None,
        claim_token: str,
    ) -> None:
        if artifact.id == expected_artifact_id:
            return
        if artifact.metadata.get("build_generation_id") != claim_token:
            raise _RecoverySkip("build_artifact_generation_mismatch")

    @staticmethod
    def _validate_document_counts(document: DocumentRecord) -> None:
        for field_name in ("chunks_count", "entity_count", "relation_count"):
            value = getattr(document, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise _RecoverySkip("document_count_invalid")

    @staticmethod
    def _committed_full_doc_payload(
        existing: Mapping[str, Any],
        committed_binding: PipelineArtifactBinding,
    ) -> dict[str, Any]:
        file_path = canonicalize_pipeline_logical_filename(existing.get("file_path"))
        payload: dict[str, Any] = {
            "content": existing.get("content", ""),
            "file_path": file_path,
            "parse_format": existing.get(
                "parse_format",
                FULL_DOCS_FORMAT_LIGHTRAG,
            ),
            "artifact_binding": committed_binding.to_dict(),
        }
        for key in (
            "parse_engine",
            "process_options",
            "chunk_options",
            "content_hash",
            "update_time",
        ):
            value = existing.get(key)
            if value is not None:
                payload[key] = value
        return payload

    def _processed_status_payload(
        self,
        full_doc: Mapping[str, Any],
        existing: Mapping[str, Any] | None,
        committed_binding: PipelineArtifactBinding,
        document: DocumentRecord,
    ) -> dict[str, Any]:
        existing = existing or {}
        body = strip_lightrag_doc_prefix(
            str(full_doc.get("content") or ""),
            str(full_doc.get("parse_format") or ""),
        )
        now = datetime.now(timezone.utc).isoformat()
        created_at = existing.get("created_at")
        if not isinstance(created_at, str) or not created_at:
            created_at = now
        summary = existing.get("content_summary")
        if not isinstance(summary, str):
            summary = get_content_summary(body)
        content_length = existing.get("content_length")
        if isinstance(content_length, bool) or not isinstance(content_length, int):
            content_length = len(body)

        metadata = doc_status_transition_metadata(existing)
        metadata["pipeline_attempt_token"] = committed_binding.claim_token
        metadata["artifact_binding"] = committed_binding.to_dict()
        payload: dict[str, Any] = {
            "status": DocStatus.PROCESSED,
            "content_summary": summary,
            "content_length": content_length,
            "created_at": created_at,
            "updated_at": now,
            "file_path": canonicalize_pipeline_logical_filename(
                full_doc.get("file_path")
            ),
            "track_id": existing.get("track_id"),
            "metadata": metadata,
        }
        content_hash = existing.get("content_hash", full_doc.get("content_hash"))
        if content_hash is not None:
            payload["content_hash"] = content_hash
        chunks_list = existing.get("chunks_list")
        if isinstance(chunks_list, list):
            payload["chunks_list"] = list(chunks_list)

        if committed_binding.operation == "build":
            for field_name in ("chunks_count", "entity_count", "relation_count"):
                value = getattr(document, field_name)
                payload[field_name] = value
                metadata[field_name] = value
        else:
            for field_name in ("chunks_count", "entity_count", "relation_count"):
                value = existing.get(field_name)
                if value is None:
                    continue
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    continue
                payload[field_name] = value
        return payload

    @classmethod
    def _status_is_exact(
        cls,
        status_row: Any,
        committed_binding: PipelineArtifactBinding,
        document: DocumentRecord,
    ) -> bool:
        if not isinstance(status_row, Mapping):
            return False
        try:
            assert_no_runtime_artifact_payload(
                status_row,
                context="pipeline artifact recovery status read-back",
            )
        except ValueError:
            return False
        if status_row.get("status") != DocStatus.PROCESSED:
            return False
        metadata = status_row.get("metadata")
        if not isinstance(metadata, Mapping):
            return False
        if metadata.get("pipeline_attempt_token") != committed_binding.claim_token:
            return False
        raw_binding = metadata.get("artifact_binding")
        if not isinstance(raw_binding, Mapping):
            return False
        try:
            status_binding = PipelineArtifactBinding.from_mapping(
                raw_binding,
                expected_workspace=committed_binding.workspace,
            )
        except (TypeError, ValueError):
            return False
        if status_binding != committed_binding:
            return False
        if committed_binding.operation == "build":
            for field_name in ("chunks_count", "entity_count", "relation_count"):
                expected = getattr(document, field_name)
                actual = status_row.get(field_name, metadata.get(field_name))
                if actual != expected:
                    return False
        return True

    @staticmethod
    def _safe_error(
        candidate: _Candidate,
        failure: _RecoveryFailure,
    ) -> PipelineArtifactRecoveryError:
        cause_name = type(failure.cause).__name__
        return PipelineArtifactRecoveryError(
            stage=failure.stage,
            error_code=failure.error_code,
            message=f"{failure.stage} failed ({cause_name})",
            document_id=candidate.document_id,
            lightrag_doc_id=candidate.lightrag_doc_id,
        )

    @staticmethod
    def _safe_discovery_error(
        *,
        stage: str,
        code: str,
        cause: BaseException,
        document_id: str | None = None,
        lightrag_doc_id: str | None = None,
    ) -> PipelineArtifactRecoveryError:
        return PipelineArtifactRecoveryError(
            stage=stage,
            error_code=code,
            message=f"{stage} failed ({type(cause).__name__})",
            document_id=document_id,
            lightrag_doc_id=lightrag_doc_id,
        )
