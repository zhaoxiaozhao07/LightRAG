from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from lightrag.api.artifact_lifecycle import (
    ARTIFACT_LIFECYCLE_MAX_PAGE_SIZE,
    ArtifactCleanupManifestRecord,
    ArtifactCleanupStatus,
    ArtifactLifecycleStateError,
    artifact_cleanup_idempotency_key,
)
from lightrag.api.config import (
    OBJECT_AUTHORITATIVE_LIFECYCLE_IMPLEMENTED,
    ArtifactCleanupConfig,
)
from lightrag.api.kb_service import (
    KnowledgeBaseConflictError,
    KnowledgeBaseNotFoundError,
    KnowledgeBaseRecord,
    KnowledgeBaseService,
    utc_now_iso,
)
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry
from lightrag.api.metadata_store import (
    InvalidJobTransitionError,
    JobRecord,
    KBLifecycleConflictError,
    MetadataRecordNotFoundError,
    SQLiteMetadataStore,
)
from lightrag.api.object_storage import ObjectStorage
from lightrag.api.postgres_metadata_store import PostgresMetadataStore
from lightrag.utils import generate_track_id, logger

MetadataStore = SQLiteMetadataStore | PostgresMetadataStore

_CLEAR_PAYLOAD_KEYS = frozenset(
    {"kb_generation", "workspace", "idempotency_fingerprint"}
)
_CLEAR_PHYSICAL_STAGE = "deleting"
_CLEAR_FINALIZING_STAGE = "finalizing"
_CLEAR_DRAINING_STAGE = "draining"

# Manifest statuses that count as "still being processed" by the cleanup
# service. The drain must wait for these to leave this set before the empty
# listing proof is collected.
_OBJECT_DRAIN_PENDING_STATUSES: tuple[ArtifactCleanupStatus, ...] = (
    "retained",
    "pending",
    "leased",
)
_OBJECT_DRAIN_BLOCKED_STATUSES: tuple[ArtifactCleanupStatus, ...] = ("blocked",)
_OBJECT_DRAIN_LISTING_PAGE_SIZE = 1000

DrainOutcome = Literal["empty", "pending", "blocked"]


@dataclass(slots=True)
class KBHardDeleteResult:
    job: JobRecord
    purged_rows: dict[str, int] = field(default_factory=dict)
    cleared_input_dir: bool = False
    cleared_object_storage: bool = False
    deleted_objects: int = 0
    dropped_storages: int = 0
    finalized_storages: bool = False
    purged_catalog: bool = False
    object_cleanup_pending: bool = False
    errors: list[str] = field(default_factory=list)


class KBHardDeleteInProgressError(RuntimeError):
    """Raised when a synchronous caller finds the generation's job running."""

    def __init__(self, job: JobRecord):
        self.job = job
        super().__init__(
            f"Knowledge base hard delete job '{job.id}' is already in progress"
        )


class KBHardDeleteUnsupportedError(RuntimeError):
    pass


def _hard_delete_capability_enabled() -> bool:
    """Return the object-authoritative hard-delete capability constant.

    Thin indirection so the hard-delete gate can be exercised in tests without
    mutating the frozen ``OBJECT_AUTHORITATIVE_LIFECYCLE_IMPLEMENTED``
    constant. The constant stays ``False`` until Gate 3; this helper is the
    only place the deletion service reads it, mirroring the
    ``_object_lifecycle_capability_enabled`` pattern in ``kb_document_routes``.
    """

    return OBJECT_AUTHORITATIVE_LIFECYCLE_IMPLEMENTED


class KBDeletionService:
    """Generation-pinned, restart-resumable hard deletion for a knowledge base.

    A single ``clear_kb`` job is identified by ``(kb_id, generation)``. Its
    payload pins both the generation and workspace, and every destructive step
    runs while the metadata store's exclusive KB deletion fence is held.
    Physical cleanup is retried idempotently, but metadata/catalog commit is
    attempted only after every physical step succeeds.
    """

    def __init__(
        self,
        kb_service: KnowledgeBaseService,
        metadata_store: MetadataStore,
        registry: LightRAGInstanceRegistry,
        *,
        input_root: Path,
        working_dir: Path | None = None,
        object_storage: ObjectStorage | None = None,
        artifact_storage_mode: str = "local",
        artifact_cleanup_config: ArtifactCleanupConfig | None = None,
    ):
        normalized_storage_mode = str(artifact_storage_mode or "local").strip().lower()
        if normalized_storage_mode not in {"local", "object"}:
            raise ValueError("artifact_storage_mode must be local or object")
        self._kb_service = kb_service
        self._metadata_store = metadata_store
        self._registry = registry
        self._input_root = Path(input_root)
        self._working_dir = Path(working_dir) if working_dir else None
        self._object_storage = object_storage
        self._artifact_storage_mode = normalized_storage_mode
        self._artifact_cleanup_config = (
            artifact_cleanup_config or ArtifactCleanupConfig()
        )

    def assert_hard_delete_supported(self) -> None:
        if self._artifact_storage_mode == "object" and not (
            _hard_delete_capability_enabled()
        ):
            raise KBHardDeleteUnsupportedError(
                "Knowledge base hard delete is disabled in object artifact mode "
                "until the OBJECT_AUTHORITATIVE_LIFECYCLE_IMPLEMENTED capability "
                "constant becomes True at Phase 3 Gate 3"
            )

    async def hard_delete(
        self, kb_id: str, *, expected_generation: str
    ) -> KBHardDeleteResult:
        """Run the generation's one durable clear job in this process.

        Failed/cancelled jobs are reset in place so retries preserve both the
        job id and idempotency key. A running job is never executed a second
        time concurrently.
        """

        self.assert_hard_delete_supported()

        job = await self._get_or_create_clear_job(
            kb_id,
            expected_generation=expected_generation,
            requeue_terminal=False,
        )
        async with self._metadata_store.job_execution_guard(
            job.id,
            wait=False,
        ) as acquired:
            if not acquired:
                current = await self._metadata_store.get_job(job.kb_id, job.id)
                if current.status == "succeeded":
                    return self._result_from_job(current)
                raise KBHardDeleteInProgressError(current)

            # A durable ``running`` row with no live execution owner is the
            # normal residue of an in-process cancellation or crash. Ownership
            # is the concurrency authority, so resume that same row directly.
            for _attempt in range(4):
                job = await self._metadata_store.get_job(job.kb_id, job.id)
                if job.status == "succeeded":
                    return self._result_from_job(job)
                try:
                    if job.status == "cancelling":
                        job = await self._metadata_store.transition_job(
                            job.kb_id,
                            job.id,
                            status="cancelled",
                        )
                    if job.status in {"failed", "cancelled"}:
                        job = await self._metadata_store.reset_job_for_retry(
                            job.kb_id,
                            job.id,
                            new_idempotency_key=None,
                        )
                    if job.status in {"queued", "retrying"}:
                        job = await self._metadata_store.transition_job(
                            job.kb_id,
                            job.id,
                            status="running",
                            progress=max(job.progress, 0.1),
                        )
                    if job.status == "running":
                        break
                except InvalidJobTransitionError:
                    continue
                raise InvalidJobTransitionError(
                    f"Cannot synchronously run clear job '{job.id}' from {job.status}"
                )
            else:
                current = await self._metadata_store.get_job(job.kb_id, job.id)
                raise InvalidJobTransitionError(
                    f"Could not stabilize clear job '{job.id}' from {current.status}"
                )
            # ``resume_hard_delete`` intentionally re-enters the same guard.
            # SQLite tracks task-local depth and PostgreSQL balances a second
            # advisory acquire/unlock on the same context-bound session.
            return await self.resume_hard_delete(job)

    async def enqueue_hard_delete(
        self, kb_id: str, *, expected_generation: str
    ) -> JobRecord:
        """Return the generation's one runnable clear job.

        A failed/cancelled generation-pinned job is reset in place. Concurrent
        enqueue calls race through the metadata store's atomic reset, so only
        one increments ``retry_count`` while the others observe the resulting
        queued/running row.
        """

        self.assert_hard_delete_supported()

        return await self._get_or_create_clear_job(
            kb_id, expected_generation=expected_generation
        )

    async def resume_hard_delete(self, job: JobRecord) -> KBHardDeleteResult:
        """Resume a persisted ``clear_kb`` job using only its pinned payload.

        The catalog is never used to infer or upgrade the generation/workspace
        carried by a legacy job. Invalid or stale jobs fail closed before any
        registry, filesystem, object-storage, metadata-purge, or catalog-purge
        side effect.
        """

        self.assert_hard_delete_supported()

        async with self._metadata_store.job_execution_guard(job.id) as acquired:
            if not acquired:  # wait=True always acquires; retained for parity.
                return self._result_from_job(job)
            return await self._resume_hard_delete_owned(job)

    async def _resume_hard_delete_owned(
        self, job: JobRecord
    ) -> KBHardDeleteResult:
        """Resume after the caller owns ``job.id`` for the complete execution."""

        try:
            current = await self._metadata_store.get_job(job.kb_id, job.id)
        except MetadataRecordNotFoundError:
            result = KBHardDeleteResult(job=job)
            result.errors.append("clear_kb job is not persisted")
            return result

        if current.status == "succeeded":
            return self._result_from_job(current)
        if current.status == "queued":
            try:
                current = await self._metadata_store.transition_job(
                    current.kb_id,
                    current.id,
                    status="running",
                    progress=max(current.progress, 0.1),
                )
            except InvalidJobTransitionError:
                current = await self._metadata_store.get_job(
                    current.kb_id, current.id
                )
        if current.status != "running":
            return self._result_from_job(current)
        return await self._execute_clear(current)

    async def _get_or_create_clear_job(
        self,
        kb_id: str,
        *,
        expected_generation: str,
        requeue_terminal: bool = True,
    ) -> JobRecord:
        if not isinstance(expected_generation, str) or not expected_generation.strip():
            raise ValueError("expected_generation must be a non-empty string")

        idempotency_key = self._idempotency_key(kb_id, expected_generation)
        existing = await self._metadata_store.get_job_by_idempotency_key(
            kb_id,
            idempotency_key,
            job_type="clear_kb",
        )
        if existing is not None:
            return await self._resolve_existing_clear_job(
                existing,
                expected_generation=expected_generation,
                requeue_terminal=requeue_terminal,
            )

        # Accommodate an already-created generation-pinned row whose key was
        # lost by an older retry implementation, without creating a second job.
        # An unfinished legacy clear with no generation is ambiguous, so return
        # that same row and let resume fail it closed rather than inventing a
        # second destructive identity.
        legacy_clear: JobRecord | None = None
        offset = 0
        while True:
            jobs, total = await self._metadata_store.list_jobs(
                kb_id,
                limit=200,
                offset=offset,
            )
            for candidate in jobs:
                if candidate.job_type != "clear_kb":
                    continue
                candidate_generation = candidate.payload.get("kb_generation")
                if candidate_generation == expected_generation:
                    return await self._resolve_existing_clear_job(
                        candidate,
                        expected_generation=expected_generation,
                        requeue_terminal=requeue_terminal,
                    )
                if (
                    (
                        not isinstance(candidate_generation, str)
                        or not candidate_generation.strip()
                    )
                    and candidate.status != "succeeded"
                    and legacy_clear is None
                ):
                    legacy_clear = candidate
            offset += len(jobs)
            if not jobs or offset >= total:
                break
        if legacy_clear is not None:
            return legacy_clear

        record = await self._load_deleted_catalog_record(
            kb_id,
            expected_generation=expected_generation,
        )
        await self._ensure_lifecycle_generation(record)

        now = utc_now_iso()
        payload = self._clear_payload(record)
        candidate = JobRecord(
            id=generate_track_id("job_clear_kb"),
            kb_id=record.id,
            workspace=record.workspace,
            batch_id=None,
            document_id=None,
            job_type="clear_kb",
            status="queued",
            stage="deleting",
            progress=0.0,
            total_items=1,
            completed_items=0,
            failed_items=0,
            idempotency_key=idempotency_key,
            config_version_id=record.active_config_version_id,
            config_hash=None,
            retry_count=0,
            max_retries=3,
            payload=payload,
            result=None,
            error_code=None,
            error_message=None,
            created_at=now,
            updated_at=now,
            queued_at=now,
            started_at=None,
            finished_at=None,
            cancelled_at=None,
        )
        try:
            created, _was_created = await self._metadata_store.create_job_once(
                candidate
            )
            return await self._resolve_existing_clear_job(
                created,
                expected_generation=expected_generation,
                requeue_terminal=requeue_terminal,
            )
        except Exception:
            # PostgreSQL callers in separate processes can both miss the
            # pre-insert lookup. The unique idempotency index admits one row;
            # after the losing transaction rolls back, return that winner.
            concurrent = await self._metadata_store.get_job_by_idempotency_key(
                record.id,
                idempotency_key,
                job_type="clear_kb",
            )
            if concurrent is not None:
                return await self._resolve_existing_clear_job(
                    concurrent,
                    expected_generation=expected_generation,
                    requeue_terminal=requeue_terminal,
                )
            raise

    async def _resolve_existing_clear_job(
        self,
        job: JobRecord,
        *,
        expected_generation: str,
        requeue_terminal: bool,
    ) -> JobRecord:
        if not requeue_terminal:
            return job
        return await self._requeue_existing_clear_job(
            job,
            expected_generation=expected_generation,
        )

    async def _requeue_existing_clear_job(
        self,
        job: JobRecord,
        *,
        expected_generation: str,
    ) -> JobRecord:
        if job.payload.get("kb_generation") != expected_generation:
            return job
        if job.status not in {"failed", "cancelled"}:
            return job
        async with self._metadata_store.job_execution_guard(job.id) as acquired:
            if not acquired:
                return job
            current = await self._metadata_store.get_job(job.kb_id, job.id)
            if current.status not in {"failed", "cancelled"}:
                return current
            try:
                return await self._metadata_store.reset_job_for_retry(
                    current.kb_id,
                    current.id,
                    new_idempotency_key=None,
                )
            except InvalidJobTransitionError:
                current = await self._metadata_store.get_job(job.kb_id, job.id)
                if current.status in {
                    "queued",
                    "running",
                    "retrying",
                    "cancelling",
                    "succeeded",
                }:
                    return current
                raise

    async def _ensure_lifecycle_generation(
        self, record: KnowledgeBaseRecord
    ) -> None:
        lifecycle = await self._metadata_store.get_kb_lifecycle(record.id)
        if (
            lifecycle is not None
            and lifecycle.state == "active"
            and lifecycle.generation == record.generation
        ):
            return
        await self._metadata_store.activate_kb_generation(
            record.id,
            record.generation,
        )

    async def _execute_clear(self, job: JobRecord) -> KBHardDeleteResult:
        result = self._result_from_job(job)
        try:
            generation, workspace = self._validate_clear_job_identity(job)
        except Exception as exc:  # noqa: BLE001 - invalid durable payload fails closed
            result.errors.append(f"invalid_clear_payload: {exc}")
            return await self._fail_result(
                result,
                error_code="kb_hard_delete_invalid_payload",
            )

        try:
            record = await self._kb_service.get(job.kb_id, include_deleted=True)
        except KnowledgeBaseNotFoundError:
            return await self._resume_tail_without_catalog(
                job,
                generation=generation,
                workspace=workspace,
                result=result,
            )
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"kb_catalog_preflight: {exc}")
            return await self._fail_result(result)

        identity_error = self._catalog_identity_error(
            record,
            generation=generation,
            workspace=workspace,
        )
        if identity_error is not None:
            result.errors.append(identity_error)
            return await self._fail_result(
                result,
                error_code="kb_hard_delete_stale_identity",
            )

        try:
            async with self._metadata_store.kb_deletion_guard(job.kb_id):
                try:
                    guarded_record = await self._kb_service.get(
                        job.kb_id, include_deleted=True
                    )
                except KnowledgeBaseNotFoundError:
                    return await self._resume_tail_without_catalog_inside_guard(
                        job,
                        generation=generation,
                        workspace=workspace,
                        result=result,
                    )

                identity_error = self._catalog_identity_error(
                    guarded_record,
                    generation=generation,
                    workspace=workspace,
                )
                if identity_error is not None:
                    result.errors.append(identity_error)
                    return await self._fail_result(
                        result,
                        error_code="kb_hard_delete_stale_identity",
                    )

                # The catalog precondition is deliberately checked while the
                # exclusive operation lock is held and before lifecycle is
                # changed. If restore won the shared-lock race, this call is
                # never reached and lifecycle remains active.
                lifecycle = await self._metadata_store.begin_kb_deletion(
                    job.kb_id,
                    generation,
                    job.id,
                )
                lifecycle_error = self._lifecycle_identity_error(
                    lifecycle,
                    generation=generation,
                    delete_job_id=job.id,
                )
                if lifecycle_error is not None:
                    result.errors.append(lifecycle_error)
                    return await self._fail_result(
                        result,
                        error_code="kb_hard_delete_stale_identity",
                    )

                if lifecycle.state == "deleted":
                    return await self._purge_catalog_tail_inside_guard(
                        guarded_record,
                        generation=generation,
                        result=result,
                    )
                if lifecycle.state != "deleting":
                    result.errors.append(
                        "kb_lifecycle: physical cleanup requires the deleting state"
                    )
                    return await self._fail_result(
                        result,
                        error_code="kb_hard_delete_stale_identity",
                    )

                object_authoritative = self._object_authoritative()
                if object_authoritative and self._object_storage is None:
                    result.errors.append(
                        "object_storage: object-authoritative hard delete requires "
                        "an object storage backend"
                    )
                    return await self._fail_result(result)

                if job.stage == _CLEAR_PHYSICAL_STAGE:
                    async with self._registry.destructive_lock(guarded_record.id):
                        await self._run_physical_cleanup(guarded_record, result)

                    if result.errors:
                        # Never commit metadata/catalog deletion after an incomplete
                        # physical clear. The deleting lifecycle fence is retained.
                        return await self._fail_result(result)

                    if object_authoritative:
                        await self._enqueue_object_drain_manifests(
                            job,
                            record=guarded_record,
                            generation=generation,
                            workspace=workspace,
                            result=result,
                        )
                        if result.errors:
                            return await self._fail_result(result)

                        drain_outcome = await self._check_object_drain_status(
                            job,
                            generation=generation,
                            workspace=workspace,
                            result=result,
                        )
                        if drain_outcome == "blocked":
                            return await self._fail_result(
                                result,
                                error_code="kb_hard_delete_drain_blocked",
                            )
                        if drain_outcome == "pending":
                            # Checkpoint the draining stage and release the
                            # exclusive fence so the cleanup service may own
                            # the workspace-prefix work. The next resume call
                            # re-acquires the fence and re-checks the drain.
                            return await self._checkpoint_draining(result, job)

                    try:
                        # Leaving the draining stage clears the pending flag
                        # so the finalizing snapshot records a clean state.
                        result.object_cleanup_pending = False
                        result.job = await self._metadata_store.update_job_progress(
                            job.kb_id,
                            job.id,
                            stage=_CLEAR_FINALIZING_STAGE,
                            result_patch=self._result_payload(result),
                        )
                    except Exception as exc:  # noqa: BLE001
                        result.errors.append(f"clear_checkpoint: {exc}")
                        return await self._fail_result(result)
                elif job.stage == _CLEAR_DRAINING_STAGE:
                    if not object_authoritative:
                        result.errors.append(
                            f"invalid_clear_payload: unsupported clear stage "
                            f"{job.stage!r}"
                        )
                        return await self._fail_result(
                            result,
                            error_code="kb_hard_delete_invalid_payload",
                        )

                    drain_outcome = await self._check_object_drain_status(
                        job,
                        generation=generation,
                        workspace=workspace,
                        result=result,
                    )
                    if drain_outcome == "blocked":
                        return await self._fail_result(
                            result,
                            error_code="kb_hard_delete_drain_blocked",
                        )
                    if drain_outcome == "pending":
                        return await self._checkpoint_draining(result, job)

                    try:
                        # Leaving the draining stage clears the pending flag
                        # so the finalizing snapshot records a clean state.
                        result.object_cleanup_pending = False
                        result.job = await self._metadata_store.update_job_progress(
                            job.kb_id,
                            job.id,
                            stage=_CLEAR_FINALIZING_STAGE,
                            result_patch=self._result_payload(result),
                        )
                    except Exception as exc:  # noqa: BLE001
                        result.errors.append(f"clear_checkpoint: {exc}")
                        return await self._fail_result(result)
                elif job.stage != _CLEAR_FINALIZING_STAGE:
                    result.errors.append(
                        f"invalid_clear_payload: unsupported clear stage {job.stage!r}"
                    )
                    return await self._fail_result(
                        result,
                        error_code="kb_hard_delete_invalid_payload",
                    )

                if object_authoritative:
                    # The verified-empty proof was collected while the fence
                    # was held. The recovery cursor is harmless residue once
                    # the metadata purge runs, so its absence is not fatal.
                    await self._delete_recovery_cursor_quiet(
                        job.kb_id, kb_generation=generation
                    )

                try:
                    result.purged_rows = (
                        await self._metadata_store.purge_kb_metadata(
                            guarded_record.id,
                            generation=generation,
                            delete_job_id=job.id,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    result.errors.append(f"metadata_purge: {exc}")
                    return await self._fail_result(result)

                try:
                    completed_lifecycle = (
                        await self._metadata_store.complete_kb_deletion(
                            guarded_record.id,
                            generation,
                            job.id,
                        )
                    )
                    lifecycle_error = self._lifecycle_identity_error(
                        completed_lifecycle,
                        generation=generation,
                        delete_job_id=job.id,
                    )
                    if (
                        lifecycle_error is not None
                        or completed_lifecycle.state != "deleted"
                    ):
                        result.errors.append(
                            lifecycle_error
                            or "complete_kb_deletion: lifecycle did not reach deleted"
                        )
                        return await self._fail_result(result)
                except Exception as exc:  # noqa: BLE001
                    result.errors.append(f"complete_kb_deletion: {exc}")
                    return await self._fail_result(result)

                return await self._purge_catalog_tail_inside_guard(
                    guarded_record,
                    generation=generation,
                    result=result,
                )
        except (KBLifecycleConflictError, KnowledgeBaseConflictError) as exc:
            result.errors.append(f"kb_lifecycle: {exc}")
            return await self._fail_result(
                result,
                error_code="kb_hard_delete_stale_identity",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Hard delete failed for KB '%s': %s", job.kb_id, exc)
            result.errors.append(str(exc))
            return await self._fail_result(result)

    async def _resume_tail_without_catalog(
        self,
        job: JobRecord,
        *,
        generation: str,
        workspace: str,
        result: KBHardDeleteResult,
    ) -> KBHardDeleteResult:
        """Finish only the post-lifecycle catalog/job tail.

        With the ordered commit protocol, a missing catalog row is valid only
        after the exact lifecycle has reached ``deleted``. A ``deleting`` row
        with no catalog is a legacy/inconsistent state and fails closed instead
        of guessing that physical cleanup or metadata purge already happened.
        """

        try:
            async with self._metadata_store.kb_deletion_guard(job.kb_id):
                return await self._resume_tail_without_catalog_inside_guard(
                    job,
                    generation=generation,
                    workspace=workspace,
                    result=result,
                )
        except (KBLifecycleConflictError, KnowledgeBaseConflictError) as exc:
            result.errors.append(f"kb_lifecycle: {exc}")
            return await self._fail_result(
                result,
                error_code="kb_hard_delete_stale_identity",
            )
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"kb_catalog_tail: {exc}")
            return await self._fail_result(result)

    async def _resume_tail_without_catalog_inside_guard(
        self,
        job: JobRecord,
        *,
        generation: str,
        workspace: str,
        result: KBHardDeleteResult,
    ) -> KBHardDeleteResult:
        """Finish a deleted lifecycle's catalog/job tail with exclusion held."""

        guarded_lifecycle = await self._metadata_store.get_kb_lifecycle(job.kb_id)
        lifecycle_error = (
            None
            if guarded_lifecycle is None
            else self._lifecycle_identity_error(
                guarded_lifecycle,
                generation=generation,
                delete_job_id=job.id,
            )
        )
        if (
            guarded_lifecycle is None
            or lifecycle_error is not None
            or guarded_lifecycle.state != "deleted"
        ):
            result.errors.append(
                lifecycle_error
                or "kb_catalog_preflight: catalog row is missing before "
                "lifecycle reached deleted"
            )
            return await self._fail_result(
                result,
                error_code="kb_hard_delete_stale_identity",
            )

        try:
            current = await self._kb_service.get(job.kb_id, include_deleted=True)
        except KnowledgeBaseNotFoundError:
            current = None
        if current is None:
            result.purged_catalog = True
            return await self._succeed_result(result)
        identity_error = self._catalog_identity_error(
            current,
            generation=generation,
            workspace=workspace,
        )
        if identity_error is not None:
            result.errors.append(identity_error)
            return await self._fail_result(
                result,
                error_code="kb_hard_delete_stale_identity",
            )
        return await self._purge_catalog_tail_inside_guard(
            current,
            generation=generation,
            result=result,
        )

    async def _purge_catalog_tail_inside_guard(
        self,
        record: KnowledgeBaseRecord,
        *,
        generation: str,
        result: KBHardDeleteResult,
    ) -> KBHardDeleteResult:
        try:
            await self._kb_service.purge(
                record.id,
                expected_generation=generation,
                expected_status="deleted",
            )
            # False means the exact row was already absent. Under the deleted
            # lifecycle fence that is an idempotent completed step.
            result.purged_catalog = True
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"kb_catalog_purge: {exc}")
            return await self._fail_result(result)
        return await self._succeed_result(result)

    @staticmethod
    def _lifecycle_identity_error(
        lifecycle: object,
        *,
        generation: str,
        delete_job_id: str,
    ) -> str | None:
        if getattr(lifecycle, "generation", None) != generation:
            return "kb_lifecycle: lifecycle generation changed"
        if getattr(lifecycle, "delete_job_id", None) != delete_job_id:
            return "kb_lifecycle: lifecycle delete job changed"
        return None

    async def _run_physical_cleanup(
        self,
        record: KnowledgeBaseRecord,
        result: KBHardDeleteResult,
    ) -> None:
        try:
            await self._registry.force_evict(record.id)
            result.finalized_storages = True
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"force_evict: {exc}")

        try:
            drop_summary = await self._registry.drop_kb_data(record)
            result.dropped_storages = int(drop_summary.get("dropped", 0))
            drop_errors = list(drop_summary.get("errors", []))
            if int(drop_summary.get("failed", 0)) and not drop_errors:
                drop_errors.append("one or more storage drops failed")
            result.errors.extend(f"drop_storages: {error}" for error in drop_errors)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"drop_storages: {exc}")

        if self._working_dir is not None:
            workspace_dir = (self._working_dir / record.workspace).resolve()
            self._safe_rmtree(workspace_dir, result, label="working_dir")

        input_workspace = (self._input_root / record.workspace).resolve()
        if input_workspace.exists():
            result.cleared_input_dir = self._safe_rmtree(
                input_workspace,
                result,
                label="input_dir",
            )

        if self._object_storage is not None and not self._object_authoritative():
            # Local artifact-storage mode keeps the legacy unvalidated bulk
            # deletion; the workspace owns no durable authority that a manifest
            # drain must respect. Object-authoritative mode replaces this with
            # the manifest-driven drain in :meth:`_execute_clear`.
            try:
                result.deleted_objects = await self._object_storage.delete_workspace(
                    record.workspace
                )
                result.cleared_object_storage = True
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"object_storage: {exc}")

    def _object_authoritative(self) -> bool:
        """Object-authoritative mode drives a manifest-driven workspace drain.

        ``assert_hard_delete_supported`` rejects object mode at the route
        boundary (HTTP 503) only while
        ``OBJECT_AUTHORITATIVE_LIFECYCLE_IMPLEMENTED`` is still ``False``;
        once that capability constant flips True the gate opens and this
        branch runs in production. Direct service callers and tests exercise
        the drain path via the ``_hard_delete_capability_enabled``
        indirection rather than bypassing the gate.
        """

        return self._artifact_storage_mode == "object"

    @staticmethod
    def _kb_delete_manifest_group_id(
        *, kb_id: str, kb_generation: str, workspace: str, job_id: str
    ) -> str:
        canonical = json.dumps(
            {
                "version": "kb-delete-manifest-group:v1",
                "kb_id": kb_id,
                "kb_generation": kb_generation,
                "workspace": workspace,
                "job_id": job_id,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return f"kbmg_{hashlib.sha256(canonical).hexdigest()}"

    def _workspace_prefix_uri(self, workspace: str) -> str:
        if self._object_storage is None:
            raise RuntimeError(
                "object-authoritative drain requires an object storage backend"
            )
        return self._object_storage.object_prefix_uri_for_key(
            f"workspaces/{workspace}"
        )

    def _build_kb_delete_manifest(
        self,
        *,
        job: JobRecord,
        record: KnowledgeBaseRecord,
        generation: str,
        workspace: str,
    ) -> ArtifactCleanupManifestRecord:
        target_uri = self._workspace_prefix_uri(workspace)
        group_id = self._kb_delete_manifest_group_id(
            kb_id=record.id,
            kb_generation=generation,
            workspace=workspace,
            job_id=job.id,
        )
        idempotency_key = artifact_cleanup_idempotency_key(
            reason="kb_delete",
            kb_id=record.id,
            kb_generation=generation,
            workspace=workspace,
            target_kind="prefix",
            target_namespace="workspace",
            target_uri=target_uri,
        )
        now = datetime.now(timezone.utc)
        delete_after = now
        cleanup_deadline_at = now + timedelta(
            seconds=self._artifact_cleanup_config.cleanup_slo_seconds
        )
        audit_retain_until = now + timedelta(
            days=self._artifact_cleanup_config.successful_audit_retention_days
        )
        return ArtifactCleanupManifestRecord(
            id=generate_track_id("kb_delete_manifest"),
            idempotency_key=idempotency_key,
            manifest_group_id=group_id,
            kb_id=record.id,
            kb_generation=generation,
            workspace=workspace,
            document_id=None,
            artifact_id=None,
            source_generation_id=None,
            origin_job_id=job.id,
            origin_attempt_token=None,
            reason="kb_delete",
            target_kind="prefix",
            target_namespace="workspace",
            disposition="delete",
            status="pending",
            target_uri=target_uri,
            delete_after=delete_after,
            cleanup_deadline_at=cleanup_deadline_at,
            audit_retain_until=audit_retain_until,
            next_attempt_at=now,
            created_at=now,
            updated_at=now,
        )

    async def _enqueue_object_drain_manifests(
        self,
        job: JobRecord,
        *,
        record: KnowledgeBaseRecord,
        generation: str,
        workspace: str,
        result: KBHardDeleteResult,
    ) -> None:
        """Enqueue the one workspace-prefix manifest and release prior retains.

        The manifest's idempotency key is deterministic in ``(kb_id,
        kb_generation, workspace, target_uri)``, so retries replay the same
        durable row instead of producing duplicate work. ``origin_job_id`` is
        the hard-delete job id, which ``begin_kb_deletion`` pinned as
        ``lifecycle.delete_job_id`` so the cleanup service admits the manifest.
        Any retained manifests for the same KB generation (left over from
        earlier document mutations) are released so they may drain alongside
        the workspace-prefix authority while lifecycle is still ``deleting``.
        """

        manifest = self._build_kb_delete_manifest(
            job=job,
            record=record,
            generation=generation,
            workspace=workspace,
        )
        try:
            await self._metadata_store.enqueue_artifact_cleanup_manifest(manifest)
        except ArtifactLifecycleStateError as exc:
            result.errors.append(f"object_drain_enqueue: {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"object_drain_enqueue: {exc}")
            return

        await self._release_retained_manifests_for_generation(
            record.id,
            generation=generation,
            current_manifest=manifest,
            result=result,
        )

    async def _release_retained_manifests_for_generation(
        self,
        kb_id: str,
        *,
        generation: str,
        current_manifest: ArtifactCleanupManifestRecord,
        result: KBHardDeleteResult,
    ) -> None:
        """Release every retained manifest still held for this KB generation.

        The just-enqueued workspace-prefix manifest is ``pending`` (not
        ``retained``), so this call is a no-op for it. Older retained manifests
        from prior document mutations are grouped by their ``manifest_group_id``
        and released so the cleanup service may drain them under the current
        ``deleting`` lifecycle.
        """

        try:
            retained, total = await self._metadata_store.list_artifact_cleanup_manifests(
                kb_id=kb_id,
                kb_generation=generation,
                status="retained",
                limit=ARTIFACT_LIFECYCLE_MAX_PAGE_SIZE,
            )
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"object_drain_release_lookup: {exc}")
            return
        if total < len(retained):
            result.errors.append("object_drain_release_lookup: retained overflow")
            return
        grouped: dict[str, list[str]] = {}
        for manifest in retained:
            grouped.setdefault(manifest.manifest_group_id, []).append(manifest.id)
        for group_id, manifest_ids in grouped.items():
            try:
                await self._metadata_store.release_retained_artifact_cleanup_manifests(
                    kb_id,
                    generation,
                    group_id,
                    manifest_ids,
                )
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"object_drain_release: {exc}")
                return

    async def _check_object_drain_status(
        self,
        job: JobRecord,
        *,
        generation: str,
        workspace: str,
        result: KBHardDeleteResult,
    ) -> DrainOutcome:
        """Return whether the workspace-prefix drain is complete.

        - ``blocked``: at least one manifest for this generation is permanently
          blocked; the hard-delete job fails closed so an operator inspects it.
        - ``pending``: at least one manifest is still being processed, or the
          post-cleanup listing still observes objects. The drain checkpoints
          ``draining`` and re-acquires the fence on the next resume.
        - ``empty``: zero pending/blocked manifests and one ``list_objects_page``
          call returns zero entries. The drain is verified empty.
        """

        try:
            blocked_count = (
                await self._metadata_store.count_artifact_cleanup_manifests(
                    kb_id=job.kb_id,
                    kb_generation=generation,
                    statuses=_OBJECT_DRAIN_BLOCKED_STATUSES,
                )
            )
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"object_drain_status: {exc}")
            return "blocked"
        if blocked_count > 0:
            result.errors.append(
                f"object_drain: {blocked_count} blocked manifest(s) for generation"
            )
            return "blocked"

        try:
            pending_count = (
                await self._metadata_store.count_artifact_cleanup_manifests(
                    kb_id=job.kb_id,
                    kb_generation=generation,
                    statuses=_OBJECT_DRAIN_PENDING_STATUSES,
                )
            )
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"object_drain_status: {exc}")
            return "blocked"
        if pending_count > 0:
            return "pending"

        if not await self._verify_object_drain_empty(workspace, result=result):
            return "pending"
        return "empty"

    async def _verify_object_drain_empty(
        self,
        workspace: str,
        *,
        result: KBHardDeleteResult,
    ) -> bool:
        """One bounded ``list_objects_page`` returning zero entries.

        Called only after the cleanup service reports zero pending manifests
        for the KB generation. Any non-empty page means the workspace still
        holds objects and the drain must retry.
        """

        if self._object_storage is None:
            result.errors.append("object_drain_proof: object storage is not configured")
            return False
        prefix_uri = self._workspace_prefix_uri(workspace)
        try:
            page = await self._object_storage.list_objects_page(
                prefix_uri,
                max_keys=_OBJECT_DRAIN_LISTING_PAGE_SIZE,
            )
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"object_drain_proof: {exc}")
            return False
        if page.entries:
            return False
        result.cleared_object_storage = True
        return True

    async def _checkpoint_draining(
        self, result: KBHardDeleteResult, job: JobRecord
    ) -> KBHardDeleteResult:
        """Persist the ``draining`` checkpoint and return without terminating.

        The job remains ``running`` with stage ``draining`` and
        ``object_cleanup_pending=True`` in its result snapshot. The exclusive
        fence is released as the call returns; the next resume re-acquires it.
        """

        result.object_cleanup_pending = True
        try:
            result.job = await self._metadata_store.update_job_progress(
                job.kb_id,
                job.id,
                stage=_CLEAR_DRAINING_STAGE,
                result_patch=self._result_payload(result),
            )
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"clear_checkpoint: {exc}")
            result.object_cleanup_pending = False
            return await self._fail_result(result)
        return result

    async def _delete_recovery_cursor_quiet(
        self, kb_id: str, *, kb_generation: str
    ) -> None:
        """Best-effort recovery-cursor removal before the metadata purge.

        ``delete_artifact_recovery_cursor`` is the additive Writer-A entry
        point. If a slightly older store does not yet expose it, the cursor
        becomes harmless residue that ``purge_kb_metadata`` does not touch.
        Never propagate an error from this call into the hard-delete result.
        """

        method = getattr(self._metadata_store, "delete_artifact_recovery_cursor", None)
        if method is None:
            return
        try:
            await method(kb_id, kb_generation)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Recovery cursor removal failed for KB '%s' (harmless residue): %s",
                kb_id,
                exc,
            )

    async def _load_deleted_catalog_record(
        self,
        kb_id: str,
        *,
        expected_generation: str,
    ) -> KnowledgeBaseRecord:
        record = await self._kb_service.get(kb_id, include_deleted=True)
        if record.generation != expected_generation:
            raise KnowledgeBaseConflictError(
                f"Knowledge base '{record.id}' changed generation"
            )
        if record.status != "deleted":
            raise KnowledgeBaseConflictError(
                f"Knowledge base '{record.id}' must be deleted before hard delete"
            )
        return record

    @staticmethod
    def _catalog_identity_error(
        record: KnowledgeBaseRecord,
        *,
        generation: str,
        workspace: str,
    ) -> str | None:
        if record.generation != generation:
            return "kb_catalog_preflight: catalog generation changed"
        if record.workspace != workspace:
            return "kb_catalog_preflight: catalog workspace changed"
        if record.status != "deleted":
            return "kb_catalog_preflight: catalog status is not deleted"
        return None

    @classmethod
    def _clear_payload(cls, record: KnowledgeBaseRecord) -> dict[str, str]:
        return {
            "kb_generation": record.generation,
            "workspace": record.workspace,
            "idempotency_fingerprint": cls._fingerprint(
                record.id,
                record.generation,
                record.workspace,
            ),
        }

    @classmethod
    def _validate_clear_job_identity(cls, job: JobRecord) -> tuple[str, str]:
        if job.job_type != "clear_kb":
            raise ValueError("job_type must be clear_kb")
        if set(job.payload) != _CLEAR_PAYLOAD_KEYS:
            raise ValueError("clear_kb payload must contain only pinned identity fields")
        generation = job.payload.get("kb_generation")
        workspace = job.payload.get("workspace")
        fingerprint = job.payload.get("idempotency_fingerprint")
        if not isinstance(generation, str) or not generation.strip():
            raise ValueError("kb_generation is required")
        if not isinstance(workspace, str) or not workspace.strip():
            raise ValueError("workspace is required")
        if workspace != job.workspace:
            raise ValueError("payload workspace does not match job workspace")
        expected_key = cls._idempotency_key(job.kb_id, generation)
        if job.idempotency_key != expected_key:
            raise ValueError("clear_kb idempotency key does not match generation")
        expected_fingerprint = cls._fingerprint(job.kb_id, generation, workspace)
        if fingerprint != expected_fingerprint:
            raise ValueError("clear_kb idempotency fingerprint does not match identity")
        return generation, workspace

    @staticmethod
    def _idempotency_key(kb_id: str, generation: str) -> str:
        return f"clear_kb:{kb_id}:{generation}"

    @staticmethod
    def _fingerprint(kb_id: str, generation: str, workspace: str) -> str:
        encoded = json.dumps(
            {
                "kb_generation": generation,
                "kb_id": kb_id,
                "workspace": workspace,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def _result_from_job(cls, job: JobRecord) -> KBHardDeleteResult:
        result = KBHardDeleteResult(job=job)
        snapshot = job.result or {}
        purged_rows = snapshot.get("purged_rows")
        if isinstance(purged_rows, dict):
            result.purged_rows = {
                str(key): int(value)
                for key, value in purged_rows.items()
                if isinstance(value, int)
            }
        for field_name in (
            "cleared_input_dir",
            "cleared_object_storage",
            "finalized_storages",
            "purged_catalog",
            "object_cleanup_pending",
        ):
            setattr(result, field_name, bool(snapshot.get(field_name, False)))
        for field_name in ("deleted_objects", "dropped_storages"):
            value = snapshot.get(field_name, 0)
            setattr(result, field_name, int(value) if isinstance(value, int) else 0)
        errors = snapshot.get("errors")
        if isinstance(errors, list):
            result.errors = [str(error) for error in errors]
        return result

    @staticmethod
    def _result_payload(result: KBHardDeleteResult) -> dict[str, object]:
        return {
            "purged_rows": result.purged_rows,
            "cleared_input_dir": result.cleared_input_dir,
            "cleared_object_storage": result.cleared_object_storage,
            "deleted_objects": result.deleted_objects,
            "dropped_storages": result.dropped_storages,
            "finalized_storages": result.finalized_storages,
            "purged_catalog": result.purged_catalog,
            "object_cleanup_pending": result.object_cleanup_pending,
            "errors": result.errors,
        }

    async def _fail_result(
        self,
        result: KBHardDeleteResult,
        *,
        error_code: str = "kb_hard_delete_failed",
    ) -> KBHardDeleteResult:
        try:
            current = await self._metadata_store.get_job(
                result.job.kb_id, result.job.id
            )
            if current.status in {"queued", "running", "retrying", "cancelling"}:
                result.job = await self._metadata_store.transition_job(
                    current.kb_id,
                    current.id,
                    status="failed",
                    progress=1.0,
                    completed_items=0,
                    failed_items=1,
                    result=self._result_payload(result),
                    error_code=error_code,
                    error_message="; ".join(result.errors),
                )
            else:
                result.job = current
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Could not mark hard delete job '%s' failed: %s",
                result.job.id,
                exc,
            )
            result.errors.append(f"job_transition_failed: {exc}")
        return result

    async def _succeed_result(
        self, result: KBHardDeleteResult
    ) -> KBHardDeleteResult:
        current = await self._metadata_store.get_job(result.job.kb_id, result.job.id)
        if current.status == "succeeded":
            result.job = current
            return result
        result.job = await self._metadata_store.transition_job(
            current.kb_id,
            current.id,
            status="succeeded",
            progress=1.0,
            completed_items=1,
            failed_items=0,
            result=self._result_payload(result),
            error_code=None,
            error_message=None,
        )
        return result

    @staticmethod
    def _safe_rmtree(
        path: Path, result: KBHardDeleteResult, *, label: str
    ) -> bool:
        try:
            if path.exists():
                shutil.rmtree(path)
            return True
        except OSError as exc:
            result.errors.append(f"{label}: {exc}")
            return False
