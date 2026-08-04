"""Leased, authority-revalidated cleanup for durable artifact manifests."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import re
from typing import Any, Literal

from lightrag.api.artifact_lifecycle import (
    ARTIFACT_LIFECYCLE_MAX_PAGE_SIZE,
    ArtifactCleanupManifestRecord,
    ArtifactLifecycleLeaseError,
    normalize_artifact_target_uri,
    sanitize_artifact_lifecycle_error_code,
)
from lightrag.api.config import ArtifactCleanupConfig
from lightrag.api.kb_service import sanitize_workspace
from lightrag.api.metadata_store import MetadataRecordNotFoundError
from lightrag.api.object_storage import (
    ArtifactCleanupTarget,
    DisabledObjectStorage,
    ObjectStorage,
    ObjectStorageProofError,
    VerifiedDeleteResult,
)

_ACTIVE_JOB_STATUSES = ("queued", "running", "retrying", "cancelling")
_CURRENT_ARTIFACT_ID_FIELDS = (
    "current_original_artifact_id",
    "current_sidecar_artifact_id",
    "current_blocks_artifact_id",
    "current_markdown_artifact_id",
)
_CURRENT_ARTIFACT_ID_LIST_FIELDS = (
    "current_artifact_ids",
    "current_raw_artifact_ids",
)
_ORIGIN_ATTEMPT_FIELDS = (
    "origin_attempt_token",
    "attempt_token",
    "claim_token",
    "parse_attempt_token",
    "build_attempt_token",
    "pipeline_attempt_token",
    "delete_attempt_token",
    "replace_attempt_token",
)

CleanupOutcome = Literal["succeeded", "retried", "blocked", "stale_lease"]


@dataclass(frozen=True, slots=True)
class ArtifactCleanupError:
    """Durable-safe per-manifest outcome; never carries exception text or URIs."""

    manifest_id: str
    outcome: CleanupOutcome
    error_code: str

    def __post_init__(self) -> None:
        if not isinstance(self.manifest_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._:-]{0,511}", self.manifest_id
        ):
            raise ValueError("manifest_id must be a safe non-empty identity")
        if self.outcome not in {"succeeded", "retried", "blocked", "stale_lease"}:
            raise ValueError("cleanup outcome is invalid")
        normalized = sanitize_artifact_lifecycle_error_code(self.error_code)
        if normalized != self.error_code:
            raise ValueError("cleanup error_code must be normalized and durable-safe")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ArtifactCleanupRunSummary:
    started_at: str
    finished_at: str
    recovered_leases: int
    claimed_manifests: int
    succeeded: int
    retried: int
    blocked: int
    stale_leases: int
    outcomes: tuple[ArtifactCleanupError, ...]

    def __post_init__(self) -> None:
        started = _parse_utc_datetime(self.started_at)
        finished = _parse_utc_datetime(self.finished_at)
        if finished < started:
            raise ValueError("cleanup summary cannot finish before it starts")
        if not isinstance(self.outcomes, tuple) or not all(
            isinstance(outcome, ArtifactCleanupError) for outcome in self.outcomes
        ):
            raise ValueError("cleanup summary outcomes must be an immutable tuple")
        for field_name in (
            "recovered_leases",
            "claimed_manifests",
            "succeeded",
            "retried",
            "blocked",
            "stale_leases",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.claimed_manifests != len(self.outcomes):
            raise ValueError("claimed manifest count must match outcome count")
        if (
            self.succeeded + self.retried + self.blocked + self.stale_leases
            != self.claimed_manifests
        ):
            raise ValueError("cleanup outcome counters must partition claimed work")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["outcomes"] = [outcome.to_dict() for outcome in self.outcomes]
        return value


@dataclass(frozen=True, slots=True)
class _ManifestOutcome:
    manifest_id: str
    outcome: CleanupOutcome
    error_code: str


class _CleanupDecision(RuntimeError):
    def __init__(self, error_code: str) -> None:
        normalized = sanitize_artifact_lifecycle_error_code(error_code)
        self.error_code = normalized or "artifact_cleanup_error"
        super().__init__(self.error_code)


class _BlockCleanup(_CleanupDecision):
    pass


class _RetryCleanup(_CleanupDecision):
    pass


class ArtifactCleanupService:
    """Process one bounded cleanup-manifest claim page."""

    def __init__(
        self,
        metadata_store: Any,
        object_storage: ObjectStorage,
        config: ArtifactCleanupConfig,
        *,
        workspace_resolver: Callable[[str], str] = sanitize_workspace,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if isinstance(object_storage, DisabledObjectStorage):
            raise ValueError("Artifact cleanup requires enabled object storage")
        if not isinstance(object_storage, ObjectStorage):
            raise TypeError("object_storage must implement ObjectStorage")
        if not isinstance(config, ArtifactCleanupConfig):
            raise TypeError("config must be an ArtifactCleanupConfig")
        self._metadata_store = metadata_store
        self._object_storage = object_storage
        self._config = config
        self._workspace_resolver = workspace_resolver
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    async def run_once(
        self, now: datetime | str, lease_owner: str
    ) -> ArtifactCleanupRunSummary:
        started = _parse_utc_datetime(now)
        recovered = (
            await self._metadata_store.recover_expired_artifact_cleanup_manifest_leases(
                now=started,
                next_attempt_at=started,
                limit=self._config.expired_lease_recovery_limit,
            )
        )
        claimed: list[
            ArtifactCleanupManifestRecord
        ] = await self._metadata_store.claim_due_artifact_cleanup_manifests(
            lease_owner=lease_owner,
            lease_duration_seconds=self._config.lease_duration_seconds,
            limit=self._config.claim_limit,
            now=started,
        )
        semaphore = asyncio.Semaphore(self._config.max_concurrent_manifests)

        async def bounded_process(
            manifest: ArtifactCleanupManifestRecord,
        ) -> _ManifestOutcome:
            async with semaphore:
                return await self._process_manifest(
                    manifest,
                    lease_owner=lease_owner,
                    run_floor=started,
                )

        outcomes = tuple(
            await asyncio.gather(*(bounded_process(item) for item in claimed))
        )
        finished = self._effective_now(started)
        errors = tuple(
            ArtifactCleanupError(
                manifest_id=outcome.manifest_id,
                outcome=outcome.outcome,
                error_code=outcome.error_code,
            )
            for outcome in outcomes
        )
        return ArtifactCleanupRunSummary(
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
            recovered_leases=len(recovered),
            claimed_manifests=len(claimed),
            succeeded=sum(item.outcome == "succeeded" for item in outcomes),
            retried=sum(item.outcome == "retried" for item in outcomes),
            blocked=sum(item.outcome == "blocked" for item in outcomes),
            stale_leases=sum(item.outcome == "stale_lease" for item in outcomes),
            outcomes=errors,
        )

    async def _process_manifest(
        self,
        manifest: ArtifactCleanupManifestRecord,
        *,
        lease_owner: str,
        run_floor: datetime,
    ) -> _ManifestOutcome:
        lease_token = manifest.lease_token
        if (
            manifest.status != "leased"
            or manifest.lease_owner != lease_owner
            or not lease_token
        ):
            return _ManifestOutcome(
                manifest.id, "stale_lease", "artifact_cleanup_stale_lease"
            )
        try:
            target = await self._revalidate_authority(manifest, run_floor=run_floor)

            async def renew_lease() -> None:
                await self._metadata_store.renew_artifact_cleanup_manifest_lease(
                    manifest.id,
                    lease_owner=lease_owner,
                    lease_token=lease_token,
                    lease_duration_seconds=self._config.lease_duration_seconds,
                    now=self._effective_now(run_floor),
                )

            async def renew_and_revalidate() -> None:
                await renew_lease()
                revalidated_target = await self._revalidate_authority(
                    manifest,
                    run_floor=run_floor,
                )
                if revalidated_target != target:
                    raise _BlockCleanup("cleanup_target_authority_changed")

            await renew_lease()
            result = await self._object_storage.verified_delete_cleanup_target(
                target,
                expected_size_bytes=manifest.expected_size_bytes,
                expected_checksum=manifest.expected_checksum,
                expected_etag=manifest.expected_etag,
                expected_version_id=manifest.expected_version_id,
                object_page_size=self._config.object_page_size,
                delete_batch_size=self._config.delete_batch_size,
                max_prefix_pages=(self._config.max_prefix_pages_per_manifest_attempt),
                before_exact_step=(
                    renew_and_revalidate if target.kind == "object" else None
                ),
                before_prefix_page=(
                    renew_and_revalidate if target.kind == "prefix" else None
                ),
            )
            if not isinstance(result, VerifiedDeleteResult) or not result.absent:
                raise _RetryCleanup("artifact_cleanup_absence_unproved")
            await self._metadata_store.succeed_artifact_cleanup_manifest(
                manifest.id,
                lease_owner=lease_owner,
                lease_token=lease_token,
                checked_at=self._effective_now(run_floor),
            )
            return _ManifestOutcome(
                manifest.id, "succeeded", "artifact_cleanup_succeeded"
            )
        except ArtifactLifecycleLeaseError:
            return _ManifestOutcome(
                manifest.id, "stale_lease", "artifact_cleanup_stale_lease"
            )
        except _BlockCleanup as exc:
            return await self._block_manifest(
                manifest,
                lease_owner=lease_owner,
                lease_token=lease_token,
                error_code=exc.error_code,
                run_floor=run_floor,
            )
        except _RetryCleanup as exc:
            return await self._retry_manifest(
                manifest,
                lease_owner=lease_owner,
                lease_token=lease_token,
                error_code=exc.error_code,
                run_floor=run_floor,
            )
        except ObjectStorageProofError as exc:
            if exc.retryable:
                return await self._retry_manifest(
                    manifest,
                    lease_owner=lease_owner,
                    lease_token=lease_token,
                    error_code=exc.error_code,
                    run_floor=run_floor,
                )
            return await self._block_manifest(
                manifest,
                lease_owner=lease_owner,
                lease_token=lease_token,
                error_code=exc.error_code,
                run_floor=run_floor,
            )
        except Exception:
            return await self._retry_manifest(
                manifest,
                lease_owner=lease_owner,
                lease_token=lease_token,
                error_code="artifact_cleanup_internal_error",
                run_floor=run_floor,
            )

    async def _revalidate_authority(
        self,
        manifest: ArtifactCleanupManifestRecord,
        *,
        run_floor: datetime,
    ) -> ArtifactCleanupTarget:
        lifecycle = await self._metadata_store.get_kb_lifecycle(manifest.kb_id)
        if lifecycle is None:
            raise _BlockCleanup("kb_lifecycle_missing")
        if lifecycle.generation != manifest.kb_generation:
            raise _BlockCleanup("kb_generation_stale")

        try:
            owned_workspace = self._workspace_resolver(manifest.kb_id)
        except Exception as exc:
            raise _BlockCleanup("workspace_resolution_failed") from exc
        if manifest.workspace != owned_workspace:
            raise _BlockCleanup("cleanup_workspace_mismatch")

        kb_delete_authority = (
            manifest.reason == "kb_delete"
            and manifest.target_namespace == "workspace"
            and manifest.target_kind == "prefix"
            and manifest.document_id is None
            and manifest.artifact_id is None
            and manifest.source_generation_id is None
        )
        if (
            manifest.target_namespace == "workspace" or manifest.reason == "kb_delete"
        ) and not kb_delete_authority:
            raise _BlockCleanup("kb_delete_authority_mismatch")
        if lifecycle.state == "active":
            if kb_delete_authority:
                raise _BlockCleanup("kb_delete_lifecycle_not_deleting")
        elif lifecycle.state == "deleting" and kb_delete_authority:
            if (
                manifest.origin_job_id is None
                or lifecycle.delete_job_id != manifest.origin_job_id
            ):
                raise _BlockCleanup("kb_delete_authority_mismatch")
        elif lifecycle.state == "deleted":
            raise _BlockCleanup("kb_lifecycle_deleted")
        else:
            raise _BlockCleanup("kb_lifecycle_not_active")

        try:
            target = self._object_storage.validate_cleanup_target(
                manifest.target_uri,
                target_kind=manifest.target_kind,
                target_namespace=manifest.target_namespace,
                workspace=manifest.workspace,
                document_id=manifest.document_id,
                artifact_id=manifest.artifact_id,
                source_generation_id=manifest.source_generation_id,
                origin_job_id=manifest.origin_job_id,
                origin_attempt_token=manifest.origin_attempt_token,
            )
        except ObjectStorageProofError:
            raise
        except Exception as exc:
            raise _BlockCleanup("cleanup_target_malformed") from exc

        await self._check_same_target_manifests(
            manifest,
            now=self._effective_now(run_floor),
        )
        origin_job = await self._check_origin_job(manifest)

        if manifest.document_id is not None:
            document = await self._metadata_store.get_document_lifecycle(
                manifest.kb_id, manifest.document_id
            )
            if document is None:
                raise _BlockCleanup("document_lifecycle_missing")
            if document.workspace != manifest.workspace:
                raise _BlockCleanup("document_workspace_mismatch")
            if not isinstance(document.metadata, dict):
                raise _BlockCleanup("document_metadata_malformed")
            await self._check_active_document_jobs(manifest)
            if document.deleted_at is not None:
                self._validate_tombstone_authority(manifest, document.metadata)
            else:
                if manifest.reason == "document_delete":
                    raise _BlockCleanup("document_delete_tombstone_missing")
                await self._check_live_document_references(manifest, document)
            self._check_document_origin_lineage(manifest, document.metadata)
        elif manifest.target_namespace in {
            "source",
            "legacy_source",
            "artifact",
        }:
            raise _BlockCleanup("cleanup_document_authority_missing")

        if (
            origin_job is not None
            and origin_job.status in _ACTIVE_JOB_STATUSES
            and not kb_delete_authority
        ):
            raise _RetryCleanup("origin_job_active")
        return target

    async def _check_same_target_manifests(
        self, manifest: ArtifactCleanupManifestRecord, *, now: datetime
    ) -> None:
        siblings, total = await self._metadata_store.list_artifact_cleanup_manifests(
            target_uri=manifest.target_uri,
            limit=ARTIFACT_LIFECYCLE_MAX_PAGE_SIZE,
        )
        if total < len(siblings):
            raise _BlockCleanup("same_target_authority_malformed")
        if total > len(siblings):
            raise _BlockCleanup("same_target_authority_overflow")
        for sibling in siblings:
            if sibling.id == manifest.id:
                continue
            if (
                sibling.workspace != manifest.workspace
                or sibling.kb_id != manifest.kb_id
                or sibling.kb_generation != manifest.kb_generation
            ):
                raise _BlockCleanup("same_target_ownership_conflict")
            if sibling.status in {"retained", "blocked"}:
                raise _BlockCleanup("same_target_manifest_conflict")
            if sibling.status == "leased":
                if sibling.lease_expires_at is None:
                    raise _RetryCleanup("same_target_competing_lease")
                expires_at = _parse_utc_datetime(sibling.lease_expires_at)
                same_run_higher_sibling = (
                    sibling.lease_owner == manifest.lease_owner
                    and sibling.id > manifest.id
                )
                if expires_at > now and not same_run_higher_sibling:
                    raise _RetryCleanup("same_target_competing_lease")
            if sibling.status == "pending" and sibling.id < manifest.id:
                raise _RetryCleanup("same_target_manifest_pending")

    async def _check_origin_job(self, manifest: ArtifactCleanupManifestRecord) -> Any:
        if manifest.origin_job_id is None:
            return None
        try:
            job = await self._metadata_store.get_job(
                manifest.kb_id, manifest.origin_job_id
            )
        except MetadataRecordNotFoundError as exc:
            raise _BlockCleanup("origin_job_missing") from exc
        if job.workspace != manifest.workspace:
            raise _BlockCleanup("origin_job_workspace_mismatch")
        if job.document_id not in {None, manifest.document_id}:
            raise _BlockCleanup("origin_job_document_mismatch")
        if _job_has_unknown_commit_outcome(job):
            raise _BlockCleanup("metadata_commit_outcome_unknown")
        if manifest.origin_attempt_token is not None:
            attempts = _explicit_job_attempt_tokens(job)
            if attempts and manifest.origin_attempt_token not in attempts:
                raise _BlockCleanup("origin_attempt_lineage_mismatch")
            if manifest.target_namespace == "staging" and not attempts:
                raise _BlockCleanup("origin_attempt_lineage_unknown")
        return job

    async def _check_active_document_jobs(
        self, manifest: ArtifactCleanupManifestRecord
    ) -> None:
        jobs, total = await self._metadata_store.list_jobs(
            manifest.kb_id,
            statuses=_ACTIVE_JOB_STATUSES,
            document_id=manifest.document_id,
            limit=self._config.active_job_query_limit,
            offset=0,
        )
        if total < len(jobs):
            raise _RetryCleanup("active_job_query_malformed")
        if total > len(jobs):
            raise _RetryCleanup("active_job_query_overflow")
        if jobs:
            raise _RetryCleanup("document_job_active")

    async def _check_live_document_references(
        self, manifest: ArtifactCleanupManifestRecord, document: Any
    ) -> None:
        metadata = document.metadata if isinstance(document.metadata, dict) else {}
        target_uri = normalize_artifact_target_uri(manifest.target_uri)
        if manifest.target_namespace in {"source", "legacy_source"}:
            source_uris = _explicit_string_values(
                metadata,
                ("current_source_object_uri", "source_object_uri"),
            )
            if isinstance(document.source_uri, str) and document.source_uri:
                source_uris.add(document.source_uri)
            if any(
                _reference_matches_target(
                    target_uri,
                    target_kind=manifest.target_kind,
                    reference_uri=value,
                )
                for value in source_uris
            ):
                raise _BlockCleanup("current_source_reference")
            if manifest.target_namespace == "source":
                current_generations = _explicit_string_values(
                    metadata,
                    ("current_source_generation_id", "source_generation_id"),
                )
                if manifest.source_generation_id in current_generations:
                    raise _BlockCleanup("current_source_generation")

        current_artifact_ids = _current_artifact_ids(metadata)
        if (
            manifest.target_namespace == "artifact"
            and manifest.artifact_id in current_artifact_ids
        ):
            raise _BlockCleanup("current_artifact_reference")

        artifacts, total = await self._metadata_store.list_document_artifacts(
            manifest.kb_id,
            manifest.document_id,
            limit=self._config.active_job_query_limit,
            offset=0,
        )
        if total < len(artifacts):
            raise _BlockCleanup("artifact_reference_query_malformed")
        if total > len(artifacts):
            raise _BlockCleanup("artifact_reference_query_overflow")
        for artifact in artifacts:
            references = _artifact_reference_uris(artifact)
            normalized_references = {
                normalized
                for value in references
                if (normalized := _normalize_reference_uri(value)) is not None
            }
            if any(
                _reference_matches_target(
                    target_uri,
                    target_kind=manifest.target_kind,
                    reference_uri=reference,
                )
                for reference in normalized_references
            ):
                raise _BlockCleanup("current_artifact_uri_reference")
            if (
                manifest.target_namespace == "artifact"
                and artifact.id == manifest.artifact_id
            ):
                raise _BlockCleanup("artifact_authority_mismatch")

    @staticmethod
    def _validate_tombstone_authority(
        manifest: ArtifactCleanupManifestRecord, metadata: Mapping[str, Any]
    ) -> None:
        if manifest.reason != "document_delete" or manifest.origin_job_id is None:
            raise _BlockCleanup("document_tombstone_authority_mismatch")
        recorded_job = _first_explicit_string(
            metadata,
            ("last_delete_job_id", "document_delete_job_id", "delete_job_id"),
        )
        if recorded_job != manifest.origin_job_id:
            raise _BlockCleanup("document_tombstone_authority_mismatch")
        if manifest.origin_attempt_token is not None:
            recorded_attempt = _first_explicit_string(
                metadata,
                (
                    "last_delete_attempt_token",
                    "document_delete_attempt_token",
                    "delete_attempt_token",
                ),
            )
            if (
                recorded_attempt is not None
                and recorded_attempt != manifest.origin_attempt_token
            ):
                raise _BlockCleanup("origin_attempt_lineage_mismatch")

    @staticmethod
    def _check_document_origin_lineage(
        manifest: ArtifactCleanupManifestRecord, metadata: Mapping[str, Any]
    ) -> None:
        if manifest.origin_job_id is None:
            return
        reason_fields: dict[str, tuple[str, ...]] = {
            "replace": ("last_replace_job_id",),
            "document_delete": ("last_delete_job_id",),
            "orphan_reconcile": (
                "last_replace_job_id",
                "last_failed_replace_job_id",
            ),
        }
        known_attempts = (
            _document_attempt_tokens(metadata, manifest.reason)
            if manifest.origin_attempt_token is not None
            else set()
        )
        # A replace/orphan_reconcile manifest whose origin attempt token is
        # still present in the durable attempt-token history remains
        # authoritative during the grace window even after a newer replace has
        # advanced ``last_replace_job_id``.  The current-source/current-
        # generation checks elsewhere still block any candidate that has become
        # the live pointer, so this never authorizes deleting the current copy.
        attempt_authorized = (
            manifest.origin_attempt_token is not None
            and bool(known_attempts)
            and manifest.origin_attempt_token in known_attempts
            and manifest.reason in {"replace", "orphan_reconcile"}
        )
        fields = reason_fields.get(manifest.reason)
        if fields is not None:
            recorded = _first_explicit_string(metadata, fields)
            if recorded is not None and recorded != manifest.origin_job_id:
                if not attempt_authorized:
                    raise _BlockCleanup("origin_job_lineage_mismatch")
        if manifest.origin_attempt_token is not None:
            if known_attempts and manifest.origin_attempt_token not in known_attempts:
                raise _BlockCleanup("origin_attempt_lineage_mismatch")
            if manifest.reason == "orphan_reconcile" and not known_attempts:
                # Candidate compensation for a rolled-back replace requires a
                # durable historical replace/failed attempt token; a tokenless
                # orphan_reconcile manifest is never authorized.
                raise _BlockCleanup("origin_attempt_lineage_mismatch")

    async def _retry_manifest(
        self,
        manifest: ArtifactCleanupManifestRecord,
        *,
        lease_owner: str,
        lease_token: str,
        error_code: str,
        run_floor: datetime,
    ) -> _ManifestOutcome:
        checked_at = self._effective_now(run_floor)
        next_attempt_at = checked_at + timedelta(
            seconds=self._backoff_seconds(manifest.attempt_count)
        )
        safe_code = sanitize_artifact_lifecycle_error_code(error_code) or (
            "artifact_cleanup_error"
        )
        try:
            await self._metadata_store.retry_artifact_cleanup_manifest(
                manifest.id,
                lease_owner=lease_owner,
                lease_token=lease_token,
                next_attempt_at=next_attempt_at,
                error_code=safe_code,
                checked_at=checked_at,
            )
        except ArtifactLifecycleLeaseError:
            return _ManifestOutcome(
                manifest.id, "stale_lease", "artifact_cleanup_stale_lease"
            )
        except Exception:
            return _ManifestOutcome(
                manifest.id, "stale_lease", "artifact_cleanup_state_write_unresolved"
            )
        return _ManifestOutcome(manifest.id, "retried", safe_code)

    async def _block_manifest(
        self,
        manifest: ArtifactCleanupManifestRecord,
        *,
        lease_owner: str,
        lease_token: str,
        error_code: str,
        run_floor: datetime,
    ) -> _ManifestOutcome:
        safe_code = sanitize_artifact_lifecycle_error_code(error_code) or (
            "artifact_cleanup_error"
        )
        try:
            await self._metadata_store.block_artifact_cleanup_manifest(
                manifest.id,
                lease_owner=lease_owner,
                lease_token=lease_token,
                error_code=safe_code,
                checked_at=self._effective_now(run_floor),
            )
        except ArtifactLifecycleLeaseError:
            return _ManifestOutcome(
                manifest.id, "stale_lease", "artifact_cleanup_stale_lease"
            )
        except Exception:
            return _ManifestOutcome(
                manifest.id, "stale_lease", "artifact_cleanup_state_write_unresolved"
            )
        return _ManifestOutcome(manifest.id, "blocked", safe_code)

    def _backoff_seconds(self, attempt_count: int) -> float:
        delay = self._config.backoff_base_seconds
        for _ in range(min(max(attempt_count, 0), 63)):
            if delay >= self._config.backoff_max_seconds:
                break
            delay = min(self._config.backoff_max_seconds, delay * 2.0)
        return min(delay, self._config.backoff_max_seconds)

    def _effective_now(self, floor: datetime) -> datetime:
        current = _parse_utc_datetime(self._clock())
        return current if current > floor else floor


def _parse_utc_datetime(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("cleanup time must be a valid UTC timestamp") from exc
    else:
        raise ValueError("cleanup time must be a datetime or ISO timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("cleanup time must include a timezone")
    return parsed.astimezone(timezone.utc)


def _explicit_string_values(
    mapping: Mapping[str, Any], fields: Sequence[str]
) -> set[str]:
    return {
        value
        for field_name in fields
        if isinstance((value := mapping.get(field_name)), str) and value
    }


def _first_explicit_string(
    mapping: Mapping[str, Any], fields: Sequence[str]
) -> str | None:
    for field_name in fields:
        value = mapping.get(field_name)
        if isinstance(value, str) and value:
            return value
    return None


def _normalize_reference_uri(value: str) -> str | None:
    try:
        return normalize_artifact_target_uri(value)
    except (TypeError, ValueError):
        if isinstance(value, str) and value.strip().lower().startswith("s3:"):
            raise _BlockCleanup("current_reference_malformed")
        return None


def _reference_matches_target(
    target_uri: str,
    *,
    target_kind: str,
    reference_uri: str,
) -> bool:
    normalized_reference = _normalize_reference_uri(reference_uri)
    if normalized_reference is None:
        return False
    if target_kind == "object":
        return normalized_reference == target_uri
    return normalized_reference.startswith(target_uri)


def _current_artifact_ids(metadata: Mapping[str, Any]) -> set[str]:
    result = _explicit_string_values(metadata, _CURRENT_ARTIFACT_ID_FIELDS)
    for field_name in _CURRENT_ARTIFACT_ID_LIST_FIELDS:
        values = metadata.get(field_name)
        if isinstance(values, list):
            result.update(value for value in values if isinstance(value, str) and value)
    binding = metadata.get("artifact_binding")
    if isinstance(binding, dict):
        result.update(
            _explicit_string_values(
                binding,
                ("sidecar_artifact_id", "blocks_artifact_id"),
            )
        )
        raw_ids = binding.get("raw_artifact_ids")
        if isinstance(raw_ids, list):
            result.update(
                value for value in raw_ids if isinstance(value, str) and value
            )
    return result


def _artifact_reference_uris(artifact: Any) -> set[str]:
    values: set[str] = set()
    if isinstance(artifact.uri, str) and artifact.uri:
        values.add(artifact.uri)
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    values.update(
        _explicit_string_values(metadata, ("object_uri", "object_prefix_uri"))
    )
    return values


def _document_attempt_tokens(metadata: Mapping[str, Any], reason: str) -> set[str]:
    operation = {
        "replace": "replace",
        "orphan_reconcile": "replace",
        "document_delete": "delete",
    }.get(reason)
    if operation is None:
        return set()
    fields = (
        f"current_{operation}_attempt_token",
        f"last_{operation}_attempt_token",
        f"{operation}_attempt_token",
    )
    if operation == "replace":
        # Earlier failed replace attempts remain durably authorized through the
        # persistent attempt-token history, including the canonical failed-
        # replace token recorded when a replacement rolls back.
        fields = (*fields, "last_failed_replace_attempt_token")
    result = _explicit_string_values(metadata, fields)
    history = metadata.get(f"{operation}_attempt_token_history")
    if isinstance(history, list):
        result.update(value for value in history if isinstance(value, str) and value)
    return result


def _job_has_unknown_commit_outcome(job: Any) -> bool:
    if job.error_code == "metadata_commit_outcome_unknown":
        return True
    for mapping in (job.payload, job.result):
        if not isinstance(mapping, dict):
            continue
        if mapping.get("metadata_commit_outcome_unknown") is True:
            return True
        for field_name in (
            "error_code",
            "metadata_commit_error_code",
            "commit_outcome",
            "metadata_commit_outcome",
        ):
            value = mapping.get(field_name)
            if isinstance(value, str) and value.strip().lower() in {
                "indeterminate",
                "pending",
                "unresolved",
                "unknown",
                "metadata_commit_outcome_unknown",
            }:
                return True
    return False


def _explicit_job_attempt_tokens(job: Any) -> set[str]:
    result: set[str] = set()
    for mapping in (job.payload, job.result):
        if not isinstance(mapping, dict):
            continue
        result.update(_explicit_string_values(mapping, _ORIGIN_ATTEMPT_FIELDS))
    return result


__all__ = [
    "ArtifactCleanupError",
    "ArtifactCleanupRunSummary",
    "ArtifactCleanupService",
]
