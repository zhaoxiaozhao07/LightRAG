#!/usr/bin/env python3
"""Migrate LightRAG on-disk artifacts into object-storage authority.

This tool scans one or more explicit ``LABEL=/absolute/root`` legacy roots,
builds a durable dry-run maintenance plan (using the frozen Phase 3.1-A
``ArtifactMaintenanceRunRecord``/``ArtifactMaintenanceItemRecord`` authority),
and then optionally applies the plan by uploading each file via the frozen
``upload_file_if_absent`` primitive and committing the resulting object
pointers through existing public metadata-store APIs.

The plan/apply/resume state machine is durable: every item progresses
through ``planned -> uploaded -> applied -> verified`` and a crash never
strands partially-applied bytes.  Online KB mutation is rejected during
apply.  All audit/JSON output is redacted so credentials, DSNs, scratch
paths, and absolute local roots cannot leak.

Usage::

    lightrag-migrate-artifacts-to-object \\
        --working-dir ./rag_storage \\
        --bucket lightrag-kb \\
        legacyA=/srv/rag/legacy-a legacyB=/srv/rag/legacy-b

    lightrag-migrate-artifacts-to-object \\
        --working-dir ./rag_storage \\
        --bucket lightrag-kb \\
        --plan-id <plan_id_from_dry_run> --yes
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import secrets
import stat
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from dotenv import load_dotenv

# Add project root to path for direct ``python lightrag/tools/...`` execution.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from lightrag.api.artifact_lifecycle import (
    ArtifactLifecycleConflictError,
    ArtifactLifecycleError,
    ArtifactLifecycleStateError,
    ArtifactMaintenanceItemRecord,
    ArtifactMaintenanceItemState,
    ArtifactMaintenanceMetadataBackend,
    ArtifactMaintenanceRunRecord,
    artifact_maintenance_item_key,
    artifact_target_uri_digest,
    normalize_artifact_relative_object_id,
    normalize_artifact_root_label,
    normalize_artifact_target_uri_authority,
)
from lightrag.api.kb_service import utc_now_iso
from lightrag.api.metadata_store import SQLiteMetadataStore
from lightrag.api.object_storage import (
    ObjectStorage,
    ObjectStorageConfig,
    S3ObjectStorage,
)

load_dotenv(dotenv_path=".env", override=False)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_LEASE_SECONDS = 600.0
_MAX_ITEMS_PER_RUN = 500  # bounded maintenance item batch (frozen store cap)
_MAINTENANCE_KIND = "migration"
_MAINTENANCE_BACKEND_FINGERPRINT = "sha256:migrate-artifacts-to-object:v1"
_TERMINAL_ITEM_STATES: frozenset[ArtifactMaintenanceItemState] = frozenset(
    {"verified", "skipped", "blocked", "failed"}
)
_ACTIVE_MUTATION_JOB_STATUSES: tuple[str, ...] = (
    "queued",
    "running",
    "retrying",
    "cancelling",
)
_DEFAULT_OBJECT_PREFIX = "kb"

# Redaction patterns for audit/JSON output.  Mirrors the existing
# ``_redact_scratch_references`` and durable-safe rules: credentials, DSNs,
# presigned URLs, scratch roots, and absolute local paths never appear in
# durable output.
_SCRATCH_RE = re.compile(
    r"(?:file://)?[^\s\"']*\.lightrag-scratch[/\\][^\s\"']*",
    re.IGNORECASE,
)
_DSN_RE = re.compile(
    r"(?:postgres(?:ql)?|mysql|redis|mongodb)(?:\+[\w]+)?://[^\s\"'<>]+",
    re.IGNORECASE,
)
_QUERY_DSN_RE = re.compile(
    r"(?:^|\s)(?:host|hostaddr|port|dbname|database|user|password|sslmode)\s*=",
    re.IGNORECASE,
)
_SECRET_KEY_RE = re.compile(
    r"(?:aws[_-]?access[_-]?key(?:[_-]?id)?|"
    r"aws[_-]?secret[_-]?access[_-]?key|"
    r"access[_-]?key(?:[_-]?id)?|secret[_-]?(?:access[_-]?)?key|"
    r"x-amz-(?:credential|signature|security-token)|"
    r"x-goog-(?:credential|signature)|password|secret|token)"
    r"\s*[=:]\s*[^\s\"'<>]+",
    re.IGNORECASE,
)
_ABSOLUTE_ROOT_RE = re.compile(r"(?:^|[\s\"'<>])(/[^\"'<>\s]+)")
_SCRATCH_TOKEN_RE = re.compile(
    r"(?:\.lightrag-scratch|\.sync-staging|\.replace-staging)",
    re.IGNORECASE,
)


class MigrationError(RuntimeError):
    """Base class for migration CLI errors. Messages are redaction-safe."""


class MigrationSecurityError(MigrationError):
    """A legacy root or file failed the descriptor-safe validation."""


class MigrationPlanError(MigrationError):
    """The requested plan is missing, ambiguous, or not yet succeeded."""


class MigrationApplyGuardError(MigrationError):
    """An online KB mutation or ownership precondition blocked apply."""


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def _redact_value(value: object) -> str:
    """Redact scratch paths, DSNs, credentials, and absolute roots from output."""

    text = str(value)
    text = _SCRATCH_RE.sub("<artifact-materialization>", text)
    text = _DSN_RE.sub("<redacted-dsn>", text)
    text = _SECRET_KEY_RE.sub("<redacted-credential>", text)
    text = _QUERY_DSN_RE.sub("<redacted-dsn>", text)
    text = _SCRATCH_TOKEN_RE.sub("<artifact-materialization>", text)
    # Drop absolute local-path references; preserve object URIs (s3://...).
    text = _ABSOLUTE_ROOT_RE.sub(
        lambda match: f"{match.group(1)[:1]}<redacted-root>", text
    )
    return text


def _redact_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            redacted[key] = _redact_mapping(item)
        elif isinstance(item, str):
            redacted[key] = _redact_value(item)
        else:
            redacted[key] = item
    return redacted


# ---------------------------------------------------------------------------
# Security: explicit-root validation + descriptor-relative no-follow reads
# ---------------------------------------------------------------------------


_NETWORK_URI_SCHEMES = frozenset(
    {"http", "https", "ftp", "sftp", "ssh", "scp", "s3", "gs", "az", "abfs"}
)


def _parse_label_root_spec(spec: str) -> tuple[str, Path]:
    """Split a ``LABEL=/absolute/root`` argument into its label and path."""

    if "=" not in spec:
        raise MigrationSecurityError(
            "Legacy root must be specified as LABEL=/absolute/root"
        )
    label, raw_path = spec.split("=", 1)
    if not label or not raw_path:
        raise MigrationSecurityError(
            "Legacy root must be specified as LABEL=/absolute/root"
        )
    try:
        normalize_artifact_root_label(label)
    except ValueError as exc:
        raise MigrationSecurityError(
            f"Legacy root label is not a safe display label: {exc}"
        ) from exc
    return label, Path(raw_path)


def _validate_absolute_root(label: str, root: Path) -> None:
    """Reject anything that is not an explicit, local, regular directory root.

    Security model (handoff lines 366-372):
      * reject relative paths (no inference from CWD/env/metadata parent);
      * reject a symlinked root entry itself;
      * reject ``..`` traversal segments;
      * reject devices, FIFOs, sockets, and network/non-file URIs;
      * reject ambiguous roots (whitespace, control chars, encoded separators).
    """

    raw = str(root)
    if not raw.startswith("/"):
        raise MigrationSecurityError(
            f"Legacy root '{label}' must be an absolute local path"
        )
    if "://" in raw or "\\" in raw:
        raise MigrationSecurityError(
            f"Legacy root '{label}' must not be a network URI or Windows path"
        )
    parsed = urlparse(raw)
    if parsed.scheme:
        raise MigrationSecurityError(f"Legacy root '{label}' must not be a network URI")
    if "\x00" in raw or any(ch.isspace() for ch in raw):
        raise MigrationSecurityError(
            f"Legacy root '{label}' must not contain whitespace or control bytes"
        )
    parts = root.parts
    if not parts or parts[0] != "/":
        raise MigrationSecurityError(f"Legacy root '{label}' must be an absolute path")
    if any(segment in {".", ".."} for segment in parts[1:]):
        raise MigrationSecurityError(
            f"Legacy root '{label}' contains a traversal segment"
        )

    # Per-component prefix symlink detection is intentionally not performed
    # so that well-known system symlinks (e.g. macOS ``/var -> /private/var``)
    # do not produce false positives. Symlink containment within the migrated
    # tree is enforced per-file during ``_iter_regular_files`` (no-follow walk)
    # and ``_open_no_follow_and_validate`` (lstat + O_NOFOLLOW + fstat).

    try:
        root_stat = os.lstat(root)
    except OSError as exc:
        raise MigrationSecurityError(f"Legacy root '{label}' is not readable") from exc
    if stat.S_ISLNK(root_stat.st_mode):
        raise MigrationSecurityError(
            f"Legacy root '{label}' must not itself be a symlink"
        )
    if not stat.S_ISDIR(root_stat.st_mode):
        raise MigrationSecurityError(f"Legacy root '{label}' must be a directory")
    if stat.S_ISCHR(root_stat.st_mode) or stat.S_ISBLK(root_stat.st_mode):
        raise MigrationSecurityError(f"Legacy root '{label}' must not be a device")
    if stat.S_ISFIFO(root_stat.st_mode):
        raise MigrationSecurityError(f"Legacy root '{label}' must not be a FIFO")
    if stat.S_ISSOCK(root_stat.st_mode):
        raise MigrationSecurityError(f"Legacy root '{label}' must not be a socket")


def _open_no_follow_and_validate(path: Path) -> tuple[int, os.stat_result]:
    """Descriptor-relative, no-follow open + ``fstat`` revalidation.

    Returns a file descriptor owned by the caller.  The descriptor was opened
    with ``O_NOFOLLOW`` after a no-follow ``lstat`` confirmed the entry is a
    regular file (not a symlink, device, FIFO, or socket); ``fstat`` is then
    re-run on the descriptor itself to prove the open file matches.
    """

    try:
        lstat_result = os.lstat(path)
    except FileNotFoundError as exc:
        raise MigrationSecurityError("Legacy artifact disappeared before open") from exc
    except OSError as exc:
        raise MigrationSecurityError("Legacy artifact unreadable before open") from exc
    if stat.S_ISLNK(lstat_result.st_mode):
        raise MigrationSecurityError("Legacy artifact must not be a symlink")
    if not stat.S_ISREG(lstat_result.st_mode):
        raise MigrationSecurityError("Legacy artifact must be a regular file")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise MigrationSecurityError("Legacy artifact open failed") from exc
    try:
        fstat_result = os.fstat(fd)
    except OSError as exc:
        os.close(fd)
        raise MigrationSecurityError("Legacy artifact fstat failed") from exc
    if (
        not stat.S_ISREG(fstat_result.st_mode)
        or fstat_result.st_ino != lstat_result.st_ino
        or fstat_result.st_dev != lstat_result.st_dev
    ):
        os.close(fd)
        raise MigrationSecurityError(
            "Legacy artifact fstat did not match the lstat entry"
        )
    return fd, fstat_result


def _compute_sha256_and_size(fd: int) -> tuple[str, int]:
    """Stream-hash an open descriptor and return ``(sha256_hex, size_bytes)``."""

    hasher = hashlib.sha256()
    total = 0
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        hasher.update(chunk)
        total += len(chunk)
    return hasher.hexdigest(), total


# ---------------------------------------------------------------------------
# Target object key construction
# ---------------------------------------------------------------------------


def _target_object_key(
    *,
    prefix: str,
    root_label: str,
    relative_path: str,
) -> str:
    """Build a deterministic, immutable target object key for one source file.

    Migration keys live under ``<prefix>/migrate/<root_label>/<rel>``.  The
    ``migrate`` segment isolates migrated objects from the canonical
    per-document ``source/generations/`` namespace used by online writes.
    """

    rel_normalized = relative_path.replace("\\", "/").lstrip("/")
    rel = normalize_artifact_relative_object_id(rel_normalized)
    key = f"{prefix.rstrip('/')}/migrate/{root_label}/{rel}"
    return normalize_artifact_relative_object_id(key)


def _strip_target_key_prefix(
    *,
    relative_object_id: str,
    prefix: str,
    root_label: str,
) -> str | None:
    """Recover the legacy relative path from a migration target object key."""

    expected_prefix = f"{prefix.rstrip('/')}/migrate/{root_label}/"
    if not relative_object_id.startswith(expected_prefix):
        return None
    return relative_object_id[len(expected_prefix) :]


# ---------------------------------------------------------------------------
# In-memory item spec
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MigrationItemSpec:
    """One file scanned from an explicit legacy root, awaiting persistence."""

    root_label: str
    relative_object_id: str
    expected_checksum: str
    expected_size_bytes: int
    subject_kind: str
    subject_id: str
    local_path: Path
    kb_id: str | None = None
    kb_generation: str | None = None
    workspace: str | None = None
    document_id: str | None = None
    artifact_id: str | None = None

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "root_label": self.root_label,
            "relative_object_id": self.relative_object_id,
            "expected_checksum": self.expected_checksum,
            "expected_size_bytes": self.expected_size_bytes,
            "subject_kind": self.subject_kind,
            "subject_id": self.subject_id,
            "kb_id": self.kb_id,
            "workspace": self.workspace,
            "document_id": self.document_id,
            "artifact_id": self.artifact_id,
            # local_path intentionally omitted: never durable.
        }


@dataclass(slots=True)
class MigrationPlanSummary:
    plan_id: str
    item_count: int
    items: list[MigrationItemSpec]
    apply_run_id: str | None
    metadata_backend: ArtifactMaintenanceMetadataBackend

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "item_count": self.item_count,
            "metadata_backend": self.metadata_backend,
            "apply_run_id": self.apply_run_id,
            "items": [item.to_audit_dict() for item in self.items],
        }


@dataclass(slots=True)
class MigrationApplySummary:
    plan_id: str
    apply_run_id: str
    items_total: int
    items_verified: int
    items_skipped: int
    items_failed: int
    items_blocked: int
    counters: dict[str, int]
    issues: list[str] = field(default_factory=list)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "apply_run_id": self.apply_run_id,
            "items_total": self.items_total,
            "items_verified": self.items_verified,
            "items_skipped": self.items_skipped,
            "items_failed": self.items_failed,
            "items_blocked": self.items_blocked,
            "counters": dict(self.counters),
            "issues": list(self.issues),
        }


# ---------------------------------------------------------------------------
# Backend construction
# ---------------------------------------------------------------------------


def _resolve_metadata_backend(
    args: argparse.Namespace,
) -> ArtifactMaintenanceMetadataBackend:
    """Decide the metadata backend label (sqlite vs postgres) from args/env."""

    forced = getattr(args, "metadata_backend", None)
    if forced:
        normalized = forced.strip().lower()
        if normalized in {"sqlite", "postgres"}:
            return normalized  # type: ignore[return-value]
        raise MigrationError(
            "Unsupported --metadata-backend value; expected sqlite or postgres"
        )
    env_value = os.getenv("LIGHTRAG_KB_METADATA_BACKEND", "local").strip().lower()
    if env_value in {"postgres", "postgresql"}:
        return "postgres"
    return "sqlite"


def _metadata_store_from_args(
    args: argparse.Namespace, backend: ArtifactMaintenanceMetadataBackend
) -> Any:
    if backend == "postgres":
        from lightrag.api.postgres_metadata_store import PostgresMetadataStore

        if getattr(args, "postgres_dsn", None):
            return PostgresMetadataStore(dsn=args.postgres_dsn)
        return PostgresMetadataStore.from_env()
    sqlite_path = Path(args.working_dir) / "metadata" / "metadata.sqlite3"
    if not sqlite_path.exists():
        raise MigrationError(
            "SQLite metadata store not found under the configured --working-dir"
        )
    return SQLiteMetadataStore(sqlite_path)


def _object_storage_from_args(args: argparse.Namespace) -> ObjectStorage:
    config = ObjectStorageConfig.from_env()
    overrides: dict[str, Any] = {}
    if args.object_storage_endpoint:
        overrides["endpoint_url"] = args.object_storage_endpoint
    if args.bucket:
        overrides["bucket"] = args.bucket
    if args.prefix:
        overrides["prefix"] = args.prefix.strip("/")
    if args.use_ssl is not None:
        overrides["use_ssl"] = args.use_ssl
    if overrides:
        config = type(config)(**{**config.__dict__, **overrides})
    return S3ObjectStorage(config)


def _resolve_bucket_name(
    args: argparse.Namespace, object_storage: ObjectStorage
) -> str:
    """Resolve the configured bucket name without depending on private attrs."""

    if args.bucket:
        return args.bucket
    config_attr = getattr(object_storage, "_config", None)
    if config_attr is not None:
        candidate = getattr(config_attr, "bucket", None)
        if isinstance(candidate, str) and candidate:
            return candidate
    env_bucket = os.getenv("LIGHTRAG_OBJECT_STORAGE_BUCKET")
    if env_bucket:
        return env_bucket.strip()
    return "lightrag-kb"


def _target_uri_authority_for(object_storage: ObjectStorage) -> str:
    """Inspect the configured storage and return its authority URI."""

    probe_key = _DEFAULT_OBJECT_PREFIX + "/migrate-probe"
    try:
        probe_uri = object_storage.object_uri_for_key(probe_key)
    except Exception as exc:  # pragma: no cover - defensive
        raise MigrationError(
            f"Object storage authority could not be derived: {_redact_value(exc)}"
        ) from exc
    parsed = urlparse(probe_uri)
    return normalize_artifact_target_uri_authority(f"{parsed.scheme}://{parsed.netloc}")


# ---------------------------------------------------------------------------
# Migration driver
# ---------------------------------------------------------------------------


class ArtifactObjectMigrator:
    """Drives the plan/apply/resume state machine against the frozen stores."""

    def __init__(
        self,
        *,
        metadata_store: Any,
        object_storage: ObjectStorage,
        metadata_backend: ArtifactMaintenanceMetadataBackend,
        bucket: str,
        prefix: str,
        lease_duration_seconds: float = _DEFAULT_LEASE_SECONDS,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._metadata_store = metadata_store
        self._object_storage = object_storage
        self._metadata_backend = metadata_backend
        self._bucket = bucket
        self._prefix = prefix.strip("/") or _DEFAULT_OBJECT_PREFIX
        self._lease_duration_seconds = lease_duration_seconds
        self._target_uri_authority = _target_uri_authority_for(object_storage)
        self._now = now or _default_now

    # -- Plan phase -------------------------------------------------------

    async def create_plan(
        self,
        label_root_pairs: Sequence[tuple[str, Path]],
        *,
        actor_id: str | None = None,
    ) -> MigrationPlanSummary:
        """Walk the explicit roots and persist a durable dry-run plan."""

        if not label_root_pairs:
            raise MigrationError("At least one LABEL=/absolute/root is required")
        for label, root in label_root_pairs:
            _validate_absolute_root(label, root)
        items = await self._collect_items(label_root_pairs)
        # Empty plans are permitted: they record an explicit no-op scan result
        # so audit has a durable record that the roots were inspected.

        scope_payload = _build_scope_payload(label_root_pairs)
        scope_fingerprint = _fingerprint_json(scope_payload)
        config_payload = {
            "bucket": self._bucket,
            "prefix": self._prefix,
            "target_uri_authority": self._target_uri_authority,
        }
        config_fingerprint = _fingerprint_json(config_payload)
        run = ArtifactMaintenanceRunRecord(
            id=self._mint_run_id("plan"),
            kind=_MAINTENANCE_KIND,
            mode="dry_run",
            status="planned",
            metadata_backend=self._metadata_backend,
            parent_plan_id=None,
            backend_fingerprint=_MAINTENANCE_BACKEND_FINGERPRINT,
            scope_fingerprint=scope_fingerprint,
            config_fingerprint=config_fingerprint,
            scope_json=scope_payload,
            actor_id=actor_id,
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
        )
        run = await self._metadata_store.create_artifact_maintenance_run(run)

        maintenance_items = [
            self._spec_to_item(run.id, ordinal, spec)
            for ordinal, spec in enumerate(items)
        ]
        # Insert in bounded batches to honor the frozen store cap.
        for offset in range(0, len(maintenance_items), _MAX_ITEMS_PER_RUN):
            batch = maintenance_items[offset : offset + _MAX_ITEMS_PER_RUN]
            await self._metadata_store.create_artifact_maintenance_items(batch)

        claimed = await self._metadata_store.claim_artifact_maintenance_run(
            run.id,
            lease_owner="migrate-artifacts-to-object-plan",
            lease_duration_seconds=self._lease_duration_seconds,
        )
        # Plan is just a durable inventory; it never performs side effects.
        await self._metadata_store.transition_artifact_maintenance_run(
            run.id,
            expected_status="running",
            new_status="succeeded",
            lease_owner=claimed.lease_owner,
            lease_token=claimed.lease_token,
            counters={
                "total_items": len(items),
                "planned_items": len(items),
            },
        )
        return MigrationPlanSummary(
            plan_id=run.id,
            item_count=len(items),
            items=items,
            apply_run_id=None,
            metadata_backend=self._metadata_backend,
        )

    async def _collect_items(
        self, label_root_pairs: Sequence[tuple[str, Path]]
    ) -> list[MigrationItemSpec]:
        items: list[MigrationItemSpec] = []
        seen_keys: set[str] = set()
        for label, root in label_root_pairs:
            for path in self._iter_regular_files(root):
                rel = path.relative_to(root)
                relative_object_id = normalize_artifact_relative_object_id(
                    str(rel).replace("\\", "/")
                )
                fd, _ = _open_no_follow_and_validate(path)
                try:
                    checksum, size = _compute_sha256_and_size(fd)
                finally:
                    os.close(fd)
                target_key = _target_object_key(
                    prefix=self._prefix,
                    root_label=label,
                    relative_path=relative_object_id,
                )
                if target_key in seen_keys:
                    raise MigrationSecurityError(
                        "Duplicate target object key derived from legacy roots"
                    )
                seen_keys.add(target_key)
                # subject_id is a non-path authority identifier, so the legacy
                # relative path is encoded as a short SHA-256 fingerprint. The
                # full legacy rel is reconstructed during apply from the plan
                # scope + root_label.
                rel_fingerprint = hashlib.sha256(
                    relative_object_id.encode("utf-8")
                ).hexdigest()[:16]
                subject_id = f"{label}:{rel_fingerprint}"
                items.append(
                    MigrationItemSpec(
                        root_label=label,
                        relative_object_id=target_key,
                        expected_checksum=checksum,
                        expected_size_bytes=size,
                        subject_kind="source",
                        subject_id=subject_id,
                        local_path=path,
                    )
                )
        items.sort(key=lambda spec: (spec.root_label, spec.relative_object_id))
        return items

    def _iter_regular_files(self, root: Path) -> Iterable[Path]:
        # No-follow directory walk: ``os.walk(followlinks=False)`` is the
        # default but we make it explicit and skip any symlinked entry.
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dir_path = Path(dirpath)
            # Filter symlinked subdirectories in-place so walk does not descend.
            kept_dirnames = [
                name for name in dirnames if not (dir_path / name).is_symlink()
            ]
            dirnames[:] = kept_dirnames
            for name in filenames:
                candidate = dir_path / name
                if candidate.is_symlink():
                    continue
                yield candidate

    # -- Apply phase ------------------------------------------------------

    async def apply_plan(
        self,
        plan_id: str,
        *,
        label_root_pairs: Sequence[tuple[str, Path]] | None = None,
        document_resolver: Callable[[str, str], tuple[str, str, str] | None]
        | None = None,
        resume: bool = False,
        actor_id: str | None = None,
    ) -> MigrationApplySummary:
        """Apply (or resume) a previously-succeeded dry-run plan.

        ``label_root_pairs`` re-resolves the explicit legacy roots at apply
        time so a moved-root deployment (where the same labels map to a
        different absolute path) can resume without re-creating the plan.
        ``document_resolver`` is an optional callback
        ``(root_label, relative_legacy_path) -> (kb_id, workspace, document_id)``
        used to bind each item to the document whose source pointer should be
        rewritten.  Items without a binding skip the metadata pointer commit
        but still produce verified object authority (e.g. for orphan-only keys).
        """

        parent = await self._metadata_store.get_artifact_maintenance_run(plan_id)
        if parent.mode != "dry_run" or parent.kind != _MAINTENANCE_KIND:
            raise MigrationPlanError(
                "Provided --plan-id is not a migration dry-run plan"
            )
        if parent.status != "succeeded":
            raise MigrationPlanError("Dry-run plan must reach 'succeeded' before apply")

        await self._assert_no_online_mutation()

        apply_run = await self._ensure_apply_run(parent, actor_id=actor_id)

        # If the apply run already reached a terminal state, return the
        # current counters without re-claiming. This makes resume calls pure
        # no-ops once every item is verified.
        if apply_run.status in {"succeeded", "cancelled"}:
            counters = await self._metadata_store.aggregate_artifact_maintenance_items(
                apply_run.id
            )
            return MigrationApplySummary(
                plan_id=plan_id,
                apply_run_id=apply_run.id,
                items_total=counters.get("total", 0),
                items_verified=counters.get("verified", 0),
                items_skipped=counters.get("skipped", 0),
                items_failed=counters.get("failed", 0),
                items_blocked=counters.get("blocked", 0),
                counters=counters,
                issues=[],
            )

        claimed = await self._claim_apply_run(apply_run.id, resume=resume)
        lease_token = claimed.lease_token or ""

        items, _total = await self._metadata_store.list_artifact_maintenance_items(
            apply_run.id,
            limit=_MAX_ITEMS_PER_RUN,
        )
        local_paths = await self._resolve_local_paths_for_apply(
            parent.id, apply_run.id, label_root_pairs
        )

        issues: list[str] = []
        for item in items:
            binding = None
            if document_resolver is not None:
                binding = self._resolve_document_binding(item, document_resolver)
            try:
                await self._advance_item(
                    item=item,
                    run_lease_token=lease_token,
                    local_path=local_paths.get(item.item_key),
                    document_binding=binding,
                )
            except MigrationApplyGuardError as exc:
                issues.append(f"item {item.item_key} blocked: {_redact_value(exc)}")
            except ArtifactLifecycleError as exc:
                issues.append(
                    f"item {item.item_key} lifecycle error: {_redact_value(exc)}"
                )

        counters = await self._metadata_store.aggregate_artifact_maintenance_items(
            apply_run.id
        )
        verified = counters.get("verified", 0)
        failed = counters.get("failed", 0) + counters.get("blocked", 0)
        if verified == counters.get("total", 0) and failed == 0:
            await self._metadata_store.transition_artifact_maintenance_run(
                apply_run.id,
                expected_status="running",
                new_status="succeeded",
                lease_owner=claimed.lease_owner,
                lease_token=claimed.lease_token,
                counters={
                    "total_items": counters.get("total", 0),
                    "verified_items": verified,
                    "planned_items": counters.get("planned", 0),
                    "uploaded_items": counters.get("uploaded", 0),
                    "applied_items": counters.get("applied", 0),
                    "failed_items": counters.get("failed", 0),
                    "blocked_items": counters.get("blocked", 0),
                    "skipped_items": counters.get("skipped", 0),
                },
            )
        else:
            # Leave the run in 'running' so a future resume can re-drive it;
            # release the lease only if everything verified.
            await self._metadata_store.recover_expired_artifact_maintenance_run_leases(
                limit=10
            )

        return MigrationApplySummary(
            plan_id=plan_id,
            apply_run_id=apply_run.id,
            items_total=counters.get("total", 0),
            items_verified=verified,
            items_skipped=counters.get("skipped", 0),
            items_failed=counters.get("failed", 0),
            items_blocked=counters.get("blocked", 0),
            counters=counters,
            issues=issues,
        )

    async def _assert_no_online_mutation(self) -> None:
        # Migration apply rewrites document pointers and artifact attachments
        # across potentially many KBs, so it MUST NOT race with any concurrent
        # parse/build/replace/sync mutation job anywhere in the store. The
        # previous per-status ``list_jobs("__any__", ...)`` loop was a silent
        # no-op because ``list_jobs`` scopes strictly by ``kb_id`` in both
        # backends, so the synthetic id matched no real KB and always returned
        # ``total=0``; the ``except Exception: continue`` doubly masked it. The
        # guard now asks the store for a single cross-KB aggregate count and
        # FAILS CLOSED on any store error: apply may proceed only when the
        # store proves the system is quiescent.
        statuses = list(_ACTIVE_MUTATION_JOB_STATUSES)
        try:
            active = await self._metadata_store.count_active_jobs_globally(statuses)
        except Exception as exc:
            raise MigrationApplyGuardError(
                "Online KB mutation guard could not verify the system was "
                "quiescent; refusing to apply"
            ) from exc
        if active:
            raise MigrationApplyGuardError(
                f"Online KB mutation is in progress: {active} active job(s) "
                f"in status(es) {sorted(statuses)}; retry apply later"
            )

    async def _ensure_apply_run(
        self,
        parent: ArtifactMaintenanceRunRecord,
        *,
        actor_id: str | None,
    ) -> ArtifactMaintenanceRunRecord:
        apply_run = ArtifactMaintenanceRunRecord(
            id=self._mint_run_id("apply", parent_id=parent.id),
            kind=_MAINTENANCE_KIND,
            mode="apply",
            status="planned",
            metadata_backend=self._metadata_backend,
            parent_plan_id=parent.id,
            backend_fingerprint=parent.backend_fingerprint,
            scope_fingerprint=parent.scope_fingerprint,
            config_fingerprint=parent.config_fingerprint,
            scope_json=parent.scope_json,
            actor_id=actor_id,
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
        )
        try:
            created = await self._metadata_store.create_artifact_maintenance_run(
                apply_run
            )
        except ArtifactLifecycleConflictError as exc:
            # Idempotent replay: a previous apply run with the same fingerprints
            # already exists. Find it.
            runs, _ = await self._metadata_store.list_artifact_maintenance_runs(
                kind=_MAINTENANCE_KIND,
                mode="apply",
                parent_plan_id=parent.id,
                limit=10,
            )
            for candidate in runs:
                if candidate.idempotency_key == apply_run.idempotency_key:
                    return candidate
            raise MigrationPlanError(
                "Apply run could not be established for this plan"
            ) from exc
        # Seed the apply run with a copy of each plan item (item_key is derived
        # from run_id so the apply run needs its own rows). Idempotent: if the
        # rows already exist (resume), ``create_artifact_maintenance_items``
        # returns the existing records without raising.
        plan_items, _ = await self._metadata_store.list_artifact_maintenance_items(
            parent.id, limit=_MAX_ITEMS_PER_RUN
        )
        if plan_items:
            apply_items: list[ArtifactMaintenanceItemRecord] = []
            for ordinal, plan_item in enumerate(plan_items):
                apply_items.append(
                    self._clone_item_for_run(created.id, ordinal, plan_item)
                )
            for offset in range(0, len(apply_items), _MAX_ITEMS_PER_RUN):
                batch = apply_items[offset : offset + _MAX_ITEMS_PER_RUN]
                await self._metadata_store.create_artifact_maintenance_items(batch)
        return created

    def _clone_item_for_run(
        self,
        run_id: str,
        ordinal: int,
        plan_item: ArtifactMaintenanceItemRecord,
    ) -> ArtifactMaintenanceItemRecord:
        """Build an apply-run item that preserves the durable plan payload."""

        new_item_key = artifact_maintenance_item_key(
            run_id=run_id,
            subject_kind=plan_item.subject_kind,
            subject_id=plan_item.subject_id,
            kb_id=plan_item.kb_id,
            kb_generation=plan_item.kb_generation,
            workspace=plan_item.workspace,
            document_id=plan_item.document_id,
            artifact_id=plan_item.artifact_id,
            logical_group_id=plan_item.logical_group_id,
            relative_object_id=plan_item.relative_object_id,
            root_label=plan_item.root_label,
            expected_checksum=plan_item.expected_checksum,
            expected_size_bytes=plan_item.expected_size_bytes,
            target_uri_authority=plan_item.target_uri_authority,
            target_uri_digest=plan_item.target_uri_digest,
            payload_json=plan_item.payload_json,
        )
        return ArtifactMaintenanceItemRecord(
            id=f"{run_id}-item-{ordinal:04d}",
            run_id=run_id,
            item_key=new_item_key,
            state="planned",
            ordinal=ordinal,
            subject_kind=plan_item.subject_kind,
            subject_id=plan_item.subject_id,
            kb_id=plan_item.kb_id,
            kb_generation=plan_item.kb_generation,
            workspace=plan_item.workspace,
            document_id=plan_item.document_id,
            artifact_id=plan_item.artifact_id,
            logical_group_id=plan_item.logical_group_id,
            relative_object_id=plan_item.relative_object_id,
            root_label=plan_item.root_label,
            expected_checksum=plan_item.expected_checksum,
            expected_size_bytes=plan_item.expected_size_bytes,
            target_uri_authority=plan_item.target_uri_authority,
            target_uri_digest=plan_item.target_uri_digest,
            payload_json=plan_item.payload_json,
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
        )

    async def _claim_apply_run(
        self, apply_run_id: str, *, resume: bool
    ) -> ArtifactMaintenanceRunRecord:
        if resume:
            # Best-effort: reap stale leases so a crashed apply becomes
            # claimable again.
            await self._metadata_store.recover_expired_artifact_maintenance_run_leases(
                limit=20
            )
        try:
            return await self._metadata_store.claim_artifact_maintenance_run(
                apply_run_id,
                lease_owner="migrate-artifacts-to-object-apply",
                lease_duration_seconds=self._lease_duration_seconds,
            )
        except ArtifactLifecycleStateError as exc:
            raise MigrationApplyGuardError(
                "Apply run is not claimable in its current state; "
                "use --resume after the lease expires"
            ) from exc

    async def _resolve_local_paths_for_apply(
        self,
        plan_id: str,
        apply_run_id: str,
        label_root_pairs: Sequence[tuple[str, Path]] | None,
    ) -> dict[str, Path]:
        """Re-resolve the item_key -> local_path map from explicit apply roots.

        The maintenance items persist only the durable relative object id and
        root label; the on-disk source path is intentionally re-derived on each
        apply/resume so a moved-root deployment can resolve the same files at a
        new absolute path.  If a file is no longer present, the item is left
        untouched and apply will report it as a blocking issue.
        """

        # Apply-time roots take precedence (moved-root support).
        root_by_label: dict[str, Path] = {}
        if label_root_pairs:
            for label, root in label_root_pairs:
                root_by_label[label] = root
        else:
            return {}

        result: dict[str, Path] = {}
        run_items, _ = await self._metadata_store.list_artifact_maintenance_items(
            apply_run_id, limit=_MAX_ITEMS_PER_RUN
        )
        for item in run_items:
            root_label = item.root_label
            if not root_label:
                continue
            root_path = root_by_label.get(root_label)
            if root_path is None:
                continue
            legacy_rel = _strip_target_key_prefix(
                relative_object_id=item.relative_object_id,
                prefix=self._prefix,
                root_label=root_label,
            )
            if legacy_rel is None:
                continue
            candidate = root_path / legacy_rel
            try:
                if candidate.exists() and candidate.is_file():
                    result[item.item_key] = candidate
            except OSError:
                continue
        return result

    def _resolve_document_binding(
        self,
        item: ArtifactMaintenanceItemRecord,
        resolver: Callable[[str, str], tuple[str, str, str] | None],
    ) -> tuple[str, str, str] | None:
        """Look up the (kb_id, workspace, document_id) for one item.

        The resolver receives ``(root_label, legacy_relative_path)``.  The
        legacy rel is recovered by stripping the
        ``<prefix>/migrate/<label>/`` prefix from the durable
        ``relative_object_id``.
        """

        if not item.root_label:
            return None
        legacy_rel = _strip_target_key_prefix(
            relative_object_id=item.relative_object_id,
            prefix=self._prefix,
            root_label=item.root_label,
        )
        if legacy_rel is None:
            return None
        return resolver(item.root_label, legacy_rel)

    async def _advance_item(
        self,
        *,
        item: ArtifactMaintenanceItemRecord,
        run_lease_token: str,
        local_path: Path | None,
        document_binding: tuple[str, str, str] | None = None,
    ) -> None:
        """Drive one item from its current state through ``verified``."""

        if item.state in _TERMINAL_ITEM_STATES:
            return

        if item.state == "planned":
            if local_path is None:
                await self._fail_item(
                    item, run_lease_token, "legacy_source_unavailable"
                )
                return
            await self._do_upload(item, local_path, run_lease_token)
            # Refresh item after transition.
            item = await self._metadata_store.get_artifact_maintenance_item(
                item.run_id, item.item_key
            )

        if item.state == "uploaded":
            await self._do_apply(item, run_lease_token, document_binding)
            item = await self._metadata_store.get_artifact_maintenance_item(
                item.run_id, item.item_key
            )

        if item.state == "applied":
            await self._do_verify(item, run_lease_token)

    async def _do_upload(
        self,
        item: ArtifactMaintenanceItemRecord,
        local_path: Path,
        run_lease_token: str,
    ) -> None:
        fd, _ = _open_no_follow_and_validate(local_path)
        try:
            checksum, size = _compute_sha256_and_size(fd)
        finally:
            os.close(fd)
        if (
            item.expected_checksum is not None and checksum != item.expected_checksum
        ) or (
            item.expected_size_bytes is not None and size != item.expected_size_bytes
        ):
            await self._fail_item(item, run_lease_token, "checksum_mismatch")
            return
        key = item.relative_object_id
        try:
            object_uri, _created = await self._object_storage.upload_file_if_absent(
                local_path,
                key=key,
                content_type="application/octet-stream",
                expected_sha256=checksum,
            )
        except Exception as exc:
            await self._fail_item(item, run_lease_token, "object_upload_failed")
            raise MigrationApplyGuardError(
                f"object upload failed for item: {_redact_value(exc)}"
            ) from exc

        expected_uri = self._object_storage.object_uri_for_key(key)
        if object_uri != expected_uri:
            await self._fail_item(item, run_lease_token, "object_uri_mismatch")
            raise MigrationApplyGuardError(
                "Uploaded object URI did not match the planned authority"
            )
        await self._transition_item(item, run_lease_token, "planned", "uploaded")

    async def _do_apply(
        self,
        item: ArtifactMaintenanceItemRecord,
        run_lease_token: str,
        document_binding: tuple[str, str, str] | None,
    ) -> None:
        # Document pointer swap (source_object_uri + source_generation_id).
        # The atomic commit is the store's row update; the maintenance item
        # transition is the durable record of progress.
        kb_id = item.kb_id
        document_id = item.document_id
        if document_binding is not None:
            kb_id, _workspace, document_id = document_binding
        if kb_id and document_id:
            try:
                await self._metadata_store.get_document(kb_id, document_id)
            except Exception as exc:
                await self._fail_item(item, run_lease_token, "document_lookup_failed")
                raise MigrationApplyGuardError(
                    "Document lookup failed during apply"
                ) from exc
            source_uri = self._object_storage.object_uri_for_key(
                item.relative_object_id
            )
            source_generation_id = f"mig_{item.expected_checksum}"
            metadata_patch = {
                "source_object_uri": source_uri,
                "source_generation_id": source_generation_id,
            }
            try:
                await self._metadata_store.update_document(
                    kb_id,
                    document_id,
                    metadata_patch=metadata_patch,
                )
            except Exception as exc:
                await self._fail_item(
                    item, run_lease_token, "metadata_pointer_commit_failed"
                )
                raise MigrationApplyGuardError(
                    "Metadata pointer commit failed during apply"
                ) from exc
        await self._transition_item(item, run_lease_token, "uploaded", "applied")

    async def _do_verify(
        self,
        item: ArtifactMaintenanceItemRecord,
        run_lease_token: str,
    ) -> None:
        object_uri = self._object_storage.object_uri_for_key(item.relative_object_id)
        try:
            readback = await self._object_storage.inspect_object(object_uri)
        except Exception as exc:
            await self._fail_item(item, run_lease_token, "object_readback_failed")
            raise MigrationApplyGuardError(
                "Object readback failed during verify"
            ) from exc
        if not readback.present or readback.stat is None:
            await self._fail_item(item, run_lease_token, "object_absent")
            raise MigrationApplyGuardError(
                "Verified state requires metadata-only inspect_object proof"
            )
        if item.expected_size_bytes is not None:
            if readback.stat.size != item.expected_size_bytes:
                await self._fail_item(item, run_lease_token, "size_proof_mismatch")
                raise MigrationApplyGuardError(
                    "Verified state requires matching object size proof"
                )
        await self._transition_item(item, run_lease_token, "applied", "verified")

    async def _transition_item(
        self,
        item: ArtifactMaintenanceItemRecord,
        run_lease_token: str,
        expected_state: ArtifactMaintenanceItemState,
        new_state: ArtifactMaintenanceItemState,
    ) -> ArtifactMaintenanceItemRecord:
        return await self._metadata_store.transition_artifact_maintenance_item(
            item.run_id,
            item.item_key,
            expected_state=expected_state,
            new_state=new_state,
            run_lease_token=run_lease_token,
            increment_attempt=(new_state in {"verified"}),
        )

    async def _fail_item(
        self,
        item: ArtifactMaintenanceItemRecord,
        run_lease_token: str,
        error_code: str,
    ) -> None:
        try:
            await self._metadata_store.transition_artifact_maintenance_item(
                item.run_id,
                item.item_key,
                expected_state=item.state,
                new_state="blocked",
                run_lease_token=run_lease_token,
                error_code=error_code,
            )
        except ArtifactLifecycleStateError:
            # Item advanced concurrently; nothing more to do here.
            pass

    # -- Record builders --------------------------------------------------

    def _mint_run_id(self, kind: str, *, parent_id: str | None = None) -> str:
        token = secrets.token_urlsafe(9)
        suffix = re.sub(r"[^A-Za-z0-9._-]", "", token)[:12]
        if kind == "plan":
            return f"mig-plan-{suffix}"
        if parent_id:
            parent_suffix = parent_id.rsplit("-", 1)[-1]
            return f"mig-apply-{parent_suffix}"
        return f"mig-apply-{suffix}"

    def _spec_to_item(
        self, run_id: str, ordinal: int, spec: MigrationItemSpec
    ) -> ArtifactMaintenanceItemRecord:
        target_uri = self._object_storage.object_uri_for_key(spec.relative_object_id)
        target_uri_digest = artifact_target_uri_digest(target_uri)
        payload = {"migration_stage": "plan"}
        item_key = artifact_maintenance_item_key(
            run_id=run_id,
            subject_kind=spec.subject_kind,
            subject_id=spec.subject_id,
            kb_id=spec.kb_id,
            kb_generation=spec.kb_generation,
            workspace=spec.workspace,
            document_id=spec.document_id,
            artifact_id=spec.artifact_id,
            logical_group_id=f"migrate:{spec.root_label}",
            relative_object_id=spec.relative_object_id,
            root_label=spec.root_label,
            expected_checksum=spec.expected_checksum,
            expected_size_bytes=spec.expected_size_bytes,
            target_uri_authority=self._target_uri_authority,
            target_uri_digest=target_uri_digest,
            payload_json=payload,
        )
        return ArtifactMaintenanceItemRecord(
            id=f"{run_id}-item-{ordinal:04d}",
            run_id=run_id,
            item_key=item_key,
            state="planned",
            ordinal=ordinal,
            subject_kind=spec.subject_kind,
            subject_id=spec.subject_id,
            kb_id=spec.kb_id,
            kb_generation=spec.kb_generation,
            workspace=spec.workspace,
            document_id=spec.document_id,
            artifact_id=spec.artifact_id,
            logical_group_id=f"migrate:{spec.root_label}",
            relative_object_id=spec.relative_object_id,
            root_label=spec.root_label,
            expected_checksum=spec.expected_checksum,
            expected_size_bytes=spec.expected_size_bytes,
            target_uri_authority=self._target_uri_authority,
            target_uri_digest=target_uri_digest,
            payload_json=payload,
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
        )


# Needed by Iterable typing.
from typing import Callable, Iterable  # noqa: E402


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


def _build_scope_payload(
    label_root_pairs: Sequence[tuple[str, Path]],
) -> dict[str, Any]:
    """Build a durable-safe scope payload (no absolute paths).

    Absolute local roots are intentionally absent: only the root label, a
    content fingerprint of its absolute path bytes, the file count, and the
    operation kind are persisted. Apply/resume re-resolves local paths from
    the LABEL=/absolute/root arguments passed at apply time (which support
    moved-root deployments).
    """

    roots: dict[str, dict[str, Any]] = {}
    for label, root in label_root_pairs:
        path_fingerprint = (
            "sha256:" + hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:32]
        )
        roots[label] = {
            "path_fingerprint": path_fingerprint,
            "kind": "legacy_root",
        }
    return {
        "roots": roots,
        "object_prefix": _DEFAULT_OBJECT_PREFIX,
        "kind": _MAINTENANCE_KIND,
    }


def _fingerprint_json(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lightrag-migrate-artifacts-to-object",
        description=(
            "Migrate LightRAG on-disk artifacts into durable object-storage "
            "authority using a dry-run-first, resumable plan."
        ),
    )
    parser.add_argument(
        "roots",
        nargs="+",
        metavar="LABEL=/absolute/root",
        help=(
            "One or more explicit legacy roots, each as LABEL=/absolute/root. "
            "LABEL is a display-safe identifier; the path must be absolute and "
            "free of symlink/traversal components."
        ),
    )
    parser.add_argument(
        "--working-dir",
        required=True,
        help="Canonical LightRAG working directory containing metadata.sqlite3.",
    )
    parser.add_argument(
        "--object-storage-endpoint",
        default=None,
        help="S3/MinIO endpoint URL (overrides LIGHTRAG_OBJECT_STORAGE_ENDPOINT).",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="Target bucket name (overrides LIGHTRAG_OBJECT_STORAGE_BUCKET).",
    )
    parser.add_argument(
        "--prefix",
        default=_DEFAULT_OBJECT_PREFIX,
        help="Object key prefix (default: 'kb').",
    )
    parser.add_argument(
        "--metadata-backend",
        choices=("sqlite", "postgres"),
        default=None,
        help="Force metadata backend (default: derived from LIGHTRAG_KB_METADATA_BACKEND).",
    )
    parser.add_argument(
        "--postgres-dsn",
        default=None,
        help="PostgreSQL DSN when --metadata-backend=postgres (or env equivalent).",
    )
    parser.add_argument(
        "--use-ssl",
        action="store_true",
        default=None,
        help="Use TLS for the S3 endpoint (overrides LIGHTRAG_OBJECT_STORAGE_USE_SSL).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Force dry-run plan creation even when --plan-id/--yes are present. "
            "Default behaviour: plan creation unless both --plan-id and --yes are set."
        ),
    )
    parser.add_argument(
        "--plan-id",
        default=None,
        help="Apply a previously-created plan (requires --yes).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm destructive apply (required with --plan-id).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an in-progress apply run (re-claims after lease expiry).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary.",
    )
    return parser


def _print_summary(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    mode = payload.get("mode", "plan")
    if mode == "plan":
        print(f"Migration plan created: {payload['plan_id']}")
        print(f"  item_count: {payload['item_count']}")
        print(f"  metadata_backend: {payload['metadata_backend']}")
    else:
        print(f"Migration apply summary for plan {payload['plan_id']}")
        print(f"  apply_run_id: {payload['apply_run_id']}")
        print(f"  items_total: {payload['items_total']}")
        print(f"  items_verified: {payload['items_verified']}")
        print(f"  items_skipped: {payload['items_skipped']}")
        print(f"  items_failed: {payload['items_failed']}")
        print(f"  items_blocked: {payload['items_blocked']}")
        for issue in payload.get("issues", []):
            print(f"  issue: {issue}")


async def _run_plan(args: argparse.Namespace) -> dict[str, Any]:
    backend = _resolve_metadata_backend(args)
    metadata_store = _metadata_store_from_args(args, backend)
    object_storage = _object_storage_from_args(args)
    bucket = _resolve_bucket_name(args, object_storage)
    await metadata_store.initialize()
    try:
        await object_storage.initialize()
        try:
            pairs = [_parse_label_root_spec(spec) for spec in args.roots]
            migrator = ArtifactObjectMigrator(
                metadata_store=metadata_store,
                object_storage=object_storage,
                metadata_backend=backend,
                bucket=bucket,
                prefix=args.prefix,
            )
            summary = await migrator.create_plan(pairs)
            return {
                "mode": "plan",
                **_redact_mapping(summary.to_audit_dict()),
            }
        finally:
            await object_storage.close()
    finally:
        await _close_store(metadata_store)


async def _run_apply(args: argparse.Namespace) -> dict[str, Any]:
    backend = _resolve_metadata_backend(args)
    metadata_store = _metadata_store_from_args(args, backend)
    object_storage = _object_storage_from_args(args)
    bucket = _resolve_bucket_name(args, object_storage)
    await metadata_store.initialize()
    try:
        await object_storage.initialize()
        try:
            migrator = ArtifactObjectMigrator(
                metadata_store=metadata_store,
                object_storage=object_storage,
                metadata_backend=backend,
                bucket=bucket,
                prefix=args.prefix,
            )
            pairs = [_parse_label_root_spec(spec) for spec in args.roots]
            summary = await migrator.apply_plan(
                args.plan_id,
                label_root_pairs=pairs,
                resume=args.resume,
            )
            return {
                "mode": "apply",
                **_redact_mapping(summary.to_audit_dict()),
            }
        finally:
            await object_storage.close()
    finally:
        await _close_store(metadata_store)


async def _close_store(metadata_store: Any) -> None:
    close = getattr(metadata_store, "close", None)
    if close is None:
        return
    try:
        await close()
    except Exception:
        return


async def _async_main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.plan_id and not args.yes:
        parser.error("--plan-id requires --yes to confirm apply")
    if args.yes and not args.plan_id:
        parser.error("--yes requires --plan-id")
    if args.resume and not args.plan_id:
        parser.error("--resume requires --plan-id")
    try:
        if args.plan_id and args.yes and not args.dry_run:
            payload = await _run_apply(args)
        else:
            payload = await _run_plan(args)
    except MigrationError as exc:
        sys.stderr.write(f"migration failed: {_redact_value(exc)}\n")
        raise SystemExit(2)
    except KeyboardInterrupt:  # pragma: no cover - operator-driven
        sys.stderr.write(
            "migration interrupted; resume with --plan-id --yes --resume\n"
        )
        raise SystemExit(130)
    _print_summary(payload, as_json=args.json)
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
