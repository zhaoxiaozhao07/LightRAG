"""Durable, credential-free contracts for artifact lifecycle authority.

This module intentionally contains no object-storage side effects.  It defines
the immutable records and validation shared by the SQLite and PostgreSQL
metadata stores.  Durable values are normalized before they can be written so
cleanup and maintenance rows remain safe to expose in audit and health output.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Mapping, Sequence, cast
from urllib.parse import quote, unquote, urlsplit, urlunsplit

ArtifactCleanupReason = Literal[
    "replace",
    "document_delete",
    "kb_delete",
    "staging_terminal",
    "migration_compensation",
    "orphan_reconcile",
]
ArtifactCleanupTargetKind = Literal["object", "prefix"]
ArtifactCleanupTargetNamespace = Literal[
    "source", "legacy_source", "artifact", "staging", "workspace"
]
ArtifactCleanupDisposition = Literal["delete", "retain"]
ArtifactCleanupStatus = Literal["retained", "pending", "leased", "blocked", "succeeded"]

ArtifactMaintenanceRunKind = Literal["migration", "orphan_reconcile"]
ArtifactMaintenanceRunMode = Literal["dry_run", "apply"]
ArtifactMaintenanceMetadataBackend = Literal["sqlite", "postgres"]
ArtifactMaintenanceRunStatus = Literal[
    "planned", "running", "waiting_cleanup", "succeeded", "failed", "cancelled"
]
ArtifactMaintenanceItemState = Literal[
    "planned", "uploaded", "applied", "verified", "skipped", "blocked", "failed"
]
ArtifactRecoveryStatus = Literal["parsed", "ready"]

ARTIFACT_CLEANUP_OPERATION_VERSION = "artifact-cleanup:v1"
ARTIFACT_MAINTENANCE_OPERATION_VERSION = "artifact-maintenance:v2"
ARTIFACT_MAINTENANCE_ITEM_VERSION = "artifact-maintenance-item:v2"
ARTIFACT_LIFECYCLE_MAX_PAGE_SIZE = 500
ARTIFACT_RECOVERY_MAX_PAGE_SIZE = 200
ARTIFACT_LIFECYCLE_MAX_COUNT = 2**63 - 1
ARTIFACT_CLEANUP_MIN_AUDIT_RETENTION_DAYS = 30

_CLEANUP_REASONS = frozenset(
    {
        "replace",
        "document_delete",
        "kb_delete",
        "staging_terminal",
        "migration_compensation",
        "orphan_reconcile",
    }
)
_TARGET_KINDS = frozenset({"object", "prefix"})
_TARGET_NAMESPACES = frozenset(
    {"source", "legacy_source", "artifact", "staging", "workspace"}
)
_DISPOSITIONS = frozenset({"delete", "retain"})
_CLEANUP_STATUSES = frozenset({"retained", "pending", "leased", "blocked", "succeeded"})
_RUN_KINDS = frozenset({"migration", "orphan_reconcile"})
_RUN_MODES = frozenset({"dry_run", "apply"})
_METADATA_BACKENDS = frozenset({"sqlite", "postgres"})
_RUN_STATUSES = frozenset(
    {"planned", "running", "waiting_cleanup", "succeeded", "failed", "cancelled"}
)
_ITEM_STATES = frozenset(
    {"planned", "uploaded", "applied", "verified", "skipped", "blocked", "failed"}
)
_RECOVERY_STATUSES = frozenset({"parsed", "ready"})

_CONTROL_CHARACTER_RE = re.compile(r"[\x00-\x1f\x7f]")
_PERCENT_ESCAPE_RE = re.compile(r"%[0-9a-fA-F]{2}")
_ENCODED_PATH_SEPARATOR_RE = re.compile(r"%(?:2f|5c)", re.IGNORECASE)
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[a-zA-Z]:[\\/]")
_ERROR_CODE_RE = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ROOT_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DSN_TEXT_RE = re.compile(
    r"(?:^|\s)(?:host|hostaddr|port|dbname|database|user|password|sslmode)\s*=",
    re.IGNORECASE,
)
_DURABLE_SECRET_RE = re.compile(
    r"(?:aws[_-]?access[_-]?key(?:[_-]?id)?|"
    r"aws[_-]?secret[_-]?access[_-]?key|"
    r"access[_-]?key(?:[_-]?id)?|secret[_-]?(?:access[_-]?)?key|"
    r"x-amz-(?:credential|signature|security-token)|"
    r"x-goog-(?:credential|signature)|"
    r"signature=|credential=)",
    re.IGNORECASE,
)
_ACCESS_KEY_VALUE_RE = re.compile(
    r"(?:A3T|AKIA|ASIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASCA)[A-Z0-9]{16}"
)
_SCRATCH_RE = re.compile(
    r"(?:\.lightrag-scratch|(?:^|[/_.-])scratch(?:[/_.-]|$)|"
    r"\.sync-staging|\.replace-staging)",
    re.IGNORECASE,
)
_UNSAFE_JSON_KEY_RE = re.compile(
    r"(?:password|passwd|secret|credential|authorization|access[_-]?key|"
    r"secret[_-]?key|presigned|pre_signed|stdout|stderr|raw[_-]?output|"
    r"full[_-]?output|absolute[_-]?root|legacy[_-]?root)",
    re.IGNORECASE,
)
_SAFE_ERROR_ENTITIES = frozenset(
    {
        "artifact lifecycle record",
        "artifact cleanup manifest",
        "artifact cleanup manifest enqueue",
        "artifact cleanup manifest release scope",
        "artifact maintenance run",
        "artifact maintenance item",
        "artifact maintenance parent plan",
        "artifact recovery cursor",
        "artifact recovery generation",
    }
)


def _safe_error_entity(value: str) -> str:
    return value if value in _SAFE_ERROR_ENTITIES else "artifact lifecycle record"


class ArtifactLifecycleError(RuntimeError):
    """Base class whose messages are safe for durable audit output."""


class ArtifactLifecycleConflictError(ArtifactLifecycleError):
    """A deterministic key already owns a different durable operation."""

    def __init__(self, entity_type: str = "artifact lifecycle record") -> None:
        self.entity_type = _safe_error_entity(entity_type)
        super().__init__(
            f"{self.entity_type} conflicts with existing durable authority"
        )


class ArtifactLifecycleLeaseError(ArtifactLifecycleError):
    """A lease token or lease owner no longer matches durable authority."""

    def __init__(self, entity_type: str = "artifact lifecycle record") -> None:
        self.entity_type = _safe_error_entity(entity_type)
        super().__init__(f"{self.entity_type} lease ownership does not match")


class ArtifactLifecycleStateError(ArtifactLifecycleError):
    """A fenced lifecycle transition did not match the current state."""

    def __init__(self, entity_type: str = "artifact lifecycle record") -> None:
        self.entity_type = _safe_error_entity(entity_type)
        super().__init__(f"{self.entity_type} state transition does not match")


class ArtifactLifecycleNotFoundError(ArtifactLifecycleError):
    """The requested lifecycle authority row does not exist."""

    def __init__(self, entity_type: str = "artifact lifecycle record") -> None:
        self.entity_type = _safe_error_entity(entity_type)
        super().__init__(f"{self.entity_type} was not found")


class ArtifactRecoveryGenerationError(ArtifactLifecycleConflictError):
    """A stale KB generation attempted to reserve a recovery page."""

    def __init__(self) -> None:
        super().__init__("artifact recovery generation")


def normalize_utc_datetime(value: str | datetime, *, field_name: str) -> str:
    """Return one canonical UTC ISO-8601 value or reject a non-UTC timestamp."""

    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        candidate = value.strip()
        if not candidate or candidate != value or _CONTROL_CHARACTER_RE.search(value):
            raise ValueError(f"{field_name} must be a normalized UTC datetime")
        try:
            parsed = datetime.fromisoformat(
                candidate[:-1] + "+00:00" if candidate.endswith("Z") else candidate
            )
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a valid UTC datetime") from exc
    else:
        raise ValueError(f"{field_name} must be a UTC datetime")
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must include the UTC timezone")
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")


def normalize_artifact_target_uri(value: str) -> str:
    """Normalize a credential-free authority URI used by cleanup manifests."""

    candidate = _normalized_text(value, field_name="target_uri", max_length=8192)
    _assert_durable_safe_text(candidate, field_name="target_uri")
    parsed = urlsplit(candidate)
    if not parsed.scheme or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(
            "target_uri must be an authority URI without query or fragment"
        )
    scheme, netloc = _normalize_artifact_uri_authority(parsed, field_name="target_uri")
    if parsed.path in {"", "/"} or not parsed.path.startswith("/"):
        raise ValueError("target_uri must identify a non-empty object or prefix")
    if parsed.path.startswith("//") or "//" in parsed.path[1:]:
        raise ValueError("target_uri contains an unnormalized path")
    if _ENCODED_PATH_SEPARATOR_RE.search(parsed.path):
        raise ValueError("target_uri contains an encoded path separator")
    decoded_once = unquote(parsed.path)
    if "\\" in decoded_once:
        raise ValueError("target_uri contains an unnormalized path")
    segments = decoded_once.split("/")
    if any(segment in {".", ".."} for segment in segments):
        raise ValueError("target_uri contains a non-canonical path segment")
    _assert_durable_safe_text(decoded_once, field_name="target_uri")
    path = quote(decoded_once, safe="/-._~")
    path = _PERCENT_ESCAPE_RE.sub(lambda match: match.group(0).upper(), path)
    return urlunsplit((scheme, netloc, path, "", ""))


def normalize_artifact_relative_object_id(value: str) -> str:
    """Return one canonical relative object identity with URI-path semantics."""

    candidate = _normalized_text(
        value, field_name="relative_object_id", max_length=4096
    )
    _assert_durable_safe_text(candidate, field_name="relative_object_id")
    if (
        candidate.startswith("/")
        or _WINDOWS_ABSOLUTE_RE.match(candidate)
        or "://" in candidate
        or "\\" in candidate
        or candidate.startswith("//")
        or "//" in candidate
        or _ENCODED_PATH_SEPARATOR_RE.search(candidate)
    ):
        raise ValueError("relative_object_id must be a normalized relative identity")
    decoded_once = unquote(candidate)
    _assert_durable_safe_text(decoded_once, field_name="relative_object_id")
    if "\\" in decoded_once:
        raise ValueError("relative_object_id must not contain a backslash")
    segments = decoded_once.split("/")
    if any(not segment or segment in {".", ".."} for segment in segments):
        raise ValueError("relative_object_id contains a non-canonical segment")
    normalized = quote(decoded_once, safe="/-._~")
    return _PERCENT_ESCAPE_RE.sub(lambda match: match.group(0).upper(), normalized)


def normalize_artifact_root_label(value: str) -> str:
    """Validate a display-safe storage-root label without accepting a path."""

    candidate = _normalized_text(value, field_name="root_label", max_length=128)
    _assert_durable_safe_text(candidate, field_name="root_label")
    if (
        _ROOT_LABEL_RE.fullmatch(candidate) is None
        or candidate in {".", ".."}
        or "/" in candidate
        or "\\" in candidate
        or ":" in candidate
    ):
        raise ValueError("root_label must be a safe non-path label")
    return candidate


def normalize_artifact_target_uri_authority(value: str) -> str:
    """Canonicalize only a target URI's scheme and host/bucket authority."""

    candidate = _normalized_text(
        value, field_name="target_uri_authority", max_length=2048
    )
    _assert_durable_safe_text(candidate, field_name="target_uri_authority")
    parsed = urlsplit(candidate)
    if (
        not parsed.scheme
        or not parsed.netloc
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "target_uri_authority must contain only a URI scheme and authority"
        )
    scheme, netloc = _normalize_artifact_uri_authority(
        parsed, field_name="target_uri_authority"
    )
    return urlunsplit((scheme, netloc, "", "", ""))


def normalize_artifact_target_uri_digest(value: str) -> str:
    """Validate a canonical lowercase SHA-256 digest without accepting a URI."""

    candidate = _normalized_text(value, field_name="target_uri_digest", max_length=64)
    if _HEX_SHA256_RE.fullmatch(candidate) is None:
        raise ValueError("target_uri_digest must be 64 lowercase hexadecimal digits")
    return candidate


def artifact_target_uri_digest(target_uri: str) -> str:
    """Hash a canonical full target URI so the URI itself need not be persisted."""

    canonical = normalize_artifact_target_uri(target_uri)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sanitize_artifact_lifecycle_error_code(
    value: str | None,
    *,
    fallback: str = "artifact_lifecycle_error",
) -> str | None:
    """Return a bounded diagnostic code without persisting exception text."""

    if value is None:
        return None
    if (
        not isinstance(fallback, str)
        or _ERROR_CODE_RE.fullmatch(fallback) is None
        or _DURABLE_SECRET_RE.search(fallback)
        or _SCRATCH_RE.search(fallback)
    ):
        raise ValueError("fallback must be a normalized safe error code")
    if not isinstance(value, str):
        return fallback
    candidate = value.strip().lower().replace(" ", "_")
    if (
        candidate != value
        or _ERROR_CODE_RE.fullmatch(candidate) is None
        or _DURABLE_SECRET_RE.search(candidate)
        or _SCRATCH_RE.search(candidate)
    ):
        return fallback
    return candidate


def canonical_safe_json(
    value: str | Mapping[str, Any] | Sequence[Any] | None,
    *,
    field_name: str,
    allow_none: bool = False,
) -> str | None:
    """Validate and canonicalize bounded JSON without paths or secret output."""

    if value is None:
        if allow_none:
            return None
        value = {}
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be valid JSON") from exc
    else:
        decoded = value
    _validate_safe_json_value(decoded, field_name=field_name, depth=0)
    try:
        encoded = json.dumps(
            decoded,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain JSON values") from exc
    if len(encoded.encode("utf-8")) > 64 * 1024:
        raise ValueError(f"{field_name} exceeds the durable JSON size limit")
    return encoded


def canonical_maintenance_payload_json(
    value: str | Mapping[str, Any] | Sequence[Any] | None,
) -> str:
    """Canonicalize a small, non-authoritative maintenance options payload."""

    if value is None:
        decoded: Any = {}
    elif isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("payload_json must be valid JSON") from exc
    else:
        decoded = value
    _validate_safe_json_value(decoded, field_name="payload_json", depth=0)
    _validate_maintenance_payload_value(decoded, depth=0)
    try:
        encoded = json.dumps(
            decoded,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("payload_json must contain JSON values") from exc
    if len(encoded.encode("utf-8")) > 8 * 1024:
        raise ValueError("payload_json exceeds the maintenance options size limit")
    return encoded


def artifact_cleanup_idempotency_key(
    *,
    reason: ArtifactCleanupReason,
    kb_id: str,
    kb_generation: str,
    workspace: str,
    target_kind: ArtifactCleanupTargetKind,
    target_namespace: ArtifactCleanupTargetNamespace,
    target_uri: str,
    document_id: str | None = None,
    artifact_id: str | None = None,
    source_generation_id: str | None = None,
    operation_version: str = ARTIFACT_CLEANUP_OPERATION_VERSION,
) -> str:
    """SHA-256 of the canonical cleanup operation and exact ownership scope."""

    canonical = {
        "artifact_id": _optional_identity(artifact_id, "artifact_id"),
        "document_id": _optional_identity(document_id, "document_id"),
        "kb_generation": _identity(kb_generation, "kb_generation"),
        "kb_id": _identity(kb_id, "kb_id"),
        "operation_version": _identity(operation_version, "operation_version"),
        "reason": _literal(reason, _CLEANUP_REASONS, "reason"),
        "source_generation_id": _optional_identity(
            source_generation_id, "source_generation_id"
        ),
        "target_kind": _literal(target_kind, _TARGET_KINDS, "target_kind"),
        "target_namespace": _literal(
            target_namespace, _TARGET_NAMESPACES, "target_namespace"
        ),
        "target_uri": normalize_artifact_target_uri(target_uri),
        "workspace": _identity(workspace, "workspace"),
    }
    encoded = json.dumps(canonical, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def artifact_maintenance_run_key(
    *,
    kind: ArtifactMaintenanceRunKind,
    mode: ArtifactMaintenanceRunMode,
    metadata_backend: ArtifactMaintenanceMetadataBackend,
    parent_plan_id: str | None,
    backend_fingerprint: str,
    scope_fingerprint: str,
    config_fingerprint: str,
) -> str:
    canonical = {
        "backend_fingerprint": _identity(backend_fingerprint, "backend_fingerprint"),
        "config_fingerprint": _identity(config_fingerprint, "config_fingerprint"),
        "kind": _literal(kind, _RUN_KINDS, "kind"),
        "metadata_backend": _literal(
            metadata_backend, _METADATA_BACKENDS, "metadata_backend"
        ),
        "mode": _literal(mode, _RUN_MODES, "mode"),
        "operation_version": ARTIFACT_MAINTENANCE_OPERATION_VERSION,
        "parent_plan_id": _optional_identity(parent_plan_id, "parent_plan_id"),
        "scope_fingerprint": _identity(scope_fingerprint, "scope_fingerprint"),
    }
    encoded = json.dumps(canonical, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def artifact_maintenance_item_key(
    *,
    run_id: str,
    subject_kind: str,
    subject_id: str,
    kb_id: str | None,
    kb_generation: str | None,
    workspace: str | None,
    document_id: str | None,
    artifact_id: str | None,
    logical_group_id: str,
    relative_object_id: str,
    root_label: str | None,
    expected_checksum: str | None,
    expected_size_bytes: int | None,
    target_uri_authority: str | None,
    target_uri_digest: str | None,
    payload_json: str | Mapping[str, Any] | Sequence[Any] | None = None,
) -> str:
    canonical = {
        "artifact_id": _optional_maintenance_authority_identity(
            artifact_id, "artifact_id"
        ),
        "document_id": _optional_maintenance_authority_identity(
            document_id, "document_id"
        ),
        "expected_checksum": _optional_maintenance_checksum(expected_checksum),
        "expected_size_bytes": (
            None
            if expected_size_bytes is None
            else _bounded_count(expected_size_bytes, "expected_size_bytes")
        ),
        "kb_generation": _optional_maintenance_authority_identity(
            kb_generation, "kb_generation"
        ),
        "kb_id": _optional_maintenance_authority_identity(kb_id, "kb_id"),
        "logical_group_id": _maintenance_authority_identity(
            logical_group_id, "logical_group_id"
        ),
        "operation_version": ARTIFACT_MAINTENANCE_ITEM_VERSION,
        "payload_json": canonical_maintenance_payload_json(payload_json),
        "relative_object_id": normalize_artifact_relative_object_id(relative_object_id),
        "root_label": (
            None if root_label is None else normalize_artifact_root_label(root_label)
        ),
        "run_id": _identity(run_id, "run_id"),
        "subject_id": _maintenance_authority_identity(subject_id, "subject_id"),
        "subject_kind": _maintenance_authority_identity(subject_kind, "subject_kind"),
        "target_uri_authority": (
            None
            if target_uri_authority is None
            else normalize_artifact_target_uri_authority(target_uri_authority)
        ),
        "target_uri_digest": (
            None
            if target_uri_digest is None
            else normalize_artifact_target_uri_digest(target_uri_digest)
        ),
        "workspace": _optional_maintenance_authority_identity(workspace, "workspace"),
    }
    encoded = json.dumps(canonical, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactCleanupManifestRecord:
    id: str
    idempotency_key: str
    manifest_group_id: str
    kb_id: str
    kb_generation: str
    workspace: str
    reason: ArtifactCleanupReason
    target_kind: ArtifactCleanupTargetKind
    target_namespace: ArtifactCleanupTargetNamespace
    disposition: ArtifactCleanupDisposition
    status: ArtifactCleanupStatus
    target_uri: str
    delete_after: str | datetime
    cleanup_deadline_at: str | datetime
    audit_retain_until: str | datetime
    next_attempt_at: str | datetime
    created_at: str | datetime
    updated_at: str | datetime
    document_id: str | None = None
    artifact_id: str | None = None
    source_generation_id: str | None = None
    origin_job_id: str | None = None
    origin_attempt_token: str | None = None
    expected_checksum: str | None = None
    expected_etag: str | None = None
    expected_version_id: str | None = None
    expected_size_bytes: int | None = None
    attempt_count: int = 0
    lease_owner: str | None = None
    lease_token: str | None = None
    lease_expires_at: str | datetime | None = None
    last_error_code: str | None = None
    last_checked_at: str | datetime | None = None
    completed_at: str | datetime | None = None

    def __post_init__(self) -> None:
        for name in (
            "id",
            "idempotency_key",
            "manifest_group_id",
            "kb_id",
            "kb_generation",
            "workspace",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        for name in (
            "document_id",
            "artifact_id",
            "source_generation_id",
            "origin_job_id",
            "origin_attempt_token",
            "lease_owner",
            "lease_token",
        ):
            object.__setattr__(
                self, name, _optional_identity(getattr(self, name), name)
            )
        object.__setattr__(
            self, "reason", _literal(self.reason, _CLEANUP_REASONS, "reason")
        )
        object.__setattr__(
            self,
            "target_kind",
            _literal(self.target_kind, _TARGET_KINDS, "target_kind"),
        )
        object.__setattr__(
            self,
            "target_namespace",
            _literal(self.target_namespace, _TARGET_NAMESPACES, "target_namespace"),
        )
        object.__setattr__(
            self,
            "disposition",
            _literal(self.disposition, _DISPOSITIONS, "disposition"),
        )
        object.__setattr__(
            self,
            "status",
            _literal(self.status, _CLEANUP_STATUSES, "status"),
        )
        object.__setattr__(
            self, "target_uri", normalize_artifact_target_uri(self.target_uri)
        )
        if self.target_kind == "prefix" and not self.target_uri.endswith("/"):
            raise ValueError("prefix cleanup targets must end with a slash")
        if self.target_kind == "object" and self.target_uri.endswith("/"):
            raise ValueError("object cleanup targets must not end with a slash")
        expected_idempotency_key = artifact_cleanup_idempotency_key(
            reason=self.reason,
            kb_id=self.kb_id,
            kb_generation=self.kb_generation,
            workspace=self.workspace,
            document_id=self.document_id,
            artifact_id=self.artifact_id,
            source_generation_id=self.source_generation_id,
            target_kind=self.target_kind,
            target_namespace=self.target_namespace,
            target_uri=self.target_uri,
        )
        if (
            _HEX_SHA256_RE.fullmatch(self.idempotency_key) is None
            or self.idempotency_key != expected_idempotency_key
        ):
            raise ValueError(
                "idempotency_key does not match the canonical cleanup operation"
            )
        for name in (
            "expected_checksum",
            "expected_etag",
            "expected_version_id",
        ):
            object.__setattr__(
                self,
                name,
                _optional_safe_text(getattr(self, name), field_name=name),
            )
        _bounded_count(self.attempt_count, "attempt_count")
        if self.expected_size_bytes is not None:
            _bounded_count(self.expected_size_bytes, "expected_size_bytes")
        for name in (
            "delete_after",
            "cleanup_deadline_at",
            "audit_retain_until",
            "next_attempt_at",
            "created_at",
            "updated_at",
        ):
            object.__setattr__(
                self,
                name,
                normalize_utc_datetime(getattr(self, name), field_name=name),
            )
        for name in ("lease_expires_at", "last_checked_at", "completed_at"):
            value = getattr(self, name)
            object.__setattr__(
                self,
                name,
                None
                if value is None
                else normalize_utc_datetime(value, field_name=name),
            )
        if cast(str, self.cleanup_deadline_at) < cast(str, self.delete_after):
            raise ValueError("cleanup_deadline_at must not precede delete_after")
        if cast(str, self.audit_retain_until) < cast(str, self.created_at):
            raise ValueError("audit_retain_until must not precede created_at")
        if cast(str, self.updated_at) < cast(str, self.created_at):
            raise ValueError("updated_at must not precede created_at")
        safe_error = sanitize_artifact_lifecycle_error_code(self.last_error_code)
        if safe_error != self.last_error_code:
            raise ValueError("last_error_code must be a normalized safe error code")
        _validate_manifest_state(self)

    @property
    def size_bytes(self) -> int | None:
        """Compatibility alias for the expected object size."""

        return self.expected_size_bytes

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArtifactCleanupManifestRecord":
        data = dict(value)
        if "size_bytes" in data and "expected_size_bytes" not in data:
            data["expected_size_bytes"] = data.pop("size_bytes")
        return cls(**data)

    def operation_payload(self) -> dict[str, Any]:
        """Fields that must match when an idempotency key is replayed."""

        return {
            "manifest_group_id": self.manifest_group_id,
            "kb_id": self.kb_id,
            "kb_generation": self.kb_generation,
            "workspace": self.workspace,
            "document_id": self.document_id,
            "artifact_id": self.artifact_id,
            "source_generation_id": self.source_generation_id,
            "origin_job_id": self.origin_job_id,
            "origin_attempt_token": self.origin_attempt_token,
            "reason": self.reason,
            "target_kind": self.target_kind,
            "target_namespace": self.target_namespace,
            "target_uri": self.target_uri,
            "expected_checksum": self.expected_checksum,
            "expected_etag": self.expected_etag,
            "expected_version_id": self.expected_version_id,
            "expected_size_bytes": self.expected_size_bytes,
            "delete_after": self.delete_after,
            "cleanup_deadline_at": self.cleanup_deadline_at,
            "audit_retain_until": self.audit_retain_until,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactMaintenanceRunRecord:
    id: str
    kind: ArtifactMaintenanceRunKind
    mode: ArtifactMaintenanceRunMode
    status: ArtifactMaintenanceRunStatus
    metadata_backend: ArtifactMaintenanceMetadataBackend
    backend_fingerprint: str
    scope_fingerprint: str
    config_fingerprint: str
    scope_json: str | Mapping[str, Any] | Sequence[Any]
    created_at: str | datetime
    updated_at: str | datetime
    parent_plan_id: str | None = None
    idempotency_key: str | None = None
    cursor_json: str | Mapping[str, Any] | Sequence[Any] | None = None
    total_items: int = 0
    planned_items: int = 0
    uploaded_items: int = 0
    applied_items: int = 0
    verified_items: int = 0
    skipped_items: int = 0
    blocked_items: int = 0
    failed_items: int = 0
    actor_id: str | None = None
    lease_owner: str | None = None
    lease_token: str | None = None
    lease_expires_at: str | datetime | None = None
    started_at: str | datetime | None = None
    completed_at: str | datetime | None = None
    last_error_code: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _identity(self.id, "id"))
        object.__setattr__(self, "kind", _literal(self.kind, _RUN_KINDS, "kind"))
        object.__setattr__(self, "mode", _literal(self.mode, _RUN_MODES, "mode"))
        object.__setattr__(
            self,
            "metadata_backend",
            _literal(self.metadata_backend, _METADATA_BACKENDS, "metadata_backend"),
        )
        object.__setattr__(
            self, "status", _literal(self.status, _RUN_STATUSES, "status")
        )
        for name in (
            "backend_fingerprint",
            "scope_fingerprint",
            "config_fingerprint",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        for name in (
            "parent_plan_id",
            "actor_id",
            "lease_owner",
            "lease_token",
        ):
            object.__setattr__(
                self, name, _optional_identity(getattr(self, name), name)
            )
        if self.mode == "dry_run" and self.parent_plan_id is not None:
            raise ValueError("dry_run maintenance records cannot have a parent plan")
        if self.mode == "apply" and self.parent_plan_id is None:
            raise ValueError("apply maintenance records require a parent plan")
        object.__setattr__(
            self,
            "scope_json",
            canonical_safe_json(self.scope_json, field_name="scope_json"),
        )
        object.__setattr__(
            self,
            "cursor_json",
            canonical_safe_json(
                self.cursor_json, field_name="cursor_json", allow_none=True
            ),
        )
        expected_key = artifact_maintenance_run_key(
            kind=self.kind,
            mode=self.mode,
            metadata_backend=self.metadata_backend,
            parent_plan_id=self.parent_plan_id,
            backend_fingerprint=self.backend_fingerprint,
            scope_fingerprint=self.scope_fingerprint,
            config_fingerprint=self.config_fingerprint,
        )
        resolved_key = self.idempotency_key or expected_key
        normalized_key = _identity(resolved_key, "idempotency_key")
        object.__setattr__(self, "idempotency_key", normalized_key)
        if (
            _HEX_SHA256_RE.fullmatch(normalized_key) is None
            or normalized_key != expected_key
        ):
            raise ValueError(
                "idempotency_key does not match the canonical maintenance operation"
            )
        for name in (
            "total_items",
            "planned_items",
            "uploaded_items",
            "applied_items",
            "verified_items",
            "skipped_items",
            "blocked_items",
            "failed_items",
        ):
            _bounded_count(getattr(self, name), name)
        for name in ("created_at", "updated_at"):
            object.__setattr__(
                self,
                name,
                normalize_utc_datetime(getattr(self, name), field_name=name),
            )
        for name in ("lease_expires_at", "started_at", "completed_at"):
            value = getattr(self, name)
            object.__setattr__(
                self,
                name,
                None
                if value is None
                else normalize_utc_datetime(value, field_name=name),
            )
        if cast(str, self.updated_at) < cast(str, self.created_at):
            raise ValueError("updated_at must not precede created_at")
        safe_error = sanitize_artifact_lifecycle_error_code(self.last_error_code)
        if safe_error != self.last_error_code:
            raise ValueError("last_error_code must be a normalized safe error code")
        _validate_maintenance_run_state(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArtifactMaintenanceRunRecord":
        return cls(**dict(value))

    def operation_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "mode": self.mode,
            "metadata_backend": self.metadata_backend,
            "parent_plan_id": self.parent_plan_id,
            "backend_fingerprint": self.backend_fingerprint,
            "scope_fingerprint": self.scope_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "scope_json": self.scope_json,
            "actor_id": self.actor_id,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactMaintenanceItemRecord:
    id: str
    run_id: str
    item_key: str
    state: ArtifactMaintenanceItemState
    ordinal: int
    subject_kind: str
    subject_id: str
    kb_id: str | None
    kb_generation: str | None
    workspace: str | None
    document_id: str | None
    artifact_id: str | None
    logical_group_id: str
    relative_object_id: str
    root_label: str | None
    expected_checksum: str | None
    expected_size_bytes: int | None
    target_uri_authority: str | None
    target_uri_digest: str | None
    payload_json: str | Mapping[str, Any] | Sequence[Any]
    created_at: str | datetime
    updated_at: str | datetime
    attempt_count: int = 0
    completed_at: str | datetime | None = None
    last_error_code: str | None = None

    def __post_init__(self) -> None:
        for name in ("id", "run_id", "item_key"):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        for name in ("subject_kind", "subject_id", "logical_group_id"):
            object.__setattr__(
                self,
                name,
                _maintenance_authority_identity(getattr(self, name), name),
            )
        for name in (
            "kb_id",
            "kb_generation",
            "workspace",
            "document_id",
            "artifact_id",
        ):
            object.__setattr__(
                self,
                name,
                _optional_maintenance_authority_identity(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "relative_object_id",
            normalize_artifact_relative_object_id(self.relative_object_id),
        )
        object.__setattr__(
            self,
            "root_label",
            None
            if self.root_label is None
            else normalize_artifact_root_label(self.root_label),
        )
        object.__setattr__(
            self,
            "expected_checksum",
            _optional_maintenance_checksum(self.expected_checksum),
        )
        if self.expected_size_bytes is not None:
            _bounded_count(self.expected_size_bytes, "expected_size_bytes")
        object.__setattr__(
            self,
            "target_uri_authority",
            None
            if self.target_uri_authority is None
            else normalize_artifact_target_uri_authority(self.target_uri_authority),
        )
        object.__setattr__(
            self,
            "target_uri_digest",
            None
            if self.target_uri_digest is None
            else normalize_artifact_target_uri_digest(self.target_uri_digest),
        )
        object.__setattr__(self, "state", _literal(self.state, _ITEM_STATES, "state"))
        _bounded_count(self.ordinal, "ordinal")
        _bounded_count(self.attempt_count, "attempt_count")
        object.__setattr__(
            self,
            "payload_json",
            canonical_maintenance_payload_json(self.payload_json),
        )
        expected_key = artifact_maintenance_item_key(
            run_id=self.run_id,
            subject_kind=self.subject_kind,
            subject_id=self.subject_id,
            kb_id=self.kb_id,
            kb_generation=self.kb_generation,
            workspace=self.workspace,
            document_id=self.document_id,
            artifact_id=self.artifact_id,
            logical_group_id=self.logical_group_id,
            relative_object_id=self.relative_object_id,
            root_label=self.root_label,
            expected_checksum=self.expected_checksum,
            expected_size_bytes=self.expected_size_bytes,
            target_uri_authority=self.target_uri_authority,
            target_uri_digest=self.target_uri_digest,
            payload_json=self.payload_json,
        )
        if self.item_key != expected_key:
            raise ValueError("item_key does not match the deterministic item payload")
        for name in ("created_at", "updated_at"):
            object.__setattr__(
                self,
                name,
                normalize_utc_datetime(getattr(self, name), field_name=name),
            )
        object.__setattr__(
            self,
            "completed_at",
            None
            if self.completed_at is None
            else normalize_utc_datetime(self.completed_at, field_name="completed_at"),
        )
        if cast(str, self.updated_at) < cast(str, self.created_at):
            raise ValueError("updated_at must not precede created_at")
        safe_error = sanitize_artifact_lifecycle_error_code(self.last_error_code)
        if safe_error != self.last_error_code:
            raise ValueError("last_error_code must be a normalized safe error code")
        if self.state in {"verified", "skipped", "blocked", "failed"}:
            if self.completed_at is None:
                raise ValueError("terminal maintenance items require completed_at")
        elif self.completed_at is not None:
            raise ValueError("non-terminal maintenance items cannot have completed_at")
        if self.state in {"blocked", "failed"} and self.last_error_code is None:
            raise ValueError(
                "blocked or failed maintenance items require an error code"
            )
        if self.completed_at is not None and str(self.completed_at) > str(
            self.updated_at
        ):
            raise ValueError("completed_at must not follow updated_at")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArtifactMaintenanceItemRecord":
        return cls(**dict(value))

    def operation_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "item_key": self.item_key,
            "ordinal": self.ordinal,
            "subject_kind": self.subject_kind,
            "subject_id": self.subject_id,
            "kb_id": self.kb_id,
            "kb_generation": self.kb_generation,
            "workspace": self.workspace,
            "document_id": self.document_id,
            "artifact_id": self.artifact_id,
            "logical_group_id": self.logical_group_id,
            "relative_object_id": self.relative_object_id,
            "root_label": self.root_label,
            "expected_checksum": self.expected_checksum,
            "expected_size_bytes": self.expected_size_bytes,
            "target_uri_authority": self.target_uri_authority,
            "target_uri_digest": self.target_uri_digest,
            "payload_json": self.payload_json,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactRecoveryCursorRecord:
    kb_id: str
    kb_generation: str
    status: ArtifactRecoveryStatus
    last_created_at: str | datetime | None
    last_document_id: str | None
    sweep: int
    version: int
    updated_at: str | datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "kb_id", _identity(self.kb_id, "kb_id"))
        object.__setattr__(
            self, "kb_generation", _identity(self.kb_generation, "kb_generation")
        )
        object.__setattr__(
            self, "status", _literal(self.status, _RECOVERY_STATUSES, "status")
        )
        object.__setattr__(
            self,
            "last_document_id",
            _optional_identity(self.last_document_id, "last_document_id"),
        )
        object.__setattr__(
            self,
            "last_created_at",
            None
            if self.last_created_at is None
            else (
                normalize_utc_datetime(
                    self.last_created_at, field_name="last_created_at"
                )
                if isinstance(self.last_created_at, datetime)
                else _validated_utc_text(
                    self.last_created_at, field_name="last_created_at"
                )
            ),
        )
        object.__setattr__(
            self,
            "updated_at",
            normalize_utc_datetime(self.updated_at, field_name="updated_at"),
        )
        _bounded_count(self.sweep, "sweep")
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise ValueError("version must be a positive integer")
        if self.version < 1 or self.version > ARTIFACT_LIFECYCLE_MAX_COUNT:
            raise ValueError("version must be a positive bounded integer")
        if (self.last_created_at is None) != (self.last_document_id is None):
            raise ValueError("recovery keyset fields must be both set or both null")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArtifactRecoveryCursorRecord":
        return cls(**dict(value))


def _validate_manifest_state(record: ArtifactCleanupManifestRecord) -> None:
    lease_values = (
        record.lease_owner,
        record.lease_token,
        record.lease_expires_at,
    )
    if record.status == "leased":
        if any(value is None for value in lease_values):
            raise ValueError("leased manifests require complete lease ownership")
        if str(record.lease_expires_at) <= str(record.updated_at):
            raise ValueError("leased manifests require a future lease expiry")
    elif any(value is not None for value in lease_values):
        raise ValueError("only leased manifests may contain lease ownership")
    if record.status == "retained":
        if record.disposition != "retain":
            raise ValueError("retained manifests require retain disposition")
    elif record.disposition != "delete":
        raise ValueError("active cleanup states require delete disposition")
    if record.status == "succeeded":
        if record.completed_at is None or record.last_checked_at is None:
            raise ValueError("succeeded manifests require completion verification")
    elif record.completed_at is not None:
        raise ValueError("only succeeded manifests may have completed_at")
    if record.status == "blocked" and record.last_error_code is None:
        raise ValueError("blocked manifests require a safe error code")
    if record.last_checked_at is not None and str(record.last_checked_at) > str(
        record.updated_at
    ):
        raise ValueError("last_checked_at must not follow updated_at")
    if record.completed_at is not None and str(record.completed_at) > str(
        record.updated_at
    ):
        raise ValueError("completed_at must not follow updated_at")


def _validate_maintenance_run_state(record: ArtifactMaintenanceRunRecord) -> None:
    lease_values = (
        record.lease_owner,
        record.lease_token,
        record.lease_expires_at,
    )
    if record.status == "running":
        if any(value is None for value in lease_values) or record.started_at is None:
            raise ValueError("running maintenance records require a complete lease")
        if str(record.lease_expires_at) <= str(record.updated_at):
            raise ValueError(
                "running maintenance records require a future lease expiry"
            )
    elif any(value is not None for value in lease_values):
        raise ValueError("only running maintenance records may contain a lease")
    terminal = record.status in {"succeeded", "failed", "cancelled"}
    if terminal != (record.completed_at is not None):
        raise ValueError("maintenance completion timestamp does not match status")
    if record.status == "failed" and record.last_error_code is None:
        raise ValueError("failed maintenance records require a safe error code")
    if record.started_at is not None and str(record.started_at) > str(
        record.updated_at
    ):
        raise ValueError("started_at must not follow updated_at")
    if record.completed_at is not None and str(record.completed_at) > str(
        record.updated_at
    ):
        raise ValueError("completed_at must not follow updated_at")


def _validate_safe_json_value(value: Any, *, field_name: str, depth: int) -> None:
    if depth > 12:
        raise ValueError(f"{field_name} exceeds the maximum JSON nesting depth")
    if value is None or isinstance(value, (bool, int)):
        if isinstance(value, int) and not isinstance(value, bool):
            _bounded_signed_integer(value, field_name)
        return
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError(f"{field_name} contains a non-finite number")
        return
    if isinstance(value, str):
        _assert_durable_safe_text(value, field_name=field_name)
        if value.startswith("/") or _WINDOWS_ABSOLUTE_RE.match(value):
            raise ValueError(f"{field_name} cannot contain an absolute local root")
        if value.lower().startswith("file://"):
            raise ValueError(f"{field_name} cannot contain a local file URI")
        if "://" in value:
            parsed = urlsplit(value)
            if (
                parsed.username is not None
                or parsed.password is not None
                or "@" in unquote(parsed.netloc)
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(f"{field_name} contains an unsafe URI")
        return
    if isinstance(value, Mapping):
        if len(value) > 1000:
            raise ValueError(f"{field_name} contains too many JSON keys")
        for key, item in value.items():
            if not isinstance(key, str) or not key or key != key.strip():
                raise ValueError(f"{field_name} contains an invalid JSON key")
            if _UNSAFE_JSON_KEY_RE.search(key):
                raise ValueError(f"{field_name} contains a forbidden durable field")
            _assert_durable_safe_text(key, field_name=field_name)
            _validate_safe_json_value(item, field_name=field_name, depth=depth + 1)
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        if len(value) > 5000:
            raise ValueError(f"{field_name} contains too many JSON items")
        for item in value:
            _validate_safe_json_value(item, field_name=field_name, depth=depth + 1)
        return
    raise ValueError(f"{field_name} contains a non-JSON value")


def _validate_maintenance_payload_value(value: Any, *, depth: int) -> None:
    if depth > 12:
        raise ValueError("payload_json exceeds the maximum JSON nesting depth")
    if value is None or isinstance(value, (bool, int, float)):
        return
    if isinstance(value, str):
        decoded_once = unquote(value)
        _assert_durable_safe_text(decoded_once, field_name="payload_json")
        for candidate in (value, decoded_once):
            lowered = candidate.lower()
            if (
                "://" in candidate
                or candidate.startswith("/")
                or _WINDOWS_ABSOLUTE_RE.match(candidate)
                or "/" in candidate
                or "\\" in candidate
                or lowered.startswith("file:")
                or _DSN_TEXT_RE.search(candidate)
            ):
                raise ValueError(
                    "payload_json cannot contain a URI, object identity, local path, or DSN"
                )
        return
    if isinstance(value, Mapping):
        forbidden_keys = {
            "artifact_id",
            "access_token",
            "api_key",
            "body",
            "bytes",
            "checksum",
            "content",
            "data",
            "document_id",
            "dsn",
            "expected_checksum",
            "expected_size_bytes",
            "kb_generation",
            "kb_id",
            "legacy_root",
            "local_path",
            "logical_group_id",
            "object_id",
            "object_key",
            "relative_object_id",
            "root_label",
            "refresh_token",
            "size_bytes",
            "target_uri",
            "target_uri_authority",
            "target_uri_digest",
            "target_url",
            "token",
            "workspace",
        }
        forbidden_suffixes = (
            "_credential",
            "_dsn",
            "_path",
            "_password",
            "_root",
            "_secret",
            "_token",
            "_uri",
            "_url",
        )
        for key, item in value.items():
            normalized_key = key.lower().replace("-", "_")
            if (
                normalized_key in forbidden_keys
                or normalized_key.endswith(forbidden_suffixes)
                or "presigned" in normalized_key
                or "pre_signed" in normalized_key
            ):
                raise ValueError(
                    "payload_json contains authoritative or unsafe maintenance data"
                )
            _validate_maintenance_payload_value(item, depth=depth + 1)
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            _validate_maintenance_payload_value(item, depth=depth + 1)
        return


def _normalize_artifact_uri_authority(
    parsed: Any,
    *,
    field_name: str,
) -> tuple[str, str]:
    decoded_netloc = unquote(parsed.netloc)
    _assert_durable_safe_text(decoded_netloc, field_name=field_name)
    _assert_durable_safe_text(unquote(decoded_netloc), field_name=field_name)
    if (
        parsed.username is not None
        or parsed.password is not None
        or "@" in parsed.netloc
        or "@" in decoded_netloc
        or "%" in parsed.netloc
        or any(character.isspace() for character in decoded_netloc)
    ):
        raise ValueError(f"{field_name} must not contain URI userinfo or encoding")
    scheme = parsed.scheme.lower()
    if scheme == "file":
        raise ValueError(f"{field_name} cannot contain a local file URI")
    hostname = parsed.hostname
    if hostname is None or not hostname:
        raise ValueError(f"{field_name} must contain a valid authority")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{field_name} contains an invalid port") from exc
    normalized_host = hostname.lower()
    if ":" in normalized_host and not normalized_host.startswith("["):
        normalized_host = f"[{normalized_host}]"
    netloc = normalized_host if port is None else f"{normalized_host}:{port}"
    return scheme, netloc


def _validated_utc_text(value: str, *, field_name: str) -> str:
    normalize_utc_datetime(value, field_name=field_name)
    return value


def _identity(value: Any, field_name: str) -> str:
    candidate = _normalized_text(value, field_name=field_name, max_length=512)
    _assert_durable_safe_text(candidate, field_name=field_name)
    if candidate.startswith("/") or _WINDOWS_ABSOLUTE_RE.match(candidate):
        raise ValueError(f"{field_name} cannot be an absolute local path")
    return candidate


def _maintenance_authority_identity(value: Any, field_name: str) -> str:
    candidate = _identity(value, field_name)
    decoded_once = unquote(candidate)
    _assert_durable_safe_text(decoded_once, field_name=field_name)
    for representation in (candidate, decoded_once):
        if (
            "://" in representation
            or "/" in representation
            or "\\" in representation
            or _DSN_TEXT_RE.search(representation)
        ):
            raise ValueError(f"{field_name} must be a non-path authority identifier")
    return candidate


def _optional_maintenance_authority_identity(value: Any, field_name: str) -> str | None:
    return None if value is None else _maintenance_authority_identity(value, field_name)


def _optional_maintenance_checksum(value: Any) -> str | None:
    if value is None:
        return None
    candidate = _normalized_text(value, field_name="expected_checksum", max_length=2048)
    _assert_durable_safe_text(candidate, field_name="expected_checksum")
    decoded_once = unquote(candidate)
    _assert_durable_safe_text(decoded_once, field_name="expected_checksum")
    for representation in (candidate, decoded_once):
        if (
            "://" in representation
            or "/" in representation
            or "\\" in representation
            or _WINDOWS_ABSOLUTE_RE.match(representation)
            or _DSN_TEXT_RE.search(representation)
        ):
            raise ValueError("expected_checksum cannot contain a URI, path, or DSN")
    return candidate


def _optional_identity(value: Any, field_name: str) -> str | None:
    return None if value is None else _identity(value, field_name)


def _optional_safe_text(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    candidate = _normalized_text(value, field_name=field_name, max_length=2048)
    _assert_durable_safe_text(candidate, field_name=field_name)
    return candidate


def _normalized_text(value: Any, *, field_name: str, max_length: int) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a normalized non-empty string")
    if len(value) > max_length or _CONTROL_CHARACTER_RE.search(value):
        raise ValueError(f"{field_name} exceeds the durable text contract")
    return value


def _assert_durable_safe_text(value: str, *, field_name: str) -> None:
    if _CONTROL_CHARACTER_RE.search(value):
        raise ValueError(f"{field_name} contains forbidden control text")
    if _DURABLE_SECRET_RE.search(value) or _ACCESS_KEY_VALUE_RE.search(value):
        raise ValueError(f"{field_name} contains forbidden credential text")
    if _SCRATCH_RE.search(value):
        raise ValueError(f"{field_name} contains a forbidden scratch marker")


def _literal(value: Any, allowed: frozenset[str], field_name: str) -> Any:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{field_name} has an unsupported value")
    return value


def _bounded_count(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a non-negative integer")
    if value < 0 or value > ARTIFACT_LIFECYCLE_MAX_COUNT:
        raise ValueError(f"{field_name} must be a bounded non-negative integer")
    return value


def _bounded_signed_integer(value: int, field_name: str) -> None:
    if value < -(2**63) or value > ARTIFACT_LIFECYCLE_MAX_COUNT:
        raise ValueError(f"{field_name} contains an out-of-range integer")


__all__ = [
    "ARTIFACT_CLEANUP_OPERATION_VERSION",
    "ARTIFACT_CLEANUP_MIN_AUDIT_RETENTION_DAYS",
    "ARTIFACT_LIFECYCLE_MAX_COUNT",
    "ARTIFACT_LIFECYCLE_MAX_PAGE_SIZE",
    "ARTIFACT_MAINTENANCE_ITEM_VERSION",
    "ARTIFACT_MAINTENANCE_OPERATION_VERSION",
    "ARTIFACT_RECOVERY_MAX_PAGE_SIZE",
    "ArtifactCleanupDisposition",
    "ArtifactCleanupManifestRecord",
    "ArtifactCleanupReason",
    "ArtifactCleanupStatus",
    "ArtifactCleanupTargetKind",
    "ArtifactCleanupTargetNamespace",
    "ArtifactLifecycleConflictError",
    "ArtifactLifecycleError",
    "ArtifactLifecycleLeaseError",
    "ArtifactLifecycleNotFoundError",
    "ArtifactLifecycleStateError",
    "ArtifactMaintenanceItemRecord",
    "ArtifactMaintenanceItemState",
    "ArtifactMaintenanceMetadataBackend",
    "ArtifactMaintenanceRunKind",
    "ArtifactMaintenanceRunMode",
    "ArtifactMaintenanceRunRecord",
    "ArtifactMaintenanceRunStatus",
    "ArtifactRecoveryCursorRecord",
    "ArtifactRecoveryGenerationError",
    "ArtifactRecoveryStatus",
    "artifact_cleanup_idempotency_key",
    "artifact_maintenance_item_key",
    "artifact_maintenance_run_key",
    "artifact_target_uri_digest",
    "canonical_maintenance_payload_json",
    "canonical_safe_json",
    "normalize_artifact_relative_object_id",
    "normalize_artifact_root_label",
    "normalize_artifact_target_uri",
    "normalize_artifact_target_uri_authority",
    "normalize_artifact_target_uri_digest",
    "normalize_utc_datetime",
    "sanitize_artifact_lifecycle_error_code",
]
