"""Core durable artifact bindings for stateless pipeline execution.

This module intentionally has no dependency on :mod:`lightrag.api`.  The
binding is durable metadata only; runtime materialization is represented by
Protocols that later H2 lanes will implement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path, PurePath
from typing import Any, Literal, Mapping, Protocol, cast, runtime_checkable


_BINDING_VERSION = 1
_BINDING_AUTHORITY = "kb_metadata"
_SCRATCH_MARKER = ".lightrag-scratch"
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_BINDING_FIELDS = frozenset(
    {
        "version",
        "authority",
        "state",
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
        "parse_generation_id",
        "index_hash",
        "sidecar_artifact_id",
        "blocks_artifact_id",
        "expected_current_sidecar_artifact_id",
        "expected_current_blocks_artifact_id",
        "raw_artifact_ids",
    }
)
_REQUIRED_IDENTITY_FIELDS = (
    "kb_id",
    "kb_generation",
    "workspace",
    "document_id",
    "lightrag_doc_id",
    "job_id",
    "claim_token",
)
_FORBIDDEN_DURABLE_FIELDS = frozenset(
    {
        "sidecar_location",
        "blocks_path",
        "source_uri",
        "source_path",
        "runtime_path",
        "runtime_source_path",
        "object_uri",
        "object_prefix_uri",
        "presigned_url",
        "scratch_path",
        "scratch_lease_id",
    }
)

PipelineAttemptRowKind = Literal["full_docs", "doc_status"]


class PipelineTerminalOutcome(str, Enum):
    """Terminal result passed to a future processing-owner session."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class PipelineArtifactCommitOutcome(str, Enum):
    """Durable metadata outcome returned by processing-owner finalization."""

    COMMITTED = "committed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class PipelineArtifactFinalizationResult:
    """Durable result handed from the artifact owner back to the pipeline.

    ``COMMITTED`` always carries the exact committed binding that the pipeline
    may patch into ``full_docs``. ``UNKNOWN`` intentionally carries no binding:
    the pipeline must close its runtime without publishing a terminal status so
    recovery can reconcile PostgreSQL and the retained immutable objects.
    """

    outcome: PipelineArtifactCommitOutcome
    committed_binding: PipelineArtifactBinding | None = None
    chunks_count: int | None = None
    entity_count: int | None = None
    relation_count: int | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.outcome is PipelineArtifactCommitOutcome.COMMITTED:
            if self.committed_binding is None:
                raise ValueError("Committed artifact finalization requires a binding")
            if self.committed_binding.state != "committed":
                raise ValueError("Artifact finalization binding is not committed")
        elif self.committed_binding is not None:
            raise ValueError("Unknown artifact finalization cannot publish a binding")


class PipelineArtifactRuntimeError(RuntimeError):
    """Durable-safe failure raised by the core runtime-session contract."""

    error_code = "pipeline_artifact_runtime_error"


class PipelineArtifactMaterializerMissingError(PipelineArtifactRuntimeError):
    """A binding reached the drain owner without a materializer callback."""

    error_code = "artifact_materializer_required"

    def __init__(self, document_id: str) -> None:
        super().__init__(
            f"{self.error_code}: binding document {document_id!r} requires "
            "a processing-owner artifact materializer"
        )


class PipelineArtifactSessionMismatchError(PipelineArtifactRuntimeError):
    """A callback returned a session for a different durable binding."""

    error_code = "artifact_session_binding_mismatch"

    def __init__(self, document_id: str) -> None:
        super().__init__(
            f"{self.error_code}: runtime session does not match binding "
            f"document {document_id!r}"
        )


class PipelineAttemptCommitError(RuntimeError):
    """Base durable-safe failure for an attempt-fenced storage commit."""

    error_code = "pipeline_attempt_commit_error"


class PipelineAttemptCommitCapabilityError(PipelineAttemptCommitError):
    """An object-authoritative storage lacks the atomic attempt capability."""

    error_code = "pipeline_attempt_commit_capability_missing"

    def __init__(self, row_kind: PipelineAttemptRowKind) -> None:
        self.row_kind = row_kind
        super().__init__(
            f"{self.error_code}: {row_kind} storage does not implement "
            "compare_and_commit_pipeline_attempt"
        )


class PipelineAttemptCommitStaleError(PipelineAttemptCommitError):
    """The current durable row belongs to a different or missing attempt."""

    error_code = "pipeline_attempt_stale"

    def __init__(self, key: str, *, row_kind: PipelineAttemptRowKind) -> None:
        self.key = key
        self.row_kind = row_kind
        super().__init__(
            f"{self.error_code}: {row_kind} row {key!r} is no longer owned by "
            "the expected pipeline attempt"
        )


class PipelineAttemptCommitOutcomeUnknownError(PipelineAttemptCommitError):
    """A backend transport failure left the atomic commit outcome unknown.

    Backends must raise this exception only when the request may already have
    committed durably but the acknowledgement could not be observed.  Core
    callers deliberately do not retry such an operation: recovery or an exact
    durable read-back must determine the outcome first.
    """

    error_code = "pipeline_attempt_commit_outcome_unknown"

    def __init__(
        self,
        key: str,
        *,
        row_kind: PipelineAttemptRowKind,
        reason: str | None = None,
    ) -> None:
        self.key = key
        self.row_kind = row_kind
        self.reason = _durable_safe_pipeline_attempt_reason(reason)
        super().__init__(
            f"{self.error_code}: {row_kind} row {key!r} commit outcome is "
            f"unknown ({self.reason})"
        )


@runtime_checkable
class PipelineAttemptCompareAndCommitStorage(Protocol):
    """Optional atomic storage capability used by object-authoritative rows.

    The implementation must atomically compare the canonical token in the
    current durable row and replace the whole row only when it equals
    ``expected_attempt_token``.  ``full_docs`` compares
    ``artifact_binding.claim_token``; ``doc_status`` compares
    ``metadata.pipeline_attempt_token``.  A missing row/token or mismatch must
    return ``False`` with zero mutation.

    Returning ``True`` guarantees that ``payload`` is durable and visible to
    other processes; any backend-specific flush/refresh is therefore part of
    this method.  If transport ambiguity means the write may have committed but
    its acknowledgement was lost, the implementation must raise
    :class:`PipelineAttemptCommitOutcomeUnknownError` instead of returning a
    guess or internally retrying the mutation.
    """

    async def compare_and_commit_pipeline_attempt(
        self,
        key: str,
        payload: Mapping[str, Any],
        *,
        expected_attempt_token: str,
        row_kind: PipelineAttemptRowKind,
    ) -> bool: ...


def extract_pipeline_attempt_token(
    row: Mapping[str, Any] | None,
    *,
    row_kind: PipelineAttemptRowKind,
) -> str | None:
    """Return the canonical attempt token for one durable pipeline row."""

    if row_kind not in {"full_docs", "doc_status"}:
        raise ValueError(f"Unsupported pipeline attempt row kind: {row_kind!r}")
    if not isinstance(row, Mapping):
        return None
    if row_kind == "full_docs":
        container = row.get("artifact_binding")
        token_key = "claim_token"
    else:
        container = row.get("metadata")
        token_key = "pipeline_attempt_token"
    if not isinstance(container, Mapping):
        return None
    token = container.get(token_key)
    if not isinstance(token, str) or not token:
        return None
    return token


async def commit_pipeline_attempt_if_current(
    storage: object,
    key: str,
    payload: Mapping[str, Any],
    *,
    expected_attempt_token: str,
    row_kind: PipelineAttemptRowKind,
) -> None:
    """Require and invoke the atomic capability for one same-attempt write.

    A backend ``False`` result is promoted to the explicit stale disposition;
    an unknown transport outcome remains distinguishable from ordinary backend
    errors through :class:`PipelineAttemptCommitOutcomeUnknownError`.
    """

    if not isinstance(key, str) or not key:
        raise ValueError("Pipeline attempt commit key must be a non-empty string")
    if not isinstance(expected_attempt_token, str) or not expected_attempt_token:
        raise ValueError("Expected pipeline attempt token must be non-empty")
    if row_kind not in {"full_docs", "doc_status"}:
        raise ValueError(f"Unsupported pipeline attempt row kind: {row_kind!r}")
    if not isinstance(payload, Mapping):
        raise TypeError("Pipeline attempt commit payload must be a mapping")
    payload_token = extract_pipeline_attempt_token(payload, row_kind=row_kind)
    if payload_token != expected_attempt_token:
        raise ValueError(
            f"{row_kind} payload does not carry the expected pipeline attempt token"
        )
    if not isinstance(storage, PipelineAttemptCompareAndCommitStorage):
        raise PipelineAttemptCommitCapabilityError(row_kind)
    compare_and_commit = cast(
        PipelineAttemptCompareAndCommitStorage,
        storage,
    ).compare_and_commit_pipeline_attempt
    if not callable(compare_and_commit):  # pragma: no cover - protocol guard
        raise PipelineAttemptCommitCapabilityError(row_kind)
    committed = await compare_and_commit(
        key,
        payload,
        expected_attempt_token=expected_attempt_token,
        row_kind=row_kind,
    )
    if type(committed) is not bool:
        raise TypeError("compare_and_commit_pipeline_attempt must return an exact bool")
    if not committed:
        raise PipelineAttemptCommitStaleError(key, row_kind=row_kind)


def _durable_safe_pipeline_attempt_reason(reason: object) -> str:
    """Reduce backend diagnostics to non-sensitive stage/type identifiers."""

    if not isinstance(reason, str) or not reason.strip():
        return "backend_transport_ambiguity"
    parts: list[str] = []
    for raw_part in reason.split(":", 2)[:2]:
        part = raw_part.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}", part):
            break
        parts.append(part)
    return ":".join(parts) if parts else "backend_transport_ambiguity"


@dataclass(frozen=True, slots=True)
class PipelineArtifactBinding:
    """Versioned pointer from pipeline storage to metadata authority."""

    version: int
    authority: Literal["kb_metadata"]
    state: Literal["claimed", "committed"]
    operation: Literal["parse", "build"]
    kb_id: str
    kb_generation: str
    workspace: str
    document_id: str
    lightrag_doc_id: str
    job_id: str
    claim_token: str
    source_hash: str | None
    parser_hash: str | None
    parse_generation_id: str | None
    index_hash: str | None
    sidecar_artifact_id: str | None
    blocks_artifact_id: str | None
    expected_current_sidecar_artifact_id: str | None
    expected_current_blocks_artifact_id: str | None
    raw_artifact_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != _BINDING_VERSION:
            raise ValueError(
                f"Unsupported pipeline artifact binding version: {self.version!r}"
            )
        if type(self.authority) is not str or self.authority != _BINDING_AUTHORITY:
            raise ValueError(
                f"Unsupported pipeline artifact authority: {self.authority!r}"
            )
        if type(self.state) is not str or self.state not in {"claimed", "committed"}:
            raise ValueError(f"Invalid pipeline artifact binding state: {self.state!r}")
        if type(self.operation) is not str or self.operation not in {"parse", "build"}:
            raise ValueError(
                f"Invalid pipeline artifact binding operation: {self.operation!r}"
            )
        for field_name in _REQUIRED_IDENTITY_FIELDS:
            _validate_binding_identity(field_name, getattr(self, field_name))
        for field_name in (
            "source_hash",
            "parser_hash",
            "parse_generation_id",
            "index_hash",
            "sidecar_artifact_id",
            "blocks_artifact_id",
            "expected_current_sidecar_artifact_id",
            "expected_current_blocks_artifact_id",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _validate_binding_identity(field_name, value)
        if not isinstance(self.raw_artifact_ids, tuple):
            raise TypeError("raw_artifact_ids must be a tuple of strings")
        if len(set(self.raw_artifact_ids)) != len(self.raw_artifact_ids):
            raise ValueError("raw_artifact_ids must be unique")
        for artifact_id in self.raw_artifact_ids:
            _validate_binding_identity("raw_artifact_ids", artifact_id)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        expected_workspace: str | None = None,
    ) -> PipelineArtifactBinding:
        if not isinstance(value, Mapping):
            raise TypeError("Pipeline artifact binding must be a mapping")
        supplied_fields = set(value)
        unknown_fields = supplied_fields - _BINDING_FIELDS
        missing_fields = _BINDING_FIELDS - supplied_fields
        if unknown_fields:
            raise ValueError(
                "Unknown pipeline artifact binding fields: "
                + ", ".join(sorted(str(field) for field in unknown_fields))
            )
        if missing_fields:
            raise ValueError(
                "Missing pipeline artifact binding fields: "
                + ", ".join(sorted(missing_fields))
            )
        raw_ids = value["raw_artifact_ids"]
        if not isinstance(raw_ids, (list, tuple)) or isinstance(raw_ids, str):
            raise TypeError("raw_artifact_ids must be a list or tuple of strings")
        binding = cls(
            version=value["version"],
            authority=value["authority"],
            state=value["state"],
            operation=value["operation"],
            kb_id=value["kb_id"],
            kb_generation=value["kb_generation"],
            workspace=value["workspace"],
            document_id=value["document_id"],
            lightrag_doc_id=value["lightrag_doc_id"],
            job_id=value["job_id"],
            claim_token=value["claim_token"],
            source_hash=value["source_hash"],
            parser_hash=value["parser_hash"],
            parse_generation_id=value["parse_generation_id"],
            index_hash=value["index_hash"],
            sidecar_artifact_id=value["sidecar_artifact_id"],
            blocks_artifact_id=value["blocks_artifact_id"],
            expected_current_sidecar_artifact_id=value[
                "expected_current_sidecar_artifact_id"
            ],
            expected_current_blocks_artifact_id=value[
                "expected_current_blocks_artifact_id"
            ],
            raw_artifact_ids=tuple(raw_ids),
        )
        if expected_workspace is not None and binding.workspace != expected_workspace:
            raise ValueError(
                "Pipeline artifact binding workspace mismatch: "
                f"expected {expected_workspace!r}, got {binding.workspace!r}"
            )
        return binding

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            "version": self.version,
            "authority": self.authority,
            "state": self.state,
            "operation": self.operation,
            "kb_id": self.kb_id,
            "kb_generation": self.kb_generation,
            "workspace": self.workspace,
            "document_id": self.document_id,
            "lightrag_doc_id": self.lightrag_doc_id,
            "job_id": self.job_id,
            "claim_token": self.claim_token,
            "source_hash": self.source_hash,
            "parser_hash": self.parser_hash,
            "parse_generation_id": self.parse_generation_id,
            "index_hash": self.index_hash,
            "sidecar_artifact_id": self.sidecar_artifact_id,
            "blocks_artifact_id": self.blocks_artifact_id,
            "expected_current_sidecar_artifact_id": (
                self.expected_current_sidecar_artifact_id
            ),
            "expected_current_blocks_artifact_id": (
                self.expected_current_blocks_artifact_id
            ),
            "raw_artifact_ids": list(self.raw_artifact_ids),
        }

    def committed(
        self,
        *,
        parse_generation_id: str | None,
        index_hash: str | None,
        sidecar_artifact_id: str | None,
        blocks_artifact_id: str | None,
        raw_artifact_ids: tuple[str, ...] | list[str],
    ) -> PipelineArtifactBinding:
        """Return an immutable committed binding for the same attempt."""

        return replace(
            self,
            state="committed",
            parse_generation_id=parse_generation_id,
            index_hash=index_hash,
            sidecar_artifact_id=sidecar_artifact_id,
            blocks_artifact_id=blocks_artifact_id,
            raw_artifact_ids=tuple(raw_artifact_ids),
        )


@runtime_checkable
class PipelineArtifactSession(Protocol):
    """Operation-scoped runtime owned by the actual pipeline drain worker.

    Runtime paths are deliberately exposed only on this in-memory protocol;
    they must never be copied into ``full_docs`` or ``doc_status``.  A parse
    binding consumes ``source_path`` while a build binding consumes the exact
    ``sidecar_dir`` / ``blocks_path`` pair.
    """

    @property
    def binding(self) -> PipelineArtifactBinding: ...

    @property
    def source_path(self) -> Path | None: ...

    @property
    def sidecar_dir(self) -> Path | None: ...

    @property
    def blocks_path(self) -> Path | None: ...

    @property
    def producer_active(self) -> bool:
        """Whether a producer may still be using the runtime tree."""

        ...

    def redact(self, error: object) -> str:
        """Return a durable-safe error without runtime/object credentials."""

        ...

    def defer_cleanup(self) -> None:
        """Retain the runtime when a cancelled producer may still use it."""

        ...

    async def finish(self, outcome: PipelineTerminalOutcome) -> None: ...

    async def handoff_success(
        self,
        *,
        parsed_data: Mapping[str, Any] | None = None,
        chunks_count: int | None = None,
    ) -> PipelineArtifactFinalizationResult:
        """Promote outputs and return only a confirmed durable outcome."""

        ...

    async def aclose(self) -> None: ...


@runtime_checkable
class PipelineArtifactMaterializer(Protocol):
    """Callback that opens a processing-owner runtime session."""

    async def __call__(
        self, binding: PipelineArtifactBinding
    ) -> PipelineArtifactSession: ...


def canonicalize_pipeline_logical_filename(value: object) -> str:
    """Validate and return a durable logical basename."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("Pipeline binding file_path must be a non-empty basename")
    filename = value.strip()
    _reject_runtime_locator("file_path", filename)
    if _URI_SCHEME.match(filename):
        raise ValueError("Pipeline binding file_path must not be a URI")
    if any(ord(character) < 32 or ord(character) == 127 for character in filename):
        raise ValueError(
            "Pipeline binding file_path must not contain control characters"
        )
    if filename in {".", ".."} or PurePath(filename).name != filename:
        raise ValueError("Pipeline binding file_path must be a logical basename")
    if "/" in filename or "\\" in filename:
        raise ValueError("Pipeline binding file_path must not contain path separators")
    return filename


def assert_no_runtime_artifact_payload(
    value: object,
    *,
    context: str = "pipeline durable write",
) -> None:
    """Reject scratch/runtime locators before a binding-mode durable write."""

    def walk(current: object, key: str | None = None) -> None:
        if isinstance(current, Mapping):
            for raw_key, nested in current.items():
                field_name = str(raw_key)
                lowered = field_name.lower()
                if lowered in _FORBIDDEN_DURABLE_FIELDS or (
                    lowered != "file_path"
                    and (
                        lowered.startswith("runtime")
                        or lowered.startswith("scratch")
                        or lowered.startswith("presigned")
                        or lowered.endswith("_presigned_url")
                        or lowered.startswith("object_")
                        or lowered.endswith("_object_uri")
                    )
                ):
                    raise ValueError(
                        f"{context} contains forbidden runtime field {field_name!r}"
                    )
                walk(nested, field_name)
            return
        if isinstance(current, (list, tuple, set)):
            for nested in current:
                walk(nested, key)
            return
        if isinstance(current, str):
            if _SCRATCH_MARKER in current.lower():
                raise ValueError(f"{context} contains a scratch runtime reference")
            if key == "file_path":
                canonicalize_pipeline_logical_filename(current)

    walk(value)


def _validate_binding_identity(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Pipeline artifact binding {field_name} must be non-empty")
    if value != value.strip():
        raise ValueError(
            f"Pipeline artifact binding {field_name} must not contain outer whitespace"
        )
    if "/" in value or "\\" in value:
        raise ValueError(
            f"Pipeline artifact binding {field_name} must be an opaque identity"
        )
    if any(character.isspace() for character in value):
        raise ValueError(
            f"Pipeline artifact binding {field_name} must not contain whitespace"
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(
            f"Pipeline artifact binding {field_name} must not contain control characters"
        )
    if _URI_SCHEME.match(value) and not (
        field_name in {"source_hash", "parser_hash", "index_hash"}
        and value.lower().startswith("sha256:")
    ):
        raise ValueError(f"Pipeline artifact binding {field_name} contains a URI")
    _reject_runtime_locator(field_name, value)


def _reject_runtime_locator(field_name: str, value: str) -> None:
    lowered = value.lower()
    if _SCRATCH_MARKER in lowered:
        raise ValueError(
            f"Pipeline artifact binding {field_name} contains a scratch path"
        )
    if _WINDOWS_ABSOLUTE_PATH.match(value) or value.startswith(("/", "\\")):
        raise ValueError(
            f"Pipeline artifact binding {field_name} contains an absolute path"
        )
    if "://" in value or lowered.startswith("file:"):
        raise ValueError(f"Pipeline artifact binding {field_name} contains a URI")
