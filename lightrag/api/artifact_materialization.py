"""Operation-scoped, containment-safe object artifact materialization.

This module intentionally does not participate in the document lifecycle yet.
It provides the scratch lease and cleanup primitives needed by a later phase
without making any durable metadata depend on a local scratch path.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from lightrag.api.object_storage import ObjectStorage
from lightrag.utils_pipeline import (
    configured_input_dir,
    get_canonical_input_root,
)

try:  # pragma: no cover - exercised on POSIX CI; fallback is conservative.
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]


SCRATCH_DIR_NAME = ".lightrag-scratch"
ACTIVE_LEASE_MARKER = ".active-lease"
DEFAULT_MATERIALIZATION_MAX_OBJECTS = 10_000
DEFAULT_MATERIALIZATION_MAX_BYTES = 1024 * 1024 * 1024
DEFAULT_MATERIALIZATION_STALE_TTL_SECONDS = 24 * 60 * 60

MATERIALIZATION_MAX_OBJECTS_ENV = (
    "LIGHTRAG_ARTIFACT_MATERIALIZATION_MAX_OBJECTS"
)
MATERIALIZATION_MAX_BYTES_ENV = "LIGHTRAG_ARTIFACT_MATERIALIZATION_MAX_BYTES"
MATERIALIZATION_STALE_TTL_ENV = (
    "LIGHTRAG_ARTIFACT_MATERIALIZATION_STALE_TTL_SECONDS"
)

_LEASE_NAME_RE = re.compile(r"^op-[0-9a-f]{32}$")
_TARGET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")
_active_lease_paths: set[Path] = set()
_active_lease_paths_lock = threading.RLock()


class ArtifactMaterializationError(RuntimeError):
    """Raised when scratch materialization cannot be performed safely."""


class MaterializationLimitError(ArtifactMaterializationError):
    """Raised when an operation exceeds its configured object/byte budget."""


@dataclass(frozen=True, slots=True)
class MaterializationLimits:
    max_objects: int = DEFAULT_MATERIALIZATION_MAX_OBJECTS
    max_total_bytes: int = DEFAULT_MATERIALIZATION_MAX_BYTES
    stale_ttl_seconds: int = DEFAULT_MATERIALIZATION_STALE_TTL_SECONDS

    def __post_init__(self) -> None:
        _validate_positive_int("materialization max_objects", self.max_objects)
        _validate_positive_int(
            "materialization max_total_bytes", self.max_total_bytes
        )
        _validate_non_negative_int(
            "materialization stale_ttl_seconds", self.stale_ttl_seconds
        )

    @classmethod
    def from_env(cls) -> "MaterializationLimits":
        return cls(
            max_objects=_env_int(
                MATERIALIZATION_MAX_OBJECTS_ENV,
                DEFAULT_MATERIALIZATION_MAX_OBJECTS,
            ),
            max_total_bytes=_env_int(
                MATERIALIZATION_MAX_BYTES_ENV,
                DEFAULT_MATERIALIZATION_MAX_BYTES,
            ),
            stale_ttl_seconds=_env_int(
                MATERIALIZATION_STALE_TTL_ENV,
                DEFAULT_MATERIALIZATION_STALE_TTL_SECONDS,
            ),
        )


@dataclass(frozen=True, slots=True)
class MaterializedDocumentTree:
    """Parser-compatible private tree owned by one materialization lease."""

    document_root: Path
    source_path: Path
    parsed_root: Path


def materialization_limits_from_args(args: Any) -> MaterializationLimits:
    """Build limits only from an already validated server args snapshot."""

    missing = [
        attribute
        for attribute in (
            "artifact_materialization_max_objects",
            "artifact_materialization_max_bytes",
            "artifact_materialization_stale_ttl_seconds",
        )
        if not hasattr(args, attribute)
    ]
    if missing:
        raise ValueError(
            "Validated artifact materialization args are missing: "
            + ", ".join(missing)
        )
    return materialization_limits_from_values(
        max_objects=args.artifact_materialization_max_objects,
        max_total_bytes=args.artifact_materialization_max_bytes,
        stale_ttl_seconds=args.artifact_materialization_stale_ttl_seconds,
    )


def materialization_limits_from_values(
    *,
    max_objects: Any = DEFAULT_MATERIALIZATION_MAX_OBJECTS,
    max_total_bytes: Any = DEFAULT_MATERIALIZATION_MAX_BYTES,
    stale_ttl_seconds: Any = DEFAULT_MATERIALIZATION_STALE_TTL_SECONDS,
) -> MaterializationLimits:
    """Coerce programmatic/env-like values through the same strict validator."""

    return MaterializationLimits(
        max_objects=_coerce_int("materialization max_objects", max_objects),
        max_total_bytes=_coerce_int(
            "materialization max_total_bytes", max_total_bytes
        ),
        stale_ttl_seconds=_coerce_int(
            "materialization stale_ttl_seconds", stale_ttl_seconds
        ),
    )


def require_posix_materialization_support() -> None:
    """Fail closed unless advisory ``fcntl`` leases are available."""

    if os.name != "posix" or fcntl is None:
        raise ArtifactMaterializationError(
            "Object artifact materialization requires POSIX fcntl file locking"
        )


def ensure_materialization_scratch_root(
    input_root: Path | str, *, probe_writable: bool = False
) -> Path:
    """Create and validate ``<INPUT_DIR>/.lightrag-scratch`` with mode 0700."""

    try:
        root = Path(input_root).expanduser().resolve(strict=False)
        if root.exists() and not root.is_dir():
            raise ArtifactMaterializationError(
                f"Canonical INPUT_DIR is not a directory: {root}"
            )
        root.mkdir(parents=True, exist_ok=True)

        scratch_path = root / SCRATCH_DIR_NAME
        if scratch_path.is_symlink():
            raise ArtifactMaterializationError(
                "Materialization scratch root cannot be a symlink"
            )
        scratch_path.mkdir(mode=0o700, exist_ok=True)
        if not scratch_path.is_dir():
            raise ArtifactMaterializationError(
                f"Materialization scratch root is not a directory: {scratch_path}"
            )
        os.chmod(scratch_path, 0o700)
        scratch_root = scratch_path.resolve(strict=True)
        if scratch_root.parent != root:
            raise ArtifactMaterializationError(
                "Materialization scratch root escapes canonical INPUT_DIR"
            )

        if probe_writable:
            probe = scratch_root / f".write-probe-{uuid4().hex}"
            fd: int | None = None
            try:
                _assert_contained(scratch_root, probe)
                fd = os.open(
                    probe,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                os.write(fd, b"ok")
            finally:
                if fd is not None:
                    os.close(fd)
                _safe_unlink(scratch_root, probe, missing_ok=True)
        return scratch_root
    except ArtifactMaterializationError:
        raise
    except OSError as exc:
        raise ArtifactMaterializationError(
            f"Canonical INPUT_DIR scratch is not writable: {input_root}"
        ) from exc


class ArtifactMaterializer:
    """Factory for unique operation-scoped materialization leases."""

    def __init__(
        self,
        object_storage: ObjectStorage,
        *,
        input_root: Path | str | None = None,
        limits: MaterializationLimits,
    ) -> None:
        require_posix_materialization_support()
        if object_storage is None:
            raise ArtifactMaterializationError("Object storage is required")
        for validation_method in (
            "stat_object",
            "validate_document_file_uri",
            "validate_document_prefix_uri",
        ):
            implementation = getattr(type(object_storage), validation_method, None)
            if implementation is None or implementation is getattr(
                ObjectStorage, validation_method
            ):
                raise ArtifactMaterializationError(
                    "Object storage must implement fail-closed stat and document URI "
                    "validation"
                )
        if not isinstance(limits, MaterializationLimits):
            raise ArtifactMaterializationError(
                "ArtifactMaterializer requires explicit validated "
                "MaterializationLimits"
            )

        canonical_root = get_canonical_input_root()
        requested_root = (
            Path(input_root).expanduser().resolve(strict=False)
            if input_root is not None
            else None
        )
        if (
            canonical_root is not None
            and requested_root is not None
            and requested_root != canonical_root
        ):
            raise ArtifactMaterializationError(
                "Materializer input root conflicts with canonical INPUT_DIR"
            )

        self.input_root = (
            requested_root
            or canonical_root
            or configured_input_dir().expanduser().resolve(strict=False)
        )
        self.scratch_root = ensure_materialization_scratch_root(self.input_root)
        self.object_storage = object_storage
        self.limits = limits
        self._deferred_leases: set[ArtifactMaterializationLease] = set()
        self._deferred_leases_lock = threading.RLock()
        self.cleanup_stale_leases()

    def create_lease(self) -> "ArtifactMaterializationLease":
        return ArtifactMaterializationLease(self)

    def _retain_deferred_lease(self, lease: "ArtifactMaterializationLease") -> None:
        """Keep a deferred lease (and therefore its fcntl lock) strongly alive."""

        with self._deferred_leases_lock:
            self._deferred_leases.add(lease)

    def _release_deferred_lease(self, lease: "ArtifactMaterializationLease") -> None:
        with self._deferred_leases_lock:
            self._deferred_leases.discard(lease)

    def cleanup_stale_leases(self, *, now: float | None = None) -> int:
        """Delete inactive direct-child leases older than the configured TTL."""

        ttl = self.limits.stale_ttl_seconds
        if ttl == 0:
            return 0
        current_time = time.time() if now is None else float(now)
        removed = 0

        try:
            candidates = list(self.scratch_root.iterdir())
        except FileNotFoundError:
            return 0

        for candidate in candidates:
            if not _LEASE_NAME_RE.fullmatch(candidate.name):
                continue
            if candidate.is_symlink() or not candidate.is_dir():
                continue
            _validate_lease_path(self.scratch_root, candidate)
            if not _is_stale(candidate, current_time, ttl):
                continue

            marker = candidate / ACTIVE_LEASE_MARKER
            marker_fd: int | None = None
            if marker.is_symlink():
                continue
            if marker.exists():
                marker_fd = _try_lock_inactive_marker(candidate, marker)
                if marker_fd is None:
                    continue
            try:
                # Re-check after obtaining the inactivity proof; a concurrent
                # writer may have refreshed the lease between the first stat
                # and lock acquisition.
                if not _is_stale(candidate, current_time, ttl):
                    continue
                _safe_remove_lease_tree(self.scratch_root, candidate)
                removed += 1
            finally:
                if marker_fd is not None:
                    _unlock_and_close(marker_fd)
        return removed


class ArtifactMaterializationLease:
    """One active, exclusively-created scratch directory for one operation."""

    def __init__(self, materializer: ArtifactMaterializer) -> None:
        self._materializer = materializer
        self._defer_cleanup = False
        self._active = False
        self._removed = False
        self._object_count = 0
        self._total_bytes = 0
        self.operation_id = uuid4().hex
        self.path = self._create_unique_lease_path()
        self._marker_path = self.path / ACTIVE_LEASE_MARKER
        self._marker_fd = self._create_active_marker()
        with _active_lease_paths_lock:
            _active_lease_paths.add(self.path)
        self._active = True

    @property
    def object_count(self) -> int:
        return self._object_count

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def __enter__(self) -> "ArtifactMaterializationLease":
        self._require_active()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            if self._defer_cleanup:
                # Keep the marker lock active: defer is used specifically when
                # a producer may still be consuming scratch after this scope.
                # The caller must retain the lease and call cleanup() when the
                # producer reaches a terminal state. A process crash releases
                # the OS lock, making the old directory janitor-eligible.
                if self.path.exists():
                    os.utime(self.path, None, follow_symlinks=False)
            else:
                self.cleanup()
        except Exception:
            if exc_type is None:
                raise
            # Preserve the operation's original exception. The inactive lease
            # remains eligible for the stale janitor if cleanup itself failed.
            self._release_active_marker()
        return False

    async def __aenter__(self) -> "ArtifactMaterializationLease":
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        return self.__exit__(exc_type, exc, traceback)

    def __del__(self) -> None:  # pragma: no cover - best-effort leak fallback
        try:
            self._release_active_marker()
        except Exception:
            pass

    def defer_cleanup(self) -> Path:
        """Keep this lease after context exit for a still-running producer."""

        self._require_active()
        self._defer_cleanup = True
        self._materializer._retain_deferred_lease(self)
        return self.path

    @property
    def cleanup_deferred(self) -> bool:
        return self._defer_cleanup

    def release_deferred_cleanup_for_janitor(self) -> Path:
        """Release a deferred lock after its producer is proven terminal.

        The lease tree is intentionally left in place for the normal TTL
        janitor.  This is primarily useful for explicit recovery/test control;
        callers must never invoke it while a producer can still use the tree.
        """

        self._require_active()
        if not self._defer_cleanup:
            raise ArtifactMaterializationError("Lease cleanup is not deferred")
        path = self.path
        self._defer_cleanup = False
        self._materializer._release_deferred_lease(self)
        self._release_active_marker()
        self._removed = True
        return path

    def create_document_tree(self, source_name: str) -> MaterializedDocumentTree:
        """Create a nested private source + ``__parsed__`` parser layout.

        ``source_name`` is preserved exactly (including non-ASCII characters),
        but must already be one safe filename segment.  The resulting layout is
        ``<lease>/document/<source_name>`` with parser artifacts rooted at
        ``<lease>/document/__parsed__/``.
        """

        self._require_active()
        safe_source_name = _validate_safe_path_segment(
            source_name, label="document source name"
        )
        document_root = self._new_target("document")
        document_root.mkdir(mode=0o700, exist_ok=False)
        os.chmod(document_root, 0o700)
        parsed_root = document_root / "__parsed__"
        _assert_safe_new_target(self.path, parsed_root)
        parsed_root.mkdir(mode=0o700, exist_ok=False)
        os.chmod(parsed_root, 0o700)
        source_path = document_root / safe_source_name
        _assert_safe_new_target(self.path, source_path)
        return MaterializedDocumentTree(
            document_root=document_root,
            source_path=source_path,
            parsed_root=parsed_root,
        )

    async def materialize_document_source(
        self,
        object_uri: str,
        *,
        workspace: str,
        document_id: str,
        source_name: str,
    ) -> MaterializedDocumentTree:
        """Restore one owned source object into a parser-compatible tree."""

        tree = self.create_document_tree(source_name)
        await self._materialize_file_to_target(
            object_uri,
            workspace=workspace,
            document_id=document_id,
            namespace="source",
            artifact_id=None,
            target=tree.source_path,
        )
        return tree

    def link_document_source(
        self, local_path: Path | str, *, source_name: str
    ) -> MaterializedDocumentTree:
        """Hard-link one safe canonical local fallback into a private tree."""

        self._require_active()
        if self._object_count + 1 > self._materializer.limits.max_objects:
            raise MaterializationLimitError("Materialization object limit exceeded")
        source = Path(local_path)
        if source.is_symlink() or not source.is_file():
            raise ArtifactMaterializationError(
                "Local document fallback must be a regular non-symlink file"
            )
        mode = source.stat(follow_symlinks=False).st_mode
        if not stat.S_ISREG(mode):
            raise ArtifactMaterializationError(
                "Local document fallback must be a regular file"
            )
        size = source.stat(follow_symlinks=False).st_size
        if self._total_bytes + size > self._materializer.limits.max_total_bytes:
            raise MaterializationLimitError(
                "Materialization total byte limit exceeded"
            )

        tree = self.create_document_tree(source_name)
        try:
            os.link(source, tree.source_path, follow_symlinks=False)
            _assert_regular_contained_file(self.path, tree.source_path)
        except Exception:
            _safe_remove_subtree(self.path, tree.document_root)
            raise
        self._object_count += 1
        self._total_bytes += size
        return tree

    async def materialize_document_prefix(
        self,
        prefix_uri: str,
        *,
        workspace: str,
        document_id: str,
        artifact_id: str,
        tree: MaterializedDocumentTree,
        directory_name: str,
    ) -> Path:
        """Restore an owned artifact prefix beside the parser sidecar tree."""

        self._require_active()
        self._validate_document_tree(tree)
        safe_directory_name = _validate_safe_path_segment(
            directory_name, label="raw artifact directory name"
        )
        target = tree.parsed_root / safe_directory_name
        return await self._materialize_prefix_to_target(
            prefix_uri,
            workspace=workspace,
            document_id=document_id,
            namespace="artifacts",
            artifact_id=artifact_id,
            target_dir=target,
        )

    async def materialize_document_artifact_file(
        self,
        object_uri: str,
        *,
        workspace: str,
        document_id: str,
        artifact_id: str,
        tree: MaterializedDocumentTree,
        directory: Path,
        filename: str,
    ) -> Path:
        """Restore one artifact file into an existing private document tree."""

        self._require_active()
        self._validate_document_tree(tree)
        safe_filename = _validate_safe_path_segment(
            filename, label="artifact filename"
        )
        resolved_directory = directory.resolve(strict=False)
        document_root = tree.document_root.resolve(strict=False)
        if not resolved_directory.is_relative_to(document_root):
            raise ArtifactMaterializationError(
                "Artifact target directory escapes the document tree"
            )
        if not resolved_directory.is_dir() or resolved_directory.is_symlink():
            raise ArtifactMaterializationError(
                "Artifact target directory must be an existing private directory"
            )
        target = resolved_directory / safe_filename
        return await self._materialize_file_to_target(
            object_uri,
            workspace=workspace,
            document_id=document_id,
            namespace="artifacts",
            artifact_id=artifact_id,
            target=target,
        )

    async def materialize_file(
        self,
        object_uri: str,
        *,
        workspace: str,
        document_id: str,
        namespace: str,
        artifact_id: str | None = None,
        target_name: str | None = None,
    ) -> Path:
        """Safely download one owned object into a caller-independent name."""

        final_name = target_name or f"file-{uuid4().hex}"
        target = self._new_target(final_name)
        return await self._materialize_file_to_target(
            object_uri,
            workspace=workspace,
            document_id=document_id,
            namespace=namespace,
            artifact_id=artifact_id,
            target=target,
        )

    async def _materialize_file_to_target(
        self,
        object_uri: str,
        *,
        workspace: str,
        document_id: str,
        namespace: str,
        artifact_id: str | None,
        target: Path,
    ) -> Path:
        self._require_active()
        if self._object_count + 1 > self._materializer.limits.max_objects:
            raise MaterializationLimitError("Materialization object limit exceeded")

        _validate_materialization_scope(namespace, artifact_id)
        self._materializer.object_storage.validate_document_file_uri(
            object_uri,
            workspace=workspace,
            document_id=document_id,
            namespace=namespace,
            artifact_id=artifact_id,
        )
        _assert_safe_new_target(self.path, target)
        remaining_bytes = (
            self._materializer.limits.max_total_bytes - self._total_bytes
        )
        object_stat = await self._materializer.object_storage.stat_object(object_uri)
        advertised_size = getattr(object_stat, "size", None)
        if (
            isinstance(advertised_size, bool)
            or not isinstance(advertised_size, int)
            or advertised_size < 0
        ):
            raise ArtifactMaterializationError(
                "Object storage returned an invalid object size"
            )
        if advertised_size > remaining_bytes:
            raise MaterializationLimitError(
                "Materialization total byte limit exceeded before download"
            )
        temp_target = target.parent / f".tmp-{uuid4().hex}"
        _assert_safe_new_target(self.path, temp_target)
        committed = False
        try:
            await self._materializer.object_storage.download_file(
                object_uri, temp_target
            )
            _assert_regular_contained_file(self.path, temp_target)
            size = temp_target.stat().st_size
            if size > remaining_bytes:
                raise MaterializationLimitError(
                    "Materialization total byte limit exceeded"
                )
            os.chmod(temp_target, 0o600)
            _assert_safe_new_target(self.path, target)
            os.link(temp_target, target, follow_symlinks=False)
            committed = True
            _safe_unlink(self.path, temp_target, missing_ok=False)
            self._object_count += 1
            self._total_bytes += size
            return target
        except Exception:
            _safe_unlink(self.path, temp_target, missing_ok=True)
            if committed:
                _safe_unlink(self.path, target, missing_ok=True)
            raise

    async def materialize_prefix(
        self,
        prefix_uri: str,
        *,
        workspace: str,
        document_id: str,
        namespace: str,
        artifact_id: str | None = None,
        target_name: str | None = None,
    ) -> Path:
        """Safely restore an owned object prefix into a fresh lease subdir."""

        directory_name = target_name or f"prefix-{uuid4().hex}"
        target_dir = self._new_target(directory_name)
        return await self._materialize_prefix_to_target(
            prefix_uri,
            workspace=workspace,
            document_id=document_id,
            namespace=namespace,
            artifact_id=artifact_id,
            target_dir=target_dir,
        )

    async def _materialize_prefix_to_target(
        self,
        prefix_uri: str,
        *,
        workspace: str,
        document_id: str,
        namespace: str,
        artifact_id: str | None,
        target_dir: Path,
    ) -> Path:
        self._require_active()
        remaining_objects = (
            self._materializer.limits.max_objects - self._object_count
        )
        remaining_bytes = (
            self._materializer.limits.max_total_bytes - self._total_bytes
        )
        if remaining_objects <= 0:
            raise MaterializationLimitError("Materialization object limit exceeded")

        _validate_materialization_scope(namespace, artifact_id)
        self._materializer.object_storage.validate_document_prefix_uri(
            prefix_uri,
            workspace=workspace,
            document_id=document_id,
            namespace=namespace,
            artifact_id=artifact_id,
        )
        _assert_safe_new_target(self.path, target_dir)
        try:
            target_dir.mkdir(mode=0o700, exist_ok=False)
            os.chmod(target_dir, 0o700)
            _assert_contained(self.path, target_dir)
            downloaded = await self._materializer.object_storage.download_prefix(
                prefix_uri,
                target_dir,
                max_objects=remaining_objects,
                max_total_bytes=remaining_bytes,
            )
            file_count, total_bytes = _inspect_materialized_tree(
                self.path, target_dir
            )
            if downloaded != file_count:
                raise ArtifactMaterializationError(
                    "Object storage prefix count did not match materialized files"
                )
            if file_count > remaining_objects:
                raise MaterializationLimitError(
                    "Materialization object limit exceeded"
                )
            if total_bytes > remaining_bytes:
                raise MaterializationLimitError(
                    "Materialization total byte limit exceeded"
                )
            self._object_count += file_count
            self._total_bytes += total_bytes
            return target_dir
        except Exception:
            _safe_remove_subtree(self.path, target_dir)
            raise

    def cleanup(self) -> None:
        """Remove the lease after a final containment check."""

        if self._removed:
            return
        try:
            if self.path.exists():
                _safe_remove_lease_tree(
                    self._materializer.scratch_root, self.path
                )
            self._removed = True
        finally:
            self._defer_cleanup = False
            self._materializer._release_deferred_lease(self)
            self._release_active_marker()

    def _create_unique_lease_path(self) -> Path:
        for _ in range(16):
            operation_id = uuid4().hex
            candidate = self._materializer.scratch_root / f"op-{operation_id}"
            _validate_lease_path(self._materializer.scratch_root, candidate)
            try:
                candidate.mkdir(mode=0o700, exist_ok=False)
            except FileExistsError:
                continue
            os.chmod(candidate, 0o700)
            self.operation_id = operation_id
            return candidate.resolve(strict=True)
        raise ArtifactMaterializationError(
            "Unable to allocate a unique materialization operation id"
        )

    def _create_active_marker(self) -> int:
        try:
            fd = os.open(
                self._marker_path,
                os.O_RDWR | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            _lock_fd(fd)
            payload = json.dumps(
                {
                    "operation_id": self.operation_id,
                    "pid": os.getpid(),
                    "created_at": time.time(),
                },
                separators=(",", ":"),
            ).encode("utf-8")
            os.write(fd, payload)
            os.fsync(fd)
            return fd
        except Exception:
            _safe_remove_lease_tree(self._materializer.scratch_root, self.path)
            raise

    def _release_active_marker(self) -> None:
        if self._active:
            with _active_lease_paths_lock:
                _active_lease_paths.discard(self.path)
            self._active = False
        fd = getattr(self, "_marker_fd", None)
        if fd is not None:
            _unlock_and_close(fd)
            self._marker_fd = None

    def _new_target(self, name: str) -> Path:
        if not isinstance(name, str) or not _TARGET_NAME_RE.fullmatch(name):
            raise ArtifactMaterializationError(
                "Materialization target name must be one safe path segment"
            )
        target = self.path / name
        _assert_safe_new_target(self.path, target)
        return target

    def _validate_document_tree(self, tree: MaterializedDocumentTree) -> None:
        if not isinstance(tree, MaterializedDocumentTree):
            raise ArtifactMaterializationError("Invalid materialized document tree")
        for candidate in (tree.document_root, tree.source_path, tree.parsed_root):
            _assert_contained(self.path, candidate)
        if tree.document_root.parent != self.path:
            raise ArtifactMaterializationError(
                "Materialized document root is not owned by this lease"
            )
        if tree.source_path.parent != tree.document_root:
            raise ArtifactMaterializationError(
                "Materialized document source is outside its document root"
            )
        if tree.parsed_root.parent != tree.document_root or tree.parsed_root.name != "__parsed__":
            raise ArtifactMaterializationError(
                "Materialized parser tree has an invalid __parsed__ root"
            )

    def _require_active(self) -> None:
        if not self._active or self._removed or not self.path.is_dir():
            raise ArtifactMaterializationError(
                "Materialization lease is not active"
            )


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return _coerce_int(name, raw)


def _coerce_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value.strip())
    raise ValueError(f"{name} must be an integer")


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_materialization_scope(
    namespace: str, artifact_id: str | None
) -> None:
    normalized = str(namespace).strip().lower()
    if normalized == "artifact":
        normalized = "artifacts"
    if normalized not in {"source", "artifacts"}:
        raise ArtifactMaterializationError(
            "Materialization namespace must be source or artifacts"
        )
    if normalized == "artifacts" and not artifact_id:
        raise ArtifactMaterializationError(
            "Artifact materialization requires an artifact id"
        )
    if normalized == "source" and artifact_id is not None:
        raise ArtifactMaterializationError(
            "Source materialization cannot specify an artifact id"
        )


def _validate_safe_path_segment(value: str, *, label: str) -> str:
    if not isinstance(value, str):
        raise ArtifactMaterializationError(f"{label} must be a string")
    if not value or value in {".", ".."}:
        raise ArtifactMaterializationError(f"{label} must be a safe filename")
    if "/" in value or "\\" in value:
        raise ArtifactMaterializationError(f"{label} must be one path segment")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise ArtifactMaterializationError(f"{label} contains control characters")
    if len(value.encode("utf-8")) > 255:
        raise ArtifactMaterializationError(f"{label} is too long")
    return value


def _validate_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be non-negative")


def _assert_contained(root: Path, candidate: Path) -> Path:
    root_resolved = root.resolve(strict=True)
    candidate_resolved = candidate.resolve(strict=False)
    if not candidate_resolved.is_relative_to(root_resolved):
        raise ArtifactMaterializationError(
            f"Scratch path escapes operation directory: {candidate}"
        )
    return candidate_resolved


def _assert_safe_new_target(root: Path, target: Path) -> None:
    _assert_contained(root, target)
    if target.exists() or target.is_symlink():
        raise ArtifactMaterializationError(
            f"Materialization target already exists: {target.name}"
        )
    current = root
    for part in target.relative_to(root).parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ArtifactMaterializationError(
                "Materialization target contains a symlink component"
            )


def _assert_regular_contained_file(root: Path, path: Path) -> None:
    _assert_contained(root, path)
    if path.is_symlink():
        raise ArtifactMaterializationError(
            "Object storage created a symlink instead of a file"
        )
    mode = path.stat().st_mode
    if not stat.S_ISREG(mode):
        raise ArtifactMaterializationError(
            "Object storage did not materialize a regular file"
        )


def _inspect_materialized_tree(lease_root: Path, target_dir: Path) -> tuple[int, int]:
    _assert_contained(lease_root, target_dir)
    if target_dir.is_symlink() or not target_dir.is_dir():
        raise ArtifactMaterializationError(
            "Materialized prefix root is not a safe directory"
        )

    file_count = 0
    total_bytes = 0
    for current_root, dir_names, file_names in os.walk(
        target_dir, topdown=True, followlinks=False
    ):
        current = Path(current_root)
        _assert_contained(lease_root, current)
        os.chmod(current, 0o700)
        for directory_name in dir_names:
            directory = current / directory_name
            if directory.is_symlink():
                raise ArtifactMaterializationError(
                    "Materialized prefix contains a directory symlink"
                )
            _assert_contained(lease_root, directory)
        for file_name in file_names:
            path = current / file_name
            _assert_regular_contained_file(lease_root, path)
            file_count += 1
            total_bytes += path.stat().st_size
            os.chmod(path, 0o600)
    return file_count, total_bytes


def _validate_lease_path(scratch_root: Path, lease_path: Path) -> None:
    scratch_resolved = scratch_root.resolve(strict=True)
    if not _LEASE_NAME_RE.fullmatch(lease_path.name):
        raise ArtifactMaterializationError("Invalid scratch lease name")
    if lease_path.parent.resolve(strict=True) != scratch_resolved:
        raise ArtifactMaterializationError(
            "Scratch lease is not a direct child of the scratch root"
        )
    if lease_path.is_symlink():
        raise ArtifactMaterializationError("Scratch lease cannot be a symlink")
    resolved = lease_path.resolve(strict=False)
    if resolved.parent != scratch_resolved:
        raise ArtifactMaterializationError("Scratch lease escapes scratch root")


def _safe_unlink(root: Path, path: Path, *, missing_ok: bool) -> None:
    if not path.exists() and not path.is_symlink():
        if missing_ok:
            return
        raise FileNotFoundError(path)
    _assert_contained(root, path)
    if path.is_symlink():
        raise ArtifactMaterializationError(
            f"Refusing to unlink scratch symlink: {path}"
        )
    path.unlink(missing_ok=missing_ok)


def _safe_remove_subtree(lease_root: Path, path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    _assert_contained(lease_root, path)
    if path.is_symlink():
        raise ArtifactMaterializationError(
            f"Refusing to remove scratch symlink: {path}"
        )
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _safe_remove_lease_tree(scratch_root: Path, lease_path: Path) -> None:
    if not lease_path.exists() and not lease_path.is_symlink():
        return
    _validate_lease_path(scratch_root, lease_path)
    shutil.rmtree(lease_path)


def _is_stale(path: Path, now: float, ttl: int) -> bool:
    try:
        modified_at = path.stat(follow_symlinks=False).st_mtime
    except FileNotFoundError:
        return False
    return now - modified_at >= ttl


def _lock_fd(fd: int) -> None:
    require_posix_materialization_support()
    assert fcntl is not None
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _try_lock_inactive_marker(lease_path: Path, marker: Path) -> int | None:
    with _active_lease_paths_lock:
        if lease_path in _active_lease_paths:
            return None
    if fcntl is None:
        raise ArtifactMaterializationError(
            "Stale lease cleanup requires POSIX fcntl file locking"
        )
    try:
        fd = os.open(marker, os.O_RDWR)
    except FileNotFoundError:
        return -1
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def _unlock_and_close(fd: int) -> None:
    if fd < 0:
        return
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
