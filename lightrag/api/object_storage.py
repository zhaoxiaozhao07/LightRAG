from __future__ import annotations

import base64
import binascii
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import hmac
import inspect
import json
import mimetypes
import os
import re
import secrets
import stat
import time
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, unquote, urlparse
from uuid import uuid4

# Cached S3 readiness probe TTL. A successful ``head_bucket`` is reused for
# this many seconds so /health never amplifies into per-request bucket probes
# (and never lists or downloads objects).
_S3_READINESS_PROBE_TTL_SECONDS = 30.0


_LISTING_TOKEN_VERSION = 2
_LISTING_TOKEN_PREFIX = f"lrag-list-v{_LISTING_TOKEN_VERSION}."
_LISTING_TOKEN_INTEGRITY_DOMAIN = (
    b"LightRAG:S3ObjectStorage:list_objects_page:continuation:v2\x00"
)


@dataclass(frozen=True, slots=True)
class ObjectStorageConfig:
    backend: str
    bucket: str
    endpoint_url: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    region_name: str | None = None
    prefix: str = "kb"
    use_ssl: bool = False
    create_bucket: bool = True
    disable_expect_header: bool = True
    sdk_retry_mode: str = "standard"
    sdk_total_max_attempts: int = 4
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 30.0
    request_checksum_calculation: str = "when_required"
    response_checksum_validation: str = "when_required"

    @classmethod
    def from_env(cls) -> "ObjectStorageConfig":
        backend = os.getenv("LIGHTRAG_OBJECT_STORAGE", "local").strip().lower()
        raw_prefix = os.getenv("LIGHTRAG_OBJECT_STORAGE_PREFIX", "kb").strip()
        return cls(
            backend=backend,
            bucket=os.getenv("LIGHTRAG_OBJECT_STORAGE_BUCKET", "lightrag-kb").strip(),
            endpoint_url=os.getenv("LIGHTRAG_OBJECT_STORAGE_ENDPOINT"),
            access_key_id=os.getenv("LIGHTRAG_OBJECT_STORAGE_ACCESS_KEY_ID")
            or os.getenv("MINIO_ACCESS_KEY_ID"),
            secret_access_key=os.getenv("LIGHTRAG_OBJECT_STORAGE_SECRET_ACCESS_KEY")
            or os.getenv("MINIO_SECRET_ACCESS_KEY"),
            region_name=os.getenv("LIGHTRAG_OBJECT_STORAGE_REGION") or "us-east-1",
            # Preserve the historical configuration contract: boundary slashes
            # are normalization, while internal key segments remain strict.
            prefix=raw_prefix.strip("/"),
            use_ssl=_env_bool("LIGHTRAG_OBJECT_STORAGE_USE_SSL", default=False),
            create_bucket=_env_bool(
                "LIGHTRAG_OBJECT_STORAGE_CREATE_BUCKET", default=True
            ),
            disable_expect_header=_env_bool(
                "LIGHTRAG_OBJECT_STORAGE_DISABLE_EXPECT_HEADER", default=True
            ),
            sdk_retry_mode=os.getenv(
                "LIGHTRAG_OBJECT_STORAGE_SDK_RETRY_MODE", "standard"
            )
            .strip()
            .lower(),
            sdk_total_max_attempts=_env_int(
                "LIGHTRAG_OBJECT_STORAGE_SDK_TOTAL_MAX_ATTEMPTS", 4
            ),
            connect_timeout_seconds=_env_float(
                "LIGHTRAG_OBJECT_STORAGE_CONNECT_TIMEOUT_SECONDS", 5.0
            ),
            read_timeout_seconds=_env_float(
                "LIGHTRAG_OBJECT_STORAGE_READ_TIMEOUT_SECONDS", 30.0
            ),
            request_checksum_calculation=os.getenv(
                "LIGHTRAG_OBJECT_STORAGE_REQUEST_CHECKSUM_CALCULATION",
                "when_required",
            )
            .strip()
            .lower(),
            response_checksum_validation=os.getenv(
                "LIGHTRAG_OBJECT_STORAGE_RESPONSE_CHECKSUM_VALIDATION",
                "when_required",
            )
            .strip()
            .lower(),
        )


@dataclass(frozen=True, slots=True)
class ObjectStat:
    size: int
    etag: str | None = None
    checksum: str | None = None
    checksum_algorithm: str | None = None
    version_id: str | None = None
    last_modified: datetime | None = None

    def __post_init__(self) -> None:
        _validate_nonnegative_size(self.size)
        if self.last_modified is not None:
            object.__setattr__(
                self,
                "last_modified",
                _normalize_utc_datetime(self.last_modified),
            )


@dataclass(frozen=True, slots=True)
class ObjectListEntry:
    """One bounded metadata-only object listing entry."""

    uri: str
    key: str
    size: int
    last_modified: datetime
    etag: str | None = None
    version_id: str | None = None
    checksum: str | None = None
    checksum_algorithm: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.uri, str) or not self.uri:
            raise ValueError("Object list entry URI must be non-empty")
        _validate_raw_object_key(
            self.key,
            label="object list entry key",
            allow_empty=False,
            allow_trailing_slash=True,
        )
        _validate_nonnegative_size(self.size)
        object.__setattr__(
            self,
            "last_modified",
            _normalize_utc_datetime(self.last_modified),
        )


@dataclass(frozen=True, slots=True)
class ObjectListPage:
    entries: tuple[ObjectListEntry, ...]
    next_token: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple) or not all(
            isinstance(entry, ObjectListEntry) for entry in self.entries
        ):
            raise ValueError("Object list page entries must be an immutable tuple")
        if self.next_token is not None and (
            not isinstance(self.next_token, str) or not self.next_token
        ):
            raise ValueError("Object list page token must be opaque non-empty text")


@dataclass(frozen=True, slots=True)
class ObjectReadback:
    """Metadata-only presence proof for an exact object or version."""

    present: bool
    stat: ObjectStat | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.present, bool):
            raise ValueError("Object readback presence must be boolean")
        if self.present != (self.stat is not None):
            raise ValueError("Object readback presence and metadata do not match")


ArtifactCleanupTargetKind = Literal["object", "prefix"]
ArtifactCleanupTargetNamespace = Literal[
    "source", "legacy_source", "artifact", "staging", "workspace"
]


@dataclass(frozen=True, slots=True)
class ArtifactCleanupTarget:
    """Validated, storage-bound authority for one cleanup side effect."""

    uri: str
    bucket: str
    key: str
    kind: ArtifactCleanupTargetKind
    namespace: ArtifactCleanupTargetNamespace
    workspace: str
    document_id: str | None = None
    artifact_id: str | None = None
    source_generation_id: str | None = None
    origin_job_id: str | None = None
    origin_attempt_token: str | None = None
    _validation_tag: str = field(default="", repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class VerifiedDeleteResult:
    """Positive absence proof produced by a validated cleanup primitive."""

    absent: bool
    already_absent: bool
    deleted_entries: int
    pages_examined: int
    version_aware: bool

    def __post_init__(self) -> None:
        if not isinstance(self.absent, bool) or not self.absent:
            raise ValueError("Verified delete results must prove absence")
        if not isinstance(self.already_absent, bool) or not isinstance(
            self.version_aware, bool
        ):
            raise ValueError("Verified delete flags must be boolean")
        for field_name in ("deleted_entries", "pages_examined"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")


class ObjectStorageError(RuntimeError):
    pass


class ObjectStorageProofError(ObjectStorageError):
    """Safe, classified failure while establishing object absence."""

    error_code = "object_storage_proof_error"
    retryable = False

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.error_code.replace("_", " "))


class ObjectStorageNotFoundError(ObjectStorageProofError):
    error_code = "object_absent"

    def __init__(self) -> None:
        super().__init__("missing object")


class ObjectStorageForbiddenError(ObjectStorageProofError):
    error_code = "object_presence_unprovable"


class ObjectStorageTransportError(ObjectStorageProofError):
    error_code = "object_storage_transport"
    retryable = True


class ObjectStorageMalformedResponseError(ObjectStorageProofError):
    error_code = "object_storage_malformed_response"


class ObjectStorageOwnershipError(ObjectStorageProofError):
    error_code = "object_ownership_conflict"


class ObjectStorageIntegrityError(ObjectStorageProofError):
    error_code = "object_integrity_conflict"


class ObjectStorageVersionProofError(ObjectStorageProofError):
    error_code = "object_version_proof_unavailable"


class ObjectStorageStillPresentError(ObjectStorageProofError):
    error_code = "object_still_present"
    retryable = True


class ObjectStoragePageBudgetError(ObjectStorageProofError):
    error_code = "object_prefix_page_budget"
    retryable = True


class ObjectStorageDeleteError(ObjectStorageProofError):
    error_code = "object_delete_failed"


class ObjectStorage:
    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def upload_file(
        self, local_path: Path, *, key: str, content_type: str | None = None
    ) -> str:
        raise NotImplementedError

    async def upload_directory(self, local_dir: Path, *, prefix: str) -> str:
        raise NotImplementedError

    async def upload_file_if_absent(
        self,
        local_path: Path,
        *,
        key: str,
        content_type: str | None = None,
        expected_sha256: str | None = None,
    ) -> tuple[str, bool]:
        """Create an immutable object without overwriting an existing key.

        Returns ``(uri, created)``. Backends must make the create conditional;
        callers verify the bytes when ``created`` is false.
        """

        raise NotImplementedError

    def object_uri_for_key(self, key: str) -> str:
        raise NotImplementedError

    def object_prefix_uri_for_key(self, prefix: str) -> str:
        raise NotImplementedError

    async def download_file(self, object_uri: str, local_path: Path) -> None:
        raise NotImplementedError

    async def stat_object(self, object_uri: str) -> ObjectStat:
        raise NotImplementedError

    async def inspect_object(
        self, object_uri: str, *, version_id: str | None = None
    ) -> ObjectReadback:
        raise NotImplementedError

    async def list_objects_page(
        self,
        prefix_uri: str,
        *,
        max_keys: int = 1000,
        continuation_token: str | None = None,
    ) -> ObjectListPage:
        raise NotImplementedError

    async def download_prefix(
        self,
        prefix_uri: str,
        local_dir: Path,
        *,
        max_objects: int | None = None,
        max_total_bytes: int | None = None,
    ) -> int:
        raise NotImplementedError

    async def delete_uri(self, object_uri: str) -> bool:
        raise NotImplementedError

    async def delete_prefix(self, prefix_uri: str) -> int:
        raise NotImplementedError

    async def delete_workspace(self, workspace: str) -> int:
        raise NotImplementedError

    def validate_cleanup_target(
        self,
        target_uri: str,
        *,
        target_kind: ArtifactCleanupTargetKind,
        target_namespace: ArtifactCleanupTargetNamespace,
        workspace: str,
        document_id: str | None = None,
        artifact_id: str | None = None,
        source_generation_id: str | None = None,
        origin_job_id: str | None = None,
        origin_attempt_token: str | None = None,
    ) -> ArtifactCleanupTarget:
        raise NotImplementedError

    async def verified_delete_cleanup_target(
        self,
        target: ArtifactCleanupTarget,
        *,
        expected_size_bytes: int | None = None,
        expected_checksum: str | None = None,
        expected_etag: str | None = None,
        expected_version_id: str | None = None,
        object_page_size: int = 1000,
        delete_batch_size: int = 1000,
        max_prefix_pages: int = 32,
        before_exact_step: Callable[[], Awaitable[None]] | None = None,
        before_prefix_page: Callable[[], Awaitable[None]] | None = None,
    ) -> VerifiedDeleteResult:
        raise NotImplementedError

    async def presign_download_url(
        self, object_uri: str, *, expires_in_seconds: int = 3600
    ) -> str:
        raise NotImplementedError

    def validate_document_file_uri(
        self,
        object_uri: str,
        *,
        workspace: str,
        document_id: str,
        namespace: str | None = None,
        artifact_id: str | None = None,
    ) -> None:
        return None

    def validate_document_prefix_uri(
        self,
        prefix_uri: str,
        *,
        workspace: str,
        document_id: str,
        namespace: str | None = None,
        artifact_id: str | None = None,
    ) -> None:
        return None

    async def readiness_probe(self) -> bool:
        """Return True when the backend is reachable and writable enough for /health.

        The default is a safe ``False`` so any backend that does not override
        the probe (including lightweight test doubles) reports not-ready
        rather than raising. Implementations must NEVER raise — /health wraps
        every probe in ``wait_for`` but a raising probe would still surface in
        other callers. On success return ``True``; on any transport/auth error
        return ``False``.
        """

        return False


class DisabledObjectStorage(ObjectStorage):
    async def upload_file(
        self, local_path: Path, *, key: str, content_type: str | None = None
    ) -> str:
        raise ObjectStorageError("Object storage is disabled")

    async def upload_directory(self, local_dir: Path, *, prefix: str) -> str:
        raise ObjectStorageError("Object storage is disabled")

    async def upload_file_if_absent(
        self,
        local_path: Path,
        *,
        key: str,
        content_type: str | None = None,
        expected_sha256: str | None = None,
    ) -> tuple[str, bool]:
        raise ObjectStorageError("Object storage is disabled")

    def object_uri_for_key(self, key: str) -> str:
        raise ObjectStorageError("Object storage is disabled")

    def object_prefix_uri_for_key(self, prefix: str) -> str:
        raise ObjectStorageError("Object storage is disabled")

    async def download_file(self, object_uri: str, local_path: Path) -> None:
        raise ObjectStorageError("Object storage is disabled")

    async def stat_object(self, object_uri: str) -> ObjectStat:
        raise ObjectStorageError("Object storage is disabled")

    async def inspect_object(
        self, object_uri: str, *, version_id: str | None = None
    ) -> ObjectReadback:
        raise ObjectStorageError("Object storage is disabled")

    async def list_objects_page(
        self,
        prefix_uri: str,
        *,
        max_keys: int = 1000,
        continuation_token: str | None = None,
    ) -> ObjectListPage:
        raise ObjectStorageError("Object storage is disabled")

    async def download_prefix(
        self,
        prefix_uri: str,
        local_dir: Path,
        *,
        max_objects: int | None = None,
        max_total_bytes: int | None = None,
    ) -> int:
        raise ObjectStorageError("Object storage is disabled")

    async def delete_uri(self, object_uri: str) -> bool:
        return False

    async def delete_prefix(self, prefix_uri: str) -> int:
        return 0

    async def delete_workspace(self, workspace: str) -> int:
        return 0

    def validate_cleanup_target(
        self,
        target_uri: str,
        *,
        target_kind: ArtifactCleanupTargetKind,
        target_namespace: ArtifactCleanupTargetNamespace,
        workspace: str,
        document_id: str | None = None,
        artifact_id: str | None = None,
        source_generation_id: str | None = None,
        origin_job_id: str | None = None,
        origin_attempt_token: str | None = None,
    ) -> ArtifactCleanupTarget:
        raise ObjectStorageError("Object storage is disabled")

    async def verified_delete_cleanup_target(
        self,
        target: ArtifactCleanupTarget,
        *,
        expected_size_bytes: int | None = None,
        expected_checksum: str | None = None,
        expected_etag: str | None = None,
        expected_version_id: str | None = None,
        object_page_size: int = 1000,
        delete_batch_size: int = 1000,
        max_prefix_pages: int = 32,
        before_exact_step: Callable[[], Awaitable[None]] | None = None,
        before_prefix_page: Callable[[], Awaitable[None]] | None = None,
    ) -> VerifiedDeleteResult:
        raise ObjectStorageError("Object storage is disabled")

    async def presign_download_url(
        self, object_uri: str, *, expires_in_seconds: int = 3600
    ) -> str:
        raise ObjectStorageError("Object storage is disabled")

    def validate_document_file_uri(
        self,
        object_uri: str,
        *,
        workspace: str,
        document_id: str,
        namespace: str | None = None,
        artifact_id: str | None = None,
    ) -> None:
        raise ObjectStorageError("Object storage is disabled")

    def validate_document_prefix_uri(
        self,
        prefix_uri: str,
        *,
        workspace: str,
        document_id: str,
        namespace: str | None = None,
        artifact_id: str | None = None,
    ) -> None:
        raise ObjectStorageError("Object storage is disabled")

    async def readiness_probe(self) -> bool:
        # Disabled storage is never ready for an object-authoritative health
        # signal. Returning ``False`` (instead of raising) keeps /health fast
        # and non-blocking in local mode.
        return False


class S3ObjectStorage(ObjectStorage):
    """S3-compatible object storage used for MinIO deployments."""

    def __init__(self, config: ObjectStorageConfig):
        _validate_bucket_name(config.bucket)
        self._configured_prefix = _validate_raw_object_key(
            config.prefix,
            label="LIGHTRAG_OBJECT_STORAGE_PREFIX",
            allow_empty=True,
            allow_trailing_slash=False,
        )
        _validate_sdk_configuration(config)
        self._config = config
        self._session: Any | None = None
        self._target_validation_secret = secrets.token_bytes(32)
        # Cached readiness probe (monotonic-clock TTL). Populated lazily by
        # ``readiness_probe``; ``-inf`` sentinel forces a fresh first probe.
        self._readiness_probe_value: bool = False
        self._readiness_probe_at: float = float("-inf")

    async def initialize(self) -> None:
        self._session = self._new_session()
        async with self._client() as client:
            try:
                await client.head_bucket(Bucket=self._config.bucket)
            except Exception as exc:
                if not self._config.create_bucket or not _is_bucket_not_found_error(
                    exc
                ):
                    raise
                await client.create_bucket(Bucket=self._config.bucket)
                await client.head_bucket(Bucket=self._config.bucket)

    async def readiness_probe(self) -> bool:
        """Bounded, cached HeadBucket probe for /health (never raises).

        Issues at most one ``head_bucket`` per
        :data:`_S3_READINESS_PROBE_TTL_SECONDS` window and caches the result.
        On success the cached value is ``True``; on ANY exception
        (transport, auth, malformed response) it is ``False``. This method
        NEVER lists bucket contents or downloads objects — it is a single
        metadata-only reachability check.
        """

        now = time.monotonic()
        if now - self._readiness_probe_at < _S3_READINESS_PROBE_TTL_SECONDS:
            return self._readiness_probe_value
        try:
            async with self._client() as client:
                await client.head_bucket(Bucket=self._config.bucket)
            self._readiness_probe_value = True
        except Exception:  # noqa: BLE001 - probe must never raise
            self._readiness_probe_value = False
        self._readiness_probe_at = now
        return self._readiness_probe_value

    async def upload_file(
        self, local_path: Path, *, key: str, content_type: str | None = None
    ) -> str:
        object_key = self._normalize_key(key)
        extra_args: dict[str, Any] = {}
        media_type = content_type or mimetypes.guess_type(local_path.name)[0]
        if media_type:
            extra_args["ContentType"] = media_type
        async with self._client() as client:
            await client.upload_file(
                str(local_path),
                self._config.bucket,
                object_key,
                ExtraArgs=extra_args or None,
            )
        return self._uri(object_key)

    async def upload_directory(self, local_dir: Path, *, prefix: str) -> str:
        if not local_dir.is_dir():
            raise ObjectStorageError(f"Directory not found: {local_dir}")
        object_prefix = self._normalize_prefix(prefix)
        async with self._client() as client:
            for path in sorted(local_dir.rglob("*")):
                if not path.is_file() or path.is_symlink():
                    continue
                rel = path.relative_to(local_dir).as_posix()
                key = _validate_raw_object_key(
                    f"{object_prefix}/{rel}",
                    label="object key",
                    allow_empty=False,
                    allow_trailing_slash=False,
                )
                extra_args = {}
                media_type = mimetypes.guess_type(path.name)[0]
                if media_type:
                    extra_args["ContentType"] = media_type
                await client.upload_file(
                    str(path),
                    self._config.bucket,
                    key,
                    ExtraArgs=extra_args or None,
                )
        return self._prefix_uri(object_prefix)

    async def upload_file_if_absent(
        self,
        local_path: Path,
        *,
        key: str,
        content_type: str | None = None,
        expected_sha256: str | None = None,
    ) -> tuple[str, bool]:
        """Conditionally create and, when requested, prove an immutable object.

        ``expected_sha256`` activates a metadata-only size/SHA-256 proof for every
        successful outcome, including a conditional-create loser and a lost PUT
        acknowledgement.  The boolean is ``True`` only when a non-ambiguous PUT
        acknowledgement was received; an ACK-loss readback winner is deliberately
        reported as pre-existing because exclusive creation ownership is unknown.
        """

        if not local_path.is_file() or local_path.is_symlink():
            raise ObjectStorageError(f"File not found: {local_path}")
        expected_digest: str | None = None
        local_size: int | None = None
        if expected_sha256 is not None:
            expected_digest = _normalize_sha256_checksum(
                expected_sha256,
                allow_base64=False,
            )
            if expected_digest is None:
                raise ObjectStorageIntegrityError(
                    "Expected immutable upload checksum is not canonical SHA-256"
                )
            local_size, local_digest = _local_file_sha256_authority(local_path)
            if local_digest != expected_digest:
                raise ObjectStorageIntegrityError(
                    "Local immutable upload bytes do not match expected SHA-256"
                )
        object_key = self._normalize_key(key)
        kwargs: dict[str, Any] = {
            "Bucket": self._config.bucket,
            "Key": object_key,
            "IfNoneMatch": "*",
        }
        media_type = content_type or mimetypes.guess_type(local_path.name)[0]
        if media_type:
            kwargs["ContentType"] = media_type
        if expected_digest is not None:
            kwargs["Metadata"] = {"sha256": expected_digest}
            kwargs["ChecksumSHA256"] = base64.b64encode(
                bytes.fromhex(expected_digest)
            ).decode("ascii")

        object_uri = self._uri(object_key)
        async with self._client() as client:
            created = False
            ambiguous_error: ObjectStorageProofError | None = None
            try:
                try:
                    with local_path.open("rb") as body:
                        await client.put_object(Body=body, **kwargs)
                except Exception as exc:
                    if (
                        expected_digest is None
                        or not _is_checksum_request_unsupported_error(exc)
                    ):
                        raise
                    fallback_kwargs = dict(kwargs)
                    fallback_kwargs.pop("ChecksumSHA256", None)
                    with local_path.open("rb") as body:
                        await client.put_object(Body=body, **fallback_kwargs)
                created = True
            except Exception as exc:
                if _is_precondition_failed_error(exc):
                    created = False
                elif expected_digest is None:
                    raise
                else:
                    ambiguous_error = _classify_s3_exception(exc)

            if expected_digest is None:
                return object_uri, created

            readback = await self._head_object_readback(
                client,
                bucket=self._config.bucket,
                key=object_key,
                version_id=None,
            )
            if not readback.present or readback.stat is None:
                if ambiguous_error is not None:
                    raise ambiguous_error
                raise ObjectStorageNotFoundError()
            assert local_size is not None
            _compare_expected_object_metadata(
                readback.stat,
                expected_size_bytes=local_size,
                expected_checksum=expected_digest,
                expected_etag=None,
                expected_version_id=None,
            )
            return object_uri, created if ambiguous_error is None else False

    def object_uri_for_key(self, key: str) -> str:
        return self._uri(self._normalize_key(key))

    def object_prefix_uri_for_key(self, prefix: str) -> str:
        return self._prefix_uri(self._normalize_prefix(prefix.rstrip("/")))

    async def download_file(self, object_uri: str, local_path: Path) -> None:
        bucket, key = self._parse_uri(object_uri)
        if key.endswith("/"):
            raise ObjectStorageError("Object URI points to a prefix, not a file")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._client() as client:
            await client.download_file(bucket, key, str(local_path))

    async def stat_object(self, object_uri: str) -> ObjectStat:
        readback = await self.inspect_object(object_uri)
        if not readback.present or readback.stat is None:
            raise ObjectStorageNotFoundError()
        return readback.stat

    async def inspect_object(
        self, object_uri: str, *, version_id: str | None = None
    ) -> ObjectReadback:
        bucket, key = self._parse_uri(object_uri)
        if key.endswith("/"):
            raise ObjectStorageOwnershipError(
                "Object metadata inspection requires an exact object"
            )
        if version_id is not None:
            version_id = _validate_opaque_identity("version_id", version_id)
        async with self._client() as client:
            return await self._head_object_readback(
                client,
                bucket=bucket,
                key=key,
                version_id=version_id,
            )

    async def list_objects_page(
        self,
        prefix_uri: str,
        *,
        max_keys: int = 1000,
        continuation_token: str | None = None,
    ) -> ObjectListPage:
        max_keys = _validate_page_limit("max_keys", max_keys, maximum=1000)
        bucket, key = self._parse_uri(prefix_uri)
        if not key.endswith("/"):
            raise ObjectStorageOwnershipError(
                "Object listing requires an exact trailing-slash prefix"
            )
        backend_token = None
        after_key = None
        if continuation_token is not None:
            backend_token, after_key = self._decode_listing_token(
                continuation_token,
                bucket=bucket,
                prefix=key,
                max_keys=max_keys,
            )
        kwargs: dict[str, Any] = {
            "Bucket": bucket,
            "Prefix": key,
            "MaxKeys": max_keys,
        }
        if backend_token is not None:
            kwargs["ContinuationToken"] = backend_token
        try:
            async with self._client() as client:
                response = await client.list_objects_v2(**kwargs)
        except Exception as exc:
            raise _classify_s3_exception(exc) from exc
        return self._parse_object_list_page(
            response,
            bucket=bucket,
            prefix=key,
            max_keys=max_keys,
            after_key=after_key,
        )

    async def download_prefix(
        self,
        prefix_uri: str,
        local_dir: Path,
        *,
        max_objects: int | None = None,
        max_total_bytes: int | None = None,
    ) -> int:
        _validate_optional_limit("max_objects", max_objects)
        _validate_optional_limit("max_total_bytes", max_total_bytes)
        bucket, parsed_prefix = self._parse_uri(prefix_uri)
        prefix = parsed_prefix.rstrip("/") + "/"
        local_root = _prepare_local_download_root(local_dir)

        objects: list[tuple[str, str, int | None, bool]] = []
        advertised_total = 0
        seen_relative_keys: set[str] = set()
        async with self._client() as client:
            continuation_token: str | None = None
            while True:
                kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
                if continuation_token:
                    kwargs["ContinuationToken"] = continuation_token
                page = await client.list_objects_v2(**kwargs)
                for item in page.get("Contents", []):
                    key = item.get("Key")
                    if not isinstance(key, str):
                        raise ObjectStorageError(
                            "Object listing returned an invalid key"
                        )
                    _validate_raw_object_key(
                        key,
                        label="listed object key",
                        allow_empty=False,
                        allow_trailing_slash=True,
                    )
                    if not key.startswith(prefix):
                        raise ObjectStorageError(
                            "Listed object key escapes the requested prefix"
                        )
                    relative_key = key[len(prefix) :]
                    if not relative_key:
                        continue
                    _validate_raw_object_key(
                        relative_key,
                        label="relative object key",
                        allow_empty=False,
                        allow_trailing_slash=True,
                    )
                    if relative_key in seen_relative_keys:
                        raise ObjectStorageError(
                            "Object listing returned a duplicate target key"
                        )
                    seen_relative_keys.add(relative_key)

                    if max_objects is not None and len(objects) + 1 > max_objects:
                        raise ObjectStorageError(
                            "Object prefix exceeds max_objects limit"
                        )
                    raw_size = item.get("Size")
                    advertised_size = (
                        raw_size
                        if isinstance(raw_size, int) and not isinstance(raw_size, bool)
                        else None
                    )
                    if advertised_size is not None:
                        if advertised_size < 0:
                            raise ObjectStorageError(
                                "Object listing returned a negative size"
                            )
                        advertised_total += advertised_size
                        if (
                            max_total_bytes is not None
                            and advertised_total > max_total_bytes
                        ):
                            raise ObjectStorageError(
                                "Object prefix exceeds max_total_bytes limit"
                            )
                    objects.append(
                        (key, relative_key, advertised_size, key.endswith("/"))
                    )
                if not page.get("IsTruncated"):
                    break
                continuation_token = page.get("NextContinuationToken")
                if not continuation_token:
                    raise ObjectStorageError(
                        "Truncated object listing omitted continuation token"
                    )

            downloaded = 0
            actual_total = 0
            for key, relative_key, _advertised_size, is_directory_marker in objects:
                target = local_root.joinpath(*relative_key.rstrip("/").split("/"))
                _assert_safe_local_target(local_root, target)
                if is_directory_marker:
                    if target.exists() and not target.is_dir():
                        raise ObjectStorageError(
                            "Object directory marker conflicts with a local file"
                        )
                    target.mkdir(parents=True, exist_ok=True)
                    _assert_safe_local_target(local_root, target)
                    continue

                target.parent.mkdir(parents=True, exist_ok=True)
                _assert_safe_local_target(local_root, target)
                if target.is_symlink():
                    raise ObjectStorageError(
                        "Object download target cannot be a symlink"
                    )
                temp_target = target.parent / f".lightrag-{uuid4().hex}.part"
                _assert_safe_local_target(local_root, temp_target)
                try:
                    await client.download_file(bucket, key, str(temp_target))
                    _assert_safe_local_target(local_root, temp_target)
                    if temp_target.is_symlink() or not temp_target.is_file():
                        raise ObjectStorageError(
                            "Object download did not create a regular file"
                        )
                    mode = temp_target.stat().st_mode
                    if not stat.S_ISREG(mode):
                        raise ObjectStorageError(
                            "Object download did not create a regular file"
                        )
                    size = temp_target.stat().st_size
                    if (
                        max_total_bytes is not None
                        and actual_total + size > max_total_bytes
                    ):
                        raise ObjectStorageError(
                            "Object prefix exceeds max_total_bytes limit"
                        )
                    _assert_safe_local_target(local_root, target)
                    if target.is_symlink() or target.is_dir():
                        raise ObjectStorageError(
                            "Object download target is not a safe file path"
                        )
                    os.replace(temp_target, target)
                    actual_total += size
                    downloaded += 1
                finally:
                    if temp_target.exists() and not temp_target.is_symlink():
                        temp_target.unlink()
        return downloaded

    async def delete_uri(self, object_uri: str) -> bool:
        bucket, key = self._parse_uri(object_uri)
        async with self._client() as client:
            await client.delete_object(Bucket=bucket, Key=key)
        return True

    async def delete_prefix(self, prefix_uri: str) -> int:
        bucket, prefix = self._parse_uri(prefix_uri)
        if prefix and not prefix.endswith("/"):
            prefix = f"{prefix}/"
        deleted = 0
        async with self._client() as client:
            while True:
                page = await client.list_objects_v2(
                    Bucket=bucket,
                    Prefix=prefix,
                    MaxKeys=1000,
                )
                objects = _validated_delete_keys_from_list_response(
                    page,
                    prefix=prefix,
                    maximum=1000,
                )
                if objects:
                    response = await client.delete_objects(
                        Bucket=bucket,
                        Delete={"Objects": objects, "Quiet": False},
                    )
                    _raise_for_delete_objects_errors(response)
                    deleted += len(objects)
                if not objects:
                    break
        return deleted

    async def delete_workspace(self, workspace: str) -> int:
        _validate_scope_component("workspace", workspace)
        return await self.delete_prefix(
            self._prefix_uri(self._normalize_key(f"workspaces/{workspace}"))
        )

    def validate_cleanup_target(
        self,
        target_uri: str,
        *,
        target_kind: ArtifactCleanupTargetKind,
        target_namespace: ArtifactCleanupTargetNamespace,
        workspace: str,
        document_id: str | None = None,
        artifact_id: str | None = None,
        source_generation_id: str | None = None,
        origin_job_id: str | None = None,
        origin_attempt_token: str | None = None,
    ) -> ArtifactCleanupTarget:
        if target_kind not in {"object", "prefix"}:
            raise ObjectStorageOwnershipError("Cleanup target kind is invalid")
        if target_namespace not in {
            "source",
            "legacy_source",
            "artifact",
            "staging",
            "workspace",
        }:
            raise ObjectStorageOwnershipError("Cleanup target namespace is invalid")
        workspace = _validate_scope_component("workspace", workspace)
        document_id = _validate_optional_scope_component("document_id", document_id)
        artifact_id = _validate_optional_scope_component("artifact_id", artifact_id)
        source_generation_id = _validate_optional_scope_component(
            "source_generation_id", source_generation_id
        )
        origin_job_id = _validate_optional_scope_component(
            "origin_job_id", origin_job_id
        )
        origin_attempt_token = _validate_optional_scope_component(
            "origin_attempt_token", origin_attempt_token
        )

        bucket, key = self._parse_uri(target_uri)
        is_prefix = key.endswith("/")
        if (target_kind == "prefix") != is_prefix:
            raise ObjectStorageOwnershipError(
                "Cleanup target kind does not match trailing-slash semantics"
            )
        key_parts = key.rstrip("/").split("/")
        configured_parts = (
            self._configured_prefix.split("/") if self._configured_prefix else []
        )
        workspace_parts = [*configured_parts, "workspaces", workspace]
        if key_parts[: len(workspace_parts)] != workspace_parts:
            raise ObjectStorageOwnershipError(
                "Cleanup target is outside the owned workspace namespace"
            )
        remainder = key_parts[len(workspace_parts) :]

        if target_namespace == "workspace":
            if (
                target_kind != "prefix"
                or remainder
                or document_id is not None
                or artifact_id is not None
                or source_generation_id is not None
            ):
                raise ObjectStorageOwnershipError(
                    "Workspace cleanup requires the exact workspace prefix"
                )
        elif target_namespace == "staging":
            if (
                origin_job_id is None
                or origin_attempt_token is None
                or document_id is not None
                or artifact_id is not None
                or source_generation_id is not None
            ):
                raise ObjectStorageOwnershipError(
                    "Staging cleanup requires exact job and attempt authority"
                )
            expected = ["staging", origin_job_id, origin_attempt_token]
            if remainder[:3] != expected:
                raise ObjectStorageOwnershipError(
                    "Staging cleanup target does not match origin authority"
                )
            if target_kind == "object" and len(remainder) < 4:
                raise ObjectStorageOwnershipError(
                    "Staging object cleanup requires an exact object"
                )
        else:
            if document_id is None:
                raise ObjectStorageOwnershipError(
                    "Document cleanup target requires a document id"
                )
            document_parts = ["documents", document_id]
            if remainder[:2] != document_parts:
                raise ObjectStorageOwnershipError(
                    "Cleanup target is outside the owned document namespace"
                )
            document_remainder = remainder[2:]
            if target_namespace == "source":
                if artifact_id is not None or source_generation_id is None:
                    raise ObjectStorageOwnershipError(
                        "Source cleanup requires exact generation authority"
                    )
                expected = ["source", "generations", source_generation_id]
                if document_remainder[:3] != expected:
                    raise ObjectStorageOwnershipError(
                        "Source cleanup target does not match the source generation"
                    )
                if target_kind == "object" and len(document_remainder) < 4:
                    raise ObjectStorageOwnershipError(
                        "Source cleanup requires an exact object or generation prefix"
                    )
            elif target_namespace == "legacy_source":
                if (
                    target_kind != "object"
                    or artifact_id is not None
                    or source_generation_id is not None
                    or len(document_remainder) != 2
                    or document_remainder[0] != "source"
                    or document_remainder[1] == "generations"
                ):
                    raise ObjectStorageOwnershipError(
                        "Legacy source cleanup requires one exact historical object"
                    )
            else:
                if artifact_id is None or source_generation_id is not None:
                    raise ObjectStorageOwnershipError(
                        "Artifact cleanup requires exact artifact authority"
                    )
                if (
                    len(document_remainder) < 3
                    or document_remainder[0] != "artifacts"
                    or document_remainder[2] != artifact_id
                ):
                    raise ObjectStorageOwnershipError(
                        "Artifact cleanup target does not match the artifact id"
                    )
                if target_kind == "object" and len(document_remainder) < 4:
                    raise ObjectStorageOwnershipError(
                        "Artifact object cleanup requires an exact object"
                    )

        canonical_uri = self._uri_for_parsed_key(key)
        target = ArtifactCleanupTarget(
            uri=canonical_uri,
            bucket=bucket,
            key=key,
            kind=target_kind,
            namespace=target_namespace,
            workspace=workspace,
            document_id=document_id,
            artifact_id=artifact_id,
            source_generation_id=source_generation_id,
            origin_job_id=origin_job_id,
            origin_attempt_token=origin_attempt_token,
        )
        return ArtifactCleanupTarget(
            **{
                field_name: getattr(target, field_name)
                for field_name in (
                    "uri",
                    "bucket",
                    "key",
                    "kind",
                    "namespace",
                    "workspace",
                    "document_id",
                    "artifact_id",
                    "source_generation_id",
                    "origin_job_id",
                    "origin_attempt_token",
                )
            },
            _validation_tag=self._cleanup_target_tag(target),
        )

    async def verified_delete_cleanup_target(
        self,
        target: ArtifactCleanupTarget,
        *,
        expected_size_bytes: int | None = None,
        expected_checksum: str | None = None,
        expected_etag: str | None = None,
        expected_version_id: str | None = None,
        object_page_size: int = 1000,
        delete_batch_size: int = 1000,
        max_prefix_pages: int = 32,
        before_exact_step: Callable[[], Awaitable[None]] | None = None,
        before_prefix_page: Callable[[], Awaitable[None]] | None = None,
    ) -> VerifiedDeleteResult:
        self._assert_cleanup_target(target)
        object_page_size = _validate_page_limit(
            "object_page_size", object_page_size, maximum=1000
        )
        delete_batch_size = _validate_page_limit(
            "delete_batch_size", delete_batch_size, maximum=1000
        )
        max_prefix_pages = _validate_page_limit(
            "max_prefix_pages", max_prefix_pages, maximum=100_000
        )
        expected_size_bytes = _validate_optional_nonnegative_int(
            "expected_size_bytes", expected_size_bytes
        )
        expected_checksum = _validate_optional_safe_text(
            "expected_checksum", expected_checksum
        )
        expected_etag = _validate_optional_safe_text("expected_etag", expected_etag)
        expected_version_id = _validate_optional_safe_text(
            "expected_version_id", expected_version_id
        )
        if target.kind == "prefix" and any(
            value is not None
            for value in (
                expected_size_bytes,
                expected_checksum,
                expected_etag,
                expected_version_id,
            )
        ):
            raise ObjectStorageIntegrityError(
                "Prefix cleanup cannot carry exact-object integrity expectations"
            )

        async with self._client() as client:
            versioned = await self._bucket_is_versioned(client, target.bucket)
            if target.kind == "object":
                return await self._verified_delete_exact_object(
                    client,
                    target,
                    versioned=versioned,
                    expected_size_bytes=expected_size_bytes,
                    expected_checksum=expected_checksum,
                    expected_etag=expected_etag,
                    expected_version_id=expected_version_id,
                    before_exact_step=before_exact_step,
                )
            return await self._verified_delete_prefix_target(
                client,
                target,
                versioned=versioned,
                object_page_size=object_page_size,
                delete_batch_size=delete_batch_size,
                max_prefix_pages=max_prefix_pages,
                before_prefix_page=before_prefix_page,
            )

    async def _verified_delete_exact_object(
        self,
        client: Any,
        target: ArtifactCleanupTarget,
        *,
        versioned: bool,
        expected_size_bytes: int | None,
        expected_checksum: str | None,
        expected_etag: str | None,
        expected_version_id: str | None,
        before_exact_step: Callable[[], Awaitable[None]] | None,
    ) -> VerifiedDeleteResult:
        if versioned and expected_version_id is None:
            raise ObjectStorageVersionProofError(
                "Versioned exact cleanup requires an expected version id"
            )
        if not versioned and expected_version_id is not None:
            raise ObjectStorageVersionProofError(
                "Expected version id cannot be proved in an unversioned bucket"
            )
        if before_exact_step is not None:
            await before_exact_step()
        readback = await self._head_object_readback(
            client,
            bucket=target.bucket,
            key=target.key,
            version_id=expected_version_id,
        )
        if not readback.present:
            if await self._bucket_is_versioned(client, target.bucket) != versioned:
                raise ObjectStorageVersionProofError(
                    "Bucket versioning state changed during exact cleanup proof"
                )
            return VerifiedDeleteResult(
                absent=True,
                already_absent=True,
                deleted_entries=0,
                pages_examined=0,
                version_aware=versioned,
            )
        assert readback.stat is not None
        _compare_expected_object_metadata(
            readback.stat,
            expected_size_bytes=expected_size_bytes,
            expected_checksum=expected_checksum,
            expected_etag=expected_etag,
            expected_version_id=expected_version_id,
        )
        kwargs: dict[str, Any] = {"Bucket": target.bucket, "Key": target.key}
        if expected_version_id is not None:
            kwargs["VersionId"] = expected_version_id
        if before_exact_step is not None:
            await before_exact_step()
        try:
            await client.delete_object(**kwargs)
        except Exception as exc:
            classified = _classify_s3_exception(exc)
            if not classified.retryable:
                raise classified from exc

        if before_exact_step is not None:
            await before_exact_step()
        post_delete = await self._head_object_readback(
            client,
            bucket=target.bucket,
            key=target.key,
            version_id=expected_version_id,
        )
        if post_delete.present:
            assert post_delete.stat is not None
            _compare_expected_object_metadata(
                post_delete.stat,
                expected_size_bytes=expected_size_bytes,
                expected_checksum=expected_checksum,
                expected_etag=expected_etag,
                expected_version_id=expected_version_id,
            )
            raise ObjectStorageStillPresentError()
        if await self._bucket_is_versioned(client, target.bucket) != versioned:
            raise ObjectStorageVersionProofError(
                "Bucket versioning state changed during exact cleanup proof"
            )
        return VerifiedDeleteResult(
            absent=True,
            already_absent=False,
            deleted_entries=1,
            pages_examined=0,
            version_aware=versioned,
        )

    async def _verified_delete_prefix_target(
        self,
        client: Any,
        target: ArtifactCleanupTarget,
        *,
        versioned: bool,
        object_page_size: int,
        delete_batch_size: int,
        max_prefix_pages: int,
        before_prefix_page: Callable[[], Awaitable[None]] | None,
    ) -> VerifiedDeleteResult:
        pages_examined = 0
        deleted_entries = 0
        pending_delete_error: ObjectStorageProofError | None = None
        while True:
            if pages_examined >= max_prefix_pages:
                raise ObjectStoragePageBudgetError()
            if before_prefix_page is not None:
                await before_prefix_page()
            pages_examined += 1
            if versioned:
                entries = await self._list_first_version_page(
                    client,
                    bucket=target.bucket,
                    prefix=target.key,
                    max_keys=object_page_size,
                )
            else:
                entries = await self._list_first_current_page(
                    client,
                    bucket=target.bucket,
                    prefix=target.key,
                    max_keys=object_page_size,
                )
            if not entries:
                current_versioned = await self._bucket_is_versioned(
                    client, target.bucket
                )
                if current_versioned != versioned:
                    if versioned or not current_versioned:
                        raise ObjectStorageVersionProofError(
                            "Bucket versioning state changed during prefix proof"
                        )
                    versioned = True
                    continue
                return VerifiedDeleteResult(
                    absent=True,
                    already_absent=deleted_entries == 0,
                    deleted_entries=deleted_entries,
                    pages_examined=pages_examined,
                    version_aware=versioned,
                )
            if pending_delete_error is not None:
                raise pending_delete_error
            batch = entries[:delete_batch_size]
            if before_prefix_page is not None:
                await before_prefix_page()
            try:
                response = await client.delete_objects(
                    Bucket=target.bucket,
                    Delete={"Objects": batch, "Quiet": False},
                )
                pending_delete_error = _delete_objects_response_error(response)
            except Exception as exc:
                pending_delete_error = _classify_s3_exception(exc)
            deleted_entries += len(batch)

    async def presign_download_url(
        self, object_uri: str, *, expires_in_seconds: int = 3600
    ) -> str:
        if expires_in_seconds <= 0:
            raise ObjectStorageError("expires_in_seconds must be positive")
        bucket, key = self._parse_uri(object_uri)
        async with self._client() as client:
            url = client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=expires_in_seconds,
            )
            if inspect.isawaitable(url):
                url = await url
        return str(url)

    def validate_document_file_uri(
        self,
        object_uri: str,
        *,
        workspace: str,
        document_id: str,
        namespace: str | None = None,
        artifact_id: str | None = None,
    ) -> None:
        _bucket, key = self._parse_uri(object_uri)
        if key.endswith("/"):
            raise ObjectStorageError("Object URI points to a prefix, not a file")
        self._validate_document_scope(
            key,
            workspace=workspace,
            document_id=document_id,
            namespace=namespace,
            artifact_id=artifact_id,
        )

    def validate_document_prefix_uri(
        self,
        prefix_uri: str,
        *,
        workspace: str,
        document_id: str,
        namespace: str | None = None,
        artifact_id: str | None = None,
    ) -> None:
        _bucket, key = self._parse_uri(prefix_uri)
        self._validate_document_scope(
            key.rstrip("/"),
            workspace=workspace,
            document_id=document_id,
            namespace=namespace,
            artifact_id=artifact_id,
        )

    def _validate_document_scope(
        self,
        key: str,
        *,
        workspace: str,
        document_id: str,
        namespace: str | None,
        artifact_id: str | None,
    ) -> None:
        workspace = _validate_scope_component("workspace", workspace)
        document_id = _validate_scope_component("document_id", document_id)
        expected_namespace = None
        if namespace is not None:
            expected_namespace = str(namespace).strip().lower()
            if expected_namespace == "artifact":
                expected_namespace = "artifacts"
            if expected_namespace not in {"source", "artifacts"}:
                raise ObjectStorageError("Object namespace must be source or artifacts")
        if artifact_id is not None:
            artifact_id = _validate_scope_component("artifact_id", artifact_id)

        prefix_parts = (
            self._configured_prefix.split("/") if self._configured_prefix else []
        )
        parts = key.rstrip("/").split("/")
        document_parts = [
            *prefix_parts,
            "workspaces",
            workspace,
            "documents",
            document_id,
        ]
        if parts[: len(document_parts)] != document_parts:
            raise ObjectStorageError("Object URI is outside the document object prefix")
        remainder = parts[len(document_parts) :]
        if len(remainder) < 2:
            raise ObjectStorageError(
                "Object URI is missing a source/artifact namespace target"
            )
        actual_namespace = remainder[0]
        if actual_namespace not in {"source", "artifacts"}:
            raise ObjectStorageError(
                "Object URI has an invalid source/artifact namespace"
            )
        if expected_namespace is not None and actual_namespace != expected_namespace:
            raise ObjectStorageError(
                "Object URI namespace does not match expected namespace"
            )
        if actual_namespace == "source" and artifact_id is not None:
            raise ObjectStorageError("Source object URI cannot carry an artifact id")
        if actual_namespace == "artifacts" and artifact_id is not None:
            # Canonical artifact keys are:
            # artifacts/<artifact-type>/<artifact-id>/<file-or-prefix>/...
            if len(remainder) < 4 or remainder[2] != artifact_id:
                raise ObjectStorageError(
                    "Object URI artifact id does not match expected artifact"
                )

    def _uri_for_parsed_key(self, key: str) -> str:
        _validate_raw_object_key(
            key,
            label="object key",
            allow_empty=False,
            allow_trailing_slash=True,
        )
        return f"s3://{self._config.bucket}/{quote(key, safe='/')}"

    def _cleanup_target_tag(self, target: ArtifactCleanupTarget) -> str:
        payload = json.dumps(
            {
                "artifact_id": target.artifact_id,
                "bucket": target.bucket,
                "document_id": target.document_id,
                "key": target.key,
                "kind": target.kind,
                "namespace": target.namespace,
                "origin_attempt_token": target.origin_attempt_token,
                "origin_job_id": target.origin_job_id,
                "source_generation_id": target.source_generation_id,
                "uri": target.uri,
                "workspace": target.workspace,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hmac.new(
            self._target_validation_secret, payload, hashlib.sha256
        ).hexdigest()

    def _assert_cleanup_target(self, target: ArtifactCleanupTarget) -> None:
        if not isinstance(target, ArtifactCleanupTarget):
            raise ObjectStorageOwnershipError("Cleanup target was not validated")
        expected = self._cleanup_target_tag(target)
        if not target._validation_tag or not hmac.compare_digest(
            target._validation_tag, expected
        ):
            raise ObjectStorageOwnershipError(
                "Cleanup target does not belong to this storage validator"
            )

    def _encode_listing_token(
        self,
        *,
        bucket: str,
        prefix: str,
        max_keys: int,
        token: str,
        last_key: str,
    ) -> str:
        payload = json.dumps(
            {
                "bucket": bucket,
                "max_keys": max_keys,
                "prefix": prefix,
                "token": token,
                "last_key": last_key,
                "version": _LISTING_TOKEN_VERSION,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        integrity_digest = hashlib.sha256(
            _LISTING_TOKEN_INTEGRITY_DOMAIN + payload
        ).digest()
        return _LISTING_TOKEN_PREFIX + _urlsafe_b64encode(payload + integrity_digest)

    def _decode_listing_token(
        self, value: str, *, bucket: str, prefix: str, max_keys: int
    ) -> tuple[str, str]:
        if (
            not isinstance(value, str)
            or len(value) > 16_384
            or not value.startswith(_LISTING_TOKEN_PREFIX)
        ):
            raise ObjectStorageOwnershipError("Object listing token is invalid")
        try:
            decoded = _urlsafe_b64decode(value.removeprefix(_LISTING_TOKEN_PREFIX))
        except (ValueError, binascii.Error) as exc:
            raise ObjectStorageOwnershipError(
                "Object listing token is invalid"
            ) from exc
        if len(decoded) <= hashlib.sha256().digest_size:
            raise ObjectStorageOwnershipError("Object listing token is invalid")
        payload = decoded[: -hashlib.sha256().digest_size]
        integrity_digest = decoded[-hashlib.sha256().digest_size :]
        expected = hashlib.sha256(_LISTING_TOKEN_INTEGRITY_DOMAIN + payload).digest()
        if not hmac.compare_digest(integrity_digest, expected):
            raise ObjectStorageOwnershipError("Object listing token is invalid")
        try:
            data = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ObjectStorageOwnershipError(
                "Object listing token is invalid"
            ) from exc
        if not isinstance(data, dict):
            raise ObjectStorageOwnershipError("Object listing token is invalid")
        if (
            set(data)
            != {
                "bucket",
                "last_key",
                "max_keys",
                "prefix",
                "token",
                "version",
            }
            or data.get("version") != _LISTING_TOKEN_VERSION
        ):
            raise ObjectStorageOwnershipError("Object listing token is invalid")
        token = data.get("token")
        last_key = data.get("last_key")
        if (
            data.get("bucket") != bucket
            or data.get("max_keys") != max_keys
            or data.get("prefix") != prefix
        ):
            raise ObjectStorageOwnershipError(
                "Object listing token does not match the exact prefix request"
            )
        if (
            not isinstance(token, str)
            or not token
            or len(token) > 8192
            or any(ord(character) < 32 or ord(character) == 127 for character in token)
            or not isinstance(last_key, str)
            or not last_key
        ):
            raise ObjectStorageOwnershipError("Object listing token is invalid")
        try:
            _validate_raw_object_key(
                last_key,
                label="object listing token last key",
                allow_empty=False,
                allow_trailing_slash=True,
            )
        except ObjectStorageError as exc:
            raise ObjectStorageOwnershipError(
                "Object listing token is invalid"
            ) from exc
        if not last_key.startswith(prefix):
            raise ObjectStorageOwnershipError(
                "Object listing token does not match the exact prefix request"
            )
        return token, last_key

    def _parse_object_list_page(
        self,
        response: Any,
        *,
        bucket: str,
        prefix: str,
        max_keys: int,
        after_key: str | None = None,
    ) -> ObjectListPage:
        if not isinstance(response, dict):
            raise ObjectStorageMalformedResponseError()
        contents = response.get("Contents", [])
        if not isinstance(contents, list) or len(contents) > max_keys:
            raise ObjectStorageMalformedResponseError()
        truncated = response.get("IsTruncated")
        if not isinstance(truncated, bool):
            raise ObjectStorageMalformedResponseError()
        seen: set[str] = set()
        previous_key = after_key
        entries: list[ObjectListEntry] = []
        for item in contents:
            if not isinstance(item, dict):
                raise ObjectStorageMalformedResponseError()
            key = item.get("Key")
            if not isinstance(key, str) or not key:
                raise ObjectStorageMalformedResponseError()
            try:
                _validate_raw_object_key(
                    key,
                    label="listed object key",
                    allow_empty=False,
                    allow_trailing_slash=True,
                )
            except ObjectStorageError as exc:
                raise ObjectStorageMalformedResponseError() from exc
            if (
                not key.startswith(prefix)
                or key in seen
                or (previous_key is not None and key <= previous_key)
            ):
                raise ObjectStorageMalformedResponseError()
            seen.add(key)
            previous_key = key
            size = item.get("Size")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ObjectStorageMalformedResponseError()
            last_modified = _parse_utc_last_modified(item.get("LastModified"))
            etag = _optional_opaque_response_text(item.get("ETag"))
            version_id = _optional_opaque_response_text(item.get("VersionId"))
            checksum, checksum_algorithm = _listing_checksum_metadata(item)
            entries.append(
                ObjectListEntry(
                    uri=self._uri_for_parsed_key(key),
                    key=key,
                    size=size,
                    last_modified=last_modified,
                    etag=etag,
                    version_id=version_id,
                    checksum=checksum,
                    checksum_algorithm=checksum_algorithm,
                )
            )
        raw_next_token = response.get("NextContinuationToken")
        if truncated:
            if (
                not isinstance(raw_next_token, str)
                or not raw_next_token
                or len(raw_next_token) > 8192
                or any(
                    ord(character) < 32 or ord(character) == 127
                    for character in raw_next_token
                )
                or not entries
            ):
                raise ObjectStorageMalformedResponseError()
            next_token = self._encode_listing_token(
                bucket=bucket,
                prefix=prefix,
                max_keys=max_keys,
                token=raw_next_token,
                last_key=entries[-1].key,
            )
        else:
            next_token = None
        return ObjectListPage(entries=tuple(entries), next_token=next_token)

    async def _head_object_readback(
        self,
        client: Any,
        *,
        bucket: str,
        key: str,
        version_id: str | None,
    ) -> ObjectReadback:
        kwargs: dict[str, Any] = {
            "Bucket": bucket,
            "Key": key,
            "ChecksumMode": "ENABLED",
        }
        if version_id is not None:
            kwargs["VersionId"] = version_id
        checksum_requested = True
        try:
            response = await client.head_object(**kwargs)
        except Exception as exc:
            if _is_checksum_mode_unsupported_error(exc):
                kwargs.pop("ChecksumMode", None)
                checksum_requested = False
                try:
                    response = await client.head_object(**kwargs)
                except Exception as fallback_exc:
                    classified = _classify_s3_exception(fallback_exc)
                    if isinstance(classified, ObjectStorageNotFoundError):
                        return ObjectReadback(present=False)
                    raise classified from fallback_exc
            else:
                classified = _classify_s3_exception(exc)
                if isinstance(classified, ObjectStorageNotFoundError):
                    return ObjectReadback(present=False)
                raise classified from exc
        stat_result = _object_stat_from_head_response(
            response,
            checksum_requested=checksum_requested,
        )
        return ObjectReadback(present=True, stat=stat_result)

    async def _bucket_is_versioned(self, client: Any, bucket: str) -> bool:
        try:
            response = await client.get_bucket_versioning(Bucket=bucket)
        except Exception as exc:
            code, status = _s3_error_code_and_status(exc)
            if code in {"notimplemented", "notsupported", "unsupported"} or (
                status == 501
            ):
                raise ObjectStorageVersionProofError(
                    "Bucket versioning state cannot be proved"
                ) from exc
            raise _classify_s3_exception(exc) from exc
        if not isinstance(response, dict):
            raise ObjectStorageMalformedResponseError()
        status = response.get("Status")
        if status is None:
            return False
        if status in {"Enabled", "Suspended"}:
            return True
        raise ObjectStorageMalformedResponseError()

    async def _list_first_current_page(
        self,
        client: Any,
        *,
        bucket: str,
        prefix: str,
        max_keys: int,
    ) -> list[dict[str, str]]:
        try:
            response = await client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                MaxKeys=max_keys,
            )
        except Exception as exc:
            raise _classify_s3_exception(exc) from exc
        page = self._parse_object_list_page(
            response,
            bucket=bucket,
            prefix=prefix,
            max_keys=max_keys,
        )
        return [{"Key": entry.key} for entry in page.entries]

    async def _list_first_version_page(
        self,
        client: Any,
        *,
        bucket: str,
        prefix: str,
        max_keys: int,
    ) -> list[dict[str, str]]:
        try:
            response = await client.list_object_versions(
                Bucket=bucket,
                Prefix=prefix,
                MaxKeys=max_keys,
            )
        except Exception as exc:
            raise _classify_s3_exception(exc) from exc
        return _validated_version_delete_entries(
            response,
            prefix=prefix,
            maximum=max_keys,
        )

    def _new_session(self) -> Any:
        import importlib

        try:
            aioboto3 = importlib.import_module("aioboto3")
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise ObjectStorageError(
                "S3/MinIO object storage requires aioboto3. "
                "Install LightRAG with the api/offline-llm extras or install aioboto3."
            ) from exc
        return aioboto3.Session()

    @asynccontextmanager
    async def _client(self) -> AsyncIterator[Any]:
        session = self._session
        if session is None:
            session = self._new_session()
            self._session = session
        async with session.client(
            "s3",
            endpoint_url=self._config.endpoint_url,
            aws_access_key_id=self._config.access_key_id,
            aws_secret_access_key=self._config.secret_access_key,
            region_name=self._config.region_name,
            use_ssl=self._config.use_ssl,
            config=self._botocore_config(),
        ) as client:
            self._configure_client(client)
            yield client

    def _botocore_config(self) -> Any:
        import importlib

        try:
            Config = importlib.import_module("botocore.config").Config
        except ImportError as exc:  # pragma: no cover - aioboto3 depends on botocore
            raise ObjectStorageError(
                "S3/MinIO object storage requires botocore configuration support"
            ) from exc
        return Config(
            retries={
                "mode": self._config.sdk_retry_mode,
                "total_max_attempts": self._config.sdk_total_max_attempts,
            },
            connect_timeout=self._config.connect_timeout_seconds,
            read_timeout=self._config.read_timeout_seconds,
            request_checksum_calculation=self._config.request_checksum_calculation,
            response_checksum_validation=self._config.response_checksum_validation,
        )

    def _configure_client(self, client: Any) -> None:
        if not self._config.disable_expect_header:
            return
        events = getattr(getattr(client, "meta", None), "events", None)
        register = getattr(events, "register", None)
        if register is None:
            return
        register("before-send.s3.PutObject", self._remove_expect_header)
        register("before-send.s3.UploadPart", self._remove_expect_header)

    @staticmethod
    def _remove_expect_header(request: Any, **_: Any) -> None:
        headers = getattr(request, "headers", None)
        if headers is None:
            return
        if "Expect" in headers:
            del headers["Expect"]
            return
        for header_name in list(headers):
            if str(header_name).lower() == "expect":
                del headers[header_name]
                return

    def _normalize_key(self, key: str) -> str:
        normalized = _validate_raw_object_key(
            key,
            label="object key",
            allow_empty=False,
            allow_trailing_slash=False,
        )
        if self._configured_prefix:
            normalized = f"{self._configured_prefix}/{normalized}"
        return _validate_raw_object_key(
            normalized,
            label="object key",
            allow_empty=False,
            allow_trailing_slash=False,
        )

    def _normalize_prefix(self, prefix: str) -> str:
        return self._normalize_key(prefix)

    def _uri(self, key: str) -> str:
        key = _validate_raw_object_key(
            key,
            label="object key",
            allow_empty=False,
            allow_trailing_slash=False,
        )
        return f"s3://{self._config.bucket}/{quote(key, safe='/')}"

    def _prefix_uri(self, prefix: str) -> str:
        prefix = _validate_raw_object_key(
            prefix.rstrip("/"),
            label="object prefix",
            allow_empty=False,
            allow_trailing_slash=False,
        )
        return f"s3://{self._config.bucket}/{quote(prefix, safe='/')}/"

    def _parse_uri(self, object_uri: str) -> tuple[str, str]:
        parsed = urlparse(object_uri)
        if (
            parsed.scheme != "s3"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.params
        ):
            raise ObjectStorageError("Unsupported or unsafe object URI")
        if parsed.netloc != self._config.bucket:
            raise ObjectStorageError(
                "Object URI bucket does not match configured bucket"
            )
        if not parsed.path.startswith("/") or parsed.path.startswith("//"):
            raise ObjectStorageError("Object URI has an invalid key path")
        raw_key = parsed.path[1:]
        if not raw_key:
            raise ObjectStorageError("Object URI missing key")
        key = _decode_object_uri_key_once(raw_key)
        if not key:
            raise ObjectStorageError("Object URI missing key")
        key = _validate_raw_object_key(
            key,
            label="object URI key",
            allow_empty=False,
            allow_trailing_slash=True,
        )
        key_without_slash = key.rstrip("/")
        if self._configured_prefix and not (
            key_without_slash == self._configured_prefix
            or key_without_slash.startswith(f"{self._configured_prefix}/")
        ):
            raise ObjectStorageError(
                "Object URI key does not match configured object prefix"
            )
        return parsed.netloc, key


def create_object_storage(config: ObjectStorageConfig) -> ObjectStorage | None:
    if config.backend in {"", "local", "disabled", "none"}:
        return None
    if config.backend in {"s3", "minio"}:
        return S3ObjectStorage(config)
    raise ObjectStorageError(f"Unsupported LIGHTRAG_OBJECT_STORAGE: {config.backend}")


def create_object_storage_from_env() -> ObjectStorage | None:
    return create_object_storage(ObjectStorageConfig.from_env())


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except (TypeError, ValueError) as exc:
        raise ObjectStorageError(f"{name} must be an integer") from exc
    if isinstance(value, bool):  # pragma: no cover - int() never returns bool
        raise ObjectStorageError(f"{name} must be an integer")
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except (TypeError, ValueError) as exc:
        raise ObjectStorageError(f"{name} must be numeric") from exc


def _validate_sdk_configuration(config: ObjectStorageConfig) -> None:
    if config.sdk_retry_mode not in {"standard", "adaptive"}:
        raise ObjectStorageError("Object storage SDK retry mode is invalid")
    _validate_page_limit(
        "sdk_total_max_attempts", config.sdk_total_max_attempts, maximum=10
    )
    for field_name in ("connect_timeout_seconds", "read_timeout_seconds"):
        value = getattr(config, field_name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 < float(value) <= 300
        ):
            raise ObjectStorageError(
                f"Object storage {field_name} must be within (0, 300]"
            )
    for field_name in (
        "request_checksum_calculation",
        "response_checksum_validation",
    ):
        if getattr(config, field_name) not in {"when_required", "when_supported"}:
            raise ObjectStorageError(f"Object storage {field_name} policy is invalid")


def _validate_page_limit(name: str, value: int, *, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > maximum
    ):
        raise ObjectStorageError(
            f"{name} must be a positive integer no greater than {maximum}"
        )
    return value


def _validate_optional_nonnegative_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ObjectStorageIntegrityError(
            f"{name} must be a non-negative integer when provided"
        )
    return value


def _validate_nonnegative_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Object size must be a non-negative integer")
    return value


def _normalize_utc_datetime(value: datetime) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError("Object last-modified time must be timezone-aware")
    return value.astimezone(timezone.utc)


def _validate_optional_safe_text(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 1024
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ObjectStorageIntegrityError(f"{name} is not safe comparison metadata")
    return value


def _validate_opaque_identity(name: str, value: str) -> str:
    validated = _validate_optional_safe_text(name, value)
    assert validated is not None
    return validated


def _validate_optional_scope_component(label: str, value: str | None) -> str | None:
    if value is None:
        return None
    return _validate_scope_component(label, value)


def _urlsafe_b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _urlsafe_b64decode(value: str) -> bytes:
    if not value or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("invalid base64 token")
    padding = "=" * (-len(value) % 4)
    return base64.b64decode(value + padding, altchars=b"-_", validate=True)


def _s3_error_code_and_status(exc: Exception) -> tuple[str, int | None]:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return "", None
    error = response.get("Error")
    code = error.get("Code") if isinstance(error, dict) else None
    metadata = response.get("ResponseMetadata")
    status = metadata.get("HTTPStatusCode") if isinstance(metadata, dict) else None
    return str(code or "").strip().lower(), status if isinstance(status, int) else None


def _classify_s3_exception(exc: Exception) -> ObjectStorageProofError:
    code, status = _s3_error_code_and_status(exc)
    if code in {"404", "nosuchkey", "nosuchversion", "notfound"} or status == 404:
        return ObjectStorageNotFoundError()
    if (
        code
        in {
            "403",
            "accessdenied",
            "allaccessdisabled",
            "invalidaccesskeyid",
            "signaturedoesnotmatch",
        }
        or status == 403
    ):
        return ObjectStorageForbiddenError()
    retryable_codes = {
        "408",
        "429",
        "internalerror",
        "requesttimeout",
        "requesttimeoutexception",
        "serviceunavailable",
        "slowdown",
        "throttling",
        "throttlingexception",
    }
    class_name = type(exc).__name__.lower()
    if (
        isinstance(exc, (TimeoutError, ConnectionError, OSError))
        or status in {408, 429}
        or (status is not None and status >= 500)
        or code in retryable_codes
        or any(
            fragment in class_name
            for fragment in (
                "connectionclosed",
                "connecttimeout",
                "endpointconnection",
                "readtimeout",
            )
        )
    ):
        return ObjectStorageTransportError()
    return ObjectStorageDeleteError()


def _is_checksum_mode_unsupported_error(exc: Exception) -> bool:
    if type(exc).__name__ == "ParamValidationError" and (
        "checksummode" in str(exc).replace("_", "").lower()
    ):
        return True
    code, status = _s3_error_code_and_status(exc)
    if code in {"notimplemented", "notsupported", "unsupportedargument"} or (
        status == 501
    ):
        return True
    if code not in {"invalidargument", "invalidrequest"}:
        return False
    response = getattr(exc, "response", None)
    error = response.get("Error") if isinstance(response, dict) else None
    if not isinstance(error, dict):
        return False
    evidence = " ".join(
        str(error.get(field_name) or "") for field_name in ("ArgumentName", "Message")
    ).lower()
    return "checksum" in evidence


def _is_checksum_request_unsupported_error(exc: Exception) -> bool:
    """Return whether a backend explicitly rejected the PUT checksum option."""

    compact_message = str(exc).replace("_", "").replace("-", "").lower()
    if type(exc).__name__ == "ParamValidationError" and (
        "checksumsha256" in compact_message or "checksum" in compact_message
    ):
        return True
    code, status = _s3_error_code_and_status(exc)
    if (
        code
        in {
            "notimplemented",
            "notsupported",
            "unsupported",
            "unsupportedargument",
        }
        or status == 501
    ):
        return True
    if code not in {"invalidargument", "invalidrequest"}:
        return False
    response = getattr(exc, "response", None)
    error = response.get("Error") if isinstance(response, dict) else None
    if not isinstance(error, dict):
        return False
    evidence = " ".join(
        str(error.get(field_name) or "") for field_name in ("ArgumentName", "Message")
    ).lower()
    return "checksum" in evidence


def _parse_utc_last_modified(value: Any) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ObjectStorageMalformedResponseError() from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise ObjectStorageMalformedResponseError()
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ObjectStorageMalformedResponseError()
    return parsed.astimezone(timezone.utc)


def _optional_opaque_response_text(value: Any) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 1024
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ObjectStorageMalformedResponseError()
    return value


def _normalize_sha256_checksum(value: str, *, allow_base64: bool = True) -> str | None:
    if not isinstance(value, str) or not value or value != value.strip():
        return None
    candidate = value
    prefix_match = re.match(r"(?i)^sha-?256\s*[:=]\s*(.+)$", candidate)
    if prefix_match is not None:
        candidate = prefix_match.group(1)
    if re.fullmatch(r"[0-9A-Fa-f]{64}", candidate):
        return candidate.lower()
    if not allow_base64:
        return None
    try:
        decoded = base64.b64decode(candidate, validate=True)
    except (ValueError, binascii.Error):
        return None
    if len(decoded) != hashlib.sha256().digest_size:
        return None
    return decoded.hex()


def _local_file_sha256_authority(path: Path) -> tuple[int, str]:
    """Hash one stable regular file without following a replacement symlink."""

    try:
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
    except OSError as exc:
        raise ObjectStorageIntegrityError(
            "Local immutable upload authority could not be read"
        ) from exc
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity or path.is_symlink() or not path.is_file():
        raise ObjectStorageIntegrityError(
            "Local immutable upload authority changed while hashing"
        )
    return after.st_size, digest.hexdigest()


def _listing_checksum_metadata(item: dict[str, Any]) -> tuple[str | None, str | None]:
    checksum_value = item.get("ChecksumSHA256")
    if checksum_value is not None:
        if not isinstance(checksum_value, str):
            raise ObjectStorageMalformedResponseError()
        checksum_type = item.get("ChecksumType")
        if checksum_type not in {None, "FULL_OBJECT", "COMPOSITE"}:
            raise ObjectStorageMalformedResponseError()
        if checksum_type == "COMPOSITE":
            if not checksum_value:
                raise ObjectStorageMalformedResponseError()
            return None, "sha256"
        normalized = _normalize_sha256_checksum(checksum_value)
        if normalized is None:
            raise ObjectStorageMalformedResponseError()
        return f"sha256:{normalized}", "sha256"
    algorithms = item.get("ChecksumAlgorithm")
    if algorithms is None:
        return None, None
    if isinstance(algorithms, str):
        values = [algorithms]
    elif (
        isinstance(algorithms, list)
        and algorithms
        and all(isinstance(value, str) and value for value in algorithms)
    ):
        values = algorithms
    else:
        raise ObjectStorageMalformedResponseError()
    known_algorithms = {
        "crc32": "crc32",
        "crc32c": "crc32c",
        "crc64nvme": "crc64nvme",
        "sha1": "sha1",
        "sha256": "sha256",
    }
    algorithm = next(
        (
            known_algorithms[normalized]
            for value in values
            if (normalized := value.replace("-", "").lower()) in known_algorithms
        ),
        None,
    )
    return None, algorithm


def _object_stat_from_head_response(
    response: Any, *, checksum_requested: bool
) -> ObjectStat:
    if not isinstance(response, dict):
        raise ObjectStorageMalformedResponseError()
    size = response.get("ContentLength")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ObjectStorageMalformedResponseError()
    etag = _optional_opaque_response_text(response.get("ETag"))
    version_id = _optional_opaque_response_text(response.get("VersionId"))
    last_modified_value = response.get("LastModified")
    last_modified = (
        None
        if last_modified_value is None
        else _parse_utc_last_modified(last_modified_value)
    )
    checksum: str | None = None
    checksum_algorithm: str | None = None
    if checksum_requested:
        raw_s3_checksum = response.get("ChecksumSHA256")
        if raw_s3_checksum is not None:
            if (
                not isinstance(raw_s3_checksum, str)
                or not raw_s3_checksum
                or len(raw_s3_checksum) > 1024
                or any(
                    ord(character) < 32 or ord(character) == 127
                    for character in raw_s3_checksum
                )
            ):
                raise ObjectStorageMalformedResponseError()
            checksum_type = response.get("ChecksumType")
            if checksum_type not in {None, "FULL_OBJECT", "COMPOSITE"}:
                raise ObjectStorageMalformedResponseError()
            checksum_algorithm = "sha256"
            if checksum_type != "COMPOSITE":
                normalized = _normalize_sha256_checksum(raw_s3_checksum)
                if normalized is None:
                    raise ObjectStorageMalformedResponseError()
                checksum = f"sha256:{normalized}"
    if checksum is None:
        metadata = response.get("Metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ObjectStorageMalformedResponseError()
        if not all(isinstance(key, str) for key in metadata):
            raise ObjectStorageMalformedResponseError()
        lowered = {str(key).strip().lower(): value for key, value in metadata.items()}
        for key in (
            "sha256",
            "checksum-sha256",
            "checksum_sha256",
            "content-sha256",
            "checksum",
        ):
            raw_metadata_checksum = lowered.get(key)
            if raw_metadata_checksum is None:
                continue
            if not isinstance(raw_metadata_checksum, str):
                raise ObjectStorageMalformedResponseError()
            normalized = _normalize_sha256_checksum(raw_metadata_checksum)
            if normalized is not None:
                checksum = f"sha256:{normalized}"
                checksum_algorithm = "sha256"
                break
    return ObjectStat(
        size=size,
        etag=etag,
        checksum=checksum,
        checksum_algorithm=checksum_algorithm,
        version_id=version_id,
        last_modified=last_modified,
    )


def _canonical_etag(value: str) -> str:
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    return value


def _compare_expected_object_metadata(
    actual: ObjectStat,
    *,
    expected_size_bytes: int | None,
    expected_checksum: str | None,
    expected_etag: str | None,
    expected_version_id: str | None,
) -> None:
    if expected_size_bytes is not None and actual.size != expected_size_bytes:
        raise ObjectStorageIntegrityError(
            "Object size does not match cleanup authority"
        )
    if expected_etag is not None and (
        actual.etag is None
        or _canonical_etag(actual.etag) != _canonical_etag(expected_etag)
    ):
        raise ObjectStorageIntegrityError(
            "Object ETag does not match cleanup authority"
        )
    if expected_version_id is not None and actual.version_id != expected_version_id:
        raise ObjectStorageVersionProofError(
            "Object version does not match cleanup authority"
        )
    if expected_checksum is not None:
        expected_sha256 = _normalize_sha256_checksum(expected_checksum)
        if expected_sha256 is None:
            raise ObjectStorageIntegrityError(
                "Expected checksum is not a comparable SHA-256 value"
            )
        if actual.checksum is None:
            raise ObjectStorageIntegrityError(
                "Required object checksum evidence is unavailable"
            )
        actual_sha256 = _normalize_sha256_checksum(actual.checksum)
        if actual_sha256 is None or actual_sha256 != expected_sha256:
            raise ObjectStorageIntegrityError(
                "Object checksum does not match cleanup authority"
            )


def _validated_delete_keys_from_list_response(
    response: Any, *, prefix: str, maximum: int
) -> list[dict[str, str]]:
    if not isinstance(response, dict):
        raise ObjectStorageMalformedResponseError()
    contents = response.get("Contents", [])
    if not isinstance(contents, list) or len(contents) > maximum:
        raise ObjectStorageMalformedResponseError()
    truncated = response.get("IsTruncated")
    if not isinstance(truncated, bool):
        raise ObjectStorageMalformedResponseError()
    if truncated and not response.get("NextContinuationToken"):
        raise ObjectStorageMalformedResponseError()
    if truncated and not contents:
        raise ObjectStorageMalformedResponseError()
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    for item in contents:
        if not isinstance(item, dict):
            raise ObjectStorageMalformedResponseError()
        key = item.get("Key")
        if not isinstance(key, str) or not key or key in seen:
            raise ObjectStorageMalformedResponseError()
        try:
            _validate_raw_object_key(
                key,
                label="listed object key",
                allow_empty=False,
                allow_trailing_slash=True,
            )
        except ObjectStorageError as exc:
            raise ObjectStorageMalformedResponseError() from exc
        if not key.startswith(prefix):
            raise ObjectStorageMalformedResponseError()
        seen.add(key)
        result.append({"Key": key})
    return result


def _validated_version_delete_entries(
    response: Any, *, prefix: str, maximum: int
) -> list[dict[str, str]]:
    if not isinstance(response, dict):
        raise ObjectStorageMalformedResponseError()
    versions = response.get("Versions", [])
    delete_markers = response.get("DeleteMarkers", [])
    if not isinstance(versions, list) or not isinstance(delete_markers, list):
        raise ObjectStorageMalformedResponseError()
    if len(versions) + len(delete_markers) > maximum:
        raise ObjectStorageMalformedResponseError()
    truncated = response.get("IsTruncated")
    if not isinstance(truncated, bool):
        raise ObjectStorageMalformedResponseError()
    if truncated:
        if not versions and not delete_markers:
            raise ObjectStorageMalformedResponseError()
        marker = response.get("NextKeyMarker")
        if not isinstance(marker, str) or not marker:
            raise ObjectStorageMalformedResponseError()
        try:
            _validate_raw_object_key(
                marker,
                label="version key marker",
                allow_empty=False,
                allow_trailing_slash=True,
            )
        except ObjectStorageError as exc:
            raise ObjectStorageMalformedResponseError() from exc
        if not marker.startswith(prefix):
            raise ObjectStorageMalformedResponseError()
        next_version_marker = response.get("NextVersionIdMarker")
        if next_version_marker is not None and (
            not isinstance(next_version_marker, str) or not next_version_marker
        ):
            raise ObjectStorageMalformedResponseError()
        if next_version_marker is not None:
            _optional_opaque_response_text(next_version_marker)
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, str]] = []
    for item in [*versions, *delete_markers]:
        if not isinstance(item, dict):
            raise ObjectStorageMalformedResponseError()
        key = item.get("Key")
        version_id = item.get("VersionId")
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(version_id, str)
            or not version_id
        ):
            raise ObjectStorageMalformedResponseError()
        _optional_opaque_response_text(version_id)
        try:
            _validate_raw_object_key(
                key,
                label="listed version key",
                allow_empty=False,
                allow_trailing_slash=True,
            )
        except ObjectStorageError as exc:
            raise ObjectStorageMalformedResponseError() from exc
        if not key.startswith(prefix) or (key, version_id) in seen:
            raise ObjectStorageMalformedResponseError()
        _parse_utc_last_modified(item.get("LastModified"))
        seen.add((key, version_id))
        result.append({"Key": key, "VersionId": version_id})
    return result


def _delete_objects_response_error(
    response: Any,
) -> ObjectStorageProofError | None:
    if not isinstance(response, dict):
        return ObjectStorageMalformedResponseError()
    errors = response.get("Errors", [])
    if not isinstance(errors, list):
        return ObjectStorageMalformedResponseError()
    result: ObjectStorageProofError | None = None
    for item in errors:
        if not isinstance(item, dict):
            return ObjectStorageMalformedResponseError()
        code = item.get("Code")
        key = item.get("Key")
        if not isinstance(code, str) or not code or not isinstance(key, str) or not key:
            return ObjectStorageMalformedResponseError()
        normalized = code.strip().lower()
        if normalized in {"accessdenied", "allaccessdisabled", "invalidaccesskeyid"}:
            return ObjectStorageForbiddenError()
        if normalized in {
            "internalerror",
            "requesttimeout",
            "serviceunavailable",
            "slowdown",
            "throttling",
        }:
            result = ObjectStorageTransportError()
        elif result is None:
            result = ObjectStorageDeleteError()
    return result


def _raise_for_delete_objects_errors(response: Any) -> None:
    error = _delete_objects_response_error(response)
    if error is not None:
        raise error


def _is_bucket_not_found_error(exc: Exception) -> bool:
    """Recognize only explicit S3 bucket-not-found responses."""

    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error")
    code = error.get("Code") if isinstance(error, dict) else None
    normalized_code = str(code or "").strip().lower()
    metadata = response.get("ResponseMetadata")
    status = metadata.get("HTTPStatusCode") if isinstance(metadata, dict) else None
    return normalized_code in {"404", "nosuchbucket", "notfound"} or status == 404


def _is_precondition_failed_error(exc: Exception) -> bool:
    """Recognize the conditional-create loser without masking other errors."""

    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error")
    code = error.get("Code") if isinstance(error, dict) else None
    normalized_code = str(code or "").strip().lower()
    metadata = response.get("ResponseMetadata")
    status = metadata.get("HTTPStatusCode") if isinstance(metadata, dict) else None
    return (
        normalized_code
        in {
            "412",
            "conditionalrequestconflict",
            "preconditionfailed",
        }
        or status == 412
    )


_ENCODED_SEPARATOR_RE = re.compile(r"%(?:2f|5c)", re.IGNORECASE)
_PERCENT_ESCAPE_RE = re.compile(r"%[0-9A-Fa-f]{2}")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:.*$")
_BUCKET_NAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9.-]{0,61}[A-Za-z0-9])?$")


def _validate_bucket_name(bucket: str) -> str:
    if not isinstance(bucket, str) or not bucket:
        raise ObjectStorageError("LIGHTRAG_OBJECT_STORAGE_BUCKET is required")
    if bucket != bucket.strip():
        raise ObjectStorageError(
            "Object storage bucket cannot contain outer whitespace"
        )
    if any(ord(char) < 32 or ord(char) == 127 for char in bucket):
        raise ObjectStorageError("Object storage bucket contains control characters")
    if not _BUCKET_NAME_RE.fullmatch(bucket) or bucket in {".", ".."} or ".." in bucket:
        raise ObjectStorageError("Object storage bucket is invalid")
    return bucket


def _validate_raw_object_key(
    value: str,
    *,
    label: str,
    allow_empty: bool,
    allow_trailing_slash: bool,
) -> str:
    if not isinstance(value, str):
        raise ObjectStorageError(f"{label} must be a string")
    if not value:
        if allow_empty:
            return value
        raise ObjectStorageError(f"{label} cannot be empty")
    if value != value.strip():
        raise ObjectStorageError(f"{label} cannot contain outer whitespace")
    if len(value.encode("utf-8")) > 1024:
        raise ObjectStorageError(f"{label} exceeds the S3 key length limit")

    if "\\" in value:
        raise ObjectStorageError(f"{label} cannot contain backslashes")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ObjectStorageError(f"{label} contains control characters")
    if value.startswith("/"):
        raise ObjectStorageError(f"{label} cannot be absolute")

    has_trailing_slash = value.endswith("/")
    if has_trailing_slash and not allow_trailing_slash:
        raise ObjectStorageError(f"{label} cannot end with a slash")
    structural = value[:-1] if has_trailing_slash else value
    if not structural:
        if allow_empty:
            return value
        raise ObjectStorageError(f"{label} cannot be empty")
    parts = structural.split("/")
    if any(part == "" for part in parts):
        raise ObjectStorageError(f"{label} contains an empty path segment")
    if any(part in {".", ".."} for part in parts):
        raise ObjectStorageError(f"{label} contains path traversal")
    if any(_WINDOWS_DRIVE_RE.fullmatch(part) for part in parts):
        raise ObjectStorageError(f"{label} contains an absolute drive path")
    return value


def _decode_object_uri_key_once(encoded_key: str) -> str:
    """Decode one URI layer, then validate the resulting raw S3 key."""

    if _ENCODED_SEPARATOR_RE.search(encoded_key):
        raise ObjectStorageError(
            "Encoded object URI key contains an encoded path separator"
        )
    index = 0
    while index < len(encoded_key):
        if encoded_key[index] != "%":
            index += 1
            continue
        if not _PERCENT_ESCAPE_RE.match(encoded_key, index):
            raise ObjectStorageError("Object URI key has invalid encoding")
        index += 3
    try:
        return unquote(encoded_key, errors="strict")
    except (UnicodeDecodeError, ValueError) as exc:
        raise ObjectStorageError("Object URI key has invalid encoding") from exc


def _validate_scope_component(label: str, value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ObjectStorageError(f"{label} must be a non-empty path component")
    _validate_raw_object_key(
        value,
        label=label,
        allow_empty=False,
        allow_trailing_slash=False,
    )
    if "/" in value:
        raise ObjectStorageError(f"{label} must be one path component")
    return value


def _validate_optional_limit(name: str, value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ObjectStorageError(f"{name} must be a non-negative integer")


def _prepare_local_download_root(local_dir: Path) -> Path:
    if local_dir.is_symlink():
        raise ObjectStorageError("Prefix download target cannot be a symlink")
    try:
        local_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ObjectStorageError(
            f"Unable to create prefix download target: {local_dir}"
        ) from exc
    if local_dir.is_symlink() or not local_dir.is_dir():
        raise ObjectStorageError("Prefix download target is not a safe directory")
    return local_dir.resolve(strict=True)


def _assert_safe_local_target(local_root: Path, target: Path) -> None:
    try:
        relative_parts = target.relative_to(local_root).parts
    except ValueError as exc:
        raise ObjectStorageError("Object key escapes prefix download target") from exc

    current = local_root
    for part in relative_parts:
        current = current / part
        if current.is_symlink():
            raise ObjectStorageError(
                "Object key reaches a symlink inside prefix download target"
            )
    try:
        resolved = target.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ObjectStorageError("Unable to resolve prefix download target") from exc
    if not resolved.is_relative_to(local_root):
        raise ObjectStorageError("Object key escapes prefix download target")
