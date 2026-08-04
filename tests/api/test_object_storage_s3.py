"""Contract tests for the real :class:`S3ObjectStorage` implementation.

These exercise the *actual* production code paths in
``lightrag/api/object_storage.py`` (key normalization, URI build/parse,
multipart-free upload/download, list_objects_v2 pagination + continuation
tokens, prefix/workspace deletion, bucket auto-create) by stubbing only the
boto3 client boundary. ``aioboto3`` is imported lazily inside
``S3ObjectStorage._new_session`` so we can inject a fake session and run fully
offline — no MinIO/S3 and no aioboto3 install required.

This closes the gap where object storage was previously covered only by an
in-test ``FakeObjectStorage`` (which re-implemented the interface rather than
testing the shipped code).
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

import pytest

from lightrag.api.object_storage import (
    ArtifactCleanupTarget,
    DisabledObjectStorage,
    ObjectStorageForbiddenError,
    ObjectStorageConfig,
    ObjectStorageError,
    ObjectStorageIntegrityError,
    ObjectStorageMalformedResponseError,
    ObjectStoragePageBudgetError,
    ObjectStorageStillPresentError,
    ObjectStorageTransportError,
    ObjectStorageVersionProofError,
    S3ObjectStorage,
    create_object_storage_from_env,
)

pytestmark = pytest.mark.offline


class _FakeS3Error(RuntimeError):
    def __init__(self, code: str, status: int, message: str | None = None):
        super().__init__(message or code)
        self.response = {
            "Error": {"Code": code, "Message": message or code},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }


class _FakeS3Client:
    """Minimal in-memory S3 client mirroring the boto3 async surface used.

    ``store`` is shared with the owning :class:`_FakeSession` so multiple
    ``async with session.client(...)`` blocks observe the same objects.
    """

    def __init__(self, state: "_FakeS3State", *, page_size: int = 2):
        self._state = state
        self._page_size = page_size
        self.meta = _FakeClientMeta(state)

    async def __aenter__(self) -> "_FakeS3Client":
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def head_bucket(self, *, Bucket: str) -> None:
        self._state.calls.append(("head_bucket", Bucket))
        if self._state.head_bucket_errors:
            error = self._state.head_bucket_errors.pop(0)
            if error is not None:
                raise error
        if Bucket not in self._state.buckets:
            raise _FakeS3Error("NoSuchBucket", 404, "bucket missing")

    async def create_bucket(self, *, Bucket: str) -> None:
        self._state.calls.append(("create_bucket", Bucket))
        self._state.buckets.add(Bucket)

    async def upload_file(self, Filename, Bucket, Key, ExtraArgs=None) -> None:
        self._state.calls.append(("upload_file", Bucket, Key))
        self._state.objects[(Bucket, Key)] = Path(Filename).read_bytes()
        self._state.extra_args[(Bucket, Key)] = ExtraArgs
        if isinstance(ExtraArgs, dict) and isinstance(ExtraArgs.get("Metadata"), dict):
            metadata = dict(ExtraArgs["Metadata"])
            self._state.head_metadata[(Bucket, Key)] = metadata
            if isinstance(metadata.get("sha256"), str):
                self._state.sha256_checksums[(Bucket, Key)] = metadata["sha256"]

    async def put_object(self, *, Body, Bucket, Key, IfNoneMatch=None, **kwargs):
        self._state.calls.append(("put_object", Bucket, Key, IfNoneMatch))
        if IfNoneMatch == "*" and (Bucket, Key) in self._state.objects:
            raise _FakeS3Error("PreconditionFailed", 412)
        # Optional transport failure raised BEFORE any bytes are durably
        # stored, modelling a request that never reached the backend.
        pre_error = self._state.put_object_pre_store_errors.pop((Bucket, Key), None)
        if pre_error is not None:
            raise pre_error
        body = Body.read() if hasattr(Body, "read") else Body
        self._state.objects[(Bucket, Key)] = bytes(body)
        metadata = kwargs.get("Metadata")
        if isinstance(metadata, dict):
            self._state.head_metadata[(Bucket, Key)] = dict(metadata)
            if isinstance(metadata.get("sha256"), str):
                self._state.sha256_checksums[(Bucket, Key)] = metadata["sha256"]
        # Simulate a lost PUT acknowledgement: the object is durably stored but
        # the client never receives the success response. The conditional-create
        # proof path must recover via HEAD readback.
        if (Bucket, Key) in self._state.put_object_ack_loss_keys:
            self._state.put_object_ack_loss_keys.discard((Bucket, Key))
            raise TimeoutError("put object acknowledgement lost")
        return {"ETag": '"put-etag"'}

    async def download_file(self, Bucket, Key, Filename) -> None:
        self._state.calls.append(("download_file", Bucket, Key))
        error = self._state.download_errors.get((Bucket, Key))
        if error is not None:
            raise error
        if (Bucket, Key) not in self._state.objects:
            raise RuntimeError(f"missing object {Bucket}/{Key}")
        Path(Filename).write_bytes(self._state.objects[(Bucket, Key)])

    async def head_object(
        self,
        *,
        Bucket,
        Key,
        ChecksumMode=None,
        VersionId=None,
    ):
        self._state.calls.append(("head_object", Bucket, Key))
        self._state.head_requests.append(
            {
                "Bucket": Bucket,
                "Key": Key,
                "ChecksumMode": ChecksumMode,
                "VersionId": VersionId,
            }
        )
        error_key = (Bucket, Key, VersionId)
        errors = self._state.head_object_errors.get(error_key)
        if errors:
            error = errors.pop(0)
            if error is not None:
                raise error
        overrides = self._state.head_response_overrides.get(error_key)
        if overrides:
            return overrides.pop(0)
        if ChecksumMode and self._state.reject_checksum_mode:
            raise _FakeS3Error("NotImplemented", 501, "checksum mode unsupported")
        version = None
        if VersionId is not None:
            version = next(
                (
                    item
                    for item in self._state.versions.get((Bucket, Key), [])
                    if item["VersionId"] == VersionId
                ),
                None,
            )
            if version is None:
                raise _FakeS3Error("NoSuchVersion", 404)
            body = version["Body"]
        else:
            if (Bucket, Key) not in self._state.objects:
                raise _FakeS3Error("NoSuchKey", 404)
            body = self._state.objects[(Bucket, Key)]
        size = self._state.head_object_sizes.get((Bucket, Key), len(body))
        metadata = self._state.head_metadata.get((Bucket, Key), {})
        response: dict[str, Any] = {
            "ContentLength": size,
            "ETag": self._state.etags.get((Bucket, Key), f'"etag-{size}"'),
            "LastModified": self._state.last_modified.get(
                (Bucket, Key), datetime(2026, 8, 3, 12, tzinfo=timezone.utc)
            ),
            "Metadata": metadata,
        }
        checksum = self._state.sha256_checksums.get((Bucket, Key))
        if checksum is not None and ChecksumMode:
            response["ChecksumSHA256"] = base64.b64encode(
                bytes.fromhex(checksum)
            ).decode("ascii")
        if VersionId is not None:
            response["VersionId"] = VersionId
        elif (
            version_id := self._state.current_version_ids.get((Bucket, Key))
        ) is not None:
            response["VersionId"] = version_id
        return response

    async def list_objects_v2(
        self,
        *,
        Bucket,
        Prefix,
        MaxKeys=None,
        ContinuationToken=None,
    ):
        self._state.calls.append(("list_objects_v2", Bucket, Prefix, ContinuationToken))
        self._state.list_requests.append(
            {
                "Bucket": Bucket,
                "Prefix": Prefix,
                "MaxKeys": MaxKeys,
                "ContinuationToken": ContinuationToken,
            }
        )
        if self._state.list_errors:
            error = self._state.list_errors.pop(0)
            if error is not None:
                raise error
        if self._state.list_response_overrides:
            return self._state.list_response_overrides.pop(0)
        keys = sorted(
            key
            for (bucket, key) in self._state.objects
            if bucket == Bucket and key.startswith(Prefix)
        )
        # Model real S3: the continuation token is a stable cursor encoding the
        # last key returned, NOT an integer offset. This stays correct even when
        # the caller deletes objects between pages (delete_prefix does exactly
        # that) — an offset-based cursor would skip items as the list shrinks.
        if ContinuationToken:
            keys = [key for key in keys if key > ContinuationToken]
        requested = 1000 if MaxKeys is None else MaxKeys
        effective_page_size = min(self._page_size, requested)
        page = keys[:effective_page_size]
        truncated = len(keys) > effective_page_size
        result: dict[str, Any] = {
            "Contents": [
                {
                    "Key": key,
                    "Size": self._state.listed_sizes.get(
                        (Bucket, key), len(self._state.objects[(Bucket, key)])
                    ),
                    "LastModified": self._state.last_modified.get(
                        (Bucket, key),
                        datetime(2026, 8, 3, 12, tzinfo=timezone.utc),
                    ),
                    "ETag": self._state.etags.get(
                        (Bucket, key),
                        f'"etag-{len(self._state.objects[(Bucket, key)])}"',
                    ),
                    **(
                        {"ChecksumAlgorithm": ["SHA256"]}
                        if (Bucket, key) in self._state.sha256_checksums
                        else {}
                    ),
                    **(
                        {"VersionId": self._state.current_version_ids[(Bucket, key)]}
                        if (Bucket, key) in self._state.current_version_ids
                        else {}
                    ),
                }
                for key in page
            ]
        }
        if truncated:
            result["IsTruncated"] = True
            result["NextContinuationToken"] = page[-1]
        else:
            result["IsTruncated"] = False
        return result

    async def get_bucket_versioning(self, *, Bucket):
        self._state.calls.append(("get_bucket_versioning", Bucket))
        if self._state.versioning_errors:
            error = self._state.versioning_errors.pop(0)
            if error is not None:
                raise error
        status = self._state.bucket_versioning.get(Bucket)
        return {} if status is None else {"Status": status}

    async def list_object_versions(self, *, Bucket, Prefix, MaxKeys):
        self._state.calls.append(("list_object_versions", Bucket, Prefix, MaxKeys))
        if self._state.version_list_overrides:
            return self._state.version_list_overrides.pop(0)
        versions: list[dict[str, Any]] = []
        markers: list[dict[str, Any]] = []
        for (bucket, key), items in sorted(self._state.versions.items()):
            if bucket != Bucket or not key.startswith(Prefix):
                continue
            versions.extend(
                {
                    "Key": key,
                    "VersionId": item["VersionId"],
                    "LastModified": item.get(
                        "LastModified",
                        datetime(2026, 8, 3, 12, tzinfo=timezone.utc),
                    ),
                }
                for item in items
            )
        for bucket, key, version_id in sorted(self._state.delete_markers):
            if bucket == Bucket and key.startswith(Prefix):
                markers.append(
                    {
                        "Key": key,
                        "VersionId": version_id,
                        "LastModified": datetime(2026, 8, 3, 12, tzinfo=timezone.utc),
                    }
                )
        combined = [(False, item) for item in versions] + [
            (True, item) for item in markers
        ]
        selected = combined[:MaxKeys]
        result: dict[str, Any] = {
            "Versions": [item for marker, item in selected if not marker],
            "DeleteMarkers": [item for marker, item in selected if marker],
            "IsTruncated": len(combined) > MaxKeys,
        }
        if result["IsTruncated"]:
            result["NextKeyMarker"] = selected[-1][1]["Key"]
            result["NextVersionIdMarker"] = selected[-1][1]["VersionId"]
        return result

    async def delete_object(self, *, Bucket, Key, VersionId=None) -> None:
        self._state.calls.append(("delete_object", Bucket, Key))
        self._state.delete_object_requests.append(
            {"Bucket": Bucket, "Key": Key, "VersionId": VersionId}
        )
        behavior = self._state.delete_object_behaviors.pop((Bucket, Key), None)
        if behavior == "error_before":
            raise TimeoutError("delete timeout")
        if VersionId is not None:
            versions = self._state.versions.get((Bucket, Key), [])
            self._state.versions[(Bucket, Key)] = [
                item for item in versions if item["VersionId"] != VersionId
            ]
        else:
            self._state.objects.pop((Bucket, Key), None)
        if behavior == "ack_loss":
            raise TimeoutError("delete acknowledgement lost")
        if behavior == "ack_loss_present":
            self._state.objects[(Bucket, Key)] = b"still-present"
            raise TimeoutError("delete acknowledgement lost")

    async def delete_objects(self, *, Bucket, Delete):
        self._state.calls.append(
            (
                "delete_objects",
                Bucket,
                tuple(item["Key"] for item in Delete["Objects"]),
            )
        )
        self._state.delete_objects_requests.append({"Bucket": Bucket, "Delete": Delete})
        if self._state.delete_objects_errors_before:
            error = self._state.delete_objects_errors_before.pop(0)
            if error is not None:
                raise error
        errors: list[dict[str, str]] = []
        deleted: list[dict[str, str]] = []
        for obj in Delete["Objects"]:
            key = obj["Key"]
            version_id = obj.get("VersionId")
            code = self._state.per_key_delete_errors.get((Bucket, key, version_id))
            if code is not None:
                errors.append({"Key": key, "VersionId": version_id or "", "Code": code})
                continue
            if version_id is None:
                self._state.objects.pop((Bucket, key), None)
            else:
                self._state.versions[(Bucket, key)] = [
                    item
                    for item in self._state.versions.get((Bucket, key), [])
                    if item["VersionId"] != version_id
                ]
                self._state.delete_markers.discard((Bucket, key, version_id))
            deleted.append(
                {"Key": key, **({"VersionId": version_id} if version_id else {})}
            )
        if self._state.delete_objects_ack_loss:
            self._state.delete_objects_ack_loss = False
            raise TimeoutError("delete acknowledgement lost")
        return {"Deleted": deleted, "Errors": errors}

    def generate_presigned_url(self, ClientMethod, Params, ExpiresIn):
        self._state.calls.append(
            ("generate_presigned_url", ClientMethod, Params, ExpiresIn)
        )
        return (
            f"https://objects.example/{Params['Bucket']}/{Params['Key']}"
            f"?method={ClientMethod}&expires={ExpiresIn}"
        )


class _FakeS3State:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.extra_args: dict[tuple[str, str], dict | None] = {}
        self.buckets: set[str] = set()
        self.calls: list[tuple] = []
        self.head_requests: list[dict[str, Any]] = []
        self.list_requests: list[dict[str, Any]] = []
        self.delete_object_requests: list[dict[str, Any]] = []
        self.delete_objects_requests: list[dict[str, Any]] = []
        self.registered_events: list[str] = []
        self.head_bucket_errors: list[Exception | None] = []
        self.head_object_sizes: dict[tuple[str, str], int] = {}
        self.listed_sizes: dict[tuple[str, str], int] = {}
        self.download_errors: dict[tuple[str, str], Exception] = {}
        self.last_modified: dict[tuple[str, str], datetime] = {}
        self.etags: dict[tuple[str, str], str] = {}
        self.sha256_checksums: dict[tuple[str, str], str] = {}
        self.head_metadata: dict[tuple[str, str], dict[str, str]] = {}
        self.current_version_ids: dict[tuple[str, str], str] = {}
        self.head_object_errors: dict[
            tuple[str, str, str | None], list[Exception | None]
        ] = {}
        self.head_response_overrides: dict[
            tuple[str, str, str | None], list[dict[str, Any]]
        ] = {}
        self.reject_checksum_mode = False
        self.list_errors: list[Exception | None] = []
        self.list_response_overrides: list[dict[str, Any]] = []
        self.bucket_versioning: dict[str, str] = {}
        self.versioning_errors: list[Exception | None] = []
        self.versions: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.delete_markers: set[tuple[str, str, str]] = set()
        self.version_list_overrides: list[dict[str, Any]] = []
        self.delete_object_behaviors: dict[tuple[str, str], str] = {}
        self.delete_objects_errors_before: list[Exception | None] = []
        self.delete_objects_ack_loss = False
        self.per_key_delete_errors: dict[tuple[str, str, str | None], str] = {}
        self.put_object_ack_loss_keys: set[tuple[str, str]] = set()
        self.put_object_pre_store_errors: dict[tuple[str, str], Exception] = {}


class _FakeEvents:
    def __init__(self, state: _FakeS3State) -> None:
        self._state = state

    def register(self, event_name: str, handler) -> None:
        self._state.registered_events.append(event_name)


class _FakeClientMeta:
    def __init__(self, state: _FakeS3State) -> None:
        self.events = _FakeEvents(state)


class _FakeSession:
    def __init__(self, state: _FakeS3State, *, page_size: int = 2):
        self._state = state
        self._page_size = page_size
        self.client_kwargs: list[dict] = []

    def client(self, service, **kwargs):
        self.client_kwargs.append({"service": service, **kwargs})
        return _FakeS3Client(self._state, page_size=self._page_size)


def _make_storage(
    *,
    bucket: str = "lightrag-kb",
    prefix: str = "kb",
    create_bucket: bool = True,
    page_size: int = 2,
    state: _FakeS3State | None = None,
) -> tuple[S3ObjectStorage, _FakeS3State, _FakeSession]:
    config = ObjectStorageConfig(
        backend="minio",
        bucket=bucket,
        endpoint_url="http://fake:9000",
        access_key_id="admin",
        secret_access_key="admin123",
        region_name="us-east-1",
        prefix=prefix,
        use_ssl=False,
        create_bucket=create_bucket,
    )
    if state is None:
        state = _FakeS3State()
    session = _FakeSession(state, page_size=page_size)
    storage = S3ObjectStorage(config)
    # Inject the fake session at the lazy-import boundary so initialize() and
    # _client() both use it instead of importing aioboto3.
    storage._new_session = lambda: session  # type: ignore[method-assign]
    return storage, state, session


async def test_initialize_creates_bucket_when_missing():
    storage, state, session = _make_storage()
    await storage.initialize()
    # Missing is the only error that permits creation, and reachability is
    # proved again after create_bucket returns.
    assert state.calls.count(("head_bucket", "lightrag-kb")) == 2
    assert ("create_bucket", "lightrag-kb") in state.calls
    assert "lightrag-kb" in state.buckets
    # The s3 client is configured with the endpoint/credentials from config.
    assert session.client_kwargs[0]["endpoint_url"] == "http://fake:9000"
    assert session.client_kwargs[0]["aws_access_key_id"] == "admin"
    sdk_config = session.client_kwargs[0]["config"]
    assert sdk_config.retries == {"mode": "standard", "total_max_attempts": 4}
    assert sdk_config.connect_timeout == 5.0
    assert sdk_config.read_timeout == 30.0
    assert sdk_config.request_checksum_calculation == "when_required"
    assert sdk_config.response_checksum_validation == "when_required"
    assert "before-send.s3.PutObject" in state.registered_events
    assert "before-send.s3.UploadPart" in state.registered_events


async def test_initialize_skips_create_when_bucket_present():
    storage, state, session = _make_storage()
    state.buckets.add("lightrag-kb")
    await storage.initialize()
    assert ("head_bucket", "lightrag-kb") in state.calls
    assert ("create_bucket", "lightrag-kb") not in state.calls


async def test_initialize_heads_bucket_even_when_creation_is_disabled():
    storage, state, _ = _make_storage(create_bucket=False)
    state.buckets.add("lightrag-kb")

    await storage.initialize()

    assert state.calls == [("head_bucket", "lightrag-kb")]


async def test_initialize_missing_bucket_fails_when_creation_is_disabled():
    storage, state, _ = _make_storage(create_bucket=False)

    with pytest.raises(_FakeS3Error, match="bucket missing"):
        await storage.initialize()

    assert state.calls == [("head_bucket", "lightrag-kb")]


@pytest.mark.parametrize(
    "error",
    [
        _FakeS3Error("AccessDenied", 403),
        _FakeS3Error("InvalidAccessKeyId", 403),
        RuntimeError("network unavailable"),
    ],
)
async def test_initialize_does_not_create_bucket_for_non_missing_errors(error):
    storage, state, _ = _make_storage(create_bucket=True)
    state.head_bucket_errors.append(error)

    with pytest.raises(type(error), match=str(error)):
        await storage.initialize()

    assert state.calls == [("head_bucket", "lightrag-kb")]


async def test_initialize_rechecks_bucket_and_propagates_verification_failure():
    storage, state, _ = _make_storage(create_bucket=True)
    verification_error = _FakeS3Error("AccessDenied", 403, "verification denied")
    state.head_bucket_errors.extend([_FakeS3Error("NotFound", 404), verification_error])

    with pytest.raises(_FakeS3Error, match="verification denied"):
        await storage.initialize()

    assert state.calls == [
        ("head_bucket", "lightrag-kb"),
        ("create_bucket", "lightrag-kb"),
        ("head_bucket", "lightrag-kb"),
    ]


@pytest.mark.parametrize(
    ("configured_prefix", "normalized_prefix", "expected_key"),
    [
        (
            "/tenant/kb/",
            "tenant/kb",
            "tenant/kb/workspaces/ws/documents/doc/source/report.pdf",
        ),
        ("/", "", "workspaces/ws/documents/doc/source/report.pdf"),
    ],
)
async def test_from_env_preserves_legacy_prefix_boundary_slash_normalization(
    tmp_path: Path,
    monkeypatch,
    configured_prefix: str,
    normalized_prefix: str,
    expected_key: str,
):
    monkeypatch.setenv("LIGHTRAG_OBJECT_STORAGE", "minio")
    monkeypatch.setenv("LIGHTRAG_OBJECT_STORAGE_BUCKET", "lightrag-kb")
    monkeypatch.setenv("LIGHTRAG_OBJECT_STORAGE_PREFIX", configured_prefix)
    monkeypatch.setenv("LIGHTRAG_OBJECT_STORAGE_CREATE_BUCKET", "true")

    config = ObjectStorageConfig.from_env()
    assert config.prefix == normalized_prefix

    state = _FakeS3State()
    session = _FakeSession(state)
    storage = S3ObjectStorage(config)
    storage._new_session = lambda: session  # type: ignore[method-assign]
    await storage.initialize()

    local = tmp_path / "report.pdf"
    local.write_bytes(b"pdf")
    uri = await storage.upload_file(
        local,
        key="workspaces/ws/documents/doc/source/report.pdf",
    )

    assert uri == f"s3://lightrag-kb/{expected_key}"
    assert state.objects[("lightrag-kb", expected_key)] == b"pdf"


async def test_upload_file_applies_prefix_and_returns_uri(tmp_path: Path):
    storage, state, _ = _make_storage(prefix="kb")
    await storage.initialize()
    local = tmp_path / "a.pdf"
    local.write_bytes(b"pdf-bytes")

    uri = await storage.upload_file(
        local, key="workspaces/ws1/documents/doc1/source/a.pdf"
    )
    expected_key = "kb/workspaces/ws1/documents/doc1/source/a.pdf"
    assert uri == f"s3://lightrag-kb/{expected_key}"
    assert state.objects[("lightrag-kb", expected_key)] == b"pdf-bytes"
    # mime type guessed from extension.
    extra_args = state.extra_args[("lightrag-kb", expected_key)]
    assert extra_args is not None
    assert extra_args["ContentType"] == "application/pdf"


async def test_upload_and_download_roundtrip(tmp_path: Path):
    storage, _, _ = _make_storage()
    await storage.initialize()
    local = tmp_path / "src.bin"
    local.write_bytes(b"binary-payload")
    uri = await storage.upload_file(
        local, key="workspaces/ws1/documents/d/source/src.bin"
    )

    out = tmp_path / "restored" / "src.bin"
    await storage.download_file(uri, out)
    assert out.read_bytes() == b"binary-payload"


async def test_stat_object_uses_head_object(tmp_path: Path):
    storage, state, _ = _make_storage()
    await storage.initialize()
    local = tmp_path / "src.bin"
    local.write_bytes(b"binary-payload")
    uri = await storage.upload_file(
        local, key="workspaces/ws1/documents/d/source/src.bin"
    )

    result = await storage.stat_object(uri)

    assert result.size == 14
    assert result.etag == '"etag-14"'
    assert result.last_modified == datetime(2026, 8, 3, 12, tzinfo=timezone.utc)
    assert (
        "head_object",
        "lightrag-kb",
        "kb/workspaces/ws1/documents/d/source/src.bin",
    ) in state.calls
    assert state.head_requests[-1]["ChecksumMode"] == "ENABLED"


@pytest.mark.parametrize(
    "raw_name",
    ["report%2Ffinal.pdf", "notes%2e%2e.txt"],
)
async def test_literal_percent_encoded_raw_key_roundtrips_once(
    tmp_path: Path, raw_name: str
):
    storage, state, _ = _make_storage()
    await storage.initialize()
    local = tmp_path / "source.bin"
    local.write_bytes(b"historical-bytes")
    raw_key = f"workspaces/ws/documents/doc/source/{raw_name}"

    uri = await storage.upload_file(local, key=raw_key)

    assert "%25" in uri
    expected_key = f"kb/{raw_key}"
    assert state.objects[("lightrag-kb", expected_key)] == b"historical-bytes"
    assert (await storage.stat_object(uri)).size == len(b"historical-bytes")
    restored = tmp_path / "restored" / raw_name
    await storage.download_file(uri, restored)
    assert restored.read_bytes() == b"historical-bytes"
    assert ("download_file", "lightrag-kb", expected_key) in state.calls


async def test_prefix_listing_treats_percent_sequences_as_literal_raw_key_text(
    tmp_path: Path,
):
    storage, state, _ = _make_storage()
    await storage.initialize()
    prefix = "kb/workspaces/ws/documents/doc/artifacts/raw/artifact-1/bundle"
    literal_name = "report%2Ffinal.pdf"
    state.objects[("lightrag-kb", f"{prefix}/{literal_name}")] = b"payload"

    restored = tmp_path / "restored"
    count = await storage.download_prefix(f"s3://lightrag-kb/{prefix}/", restored)

    assert count == 1
    assert (restored / literal_name).read_bytes() == b"payload"


async def test_upload_directory_uploads_each_file_under_prefix(tmp_path: Path):
    storage, state, _ = _make_storage()
    await storage.initialize()
    raw = tmp_path / "raw"
    (raw / "images").mkdir(parents=True)
    (raw / "full.md").write_text("# md", encoding="utf-8")
    (raw / "images" / "p1.png").write_bytes(b"img")

    prefix_uri = await storage.upload_directory(
        raw, prefix="workspaces/ws1/documents/d/artifacts/raw"
    )
    assert prefix_uri == "s3://lightrag-kb/kb/workspaces/ws1/documents/d/artifacts/raw/"
    uploaded_keys = {key for (_, key) in state.objects}
    assert "kb/workspaces/ws1/documents/d/artifacts/raw/full.md" in uploaded_keys
    assert "kb/workspaces/ws1/documents/d/artifacts/raw/images/p1.png" in uploaded_keys


async def test_download_prefix_paginates_and_restores_tree(tmp_path: Path):
    storage, _, _ = _make_storage(page_size=2)
    await storage.initialize()
    src = tmp_path / "bundle"
    (src / "sub").mkdir(parents=True)
    for name in ("a.txt", "b.txt", "c.txt", "d.txt"):
        (src / name).write_text(name, encoding="utf-8")
    (src / "sub" / "e.txt").write_text("e", encoding="utf-8")
    prefix_uri = await storage.upload_directory(src, prefix="workspaces/ws/d/raw")

    out = tmp_path / "restored"
    count = await storage.download_prefix(prefix_uri, out)
    assert count == 5  # forced multi-page (page_size=2) -> continuation tokens used
    assert (out / "a.txt").read_text(encoding="utf-8") == "a.txt"
    assert (out / "sub" / "e.txt").read_text(encoding="utf-8") == "e"


async def test_delete_uri_removes_single_object(tmp_path: Path):
    storage, state, _ = _make_storage()
    await storage.initialize()
    local = tmp_path / "x.bin"
    local.write_bytes(b"x")
    uri = await storage.upload_file(local, key="workspaces/ws/d/source/x.bin")
    assert await storage.delete_uri(uri) is True
    assert ("lightrag-kb", "kb/workspaces/ws/d/source/x.bin") not in state.objects


async def test_delete_prefix_paginates_and_counts(tmp_path: Path):
    storage, state, _ = _make_storage(page_size=2)
    await storage.initialize()
    src = tmp_path / "bundle"
    src.mkdir()
    for name in ("a.txt", "b.txt", "c.txt", "d.txt", "e.txt"):
        (src / name).write_text(name, encoding="utf-8")
    prefix_uri = await storage.upload_directory(src, prefix="workspaces/ws/d/raw")

    deleted = await storage.delete_prefix(prefix_uri)
    assert deleted == 5
    remaining = [
        key for (_, key) in state.objects if key.startswith("kb/workspaces/ws/d/raw")
    ]
    assert remaining == []


async def test_delete_workspace_removes_all_workspace_objects(tmp_path: Path):
    storage, state, _ = _make_storage(page_size=3)
    await storage.initialize()
    # Two documents in the same workspace + one in another workspace.
    for doc in ("doc1", "doc2"):
        local = tmp_path / f"{doc}.pdf"
        local.write_bytes(b"pdf")
        await storage.upload_file(
            local, key=f"workspaces/ws_target/documents/{doc}/source/{doc}.pdf"
        )
    other = tmp_path / "other.pdf"
    other.write_bytes(b"pdf")
    await storage.upload_file(
        other, key="workspaces/ws_other/documents/d/source/other.pdf"
    )

    deleted = await storage.delete_workspace("ws_target")
    assert deleted == 2
    survivors = {key for (_, key) in state.objects}
    assert survivors == {"kb/workspaces/ws_other/documents/d/source/other.pdf"}


async def test_presign_download_url_uses_get_object_params(tmp_path: Path):
    storage, state, _ = _make_storage()
    await storage.initialize()
    local = tmp_path / "source.bin"
    local.write_bytes(b"payload")
    uri = await storage.upload_file(local, key="workspaces/ws/doc/source.bin")

    url = await storage.presign_download_url(uri, expires_in_seconds=900)

    assert url == (
        "https://objects.example/lightrag-kb/kb/workspaces/ws/doc/source.bin"
        "?method=get_object&expires=900"
    )
    assert (
        "generate_presigned_url",
        "get_object",
        {"Bucket": "lightrag-kb", "Key": "kb/workspaces/ws/doc/source.bin"},
        900,
    ) in state.calls


async def test_presign_download_url_unquotes_encoded_s3_keys():
    storage, state, _ = _make_storage()
    await storage.initialize()
    uri = "s3://lightrag-kb/kb/workspaces/ws/documents/doc/source/report%20one.pdf"

    url = await storage.presign_download_url(uri, expires_in_seconds=300)

    assert url == (
        "https://objects.example/lightrag-kb/"
        "kb/workspaces/ws/documents/doc/source/report one.pdf"
        "?method=get_object&expires=300"
    )
    assert (
        "generate_presigned_url",
        "get_object",
        {
            "Bucket": "lightrag-kb",
            "Key": "kb/workspaces/ws/documents/doc/source/report one.pdf",
        },
        300,
    ) in state.calls


async def test_presign_download_url_rejects_empty_s3_key():
    storage, _, _ = _make_storage()
    with pytest.raises(ObjectStorageError, match="Object URI missing key"):
        await storage.presign_download_url("s3://lightrag-kb/", expires_in_seconds=300)


def test_validate_document_file_uri_accepts_current_document_scope():
    storage, _, _ = _make_storage()

    storage.validate_document_file_uri(
        "s3://lightrag-kb/kb/workspaces/ws/documents/doc/artifacts/blocks.jsonl",
        workspace="ws",
        document_id="doc",
    )


def test_validate_document_uri_checks_configured_prefix_namespace_and_artifact():
    storage, _, _ = _make_storage()

    storage.validate_document_file_uri(
        "s3://lightrag-kb/kb/workspaces/ws/documents/doc/source/report.pdf",
        workspace="ws",
        document_id="doc",
        namespace="source",
    )
    storage.validate_document_prefix_uri(
        "s3://lightrag-kb/kb/workspaces/ws/documents/doc/"
        "artifacts/raw/artifact-1/bundle/",
        workspace="ws",
        document_id="doc",
        namespace="artifacts",
        artifact_id="artifact-1",
    )

    with pytest.raises(ObjectStorageError, match="configured object prefix"):
        storage.validate_document_file_uri(
            "s3://lightrag-kb/other/workspaces/ws/documents/doc/source/report.pdf",
            workspace="ws",
            document_id="doc",
            namespace="source",
        )
    with pytest.raises(ObjectStorageError, match="namespace"):
        storage.validate_document_file_uri(
            "s3://lightrag-kb/kb/workspaces/ws/documents/doc/"
            "artifacts/raw/artifact-1/report.json",
            workspace="ws",
            document_id="doc",
            namespace="source",
        )
    with pytest.raises(ObjectStorageError, match="artifact id"):
        storage.validate_document_file_uri(
            "s3://lightrag-kb/kb/workspaces/ws/documents/doc/"
            "artifacts/raw/artifact-2/report.json",
            workspace="ws",
            document_id="doc",
            namespace="artifacts",
            artifact_id="artifact-1",
        )


async def test_scope_validation_accepts_actual_document_lifecycle_key_layouts(
    tmp_path: Path,
):
    from lightrag.api.document_lifecycle_service import (
        DocumentLifecycleService,
        DocumentParseExecution,
        DocumentParsePlan,
        _build_parse_artifacts,
    )
    from lightrag.api.metadata_store import DocumentRecord
    from lightrag.constants import PARSER_ENGINE_MINERU

    storage, _, _ = _make_storage()
    await storage.initialize()
    document_dir = tmp_path / "inputs" / "ws" / "doc"
    document_dir.mkdir(parents=True)
    source = document_dir / "report.pdf"
    source.write_bytes(b"pdf")
    parsed_root = document_dir / "__parsed__"
    parsed_root.mkdir()
    sidecar = parsed_root / "report.pdf.parsed"
    sidecar.mkdir()
    blocks = sidecar / "report.pdf.blocks.jsonl"
    blocks.write_text('{"text":"block"}\n', encoding="utf-8")
    (sidecar / "full.md").write_text("# parsed", encoding="utf-8")
    raw_dir = parsed_root / "report.pdf.mineru_raw"
    raw_dir.mkdir()
    (raw_dir / "content_list.json").write_text("[]", encoding="utf-8")

    document = DocumentRecord(
        id="doc",
        kb_id="kb-id",
        workspace="ws",
        lightrag_doc_id="doc-lightrag",
        source_type="upload",
        source_name=source.name,
        source_uri=str(source),
        source_hash="sha256:source",
        content_type="application/pdf",
        size_bytes=source.stat().st_size,
        parser_hash="sha256:parser",
        index_hash=None,
        status="parsed",
        enabled=True,
        archived=False,
        chunks_count=None,
        entity_count=None,
        relation_count=None,
        error_code=None,
        error_message=None,
        metadata={},
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        deleted_at=None,
    )
    service = object.__new__(DocumentLifecycleService)
    service._object_storage = storage
    service._artifact_storage_mode = "local"

    source_uri = await service._persist_source_file(
        document.workspace,
        document.id,
        source,
        content_type=document.content_type,
    )
    assert source_uri == (
        "s3://lightrag-kb/kb/workspaces/ws/documents/doc/source/report.pdf"
    )
    assert source_uri is not None
    document.metadata["source_object_uri"] = source_uri
    plan = DocumentParsePlan(
        document=document,
        source_name=source.name,
        source_object_uri=source_uri,
        raw_object_refs=(),
        parser_engine=PARSER_ENGINE_MINERU,
        process_options="",
        parser_hash="sha256:parser",
        lightrag_doc_id="doc-lightrag",
        force_reparse=False,
        auto_index=False,
    )
    execution = DocumentParseExecution(
        lease=None,
        scratch_document_root=document_dir,
        source_path=source,
        parsed_tree=parsed_root,
        canonical_document_root=document_dir,
    )
    artifacts = _build_parse_artifacts(
        plan,
        execution,
        {"blocks_path": str(blocks)},
        object_authoritative=False,
    )
    persisted, _uploaded = await service._persist_parse_artifacts(plan, artifacts)
    by_type = {artifact.artifact_type: artifact for artifact in persisted}
    assert {"original", "sidecar", "blocks", "raw_dir", "markdown"} <= by_type.keys()

    # The source and original artifact share the source namespace and never
    # carry the original artifact record id in their key.
    storage.validate_document_file_uri(
        source_uri,
        workspace=document.workspace,
        document_id=document.id,
        namespace="source",
    )
    original = by_type["original"]
    assert original.metadata["object_uri"] == source_uri
    assert original.id not in source_uri
    storage.validate_document_file_uri(
        original.metadata["object_uri"],
        workspace=document.workspace,
        document_id=document.id,
        namespace="source",
    )

    for artifact_type in ("sidecar", "raw_dir"):
        artifact = by_type[artifact_type]
        prefix_uri = artifact.metadata["object_prefix_uri"]
        assert f"/artifacts/{artifact.artifact_type}/{artifact.id}/" in prefix_uri
        storage.validate_document_prefix_uri(
            prefix_uri,
            workspace=document.workspace,
            document_id=document.id,
            namespace="artifacts",
            artifact_id=artifact.id,
        )

    for artifact_type in ("blocks", "markdown"):
        artifact = by_type[artifact_type]
        object_uri = artifact.metadata["object_uri"]
        assert f"/artifacts/{artifact.artifact_type}/{artifact.id}/" in object_uri
        storage.validate_document_file_uri(
            object_uri,
            workspace=document.workspace,
            document_id=document.id,
            namespace="artifacts",
            artifact_id=artifact.id,
        )


@pytest.mark.parametrize(
    "prefix",
    ["../kb", "/kb", "kb\\escape", "kb/../escape"],
)
def test_storage_rejects_unsafe_configured_prefix(prefix: str):
    with pytest.raises(ObjectStorageError):
        S3ObjectStorage(
            ObjectStorageConfig(
                backend="minio",
                bucket="lightrag-kb",
                prefix=prefix,
            )
        )


@pytest.mark.parametrize(
    "bucket", ["", "bad/bucket", "bad?bucket", "bad..bucket", "-bad"]
)
def test_storage_rejects_unsafe_configured_bucket(bucket: str):
    with pytest.raises(ObjectStorageError):
        S3ObjectStorage(
            ObjectStorageConfig(
                backend="minio",
                bucket=bucket,
                prefix="kb",
            )
        )


@pytest.mark.parametrize(
    "malicious_relative_key",
    [
        "../escape.txt",
        "sub/../escape.txt",
        "sub\\..\\escape.txt",
        "/absolute.txt",
        "C:/absolute.txt",
    ],
)
async def test_download_prefix_rejects_traversal_keys(
    tmp_path: Path, malicious_relative_key: str
):
    storage, state, _ = _make_storage()
    await storage.initialize()
    prefix = "kb/workspaces/ws/documents/doc/artifacts/raw/artifact-1/bundle"
    state.objects[("lightrag-kb", f"{prefix}/{malicious_relative_key}")] = b"escape"
    out = tmp_path / "materialized"

    with pytest.raises(ObjectStorageError):
        await storage.download_prefix(
            f"s3://lightrag-kb/{prefix}/",
            out,
            max_objects=10,
            max_total_bytes=1024,
        )

    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.parametrize(
    "encoded_suffix",
    [
        "report%2Ffinal.pdf",
        "report%5Cfinal.pdf",
        "%2e%2e/escape.pdf",
        ".%2e/escape.pdf",
    ],
)
def test_uri_validation_rejects_encoded_separator_or_decoded_traversal(
    encoded_suffix: str,
):
    storage, _, _ = _make_storage()

    with pytest.raises(ObjectStorageError):
        storage.validate_document_file_uri(
            "s3://lightrag-kb/kb/workspaces/ws/documents/doc/source/" + encoded_suffix,
            workspace="ws",
            document_id="doc",
            namespace="source",
        )


async def test_download_prefix_rejects_symlink_escape(tmp_path: Path):
    storage, state, _ = _make_storage()
    await storage.initialize()
    prefix = "kb/workspaces/ws/documents/doc/artifacts/raw/artifact-1/bundle"
    state.objects[("lightrag-kb", f"{prefix}/linked/escape.txt")] = b"escape"
    out = tmp_path / "materialized"
    outside = tmp_path / "outside"
    out.mkdir()
    outside.mkdir()
    (out / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ObjectStorageError, match="symlink|escapes"):
        await storage.download_prefix(f"s3://lightrag-kb/{prefix}/", out)

    assert not (outside / "escape.txt").exists()


async def test_download_prefix_enforces_object_and_byte_limits(tmp_path: Path):
    storage, state, _ = _make_storage()
    await storage.initialize()
    prefix = "kb/workspaces/ws/documents/doc/artifacts/raw/artifact-1/bundle"
    state.objects[("lightrag-kb", f"{prefix}/a.bin")] = b"aa"
    state.objects[("lightrag-kb", f"{prefix}/b.bin")] = b"bb"
    uri = f"s3://lightrag-kb/{prefix}/"

    with pytest.raises(ObjectStorageError, match="max_objects"):
        await storage.download_prefix(uri, tmp_path / "count", max_objects=1)
    with pytest.raises(ObjectStorageError, match="max_total_bytes"):
        await storage.download_prefix(
            uri, tmp_path / "bytes", max_objects=10, max_total_bytes=3
        )

    assert list((tmp_path / "count").iterdir()) == []
    assert list((tmp_path / "bytes").iterdir()) == []


@pytest.mark.parametrize(
    ("object_uri", "message"),
    [
        (
            "s3://other-bucket/kb/workspaces/ws/documents/doc/artifacts/blocks.jsonl",
            "bucket does not match",
        ),
        (
            "s3://lightrag-kb/kb/workspaces/other/documents/doc/artifacts/blocks.jsonl",
            "outside the document object prefix",
        ),
        (
            "s3://lightrag-kb/kb/workspaces/ws/documents/other/artifacts/blocks.jsonl",
            "outside the document object prefix",
        ),
        (
            "s3://lightrag-kb/kb/workspaces/ws/documents/doc-extra/artifacts/blocks.jsonl",
            "outside the document object prefix",
        ),
        (
            "s3://lightrag-kb/kb/workspaces/ws/documents/doc/artifacts/",
            "points to a prefix",
        ),
    ],
)
def test_validate_document_file_uri_rejects_untrusted_scope(
    object_uri: str, message: str
):
    storage, _, _ = _make_storage()

    with pytest.raises(ObjectStorageError, match=message):
        storage.validate_document_file_uri(
            object_uri,
            workspace="ws",
            document_id="doc",
        )


async def test_presign_download_url_rejects_invalid_ttl():
    storage, _, _ = _make_storage()
    with pytest.raises(ObjectStorageError):
        await storage.presign_download_url("s3://lightrag-kb/k", expires_in_seconds=0)


async def test_parse_uri_rejects_non_s3_scheme():
    storage, _, _ = _make_storage()
    with pytest.raises(ObjectStorageError):
        await storage.download_file("http://example.com/x", Path("/tmp/x"))


async def test_disabled_object_storage_semantics():
    disabled = DisabledObjectStorage()
    assert await disabled.delete_uri("s3://b/k") is False
    assert await disabled.delete_prefix("s3://b/k/") == 0
    assert await disabled.delete_workspace("ws") == 0
    with pytest.raises(ObjectStorageError):
        await disabled.upload_file(Path("/tmp/x"), key="k")
    with pytest.raises(ObjectStorageError):
        await disabled.stat_object("s3://b/k")


def test_create_object_storage_from_env_selection(monkeypatch):
    monkeypatch.setenv("LIGHTRAG_OBJECT_STORAGE", "local")
    assert create_object_storage_from_env() is None

    monkeypatch.setenv("LIGHTRAG_OBJECT_STORAGE", "minio")
    monkeypatch.setenv("LIGHTRAG_OBJECT_STORAGE_BUCKET", "b")
    storage = create_object_storage_from_env()
    assert isinstance(storage, S3ObjectStorage)

    monkeypatch.setenv("LIGHTRAG_OBJECT_STORAGE", "weird-backend")
    with pytest.raises(ObjectStorageError):
        create_object_storage_from_env()


def test_remove_expect_header_is_case_insensitive():
    class _Request:
        headers = {"expect": "100-continue", "Content-Length": "5"}

    S3ObjectStorage._remove_expect_header(_Request())

    assert _Request.headers == {"Content-Length": "5"}


def test_object_storage_config_reads_disable_expect_flag(monkeypatch):
    monkeypatch.setenv("LIGHTRAG_OBJECT_STORAGE", "minio")
    monkeypatch.setenv("LIGHTRAG_OBJECT_STORAGE_DISABLE_EXPECT_HEADER", "false")

    config = ObjectStorageConfig.from_env()

    assert config.disable_expect_header is False


async def test_configure_client_can_keep_expect_header_registration_disabled():
    storage, state, _ = _make_storage()
    storage._config = ObjectStorageConfig(
        backend="minio",
        bucket="lightrag-kb",
        endpoint_url="http://fake:9000",
        access_key_id="admin",
        secret_access_key="admin123",
        region_name="us-east-1",
        prefix="kb",
        use_ssl=False,
        create_bucket=True,
        disable_expect_header=False,
    )

    await storage.initialize()

    assert state.registered_events == []


async def test_bounded_listing_uses_max_keys_opaque_token_and_metadata_only():
    storage, state, _ = _make_storage(page_size=1)
    await storage.initialize()
    prefix = "kb/workspaces/ws/documents/doc/artifacts/raw/artifact-1/"
    first_key = f"{prefix}a.bin"
    second_key = f"{prefix}b.bin"
    state.objects[("lightrag-kb", first_key)] = b"a"
    state.objects[("lightrag-kb", second_key)] = b"bb"
    state.last_modified[("lightrag-kb", first_key)] = datetime(
        2026, 8, 3, 14, 30, tzinfo=timezone.utc
    )
    state.sha256_checksums[("lightrag-kb", first_key)] = hashlib.sha256(
        b"a"
    ).hexdigest()
    state.current_version_ids[("lightrag-kb", first_key)] = "version-1"

    first = await storage.list_objects_page(f"s3://lightrag-kb/{prefix}", max_keys=1)

    assert len(first.entries) == 1
    assert first.entries[0].key == first_key
    assert first.entries[0].uri == f"s3://lightrag-kb/{first_key}"
    assert first.entries[0].size == 1
    assert first.entries[0].last_modified.tzinfo == timezone.utc
    assert first.entries[0].etag == '"etag-1"'
    assert first.entries[0].version_id == "version-1"
    assert first.entries[0].checksum is None
    assert first.entries[0].checksum_algorithm == "sha256"
    assert first.next_token is not None
    assert first.next_token != first_key
    assert state.list_requests[-1] == {
        "Bucket": "lightrag-kb",
        "Prefix": prefix,
        "MaxKeys": 1,
        "ContinuationToken": None,
    }

    second = await storage.list_objects_page(
        f"s3://lightrag-kb/{prefix}",
        max_keys=1,
        continuation_token=first.next_token,
    )
    assert [entry.key for entry in second.entries] == [second_key]
    assert second.next_token is None
    assert not any(call[0] == "download_file" for call in state.calls)

    other_prefix = "kb/workspaces/ws/documents/other/artifacts/raw/a/"
    with pytest.raises(ObjectStorageError, match="exact prefix"):
        await storage.list_objects_page(
            f"s3://lightrag-kb/{other_prefix}",
            max_keys=1,
            continuation_token=first.next_token,
        )
    with pytest.raises(ObjectStorageError, match="exact prefix"):
        await storage.list_objects_page(
            f"s3://lightrag-kb/{prefix}",
            max_keys=2,
            continuation_token=first.next_token,
        )


async def test_bounded_listing_token_resumes_after_process_reinitialization():
    storage_a, state, session_a = _make_storage(page_size=1)
    await storage_a.initialize()
    prefix = "kb/workspaces/ws/documents/doc/artifacts/raw/artifact-1/"
    first_key = f"{prefix}a.bin"
    second_key = f"{prefix}b.bin"
    state.objects[("lightrag-kb", first_key)] = b"a"
    state.objects[("lightrag-kb", second_key)] = b"b"

    first = await storage_a.list_objects_page(f"s3://lightrag-kb/{prefix}", max_keys=1)
    assert first.next_token is not None

    await storage_a.close()
    del storage_a
    storage_b, shared_state, session_b = _make_storage(
        page_size=1,
        state=state,
    )
    assert shared_state is state
    assert session_b is not session_a
    await storage_b.initialize()

    second = await storage_b.list_objects_page(
        f"s3://lightrag-kb/{prefix}",
        max_keys=1,
        continuation_token=first.next_token,
    )

    assert [entry.key for entry in second.entries] == [second_key]
    assert state.list_requests[-1]["ContinuationToken"] == first_key


async def test_bounded_listing_token_rejects_different_storage_scope_and_limit():
    storage, state, _ = _make_storage(page_size=1)
    prefix = "kb/workspaces/ws/"
    state.objects[("lightrag-kb", f"{prefix}a.bin")] = b"a"
    state.objects[("lightrag-kb", f"{prefix}b.bin")] = b"b"
    first = await storage.list_objects_page(f"s3://lightrag-kb/{prefix}", max_keys=1)
    assert first.next_token is not None

    different_prefix_storage, _, _ = _make_storage(prefix="other", state=state)
    different_bucket_storage, _, _ = _make_storage(
        bucket="other-bucket",
        state=state,
    )
    for scoped_storage, prefix_uri, max_keys in (
        (
            different_prefix_storage,
            "s3://lightrag-kb/other/workspaces/ws/",
            1,
        ),
        (
            different_bucket_storage,
            "s3://other-bucket/kb/workspaces/ws/",
            1,
        ),
        (storage, f"s3://lightrag-kb/{prefix}", 2),
    ):
        with pytest.raises(ObjectStorageError, match="exact prefix"):
            await scoped_storage.list_objects_page(
                prefix_uri,
                max_keys=max_keys,
                continuation_token=first.next_token,
            )


async def test_bounded_listing_token_rejects_malformed_truncated_and_tampered():
    storage, state, _ = _make_storage(page_size=1)
    prefix = "kb/workspaces/ws/"
    state.objects[("lightrag-kb", f"{prefix}a.bin")] = b"a"
    state.objects[("lightrag-kb", f"{prefix}b.bin")] = b"b"
    first = await storage.list_objects_page(f"s3://lightrag-kb/{prefix}", max_keys=1)
    assert first.next_token is not None
    token_prefix, encoded = first.next_token.split(".", 1)
    tamper_index = len(encoded) // 2
    replacement = "A" if encoded[tamper_index] != "A" else "B"
    tampered = (
        f"{token_prefix}."
        f"{encoded[:tamper_index]}{replacement}{encoded[tamper_index + 1 :]}"
    )

    for invalid_token in ("not-a-token", first.next_token[:-1], tampered):
        with pytest.raises(ObjectStorageError, match="token is invalid"):
            await storage.list_objects_page(
                f"s3://lightrag-kb/{prefix}",
                max_keys=1,
                continuation_token=invalid_token,
            )


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1001])
async def test_bounded_listing_rejects_invalid_page_limit(invalid):
    storage, _, _ = _make_storage()
    with pytest.raises(ObjectStorageError, match="max_keys"):
        await storage.list_objects_page(
            "s3://lightrag-kb/kb/workspaces/ws/", max_keys=invalid
        )


@pytest.mark.parametrize(
    "response",
    [
        {"Contents": [], "IsTruncated": True},
        {
            "Contents": [
                {
                    "Key": f"kb/workspaces/ws/{index}",
                    "Size": 1,
                    "LastModified": datetime(2026, 8, 3, tzinfo=timezone.utc),
                }
                for index in range(2)
            ],
            "IsTruncated": False,
        },
        {
            "Contents": [
                {
                    "Key": "kb/workspaces/ws/a",
                    "Size": 1,
                    "LastModified": datetime(2026, 8, 3, tzinfo=timezone.utc),
                },
                {
                    "Key": "kb/workspaces/ws/a",
                    "Size": 1,
                    "LastModified": datetime(2026, 8, 3, tzinfo=timezone.utc),
                },
            ],
            "IsTruncated": False,
        },
        {
            "Contents": [
                {
                    "Key": "kb/workspaces/other/a",
                    "Size": 1,
                    "LastModified": datetime(2026, 8, 3, tzinfo=timezone.utc),
                }
            ],
            "IsTruncated": False,
        },
        {
            "Contents": [
                {
                    "Key": None,
                    "Size": 1,
                    "LastModified": datetime(2026, 8, 3, tzinfo=timezone.utc),
                }
            ],
            "IsTruncated": False,
        },
    ],
)
async def test_bounded_listing_rejects_malformed_pages(response):
    storage, state, _ = _make_storage()
    state.list_response_overrides.append(response)
    with pytest.raises(ObjectStorageMalformedResponseError):
        await storage.list_objects_page(
            "s3://lightrag-kb/kb/workspaces/ws/", max_keys=1
        )


async def test_bounded_listing_rejects_duplicate_across_continuation_pages():
    storage, state, _ = _make_storage()
    entry = {
        "Key": "kb/workspaces/ws/a",
        "Size": 1,
        "LastModified": datetime(2026, 8, 3, tzinfo=timezone.utc),
    }
    state.list_response_overrides.extend(
        [
            {
                "Contents": [entry],
                "IsTruncated": True,
                "NextContinuationToken": "backend-token",
            },
            {"Contents": [entry], "IsTruncated": False},
        ]
    )
    first = await storage.list_objects_page(
        "s3://lightrag-kb/kb/workspaces/ws/", max_keys=1
    )
    assert first.next_token is not None
    with pytest.raises(ObjectStorageMalformedResponseError):
        await storage.list_objects_page(
            "s3://lightrag-kb/kb/workspaces/ws/",
            max_keys=1,
            continuation_token=first.next_token,
        )


def test_cleanup_target_validators_cover_all_namespaces():
    storage, _, _ = _make_storage()
    source = storage.validate_cleanup_target(
        "s3://lightrag-kb/kb/workspaces/ws/documents/doc/source/"
        "generations/srcg-1/report.pdf",
        target_kind="object",
        target_namespace="source",
        workspace="ws",
        document_id="doc",
        source_generation_id="srcg-1",
        origin_job_id="job-1",
        origin_attempt_token="attempt-1",
    )
    legacy = storage.validate_cleanup_target(
        "s3://lightrag-kb/kb/workspaces/ws/documents/doc/source/report.pdf",
        target_kind="object",
        target_namespace="legacy_source",
        workspace="ws",
        document_id="doc",
    )
    artifact = storage.validate_cleanup_target(
        "s3://lightrag-kb/kb/workspaces/ws/documents/doc/artifacts/raw/artifact-1/",
        target_kind="prefix",
        target_namespace="artifact",
        workspace="ws",
        document_id="doc",
        artifact_id="artifact-1",
    )
    staging = storage.validate_cleanup_target(
        "s3://lightrag-kb/kb/workspaces/ws/staging/job-1/attempt-1/candidate.bin",
        target_kind="object",
        target_namespace="staging",
        workspace="ws",
        origin_job_id="job-1",
        origin_attempt_token="attempt-1",
    )
    workspace = storage.validate_cleanup_target(
        "s3://lightrag-kb/kb/workspaces/ws/",
        target_kind="prefix",
        target_namespace="workspace",
        workspace="ws",
        origin_job_id="job-delete",
    )

    assert isinstance(source, ArtifactCleanupTarget)
    assert source.source_generation_id == "srcg-1"
    assert legacy.namespace == "legacy_source"
    assert artifact.artifact_id == "artifact-1"
    assert staging.origin_attempt_token == "attempt-1"
    assert workspace.key == "kb/workspaces/ws/"


@pytest.mark.parametrize(
    ("uri", "kwargs"),
    [
        (
            "s3://other/kb/workspaces/ws/",
            dict(
                target_kind="prefix",
                target_namespace="workspace",
                workspace="ws",
            ),
        ),
        (
            "s3://lightrag-kb/other/workspaces/ws/",
            dict(
                target_kind="prefix",
                target_namespace="workspace",
                workspace="ws",
            ),
        ),
        (
            "s3://lightrag-kb/kb/workspaces/ws/documents/doc/source%2Fescape",
            dict(
                target_kind="object",
                target_namespace="legacy_source",
                workspace="ws",
                document_id="doc",
            ),
        ),
        (
            "s3://lightrag-kb/kb/workspaces/ws/documents/doc/source/%2e%2e/x",
            dict(
                target_kind="object",
                target_namespace="legacy_source",
                workspace="ws",
                document_id="doc",
            ),
        ),
        (
            "s3://lightrag-kb/kb/workspaces/ws/documents/doc/source/generations/x",
            dict(
                target_kind="object",
                target_namespace="legacy_source",
                workspace="ws",
                document_id="doc",
            ),
        ),
        (
            "s3://lightrag-kb/kb/workspaces/ws/narrower/",
            dict(
                target_kind="prefix",
                target_namespace="workspace",
                workspace="ws",
            ),
        ),
        (
            "s3://lightrag-kb/kb/workspaces/ws/staging/job-1/wrong/c.bin",
            dict(
                target_kind="object",
                target_namespace="staging",
                workspace="ws",
                origin_job_id="job-1",
                origin_attempt_token="attempt-1",
            ),
        ),
    ],
)
def test_cleanup_target_validation_rejects_unowned_or_malformed(uri, kwargs):
    storage, _, _ = _make_storage()
    with pytest.raises(ObjectStorageError):
        storage.validate_cleanup_target(uri, **kwargs)


def _validated_source_target(storage: S3ObjectStorage) -> ArtifactCleanupTarget:
    return storage.validate_cleanup_target(
        "s3://lightrag-kb/kb/workspaces/ws/documents/doc/source/"
        "generations/srcg-1/report.bin",
        target_kind="object",
        target_namespace="source",
        workspace="ws",
        document_id="doc",
        source_generation_id="srcg-1",
        origin_job_id="job-1",
        origin_attempt_token="attempt-1",
    )


async def test_verified_exact_delete_compares_integrity_and_is_idempotent():
    storage, state, _ = _make_storage()
    target = _validated_source_target(storage)
    body = b"authoritative"
    state.objects[(target.bucket, target.key)] = body
    digest = hashlib.sha256(body).hexdigest()
    state.sha256_checksums[(target.bucket, target.key)] = digest
    state.etags[(target.bucket, target.key)] = '"opaque-etag"'

    result = await storage.verified_delete_cleanup_target(
        target,
        expected_size_bytes=len(body),
        expected_checksum=f"sha256:{digest}",
        expected_etag="opaque-etag",
    )
    assert result.absent and not result.already_absent
    assert (target.bucket, target.key) not in state.objects

    repeated = await storage.verified_delete_cleanup_target(
        target,
        expected_size_bytes=len(body),
        expected_checksum=f"sha256:{digest}",
        expected_etag="opaque-etag",
    )
    assert repeated.absent and repeated.already_absent


@pytest.mark.parametrize(
    "expectations",
    [
        {"expected_size_bytes": 999},
        {"expected_checksum": "sha256:" + "0" * 64},
        {"expected_etag": "wrong"},
    ],
)
async def test_verified_exact_delete_blocks_integrity_mismatch(expectations):
    storage, state, _ = _make_storage()
    target = _validated_source_target(storage)
    body = b"authoritative"
    state.objects[(target.bucket, target.key)] = body
    state.sha256_checksums[(target.bucket, target.key)] = hashlib.sha256(
        body
    ).hexdigest()
    with pytest.raises(ObjectStorageIntegrityError):
        await storage.verified_delete_cleanup_target(target, **expectations)
    assert (target.bucket, target.key) in state.objects


async def test_verified_delete_uses_custom_sha256_metadata_after_checksum_rejection():
    storage, state, _ = _make_storage()
    target = _validated_source_target(storage)
    state.objects[(target.bucket, target.key)] = b"body"
    state.reject_checksum_mode = True
    digest = hashlib.sha256(b"body").hexdigest()
    state.head_metadata[(target.bucket, target.key)] = {"sha256": digest}

    result = await storage.verified_delete_cleanup_target(
        target,
        expected_checksum=digest,
    )

    assert result.absent
    assert (target.bucket, target.key) not in state.objects
    assert [request["ChecksumMode"] for request in state.head_requests[:2]] == [
        "ENABLED",
        None,
    ]


async def test_verified_delete_blocks_when_required_checksum_is_unavailable():
    storage, state, _ = _make_storage()
    target = _validated_source_target(storage)
    body = b"body"
    digest = hashlib.sha256(body).hexdigest()
    state.objects[(target.bucket, target.key)] = body
    state.sha256_checksums[(target.bucket, target.key)] = digest
    state.reject_checksum_mode = True

    with pytest.raises(ObjectStorageIntegrityError, match="unavailable"):
        await storage.verified_delete_cleanup_target(
            target,
            expected_checksum=digest,
        )
    assert (target.bucket, target.key) in state.objects
    assert state.delete_object_requests == []


async def test_verified_delete_reconciles_ack_loss_absent_vs_present():
    storage, state, _ = _make_storage()
    target = _validated_source_target(storage)
    state.objects[(target.bucket, target.key)] = b"body"
    state.delete_object_behaviors[(target.bucket, target.key)] = "ack_loss"

    result = await storage.verified_delete_cleanup_target(target)
    assert result.absent

    state.objects[(target.bucket, target.key)] = b"body"
    state.delete_object_behaviors[(target.bucket, target.key)] = "ack_loss_present"
    with pytest.raises(ObjectStorageStillPresentError):
        await storage.verified_delete_cleanup_target(target)


async def test_verified_delete_classifies_forbidden_and_transport_readback():
    storage, state, _ = _make_storage()
    target = _validated_source_target(storage)
    state.objects[(target.bucket, target.key)] = b"body"
    state.head_object_errors[(target.bucket, target.key, None)] = [
        _FakeS3Error("AccessDenied", 403)
    ]
    with pytest.raises(ObjectStorageForbiddenError):
        await storage.verified_delete_cleanup_target(target)

    state.head_object_errors[(target.bucket, target.key, None)] = [TimeoutError()]
    with pytest.raises(ObjectStorageTransportError):
        await storage.verified_delete_cleanup_target(target)


def _validated_artifact_prefix(storage: S3ObjectStorage) -> ArtifactCleanupTarget:
    return storage.validate_cleanup_target(
        "s3://lightrag-kb/kb/workspaces/ws/documents/doc/artifacts/raw/artifact-1/",
        target_kind="prefix",
        target_namespace="artifact",
        workspace="ws",
        document_id="doc",
        artifact_id="artifact-1",
    )


async def test_verified_prefix_relists_first_page_and_proves_empty():
    storage, state, _ = _make_storage(page_size=2)
    target = _validated_artifact_prefix(storage)
    for index in range(5):
        state.objects[(target.bucket, f"{target.key}{index}.bin")] = bytes([index])
    renewals = 0

    async def renew():
        nonlocal renewals
        renewals += 1

    result = await storage.verified_delete_cleanup_target(
        target,
        object_page_size=2,
        delete_batch_size=2,
        max_prefix_pages=10,
        before_prefix_page=renew,
    )
    assert result.absent
    assert result.pages_examined == 4
    assert renewals == 7  # every list plus every destructive batch
    assert len(state.list_requests) == 4
    assert all(request["ContinuationToken"] is None for request in state.list_requests)
    assert not any(key.startswith(target.key) for _, key in state.objects)


async def test_verified_prefix_page_budget_and_per_key_errors_fail_closed():
    storage, state, _ = _make_storage(page_size=1)
    target = _validated_artifact_prefix(storage)
    key = f"{target.key}a.bin"
    state.objects[(target.bucket, key)] = b"a"
    with pytest.raises(ObjectStoragePageBudgetError):
        await storage.verified_delete_cleanup_target(
            target,
            object_page_size=1,
            delete_batch_size=1,
            max_prefix_pages=1,
        )

    state.objects[(target.bucket, key)] = b"a"
    state.per_key_delete_errors[(target.bucket, key, None)] = "AccessDenied"
    with pytest.raises(ObjectStorageForbiddenError):
        await storage.verified_delete_cleanup_target(
            target,
            object_page_size=1,
            delete_batch_size=1,
            max_prefix_pages=3,
        )
    assert (target.bucket, key) in state.objects


async def test_versioned_exact_requires_and_deletes_exact_version():
    storage, state, _ = _make_storage()
    target = _validated_source_target(storage)
    state.bucket_versioning[target.bucket] = "Enabled"
    state.versions[(target.bucket, target.key)] = [
        {"VersionId": "v1", "Body": b"old"},
        {"VersionId": "v2", "Body": b"new"},
    ]
    state.sha256_checksums[(target.bucket, target.key)] = hashlib.sha256(
        b"old"
    ).hexdigest()

    with pytest.raises(ObjectStorageVersionProofError):
        await storage.verified_delete_cleanup_target(target)

    result = await storage.verified_delete_cleanup_target(
        target,
        expected_version_id="v1",
        expected_size_bytes=3,
        expected_checksum=hashlib.sha256(b"old").hexdigest(),
    )
    assert result.version_aware and result.absent
    assert [
        item["VersionId"] for item in state.versions[(target.bucket, target.key)]
    ] == ["v2"]


async def test_versioned_exact_blocks_mismatched_head_version_identity():
    storage, state, _ = _make_storage()
    target = _validated_source_target(storage)
    state.bucket_versioning[target.bucket] = "Enabled"
    state.head_response_overrides[(target.bucket, target.key, "expected-v1")] = [
        {
            "ContentLength": 3,
            "ETag": '"etag"',
            "VersionId": "different-v2",
            "LastModified": datetime(2026, 8, 3, tzinfo=timezone.utc),
            "Metadata": {},
        }
    ]

    with pytest.raises(ObjectStorageVersionProofError, match="does not match"):
        await storage.verified_delete_cleanup_target(
            target,
            expected_version_id="expected-v1",
        )


async def test_versioned_prefix_deletes_versions_and_delete_markers():
    storage, state, _ = _make_storage(page_size=2)
    target = _validated_artifact_prefix(storage)
    state.bucket_versioning[target.bucket] = "Suspended"
    state.versions[(target.bucket, f"{target.key}a.bin")] = [
        {"VersionId": "v1", "Body": b"a"},
        {"VersionId": "v2", "Body": b"b"},
    ]
    state.delete_markers.add((target.bucket, f"{target.key}a.bin", "marker-1"))

    result = await storage.verified_delete_cleanup_target(
        target,
        object_page_size=2,
        delete_batch_size=2,
        max_prefix_pages=5,
    )
    assert result.absent and result.version_aware
    assert state.versions[(target.bucket, f"{target.key}a.bin")] == []
    assert not state.delete_markers
    assert any(call[0] == "list_object_versions" for call in state.calls)


async def test_verified_delete_fails_closed_when_version_state_is_unprovable():
    storage, state, _ = _make_storage()
    target = _validated_source_target(storage)
    state.objects[(target.bucket, target.key)] = b"body"
    state.versioning_errors.append(_FakeS3Error("NotImplemented", 501))

    with pytest.raises(ObjectStorageVersionProofError):
        await storage.verified_delete_cleanup_target(target)
    assert (target.bucket, target.key) in state.objects


# ---------------------------------------------------------------------------
# Immutable upload proof (upload_file_if_absent with expected_sha256).
#
# These exercise the Section A production path: every successful outcome is
# proved by a metadata-only HEAD (size + SHA-256) readback. No test issues a
# GetObject/download; the proof is established purely from HEAD metadata.
# ---------------------------------------------------------------------------

_SOURCE_KEY = "workspaces/ws1/documents/doc1/source/generations/srcg-1/source.pdf"


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def test_upload_file_if_absent_proof_creates_and_head_matches(
    tmp_path: Path,
):
    storage, state, _ = _make_storage()
    await storage.initialize()
    payload = b"immutable-upload-bytes"
    local = tmp_path / "src.bin"
    local.write_bytes(payload)
    digest = _sha256_hex(payload)

    uri, created = await storage.upload_file_if_absent(
        local,
        key=_SOURCE_KEY,
        expected_sha256=digest,
    )

    assert created is True
    expected_key = "kb/" + _SOURCE_KEY
    assert uri == f"s3://lightrag-kb/{expected_key}"
    # The PUT attached both the human-readable SHA-256 metadata and the
    # base64 ChecksumSHA256 trailer; HEAD readback proves the object matches.
    assert state.sha256_checksums[("lightrag-kb", expected_key)] == digest
    assert state.head_metadata[("lightrag-kb", expected_key)] == {"sha256": digest}
    stat = await storage.stat_object(uri)
    assert stat.size == len(payload)
    assert stat.checksum == f"sha256:{digest}"


async def test_upload_file_if_absent_proof_precondition_loser_returns_existing(
    tmp_path: Path,
):
    storage, state, _ = _make_storage()
    await storage.initialize()
    payload = b"immutable-upload-bytes"
    local = tmp_path / "src.bin"
    local.write_bytes(payload)
    digest = _sha256_hex(payload)
    # A prior writer already created the identical immutable object.
    expected_key = "kb/" + _SOURCE_KEY
    state.objects[("lightrag-kb", expected_key)] = payload
    state.sha256_checksums[("lightrag-kb", expected_key)] = digest
    state.head_metadata[("lightrag-kb", expected_key)] = {"sha256": digest}

    uri, created = await storage.upload_file_if_absent(
        local,
        key=_SOURCE_KEY,
        expected_sha256=digest,
    )

    assert created is False
    assert uri == f"s3://lightrag-kb/{expected_key}"
    # The loser never overwrote the object: readback still matches the proof.
    assert state.objects[("lightrag-kb", expected_key)] == payload


@pytest.mark.parametrize(
    "mismatch",
    ["size", "checksum"],
)
async def test_upload_file_if_absent_proof_precondition_loser_mismatch_blocks(
    tmp_path: Path,
    mismatch: str,
):
    storage, state, _ = _make_storage()
    await storage.initialize()
    payload = b"immutable-upload-bytes"
    local = tmp_path / "src.bin"
    local.write_bytes(payload)
    digest = _sha256_hex(payload)
    expected_key = "kb/" + _SOURCE_KEY
    # A different object already occupies the immutable key.
    state.objects[("lightrag-kb", expected_key)] = payload
    state.sha256_checksums[("lightrag-kb", expected_key)] = digest
    state.head_metadata[("lightrag-kb", expected_key)] = {"sha256": digest}
    if mismatch == "size":
        state.head_object_sizes[("lightrag-kb", expected_key)] = len(payload) + 1
    else:
        # Different checksum on the existing object.
        other = payload + b"-different"
        state.objects[("lightrag-kb", expected_key)] = other
        other_digest = _sha256_hex(other)
        state.sha256_checksums[("lightrag-kb", expected_key)] = other_digest
        state.head_metadata[("lightrag-kb", expected_key)] = {"sha256": other_digest}

    with pytest.raises(ObjectStorageIntegrityError):
        await storage.upload_file_if_absent(
            local,
            key=_SOURCE_KEY,
            expected_sha256=digest,
        )


async def test_upload_file_if_absent_proof_readback_unavailable_checksum_blocks(
    tmp_path: Path,
):
    storage, state, _ = _make_storage()
    await storage.initialize()
    payload = b"immutable-upload-bytes"
    local = tmp_path / "src.bin"
    local.write_bytes(payload)
    digest = _sha256_hex(payload)
    expected_key = "kb/" + _SOURCE_KEY
    # Object present but the backend reports no checksum even under
    # ChecksumMode=ENABLED, so the proof cannot be established.
    state.objects[("lightrag-kb", expected_key)] = payload
    state.head_metadata[("lightrag-kb", expected_key)] = {}
    state.sha256_checksums.pop(("lightrag-kb", expected_key), None)

    with pytest.raises(ObjectStorageIntegrityError):
        await storage.upload_file_if_absent(
            local,
            key=_SOURCE_KEY,
            expected_sha256=digest,
        )


@pytest.mark.parametrize(
    "error",
    [
        _FakeS3Error("AccessDenied", 403),
        _FakeS3Error("InvalidAccessKeyId", 403),
    ],
)
async def test_upload_file_if_absent_proof_readback_forbidden_blocks(
    tmp_path: Path,
    error,
):
    storage, state, _ = _make_storage()
    await storage.initialize()
    payload = b"immutable-upload-bytes"
    local = tmp_path / "src.bin"
    local.write_bytes(payload)
    digest = _sha256_hex(payload)
    expected_key = "kb/" + _SOURCE_KEY
    # Conditional-create loser path: the object exists but HEAD is forbidden,
    # so absence/presence cannot be proved and the caller must fail closed.
    state.objects[("lightrag-kb", expected_key)] = payload
    state.sha256_checksums[("lightrag-kb", expected_key)] = digest
    state.head_metadata[("lightrag-kb", expected_key)] = {"sha256": digest}
    state.head_object_errors[("lightrag-kb", expected_key, None)] = [error]

    with pytest.raises(ObjectStorageForbiddenError):
        await storage.upload_file_if_absent(
            local,
            key=_SOURCE_KEY,
            expected_sha256=digest,
        )


async def test_upload_file_if_absent_proof_ack_loss_readback_proves_present(
    tmp_path: Path,
):
    storage, state, _ = _make_storage()
    await storage.initialize()
    payload = b"immutable-upload-bytes"
    local = tmp_path / "src.bin"
    local.write_bytes(payload)
    digest = _sha256_hex(payload)
    expected_key = "kb/" + _SOURCE_KEY
    # The PUT reaches the backend (object stored + checksum recorded) but the
    # acknowledgement is lost. The proof path must recover via HEAD readback.
    state.put_object_ack_loss_keys.add(("lightrag-kb", expected_key))

    uri, created = await storage.upload_file_if_absent(
        local,
        key=_SOURCE_KEY,
        expected_sha256=digest,
    )

    # An ACK-loss winner is deliberately reported as pre-existing because
    # exclusive creation ownership cannot be proved.
    assert created is False
    assert uri == f"s3://lightrag-kb/{expected_key}"
    assert state.objects[("lightrag-kb", expected_key)] == payload
    assert state.sha256_checksums[("lightrag-kb", expected_key)] == digest


async def test_upload_file_if_absent_proof_readback_absent_after_ambiguous_blocks(
    tmp_path: Path,
):
    storage, state, _ = _make_storage()
    await storage.initialize()
    payload = b"immutable-upload-bytes"
    local = tmp_path / "src.bin"
    local.write_bytes(payload)
    digest = _sha256_hex(payload)
    expected_key = "kb/" + _SOURCE_KEY
    # The PUT never reached the backend (transport failure before any bytes
    # were stored). The proof path cannot establish presence, so the caller
    # fails closed (no false "created" claim).
    state.put_object_pre_store_errors[("lightrag-kb", expected_key)] = TimeoutError(
        "upload transport lost"
    )

    with pytest.raises(ObjectStorageError):
        await storage.upload_file_if_absent(
            local,
            key=_SOURCE_KEY,
            expected_sha256=digest,
        )
    # No object was durably recorded from the caller's perspective.
    assert ("lightrag-kb", expected_key) not in state.objects


async def test_upload_file_if_absent_proof_local_checksum_mismatch_blocks(
    tmp_path: Path,
):
    storage, state, _ = _make_storage()
    await storage.initialize()
    payload = b"immutable-upload-bytes"
    local = tmp_path / "src.bin"
    local.write_bytes(payload)
    # Expected checksum does not match the local file: never reach the wire.
    wrong_digest = _sha256_hex(b"completely-different-bytes")

    with pytest.raises(ObjectStorageIntegrityError):
        await storage.upload_file_if_absent(
            local,
            key=_SOURCE_KEY,
            expected_sha256=wrong_digest,
        )
    assert not any(call[0] == "put_object" for call in state.calls)
    expected_key = "kb/" + _SOURCE_KEY
    assert ("lightrag-kb", expected_key) not in state.objects


async def test_upload_file_if_absent_proof_rejects_non_canonical_checksum(
    tmp_path: Path,
):
    storage, _, _ = _make_storage()
    await storage.initialize()
    local = tmp_path / "src.bin"
    local.write_bytes(b"x")

    with pytest.raises(ObjectStorageIntegrityError):
        await storage.upload_file_if_absent(
            local,
            key=_SOURCE_KEY,
            expected_sha256="not-a-hex-digest",
        )


async def test_upload_file_if_absent_without_proof_skips_readback(tmp_path: Path):
    storage, state, _ = _make_storage()
    await storage.initialize()
    payload = b"legacy-upload-bytes"
    local = tmp_path / "src.bin"
    local.write_bytes(payload)

    uri, created = await storage.upload_file_if_absent(local, key=_SOURCE_KEY)

    assert created is True
    expected_key = "kb/" + _SOURCE_KEY
    assert uri == f"s3://lightrag-kb/{expected_key}"
    assert state.objects[("lightrag-kb", expected_key)] == payload
    # No proof requested -> no HEAD readback is required for the happy path.
    assert not any(call[0] == "head_object" for call in state.calls)
