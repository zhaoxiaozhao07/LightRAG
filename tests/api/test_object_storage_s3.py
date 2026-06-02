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

from pathlib import Path
from typing import Any

import pytest

from lightrag.api.object_storage import (
    DisabledObjectStorage,
    ObjectStorageConfig,
    ObjectStorageError,
    S3ObjectStorage,
    create_object_storage_from_env,
)

pytestmark = pytest.mark.offline


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
        if Bucket not in self._state.buckets:
            raise RuntimeError("404 bucket missing")

    async def create_bucket(self, *, Bucket: str) -> None:
        self._state.calls.append(("create_bucket", Bucket))
        self._state.buckets.add(Bucket)

    async def upload_file(self, Filename, Bucket, Key, ExtraArgs=None) -> None:
        self._state.calls.append(("upload_file", Bucket, Key))
        self._state.objects[(Bucket, Key)] = Path(Filename).read_bytes()
        self._state.extra_args[(Bucket, Key)] = ExtraArgs

    async def download_file(self, Bucket, Key, Filename) -> None:
        self._state.calls.append(("download_file", Bucket, Key))
        if (Bucket, Key) not in self._state.objects:
            raise RuntimeError(f"missing object {Bucket}/{Key}")
        Path(Filename).write_bytes(self._state.objects[(Bucket, Key)])

    async def list_objects_v2(self, *, Bucket, Prefix, ContinuationToken=None):
        self._state.calls.append(("list_objects_v2", Bucket, Prefix, ContinuationToken))
        keys = sorted(
            key for (bucket, key) in self._state.objects if bucket == Bucket and key.startswith(Prefix)
        )
        # Model real S3: the continuation token is a stable cursor encoding the
        # last key returned, NOT an integer offset. This stays correct even when
        # the caller deletes objects between pages (delete_prefix does exactly
        # that) — an offset-based cursor would skip items as the list shrinks.
        if ContinuationToken:
            keys = [key for key in keys if key > ContinuationToken]
        page = keys[: self._page_size]
        truncated = len(keys) > self._page_size
        result: dict[str, Any] = {"Contents": [{"Key": key} for key in page]}
        if truncated:
            result["IsTruncated"] = True
            result["NextContinuationToken"] = page[-1]
        else:
            result["IsTruncated"] = False
        return result

    async def delete_object(self, *, Bucket, Key) -> None:
        self._state.calls.append(("delete_object", Bucket, Key))
        self._state.objects.pop((Bucket, Key), None)

    async def delete_objects(self, *, Bucket, Delete) -> None:
        self._state.calls.append(("delete_objects", Bucket, tuple(o["Key"] for o in Delete["Objects"])))
        for obj in Delete["Objects"]:
            self._state.objects.pop((Bucket, obj["Key"]), None)

    def generate_presigned_url(self, ClientMethod, Params, ExpiresIn):
        self._state.calls.append(("generate_presigned_url", ClientMethod, Params, ExpiresIn))
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
        self.registered_events: list[str] = []


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
    *, prefix: str = "kb", create_bucket: bool = True, page_size: int = 2
) -> tuple[S3ObjectStorage, _FakeS3State, _FakeSession]:
    config = ObjectStorageConfig(
        backend="minio",
        bucket="lightrag-kb",
        endpoint_url="http://fake:9000",
        access_key_id="admin",
        secret_access_key="admin123",
        region_name="us-east-1",
        prefix=prefix,
        use_ssl=False,
        create_bucket=create_bucket,
    )
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
    # head_bucket fails (bucket absent) -> create_bucket invoked.
    assert ("head_bucket", "lightrag-kb") in state.calls
    assert ("create_bucket", "lightrag-kb") in state.calls
    assert "lightrag-kb" in state.buckets
    # The s3 client is configured with the endpoint/credentials from config.
    assert session.client_kwargs[0]["endpoint_url"] == "http://fake:9000"
    assert session.client_kwargs[0]["aws_access_key_id"] == "admin"
    assert "before-send.s3.PutObject" in state.registered_events
    assert "before-send.s3.UploadPart" in state.registered_events


async def test_initialize_skips_create_when_bucket_present():
    storage, state, session = _make_storage()
    state.buckets.add("lightrag-kb")
    await storage.initialize()
    assert ("head_bucket", "lightrag-kb") in state.calls
    assert ("create_bucket", "lightrag-kb") not in state.calls


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
    uri = await storage.upload_file(local, key="workspaces/ws1/documents/d/source/src.bin")

    out = tmp_path / "restored" / "src.bin"
    await storage.download_file(uri, out)
    assert out.read_bytes() == b"binary-payload"


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
    remaining = [key for (_, key) in state.objects if key.startswith("kb/workspaces/ws/d/raw")]
    assert remaining == []


async def test_delete_workspace_removes_all_workspace_objects(tmp_path: Path):
    storage, state, _ = _make_storage(page_size=3)
    await storage.initialize()
    # Two documents in the same workspace + one in another workspace.
    for doc in ("doc1", "doc2"):
        local = tmp_path / f"{doc}.pdf"
        local.write_bytes(b"pdf")
        await storage.upload_file(local, key=f"workspaces/ws_target/documents/{doc}/source/{doc}.pdf")
    other = tmp_path / "other.pdf"
    other.write_bytes(b"pdf")
    await storage.upload_file(other, key="workspaces/ws_other/documents/d/source/other.pdf")

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
    uri = (
        "s3://lightrag-kb/"
        "kb/workspaces/ws/documents/doc/source/report%20one.pdf"
    )

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
        await storage.presign_download_url(
            "s3://lightrag-kb/", expires_in_seconds=300
        )


def test_validate_document_file_uri_accepts_current_document_scope():
    storage, _, _ = _make_storage()

    storage.validate_document_file_uri(
        "s3://lightrag-kb/kb/workspaces/ws/documents/doc/artifacts/blocks.jsonl",
        workspace="ws",
        document_id="doc",
    )


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
