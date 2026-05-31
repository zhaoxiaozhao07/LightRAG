from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse


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

    @classmethod
    def from_env(cls) -> "ObjectStorageConfig":
        backend = os.getenv("LIGHTRAG_OBJECT_STORAGE", "local").strip().lower()
        return cls(
            backend=backend,
            bucket=os.getenv("LIGHTRAG_OBJECT_STORAGE_BUCKET", "lightrag-kb"),
            endpoint_url=os.getenv("LIGHTRAG_OBJECT_STORAGE_ENDPOINT"),
            access_key_id=os.getenv("LIGHTRAG_OBJECT_STORAGE_ACCESS_KEY_ID")
            or os.getenv("MINIO_ACCESS_KEY_ID"),
            secret_access_key=os.getenv("LIGHTRAG_OBJECT_STORAGE_SECRET_ACCESS_KEY")
            or os.getenv("MINIO_SECRET_ACCESS_KEY"),
            region_name=os.getenv("LIGHTRAG_OBJECT_STORAGE_REGION") or "us-east-1",
            prefix=os.getenv("LIGHTRAG_OBJECT_STORAGE_PREFIX", "kb").strip("/"),
            use_ssl=_env_bool("LIGHTRAG_OBJECT_STORAGE_USE_SSL", default=False),
            create_bucket=_env_bool("LIGHTRAG_OBJECT_STORAGE_CREATE_BUCKET", default=True),
        )


class ObjectStorageError(RuntimeError):
    pass


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

    async def download_file(self, object_uri: str, local_path: Path) -> None:
        raise NotImplementedError

    async def download_prefix(self, prefix_uri: str, local_dir: Path) -> int:
        raise NotImplementedError

    async def delete_uri(self, object_uri: str) -> bool:
        raise NotImplementedError

    async def delete_prefix(self, prefix_uri: str) -> int:
        raise NotImplementedError

    async def delete_workspace(self, workspace: str) -> int:
        raise NotImplementedError


class DisabledObjectStorage(ObjectStorage):
    async def upload_file(
        self, local_path: Path, *, key: str, content_type: str | None = None
    ) -> str:
        raise ObjectStorageError("Object storage is disabled")

    async def upload_directory(self, local_dir: Path, *, prefix: str) -> str:
        raise ObjectStorageError("Object storage is disabled")

    async def download_file(self, object_uri: str, local_path: Path) -> None:
        raise ObjectStorageError("Object storage is disabled")

    async def download_prefix(self, prefix_uri: str, local_dir: Path) -> int:
        raise ObjectStorageError("Object storage is disabled")

    async def delete_uri(self, object_uri: str) -> bool:
        return False

    async def delete_prefix(self, prefix_uri: str) -> int:
        return 0

    async def delete_workspace(self, workspace: str) -> int:
        return 0


class S3ObjectStorage(ObjectStorage):
    """S3-compatible object storage used for MinIO deployments."""

    def __init__(self, config: ObjectStorageConfig):
        self._config = config
        self._session: Any | None = None

    async def initialize(self) -> None:
        if not self._config.bucket:
            raise ObjectStorageError("LIGHTRAG_OBJECT_STORAGE_BUCKET is required")
        self._session = self._new_session()
        if self._config.create_bucket:
            async with self._client() as client:
                try:
                    await client.head_bucket(Bucket=self._config.bucket)
                except Exception:
                    await client.create_bucket(Bucket=self._config.bucket)

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
                key = f"{object_prefix}/{rel}"
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

    async def download_file(self, object_uri: str, local_path: Path) -> None:
        bucket, key = self._parse_uri(object_uri)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._client() as client:
            await client.download_file(bucket, key, str(local_path))

    async def download_prefix(self, prefix_uri: str, local_dir: Path) -> int:
        bucket, prefix = self._parse_uri(prefix_uri)
        if prefix and not prefix.endswith("/"):
            prefix = f"{prefix}/"
        local_dir.mkdir(parents=True, exist_ok=True)
        downloaded = 0
        async with self._client() as client:
            continuation_token: str | None = None
            while True:
                kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
                if continuation_token:
                    kwargs["ContinuationToken"] = continuation_token
                page = await client.list_objects_v2(**kwargs)
                for item in page.get("Contents", []):
                    key = item["Key"]
                    relative_key = key[len(prefix) :] if key.startswith(prefix) else Path(key).name
                    if not relative_key:
                        continue
                    target = local_dir / relative_key
                    target.parent.mkdir(parents=True, exist_ok=True)
                    await client.download_file(bucket, key, str(target))
                    downloaded += 1
                if not page.get("IsTruncated"):
                    break
                continuation_token = page.get("NextContinuationToken")
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
            continuation_token: str | None = None
            while True:
                kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
                if continuation_token:
                    kwargs["ContinuationToken"] = continuation_token
                page = await client.list_objects_v2(**kwargs)
                objects = [{"Key": item["Key"]} for item in page.get("Contents", [])]
                if objects:
                    await client.delete_objects(
                        Bucket=bucket,
                        Delete={"Objects": objects, "Quiet": True},
                    )
                    deleted += len(objects)
                if not page.get("IsTruncated"):
                    break
                continuation_token = page.get("NextContinuationToken")
        return deleted

    async def delete_workspace(self, workspace: str) -> int:
        return await self.delete_prefix(
            self._prefix_uri(self._normalize_key(f"workspaces/{workspace}"))
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

    def _client(self) -> Any:
        session = self._session
        if session is None:
            session = self._new_session()
            self._session = session
        return session.client(
            "s3",
            endpoint_url=self._config.endpoint_url,
            aws_access_key_id=self._config.access_key_id,
            aws_secret_access_key=self._config.secret_access_key,
            region_name=self._config.region_name,
            use_ssl=self._config.use_ssl,
        )

    def _normalize_key(self, key: str) -> str:
        key = key.strip("/")
        if self._config.prefix:
            key = f"{self._config.prefix}/{key}"
        return key

    def _normalize_prefix(self, prefix: str) -> str:
        return self._normalize_key(prefix).rstrip("/")

    def _uri(self, key: str) -> str:
        return f"s3://{self._config.bucket}/{quote(key, safe='/')}"

    def _prefix_uri(self, prefix: str) -> str:
        return f"s3://{self._config.bucket}/{quote(prefix.rstrip('/'), safe='/')}/"

    @staticmethod
    def _parse_uri(object_uri: str) -> tuple[str, str]:
        parsed = urlparse(object_uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ObjectStorageError(f"Unsupported object URI: {object_uri}")
        return parsed.netloc, parsed.path.lstrip("/")


def create_object_storage_from_env() -> ObjectStorage | None:
    config = ObjectStorageConfig.from_env()
    if config.backend in {"", "local", "disabled", "none"}:
        return None
    if config.backend in {"s3", "minio"}:
        return S3ObjectStorage(config)
    raise ObjectStorageError(f"Unsupported LIGHTRAG_OBJECT_STORAGE: {config.backend}")


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
