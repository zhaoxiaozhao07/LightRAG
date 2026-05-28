from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from lightrag.file_atomic import atomic_write

KB_STATUS_VALUES = {
    "creating",
    "active",
    "disabled",
    "deleting",
    "deleted",
    "error",
}
VISIBILITY_VALUES = {"private", "public", "internal"}
KB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
WORKSPACE_RE = re.compile(r"^[A-Za-z0-9_]+$")
LOCK_POLL_SECONDS = 0.05

KnowledgeBaseStatus = Literal[
    "creating", "active", "disabled", "deleting", "deleted", "error"
]
KnowledgeBaseVisibility = Literal["private", "public", "internal"]


class _UnsetType:
    pass


_UNSET = _UnsetType()
UpdateField = str | None | _UnsetType


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_kb_id(kb_id: str) -> str:
    normalized = kb_id.strip()
    if not normalized:
        raise ValueError("Knowledge base id cannot be empty")
    if not KB_ID_RE.fullmatch(normalized):
        raise ValueError(
            "Knowledge base id must start with a letter or digit and contain only "
            "letters, digits, underscores, or hyphens"
        )
    return normalized


def sanitize_workspace(kb_id: str) -> str:
    validated = validate_kb_id(kb_id)
    encoded = "".join(_encode_workspace_char(char) for char in validated)
    workspace = f"kb_{encoded}"
    if not WORKSPACE_RE.fullmatch(workspace) or workspace == "pipeline_status":
        raise ValueError("Knowledge base id maps to an invalid workspace")
    return workspace


def _encode_workspace_char(char: str) -> str:
    if char.isdigit() or "A" <= char <= "Z" or "a" <= char <= "z":
        return char
    if char == "_":
        return "_u"
    if char == "-":
        return "_d"
    raise ValueError("Knowledge base id contains an unsupported workspace character")


class _MetadataFileLock:
    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self._file: Any | None = None

    def __enter__(self) -> "_MetadataFileLock":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        file = self.lock_path.open("a+b")
        self._file = file
        try:
            if file.tell() == 0 and file.seek(0, os.SEEK_END) == 0:
                file.write(b"0")
                file.flush()
                os.fsync(file.fileno())
            file.seek(0)
            _lock_file_region(file)
        except BaseException:
            file.close()
            self._file = None
            raise
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._file is None:
            return
        try:
            self._file.seek(0)
            _unlock_file_region(self._file)
        finally:
            self._file.close()
            self._file = None


def _lock_file_region(file: Any) -> None:
    if os.name == "nt":
        import msvcrt

        while True:
            try:
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                time.sleep(LOCK_POLL_SECONDS)
    else:
        import fcntl

        fcntl.flock(file.fileno(), fcntl.LOCK_EX)


def _unlock_file_region(file: Any) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, field_name)


@dataclass(slots=True)
class KnowledgeBaseRecord:
    id: str
    name: str
    description: str | None
    workspace: str
    status: KnowledgeBaseStatus
    active_config_version_id: str | None
    owner_id: str | None
    tenant_id: str | None
    visibility: KnowledgeBaseVisibility
    created_at: str
    updated_at: str
    deleted_at: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeBaseRecord":
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            description=data.get("description"),
            workspace=str(data["workspace"]),
            status=data.get("status", "active"),
            active_config_version_id=data.get("active_config_version_id"),
            owner_id=data.get("owner_id"),
            tenant_id=data.get("tenant_id"),
            visibility=data.get("visibility", "private"),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            deleted_at=data.get("deleted_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class KnowledgeBaseServiceError(RuntimeError):
    pass


class KnowledgeBaseNotFoundError(KnowledgeBaseServiceError):
    pass


class KnowledgeBaseConflictError(KnowledgeBaseServiceError):
    pass


class KnowledgeBaseStorageError(KnowledgeBaseServiceError):
    pass


class KnowledgeBaseService:
    def __init__(self, metadata_path: str | Path):
        self.metadata_path = Path(metadata_path)
        self.lock_path = Path(f"{self.metadata_path}.lock")
        self._lock = asyncio.Lock()
        self._loaded = False
        self._records: dict[str, KnowledgeBaseRecord] = {}

    async def initialize(self) -> None:
        async with self._lock:
            self._reload_metadata_locked()

    async def create(
        self,
        *,
        name: str,
        kb_id: str | None = None,
        description: str | None = None,
        owner_id: str | None = None,
        tenant_id: str | None = None,
        visibility: KnowledgeBaseVisibility = "private",
    ) -> KnowledgeBaseRecord:
        normalized_id = (
            validate_kb_id(kb_id) if kb_id is not None else f"kb_{uuid4().hex[:12]}"
        )
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("Knowledge base name cannot be empty")
        if visibility not in VISIBILITY_VALUES:
            raise ValueError("Invalid knowledge base visibility")

        async with self._lock:
            with _MetadataFileLock(self.lock_path):
                self._reload_metadata_locked()
                existing = self._records.get(normalized_id)
                if existing is not None:
                    raise KnowledgeBaseConflictError(
                        f"Knowledge base '{normalized_id}' already exists"
                    )

                now = utc_now_iso()
                record = KnowledgeBaseRecord(
                    id=normalized_id,
                    name=normalized_name,
                    description=description,
                    workspace=sanitize_workspace(normalized_id),
                    status="active",
                    active_config_version_id=None,
                    owner_id=owner_id,
                    tenant_id=tenant_id,
                    visibility=visibility,
                    created_at=now,
                    updated_at=now,
                    deleted_at=None,
                )
                self._records[normalized_id] = record
                self._write_metadata_locked()
                return record

    async def list(self, *, include_deleted: bool = False) -> list[KnowledgeBaseRecord]:
        async with self._lock:
            self._reload_metadata_locked()
            records = [
                record
                for record in self._records.values()
                if include_deleted or record.status != "deleted"
            ]
            return sorted(records, key=lambda record: (record.created_at, record.id))

    async def get(
        self, kb_id: str, *, include_deleted: bool = False
    ) -> KnowledgeBaseRecord:
        normalized_id = validate_kb_id(kb_id)
        async with self._lock:
            self._reload_metadata_locked()
            record = self._records.get(normalized_id)
            if record is None or (record.status == "deleted" and not include_deleted):
                raise KnowledgeBaseNotFoundError(
                    f"Knowledge base '{normalized_id}' not found"
                )
            return record

    async def update(
        self,
        kb_id: str,
        *,
        name: UpdateField = _UNSET,
        description: UpdateField = _UNSET,
        status: UpdateField = _UNSET,
        owner_id: UpdateField = _UNSET,
        tenant_id: UpdateField = _UNSET,
        visibility: UpdateField = _UNSET,
        active_config_version_id: UpdateField = _UNSET,
    ) -> KnowledgeBaseRecord:
        normalized_id = validate_kb_id(kb_id)
        async with self._lock:
            with _MetadataFileLock(self.lock_path):
                self._reload_metadata_locked()
                record = self._records.get(normalized_id)
                if record is None or record.status == "deleted":
                    raise KnowledgeBaseNotFoundError(
                        f"Knowledge base '{normalized_id}' not found"
                    )

                updated = record.to_dict()
                if name is not _UNSET:
                    normalized_name = _require_string(name, "Knowledge base name").strip()
                    if not normalized_name:
                        raise ValueError("Knowledge base name cannot be empty")
                    updated["name"] = normalized_name
                if description is not _UNSET:
                    updated["description"] = _optional_string(description, "Description")
                if status is not _UNSET:
                    status_value = _require_string(status, "Knowledge base status")
                    if status_value not in KB_STATUS_VALUES or status_value == "deleted":
                        raise ValueError("Invalid knowledge base status")
                    updated["status"] = status_value
                if owner_id is not _UNSET:
                    updated["owner_id"] = _optional_string(owner_id, "Owner id")
                if tenant_id is not _UNSET:
                    updated["tenant_id"] = _optional_string(tenant_id, "Tenant id")
                if visibility is not _UNSET:
                    visibility_value = _require_string(visibility, "Knowledge base visibility")
                    if visibility_value not in VISIBILITY_VALUES:
                        raise ValueError("Invalid knowledge base visibility")
                    updated["visibility"] = visibility_value
                if active_config_version_id is not _UNSET:
                    updated["active_config_version_id"] = _optional_string(
                        active_config_version_id, "Active config version id"
                    )
                updated["updated_at"] = utc_now_iso()

                next_record = KnowledgeBaseRecord.from_dict(updated)
                self._records[normalized_id] = next_record
                self._write_metadata_locked()
                return next_record

    async def delete(self, kb_id: str) -> KnowledgeBaseRecord:
        normalized_id = validate_kb_id(kb_id)
        async with self._lock:
            with _MetadataFileLock(self.lock_path):
                self._reload_metadata_locked()
                record = self._records.get(normalized_id)
                if record is None or record.status == "deleted":
                    raise KnowledgeBaseNotFoundError(
                        f"Knowledge base '{normalized_id}' not found"
                    )

                now = utc_now_iso()
                updated = record.to_dict()
                updated["status"] = "deleted"
                updated["updated_at"] = now
                updated["deleted_at"] = now
                deleted_record = KnowledgeBaseRecord.from_dict(updated)
                self._records[normalized_id] = deleted_record
                self._write_metadata_locked()
                return deleted_record

    async def _ensure_loaded_locked(self) -> None:
        if not self._loaded:
            self._reload_metadata_locked()

    def _reload_metadata_locked(self) -> None:
        self._records = self._read_metadata()
        self._loaded = True

    def _read_metadata(self) -> dict[str, KnowledgeBaseRecord]:
        if not self.metadata_path.exists():
            return {}
        try:
            with self.metadata_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except json.JSONDecodeError as exc:
            raise KnowledgeBaseStorageError(
                f"Invalid knowledge base metadata JSON: {self.metadata_path}"
            ) from exc
        except OSError as exc:
            raise KnowledgeBaseStorageError(
                f"Failed to read knowledge base metadata: {self.metadata_path}"
            ) from exc

        raw_records = payload.get("knowledge_bases", {})
        if not isinstance(raw_records, dict):
            raise KnowledgeBaseStorageError("Invalid knowledge base metadata schema")

        records: dict[str, KnowledgeBaseRecord] = {}
        for kb_id, value in raw_records.items():
            if not isinstance(value, dict):
                raise KnowledgeBaseStorageError(
                    f"Invalid metadata record for knowledge base '{kb_id}'"
                )
            record = KnowledgeBaseRecord.from_dict(value)
            records[record.id] = record
        return records

    def _write_metadata_locked(self) -> None:
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "knowledge_bases": {
                kb_id: record.to_dict()
                for kb_id, record in sorted(self._records.items(), key=lambda item: item[0])
            },
        }

        def write_payload(path: str) -> None:
            with open(path, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())

        try:
            atomic_write(str(self.metadata_path), write_payload, workspace="metadata")
        except OSError as exc:
            raise KnowledgeBaseStorageError(
                f"Failed to write knowledge base metadata: {self.metadata_path}"
            ) from exc