"""``_manifest.json`` schema for ``*.mineru_raw/`` bundles.

The manifest is the *atomic success marker* for a raw bundle. Its presence
implies "all files in this directory finished downloading"; its content is
the cache key for "is this bundle for the same source file, the same MinerU
parser options, engine version, and endpoint we are using right now?".

Write path: ``write_manifest(path, manifest)`` writes a temp file then
atomically renames to ``_manifest.json``. Local-mode submissions also write
``_pending_task.json`` atomically before polling; a crash mid-download leaves
no success manifest, so the next ``parse_mineru`` call resumes that task when
its source/endpoint/options identity still matches.

Read path: ``load_manifest(path)`` returns ``None`` if absent or malformed
— either way the bundle is treated as stale.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

MANIFEST_FILENAME = "_manifest.json"
MANIFEST_VERSION = "1.0"
MANIFEST_ENGINE = "mineru"
PENDING_TASK_FILENAME = "_pending_task.json"
PENDING_TASK_VERSION = "1.0"


@dataclass
class ManifestFile:
    """One file entry inside the bundle. Size always; sha256 only for the
    critical file (content_list.json) — see :class:`Manifest.critical_file`.
    """

    path: str  # relative to the raw dir
    size: int
    sha256: str | None = None  # ``"sha256:<hex>"`` form or ``None``


@dataclass
class Manifest:
    """Schema for ``_manifest.json``. Backward-compat policy: new optional
    fields can be added without bumping version; **any** mismatch on existing
    field semantics requires a version bump.
    """

    source_content_hash: str  # ``"sha256:<hex>"`` of source file
    source_size_bytes: int
    source_filename_at_parse: str
    critical_file: ManifestFile  # content_list.json; size + sha256
    files: list[ManifestFile]  # other files; size only
    total_size_bytes: int
    task_id: str = ""
    api_mode: str = ""
    engine_version: str = ""
    endpoint_signature: str = ""
    options_signature: str = ""
    downloaded_at: str = ""
    version: str = MANIFEST_VERSION
    engine: str = MANIFEST_ENGINE

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "engine": self.engine,
            "api_mode": self.api_mode,
            "engine_version": self.engine_version,
            "endpoint_signature": self.endpoint_signature,
            "options_signature": self.options_signature,
            "source_content_hash": self.source_content_hash,
            "source_size_bytes": int(self.source_size_bytes),
            "source_filename_at_parse": self.source_filename_at_parse,
            "task_id": self.task_id,
            "downloaded_at": self.downloaded_at,
            "critical_file": asdict(self.critical_file),
            "files": [asdict(f) for f in self.files],
            "total_size_bytes": int(self.total_size_bytes),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Manifest":
        critical_raw = payload.get("critical_file") or {}
        files_raw = payload.get("files") or []
        return cls(
            version=str(payload.get("version") or MANIFEST_VERSION),
            engine=str(payload.get("engine") or MANIFEST_ENGINE),
            api_mode=str(payload.get("api_mode") or ""),
            engine_version=str(payload.get("engine_version") or ""),
            endpoint_signature=str(payload.get("endpoint_signature") or ""),
            options_signature=str(payload.get("options_signature") or ""),
            source_content_hash=str(payload.get("source_content_hash") or ""),
            source_size_bytes=int(payload.get("source_size_bytes") or 0),
            source_filename_at_parse=str(
                payload.get("source_filename_at_parse") or ""
            ),
            task_id=str(payload.get("task_id") or ""),
            downloaded_at=str(payload.get("downloaded_at") or ""),
            critical_file=ManifestFile(
                path=str(critical_raw.get("path") or ""),
                size=int(critical_raw.get("size") or 0),
                sha256=(
                    str(critical_raw["sha256"])
                    if critical_raw.get("sha256")
                    else None
                ),
            ),
            files=[
                ManifestFile(
                    path=str(f.get("path") or ""),
                    size=int(f.get("size") or 0),
                    sha256=(str(f["sha256"]) if f.get("sha256") else None),
                )
                for f in files_raw
                if isinstance(f, dict)
            ],
            total_size_bytes=int(payload.get("total_size_bytes") or 0),
        )


@dataclass
class PendingLocalTask:
    """Durable identity for a submitted local MinerU task awaiting download."""

    task_id: str
    source_content_hash: str
    source_size_bytes: int
    source_filename_at_parse: str
    endpoint_signature: str
    options_signature: str
    submitted_at: str
    api_mode: str = "local"
    version: str = PENDING_TASK_VERSION
    engine: str = MANIFEST_ENGINE

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "engine": self.engine,
            "api_mode": self.api_mode,
            "task_id": self.task_id,
            "source_content_hash": self.source_content_hash,
            "source_size_bytes": int(self.source_size_bytes),
            "source_filename_at_parse": self.source_filename_at_parse,
            "endpoint_signature": self.endpoint_signature,
            "options_signature": self.options_signature,
            "submitted_at": self.submitted_at,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "PendingLocalTask":
        return cls(
            version=str(payload.get("version") or PENDING_TASK_VERSION),
            engine=str(payload.get("engine") or MANIFEST_ENGINE),
            api_mode=str(payload.get("api_mode") or ""),
            task_id=str(payload.get("task_id") or ""),
            source_content_hash=str(payload.get("source_content_hash") or ""),
            source_size_bytes=int(payload.get("source_size_bytes") or 0),
            source_filename_at_parse=str(
                payload.get("source_filename_at_parse") or ""
            ),
            endpoint_signature=str(payload.get("endpoint_signature") or ""),
            options_signature=str(payload.get("options_signature") or ""),
            submitted_at=str(payload.get("submitted_at") or ""),
        )

def manifest_path(raw_dir: Path) -> Path:
    return raw_dir / MANIFEST_FILENAME


def pending_task_path(raw_dir: Path) -> Path:
    return raw_dir / PENDING_TASK_FILENAME


def load_manifest(raw_dir: Path) -> Manifest | None:
    """Return the parsed manifest or ``None`` if absent / malformed."""
    p = manifest_path(raw_dir)
    if not p.is_file():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != MANIFEST_VERSION:
        return None
    if payload.get("engine") != MANIFEST_ENGINE:
        return None
    try:
        return Manifest.from_dict(payload)
    except (TypeError, ValueError):
        return None


def write_manifest(raw_dir: Path, manifest: Manifest) -> None:
    """Atomically write the manifest. The temp-file + rename pattern
    guarantees the manifest never appears in a partially-written state."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    final = manifest_path(raw_dir)
    tmp = final.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, final)


def load_pending_task(raw_dir: Path) -> PendingLocalTask | None:
    """Return a valid pending-task record, or ``None`` when absent/malformed."""
    path = pending_task_path(raw_dir)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != PENDING_TASK_VERSION:
        return None
    if payload.get("engine") != MANIFEST_ENGINE:
        return None
    try:
        pending = PendingLocalTask.from_dict(payload)
    except (TypeError, ValueError):
        return None
    if pending.api_mode != "local" or not pending.task_id:
        return None
    return pending


def write_pending_task(raw_dir: Path, pending: PendingLocalTask) -> None:
    """Atomically persist a submitted local task before polling begins."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    final = pending_task_path(raw_dir)
    tmp = final.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(pending.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, final)


def delete_pending_task(raw_dir: Path) -> None:
    pending_task_path(raw_dir).unlink(missing_ok=True)


# Re-exported for convenience.
__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_VERSION",
    "MANIFEST_ENGINE",
    "Manifest",
    "ManifestFile",
    "PENDING_TASK_FILENAME",
    "PENDING_TASK_VERSION",
    "PendingLocalTask",
    "delete_pending_task",
    "load_manifest",
    "load_pending_task",
    "manifest_path",
    "pending_task_path",
    "write_manifest",
    "write_pending_task",
]
