"""KB build must survive an INPUT_DIR move without re-parsing.

``artifacts.uri`` is the absolute path the parser wrote to, captured at parse
time. Relocating a deployment (bare metal ``<repo>/inputs`` -> container
``/app/inputs``) leaves every pre-move row pointing outside the current
INPUT_DIR, and the re-enqueue used to reject it with "LightRAG document sidecar
path must stay under INPUT_DIR" even though ``inputs/`` had been copied across.
A KB document always lives at ``<INPUT_DIR>/<workspace>/<document_id>/``, so the
build can re-anchor the recorded path exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lightrag.api.index_build_service import IndexBuildService
from lightrag.api.metadata_store import ArtifactRecord, DocumentRecord
from lightrag.utils_pipeline import resolve_sidecar_uri

WORKSPACE = "kb_demo"
DOCUMENT_ID = "doc-abc"


def _document() -> DocumentRecord:
    return DocumentRecord(
        id=DOCUMENT_ID,
        kb_id="kb_demo",
        workspace=WORKSPACE,
        lightrag_doc_id="doc-lr",
        source_type="upload",
        source_name="a.docx",
        source_uri="",
        source_hash="sha256:x",
        content_type=None,
        size_bytes=1,
        parser_hash=None,
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
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        deleted_at=None,
    )


def _artifact(artifact_type: str, uri: Path) -> ArtifactRecord:
    return ArtifactRecord(
        id=f"art-{artifact_type}",
        kb_id="kb_demo",
        workspace=WORKSPACE,
        document_id=DOCUMENT_ID,
        artifact_type=artifact_type,
        uri=str(uri),
        checksum=None,
        size_bytes=None,
        metadata={},
        created_at="2026-01-01T00:00:00Z",
    )


class _ArtifactsOnlyService:
    """Minimal document-service stand-in: _resolve_artifacts only lists."""

    def __init__(self, artifacts: list[ArtifactRecord]):
        self._artifacts = artifacts

    async def list_document_artifacts(self, kb_id, document_id, limit=200):
        return list(self._artifacts), len(self._artifacts)


def _write_sidecar(parsed_dir: Path) -> Path:
    parsed_dir.mkdir(parents=True, exist_ok=True)
    blocks = parsed_dir / "a.blocks.jsonl"
    blocks.write_text(
        json.dumps({"type": "meta"}) + "\n"
        + json.dumps({"type": "content", "content": "body"}) + "\n",
        encoding="utf-8",
    )
    return blocks


@pytest.mark.asyncio
async def test_build_resolves_artifacts_recorded_under_a_former_input_dir(
    tmp_path, monkeypatch
):
    old_parsed = (
        tmp_path
        / "srv"
        / "LightRAG-API-Server"
        / "inputs"
        / WORKSPACE
        / DOCUMENT_ID
        / "__parsed__"
        / "a.docx.parsed"
    )
    old_blocks = _write_sidecar(old_parsed)

    new_root = tmp_path / "app" / "inputs"
    new_parsed = new_root / WORKSPACE / DOCUMENT_ID / "__parsed__" / "a.docx.parsed"
    new_blocks = _write_sidecar(new_parsed)
    monkeypatch.setenv("INPUT_DIR", str(new_root))

    service = IndexBuildService(
        _ArtifactsOnlyService(
            [_artifact("sidecar", old_parsed), _artifact("blocks", old_blocks)]
        )  # type: ignore[arg-type]
    )
    sidecar_uri, blocks_path = await service._resolve_artifacts("kb_demo", _document())

    assert resolve_sidecar_uri(sidecar_uri) == new_parsed.resolve()
    assert blocks_path == str(new_blocks.resolve())


@pytest.mark.asyncio
async def test_build_leaves_artifacts_alone_when_input_dir_did_not_move(
    tmp_path, monkeypatch
):
    new_root = tmp_path / "app" / "inputs"
    parsed = new_root / WORKSPACE / DOCUMENT_ID / "__parsed__" / "a.docx.parsed"
    blocks = _write_sidecar(parsed)
    monkeypatch.setenv("INPUT_DIR", str(new_root))

    service = IndexBuildService(
        _ArtifactsOnlyService([_artifact("sidecar", parsed), _artifact("blocks", blocks)])  # type: ignore[arg-type]
    )
    sidecar_uri, blocks_path = await service._resolve_artifacts("kb_demo", _document())

    assert resolve_sidecar_uri(sidecar_uri) == parsed.resolve()
    assert blocks_path == str(blocks.resolve())


@pytest.mark.asyncio
async def test_build_keeps_the_recorded_uri_when_nothing_was_copied(
    tmp_path, monkeypatch
):
    """No silent fallback to a different document: the build still fails."""
    old_parsed = (
        tmp_path / "srv" / "inputs" / WORKSPACE / DOCUMENT_ID / "__parsed__" / "a.docx.parsed"
    )
    _write_sidecar(old_parsed)

    new_root = tmp_path / "app" / "inputs"
    new_root.mkdir(parents=True)
    monkeypatch.setenv("INPUT_DIR", str(new_root))

    service = IndexBuildService(
        _ArtifactsOnlyService([_artifact("sidecar", old_parsed)])  # type: ignore[arg-type]
    )
    sidecar_uri, _blocks_path = await service._resolve_artifacts("kb_demo", _document())

    # Unrelocatable -> the stale location is passed through so the enqueue guard
    # raises with the actionable message instead of indexing the wrong sidecar.
    assert resolve_sidecar_uri(sidecar_uri) == old_parsed.resolve()


@pytest.mark.offline
def test_rehydration_target_cannot_escape_the_document_directory(tmp_path):
    """The fallback path is a write target, so a ``..`` tail must be clamped.

    ``_ensure_source_cached`` hands this path to ``ObjectStorage.download_file``,
    which mkdirs the parent. A recorded URI whose tail walks upwards would
    otherwise let an object-storage rehydration write outside INPUT_DIR.
    """
    from types import SimpleNamespace

    from lightrag.api.document_lifecycle_service import _local_document_path

    source_root = (tmp_path / "app" / "inputs").resolve()
    document = SimpleNamespace(workspace="ws", id="doc1")
    document_dir = source_root / "ws" / "doc1"

    escaped = _local_document_path(
        source_root,
        document,
        "/old/inputs/ws/doc1/__parsed__/../../../../../../etc/passwd",
    )
    assert escaped.is_relative_to(document_dir)
    assert escaped == document_dir / "passwd"


@pytest.mark.offline
def test_rehydration_target_keeps_a_normal_parsed_tail(tmp_path):
    """Clamping must not disturb the ordinary relocation it guards."""
    from types import SimpleNamespace

    from lightrag.api.document_lifecycle_service import _local_document_path

    source_root = (tmp_path / "app" / "inputs").resolve()
    document = SimpleNamespace(workspace="ws", id="doc1")

    resolved = _local_document_path(
        source_root, document, "/old/inputs/ws/doc1/__parsed__/a.docx.parsed"
    )
    assert resolved == source_root / "ws" / "doc1" / "__parsed__" / "a.docx.parsed"
