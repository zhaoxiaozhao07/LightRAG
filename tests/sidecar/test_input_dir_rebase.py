"""Unit tests for re-anchoring recorded paths after an INPUT_DIR move.

Source files and parse artifacts are recorded in the metadata DB as absolute
paths captured at parse time (``full_docs.sidecar_location``, KB
``artifacts.uri``, ``documents.source_uri``). Relocating a deployment — bare
metal ``<repo>/inputs`` -> container ``/app/inputs`` — leaves every pre-move row
pointing outside the current INPUT_DIR even when the files were copied across,
which surfaced as "LightRAG document sidecar path must stay under INPUT_DIR" on
rebuild. These tests pin the recovery behaviour and its limits.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pytest

from lightrag.utils_pipeline import (
    discovered_legacy_input_dir_roots,
    legacy_input_dir_roots,
    rebase_under_input_dir,
    resolve_sidecar_dir,
    sidecar_blocks_path,
    sidecar_uri_for,
)


def _write_sidecar(parsed_dir: Path, base: str) -> Path:
    parsed_dir.mkdir(parents=True, exist_ok=True)
    blocks = parsed_dir / f"{base}.blocks.jsonl"
    blocks.write_text(
        json.dumps({"type": "meta"}) + "\n"
        + json.dumps({"type": "content", "content": "body"}) + "\n",
        encoding="utf-8",
    )
    return blocks


@pytest.fixture
def relocated(tmp_path, monkeypatch):
    """A KB document parsed under an old INPUT_DIR, copied to a new one."""
    old_root = tmp_path / "srv" / "LightRAG-API-Server" / "inputs"
    new_root = tmp_path / "app" / "inputs"
    rel = Path("kb_demo") / "doc-abc" / "__parsed__" / "a.docx.parsed"
    _write_sidecar(old_root / rel, "a")
    _write_sidecar(new_root / rel, "a")
    monkeypatch.setenv("INPUT_DIR", str(new_root))
    return old_root, new_root, rel


@pytest.mark.offline
def test_path_already_under_input_dir_is_returned_unchanged(relocated):
    _old_root, new_root, rel = relocated
    assert rebase_under_input_dir(new_root / rel) == (new_root / rel).resolve()


@pytest.mark.offline
def test_relative_path_is_anchored_on_input_dir(relocated):
    _old_root, new_root, rel = relocated
    assert rebase_under_input_dir(rel) == (new_root / rel).resolve()


@pytest.mark.offline
def test_legacy_absolute_path_is_rebased_by_tail_match(relocated):
    old_root, new_root, rel = relocated
    assert rebase_under_input_dir(old_root / rel) == (new_root / rel).resolve()


@pytest.mark.offline
def test_declared_legacy_root_rebases_by_prefix_swap(relocated, monkeypatch):
    old_root, new_root, rel = relocated
    monkeypatch.setenv("LIGHTRAG_INPUT_DIR_LEGACY_ROOTS", str(old_root))
    assert legacy_input_dir_roots() == [old_root.resolve()]
    assert rebase_under_input_dir(old_root / rel) == (new_root / rel).resolve()


@pytest.fixture
def _propagate_lightrag_logs():
    """The ``lightrag`` logger sets ``propagate=False``, so caplog's root
    handler would miss its records. Re-enable propagation for the tests that
    assert on the rebase log level."""
    lg = logging.getLogger("lightrag")
    old = lg.propagate
    lg.propagate = True
    try:
        yield
    finally:
        lg.propagate = old


@pytest.mark.offline
def test_configured_migration_does_not_warn(
    relocated, monkeypatch, caplog, _propagate_lightrag_logs
):
    """A declared legacy root is the expected path, so it must not warn.

    Warning once per document for the life of a migrated deployment would bury
    real problems, and the advice the warning carries is already followed.
    """
    old_root, _new_root, rel = relocated
    monkeypatch.setenv("LIGHTRAG_INPUT_DIR_LEGACY_ROOTS", str(old_root))
    with caplog.at_level("INFO", logger="lightrag"):
        assert rebase_under_input_dir(old_root / rel) is not None
    levels = {r.levelname for r in caplog.records}
    assert "WARNING" not in levels, caplog.text
    assert "INFO" in levels
    assert "former INPUT_DIR" in caplog.text


@pytest.mark.offline
def test_heuristic_tail_match_warns_with_remediation(
    relocated, caplog, _propagate_lightrag_logs
):
    """The tail search is a guess, so it stays a warning that says what to set."""
    old_root, _new_root, rel = relocated
    with caplog.at_level("INFO", logger="lightrag"):
        assert rebase_under_input_dir(old_root / rel) is not None
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "LIGHTRAG_INPUT_DIR_LEGACY_ROOTS" in warnings[0].getMessage()


@pytest.mark.offline
def test_anchor_splices_the_parsed_tail_under_the_document_dir(relocated):
    old_root, new_root, _rel = relocated
    # The recorded path carries a *different* workspace/doc prefix (e.g. the KB
    # was re-keyed); the anchor is authoritative, the ``__parsed__`` tail is not.
    recorded = old_root / "stale_ws" / "stale-doc" / "__parsed__" / "a.docx.parsed"
    rebased = rebase_under_input_dir(recorded, anchor=("kb_demo", "doc-abc"))
    assert rebased == (new_root / "kb_demo" / "doc-abc" / "__parsed__" / "a.docx.parsed").resolve()


@pytest.mark.offline
def test_source_file_without_parsed_segment_rebases_on_basename(tmp_path, monkeypatch):
    new_root = tmp_path / "app" / "inputs"
    document_dir = new_root / "kb_demo" / "doc-abc"
    document_dir.mkdir(parents=True)
    (document_dir / "a.docx").write_bytes(b"x")
    monkeypatch.setenv("INPUT_DIR", str(new_root))

    recorded = tmp_path / "srv" / "inputs" / "kb_demo" / "doc-abc" / "a.docx"
    rebased = rebase_under_input_dir(recorded, anchor=("kb_demo", "doc-abc"))
    assert rebased == (document_dir / "a.docx").resolve()


@pytest.mark.offline
def test_missing_artifact_is_not_invented(relocated):
    old_root, _new_root, _rel = relocated
    missing = old_root / "kb_gone" / "doc-x" / "__parsed__" / "z.parsed"
    assert rebase_under_input_dir(missing) is None
    assert rebase_under_input_dir(missing, anchor=("kb_gone", "doc-x")) is None


@pytest.mark.offline
def test_rebase_never_drops_the_parsed_segment(tmp_path, monkeypatch):
    """A tail search must not bind a KB doc to the global ``__parsed__`` root."""
    new_root = tmp_path / "app" / "inputs"
    _write_sidecar(new_root / "__parsed__" / "a.docx.parsed", "a")
    monkeypatch.setenv("INPUT_DIR", str(new_root))

    # Recorded under a KB document dir that was NOT copied across. The only
    # candidate whose tail still starts at ``__parsed__`` is the global one,
    # which belongs to a different document -- so it must be accepted only
    # because the tail is complete, never by shortening past ``__parsed__``.
    recorded = tmp_path / "srv" / "inputs" / "kb_demo" / "doc-abc" / "__parsed__" / "a.docx.parsed"
    assert rebase_under_input_dir(recorded) == (
        new_root / "__parsed__" / "a.docx.parsed"
    ).resolve()

    # Shortening past ``__parsed__`` would let a bare directory name match; it
    # must not.
    stray = tmp_path / "srv" / "inputs" / "kb_demo" / "doc-abc" / "__parsed__" / "b.docx.parsed"
    (new_root / "b.docx.parsed").mkdir()
    assert rebase_under_input_dir(stray) is None


@pytest.mark.offline
def test_sidecar_readers_recover_a_relocated_blocks_file(relocated):
    old_root, new_root, rel = relocated
    recorded_uri = sidecar_uri_for(old_root / rel)
    # The container never has the old host layout.
    shutil.rmtree(old_root)

    assert resolve_sidecar_dir(recorded_uri) == (new_root / rel).resolve()
    assert sidecar_blocks_path(recorded_uri) == str(
        (new_root / rel / "a.blocks.jsonl").resolve()
    )


@pytest.mark.offline
def test_sidecar_readers_prefer_the_recorded_path_when_it_still_exists(relocated):
    old_root, _new_root, rel = relocated
    recorded_uri = sidecar_uri_for(old_root / rel)
    # Both copies exist here, so the recorded location wins -- rebasing is a
    # recovery path, not a redirect.
    assert resolve_sidecar_dir(recorded_uri) == (old_root / rel).resolve()


@pytest.mark.offline
def test_first_tail_match_learns_the_former_root_and_stops_warning(
    relocated, caplog, _propagate_lightrag_logs
):
    """One warning per former root, not one per document.

    An operator migrating a deployment does not necessarily know the old
    absolute path, and warning for every document would make the log useless
    exactly when it is being read. The prefix replaced by the first successful
    tail match *is* the former INPUT_DIR, so remembering it turns every later
    document into the deterministic swap LIGHTRAG_INPUT_DIR_LEGACY_ROOTS
    would have configured.
    """
    old_root, new_root, rel = relocated
    second = Path("kb_demo") / "doc-def" / "__parsed__" / "b.docx.parsed"
    _write_sidecar(old_root / second, "b")
    _write_sidecar(new_root / second, "b")

    with caplog.at_level("INFO", logger="lightrag"):
        assert rebase_under_input_dir(old_root / rel) == (new_root / rel).resolve()
        first_warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(first_warnings) == 1

        # The former root is now known ...
        assert old_root.resolve() in discovered_legacy_input_dir_roots()

        # ... so the next document resolves deterministically and adds no warning.
        caplog.clear()
        assert rebase_under_input_dir(old_root / second) == (
            new_root / second
        ).resolve()

    assert [r.levelname for r in caplog.records] == ["INFO"]
    assert "former INPUT_DIR" in caplog.text


@pytest.mark.offline
def test_learned_root_never_invents_a_missing_artifact(relocated):
    """Learning a prefix must not weaken the existence requirement."""
    old_root, _new_root, rel = relocated
    assert rebase_under_input_dir(old_root / rel) is not None
    assert old_root.resolve() in discovered_legacy_input_dir_roots()

    missing = old_root / "kb_demo" / "doc-gone" / "__parsed__" / "z.docx.parsed"
    assert rebase_under_input_dir(missing) is None
