"""Offline contract tests for the reconcile-orphans CLI.

Covers parser surface, redacted JSON output, end-to-end plan/apply through
``main``, and the leak-scan invariant that apply never deletes objects.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest

from lightrag.api.metadata_store import SQLiteMetadataStore
from lightrag.api.object_storage import (
    ObjectListEntry,
    ObjectListPage,
    ObjectReadback,
    ObjectStat,
)
from lightrag.tools.reconcile_orphans import build_parser, main


pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Fakes (kept narrow and local so this test file is self-contained)
# ---------------------------------------------------------------------------


@dataclass
class FakeObjectStorage:
    bucket: str = "lightrag-kb"
    prefix: str = "kb"
    entries: list[ObjectListEntry] = field(default_factory=list)
    deleted_uris: set[str] = field(default_factory=set)

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    def object_uri_for_key(self, key: str) -> str:
        return f"s3://{self.bucket}/{key}"

    def object_prefix_uri_for_key(self, prefix: str) -> str:
        return f"s3://{self.bucket}/{prefix.rstrip('/')}/"

    async def inspect_object(
        self, object_uri: str, *, version_id: str | None = None
    ) -> ObjectReadback:
        key = urlparse(object_uri).path.lstrip("/")
        matching = [entry for entry in self.entries if entry.key == key]
        if not matching:
            return ObjectReadback(present=False, stat=None)
        entry = matching[0]
        return ObjectReadback(
            present=True,
            stat=ObjectStat(
                size=entry.size,
                etag=entry.etag,
                checksum=entry.checksum,
                version_id=entry.version_id,
                last_modified=entry.last_modified,
            ),
        )

    async def list_objects_page(
        self,
        prefix_uri: str,
        *,
        max_keys: int = 1000,
        continuation_token: str | None = None,
    ) -> ObjectListPage:
        prefix = urlparse(prefix_uri).path.lstrip("/")
        if not prefix.endswith("/"):
            prefix = prefix + "/"
        matching = sorted(
            (entry for entry in self.entries if entry.key.startswith(prefix)),
            key=lambda entry: entry.key,
        )
        return ObjectListPage(entries=tuple(matching[:max_keys]), next_token=None)

    async def delete_uri(self, object_uri: str) -> bool:
        self.deleted_uris.add(object_uri)
        return True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_store(tmp_path: Path) -> SQLiteMetadataStore:
    store = SQLiteMetadataStore(tmp_path / "metadata.sqlite3")
    asyncio.run(store.initialize())
    yield store
    asyncio.run(store.close())


@pytest.fixture
def fake_object_store() -> FakeObjectStorage:
    return FakeObjectStorage()


def _entry(key: str, *, last_modified: datetime | None = None) -> ObjectListEntry:
    return ObjectListEntry(
        uri=f"s3://lightrag-kb/{key}",
        key=key,
        size=0,
        last_modified=last_modified or (datetime.now(timezone.utc) - timedelta(days=7)),
        etag=f"etag-{key}",
    )


def _seed_kb(store: SQLiteMetadataStore, kb_id: str, generation: str) -> Any:
    return asyncio.run(store.activate_kb_generation(kb_id, generation))


# ---------------------------------------------------------------------------
# Parser surface
# ---------------------------------------------------------------------------


def test_parser_supports_all_specified_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--working-dir",
            "./rag_storage",
            "--object-storage-endpoint",
            "http://localhost:9000",
            "--bucket",
            "lightrag-kb",
            "--prefix",
            "kb",
            "--min-age-hours",
            "48",
            "--metadata-backend",
            "sqlite",
            "--use-ssl",
            "--json",
        ]
    )
    assert args.working_dir == "./rag_storage"
    assert args.bucket == "lightrag-kb"
    assert args.prefix == "kb"
    assert args.min_age_hours == 48
    assert args.metadata_backend == "sqlite"
    assert args.use_ssl is True
    assert args.json is True
    assert args.plan_id is None
    assert args.apply is False
    assert args.yes is False
    assert args.release_retained is False
    assert args.dry_run is False


def test_parser_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["--working-dir", "./rag_storage"])
    assert args.min_age_hours == 24
    assert args.prefix == "kb"
    assert args.release_retained is False
    assert args.apply is False
    assert args.yes is False


def test_parser_requires_working_dir() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


# ---------------------------------------------------------------------------
# main(): plan emits redacted JSON
# ---------------------------------------------------------------------------


def test_main_plan_emits_redacted_json(
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end plan creation through main() with --json output."""

    kb = _seed_kb(sqlite_store, "clikb", "gen-1")
    orphan_key = (
        "kb/workspaces/kb_clikb/documents/doc-1/source/generations/srcg_cli/f.txt"
    )
    fake_object_store.entries = [_entry(orphan_key)]

    import lightrag.tools.reconcile_orphans as cli
    import lightrag.api.orphan_reconcile_service as svc_mod

    monkeypatch.setattr(
        cli,
        "_metadata_store_from_args",
        lambda args, backend: sqlite_store,
    )
    monkeypatch.setattr(
        cli, "_object_storage_from_args", lambda args: fake_object_store
    )

    # Inject the KB lifecycle provider through the service constructor.
    original_init = svc_mod.OrphanReconcileService.__init__

    def patched_init(self: Any, **kwargs: Any) -> None:
        kwargs.setdefault(
            "kb_lifecycle_provider",
            (lambda: _provider([kb])),
        )
        original_init(self, **kwargs)

    monkeypatch.setattr(svc_mod.OrphanReconcileService, "__init__", patched_init)

    rc = main(
        [
            "--working-dir",
            str(tmp_path),
            "--bucket",
            fake_object_store.bucket,
            "--json",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["mode"] == "plan"
    assert payload["item_count"] == 1
    assert payload["classifications"]["eligible"] == 1
    # Redaction: no scratch markers or absolute local roots leak.
    assert ".lightrag-scratch" not in captured
    assert str(tmp_path) not in captured


async def _provider(records: list[Any]) -> list[Any]:
    return list(records)


# ---------------------------------------------------------------------------
# main(): apply requires --plan-id + --apply + --yes (parser validation)
# ---------------------------------------------------------------------------


def test_main_apply_requires_plan_id_apply_and_yes(
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Applying without --apply/--yes must error before contacting the store."""

    import lightrag.tools.reconcile_orphans as cli

    monkeypatch.setattr(
        cli,
        "_metadata_store_from_args",
        lambda args, backend: sqlite_store,
    )
    monkeypatch.setattr(
        cli, "_object_storage_from_args", lambda args: fake_object_store
    )

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--working-dir",
                str(tmp_path),
                "--bucket",
                fake_object_store.bucket,
                "--plan-id",
                "or-plan-1",
            ]
        )
    assert excinfo.value.code == 2
    captured = capsys.readouterr().err
    assert "--apply" in captured or "--yes" in captured


def test_main_apply_requires_yes(
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--plan-id --apply without --yes must error."""

    import lightrag.tools.reconcile_orphans as cli

    monkeypatch.setattr(
        cli,
        "_metadata_store_from_args",
        lambda args, backend: sqlite_store,
    )
    monkeypatch.setattr(
        cli, "_object_storage_from_args", lambda args: fake_object_store
    )

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--working-dir",
                str(tmp_path),
                "--bucket",
                fake_object_store.bucket,
                "--plan-id",
                "or-plan-1",
                "--apply",
            ]
        )
    assert excinfo.value.code == 2
    captured = capsys.readouterr().err
    assert "--yes" in captured


def test_main_release_retained_requires_full_apply_chain(
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--release-retained without --plan-id/--apply/--yes must error."""

    import lightrag.tools.reconcile_orphans as cli

    monkeypatch.setattr(
        cli,
        "_metadata_store_from_args",
        lambda args, backend: sqlite_store,
    )
    monkeypatch.setattr(
        cli, "_object_storage_from_args", lambda args: fake_object_store
    )

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--working-dir",
                str(tmp_path),
                "--bucket",
                fake_object_store.bucket,
                "--release-retained",
            ]
        )
    assert excinfo.value.code == 2
    captured = capsys.readouterr().err
    assert "--release-retained" in captured


# ---------------------------------------------------------------------------
# main(): end-to-end plan -> apply through main()
# ---------------------------------------------------------------------------


def test_main_plan_then_apply_enqueues_manifest(
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full plan+apply through main() enqueues exactly one manifest.

    Leak-scan: ``fake_object_store.deleted_uris`` must remain empty because
    apply only enqueues; the cleanup service does the verified deletion.
    """

    kb = _seed_kb(sqlite_store, "e2ekb", "gen-1")
    orphan_key = (
        "kb/workspaces/kb_e2ekb/documents/doc-1/source/generations/srcg_e2e/f.txt"
    )
    fake_object_store.entries = [_entry(orphan_key)]

    import lightrag.tools.reconcile_orphans as cli
    import lightrag.api.orphan_reconcile_service as svc_mod

    monkeypatch.setattr(
        cli,
        "_metadata_store_from_args",
        lambda args, backend: sqlite_store,
    )
    monkeypatch.setattr(
        cli, "_object_storage_from_args", lambda args: fake_object_store
    )
    original_init = svc_mod.OrphanReconcileService.__init__

    def patched_init(self: Any, **kwargs: Any) -> None:
        kwargs.setdefault(
            "kb_lifecycle_provider",
            (lambda: _provider([kb])),
        )
        original_init(self, **kwargs)

    monkeypatch.setattr(svc_mod.OrphanReconcileService, "__init__", patched_init)

    rc = main(
        [
            "--working-dir",
            str(tmp_path),
            "--bucket",
            fake_object_store.bucket,
            "--json",
        ]
    )
    assert rc == 0
    plan_payload = json.loads(capsys.readouterr().out)
    plan_id = plan_payload["plan_id"]
    assert plan_payload["classifications"]["eligible"] == 1

    rc = main(
        [
            "--working-dir",
            str(tmp_path),
            "--bucket",
            fake_object_store.bucket,
            "--plan-id",
            plan_id,
            "--apply",
            "--yes",
            "--json",
        ]
    )
    assert rc == 0
    apply_payload = json.loads(capsys.readouterr().out)
    assert apply_payload["mode"] == "apply"
    assert apply_payload["items_enqueued"] == 1
    assert apply_payload["items_skipped"] == 0

    manifests, total = asyncio.run(
        sqlite_store.list_artifact_cleanup_manifests(
            reason="orphan_reconcile", limit=10
        )
    )
    assert total == 1
    assert manifests[0].reason == "orphan_reconcile"
    assert manifests[0].status == "pending"
    assert manifests[0].disposition == "delete"

    # Leak-scan: apply must NEVER delete directly.
    assert fake_object_store.deleted_uris == set()
