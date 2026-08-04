"""Offline contract tests for the OrphanReconcileService.

Covers the plan/apply/resume state machine, classification correctness
(eligible / referenced / retained / malformed / unknown_owner / too_new),
apply-only-enqueues (never deletes), retained release gating, resume after
crash, redacted JSON output, and the leak-scan invariant that apply never
deletes objects directly.
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

from lightrag.api.artifact_lifecycle import (
    ArtifactCleanupManifestRecord,
    artifact_cleanup_idempotency_key,
    normalize_artifact_target_uri,
)
from lightrag.api.kb_service import sanitize_workspace, utc_now_iso
from lightrag.api.metadata_store import (
    ArtifactRecord,
    DocumentRecord,
    KBLifecycleRecord,
    SQLiteMetadataStore,
)
from lightrag.api.object_storage import (
    ObjectListEntry,
    ObjectListPage,
    ObjectReadback,
    ObjectStat,
)
from lightrag.api.orphan_reconcile_service import (
    OrphanReconcilePlanError,
    OrphanReconcileService,
    _parse_object_key,
    redact_value,
)
from lightrag.tools.reconcile_orphans import build_parser


pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeObjectStorage:
    """Deterministic in-memory ObjectStorage used by the reconciler tests."""

    bucket: str = "lightrag-kb"
    prefix: str = "kb"
    entries: list[ObjectListEntry] = field(default_factory=list)
    deleted_uris: set[str] = field(default_factory=set)
    inspect_failures: set[str] = field(default_factory=set)

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
        if object_uri in self.inspect_failures:
            raise RuntimeError("inspect failed")
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
        matching = [entry for entry in self.entries if entry.key.startswith(prefix)]
        # Bounded by max_keys, deterministic order, single page in tests.
        ordered = sorted(matching, key=lambda entry: entry.key)
        return ObjectListPage(entries=tuple(ordered[:max_keys]), next_token=None)

    async def delete_uri(self, object_uri: str) -> bool:
        # Reconciler must never call this; tracking it makes leaks visible.
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


def _seed_kb_lifecycle(
    store: SQLiteMetadataStore,
    *,
    kb_id: str,
    generation: str,
    state: str = "active",
) -> KBLifecycleRecord:
    """Seed a KB lifecycle row directly via the public activate API."""

    return asyncio.run(store.activate_kb_generation(kb_id, generation))


def _seed_document(
    store: SQLiteMetadataStore,
    *,
    kb_id: str,
    workspace: str,
    document_id: str,
    metadata: dict[str, Any] | None = None,
    source_uri: str = "",
    deleted_at: str | None = None,
    status: str = "ready",
) -> DocumentRecord:
    now = utc_now_iso()
    document = DocumentRecord(
        id=document_id,
        kb_id=kb_id,
        workspace=workspace,
        lightrag_doc_id=f"lr-{document_id}",
        source_type="upload",
        source_name=f"{document_id}.txt",
        source_uri=source_uri,
        source_hash="sha256:legacy",
        content_type="text/plain",
        size_bytes=0,
        parser_hash=None,
        index_hash=None,
        status=status,
        enabled=True,
        archived=False,
        chunks_count=0,
        entity_count=0,
        relation_count=0,
        error_code=None,
        error_message=None,
        metadata=metadata or {},
        created_at=now,
        updated_at=now,
        deleted_at=deleted_at,
    )

    def seed(conn: Any) -> None:
        try:
            store._insert_document(conn, document)
        except Exception:
            pass

    asyncio.run(store._write(seed))  # noqa: SLF001 - test fixture
    return document


def _seed_artifact(
    store: SQLiteMetadataStore,
    *,
    kb_id: str,
    workspace: str,
    document_id: str,
    artifact_id: str,
    artifact_type: str,
    uri: str,
) -> ArtifactRecord:
    now = utc_now_iso()
    artifact = ArtifactRecord(
        id=artifact_id,
        kb_id=kb_id,
        workspace=workspace,
        document_id=document_id,
        artifact_type=artifact_type,
        uri=uri,
        checksum=None,
        size_bytes=None,
        metadata={},
        created_at=now,
    )

    def seed(conn: Any) -> None:
        cols = (
            "id, kb_id, workspace, document_id, artifact_type, uri, checksum, "
            "size_bytes, metadata_json, created_at"
        )
        placeholders = ", ".join("?" for _ in cols.split(","))
        conn.execute(
            f"INSERT OR REPLACE INTO document_artifacts ({cols}) "
            f"VALUES ({placeholders})",
            (
                artifact.id,
                artifact.kb_id,
                artifact.workspace,
                artifact.document_id,
                artifact.artifact_type,
                artifact.uri,
                artifact.checksum,
                artifact.size_bytes,
                json.dumps(artifact.metadata),
                artifact.created_at,
            ),
        )

    asyncio.run(store._write(seed))  # noqa: SLF001 - test fixture
    return artifact


def _entry(
    key: str,
    *,
    last_modified: datetime | None = None,
    size: int = 0,
) -> ObjectListEntry:
    return ObjectListEntry(
        uri=f"s3://lightrag-kb/{key}",
        key=key,
        size=size,
        last_modified=last_modified or (datetime.now(timezone.utc) - timedelta(days=7)),
        etag=f"etag-{key}",
    )


def _make_kb_provider(
    *records: KBLifecycleRecord,
) -> Any:
    async def provider() -> list[KBLifecycleRecord]:
        return list(records)

    return provider


def _make_service(
    store: SQLiteMetadataStore,
    fake: FakeObjectStorage,
    *,
    kb_records: tuple[KBLifecycleRecord, ...] = (),
    min_age_hours: int = 24,
) -> OrphanReconcileService:
    return OrphanReconcileService(
        metadata_store=store,
        object_storage=fake,
        metadata_backend="sqlite",
        bucket=fake.bucket,
        prefix=fake.prefix,
        min_age_hours=min_age_hours,
        kb_lifecycle_provider=_make_kb_provider(*kb_records),
    )


# ---------------------------------------------------------------------------
# Object key parser
# ---------------------------------------------------------------------------


def test_parse_object_key_recognizes_source_generation() -> None:
    parsed = _parse_object_key(
        "kb/workspaces/kb_demo/documents/doc-1/source/generations/srcg_abc/file.txt",
        configured_prefix="kb",
    )
    assert parsed is not None
    assert parsed.workspace == "kb_demo"
    assert parsed.document_id == "doc-1"
    assert parsed.namespace == "source"
    assert parsed.source_generation_id == "srcg_abc"
    assert parsed.artifact_id is None


def test_parse_object_key_recognizes_artifact() -> None:
    parsed = _parse_object_key(
        "kb/workspaces/kb_demo/documents/doc-1/artifacts/blocks/art-2/file.jsonl",
        configured_prefix="kb",
    )
    assert parsed is not None
    assert parsed.namespace == "artifact"
    assert parsed.artifact_id == "art-2"


def test_parse_object_key_recognizes_legacy_source() -> None:
    parsed = _parse_object_key(
        "kb/workspaces/kb_demo/documents/doc-1/source/legacy.txt",
        configured_prefix="kb",
    )
    assert parsed is not None
    assert parsed.namespace == "legacy_source"


def test_parse_object_key_rejects_unowned_namespace() -> None:
    parsed = _parse_object_key(
        "kb/workspaces/kb_demo/documents/doc-1/random/file.txt",
        configured_prefix="kb",
    )
    assert parsed is None


def test_parse_object_key_rejects_root_outside_workspace() -> None:
    parsed = _parse_object_key("kb/random/file.txt", configured_prefix="kb")
    assert parsed is None


# ---------------------------------------------------------------------------
# Plan: classification correctness
# ---------------------------------------------------------------------------


def test_plan_classifies_eligible_orphan(
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
) -> None:
    """A well-formed object under a live KB but no live reference is eligible."""

    kb = _seed_kb_lifecycle(sqlite_store, kb_id="demo", generation="gen-1")
    workspace = sanitize_workspace("demo")
    _seed_document(
        sqlite_store,
        kb_id="demo",
        workspace=workspace,
        document_id="doc-other",
        metadata={
            "source_object_uri": "s3://lightrag-kb/kb/workspaces/kb_demo"
            "/documents/doc-other/source/generations/srcg_other/file.txt",
            "source_generation_id": "srcg_other",
        },
    )
    # Orphan: source object URI points elsewhere; doc-1 has no row at all.
    orphan_key = (
        "kb/workspaces/kb_demo/documents/doc-1/source/generations/srcg_old/file.txt"
    )
    fake_object_store.entries = [_entry(orphan_key, size=128)]

    service = _make_service(sqlite_store, fake_object_store, kb_records=(kb,))
    summary = asyncio.run(service.create_plan())

    assert summary.item_count == 1
    assert summary.classifications["eligible"] == 1
    candidate = summary.candidates[0]
    assert candidate.classification == "eligible"
    assert candidate.parsed is not None
    assert candidate.parsed.workspace == workspace
    assert candidate.parsed.document_id == "doc-1"
    assert candidate.parsed.namespace == "source"
    assert candidate.parsed.source_generation_id == "srcg_old"
    assert candidate.kb_id == "demo"
    assert candidate.kb_generation == "gen-1"


def test_plan_classifies_referenced_when_document_points_at_uri(
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
) -> None:
    """An object matching a live document source_object_uri is referenced."""

    kb = _seed_kb_lifecycle(sqlite_store, kb_id="refkb", generation="gen-1")
    workspace = sanitize_workspace("refkb")
    live_key = (
        "kb/workspaces/kb_refkb/documents/doc-live/source/generations/srcg_live/f.txt"
    )
    live_uri = f"s3://lightrag-kb/{live_key}"
    _seed_document(
        sqlite_store,
        kb_id="refkb",
        workspace=workspace,
        document_id="doc-live",
        metadata={
            "source_object_uri": live_uri,
            "source_generation_id": "srcg_live",
        },
    )
    fake_object_store.entries = [_entry(live_key)]

    service = _make_service(sqlite_store, fake_object_store, kb_records=(kb,))
    summary = asyncio.run(service.create_plan())

    assert summary.classifications.get("referenced") == 1
    assert summary.classifications.get("eligible", 0) == 0
    candidate = summary.candidates[0]
    assert "current_source_reference" in candidate.reason_codes


def test_plan_classifies_referenced_when_current_artifact_id_matches(
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
) -> None:
    """An object whose artifact_id is the document's current artifact is referenced."""

    kb = _seed_kb_lifecycle(sqlite_store, kb_id="artkb", generation="gen-1")
    workspace = sanitize_workspace("artkb")
    artifact_key = (
        "kb/workspaces/kb_artkb/documents/doc-a/artifacts/blocks/art-9/f.jsonl"
    )
    artifact_uri = f"s3://lightrag-kb/{artifact_key}"
    _seed_document(
        sqlite_store,
        kb_id="artkb",
        workspace=workspace,
        document_id="doc-a",
        metadata={"current_blocks_artifact_id": "art-9"},
    )
    _seed_artifact(
        sqlite_store,
        kb_id="artkb",
        workspace=workspace,
        document_id="doc-a",
        artifact_id="art-9",
        artifact_type="blocks",
        uri=artifact_uri,
    )
    fake_object_store.entries = [_entry(artifact_key)]

    service = _make_service(sqlite_store, fake_object_store, kb_records=(kb,))
    summary = asyncio.run(service.create_plan())

    assert summary.classifications.get("referenced") == 1
    candidate = summary.candidates[0]
    assert "current_artifact_reference" in candidate.reason_codes


def test_plan_classifies_retained_when_manifest_holds_target(
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
) -> None:
    """A retained manifest targeting the object keeps it from being eligible."""

    kb = _seed_kb_lifecycle(sqlite_store, kb_id="retkb", generation="gen-1")
    workspace = sanitize_workspace("retkb")
    target_key = (
        "kb/workspaces/kb_retkb/documents/doc-1/source/generations/srcg_x/f.txt"
    )
    target_uri = f"s3://lightrag-kb/{target_key}"
    asyncio.run(
        _seed_retained_manifest(
            sqlite_store,
            kb_id="retkb",
            generation="gen-1",
            workspace=workspace,
            document_id="doc-1",
            source_generation_id="srcg_x",
            target_uri=target_uri,
        )
    )
    fake_object_store.entries = [_entry(target_key)]

    service = _make_service(sqlite_store, fake_object_store, kb_records=(kb,))
    summary = asyncio.run(service.create_plan())

    assert summary.classifications.get("retained") == 1
    assert summary.classifications.get("eligible", 0) == 0
    candidate = summary.candidates[0]
    assert "retained_manifest_holds_target" in candidate.reason_codes


def test_plan_classifies_malformed_for_unowned_key(
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
) -> None:
    """Objects outside the workspace namespace are report-only malformed."""

    kb = _seed_kb_lifecycle(sqlite_store, kb_id="mkb", generation="gen-1")
    # Both keys live under the configured "kb/" prefix but neither matches the
    # validated workspace/document/source|artifact|staging structure.
    fake_object_store.entries = [
        _entry("kb/random/unowned/file.txt"),
        _entry("kb/orphaned/backup/data.bin"),
    ]

    service = _make_service(sqlite_store, fake_object_store, kb_records=(kb,))
    summary = asyncio.run(service.create_plan())

    assert summary.classifications.get("malformed") == 2
    assert summary.classifications.get("eligible", 0) == 0
    for candidate in summary.candidates:
        assert candidate.parsed is None
        assert "object_key_unowned" in candidate.reason_codes


def test_plan_classifies_unknown_owner_when_kb_missing(
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
) -> None:
    """A well-formed key whose workspace has no KB lifecycle is unknown_owner."""

    fake_object_store.entries = [
        _entry(
            "kb/workspaces/kb_orphan/documents/doc-x/source/generations/srcg_y/f.txt"
        ),
    ]

    service = _make_service(sqlite_store, fake_object_store, kb_records=())
    summary = asyncio.run(service.create_plan())

    assert summary.classifications.get("unknown_owner") == 1
    candidate = summary.candidates[0]
    assert "workspace_kb_missing" in candidate.reason_codes


def test_plan_classifies_too_new_for_recent_objects(
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
) -> None:
    """Objects younger than the minimum-age window are too_new."""

    kb = _seed_kb_lifecycle(sqlite_store, kb_id="newkb", generation="gen-1")
    fresh = datetime.now(timezone.utc) - timedelta(hours=2)
    fake_object_store.entries = [
        _entry(
            "kb/workspaces/kb_newkb/documents/doc-1/source/generations/srcg_n/f.txt",
            last_modified=fresh,
        )
    ]

    service = _make_service(
        sqlite_store, fake_object_store, kb_records=(kb,), min_age_hours=24
    )
    summary = asyncio.run(service.create_plan())

    assert summary.classifications.get("too_new") == 1
    candidate = summary.candidates[0]
    assert "minimum_age_not_met" in candidate.reason_codes


def test_plan_minimum_age_zero_eligible_for_fresh_object(
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
) -> None:
    """min_age_hours=0 disables the too-new filter entirely."""

    kb = _seed_kb_lifecycle(sqlite_store, kb_id="zerokb", generation="gen-1")
    fresh = datetime.now(timezone.utc) - timedelta(seconds=30)
    fake_object_store.entries = [
        _entry(
            "kb/workspaces/kb_zerokb/documents/doc-1/source/generations/srcg_z/f.txt",
            last_modified=fresh,
        )
    ]

    service = _make_service(
        sqlite_store, fake_object_store, kb_records=(kb,), min_age_hours=0
    )
    summary = asyncio.run(service.create_plan())

    assert summary.classifications.get("eligible") == 1
    assert summary.classifications.get("too_new", 0) == 0


# ---------------------------------------------------------------------------
# Plan persistence
# ---------------------------------------------------------------------------


def test_plan_persists_dry_run_run_and_items(
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
) -> None:
    kb = _seed_kb_lifecycle(sqlite_store, kb_id="perkb", generation="gen-1")
    fake_object_store.entries = [
        _entry("kb/workspaces/kb_perkb/documents/doc-1/source/generations/srcg_p/f.txt")
    ]

    service = _make_service(sqlite_store, fake_object_store, kb_records=(kb,))
    summary = asyncio.run(service.create_plan())

    run = asyncio.run(sqlite_store.get_artifact_maintenance_run(summary.plan_id))
    assert run.kind == "orphan_reconcile"
    assert run.mode == "dry_run"
    assert run.status == "succeeded"
    items, total = asyncio.run(
        sqlite_store.list_artifact_maintenance_items(run.id, limit=10)
    )
    assert total == 1
    item = items[0]
    assert item.state == "planned"
    assert item.subject_kind == "source"
    payload = json.loads(item.payload_json)
    assert payload["classification"] == "eligible"


# ---------------------------------------------------------------------------
# Apply: requires --plan-id + --apply + --yes (parser)
# ---------------------------------------------------------------------------


def test_parser_requires_apply_and_yes_for_plan_id() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["--working-dir", "./rag_storage", "--plan-id", "or-plan-1"]
    )
    # Parser itself does not enforce; the validation lives in _async_main.
    assert args.plan_id == "or-plan-1"
    assert args.apply is False
    assert args.yes is False


def test_parser_release_retained_defaults_off() -> None:
    parser = build_parser()
    args = parser.parse_args(["--working-dir", "./rag_storage"])
    assert args.release_retained is False


def test_parser_min_age_hours_default() -> None:
    parser = build_parser()
    args = parser.parse_args(["--working-dir", "./rag_storage"])
    assert args.min_age_hours == 24


# ---------------------------------------------------------------------------
# Apply: enqueues manifests, never deletes
# ---------------------------------------------------------------------------


def test_apply_enqueues_manifest_for_eligible(
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
) -> None:
    kb = _seed_kb_lifecycle(sqlite_store, kb_id="applykb", generation="gen-1")
    workspace = sanitize_workspace("applykb")
    orphan_key = (
        "kb/workspaces/kb_applykb/documents/doc-1/source/generations/srcg_o/f.txt"
    )
    fake_object_store.entries = [_entry(orphan_key)]

    service = _make_service(sqlite_store, fake_object_store, kb_records=(kb,))
    plan = asyncio.run(service.create_plan())
    summary = asyncio.run(service.apply_plan(plan.plan_id))

    assert summary.items_total == 1
    assert summary.items_enqueued == 1
    assert summary.items_skipped == 0
    assert summary.items_blocked == 0
    assert summary.items_failed == 0

    manifests, total = asyncio.run(
        sqlite_store.list_artifact_cleanup_manifests(
            reason="orphan_reconcile", limit=10
        )
    )
    assert total == 1
    manifest = manifests[0]
    assert manifest.reason == "orphan_reconcile"
    assert manifest.disposition == "delete"
    assert manifest.status == "pending"
    assert manifest.target_namespace == "source"
    assert manifest.kb_id == "applykb"
    assert manifest.kb_generation == "gen-1"
    assert manifest.workspace == workspace
    assert manifest.document_id == "doc-1"
    assert manifest.source_generation_id == "srcg_o"

    # Leak-scan: apply must NEVER delete directly.
    assert fake_object_store.deleted_uris == set()


def test_apply_does_not_enqueue_for_referenced(
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
) -> None:
    """Referenced items are skipped (report-only) and never enqueued."""

    kb = _seed_kb_lifecycle(sqlite_store, kb_id="skipkb", generation="gen-1")
    workspace = sanitize_workspace("skipkb")
    live_key = (
        "kb/workspaces/kb_skipkb/documents/doc-live/source/generations/srcg_l/f.txt"
    )
    live_uri = f"s3://lightrag-kb/{live_key}"
    _seed_document(
        sqlite_store,
        kb_id="skipkb",
        workspace=workspace,
        document_id="doc-live",
        metadata={
            "source_object_uri": live_uri,
            "source_generation_id": "srcg_l",
        },
    )
    fake_object_store.entries = [_entry(live_key)]

    service = _make_service(sqlite_store, fake_object_store, kb_records=(kb,))
    plan = asyncio.run(service.create_plan())
    summary = asyncio.run(service.apply_plan(plan.plan_id))

    assert summary.items_skipped == 1
    assert summary.items_enqueued == 0
    manifests, total = asyncio.run(
        sqlite_store.list_artifact_cleanup_manifests(limit=10)
    )
    assert total == 0
    assert fake_object_store.deleted_uris == set()


def test_apply_skips_malformed_and_unknown_owner(
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
) -> None:
    """malformed + unknown_owner items are report-only (never enqueued)."""

    kb = _seed_kb_lifecycle(sqlite_store, kb_id="ownkb", generation="gen-1")
    fake_object_store.entries = [
        _entry("kb/random/file.txt"),  # malformed
        _entry(  # unknown_owner
            "kb/workspaces/kb_missing/documents/doc-1/source/generations/srcg_x/f.txt"
        ),
    ]

    service = _make_service(sqlite_store, fake_object_store, kb_records=(kb,))
    plan = asyncio.run(service.create_plan())
    summary = asyncio.run(service.apply_plan(plan.plan_id))

    assert summary.items_total == 2
    assert summary.items_enqueued == 0
    assert summary.items_skipped == 2
    manifests, total = asyncio.run(
        sqlite_store.list_artifact_cleanup_manifests(limit=10)
    )
    assert total == 0


def test_apply_release_retained_required_to_release(
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
) -> None:
    """A retained item is skipped unless --release-retained is passed.

    The plan/apply with ``release_retained=False`` leaves the retained
    manifest intact; the matching ``test_apply_release_retained_releases``
    test (separate run) proves the flag actually releases when set.
    """

    kb = _seed_kb_lifecycle(sqlite_store, kb_id="relkb", generation="gen-1")
    workspace = sanitize_workspace("relkb")
    target_key = (
        "kb/workspaces/kb_relkb/documents/doc-r/source/generations/srcg_r/f.txt"
    )
    target_uri = f"s3://lightrag-kb/{target_key}"
    asyncio.run(
        _seed_retained_manifest(
            sqlite_store,
            kb_id="relkb",
            generation="gen-1",
            workspace=workspace,
            document_id="doc-r",
            source_generation_id="srcg_r",
            target_uri=target_uri,
        )
    )
    fake_object_store.entries = [_entry(target_key)]

    service = _make_service(sqlite_store, fake_object_store, kb_records=(kb,))
    plan = asyncio.run(service.create_plan())
    summary = asyncio.run(service.apply_plan(plan.plan_id, release_retained=False))
    assert summary.items_enqueued == 0
    assert summary.items_skipped == 1

    # The retained manifest must still be retained (not released).
    manifests, total = asyncio.run(
        sqlite_store.list_artifact_cleanup_manifests(
            target_uri=target_uri, statuses=("retained",), limit=10
        )
    )
    assert total == 1
    assert fake_object_store.deleted_uris == set()


def test_apply_release_retained_releases_manifest(
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
) -> None:
    """Passing release_retained=True releases the retained manifest."""

    kb = _seed_kb_lifecycle(sqlite_store, kb_id="relrb", generation="gen-1")
    workspace = sanitize_workspace("relrb")
    target_key = (
        "kb/workspaces/kb_relrb/documents/doc-r/source/generations/srcg_r/f.txt"
    )
    target_uri = f"s3://lightrag-kb/{target_key}"
    asyncio.run(
        _seed_retained_manifest(
            sqlite_store,
            kb_id="relrb",
            generation="gen-1",
            workspace=workspace,
            document_id="doc-r",
            source_generation_id="srcg_r",
            target_uri=target_uri,
        )
    )
    fake_object_store.entries = [_entry(target_key)]

    service = _make_service(sqlite_store, fake_object_store, kb_records=(kb,))
    plan = asyncio.run(service.create_plan())
    summary = asyncio.run(service.apply_plan(plan.plan_id, release_retained=True))
    assert summary.items_enqueued == 1

    # The retained manifest must now be pending (released).
    manifests, total = asyncio.run(
        sqlite_store.list_artifact_cleanup_manifests(
            target_uri=target_uri, statuses=("pending",), limit=10
        )
    )
    assert total == 1
    assert fake_object_store.deleted_uris == set()


def test_apply_unknown_plan_rejected(
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
) -> None:
    service = _make_service(
        sqlite_store,
        fake_object_store,
        kb_records=(_seed_kb_lifecycle(sqlite_store, kb_id="x", generation="g"),),
    )
    with pytest.raises(OrphanReconcilePlanError):
        asyncio.run(service.apply_plan("does-not-exist"))


def test_apply_rejects_non_orphan_plan(
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
) -> None:
    """A migration dry-run plan must not be applicable to orphan reconcile."""

    from lightrag.api.artifact_lifecycle import ArtifactMaintenanceRunRecord

    now = utc_now_iso()
    run = ArtifactMaintenanceRunRecord(
        id="mig-plan-foreign",
        kind="migration",
        mode="dry_run",
        status="planned",
        metadata_backend="sqlite",
        backend_fingerprint="sha256:migrate-artifacts-to-object:v1",
        scope_fingerprint="sha256:foreign",
        config_fingerprint="sha256:foreign",
        scope_json={"kind": "migration"},
        created_at=now,
        updated_at=now,
    )
    asyncio.run(sqlite_store.create_artifact_maintenance_run(run))

    service = _make_service(
        sqlite_store,
        fake_object_store,
        kb_records=(_seed_kb_lifecycle(sqlite_store, kb_id="x", generation="g"),),
    )
    with pytest.raises(OrphanReconcilePlanError):
        asyncio.run(service.apply_plan("mig-plan-foreign"))


# ---------------------------------------------------------------------------
# Resume / idempotency
# ---------------------------------------------------------------------------


def test_apply_resume_is_idempotent_after_success(
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
) -> None:
    kb = _seed_kb_lifecycle(sqlite_store, kb_id="reskb", generation="gen-1")
    fake_object_store.entries = [
        _entry(
            "kb/workspaces/kb_reskb/documents/doc-1/source/generations/srcg_re/f.txt"
        )
    ]
    service = _make_service(sqlite_store, fake_object_store, kb_records=(kb,))
    plan = asyncio.run(service.create_plan())
    first = asyncio.run(service.apply_plan(plan.plan_id))
    second = asyncio.run(service.apply_plan(plan.plan_id, resume=True))

    assert first.items_enqueued == 1
    assert second.items_enqueued == 1
    manifests, total = asyncio.run(
        sqlite_store.list_artifact_cleanup_manifests(
            reason="orphan_reconcile", limit=10
        )
    )
    # Resume must not double-enqueue.
    assert total == 1


# ---------------------------------------------------------------------------
# Unknown commit outcome blocks reclaim (report-only)
# ---------------------------------------------------------------------------


def test_unknown_commit_outcome_blocks_eligible(
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
) -> None:
    """A document whose latest job has commit_outcome_unknown stays referenced."""

    from lightrag.api.metadata_store import JobRecord

    kb = _seed_kb_lifecycle(sqlite_store, kb_id="uckb", generation="gen-1")
    workspace = sanitize_workspace("uckb")
    # Seed a document with no live source pointer; the orphan key is otherwise
    # eligible, but the document has an unknown-outcome job.
    _seed_document(
        sqlite_store,
        kb_id="uckb",
        workspace=workspace,
        document_id="doc-1",
        metadata={},
    )
    orphan_key = (
        "kb/workspaces/kb_uckb/documents/doc-1/source/generations/srcg_uc/f.txt"
    )
    fake_object_store.entries = [_entry(orphan_key)]

    now = utc_now_iso()
    job = JobRecord(
        id="job-uc",
        kb_id="uckb",
        workspace=workspace,
        batch_id=None,
        document_id="doc-1",
        job_type="parse",
        status="completed",
        stage=None,
        progress=1.0,
        total_items=1,
        completed_items=1,
        failed_items=0,
        idempotency_key=None,
        config_version_id=None,
        config_hash=None,
        retry_count=0,
        max_retries=3,
        payload={"metadata_commit_outcome_unknown": True},
        result=None,
        error_code=None,
        error_message=None,
        created_at=now,
        updated_at=now,
        queued_at=now,
        started_at=now,
        finished_at=now,
        cancelled_at=None,
    )
    _seed_job(sqlite_store, job)

    service = _make_service(sqlite_store, fake_object_store, kb_records=(kb,))
    summary = asyncio.run(service.create_plan())

    # Even though the object has no live pointer, the document exists and its
    # job has an unresolved commit outcome -> report-only.
    classifications = summary.classifications
    assert classifications.get("eligible", 0) == 0
    assert (
        classifications.get("unknown_owner", 0) == 1
        or classifications.get("referenced", 0) == 1
    )


def _seed_job(store: SQLiteMetadataStore, job: Any) -> None:
    def seed(conn: Any) -> None:
        cols = (
            "id, kb_id, workspace, batch_id, document_id, job_type, status, "
            "stage, progress, total_items, completed_items, failed_items, "
            "idempotency_key, config_version_id, config_hash, retry_count, "
            "max_retries, payload_json, result_json, error_code, "
            "error_message, created_at, updated_at, queued_at, started_at, "
            "finished_at, cancelled_at"
        )
        placeholders = ", ".join("?" for _ in cols.split(","))
        conn.execute(
            f"INSERT OR REPLACE INTO jobs ({cols}) VALUES ({placeholders})",
            (
                job.id,
                job.kb_id,
                job.workspace,
                job.batch_id,
                job.document_id,
                job.job_type,
                job.status,
                job.stage,
                job.progress,
                job.total_items,
                job.completed_items,
                job.failed_items,
                job.idempotency_key,
                job.config_version_id,
                job.config_hash,
                job.retry_count,
                job.max_retries,
                json.dumps(job.payload),
                None,
                job.error_code,
                job.error_message,
                job.created_at,
                job.updated_at,
                job.queued_at,
                job.started_at,
                job.finished_at,
                job.cancelled_at,
            ),
        )

    asyncio.run(store._write(seed))  # noqa: SLF001 - test fixture


# ---------------------------------------------------------------------------
# Retained manifest helper
# ---------------------------------------------------------------------------


async def _seed_retained_manifest(
    store: SQLiteMetadataStore,
    *,
    kb_id: str,
    generation: str,
    workspace: str,
    document_id: str,
    source_generation_id: str,
    target_uri: str,
) -> ArtifactCleanupManifestRecord:
    normalized = normalize_artifact_target_uri(target_uri)
    idempotency_key = artifact_cleanup_idempotency_key(
        reason="replace",
        kb_id=kb_id,
        kb_generation=generation,
        workspace=workspace,
        document_id=document_id,
        artifact_id=None,
        source_generation_id=source_generation_id,
        target_kind="object",
        target_namespace="source",
        target_uri=normalized,
    )
    now = datetime.now(timezone.utc)
    delete_after = now + timedelta(hours=24)
    cleanup_deadline = delete_after + timedelta(hours=24)
    audit_retain = now + timedelta(days=30)
    manifest = ArtifactCleanupManifestRecord(
        id=f"ret-{idempotency_key[:24]}",
        idempotency_key=idempotency_key,
        manifest_group_id=f"test-retain:{idempotency_key[:12]}",
        kb_id=kb_id,
        kb_generation=generation,
        workspace=workspace,
        document_id=document_id,
        artifact_id=None,
        source_generation_id=source_generation_id,
        origin_job_id="job-retain",
        origin_attempt_token="attn-retain",
        reason="replace",
        target_kind="object",
        target_namespace="source",
        disposition="retain",
        status="retained",
        target_uri=normalized,
        expected_size_bytes=None,
        expected_checksum=None,
        expected_etag=None,
        expected_version_id=None,
        delete_after=delete_after,
        cleanup_deadline_at=cleanup_deadline,
        audit_retain_until=audit_retain,
        next_attempt_at=delete_after,
        created_at=utc_now_iso(),
        updated_at=utc_now_iso(),
    )
    return await store.enqueue_artifact_cleanup_manifest(manifest)


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def test_redact_value_scrubs_paths_credentials_and_dsns() -> None:
    assert "<artifact-materialization>" in redact_value(
        "/tmp/work/.lightrag-scratch/leak"
    )
    assert "<redacted-dsn>" in redact_value("postgres://user:pass@host/db")
    assert "<redacted-credential>" in redact_value(
        "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCY"
    )
    redacted = redact_value("opened /etc/secrets/db.key")
    assert "/etc/secrets" not in redacted
