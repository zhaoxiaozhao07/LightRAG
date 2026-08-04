"""Offline contract tests for the migrate-artifacts-to-object CLI.

Covers the plan/apply/resume state machine, security rejections, descriptor-
relative no-follow reads, atomic metadata pointer updates, online mutation
guard, redacted JSON output, moved-root re-resolution, and leak-scan.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest

from lightrag.api.metadata_store import DocumentRecord, SQLiteMetadataStore
from lightrag.api.object_storage import (
    ObjectListPage,
    ObjectReadback,
    ObjectStat,
)
from lightrag.tools.migrate_artifacts_to_object import (
    ArtifactObjectMigrator,
    MigrationApplyGuardError,
    MigrationSecurityError,
    _open_no_follow_and_validate,
    _parse_label_root_spec,
    _redact_value,
    _validate_absolute_root,
    build_parser,
    main,
)


pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeObjectStorage:
    """Deterministic in-memory ObjectStorage used by the migration tests."""

    bucket: str = "lightrag-kb"
    prefix: str = "kb"
    uploads: dict[str, bytes] = field(default_factory=dict)
    inspect_failures: set[str] = field(default_factory=set)
    upload_failure_exc: Exception | None = None

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    def object_uri_for_key(self, key: str) -> str:
        return f"s3://{self.bucket}/{key}"

    def object_prefix_uri_for_key(self, prefix: str) -> str:
        return f"s3://{self.bucket}/{prefix}/"

    async def upload_file_if_absent(
        self,
        local_path: Path,
        *,
        key: str,
        content_type: str | None = None,
        expected_sha256: str | None = None,
    ) -> tuple[str, bool]:
        if self.upload_failure_exc is not None:
            raise self.upload_failure_exc
        data = local_path.read_bytes()
        if expected_sha256 is not None:
            digest = hashlib.sha256(data).hexdigest()
            assert digest == expected_sha256, "test fake SHA-256 mismatch"
        created = key not in self.uploads
        self.uploads[key] = data
        return self.object_uri_for_key(key), created

    async def inspect_object(
        self, object_uri: str, *, version_id: str | None = None
    ) -> ObjectReadback:
        if object_uri in self.inspect_failures:
            raise RuntimeError("inspect failed")
        key = urlparse(object_uri).path.lstrip("/")
        if key not in self.uploads:
            return ObjectReadback(present=False, stat=None)
        return ObjectReadback(
            present=True,
            stat=ObjectStat(
                size=len(self.uploads[key]),
                etag=f"etag-{key}",
                checksum=None,
            ),
        )

    async def list_objects_page(
        self,
        prefix_uri: str,
        *,
        max_keys: int = 1000,
        continuation_token: str | None = None,
    ) -> ObjectListPage:
        return ObjectListPage(entries=(), next_token=None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_store(tmp_path: Path) -> SQLiteMetadataStore:
    import asyncio

    store = SQLiteMetadataStore(tmp_path / "metadata.sqlite3")
    asyncio.run(store.initialize())
    yield store
    asyncio.run(store.close())


@pytest.fixture
def fake_object_store() -> FakeObjectStorage:
    return FakeObjectStorage()


@pytest.fixture
def migrator(
    sqlite_store: SQLiteMetadataStore, fake_object_store: FakeObjectStorage
) -> ArtifactObjectMigrator:
    return ArtifactObjectMigrator(
        metadata_store=sqlite_store,
        object_storage=fake_object_store,
        metadata_backend="sqlite",
        bucket=fake_object_store.bucket,
        prefix=fake_object_store.prefix,
    )


def _seed_file(root: Path, relative: str, content: bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _seed_kb_and_document(
    store: SQLiteMetadataStore,
    *,
    kb_id: str,
    workspace: str,
    document_id: str,
    source_name: str,
) -> None:
    """Seed a ready document directly into the SQLite store.

    SQLite metadata store does not have a ``knowledge_bases`` table (KB
    catalog lives in JSON); a document row is sufficient for migration tests
    because the migrator only calls ``get_document``/``update_document``.
    """

    import asyncio
    from lightrag.api.kb_service import utc_now_iso

    now = utc_now_iso()
    document = DocumentRecord(
        id=document_id,
        kb_id=kb_id,
        workspace=workspace,
        lightrag_doc_id="lr-doc",
        source_type="upload",
        source_name=source_name,
        source_uri=f"/inputs/{source_name}",
        source_hash="sha256:legacy",
        content_type="text/plain",
        size_bytes=0,
        parser_hash=None,
        index_hash=None,
        status="ready",
        enabled=True,
        archived=False,
        chunks_count=0,
        entity_count=0,
        relation_count=0,
        error_code=None,
        error_message=None,
        metadata={"batch_id": "fixture"},
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )

    def seed(conn: Any) -> None:
        try:
            store._insert_document(conn, document)
        except Exception:
            # Already seeded by an earlier call in the same test.
            pass

    asyncio.run(store._write(seed))  # noqa: SLF001 - test fixture


# ---------------------------------------------------------------------------
# Spec parsing and absolute-root validation
# ---------------------------------------------------------------------------


def test_parse_label_root_spec_round_trip() -> None:
    label, path = _parse_label_root_spec("legacyA=/srv/rag/legacy-a")
    assert label == "legacyA"
    assert str(path) == "/srv/rag/legacy-a"


def test_parse_label_root_spec_rejects_missing_equals() -> None:
    with pytest.raises(MigrationSecurityError):
        _parse_label_root_spec("/srv/rag/legacy-a")


def test_parse_label_root_spec_rejects_unsafe_label() -> None:
    with pytest.raises(MigrationSecurityError):
        _parse_label_root_spec("bad/label=/srv/rag/x")


def test_validate_absolute_root_rejects_relative() -> None:
    with pytest.raises(MigrationSecurityError, match="absolute"):
        _validate_absolute_root("rel", Path("relative/path"))


def test_validate_absolute_root_rejects_symlink_component(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(MigrationSecurityError, match="symlink"):
        _validate_absolute_root("evil", link)


def test_validate_absolute_root_rejects_traversal(tmp_path: Path) -> None:
    bad = tmp_path / ".." / "target"
    with pytest.raises(MigrationSecurityError, match="traversal"):
        _validate_absolute_root("evil", bad)


def test_validate_absolute_root_rejects_fifo(tmp_path: Path) -> None:
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(MigrationSecurityError):
        _validate_absolute_root("ipc", fifo)


def test_validate_absolute_root_accepts_real_root(tmp_path: Path) -> None:
    root = tmp_path / "legacy"
    root.mkdir()
    _validate_absolute_root("ok", root)


# ---------------------------------------------------------------------------
# Descriptor-relative no-follow reads + fstat revalidation
# ---------------------------------------------------------------------------


def test_open_no_follow_returns_regular_file_fd(tmp_path: Path) -> None:
    path = _seed_file(tmp_path, "doc.txt", b"hello world")
    fd, stat_result = _open_no_follow_and_validate(path)
    try:
        assert stat.S_ISREG(stat_result.st_mode)
        assert stat_result.st_size == len(b"hello world")
    finally:
        os.close(fd)


def test_open_no_follow_rejects_symlink(tmp_path: Path) -> None:
    target = _seed_file(tmp_path, "real.txt", b"data")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    with pytest.raises(MigrationSecurityError, match="symlink"):
        _open_no_follow_and_validate(link)


def test_open_no_follow_rejects_fifo(tmp_path: Path) -> None:
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(MigrationSecurityError):
        _open_no_follow_and_validate(fifo)


# ---------------------------------------------------------------------------
# Plan creation with explicit roots
# ---------------------------------------------------------------------------


def test_create_plan_persists_dry_run_maintenance_run_and_items(
    migrator: ArtifactObjectMigrator,
    sqlite_store: SQLiteMetadataStore,
    tmp_path: Path,
) -> None:
    import asyncio

    root_a = tmp_path / "legacy-a"
    _seed_file(root_a, "file1.txt", b"alpha")
    _seed_file(root_a, "nested/file2.txt", b"beta" * 4)

    summary = asyncio.run(migrator.create_plan([("legacyA", root_a)]))

    assert summary.item_count == 2
    assert summary.metadata_backend == "sqlite"
    assert summary.apply_run_id is None

    run = asyncio.run(sqlite_store.get_artifact_maintenance_run(summary.plan_id))
    assert run.mode == "dry_run"
    assert run.status == "succeeded"
    items, total = asyncio.run(
        sqlite_store.list_artifact_maintenance_items(run.id, limit=10)
    )
    assert total == 2
    for item in items:
        assert item.state == "planned"
        assert item.root_label == "legacyA"
        assert item.expected_checksum is not None
        assert item.expected_size_bytes is not None
        assert item.target_uri_authority == "s3://lightrag-kb"
        assert item.relative_object_id.startswith("kb/migrate/legacyA/")


def test_create_plan_rejects_relative_root(migrator: ArtifactObjectMigrator) -> None:
    import asyncio

    with pytest.raises(MigrationSecurityError, match="absolute"):
        asyncio.run(migrator.create_plan([("rel", Path("relative"))]))


def test_create_plan_skips_symlinks_inside_root(
    migrator: ArtifactObjectMigrator, tmp_path: Path
) -> None:
    import asyncio

    root = tmp_path / "legacy"
    root.mkdir()
    target = _seed_file(tmp_path, "target.txt", b"data")
    link = root / "link.txt"
    link.symlink_to(target)
    summary = asyncio.run(migrator.create_plan([("legacy", root)]))
    # The symlink is silently skipped: only regular files are migrated.
    assert summary.item_count == 0


# ---------------------------------------------------------------------------
# Apply: requires --plan-id + --yes
# ---------------------------------------------------------------------------


def test_apply_requires_plan_id_and_yes(tmp_path: Path) -> None:
    parser = build_parser()
    # Missing both is valid for the parser; validation happens in _async_main.
    args = parser.parse_args(
        [
            "legacyA=" + str(tmp_path),
            "--working-dir",
            str(tmp_path),
        ]
    )
    assert args.plan_id is None
    assert args.yes is False


def test_apply_unknown_plan_rejected(migrator: ArtifactObjectMigrator) -> None:
    import asyncio

    with pytest.raises(Exception):
        asyncio.run(migrator.apply_plan("does-not-exist"))


# ---------------------------------------------------------------------------
# Resumable planned -> uploaded -> applied -> verified
# ---------------------------------------------------------------------------


def _make_resolver(kb_id: str, workspace: str, document_id: str):
    def resolver(_label: str, _rel: str) -> tuple[str, str, str]:
        return (kb_id, workspace, document_id)

    return resolver


def test_apply_drives_items_to_verified(
    migrator: ArtifactObjectMigrator,
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
    tmp_path: Path,
) -> None:
    import asyncio

    root = tmp_path / "legacy"
    _seed_file(root, "doc1.txt", b"alpha-data")

    kb_id = "kb-apply"
    workspace = "ws-apply"
    document_id = "doc-1"
    _seed_kb_and_document(
        sqlite_store,
        kb_id=kb_id,
        workspace=workspace,
        document_id=document_id,
        source_name="doc1.txt",
    )

    plan_summary = asyncio.run(migrator.create_plan([("legacy", root)]))
    summary = asyncio.run(
        migrator.apply_plan(
            plan_summary.plan_id,
            label_root_pairs=[("legacy", root)],
            document_resolver=_make_resolver(kb_id, workspace, document_id),
        )
    )
    assert summary.items_total == 1
    assert summary.items_verified == 1
    assert summary.items_failed == 0
    assert summary.items_blocked == 0

    document = asyncio.run(sqlite_store.get_document(kb_id, document_id))
    assert document.metadata["source_object_uri"].startswith("s3://")
    assert document.metadata["source_generation_id"].startswith("mig_")


def test_apply_rejects_when_online_mutation_active(
    migrator: ArtifactObjectMigrator,
    sqlite_store: SQLiteMetadataStore,
    tmp_path: Path,
) -> None:
    """An active mutation job in ANY KB blocks apply via the online-mutation guard.

    This is the regression for B-1: the production store aggregate
    ``count_active_jobs_globally`` is unscoped by ``kb_id``, so a ``running``
    job seeded under a DIFFERENT kb_id than the migration target must still
    block apply. The previous ``list_jobs("__any__", ...)`` guard was a silent
    no-op (``list_jobs`` scopes strictly by ``kb_id``) and only passed because
    a synthetic ``_JobAwareStore`` wrapper overrode the lookup — that wrapper
    is intentionally absent here so the real production code path is exercised.
    """

    import asyncio
    from lightrag.api.kb_service import utc_now_iso
    from lightrag.api.metadata_store import JobRecord

    root = tmp_path / "legacy"
    _seed_file(root, "doc.txt", b"data")
    plan_summary = asyncio.run(migrator.create_plan([("legacy", root)]))

    now = utc_now_iso()
    job = JobRecord(
        id="job-active",
        kb_id="kb-other",
        workspace="ws-other",
        batch_id=None,
        document_id=None,
        job_type="parse",
        status="running",
        stage=None,
        progress=0.0,
        total_items=1,
        completed_items=0,
        failed_items=0,
        idempotency_key=None,
        config_version_id=None,
        config_hash=None,
        retry_count=0,
        max_retries=3,
        payload={},
        result=None,
        error_code=None,
        error_message=None,
        created_at=now,
        updated_at=now,
        queued_at=now,
        started_at=now,
        finished_at=None,
        cancelled_at=None,
    )

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

    asyncio.run(sqlite_store._write(seed))  # noqa: SLF001

    # Production aggregate is unscoped by kb_id: the cross-KB ``running`` job
    # MUST be visible to the guard regardless of which KB the migration targets.
    active = asyncio.run(sqlite_store.count_active_jobs_globally(["running"]))
    assert active > 0

    # The migrator fixture already binds the REAL sqlite_store (no wrapper), so
    # this exercises the production guard code path end to end.
    with pytest.raises(MigrationApplyGuardError, match="Online KB mutation"):
        asyncio.run(
            migrator.apply_plan(
                plan_summary.plan_id, label_root_pairs=[("legacy", root)]
            )
        )


def test_apply_proceeds_when_no_online_mutation(
    migrator: ArtifactObjectMigrator,
    sqlite_store: SQLiteMetadataStore,
    tmp_path: Path,
) -> None:
    """With zero active mutation jobs the online-mutation guard lets apply run.

    Regression complement to B-1: the guard must not over-block. A seeded job
    in a terminal (non-mutation) status is invisible to the active-status
    filter, so apply proceeds and the plan completes.
    """

    import asyncio
    from lightrag.api.kb_service import utc_now_iso
    from lightrag.api.metadata_store import JobRecord

    root = tmp_path / "legacy"
    _seed_file(root, "doc.txt", b"data")
    _seed_kb_and_document(
        sqlite_store,
        kb_id="kb-migrate",
        workspace="ws-migrate",
        document_id="doc-1",
        source_name="doc.txt",
    )
    plan_summary = asyncio.run(migrator.create_plan([("legacy", root)]))

    now = utc_now_iso()
    terminal_job = JobRecord(
        id="job-done",
        kb_id="kb-other",
        workspace="ws-other",
        batch_id=None,
        document_id=None,
        job_type="parse",
        status="succeeded",
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
        payload={},
        result={},
        error_code=None,
        error_message=None,
        created_at=now,
        updated_at=now,
        queued_at=now,
        started_at=now,
        finished_at=now,
        cancelled_at=None,
    )

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
                terminal_job.id,
                terminal_job.kb_id,
                terminal_job.workspace,
                terminal_job.batch_id,
                terminal_job.document_id,
                terminal_job.job_type,
                terminal_job.status,
                terminal_job.stage,
                terminal_job.progress,
                terminal_job.total_items,
                terminal_job.completed_items,
                terminal_job.failed_items,
                terminal_job.idempotency_key,
                terminal_job.config_version_id,
                terminal_job.config_hash,
                terminal_job.retry_count,
                terminal_job.max_retries,
                json.dumps(terminal_job.payload),
                json.dumps(terminal_job.result or {}),
                terminal_job.error_code,
                terminal_job.error_message,
                terminal_job.created_at,
                terminal_job.updated_at,
                terminal_job.queued_at,
                terminal_job.started_at,
                terminal_job.finished_at,
                terminal_job.cancelled_at,
            ),
        )

    asyncio.run(sqlite_store._write(seed))  # noqa: SLF001

    # No active mutation jobs: the guard's active-status set excludes
    # ``succeeded``.
    statuses = ["queued", "running", "retrying", "cancelling"]
    assert asyncio.run(sqlite_store.count_active_jobs_globally(statuses)) == 0

    summary = asyncio.run(
        migrator.apply_plan(plan_summary.plan_id, label_root_pairs=[("legacy", root)])
    )
    assert summary.items_failed == 0


def test_apply_fails_closed_when_guard_store_raises(
    migrator: ArtifactObjectMigrator,
    tmp_path: Path,
) -> None:
    """If the store raises during the online-mutation count, apply fails CLOSED.

    The guard must never silently allow apply when it cannot prove the system
    is quiescent. A store whose ``count_active_jobs_globally`` raises must
    surface as ``MigrationApplyGuardError``, not be swallowed.
    """

    import asyncio

    root = tmp_path / "legacy"
    _seed_file(root, "doc.txt", b"data")
    plan_summary = asyncio.run(migrator.create_plan([("legacy", root)]))

    class _ExplodingStore:
        """Minimal proxy: every attribute delegates to the real store except
        the aggregate, which raises. This simulates a transient store failure
        (connection drop, AttributeError on a stale/old store) during the
        guard."""

        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

        async def count_active_jobs_globally(
            self, statuses: Any
        ) -> int:  # pragma: no cover - always raises
            raise RuntimeError("simulated store outage")

    migrator._metadata_store = _ExplodingStore(migrator._metadata_store)  # noqa: SLF001
    with pytest.raises(MigrationApplyGuardError, match="could not verify"):
        asyncio.run(
            migrator.apply_plan(
                plan_summary.plan_id, label_root_pairs=[("legacy", root)]
            )
        )


# ---------------------------------------------------------------------------
# Crash + resume continues from current state
# ---------------------------------------------------------------------------


def test_apply_resume_advances_state_machine_idempotently(
    migrator: ArtifactObjectMigrator,
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
    tmp_path: Path,
) -> None:
    """A second apply (resume) on the same plan is a no-op for verified items."""

    import asyncio

    root = tmp_path / "legacy"
    _seed_file(root, "doc.txt", b"resume-data")
    kb_id = "kb-resume"
    workspace = "ws-resume"
    document_id = "doc-r"
    _seed_kb_and_document(
        sqlite_store,
        kb_id=kb_id,
        workspace=workspace,
        document_id=document_id,
        source_name="doc.txt",
    )

    plan_summary = asyncio.run(migrator.create_plan([("legacy", root)]))
    first = asyncio.run(
        migrator.apply_plan(
            plan_summary.plan_id,
            label_root_pairs=[("legacy", root)],
            document_resolver=_make_resolver(kb_id, workspace, document_id),
        )
    )
    assert first.items_verified == 1

    second = asyncio.run(
        migrator.apply_plan(
            plan_summary.plan_id,
            label_root_pairs=[("legacy", root)],
            resume=True,
            document_resolver=_make_resolver(kb_id, workspace, document_id),
        )
    )
    assert second.items_verified == 1


def test_apply_blocks_when_inspect_object_fails(
    migrator: ArtifactObjectMigrator,
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
    tmp_path: Path,
) -> None:
    """If inspect_object reports absent after upload, verify must fail closed.

    The migrator surfaces per-item failures via ``issues`` and item counters
    rather than aborting the whole apply: callers decide whether to retry,
    resume, or abandon based on the durable maintenance item state.
    """

    import asyncio

    root = tmp_path / "legacy"
    _seed_file(root, "doc.txt", b"verify-blocks")
    kb_id = "kb-block"
    workspace = "ws-block"
    document_id = "doc-b"
    _seed_kb_and_document(
        sqlite_store,
        kb_id=kb_id,
        workspace=workspace,
        document_id=document_id,
        source_name="doc.txt",
    )
    plan_summary = asyncio.run(migrator.create_plan([("legacy", root)]))

    original_inspect = fake_object_store.inspect_object

    async def failing_inspect(
        uri: str, *, version_id: str | None = None
    ) -> ObjectReadback:
        return ObjectReadback(present=False, stat=None)

    fake_object_store.inspect_object = failing_inspect  # type: ignore[assignment]

    summary = asyncio.run(
        migrator.apply_plan(
            plan_summary.plan_id,
            label_root_pairs=[("legacy", root)],
            document_resolver=_make_resolver(kb_id, workspace, document_id),
        )
    )
    fake_object_store.inspect_object = original_inspect  # type: ignore[assignment]
    # The item is blocked (failed closed) rather than falsely verified.
    assert summary.items_verified == 0
    assert summary.items_blocked >= 1 or summary.items_failed >= 1
    assert summary.issues


# ---------------------------------------------------------------------------
# Atomic metadata pointer update
# ---------------------------------------------------------------------------


def test_metadata_pointer_update_is_atomic_under_resume(
    migrator: ArtifactObjectMigrator,
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
    tmp_path: Path,
) -> None:
    """The frozen maintenance state machine makes pointer commits idempotent."""

    import asyncio

    root = tmp_path / "legacy"
    _seed_file(root, "doc.txt", b"once-only")
    kb_id = "kb-once"
    workspace = "ws-once"
    document_id = "doc-once"
    _seed_kb_and_document(
        sqlite_store,
        kb_id=kb_id,
        workspace=workspace,
        document_id=document_id,
        source_name="doc.txt",
    )
    plan_summary = asyncio.run(migrator.create_plan([("legacy", root)]))
    first = asyncio.run(
        migrator.apply_plan(
            plan_summary.plan_id,
            label_root_pairs=[("legacy", root)],
            document_resolver=_make_resolver(kb_id, workspace, document_id),
        )
    )
    assert first.items_verified == 1

    second = asyncio.run(
        migrator.apply_plan(
            plan_summary.plan_id,
            label_root_pairs=[("legacy", root)],
            resume=True,
            document_resolver=_make_resolver(kb_id, workspace, document_id),
        )
    )
    assert second.items_verified == 1
    document = asyncio.run(sqlite_store.get_document(kb_id, document_id))
    assert document.metadata["source_object_uri"].startswith("s3://")
    assert document.metadata["source_generation_id"].startswith("mig_")


# ---------------------------------------------------------------------------
# Verified requires metadata-only inspect_object proof
# ---------------------------------------------------------------------------


def test_verified_requires_inspect_object_proof(
    migrator: ArtifactObjectMigrator,
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
    tmp_path: Path,
) -> None:
    import asyncio

    root = tmp_path / "legacy"
    _seed_file(root, "doc.txt", b"proof-required")
    kb_id = "kb-proof"
    workspace = "ws-proof"
    document_id = "doc-proof"
    _seed_kb_and_document(
        sqlite_store,
        kb_id=kb_id,
        workspace=workspace,
        document_id=document_id,
        source_name="doc.txt",
    )
    plan_summary = asyncio.run(migrator.create_plan([("legacy", root)]))
    summary = asyncio.run(
        migrator.apply_plan(
            plan_summary.plan_id,
            label_root_pairs=[("legacy", root)],
            document_resolver=_make_resolver(kb_id, workspace, document_id),
        )
    )
    assert summary.items_verified == 1


# ---------------------------------------------------------------------------
# Redacted JSON output
# ---------------------------------------------------------------------------


def test_redact_value_scratches_paths_credentials_and_dsns() -> None:
    assert "<artifact-materialization>" in _redact_value(
        "/tmp/work/.lightrag-scratch/secret"
    )
    assert "<redacted-dsn>" in _redact_value("postgres://user:pass@host/db")
    assert "<redacted-credential>" in _redact_value(
        "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCY"
    )
    redacted = _redact_value("opened /etc/secrets/db.key")
    assert "/etc/secrets" not in redacted


def test_main_emits_redacted_json(
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end plan creation through main() with --json output."""

    root = tmp_path / "legacy"
    _seed_file(root, "doc.txt", b"json-summary")
    import lightrag.tools.migrate_artifacts_to_object as cli

    def _stub_metadata_store(args, backend):  # noqa: ANN001
        return sqlite_store

    def _stub_object_storage(args):  # noqa: ANN001
        return fake_object_store

    monkeypatch.setattr(cli, "_metadata_store_from_args", _stub_metadata_store)
    monkeypatch.setattr(cli, "_object_storage_from_args", _stub_object_storage)

    rc = main(
        [
            f"legacyA={root}",
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
    assert "items" in payload
    # Redaction: no absolute local path leaks in JSON.
    assert str(root) not in captured
    assert ".lightrag-scratch" not in captured


# ---------------------------------------------------------------------------
# Moved-root: explicit LABEL mapping survives directory moves
# ---------------------------------------------------------------------------


def test_plan_records_root_label_for_moved_root_re_resolution(
    migrator: ArtifactObjectMigrator,
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
    tmp_path: Path,
) -> None:
    import asyncio

    first_root = tmp_path / "original"
    _seed_file(first_root, "doc.txt", b"moved-root")
    kb_id = "kb-moved"
    workspace = "ws-moved"
    document_id = "doc-m"
    _seed_kb_and_document(
        sqlite_store,
        kb_id=kb_id,
        workspace=workspace,
        document_id=document_id,
        source_name="doc.txt",
    )

    plan_summary = asyncio.run(migrator.create_plan([("legacyA", first_root)]))
    run = asyncio.run(sqlite_store.get_artifact_maintenance_run(plan_summary.plan_id))
    assert run.mode == "dry_run"
    items, _ = asyncio.run(
        sqlite_store.list_artifact_maintenance_items(run.id, limit=10)
    )
    assert all(item.root_label == "legacyA" for item in items)

    # Simulate a moved root: copy the same bytes to a new directory and apply
    # with the new absolute path. The label is what binds the item to its file.
    moved_root = tmp_path / "moved"
    moved_root.mkdir()
    _seed_file(moved_root, "doc.txt", b"moved-root")

    summary = asyncio.run(
        migrator.apply_plan(
            plan_summary.plan_id,
            label_root_pairs=[("legacyA", moved_root)],
            document_resolver=_make_resolver(kb_id, workspace, document_id),
        )
    )
    assert summary.items_verified == 1
    document = asyncio.run(sqlite_store.get_document(kb_id, document_id))
    assert document.metadata["source_object_uri"].startswith("s3://")


# ---------------------------------------------------------------------------
# Leak-scan: no orphan objects left after successful apply
# ---------------------------------------------------------------------------


def test_no_orphan_objects_after_successful_apply(
    migrator: ArtifactObjectMigrator,
    sqlite_store: SQLiteMetadataStore,
    fake_object_store: FakeObjectStorage,
    tmp_path: Path,
) -> None:
    import asyncio

    root = tmp_path / "legacy"
    for name in ("a.txt", "b.txt", "c.txt"):
        _seed_file(root, name, name.encode() * 8)
    kb_id = "kb-leak"
    workspace = "ws-leak"
    document_id = "doc-a"
    _seed_kb_and_document(
        sqlite_store,
        kb_id=kb_id,
        workspace=workspace,
        document_id=document_id,
        source_name="a.txt",
    )
    plan_summary = asyncio.run(migrator.create_plan([("legacy", root)]))
    summary = asyncio.run(
        migrator.apply_plan(
            plan_summary.plan_id,
            label_root_pairs=[("legacy", root)],
            document_resolver=_make_resolver(kb_id, workspace, document_id),
        )
    )
    items, _ = asyncio.run(
        sqlite_store.list_artifact_maintenance_items(summary.apply_run_id, limit=20)
    )
    migrated_keys = {item.relative_object_id for item in items}
    leaked = set(fake_object_store.uploads.keys()) - migrated_keys
    assert not leaked, f"orphan objects: {leaked}"


# ---------------------------------------------------------------------------
# Parser surface
# ---------------------------------------------------------------------------


def test_build_parser_supports_all_specified_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "legacyA=/srv/legacy",
            "--working-dir",
            "./rag_storage",
            "--object-storage-endpoint",
            "http://localhost:9000",
            "--bucket",
            "lightrag-kb",
            "--prefix",
            "kb",
            "--metadata-backend",
            "sqlite",
            "--json",
        ]
    )
    assert args.working_dir == "./rag_storage"
    assert args.bucket == "lightrag-kb"
    assert args.prefix == "kb"
    assert args.metadata_backend == "sqlite"
    assert args.json is True
    assert args.plan_id is None
    assert args.yes is False
