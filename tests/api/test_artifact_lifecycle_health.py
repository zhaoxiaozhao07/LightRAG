"""Phase 3.3 Writer Health — ``/health`` ``artifact_lifecycle`` block tests.

These tests verify the *additive sibling* health block introduced in Phase 3.3:

* the block reports bounded indexed aggregates (manifest totals +
  ``oldest_due_at``, non-terminal maintenance runs, migration blockers,
  unresolved commit-unknown jobs, stale recovery cursors);
* the object-store readiness probe is a **cached, bounded HeadBucket** (one
  ``head_bucket`` per TTL window) and never raises;
* every probe collapses to ``"not_reported"`` (or ``False`` for the readiness
  probe) on timeout or error so /health latency stays bounded;
* the block performs **NO bucket listing and NO object download** — it is
  aggregates + a single HeadBucket only.

Store-level regression coverage for the three new additive aggregates
(``oldest_due_at``, ``count_unresolved_commit_unknown_jobs``,
``count_stale_artifact_recovery_cursors``) runs against the real SQLite
backend; the block-level coverage uses lightweight fakes for determinism.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from lightrag.api.metadata_store import SQLiteMetadataStore
from lightrag.api.object_storage import DisabledObjectStorage, ObjectStorageConfig
from lightrag.utils_pipeline import reset_canonical_input_root_for_tests

pytestmark = pytest.mark.offline

_NOW = datetime(2026, 8, 4, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# SQLite store-level regression: additive aggregates.
# ---------------------------------------------------------------------------


def _make_sqlite_store(tmp_path: Path) -> SQLiteMetadataStore:
    return SQLiteMetadataStore(tmp_path / "health.sqlite3")


async def _seed_manifest(
    store: SQLiteMetadataStore,
    manifest_id: str,
    *,
    status: str,
    delete_after: datetime,
    kb_id: str = "kb_health",
) -> None:
    from lightrag.api.artifact_lifecycle import (
        ArtifactCleanupManifestRecord,
        artifact_cleanup_idempotency_key,
    )

    target_uri = f"s3://artifact-bucket/kb/ws/source/{manifest_id}"
    disposition = "retain" if status == "retained" else "delete"
    # ``blocked`` manifests require a safe error code at the lifecycle layer.
    last_error_code = "artifact_cleanup_blocked" if status == "blocked" else None
    key = artifact_cleanup_idempotency_key(
        reason="replace",
        kb_id=kb_id,
        kb_generation="generation-1",
        workspace="workspace-1",
        document_id="document-1",
        artifact_id=None,
        source_generation_id="source-generation-1",
        target_kind="object",
        target_namespace="source",
        target_uri=target_uri,
    )
    manifest = ArtifactCleanupManifestRecord(
        id=manifest_id,
        idempotency_key=key,
        manifest_group_id="manifest-group-1",
        kb_id=kb_id,
        kb_generation="generation-1",
        workspace="workspace-1",
        document_id="document-1",
        artifact_id=None,
        source_generation_id="source-generation-1",
        origin_job_id="job-1",
        origin_attempt_token="attempt-1",
        reason="replace",
        target_kind="object",
        target_namespace="source",
        disposition=disposition,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        target_uri=target_uri,
        expected_checksum="sha256:abc123",
        expected_etag="etag-1",
        expected_version_id="version-1",
        expected_size_bytes=123,
        delete_after=delete_after,
        cleanup_deadline_at=delete_after + timedelta(days=1),
        audit_retain_until=delete_after + timedelta(days=30),
        next_attempt_at=delete_after,
        attempt_count=0,
        created_at=delete_after,
        updated_at=delete_after,
        last_error_code=last_error_code,
    )
    await store.enqueue_artifact_cleanup_manifest(manifest)


async def test_aggregate_reports_oldest_due_at_additively(tmp_path: Path) -> None:
    """``oldest_due_at`` is the MIN(delete_after) among pending/leased rows.

    Additive: the existing integer keys are unchanged; the new key is a
    timestamp string (or None when no pending/leased row exists).
    """

    store = _make_sqlite_store(tmp_path)
    await store.initialize()
    try:
        # Two pending manifests with different due times + one retained
        # (retained must NOT contribute to oldest_due_at).
        await _seed_manifest(
            store,
            "manifest_due_later",
            status="pending",
            delete_after=_NOW + timedelta(hours=2),
        )
        await _seed_manifest(
            store,
            "manifest_due_sooner",
            status="pending",
            delete_after=_NOW + timedelta(hours=1),
        )
        await _seed_manifest(
            store,
            "manifest_retained",
            status="retained",
            delete_after=_NOW + timedelta(minutes=5),
        )

        aggregate = await store.aggregate_artifact_cleanup_manifests(
            kb_id="kb_health", now=_NOW
        )
        # Existing integer assertions still hold.
        assert aggregate["total"] == 3
        assert aggregate["pending"] == 2
        assert aggregate["retained"] == 1
        # Additive oldest_due_at == the earliest pending/leased delete_after.
        oldest = aggregate["oldest_due_at"]
        assert oldest is not None
        assert str(oldest).startswith("2026-08-04T13:00:00")
    finally:
        await store.close()


async def test_aggregate_oldest_due_at_none_when_no_pending_or_leased(
    tmp_path: Path,
) -> None:
    store = _make_sqlite_store(tmp_path)
    await store.initialize()
    try:
        # A single retained manifest does NOT contribute to oldest_due_at
        # (only pending/leased rows do), so the MIN is None.
        await _seed_manifest(
            store,
            "manifest_retained_only",
            status="retained",
            delete_after=_NOW,
        )
        aggregate = await store.aggregate_artifact_cleanup_manifests(
            kb_id="kb_health", now=_NOW
        )
        assert aggregate["oldest_due_at"] is None
    finally:
        await store.close()


async def test_count_unresolved_commit_unknown_jobs_is_bounded(tmp_path: Path) -> None:
    """The count matches exactly the jobs carrying the commit-unknown sentinel."""

    from tests.api.test_metadata_store_contract import _doc, _job

    store = _make_sqlite_store(tmp_path)
    await store.initialize()
    kb_id = "kb_health_jobs"
    try:
        await store.create_documents_and_job(
            [_doc(kb_id, "doc_unknown_a")],
            _job(kb_id, "job_unknown_a", document_id="doc_unknown_a"),
        )
        await store.transition_job(kb_id, "job_unknown_a", status="running")
        await store.transition_job(
            kb_id,
            "job_unknown_a",
            status="failed",
            error_code="metadata_commit_outcome_unknown",
        )
        await store.create_documents_and_job(
            [_doc(kb_id, "doc_unknown_b")],
            _job(kb_id, "job_unknown_b", document_id="doc_unknown_b"),
        )
        await store.transition_job(kb_id, "job_unknown_b", status="running")
        await store.transition_job(
            kb_id,
            "job_unknown_b",
            status="failed",
            error_code="metadata_commit_outcome_unknown",
        )
        # A plain failure with a different error code is NOT counted.
        await store.create_documents_and_job(
            [_doc(kb_id, "doc_other")],
            _job(kb_id, "job_other", document_id="doc_other"),
        )
        await store.transition_job(kb_id, "job_other", status="running")
        await store.transition_job(
            kb_id, "job_other", status="failed", error_code="build_failed"
        )

        count = await store.count_unresolved_commit_unknown_jobs()
        assert count == 2
    finally:
        await store.close()


async def test_count_stale_artifact_recovery_cursors_is_bounded(tmp_path: Path) -> None:
    """Cursors older than the cutoff are counted; fresh ones are not."""

    import sqlite3

    store = _make_sqlite_store(tmp_path)
    await store.initialize()
    try:
        stale_iso = (_NOW - timedelta(hours=12)).isoformat()
        fresh_iso = (_NOW - timedelta(minutes=5)).isoformat()
        # Insert two cursor rows directly (the public reserve path requires a
        # KB lifecycle; the count method itself only needs the table row).
        db_path = store.db_path
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                "INSERT INTO artifact_recovery_cursors "
                "(kb_id, kb_generation, status, last_created_at, "
                " last_document_id, sweep, version, updated_at) "
                "VALUES (?, ?, 'parsed', NULL, NULL, 0, 1, ?)",
                ("kb_stale", "gen-1", stale_iso),
            )
            conn.execute(
                "INSERT INTO artifact_recovery_cursors "
                "(kb_id, kb_generation, status, last_created_at, "
                " last_document_id, sweep, version, updated_at) "
                "VALUES (?, ?, 'parsed', NULL, NULL, 0, 1, ?)",
                ("kb_fresh", "gen-1", fresh_iso),
            )
            conn.commit()
        finally:
            conn.close()

        cutoff = _NOW - timedelta(hours=6)
        count = await store.count_stale_artifact_recovery_cursors(stale_before=cutoff)
        assert count == 1
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Block-level unit tests: fake store + fake object storage.
# ---------------------------------------------------------------------------


class _RecordingObjectStorage:
    """Object-storage double that records listing/download attempts.

    The health block must NEVER call list/download — it should only invoke the
    readiness probe. This fake records every list/download call so a test can
    assert the block stayed aggregate-only.
    """

    def __init__(self, *, ready: bool = True) -> None:
        self._ready = ready
        self.list_calls: int = 0
        self.download_calls: int = 0
        self.stat_calls: int = 0
        self.probe_calls: int = 0

    async def readiness_probe(self) -> bool:
        self.probe_calls += 1
        return self._ready

    async def list_objects_page(self, *args: Any, **kwargs: Any) -> Any:
        self.list_calls += 1
        raise AssertionError("health block must not list objects")

    async def download_file(self, *args: Any, **kwargs: Any) -> None:
        self.download_calls += 1
        raise AssertionError("health block must not download objects")

    async def download_prefix(self, *args: Any, **kwargs: Any) -> int:
        self.download_calls += 1
        raise AssertionError("health block must not download prefixes")

    async def stat_object(self, *args: Any, **kwargs: Any) -> Any:
        self.stat_calls += 1
        raise AssertionError("health block must not stat objects")


class _FakeMetadataStore:
    """Metadata-store double returning controlled aggregate values."""

    def __init__(
        self,
        *,
        manifest_aggregate: dict[str, Any] | None = None,
        non_terminal_runs: int = 0,
        migration_blockers: int = 0,
        unresolved_commit_unknown: int = 0,
        stale_cursors: int = 0,
    ) -> None:
        self._manifest_aggregate = manifest_aggregate or {
            "total": 5,
            "retained": 1,
            "pending": 2,
            "leased": 1,
            "blocked": 0,
            "succeeded": 1,
            "due_pending": 1,
            "expired_leases": 0,
            "cleanup_deadline_overdue": 0,
            "oldest_due_at": "2026-08-04T13:00:00+00:00",
        }
        self._non_terminal_runs = non_terminal_runs
        self._migration_blockers = migration_blockers
        self._unresolved_commit_unknown = unresolved_commit_unknown
        self._stale_cursors = stale_cursors
        self.aggregate_calls: int = 0
        self.list_runs_calls: int = 0
        self.count_active_calls: int = 0
        self.count_unknown_calls: int = 0
        self.count_stale_calls: int = 0

    async def aggregate_artifact_cleanup_manifests(
        self, *, now: Any = None
    ) -> dict[str, Any]:
        self.aggregate_calls += 1
        return dict(self._manifest_aggregate)

    async def list_artifact_maintenance_runs(
        self, *, status: str | None = None, limit: int = 100
    ) -> tuple[list[Any], int]:
        self.list_runs_calls += 1
        # Report the configured non-terminal count once (for ``planned``) and
        # 0 for the other non-terminal statuses, so the helper's per-status
        # sum equals the configured value rather than triple-counting it.
        total = self._non_terminal_runs if status == "planned" else 0
        return [], total

    async def count_active_jobs_globally(self, statuses: Any) -> int:
        self.count_active_calls += 1
        return self._migration_blockers

    async def count_unresolved_commit_unknown_jobs(self) -> int:
        self.count_unknown_calls += 1
        return self._unresolved_commit_unknown

    async def count_stale_artifact_recovery_cursors(self, *, stale_before: Any) -> int:
        self.count_stale_calls += 1
        return self._stale_cursors


async def test_build_block_reports_full_aggregate_shape() -> None:
    """The block surfaces every documented field with correct values."""

    from lightrag.api.lightrag_server import _build_artifact_lifecycle_health_block

    object_storage = _RecordingObjectStorage(ready=True)
    store = _FakeMetadataStore(
        non_terminal_runs=3,
        migration_blockers=2,
        unresolved_commit_unknown=1,
        stale_cursors=4,
    )
    block = await _build_artifact_lifecycle_health_block(
        artifact_storage_mode="object",
        object_storage=object_storage,
        metadata_store=store,
        capability_implemented=False,
        admission_allows_object_mode=False,
        now=_NOW,
    )

    assert block["mode"] == "object"
    assert block["backend"] == "_RecordingObjectStorage"
    assert block["capability_admitted"] == {
        "implemented": False,
        "admission_gate_allows_object_mode": False,
    }
    assert block["object_store_ready"] is True

    manifests = block["manifests"]
    assert manifests["total"] == 5
    assert manifests["pending"] == 2
    assert manifests["oldest_due_at"] == "2026-08-04T13:00:00+00:00"

    assert block["maintenance_runs"] == 3
    assert block["migration_blockers"] == 2
    assert block["unresolved_commit_unknown"] == 1
    assert block["recovery_cursor_stale"] == 4

    # The block queried each bounded probe exactly the expected number of
    # times (3 non-terminal statuses -> 3 list calls).
    assert store.aggregate_calls == 1
    assert store.list_runs_calls == 3
    assert store.count_active_calls == 1
    assert store.count_unknown_calls == 1
    assert store.count_stale_calls == 1
    assert object_storage.probe_calls == 1


async def test_build_block_performs_no_listing_or_download() -> None:
    """The block is aggregates + HeadBucket only: never lists/downloads/stats."""

    from lightrag.api.lightrag_server import _build_artifact_lifecycle_health_block

    object_storage = _RecordingObjectStorage(ready=True)
    store = _FakeMetadataStore()
    await _build_artifact_lifecycle_health_block(
        artifact_storage_mode="object",
        object_storage=object_storage,
        metadata_store=store,
        capability_implemented=False,
        admission_allows_object_mode=False,
        now=_NOW,
    )
    assert object_storage.list_calls == 0
    assert object_storage.download_calls == 0
    assert object_storage.stat_calls == 0


async def test_build_block_falls_back_on_store_timeout() -> None:
    """A store probe slower than the health timeout collapses to not_reported."""

    from lightrag.api.lightrag_server import _build_artifact_lifecycle_health_block

    class _SlowStore(_FakeMetadataStore):
        async def aggregate_artifact_cleanup_manifests(
            self, *, now: Any = None
        ) -> dict[str, Any]:
            await asyncio.sleep(5.0)
            return {}  # pragma: no cover - never reached

    object_storage = _RecordingObjectStorage(ready=True)
    block = await _build_artifact_lifecycle_health_block(
        artifact_storage_mode="object",
        object_storage=object_storage,
        metadata_store=_SlowStore(),
        capability_implemented=False,
        admission_allows_object_mode=False,
        now=_NOW,
    )
    assert block["manifests"] == "not_reported"
    # The other (fast) probes still report their values.
    assert isinstance(block["maintenance_runs"], int)
    assert block["object_store_ready"] is True


async def test_build_block_falls_back_on_store_error() -> None:
    """A raising store probe collapses to not_reported (never propagates)."""

    from lightrag.api.lightrag_server import _build_artifact_lifecycle_health_block

    class _ExplodingStore(_FakeMetadataStore):
        async def count_unresolved_commit_unknown_jobs(self) -> int:
            raise RuntimeError("backend unavailable")

        async def count_active_jobs_globally(self, statuses: Any) -> int:
            raise ValueError("bad statuses")

    block = await _build_artifact_lifecycle_health_block(
        artifact_storage_mode="object",
        object_storage=_RecordingObjectStorage(ready=True),
        metadata_store=_ExplodingStore(),
        capability_implemented=False,
        admission_allows_object_mode=False,
        now=_NOW,
    )
    assert block["unresolved_commit_unknown"] == "not_reported"
    assert block["migration_blockers"] == "not_reported"


async def test_build_block_not_reported_when_methods_absent() -> None:
    """An older/fake store without the new methods reports not_reported."""

    from lightrag.api.lightrag_server import _build_artifact_lifecycle_health_block

    # A plain object exposes none of the aggregate methods.
    block = await _build_artifact_lifecycle_health_block(
        artifact_storage_mode="local",
        object_storage=DisabledObjectStorage(),
        metadata_store=object(),
        capability_implemented=False,
        admission_allows_object_mode=False,
        now=_NOW,
    )
    assert block["mode"] == "local"
    assert block["backend"] == "disabled"
    assert block["object_store_ready"] is False
    assert block["manifests"] == "not_reported"
    assert block["maintenance_runs"] == "not_reported"
    assert block["migration_blockers"] == "not_reported"
    assert block["unresolved_commit_unknown"] == "not_reported"
    assert block["recovery_cursor_stale"] == "not_reported"


async def test_build_block_object_store_ready_false_on_missing_probe() -> None:
    """When the object storage lacks readiness_probe, the block reports False."""

    from lightrag.api.lightrag_server import _build_artifact_lifecycle_health_block

    class _NoProbeObjectStorage:
        pass

    block = await _build_artifact_lifecycle_health_block(
        artifact_storage_mode="object",
        object_storage=_NoProbeObjectStorage(),
        metadata_store=_FakeMetadataStore(),
        capability_implemented=False,
        admission_allows_object_mode=False,
        now=_NOW,
    )
    assert block["object_store_ready"] is False


# ---------------------------------------------------------------------------
# S3 readiness_probe caching (real S3ObjectStorage + fake session).
# ---------------------------------------------------------------------------


def _make_s3_with_fake_session() -> tuple[Any, Any]:
    from lightrag.api.object_storage import S3ObjectStorage

    class _FakeClient:
        def __init__(self, state: dict[str, Any]) -> None:
            self._state = state

        async def head_bucket(self, *, Bucket: str) -> None:
            self._state["head_bucket_calls"] += 1
            if self._state.get("raise"):
                raise RuntimeError("head_bucket failed")

    class _FakeSession:
        def __init__(self, state: dict[str, Any]) -> None:
            self._state = state

        def client(self, service: str, **kwargs: Any):  # noqa: ANN201
            del service, kwargs
            return _FakeCM(self._state)

    class _FakeCM:
        def __init__(self, state: dict[str, Any]) -> None:
            self._state = state

        async def __aenter__(self) -> _FakeClient:
            return _FakeClient(self._state)

        async def __aexit__(self, *exc: Any) -> None:
            return None

    config = ObjectStorageConfig(
        backend="minio",
        bucket="health-bucket",
        endpoint_url="http://fake:9000",
        access_key_id="admin",
        secret_access_key="admin123",
        region_name="us-east-1",
        prefix="kb",
        use_ssl=False,
        create_bucket=False,
    )
    state: dict[str, Any] = {"head_bucket_calls": 0, "raise": False}
    storage = S3ObjectStorage(config)
    storage._new_session = lambda: _FakeSession(state)  # type: ignore[method-assign]
    return storage, state


async def test_s3_readiness_probe_caches_within_ttl() -> None:
    storage, state = _make_s3_with_fake_session()
    # Two probes back-to-back share one HeadBucket (TTL cache).
    first = await storage.readiness_probe()
    second = await storage.readiness_probe()
    assert first is True
    assert second is True
    assert state["head_bucket_calls"] == 1


async def test_s3_readiness_probe_refreshes_after_ttl() -> None:
    storage, state = _make_s3_with_fake_session()
    # Force the cached timestamp into the distant past so the next probe
    # performs a fresh HeadBucket.
    storage._readiness_probe_at = float("-inf")  # type: ignore[attr-defined]
    await storage.readiness_probe()
    assert state["head_bucket_calls"] == 1
    # Expire the cache again and probe once more -> second HeadBucket.
    storage._readiness_probe_at = float("-inf")  # type: ignore[attr-defined]
    await storage.readiness_probe()
    assert state["head_bucket_calls"] == 2


async def test_s3_readiness_probe_returns_false_on_error_and_never_raises() -> None:
    storage, state = _make_s3_with_fake_session()
    state["raise"] = True
    result = await storage.readiness_probe()
    assert result is False
    # The error also caches (still only one HeadBucket this TTL window).
    assert state["head_bucket_calls"] == 1


async def test_disabled_object_storage_readiness_probe_is_false() -> None:
    storage = DisabledObjectStorage()
    assert await storage.readiness_probe() is False


# ---------------------------------------------------------------------------
# End-to-end: /health emits the additive sibling block.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_canonical_root() -> Any:
    reset_canonical_input_root_for_tests()
    try:
        yield
    finally:
        reset_canonical_input_root_for_tests()


def _make_server_app(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    object_mode: bool,
) -> Any:
    from lightrag.api import lightrag_server
    from tests.api.test_artifact_storage_foundation import _complete_server_args

    args = _complete_server_args(tmp_path, monkeypatch)
    monkeypatch.setenv("LIGHTRAG_KB_JOB_WORKER", "true")
    monkeypatch.setattr(lightrag_server, "check_frontend_build", lambda: (True, False))

    class _ServerRAG:
        def __init__(self, **kwargs: Any) -> None:
            self.workspace = kwargs["workspace"]
            self.ollama_server_infos = kwargs["ollama_server_infos"]
            self.pipeline_artifact_materializer = None
            self.role_llm_builder = None

        def register_role_llm_builder(self, builder: Any) -> None:
            self.role_llm_builder = builder

        def get_llm_role_config(self) -> dict[str, Any]:
            return {}

        async def aupdate_llm_role_config(self, _role: str, **_kwargs: Any) -> None:
            return None

        async def adrop_all_storages(self) -> dict[str, Any]:
            return {"dropped": 0, "failed": 0, "errors": []}

        async def initialize_storages(self) -> None:
            from lightrag.kg.shared_storage import (
                initialize_pipeline_status,
                initialize_share_data,
                set_default_workspace,
            )

            initialize_share_data()
            set_default_workspace(self.workspace)
            await initialize_pipeline_status(workspace=self.workspace)

        async def check_and_migrate_data(self) -> None:
            return None

        async def finalize_storages(self) -> None:
            return None

        async def get_llm_queue_status(
            self, include_base: bool = False
        ) -> dict[str, Any]:
            return {"include_base": include_base}

        async def get_embedding_queue_status(self) -> dict[str, Any]:
            return {}

        async def get_rerank_queue_status(self) -> dict[str, Any]:
            return {}

    monkeypatch.setattr(lightrag_server, "LightRAG", _ServerRAG)
    if object_mode:
        from tests.api.test_artifact_storage_phase2a import _FakeObjectStorage

        args.artifact_storage_mode = "object"
        monkeypatch.setenv("LIGHTRAG_ARTIFACT_STORAGE_MODE", "object")
        storage = _FakeObjectStorage()
        monkeypatch.setattr(
            lightrag_server, "create_object_storage", lambda _config: storage
        )
        monkeypatch.setattr(
            lightrag_server,
            "validate_artifact_storage_configuration",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            lightrag_server,
            "validate_artifact_storage_server_admission",
            lambda *args, **kwargs: None,
        )
    else:
        args.artifact_storage_mode = "local"
        monkeypatch.setenv("LIGHTRAG_ARTIFACT_STORAGE_MODE", "local")

    return lightrag_server.create_app(args)


def test_health_endpoint_emits_artifact_lifecycle_block_in_object_mode(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """/health exposes ``artifact_lifecycle`` alongside ``artifact_cleanup``."""

    app = _make_server_app(tmp_path, monkeypatch, object_mode=True)
    worker = app.state.job_worker
    assert worker is not None
    worker.start = MagicMock()
    worker.stop = AsyncMock()

    with patch("lightrag.api.lightrag_server.finalize_share_data"):
        with TestClient(app) as client:
            response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()

    # The legacy block is preserved unchanged.
    assert "artifact_cleanup" in payload
    # The additive sibling is present with the documented shape.
    assert "artifact_lifecycle" in payload
    block = payload["artifact_lifecycle"]
    assert block["mode"] == "object"
    assert set(block.keys()) == {
        "mode",
        "backend",
        "capability_admitted",
        "object_store_ready",
        "manifests",
        "maintenance_runs",
        "migration_blockers",
        "unresolved_commit_unknown",
        "recovery_cursor_stale",
    }
    assert block["capability_admitted"]["implemented"] is True
    assert block["capability_admitted"]["admission_gate_allows_object_mode"] is True
    # The real SQLite store exposes every aggregate method, so all probes
    # report concrete values (not "not_reported").
    assert isinstance(block["manifests"], dict)
    assert "oldest_due_at" in block["manifests"]
    assert isinstance(block["maintenance_runs"], int)
    assert isinstance(block["migration_blockers"], int)
    assert isinstance(block["unresolved_commit_unknown"], int)
    assert isinstance(block["recovery_cursor_stale"], int)


def test_health_endpoint_emits_artifact_lifecycle_block_in_local_mode(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In local mode the block reports disabled storage + still-bounded aggregates."""

    app = _make_server_app(tmp_path, monkeypatch, object_mode=False)
    worker = app.state.job_worker
    assert worker is not None
    worker.start = MagicMock()
    worker.stop = AsyncMock()

    with patch("lightrag.api.lightrag_server.finalize_share_data"):
        with TestClient(app) as client:
            response = client.get("/health")
    assert response.status_code == 200
    block = response.json()["artifact_lifecycle"]
    assert block["mode"] == "local"
    assert block["backend"] == "none"
    assert block["object_store_ready"] is False
