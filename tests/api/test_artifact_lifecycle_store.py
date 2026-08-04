from __future__ import annotations

import asyncio
import inspect
import json
import os
import sqlite3
import uuid
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from lightrag.api.artifact_lifecycle import (
    ArtifactCleanupManifestRecord,
    ArtifactLifecycleConflictError,
    ArtifactLifecycleLeaseError,
    ArtifactLifecycleNotFoundError,
    ArtifactLifecycleStateError,
    ArtifactMaintenanceItemRecord,
    ArtifactMaintenanceRunRecord,
    ArtifactRecoveryGenerationError,
    artifact_cleanup_idempotency_key,
    artifact_maintenance_item_key,
    artifact_maintenance_run_key,
    artifact_target_uri_digest,
    normalize_artifact_relative_object_id,
    normalize_artifact_target_uri,
    normalize_artifact_target_uri_authority,
    sanitize_artifact_lifecycle_error_code,
)
from lightrag.api.metadata_store import (
    DocumentRecord,
    JobRecord,
    MetadataRecordNotFoundError,
    SQLiteMetadataStore,
)

pytestmark = pytest.mark.offline

_NOW = datetime(2026, 8, 3, 12, 0, 0, tzinfo=timezone.utc)
_POSTGRES_DSN = os.getenv("LIGHTRAG_KB_POSTGRES_TEST_DSN") or os.getenv(
    "POSTGRES_TEST_DSN"
)


def _manifest(
    manifest_id: str,
    *,
    kb_id: str = "kb_lifecycle",
    kb_generation: str = "generation-1",
    group_id: str = "manifest-group-1",
    document_id: str | None = "document-1",
    artifact_id: str | None = None,
    source_generation_id: str | None = "source-generation-1",
    status: str = "pending",
    disposition: str | None = None,
    target_namespace: str = "source",
    target_suffix: str | None = None,
    delete_after: datetime = _NOW,
    audit_retain_until: datetime = _NOW + timedelta(days=30),
    expected_checksum: str | None = "sha256:abc123",
) -> ArtifactCleanupManifestRecord:
    target_uri = (
        f"s3://artifact-bucket/kb/workspace/source/{target_suffix or manifest_id}"
    )
    resolved_disposition = disposition or (
        "retain" if status == "retained" else "delete"
    )
    key = artifact_cleanup_idempotency_key(
        reason="replace",
        kb_id=kb_id,
        kb_generation=kb_generation,
        workspace="workspace-1",
        document_id=document_id,
        artifact_id=artifact_id,
        source_generation_id=source_generation_id,
        target_kind="object",
        target_namespace=target_namespace,  # type: ignore[arg-type]
        target_uri=target_uri,
    )
    return ArtifactCleanupManifestRecord(
        id=manifest_id,
        idempotency_key=key,
        manifest_group_id=group_id,
        kb_id=kb_id,
        kb_generation=kb_generation,
        workspace="workspace-1",
        document_id=document_id,
        artifact_id=artifact_id,
        source_generation_id=source_generation_id,
        origin_job_id="job-1",
        origin_attempt_token="attempt-1",
        reason="replace",
        target_kind="object",
        target_namespace=target_namespace,  # type: ignore[arg-type]
        disposition=resolved_disposition,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        target_uri=target_uri,
        expected_checksum=expected_checksum,
        expected_etag="etag-1",
        expected_version_id="version-1",
        expected_size_bytes=123,
        delete_after=delete_after,
        cleanup_deadline_at=delete_after + timedelta(days=1),
        audit_retain_until=audit_retain_until,
        next_attempt_at=delete_after,
        attempt_count=0,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _document(
    kb_id: str,
    document_id: str,
    *,
    status: str,
    created_at: str,
) -> DocumentRecord:
    return DocumentRecord(
        id=document_id,
        kb_id=kb_id,
        workspace=f"workspace-{kb_id}",
        lightrag_doc_id=f"doc-{document_id}",
        source_type="upload",
        source_name=f"{document_id}.txt",
        source_uri=f"/legacy-input/{document_id}.txt",
        source_hash=f"sha256:{document_id}",
        content_type="text/plain",
        size_bytes=1,
        parser_hash="parser-1",
        index_hash="index-1" if status == "ready" else None,
        status=status,
        enabled=True,
        archived=False,
        chunks_count=None,
        entity_count=None,
        relation_count=None,
        error_code=None,
        error_message=None,
        metadata={},
        created_at=created_at,
        updated_at=created_at,
        deleted_at=None,
    )


def _job(kb_id: str, job_id: str, total_items: int) -> JobRecord:
    now = _NOW.isoformat()
    return JobRecord(
        id=job_id,
        kb_id=kb_id,
        workspace=f"workspace-{kb_id}",
        batch_id=None,
        document_id=None,
        job_type="reindex",
        status="succeeded",
        stage=None,
        progress=1.0,
        total_items=total_items,
        completed_items=total_items,
        failed_items=0,
        idempotency_key=f"idempotency-{job_id}",
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


async def _assert_exact_manifest_filter_contract(
    store,
    source_manifest: ArtifactCleanupManifestRecord,
    artifact_manifest: ArtifactCleanupManifestRecord,
) -> None:
    assert source_manifest.manifest_group_id == artifact_manifest.manifest_group_id
    base_filters = {
        "kb_id": source_manifest.kb_id,
        "kb_generation": source_manifest.kb_generation,
        "manifest_group_id": source_manifest.manifest_group_id,
    }
    bounded, total = await store.list_artifact_cleanup_manifests(
        **base_filters, limit=1
    )
    assert len(bounded) == 1
    assert total == 2
    assert await store.count_artifact_cleanup_manifests(**base_filters) == total

    assert source_manifest.document_id is not None
    assert source_manifest.source_generation_id is not None
    assert artifact_manifest.artifact_id is not None
    normalized_lookup_uri = source_manifest.target_uri.replace(
        "s3://artifact-bucket", "S3://ARTIFACT-BUCKET", 1
    )
    exact_filters = (
        ({"document_id": source_manifest.document_id}, source_manifest),
        ({"target_uri": normalized_lookup_uri}, source_manifest),
        ({"artifact_id": artifact_manifest.artifact_id}, artifact_manifest),
        (
            {"source_generation_id": source_manifest.source_generation_id},
            source_manifest,
        ),
    )
    for filters, expected in exact_filters:
        records, filtered_total = await store.list_artifact_cleanup_manifests(
            **base_filters, **filters, limit=1
        )
        filtered_count = await store.count_artifact_cleanup_manifests(
            **base_filters, **filters
        )
        assert filtered_total == filtered_count == 1
        assert records == [expected]

    with pytest.raises(ValueError):
        await store.count_artifact_cleanup_manifests(
            **base_filters,
            target_uri=f"{source_manifest.target_uri}?X-Amz-Signature=unsafe",
        )


async def _seed_recovery_documents(
    store,
    *,
    kb_id: str,
    kb_generation: str,
    count: int,
    parsed_count: int,
) -> list[str]:
    await store.activate_kb_generation(kb_id, kb_generation)
    # Deliberately use one exact timestamp without fractional seconds. This
    # proves the keyset tie-breaker is document_id rather than an offset or a
    # timestamp formatting accident.
    created_at = "2026-08-03T01:00:00+00:00"
    documents = [
        _document(
            kb_id,
            f"{kb_id}-document-{index:04d}",
            status="parsed" if index < parsed_count else "ready",
            created_at=created_at,
        )
        for index in range(count)
    ]
    await store.create_documents_and_job(
        documents,
        _job(kb_id, f"job-{kb_generation}-{uuid.uuid4().hex}", count),
    )
    return [document.id for document in documents]


def _dry_run(
    run_id: str,
    *,
    status: str = "planned",
    metadata_backend: str = "sqlite",
) -> ArtifactMaintenanceRunRecord:
    terminal = status in {"succeeded", "failed", "cancelled"}
    return ArtifactMaintenanceRunRecord(
        id=run_id,
        kind="migration",
        mode="dry_run",
        status=status,  # type: ignore[arg-type]
        metadata_backend=metadata_backend,  # type: ignore[arg-type]
        parent_plan_id=None,
        backend_fingerprint="sha256:backend",
        scope_fingerprint="sha256:scope",
        config_fingerprint="sha256:config",
        scope_json={"kb_id": "kb_lifecycle", "object_prefix": "kb/workspace"},
        total_items=0,
        actor_id="operator-1",
        created_at=_NOW,
        updated_at=_NOW,
        completed_at=_NOW if terminal else None,
        last_error_code="maintenance_failed" if status == "failed" else None,
    )


def _apply_run(
    run_id: str,
    parent_plan_id: str,
    *,
    metadata_backend: str = "sqlite",
) -> ArtifactMaintenanceRunRecord:
    return ArtifactMaintenanceRunRecord(
        id=run_id,
        kind="migration",
        mode="apply",
        status="planned",
        metadata_backend=metadata_backend,  # type: ignore[arg-type]
        parent_plan_id=parent_plan_id,
        backend_fingerprint="sha256:backend",
        scope_fingerprint="sha256:scope",
        config_fingerprint="sha256:config",
        scope_json={"kb_id": "kb_lifecycle", "object_prefix": "kb/workspace"},
        actor_id="operator-1",
        created_at=_NOW + timedelta(minutes=1),
        updated_at=_NOW + timedelta(minutes=1),
    )


def _maintenance_item(
    run_id: str,
    *,
    ordinal: int = 0,
    kb_id: str = "kb_lifecycle",
    kb_generation: str = "generation-1",
    workspace: str = "workspace-1",
    document_id: str | None = "document-1",
    artifact_id: str | None = "artifact-1",
    logical_group_id: str = "logical-group-1",
    relative_object_id: str = "source/document-1.txt",
    target_uri: str = "s3://artifact-bucket/kb/workspace/source/document-1.txt",
) -> ArtifactMaintenanceItemRecord:
    payload = {"overwrite": False, "verification_mode": "checksum"}
    target_uri_authority = "s3://artifact-bucket"
    target_uri_digest = artifact_target_uri_digest(target_uri)
    item_key = artifact_maintenance_item_key(
        run_id=run_id,
        subject_kind="document",
        subject_id=document_id or artifact_id or f"item-{ordinal}",
        kb_id=kb_id,
        kb_generation=kb_generation,
        workspace=workspace,
        document_id=document_id,
        artifact_id=artifact_id,
        logical_group_id=logical_group_id,
        relative_object_id=relative_object_id,
        root_label="legacy-source",
        expected_checksum="sha256:a",
        expected_size_bytes=123,
        target_uri_authority=target_uri_authority,
        target_uri_digest=target_uri_digest,
        payload_json=payload,
    )
    return ArtifactMaintenanceItemRecord(
        id=f"item-{run_id}-{ordinal}",
        run_id=run_id,
        item_key=item_key,
        state="planned",
        ordinal=ordinal,
        subject_kind="document",
        subject_id=document_id or artifact_id or f"item-{ordinal}",
        kb_id=kb_id,
        kb_generation=kb_generation,
        workspace=workspace,
        document_id=document_id,
        artifact_id=artifact_id,
        logical_group_id=logical_group_id,
        relative_object_id=relative_object_id,
        root_label="legacy-source",
        expected_checksum="sha256:a",
        expected_size_bytes=123,
        target_uri_authority=target_uri_authority,
        target_uri_digest=target_uri_digest,
        payload_json=payload,
        created_at=_NOW,
        updated_at=_NOW,
    )


def test_record_validation_redaction_immutability_and_idempotency() -> None:
    for unsafe_encoded_uri in (
        "S3://Artifact-Bucket/kb/workspace/source/a%2Fb.txt",
        "s3://artifact-bucket/kb/workspace/source/a%5Cb.txt",
        "s3://artifact-bucket/kb/workspace/source/%2e%2e/file.txt",
        "s3://artifact-bucket/kb/workspace/source/.%2E/file.txt",
    ):
        with pytest.raises(ValueError):
            normalize_artifact_target_uri(unsafe_encoded_uri)

    literal_percent_uri = "S3://Artifact-Bucket/kb/workspace/source/a%252Fb.txt"
    assert normalize_artifact_target_uri(literal_percent_uri) == (
        "s3://artifact-bucket/kb/workspace/source/a%252Fb.txt"
    )
    key = artifact_cleanup_idempotency_key(
        reason="replace",
        kb_id="kb-1",
        kb_generation="generation-1",
        workspace="workspace-1",
        document_id="document-1",
        target_kind="object",
        target_namespace="source",
        target_uri=literal_percent_uri,
    )
    assert len(key) == 64

    sqlite_run_key = artifact_maintenance_run_key(
        kind="migration",
        mode="dry_run",
        metadata_backend="sqlite",
        parent_plan_id=None,
        backend_fingerprint="sha256:backend",
        scope_fingerprint="sha256:scope",
        config_fingerprint="sha256:config",
    )
    postgres_run_key = artifact_maintenance_run_key(
        kind="migration",
        mode="dry_run",
        metadata_backend="postgres",
        parent_plan_id=None,
        backend_fingerprint="sha256:backend",
        scope_fingerprint="sha256:scope",
        config_fingerprint="sha256:config",
    )
    assert sqlite_run_key != postgres_run_key

    record = _manifest("manifest-record")
    assert (
        record.target_uri == "s3://artifact-bucket/kb/workspace/source/manifest-record"
    )
    assert str(record.created_at).endswith(".000000+00:00")
    with pytest.raises(FrozenInstanceError):
        record.status = "blocked"  # type: ignore[misc]

    for unsafe_uri in (
        "s3://access:secret@artifact-bucket/key",
        "s3://user%40artifact-bucket/key",
        "s3://artifact-bucket/key?X-Amz-Signature=secret",
        "s3://artifact-bucket/.lightrag-scratch/op/file",
        "s3://artifact-bucket/AKIAABCDEFGHIJKLMNOP",
        "s3://artifact-bucket/key#fragment",
        "file://localhost/private/object",
    ):
        with pytest.raises(ValueError):
            replace(record, target_uri=unsafe_uri)

    with pytest.raises(ValueError):
        replace(record, idempotency_key="0" * 64)
    with pytest.raises(ValueError):
        replace(record, target_kind="prefix")
    with pytest.raises(ValueError):
        replace(record, status="leased")
    with pytest.raises(ValueError):
        replace(record, created_at="2026-08-03T12:00:00")
    with pytest.raises(ValueError):
        replace(record, expected_size_bytes=-1)
    with pytest.raises(ValueError):
        ArtifactMaintenanceRunRecord(
            id="unsafe-run",
            kind="migration",
            mode="dry_run",
            status="planned",
            metadata_backend="sqlite",
            backend_fingerprint="sha256:b",
            scope_fingerprint="sha256:s",
            config_fingerprint="sha256:c",
            scope_json={"legacy_root": "/private/legacy"},
            created_at=_NOW,
            updated_at=_NOW,
        )

    item = _maintenance_item("validation-run")
    assert normalize_artifact_relative_object_id("a%252Fb/file.txt") == (
        "a%252Fb/file.txt"
    )
    assert normalize_artifact_target_uri_authority("S3://Artifact-Bucket") == (
        "s3://artifact-bucket"
    )
    changed_item_key = artifact_maintenance_item_key(
        run_id=item.run_id,
        subject_kind=item.subject_kind,
        subject_id=item.subject_id,
        kb_id=item.kb_id,
        kb_generation=item.kb_generation,
        workspace=item.workspace,
        document_id="document-2",
        artifact_id=item.artifact_id,
        logical_group_id=item.logical_group_id,
        relative_object_id=item.relative_object_id,
        root_label=item.root_label,
        expected_checksum=item.expected_checksum,
        expected_size_bytes=item.expected_size_bytes,
        target_uri_authority=item.target_uri_authority,
        target_uri_digest=item.target_uri_digest,
        payload_json=item.payload_json,
    )
    assert changed_item_key != item.item_key
    with pytest.raises(ValueError):
        replace(item, document_id="document-2")
    for changes in (
        {"relative_object_id": "/private/object"},
        {"relative_object_id": "../object"},
        {"relative_object_id": "folder//object"},
        {"relative_object_id": "folder\\object"},
        {"relative_object_id": "s3://bucket/object"},
        {"root_label": "/private/root"},
        {"expected_checksum": "/private/checksum"},
        {"target_uri_authority": "s3://bucket/path"},
        {"target_uri_authority": "s3://user@bucket"},
        {"target_uri_digest": "A" * 64},
        {"target_uri_digest": "s3://bucket/object"},
        {"kb_id": "s3://bucket/object"},
    ):
        with pytest.raises(ValueError):
            replace(item, **changes)
    for payload in (
        {"target_uri": "s3://bucket/private/object"},
        {"object_key": "kb/workspace/private/object"},
        {"local_path": "/private/object"},
        {"dsn": "postgresql://user:password@db/private"},
        {"content": "private document bytes"},
        {"password": "must-not-persist"},
    ):
        with pytest.raises(ValueError):
            replace(item, payload_json=payload)
    with pytest.raises(ValueError):
        replace(
            item,
            state="blocked",
            last_error_code="maintenance_blocked",
            completed_at=None,
        )

    assert (
        sanitize_artifact_lifecycle_error_code("AWS_SECRET_ACCESS_KEY=must-not-persist")
        == "artifact_lifecycle_error"
    )
    conflict = ArtifactLifecycleConflictError("artifact cleanup manifest")
    assert "secret" not in str(conflict).lower()
    assert "/" not in str(conflict)
    redacted_conflict = ArtifactLifecycleConflictError(
        "AWS_SECRET_ACCESS_KEY=must-not-persist"
    )
    assert "secret" not in str(redacted_conflict).lower()


async def test_sqlite_schema_round_trip_and_existing_schema_initialize(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "metadata.sqlite3"
    store = SQLiteMetadataStore(db_path)
    await store.initialize()
    manifest = _manifest("manifest-round-trip")
    assert await store.enqueue_artifact_cleanup_manifest(manifest) == manifest
    await store.close()

    # Simulate an existing Phase 2 schema that has not yet received lifecycle
    # tables, then prove repeated initialization converges idempotently.
    with sqlite3.connect(db_path) as conn:
        for table in (
            "artifact_maintenance_items",
            "artifact_maintenance_runs",
            "artifact_recovery_cursors",
            "artifact_cleanup_manifests",
        ):
            conn.execute(f"DROP TABLE {table}")
        conn.execute("DELETE FROM metadata_schema WHERE version = 13")
        conn.commit()

    upgraded = SQLiteMetadataStore(db_path)
    await upgraded.initialize()
    await upgraded.initialize()
    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {
            "artifact_cleanup_manifests",
            "artifact_maintenance_runs",
            "artifact_maintenance_items",
            "artifact_recovery_cursors",
        } <= tables
        assert (
            conn.execute(
                "PRAGMA foreign_key_list(artifact_cleanup_manifests)"
            ).fetchall()
            == []
        )
        manifest_indexes = {
            row[1]
            for row in conn.execute("PRAGMA index_list(artifact_cleanup_manifests)")
        }
        assert {
            "idx_artifact_cleanup_manifest_due",
            "idx_artifact_cleanup_manifest_status",
            "idx_artifact_cleanup_manifest_kb",
            "idx_artifact_cleanup_manifest_group_lookup",
        } <= manifest_indexes
        run_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(artifact_maintenance_runs)")
        }
        item_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(artifact_maintenance_items)")
        }
        assert "metadata_backend" in run_columns
        assert {
            "kb_id",
            "kb_generation",
            "workspace",
            "document_id",
            "artifact_id",
            "logical_group_id",
            "relative_object_id",
            "root_label",
            "expected_checksum",
            "expected_size_bytes",
            "target_uri_authority",
            "target_uri_digest",
        } <= item_columns
        run_indexes = {
            row[1]
            for row in conn.execute("PRAGMA index_list(artifact_maintenance_runs)")
        }
        item_indexes = {
            row[1]
            for row in conn.execute("PRAGMA index_list(artifact_maintenance_items)")
        }
        assert "idx_artifact_maintenance_run_backend" in run_indexes
        assert {
            "idx_artifact_maintenance_item_run_state",
            "idx_artifact_maintenance_item_kb",
            "idx_artifact_maintenance_item_document",
            "idx_artifact_maintenance_item_artifact",
            "idx_artifact_maintenance_item_group",
            "idx_artifact_maintenance_item_uri_digest",
        } <= item_indexes
        versions = {
            row[0] for row in conn.execute("SELECT version FROM metadata_schema")
        }
        assert 13 in versions

    second = _manifest("manifest-after-upgrade", target_suffix="after-upgrade")
    assert await upgraded.enqueue_artifact_cleanup_manifest(second) == second
    assert await upgraded.get_artifact_cleanup_manifest(second.id) == second


async def test_sqlite_fix18_maintenance_schema_upgrade_is_safe_and_idempotent(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "metadata.sqlite3"
    initial = SQLiteMetadataStore(db_path)
    await initial.initialize()
    await initial.close()

    legacy_run_id = "legacy-maintenance-run"
    legacy_item_key = "b" * 64
    timestamp = _NOW.isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            DROP TABLE artifact_maintenance_items;
            DROP TABLE artifact_maintenance_runs;
            CREATE TABLE artifact_maintenance_runs (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL CHECK (kind IN ('migration', 'orphan_reconcile')),
                mode TEXT NOT NULL CHECK (mode IN ('dry_run', 'apply')),
                status TEXT NOT NULL CHECK (status IN (
                    'planned', 'running', 'waiting_cleanup', 'succeeded',
                    'failed', 'cancelled'
                )),
                backend_fingerprint TEXT NOT NULL,
                scope_fingerprint TEXT NOT NULL,
                config_fingerprint TEXT NOT NULL,
                scope_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                parent_plan_id TEXT,
                idempotency_key TEXT NOT NULL UNIQUE,
                cursor_json TEXT,
                total_items INTEGER NOT NULL DEFAULT 0,
                planned_items INTEGER NOT NULL DEFAULT 0,
                uploaded_items INTEGER NOT NULL DEFAULT 0,
                applied_items INTEGER NOT NULL DEFAULT 0,
                verified_items INTEGER NOT NULL DEFAULT 0,
                skipped_items INTEGER NOT NULL DEFAULT 0,
                blocked_items INTEGER NOT NULL DEFAULT 0,
                failed_items INTEGER NOT NULL DEFAULT 0,
                actor_id TEXT,
                lease_owner TEXT,
                lease_token TEXT,
                lease_expires_at TEXT,
                started_at TEXT,
                completed_at TEXT,
                last_error_code TEXT,
                CHECK (total_items >= 0 AND planned_items >= 0
                    AND uploaded_items >= 0 AND applied_items >= 0
                    AND verified_items >= 0 AND skipped_items >= 0
                    AND blocked_items >= 0 AND failed_items >= 0),
                CHECK (
                    (mode = 'dry_run' AND parent_plan_id IS NULL)
                    OR (mode = 'apply' AND parent_plan_id IS NOT NULL)
                ),
                CHECK (
                    (status = 'running' AND lease_owner IS NOT NULL
                        AND lease_token IS NOT NULL AND lease_expires_at IS NOT NULL
                        AND started_at IS NOT NULL)
                    OR
                    (status <> 'running' AND lease_owner IS NULL
                        AND lease_token IS NULL AND lease_expires_at IS NULL)
                ),
                CHECK (
                    (status IN ('succeeded', 'failed', 'cancelled')
                        AND completed_at IS NOT NULL)
                    OR
                    (status NOT IN ('succeeded', 'failed', 'cancelled')
                        AND completed_at IS NULL)
                )
            );
            CREATE TABLE artifact_maintenance_items (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                item_key TEXT NOT NULL,
                state TEXT NOT NULL CHECK (state IN (
                    'planned', 'uploaded', 'applied', 'verified', 'skipped',
                    'blocked', 'failed'
                )),
                ordinal INTEGER NOT NULL,
                subject_kind TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                completed_at TEXT,
                last_error_code TEXT,
                UNIQUE (run_id, item_key),
                CHECK (ordinal >= 0 AND attempt_count >= 0),
                CHECK (
                    (state IN ('verified', 'skipped', 'failed')
                        AND completed_at IS NOT NULL)
                    OR
                    (state NOT IN ('verified', 'skipped', 'failed')
                        AND completed_at IS NULL)
                )
            );
            """
        )
        conn.execute(
            """
            INSERT INTO artifact_maintenance_runs (
                id, kind, mode, status, backend_fingerprint,
                scope_fingerprint, config_fingerprint, scope_json,
                created_at, updated_at, parent_plan_id, idempotency_key,
                cursor_json, total_items, planned_items, uploaded_items,
                applied_items, verified_items, skipped_items, blocked_items,
                failed_items, actor_id, lease_owner, lease_token,
                lease_expires_at, started_at, completed_at, last_error_code
            ) VALUES (?, 'migration', 'dry_run', 'planned', ?, ?, ?, '{}',
                ?, ?, NULL, ?, NULL, 1, 1, 0, 0, 0, 0, 0, 0,
                'legacy-operator', NULL, NULL, NULL, NULL, NULL, NULL)
            """,
            (
                legacy_run_id,
                "sha256:backend",
                "sha256:scope",
                "sha256:config",
                timestamp,
                timestamp,
                "a" * 64,
            ),
        )
        conn.execute(
            """
            INSERT INTO artifact_maintenance_items (
                id, run_id, item_key, state, ordinal, subject_kind,
                subject_id, payload_json, created_at, updated_at,
                attempt_count, completed_at, last_error_code
            ) VALUES (?, ?, ?, 'planned', 0, 'document', 'legacy-document',
                ?, ?, ?, 0, NULL, NULL)
            """,
            (
                "legacy-maintenance-item",
                legacy_run_id,
                legacy_item_key,
                json.dumps(
                    {
                        "object_key": "kb/workspace/private/object",
                        "target_uri": "s3://private-bucket/private/object",
                    }
                ),
                timestamp,
                timestamp,
            ),
        )
        conn.execute("DELETE FROM metadata_schema WHERE version = 13")
        conn.execute(
            "INSERT OR IGNORE INTO metadata_schema(version, applied_at) VALUES (12, ?)",
            (timestamp,),
        )
        conn.commit()

    upgraded = SQLiteMetadataStore(db_path)
    await upgraded.initialize()
    run = await upgraded.get_artifact_maintenance_run(legacy_run_id)
    assert run.metadata_backend == "sqlite"
    assert run.idempotency_key == artifact_maintenance_run_key(
        kind="migration",
        mode="dry_run",
        metadata_backend="sqlite",
        parent_plan_id=None,
        backend_fingerprint="sha256:backend",
        scope_fingerprint="sha256:scope",
        config_fingerprint="sha256:config",
    )
    assert run.total_items == run.blocked_items == 1
    assert run.planned_items == 0

    items, total = await upgraded.list_artifact_maintenance_items(legacy_run_id)
    assert total == 1
    blocked = items[0]
    assert blocked.item_key != legacy_item_key
    assert blocked.state == "blocked"
    assert blocked.completed_at is not None
    assert blocked.last_error_code == "maintenance_schema_upgrade_required"
    assert blocked.kb_id is None and blocked.document_id is None
    assert blocked.logical_group_id == "maintenance-schema-upgrade-required"
    assert blocked.relative_object_id.startswith("maintenance-schema-upgrade-required/")
    assert blocked.payload_json == '{"schema_upgrade_required":true}'
    assert "s3://" not in blocked.payload_json
    assert "private/object" not in blocked.payload_json

    first_snapshot = (run, blocked)
    await upgraded.initialize()
    repeated_run = await upgraded.get_artifact_maintenance_run(legacy_run_id)
    repeated_items, repeated_total = await upgraded.list_artifact_maintenance_items(
        legacy_run_id
    )
    assert repeated_total == 1
    assert (repeated_run, repeated_items[0]) == first_snapshot
    with sqlite3.connect(db_path) as conn:
        assert conn.execute(
            "SELECT 1 FROM metadata_schema WHERE version = 13"
        ).fetchone() == (1,)


async def test_manifest_idempotency_and_full_lease_state_machine(
    tmp_path: Path,
) -> None:
    store = SQLiteMetadataStore(tmp_path / "metadata.sqlite3")
    await store.initialize()

    pending = _manifest("manifest-main")
    assert await store.enqueue_artifact_cleanup_manifest(pending) == pending
    assert await store.enqueue_artifact_cleanup_manifest(pending) == pending
    mismatch = replace(pending, expected_checksum="sha256:different")
    with pytest.raises(ArtifactLifecycleConflictError):
        await store.enqueue_artifact_cleanup_manifest(mismatch)
    with pytest.raises(ArtifactLifecycleConflictError):
        await store.enqueue_artifact_cleanup_manifest(
            replace(pending, status="retained", disposition="retain")
        )

    retained = _manifest("manifest-retained", status="retained")
    await store.enqueue_artifact_cleanup_manifest(retained)
    claimed = await store.claim_due_artifact_cleanup_manifests(
        lease_owner="worker-1", limit=10, now=_NOW + timedelta(seconds=1)
    )
    assert [record.id for record in claimed] == [pending.id]
    lease = claimed[0]
    assert lease.status == "leased"
    with pytest.raises(ArtifactLifecycleLeaseError):
        await store.renew_artifact_cleanup_manifest_lease(
            lease.id,
            lease_owner="worker-1",
            lease_token="stale-token",
        )
    renewed = await store.renew_artifact_cleanup_manifest_lease(
        lease.id,
        lease_owner="worker-1",
        lease_token=lease.lease_token or "",
        lease_duration_seconds=120,
        now=_NOW + timedelta(seconds=2),
    )
    assert renewed.lease_expires_at is not None
    assert lease.lease_expires_at is not None
    assert str(renewed.lease_expires_at) > str(lease.lease_expires_at)

    retried = await store.retry_artifact_cleanup_manifest(
        renewed.id,
        lease_owner="worker-1",
        lease_token=renewed.lease_token or "",
        next_attempt_at=_NOW + timedelta(seconds=10),
        error_code="AWS_SECRET_ACCESS_KEY=redact-me",
        checked_at=_NOW + timedelta(seconds=3),
        max_attempt_count=1,
    )
    assert retried.status == "pending"
    assert retried.attempt_count == 1
    assert retried.last_error_code == "artifact_lifecycle_error"
    assert not await store.claim_due_artifact_cleanup_manifests(
        lease_owner="worker-too-early", now=_NOW + timedelta(seconds=9)
    )
    claimed_again = await store.claim_due_artifact_cleanup_manifests(
        lease_owner="worker-2", now=_NOW + timedelta(seconds=11), limit=1
    )
    blocked = await store.block_artifact_cleanup_manifest(
        retried.id,
        lease_owner="worker-2",
        lease_token=claimed_again[0].lease_token or "",
        error_code="ownership_conflict",
        checked_at=_NOW + timedelta(seconds=12),
        max_attempt_count=1,
    )
    assert blocked.status == "blocked"
    assert blocked.attempt_count == 1
    assert not await store.claim_due_artifact_cleanup_manifests(
        lease_owner="worker-3", now=_NOW + timedelta(days=1)
    )

    expiring = _manifest(
        "manifest-expiring",
        target_suffix="expiring",
        delete_after=_NOW + timedelta(seconds=20),
    )
    await store.enqueue_artifact_cleanup_manifest(expiring)
    expiring_lease = (
        await store.claim_due_artifact_cleanup_manifests(
            lease_owner="worker-expiring",
            lease_duration_seconds=1,
            now=_NOW + timedelta(seconds=21),
            limit=1,
        )
    )[0]
    with pytest.raises(ArtifactLifecycleLeaseError):
        await store.renew_artifact_cleanup_manifest_lease(
            expiring_lease.id,
            lease_owner="worker-expiring",
            lease_token=expiring_lease.lease_token or "",
            now=_NOW + timedelta(seconds=23),
        )
    recovered = await store.recover_expired_artifact_cleanup_manifest_leases(
        now=_NOW + timedelta(seconds=23),
        next_attempt_at=_NOW + timedelta(days=2),
        max_attempt_count=1,
    )
    assert [record.id for record in recovered] == [expiring_lease.id]
    assert recovered[0].status == "pending"
    assert recovered[0].last_error_code == "lease_expired"
    assert recovered[0].attempt_count == 1

    released = await store.release_retained_artifact_cleanup_manifests(
        retained.kb_id,
        retained.kb_generation,
        retained.manifest_group_id,
        [retained.id],
        released_at=_NOW + timedelta(seconds=40),
    )
    assert released[0].status == "pending"
    assert released[0].disposition == "delete"
    assert (
        await store.release_retained_artifact_cleanup_manifests(
            retained.kb_id,
            retained.kb_generation,
            retained.manifest_group_id,
            [retained.id],
            released_at=_NOW + timedelta(seconds=41),
        )
        == released
    )
    assert await store.enqueue_artifact_cleanup_manifest(retained) == released[0]
    retained_lease = (
        await store.claim_due_artifact_cleanup_manifests(
            lease_owner="worker-retained",
            now=_NOW + timedelta(seconds=42),
            limit=1,
        )
    )[0]
    succeeded = await store.complete_artifact_cleanup_manifest(
        retained.id,
        lease_owner="worker-retained",
        lease_token=retained_lease.lease_token or "",
        checked_at=_NOW + timedelta(seconds=43),
    )
    assert succeeded.status == "succeeded"
    assert datetime.fromisoformat(str(succeeded.audit_retain_until)) >= (
        datetime.fromisoformat(str(succeeded.completed_at)) + timedelta(days=30)
    )
    assert await store.enqueue_artifact_cleanup_manifest(retained) == succeeded
    assert (
        await store.prune_succeeded_artifact_cleanup_manifests(
            now=_NOW + timedelta(days=29)
        )
        == 0
    )
    assert (
        await store.prune_succeeded_artifact_cleanup_manifests(
            now=_NOW + timedelta(days=31)
        )
        == 1
    )
    with pytest.raises(ArtifactLifecycleNotFoundError):
        await store.get_artifact_cleanup_manifest(retained.id)
    assert (await store.get_artifact_cleanup_manifest(blocked.id)).status == "blocked"
    assert (await store.get_artifact_cleanup_manifest(expiring.id)).status == "pending"

    records, total = await store.list_artifact_cleanup_manifests(
        kb_id=pending.kb_id, limit=500
    )
    assert total == len(records) == 2
    aggregate = await store.aggregate_artifact_cleanup_manifests(
        kb_id=pending.kb_id,
        now=_NOW + timedelta(days=32),
    )
    assert aggregate["blocked"] == 1
    assert aggregate["pending"] == 1


async def test_independent_sqlite_manifest_claim_is_single_winner(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "metadata.sqlite3"
    first = SQLiteMetadataStore(db_path)
    second = SQLiteMetadataStore(db_path)
    await first.initialize()
    await second.initialize()
    await first.enqueue_artifact_cleanup_manifest(_manifest("manifest-race"))

    left, right = await asyncio.gather(
        first.claim_due_artifact_cleanup_manifests(
            lease_owner="worker-left", now=_NOW + timedelta(seconds=1), limit=1
        ),
        second.claim_due_artifact_cleanup_manifests(
            lease_owner="worker-right", now=_NOW + timedelta(seconds=1), limit=1
        ),
    )
    assert sorted([len(left), len(right)]) == [0, 1]
    winner = (left or right)[0]
    assert winner.status == "leased"


async def test_manifest_exact_cleanup_safety_filters_are_bounded(
    tmp_path: Path,
) -> None:
    store = SQLiteMetadataStore(tmp_path / "metadata.sqlite3")
    await store.initialize()
    group_id = "manifest-filter-group"
    source_manifest = _manifest(
        "manifest-filter-source",
        group_id=group_id,
        document_id="document-filter-source",
        source_generation_id="source-generation-filter",
        status="retained",
        target_suffix="filters/source-object",
    )
    artifact_manifest = _manifest(
        "manifest-filter-artifact",
        group_id=group_id,
        document_id="document-filter-artifact",
        artifact_id="artifact-filter",
        source_generation_id=None,
        status="retained",
        target_namespace="artifact",
        target_suffix="filters/artifact-object",
    )
    await store.enqueue_artifact_cleanup_manifests([source_manifest, artifact_manifest])

    await _assert_exact_manifest_filter_contract(
        store, source_manifest, artifact_manifest
    )


async def test_manifest_survives_document_delete_and_kb_metadata_purge(
    tmp_path: Path,
) -> None:
    store = SQLiteMetadataStore(tmp_path / "metadata.sqlite3")
    await store.initialize()
    kb_id = "kb-survival"
    generation = "generation-survival"
    await store.activate_kb_generation(kb_id, generation)
    document = _document(
        kb_id,
        "document-survival",
        status="uploaded",
        created_at=_NOW.isoformat(),
    )
    await store.create_documents_and_job([document], _job(kb_id, "job-survival", 1))
    manifest = _manifest(
        "manifest-survival",
        kb_id=kb_id,
        kb_generation=generation,
        document_id=document.id,
    )
    await store.enqueue_artifact_cleanup_manifest(manifest)

    deleted = await store.complete_document_delete(
        kb_id, document.id, metadata_patch={}
    )
    tombstone = await store.get_document_lifecycle(kb_id, document.id)
    assert tombstone is not None
    assert tombstone == deleted
    assert tombstone.status == "deleted"
    assert tombstone.deleted_at is not None
    assert await store.get_artifact_cleanup_manifest(manifest.id) == manifest
    filtered, filtered_total = await store.list_artifact_cleanup_manifests(
        kb_id=kb_id,
        kb_generation=generation,
        document_id=document.id,
        target_uri=manifest.target_uri,
        limit=1,
    )
    assert filtered == [manifest]
    assert filtered_total == 1
    assert (
        await store.count_artifact_cleanup_manifests(
            kb_id=kb_id,
            kb_generation=generation,
            document_id=document.id,
            target_uri=manifest.target_uri,
        )
        == 1
    )
    counts = await store.purge_kb_metadata(kb_id, generation)
    assert counts["documents"] == 1
    assert "artifact_cleanup_manifests" not in counts
    assert await store.get_artifact_cleanup_manifest(manifest.id) == manifest
    with pytest.raises(MetadataRecordNotFoundError):
        await store.get_document(kb_id, document.id)


async def test_maintenance_plan_item_idempotency_and_transition_fencing(
    tmp_path: Path,
) -> None:
    store = SQLiteMetadataStore(tmp_path / "metadata.sqlite3")
    await store.initialize()
    dry_run = _dry_run("maintenance-dry-run")
    assert await store.create_artifact_maintenance_run(dry_run) == dry_run
    assert await store.create_artifact_maintenance_run(dry_run) == dry_run
    with pytest.raises(ArtifactLifecycleConflictError):
        await store.create_artifact_maintenance_run(
            replace(dry_run, actor_id="operator-2")
        )

    apply = _apply_run("maintenance-apply", dry_run.id)
    with pytest.raises(ArtifactLifecycleConflictError):
        await store.create_artifact_maintenance_run(apply)

    claimed = await store.claim_artifact_maintenance_run(
        dry_run.id,
        lease_owner="maintenance-worker",
        now=_NOW + timedelta(seconds=1),
    )
    with pytest.raises(ArtifactLifecycleLeaseError):
        await store.transition_artifact_maintenance_run(
            dry_run.id,
            expected_status="running",
            new_status="succeeded",
            lease_owner="maintenance-worker",
            lease_token="stale-token",
            now=_NOW + timedelta(seconds=2),
        )

    item = _maintenance_item(dry_run.id)
    assert await store.create_artifact_maintenance_item(item) == item
    with pytest.raises(ArtifactLifecycleLeaseError):
        await store.transition_artifact_maintenance_item(
            dry_run.id,
            item.item_key,
            expected_state="planned",
            new_state="uploaded",
            now=_NOW + timedelta(seconds=2),
        )
    blocked_item = await store.transition_artifact_maintenance_item(
        dry_run.id,
        item.item_key,
        expected_state="planned",
        new_state="blocked",
        expected_updated_at=item.updated_at,
        run_lease_token=claimed.lease_token,
        error_code="ownership_unresolved",
        now=_NOW + timedelta(seconds=2),
    )
    assert blocked_item.state == "blocked"
    assert blocked_item.completed_at == blocked_item.updated_at
    assert blocked_item.last_error_code == "ownership_unresolved"
    with pytest.raises(ArtifactLifecycleStateError):
        await store.transition_artifact_maintenance_item(
            dry_run.id,
            item.item_key,
            expected_state="blocked",
            new_state="uploaded",
            run_lease_token=claimed.lease_token,
            now=_NOW + timedelta(seconds=3),
        )
    reopened = await store.transition_artifact_maintenance_item(
        dry_run.id,
        item.item_key,
        expected_state="blocked",
        new_state="planned",
        expected_updated_at=blocked_item.updated_at,
        run_lease_token=claimed.lease_token,
        now=_NOW + timedelta(seconds=3),
    )
    assert reopened.completed_at is None
    assert reopened.last_error_code is None
    with pytest.raises(ArtifactLifecycleStateError):
        await store.transition_artifact_maintenance_item(
            dry_run.id,
            item.item_key,
            expected_state="planned",
            new_state="uploaded",
            expected_updated_at=blocked_item.updated_at,
            run_lease_token=claimed.lease_token,
            now=_NOW + timedelta(seconds=4),
        )
    uploaded = await store.transition_artifact_maintenance_item(
        dry_run.id,
        item.item_key,
        expected_state="planned",
        new_state="uploaded",
        expected_updated_at=reopened.updated_at,
        run_lease_token=claimed.lease_token,
        increment_attempt=True,
        now=_NOW + timedelta(seconds=4),
    )
    assert uploaded.state == "uploaded"
    assert uploaded.attempt_count == 1
    # Replaying the immutable plan item returns current checkpoint authority.
    assert await store.create_artifact_maintenance_item(item) == uploaded
    with pytest.raises(ArtifactLifecycleStateError):
        await store.transition_artifact_maintenance_item(
            dry_run.id,
            item.item_key,
            expected_state="planned",
            new_state="uploaded",
            run_lease_token=claimed.lease_token,
            now=_NOW + timedelta(seconds=5),
        )
    conflict_item = replace(item, ordinal=2)
    with pytest.raises(ArtifactLifecycleConflictError):
        await store.create_artifact_maintenance_item(conflict_item)

    applied_item = await store.transition_artifact_maintenance_item(
        dry_run.id,
        item.item_key,
        expected_state="uploaded",
        new_state="applied",
        run_lease_token=claimed.lease_token,
        now=_NOW + timedelta(seconds=5),
    )
    verified = await store.transition_artifact_maintenance_item(
        dry_run.id,
        item.item_key,
        expected_state="applied",
        new_state="verified",
        run_lease_token=claimed.lease_token,
        now=_NOW + timedelta(seconds=6),
    )
    assert applied_item.state == "applied"
    assert verified.state == "verified"
    assert await store.aggregate_artifact_maintenance_items(dry_run.id) == {
        "planned": 0,
        "uploaded": 0,
        "applied": 0,
        "verified": 1,
        "skipped": 0,
        "blocked": 0,
        "failed": 0,
        "total": 1,
    }

    succeeded = await store.transition_artifact_maintenance_run(
        dry_run.id,
        expected_status="running",
        new_status="succeeded",
        lease_owner="maintenance-worker",
        lease_token=claimed.lease_token,
        counters={"total_items": 1, "verified_items": 1},
        now=_NOW + timedelta(seconds=7),
    )
    assert succeeded.status == "succeeded"
    assert (await store.create_artifact_maintenance_run(dry_run)).status == "succeeded"

    mismatched_backend_apply = _apply_run(
        "maintenance-apply-wrong-backend",
        dry_run.id,
        metadata_backend="postgres",
    )
    with pytest.raises(ArtifactLifecycleConflictError):
        await store.create_artifact_maintenance_run(mismatched_backend_apply)

    assert await store.create_artifact_maintenance_run(apply) == apply
    assert await store.create_artifact_maintenance_run(apply) == apply
    apply_claim = await store.claim_artifact_maintenance_run(
        apply.id,
        lease_owner="apply-worker",
        lease_duration_seconds=1,
        now=_NOW + timedelta(minutes=2),
    )
    recovered_runs = await store.recover_expired_artifact_maintenance_run_leases(
        now=_NOW + timedelta(minutes=2, seconds=2)
    )
    assert [run.id for run in recovered_runs] == [apply_claim.id]
    assert recovered_runs[0].status == "failed"
    assert recovered_runs[0].last_error_code == "lease_expired"
    resumed = await store.claim_artifact_maintenance_run(
        apply.id,
        lease_owner="apply-worker-resumed",
        now=_NOW + timedelta(minutes=2, seconds=3),
    )
    assert resumed.status == "running"
    runs, total = await store.list_artifact_maintenance_runs(
        kind="migration", metadata_backend="sqlite"
    )
    assert total == 2
    assert {run.id for run in runs} == {dry_run.id, apply.id}
    assert await store.list_artifact_maintenance_runs(metadata_backend="postgres") == (
        [],
        0,
    )


async def test_maintenance_item_authority_filters_and_indexes(
    tmp_path: Path,
) -> None:
    store = SQLiteMetadataStore(tmp_path / "metadata.sqlite3")
    await store.initialize()
    run = _dry_run("maintenance-filter-run")
    await store.create_artifact_maintenance_run(run)
    items = [
        _maintenance_item(
            run.id,
            ordinal=0,
            kb_id="kb-filter-a",
            kb_generation="generation-a",
            workspace="workspace-a",
            document_id="document-a",
            artifact_id="artifact-a",
            logical_group_id="logical-group-a",
            relative_object_id="source/document-a.txt",
            target_uri="s3://artifact-bucket/a/document-a.txt",
        ),
        _maintenance_item(
            run.id,
            ordinal=1,
            kb_id="kb-filter-a",
            kb_generation="generation-a",
            workspace="workspace-a",
            document_id="document-b",
            artifact_id="artifact-b",
            logical_group_id="logical-group-b",
            relative_object_id="source/document-b.txt",
            target_uri="s3://artifact-bucket/a/document-b.txt",
        ),
        _maintenance_item(
            run.id,
            ordinal=2,
            kb_id="kb-filter-b",
            kb_generation="generation-b",
            workspace="workspace-b",
            document_id="document-c",
            artifact_id="artifact-c",
            logical_group_id="logical-group-a",
            relative_object_id="artifact/document-c.bin",
            target_uri="s3://artifact-bucket/b/document-c.bin",
        ),
    ]
    assert await store.create_artifact_maintenance_items(items) == items
    claimed = await store.claim_artifact_maintenance_run(
        run.id,
        lease_owner="filter-worker",
        now=_NOW + timedelta(seconds=1),
    )
    blocked = await store.transition_artifact_maintenance_item(
        run.id,
        items[1].item_key,
        expected_state="planned",
        new_state="blocked",
        run_lease_token=claimed.lease_token,
        error_code="manual_review_required",
        now=_NOW + timedelta(seconds=2),
    )
    assert blocked.completed_at is not None

    by_kb, total = await store.list_artifact_maintenance_items(
        run.id, kb_id="kb-filter-a", kb_generation="generation-a"
    )
    assert total == 2
    assert {item.document_id for item in by_kb} == {"document-a", "document-b"}
    by_document_artifact, total = await store.list_artifact_maintenance_items(
        run.id,
        document_id="document-b",
        artifact_id="artifact-b",
        state="blocked",
    )
    assert total == 1 and by_document_artifact == [blocked]
    by_group, total = await store.list_artifact_maintenance_items(
        run.id, logical_group_id="logical-group-a"
    )
    assert total == 2
    assert {item.ordinal for item in by_group} == {0, 2}
    planned, total = await store.list_artifact_maintenance_items(
        run.id, state="planned"
    )
    assert total == 2
    assert {item.ordinal for item in planned} == {0, 2}
    by_digest, total = await store.list_artifact_maintenance_items(
        run.id, target_uri_digest=items[0].target_uri_digest
    )
    assert total == 1 and by_digest == [items[0]]
    by_workspace, total = await store.list_artifact_maintenance_items(
        run.id, workspace="workspace-b"
    )
    assert total == 1 and by_workspace == [items[2]]
    assert await store.aggregate_artifact_maintenance_items(run.id) == {
        "planned": 2,
        "uploaded": 0,
        "applied": 0,
        "verified": 0,
        "skipped": 0,
        "blocked": 1,
        "failed": 0,
        "total": 3,
    }


async def test_durable_recovery_cursor_rotation_restart_and_stale_generation(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "metadata.sqlite3"
    first = SQLiteMetadataStore(db_path)
    await first.initialize()
    expected = await _seed_recovery_documents(
        first,
        kb_id="kb-recovery",
        kb_generation="generation-1",
        count=451,
        parsed_count=301,
    )
    page_one = await first.reserve_pipeline_artifact_recovery_page(
        "kb-recovery", "generation-1", 137
    )
    assert [document.id for document in page_one] == expected[:137]
    first_cursor = await first.get_artifact_recovery_cursor(
        "kb-recovery", "generation-1"
    )
    assert first_cursor is not None
    assert first_cursor.last_created_at == "2026-08-03T01:00:00+00:00"
    await first.close()

    restarted = SQLiteMetadataStore(db_path)
    await restarted.initialize()
    pages = [
        await restarted.reserve_pipeline_artifact_recovery_page(
            "kb-recovery", "generation-1", 137
        )
        for _ in range(3)
    ]
    remaining = [document.id for page in pages for document in page]
    assert remaining == expected[137:]
    assert len(page_one) + sum(map(len, pages)) == 451
    cursor = await restarted.get_artifact_recovery_cursor("kb-recovery", "generation-1")
    assert cursor is not None
    assert cursor.status == "parsed"
    assert cursor.sweep == 1
    assert cursor.last_created_at is None
    assert cursor.last_document_id is None

    with pytest.raises(ValueError):
        await restarted.reserve_pipeline_artifact_recovery_page(
            "kb-recovery", "generation-1", 0
        )
    next_sweep = await restarted.reserve_pipeline_artifact_recovery_page(
        "kb-recovery", "generation-1", 999
    )
    assert len(next_sweep) == 200
    assert [document.id for document in next_sweep] == expected[:200]

    await restarted.purge_kb_metadata("kb-recovery", "generation-1")
    await restarted.activate_kb_generation("kb-recovery", "generation-2")
    await _seed_recovery_documents(
        restarted,
        kb_id="kb-recovery",
        kb_generation="generation-2",
        count=3,
        parsed_count=2,
    )
    with pytest.raises(ArtifactRecoveryGenerationError):
        await restarted.reserve_pipeline_artifact_recovery_page(
            "kb-recovery", "generation-1", 10
        )
    current = await restarted.reserve_pipeline_artifact_recovery_page(
        "kb-recovery", "generation-2", 10
    )
    assert len(current) == 3


async def test_independent_sqlite_recovery_reservations_do_not_overlap(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "metadata.sqlite3"
    first = SQLiteMetadataStore(db_path)
    second = SQLiteMetadataStore(db_path)
    await first.initialize()
    await second.initialize()
    await _seed_recovery_documents(
        first,
        kb_id="kb-recovery-race",
        kb_generation="generation-race",
        count=460,
        parsed_count=310,
    )
    left, right = await asyncio.gather(
        first.reserve_pipeline_artifact_recovery_page(
            "kb-recovery-race", "generation-race", 120
        ),
        second.reserve_pipeline_artifact_recovery_page(
            "kb-recovery-race", "generation-race", 120
        ),
    )
    left_ids = {document.id for document in left}
    right_ids = {document.id for document in right}
    assert len(left_ids) == len(right_ids) == 120
    assert left_ids.isdisjoint(right_ids)
    cursor = await first.get_artifact_recovery_cursor(
        "kb-recovery-race", "generation-race"
    )
    assert cursor is not None
    assert cursor.version == 3


@pytest.fixture(params=["sqlite", "postgres"])
async def _recovery_cursor_store(request, tmp_path: Path):
    """Parametrized store for recovery-cursor delete tests.

    ``sqlite`` always runs; ``postgres`` runs live against a real PostgreSQL
    only when ``LIGHTRAG_KB_POSTGRES_TEST_DSN`` (or ``POSTGRES_TEST_DSN``) is
    set, otherwise it is skipped.
    """

    backend = request.param
    if backend == "postgres" and not _POSTGRES_DSN:
        pytest.skip(
            "live PostgreSQL recovery-cursor delete test skipped: set "
            "LIGHTRAG_KB_POSTGRES_TEST_DSN to enable"
        )
    suffix = uuid.uuid4().hex
    if backend == "sqlite":
        store = SQLiteMetadataStore(tmp_path / f"recovery-delete-{suffix}.sqlite3")
    else:
        from lightrag.api.postgres_metadata_store import PostgresMetadataStore

        store = PostgresMetadataStore(dsn=_POSTGRES_DSN, min_size=1, max_size=1)
    await store.initialize()
    tracked_kb_ids: list[str] = []
    store._test_recovery_cursor_kb_ids = tracked_kb_ids  # type: ignore[attr-defined]
    try:
        yield store
    finally:
        if backend == "postgres":
            # Recovery cursors intentionally survive purge_kb_metadata; remove
            # them directly alongside the rest of the KB control-plane rows so
            # the disposable shared database stays clean across runs.
            for kb_id in tracked_kb_ids:
                try:
                    async with store._pool_or_raise().acquire() as conn:  # type: ignore[attr-defined]
                        async with conn.transaction():
                            await conn.execute(
                                "DELETE FROM kb_artifact_recovery_cursors "
                                "WHERE kb_id = $1",
                                kb_id,
                            )
                            await conn.execute(
                                "DELETE FROM kb_jobs WHERE kb_id = $1", kb_id
                            )
                            await conn.execute(
                                "DELETE FROM kb_documents WHERE kb_id = $1", kb_id
                            )
                            await conn.execute(
                                "DELETE FROM enterprise_kb_lifecycle WHERE kb_id = $1",
                                kb_id,
                            )
                except Exception:
                    pass
        await store.close()


def _track_recovery_kb(store, kb_id: str) -> str:
    tracked = getattr(store, "_test_recovery_cursor_kb_ids", None)
    if tracked is not None:
        tracked.append(kb_id)
    return kb_id


async def test_delete_recovery_cursor_after_reserve_returns_true(
    _recovery_cursor_store,
) -> None:
    store = _recovery_cursor_store
    kb_id = _track_recovery_kb(store, f"kb-recovery-delete-after-{uuid.uuid4().hex}")
    generation = f"generation-{uuid.uuid4().hex}"
    await _seed_recovery_documents(
        store,
        kb_id=kb_id,
        kb_generation=generation,
        count=5,
        parsed_count=3,
    )
    await store.reserve_pipeline_artifact_recovery_page(kb_id, generation, 5)
    assert await store.get_artifact_recovery_cursor(kb_id, generation) is not None
    assert await store.delete_artifact_recovery_cursor(kb_id, generation) is True
    assert await store.get_artifact_recovery_cursor(kb_id, generation) is None


async def test_delete_recovery_cursor_when_absent_returns_false(
    _recovery_cursor_store,
) -> None:
    store = _recovery_cursor_store
    kb_id = _track_recovery_kb(store, f"kb-recovery-delete-absent-{uuid.uuid4().hex}")
    generation = f"generation-{uuid.uuid4().hex}"
    # No prior reserve: no cursor exists.
    assert await store.get_artifact_recovery_cursor(kb_id, generation) is None
    assert await store.delete_artifact_recovery_cursor(kb_id, generation) is False


async def test_delete_recovery_cursor_is_idempotent(
    _recovery_cursor_store,
) -> None:
    store = _recovery_cursor_store
    kb_id = _track_recovery_kb(
        store, f"kb-recovery-delete-idempotent-{uuid.uuid4().hex}"
    )
    generation = f"generation-{uuid.uuid4().hex}"
    await _seed_recovery_documents(
        store,
        kb_id=kb_id,
        kb_generation=generation,
        count=3,
        parsed_count=2,
    )
    await store.reserve_pipeline_artifact_recovery_page(kb_id, generation, 3)
    first = await store.delete_artifact_recovery_cursor(kb_id, generation)
    second = await store.delete_artifact_recovery_cursor(kb_id, generation)
    assert first is True
    assert second is False
    assert await store.get_artifact_recovery_cursor(kb_id, generation) is None


async def test_delete_recovery_cursor_does_not_affect_other_kb_cursors(
    _recovery_cursor_store,
) -> None:
    store = _recovery_cursor_store
    kb_a = _track_recovery_kb(store, f"kb-recovery-delete-other-a-{uuid.uuid4().hex}")
    kb_b = _track_recovery_kb(store, f"kb-recovery-delete-other-b-{uuid.uuid4().hex}")
    gen_a = f"generation-{uuid.uuid4().hex}"
    gen_b = f"generation-{uuid.uuid4().hex}"
    await _seed_recovery_documents(
        store, kb_id=kb_a, kb_generation=gen_a, count=3, parsed_count=2
    )
    await _seed_recovery_documents(
        store, kb_id=kb_b, kb_generation=gen_b, count=3, parsed_count=2
    )
    await store.reserve_pipeline_artifact_recovery_page(kb_a, gen_a, 3)
    await store.reserve_pipeline_artifact_recovery_page(kb_b, gen_b, 3)

    assert await store.delete_artifact_recovery_cursor(kb_a, gen_a) is True
    # KB B cursor is untouched.
    assert await store.get_artifact_recovery_cursor(kb_b, gen_b) is not None
    assert await store.get_artifact_recovery_cursor(kb_a, gen_a) is None
    # Deleting KB B now succeeds independently.
    assert await store.delete_artifact_recovery_cursor(kb_b, gen_b) is True


async def test_delete_recovery_cursor_with_stale_generation_still_works(
    _recovery_cursor_store,
) -> None:
    """Delete does not require any specific lifecycle state.

    After the KB's active generation is purged and a new generation activated,
    the stale cursor row for the old generation can still be removed. This
    mirrors the post-hard-delete-drain cleanup path.
    """

    store = _recovery_cursor_store
    kb_id = _track_recovery_kb(store, f"kb-recovery-delete-stale-{uuid.uuid4().hex}")
    old_generation = f"generation-old-{uuid.uuid4().hex}"
    await _seed_recovery_documents(
        store,
        kb_id=kb_id,
        kb_generation=old_generation,
        count=3,
        parsed_count=2,
    )
    await store.reserve_pipeline_artifact_recovery_page(kb_id, old_generation, 3)
    # Cursor survives purge (recovery cursors are durable authority).
    await store.purge_kb_metadata(kb_id, old_generation)
    assert await store.get_artifact_recovery_cursor(kb_id, old_generation) is not None
    # Activating a new generation does not block deletion of the old cursor.
    new_generation = f"generation-new-{uuid.uuid4().hex}"
    await store.activate_kb_generation(kb_id, new_generation)
    assert await store.delete_artifact_recovery_cursor(kb_id, old_generation) is True
    assert await store.get_artifact_recovery_cursor(kb_id, old_generation) is None


async def test_concurrent_delete_recovery_cursor_safety(
    _recovery_cursor_store,
) -> None:
    """Two concurrent deletes on the same cursor: exactly one wins.

    The loser returns False without error. Both SQLite (file-lock serialization)
    and PostgreSQL (row-lock serialization) guarantee mutual exclusion.
    """

    store = _recovery_cursor_store
    kb_id = _track_recovery_kb(
        store, f"kb-recovery-delete-concurrent-{uuid.uuid4().hex}"
    )
    generation = f"generation-{uuid.uuid4().hex}"
    await _seed_recovery_documents(
        store,
        kb_id=kb_id,
        kb_generation=generation,
        count=5,
        parsed_count=3,
    )
    await store.reserve_pipeline_artifact_recovery_page(kb_id, generation, 5)

    if isinstance(store, SQLiteMetadataStore):
        peer: Any = SQLiteMetadataStore(store.db_path)
    else:
        from lightrag.api.postgres_metadata_store import PostgresMetadataStore

        peer = PostgresMetadataStore(dsn=_POSTGRES_DSN, min_size=1, max_size=1)
    await peer.initialize()
    try:
        left, right = await asyncio.gather(
            store.delete_artifact_recovery_cursor(kb_id, generation),
            peer.delete_artifact_recovery_cursor(kb_id, generation),
        )
        assert {left, right} == {True, False}
        assert await store.get_artifact_recovery_cursor(kb_id, generation) is None
    finally:
        await peer.close()


def test_sqlite_postgres_lifecycle_api_signatures_match() -> None:
    from lightrag.api.postgres_metadata_store import PostgresMetadataStore

    public_names = {
        name
        for name in dir(SQLiteMetadataStore)
        if not name.startswith("_")
        and (
            "artifact_cleanup" in name
            or "artifact_maintenance" in name
            or "artifact_recovery" in name
            or name == "get_document_lifecycle"
        )
    }
    assert public_names
    for name in public_names:
        assert hasattr(PostgresMetadataStore, name), name
        assert inspect.signature(
            getattr(SQLiteMetadataStore, name)
        ) == inspect.signature(getattr(PostgresMetadataStore, name))


@pytest.mark.skipif(
    not _POSTGRES_DSN,
    reason="live PostgreSQL lifecycle proof requires LIGHTRAG_KB_POSTGRES_TEST_DSN",
)
async def test_postgres_live_artifact_lifecycle_parity() -> None:
    from lightrag.api.postgres_metadata_store import PostgresMetadataStore

    suffix = uuid.uuid4().hex
    kb_id = f"kb_artifact_lifecycle_{suffix}"
    generation = f"generation-{suffix}"
    manifest = _manifest(
        f"manifest-{suffix}",
        kb_id=kb_id,
        kb_generation=generation,
        group_id=f"group-{suffix}",
        document_id=f"document-{suffix}",
    )
    filter_group_id = f"filter-group-{suffix}"
    source_filter_manifest = _manifest(
        f"manifest-filter-source-{suffix}",
        kb_id=kb_id,
        kb_generation=generation,
        group_id=filter_group_id,
        document_id=f"document-filter-source-{suffix}",
        source_generation_id=f"source-generation-filter-{suffix}",
        status="retained",
        target_suffix=f"filters/{suffix}/source-object",
    )
    artifact_filter_manifest = _manifest(
        f"manifest-filter-artifact-{suffix}",
        kb_id=kb_id,
        kb_generation=generation,
        group_id=filter_group_id,
        document_id=f"document-filter-artifact-{suffix}",
        artifact_id=f"artifact-filter-{suffix}",
        source_generation_id=None,
        status="retained",
        target_namespace="artifact",
        target_suffix=f"filters/{suffix}/artifact-object",
    )
    store = PostgresMetadataStore(
        dsn=_POSTGRES_DSN,
        min_size=1,
        max_size=2,
        operation_lock_pool_max_size=2,
    )
    peer = PostgresMetadataStore(
        dsn=_POSTGRES_DSN,
        min_size=1,
        max_size=1,
        operation_lock_pool_max_size=1,
    )
    await store.initialize()
    await peer.initialize()
    try:
        async with store._pool_or_raise().acquire() as conn:
            table_names = {
                await conn.fetchval(
                    "SELECT to_regclass('kb_artifact_cleanup_manifests')"
                ),
                await conn.fetchval(
                    "SELECT to_regclass('kb_artifact_maintenance_runs')"
                ),
                await conn.fetchval(
                    "SELECT to_regclass('kb_artifact_maintenance_items')"
                ),
                await conn.fetchval(
                    "SELECT to_regclass('kb_artifact_recovery_cursors')"
                ),
            }
            cleanup_foreign_keys = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM pg_constraint
                WHERE contype = 'f'
                  AND conrelid = 'kb_artifact_cleanup_manifests'::regclass
                """
            )
            maintenance_columns = {
                (str(row["table_name"]), str(row["column_name"]))
                for row in await conn.fetch(
                    """
                    SELECT table_name, column_name
                    FROM information_schema.columns
                    WHERE table_schema = current_schema()
                      AND table_name IN (
                          'kb_artifact_maintenance_runs',
                          'kb_artifact_maintenance_items'
                      )
                    """
                )
            }
            maintenance_indexes = {
                str(row["indexname"])
                for row in await conn.fetch(
                    """
                    SELECT indexname
                    FROM pg_indexes
                    WHERE schemaname = current_schema()
                      AND tablename IN (
                          'kb_artifact_maintenance_runs',
                          'kb_artifact_maintenance_items'
                      )
                    """
                )
            }
            schema_v6 = await conn.fetchval(
                "SELECT 1 FROM kb_metadata_schema WHERE version = 6"
            )
        assert table_names == {
            "kb_artifact_cleanup_manifests",
            "kb_artifact_maintenance_runs",
            "kb_artifact_maintenance_items",
            "kb_artifact_recovery_cursors",
        }
        assert cleanup_foreign_keys == 0
        assert (
            "kb_artifact_maintenance_runs",
            "metadata_backend",
        ) in maintenance_columns
        assert {
            ("kb_artifact_maintenance_items", "kb_id"),
            ("kb_artifact_maintenance_items", "kb_generation"),
            ("kb_artifact_maintenance_items", "document_id"),
            ("kb_artifact_maintenance_items", "artifact_id"),
            ("kb_artifact_maintenance_items", "logical_group_id"),
            ("kb_artifact_maintenance_items", "relative_object_id"),
            ("kb_artifact_maintenance_items", "target_uri_authority"),
            ("kb_artifact_maintenance_items", "target_uri_digest"),
        } <= maintenance_columns
        assert {
            "idx_kb_artifact_maintenance_run_backend",
            "idx_kb_artifact_maintenance_item_run_state",
            "idx_kb_artifact_maintenance_item_kb",
            "idx_kb_artifact_maintenance_item_document",
            "idx_kb_artifact_maintenance_item_artifact",
            "idx_kb_artifact_maintenance_item_group",
            "idx_kb_artifact_maintenance_item_uri_digest",
        } <= maintenance_indexes
        assert schema_v6 == 1

        # Exercise the real PostgreSQL v5 -> v6 backfill inside a rollback-only
        # DDL transaction so the disposable test database is left upgraded.
        async with store._pool_or_raise().acquire() as conn:
            migration_transaction = conn.transaction()
            await migration_transaction.start()
            try:
                await conn.execute(
                    """
                    ALTER TABLE kb_artifact_maintenance_runs
                        ALTER COLUMN metadata_backend DROP NOT NULL;
                    ALTER TABLE kb_artifact_maintenance_items
                        ALTER COLUMN logical_group_id DROP NOT NULL,
                        ALTER COLUMN relative_object_id DROP NOT NULL;
                    """
                )
                legacy_run_id = f"legacy-run-{suffix}"
                legacy_item_id = f"legacy-item-{suffix}"
                await conn.execute(
                    """
                    INSERT INTO kb_artifact_maintenance_runs (
                        id, kind, mode, status, backend_fingerprint,
                        scope_fingerprint, config_fingerprint, scope_json,
                        created_at, updated_at, parent_plan_id, idempotency_key,
                        cursor_json, total_items, planned_items, uploaded_items,
                        applied_items, verified_items, skipped_items,
                        blocked_items, failed_items, actor_id, lease_owner,
                        lease_token, lease_expires_at, started_at, completed_at,
                        last_error_code
                    ) VALUES (
                        $1, 'migration', 'dry_run', 'planned', 'sha256:backend',
                        'sha256:scope', 'sha256:config', '{}', $2, $2, NULL,
                        $3, NULL, 1, 1, 0, 0, 0, 0, 0, 0,
                        'legacy-operator', NULL, NULL, NULL, NULL, NULL, NULL
                    )
                    """,
                    legacy_run_id,
                    _NOW.isoformat(),
                    "c" * 64,
                )
                await conn.execute(
                    """
                    INSERT INTO kb_artifact_maintenance_items (
                        id, run_id, item_key, state, ordinal, subject_kind,
                        subject_id, payload_json, created_at, updated_at,
                        attempt_count, completed_at, last_error_code
                    ) VALUES (
                        $1, $2, $3, 'planned', 0, 'document',
                        'legacy-document', $4, $5, $5, 0, NULL, NULL
                    )
                    """,
                    legacy_item_id,
                    legacy_run_id,
                    "d" * 64,
                    json.dumps(
                        {
                            "object_key": "kb/workspace/private/object",
                            "target_uri": "s3://private-bucket/private/object",
                        }
                    ),
                    _NOW.isoformat(),
                )
                await store._migrate_artifact_lifecycle_schema_v6(conn)
                legacy_run = await conn.fetchrow(
                    "SELECT * FROM kb_artifact_maintenance_runs WHERE id = $1",
                    legacy_run_id,
                )
                legacy_item = await conn.fetchrow(
                    "SELECT * FROM kb_artifact_maintenance_items WHERE id = $1",
                    legacy_item_id,
                )
                assert legacy_run is not None
                assert legacy_run["metadata_backend"] == "postgres"
                assert legacy_run["blocked_items"] == 1
                assert legacy_item is not None
                assert legacy_item["state"] == "blocked"
                assert legacy_item["completed_at"] is not None
                assert (
                    legacy_item["last_error_code"]
                    == "maintenance_schema_upgrade_required"
                )
                assert legacy_item["payload_json"] == (
                    '{"schema_upgrade_required":true}'
                )
                assert legacy_item["target_uri_authority"] is None
                assert legacy_item["target_uri_digest"] is None
            finally:
                await migration_transaction.rollback()

        await store.activate_kb_generation(kb_id, generation)
        assert manifest.document_id is not None
        document = _document(
            kb_id,
            manifest.document_id,
            status="uploaded",
            created_at=_NOW.isoformat(),
        )
        await store.create_documents_and_job(
            [document], _job(kb_id, f"job-manifest-{suffix}", 1)
        )
        assert await store.enqueue_artifact_cleanup_manifest(manifest) == manifest
        await store.enqueue_artifact_cleanup_manifests(
            [source_filter_manifest, artifact_filter_manifest]
        )
        await _assert_exact_manifest_filter_contract(
            store, source_filter_manifest, artifact_filter_manifest
        )
        deleted = await store.complete_document_delete(
            kb_id, document.id, metadata_patch={}
        )
        tombstone = await store.get_document_lifecycle(kb_id, document.id)
        assert tombstone is not None
        assert tombstone == deleted
        assert tombstone.status == "deleted"
        assert tombstone.deleted_at is not None
        with pytest.raises(MetadataRecordNotFoundError):
            await store.get_document(kb_id, document.id)
        filtered, filtered_total = await store.list_artifact_cleanup_manifests(
            kb_id=kb_id,
            kb_generation=generation,
            document_id=document.id,
            target_uri=manifest.target_uri,
            limit=1,
        )
        assert filtered == [manifest]
        assert filtered_total == 1
        assert (
            await store.count_artifact_cleanup_manifests(
                kb_id=kb_id,
                kb_generation=generation,
                document_id=document.id,
                target_uri=manifest.target_uri,
            )
            == 1
        )
        left, right = await asyncio.gather(
            store.claim_due_artifact_cleanup_manifests(
                lease_owner=f"worker-left-{suffix}", now=_NOW + timedelta(seconds=1)
            ),
            peer.claim_due_artifact_cleanup_manifests(
                lease_owner=f"worker-right-{suffix}", now=_NOW + timedelta(seconds=1)
            ),
        )
        assert sorted([len(left), len(right)]) == [0, 1]
        lease = (left or right)[0]
        owner = lease.lease_owner or ""
        completed = await store.complete_artifact_cleanup_manifest(
            manifest.id,
            lease_owner=owner,
            lease_token=lease.lease_token or "",
            checked_at=_NOW + timedelta(seconds=2),
        )
        assert completed.status == "succeeded"

        maintenance_run = _dry_run(
            f"maintenance-dry-{suffix}", metadata_backend="postgres"
        )
        assert (
            await store.create_artifact_maintenance_run(maintenance_run)
            == maintenance_run
        )
        maintenance_claim = await store.claim_artifact_maintenance_run(
            maintenance_run.id,
            lease_owner=f"maintenance-worker-{suffix}",
            now=_NOW + timedelta(seconds=10),
        )
        maintenance_item = _maintenance_item(
            maintenance_run.id,
            kb_id=kb_id,
            kb_generation=generation,
            workspace=f"workspace-{suffix}",
            document_id=f"document-{suffix}",
            artifact_id=f"artifact-{suffix}",
            logical_group_id=f"logical-group-{suffix}",
            relative_object_id=f"source/document-{suffix}.txt",
            target_uri=(f"s3://artifact-bucket/{kb_id}/source/document-{suffix}.txt"),
        )
        await store.create_artifact_maintenance_item(maintenance_item)
        filtered_items, filtered_total = await store.list_artifact_maintenance_items(
            maintenance_run.id,
            kb_id=kb_id,
            kb_generation=generation,
            document_id=f"document-{suffix}",
            artifact_id=f"artifact-{suffix}",
            logical_group_id=f"logical-group-{suffix}",
            target_uri_digest=maintenance_item.target_uri_digest,
            state="planned",
        )
        assert filtered_total == 1 and filtered_items == [maintenance_item]
        verified_item = await store.transition_artifact_maintenance_item(
            maintenance_run.id,
            maintenance_item.item_key,
            expected_state="planned",
            new_state="verified",
            run_lease_token=maintenance_claim.lease_token,
            now=_NOW + timedelta(seconds=11),
        )
        assert verified_item.completed_at is not None
        await store.transition_artifact_maintenance_run(
            maintenance_run.id,
            expected_status="running",
            new_status="succeeded",
            lease_owner=maintenance_claim.lease_owner,
            lease_token=maintenance_claim.lease_token,
            counters={"total_items": 1, "verified_items": 1},
            now=_NOW + timedelta(seconds=12),
        )
        apply_run = _apply_run(
            f"maintenance-apply-{suffix}",
            maintenance_run.id,
            metadata_backend="postgres",
        )
        assert await store.create_artifact_maintenance_run(apply_run) == apply_run
        with pytest.raises(ArtifactLifecycleConflictError):
            await store.create_artifact_maintenance_run(
                _apply_run(
                    f"maintenance-apply-wrong-backend-{suffix}",
                    maintenance_run.id,
                    metadata_backend="sqlite",
                )
            )
        postgres_runs, postgres_run_total = await store.list_artifact_maintenance_runs(
            metadata_backend="postgres", kind="migration"
        )
        assert postgres_run_total >= 2
        assert {maintenance_run.id, apply_run.id} <= {run.id for run in postgres_runs}

        expected = await _seed_recovery_documents(
            store,
            kb_id=kb_id,
            kb_generation=generation,
            count=5,
            parsed_count=3,
        )
        reserved = await store.reserve_pipeline_artifact_recovery_page(
            kb_id, generation, 5
        )
        assert [document.id for document in reserved] == expected
        await store.purge_kb_metadata(kb_id, generation)
        assert (
            await store.get_artifact_cleanup_manifest(manifest.id)
        ).status == "succeeded"
    finally:
        # The DSN gate is explicitly test-only. Cleanup exact test identities;
        # do not rely on purge because manifests intentionally survive it.
        try:
            async with store._pool_or_raise().acquire() as conn:
                async with conn.transaction():
                    await conn.execute(
                        "DELETE FROM kb_artifact_maintenance_items WHERE run_id LIKE $1",
                        f"%{suffix}%",
                    )
                    await conn.execute(
                        "DELETE FROM kb_artifact_maintenance_runs WHERE id LIKE $1",
                        f"%{suffix}%",
                    )
                    await conn.execute(
                        "DELETE FROM kb_artifact_cleanup_manifests WHERE kb_id = $1",
                        kb_id,
                    )
                    await conn.execute(
                        "DELETE FROM kb_artifact_recovery_cursors WHERE kb_id = $1",
                        kb_id,
                    )
                    await conn.execute("DELETE FROM kb_jobs WHERE kb_id = $1", kb_id)
                    await conn.execute(
                        "DELETE FROM kb_documents WHERE kb_id = $1", kb_id
                    )
                    await conn.execute(
                        "DELETE FROM enterprise_kb_lifecycle WHERE kb_id = $1", kb_id
                    )
        finally:
            await peer.close()
            await store.close()


async def test_sqlite_count_active_jobs_globally_aggregates_across_kbs(
    tmp_path: Path,
) -> None:
    """SQLite ``count_active_jobs_globally`` rolls up jobs across ALL KBs.

    Focused regression for B-1: the migration online-mutation guard needs a
    cross-KB aggregate. The existing per-KB ``list_jobs`` was strictly scoped
    by ``kb_id`` and so could never see a mutation job seeded under a
    different KB — the guard silently passed. This test seeds jobs in several
    KBs with mixed statuses and asserts the unscoped count observes every
    active row regardless of its KB.
    """

    store = SQLiteMetadataStore(tmp_path / "metadata.sqlite3")
    await store.initialize()
    try:
        kb_a = "kb_global_a"
        kb_b = "kb_global_b"
        kb_c = "kb_global_c"
        kb_d = "kb_global_d"
        now = _NOW.isoformat()
        # Four different active statuses spread across four different KBs,
        # plus two terminal jobs that must NOT be counted.
        seeds = [
            (kb_a, "job-a-queued", "queued"),
            (kb_a, "job-a-running", "running"),
            (kb_a, "job-a-succeeded", "succeeded"),
            (kb_b, "job-b-cancelling", "cancelling"),
            (kb_c, "job-c-queued", "queued"),
            (kb_c, "job-c-failed", "failed"),
            (kb_d, "job-d-retrying", "retrying"),
        ]
        for kb_id, job_id, status in seeds:
            await store.create_job(
                replace(
                    _job(kb_id, job_id, 1),
                    status=status,
                    created_at=now,
                    updated_at=now,
                )
            )

        mutation_statuses = ["queued", "running", "retrying", "cancelling"]
        # 2 queued (KB-A, KB-C) + 1 running (KB-A)
        # + 1 retrying (KB-D) + 1 cancelling (KB-B) == 5.
        # The ``succeeded`` and ``failed`` jobs are excluded by status.
        assert await store.count_active_jobs_globally(mutation_statuses) == 5

        # Each individual active status rolls up across KBs too.
        assert await store.count_active_jobs_globally(["queued"]) == 2
        assert await store.count_active_jobs_globally(["running"]) == 1
        assert await store.count_active_jobs_globally(["retrying"]) == 1
        assert await store.count_active_jobs_globally(["cancelling"]) == 1

        # A status matching no row returns 0 (not an error).
        assert await store.count_active_jobs_globally(["nonexistent"]) == 0

        # Defensive input validation: a malformed query would silently
        # under-count and defeat the guard, so bad input must raise.
        with pytest.raises(ValueError):
            await store.count_active_jobs_globally([])
        with pytest.raises(ValueError):
            await store.count_active_jobs_globally(["running", 1])  # type: ignore[list-item]
    finally:
        await store.close()
