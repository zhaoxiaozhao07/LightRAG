from __future__ import annotations

import argparse
import asyncio
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from lightrag.api.artifact_cleanup_service import ArtifactCleanupService
from lightrag.api.artifact_lifecycle import (
    ArtifactCleanupManifestRecord,
    artifact_cleanup_idempotency_key,
)
from lightrag.api.config import (
    ArtifactCleanupConfig,
    artifact_cleanup_config_from_args,
    artifact_cleanup_config_from_values,
    configure_artifact_storage_args,
)
from lightrag.api.kb_service import sanitize_workspace
from lightrag.api.metadata_store import (
    ArtifactRecord,
    DocumentRecord,
    JobRecord,
    SQLiteMetadataStore,
)
from lightrag.api.object_storage import VerifiedDeleteResult
from tests.api.test_object_storage_s3 import _make_storage

pytestmark = pytest.mark.offline

_NOW = datetime(2026, 8, 3, 12, 0, 0, tzinfo=timezone.utc)
_KB_ID = "cleanup"
_GENERATION = "generation-1"
_WORKSPACE = sanitize_workspace(_KB_ID)


def _job(
    job_id: str = "job-origin",
    *,
    document_id: str | None = None,
    status: str = "succeeded",
    error_code: str | None = None,
    attempt_token: str = "attempt-1",
) -> JobRecord:
    timestamp = _NOW.isoformat()
    return JobRecord(
        id=job_id,
        kb_id=_KB_ID,
        workspace=_WORKSPACE,
        batch_id=None,
        document_id=document_id,
        job_type="replace",
        status=status,
        stage=None,
        progress=1.0 if status == "succeeded" else 0.0,
        total_items=1,
        completed_items=1 if status == "succeeded" else 0,
        failed_items=0,
        idempotency_key=f"idem-{job_id}",
        config_version_id=None,
        config_hash=None,
        retry_count=0,
        max_retries=3,
        payload={"attempt_token": attempt_token},
        result={},
        error_code=error_code,
        error_message=None,
        created_at=timestamp,
        updated_at=timestamp,
        queued_at=timestamp,
        started_at=timestamp if status != "queued" else None,
        finished_at=timestamp if status in {"succeeded", "failed"} else None,
        cancelled_at=None,
    )


def _document(
    document_id: str = "doc-1",
    *,
    metadata: dict[str, Any] | None = None,
) -> DocumentRecord:
    timestamp = _NOW.isoformat()
    return DocumentRecord(
        id=document_id,
        kb_id=_KB_ID,
        workspace=_WORKSPACE,
        lightrag_doc_id=f"lr-{document_id}",
        source_type="upload",
        source_name=f"{document_id}.bin",
        source_uri=f"/legacy/{document_id}.bin",
        source_hash="sha256:" + "1" * 64,
        content_type="application/octet-stream",
        size_bytes=4,
        parser_hash="parser-1",
        index_hash="index-1",
        status="ready",
        enabled=True,
        archived=False,
        chunks_count=1,
        entity_count=0,
        relation_count=0,
        error_code=None,
        error_message=None,
        metadata=dict(metadata or {}),
        created_at=timestamp,
        updated_at=timestamp,
        deleted_at=None,
    )


def _source_uri(
    document_id: str = "doc-1", source_generation_id: str = "srcg-old"
) -> str:
    return (
        f"s3://lightrag-kb/kb/workspaces/{_WORKSPACE}/documents/{document_id}/"
        f"source/generations/{source_generation_id}/{document_id}.bin"
    )


def _artifact_uri(document_id: str = "doc-1", artifact_id: str = "artifact-old") -> str:
    return (
        f"s3://lightrag-kb/kb/workspaces/{_WORKSPACE}/documents/{document_id}/"
        f"artifacts/raw/{artifact_id}/bundle.bin"
    )


def _artifact_prefix_uri(
    document_id: str = "doc-1", artifact_id: str = "artifact-old"
) -> str:
    return (
        f"s3://lightrag-kb/kb/workspaces/{_WORKSPACE}/documents/{document_id}/"
        f"artifacts/raw/{artifact_id}/"
    )


def _manifest(
    manifest_id: str = "manifest-1",
    *,
    reason: str = "replace",
    kb_generation: str = _GENERATION,
    workspace: str = _WORKSPACE,
    document_id: str | None = "doc-1",
    artifact_id: str | None = None,
    source_generation_id: str | None = "srcg-old",
    target_kind: str = "object",
    target_namespace: str = "source",
    target_uri: str | None = None,
    origin_job_id: str | None = "job-origin",
    origin_attempt_token: str | None = "attempt-1",
    status: str = "pending",
    disposition: str | None = None,
    expected_checksum: str | None = None,
) -> ArtifactCleanupManifestRecord:
    resolved_target = target_uri or _source_uri(
        document_id or "doc-1", source_generation_id or "srcg-old"
    )
    resolved_disposition = disposition or (
        "retain" if status == "retained" else "delete"
    )
    idempotency_key = artifact_cleanup_idempotency_key(
        reason=reason,  # type: ignore[arg-type]
        kb_id=_KB_ID,
        kb_generation=kb_generation,
        workspace=workspace,
        document_id=document_id,
        artifact_id=artifact_id,
        source_generation_id=source_generation_id,
        target_kind=target_kind,  # type: ignore[arg-type]
        target_namespace=target_namespace,  # type: ignore[arg-type]
        target_uri=resolved_target,
    )
    return ArtifactCleanupManifestRecord(
        id=manifest_id,
        idempotency_key=idempotency_key,
        manifest_group_id=f"group-{manifest_id}",
        kb_id=_KB_ID,
        kb_generation=kb_generation,
        workspace=workspace,
        document_id=document_id,
        artifact_id=artifact_id,
        source_generation_id=source_generation_id,
        origin_job_id=origin_job_id,
        origin_attempt_token=origin_attempt_token,
        reason=reason,  # type: ignore[arg-type]
        target_kind=target_kind,  # type: ignore[arg-type]
        target_namespace=target_namespace,  # type: ignore[arg-type]
        disposition=resolved_disposition,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        target_uri=resolved_target,
        expected_checksum=expected_checksum,
        delete_after=_NOW,
        cleanup_deadline_at=_NOW + timedelta(days=1),
        audit_retain_until=_NOW + timedelta(days=30),
        next_attempt_at=_NOW,
        created_at=_NOW,
        updated_at=_NOW,
    )


async def _store(
    tmp_path: Path,
    *,
    documents: list[DocumentRecord] | None = None,
    origin_job: JobRecord | None = None,
) -> SQLiteMetadataStore:
    store = SQLiteMetadataStore(tmp_path / "metadata.sqlite3")
    await store.initialize()
    await store.activate_kb_generation(
        _KB_ID, _GENERATION, activated_at=_NOW.isoformat()
    )
    docs = list(documents or [])
    job = origin_job or _job(document_id=None)
    await store.create_documents_and_job(docs, job)
    return store


def _config(**overrides: Any) -> ArtifactCleanupConfig:
    values = {
        "claim_limit": 16,
        "max_concurrent_manifests": 4,
        "backoff_base_seconds": 5,
        "backoff_max_seconds": 20,
        "object_page_size": 2,
        "delete_batch_size": 2,
        "max_prefix_pages_per_manifest_attempt": 8,
    }
    values.update(overrides)
    return artifact_cleanup_config_from_values(**values)


def _put_manifest_object(state, manifest: ArtifactCleanupManifestRecord) -> str:
    key = manifest.target_uri.split("lightrag-kb/", 1)[1].rstrip("/")
    state.objects[("lightrag-kb", key)] = b"old"
    return key


async def test_obsolete_immutable_generation_cleans_while_document_is_live(
    tmp_path: Path,
):
    current_uri = _source_uri(source_generation_id="srcg-current")
    store = await _store(
        tmp_path,
        documents=[
            _document(
                metadata={
                    "current_source_generation_id": "srcg-current",
                    "source_object_uri": current_uri,
                    "last_replace_job_id": "job-origin",
                }
            )
        ],
    )
    manifest = _manifest()
    await store.enqueue_artifact_cleanup_manifest(manifest)
    storage, state, _ = _make_storage()
    old_key = _put_manifest_object(state, manifest)

    summary = await ArtifactCleanupService(
        store, storage, _config(), clock=lambda: _NOW
    ).run_once(_NOW, "worker-1")

    assert summary.succeeded == 1
    assert summary.to_dict()["outcomes"][0]["error_code"] == (
        "artifact_cleanup_succeeded"
    )
    assert ("lightrag-kb", old_key) not in state.objects
    stored = await store.get_artifact_cleanup_manifest(manifest.id)
    assert stored.status == "succeeded"
    assert datetime.fromisoformat(str(stored.audit_retain_until)) >= (
        datetime.fromisoformat(str(stored.completed_at)) + timedelta(days=30)
    )


@pytest.mark.parametrize(
    (
        "metadata",
        "namespace",
        "artifact_id",
        "source_generation_id",
        "target_uri",
        "code",
    ),
    [
        (
            {
                "current_source_generation_id": "srcg-old",
                "source_object_uri": _source_uri(),
                "last_replace_job_id": "job-origin",
            },
            "source",
            None,
            "srcg-old",
            _source_uri(),
            "current_source_reference",
        ),
        (
            {
                "current_sidecar_artifact_id": "artifact-old",
                "last_replace_job_id": "job-origin",
            },
            "artifact",
            "artifact-old",
            None,
            _artifact_uri(),
            "current_artifact_reference",
        ),
    ],
)
async def test_exact_current_source_or_artifact_reference_blocks(
    tmp_path: Path,
    metadata,
    namespace,
    artifact_id,
    source_generation_id,
    target_uri,
    code,
):
    store = await _store(tmp_path, documents=[_document(metadata=metadata)])
    manifest = _manifest(
        target_namespace=namespace,
        artifact_id=artifact_id,
        source_generation_id=source_generation_id,
        target_uri=target_uri,
    )
    await store.enqueue_artifact_cleanup_manifest(manifest)
    storage, state, _ = _make_storage()
    key = _put_manifest_object(state, manifest)

    summary = await ArtifactCleanupService(
        store, storage, _config(), clock=lambda: _NOW
    ).run_once(_NOW, "worker-1")

    assert summary.blocked == 1
    assert summary.outcomes[0].error_code == code
    assert ("lightrag-kb", key) in state.objects


async def test_exact_artifact_row_reference_blocks(tmp_path: Path):
    store = await _store(
        tmp_path,
        documents=[_document(metadata={"last_replace_job_id": "job-origin"})],
    )
    artifact = ArtifactRecord(
        id="artifact-row",
        kb_id=_KB_ID,
        workspace=_WORKSPACE,
        document_id="doc-1",
        artifact_type="raw",
        uri="/compat/local",
        checksum="sha256:" + "2" * 64,
        size_bytes=3,
        metadata={"object_uri": _artifact_uri(artifact_id="artifact-row")},
        created_at=_NOW.isoformat(),
    )
    await store.complete_document_parse(
        _KB_ID,
        "doc-1",
        parser_hash="parser-2",
        lightrag_doc_id="lr-doc-1",
        metadata_patch={},
        artifacts=[artifact],
    )
    manifest = _manifest(
        artifact_id="artifact-row",
        source_generation_id=None,
        target_namespace="artifact",
        target_uri=_artifact_uri(artifact_id="artifact-row"),
    )
    await store.enqueue_artifact_cleanup_manifest(manifest)
    storage, _, _ = _make_storage()

    summary = await ArtifactCleanupService(
        store, storage, _config(), clock=lambda: _NOW
    ).run_once(_NOW, "worker-1")

    assert summary.blocked == 1
    assert summary.outcomes[0].error_code == "current_artifact_uri_reference"


async def test_active_job_defers_with_capped_exponential_backoff(tmp_path: Path):
    store = await _store(
        tmp_path,
        documents=[_document(metadata={"last_replace_job_id": "job-origin"})],
    )
    await store.create_job(_job("job-active", document_id="doc-1", status="running"))
    manifest = replace(_manifest(), attempt_count=2)
    await store.enqueue_artifact_cleanup_manifest(manifest)
    storage, state, _ = _make_storage()
    key = _put_manifest_object(state, manifest)

    summary = await ArtifactCleanupService(
        store, storage, _config(), clock=lambda: _NOW
    ).run_once(_NOW, "worker-1")

    assert summary.retried == 1
    assert summary.outcomes[0].error_code == "document_job_active"
    pending = await store.get_artifact_cleanup_manifest(manifest.id)
    assert pending.status == "pending"
    assert datetime.fromisoformat(str(pending.next_attempt_at)) == _NOW + timedelta(
        seconds=20
    )
    assert ("lightrag-kb", key) in state.objects


async def test_unknown_origin_outcome_and_attempt_mismatch_block(tmp_path: Path):
    for suffix, job, expected_code in (
        (
            "unknown",
            _job(error_code="metadata_commit_outcome_unknown"),
            "metadata_commit_outcome_unknown",
        ),
        (
            "attempt",
            _job(attempt_token="different-attempt"),
            "origin_attempt_lineage_mismatch",
        ),
    ):
        case = tmp_path / suffix
        case.mkdir()
        store = await _store(
            case,
            documents=[_document(metadata={"last_replace_job_id": "job-origin"})],
            origin_job=job,
        )
        manifest = _manifest(manifest_id=f"manifest-{suffix}")
        await store.enqueue_artifact_cleanup_manifest(manifest)
        storage, _, _ = _make_storage()
        summary = await ArtifactCleanupService(
            store, storage, _config(), clock=lambda: _NOW
        ).run_once(_NOW, "worker-1")
        assert summary.blocked == 1
        assert summary.outcomes[0].error_code == expected_code


@pytest.mark.parametrize(
    ("manifest", "expected_code"),
    [
        (_manifest(kb_generation="stale-generation"), "kb_generation_stale"),
        (
            _manifest(
                workspace="kb_other",
                target_uri=(
                    "s3://lightrag-kb/kb/workspaces/kb_other/documents/doc-1/"
                    "source/generations/srcg-old/doc-1.bin"
                ),
            ),
            "cleanup_workspace_mismatch",
        ),
        (
            _manifest(
                target_uri=(
                    f"s3://lightrag-kb/kb/workspaces/{_WORKSPACE}/documents/"
                    "doc-1/not-a-cleanup-namespace/file.bin"
                )
            ),
            "object_ownership_conflict",
        ),
    ],
)
async def test_stale_generation_workspace_and_malformed_target_block(
    tmp_path: Path, manifest, expected_code
):
    store = await _store(
        tmp_path,
        documents=[_document(metadata={"last_replace_job_id": "job-origin"})],
    )
    await store.enqueue_artifact_cleanup_manifest(manifest)
    storage, _, _ = _make_storage()
    summary = await ArtifactCleanupService(
        store, storage, _config(), clock=lambda: _NOW
    ).run_once(_NOW, "worker-1")
    assert summary.blocked == 1
    assert summary.outcomes[0].error_code == expected_code


async def test_missing_active_document_blocks(tmp_path: Path):
    store = await _store(tmp_path, documents=[])
    manifest = _manifest()
    await store.enqueue_artifact_cleanup_manifest(manifest)
    storage, _, _ = _make_storage()

    summary = await ArtifactCleanupService(
        store, storage, _config(), clock=lambda: _NOW
    ).run_once(_NOW, "worker-1")

    assert summary.blocked == 1
    assert summary.outcomes[0].error_code == "document_lifecycle_missing"


async def test_matching_document_tombstone_authority_allows_cleanup(tmp_path: Path):
    store = await _store(tmp_path, documents=[_document()])
    await store.complete_document_delete(
        _KB_ID,
        "doc-1",
        metadata_patch={
            "last_delete_job_id": "job-origin",
            "last_delete_attempt_token": "attempt-1",
        },
    )
    manifest = _manifest(reason="document_delete")
    await store.enqueue_artifact_cleanup_manifest(manifest)
    storage, state, _ = _make_storage()
    _put_manifest_object(state, manifest)

    summary = await ArtifactCleanupService(
        store, storage, _config(), clock=lambda: _NOW
    ).run_once(_NOW, "worker-1")

    assert summary.succeeded == 1


async def test_non_delete_manifest_cannot_use_document_tombstone(tmp_path: Path):
    store = await _store(tmp_path, documents=[_document()])
    await store.complete_document_delete(
        _KB_ID,
        "doc-1",
        metadata_patch={"last_delete_job_id": "job-origin"},
    )
    manifest = _manifest(reason="replace")
    await store.enqueue_artifact_cleanup_manifest(manifest)
    storage, _, _ = _make_storage()

    summary = await ArtifactCleanupService(
        store, storage, _config(), clock=lambda: _NOW
    ).run_once(_NOW, "worker-1")

    assert summary.blocked == 1
    assert summary.outcomes[0].error_code == ("document_tombstone_authority_mismatch")


async def test_retained_same_target_manifest_blocks(tmp_path: Path):
    store = await _store(
        tmp_path,
        documents=[_document(metadata={"last_replace_job_id": "job-origin"})],
    )
    retained = _manifest(
        manifest_id="manifest-retained",
        reason="orphan_reconcile",
        status="retained",
    )
    pending = _manifest(manifest_id="manifest-pending")
    await store.enqueue_artifact_cleanup_manifests([retained, pending])
    storage, _, _ = _make_storage()

    summary = await ArtifactCleanupService(
        store, storage, _config(), clock=lambda: _NOW
    ).run_once(_NOW, "worker-1")

    assert summary.blocked == 1
    assert summary.outcomes[0].error_code == "same_target_manifest_conflict"


async def test_competing_same_target_lease_defers_deterministic_loser(tmp_path: Path):
    store = await _store(
        tmp_path,
        documents=[_document(metadata={"last_replace_job_id": "job-origin"})],
    )
    winner = _manifest(manifest_id="a-winner", reason="orphan_reconcile")
    loser = _manifest(manifest_id="b-loser")
    await store.enqueue_artifact_cleanup_manifest(winner)
    leased = await store.claim_due_artifact_cleanup_manifests(
        lease_owner="other-worker", now=_NOW, limit=1
    )
    assert leased[0].id == winner.id
    await store.enqueue_artifact_cleanup_manifest(loser)
    storage, _, _ = _make_storage()

    summary = await ArtifactCleanupService(
        store, storage, _config(), clock=lambda: _NOW
    ).run_once(_NOW, "worker-1")

    assert summary.retried == 1
    assert summary.outcomes[0].error_code == "same_target_competing_lease"


async def test_succeeded_same_target_is_confirmed_by_readback(tmp_path: Path):
    store = await _store(
        tmp_path,
        documents=[
            _document(
                metadata={
                    "last_replace_job_id": "job-origin",
                    "last_replace_attempt_token": "attempt-1",
                    "replace_attempt_token_history": ["attempt-1"],
                }
            )
        ],
    )
    first = _manifest(manifest_id="manifest-first")
    await store.enqueue_artifact_cleanup_manifest(first)
    storage, state, _ = _make_storage()
    _put_manifest_object(state, first)
    service = ArtifactCleanupService(store, storage, _config(), clock=lambda: _NOW)
    assert (await service.run_once(_NOW, "worker-1")).succeeded == 1

    second = _manifest(manifest_id="manifest-second", reason="orphan_reconcile")
    await store.enqueue_artifact_cleanup_manifest(second)
    summary = await service.run_once(_NOW + timedelta(seconds=1), "worker-2")
    assert summary.succeeded == 1
    assert (await store.get_artifact_cleanup_manifest(second.id)).status == (
        "succeeded"
    )


async def test_expired_worker_lease_is_recovered_and_reprocessed(tmp_path: Path):
    store = await _store(
        tmp_path,
        documents=[_document(metadata={"last_replace_job_id": "job-origin"})],
    )
    manifest = _manifest()
    await store.enqueue_artifact_cleanup_manifest(manifest)
    await store.claim_due_artifact_cleanup_manifests(
        lease_owner="crashed-worker",
        lease_duration_seconds=1,
        limit=1,
        now=_NOW,
    )
    storage, state, _ = _make_storage()
    _put_manifest_object(state, manifest)

    summary = await ArtifactCleanupService(
        store, storage, _config(), clock=lambda: _NOW
    ).run_once(_NOW + timedelta(seconds=2), "recovery-worker")

    assert summary.recovered_leases == 1
    assert summary.succeeded == 1


async def test_duplicate_service_claims_have_one_winner(tmp_path: Path):
    db_path = tmp_path / "metadata.sqlite3"
    first_store = SQLiteMetadataStore(db_path)
    second_store = SQLiteMetadataStore(db_path)
    await first_store.initialize()
    await second_store.initialize()
    await first_store.activate_kb_generation(
        _KB_ID, _GENERATION, activated_at=_NOW.isoformat()
    )
    await first_store.create_documents_and_job(
        [_document(metadata={"last_replace_job_id": "job-origin"})], _job()
    )
    manifest = _manifest()
    await first_store.enqueue_artifact_cleanup_manifest(manifest)
    storage, state, _ = _make_storage()
    _put_manifest_object(state, manifest)

    left, right = await asyncio.gather(
        ArtifactCleanupService(
            first_store, storage, _config(), clock=lambda: _NOW
        ).run_once(_NOW, "worker-left"),
        ArtifactCleanupService(
            second_store, storage, _config(), clock=lambda: _NOW
        ).run_once(_NOW, "worker-right"),
    )

    assert left.claimed_manifests + right.claimed_manifests == 1
    assert left.succeeded + right.succeeded == 1


async def test_stale_lease_completion_cannot_overwrite_new_claim(tmp_path: Path):
    store = await _store(
        tmp_path,
        documents=[_document(metadata={"last_replace_job_id": "job-origin"})],
    )
    manifest = _manifest()
    await store.enqueue_artifact_cleanup_manifest(manifest)
    storage, _, _ = _make_storage()

    async def steal_lease(_target, **_kwargs):
        await store.recover_expired_artifact_cleanup_manifest_leases(
            now=_NOW + timedelta(minutes=2),
            next_attempt_at=_NOW + timedelta(minutes=2),
            limit=1,
        )
        claimed = await store.claim_due_artifact_cleanup_manifests(
            lease_owner="new-worker",
            now=_NOW + timedelta(minutes=2),
            limit=1,
        )
        assert claimed and claimed[0].lease_owner == "new-worker"
        return VerifiedDeleteResult(
            absent=True,
            already_absent=True,
            deleted_entries=0,
            pages_examined=0,
            version_aware=False,
        )

    storage.verified_delete_cleanup_target = steal_lease  # type: ignore[method-assign]
    summary = await ArtifactCleanupService(
        store, storage, _config(), clock=lambda: _NOW
    ).run_once(_NOW, "old-worker")

    assert summary.stale_leases == 1
    current = await store.get_artifact_cleanup_manifest(manifest.id)
    assert current.status == "leased"
    assert current.lease_owner == "new-worker"


async def test_prefix_cleanup_renews_lease_between_bounded_pages(tmp_path: Path):
    store = await _store(
        tmp_path,
        documents=[_document(metadata={"last_replace_job_id": "job-origin"})],
    )
    manifest = _manifest(
        artifact_id="artifact-old",
        source_generation_id=None,
        target_kind="prefix",
        target_namespace="artifact",
        target_uri=_artifact_prefix_uri(),
    )
    await store.enqueue_artifact_cleanup_manifest(manifest)
    storage, state, _ = _make_storage(page_size=1)
    prefix_key = manifest.target_uri.split("lightrag-kb/", 1)[1]
    for index in range(3):
        state.objects[("lightrag-kb", f"{prefix_key}{index}.bin")] = b"x"
    renewals = 0
    original_renew = store.renew_artifact_cleanup_manifest_lease

    async def counted_renew(*args, **kwargs):
        nonlocal renewals
        renewals += 1
        return await original_renew(*args, **kwargs)

    store.renew_artifact_cleanup_manifest_lease = counted_renew  # type: ignore[method-assign]
    summary = await ArtifactCleanupService(
        store,
        storage,
        _config(object_page_size=1, delete_batch_size=1),
        clock=lambda: _NOW,
    ).run_once(_NOW, "worker-1")

    assert summary.succeeded == 1
    assert renewals >= 5  # pre-side-effect + each destructive/final proof page


async def test_prefix_page_budget_maps_to_fenced_retry(tmp_path: Path):
    store = await _store(
        tmp_path,
        documents=[_document(metadata={"last_replace_job_id": "job-origin"})],
    )
    manifest = _manifest(
        artifact_id="artifact-old",
        source_generation_id=None,
        target_kind="prefix",
        target_namespace="artifact",
        target_uri=_artifact_prefix_uri(),
    )
    await store.enqueue_artifact_cleanup_manifest(manifest)
    storage, state, _ = _make_storage(page_size=1)
    prefix_key = manifest.target_uri.split("lightrag-kb/", 1)[1]
    for index in range(2):
        state.objects[("lightrag-kb", f"{prefix_key}{index}.bin")] = b"x"

    summary = await ArtifactCleanupService(
        store,
        storage,
        _config(
            object_page_size=1,
            delete_batch_size=1,
            max_prefix_pages_per_manifest_attempt=1,
        ),
        clock=lambda: _NOW,
    ).run_once(_NOW, "worker-1")

    assert summary.retried == 1
    assert summary.outcomes[0].error_code == "object_prefix_page_budget"
    assert (await store.get_artifact_cleanup_manifest(manifest.id)).status == "pending"


async def test_concurrency_is_semaphore_bounded(tmp_path: Path):
    documents = [
        _document(f"doc-{index}", metadata={"last_replace_job_id": "job-origin"})
        for index in range(4)
    ]
    store = await _store(tmp_path, documents=documents)
    manifests = [
        _manifest(
            manifest_id=f"manifest-{index}",
            document_id=f"doc-{index}",
            target_uri=_source_uri(f"doc-{index}"),
        )
        for index in range(4)
    ]
    await store.enqueue_artifact_cleanup_manifests(manifests)
    storage, state, _ = _make_storage()
    for manifest in manifests:
        _put_manifest_object(state, manifest)
    original_delete = storage.verified_delete_cleanup_target
    active = 0
    maximum = 0

    async def delayed_delete(target, **kwargs):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        try:
            await asyncio.sleep(0.01)
            return await original_delete(target, **kwargs)
        finally:
            active -= 1

    storage.verified_delete_cleanup_target = delayed_delete  # type: ignore[method-assign]
    summary = await ArtifactCleanupService(
        store,
        storage,
        _config(max_concurrent_manifests=2),
        clock=lambda: _NOW,
    ).run_once(_NOW, "worker-1")

    assert summary.succeeded == 4
    assert maximum == 2


async def test_kb_delete_workspace_prefix_may_run_while_lifecycle_is_deleting(
    tmp_path: Path,
):
    store = await _store(
        tmp_path,
        documents=[],
        origin_job=_job(status="running"),
    )
    await store.begin_kb_deletion(_KB_ID, _GENERATION, "job-origin")
    target_uri = f"s3://lightrag-kb/kb/workspaces/{_WORKSPACE}/"
    manifest = _manifest(
        reason="kb_delete",
        document_id=None,
        source_generation_id=None,
        target_kind="prefix",
        target_namespace="workspace",
        target_uri=target_uri,
        origin_attempt_token=None,
    )
    await store.enqueue_artifact_cleanup_manifest(manifest)
    storage, state, _ = _make_storage(page_size=1)
    state.objects[("lightrag-kb", f"kb/workspaces/{_WORKSPACE}/a.bin")] = b"a"

    summary = await ArtifactCleanupService(
        store, storage, _config(), clock=lambda: _NOW
    ).run_once(_NOW, "worker-1")

    assert summary.succeeded == 1


async def test_kb_delete_workspace_prefix_cannot_run_while_lifecycle_is_active(
    tmp_path: Path,
):
    store = await _store(tmp_path, documents=[])
    target_uri = f"s3://lightrag-kb/kb/workspaces/{_WORKSPACE}/"
    manifest = _manifest(
        reason="kb_delete",
        document_id=None,
        source_generation_id=None,
        target_kind="prefix",
        target_namespace="workspace",
        target_uri=target_uri,
        origin_attempt_token=None,
    )
    await store.enqueue_artifact_cleanup_manifest(manifest)
    storage, state, _ = _make_storage()
    key = f"kb/workspaces/{_WORKSPACE}/a.bin"
    state.objects[("lightrag-kb", key)] = b"a"

    summary = await ArtifactCleanupService(
        store, storage, _config(), clock=lambda: _NOW
    ).run_once(_NOW, "worker-1")

    assert summary.blocked == 1
    assert summary.outcomes[0].error_code == "kb_delete_lifecycle_not_deleting"
    assert ("lightrag-kb", key) in state.objects


async def test_service_outputs_and_persisted_errors_are_durable_safe(tmp_path: Path):
    store = await _store(
        tmp_path,
        documents=[_document(metadata={"last_replace_job_id": "job-origin"})],
    )
    manifest = _manifest()
    await store.enqueue_artifact_cleanup_manifest(manifest)
    storage, _, _ = _make_storage()

    async def unsafe_failure(_target, **_kwargs):
        raise RuntimeError(
            "AWS_SECRET_ACCESS_KEY=secret s3://user:pass@bucket/key?X-Amz-Signature=x"
        )

    storage.verified_delete_cleanup_target = unsafe_failure  # type: ignore[method-assign]
    summary = await ArtifactCleanupService(
        store, storage, _config(), clock=lambda: _NOW
    ).run_once(_NOW, "worker-1")
    serialized = str(summary.to_dict())
    assert "secret" not in serialized
    assert "X-Amz" not in serialized
    assert "s3://" not in serialized
    assert summary.retried == 1
    stored = await store.get_artifact_cleanup_manifest(manifest.id)
    assert stored.last_error_code == "artifact_cleanup_internal_error"


def test_cleanup_config_defaults_floor_and_frozen_contract():
    config = ArtifactCleanupConfig()
    assert config.replacement_grace_seconds == 24 * 60 * 60
    assert config.staging_grace_seconds == 24 * 60 * 60
    assert config.cleanup_slo_seconds == 24 * 60 * 60
    assert config.successful_audit_retention_days == 30
    assert config.lease_duration_seconds == 60
    assert config.max_concurrent_manifests <= 8
    assert config.claim_limit <= 500
    assert config.object_page_size <= 1000
    assert config.delete_batch_size <= 1000
    assert config.expired_lease_recovery_limit <= 500
    with pytest.raises(FrozenInstanceError):
        config.claim_limit = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"successful_audit_retention_days": 29},
        {"lease_duration_seconds": 0},
        {"max_concurrent_manifests": True},
        {"claim_limit": 501},
        {"object_page_size": 1001},
        {"delete_batch_size": 0},
        {"max_prefix_pages_per_manifest_attempt": 0},
        {"expired_lease_recovery_limit": 501},
        {"backoff_base_seconds": 20, "backoff_max_seconds": 10},
    ],
)
def test_cleanup_config_rejects_unsafe_bounds(overrides):
    with pytest.raises(ValueError):
        artifact_cleanup_config_from_values(**overrides)


def test_configure_artifact_storage_args_normalizes_cleanup_values(monkeypatch):
    monkeypatch.setenv("LIGHTRAG_ARTIFACT_CLEANUP_CLAIM_LIMIT", "7")
    monkeypatch.setenv("LIGHTRAG_ARTIFACT_CLEANUP_LEASE_DURATION_SECONDS", "45.5")
    args = argparse.Namespace()

    configure_artifact_storage_args(args)
    config = artifact_cleanup_config_from_args(args)

    assert args.artifact_cleanup_claim_limit == 7
    assert args.artifact_cleanup_lease_duration_seconds == 45.5
    assert config.claim_limit == 7
    assert config.lease_duration_seconds == 45.5


# ---------------------------------------------------------------------------
# Phase 3.1-C Section F regression: cleanup authority lineage correction.
#
# The singular ``last_replace_job_id`` must not invalidate older replace
# manifests during the grace window. A replace/orphan_reconcile manifest whose
# origin attempt token is still in the durable attempt-token history remains
# authorized even after a newer replace has advanced ``last_replace_job_id``.
# ---------------------------------------------------------------------------


async def test_older_replace_manifest_still_authorized_after_newer_replace_commits(
    tmp_path: Path,
):
    """A's manifest cleans after B commits because attempt-a is in history."""

    job_a = _job(
        job_id="job-a",
        document_id="doc-1",
        status="succeeded",
        attempt_token="attempt-a",
    )
    store = await _store(
        tmp_path,
        documents=[
            _document(
                metadata={
                    "current_source_generation_id": "srcg-b",
                    "source_object_uri": _source_uri(source_generation_id="srcg-b"),
                    "last_replace_job_id": "job-b",
                    "last_replace_attempt_token": "attempt-b",
                    "replace_attempt_token_history": ["attempt-a", "attempt-b"],
                    "current_replace_job_id": "job-b",
                    "current_replace_claim_token": "attempt-b",
                }
            )
        ],
        origin_job=job_a,
    )
    # A's manifest targets the OLD source (srcg-old), authorized by attempt-a.
    manifest = _manifest(
        manifest_id="manifest-a-old",
        source_generation_id="srcg-old",
        target_uri=_source_uri(source_generation_id="srcg-old"),
        origin_job_id="job-a",
        origin_attempt_token="attempt-a",
    )
    await store.enqueue_artifact_cleanup_manifest(manifest)
    storage, state, _ = _make_storage()
    key = _put_manifest_object(state, manifest)

    summary = await ArtifactCleanupService(
        store, storage, _config(), clock=lambda: _NOW
    ).run_once(_NOW, "worker-1")

    # A's old-target manifest is authorized by the durable history and the
    # current source (srcg-b) is different, so cleanup succeeds.
    assert summary.succeeded == 1
    assert ("lightrag-kb", key) not in state.objects


async def test_newer_current_source_blocks_cleanup_of_current_pointer(
    tmp_path: Path,
):
    """A manifest targeting the current source is blocked even with history."""

    current_uri = _source_uri(source_generation_id="srcg-current")
    store = await _store(
        tmp_path,
        documents=[
            _document(
                metadata={
                    "current_source_generation_id": "srcg-current",
                    "source_object_uri": current_uri,
                    "last_replace_job_id": "job-origin",
                    "last_replace_attempt_token": "attempt-1",
                    "replace_attempt_token_history": ["attempt-1"],
                }
            )
        ],
    )
    # This manifest targets the CURRENT source — must be blocked regardless of
    # attempt-token authorization.
    manifest = _manifest(
        manifest_id="manifest-current",
        source_generation_id="srcg-current",
        target_uri=current_uri,
        origin_job_id="job-origin",
        origin_attempt_token="attempt-1",
    )
    await store.enqueue_artifact_cleanup_manifest(manifest)
    storage, state, _ = _make_storage()
    key = _put_manifest_object(state, manifest)

    summary = await ArtifactCleanupService(
        store, storage, _config(), clock=lambda: _NOW
    ).run_once(_NOW, "worker-1")

    assert summary.blocked == 1
    assert summary.outcomes[0].error_code == "current_source_reference"
    assert ("lightrag-kb", key) in state.objects


async def test_orphan_reconcile_authorized_by_failed_replace_attempt_token(
    tmp_path: Path,
):
    """orphan_reconcile/source cleanup is authorized by last_failed_replace_attempt_token."""

    job_failed = _job(
        job_id="job-failed",
        document_id="doc-1",
        status="failed",
        attempt_token="attempt-failed",
    )
    store = await _store(
        tmp_path,
        documents=[
            _document(
                metadata={
                    "current_source_generation_id": "srcg-current",
                    "source_object_uri": _source_uri(source_generation_id="srcg-current"),
                    "last_replace_job_id": "job-ok",
                    "last_replace_attempt_token": "attempt-ok",
                    "last_failed_replace_attempt_token": "attempt-failed",
                    "replace_attempt_token_history": ["attempt-failed", "attempt-ok"],
                }
            )
        ],
        origin_job=job_failed,
    )
    # Candidate from a rolled-back replace (attempt-failed).
    candidate_uri = _source_uri(source_generation_id="srcg-failed")
    manifest = _manifest(
        manifest_id="manifest-orphan",
        reason="orphan_reconcile",
        source_generation_id="srcg-failed",
        target_uri=candidate_uri,
        origin_job_id="job-failed",
        origin_attempt_token="attempt-failed",
    )
    await store.enqueue_artifact_cleanup_manifest(manifest)
    storage, state, _ = _make_storage()
    key = _put_manifest_object(state, manifest)

    summary = await ArtifactCleanupService(
        store, storage, _config(), clock=lambda: _NOW
    ).run_once(_NOW, "worker-1")

    # Authorized by last_failed_replace_attempt_token; current source preserved.
    assert summary.succeeded == 1
    assert ("lightrag-kb", key) not in state.objects


async def test_replace_manifest_blocks_when_attempt_token_not_in_history(
    tmp_path: Path,
):
    """A replace manifest whose token is absent from history is blocked."""

    store = await _store(
        tmp_path,
        documents=[
            _document(
                metadata={
                    "last_replace_job_id": "job-origin",
                    "last_replace_attempt_token": "attempt-recorded",
                    "replace_attempt_token_history": ["attempt-recorded"],
                }
            )
        ],
    )
    manifest = _manifest(
        manifest_id="manifest-foreign",
        origin_job_id="job-origin",
        origin_attempt_token="attempt-foreign",
    )
    await store.enqueue_artifact_cleanup_manifest(manifest)
    storage, state, _ = _make_storage()
    _put_manifest_object(state, manifest)

    summary = await ArtifactCleanupService(
        store, storage, _config(), clock=lambda: _NOW
    ).run_once(_NOW, "worker-1")

    assert summary.blocked == 1
    assert summary.outcomes[0].error_code == "origin_attempt_lineage_mismatch"
