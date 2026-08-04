from __future__ import annotations

import inspect
import os
import uuid
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from lightrag.api.artifact_lifecycle import (
    ArtifactCleanupManifestRecord,
    ArtifactLifecycleConflictError,
    artifact_cleanup_idempotency_key,
)
from lightrag.api.commit_reconciliation import MetadataCommitOutcome
from lightrag.api.kb_service import utc_now_iso
from lightrag.api.metadata_store import (
    ArtifactRecord,
    DocumentAttemptOwnershipError,
    DocumentMutationClaimResult,
    DocumentRecord,
    JobRecord,
    KBLifecycleConflictError,
    MetadataConflictError,
    MetadataStoreError,
    SQLiteMetadataStore,
    document_mutation_manifest_group_id,
    document_mutation_snapshot,
    document_source_generation_id,
)
from lightrag.api.postgres_metadata_store import PostgresMetadataStore

pytestmark = pytest.mark.offline

_POSTGRES_DSN = os.getenv("LIGHTRAG_KB_POSTGRES_TEST_DSN") or os.getenv(
    "POSTGRES_TEST_DSN"
)
_NEW_SOURCE_HASH = "sha256:new-source"
_DEFAULT_SYNC_ITEMS = object()


@pytest.fixture(params=["sqlite", "postgres"])
async def cow_store(request, tmp_path):
    backend = request.param
    if backend == "postgres" and not _POSTGRES_DSN:
        pytest.skip(
            "live PostgreSQL COW test skipped: set "
            "LIGHTRAG_KB_POSTGRES_TEST_DSN to enable"
        )
    if backend == "sqlite":
        store: Any = SQLiteMetadataStore(tmp_path / "cow.sqlite3")
    else:
        store = PostgresMetadataStore(
            dsn=_POSTGRES_DSN,
            min_size=1,
            max_size=1,
            operation_lock_pool_max_size=4,
        )
    await store.initialize()
    store._cow_backend = backend
    store._cow_kb_ids = []
    store._cow_generations = {}
    # Record every activated generation so the PostgreSQL teardown can purge
    # with the matching generation (the lifecycle-correct cleanup path). The
    # wrapper is fully transparent (forwards ``*args``/``**kwargs``, returns
    # the lifecycle record) so it also captures tests that activate a
    # generation directly instead of going through ``_seed``/``_seed_sync``,
    # as well as any internal caller (e.g. ``register_kb_generation``).
    _original_activate_generation = store.activate_kb_generation

    async def _tracking_activate_generation(*args: Any, **kwargs: Any) -> Any:
        record = await _original_activate_generation(*args, **kwargs)
        kb_id = getattr(record, "kb_id", None)
        if isinstance(kb_id, str):
            store._cow_generations[kb_id] = str(getattr(record, "generation"))
        return record

    store.activate_kb_generation = _tracking_activate_generation
    try:
        yield store
    finally:
        if backend == "postgres":
            for kb_id in store._cow_kb_ids:
                generation = store._cow_generations.get(kb_id)
                try:
                    await store.purge_kb_metadata(kb_id, generation=generation)
                except Exception:
                    pass
                # ``purge_kb_metadata`` may still raise (lifecycle stuck in
                # ``deleting``, or a missing/mismatched generation for a KB
                # whose activation did not round-trip through the wrapper) and
                # it does NOT delete ``kb_artifact_cleanup_manifests``. The
                # tables below all carry a GLOBAL primary key and these tests
                # reuse hardcoded ids across distinct KBs, so unconditionally
                # guarantee every row owned by this KB is removed to prevent
                # cross-test collisions on the shared PostgreSQL database.
                try:
                    await _force_purge_kb_rows(store, kb_id)
                except Exception:
                    pass
        await store.close()


async def _force_purge_kb_rows(store: Any, kb_id: str) -> None:
    """Test-only PostgreSQL cleanup: delete every row owned by the KB,
    bypassing lifecycle state checks. Mirrors ``purge_kb_metadata`` for the
    global-PK tables tests insert into (documents, jobs, artifacts) and
    additionally drops ``kb_artifact_cleanup_manifests`` rows, which the
    production purge leaves in place. This is the guarantee that hardcoded
    ids (``document-1``, ``job-1``, ``artifact-document-1``,
    ``manifest-replace-source-document-1``, ...) never leak across tests on
    the shared PostgreSQL backend.
    """

    async def write(conn: Any) -> None:
        await conn.execute(
            "DELETE FROM kb_artifact_cleanup_manifests WHERE kb_id = $1",
            kb_id,
        )
        await conn.execute("DELETE FROM kb_document_artifacts WHERE kb_id = $1", kb_id)
        await conn.execute("DELETE FROM kb_jobs WHERE kb_id = $1", kb_id)
        await conn.execute("DELETE FROM kb_documents WHERE kb_id = $1", kb_id)

    await store._write(write)


def _kb_id(store: Any) -> str:
    kb_id = f"kb_cow_{uuid.uuid4().hex[:12]}"
    store._cow_kb_ids.append(kb_id)
    return kb_id


def _document(kb_id: str, document_id: str) -> DocumentRecord:
    now = utc_now_iso()
    workspace = f"ws-{kb_id}"
    source_uri = (
        f"s3://cow-bucket/kb/{workspace}/{document_id}/source/"
        "generations/old-generation/source.pdf"
    )
    return DocumentRecord(
        id=document_id,
        kb_id=kb_id,
        workspace=workspace,
        lightrag_doc_id=f"engine-{document_id}",
        source_type="upload",
        source_name="source.pdf",
        source_uri=source_uri,
        source_hash="sha256:old-source",
        content_type="application/pdf",
        size_bytes=101,
        parser_hash="parser-old",
        index_hash="index-old",
        status="ready",
        enabled=True,
        archived=False,
        chunks_count=7,
        entity_count=5,
        relation_count=3,
        error_code=None,
        error_message=None,
        metadata={
            "source_key": f"source-key-{document_id}",
            "source_object_uri": source_uri,
            "source_generation_id": "old-generation",
            "current_sidecar_artifact_id": f"artifact-{document_id}",
            "current_artifact_ids": [f"artifact-{document_id}"],
            "artifact_binding": {
                "version": 1,
                "state": "committed",
                "document_id": document_id,
                "sidecar_artifact_id": f"artifact-{document_id}",
                "parse_generation_id": "parse-generation-old",
            },
            "current_parse_generation_id": "parse-generation-old",
            "current_build_generation_id": "build-generation-old",
        },
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )


def _artifact(document: DocumentRecord) -> ArtifactRecord:
    now = utc_now_iso()
    artifact_id = f"artifact-{document.id}"
    object_uri = (
        f"s3://cow-bucket/kb/{document.workspace}/{document.id}/artifacts/"
        f"{artifact_id}/sidecar.json"
    )
    return ArtifactRecord(
        id=artifact_id,
        kb_id=document.kb_id,
        workspace=document.workspace,
        document_id=document.id,
        artifact_type="sidecar",
        uri=object_uri,
        checksum="sha256:old-artifact",
        size_bytes=71,
        metadata={
            "object_uri": object_uri,
            "parse_generation_id": "parse-generation-old",
            "safe": {"purpose": "parse-sidecar"},
            "blocks_path": "/tmp/.lightrag-scratch/unsafe/blocks.jsonl",
        },
        created_at=now,
    )


def _job(
    document: DocumentRecord,
    job_id: str,
    *,
    operation: str,
    document_ids: list[str] | None = None,
) -> JobRecord:
    now = utc_now_iso()
    is_batch = document_ids is not None
    payload: dict[str, Any] = {"idempotency_fingerprint": "sha256:cow"}
    if operation == "delete" and is_batch:
        payload["document_ids"] = document_ids
    elif operation == "delete":
        payload["document_id"] = document.id
    return JobRecord(
        id=job_id,
        kb_id=document.kb_id,
        workspace=document.workspace,
        batch_id=f"batch-{job_id}" if is_batch else None,
        document_id=None if is_batch else document.id,
        job_type=operation,
        status="running",
        stage="replacing" if operation == "replace" else "deleting",
        progress=0.1,
        total_items=len(document_ids) if document_ids is not None else 1,
        completed_items=0,
        failed_items=0,
        idempotency_key=f"idem-{job_id}",
        config_version_id=None,
        config_hash=None,
        retry_count=0,
        max_retries=3,
        payload=payload,
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


def _sync_item(
    source_key: Any,
    *,
    source_hash: Any = _NEW_SOURCE_HASH,
    source_name: Any = "replacement.pdf",
    source_type: Any = "upload",
    content_type: Any = "application/pdf",
    size_bytes: Any = 202,
) -> dict[str, Any]:
    return {
        "source_key": source_key,
        "source_hash": source_hash,
        "source_name": source_name,
        "source_type": source_type,
        "content_type": content_type,
        "size_bytes": size_bytes,
    }


def _sync_job(
    document: DocumentRecord,
    job_id: str,
    *,
    items: Any,
    include_items: bool = True,
) -> JobRecord:
    job = _job(document, job_id, operation="replace")
    payload: dict[str, Any] = {"idempotency_fingerprint": "sha256:cow-sync"}
    if include_items:
        payload["items"] = items
    return replace(
        job,
        batch_id=f"batch-{job_id}",
        document_id=None,
        job_type="sync",
        stage="syncing",
        total_items=len(items) if isinstance(items, list) else 1,
        payload=payload,
    )


async def _create_artifact(store: Any, artifact: ArtifactRecord) -> None:
    if store._cow_backend == "sqlite":
        await store._write(lambda conn: store._insert_artifact(conn, artifact))
        return

    async def write(conn: Any) -> None:
        await store._insert_artifact(conn, artifact)

    await store._write(write)


async def _seed(
    store: Any,
    *,
    operation: str,
    document_id: str = "document-1",
    job_id: str = "job-1",
) -> tuple[str, str, DocumentRecord, ArtifactRecord, JobRecord]:
    kb_id = _kb_id(store)
    generation = f"generation-{uuid.uuid4().hex[:8]}"
    await store.activate_kb_generation(kb_id, generation)
    document = _document(kb_id, document_id)
    artifact = _artifact(document)
    job = _job(document, job_id, operation=operation)
    await store.create_documents_and_job([document], job)
    await _create_artifact(store, artifact)
    return kb_id, generation, document, artifact, job


async def _seed_sync(
    store: Any,
    *,
    document_id: str = "document-sync",
    job_id: str = "job-sync",
    items: Any = _DEFAULT_SYNC_ITEMS,
    include_items: bool = True,
    remove_document_source_key: bool = False,
) -> tuple[str, str, DocumentRecord, ArtifactRecord, JobRecord]:
    kb_id = _kb_id(store)
    generation = f"generation-{uuid.uuid4().hex[:8]}"
    await store.activate_kb_generation(kb_id, generation)
    document = _document(kb_id, document_id)
    durable_source_key = str(document.metadata["source_key"])
    if remove_document_source_key:
        document.metadata.pop("source_key")
    resolved_items = (
        [_sync_item(durable_source_key)] if items is _DEFAULT_SYNC_ITEMS else items
    )
    artifact = _artifact(document)
    job = _sync_job(
        document,
        job_id,
        items=resolved_items,
        include_items=include_items,
    )
    await store.create_documents_and_job([document], job)
    await _create_artifact(store, artifact)
    return kb_id, generation, document, artifact, job


async def _assert_zero_document_manifest_mutation(
    store: Any,
    document: DocumentRecord,
    artifact: ArtifactRecord,
) -> None:
    assert await store.get_document(document.kb_id, document.id) == document
    artifacts, total = await store.list_document_artifacts(
        document.kb_id,
        document.id,
        limit=20,
    )
    assert artifacts == [artifact]
    assert total == 1
    manifests, total = await store.list_artifact_cleanup_manifests(
        kb_id=document.kb_id,
        document_id=document.id,
        limit=20,
    )
    assert manifests == []
    assert total == 0


def _new_source(
    document: DocumentRecord,
    *,
    generation: str,
    job_id: str,
    attempt_token: str,
) -> dict[str, Any]:
    source_hash = _NEW_SOURCE_HASH
    source_generation = document_source_generation_id(
        kb_id=document.kb_id,
        kb_generation=generation,
        document_id=document.id,
        job_id=job_id,
        attempt_token=attempt_token,
        source_hash=source_hash,
    )
    object_uri = (
        f"s3://cow-bucket/kb/{document.workspace}/{document.id}/source/"
        f"candidates/{job_id}/{attempt_token}/source.bin"
    )
    return {
        "new_source_type": "upload",
        "new_source_name": "replacement.pdf",
        "new_source_uri": object_uri,
        "new_source_hash": source_hash,
        "new_content_type": "application/pdf",
        "new_size_bytes": 202,
        "new_source_object_uri": object_uri,
        "new_source_generation_id": source_generation,
    }


def _manifest(
    *,
    manifest_id: str,
    operation: str,
    document: DocumentRecord,
    kb_generation: str,
    job_id: str,
    attempt_token: str,
    snapshot_digest: str,
    target_uri: str,
    target_namespace: str,
    artifact_id: str | None,
    source_generation_id: str | None,
    disposition: str = "delete",
) -> ArtifactCleanupManifestRecord:
    now = datetime.now(timezone.utc)
    reason = "replace" if operation == "replace" else "document_delete"
    status = "retained" if disposition == "retain" else "pending"
    group_id = document_mutation_manifest_group_id(
        operation,  # type: ignore[arg-type]
        kb_id=document.kb_id,
        kb_generation=kb_generation,
        document_id=document.id,
        job_id=job_id,
        attempt_token=attempt_token,
        snapshot_digest=snapshot_digest,
    )
    key = artifact_cleanup_idempotency_key(
        reason=reason,  # type: ignore[arg-type]
        kb_id=document.kb_id,
        kb_generation=kb_generation,
        workspace=document.workspace,
        document_id=document.id,
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
        kb_id=document.kb_id,
        kb_generation=kb_generation,
        workspace=document.workspace,
        document_id=document.id,
        artifact_id=artifact_id,
        source_generation_id=source_generation_id,
        origin_job_id=job_id,
        origin_attempt_token=attempt_token,
        reason=reason,  # type: ignore[arg-type]
        target_kind="object",
        target_namespace=target_namespace,  # type: ignore[arg-type]
        disposition=disposition,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        target_uri=target_uri,
        expected_checksum=None,
        expected_etag=None,
        expected_version_id=None,
        expected_size_bytes=None,
        delete_after=now,
        cleanup_deadline_at=now + timedelta(days=7),
        audit_retain_until=now + timedelta(days=30),
        next_attempt_at=now,
        attempt_count=0,
        created_at=now,
        updated_at=now,
    )


def _manifests(
    *,
    operation: str,
    document: DocumentRecord,
    artifact: ArtifactRecord,
    kb_generation: str,
    job_id: str,
    claim: DocumentMutationClaimResult,
    retain_artifact: bool = False,
) -> tuple[ArtifactCleanupManifestRecord, ...]:
    return (
        _manifest(
            manifest_id=f"manifest-{operation}-source-{document.id}",
            operation=operation,
            document=document,
            kb_generation=kb_generation,
            job_id=job_id,
            attempt_token=claim.attempt_token,
            snapshot_digest=claim.snapshot_digest,
            target_uri=str(document.metadata["source_object_uri"]),
            target_namespace="source",
            artifact_id=None,
            source_generation_id=str(document.metadata["source_generation_id"]),
        ),
        _manifest(
            manifest_id=f"manifest-{operation}-artifact-{document.id}",
            operation=operation,
            document=document,
            kb_generation=kb_generation,
            job_id=job_id,
            attempt_token=claim.attempt_token,
            snapshot_digest=claim.snapshot_digest,
            target_uri=str(artifact.metadata["object_uri"]),
            target_namespace="artifact",
            artifact_id=artifact.id,
            source_generation_id=None,
            disposition="retain" if retain_artifact else "delete",
        ),
    )


def test_document_cow_public_signatures_match() -> None:
    names = (
        "claim_document_replacing_cow",
        "claim_document_deleting_cow",
        "claim_documents_deleting_cow",
        "commit_document_replace_cow",
        "commit_document_delete_cow",
        "finalize_document_replace_cow",
        "record_document_replace_engine_cleanup_failure_cow",
        "fail_document_replace_cow",
        "fail_document_delete_cow",
        "reconcile_document_replace_cow_commit",
        "reconcile_document_delete_cow_commit",
    )
    for name in names:
        assert inspect.signature(
            getattr(SQLiteMetadataStore, name)
        ) == inspect.signature(getattr(PostgresMetadataStore, name))


def test_document_mutation_result_types_are_frozen() -> None:
    document = _document("kb-frozen", "document-frozen")
    artifact = _artifact(document)
    result = DocumentMutationClaimResult(
        operation="replace",
        document=document,
        attempt_token="attempt-frozen",
        snapshot_digest=document_mutation_snapshot(
            document,
            [artifact],
            operation="replace",
        ),
        snapshot_version=1,
        artifacts=(artifact,),
        old_source_object_uri=str(document.metadata["source_object_uri"]),
        old_source_generation_id=str(document.metadata["source_generation_id"]),
        previous_lightrag_doc_id=document.lightrag_doc_id,
    )
    with pytest.raises(FrozenInstanceError):
        result.operation = "delete"  # type: ignore[misc]


def test_document_mutation_snapshot_is_deterministic_and_exact() -> None:
    document = _document("kb-snapshot", "document-snapshot")
    artifact = _artifact(document)
    second = replace(
        artifact,
        id="artifact-second",
        uri=artifact.uri.replace(artifact.id, "artifact-second"),
        metadata={
            **artifact.metadata,
            "object_uri": str(artifact.metadata["object_uri"]).replace(
                artifact.id,
                "artifact-second",
            ),
        },
    )
    document.metadata["current_artifact_ids"] = [artifact.id, second.id]
    digest = document_mutation_snapshot(
        document,
        [artifact, second],
        operation="replace",
    )
    assert digest == document_mutation_snapshot(
        replace(document, updated_at=utc_now_iso(), error_message="ignored"),
        [second, artifact],
        operation="replace",
    )
    assert digest == document_mutation_snapshot(
        replace(
            document,
            metadata={
                **document.metadata,
                "current_artifact_ids": [second.id, artifact.id],
            },
        ),
        [second, artifact],
        operation="replace",
    )
    claimed_controls = replace(
        document,
        status="replacing",
        metadata={
            **document.metadata,
            "pending_replace_job_id": "job",
            "pending_replace_claim_token": "attempt",
            "replace_attempt_token_history": ["attempt"],
            "replace_mutation_snapshot_digest": digest,
            "replace_mutation_snapshot_version": 1,
            "replace_snapshot_status": "ready",
            "replace_phase": "pre_commit",
        },
    )
    assert digest == document_mutation_snapshot(
        claimed_controls,
        [artifact, second],
        operation="replace",
    )
    mutations = (
        replace(document, source_hash="sha256:changed"),
        replace(
            document,
            metadata={**document.metadata, "source_generation_id": "changed"},
        ),
        replace(document, lightrag_doc_id="engine-changed"),
        replace(document, parser_hash="parser-changed"),
        replace(document, chunks_count=99),
        replace(
            document,
            metadata={
                **document.metadata,
                "current_sidecar_artifact_id": "different-artifact",
            },
        ),
    )
    for mutated in mutations:
        assert digest != document_mutation_snapshot(
            mutated,
            [artifact, second],
            operation="replace",
        )
    changed_artifact = replace(artifact, checksum="sha256:changed-artifact")
    assert digest != document_mutation_snapshot(
        document,
        [changed_artifact, second],
        operation="replace",
    )


@pytest.mark.asyncio
async def test_sync_replace_claim_accepts_only_exact_normalized_source_key(
    cow_store,
) -> None:
    document_id = "document-sync-exact"
    source_key = f"source-key-{document_id}"
    kb_id, generation, document, _artifact_record, job = await _seed_sync(
        cow_store,
        document_id=document_id,
        items=[
            _sync_item(
                f"  {source_key}  ",
                source_hash=f"  {_NEW_SOURCE_HASH}  ",
            )
        ],
    )
    claim = await cow_store.claim_document_replacing_cow(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        claim_token="sync-exact-attempt",
    )
    assert claim.document.status == "replacing"
    assert claim.document.metadata["pending_replace_job_id"] == job.id
    assert claim.document.metadata["source_key"] == source_key


@pytest.mark.asyncio
async def test_sync_replace_claim_rejects_unrelated_nonempty_items_atomically(
    cow_store,
) -> None:
    kb_id, generation, document, artifact, job = await _seed_sync(
        cow_store,
        document_id="document-sync-unrelated",
        items=[_sync_item("source-key-for-another-document")],
    )
    with pytest.raises(DocumentAttemptOwnershipError):
        await cow_store.claim_document_replacing_cow(
            kb_id,
            document.id,
            kb_generation=generation,
            job_id=job.id,
            claim_token="sync-unrelated-attempt",
        )
    await _assert_zero_document_manifest_mutation(cow_store, document, artifact)


@pytest.mark.parametrize(
    "authority_case",
    [
        "missing_document_source_key",
        "missing_items",
        "items_not_list",
        "empty_items",
        "item_not_object",
        "missing_item_source_key",
        "malformed_item_source_key",
        "malformed_item_checksum",
        "duplicate_checksum_conflict",
        "duplicate_other_authority_conflict",
    ],
)
@pytest.mark.asyncio
async def test_sync_replace_claim_rejects_malformed_or_conflicting_authority(
    cow_store,
    authority_case: str,
) -> None:
    document_id = f"document-sync-{authority_case}"
    source_key = f"source-key-{document_id}"
    include_items = authority_case != "missing_items"
    remove_document_source_key = authority_case == "missing_document_source_key"
    if authority_case == "items_not_list":
        items: Any = {"source_key": source_key, "source_hash": _NEW_SOURCE_HASH}
    elif authority_case == "empty_items":
        items = []
    elif authority_case == "item_not_object":
        items = [source_key]
    elif authority_case == "missing_item_source_key":
        items = [{"source_hash": _NEW_SOURCE_HASH}]
    elif authority_case == "malformed_item_source_key":
        items = [_sync_item(42)]
    elif authority_case == "malformed_item_checksum":
        items = [_sync_item(source_key, source_hash=None)]
    elif authority_case == "duplicate_checksum_conflict":
        items = [
            _sync_item(source_key, source_hash="sha256:first"),
            _sync_item(f" {source_key} ", source_hash="sha256:second"),
        ]
    elif authority_case == "duplicate_other_authority_conflict":
        items = [
            _sync_item(source_key, source_name="first.pdf"),
            _sync_item(f" {source_key} ", source_name="second.pdf"),
        ]
    else:
        items = [_sync_item(source_key)]
    kb_id, generation, document, artifact, job = await _seed_sync(
        cow_store,
        document_id=document_id,
        items=items,
        include_items=include_items,
        remove_document_source_key=remove_document_source_key,
    )
    with pytest.raises(DocumentAttemptOwnershipError):
        await cow_store.claim_document_replacing_cow(
            kb_id,
            document.id,
            kb_generation=generation,
            job_id=job.id,
            claim_token=f"sync-{authority_case}-attempt",
        )
    await _assert_zero_document_manifest_mutation(cow_store, document, artifact)


@pytest.mark.asyncio
async def test_sync_replace_commit_checksum_mismatch_is_atomic(cow_store) -> None:
    kb_id, generation, document, artifact, job = await _seed_sync(
        cow_store,
        document_id="document-sync-checksum-mismatch",
        items=[
            _sync_item(
                "source-key-document-sync-checksum-mismatch",
                source_hash="sha256:different-source",
            )
        ],
    )
    claim = await cow_store.claim_document_replacing_cow(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        claim_token="sync-checksum-mismatch-attempt",
    )
    source = _new_source(
        document,
        generation=generation,
        job_id=job.id,
        attempt_token=claim.attempt_token,
    )
    manifests = _manifests(
        operation="replace",
        document=document,
        artifact=artifact,
        kb_generation=generation,
        job_id=job.id,
        claim=claim,
    )
    with pytest.raises(DocumentAttemptOwnershipError):
        await cow_store.reconcile_document_replace_cow_commit(
            kb_id,
            document.id,
            kb_generation=generation,
            job_id=job.id,
            attempt_token=claim.attempt_token,
            expected_snapshot=claim.snapshot_digest,
            manifests=manifests,
            **source,
        )
    with pytest.raises(DocumentAttemptOwnershipError):
        await cow_store.commit_document_replace_cow(
            kb_id,
            document.id,
            kb_generation=generation,
            job_id=job.id,
            attempt_token=claim.attempt_token,
            expected_snapshot=claim.snapshot_digest,
            metadata_patch=None,
            manifests=manifests,
            **source,
        )
    await _assert_zero_document_manifest_mutation(cow_store, claim.document, artifact)


@pytest.mark.asyncio
async def test_sync_replace_exact_item_checksum_commit_and_reconciliation(
    cow_store,
) -> None:
    document_id = "document-sync-commit"
    source_key = f"source-key-{document_id}"
    kb_id, generation, document, artifact, job = await _seed_sync(
        cow_store,
        document_id=document_id,
        items=[
            _sync_item(
                f" {source_key} ",
                source_hash=f" {_NEW_SOURCE_HASH} ",
            )
        ],
    )
    claim = await cow_store.claim_document_replacing_cow(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        claim_token="sync-commit-attempt",
    )
    source = _new_source(
        document,
        generation=generation,
        job_id=job.id,
        attempt_token=claim.attempt_token,
    )
    manifests = _manifests(
        operation="replace",
        document=document,
        artifact=artifact,
        kb_generation=generation,
        job_id=job.id,
        claim=claim,
    )
    before = await cow_store.reconcile_document_replace_cow_commit(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        attempt_token=claim.attempt_token,
        expected_snapshot=claim.snapshot_digest,
        manifests=manifests,
        **source,
    )
    assert before.outcome is MetadataCommitOutcome.ROLLED_BACK
    committed = await cow_store.commit_document_replace_cow(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        attempt_token=claim.attempt_token,
        expected_snapshot=claim.snapshot_digest,
        metadata_patch=None,
        manifests=manifests,
        **source,
    )
    assert committed.document.source_hash == _NEW_SOURCE_HASH
    assert committed.document.metadata["source_key"] == source_key
    after = await cow_store.reconcile_document_replace_cow_commit(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        attempt_token=claim.attempt_token,
        expected_snapshot=claim.snapshot_digest,
        manifests=manifests,
        **source,
    )
    assert after.outcome is MetadataCommitOutcome.COMMITTED


@pytest.mark.asyncio
async def test_direct_replace_still_requires_exact_job_document_id(cow_store) -> None:
    kb_id, generation, document, artifact, job = await _seed(
        cow_store,
        operation="replace",
        document_id="document-direct-authority",
    )
    wrong_job = replace(
        job,
        id="job-direct-wrong-document",
        document_id=None,
        idempotency_key="idem-job-direct-wrong-document",
    )
    await cow_store.create_job(wrong_job)
    with pytest.raises(MetadataStoreError):
        await cow_store.claim_document_replacing_cow(
            kb_id,
            document.id,
            kb_generation=generation,
            job_id=wrong_job.id,
            claim_token="direct-wrong-document-attempt",
        )
    await _assert_zero_document_manifest_mutation(cow_store, document, artifact)


@pytest.mark.asyncio
async def test_claim_idempotency_takeover_and_conflict_fences(cow_store) -> None:
    kb_id, generation, document, _artifact_record, job = await _seed(
        cow_store,
        operation="replace",
    )
    initial = document_mutation_snapshot(
        document, [_artifact_record], operation="replace"
    )
    first = await cow_store.claim_document_replacing_cow(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        claim_token="replace-attempt-a",
        expected_snapshot=initial,
        metadata_patch={"delete_artifacts": True},
    )
    same = await cow_store.claim_document_replacing_cow(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
    )
    assert same.attempt_token == first.attempt_token
    assert same.snapshot_digest == first.snapshot_digest
    assert same.document == first.document
    takeover = await cow_store.claim_document_replacing_cow(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        claim_token="replace-attempt-b",
        expected_snapshot=first.snapshot_digest,
    )
    assert takeover.attempt_token == "replace-attempt-b"
    assert takeover.document.metadata["replace_attempt_token_history"] == [
        "replace-attempt-a",
        "replace-attempt-b",
    ]
    with pytest.raises(MetadataStoreError):
        await cow_store.claim_document_replacing_cow(
            kb_id,
            document.id,
            kb_generation=generation,
            job_id=job.id,
            claim_token="replace-attempt-a",
        )

    other_job = replace(job, id="job-other", idempotency_key="idem-job-other")
    await cow_store.create_job(other_job)
    with pytest.raises(MetadataStoreError):
        await cow_store.claim_document_replacing_cow(
            kb_id,
            document.id,
            kb_generation=generation,
            job_id=other_job.id,
        )


@pytest.mark.asyncio
async def test_replace_commit_final_failure_and_reconciliation(cow_store) -> None:
    kb_id, generation, document, artifact, job = await _seed(
        cow_store,
        operation="replace",
    )
    claim = await cow_store.claim_document_replacing_cow(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        claim_token="replace-attempt",
    )
    source = _new_source(
        document,
        generation=generation,
        job_id=job.id,
        attempt_token=claim.attempt_token,
    )
    manifests = _manifests(
        operation="replace",
        document=document,
        artifact=artifact,
        kb_generation=generation,
        job_id=job.id,
        claim=claim,
        retain_artifact=True,
    )
    before = await cow_store.reconcile_document_replace_cow_commit(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        attempt_token=claim.attempt_token,
        expected_snapshot=claim.snapshot_digest,
        manifests=manifests,
        **source,
    )
    assert before.outcome is MetadataCommitOutcome.ROLLED_BACK

    committed = await cow_store.commit_document_replace_cow(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        attempt_token=claim.attempt_token,
        expected_snapshot=claim.snapshot_digest,
        metadata_patch={"auto_parse": True},
        manifests=manifests,
        **source,
    )
    assert committed.document.status == "replacing"
    assert committed.document.metadata["replace_phase"] == "engine_cleanup_pending"
    assert (
        committed.document.metadata["source_object_uri"]
        == source["new_source_object_uri"]
    )
    assert (
        committed.document.metadata["source_generation_id"]
        == source["new_source_generation_id"]
    )
    assert "current_source_object_uri" not in committed.document.metadata
    assert committed.document.lightrag_doc_id == document.lightrag_doc_id
    assert committed.document.parser_hash == document.parser_hash
    assert committed.document.index_hash == document.index_hash
    assert committed.document.chunks_count == document.chunks_count
    assert committed.pending_cleanup_count == 1
    assert committed.retained_cleanup_count == 1
    artifacts, total = await cow_store.list_document_artifacts(
        kb_id,
        document.id,
        limit=20,
    )
    assert artifacts == []
    assert total == 0

    replay = await cow_store.commit_document_replace_cow(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        attempt_token=claim.attempt_token,
        expected_snapshot=claim.snapshot_digest,
        metadata_patch=None,
        manifests=manifests,
        **source,
    )
    assert replay.manifest_ids == committed.manifest_ids
    incomplete_readback = await cow_store.reconcile_document_replace_cow_commit(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        attempt_token=claim.attempt_token,
        expected_snapshot=claim.snapshot_digest,
        manifests=manifests[:1],
        **source,
    )
    assert incomplete_readback.outcome is MetadataCommitOutcome.UNKNOWN
    with pytest.raises(MetadataStoreError):
        await cow_store.claim_document_replacing_cow(
            kb_id,
            document.id,
            kb_generation=generation,
            job_id=job.id,
            claim_token="replace-attempt-rotated",
        )

    await cow_store.update_document(
        kb_id,
        document.id,
        metadata_patch={"current_parse_generation_id": "raced-parse-generation"},
    )
    with pytest.raises(MetadataStoreError):
        await cow_store.finalize_document_replace_cow(
            kb_id,
            document.id,
            kb_generation=generation,
            job_id=job.id,
            attempt_token=claim.attempt_token,
            source_object_uri=str(source["new_source_object_uri"]),
            source_generation_id=str(source["new_source_generation_id"]),
            manifest_group_id=committed.manifest_group_id,
        )
    await cow_store.update_document(
        kb_id,
        document.id,
        metadata_patch={"current_parse_generation_id": "parse-generation-old"},
    )

    failed_cleanup = await cow_store.record_document_replace_engine_cleanup_failure_cow(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        attempt_token=claim.attempt_token,
        source_object_uri=str(source["new_source_object_uri"]),
        source_generation_id=str(source["new_source_generation_id"]),
        manifest_group_id=committed.manifest_group_id,
        error_code="engine_cleanup_failed",
        error_message="Safe engine cleanup failure",
    )
    assert failed_cleanup.status == "replacing"
    assert failed_cleanup.metadata["replace_phase"] == "engine_cleanup_pending"
    assert failed_cleanup.error_code == "engine_cleanup_failed"

    finalized = await cow_store.finalize_document_replace_cow(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        attempt_token=claim.attempt_token,
        source_object_uri=str(source["new_source_object_uri"]),
        source_generation_id=str(source["new_source_generation_id"]),
        manifest_group_id=committed.manifest_group_id,
    )
    assert finalized.status == "uploaded"
    assert finalized.metadata["replace_phase"] == "completed"
    assert finalized.lightrag_doc_id is None
    assert finalized.parser_hash is None
    assert finalized.index_hash is None
    assert finalized.chunks_count is None
    assert finalized.metadata["source_object_uri"] == source["new_source_object_uri"]

    after = await cow_store.reconcile_document_replace_cow_commit(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        attempt_token=claim.attempt_token,
        expected_snapshot=claim.snapshot_digest,
        manifests=manifests,
        **source,
    )
    assert after.outcome is MetadataCommitOutcome.COMMITTED


@pytest.mark.asyncio
async def test_replace_stale_generation_snapshot_and_manifest_roll_back(
    cow_store,
) -> None:
    kb_id, generation, document, artifact, job = await _seed(
        cow_store,
        operation="replace",
    )
    claim = await cow_store.claim_document_replacing_cow(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        claim_token="replace-attempt",
    )
    source = _new_source(
        document,
        generation=generation,
        job_id=job.id,
        attempt_token=claim.attempt_token,
    )
    manifests = _manifests(
        operation="replace",
        document=document,
        artifact=artifact,
        kb_generation=generation,
        job_id=job.id,
        claim=claim,
    )
    wrong_workspace_document = replace(document, workspace="wrong-workspace")
    malformed = (
        _manifest(
            manifest_id=manifests[0].id,
            operation="replace",
            document=wrong_workspace_document,
            kb_generation=generation,
            job_id=job.id,
            attempt_token=claim.attempt_token,
            snapshot_digest=claim.snapshot_digest,
            target_uri=str(document.metadata["source_object_uri"]),
            target_namespace="source",
            artifact_id=None,
            source_generation_id=str(document.metadata["source_generation_id"]),
        ),
        manifests[1],
    )
    with pytest.raises((MetadataStoreError, ValueError)):
        await cow_store.commit_document_replace_cow(
            kb_id,
            document.id,
            kb_generation=generation,
            job_id=job.id,
            attempt_token=claim.attempt_token,
            expected_snapshot=claim.snapshot_digest,
            metadata_patch=None,
            manifests=malformed,
            **source,
        )
    current = await cow_store.get_document(kb_id, document.id)
    assert current.source_hash == document.source_hash
    rows, total = await cow_store.list_artifact_cleanup_manifests(
        manifest_group_id=manifests[0].manifest_group_id,
        limit=20,
    )
    assert rows == []
    assert total == 0

    changed_artifact = replace(
        artifact,
        id="artifact-raced",
        uri=artifact.uri.replace(artifact.id, "artifact-raced"),
        checksum="sha256:raced",
        metadata={
            **artifact.metadata,
            "object_uri": str(artifact.metadata["object_uri"]).replace(
                artifact.id,
                "artifact-raced",
            ),
        },
    )
    await _create_artifact(cow_store, changed_artifact)
    with pytest.raises(MetadataConflictError):
        await cow_store.commit_document_replace_cow(
            kb_id,
            document.id,
            kb_generation=generation,
            job_id=job.id,
            attempt_token=claim.attempt_token,
            expected_snapshot=claim.snapshot_digest,
            metadata_patch=None,
            manifests=manifests,
            **source,
        )

    with pytest.raises(KBLifecycleConflictError):
        await cow_store.commit_document_replace_cow(
            kb_id,
            document.id,
            kb_generation="stale-generation",
            job_id=job.id,
            attempt_token=claim.attempt_token,
            expected_snapshot=claim.snapshot_digest,
            metadata_patch=None,
            manifests=manifests,
            **source,
        )


@pytest.mark.asyncio
async def test_replace_source_pointer_race_has_zero_partial_writes(cow_store) -> None:
    kb_id, generation, document, artifact, job = await _seed(
        cow_store,
        operation="replace",
    )
    claim = await cow_store.claim_document_replacing_cow(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        claim_token="replace-attempt",
    )
    source = _new_source(
        document,
        generation=generation,
        job_id=job.id,
        attempt_token=claim.attempt_token,
    )
    manifests = _manifests(
        operation="replace",
        document=document,
        artifact=artifact,
        kb_generation=generation,
        job_id=job.id,
        claim=claim,
    )
    raced_uri = (
        f"s3://cow-bucket/kb/{document.workspace}/{document.id}/source/"
        "generations/raced-generation/source.pdf"
    )
    await cow_store.update_document(
        kb_id,
        document.id,
        metadata_patch={
            "source_object_uri": raced_uri,
            "source_generation_id": "raced-generation",
        },
    )
    with pytest.raises(MetadataConflictError):
        await cow_store.commit_document_replace_cow(
            kb_id,
            document.id,
            kb_generation=generation,
            job_id=job.id,
            attempt_token=claim.attempt_token,
            expected_snapshot=claim.snapshot_digest,
            metadata_patch=None,
            manifests=manifests,
            **source,
        )
    current = await cow_store.get_document(kb_id, document.id)
    assert current.source_hash == document.source_hash
    assert current.metadata["source_object_uri"] == raced_uri
    assert current.metadata["replace_phase"] == "pre_commit"
    rows, total = await cow_store.list_artifact_cleanup_manifests(
        manifest_group_id=manifests[0].manifest_group_id,
        limit=20,
    )
    assert rows == []
    assert total == 0
    artifacts, total = await cow_store.list_document_artifacts(
        kb_id,
        document.id,
        limit=20,
    )
    assert [item.id for item in artifacts] == [artifact.id]
    assert total == 1


@pytest.mark.asyncio
async def test_delete_commit_tombstone_and_reconciliation(cow_store) -> None:
    kb_id, generation, document, artifact, job = await _seed(
        cow_store,
        operation="delete",
    )
    claim = await cow_store.claim_document_deleting_cow(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        claim_token="delete-attempt",
    )
    manifests = _manifests(
        operation="delete",
        document=document,
        artifact=artifact,
        kb_generation=generation,
        job_id=job.id,
        claim=claim,
        retain_artifact=True,
    )
    before = await cow_store.reconcile_document_delete_cow_commit(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        attempt_token=claim.attempt_token,
        expected_snapshot=claim.snapshot_digest,
        manifests=manifests,
    )
    assert before.outcome is MetadataCommitOutcome.ROLLED_BACK
    committed = await cow_store.commit_document_delete_cow(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        attempt_token=claim.attempt_token,
        expected_snapshot=claim.snapshot_digest,
        metadata_patch={"delete_audit": "preserved"},
        manifests=manifests,
    )
    assert committed.document.status == "deleted"
    assert committed.document.deleted_at is not None
    assert committed.document.enabled is False
    assert committed.document.archived is True
    assert committed.document.lightrag_doc_id == document.lightrag_doc_id
    assert committed.document.parser_hash == document.parser_hash
    assert committed.document.metadata["last_delete_manifest_ids"] == list(
        committed.manifest_ids
    )
    assert committed.pending_cleanup_count == 1
    assert committed.retained_cleanup_count == 1
    assert (
        await cow_store.get_document_lifecycle(kb_id, document.id) == committed.document
    )
    with pytest.raises(Exception):
        await cow_store.get_document(kb_id, document.id)
    artifacts, total = await cow_store.list_document_artifacts(
        kb_id,
        document.id,
        limit=20,
    )
    assert artifacts == []
    assert total == 0
    by_source = await cow_store.get_documents_by_source_keys(
        kb_id,
        [str(document.metadata["source_key"])],
    )
    assert by_source == {}

    replay = await cow_store.commit_document_delete_cow(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        attempt_token=claim.attempt_token,
        expected_snapshot=claim.snapshot_digest,
        metadata_patch=None,
        manifests=manifests,
    )
    assert replay.manifest_ids == committed.manifest_ids
    after = await cow_store.reconcile_document_delete_cow_commit(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        attempt_token=claim.attempt_token,
        expected_snapshot=claim.snapshot_digest,
        manifests=manifests,
    )
    assert after.outcome is MetadataCommitOutcome.COMMITTED


@pytest.mark.asyncio
async def test_empty_replace_group_is_durable_and_classifiable(cow_store) -> None:
    kb_id = _kb_id(cow_store)
    generation = f"generation-{uuid.uuid4().hex[:8]}"
    await cow_store.activate_kb_generation(kb_id, generation)
    document = _document(kb_id, "document-empty-group")
    document.metadata.pop("source_object_uri")
    document.metadata.pop("source_generation_id")
    document.metadata.pop("current_sidecar_artifact_id")
    document.metadata.pop("current_artifact_ids")
    document.metadata.pop("artifact_binding")
    job = _job(document, "job-empty-group", operation="replace")
    await cow_store.create_documents_and_job([document], job)
    claim = await cow_store.claim_document_replacing_cow(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        claim_token="replace-empty-attempt",
    )
    source = _new_source(
        document,
        generation=generation,
        job_id=job.id,
        attempt_token=claim.attempt_token,
    )
    committed = await cow_store.commit_document_replace_cow(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        attempt_token=claim.attempt_token,
        expected_snapshot=claim.snapshot_digest,
        metadata_patch=None,
        manifests=(),
        **source,
    )
    assert committed.manifest_ids == ()
    assert committed.manifest_records == ()
    assert committed.document.metadata["last_replace_manifest_group_id"]
    assert committed.document.metadata["last_replace_manifest_ids"] == []
    reconciliation = await cow_store.reconcile_document_replace_cow_commit(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        attempt_token=claim.attempt_token,
        expected_snapshot=claim.snapshot_digest,
        manifests=(),
        **source,
    )
    assert reconciliation.outcome is MetadataCommitOutcome.COMMITTED


@pytest.mark.asyncio
async def test_precommit_delete_failure_is_fenced_and_preserves_authority(
    cow_store,
) -> None:
    kb_id, generation, document, artifact, job = await _seed(
        cow_store,
        operation="delete",
    )
    claim = await cow_store.claim_document_deleting_cow(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        claim_token="delete-failure-attempt",
    )
    with pytest.raises(MetadataStoreError):
        await cow_store.fail_document_delete_cow(
            kb_id,
            document.id,
            kb_generation=generation,
            job_id=job.id,
            attempt_token="stale-delete-attempt",
            error_code="delete_failed",
            error_message="Safe failure",
        )
    failed = await cow_store.fail_document_delete_cow(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        attempt_token=claim.attempt_token,
        error_code="delete_failed",
        error_message="postgresql://user:secret@db.example.invalid/database",
    )
    assert failed.status == "delete_failed"
    assert failed.deleted_at is None
    assert failed.source_hash == document.source_hash
    assert failed.lightrag_doc_id == document.lightrag_doc_id
    assert failed.error_message == "Document mutation failed"
    artifacts, total = await cow_store.list_document_artifacts(
        kb_id,
        document.id,
        limit=20,
    )
    assert [item.id for item in artifacts] == [artifact.id]
    assert total == 1


@pytest.mark.asyncio
async def test_reconciliation_mixed_state_is_unknown(cow_store) -> None:
    kb_id, generation, document, artifact, job = await _seed(
        cow_store,
        operation="replace",
    )
    claim = await cow_store.claim_document_replacing_cow(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        claim_token="replace-attempt",
    )
    source = _new_source(
        document,
        generation=generation,
        job_id=job.id,
        attempt_token=claim.attempt_token,
    )
    manifests = _manifests(
        operation="replace",
        document=document,
        artifact=artifact,
        kb_generation=generation,
        job_id=job.id,
        claim=claim,
    )
    await cow_store.enqueue_artifact_cleanup_manifests(manifests)
    outcome = await cow_store.reconcile_document_replace_cow_commit(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        attempt_token=claim.attempt_token,
        expected_snapshot=claim.snapshot_digest,
        manifests=manifests,
        **source,
    )
    assert outcome.outcome is MetadataCommitOutcome.UNKNOWN


@pytest.mark.asyncio
async def test_batch_delete_claims_preserve_per_item_progress(cow_store) -> None:
    kb_id = _kb_id(cow_store)
    generation = f"generation-{uuid.uuid4().hex[:8]}"
    await cow_store.activate_kb_generation(kb_id, generation)
    first = _document(kb_id, "document-a")
    second = _document(kb_id, "document-b")
    job = _job(
        first,
        "job-batch-delete",
        operation="delete",
        document_ids=[first.id, second.id, "missing-document"],
    )
    await cow_store.create_documents_and_job([first, second], job)
    await _create_artifact(cow_store, _artifact(first))
    await _create_artifact(cow_store, _artifact(second))
    results, failures = await cow_store.claim_documents_deleting_cow(
        kb_id,
        [first.id, "missing-document", second.id],
        kb_generation=generation,
        job_id=job.id,
        claim_tokens={first.id: "delete-attempt-a", second.id: "delete-attempt-b"},
    )
    assert [result.document.id for result in results] == [first.id, second.id]
    assert [failure["document_id"] for failure in failures] == ["missing-document"]
    replay, replay_failures = await cow_store.claim_documents_deleting_cow(
        kb_id,
        [first.id, "missing-document", second.id],
        kb_generation=generation,
        job_id=job.id,
    )
    assert [result.attempt_token for result in replay] == [
        "delete-attempt-a",
        "delete-attempt-b",
    ]
    assert [failure["document_id"] for failure in replay_failures] == [
        "missing-document"
    ]


@pytest.mark.asyncio
async def test_injected_manifest_insert_failure_rolls_back_whole_commit(
    cow_store,
    monkeypatch,
) -> None:
    kb_id, generation, document, artifact, job = await _seed(
        cow_store,
        operation="replace",
    )
    claim = await cow_store.claim_document_replacing_cow(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        claim_token="replace-attempt",
    )
    source = _new_source(
        document,
        generation=generation,
        job_id=job.id,
        attempt_token=claim.attempt_token,
    )
    manifests = _manifests(
        operation="replace",
        document=document,
        artifact=artifact,
        kb_generation=generation,
        job_id=job.id,
        claim=claim,
    )
    if cow_store._cow_backend == "sqlite":
        original = cow_store._insert_artifact_cleanup_manifest
        calls = 0

        def fail_sqlite_after_insert(conn, manifest):
            nonlocal calls
            original(conn, manifest)
            calls += 1
            if calls == 1:
                raise RuntimeError("injected transaction failure")

        monkeypatch.setattr(
            cow_store,
            "_insert_artifact_cleanup_manifest",
            fail_sqlite_after_insert,
        )
    else:
        original = cow_store._insert_postgres_artifact_manifest
        calls = 0

        async def fail_postgres_after_insert(conn, manifest, *, ignore_conflict):
            nonlocal calls
            inserted = await original(
                conn,
                manifest,
                ignore_conflict=ignore_conflict,
            )
            calls += 1
            if calls == 1:
                raise RuntimeError("injected transaction failure")
            return inserted

        monkeypatch.setattr(
            cow_store,
            "_insert_postgres_artifact_manifest",
            fail_postgres_after_insert,
        )
    with pytest.raises(RuntimeError, match="injected transaction failure"):
        await cow_store.commit_document_replace_cow(
            kb_id,
            document.id,
            kb_generation=generation,
            job_id=job.id,
            attempt_token=claim.attempt_token,
            expected_snapshot=claim.snapshot_digest,
            metadata_patch=None,
            manifests=manifests,
            **source,
        )
    current = await cow_store.get_document(kb_id, document.id)
    assert current.source_hash == document.source_hash
    assert current.status == "replacing"
    rows, total = await cow_store.list_artifact_cleanup_manifests(
        manifest_group_id=manifests[0].manifest_group_id,
        limit=20,
    )
    assert rows == []
    assert total == 0
    artifacts, total = await cow_store.list_document_artifacts(
        kb_id,
        document.id,
        limit=20,
    )
    assert [item.id for item in artifacts] == [artifact.id]
    assert total == 1


@pytest.mark.asyncio
async def test_manifest_idempotency_conflict_rolls_back_document(cow_store) -> None:
    kb_id, generation, document, artifact, job = await _seed(
        cow_store,
        operation="replace",
    )
    claim = await cow_store.claim_document_replacing_cow(
        kb_id,
        document.id,
        kb_generation=generation,
        job_id=job.id,
        claim_token="replace-attempt",
    )
    source = _new_source(
        document,
        generation=generation,
        job_id=job.id,
        attempt_token=claim.attempt_token,
    )
    manifests = _manifests(
        operation="replace",
        document=document,
        artifact=artifact,
        kb_generation=generation,
        job_id=job.id,
        claim=claim,
    )
    conflict = _manifest(
        manifest_id=manifests[0].id,
        operation="replace",
        document=document,
        kb_generation=generation,
        job_id=job.id,
        attempt_token=claim.attempt_token,
        snapshot_digest=claim.snapshot_digest,
        target_uri=str(artifact.metadata["object_uri"]),
        target_namespace="source",
        artifact_id=None,
        source_generation_id=str(document.metadata["source_generation_id"]),
    )
    await cow_store.enqueue_artifact_cleanup_manifest(conflict)
    with pytest.raises(ArtifactLifecycleConflictError):
        await cow_store.commit_document_replace_cow(
            kb_id,
            document.id,
            kb_generation=generation,
            job_id=job.id,
            attempt_token=claim.attempt_token,
            expected_snapshot=claim.snapshot_digest,
            metadata_patch=None,
            manifests=manifests,
            **source,
        )
    current = await cow_store.get_document(kb_id, document.id)
    assert current.source_hash == document.source_hash
    assert current.metadata["replace_phase"] == "pre_commit"
