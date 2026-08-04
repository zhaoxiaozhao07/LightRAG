from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import inspect
from pathlib import Path
from typing import Any

import pytest

from lightrag.api.kb_service import utc_now_iso
from lightrag.api.metadata_store import (
    DocumentAttemptClaim,
    DocumentAttemptOwnershipError,
    DocumentRecord,
    DocumentSnapshotConflictError,
    JobRecord,
    SQLiteMetadataStore,
    document_state_snapshot,
)
from lightrag.api.postgres_metadata_store import PostgresMetadataStore

pytestmark = pytest.mark.offline


def test_backend_attempt_method_signatures_match():
    method_names = (
        "mark_document_parse_queued",
        "claim_documents_parse_queued",
        "mark_document_parsing",
        "complete_document_parse",
        "fail_document_parse",
        "release_document_parse_if_owned",
        "claim_document_build_queued",
        "claim_documents_build_queued",
        "mark_document_building",
        "complete_document_build",
        "complete_document_build_with_artifact_promotion",
        "fail_document_build",
        "release_document_build_if_owned",
    )
    for method_name in method_names:
        assert inspect.signature(getattr(SQLiteMetadataStore, method_name)) == (
            inspect.signature(getattr(PostgresMetadataStore, method_name))
        )


def _document(
    kb_id: str,
    document_id: str,
    *,
    status: str = "uploaded",
    parser_hash: str | None = None,
    index_hash: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> DocumentRecord:
    now = utc_now_iso()
    return DocumentRecord(
        id=document_id,
        kb_id=kb_id,
        workspace=f"ws_{kb_id}",
        lightrag_doc_id=None,
        source_type="upload",
        source_name=f"{document_id}.pdf",
        source_uri=f"/inputs/{document_id}.pdf",
        source_hash="sha256:source",
        content_type="application/pdf",
        size_bytes=10,
        parser_hash=parser_hash,
        index_hash=index_hash,
        status=status,
        enabled=True,
        archived=False,
        chunks_count=3,
        entity_count=2,
        relation_count=1,
        error_code=None,
        error_message=None,
        metadata=dict(metadata or {}),
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )


def _seed_job(kb_id: str, document_id: str) -> JobRecord:
    now = utc_now_iso()
    return JobRecord(
        id=f"job_seed_{document_id}",
        kb_id=kb_id,
        workspace=f"ws_{kb_id}",
        batch_id=None,
        document_id=document_id,
        job_type="upload",
        status="succeeded",
        stage="uploading",
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
        result=None,
        error_code=None,
        error_message=None,
        created_at=now,
        updated_at=now,
        queued_at=None,
        started_at=now,
        finished_at=now,
        cancelled_at=None,
    )


async def _sqlite_store(
    tmp_path: Path, kb_id: str, documents: list[DocumentRecord]
) -> SQLiteMetadataStore:
    store = SQLiteMetadataStore(tmp_path / f"{kb_id}.sqlite3")
    await store.create_documents_and_job(
        documents, _seed_job(kb_id, documents[0].id)
    )
    return store


async def test_sqlite_snapshot_conflict_has_zero_state_change_in_real_transaction(
    tmp_path: Path,
):
    kb_id = "kb_snapshot_tx"
    document = _document(kb_id, "doc_parse")
    store = await _sqlite_store(tmp_path, kb_id, [document])
    try:
        before = await store.get_document(kb_id, document.id)
        stale_snapshot = document_state_snapshot(before)
        stale_snapshot["source_hash"] = "sha256:stale"

        with pytest.raises(DocumentSnapshotConflictError):
            await store.mark_document_parse_queued(
                kb_id,
                document.id,
                metadata_patch={
                    "pending_parse_job_id": "job-parse",
                    "pending_parser_hash": "sha256:parser",
                },
                expected_snapshot=stale_snapshot,
                claim_token="parse-pre-generated",
            )

        after = await store.get_document(kb_id, document.id)
        assert asdict(after) == asdict(before)
        assert "pending_parse_claim_token" not in after.metadata
        assert "parse_attempt_token_history" not in after.metadata
    finally:
        await store.close()


async def test_batch_claim_contract_accepts_dataclass_and_four_tuple_tokens(
    tmp_path: Path,
):
    kb_id = "kb_batch_claim_contract"
    parse_document = _document(kb_id, "doc_parse")
    build_document = _document(
        kb_id,
        "doc_build",
        status="ready",
        parser_hash="sha256:parser",
        index_hash="sha256:index",
        metadata={
            "current_parse_generation_id": "parse-generation-0",
            "current_sidecar_artifact_id": "sidecar-0",
            "current_blocks_artifact_id": "blocks-0",
        },
    )
    store = await _sqlite_store(
        tmp_path, kb_id, [parse_document, build_document]
    )
    try:
        parse_patch = {
            "pending_parse_job_id": "job-batch-parse",
            "pending_parser_hash": "sha256:parser",
        }
        parse_claims, parse_failures = await store.claim_documents_parse_queued(
            kb_id,
            [
                DocumentAttemptClaim(
                    document_id=parse_document.id,
                    metadata_patch=parse_patch,
                    expected_snapshot=document_state_snapshot(parse_document),
                    claim_token="parse-batch-token",
                )
            ],
        )
        assert parse_failures == []
        assert parse_claims[0].metadata["pending_parse_claim_token"] == (
            "parse-batch-token"
        )

        duplicate = await store.mark_document_parse_queued(
            kb_id,
            parse_document.id,
            metadata_patch=parse_patch,
            expected_snapshot=document_state_snapshot(parse_document),
            claim_token="parse-batch-token",
        )
        assert asdict(duplicate) == asdict(parse_claims[0])

        build_patch = {
            "pending_build_job_id": "job-batch-build",
            "pending_index_hash": "sha256:index-next",
        }
        build_claims, build_failures = await store.claim_documents_build_queued(
            kb_id,
            [
                (
                    build_document.id,
                    build_patch,
                    document_state_snapshot(build_document),
                    "build-batch-token",
                )
            ],
        )
        assert build_failures == []
        assert build_claims[0].metadata["pending_build_claim_token"] == (
            "build-batch-token"
        )
    finally:
        await store.close()


async def test_parse_retry_rotates_token_and_fences_late_terminal_writes(
    tmp_path: Path,
):
    kb_id = "kb_parse_retry_store"
    document = _document(kb_id, "doc_parse")
    store = await _sqlite_store(tmp_path, kb_id, [document])
    job_id = "job-retry"
    try:
        first = await store.mark_document_parse_queued(
            kb_id,
            document.id,
            metadata_patch={"pending_parse_job_id": job_id},
            expected_snapshot=document_state_snapshot(document),
            claim_token="parse-old-token",
        )
        old_token = first.metadata["pending_parse_claim_token"]
        await store.mark_document_parsing(
            kb_id,
            document.id,
            metadata_patch={"current_parse_job_id": job_id},
            job_id=job_id,
            claim_token=old_token,
        )
        await store.release_document_parse_if_owned(
            kb_id,
            document.id,
            job_id=job_id,
            claim_token=old_token,
            error_code="retryable",
            error_message="retry",
        )

        released = await store.get_document(kb_id, document.id)
        retry_snapshot = document_state_snapshot(released)
        with pytest.raises(DocumentAttemptOwnershipError):
            await store.mark_document_parse_queued(
                kb_id,
                document.id,
                metadata_patch={"pending_parse_job_id": job_id},
                expected_snapshot=retry_snapshot,
                claim_token=old_token,
            )

        second = await store.mark_document_parse_queued(
            kb_id,
            document.id,
            metadata_patch={"pending_parse_job_id": job_id},
            expected_snapshot=retry_snapshot,
        )
        new_token = second.metadata["pending_parse_claim_token"]
        assert new_token != old_token
        await store.mark_document_parsing(
            kb_id,
            document.id,
            metadata_patch={"current_parse_job_id": job_id},
            job_id=job_id,
            claim_token=new_token,
        )

        with pytest.raises(DocumentAttemptOwnershipError):
            await store.complete_document_parse(
                kb_id,
                document.id,
                parser_hash="sha256:late",
                lightrag_doc_id="doc-late",
                metadata_patch={"current_parse_generation_id": job_id},
                artifacts=[],
                job_id=job_id,
                claim_token=old_token,
            )
        with pytest.raises(DocumentAttemptOwnershipError):
            await store.fail_document_parse(
                kb_id,
                document.id,
                error_code="late_failure",
                error_message="late",
                metadata_patch={},
                job_id=job_id,
                claim_token=old_token,
            )
        before_release = await store.get_document(kb_id, document.id)
        after_release = await store.release_document_parse_if_owned(
            kb_id,
            document.id,
            job_id=job_id,
            claim_token=old_token,
            error_code="late_release",
            error_message="late",
        )
        assert asdict(after_release) == asdict(before_release)

        completed, _artifacts = await store.complete_document_parse(
            kb_id,
            document.id,
            parser_hash="sha256:parser-new",
            lightrag_doc_id="doc-current",
            metadata_patch={"current_parse_generation_id": job_id},
            artifacts=[],
            job_id=job_id,
            claim_token=new_token,
        )
        assert completed.metadata["current_parse_generation_id"] == new_token
        assert completed.metadata["parse_attempt_token_history"] == [
            old_token,
            new_token,
        ]
    finally:
        await store.close()


async def test_legacy_no_token_transition_does_not_weaken_tokenized_owner(
    tmp_path: Path,
):
    kb_id = "kb_legacy_attempt_store"
    legacy = _document(
        kb_id,
        "doc_legacy",
        status="parsing",
        metadata={"current_parse_job_id": "job-legacy"},
    )
    fenced = _document(kb_id, "doc_fenced")
    store = await _sqlite_store(tmp_path, kb_id, [legacy, fenced])
    try:
        completed_legacy, _artifacts = await store.complete_document_parse(
            kb_id,
            legacy.id,
            parser_hash="sha256:legacy-parser",
            lightrag_doc_id="doc-legacy-lr",
            metadata_patch={},
            artifacts=[],
        )
        legacy_token = completed_legacy.metadata["current_parse_generation_id"]
        assert legacy_token.startswith("parse_attempt_")
        assert completed_legacy.metadata["parse_attempt_token_history"] == [
            legacy_token
        ]

        claimed = await store.mark_document_parse_queued(
            kb_id,
            fenced.id,
            metadata_patch={"pending_parse_job_id": "job-fenced"},
            expected_snapshot=document_state_snapshot(fenced),
            claim_token="parse-fenced-token",
        )
        await store.mark_document_parsing(
            kb_id,
            fenced.id,
            metadata_patch={"current_parse_job_id": "job-fenced"},
        )
        before = await store.get_document(kb_id, fenced.id)
        assert before.metadata["current_parse_claim_token"] == (
            claimed.metadata["pending_parse_claim_token"]
        )
        with pytest.raises(DocumentAttemptOwnershipError):
            await store.complete_document_parse(
                kb_id,
                fenced.id,
                parser_hash="sha256:unfenced",
                lightrag_doc_id="doc-unfenced",
                metadata_patch={},
                artifacts=[],
            )
        after = await store.get_document(kb_id, fenced.id)
        assert asdict(after) == asdict(before)
    finally:
        await store.close()


async def test_build_retry_rotates_token_and_fences_late_terminal_writes(
    tmp_path: Path,
):
    kb_id = "kb_build_retry_store"
    document = _document(
        kb_id,
        "doc_build",
        status="ready",
        parser_hash="sha256:parser",
        index_hash="sha256:index-old",
        metadata={
            "current_parse_generation_id": "parse-generation-0",
            "current_build_generation_id": "build-generation-0",
            "current_sidecar_artifact_id": "sidecar-0",
            "current_blocks_artifact_id": "blocks-0",
        },
    )
    store = await _sqlite_store(tmp_path, kb_id, [document])
    job_id = "job-build-retry"
    try:
        first = await store.claim_document_build_queued(
            kb_id,
            document.id,
            metadata_patch={"pending_build_job_id": job_id},
            expected_snapshot=document_state_snapshot(document),
            claim_token="build-old-token",
        )
        old_token = first.metadata["pending_build_claim_token"]
        await store.mark_document_building(
            kb_id,
            document.id,
            metadata_patch={"current_build_job_id": job_id},
            job_id=job_id,
            claim_token=old_token,
        )
        await store.release_document_build_if_owned(
            kb_id,
            document.id,
            job_id=job_id,
            claim_token=old_token,
            error_code="retryable",
            error_message="retry",
        )

        released = await store.get_document(kb_id, document.id)
        retry_snapshot = document_state_snapshot(released)
        with pytest.raises(DocumentAttemptOwnershipError):
            await store.claim_document_build_queued(
                kb_id,
                document.id,
                metadata_patch={"pending_build_job_id": job_id},
                expected_snapshot=retry_snapshot,
                claim_token=old_token,
            )

        second = await store.claim_document_build_queued(
            kb_id,
            document.id,
            metadata_patch={"pending_build_job_id": job_id},
            expected_snapshot=retry_snapshot,
        )
        new_token = second.metadata["pending_build_claim_token"]
        assert new_token != old_token
        await store.mark_document_building(
            kb_id,
            document.id,
            metadata_patch={"current_build_job_id": job_id},
            job_id=job_id,
            claim_token=new_token,
        )

        with pytest.raises(DocumentAttemptOwnershipError):
            await store.complete_document_build(
                kb_id,
                document.id,
                index_hash="sha256:late",
                metadata_patch={"current_build_generation_id": job_id},
                job_id=job_id,
                claim_token=old_token,
            )
        with pytest.raises(DocumentAttemptOwnershipError):
            await store.fail_document_build(
                kb_id,
                document.id,
                error_code="late_failure",
                error_message="late",
                metadata_patch={},
                job_id=job_id,
                claim_token=old_token,
            )
        before_release = await store.get_document(kb_id, document.id)
        after_release = await store.release_document_build_if_owned(
            kb_id,
            document.id,
            job_id=job_id,
            claim_token=old_token,
            error_code="late_release",
            error_message="late",
        )
        assert asdict(after_release) == asdict(before_release)

        completed = await store.complete_document_build(
            kb_id,
            document.id,
            index_hash="sha256:index-new",
            metadata_patch={"current_build_generation_id": job_id},
            job_id=job_id,
            claim_token=new_token,
        )
        assert completed.metadata["current_build_generation_id"] == new_token
        assert completed.metadata["build_attempt_token_history"] == [
            "build-generation-0",
            old_token,
            new_token,
        ]
        with pytest.raises(DocumentAttemptOwnershipError):
            await store.claim_document_build_queued(
                kb_id,
                document.id,
                metadata_patch={"pending_build_job_id": "job-after-success"},
                expected_snapshot=document_state_snapshot(completed),
                claim_token="build-generation-0",
            )
    finally:
        await store.close()


@pytest.mark.parametrize(
    "field",
    [
        "status",
        "source_hash",
        "parser_hash",
        "current_parse_generation_id",
        "current_sidecar_artifact_id",
        "current_blocks_artifact_id",
        "index_hash",
    ],
)
async def test_skipped_build_snapshot_compares_every_planning_field(
    tmp_path: Path, field: str
):
    kb_id = f"kb_skipped_snapshot_{field}"
    document = _document(
        kb_id,
        "doc_build",
        status="ready",
        parser_hash="sha256:parser",
        index_hash="sha256:index",
        metadata={
            "current_parse_generation_id": "parse-generation-0",
            "current_sidecar_artifact_id": "sidecar-0",
            "current_blocks_artifact_id": "blocks-0",
        },
    )
    store = await _sqlite_store(tmp_path, kb_id, [document])
    try:
        before = await store.get_document(kb_id, document.id)
        expected = document_state_snapshot(before)
        expected[field] = f"stale:{expected[field]}"
        with pytest.raises(DocumentSnapshotConflictError):
            await store.claim_document_build_queued(
                kb_id,
                document.id,
                metadata_patch={"pending_build_job_id": "job-skipped"},
                expected_snapshot=expected,
                claim_token=f"build-stale-{field}",
            )
        after = await store.get_document(kb_id, document.id)
        assert asdict(after) == asdict(before)
    finally:
        await store.close()


class _FakePostgresConnection:
    def __init__(self, document: DocumentRecord):
        self.document = document
        self.events: list[tuple[Any, ...]] = []

    async def fetchrow(self, query: str, *_args: Any) -> dict[str, Any]:
        self.events.append(("select", query))
        return {"data_json": asdict(self.document)}


async def test_postgres_claim_and_owner_cas_lock_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
):
    document = _document(
        "kb-pg",
        "doc-pg",
        status="ready",
        parser_hash="sha256:parser",
        index_hash="sha256:index",
        metadata={
            "current_parse_generation_id": "parse-generation-0",
            "current_sidecar_artifact_id": "sidecar-0",
            "current_blocks_artifact_id": "blocks-0",
        },
    )
    conn = _FakePostgresConnection(document)
    store = PostgresMetadataStore(dsn="postgresql://unused")

    async def ensure() -> None:
        return None

    async def write(callback):
        return await callback(conn)

    async def update_state(
        _conn,
        _kb_id,
        _document_id,
        *,
        status,
        metadata_patch,
        **_kwargs,
    ):
        conn.events.append(("update", status, dict(metadata_patch)))
        result = deepcopy(conn.document)
        result.status = status
        result.metadata.update(metadata_patch)
        return result

    monkeypatch.setattr(store, "_ensure_initialized", ensure)
    monkeypatch.setattr(store, "_write", write)
    monkeypatch.setattr(store, "_update_document_parse_state", update_state)

    stale_snapshot = document_state_snapshot(document)
    stale_snapshot["parser_hash"] = "sha256:stale"
    with pytest.raises(DocumentSnapshotConflictError):
        await store.claim_document_build_queued(
            "kb-pg",
            document.id,
            metadata_patch={"pending_build_job_id": "job-pg"},
            expected_snapshot=stale_snapshot,
            claim_token="build-pg-token",
        )
    assert [event[0] for event in conn.events] == ["select"]
    assert "FOR UPDATE" in conn.events[0][1]

    conn.events.clear()
    claimed = await store.claim_document_build_queued(
        "kb-pg",
        document.id,
        metadata_patch={"pending_build_job_id": "job-pg"},
        expected_snapshot=document_state_snapshot(document),
        claim_token="build-pg-token",
    )
    assert [event[0] for event in conn.events] == ["select", "update"]
    assert "FOR UPDATE" in conn.events[0][1]
    assert conn.events[1][2]["pending_build_claim_token"] == "build-pg-token"

    conn.document = claimed
    conn.events.clear()
    with pytest.raises(DocumentAttemptOwnershipError):
        await store.mark_document_building(
            "kb-pg",
            document.id,
            metadata_patch={"current_build_job_id": "job-pg"},
            job_id="job-pg",
            claim_token="build-stale-token",
        )
    assert [event[0] for event in conn.events] == ["select"]
    assert "FOR UPDATE" in conn.events[0][1]

    conn.events.clear()
    building = await store.mark_document_building(
        "kb-pg",
        document.id,
        metadata_patch={"current_build_job_id": "job-pg"},
        job_id="job-pg",
        claim_token="build-pg-token",
    )
    assert [event[0] for event in conn.events] == ["select", "update"]
    owner_patch = conn.events[1][2]
    assert owner_patch["pending_build_job_id"] is None
    assert owner_patch["pending_build_claim_token"] is None
    assert owner_patch["current_build_job_id"] == "job-pg"
    assert owner_patch["current_build_claim_token"] == "build-pg-token"

    conn.document = building
    conn.events.clear()
    released = await store.release_document_build_if_owned(
        "kb-pg",
        document.id,
        job_id="job-pg",
        claim_token="build-stale-token",
        error_code="late",
        error_message="late",
    )
    assert asdict(released) == asdict(building)
    assert [event[0] for event in conn.events] == ["select"]
    assert "FOR UPDATE" in conn.events[0][1]
