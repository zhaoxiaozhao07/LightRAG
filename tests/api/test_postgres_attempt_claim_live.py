from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from typing import Any

import pytest

from lightrag.api.kb_service import utc_now_iso
from lightrag.api.metadata_store import (
    DocumentAttemptOwnershipError,
    DocumentRecord,
    DocumentSnapshotConflictError,
    JobRecord,
    document_state_snapshot,
)
from lightrag.api.postgres_metadata_store import PostgresMetadataStore

pytestmark = pytest.mark.offline

_POSTGRES_DSN = os.getenv("LIGHTRAG_KB_POSTGRES_TEST_DSN") or os.getenv(
    "POSTGRES_TEST_DSN"
)
_WriteCallback = Callable[[Any], Awaitable[Any]]
_WriteMethod = Callable[[_WriteCallback], Awaitable[Any]]


def _ready_document(kb_id: str, document_id: str, token: str) -> DocumentRecord:
    now = utc_now_iso()
    return DocumentRecord(
        id=document_id,
        kb_id=kb_id,
        workspace=f"ws_{token}",
        lightrag_doc_id=f"lightrag_{document_id}",
        source_type="upload",
        source_name=f"{document_id}.pdf",
        source_uri=f"/inputs/{document_id}.pdf",
        source_hash=f"sha256:source-{token}",
        content_type="application/pdf",
        size_bytes=10,
        parser_hash=f"sha256:parser-{token}",
        index_hash=f"sha256:index-{token}",
        status="ready",
        enabled=True,
        archived=False,
        chunks_count=3,
        entity_count=2,
        relation_count=1,
        error_code=None,
        error_message=None,
        metadata={
            "current_parse_generation_id": f"parse-generation-{token}",
            "current_build_generation_id": f"build-generation-{token}",
        },
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )


def _seed_job(kb_id: str, document_id: str, token: str) -> JobRecord:
    now = utc_now_iso()
    return JobRecord(
        id=f"job_seed_{token}",
        kb_id=kb_id,
        workspace=f"ws_{token}",
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


def _synchronized_write(
    original_write: _WriteMethod,
    entered: asyncio.Event,
    peer_entered: asyncio.Event,
) -> _WriteMethod:
    async def write(callback: _WriteCallback) -> Any:
        async def synchronized_callback(conn: Any) -> Any:
            entered.set()
            await asyncio.wait_for(peer_entered.wait(), timeout=10)
            return await callback(conn)

        return await original_write(synchronized_callback)

    return write


async def _backend_pids(
    first: PostgresMetadataStore,
    second: PostgresMetadataStore,
) -> tuple[int, int]:
    async with first._pool_or_raise().acquire() as first_conn:
        async with second._pool_or_raise().acquire() as second_conn:
            first_pid, second_pid = await asyncio.gather(
                first_conn.fetchval("SELECT pg_backend_pid()"),
                second_conn.fetchval("SELECT pg_backend_pid()"),
            )
    return int(first_pid), int(second_pid)


async def _delete_seeded_kb_metadata(store: PostgresMetadataStore, kb_id: str) -> None:
    async def delete(conn: Any) -> None:
        for table in (
            "kb_document_artifacts",
            "kb_config_versions",
            "kb_jobs",
            "kb_documents",
        ):
            await conn.execute(f"DELETE FROM {table} WHERE kb_id = $1", kb_id)

    await store._write(delete)


@pytest.mark.skipif(
    not _POSTGRES_DSN,
    reason=(
        "live PostgreSQL attempt-claim contract skipped: set "
        "LIGHTRAG_KB_POSTGRES_TEST_DSN or POSTGRES_TEST_DSN"
    ),
)
async def test_postgres_build_attempt_claim_has_one_durable_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _POSTGRES_DSN is not None
    token = uuid.uuid4().hex
    kb_id = f"kb_pg_attempt_claim_{token}"
    document_id = f"doc_pg_attempt_claim_{token}"
    stores = (
        PostgresMetadataStore(dsn=_POSTGRES_DSN, min_size=1, max_size=1),
        PostgresMetadataStore(dsn=_POSTGRES_DSN, min_size=1, max_size=1),
    )
    seeded = False

    try:
        await stores[0].initialize()
        await stores[1].initialize()

        first_pid, second_pid = await _backend_pids(*stores)
        assert first_pid > 0
        assert second_pid > 0
        assert first_pid != second_pid

        document = _ready_document(kb_id, document_id, token)
        created_documents, created_job, created = await stores[
            0
        ].create_documents_and_job([document], _seed_job(kb_id, document_id, token))
        seeded = True
        assert created is True
        assert created_documents == [document]
        assert created_job.document_id == document_id

        initial_documents = await asyncio.gather(
            stores[0].get_document(kb_id, document_id),
            stores[1].get_document(kb_id, document_id),
        )
        initial_snapshot = document_state_snapshot(initial_documents[0])
        assert document_state_snapshot(initial_documents[1]) == initial_snapshot

        job_ids = (f"job_claim_a_{token}", f"job_claim_b_{token}")
        claim_tokens = (f"build-claim-a-{token}", f"build-claim-b-{token}")
        pending_index_hash = f"sha256:index-next-{token}"

        async def claim(index: int) -> DocumentRecord:
            return await stores[index].claim_document_build_queued(
                kb_id,
                document_id,
                metadata_patch={
                    "pending_build_job_id": job_ids[index],
                    "pending_index_hash": pending_index_hash,
                },
                expected_snapshot=initial_snapshot,
                claim_token=claim_tokens[index],
            )

        entered = (asyncio.Event(), asyncio.Event())
        with monkeypatch.context() as claim_gate:
            claim_gate.setattr(
                stores[0],
                "_write",
                _synchronized_write(stores[0]._write, entered[0], entered[1]),
            )
            claim_gate.setattr(
                stores[1],
                "_write",
                _synchronized_write(stores[1]._write, entered[1], entered[0]),
            )
            outcomes = await asyncio.wait_for(
                asyncio.gather(claim(0), claim(1), return_exceptions=True),
                timeout=20,
            )

        assert entered[0].is_set()
        assert entered[1].is_set()
        winner_indexes = [
            index
            for index, outcome in enumerate(outcomes)
            if isinstance(outcome, DocumentRecord)
        ]
        assert len(winner_indexes) == 1, outcomes
        winner_index = winner_indexes[0]
        loser_index = 1 - winner_index
        winner_claim = outcomes[winner_index]
        loser_error = outcomes[loser_index]
        assert isinstance(winner_claim, DocumentRecord)
        assert type(loser_error) is DocumentSnapshotConflictError
        assert loser_error.entity_type == "document_snapshot"
        assert loser_error.entity_id == document_id
        assert loser_error.expected == initial_snapshot
        expected_loser_current = dict(initial_snapshot)
        expected_loser_current["status"] = "build_queued"
        assert loser_error.current == expected_loser_current

        winner_job_id = job_ids[winner_index]
        winner_claim_token = claim_tokens[winner_index]
        loser_job_id = job_ids[loser_index]
        loser_claim_token = claim_tokens[loser_index]
        assert winner_claim.status == "build_queued"
        assert winner_claim.metadata["pending_build_job_id"] == winner_job_id
        assert winner_claim.metadata["pending_build_claim_token"] == winner_claim_token

        durable_pending = await asyncio.gather(
            stores[0].get_document(kb_id, document_id),
            stores[1].get_document(kb_id, document_id),
        )
        for current in durable_pending:
            assert current.status == "build_queued"
            assert current.metadata["pending_build_job_id"] == winner_job_id
            assert current.metadata["pending_build_claim_token"] == winner_claim_token

        building = await stores[winner_index].mark_document_building(
            kb_id,
            document_id,
            metadata_patch={},
            job_id=winner_job_id,
            claim_token=winner_claim_token,
        )
        assert building.status == "building"
        assert building.metadata["pending_build_job_id"] is None
        assert building.metadata["pending_build_claim_token"] is None
        assert building.metadata["current_build_job_id"] == winner_job_id
        assert building.metadata["current_build_claim_token"] == winner_claim_token

        before_stale_write = await stores[loser_index].get_document(kb_id, document_id)
        with pytest.raises(DocumentAttemptOwnershipError) as ownership_conflict:
            await stores[loser_index].fail_document_build(
                kb_id,
                document_id,
                error_code="stale_attempt",
                error_message="stale attempt must not mutate the winner",
                metadata_patch={"stale_mutation": True},
                job_id=loser_job_id,
                claim_token=loser_claim_token,
            )
        assert type(ownership_conflict.value) is DocumentAttemptOwnershipError
        assert ownership_conflict.value.entity_type == "document_build_attempt"
        assert ownership_conflict.value.entity_id == document_id
        assert ownership_conflict.value.expected == {
            "status": "building",
            "job_id": loser_job_id,
            "claim_token": loser_claim_token,
        }
        assert ownership_conflict.value.current == {
            "status": "building",
            "job_id": winner_job_id,
            "claim_token": winner_claim_token,
        }
        after_stale_write = await stores[winner_index].get_document(kb_id, document_id)
        assert asdict(after_stale_write) == asdict(before_stale_write)

        stale_release = await stores[loser_index].release_document_build_if_owned(
            kb_id,
            document_id,
            job_id=loser_job_id,
            claim_token=loser_claim_token,
            error_code="stale_release",
            error_message="stale release must be a no-op",
            metadata_patch={"stale_release": True},
        )
        assert asdict(stale_release) == asdict(before_stale_write)

        final_documents = await asyncio.gather(
            stores[0].get_document(kb_id, document_id),
            stores[1].get_document(kb_id, document_id),
        )
        for current in final_documents:
            assert asdict(current) == asdict(before_stale_write)
            assert current.status == "building"
            assert current.metadata["current_build_job_id"] == winner_job_id
            assert current.metadata["current_build_claim_token"] == winner_claim_token
            assert "stale_mutation" not in current.metadata
            assert "stale_release" not in current.metadata
    finally:
        try:
            if seeded:
                await _delete_seeded_kb_metadata(stores[0], kb_id)
        finally:
            try:
                await stores[0].close()
            finally:
                await stores[1].close()
