"""Live PostgreSQL contract for cross-store artifact pointer promotion CAS."""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import cast

import pytest

from lightrag.api.kb_service import utc_now_iso
from lightrag.api.metadata_store import (
    ArtifactPointerConflictError,
    ArtifactRecord,
    DocumentAttemptOwnershipError,
    DocumentRecord,
    JobRecord,
    document_state_snapshot,
)
from lightrag.api.postgres_metadata_store import PostgresMetadataStore

_POSTGRES_TEST_DSN = os.getenv("LIGHTRAG_KB_POSTGRES_TEST_DSN") or os.getenv(
    "POSTGRES_TEST_DSN"
)
_INDEX_HASH = "sha256:postgres-pointer-cas-index"

pytestmark = [
    pytest.mark.offline,
    pytest.mark.requires_db,
    pytest.mark.skipif(
        not _POSTGRES_TEST_DSN,
        reason=(
            "live PostgreSQL pointer CAS test skipped: set "
            "LIGHTRAG_KB_POSTGRES_TEST_DSN or POSTGRES_TEST_DSN to enable"
        ),
    ),
]


def _document(kb_id: str, workspace: str, document_id: str) -> DocumentRecord:
    now = utc_now_iso()
    return DocumentRecord(
        id=document_id,
        kb_id=kb_id,
        workspace=workspace,
        lightrag_doc_id=None,
        source_type="upload",
        source_name=f"{document_id}.pdf",
        source_uri=f"/inputs/{document_id}.pdf",
        source_hash=f"sha256:source-{document_id}",
        content_type="application/pdf",
        size_bytes=17,
        parser_hash=None,
        index_hash=None,
        status="uploaded",
        enabled=True,
        archived=False,
        chunks_count=None,
        entity_count=None,
        relation_count=None,
        error_code=None,
        error_message=None,
        metadata={
            "process_options": "",
            "source_key": f"postgres-pointer-cas/{document_id}.pdf",
        },
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )


def _job(
    kb_id: str,
    workspace: str,
    document_id: str,
    job_id: str,
    *,
    job_type: str,
    status: str,
) -> JobRecord:
    now = utc_now_iso()
    terminal = status == "succeeded"
    started = status in {"running", "succeeded"}
    return JobRecord(
        id=job_id,
        kb_id=kb_id,
        workspace=workspace,
        batch_id=None,
        document_id=document_id,
        job_type=job_type,
        status=status,
        stage=job_type,
        progress=1.0 if terminal else 0.0,
        total_items=1,
        completed_items=1 if terminal else 0,
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
        started_at=now if started else None,
        finished_at=now if terminal else None,
        cancelled_at=None,
    )


def _artifact_pair(
    *,
    kb_id: str,
    workspace: str,
    document_id: str,
    sidecar_id: str,
    blocks_id: str,
    generation: str,
) -> list[ArtifactRecord]:
    now = utc_now_iso()
    base_uri = f"s3://postgres-pointer-cas.invalid/{workspace}/documents/{document_id}"
    return [
        ArtifactRecord(
            id=sidecar_id,
            kb_id=kb_id,
            workspace=workspace,
            document_id=document_id,
            artifact_type="sidecar",
            uri=f"{base_uri}/sidecar/{sidecar_id}/",
            checksum=f"sha256:{generation}-sidecar",
            size_bytes=None,
            metadata={"generation": generation, "is_directory": True},
            created_at=now,
        ),
        ArtifactRecord(
            id=blocks_id,
            kb_id=kb_id,
            workspace=workspace,
            document_id=document_id,
            artifact_type="blocks",
            uri=f"{base_uri}/blocks/{blocks_id}.jsonl",
            checksum=f"sha256:{generation}-blocks",
            size_bytes=23,
            metadata={"generation": generation},
            created_at=now,
        ),
    ]


async def _backend_pid(store: PostgresMetadataStore) -> int:
    # The metadata API has no public connection-identity probe. This narrow
    # live test uses the store's test-established pool accessor only to prove
    # that the two public API calls below run through independent sessions.
    async with store._pool_or_raise().acquire() as connection:
        return int(await connection.fetchval("SELECT pg_backend_pid()"))


async def test_postgres_artifact_pointer_promotion_is_cross_store_cas() -> None:
    run_id = uuid.uuid4().hex
    kb_id = f"kb_pg_pointer_cas_{run_id}"
    workspace = f"ws_pg_pointer_cas_{run_id}"
    document_id = f"doc_pg_pointer_cas_{run_id}"
    seed_job_id = f"job_pg_pointer_seed_{run_id}"
    build_job_id = f"job_pg_pointer_build_{run_id}"
    claim_token = f"build_attempt_pg_pointer_{run_id}"

    old_sidecar_id = f"artifact_pg_sidecar_old_{run_id}"
    old_blocks_id = f"artifact_pg_blocks_old_{run_id}"
    candidate_ids = {
        "alpha": (
            f"artifact_pg_sidecar_alpha_{run_id}",
            f"artifact_pg_blocks_alpha_{run_id}",
        ),
        "beta": (
            f"artifact_pg_sidecar_beta_{run_id}",
            f"artifact_pg_blocks_beta_{run_id}",
        ),
    }
    candidate_metrics = {
        "alpha": (11, 7, 5),
        "beta": (13, 9, 6),
    }

    store_alpha = PostgresMetadataStore(
        dsn=_POSTGRES_TEST_DSN,
        min_size=1,
        max_size=1,
        operation_lock_pool_max_size=1,
    )
    store_beta = PostgresMetadataStore(
        dsn=_POSTGRES_TEST_DSN,
        min_size=1,
        max_size=1,
        operation_lock_pool_max_size=1,
    )
    seeded = False

    try:
        await store_alpha.initialize()
        await store_beta.initialize()

        alpha_pid, beta_pid = await asyncio.gather(
            _backend_pid(store_alpha),
            _backend_pid(store_beta),
        )
        assert alpha_pid != beta_pid

        document = _document(kb_id, workspace, document_id)
        await store_alpha.create_documents_and_job(
            [document],
            _job(
                kb_id,
                workspace,
                document_id,
                seed_job_id,
                job_type="upload",
                status="succeeded",
            ),
        )
        seeded = True

        old_artifacts = _artifact_pair(
            kb_id=kb_id,
            workspace=workspace,
            document_id=document_id,
            sidecar_id=old_sidecar_id,
            blocks_id=old_blocks_id,
            generation="old",
        )
        parsed, created_old_artifacts = await store_alpha.complete_document_parse(
            kb_id,
            document_id,
            parser_hash="sha256:postgres-pointer-cas-parser",
            lightrag_doc_id=f"lightrag_{document_id}",
            metadata_patch={
                "current_sidecar_artifact_id": old_sidecar_id,
                "current_blocks_artifact_id": old_blocks_id,
                "last_parsed_at": utc_now_iso(),
            },
            artifacts=old_artifacts,
            expected_snapshot=document_state_snapshot(document),
        )
        assert {artifact.id for artifact in created_old_artifacts} == {
            old_sidecar_id,
            old_blocks_id,
        }

        ready = await store_alpha.complete_document_build(
            kb_id,
            document_id,
            index_hash=_INDEX_HASH,
            chunks_count=5,
            entity_count=3,
            relation_count=2,
            metadata_patch={"last_built_at": utc_now_iso()},
            expected_snapshot=document_state_snapshot(parsed),
        )
        assert ready.status == "ready"
        assert ready.metadata["current_sidecar_artifact_id"] == old_sidecar_id
        assert ready.metadata["current_blocks_artifact_id"] == old_blocks_id

        await store_alpha.create_job(
            _job(
                kb_id,
                workspace,
                document_id,
                build_job_id,
                job_type="build_kg",
                status="queued",
            )
        )
        await store_alpha.transition_job(
            kb_id,
            build_job_id,
            status="running",
            stage="building",
        )

        expected_snapshot = document_state_snapshot(ready)
        claimed = await store_alpha.claim_document_build_queued(
            kb_id,
            document_id,
            metadata_patch={
                "pending_build_job_id": build_job_id,
                "pending_index_hash": _INDEX_HASH,
                "force_rechunk": True,
                "force_extract": False,
                "force_embedding": False,
            },
            expected_snapshot=expected_snapshot,
            claim_token=claim_token,
        )
        assert claimed.metadata["pending_build_job_id"] == build_job_id
        assert claimed.metadata["pending_build_claim_token"] == claim_token

        building = await store_alpha.mark_document_building(
            kb_id,
            document_id,
            metadata_patch={
                "current_build_job_id": build_job_id,
                "build_started_at": utc_now_iso(),
            },
            job_id=build_job_id,
            claim_token=claim_token,
        )
        assert building.status == "building"
        assert building.metadata["current_build_job_id"] == build_job_id
        assert building.metadata["current_build_claim_token"] == claim_token
        assert building.metadata["pending_build_job_id"] is None
        assert building.metadata["pending_build_claim_token"] is None
        assert building.metadata["current_build_legacy_attempt"] is False
        assert building.metadata["pending_index_hash"] == _INDEX_HASH
        assert building.metadata["current_sidecar_artifact_id"] == old_sidecar_id
        assert building.metadata["current_blocks_artifact_id"] == old_blocks_id
        assert (
            await store_beta.get_document(kb_id, document_id)
        ).to_dict() == building.to_dict()
        assert (await store_beta.get_job(kb_id, build_job_id)).status == "running"

        (
            seeded_artifacts,
            seeded_artifact_total,
        ) = await store_beta.list_document_artifacts(
            kb_id,
            document_id,
            limit=20,
        )
        assert seeded_artifact_total == 2
        assert {artifact.id for artifact in seeded_artifacts} == {
            old_sidecar_id,
            old_blocks_id,
        }

        candidates: dict[str, list[ArtifactRecord]] = {}
        for label, (sidecar_id, blocks_id) in candidate_ids.items():
            candidates[label] = _artifact_pair(
                kb_id=kb_id,
                workspace=workspace,
                document_id=document_id,
                sidecar_id=sidecar_id,
                blocks_id=blocks_id,
                generation=label,
            )

        arrivals = 0
        arrival_lock = asyncio.Lock()
        both_arrived = asyncio.Event()

        async def promote(
            store: PostgresMetadataStore, label: str
        ) -> tuple[DocumentRecord, list[ArtifactRecord]]:
            nonlocal arrivals
            async with arrival_lock:
                arrivals += 1
                if arrivals == 2:
                    both_arrived.set()
            await asyncio.wait_for(both_arrived.wait(), timeout=10)

            sidecar_id, blocks_id = candidate_ids[label]
            chunks_count, entity_count, relation_count = candidate_metrics[label]
            return await store.complete_document_build_with_artifact_promotion(
                kb_id,
                document_id,
                index_hash=_INDEX_HASH,
                expected_current_sidecar_artifact_id=old_sidecar_id,
                expected_current_blocks_artifact_id=old_blocks_id,
                current_sidecar_artifact_id=sidecar_id,
                current_blocks_artifact_id=blocks_id,
                artifacts=candidates[label],
                chunks_count=chunks_count,
                entity_count=entity_count,
                relation_count=relation_count,
                metadata_patch={
                    "last_build_job_id": build_job_id,
                    "last_built_at": utc_now_iso(),
                    "build_skipped": False,
                    "pending_index_hash": None,
                    "promotion_candidate": label,
                },
                job_id=build_job_id,
                claim_token=claim_token,
                expected_snapshot=expected_snapshot,
            )

        labels = ("alpha", "beta")
        outcomes = await asyncio.wait_for(
            asyncio.gather(
                promote(store_alpha, labels[0]),
                promote(store_beta, labels[1]),
                return_exceptions=True,
            ),
            timeout=30,
        )
        success_indexes = [
            index
            for index, outcome in enumerate(outcomes)
            if not isinstance(outcome, BaseException)
        ]
        failure_indexes = [
            index
            for index, outcome in enumerate(outcomes)
            if isinstance(outcome, BaseException)
        ]
        assert len(success_indexes) == 1, outcomes
        assert len(failure_indexes) == 1, outcomes

        winner_index = success_indexes[0]
        loser_index = failure_indexes[0]
        winner_label = labels[winner_index]
        loser_label = labels[loser_index]
        completed_document, created_artifacts = cast(
            tuple[DocumentRecord, list[ArtifactRecord]], outcomes[winner_index]
        )
        conflict = outcomes[loser_index]
        assert isinstance(
            conflict,
            (ArtifactPointerConflictError, DocumentAttemptOwnershipError),
        )

        winner_sidecar_id, winner_blocks_id = candidate_ids[winner_label]
        loser_sidecar_id, loser_blocks_id = candidate_ids[loser_label]
        assert {artifact.id for artifact in created_artifacts} == {
            winner_sidecar_id,
            winner_blocks_id,
        }

        final_document = await store_alpha.get_document(kb_id, document_id)
        peer_document = await store_beta.get_document(kb_id, document_id)
        assert peer_document.to_dict() == final_document.to_dict()
        assert completed_document.to_dict() == final_document.to_dict()
        assert final_document.status == "ready"
        assert final_document.index_hash == _INDEX_HASH
        assert final_document.metadata["promotion_candidate"] == winner_label
        assert final_document.metadata["current_build_generation_id"] == claim_token
        assert final_document.metadata["current_build_job_id"] is None
        assert final_document.metadata["current_build_claim_token"] is None
        assert (
            final_document.metadata["current_sidecar_artifact_id"] == winner_sidecar_id
        )
        assert final_document.metadata["current_blocks_artifact_id"] == winner_blocks_id
        assert (
            final_document.chunks_count,
            final_document.entity_count,
            final_document.relation_count,
        ) == candidate_metrics[winner_label]

        expected_pointer_state = {
            "current_sidecar_artifact_id": old_sidecar_id,
            "current_blocks_artifact_id": old_blocks_id,
        }
        winning_pointer_state = {
            "current_sidecar_artifact_id": winner_sidecar_id,
            "current_blocks_artifact_id": winner_blocks_id,
        }
        if isinstance(conflict, ArtifactPointerConflictError):
            assert conflict.entity_type == "document_artifact_pointer"
            assert conflict.expected == expected_pointer_state
            assert conflict.current == winning_pointer_state
        else:
            assert conflict.entity_type == "document_build_attempt"
            assert conflict.expected == {
                "status": "building",
                "job_id": build_job_id,
                "claim_token": claim_token,
            }
            assert conflict.current == {
                "status": "ready",
                "job_id": None,
                "claim_token": None,
            }

        persisted_artifacts, total = await store_beta.list_document_artifacts(
            kb_id,
            document_id,
            limit=20,
        )
        persisted_by_id = {artifact.id: artifact for artifact in persisted_artifacts}
        assert total == 4
        assert set(persisted_by_id) == {
            old_sidecar_id,
            old_blocks_id,
            winner_sidecar_id,
            winner_blocks_id,
        }
        assert loser_sidecar_id not in persisted_by_id
        assert loser_blocks_id not in persisted_by_id
        assert persisted_by_id[winner_sidecar_id].metadata["generation"] == winner_label
        assert persisted_by_id[winner_blocks_id].metadata["generation"] == winner_label
        assert persisted_by_id[old_sidecar_id].metadata["generation"] == "old"
        assert persisted_by_id[old_blocks_id].metadata["generation"] == "old"

        stale_store = store_alpha if loser_index == 0 else store_beta
        before_stale_release = final_document.to_dict()
        stale_release = await stale_store.release_document_build_if_owned(
            kb_id,
            document_id,
            job_id=build_job_id,
            claim_token=claim_token,
            error_code="artifact_pointer_conflict",
            error_message="stale contender lost pointer promotion CAS",
            metadata_patch={"stale_release_candidate": loser_label},
        )
        assert stale_release.to_dict() == before_stale_release
        assert (
            await store_alpha.get_document(kb_id, document_id)
        ).to_dict() == before_stale_release
    finally:
        try:
            if seeded:
                await store_alpha.purge_kb_metadata(kb_id)
        finally:
            await asyncio.gather(store_alpha.close(), store_beta.close())
