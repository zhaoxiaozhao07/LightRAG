from __future__ import annotations

import asyncio
from typing import cast

import pytest

from lightrag.api.index_build_service import IndexBuildService
from lightrag.api.job_worker import build_delete_executor
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry
from tests.api.routes import test_kb_build_kg_routes as kg

pytestmark = pytest.mark.offline


def test_durable_delete_executor_rebuild_subgraph_uses_pre_delete_footprint(
    tmp_path,
):
    """Durable delete worker preserves rebuild_subgraph precision after restart."""
    client, kb_service, document_service, job_service, probe = kg._build_client(tmp_path)
    kb_id = "kb_delexec_subgraph"
    kg._create_kb(client, kb_id)
    doc_drop = kg._upload_and_parse(client, kb_id, filename="drop.pdf")
    doc_aff = kg._upload_and_parse(client, kb_id, filename="affected.pdf")
    doc_unrel = kg._upload_and_parse(client, kb_id, filename="unrelated.pdf")
    lightrag_ids: dict[str, str] = {}
    for document_id in (doc_drop, doc_aff, doc_unrel):
        built = client.post(
            f"/kbs/{kb_id}/documents/{document_id}:build-kg",
            json={},
            headers=kg._HEADERS,
        )
        assert built.status_code == 200, built.text
        detail = client.get(
            f"/kbs/{kb_id}/documents/{document_id}", headers=kg._HEADERS
        ).json()
        lightrag_ids[document_id] = detail["lightrag_doc_id"]

    metadata_store = document_service.metadata_store
    index_service = IndexBuildService(document_service)
    registry = LightRAGInstanceRegistry(kb_service, probe.build, probe.finalize)

    async def _drive() -> tuple[str, kg.FakeRAG]:
        worker_rag = cast(kg.FakeRAG, await registry.get(kb_id))
        worker_rag.full_entities.seed(
            lightrag_ids[doc_drop], {"entity_names": ["SHARED", "GONE"]}
        )
        worker_rag.full_entities.seed(
            lightrag_ids[doc_aff], {"entity_names": ["SHARED", "OTHER"]}
        )
        worker_rag.full_entities.seed(
            lightrag_ids[doc_unrel], {"entity_names": ["LONELY"]}
        )

        job, _created = await job_service.create_delete_job_once(
            kb_id,
            document_id=doc_drop,
            lightrag_doc_id=lightrag_ids[doc_drop],
            delete_source_file=False,
            delete_artifacts=False,
            delete_llm_cache=False,
            strategy="rebuild_subgraph",
        )
        await metadata_store.claim_document_deleting(
            kb_id,
            doc_drop,
            metadata_patch={"pending_delete_job_id": job.id},
        )
        await document_service.fail_delete(
            kb_id,
            doc_drop,
            job_id=job.id,
            error_code="worker_orphaned",
            error_message="crash",
        )
        claimed = await metadata_store.claim_next_worker_job(
            job_types=["delete"], max_queued_at=None
        )
        assert claimed is not None and claimed.id == job.id
        executor = build_delete_executor(
            document_service=document_service,
            registry=registry,
            job_service=job_service,
            index_service=index_service,
        )
        await executor(claimed)
        return job.id, worker_rag

    try:
        job_id, worker_rag = asyncio.run(_drive())
    finally:
        asyncio.run(registry.shutdown())

    job = client.get(f"/kbs/{kb_id}/jobs/{job_id}", headers=kg._HEADERS).json()
    assert job["status"] == "succeeded", job
    assert job["result"]["resumed_by_worker"] is True
    rebuild = job["result"]["rebuild"]
    assert rebuild["strategy"] == "rebuild_subgraph"
    assert rebuild["footprint_entities"] == 2
    assert rebuild["affected_documents"] == 1
    assert rebuild["rebuilt_documents"] == 1
    assert rebuild["failed_documents"] == 0
    assert worker_rag.delete_calls == [
        (lightrag_ids[doc_drop], False),
        (lightrag_ids[doc_aff], False),
    ]
    assert worker_rag.full_entities.rows.get(lightrag_ids[doc_drop]) is None
    assert worker_rag.doc_status.stamp_counts.get(lightrag_ids[doc_aff], 0) == 1
    assert worker_rag.doc_status.stamp_counts.get(lightrag_ids[doc_unrel], 0) == 0


def test_durable_delete_executor_rebuild_subgraph_resumes_after_delete_completed(
    tmp_path,
):
    """Retry completes subgraph rebuild when crash happens after destructive delete."""
    client, kb_service, document_service, job_service, probe = kg._build_client(tmp_path)
    kb_id = "kb_delexec_subgraph_postdelete"
    kg._create_kb(client, kb_id)
    doc_drop = kg._upload_and_parse(client, kb_id, filename="drop.pdf")
    doc_aff = kg._upload_and_parse(client, kb_id, filename="affected.pdf")
    doc_unrel = kg._upload_and_parse(client, kb_id, filename="unrelated.pdf")
    lightrag_ids: dict[str, str] = {}
    for document_id in (doc_drop, doc_aff, doc_unrel):
        built = client.post(
            f"/kbs/{kb_id}/documents/{document_id}:build-kg",
            json={},
            headers=kg._HEADERS,
        )
        assert built.status_code == 200, built.text
        detail = client.get(
            f"/kbs/{kb_id}/documents/{document_id}", headers=kg._HEADERS
        ).json()
        lightrag_ids[document_id] = detail["lightrag_doc_id"]

    metadata_store = document_service.metadata_store
    index_service = IndexBuildService(document_service)
    registry = LightRAGInstanceRegistry(kb_service, probe.build, probe.finalize)

    async def _drive() -> tuple[str, kg.FakeRAG]:
        worker_rag = cast(kg.FakeRAG, await registry.get(kb_id))
        worker_rag.full_entities.seed(
            lightrag_ids[doc_drop], {"entity_names": ["SHARED", "GONE"]}
        )
        worker_rag.full_entities.seed(
            lightrag_ids[doc_aff], {"entity_names": ["SHARED", "OTHER"]}
        )
        worker_rag.full_entities.seed(
            lightrag_ids[doc_unrel], {"entity_names": ["LONELY"]}
        )
        job, _created = await job_service.create_delete_job_once(
            kb_id,
            document_id=doc_drop,
            lightrag_doc_id=lightrag_ids[doc_drop],
            delete_source_file=False,
            delete_artifacts=False,
            delete_llm_cache=False,
            strategy="rebuild_subgraph",
        )
        document = await document_service.claim_delete(
            kb_id,
            doc_drop,
            job=job,
            delete_source_file=False,
            delete_artifacts=False,
        )
        await job_service.transition_job(kb_id, job.id, status="running", progress=0.2)
        footprint = await kg._kb_document_routes._capture_graph_footprint(
            rag=worker_rag,
            lightrag_doc_id=document.lightrag_doc_id,
        )
        await job_service.update_job_payload_patch(
            kb_id,
            job.id,
            payload_patch={
                "rebuild_subgraph_footprints": [
                    kg._kb_document_routes._serialize_graph_footprint(
                        footprint,
                        document_id=document.id,
                        lightrag_doc_id=document.lightrag_doc_id,
                    )
                ]
            },
        )
        item = await kg._kb_document_routes._execute_delete_document_impl(
            document_service=document_service,
            kb_id=kb_id,
            job_id=job.id,
            document=document,
            active_registry=registry,
            delete_source_file=False,
            delete_artifacts=False,
            delete_llm_cache=False,
        )
        assert item["status"] == "succeeded", item
        await job_service.transition_job(
            kb_id,
            job.id,
            status="failed",
            progress=1.0,
            failed_items=1,
            error_code="worker_orphaned",
            error_message="crash after delete before rebuild",
        )
        await job_service.retry_job(kb_id, job.id)
        claimed = await metadata_store.claim_next_worker_job(
            job_types=["delete"], max_queued_at=None
        )
        assert claimed is not None and claimed.id == job.id
        executor = build_delete_executor(
            document_service=document_service,
            registry=registry,
            job_service=job_service,
            index_service=index_service,
        )
        await executor(claimed)
        return job.id, worker_rag

    try:
        job_id, worker_rag = asyncio.run(_drive())
    finally:
        asyncio.run(registry.shutdown())

    job = client.get(f"/kbs/{kb_id}/jobs/{job_id}", headers=kg._HEADERS).json()
    assert job["status"] == "succeeded", job
    assert job["result"]["resumed_by_worker"] is True
    rebuild = job["result"]["rebuild"]
    assert rebuild["strategy"] == "rebuild_subgraph"
    assert rebuild["footprint_entities"] == 2
    assert rebuild["affected_documents"] == 1
    assert rebuild["rebuilt_documents"] == 1
    assert rebuild["failed_documents"] == 0
    assert worker_rag.delete_calls == [
        (lightrag_ids[doc_drop], False),
        (lightrag_ids[doc_aff], False),
    ]
    assert worker_rag.doc_status.stamp_counts.get(lightrag_ids[doc_aff], 0) == 1
    assert worker_rag.doc_status.stamp_counts.get(lightrag_ids[doc_unrel], 0) == 0


def test_durable_batch_delete_executor_rebuild_subgraph_resumes_after_deletes_completed(
    tmp_path,
):
    """Batch retry rebuilds from persisted footprints after all deletes finished."""
    client, kb_service, document_service, job_service, probe = kg._build_client(tmp_path)
    kb_id = "kb_batch_delexec_subgraph_postdelete"
    kg._create_kb(client, kb_id)
    doc_drop_a = kg._upload_and_parse(client, kb_id, filename="drop-a.pdf")
    doc_drop_b = kg._upload_and_parse(client, kb_id, filename="drop-b.pdf")
    doc_aff = kg._upload_and_parse(client, kb_id, filename="affected.pdf")
    doc_unrel = kg._upload_and_parse(client, kb_id, filename="unrelated.pdf")
    lightrag_ids: dict[str, str] = {}
    for document_id in (doc_drop_a, doc_drop_b, doc_aff, doc_unrel):
        built = client.post(
            f"/kbs/{kb_id}/documents/{document_id}:build-kg",
            json={},
            headers=kg._HEADERS,
        )
        assert built.status_code == 200, built.text
        detail = client.get(
            f"/kbs/{kb_id}/documents/{document_id}", headers=kg._HEADERS
        ).json()
        lightrag_ids[document_id] = detail["lightrag_doc_id"]

    metadata_store = document_service.metadata_store
    index_service = IndexBuildService(document_service)
    registry = LightRAGInstanceRegistry(kb_service, probe.build, probe.finalize)

    async def _drive() -> tuple[str, kg.FakeRAG]:
        worker_rag = cast(kg.FakeRAG, await registry.get(kb_id))
        worker_rag.full_entities.seed(
            lightrag_ids[doc_drop_a], {"entity_names": ["SHARED", "GONE_A"]}
        )
        worker_rag.full_entities.seed(
            lightrag_ids[doc_drop_b], {"entity_names": ["GONE_B"]}
        )
        worker_rag.full_entities.seed(
            lightrag_ids[doc_aff], {"entity_names": ["SHARED", "OTHER"]}
        )
        worker_rag.full_entities.seed(
            lightrag_ids[doc_unrel], {"entity_names": ["LONELY"]}
        )
        job, _created = await job_service.create_batch_delete_job_once(
            kb_id,
            batch_id="batch_delete_subgraph_resume_after_delete",
            document_ids=[doc_drop_a, doc_drop_b],
            delete_source_file=False,
            delete_artifacts=False,
            delete_llm_cache=False,
            strategy="rebuild_subgraph",
        )
        documents, failures = await document_service.claim_batch_delete(
            kb_id,
            [doc_drop_a, doc_drop_b],
            job=job,
            delete_source_file=False,
            delete_artifacts=False,
        )
        assert not failures
        await job_service.transition_job(kb_id, job.id, status="running", progress=0.2)
        serialized_footprints: list[dict[str, object]] = []
        for document in documents:
            footprint = await kg._kb_document_routes._capture_graph_footprint(
                rag=worker_rag,
                lightrag_doc_id=document.lightrag_doc_id,
            )
            serialized_footprints.append(
                kg._kb_document_routes._serialize_graph_footprint(
                    footprint,
                    document_id=document.id,
                    lightrag_doc_id=document.lightrag_doc_id,
                )
            )
        await job_service.update_job_payload_patch(
            kb_id,
            job.id,
            payload_patch={"rebuild_subgraph_footprints": serialized_footprints},
        )
        for document in documents:
            item = await kg._kb_document_routes._execute_delete_document_impl(
                document_service=document_service,
                kb_id=kb_id,
                job_id=job.id,
                document=document,
                active_registry=registry,
                delete_source_file=False,
                delete_artifacts=False,
                delete_llm_cache=False,
            )
            assert item["status"] == "succeeded", item
        await job_service.transition_job(
            kb_id,
            job.id,
            status="failed",
            progress=1.0,
            completed_items=2,
            failed_items=0,
            error_code="worker_orphaned",
            error_message="crash after batch delete before rebuild",
        )
        await job_service.retry_job(kb_id, job.id)
        claimed = await metadata_store.claim_next_worker_job(
            job_types=["delete"], max_queued_at=None
        )
        assert claimed is not None and claimed.id == job.id
        executor = build_delete_executor(
            document_service=document_service,
            registry=registry,
            job_service=job_service,
            index_service=index_service,
        )
        await executor(claimed)
        return job.id, worker_rag

    try:
        job_id, worker_rag = asyncio.run(_drive())
    finally:
        asyncio.run(registry.shutdown())

    job = client.get(f"/kbs/{kb_id}/jobs/{job_id}", headers=kg._HEADERS).json()
    assert job["status"] == "succeeded", job
    assert job["result"]["resumed_by_worker"] is True
    assert job["completed_items"] == 2
    assert job["failed_items"] == 0
    rebuild = job["result"]["rebuild"]
    assert rebuild["strategy"] == "rebuild_subgraph"
    assert rebuild["footprint_entities"] == 3
    assert rebuild["affected_documents"] == 1
    assert rebuild["rebuilt_documents"] == 1
    assert rebuild["failed_documents"] == 0
    assert worker_rag.delete_calls == [
        (lightrag_ids[doc_drop_a], False),
        (lightrag_ids[doc_drop_b], False),
        (lightrag_ids[doc_aff], False),
    ]
    assert worker_rag.doc_status.stamp_counts.get(lightrag_ids[doc_aff], 0) == 1
    assert worker_rag.doc_status.stamp_counts.get(lightrag_ids[doc_unrel], 0) == 0
