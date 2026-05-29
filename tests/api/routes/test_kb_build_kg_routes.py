from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightrag.api.document_lifecycle_service import DocumentLifecycleService
from lightrag.api.index_build_service import (
    IndexBuildPlan,
    IndexBuildService,
    _collect_doc_status,
)
from lightrag.api.job_service import JobService
from lightrag.api.kb_service import KnowledgeBaseService
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry, LightRAGLike
from lightrag.api.metadata_store import DocumentRecord, SQLiteMetadataStore

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_kb_routes = importlib.import_module("lightrag.api.routers.kb_routes")
_kb_document_routes = importlib.import_module("lightrag.api.routers.kb_document_routes")
sys.argv = _original_argv

create_kb_routes = _kb_routes.create_kb_routes
create_kb_document_routes = _kb_document_routes.create_kb_document_routes

pytestmark = pytest.mark.offline

_API_KEY = "test-key"
_HEADERS = {"X-API-Key": _API_KEY}


class FakeDocStatus:
    def __init__(self):
        self.rows: dict[str, dict] = {}

    async def get_by_ids(self, ids):
        return [self.rows.get(item_id) for item_id in ids]

    def stamp_processed(
        self,
        doc_id: str,
        *,
        chunks_count: int = 5,
        entity_count: int = 12,
        relation_count: int = 7,
    ) -> None:
        self.rows[doc_id] = {
            "status": "processed",
            "chunks_count": chunks_count,
            "entity_count": entity_count,
            "relation_count": relation_count,
        }


class FakeDeletionResult:
    def __init__(self, doc_id: str, delete_llm_cache: bool):
        self.status = "success"
        self.doc_id = doc_id
        self.delete_llm_cache = delete_llm_cache


def _document_record(
    *,
    document_id: str = "doc_kb",
    lightrag_doc_id: str = "doc-lr",
) -> DocumentRecord:
    return DocumentRecord(
        id=document_id,
        kb_id="kb_build",
        workspace="workspace",
        lightrag_doc_id=lightrag_doc_id,
        source_type="file",
        source_name="paper.pdf",
        source_uri="paper.pdf",
        source_hash="sha256:source",
        content_type="application/pdf",
        size_bytes=10,
        parser_hash="sha256:parser",
        index_hash=None,
        status="parsed",
        enabled=True,
        archived=False,
        chunks_count=None,
        entity_count=None,
        relation_count=None,
        error_code=None,
        error_message=None,
        metadata={},
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        deleted_at=None,
    )


def _build_plan(document: DocumentRecord) -> IndexBuildPlan:
    return IndexBuildPlan(
        document=document,
        sidecar_uri=None,
        blocks_path=None,
        parser_hash=document.parser_hash or "",
        index_hash="sha256:index",
        process_options="",
        force_rechunk=False,
        force_extract=False,
        force_embedding=False,
    )


class FakeRAG:
    def __init__(
        self,
        workspace: str,
        *,
        build_should_fail_for: set[str] | None = None,
    ):
        self.workspace = workspace
        self.embedding_dim = 768
        self.chunk_token_size = 512
        self.chunk_overlap_token_size = 64
        self.tiktoken_model_name = "gpt-4o-mini"
        self.summary_max_tokens = 800
        self.force_llm_summary_on_merge = False
        self.addon_params = {
            "chunker": {"strategy": "F", "F": {"chunk_size": 512}},
            "entity_types": ["concept", "person"],
            "language": "en",
            "extraction": {"prompt_version": "v1"},
        }

        class _EmbeddingFunc:
            __name__ = "fake_embed"
            func_name = "fake_embed"

        self.embedding_func = _EmbeddingFunc()
        self.doc_status = FakeDocStatus()
        self.build_should_fail_for = build_should_fail_for or set()
        self.enqueue_calls: list[dict] = []
        self.delete_calls: list[tuple[str, bool]] = []
        self.process_calls: int = 0

    async def finalize_storages(self) -> None:
        return None

    async def adelete_by_doc_id(self, doc_id, *, delete_llm_cache=False):
        self.delete_calls.append((doc_id, delete_llm_cache))
        return FakeDeletionResult(doc_id, delete_llm_cache)

    async def parse_native(self, doc_id, file_path, content_data):
        return await self._parse("native", doc_id, file_path, content_data)

    async def parse_mineru(self, doc_id, file_path, content_data):
        return await self._parse("mineru", doc_id, file_path, content_data)

    async def parse_docling(self, doc_id, file_path, content_data):
        return await self._parse("docling", doc_id, file_path, content_data)

    async def _parse(self, engine, doc_id, file_path, content_data):
        source_path = Path(file_path)
        parsed_dir = source_path.parent / "__parsed__" / f"{source_path.name}.parsed"
        parsed_dir.mkdir(parents=True, exist_ok=True)
        blocks_path = parsed_dir / f"{source_path.stem}.blocks.jsonl"
        blocks_path.write_text(
            '{"type":"meta"}\n{"type":"content","content":"hello"}\n', encoding="utf-8"
        )
        return {
            "doc_id": doc_id,
            "file_path": file_path,
            "parse_format": "lightrag",
            "content": "hello",
            "blocks_path": str(blocks_path),
            "parse_stage_skipped": False,
        }

    async def apipeline_enqueue_documents(
        self,
        input,
        *,
        ids,
        file_paths,
        track_id,
        docs_format,
        lightrag_document_paths,
        parse_engine=None,
        process_options=None,
    ):
        call = {
            "ids": list(ids),
            "file_paths": list(file_paths),
            "docs_format": docs_format,
            "lightrag_document_paths": list(lightrag_document_paths),
            "process_options": process_options,
        }
        self.enqueue_calls.append(call)
        for doc_id in ids:
            doc_record_id = doc_id
            if doc_id in self.build_should_fail_for:
                raise RuntimeError(f"index pipeline exploded for {doc_id}")
            self.doc_status.stamp_processed(doc_record_id)
        return track_id

    async def apipeline_process_enqueue_documents(self):
        self.process_calls += 1


class BuilderProbe:
    def __init__(self, *, build_should_fail_for: set[str] | None = None):
        self.build_should_fail_for = build_should_fail_for or set()
        self.instances: list[FakeRAG] = []

    async def build(self, record) -> FakeRAG:
        rag = FakeRAG(
            record.workspace, build_should_fail_for=self.build_should_fail_for
        )
        self.instances.append(rag)
        return rag

    async def finalize(self, rag: LightRAGLike) -> None:
        return None


def _build_client(
    tmp_path: Path,
    *,
    probe: BuilderProbe | None = None,
    index_service_factory: Callable[[DocumentLifecycleService], IndexBuildService] | None = None,
):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "knowledge_bases.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    document_service = DocumentLifecycleService(
        kb_service, metadata_store, tmp_path / "inputs"
    )
    job_service = JobService(kb_service, metadata_store)
    index_service = (
        index_service_factory(document_service)
        if index_service_factory is not None
        else IndexBuildService(document_service)
    )
    probe = probe or BuilderProbe()
    registry = LightRAGInstanceRegistry(kb_service, probe.build, probe.finalize)
    app = FastAPI()
    app.include_router(
        create_kb_routes(kb_service, registry, api_key=_API_KEY, job_service=job_service)
    )
    app.include_router(
        create_kb_document_routes(
            document_service,
            job_service,
            api_key=_API_KEY,
            registry=registry,
            index_service=index_service,
        )
    )
    return TestClient(app), kb_service, document_service, job_service, probe


def _create_kb(client: TestClient, kb_id: str):
    response = client.post("/kbs", json={"id": kb_id, "name": kb_id}, headers=_HEADERS)
    assert response.status_code == 200
    return response.json()


def _upload_and_parse(
    client: TestClient,
    kb_id: str,
    *,
    filename: str = "paper.pdf",
    content: bytes = b"pdf-bytes",
    engine: str = "mineru",
):
    upload = client.post(
        f"/kbs/{kb_id}/documents:upload",
        files=[("files", (filename, content, "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200, upload.text
    document_id = upload.json()["documents"][0]["id"]
    parse = client.post(
        f"/kbs/{kb_id}/documents/{document_id}:parse",
        json={"engine": engine, "process_options": "iF"},
        headers=_HEADERS,
    )
    assert parse.status_code == 200, parse.text
    detail = client.get(f"/kbs/{kb_id}/documents/{document_id}", headers=_HEADERS)
    assert detail.status_code == 200
    assert detail.json()["status"] == "parsed"
    return document_id


async def test_collect_doc_status_missing_row_fails():
    rag = FakeRAG("workspace")
    plan = _build_plan(_document_record())

    with pytest.raises(RuntimeError, match="did not create doc_status row"):
        await _collect_doc_status(rag, plan)


def test_build_kg_succeeds_and_stamps_index_hash(tmp_path):
    client, *_ = _build_client(tmp_path)
    _create_kb(client, "kb_build")
    document_id = _upload_and_parse(client, "kb_build")

    response = client.post(
        f"/kbs/kb_build/documents/{document_id}:build-kg",
        json={},
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    job_id = response.json()["id"]
    job = client.get(f"/kbs/kb_build/jobs/{job_id}", headers=_HEADERS).json()
    assert job["job_type"] == "build_kg"
    assert job["status"] == "succeeded"
    assert job["result"]["chunks_count"] == 5
    assert job["result"]["entity_count"] == 12
    assert job["result"]["relation_count"] == 7
    assert job["result"]["skipped"] is False
    assert job["result"]["index_hash"].startswith("sha256:")

    detail = client.get(f"/kbs/kb_build/documents/{document_id}", headers=_HEADERS)
    payload = detail.json()
    assert payload["status"] == "ready"
    assert payload["index_hash"] == job["result"]["index_hash"]
    assert payload["chunks_count"] == 5
    assert payload["entity_count"] == 12
    assert payload["relation_count"] == 7


def test_build_kg_skips_when_index_hash_matches(tmp_path):
    client, *_, probe = _build_client(tmp_path)
    _create_kb(client, "kb_skip")
    document_id = _upload_and_parse(client, "kb_skip")

    first = client.post(
        f"/kbs/kb_skip/documents/{document_id}:build-kg",
        json={},
        headers=_HEADERS,
    )
    assert first.status_code == 200
    rag = probe.instances[0]
    assert len(rag.enqueue_calls) == 1
    assert rag.process_calls == 1

    second = client.post(
        f"/kbs/kb_skip/documents/{document_id}:build-kg",
        json={},
        headers=_HEADERS,
    )
    assert second.status_code == 200
    body = second.json()
    assert body["status"] == "succeeded"
    assert body["result"]["skipped"] is True
    assert body["result"]["skip_reason"] == "index_hash_match"
    # Skipped path must not re-run the pipeline
    assert len(rag.enqueue_calls) == 1
    assert rag.process_calls == 1


def test_replace_then_build_reindexes_new_source(tmp_path):
    client, *_, probe = _build_client(tmp_path)
    _create_kb(client, "kb_replace_build")
    document_id = _upload_and_parse(client, "kb_replace_build")
    first = client.post(
        f"/kbs/kb_replace_build/documents/{document_id}:build-kg",
        json={},
        headers=_HEADERS,
    )
    assert first.status_code == 200
    rag = probe.instances[0]
    assert len(rag.enqueue_calls) == 1
    ready = client.get(f"/kbs/kb_replace_build/documents/{document_id}", headers=_HEADERS)
    assert ready.status_code == 200
    old_lightrag_doc_id = ready.json()["lightrag_doc_id"]
    assert ready.json()["status"] == "ready"

    replace = client.post(
        f"/kbs/kb_replace_build/documents/{document_id}:replace"
        "?auto_parse=true&parser_engine=mineru&process_options=iF",
        files={"file": ("paper-v2.pdf", b"new-pdf", "application/pdf")},
        headers=_HEADERS,
    )
    assert replace.status_code == 200, replace.text
    replace_job = client.get(
        f"/kbs/kb_replace_build/jobs/{replace.json()['id']}", headers=_HEADERS
    )
    assert replace_job.status_code == 200
    assert replace_job.json()["status"] == "succeeded"
    assert rag.delete_calls == [(old_lightrag_doc_id, False)]
    after_replace = client.get(
        f"/kbs/kb_replace_build/documents/{document_id}", headers=_HEADERS
    )
    assert after_replace.status_code == 200
    after_payload = after_replace.json()
    assert after_payload["status"] == "parsed"
    assert after_payload["index_hash"] is None
    assert after_payload["source_name"] == "paper-v2.pdf"

    second = client.post(
        f"/kbs/kb_replace_build/documents/{document_id}:build-kg",
        json={},
        headers=_HEADERS,
    )
    assert second.status_code == 200
    second_job = client.get(
        f"/kbs/kb_replace_build/jobs/{second.json()['id']}", headers=_HEADERS
    )
    assert second_job.status_code == 200
    assert second_job.json()["status"] == "succeeded"
    assert second_job.json()["result"]["skipped"] is False
    assert len(rag.enqueue_calls) == 2
    assert rag.process_calls == 2


def test_skipped_build_claim_blocks_delete_race(tmp_path):
    class RacingIndexBuildService(IndexBuildService):
        async def create_build_plan(
            self,
            kb_id,
            document_id,
            *,
            rag,
            force_rechunk=False,
            force_extract=False,
            force_embedding=False,
        ):
            plan = await super().create_build_plan(
                kb_id,
                document_id,
                rag=rag,
                force_rechunk=force_rechunk,
                force_extract=force_extract,
                force_embedding=force_embedding,
            )
            if plan.skipped:
                await self._document_service.metadata_store.claim_document_deleting(
                    kb_id,
                    document_id,
                    metadata_patch={"pending_delete_job_id": "job_delete_race"},
                )
            return plan

    client, _kb_service, _document_service, _job_service, _probe = _build_client(
        tmp_path,
        index_service_factory=RacingIndexBuildService,
    )
    _create_kb(client, "kb_skip_race")
    document_id = _upload_and_parse(client, "kb_skip_race")

    first = client.post(
        f"/kbs/kb_skip_race/documents/{document_id}:build-kg",
        json={},
        headers=_HEADERS,
    )
    assert first.status_code == 200

    second = client.post(
        f"/kbs/kb_skip_race/documents/{document_id}:build-kg",
        json={},
        headers=_HEADERS,
    )

    assert second.status_code == 409
    detail = second.json()["detail"]
    assert detail["error_code"] == "delete_job_active"
    assert detail["existing_job_id"] == "job_delete_race"
    document = client.get(f"/kbs/kb_skip_race/documents/{document_id}", headers=_HEADERS)
    assert document.status_code == 200
    assert document.json()["status"] == "deleting"


def test_batch_skipped_build_claim_blocks_delete_race(tmp_path):
    class RacingIndexBuildService(IndexBuildService):
        def __init__(self, document_service: DocumentLifecycleService):
            super().__init__(document_service)
            self.raced = False

        async def create_build_plan(
            self,
            kb_id,
            document_id,
            *,
            rag,
            force_rechunk=False,
            force_extract=False,
            force_embedding=False,
        ):
            plan = await super().create_build_plan(
                kb_id,
                document_id,
                rag=rag,
                force_rechunk=force_rechunk,
                force_extract=force_extract,
                force_embedding=force_embedding,
            )
            if plan.skipped and not self.raced:
                self.raced = True
                await self._document_service.metadata_store.claim_document_deleting(
                    kb_id,
                    document_id,
                    metadata_patch={"pending_delete_job_id": "job_batch_delete_race"},
                )
            return plan

    client, _kb_service, _document_service, _job_service, _probe = _build_client(
        tmp_path,
        index_service_factory=RacingIndexBuildService,
    )
    _create_kb(client, "kb_batch_skip_race")
    doc_a = _upload_and_parse(client, "kb_batch_skip_race", filename="a.pdf")
    doc_b = _upload_and_parse(client, "kb_batch_skip_race", filename="b.pdf")
    for document_id in (doc_a, doc_b):
        first = client.post(
            f"/kbs/kb_batch_skip_race/documents/{document_id}:build-kg",
            json={},
            headers=_HEADERS,
        )
        assert first.status_code == 200

    response = client.post(
        "/kbs/kb_batch_skip_race/documents:batch-build-kg",
        json={"document_ids": [doc_a, doc_b]},
        headers=_HEADERS,
    )

    assert response.status_code == 200
    job = client.get(f"/kbs/kb_batch_skip_race/jobs/{response.json()['job_id']}", headers=_HEADERS)
    assert job.status_code == 200
    payload = job.json()
    assert payload["status"] == "failed"
    assert payload["completed_items"] == 1
    assert payload["failed_items"] == 1
    assert payload["result"]["summary"]["outcome"] == "partial_failure"
    failures = [
        item for item in payload["result"]["items"] if item["status"] == "failed"
    ]
    assert failures[0]["error_code"] == "delete_job_active"
    assert failures[0]["existing_job_id"] == "job_batch_delete_race"


def test_reindex_forces_rebuild_even_when_index_hash_matches(tmp_path):
    client, *_, probe = _build_client(tmp_path)
    _create_kb(client, "kb_reindex")
    document_id = _upload_and_parse(client, "kb_reindex")

    client.post(
        f"/kbs/kb_reindex/documents/{document_id}:build-kg",
        json={},
        headers=_HEADERS,
    )
    rag = probe.instances[0]
    assert len(rag.enqueue_calls) == 1

    response = client.post(
        f"/kbs/kb_reindex/documents/{document_id}:reindex",
        json={},
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    job_id = response.json()["id"]
    final = client.get(f"/kbs/kb_reindex/jobs/{job_id}", headers=_HEADERS).json()
    assert final["status"] == "succeeded"
    assert final["result"]["skipped"] is False
    assert len(rag.enqueue_calls) == 2


def test_build_kg_rejects_unparsed_document(tmp_path):
    client, *_ = _build_client(tmp_path)
    _create_kb(client, "kb_unparsed")
    upload = client.post(
        "/kbs/kb_unparsed/documents:upload",
        files=[("files", ("a.pdf", b"raw", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]

    response = client.post(
        f"/kbs/kb_unparsed/documents/{document_id}:build-kg",
        json={},
        headers=_HEADERS,
    )
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["error_code"] == "document_not_parsed"
    assert detail["current_status"] == "uploaded"


def test_batch_build_kg_isolates_per_item_failures(tmp_path):
    probe = BuilderProbe()
    client, *_ = _build_client(tmp_path, probe=probe)
    _create_kb(client, "kb_batch")

    doc_a = _upload_and_parse(client, "kb_batch", filename="a.pdf", content=b"a-bytes")
    doc_b = _upload_and_parse(client, "kb_batch", filename="b.pdf", content=b"b-bytes")
    detail_b = client.get(
        f"/kbs/kb_batch/documents/{doc_b}", headers=_HEADERS
    ).json()
    fail_doc_id = detail_b["lightrag_doc_id"]

    # Tell the rag instance to fail when ingesting doc_b
    rag = probe.instances[0]
    rag.build_should_fail_for = {fail_doc_id}

    response = client.post(
        "/kbs/kb_batch/documents:batch-build-kg",
        json={"document_ids": [doc_a, doc_b]},
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]

    job = client.get(f"/kbs/kb_batch/jobs/{job_id}", headers=_HEADERS).json()
    assert job["status"] == "failed"
    assert job["completed_items"] == 1
    assert job["failed_items"] == 1
    items_by_doc = {item["document_id"]: item for item in job["result"]["items"]}
    assert items_by_doc[doc_a]["status"] == "succeeded"
    assert items_by_doc[doc_b]["status"] == "failed"

    detail_a = client.get(f"/kbs/kb_batch/documents/{doc_a}", headers=_HEADERS).json()
    detail_b_after = client.get(
        f"/kbs/kb_batch/documents/{doc_b}", headers=_HEADERS
    ).json()
    assert detail_a["status"] == "ready"
    assert detail_b_after["status"] == "build_failed"


def test_incremental_build_does_not_touch_existing_documents(tmp_path):
    """Adding a new file to a KB that already has a built document must only
    rebuild the new document; the existing document's state and stats remain."""
    probe = BuilderProbe()
    client, *_ = _build_client(tmp_path, probe=probe)
    _create_kb(client, "kb_increment")

    doc_a = _upload_and_parse(
        client, "kb_increment", filename="a.pdf", content=b"a-bytes"
    )
    first_build = client.post(
        f"/kbs/kb_increment/documents/{doc_a}:build-kg",
        json={},
        headers=_HEADERS,
    )
    assert first_build.status_code == 200
    detail_a_before = client.get(
        f"/kbs/kb_increment/documents/{doc_a}", headers=_HEADERS
    ).json()
    assert detail_a_before["status"] == "ready"
    rag = probe.instances[0]
    enqueue_count_before = len(rag.enqueue_calls)

    # Upload a new file and build only the new document
    doc_b = _upload_and_parse(
        client, "kb_increment", filename="b.pdf", content=b"b-bytes"
    )
    second_build = client.post(
        f"/kbs/kb_increment/documents/{doc_b}:build-kg",
        json={},
        headers=_HEADERS,
    )
    assert second_build.status_code == 200
    job_id = second_build.json()["id"]
    final = client.get(f"/kbs/kb_increment/jobs/{job_id}", headers=_HEADERS).json()
    assert final["status"] == "succeeded"
    assert final["result"]["skipped"] is False

    # Pipeline ran exactly once more — for the new doc only
    assert len(rag.enqueue_calls) == enqueue_count_before + 1
    last_call = rag.enqueue_calls[-1]
    detail_b_after = client.get(
        f"/kbs/kb_increment/documents/{doc_b}", headers=_HEADERS
    ).json()
    assert last_call["ids"] == [detail_b_after["lightrag_doc_id"]]

    # doc_a remains ready with same index_hash and unchanged counts
    detail_a_after = client.get(
        f"/kbs/kb_increment/documents/{doc_a}", headers=_HEADERS
    ).json()
    assert detail_a_after["status"] == "ready"
    assert detail_a_after["index_hash"] == detail_a_before["index_hash"]
    assert detail_a_after["chunks_count"] == detail_a_before["chunks_count"]
    assert detail_a_after["entity_count"] == detail_a_before["entity_count"]


def test_active_build_conflict_returns_409(tmp_path):
    client, _kb_service, document_service, _job_service, _probe = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_active")
    document_id = _upload_and_parse(client, "kb_active")

    # Mark document as build_queued via the metadata store directly
    import asyncio

    async def claim():
        store = document_service.metadata_store
        await store.claim_document_build_queued(
            "kb_active",
            document_id,
            metadata_patch={"pending_build_job_id": "job_in_flight"},
        )

    asyncio.run(claim())
    response = client.post(
        f"/kbs/kb_active/documents/{document_id}:build-kg",
        json={},
        headers=_HEADERS,
    )
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["error_code"] == "build_job_active"
    assert detail["document_id"] == document_id


def test_cancel_queued_job_marks_cancelled(tmp_path):
    client, *_, _ = _build_client(tmp_path)
    _create_kb(client, "kb_cancel")
    upload = client.post(
        "/kbs/kb_cancel/documents:upload?auto_parse=true",
        files=[("files", ("a.pdf", b"raw-bytes", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    job_id = upload.json()["job_id"]

    cancel = client.post(
        f"/kbs/kb_cancel/jobs/{job_id}:cancel", headers=_HEADERS
    )
    assert cancel.status_code == 200, cancel.text
    body = cancel.json()
    assert body["status"] == "cancelled"
    assert body["error_code"] == "cancelled_by_user"


def test_retry_failed_job_resets_to_queued(tmp_path):
    client, _kb_service, _document_service, job_service, _probe = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_retry")
    document_id = _upload_and_parse(client, "kb_retry")

    # Manually create + fail a build job to test retry without running the worker
    import asyncio

    async def setup_failed_job():
        record = await _kb_service.get("kb_retry")
        from lightrag.api.metadata_store import JobRecord
        from lightrag.utils import generate_track_id
        from lightrag.api.kb_service import utc_now_iso

        now = utc_now_iso()
        job = JobRecord(
            id=generate_track_id("job_build"),
            kb_id=record.id,
            workspace=record.workspace,
            batch_id=None,
            document_id=document_id,
            job_type="build_kg",
            status="queued",
            stage="building",
            progress=0.0,
            total_items=1,
            completed_items=0,
            failed_items=0,
            idempotency_key="retry-key",
            config_version_id=None,
            config_hash=None,
            retry_count=0,
            max_retries=3,
            payload={"document_id": document_id},
            result=None,
            error_code=None,
            error_message=None,
            created_at=now,
            updated_at=now,
            queued_at=now,
            started_at=None,
            finished_at=None,
            cancelled_at=None,
        )
        created = await _document_service.metadata_store.create_job(job)
        await _document_service.metadata_store.transition_job(
            record.id,
            created.id,
            status="failed",
            progress=1.0,
            failed_items=1,
            error_code="build_failed",
            error_message="boom",
        )
        return created.id

    job_id = asyncio.run(setup_failed_job())

    retry = client.post(
        f"/kbs/kb_retry/jobs/{job_id}:retry",
        json={"idempotency_key": "retry-key-2"},
        headers=_HEADERS,
    )
    assert retry.status_code == 200, retry.text
    payload = retry.json()
    assert payload["status"] == "queued"
    assert payload["retry_count"] == 1
    assert payload["error_code"] is None
    assert payload["idempotency_key"] == "retry-key-2"

    # Retrying again until exhausted
    retry_again = client.post(
        f"/kbs/kb_retry/jobs/{job_id}:retry", json={}, headers=_HEADERS
    )
    # Cannot retry queued — must fail it first
    assert retry_again.status_code == 409


def test_wait_for_job_returns_terminal_state(tmp_path):
    client, *_ = _build_client(tmp_path)
    _create_kb(client, "kb_wait")
    document_id = _upload_and_parse(client, "kb_wait")

    response = client.post(
        f"/kbs/kb_wait/documents/{document_id}:build-kg",
        json={},
        headers=_HEADERS,
    )
    assert response.status_code == 200
    job_id = response.json()["id"]

    waited = client.post(
        f"/kbs/kb_wait/jobs/{job_id}:wait?timeout_seconds=10",
        headers=_HEADERS,
    )
    assert waited.status_code == 200, waited.text
    assert waited.json()["status"] == "succeeded"


def test_wait_for_job_returns_408_on_timeout(tmp_path):
    """If the job never reaches a terminal state inside the timeout window,
    :wait returns 408 with the current status — not 200."""
    client, _kb_service, document_service, _job_service, _probe = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_wait_timeout")

    import asyncio

    from lightrag.api.metadata_store import JobRecord
    from lightrag.api.kb_service import utc_now_iso
    from lightrag.utils import generate_track_id

    async def seed_running_job() -> str:
        record = await document_service.kb_service.get("kb_wait_timeout")
        now = utc_now_iso()
        job = JobRecord(
            id=generate_track_id("job_running"),
            kb_id=record.id,
            workspace=record.workspace,
            batch_id=None,
            document_id=None,
            job_type="parse",
            status="queued",
            stage="parsing",
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
            started_at=None,
            finished_at=None,
            cancelled_at=None,
        )
        created = await document_service.metadata_store.create_job(job)
        await document_service.metadata_store.transition_job(
            record.id, created.id, status="running", progress=0.5
        )
        return created.id

    job_id = asyncio.run(seed_running_job())
    response = client.post(
        f"/kbs/kb_wait_timeout/jobs/{job_id}:wait?timeout_seconds=0.3&poll_interval_seconds=0.1",
        headers=_HEADERS,
    )
    assert response.status_code == 408
    detail = response.json()["detail"]
    assert detail["error_code"] == "wait_timeout"
    assert detail["current_status"] == "running"
