import asyncio
import importlib
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from lightrag.api.document_lifecycle_service import (
    DocumentLifecycleService,
    DocumentSourceInput,
)
from lightrag.api.job_service import JobService
from lightrag.api.kb_service import KnowledgeBaseService, sanitize_workspace
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry, LightRAGLike
from lightrag.api.metadata_store import InvalidJobTransitionError, SQLiteMetadataStore
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

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


class FakeRAG:
    def __init__(
        self,
        workspace: str,
        *,
        should_fail: bool = False,
        fail_source_names: set[str] | None = None,
    ):
        self.workspace = workspace
        self.should_fail = should_fail
        self.fail_source_names = fail_source_names or set()
        self.parse_calls = []
        self.delete_calls = []

    async def finalize_storages(self) -> None:
        return None

    async def parse_native(self, doc_id: str, file_path: str, content_data):
        return await self._parse("native", doc_id, file_path, content_data)

    async def parse_mineru(self, doc_id: str, file_path: str, content_data):
        return await self._parse("mineru", doc_id, file_path, content_data)

    async def parse_docling(self, doc_id: str, file_path: str, content_data):
        return await self._parse("docling", doc_id, file_path, content_data)

    async def apipeline_enqueue_documents(self, *args, **kwargs):
        raise AssertionError("KB parse endpoint must not enqueue indexing pipeline")

    async def apipeline_process_enqueue_documents(self, *args, **kwargs):
        raise AssertionError("KB parse endpoint must not process indexing pipeline")

    async def adelete_by_doc_id(self, doc_id: str, delete_llm_cache: bool = False):
        self.delete_calls.append((doc_id, delete_llm_cache))
        return FakeDeletionResult(
            status="success",
            doc_id=doc_id,
            message="deleted",
            status_code=200,
            file_path="",
        )

    async def _parse(self, engine: str, doc_id: str, file_path: str, content_data):
        self.parse_calls.append((engine, doc_id, file_path, content_data))
        source_path = Path(file_path)
        if self.should_fail or source_path.name in self.fail_source_names:
            raise RuntimeError("parser exploded")
        parsed_dir = source_path.parent / "__parsed__" / f"{source_path.name}.parsed"
        parsed_dir.mkdir(parents=True, exist_ok=True)
        blocks_path = parsed_dir / f"{source_path.stem}.blocks.jsonl"
        blocks_path.write_text('{"type":"content","text":"parsed"}\n', encoding="utf-8")
        if engine == "mineru":
            raw_dir = parsed_dir.parent / f"{source_path.name}.mineru_raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / "full.md").write_text("# parsed", encoding="utf-8")
        if content_data.get("archive_source_after_parse", True):
            source_path.unlink()
        return {
            "doc_id": doc_id,
            "file_path": file_path,
            "parse_format": "lightrag",
            "content": "parsed",
            "blocks_path": str(blocks_path),
            "parse_stage_skipped": False,
        }


class FakeDeletionResult(BaseModel):
    status: str
    doc_id: str
    message: str
    status_code: int
    file_path: str | None = None


class BuilderProbe:
    def __init__(
        self,
        *,
        should_fail: bool = False,
        fail_source_names: set[str] | None = None,
    ):
        self.should_fail = should_fail
        self.fail_source_names = fail_source_names or set()
        self.instances: list[FakeRAG] = []

    async def build(self, record) -> FakeRAG:
        rag = FakeRAG(
            record.workspace,
            should_fail=self.should_fail,
            fail_source_names=self.fail_source_names,
        )
        self.instances.append(rag)
        return rag

    async def finalize(self, rag: LightRAGLike) -> None:
        return None


def _build_client(tmp_path: Path, *, probe: BuilderProbe | None = None):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "knowledge_bases.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    document_service = DocumentLifecycleService(
        kb_service, metadata_store, tmp_path / "inputs"
    )
    job_service = JobService(kb_service, metadata_store)
    probe = probe or BuilderProbe()
    registry = LightRAGInstanceRegistry(kb_service, probe.build, probe.finalize)
    app = FastAPI()
    app.include_router(
        create_kb_routes(
            kb_service, registry, api_key=_API_KEY, job_service=job_service
        )
    )
    app.include_router(
        create_kb_document_routes(
            document_service, job_service, api_key=_API_KEY, registry=registry
        )
    )
    return TestClient(app), kb_service, metadata_store, document_service, job_service


def _create_kb(client: TestClient, kb_id: str):
    response = client.post("/kbs", json={"id": kb_id, "name": kb_id}, headers=_HEADERS)
    assert response.status_code == 200
    return response.json()


def _upload_and_parse_document(
    client: TestClient,
    kb_id: str,
    *,
    filename: str = "paper.pdf",
    content: bytes = b"pdf",
    content_type: str = "application/pdf",
):
    upload = client.post(
        f"/kbs/{kb_id}/documents:upload",
        files=[("files", (filename, content, content_type))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]

    parse = client.post(
        f"/kbs/{kb_id}/documents/{document_id}:parse",
        json={"engine": "mineru", "process_options": "iF"},
        headers=_HEADERS,
    )
    assert parse.status_code == 200

    artifacts = client.get(
        f"/kbs/{kb_id}/documents/{document_id}/artifacts", headers=_HEADERS
    )
    assert artifacts.status_code == 200
    artifacts_by_type = {
        item["artifact_type"]: item for item in artifacts.json()["artifacts"]
    }
    return document_id, artifacts_by_type


def test_upload_persists_documents_jobs_and_running_status(tmp_path):
    initialize_share_data()
    try:
        client, _kb_service, _store, _document_service, _job_service = _build_client(
            tmp_path
        )
        kb = _create_kb(client, "kb_upload")

        response = client.post(
            "/kbs/kb_upload/documents:upload?auto_parse=true&auto_index=false",
            files=[
                ("files", ("alpha.txt", b"alpha", "text/plain")),
                ("files", ("beta.txt", b"beta", "text/plain")),
            ],
            headers=_HEADERS,
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["job_id"].startswith("job_parse_")
        assert payload["batch_id"].startswith("batch_")
        assert [doc["status"] for doc in payload["documents"]] == [
            "parse_queued",
            "parse_queued",
        ]
        assert {doc["workspace"] for doc in payload["documents"]} == {kb["workspace"]}
        assert all(Path(doc["source_uri"]).exists() for doc in payload["documents"])
        assert (tmp_path / "metadata" / "metadata.sqlite3").exists()

        list_response = client.get("/kbs/kb_upload/documents", headers=_HEADERS)
        assert list_response.status_code == 200
        listed = list_response.json()
        assert listed["total"] == 2
        assert {doc["source_name"] for doc in listed["documents"]} == {
            "alpha.txt",
            "beta.txt",
        }

        document_id = payload["documents"][0]["id"]
        detail_response = client.get(
            f"/kbs/kb_upload/documents/{document_id}", headers=_HEADERS
        )
        assert detail_response.status_code == 200
        assert detail_response.json()["id"] == document_id

        jobs_response = client.get(
            "/kbs/kb_upload/jobs?status=queued", headers=_HEADERS
        )
        assert jobs_response.status_code == 200
        jobs = jobs_response.json()
        assert jobs["total"] == 1
        assert jobs["jobs"][0]["id"] == payload["job_id"]
        assert jobs["jobs"][0]["job_type"] == "parse"

        status_response = client.get("/kbs/kb_upload/status", headers=_HEADERS)
        assert status_response.status_code == 200
        status = status_response.json()
        assert status["kb"]["workspace"] == sanitize_workspace("kb_upload")
        assert [job["id"] for job in status["running_jobs"]] == [payload["job_id"]]
    finally:
        finalize_share_data()


def test_text_import_is_metadata_only_and_kb_scoped(tmp_path):
    client, _kb_service, store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_texts")
    _create_kb(client, "kb_other")

    response = client.post(
        "/kbs/kb_texts/documents:texts",
        json={
            "documents": [
                {
                    "text": "hello metadata",
                    "source_name": "note.md",
                    "metadata": {"tag": "unit"},
                }
            ],
            "auto_parse": False,
        },
        headers=_HEADERS,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"].startswith("job_upload_")
    assert payload["documents"][0]["status"] == "uploaded"
    assert payload["documents"][0]["metadata"]["tag"] == "unit"

    own_response = client.get("/kbs/kb_texts/documents", headers=_HEADERS)
    other_response = client.get("/kbs/kb_other/documents", headers=_HEADERS)
    assert own_response.status_code == 200
    assert other_response.status_code == 200
    assert own_response.json()["total"] == 1
    assert other_response.json()["total"] == 0

    reopened = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    documents, total = asyncio.run(reopened.list_documents("kb_texts"))
    assert total == 1
    assert documents[0].source_name == "note.md"
    assert documents[0].status == "uploaded"

    jobs, total_jobs = asyncio.run(store.list_jobs("kb_texts"))
    assert total_jobs == 1
    assert jobs[0].status == "succeeded"
    assert jobs[0].progress == 1.0


def test_text_import_idempotency_key_reuses_existing_batch(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_text_idem")
    request = {
        "documents": [
            {
                "text": "hello idempotency",
                "source_name": "idem.txt",
                "metadata": {"tag": "first"},
            }
        ],
        "idempotency_key": "idem-text-1",
    }

    first = client.post(
        "/kbs/kb_text_idem/documents:texts", json=request, headers=_HEADERS
    )
    second = client.post(
        "/kbs/kb_text_idem/documents:texts",
        json=request,
        headers=_HEADERS,
    )

    assert first.status_code == 200
    assert second.status_code == 200
    first_payload = first.json()
    second_payload = second.json()
    assert second_payload["job_id"] == first_payload["job_id"]
    assert second_payload["batch_id"] == first_payload["batch_id"]
    assert second_payload["documents"][0]["id"] == first_payload["documents"][0]["id"]
    assert second_payload["documents"][0]["source_name"] == "idem.txt"

    listed = client.get("/kbs/kb_text_idem/documents", headers=_HEADERS)
    assert listed.status_code == 200
    assert listed.json()["total"] == 1
    jobs = client.get("/kbs/kb_text_idem/jobs", headers=_HEADERS)
    assert jobs.status_code == 200
    assert jobs.json()["total"] == 1

    conflict = client.post(
        "/kbs/kb_text_idem/documents:texts",
        json={
            **request,
            "documents": [
                {
                    "text": "different body must conflict",
                    "source_name": "different.txt",
                }
            ],
        },
        headers=_HEADERS,
    )
    assert conflict.status_code == 409


@pytest.mark.asyncio
async def test_text_import_idempotency_key_is_atomic_for_concurrent_batches(tmp_path):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "knowledge_bases.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    document_service = DocumentLifecycleService(
        kb_service, metadata_store, tmp_path / "inputs"
    )
    await kb_service.create(kb_id="kb_text_concurrent", name="Concurrent")
    source = DocumentSourceInput(
        source_name="same.txt",
        content=b"same content",
        source_type="text",
        content_type="text/plain",
        metadata={"tag": "same"},
    )

    first, second = await asyncio.gather(
        document_service.create_source_batch(
            "kb_text_concurrent", [source], idempotency_key="same-key"
        ),
        document_service.create_source_batch(
            "kb_text_concurrent", [source], idempotency_key="same-key"
        ),
    )

    assert first.job.id == second.job.id
    assert first.documents[0].id == second.documents[0].id
    documents, total = await document_service.list_documents("kb_text_concurrent")
    assert total == 1
    assert documents[0].source_name == "same.txt"
    jobs, total_jobs = await metadata_store.list_jobs("kb_text_concurrent")
    assert total_jobs == 1
    assert jobs[0].id == first.job.id
    workspace_dir = tmp_path / "inputs" / sanitize_workspace("kb_text_concurrent")
    assert [path.name for path in workspace_dir.iterdir()] == [documents[0].id]


def test_list_documents_source_name_filter_and_patch_metadata_flags(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_patch_doc")
    upload = client.post(
        "/kbs/kb_patch_doc/documents:upload",
        files=[
            ("files", ("Alpha Report.pdf", b"alpha", "application/pdf")),
            ("files", ("beta-notes.txt", b"beta", "text/plain")),
        ],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    alpha = next(
        document
        for document in upload.json()["documents"]
        if document["source_name"] == "Alpha Report.pdf"
    )

    filtered = client.get(
        "/kbs/kb_patch_doc/documents?source_name=alpha", headers=_HEADERS
    )
    assert filtered.status_code == 200
    assert filtered.json()["total"] == 1
    assert filtered.json()["documents"][0]["id"] == alpha["id"]

    patched = client.patch(
        f"/kbs/kb_patch_doc/documents/{alpha['id']}",
        json={"metadata": {"reviewed": True}, "enabled": False, "archived": True},
        headers=_HEADERS,
    )
    assert patched.status_code == 200
    patched_payload = patched.json()
    assert patched_payload["enabled"] is False
    assert patched_payload["archived"] is True
    assert patched_payload["metadata"]["reviewed"] is True
    assert patched_payload["metadata"]["batch_id"] == alpha["metadata"]["batch_id"]

    empty_patch = client.patch(
        f"/kbs/kb_patch_doc/documents/{alpha['id']}", json={}, headers=_HEADERS
    )
    assert empty_patch.status_code == 400

    reserved_patch = client.patch(
        f"/kbs/kb_patch_doc/documents/{alpha['id']}",
        json={"metadata": {"pending_parse_job_id": "job_fake"}},
        headers=_HEADERS,
    )
    assert reserved_patch.status_code == 422


def test_enable_disable_document_actions_update_metadata_only(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_enable_disable")
    upload = client.post(
        "/kbs/kb_enable_disable/documents:upload",
        files=[("files", ("paper.pdf", b"pdf", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]

    disabled = client.post(
        f"/kbs/kb_enable_disable/documents/{document_id}:disable",
        headers=_HEADERS,
    )
    enabled = client.post(
        f"/kbs/kb_enable_disable/documents/{document_id}:enable",
        headers=_HEADERS,
    )

    assert disabled.status_code == 200
    assert disabled.json()["enabled"] is False
    assert disabled.json()["status"] == "uploaded"
    assert enabled.status_code == 200
    assert enabled.json()["enabled"] is True
    assert enabled.json()["status"] == "uploaded"


def test_delete_uploaded_unindexed_document_soft_deletes_without_lightrag(tmp_path):
    probe = BuilderProbe()
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_delete_uploaded")
    upload = client.post(
        "/kbs/kb_delete_uploaded/documents:upload",
        files=[("files", ("draft.txt", b"draft", "text/plain"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document = upload.json()["documents"][0]

    response = client.delete(
        f"/kbs/kb_delete_uploaded/documents/{document['id']}?idempotency_key=delete-draft",
        headers=_HEADERS,
    )

    assert response.status_code == 200
    job_id = response.json()["id"]
    retry = client.delete(
        f"/kbs/kb_delete_uploaded/documents/{document['id']}?idempotency_key=delete-draft",
        headers=_HEADERS,
    )
    assert retry.status_code == 200
    assert retry.json()["id"] == job_id
    conflict = client.delete(
        f"/kbs/kb_delete_uploaded/documents/{document['id']}"
        "?idempotency_key=delete-draft&delete_source_file=true",
        headers=_HEADERS,
    )
    assert conflict.status_code == 409
    job = client.get(f"/kbs/kb_delete_uploaded/jobs/{job_id}", headers=_HEADERS)
    assert job.status_code == 200
    assert job.json()["status"] == "succeeded"
    assert (
        job.json()["result"]["items"][0]["lightrag_delete_result"]["status"]
        == "skipped"
    )
    assert probe.instances == []

    assert (
        client.get(
            f"/kbs/kb_delete_uploaded/documents/{document['id']}", headers=_HEADERS
        ).status_code
        == 404
    )
    listed = client.get("/kbs/kb_delete_uploaded/documents", headers=_HEADERS)
    assert listed.status_code == 200
    assert listed.json()["total"] == 0
    assert Path(document["source_uri"]).exists()


def test_delete_ready_document_invokes_lightrag_and_removes_files_when_requested(
    tmp_path,
):
    probe = BuilderProbe()
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_delete_ready")
    document_id, artifacts = _upload_and_parse_document(client, "kb_delete_ready")
    document = client.get(
        f"/kbs/kb_delete_ready/documents/{document_id}", headers=_HEADERS
    )
    assert document.status_code == 200
    document_payload = document.json()
    source_path = Path(document_payload["source_uri"])
    sidecar_path = Path(artifacts["sidecar"]["uri"])
    lightrag_doc_id = document_payload["lightrag_doc_id"]

    response = client.delete(
        f"/kbs/kb_delete_ready/documents/{document_id}"
        "?delete_source_file=true&delete_artifacts=true&delete_llm_cache=true",
        headers=_HEADERS,
    )

    assert response.status_code == 200
    job = client.get(
        f"/kbs/kb_delete_ready/jobs/{response.json()['id']}", headers=_HEADERS
    )
    assert job.status_code == 200
    assert job.json()["status"] == "succeeded"
    assert probe.instances[-1].delete_calls == [(lightrag_doc_id, True)]
    assert not source_path.exists()
    assert not sidecar_path.exists()
    assert (
        client.get(
            f"/kbs/kb_delete_ready/documents/{document_id}", headers=_HEADERS
        ).status_code
        == 404
    )


def test_batch_delete_partial_failure_for_active_build_and_missing_doc(tmp_path):
    client, _kb_service, store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_batch_delete")
    upload = client.post(
        "/kbs/kb_batch_delete/documents:upload",
        files=[
            ("files", ("active.txt", b"active", "text/plain")),
            ("files", ("ok.txt", b"ok", "text/plain")),
        ],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    active_id, ok_id = [document["id"] for document in upload.json()["documents"]]
    asyncio.run(
        store.claim_document_build_queued(
            "kb_batch_delete",
            active_id,
            metadata_patch={"pending_build_job_id": "job_active_build"},
            require_parsed=False,
        )
    )

    response = client.post(
        "/kbs/kb_batch_delete/documents:batch-delete",
        json={"document_ids": [active_id, ok_id, "doc_missing"]},
        headers=_HEADERS,
    )

    assert response.status_code == 200
    job = client.get(
        f"/kbs/kb_batch_delete/jobs/{response.json()['id']}", headers=_HEADERS
    )
    assert job.status_code == 200
    job_payload = job.json()
    assert job_payload["status"] == "failed"
    assert job_payload["completed_items"] == 1
    assert job_payload["failed_items"] == 2
    assert job_payload["result"]["summary"]["outcome"] == "partial_failure"
    failures = {
        item["document_id"]: item
        for item in job_payload["result"]["items"]
        if item["status"] == "failed"
    }
    assert failures[active_id]["error_code"] == "build_job_active"
    assert failures[active_id]["existing_job_id"] == "job_active_build"
    assert failures["doc_missing"]["error_code"] == "document_not_found"
    assert (
        client.get(
            f"/kbs/kb_batch_delete/documents/{ok_id}", headers=_HEADERS
        ).status_code
        == 404
    )
    active = client.get(f"/kbs/kb_batch_delete/documents/{active_id}", headers=_HEADERS)
    assert active.status_code == 200
    assert active.json()["status"] == "build_queued"


def test_active_delete_blocks_parse_claim(tmp_path):
    client, _kb_service, store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_delete_parse_guard")
    upload = client.post(
        "/kbs/kb_delete_parse_guard/documents:upload",
        files=[("files", ("guard.pdf", b"guard", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]
    asyncio.run(
        store.claim_document_deleting(
            "kb_delete_parse_guard",
            document_id,
            metadata_patch={"pending_delete_job_id": "job_delete_guard"},
        )
    )

    response = client.post(
        f"/kbs/kb_delete_parse_guard/documents/{document_id}:parse",
        json={"engine": "mineru", "process_options": "iF"},
        headers=_HEADERS,
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["error_code"] == "delete_job_active"
    assert detail["existing_job_id"] == "job_delete_guard"
    document = client.get(
        f"/kbs/kb_delete_parse_guard/documents/{document_id}", headers=_HEADERS
    )
    assert document.status_code == 200
    assert document.json()["status"] == "deleting"


def test_active_build_blocks_parse_claim(tmp_path):
    client, _kb_service, store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_build_parse_guard")
    upload = client.post(
        "/kbs/kb_build_parse_guard/documents:upload",
        files=[("files", ("guard.pdf", b"guard", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]
    asyncio.run(
        store.claim_document_build_queued(
            "kb_build_parse_guard",
            document_id,
            metadata_patch={"pending_build_job_id": "job_build_guard"},
            require_parsed=False,
        )
    )

    response = client.post(
        f"/kbs/kb_build_parse_guard/documents/{document_id}:parse",
        json={"engine": "mineru", "process_options": "iF"},
        headers=_HEADERS,
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["error_code"] == "build_job_active"
    assert detail["existing_job_id"] == "job_build_guard"
    document = client.get(
        f"/kbs/kb_build_parse_guard/documents/{document_id}", headers=_HEADERS
    )
    assert document.status_code == 200
    assert document.json()["status"] == "build_queued"

    jobs, _total = asyncio.run(store.list_jobs("kb_build_parse_guard"))
    failed_parse_jobs = [
        job
        for job in jobs
        if job.job_type == "parse" and job.document_id == document_id
    ]
    assert len(failed_parse_jobs) == 1
    assert failed_parse_jobs[0].status == "failed"
    assert failed_parse_jobs[0].error_code == "build_job_active"


def test_delete_artifact_cleanup_rejects_workspace_escape(tmp_path):
    probe = BuilderProbe()
    client, _kb_service, store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_delete_escape")
    document_id, artifacts = _upload_and_parse_document(client, "kb_delete_escape")
    sibling_dir = Path(artifacts["original"]["uri"]).parent.parent / "doc_sibling"
    sibling_dir.mkdir()
    escaped_path = sibling_dir / "escaped-delete.txt"
    escaped_path.write_text("outside", encoding="utf-8")
    with store._connect() as conn:
        conn.execute(
            "UPDATE document_artifacts SET uri = ? WHERE id = ?",
            (str(escaped_path), artifacts["blocks"]["id"]),
        )
        conn.commit()

    response = client.delete(
        f"/kbs/kb_delete_escape/documents/{document_id}?delete_artifacts=true",
        headers=_HEADERS,
    )

    assert response.status_code == 200
    job = client.get(
        f"/kbs/kb_delete_escape/jobs/{response.json()['id']}", headers=_HEADERS
    )
    assert job.status_code == 200
    assert job.json()["status"] == "failed"
    assert job.json()["error_code"] == "delete_failed"
    assert escaped_path.exists()
    document = client.get(
        f"/kbs/kb_delete_escape/documents/{document_id}", headers=_HEADERS
    )
    assert document.status_code == 200
    assert document.json()["status"] == "delete_failed"


def test_replace_ready_document_resets_source_artifacts_and_old_index(tmp_path):
    probe = BuilderProbe()
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_replace_ready")
    document_id, artifacts = _upload_and_parse_document(client, "kb_replace_ready")
    before = client.get(
        f"/kbs/kb_replace_ready/documents/{document_id}", headers=_HEADERS
    )
    assert before.status_code == 200
    before_payload = before.json()
    old_source_path = Path(before_payload["source_uri"])
    old_sidecar_path = Path(artifacts["sidecar"]["uri"])
    old_lightrag_doc_id = before_payload["lightrag_doc_id"]

    response = client.post(
        f"/kbs/kb_replace_ready/documents/{document_id}:replace"
        "?delete_llm_cache=true&idempotency_key=replace-ready-1",
        files={"file": ("paper-v2.pdf", b"new-pdf", "application/pdf")},
        headers=_HEADERS,
    )

    assert response.status_code == 200, response.text
    job = client.get(
        f"/kbs/kb_replace_ready/jobs/{response.json()['id']}", headers=_HEADERS
    )
    assert job.status_code == 200
    job_payload = job.json()
    assert job_payload["status"] == "succeeded"
    assert job_payload["job_type"] == "replace"
    assert job_payload["result"]["previous_lightrag_doc_id"] == old_lightrag_doc_id
    assert probe.instances[-1].delete_calls == [(old_lightrag_doc_id, True)]

    detail = client.get(
        f"/kbs/kb_replace_ready/documents/{document_id}", headers=_HEADERS
    )
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["id"] == document_id
    assert payload["status"] == "uploaded"
    assert payload["source_name"] == "paper-v2.pdf"
    assert payload["source_hash"] != before_payload["source_hash"]
    assert payload["lightrag_doc_id"] is None
    assert payload["parser_hash"] is None
    assert payload["index_hash"] is None
    assert payload["chunks_count"] is None
    assert payload["metadata"]["last_replace_job_id"] == response.json()["id"]
    assert "blocks_path" not in payload["metadata"]
    new_source_path = Path(payload["source_uri"])
    assert new_source_path.exists()
    assert new_source_path.read_bytes() == b"new-pdf"
    assert not old_source_path.exists()
    assert not old_sidecar_path.exists()

    artifacts_after = client.get(
        f"/kbs/kb_replace_ready/documents/{document_id}/artifacts", headers=_HEADERS
    )
    assert artifacts_after.status_code == 200
    assert artifacts_after.json()["total"] == 0


def test_replace_idempotency_key_reuses_existing_job_and_conflicts(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_replace_idem")
    upload = client.post(
        "/kbs/kb_replace_idem/documents:upload",
        files=[("files", ("paper.pdf", b"pdf", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]

    first = client.post(
        f"/kbs/kb_replace_idem/documents/{document_id}:replace"
        "?idempotency_key=replace-idem-1",
        files={"file": ("paper-v2.pdf", b"new", "application/pdf")},
        headers=_HEADERS,
    )
    second = client.post(
        f"/kbs/kb_replace_idem/documents/{document_id}:replace"
        "?idempotency_key=replace-idem-1",
        files={"file": ("paper-v2.pdf", b"new", "application/pdf")},
        headers=_HEADERS,
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]

    conflict = client.post(
        f"/kbs/kb_replace_idem/documents/{document_id}:replace"
        "?idempotency_key=replace-idem-1",
        files={"file": ("paper-v3.pdf", b"different", "application/pdf")},
        headers=_HEADERS,
    )
    assert conflict.status_code == 409


def test_replace_auto_index_requires_auto_parse(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_replace_auto_index")
    upload = client.post(
        "/kbs/kb_replace_auto_index/documents:upload",
        files=[("files", ("paper.pdf", b"pdf", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]

    response = client.post(
        f"/kbs/kb_replace_auto_index/documents/{document_id}:replace"
        "?auto_index=true&auto_parse=false",
        files={"file": ("paper-v2.pdf", b"new", "application/pdf")},
        headers=_HEADERS,
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "auto_index requires auto_parse for document replacement"
    )


def test_active_replace_blocks_parse_claim(tmp_path):
    client, _kb_service, store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_replace_guard")
    upload = client.post(
        "/kbs/kb_replace_guard/documents:upload",
        files=[("files", ("guard.pdf", b"guard", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]
    asyncio.run(
        store.claim_document_replacing(
            "kb_replace_guard",
            document_id,
            metadata_patch={"pending_replace_job_id": "job_replace_guard"},
        )
    )

    response = client.post(
        f"/kbs/kb_replace_guard/documents/{document_id}:parse",
        json={"engine": "mineru", "process_options": "iF"},
        headers=_HEADERS,
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["error_code"] == "replace_job_active"
    assert detail["existing_job_id"] == "job_replace_guard"
    document = client.get(
        f"/kbs/kb_replace_guard/documents/{document_id}", headers=_HEADERS
    )
    assert document.status_code == 200
    assert document.json()["status"] == "replacing"


def test_replace_artifact_cleanup_rejects_workspace_escape(tmp_path):
    probe = BuilderProbe()
    client, _kb_service, store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_replace_escape")
    document_id, artifacts = _upload_and_parse_document(client, "kb_replace_escape")
    before = client.get(
        f"/kbs/kb_replace_escape/documents/{document_id}", headers=_HEADERS
    )
    assert before.status_code == 200
    old_lightrag_doc_id = before.json()["lightrag_doc_id"]
    escaped_path = tmp_path / "escaped-replace.txt"
    escaped_path.write_text("outside", encoding="utf-8")
    with store._connect() as conn:
        conn.execute(
            "UPDATE document_artifacts SET uri = ? WHERE id = ?",
            (str(escaped_path), artifacts["blocks"]["id"]),
        )
        conn.commit()

    response = client.post(
        f"/kbs/kb_replace_escape/documents/{document_id}:replace",
        files={"file": ("new.pdf", b"new", "application/pdf")},
        headers=_HEADERS,
    )

    assert response.status_code == 200
    job = client.get(
        f"/kbs/kb_replace_escape/jobs/{response.json()['id']}", headers=_HEADERS
    )
    assert job.status_code == 200
    assert job.json()["status"] == "failed"
    assert job.json()["error_code"] == "replace_failed"
    assert escaped_path.exists()
    assert probe.instances[-1].delete_calls == []
    document = client.get(
        f"/kbs/kb_replace_escape/documents/{document_id}", headers=_HEADERS
    )
    assert document.status_code == 200
    payload = document.json()
    assert payload["status"] == "replace_failed"
    assert payload["lightrag_doc_id"] == old_lightrag_doc_id


def test_replace_failure_after_old_index_delete_clears_index_metadata(
    tmp_path, monkeypatch
):
    probe = BuilderProbe()
    client, _kb_service, _store, document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_replace_partial_fail")
    document_id, _artifacts = _upload_and_parse_document(
        client, "kb_replace_partial_fail"
    )
    before = client.get(
        f"/kbs/kb_replace_partial_fail/documents/{document_id}", headers=_HEADERS
    )
    assert before.status_code == 200
    old_lightrag_doc_id = before.json()["lightrag_doc_id"]

    async def fail_after_lightrag_delete(*_args, **_kwargs):
        raise RuntimeError("file replacement exploded")

    monkeypatch.setattr(
        document_service,
        "replace_document_source",
        fail_after_lightrag_delete,
    )

    response = client.post(
        f"/kbs/kb_replace_partial_fail/documents/{document_id}:replace",
        files={"file": ("new.pdf", b"new", "application/pdf")},
        headers=_HEADERS,
    )

    assert response.status_code == 200
    job = client.get(
        f"/kbs/kb_replace_partial_fail/jobs/{response.json()['id']}",
        headers=_HEADERS,
    )
    assert job.status_code == 200
    assert job.json()["status"] == "failed"
    assert probe.instances[-1].delete_calls == [(old_lightrag_doc_id, False)]
    document = client.get(
        f"/kbs/kb_replace_partial_fail/documents/{document_id}", headers=_HEADERS
    )
    assert document.status_code == 200
    payload = document.json()
    assert payload["status"] == "replace_failed"
    assert payload["lightrag_doc_id"] is None
    assert payload["index_hash"] is None


def test_replace_background_start_failure_releases_replacing_claim(
    tmp_path, monkeypatch
):
    client, _kb_service, _store, _document_service, job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_replace_start_fail")
    upload = client.post(
        "/kbs/kb_replace_start_fail/documents:upload",
        files=[("files", ("paper.pdf", b"pdf", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]
    original_transition_job = job_service.transition_job

    async def fail_running_transition(kb_id, job_id, **kwargs):
        if kwargs.get("status") == "running":
            raise RuntimeError("transition exploded")
        return await original_transition_job(kb_id, job_id, **kwargs)

    monkeypatch.setattr(job_service, "transition_job", fail_running_transition)

    response = client.post(
        f"/kbs/kb_replace_start_fail/documents/{document_id}:replace",
        files={"file": ("paper-v2.pdf", b"new", "application/pdf")},
        headers=_HEADERS,
    )

    assert response.status_code == 200
    job = client.get(
        f"/kbs/kb_replace_start_fail/jobs/{response.json()['id']}",
        headers=_HEADERS,
    )
    assert job.status_code == 200
    assert job.json()["status"] == "failed"
    document = client.get(
        f"/kbs/kb_replace_start_fail/documents/{document_id}", headers=_HEADERS
    )
    assert document.status_code == 200
    assert document.json()["status"] == "replace_failed"


def test_parse_document_succeeds_and_persists_artifacts(tmp_path):
    probe = BuilderProbe()
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_parse")
    upload = client.post(
        "/kbs/kb_parse/documents:upload",
        files=[("files", ("paper.pdf", b"pdf", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]

    response = client.post(
        f"/kbs/kb_parse/documents/{document_id}:parse",
        json={"engine": "mineru", "process_options": "iF", "force_reparse": True},
        headers=_HEADERS,
    )

    assert response.status_code == 200
    job_id = response.json()["id"]
    assert response.json()["job_type"] == "parse"
    assert response.json()["document_id"] == document_id

    job = client.get(f"/kbs/kb_parse/jobs/{job_id}", headers=_HEADERS)
    assert job.status_code == 200
    assert job.json()["status"] == "succeeded"
    assert job.json()["completed_items"] == 1
    assert job.json()["result"]["artifact_count"] >= 3

    document = client.get(f"/kbs/kb_parse/documents/{document_id}", headers=_HEADERS)
    assert document.status_code == 200
    document_payload = document.json()
    assert document_payload["status"] == "parsed"
    assert document_payload["parser_hash"].startswith("sha256:")
    assert document_payload["lightrag_doc_id"].startswith("doc-")
    assert Path(document_payload["source_uri"]).exists()

    assert probe.instances
    _engine, _doc_id, _file_path, content_data = probe.instances[0].parse_calls[0]
    assert content_data["force_reparse"] is True
    assert content_data["archive_source_after_parse"] is False

    artifacts = client.get(
        f"/kbs/kb_parse/documents/{document_id}/artifacts", headers=_HEADERS
    )
    assert artifacts.status_code == 200
    artifact_payload = artifacts.json()
    artifact_types = {item["artifact_type"] for item in artifact_payload["artifacts"]}
    assert {"original", "sidecar", "blocks", "raw_dir"}.issubset(artifact_types)
    original = next(
        item
        for item in artifact_payload["artifacts"]
        if item["artifact_type"] == "original"
    )
    assert original["checksum"].startswith("sha256:")
    assert original["size_bytes"] == 3

    artifact_id = artifact_payload["artifacts"][0]["id"]
    detail = client.get(
        f"/kbs/kb_parse/documents/{document_id}/artifacts/{artifact_id}",
        headers=_HEADERS,
    )
    assert detail.status_code == 200
    assert detail.json()["id"] == artifact_id


def test_parse_document_idempotency_key_reuses_existing_job(tmp_path):
    probe = BuilderProbe()
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_parse_idem")
    upload = client.post(
        "/kbs/kb_parse_idem/documents:upload",
        files=[("files", ("paper.pdf", b"pdf", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]

    first = client.post(
        f"/kbs/kb_parse_idem/documents/{document_id}:parse",
        json={"engine": "mineru", "idempotency_key": "idem-parse-1"},
        headers=_HEADERS,
    )
    second = client.post(
        f"/kbs/kb_parse_idem/documents/{document_id}:parse",
        json={"engine": "mineru", "idempotency_key": "idem-parse-1"},
        headers=_HEADERS,
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert len(probe.instances) == 1
    assert len(probe.instances[0].parse_calls) == 1

    conflict = client.post(
        f"/kbs/kb_parse_idem/documents/{document_id}:parse",
        json={
            "engine": "mineru",
            "force_reparse": True,
            "idempotency_key": "idem-parse-1",
        },
        headers=_HEADERS,
    )
    assert conflict.status_code == 409


def test_batch_parse_documents_succeeds_and_persists_artifacts(tmp_path):
    probe = BuilderProbe()
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_batch_parse")
    upload = client.post(
        "/kbs/kb_batch_parse/documents:upload",
        files=[
            ("files", ("alpha.pdf", b"alpha", "application/pdf")),
            ("files", ("beta.pdf", b"beta", "application/pdf")),
        ],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_ids = [document["id"] for document in upload.json()["documents"]]

    response = client.post(
        "/kbs/kb_batch_parse/documents:batch-parse",
        json={
            "document_ids": document_ids,
            "engine": "mineru",
            "process_options": "iF",
            "force_reparse": True,
        },
        headers=_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"].startswith("job_parse_")
    assert payload["batch_id"].startswith("batch_")
    assert payload["documents"]
    assert {document["status"] for document in payload["documents"]} == {"parse_queued"}

    job = client.get(f"/kbs/kb_batch_parse/jobs/{payload['job_id']}", headers=_HEADERS)
    assert job.status_code == 200
    job_payload = job.json()
    assert job_payload["status"] == "succeeded"
    assert job_payload["job_type"] == "parse"
    assert job_payload["document_id"] is None
    assert job_payload["batch_id"] == payload["batch_id"]
    assert job_payload["total_items"] == 2
    assert job_payload["completed_items"] == 2
    assert job_payload["failed_items"] == 0
    assert job_payload["result"]["summary"]["outcome"] == "succeeded"
    assert {item["status"] for item in job_payload["result"]["items"]} == {"succeeded"}

    for document_id in document_ids:
        document = client.get(
            f"/kbs/kb_batch_parse/documents/{document_id}", headers=_HEADERS
        )
        assert document.status_code == 200
        assert document.json()["status"] == "parsed"
        artifacts = client.get(
            f"/kbs/kb_batch_parse/documents/{document_id}/artifacts",
            headers=_HEADERS,
        )
        assert artifacts.status_code == 200
        assert artifacts.json()["total"] >= 3

    assert len(probe.instances) == 1
    assert len(probe.instances[0].parse_calls) == 2


def test_batch_parse_idempotency_key_reuses_existing_job(tmp_path):
    probe = BuilderProbe()
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_batch_idem")
    upload = client.post(
        "/kbs/kb_batch_idem/documents:upload",
        files=[
            ("files", ("alpha.pdf", b"alpha", "application/pdf")),
            ("files", ("beta.pdf", b"beta", "application/pdf")),
        ],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_ids = [document["id"] for document in upload.json()["documents"]]
    request = {
        "document_ids": document_ids,
        "engine": "mineru",
        "idempotency_key": "idem-batch-parse-1",
    }

    first = client.post(
        "/kbs/kb_batch_idem/documents:batch-parse", json=request, headers=_HEADERS
    )
    second = client.post(
        "/kbs/kb_batch_idem/documents:batch-parse", json=request, headers=_HEADERS
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["job_id"] == first.json()["job_id"]
    assert second.json()["batch_id"] == first.json()["batch_id"]
    assert [document["id"] for document in second.json()["documents"]] == document_ids
    assert len(probe.instances) == 1
    assert len(probe.instances[0].parse_calls) == 2

    conflict = client.post(
        "/kbs/kb_batch_idem/documents:batch-parse",
        json={**request, "force_reparse": True},
        headers=_HEADERS,
    )
    assert conflict.status_code == 409


def test_batch_parse_partial_failure_marks_job_failed_and_continues(tmp_path):
    probe = BuilderProbe(fail_source_names={"bad.pdf"})
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_batch_partial")
    upload = client.post(
        "/kbs/kb_batch_partial/documents:upload",
        files=[
            ("files", ("good.pdf", b"good", "application/pdf")),
            ("files", ("bad.pdf", b"bad", "application/pdf")),
        ],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    documents = upload.json()["documents"]
    document_ids = [document["id"] for document in documents]
    names_by_id = {document["id"]: document["source_name"] for document in documents}

    response = client.post(
        "/kbs/kb_batch_partial/documents:batch-parse",
        json={"document_ids": document_ids, "engine": "mineru"},
        headers=_HEADERS,
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    job = client.get(f"/kbs/kb_batch_partial/jobs/{job_id}", headers=_HEADERS)
    assert job.status_code == 200
    job_payload = job.json()
    assert job_payload["status"] == "failed"
    assert job_payload["completed_items"] == 1
    assert job_payload["failed_items"] == 1
    assert job_payload["error_code"] == "partial_parse_failed"
    assert job_payload["result"]["summary"]["outcome"] == "partial_failure"
    assert {item["status"] for item in job_payload["result"]["items"]} == {
        "succeeded",
        "failed",
    }

    for document_id in document_ids:
        document = client.get(
            f"/kbs/kb_batch_partial/documents/{document_id}", headers=_HEADERS
        )
        assert document.status_code == 200
        expected_status = (
            "parse_failed" if names_by_id[document_id] == "bad.pdf" else "parsed"
        )
        assert document.json()["status"] == expected_status


def test_batch_parse_treats_active_parse_as_per_item_failure(tmp_path):
    probe = BuilderProbe()
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_batch_active")
    active_upload = client.post(
        "/kbs/kb_batch_active/documents:upload?auto_parse=true",
        files=[("files", ("active.pdf", b"active", "application/pdf"))],
        headers=_HEADERS,
    )
    assert active_upload.status_code == 200
    active_payload = active_upload.json()
    active_job_id = active_payload["job_id"]
    active_document_id = active_payload["documents"][0]["id"]
    valid_upload = client.post(
        "/kbs/kb_batch_active/documents:upload",
        files=[("files", ("valid.pdf", b"valid", "application/pdf"))],
        headers=_HEADERS,
    )
    assert valid_upload.status_code == 200
    valid_document_id = valid_upload.json()["documents"][0]["id"]

    response = client.post(
        "/kbs/kb_batch_active/documents:batch-parse",
        json={
            "document_ids": [active_document_id, valid_document_id],
            "engine": "mineru",
        },
        headers=_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert [document["id"] for document in payload["documents"]] == [valid_document_id]
    job = client.get(f"/kbs/kb_batch_active/jobs/{payload['job_id']}", headers=_HEADERS)
    assert job.status_code == 200
    job_payload = job.json()
    assert job_payload["status"] == "failed"
    assert job_payload["completed_items"] == 1
    assert job_payload["failed_items"] == 1
    assert job_payload["result"]["summary"]["outcome"] == "partial_failure"
    failure = next(
        item for item in job_payload["result"]["items"] if item["status"] == "failed"
    )
    assert failure["document_id"] == active_document_id
    assert failure["error_code"] == "parse_job_active"
    assert failure["existing_job_id"] == active_job_id

    active_document = client.get(
        f"/kbs/kb_batch_active/documents/{active_document_id}", headers=_HEADERS
    )
    assert active_document.status_code == 200
    assert active_document.json()["status"] == "parse_queued"
    assert active_document.json()["metadata"]["pending_parse_job_id"] == active_job_id
    valid_document = client.get(
        f"/kbs/kb_batch_active/documents/{valid_document_id}", headers=_HEADERS
    )
    assert valid_document.status_code == 200
    assert valid_document.json()["status"] == "parsed"
    assert len(probe.instances) == 1
    assert len(probe.instances[0].parse_calls) == 1


def test_batch_parse_missing_document_and_source_are_per_item_failures(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_batch_missing")
    upload = client.post(
        "/kbs/kb_batch_missing/documents:upload",
        files=[
            ("files", ("ok.pdf", b"ok", "application/pdf")),
            ("files", ("lost.pdf", b"lost", "application/pdf")),
        ],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    documents = upload.json()["documents"]
    ok_id = documents[0]["id"]
    lost = documents[1]
    Path(lost["source_uri"]).unlink()

    response = client.post(
        "/kbs/kb_batch_missing/documents:batch-parse",
        json={"document_ids": [ok_id, lost["id"], "doc_missing"], "engine": "mineru"},
        headers=_HEADERS,
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    job = client.get(f"/kbs/kb_batch_missing/jobs/{job_id}", headers=_HEADERS)
    assert job.status_code == 200
    job_payload = job.json()
    assert job_payload["status"] == "failed"
    assert job_payload["completed_items"] == 1
    assert job_payload["failed_items"] == 2
    result_items = job_payload["result"]["items"]
    assert {
        item["error_code"] for item in result_items if item["status"] == "failed"
    } == {
        "source_not_found",
        "document_not_found",
    }

    ok_document = client.get(
        f"/kbs/kb_batch_missing/documents/{ok_id}", headers=_HEADERS
    )
    assert ok_document.status_code == 200
    assert ok_document.json()["status"] == "parsed"
    lost_document = client.get(
        f"/kbs/kb_batch_missing/documents/{lost['id']}", headers=_HEADERS
    )
    assert lost_document.status_code == 200
    assert lost_document.json()["status"] == "uploaded"


def test_batch_parse_rejects_invalid_options_duplicates_and_cross_kb(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_batch_invalid")
    _create_kb(client, "kb_batch_other")
    upload = client.post(
        "/kbs/kb_batch_invalid/documents:upload",
        files=[("files", ("paper.pdf", b"pdf", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]

    invalid_options = client.post(
        "/kbs/kb_batch_invalid/documents:batch-parse",
        json={
            "document_ids": [document_id],
            "engine": "mineru",
            "process_options": "iZ",
        },
        headers=_HEADERS,
    )
    assert invalid_options.status_code == 400
    assert "unsupported character" in invalid_options.json()["detail"]

    duplicates = client.post(
        "/kbs/kb_batch_invalid/documents:batch-parse",
        json={"document_ids": [document_id, document_id], "engine": "mineru"},
        headers=_HEADERS,
    )
    assert duplicates.status_code == 422

    cross_kb = client.post(
        "/kbs/kb_batch_other/documents:batch-parse",
        json={"document_ids": [document_id], "engine": "mineru"},
        headers=_HEADERS,
    )
    assert cross_kb.status_code == 200
    job = client.get(
        f"/kbs/kb_batch_other/jobs/{cross_kb.json()['job_id']}", headers=_HEADERS
    )
    assert job.status_code == 200
    assert job.json()["status"] == "failed"
    assert job.json()["failed_items"] == 1
    assert job.json()["result"]["items"][0]["error_code"] == "document_not_found"


def test_download_document_file_artifacts_returns_bytes(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_artifact_download")
    document_id, artifacts = _upload_and_parse_document(
        client,
        "kb_artifact_download",
        filename="paper.pdf",
        content=b"pdf-body",
    )

    original = artifacts["original"]
    original_response = client.get(
        f"/kbs/kb_artifact_download/documents/{document_id}/artifacts/{original['id']}:download",
        headers=_HEADERS,
    )
    assert original_response.status_code == 200
    assert original_response.content == b"pdf-body"
    assert original_response.headers["content-type"].startswith("application/pdf")
    assert "paper.pdf" in original_response.headers["content-disposition"]

    blocks = artifacts["blocks"]
    blocks_response = client.get(
        f"/kbs/kb_artifact_download/documents/{document_id}/artifacts/{blocks['id']}:download",
        headers=_HEADERS,
    )
    assert blocks_response.status_code == 200
    assert blocks_response.content.replace(b"\r\n", b"\n") == (
        b'{"type":"content","text":"parsed"}\n'
    )
    assert blocks_response.headers["content-type"].startswith("application/x-ndjson")


def test_download_document_artifact_streams_directory_as_zip(tmp_path):
    import io
    import zipfile

    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_artifact_zip")
    document_id, artifacts = _upload_and_parse_document(client, "kb_artifact_zip")

    sidecar = artifacts["sidecar"]
    response = client.get(
        f"/kbs/kb_artifact_zip/documents/{document_id}/artifacts/{sidecar['id']}:download",
        headers=_HEADERS,
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    names = archive.namelist()
    assert any(name.endswith(".blocks.jsonl") for name in names)


def test_download_document_artifact_rejects_directories_and_missing_files(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_artifact_errors")
    document_id, artifacts = _upload_and_parse_document(client, "kb_artifact_errors")

    missing_response = client.get(
        f"/kbs/kb_artifact_errors/documents/{document_id}/artifacts/artifact_missing:download",
        headers=_HEADERS,
    )
    assert missing_response.status_code == 404

    blocks = artifacts["blocks"]
    Path(blocks["uri"]).unlink()
    missing_file_response = client.get(
        f"/kbs/kb_artifact_errors/documents/{document_id}/artifacts/{blocks['id']}:download",
        headers=_HEADERS,
    )
    assert missing_file_response.status_code == 404
    assert "Artifact file not found" in missing_file_response.json()["detail"]


def test_download_document_artifact_rejects_cross_kb_and_path_escape(tmp_path):
    client, _kb_service, store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_artifact_owner")
    _create_kb(client, "kb_artifact_other")
    document_id, artifacts = _upload_and_parse_document(client, "kb_artifact_owner")
    original = artifacts["original"]

    cross_kb_response = client.get(
        f"/kbs/kb_artifact_other/documents/{document_id}/artifacts/{original['id']}:download",
        headers=_HEADERS,
    )
    assert cross_kb_response.status_code == 404

    escaped_path = tmp_path / "escaped.txt"
    escaped_path.write_text("escaped", encoding="utf-8")
    with store._connect() as conn:
        conn.execute(
            "UPDATE document_artifacts SET uri = ? WHERE id = ?",
            (str(escaped_path), original["id"]),
        )
        conn.commit()

    escape_response = client.get(
        f"/kbs/kb_artifact_owner/documents/{document_id}/artifacts/{original['id']}:download",
        headers=_HEADERS,
    )
    assert escape_response.status_code == 400
    assert "escapes document directory" in escape_response.json()["detail"]


def test_parse_document_failure_marks_job_and_document_failed(tmp_path):
    probe = BuilderProbe(should_fail=True)
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_parse_fail")
    upload = client.post(
        "/kbs/kb_parse_fail/documents:upload",
        files=[("files", ("paper.pdf", b"pdf", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]

    response = client.post(
        f"/kbs/kb_parse_fail/documents/{document_id}:parse",
        json={"engine": "mineru"},
        headers=_HEADERS,
    )

    assert response.status_code == 200
    job_id = response.json()["id"]
    job = client.get(f"/kbs/kb_parse_fail/jobs/{job_id}", headers=_HEADERS)
    assert job.status_code == 200
    assert job.json()["status"] == "failed"
    assert job.json()["failed_items"] == 1
    assert job.json()["error_code"] == "parse_failed"

    document = client.get(
        f"/kbs/kb_parse_fail/documents/{document_id}", headers=_HEADERS
    )
    assert document.status_code == 200
    assert document.json()["status"] == "parse_failed"
    assert document.json()["error_code"] == "parse_failed"


def test_parse_document_missing_source_returns_404(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_missing_source")
    upload = client.post(
        "/kbs/kb_missing_source/documents:upload",
        files=[("files", ("paper.pdf", b"pdf", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document = upload.json()["documents"][0]
    Path(document["source_uri"]).unlink()

    response = client.post(
        f"/kbs/kb_missing_source/documents/{document['id']}:parse",
        json={"engine": "mineru"},
        headers=_HEADERS,
    )

    assert response.status_code == 404
    assert "Document source not found" in response.json()["detail"]


def test_parse_document_rejects_invalid_process_options(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_bad_options")
    upload = client.post(
        "/kbs/kb_bad_options/documents:upload",
        files=[("files", ("paper.pdf", b"pdf", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]

    response = client.post(
        f"/kbs/kb_bad_options/documents/{document_id}:parse",
        json={"engine": "mineru", "process_options": "iZ"},
        headers=_HEADERS,
    )

    assert response.status_code == 400
    assert "unsupported character" in response.json()["detail"]


def test_parse_document_rejects_existing_active_parse_job(tmp_path):
    probe = BuilderProbe()
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_parse_active")
    upload = client.post(
        "/kbs/kb_parse_active/documents:upload?auto_parse=true",
        files=[("files", ("paper.pdf", b"pdf", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    active_job_id = upload.json()["job_id"]
    document_id = upload.json()["documents"][0]["id"]

    response = client.post(
        f"/kbs/kb_parse_active/documents/{document_id}:parse",
        json={"engine": "mineru"},
        headers=_HEADERS,
    )

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "error_code": "parse_job_active",
        "document_id": document_id,
        "existing_job_id": active_job_id,
        "message": f"Document '{document_id}' already has an active parse job",
    }
    document = client.get(
        f"/kbs/kb_parse_active/documents/{document_id}", headers=_HEADERS
    )
    assert document.status_code == 200
    assert document.json()["status"] == "parse_queued"
    assert document.json()["metadata"]["pending_parse_job_id"] == active_job_id
    failed_jobs = client.get(
        f"/kbs/kb_parse_active/jobs?status=failed&document_id={document_id}",
        headers=_HEADERS,
    )
    assert failed_jobs.status_code == 200
    assert failed_jobs.json()["total"] == 1
    assert failed_jobs.json()["jobs"][0]["error_code"] == "parse_job_active"
    assert probe.instances == []


def test_same_name_uploads_use_distinct_exclusive_source_paths(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_same_name")

    response = client.post(
        "/kbs/kb_same_name/documents:upload",
        files=[
            ("files", ("same.txt", b"first", "text/plain")),
            ("files", ("same.txt", b"second", "text/plain")),
        ],
        headers=_HEADERS,
    )

    assert response.status_code == 200
    documents = response.json()["documents"]
    paths = [Path(document["source_uri"]) for document in documents]
    assert len({str(path) for path in paths}) == 2
    assert all(path.exists() for path in paths)
    assert {path.parent.name for path in paths} == {
        document["id"] for document in documents
    }
    assert {path.read_bytes() for path in paths} == {b"first", b"second"}


def test_upload_rejects_unsupported_file_type(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_ext")

    response = client.post(
        "/kbs/kb_ext/documents:upload",
        files=[("files", ("malware.exe", b"nope", "application/octet-stream"))],
        headers=_HEADERS,
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_sanitizes_unsafe_filename_inside_document_directory(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_safe_name")

    response = client.post(
        "/kbs/kb_safe_name/documents:upload",
        files=[("files", ("../unsafe?.txt", b"safe", "text/plain"))],
        headers=_HEADERS,
    )

    assert response.status_code == 200
    document = response.json()["documents"][0]
    source_path = Path(document["source_uri"])
    # ".." is dropped (left-over leading slash from "/" then sanitized to "_"),
    # the unsafe "?" character becomes "_". CJK / spaces / dashes survive.
    assert document["source_name"] == "_unsafe_.txt"
    assert source_path.parent.name == document["id"]
    assert source_path.parent.parent == tmp_path / "inputs" / sanitize_workspace(
        "kb_safe_name"
    )
    assert source_path.read_bytes() == b"safe"


def test_upload_rejects_oversized_file(tmp_path, monkeypatch):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_big")
    monkeypatch.setattr(_kb_document_routes.global_args, "max_upload_size", 3)

    response = client.post(
        "/kbs/kb_big/documents:upload",
        files=[("files", ("big.txt", b"1234", "text/plain"))],
        headers=_HEADERS,
    )

    assert response.status_code == 413
    assert "File too large" in response.json()["detail"]
    assert not any((tmp_path / "inputs" / sanitize_workspace("kb_big")).glob("**/*"))


def test_upload_rejects_aggregate_oversized_batch(tmp_path, monkeypatch):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_big_batch")
    monkeypatch.setattr(_kb_document_routes.global_args, "max_upload_size", 6)

    response = client.post(
        "/kbs/kb_big_batch/documents:upload",
        files=[
            ("files", ("one.txt", b"1234", "text/plain")),
            ("files", ("two.txt", b"5678", "text/plain")),
        ],
        headers=_HEADERS,
    )

    assert response.status_code == 413
    assert "Upload batch too large" in response.json()["detail"]
    assert not any(
        (tmp_path / "inputs" / sanitize_workspace("kb_big_batch")).glob("**/*")
    )


def test_upload_rejects_unlimited_max_upload_size(tmp_path, monkeypatch):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_unlimited")
    monkeypatch.setattr(_kb_document_routes.global_args, "max_upload_size", 0)

    response = client.post(
        "/kbs/kb_unlimited/documents:upload",
        files=[("files", ("tiny.txt", b"tiny", "text/plain"))],
        headers=_HEADERS,
    )

    assert response.status_code == 413
    assert "MAX_UPLOAD_SIZE" in response.json()["detail"]


def test_upload_rejects_too_many_files(tmp_path, monkeypatch):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_many")
    monkeypatch.setattr(_kb_document_routes, "_MAX_KB_UPLOAD_FILES", 1)

    response = client.post(
        "/kbs/kb_many/documents:upload",
        files=[
            ("files", ("one.txt", b"one", "text/plain")),
            ("files", ("two.txt", b"two", "text/plain")),
        ],
        headers=_HEADERS,
    )

    assert response.status_code == 413
    assert "Too many files" in response.json()["detail"]


def test_text_import_rejects_oversized_text(tmp_path, monkeypatch):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_big_text")
    monkeypatch.setattr(_kb_document_routes, "_MAX_TEXT_DOCUMENT_BYTES", 4)

    response = client.post(
        "/kbs/kb_big_text/documents:texts",
        json={"documents": [{"text": "12345", "source_name": "big.txt"}]},
        headers=_HEADERS,
    )

    assert response.status_code == 413
    assert "Text document too large" in response.json()["detail"]


@pytest.mark.asyncio
async def test_job_transition_rules(tmp_path):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "knowledge_bases.json")
    store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    job_service = JobService(kb_service, store)
    await kb_service.create(kb_id="kb_jobs", name="Jobs")

    job = await job_service.create_job("kb_jobs", job_type="parse", stage="parsing")
    running = await job_service.transition_job(
        "kb_jobs", job.id, status="running", progress=0.5
    )
    assert running.status == "running"
    assert running.started_at is not None
    assert running.progress == 0.5

    succeeded = await job_service.transition_job(
        "kb_jobs", job.id, status="succeeded", progress=1.0, result={"ok": True}
    )
    assert succeeded.status == "succeeded"
    assert succeeded.finished_at is not None
    assert succeeded.result == {"ok": True}

    with pytest.raises(InvalidJobTransitionError):
        await job_service.transition_job("kb_jobs", job.id, status="running")


def test_missing_kb_document_routes_return_404(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )

    response = client.get("/kbs/missing/documents", headers=_HEADERS)
    assert response.status_code == 404

    upload = client.post(
        "/kbs/missing/documents:upload",
        files=[("files", ("missing.txt", b"missing", "text/plain"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 404
