from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightrag.api.document_lifecycle_service import (
    DocumentLifecycleService,
    DocumentSourceInput,
)
from lightrag.api.job_service import JobService
from lightrag.api.kb_service import KnowledgeBaseService
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry, LightRAGLike
from lightrag.api.metadata_store import SQLiteMetadataStore

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_kb_routes = importlib.import_module("lightrag.api.routers.kb_routes")
_kb_query_routes = importlib.import_module("lightrag.api.routers.kb_query_routes")
_kb_document_routes = importlib.import_module(
    "lightrag.api.routers.kb_document_routes"
)
sys.argv = _original_argv

create_kb_routes = _kb_routes.create_kb_routes
create_kb_query_routes = _kb_query_routes.create_kb_query_routes
create_kb_document_routes = _kb_document_routes.create_kb_document_routes

pytestmark = pytest.mark.offline

_API_KEY = "test-key"
_HEADERS = {"X-API-Key": _API_KEY}


class FakeRAG:
    """Minimal LightRAG stand-in that records query calls per workspace.

    The fake builds responses that include the workspace name so a test
    can assert KB A never sees KB B content.
    """

    def __init__(self, workspace: str):
        self.workspace = workspace
        self.queries: list[tuple[str, str]] = []  # (query, mode)

    async def finalize_storages(self) -> None:
        return None

    async def aquery_llm(self, query: str, *, param):
        self.queries.append((query, param.mode))
        return {
            "llm_response": {
                "content": f"answer-from-{self.workspace}: {query}",
                "is_streaming": False,
            },
            "data": {
                "references": [
                    {
                        "reference_id": "1",
                        "file_path": f"{self.workspace}/source.pdf",
                    }
                ],
                "chunks": [
                    {
                        "reference_id": "1",
                        "content": f"chunk content from {self.workspace}",
                    }
                ],
            },
        }

    async def aquery_data(self, query: str, *, param):
        self.queries.append((query, param.mode))
        return {
            "status": "success",
            "message": "ok",
            "data": {
                "entities": [
                    {"entity_name": f"{self.workspace}-entity", "reference_id": "1"}
                ],
                "relationships": [],
                "chunks": [
                    {
                        "reference_id": "1",
                        "content": f"chunk content from {self.workspace}",
                    }
                ],
                "references": [
                    {
                        "reference_id": "1",
                        "file_path": f"{self.workspace}/source.pdf",
                    }
                ],
            },
            "metadata": {"query_mode": param.mode},
        }


class StreamingFakeRAG(FakeRAG):
    async def aquery_llm(self, query: str, *, param):
        self.queries.append((query, param.mode))

        async def chunks():
            yield "first "
            yield "second "
            yield "third"

        return {
            "llm_response": {
                "content": "",
                "is_streaming": True,
                "response_iterator": chunks(),
            },
            "data": {
                "references": [
                    {
                        "reference_id": "1",
                        "file_path": f"{self.workspace}/source.pdf",
                    }
                ],
                "chunks": [],
            },
        }


class BuilderProbe:
    def __init__(self, *, streaming: bool = False):
        self.streaming = streaming
        self.instances: dict[str, FakeRAG] = {}

    async def build(self, record) -> FakeRAG:
        cls = StreamingFakeRAG if self.streaming else FakeRAG
        rag = cls(record.workspace)
        self.instances[record.id] = rag
        return rag

    async def finalize(self, rag: LightRAGLike) -> None:
        return None


def _build_client(tmp_path: Path, *, streaming: bool = False):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kb.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    document_service = DocumentLifecycleService(
        kb_service, metadata_store, tmp_path / "inputs"
    )
    job_service = JobService(kb_service, metadata_store)
    probe = BuilderProbe(streaming=streaming)
    registry = LightRAGInstanceRegistry(kb_service, probe.build, probe.finalize)
    app = FastAPI()
    app.include_router(
        create_kb_routes(kb_service, registry, api_key=_API_KEY, job_service=job_service)
    )
    app.include_router(
        create_kb_document_routes(
            document_service, job_service, api_key=_API_KEY, registry=registry
        )
    )
    app.include_router(
        create_kb_query_routes(document_service, registry, api_key=_API_KEY)
    )
    return TestClient(app), kb_service, document_service, registry, probe


def _create_kb(client: TestClient, kb_id: str):
    response = client.post("/kbs", json={"id": kb_id, "name": kb_id}, headers=_HEADERS)
    assert response.status_code == 200
    return response.json()


def test_kb_query_returns_workspace_specific_answer(tmp_path):
    client, _kb_service, *_, probe = _build_client(tmp_path)
    alpha = _create_kb(client, "kb_alpha")
    beta = _create_kb(client, "kb_beta")
    alpha_workspace = alpha["workspace"]
    beta_workspace = beta["workspace"]

    response_alpha = client.post(
        "/kbs/kb_alpha/query",
        json={"query": "what is alpha?", "mode": "mix"},
        headers=_HEADERS,
    )
    assert response_alpha.status_code == 200, response_alpha.text
    body_alpha = response_alpha.json()
    assert body_alpha["kb_id"] == "kb_alpha"
    assert body_alpha["mode"] == "mix"
    assert f"answer-from-{alpha_workspace}" in body_alpha["response"]
    assert body_alpha["references"][0]["file_path"].startswith(alpha_workspace + "/")

    response_beta = client.post(
        "/kbs/kb_beta/query",
        json={"query": "what is beta?", "mode": "mix"},
        headers=_HEADERS,
    )
    assert response_beta.status_code == 200
    body_beta = response_beta.json()
    assert f"answer-from-{beta_workspace}" in body_beta["response"]
    assert body_beta["references"][0]["file_path"].startswith(beta_workspace + "/")

    # Each KB instance only saw its own query
    alpha_rag = probe.instances["kb_alpha"]
    beta_rag = probe.instances["kb_beta"]
    assert alpha_rag.queries == [("what is alpha?", "mix")]
    assert beta_rag.queries == [("what is beta?", "mix")]


def test_kb_query_can_disable_references(tmp_path):
    client, *_ = _build_client(tmp_path)
    _create_kb(client, "kb_no_refs")

    response = client.post(
        "/kbs/kb_no_refs/query",
        json={"query": "ignore references", "include_references": False},
        headers=_HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["references"] is None


def test_kb_query_includes_chunk_content_when_requested(tmp_path):
    client, *_ = _build_client(tmp_path)
    kb = _create_kb(client, "kb_chunks")

    response = client.post(
        "/kbs/kb_chunks/query",
        json={
            "query": "include chunks",
            "include_references": True,
            "include_chunk_content": True,
        },
        headers=_HEADERS,
    )
    assert response.status_code == 200
    refs = response.json()["references"]
    assert refs[0]["content"] == [f"chunk content from {kb['workspace']}"]


def test_kb_query_data_returns_structured_payload(tmp_path):
    client, *_ = _build_client(tmp_path)
    kb = _create_kb(client, "kb_data")

    response = client.post(
        "/kbs/kb_data/query/data",
        json={"query": "structured retrieval", "mode": "local"},
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kb_id"] == "kb_data"
    assert body["status"] == "success"
    assert body["data"]["entities"][0]["entity_name"] == f"{kb['workspace']}-entity"
    assert body["metadata"]["query_mode"] == "local"


def test_kb_retrieve_aliases_query_data(tmp_path):
    client, *_ = _build_client(tmp_path)
    kb = _create_kb(client, "kb_retrieve")

    response = client.post(
        "/kbs/kb_retrieve/retrieve",
        json={"query": "alias check", "mode": "naive"},
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    assert (
        response.json()["data"]["entities"][0]["entity_name"]
        == f"{kb['workspace']}-entity"
    )


def test_kb_query_stream_returns_ndjson(tmp_path):
    client, *_ = _build_client(tmp_path, streaming=True)
    kb = _create_kb(client, "kb_stream")

    with client.stream(
        "POST",
        "/kbs/kb_stream/query/stream",
        json={"query": "streaming please", "stream": True},
        headers=_HEADERS,
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        body = b"".join(response.iter_bytes()).decode("utf-8")
    lines = [line for line in body.split("\n") if line]
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["kb_id"] == "kb_stream"
    assert parsed[0]["references"][0]["file_path"].startswith(kb["workspace"] + "/")
    chunks = [item["response"] for item in parsed[1:]]
    assert chunks == ["first ", "second ", "third"]


def test_kb_query_filters_doc_ids_must_belong_to_kb(tmp_path):
    client, _kb_service, document_service, _registry, _probe = _build_client(tmp_path)
    _create_kb(client, "kb_owner")
    _create_kb(client, "kb_other")

    # Seed a document into kb_owner via the lifecycle service
    async def seed_document() -> str:
        result = await document_service.create_source_batch(
            "kb_owner",
            [
                DocumentSourceInput(
                    source_name="doc.txt",
                    content=b"hello",
                    source_type="upload",
                    content_type="text/plain",
                    metadata={},
                )
            ],
            auto_parse=False,
            auto_index=False,
        )
        return result.documents[0].id

    document_id = asyncio.run(seed_document())

    # Filter referencing a doc that exists inside kb_owner is OK
    response_ok = client.post(
        "/kbs/kb_owner/query",
        json={
            "query": "owner search",
            "filters": {"doc_ids": [document_id]},
        },
        headers=_HEADERS,
    )
    assert response_ok.status_code == 200, response_ok.text

    # Filter referencing the same doc against a different KB is rejected
    response_rejected = client.post(
        "/kbs/kb_other/query",
        json={
            "query": "should fail",
            "filters": {"doc_ids": [document_id]},
        },
        headers=_HEADERS,
    )
    assert response_rejected.status_code == 400
    detail = response_rejected.json()["detail"]
    assert detail["error_code"] == "doc_ids_not_in_kb"
    assert document_id in detail["missing"]


def test_kb_query_rejects_documents_in_active_replace(tmp_path):
    client, _kb_service, document_service, _registry, probe = _build_client(tmp_path)
    _create_kb(client, "kb_query_replace")

    async def seed_replacing_document() -> tuple[str, str]:
        result = await document_service.create_source_batch(
            "kb_query_replace",
            [
                DocumentSourceInput(
                    source_name="doc.txt",
                    content=b"hello",
                    source_type="upload",
                    content_type="text/plain",
                    metadata={},
                )
            ],
            auto_parse=False,
            auto_index=False,
        )
        document = result.documents[0]
        replacement = document_service.prepare_replacement_source(
            DocumentSourceInput(
                source_name="doc-v2.txt",
                content=b"replacement",
                source_type="upload",
                content_type="text/plain",
                metadata={},
            )
        )
        claimed = await document_service.claim_replace(
            "kb_query_replace",
            document.id,
            job=result.job,
            replacement=replacement,
        )
        return claimed.id, result.job.id

    document_id, job_id = asyncio.run(seed_replacing_document())

    filtered = client.post(
        "/kbs/kb_query_replace/query",
        json={"query": "active document", "filters": {"doc_ids": [document_id]}},
        headers=_HEADERS,
    )
    assert filtered.status_code == 409
    detail = filtered.json()["detail"]
    assert detail["error_code"] == "replace_job_active"
    assert detail["document_id"] == document_id
    assert detail["existing_job_id"] == job_id

    kb_wide = client.post(
        "/kbs/kb_query_replace/query/data",
        json={"query": "active document"},
        headers=_HEADERS,
    )
    assert kb_wide.status_code == 409
    assert kb_wide.json()["detail"]["error_code"] == "replace_job_active"
    assert "kb_query_replace" not in probe.instances


def test_kb_query_404_when_kb_missing(tmp_path):
    client, *_ = _build_client(tmp_path)
    response = client.post(
        "/kbs/kb_unknown/query",
        json={"query": "where am I?"},
        headers=_HEADERS,
    )
    assert response.status_code == 404


def test_kb_query_short_query_rejected(tmp_path):
    client, *_ = _build_client(tmp_path)
    _create_kb(client, "kb_short")
    response = client.post(
        "/kbs/kb_short/query",
        json={"query": "ab"},
        headers=_HEADERS,
    )
    assert response.status_code == 422
