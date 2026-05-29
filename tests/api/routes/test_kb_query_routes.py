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
from lightrag.base import QueryParam

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
        self.query_params: list[QueryParam] = []
        self.kb_active_query_config: dict[str, object] = {}
        self.kb_active_config_version_id: str | None = None
        self.kb_active_parser_hash: str | None = None
        self.kb_active_index_hash: str | None = None
        self.kb_active_query_hash: str | None = None

    async def finalize_storages(self) -> None:
        return None

    async def aquery_llm(self, query: str, *, param):
        self.queries.append((query, param.mode))
        self.query_params.append(param)
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
        self.query_params.append(param)
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
        self.active_query_config: dict[str, object] = {}
        self.active_config_version_id: str | None = None
        self.active_parser_hash: str | None = None
        self.active_index_hash: str | None = None
        self.active_query_hash: str | None = None

    async def build(self, record) -> FakeRAG:
        cls = StreamingFakeRAG if self.streaming else FakeRAG
        rag = cls(record.workspace)
        if self.active_query_config:
            rag.kb_active_query_config = dict(self.active_query_config)
        if self.active_config_version_id:
            rag.kb_active_config_version_id = self.active_config_version_id
        if self.active_parser_hash:
            rag.kb_active_parser_hash = self.active_parser_hash
        if self.active_index_hash:
            rag.kb_active_index_hash = self.active_index_hash
        if self.active_query_hash:
            rag.kb_active_query_hash = self.active_query_hash
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


def test_kb_query_uses_active_query_defaults_and_metadata(tmp_path):
    client, _kb_service, _document_service, _registry, probe = _build_client(tmp_path)
    probe.active_query_config = {
        "mode": "local",
        "top_k": 7,
        "chunk_top_k": 3,
        "include_references": False,
    }
    probe.active_config_version_id = "cfg_active"
    probe.active_parser_hash = "sha256:parser-active"
    probe.active_index_hash = "sha256:index-active"
    probe.active_query_hash = "sha256:query-active"
    _create_kb(client, "kb_active_query")

    response = client.post(
        "/kbs/kb_active_query/query",
        json={"query": "active query defaults"},
        headers=_HEADERS,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "local"
    assert body["references"] is None
    assert body["metadata"] == {
        "config_version_id": "cfg_active",
        "parser_hash": "sha256:parser-active",
        "index_hash": "sha256:index-active",
        "query_hash": "sha256:query-active",
    }
    rag = probe.instances["kb_active_query"]
    param = rag.query_params[0]
    assert param.mode == "local"
    assert param.top_k == 7
    assert param.chunk_top_k == 3
    assert param.include_references is False


def test_kb_query_request_overrides_active_query_defaults(tmp_path):
    client, _kb_service, _document_service, _registry, probe = _build_client(tmp_path)
    probe.active_query_config = {
        "mode": "local",
        "top_k": 7,
        "include_references": False,
    }
    _create_kb(client, "kb_active_query_override")

    response = client.post(
        "/kbs/kb_active_query_override/query/data",
        json={
            "query": "override active defaults",
            "mode": "global",
            "top_k": 11,
            "include_references": True,
        },
        headers=_HEADERS,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["metadata"]["query_mode"] == "global"
    rag = probe.instances["kb_active_query_override"]
    param = rag.query_params[0]
    assert param.mode == "global"
    assert param.top_k == 11
    assert param.include_references is True


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


def test_kb_query_stream_includes_metadata_when_references_disabled(tmp_path):
    client, _kb_service, _document_service, _registry, probe = _build_client(
        tmp_path, streaming=True
    )
    probe.active_query_config = {"include_references": False}
    probe.active_config_version_id = "cfg_stream_active"
    probe.active_parser_hash = "sha256:stream-parser"
    probe.active_index_hash = "sha256:stream-index"
    probe.active_query_hash = "sha256:stream-query"
    _create_kb(client, "kb_stream_no_refs")

    with client.stream(
        "POST",
        "/kbs/kb_stream_no_refs/query/stream",
        json={"query": "streaming no references", "stream": True},
        headers=_HEADERS,
    ) as response:
        assert response.status_code == 200
        body = b"".join(response.iter_bytes()).decode("utf-8")

    lines = [line for line in body.split("\n") if line]
    parsed = [json.loads(line) for line in lines]
    assert parsed[0] == {
        "kb_id": "kb_stream_no_refs",
        "metadata": {
            "config_version_id": "cfg_stream_active",
            "parser_hash": "sha256:stream-parser",
            "index_hash": "sha256:stream-index",
            "query_hash": "sha256:stream-query",
        },
    }
    assert [item["response"] for item in parsed[1:]] == ["first ", "second ", "third"]


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


# ---------------------------------------------------------------------------
# Per-document retrieval scoping (enabled/archived + filters.doc_ids → ids)
# ---------------------------------------------------------------------------


def _doc_record(
    doc_id: str,
    *,
    lightrag_doc_id: str | None,
    enabled: bool = True,
    archived: bool = False,
):
    from lightrag.api.metadata_store import DocumentRecord

    return DocumentRecord(
        id=doc_id,
        kb_id="kb_scope",
        workspace="kb_scope_ws",
        lightrag_doc_id=lightrag_doc_id,
        source_type="upload",
        source_name=f"{doc_id}.txt",
        source_uri=f"/tmp/{doc_id}.txt",
        source_hash="sha256:x",
        content_type="text/plain",
        size_bytes=1,
        parser_hash=None,
        index_hash=None,
        status="ready",
        enabled=enabled,
        archived=archived,
        chunks_count=None,
        entity_count=None,
        relation_count=None,
        error_code=None,
        error_message=None,
        metadata={},
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        deleted_at=None,
    )


class _FakeDocService:
    def __init__(self, documents):
        self._documents = documents

    async def list_documents(self, kb_id, *, limit=50, offset=0, **kwargs):
        total = len(self._documents)
        return self._documents[offset : offset + limit], total


def test_resolve_doc_scope_all_enabled_returns_none():
    from lightrag.api.routers.kb_query_routes import _resolve_doc_id_scope

    docs = [
        _doc_record("doc_a", lightrag_doc_id="lr-a"),
        _doc_record("doc_b", lightrag_doc_id="lr-b"),
    ]
    scope = asyncio.run(
        _resolve_doc_id_scope(_FakeDocService(docs), "kb_scope", None)
    )
    assert scope is None  # unrestricted, full recall


def test_resolve_doc_scope_excludes_disabled_and_archived():
    from lightrag.api.routers.kb_query_routes import _resolve_doc_id_scope

    docs = [
        _doc_record("doc_a", lightrag_doc_id="lr-a"),
        _doc_record("doc_b", lightrag_doc_id="lr-b", enabled=False),
        _doc_record("doc_c", lightrag_doc_id="lr-c", archived=True),
    ]
    scope = asyncio.run(
        _resolve_doc_id_scope(_FakeDocService(docs), "kb_scope", None)
    )
    assert scope == ["lr-a"]


def test_resolve_doc_scope_intersects_doc_ids_with_retrievable():
    from lightrag.api.routers.kb_query_routes import _resolve_doc_id_scope

    docs = [
        _doc_record("doc_a", lightrag_doc_id="lr-a"),
        _doc_record("doc_b", lightrag_doc_id="lr-b", enabled=False),
        _doc_record("doc_c", lightrag_doc_id="lr-c"),
    ]
    # Requesting a disabled doc drops it; only retrievable requested docs remain.
    scope = asyncio.run(
        _resolve_doc_id_scope(
            _FakeDocService(docs), "kb_scope", ["doc_a", "doc_b"]
        )
    )
    assert scope == ["lr-a"]


def test_resolve_doc_scope_unindexed_docs_have_no_lightrag_id():
    from lightrag.api.routers.kb_query_routes import _resolve_doc_id_scope

    docs = [
        _doc_record("doc_a", lightrag_doc_id="lr-a"),
        _doc_record("doc_b", lightrag_doc_id=None, enabled=False),
    ]
    scope = asyncio.run(
        _resolve_doc_id_scope(_FakeDocService(docs), "kb_scope", None)
    )
    assert scope == ["lr-a"]


# ---------------------------------------------------------------------------
# Engine-level chunk filter helpers (operate.py)
# ---------------------------------------------------------------------------


def test_chunk_doc_scope_helpers():
    from lightrag.operate import _chunk_in_doc_scope, _normalize_doc_id_allowlist

    # None ids => no scoping
    assert _normalize_doc_id_allowlist(QueryParam(ids=None)) is None
    assert _chunk_in_doc_scope({"full_doc_id": "lr-a"}, None) is True

    allow = _normalize_doc_id_allowlist(QueryParam(ids=["lr-a", "lr-b"]))
    assert allow == {"lr-a", "lr-b"}
    assert _chunk_in_doc_scope({"full_doc_id": "lr-a"}, allow) is True
    assert _chunk_in_doc_scope({"full_doc_id": "lr-z"}, allow) is False
    # Records without full_doc_id are excluded when filtering is active.
    assert _chunk_in_doc_scope({"content": "x"}, allow) is False
    # Empty allow-list => nothing is in scope.
    empty = _normalize_doc_id_allowlist(QueryParam(ids=[]))
    assert empty == set()
    assert _chunk_in_doc_scope({"full_doc_id": "lr-a"}, empty) is False
