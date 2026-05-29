"""端到端流水线集成测试。

使用 TestClient 模拟客户端，覆盖整条链路：

    1. 创建 KB
    2. 上传两份"文件"（auto_parse=true）
    3. 等 parse / build_kg job 终态
    4. 问答（不带历史会话）
    5. 第二次运行：第一份文件未变 -> hash 命中 skip；第二份文件内容修改 -> 重新 parse + build；新增第三份文件 -> 走完整流程
    6. 再次问答

不依赖真实 LLM —— 通过 FakeRAG 模拟解析与索引。
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightrag.api.document_lifecycle_service import DocumentLifecycleService
from lightrag.api.document_lifecycle_service import DocumentSourceInput
from lightrag.api.index_build_service import IndexBuildService
from lightrag.api.job_service import JobService
from lightrag.api.kb_service import KnowledgeBaseService
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry, LightRAGLike
from lightrag.api.metadata_store import (
    DuplicateDocumentSourceKeyError,
    SQLiteMetadataStore,
)

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_kb_routes = importlib.import_module("lightrag.api.routers.kb_routes")
_kb_document_routes = importlib.import_module("lightrag.api.routers.kb_document_routes")
_kb_query_routes = importlib.import_module("lightrag.api.routers.kb_query_routes")
sys.argv = _original_argv

create_kb_routes = _kb_routes.create_kb_routes
create_kb_document_routes = _kb_document_routes.create_kb_document_routes
create_kb_query_routes = _kb_query_routes.create_kb_query_routes

pytestmark = pytest.mark.offline

_API_KEY = "test-key"
_HEADERS = {"X-API-Key": _API_KEY}


class FakeDocStatus:
    def __init__(self):
        self.rows: dict[str, dict] = {}

    async def get_by_ids(self, ids):
        return [self.rows.get(item_id) for item_id in ids]

    def stamp(self, doc_id: str) -> None:
        self.rows[doc_id] = {
            "status": "processed",
            "chunks_count": 4,
            "entity_count": 9,
            "relation_count": 6,
        }


class FakeDeletionResult:
    def __init__(self, doc_id: str, delete_llm_cache: bool):
        self.status = "success"
        self.doc_id = doc_id
        self.delete_llm_cache = delete_llm_cache
        self.message = "deleted"
        self.status_code = 200
        self.file_path = None


class FakeRAG:
    """足以驱动 parse + build + query 的 LightRAG 替身。"""

    def __init__(self, workspace: str):
        self.workspace = workspace
        self.embedding_dim = 768
        self.chunk_token_size = 512
        self.chunk_overlap_token_size = 64
        self.tiktoken_model_name = "gpt-4o-mini"
        self.summary_max_tokens = 800
        self.force_llm_summary_on_merge = False
        self.addon_params = {
            "chunker": {"strategy": "F"},
            "entity_types": ["concept"],
            "language": "en",
            "extraction": {"prompt_version": "v1"},
        }

        class _Embed:
            __name__ = "fake_embed"
            func_name = "fake_embed"

        self.embedding_func = _Embed()
        self.doc_status = FakeDocStatus()
        self.documents_indexed: list[str] = []
        self.delete_calls: list[tuple[str, bool]] = []
        self.queries: list[str] = []

    async def finalize_storages(self) -> None:
        return None

    async def adelete_by_doc_id(self, doc_id: str, *, delete_llm_cache: bool = False):
        self.delete_calls.append((doc_id, delete_llm_cache))
        return FakeDeletionResult(doc_id, delete_llm_cache)

    async def parse_native(self, doc_id, file_path, content_data):
        return await self._parse(doc_id, file_path)

    async def parse_mineru(self, doc_id, file_path, content_data):
        return await self._parse(doc_id, file_path)

    async def parse_docling(self, doc_id, file_path, content_data):
        return await self._parse(doc_id, file_path)

    async def _parse(self, doc_id, file_path):
        source = Path(file_path)
        parsed_dir = source.parent / "__parsed__" / f"{source.name}.parsed"
        parsed_dir.mkdir(parents=True, exist_ok=True)
        blocks_path = parsed_dir / f"{source.stem}.blocks.jsonl"
        blocks_path.write_text(
            '{"type":"meta"}\n{"type":"content","content":"hello"}\n',
            encoding="utf-8",
        )
        return {
            "doc_id": doc_id,
            "file_path": str(source),
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
        for doc_id in ids:
            self.documents_indexed.append(doc_id)
            self.doc_status.stamp(doc_id)
        return track_id

    async def apipeline_process_enqueue_documents(self):
        return None

    async def aquery_llm(self, query: str, *, param):
        self.queries.append(query)
        return {
            "llm_response": {
                "content": (
                    f"[{self.workspace}] indexed {len(self.documents_indexed)} docs; "
                    f"answering: {query}"
                ),
                "is_streaming": False,
            },
            "data": {
                "references": [
                    {
                        "reference_id": str(idx + 1),
                        "file_path": f"{self.workspace}/doc_{idx}.txt",
                    }
                    for idx in range(len(self.documents_indexed))
                ],
                "chunks": [],
            },
        }

    async def aquery_data(self, query: str, *, param):
        return {
            "status": "success",
            "message": "ok",
            "data": {
                "entities": [],
                "relationships": [],
                "chunks": [],
                "references": [],
            },
            "metadata": {"query_mode": param.mode},
        }


class BuilderProbe:
    def __init__(self):
        self.instances: dict[str, FakeRAG] = {}

    async def build(self, record) -> FakeRAG:
        rag = FakeRAG(record.workspace)
        self.instances[record.id] = rag
        return rag

    async def finalize(self, rag: LightRAGLike) -> None:
        return None


def _build_client(tmp_path: Path):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kb.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    document_service = DocumentLifecycleService(
        kb_service, metadata_store, tmp_path / "inputs"
    )
    job_service = JobService(kb_service, metadata_store)
    index_service = IndexBuildService(document_service)
    probe = BuilderProbe()
    registry = LightRAGInstanceRegistry(kb_service, probe.build, probe.finalize)
    app = FastAPI()
    app.state.document_service = document_service
    app.state.metadata_store = metadata_store
    app.include_router(
        create_kb_routes(
            kb_service, registry, api_key=_API_KEY, job_service=job_service
        )
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
    app.include_router(
        create_kb_query_routes(document_service, registry, api_key=_API_KEY)
    )
    return TestClient(app), probe


def _create_kb(client: TestClient, kb_id: str) -> dict:
    response = client.post("/kbs", json={"id": kb_id, "name": kb_id}, headers=_HEADERS)
    assert response.status_code == 200
    return response.json()


def _upload(
    client: TestClient,
    kb_id: str,
    *,
    name: str,
    content: bytes,
    mime: str = "text/plain",
) -> dict:
    response = client.post(
        f"/kbs/{kb_id}/documents:upload",
        files=[("files", (name, content, mime))],
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    return response.json()


def _parse(client: TestClient, kb_id: str, doc_id: str) -> dict:
    response = client.post(
        f"/kbs/{kb_id}/documents/{doc_id}:parse",
        json={"engine": "mineru", "process_options": "iF"},
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    return response.json()


def _wait(client: TestClient, kb_id: str, job_id: str) -> dict:
    response = client.post(
        f"/kbs/{kb_id}/jobs/{job_id}:wait?timeout_seconds=10",
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    return response.json()


def _build_kg(client: TestClient, kb_id: str, doc_id: str) -> dict:
    response = client.post(
        f"/kbs/{kb_id}/documents/{doc_id}:build-kg",
        json={},
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    return response.json()


def _sync(
    client: TestClient,
    kb_id: str,
    items: list[tuple[str, str, bytes]],
    *,
    process_options: str = "iF",
    force_reparse: bool = False,
    idempotency_key: str | None = None,
) -> dict:
    files = [
        ("files", (filename, content, "application/pdf"))
        for _source_key, filename, content in items
    ]
    data = {"source_keys": [source_key for source_key, _filename, _content in items]}
    params = {
        "parser_engine": "mineru",
        "process_options": process_options,
    }
    if force_reparse:
        params["force_reparse"] = "true"
    if idempotency_key is not None:
        params["idempotency_key"] = idempotency_key
    response = client.post(
        f"/kbs/{kb_id}/documents:sync",
        params=params,
        data=data,
        files=files,
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_sync_documents_runs_incremental_update_to_query_ready(tmp_path):
    client, probe = _build_client(tmp_path)
    kb = _create_kb(client, "kb_sync")

    first_job = _sync(
        client,
        "kb_sync",
        [
            ("manual/doc_a.pdf", "doc_a.pdf", b"alpha document v1"),
            ("manual/doc_b.pdf", "doc_b.pdf", b"beta document v1"),
        ],
        idempotency_key="sync-round-1",
    )
    first_final = _wait(client, "kb_sync", first_job["id"])
    assert first_final["status"] == "succeeded"
    assert first_final["job_type"] == "sync"
    assert first_final["completed_items"] == 2
    assert {item["action"] for item in first_final["result"]["items"]} == {"created"}
    assert all(
        item["build_result"]["status"] == "succeeded"
        for item in first_final["result"]["items"]
    )

    listing = client.get("/kbs/kb_sync/documents", headers=_HEADERS)
    assert listing.status_code == 200
    documents = listing.json()["documents"]
    assert {document["status"] for document in documents} == {"ready"}
    assert {document["metadata"]["source_key"] for document in documents} == {
        "manual/doc_a.pdf",
        "manual/doc_b.pdf",
    }

    rag = probe.instances["kb_sync"]
    assert len(rag.documents_indexed) == 2

    second_job = _sync(
        client,
        "kb_sync",
        [
            ("manual/doc_a.pdf", "doc_a.pdf", b"alpha document v1"),
            ("manual/doc_b.pdf", "doc_b.pdf", b"beta document v1"),
        ],
        idempotency_key="sync-round-2",
    )
    second_final = _wait(client, "kb_sync", second_job["id"])
    assert second_final["status"] == "succeeded"
    assert second_final["result"]["skipped_items"] == 2
    assert {item["status"] for item in second_final["result"]["items"]} == {"skipped"}
    assert len(rag.documents_indexed) == 2

    third_job = _sync(
        client,
        "kb_sync",
        [
            ("manual/doc_a.pdf", "doc_a.pdf", b"alpha document v1"),
            ("manual/doc_b.pdf", "doc_b.pdf", b"beta document v2"),
        ],
        idempotency_key="sync-round-3",
    )
    third_final = _wait(client, "kb_sync", third_job["id"])
    assert third_final["status"] == "succeeded"
    actions_by_key = {
        item["source_key"]: item["action"] for item in third_final["result"]["items"]
    }
    assert actions_by_key == {
        "manual/doc_a.pdf": "skipped",
        "manual/doc_b.pdf": "replaced",
    }
    assert len(rag.delete_calls) == 1
    assert len(rag.documents_indexed) == 3

    answer = client.post(
        "/kbs/kb_sync/query",
        json={
            "query": "what changed?",
            "mode": "mix",
            "include_references": True,
            "stream": False,
        },
        headers=_HEADERS,
    )
    assert answer.status_code == 200
    payload = answer.json()
    assert payload["kb_id"] == "kb_sync"
    assert kb["workspace"] in payload["response"]
    assert "indexed 3 docs" in payload["response"]


def test_sync_documents_reuses_idempotency_key(tmp_path):
    client, _probe = _build_client(tmp_path)
    _create_kb(client, "kb_sync_idempotent")

    first = _sync(
        client,
        "kb_sync_idempotent",
        [("manual/doc.pdf", "doc.pdf", b"same")],
        idempotency_key="sync-repeat",
    )
    repeated = _sync(
        client,
        "kb_sync_idempotent",
        [("manual/doc.pdf", "doc.pdf", b"same")],
        idempotency_key="sync-repeat",
    )

    assert repeated["id"] == first["id"]
    conflict = client.post(
        "/kbs/kb_sync_idempotent/documents:sync",
        params={
            "parser_engine": "mineru",
            "process_options": "iF",
            "idempotency_key": "sync-repeat",
        },
        data={"source_keys": ["manual/doc.pdf"]},
        files=[("files", ("doc.pdf", b"changed", "application/pdf"))],
        headers=_HEADERS,
    )
    assert conflict.status_code == 409


def test_sync_reparses_unchanged_source_when_parser_hash_changes(tmp_path):
    client, probe = _build_client(tmp_path)
    _create_kb(client, "kb_sync_parser_hash")

    first_job = _sync(
        client,
        "kb_sync_parser_hash",
        [("manual/doc.pdf", "doc.pdf", b"same bytes")],
        process_options="iF",
        idempotency_key="parser-hash-1",
    )
    first_final = _wait(client, "kb_sync_parser_hash", first_job["id"])
    assert first_final["status"] == "succeeded"
    first_item = first_final["result"]["items"][0]
    first_parser_hash = first_item["parse_result"]["parser_hash"]

    rag = probe.instances["kb_sync_parser_hash"]
    assert len(rag.documents_indexed) == 1

    second_job = _sync(
        client,
        "kb_sync_parser_hash",
        [("manual/doc.pdf", "doc.pdf", b"same bytes")],
        process_options="itF",
        idempotency_key="parser-hash-2",
    )
    second_final = _wait(client, "kb_sync_parser_hash", second_job["id"])

    assert second_final["status"] == "succeeded"
    assert second_final["result"]["skipped_items"] == 0
    second_item = second_final["result"]["items"][0]
    assert second_item["action"] == "reparsed"
    assert second_item["status"] == "succeeded"
    assert second_item["parse_result"]["parser_hash"] != first_parser_hash
    assert second_item["build_result"]["status"] == "succeeded"
    assert second_item["build_result"].get("skipped") is not True
    assert len(rag.documents_indexed) == 2


def test_sync_force_reparse_preserves_active_build_conflict(tmp_path):
    client, _probe = _build_client(tmp_path)
    _create_kb(client, "kb_sync_active_build")

    first_job = _sync(
        client,
        "kb_sync_active_build",
        [("manual/doc.pdf", "doc.pdf", b"active bytes")],
        idempotency_key="active-build-1",
    )
    first_final = _wait(client, "kb_sync_active_build", first_job["id"])
    assert first_final["status"] == "succeeded"
    document_id = first_final["result"]["items"][0]["document_id"]

    asyncio.run(
        cast(Any, client.app).state.metadata_store.claim_document_build_queued(
            "kb_sync_active_build",
            document_id,
            metadata_patch={"pending_build_job_id": "job_active_build"},
        )
    )

    second_job = _sync(
        client,
        "kb_sync_active_build",
        [("manual/doc.pdf", "doc.pdf", b"active bytes")],
        force_reparse=True,
        idempotency_key="active-build-2",
    )
    second_final = _wait(client, "kb_sync_active_build", second_job["id"])

    assert second_final["status"] == "failed"
    assert second_final["failed_items"] == 1
    item = second_final["result"]["items"][0]
    assert item["status"] == "failed"
    assert item["error_code"] == "build_job_active"
    assert item["existing_job_id"] == "job_active_build"
    assert item["document_id"] == document_id


def test_source_key_identity_is_unique_in_metadata_store(tmp_path):
    client, _probe = _build_client(tmp_path)
    _create_kb(client, "kb_source_key_unique")
    document_service: DocumentLifecycleService = cast(
        Any, client.app
    ).state.document_service

    asyncio.run(
        document_service.create_source_batch(
            "kb_source_key_unique",
            [
                DocumentSourceInput(
                    source_name="first.txt",
                    content=b"first",
                    source_type="text",
                    content_type="text/plain",
                    metadata={"source_key": "manual/same.txt"},
                )
            ],
        )
    )

    with pytest.raises(DuplicateDocumentSourceKeyError):
        asyncio.run(
            document_service.create_source_batch(
                "kb_source_key_unique",
                [
                    DocumentSourceInput(
                        source_name="second.txt",
                        content=b"second",
                        source_type="text",
                        content_type="text/plain",
                        metadata={"source_key": "manual/same.txt"},
                    )
                ],
            )
        )

    documents = client.get("/kbs/kb_source_key_unique/documents", headers=_HEADERS)
    assert documents.status_code == 200
    assert documents.json()["total"] == 1


def test_full_pipeline_runs_and_supports_incremental_updates(tmp_path):
    client, probe = _build_client(tmp_path)
    kb = _create_kb(client, "kb_pipeline")

    # ===== 第一次运行：上传 + 解析 + 构建 + 问答 =====
    upload_a = _upload(
        client,
        "kb_pipeline",
        name="doc_a.pdf",
        content=b"alpha document v1",
        mime="application/pdf",
    )
    upload_b = _upload(
        client,
        "kb_pipeline",
        name="doc_b.pdf",
        content=b"beta document v1",
        mime="application/pdf",
    )
    parse_a_doc = upload_a["documents"][0]
    parse_b_doc = upload_b["documents"][0]
    parse_a_job = _parse(client, "kb_pipeline", parse_a_doc["id"])
    parse_b_job = _parse(client, "kb_pipeline", parse_b_doc["id"])
    parse_a_final = _wait(client, "kb_pipeline", parse_a_job["id"])
    parse_b_final = _wait(client, "kb_pipeline", parse_b_job["id"])
    assert parse_a_final["status"] == "succeeded"
    assert parse_b_final["status"] == "succeeded"

    build_a = _build_kg(client, "kb_pipeline", parse_a_doc["id"])
    build_b = _build_kg(client, "kb_pipeline", parse_b_doc["id"])
    final_a = _wait(client, "kb_pipeline", build_a["id"])
    final_b = _wait(client, "kb_pipeline", build_b["id"])
    assert final_a["status"] == "succeeded"
    assert final_b["status"] == "succeeded"

    rag = probe.instances["kb_pipeline"]
    enqueue_count_round_one = len(rag.documents_indexed)
    assert enqueue_count_round_one == 2

    answer_one = client.post(
        "/kbs/kb_pipeline/query",
        json={
            "query": "summarise the documents",
            "mode": "mix",
            "include_references": True,
            "stream": False,
        },
        headers=_HEADERS,
    )
    assert answer_one.status_code == 200
    payload_one = answer_one.json()
    assert payload_one["kb_id"] == "kb_pipeline"
    assert kb["workspace"] in payload_one["response"]
    assert "indexed 2 docs" in payload_one["response"]

    # ===== 第二次运行：模拟目录变化 =====
    # doc_a 内容不变 -> 客户端不应再上传；服务端 build_kg 命中 hash 也跳过
    # doc_b 内容更新 -> 重新走解析 + 构建
    # doc_c 新增 -> 走完整流程
    rebuild_a = _build_kg(client, "kb_pipeline", parse_a_doc["id"])
    final_rebuild_a = _wait(client, "kb_pipeline", rebuild_a["id"])
    assert final_rebuild_a["status"] == "succeeded"
    assert final_rebuild_a["result"]["skipped"] is True
    assert final_rebuild_a["result"]["skip_reason"] == "index_hash_match"

    upload_b_v2 = _upload(
        client,
        "kb_pipeline",
        name="doc_b.pdf",
        content=b"beta document v2 with more content",
        mime="application/pdf",
    )
    parse_b_v2_doc = upload_b_v2["documents"][0]
    assert parse_b_v2_doc["id"] != parse_b_doc["id"]
    parse_b_v2_job = _parse(client, "kb_pipeline", parse_b_v2_doc["id"])
    assert _wait(client, "kb_pipeline", parse_b_v2_job["id"])["status"] == "succeeded"
    rebuild_b_v2 = _build_kg(client, "kb_pipeline", parse_b_v2_doc["id"])
    final_rebuild_b = _wait(client, "kb_pipeline", rebuild_b_v2["id"])
    assert final_rebuild_b["status"] == "succeeded"
    assert final_rebuild_b["result"]["skipped"] is False

    upload_c = _upload(
        client,
        "kb_pipeline",
        name="doc_c.pdf",
        content=b"gamma document",
        mime="application/pdf",
    )
    parse_c_doc = upload_c["documents"][0]
    parse_c_job = _parse(client, "kb_pipeline", parse_c_doc["id"])
    assert _wait(client, "kb_pipeline", parse_c_job["id"])["status"] == "succeeded"
    build_c = _build_kg(client, "kb_pipeline", parse_c_doc["id"])
    final_c = _wait(client, "kb_pipeline", build_c["id"])
    assert final_c["status"] == "succeeded"

    # 索引调用次数：第一轮 2 条；第二轮 doc_a 命中 skip，doc_b_v2 + doc_c 各执行 1 次
    enqueue_count_round_two = len(rag.documents_indexed)
    assert enqueue_count_round_two == enqueue_count_round_one + 2

    # ===== 第二次问答（不带历史会话）=====
    answer_two = client.post(
        "/kbs/kb_pipeline/query",
        json={
            "query": "after incremental update",
            "mode": "mix",
            "include_references": True,
            "stream": False,
        },
        headers=_HEADERS,
    )
    assert answer_two.status_code == 200
    payload_two = answer_two.json()
    assert "indexed 4 docs" in payload_two["response"]
    refs = payload_two["references"]
    assert len(refs) == 4
    assert all(ref["file_path"].startswith(kb["workspace"] + "/") for ref in refs)

    # 文档列表应能枚举出所有 4 个文档
    listing = client.get("/kbs/kb_pipeline/documents", headers=_HEADERS)
    assert listing.status_code == 200
    documents = listing.json()
    assert documents["total"] == 4
    statuses = {item["status"] for item in documents["documents"]}
    assert statuses == {"ready"}
