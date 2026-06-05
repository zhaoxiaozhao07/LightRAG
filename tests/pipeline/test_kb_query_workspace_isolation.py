"""Engine-level workspace isolation + doc-id allow-list for KB query.

The KB query route tests (``tests/api/routes/test_kb_query_routes.py``) use a
``FakeRAG`` whose response merely embeds the workspace name, and the doc-scope
filtering is covered by ``operate.py`` unit tests on fake VDBs. Neither runs the
real retrieval path end to end. This test fills that gap with real (in-memory)
:class:`LightRAG` instances and the real ``aquery_data`` retrieval pipeline:

* two KBs on different workspaces must never surface each other's chunks
  through a real ``naive`` retrieval (workspace isolation at the engine level),
  and
* ``QueryParam.ids`` (the allow-list the route derives from
  ``enabled``/``archived``/``filters.doc_ids``) must actually drop out-of-scope
  chunks in the real retrieval, not just in the ``operate.py`` helper unit
  tests.

Offline: uses fake LLM/embedding and in-memory storages, so it always runs.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from lightrag.base import QueryParam
from lightrag.lightrag import LightRAG
from lightrag.utils import EmbeddingFunc, Tokenizer

pytestmark = pytest.mark.offline


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    # Constant vectors: every chunk is equally "similar", so naive retrieval
    # returns whatever is in *this* instance's chunks_vdb — isolation/allow-list
    # filtering, not ranking, is what this test exercises.
    return np.ones((len(texts), 8), dtype=float)


async def _dummy_llm(*args, **kwargs) -> str:
    return "ok"


async def _build_rag(tmp_path, name: str, workspace: str) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / name),
        workspace=workspace,
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("test-tokenizer", _SimpleTokenizerImpl()),
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    return rag


async def _seed_chunk(rag: LightRAG, chunk_id: str, *, content: str, full_doc_id: str):
    payload = {
        chunk_id: {
            "content": content,
            "file_path": f"{full_doc_id}.txt",
            "full_doc_id": full_doc_id,
        }
    }
    await rag.text_chunks.upsert(payload)
    await rag.chunks_vdb.upsert(payload)
    await rag.text_chunks.index_done_callback()
    await rag.chunks_vdb.index_done_callback()


def _chunk_contents(result: dict) -> str:
    chunks = result.get("data", {}).get("chunks", []) or []
    return " ".join(c.get("content", "") for c in chunks)


@pytest.mark.asyncio
async def test_two_kbs_isolate_query_retrieval(tmp_path):
    """Two real LightRAG instances on different workspaces: a naive query on KB
    A returns A's chunk and never B's, and vice versa."""
    suffix = uuid4().hex[:8]
    rag_a = await _build_rag(tmp_path, "qa", f"wsq_a_{suffix}")
    rag_b = await _build_rag(tmp_path, "qb", f"wsq_b_{suffix}")
    try:
        await _seed_chunk(
            rag_a, "chunk-a", content="ALPHA-ONLY-CONTENT", full_doc_id="doc-a"
        )
        await _seed_chunk(
            rag_b, "chunk-b", content="BETA-ONLY-CONTENT", full_doc_id="doc-b"
        )

        param = QueryParam(mode="naive", top_k=10, chunk_top_k=10)
        res_a = await rag_a.aquery_data("anything", param=param)
        res_b = await rag_b.aquery_data("anything", param=param)

        contents_a = _chunk_contents(res_a)
        contents_b = _chunk_contents(res_b)

        # Sanity: each KB retrieves its own seeded chunk.
        assert "ALPHA-ONLY-CONTENT" in contents_a, res_a
        assert "BETA-ONLY-CONTENT" in contents_b, res_b
        # Isolation: neither KB ever surfaces the other's content.
        assert "BETA-ONLY-CONTENT" not in contents_a, (
            f"KB A leaked KB B content: {contents_a}"
        )
        assert "ALPHA-ONLY-CONTENT" not in contents_b, (
            f"KB B leaked KB A content: {contents_b}"
        )
    finally:
        await rag_a.finalize_storages()
        await rag_b.finalize_storages()


@pytest.mark.asyncio
async def test_query_ids_allowlist_filters_retrieval_end_to_end(tmp_path):
    """``QueryParam.ids`` (full_doc_id allow-list) drops out-of-scope chunks in
    the real ``naive`` retrieval — the end-to-end counterpart to the operate.py
    helper unit tests. This is how the KB route enforces
    enabled/archived/filters.doc_ids at query time."""
    suffix = uuid4().hex[:8]
    rag = await _build_rag(tmp_path, "scope", f"wsq_scope_{suffix}")
    try:
        await _seed_chunk(
            rag, "c-in", content="IN-SCOPE-CONTENT", full_doc_id="doc-in"
        )
        await _seed_chunk(
            rag, "c-out", content="OUT-OF-SCOPE-CONTENT", full_doc_id="doc-out"
        )

        # No allow-list: both chunks are retrievable.
        res_all = await rag.aquery_data(
            "anything", param=QueryParam(mode="naive", top_k=10, chunk_top_k=10)
        )
        contents_all = _chunk_contents(res_all)
        assert "IN-SCOPE-CONTENT" in contents_all, res_all
        assert "OUT-OF-SCOPE-CONTENT" in contents_all, res_all

        # Allow-list pinned to doc-in: only its chunk survives retrieval.
        res_scoped = await rag.aquery_data(
            "anything",
            param=QueryParam(
                mode="naive", top_k=10, chunk_top_k=10, ids=["doc-in"]
            ),
        )
        contents_scoped = _chunk_contents(res_scoped)
        assert "IN-SCOPE-CONTENT" in contents_scoped, res_scoped
        assert "OUT-OF-SCOPE-CONTENT" not in contents_scoped, (
            f"allow-list did not drop the out-of-scope chunk: {contents_scoped}"
        )
    finally:
        await rag.finalize_storages()
