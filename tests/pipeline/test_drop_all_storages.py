"""Engine-level test for ``LightRAG.adrop_all_storages``.

Hard-deleting a KB used to only remove the on-disk ``working_dir/<workspace>``
folder, which purges file-based backends but leaves external backends
(PostgreSQL / Milvus / Neo4j / Qdrant / Redis / Mongo / OpenSearch) holding the
KB's vectors / graph / chunks / doc-status — orphaned data that a future KB
reusing the same workspace would read. ``adrop_all_storages`` closes that gap by
calling ``drop()`` on every storage; this test seeds a real (in-memory) LightRAG
across all storage classes and asserts they are all emptied and the per-storage
outcome is aggregated.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

from lightrag.base import DocStatus
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.lightrag import LightRAG
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id

pytestmark = pytest.mark.offline


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _dummy_llm(*args, **kwargs) -> str:
    return "ok"


async def _build_rag(tmp_path) -> LightRAG:
    workspace = f"drop_all_{uuid4().hex[:8]}"
    rag = LightRAG(
        working_dir=str(tmp_path / "drop_all"),
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


async def _seed(rag: LightRAG) -> dict[str, str]:
    doc = "doc-1"
    chunk = "chunk-1"
    entity = "ENTITY-1"
    now = datetime.now(timezone.utc).isoformat()
    created_at = int(datetime.now(timezone.utc).timestamp())

    await rag.full_docs.upsert({doc: {"content": "hi", "file_path": "a.txt"}})
    await rag.doc_status.upsert(
        {
            doc: {
                "status": DocStatus.PROCESSED,
                "content_summary": "hi",
                "content_length": 2,
                "chunks_count": 1,
                "chunks_list": [chunk],
                "created_at": now,
                "updated_at": now,
                "file_path": "a.txt",
                "track_id": "t-1",
                "error_msg": "",
                "metadata": {},
            }
        }
    )
    chunk_payload = {
        chunk: {"content": "hi chunk", "file_path": "a.txt", "full_doc_id": doc}
    }
    await rag.text_chunks.upsert(chunk_payload)
    await rag.chunks_vdb.upsert(chunk_payload)
    await rag.full_entities.upsert({doc: {"entity_names": [entity]}})
    await rag.full_relations.upsert({doc: {"relation_pairs": []}})
    await rag.entity_chunks.upsert({entity: {"chunk_ids": [chunk], "count": 1}})
    await rag.chunk_entity_relation_graph.upsert_node(
        entity,
        {
            "entity_id": entity,
            "source_id": GRAPH_FIELD_SEP.join([chunk]),
            "description": "an entity",
            "entity_type": "test",
            "file_path": "a.txt",
            "created_at": created_at,
            "truncate": "",
        },
    )
    await rag.entities_vdb.upsert(
        {
            compute_mdhash_id(entity, prefix="ent-"): {
                "content": f"{entity}\nan entity",
                "entity_name": entity,
                "source_id": chunk,
                "description": "an entity",
                "entity_type": "test",
                "file_path": "a.txt",
            }
        }
    )
    return {"doc": doc, "chunk": chunk, "entity": entity}


@pytest.mark.asyncio
async def test_adrop_all_storages_empties_every_storage(tmp_path):
    rag = await _build_rag(tmp_path)
    try:
        ids = await _seed(rag)
        # Sanity: data is present before the drop.
        assert await rag.full_docs.get_by_id(ids["doc"]) is not None
        assert await rag.text_chunks.get_by_id(ids["chunk"]) is not None
        assert await rag.doc_status.get_by_id(ids["doc"]) is not None
        assert (
            await rag.chunk_entity_relation_graph.get_node(ids["entity"]) is not None
        )
        assert (
            await rag.entities_vdb.get_by_id(
                compute_mdhash_id(ids["entity"], prefix="ent-")
            )
            is not None
        )

        summary = await rag.adrop_all_storages()

        # Every storage drop reported success and none failed.
        assert summary["failed"] == 0, summary
        assert summary["errors"] == []
        # All 12 storages (full_docs..doc_status incl. llm_response_cache) dropped.
        assert summary["dropped"] == 12, summary

        # The actual data is gone from each backend.
        assert await rag.full_docs.get_by_id(ids["doc"]) is None
        assert await rag.text_chunks.get_by_id(ids["chunk"]) is None
        assert await rag.doc_status.get_by_id(ids["doc"]) is None
        assert await rag.chunk_entity_relation_graph.get_node(ids["entity"]) is None
        assert await rag.full_entities.get_by_id(ids["doc"]) is None
        assert await rag.entity_chunks.get_by_id(ids["entity"]) is None
        assert (
            await rag.entities_vdb.get_by_id(
                compute_mdhash_id(ids["entity"], prefix="ent-")
            )
            is None
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_adrop_all_storages_aggregates_failures(tmp_path, monkeypatch):
    """A storage whose ``drop()`` raises or returns an error status is counted as
    a failure and reported in ``errors`` — without aborting the other drops."""
    rag = await _build_rag(tmp_path)
    try:
        await _seed(rag)

        async def _boom() -> dict[str, str]:
            raise RuntimeError("backend unreachable")

        async def _error_status() -> dict[str, str]:
            return {"status": "error", "message": "vdb refused"}

        monkeypatch.setattr(rag.full_docs, "drop", _boom)
        monkeypatch.setattr(rag.entities_vdb, "drop", _error_status)

        summary = await rag.adrop_all_storages()

        assert summary["failed"] == 2, summary
        assert summary["dropped"] == 10, summary
        joined = " ".join(summary["errors"])
        assert "full_docs" in joined
        assert "entities_vdb" in joined
        # A non-failing storage was still dropped.
        assert await rag.text_chunks.get_by_id("chunk-1") is None
    finally:
        await rag.finalize_storages()
