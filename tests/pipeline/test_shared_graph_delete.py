"""Engine-level correctness test for shared-graph deletion (source attribution).

The API delete tests assert only that ``adelete_by_doc_id`` is *called*; they do
not prove what the engine actually does to a graph that is *shared* between
documents. This test fills that gap with a real :class:`LightRAG` (in-memory
storages, fake LLM/embedding) seeded so that deleting one document must:

* **keep** an entity/relation still sourced by a surviving document, with its
  source attribution **narrowed** to the remaining chunks (rebuild path), and
* **prune** an entity/relation whose *only* source was the deleted document
  (orphan path) — removing it from both the graph and the entity/relation
  vector stores.

This is the "safe" / "rebuild_doc_scope" strategy contract: shared entities are
never lost, orphans never linger. It mirrors the seed structure proven in
``tests/pipeline/test_doc_status_chunk_preservation.py`` but adds an explicit
orphan to assert pruning.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

import lightrag.lightrag as lightrag_module
from lightrag.base import DocStatus
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.lightrag import LightRAG
from lightrag.utils import (
    EmbeddingFunc,
    Tokenizer,
    compute_mdhash_id,
    make_relation_chunk_key,
)

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


async def _build_rag(tmp_path, test_name: str) -> LightRAG:
    workspace = f"{test_name}_{uuid4().hex[:8]}"
    rag = LightRAG(
        working_dir=str(tmp_path / test_name),
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


async def _rebuild_from_remaining_chunks(
    entities_to_rebuild,
    relationships_to_rebuild,
    knowledge_graph_inst,
    entities_vdb,
    relationships_vdb,
    **kwargs,
):
    """Faithful stand-in for ``rebuild_knowledge_from_chunks``: narrows the
    ``source_id`` of surviving entities/relations to the remaining chunks
    (no LLM needed)."""
    for entity_name, remaining_chunk_ids in entities_to_rebuild.items():
        node = await knowledge_graph_inst.get_node(entity_name)
        assert node is not None
        updated = {**node, "source_id": GRAPH_FIELD_SEP.join(remaining_chunk_ids)}
        await knowledge_graph_inst.upsert_node(entity_name, updated)
        await entities_vdb.upsert(
            {
                compute_mdhash_id(entity_name, prefix="ent-"): {
                    "content": f"{entity_name}\n{updated['description']}",
                    "entity_name": entity_name,
                    "source_id": updated["source_id"],
                    "description": updated["description"],
                    "entity_type": updated["entity_type"],
                    "file_path": updated["file_path"],
                }
            }
        )
    for (src, tgt), remaining_chunk_ids in relationships_to_rebuild.items():
        edge = await knowledge_graph_inst.get_edge(src, tgt)
        assert edge is not None
        updated = {**edge, "source_id": GRAPH_FIELD_SEP.join(remaining_chunk_ids)}
        await knowledge_graph_inst.upsert_edge(src, tgt, updated)
        await relationships_vdb.upsert(
            {
                compute_mdhash_id(src + tgt, prefix="rel-"): {
                    "content": f"{updated['keywords']}\t{src}\n{tgt}\n{updated['description']}",
                    "src_id": src,
                    "tgt_id": tgt,
                    "source_id": updated["source_id"],
                    "description": updated["description"],
                    "keywords": updated["keywords"],
                    "weight": updated["weight"],
                    "file_path": updated["file_path"],
                }
            }
        )


async def _seed_shared_graph(rag: LightRAG) -> dict[str, str]:
    """Seed two docs that share entity SHARED + relation, where doc_drop also
    uniquely contributes orphan entity ORPHAN + relation SHARED-ORPHAN.

    chunk ownership:
      - chunk_keep  -> doc_keep (survivor)
      - chunk_drop  -> doc_drop (to be deleted)

    SHARED entity sourced by [chunk_keep, chunk_drop]   -> survives, narrowed.
    ORPHAN entity sourced by [chunk_drop] only           -> pruned.
    """
    doc_keep = "doc-keep"
    doc_drop = "doc-drop"
    chunk_keep = "chunk-keep"
    chunk_drop = "chunk-drop"
    shared = "SHARED-ENTITY"
    orphan = "ORPHAN-ENTITY"
    shared_orphan_rel = make_relation_chunk_key(shared, orphan)
    now = datetime.now(timezone.utc).isoformat()
    created_at = int(datetime.now(timezone.utc).timestamp())

    # --- documents + doc_status (doc_drop owns chunk_drop) ---
    await rag.full_docs.upsert(
        {
            doc_keep: {"content": "keep doc", "file_path": "keep.txt"},
            doc_drop: {"content": "drop doc", "file_path": "drop.txt"},
        }
    )
    await rag.doc_status.upsert(
        {
            doc_drop: {
                "status": DocStatus.PROCESSED,
                "content_summary": "drop",
                "content_length": 8,
                "chunks_count": 1,
                "chunks_list": [chunk_drop],
                "created_at": now,
                "updated_at": now,
                "file_path": "drop.txt",
                "track_id": "track-drop",
                "error_msg": "",
                "metadata": {},
            }
        }
    )

    # --- chunks (full_doc_id attributes each chunk to its owning doc) ---
    chunk_payload = {
        chunk_keep: {"content": "keep chunk", "file_path": "keep.txt", "full_doc_id": doc_keep},
        chunk_drop: {"content": "drop chunk", "file_path": "drop.txt", "full_doc_id": doc_drop},
    }
    await rag.text_chunks.upsert(chunk_payload)
    await rag.chunks_vdb.upsert(chunk_payload)

    # --- per-doc entity/relation contributions (drives affected_nodes) ---
    await rag.full_entities.upsert({doc_drop: {"entity_names": [shared, orphan]}})
    await rag.full_relations.upsert(
        {doc_drop: {"relation_pairs": [(shared, orphan)]}}
    )

    # --- entity/relation chunk tracking (decides rebuild vs delete) ---
    await rag.entity_chunks.upsert(
        {
            shared: {"chunk_ids": [chunk_keep, chunk_drop], "count": 2},
            orphan: {"chunk_ids": [chunk_drop], "count": 1},
        }
    )
    await rag.relation_chunks.upsert(
        {shared_orphan_rel: {"chunk_ids": [chunk_drop], "count": 1}}
    )

    # --- graph nodes ---
    await rag.chunk_entity_relation_graph.upsert_node(
        shared,
        {
            "entity_id": shared,
            "source_id": GRAPH_FIELD_SEP.join([chunk_keep, chunk_drop]),
            "description": "shared entity",
            "entity_type": "test",
            "file_path": "keep.txt",
            "created_at": created_at,
            "truncate": "",
        },
    )
    await rag.chunk_entity_relation_graph.upsert_node(
        orphan,
        {
            "entity_id": orphan,
            "source_id": chunk_drop,
            "description": "orphan entity",
            "entity_type": "test",
            "file_path": "drop.txt",
            "created_at": created_at,
            "truncate": "",
        },
    )
    await rag.chunk_entity_relation_graph.upsert_edge(
        shared,
        orphan,
        {
            "source": shared,
            "target": orphan,
            "source_id": chunk_drop,
            "description": "shared-orphan relation",
            "keywords": "test",
            "weight": 1.0,
            "file_path": "drop.txt",
        },
    )

    # --- vector stores ---
    await rag.entities_vdb.upsert(
        {
            compute_mdhash_id(shared, prefix="ent-"): {
                "content": f"{shared}\nshared entity",
                "entity_name": shared,
                "source_id": GRAPH_FIELD_SEP.join([chunk_keep, chunk_drop]),
                "description": "shared entity",
                "entity_type": "test",
                "file_path": "keep.txt",
            },
            compute_mdhash_id(orphan, prefix="ent-"): {
                "content": f"{orphan}\norphan entity",
                "entity_name": orphan,
                "source_id": chunk_drop,
                "description": "orphan entity",
                "entity_type": "test",
                "file_path": "drop.txt",
            },
        }
    )
    await rag.relationships_vdb.upsert(
        {
            compute_mdhash_id(shared + orphan, prefix="rel-"): {
                "content": f"test\t{shared}\n{orphan}\nshared-orphan relation",
                "src_id": shared,
                "tgt_id": orphan,
                "source_id": chunk_drop,
                "description": "shared-orphan relation",
                "keywords": "test",
                "weight": 1.0,
                "file_path": "drop.txt",
            }
        }
    )

    return {
        "doc_keep": doc_keep,
        "doc_drop": doc_drop,
        "chunk_keep": chunk_keep,
        "chunk_drop": chunk_drop,
        "shared": shared,
        "orphan": orphan,
    }


async def test_delete_keeps_shared_entity_and_prunes_orphan(tmp_path, monkeypatch):
    rag = await _build_rag(tmp_path, "shared_graph_delete")
    monkeypatch.setattr(
        lightrag_module,
        "rebuild_knowledge_from_chunks",
        _rebuild_from_remaining_chunks,
    )
    try:
        ids = await _seed_shared_graph(rag)
        shared = ids["shared"]
        orphan = ids["orphan"]
        chunk_keep = ids["chunk_keep"]
        chunk_drop = ids["chunk_drop"]

        result = await rag.adelete_by_doc_id(ids["doc_drop"])
        assert result.status == "success"

        # --- the deleted document is gone ---
        assert await rag.doc_status.get_by_id(ids["doc_drop"]) is None
        assert await rag.full_docs.get_by_id(ids["doc_drop"]) is None
        assert await rag.text_chunks.get_by_id(chunk_drop) is None
        # the surviving document's chunk is untouched
        assert await rag.text_chunks.get_by_id(chunk_keep) is not None

        # --- SHARED entity SURVIVES with narrowed source attribution ---
        shared_node = await rag.chunk_entity_relation_graph.get_node(shared)
        assert shared_node is not None, "shared entity must not be deleted"
        shared_sources = [
            c for c in shared_node["source_id"].split(GRAPH_FIELD_SEP) if c
        ]
        assert shared_sources == [chunk_keep], "source attribution must narrow to survivor"
        shared_tracking = await rag.entity_chunks.get_by_id(shared)
        assert shared_tracking is not None
        assert shared_tracking["chunk_ids"] == [chunk_keep]
        # still retrievable in the entity vector store
        assert (
            await rag.entities_vdb.get_by_id(compute_mdhash_id(shared, prefix="ent-"))
            is not None
        )

        # --- ORPHAN entity is PRUNED from graph + vector store ---
        assert (
            await rag.chunk_entity_relation_graph.get_node(orphan) is None
        ), "orphan entity (last source removed) must be pruned"
        assert (
            await rag.entities_vdb.get_by_id(compute_mdhash_id(orphan, prefix="ent-"))
            is None
        ), "orphan entity vector must be pruned"

        # --- the shared-orphan relation is pruned (its only source was dropped) ---
        assert (
            await rag.chunk_entity_relation_graph.get_edge(shared, orphan) is None
        )
        assert (
            await rag.relationships_vdb.get_by_id(
                compute_mdhash_id(shared + orphan, prefix="rel-")
            )
            is None
        )
    finally:
        await rag.finalize_storages()
