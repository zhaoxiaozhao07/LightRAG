"""Engine-level test: rebuild_knowledge_from_chunks really re-summarizes via the
LLM from cached extractions — not the hand-written stand-in that
``test_shared_graph_delete.py`` patches in.

``test_shared_graph_delete.py`` monkeypatches ``rebuild_knowledge_from_chunks``
with ``_rebuild_from_remaining_chunks`` (a no-LLM stand-in) to assert source-id
narrowing. That leaves the real rebuild — which reconstructs descriptions from
cached extraction results and invokes the LLM map-reduce summary when an entity
keeps >= ``force_llm_summary_on_merge`` descriptions — unverified. This drives
the REAL ``rebuild_knowledge_from_chunks``:

* seed three chunks' cached extraction results for one shared entity (a distinct
  description per chunk),
* run the real rebuild with ``force_llm_summary_on_merge=3`` so the three
  surviving descriptions must go through the LLM summary,
* assert the LLM summary was actually invoked and the rebuilt description (not
  the stale seed) is written back to the graph + entity vector store.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.lightrag import LightRAG
from lightrag.operate import rebuild_knowledge_from_chunks
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id

pytestmark = pytest.mark.offline


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


def _extract_record(entity: str, description: str) -> str:
    # MinerU/extract text-mode record the rebuild parser understands
    # (tuple_delimiter "<|#|>", completion "<|COMPLETE|>").
    return f"(entity<|#|>{entity}<|#|>PERSON<|#|>{description})<|COMPLETE|>"


@pytest.mark.asyncio
async def test_rebuild_resummarizes_shared_entity_from_cache(tmp_path):
    summary_prompts: list[str] = []

    async def _spy_llm(prompt, **kwargs) -> str:
        # The ONLY LLM call the rebuild makes is the description summary —
        # extraction itself is read from cache. Record it and return a sentinel.
        summary_prompts.append(str(prompt))
        return "REBUILT-MERGED-DESCRIPTION"

    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"rebuild_{uuid4().hex[:8]}",
        llm_model_func=_spy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("test-tokenizer", _SimpleTokenizerImpl()),
        # >= 3 surviving descriptions must be merged via the LLM (min allowed is 3).
        force_llm_summary_on_merge=3,
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    try:
        entity = "SHARED-ENTITY"
        chunks = ["chunk-a", "chunk-b", "chunk-c"]
        ent_vid = compute_mdhash_id(entity, prefix="ent-")

        # text_chunks each reference one cached extraction result.
        await rag.text_chunks.upsert(
            {
                chunk_id: {
                    "content": f"{chunk_id} content",
                    "file_path": f"{chunk_id}.txt",
                    "full_doc_id": f"doc-{chunk_id}",
                    "llm_cache_list": [f"cache-{chunk_id}"],
                }
                for chunk_id in chunks
            }
        )
        # Cached extraction results: the shared entity with a distinct
        # description per chunk (so the rebuild has 3 to reconcile).
        await rag.llm_response_cache.upsert(
            {
                f"cache-{chunk_id}": {
                    "cache_type": "extract",
                    "chunk_id": chunk_id,
                    "create_time": idx + 1,
                    "return": _extract_record(
                        entity, f"description-from-{chunk_id}"
                    ),
                }
                for idx, chunk_id in enumerate(chunks)
            }
        )
        # A stale graph node + vector that the rebuild must overwrite.
        await rag.chunk_entity_relation_graph.upsert_node(
            entity,
            {
                "entity_id": entity,
                "source_id": GRAPH_FIELD_SEP.join(chunks),
                "description": "STALE-DESCRIPTION",
                "entity_type": "PERSON",
                "file_path": "chunk-a.txt",
                "created_at": 1,
                "truncate": "",
            },
        )
        await rag.entities_vdb.upsert(
            {
                ent_vid: {
                    "content": f"{entity}\nSTALE-DESCRIPTION",
                    "entity_name": entity,
                    "source_id": GRAPH_FIELD_SEP.join(chunks),
                    "description": "STALE-DESCRIPTION",
                    "entity_type": "PERSON",
                    "file_path": "chunk-a.txt",
                }
            }
        )
        await rag.entity_chunks.upsert(
            {entity: {"chunk_ids": list(chunks), "count": len(chunks)}}
        )

        # Run the REAL rebuild (not the patched stand-in).
        await rebuild_knowledge_from_chunks(
            entities_to_rebuild={entity: list(chunks)},
            relationships_to_rebuild={},
            knowledge_graph_inst=rag.chunk_entity_relation_graph,
            entities_vdb=rag.entities_vdb,
            relationships_vdb=rag.relationships_vdb,
            text_chunks_storage=rag.text_chunks,
            llm_response_cache=rag.llm_response_cache,
            global_config=rag._build_global_config(),
            entity_chunks_storage=rag.entity_chunks,
            relation_chunks_storage=rag.relation_chunks,
        )

        # The LLM summary path was actually exercised (not skipped by a patch).
        assert summary_prompts, (
            "rebuild must invoke the LLM summary to merge the 3 cached descriptions"
        )
        # The stale description was replaced by the rebuilt/summarized one.
        node = await rag.chunk_entity_relation_graph.get_node(entity)
        assert node is not None
        assert node["description"] != "STALE-DESCRIPTION", node
        assert node["description"] == "REBUILT-MERGED-DESCRIPTION", node
        # The entity vector store was rebuilt in lock-step with the graph.
        vector = await rag.entities_vdb.get_by_id(ent_vid)
        assert vector is not None
        assert "STALE-DESCRIPTION" not in (vector.get("content") or "")
    finally:
        await rag.finalize_storages()
