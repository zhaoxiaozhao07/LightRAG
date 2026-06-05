"""Gated integration test: two KBs on the SAME Neo4j database must not leak
graph data across workspaces.

When ``LIGHTRAG_GRAPH_STORAGE=Neo4JStorage`` runs with a single
``NEO4J_DATABASE`` (the recommended single-DB-per-deployment setup), different
KBs are isolated only by a per-workspace node label — there is no separate
database per KB. This test seeds two ``LightRAG`` instances bound to different
workspaces in the SAME Neo4j database and asserts each instance sees only its
own entities, and that dropping one workspace's subgraph leaves the other
intact. This is the workspace-label isolation that KB-level graph queries and
KB hard delete both depend on, and which the file-based default backend
(``NetworkXStorage``, one file per workspace) cannot exercise.

Gated: marked ``integration`` (skipped unless ``--run-integration``) AND skips
with a clear reason when ``NEO4J_URI`` / ``NEO4J_USERNAME`` / ``NEO4J_PASSWORD``
are unset. Each run uses unique workspaces and drops them, so it is safe against
a shared Neo4j service.

Run with e.g.::

    NEO4J_URI=bolt://127.0.0.1:7687 NEO4J_USERNAME=neo4j NEO4J_PASSWORD=... \
        uv run pytest tests/pipeline/test_neo4j_kb_isolation.py --run-integration -q
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.lightrag import LightRAG
from lightrag.utils import EmbeddingFunc, Tokenizer

_NEO4J_ENV = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")

pytestmark = [pytest.mark.integration, pytest.mark.requires_db]


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _dummy_llm(*args, **kwargs) -> str:
    return "ok"


async def _build_neo4j_rag(tmp_path, workspace: str) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / workspace),
        workspace=workspace,
        graph_storage="Neo4JStorage",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("test-tokenizer", _SimpleTokenizerImpl()),
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    return rag


async def _seed_entity(rag: LightRAG, entity: str) -> None:
    created_at = int(datetime.now(timezone.utc).timestamp())
    await rag.chunk_entity_relation_graph.upsert_node(
        entity,
        {
            "entity_id": entity,
            "source_id": GRAPH_FIELD_SEP.join(["chunk-1"]),
            "description": "an entity",
            "entity_type": "test",
            "file_path": "a.txt",
            "created_at": created_at,
            "truncate": "",
        },
    )
    await rag.chunk_entity_relation_graph.index_done_callback()


@pytest.mark.asyncio
async def test_two_workspaces_share_db_but_isolate_graph(tmp_path):
    if not all(os.getenv(v) for v in _NEO4J_ENV):
        pytest.skip("Neo4j not configured (set NEO4J_URI/USERNAME/PASSWORD)")

    suffix = uuid4().hex[:10]
    ws_a = f"kbiso_a_{suffix}"
    ws_b = f"kbiso_b_{suffix}"
    entity_a = f"ENTITY-A-{suffix}"
    entity_b = f"ENTITY-B-{suffix}"

    rag_a = await _build_neo4j_rag(tmp_path, ws_a)
    rag_b = await _build_neo4j_rag(tmp_path, ws_b)
    try:
        await _seed_entity(rag_a, entity_a)
        await _seed_entity(rag_b, entity_b)

        # Each workspace sees its own entity...
        assert await rag_a.chunk_entity_relation_graph.get_node(entity_a) is not None
        assert await rag_b.chunk_entity_relation_graph.get_node(entity_b) is not None

        # ...but NOT the other workspace's entity, even though both live in the
        # same Neo4j database (label-based isolation).
        assert await rag_a.chunk_entity_relation_graph.get_node(entity_b) is None, (
            "workspace B's entity leaked into workspace A"
        )
        assert await rag_b.chunk_entity_relation_graph.get_node(entity_a) is None, (
            "workspace A's entity leaked into workspace B"
        )

        labels_a = await rag_a.chunk_entity_relation_graph.get_all_labels()
        labels_b = await rag_b.chunk_entity_relation_graph.get_all_labels()
        assert entity_a in labels_a and entity_b not in labels_a, labels_a
        assert entity_b in labels_b and entity_a not in labels_b, labels_b

        # Dropping one workspace's subgraph must not touch the other — this is
        # what KB hard delete relies on when KBs share a Neo4j database.
        await rag_a.chunk_entity_relation_graph.drop()
        assert await rag_a.chunk_entity_relation_graph.get_node(entity_a) is None
        assert await rag_b.chunk_entity_relation_graph.get_node(entity_b) is not None, (
            "dropping workspace A also cleared workspace B"
        )
    finally:
        for rag in (rag_a, rag_b):
            try:
                await rag.chunk_entity_relation_graph.drop()
            except Exception:  # noqa: BLE001 — teardown best-effort
                pass
            await rag.finalize_storages()
