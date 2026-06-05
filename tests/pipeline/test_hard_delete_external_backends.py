"""Gated integration test: hard delete must purge EXTERNAL engine backends,
not just the on-disk ``working_dir`` folder.

Why this test exists
--------------------
``LightRAG.adrop_all_storages`` (driven by ``KBDeletionService`` →
``LightRAGInstanceRegistry.drop_kb_data`` during a KB hard delete) was added to
fix a real isolation bug: hard delete used to only ``rmtree`` the on-disk
``working_dir/<workspace>`` folder, which purges *file-based* backends but
leaves *external* backends (PostgreSQL / Milvus / Neo4j / Qdrant / Redis /
Mongo / OpenSearch) holding the KB's vectors / graph / chunks. That orphaned
data is then visible to a future KB that reuses the same ``workspace`` — an
isolation/correctness bug, not just cleanup hygiene.

The existing coverage does NOT touch that failure path:

* ``tests/pipeline/test_drop_all_storages.py`` runs the real engine but only on
  the in-memory **file-based** default backends (JsonKV / NanoVectorDB /
  NetworkX) — exactly the case where the bug does not manifest.
* ``tests/api/test_kb_hard_delete.py`` uses a ``FakeRAG`` whose
  ``adrop_all_storages`` is a stub that drops nothing.

This test closes the gap by exercising the drop against a **real external
backend** and asserting the workspace-reuse isolation property end to end:

    seed(rag1) → flush → rag2[same workspace] reads it back (proves the data
    lives in the shared/remote store, not just rag1's memory) → drop →
    rag3[same workspace] reads NOTHING (proves drop reached the remote backend).

If hard delete regressed to "remove the working_dir folder only", the external
backend (graph for Neo4j, vectors for Milvus) would still hold rag1's data and
the final ``rag3`` assertion would fail.

Gating
------
* The ``file`` parametrization always runs (offline): it validates the test's
  own seed → flush → reuse → drop → reuse logic on the default file backends,
  so the assertions are never silently dead.
* ``neo4j-graph`` / ``milvus-vector`` are marked ``integration`` (skipped by
  default; enable with ``--run-integration``) AND skip with a clear reason when
  the backend env vars are unset. Each run uses a unique ``workspace`` and
  drops it, so it is safe against a shared service.

Run live coverage with e.g.::

    NEO4J_URI=bolt://127.0.0.1:7687 NEO4J_USERNAME=neo4j NEO4J_PASSWORD=... \
    MILVUS_URI=http://127.0.0.1:19530 MILVUS_DB_NAME=lightrag \
        uv run pytest tests/pipeline/test_hard_delete_external_backends.py \
        --run-integration -q
"""

from __future__ import annotations

import asyncio
import os
import shutil
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.lightrag import LightRAG
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id

_NEO4J_ENV = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")
_MILVUS_ENV = ("MILVUS_URI", "MILVUS_DB_NAME")


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _dummy_llm(*args, **kwargs) -> str:
    return "ok"


def _make_rag(working_dir, workspace: str, backend: str) -> LightRAG:
    """Build a LightRAG bound to ``workspace`` with the parametrized backend.

    Only the storage under test is switched to the external backend; the rest
    stay file-based. ``adrop_all_storages`` still drops every storage, so the
    workspace-reuse assertion holds regardless of which one is external.
    """
    kwargs = dict(
        working_dir=str(working_dir),
        workspace=workspace,
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("test-tokenizer", _SimpleTokenizerImpl()),
        max_parallel_insert=1,
    )
    if backend == "neo4j-graph":
        kwargs["graph_storage"] = "Neo4JStorage"
    elif backend == "milvus-vector":
        kwargs["vector_storage"] = "MilvusVectorDBStorage"
    # "file": all defaults (NetworkX / NanoVectorDB / JsonKV / JsonDocStatus)
    return LightRAG(**kwargs)


def _iter_storages(rag: LightRAG):
    for storage in (
        rag.full_docs,
        rag.text_chunks,
        rag.chunks_vdb,
        rag.entities_vdb,
        rag.relationships_vdb,
        rag.chunk_entity_relation_graph,
        rag.doc_status,
    ):
        if storage is not None:
            yield storage


def _ids() -> dict[str, str]:
    entity = "ENTITY-1"
    return {
        "doc": "doc-1",
        "chunk": "chunk-1",
        "entity": entity,
        "ent_vid": compute_mdhash_id(entity, prefix="ent-"),
    }


async def _seed(rag: LightRAG, ids: dict[str, str]) -> None:
    created_at = int(datetime.now(timezone.utc).timestamp())
    chunk_payload = {
        ids["chunk"]: {
            "content": "hi chunk",
            "file_path": "a.txt",
            "full_doc_id": ids["doc"],
        }
    }
    await rag.full_docs.upsert({ids["doc"]: {"content": "hi", "file_path": "a.txt"}})
    await rag.text_chunks.upsert(chunk_payload)
    await rag.chunks_vdb.upsert(chunk_payload)
    await rag.chunk_entity_relation_graph.upsert_node(
        ids["entity"],
        {
            "entity_id": ids["entity"],
            "source_id": GRAPH_FIELD_SEP.join([ids["chunk"]]),
            "description": "an entity",
            "entity_type": "test",
            "file_path": "a.txt",
            "created_at": created_at,
            "truncate": "",
        },
    )
    await rag.entities_vdb.upsert(
        {
            ids["ent_vid"]: {
                "content": f"{ids['entity']}\nan entity",
                "entity_name": ids["entity"],
                "source_id": ids["chunk"],
                "description": "an entity",
                "entity_type": "test",
                "file_path": "a.txt",
            }
        }
    )
    # Commit so a fresh instance bound to the same workspace can read it back
    # (file backends persist on this callback; external backends are no-ops).
    for storage in _iter_storages(rag):
        await storage.index_done_callback()


async def _read_presence(rag: LightRAG, ids: dict[str, str]) -> dict[str, bool]:
    """Return which seeded rows are visible. Used both to confirm presence
    (after seed) and absence (after drop)."""
    return {
        "doc": await rag.full_docs.get_by_id(ids["doc"]) is not None,
        "chunk": await rag.text_chunks.get_by_id(ids["chunk"]) is not None,
        "graph_node": await rag.chunk_entity_relation_graph.get_node(ids["entity"])
        is not None,
        "entity_vector": await rag.entities_vdb.get_by_id(ids["ent_vid"]) is not None,
    }


async def _poll_present(
    rag: LightRAG,
    ids: dict[str, str],
    key_field: str,
    *,
    attempts: int = 30,
    delay: float = 0.5,
) -> dict[str, bool]:
    """Poll one instance until the seeded ``key_field`` row becomes visible.

    File and Neo4j backends are visible on the first read. Milvus uses bounded
    read consistency, so a freshly-opened client (a new session) may not see a
    just-committed vector for a few seconds. Polling the SAME instance lets the
    staleness window elapse, making the pre-drop presence check reliable. The
    post-drop absence check needs no polling: dropping a Milvus collection is an
    immediate synchronous op, so a workspace-reusing instance binds to a freshly
    recreated empty collection (and if the drop had silently NOT happened, the
    data — already past its staleness window here — would still be visible, so
    the absence assertion still catches a regressed drop).
    """
    present = await _read_presence(rag, ids)
    for _ in range(attempts):
        if present[key_field]:
            return present
        await asyncio.sleep(delay)
        present = await _read_presence(rag, ids)
    return present


async def _finalize_quietly(rag: LightRAG) -> None:
    try:
        await rag.finalize_storages()
    except Exception:  # noqa: BLE001 — teardown best-effort
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("file", marks=pytest.mark.offline),
        pytest.param(
            "neo4j-graph",
            marks=[pytest.mark.integration, pytest.mark.requires_db],
        ),
        pytest.param(
            "milvus-vector",
            marks=[pytest.mark.integration, pytest.mark.requires_db],
        ),
    ],
)
async def test_hard_delete_clears_backend_and_workspace_reuse_reads_nothing(
    backend: str, tmp_path
):
    if backend == "neo4j-graph" and not all(os.getenv(v) for v in _NEO4J_ENV):
        pytest.skip("Neo4j not configured (set NEO4J_URI/USERNAME/PASSWORD)")
    if backend == "milvus-vector" and not all(os.getenv(v) for v in _MILVUS_ENV):
        pytest.skip("Milvus not configured (set MILVUS_URI/MILVUS_DB_NAME)")

    # Unique workspace so a shared external service is never polluted across runs.
    workspace = f"hdext_{uuid4().hex[:10]}"
    working_dir = tmp_path / "hd_ext"
    ids = _ids()
    # The storage whose external-backend purge we most care about for this param.
    key_field = "graph_node" if backend == "neo4j-graph" else "entity_vector"

    # 1) Seed a KB's data and commit it to the (possibly remote) workspace store.
    rag1 = _make_rag(working_dir, workspace, backend)
    await rag1.initialize_storages()
    try:
        await _seed(rag1, ids)
        # Poll: Milvus flush clears the read-your-writes buffer, so even the
        # same instance reads committed vectors via a bounded-consistency query
        # that may lag a few seconds; file/Neo4j read on the first try.
        present_same = await _poll_present(rag1, ids, key_field)
        assert present_same[key_field] is True, present_same
    finally:
        await _finalize_quietly(rag1)

    # 2) A FRESH instance bound to the SAME workspace must read the seeded data
    #    back. This proves the data lives in the shared/remote workspace store
    #    (not just rag1's process memory) — otherwise the final "reads nothing"
    #    assertion would pass vacuously.
    rag2 = _make_rag(working_dir, workspace, backend)
    await rag2.initialize_storages()
    try:
        before = await _poll_present(rag2, ids, key_field)
        assert before[key_field] is True, (
            f"seeded data not visible to a second instance on backend "
            f"'{backend}' before drop: {before}"
        )
    finally:
        await _finalize_quietly(rag2)

    # 3) Drop ALL engine storages for the workspace — this is what hard delete
    #    runs via registry.drop_kb_data → adrop_all_storages. For external
    #    backends this must reach the remote service, not just local files.
    rag_drop = _make_rag(working_dir, workspace, backend)
    await rag_drop.initialize_storages()
    try:
        summary = await rag_drop.adrop_all_storages()
        assert summary["failed"] == 0, summary
        assert summary["dropped"] >= 1, summary
    finally:
        await _finalize_quietly(rag_drop)

    # Remove the on-disk workspace folder too (mirrors KBDeletionService step 4).
    # For external backends this alone would NOT purge remote data — the drop
    # above is what does.
    workspace_dir = working_dir / workspace
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir, ignore_errors=True)

    # 4) Reuse the SAME workspace with another fresh instance. It must read
    #    NOTHING. If the external backend still held rag1's data (the bug),
    #    key_field would come back True here.
    rag3 = _make_rag(working_dir, workspace, backend)
    await rag3.initialize_storages()
    try:
        after = await _read_presence(rag3, ids)
        assert after[key_field] is False, (
            f"backend '{backend}' still exposes dropped data to a "
            f"workspace-reusing instance — hard delete left orphaned remote "
            f"data: {after}"
        )
        if backend == "file":
            # All-file-based: drop + working-dir removal clears the entire KB
            # footprint for a reused workspace. External-backend params only
            # assert the external storage itself is purged (key_field above) —
            # the remaining file-based KV/graph cleanup across in-process
            # instances (shared_storage cache) is covered by
            # tests/pipeline/test_drop_all_storages.py.
            assert not any(after.values()), after
    finally:
        await _finalize_quietly(rag3)
        # Best-effort cleanup in case an assertion above fired before the drop.
        cleanup = _make_rag(working_dir, workspace, backend)
        try:
            await cleanup.initialize_storages()
            await cleanup.adrop_all_storages()
        except Exception:  # noqa: BLE001
            pass
        finally:
            await _finalize_quietly(cleanup)
