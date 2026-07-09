"""Pin the LightRAG facade → utils_graph wiring for entity/relation deletes.

The API graph-curation delete endpoints go through
``LightRAG.adelete_by_entity`` / ``adelete_by_relation``. Regression: the
facade must forward the chunk-tracking storages (``entity_chunks`` /
``relation_chunks``) exactly like ``aedit_entity`` already does — otherwise
a graph-level delete leaves stale chunk-tracking rows behind and a later
document deletion resurrects the deleted entity via
``rebuild_knowledge_from_chunks``.
"""

from unittest.mock import AsyncMock

import pytest

from lightrag import LightRAG
from lightrag.base import DeletionResult

pytestmark = pytest.mark.offline


def _bare_rag() -> LightRAG:
    rag = LightRAG.__new__(LightRAG)
    rag.chunk_entity_relation_graph = object()
    rag.entities_vdb = object()
    rag.relationships_vdb = object()
    rag.entity_chunks = object()
    rag.relation_chunks = object()
    return rag


@pytest.mark.asyncio
async def test_adelete_by_entity_forwards_chunk_tracking_storages(monkeypatch):
    import lightrag.utils_graph as utils_graph

    spy = AsyncMock(
        return_value=DeletionResult(status="success", doc_id="X", message="")
    )
    monkeypatch.setattr(utils_graph, "adelete_by_entity", spy)
    rag = _bare_rag()

    result = await LightRAG.adelete_by_entity(rag, "X")

    assert result.status == "success"
    spy.assert_awaited_once_with(
        rag.chunk_entity_relation_graph,
        rag.entities_vdb,
        rag.relationships_vdb,
        "X",
        entity_chunks_storage=rag.entity_chunks,
        relation_chunks_storage=rag.relation_chunks,
    )


@pytest.mark.asyncio
async def test_adelete_by_relation_forwards_chunk_tracking_storage(monkeypatch):
    import lightrag.utils_graph as utils_graph

    spy = AsyncMock(
        return_value=DeletionResult(status="success", doc_id="X -> Y", message="")
    )
    monkeypatch.setattr(utils_graph, "adelete_by_relation", spy)
    rag = _bare_rag()

    result = await LightRAG.adelete_by_relation(rag, "X", "Y")

    assert result.status == "success"
    spy.assert_awaited_once_with(
        rag.chunk_entity_relation_graph,
        rag.relationships_vdb,
        "X",
        "Y",
        relation_chunks_storage=rag.relation_chunks,
    )
