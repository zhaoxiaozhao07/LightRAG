"""Pin the vdb payload contract for manual graph-curation writes.

Milvus collections declare ``file_path`` in the schema; collections created
before the schema switched to ``nullable=True`` reject any insert omitting
the field ("Insert missed an field `file_path` to collection without set
nullable==true or set default_value"). Because Milvus upserts are buffered
and only flushed by a later ``index_done_callback``, one bad payload from an
edit also breaks the NEXT curation operation's flush — deletes included, so
the symptom shows up on every entity/relation curation call. Every curation
upsert must therefore always carry ``file_path``.

``acreate_entity`` / ``acreate_relation`` / ``amerge_entities`` already set
it; these tests pin the edit paths that historically omitted it.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightrag import utils_graph

pytestmark = pytest.mark.offline


@asynccontextmanager
async def _noop_lock():
    yield


def _noop_keyed_lock(keys, namespace="default", enable_logging=False):
    return _noop_lock()


def _make_graph_mock(
    node_data: dict,
    edge_data: dict,
    *,
    existing_entity: str = "X",
    edges_for_entity: list[tuple[str, str]] | None = None,
):
    graph = MagicMock()
    graph.has_node = AsyncMock(side_effect=lambda name: name == existing_entity)
    graph.has_edge = AsyncMock(return_value=True)
    graph.get_node = AsyncMock(return_value=dict(node_data))
    graph.get_edge = AsyncMock(return_value=dict(edge_data))
    graph.get_node_edges = AsyncMock(return_value=edges_for_entity or [])
    graph.upsert_node = AsyncMock(return_value=None)
    graph.upsert_edge = AsyncMock(return_value=None)
    graph.delete_node = AsyncMock(return_value=None)
    graph.index_done_callback = AsyncMock(return_value=None)
    return graph


def _make_vdb_mock():
    vdb = MagicMock()
    vdb.global_config = {"workspace": "ws1"}
    vdb.upsert = AsyncMock(return_value=None)
    vdb.delete = AsyncMock(return_value=None)
    vdb.get_by_id = AsyncMock(return_value=None)
    vdb.index_done_callback = AsyncMock(return_value=None)
    return vdb


def _upserted_records(vdb) -> list[dict]:
    records: list[dict] = []
    for call in vdb.upsert.await_args_list:
        payload = call.args[0] if call.args else call.kwargs["data"]
        records.extend(payload.values())
    return records


_NODE = {
    "entity_id": "X",
    "description": "old description",
    "entity_type": "PERSON",
    "source_id": "chunk-1",
    "file_path": "doc-a.txt",
}

_EDGE = {
    "weight": 1.0,
    "description": "rel",
    "keywords": "k",
    "source_id": "chunk-1",
    "file_path": "doc-b.txt",
    "created_at": 0,
}


@pytest.mark.asyncio
async def test_aedit_entity_update_payload_carries_file_path():
    graph = _make_graph_mock(_NODE, _EDGE)
    entities_vdb = _make_vdb_mock()
    relationships_vdb = _make_vdb_mock()

    with patch.object(utils_graph, "get_storage_keyed_lock", _noop_keyed_lock):
        await utils_graph.aedit_entity(
            chunk_entity_relation_graph=graph,
            entities_vdb=entities_vdb,
            relationships_vdb=relationships_vdb,
            entity_name="X",
            updated_data={"description": "new description"},
            allow_rename=False,
        )

    records = _upserted_records(entities_vdb)
    assert records, "entity edit must upsert the entity vector"
    assert all(record["file_path"] == "doc-a.txt" for record in records)


@pytest.mark.asyncio
async def test_aedit_entity_falls_back_to_unknown_source_without_file_path():
    node = {key: value for key, value in _NODE.items() if key != "file_path"}
    graph = _make_graph_mock(node, _EDGE)
    entities_vdb = _make_vdb_mock()
    relationships_vdb = _make_vdb_mock()

    with patch.object(utils_graph, "get_storage_keyed_lock", _noop_keyed_lock):
        await utils_graph.aedit_entity(
            chunk_entity_relation_graph=graph,
            entities_vdb=entities_vdb,
            relationships_vdb=relationships_vdb,
            entity_name="X",
            updated_data={"description": "new description"},
            allow_rename=False,
        )

    records = _upserted_records(entities_vdb)
    assert records
    assert all(record["file_path"] == "unknown_source" for record in records)


@pytest.mark.asyncio
async def test_aedit_entity_rename_relation_reupserts_carry_file_path():
    graph = _make_graph_mock(_NODE, _EDGE, edges_for_entity=[("X", "Z")])
    entities_vdb = _make_vdb_mock()
    relationships_vdb = _make_vdb_mock()

    with patch.object(utils_graph, "get_storage_keyed_lock", _noop_keyed_lock):
        await utils_graph.aedit_entity(
            chunk_entity_relation_graph=graph,
            entities_vdb=entities_vdb,
            relationships_vdb=relationships_vdb,
            entity_name="X",
            updated_data={"entity_name": "Y"},
            allow_rename=True,
        )

    entity_records = _upserted_records(entities_vdb)
    relation_records = _upserted_records(relationships_vdb)
    assert entity_records and relation_records
    assert all(record["file_path"] == "doc-a.txt" for record in entity_records)
    assert all(record["file_path"] == "doc-b.txt" for record in relation_records)


@pytest.mark.asyncio
async def test_aedit_relation_payload_carries_file_path():
    graph = _make_graph_mock(_NODE, _EDGE)
    entities_vdb = _make_vdb_mock()
    relationships_vdb = _make_vdb_mock()

    with patch.object(utils_graph, "get_storage_keyed_lock", _noop_keyed_lock):
        await utils_graph.aedit_relation(
            chunk_entity_relation_graph=graph,
            entities_vdb=entities_vdb,
            relationships_vdb=relationships_vdb,
            source_entity="X",
            target_entity="Z",
            updated_data={"description": "updated rel"},
        )

    records = _upserted_records(relationships_vdb)
    assert records, "relation edit must upsert the relation vector"
    assert all(record["file_path"] == "doc-b.txt" for record in records)


@pytest.mark.asyncio
async def test_aedit_relation_falls_back_to_unknown_source_without_file_path():
    edge = {key: value for key, value in _EDGE.items() if key != "file_path"}
    graph = _make_graph_mock(_NODE, edge)
    entities_vdb = _make_vdb_mock()
    relationships_vdb = _make_vdb_mock()

    with patch.object(utils_graph, "get_storage_keyed_lock", _noop_keyed_lock):
        await utils_graph.aedit_relation(
            chunk_entity_relation_graph=graph,
            entities_vdb=entities_vdb,
            relationships_vdb=relationships_vdb,
            source_entity="X",
            target_entity="Z",
            updated_data={"description": "updated rel"},
        )

    records = _upserted_records(relationships_vdb)
    assert records
    assert all(record["file_path"] == "unknown_source" for record in records)
