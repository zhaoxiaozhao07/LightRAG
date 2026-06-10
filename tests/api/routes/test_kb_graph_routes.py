from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightrag.api.kb_service import KnowledgeBaseService
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry, LightRAGLike
from lightrag.base import DeletionResult
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_kb_routes = importlib.import_module("lightrag.api.routers.kb_routes")
_kb_graph_routes = importlib.import_module("lightrag.api.routers.kb_graph_routes")
sys.argv = _original_argv

create_kb_routes = _kb_routes.create_kb_routes
create_kb_graph_routes = _kb_graph_routes.create_kb_graph_routes

pytestmark = pytest.mark.offline

_API_KEY = "test-key"
_HEADERS = {"X-API-Key": _API_KEY}


class _FakeGraphStore:
    def __init__(self, labels: list[str]):
        self._labels = labels

    async def search_labels(self, query: str, limit: int) -> list[str]:
        matched = [label for label in self._labels if query.lower() in label.lower()]
        return matched[:limit]


class FakeGraphRAG:
    def __init__(self, workspace: str, labels: list[str] | None = None):
        self.workspace = workspace
        self._labels = labels if labels is not None else ["Alice", "Bob", "Acme", "Paris"]
        self.chunk_entity_relation_graph = _FakeGraphStore(self._labels)
        self.edit_entity_calls: list[tuple] = []
        self.relation_edit_calls: list[tuple] = []

    async def finalize_storages(self) -> None:
        return None

    async def aedit_entity(
        self,
        entity_name: str,
        updated_data: dict,
        allow_rename: bool = True,
        allow_merge: bool = False,
    ) -> dict:
        if entity_name not in self._labels:
            raise ValueError(f"Entity '{entity_name}' does not exist")
        self.edit_entity_calls.append(
            (entity_name, updated_data, allow_rename, allow_merge)
        )
        final_name = updated_data.get("entity_name", entity_name)
        return {
            "entity_name": final_name,
            "description": updated_data.get("description", "unchanged"),
            "operation_summary": {
                "merged": False,
                "merge_status": "not_attempted",
                "merge_error": None,
                "operation_status": "success",
                "target_entity": None,
                "final_entity": final_name,
                "renamed": final_name != entity_name,
            },
        }

    async def acreate_entity(self, entity_name: str, entity_data: dict) -> dict:
        if entity_name in self._labels:
            raise ValueError(f"Entity '{entity_name}' already exists")
        self._labels.append(entity_name)
        return {"entity_name": entity_name, **entity_data}

    async def amerge_entities(
        self,
        source_entities: list[str],
        target_entity: str,
        merge_strategy: dict | None = None,
        target_entity_data: dict | None = None,
    ) -> dict:
        missing = [name for name in source_entities if name not in self._labels]
        if missing:
            raise ValueError(f"Entities do not exist: {missing}")
        for name in source_entities:
            self._labels.remove(name)
        return {"entity_name": target_entity}

    async def aedit_relation(
        self, source_entity: str, target_entity: str, updated_data: dict
    ) -> dict:
        self.relation_edit_calls.append((source_entity, target_entity, updated_data))
        return {"src_id": source_entity, "tgt_id": target_entity, **updated_data}

    async def acreate_relation(
        self, source_entity: str, target_entity: str, relation_data: dict
    ) -> dict:
        return {"src_id": source_entity, "tgt_id": target_entity, **relation_data}

    async def adelete_by_entity(self, entity_name: str) -> DeletionResult:
        if entity_name not in self._labels:
            return DeletionResult(
                status="not_found",
                doc_id=entity_name,
                message=f"Entity '{entity_name}' not found",
                status_code=404,
                file_path="",
            )
        self._labels.remove(entity_name)
        return DeletionResult(
            status="success",
            doc_id=entity_name,
            message="deleted",
            status_code=200,
            file_path="",
        )

    async def adelete_by_relation(
        self, source_entity: str, target_entity: str
    ) -> DeletionResult:
        return DeletionResult(
            status="success",
            doc_id=f"{source_entity}->{target_entity}",
            message="deleted",
            status_code=200,
            file_path="",
        )

    async def get_graph_labels(self) -> list[str]:
        return list(self._labels)

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 3, max_nodes: int | None = None
    ) -> KnowledgeGraph:
        # Honor node_label: "*" => whole graph; otherwise just the matching node.
        if node_label == "*":
            selected = list(self._labels)
        else:
            selected = [label for label in self._labels if label == node_label]
        # Honor max_nodes: truncate and flag like the real backends do.
        is_truncated = False
        if max_nodes is not None and len(selected) > max_nodes:
            selected = selected[:max_nodes]
            is_truncated = True
        node_set = set(selected)
        all_edges = [
            KnowledgeGraphEdge(
                id="e1",
                type="KNOWS",
                source="Alice",
                target="Bob",
                properties={"weight": 1.0},
            ),
            KnowledgeGraphEdge(
                id="e2",
                type="WORKS_AT",
                source="Bob",
                target="Acme",
                properties={},
            ),
        ]
        nodes = [
            KnowledgeGraphNode(id=label, labels=[label], properties={})
            for label in selected
        ]
        # Whole-graph scan keeps all edges; subgraph keeps only edges whose
        # endpoints are in scope.
        if node_label == "*":
            edges = all_edges
        else:
            edges = [
                edge
                for edge in all_edges
                if edge.source in node_set and edge.target in node_set
            ]
        return KnowledgeGraph(nodes=nodes, edges=edges, is_truncated=is_truncated)


class GraphBuilderProbe:
    def __init__(self):
        self.instances: dict[str, FakeGraphRAG] = {}

    async def build(self, record) -> FakeGraphRAG:
        rag = FakeGraphRAG(record.workspace)
        self.instances[record.id] = rag
        return rag

    async def finalize(self, rag: LightRAGLike) -> None:
        return None


def _build_client(tmp_path: Path):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kb.json")
    probe = GraphBuilderProbe()
    registry = LightRAGInstanceRegistry(kb_service, probe.build, probe.finalize)
    app = FastAPI()
    app.include_router(create_kb_routes(kb_service, registry, api_key=_API_KEY))
    app.include_router(create_kb_graph_routes(registry, api_key=_API_KEY))
    return TestClient(app), probe


def _create_kb(client: TestClient, kb_id: str):
    response = client.post("/kbs", json={"id": kb_id, "name": kb_id}, headers=_HEADERS)
    assert response.status_code == 200


def test_kb_graph_status(tmp_path):
    client, _probe = _build_client(tmp_path)
    _create_kb(client, "kb_graph")
    response = client.get("/kbs/kb_graph/graph/status", headers=_HEADERS)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kb_id"] == "kb_graph"
    assert body["label_count"] == 4
    assert body["node_count"] == 4
    assert body["edge_count"] == 2
    assert body["is_truncated"] is False


def test_kb_graph_entities_pagination(tmp_path):
    client, _probe = _build_client(tmp_path)
    _create_kb(client, "kb_graph")
    response = client.get(
        "/kbs/kb_graph/graph/entities?limit=2&offset=0", headers=_HEADERS
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 4
    assert body["entities"] == ["Alice", "Bob"]

    page2 = client.get(
        "/kbs/kb_graph/graph/entities?limit=2&offset=2", headers=_HEADERS
    ).json()
    assert page2["entities"] == ["Acme", "Paris"]


def test_kb_graph_entities_search(tmp_path):
    client, _probe = _build_client(tmp_path)
    _create_kb(client, "kb_graph")
    response = client.get(
        "/kbs/kb_graph/graph/entities?q=a", headers=_HEADERS
    )
    assert response.status_code == 200, response.text
    body = response.json()
    # Fuzzy "a" matches Alice, Acme, Paris (case-insensitive).
    assert set(body["entities"]) == {"Alice", "Acme", "Paris"}


def test_kb_graph_relations(tmp_path):
    client, _probe = _build_client(tmp_path)
    _create_kb(client, "kb_graph")
    response = client.get("/kbs/kb_graph/graph/relations", headers=_HEADERS)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 2
    types = {rel["type"] for rel in body["relations"]}
    assert types == {"KNOWS", "WORKS_AT"}


def test_kb_subgraph(tmp_path):
    client, _probe = _build_client(tmp_path)
    _create_kb(client, "kb_graph")
    response = client.get(
        "/kbs/kb_graph/graph?label=*&max_nodes=10", headers=_HEADERS
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["nodes"]) == 4
    assert len(body["edges"]) == 2


def test_kb_graph_status_unknown_kb_404(tmp_path):
    client, _probe = _build_client(tmp_path)
    response = client.get("/kbs/kb_missing/graph/status", headers=_HEADERS)
    assert response.status_code == 404


def test_kb_graph_status_truncation_path(tmp_path, monkeypatch):
    """When the bounded scan hits the node cap, is_truncated=True and
    node_count is clamped to the cap (the load-bearing OOM guard)."""
    monkeypatch.setattr(_kb_graph_routes, "_MAX_GRAPH_STATUS_NODES", 2)
    client, _probe = _build_client(tmp_path)
    _create_kb(client, "kb_trunc")
    response = client.get("/kbs/kb_trunc/graph/status", headers=_HEADERS)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["is_truncated"] is True
    assert body["node_count"] == 2
    assert body["max_nodes_scanned"] == 2


def test_kb_subgraph_filters_by_label_and_max_nodes(tmp_path):
    """Subgraph honors a specific label (not '*') and max_nodes, returning only
    the in-scope node and its incident edges."""
    client, _probe = _build_client(tmp_path)
    _create_kb(client, "kb_sub")
    response = client.get(
        "/kbs/kb_sub/graph?label=Alice&max_nodes=10", headers=_HEADERS
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert [node["id"] for node in body["nodes"]] == ["Alice"]
    # Alice's only edge (Alice->Bob) drops out because Bob is not in scope.
    assert body["edges"] == []


def test_kb_graph_entities_search_total_is_accurate_under_pagination(tmp_path):
    """The q-search total must reflect ALL matches, not limit+offset, so a
    client can page through a filtered set correctly."""
    client, probe = _build_client(tmp_path)
    _create_kb(client, "kb_search_total")
    # Trigger a lazy build, then swap in an instance with many "a"-matching labels.
    client.get("/kbs/kb_search_total/graph/status", headers=_HEADERS)
    many = [f"alpha{i}" for i in range(10)]
    probe.instances["kb_search_total"]._labels = many
    probe.instances["kb_search_total"].chunk_entity_relation_graph = _FakeGraphStore(many)

    response = client.get(
        "/kbs/kb_search_total/graph/entities?q=alpha&limit=3&offset=0",
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    # total reflects all 10 matches even though only 3 are returned.
    assert body["total"] == 10
    assert len(body["entities"]) == 3


def test_kb_graph_entities_relations_subgraph_404_for_unknown_kb(tmp_path):
    """All read-only graph endpoints 404 for a missing KB (not just /status)."""
    client, _probe = _build_client(tmp_path)
    assert (
        client.get("/kbs/nope/graph/entities", headers=_HEADERS).status_code == 404
    )
    assert (
        client.get("/kbs/nope/graph/relations", headers=_HEADERS).status_code == 404
    )
    assert (
        client.get("/kbs/nope/graph?label=*", headers=_HEADERS).status_code == 404
    )


def test_kb_graph_write_endpoints_roundtrip(tmp_path):
    # The busy-guard reads shared pipeline state; bootstrap it like the real
    # server lifespan does so the guard takes its silent no-op path.
    initialize_share_data()
    try:
        _run_kb_graph_write_endpoints_roundtrip(tmp_path)
    finally:
        finalize_share_data()


def _run_kb_graph_write_endpoints_roundtrip(tmp_path):
    client, probe = _build_client(tmp_path)
    _create_kb(client, "kb_curate")

    created = client.post(
        "/kbs/kb_curate/graph/entity:create",
        json={
            "entity_name": "Tesla",
            "entity_data": {"description": "EV maker", "entity_type": "ORGANIZATION"},
        },
        headers=_HEADERS,
    )
    assert created.status_code == 200, created.text
    assert created.json()["data"]["entity_name"] == "Tesla"

    # Engine ValueError (duplicate entity) maps to 400.
    duplicate = client.post(
        "/kbs/kb_curate/graph/entity:create",
        json={"entity_name": "Tesla", "entity_data": {}},
        headers=_HEADERS,
    )
    assert duplicate.status_code == 400

    edited = client.post(
        "/kbs/kb_curate/graph/entity:edit",
        json={"entity_name": "Alice", "updated_data": {"description": "a person"}},
        headers=_HEADERS,
    )
    assert edited.status_code == 200, edited.text
    assert edited.json()["operation_summary"]["operation_status"] == "success"
    assert probe.instances["kb_curate"].edit_entity_calls[0][0] == "Alice"

    missing_edit = client.post(
        "/kbs/kb_curate/graph/entity:edit",
        json={"entity_name": "Ghost", "updated_data": {"description": "x"}},
        headers=_HEADERS,
    )
    assert missing_edit.status_code == 400

    relation = client.post(
        "/kbs/kb_curate/graph/relation:create",
        json={
            "source_entity": "Alice",
            "target_entity": "Tesla",
            "relation_data": {"description": "works at", "weight": 1.0},
        },
        headers=_HEADERS,
    )
    assert relation.status_code == 200, relation.text
    assert relation.json()["data"]["src_id"] == "Alice"

    relation_edit = client.post(
        "/kbs/kb_curate/graph/relation:edit",
        json={
            "source_entity": "Alice",
            "target_entity": "Tesla",
            "updated_data": {"weight": 2.0},
        },
        headers=_HEADERS,
    )
    assert relation_edit.status_code == 200, relation_edit.text
    assert probe.instances["kb_curate"].relation_edit_calls[0][2] == {"weight": 2.0}

    merged = client.post(
        "/kbs/kb_curate/graph/entities:merge",
        json={"source_entities": ["Bob"], "target_entity": "Alice"},
        headers=_HEADERS,
    )
    assert merged.status_code == 200, merged.text
    assert merged.json()["data"]["entity_name"] == "Alice"

    relation_deleted = client.post(
        "/kbs/kb_curate/graph/relation:delete",
        json={"source_entity": "Alice", "target_entity": "Tesla"},
        headers=_HEADERS,
    )
    assert relation_deleted.status_code == 200, relation_deleted.text
    assert relation_deleted.json()["status"] == "success"

    entity_deleted = client.post(
        "/kbs/kb_curate/graph/entity:delete",
        json={"entity_name": "Tesla"},
        headers=_HEADERS,
    )
    assert entity_deleted.status_code == 200, entity_deleted.text

    # Deleting again maps the engine's not_found result to 404.
    assert (
        client.post(
            "/kbs/kb_curate/graph/entity:delete",
            json={"entity_name": "Tesla"},
            headers=_HEADERS,
        ).status_code
        == 404
    )

    # Unknown KB is 404 before any engine work.
    assert (
        client.post(
            "/kbs/kb_ghost/graph/entity:create",
            json={"entity_name": "X", "entity_data": {}},
            headers=_HEADERS,
        ).status_code
        == 404
    )
