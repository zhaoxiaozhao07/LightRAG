"""KB-scoped knowledge-graph inspection endpoints.

Mirrors the global ``/graph/*`` routes but resolves the per-KB LightRAG
instance through :class:`LightRAGInstanceRegistry`, so graph stats / labels /
subgraphs are workspace-isolated to a single knowledge base.

Endpoints:

- ``GET /kbs/{kb_id}/graph/status``    — node/edge/label counts (bounded scan)
- ``GET /kbs/{kb_id}/graph/entities``  — paginated entity labels
- ``GET /kbs/{kb_id}/graph/relations`` — relation (edge) listing
- ``GET /kbs/{kb_id}/graph``           — connected subgraph for a label
"""

from __future__ import annotations

from typing import Any, Optional, cast

from fastapi import APIRouter, Depends, HTTPException, Query

from lightrag.api.kb_service import KnowledgeBaseNotFoundError
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.utils import logger

_MAX_GRAPH_STATUS_NODES = 100_000


def create_kb_graph_routes(
    registry: LightRAGInstanceRegistry,
    api_key: Optional[str] = None,
):
    router = APIRouter(prefix="/kbs", tags=["knowledge-base-graph"])
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get(
        "/{kb_id}/graph/status",
        dependencies=[Depends(combined_auth)],
        summary="Summary statistics for a KB's knowledge graph",
    )
    async def kb_graph_status(kb_id: str):
        try:
            rag = cast(Any, await registry.get(kb_id))
            labels = await rag.get_graph_labels()
            # Bounded full-graph scan via the "*" wildcard so very large
            # graphs cannot OOM the status endpoint; is_truncated signals the
            # cap was hit.
            graph = await rag.get_knowledge_graph(
                node_label="*",
                max_depth=1,
                max_nodes=_MAX_GRAPH_STATUS_NODES,
            )
            return {
                "kb_id": kb_id,
                "label_count": len(labels),
                "node_count": len(graph.nodes),
                "edge_count": len(graph.edges),
                "is_truncated": bool(graph.is_truncated),
                "max_nodes_scanned": _MAX_GRAPH_STATUS_NODES,
            }
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("KB graph status failed for '%s': %s", kb_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.get(
        "/{kb_id}/graph/entities",
        dependencies=[Depends(combined_auth)],
        summary="Paginated list of entity labels in a KB graph",
    )
    async def kb_graph_entities(
        kb_id: str,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        q: Optional[str] = Query(
            None, description="Optional fuzzy label search (case-insensitive)"
        ),
    ):
        try:
            rag = cast(Any, await registry.get(kb_id))
            if q:
                labels = await rag.chunk_entity_relation_graph.search_labels(
                    q, limit + offset
                )
            else:
                labels = await rag.get_graph_labels()
            total = len(labels)
            page = labels[offset : offset + limit]
            return {
                "kb_id": kb_id,
                "total": total,
                "limit": limit,
                "offset": offset,
                "entities": page,
            }
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("KB graph entities failed for '%s': %s", kb_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.get(
        "/{kb_id}/graph/relations",
        dependencies=[Depends(combined_auth)],
        summary="List relations (edges) in a KB graph",
    )
    async def kb_graph_relations(
        kb_id: str,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ):
        try:
            rag = cast(Any, await registry.get(kb_id))
            graph = await rag.get_knowledge_graph(
                node_label="*",
                max_depth=1,
                max_nodes=_MAX_GRAPH_STATUS_NODES,
            )
            edges = graph.edges
            total = len(edges)
            page = edges[offset : offset + limit]
            return {
                "kb_id": kb_id,
                "total": total,
                "limit": limit,
                "offset": offset,
                "is_truncated": bool(graph.is_truncated),
                "relations": [
                    {
                        "id": edge.id,
                        "type": edge.type,
                        "source": edge.source,
                        "target": edge.target,
                        "properties": edge.properties,
                    }
                    for edge in page
                ],
            }
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("KB graph relations failed for '%s': %s", kb_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.get(
        "/{kb_id}/graph",
        dependencies=[Depends(combined_auth)],
        summary="Connected subgraph for a label within a KB",
    )
    async def kb_subgraph(
        kb_id: str,
        label: str = Query(..., description="Starting node label; '*' for whole graph"),
        max_depth: int = Query(3, ge=1),
        max_nodes: int = Query(1000, ge=1),
    ):
        try:
            rag = cast(Any, await registry.get(kb_id))
            return await rag.get_knowledge_graph(
                node_label=label,
                max_depth=max_depth,
                max_nodes=max_nodes,
            )
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("KB subgraph failed for '%s': %s", kb_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return router
