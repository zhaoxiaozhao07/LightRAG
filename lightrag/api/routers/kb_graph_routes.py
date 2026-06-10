"""KB-scoped knowledge-graph inspection and curation endpoints.

Mirrors the global ``/graph/*`` routes but resolves the per-KB LightRAG
instance through :class:`LightRAGInstanceRegistry`, so graph stats / labels /
subgraphs are workspace-isolated to a single knowledge base.

Read endpoints (``kb_viewer``+ in enterprise mode):

- ``GET /kbs/{kb_id}/graph/status``    — node/edge/label counts (bounded scan)
- ``GET /kbs/{kb_id}/graph/entities``  — paginated entity labels
- ``GET /kbs/{kb_id}/graph/relations`` — relation (edge) listing
- ``GET /kbs/{kb_id}/graph``           — connected subgraph for a label

Write endpoints (``kb_admin``+ in enterprise mode; the enterprise middleware
escalates non-GET ``/graph`` paths). These wrap the engine's curation methods
with the KB workspace boundary — with global ``/graph/*`` routes disabled in
enterprise mode they are the only graph-surgery entry point. Manual edits
live in engine storage and are overwritten when the source documents are
force-reindexed / rebuilt:

- ``POST /kbs/{kb_id}/graph/entity:edit``     — update / rename / auto-merge
- ``POST /kbs/{kb_id}/graph/entity:create``   — create a standalone entity
- ``POST /kbs/{kb_id}/graph/entity:delete``   — delete entity + relations
- ``POST /kbs/{kb_id}/graph/entities:merge``  — merge duplicates into one
- ``POST /kbs/{kb_id}/graph/relation:edit``   — update relation properties
- ``POST /kbs/{kb_id}/graph/relation:create`` — create relation between entities
- ``POST /kbs/{kb_id}/graph/relation:delete`` — delete one relation
"""

from __future__ import annotations

from typing import Any, Optional, cast

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from lightrag.api.enterprise_auth import append_enterprise_audit_event
from lightrag.api.kb_service import KnowledgeBaseNotFoundError
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.base import DeletionResult
from lightrag.utils import logger

from .document_routes import check_pipeline_busy_or_raise

_MAX_GRAPH_STATUS_NODES = 100_000


class KBEntityEditRequest(BaseModel):
    entity_name: str = Field(min_length=1)
    updated_data: dict[str, Any]
    allow_rename: bool = False
    allow_merge: bool = False


class KBEntityCreateRequest(BaseModel):
    entity_name: str = Field(min_length=1)
    entity_data: dict[str, Any]


class KBEntityDeleteRequest(BaseModel):
    entity_name: str = Field(min_length=1)


class KBEntitiesMergeRequest(BaseModel):
    source_entities: list[str] = Field(min_length=1)
    target_entity: str = Field(min_length=1)


class KBRelationEditRequest(BaseModel):
    source_entity: str = Field(min_length=1)
    target_entity: str = Field(min_length=1)
    updated_data: dict[str, Any]


class KBRelationCreateRequest(BaseModel):
    source_entity: str = Field(min_length=1)
    target_entity: str = Field(min_length=1)
    relation_data: dict[str, Any]


class KBRelationDeleteRequest(BaseModel):
    source_entity: str = Field(min_length=1)
    target_entity: str = Field(min_length=1)


def create_kb_graph_routes(
    registry: LightRAGInstanceRegistry,
    api_key: Optional[str] = None,
):
    router = APIRouter(prefix="/kbs", tags=["knowledge-base-graph"])
    combined_auth = get_combined_auth_dependency(api_key)

    async def _resolve_rag(kb_id: str) -> Any:
        try:
            return cast(Any, await registry.get(kb_id))
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    async def _audit_graph_write(
        request: Request, event_type: str, kb_id: str, metadata: dict[str, Any]
    ) -> None:
        await append_enterprise_audit_event(
            request,
            event_type,
            target_type="kb",
            target_id=kb_id,
            metadata=metadata,
        )

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
                # Fetch a bounded full match set so ``total`` reflects ALL
                # matching labels, not just ``limit + offset`` (otherwise paging
                # through a large filtered set reports a truncated total and the
                # client cannot tell there are more pages).
                labels = await rag.chunk_entity_relation_graph.search_labels(
                    q, _MAX_GRAPH_STATUS_NODES
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

    @router.post(
        "/{kb_id}/graph/entity:edit",
        dependencies=[Depends(combined_auth)],
        summary="Update (and optionally rename / auto-merge) one entity in a KB graph",
    )
    async def kb_graph_entity_edit(
        kb_id: str, request: Request, body: KBEntityEditRequest
    ):
        rag = await _resolve_rag(kb_id)
        try:
            await check_pipeline_busy_or_raise(rag)
            result = await rag.aedit_entity(
                entity_name=body.entity_name,
                updated_data=body.updated_data,
                allow_rename=body.allow_rename,
                allow_merge=body.allow_merge,
            )
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("KB graph entity edit failed for '%s': %s", kb_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        operation_summary = result.get("operation_summary") if isinstance(result, dict) else None
        entity_data = dict(result) if isinstance(result, dict) else {"result": result}
        entity_data.pop("operation_summary", None)
        await _audit_graph_write(
            request,
            "kb_graph_entity_edited",
            kb_id,
            {
                "operation": "graph_entity_edit",
                "entity_name": body.entity_name,
                "renamed": bool(
                    body.updated_data.get("entity_name", body.entity_name)
                    != body.entity_name
                ),
            },
        )
        return {
            "kb_id": kb_id,
            "status": "success",
            "data": entity_data,
            "operation_summary": operation_summary,
        }

    @router.post(
        "/{kb_id}/graph/entity:create",
        dependencies=[Depends(combined_auth)],
        summary="Create a new entity in a KB graph",
    )
    async def kb_graph_entity_create(
        kb_id: str, request: Request, body: KBEntityCreateRequest
    ):
        rag = await _resolve_rag(kb_id)
        try:
            await check_pipeline_busy_or_raise(rag)
            result = await rag.acreate_entity(
                entity_name=body.entity_name,
                entity_data=body.entity_data,
            )
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("KB graph entity create failed for '%s': %s", kb_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        await _audit_graph_write(
            request,
            "kb_graph_entity_created",
            kb_id,
            {"operation": "graph_entity_create", "entity_name": body.entity_name},
        )
        return {"kb_id": kb_id, "status": "success", "data": result}

    @router.post(
        "/{kb_id}/graph/entity:delete",
        response_model=DeletionResult,
        dependencies=[Depends(combined_auth)],
        summary="Delete one entity (and its relations) from a KB graph",
    )
    async def kb_graph_entity_delete(
        kb_id: str, request: Request, body: KBEntityDeleteRequest
    ):
        rag = await _resolve_rag(kb_id)
        try:
            await check_pipeline_busy_or_raise(rag)
            result = await rag.adelete_by_entity(entity_name=body.entity_name)
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("KB graph entity delete failed for '%s': %s", kb_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if result.status == "not_found":
            raise HTTPException(status_code=404, detail=result.message)
        if result.status == "fail":
            raise HTTPException(status_code=500, detail=result.message)
        result.doc_id = ""
        await _audit_graph_write(
            request,
            "kb_graph_entity_deleted",
            kb_id,
            {"operation": "graph_entity_delete", "entity_name": body.entity_name},
        )
        return result

    @router.post(
        "/{kb_id}/graph/entities:merge",
        dependencies=[Depends(combined_auth)],
        summary="Merge duplicate entities into one target entity in a KB graph",
    )
    async def kb_graph_entities_merge(
        kb_id: str, request: Request, body: KBEntitiesMergeRequest
    ):
        rag = await _resolve_rag(kb_id)
        try:
            await check_pipeline_busy_or_raise(rag)
            result = await rag.amerge_entities(
                source_entities=body.source_entities,
                target_entity=body.target_entity,
            )
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("KB graph entities merge failed for '%s': %s", kb_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        await _audit_graph_write(
            request,
            "kb_graph_entities_merged",
            kb_id,
            {
                "operation": "graph_entities_merge",
                "source_count": len(body.source_entities),
                "target_entity": body.target_entity,
            },
        )
        return {"kb_id": kb_id, "status": "success", "data": result}

    @router.post(
        "/{kb_id}/graph/relation:edit",
        dependencies=[Depends(combined_auth)],
        summary="Update one relation's properties in a KB graph",
    )
    async def kb_graph_relation_edit(
        kb_id: str, request: Request, body: KBRelationEditRequest
    ):
        rag = await _resolve_rag(kb_id)
        try:
            await check_pipeline_busy_or_raise(rag)
            result = await rag.aedit_relation(
                source_entity=body.source_entity,
                target_entity=body.target_entity,
                updated_data=body.updated_data,
            )
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("KB graph relation edit failed for '%s': %s", kb_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        await _audit_graph_write(
            request,
            "kb_graph_relation_edited",
            kb_id,
            {
                "operation": "graph_relation_edit",
                "source_entity": body.source_entity,
                "target_entity": body.target_entity,
            },
        )
        return {"kb_id": kb_id, "status": "success", "data": result}

    @router.post(
        "/{kb_id}/graph/relation:create",
        dependencies=[Depends(combined_auth)],
        summary="Create a relation between two existing entities in a KB graph",
    )
    async def kb_graph_relation_create(
        kb_id: str, request: Request, body: KBRelationCreateRequest
    ):
        rag = await _resolve_rag(kb_id)
        try:
            await check_pipeline_busy_or_raise(rag)
            result = await rag.acreate_relation(
                source_entity=body.source_entity,
                target_entity=body.target_entity,
                relation_data=body.relation_data,
            )
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("KB graph relation create failed for '%s': %s", kb_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        await _audit_graph_write(
            request,
            "kb_graph_relation_created",
            kb_id,
            {
                "operation": "graph_relation_create",
                "source_entity": body.source_entity,
                "target_entity": body.target_entity,
            },
        )
        return {"kb_id": kb_id, "status": "success", "data": result}

    @router.post(
        "/{kb_id}/graph/relation:delete",
        response_model=DeletionResult,
        dependencies=[Depends(combined_auth)],
        summary="Delete one relation from a KB graph",
    )
    async def kb_graph_relation_delete(
        kb_id: str, request: Request, body: KBRelationDeleteRequest
    ):
        rag = await _resolve_rag(kb_id)
        try:
            await check_pipeline_busy_or_raise(rag)
            result = await rag.adelete_by_relation(
                source_entity=body.source_entity,
                target_entity=body.target_entity,
            )
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("KB graph relation delete failed for '%s': %s", kb_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if result.status == "not_found":
            raise HTTPException(status_code=404, detail=result.message)
        if result.status == "fail":
            raise HTTPException(status_code=500, detail=result.message)
        result.doc_id = ""
        await _audit_graph_write(
            request,
            "kb_graph_relation_deleted",
            kb_id,
            {
                "operation": "graph_relation_delete",
                "source_entity": body.source_entity,
                "target_entity": body.target_entity,
            },
        )
        return result

    return router
