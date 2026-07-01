from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from lightrag.api.agent_query_service import AgentQueryRequest, AgentQueryService
from lightrag.api.document_lifecycle_service import DocumentLifecycleService
from lightrag.api.kb_service import KnowledgeBaseService
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry
from lightrag.api.query_tool_service import QueryToolService
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.utils import logger


def create_agent_routes(
    *,
    kb_service: KnowledgeBaseService,
    document_service: DocumentLifecycleService,
    registry: LightRAGInstanceRegistry,
    api_key: Optional[str] = None,
) -> APIRouter:
    router = APIRouter(prefix="/agent", tags=["agent-query"])
    combined_auth = get_combined_auth_dependency(api_key)
    service = AgentQueryService(
        kb_service=kb_service,
        query_tool_service=QueryToolService(document_service, registry),
    )

    @router.post(
        "/query",
        dependencies=[Depends(combined_auth)],
        summary="Run a server-side Agent query over authorized knowledge bases",
    )
    async def agent_query(body: AgentQueryRequest, request: Request):
        try:
            result = await service.run(request=request, body=body, stream=False)
            return {
                "status": result.status,
                "session_id": result.session_id,
                "answer": result.answer,
                "clarification_question": result.clarification_question,
                "references": result.references,
                "steps_summary": result.steps_summary,
                "metadata": result.metadata,
            }
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Agent query failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post(
        "/query/stream",
        dependencies=[Depends(combined_auth)],
        summary="Run a server-side Agent query as NDJSON events",
    )
    async def agent_query_stream(body: AgentQueryRequest, request: Request):
        return StreamingResponse(
            service.stream_events(request=request, body=body),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "application/x-ndjson",
                "X-Accel-Buffering": "no",
            },
        )

    return router
