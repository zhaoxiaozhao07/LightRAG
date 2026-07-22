from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from lightrag.api.agent_query_service import AgentQueryRequest, AgentQueryService
from lightrag.api.chat_memory_routing import authorize_memory_context
from lightrag.api.chat_memory_service import (
    CHAT_MEMORY_FINAL_REQUEST_BUILDER_INVALID,
)
from lightrag.api.document_lifecycle_service import DocumentLifecycleService
from lightrag.api.kb_service import KnowledgeBaseService
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry
from lightrag.api.query_tool_service import QueryToolService
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.sensitive_context import (
    CHAT_MEMORY_QUERY_LLM_EGRESS_NOT_ALLOWED,
    SensitiveContextPolicyError,
)
from lightrag.utils import logger


def _sensitive_context_error_response(
    exc: SensitiveContextPolicyError,
) -> tuple[int, dict[str, str]]:
    """Map a content-free policy failure to its stable API contract."""

    if exc.error_code == CHAT_MEMORY_QUERY_LLM_EGRESS_NOT_ALLOWED:
        status_code = 403
    elif exc.error_code == CHAT_MEMORY_FINAL_REQUEST_BUILDER_INVALID:
        status_code = 500
    else:
        # Request, query-length, and final-synthesis contract failures are
        # client errors. Keep the response independent of exception/provider
        # text by using only the stable policy code at the API boundary.
        status_code = 400
    detail = {
        "error_code": exc.error_code,
        "message": exc.error_code,
    }
    return status_code, detail


def _sensitive_context_http_exception(
    exc: SensitiveContextPolicyError,
) -> HTTPException:
    status_code, detail = _sensitive_context_error_response(exc)
    return HTTPException(status_code=status_code, detail=detail)


async def _stream_sensitive_context_errors(
    events: AsyncIterator[str],
) -> AsyncIterator[str]:
    """Convert late memory-policy failures into one terminal NDJSON event."""

    try:
        async for event in events:
            yield event
    except SensitiveContextPolicyError as exc:
        status_code, detail = _sensitive_context_error_response(exc)
        yield (
            json.dumps(
                {
                    "event": "error",
                    "error_code": detail["error_code"],
                    "status_code": status_code,
                    "message": detail["message"],
                },
                ensure_ascii=False,
            )
            + "\n"
        )


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
            sensitive_context = await authorize_memory_context(
                request, body.memory, body.query
            )
            result = await service.run(
                request=request,
                body=body,
                stream=False,
                sensitive_context=sensitive_context,
            )
            return {
                "status": result.status,
                "session_id": result.session_id,
                "answer": result.answer,
                "clarification_question": result.clarification_question,
                "references": result.references,
                "steps_summary": result.steps_summary,
                "metadata": result.metadata,
            }
        except SensitiveContextPolicyError as exc:
            raise _sensitive_context_http_exception(exc) from None
        except HTTPException as exc:
            if body.memory is not None and exc.status_code >= 500:
                logger.error("Memory-scoped Agent query failed")
                raise HTTPException(
                    status_code=exc.status_code, detail="Agent query failed"
                ) from None
            raise
        except Exception as exc:  # noqa: BLE001
            if body.memory is not None:
                logger.error("Memory-scoped Agent query failed")
                raise HTTPException(
                    status_code=500, detail="Agent query failed"
                ) from None
            logger.error("Agent query failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post(
        "/query/stream",
        dependencies=[Depends(combined_auth)],
        summary="Run a server-side Agent query as NDJSON events",
    )
    async def agent_query_stream(body: AgentQueryRequest, request: Request):
        try:
            sensitive_context = await authorize_memory_context(
                request, body.memory, body.query
            )
            events = service.stream_events(
                request=request,
                body=body,
                sensitive_context=sensitive_context,
            )
        except SensitiveContextPolicyError as exc:
            raise _sensitive_context_http_exception(exc) from None

        # Preserve the exact no-memory iterator and bytes. Only memory-scoped
        # streams need a route boundary for policy errors raised during final
        # synthesis, after the StreamingResponse has already started.
        if sensitive_context is not None:
            events = _stream_sensitive_context_errors(events)
        return StreamingResponse(
            events,
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "application/x-ndjson",
                "X-Accel-Buffering": "no",
            },
        )

    return router
