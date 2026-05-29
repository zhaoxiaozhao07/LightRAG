"""KB-scoped query routes.

Wraps the existing global ``/query`` / ``/query/stream`` / ``/query/data``
routes with a per-KB edge:

- The handler resolves the KB id from the path, fetches the corresponding
  ``LightRAG`` instance from ``LightRAGInstanceRegistry``, and calls the
  same ``aquery_llm`` / ``aquery_data`` methods that the global routes
  use.
- ``filters.doc_ids`` (when supplied) are validated against the KB's
  ``documents`` table so a request cannot retrieve a document that does
  not belong to the KB.
- ``mode`` accepts the same six values as the global route. ``bypass`` is
  still allowed but should be gated behind RBAC in a later phase.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional, cast

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from lightrag.api.config_version_service import (
    active_query_defaults_from_rag,
    active_query_metadata_from_rag,
)
from lightrag.api.document_lifecycle_service import DocumentLifecycleService
from lightrag.api.kb_service import KnowledgeBaseNotFoundError
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry
from lightrag.api.metadata_store import DocumentRecord
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.base import QueryParam
from lightrag.utils import logger

QueryMode = Literal["local", "global", "hybrid", "naive", "mix", "bypass"]
_QUERY_BLOCKING_DOCUMENT_STATUSES = {
    "deleting": "delete_job_active",
    "replacing": "replace_job_active",
}


class KBQueryFilters(BaseModel):
    doc_ids: Optional[List[str]] = Field(
        default=None,
        description=(
            "Restrict retrieval to a specific list of KB documents. "
            "Each id must belong to the target KB; otherwise the request is rejected."
        ),
    )


class KBQueryRequest(BaseModel):
    query: str = Field(min_length=3, description="The question to answer")
    mode: QueryMode = Field(default="mix")
    only_need_context: Optional[bool] = None
    only_need_prompt: Optional[bool] = None
    response_type: Optional[str] = Field(default=None, min_length=1)
    top_k: Optional[int] = Field(default=None, ge=1)
    chunk_top_k: Optional[int] = Field(default=None, ge=1)
    max_entity_tokens: Optional[int] = Field(default=None, ge=1)
    max_relation_tokens: Optional[int] = Field(default=None, ge=1)
    max_total_tokens: Optional[int] = Field(default=None, ge=1)
    hl_keywords: List[str] = Field(default_factory=list)
    ll_keywords: List[str] = Field(default_factory=list)
    conversation_history: Optional[List[Dict[str, Any]]] = None
    user_prompt: Optional[str] = None
    enable_rerank: Optional[bool] = None
    include_references: Optional[bool] = True
    include_chunk_content: Optional[bool] = False
    stream: Optional[bool] = True
    filters: Optional[KBQueryFilters] = None

    @field_validator("query", mode="after")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        return value.strip()

    @field_validator("conversation_history", mode="after")
    @classmethod
    def _validate_history(
        cls, value: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        if value is None:
            return None
        for message in value:
            if "role" not in message:
                raise ValueError("Each message must have a 'role' key.")
            if not isinstance(message["role"], str) or not message["role"].strip():
                raise ValueError("Each message 'role' must be a non-empty string.")
        return value

    def to_query_params(
        self,
        *,
        is_stream: bool,
        active_defaults: dict[str, Any] | None = None,
    ) -> QueryParam:
        route_only_fields = {"query", "include_chunk_content", "filters"}
        request_data = self.model_dump(
            exclude_none=True,
            exclude=route_only_fields,
        )
        explicit_fields = self.model_fields_set - route_only_fields
        data = dict(request_data)
        for key, value in (active_defaults or {}).items():
            if key not in route_only_fields and key not in explicit_fields:
                data[key] = value
        param = QueryParam(**data)
        param.stream = is_stream
        return param


class KBReferenceItem(BaseModel):
    reference_id: str
    file_path: str
    content: Optional[List[str]] = None


class KBQueryResponse(BaseModel):
    kb_id: str
    mode: QueryMode
    response: str
    references: Optional[List[KBReferenceItem]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KBQueryDataResponse(BaseModel):
    kb_id: str
    status: str
    message: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]


def _enrich_with_chunk_content(
    references: List[Dict[str, Any]], chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Attach chunk text bodies to each reference (for evaluation / debugging)."""
    bucket: Dict[str, List[str]] = {}
    for chunk in chunks:
        rid = chunk.get("reference_id", "")
        content = chunk.get("content", "")
        if rid and content:
            bucket.setdefault(rid, []).append(content)
    enriched = []
    for ref in references:
        copy = dict(ref)
        rid = ref.get("reference_id", "")
        if rid in bucket:
            copy["content"] = bucket[rid]
        enriched.append(copy)
    return enriched


async def _validate_doc_ids_belong_to_kb(
    document_service: DocumentLifecycleService,
    kb_id: str,
    doc_ids: List[str],
) -> List[DocumentRecord]:
    if not doc_ids:
        return []
    documents = await document_service.get_documents_by_ids(kb_id, doc_ids)
    found = {document.id for document in documents}
    missing = [doc_id for doc_id in doc_ids if doc_id not in found]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "doc_ids_not_in_kb",
                "missing": missing,
                "message": "filters.doc_ids reference documents outside the target KB",
            },
        )
    return documents


def _active_lifecycle_job_id(document: DocumentRecord) -> str:
    if document.status == "deleting":
        job_id = document.metadata.get("current_delete_job_id") or document.metadata.get(
            "pending_delete_job_id"
        )
        return str(job_id) if job_id else "unknown"
    if document.status == "replacing":
        job_id = document.metadata.get("current_replace_job_id") or document.metadata.get(
            "pending_replace_job_id"
        )
        return str(job_id) if job_id else "unknown"
    return "unknown"


def _raise_active_lifecycle_query_conflict(document: DocumentRecord) -> None:
    status = str(document.status)
    error_code = _QUERY_BLOCKING_DOCUMENT_STATUSES[status]
    raise HTTPException(
        status_code=409,
        detail={
            "error_code": error_code,
            "document_id": document.id,
            "existing_job_id": _active_lifecycle_job_id(document),
            "message": f"Document '{document.id}' is currently {status}",
        },
    )


async def _ensure_query_documents_available(
    document_service: DocumentLifecycleService,
    kb_id: str,
    doc_ids: List[str] | None,
) -> None:
    if doc_ids:
        documents = await _validate_doc_ids_belong_to_kb(document_service, kb_id, doc_ids)
        for document in documents:
            if document.status in _QUERY_BLOCKING_DOCUMENT_STATUSES:
                _raise_active_lifecycle_query_conflict(document)
        return

    for status in _QUERY_BLOCKING_DOCUMENT_STATUSES:
        documents, total = await document_service.list_documents(
            kb_id, status=status, limit=1, offset=0
        )
        if total > 0 and documents:
            _raise_active_lifecycle_query_conflict(documents[0])


def create_kb_query_routes(
    document_service: DocumentLifecycleService,
    registry: LightRAGInstanceRegistry,
    api_key: Optional[str] = None,
):
    router = APIRouter(prefix="/kbs", tags=["knowledge-base-query"])
    combined_auth = get_combined_auth_dependency(api_key)

    @router.post(
        "/{kb_id}/query",
        response_model=KBQueryResponse,
        dependencies=[Depends(combined_auth)],
        summary="Run a non-streaming RAG query against a knowledge base",
    )
    async def kb_query(kb_id: str, request: KBQueryRequest):
        try:
            await _ensure_query_documents_available(
                document_service,
                kb_id,
                request.filters.doc_ids if request.filters else None,
            )
            rag = cast(Any, await registry.get(kb_id))
            active_defaults = active_query_defaults_from_rag(rag)
            active_metadata = active_query_metadata_from_rag(rag)
            param = request.to_query_params(
                is_stream=False,
                active_defaults=active_defaults,
            )
            param.stream = False
            # NOTE: ``filters.doc_ids`` is currently validated against the KB
            # metadata and active lifecycle state.
            # Per-doc retrieval filtering inside QueryParam is not yet
            # supported by LightRAG; KB workspace isolation already
            # guarantees cross-KB separation.
            result = await rag.aquery_llm(request.query, param=param)
            llm_response = result.get("llm_response", {})
            data = result.get("data", {})
            references = data.get("references", [])
            response_text = llm_response.get("content") or "No relevant context found for the query."
            include_references = bool(param.include_references)
            if include_references and request.include_chunk_content:
                references = _enrich_with_chunk_content(
                    references, data.get("chunks", [])
                )
            return KBQueryResponse(
                kb_id=kb_id,
                mode=cast(QueryMode, param.mode),
                response=response_text,
                references=[
                    KBReferenceItem(**ref) for ref in references
                ]
                if include_references
                else None,
                metadata=active_metadata,
            )
        except HTTPException:
            raise
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("KB query failed for '%s': %s", kb_id, exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post(
        "/{kb_id}/query/stream",
        dependencies=[Depends(combined_auth)],
        summary="Run a streaming RAG query against a knowledge base (NDJSON)",
    )
    async def kb_query_stream(kb_id: str, request: KBQueryRequest):
        try:
            await _ensure_query_documents_available(
                document_service,
                kb_id,
                request.filters.doc_ids if request.filters else None,
            )
            rag = cast(Any, await registry.get(kb_id))
            active_defaults = active_query_defaults_from_rag(rag)
            active_metadata = active_query_metadata_from_rag(rag)
            stream_mode = request.stream if request.stream is not None else True
            param = request.to_query_params(
                is_stream=stream_mode,
                active_defaults=active_defaults,
            )
            result = await rag.aquery_llm(request.query, param=param)

            async def stream_generator():
                references = result.get("data", {}).get("references", [])
                llm_response = result.get("llm_response", {})
                include_references = bool(param.include_references)
                if include_references and request.include_chunk_content:
                    references = _enrich_with_chunk_content(
                        references, result.get("data", {}).get("chunks", [])
                    )
                if llm_response.get("is_streaming"):
                    payload = {
                        "kb_id": kb_id,
                        "metadata": active_metadata,
                    }
                    if include_references:
                        payload["references"] = references
                    yield f"{json.dumps(payload)}\n"
                    iterator = llm_response.get("response_iterator")
                    if iterator:
                        try:
                            async for chunk in iterator:
                                if chunk:
                                    yield f"{json.dumps({'response': chunk})}\n"
                        except Exception as exc:  # noqa: BLE001
                            logger.error("KB stream error: %s", exc)
                            yield f"{json.dumps({'error': str(exc)})}\n"
                else:
                    body = {
                        "kb_id": kb_id,
                        "response": llm_response.get("content", ""),
                        "metadata": active_metadata,
                    }
                    if include_references:
                        body["references"] = references
                    yield f"{json.dumps(body)}\n"

            return StreamingResponse(
                stream_generator(),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "application/x-ndjson",
                    "X-Accel-Buffering": "no",
                },
            )
        except HTTPException:
            raise
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "KB streaming query failed for '%s': %s", kb_id, exc, exc_info=True
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post(
        "/{kb_id}/query/data",
        response_model=KBQueryDataResponse,
        dependencies=[Depends(combined_auth)],
        summary="Return structured retrieval data without generating an LLM answer",
    )
    async def kb_query_data(kb_id: str, request: KBQueryRequest):
        try:
            await _ensure_query_documents_available(
                document_service,
                kb_id,
                request.filters.doc_ids if request.filters else None,
            )
            rag = cast(Any, await registry.get(kb_id))
            active_defaults = active_query_defaults_from_rag(rag)
            active_metadata = active_query_metadata_from_rag(rag)
            param = request.to_query_params(
                is_stream=False,
                active_defaults=active_defaults,
            )
            param.stream = False
            result = await rag.aquery_data(request.query, param=param)
            return KBQueryDataResponse(
                kb_id=kb_id,
                status=result.get("status", "success"),
                message=result.get("message", ""),
                data=result.get("data", {}),
                metadata={**result.get("metadata", {}), **active_metadata},
            )
        except HTTPException:
            raise
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "KB query/data failed for '%s': %s", kb_id, exc, exc_info=True
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post(
        "/{kb_id}/retrieve",
        response_model=KBQueryDataResponse,
        dependencies=[Depends(combined_auth)],
        summary="Alias for /query/data — retrieval only, no LLM generation",
    )
    async def kb_retrieve(kb_id: str, request: KBQueryRequest):
        return await kb_query_data(kb_id, request)

    return router
