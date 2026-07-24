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
- ``mode`` accepts the same six values as the global route. In enterprise
  mode, effective ``bypass`` mode is gated after active KB defaults are merged.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any, Dict, List, Literal, Optional, Protocol, cast

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from lightrag.api.bilingual_query_service import (
    answer_language_rules,
    apply_plan_keywords_to_param,
    bilingual_applies,
    bilingual_audit_fields,
    bilingual_mode_from_rag,
    bilingual_query_data,
    bilingual_query_llm,
    dual_aquery_data,
    prepare_bilingual_queries,
    resolve_bilingual_mode,
)
from lightrag.api.config_version_service import (
    active_query_defaults_from_rag,
    active_query_metadata_from_rag,
)
from lightrag.api.document_lifecycle_service import DocumentLifecycleService
from lightrag.api.enterprise_auth import (
    KB_ROLE_VIEWER,
    UserKBQuerySettingsService,
    append_enterprise_audit_event,
    enterprise_auth_enabled,
    get_enterprise_authorization_service,
    get_request_principal,
)
from lightrag.api.chat_memory_routing import (
    ChatMemoryScope,
    authorize_memory_context,
    memory_audit_fields,
)
from lightrag.api.chat_memory_service import (
    CHAT_MEMORY_FINAL_REQUEST_BUILDER_INVALID,
    MEMORY_QUERY_MAX_LENGTH,
    AuthorizedChatMemoryHandle,
)
from lightrag.api.kb_service import KnowledgeBaseNotFoundError
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry
from lightrag.api.metadata_store import DocumentRecord
from lightrag.api.streaming_lifecycle import (
    ClientGoneError,
    abort_if_client_gone,
    await_with_disconnect_check,
    client_closed_response,
    safe_aclose,
    stream_with_disconnect_guard,
)
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.base import QueryParam
from lightrag.prompt import PROMPTS
from lightrag.sensitive_context import (
    CHAT_MEMORY_QUERY_LLM_EGRESS_NOT_ALLOWED,
    SensitiveContextPayload,
    SensitiveContextPolicyError,
    bind_sensitive_context_endpoint,
    mark_sensitive_context_not_used,
    serialize_sensitive_final_request,
)
from lightrag.utils import (
    generate_reference_list_from_chunks,
    get_llm_cache_identity,
    logger,
    process_chunks_unified,
)

QueryMode = Literal["local", "global", "hybrid", "naive", "mix", "bypass"]

_CHAT_MEMORY_REQUIRES_FINAL_SYNTHESIS = "chat_memory_requires_final_synthesis"
_CHAT_MEMORY_QUERY_TOO_LONG = "chat_memory_query_too_long"
_CHAT_MEMORY_INTERNAL_ERROR = "chat_memory_sensitive_context_failed"


def _chat_memory_http_error(status_code: int, error_code: str) -> HTTPException:
    """Build a stable, content-free Chat Memory API error."""

    return HTTPException(
        status_code=status_code,
        detail={"error_code": error_code, "message": error_code},
    )


def _reject_memory_without_final_synthesis(
    scope: ChatMemoryScope | None,
    param: QueryParam | None = None,
    *,
    final_synthesis: bool = True,
) -> None:
    """Reject memory whenever the request cannot execute a final query LLM."""

    if scope is None:
        return
    if (
        not final_synthesis
        or param is None
        or param.mode == "bypass"
        or bool(param.only_need_context)
        or bool(param.only_need_prompt)
    ):
        raise _chat_memory_http_error(
            400,
            _CHAT_MEMORY_REQUIRES_FINAL_SYNTHESIS,
        )


async def _authorize_memory_handle(
    http_request: Request,
    scope: ChatMemoryScope | None,
    query: str,
) -> AuthorizedChatMemoryHandle | None:
    """Authorize before KB work while keeping fact search lazy."""

    if scope is None:
        return None
    if len(query) > MEMORY_QUERY_MAX_LENGTH:
        raise _chat_memory_http_error(400, _CHAT_MEMORY_QUERY_TOO_LONG)
    return await authorize_memory_context(http_request, scope, query)


def _map_sensitive_context_policy_error(
    exc: SensitiveContextPolicyError,
) -> HTTPException:
    """Map stable sensitive-context policy errors without exposing internals."""

    if exc.error_code == CHAT_MEMORY_QUERY_LLM_EGRESS_NOT_ALLOWED:
        return _chat_memory_http_error(403, exc.error_code)
    if exc.error_code in {
        _CHAT_MEMORY_REQUIRES_FINAL_SYNTHESIS,
        _CHAT_MEMORY_QUERY_TOO_LONG,
    }:
        return _chat_memory_http_error(400, exc.error_code)
    if exc.error_code == CHAT_MEMORY_FINAL_REQUEST_BUILDER_INVALID:
        return _chat_memory_http_error(500, exc.error_code)
    return _chat_memory_http_error(500, _CHAT_MEMORY_INTERNAL_ERROR)


async def _reject_explicit_memory_without_final_synthesis(
    request: Request,
) -> None:
    """Reject memory-scoped bypass before enterprise bypass gating.

    Enterprise request authorization checks explicit ``mode=bypass`` before the
    route handler runs. This lightweight body dependency must therefore run
    first so a memory-scoped bypass request receives the stable memory contract
    error instead of the unrelated bypass-capability response. The body remains
    cached on ``Request`` for normal FastAPI model validation.
    """

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 - normal validation owns malformed JSON
        return
    if not isinstance(body, dict) or body.get("memory") is None:
        return
    if body.get("mode") == "bypass":
        raise _chat_memory_http_error(
            400,
            _CHAT_MEMORY_REQUIRES_FINAL_SYNTHESIS,
        )


class _DocumentListService(Protocol):
    async def list_documents(
        self,
        kb_id: str,
        *,
        status: str | None = None,
        source_name: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[DocumentRecord], int]: ...
_QUERY_BLOCKING_DOCUMENT_STATUSES = {
    "deleting": "delete_job_active",
    "replacing": "replace_job_active",
}
_MAX_METADATA_FILTER_BYTES = 64 * 1024
_METADATA_FILTER_SCALAR_TYPES = (str, int, float, bool, type(None))


def _enforce_resolved_bypass_permission(request: Request, param: QueryParam) -> None:
    if param.mode != "bypass" or not enterprise_auth_enabled():
        return
    get_enterprise_authorization_service(request).require_bypass_query(
        get_request_principal(request)
    )


async def _merge_user_query_defaults(
    request: Request,
    kb_id: str,
    active_defaults: dict[str, Any],
) -> dict[str, Any]:
    """Overlay per-user persisted KB query prompt on KB active defaults.

    Request fields still win because ``KBQueryRequest.to_query_params`` only
    applies defaults for fields the request did not explicitly provide.
    """
    defaults = dict(active_defaults)
    if not enterprise_auth_enabled():
        return defaults
    principal = get_request_principal(request)
    if principal is None or principal.auth_method != "jwt":
        return defaults
    service = getattr(request.app.state, "enterprise_user_kb_query_settings_service", None)
    if not isinstance(service, UserKBQuerySettingsService):
        return defaults
    settings = await service.get_settings(principal.user_id, kb_id)
    if settings is not None and settings.user_prompt:
        defaults["user_prompt"] = settings.user_prompt
    return defaults


def _query_audit_metadata(
    body: "KBQueryRequest",
    param: QueryParam,
    active_metadata: dict[str, Any],
    *,
    route: str,
    stream: bool,
    bilingual_info: dict[str, Any] | None = None,
    memory_handle: AuthorizedChatMemoryHandle | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "route": route,
        "mode": param.mode,
        "stream": stream,
        "query_hash": hashlib.sha256(body.query.encode("utf-8")).hexdigest(),
        "top_k": getattr(param, "top_k", None),
        "chunk_top_k": getattr(param, "chunk_top_k", None),
        "only_need_context": getattr(param, "only_need_context", None),
        "only_need_prompt": getattr(param, "only_need_prompt", None),
        "has_doc_filters": bool(body.filters and body.filters.doc_ids),
        "doc_filter_count": len(body.filters.doc_ids)
        if body.filters and body.filters.doc_ids
        else 0,
        "metadata_filter_keys": sorted((body.filters.metadata or {}).keys())
        if body.filters and body.filters.metadata
        else [],
    }
    for key in ("config_version_id", "parser_hash", "index_hash", "query_hash"):
        if key in active_metadata:
            metadata[f"active_{key}"] = active_metadata[key]
    if bilingual_info is not None:
        metadata.update(bilingual_audit_fields(bilingual_info))
    metadata.update(
        memory_audit_fields(memory_handle.info if memory_handle is not None else None)
    )
    return metadata


async def _maybe_bilingual_plan(
    rag: Any, request: "KBQueryRequest", param: QueryParam
):
    """Resolve the bilingual mode for one KB query and preprocess if it applies.

    Returns ``(plan, info)``:

    - ``(plan, None)`` — dual-path should run; keywords already seeded on
      ``param``; the caller builds the final info block from the dual result.
    - ``(None, info)`` — dual-path was requested but preprocessing was
      unavailable; ``info`` explains the single-path fallback.
    - ``(None, None)`` — bilingual is off / not applicable; responses stay
      byte-identical to deployments that never enable the feature.
    """
    mode = resolve_bilingual_mode(request.bilingual, bilingual_mode_from_rag(rag))
    if not bilingual_applies(mode, request.query, param):
        return None, None
    plan = await prepare_bilingual_queries(rag, request.query)
    if plan is None:
        return None, {
            "enabled": False,
            "mode": mode,
            "reason": "preprocess_unavailable",
        }
    apply_plan_keywords_to_param(param, plan)
    return plan, None


class KBQueryFilters(BaseModel):
    doc_ids: Optional[List[str]] = Field(
        default=None,
        description=(
            "Restrict retrieval to a specific list of KB documents. "
            "Each id must belong to the target KB; otherwise the request is rejected."
        ),
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Restrict retrieval to documents whose metadata exactly matches these "
            "key/value filters. Values may be scalars or lists of scalar OR values."
        ),
    )

    @field_validator("metadata", mode="after")
    @classmethod
    def _validate_metadata_filters(
        cls, value: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        if len(encoded) > _MAX_METADATA_FILTER_BYTES:
            raise ValueError(
                "filters.metadata is too large. Maximum size: "
                f"{_MAX_METADATA_FILTER_BYTES} bytes"
            )
        for key, filter_value in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("filters.metadata keys must be non-empty strings")
            if isinstance(filter_value, list):
                if not all(
                    isinstance(item, _METADATA_FILTER_SCALAR_TYPES)
                    for item in filter_value
                ):
                    raise ValueError(
                        "filters.metadata values must be scalars or lists of scalars"
                    )
                continue
            if not isinstance(filter_value, _METADATA_FILTER_SCALAR_TYPES):
                raise ValueError(
                    "filters.metadata values must be scalars or lists of scalars"
                )
        return value


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
    memory: Optional[ChatMemoryScope] = Field(
        default=None,
        description=(
            "Server-side chat memory injection. Provide the chat project_id to "
            "authorize lazy, budgeted project-memory context for final synthesis "
            "(docs/ChatMemory-zh.md §6). "
            "Requires an interactive user owning the project and "
            "LIGHTRAG_CHAT_MEMORY_ENABLED."
        ),
    )
    bilingual: Optional[bool] = Field(
        default=None,
        description=(
            "Explicit dual-path bilingual retrieval override. True forces it "
            "on, False forces it off; omit to follow the KB's "
            "query_config.bilingual_query and the deployment default. The "
            "BILINGUAL_QUERY_ENABLED master switch must be on either way."
        ),
    )

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
        route_only_fields = {
            "query",
            "include_chunk_content",
            "filters",
            "bilingual",
            "memory",
        }
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


async def _list_all_kb_documents(
    document_service: _DocumentListService,
    kb_id: str,
) -> list[DocumentRecord]:
    page_size = 200
    offset = 0
    all_documents: list[DocumentRecord] = []
    while True:
        documents, total = await document_service.list_documents(
            kb_id, limit=page_size, offset=offset
        )
        all_documents.extend(documents)
        offset += page_size
        if offset >= total or not documents:
            break
    return all_documents


def _metadata_filter_matches(document: DocumentRecord, filters: Dict[str, Any]) -> bool:
    for key, filter_value in filters.items():
        allowed_values = filter_value if isinstance(filter_value, list) else [filter_value]
        document_value = document.metadata.get(key)
        if isinstance(document_value, list):
            if not any(item in allowed_values for item in document_value):
                return False
            continue
        if document_value not in allowed_values:
            return False
    return True


def _has_metadata_filters(filters: KBQueryFilters | None) -> bool:
    return bool(filters and filters.metadata)


def _has_doc_id_filter(filters: KBQueryFilters | None) -> bool:
    return bool(filters and filters.doc_ids is not None)


def _coerce_filters(
    filters_or_doc_ids: KBQueryFilters | List[str] | None,
) -> KBQueryFilters | None:
    if filters_or_doc_ids is None or isinstance(filters_or_doc_ids, KBQueryFilters):
        return filters_or_doc_ids
    return KBQueryFilters(doc_ids=filters_or_doc_ids)


def _effective_candidate_documents(
    all_documents: list[DocumentRecord], filters: KBQueryFilters | None
) -> list[DocumentRecord]:
    candidates = all_documents
    if _has_doc_id_filter(filters):
        requested = set(filters.doc_ids or []) if filters else set()
        candidates = [document for document in candidates if document.id in requested]
    if _has_metadata_filters(filters) and filters and filters.metadata:
        candidates = [
            document
            for document in candidates
            if _metadata_filter_matches(document, filters.metadata)
        ]
    return candidates


def _validate_doc_ids_from_documents(
    all_documents: list[DocumentRecord], doc_ids: list[str]
) -> None:
    found = {document.id for document in all_documents}
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


async def _ensure_query_filter_documents_available(
    document_service: DocumentLifecycleService,
    kb_id: str,
    filters: KBQueryFilters | None,
) -> None:
    if not (_has_doc_id_filter(filters) or _has_metadata_filters(filters)):
        await _ensure_query_documents_available(document_service, kb_id, None)
        return

    all_documents = await _list_all_kb_documents(document_service, kb_id)
    if _has_doc_id_filter(filters) and filters:
        _validate_doc_ids_from_documents(all_documents, filters.doc_ids or [])
    for document in _effective_candidate_documents(all_documents, filters):
        if document.status in _QUERY_BLOCKING_DOCUMENT_STATUSES:
            _raise_active_lifecycle_query_conflict(document)


async def _resolve_doc_id_scope(
    document_service: _DocumentListService,
    kb_id: str,
    filters_or_doc_ids: KBQueryFilters | List[str] | None,
) -> List[str] | None:
    """Compute the ``lightrag_doc_id`` allow-list to pass to ``QueryParam.ids``.

    Retrieval scope rules:

    - A document is *retrievable* only when ``enabled`` and not ``archived``
      and it has an indexed ``lightrag_doc_id``.
    - When ``filters.doc_ids`` is supplied, the scope is exactly those
      documents (already validated to belong to the KB) intersected with the
      retrievable set — disabled/archived ids silently drop out so they can
      never leak into an answer.
    - When no ``filters.doc_ids`` is supplied and every document is retrievable,
      return ``None`` (unrestricted retrieval, full recall, zero overhead).
    - When no ``filters.doc_ids`` is supplied but some documents are disabled or
      archived, return the retrievable allow-list so excluded documents are
      filtered out at retrieval time.

    Returns ``None`` for "no scoping", otherwise a (possibly empty) list of
    ``lightrag_doc_id`` values.
    """
    filters = _coerce_filters(filters_or_doc_ids)
    all_documents = await _list_all_kb_documents(document_service, kb_id)
    if _has_doc_id_filter(filters) and filters:
        _validate_doc_ids_from_documents(all_documents, filters.doc_ids or [])

    def _retrievable(document: DocumentRecord) -> bool:
        return (
            document.enabled
            and not document.archived
            and bool(document.lightrag_doc_id)
        )

    has_excluded = any(
        (not document.enabled or document.archived or not document.lightrag_doc_id)
        for document in all_documents
    )

    if _has_doc_id_filter(filters) or _has_metadata_filters(filters):
        return [
            str(document.lightrag_doc_id)
            for document in _effective_candidate_documents(all_documents, filters)
            if _retrievable(document)
        ]

    if not has_excluded:
        return None

    return [
        str(document.lightrag_doc_id)
        for document in all_documents
        if _retrievable(document)
    ]


_MAX_MULTI_KB = 10


class MultiKBQueryRequest(BaseModel):
    """Query several knowledge bases at once and synthesize one answer.

    All target KBs must share the same embedding model/dim (so cross-KB
    relevance is comparable); KB isolation is preserved — each KB is retrieved
    through its own instance and results are merged at the retrieval layer.
    """

    kb_ids: List[str] = Field(min_length=1, max_length=_MAX_MULTI_KB)
    query: str = Field(min_length=3)
    mode: QueryMode = Field(default="mix")
    response_type: Optional[str] = Field(default=None, min_length=1)
    top_k: Optional[int] = Field(default=None, ge=1)
    chunk_top_k: Optional[int] = Field(default=None, ge=1)
    max_total_tokens: Optional[int] = Field(default=None, ge=1)
    conversation_history: Optional[List[Dict[str, Any]]] = None
    user_prompt: Optional[str] = None
    enable_rerank: Optional[bool] = None
    include_references: Optional[bool] = True
    include_chunk_content: Optional[bool] = False
    filters: Optional[KBQueryFilters] = None
    memory: Optional[ChatMemoryScope] = Field(
        default=None,
        description=(
            "Server-side chat memory injection (docs/ChatMemory-zh.md §6). "
            "Provide the chat project_id to authorize lazy, budgeted project "
            "memory for final synthesis. Requires an interactive user owning the "
            "project and LIGHTRAG_CHAT_MEMORY_ENABLED."
        ),
    )
    bilingual: Optional[bool] = Field(
        default=None,
        description=(
            "Explicit dual-path bilingual retrieval override for every target "
            "KB. Multi-KB queries do not consult per-KB query_config; omit to "
            "follow the deployment default mode."
        ),
    )

    @field_validator("query", mode="after")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        return value.strip()

    @field_validator("kb_ids", mode="after")
    @classmethod
    def _dedup_kb_ids(cls, value: List[str]) -> List[str]:
        seen: set[str] = set()
        result: List[str] = []
        for raw in value:
            kb_id = raw.strip()
            if not kb_id:
                raise ValueError("kb_ids entries must be non-empty strings")
            if kb_id not in seen:
                seen.add(kb_id)
                result.append(kb_id)
        if not result:
            raise ValueError("kb_ids must contain at least one knowledge base id")
        return result

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
        self, *, active_defaults: dict[str, Any] | None = None
    ) -> QueryParam:
        route_only_fields = {
            "query",
            "kb_ids",
            "include_chunk_content",
            "include_references",
            "filters",
            "bilingual",
            "memory",
        }
        request_data = self.model_dump(exclude_none=True, exclude=route_only_fields)
        explicit_fields = self.model_fields_set - route_only_fields
        data = dict(request_data)
        for key, value in (active_defaults or {}).items():
            if key not in route_only_fields and key not in explicit_fields:
                data[key] = value
        param = QueryParam(**data)
        param.stream = False
        return param


class MultiKBReferenceItem(BaseModel):
    reference_id: str
    file_path: str
    kb_id: str
    content: Optional[List[str]] = None


class MultiKBQueryResponse(BaseModel):
    kb_ids: List[str]
    mode: QueryMode
    response: str
    references: Optional[List[MultiKBReferenceItem]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MultiKBQueryDataResponse(BaseModel):
    kb_ids: List[str]
    status: str
    message: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)


def _dedup_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop chunks that are exact duplicates across KBs, keeping first seen."""
    seen: set = set()
    result: List[Dict[str, Any]] = []
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if chunk_id:
            key = ("id", chunk.get("file_path"), chunk_id)
        else:
            key = ("content", chunk.get("content"))
        if key in seen:
            continue
        seen.add(key)
        result.append(chunk)
    return result


def _multi_kb_query_audit_metadata(
    request: MultiKBQueryRequest,
    param: QueryParam | None,
    *,
    kb_ids: List[str],
    skipped: List[Dict[str, Any]],
    reranked: bool,
    final_count: int,
    bilingual_info: Dict[str, Any] | None = None,
    memory_handle: AuthorizedChatMemoryHandle | None = None,
) -> Dict[str, Any]:
    metadata = {
        "route": "multi_query",
        "mode": param.mode if param is not None else request.mode,
        "kb_ids": kb_ids,
        "kb_count": len(kb_ids),
        "skipped_count": len(skipped),
        "query_hash": hashlib.sha256(request.query.encode("utf-8")).hexdigest(),
        "top_k": getattr(param, "top_k", None) if param is not None else None,
        "chunk_top_k": getattr(param, "chunk_top_k", None)
        if param is not None
        else None,
        "reranked": reranked,
        "final_chunk_count": final_count,
    }
    if bilingual_info is not None:
        metadata.update(bilingual_audit_fields(bilingual_info))
    metadata.update(
        memory_audit_fields(memory_handle.info if memory_handle is not None else None)
    )
    return metadata


async def _resolve_multi_kb_doc_id_filters(
    document_service: DocumentLifecycleService,
    kb_ids: List[str],
    filters: "KBQueryFilters | None",
) -> Dict[str, "KBQueryFilters | None"]:
    """Map each target KB to the filters it should apply.

    ``metadata`` filters apply uniformly to every KB. ``doc_ids`` are KB-scoped:
    each KB receives only the requested ids that actually belong to it, and
    every requested id must belong to at least one target KB (otherwise 400 —
    a genuine mistake). This avoids the single-KB strict validation rejecting
    ids that legitimately live in a sibling target KB.
    """
    if not _has_doc_id_filter(filters):
        return {kb_id: filters for kb_id in kb_ids}
    requested = list(filters.doc_ids or []) if filters else []
    per_kb: Dict[str, "KBQueryFilters | None"] = {}
    belonging_union: set[str] = set()
    metadata = filters.metadata if filters else None
    for kb_id in kb_ids:
        docs = await document_service.get_documents_by_ids(kb_id, requested)
        ids_here = [doc.id for doc in docs]
        belonging_union.update(ids_here)
        per_kb[kb_id] = KBQueryFilters(doc_ids=ids_here, metadata=metadata)
    missing = [doc_id for doc_id in requested if doc_id not in belonging_union]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "doc_ids_not_in_kb",
                "missing": missing,
                "message": "filters.doc_ids reference documents outside all target KBs",
            },
        )
    return per_kb


async def _prepare_multi_kb_synthesis(
    request: "MultiKBQueryRequest",
    merged: List[Dict[str, Any]],
    synth_rag: Any,
    synth_param: QueryParam | None,
    *,
    bilingual_info: Dict[str, Any] | None = None,
    sensitive_context: AuthorizedChatMemoryHandle | None = None,
):
    """Process merged chunks and build the single-synthesis system prompt.

    Returns ``(sys_prompt, use_model_func, references, reranked, final_count)``;
    ``sys_prompt``/``use_model_func`` are ``None`` when there is nothing to
    synthesize (no merged chunks). The caller invokes ``use_model_func`` with
    the desired ``stream`` flag so non-streaming and streaming share this logic.
    """
    if not (merged and synth_rag is not None and synth_param is not None):
        mark_sensitive_context_not_used(sensitive_context, "no_kb_evidence")
        return None, None, [], False, 0

    global_config = synth_rag._build_global_config()
    tokenizer = global_config.get("tokenizer")
    response_type = synth_param.response_type or "Multiple Paragraphs"
    user_prompt_parts: List[str] = []
    if bilingual_info and bilingual_info.get("enabled"):
        user_prompt_parts.append(
            answer_language_rules(
                str(bilingual_info.get("source_language") or "zh")
            )
        )
    if synth_param.user_prompt:
        user_prompt_parts.append(synth_param.user_prompt)
    joined_user_prompt = "\n\n".join(user_prompt_parts)
    user_prompt = f"\n\n{joined_user_prompt}" if joined_user_prompt else "n/a"
    max_total_tokens = (
        getattr(synth_param, "max_total_tokens", None)
        or global_config.get("max_total_tokens")
        or 30000
    )
    chunk_token_limit: int | None = None
    if tokenizer:
        pre_sys = PROMPTS["naive_rag_response"].format(
            response_type=response_type, user_prompt=user_prompt, content_data=""
        )
        chunk_token_limit = max_total_tokens - (
            len(tokenizer.encode(pre_sys)) + len(tokenizer.encode(request.query)) + 200
        )

    processed = await process_chunks_unified(
        query=request.query,
        unique_chunks=merged,
        query_param=synth_param,
        global_config=global_config,
        source_type="multi_kb",
        chunk_token_limit=chunk_token_limit,
    )
    reranked = bool(global_config.get("rerank_model_func")) and bool(
        synth_param.enable_rerank
    )
    final_count = len(processed)

    reference_list, processed_with_ids = generate_reference_list_from_chunks(processed)
    if sensitive_context is not None and not processed_with_ids:
        mark_sensitive_context_not_used(sensitive_context, "no_kb_evidence")
        return None, None, [], reranked, 0

    chunks_context = [
        {"reference_id": c["reference_id"], "content": c["content"]}
        for c in processed_with_ids
        if c.get("reference_id")
    ]
    text_units_str = "\n".join(
        json.dumps(unit, ensure_ascii=False) for unit in chunks_context
    )
    reference_list_str = "\n".join(
        f"[{ref['reference_id']}] {ref['file_path']}"
        for ref in reference_list
        if ref["reference_id"]
    )
    content_data = PROMPTS["naive_query_context"].format(
        text_chunks_str=text_units_str, reference_list_str=reference_list_str
    )
    def build_system_prompt(payload: SensitiveContextPayload | None) -> str:
        effective_user_prompt = user_prompt
        effective_content_data = content_data
        if payload is not None:
            # The trusted policy is the last server instruction. Only the
            # untrusted JSONL records enter the Context section.
            effective_user_prompt = (
                f"{effective_user_prompt}\n\n{payload.trusted_policy}"
            )
            effective_content_data = (
                f"{effective_content_data}\n\n{payload.context_data}"
            )
        return PROMPTS["naive_rag_response"].format(
            response_type=response_type,
            user_prompt=effective_user_prompt,
            content_data=effective_content_data,
        )

    sys_prompt = build_system_prompt(None)

    ref_kb: Dict[str, str] = {}
    ref_content: Dict[str, List[str]] = {}
    for chunk in processed_with_ids:
        rid = chunk.get("reference_id")
        if not rid:
            continue
        ref_kb.setdefault(rid, chunk.get("kb_id", ""))
        if request.include_chunk_content:
            ref_content.setdefault(rid, []).append(chunk.get("content", ""))
    references = [
        MultiKBReferenceItem(
            reference_id=ref["reference_id"],
            file_path=ref["file_path"],
            kb_id=ref_kb.get(ref["reference_id"], ""),
            content=ref_content.get(ref["reference_id"])
            if request.include_chunk_content
            else None,
        )
        for ref in reference_list
        if ref["reference_id"]
    ]
    use_model_func = global_config["role_llm_funcs"]["query"]
    if sensitive_context is not None:
        # Re-read the exact runtime used for final synthesis only after merged,
        # processed authoritative evidence and the synthesis RAG are known.
        final_global_config = synth_rag._build_global_config()
        final_identity = get_llm_cache_identity(final_global_config, "query")
        bind_sensitive_context_endpoint(
            sensitive_context,
            final_identity.get("host")
            if isinstance(final_identity, dict)
            else None,
        )

        def build_final_request(
            payload: SensitiveContextPayload | None,
        ) -> str:
            return serialize_sensitive_final_request(
                build_system_prompt(payload),
                request.query,
                synth_param.conversation_history,
            )

        final_max_total_tokens = (
            getattr(synth_param, "max_total_tokens", None)
            or final_global_config.get("max_total_tokens")
            or 30000
        )
        payload = await sensitive_context.resolve_for_final_request(
            final_global_config.get("tokenizer"),
            final_max_total_tokens,
            build_final_request,
        )
        sys_prompt = build_system_prompt(payload)
        use_model_func = final_global_config["role_llm_funcs"]["query"]
    return sys_prompt, use_model_func, references, reranked, final_count


async def _multi_kb_retrieve(
    document_service: DocumentLifecycleService,
    registry: LightRAGInstanceRegistry,
    request: MultiKBQueryRequest,
    http_request: Request,
) -> tuple[
    List[Dict[str, Any]],
    Any,
    QueryParam | None,
    List[str],
    List[Dict[str, Any]],
    Dict[str, int],
    Dict[str, Any] | None,
]:
    """Fan out retrieval across ``request.kb_ids`` and merge the chunks.

    Returns ``(merged_chunks, synth_rag, synth_param, queried_kb_ids,
    skipped_kbs, per_kb_chunk_counts, bilingual_info)``. ``synth_rag``/
    ``synth_param`` come from the first KB that retrieved successfully
    (drive synthesis).

    SECURITY: the central middleware ``enforce_enterprise_request_access`` does
    NOT cover the collection-level ``/kbs:query`` / ``/kbs:retrieve`` paths
    (``_extract_kb_id`` returns ``None`` for them), so this function MUST
    enforce ``kb_viewer`` on every target KB itself — fail closed.
    """
    if request.mode == "bypass":
        raise HTTPException(
            status_code=400,
            detail="bypass mode is not supported for multi-KB query",
        )

    kb_ids = request.kb_ids
    if enterprise_auth_enabled():
        principal = get_request_principal(http_request)
        authz = get_enterprise_authorization_service(http_request)
        for kb_id in kb_ids:
            await authz.require_kb_role(principal, kb_id, KB_ROLE_VIEWER)

    # Multi-KB queries deliberately use request-level params only (see the
    # comment in _retrieve_one), so bilingual mode also resolves from the
    # request flag plus the deployment default — per-KB query_config is not
    # consulted. Preprocessing runs once and is shared by every KB.
    plan = None
    bilingual_info: Dict[str, Any] | None = None
    bilingual_mode = resolve_bilingual_mode(request.bilingual, None)
    if bilingual_applies(bilingual_mode, request.query, None):
        try:
            preprocess_rag = cast(Any, await registry.get(kb_ids[0]))
        except Exception:  # noqa: BLE001 — KB errors surface in the fan-out
            preprocess_rag = None
        if preprocess_rag is not None:
            plan = await prepare_bilingual_queries(preprocess_rag, request.query)
        if plan is None:
            bilingual_info = {
                "enabled": False,
                "mode": bilingual_mode,
                "reason": "preprocess_unavailable",
            }

    # Per-KB filters: metadata applies uniformly; doc_ids are split to the KB
    # they belong to (with union validation). None when no filters supplied.
    per_kb_filters = await _resolve_multi_kb_doc_id_filters(
        document_service, kb_ids, request.filters
    )

    secondary_counts: Dict[str, int] = {}
    secondary_failed_kbs: List[str] = []

    async def _retrieve_one(kb_id: str):
        kb_filters = per_kb_filters.get(kb_id)
        await _ensure_query_filter_documents_available(
            document_service, kb_id, kb_filters
        )
        rag = cast(Any, await registry.get(kb_id))
        # All KBs share one model service, so retrieve every KB with the SAME
        # request-level params (not each KB's own active query_config) for a
        # fair, consistent cross-KB merge.
        param = request.to_query_params()
        param.stream = False
        param.ids = await _resolve_doc_id_scope(document_service, kb_id, kb_filters)
        if plan is not None:
            apply_plan_keywords_to_param(param, plan)
            dual = await dual_aquery_data(rag, request.query, param, plan)
            secondary_counts[kb_id] = len(dual.secondary_chunks)
            if dual.secondary_failed:
                secondary_failed_kbs.append(kb_id)
            data = {"data": {"chunks": dual.merged_chunks()}}
            return rag, param, data
        data = await rag.aquery_data(request.query, param=param)
        return rag, param, data

    gathered = await asyncio.gather(
        *(_retrieve_one(kb_id) for kb_id in kb_ids), return_exceptions=True
    )

    merged: List[Dict[str, Any]] = []
    per_kb_counts: Dict[str, int] = {}
    skipped: List[Dict[str, Any]] = []
    synth_rag: Any = None
    synth_param: QueryParam | None = None
    for kb_id, outcome in zip(kb_ids, gathered):
        if isinstance(outcome, BaseException):
            # A KB mid delete/replace (409) must not be silently answered over.
            if isinstance(outcome, HTTPException) and outcome.status_code == 409:
                raise outcome
            if isinstance(outcome, KnowledgeBaseNotFoundError) or (
                isinstance(outcome, HTTPException) and outcome.status_code == 404
            ):
                skipped.append({"kb_id": kb_id, "reason": "not_found"})
            else:
                logger.error(
                    "Multi-KB retrieve failed for '%s': %s", kb_id, outcome
                )
                skipped.append({"kb_id": kb_id, "reason": "error"})
            continue
        rag, param, data = outcome
        if synth_rag is None:
            synth_rag, synth_param = rag, param
        chunks = ((data or {}).get("data", {}) or {}).get("chunks", []) or []
        per_kb_counts[kb_id] = len(chunks)
        for chunk in chunks:
            tagged = dict(chunk)
            tagged["kb_id"] = kb_id
            merged.append(tagged)

    skipped_ids = {entry["kb_id"] for entry in skipped}
    queried = [kb_id for kb_id in kb_ids if kb_id not in skipped_ids]
    if not queried:
        raise HTTPException(
            status_code=502,
            detail="All target knowledge bases failed to retrieve",
        )
    if plan is not None:
        bilingual_info = {
            "enabled": True,
            "mode": bilingual_mode,
            "source_language": plan.source_language,
            "translated_query": plan.secondary_query,
            "translation_cached": plan.from_cache,
            "per_kb_secondary_chunks": secondary_counts,
            "secondary_chunks": sum(secondary_counts.values()),
        }
        if secondary_failed_kbs:
            bilingual_info["secondary_failed_kbs"] = secondary_failed_kbs
    return (
        _dedup_chunks(merged),
        synth_rag,
        synth_param,
        queried,
        skipped,
        per_kb_counts,
        bilingual_info,
    )


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
        dependencies=[
            Depends(_reject_explicit_memory_without_final_synthesis),
            Depends(combined_auth),
        ],
        summary="Run a non-streaming RAG query against a knowledge base",
    )
    async def kb_query(kb_id: str, request: KBQueryRequest, http_request: Request):
        try:
            if request.memory is None:
                await _ensure_query_filter_documents_available(
                    document_service,
                    kb_id,
                    request.filters,
                )
            rag = cast(Any, await registry.get(kb_id))
            active_defaults = await _merge_user_query_defaults(
                http_request,
                kb_id,
                active_query_defaults_from_rag(rag),
            )
            active_metadata = active_query_metadata_from_rag(rag)
            param = request.to_query_params(
                is_stream=False,
                active_defaults=active_defaults,
            )
            _reject_memory_without_final_synthesis(request.memory, param)
            _enforce_resolved_bypass_permission(http_request, param)
            param.stream = False
            memory_handle = await _authorize_memory_handle(
                http_request,
                request.memory,
                request.query,
            )
            if memory_handle is not None:
                await _ensure_query_filter_documents_available(
                    document_service,
                    kb_id,
                    request.filters,
                )
            # Enforce per-document retrieval scoping: ``filters.doc_ids`` plus
            # the enabled/archived control-plane state are translated into a
            # ``lightrag_doc_id`` allow-list applied inside retrieval. Cross-KB
            # isolation is still guaranteed by workspace partitioning.
            param.ids = await _resolve_doc_id_scope(
                document_service,
                kb_id,
                request.filters,
            )
            plan, bilingual_info = await await_with_disconnect_check(
                http_request, _maybe_bilingual_plan(rag, request, param)
            )
            if plan is not None:
                if memory_handle is None:
                    result, bilingual_info = await await_with_disconnect_check(
                        http_request,
                        bilingual_query_llm(
                            rag, request.query, param, plan, stream=False
                        ),
                    )
                else:
                    result, bilingual_info = await await_with_disconnect_check(
                        http_request,
                        bilingual_query_llm(
                            rag,
                            request.query,
                            param,
                            plan,
                            stream=False,
                            sensitive_context=memory_handle,
                        ),
                    )
            else:
                if memory_handle is None:
                    result = await await_with_disconnect_check(
                        http_request, rag.aquery_llm(request.query, param=param)
                    )
                else:
                    result = await await_with_disconnect_check(
                        http_request,
                        rag.aquery_llm(
                            request.query,
                            param=param,
                            sensitive_context=memory_handle,
                        ),
                    )
            memory_info = memory_handle.info if memory_handle is not None else None
            llm_response = result.get("llm_response", {})
            data = result.get("data", {})
            references = data.get("references", [])
            response_text = llm_response.get("content") or "No relevant context found for the query."
            include_references = bool(param.include_references)
            if include_references and request.include_chunk_content:
                references = _enrich_with_chunk_content(
                    references, data.get("chunks", [])
                )
            response_metadata = dict(active_metadata)
            if bilingual_info is not None:
                response_metadata["bilingual"] = bilingual_info
            if memory_info is not None:
                response_metadata["memory"] = memory_info
            await append_enterprise_audit_event(
                http_request,
                "query_executed",
                target_type="kb",
                target_id=kb_id,
                metadata=_query_audit_metadata(
                    request,
                    param,
                    active_metadata,
                    route="query",
                    stream=False,
                    bilingual_info=bilingual_info,
                    memory_handle=memory_handle,
                ),
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
                metadata=response_metadata,
            )
        except HTTPException:
            raise
        except ClientGoneError:
            return client_closed_response()
        except SensitiveContextPolicyError as exc:
            raise _map_sensitive_context_policy_error(exc) from None
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            if request.memory is not None:
                logger.error(
                    "Sensitive KB query failed for '%s' (%s)",
                    kb_id,
                    type(exc).__name__,
                )
                raise _chat_memory_http_error(
                    500, _CHAT_MEMORY_INTERNAL_ERROR
                ) from None
            logger.error("KB query failed for '%s': %s", kb_id, exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post(
        "/{kb_id}/query/stream",
        dependencies=[
            Depends(_reject_explicit_memory_without_final_synthesis),
            Depends(combined_auth),
        ],
        summary="Run a streaming RAG query against a knowledge base (NDJSON)",
    )
    async def kb_query_stream(
        kb_id: str, request: KBQueryRequest, http_request: Request
    ):
        try:
            if request.memory is None:
                await _ensure_query_filter_documents_available(
                    document_service,
                    kb_id,
                    request.filters,
                )
            rag = cast(Any, await registry.get(kb_id))
            active_defaults = await _merge_user_query_defaults(
                http_request,
                kb_id,
                active_query_defaults_from_rag(rag),
            )
            active_metadata = active_query_metadata_from_rag(rag)
            stream_mode = request.stream if request.stream is not None else True
            param = request.to_query_params(
                is_stream=stream_mode,
                active_defaults=active_defaults,
            )
            _reject_memory_without_final_synthesis(request.memory, param)
            _enforce_resolved_bypass_permission(http_request, param)
            memory_handle = await _authorize_memory_handle(
                http_request,
                request.memory,
                request.query,
            )
            if memory_handle is not None:
                await _ensure_query_filter_documents_available(
                    document_service,
                    kb_id,
                    request.filters,
                )
            param.ids = await _resolve_doc_id_scope(
                document_service,
                kb_id,
                request.filters,
            )
            plan, bilingual_info = await await_with_disconnect_check(
                http_request, _maybe_bilingual_plan(rag, request, param)
            )
            if plan is not None:
                if memory_handle is None:
                    result, bilingual_info = await await_with_disconnect_check(
                        http_request,
                        bilingual_query_llm(
                            rag, request.query, param, plan, stream=stream_mode
                        ),
                    )
                else:
                    result, bilingual_info = await await_with_disconnect_check(
                        http_request,
                        bilingual_query_llm(
                            rag,
                            request.query,
                            param,
                            plan,
                            stream=stream_mode,
                            sensitive_context=memory_handle,
                        ),
                    )
            else:
                if memory_handle is None:
                    result = await await_with_disconnect_check(
                        http_request, rag.aquery_llm(request.query, param=param)
                    )
                else:
                    result = await await_with_disconnect_check(
                        http_request,
                        rag.aquery_llm(
                            request.query,
                            param=param,
                            sensitive_context=memory_handle,
                        ),
                    )
            memory_info = memory_handle.info if memory_handle is not None else None
            response_metadata = dict(active_metadata)
            if bilingual_info is not None:
                response_metadata["bilingual"] = bilingual_info
            if memory_info is not None:
                response_metadata["memory"] = memory_info
            await append_enterprise_audit_event(
                http_request,
                "query_stream_started",
                target_type="kb",
                target_id=kb_id,
                metadata=_query_audit_metadata(
                    request,
                    param,
                    active_metadata,
                    route="query_stream",
                    stream=True,
                    bilingual_info=bilingual_info,
                    memory_handle=memory_handle,
                ),
            )

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
                        "metadata": response_metadata,
                    }
                    if include_references:
                        payload["references"] = references
                    yield f"{json.dumps(payload)}\n"
                    iterator = llm_response.get("response_iterator")
                    if iterator:
                        # Drive the upstream LLM stream through a disconnect guard:
                        # a client abort stops pulling tokens promptly and the
                        # underlying response is released (aclose) on exit.
                        guarded = stream_with_disconnect_guard(
                            iterator, http_request
                        )
                        try:
                            async for chunk in guarded:
                                if chunk:
                                    yield f"{json.dumps({'response': chunk})}\n"
                        except Exception as exc:  # noqa: BLE001
                            if memory_handle is not None:
                                logger.error(
                                    "Sensitive KB stream failed (%s)",
                                    type(exc).__name__,
                                )
                                yield f"{json.dumps({'error': 'Sensitive LLM call failed'})}\n"
                                return
                            logger.error("KB stream error: %s", exc)
                            yield f"{json.dumps({'error': str(exc)})}\n"
                        finally:
                            await safe_aclose(iterator)
                else:
                    body = {
                        "kb_id": kb_id,
                        "response": llm_response.get("content", ""),
                        "metadata": response_metadata,
                    }
                    if include_references:
                        body["references"] = references
                    yield f"{json.dumps(body)}\n"

            # If the client vanished while the pre-stream work finished, the
            # generator below may never start (so never clean up) — release the
            # already-open upstream stream now and unwind via the 499 path.
            await abort_if_client_gone(
                http_request, result.get("llm_response", {}).get("response_iterator")
            )

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
        except ClientGoneError:
            return client_closed_response()
        except SensitiveContextPolicyError as exc:
            raise _map_sensitive_context_policy_error(exc) from None
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            if request.memory is not None:
                logger.error(
                    "Sensitive KB streaming query failed for '%s' (%s)",
                    kb_id,
                    type(exc).__name__,
                )
                raise _chat_memory_http_error(
                    500, _CHAT_MEMORY_INTERNAL_ERROR
                ) from None
            logger.error(
                "KB streaming query failed for '%s': %s", kb_id, exc, exc_info=True
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post(
        "/{kb_id}/query/data",
        response_model=KBQueryDataResponse,
        dependencies=[
            Depends(_reject_explicit_memory_without_final_synthesis),
            Depends(combined_auth),
        ],
        summary="Return structured retrieval data without generating an LLM answer",
    )
    async def kb_query_data(kb_id: str, request: KBQueryRequest, http_request: Request):
        try:
            _reject_memory_without_final_synthesis(
                request.memory,
                final_synthesis=False,
            )
            await _ensure_query_filter_documents_available(
                document_service,
                kb_id,
                request.filters,
            )
            rag = cast(Any, await registry.get(kb_id))
            active_defaults = await _merge_user_query_defaults(
                http_request,
                kb_id,
                active_query_defaults_from_rag(rag),
            )
            active_metadata = active_query_metadata_from_rag(rag)
            param = request.to_query_params(
                is_stream=False,
                active_defaults=active_defaults,
            )
            _enforce_resolved_bypass_permission(http_request, param)
            param.stream = False
            param.ids = await _resolve_doc_id_scope(
                document_service,
                kb_id,
                request.filters,
            )
            plan, bilingual_info = await await_with_disconnect_check(
                http_request, _maybe_bilingual_plan(rag, request, param)
            )
            if plan is not None:
                result = await await_with_disconnect_check(
                    http_request,
                    bilingual_query_data(rag, request.query, param, plan),
                )
                bilingual_info = (result.get("metadata") or {}).get("bilingual")
            else:
                result = await await_with_disconnect_check(
                    http_request, rag.aquery_data(request.query, param=param)
                )
                if bilingual_info is not None:
                    result.setdefault("metadata", {})["bilingual"] = bilingual_info
            await append_enterprise_audit_event(
                http_request,
                "retrieve_executed",
                target_type="kb",
                target_id=kb_id,
                metadata=_query_audit_metadata(
                    request,
                    param,
                    active_metadata,
                    route="retrieve"
                    if http_request.url.path.endswith("/retrieve")
                    else "query_data",
                    stream=False,
                    bilingual_info=bilingual_info,
                ),
            )
            return KBQueryDataResponse(
                kb_id=kb_id,
                status=result.get("status", "success"),
                message=result.get("message", ""),
                data=result.get("data", {}),
                metadata={**result.get("metadata", {}), **active_metadata},
            )
        except HTTPException:
            raise
        except ClientGoneError:
            return client_closed_response()
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
        dependencies=[
            Depends(_reject_explicit_memory_without_final_synthesis),
            Depends(combined_auth),
        ],
        summary="Alias for /query/data — retrieval only, no LLM generation",
    )
    async def kb_retrieve(kb_id: str, request: KBQueryRequest, http_request: Request):
        return await kb_query_data(kb_id, request, http_request)

    @router.post(
        ":query",
        response_model=MultiKBQueryResponse,
        dependencies=[
            Depends(_reject_explicit_memory_without_final_synthesis),
            Depends(combined_auth),
        ],
        summary="Run one synthesized RAG answer across multiple knowledge bases",
    )
    async def multi_kb_query(request: MultiKBQueryRequest, http_request: Request):
        try:
            if request.memory is not None and request.mode == "bypass":
                _reject_memory_without_final_synthesis(request.memory)
            memory_handle = await _authorize_memory_handle(
                http_request,
                request.memory,
                request.query,
            )
            (
                merged,
                synth_rag,
                synth_param,
                queried,
                skipped,
                per_kb_counts,
                bilingual_info,
            ) = await await_with_disconnect_check(
                http_request,
                _multi_kb_retrieve(
                    document_service, registry, request, http_request
                ),
            )

            response_text = "No relevant context found for the query."
            (
                sys_prompt,
                use_model_func,
                references_out,
                reranked,
                final_count,
            ) = await await_with_disconnect_check(
                http_request,
                _prepare_multi_kb_synthesis(
                    request,
                    merged,
                    synth_rag,
                    synth_param,
                    bilingual_info=bilingual_info,
                    sensitive_context=memory_handle,
                ),
            )
            if sys_prompt is not None and use_model_func is not None:
                llm_kwargs: Dict[str, Any] = {
                    "system_prompt": sys_prompt,
                    "history_messages": (
                        synth_param.conversation_history
                        if synth_param is not None
                        else None
                    ),
                    "enable_cot": True,
                    "stream": False,
                }
                if memory_handle is not None:
                    llm_kwargs["_sensitive"] = True
                llm_out = await await_with_disconnect_check(
                    http_request, use_model_func(request.query, **llm_kwargs)
                )
                if isinstance(llm_out, str) and llm_out.strip():
                    response_text = llm_out.strip()
            memory_info = memory_handle.info if memory_handle is not None else None

            metadata = {
                "requested_kb_count": len(request.kb_ids),
                "per_kb_chunk_counts": per_kb_counts,
                "merged_chunk_count": len(merged),
                "final_chunk_count": final_count,
                "reranked": reranked,
                "skipped_kbs": skipped,
                "synthesis_kb_id": queried[0] if queried else None,
            }
            if bilingual_info is not None:
                metadata["bilingual"] = bilingual_info

            await append_enterprise_audit_event(
                http_request,
                "multi_kb_query_executed",
                target_type="kb_group",
                target_id=None,
                metadata=_multi_kb_query_audit_metadata(
                    request,
                    synth_param,
                    kb_ids=queried,
                    skipped=skipped,
                    reranked=reranked,
                    final_count=final_count,
                    bilingual_info=bilingual_info,
                    memory_handle=memory_handle,
                ),
            )
            return MultiKBQueryResponse(
                kb_ids=queried,
                mode=cast(QueryMode, request.mode),
                response=response_text,
                references=references_out if request.include_references else None,
                metadata={**metadata, **({"memory": memory_info} if memory_info else {})},
            )
        except HTTPException:
            raise
        except ClientGoneError:
            return client_closed_response()
        except SensitiveContextPolicyError as exc:
            raise _map_sensitive_context_policy_error(exc) from None
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            if request.memory is not None:
                logger.error(
                    "Sensitive multi-KB query failed (%s)",
                    type(exc).__name__,
                )
                raise _chat_memory_http_error(
                    500, _CHAT_MEMORY_INTERNAL_ERROR
                ) from None
            logger.error("Multi-KB query failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post(
        ":query/stream",
        dependencies=[
            Depends(_reject_explicit_memory_without_final_synthesis),
            Depends(combined_auth),
        ],
        summary="Stream one synthesized RAG answer across multiple KBs (NDJSON)",
    )
    async def multi_kb_query_stream(
        request: MultiKBQueryRequest, http_request: Request
    ):
        try:
            if request.memory is not None and request.mode == "bypass":
                _reject_memory_without_final_synthesis(request.memory)
            memory_handle = await _authorize_memory_handle(
                http_request,
                request.memory,
                request.query,
            )
            (
                merged,
                synth_rag,
                synth_param,
                queried,
                skipped,
                per_kb_counts,
                bilingual_info,
            ) = await await_with_disconnect_check(
                http_request,
                _multi_kb_retrieve(
                    document_service, registry, request, http_request
                ),
            )
            (
                sys_prompt,
                use_model_func,
                references_out,
                reranked,
                final_count,
            ) = await await_with_disconnect_check(
                http_request,
                _prepare_multi_kb_synthesis(
                    request,
                    merged,
                    synth_rag,
                    synth_param,
                    bilingual_info=bilingual_info,
                    sensitive_context=memory_handle,
                ),
            )
            prepared_llm_out: Any = None
            if (
                memory_handle is not None
                and sys_prompt is not None
                and use_model_func is not None
            ):
                prepared_llm_out = await await_with_disconnect_check(
                    http_request,
                    use_model_func(
                        request.query,
                        system_prompt=sys_prompt,
                        history_messages=(
                            synth_param.conversation_history
                            if synth_param is not None
                            else None
                        ),
                        enable_cot=True,
                        stream=True,
                        _sensitive=True,
                    ),
                )
            memory_info = memory_handle.info if memory_handle is not None else None
            metadata = {
                "requested_kb_count": len(request.kb_ids),
                "per_kb_chunk_counts": per_kb_counts,
                "merged_chunk_count": len(merged),
                "final_chunk_count": final_count,
                "reranked": reranked,
                "skipped_kbs": skipped,
                "synthesis_kb_id": queried[0] if queried else None,
            }
            if bilingual_info is not None:
                metadata["bilingual"] = bilingual_info
            if memory_info is not None:
                metadata["memory"] = memory_info
            await append_enterprise_audit_event(
                http_request,
                "multi_kb_query_stream_started",
                target_type="kb_group",
                target_id=None,
                metadata=_multi_kb_query_audit_metadata(
                    request,
                    synth_param,
                    kb_ids=queried,
                    skipped=skipped,
                    reranked=reranked,
                    final_count=final_count,
                    bilingual_info=bilingual_info,
                    memory_handle=memory_handle,
                ),
            )

            async def stream_generator():
                head: Dict[str, Any] = {"kb_ids": queried, "metadata": metadata}
                if request.include_references:
                    head["references"] = [ref.model_dump() for ref in references_out]
                yield f"{json.dumps(head)}\n"
                if sys_prompt is None or use_model_func is None:
                    yield (
                        f"{json.dumps({'response': 'No relevant context found for the query.'})}\n"
                    )
                    return
                if memory_handle is not None:
                    llm_out = prepared_llm_out
                else:
                    try:
                        llm_out = await await_with_disconnect_check(
                            http_request,
                            use_model_func(
                                request.query,
                                system_prompt=sys_prompt,
                                history_messages=(
                                    synth_param.conversation_history
                                    if synth_param is not None
                                    else None
                                ),
                                enable_cot=True,
                                stream=True,
                            ),
                        )
                    except ClientGoneError:
                        # The head line already went out, so a 499 is impossible
                        # here — end the body quietly instead of letting the
                        # error escape the generator into the server log.
                        return
                if isinstance(llm_out, str):
                    if llm_out.strip():
                        yield f"{json.dumps({'response': llm_out.strip()})}\n"
                    return
                try:
                    # Disconnect guard: a client abort stops pulling tokens and the
                    # upstream stream is released (aclose) on exit.
                    guarded = stream_with_disconnect_guard(llm_out, http_request)
                    async for chunk in guarded:
                        if chunk:
                            yield f"{json.dumps({'response': chunk})}\n"
                except Exception as exc:  # noqa: BLE001
                    if memory_handle is not None:
                        logger.error(
                            "Sensitive multi-KB stream failed (%s)",
                            type(exc).__name__,
                        )
                        yield f"{json.dumps({'error': 'Sensitive LLM call failed'})}\n"
                        return
                    logger.error("Multi-KB stream error: %s", exc)
                    yield f"{json.dumps({'error': str(exc)})}\n"
                finally:
                    await safe_aclose(llm_out)

            # If the client vanished while the pre-stream work finished, the
            # generator below may never start (so never clean up) — release the
            # pre-opened synthesis stream now and unwind via the 499 path.
            await abort_if_client_gone(http_request, prepared_llm_out)

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
        except ClientGoneError:
            return client_closed_response()
        except SensitiveContextPolicyError as exc:
            raise _map_sensitive_context_policy_error(exc) from None
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            if request.memory is not None:
                logger.error(
                    "Sensitive multi-KB streaming query failed (%s)",
                    type(exc).__name__,
                )
                raise _chat_memory_http_error(
                    500, _CHAT_MEMORY_INTERNAL_ERROR
                ) from None
            logger.error("Multi-KB streaming query failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post(
        ":retrieve",
        response_model=MultiKBQueryDataResponse,
        dependencies=[
            Depends(_reject_explicit_memory_without_final_synthesis),
            Depends(combined_auth),
        ],
        summary="Retrieve and merge chunks across multiple knowledge bases (no LLM)",
    )
    async def multi_kb_retrieve(request: MultiKBQueryRequest, http_request: Request):
        try:
            _reject_memory_without_final_synthesis(
                request.memory,
                final_synthesis=False,
            )
            (
                merged,
                synth_rag,
                synth_param,
                queried,
                skipped,
                per_kb_counts,
                bilingual_info,
            ) = await await_with_disconnect_check(
                http_request,
                _multi_kb_retrieve(
                    document_service, registry, request, http_request
                ),
            )

            reranked = False
            processed: List[Dict[str, Any]] = merged
            if merged and synth_rag is not None and synth_param is not None:
                global_config = synth_rag._build_global_config()
                processed = await process_chunks_unified(
                    query=request.query,
                    unique_chunks=merged,
                    query_param=synth_param,
                    global_config=global_config,
                    source_type="multi_kb",
                    chunk_token_limit=None,
                )
                reranked = bool(global_config.get("rerank_model_func")) and bool(
                    synth_param.enable_rerank
                )

            reference_list, processed_with_ids = (
                generate_reference_list_from_chunks(processed)
                if processed
                else ([], [])
            )
            ref_kb: Dict[str, str] = {}
            for chunk in processed_with_ids:
                rid = chunk.get("reference_id")
                if rid:
                    ref_kb.setdefault(rid, chunk.get("kb_id", ""))
            references = [
                {
                    "reference_id": ref["reference_id"],
                    "file_path": ref["file_path"],
                    "kb_id": ref_kb.get(ref["reference_id"], ""),
                }
                for ref in reference_list
                if ref["reference_id"]
            ]
            metadata = {
                "requested_kb_count": len(request.kb_ids),
                "per_kb_chunk_counts": per_kb_counts,
                "merged_chunk_count": len(merged),
                "final_chunk_count": len(processed_with_ids),
                "reranked": reranked,
                "skipped_kbs": skipped,
            }
            if bilingual_info is not None:
                metadata["bilingual"] = bilingual_info
            await append_enterprise_audit_event(
                http_request,
                "multi_kb_retrieve_executed",
                target_type="kb_group",
                target_id=None,
                metadata=_multi_kb_query_audit_metadata(
                    request,
                    synth_param,
                    kb_ids=queried,
                    skipped=skipped,
                    reranked=reranked,
                    final_count=len(processed_with_ids),
                    bilingual_info=bilingual_info,
                ),
            )
            return MultiKBQueryDataResponse(
                kb_ids=queried,
                status="success",
                message="ok",
                data={"chunks": processed_with_ids, "references": references},
                metadata=metadata,
            )
        except HTTPException:
            raise
        except ClientGoneError:
            return client_closed_response()
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("Multi-KB retrieve failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return router
