from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Literal, Optional, cast

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from lightrag.api.bilingual_query_service import (
    alt_retrieval_param,
    usable_alt_query,
)
from lightrag.api.config_version_service import (
    active_query_defaults_from_rag,
    active_query_metadata_from_rag,
)
from lightrag.api.document_lifecycle_service import DocumentLifecycleService
from lightrag.api.enterprise_auth import (
    INTERACTIVE_AUTH_METHODS,
    KB_ROLE_VIEWER,
    UserKBQuerySettingsService,
    enterprise_auth_enabled,
    get_enterprise_authorization_service,
    get_request_principal,
)
from lightrag.api.kb_service import KnowledgeBaseNotFoundError
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry
from lightrag.api.metadata_store import DocumentRecord
from lightrag.base import QueryParam
from lightrag.utils import logger

QueryMode = Literal["local", "global", "hybrid", "naive", "mix", "bypass"]
_QUERY_BLOCKING_DOCUMENT_STATUSES = {
    "deleting": "delete_job_active",
    "replacing": "replace_job_active",
}
_MAX_METADATA_FILTER_BYTES = 64 * 1024
_METADATA_FILTER_SCALAR_TYPES = (str, int, float, bool, type(None))


class KBQueryFilters(BaseModel):
    doc_ids: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

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
    query: str = Field(min_length=3)
    mode: QueryMode = "mix"
    only_need_context: bool | None = None
    only_need_prompt: bool | None = None
    response_type: str | None = None
    top_k: int | None = Field(default=None, ge=1)
    chunk_top_k: int | None = Field(default=None, ge=1)
    max_entity_tokens: int | None = Field(default=None, ge=1)
    max_relation_tokens: int | None = Field(default=None, ge=1)
    max_total_tokens: int | None = Field(default=None, ge=1)
    hl_keywords: list[str] = Field(default_factory=list)
    ll_keywords: list[str] = Field(default_factory=list)
    conversation_history: list[dict[str, Any]] | None = None
    user_prompt: str | None = None
    enable_rerank: bool | None = None
    include_references: bool | None = True
    include_chunk_content: bool | None = False
    stream: bool | None = True
    filters: KBQueryFilters | None = None

    def to_query_params(self, *, is_stream: bool, active_defaults: dict[str, Any] | None = None) -> QueryParam:
        route_only_fields = {"query", "include_chunk_content", "filters"}
        request_data = self.model_dump(exclude_none=True, exclude=route_only_fields)
        explicit_fields = self.model_fields_set - route_only_fields
        data = dict(request_data)
        for key, value in (active_defaults or {}).items():
            if key not in route_only_fields and key not in explicit_fields:
                data[key] = value
        param = QueryParam(**data)
        param.stream = is_stream
        return param


async def _merge_user_query_defaults(
    request: Request,
    kb_id: str,
    active_defaults: dict[str, Any],
) -> dict[str, Any]:
    defaults = dict(active_defaults)
    if not enterprise_auth_enabled():
        return defaults
    principal = get_request_principal(request)
    if principal is None or principal.auth_method not in INTERACTIVE_AUTH_METHODS:
        return defaults
    service = getattr(request.app.state, "enterprise_user_kb_query_settings_service", None)
    if not isinstance(service, UserKBQuerySettingsService):
        return defaults
    settings = await service.get_settings(principal.user_id, kb_id)
    if settings is not None and settings.user_prompt:
        defaults["user_prompt"] = settings.user_prompt
    return defaults


def _has_metadata_filters(filters: KBQueryFilters | None) -> bool:
    return bool(filters and filters.metadata)


def _has_doc_id_filter(filters: KBQueryFilters | None) -> bool:
    return bool(filters and filters.doc_ids is not None)


async def _list_all_kb_documents(
    document_service: DocumentLifecycleService,
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


def _validate_doc_ids_from_documents(all_documents: list[DocumentRecord], doc_ids: list[str]) -> None:
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


def _raise_active_lifecycle_query_conflict(document: DocumentRecord) -> None:
    status = str(document.status)
    error_code = _QUERY_BLOCKING_DOCUMENT_STATUSES[status]
    raise HTTPException(
        status_code=409,
        detail={
            "error_code": error_code,
            "document_id": document.id,
            "existing_job_id": str(
                document.metadata.get("current_delete_job_id")
                or document.metadata.get("pending_delete_job_id")
                or document.metadata.get("current_replace_job_id")
                or document.metadata.get("pending_replace_job_id")
                or "unknown"
            ),
            "message": f"Document '{document.id}' is currently {status}",
        },
    )


async def _ensure_query_filter_documents_available(
    document_service: DocumentLifecycleService,
    kb_id: str,
    filters: KBQueryFilters | None,
) -> None:
    all_documents = await _list_all_kb_documents(document_service, kb_id)
    if _has_doc_id_filter(filters) and filters:
        _validate_doc_ids_from_documents(all_documents, filters.doc_ids or [])
    candidates = (
        _effective_candidate_documents(all_documents, filters)
        if (_has_doc_id_filter(filters) or _has_metadata_filters(filters))
        else all_documents
    )
    for document in candidates:
        if document.status in _QUERY_BLOCKING_DOCUMENT_STATUSES:
            _raise_active_lifecycle_query_conflict(document)


async def _resolve_doc_id_scope(
    document_service: DocumentLifecycleService,
    kb_id: str,
    filters: KBQueryFilters | None,
) -> list[str] | None:
    all_documents = await _list_all_kb_documents(document_service, kb_id)
    if _has_doc_id_filter(filters) and filters:
        _validate_doc_ids_from_documents(all_documents, filters.doc_ids or [])

    def _retrievable(document: DocumentRecord) -> bool:
        return document.enabled and not document.archived and bool(document.lightrag_doc_id)

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
    return [str(document.lightrag_doc_id) for document in all_documents if _retrievable(document)]


async def _resolve_multi_kb_doc_id_filters(
    document_service: DocumentLifecycleService,
    kb_ids: list[str],
    filters: KBQueryFilters | None,
) -> dict[str, KBQueryFilters | None]:
    if not _has_doc_id_filter(filters):
        return {kb_id: filters for kb_id in kb_ids}
    requested = list(filters.doc_ids or []) if filters else []
    per_kb: dict[str, KBQueryFilters | None] = {}
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


def _dedup_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set = set()
    result: list[dict[str, Any]] = []
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if chunk_id:
            key = ("id", chunk.get("kb_id"), chunk.get("file_path"), chunk_id)
        else:
            key = ("content", chunk.get("kb_id"), chunk.get("content"))
        if key in seen:
            continue
        seen.add(key)
        result.append(chunk)
    return result


@dataclass(slots=True)
class QueryToolResult:
    chunks: list[dict[str, Any]]
    rag: Any | None
    param: QueryParam | None
    queried_kb_ids: list[str]
    skipped_kbs: list[dict[str, Any]] = field(default_factory=list)
    per_kb_chunk_counts: dict[str, int] = field(default_factory=dict)
    active_metadata: dict[str, Any] = field(default_factory=dict)
    alt_chunk_counts: dict[str, int] = field(default_factory=dict)
    alt_failed_kbs: list[str] = field(default_factory=list)


class QueryToolService:
    """Safe internal retrieve tools used by server-side Agent orchestration.

    This class deliberately reuses the same KB route helpers for RBAC, document
    lifecycle checks, metadata/doc-id filters, and active query defaults so Agent
    planning cannot bypass controls by calling LightRAG instances directly.
    """

    def __init__(
        self,
        document_service: DocumentLifecycleService,
        registry: LightRAGInstanceRegistry,
    ):
        self._document_service = document_service
        self._registry = registry

    async def get_rag(self, kb_id: str) -> Any:
        return cast(Any, await self._registry.get(kb_id))

    async def retrieve_serial(
        self,
        *,
        http_request: Request,
        kb_ids: list[str],
        query: str,
        mode: QueryMode,
        filters: KBQueryFilters | None = None,
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        max_entity_tokens: int | None = None,
        max_relation_tokens: int | None = None,
        max_total_tokens: int | None = None,
        enable_rerank: bool | None = None,
        hl_keywords: list[str] | None = None,
        ll_keywords: list[str] | None = None,
        query_alt: str | None = None,
        hl_keywords_alt: list[str] | None = None,
        ll_keywords_alt: list[str] | None = None,
    ) -> QueryToolResult:
        if not kb_ids:
            raise HTTPException(status_code=400, detail="At least one KB is required")
        if mode == "bypass":
            raise HTTPException(status_code=400, detail="bypass mode is not supported for Agent query")

        # Bilingual dual-path: a usable alternate-language query triggers a
        # second retrieval per KB whose chunks join the merged pool. The alt
        # path is best-effort — its failure never fails the step.
        alt_query = usable_alt_query(query, query_alt)

        if enterprise_auth_enabled():
            principal = get_request_principal(http_request)
            authz = get_enterprise_authorization_service(http_request)
            for kb_id in kb_ids:
                await authz.require_kb_role(principal, kb_id, KB_ROLE_VIEWER)

        per_kb_filters = await _resolve_multi_kb_doc_id_filters(
            self._document_service, kb_ids, filters
        )

        merged: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        queried: list[str] = []
        per_kb_counts: dict[str, int] = {}
        alt_counts: dict[str, int] = {}
        alt_failed: list[str] = []
        synth_rag: Any | None = None
        synth_param: QueryParam | None = None
        active_metadata: dict[str, Any] = {}

        for kb_id in kb_ids:
            try:
                kb_filters = per_kb_filters.get(kb_id)
                await _ensure_query_filter_documents_available(
                    self._document_service, kb_id, kb_filters
                )
                rag = cast(Any, await self._registry.get(kb_id))
                active_defaults = await _merge_user_query_defaults(
                    http_request,
                    kb_id,
                    active_query_defaults_from_rag(rag),
                )
                request = KBQueryRequest(
                    query=query,
                    mode=mode,
                    top_k=top_k,
                    chunk_top_k=chunk_top_k,
                    max_entity_tokens=max_entity_tokens,
                    max_relation_tokens=max_relation_tokens,
                    max_total_tokens=max_total_tokens,
                    hl_keywords=hl_keywords or [],
                    ll_keywords=ll_keywords or [],
                    enable_rerank=enable_rerank,
                    include_references=True,
                    include_chunk_content=True,
                    stream=False,
                    filters=kb_filters,
                )
                param = request.to_query_params(
                    is_stream=False,
                    active_defaults=active_defaults,
                )
                if param.mode == "bypass":
                    raise HTTPException(
                        status_code=400,
                        detail="bypass mode is not supported for Agent query",
                    )
                param.stream = False
                param.ids = await _resolve_doc_id_scope(
                    self._document_service,
                    kb_id,
                    kb_filters,
                )
                data = await rag.aquery_data(query, param=param)
            except HTTPException as exc:
                if exc.status_code == 409:
                    raise
                if exc.status_code == 404:
                    skipped.append({"kb_id": kb_id, "reason": "not_found"})
                    continue
                raise
            except KnowledgeBaseNotFoundError:
                skipped.append({"kb_id": kb_id, "reason": "not_found"})
                continue
            except Exception as exc:  # noqa: BLE001
                logger.error("Agent retrieve failed for '%s': %s", kb_id, exc, exc_info=True)
                skipped.append({"kb_id": kb_id, "reason": "error"})
                continue

            alt_chunks: list[dict[str, Any]] = []
            if alt_query is not None:
                try:
                    alt_param = alt_retrieval_param(
                        param, alt_query, hl_keywords_alt, ll_keywords_alt
                    )
                    alt_data = await rag.aquery_data(alt_query, param=alt_param)
                    alt_chunks = (
                        ((alt_data or {}).get("data", {}) or {}).get("chunks", []) or []
                    )
                    alt_counts[kb_id] = len(alt_chunks)
                except Exception as exc:  # noqa: BLE001 — alt path is best-effort
                    logger.warning(
                        "Agent bilingual alt retrieval failed for '%s': %s", kb_id, exc
                    )
                    alt_failed.append(kb_id)

            if synth_rag is None:
                synth_rag = rag
                synth_param = param
                active_metadata = active_query_metadata_from_rag(rag)
            queried.append(kb_id)
            chunks = ((data or {}).get("data", {}) or {}).get("chunks", []) or []
            per_kb_counts[kb_id] = len(chunks)
            for chunk in chunks:
                tagged = dict(chunk)
                tagged["kb_id"] = kb_id
                merged.append(tagged)
            for chunk in alt_chunks:
                tagged = dict(chunk)
                tagged["kb_id"] = kb_id
                tagged.setdefault("retrieval_path", "secondary")
                merged.append(tagged)

        if not queried:
            raise HTTPException(status_code=502, detail="All target knowledge bases failed to retrieve")

        return QueryToolResult(
            chunks=_dedup_chunks(merged),
            rag=synth_rag,
            param=synth_param,
            queried_kb_ids=queried,
            skipped_kbs=skipped,
            per_kb_chunk_counts=per_kb_counts,
            active_metadata=active_metadata,
            alt_chunk_counts=alt_counts,
            alt_failed_kbs=alt_failed,
        )
