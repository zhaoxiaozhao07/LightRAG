"""Chat-memory authorization shared by query and Agent endpoints.

``authorize_memory_context`` validates request ownership without searching and
returns a process-local handle that Phase 4B call sites resolve only at final
synthesis. ``resolve_memory_injection`` remains as a compatibility wrapper for
the existing pre-Phase-4B routes.

Failure semantics:

- ``memory`` omitted        → untouched request, byte-identical responses.
- feature disabled          → 503 (explicit client intent hit a config gap).
- non-interactive principal → 403 (memory partitions are user-owned).
- foreign/missing project   → 404 (no existence leak).
- backend unavailable       → fail-open: no injection, ``metadata.memory``
  reports ``{"enabled": false, "reason": "unavailable"}``.
"""

from __future__ import annotations

from typing import Any, cast

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field

from lightrag.api.chat_memory_service import (
    MEMORY_QUERY_MAX_LENGTH,
    MEMORY_SEARCH_MAX_LIMIT,
    AuthorizedChatMemoryHandle,
)
from lightrag.api.enterprise_auth import (
    INTERACTIVE_AUTH_METHODS,
    get_enterprise_chat_memory_service,
    get_request_principal,
)


_MEMORY_AUDIT_ALLOWED_KEYS = frozenset(
    {
        "memory_enabled",
        "memory_fact_count",
        "memory_injected_count",
        "memory_status",
        "memory_truncated",
        "memory_reason",
    }
)
_MEMORY_AUDIT_STATUS_VALUES = frozenset(
    {"injected", "empty", "budget_exhausted", "unavailable", "not_used"}
)
_MEMORY_AUDIT_REASON_VALUES = frozenset(
    {"unavailable", "clarification_required", "no_kb_evidence"}
)


class ChatMemoryScope(BaseModel):
    """Request opt-in for server-side memory injection."""

    project_id: str = Field(min_length=1, max_length=128)
    # Omitted limit falls back to the deployment default (MEMORY_SEARCH_LIMIT).
    limit: int | None = Field(default=None, ge=1, le=MEMORY_SEARCH_MAX_LIMIT)


async def authorize_memory_context(
    request: Request,
    scope: ChatMemoryScope | None,
    query: str,
    *,
    query_llm_endpoint: str | None = None,
) -> AuthorizedChatMemoryHandle | None:
    """Authorize a fact-free handle without searching Chat Memory.

    Phase 4B call sites bind (or pass) the exact final-synthesis endpoint and
    invoke the handle only after authoritative KB/Agent evidence is complete.
    """

    if scope is None:
        return None
    if len(query) > MEMORY_QUERY_MAX_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Chat memory query exceeds {MEMORY_QUERY_MAX_LENGTH} characters",
        )
    service = get_enterprise_chat_memory_service(request)
    if service is None:
        raise HTTPException(status_code=503, detail="Chat memory is not enabled")
    principal = get_request_principal(request)
    if principal is None:
        raise HTTPException(status_code=401, detail="Login required")
    if principal.auth_method not in INTERACTIVE_AUTH_METHODS:
        raise HTTPException(
            status_code=403,
            detail="Chat memory requires an interactive user",
        )
    conversation = getattr(
        request.app.state, "enterprise_chat_conversation_service", None
    )
    if conversation is None:
        raise HTTPException(
            status_code=500,
            detail="Enterprise chat conversation service unavailable",
        )
    project = await conversation.get_project(principal.user_id, scope.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Chat project not found")
    create_handle = getattr(service, "create_authorized_handle", None)
    if callable(create_handle):
        return cast(
            AuthorizedChatMemoryHandle,
            create_handle(
                user_id=principal.user_id,
                project_id=scope.project_id,
                query=query,
                limit=scope.limit,
                query_llm_endpoint=query_llm_endpoint,
            ),
        )
    return AuthorizedChatMemoryHandle(
        service,
        user_id=principal.user_id,
        project_id=scope.project_id,
        query=query,
        limit=scope.limit,
        query_llm_endpoint=query_llm_endpoint,
    )


# Explicit alias for Phase 4B integration lanes.
authorize_chat_memory_context = authorize_memory_context


async def resolve_memory_injection(
    request: Request, scope: ChatMemoryScope | None, query: str
) -> tuple[str | None, dict[str, Any] | None]:
    """Resolve the memory block for one query request.

    Returns ``(memory_block, memory_info)``; both ``None`` when the request
    did not opt in. ``memory_block`` is ``None`` with an info dict when there
    are no facts or the backend is temporarily unavailable (fail-open).
    """
    if scope is None:
        return None, None
    service = get_enterprise_chat_memory_service(request)
    if service is None:
        raise HTTPException(status_code=503, detail="Chat memory is not enabled")
    principal = get_request_principal(request)
    if principal is None:
        raise HTTPException(status_code=401, detail="Login required")
    if principal.auth_method not in INTERACTIVE_AUTH_METHODS:
        raise HTTPException(
            status_code=403,
            detail="Chat memory requires an interactive user",
        )
    conversation = getattr(
        request.app.state, "enterprise_chat_conversation_service", None
    )
    if conversation is None:
        raise HTTPException(
            status_code=500,
            detail="Enterprise chat conversation service unavailable",
        )
    project = await conversation.get_project(principal.user_id, scope.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Chat project not found")
    block, info = await service.build_memory_block(
        user_id=principal.user_id,
        project_id=scope.project_id,
        query=query,
        limit=scope.limit,
    )
    return block, info


def merge_memory_block(block: str | None, user_prompt: str | None) -> str | None:
    """Prepend the memory block to an (optional) effective user prompt."""
    if not block:
        return user_prompt
    if user_prompt:
        return f"{block}\n\n{user_prompt}"
    return block


def memory_audit_fields(info: dict[str, Any] | None) -> dict[str, Any]:
    """Whitelisted audit fields — ids/counts only, never fact text."""
    if info is None:
        return {}
    fields: dict[str, Any] = {"memory_enabled": bool(info.get("enabled"))}
    if "fact_count" in info:
        fields["memory_fact_count"] = info["fact_count"]
    if "injected_count" in info:
        fields["memory_injected_count"] = info["injected_count"]
    status = info.get("status")
    if status.__class__ is str and status in _MEMORY_AUDIT_STATUS_VALUES:
        fields["memory_status"] = status
    if "truncated" in info:
        fields["memory_truncated"] = bool(info["truncated"])
    reason = info.get("reason")
    if reason.__class__ is str and reason in _MEMORY_AUDIT_REASON_VALUES:
        fields["memory_reason"] = reason
    return {
        key: value
        for key, value in fields.items()
        if key in _MEMORY_AUDIT_ALLOWED_KEYS
    }
