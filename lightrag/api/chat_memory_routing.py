"""Server-side chat-memory injection shared by query and agent endpoints.

Endpoints opt in per request with ``memory: {"project_id": "...", "limit": n}``.
The server validates that the caller is an interactive user who owns the chat
project, searches the project's graphiti memory partition with the request
query, and prepends the formatted fact block to the effective ``user_prompt``.
See docs/ChatMemory-zh.md §6.

Failure semantics:

- ``memory`` omitted        → untouched request, byte-identical responses.
- feature disabled          → 503 (explicit client intent hit a config gap).
- non-interactive principal → 403 (memory partitions are user-owned).
- foreign/missing project   → 404 (no existence leak).
- backend unavailable       → fail-open: no injection, ``metadata.memory``
  reports ``{"enabled": false, "reason": "unavailable"}``.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field

from lightrag.api.chat_memory_service import MEMORY_SEARCH_MAX_LIMIT
from lightrag.api.enterprise_auth import (
    get_enterprise_chat_memory_service,
    get_request_principal,
)


class ChatMemoryScope(BaseModel):
    """Request opt-in for server-side memory injection."""

    project_id: str = Field(min_length=1, max_length=128)
    # Omitted limit falls back to the deployment default (MEMORY_SEARCH_LIMIT).
    limit: int | None = Field(default=None, ge=1, le=MEMORY_SEARCH_MAX_LIMIT)


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
    if principal.auth_method != "jwt":
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
    if "project_id" in info:
        fields["memory_project_id"] = info["project_id"]
    if "fact_count" in info:
        fields["memory_fact_count"] = info["fact_count"]
    if "reason" in info:
        fields["memory_reason"] = info["reason"]
    return fields
