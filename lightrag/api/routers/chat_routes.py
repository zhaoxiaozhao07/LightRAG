"""Per-user chat conversation management routes (projects + sessions + messages).

Hierarchy: user > project > session > message — one user owns many projects,
one project holds many sessions, one session holds the persisted Q&A messages
so history syncs across browsers/devices. Records live in the KB control-plane
metadata store (SQLite in ``local`` mode, PostgreSQL with
``LIGHTRAG_KB_METADATA_BACKEND=postgres``) and never touch LightRAG engine
storage.

The router is mounted only in enterprise mode and every endpoint is restricted
to interactive JWT users; all reads/writes are scoped to the authenticated
user's own records, so a missing or foreign resource uniformly yields ``404``.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from lightrag.api.chat_memory_service import (
    MEMORY_QUERY_MAX_LENGTH,
    MEMORY_SEARCH_MAX_LIMIT,
    ChatMemoryUnavailableError,
)
from lightrag.api.enterprise_auth import (
    ChatMessageRecord,
    ChatProjectRecord,
    ChatSessionRecord,
    Principal,
    get_enterprise_chat_conversation_service,
    get_enterprise_chat_memory_service,
    get_request_principal,
)
from lightrag.api.utils_api import get_combined_auth_dependency

CHAT_NAME_MAX_LENGTH = 256
CHAT_MESSAGE_CONTENT_MAX_LENGTH = 1_000_000
CHAT_MESSAGE_METADATA_MAX_BYTES = 64 * 1024
CHAT_MESSAGES_MAX_BATCH = 20


class ChatProjectCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=CHAT_NAME_MAX_LENGTH)


class ChatProjectUpdateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=CHAT_NAME_MAX_LENGTH)


class ChatProjectResponse(BaseModel):
    id: str
    user_id: str
    name: str
    created_at: str
    updated_at: str


class ChatProjectListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    projects: list[ChatProjectResponse]


class ChatProjectDeleteResponse(BaseModel):
    id: str
    deleted: bool
    deleted_sessions: int
    deleted_messages: int


class ChatSessionCreateRequest(BaseModel):
    # Blank/omitted name falls back to a creation-time default on the server.
    name: str | None = Field(default=None, max_length=CHAT_NAME_MAX_LENGTH)
    # Conversation rounds sent to the LLM (-1 = full history); omitted falls
    # back to the deployment default (CHAT_SESSION_DEFAULT_CONTEXT_ROUNDS).
    context_rounds: int | None = None


class ChatSessionUpdateRequest(BaseModel):
    name: str | None = Field(
        default=None, min_length=1, max_length=CHAT_NAME_MAX_LENGTH
    )
    context_rounds: int | None = None


class ChatSessionResponse(BaseModel):
    id: str
    project_id: str
    user_id: str
    name: str
    context_rounds: int
    created_at: str
    updated_at: str


class ChatSessionListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    sessions: list[ChatSessionResponse]


class ChatSessionDeleteResponse(BaseModel):
    id: str
    project_id: str
    deleted: bool
    deleted_messages: int


class ChatMessageAppendItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=CHAT_MESSAGE_CONTENT_MAX_LENGTH)
    # Free-form client payload (e.g. references/citations shown with an
    # answer, query mode, agent session id); serialized size is capped.
    metadata: dict[str, Any] | None = None


class ChatMessagesAppendRequest(BaseModel):
    messages: list[ChatMessageAppendItem] = Field(
        min_length=1, max_length=CHAT_MESSAGES_MAX_BATCH
    )


class ChatMessageResponse(BaseModel):
    id: str
    session_id: str
    project_id: str
    user_id: str
    role: str
    content: str
    metadata: dict[str, Any]
    seq: int
    created_at: str


class ChatMessagesAppendResponse(BaseModel):
    session_id: str
    project_id: str
    messages: list[ChatMessageResponse]


class ChatMessageListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    messages: list[ChatMessageResponse]


class ChatMessageDeleteResponse(BaseModel):
    id: str
    session_id: str
    project_id: str
    deleted: bool


class ChatMemorySearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MEMORY_QUERY_MAX_LENGTH)
    # Omitted limit falls back to the deployment default (MEMORY_SEARCH_LIMIT).
    limit: int | None = Field(default=None, ge=1, le=MEMORY_SEARCH_MAX_LIMIT)


class ChatMemoryFact(BaseModel):
    uuid: str
    name: str
    fact: str
    # valid_at/invalid_at are fact-world validity; a non-null invalid_at marks
    # a fact that has since been superseded (kept for history).
    valid_at: str | None = None
    invalid_at: str | None = None
    created_at: str | None = None
    expired_at: str | None = None


class ChatMemorySearchResponse(BaseModel):
    project_id: str
    total: int
    facts: list[ChatMemoryFact]


class ChatMemoryOverviewResponse(BaseModel):
    project_id: str
    enabled: bool
    available: bool
    episode_count: int
    last_ingested_at: str | None = None


def _project_response(record: ChatProjectRecord) -> ChatProjectResponse:
    return ChatProjectResponse(
        id=record.id,
        user_id=record.user_id,
        name=record.name,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _session_response(record: ChatSessionRecord) -> ChatSessionResponse:
    return ChatSessionResponse(
        id=record.id,
        project_id=record.project_id,
        user_id=record.user_id,
        name=record.name,
        context_rounds=record.context_rounds,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _message_response(record: ChatMessageRecord) -> ChatMessageResponse:
    return ChatMessageResponse(
        id=record.id,
        session_id=record.session_id,
        project_id=record.project_id,
        user_id=record.user_id,
        role=record.role,
        content=record.content,
        metadata=record.metadata,
        seq=record.seq,
        created_at=record.created_at,
    )


def _required_name(name: str) -> str:
    stripped = name.strip()
    if not stripped:
        raise HTTPException(status_code=400, detail="Name must not be blank")
    return stripped


def _validated_message_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    serialized = json.dumps(metadata, ensure_ascii=False)
    if len(serialized.encode("utf-8")) > CHAT_MESSAGE_METADATA_MAX_BYTES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Message metadata exceeds "
                f"{CHAT_MESSAGE_METADATA_MAX_BYTES} bytes when serialized"
            ),
        )
    return metadata


def _validated_context_rounds(value: int) -> int:
    if value == -1 or value >= 1:
        return value
    raise HTTPException(
        status_code=400,
        detail="context_rounds must be -1 (full history) or a positive integer",
    )


def create_chat_routes(api_key: str | None = None) -> APIRouter:
    router = APIRouter(prefix="/chat", tags=["chat"])
    combined_auth = get_combined_auth_dependency(api_key)

    def require_interactive_user_principal(request: Request) -> Principal:
        principal = get_request_principal(request)
        if principal is None:
            raise HTTPException(status_code=401, detail="Login required")
        if principal.auth_method != "jwt":
            raise HTTPException(
                status_code=403,
                detail="Only available for interactive users",
            )
        return principal

    @router.post(
        "/projects",
        response_model=ChatProjectResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def create_chat_project(request: Request, body: ChatProjectCreateRequest):
        principal = require_interactive_user_principal(request)
        project = await get_enterprise_chat_conversation_service(
            request
        ).create_project(
            user_id=principal.user_id,
            name=_required_name(body.name),
            actor_user_id=principal.user_id,
        )
        return _project_response(project)

    @router.get(
        "/projects",
        response_model=ChatProjectListResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def list_chat_projects(
        request: Request,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ):
        principal = require_interactive_user_principal(request)
        projects, total = await get_enterprise_chat_conversation_service(
            request
        ).list_projects(principal.user_id, limit=limit, offset=offset)
        return ChatProjectListResponse(
            total=total,
            limit=limit,
            offset=offset,
            projects=[_project_response(project) for project in projects],
        )

    @router.get(
        "/projects/{project_id}",
        response_model=ChatProjectResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_chat_project(project_id: str, request: Request):
        principal = require_interactive_user_principal(request)
        project = await get_enterprise_chat_conversation_service(
            request
        ).get_project(principal.user_id, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Chat project not found")
        return _project_response(project)

    @router.patch(
        "/projects/{project_id}",
        response_model=ChatProjectResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def rename_chat_project(
        project_id: str, request: Request, body: ChatProjectUpdateRequest
    ):
        principal = require_interactive_user_principal(request)
        project = await get_enterprise_chat_conversation_service(
            request
        ).rename_project(
            user_id=principal.user_id,
            project_id=project_id,
            name=_required_name(body.name),
            actor_user_id=principal.user_id,
        )
        if project is None:
            raise HTTPException(status_code=404, detail="Chat project not found")
        return _project_response(project)

    @router.delete(
        "/projects/{project_id}",
        response_model=ChatProjectDeleteResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def delete_chat_project(project_id: str, request: Request):
        principal = require_interactive_user_principal(request)
        deleted, deleted_sessions, deleted_messages = (
            await get_enterprise_chat_conversation_service(request).delete_project(
                user_id=principal.user_id,
                project_id=project_id,
                actor_user_id=principal.user_id,
            )
        )
        if not deleted:
            raise HTTPException(status_code=404, detail="Chat project not found")
        memory_service = get_enterprise_chat_memory_service(request)
        if memory_service is not None:
            # Best-effort background cleanup of the project's memory graph.
            memory_service.schedule_purge(principal.user_id, [project_id])
        return ChatProjectDeleteResponse(
            id=project_id,
            deleted=True,
            deleted_sessions=deleted_sessions,
            deleted_messages=deleted_messages,
        )

    @router.post(
        "/projects/{project_id}/memory:search",
        response_model=ChatMemorySearchResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def search_chat_project_memory(
        project_id: str, request: Request, body: ChatMemorySearchRequest
    ):
        """Hybrid search over this project's long-term memory graph.

        Returns temporal facts distilled from the project's earlier sessions;
        clients weave them into ``user_prompt``/``conversation_history`` of
        the next query, or let the query/agent endpoints inject them
        server-side via the ``memory`` field (docs/ChatMemory-zh.md §6).
        """
        principal = require_interactive_user_principal(request)
        memory_service = get_enterprise_chat_memory_service(request)
        if memory_service is None:
            raise HTTPException(status_code=503, detail="Chat memory is not enabled")
        project = await get_enterprise_chat_conversation_service(
            request
        ).get_project(principal.user_id, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Chat project not found")
        try:
            facts = await memory_service.search(
                user_id=principal.user_id,
                project_id=project_id,
                query=body.query,
                limit=body.limit,
            )
        except ChatMemoryUnavailableError:
            raise HTTPException(
                status_code=503, detail="Chat memory is temporarily unavailable"
            )
        return ChatMemorySearchResponse(
            project_id=project_id,
            total=len(facts),
            facts=[ChatMemoryFact(**fact) for fact in facts],
        )

    @router.get(
        "/projects/{project_id}/memory",
        response_model=ChatMemoryOverviewResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_chat_project_memory_overview(project_id: str, request: Request):
        """How much memory a project has accumulated (episode count + last
        ingest time) plus the enabled/available flags — lets the front end show
        a "N memories, updated X" indicator without a search."""
        principal = require_interactive_user_principal(request)
        memory_service = get_enterprise_chat_memory_service(request)
        if memory_service is None:
            raise HTTPException(status_code=503, detail="Chat memory is not enabled")
        project = await get_enterprise_chat_conversation_service(
            request
        ).get_project(principal.user_id, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Chat project not found")
        overview = await memory_service.project_overview(
            principal.user_id, project_id
        )
        return ChatMemoryOverviewResponse(**overview)

    @router.post(
        "/projects/{project_id}/sessions",
        response_model=ChatSessionResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def create_chat_session(
        project_id: str,
        request: Request,
        body: ChatSessionCreateRequest | None = None,
    ):
        principal = require_interactive_user_principal(request)
        context_rounds = body.context_rounds if body is not None else None
        if context_rounds is not None:
            context_rounds = _validated_context_rounds(context_rounds)
        session = await get_enterprise_chat_conversation_service(
            request
        ).create_session(
            user_id=principal.user_id,
            project_id=project_id,
            name=body.name if body is not None else None,
            context_rounds=context_rounds,
            actor_user_id=principal.user_id,
        )
        if session is None:
            raise HTTPException(status_code=404, detail="Chat project not found")
        return _session_response(session)

    @router.get(
        "/projects/{project_id}/sessions",
        response_model=ChatSessionListResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def list_chat_sessions(
        project_id: str,
        request: Request,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ):
        principal = require_interactive_user_principal(request)
        service = get_enterprise_chat_conversation_service(request)
        project = await service.get_project(principal.user_id, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Chat project not found")
        sessions, total = await service.list_sessions(
            principal.user_id, project_id, limit=limit, offset=offset
        )
        return ChatSessionListResponse(
            total=total,
            limit=limit,
            offset=offset,
            sessions=[_session_response(session) for session in sessions],
        )

    @router.get(
        "/projects/{project_id}/sessions/{session_id}",
        response_model=ChatSessionResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_chat_session(project_id: str, session_id: str, request: Request):
        principal = require_interactive_user_principal(request)
        session = await get_enterprise_chat_conversation_service(
            request
        ).get_session(principal.user_id, project_id, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Chat session not found")
        return _session_response(session)

    @router.patch(
        "/projects/{project_id}/sessions/{session_id}",
        response_model=ChatSessionResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def update_chat_session(
        project_id: str,
        session_id: str,
        request: Request,
        body: ChatSessionUpdateRequest,
    ):
        principal = require_interactive_user_principal(request)
        if body.name is None and body.context_rounds is None:
            raise HTTPException(
                status_code=400,
                detail="At least one of name or context_rounds is required",
            )
        name = _required_name(body.name) if body.name is not None else None
        context_rounds = (
            _validated_context_rounds(body.context_rounds)
            if body.context_rounds is not None
            else None
        )
        session = await get_enterprise_chat_conversation_service(
            request
        ).update_session(
            user_id=principal.user_id,
            project_id=project_id,
            session_id=session_id,
            name=name,
            context_rounds=context_rounds,
            actor_user_id=principal.user_id,
        )
        if session is None:
            raise HTTPException(status_code=404, detail="Chat session not found")
        return _session_response(session)

    @router.delete(
        "/projects/{project_id}/sessions/{session_id}",
        response_model=ChatSessionDeleteResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def delete_chat_session(
        project_id: str, session_id: str, request: Request
    ):
        principal = require_interactive_user_principal(request)
        deleted, deleted_messages = await get_enterprise_chat_conversation_service(
            request
        ).delete_session(
            user_id=principal.user_id,
            project_id=project_id,
            session_id=session_id,
            actor_user_id=principal.user_id,
        )
        if not deleted:
            raise HTTPException(status_code=404, detail="Chat session not found")
        memory_service = get_enterprise_chat_memory_service(request)
        if memory_service is not None:
            # Best-effort: forget the memory episodes distilled from this session.
            memory_service.schedule_forget_session(
                user_id=principal.user_id,
                project_id=project_id,
                session_id=session_id,
            )
        return ChatSessionDeleteResponse(
            id=session_id,
            project_id=project_id,
            deleted=True,
            deleted_messages=deleted_messages,
        )

    @router.post(
        "/projects/{project_id}/sessions/{session_id}/messages",
        response_model=ChatMessagesAppendResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def append_chat_messages(
        project_id: str,
        session_id: str,
        request: Request,
        body: ChatMessagesAppendRequest,
    ):
        principal = require_interactive_user_principal(request)
        messages = [
            {
                "role": item.role,
                "content": item.content,
                "metadata": _validated_message_metadata(item.metadata),
            }
            for item in body.messages
        ]
        saved = await get_enterprise_chat_conversation_service(
            request
        ).append_messages(
            user_id=principal.user_id,
            project_id=project_id,
            session_id=session_id,
            messages=messages,
            actor_user_id=principal.user_id,
        )
        if saved is None:
            raise HTTPException(status_code=404, detail="Chat session not found")
        memory_service = get_enterprise_chat_memory_service(request)
        if memory_service is not None:
            # Fire-and-forget: distill the persisted turn into the project's
            # memory graph without delaying the request.
            memory_service.schedule_ingest(
                user_id=principal.user_id,
                project_id=project_id,
                session_id=session_id,
                messages=saved,
            )
        return ChatMessagesAppendResponse(
            session_id=session_id,
            project_id=project_id,
            messages=[_message_response(record) for record in saved],
        )

    @router.get(
        "/projects/{project_id}/sessions/{session_id}/messages",
        response_model=ChatMessageListResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def list_chat_messages(
        project_id: str,
        session_id: str,
        request: Request,
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ):
        principal = require_interactive_user_principal(request)
        service = get_enterprise_chat_conversation_service(request)
        session = await service.get_session(
            principal.user_id, project_id, session_id
        )
        if session is None:
            raise HTTPException(status_code=404, detail="Chat session not found")
        messages, total = await service.list_messages(
            principal.user_id, project_id, session_id, limit=limit, offset=offset
        )
        return ChatMessageListResponse(
            total=total,
            limit=limit,
            offset=offset,
            messages=[_message_response(record) for record in messages],
        )

    @router.delete(
        "/projects/{project_id}/sessions/{session_id}/messages/{message_id}",
        response_model=ChatMessageDeleteResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def delete_chat_message(
        project_id: str, session_id: str, message_id: str, request: Request
    ):
        principal = require_interactive_user_principal(request)
        service = get_enterprise_chat_conversation_service(request)
        memory_service = get_enterprise_chat_memory_service(request)
        message_seq: int | None = None
        if memory_service is not None:
            # Capture the seq before deletion; it locates the memory episode(s)
            # distilled from this message.
            record = await service.get_message(
                principal.user_id, project_id, session_id, message_id
            )
            message_seq = record.seq if record is not None else None
        deleted = await service.delete_message(
            user_id=principal.user_id,
            project_id=project_id,
            session_id=session_id,
            message_id=message_id,
            actor_user_id=principal.user_id,
        )
        if not deleted:
            raise HTTPException(status_code=404, detail="Chat message not found")
        if memory_service is not None and message_seq is not None:
            memory_service.schedule_forget_message(
                user_id=principal.user_id,
                project_id=project_id,
                session_id=session_id,
                seq=message_seq,
            )
        return ChatMessageDeleteResponse(
            id=message_id,
            session_id=session_id,
            project_id=project_id,
            deleted=True,
        )

    return router
