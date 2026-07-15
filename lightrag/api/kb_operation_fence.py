from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from lightrag.api.kb_service import KnowledgeBaseNotFoundError, validate_kb_id
from lightrag.api.metadata_store import KBLifecycleConflictError


KB_GENERATION_STATE_KEY = "kb_generation"
KB_ID_STATE_KEY = "kb_id"
_WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


@dataclass(frozen=True, slots=True)
class KBWriteTarget:
    kb_id: str


class ConflictingKBWriteTargetsError(ValueError):
    """A write path names more than one distinct KB identity."""

    def __init__(self, kb_ids: set[str]) -> None:
        self.kb_ids = tuple(sorted(kb_ids))
        super().__init__(
            "Request path contains conflicting KB identities: "
            + ", ".join(self.kb_ids)
        )


def _decoded_route_path(scope: Scope) -> str:
    """Return the decoded application route without scanning ``root_path``."""

    path_value = scope.get("path")
    if not path_value:
        raw_path = scope.get("raw_path")
        if isinstance(raw_path, bytes):
            path_value = raw_path.decode("utf-8", errors="replace")
    path = unquote(str(path_value or "/"))
    root_path = unquote(str(scope.get("root_path") or ""))
    if root_path and root_path != "/":
        normalized_root = f"/{root_path.strip('/')}"
        if path == normalized_root:
            return "/"
        if path.startswith(f"{normalized_root}/"):
            return path[len(normalized_root) :]
    return path


def kb_write_target_from_scope(scope: Scope) -> KBWriteTarget | None:
    """Resolve the KB protected by a state-changing HTTP request.

    ASGI ``path`` is intentionally used instead of ``raw_path``. Servers and
    proxies may decode ``%2F`` before routing; inspecting only the raw bytes
    would let an encoded separator reach a real ``/kbs/{kb_id}`` route without
    admission control. Segment comparison remains exact, so ``/kbsX`` is not a
    match.
    """

    if scope.get("type") != "http":
        return None
    method = str(scope.get("method") or "").upper()
    if method not in _WRITE_METHODS:
        return None

    route_path = _decoded_route_path(scope)

    # KB creation and KB lifecycle transitions own their own admission rules.
    if method == "POST" and route_path == "/kbs":
        return None

    candidates: set[str] = set()
    segments = route_path.split("/")
    for index, segment in enumerate(segments):
        if segment != "kbs" or index + 1 >= len(segments):
            continue
        candidate = segments[index + 1]
        if not candidate:
            continue

        # FastAPI action routes encode the action in the same segment as the KB
        # id (``/{kb_id}:rebuild``, ``:reparse``, ...). Admission protects the
        # identity before the first colon; exact lifecycle exclusions are
        # applied only after all path identities have been collected.
        if ":" in candidate:
            candidate = candidate.split(":", 1)[0]
            if not candidate:
                continue

        try:
            candidate = validate_kb_id(candidate)
        except ValueError:
            continue

        candidates.add(candidate)

    if len(candidates) > 1:
        raise ConflictingKBWriteTargetsError(candidates)
    if not candidates:
        return None

    candidate = next(iter(candidates))
    if method == "POST" and route_path == f"/kbs/{candidate}:restore":
        return None
    if method == "DELETE" and route_path == f"/kbs/{candidate}":
        return None
    return KBWriteTarget(kb_id=candidate)


class KBWriteAdmissionMiddleware:
    """Pure-ASGI shared admission fence for KB-scoped writes.

    The guard surrounds the complete downstream ASGI call. Unlike
    ``BaseHTTPMiddleware``, that includes response streaming and Starlette /
    FastAPI response background tasks, so hard-delete's exclusive fence cannot
    overtake request-owned staging or mutation work.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        kb_service: Any,
        metadata_store: Any,
    ) -> None:
        self.app = app
        self.kb_service = kb_service
        self.metadata_store = metadata_store

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            target = kb_write_target_from_scope(scope)
        except ConflictingKBWriteTargetsError as exc:
            await self._send_conflicting_targets(scope, receive, send, exc.kb_ids)
            return
        if target is None:
            await self.app(scope, receive, send)
            return

        try:
            # include_deleted distinguishes a genuinely absent catalog row
            # (which must retain the downstream route's normal 404) from a
            # soft-deleted row whose hard-delete lifecycle is already active.
            record = await self.kb_service.get(
                target.kb_id,
                include_deleted=True,
            )
        except (KnowledgeBaseNotFoundError, ValueError):
            await self.app(scope, receive, send)
            return

        if getattr(record, "status", None) != "active":
            lifecycle = await self.metadata_store.get_kb_lifecycle(record.id)
            if lifecycle is not None and lifecycle.state == "deleting":
                await self._send_conflict(scope, receive, send, record.id)
                return
            await self.app(scope, receive, send)
            return

        response_started = False

        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            async with self.metadata_store.kb_write_guard(
                record.id,
                record.generation,
            ):
                state = scope.get("state")
                if not isinstance(state, dict):
                    state = {}
                    scope["state"] = state
                state[KB_ID_STATE_KEY] = record.id
                state[KB_GENERATION_STATE_KEY] = record.generation
                await self.app(scope, receive, tracked_send)
        except KBLifecycleConflictError:
            if response_started:
                raise
            await self._send_conflict(scope, receive, send, record.id)

    @staticmethod
    async def _send_conflict(
        scope: Scope,
        receive: Receive,
        send: Send,
        kb_id: str,
    ) -> None:
        response = JSONResponse(
            status_code=409,
            content={
                "detail": {
                    "error_code": "kb_write_admission_conflict",
                    "kb_id": kb_id,
                    "message": "Knowledge base is being deleted or changed generation",
                }
            },
        )
        await response(scope, receive, send)

    @staticmethod
    async def _send_conflicting_targets(
        scope: Scope,
        receive: Receive,
        send: Send,
        kb_ids: tuple[str, ...],
    ) -> None:
        response = JSONResponse(
            status_code=400,
            content={
                "detail": {
                    "error_code": "conflicting_kb_write_targets",
                    "kb_ids": list(kb_ids),
                    "message": "Request path contains conflicting KB identities",
                }
            },
        )
        await response(scope, receive, send)
