"""Client-abort lifecycle helpers for streaming query routes.

The RAG query path has a gap: Starlette's ``StreamingResponse`` only starts its
disconnect watcher *after* the response object is returned, but the expensive
retrieval / synthesis / non-streaming LLM work happens *before* that point (the
handler awaits ``rag.aquery_llm(...)`` then returns ``StreamingResponse``). If
the user hits "stop" during that window the server keeps running the whole
pipeline to completion and discards the result.

This module closes four gaps while preserving byte-identical normal behavior:

1. **Pre-stream cancellation** — ``await_with_disconnect_check`` races the
   retrieval/synthesis coroutine against a client-disconnect poller and cancels
   the work (raising :class:`ClientGoneError`) the moment the client is gone.
2. **In-stream cancellation** — ``stream_with_disconnect_guard`` wraps a chunk
   iterator so a mid-stream disconnect stops pulling tokens and promptly closes
   the upstream LLM response.
3. **Deterministic upstream release** — ``safe_aclose`` closes an LLM stream
   iterator explicitly instead of relying on GC finalization timing.
4. **Pre-response abort** — ``abort_if_client_gone`` is a last check right
   before a handler returns ``StreamingResponse``: if the client vanished after
   the guarded work finished, the response generator may never start (so its
   cleanup never runs); this releases the already-open upstream stream now.

All helpers degrade gracefully (never raise) when a disconnect probe is
unavailable, so a transport that cannot report ``http.disconnect`` simply
behaves like the old code.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from typing import Any, AsyncIterator, Awaitable, TypeVar

from fastapi import Response
from starlette.requests import Request

from lightrag.utils import logger

T = TypeVar("T")

#: HTTP status returned when the client closed the request before we finished.
#: Mirrors nginx's non-standard "Client Closed Request" code; the body is never
#: delivered (the socket is gone) — the status only keeps ASGI teardown clean
#: and prevents a misleading 500 in logs.
_CLIENT_CLOSED_STATUS = 499

#: How often (seconds) the disconnect poller wakes while awaiting pre-stream
#: work. Snappy enough to feel responsive to a human, cheap enough that the
#: probe (one non-blocking ASGI receive) is negligible.
_DISCONNECT_POLL_INTERVAL = 0.5

#: How long (seconds) a cancelled work task is allowed to run its cleanup
#: before being abandoned. Waiting forever would let a pipeline that swallows
#: ``CancelledError`` (or is stuck in a synchronous call) hold the connection
#: slot and delay the 499 response indefinitely.
_CANCEL_GRACE_SECONDS = 5.0


class ClientGoneError(Exception):
    """The client disconnected while the response was still being prepared.

    Route handlers translate this into a :data:`_CLIENT_CLOSED_STATUS` response
    so the teardown is clean and nothing is logged as a server error.
    """


def client_closed_response() -> Response:
    """Return an empty 499 response for an already-aborted request."""

    return Response(status_code=_CLIENT_CLOSED_STATUS)


async def is_client_disconnected(request: Request | None) -> bool:
    """Best-effort async disconnect check that never raises.

    Returns ``False`` (i.e. "still connected") whenever the probe is missing or
    cannot be evaluated, so callers can treat "unknown" as "proceed".
    """

    if request is None:
        return False
    is_disconnected = getattr(request, "is_disconnected", None)
    if not callable(is_disconnected):
        return False
    try:
        result = is_disconnected()
        if inspect.isawaitable(result):
            result = await result
        return bool(result)
    except Exception:  # noqa: BLE001 — a probe failure must never abort a request
        return False


def _consume_task_result(task: "asyncio.Future[Any]") -> None:
    """Retrieve a finished task's outcome so asyncio never logs it as unretrieved."""

    with contextlib.suppress(BaseException):
        task.result()


async def _cancel_and_reap(
    task: "asyncio.Future[Any]", *, grace: float | None = None
) -> None:
    """Cancel ``task`` and wait a bounded time for its cleanup to finish.

    Normally the cancelled work acknowledges within milliseconds and its
    resource cleanup (LLM stream close, lock release) completes before we
    return. If it is still running after ``grace`` seconds (stuck in a blocking
    sync call, or swallowing ``CancelledError``) it is abandoned with a warning
    so the caller — typically a 499 teardown — is not stalled indefinitely. The
    done-callback consumes the task's eventual outcome either way, including
    when this coroutine is itself cancelled mid-wait.
    """

    if grace is None:
        grace = _CANCEL_GRACE_SECONDS
    if task.done():
        _consume_task_result(task)
        return
    task.add_done_callback(_consume_task_result)
    task.cancel()
    _done, pending = await asyncio.wait({task}, timeout=grace)
    if pending:
        logger.warning(
            "Cancelled query work is still running after %.1fs "
            "(likely blocked in sync code or swallowing CancelledError); "
            "abandoning it",
            grace,
        )


async def await_with_disconnect_check(
    request: Request | None,
    awaitable: Awaitable[T],
    *,
    poll_interval: float = _DISCONNECT_POLL_INTERVAL,
) -> T:
    """Await ``awaitable``, cancelling it if the client disconnects.

    The retrieval/synthesis phase (before the first token streams) can take
    several seconds and Starlette's disconnect watcher is not active yet. This
    coroutine polls ``request.is_disconnected()`` between short waits; on
    disconnect it cancels the in-flight work, waits (bounded by
    :data:`_CANCEL_GRACE_SECONDS`) for its cleanup, and raises
    :class:`ClientGoneError`.

    With ``poll_interval <= 0`` the work is awaited directly with no polling
    (useful in tests where disconnect is simulated elsewhere).

    Normal completion preserves the awaited result and any exception it raises
    verbatim; only a client disconnect produces ``ClientGoneError``. A
    :class:`asyncio.CancelledError` injected into *this* coroutine (e.g. server
    shutdown) also cancels the child task and re-raises.
    """

    if poll_interval <= 0 or request is None:
        # No polling possible/requested: await directly so behavior is exactly
        # the legacy path (no extra task layer, no probe).
        return await awaitable

    task = asyncio.ensure_future(awaitable)
    try:
        while True:
            done, _pending = await asyncio.wait(
                {task}, timeout=poll_interval, return_when=asyncio.FIRST_COMPLETED
            )
            if done:
                # Work finished (or raised) — surface its real outcome verbatim.
                return task.result()
            if await is_client_disconnected(request):
                logger.debug(
                    "Client disconnected during query preparation; cancelling work"
                )
                # The finally below cancels the work and waits (bounded) for its
                # cleanup before this exception reaches the route handler.
                raise ClientGoneError(
                    "client disconnected before the response started streaming"
                )
    finally:
        if not task.done():
            await _cancel_and_reap(task)


async def safe_aclose(iterator: Any) -> None:
    """Close an upstream async iterator promptly, never raising.

    When a stream aborts (client disconnect or an error mid-stream) the
    OpenAI/httpx response backing the LLM stream stays open on the provider
    side. Starlette cancels the generator and Python's GC finalizes it
    eventually, but that timing is non-deterministic and may continue pulling
    tokens briefly. Calling ``aclose()`` explicitly releases the connection now.

    Best-effort: a cleanup failure must never mask the real error.
    """

    close = getattr(iterator, "aclose", None)
    if not callable(close):
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await result
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 — cleanup must never replace the real error
        return


async def abort_if_client_gone(request: Request | None, *upstreams: Any) -> None:
    """Raise :class:`ClientGoneError` now if the client already disconnected.

    Final check for streaming handlers, right before ``return
    StreamingResponse(...)``: between the guarded pre-stream work completing
    (upstream LLM stream already open) and the response generator actually
    starting there is a small window in which a disconnect means the generator
    may never run — so its ``finally`` cleanup never fires either. On a
    detected disconnect this closes every non-``None`` ``upstream`` via
    :func:`safe_aclose` and raises, letting the handler unwind through its
    normal ``ClientGoneError`` → 499 path. No-op while the client is connected
    or the probe is unavailable.
    """

    if not await is_client_disconnected(request):
        return
    for upstream in upstreams:
        if upstream is not None:
            await safe_aclose(upstream)
    raise ClientGoneError("client disconnected before the response started streaming")


async def stream_with_disconnect_guard(
    iterator: AsyncIterator[T],
    request: Request | None,
    *,
    poll_interval: float = _DISCONNECT_POLL_INTERVAL,
) -> AsyncIterator[T]:
    """Yield from ``iterator`` but stop early on client disconnect.

    Starlette's own disconnect watcher cancels the streaming generator, which is
    the primary cancellation mechanism. This wrapper adds a cooperative layer:
    between chunks it also polls the client state so a disconnect detected on a
    quiet connection (no chunk in flight) is honored without waiting for the
    next chunk. On any exit (normal, error, or abort) the upstream iterator is
    closed via :func:`safe_aclose`.
    """

    try:
        if poll_interval <= 0 or request is None:
            async for item in iterator:
                yield item
            return
        while True:
            # Wait for the next chunk, but also wake periodically to re-check
            # the client. aiter()/anext() keeps iteration cooperative so the
            # Starlette cancel still wins immediately on a busy stream.
            nxt = asyncio.ensure_future(iterator.__anext__())
            try:
                while True:
                    done, _pending = await asyncio.wait(
                        {nxt}, timeout=poll_interval, return_when=asyncio.FIRST_COMPLETED
                    )
                    if done:
                        try:
                            item = nxt.result()
                        except StopAsyncIteration:
                            return
                        yield item
                        break  # fetch the next chunk
                    if await is_client_disconnected(request):
                        logger.debug(
                            "Client disconnected during streaming; stopping output"
                        )
                        return
            finally:
                if not nxt.done():
                    await _cancel_and_reap(nxt)
    finally:
        await safe_aclose(iterator)
