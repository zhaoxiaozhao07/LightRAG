"""Tests for ``lightrag.api.streaming_lifecycle`` disconnect helpers.

These exercise the cancellation semantics directly (no HTTP/ASGI layer) so the
behavior is provable regardless of Starlette's transport internals.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from starlette.requests import Request

from lightrag.api.streaming_lifecycle import (
    ClientGoneError,
    abort_if_client_gone,
    await_with_disconnect_check,
    client_closed_response,
    is_client_disconnected,
    safe_aclose,
    stream_with_disconnect_guard,
)


def _client_closed_status() -> int:
    # The numeric 499 status is an internal symbol; import via the module to
    # avoid coupling tests to a private name.
    import lightrag.api.streaming_lifecycle as mod

    return mod._CLIENT_CLOSED_STATUS  # noqa: SLF001 — intentional internal read


# ---------------------------------------------------------------------------
# is_client_disconnected
# ---------------------------------------------------------------------------


def _make_request(*, disconnected: bool = False, broken: bool = False) -> Any:
    class _StubRequest:
        def __init__(self) -> None:
            self.calls = 0
            self._broken = broken
            self._disconnected = disconnected

        async def is_disconnected(self) -> bool:
            self.calls += 1
            if self._broken:
                raise RuntimeError("probe exploded")
            return self._disconnected

    return _StubRequest()


async def test_is_client_disconnected_returns_false_for_none():
    assert await is_client_disconnected(None) is False


async def test_is_client_disconnected_returns_false_when_still_connected():
    req = _make_request(disconnected=False)
    assert await is_client_disconnected(req) is False
    assert req.calls == 1


async def test_is_client_disconnected_returns_true_when_gone():
    req = _make_request(disconnected=True)
    assert await is_client_disconnected(req) is True


async def test_is_client_disconnected_never_raises():
    req = _make_request(broken=True)
    assert await is_client_disconnected(req) is False


# ---------------------------------------------------------------------------
# await_with_disconnect_check
# ---------------------------------------------------------------------------


async def test_returns_result_when_client_stays_connected():
    async def work() -> str:
        await asyncio.sleep(0)
        return "ok"

    result = await await_with_disconnect_check(_make_request(), work(), poll_interval=0.01)
    assert result == "ok"


async def test_propagates_real_exception_unchanged():
    sentinel = ValueError("boom")

    async def work() -> None:
        raise sentinel

    with pytest.raises(ValueError) as exc_info:
        await await_with_disconnect_check(_make_request(), work(), poll_interval=0.01)
    assert exc_info.value is sentinel


async def test_no_polling_direct_await_when_request_none():
    marker = object()

    async def work() -> Any:
        return marker

    # request=None must short-circuit to a plain await (no task wrapping/probe)
    result = await await_with_disconnect_check(None, work(), poll_interval=0.5)
    assert result is marker


async def test_cancels_work_and_raises_client_gone_on_disconnect():
    cleanup = {"ran": False}

    async def work() -> str:
        try:
            # Blocks forever unless cancelled.
            await asyncio.sleep(30)
            return "unreachable"
        except asyncio.CancelledError:
            cleanup["ran"] = True
            raise

    req = _make_request(disconnected=True)
    with pytest.raises(ClientGoneError):
        await await_with_disconnect_check(req, work(), poll_interval=0.01)

    # The cancelled work must have run its cleanup, not be left dangling.
    assert cleanup["ran"] is True


async def test_does_not_cry_on_already_done_work():
    async def work() -> int:
        return 7

    req = _make_request(disconnected=True)
    # Work resolves immediately on the first tick; disconnect must not win.
    result = await await_with_disconnect_check(req, work(), poll_interval=0.01)
    assert result == 7


async def test_disconnect_waits_for_cooperative_cleanup():
    """Cleanup that acknowledges cancellation promptly completes before raising."""

    state = {"cleaned": False}

    async def work() -> None:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            await asyncio.sleep(0)  # cleanup itself awaits
            state["cleaned"] = True
            raise

    with pytest.raises(ClientGoneError):
        await await_with_disconnect_check(
            _make_request(disconnected=True), work(), poll_interval=0.01
        )
    assert state["cleaned"] is True


async def test_disconnect_abandons_stuck_cleanup_after_grace(monkeypatch):
    """Work whose cleanup outlives the grace period must not stall the 499."""

    import lightrag.api.streaming_lifecycle as mod

    monkeypatch.setattr(mod, "_CANCEL_GRACE_SECONDS", 0.05)
    warnings: list[str] = []
    # The lightrag logger has propagate=False, so capture directly.
    monkeypatch.setattr(
        mod.logger, "warning", lambda msg, *args, **kw: warnings.append(msg % args)
    )

    release = asyncio.Event()
    state = {"finished": False}

    async def stubborn() -> None:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            await release.wait()  # cleanup stuck far beyond the grace window
            state["finished"] = True
            raise

    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(ClientGoneError):
        await await_with_disconnect_check(
            _make_request(disconnected=True), stubborn(), poll_interval=0.01
        )
    # Bounded: the caller got its ClientGoneError long before the 30s sleep,
    # while the stuck cleanup was abandoned with a warning.
    assert loop.time() - started < 2
    assert state["finished"] is False
    assert any("abandoning it" in w for w in warnings)

    # Let the abandoned task finish so the loop closes clean; its outcome is
    # consumed by the reap callback (no "exception was never retrieved").
    release.set()
    await asyncio.sleep(0.01)
    assert state["finished"] is True


# ---------------------------------------------------------------------------
# client_closed_response
# ---------------------------------------------------------------------------


def test_client_closed_response_is_499():
    response = client_closed_response()
    assert response.status_code == _client_closed_status()
    assert response.body == b""


# ---------------------------------------------------------------------------
# safe_aclose
# ---------------------------------------------------------------------------


async def test_safe_aclose_calls_aclose_once():
    calls = {"n": 0}

    class _Iter:
        async def aclose(self) -> None:
            calls["n"] += 1

    await safe_aclose(_Iter())
    assert calls["n"] == 1


async def test_safe_aclose_awaits_coroutine_aclose():
    calls = {"n": 0}

    class _Iter:
        async def aclose(self) -> None:
            await asyncio.sleep(0)
            calls["n"] += 1

    await safe_aclose(_Iter())
    assert calls["n"] == 1


async def test_safe_aclose_swallows_errors():
    class _Iter:
        async def aclose(self) -> None:
            raise RuntimeError("cleanup failed")

    # Must not raise.
    await safe_aclose(_Iter())


async def test_safe_aclose_noop_when_no_aclose():
    await safe_aclose(object())  # no attribute -> no-op, no raise


async def test_safe_aclose_propagates_cancelled_error():
    class _Iter:
        async def aclose(self) -> None:
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await safe_aclose(_Iter())


# ---------------------------------------------------------------------------
# stream_with_disconnect_guard
# ---------------------------------------------------------------------------


async def test_stream_passes_all_chunks_when_connected():
    async def src():
        for piece in ("a", "b", "c"):
            yield piece

    out = [
        item
        async for item in stream_with_disconnect_guard(
            src(), _make_request(disconnected=False), poll_interval=0.01
        )
    ]
    assert out == ["a", "b", "c"]


async def test_stream_closes_upstream_on_completion():
    closed = {"n": 0}

    class _Src:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def aclose(self) -> None:
            closed["n"] += 1

    out = [
        item
        async for item in stream_with_disconnect_guard(
            _Src(), _make_request(disconnected=False), poll_interval=0.01
        )
    ]
    assert out == []
    assert closed["n"] == 1


async def test_stream_stops_and_closes_on_disconnect():
    closed = {"n": 0}

    async def src():
        try:
            yield "first"
            await asyncio.sleep(30)  # stalls after first chunk
            yield "unreachable"
        finally:
            closed["n"] += 1

    req = _make_request(disconnected=True)
    out = [
        item
        async for item in stream_with_disconnect_guard(src(), req, poll_interval=0.01)
    ]
    assert out == ["first"]
    assert closed["n"] == 1


async def test_stream_no_polling_request_none_yields_everything():
    async def src():
        for piece in ("x", "y"):
            yield piece

    out = [item async for item in stream_with_disconnect_guard(src(), None)]
    assert out == ["x", "y"]


# ---------------------------------------------------------------------------
# abort_if_client_gone
# ---------------------------------------------------------------------------


class _ClosableUpstream:
    def __init__(self) -> None:
        self.closed = 0

    async def aclose(self) -> None:
        self.closed += 1


async def test_abort_if_client_gone_noop_while_connected():
    upstream = _ClosableUpstream()
    await abort_if_client_gone(_make_request(disconnected=False), upstream)
    assert upstream.closed == 0


async def test_abort_if_client_gone_noop_without_request():
    await abort_if_client_gone(None, _ClosableUpstream())  # must not raise


async def test_abort_if_client_gone_closes_upstreams_and_raises():
    first, second = _ClosableUpstream(), _ClosableUpstream()
    with pytest.raises(ClientGoneError):
        # None entries (e.g. a non-streaming result) are skipped silently.
        await abort_if_client_gone(
            _make_request(disconnected=True), first, None, second
        )
    assert first.closed == 1
    assert second.closed == 1


# Sanity: Request typing is importable and is what the helpers accept.
def test_request_type_alias_imported():
    assert Request is not None
