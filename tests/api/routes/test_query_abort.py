"""End-to-end client-abort tests for the streaming query routes.

These prove the behavior the unit tests in ``test_streaming_lifecycle.py`` only
assert in isolation: when the HTTP client disconnects while the server is still
in the expensive pre-stream phase (retrieval / synthesis), the route cancels the
in-flight RAG work instead of running it to completion and discarding the result.

We drive the ASGI app directly (rather than via httpx's ``ASGITransport``, which
always waits for the response to finish and therefore cannot model a mid-request
disconnect). A custom ``receive()`` returns one ``http.request`` then
``http.disconnect`` — exactly what a real transport delivers when the socket
closes — so both Starlette's own disconnect watcher and our cooperative polling
(see ``streaming_lifecycle``) observe the abort.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_query_routes = importlib.import_module("lightrag.api.routers.query_routes")
_kb_routes = importlib.import_module("lightrag.api.routers.kb_routes")
_kb_query_routes = importlib.import_module("lightrag.api.routers.kb_query_routes")
_kb_document_routes = importlib.import_module(
    "lightrag.api.routers.kb_document_routes"
)
sys.argv = _original_argv

create_query_routes = _query_routes.create_query_routes
create_kb_routes = _kb_routes.create_kb_routes
create_kb_query_routes = _kb_query_routes.create_kb_query_routes
create_kb_document_routes = _kb_document_routes.create_kb_document_routes

pytestmark = pytest.mark.offline

_API_KEY = "test-key"


@pytest.fixture(autouse=True)
def _disable_enterprise_auth(monkeypatch):
    from lightrag.api import config as api_config

    monkeypatch.setattr(
        api_config,
        "global_args",
        SimpleNamespace(
            enterprise_auth_enabled=False,
            token_auto_renew=False,
            token_renew_threshold=0.5,
        ),
    )


# ---------------------------------------------------------------------------
# ASGI driver that models a client which disconnects after sending the body
# ---------------------------------------------------------------------------


async def _drive(
    app: FastAPI,
    *,
    method: str,
    path: str,
    body: dict[str, Any] | None,
    disconnect_after_body: bool = True,
    spec_version: str = "2.3",
) -> tuple[int, bytes]:
    """Run ``app`` with a receive() that yields the request then disconnects.

    Returns ``(status_code, body_bytes)`` captured from the ASGI ``send``.
    """

    payload = json.dumps(body).encode() if body is not None else b""
    receive_calls = {"n": 0}

    async def receive():
        receive_calls["n"] += 1
        if receive_calls["n"] == 1:
            return {"type": "http.request", "body": payload, "more_body": False}
        # From the second call onward the client is gone (or probing).
        return {"type": "http.disconnect"}

    status_code: list[int] = []
    body_parts: list[bytes] = []

    async def send(message):
        if message["type"] == "http.response.start":
            status_code.append(message["status"])
        elif message["type"] == "http.response.body":
            body_parts.append(message.get("body", b""))

    scope = {
        "type": "http",
        # spec_version 2.3 (< 2.4) is what uvicorn reports, which forces the
        # anyio task-group disconnect-listening branch in StreamingResponse;
        # pass "2.4" to exercise the branch with no listener task, where the
        # cooperative poller is the only disconnect detector.
        "asgi": {"version": "3.0", "spec_version": spec_version},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [
            (b"x-api-key", _API_KEY.encode()),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(payload)).encode()),
        ],
        "client": ("testclient", 123),
        "server": ("testserver", 80),
        "root_path": "",
    }
    await app(scope, receive, send)
    status = status_code[0] if status_code else 0
    return status, b"".join(body_parts)


# ---------------------------------------------------------------------------
# Fake RAG whose aquery_llm blocks forever unless cancelled
# ---------------------------------------------------------------------------


class _BlockingFakeRAG:
    """A RAG stand-in that stalls in ``aquery_llm`` so an abort lands mid-work.

    Records whether it was cancelled (cleanup ran) vs. allowed to complete.
    """

    def __init__(self) -> None:
        self.cancelled = False
        self.completed = False

    async def aquery_llm(self, query: str, *, param):
        try:
            # Simulate the long retrieval + synthesis window before any token.
            await asyncio.sleep(30)
            self.completed = True
            return {
                "llm_response": {"content": "unreachable", "is_streaming": False},
                "data": {"references": []},
            }
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class _StreamingBlockingFakeRAG(_BlockingFakeRAG):
    """Variant whose (never-reached) result would have been a streaming one."""

    async def aquery_llm(self, query: str, *, param):
        try:
            await asyncio.sleep(30)
            self.completed = True

            async def _chunks():
                yield "unreachable"

            return {
                "llm_response": {
                    "is_streaming": True,
                    "response_iterator": _chunks(),
                },
                "data": {"references": []},
            }
        except asyncio.CancelledError:
            self.cancelled = True
            raise


# ---------------------------------------------------------------------------
# Global /query/* abort behavior
# ---------------------------------------------------------------------------


async def test_query_stream_aborts_retrieval_on_client_disconnect():
    rag = _StreamingBlockingFakeRAG()
    app = FastAPI()
    app.include_router(create_query_routes(rag, api_key=_API_KEY))

    status, _body = await _drive(
        app,
        method="POST",
        path="/query/stream",
        body={"query": "streaming please", "stream": True},
    )

    # 499 = client closed the request; nothing was generated.
    assert status == 499
    assert rag.cancelled is True
    assert rag.completed is False


async def test_query_non_stream_aborts_retrieval_on_client_disconnect():
    rag = _BlockingFakeRAG()
    app = FastAPI()
    app.include_router(create_query_routes(rag, api_key=_API_KEY))

    status, _body = await _drive(
        app,
        method="POST",
        path="/query",
        body={"query": "non streaming please"},
    )

    assert status == 499
    assert rag.cancelled is True
    assert rag.completed is False


async def test_query_data_aborts_retrieval_on_client_disconnect():
    rag = _BlockingFakeRAG()
    # /query/data calls aquery_data, not aquery_llm.
    async def aquery_data(query, *, param):
        try:
            await asyncio.sleep(30)
            rag.completed = True
            return {"status": "success", "message": "ok", "data": {}, "metadata": {}}
        except asyncio.CancelledError:
            rag.cancelled = True
            raise

    rag.aquery_data = aquery_data  # type: ignore[assignment]
    app = FastAPI()
    app.include_router(create_query_routes(rag, api_key=_API_KEY))

    status, _body = await _drive(
        app,
        method="POST",
        path="/query/data",
        body={"query": "data please"},
    )

    assert status == 499
    assert rag.cancelled is True
    assert rag.completed is False


async def test_query_stream_abort_without_starlette_listener_spec_2_4():
    """spec_version >= 2.4: Starlette starts no listen_for_disconnect task, so
    the cooperative poller is the ONLY pre-stream disconnect detector."""

    rag = _StreamingBlockingFakeRAG()
    app = FastAPI()
    app.include_router(create_query_routes(rag, api_key=_API_KEY))

    status, _body = await _drive(
        app,
        method="POST",
        path="/query/stream",
        body={"query": "streaming please", "stream": True},
        spec_version="2.4",
    )

    assert status == 499
    assert rag.cancelled is True
    assert rag.completed is False


class _NeverStartedUpstream:
    """An upstream LLM stream whose closure is observable even if never iterated.

    (An unstarted async *generator*'s ``finally`` never runs, so it cannot
    witness the pre-response release; a plain object with ``aclose`` can.)
    """

    def __init__(self) -> None:
        self.closed = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(30)
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed += 1


async def test_query_stream_disconnect_after_prepare_releases_upstream():
    """Disconnect landing between aquery_llm returning (upstream open) and the
    response generator starting must aclose the upstream deterministically and
    return 499 — not leave the stream to GC finalization."""

    upstream = _NeverStartedUpstream()

    class _InstantRAG:
        async def aquery_llm(self, query: str, *, param):
            return {
                "llm_response": {
                    "is_streaming": True,
                    "response_iterator": upstream,
                },
                "data": {"references": []},
            }

    app = FastAPI()
    app.include_router(create_query_routes(_InstantRAG(), api_key=_API_KEY))

    # aquery_llm resolves instantly (before any poll tick fires), so the
    # disconnect is only observable at the final pre-response check.
    status, _body = await _drive(
        app,
        method="POST",
        path="/query/stream",
        body={"query": "streaming please", "stream": True},
    )

    assert status == 499
    assert upstream.closed >= 1


async def test_query_stream_normal_path_is_unchanged():
    """Regression: no disconnect -> byte-identical NDJSON streaming output."""

    class _OkRAG:
        async def aquery_llm(self, query: str, *, param):
            async def chunks():
                yield "first "
                yield "second"

            return {
                "llm_response": {
                    "is_streaming": True,
                    "response_iterator": chunks(),
                },
                "data": {
                    "references": [
                        {"reference_id": "1", "file_path": "inputs/source.txt"}
                    ]
                },
            }

    app = FastAPI()
    app.include_router(create_query_routes(_OkRAG(), api_key=_API_KEY))

    # A receive() that NEVER disconnects: deliver the body once, then block on a
    # never-set event so is_disconnected() probes never see http.disconnect.
    never = asyncio.Event()
    delivered = {"body": False}

    async def receive():
        if not delivered["body"]:
            delivered["body"] = True
            return {
                "type": "http.request",
                "body": json.dumps(
                    {"query": "streaming please", "stream": True}
                ).encode(),
                "more_body": False,
            }
        await never.wait()  # blocks forever; probes are cancelled by the scope
        return {"type": "http.disconnect"}

    status: list[int] = []
    body_parts: list[bytes] = []

    async def send(message):
        if message["type"] == "http.response.start":
            status.append(message["status"])
        elif message["type"] == "http.response.body":
            body_parts.append(message.get("body", b""))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/query/stream",
        "raw_path": b"/query/stream",
        "query_string": b"",
        "headers": [
            (b"x-api-key", _API_KEY.encode()),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(json.dumps({"query": "streaming please", "stream": True}).encode())).encode()),
        ],
        "client": ("testclient", 123),
        "server": ("testserver", 80),
        "root_path": "",
    }
    await app(scope, receive, send)

    assert status == [200]
    lines = [ln for ln in b"".join(body_parts).decode().split("\n") if ln]
    parsed = [json.loads(ln) for ln in lines]
    assert parsed == [
        {"references": [{"reference_id": "1", "file_path": "inputs/source.txt"}]},
        {"response": "first "},
        {"response": "second"},
    ]


# ---------------------------------------------------------------------------
# KB-scoped /kbs/{kb_id}/query/stream abort behavior
# ---------------------------------------------------------------------------


class _BlockingKBProbe:
    """Registry probe that returns a blocking RAG for every KB."""

    def __init__(self) -> None:
        self.instances: dict[str, _BlockingFakeRAG] = {}

    async def build(self, record) -> _BlockingFakeRAG:
        rag = _BlockingFakeRAG()
        self.instances[record.id] = rag
        return rag

    async def finalize(self, rag) -> None:
        return None


def _build_kb_app(tmp_path: Path) -> tuple[FastAPI, _BlockingKBProbe]:
    from lightrag.api.document_lifecycle_service import DocumentLifecycleService
    from lightrag.api.job_service import JobService
    from lightrag.api.kb_service import KnowledgeBaseService
    from lightrag.api.lightrag_registry import LightRAGInstanceRegistry
    from lightrag.api.metadata_store import SQLiteMetadataStore

    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kb.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    document_service = DocumentLifecycleService(
        kb_service, metadata_store, tmp_path / "inputs"
    )
    job_service = JobService(kb_service, metadata_store)
    probe = _BlockingKBProbe()
    registry = LightRAGInstanceRegistry(kb_service, probe.build, probe.finalize)
    app = FastAPI()
    app.include_router(
        create_kb_routes(
            kb_service, registry, api_key=_API_KEY, job_service=job_service
        )
    )
    app.include_router(
        create_kb_document_routes(
            document_service, job_service, api_key=_API_KEY, registry=registry
        )
    )
    app.include_router(
        create_kb_query_routes(document_service, registry, api_key=_API_KEY)
    )
    return app, probe


async def test_kb_query_stream_aborts_retrieval_on_client_disconnect(tmp_path):
    from fastapi.testclient import TestClient

    app, probe = _build_kb_app(tmp_path)
    client = TestClient(app)
    kb = client.post(
        "/kbs", json={"id": "kb_abort", "name": "kb_abort"}, headers={"X-API-Key": _API_KEY}
    ).json()

    # Warm the instance so the blocking RAG exists, then issue the streaming
    # query through the raw ASGI driver to simulate a mid-request disconnect.
    _ = kb["id"]
    status, _body = await _drive(
        app,
        method="POST",
        path=f"/kbs/{kb['id']}/query/stream",
        body={"query": "streaming please", "stream": True},
    )

    assert status == 499
    rag = probe.instances[kb["id"]]
    assert rag.cancelled is True
    assert rag.completed is False
