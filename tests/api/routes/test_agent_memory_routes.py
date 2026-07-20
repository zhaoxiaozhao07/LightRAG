from __future__ import annotations

import importlib
import json
import sys
from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightrag.api.agent_query_service import AgentQueryService, AgentRunResult
from lightrag.api.chat_memory_service import (
    CHAT_MEMORY_FINAL_REQUEST_BUILDER_INVALID,
    MEMORY_QUERY_MAX_LENGTH,
)
from lightrag.sensitive_context import (
    CHAT_MEMORY_QUERY_LLM_EGRESS_NOT_ALLOWED,
    SensitiveContextPolicyError,
)
from tests.api.test_agent_chat_memory_service import (
    MEMORY_SENTINEL,
    _CountingMemoryHandle,
    _MemoryService,
    _build_case,
)

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
agent_routes = importlib.import_module("lightrag.api.routers.agent_routes")
sys.argv = _original_argv


pytestmark = pytest.mark.offline

QUERY = "How should the current authoritative evidence be applied?"
MEMORY_SCOPE = {"project_id": "project-1"}


class _RouteHandle:
    def __init__(self) -> None:
        self.resolve_calls = 0
        self.search_calls = 0
        self.info = {
            "enabled": True,
            "project_id": "project-1",
            "status": "injected",
            "fact_count": 1,
            "injected_count": 1,
            "truncated": False,
            "references": [
                {
                    "reference_id": "M1",
                    "fact_id": "memory-edge-1",
                    "valid_at": "2026-07-01",
                }
            ],
        }

    async def resolve_for_final_request(self, *_args: Any, **_kwargs: Any) -> None:
        self.resolve_calls += 1
        raise AssertionError("The Agent route must not resolve memory")

    async def search(self, **_kwargs: Any) -> None:
        self.search_calls += 1
        raise AssertionError("The Agent route must not search memory")

    def mark_not_used(self, reason: str) -> None:
        self.info.update(
            {
                "status": "not_used",
                "fact_count": 0,
                "injected_count": 0,
                "truncated": False,
                "references": [],
                "reason": reason,
            }
        )


class _FixedIterator(AsyncIterator[str]):
    def __init__(self, lines: list[str]) -> None:
        self._lines = iter(lines)

    def __aiter__(self) -> _FixedIterator:
        return self

    async def __anext__(self) -> str:
        try:
            return next(self._lines)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


RunImpl = Callable[..., Any]
StreamFactory = Callable[..., AsyncIterator[str]]


class _AgentServiceProbe:
    def __init__(
        self,
        *,
        timeline: list[str] | None = None,
        run_impl: RunImpl | None = None,
        stream_factory: StreamFactory | None = None,
    ) -> None:
        self.timeline = timeline if timeline is not None else []
        self.run_impl = run_impl
        self.stream_factory = stream_factory
        self.calls: list[dict[str, Any]] = []
        self.last_stream: AsyncIterator[str] | None = None

    async def run(
        self,
        *,
        request: Any,
        body: Any,
        stream: bool = False,
        sensitive_context: Any = None,
    ) -> AgentRunResult:
        self.timeline.append(f"service:{body.workflow}:run")
        self.calls.append(
            {
                "method": "run",
                "request": request,
                "body": body,
                "stream": stream,
                "sensitive_context": sensitive_context,
            }
        )
        if self.run_impl is not None:
            result = self.run_impl(body=body, sensitive_context=sensitive_context)
            if hasattr(result, "__await__"):
                result = await result
            return result
        return AgentRunResult(
            status="success",
            session_id="agent-route-session",
            answer="Answer grounded in [A1]",
            references=[{"reference_id": "A1", "kb_id": "kb1"}],
            steps_summary=[{"status": "ok"}],
            metadata={
                "workflow": body.workflow,
                "memory": sensitive_context.info,
                "service_marker": "preserved",
            },
        )

    def stream_events(
        self,
        *,
        request: Any,
        body: Any,
        sensitive_context: Any = None,
    ) -> AsyncIterator[str]:
        self.timeline.append(f"service:{body.workflow}:stream_events")
        self.calls.append(
            {
                "method": "stream_events",
                "request": request,
                "body": body,
                "sensitive_context": sensitive_context,
            }
        )
        if self.stream_factory is not None:
            stream = self.stream_factory(
                body=body,
                sensitive_context=sensitive_context,
            )
        else:
            stream = _FixedIterator(
                [
                    json.dumps(
                        {
                            "event": "references",
                            "references": [{"reference_id": "A1"}],
                        },
                        separators=(",", ":"),
                    )
                    + "\n",
                    json.dumps(
                        {
                            "event": "done",
                            "session_id": "agent-route-session",
                            "metadata": {"memory": sensitive_context.info},
                        },
                        separators=(",", ":"),
                    )
                    + "\n",
                ]
            )
        self.last_stream = stream
        return stream


def _build_client(
    monkeypatch: pytest.MonkeyPatch,
    service: Any,
    *,
    authorize: Callable[..., Any] | None = None,
) -> TestClient:
    async def allow_request() -> None:
        return None

    monkeypatch.setattr(
        agent_routes,
        "get_combined_auth_dependency",
        lambda _api_key: allow_request,
    )
    monkeypatch.setattr(
        agent_routes,
        "AgentQueryService",
        lambda **_kwargs: service,
    )
    if authorize is not None:
        monkeypatch.setattr(agent_routes, "authorize_memory_context", authorize)

    app = FastAPI()
    app.include_router(
        agent_routes.create_agent_routes(
            kb_service=SimpleNamespace(),
            document_service=SimpleNamespace(),
            registry=SimpleNamespace(),
        )
    )
    return TestClient(app)


def _prepare_real_service(
    monkeypatch: pytest.MonkeyPatch,
    service: AgentQueryService,
) -> None:
    monkeypatch.setattr(service, "_require_agent_access", lambda _request: None)

    async def effective_kbs(_request: Any, _candidate_kb_ids: Any):
        return await service._kb_service.list(include_deleted=False)

    monkeypatch.setattr(service, "_effective_kbs", effective_kbs)


def _memory_payload(workflow: str, *, query: str = QUERY) -> dict[str, Any]:
    return {
        "query": query,
        "workflow": workflow,
        "memory": MEMORY_SCOPE,
    }


@pytest.mark.parametrize("workflow", ["plan", "staged"])
@pytest.mark.parametrize(
    ("path", "service_method"),
    [
        ("/agent/query", "run"),
        ("/agent/query/stream", "stream_events"),
    ],
)
def test_agent_routes_authorize_once_before_service_and_pass_exact_handle(
    monkeypatch: pytest.MonkeyPatch,
    workflow: str,
    path: str,
    service_method: str,
) -> None:
    timeline: list[str] = []
    handle = _RouteHandle()
    authorization_calls: list[tuple[Any, Any, str]] = []

    async def authorize(request: Any, scope: Any, query: str) -> _RouteHandle:
        timeline.append("authorize")
        authorization_calls.append((request, scope, query))
        return handle

    service = _AgentServiceProbe(timeline=timeline)
    client = _build_client(monkeypatch, service, authorize=authorize)

    response = client.post(path, json=_memory_payload(workflow))

    assert response.status_code == 200
    assert len(authorization_calls) == 1
    assert authorization_calls[0][1].project_id == "project-1"
    assert authorization_calls[0][2] == QUERY
    assert timeline == ["authorize", f"service:{workflow}:{service_method}"]
    assert len(service.calls) == 1
    assert service.calls[0]["sensitive_context"] is handle
    assert handle.resolve_calls == 0
    assert handle.search_calls == 0

    if path.endswith("/stream"):
        events = [json.loads(line) for line in response.text.splitlines()]
        references = next(
            event["references"] for event in events if event["event"] == "references"
        )
        done = events[-1]
        assert done["metadata"]["memory"] == handle.info
    else:
        payload = response.json()
        references = payload["references"]
        assert payload["metadata"] == {
            "workflow": workflow,
            "memory": handle.info,
            "service_marker": "preserved",
        }

    assert references
    assert all(ref["reference_id"].startswith("A") for ref in references)
    assert all(not ref["reference_id"].startswith("M") for ref in references)


@pytest.mark.parametrize(
    ("workflow", "path", "reason", "expected_status"),
    [
        ("plan", "/agent/query", "clarification_required", "clarification_required"),
        ("staged", "/agent/query", "no_kb_evidence", "success"),
        (
            "plan",
            "/agent/query/stream",
            "clarification_required",
            "clarification_required",
        ),
        ("staged", "/agent/query/stream", "no_kb_evidence", "success"),
    ],
)
def test_agent_not_used_outcomes_are_preserved_without_route_search(
    monkeypatch: pytest.MonkeyPatch,
    workflow: str,
    path: str,
    reason: str,
    expected_status: str,
) -> None:
    handle = _RouteHandle()

    async def authorize(_request: Any, _scope: Any, _query: str) -> _RouteHandle:
        return handle

    async def run_impl(*, body: Any, sensitive_context: _RouteHandle):
        sensitive_context.mark_not_used(reason)
        return AgentRunResult(
            status=expected_status,
            session_id="not-used-session",
            clarification_question=(
                "Please clarify." if reason == "clarification_required" else None
            ),
            metadata={
                "workflow": body.workflow,
                "memory": sensitive_context.info,
            },
        )

    def stream_factory(
        *, body: Any, sensitive_context: _RouteHandle
    ) -> AsyncIterator[str]:
        sensitive_context.mark_not_used(reason)
        return _FixedIterator(
            [
                json.dumps(
                    {
                        "event": "done",
                        "session_id": "not-used-session",
                        "metadata": {"memory": sensitive_context.info},
                    },
                    separators=(",", ":"),
                )
                + "\n"
            ]
        )

    service = _AgentServiceProbe(
        run_impl=run_impl,
        stream_factory=stream_factory,
    )
    client = _build_client(monkeypatch, service, authorize=authorize)

    response = client.post(path, json=_memory_payload(workflow))

    assert response.status_code == 200
    if path.endswith("/stream"):
        memory_info = json.loads(response.text)["metadata"]["memory"]
    else:
        payload = response.json()
        assert payload["status"] == expected_status
        memory_info = payload["metadata"]["memory"]
    assert memory_info["status"] == "not_used"
    assert memory_info["reason"] == reason
    assert handle.resolve_calls == 0
    assert handle.search_calls == 0


@pytest.mark.parametrize(
    ("error_code", "expected_status"),
    [
        (CHAT_MEMORY_QUERY_LLM_EGRESS_NOT_ALLOWED, 403),
        ("chat_memory_query_too_long", 400),
        ("chat_memory_requires_final_synthesis", 400),
        (CHAT_MEMORY_FINAL_REQUEST_BUILDER_INVALID, 500),
    ],
)
def test_agent_non_stream_policy_errors_use_stable_content_free_mapping(
    monkeypatch: pytest.MonkeyPatch,
    error_code: str,
    expected_status: int,
) -> None:
    handle = _RouteHandle()
    secret = "https://provider.invalid/fact/MEMORY-SECRET"

    async def authorize(_request: Any, _scope: Any, _query: str) -> _RouteHandle:
        return handle

    async def run_impl(**_kwargs: Any) -> AgentRunResult:
        raise SensitiveContextPolicyError(error_code, secret)

    service = _AgentServiceProbe(run_impl=run_impl)
    client = _build_client(monkeypatch, service, authorize=authorize)

    response = client.post("/agent/query", json=_memory_payload("plan"))

    assert response.status_code == expected_status
    assert response.json() == {
        "detail": {
            "error_code": error_code,
            "message": error_code,
        }
    }
    assert secret not in response.text


@pytest.mark.parametrize("workflow", ["plan", "staged"])
def test_agent_memory_stream_maps_late_egress_denial_to_one_terminal_event(
    monkeypatch: pytest.MonkeyPatch,
    workflow: str,
) -> None:
    handle = _RouteHandle()
    secret = "provider endpoint and fact must not escape"

    async def authorize(_request: Any, _scope: Any, _query: str) -> _RouteHandle:
        return handle

    async def stream() -> AsyncIterator[str]:
        yield '{"event":"session_started","session_id":"late-error"}\n'
        raise SensitiveContextPolicyError(
            CHAT_MEMORY_QUERY_LLM_EGRESS_NOT_ALLOWED,
            secret,
        )

    def stream_factory(**_kwargs: Any) -> AsyncIterator[str]:
        return stream()

    service = _AgentServiceProbe(stream_factory=stream_factory)
    client = _build_client(monkeypatch, service, authorize=authorize)

    response = client.post(
        "/agent/query/stream",
        json=_memory_payload(workflow),
    )

    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0]["event"] == "session_started"
    assert events[-1] == {
        "event": "error",
        "error_code": CHAT_MEMORY_QUERY_LLM_EGRESS_NOT_ALLOWED,
        "status_code": 403,
        "message": CHAT_MEMORY_QUERY_LLM_EGRESS_NOT_ALLOWED,
    }
    assert sum(event["event"] == "error" for event in events) == 1
    assert secret not in response.text


def test_staged_real_service_stream_preserves_late_egress_policy_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory_endpoint = "https://memory-provider.invalid/v1"
    query_endpoint = "https://query-provider.invalid/v1"
    memory_service = _MemoryService(
        [{"uuid": "memory-edge", "fact": MEMORY_SENTINEL, "valid_at": None}],
        endpoint=memory_endpoint,
    )
    service, _body, rag, _tool, _request, audits = _build_case(
        monkeypatch,
        "staged",
        memory_service=memory_service,
        query_endpoint=query_endpoint,
    )
    _prepare_real_service(monkeypatch, service)
    handle = _CountingMemoryHandle(memory_service, query=QUERY)

    async def authorize(_request: Any, _scope: Any, _query: str):
        return handle

    client = _build_client(monkeypatch, service, authorize=authorize)

    response = client.post(
        "/agent/query/stream",
        json=_memory_payload("staged"),
    )

    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[-1] == {
        "event": "error",
        "error_code": CHAT_MEMORY_QUERY_LLM_EGRESS_NOT_ALLOWED,
        "status_code": 403,
        "message": CHAT_MEMORY_QUERY_LLM_EGRESS_NOT_ALLOWED,
    }
    assert sum(event["event"] == "error" for event in events) == 1
    assert all(event.get("error_code") != "agent_error" for event in events)
    assert memory_service.calls == []
    assert rag.query_calls == []
    assert not any(record["event"] == "agent_session_failed" for record in audits)
    for secret in (memory_endpoint, query_endpoint, MEMORY_SENTINEL):
        assert secret not in response.text


@pytest.mark.parametrize("workflow", ["plan", "staged"])
def test_real_agent_route_blank_final_evidence_skips_memory_and_final_llm(
    monkeypatch: pytest.MonkeyPatch,
    workflow: str,
) -> None:
    if workflow == "staged":
        monkeypatch.setattr(
            "lightrag.api.agent_staged_service.agent_staged_max_retrievals",
            lambda: 1,
        )
    memory_service = _MemoryService(
        [{"uuid": "memory-edge", "fact": MEMORY_SENTINEL, "valid_at": None}]
    )
    service, _body, rag, _tool, _request, audits = _build_case(
        monkeypatch,
        workflow,
        memory_service=memory_service,
        chunk_contents=[None, "", " \t\r\n ", 7],
    )
    _prepare_real_service(monkeypatch, service)
    handle = _CountingMemoryHandle(memory_service, query=QUERY)

    async def authorize(_request: Any, _scope: Any, _query: str):
        return handle

    client = _build_client(monkeypatch, service, authorize=authorize)

    response = client.post("/agent/query", json=_memory_payload(workflow))

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["answer"] == "未检索到可用于回答的证据。"
    assert payload["references"] == []
    assert payload["metadata"]["memory"]["status"] == "not_used"
    assert payload["metadata"]["memory"]["reason"] == "no_kb_evidence"
    assert handle.resolve_calls == 0
    assert memory_service.calls == []
    assert rag.query_calls == []
    completion = next(
        record["metadata"]
        for record in audits
        if record["event"] == "agent_query_completed"
    )
    assert completion["reference_count"] == 0
    assert MEMORY_SENTINEL not in repr(completion)


def test_agent_memory_non_stream_generic_500_is_content_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = _RouteHandle()
    secret = "https://provider.invalid/fact/MEMORY-SECRET"
    log_calls: list[tuple[Any, ...]] = []

    async def authorize(_request: Any, _scope: Any, _query: str) -> _RouteHandle:
        return handle

    async def run_impl(**_kwargs: Any) -> AgentRunResult:
        raise RuntimeError(secret)

    def record_error(*args: Any, **kwargs: Any) -> None:
        log_calls.append((*args, kwargs))

    monkeypatch.setattr(agent_routes.logger, "error", record_error)
    service = _AgentServiceProbe(run_impl=run_impl)
    client = _build_client(monkeypatch, service, authorize=authorize)

    response = client.post("/agent/query", json=_memory_payload("plan"))

    assert response.status_code == 500
    assert response.json() == {"detail": "Agent query failed"}
    assert secret not in response.text
    assert secret not in repr(log_calls)


@pytest.mark.parametrize("workflow", ["plan", "staged"])
@pytest.mark.parametrize("path", ["/agent/query", "/agent/query/stream"])
def test_agent_query_length_is_rejected_before_service_or_stream_response(
    monkeypatch: pytest.MonkeyPatch,
    workflow: str,
    path: str,
) -> None:
    service = _AgentServiceProbe()
    client = _build_client(monkeypatch, service)
    oversized_query = "q" * (MEMORY_QUERY_MAX_LENGTH + 1)

    response = client.post(
        path,
        json=_memory_payload(workflow, query=oversized_query),
    )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {
        "detail": (f"Chat memory query exceeds {MEMORY_QUERY_MAX_LENGTH} characters")
    }
    assert service.calls == []


@pytest.mark.parametrize("workflow", ["plan", "staged"])
def test_agent_no_memory_stream_keeps_exact_iterator_bytes_and_done_shape(
    monkeypatch: pytest.MonkeyPatch,
    workflow: str,
) -> None:
    expected_lines = [
        '{"event": "session_started", "session_id": "plain"}\n',
        '{"event": "done", "session_id": "plain"}\n',
    ]
    fixed_stream = _FixedIterator(expected_lines)

    def stream_factory(**_kwargs: Any) -> AsyncIterator[str]:
        return fixed_stream

    service = _AgentServiceProbe(stream_factory=stream_factory)
    original_authorize = agent_routes.authorize_memory_context
    authorization_calls: list[tuple[Any, Any, str]] = []

    async def authorize(request: Any, scope: Any, query: str):
        authorization_calls.append((request, scope, query))
        return await original_authorize(request, scope, query)

    def forbidden_wrapper(_events: AsyncIterator[str]) -> AsyncIterator[str]:
        raise AssertionError("No-memory Agent streams must not be wrapped")

    monkeypatch.setattr(
        agent_routes,
        "_stream_sensitive_context_errors",
        forbidden_wrapper,
    )
    client = _build_client(monkeypatch, service, authorize=authorize)

    response = client.post(
        "/agent/query/stream",
        json={"query": QUERY, "workflow": workflow},
    )

    assert response.status_code == 200
    assert response.content == "".join(expected_lines).encode()
    assert len(authorization_calls) == 1
    assert authorization_calls[0][1] is None
    assert service.calls[0]["sensitive_context"] is None
    done = json.loads(response.text.splitlines()[-1])
    assert done == {"event": "done", "session_id": "plain"}
