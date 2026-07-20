from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from lightrag.api.agent_query_service import AgentQueryRequest, AgentQueryService
from lightrag.api.chat_memory_routing import memory_audit_fields
from lightrag.api.chat_memory_service import (
    CHAT_MEMORY_AGENT_POLICY_SUFFIX,
    CHAT_MEMORY_UNIVERSAL_POLICY,
    AuthorizedChatMemoryHandle,
    ChatMemoryConfig,
    ChatMemoryUnavailableError,
)
from lightrag.api.enterprise_auth import (
    AuthorizationService,
    Principal,
    SYSTEM_ROLE_SUPER_ADMIN,
    USER_STATUS_ACTIVE,
)
from lightrag.api.kb_service import KnowledgeBaseRecord, utc_now_iso
from lightrag.api.query_tool_service import QueryToolResult
from lightrag.base import QueryParam
from lightrag.sensitive_context import SensitiveContextPolicyError


pytestmark = pytest.mark.offline

QUERY = "How should the authoritative evidence be applied?"
QUERY_ENDPOINT = "https://llm.example/v1"
MEMORY_SENTINEL = "MEMORY-FACT-SENTINEL-4B"
USER_PROMPT_SENTINEL = "USER-PROMPT-SENTINEL-4B"


class _CharTokenizer:
    def encode(self, content: str) -> list[int]:
        return [ord(character) for character in content]


class _MemoryService:
    def __init__(
        self,
        facts: list[dict[str, Any]] | None = None,
        *,
        endpoint: str = QUERY_ENDPOINT,
        error: Exception | None = None,
    ) -> None:
        self.config = ChatMemoryConfig(
            enabled=True,
            llm_base_url=endpoint,
            prompt_max_tokens=100_000,
            prompt_max_chars=100_000,
        )
        self.facts = list(facts or [])
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def search(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(dict(kwargs))
        if self.error is not None:
            raise self.error
        return list(self.facts)


class _CountingMemoryHandle(AuthorizedChatMemoryHandle):
    def __init__(self, service: _MemoryService, *, query: str = QUERY) -> None:
        super().__init__(
            service,
            user_id="user-1",
            project_id="project-1",
            query=query,
            limit=5,
            query_llm_endpoint=None,
        )
        self.resolve_calls = 0
        self.bound_endpoints: list[str | None] = []

    def bind_final_llm_endpoint(self, endpoint: str | None) -> None:
        self.bound_endpoints.append(endpoint)
        super().bind_final_llm_endpoint(endpoint)

    async def resolve_for_final_request(
        self,
        tokenizer: Any,
        max_total_tokens: int,
        build_final_request: Any,
        policy_suffix: str = "",
    ):
        self.resolve_calls += 1
        return await super().resolve_for_final_request(
            tokenizer,
            max_total_tokens,
            build_final_request,
            policy_suffix=policy_suffix,
        )


class _FakeRAG:
    def __init__(
        self,
        agent_responses: list[str],
        *,
        memory_service: _MemoryService | None = None,
        query_endpoint: str = QUERY_ENDPOINT,
        tokenizer: Any = ...,
        query_error: Exception | None = None,
    ) -> None:
        self.agent_responses = list(agent_responses)
        self.memory_service = memory_service
        self.query_endpoint = query_endpoint
        self.tokenizer = _CharTokenizer() if tokenizer is ... else tokenizer
        self.query_error = query_error
        self.agent_prompts: list[str] = []
        self.agent_search_counts: list[int] = []
        self.query_calls: list[dict[str, Any]] = []
        self.query_search_counts: list[int] = []

    def _search_count(self) -> int:
        return len(self.memory_service.calls) if self.memory_service else 0

    def _build_global_config(self) -> dict[str, Any]:
        async def agent_func(prompt: str, **_kwargs: Any) -> str:
            self.agent_search_counts.append(self._search_count())
            self.agent_prompts.append(prompt)
            if not self.agent_responses:
                raise AssertionError("Unexpected Agent decision call")
            return self.agent_responses.pop(0)

        async def query_func(query: str, stream: bool = False, **kwargs: Any):
            self.query_search_counts.append(self._search_count())
            self.query_calls.append({"query": query, "stream": stream, **kwargs})
            if self.query_error is not None:
                raise self.query_error
            if stream:

                async def chunks():
                    yield "final "
                    yield "answer [A1]"

                return chunks()
            return "final answer [A1]"

        return {
            "role_llm_funcs": {"agent": agent_func, "query": query_func},
            "tokenizer": self.tokenizer,
            "max_total_tokens": 100_000,
            "llm_cache_identities": {
                "query": {
                    "role": "query",
                    "binding": "openai",
                    "model": "query-model",
                    "host": self.query_endpoint,
                }
            },
        }


class _QueryTool:
    def __init__(
        self,
        rag: _FakeRAG,
        *,
        empty: bool = False,
        chunk_contents: list[Any] | None = None,
    ) -> None:
        self.rag = rag
        self.empty = empty
        self.chunk_contents = chunk_contents
        self.calls: list[dict[str, Any]] = []

    async def get_rag(self, _kb_id: str) -> _FakeRAG:
        return self.rag

    async def retrieve_serial(self, **kwargs: Any) -> QueryToolResult:
        self.calls.append(dict(kwargs))
        call_number = len(self.calls)
        chunks = []
        if not self.empty:
            contents = self.chunk_contents
            if contents is None:
                contents = [f"AUTHORITATIVE KB EVIDENCE {call_number}"]
            chunks = [
                {
                    "kb_id": kwargs["kb_ids"][0],
                    "chunk_id": f"chunk-{call_number}-{index}",
                    "file_path": f"authoritative-{call_number}-{index}.md",
                    "content": content,
                }
                for index, content in enumerate(contents, start=1)
            ]
        return QueryToolResult(
            chunks=chunks,
            rag=self.rag,
            param=QueryParam(mode=kwargs["mode"], max_total_tokens=100_000),
            queried_kb_ids=list(kwargs["kb_ids"]),
            per_kb_chunk_counts={kwargs["kb_ids"][0]: len(chunks)},
        )


class _KBService:
    def __init__(self) -> None:
        now = utc_now_iso()
        self.records = [
            KnowledgeBaseRecord(
                id="kb1",
                name="Authoritative KB",
                description="Current authoritative evidence",
                workspace="kb_kb1",
                status="active",
                active_config_version_id=None,
                owner_id=None,
                tenant_id=None,
                visibility="private",
                created_at=now,
                updated_at=now,
                metadata={"agent_priority": 10},
            )
        ]

    async def list(self, *, include_deleted: bool = False):
        return self.records


def _request(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(
        "lightrag.api.agent_query_service.enterprise_auth_enabled", lambda: True
    )
    monkeypatch.setattr(
        "lightrag.api.agent_query_service.agent_query_enabled", lambda: True
    )
    principal = Principal(
        user_id="admin",
        username="admin",
        system_role=SYSTEM_ROLE_SUPER_ADMIN,
        status=USER_STATUS_ACTIVE,
        tenant_id=None,
        tenant_roles={},
        can_create_kb=True,
        can_use_bypass_query=True,
        token_version=1,
        auth_method="api_key",
        metadata={},
        can_use_agent_query=True,
    )
    state = SimpleNamespace(
        principal=principal,
        enterprise_authorization_service=AuthorizationService(
            metadata_store=SimpleNamespace()
        ),
    )
    return SimpleNamespace(state=state, app=SimpleNamespace(state=state))


def _install_audit_recorder(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    async def record(
        _request: Any,
        event: str,
        *,
        metadata: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> None:
        records.append({"event": event, "metadata": metadata or {}})

    monkeypatch.setattr(
        "lightrag.api.agent_query_service.append_enterprise_audit_event", record
    )
    monkeypatch.setattr(
        "lightrag.api.agent_staged_service.append_enterprise_audit_event", record
    )
    return records


def _plan_response(*, clarification: bool = False) -> str:
    if clarification:
        return json.dumps(
            {
                "type": "plan",
                "clarification_required": True,
                "clarification_question": "Please provide the missing constraint.",
                "steps": [],
            }
        )
    return json.dumps(
        {
            "type": "plan",
            "clarification_required": False,
            "steps": [
                {
                    "step_index": 1,
                    "title": "Retrieve authoritative evidence",
                    "query": "Retrieve current authoritative evidence",
                    "kb_ids": ["kb1"],
                    "mode": "mix",
                    "priority": "P0",
                }
            ],
        }
    )


def _staged_responses(*, clarification: bool = False) -> list[str]:
    requirement = {
        "type": "requirement",
        "clarification_required": clarification,
        "clarification_question": (
            "Please provide the target application." if clarification else None
        ),
        "application": "target application",
        "conditions": ["current conditions"],
        "target_properties": (
            []
            if clarification
            else [{"name": "safety", "why": "required", "priority": "P0"}]
        ),
        "constraints": [],
    }
    if clarification:
        return [json.dumps(requirement)]
    return [
        json.dumps(requirement),
        json.dumps(
            {
                "type": "skeleton_plan",
                "kb_roles": {"kb1": "reference_formula"},
                "steps": [
                    {
                        "step_index": 1,
                        "title": "Retrieve a current baseline",
                        "query": "Retrieve a current authoritative baseline",
                        "kb_ids": ["kb1"],
                        "mode": "mix",
                        "priority": "P0",
                    }
                ],
            }
        ),
        json.dumps(
            {
                "type": "skeleton",
                "components": [],
                "open_questions": [],
                "rationale": "No unsupported formula inference.",
            }
        ),
        json.dumps(
            {
                "type": "verdicts",
                "verdicts": [
                    {
                        "property": "safety",
                        "verdict": "supported",
                        "evidence_refs": ["A2"],
                        "note": "Supported by current KB evidence.",
                    }
                ],
            }
        ),
    ]


def _agent_responses(workflow: str, *, clarification: bool = False) -> list[str]:
    if workflow == "staged":
        return _staged_responses(clarification=clarification)
    return [_plan_response(clarification=clarification)]


def _build_case(
    monkeypatch: pytest.MonkeyPatch,
    workflow: str,
    *,
    memory_service: _MemoryService | None = None,
    query_endpoint: str = QUERY_ENDPOINT,
    tokenizer: Any = ...,
    empty: bool = False,
    chunk_contents: list[Any] | None = None,
    query_error: Exception | None = None,
    clarification: bool = False,
):
    audits = _install_audit_recorder(monkeypatch)
    rag = _FakeRAG(
        _agent_responses(workflow, clarification=clarification),
        memory_service=memory_service,
        query_endpoint=query_endpoint,
        tokenizer=tokenizer,
        query_error=query_error,
    )
    tool = _QueryTool(rag, empty=empty, chunk_contents=chunk_contents)
    service = AgentQueryService(kb_service=_KBService(), query_tool_service=tool)
    body = AgentQueryRequest(
        query=QUERY,
        workflow=workflow,
        candidate_kb_ids=["kb1"],
        user_prompt=USER_PROMPT_SENTINEL,
        conversation_history=[
            {"role": "user", "content": "Earlier user context"},
            {"role": "assistant", "content": "Earlier assistant context"},
        ],
    )
    return service, body, rag, tool, _request(monkeypatch), audits


def _completion_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    return next(
        record["metadata"]
        for record in reversed(records)
        if record["event"] == "agent_query_completed"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow", ["plan", "staged"])
async def test_agent_memory_is_final_only_and_preserves_a_m_trust_boundary(
    monkeypatch: pytest.MonkeyPatch, workflow: str
) -> None:
    memory_service = _MemoryService(
        [
            {
                "uuid": "memory-edge-1",
                "fact": MEMORY_SENTINEL,
                "valid_at": "2026-07-01",
            }
        ]
    )
    service, body, rag, _tool, request, audits = _build_case(
        monkeypatch, workflow, memory_service=memory_service
    )
    handle = _CountingMemoryHandle(memory_service)

    result = await service.run(
        request=request,
        body=body,
        sensitive_context=handle,
    )

    assert handle.resolve_calls == 1
    assert handle.bound_endpoints == [QUERY_ENDPOINT]
    assert len(memory_service.calls) == 1
    assert rag.agent_search_counts and set(rag.agent_search_counts) == {0}
    assert all(MEMORY_SENTINEL not in prompt for prompt in rag.agent_prompts)
    assert rag.query_search_counts == [1]
    assert len(rag.query_calls) == 1
    query_call = rag.query_calls[0]
    assert query_call["_sensitive"] is True
    assert query_call["history_messages"] == body.conversation_history

    prompt = query_call["system_prompt"]
    context_index = prompt.index("---Context---")
    policy_index = prompt.index(CHAT_MEMORY_UNIVERSAL_POLICY)
    assert USER_PROMPT_SENTINEL in prompt[:policy_index]
    assert policy_index < context_index
    assert CHAT_MEMORY_AGENT_POLICY_SUFFIX in prompt[policy_index:context_index]
    assert MEMORY_SENTINEL in prompt[context_index:]
    assert '"reference_id":"M1"' in prompt[context_index:]
    assert "[A1]" in prompt[context_index:]

    assert result.metadata["memory"] is handle.info
    assert handle.info["status"] == "injected"
    assert handle.info["references"][0]["reference_id"] == "M1"
    assert result.references
    assert all(ref["reference_id"].startswith("A") for ref in result.references)
    assert all(ref["reference_id"] != "M1" for ref in result.references)
    if workflow == "staged":
        assert all(
            ref.startswith("A")
            for verdict in result.metadata["property_verdicts"]
            for ref in verdict["evidence_refs"]
        )

    completion = _completion_audit(audits)
    expected_audit = memory_audit_fields(handle.info)
    assert {key: completion[key] for key in expected_audit} == expected_audit
    assert "references" not in completion
    assert "memory-edge-1" not in repr(completion)
    assert body.user_prompt == USER_PROMPT_SENTINEL


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow", ["plan", "staged"])
@pytest.mark.parametrize(
    ("status", "expected_searches", "enabled", "reason"),
    [
        ("empty", 1, True, None),
        ("unavailable", 1, False, "unavailable"),
        ("budget_exhausted", 0, True, None),
    ],
)
async def test_agent_memory_terminal_statuses_still_use_sensitive_final_llm(
    monkeypatch: pytest.MonkeyPatch,
    workflow: str,
    status: str,
    expected_searches: int,
    enabled: bool,
    reason: str | None,
) -> None:
    error = (
        ChatMemoryUnavailableError("backend unavailable")
        if status == "unavailable"
        else None
    )
    facts = (
        []
        if status == "empty"
        else [{"uuid": "edge", "fact": MEMORY_SENTINEL, "valid_at": None}]
    )
    memory_service = _MemoryService(facts, error=error)
    tokenizer = None if status == "budget_exhausted" else ...
    service, body, rag, _tool, request, audits = _build_case(
        monkeypatch,
        workflow,
        memory_service=memory_service,
        tokenizer=tokenizer,
    )
    handle = _CountingMemoryHandle(memory_service)

    result = await service.run(
        request=request,
        body=body,
        sensitive_context=handle,
    )

    assert handle.resolve_calls == 1
    assert len(memory_service.calls) == expected_searches
    assert handle.info["status"] == status
    assert handle.info["enabled"] is enabled
    if reason is not None:
        assert handle.info["reason"] == reason
    assert result.metadata["memory"] is handle.info
    assert rag.query_calls[0]["_sensitive"] is True
    assert MEMORY_SENTINEL not in rag.query_calls[0]["system_prompt"]
    completion = _completion_audit(audits)
    expected_audit = memory_audit_fields(handle.info)
    assert {key: completion[key] for key in expected_audit} == expected_audit


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow", ["plan", "staged"])
async def test_agent_clarification_marks_memory_not_used_without_search(
    monkeypatch: pytest.MonkeyPatch, workflow: str
) -> None:
    memory_service = _MemoryService(
        [{"uuid": "edge", "fact": MEMORY_SENTINEL, "valid_at": None}]
    )
    service, body, rag, tool, request, _audits = _build_case(
        monkeypatch,
        workflow,
        memory_service=memory_service,
        clarification=True,
    )
    handle = _CountingMemoryHandle(memory_service)

    result = await service.run(
        request=request,
        body=body,
        sensitive_context=handle,
    )

    assert result.status == "clarification_required"
    assert result.metadata["memory"] is handle.info
    assert handle.info["status"] == "not_used"
    assert handle.info["reason"] == "clarification_required"
    assert handle.resolve_calls == 0
    assert memory_service.calls == []
    assert tool.calls == []
    assert rag.query_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow", ["plan", "staged"])
async def test_agent_no_kb_evidence_marks_memory_not_used_without_search(
    monkeypatch: pytest.MonkeyPatch, workflow: str
) -> None:
    if workflow == "staged":
        monkeypatch.setattr(
            "lightrag.api.agent_staged_service.agent_staged_max_retrievals",
            lambda: 1,
        )
    memory_service = _MemoryService(
        [{"uuid": "edge", "fact": MEMORY_SENTINEL, "valid_at": None}]
    )
    service, body, rag, _tool, request, audits = _build_case(
        monkeypatch,
        workflow,
        memory_service=memory_service,
        empty=True,
    )
    handle = _CountingMemoryHandle(memory_service)

    result = await service.run(
        request=request,
        body=body,
        sensitive_context=handle,
    )

    assert result.status == "success"
    assert result.references == []
    assert result.metadata["memory"] is handle.info
    assert handle.info["status"] == "not_used"
    assert handle.info["reason"] == "no_kb_evidence"
    assert handle.resolve_calls == 0
    assert memory_service.calls == []
    assert rag.query_calls == []
    completion = _completion_audit(audits)
    assert completion["memory_status"] == "not_used"
    assert completion["memory_reason"] == "no_kb_evidence"


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow", ["plan", "staged"])
async def test_agent_blank_final_evidence_skips_memory_and_final_llm(
    monkeypatch: pytest.MonkeyPatch, workflow: str
) -> None:
    if workflow == "staged":
        monkeypatch.setattr(
            "lightrag.api.agent_staged_service.agent_staged_max_retrievals",
            lambda: 1,
        )
    memory_service = _MemoryService(
        [{"uuid": "edge", "fact": MEMORY_SENTINEL, "valid_at": None}]
    )
    service, body, rag, _tool, request, audits = _build_case(
        monkeypatch,
        workflow,
        memory_service=memory_service,
        chunk_contents=[None, "", " \t\r\n ", 7],
    )
    handle = _CountingMemoryHandle(memory_service)

    result = await service.run(
        request=request,
        body=body,
        sensitive_context=handle,
    )

    assert result.status == "success"
    assert result.answer == "未检索到可用于回答的证据。"
    assert result.references == []
    assert result.metadata["memory"] is handle.info
    assert handle.info["status"] == "not_used"
    assert handle.info["reason"] == "no_kb_evidence"
    assert handle.resolve_calls == 0
    assert memory_service.calls == []
    assert rag.query_calls == []
    assert all(summary["chunk_count"] == 0 for summary in result.steps_summary)
    completion = _completion_audit(audits)
    assert completion["reference_count"] == 0
    assert completion["memory_status"] == "not_used"
    assert completion["memory_reason"] == "no_kb_evidence"
    assert MEMORY_SENTINEL not in repr(completion)


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow", ["plan", "staged"])
async def test_agent_hard_memory_policy_error_propagates_without_final_llm(
    monkeypatch: pytest.MonkeyPatch, workflow: str
) -> None:
    memory_service = _MemoryService(
        [{"uuid": "edge", "fact": MEMORY_SENTINEL, "valid_at": None}],
        endpoint="https://memory.example/v1",
    )
    service, body, rag, _tool, request, audits = _build_case(
        monkeypatch,
        workflow,
        memory_service=memory_service,
        query_endpoint="https://query.example/v1",
    )
    handle = _CountingMemoryHandle(memory_service)

    with pytest.raises(SensitiveContextPolicyError) as exc_info:
        await service.run(
            request=request,
            body=body,
            sensitive_context=handle,
        )

    assert exc_info.value.error_code == "chat_memory_query_llm_egress_not_allowed"
    assert handle.resolve_calls == 1
    assert memory_service.calls == []
    assert rag.query_calls == []
    assert not any(record["event"] == "agent_session_failed" for record in audits)


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow", ["plan", "staged"])
async def test_agent_memory_generic_stream_failure_is_content_free(
    monkeypatch: pytest.MonkeyPatch,
    workflow: str,
) -> None:
    secret = "https://provider.invalid/fact/MEMORY-SECRET"
    memory_service = _MemoryService(
        [{"uuid": "edge", "fact": MEMORY_SENTINEL, "valid_at": None}]
    )
    service, body, _rag, _tool, request, audits = _build_case(
        monkeypatch,
        workflow,
        memory_service=memory_service,
        query_error=RuntimeError(secret),
    )
    handle = _CountingMemoryHandle(memory_service)
    log_calls: list[tuple[Any, ...]] = []

    def record_error(*args: Any, **kwargs: Any) -> None:
        log_calls.append((*args, kwargs))

    monkeypatch.setattr(
        "lightrag.api.agent_query_service.logger.error", record_error
    )
    events: list[dict[str, Any]] = []

    async for line in service.stream_events(
        request=request,
        body=body,
        sensitive_context=handle,
    ):
        events.append(json.loads(line))

    assert events[-1] == {
        "event": "error",
        "error_code": "agent_error",
        "message": "Agent query failed",
    }
    failed = next(
        record["metadata"]
        for record in audits
        if record["event"] == "agent_session_failed"
    )
    assert failed == {
        "error": "Agent query failed",
        "error_code": "agent_error",
        "status_code": None,
    }
    assert secret not in repr(events)
    assert secret not in repr(failed)
    assert secret not in repr(log_calls)


@pytest.mark.asyncio
async def test_agent_stream_propagates_hard_memory_policy_error_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory_service = _MemoryService(
        [{"uuid": "edge", "fact": MEMORY_SENTINEL, "valid_at": None}],
        endpoint="https://memory.example/v1",
    )
    service, body, _rag, _tool, request, _audits = _build_case(
        monkeypatch,
        "plan",
        memory_service=memory_service,
        query_endpoint="https://query.example/v1",
    )
    handle = _CountingMemoryHandle(memory_service)
    emitted: list[dict[str, Any]] = []

    with pytest.raises(SensitiveContextPolicyError) as exc_info:
        async for line in service.stream_events(
            request=request,
            body=body,
            sensitive_context=handle,
        ):
            emitted.append(json.loads(line))

    assert emitted[0]["event"] == "session_started"
    assert exc_info.value.error_code == "chat_memory_query_llm_egress_not_allowed"
    assert handle.resolve_calls == 1
    assert memory_service.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow", ["plan", "staged"])
async def test_agent_memory_stream_done_event_has_only_memory_metadata(
    monkeypatch: pytest.MonkeyPatch, workflow: str
) -> None:
    memory_service = _MemoryService(
        [{"uuid": "edge", "fact": MEMORY_SENTINEL, "valid_at": None}]
    )
    service, body, rag, _tool, request, _audits = _build_case(
        monkeypatch, workflow, memory_service=memory_service
    )
    handle = _CountingMemoryHandle(memory_service)
    events: list[dict[str, Any]] = []

    async for line in service.stream_events(
        request=request,
        body=body,
        sensitive_context=handle,
    ):
        events.append(json.loads(line))

    done = events[-1]
    assert done == {
        "event": "done",
        "session_id": done["session_id"],
        "metadata": {"memory": handle.info},
    }
    assert handle.resolve_calls == 1
    assert len(memory_service.calls) == 1
    assert rag.query_calls[0]["_sensitive"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow", ["plan", "staged"])
async def test_agent_no_memory_stream_and_audit_shapes_remain_compatible(
    monkeypatch: pytest.MonkeyPatch, workflow: str
) -> None:
    service, body, rag, _tool, request, audits = _build_case(monkeypatch, workflow)
    events: list[dict[str, Any]] = []

    async for line in service.stream_events(request=request, body=body):
        events.append(json.loads(line))

    done = events[-1]
    assert set(done) == {"event", "session_id"}
    assert done["event"] == "done"
    assert all("memory" not in (event.get("metadata") or {}) for event in events)
    assert "_sensitive" not in rag.query_calls[0]
    completion = _completion_audit(audits)
    assert not any(key.startswith("memory_") for key in completion)
