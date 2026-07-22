"""Server-side lazy Chat Memory integration on KB query routes."""

from __future__ import annotations

# ruff: noqa: E402 - LightRAG API modules parse sys.argv at import time.

import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from lightrag.api.auth import auth_handler
from lightrag.api.bilingual_query_service import BILINGUAL_PREPROCESS_SYSTEM_PROMPT
from lightrag.api.chat_memory_service import (
    CHAT_MEMORY_FINAL_REQUEST_BUILDER_INVALID,
    AuthorizedChatMemoryHandle,
    ChatMemoryConfig,
    ChatMemoryUnavailableError,
)
from lightrag.api.enterprise_auth import (
    AuditService,
    AuthorizationService,
    ChatConversationService,
    SystemSettingsService,
    UserService,
)
from lightrag.api.kb_service import KnowledgeBaseService
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry
from lightrag.api.metadata_store import SQLiteMetadataStore
from lightrag.api.routers.chat_routes import create_chat_routes
from lightrag.api.routers.kb_query_routes import create_kb_query_routes
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.sensitive_context import (
    SensitiveContextPayload,
    SensitiveContextPolicyError,
    serialize_sensitive_final_request,
)
sys.argv = _original_argv

pytestmark = pytest.mark.offline

_API_KEY = "test-key"


class FakeRAG:
    def __init__(
        self,
        workspace: str,
        *,
        events: list[str] | None = None,
        empty_data: bool = False,
        final_endpoint: str = "https://memory.example/v1",
        policy_error: SensitiveContextPolicyError | None = None,
    ):
        self.workspace = workspace
        self.events = events if events is not None else []
        self.empty_data = empty_data
        self.final_endpoint = final_endpoint
        self.policy_error = policy_error
        self.query_params: list = []
        self.sensitive_contexts: list = []
        self.memory_payloads: list[SensitiveContextPayload | None] = []
        self.final_llm_calls: list[dict[str, Any]] = []
        self.tokenizers: list[_CharTokenizer] = []
        self.kb_active_query_config: dict = {}
        self.kb_active_config_version_id = None
        self.kb_active_parser_hash = None
        self.kb_active_index_hash = None
        self.kb_active_query_hash = None
        self.llm_response_cache = None

    async def aquery_llm(self, query: str, *, param, sensitive_context=None):
        self.events.append(f"single_retrieval:{self.workspace}")
        self.query_params.append(param)
        self.sensitive_contexts.append(sensitive_context)
        if self.policy_error is not None:
            raise self.policy_error
        if self.empty_data:
            if sensitive_context is not None:
                sensitive_context.mark_not_used("no_kb_evidence")
            return {
                "llm_response": {"content": "", "is_streaming": False},
                "data": {"references": [], "chunks": []},
            }
        if sensitive_context is not None:
            sensitive_context.bind_final_llm_endpoint(self.final_endpoint)

            def build_final_request(payload):
                system_prompt = "AUTHORITATIVE KB EVIDENCE"
                if payload is not None:
                    system_prompt += (
                        f"\n{payload.trusted_policy}\n{payload.context_data}"
                    )
                return serialize_sensitive_final_request(
                    system_prompt,
                    query,
                    param.conversation_history,
                )

            payload = await sensitive_context.resolve_for_final_request(
                _CharTokenizer(),
                param.max_total_tokens,
                build_final_request,
            )
            self.memory_payloads.append(payload)
        self.events.append("final_llm")
        if param.stream:

            async def chunks():
                yield "answer: "
                yield query

            return {
                "llm_response": {
                    "content": "",
                    "is_streaming": True,
                    "response_iterator": chunks(),
                },
                "data": {
                    "references": [
                        {"reference_id": "1", "file_path": "single.pdf"}
                    ],
                    "chunks": [
                        {
                            "reference_id": "1",
                            "content": "single authoritative evidence",
                        }
                    ],
                },
            }
        return {
            "llm_response": {
                "content": f"answer: {query}",
                "is_streaming": False,
            },
            "data": {
                "references": [
                    {"reference_id": "1", "file_path": "single.pdf"}
                ],
                "chunks": [
                    {
                        "reference_id": "1",
                        "content": "single authoritative evidence",
                    }
                ],
            },
        }

    async def aquery_data(self, query: str, *, param):
        self.events.append(f"retrieve:{self.workspace}:{query}")
        self.query_params.append(param)
        chunks = []
        if not self.empty_data:
            chunks = [
                {
                    "reference_id": "1",
                    "chunk_id": f"{self.workspace}-{len(self.query_params)}",
                    "content": f"authoritative evidence from {self.workspace}: {query}",
                    "file_path": f"{self.workspace}/source.pdf",
                }
            ]
        return {
            "status": "success",
            "message": "ok",
            "data": {
                "entities": [],
                "relationships": [],
                "chunks": chunks,
                "references": [],
            },
            "metadata": {"query_mode": param.mode},
        }

    def _build_global_config(self):
        tokenizer = _CharTokenizer()
        self.tokenizers.append(tokenizer)

        async def query_llm(query, *, system_prompt=None, stream=False, **kwargs):
            if system_prompt == BILINGUAL_PREPROCESS_SYSTEM_PROMPT:
                return json.dumps(
                    {
                        "query_zh": query,
                        "query_en": "translated memory query",
                        "hl_keywords_zh": ["记忆"],
                        "ll_keywords_zh": ["项目"],
                        "hl_keywords_en": ["memory"],
                        "ll_keywords_en": ["project"],
                    },
                    ensure_ascii=False,
                )
            self.events.append("final_llm")
            self.final_llm_calls.append(
                {
                    "query": query,
                    "system_prompt": system_prompt,
                    "stream": stream,
                    **kwargs,
                }
            )
            answer = f"synthesized: {query}"
            if stream:

                async def chunks():
                    yield "synthesized: "
                    yield query

                return chunks()
            return answer

        return {
            "role_llm_funcs": {"query": query_llm},
            "llm_cache_identities": {
                "query": {
                    "role": "query",
                    "binding": "openai",
                    "model": "query-model",
                    "host": self.final_endpoint,
                }
            },
            "tokenizer": tokenizer,
            "max_total_tokens": 100_000,
            "min_rerank_score": 0.0,
            "rerank_model_func": None,
        }


class BuilderProbe:
    def __init__(self, events: list[str] | None = None):
        self.events = events if events is not None else []
        self.instances: dict[str, FakeRAG] = {}
        self.empty_data = False
        self.final_endpoint = "https://memory.example/v1"
        self.policy_error: SensitiveContextPolicyError | None = None

    async def build(self, record) -> FakeRAG:
        rag = FakeRAG(
            record.workspace,
            events=self.events,
            empty_data=self.empty_data,
            final_endpoint=self.final_endpoint,
            policy_error=self.policy_error,
        )
        self.instances[record.id] = rag
        return rag

    async def finalize(self, rag) -> None:
        return None


class _RecordingHandle(AuthorizedChatMemoryHandle):
    def __init__(self, *args, events: list[str], **kwargs):
        super().__init__(*args, **kwargs)
        self.events = events
        self.resolve_calls = 0
        self.bound_endpoints: list[str | None] = []
        self.resolver_tokenizer: Any = None
        self.resolver_max_total_tokens: int | None = None

    def bind_final_llm_endpoint(self, endpoint: str | None) -> None:
        self.events.append("bind_endpoint")
        self.bound_endpoints.append(endpoint)
        super().bind_final_llm_endpoint(endpoint)

    async def resolve_for_final_request(
        self,
        tokenizer,
        max_total_tokens,
        build_final_request,
        policy_suffix="",
    ):
        self.events.append("resolve_context")
        self.resolve_calls += 1
        self.resolver_tokenizer = tokenizer
        self.resolver_max_total_tokens = max_total_tokens
        return await super().resolve_for_final_request(
            tokenizer,
            max_total_tokens,
            build_final_request,
            policy_suffix=policy_suffix,
        )


class FakeMemoryService:
    def __init__(self, events: list[str] | None = None):
        self.events = events if events is not None else []
        self.config = ChatMemoryConfig(
            enabled=True,
            llm_base_url="https://memory.example/v1",
            prompt_max_tokens=100_000,
            prompt_max_chars=100_000,
        )
        self.facts = [
            {
                "uuid": "edge-1",
                "fact": "采用 NR/BR 并用",
                "valid_at": "2026-07-01T00:00:00Z",
            }
        ]
        self.raise_unavailable = False
        self.calls: list[dict] = []
        self.handles: list[_RecordingHandle] = []

    def create_authorized_handle(self, **kwargs):
        self.events.append("handle_created")
        handle = _RecordingHandle(self, events=self.events, **kwargs)
        self.handles.append(handle)
        return handle

    async def search(self, **kwargs):
        self.events.append("memory_search")
        self.calls.append(dict(kwargs))
        if self.raise_unavailable:
            raise ChatMemoryUnavailableError("private backend detail")
        return list(self.facts)


class _CharTokenizer:
    def __init__(self):
        self.encoded: list[str] = []

    def encode(self, content: str) -> list[int]:
        self.encoded.append(content)
        return [ord(character) for character in content]


def _enterprise_args(**overrides):
    values = {
        "enterprise_auth_enabled": True,
        "enterprise_legacy_api_key_superadmin": False,
        "enterprise_disable_global_routes": True,
        "enterprise_rate_limit_enabled": False,
        "enterprise_rate_limit_requests": 60,
        "enterprise_rate_limit_window_seconds": 60.0,
        "enterprise_tenant_rate_limit_requests": 0,
        "enterprise_tenant_rate_limit_window_seconds": 60.0,
        "enterprise_quota_requests": 0,
        "enterprise_quota_window_seconds": 86400.0,
        "enterprise_tenant_quota_requests": 0,
        "enterprise_tenant_quota_window_seconds": 86400.0,
        "enterprise_mask_storage_uris": True,
        "chat_session_default_context_rounds": 1,
        "token_auto_renew": False,
        "token_renew_threshold": 0.5,
        "bilingual_query_enabled": False,
        "bilingual_query_default_mode": "off",
        "top_k": 40,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _patch_enterprise_args(monkeypatch, args) -> None:
    from lightrag.api import config as api_config
    import lightrag.api.utils_api as utils_api

    monkeypatch.setattr(api_config, "global_args", args)
    monkeypatch.setattr(utils_api, "global_args", args)
    dependency_functions = [get_combined_auth_dependency]
    for factory in (create_chat_routes, create_kb_query_routes):
        dependency = factory.__globals__["get_combined_auth_dependency"]
        if dependency not in dependency_functions:
            dependency_functions.append(dependency)
    for dependency in dependency_functions:
        monkeypatch.setitem(dependency.__globals__, "global_args", args)


def _token(user_service, user) -> str:
    return auth_handler.create_token(
        username=user.username,
        role=user.system_role,
        metadata=user_service.token_metadata_for_user(user),
    )


def _build(monkeypatch, tmp_path: Path, *, with_memory=True):
    args = _enterprise_args()
    _patch_enterprise_args(monkeypatch, args)

    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kb.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    from lightrag.api.document_lifecycle_service import DocumentLifecycleService

    document_service = DocumentLifecycleService(
        kb_service, metadata_store, tmp_path / "inputs"
    )
    audit_service = AuditService(metadata_store)
    user_service = UserService(metadata_store, audit_service)
    settings_service = SystemSettingsService(metadata_store)
    chat_service = ChatConversationService(metadata_store, audit_service)
    authz_service = AuthorizationService(
        metadata_store, audit_service, kb_service=kb_service
    )
    events: list[str] = []
    probe = BuilderProbe(events)
    registry = LightRAGInstanceRegistry(kb_service, probe.build, probe.finalize)
    memory_service = FakeMemoryService(events) if with_memory else None

    async def seed():
        await kb_service.initialize()
        await metadata_store.initialize()
        await settings_service.initialize_registration_setting(False)
        admin = await user_service.bootstrap_super_admin(
            username="admin", password="admin-pass", password_hash=None
        )
        alice = await user_service.create_user(
            username="alice", password="alice-pass", can_create_kb=True
        )
        bob = await user_service.create_user(username="bob", password="bob-pass")
        kb = await kb_service.create(name="KB", kb_id="kb_mem")
        kb_two = await kb_service.create(name="KB 2", kb_id="kb_mem_2")
        await authz_service.grant_kb_role(
            kb.id, alice.id, "kb_viewer", granted_by=admin.id
        )
        await authz_service.grant_kb_role(
            kb_two.id, alice.id, "kb_viewer", granted_by=admin.id
        )
        return alice, bob

    alice, bob = asyncio.run(seed())

    app = FastAPI()
    app.state.enterprise_enabled = True
    app.state.metadata_store = metadata_store
    app.state.enterprise_user_service = user_service
    app.state.enterprise_settings_service = settings_service
    app.state.enterprise_chat_conversation_service = chat_service
    app.state.enterprise_authorization_service = authz_service
    app.state.enterprise_audit_service = audit_service
    if memory_service is not None:
        app.state.enterprise_chat_memory_service = memory_service
    app.include_router(create_chat_routes(api_key=_API_KEY))
    app.include_router(
        create_kb_query_routes(document_service, registry, api_key=_API_KEY)
    )
    return TestClient(app), user_service, alice, bob, probe, memory_service


def _project(client, headers) -> str:
    resp = client.post("/chat/projects", json={"name": "记忆项目"}, headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


def test_kb_query_resolves_memory_through_sensitive_context(monkeypatch, tmp_path):
    client, user_service, alice, _bob, probe, memory = _build(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    project_id = _project(client, headers)

    resp = client.post(
        "/kbs/kb_mem/query",
        json={
            "query": "低温性能怎么做？",
            "mode": "mix",
            "user_prompt": "请用中文回答",
            "memory": {"project_id": project_id, "limit": 5},
        },
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Response metadata advertises the late, budgeted injection.
    assert body["metadata"]["memory"]["enabled"] is True
    assert body["metadata"]["memory"]["fact_count"] == 1
    # The memory service was called with the owning user + project + query.
    assert memory.calls == [
        {
            "user_id": alice.id,
            "project_id": project_id,
            "query": "低温性能怎么做？",
            "limit": 5,
        }
    ]
    rag = probe.instances["kb_mem"]
    param = rag.query_params[-1]
    # Raw facts never enter QueryParam/user_prompt.
    assert param.user_prompt == "请用中文回答"
    assert rag.sensitive_contexts[-1] is not None
    payload = rag.memory_payloads[-1]
    assert payload is not None
    assert "采用 NR/BR 并用" in payload.context_data
    assert "current authoritative KB evidence" in payload.trusted_policy


def test_kb_query_without_memory_field_is_untouched(monkeypatch, tmp_path):
    client, user_service, alice, _bob, probe, memory = _build(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    resp = client.post(
        "/kbs/kb_mem/query",
        json={"query": "无记忆注入", "mode": "mix"},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    assert "memory" not in resp.json()["metadata"]
    assert memory.calls == []
    rag = probe.instances["kb_mem"]
    param = rag.query_params[-1]
    assert param.user_prompt in (None, "", "n/a")
    assert rag.sensitive_contexts[-1] is None


def test_kb_query_memory_foreign_project_is_404(monkeypatch, tmp_path):
    client, user_service, alice, _bob, _probe, _memory = _build(monkeypatch, tmp_path)
    alice_headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    _project(client, alice_headers)
    # A project id alice does not own is an ownership 404 (no existence leak),
    # not a memory-search error.
    resp = client.post(
        "/kbs/kb_mem/query",
        json={"query": "abc", "mode": "mix", "memory": {"project_id": "proj_ghost"}},
        headers=alice_headers,
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Chat project not found"


def test_kb_query_memory_disabled_returns_503(monkeypatch, tmp_path):
    client, user_service, alice, _bob, _probe, _memory = _build(
        monkeypatch, tmp_path, with_memory=False
    )
    headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    project_id = _project(client, headers)
    resp = client.post(
        "/kbs/kb_mem/query",
        json={"query": "abc", "mode": "mix", "memory": {"project_id": project_id}},
        headers=headers,
    )
    assert resp.status_code == 503
    assert resp.json()["detail"] == "Chat memory is not enabled"


def test_kb_query_memory_unavailable_fails_open(monkeypatch, tmp_path):
    client, user_service, alice, _bob, probe, memory = _build(monkeypatch, tmp_path)
    memory.raise_unavailable = True
    headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    project_id = _project(client, headers)
    resp = client.post(
        "/kbs/kb_mem/query",
        json={
            "query": "后端挂了也要能答",
            "mode": "mix",
            "memory": {"project_id": project_id},
        },
        headers=headers,
    )
    # Fail-open: the query still succeeds, memory just reports unavailable.
    assert resp.status_code == 200, resp.text
    memory_info = resp.json()["metadata"]["memory"]
    assert memory_info["enabled"] is False
    assert memory_info["status"] == "unavailable"
    assert memory_info["reason"] == "unavailable"
    param = probe.instances["kb_mem"].query_params[-1]
    assert param.user_prompt in (None, "", "n/a")


def _memory_headers_and_project(client, user_service, user):
    headers = {"Authorization": f"Bearer {_token(user_service, user)}"}
    return headers, _project(client, headers)


def _assert_memory_error(response, status_code: int, error_code: str) -> None:
    assert response.status_code == status_code, response.text
    assert response.json()["detail"] == {
        "error_code": error_code,
        "message": error_code,
    }


def test_kb_query_stream_resolves_once_and_emits_memory_head(monkeypatch, tmp_path):
    client, user_service, alice, _bob, probe, memory = _build(monkeypatch, tmp_path)
    headers, project_id = _memory_headers_and_project(client, user_service, alice)

    response = client.post(
        "/kbs/kb_mem/query/stream",
        json={
            "query": "stream memory answer",
            "stream": True,
            "memory": {"project_id": project_id},
        },
        headers=headers,
    )

    assert response.status_code == 200, response.text
    lines = [json.loads(line) for line in response.text.splitlines() if line]
    head = lines[0]
    assert head["metadata"]["memory"]["status"] == "injected"
    assert [ref["reference_id"] for ref in head["references"]] == ["1"]
    assert all(not ref["reference_id"].startswith("M") for ref in head["references"])
    assert "".join(line.get("response", "") for line in lines[1:]) == (
        "answer: stream memory answer"
    )
    assert len(memory.calls) == 1
    assert len(memory.handles) == 1
    assert memory.handles[0].resolve_calls == 1
    assert memory.events.index("handle_created") < next(
        index
        for index, event in enumerate(memory.events)
        if event.startswith("single_retrieval:")
    )
    assert memory.events.index("resolve_context") < memory.events.index(
        "memory_search"
    )
    assert memory.events.index("memory_search") < memory.events.index("final_llm")
    assert probe.instances["kb_mem"].sensitive_contexts[-1] is memory.handles[0]


def test_bilingual_kb_query_uses_same_handle_and_separate_references(
    monkeypatch, tmp_path
):
    client, user_service, alice, _bob, probe, memory = _build(monkeypatch, tmp_path)
    from lightrag.api import config as api_config

    api_config.global_args.bilingual_query_enabled = True
    api_config.global_args.bilingual_query_default_mode = "on"
    headers, project_id = _memory_headers_and_project(client, user_service, alice)

    response = client.post(
        "/kbs/kb_mem/query",
        json={
            "query": "双语记忆查询",
            "bilingual": True,
            "memory": {"project_id": project_id},
        },
        headers=headers,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["metadata"]["bilingual"]["enabled"] is True
    assert body["metadata"]["memory"]["status"] == "injected"
    assert len(memory.calls) == 1
    assert memory.handles[0].resolve_calls == 1
    rag = probe.instances["kb_mem"]
    assert len([event for event in memory.events if event.startswith("retrieve:")]) == 2
    assert max(
        index
        for index, event in enumerate(memory.events)
        if event.startswith("retrieve:")
    ) < memory.events.index("resolve_context")
    assert rag.final_llm_calls[-1]["_sensitive"] is True
    prompt = str(rag.final_llm_calls[-1]["system_prompt"])
    instructions, context = prompt.split("---Context---", maxsplit=1)
    assert instructions.rstrip().endswith(
        "Keep the generated ### References section and all top-level reference arrays KB-only."
    )
    assert '"reference_id":"M1"' in context
    assert all(not ref["reference_id"].startswith("M") for ref in body["references"])


def test_bilingual_kb_stream_uses_same_handle_and_memory_head(monkeypatch, tmp_path):
    client, user_service, alice, _bob, probe, memory = _build(monkeypatch, tmp_path)
    from lightrag.api import config as api_config

    api_config.global_args.bilingual_query_enabled = True
    api_config.global_args.bilingual_query_default_mode = "on"
    headers, project_id = _memory_headers_and_project(client, user_service, alice)

    response = client.post(
        "/kbs/kb_mem/query/stream",
        json={
            "query": "双语流式记忆查询",
            "bilingual": True,
            "stream": True,
            "memory": {"project_id": project_id},
        },
        headers=headers,
    )

    assert response.status_code == 200, response.text
    head = json.loads(response.text.splitlines()[0])
    assert head["metadata"]["bilingual"]["enabled"] is True
    assert head["metadata"]["memory"]["status"] == "injected"
    assert memory.handles[0].resolve_calls == 1
    assert len(memory.calls) == 1
    final_call = probe.instances["kb_mem"].final_llm_calls[-1]
    assert final_call["_sensitive"] is True
    assert final_call["stream"] is True


def test_multi_kb_memory_resolves_after_merge_with_sensitive_final_call(
    monkeypatch, tmp_path
):
    client, user_service, alice, _bob, probe, memory = _build(monkeypatch, tmp_path)
    headers, project_id = _memory_headers_and_project(client, user_service, alice)

    response = client.post(
        "/kbs:query",
        json={
            "kb_ids": ["kb_mem", "kb_mem_2"],
            "query": "compare both knowledge bases",
            "conversation_history": [
                {"role": "user", "content": "history sentinel"}
            ],
            "max_total_tokens": 54_321,
            "memory": {"project_id": project_id},
        },
        headers=headers,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["metadata"]["memory"]["status"] == "injected"
    assert len(memory.calls) == 1
    assert memory.handles[0].resolve_calls == 1
    assert memory.handles[0].bound_endpoints == ["https://memory.example/v1"]
    assert memory.handles[0].resolver_max_total_tokens == 54_321
    assert memory.handles[0].resolver_tokenizer is probe.instances[
        "kb_mem"
    ].tokenizers[-1]
    complete_requests = [
        content
        for content in memory.handles[0].resolver_tokenizer.encoded
        if content.startswith("---LIGHTRAG FINAL SYSTEM PROMPT---")
    ]
    assert complete_requests
    assert all("history sentinel" in content for content in complete_requests)
    retrieval_indexes = [
        index
        for index, event in enumerate(memory.events)
        if event.startswith("retrieve:")
    ]
    assert len(retrieval_indexes) == 2
    assert memory.events.index("handle_created") < min(retrieval_indexes)
    assert max(retrieval_indexes) < memory.events.index("bind_endpoint")
    assert memory.events.index("resolve_context") < memory.events.index(
        "memory_search"
    )
    assert memory.events.index("memory_search") < memory.events.index("final_llm")

    synth_rag = probe.instances["kb_mem"]
    final_call = synth_rag.final_llm_calls[-1]
    assert final_call["_sensitive"] is True
    assert final_call["history_messages"] == [
        {"role": "user", "content": "history sentinel"}
    ]
    prompt = str(final_call["system_prompt"])
    instructions, context = prompt.split("---Context---", maxsplit=1)
    assert instructions.rstrip().endswith(
        "Keep the generated ### References section and all top-level reference arrays KB-only."
    )
    assert '"reference_id":"M1"' in context
    reference_ids = [ref["reference_id"] for ref in body["references"]]
    assert reference_ids == ["1", "2"]
    assert all(not reference_id.startswith("M") for reference_id in reference_ids)


def test_multi_kb_stream_memory_metadata_and_sensitive_setup(monkeypatch, tmp_path):
    client, user_service, alice, _bob, probe, memory = _build(monkeypatch, tmp_path)
    headers, project_id = _memory_headers_and_project(client, user_service, alice)

    response = client.post(
        "/kbs:query/stream",
        json={
            "kb_ids": ["kb_mem", "kb_mem_2"],
            "query": "stream both knowledge bases",
            "memory": {"project_id": project_id},
        },
        headers=headers,
    )

    assert response.status_code == 200, response.text
    lines = [json.loads(line) for line in response.text.splitlines() if line]
    assert lines[0]["metadata"]["memory"]["status"] == "injected"
    assert [ref["reference_id"] for ref in lines[0]["references"]] == ["1", "2"]
    assert len(memory.calls) == 1
    assert memory.handles[0].resolve_calls == 1
    synth_rag = probe.instances["kb_mem"]
    assert synth_rag.final_llm_calls[-1]["_sensitive"] is True
    assert synth_rag.final_llm_calls[-1]["stream"] is True


@pytest.mark.parametrize(
    "case,expected_status,expected_searches",
    [
        ("empty", "empty", 1),
        ("unavailable", "unavailable", 1),
        ("budget", "budget_exhausted", 0),
    ],
)
def test_multi_kb_noninjected_outcomes_still_use_sensitive_final_call(
    monkeypatch,
    tmp_path,
    case,
    expected_status,
    expected_searches,
):
    client, user_service, alice, _bob, probe, memory = _build(monkeypatch, tmp_path)
    if case == "empty":
        memory.facts = []
    elif case == "unavailable":
        memory.raise_unavailable = True
    else:
        memory.config = replace(memory.config, prompt_max_chars=1)
    headers, project_id = _memory_headers_and_project(client, user_service, alice)

    response = client.post(
        "/kbs:query",
        json={
            "kb_ids": ["kb_mem", "kb_mem_2"],
            "query": f"multi {case} memory outcome",
            "memory": {"project_id": project_id},
        },
        headers=headers,
    )

    assert response.status_code == 200, response.text
    info = response.json()["metadata"]["memory"]
    assert info["status"] == expected_status
    assert len(memory.calls) == expected_searches
    final_call = probe.instances["kb_mem"].final_llm_calls[-1]
    assert final_call["_sensitive"] is True
    assert "---Untrusted Project Memory Data---" not in str(
        final_call["system_prompt"]
    )


@pytest.mark.parametrize("route", ["single", "multi"])
def test_no_kb_evidence_marks_not_used_without_search_or_final_llm(
    monkeypatch, tmp_path, route
):
    client, user_service, alice, _bob, probe, memory = _build(monkeypatch, tmp_path)
    probe.empty_data = True
    headers, project_id = _memory_headers_and_project(client, user_service, alice)
    if route == "single":
        path = "/kbs/kb_mem/query"
        payload = {
            "query": "no evidence single",
            "memory": {"project_id": project_id},
        }
    else:
        path = "/kbs:query"
        payload = {
            "kb_ids": ["kb_mem", "kb_mem_2"],
            "query": "no evidence multi",
            "memory": {"project_id": project_id},
        }

    response = client.post(path, json=payload, headers=headers)

    assert response.status_code == 200, response.text
    info = response.json()["metadata"]["memory"]
    assert info["enabled"] is True
    assert info["status"] == "not_used"
    assert info["reason"] == "no_kb_evidence"
    assert memory.calls == []
    assert memory.handles[0].resolve_calls == 0
    assert "memory_search" not in memory.events
    assert "final_llm" not in memory.events


def test_stream_memory_backend_unavailable_fails_open(monkeypatch, tmp_path):
    client, user_service, alice, _bob, _probe, memory = _build(monkeypatch, tmp_path)
    memory.raise_unavailable = True
    headers, project_id = _memory_headers_and_project(client, user_service, alice)

    response = client.post(
        "/kbs/kb_mem/query/stream",
        json={
            "query": "stream despite backend failure",
            "memory": {"project_id": project_id},
        },
        headers=headers,
    )

    assert response.status_code == 200, response.text
    head = json.loads(response.text.splitlines()[0])
    info = head["metadata"]["memory"]
    assert info["enabled"] is False
    assert info["status"] == "unavailable"
    assert info["reason"] == "unavailable"
    assert len(memory.calls) == 1


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/kbs/kb_mem/query", {"query": "egress single"}),
        ("/kbs/kb_mem/query/stream", {"query": "egress stream"}),
        (
            "/kbs:query",
            {"kb_ids": ["kb_mem", "kb_mem_2"], "query": "egress multi"},
        ),
    ],
)
def test_egress_denial_is_hard_403_before_memory_search(
    monkeypatch, tmp_path, path, payload
):
    client, user_service, alice, _bob, probe, memory = _build(monkeypatch, tmp_path)
    probe.final_endpoint = "https://different-query-provider.example/v1"
    headers, project_id = _memory_headers_and_project(client, user_service, alice)
    payload = {**payload, "memory": {"project_id": project_id}}

    response = client.post(path, json=payload, headers=headers)

    _assert_memory_error(
        response,
        403,
        "chat_memory_query_llm_egress_not_allowed",
    )
    assert memory.calls == []
    assert "memory_search" not in memory.events
    assert "final_llm" not in memory.events
    assert "different-query-provider" not in response.text


def test_memory_query_length_rejected_before_authorization_or_search(
    monkeypatch, tmp_path
):
    client, user_service, alice, _bob, _probe, memory = _build(monkeypatch, tmp_path)
    headers, project_id = _memory_headers_and_project(client, user_service, alice)

    response = client.post(
        "/kbs/kb_mem/query",
        json={
            "query": "q" * 4097,
            "memory": {"project_id": project_id},
        },
        headers=headers,
    )

    _assert_memory_error(response, 400, "chat_memory_query_too_long")
    assert memory.handles == []
    assert memory.calls == []
    assert "memory_search" not in memory.events


@pytest.mark.parametrize(
    "path,extra",
    [
        ("/kbs/kb_mem/query", {"mode": "bypass"}),
        ("/kbs/kb_mem/query", {"only_need_context": True}),
        ("/kbs/kb_mem/query", {"only_need_prompt": True}),
        (
            "/kbs/kb_mem/query",
            {"only_need_context": True, "only_need_prompt": True},
        ),
        ("/kbs/kb_mem/query/stream", {"mode": "bypass"}),
        ("/kbs/kb_mem/query/stream", {"only_need_context": True}),
        ("/kbs/kb_mem/query/stream", {"only_need_prompt": True}),
        (
            "/kbs/kb_mem/query/stream",
            {"only_need_context": True, "only_need_prompt": True},
        ),
        ("/kbs/kb_mem/query/data", {}),
        ("/kbs/kb_mem/retrieve", {}),
        ("/kbs:retrieve", {"kb_ids": ["kb_mem", "kb_mem_2"]}),
        ("/kbs:query", {"kb_ids": ["kb_mem"], "mode": "bypass"}),
    ],
)
def test_memory_rejected_for_every_no_final_synthesis_path(
    monkeypatch, tmp_path, path, extra
):
    client, user_service, alice, _bob, _probe, memory = _build(monkeypatch, tmp_path)
    headers, project_id = _memory_headers_and_project(client, user_service, alice)
    payload = {
        "query": "must require final synthesis",
        "memory": {"project_id": project_id},
        **extra,
    }

    response = client.post(path, json=payload, headers=headers)

    _assert_memory_error(
        response,
        400,
        "chat_memory_requires_final_synthesis",
    )
    assert memory.handles == []
    assert memory.calls == []
    assert "memory_search" not in memory.events
    assert not any(
        event.startswith(("single_retrieval:", "retrieve:"))
        for event in memory.events
    )


def test_builder_contract_error_is_content_free_500(monkeypatch, tmp_path):
    client, user_service, alice, _bob, probe, memory = _build(monkeypatch, tmp_path)
    sentinel = "PRIVATE-BUILDER-DETAIL"
    probe.policy_error = SensitiveContextPolicyError(
        CHAT_MEMORY_FINAL_REQUEST_BUILDER_INVALID,
        sentinel,
    )
    headers, project_id = _memory_headers_and_project(client, user_service, alice)

    response = client.post(
        "/kbs/kb_mem/query",
        json={
            "query": "builder contract",
            "memory": {"project_id": project_id},
        },
        headers=headers,
    )

    _assert_memory_error(
        response,
        500,
        CHAT_MEMORY_FINAL_REQUEST_BUILDER_INVALID,
    )
    assert sentinel not in response.text
    assert memory.calls == []


def test_all_kb_memory_audits_use_content_free_projection(monkeypatch, tmp_path):
    audits: list[dict[str, Any]] = []

    async def record_audit(
        _request,
        event_type,
        *,
        target_type,
        target_id,
        metadata,
    ):
        audits.append(
            {
                "event_type": event_type,
                "target_type": target_type,
                "target_id": target_id,
                "metadata": metadata,
            }
        )

    monkeypatch.setitem(
        create_kb_query_routes.__globals__,
        "append_enterprise_audit_event",
        record_audit,
    )
    client, user_service, alice, _bob, _probe, memory = _build(monkeypatch, tmp_path)
    headers, project_id = _memory_headers_and_project(client, user_service, alice)
    base_memory = {"memory": {"project_id": project_id}}

    responses = [
        client.post(
            "/kbs/kb_mem/query",
            json={"query": "audit single", **base_memory},
            headers=headers,
        ),
        client.post(
            "/kbs/kb_mem/query/stream",
            json={"query": "audit stream", **base_memory},
            headers=headers,
        ),
        client.post(
            "/kbs:query",
            json={
                "kb_ids": ["kb_mem", "kb_mem_2"],
                "query": "audit multi",
                **base_memory,
            },
            headers=headers,
        ),
    ]
    from lightrag.api import config as api_config

    api_config.global_args.bilingual_query_enabled = True
    responses.append(
        client.post(
            "/kbs/kb_mem/query",
            json={
                "query": "审计双语",
                "bilingual": True,
                **base_memory,
            },
            headers=headers,
        )
    )

    assert all(response.status_code == 200 for response in responses)
    expected_events = {
        "query_executed",
        "query_stream_started",
        "multi_kb_query_executed",
    }
    assert expected_events.issubset({audit["event_type"] for audit in audits})
    for audit in audits:
        metadata = audit["metadata"]
        assert metadata["memory_enabled"] is True
        assert metadata["memory_status"] == "injected"
        assert metadata["memory_fact_count"] == 1
        assert metadata["memory_injected_count"] == 1
        serialized = json.dumps(metadata, ensure_ascii=False)
        assert "edge-1" not in serialized
        assert "采用 NR/BR 并用" not in serialized
        assert project_id not in serialized
        assert "references" not in metadata
    assert len(memory.calls) == 4


def test_no_memory_json_and_ndjson_bytes_remain_compatible(monkeypatch, tmp_path):
    client, user_service, alice, _bob, probe, memory = _build(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}

    response = client.post(
        "/kbs/kb_mem/query",
        json={"query": "no memory", "mode": "mix"},
        headers=headers,
    )
    assert response.content == (
        b'{"kb_id":"kb_mem","mode":"mix","response":"answer: no memory",'
        b'"references":[{"reference_id":"1","file_path":"single.pdf",'
        b'"content":null}],"metadata":{}}'
    )

    stream_response = client.post(
        "/kbs/kb_mem/query/stream",
        json={"query": "stream no memory", "stream": True},
        headers=headers,
    )
    assert stream_response.content == (
        b'{"kb_id": "kb_mem", "metadata": {}, "references": '
        b'[{"reference_id": "1", "file_path": "single.pdf"}]}\n'
        b'{"response": "answer: "}\n'
        b'{"response": "stream no memory"}\n'
    )
    assert memory.calls == []
    assert all(
        context is None
        for context in probe.instances["kb_mem"].sensitive_contexts
    )


def test_multi_kb_no_memory_shapes_do_not_add_sensitive_fields(
    monkeypatch, tmp_path
):
    client, user_service, alice, _bob, probe, memory = _build(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {_token(user_service, alice)}"}
    payload = {
        "kb_ids": ["kb_mem", "kb_mem_2"],
        "query": "multi no memory",
    }

    response = client.post("/kbs:query", json=payload, headers=headers)
    assert response.status_code == 200, response.text
    assert "memory" not in response.json()["metadata"]
    assert "_sensitive" not in probe.instances["kb_mem"].final_llm_calls[-1]

    stream_response = client.post(
        "/kbs:query/stream",
        json={**payload, "query": "multi stream no memory"},
        headers=headers,
    )
    assert stream_response.status_code == 200, stream_response.text
    head = json.loads(stream_response.text.splitlines()[0])
    assert "memory" not in head["metadata"]
    assert "_sensitive" not in probe.instances["kb_mem"].final_llm_calls[-1]
    assert memory.calls == []
