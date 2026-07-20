"""Sensitive final-LLM scope, logging, tracing, and error isolation tests."""

from __future__ import annotations

import asyncio
import logging
import traceback
from types import SimpleNamespace

import pytest

from lightrag.base import QueryParam
from lightrag.api.chat_memory_service import (
    AuthorizedChatMemoryHandle,
    ChatMemoryConfig,
)
from lightrag.lightrag import LightRAG
from lightrag.llm_roles import _RoleLLMMixin
from lightrag.sensitive_context import (
    SensitiveContextPolicyError,
    SensitiveLLMError,
    is_sensitive_call,
    sensitive_call_scope,
)
from lightrag.utils import logger, set_verbose_debug, verbose_debug


pytestmark = pytest.mark.offline
_SENTINEL = "PRIVATE-MEMORY-SENTINEL-4A"


@pytest.fixture
def propagating_logger(monkeypatch):
    monkeypatch.setattr(logger, "propagate", True)


def _role_wrapper(raw_func, *, max_async=2):
    mixin = _RoleLLMMixin()
    mixin.llm_response_cache = None
    return mixin._wrap_llm_role_func(
        "query",
        raw_func,
        max_async=max_async,
        timeout=30,
        model_kwargs={},
    )


def _exception_text_and_chain(exc: BaseException) -> str:
    values = ["".join(traceback.format_exception(exc)), str(exc), repr(exc)]
    seen: set[int] = set()
    current = exc.__cause__ or exc.__context__
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        values.extend((str(current), repr(current)))
        current = current.__cause__ or current.__context__
    return "\n".join(values)


def test_verbose_debug_emits_no_content_in_sensitive_scope(
    caplog, propagating_logger
):
    set_verbose_debug(True)
    try:
        with caplog.at_level(logging.DEBUG, logger="lightrag"):
            with sensitive_call_scope():
                verbose_debug(f"prompt={_SENTINEL}")
            verbose_debug("ordinary-visible")
    finally:
        set_verbose_debug(False)

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert _SENTINEL not in messages
    assert "ordinary-visible" in messages


def test_sensitive_openai_uses_standard_uninstrumented_client(monkeypatch):
    import lightrag.llm.openai as openai_module

    created: list[str] = []

    class _InstrumentedClient:
        def __init__(self, **_kwargs):
            created.append("instrumented")

    class _StandardClient:
        def __init__(self, **_kwargs):
            created.append("standard")

    monkeypatch.setattr(openai_module, "AsyncOpenAI", _InstrumentedClient)
    monkeypatch.setattr(openai_module, "StandardAsyncOpenAI", _StandardClient)
    monkeypatch.setattr(openai_module, "LANGFUSE_ENABLED", True)

    openai_module.create_openai_async_client(
        api_key="test-key", base_url="https://example.invalid/v1"
    )
    with sensitive_call_scope():
        openai_module.create_openai_async_client(
            api_key="test-key", base_url="https://example.invalid/v1"
        )

    assert created == ["instrumented", "standard"]


@pytest.mark.asyncio
async def test_sensitive_openai_stream_failure_logs_no_prompt_or_provider_body(
    monkeypatch, caplog, propagating_logger
):
    import lightrag.llm.openai as openai_module

    class _FailingStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise RuntimeError(_SENTINEL)

        async def aclose(self):
            return None

    class _Completions:
        async def create(self, **_kwargs):
            return _FailingStream()

    class _Client:
        chat = SimpleNamespace(completions=_Completions())

        async def close(self):
            return None

    monkeypatch.setattr(
        openai_module,
        "create_openai_async_client",
        lambda **_kwargs: _Client(),
    )

    async def raw(prompt, **kwargs):
        return await openai_module.openai_complete_if_cache(
            "test-model",
            prompt,
            api_key="test-key",
            base_url="https://example.invalid/v1",
            **kwargs,
        )

    wrapped = _role_wrapper(raw, max_async=1)
    set_verbose_debug(True)
    try:
        with caplog.at_level(logging.DEBUG, logger="lightrag"):
            iterator = await wrapped(
                f"prompt {_SENTINEL}", stream=True, _sensitive=True
            )
            with pytest.raises(SensitiveLLMError) as exc_info:
                await anext(iterator)
    finally:
        set_verbose_debug(False)

    assert _SENTINEL not in _exception_text_and_chain(exc_info.value)
    assert _SENTINEL not in "\n".join(
        record.getMessage() for record in caplog.records
    )
    assert is_sensitive_call() is False
    await wrapped.shutdown()


@pytest.mark.asyncio
async def test_sensitive_and_ordinary_calls_are_context_isolated_in_role_queue():
    started = 0
    both_started = asyncio.Event()
    release = asyncio.Event()
    observed: dict[str, bool] = {}

    async def raw(label, **_kwargs):
        nonlocal started
        observed[label] = is_sensitive_call()
        started += 1
        if started == 2:
            both_started.set()
        await release.wait()
        return label

    wrapped = _role_wrapper(raw)
    sensitive = asyncio.create_task(wrapped("sensitive", _sensitive=True))
    ordinary = asyncio.create_task(wrapped("ordinary"))
    await both_started.wait()
    release.set()

    assert await asyncio.gather(sensitive, ordinary) == ["sensitive", "ordinary"]
    assert observed == {"sensitive": True, "ordinary": False}
    assert is_sensitive_call() is False
    await wrapped.shutdown()


@pytest.mark.asyncio
async def test_sensitive_initial_failure_is_content_free_and_has_no_raw_chain(
    caplog, propagating_logger
):
    async def raw(**_kwargs):
        assert is_sensitive_call() is True
        raise RuntimeError(_SENTINEL)

    wrapped = _role_wrapper(raw, max_async=1)
    with caplog.at_level(logging.DEBUG, logger="lightrag"):
        with pytest.raises(SensitiveLLMError) as exc_info:
            await wrapped(_sensitive=True)

    assert _SENTINEL not in _exception_text_and_chain(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert _SENTINEL not in "\n".join(
        record.getMessage() for record in caplog.records
    )
    assert is_sensitive_call() is False
    await wrapped.shutdown()


@pytest.mark.asyncio
async def test_sensitive_stream_failure_is_sanitized_and_scope_restored(
    caplog, propagating_logger
):
    observations: list[bool] = []

    async def raw(**_kwargs):
        assert is_sensitive_call() is True

        async def stream():
            observations.append(is_sensitive_call())
            yield "first"
            observations.append(is_sensitive_call())
            raise RuntimeError(_SENTINEL)

        return stream()

    wrapped = _role_wrapper(raw, max_async=1)
    iterator = await wrapped(_sensitive=True)
    assert is_sensitive_call() is False
    assert await anext(iterator) == "first"
    assert is_sensitive_call() is False
    with caplog.at_level(logging.DEBUG, logger="lightrag"):
        with pytest.raises(SensitiveLLMError) as exc_info:
            await anext(iterator)

    assert observations == [True, True]
    assert _SENTINEL not in _exception_text_and_chain(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert _SENTINEL not in "\n".join(
        record.getMessage() for record in caplog.records
    )
    assert is_sensitive_call() is False
    await wrapped.shutdown()


@pytest.mark.asyncio
async def test_sensitive_stream_cancellation_and_aclose_restore_context():
    started = asyncio.Event()
    cancellation_observations: list[bool] = []

    async def cancellable_raw(**_kwargs):
        async def stream():
            try:
                cancellation_observations.append(is_sensitive_call())
                started.set()
                await asyncio.Event().wait()
                yield "never"
            finally:
                cancellation_observations.append(is_sensitive_call())

        return stream()

    wrapped = _role_wrapper(cancellable_raw, max_async=1)
    iterator = await wrapped(_sensitive=True)
    pending = asyncio.create_task(anext(iterator))
    await started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert cancellation_observations == [True, True]
    assert is_sensitive_call() is False
    await wrapped.shutdown()

    close_observations: list[bool] = []

    async def closable_raw(**_kwargs):
        async def stream():
            try:
                close_observations.append(is_sensitive_call())
                yield "one"
            finally:
                close_observations.append(is_sensitive_call())

        return stream()

    wrapped = _role_wrapper(closable_raw, max_async=1)
    iterator = await wrapped(_sensitive=True)
    assert await anext(iterator) == "one"
    await iterator.aclose()
    assert close_observations == [True, True]
    assert is_sensitive_call() is False
    await wrapped.shutdown()


@pytest.mark.asyncio
async def test_aquery_llm_rethrows_typed_policy_error(monkeypatch):
    import lightrag.lightrag as lightrag_module

    policy_error = SensitiveContextPolicyError("stable_code", "stable_message")

    async def fail_policy(*_args, **_kwargs):
        raise policy_error

    monkeypatch.setattr(lightrag_module, "kg_query", fail_policy)
    fake_rag = SimpleNamespace(
        _build_global_config=lambda: {},
        chunk_entity_relation_graph=object(),
        entities_vdb=object(),
        relationships_vdb=object(),
        text_chunks=object(),
        llm_response_cache=None,
        chunks_vdb=object(),
    )

    with pytest.raises(SensitiveContextPolicyError) as exc_info:
        await LightRAG.aquery_llm(
            fake_rag,
            "query",
            QueryParam(mode="local"),
            sensitive_context=SimpleNamespace(),
        )
    assert exc_info.value is policy_error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "param",
    [
        QueryParam(mode="bypass"),
        QueryParam(mode="local", only_need_context=True),
        QueryParam(mode="local", only_need_prompt=True),
        QueryParam(
            mode="local", only_need_context=True, only_need_prompt=True
        ),
    ],
)
async def test_aquery_rejects_sensitive_context_without_final_synthesis(param):
    fake_rag = SimpleNamespace(
        _build_global_config=lambda: (_ for _ in ()).throw(
            AssertionError("must reject before building runtime config")
        )
    )
    with pytest.raises(SensitiveContextPolicyError) as exc_info:
        await LightRAG.aquery_llm(
            fake_rag,
            "query",
            param,
            sensitive_context=SimpleNamespace(),
        )
    assert exc_info.value.error_code == "chat_memory_requires_final_synthesis"


@pytest.mark.asyncio
async def test_both_unknown_egress_propagates_through_aquery_without_search_or_llm(
    monkeypatch,
):
    import lightrag.lightrag as lightrag_module

    class _Service:
        config = ChatMemoryConfig(enabled=True, llm_base_url=None)

        def __init__(self):
            self.search_calls = 0

        async def search(self, **_kwargs):
            self.search_calls += 1
            return []

    class _Tokenizer:
        def encode(self, content):
            return list(content)

    service = _Service()
    handle = AuthorizedChatMemoryHandle(
        service,
        user_id="usr",
        project_id="proj",
        query="query",
        query_llm_endpoint=None,
    )
    final_llm_calls = 0

    async def resolve_context(*_args, sensitive_context=None, **_kwargs):
        nonlocal final_llm_calls
        await sensitive_context.resolve_for_final_request(
            _Tokenizer(),
            100_000,
            lambda payload: payload.trusted_policy if payload else "base",
        )
        final_llm_calls += 1
        raise AssertionError("final LLM must not run")

    monkeypatch.setattr(lightrag_module, "kg_query", resolve_context)
    fake_rag = SimpleNamespace(
        _build_global_config=lambda: {},
        chunk_entity_relation_graph=object(),
        entities_vdb=object(),
        relationships_vdb=object(),
        text_chunks=object(),
        llm_response_cache=None,
        chunks_vdb=object(),
    )

    with pytest.raises(SensitiveContextPolicyError) as exc_info:
        await LightRAG.aquery_llm(
            fake_rag,
            "query",
            QueryParam(mode="local"),
            sensitive_context=handle,
        )
    assert exc_info.value.error_code == "chat_memory_query_llm_egress_not_allowed"
    assert service.search_calls == 0
    assert final_llm_calls == 0


@pytest.mark.asyncio
async def test_aquery_llm_generic_sensitive_failure_is_content_free(
    monkeypatch, caplog, propagating_logger
):
    import lightrag.lightrag as lightrag_module

    async def fail_generic(*_args, **_kwargs):
        raise RuntimeError(_SENTINEL)

    monkeypatch.setattr(lightrag_module, "kg_query", fail_generic)
    fake_rag = SimpleNamespace(
        _build_global_config=lambda: {},
        chunk_entity_relation_graph=object(),
        entities_vdb=object(),
        relationships_vdb=object(),
        text_chunks=object(),
        llm_response_cache=None,
        chunks_vdb=object(),
    )

    with caplog.at_level(logging.DEBUG, logger="lightrag"):
        result = await LightRAG.aquery_llm(
            fake_rag,
            "query",
            QueryParam(mode="local"),
            sensitive_context=SimpleNamespace(),
        )
    assert result["status"] == "failure"
    assert result["message"] == "Query failed"
    assert _SENTINEL not in str(result)
    assert _SENTINEL not in "\n".join(
        record.getMessage() for record in caplog.records
    )
