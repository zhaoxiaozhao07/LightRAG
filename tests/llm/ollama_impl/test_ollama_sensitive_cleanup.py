"""Sensitive-scope cleanup regressions for the Ollama provider."""

from __future__ import annotations

import importlib
import logging
import sys
import traceback
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from lightrag.llm_roles import _RoleLLMMixin
from lightrag.sensitive_context import SensitiveLLMError, is_sensitive_call
from lightrag.utils import logger


pytestmark = pytest.mark.offline


def _load_ollama_module(monkeypatch, request, client_factory):
    fake_pm = SimpleNamespace(
        is_installed=lambda _name: True,
        install=lambda _name: None,
    )
    fake_ollama = ModuleType("ollama")
    setattr(fake_ollama, "AsyncClient", client_factory)

    monkeypatch.setitem(sys.modules, "pipmaster", fake_pm)
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

    parent = sys.modules.get("lightrag.llm")
    original_module = sys.modules.get("lightrag.llm.ollama")
    original_parent_attr = getattr(parent, "ollama", None) if parent else None
    sys.modules.pop("lightrag.llm.ollama", None)
    if parent is not None and hasattr(parent, "ollama"):
        delattr(parent, "ollama")

    def restore_module():
        if original_module is not None:
            sys.modules["lightrag.llm.ollama"] = original_module
        else:
            sys.modules.pop("lightrag.llm.ollama", None)
        if parent is not None:
            if original_parent_attr is not None:
                setattr(parent, "ollama", original_parent_attr)
            elif hasattr(parent, "ollama"):
                delattr(parent, "ollama")

    request.addfinalizer(restore_module)
    return importlib.import_module("lightrag.llm.ollama")


def _role_wrapper(raw_func) -> Any:
    mixin = _RoleLLMMixin()
    setattr(mixin, "llm_response_cache", None)
    return mixin._wrap_llm_role_func(
        "query",
        raw_func,
        max_async=1,
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


@pytest.mark.asyncio
async def test_sensitive_non_streaming_cleanup_failure_is_content_free(
    monkeypatch, request, caplog
):
    provider_sentinel = "OLLAMA-NONSTREAM-PROVIDER-SENTINEL"
    cleanup_sentinel = "OLLAMA-NONSTREAM-CLEANUP-SENTINEL"

    class NonStreamingCleanupError(RuntimeError):
        pass

    class FakeTransport:
        async def aclose(self):
            raise NonStreamingCleanupError(cleanup_sentinel)

    class FakeClient:
        def __init__(self, **_kwargs):
            self._client = FakeTransport()

        async def chat(self, **_kwargs):
            raise RuntimeError(provider_sentinel)

    ollama_module = _load_ollama_module(monkeypatch, request, FakeClient)
    monkeypatch.setattr(logger, "propagate", True)

    async def raw(prompt, **kwargs):
        return await ollama_module._ollama_model_if_cache(
            "test-model", prompt, **kwargs
        )

    wrapped = _role_wrapper(raw)
    try:
        with caplog.at_level(logging.DEBUG, logger="lightrag"):
            with pytest.raises(SensitiveLLMError) as exc_info:
                await wrapped("private prompt", _sensitive=True)
    finally:
        await wrapped.shutdown()

    messages = "\n".join(record.getMessage() for record in caplog.records)
    exception_text = _exception_text_and_chain(exc_info.value)
    assert cleanup_sentinel not in messages
    assert provider_sentinel not in messages
    assert cleanup_sentinel not in exception_text
    assert provider_sentinel not in exception_text
    assert "NonStreamingCleanupError" in messages
    assert is_sensitive_call() is False


@pytest.mark.asyncio
async def test_sensitive_streaming_cleanup_failure_is_content_free(
    monkeypatch, request, caplog
):
    provider_sentinel = "OLLAMA-STREAM-PROVIDER-SENTINEL"
    cleanup_sentinel = "OLLAMA-STREAM-CLEANUP-SENTINEL"

    class StreamingCleanupError(RuntimeError):
        pass

    class FakeTransport:
        async def aclose(self):
            raise StreamingCleanupError(cleanup_sentinel)

    class FakeStream:
        def __init__(self):
            self._yielded = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._yielded:
                self._yielded = True
                return {"message": {"content": "first"}}
            raise RuntimeError(provider_sentinel)

    class FakeClient:
        def __init__(self, **_kwargs):
            self._client = FakeTransport()

        async def chat(self, **_kwargs):
            return FakeStream()

    ollama_module = _load_ollama_module(monkeypatch, request, FakeClient)
    monkeypatch.setattr(logger, "propagate", True)

    async def raw(prompt, **kwargs):
        return await ollama_module._ollama_model_if_cache(
            "test-model", prompt, **kwargs
        )

    wrapped = _role_wrapper(raw)
    try:
        iterator = await wrapped(
            "private prompt", stream=True, _sensitive=True
        )
        assert await anext(iterator) == "first"
        with caplog.at_level(logging.DEBUG, logger="lightrag"):
            with pytest.raises(SensitiveLLMError) as exc_info:
                await anext(iterator)
    finally:
        await wrapped.shutdown()

    messages = "\n".join(record.getMessage() for record in caplog.records)
    exception_text = _exception_text_and_chain(exc_info.value)
    assert cleanup_sentinel not in messages
    assert provider_sentinel not in messages
    assert cleanup_sentinel not in exception_text
    assert provider_sentinel not in exception_text
    assert "StreamingCleanupError" in messages
    assert is_sensitive_call() is False
