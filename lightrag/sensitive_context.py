"""Private sensitive-context contracts used by final LLM synthesis.

Sensitive context is process-local input that must never be persisted in query
configuration or cache metadata.  This module deliberately has no dependency on
the API package so core query and provider code can share the same typed error,
payload, and context-local call scope.
"""

from __future__ import annotations

import asyncio
import inspect
import ipaddress
import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable
from urllib.parse import urlsplit


SENSITIVE_CONTEXT_CHAT_FRAMING_RESERVE_TOKENS = 64
CHAT_MEMORY_QUERY_LLM_EGRESS_NOT_ALLOWED = (
    "chat_memory_query_llm_egress_not_allowed"
)


class SensitiveContextPolicyError(RuntimeError):
    """A stable, content-free hard policy failure.

    ``error_code`` and ``message`` are safe for API/NDJSON boundaries.  Provider
    exception text, endpoints, prompts, and fact content must never be attached
    to this exception or chained behind it.
    """

    def __init__(self, error_code: str, message: str):
        self.error_code = str(error_code)
        self.message = str(message)
        super().__init__(self.message)


class SensitiveLLMError(RuntimeError):
    """Content-free replacement for failures from a sensitive final LLM call."""

    def __init__(self) -> None:
        super().__init__("Sensitive LLM call failed")


@dataclass(frozen=True)
class SensitiveContextPayload:
    """Process-local split between trusted policy and untrusted context data."""

    trusted_policy: str
    context_data: str


FinalRequestBuilder = Callable[
    [SensitiveContextPayload | None], str | Awaitable[str]
]


@runtime_checkable
class SensitiveContext(Protocol):
    """Private lazy context supplied directly to final synthesis code."""

    async def resolve_for_final_request(
        self,
        tokenizer: Any,
        max_total_tokens: int,
        build_final_request: FinalRequestBuilder,
        policy_suffix: str = "",
    ) -> SensitiveContextPayload | None: ...


_SENSITIVE_LLM_CALL: ContextVar[bool] = ContextVar(
    "lightrag_sensitive_llm_call", default=False
)


def is_sensitive_call() -> bool:
    """Return whether the current provider execution carries sensitive context."""

    return _SENSITIVE_LLM_CALL.get()


@contextmanager
def sensitive_call_scope(enabled: bool = True):
    """Activate the context-local sensitive provider scope for one operation."""

    if not enabled:
        yield
        return
    token = _SENSITIVE_LLM_CALL.set(True)
    try:
        yield
    finally:
        _SENSITIVE_LLM_CALL.reset(token)


async def _close_async_iterator(iterator: Any) -> None:
    close = getattr(iterator, "aclose", None)
    if not callable(close):
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await cast(Awaitable[Any], result)
    except asyncio.CancelledError:
        raise
    except Exception:
        # Cleanup is best effort.  In particular, never replace a cancellation
        # or sanitized stream failure with provider-controlled close text.
        return


class SensitiveAsyncIterator(AsyncIterator[Any]):
    """Keep provider iteration/cleanup sensitive and sanitize stream failures."""

    def __init__(self, source: Any):
        self._source = source
        self._iterator: Any = None
        self._closed = False

    def _active_iterator(self) -> Any:
        if self._iterator is None:
            self._iterator = self._source.__aiter__()
        return self._iterator

    def _close_target(self) -> Any:
        return self._iterator if self._iterator is not None else self._source

    def __aiter__(self) -> "SensitiveAsyncIterator":
        return self

    async def __anext__(self) -> Any:
        token: Token[bool] = _SENSITIVE_LLM_CALL.set(True)
        sanitized_error: SensitiveLLMError | None = None
        try:
            return await self._active_iterator().__anext__()
        except StopAsyncIteration:
            self._closed = True
            raise
        except asyncio.CancelledError:
            await _close_async_iterator(self._close_target())
            self._closed = True
            raise
        except SensitiveContextPolicyError:
            await _close_async_iterator(self._close_target())
            self._closed = True
            raise
        except Exception:
            await _close_async_iterator(self._close_target())
            self._closed = True
            sanitized_error = SensitiveLLMError()
        finally:
            _SENSITIVE_LLM_CALL.reset(token)
        if sanitized_error is not None:
            raise sanitized_error
        raise StopAsyncIteration

    async def aclose(self) -> None:
        if self._closed:
            return
        token: Token[bool] = _SENSITIVE_LLM_CALL.set(True)
        sanitized_error: SensitiveLLMError | None = None
        try:
            await _close_async_iterator(self._close_target())
            self._closed = True
        except asyncio.CancelledError:
            self._closed = True
            raise
        except Exception:
            self._closed = True
            sanitized_error = SensitiveLLMError()
        finally:
            _SENSITIVE_LLM_CALL.reset(token)
        if sanitized_error is not None:
            raise sanitized_error


def wrap_sensitive_async_iterator(result: Any) -> Any:
    """Wrap an async iterator; leave ordinary scalar results unchanged."""

    if hasattr(result, "__aiter__"):
        return SensitiveAsyncIterator(result)
    return result


def _stable_message_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        return str(value)


def serialize_sensitive_final_request(
    system_prompt: str | None,
    query: str,
    history_messages: Sequence[Mapping[str, Any]] | None,
) -> str:
    """Serialize every final-request text field with deterministic separators.

    Token callers add :data:`SENSITIVE_CONTEXT_CHAT_FRAMING_RESERVE_TOKENS` to
    the encoded result for provider-specific chat framing that is not present in
    these textual fields.
    """

    parts = [
        "---LIGHTRAG FINAL SYSTEM PROMPT---",
        system_prompt or "",
        "---LIGHTRAG CONVERSATION HISTORY---",
    ]
    for index, message in enumerate(history_messages or ()):
        parts.extend(
            (
                f"---HISTORY {index} ROLE---",
                _stable_message_value(message.get("role", "")),
                f"---HISTORY {index} CONTENT---",
                _stable_message_value(message.get("content", "")),
            )
        )
    parts.extend(("---LIGHTRAG USER QUERY---", query))
    return "\n".join(parts)


def canonicalize_endpoint_identity(endpoint: str | None) -> str | None:
    """Return a credential-free canonical HTTP(S) endpoint identity.

    Host and scheme case, redundant/trailing slashes, default ports, IPv6
    spelling, credentials, query strings, and fragments do not affect identity.
    A scheme-less host uses the HTTPS default.  Malformed/non-HTTP endpoints are
    unknown and return ``None``.
    """

    raw = str(endpoint or "").strip()
    if not raw:
        return None
    if raw.startswith("//"):
        candidate = f"https:{raw}"
    elif "://" not in raw:
        candidate = f"https://{raw}"
    else:
        candidate = raw

    try:
        parsed = urlsplit(candidate)
        scheme = parsed.scheme.lower()
        if scheme not in {"http", "https"}:
            return None
        hostname = (parsed.hostname or "").strip().lower().rstrip(".")
        if not hostname:
            return None
        port = parsed.port
    except (TypeError, ValueError):
        return None

    try:
        normalized_ip = ipaddress.ip_address(hostname)
    except ValueError:
        try:
            hostname = hostname.encode("idna").decode("ascii").lower()
        except UnicodeError:
            return None
    else:
        hostname = normalized_ip.compressed.lower()

    rendered_host = f"[{hostname}]" if ":" in hostname else hostname
    default_port = 443 if scheme == "https" else 80
    rendered_port = f":{port}" if port is not None and port != default_port else ""
    path = re.sub(r"/+", "/", parsed.path or "").rstrip("/")
    return f"{scheme}://{rendered_host}{rendered_port}{path}"


def is_chat_memory_query_llm_egress_allowed(
    memory_endpoint: str | None,
    query_llm_endpoint: str | None,
    *,
    allow_cross_provider: bool = False,
) -> bool:
    """Apply the default-deny Chat Memory final-query egress policy."""

    if allow_cross_provider:
        return True
    memory_identity = canonicalize_endpoint_identity(memory_endpoint)
    query_identity = canonicalize_endpoint_identity(query_llm_endpoint)
    return (
        memory_identity is not None
        and query_identity is not None
        and memory_identity == query_identity
    )


def ensure_chat_memory_query_llm_egress_allowed(
    memory_endpoint: str | None,
    query_llm_endpoint: str | None,
    *,
    allow_cross_provider: bool = False,
) -> None:
    """Raise the stable content-free hard error when egress is not allowed."""

    if is_chat_memory_query_llm_egress_allowed(
        memory_endpoint,
        query_llm_endpoint,
        allow_cross_provider=allow_cross_provider,
    ):
        return
    raise SensitiveContextPolicyError(
        CHAT_MEMORY_QUERY_LLM_EGRESS_NOT_ALLOWED,
        CHAT_MEMORY_QUERY_LLM_EGRESS_NOT_ALLOWED,
    ) from None


# Descriptive aliases for callers/tests that do not need the Chat Memory prefix.
canonicalize_llm_endpoint_identity = canonicalize_endpoint_identity
validate_chat_memory_query_llm_egress = ensure_chat_memory_query_llm_egress_allowed


def bind_sensitive_context_endpoint(
    sensitive_context: SensitiveContext | None, endpoint: str | None
) -> None:
    """Bind the exact final-synthesis runtime endpoint when supported."""

    if sensitive_context is None:
        return
    bind = getattr(sensitive_context, "bind_final_llm_endpoint", None)
    if callable(bind):
        bind(endpoint)


def mark_sensitive_context_not_used(
    sensitive_context: SensitiveContext | None, reason: str
) -> None:
    """Freeze a requested context as not used without invoking its search."""

    if sensitive_context is None:
        return
    mark = getattr(sensitive_context, "mark_not_used", None)
    if callable(mark):
        mark(reason)
