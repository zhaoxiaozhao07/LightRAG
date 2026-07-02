"""Shared helpers for role-LLM calls that must return strict JSON.

Used by the Agent orchestration (plan generation) and the Agent profile
generation (document/KB profiles). Both call OpenAI-compatible role LLM
functions that are asked for strict JSON output; local models occasionally
wrap the payload in code fences or emit invalid JSON, so parsing failures
are retried a bounded number of times before surfacing an error.
"""

from __future__ import annotations

import json
from functools import partial
from typing import Any, Callable, TypeVar

from pydantic import ValidationError

from lightrag.utils import logger

T = TypeVar("T")

DEFAULT_LLM_JSON_ATTEMPTS = 3


class LLMJsonError(ValueError):
    """Raised when an LLM did not produce a parseable/valid JSON payload
    within the allowed number of attempts."""


def strip_code_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    return value


async def call_llm_json(
    llm_func: Callable[..., Any],
    prompt: str,
    *,
    system_prompt: str,
    priority: int,
    parse: Callable[[Any], T],
    attempts: int = DEFAULT_LLM_JSON_ATTEMPTS,
    label: str = "llm_json",
) -> T:
    """Call a role LLM function and parse its JSON output with bounded retries.

    ``parse`` receives the decoded JSON value (or a raw dict returned by the
    binding) and must return the validated result; it should raise
    ``ValueError`` / ``ValidationError`` for semantically invalid payloads so
    the call is retried. Bindings that do not accept ``response_format``
    raise ``TypeError`` on the first call; the fallback (without the kwarg)
    is remembered for subsequent attempts.
    """
    attempts = max(1, int(attempts))
    bound = partial(llm_func, _priority=priority)
    supports_response_format = True
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        if supports_response_format:
            try:
                raw = await bound(
                    prompt,
                    system_prompt=system_prompt,
                    stream=False,
                    enable_cot=False,
                    response_format={"type": "json_object"},
                )
            except TypeError:
                supports_response_format = False
                raw = await bound(
                    prompt,
                    system_prompt=system_prompt,
                    stream=False,
                    enable_cot=False,
                )
        else:
            raw = await bound(
                prompt,
                system_prompt=system_prompt,
                stream=False,
                enable_cot=False,
            )
        try:
            data = raw if isinstance(raw, dict) else json.loads(strip_code_fence(str(raw)))
            return parse(data)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "%s: attempt %d/%d returned invalid JSON payload: %s",
                label,
                attempt,
                attempts,
                exc,
            )
    raise LLMJsonError(str(last_error))
