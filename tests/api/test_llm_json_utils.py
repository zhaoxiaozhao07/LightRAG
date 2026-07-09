"""Tests for the shared JSON-constrained role-LLM call helper.

Focus: forwarding of schema response_format payloads and the degradation
chain when a backend rejects them (json_schema → json_object → no
response_format kwarg).
"""

import pytest

from lightrag.api.llm_json_utils import call_llm_json

SCHEMA_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "demo", "schema": {"type": "object"}},
}


class _ScriptedLLM:
    """Async LLM stub that replays a script of return values / exceptions
    and records the kwargs of every call."""

    def __init__(self, script):
        self.calls: list[dict] = []
        self._script = list(script)

    async def __call__(self, prompt, **kwargs):
        self.calls.append(kwargs)
        action = self._script.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


def _formats(llm: _ScriptedLLM) -> list:
    return [call.get("response_format") for call in llm.calls]


async def _call(llm: _ScriptedLLM, **overrides):
    kwargs = {
        "system_prompt": "sys",
        "priority": 5,
        "parse": lambda data: data,
        "label": "test_json",
        "response_format": SCHEMA_FORMAT,
    }
    kwargs.update(overrides)
    return await call_llm_json(llm, "prompt", **kwargs)


@pytest.mark.asyncio
async def test_schema_response_format_is_forwarded_verbatim():
    llm = _ScriptedLLM(['{"ok": true}'])

    result = await _call(llm)

    assert result == {"ok": True}
    assert _formats(llm) == [SCHEMA_FORMAT]


@pytest.mark.asyncio
async def test_schema_rejection_downgrades_to_json_object():
    llm = _ScriptedLLM(
        [
            RuntimeError("response_format type json_schema is not supported"),
            '{"ok": 1}',
        ]
    )

    result = await _call(llm)

    assert result == {"ok": 1}
    assert _formats(llm) == [SCHEMA_FORMAT, {"type": "json_object"}]


@pytest.mark.asyncio
async def test_downgrade_persists_across_retry_attempts():
    llm = _ScriptedLLM(
        [
            RuntimeError("failed to parse grammar from JSON schema"),
            "这不是 JSON",
            '{"ok": 1}',
        ]
    )

    result = await _call(llm)

    assert result == {"ok": 1}
    # After the downgrade, later attempts go straight to json_object instead
    # of re-trying the rejected schema payload.
    assert _formats(llm) == [
        SCHEMA_FORMAT,
        {"type": "json_object"},
        {"type": "json_object"},
    ]


@pytest.mark.asyncio
async def test_unrelated_llm_error_propagates_immediately():
    llm = _ScriptedLLM([RuntimeError("rate limit exceeded")])

    with pytest.raises(RuntimeError, match="rate limit"):
        await _call(llm)

    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_downgrade_typeerror_drops_response_format_kwarg():
    llm = _ScriptedLLM(
        [
            RuntimeError("unsupported json_schema response_format"),
            TypeError("unexpected keyword argument 'response_format'"),
            '{"ok": 1}',
        ]
    )

    result = await _call(llm)

    assert result == {"ok": 1}
    assert _formats(llm)[:2] == [SCHEMA_FORMAT, {"type": "json_object"}]
    assert "response_format" not in llm.calls[2]
