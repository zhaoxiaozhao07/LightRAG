from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from lightrag.artifact_runtime import (
    PipelineAttemptCommitOutcomeUnknownError,
    extract_pipeline_attempt_token,
)
from lightrag.kg import redis_impl
from lightrag.namespace import NameSpace

pytestmark = pytest.mark.offline
RedisConnectionError = redis_impl.ConnectionError


class _DummyEmbeddingFunc:
    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


@dataclass(frozen=True)
class _RowCase:
    storage_class: type
    namespace: str
    row_kind: str

    def row(self, token: str | None, marker: str) -> dict[str, Any]:
        if self.row_kind == "full_docs":
            binding = {} if token is None else {"claim_token": token}
            return {"artifact_binding": binding, "content": marker}
        metadata = {} if token is None else {"pipeline_attempt_token": token}
        return {"metadata": metadata, "status": marker}


_ROW_CASES = (
    _RowCase(
        redis_impl.RedisKVStorage,
        NameSpace.KV_STORE_FULL_DOCS,
        "full_docs",
    ),
    _RowCase(
        redis_impl.RedisDocStatusStorage,
        NameSpace.DOC_STATUS,
        "doc_status",
    ),
)


class _ScriptRedis:
    """Script-capable fake for the Redis string-key encoding used by LightRAG."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.ttl_ms: dict[str, int] = {}
        self.eval_calls: list[tuple[Any, ...]] = []
        self.get_calls: list[str] = []
        self.write_count = 0
        self.raise_after_eval = False
        self.after_eval_value: str | None = None

    async def eval(self, script: str, numkeys: int, *keys_and_args: Any) -> int:
        self.eval_calls.append((script, numkeys, *keys_and_args))
        assert numkeys == 1
        redis_key, row_kind, expected_token, candidate = keys_and_args

        current_raw = self.store.get(redis_key)
        current = json.loads(current_raw) if current_raw is not None else None
        current_token = extract_pipeline_attempt_token(current, row_kind=row_kind)
        result = 0
        if current_token == expected_token:
            # Mirrors SET ... KEEPTTL in the production Lua script.
            self.store[redis_key] = candidate
            self.write_count += 1
            result = 1

        if self.after_eval_value is not None:
            self.store[redis_key] = self.after_eval_value
        if self.raise_after_eval:
            raise RedisConnectionError("script response lost")
        return result

    async def get(self, key: str) -> str | None:
        self.get_calls.append(key)
        return self.store.get(key)


def _make_storage(
    monkeypatch: pytest.MonkeyPatch,
    case: _RowCase,
) -> tuple[Any, _ScriptRedis]:
    fake = _ScriptRedis()
    monkeypatch.setattr(
        redis_impl.RedisConnectionManager,
        "get_pool",
        lambda _redis_url: object(),
    )
    monkeypatch.setattr(
        redis_impl.RedisConnectionManager,
        "release_pool",
        lambda _redis_url: None,
    )
    monkeypatch.setattr(
        redis_impl,
        "Redis",
        lambda connection_pool=None: fake,
    )
    storage = case.storage_class(
        namespace=case.namespace,
        global_config={},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="cas-test",
    )
    return storage, fake


@pytest.mark.parametrize("case", _ROW_CASES, ids=lambda case: case.row_kind)
async def test_pipeline_attempt_cas_uses_exact_atomic_script_key_and_payload(
    monkeypatch: pytest.MonkeyPatch,
    case: _RowCase,
):
    storage, fake = _make_storage(monkeypatch, case)
    redis_key = f"{storage.final_namespace}:doc-1"
    current = case.row("attempt-1", "before")
    candidate = case.row("attempt-1", "after")
    candidate_before = json.loads(json.dumps(candidate))
    candidate_json = json.dumps(candidate)
    fake.store[redis_key] = json.dumps(current)
    fake.ttl_ms[redis_key] = 60_000

    committed = await storage.compare_and_commit_pipeline_attempt(
        "doc-1",
        candidate,
        expected_attempt_token="attempt-1",
        row_kind=case.row_kind,
    )

    assert committed is True
    assert fake.store[redis_key] == candidate_json
    assert candidate == candidate_before
    assert fake.write_count == 1
    assert fake.ttl_ms[redis_key] == 60_000
    assert fake.get_calls == []
    assert fake.eval_calls == [
        (
            redis_impl._PIPELINE_ATTEMPT_COMPARE_AND_SET_LUA,
            1,
            redis_key,
            case.row_kind,
            "attempt-1",
            candidate_json,
        )
    ]
    assert (
        'redis.call("SET", KEYS[1], ARGV[3], "KEEPTTL")'
        in redis_impl._PIPELINE_ATTEMPT_COMPARE_AND_SET_LUA
    )
    assert (
        'decoded["artifact_binding"]'
        in redis_impl._PIPELINE_ATTEMPT_COMPARE_AND_SET_LUA
    )
    assert 'binding["claim_token"]' in redis_impl._PIPELINE_ATTEMPT_COMPARE_AND_SET_LUA
    assert 'decoded["metadata"]' in redis_impl._PIPELINE_ATTEMPT_COMPARE_AND_SET_LUA
    assert (
        'metadata["pipeline_attempt_token"]'
        in redis_impl._PIPELINE_ATTEMPT_COMPARE_AND_SET_LUA
    )


@pytest.mark.parametrize("case", _ROW_CASES, ids=lambda case: case.row_kind)
@pytest.mark.parametrize("stored_token", ["newer-attempt", None])
async def test_pipeline_attempt_cas_mismatch_or_missing_token_has_zero_writes(
    monkeypatch: pytest.MonkeyPatch,
    case: _RowCase,
    stored_token: str | None,
):
    storage, fake = _make_storage(monkeypatch, case)
    redis_key = f"{storage.final_namespace}:doc-1"
    current_json = json.dumps(case.row(stored_token, "unchanged"))
    fake.store[redis_key] = current_json

    committed = await storage.compare_and_commit_pipeline_attempt(
        "doc-1",
        case.row("attempt-1", "candidate"),
        expected_attempt_token="attempt-1",
        row_kind=case.row_kind,
    )

    assert committed is False
    assert fake.store[redis_key] == current_json
    assert fake.write_count == 0
    assert fake.get_calls == []
    assert len(fake.eval_calls) == 1


@pytest.mark.parametrize("case", _ROW_CASES, ids=lambda case: case.row_kind)
@pytest.mark.parametrize("readback_outcome", ["exact", "different-token", "same-token"])
async def test_pipeline_attempt_cas_reconciles_ambiguous_script_transport_errors(
    monkeypatch: pytest.MonkeyPatch,
    case: _RowCase,
    readback_outcome: str,
):
    storage, fake = _make_storage(monkeypatch, case)
    redis_key = f"{storage.final_namespace}:doc-1"
    candidate = case.row("attempt-1", "candidate")
    candidate_json = json.dumps(candidate)
    fake.store[redis_key] = json.dumps(case.row("attempt-1", "before"))
    fake.raise_after_eval = True

    if readback_outcome == "different-token":
        fake.after_eval_value = json.dumps(case.row("attempt-2", "winner"))
    elif readback_outcome == "same-token":
        fake.after_eval_value = json.dumps(case.row("attempt-1", "other-payload"))

    call = storage.compare_and_commit_pipeline_attempt(
        "doc-1",
        candidate,
        expected_attempt_token="attempt-1",
        row_kind=case.row_kind,
    )
    if readback_outcome == "same-token":
        with pytest.raises(PipelineAttemptCommitOutcomeUnknownError) as exc_info:
            await call
        assert exc_info.value.key == "doc-1"
        assert exc_info.value.row_kind == case.row_kind
        assert exc_info.value.reason == "ConnectionError"
        assert isinstance(exc_info.value.__cause__, RedisConnectionError)
    else:
        assert await call is (readback_outcome == "exact")

    assert fake.write_count == 1
    assert fake.get_calls == [redis_key]
    assert len(fake.eval_calls) == 1
    if readback_outcome == "exact":
        assert fake.store[redis_key] == candidate_json
