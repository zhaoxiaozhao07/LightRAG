import pytest

from lightrag.base import QueryContextResult, QueryParam
from lightrag.operate import kg_query, naive_query
from lightrag.sensitive_context import SensitiveContextPayload


class _FakeTokenizer:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


class _FakeKVStorage:
    def __init__(self):
        self.global_config = {"enable_llm_cache": True}
        self._store = {}

    async def get_by_id(self, key):
        return self._store.get(key)

    async def upsert(self, entries):
        self._store.update(entries)


class _FakeChunksVDB:
    cosine_better_than_threshold = 0.0

    async def query(self, *_args, **_kwargs):
        return [
            {
                "id": "chunk-1",
                "content": "LightRAG cache identity test chunk.",
                "file_path": "test.md",
            }
        ]


class _NoContentSensitiveContext:
    def __init__(self):
        self.endpoint = None
        self.resolve_calls = 0
        self.final_requests: list[str] = []

    def bind_final_llm_endpoint(self, endpoint):
        self.endpoint = endpoint

    async def resolve_for_final_request(
        self, tokenizer, max_total_tokens, build_final_request, policy_suffix=""
    ):
        self.resolve_calls += 1
        self.final_requests.append(build_final_request(None))
        return None


class _PayloadSensitiveContext(_NoContentSensitiveContext):
    async def resolve_for_final_request(
        self, tokenizer, max_total_tokens, build_final_request, policy_suffix=""
    ):
        self.resolve_calls += 1
        payload = SensitiveContextPayload(
            trusted_policy="TRUSTED MEMORY POLICY",
            context_data="UNTRUSTED MEMORY JSONL",
        )
        self.final_requests.append(build_final_request(payload))
        return payload


def _query_global_config(model: str, llm_func) -> dict:
    return {
        "tokenizer": _FakeTokenizer(),
        "role_llm_funcs": {"query": llm_func},
        "llm_cache_identities": {
            "query": {
                "role": "query",
                "binding": "openai",
                "model": model,
                "host": "https://api.example.com/v1",
            }
        },
        "min_rerank_score": 0.0,
        "max_total_tokens": 4096,
    }


@pytest.mark.offline
@pytest.mark.asyncio
async def test_naive_query_partitions_query_cache_by_llm_identity():
    cache = _FakeKVStorage()
    chunks_vdb = _FakeChunksVDB()
    calls = 0

    async def query_model(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return f"answer-{calls}"

    param = QueryParam(mode="naive", enable_rerank=False)

    first = await naive_query(
        "same query",
        chunks_vdb,
        param,
        _query_global_config("model-a", query_model),
        hashing_kv=cache,
    )
    second = await naive_query(
        "same query",
        chunks_vdb,
        param,
        _query_global_config("model-b", query_model),
        hashing_kv=cache,
    )

    assert first.content == "answer-1"
    assert second.content == "answer-2"
    assert calls == 2
    assert len(cache._store) == 2


@pytest.mark.offline
@pytest.mark.asyncio
async def test_sensitive_naive_query_ignores_prepopulated_cache_and_does_not_write():
    cache = _FakeKVStorage()
    chunks_vdb = _FakeChunksVDB()
    calls = []

    async def query_model(*_args, **kwargs):
        calls.append(dict(kwargs))
        return f"answer-{len(calls)}"

    param = QueryParam(
        mode="naive",
        enable_rerank=False,
        conversation_history=[{"role": "user", "content": "prior"}],
    )
    config = _query_global_config("model-a", query_model)
    ordinary = await naive_query(
        "same query", chunks_vdb, param, config, hashing_kv=cache
    )
    before = dict(cache._store)
    sensitive = _NoContentSensitiveContext()
    fresh = await naive_query(
        "same query",
        chunks_vdb,
        param,
        config,
        hashing_kv=cache,
        sensitive_context=sensitive,
    )

    assert ordinary.content == "answer-1"
    assert fresh.content == "answer-2"
    assert len(calls) == 2
    assert calls[-1]["_sensitive"] is True
    assert cache._store == before
    assert sensitive.endpoint == "https://api.example.com/v1"
    assert sensitive.resolve_calls == 1
    assert "prior" in sensitive.final_requests[0]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_sensitive_kg_query_ignores_prepopulated_cache_and_does_not_write(
    monkeypatch,
):
    import lightrag.operate as operate_module

    cache = _FakeKVStorage()
    calls = []

    async def query_model(*_args, **kwargs):
        calls.append(dict(kwargs))
        return f"kg-answer-{len(calls)}"

    async def fake_keywords(*_args, **_kwargs):
        return ["high"], ["low"]

    async def fake_context(*_args, **_kwargs):
        return QueryContextResult(
            context="AUTHORITATIVE KG CONTEXT",
            raw_data={"data": {"references": [], "chunks": []}},
        )

    monkeypatch.setattr(operate_module, "get_keywords_from_query", fake_keywords)
    monkeypatch.setattr(operate_module, "_build_query_context", fake_context)

    param = QueryParam(mode="local", enable_rerank=False)
    config = _query_global_config("model-a", query_model)
    ordinary = await kg_query(
        "same query",
        object(),
        object(),
        object(),
        object(),
        param,
        config,
        hashing_kv=cache,
        chunks_vdb=object(),
    )
    before = dict(cache._store)
    sensitive = _NoContentSensitiveContext()
    fresh = await kg_query(
        "same query",
        object(),
        object(),
        object(),
        object(),
        param,
        config,
        hashing_kv=cache,
        chunks_vdb=object(),
        sensitive_context=sensitive,
    )

    assert ordinary.content == "kg-answer-1"
    assert fresh.content == "kg-answer-2"
    assert len(calls) == 2
    assert calls[-1]["_sensitive"] is True
    assert cache._store == before
    assert sensitive.resolve_calls == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_custom_naive_prompt_receives_split_sensitive_sections():
    calls = []

    async def query_model(prompt, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        return "answer"

    sensitive = _PayloadSensitiveContext()
    result = await naive_query(
        "What is the answer?",
        _FakeChunksVDB(),
        QueryParam(mode="naive", max_total_tokens=4096),
        _query_global_config("model-a", query_model),
        hashing_kv=_FakeKVStorage(),
        system_prompt="CUSTOM SYSTEM",
        sensitive_context=sensitive,
    )

    assert result is not None
    sent_prompt = calls[0]["system_prompt"]
    assert sent_prompt.startswith("CUSTOM SYSTEM")
    assert "---Additional Instructions---\n\nTRUSTED MEMORY POLICY" in sent_prompt
    assert "---Context---" in sent_prompt
    assert "UNTRUSTED MEMORY JSONL" in sent_prompt
    assert "LightRAG cache identity test chunk." in sent_prompt
    assert sent_prompt.index("TRUSTED MEMORY POLICY") < sent_prompt.index(
        "UNTRUSTED MEMORY JSONL"
    )
    assert calls[0]["_sensitive"] is True
