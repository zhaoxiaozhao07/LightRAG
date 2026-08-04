from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import pytest

pytest.importorskip(
    "opensearchpy",
    reason="opensearchpy is required for OpenSearch storage tests",
)

from opensearchpy.exceptions import (  # type: ignore
    ConflictError,
    NotFoundError,
    OpenSearchException,
    RequestError,
)

from lightrag.artifact_runtime import PipelineAttemptCommitOutcomeUnknownError
from lightrag.kg import opensearch_impl

pytestmark = pytest.mark.offline

_KEY = "doc-1"
_TOKEN = "attempt-1"


@dataclass(frozen=True)
class _RowCase:
    storage_class: type
    namespace: str
    row_kind: str
    token_container: str
    token_field: str

    def row(self, token: str | None, marker: str) -> dict[str, Any]:
        if self.row_kind == "full_docs":
            binding = {} if token is None else {"claim_token": token}
            return {"artifact_binding": binding, "content": marker}
        metadata = {} if token is None else {"pipeline_attempt_token": token}
        return {"metadata": metadata, "status": marker}


_ROW_CASES = (
    _RowCase(
        opensearch_impl.OpenSearchKVStorage,
        "full_docs",
        "full_docs",
        "artifact_binding",
        "claim_token",
    ),
    _RowCase(
        opensearch_impl.OpenSearchDocStatusStorage,
        "doc_status",
        "doc_status",
        "metadata",
        "pipeline_attempt_token",
    ),
)


class _FakeOpenSearchClient:
    """Minimal update/mget fake for the atomic OpenSearch request contract."""

    def __init__(self, source: dict[str, Any] | None) -> None:
        self.source = deepcopy(source)
        self.update_calls: list[dict[str, Any]] = []
        self.mget_calls: list[dict[str, Any]] = []
        self.bulk_calls: list[list[dict[str, Any]]] = []
        self.operations: list[tuple[str, str]] = []
        self.write_count = 0
        self.update_error: OpenSearchException | None = None
        self.raise_after_update = False
        self.after_update_source: dict[str, Any] | None = None
        self.bulk_failed: list[Any] = []
        self.indices = _FakeIndices(self)

    async def update(self, **kwargs: Any) -> dict[str, Any]:
        self.update_calls.append(deepcopy(kwargs))
        self.operations.append(("cas", kwargs["id"]))
        if self.update_error is not None and not self.raise_after_update:
            raise self.update_error

        script_params = kwargs["body"]["script"]["params"]
        current_token = None
        if isinstance(self.source, dict):
            container = self.source.get(script_params["token_container"])
            if isinstance(container, dict):
                current_token = container.get(script_params["token_field"])

        result = "not_found" if self.source is None else "noop"
        if current_token == script_params["expected_attempt_token"]:
            self.source = deepcopy(script_params["payload"])
            self.write_count += 1
            result = "updated"

        if self.after_update_source is not None:
            self.source = deepcopy(self.after_update_source)
        if self.update_error is not None:
            raise self.update_error
        return {"result": result}

    async def mget(self, **kwargs: Any) -> dict[str, Any]:
        self.mget_calls.append(deepcopy(kwargs))
        doc_id = kwargs["body"]["ids"][0]
        if self.source is None:
            return {"docs": [{"_id": doc_id, "found": False}]}
        return {
            "docs": [
                {
                    "_id": doc_id,
                    "found": True,
                    "_source": deepcopy(self.source),
                }
            ]
        }

    async def apply_bulk(self, actions: list[dict[str, Any]]) -> tuple[int, list[Any]]:
        """Apply fake bulk operations to the same server row used by CAS."""
        copied_actions = deepcopy(actions)
        self.bulk_calls.append(copied_actions)
        failed_ids = {
            payload["_id"]
            for failure in self.bulk_failed
            for payload in failure.values()
        }
        success = 0
        for action in copied_actions:
            op_type = action["_op_type"]
            doc_id = action["_id"]
            self.operations.append((f"bulk-{op_type}", doc_id))
            if doc_id in failed_ids:
                continue
            if op_type == "delete":
                self.source = None
            else:
                self.source = deepcopy(action["_source"])
            success += 1
        return success, deepcopy(self.bulk_failed)


class _FakeIndices:
    def __init__(self, client: _FakeOpenSearchClient) -> None:
        self._client = client

    async def refresh(self, *, index: str) -> None:
        self._client.operations.append(("refresh", index))


async def _fake_async_bulk(
    client: _FakeOpenSearchClient,
    actions: list[dict[str, Any]],
    **_: Any,
) -> tuple[int, list[Any]]:
    return await client.apply_bulk(actions)


def _storage(case: _RowCase, client: _FakeOpenSearchClient) -> Any:
    storage = case.storage_class.__new__(case.storage_class)
    storage.workspace = "workspace-a"
    storage.namespace = case.namespace
    storage.final_namespace = f"{storage.workspace}_{case.namespace}"
    storage._index_name = opensearch_impl._sanitize_index_name(storage.final_namespace)
    storage.client = client
    if case.storage_class is opensearch_impl.OpenSearchKVStorage:
        storage._index_ready = True
        storage._pending_upserts = {}
        storage._pending_kv_deletes = set()
        storage._flush_lock = asyncio.Lock()
        storage._max_upsert_payload_bytes = 1024 * 1024
        storage._max_upsert_records_per_batch = 128
        storage._max_delete_records_per_batch = 1000
    return storage


def _stored_source(case: _RowCase, token: str | None, marker: str) -> dict[str, Any]:
    return {**case.row(token, marker), "__mirrored_id": _KEY}


@pytest.mark.parametrize("case", _ROW_CASES, ids=lambda case: case.row_kind)
async def test_pipeline_attempt_cas_uses_exact_atomic_update_request(
    case: _RowCase,
) -> None:
    client = _FakeOpenSearchClient(_stored_source(case, _TOKEN, "before"))
    storage = _storage(case, client)
    payload = {**case.row(_TOKEN, "after"), "_id": _KEY}
    payload_before = deepcopy(payload)

    committed = await storage.compare_and_commit_pipeline_attempt(
        _KEY,
        payload,
        expected_attempt_token=_TOKEN,
        row_kind=case.row_kind,
    )

    replacement = {**case.row(_TOKEN, "after"), "__mirrored_id": _KEY}
    assert committed is True
    assert payload == payload_before
    assert client.source == replacement
    assert client.write_count == 1
    assert client.mget_calls == []
    assert client.update_calls == [
        {
            "index": f"workspace-a_{case.namespace}",
            "id": _KEY,
            "body": {
                "script": {
                    "lang": "painless",
                    "source": opensearch_impl._PIPELINE_ATTEMPT_CAS_SCRIPT,
                    "params": {
                        "token_container": case.token_container,
                        "token_field": case.token_field,
                        "expected_attempt_token": _TOKEN,
                        "payload": replacement,
                    },
                }
            },
            "params": {"refresh": "wait_for", "retry_on_conflict": 3},
        }
    ]
    script = opensearch_impl._PIPELINE_ATTEMPT_CAS_SCRIPT
    assert "ctx.op = 'noop'" in script
    assert "ctx._source.clear()" in script
    assert "ctx._source.putAll(params.payload)" in script


@pytest.mark.parametrize("case", _ROW_CASES, ids=lambda case: case.row_kind)
@pytest.mark.parametrize("stored_token", ["newer-attempt", None])
async def test_pipeline_attempt_cas_mismatch_or_missing_token_has_zero_mutation(
    case: _RowCase,
    stored_token: str | None,
) -> None:
    original = _stored_source(case, stored_token, "unchanged")
    client = _FakeOpenSearchClient(original)
    storage = _storage(case, client)

    committed = await storage.compare_and_commit_pipeline_attempt(
        _KEY,
        case.row(_TOKEN, "candidate"),
        expected_attempt_token=_TOKEN,
        row_kind=case.row_kind,
    )

    assert committed is False
    assert client.source == original
    assert client.write_count == 0
    assert client.mget_calls == []
    assert len(client.update_calls) == 1


@pytest.mark.parametrize("case", _ROW_CASES, ids=lambda case: case.row_kind)
async def test_pipeline_attempt_cas_missing_row_is_false(case: _RowCase) -> None:
    client = _FakeOpenSearchClient(None)
    client.update_error = NotFoundError(404, "not_found", {})
    storage = _storage(case, client)

    committed = await storage.compare_and_commit_pipeline_attempt(
        _KEY,
        case.row(_TOKEN, "candidate"),
        expected_attempt_token=_TOKEN,
        row_kind=case.row_kind,
    )

    assert committed is False
    assert client.write_count == 0
    assert client.mget_calls == []


@pytest.mark.parametrize(
    "write_error",
    [
        ConflictError(409, "version_conflict_engine_exception", {}),
        RequestError(400, "script_exception", {}),
    ],
    ids=["version-conflict", "script-conflict"],
)
async def test_pipeline_attempt_cas_preserves_known_version_and_script_errors(
    write_error: OpenSearchException,
) -> None:
    case = _ROW_CASES[0]
    original = _stored_source(case, _TOKEN, "before")
    client = _FakeOpenSearchClient(original)
    client.update_error = write_error
    storage = _storage(case, client)

    with pytest.raises(type(write_error)) as exc_info:
        await storage.compare_and_commit_pipeline_attempt(
            _KEY,
            case.row(_TOKEN, "candidate"),
            expected_attempt_token=_TOKEN,
            row_kind=case.row_kind,
        )

    assert exc_info.value is write_error
    assert client.source == original
    assert client.write_count == 0
    assert client.mget_calls == []
    assert client.update_calls[0]["params"]["retry_on_conflict"] == 3


@pytest.mark.parametrize("case", _ROW_CASES, ids=lambda case: case.row_kind)
@pytest.mark.parametrize(
    ("readback_outcome", "expected_result", "raises_unknown"),
    [
        ("exact", True, False),
        ("different-token", False, False),
        ("same-token-different-payload", None, True),
    ],
)
async def test_pipeline_attempt_cas_reconciles_ambiguous_transport_errors(
    case: _RowCase,
    readback_outcome: str,
    expected_result: bool | None,
    raises_unknown: bool,
) -> None:
    candidate = case.row(_TOKEN, "candidate")
    replacement = {**deepcopy(candidate), "__mirrored_id": _KEY}
    client = _FakeOpenSearchClient(_stored_source(case, _TOKEN, "before"))
    write_error = OpenSearchException("update acknowledgement was lost")
    client.update_error = write_error
    client.raise_after_update = True

    if readback_outcome == "different-token":
        client.after_update_source = _stored_source(case, "newer-attempt", "winner")
    elif readback_outcome == "same-token-different-payload":
        client.after_update_source = _stored_source(case, _TOKEN, "other-payload")

    storage = _storage(case, client)
    commit = storage.compare_and_commit_pipeline_attempt(
        _KEY,
        candidate,
        expected_attempt_token=_TOKEN,
        row_kind=case.row_kind,
    )
    if raises_unknown:
        with pytest.raises(PipelineAttemptCommitOutcomeUnknownError) as exc_info:
            await commit
        assert exc_info.value.key == _KEY
        assert exc_info.value.row_kind == case.row_kind
        assert exc_info.value.__cause__ is write_error
    else:
        assert await commit is expected_result

    assert client.write_count == 1
    assert client.update_calls[0]["body"]["script"]["params"]["payload"] == (
        replacement
    )
    assert client.mget_calls == [
        {
            "index": f"workspace-a_{case.namespace}",
            "body": {"ids": [_KEY]},
        }
    ]


async def test_kv_pipeline_attempt_cas_flushes_newer_pending_row_before_cas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _ROW_CASES[0]
    client = _FakeOpenSearchClient(_stored_source(case, _TOKEN, "server-old"))
    storage = _storage(case, client)
    monkeypatch.setattr(opensearch_impl.helpers, "async_bulk", _fake_async_bulk)

    await storage.upsert({_KEY: case.row("attempt-newer", "pending-newer")})
    buffered_source = deepcopy(storage._pending_upserts[_KEY])

    committed = await asyncio.wait_for(
        storage.compare_and_commit_pipeline_attempt(
            _KEY,
            case.row(_TOKEN, "stale-candidate"),
            expected_attempt_token=_TOKEN,
            row_kind=case.row_kind,
        ),
        timeout=1,
    )

    assert committed is False
    assert client.operations == [("bulk-index", _KEY), ("cas", _KEY)]
    assert client.bulk_calls == [
        [
            {
                "_op_type": "index",
                "_index": storage._index_name,
                "_id": _KEY,
                "_source": buffered_source,
            }
        ]
    ]
    assert client.source == buffered_source
    assert client.write_count == 0
    assert storage._pending_upserts == {}
    assert storage._pending_kv_deletes == set()

    source_after_cas = deepcopy(client.source)
    await asyncio.wait_for(storage.index_done_callback(), timeout=1)

    assert client.source == source_after_cas
    assert client.bulk_calls == [
        [
            {
                "_op_type": "index",
                "_index": storage._index_name,
                "_id": _KEY,
                "_source": buffered_source,
            }
        ]
    ]
    assert client.operations == [
        ("bulk-index", _KEY),
        ("cas", _KEY),
        ("refresh", storage._index_name),
    ]


async def test_kv_pipeline_attempt_cas_flushes_same_token_then_commits_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _ROW_CASES[0]
    client = _FakeOpenSearchClient(_stored_source(case, _TOKEN, "server-old"))
    storage = _storage(case, client)
    monkeypatch.setattr(opensearch_impl.helpers, "async_bulk", _fake_async_bulk)

    await storage.upsert({_KEY: case.row(_TOKEN, "pending-same-token")})
    buffered_source = deepcopy(storage._pending_upserts[_KEY])
    candidate = case.row(_TOKEN, "committed")

    committed = await asyncio.wait_for(
        storage.compare_and_commit_pipeline_attempt(
            _KEY,
            candidate,
            expected_attempt_token=_TOKEN,
            row_kind=case.row_kind,
        ),
        timeout=1,
    )

    assert committed is True
    assert client.operations == [("bulk-index", _KEY), ("cas", _KEY)]
    assert client.bulk_calls == [
        [
            {
                "_op_type": "index",
                "_index": storage._index_name,
                "_id": _KEY,
                "_source": buffered_source,
            }
        ]
    ]
    assert client.source == {**candidate, "__mirrored_id": _KEY}
    assert client.write_count == 1
    assert len(client.update_calls) == 1
    assert storage._pending_upserts == {}
    assert storage._pending_kv_deletes == set()


async def test_kv_pipeline_attempt_cas_flushes_pending_delete_before_cas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _ROW_CASES[0]
    client = _FakeOpenSearchClient(_stored_source(case, _TOKEN, "server-old"))
    storage = _storage(case, client)
    monkeypatch.setattr(opensearch_impl.helpers, "async_bulk", _fake_async_bulk)

    await storage.delete([_KEY])

    committed = await asyncio.wait_for(
        storage.compare_and_commit_pipeline_attempt(
            _KEY,
            case.row(_TOKEN, "candidate"),
            expected_attempt_token=_TOKEN,
            row_kind=case.row_kind,
        ),
        timeout=1,
    )

    assert committed is False
    assert client.operations == [("bulk-delete", _KEY), ("cas", _KEY)]
    assert client.bulk_calls == [
        [
            {
                "_op_type": "delete",
                "_index": storage._index_name,
                "_id": _KEY,
            }
        ]
    ]
    assert client.source is None
    assert client.write_count == 0
    assert storage._pending_upserts == {}
    assert storage._pending_kv_deletes == set()


async def test_kv_pipeline_attempt_cas_skips_cas_when_flush_must_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _ROW_CASES[0]
    original = _stored_source(case, _TOKEN, "server-old")
    client = _FakeOpenSearchClient(original)
    storage = _storage(case, client)
    monkeypatch.setattr(opensearch_impl.helpers, "async_bulk", _fake_async_bulk)

    await storage.upsert({_KEY: case.row(_TOKEN, "retry-me")})
    buffered_source = deepcopy(storage._pending_upserts[_KEY])
    client.bulk_failed = [
        {
            "index": {
                "_id": _KEY,
                "status": 503,
                "error": "temporarily unavailable",
            }
        }
    ]

    with pytest.raises(RuntimeError, match="remain buffered"):
        await asyncio.wait_for(
            storage.compare_and_commit_pipeline_attempt(
                _KEY,
                case.row(_TOKEN, "candidate"),
                expected_attempt_token=_TOKEN,
                row_kind=case.row_kind,
            ),
            timeout=1,
        )

    assert client.operations == [("bulk-index", _KEY)]
    assert client.update_calls == []
    assert client.source == original
    assert storage._pending_upserts == {_KEY: buffered_source}
    assert storage._pending_kv_deletes == set()
