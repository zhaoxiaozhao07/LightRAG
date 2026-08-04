from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("pymongo", reason="pymongo is required for Mongo storage tests")

from pymongo.errors import AutoReconnect

from lightrag.artifact_runtime import PipelineAttemptCommitOutcomeUnknownError
from lightrag.kg.mongo_impl import MongoDocStatusStorage, MongoKVStorage

pytestmark = pytest.mark.offline

_KEY = "doc-1"
_TOKEN = "attempt-1"


def _nested_value(row: dict[str, Any], dotted_path: str) -> Any:
    value: Any = row
    for part in dotted_path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


class _FakeMongoCollection:
    def __init__(self, row: dict[str, Any] | None) -> None:
        self.row = deepcopy(row)
        self.replace_calls: list[tuple[dict[str, Any], dict[str, Any], bool]] = []
        self.find_calls: list[dict[str, Any]] = []

    async def replace_one(
        self,
        atomic_filter: dict[str, Any],
        replacement: dict[str, Any],
        *,
        upsert: bool,
    ) -> SimpleNamespace:
        self.replace_calls.append(
            (deepcopy(atomic_filter), deepcopy(replacement), upsert)
        )
        matched = self.row is not None and all(
            (
                self.row.get(field) == expected
                if "." not in field
                else _nested_value(self.row, field) == expected
            )
            for field, expected in atomic_filter.items()
        )
        if matched:
            self.row = deepcopy(replacement)
        return SimpleNamespace(matched_count=int(matched))

    async def find_one(self, atomic_filter: dict[str, Any]) -> dict[str, Any] | None:
        self.find_calls.append(deepcopy(atomic_filter))
        if self.row is None or self.row.get("_id") != atomic_filter.get("_id"):
            return None
        return deepcopy(self.row)


def _storage(storage_cls: type, namespace: str, collection: Any) -> Any:
    storage = storage_cls.__new__(storage_cls)
    storage.workspace = "workspace-a"
    storage.namespace = namespace
    storage.final_namespace = f"{storage.workspace}_{namespace}"
    storage._collection_name = storage.final_namespace
    storage._data = collection
    return storage


def _binding(*, token: str = _TOKEN, state: str = "claimed") -> dict[str, Any]:
    return {
        "workspace": "workspace-a",
        "document_id": "document-1",
        "lightrag_doc_id": _KEY,
        "claim_token": token,
        "state": state,
    }


@pytest.mark.parametrize(
    ("storage_cls", "namespace", "row_kind", "token_path", "original", "payload"),
    [
        (
            MongoKVStorage,
            "kv_store_full_docs",
            "full_docs",
            "artifact_binding.claim_token",
            {
                "_id": _KEY,
                "content": "old",
                "artifact_binding": _binding(),
                "create_time": 10,
            },
            {
                "content": "committed",
                "artifact_binding": _binding(state="committed"),
                "create_time": 10,
                "update_time": 20,
            },
        ),
        (
            MongoDocStatusStorage,
            "doc_status",
            "doc_status",
            "metadata.pipeline_attempt_token",
            {
                "_id": _KEY,
                "status": "processing",
                "metadata": {
                    "pipeline_attempt_token": _TOKEN,
                    "artifact_binding": _binding(),
                },
            },
            {
                "status": "processed",
                "metadata": {
                    "pipeline_attempt_token": _TOKEN,
                    "artifact_binding": _binding(state="committed"),
                },
                "chunks_list": ["chunk-1"],
            },
        ),
    ],
)
async def test_atomic_pipeline_attempt_commit_uses_nested_fence_and_exact_identity(
    storage_cls: type,
    namespace: str,
    row_kind: str,
    token_path: str,
    original: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    collection = _FakeMongoCollection(original)
    storage = _storage(storage_cls, namespace, collection)
    payload_before = deepcopy(payload)

    committed = await storage.compare_and_commit_pipeline_attempt(
        _KEY,
        payload,
        expected_attempt_token=_TOKEN,
        row_kind=row_kind,
    )

    assert committed is True
    assert storage._collection_name == f"workspace-a_{namespace}"
    assert collection.replace_calls == [
        (
            {"_id": _KEY, token_path: _TOKEN},
            {**payload_before, "_id": _KEY},
            False,
        )
    ]
    assert collection.find_calls == []
    assert collection.row == {**payload_before, "_id": _KEY}
    assert collection.row["_id"] == _KEY
    assert payload == payload_before

    binding = (
        collection.row["artifact_binding"]
        if row_kind == "full_docs"
        else collection.row["metadata"]["artifact_binding"]
    )
    assert binding["workspace"] == "workspace-a"
    assert binding["lightrag_doc_id"] == _KEY


@pytest.mark.parametrize(
    ("storage_cls", "namespace", "row_kind", "token_path", "stored_token"),
    [
        (
            MongoKVStorage,
            "kv_store_full_docs",
            "full_docs",
            "artifact_binding.claim_token",
            "newer-attempt",
        ),
        (
            MongoKVStorage,
            "kv_store_full_docs",
            "full_docs",
            "artifact_binding.claim_token",
            None,
        ),
        (
            MongoDocStatusStorage,
            "doc_status",
            "doc_status",
            "metadata.pipeline_attempt_token",
            "newer-attempt",
        ),
        (
            MongoDocStatusStorage,
            "doc_status",
            "doc_status",
            "metadata.pipeline_attempt_token",
            None,
        ),
    ],
)
async def test_pipeline_attempt_mismatch_or_missing_token_has_zero_mutation(
    storage_cls: type,
    namespace: str,
    row_kind: str,
    token_path: str,
    stored_token: str | None,
) -> None:
    if row_kind == "full_docs":
        row = {
            "_id": _KEY,
            "content": "newer",
            "artifact_binding": (
                _binding(token=stored_token) if stored_token is not None else {}
            ),
        }
        payload = {
            "content": "stale",
            "artifact_binding": _binding(state="committed"),
        }
    else:
        metadata = {"artifact_binding": _binding(token="newer-attempt")}
        if stored_token is not None:
            metadata["pipeline_attempt_token"] = stored_token
        row = {"_id": _KEY, "status": "processing", "metadata": metadata}
        payload = {
            "status": "processed",
            "metadata": {
                "pipeline_attempt_token": _TOKEN,
                "artifact_binding": _binding(state="committed"),
            },
        }

    collection = _FakeMongoCollection(row)
    storage = _storage(storage_cls, namespace, collection)
    before = deepcopy(collection.row)

    committed = await storage.compare_and_commit_pipeline_attempt(
        _KEY,
        payload,
        expected_attempt_token=_TOKEN,
        row_kind=row_kind,
    )

    assert committed is False
    assert collection.row == before
    assert collection.replace_calls[0][0] == {"_id": _KEY, token_path: _TOKEN}
    assert collection.replace_calls[0][2] is False
    assert collection.find_calls == []


@pytest.mark.parametrize(
    ("storage_cls", "namespace", "row_kind", "token_path", "payload"),
    [
        (
            MongoKVStorage,
            "kv_store_full_docs",
            "full_docs",
            "artifact_binding.claim_token",
            {
                "content": "candidate",
                "artifact_binding": _binding(state="committed"),
            },
        ),
        (
            MongoDocStatusStorage,
            "doc_status",
            "doc_status",
            "metadata.pipeline_attempt_token",
            {
                "status": "processed",
                "metadata": {
                    "pipeline_attempt_token": _TOKEN,
                    "artifact_binding": _binding(state="committed"),
                },
            },
        ),
    ],
)
async def test_pipeline_attempt_missing_row_is_false_without_mutation(
    storage_cls: type,
    namespace: str,
    row_kind: str,
    token_path: str,
    payload: dict[str, Any],
) -> None:
    collection = _FakeMongoCollection(None)
    storage = _storage(storage_cls, namespace, collection)

    committed = await storage.compare_and_commit_pipeline_attempt(
        _KEY,
        payload,
        expected_attempt_token=_TOKEN,
        row_kind=row_kind,
    )

    assert committed is False
    assert collection.row is None
    assert collection.replace_calls[0][0] == {"_id": _KEY, token_path: _TOKEN}
    assert collection.replace_calls[0][2] is False
    assert collection.find_calls == []


@pytest.mark.parametrize("row_kind", ["full_docs", "doc_status"])
@pytest.mark.parametrize(
    ("readback_outcome", "expected_result", "raises_unknown"),
    [
        ("exact", True, False),
        ("different_token", False, False),
        ("same_token_different_payload", None, True),
        ("readback_error", None, True),
    ],
)
async def test_pipeline_attempt_ambiguous_write_reconciles_exact_row(
    row_kind: str,
    readback_outcome: str,
    expected_result: bool | None,
    raises_unknown: bool,
) -> None:
    token_path = (
        "artifact_binding.claim_token"
        if row_kind == "full_docs"
        else "metadata.pipeline_attempt_token"
    )
    namespace = "kv_store_full_docs" if row_kind == "full_docs" else "doc_status"
    storage_cls = MongoKVStorage if row_kind == "full_docs" else MongoDocStatusStorage
    if row_kind == "full_docs":
        payload = {
            "content": "committed",
            "artifact_binding": _binding(state="committed"),
        }
    else:
        payload = {
            "status": "processed",
            "metadata": {
                "pipeline_attempt_token": _TOKEN,
                "artifact_binding": _binding(state="committed"),
            },
        }
    replacement = {**deepcopy(payload), "_id": _KEY}

    if readback_outcome == "exact":
        readback = replacement
    elif readback_outcome == "different_token":
        readback = deepcopy(replacement)
        if row_kind == "full_docs":
            readback["artifact_binding"]["claim_token"] = "newer-attempt"
        else:
            readback["metadata"]["pipeline_attempt_token"] = "newer-attempt"
    else:
        readback = deepcopy(replacement)
        readback["status" if row_kind == "doc_status" else "content"] = "different"

    write_error = AutoReconnect("write acknowledgement was lost")
    read_error = AutoReconnect("read-back failed")

    class _AmbiguousCollection:
        def __init__(self) -> None:
            self.replace_calls: list[tuple[dict[str, Any], dict[str, Any], bool]] = []
            self.find_calls: list[dict[str, Any]] = []

        async def replace_one(
            self,
            atomic_filter: dict[str, Any],
            replacement_value: dict[str, Any],
            *,
            upsert: bool,
        ) -> None:
            self.replace_calls.append(
                (deepcopy(atomic_filter), deepcopy(replacement_value), upsert)
            )
            raise write_error

        async def find_one(self, atomic_filter: dict[str, Any]) -> dict[str, Any]:
            self.find_calls.append(deepcopy(atomic_filter))
            if readback_outcome == "readback_error":
                raise read_error
            return deepcopy(readback)

    collection = _AmbiguousCollection()
    storage = _storage(storage_cls, namespace, collection)

    if raises_unknown:
        with pytest.raises(PipelineAttemptCommitOutcomeUnknownError) as exc_info:
            await storage.compare_and_commit_pipeline_attempt(
                _KEY,
                payload,
                expected_attempt_token=_TOKEN,
                row_kind=row_kind,
            )
        assert exc_info.value.key == _KEY
        assert exc_info.value.row_kind == row_kind
        assert exc_info.value.__cause__ is (
            read_error if readback_outcome == "readback_error" else write_error
        )
    else:
        result = await storage.compare_and_commit_pipeline_attempt(
            _KEY,
            payload,
            expected_attempt_token=_TOKEN,
            row_kind=row_kind,
        )
        assert result is expected_result

    assert collection.replace_calls == [
        ({"_id": _KEY, token_path: _TOKEN}, replacement, False)
    ]
    assert collection.find_calls == [{"_id": _KEY}]
