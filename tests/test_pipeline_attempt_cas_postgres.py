from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import pytest

from lightrag.artifact_runtime import (
    PipelineAttemptCommitOutcomeUnknownError,
    PipelineAttemptRowKind,
)
from lightrag.kg.postgres_impl import PGDocStatusStorage, PGKVStorage
from lightrag.namespace import NameSpace

pytestmark = pytest.mark.offline

_KEY = "doc-1"
_WORKSPACE = "workspace-a"
_TOKEN = "attempt-1"
_CREATED_AT = "2026-08-03T10:00:00+00:00"
_UPDATED_AT = "2026-08-03T10:05:00+00:00"
_DB_NOW = datetime(2026, 8, 3, 10, 6, tzinfo=timezone.utc).replace(tzinfo=None)


@dataclass(frozen=True)
class _RowCase:
    row_kind: PipelineAttemptRowKind
    storage_class: type
    namespace: str
    table_name: str
    token_sql: str


_ROW_CASES = (
    _RowCase(
        row_kind="full_docs",
        storage_class=PGKVStorage,
        namespace=NameSpace.KV_STORE_FULL_DOCS,
        table_name="LIGHTRAG_DOC_FULL",
        token_sql="#>> '{artifact_binding,claim_token}'",
    ),
    _RowCase(
        row_kind="doc_status",
        storage_class=PGDocStatusStorage,
        namespace=NameSpace.DOC_STATUS,
        table_name="LIGHTRAG_DOC_STATUS",
        token_sql="->> 'pipeline_attempt_token'",
    ),
)


def _binding(token: str, marker: str) -> dict[str, Any]:
    return {
        "claim_token": token,
        "state": marker,
        "workspace": _WORKSPACE,
        "lightrag_doc_id": _KEY,
    }


def _payload(case: _RowCase, token: str, marker: str) -> dict[str, Any]:
    if case.row_kind == "full_docs":
        return {
            "content": f"content-{marker}",
            "file_path": f"{marker}.pdf",
            "sidecar_location": f"lightrag://{marker}",
            "parse_format": "lightrag",
            "content_hash": f"sha256:{marker}",
            "process_options": f"options-{marker}",
            "chunk_options": {
                "chunk_token_size": 1200,
                "marker": marker,
            },
            "parse_engine": f"engine-{marker}",
            "artifact_binding": _binding(token, marker),
            # PG's established full_docs shape stores its own update_time and
            # does not expose this logical compatibility field on reads.
            "update_time": 123,
        }
    return {
        "content_summary": f"summary-{marker}",
        "content_length": len(marker) + 100,
        "chunks_count": len(marker),
        "status": marker,
        "file_path": f"{marker}.pdf",
        "chunks_list": [f"chunk-{marker}-1", f"chunk-{marker}-2"],
        "track_id": f"track-{marker}",
        "metadata": {
            "pipeline_attempt_token": token,
            "marker": marker,
            "nested": {"value": marker},
        },
        "error_msg": None,
        "content_hash": f"sha256:{marker}",
        "created_at": _CREATED_AT,
        "updated_at": _UPDATED_AT,
    }


def _raw_row(case: _RowCase, token: str | None, marker: str) -> dict[str, Any]:
    if case.row_kind == "full_docs":
        binding = {} if token is None else _binding(token, marker)
        return {
            "workspace": _WORKSPACE,
            "id": _KEY,
            "content": f"content-{marker}",
            "doc_name": f"{marker}.pdf",
            "meta": {
                "artifact_binding": binding,
                "legacy_meta_must_survive": marker,
            },
            "sidecar_location": f"lightrag://{marker}",
            "parse_format": "old-format",
            "content_hash": f"sha256:{marker}",
            "process_options": f"old-options-{marker}",
            "chunk_options": {"old": marker},
            "parse_engine": "old-engine",
            "create_time": datetime(2026, 8, 3, 9, 0),
            "update_time": datetime(2026, 8, 3, 9, 1),
        }
    metadata: dict[str, Any] = {"old": marker}
    if token is not None:
        metadata["pipeline_attempt_token"] = token
    return {
        "workspace": _WORKSPACE,
        "id": _KEY,
        "content_summary": f"old-summary-{marker}",
        "content_length": 1,
        "chunks_count": 1,
        "status": "processing",
        "file_path": f"old-{marker}.pdf",
        "chunks_list": ["old-chunk"],
        "track_id": "old-track",
        "metadata": metadata,
        "error_msg": "old-error",
        "content_hash": "old-hash",
        "created_at": datetime(2026, 8, 3, 9, 0),
        "updated_at": datetime(2026, 8, 3, 9, 1),
    }


def _expected_raw_after_cas(
    case: _RowCase,
    payload: dict[str, Any],
    *,
    previous: dict[str, Any],
) -> dict[str, Any]:
    if case.row_kind == "full_docs":
        return {
            "workspace": _WORKSPACE,
            "id": _KEY,
            "content": payload["content"],
            "doc_name": payload["file_path"],
            "meta": {
                "artifact_binding": deepcopy(payload["artifact_binding"]),
                "legacy_meta_must_survive": previous["meta"][
                    "legacy_meta_must_survive"
                ],
            },
            "sidecar_location": payload["sidecar_location"],
            "parse_format": payload["parse_format"],
            "content_hash": payload["content_hash"],
            "process_options": payload["process_options"],
            "chunk_options": deepcopy(payload["chunk_options"]),
            "parse_engine": payload["parse_engine"],
            "create_time": previous["create_time"],
            "update_time": _DB_NOW,
        }
    return {
        "workspace": _WORKSPACE,
        "id": _KEY,
        "content_summary": payload["content_summary"],
        "content_length": payload["content_length"],
        "chunks_count": payload["chunks_count"],
        "status": payload["status"],
        "file_path": payload["file_path"],
        "chunks_list": deepcopy(payload["chunks_list"]),
        "track_id": payload["track_id"],
        "metadata": deepcopy(payload["metadata"]),
        "error_msg": payload["error_msg"],
        "content_hash": payload["content_hash"],
        "created_at": datetime.fromisoformat(payload["created_at"]).replace(
            tzinfo=None
        ),
        "updated_at": datetime.fromisoformat(payload["updated_at"]).replace(
            tzinfo=None
        ),
    }


def _raw_token(case: _RowCase, row: dict[str, Any] | None) -> str | None:
    if row is None:
        return None
    if case.row_kind == "full_docs":
        meta = row.get("meta")
        binding = meta.get("artifact_binding") if isinstance(meta, dict) else None
        return binding.get("claim_token") if isinstance(binding, dict) else None
    metadata = row.get("metadata")
    return (
        metadata.get("pipeline_attempt_token") if isinstance(metadata, dict) else None
    )


class _Transaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        return False


class _FakeConnection:
    def __init__(self, db: _FakeDB) -> None:
        self.db = db

    def transaction(self) -> _Transaction:
        return _Transaction()

    async def fetchrow(self, sql: str, *params: Any) -> dict[str, Any] | None:
        self.db.update_calls.append((sql, deepcopy(params)))
        workspace, key = params[:2]
        assert workspace == _WORKSPACE
        assert key == _KEY
        expected_token = params[-1]
        if (
            self.db.row is None
            or _raw_token(self.db.case, self.db.row) != expected_token
        ):
            return None

        previous = deepcopy(self.db.row)
        if self.db.case.row_kind == "full_docs":
            self.db.row = {
                "workspace": workspace,
                "id": key,
                "content": params[2],
                "doc_name": params[3],
                "meta": {
                    **(
                        previous["meta"]
                        if isinstance(previous.get("meta"), dict)
                        else {}
                    ),
                    **json.loads(params[4]),
                },
                "sidecar_location": params[5],
                "parse_format": params[6],
                "content_hash": params[7],
                "process_options": params[8],
                "chunk_options": json.loads(params[9]),
                "parse_engine": params[10],
                "create_time": previous["create_time"],
                "update_time": _DB_NOW,
            }
        else:
            self.db.row = {
                "workspace": workspace,
                "id": key,
                "content_summary": params[2],
                "content_length": params[3],
                "chunks_count": params[4],
                "status": params[5],
                "file_path": params[6],
                "chunks_list": json.loads(params[7]),
                "track_id": params[8],
                "metadata": json.loads(params[9]),
                "error_msg": params[10],
                "content_hash": params[11],
                "created_at": params[12],
                "updated_at": params[13],
            }

        if self.db.after_update is not None:
            self.db.row = self.db.after_update(deepcopy(self.db.row))
        if self.db.raise_after_update:
            raise self.db.write_error
        return {"id": key}

    async def executemany(self, sql: str, rows: list[tuple[Any, ...]]) -> None:
        self.db.executemany_calls.append((sql, deepcopy(rows)))
        for values in rows:
            if "LIGHTRAG_DOC_FULL" not in sql:
                raise AssertionError("offline fake only supports full_docs upsert")
            current = deepcopy(self.db.row)
            meta = json.loads(values[10]) if values[10] is not None else None
            if current is None:
                current_meta = meta
                create_time = _DB_NOW
            else:
                current_meta = deepcopy(current.get("meta"))
                if meta is not None:
                    current_meta = {**(current_meta or {}), **meta}
                create_time = current["create_time"]
            self.db.row = {
                "workspace": values[3],
                "id": values[0],
                "content": values[1],
                "doc_name": values[2],
                "meta": current_meta,
                "sidecar_location": values[4],
                "parse_format": values[5],
                "content_hash": values[6],
                "process_options": values[7],
                "chunk_options": json.loads(values[8]),
                "parse_engine": values[9],
                "create_time": create_time,
                "update_time": _DB_NOW,
            }


class _FakeDB:
    _transient_exceptions = (ConnectionError,)

    def __init__(self, case: _RowCase, row: dict[str, Any] | None) -> None:
        self.case = case
        self.row = deepcopy(row)
        self.workspace = None
        self.update_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.executemany_calls: list[tuple[str, list[tuple[Any, ...]]]] = []
        self.readback_calls = 0
        self.run_calls = 0
        self.raise_after_update = False
        self.raise_after_release = False
        self.write_error = ConnectionError("PostgreSQL acknowledgement was lost")
        self.release_error = ConnectionError("PostgreSQL pool release failed")
        self.query_error: BaseException | None = None
        self.after_update: Callable[[dict[str, Any]], dict[str, Any]] | None = None

    async def _run_with_retry(self, operation, **kwargs):
        del kwargs
        self.run_calls += 1
        return await operation(_FakeConnection(self))

    async def _run_pipeline_attempt_cas_once(self, operation):
        self.run_calls += 1
        result = await operation(_FakeConnection(self))
        if self.raise_after_release:
            raise self.release_error
        return result

    async def query(
        self,
        sql: str,
        params: list[Any] | None = None,
        multirows: bool = False,
        **kwargs: Any,
    ) -> Any:
        del kwargs
        self.readback_calls += 1
        if self.query_error is not None:
            raise self.query_error
        assert params == [_WORKSPACE, _KEY]
        if self.row is None:
            return [] if multirows else None
        if self.case.row_kind == "full_docs":
            assert "LIGHTRAG_DOC_FULL" in sql
            return {
                "id": self.row["id"],
                "content": self.row["content"],
                "file_path": self.row["doc_name"],
                "meta": json.dumps(self.row.get("meta") or {}),
                "sidecar_location": self.row["sidecar_location"],
                "parse_format": self.row["parse_format"],
                "content_hash": self.row["content_hash"],
                "process_options": self.row["process_options"],
                "chunk_options": json.dumps(self.row["chunk_options"]),
                "parse_engine": self.row["parse_engine"],
            }
        assert "LIGHTRAG_DOC_STATUS" in sql
        raw = deepcopy(self.row)
        raw["chunks_list"] = json.dumps(raw["chunks_list"])
        raw["metadata"] = json.dumps(raw["metadata"])
        return [raw]


def _storage(case: _RowCase, db: _FakeDB) -> Any:
    storage = case.storage_class.__new__(case.storage_class)
    storage.namespace = case.namespace
    storage.workspace = _WORKSPACE
    storage.global_config = {"embedding_batch_num": 10}
    storage.db = db
    storage.__post_init__()
    return storage


@pytest.mark.parametrize("case", _ROW_CASES, ids=lambda case: case.row_kind)
async def test_postgres_pipeline_attempt_cas_is_one_exact_atomic_update(
    case: _RowCase,
) -> None:
    original = _raw_row(case, _TOKEN, "claimed")
    db = _FakeDB(case, original)
    storage = _storage(case, db)
    candidate = _payload(case, _TOKEN, "committed")
    candidate_before = deepcopy(candidate)

    committed = await storage.compare_and_commit_pipeline_attempt(
        _KEY,
        candidate,
        expected_attempt_token=_TOKEN,
        row_kind=case.row_kind,
    )

    assert committed is True
    assert candidate == candidate_before
    assert db.row == _expected_raw_after_cas(
        case,
        candidate_before,
        previous=original,
    )
    assert db.run_calls == 1
    assert db.readback_calls == 0
    assert len(db.update_calls) == 1
    sql, params = db.update_calls[0]
    normalized = " ".join(sql.split())
    assert normalized.startswith(f"UPDATE {case.table_name} SET")
    assert "workspace = $1" in normalized
    assert "id = $2" in normalized
    assert case.token_sql in normalized
    assert "RETURNING id" in normalized
    assert "INSERT" not in normalized
    assert "ON CONFLICT" not in normalized
    assert params[0:2] == (_WORKSPACE, _KEY)
    assert params[-1] == _TOKEN

    durable_after_cas = deepcopy(db.row)
    await storage.index_done_callback()
    assert db.row == durable_after_cas


@pytest.mark.parametrize("case", _ROW_CASES, ids=lambda case: case.row_kind)
@pytest.mark.parametrize("stored_token", ["newer-attempt", None])
async def test_postgres_pipeline_attempt_mismatch_or_missing_token_is_zero_mutation(
    case: _RowCase,
    stored_token: str | None,
) -> None:
    original = _raw_row(case, stored_token, "newer")
    db = _FakeDB(case, original)
    storage = _storage(case, db)

    committed = await storage.compare_and_commit_pipeline_attempt(
        _KEY,
        _payload(case, _TOKEN, "stale-candidate"),
        expected_attempt_token=_TOKEN,
        row_kind=case.row_kind,
    )

    assert committed is False
    assert db.row == original
    assert len(db.update_calls) == 1
    assert db.readback_calls == 0


@pytest.mark.parametrize("case", _ROW_CASES, ids=lambda case: case.row_kind)
async def test_postgres_pipeline_attempt_missing_row_is_false(case: _RowCase) -> None:
    db = _FakeDB(case, None)
    storage = _storage(case, db)

    committed = await storage.compare_and_commit_pipeline_attempt(
        _KEY,
        _payload(case, _TOKEN, "candidate"),
        expected_attempt_token=_TOKEN,
        row_kind=case.row_kind,
    )

    assert committed is False
    assert db.row is None
    assert len(db.update_calls) == 1
    assert db.readback_calls == 0


@pytest.mark.parametrize("case", _ROW_CASES, ids=lambda case: case.row_kind)
@pytest.mark.parametrize("matches", [True, False], ids=["matched", "mismatched"])
async def test_postgres_pipeline_attempt_pool_release_error_does_not_retry_known_result(
    case: _RowCase,
    matches: bool,
) -> None:
    stored_token = _TOKEN if matches else "newer-attempt"
    original = _raw_row(case, stored_token, "before-release-error")
    db = _FakeDB(case, original)
    db.raise_after_release = True
    storage = _storage(case, db)

    committed = await storage.compare_and_commit_pipeline_attempt(
        _KEY,
        _payload(case, _TOKEN, "candidate"),
        expected_attempt_token=_TOKEN,
        row_kind=case.row_kind,
    )

    assert committed is matches
    assert db.run_calls == 1
    assert len(db.update_calls) == 1
    assert db.readback_calls == 0


@pytest.mark.parametrize("case", _ROW_CASES, ids=lambda case: case.row_kind)
@pytest.mark.parametrize(
    ("readback_outcome", "expected_result", "raises_unknown"),
    [
        ("exact", True, False),
        ("different-token", False, False),
        ("same-token-different-payload", None, True),
        ("readback-error", None, True),
    ],
)
async def test_postgres_pipeline_attempt_ambiguous_transport_reconciles_readback(
    case: _RowCase,
    readback_outcome: str,
    expected_result: bool | None,
    raises_unknown: bool,
) -> None:
    db = _FakeDB(case, _raw_row(case, _TOKEN, "claimed"))
    db.raise_after_update = True
    readback_error = ConnectionError("independent read-back failed")

    if readback_outcome == "different-token":

        def different_token(row: dict[str, Any]) -> dict[str, Any]:
            if case.row_kind == "full_docs":
                row["meta"]["artifact_binding"]["claim_token"] = "newer-attempt"
            else:
                row["metadata"]["pipeline_attempt_token"] = "newer-attempt"
            return row

        db.after_update = different_token
    elif readback_outcome == "same-token-different-payload":

        def different_payload(row: dict[str, Any]) -> dict[str, Any]:
            row["content" if case.row_kind == "full_docs" else "status"] = "other"
            return row

        db.after_update = different_payload
    elif readback_outcome == "readback-error":
        db.query_error = readback_error

    storage = _storage(case, db)
    candidate = _payload(case, _TOKEN, "candidate")
    call = storage.compare_and_commit_pipeline_attempt(
        _KEY,
        candidate,
        expected_attempt_token=_TOKEN,
        row_kind=case.row_kind,
    )

    if raises_unknown:
        with pytest.raises(PipelineAttemptCommitOutcomeUnknownError) as exc_info:
            await call
        assert exc_info.value.key == _KEY
        assert exc_info.value.row_kind == case.row_kind
        if readback_outcome == "readback-error":
            assert exc_info.value.__cause__ is readback_error
        else:
            assert exc_info.value.__cause__ is db.write_error
    else:
        assert await call is expected_result

    # The ambiguous write closure is invoked exactly once: reconciliation is a
    # separate read and never retries the mutation.
    assert db.run_calls == 1
    assert len(db.update_calls) == 1
    assert db.readback_calls == 1


async def test_full_docs_upsert_persists_and_restores_artifact_binding() -> None:
    case = _ROW_CASES[0]
    db = _FakeDB(case, None)
    storage = _storage(case, db)
    payload = _payload(case, _TOKEN, "claimed")

    await storage.upsert({_KEY: deepcopy(payload)})

    assert len(db.executemany_calls) == 1
    sql, rows = db.executemany_calls[0]
    assert "meta" in sql
    assert len(rows) == 1
    assert json.loads(rows[0][10]) == {"artifact_binding": payload["artifact_binding"]}
    stored = await storage.get_by_id(_KEY)
    assert stored is not None
    assert stored["artifact_binding"] == payload["artifact_binding"]
    assert stored["content"] == payload["content"]
