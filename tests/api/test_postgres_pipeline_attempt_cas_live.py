"""Opt-in live PostgreSQL contract for pipeline-attempt row CAS."""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import asyncpg
import pytest

from lightrag.kg.postgres_impl import (
    PGDocStatusStorage,
    PGKVStorage,
    PostgreSQLDB,
)
from lightrag.namespace import NameSpace

_POSTGRES_TEST_DSN = os.getenv("LIGHTRAG_KB_POSTGRES_TEST_DSN") or os.getenv(
    "POSTGRES_TEST_DSN"
)

pytestmark = [
    pytest.mark.offline,
    pytest.mark.requires_db,
    pytest.mark.skipif(
        not _POSTGRES_TEST_DSN,
        reason=(
            "live PostgreSQL pipeline-attempt CAS test skipped: set "
            "LIGHTRAG_KB_POSTGRES_TEST_DSN or POSTGRES_TEST_DSN to enable"
        ),
    ),
]


@dataclass
class _Backend:
    db: PostgreSQLDB
    full_docs: PGKVStorage
    doc_status: PGDocStatusStorage


def _storage(storage_class: type, namespace: str, workspace: str, db: Any) -> Any:
    storage = storage_class.__new__(storage_class)
    storage.namespace = namespace
    storage.workspace = workspace
    storage.global_config = {"embedding_batch_num": 10}
    storage.db = db
    storage.__post_init__()
    return storage


def _postgres_db(pool: asyncpg.Pool) -> PostgreSQLDB:
    """Bind the real PostgreSQLDB execution methods to an opt-in one-slot pool."""

    db = PostgreSQLDB.__new__(PostgreSQLDB)
    db.pool = pool
    db.workspace = None
    db._pool_reconnect_lock = asyncio.Lock()
    db._transient_exceptions = (
        asyncio.TimeoutError,
        TimeoutError,
        ConnectionError,
        OSError,
        asyncpg.exceptions.InterfaceError,
        asyncpg.exceptions.TooManyConnectionsError,
        asyncpg.exceptions.CannotConnectNowError,
        asyncpg.exceptions.PostgresConnectionError,
        asyncpg.exceptions.ConnectionDoesNotExistError,
        asyncpg.exceptions.ConnectionFailureError,
    )
    db.connection_retry_attempts = 1
    db.connection_retry_backoff = 0.0
    db.connection_retry_backoff_max = 0.0
    db.pool_close_timeout = 5.0
    return db


def _backend(pool: asyncpg.Pool, workspace: str) -> _Backend:
    db = _postgres_db(pool)
    return _Backend(
        db=db,
        full_docs=_storage(
            PGKVStorage,
            NameSpace.KV_STORE_FULL_DOCS,
            workspace,
            db,
        ),
        doc_status=_storage(
            PGDocStatusStorage,
            NameSpace.DOC_STATUS,
            workspace,
            db,
        ),
    )


def _binding(token: str, marker: str) -> dict[str, Any]:
    return {
        "claim_token": token,
        "state": marker,
        "workspace": "live-cas",
        "lightrag_doc_id": marker,
    }


def _full_payload(token: str, marker: str) -> dict[str, Any]:
    return {
        "content": f"content-{marker}",
        "file_path": f"{marker}.pdf",
        "sidecar_location": f"lightrag://{marker}",
        "parse_format": "lightrag",
        "content_hash": f"sha256:{marker}",
        "process_options": f"options-{marker}",
        "chunk_options": {
            "chunk_token_size": 1000 + len(marker),
            "marker": marker,
        },
        "parse_engine": "native",
        "artifact_binding": _binding(token, marker),
        "update_time": 123,
    }


def _status_payload(token: str, marker: str) -> dict[str, Any]:
    return {
        "content_summary": f"summary-{marker}",
        "content_length": 100 + len(marker),
        "chunks_count": len(marker),
        "status": marker,
        "file_path": f"{marker}.pdf",
        "chunks_list": [f"chunk-{marker}-1", f"chunk-{marker}-2"],
        "track_id": f"track-{marker}",
        "metadata": {
            "pipeline_attempt_token": token,
            "marker": marker,
            "nested": {"winner": marker},
        },
        "error_msg": None,
        "content_hash": f"sha256:{marker}",
        "created_at": "2026-08-03T10:00:00+00:00",
        "updated_at": "2026-08-03T10:05:00+00:00",
    }


def _expected_full(key: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": key,
        "content": payload["content"],
        "file_path": payload["file_path"],
        "sidecar_location": payload["sidecar_location"],
        "parse_format": payload["parse_format"],
        "content_hash": payload["content_hash"],
        "process_options": payload["process_options"],
        "chunk_options": payload["chunk_options"],
        "parse_engine": payload["parse_engine"],
        "artifact_binding": payload["artifact_binding"],
    }


def _as_pg_iso(value: str) -> str:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _expected_status(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "content_length": payload["content_length"],
        "content_summary": payload["content_summary"],
        "status": payload["status"],
        "chunks_count": payload["chunks_count"],
        "created_at": _as_pg_iso(payload["created_at"]),
        "updated_at": _as_pg_iso(payload["updated_at"]),
        "file_path": payload["file_path"],
        "chunks_list": payload["chunks_list"],
        "metadata": payload["metadata"],
        "error_msg": payload["error_msg"],
        "track_id": payload["track_id"],
        "content_hash": payload["content_hash"],
    }


async def _backend_pid(backend: _Backend) -> int:
    async with backend.db.pool.acquire() as connection:
        return int(await connection.fetchval("SELECT pg_backend_pid()"))


async def _create_tables(backend: _Backend) -> None:
    async with backend.db.pool.acquire() as connection:
        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS LIGHTRAG_DOC_FULL (
                id VARCHAR(255),
                workspace VARCHAR(255),
                doc_name VARCHAR(1024),
                content TEXT,
                meta JSONB,
                sidecar_location TEXT NULL,
                parse_format VARCHAR(32) NULL DEFAULT 'raw',
                content_hash TEXT NULL,
                process_options TEXT NULL,
                chunk_options JSONB NULL DEFAULT '{}'::jsonb,
                parse_engine VARCHAR(32) NULL,
                create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (workspace, id)
            )
            """
        )
        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS LIGHTRAG_DOC_STATUS (
                workspace VARCHAR(255) NOT NULL,
                id VARCHAR(255) NOT NULL,
                content_summary VARCHAR(255) NULL,
                content_length INT4 NULL,
                chunks_count INT4 NULL,
                status VARCHAR(64) NULL,
                file_path TEXT NULL,
                chunks_list JSONB NULL DEFAULT '[]'::jsonb,
                track_id VARCHAR(255) NULL,
                metadata JSONB NULL DEFAULT '{}'::jsonb,
                error_msg TEXT NULL,
                content_hash TEXT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (workspace, id)
            )
            """
        )


async def _clean_rows(backend: _Backend, workspace: str) -> None:
    async with backend.db.pool.acquire() as connection:
        await connection.execute(
            "DELETE FROM LIGHTRAG_DOC_STATUS WHERE workspace = $1",
            workspace,
        )
        await connection.execute(
            "DELETE FROM LIGHTRAG_DOC_FULL WHERE workspace = $1",
            workspace,
        )


async def _race(
    first_store: Any,
    second_store: Any,
    key: str,
    first_payload: dict[str, Any],
    second_payload: dict[str, Any],
    *,
    token: str,
    row_kind: str,
) -> list[bool]:
    arrivals = 0
    arrival_lock = asyncio.Lock()
    both_arrived = asyncio.Event()

    async def commit(store: Any, payload: dict[str, Any]) -> bool:
        nonlocal arrivals
        async with arrival_lock:
            arrivals += 1
            if arrivals == 2:
                both_arrived.set()
        await asyncio.wait_for(both_arrived.wait(), timeout=10)
        return await store.compare_and_commit_pipeline_attempt(
            key,
            payload,
            expected_attempt_token=token,
            row_kind=row_kind,
        )

    return list(
        await asyncio.wait_for(
            asyncio.gather(
                commit(first_store, first_payload),
                commit(second_store, second_payload),
            ),
            timeout=20,
        )
    )


@pytest.mark.asyncio
async def test_postgres_pipeline_attempt_cas_is_cross_backend_linearizable() -> None:
    assert _POSTGRES_TEST_DSN is not None
    run_id = uuid.uuid4().hex
    workspace = f"ws_pipeline_attempt_cas_{run_id}"
    first_pool = await asyncpg.create_pool(
        dsn=_POSTGRES_TEST_DSN,
        min_size=1,
        max_size=1,
    )
    second_pool: asyncpg.Pool | None = None
    tables_ready = False

    try:
        second_pool = await asyncpg.create_pool(
            dsn=_POSTGRES_TEST_DSN,
            min_size=1,
            max_size=1,
        )
        first = _backend(first_pool, workspace)
        second = _backend(second_pool, workspace)
        await _create_tables(first)
        tables_ready = True
        await _clean_rows(first, workspace)

        first_pid, second_pid = await asyncio.gather(
            _backend_pid(first),
            _backend_pid(second),
        )
        assert first_pid > 0
        assert second_pid > 0
        assert first_pid != second_pid

        row_cases = (
            (
                "full_docs",
                first.full_docs,
                second.full_docs,
                _full_payload,
                _expected_full,
            ),
            (
                "doc_status",
                first.doc_status,
                second.doc_status,
                _status_payload,
                lambda _key, payload: _expected_status(payload),
            ),
        )

        for (
            row_kind,
            first_store,
            second_store,
            payload_factory,
            expected_factory,
        ) in row_cases:
            stale_key = f"{row_kind}_stale_{run_id}"
            old_token = f"old-token-{row_kind}-{run_id}"
            newer_token = f"newer-token-{row_kind}-{run_id}"
            old_row = payload_factory(old_token, f"old-{row_kind}-{run_id}")
            stale_candidate = payload_factory(
                old_token,
                f"stale-candidate-{row_kind}-{run_id}",
            )
            newer_row = payload_factory(
                newer_token,
                f"newer-winner-{row_kind}-{run_id}",
            )
            await first_store.upsert({stale_key: old_row})

            old_logic_paused = asyncio.Event()
            release_old_logic = asyncio.Event()

            async def paused_old_commit() -> bool:
                # The old worker has already captured its expected token and
                # candidate payload, but has not issued the SQL statement yet.
                old_logic_paused.set()
                await asyncio.wait_for(release_old_logic.wait(), timeout=10)
                return await first_store.compare_and_commit_pipeline_attempt(
                    stale_key,
                    stale_candidate,
                    expected_attempt_token=old_token,
                    row_kind=row_kind,
                )

            stale_task = asyncio.create_task(paused_old_commit())
            await asyncio.wait_for(old_logic_paused.wait(), timeout=10)
            await second_store.upsert({stale_key: newer_row})
            release_old_logic.set()
            assert await asyncio.wait_for(stale_task, timeout=20) is False

            expected_newer = expected_factory(stale_key, newer_row)
            assert await first_store.get_by_id(stale_key) == expected_newer
            assert await second_store.get_by_id(stale_key) == expected_newer

            race_key = f"{row_kind}_race_{run_id}"
            shared_token = f"shared-token-{row_kind}-{run_id}"
            claimed = payload_factory(
                shared_token,
                f"claimed-{row_kind}-{run_id}",
            )
            candidates = (
                payload_factory(shared_token, f"alpha-{row_kind}-{run_id}"),
                payload_factory(shared_token, f"beta-{row_kind}-{run_id}"),
            )
            await first_store.upsert({race_key: claimed})

            outcomes = await _race(
                first_store,
                second_store,
                race_key,
                candidates[0],
                candidates[1],
                token=shared_token,
                row_kind=row_kind,
            )
            assert any(outcomes), outcomes

            expected_candidates = [
                expected_factory(race_key, candidate) for candidate in candidates
            ]
            persisted = await first_store.get_by_id(race_key)
            peer_persisted = await second_store.get_by_id(race_key)
            assert peer_persisted == persisted
            assert persisted in expected_candidates

            # The expected token intentionally survives each replacement. Thus
            # both different payloads may linearize in sequence; the durable
            # final winner must nevertheless be one complete candidate, never a
            # torn mixture. If a backend reports only one success, that success
            # must be the durable winner.
            successful_indexes = [
                index for index, outcome in enumerate(outcomes) if outcome is True
            ]
            if len(successful_indexes) == 1:
                assert persisted == expected_candidates[successful_indexes[0]]
            else:
                assert successful_indexes == [0, 1]

            # Re-applying the exact durable winner under the same token is a
            # clearly defined idempotent success and preserves the exact row.
            winner_index = expected_candidates.index(persisted)
            assert await second_store.compare_and_commit_pipeline_attempt(
                race_key,
                candidates[winner_index],
                expected_attempt_token=shared_token,
                row_kind=row_kind,
            )
            assert await first_store.get_by_id(race_key) == persisted
    finally:
        try:
            if tables_ready:
                cleanup_backend = _backend(first_pool, workspace)
                await _clean_rows(cleanup_backend, workspace)
        finally:
            if second_pool is not None:
                await second_pool.close()
            await first_pool.close()
