"""Server-integration wiring for the durable job worker.

``tests/api/test_job_worker.py`` covers the worker's runtime behavior in
isolation (atomic claim, grace window, ordering, ``_run_loop`` start/stop,
executor-error handling) by constructing ``JobWorker`` directly, and
``test_replace_durable_resume.py`` / ``test_kb_hard_delete.py`` drive the real
executors directly. What was NOT covered: the ``LIGHTRAG_KB_JOB_WORKER`` env
flag actually wiring a worker into the server via ``create_app`` — with the full
set of REAL resumable executors registered — and the absence of one when the
flag is off. This test closes that wiring gap.

The worker's lifespan ``start()`` / ``stop()`` is exercised by
``test_job_worker.py::test_worker_run_loop_consumes_then_stops``; here we only
assert ``create_app`` builds (or omits) ``app.state.job_worker`` per the env
flag, and that it registers every resumable job type with a callable executor.
"""

from __future__ import annotations

import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from lightrag.api.kb_operation_fence import KBWriteAdmissionMiddleware

pytestmark = pytest.mark.offline

# Env that the project's .env may populate at config import time; clear so the
# test is hermetic, then set minimal OpenAI-compatible bindings so create_app's
# binding validation passes without importing optional local providers.
_ENV_TO_ISOLATE = (
    "LLM_BINDING",
    "EMBEDDING_BINDING",
    "LLM_BINDING_HOST",
    "LLM_BINDING_API_KEY",
    "LLM_MODEL",
    "EMBEDDING_BINDING_HOST",
    "EMBEDDING_BINDING_API_KEY",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "RERANK_BINDING",
    "LIGHTRAG_KV_STORAGE",
    "LIGHTRAG_VECTOR_STORAGE",
    "LIGHTRAG_GRAPH_STORAGE",
    "LIGHTRAG_DOC_STATUS_STORAGE",
    "LIGHTRAG_KB_METADATA_BACKEND",
    "LIGHTRAG_OBJECT_STORAGE",
    # Keep the graphiti-backed chat memory off: a leaked
    # LIGHTRAG_CHAT_MEMORY_ENABLED=true from a developer .env would make the
    # lifespan test reach out to the configured Neo4j.
    "LIGHTRAG_CHAT_MEMORY_ENABLED",
    "LIGHTRAG_KB_JOB_WORKER",
    "LIGHTRAG_KB_JOB_WORKER_POLL_SECONDS",
    "LIGHTRAG_KB_JOB_WORKER_GRACE_SECONDS",
    "LIGHTRAG_KB_JOB_RECOVERY_INTERVAL_SECONDS",
    "LIGHTRAG_KB_JOB_RECOVERY_GRACE_SECONDS",
)

# Every resumable job type the server must register an executor for when the
# durable worker is enabled (see lightrag_server.create_app).
_EXPECTED_EXECUTORS = {
    "parse",
    "build_kg",
    "reindex",
    "delete",
    "replace",
    "sync",
    "clear_kb",
    "agent_profile",
}


def _make_app(tmp_path, monkeypatch, *, worker_enabled: bool):
    for var in _ENV_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("LLM_BINDING_HOST", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_BINDING_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("EMBEDDING_BINDING", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING_HOST", "https://api.openai.com/v1")
    monkeypatch.setenv("EMBEDDING_BINDING_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_DIM", "1536")
    monkeypatch.setenv("RERANK_BINDING", "null")
    monkeypatch.setenv("WORKING_DIR", str(tmp_path / "rag_storage"))
    if worker_enabled:
        monkeypatch.setenv("LIGHTRAG_KB_JOB_WORKER", "true")
        monkeypatch.setenv("LIGHTRAG_KB_JOB_WORKER_POLL_SECONDS", "0.05")
        monkeypatch.setenv("LIGHTRAG_KB_JOB_WORKER_GRACE_SECONDS", "0")
        monkeypatch.setenv("LIGHTRAG_KB_JOB_RECOVERY_INTERVAL_SECONDS", "0.25")
        monkeypatch.setenv("LIGHTRAG_KB_JOB_RECOVERY_GRACE_SECONDS", "0")

    from lightrag.api.config import parse_args

    # Keep sys.argv clean for the whole create_app call: create_app (and the
    # config layer it uses) reads sys.argv, so restoring it before create_app
    # would leak pytest's argv into argument parsing.
    original_argv = sys.argv.copy()
    sys.argv = ["lightrag-server"]
    try:
        args = parse_args()
        with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
            fake_rag = MagicMock()
            fake_rag.initialize_storages = AsyncMock()
            fake_rag.check_and_migrate_data = AsyncMock()
            fake_rag.finalize_storages = AsyncMock()
            mock_rag.return_value = fake_rag
            from lightrag.api.lightrag_server import create_app

            return create_app(args)
    finally:
        sys.argv = original_argv


def test_job_worker_wired_with_real_executors_when_enabled(tmp_path, monkeypatch):
    """``LIGHTRAG_KB_JOB_WORKER=true`` makes create_app build a JobWorker and
    register every resumable job type with a real, callable executor."""
    app = _make_app(tmp_path, monkeypatch, worker_enabled=True)

    worker = app.state.job_worker
    assert worker is not None, "worker must be wired when the flag is enabled"
    # All resumable job types are registered (parse/build_kg/reindex/delete/
    # replace/sync/clear_kb/agent_profile) — the durable-resume contract depends on it.
    assert worker.resumable_job_types == _EXPECTED_EXECUTORS
    # Each registered executor is a real callable (built from the KB services),
    # not a placeholder.
    for job_type in _EXPECTED_EXECUTORS:
        executor = worker._executors[job_type]
        assert callable(executor), f"executor for {job_type} is not callable"
    assert worker._recovery_interval == 0.25
    assert worker.recovery_grace_seconds == 0


def test_no_job_worker_when_disabled(tmp_path, monkeypatch):
    """With the flag unset, create_app must NOT wire a worker — behavior stays
    identical to the historical in-process-only path."""
    app = _make_app(tmp_path, monkeypatch, worker_enabled=False)
    assert app.state.job_worker is None


def test_server_wires_pure_asgi_kb_write_admission_middleware(
    tmp_path, monkeypatch
):
    app = _make_app(tmp_path, monkeypatch, worker_enabled=False)
    matches = [
        item for item in app.user_middleware if item.cls is KBWriteAdmissionMiddleware
    ]

    assert len(matches) == 1
    assert matches[0].kwargs["kb_service"] is app.state.kb_service
    assert matches[0].kwargs["metadata_store"] is app.state.metadata_store


def test_job_worker_lifespan_consumes_queued_job_and_stops(tmp_path, monkeypatch):
    """The real FastAPI lifespan starts the server-wired worker, consumes an
    eligible queued job, and stops the loop on shutdown."""
    app = _make_app(tmp_path, monkeypatch, worker_enabled=True)
    worker = app.state.job_worker
    job_service = app.state.job_service
    kb_service = app.state.kb_service
    executed_job_ids: list[str] = []

    async def fake_parse_executor(job):
        executed_job_ids.append(job.id)
        await job_service.transition_job(
            job.kb_id,
            job.id,
            status="succeeded",
            progress=1.0,
            completed_items=1,
            result={"lifespan_e2e": True},
        )

    worker._executors["parse"] = fake_parse_executor

    async def seed_job():
        await kb_service.create(name="Lifespan", kb_id="kb_lifespan")
        return await job_service.create_job(
            "kb_lifespan",
            job_type="parse",
            stage="parsing",
            payload={"lifespan_e2e": True},
        )

    with TestClient(app) as client:
        assert client.portal is not None
        portal = client.portal
        assert worker._task is not None
        assert not worker._task.done()
        assert worker._recovery_task is not None
        assert not worker._recovery_task.done()
        job = portal.call(seed_job)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            current = portal.call(job_service.get_job, "kb_lifespan", job.id)
            if current.status == "succeeded":
                break
            time.sleep(0.05)
        else:
            pytest.fail("worker did not consume queued job during FastAPI lifespan")

        assert current.result == {"lifespan_e2e": True}
        assert executed_job_ids == [job.id]

    assert worker._task is None
    assert worker._recovery_task is None
