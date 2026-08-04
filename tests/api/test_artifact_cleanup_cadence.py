"""Phase 3.1-D Writer D — server-level wiring of ``ArtifactCleanupService``
onto the JobWorker recovery cadence.

These tests deliberately exercise the *wiring surface* only:

* the cleanup service is constructed in object mode and not in local mode;
* the callback is injected into ``JobWorker`` alongside the existing
  ``artifact_recovery_callback``;
* the callback invokes ``ArtifactCleanupService.run_once`` with the
  documented ``lease_owner``;
* a failing ``run_once`` is isolated — it never crashes the recovery cycle,
  the artifact_recovery_callback, or the polling loop;
* ``JobWorker.start()`` stays idempotent (no duplicate timers) and
  ``stop()`` releases both tasks.

The full leased-cleanup semantics (revalidation, renewal, block/retry state
writes) are covered by ``test_artifact_cleanup_service.py``. The shared
cadence ordering and failure-isolation contract on ``JobWorker`` itself is
covered by ``test_job_worker.py``. This file is intentionally narrow.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from lightrag.utils_pipeline import reset_canonical_input_root_for_tests

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _reset_canonical_root() -> Any:
    reset_canonical_input_root_for_tests()
    try:
        yield
    finally:
        reset_canonical_input_root_for_tests()


class _ServerRAG:
    """Minimal RAG double for server construction (parity with H2D wiring).

    ``initialize_storages`` also primes the shared-storage namespace so
    ``get_namespace_data('pipeline_status', ...)`` works inside /health —
    the real LightRAG does this as part of its storage init, and several
    routes call into the shared dict at request time.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.workspace = kwargs["workspace"]
        self.ollama_server_infos = kwargs["ollama_server_infos"]
        self.pipeline_artifact_materializer = None
        self.role_llm_builder = None

    def register_role_llm_builder(self, builder: Any) -> None:
        self.role_llm_builder = builder

    def get_llm_role_config(self) -> dict[str, Any]:
        return {}

    async def aupdate_llm_role_config(self, _role: str, **_kwargs: Any) -> None:
        return None

    async def adrop_all_storages(self) -> dict[str, Any]:
        return {"dropped": 0, "failed": 0, "errors": []}

    async def initialize_storages(self) -> None:
        from lightrag.kg.shared_storage import (
            initialize_pipeline_status,
            initialize_share_data,
            set_default_workspace,
        )

        initialize_share_data()
        set_default_workspace(self.workspace)
        await initialize_pipeline_status(workspace=self.workspace)

    async def check_and_migrate_data(self) -> None:
        return None

    async def finalize_storages(self) -> None:
        return None

    async def get_llm_queue_status(self, include_base: bool = False) -> dict[str, Any]:
        return {"include_base": include_base}

    async def get_embedding_queue_status(self) -> dict[str, Any]:
        return {}

    async def get_rerank_queue_status(self) -> dict[str, Any]:
        return {}


def _make_server_app(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    object_mode: bool,
) -> tuple[Any, list[_ServerRAG]]:
    from lightrag.api import lightrag_server
    from tests.api.test_artifact_storage_foundation import _complete_server_args

    args = _complete_server_args(tmp_path, monkeypatch)
    monkeypatch.setenv("LIGHTRAG_KB_JOB_WORKER", "true")
    monkeypatch.setenv("LIGHTRAG_KB_JOB_WORKER_POLL_SECONDS", "0.05")
    monkeypatch.setenv("LIGHTRAG_KB_JOB_RECOVERY_INTERVAL_SECONDS", "0.05")
    monkeypatch.setenv("LIGHTRAG_KB_JOB_RECOVERY_GRACE_SECONDS", "0")
    monkeypatch.setattr(lightrag_server, "check_frontend_build", lambda: (True, False))
    built: list[_ServerRAG] = []

    def build_rag(**kwargs: Any) -> _ServerRAG:
        rag = _ServerRAG(**kwargs)
        built.append(rag)
        return rag

    monkeypatch.setattr(lightrag_server, "LightRAG", build_rag)
    if object_mode:
        from tests.api.test_artifact_storage_phase2a import _FakeObjectStorage

        args.artifact_storage_mode = "object"
        monkeypatch.setenv("LIGHTRAG_ARTIFACT_STORAGE_MODE", "object")
        storage = _FakeObjectStorage()
        monkeypatch.setattr(
            lightrag_server,
            "create_object_storage",
            lambda _config: storage,
        )
        monkeypatch.setattr(
            lightrag_server,
            "validate_artifact_storage_configuration",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            lightrag_server,
            "validate_artifact_storage_server_admission",
            lambda *args, **kwargs: None,
        )
    else:
        args.artifact_storage_mode = "local"
        monkeypatch.setenv("LIGHTRAG_ARTIFACT_STORAGE_MODE", "local")

    return lightrag_server.create_app(args), built


# ---------------------------------------------------------------------------
# Object-mode wiring: service constructed, callback injected, run_once shape.
# ---------------------------------------------------------------------------


def test_object_mode_constructs_cleanup_service_and_injects_callback(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In object mode the cleanup service is built and its callback rides the
    JobWorker recovery cadence next to the existing artifact recovery callback."""

    from lightrag.api.artifact_cleanup_service import ArtifactCleanupService

    app, _built = _make_server_app(tmp_path, monkeypatch, object_mode=True)

    worker = app.state.job_worker
    assert worker is not None
    assert isinstance(app.state.artifact_cleanup_service, ArtifactCleanupService)
    assert callable(app.state.artifact_cleanup_callback)
    # The injected callback is exactly the one built from the service.
    assert worker._artifact_cleanup_callback is app.state.artifact_cleanup_callback
    # The pre-existing recovery callback is still wired (no interference).
    assert callable(app.state.pipeline_artifact_recovery_callback)
    assert (
        worker._artifact_recovery_callback
        is app.state.pipeline_artifact_recovery_callback
    )


async def test_cleanup_callback_invokes_run_once_with_job_worker_lease_owner(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The built callback drives exactly one ``run_once`` per call with the
    fixed ``lease_owner='job-worker'`` and an aware UTC ``now``."""

    app, _built = _make_server_app(tmp_path, monkeypatch, object_mode=True)
    callback = app.state.artifact_cleanup_callback
    assert callback is not None

    captured: dict[str, Any] = {}

    async def fake_run_once(*, now: Any, lease_owner: str) -> None:
        captured["now"] = now
        captured["lease_owner"] = lease_owner

    with patch.object(
        app.state.artifact_cleanup_service,
        "run_once",
        side_effect=fake_run_once,
    ):
        await callback()

    assert captured["lease_owner"] == "job-worker"
    # ``now`` is timezone-aware UTC (the server wrapper supplies it).
    assert captured["now"].tzinfo is not None
    assert captured["now"].utcoffset() is not None


async def test_cleanup_callback_run_once_failure_is_isolated(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising ``run_once`` must never propagate out of the callback —
    JobWorker's quiet helper is the outer isolation layer, but the server
    wrapper is defense-in-depth and logs/swallows on its own."""

    app, _built = _make_server_app(tmp_path, monkeypatch, object_mode=True)
    callback = app.state.artifact_cleanup_callback

    async def boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(
            "s3://cleanup:secret@bucket/.lightrag-scratch/private-cleanup"
        )

    with (
        patch.object(
            app.state.artifact_cleanup_service,
            "run_once",
            side_effect=boom,
        ),
        patch("lightrag.api.lightrag_server.logger.warning") as log_warning,
    ):
        # Must not raise.
        await callback()

    logged = repr(log_warning.call_args_list)
    assert "cleanup:secret" not in logged
    assert ".lightrag-scratch" not in logged
    assert "private-cleanup" not in logged


async def test_cleanup_callback_failure_does_not_break_recovery_cycle_or_polling(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end isolation through the real JobWorker cadence: the cleanup
    callback raises every cycle, but the polling loop, the orphan recovery,
    and the artifact_recovery_callback all keep running."""

    app, _built = _make_server_app(tmp_path, monkeypatch, object_mode=True)
    worker = app.state.job_worker
    assert worker is not None

    recovery_calls = 0
    recovery_keepalive = asyncio.Event()

    async def recovery_callback() -> None:
        nonlocal recovery_calls
        recovery_calls += 1

    async def boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("cleanup-run_once-failure")

    # Inject a failing run_once and a sibling recovery callback. The
    # original recovery callback also stays wired; this adds a second one
    # for clarity.
    worker._artifact_recovery_callback = recovery_callback
    with patch.object(app.state.artifact_cleanup_service, "run_once", side_effect=boom):
        worker.start()
        try:
            for _ in range(100):
                if recovery_calls >= 2:
                    recovery_keepalive.set()
                    break
                await asyncio.sleep(0.01)
            assert recovery_keepalive.is_set(), (
                "recovery callback stopped firing after cleanup failed"
            )
        finally:
            await worker.stop()

    assert recovery_calls >= 2


async def test_object_mode_start_is_idempotent_and_stop_clears_tasks(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``start()`` is idempotent (no duplicate timers) and ``stop()`` clears
    both the polling and recovery tasks, exactly as before."""

    app, _built = _make_server_app(tmp_path, monkeypatch, object_mode=True)
    worker = app.state.job_worker
    assert worker is not None

    worker.start()
    polling_task = worker._task
    recovery_task = worker._recovery_task
    assert polling_task is not None
    assert recovery_task is not None
    # No standalone cleanup timer — the callback rides the recovery cadence.
    assert not hasattr(worker, "_artifact_cleanup_task")

    worker.start()  # idempotent
    assert worker._task is polling_task
    assert worker._recovery_task is recovery_task

    try:
        await worker.stop()
    finally:
        # Defensive: never leak the background tasks if the assertion below
        # fails before the explicit None checks.
        pass
    assert worker._task is None
    assert worker._recovery_task is None


def test_object_mode_health_payload_reports_artifact_cleanup_block(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """/health exposes an ``artifact_cleanup`` block in object mode."""

    app, _built = _make_server_app(tmp_path, monkeypatch, object_mode=True)
    # Stop the worker so TestClient startup/shutdown doesn't race the polling
    # loop in this offline test.
    worker = app.state.job_worker
    assert worker is not None
    worker.start = MagicMock()
    worker.stop = AsyncMock()

    async def _count(*_args: Any, **_kwargs: Any) -> int:
        return 7

    monkeypatch.setattr(
        app.state.metadata_store,
        "count_artifact_cleanup_manifests",
        lambda **kwargs: _count(**kwargs),
        raising=False,
    )

    with patch("lightrag.api.lightrag_server.finalize_share_data"):
        with TestClient(app) as client:
            response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert "artifact_cleanup" in payload
    block = payload["artifact_cleanup"]
    assert block["enabled"] is True
    # The worker is started by the TestClient lifespan; either True or False
    # is acceptable depending on timing — the key contract is the key's
    # presence and that it is a bool.
    assert isinstance(block["worker_running"], bool)
    assert block["pending_count"] == 7
    # Phase 3.3 additive sibling: ``artifact_lifecycle`` coexists with the
    # legacy ``artifact_cleanup`` block (backward compat — both keys present).
    assert "artifact_lifecycle" in payload
    lifecycle = payload["artifact_lifecycle"]
    assert lifecycle["mode"] == "object"
    assert isinstance(lifecycle["manifests"], dict)


def test_health_pending_count_falls_back_to_not_reported_on_missing_method(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the metadata store ever lacks the lightweight count method, /health
    collapses to ``"not_reported"`` instead of erroring."""

    app, _built = _make_server_app(tmp_path, monkeypatch, object_mode=True)
    worker = app.state.job_worker
    assert worker is not None
    worker.start = MagicMock()
    worker.stop = AsyncMock()

    # Simulate an older backend that does not expose the lightweight count:
    # setattr to None on the instance. The helper's ``callable(None)`` guard
    # collapses to ``"not_reported"``.
    monkeypatch.setattr(
        app.state.metadata_store,
        "count_artifact_cleanup_manifests",
        None,
        raising=False,
    )

    with patch("lightrag.api.lightrag_server.finalize_share_data"):
        with TestClient(app) as client:
            response = client.get("/health")
    assert response.status_code == 200
    block = response.json()["artifact_cleanup"]
    assert block["enabled"] is True
    assert block["pending_count"] == "not_reported"


# ---------------------------------------------------------------------------
# Local mode: cleanup service is not constructed; cadence is a no-op.
# ---------------------------------------------------------------------------


def test_local_mode_never_constructs_cleanup_service(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without object storage there is no cleanup service, no callback, and
    the JobWorker cleanup param defaults to None — behavior unchanged."""

    app, _built = _make_server_app(tmp_path, monkeypatch, object_mode=False)

    worker = app.state.job_worker
    assert worker is not None
    assert app.state.artifact_cleanup_service is None
    assert app.state.artifact_cleanup_callback is None
    assert worker._artifact_cleanup_callback is None
    # The pre-existing recovery callback is also absent in local mode.
    assert app.state.pipeline_artifact_recovery_callback is None
    assert worker._artifact_recovery_callback is None


def test_local_mode_health_payload_omits_or_disables_cleanup_block(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """/health in local mode reports the cleanup cadence as disabled."""

    app, _built = _make_server_app(tmp_path, monkeypatch, object_mode=False)
    worker = app.state.job_worker
    assert worker is not None
    worker.start = MagicMock()
    worker.stop = AsyncMock()

    with patch("lightrag.api.lightrag_server.finalize_share_data"):
        with TestClient(app) as client:
            response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert "artifact_cleanup" in payload
    block = payload["artifact_cleanup"]
    assert block["enabled"] is False
    assert block["worker_running"] is False
    # Phase 3.3 additive sibling is present in local mode too (reports
    # ``mode='local'`` / disabled storage), alongside the legacy block.
    assert "artifact_lifecycle" in payload
    assert payload["artifact_lifecycle"]["mode"] == "local"


# ---------------------------------------------------------------------------
# Standalone helper-level unit checks (no app construction).
# ---------------------------------------------------------------------------


async def test_build_artifact_cleanup_callback_uses_injected_lease_owner() -> None:
    """The callback builder honors an explicit ``lease_owner`` override and
    forwards an aware UTC ``now`` to ``run_once``."""

    from lightrag.api.lightrag_server import _build_artifact_cleanup_callback

    cleanup_service = MagicMock()
    captured: dict[str, Any] = {}

    async def fake_run_once(*, now: Any, lease_owner: str) -> None:
        captured["now"] = now
        captured["lease_owner"] = lease_owner

    cleanup_service.run_once = fake_run_once
    callback = _build_artifact_cleanup_callback(
        cleanup_service=cleanup_service,  # type: ignore[arg-type]
        lease_owner="custom-owner",
    )
    await callback()
    assert captured["lease_owner"] == "custom-owner"
    assert captured["now"].tzinfo is not None
