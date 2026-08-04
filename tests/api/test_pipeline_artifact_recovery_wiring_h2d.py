from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from lightrag.api.job_worker import JobWorker
from lightrag.api.pipeline_artifact_recovery import (
    PipelineArtifactRecoverySummary,
)
from lightrag.utils_pipeline import reset_canonical_input_root_for_tests

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _reset_canonical_root() -> Any:
    reset_canonical_input_root_for_tests()
    try:
        yield
    finally:
        reset_canonical_input_root_for_tests()


class _CallbackKBService:
    def __init__(self, records: list[Any]) -> None:
        self.records = records
        self.include_deleted_values: list[bool] = []

    async def list(self, *, include_deleted: bool = False) -> list[Any]:
        self.include_deleted_values.append(include_deleted)
        return list(self.records)


class _CallbackRegistry:
    def __init__(self) -> None:
        self.kb_ids: list[str] = []
        self.rags: dict[str, object] = {}

    async def get(self, kb_id: str) -> object:
        self.kb_ids.append(kb_id)
        return self.rags.setdefault(kb_id, object())


class _CallbackReconciler:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object, int]] = []

    async def reconcile_kb(
        self,
        kb_id: str,
        rag: object,
        *,
        limit: int,
    ) -> PipelineArtifactRecoverySummary:
        self.calls.append((kb_id, rag, limit))
        if kb_id == "kb_broken":
            raise RuntimeError(
                "s3://access:secret@bucket/.lightrag-scratch/private-artifact"
            )
        return PipelineArtifactRecoverySummary(
            discovered=3,
            finalized=1,
            skipped=2,
        )


async def test_all_kb_callback_continues_after_one_redacted_failure() -> None:
    from lightrag.api.lightrag_server import (
        _build_pipeline_artifact_recovery_callback,
    )

    kb_service = _CallbackKBService(
        [
            SimpleNamespace(id="kb_broken", status="active"),
            SimpleNamespace(id="kb_good", status="active"),
            SimpleNamespace(id="kb_deleting", status="deleting"),
        ]
    )
    registry = _CallbackRegistry()
    reconciler = _CallbackReconciler()
    callback = _build_pipeline_artifact_recovery_callback(
        kb_service=kb_service,
        registry=registry,  # type: ignore[arg-type]
        reconciler=reconciler,  # type: ignore[arg-type]
        document_limit=999,
    )

    with (
        patch("lightrag.api.lightrag_server.logger.info") as log_info,
        patch("lightrag.api.lightrag_server.logger.error") as log_error,
    ):
        await callback()

    assert kb_service.include_deleted_values == [False]
    assert registry.kb_ids == ["kb_broken", "kb_good"]
    assert [(kb_id, limit) for kb_id, _rag, limit in reconciler.calls] == [
        ("kb_broken", 200),
        ("kb_good", 200),
    ]
    info_args = log_info.call_args.args
    assert info_args[0] % info_args[1:] == (
        "Pipeline artifact terminalization reconciliation completed "
        "(discovered=3 finalized=1 skipped=2 error_count=1)"
    )
    logged = repr((log_info.call_args_list, log_error.call_args_list))
    assert "access:secret" not in logged
    assert ".lightrag-scratch" not in logged
    assert "private-artifact" not in logged


class _PeriodicJobService:
    def __init__(self) -> None:
        self.orphan_calls = 0

    async def claim_next_worker_job(self, **_kwargs: Any) -> None:
        return None

    async def recover_orphan_jobs(self, **_kwargs: Any) -> list[Any]:
        self.orphan_calls += 1
        raise RuntimeError("s3://orphan:secret@bucket/.lightrag-scratch/private-orphan")


async def test_periodic_artifact_callback_is_independent_and_uses_existing_task() -> (
    None
):
    job_service = _PeriodicJobService()
    callback_calls = 0
    callback_survived = asyncio.Event()

    async def artifact_recovery() -> None:
        nonlocal callback_calls
        callback_calls += 1
        if callback_calls == 1:
            raise RuntimeError(
                "s3://access:secret@bucket/.lightrag-scratch/private-artifact"
            )
        callback_survived.set()

    worker = JobWorker(
        job_service,  # type: ignore[arg-type]
        executors={},
        poll_interval_seconds=0.05,
        recovery_interval_seconds=0.01,
        artifact_recovery_callback=artifact_recovery,
    )
    with patch("lightrag.api.job_worker.logger.error") as log_error:
        worker.start()
        polling_task = worker._task
        recovery_task = worker._recovery_task
        worker.start()
        try:
            await asyncio.wait_for(callback_survived.wait(), timeout=2.0)
            assert worker._task is polling_task
            assert worker._recovery_task is recovery_task
            assert recovery_task is not None and not recovery_task.done()
            assert not hasattr(worker, "_artifact_recovery_task")
            assert job_service.orphan_calls >= 2
            assert callback_calls >= 2
        finally:
            await worker.stop()

    assert worker._task is None
    assert worker._recovery_task is None
    logged = repr(log_error.call_args_list)
    assert "access:secret" not in logged
    assert ".lightrag-scratch" not in logged
    assert "private-artifact" not in logged
    assert "orphan:secret" not in logged
    assert "private-orphan" not in logged

    legacy_worker = JobWorker(
        job_service,  # type: ignore[arg-type]
        executors={},
        recovery_interval_seconds=0,
    )
    assert legacy_worker._artifact_recovery_callback is None


class _ServerRAG:
    def __init__(self, **kwargs: Any) -> None:
        self.workspace = kwargs["workspace"]
        self.ollama_server_infos = kwargs["ollama_server_infos"]
        self.pipeline_artifact_materializer = None
        self.role_llm_builder = None
        self.initialize_storages = AsyncMock()
        self.check_and_migrate_data = AsyncMock()
        self.finalize_storages = AsyncMock()

    def register_role_llm_builder(self, builder: Any) -> None:
        self.role_llm_builder = builder

    def get_llm_role_config(self) -> dict[str, Any]:
        return {}

    async def aupdate_llm_role_config(self, _role: str, **_kwargs: Any) -> None:
        return None

    async def adrop_all_storages(self) -> dict[str, Any]:
        return {"dropped": 0, "failed": 0, "errors": []}


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


def test_object_startup_orders_recovery_before_worker_and_stays_fail_soft(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, built = _make_server_app(tmp_path, monkeypatch, object_mode=True)
    worker = app.state.job_worker
    assert worker is not None
    assert app.state.pipeline_artifact_reconciler is not None
    assert callable(app.state.pipeline_artifact_recovery_callback)
    assert (
        worker._artifact_recovery_callback
        is app.state.pipeline_artifact_recovery_callback
    )

    order: list[str] = []

    async def recover_orphans(**_kwargs: Any) -> list[Any]:
        order.append("orphan_recovery")
        return []

    async def initialize_rag() -> None:
        order.append("rag_initialize")

    async def migrate_rag() -> None:
        order.append("rag_migrate")

    async def artifact_recovery() -> None:
        order.append("artifact_recovery")
        raise RuntimeError(
            "s3://access:secret@bucket/.lightrag-scratch/private-artifact"
        )

    app.state.job_service.recover_orphan_jobs = recover_orphans
    built[0].initialize_storages = AsyncMock(side_effect=initialize_rag)
    built[0].check_and_migrate_data = AsyncMock(side_effect=migrate_rag)
    app.state.pipeline_artifact_recovery_callback = artifact_recovery
    worker.start = MagicMock(side_effect=lambda: order.append("worker_start"))
    worker.stop = AsyncMock()
    with (
        patch("lightrag.api.lightrag_server.finalize_share_data"),
        patch("lightrag.api.lightrag_server.logger.error") as log_error,
    ):
        with TestClient(app):
            pass
    logged = repr(log_error.call_args_list)

    assert order == [
        "orphan_recovery",
        "rag_initialize",
        "rag_migrate",
        "artifact_recovery",
        "worker_start",
    ]
    worker.stop.assert_awaited_once_with()
    assert "access:secret" not in logged
    assert ".lightrag-scratch" not in logged
    assert "private-artifact" not in logged


def test_local_mode_has_no_reconciler_callback_or_kb_scan(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _built = _make_server_app(tmp_path, monkeypatch, object_mode=False)
    worker = app.state.job_worker
    assert worker is not None
    assert app.state.pipeline_artifact_reconciler is None
    assert app.state.pipeline_artifact_recovery_callback is None
    assert worker._artifact_recovery_callback is None

    app.state.kb_service.list = AsyncMock(
        side_effect=AssertionError("local startup must not enumerate KBs")
    )
    app.state.lightrag_registry.get = AsyncMock(
        side_effect=AssertionError("local startup must not initialize KB RAGs")
    )
    app.state.pipeline_artifact_recovery_callback = AsyncMock(
        side_effect=AssertionError("local startup must not invoke artifact recovery")
    )
    app.state.job_service.recover_orphan_jobs = AsyncMock(return_value=[])
    worker.start = MagicMock()
    worker.stop = AsyncMock()

    with patch("lightrag.api.lightrag_server.finalize_share_data"):
        with TestClient(app):
            pass

    app.state.kb_service.list.assert_not_awaited()
    app.state.lightrag_registry.get.assert_not_awaited()
    app.state.pipeline_artifact_recovery_callback.assert_not_awaited()
    worker.start.assert_called_once_with()
    worker.stop.assert_awaited_once_with()
