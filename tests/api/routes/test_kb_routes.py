import asyncio
import importlib
import multiprocessing
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from lightrag.api.job_service import JobService
from lightrag.api.kb_deletion_service import KBDeletionService
from lightrag.api.kb_service import (
    KnowledgeBaseConflictError,
    KnowledgeBaseService,
    sanitize_workspace,
    validate_kb_id,
)
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry, LightRAGLike
from lightrag.api.metadata_store import SQLiteMetadataStore
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_kb_routes = importlib.import_module("lightrag.api.routers.kb_routes")
sys.argv = _original_argv

create_kb_routes = _kb_routes.create_kb_routes

pytestmark = pytest.mark.offline

_API_KEY = "test-key"
_HEADERS = {"X-API-Key": _API_KEY}


class FakeRAG:
    def __init__(self, workspace: str):
        self.workspace = workspace

    async def finalize_storages(self) -> None:
        return None

    async def adrop_all_storages(self) -> dict:
        return {"dropped": 12, "failed": 0, "errors": []}


class BuilderProbe:
    def __init__(self):
        self.calls = 0
        self.finalized: list[str] = []

    async def build(self, record) -> FakeRAG:
        self.calls += 1
        await asyncio.sleep(0.01)
        return FakeRAG(record.workspace)

    async def finalize(self, rag: LightRAGLike) -> None:
        self.finalized.append(rag.workspace)


class FakeJobWorker:
    @property
    def resumable_job_types(self) -> set[str]:
        return {"clear_kb"}


def _build_client(tmp_path: Path):
    service = KnowledgeBaseService(tmp_path / "metadata" / "knowledge_bases.json")
    probe = BuilderProbe()
    registry = LightRAGInstanceRegistry(service, probe.build, probe.finalize)
    app = FastAPI()
    app.include_router(create_kb_routes(service, registry, api_key=_API_KEY))
    return TestClient(app), service, registry, probe


def _build_hard_delete_client(tmp_path: Path):
    service = KnowledgeBaseService(tmp_path / "metadata" / "knowledge_bases.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    job_service = JobService(service, metadata_store)
    probe = BuilderProbe()
    registry = LightRAGInstanceRegistry(service, probe.build, probe.finalize)
    deletion_service = KBDeletionService(
        service,
        metadata_store,
        registry,
        input_root=tmp_path / "inputs",
        working_dir=tmp_path / "working",
    )
    app = FastAPI()
    app.include_router(
        create_kb_routes(
            service,
            registry,
            api_key=_API_KEY,
            job_service=job_service,
            deletion_service=deletion_service,
            metadata_store=metadata_store,
        )
    )
    return TestClient(app), metadata_store, registry, probe


def _build_durable_hard_delete_client(tmp_path: Path):
    service = KnowledgeBaseService(tmp_path / "metadata" / "knowledge_bases.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    job_service = JobService(service, metadata_store)
    probe = BuilderProbe()
    registry = LightRAGInstanceRegistry(service, probe.build, probe.finalize)
    deletion_service = KBDeletionService(
        service,
        metadata_store,
        registry,
        input_root=tmp_path / "inputs",
        working_dir=tmp_path / "working",
    )
    app = FastAPI()
    app.state.job_worker = FakeJobWorker()
    app.include_router(
        create_kb_routes(
            service,
            registry,
            api_key=_API_KEY,
            job_service=job_service,
            deletion_service=deletion_service,
            metadata_store=metadata_store,
        )
    )
    return (
        TestClient(app),
        service,
        metadata_store,
        job_service,
        registry,
        probe,
        deletion_service,
    )


def _create_kb_after_start_event(
    metadata_path: str,
    kb_id: str,
    ready_queue: Any,
    start_event: Any,
) -> None:
    async def run() -> None:
        service = KnowledgeBaseService(metadata_path)
        await service.initialize()
        ready_queue.put(kb_id)
        if not start_event.wait(10):
            raise TimeoutError("Timed out waiting to start metadata write")
        await service.create(kb_id=kb_id, name=kb_id)

    asyncio.run(run())


async def _list_record_ids(metadata_path: Path) -> list[str]:
    service = KnowledgeBaseService(metadata_path)
    records = await service.list()
    return [record.id for record in records]


def _route_endpoint(app: FastAPI, path: str, method: str):
    for route in app.routes:
        methods = getattr(route, "methods", None) or set()
        if getattr(route, "path", None) == path and method in methods:
            return getattr(route, "endpoint")
    raise AssertionError(f"Route not found: {method} {path}")


def _route_request(app: FastAPI, method: str, path: str) -> Request:
    return Request(
        {
            "type": "http",
            "app": app,
            "method": method,
            "path": path,
            "headers": [],
            "query_string": b"",
        }
    )


async def _build_restore_race_app(tmp_path: Path, kb_id: str):
    service = KnowledgeBaseService(tmp_path / "metadata" / "knowledge_bases.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    await service.initialize()
    await metadata_store.initialize()
    job_service = JobService(service, metadata_store)
    probe = BuilderProbe()
    registry = LightRAGInstanceRegistry(service, probe.build, probe.finalize)
    deletion_service = KBDeletionService(
        service,
        metadata_store,
        registry,
        input_root=tmp_path / "inputs",
        working_dir=tmp_path / "working",
    )
    app = FastAPI()
    app.include_router(
        create_kb_routes(
            service,
            registry,
            job_service=job_service,
            deletion_service=deletion_service,
            metadata_store=metadata_store,
        )
    )
    record = await service.create(kb_id=kb_id, name=kb_id)
    await metadata_store.activate_kb_generation(record.id, record.generation)
    return app, service, metadata_store, deletion_service, record


def test_kb_crud_flow(tmp_path):
    client, _service, _registry, _probe = _build_client(tmp_path)

    create_response = client.post(
        "/kbs",
        json={
            "id": "kb_alpha-1",
            "name": " Alpha ",
            "description": "first KB",
            "tenant_id": "tenant-a",
            "visibility": "internal",
        },
        headers=_HEADERS,
    )
    assert create_response.status_code == 200
    created = create_response.json()
    assert created["id"] == "kb_alpha-1"
    assert created["name"] == "Alpha"
    assert created["workspace"] == sanitize_workspace("kb_alpha-1")
    assert created["status"] == "active"
    assert created["tenant_id"] == "tenant-a"
    assert created["visibility"] == "internal"
    assert created["origin"] == "platform"

    duplicate = client.post(
        "/kbs", json={"id": "kb_alpha-1", "name": "Duplicate"}, headers=_HEADERS
    )
    assert duplicate.status_code == 409

    list_response = client.get("/kbs", headers=_HEADERS)
    assert list_response.status_code == 200
    listed = list_response.json()
    assert listed["total"] == 1
    assert listed["knowledge_bases"][0]["id"] == "kb_alpha-1"

    patch_response = client.patch(
        "/kbs/kb_alpha-1",
        json={
            "name": "Renamed",
            "status": "disabled",
            "visibility": "private",
            "origin": "tenant",
        },
        headers=_HEADERS,
    )
    assert patch_response.status_code == 200
    patched = patch_response.json()
    assert patched["name"] == "Renamed"
    assert patched["status"] == "disabled"
    assert patched["visibility"] == "private"
    assert patched["origin"] == "platform"

    get_response = client.get("/kbs/kb_alpha-1", headers=_HEADERS)
    assert get_response.status_code == 200
    assert get_response.json()["name"] == "Renamed"

    delete_response = client.delete("/kbs/kb_alpha-1", headers=_HEADERS)
    assert delete_response.status_code == 200
    assert delete_response.json()["status"] == "deleted"
    assert delete_response.json()["deleted_at"] is not None

    assert client.get("/kbs/kb_alpha-1", headers=_HEADERS).status_code == 404
    include_deleted = client.get("/kbs?include_deleted=true", headers=_HEADERS)
    assert include_deleted.status_code == 200
    assert include_deleted.json()["total"] == 1


def test_kb_id_validation_and_workspace_sanitization():
    assert validate_kb_id("abc_123-XYZ") == "abc_123-XYZ"
    assert sanitize_workspace("abc-123") == "kb_abc_d123"
    assert sanitize_workspace("abc_123-XYZ") == "kb_abc_u123_dXYZ"
    assert sanitize_workspace("a-b") != sanitize_workspace("a_b")

    for unsafe in ("", "../x", "..\\x", "/abs", "x/y", "x y", "-bad"):
        with pytest.raises(ValueError):
            validate_kb_id(unsafe)


@pytest.mark.asyncio
async def test_service_rejects_explicit_empty_kb_id(tmp_path):
    service = KnowledgeBaseService(tmp_path / "metadata" / "knowledge_bases.json")

    with pytest.raises(ValueError):
        await service.create(kb_id="", name="Empty")


def test_workspace_mapping_does_not_collide_for_hyphen_and_underscore(tmp_path):
    client, _service, _registry, _probe = _build_client(tmp_path)

    hyphen_response = client.post(
        "/kbs", json={"id": "a-b", "name": "Hyphen"}, headers=_HEADERS
    )
    underscore_response = client.post(
        "/kbs", json={"id": "a_b", "name": "Underscore"}, headers=_HEADERS
    )

    assert hyphen_response.status_code == 200
    assert underscore_response.status_code == 200
    assert hyphen_response.json()["workspace"] == "kb_a_db"
    assert underscore_response.json()["workspace"] == "kb_a_ub"
    assert hyphen_response.json()["workspace"] != underscore_response.json()["workspace"]


@pytest.mark.asyncio
async def test_metadata_writes_reload_before_write_across_service_instances(tmp_path):
    metadata_path = tmp_path / "metadata" / "knowledge_bases.json"
    first_service = KnowledgeBaseService(metadata_path)
    second_service = KnowledgeBaseService(metadata_path)

    await first_service.initialize()
    await second_service.initialize()
    await first_service.create(kb_id="kb_first", name="First")
    await second_service.create(kb_id="kb_second", name="Second")

    records = await first_service.list()
    assert [record.id for record in records] == ["kb_first", "kb_second"]


def test_metadata_lock_preserves_concurrent_cross_process_writes(tmp_path):
    metadata_path = tmp_path / "metadata" / "knowledge_bases.json"
    context = multiprocessing.get_context("spawn")
    ready_queue = context.Queue()
    start_event = context.Event()
    processes = [
        context.Process(
            target=_create_kb_after_start_event,
            args=(str(metadata_path), kb_id, ready_queue, start_event),
        )
        for kb_id in ("kb_proc_a", "kb_proc_b")
    ]

    for process in processes:
        process.start()
    try:
        ready_ids = {ready_queue.get(timeout=10) for _ in processes}
        assert ready_ids == {"kb_proc_a", "kb_proc_b"}
        start_event.set()
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        start_event.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert sorted(asyncio.run(_list_record_ids(metadata_path))) == [
        "kb_proc_a",
        "kb_proc_b",
    ]


@pytest.mark.asyncio
async def test_deleted_kb_id_cannot_be_recreated(tmp_path):
    service = KnowledgeBaseService(tmp_path / "metadata" / "knowledge_bases.json")

    await service.create(kb_id="kb_deleted", name="Deleted")
    await service.delete("kb_deleted")

    with pytest.raises(KnowledgeBaseConflictError) as exc_info:
        await service.create(kb_id="kb_deleted", name="Recreated")
    assert "already exists" in str(exc_info.value)


def test_patch_preserves_omitted_fields_and_clears_explicit_null(tmp_path):
    client, _service, _registry, _probe = _build_client(tmp_path)

    create_response = client.post(
        "/kbs",
        json={
            "id": "kb_patch",
            "name": "Patch",
            "description": "details",
            "owner_id": "owner-a",
            "tenant_id": "tenant-a",
        },
        headers=_HEADERS,
    )
    assert create_response.status_code == 200

    omit_response = client.patch(
        "/kbs/kb_patch", json={"name": "Renamed"}, headers=_HEADERS
    )
    assert omit_response.status_code == 200
    omitted = omit_response.json()
    assert omitted["name"] == "Renamed"
    assert omitted["description"] == "details"
    assert omitted["owner_id"] == "owner-a"
    assert omitted["tenant_id"] == "tenant-a"

    config_response = client.patch(
        "/kbs/kb_patch",
        json={"active_config_version_id": "cfg_1"},
        headers=_HEADERS,
    )
    assert config_response.status_code == 400
    assert "configs/{version_id}:activate" in config_response.json()["detail"]

    clear_response = client.patch(
        "/kbs/kb_patch",
        json={
            "description": None,
            "owner_id": None,
            "tenant_id": None,
        },
        headers=_HEADERS,
    )
    assert clear_response.status_code == 200
    cleared = clear_response.json()
    assert cleared["description"] is None
    assert cleared["owner_id"] is None
    assert cleared["tenant_id"] is None
    assert cleared["active_config_version_id"] is None

    invalid_response = client.patch(
        "/kbs/kb_patch", json={"status": None}, headers=_HEADERS
    )
    assert invalid_response.status_code == 400


@pytest.mark.asyncio
async def test_registry_single_flight_initialization(tmp_path):
    service = KnowledgeBaseService(tmp_path / "metadata" / "knowledge_bases.json")
    record = await service.create(kb_id="kb_parallel", name="Parallel")
    probe = BuilderProbe()
    registry = LightRAGInstanceRegistry(service, probe.build, probe.finalize)

    instances = await asyncio.gather(
        *(registry.get(record.id) for _ in range(8))
    )

    assert probe.calls == 1
    assert len({id(instance) for instance in instances}) == 1
    assert registry.loaded_workspaces() == {
        "kb_parallel": sanitize_workspace("kb_parallel")
    }

    assert await registry.discard(record.id) is True
    assert await registry.discard(record.id) is False
    assert probe.finalized == [sanitize_workspace("kb_parallel")]


@pytest.mark.asyncio
async def test_registry_shutdown_finalizes_each_loaded_instance(tmp_path):
    service = KnowledgeBaseService(tmp_path / "metadata" / "knowledge_bases.json")
    await service.create(kb_id="kb_a", name="A")
    await service.create(kb_id="kb_b", name="B")
    probe = BuilderProbe()
    registry = LightRAGInstanceRegistry(service, probe.build, probe.finalize)

    await registry.get("kb_a")
    await registry.get("kb_b")
    await registry.shutdown()

    assert sorted(probe.finalized) == [
        sanitize_workspace("kb_a"),
        sanitize_workspace("kb_b"),
    ]
    assert registry.loaded_workspaces() == {}


def test_status_handles_uninitialized_pipeline(tmp_path):
    initialize_share_data()
    try:
        client, _service, registry, _probe = _build_client(tmp_path)
        create_response = client.post(
            "/kbs", json={"id": "kb_status", "name": "Status"}, headers=_HEADERS
        )
        assert create_response.status_code == 200

        status_response = client.get("/kbs/kb_status/status", headers=_HEADERS)
        assert status_response.status_code == 200
        status = status_response.json()
        assert status["kb"]["workspace"] == sanitize_workspace("kb_status")
        assert status["instance_loaded"] is False
        assert status["pipeline_initialized"] is False
        assert status["pipeline_status"] == {}
        assert status["running_jobs"] == []
        assert registry.loaded_workspaces() == {}
    finally:
        finalize_share_data()


def test_hard_delete_without_service_returns_503_after_soft_delete(tmp_path):
    client, _service, _registry, _probe = _build_client(tmp_path)
    create_response = client.post(
        "/kbs", json={"id": "kb_hard_missing", "name": "Hard"}, headers=_HEADERS
    )
    assert create_response.status_code == 200

    delete_response = client.delete(
        "/kbs/kb_hard_missing?hard=true", headers=_HEADERS
    )

    assert delete_response.status_code == 503
    assert delete_response.json()["detail"] == "KB hard-delete service is not configured"
    include_deleted = client.get("/kbs?include_deleted=true", headers=_HEADERS)
    assert include_deleted.status_code == 200
    record = include_deleted.json()["knowledge_bases"][0]
    assert record["id"] == "kb_hard_missing"
    assert record["status"] == "deleted"
    assert record["deleted_at"] is not None


def test_hard_delete_route_purges_control_plane_and_files(tmp_path):
    client, metadata_store, registry, probe = _build_hard_delete_client(tmp_path)
    create_response = client.post(
        "/kbs", json={"id": "kb_hard_route", "name": "Hard Route"}, headers=_HEADERS
    )
    assert create_response.status_code == 200
    workspace = create_response.json()["workspace"]
    input_workspace = tmp_path / "inputs" / workspace
    input_workspace.mkdir(parents=True)
    (input_workspace / "source.txt").write_text("raw", encoding="utf-8")
    working_workspace = tmp_path / "working" / workspace
    working_workspace.mkdir(parents=True)
    (working_workspace / "graph.json").write_text("{}", encoding="utf-8")

    rag = asyncio.run(registry.get("kb_hard_route"))
    assert registry.is_loaded("kb_hard_route")
    assert isinstance(rag, FakeRAG)

    delete_response = client.delete(
        "/kbs/kb_hard_route?hard=true", headers=_HEADERS
    )

    assert delete_response.status_code == 200, delete_response.text
    payload = delete_response.json()
    assert payload["status"] == "deleted"
    assert payload["deleted_at"] is not None
    assert not registry.is_loaded("kb_hard_route")
    # force_evict finalizes the cached instance; drop_kb_data builds a transient
    # instance to drop engine storages (reaching external backends) and finalizes
    # it too — both for this workspace.
    assert set(probe.finalized) == {workspace}
    assert len(probe.finalized) == 2
    assert not input_workspace.exists()
    assert not working_workspace.exists()

    docs, total_docs = asyncio.run(metadata_store.list_documents("kb_hard_route"))
    assert docs == []
    assert total_docs == 0
    jobs, total_jobs = asyncio.run(metadata_store.list_jobs("kb_hard_route"))
    assert total_jobs == 1
    assert jobs[0].job_type == "clear_kb"
    assert jobs[0].status == "succeeded"
    result = jobs[0].result or {}
    assert result["cleared_input_dir"] is True
    assert result["finalized_storages"] is True
    assert result["dropped_storages"] == 12


def test_hard_delete_route_enqueues_clear_kb_when_worker_enabled(tmp_path):
    (
        client,
        _service,
        metadata_store,
        job_service,
        registry,
        probe,
        _deletion_service,
    ) = _build_durable_hard_delete_client(tmp_path)
    create_response = client.post(
        "/kbs", json={"id": "kb_hard_queued", "name": "Queued Hard"}, headers=_HEADERS
    )
    assert create_response.status_code == 200
    workspace = create_response.json()["workspace"]
    input_workspace = tmp_path / "inputs" / workspace
    input_workspace.mkdir(parents=True)
    (input_workspace / "source.txt").write_text("raw", encoding="utf-8")
    working_workspace = tmp_path / "working" / workspace
    working_workspace.mkdir(parents=True)
    (working_workspace / "graph.json").write_text("{}", encoding="utf-8")

    asyncio.run(registry.get("kb_hard_queued"))
    assert registry.is_loaded("kb_hard_queued")

    delete_response = client.delete(
        "/kbs/kb_hard_queued?hard=true", headers=_HEADERS
    )

    assert delete_response.status_code == 200, delete_response.text
    payload = delete_response.json()
    assert payload["status"] == "deleted"
    assert payload["deleted_at"] is not None
    assert payload["hard_delete_queued"] is True
    assert payload["hard_delete_job_type"] == "clear_kb"
    assert payload["hard_delete_job_status"] == "queued"
    assert payload["hard_delete_job_id"].startswith("job_clear_kb")
    # Queuing is control-plane only. The worker performs force_evict under the
    # cross-process deletion guard and local destructive lock.
    assert registry.is_loaded("kb_hard_queued")
    assert probe.finalized == []
    assert input_workspace.exists()
    assert working_workspace.exists()

    jobs, total_jobs = asyncio.run(metadata_store.list_jobs("kb_hard_queued"))
    assert total_jobs == 1
    assert jobs[0].id == payload["hard_delete_job_id"]
    assert jobs[0].job_type == "clear_kb"
    assert jobs[0].status == "queued"
    assert jobs[0].payload["kb_generation"]
    assert jobs[0].payload["workspace"] == workspace
    assert jobs[0].idempotency_key == (
        f"clear_kb:kb_hard_queued:{jobs[0].payload['kb_generation']}"
    )
    fetched = asyncio.run(
        job_service.get_job(
            "kb_hard_queued", payload["hard_delete_job_id"], include_deleted=True
        )
    )
    assert fetched.id == jobs[0].id

    duplicate = client.delete(
        "/kbs/kb_hard_queued?hard=true", headers=_HEADERS
    )
    assert duplicate.status_code == 200, duplicate.text
    assert duplicate.json()["hard_delete_job_id"] == payload["hard_delete_job_id"]
    jobs, total_jobs = asyncio.run(metadata_store.list_jobs("kb_hard_queued"))
    assert total_jobs == 1
    assert jobs[0].id == payload["hard_delete_job_id"]


def test_worker_hard_delete_route_requeues_same_failed_clear_job(
    tmp_path, monkeypatch
):
    (
        client,
        service,
        metadata_store,
        _job_service,
        _registry,
        _probe,
        deletion_service,
    ) = _build_durable_hard_delete_client(tmp_path)
    created = client.post(
        "/kbs",
        json={"id": "kb_hard_requeue", "name": "Hard Requeue"},
        headers=_HEADERS,
    )
    assert created.status_code == 200, created.text
    generation = asyncio.run(
        service.get("kb_hard_requeue", include_deleted=True)
    ).generation

    first_delete = client.delete(
        "/kbs/kb_hard_requeue?hard=true",
        headers=_HEADERS,
    )
    assert first_delete.status_code == 200, first_delete.text
    job_id = first_delete.json()["hard_delete_job_id"]
    queued = asyncio.run(metadata_store.get_job("kb_hard_requeue", job_id))
    original_key = queued.idempotency_key
    running = asyncio.run(
        metadata_store.transition_job(
            "kb_hard_requeue",
            job_id,
            status="running",
        )
    )

    async def fail_catalog_purge(*_args, **_kwargs):
        raise RuntimeError("catalog temporarily unavailable")

    monkeypatch.setattr(service, "purge", fail_catalog_purge)
    failed = asyncio.run(deletion_service.resume_hard_delete(running))
    assert failed.job.status == "failed"
    lifecycle = asyncio.run(metadata_store.get_kb_lifecycle("kb_hard_requeue"))
    assert lifecycle is not None
    assert lifecycle.state == "deleted"
    assert lifecycle.generation == generation
    assert lifecycle.delete_job_id == job_id
    assert (
        asyncio.run(service.get("kb_hard_requeue", include_deleted=True)).status
        == "deleted"
    )

    duplicate = client.delete(
        "/kbs/kb_hard_requeue?hard=true",
        headers=_HEADERS,
    )

    assert duplicate.status_code == 200, duplicate.text
    payload = duplicate.json()
    assert payload["hard_delete_queued"] is True
    assert payload["hard_delete_job_id"] == job_id
    assert payload["hard_delete_job_status"] == "queued"
    retried = asyncio.run(metadata_store.get_job("kb_hard_requeue", job_id))
    assert retried.status == "queued"
    assert retried.retry_count == 1
    assert retried.idempotency_key == original_key
    jobs, total = asyncio.run(metadata_store.list_jobs("kb_hard_requeue"))
    assert total == 1
    assert [job.id for job in jobs] == [job_id]


def test_hard_delete_route_threads_authorized_generation_to_service(
    tmp_path, monkeypatch
):
    original_hard_delete = KBDeletionService.hard_delete
    original_enqueue = KBDeletionService.enqueue_hard_delete
    captured: list[tuple[str, str]] = []

    async def hard_delete_with_generation(
        service, kb_id: str, *, expected_generation: str
    ):
        record = await service._kb_service.get(kb_id, include_deleted=True)
        assert record.generation == expected_generation
        captured.append(("hard", expected_generation))
        return await original_hard_delete(
            service,
            kb_id,
            expected_generation=expected_generation,
        )

    async def enqueue_with_generation(
        service, kb_id: str, *, expected_generation: str
    ):
        record = await service._kb_service.get(kb_id, include_deleted=True)
        assert record.generation == expected_generation
        captured.append(("queued", expected_generation))
        return await original_enqueue(
            service,
            kb_id,
            expected_generation=expected_generation,
        )

    monkeypatch.setattr(KBDeletionService, "hard_delete", hard_delete_with_generation)
    monkeypatch.setattr(KBDeletionService, "enqueue_hard_delete", enqueue_with_generation)

    sync_client, _store, _registry, _probe = _build_hard_delete_client(
        tmp_path / "sync"
    )
    sync_created = sync_client.post(
        "/kbs",
        json={"id": "kb_generation_sync", "name": "Generation Sync"},
        headers=_HEADERS,
    )
    assert sync_created.status_code == 200, sync_created.text
    sync_deleted = sync_client.delete(
        "/kbs/kb_generation_sync?hard=true", headers=_HEADERS
    )
    assert sync_deleted.status_code == 200, sync_deleted.text

    queued_client, _service, _store, _jobs, _registry, _probe, _deletion = (
        _build_durable_hard_delete_client(tmp_path / "queued")
    )
    queued_created = queued_client.post(
        "/kbs",
        json={"id": "kb_generation_queued", "name": "Generation Queued"},
        headers=_HEADERS,
    )
    assert queued_created.status_code == 200, queued_created.text
    queued_deleted = queued_client.delete(
        "/kbs/kb_generation_queued?hard=true", headers=_HEADERS
    )
    assert queued_deleted.status_code == 200, queued_deleted.text

    assert [kind for kind, _generation in captured] == ["hard", "queued"]
    assert all(generation for _kind, generation in captured)


def test_missing_kb_returns_404(tmp_path):
    client, _service, _registry, _probe = _build_client(tmp_path)
    assert client.get("/kbs/missing", headers=_HEADERS).status_code == 404
    assert client.get("/kbs/missing/status", headers=_HEADERS).status_code == 404
    assert client.delete("/kbs/missing", headers=_HEADERS).status_code == 404


def _build_client_with_jobs(tmp_path: Path):
    service = KnowledgeBaseService(tmp_path / "metadata" / "knowledge_bases.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    job_service = JobService(service, metadata_store)
    probe = BuilderProbe()
    registry = LightRAGInstanceRegistry(service, probe.build, probe.finalize)
    app = FastAPI()
    app.include_router(
        create_kb_routes(service, registry, api_key=_API_KEY, job_service=job_service)
    )
    return TestClient(app), service, job_service


def test_status_reports_running_jobs_from_job_service(tmp_path):
    """The /status endpoint surfaces queued/running jobs read from the SQLite
    JobService (the populated path, not just the empty default)."""
    initialize_share_data()
    try:
        client, _service, job_service = _build_client_with_jobs(tmp_path)
        create = client.post(
            "/kbs", json={"id": "kb_running", "name": "Running"}, headers=_HEADERS
        )
        assert create.status_code == 200

        queued = asyncio.run(
            job_service.create_job("kb_running", job_type="parse", stage="parsing")
        )

        status_response = client.get("/kbs/kb_running/status", headers=_HEADERS)
        assert status_response.status_code == 200
        running_jobs = status_response.json()["running_jobs"]
        assert len(running_jobs) == 1
        assert running_jobs[0]["id"] == queued.id
        assert running_jobs[0]["status"] == "queued"
        assert running_jobs[0]["job_type"] == "parse"
    finally:
        finalize_share_data()


def test_patch_status_deleted_is_rejected(tmp_path):
    """PATCH must not allow setting status directly to 'deleted' — the
    mutable-status enum excludes it, so the request is rejected (422)."""
    client, _service, _registry, _probe = _build_client(tmp_path)
    create = client.post(
        "/kbs", json={"id": "kb_no_delete_status", "name": "NoDelete"}, headers=_HEADERS
    )
    assert create.status_code == 200

    response = client.patch(
        "/kbs/kb_no_delete_status", json={"status": "deleted"}, headers=_HEADERS
    )
    assert response.status_code == 422

    # The KB remains active and retrievable (not soft-deleted via PATCH).
    detail = client.get("/kbs/kb_no_delete_status", headers=_HEADERS)
    assert detail.status_code == 200
    assert detail.json()["status"] != "deleted"


def test_kb_restore_roundtrip(tmp_path):
    client, _service, _registry, _probe = _build_client(tmp_path)
    created = client.post(
        "/kbs", json={"id": "kb_res", "name": "Restorable"}, headers=_HEADERS
    )
    assert created.status_code == 200, created.text

    deleted = client.delete("/kbs/kb_res", headers=_HEADERS)
    assert deleted.status_code == 200
    assert client.get("/kbs/kb_res", headers=_HEADERS).status_code == 404

    restored = client.post("/kbs/kb_res:restore", headers=_HEADERS)
    assert restored.status_code == 200, restored.text
    body = restored.json()
    assert body["status"] == "active"
    assert body["deleted_at"] is None
    assert client.get("/kbs/kb_res", headers=_HEADERS).status_code == 200

    # Restoring a live KB conflicts; restoring an unknown id is 404.
    assert client.post("/kbs/kb_res:restore", headers=_HEADERS).status_code == 409
    assert client.post("/kbs/kb_ghost:restore", headers=_HEADERS).status_code == 404


@pytest.mark.asyncio
async def test_restore_shared_guard_makes_hard_delete_wait_without_destroying(
    tmp_path, monkeypatch
):
    kb_id = "kb_restore_wins"
    app, service, metadata_store, deletion_service, record = (
        await _build_restore_race_app(tmp_path, kb_id)
    )
    await service.delete(kb_id, expected_generation=record.generation)
    restore_endpoint = _route_endpoint(app, "/kbs/{kb_id}:restore", "POST")
    delete_endpoint = _route_endpoint(app, "/kbs/{kb_id}", "DELETE")

    restore_entered = asyncio.Event()
    release_restore = asyncio.Event()
    deletion_waiting = asyncio.Event()
    deletion_entered = asyncio.Event()
    cleanup_called = asyncio.Event()
    original_restore = service.restore
    original_deletion_guard = metadata_store.kb_deletion_guard

    async def blocked_restore(kb_id: str, *, expected_generation: str | None = None):
        restore_entered.set()
        await asyncio.wait_for(release_restore.wait(), timeout=5)
        return await original_restore(
            kb_id,
            expected_generation=expected_generation,
        )

    @asynccontextmanager
    async def observed_deletion_guard(kb_id):
        deletion_waiting.set()
        async with original_deletion_guard(kb_id):
            deletion_entered.set()
            yield

    async def forbidden_cleanup(*_args, **_kwargs):
        cleanup_called.set()
        raise AssertionError("hard delete reached physical cleanup after restore")

    monkeypatch.setattr(service, "restore", blocked_restore)
    monkeypatch.setattr(
        metadata_store,
        "kb_deletion_guard",
        observed_deletion_guard,
    )
    monkeypatch.setattr(deletion_service, "_run_physical_cleanup", forbidden_cleanup)

    restore_task = asyncio.create_task(
        restore_endpoint(
            kb_id,
            _route_request(app, "POST", f"/kbs/{kb_id}:restore"),
        )
    )
    await asyncio.wait_for(restore_entered.wait(), timeout=5)
    hard_delete_task = asyncio.create_task(
        delete_endpoint(
            kb_id,
            _route_request(app, "DELETE", f"/kbs/{kb_id}"),
            hard=True,
        )
    )
    await asyncio.wait_for(deletion_waiting.wait(), timeout=5)
    await asyncio.sleep(0)
    assert not deletion_entered.is_set()
    assert not hard_delete_task.done()

    release_restore.set()
    restored = await asyncio.wait_for(restore_task, timeout=5)
    assert restored.status == "active"
    await asyncio.wait_for(deletion_entered.wait(), timeout=5)
    with pytest.raises(HTTPException) as hard_delete_error:
        await asyncio.wait_for(hard_delete_task, timeout=5)
    assert hard_delete_error.value.status_code == 500
    assert not cleanup_called.is_set()
    current = await service.get(kb_id, include_deleted=True)
    assert current.status == "active"
    assert current.generation == record.generation
    lifecycle = await metadata_store.get_kb_lifecycle(kb_id)
    assert lifecycle is not None
    assert lifecycle.state == "active"
    assert lifecycle.delete_job_id is None


@pytest.mark.asyncio
async def test_restore_returns_conflict_after_hard_delete_enters_exclusive_guard(
    tmp_path, monkeypatch
):
    kb_id = "kb_hard_delete_wins"
    app, service, _metadata_store, deletion_service, _record = (
        await _build_restore_race_app(tmp_path, kb_id)
    )
    restore_endpoint = _route_endpoint(app, "/kbs/{kb_id}:restore", "POST")
    delete_endpoint = _route_endpoint(app, "/kbs/{kb_id}", "DELETE")
    hard_delete_entered = asyncio.Event()
    release_hard_delete = asyncio.Event()

    async def blocked_cleanup(*_args, **_kwargs):
        hard_delete_entered.set()
        await asyncio.wait_for(release_hard_delete.wait(), timeout=5)

    monkeypatch.setattr(deletion_service, "_run_physical_cleanup", blocked_cleanup)
    hard_delete_task = asyncio.create_task(
        delete_endpoint(
            kb_id,
            _route_request(app, "DELETE", f"/kbs/{kb_id}"),
            hard=True,
        )
    )
    await asyncio.wait_for(hard_delete_entered.wait(), timeout=5)

    with pytest.raises(HTTPException) as restore_error:
        await restore_endpoint(
            kb_id,
            _route_request(app, "POST", f"/kbs/{kb_id}:restore"),
        )
    assert restore_error.value.status_code == 409
    detail: Any = restore_error.value.detail
    assert isinstance(detail, dict)
    assert detail["error_code"] == "kb_hard_delete_in_progress"

    release_hard_delete.set()
    deleted = await asyncio.wait_for(hard_delete_task, timeout=5)
    assert deleted.status == "deleted"


def test_kb_restore_blocked_while_clear_kb_in_flight(tmp_path):
    client, _service, _metadata_store, _job_service, _registry, _probe, _deletion = (
        _build_durable_hard_delete_client(tmp_path)
    )
    created = client.post(
        "/kbs", json={"id": "kb_res_hard", "name": "Hard Pending"}, headers=_HEADERS
    )
    assert created.status_code == 200, created.text

    delete_response = client.delete("/kbs/kb_res_hard?hard=true", headers=_HEADERS)
    assert delete_response.status_code == 200, delete_response.text
    assert delete_response.json()["hard_delete_job_status"] == "queued"

    blocked = client.post("/kbs/kb_res_hard:restore", headers=_HEADERS)
    assert blocked.status_code == 409, blocked.text
    detail = blocked.json()["detail"]
    assert detail["error_code"] == "kb_hard_delete_in_progress"
    assert detail["job_id"] == delete_response.json()["hard_delete_job_id"]


def test_kb_restore_blocked_after_clear_kb_failed_with_deleting_fence(tmp_path):
    (
        client,
        _service,
        metadata_store,
        _job_service,
        _registry,
        _probe,
        _deletion,
    ) = (
        _build_durable_hard_delete_client(tmp_path)
    )
    created = client.post(
        "/kbs",
        json={"id": "kb_res_failed", "name": "Hard Failed"},
        headers=_HEADERS,
    )
    assert created.status_code == 200, created.text
    deleted = client.delete(
        "/kbs/kb_res_failed?hard=true",
        headers=_HEADERS,
    )
    assert deleted.status_code == 200, deleted.text
    job_id = deleted.json()["hard_delete_job_id"]
    jobs, total = asyncio.run(metadata_store.list_jobs("kb_res_failed"))
    assert total == 1
    job = jobs[0]
    generation = job.payload["kb_generation"]

    async def mark_failed_after_guard() -> None:
        running = await metadata_store.transition_job(
            "kb_res_failed",
            job_id,
            status="running",
        )
        assert running.status == "running"
        async with metadata_store.kb_deletion_guard(
            "kb_res_failed",
            generation,
            job_id,
        ):
            pass
        await metadata_store.transition_job(
            "kb_res_failed",
            job_id,
            status="failed",
            error_code="physical_cleanup_failed",
            error_message="retryable",
        )

    asyncio.run(mark_failed_after_guard())
    lifecycle = asyncio.run(metadata_store.get_kb_lifecycle("kb_res_failed"))
    assert lifecycle is not None
    assert lifecycle.state == "deleting"

    blocked = client.post("/kbs/kb_res_failed:restore", headers=_HEADERS)
    assert blocked.status_code == 409, blocked.text
    detail = blocked.json()["detail"]
    assert detail["error_code"] == "kb_hard_delete_in_progress"
    assert detail["job_id"] == job_id


def test_deleting_fence_blocks_recreate_and_restore_after_metadata_purge(tmp_path):
    (
        client,
        service,
        metadata_store,
        _job_service,
        _registry,
        _probe,
        _deletion,
    ) = _build_durable_hard_delete_client(tmp_path)
    created = client.post(
        "/kbs",
        json={"id": "kb_tail_fence", "name": "Tail Fence"},
        headers=_HEADERS,
    )
    assert created.status_code == 200, created.text
    deleted = client.delete(
        "/kbs/kb_tail_fence?hard=true",
        headers=_HEADERS,
    )
    assert deleted.status_code == 200, deleted.text
    job_id = deleted.json()["hard_delete_job_id"]
    jobs, total = asyncio.run(metadata_store.list_jobs("kb_tail_fence"))
    assert total == 1
    job = jobs[0]
    generation = job.payload["kb_generation"]

    async def stop_after_metadata_purge() -> None:
        await metadata_store.transition_job(
            "kb_tail_fence",
            job_id,
            status="running",
        )
        async with metadata_store.kb_deletion_guard(
            "kb_tail_fence",
            generation,
            job_id,
        ):
            await metadata_store.purge_kb_metadata(
                "kb_tail_fence",
                generation=generation,
                delete_job_id=job_id,
            )
        await metadata_store.transition_job(
            "kb_tail_fence",
            job_id,
            status="failed",
            error_code="complete_failed",
            error_message="retryable tail",
        )

    asyncio.run(stop_after_metadata_purge())
    lifecycle = asyncio.run(metadata_store.get_kb_lifecycle("kb_tail_fence"))
    assert lifecycle is not None
    assert lifecycle.state == "deleting"

    recreated = client.post(
        "/kbs",
        json={"id": "kb_tail_fence", "name": "Must Be Fenced"},
        headers=_HEADERS,
    )
    assert recreated.status_code == 409, recreated.text
    retained = asyncio.run(service.get("kb_tail_fence", include_deleted=True))
    assert retained.status == "deleted"
    assert retained.generation == generation

    restore = client.post("/kbs/kb_tail_fence:restore", headers=_HEADERS)
    assert restore.status_code == 409, restore.text
    detail = restore.json()["detail"]
    assert detail["error_code"] == "kb_hard_delete_in_progress"
    assert detail["job_id"] == job_id


def test_kb_metadata_create_merge_and_size_cap(tmp_path):
    client, _service, _registry, _probe = _build_client(tmp_path)

    created = client.post(
        "/kbs",
        json={
            "id": "kb_meta",
            "name": "Meta",
            "metadata": {
                "tags": ["legal", "hr"],
                "team": "ops",
                "platform_provisioned": True,
                "tenant_managed": True,
                "tenant_tag": "tenant:spoofed",
            },
        },
        headers=_HEADERS,
    )
    assert created.status_code == 200, created.text
    assert created.json()["metadata"] == {"tags": ["legal", "hr"], "team": "ops"}
    assert created.json()["origin"] == "platform"

    # Old records / omitted metadata default to an empty dict.
    plain = client.post(
        "/kbs", json={"id": "kb_meta_plain", "name": "Plain"}, headers=_HEADERS
    )
    assert plain.status_code == 200
    assert plain.json()["metadata"] == {}

    # PATCH merges: provided keys overwrite, null values delete, others stay.
    patched = client.patch(
        "/kbs/kb_meta",
        json={"metadata": {"team": None, "stage": "prod"}},
        headers=_HEADERS,
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["metadata"] == {"tags": ["legal", "hr"], "stage": "prod"}

    listed = client.get("/kbs", headers=_HEADERS)
    by_id = {item["id"]: item for item in listed.json()["knowledge_bases"]}
    assert by_id["kb_meta"]["metadata"]["stage"] == "prod"

    # Top-level null is rejected (metadata must be an object).
    null_patch = client.patch(
        "/kbs/kb_meta", json={"metadata": None}, headers=_HEADERS
    )
    assert null_patch.status_code == 400

    for reserved_key in (
        "platform_provisioned",
        "tenant_managed",
        "tenant_tag",
    ):
        reserved_patch = client.patch(
            "/kbs/kb_meta",
            json={"metadata": {reserved_key: True}},
            headers=_HEADERS,
        )
        assert reserved_patch.status_code == 400
        assert "reserved" in reserved_patch.json()["detail"]

    # Oversized metadata (>16KB serialized) is rejected on create and PATCH.
    oversized = {"blob": "x" * (16 * 1024 + 1)}
    assert (
        client.post(
            "/kbs",
            json={"id": "kb_meta_big", "name": "Big", "metadata": oversized},
            headers=_HEADERS,
        ).status_code
        == 400
    )
    assert (
        client.patch(
            "/kbs/kb_meta", json={"metadata": oversized}, headers=_HEADERS
        ).status_code
        == 400
    )
