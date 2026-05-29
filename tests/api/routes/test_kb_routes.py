import asyncio
import importlib
import multiprocessing
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
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
        )
    )
    return TestClient(app), metadata_store, registry, probe


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
        json={"name": "Renamed", "status": "disabled", "visibility": "private"},
        headers=_HEADERS,
    )
    assert patch_response.status_code == 200
    patched = patch_response.json()
    assert patched["name"] == "Renamed"
    assert patched["status"] == "disabled"
    assert patched["visibility"] == "private"

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
    assert probe.finalized == [workspace]
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
