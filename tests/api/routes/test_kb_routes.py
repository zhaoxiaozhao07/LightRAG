import asyncio
import importlib
import multiprocessing
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightrag.api.kb_service import (
    KnowledgeBaseConflictError,
    KnowledgeBaseService,
    sanitize_workspace,
    validate_kb_id,
)
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry, LightRAGLike
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
    assert config_response.status_code == 200
    assert config_response.json()["active_config_version_id"] == "cfg_1"

    clear_response = client.patch(
        "/kbs/kb_patch",
        json={
            "description": None,
            "owner_id": None,
            "tenant_id": None,
            "active_config_version_id": None,
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


def test_missing_kb_returns_404(tmp_path):
    client, _service, _registry, _probe = _build_client(tmp_path)
    assert client.get("/kbs/missing", headers=_HEADERS).status_code == 404
    assert client.get("/kbs/missing/status", headers=_HEADERS).status_code == 404
    assert client.delete("/kbs/missing", headers=_HEADERS).status_code == 404
