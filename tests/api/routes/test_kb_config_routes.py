from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightrag.api.config_version_service import ConfigVersionService
from lightrag.api.job_service import JobService
from lightrag.api.kb_service import KnowledgeBaseService
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry, LightRAGLike
from lightrag.api.metadata_store import SQLiteMetadataStore

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
        self.finalized = False

    async def finalize_storages(self) -> None:
        self.finalized = True


class BuilderProbe:
    def __init__(self):
        self.instances: list[FakeRAG] = []

    async def build(self, record) -> FakeRAG:
        rag = FakeRAG(record.workspace)
        self.instances.append(rag)
        return rag

    async def finalize(self, rag: LightRAGLike) -> None:
        await rag.finalize_storages()


def _build_client(tmp_path: Path):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kb.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    job_service = JobService(kb_service, metadata_store)
    probe = BuilderProbe()
    registry = LightRAGInstanceRegistry(kb_service, probe.build, probe.finalize)
    config_service = ConfigVersionService(kb_service, metadata_store, registry)
    app = FastAPI()
    app.include_router(
        create_kb_routes(
            kb_service,
            registry,
            api_key=_API_KEY,
            job_service=job_service,
            config_service=config_service,
        )
    )
    return TestClient(app), kb_service, metadata_store, registry, probe


def _create_kb(client: TestClient, kb_id: str):
    response = client.post(
        "/kbs", json={"id": kb_id, "name": kb_id}, headers=_HEADERS
    )
    assert response.status_code == 200
    return response.json()


_BASE_CONFIG = {
    "parser_config": {"engine": "mineru"},
    "chunk_config": {"chunk_size": 512},
    "embedding_config": {"model": "bge-large", "dim": 1024},
    "llm_role_config": {"extract": "gpt-4o-mini"},
    "query_config": {"top_k": 60},
}


def test_config_version_create_list_get(tmp_path):
    client, *_ = _build_client(tmp_path)
    _create_kb(client, "kb_cfg")

    create = client.post(
        "/kbs/kb_cfg/configs",
        json={"config": _BASE_CONFIG, "created_by": "alice"},
        headers=_HEADERS,
    )
    assert create.status_code == 200
    body = create.json()
    assert body["version"] == 1
    assert body["parser_hash"].startswith("sha256:")
    assert body["index_hash"].startswith("sha256:")
    assert body["query_hash"].startswith("sha256:")
    assert body["activated_at"] is None
    assert body["created_by"] == "alice"

    second = client.post(
        "/kbs/kb_cfg/configs",
        json={"config": {**_BASE_CONFIG, "query_config": {"top_k": 80}}},
        headers=_HEADERS,
    )
    assert second.status_code == 200
    assert second.json()["version"] == 2

    listing = client.get("/kbs/kb_cfg/configs", headers=_HEADERS)
    assert listing.status_code == 200
    listing_body = listing.json()
    assert listing_body["total"] == 2
    versions = [item["version"] for item in listing_body["versions"]]
    assert versions == [2, 1]

    detail = client.get(
        f"/kbs/kb_cfg/configs/{body['id']}", headers=_HEADERS
    )
    assert detail.status_code == 200
    assert detail.json()["id"] == body["id"]


def test_config_version_activate_updates_kb_and_evicts_registry(tmp_path):
    client, kb_service, _store, registry, probe = _build_client(tmp_path)
    _create_kb(client, "kb_activate")
    # Force registry to load an instance
    import asyncio
    rag = asyncio.run(registry.get("kb_activate"))
    assert registry.is_loaded("kb_activate")

    create = client.post(
        "/kbs/kb_activate/configs",
        json={"config": _BASE_CONFIG},
        headers=_HEADERS,
    )
    version_id = create.json()["id"]

    activate = client.post(
        f"/kbs/kb_activate/configs/{version_id}:activate", headers=_HEADERS
    )
    assert activate.status_code == 200
    activated = activate.json()
    assert activated["activated_at"] is not None

    refreshed = asyncio.run(kb_service.get("kb_activate"))
    assert refreshed.active_config_version_id == version_id

    # Activation must have discarded the cached LightRAG instance
    assert not registry.is_loaded("kb_activate")
    assert rag.finalized is True


def test_config_version_diff_reports_changes(tmp_path):
    client, *_ = _build_client(tmp_path)
    _create_kb(client, "kb_diff")

    base = client.post(
        "/kbs/kb_diff/configs",
        json={"config": _BASE_CONFIG},
        headers=_HEADERS,
    )
    base_id = base.json()["id"]
    client.post(
        f"/kbs/kb_diff/configs/{base_id}:activate", headers=_HEADERS
    )

    target = client.post(
        "/kbs/kb_diff/configs",
        json={
            "config": {
                **_BASE_CONFIG,
                "embedding_config": {"model": "bge-m3", "dim": 1024},
                "chunk_config": {"chunk_size": 1024},
            }
        },
        headers=_HEADERS,
    )
    diff = client.post(
        f"/kbs/kb_diff/configs/{target.json()['id']}:diff", headers=_HEADERS
    )
    assert diff.status_code == 200
    body = diff.json()
    assert body["requires_reindex"] is True
    assert body["requires_vector_rebuild"] is True
    assert body["requires_reparse"] is False
    assert "embedding_changed" in body["reasons"]
    assert "index_hash_changed" in body["reasons"]


def test_config_version_diff_without_active_returns_full_rebuild(tmp_path):
    client, *_ = _build_client(tmp_path)
    _create_kb(client, "kb_no_active")

    new_version = client.post(
        "/kbs/kb_no_active/configs",
        json={"config": _BASE_CONFIG},
        headers=_HEADERS,
    )
    diff = client.post(
        f"/kbs/kb_no_active/configs/{new_version.json()['id']}:diff",
        headers=_HEADERS,
    )
    body = diff.json()
    assert body["active_version_id"] is None
    assert body["requires_reparse"] is True
    assert body["requires_reindex"] is True
    assert body["requires_vector_rebuild"] is True


def test_config_version_get_404(tmp_path):
    client, *_ = _build_client(tmp_path)
    _create_kb(client, "kb_404")
    response = client.get(
        "/kbs/kb_404/configs/cfg_missing", headers=_HEADERS
    )
    assert response.status_code == 404
