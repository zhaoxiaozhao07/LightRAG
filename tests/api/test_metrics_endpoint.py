from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from lightrag.api.metrics import (
    build_prometheus_metrics,
    record_http_request,
    reset_http_metrics_for_tests,
)


pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _reset_http_metrics():
    reset_http_metrics_for_tests()
    yield
    reset_http_metrics_for_tests()


class FakeKBService:
    async def list(self, *, include_deleted: bool = False):
        assert include_deleted is True
        return [
            SimpleNamespace(id="kb_alpha", status="active"),
            SimpleNamespace(id="kb_deleted", status="deleted"),
        ]


class FakeMetadataStore:
    async def list_documents(
        self,
        kb_id: str,
        *,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ):
        assert kb_id == "kb_alpha"
        assert limit == 1
        assert offset == 0
        totals = {None: 3, "ready": 2, "parse_failed": 1}
        return [], totals.get(status, 0)

    async def list_jobs(
        self,
        kb_id: str,
        *,
        statuses=None,
        document_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ):
        assert kb_id == "kb_alpha"
        assert document_id is None
        assert limit == 1
        assert offset == 0
        status = statuses[0] if statuses else None
        totals = {None: 4, "queued": 1, "failed": 2, "succeeded": 1}
        return [], totals.get(status, 0)

    async def list_audit_events(self, *, limit: int = 100):
        assert limit == 500
        return [
            SimpleNamespace(event_type="login_success"),
            SimpleNamespace(event_type="kb_acl_granted"),
            SimpleNamespace(event_type="kb_acl_granted"),
        ]


async def _build_sample_metrics() -> str:
    return await build_prometheus_metrics(
        kb_service=FakeKBService(),
        metadata_store=FakeMetadataStore(),
        enterprise_enabled=True,
        job_worker_enabled=True,
        object_storage_enabled=False,
        kb_metadata_backend="local",
    )


def test_build_prometheus_metrics_includes_kb_document_job_and_audit_counts():
    metrics = asyncio.run(_build_sample_metrics())

    assert "lightrag_enterprise_enabled 1" in metrics
    assert 'lightrag_info{kb_metadata_backend="local"} 1' in metrics
    assert "lightrag_kb_total 2" in metrics
    assert 'lightrag_kb_status_total{status="active"} 1' in metrics
    assert 'lightrag_kb_documents_total{kb_id="kb_alpha",status="all"} 3' in metrics
    assert 'lightrag_kb_documents_total{kb_id="kb_alpha",status="ready"} 2' in metrics
    assert 'lightrag_kb_jobs_total{kb_id="kb_alpha",status="failed"} 2' in metrics
    assert "lightrag_enterprise_audit_events_sampled_total 3" in metrics
    assert (
        'lightrag_enterprise_audit_events_sampled_total{event_type="kb_acl_granted"} 2'
        in metrics
    )


def test_build_prometheus_metrics_includes_http_counters_and_latency_histogram():
    record_http_request("GET", "/health", 200, 0.03)

    metrics = asyncio.run(_build_sample_metrics())

    assert "# TYPE lightrag_http_requests_total counter" in metrics
    assert (
        'lightrag_http_requests_total{method="GET",route="/health",status_code="200"} 1'
        in metrics
    )
    assert "# TYPE lightrag_http_request_duration_seconds histogram" in metrics
    assert (
        'lightrag_http_request_duration_seconds_bucket{le="0.05",method="GET",route="/health",status_code="200"} 1'
        in metrics
    )
    assert (
        'lightrag_http_request_duration_seconds_bucket{le="+Inf",method="GET",route="/health",status_code="200"} 1'
        in metrics
    )
    assert (
        'lightrag_http_request_duration_seconds_count{method="GET",route="/health",status_code="200"} 1'
        in metrics
    )


def test_metrics_endpoint_returns_prometheus_text(tmp_path, monkeypatch):
    for var in (
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
        "LIGHTRAG_API_KEY",
        "LIGHTRAG_KB_METADATA_BACKEND",
        "LIGHTRAG_OBJECT_STORAGE",
        "LIGHTRAG_KB_JOB_WORKER",
        "LIGHTRAG_ENTERPRISE_AUTH_ENABLED",
        "LIGHTRAG_ENTERPRISE_LEGACY_API_KEY_SUPERADMIN",
        "LIGHTRAG_ENTERPRISE_DISABLE_GLOBAL_ROUTES",
    ):
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
    monkeypatch.setenv("LIGHTRAG_API_KEY", "metrics-key")
    monkeypatch.setenv("LIGHTRAG_ENTERPRISE_AUTH_ENABLED", "false")
    monkeypatch.setenv("WORKING_DIR", str(tmp_path / "rag_storage"))

    from lightrag.api.config import initialize_config, parse_args

    original_argv = sys.argv.copy()
    sys.argv = ["lightrag-server"]
    try:
        args = parse_args()
        initialize_config(args, force=True)
        with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
            mock_rag.return_value = MagicMock()
            from lightrag.api.lightrag_server import create_app

            app = create_app(args)
    finally:
        sys.argv = original_argv

    client = TestClient(app)
    response = client.get("/metrics", headers={"X-API-Key": "metrics-key"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "lightrag_enterprise_enabled 0" in response.text
    assert "lightrag_kb_total" in response.text

    second = client.get("/metrics", headers={"X-API-Key": "metrics-key"})
    assert second.status_code == 200
    assert 'lightrag_http_requests_total{method="GET",route="/metrics",status_code="200"}' in second.text
    assert 'lightrag_http_request_duration_seconds_count{method="GET",route="/metrics",status_code="200"}' in second.text
