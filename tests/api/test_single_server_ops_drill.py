from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx
import pytest
import yaml

from scripts.run_single_server_ops_drill import ApiClient, run_checks


pytestmark = pytest.mark.offline

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "backup_id": "backup_test",
        "kb_id": ["kb_alpha"],
        "sample_kb_limit": 3,
        "query": "restore smoke test",
        "query_mode": "naive",
        "skip_query": True,
        "hard_delete_drill_kb_id": None,
        "hard_delete_seed_text": "seed",
        "hard_delete_auto_index": False,
        "wait_jobs": True,
        "job_timeout_seconds": 10.0,
        "poll_interval_seconds": 0.01,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize(
    "relative_path",
    [
        "deploy/monitoring/prometheus-rules.yml",
        "deploy/monitoring/slo.md",
        "deploy/monitoring/grafana-lightrag-overview.json",
    ],
)
def test_monitoring_assets_exist(relative_path: str) -> None:
    assert (PROJECT_ROOT / relative_path).is_file()


def test_prometheus_rules_are_valid_yaml_and_cover_core_alerts() -> None:
    payload = yaml.safe_load(
        (PROJECT_ROOT / "deploy/monitoring/prometheus-rules.yml").read_text(
            encoding="utf-8"
        )
    )

    alert_names = {
        rule["alert"]
        for group in payload["groups"]
        for rule in group["rules"]
        if "alert" in rule
    }
    assert "LightRAGHighServerErrors" in alert_names
    assert "LightRAGQueuedOrRunningJobsStuck" in alert_names
    assert "LightRAGExporterDown" in alert_names


def test_grafana_dashboard_is_valid_json_and_references_lightrag_metrics() -> None:
    payload = json.loads(
        (PROJECT_ROOT / "deploy/monitoring/grafana-lightrag-overview.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["uid"] == "lightrag-single-server-overview"
    exprs = [target["expr"] for panel in payload["panels"] for target in panel["targets"]]
    assert any("lightrag_http_requests_total" in expr for expr in exprs)
    assert any("lightrag_kb_jobs_total" in expr for expr in exprs)
    assert any("lightrag_http_request_duration_seconds_bucket" in expr for expr in exprs)


@pytest.mark.asyncio
async def test_single_server_ops_smoke_with_mock_transport() -> None:
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy", "auth_mode": "api_key"})
        if request.url.path == "/metrics":
            return httpx.Response(
                200,
                text="lightrag_kb_total 1\nlightrag_http_request_duration_seconds_count 1\n",
                headers={"content-type": "text/plain; version=0.0.4"},
            )
        if request.url.path == "/kbs":
            return httpx.Response(200, json=[{"id": "kb_alpha", "status": "active"}])
        if request.url.path == "/kbs/kb_alpha/status":
            return httpx.Response(200, json={"kb": {"id": "kb_alpha"}})
        if request.url.path == "/kbs/kb_alpha/documents":
            return httpx.Response(200, json={"documents": [], "total": 0})
        if request.url.path == "/kbs/kb_alpha/graph/status":
            return httpx.Response(200, json={"node_count": 0, "edge_count": 0})
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    api = ApiClient(
        "http://testserver",
        {"X-API-Key": "test"},
        timeout=5.0,
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await run_checks(api, _args())
    finally:
        await api.close()

    assert result.ok is True
    assert result.report["metrics"]["has_kb_total"] is True
    assert result.report["sampled_kbs"][0]["kb_id"] == "kb_alpha"
    assert ("GET", "/metrics") in seen


@pytest.mark.asyncio
async def test_single_server_hard_delete_drill_detects_clean_workspace_reuse() -> None:
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"})
        if request.url.path == "/metrics":
            return httpx.Response(200, text="lightrag_kb_total 0\n")
        if request.url.path == "/kbs" and request.method == "GET":
            return httpx.Response(200, json=[])
        if request.url.path == "/kbs" and request.method == "POST":
            body = json.loads(request.content.decode("utf-8"))
            return httpx.Response(200, json={"id": body["id"], "status": "active"})
        if request.url.path == "/kbs/kb_drill" and request.method == "DELETE":
            delete_count = requests.count(("DELETE", "/kbs/kb_drill"))
            if delete_count == 1:
                return httpx.Response(404, json={"detail": "not found"})
            if delete_count == 2:
                return httpx.Response(
                    200,
                    json={
                        "id": "kb_drill",
                        "hard_delete_queued": True,
                        "hard_delete_job_id": "job_hard_delete",
                    },
                )
            return httpx.Response(200, json={"id": "kb_drill"})
        if request.url.path == "/kbs/kb_drill/documents:texts":
            return httpx.Response(200, json={"job_id": "job_seed", "documents": []})
        if request.url.path == "/kbs/kb_drill/jobs/job_seed:wait":
            return httpx.Response(200, json={"id": "job_seed", "status": "succeeded"})
        if request.url.path == "/kbs/kb_drill/jobs/job_hard_delete:wait":
            return httpx.Response(
                200,
                json={"id": "job_hard_delete", "status": "succeeded"},
            )
        if request.url.path == "/kbs/kb_drill/documents":
            return httpx.Response(200, json={"documents": [], "total": 0})
        if request.url.path == "/kbs/kb_drill/graph/status":
            return httpx.Response(200, json={"node_count": 0, "edge_count": 0})
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    api = ApiClient(
        "http://testserver",
        {},
        timeout=5.0,
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await run_checks(
            api,
            _args(kb_id=[], hard_delete_drill_kb_id="kb_drill", skip_query=True),
        )
    finally:
        await api.close()

    assert result.ok is True
    assert result.report["hard_delete_drill"]["hard_delete_job"]["status"] == "succeeded"
    assert result.report["hard_delete_drill"]["recreated_workspace"] == {
        "documents_total": 0,
        "node_count": 0,
        "edge_count": 0,
    }
