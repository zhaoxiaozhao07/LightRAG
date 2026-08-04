from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import httpx
import pytest
import yaml

from scripts.run_single_server_ops_drill import ApiClient, CliResult, run_checks


pytestmark = pytest.mark.offline

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _local_artifact_lifecycle_block() -> dict[str, object]:
    """A minimal valid local-mode ``artifact_lifecycle`` health block.

    Mirrors the additive sibling shape emitted by fix-16 so the always-on
    Phase 3.3 health assertion passes for every mock-transport drill. The
    manifests aggregate carries every documented counter (retained/blocked
    included, both 0) and the bounded probes report concrete integers.
    """

    return {
        "mode": "local",
        "backend": "none",
        "capability_admitted": {
            "implemented": False,
            "admission_gate_allows_object_mode": False,
        },
        "object_store_ready": False,
        "manifests": {
            "total": 0,
            "pending": 0,
            "leased": 0,
            "retained": 0,
            "blocked": 0,
            "succeeded": 0,
            "due_pending": 0,
            "expired_leases": 0,
            "cleanup_deadline_overdue": 0,
            "oldest_due_at": None,
        },
        "maintenance_runs": 0,
        "migration_blockers": 0,
        "unresolved_commit_unknown": 0,
        "recovery_cursor_stale": 0,
    }


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
        "migration_cli_drill": False,
        "orphan_cli_drill": False,
        "cli_drill_bucket": "lightrag-kb",
        "cli_drill_prefix": "kb",
        "cli_drill_endpoint": None,
        "cli_drill_timeout_seconds": 30.0,
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
    exprs = [
        target["expr"] for panel in payload["panels"] for target in panel["targets"]
    ]
    assert any("lightrag_http_requests_total" in expr for expr in exprs)
    assert any("lightrag_kb_jobs_total" in expr for expr in exprs)
    assert any(
        "lightrag_http_request_duration_seconds_bucket" in expr for expr in exprs
    )


@pytest.mark.asyncio
async def test_single_server_ops_smoke_with_mock_transport() -> None:
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "status": "healthy",
                    "auth_mode": "api_key",
                    "artifact_lifecycle": _local_artifact_lifecycle_block(),
                },
            )
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
            return httpx.Response(
                200,
                json={
                    "status": "healthy",
                    "artifact_lifecycle": _local_artifact_lifecycle_block(),
                },
            )
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
    assert (
        result.report["hard_delete_drill"]["hard_delete_job"]["status"] == "succeeded"
    )
    assert result.report["hard_delete_drill"]["recreated_workspace"] == {
        "documents_total": 0,
        "node_count": 0,
        "edge_count": 0,
    }


# ---------------------------------------------------------------------------
# Phase 3.3 — artifact-lifecycle health assertion + CLI-invocation drills
# ---------------------------------------------------------------------------


def _object_mode_lifecycle_block() -> dict[str, object]:
    """A full object-mode artifact_lifecycle block with non-zero aggregates."""

    return {
        "mode": "object",
        "backend": "_FakeObjectStorage",
        "capability_admitted": {
            "implemented": False,
            "admission_gate_allows_object_mode": False,
        },
        "object_store_ready": True,
        "manifests": {
            "total": 5,
            "pending": 2,
            "leased": 1,
            "retained": 1,
            "blocked": 0,
            "succeeded": 1,
            "due_pending": 1,
            "expired_leases": 0,
            "cleanup_deadline_overdue": 0,
            "oldest_due_at": "2026-08-04T13:00:00+00:00",
        },
        "maintenance_runs": 3,
        "migration_blockers": 2,
        "unresolved_commit_unknown": 1,
        "recovery_cursor_stale": 4,
    }


def _offline_handler(
    lifecycle_block: dict[str, object],
) -> Callable[[httpx.Request], httpx.Response]:
    """Build a mock-transport handler with the given artifact_lifecycle block."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={"status": "healthy", "artifact_lifecycle": lifecycle_block},
            )
        if request.url.path == "/metrics":
            return httpx.Response(200, text="lightrag_kb_total 0\n")
        if request.url.path == "/kbs":
            return httpx.Response(200, json=[])
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    return handler


@pytest.mark.asyncio
async def test_artifact_lifecycle_health_block_asserts_full_key_shape() -> None:
    """Step 1: the drill requires every documented artifact_lifecycle key."""

    api = ApiClient(
        "http://testserver",
        {},
        timeout=5.0,
        transport=httpx.MockTransport(_offline_handler(_object_mode_lifecycle_block())),
    )
    try:
        result = await run_checks(api, _args(kb_id=[]))
    finally:
        await api.close()

    assert result.ok is True
    lifecycle = result.report["artifact_lifecycle"]
    assert lifecycle["mode"] == "object"
    assert lifecycle["manifests"]["retained"] == 1
    # Step 4: retained/blocked cleanup observability.
    assert result.report["retained_blocked_cleanup"] == {
        "reported": True,
        "retained": 1,
        "blocked": 0,
    }
    # Step 5: commit-unknown recovery observability.
    assert result.report["commit_unknown_recovery"] == {"unresolved_commit_unknown": 1}
    # Steps 6 & 7: documented-but-unexecuted sections are always recorded.
    documented = result.report["documented_drills"]
    assert documented["moved_root_operation"]["executed"] is False
    assert documented["production_staging_graph_vector_llm"]["executed"] is False


@pytest.mark.asyncio
async def test_artifact_lifecycle_drill_fails_when_block_missing() -> None:
    """Step 1: a server that omits artifact_lifecycle fails the drill."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            # legacy server: no artifact_lifecycle block
            return httpx.Response(200, json={"status": "healthy"})
        if request.url.path == "/metrics":
            return httpx.Response(200, text="lightrag_kb_total 0\n")
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    api = ApiClient(
        "http://testserver", {}, timeout=5.0, transport=httpx.MockTransport(handler)
    )
    try:
        result = await run_checks(api, _args(kb_id=[]))
    finally:
        await api.close()

    assert result.ok is False
    assert "artifact_lifecycle" in result.report["error"]["message"]


@pytest.mark.asyncio
async def test_retained_blocked_drill_records_not_reported_gracefully() -> None:
    """Step 4: a 'not_reported' manifests aggregate is recorded, not fatal."""

    block = _local_artifact_lifecycle_block()
    block["manifests"] = "not_reported"  # type: ignore[assignment]

    api = ApiClient(
        "http://testserver",
        {},
        timeout=5.0,
        transport=httpx.MockTransport(_offline_handler(block)),
    )
    try:
        result = await run_checks(api, _args(kb_id=[]))
    finally:
        await api.close()

    assert result.ok is True
    assert result.report["retained_blocked_cleanup"]["reported"] is False


class _FakeCliRunner:
    """Records CLI invocations and returns a canned ``CliResult``.

    Stands in for :func:`scripts.run_single_server_ops_drill._default_cli_runner`
    so the CLI drills are exercised end-to-end (temp working dir, command build,
    JSON parsing, assertions) without spawning a real subprocess or needing
    object-storage connectivity.
    """

    def __init__(
        self, payload: dict[str, object], *, returncode: int = 0, stderr: str = ""
    ) -> None:
        self._stdout = json.dumps(payload)
        self._returncode = returncode
        self._stderr = stderr
        self.commands: list[list[str]] = []
        self.last_env: object | None = None
        self.last_cwd: object | None = None

    def __call__(self, command, env, cwd, timeout):  # noqa: ANN001, ANN202
        self.commands.append(list(command))
        self.last_env = env
        self.last_cwd = cwd
        return CliResult(self._returncode, self._stdout, self._stderr)


@pytest.mark.asyncio
async def test_migration_cli_drill_parses_plan_json() -> None:
    """Step 2: migration CLI dry-run JSON is parsed into plan_id + item_count."""

    canned = {
        "mode": "plan",
        "plan_id": "mig-plan-test",
        "item_count": 1,
        "metadata_backend": "sqlite",
        "apply_run_id": None,
        "items": [{"root_label": "legacyRoot"}],
    }
    runner = _FakeCliRunner(canned)

    api = ApiClient(
        "http://testserver",
        {},
        timeout=5.0,
        transport=httpx.MockTransport(
            _offline_handler(_local_artifact_lifecycle_block())
        ),
    )
    try:
        result = await run_checks(
            api,
            _args(kb_id=[], migration_cli_drill=True),
            cli_runner=runner,
        )
    finally:
        await api.close()

    assert result.ok is True
    drill = result.report["migration_cli_drill"]
    assert drill["executed"] is True
    assert drill["plan_id"] == "mig-plan-test"
    assert drill["item_count"] == 1
    assert drill["metadata_backend"] == "sqlite"
    assert drill["applied"] is False
    # The drill never ran --yes apply.
    assert "requires" in drill["apply_note"]
    # The command targets the documented console script with dry-run + json.
    assert runner.commands
    command = runner.commands[0]
    assert "--dry-run" in command
    assert "--json" in command
    assert any(str(arg).startswith("legacyRoot=") for arg in command)
    assert "--metadata-backend" in command and "sqlite" in command


@pytest.mark.asyncio
async def test_migration_cli_drill_fails_on_nonzero_exit() -> None:
    """Step 2: a failing CLI surfaces a clear error rather than a false pass."""

    runner = _FakeCliRunner({}, returncode=2, stderr="object storage unreachable")

    api = ApiClient(
        "http://testserver",
        {},
        timeout=5.0,
        transport=httpx.MockTransport(
            _offline_handler(_local_artifact_lifecycle_block())
        ),
    )
    try:
        result = await run_checks(
            api,
            _args(kb_id=[], migration_cli_drill=True),
            cli_runner=runner,
        )
    finally:
        await api.close()

    assert result.ok is False
    assert "exited 2" in result.report["error"]["message"]


@pytest.mark.asyncio
async def test_orphan_cli_drill_parses_plan_json() -> None:
    """Step 3: orphan reconcile CLI dry-run JSON is parsed into plan_id."""

    canned = {
        "mode": "plan",
        "plan_id": "or-plan-test",
        "item_count": 0,
        "metadata_backend": "sqlite",
        "classifications": {
            "eligible": 0,
            "referenced": 0,
            "retained": 0,
            "malformed": 0,
            "unknown_owner": 0,
            "too_new": 0,
        },
    }
    runner = _FakeCliRunner(canned)

    api = ApiClient(
        "http://testserver",
        {},
        timeout=5.0,
        transport=httpx.MockTransport(
            _offline_handler(_local_artifact_lifecycle_block())
        ),
    )
    try:
        result = await run_checks(
            api,
            _args(kb_id=[], orphan_cli_drill=True),
            cli_runner=runner,
        )
    finally:
        await api.close()

    assert result.ok is True
    drill = result.report["orphan_cli_drill"]
    assert drill["executed"] is True
    assert drill["plan_id"] == "or-plan-test"
    assert drill["item_count"] == 0
    assert drill["applied"] is False
    command = runner.commands[0]
    assert "--dry-run" in command and "--json" in command
    # orphan CLI takes no positional LABEL roots.
    assert not any(str(arg).startswith("legacyRoot=") for arg in command)
