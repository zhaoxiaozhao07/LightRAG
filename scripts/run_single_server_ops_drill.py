"""Run single-server operational smoke checks for LightRAG.

The script targets exactly one LightRAG server process on one server and
covers the operational drills in scope for the current deployment model:

1. post-restore validation (``/health`` + ``/metrics`` probe, KB sampling +
   optional query/data smoke) after following
   ``docs/生产级后端备份恢复Runbook.md``;
2. optional hard-delete workspace-reuse cleanup drill on a disposable KB;
3. Phase 3.3 artifact-lifecycle observability (fix-16) — asserts ``/health``
   exposes the additive ``artifact_lifecycle`` block with its full keyed shape
   (mode/backend/capability_admitted/object_store_ready/manifests/
   maintenance_runs/migration_blockers/unresolved_commit_unknown/
   recovery_cursor_stale), plus retained/blocked cleanup and commit-unknown
   recovery observability;
4. optional migration/orphan CLI ``--dry-run --json`` drills that invoke the
   frozen Phase 3.2 operator CLIs (``lightrag-migrate-artifacts-to-object`` and
   ``lightrag-reconcile-orphans``) via subprocess against a throwaway working
   directory. Apply/resume are documented as operator reference and are NEVER
   executed by this drill;
5. documented-but-unexecuted sections for moved-root operation and the
   production-staging graph/vector/LLM drill.

PostgreSQL + MinIO consistency manipulation stays in the separate live
acceptance harness; this script performs HTTP-level steps + CLI invocation
only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import httpx
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_PATH = PROJECT_ROOT / "temp" / "single_server_ops_drill_report.json"
DEFAULT_QUERY = "restore smoke test"


def env_value(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_preview(value: Any, limit: int = 1000) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)[:limit]


def _collection_items(payload: Any, *candidate_keys: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in candidate_keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _collection_total(payload: Any, *candidate_keys: str) -> int:
    if isinstance(payload, dict):
        for key in ("total", "total_count", "count"):
            value = payload.get(key)
            if isinstance(value, int):
                return value
        items = _collection_items(payload, *candidate_keys)
        return len(items)
    if isinstance(payload, list):
        return len(payload)
    return 0


@dataclass(slots=True)
class DrillResult:
    ok: bool
    report: dict[str, Any]


class ApiClient:
    def __init__(
        self,
        base_url: str,
        headers: dict[str, str],
        *,
        timeout: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=min(timeout, 10.0)),
            trust_env=False,
            transport=transport,
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def request_json(self, method: str, path: str, **kwargs: Any) -> Any:
        response = await self.client.request(method, path, **kwargs)
        if response.status_code >= 400:
            raise RuntimeError(
                f"{method} {path} returned HTTP {response.status_code}: "
                f"{response.text[:1000]}"
            )
        if not response.content:
            return None
        return response.json()

    async def get_json(self, path: str, **kwargs: Any) -> Any:
        return await self.request_json("GET", path, **kwargs)

    async def post_json(
        self, path: str, payload: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        return await self.request_json("POST", path, json=payload or {}, **kwargs)

    async def delete_json(self, path: str, *, ignore_404: bool = False) -> Any:
        response = await self.client.delete(path)
        if ignore_404 and response.status_code == 404:
            return {"status_code": 404, "ignored": True}
        if response.status_code >= 400:
            raise RuntimeError(
                f"DELETE {path} returned HTTP {response.status_code}: "
                f"{response.text[:1000]}"
            )
        return response.json() if response.content else None


async def authenticate(args: argparse.Namespace, base_url: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    api_key = args.api_key or env_value("LIGHTRAG_API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key
        return headers

    username = args.username or env_value("LIGHTRAG_USERNAME")
    password = args.password or env_value("LIGHTRAG_PASSWORD")
    if not username or not password:
        return headers

    async with httpx.AsyncClient(
        base_url=base_url, timeout=15.0, trust_env=False
    ) as client:
        response = await client.post(
            "/login",
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Login failed: HTTP {response.status_code}: {response.text}"
            )
        token = response.json().get("access_token")
        if not token:
            raise RuntimeError("Login response did not include access_token")
        headers["Authorization"] = f"Bearer {token}"
        return headers


async def wait_job(
    api: ApiClient,
    kb_id: str,
    job_id: str,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        remaining = max(0.1, min(30.0, deadline - time.monotonic()))
        try:
            payload = await api.post_json(
                f"/kbs/{kb_id}/jobs/{job_id}:wait",
                {},
                params={
                    "timeout_seconds": remaining,
                    "poll_interval_seconds": min(max(poll_interval_seconds, 0.05), 5.0),
                },
            )
        except RuntimeError:
            payload = await api.get_json(f"/kbs/{kb_id}/jobs/{job_id}")
        if isinstance(payload, dict):
            last = payload
            if str(payload.get("status") or "").lower() in {
                "succeeded",
                "failed",
                "cancelled",
            }:
                return payload
        await asyncio.sleep(poll_interval_seconds)
    raise TimeoutError(f"Timed out waiting for job {job_id}: {_json_preview(last)}")


async def wait_hard_delete_if_queued(
    api: ApiClient,
    kb_id: str,
    payload: Any,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if not args.wait_jobs or not isinstance(payload, dict):
        return None
    hard_job_id = payload.get("hard_delete_job_id")
    if not hard_job_id:
        return None
    return await wait_job(
        api,
        kb_id,
        str(hard_job_id),
        timeout_seconds=args.job_timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
    )


async def probe_base_health(api: ApiClient, report: dict[str, Any]) -> None:
    health = await api.get_json("/health")
    report["health"] = {
        "status": health.get("status") if isinstance(health, dict) else None,
        "auth_mode": health.get("auth_mode") if isinstance(health, dict) else None,
        "working_directory": health.get("working_directory")
        if isinstance(health, dict)
        else None,
        "input_directory": health.get("input_directory")
        if isinstance(health, dict)
        else None,
    }
    if report["health"]["status"] != "healthy":
        raise RuntimeError(f"/health did not report healthy: {_json_preview(health)}")

    metrics = await api.client.get("/metrics")
    if metrics.status_code >= 400:
        raise RuntimeError(
            f"GET /metrics returned HTTP {metrics.status_code}: {metrics.text[:1000]}"
        )
    report["metrics"] = {
        "content_type": metrics.headers.get("content-type"),
        "has_kb_total": "lightrag_kb_total" in metrics.text,
        "has_http_histogram": "lightrag_http_request_duration_seconds" in metrics.text,
    }
    if not report["metrics"]["has_kb_total"]:
        raise RuntimeError("/metrics did not include lightrag_kb_total")


# ---------------------------------------------------------------------------
# Phase 3.3 — artifact-lifecycle health observability (fix-16)
# ---------------------------------------------------------------------------

_ARTIFACT_LIFECYCLE_KEYS: frozenset[str] = frozenset(
    {
        "mode",
        "backend",
        "capability_admitted",
        "object_store_ready",
        "manifests",
        "maintenance_runs",
        "migration_blockers",
        "unresolved_commit_unknown",
        "recovery_cursor_stale",
    }
)
_KNOWN_ARTIFACT_MODES: frozenset[str] = frozenset({"local", "object"})


async def probe_artifact_lifecycle_health(
    api: ApiClient, report: dict[str, Any]
) -> None:
    """Phase 3.3 step 1 — assert ``/health`` exposes the artifact_lifecycle block.

    fix-16 (parent-accepted) adds ``artifact_lifecycle`` as an *additive sibling*
    to the existing ``artifact_cleanup`` block. The block carries bounded indexed
    aggregates only (no bucket listing, no object download). This step verifies
    every documented key is present so operators can rely on the health endpoint
    for lifecycle observability after restore and during migration drills.

    In local mode the block reports ``mode == "local"``; object mode is also
    valid. The drill records whichever the server reports rather than forcing a
    single deployment model, but it does require ``mode`` to be one of the known
    values so a broken/legacy server is caught.
    """

    health = await api.get_json("/health")
    block = health.get("artifact_lifecycle") if isinstance(health, dict) else None
    if not isinstance(block, dict):
        raise RuntimeError(
            "/health did not include the 'artifact_lifecycle' block "
            "(fix-16 health extension); cannot complete lifecycle observability drill"
        )
    missing = sorted(_ARTIFACT_LIFECYCLE_KEYS - block.keys())
    if missing:
        raise RuntimeError(
            f"/health artifact_lifecycle block is missing keys: {missing}"
        )
    mode = block.get("mode")
    if not isinstance(mode, str) or mode not in _KNOWN_ARTIFACT_MODES:
        raise RuntimeError(
            f"/health artifact_lifecycle.mode has unexpected value: {mode!r}"
        )
    report["artifact_lifecycle"] = {
        "mode": mode,
        "backend": block.get("backend"),
        "capability_admitted": block.get("capability_admitted"),
        "object_store_ready": block.get("object_store_ready"),
        "manifests": block.get("manifests"),
        "maintenance_runs": block.get("maintenance_runs"),
        "migration_blockers": block.get("migration_blockers"),
        "unresolved_commit_unknown": block.get("unresolved_commit_unknown"),
        "recovery_cursor_stale": block.get("recovery_cursor_stale"),
    }


def probe_retained_blocked_cleanup(report: dict[str, Any]) -> None:
    """Phase 3.3 step 4 — retained/blocked cleanup manifest observability.

    Read-only check that ``artifact_lifecycle.manifests`` surfaces the
    ``retained`` and ``blocked`` counters (both may legitimately be 0).
    ``retained`` manifests are held back pending explicit operator release;
    ``blocked`` manifests failed verified-absence and need investigation.

    When the server could not report aggregates (timeout/error), the block
    collapses ``manifests`` to the string ``"not_reported"`` — recorded without
    failing the drill, since that is a valid degraded-health signal rather than
    a bad artifact-lifecycle contract.
    """

    block = report.get("artifact_lifecycle") or {}
    manifests = block.get("manifests") if isinstance(block, dict) else None
    if manifests == "not_reported" or not isinstance(manifests, dict):
        report["retained_blocked_cleanup"] = {
            "reported": False,
            "reason": "manifests aggregate not reported by server",
            "manifests": manifests,
        }
        return
    if "retained" not in manifests or "blocked" not in manifests:
        raise RuntimeError(
            "artifact_lifecycle.manifests is missing retained/blocked keys: "
            f"{_json_preview(manifests)}"
        )
    report["retained_blocked_cleanup"] = {
        "reported": True,
        "retained": manifests.get("retained"),
        "blocked": manifests.get("blocked"),
    }


def probe_commit_unknown_recovery(report: dict[str, Any]) -> None:
    """Phase 3.3 step 5 — unresolved commit-unknown observability.

    Operator reference — commit-unknown recovery procedure (NOT auto-resolved):
      A job that reached the metadata-commit step but could not confirm the
      outcome is flagged with error_code ``metadata_commit_outcome_unknown`` and
      counted here. The system NEVER auto-resolves these: the artifact may or may
      not have been committed, so silent retry could duplicate or lose data. The
      operator must:

        1. Inspect the job and its document's artifact attachments to determine
           the actual commit state (GET /kbs/<kb>/jobs/<job_id> + document
           metadata).
        2. If the commit succeeded (artifact present + pointer committed), mark
           the job resolved per the documented operator action.
        3. If the commit failed (no artifact/pointer), re-queue the document for
           a fresh attempt after correcting the underlying cause.
        4. Only after manual resolution does this count return to 0.

      This drill only reports the count; it never attempts recovery.
    """

    block = report.get("artifact_lifecycle") or {}
    value = block.get("unresolved_commit_unknown") if isinstance(block, dict) else None
    if value is None:
        raise RuntimeError(
            "artifact_lifecycle.unresolved_commit_unknown is absent from /health"
        )
    if not (isinstance(value, int) or value == "not_reported"):
        raise RuntimeError(
            "artifact_lifecycle.unresolved_commit_unknown has unexpected type: "
            f"{type(value).__name__}={value!r}"
        )
    report["commit_unknown_recovery"] = {
        "unresolved_commit_unknown": value,
    }


async def probe_kbs(
    api: ApiClient, args: argparse.Namespace, report: dict[str, Any]
) -> None:
    payload = await api.get_json("/kbs")
    kbs = _collection_items(payload, "items", "kbs", "knowledge_bases")
    report["kbs"] = {
        "total": _collection_total(payload, "items", "kbs", "knowledge_bases")
    }

    requested = list(args.kb_id or [])
    if not requested:
        requested = [
            str(item.get("id"))
            for item in kbs[: args.sample_kb_limit]
            if item.get("id")
        ]
    report["sampled_kbs"] = []
    for kb_id in requested:
        sample: dict[str, Any] = {"kb_id": kb_id}
        sample["status"] = await api.get_json(f"/kbs/{kb_id}/status")
        documents = await api.get_json(f"/kbs/{kb_id}/documents")
        sample["documents_total"] = _collection_total(documents, "documents", "items")
        graph_status = await api.get_json(f"/kbs/{kb_id}/graph/status")
        sample["graph"] = {
            "node_count": graph_status.get("node_count")
            if isinstance(graph_status, dict)
            else None,
            "edge_count": graph_status.get("edge_count")
            if isinstance(graph_status, dict)
            else None,
            "is_truncated": graph_status.get("is_truncated")
            if isinstance(graph_status, dict)
            else None,
        }
        if not args.skip_query:
            query = await api.post_json(
                f"/kbs/{kb_id}/query/data",
                {
                    "query": args.query,
                    "mode": args.query_mode,
                    "top_k": 5,
                    "chunk_top_k": 5,
                },
            )
            data = query.get("data") if isinstance(query, dict) else {}
            sample["query_data"] = {
                "references_count": len((data or {}).get("references") or []),
                "chunks_count": len((data or {}).get("chunks") or []),
            }
        report["sampled_kbs"].append(sample)


async def hard_delete_disposable_kb_drill(
    api: ApiClient, args: argparse.Namespace
) -> dict[str, Any]:
    kb_id = args.hard_delete_drill_kb_id
    assert kb_id
    result: dict[str, Any] = {"kb_id": kb_id, "seeded": False}

    initial_delete = await api.delete_json(f"/kbs/{kb_id}?hard=true", ignore_404=True)
    result["initial_cleanup"] = initial_delete
    initial_job = await wait_hard_delete_if_queued(api, kb_id, initial_delete, args)
    if initial_job is not None:
        result["initial_cleanup_job"] = initial_job

    await api.post_json(
        "/kbs",
        {
            "id": kb_id,
            "name": f"Single-server hard-delete drill {kb_id}",
            "description": "Disposable KB created by scripts/run_single_server_ops_drill.py",
            "visibility": "private",
        },
    )

    try:
        if args.hard_delete_seed_text:
            seed = await api.post_json(
                f"/kbs/{kb_id}/documents:texts",
                {
                    "documents": [
                        {
                            "text": args.hard_delete_seed_text,
                            "source_name": "single-server-hard-delete-drill.txt",
                            "metadata": {"ops_drill": "hard_delete"},
                        }
                    ],
                    "auto_parse": True,
                    "auto_index": args.hard_delete_auto_index,
                    "idempotency_key": f"hard-delete-drill-seed-{kb_id}",
                },
            )
            result["seeded"] = True
            result["seed_job_id"] = (
                seed.get("job_id") if isinstance(seed, dict) else None
            )
            if result["seed_job_id"] and args.wait_jobs:
                result["seed_job"] = await wait_job(
                    api,
                    kb_id,
                    result["seed_job_id"],
                    timeout_seconds=args.job_timeout_seconds,
                    poll_interval_seconds=args.poll_interval_seconds,
                )

        delete_payload = await api.delete_json(f"/kbs/{kb_id}?hard=true")
        result["delete_response"] = delete_payload
        hard_delete_job = await wait_hard_delete_if_queued(
            api, kb_id, delete_payload, args
        )
        if hard_delete_job is not None:
            result["hard_delete_job"] = hard_delete_job

        await api.post_json(
            "/kbs",
            {
                "id": kb_id,
                "name": f"Single-server hard-delete drill verify {kb_id}",
                "description": "Recreated after hard delete to verify workspace cleanup",
                "visibility": "private",
            },
        )
        documents = await api.get_json(f"/kbs/{kb_id}/documents")
        graph_status = await api.get_json(f"/kbs/{kb_id}/graph/status")
        documents_total = _collection_total(documents, "documents", "items")
        node_count = int((graph_status or {}).get("node_count") or 0)
        edge_count = int((graph_status or {}).get("edge_count") or 0)
        result["recreated_workspace"] = {
            "documents_total": documents_total,
            "node_count": node_count,
            "edge_count": edge_count,
        }
        if documents_total or node_count or edge_count:
            raise RuntimeError(
                "Hard-delete cleanup consistency violated after workspace reuse: "
                f"docs={documents_total}, nodes={node_count}, edges={edge_count}"
            )
        return result
    finally:
        cleanup_payload = await api.delete_json(
            f"/kbs/{kb_id}?hard=true", ignore_404=True
        )
        result["cleanup"] = cleanup_payload
        cleanup_job = await wait_hard_delete_if_queued(
            api, kb_id, cleanup_payload, args
        )
        if cleanup_job is not None:
            result["cleanup_job"] = cleanup_job


# ---------------------------------------------------------------------------
# Phase 3.3 — migration / orphan CLI drills (subprocess-invoked)
# ---------------------------------------------------------------------------
#
# These drills invoke the frozen Phase 3.2 operator CLIs
# (``lightrag-migrate-artifacts-to-object`` and ``lightrag-reconcile-orphans``)
# in ``--dry-run --json`` mode against a throwaway working directory. They prove
# the operator CLI surface is installed and emits the documented
# machine-readable plan shape. Apply / resume are NEVER run by this drill (apply
# requires a quiescent system and mutates object storage); they are documented
# below as operator reference.
#
# PG+MinIO consistency manipulation (online/offline write+delete against real
# PostgreSQL + MinIO) is covered by the separate live acceptance harness, NOT by
# this drill script.


@dataclass(slots=True)
class CliResult:
    """Captured subprocess outcome used by the CLI drill runner."""

    returncode: int
    stdout: str
    stderr: str


CliRunner = Callable[
    [list[str], Mapping[str, str] | None, Path | None, float], CliResult
]


def _default_cli_runner(
    command: list[str],
    env: Mapping[str, str] | None,
    cwd: Path | None,
    timeout: float,
) -> CliResult:
    """Run a CLI command via :mod:`subprocess` and capture its output."""

    try:
        proc = subprocess.run(
            command,
            env=dict(env) if env is not None else None,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"CLI executable not found for {command[0]!r}; "
            "run the drill under the project venv (e.g. via `uv run ...`)"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"CLI {command[0]!r} timed out after {timeout}s") from exc
    return CliResult(proc.returncode, proc.stdout, proc.stderr)


def _resolve_cli_entry(console_name: str, module_name: str) -> list[str]:
    """Resolve a console script, falling back to ``python -m <module>``.

    Prefers the installed console script (matches the runbook invocation
    ``lightrag-migrate-artifacts-to-object ...``). When the script is not on
    ``PATH`` (e.g. the drill was launched via ``python scripts/...`` without an
    activated venv), falls back to executing the module entrypoint with the
    current interpreter so the drill is still runnable.
    """

    resolved = shutil.which(console_name)
    if resolved:
        return [resolved]
    return [sys.executable, "-m", module_name]


def _bootstrap_cli_drill_working_dir(base: Path) -> Path:
    """Create a throwaway working dir with an empty ``metadata.sqlite3``.

    The migration/orphan CLIs require ``<working_dir>/metadata/metadata.sqlite3``
    to exist; their ``SQLiteMetadataStore.initialize()`` creates the full schema
    on first connection, so a freshly-touched empty database file is sufficient.
    The temp dir is isolated from the operator's real working directory so this
    drill never risks mutating production metadata.
    """

    working_dir = base / "working_dir"
    metadata_dir = working_dir / "metadata"
    metadata_dir.mkdir(parents=True)
    sqlite_path = metadata_dir / "metadata.sqlite3"
    sqlite_path.touch()
    return working_dir


def _bootstrap_legacy_root_sample(base: Path) -> Path:
    """Create a small explicit ``LABEL=/absolute/root`` sample for migration.

    One regular file is enough to produce a non-empty migration plan
    (``item_count >= 1``) so the drill can assert a real plan was built.
    """

    root = base / "legacy_root"
    root.mkdir(parents=True)
    (root / "sample_artifact.txt").write_text(
        "single-server ops drill migration sample artifact\n",
        encoding="utf-8",
    )
    return root


def _cli_drill_env() -> dict[str, str]:
    """Build the subprocess environment for CLI drills.

    The operator's environment is passed through (so object-storage credentials
    resolve normally) with ``PYTHON_DOTENV_DISABLED`` set to discourage the CLI
    subprocess from reading a project ``.env`` that may target a different
    deployment. Bucket / endpoint / prefix are supplied explicitly as CLI args.
    """

    env = dict(os.environ)
    env["PYTHON_DOTENV_DISABLED"] = "1"
    return env


def _build_migration_cli_command(
    args: argparse.Namespace, working_dir: Path, root: Path
) -> list[str]:
    command = _resolve_cli_entry(
        "lightrag-migrate-artifacts-to-object",
        "lightrag.tools.migrate_artifacts_to_object",
    )
    command += [
        "--working-dir",
        str(working_dir),
        "--bucket",
        args.cli_drill_bucket,
        "--prefix",
        args.cli_drill_prefix,
        "--metadata-backend",
        "sqlite",
        "--dry-run",
        "--json",
        f"legacyRoot={root}",
    ]
    if args.cli_drill_endpoint:
        command += ["--object-storage-endpoint", args.cli_drill_endpoint]
    return command


def _build_orphan_cli_command(args: argparse.Namespace, working_dir: Path) -> list[str]:
    command = _resolve_cli_entry(
        "lightrag-reconcile-orphans",
        "lightrag.tools.reconcile_orphans",
    )
    command += [
        "--working-dir",
        str(working_dir),
        "--bucket",
        args.cli_drill_bucket,
        "--prefix",
        args.cli_drill_prefix,
        "--metadata-backend",
        "sqlite",
        "--min-age-hours",
        "0",
        "--dry-run",
        "--json",
    ]
    if args.cli_drill_endpoint:
        command += ["--object-storage-endpoint", args.cli_drill_endpoint]
    return command


def run_migration_cli_drill(
    args: argparse.Namespace, cli_runner: CliRunner
) -> dict[str, Any]:
    """Phase 3.3 step 2 — invoke ``lightrag-migrate-artifacts-to-object --dry-run``.

    Operator reference — migration apply/resume flow (NOT executed by this drill):
      The dry-run plan produced here only inventories legacy roots and persists a
      durable plan record. To actually upload bytes and rewrite document
      pointers, the operator runs apply against the recorded ``plan_id`` on a
      QUIESCENT system:

        lightrag-migrate-artifacts-to-object \\
            --working-dir <real_working_dir> --bucket <bucket> --prefix kb \\
            --plan-id <plan_id_from_dry_run> --yes \\
            legacyA=/srv/rag/legacy-a

      Apply rejects when any KB mutation job is queued/running/retrying/cancelling
      (online-mutation guard). If apply is interrupted (crash or lease expiry),
      resume with the same LABEL mapping (which may point at a moved absolute
      path):

        lightrag-migrate-artifacts-to-object \\
            --working-dir <real_working_dir> --bucket <bucket> --prefix kb \\
            --plan-id <plan_id_from_dry_run> --yes --resume \\
            legacyA=/srv/rag/legacy-a

      The item state machine is ``planned -> uploaded -> applied -> verified``;
      a crash never strands partially-applied bytes. Inspect ``items_failed`` /
      ``items_blocked`` from the apply summary and re-run ``--resume`` until
      ``items_verified == items_total``.
    """

    with tempfile.TemporaryDirectory(prefix="ops-drill-migrate-") as tmp:
        tmp_path = Path(tmp)
        working_dir = _bootstrap_cli_drill_working_dir(tmp_path)
        root = _bootstrap_legacy_root_sample(tmp_path)
        command = _build_migration_cli_command(args, working_dir, root)
        result = cli_runner(
            command,
            _cli_drill_env(),
            working_dir,
            args.cli_drill_timeout_seconds,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"migration CLI exited {result.returncode}; stderr: "
                f"{result.stderr[:1000]}"
            )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"migration CLI stdout was not valid JSON: {exc}; "
                f"stdout: {result.stdout[:1000]}"
            ) from exc
        if not isinstance(payload, dict) or payload.get("mode") != "plan":
            raise RuntimeError(
                f"migration CLI did not return a plan summary: "
                f"{_json_preview(result.stdout)}"
            )
        plan_id = payload.get("plan_id")
        item_count = payload.get("item_count")
        if not isinstance(plan_id, str) or not plan_id:
            raise RuntimeError(
                f"migration plan JSON missing plan_id: {_json_preview(result.stdout)}"
            )
        if not isinstance(item_count, int) or item_count < 0:
            raise RuntimeError(
                f"migration plan JSON has invalid item_count: {item_count!r}"
            )
        return {
            "executed": True,
            "plan_id": plan_id,
            "item_count": item_count,
            "metadata_backend": payload.get("metadata_backend"),
            "command": command,
            "applied": False,
            "apply_note": (
                "dry-run only; apply requires --plan-id --yes on a quiescent "
                "system (see operator reference in source)"
            ),
        }


def run_orphan_cli_drill(
    args: argparse.Namespace, cli_runner: CliRunner
) -> dict[str, Any]:
    """Phase 3.3 step 3 — invoke ``lightrag-reconcile-orphans --dry-run``.

    Operator reference — orphan reconcile apply/resume flow (NOT executed):
      Apply NEVER deletes objects directly. It enqueues cleanup manifests for
      ``eligible`` orphans; those manifests are later drained by
      ``ArtifactCleanupService`` with verified absence. The categories
      ``referenced``, ``retained``, ``malformed``, ``unknown_owner`` and
      ``too_new`` are report-only.

        lightrag-reconcile-orphans --working-dir <wd> --bucket <bucket> \\
            --plan-id <plan_id> --apply --yes
        # resume after interruption:
        lightrag-reconcile-orphans --working-dir <wd> --bucket <bucket> \\
            --plan-id <plan_id> --apply --yes --resume
        # release retained manifests (deliberate operator action):
        lightrag-reconcile-orphans --working-dir <wd> --bucket <bucket> \\
            --plan-id <plan_id> --apply --yes --release-retained
    """

    with tempfile.TemporaryDirectory(prefix="ops-drill-orphan-") as tmp:
        tmp_path = Path(tmp)
        working_dir = _bootstrap_cli_drill_working_dir(tmp_path)
        command = _build_orphan_cli_command(args, working_dir)
        result = cli_runner(
            command,
            _cli_drill_env(),
            working_dir,
            args.cli_drill_timeout_seconds,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"orphan reconcile CLI exited {result.returncode}; stderr: "
                f"{result.stderr[:1000]}"
            )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"orphan reconcile CLI stdout was not valid JSON: {exc}; "
                f"stdout: {result.stdout[:1000]}"
            ) from exc
        if not isinstance(payload, dict) or payload.get("mode") != "plan":
            raise RuntimeError(
                f"orphan reconcile CLI did not return a plan summary: "
                f"{_json_preview(result.stdout)}"
            )
        plan_id = payload.get("plan_id")
        item_count = payload.get("item_count")
        if not isinstance(plan_id, str) or not plan_id:
            raise RuntimeError(
                f"orphan reconcile plan JSON missing plan_id: "
                f"{_json_preview(result.stdout)}"
            )
        if not isinstance(item_count, int) or item_count < 0:
            raise RuntimeError(
                f"orphan reconcile plan JSON has invalid item_count: {item_count!r}"
            )
        return {
            "executed": True,
            "plan_id": plan_id,
            "item_count": item_count,
            "metadata_backend": payload.get("metadata_backend"),
            "classifications": payload.get("classifications"),
            "command": command,
            "applied": False,
            "apply_note": (
                "dry-run only; apply enqueues cleanup manifests for verified "
                "deletion (see operator reference in source)"
            ),
        }


def record_documented_drills(report: dict[str, Any]) -> None:
    """Phase 3.3 steps 6 & 7 — documented-but-unexecuted drill sections.

    Moved-root operation (step 6):
      Object-mode staging and migration write object authority under
      ``<prefix>/source/generations/`` and ``<prefix>/migrate/<label>/``
      respectively. These object keys are independent of the local filesystem
      layout, so an object-mode deployment survives directory moves. Local-mode
      artifacts are stored under the working directory and do NOT survive a move
      unless the operator migrates them. The migration CLI's explicit
      ``LABEL=/absolute/root`` mapping re-resolves roots at apply time, enabling
      moved-root resume without re-creating the plan. No live moved-root
      manipulation is performed by this drill.

    Production-staging graph/vector/LLM drill (step 7, unexecuted):
      This environment lacks real Neo4j/Milvus/LLM backends, so the full
      production-staging E2E is documented but not run. It is a deployment
      certification prerequisite and must not be claimed as passed by this
      offline drill.
    """

    report["documented_drills"] = {
        "moved_root_operation": {
            "executed": False,
            "summary": (
                "Object-mode staging and migration write object authority under "
                "<prefix>/source/generations/ and <prefix>/migrate/<label>/ "
                "respectively. These object keys are independent of the local "
                "filesystem layout, so an object-mode deployment survives "
                "directory moves. Local-mode artifacts are stored under the "
                "working directory and do NOT survive a move unless migrated. "
                "The migration CLI's explicit LABEL=/absolute/root mapping "
                "re-resolves roots at apply time, enabling moved-root resume "
                "without re-creating the plan. No live moved-root manipulation "
                "is performed by this drill."
            ),
        },
        "production_staging_graph_vector_llm": {
            "executed": False,
            "summary": (
                "Unexecuted — requires a production staging environment with "
                "real Neo4j/Milvus/LLM backends, which are unavailable in this "
                "environment. It is a deployment certification prerequisite and "
                "must not be claimed as passed by this offline drill."
            ),
            "would_cover": [
                "real Neo4j graph writes/reads across the full document lifecycle",
                "real Milvus vector index build and hybrid query",
                "real LLM/embedding extraction and role-specific configuration",
                "full document lifecycle (ingest/parse/index/query/replace/delete)",
                "PostgreSQL + object-store backup/restore consistency",
                "rollback / mixed-version operation restrictions",
                "bucket versioning and noncurrent-version cleanup",
            ],
        },
    }


async def run_checks(
    api: ApiClient,
    args: argparse.Namespace,
    *,
    cli_runner: CliRunner | None = None,
) -> DrillResult:
    report: dict[str, Any] = {
        "started_at": utc_now(),
        "backup_id": args.backup_id,
        "base_url": api.base_url,
        "single_server_scope": True,
    }
    try:
        await probe_base_health(api, report)
        # Phase 3.3 step 1 — artifact_lifecycle health block (fix-16).
        await probe_artifact_lifecycle_health(api, report)
        # Phase 3.3 step 4 — retained/blocked cleanup observability.
        probe_retained_blocked_cleanup(report)
        # Phase 3.3 step 5 — commit-unknown recovery observability.
        probe_commit_unknown_recovery(report)
        # Phase 3.3 steps 6 & 7 — documented-but-unexecuted sections.
        record_documented_drills(report)
        await probe_kbs(api, args, report)
        if args.hard_delete_drill_kb_id:
            report["hard_delete_drill"] = await hard_delete_disposable_kb_drill(
                api, args
            )
        # Phase 3.3 step 2 — migration CLI dry-run drill.
        if getattr(args, "migration_cli_drill", False):
            runner = cli_runner or _default_cli_runner
            report["migration_cli_drill"] = run_migration_cli_drill(args, runner)
        # Phase 3.3 step 3 — orphan reconcile CLI dry-run drill.
        if getattr(args, "orphan_cli_drill", False):
            runner = cli_runner or _default_cli_runner
            report["orphan_cli_drill"] = run_orphan_cli_drill(args, runner)
        report["ok"] = True
        return DrillResult(ok=True, report=report)
    except Exception as exc:
        report["ok"] = False
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        return DrillResult(ok=False, report=report)
    finally:
        report["finished_at"] = utc_now()


def build_base_url(args: argparse.Namespace) -> str:
    if args.base_url:
        return args.base_url.rstrip("/")
    return env_value("LIGHTRAG_BASE_URL", "http://127.0.0.1:9621").rstrip("/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LightRAG single-server restore/operations smoke checks."
    )
    parser.add_argument("--base-url", default=None, help="LightRAG server base URL.")
    parser.add_argument("--api-key", default=None, help="API key for X-API-Key auth.")
    parser.add_argument(
        "--username", default=None, help="JWT login username if API key is not used."
    )
    parser.add_argument(
        "--password", default=None, help="JWT login password if API key is not used."
    )
    parser.add_argument(
        "--backup-id",
        default=env_value("BACKUP_ID"),
        help="Backup/drill identifier recorded in the report.",
    )
    parser.add_argument(
        "--kb-id",
        action="append",
        default=[],
        help="KB id to sample; repeat for multiple KBs. Defaults to first active KBs from /kbs.",
    )
    parser.add_argument(
        "--sample-kb-limit",
        type=int,
        default=3,
        help="Number of KBs to sample when --kb-id is omitted.",
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Query text for optional query/data smoke checks.",
    )
    parser.add_argument(
        "--query-mode",
        default="mix",
        choices=["local", "global", "hybrid", "naive", "mix", "bypass"],
    )
    parser.add_argument(
        "--skip-query",
        action="store_true",
        help="Skip query/data checks for sampled KBs.",
    )
    parser.add_argument(
        "--hard-delete-drill-kb-id",
        default=None,
        help="Disposable KB id used for hard-delete workspace reuse verification.",
    )
    parser.add_argument(
        "--hard-delete-seed-text",
        default="LightRAG single-server hard delete drill seed text.",
        help="Seed text for the disposable KB; pass empty string to skip seeding.",
    )
    parser.add_argument(
        "--hard-delete-auto-index",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Build the seed document before hard delete. Requires working LLM/embedding config.",
    )
    parser.add_argument(
        "--wait-jobs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait for seed/hard-delete jobs that return job IDs.",
    )
    parser.add_argument("--job-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    parser.add_argument("--http-timeout", type=float, default=60.0)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--migration-cli-drill",
        action="store_true",
        help="Run the lightrag-migrate-artifacts-to-object --dry-run --json drill against a throwaway working dir. Requires object-storage connectivity.",
    )
    parser.add_argument(
        "--orphan-cli-drill",
        action="store_true",
        help="Run the lightrag-reconcile-orphans --dry-run --json drill against a throwaway working dir. Requires object-storage connectivity.",
    )
    parser.add_argument(
        "--cli-drill-bucket",
        default="lightrag-kb",
        help="Bucket name passed to the CLI drills (default: lightrag-kb).",
    )
    parser.add_argument(
        "--cli-drill-prefix",
        default="kb",
        help="Object key prefix passed to the CLI drills (default: kb).",
    )
    parser.add_argument(
        "--cli-drill-endpoint",
        default=None,
        help="Optional S3/MinIO endpoint URL passed to the CLI drills.",
    )
    parser.add_argument(
        "--cli-drill-timeout-seconds",
        type=float,
        default=120.0,
        help="Per-CLI subprocess timeout for the CLI drills (default: 120).",
    )
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    base_url = build_base_url(args)
    headers = await authenticate(args, base_url)
    api = ApiClient(base_url, headers, timeout=args.http_timeout)
    try:
        result = await run_checks(api, args)
    finally:
        await api.close()

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        json.dumps(result.report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Report written to {args.report_path}")
    return 0 if result.ok else 1


def main() -> int:
    try:
        return asyncio.run(async_main(parse_args()))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
