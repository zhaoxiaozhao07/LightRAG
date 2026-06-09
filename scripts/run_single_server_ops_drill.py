"""Run single-server operational smoke checks for LightRAG.

The script is intended for two production operations that are in scope for the
current deployment model:

1. post-restore validation after following ``docs/生产级后端备份恢复Runbook.md``;
2. optional hard-delete workspace-reuse cleanup drill on a disposable KB.

It targets exactly one LightRAG server process on one server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

    async def post_json(self, path: str, payload: dict[str, Any] | None = None, **kwargs: Any) -> Any:
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

    async with httpx.AsyncClient(base_url=base_url, timeout=15.0, trust_env=False) as client:
        response = await client.post(
            "/login",
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Login failed: HTTP {response.status_code}: {response.text}")
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
            if str(payload.get("status") or "").lower() in {"succeeded", "failed", "cancelled"}:
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
        "working_directory": health.get("working_directory") if isinstance(health, dict) else None,
        "input_directory": health.get("input_directory") if isinstance(health, dict) else None,
    }
    if report["health"]["status"] != "healthy":
        raise RuntimeError(f"/health did not report healthy: {_json_preview(health)}")

    metrics = await api.client.get("/metrics")
    if metrics.status_code >= 400:
        raise RuntimeError(f"GET /metrics returned HTTP {metrics.status_code}: {metrics.text[:1000]}")
    report["metrics"] = {
        "content_type": metrics.headers.get("content-type"),
        "has_kb_total": "lightrag_kb_total" in metrics.text,
        "has_http_histogram": "lightrag_http_request_duration_seconds" in metrics.text,
    }
    if not report["metrics"]["has_kb_total"]:
        raise RuntimeError("/metrics did not include lightrag_kb_total")


async def probe_kbs(api: ApiClient, args: argparse.Namespace, report: dict[str, Any]) -> None:
    payload = await api.get_json("/kbs")
    kbs = _collection_items(payload, "items", "kbs", "knowledge_bases")
    report["kbs"] = {"total": _collection_total(payload, "items", "kbs", "knowledge_bases")}

    requested = list(args.kb_id or [])
    if not requested:
        requested = [str(item.get("id")) for item in kbs[: args.sample_kb_limit] if item.get("id")]
    report["sampled_kbs"] = []
    for kb_id in requested:
        sample: dict[str, Any] = {"kb_id": kb_id}
        sample["status"] = await api.get_json(f"/kbs/{kb_id}/status")
        documents = await api.get_json(f"/kbs/{kb_id}/documents")
        sample["documents_total"] = _collection_total(documents, "documents", "items")
        graph_status = await api.get_json(f"/kbs/{kb_id}/graph/status")
        sample["graph"] = {
            "node_count": graph_status.get("node_count") if isinstance(graph_status, dict) else None,
            "edge_count": graph_status.get("edge_count") if isinstance(graph_status, dict) else None,
            "is_truncated": graph_status.get("is_truncated") if isinstance(graph_status, dict) else None,
        }
        if not args.skip_query:
            query = await api.post_json(
                f"/kbs/{kb_id}/query/data",
                {"query": args.query, "mode": args.query_mode, "top_k": 5, "chunk_top_k": 5},
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
            result["seed_job_id"] = seed.get("job_id") if isinstance(seed, dict) else None
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
        hard_delete_job = await wait_hard_delete_if_queued(api, kb_id, delete_payload, args)
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
        cleanup_job = await wait_hard_delete_if_queued(api, kb_id, cleanup_payload, args)
        if cleanup_job is not None:
            result["cleanup_job"] = cleanup_job


async def run_checks(api: ApiClient, args: argparse.Namespace) -> DrillResult:
    report: dict[str, Any] = {
        "started_at": utc_now(),
        "backup_id": args.backup_id,
        "base_url": api.base_url,
        "single_server_scope": True,
    }
    try:
        await probe_base_health(api, report)
        await probe_kbs(api, args, report)
        if args.hard_delete_drill_kb_id:
            report["hard_delete_drill"] = await hard_delete_disposable_kb_drill(api, args)
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
    parser.add_argument("--username", default=None, help="JWT login username if API key is not used.")
    parser.add_argument("--password", default=None, help="JWT login password if API key is not used.")
    parser.add_argument("--backup-id", default=env_value("BACKUP_ID"), help="Backup/drill identifier recorded in the report.")
    parser.add_argument("--kb-id", action="append", default=[], help="KB id to sample; repeat for multiple KBs. Defaults to first active KBs from /kbs.")
    parser.add_argument("--sample-kb-limit", type=int, default=3, help="Number of KBs to sample when --kb-id is omitted.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Query text for optional query/data smoke checks.")
    parser.add_argument("--query-mode", default="mix", choices=["local", "global", "hybrid", "naive", "mix", "bypass"])
    parser.add_argument("--skip-query", action="store_true", help="Skip query/data checks for sampled KBs.")
    parser.add_argument("--hard-delete-drill-kb-id", default=None, help="Disposable KB id used for hard-delete workspace reuse verification.")
    parser.add_argument("--hard-delete-seed-text", default="LightRAG single-server hard delete drill seed text.", help="Seed text for the disposable KB; pass empty string to skip seeding.")
    parser.add_argument("--hard-delete-auto-index", action=argparse.BooleanOptionalAction, default=False, help="Build the seed document before hard delete. Requires working LLM/embedding config.")
    parser.add_argument("--wait-jobs", action=argparse.BooleanOptionalAction, default=True, help="Wait for seed/hard-delete jobs that return job IDs.")
    parser.add_argument("--job-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    parser.add_argument("--http-timeout", type=float, default=60.0)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
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
