"""Smoke-test the LightRAG REST API document-to-QA workflow.

The script targets a running LightRAG API server, or can start one with
``--start-server``. It uploads a PDF with a MinerU filename hint, waits for the
asynchronous pipeline to finish, then probes document, graph, structured
retrieval, normal query, streaming query, and Ollama-compatible discovery
endpoints.

Usage:
    uv run python scripts/run_lightrag_api_workflow.py
    uv run python scripts/run_lightrag_api_workflow.py --start-server
    uv run python scripts/run_lightrag_api_workflow.py --file test_pdf/demo.pdf
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import httpx
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_PATH = PROJECT_ROOT / "temp" / "api_mineru_workflow_report.json"
DEFAULT_QUERY = "请总结刚入库文档的核心内容，并列出关键实体和实体关系。"
TERMINAL_STATUSES = {"processed", "failed"}
SUCCESS_STATUS = "processed"


@dataclass
class ManagedServer:
    process: subprocess.Popen[str]
    log_file: TextIO


def env_value(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def env_int(name: str, default: int) -> int:
    value = env_value(name)
    return default if value is None else int(value)


def normalize_status(value: Any) -> str:
    return str(value or "").strip().lower()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def safe_ascii_stem(stem: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return (normalized or "document")[:80]


def discover_pdf(pdf_dir: Path) -> Path:
    pdfs = sorted(path for path in pdf_dir.iterdir() if path.suffix.lower() == ".pdf")
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}")
    return pdfs[0]


def hinted_upload_name(source: Path, process_options: str) -> str:
    stem = safe_ascii_stem(source.stem)
    hint = "mineru" if not process_options else f"mineru-{process_options}"
    return f"api-smoke-{utc_stamp()}-{stem}.[{hint}]{source.suffix.lower()}"


def build_base_url(args: argparse.Namespace) -> str:
    if args.base_url:
        return args.base_url.rstrip("/")
    port = args.port or env_value("PORT", "9621")
    return f"http://127.0.0.1:{port}".rstrip("/")


def compact_document(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": doc.get("id"),
        "status": doc.get("status"),
        "file_path": doc.get("file_path"),
        "content_length": doc.get("content_length"),
        "chunks_count": doc.get("chunks_count"),
        "error_msg": doc.get("error_msg"),
        "metadata": doc.get("metadata"),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
    }


def summarize_query_data(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data") or {}
    metadata = payload.get("metadata") or {}
    return {
        "status": payload.get("status"),
        "message": payload.get("message"),
        "entities_count": len(data.get("entities") or []),
        "relationships_count": len(data.get("relationships") or []),
        "chunks_count": len(data.get("chunks") or []),
        "references_count": len(data.get("references") or []),
        "first_entity": (data.get("entities") or [{}])[0],
        "first_relationship": (data.get("relationships") or [{}])[0],
        "metadata": metadata,
    }


def summarize_query(payload: dict[str, Any]) -> dict[str, Any]:
    response = str(payload.get("response") or "")
    references = payload.get("references") or []
    return {
        "response_preview": response[:1200],
        "response_length": len(response),
        "references_count": len(references),
        "references": references[:10],
    }


class ApiClient:
    def __init__(self, base_url: str, headers: dict[str, str], timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = headers
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=10.0),
            trust_env=False,
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

    async def post_json(self, path: str, payload: dict[str, Any]) -> Any:
        return await self.request_json("POST", path, json=payload)

    async def delete_json(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        kwargs: dict[str, Any] = {}
        if payload is not None:
            kwargs["json"] = payload
        return await self.request_json("DELETE", path, **kwargs)


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


def start_server(args: argparse.Namespace, base_url: str) -> ManagedServer | None:
    if not args.start_server:
        return None

    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("--start-server requires uv to be available on PATH")

    port = args.port or env_value("PORT", "9621")
    host = args.host or "127.0.0.1"
    log_path = args.server_log.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [uv, "run", "lightrag-server", "--host", host, "--port", str(port)]
    if args.workspace:
        command.extend(["--workspace", args.workspace])
    print(f"Starting LightRAG server for smoke test: {' '.join(command)}")
    print(f"Server log: {log_path}")
    log_file = log_path.open("w", encoding="utf-8")
    child_env = os.environ.copy()
    child_env.setdefault("PYTHONUTF8", "1")
    child_env.setdefault("PYTHONIOENCODING", "utf-8")
    fallback_api_key = (
        child_env.get("LLM_BINDING_API_KEY")
        or child_env.get("VLM_LLM_BINDING_API_KEY")
        or child_env.get("OPENAI_API_KEY")
        or "not_needed"
    )
    child_env.setdefault("OPENAI_API_KEY", fallback_api_key)
    try:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
        )
    except Exception:
        log_file.close()
        raise
    print(f"Waiting for {base_url}/health ...")
    return ManagedServer(process=process, log_file=log_file)


def stop_server(server: ManagedServer | None) -> None:
    if server is None:
        return
    process = server.process
    try:
        if process.poll() is None:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
    finally:
        server.log_file.close()


async def wait_for_health(api: ApiClient, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            health = await api.get_json("/health")
            if isinstance(health, dict) and health.get("status") == "healthy":
                return health
        except Exception as exc:  # server may still be starting
            last_error = exc
        await asyncio.sleep(2)
    raise RuntimeError(f"LightRAG server did not become healthy: {last_error}")


async def upload_pdf(api: ApiClient, source: Path, process_options: str) -> dict[str, Any]:
    upload_name = hinted_upload_name(source, process_options)
    print(f"Uploading {source} as {upload_name}")
    with source.open("rb") as file_obj:
        files = {"file": (upload_name, file_obj, "application/pdf")}
        response = await api.client.post("/documents/upload", files=files)
    if response.status_code >= 400:
        raise RuntimeError(
            f"POST /documents/upload returned HTTP {response.status_code}: "
            f"{response.text[:1000]}"
        )
    payload = response.json()
    payload["upload_name"] = upload_name
    payload["source_file"] = str(source)
    return payload


async def poll_track_status(
    api: ApiClient,
    track_id: str,
    timeout_seconds: int,
    poll_interval: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_payload: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        payload = await api.get_json(f"/documents/track_status/{track_id}")
        last_payload = payload
        documents = payload.get("documents") or []
        statuses = [normalize_status(doc.get("status")) for doc in documents]
        summary = payload.get("status_summary") or {}
        print(f"Track {track_id}: statuses={statuses or summary}")
        if documents and all(status in TERMINAL_STATUSES for status in statuses):
            return payload
        with suppress(Exception):
            pipeline = await api.get_json("/documents/pipeline_status")
            latest = pipeline.get("latest_message") or ""
            busy = pipeline.get("busy")
            print(f"  pipeline busy={busy} latest={latest[:180]}")
        await asyncio.sleep(poll_interval)
    raise TimeoutError(
        f"Timed out waiting for track_id={track_id}. Last payload: "
        f"{json.dumps(last_payload, ensure_ascii=False)[:2000]}"
    )


async def probe_graph(api: ApiClient, report: dict[str, Any]) -> None:
    graph_report: dict[str, Any] = {}
    popular = await api.get_json("/graph/label/popular", params={"limit": 10})
    graph_report["popular_labels"] = popular
    labels = await api.get_json("/graph/label/list")
    graph_report["label_count"] = len(labels) if isinstance(labels, list) else None
    label = (popular or labels or [None])[0] if isinstance(popular or labels, list) else None
    if label:
        graph_report["first_label"] = label
        graph_report["entity_exists"] = await api.get_json(
            "/graph/entity/exists", params={"name": label}
        )
        graph_report["search"] = await api.get_json(
            "/graph/label/search", params={"q": label, "limit": 10}
        )
        subgraph = await api.get_json(
            "/graphs", params={"label": label, "max_depth": 2, "max_nodes": 80}
        )
        graph_report["subgraph_keys"] = sorted(subgraph.keys()) if isinstance(subgraph, dict) else []
        graph_report["subgraph"] = subgraph
    report["graph"] = graph_report


async def probe_query_endpoints(api: ApiClient, args: argparse.Namespace, report: dict[str, Any]) -> None:
    query_payload = {
        "query": args.query,
        "mode": args.mode,
        "top_k": args.top_k,
        "chunk_top_k": args.chunk_top_k,
        "include_references": True,
        "include_chunk_content": args.include_chunk_content,
        "enable_rerank": args.enable_rerank,
    }
    query_payload = {key: value for key, value in query_payload.items() if value is not None}

    query_data = await api.post_json("/query/data", query_payload)
    report["query_data"] = summarize_query_data(query_data)

    query = await api.post_json("/query", query_payload)
    report["query"] = summarize_query(query)

    stream_payload = dict(query_payload)
    stream_payload["stream"] = False
    response = await api.client.post("/query/stream", json=stream_payload)
    if response.status_code >= 400:
        raise RuntimeError(
            f"POST /query/stream returned HTTP {response.status_code}: {response.text[:1000]}"
        )
    stream_lines = [line for line in response.text.splitlines() if line.strip()]
    decoded_lines = [json.loads(line) for line in stream_lines[:5]]
    report["query_stream"] = {
        "content_type": response.headers.get("content-type"),
        "line_count": len(stream_lines),
        "first_lines": decoded_lines,
    }


async def probe_document_endpoints(api: ApiClient, report: dict[str, Any]) -> None:
    report["documents"] = {
        "status_counts": await api.get_json("/documents/status_counts"),
        "pipeline_status": await api.get_json("/documents/pipeline_status"),
        "paginated_latest": await api.post_json(
            "/documents/paginated",
            {"page": 1, "page_size": 10, "sort_field": "updated_at", "sort_direction": "desc"},
        ),
    }


async def probe_ollama_discovery(api: ApiClient, report: dict[str, Any]) -> None:
    report["ollama_compatible"] = {
        "version": await api.get_json("/api/version"),
        "tags": await api.get_json("/api/tags"),
        "ps": await api.get_json("/api/ps"),
    }


async def run(args: argparse.Namespace) -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    base_url = build_base_url(args)
    headers = await authenticate(args, base_url)
    api = ApiClient(base_url, headers, args.http_timeout)
    server_process: ManagedServer | None = None
    report: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "warnings": [],
    }
    if args.workspace:
        report["workspace"] = args.workspace

    try:
        try:
            server_process = start_server(args, base_url)
            health = await wait_for_health(api, args.health_timeout_seconds)
            report["health"] = {
                "status": health.get("status"),
                "auth_mode": health.get("auth_mode"),
                "working_directory": health.get("working_directory"),
                "input_directory": health.get("input_directory"),
                "configuration": health.get("configuration"),
                "pipeline_active": health.get("pipeline_active"),
            }
            print("LightRAG health OK")

            if not args.skip_upload:
                source = args.file.resolve() if args.file else discover_pdf(args.pdf_dir.resolve())
                upload = await upload_pdf(api, source, args.process_options)
                report["upload"] = upload
                track_id = upload.get("track_id")
                if not track_id:
                    raise RuntimeError(f"Upload response did not include track_id: {upload}")
                track_status = await poll_track_status(
                    api, track_id, args.pipeline_timeout_seconds, args.poll_interval
                )
                report["track_status"] = {
                    "track_id": track_id,
                    "total_count": track_status.get("total_count"),
                    "status_summary": track_status.get("status_summary"),
                    "documents": [compact_document(doc) for doc in track_status.get("documents", [])],
                }
                statuses = [normalize_status(doc.get("status")) for doc in track_status.get("documents", [])]
                if any(status != SUCCESS_STATUS for status in statuses):
                    report["warnings"].append("At least one uploaded document did not reach processed status.")
                    return 2
            else:
                report["upload"] = {"skipped": True}

            await probe_document_endpoints(api, report)
            await probe_graph(api, report)
            if not report.get("graph", {}).get("popular_labels"):
                report["warnings"].append(
                    "No popular graph labels returned. Check process options, entity extraction, and LLM logs."
                )
            await probe_query_endpoints(api, args, report)
            await probe_ollama_discovery(api, report)
            return 0
        except Exception as exc:
            report["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"Report written to {args.report_path}")
        await api.close()
        stop_server(server_process)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a LightRAG API smoke test from PDF upload to KB question answering."
    )
    parser.add_argument("--base-url", default=None, help="LightRAG server base URL.")
    parser.add_argument("--host", default=None, help="Host used with --start-server.")
    parser.add_argument("--port", default=None, help="Port used for base URL and --start-server.")
    parser.add_argument("--api-key", default=None, help="API key for X-API-Key auth.")
    parser.add_argument("--username", default=None, help="JWT login username if API key is not used.")
    parser.add_argument("--password", default=None, help="JWT login password if API key is not used.")
    parser.add_argument("--start-server", action="store_true", help="Start lightrag-server before testing.")
    parser.add_argument(
        "--workspace",
        default=env_value("API_WORKFLOW_WORKSPACE"),
        help=(
            "Workspace passed to lightrag-server when --start-server is used. "
            "Use an isolated workspace to avoid existing vector collection dimension conflicts."
        ),
    )
    parser.add_argument(
        "--server-log",
        type=Path,
        default=PROJECT_ROOT / "temp" / "lightrag_api_smoke_server.log",
        help="Log file for --start-server.",
    )
    parser.add_argument("--pdf-dir", type=Path, default=PROJECT_ROOT / "test_pdf")
    parser.add_argument("--file", type=Path, default=None, help="PDF file to upload.")
    parser.add_argument(
        "--process-options",
        default="iteP",
        help="MinerU process options used in filename hint, e.g. iteP, R, P, iteP!.",
    )
    parser.add_argument("--skip-upload", action="store_true", help="Only probe existing KB endpoints.")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument(
        "--mode",
        default="mix",
        choices=["local", "global", "hybrid", "naive", "mix", "bypass"],
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--chunk-top-k", type=int, default=5)
    parser.add_argument("--enable-rerank", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--include-chunk-content", action="store_true")
    parser.add_argument("--health-timeout-seconds", type=int, default=env_int("API_HEALTH_TIMEOUT", 180))
    parser.add_argument("--pipeline-timeout-seconds", type=int, default=env_int("API_PIPELINE_TIMEOUT", 1800))
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--http-timeout", type=float, default=180.0)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main() -> int:
    try:
        return asyncio.run(run(parse_args()))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
