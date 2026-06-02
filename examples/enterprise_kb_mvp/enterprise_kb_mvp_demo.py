from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import httpx
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "Missing dependency: httpx. Run with `uv run ...` from the project root."
    ) from exc


def _prefer_utf8_stdio() -> None:
    if os.name != "nt":
        return
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


_prefer_utf8_stdio()


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "模拟文件"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "examples" / "enterprise_kb_mvp" / "runs"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_SERVER = "http://127.0.0.1:9621"
DEFAULT_API_KEY = os.environ.get("LIGHTRAG_API_KEY", "sk-123456")
DEFAULT_KB_ID = "enterprise_mvp_demo"
DEFAULT_KB_NAME = "企业知识库 MVP 模拟"
DEFAULT_PARSER_ENGINE = "mineru"
DEFAULT_PROCESS_OPTIONS = "iteP"
DEFAULT_QUERY = "请总结企业知识库中这些资料的核心主题、关键材料/工艺、实体关系和可落地价值。"
DEFAULT_ISOLATION_QUERY = "请回答：这是什么知识库？如果没有资料，请说明无法从知识库中找到答案。"
DEFAULT_HTTP_TIMEOUT = 300.0
DEFAULT_JOB_TIMEOUT = 3600.0
JOB_WAIT_SERVER_WINDOW = 600.0
JOB_WAIT_HTTP_GRACE = 10.0
ACTIVE_JOB_STATES = {"queued", "running", "retrying", "cancelling"}
TERMINAL_JOB_STATES = {"succeeded", "failed", "cancelled"}
DELETE_STRATEGIES = ("safe", "rebuild_doc_scope", "rebuild_kb", "rebuild_subgraph")
SAFE_RUN_ID_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)
SUPPORTED_SUFFIXES = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".html",
}
SENSITIVE_ENV_KEYS = {
    "AUTH_ACCOUNTS",
    "LIGHTRAG_API_KEY",
    "LIGHTRAG_KB_POSTGRES_DSN",
    "LIGHTRAG_KB_POSTGRES_PASSWORD",
    "LIGHTRAG_OBJECT_STORAGE_ACCESS_KEY_ID",
    "LIGHTRAG_OBJECT_STORAGE_SECRET_ACCESS_KEY",
    "LLM_BINDING_API_KEY",
    "OPENAI_API_KEY",
    "RERANK_BINDING_API_KEY",
    "TOKEN_SECRET",
    "VLM_LLM_BINDING_API_KEY",
}
ENV_SNAPSHOT_KEYS = (
    "HOST",
    "PORT",
    "WORKING_DIR",
    "INPUT_DIR",
    "WORKSPACE",
    "TOP_K",
    "CHUNK_TOP_K",
    "MAX_ENTITY_TOKENS",
    "MAX_RELATION_TOKENS",
    "MAX_TOTAL_TOKENS",
    "RELATED_CHUNK_NUMBER",
    "RERANK_BINDING",
    "RERANK_MODEL",
    "RERANK_BINDING_HOST",
    "RERANK_BY_DEFAULT",
    "MIN_RERANK_SCORE",
    "MAX_ASYNC_RERANK",
    "RERANK_TIMEOUT",
    "COSINE_THRESHOLD",
    "LIGHTRAG_PARSER",
    "MINERU_API_MODE",
    "MINERU_LOCAL_ENDPOINT",
    "MINERU_LOCAL_BACKEND",
    "MINERU_VLM_URL",
    "MINERU_VLM_MODEL",
    "MINERU_LOCAL_PARSE_METHOD",
    "MINERU_LOCAL_IMAGE_ANALYSIS",
    "MINERU_LANGUAGE",
    "MINERU_ENABLE_TABLE",
    "MINERU_ENABLE_FORMULA",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP_SIZE",
    "MAX_PARALLEL_INSERT",
    "MAX_PARALLEL_PARSE_MINERU",
    "MAX_PARALLEL_ANALYZE",
    "LLM_BINDING",
    "LLM_BINDING_HOST",
    "LLM_MODEL",
    "MAX_ASYNC",
    "EXTRACT_MAX_ASYNC_LLM",
    "KEYWORD_MAX_ASYNC_LLM",
    "QUERY_MAX_ASYNC_LLM",
    "OPENAI_LLM_MAX_TOKENS",
    "VLM_LLM_BINDING",
    "VLM_LLM_BINDING_HOST",
    "VLM_LLM_MODEL",
    "VLM_MAX_IMAGE_BYTES",
    "VLM_MAX_ASYNC_LLM",
    "EMBEDDING_BINDING",
    "EMBEDDING_BINDING_HOST",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "EMBEDDING_TOKEN_LIMIT",
    "EMBEDDING_SEND_DIM",
    "EMBEDDING_USE_BASE64",
    "EMBEDDING_FUNC_MAX_ASYNC",
    "EMBEDDING_BATCH_NUM",
    "EMBEDDING_TIMEOUT",
    "LIGHTRAG_KB_METADATA_BACKEND",
    "LIGHTRAG_KB_POSTGRES_HOST",
    "LIGHTRAG_KB_POSTGRES_PORT",
    "LIGHTRAG_KB_POSTGRES_USER",
    "LIGHTRAG_KB_POSTGRES_DATABASE",
    "LIGHTRAG_KB_POSTGRES_POOL_MIN_SIZE",
    "LIGHTRAG_KB_POSTGRES_POOL_MAX_SIZE",
    "LIGHTRAG_KV_STORAGE",
    "LIGHTRAG_DOC_STATUS_STORAGE",
    "LIGHTRAG_GRAPH_STORAGE",
    "LIGHTRAG_VECTOR_STORAGE",
    "LIGHTRAG_OBJECT_STORAGE",
    "LIGHTRAG_OBJECT_STORAGE_ENDPOINT",
    "LIGHTRAG_OBJECT_STORAGE_BUCKET",
    "LIGHTRAG_OBJECT_STORAGE_USE_SSL",
    "LIGHTRAG_OBJECT_STORAGE_REGION",
    "LIGHTRAG_OBJECT_STORAGE_PREFIX",
    "LIGHTRAG_OBJECT_STORAGE_CREATE_BUCKET",
    "LIGHTRAG_OBJECT_STORAGE_DISABLE_EXPECT_HEADER",
    "MILVUS_URI",
    "MILVUS_DB_NAME",
    "MILVUS_USER",
    "MILVUS_PASSWORD",
    "MILVUS_TOKEN",
)


@dataclass(frozen=True)
class SourceFile:
    path: Path
    relative_key: str
    sha256: str
    size_bytes: int
    content_type: str


class EnterpriseKBClient:
    def __init__(self, base_url: str, api_key: str, *, timeout: float):
        headers = {"X-API-Key": api_key} if api_key else {}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout
        )

    def close(self) -> None:
        self._client.close()

    def health(self) -> dict[str, Any]:
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json()

    def get_kb(self, kb_id: str) -> dict[str, Any] | None:
        response = self._client.get(f"/kbs/{kb_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def create_kb(self, kb_id: str, name: str, description: str) -> dict[str, Any]:
        response = self._client.post(
            "/kbs",
            json={
                "id": kb_id,
                "name": name,
                "description": description,
                "visibility": "private",
            },
        )
        response.raise_for_status()
        return response.json()

    def ensure_kb(self, kb_id: str, name: str, description: str) -> dict[str, Any]:
        existing = self.get_kb(kb_id)
        if existing is not None:
            return existing
        return self.create_kb(kb_id, name, description)

    def hard_delete_kb(self, kb_id: str) -> dict[str, Any] | None:
        """Hard-delete a KB so the next run starts from a clean slate.

        Calls ``DELETE /kbs/{kb_id}?hard=true`` which drops on-disk LightRAG
        storage, parser inputs/artifacts, MinIO objects, and metadata rows
        (kb_documents / kb_jobs / kb_document_artifacts / kb_config_versions
        / kb_catalog). Returns the deleted KB record on success, or ``None``
        when the KB did not exist (404). Other HTTP errors propagate.
        """
        response = self._client.delete(f"/kbs/{kb_id}", params={"hard": "true"})
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def status(self, kb_id: str) -> dict[str, Any]:
        response = self._client.get(f"/kbs/{kb_id}/status")
        response.raise_for_status()
        return response.json()

    def create_config(
        self, kb_id: str, config: dict[str, Any], *, created_by: str
    ) -> dict[str, Any]:
        response = self._client.post(
            f"/kbs/{kb_id}/configs",
            json={"config": config, "created_by": created_by},
        )
        response.raise_for_status()
        return response.json()

    def activate_config(self, kb_id: str, config_id: str) -> dict[str, Any]:
        response = self._client.post(f"/kbs/{kb_id}/configs/{config_id}:activate")
        response.raise_for_status()
        return response.json()

    def list_configs(self, kb_id: str) -> dict[str, Any]:
        response = self._client.get(f"/kbs/{kb_id}/configs")
        response.raise_for_status()
        return response.json()

    def sync_documents(
        self,
        kb_id: str,
        files: list[SourceFile],
        *,
        parser_engine: str,
        process_options: str,
        idempotency_key: str,
        auto_parse: bool = True,
        auto_index: bool = True,
    ) -> dict[str, Any]:
        params = {
            "auto_parse": str(auto_parse).lower(),
            "auto_index": str(auto_index).lower(),
            "parser_engine": parser_engine,
            "process_options": process_options,
            "idempotency_key": idempotency_key,
        }
        file_handles = []
        try:
            multipart_files = []
            source_keys: list[str] = []
            for source in files:
                handle = source.path.open("rb")
                file_handles.append(handle)
                multipart_files.append(
                    (
                        "files",
                        (source.path.name, handle, source.content_type),
                    )
                )
                source_keys.append(source.relative_key)
            response = self._client.post(
                f"/kbs/{kb_id}/documents:sync",
                params=params,
                data={"source_keys": source_keys},
                files=multipart_files,
            )
        finally:
            for handle in file_handles:
                handle.close()
        response.raise_for_status()
        return response.json()

    def upload_document(
        self,
        kb_id: str,
        source: SourceFile,
        *,
        parser_engine: str,
        process_options: str,
    ) -> dict[str, Any]:
        params = {
            "auto_parse": "false",
            "parser_engine": parser_engine,
            "process_options": process_options,
        }
        with source.path.open("rb") as handle:
            response = self._client.post(
                f"/kbs/{kb_id}/documents:upload",
                params=params,
                files={"files": (source.path.name, handle, source.content_type)},
            )
        response.raise_for_status()
        return response.json()

    def parse_document(
        self,
        kb_id: str,
        document_id: str,
        *,
        parser_engine: str,
        process_options: str,
    ) -> dict[str, Any]:
        response = self._client.post(
            f"/kbs/{kb_id}/documents/{document_id}:parse",
            json={"engine": parser_engine, "process_options": process_options},
        )
        response.raise_for_status()
        return response.json()

    def build_kg(self, kb_id: str, document_id: str) -> dict[str, Any]:
        response = self._client.post(
            f"/kbs/{kb_id}/documents/{document_id}:build-kg", json={}
        )
        response.raise_for_status()
        return response.json()

    def wait_for_job(
        self, kb_id: str, job_id: str, *, timeout_seconds: float
    ) -> dict[str, Any]:
        deadline = time.monotonic() + max(timeout_seconds, 0.1)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"job {job_id!r} did not finish within {timeout_seconds:.1f}s"
                )
            wait_window = min(remaining, JOB_WAIT_SERVER_WINDOW)
            response = self._client.post(
                f"/kbs/{kb_id}/jobs/{job_id}:wait",
                params={"timeout_seconds": wait_window},
                timeout=wait_window + JOB_WAIT_HTTP_GRACE,
            )
            wait_detail = _wait_timeout_detail(response)
            if wait_detail is not None:
                status = wait_detail.get("current_status")
                if status in ACTIVE_JOB_STATES and deadline - time.monotonic() > 0:
                    print(f"[wait] {job_id} still {status}; continuing")
                    continue
            response.raise_for_status()
            return response.json()

    def list_documents(self, kb_id: str, *, limit: int = 100) -> dict[str, Any]:
        response = self._client.get(f"/kbs/{kb_id}/documents", params={"limit": limit})
        response.raise_for_status()
        return response.json()

    def list_jobs(self, kb_id: str, *, limit: int = 50) -> dict[str, Any]:
        response = self._client.get(f"/kbs/{kb_id}/jobs", params={"limit": limit})
        response.raise_for_status()
        return response.json()

    def list_dead_letters(self, kb_id: str, *, limit: int = 50) -> dict[str, Any]:
        response = self._client.get(
            f"/kbs/{kb_id}/jobs/dead-letter", params={"limit": limit}
        )
        response.raise_for_status()
        return response.json()

    def list_artifacts(self, kb_id: str, document_id: str) -> dict[str, Any]:
        response = self._client.get(f"/kbs/{kb_id}/documents/{document_id}/artifacts")
        response.raise_for_status()
        return response.json()

    def delete_document(
        self,
        kb_id: str,
        document_id: str,
        *,
        delete_source_file: bool,
        delete_artifacts: bool,
        delete_llm_cache: bool,
        strategy: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        response = self._client.delete(
            f"/kbs/{kb_id}/documents/{document_id}",
            params={
                "delete_source_file": str(delete_source_file).lower(),
                "delete_artifacts": str(delete_artifacts).lower(),
                "delete_llm_cache": str(delete_llm_cache).lower(),
                "delete_graph_orphans": "true",
                "strategy": strategy,
                "idempotency_key": idempotency_key,
            },
        )
        response.raise_for_status()
        return response.json()

    def batch_delete_documents(
        self,
        kb_id: str,
        document_ids: list[str],
        *,
        delete_source_file: bool,
        delete_artifacts: bool,
        delete_llm_cache: bool,
        strategy: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        response = self._client.post(
            f"/kbs/{kb_id}/documents:batch-delete",
            json={
                "document_ids": document_ids,
                "delete_source_file": delete_source_file,
                "delete_artifacts": delete_artifacts,
                "delete_llm_cache": delete_llm_cache,
                "delete_graph_orphans": True,
                "strategy": strategy,
                "idempotency_key": idempotency_key,
            },
        )
        response.raise_for_status()
        return response.json()

    def graph_status(self, kb_id: str) -> dict[str, Any]:
        response = self._client.get(f"/kbs/{kb_id}/graph/status")
        response.raise_for_status()
        return response.json()

    def graph_entities(self, kb_id: str, *, limit: int = 20) -> dict[str, Any]:
        response = self._client.get(
            f"/kbs/{kb_id}/graph/entities", params={"limit": limit}
        )
        response.raise_for_status()
        return response.json()

    def graph_relations(self, kb_id: str, *, limit: int = 20) -> dict[str, Any]:
        response = self._client.get(
            f"/kbs/{kb_id}/graph/relations", params={"limit": limit}
        )
        response.raise_for_status()
        return response.json()

    def query(
        self,
        kb_id: str,
        question: str,
        *,
        mode: str,
        include_references: bool,
        include_chunk_content: bool,
        top_k: int,
        chunk_top_k: int,
        doc_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "query": question,
            "mode": mode,
            "top_k": top_k,
            "chunk_top_k": chunk_top_k,
            "include_references": include_references,
            "include_chunk_content": include_chunk_content,
            "stream": False,
        }
        if doc_ids is not None:
            body["filters"] = {"doc_ids": doc_ids}
        response = self._client.post(f"/kbs/{kb_id}/query", json=body, timeout=600.0)
        response.raise_for_status()
        return response.json()

    def query_data(
        self,
        kb_id: str,
        question: str,
        *,
        mode: str,
        top_k: int,
        chunk_top_k: int,
    ) -> dict[str, Any]:
        response = self._client.post(
            f"/kbs/{kb_id}/query/data",
            json={
                "query": question,
                "mode": mode,
                "top_k": top_k,
                "chunk_top_k": chunk_top_k,
                "include_references": True,
                "include_chunk_content": False,
                "stream": False,
            },
            timeout=600.0,
        )
        response.raise_for_status()
        return response.json()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enterprise-style KB MVP API flow: KB config, upload/sync, parse, KG build, MinIO, vector DB, graph, query, and isolation checks."
    )
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kb-id", default=DEFAULT_KB_ID)
    parser.add_argument("--kb-name", default=DEFAULT_KB_NAME)
    parser.add_argument(
        "--kb-description",
        default="企业知识库落地 MVP：模拟文档接入、对象存储、解析、实体关系抽取、向量化、图谱构建和问答隔离。",
    )
    parser.add_argument("--parser-engine", default=DEFAULT_PARSER_ENGINE)
    parser.add_argument("--process-options", default=DEFAULT_PROCESS_OPTIONS)
    parser.add_argument("--mode", default="mix", choices=("local", "global", "hybrid", "naive", "mix", "bypass"))
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--isolation-query", default=DEFAULT_ISOLATION_QUERY)
    parser.add_argument("--isolation-kb-id", default="enterprise_mvp_isolation_empty")
    parser.add_argument("--isolation-kb-name", default="企业知识库隔离空白对照")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--chunk-top-k", type=int, default=20)
    parser.add_argument("--http-timeout", type=float, default=DEFAULT_HTTP_TIMEOUT)
    parser.add_argument("--job-timeout", type=float, default=DEFAULT_JOB_TIMEOUT)
    parser.add_argument("--include-references", action="store_true", default=True)
    parser.add_argument("--no-include-references", dest="include_references", action="store_false")
    parser.add_argument("--include-chunk-content", action="store_true")
    parser.add_argument(
        "--manual-flow",
        action="store_true",
        help="Use upload -> parse -> build one document at a time instead of documents:sync auto_parse+auto_index.",
    )
    parser.add_argument("--force-rebuild", action="store_true", help="After sync, force build_kg for every discovered document.")
    parser.add_argument(
        "--delete-test",
        action="store_true",
        help="After graph inspection, pause for interactive document deletion testing.",
    )
    parser.add_argument(
        "--delete-source-file",
        action="store_true",
        help="Also delete the original uploaded source object/file during delete test.",
    )
    parser.add_argument(
        "--delete-artifacts",
        action="store_true",
        help="Also delete parser artifacts during delete test.",
    )
    parser.add_argument(
        "--delete-llm-cache",
        action="store_true",
        help="Also clear LLM cache entries for selected documents during delete test.",
    )
    parser.add_argument(
        "--delete-strategy",
        default="safe",
        choices=DELETE_STRATEGIES,
        help="Graph cleanup strategy passed to the delete API.",
    )
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-query", action="store_true")
    parser.add_argument("--skip-isolation-check", action="store_true")
    parser.add_argument("--skip-config", action="store_true")
    parser.add_argument(
        "--max-files", type=int, default=0, help="0 means all supported files."
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional stable run id for idempotency keys and report file names.",
    )
    parser.add_argument(
        "--reset-kb",
        choices=("ask", "yes", "no"),
        default="ask",
        help=(
            "Hard-delete the KB (and the isolation control KB unless "
            "--skip-isolation-check) before running. 'ask' (default) prompts "
            "interactively on stdin; 'yes' resets without asking; 'no' skips. "
            "Reset calls DELETE /kbs/{kb_id}?hard=true and clears落盘文件、"
            "Postgres 记录和对象存储 -- 也是检验 KB 硬删接口的端到端入口。"
        ),
    )
    return parser.parse_args()


def confirm_reset_kb(args: argparse.Namespace) -> bool:
    """Decide whether to hard-delete the demo KB(s) before the run.

    Honors ``--reset-kb yes/no`` without prompting; otherwise asks once on
    stdin. Falls back to False when stdin is not a TTY so non-interactive
    runs cannot accidentally wipe a KB without an explicit ``yes``.
    """
    choice = getattr(args, "reset_kb", "ask")
    if choice == "yes":
        return True
    if choice == "no":
        return False
    if not sys.stdin.isatty():
        print(
            "[warn] --reset-kb=ask but stdin is not a TTY; skipping reset. "
            "Pass --reset-kb=yes to reset in non-interactive runs."
        )
        return False
    targets = [args.kb_id]
    if not args.skip_isolation_check:
        targets.append(args.isolation_kb_id)
    print()
    print("[input] 是否全部重建知识库（硬删落盘文件、Postgres 记录和对象存储）？")
    for kb_id in targets:
        print(f"        - {kb_id}")
    print("        将调用 DELETE /kbs/{kb_id}?hard=true")
    try:
        answer = input("        输入 yes / y 确认，其他键跳过：").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def _hard_reset_demo_kbs(
    client: EnterpriseKBClient, args: argparse.Namespace
) -> dict[str, Any]:
    """Hard-delete the main demo KB and (when not skipped) the isolation KB.

    Returns a summary dict for the run report; errors propagate so the demo
    stops loudly rather than continuing on stale state.
    """
    summary: dict[str, Any] = {"performed": True, "targets": {}}
    plan: list[tuple[str, str]] = [("main", args.kb_id)]
    if not args.skip_isolation_check:
        plan.append(("isolation", args.isolation_kb_id))
    for label, kb_id in plan:
        print(f"[reset] hard-delete {label} kb_id={kb_id!r}")
        deleted = client.hard_delete_kb(kb_id)
        state = "deleted" if deleted else "not_found"
        summary["targets"][label] = {
            "kb_id": kb_id,
            "state": state,
            "record": deleted,
        }
        print(f"[reset] {label} kb_id={kb_id!r}: {state}")
    return summary


def run(args: argparse.Namespace) -> int:
    started = time.time()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = (
        normalize_run_id(args.run_id)
        if args.run_id
        else datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )

    env_snapshot = load_env_snapshot(args.env_file)
    files = discover_source_files(source_dir, max_files=args.max_files)
    if not files and not args.skip_ingest:
        raise SystemExit(f"No supported files found under: {source_dir}")

    print(f"[info] source_dir={source_dir}")
    print(f"[info] files={len(files)} kb_id={args.kb_id!r} run_id={run_id}")

    client = EnterpriseKBClient(args.server, args.api_key, timeout=args.http_timeout)
    report: dict[str, Any] = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "server": args.server,
        "source_dir": str(source_dir),
        "kb_id": args.kb_id,
        "source_files": [source_file_to_dict(item) for item in files],
        "env_snapshot": env_snapshot,
        "steps": {},
    }
    try:
        health = client.health()
        report["steps"]["health"] = health
        print(
            f"[ok] health object_storage={health.get('configuration', {}).get('object_storage')} "
            f"metadata={health.get('configuration', {}).get('kb_metadata_backend')}"
        )

        if confirm_reset_kb(args):
            report["steps"]["reset"] = _hard_reset_demo_kbs(client, args)
        else:
            report["steps"]["reset"] = {"performed": False}

        kb = client.ensure_kb(args.kb_id, args.kb_name, args.kb_description)
        report["steps"]["kb"] = kb
        print(f"[ok] kb id={kb['id']} workspace={kb['workspace']}")

        if not args.skip_config:
            config_body = build_enterprise_config(args, env_snapshot)
            config = client.create_config(
                args.kb_id, config_body, created_by="enterprise-kb-mvp-demo"
            )
            activated = client.activate_config(args.kb_id, config["id"])
            report["steps"]["config"] = {"created": config, "activated": activated}
            print(
                f"[ok] config activated id={activated['id']} version={activated.get('version')}"
            )
        else:
            print("[skip] config creation/activation")

        if not args.skip_ingest:
            if args.manual_flow:
                documents = run_manual_flow(client, args, files, run_id, report)
            else:
                documents = run_sync_flow(client, args, files, run_id, report)
            if args.force_rebuild:
                force_build_all(client, args.kb_id, documents, args.job_timeout, report)
        else:
            print("[skip] ingest")

        documents_payload = client.list_documents(args.kb_id, limit=200)
        report["steps"]["documents"] = documents_payload
        ready_documents = [
            item for item in documents_payload.get("documents", []) if item.get("status") == "ready"
        ]
        if not ready_documents and not args.skip_query:
            print(
                "[warn] no ready documents available; "
                "query will be skipped unless documents become ready later"
            )
        print(
            f"[ok] documents total={documents_payload.get('total')} ready={len(ready_documents)}"
        )

        artifact_summary = collect_artifacts(client, args.kb_id, ready_documents)
        report["steps"]["artifacts"] = artifact_summary
        print(f"[ok] artifacts checked for {len(artifact_summary)} document(s)")

        graph_status = client.graph_status(args.kb_id)
        graph_entities = client.graph_entities(args.kb_id, limit=20)
        graph_relations = client.graph_relations(args.kb_id, limit=20)
        report["steps"]["graph"] = {
            "status": graph_status,
            "entities_sample": graph_entities,
            "relations_sample": graph_relations,
        }
        print(
            f"[ok] graph nodes={graph_status.get('node_count')} edges={graph_status.get('edge_count')}"
        )

        if args.delete_test:
            delete_summary = run_delete_test(
                client, args, documents_payload, run_id, report
            )
            report["steps"]["delete_test"] = delete_summary
            documents_payload = client.list_documents(args.kb_id, limit=200)
            report["steps"]["documents_after_delete"] = documents_payload
            ready_documents = [
                item
                for item in documents_payload.get("documents", [])
                if item.get("status") == "ready"
            ]
            if delete_summary.get("deleted_count", 0) > 0:
                graph_status_after_delete = client.graph_status(args.kb_id)
                graph_entities_after_delete = client.graph_entities(args.kb_id, limit=20)
                graph_relations_after_delete = client.graph_relations(args.kb_id, limit=20)
                report["steps"]["graph_after_delete"] = {
                    "status": graph_status_after_delete,
                    "entities_sample": graph_entities_after_delete,
                    "relations_sample": graph_relations_after_delete,
                }
                print(
                    "[ok] after delete "
                    f"documents={documents_payload.get('total')} ready={len(ready_documents)} "
                    f"nodes={graph_status_after_delete.get('node_count')} "
                    f"edges={graph_status_after_delete.get('edge_count')}"
                )

        report["steps"]["jobs"] = client.list_jobs(args.kb_id, limit=100)
        report["steps"]["dead_letters"] = client.list_dead_letters(args.kb_id, limit=50)
        report["steps"]["configs"] = client.list_configs(args.kb_id)
        report["steps"]["status"] = client.status(args.kb_id)

        if not args.skip_query:
            if not ready_documents:
                report["steps"]["query"] = {
                    "skipped": True,
                    "reason": "no_ready_documents_after_delete_test"
                    if args.delete_test
                    else "no_ready_documents",
                }
                print("[skip] query: no ready documents remain")
            else:
                query_result = client.query(
                    args.kb_id,
                    args.query,
                    mode=args.mode,
                    include_references=args.include_references,
                    include_chunk_content=args.include_chunk_content,
                    top_k=args.top_k,
                    chunk_top_k=args.chunk_top_k,
                )
                doc_scoped_query = client.query(
                    args.kb_id,
                    args.query,
                    mode=args.mode,
                    include_references=True,
                    include_chunk_content=False,
                    top_k=args.top_k,
                    chunk_top_k=args.chunk_top_k,
                    doc_ids=[ready_documents[0]["id"]],
                )
                query_data = client.query_data(
                    args.kb_id,
                    args.query,
                    mode=args.mode,
                    top_k=args.top_k,
                    chunk_top_k=args.chunk_top_k,
                )
                report["steps"]["query"] = {
                    "question": args.query,
                    "result": query_result,
                    "doc_scoped_result": doc_scoped_query,
                    "data": query_data,
                }
                print(
                    f"[ok] query mode={query_result.get('mode')} refs={len(query_result.get('references') or [])}"
                )
        else:
            print("[skip] query")

        if not args.skip_isolation_check:
            isolation = run_isolation_check(client, args, report)
            report["steps"]["isolation"] = isolation
            print(f"[ok] isolation kb={args.isolation_kb_id}")
        else:
            print("[skip] isolation check")

        elapsed = time.time() - started
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["elapsed_seconds"] = round(elapsed, 3)
        report_path = write_report(report, output_dir, run_id)
        print(f"[done] report={report_path}")
        return 0
    except Exception as exc:
        elapsed = time.time() - started
        report["failed_at"] = datetime.now(timezone.utc).isoformat()
        report["elapsed_seconds"] = round(elapsed, 3)
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        report_path = write_report(report, output_dir, run_id)
        print(f"[error] partial report={report_path}", file=sys.stderr)
        raise
    finally:
        client.close()


def run_sync_flow(
    client: EnterpriseKBClient,
    args: argparse.Namespace,
    files: list[SourceFile],
    run_id: str,
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    print("[step] documents:sync auto_parse=true auto_index=true")
    job = client.sync_documents(
        args.kb_id,
        files,
        parser_engine=args.parser_engine,
        process_options=args.process_options,
        idempotency_key=f"enterprise-sync-{args.kb_id}-{run_id}",
        auto_parse=True,
        auto_index=True,
    )
    final = client.wait_for_job(args.kb_id, job["id"], timeout_seconds=args.job_timeout)
    if final["status"] != "succeeded":
        raise RuntimeError(
            f"documents:sync failed: {final.get('error_code')} {final.get('error_message')}"
        )
    report["steps"]["sync"] = {"created": job, "final": final}
    print(f"[ok] sync job={job['id']} status={final['status']}")
    documents_payload = client.list_documents(args.kb_id, limit=200)
    return list(documents_payload.get("documents", []))


def run_manual_flow(
    client: EnterpriseKBClient,
    args: argparse.Namespace,
    files: list[SourceFile],
    run_id: str,
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    print("[step] manual upload -> parse -> build-kg")
    uploaded: list[dict[str, Any]] = []
    parsed_jobs: list[dict[str, Any]] = []
    build_jobs: list[dict[str, Any]] = []
    for source in files:
        upload = client.upload_document(
            args.kb_id,
            source,
            parser_engine=args.parser_engine,
            process_options=args.process_options,
        )
        document = upload["documents"][0]
        uploaded.append(document)
        parse_job = client.parse_document(
            args.kb_id,
            document["id"],
            parser_engine=args.parser_engine,
            process_options=args.process_options,
        )
        parse_final = client.wait_for_job(
            args.kb_id, parse_job["id"], timeout_seconds=args.job_timeout
        )
        if parse_final["status"] != "succeeded":
            raise RuntimeError(f"parse failed for {source.relative_key}: {parse_final}")
        parsed_jobs.append({"created": parse_job, "final": parse_final})

        build_job = client.build_kg(args.kb_id, document["id"])
        if build_job.get("status") in TERMINAL_JOB_STATES:
            build_final = build_job
        else:
            build_final = client.wait_for_job(
                args.kb_id, build_job["id"], timeout_seconds=args.job_timeout
            )
        if build_final["status"] != "succeeded":
            raise RuntimeError(f"build failed for {source.relative_key}: {build_final}")
        build_jobs.append({"created": build_job, "final": build_final})
        print(f"[ok] manual {source.relative_key} -> {document['id']}")
    report["steps"]["manual_flow"] = {
        "run_id": run_id,
        "uploaded": uploaded,
        "parse_jobs": parsed_jobs,
        "build_jobs": build_jobs,
    }
    return uploaded


def force_build_all(
    client: EnterpriseKBClient,
    kb_id: str,
    documents: list[dict[str, Any]],
    timeout_seconds: float,
    report: dict[str, Any],
) -> None:
    results: list[dict[str, Any]] = []
    print(f"[step] force build_kg for {len(documents)} document(s)")
    for document in documents:
        job = client.build_kg(kb_id, document["id"])
        if job.get("status") in TERMINAL_JOB_STATES:
            final = job
        else:
            final = client.wait_for_job(kb_id, job["id"], timeout_seconds=timeout_seconds)
        if final["status"] != "succeeded":
            raise RuntimeError(f"build_kg failed for {document['id']}: {final}")
        results.append({"created": job, "final": final})
    report["steps"]["force_build"] = results


def run_delete_test(
    client: EnterpriseKBClient,
    args: argparse.Namespace,
    documents_payload: dict[str, Any],
    run_id: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    documents = list(documents_payload.get("documents", []))
    summary: dict[str, Any] = {
        "enabled": True,
        "before_total": documents_payload.get("total"),
        "flags": {
            "delete_source_file": args.delete_source_file,
            "delete_artifacts": args.delete_artifacts,
            "delete_llm_cache": args.delete_llm_cache,
            "strategy": args.delete_strategy,
        },
        "selected": [],
        "deleted_count": 0,
    }
    if not documents:
        print("[delete-test] no documents available")
        summary["skipped_reason"] = "no_documents"
        return summary

    print("[delete-test] 当前知识库文件列表：")
    for number, document in enumerate(documents, start=1):
        print(format_document_for_prompt(number, document))

    if not prompt_yes_no("是否删除现有文件？[y/N]: ", default=False):
        print("[delete-test] skipped by user")
        summary["skipped_reason"] = "user_declined"
        return summary

    selected_indexes = prompt_for_document_selection(len(documents))
    if not selected_indexes:
        print("[delete-test] no documents selected")
        summary["skipped_reason"] = "empty_selection"
        return summary

    selected_documents = [documents[index] for index in selected_indexes]
    summary["selected"] = [
        compact_document_for_report(document, number=index + 1)
        for index, document in zip(selected_indexes, selected_documents, strict=True)
    ]
    print("[delete-test] 将删除以下文件：")
    for index, document in zip(selected_indexes, selected_documents, strict=True):
        print(format_document_for_prompt(index + 1, document))
    print(
        "[delete-test] flags "
        f"delete_source_file={args.delete_source_file} "
        f"delete_artifacts={args.delete_artifacts} "
        f"delete_llm_cache={args.delete_llm_cache} "
        f"strategy={args.delete_strategy}"
    )
    if not prompt_yes_no("确认删除所选文件？[y/N]: ", default=False):
        print("[delete-test] cancelled by user")
        summary["skipped_reason"] = "user_cancelled_confirmation"
        return summary

    document_ids = [str(document["id"]) for document in selected_documents]
    idempotency_key = make_delete_idempotency_key(args, run_id, document_ids)
    if len(document_ids) == 1:
        job = client.delete_document(
            args.kb_id,
            document_ids[0],
            delete_source_file=args.delete_source_file,
            delete_artifacts=args.delete_artifacts,
            delete_llm_cache=args.delete_llm_cache,
            strategy=args.delete_strategy,
            idempotency_key=idempotency_key,
        )
    else:
        job = client.batch_delete_documents(
            args.kb_id,
            document_ids,
            delete_source_file=args.delete_source_file,
            delete_artifacts=args.delete_artifacts,
            delete_llm_cache=args.delete_llm_cache,
            strategy=args.delete_strategy,
            idempotency_key=idempotency_key,
        )
    final = wait_for_created_job(client, args.kb_id, job, args.job_timeout)
    summary["job"] = {"created": job, "final": final}
    summary["idempotency_key"] = idempotency_key
    summary["requested_count"] = len(document_ids)
    summary["deleted_count"] = int(final.get("completed_items") or 0)
    report["steps"]["delete_test"] = summary
    report["steps"]["delete_test_job"] = summary["job"]
    if final["status"] != "succeeded":
        summary["failed"] = True
        raise RuntimeError(
            f"delete failed: {final.get('error_code')} {final.get('error_message')}"
        )
    summary["deleted_count"] = len(document_ids)
    print(f"[ok] delete job={job['id']} deleted={len(document_ids)}")
    return summary


def wait_for_created_job(
    client: EnterpriseKBClient,
    kb_id: str,
    job: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    if job.get("status") in TERMINAL_JOB_STATES:
        return job
    return client.wait_for_job(kb_id, job["id"], timeout_seconds=timeout_seconds)


def prompt_for_document_selection(document_count: int) -> list[int]:
    while True:
        try:
            raw = input("请输入要删除的编号（例：1,3-5 或 all；回车取消）: ")
        except EOFError:
            return []
        try:
            return parse_document_selection(raw, document_count)
        except ValueError as exc:
            print(f"[delete-test] {exc}")


def parse_document_selection(raw: str, document_count: int) -> list[int]:
    text = raw.strip().lower()
    if text in {"", "n", "no", "none", "cancel", "q", "quit"}:
        return []
    if text in {"all", "*", "全部"}:
        return list(range(document_count))

    selected: set[int] = set()
    normalized = text.replace("，", ",").replace("；", ",").replace(";", ",")
    for part in normalized.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = parse_selection_number(start_text)
            end = parse_selection_number(end_text)
            if start > end:
                start, end = end, start
            for number in range(start, end + 1):
                add_selection_number(selected, number, document_count)
            continue
        add_selection_number(selected, parse_selection_number(token), document_count)
    return sorted(selected)


def parse_selection_number(value: str) -> int:
    text = value.strip()
    if not text.isdigit():
        raise ValueError(f"编号不是数字：{value!r}")
    return int(text)


def add_selection_number(selected: set[int], number: int, document_count: int) -> None:
    if number < 1 or number > document_count:
        raise ValueError(f"编号超出范围：{number}，有效范围是 1-{document_count}")
    selected.add(number - 1)


def prompt_yes_no(prompt: str, *, default: bool) -> bool:
    yes_values = {"y", "yes", "1", "true", "是", "确认", "删除"}
    no_values = {"n", "no", "0", "false", "否", "不", "取消"}
    while True:
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            return default
        if not answer:
            return default
        if answer in yes_values:
            return True
        if answer in no_values:
            return False
        print("请输入 y 或 n。")


def make_delete_idempotency_key(
    args: argparse.Namespace, run_id: str, document_ids: list[str]
) -> str:
    fingerprint = json.dumps(
        {
            "kb_id": args.kb_id,
            "run_id": run_id,
            "document_ids": document_ids,
            "delete_source_file": args.delete_source_file,
            "delete_artifacts": args.delete_artifacts,
            "delete_llm_cache": args.delete_llm_cache,
            "strategy": args.delete_strategy,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
    return f"enterprise-delete-{run_id}-{digest}"


def format_document_for_prompt(number: int, document: dict[str, Any]) -> str:
    metadata = document_metadata(document)
    source_object_uri = metadata.get("source_object_uri") or "-"
    source_name = document.get("source_name") or metadata.get("source_key") or "-"
    return (
        f"  {number:>2}. status={document.get('status', '-')} "
        f"id={document.get('id', '-')} name={source_name} "
        f"object={source_object_uri}"
    )


def compact_document_for_report(
    document: dict[str, Any], *, number: int | None = None
) -> dict[str, Any]:
    metadata = document_metadata(document)
    result = {
        "id": document.get("id"),
        "number": number,
        "status": document.get("status"),
        "source_name": document.get("source_name"),
        "source_key": metadata.get("source_key"),
        "source_object_uri": metadata.get("source_object_uri"),
        "lightrag_doc_id": document.get("lightrag_doc_id"),
    }
    return {key: value for key, value in result.items() if value is not None}


def document_metadata(document: dict[str, Any]) -> dict[str, Any]:
    metadata = document.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def normalize_run_id(raw_run_id: str) -> str:
    sanitized = "".join(
        char if char in SAFE_RUN_ID_CHARS else "_" for char in raw_run_id.strip()
    ).strip("._")
    return sanitized or "run"


def write_report(report: dict[str, Any], output_dir: Path, run_id: str) -> Path:
    report_path = output_dir / f"enterprise_kb_mvp_report_{run_id}.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report_path


def run_isolation_check(
    client: EnterpriseKBClient, args: argparse.Namespace, report: dict[str, Any]
) -> dict[str, Any]:
    isolation_kb = client.ensure_kb(
        args.isolation_kb_id,
        args.isolation_kb_name,
        "空白对照 KB，用于验证企业知识库 workspace 隔离。",
    )
    documents = client.list_documents(args.isolation_kb_id, limit=20)
    query_result: dict[str, Any] | None = None
    if not args.skip_query:
        query_result = client.query(
            args.isolation_kb_id,
            args.isolation_query,
            mode=args.mode,
            include_references=True,
            include_chunk_content=False,
            top_k=args.top_k,
            chunk_top_k=args.chunk_top_k,
        )
    primary_docs = report["steps"].get("documents", {}).get("documents", [])
    primary_doc_ids = {item.get("id") for item in primary_docs}
    isolation_doc_ids = {item.get("id") for item in documents.get("documents", [])}
    overlap = sorted(str(item) for item in primary_doc_ids & isolation_doc_ids if item)
    if overlap:
        raise RuntimeError(f"KB isolation violated; overlapping document ids: {overlap}")
    return {"kb": isolation_kb, "documents": documents, "query": query_result, "overlap": overlap}


def collect_artifacts(
    client: EnterpriseKBClient, kb_id: str, documents: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for document in documents:
        artifacts = client.list_artifacts(kb_id, document["id"])
        entries = artifacts.get("artifacts") or artifacts.get("items") or []
        object_backed = [
            item
            for item in entries
            if isinstance(item.get("metadata"), dict)
            and (item["metadata"].get("object_uri") or item["metadata"].get("object_prefix_uri"))
        ]
        source_object_uri = (document.get("metadata") or {}).get("source_object_uri")
        if not source_object_uri:
            raise RuntimeError(f"document {document['id']} missing source_object_uri")
        summary.append(
            {
                "document_id": document["id"],
                "source_name": document.get("source_name"),
                "source_object_uri": source_object_uri,
                "artifact_count": len(entries),
                "object_backed_artifact_count": len(object_backed),
                "artifacts": artifacts,
            }
        )
    return summary


def build_enterprise_config(
    args: argparse.Namespace, env_snapshot: dict[str, str | None]
) -> dict[str, Any]:
    embedding_dim = _int_from_snapshot(env_snapshot, "EMBEDDING_DIM")
    embedding_token_limit = _int_from_snapshot(env_snapshot, "EMBEDDING_TOKEN_LIMIT")
    chunk_size = _int_from_snapshot(env_snapshot, "CHUNK_SIZE")
    chunk_overlap = _int_from_snapshot(env_snapshot, "CHUNK_OVERLAP_SIZE")
    return {
        "parser_config": {
            "engine": args.parser_engine,
            "process_options": args.process_options,
            "mineru": {
                "endpoint": env_snapshot.get("MINERU_LOCAL_ENDPOINT"),
                "api_mode": env_snapshot.get("MINERU_API_MODE"),
                "local_backend": env_snapshot.get("MINERU_LOCAL_BACKEND"),
                "local_parse_method": env_snapshot.get("MINERU_LOCAL_PARSE_METHOD"),
                "local_image_analysis": _bool_from_snapshot(
                    env_snapshot, "MINERU_LOCAL_IMAGE_ANALYSIS"
                ),
                "language": env_snapshot.get("MINERU_LANGUAGE"),
                "enable_table": _bool_from_snapshot(env_snapshot, "MINERU_ENABLE_TABLE"),
                "enable_formula": _bool_from_snapshot(env_snapshot, "MINERU_ENABLE_FORMULA"),
            },
        },
        "chunk_config": {
            "chunk_size": chunk_size,
            "chunk_overlap_size": chunk_overlap,
        },
        "embedding_config": {
            "binding": env_snapshot.get("EMBEDDING_BINDING"),
            "host": env_snapshot.get("EMBEDDING_BINDING_HOST"),
            "model": env_snapshot.get("EMBEDDING_MODEL"),
            "dim": embedding_dim,
            "token_limit": embedding_token_limit,
        },
        "llm_role_config": {
            "extract": {
                "binding": env_snapshot.get("LLM_BINDING"),
                "host": env_snapshot.get("LLM_BINDING_HOST"),
                "model": env_snapshot.get("LLM_MODEL"),
                "max_async": _int_from_snapshot(env_snapshot, "EXTRACT_MAX_ASYNC_LLM"),
            },
            "keyword": {
                "binding": env_snapshot.get("LLM_BINDING"),
                "host": env_snapshot.get("LLM_BINDING_HOST"),
                "model": env_snapshot.get("LLM_MODEL"),
                "max_async": _int_from_snapshot(env_snapshot, "KEYWORD_MAX_ASYNC_LLM"),
            },
            "query": {
                "binding": env_snapshot.get("LLM_BINDING"),
                "host": env_snapshot.get("LLM_BINDING_HOST"),
                "model": env_snapshot.get("LLM_MODEL"),
                "max_async": _int_from_snapshot(env_snapshot, "QUERY_MAX_ASYNC_LLM"),
            },
            "vlm": {
                "binding": env_snapshot.get("VLM_LLM_BINDING")
                or env_snapshot.get("LLM_BINDING"),
                "host": env_snapshot.get("VLM_LLM_BINDING_HOST")
                or env_snapshot.get("LLM_BINDING_HOST"),
                "model": env_snapshot.get("VLM_LLM_MODEL"),
                "max_async": _int_from_snapshot(env_snapshot, "VLM_MAX_ASYNC_LLM"),
            },
        },
        "query_config": {
            "top_k": args.top_k,
            "chunk_top_k": args.chunk_top_k,
            "max_entity_tokens": _int_from_snapshot(env_snapshot, "MAX_ENTITY_TOKENS"),
            "max_relation_tokens": _int_from_snapshot(env_snapshot, "MAX_RELATION_TOKENS"),
            "max_total_tokens": _int_from_snapshot(env_snapshot, "MAX_TOTAL_TOKENS"),
            "related_chunk_number": _int_from_snapshot(env_snapshot, "RELATED_CHUNK_NUMBER"),
            "cosine_threshold": _float_from_snapshot(env_snapshot, "COSINE_THRESHOLD"),
            "enable_rerank": _bool_from_snapshot(env_snapshot, "RERANK_BY_DEFAULT"),
        },
        "extraction_config": {
            "language": env_snapshot.get("SUMMARY_LANGUAGE") or "Chinese",
        },
        "storage_config": {
            "metadata_backend": env_snapshot.get("LIGHTRAG_KB_METADATA_BACKEND"),
            "kv_storage": env_snapshot.get("LIGHTRAG_KV_STORAGE"),
            "doc_status_storage": env_snapshot.get("LIGHTRAG_DOC_STATUS_STORAGE"),
            "graph_storage": env_snapshot.get("LIGHTRAG_GRAPH_STORAGE"),
            "vector_storage": env_snapshot.get("LIGHTRAG_VECTOR_STORAGE"),
            "object_storage": {
                "backend": env_snapshot.get("LIGHTRAG_OBJECT_STORAGE"),
                "endpoint": env_snapshot.get("LIGHTRAG_OBJECT_STORAGE_ENDPOINT"),
                "bucket": env_snapshot.get("LIGHTRAG_OBJECT_STORAGE_BUCKET"),
                "prefix": env_snapshot.get("LIGHTRAG_OBJECT_STORAGE_PREFIX"),
                "region": env_snapshot.get("LIGHTRAG_OBJECT_STORAGE_REGION"),
                "create_bucket": _bool_from_snapshot(
                    env_snapshot, "LIGHTRAG_OBJECT_STORAGE_CREATE_BUCKET"
                ),
                "disable_expect_header": _bool_from_snapshot(
                    env_snapshot,
                    "LIGHTRAG_OBJECT_STORAGE_DISABLE_EXPECT_HEADER",
                    default=True,
                ),
            },
            "milvus": {
                "uri": env_snapshot.get("MILVUS_URI"),
                "db_name": env_snapshot.get("MILVUS_DB_NAME"),
                "user": env_snapshot.get("MILVUS_USER"),
                "password": env_snapshot.get("MILVUS_PASSWORD"),
                "token": env_snapshot.get("MILVUS_TOKEN"),
            },
            "env_snapshot": env_snapshot,
        },
        "enterprise_metadata": {
            "scenario": "enterprise_kb_mvp_landing_simulation",
            "source_dir": str(args.source_dir),
            "created_by_script": "examples/enterprise_kb_mvp/enterprise_kb_mvp_demo.py",
        },
    }


def load_env_snapshot(env_file: Path) -> dict[str, str | None]:
    values = parse_env_file(env_file)
    snapshot: dict[str, str | None] = {}
    for key in ENV_SNAPSHOT_KEYS:
        raw = values.get(key)
        snapshot[key] = redact_value(key, raw) if raw is not None else None
    return snapshot


def parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
    return values


def redact_value(key: str, value: str) -> str:
    upper_key = key.upper()
    is_sensitive = upper_key in SENSITIVE_ENV_KEYS or upper_key.endswith(
        ("_API_KEY", "_SECRET", "_PASSWORD", "_TOKEN", "_DSN")
    )
    if is_sensitive:
        if not value:
            return ""
        return f"{value[:2]}***{value[-2:]}" if len(value) > 4 else "***"
    return value


def discover_source_files(source_dir: Path, *, max_files: int) -> list[SourceFile]:
    if not source_dir.exists() or not source_dir.is_dir():
        raise SystemExit(f"Source directory does not exist: {source_dir}")
    source_root = source_dir.resolve()
    files: list[SourceFile] = []
    for path in sorted(source_dir.rglob("*")):
        if (
            path.is_symlink()
            or not path.is_file()
            or path.suffix.lower() not in SUPPORTED_SUFFIXES
        ):
            continue
        resolved_path = path.resolve()
        try:
            resolved_path.relative_to(source_root)
        except ValueError:
            continue
        relative_key = path.relative_to(source_dir).as_posix()
        files.append(
            SourceFile(
                path=resolved_path,
                relative_key=f"enterprise-demo/{relative_key}",
                sha256=hash_file(resolved_path),
                size_bytes=resolved_path.stat().st_size,
                content_type=mimetypes.guess_type(path.name)[0]
                or "application/octet-stream",
            )
        )
        if max_files > 0 and len(files) >= max_files:
            break
    return files


def source_file_to_dict(source: SourceFile) -> dict[str, Any]:
    return {
        "path": str(source.path),
        "relative_key": source.relative_key,
        "sha256": source.sha256,
        "size_bytes": source.size_bytes,
        "content_type": source.content_type,
    }


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wait_timeout_detail(response: httpx.Response) -> dict[str, Any] | None:
    if response.status_code != 408:
        return None
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    detail = payload.get("detail")
    if not isinstance(detail, dict):
        return None
    if detail.get("error_code") != "wait_timeout":
        return None
    return detail


def _int_from_snapshot(snapshot: dict[str, str | None], key: str) -> int | None:
    value = snapshot.get(key)
    if value in (None, ""):
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def _float_from_snapshot(snapshot: dict[str, str | None], key: str) -> float | None:
    value = snapshot.get(key)
    if value in (None, ""):
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def _bool_from_snapshot(
    snapshot: dict[str, str | None], key: str, *, default: bool | None = None
) -> bool | None:
    value = snapshot.get(key)
    if value in (None, ""):
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except httpx.HTTPStatusError as exc:
        print(
            f"[fatal] HTTP {exc.response.status_code} {exc.request.method} {exc.request.url}: {exc.response.text}",
            file=sys.stderr,
        )
        return 2
    except KeyboardInterrupt:
        print("[fatal] interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI should report any terminal error
        print(f"[fatal] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
