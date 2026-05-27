"""Run an end-to-end MinerU PDF ingestion pipeline with deployed services.

The script reads PDFs from test_pdf by default, parses them through MinerU,
builds LightRAG chunks/entities/relations/graph data, writes vectors to the
configured vector database, and optionally runs a mix-mode query with rerank.

Usage:
    python scripts/run_mineru_pdf_pipeline.py
    python scripts/run_mineru_pdf_pipeline.py --force-reparse-mineru --skip-query
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any, Callable

import httpx
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)

from lightrag import LightRAG, QueryParam
from lightrag.constants import (
    FULL_DOCS_FORMAT_PENDING_PARSE,
    PARSER_ENGINE_MINERU,
)
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.llm_roles import RoleLLMConfig
from lightrag.rerank import ali_rerank, cohere_rerank, jina_rerank
from lightrag.utils import EmbeddingFunc, generate_track_id


DEFAULT_QUERY = "Summarize the indexed PDFs and list key entities and relationships."


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def optional_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def local_api_key(*names: str) -> str:
    for name in names:
        value = optional_env(name)
        if value:
            return value
    return "not_needed"


def local_http_client_configs() -> dict[str, Any]:
    return {
        "http_client": httpx.AsyncClient(verify=False, trust_env=False),
    }


def openai_options(role_prefix: str | None = None) -> dict[str, Any]:
    prefix = f"{role_prefix}_OPENAI_LLM_" if role_prefix else "OPENAI_LLM_"
    options: dict[str, Any] = {}
    scalar_fields: tuple[tuple[str, str, Callable[[str], Any]], ...] = (
        ("TEMPERATURE", "temperature", float),
        ("MAX_TOKENS", "max_tokens", int),
        ("MAX_COMPLETION_TOKENS", "max_completion_tokens", int),
        ("REASONING_EFFORT", "reasoning_effort", str),
        ("TOP_P", "top_p", float),
        ("FREQUENCY_PENALTY", "frequency_penalty", float),
        ("PRESENCE_PENALTY", "presence_penalty", float),
    )
    for env_suffix, option_name, caster in scalar_fields:
        raw = optional_env(prefix + env_suffix)
        if raw is not None:
            options[option_name] = caster(raw)

    stop = optional_env(prefix + "STOP")
    if stop is not None:
        options["stop"] = json.loads(stop)

    extra_body = optional_env(prefix + "EXTRA_BODY")
    if extra_body is not None:
        options["extra_body"] = json.loads(extra_body)

    return options


def make_openai_llm_func(
    *,
    model: str,
    base_url: str,
    api_key: str,
    timeout: int,
    provider_options: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    provider_options = provider_options or {}

    async def llm_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        if history_messages is None:
            history_messages = []
        call_kwargs = dict(provider_options)
        call_kwargs.update(kwargs)
        call_kwargs.setdefault("timeout", timeout)
        call_kwargs.setdefault("openai_client_configs", local_http_client_configs())
        return await openai_complete_if_cache(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=base_url,
            api_key=api_key,
            **call_kwargs,
        )

    return llm_model_func


def build_embedding_func() -> EmbeddingFunc:
    model = os.getenv("EMBEDDING_MODEL", "bge-m3")
    base_url = os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:8000/v1")
    api_key = local_api_key("EMBEDDING_BINDING_API_KEY", "OPENAI_API_KEY")

    async def embedding_call(texts: list[str], **kwargs: Any):
        kwargs.setdefault("client_configs", local_http_client_configs())
        return await openai_embed.func(
            texts,
            model=model,
            base_url=base_url,
            api_key=api_key,
            **kwargs,
        )

    return EmbeddingFunc(
        model_name=model,
        embedding_dim=env_int("EMBEDDING_DIM", 1024),
        max_token_size=env_int("EMBEDDING_TOKEN_LIMIT", 8192),
        send_dimensions=env_bool("EMBEDDING_SEND_DIM", False),
        supports_asymmetric=True,
        func=embedding_call,
    )


def build_rerank_func() -> Callable[..., Any] | None:
    binding = os.getenv("RERANK_BINDING", "null").strip().lower()
    if binding in {"", "null", "none"}:
        return None

    rerank_functions = {
        "cohere": cohere_rerank,
        "jina": jina_rerank,
        "aliyun": ali_rerank,
    }
    if binding not in rerank_functions:
        raise ValueError(f"Unsupported RERANK_BINDING: {binding}")

    return partial(
        rerank_functions[binding],
        model=os.getenv("RERANK_MODEL", "bge-reranker"),
        base_url=os.getenv("RERANK_BINDING_HOST", "http://localhost:8000/rerank"),
        api_key=optional_env("RERANK_BINDING_API_KEY"),
    )


def build_role_llm_configs(
    base_model_func: Callable[..., Any],
) -> dict[str, RoleLLMConfig]:
    role_configs: dict[str, RoleLLMConfig] = {}
    base_binding = os.getenv("LLM_BINDING", "openai").strip().lower()

    for role in ("EXTRACT", "KEYWORD", "QUERY", "VLM"):
        model = optional_env(f"{role}_LLM_MODEL")
        binding = optional_env(f"{role}_LLM_BINDING") or base_binding
        host = optional_env(f"{role}_LLM_BINDING_HOST") or os.getenv(
            "LLM_BINDING_HOST", "http://localhost:8000/v1"
        )
        api_key = local_api_key(f"{role}_LLM_BINDING_API_KEY", "LLM_BINDING_API_KEY")
        timeout = env_int(f"{role}_LLM_TIMEOUT", env_int("LLM_TIMEOUT", 180))
        max_async = optional_env(f"{role}_MAX_ASYNC_LLM")

        if binding != "openai":
            if model or binding != base_binding or max_async is not None:
                raise ValueError(
                    f"This script currently builds role override functions only for "
                    f"OpenAI-compatible bindings; got {role}_LLM_BINDING={binding!r}."
                )
            continue

        if (
            not model
            and max_async is None
            and optional_env(f"{role}_LLM_TIMEOUT") is None
        ):
            continue

        role_lower = role.lower()
        role_func = (
            make_openai_llm_func(
                model=model or os.getenv("LLM_MODEL", "qwen3.6-36b"),
                base_url=host,
                api_key=api_key,
                timeout=timeout,
                provider_options=openai_options(role),
            )
            if model
            else base_model_func
        )
        role_configs[role_lower] = RoleLLMConfig(
            func=role_func,
            max_async=int(max_async) if max_async is not None else None,
            timeout=timeout,
            metadata={
                "binding": binding,
                "model": model or os.getenv("LLM_MODEL", "qwen3.6-36b"),
                "host": host,
            },
        )

    return role_configs


def vector_kwargs() -> dict[str, Any]:
    return {
        "cosine_better_than_threshold": env_float("COSINE_THRESHOLD", 0.2),
        "index_type": os.getenv("MILVUS_INDEX_TYPE", "AUTOINDEX"),
        "metric_type": os.getenv("MILVUS_METRIC_TYPE", "COSINE"),
    }


def discover_pdfs(pdf_dir: Path) -> list[Path]:
    return sorted(path for path in pdf_dir.iterdir() if path.suffix.lower() == ".pdf")


async def preflight_mineru_endpoint() -> None:
    api_mode = os.getenv("MINERU_API_MODE", "local").strip().lower()
    if api_mode != "local":
        return

    endpoint = os.getenv("MINERU_LOCAL_ENDPOINT", "").strip().rstrip("/")
    if not endpoint:
        raise RuntimeError("MINERU_LOCAL_ENDPOINT is required for MINERU_API_MODE=local")

    async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=3.0)) as client:
        try:
            openapi_resp = await client.get(f"{endpoint}/openapi.json")
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Cannot reach MINERU_LOCAL_ENDPOINT={endpoint!r}: {exc}"
            ) from exc

        if openapi_resp.status_code == 200:
            payload = openapi_resp.json()
            paths = set((payload.get("paths") or {}).keys())
            if "/tasks" in paths:
                return
            if "/v1/models" in paths or any(path.startswith("/v1/") for path in paths):
                detail = (
                    "The endpoint looks like a vLLM/OpenAI-compatible model server. "
                    "LightRAG's MinerU adapter needs the self-hosted mineru-api "
                    "or mineru-router base URL that provides POST /tasks, "
                    "GET /tasks/{task_id}, and GET /tasks/{task_id}/result. "
                    "Configure the vLLM MinerU model URL on the mineru-api side, "
                    "then set MINERU_LOCAL_ENDPOINT to that mineru-api/router URL."
                )
            else:
                shown_paths = ", ".join(sorted(paths)[:10]) or "<none>"
                detail = f"The endpoint OpenAPI paths do not include /tasks: {shown_paths}"
            raise RuntimeError(
                f"Invalid MINERU_LOCAL_ENDPOINT={endpoint!r}. {detail}"
            )

        tasks_resp = await client.get(f"{endpoint}/tasks")
        if tasks_resp.status_code in {200, 405, 422}:
            return
        raise RuntimeError(
            f"Invalid MINERU_LOCAL_ENDPOINT={endpoint!r}: GET /tasks returned "
            f"HTTP {tasks_resp.status_code}. Expected a mineru-api/router service "
            "with the /tasks API."
        )


def doc_status_value(doc: Any) -> str:
    status = doc.get("status") if isinstance(doc, dict) else getattr(doc, "status", "")
    return getattr(status, "value", str(status))


async def initialize_rag(args: argparse.Namespace) -> LightRAG:
    llm_timeout = env_int("LLM_TIMEOUT", 180)
    llm_model = os.getenv("LLM_MODEL", "qwen3.6-36b")
    llm_func = make_openai_llm_func(
        model=llm_model,
        base_url=os.getenv("LLM_BINDING_HOST", "http://localhost:8000/v1"),
        api_key=local_api_key("LLM_BINDING_API_KEY", "OPENAI_API_KEY"),
        timeout=llm_timeout,
        provider_options=openai_options(),
    )
    role_configs = build_role_llm_configs(llm_func)

    rag = LightRAG(
        working_dir=str(args.working_dir),
        workspace=args.workspace,
        llm_model_name=llm_model,
        llm_model_func=llm_func,
        role_llm_configs=role_configs,
        embedding_func=build_embedding_func(),
        rerank_model_func=build_rerank_func(),
        kv_storage=os.getenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage"),
        doc_status_storage=os.getenv(
            "LIGHTRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage"
        ),
        graph_storage=os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage"),
        vector_storage=os.getenv("LIGHTRAG_VECTOR_STORAGE", "MilvusVectorDBStorage"),
        vector_db_storage_cls_kwargs=vector_kwargs(),
        enable_llm_cache=env_bool("ENABLE_LLM_CACHE", True),
        enable_llm_cache_for_entity_extract=env_bool(
            "ENABLE_LLM_CACHE_FOR_EXTRACT", True
        ),
        vlm_process_enable=env_bool("VLM_PROCESS_ENABLE", True),
    )
    await rag.initialize_storages()
    return rag


async def run_pipeline(args: argparse.Namespace) -> int:
    pdf_dir = args.pdf_dir.resolve()
    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdfs = discover_pdfs(pdf_dir)
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in: {pdf_dir}")

    os.environ["INPUT_DIR"] = str(pdf_dir)
    if args.force_reparse_mineru:
        os.environ["LIGHTRAG_FORCE_REPARSE_MINERU"] = "true"

    args.working_dir.mkdir(parents=True, exist_ok=True)

    print("LightRAG MinerU PDF pipeline")
    print(f"  pdf_dir: {pdf_dir}")
    print(f"  working_dir: {args.working_dir}")
    print(f"  workspace: {args.workspace or '<default>'}")
    print(f"  mineru_endpoint: {os.getenv('MINERU_LOCAL_ENDPOINT')}")
    print(f"  mineru_backend: {os.getenv('MINERU_LOCAL_BACKEND')}")
    print(f"  vector_storage: {os.getenv('LIGHTRAG_VECTOR_STORAGE')}")
    print(f"  milvus_uri: {os.getenv('MILVUS_URI')}")
    print(f"  pdf_count: {len(pdfs)}")

    await preflight_mineru_endpoint()

    rag: LightRAG | None = None
    try:
        rag = await initialize_rag(args)

        embedding = await rag.embedding_func(["LightRAG embedding connectivity check"])
        print(f"Embedding check OK: shape={tuple(embedding.shape)}")

        track_id = generate_track_id("mineru-pdf")
        print(f"Enqueueing PDFs with track_id={track_id}")

        await rag.apipeline_enqueue_documents(
            [""] * len(pdfs),
            file_paths=[path.name for path in pdfs],
            track_id=track_id,
            docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
            parse_engine=[PARSER_ENGINE_MINERU] * len(pdfs),
            process_options=[args.process_options] * len(pdfs),
        )
        await rag.apipeline_process_enqueue_documents()

        docs = await rag.doc_status.get_docs_by_track_id(track_id)
        status_counts = Counter(doc_status_value(doc) for doc in docs.values())
        print(f"Document status for this run: {dict(status_counts)}")

        all_status_counts = await rag.doc_status.get_all_status_counts()
        print(f"All document status counts: {all_status_counts}")

        graph_nodes = await rag.chunk_entity_relation_graph.get_all_nodes()
        graph_edges = await rag.chunk_entity_relation_graph.get_all_edges()
        print(f"Knowledge graph: nodes={len(graph_nodes)}, edges={len(graph_edges)}")

        vector_hits = await rag.chunks_vdb.query(args.query, top_k=3)
        print(f"Milvus chunk vector query check: hits={len(vector_hits)}")
        for index, hit in enumerate(vector_hits, start=1):
            print(
                f"  hit{index}: id={hit.get('id', '<unknown>')} "
                f"file={hit.get('file_path', '<unknown>')}"
            )

        if not args.skip_query:
            result = await rag.aquery(
                args.query,
                param=QueryParam(
                    mode="mix",
                    stream=False,
                    enable_rerank=rag.rerank_model_func is not None,
                ),
            )
            print("\nMix query result:")
            print(result)

        failed_docs = [
            doc for doc in docs.values() if doc_status_value(doc) == "failed"
        ]
        return 2 if failed_docs else 0
    finally:
        if rag is not None:
            await rag.finalize_storages()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse PDFs with MinerU and index them into LightRAG/Milvus."
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=PROJECT_ROOT / "test_pdf",
        help="Directory containing PDF files to index.",
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=Path(os.getenv("WORKING_DIR", PROJECT_ROOT / "rag_storage_mineru_test")),
        help="LightRAG working directory for KV/doc-status/graph/cache files.",
    )
    parser.add_argument(
        "--workspace",
        default=os.getenv("WORKSPACE", "mineru_pdf_test"),
        help="Workspace name used to isolate storage collections.",
    )
    parser.add_argument(
        "--process-options",
        default="iteP",
        help=(
            "LightRAG process options for parsed PDFs. Default enables "
            "image/table/equation analysis and paragraph chunking."
        ),
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Query used for vector and mix-mode retrieval checks after indexing.",
    )
    parser.add_argument(
        "--skip-query",
        action="store_true",
        help="Only index PDFs and skip the final mix-mode query.",
    )
    parser.add_argument(
        "--force-reparse-mineru",
        action="store_true",
        help="Ignore existing MinerU raw parse cache for this run.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        return asyncio.run(run_pipeline(parse_args()))
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())