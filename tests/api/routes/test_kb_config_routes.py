from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightrag.api.config_version_service import (
    ConfigVersionService,
    active_embedding_runtime_config_from_version,
    active_llm_role_runtime_config_from_version,
    active_parser_runtime_config_from_version,
    active_query_defaults_from_rag,
    active_query_metadata_from_rag,
    apply_active_config_to_lightrag_kwargs,
    attach_active_config_metadata,
)
from lightrag.api.index_build_service import compute_index_hash
from lightrag.api.job_service import JobService
from lightrag.api.kb_service import KnowledgeBaseService
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry, LightRAGLike
from lightrag.api.metadata_store import ConfigVersionRecord, SQLiteMetadataStore
from lightrag.utils import EmbeddingFunc

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_kb_routes = importlib.import_module("lightrag.api.routers.kb_routes")
sys.argv = _original_argv

create_kb_routes = _kb_routes.create_kb_routes

pytestmark = pytest.mark.offline

_API_KEY = "test-key"
_HEADERS = {"X-API-Key": _API_KEY}
_SERVER_ENV_VARS_TO_ISOLATE = (
    "LLM_BINDING",
    "LLM_BINDING_HOST",
    "LLM_BINDING_API_KEY",
    "LLM_MODEL",
    "EMBEDDING_BINDING",
    "EMBEDDING_BINDING_HOST",
    "EMBEDDING_BINDING_API_KEY",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "RERANK_BINDING",
    "EXTRACT_LLM_BINDING",
    "EXTRACT_LLM_MODEL",
    "EXTRACT_LLM_BINDING_HOST",
    "EXTRACT_LLM_BINDING_API_KEY",
    "KEYWORD_LLM_BINDING",
    "KEYWORD_LLM_MODEL",
    "KEYWORD_LLM_BINDING_HOST",
    "KEYWORD_LLM_BINDING_API_KEY",
    "QUERY_LLM_BINDING",
    "QUERY_LLM_MODEL",
    "QUERY_LLM_BINDING_HOST",
    "QUERY_LLM_BINDING_API_KEY",
    "VLM_LLM_BINDING",
    "VLM_LLM_MODEL",
    "VLM_LLM_BINDING_HOST",
    "VLM_LLM_BINDING_API_KEY",
    "LIGHTRAG_KV_STORAGE",
    "LIGHTRAG_VECTOR_STORAGE",
    "LIGHTRAG_GRAPH_STORAGE",
    "LIGHTRAG_DOC_STATUS_STORAGE",
    "LIGHTRAG_API_KEY",
)


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
    assert isinstance(rag, FakeRAG)
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


async def _fake_embed(texts: list[str]):
    return texts


def _config_version(config: dict[str, object]) -> ConfigVersionRecord:
    return ConfigVersionRecord(
        id="cfg_active",
        kb_id="kb_runtime",
        workspace="kb_runtime_ws",
        version=1,
        config=config,
        parser_hash="sha256:parser-active",
        index_hash="sha256:index-active",
        query_hash="sha256:query-active",
        created_at="2026-01-01T00:00:00Z",
        activated_at="2026-01-01T00:00:00Z",
        created_by="tester",
    )


def test_active_config_runtime_helpers_apply_supported_fields():
    embedding_func = EmbeddingFunc(
        embedding_dim=128,
        func=_fake_embed,
        max_token_size=512,
        model_name="base-embed",
    )
    kwargs = {
        "chunk_token_size": 256,
        "chunk_overlap_token_size": 32,
        "tiktoken_model_name": "old-tokenizer",
        "top_k": 20,
        "chunk_top_k": 5,
        "max_entity_tokens": 1000,
        "max_relation_tokens": 2000,
        "max_total_tokens": 3000,
        "cosine_threshold": 0.2,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        "related_chunk_number": 2,
        "embedding_func": embedding_func,
    }
    active = _config_version(
        {
            "chunk_config": {
                "chunk_size": 1024,
                "chunk_overlap_size": 128,
                "tiktoken_model_name": "gpt2",
            },
            "embedding_config": {
                "dim": 768,
                "token_limit": 4096,
                "model": "bge-m3",
            },
            "query_config": {
                "mode": "global",
                "top_k": 77,
                "chunk_top_k": 9,
                "max_total_tokens": 9999,
                "include_references": False,
                "cosine_threshold": 0.42,
                "unsupported_field": "ignored",
            },
            "llm_role_config": {"extract": "unsupported-runtime-role"},
        }
    )

    updated = apply_active_config_to_lightrag_kwargs(dict(kwargs), active)

    assert updated["chunk_token_size"] == 1024
    assert updated["chunk_overlap_token_size"] == 128
    assert updated["tiktoken_model_name"] == "gpt2"
    assert updated["top_k"] == 77
    assert updated["chunk_top_k"] == 9
    assert updated["max_total_tokens"] == 9999
    assert updated["cosine_threshold"] == 0.42
    assert updated["cosine_better_than_threshold"] == 0.42
    assert updated["vector_db_storage_cls_kwargs"] == {
        "cosine_better_than_threshold": 0.42
    }
    assert updated["embedding_func"] is not embedding_func
    assert updated["embedding_func"].embedding_dim == 768
    assert updated["embedding_func"].max_token_size == 4096
    assert updated["embedding_func"].model_name == "bge-m3"
    assert embedding_func.embedding_dim == 128
    assert active_embedding_runtime_config_from_version(active) == {
        "embedding_dim": 768,
        "max_token_size": 4096,
        "model_name": "bge-m3",
    }

    rag = SimpleNamespace()
    attach_active_config_metadata(rag, active)
    expected_hashes = ConfigVersionService._derive_hashes(active.config)

    assert active_query_defaults_from_rag(rag) == {
        "mode": "global",
        "top_k": 77,
        "chunk_top_k": 9,
        "max_total_tokens": 9999,
        "include_references": False,
    }
    assert active_query_metadata_from_rag(rag) == {
        "config_version_id": "cfg_active",
        "parser_hash": "sha256:parser-active",
        "index_hash": expected_hashes["index_hash"],
        "query_hash": expected_hashes["query_hash"],
    }
    assert compute_index_hash(rag) == expected_hashes["index_hash"]


def test_active_parser_runtime_helper_normalizes_supported_fields():
    active = _config_version(
        {
            "parser_config": {
                "engine": "MinerU",
                "process_options": " i-F ",
            },
        }
    )

    assert active_parser_runtime_config_from_version(active) == {
        "parser_engine": "mineru",
        "process_options": "iF",
    }


def test_active_config_applies_extraction_runtime_fields():
    active = _config_version(
        {
            "extraction_config": {
                "language": "Chinese",
                "entity_types": ["Chemical", "Solvent", "Chemical"],
                "max_gleaning": 2,
                "max_extraction_records": 80,
                "max_extraction_entities": 40,
                "force_llm_summary_on_merge": 6,
            },
        }
    )

    updated = apply_active_config_to_lightrag_kwargs({}, active)

    addon = updated["addon_params"]
    assert addon["language"] == "Chinese"
    # Duplicate entity types are de-duplicated while order is preserved.
    assert addon["entity_types"] == ["Chemical", "Solvent"]
    assert "- Chemical" in addon["entity_types_guidance"]
    assert "- Solvent" in addon["entity_types_guidance"]
    assert updated["entity_extract_max_gleaning"] == 2
    assert updated["entity_extract_max_records"] == 80
    assert updated["entity_extract_max_entities"] == 40
    assert updated["force_llm_summary_on_merge"] == 6


def test_active_config_explicit_guidance_takes_precedence_and_merges_addon():
    active = _config_version(
        {
            "extraction_config": {
                "entity_types": ["Ignored"],
                "entity_types_guidance": "Custom guidance only.",
            },
        }
    )

    updated = apply_active_config_to_lightrag_kwargs(
        {"addon_params": {"language": "English", "chunker": {"keep": True}}},
        active,
    )

    addon = updated["addon_params"]
    assert addon["entity_types_guidance"] == "Custom guidance only."
    # Explicit guidance wins, so no entity_types list is injected.
    assert "entity_types" not in addon
    # Pre-existing addon keys are preserved through the merge.
    assert addon["language"] == "English"
    assert addon["chunker"] == {"keep": True}


def test_extraction_config_changes_require_reindex_via_index_hash():
    base = {"chunk_config": {"chunk_size": 512}}
    changed = {
        "chunk_config": {"chunk_size": 512},
        "extraction_config": {"entity_types": ["Chemical"]},
    }
    assert (
        ConfigVersionService._derive_hashes(base)["index_hash"]
        != ConfigVersionService._derive_hashes(changed)["index_hash"]
    )


def test_active_index_hash_ignores_unsupported_runtime_sections():
    base_config = {
        "chunk_config": {"chunk_size": 512},
        "embedding_config": {"model": "bge-m3", "dim": 1024},
        "storage_config": {"graph_storage": "Neo4j"},
    }
    changed_unsupported_config = {
        **base_config,
        "storage_config": {"graph_storage": "NetworkX"},
    }

    assert ConfigVersionService._derive_hashes(base_config)[
        "index_hash"
    ] == ConfigVersionService._derive_hashes(changed_unsupported_config)[
        "index_hash"
    ]

    changed_supported_config = {
        **base_config,
        "embedding_config": {"model": "bge-large", "dim": 1024},
    }
    assert ConfigVersionService._derive_hashes(base_config)[
        "index_hash"
    ] != ConfigVersionService._derive_hashes(changed_supported_config)[
        "index_hash"
    ]


def test_extract_role_model_change_requires_reindex_via_index_hash():
    """Changing the extraction LLM role model invalidates built KG content."""
    base = {"llm_role_config": {"extract": "model-a"}}
    changed = {"llm_role_config": {"extract": "model-b"}}
    assert (
        ConfigVersionService._derive_hashes(base)["index_hash"]
        != ConfigVersionService._derive_hashes(changed)["index_hash"]
    )


def test_vlm_role_model_change_requires_reindex_via_index_hash():
    base = {"llm_role_config": {"vlm": {"model": "vlm-a", "binding": "openai"}}}
    changed = {"llm_role_config": {"vlm": {"model": "vlm-b", "binding": "openai"}}}
    assert (
        ConfigVersionService._derive_hashes(base)["index_hash"]
        != ConfigVersionService._derive_hashes(changed)["index_hash"]
    )


def test_query_role_model_change_only_affects_query_hash():
    """Query/keyword role identity is retrieval-time only: query_hash moves,
    index_hash stays stable so no reindex is forced."""
    base = {"llm_role_config": {"query": "q-model-a"}}
    changed = {"llm_role_config": {"query": "q-model-b"}}
    base_hashes = ConfigVersionService._derive_hashes(base)
    changed_hashes = ConfigVersionService._derive_hashes(changed)
    assert base_hashes["index_hash"] == changed_hashes["index_hash"]
    assert base_hashes["query_hash"] != changed_hashes["query_hash"]


def test_llm_role_secret_and_perf_knobs_excluded_from_hash():
    """Rotating api_key or tuning max_async/timeout must not force a rebuild."""
    base = {"llm_role_config": {"extract": {"model": "m", "api_key": "k1"}}}
    rotated = {
        "llm_role_config": {
            "extract": {"model": "m", "api_key": "k2", "max_async": 8, "timeout": 90}
        }
    }
    base_hashes = ConfigVersionService._derive_hashes(base)
    rotated_hashes = ConfigVersionService._derive_hashes(rotated)
    assert base_hashes["index_hash"] == rotated_hashes["index_hash"]
    assert base_hashes["query_hash"] == rotated_hashes["query_hash"]


def test_llm_role_config_rejects_unknown_role_and_keys():
    with pytest.raises(ValueError):
        ConfigVersionService._derive_hashes(
            {"llm_role_config": {"unknown_role": "m"}}
        )
    with pytest.raises(ValueError):
        ConfigVersionService._derive_hashes(
            {"llm_role_config": {"extract": {"bogus_key": "x"}}}
        )


def test_active_llm_role_runtime_config_normalizes_shapes():
    active = _config_version(
        {
            "llm_role_config": {
                "extract": "string-model",
                "query": {
                    "model": "  q-model  ",
                    "binding": "openai",
                    "api_key": "secret",
                    "kwargs": {"temperature": 0.1},
                    "max_async": 4,
                },
                "vlm": {"model": None},
            }
        }
    )

    runtime = active_llm_role_runtime_config_from_version(active)

    # Bare string normalizes to a model override.
    assert runtime["extract"] == {"model": "string-model"}
    # ``kwargs`` aliases to ``model_kwargs``; strings are stripped; ints kept.
    assert runtime["query"] == {
        "model": "q-model",
        "binding": "openai",
        "api_key": "secret",
        "model_kwargs": {"temperature": 0.1},
        "max_async": 4,
    }
    # A role whose only key resolves to None contributes no override.
    assert "vlm" not in runtime
    # None active version yields no overrides.
    assert active_llm_role_runtime_config_from_version(None) == {}


def test_active_embedding_model_reaches_rebuilt_provider_closure(
    tmp_path, monkeypatch
):
    import asyncio

    import numpy as np

    for var in _SERVER_ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)

    for role in ("EXTRACT", "KEYWORD", "QUERY", "VLM"):
        monkeypatch.setenv(f"{role}_LLM_BINDING", "openai")
        monkeypatch.setenv(f"{role}_LLM_BINDING_HOST", "https://api.openai.com/v1")
        monkeypatch.setenv(f"{role}_LLM_BINDING_API_KEY", "test-key")
        monkeypatch.setenv(f"{role}_LLM_MODEL", "gpt-4o-mini")

    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("LLM_BINDING_HOST", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_BINDING_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("EMBEDDING_BINDING", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING_HOST", "https://api.openai.com/v1")
    monkeypatch.setenv("EMBEDDING_BINDING_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "base-env-embed")
    monkeypatch.setenv("EMBEDDING_DIM", "1536")
    monkeypatch.setenv("RERANK_BINDING", "null")
    monkeypatch.setenv("LIGHTRAG_API_KEY", _API_KEY)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lightrag-server",
            "--working-dir",
            str(tmp_path / "rag_storage"),
            "--input-dir",
            str(tmp_path / "inputs"),
        ],
    )

    from lightrag.api.config import parse_args
    from lightrag.api import lightrag_server
    from lightrag.llm import openai as openai_llm

    provider_calls: list[dict[str, object]] = []

    async def fake_openai_embed(texts, **kwargs):
        provider_calls.append(dict(kwargs))
        return np.zeros((len(texts), 1536), dtype=np.float32)

    monkeypatch.setattr(
        openai_llm,
        "openai_embed",
        EmbeddingFunc(
            embedding_dim=1536,
            func=fake_openai_embed,
            max_token_size=8192,
            model_name="base-provider-embed",
        ),
    )

    class CapturedLightRAG:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.workspace = kwargs["workspace"]
            self.embedding_func = kwargs["embedding_func"]
            self.ollama_server_infos = kwargs["ollama_server_infos"]
            self.role_llm_builder = None
            self.finalized = False

        def register_role_llm_builder(self, builder):
            self.role_llm_builder = builder

        def get_llm_role_config(self):
            return {}

        async def initialize_storages(self):
            return None

        async def check_and_migrate_data(self):
            return None

        async def finalize_storages(self):
            self.finalized = True

    monkeypatch.setattr(lightrag_server, "LightRAG", CapturedLightRAG)
    monkeypatch.setattr(
        lightrag_server, "check_frontend_build", lambda: (False, False)
    )

    async def exercise_active_embedding_provider():
        args = parse_args()
        app = lightrag_server.create_app(args)
        try:
            await app.state.kb_service.initialize()
            await app.state.metadata_store.initialize()

            kb_id = "kb_active_embed"
            await app.state.kb_service.create(kb_id=kb_id, name=kb_id)
            version = await app.state.config_version_service.create(
                kb_id,
                config={"embedding_config": {"model": "active-kb-embed"}},
            )
            await app.state.config_version_service.activate(kb_id, version.id)

            rag = await app.state.lightrag_registry.get(kb_id)
            await rag.embedding_func(["probe"])

            assert rag.embedding_func.model_name == "active-kb-embed"
            assert provider_calls[-1]["model"] == "active-kb-embed"
        finally:
            await app.state.lightrag_registry.shutdown()

    asyncio.run(exercise_active_embedding_provider())


def test_active_llm_role_config_reaches_built_instance(tmp_path, monkeypatch):
    """The KB ``llm_role_config`` overrides are applied to the freshly built
    LightRAG instance via ``aupdate_llm_role_config`` (so the registered role
    builder rebuilds the role func with the KB's model)."""
    import asyncio

    import numpy as np

    for var in _SERVER_ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)

    for role in ("EXTRACT", "KEYWORD", "QUERY", "VLM"):
        monkeypatch.setenv(f"{role}_LLM_BINDING", "openai")
        monkeypatch.setenv(f"{role}_LLM_BINDING_HOST", "https://api.openai.com/v1")
        monkeypatch.setenv(f"{role}_LLM_BINDING_API_KEY", "test-key")
        monkeypatch.setenv(f"{role}_LLM_MODEL", "gpt-4o-mini")

    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("LLM_BINDING_HOST", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_BINDING_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("EMBEDDING_BINDING", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING_HOST", "https://api.openai.com/v1")
    monkeypatch.setenv("EMBEDDING_BINDING_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "base-env-embed")
    monkeypatch.setenv("EMBEDDING_DIM", "1536")
    monkeypatch.setenv("RERANK_BINDING", "null")
    monkeypatch.setenv("LIGHTRAG_API_KEY", _API_KEY)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lightrag-server",
            "--working-dir",
            str(tmp_path / "rag_storage"),
            "--input-dir",
            str(tmp_path / "inputs"),
        ],
    )

    from lightrag.api.config import parse_args
    from lightrag.api import lightrag_server
    from lightrag.llm import openai as openai_llm

    async def fake_openai_embed(texts, **kwargs):
        return np.zeros((len(texts), 1536), dtype=np.float32)

    monkeypatch.setattr(
        openai_llm,
        "openai_embed",
        EmbeddingFunc(
            embedding_dim=1536,
            func=fake_openai_embed,
            max_token_size=8192,
            model_name="base-provider-embed",
        ),
    )

    class CapturedLightRAG:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.workspace = kwargs["workspace"]
            self.embedding_func = kwargs["embedding_func"]
            self.ollama_server_infos = kwargs["ollama_server_infos"]
            self.role_llm_builder = None
            self.role_updates: list[dict[str, object]] = []
            self.finalized = False

        def register_role_llm_builder(self, builder):
            self.role_llm_builder = builder

        async def aupdate_llm_role_config(self, role, **override):
            self.role_updates.append({"role": role, **override})

        def get_llm_role_config(self):
            return {}

        async def initialize_storages(self):
            return None

        async def check_and_migrate_data(self):
            return None

        async def finalize_storages(self):
            self.finalized = True

    monkeypatch.setattr(lightrag_server, "LightRAG", CapturedLightRAG)
    monkeypatch.setattr(
        lightrag_server, "check_frontend_build", lambda: (False, False)
    )

    async def exercise_active_role_config():
        args = parse_args()
        app = lightrag_server.create_app(args)
        try:
            await app.state.kb_service.initialize()
            await app.state.metadata_store.initialize()

            kb_id = "kb_active_role"
            await app.state.kb_service.create(kb_id=kb_id, name=kb_id)
            version = await app.state.config_version_service.create(
                kb_id,
                config={
                    "llm_role_config": {
                        "extract": "kb-extract-model",
                        "query": {"model": "kb-query-model", "max_async": 3},
                    }
                },
            )
            await app.state.config_version_service.activate(kb_id, version.id)

            rag = await app.state.lightrag_registry.get(kb_id)
            updates = {item["role"]: item for item in rag.role_updates}
            assert updates["extract"]["model"] == "kb-extract-model"
            assert updates["query"]["model"] == "kb-query-model"
            assert updates["query"]["max_async"] == 3
            # Roles without overrides are not touched.
            assert "vlm" not in updates
            assert "keyword" not in updates
        finally:
            await app.state.lightrag_registry.shutdown()

    asyncio.run(exercise_active_role_config())


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


def test_config_diff_parser_only_change_requires_reparse(tmp_path):
    """Changing ONLY parser_config flips requires_reparse (and the implied
    requires_reindex) without requiring a vector rebuild."""
    client, *_ = _build_client(tmp_path)
    _create_kb(client, "kb_parser_diff")
    base = client.post(
        "/kbs/kb_parser_diff/configs",
        json={"config": _BASE_CONFIG},
        headers=_HEADERS,
    )
    client.post(
        f"/kbs/kb_parser_diff/configs/{base.json()['id']}:activate", headers=_HEADERS
    )

    target = client.post(
        "/kbs/kb_parser_diff/configs",
        json={"config": {**_BASE_CONFIG, "parser_config": {"engine": "docling"}}},
        headers=_HEADERS,
    )
    diff = client.post(
        f"/kbs/kb_parser_diff/configs/{target.json()['id']}:diff", headers=_HEADERS
    )
    assert diff.status_code == 200
    body = diff.json()
    assert body["requires_reparse"] is True
    assert body["requires_reindex"] is True
    assert body["requires_vector_rebuild"] is False
    assert "parser_hash_changed" in body["reasons"]
    assert "embedding_changed" not in body["reasons"]


def test_config_diff_query_only_change_requires_no_rebuild(tmp_path):
    """Changing ONLY query_config (top_k) needs no reparse/reindex/vector
    rebuild — it only affects query-time defaults."""
    client, *_ = _build_client(tmp_path)
    _create_kb(client, "kb_query_diff")
    base = client.post(
        "/kbs/kb_query_diff/configs",
        json={"config": _BASE_CONFIG},
        headers=_HEADERS,
    )
    client.post(
        f"/kbs/kb_query_diff/configs/{base.json()['id']}:activate", headers=_HEADERS
    )

    target = client.post(
        "/kbs/kb_query_diff/configs",
        json={"config": {**_BASE_CONFIG, "query_config": {"top_k": 80}}},
        headers=_HEADERS,
    )
    diff = client.post(
        f"/kbs/kb_query_diff/configs/{target.json()['id']}:diff", headers=_HEADERS
    )
    assert diff.status_code == 200
    body = diff.json()
    assert body["requires_reparse"] is False
    assert body["requires_reindex"] is False
    assert body["requires_vector_rebuild"] is False
    assert body["reasons"] == ["query_hash_changed"]


def test_activate_skips_discard_while_destructive_lock_held(tmp_path):
    """Activation must still persist active_config_version_id + activated_at
    even when a destructive job holds the lock; the registry discard is
    silently skipped (cached instance left intact for the destructive job)."""
    import asyncio

    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kb.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    probe = BuilderProbe()
    registry = LightRAGInstanceRegistry(kb_service, probe.build, probe.finalize)
    config_service = ConfigVersionService(kb_service, metadata_store, registry)

    async def _exercise() -> None:
        await kb_service.initialize()
        await kb_service.create(kb_id="kb_lock", name="Lock")
        rag = await registry.get("kb_lock")
        assert isinstance(rag, FakeRAG)
        assert registry.is_loaded("kb_lock")
        version = await config_service.create("kb_lock", config=_BASE_CONFIG)

        async with registry.destructive_lock("kb_lock"):
            activated = await config_service.activate("kb_lock", version.id)
            # The write side-effects still happened ...
            assert activated.activated_at is not None
            refreshed = await kb_service.get("kb_lock")
            assert refreshed.active_config_version_id == version.id
            # ... but discard was skipped: the instance is still cached and
            # was NOT finalized while the destructive lock was held.
            assert registry.is_loaded("kb_lock")
            assert rag.finalized is False

    try:
        asyncio.run(_exercise())
    finally:
        asyncio.run(registry.shutdown())
