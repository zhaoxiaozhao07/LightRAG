"""Tests for the bilingual (zh<->en) dual-path query feature.

Covers docs/BilingualQuery-zh.md:
- service-level unit behaviour (language detection, mode resolution,
  preprocessing fail-open, merge/dedup, alt-param keyword seeding);
- KB config plumbing (query_config.bilingual_query validation + runtime read);
- single-KB / multi-KB / legacy route integration (dual retrieval, merged
  synthesis, metadata.bilingual, fallback paths);
- agent step execution (QueryToolService alt path) and staged helpers.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightrag.api.document_lifecycle_service import DocumentLifecycleService
from lightrag.api.job_service import JobService
from lightrag.api.kb_service import KnowledgeBaseService
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry
from lightrag.api.metadata_store import SQLiteMetadataStore
from lightrag.base import QueryParam

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_bilingual = importlib.import_module("lightrag.api.bilingual_query_service")
_cvs = importlib.import_module("lightrag.api.config_version_service")
_kb_routes = importlib.import_module("lightrag.api.routers.kb_routes")
_kb_query_routes = importlib.import_module("lightrag.api.routers.kb_query_routes")
_query_routes = importlib.import_module("lightrag.api.routers.query_routes")
_query_tool = importlib.import_module("lightrag.api.query_tool_service")
_agent_service = importlib.import_module("lightrag.api.agent_query_service")
_agent_staged = importlib.import_module("lightrag.api.agent_staged_service")
sys.argv = _original_argv

create_kb_routes = _kb_routes.create_kb_routes
create_kb_query_routes = _kb_query_routes.create_kb_query_routes
create_query_routes = _query_routes.create_query_routes

pytestmark = pytest.mark.offline

_API_KEY = "test-key"
_HEADERS = {"X-API-Key": _API_KEY}

_ZH_QUERY = "推荐一种耐高温的胎侧橡胶配方"
_EN_QUERY = "recommend a heat resistant sidewall rubber formula"


@pytest.fixture(autouse=True)
def _pin_global_args(monkeypatch):
    """Pin non-enterprise mode + bilingual master ON for these tests.

    Individual tests flip attributes on this namespace to exercise the
    off/auto/on matrix.
    """
    from lightrag.api import config as api_config

    namespace = SimpleNamespace(
        enterprise_auth_enabled=False,
        token_auto_renew=False,
        token_renew_threshold=0.5,
        bilingual_query_enabled=True,
        bilingual_query_default_mode="auto",
        bilingual_query_timeout=12,
    )
    monkeypatch.setattr(api_config, "global_args", namespace)
    return namespace


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class BilingualFakeRAG:
    """LightRAG stand-in returning language-specific chunks.

    Chinese queries hit ``zh.pdf`` evidence, non-Chinese queries hit
    ``en.pdf`` evidence, so a dual-path retrieval visibly widens recall.
    The query-role LLM answers the preprocessing call with fixed JSON and
    echoes seen evidence markers during synthesis.
    """

    def __init__(
        self,
        workspace: str,
        *,
        fail_preprocess: bool = False,
        fail_secondary: bool = False,
    ):
        self.workspace = workspace
        self.queries: list[tuple[str, str, str]] = []  # (api, query, mode)
        self.query_params: list[QueryParam] = []
        self.kb_active_query_config: dict[str, object] = {}
        self.kb_active_config_version_id = None
        self.kb_active_parser_hash = None
        self.kb_active_index_hash = None
        self.kb_active_query_hash = None
        self.llm_response_cache = None
        self._fail_preprocess = fail_preprocess
        self._fail_secondary = fail_secondary

    async def finalize_storages(self) -> None:
        return None

    async def aquery_llm(self, query: str, *, param):
        self.queries.append(("llm", query, param.mode))
        self.query_params.append(param)
        return {
            "llm_response": {
                "content": f"single-path-answer-from-{self.workspace}",
                "is_streaming": False,
            },
            "data": {
                "references": [
                    {"reference_id": "1", "file_path": f"{self.workspace}/zh.pdf"}
                ],
                "chunks": [
                    {"reference_id": "1", "content": f"中文证据 from {self.workspace}"}
                ],
            },
        }

    async def aquery_data(self, query: str, *, param):
        self.queries.append(("data", query, param.mode))
        self.query_params.append(param)
        if _bilingual.contains_cjk(query):
            chunk = {
                "reference_id": "1",
                "chunk_id": f"{self.workspace}-zh-c1",
                "content": f"中文证据 from {self.workspace}",
                "file_path": f"{self.workspace}/zh.pdf",
            }
            entities = [{"entity_name": "硫化促进剂", "reference_id": "1"}]
        else:
            if self._fail_secondary:
                raise RuntimeError("secondary retrieval boom")
            chunk = {
                "reference_id": "1",
                "chunk_id": f"{self.workspace}-en-c1",
                "content": f"english evidence from {self.workspace}",
                "file_path": f"{self.workspace}/en.pdf",
            }
            entities = [
                {"entity_name": "vulcanization accelerator", "reference_id": "1"}
            ]
        return {
            "status": "success",
            "message": "ok",
            "data": {
                "entities": entities,
                "relationships": [
                    {"src_id": entities[0]["entity_name"], "tgt_id": "rubber", "reference_id": "1"}
                ],
                "chunks": [chunk],
                "references": [
                    {"reference_id": "1", "file_path": chunk["file_path"]}
                ],
            },
            "metadata": {"query_mode": param.mode},
        }

    def _build_global_config(self):
        async def fake_query_llm(
            query,
            *,
            system_prompt=None,
            history_messages=None,
            enable_cot=True,
            stream=False,
            response_format=None,
            _priority=None,
        ):
            if system_prompt == _bilingual.BILINGUAL_PREPROCESS_SYSTEM_PROMPT:
                if self._fail_preprocess:
                    return "not json {{{"
                return json.dumps(
                    {
                        "query_zh": "耐高温胎侧橡胶配方推荐",
                        "query_en": "heat resistant sidewall rubber formula recommendation",
                        "hl_keywords_zh": ["橡胶配方"],
                        "ll_keywords_zh": ["耐高温", "胎侧"],
                        "hl_keywords_en": ["rubber formulation"],
                        "ll_keywords_en": ["heat resistance", "sidewall"],
                    },
                    ensure_ascii=False,
                )
            markers = [
                marker
                for marker in ("中文证据", "english evidence")
                if marker in (system_prompt or "")
            ]
            answer = "bilingual-synth[" + "|".join(markers) + "]: " + query
            if stream:

                async def _gen():
                    mid = len(answer) // 2
                    yield answer[:mid]
                    yield answer[mid:]

                return _gen()
            return answer

        return {
            "role_llm_funcs": {"query": fake_query_llm},
            "tokenizer": None,
            "max_total_tokens": 30000,
            "min_rerank_score": 0.0,
            "rerank_model_func": None,
        }


def _build_client(
    tmp_path: Path,
    *,
    active_query_config: dict | None = None,
    fail_preprocess: bool = False,
    fail_secondary: bool = False,
):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kb.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    document_service = DocumentLifecycleService(
        kb_service, metadata_store, tmp_path / "inputs"
    )
    job_service = JobService(kb_service, metadata_store)
    instances: dict[str, BilingualFakeRAG] = {}

    async def build(record):
        rag = BilingualFakeRAG(
            record.workspace,
            fail_preprocess=fail_preprocess,
            fail_secondary=fail_secondary,
        )
        if active_query_config:
            rag.kb_active_query_config = dict(active_query_config)
        instances[record.id] = rag
        return rag

    async def finalize(rag):
        return None

    registry = LightRAGInstanceRegistry(kb_service, build, finalize)
    app = FastAPI()
    app.include_router(
        create_kb_routes(kb_service, registry, api_key=_API_KEY, job_service=job_service)
    )
    app.include_router(
        create_kb_query_routes(document_service, registry, api_key=_API_KEY)
    )
    return TestClient(app), instances, document_service, registry


def _create_kb(client: TestClient, kb_id: str):
    response = client.post("/kbs", json={"id": kb_id, "name": kb_id}, headers=_HEADERS)
    assert response.status_code == 200, response.text
    return response.json()


def _data_calls(rag: BilingualFakeRAG) -> list[tuple[str, str, str]]:
    return [entry for entry in rag.queries if entry[0] == "data"]


# ---------------------------------------------------------------------------
# Service unit tests
# ---------------------------------------------------------------------------


def test_contains_cjk_and_query_language():
    assert _bilingual.contains_cjk("胎侧橡胶")
    assert _bilingual.contains_cjk("NR/BR并用比")
    assert not _bilingual.contains_cjk("pure english 123")
    assert _bilingual.query_language(_ZH_QUERY) == "zh"
    assert _bilingual.query_language(_EN_QUERY) == "en"


def test_resolve_bilingual_mode_matrix(_pin_global_args):
    # request flag wins
    assert _bilingual.resolve_bilingual_mode(True, "off") == "on"
    assert _bilingual.resolve_bilingual_mode(False, "on") == "off"
    # KB config next
    assert _bilingual.resolve_bilingual_mode(None, "on") == "on"
    assert _bilingual.resolve_bilingual_mode(None, "off") == "off"
    assert _bilingual.resolve_bilingual_mode(None, "AUTO") == "auto"
    # invalid KB value falls through to env default
    assert _bilingual.resolve_bilingual_mode(None, "banana") == "auto"
    assert _bilingual.resolve_bilingual_mode(None, None) == "auto"
    # env default mode respected
    _pin_global_args.bilingual_query_default_mode = "off"
    assert _bilingual.resolve_bilingual_mode(None, None) == "off"
    # master kill-switch beats everything
    _pin_global_args.bilingual_query_enabled = False
    assert _bilingual.resolve_bilingual_mode(True, "on") == "off"


def test_bilingual_applies_guards():
    assert not _bilingual.bilingual_applies("off", _ZH_QUERY)
    # bypass / only_need_context / caller keywords force single-path
    param = QueryParam(mode="bypass")
    assert not _bilingual.bilingual_applies("on", _ZH_QUERY, param)
    param = QueryParam(mode="mix", only_need_context=True)
    assert not _bilingual.bilingual_applies("on", _ZH_QUERY, param)
    param = QueryParam(mode="mix", hl_keywords=["已有关键词"])
    assert not _bilingual.bilingual_applies("on", _ZH_QUERY, param)
    # auto: CJK only
    param = QueryParam(mode="mix")
    assert _bilingual.bilingual_applies("auto", _ZH_QUERY, param)
    assert not _bilingual.bilingual_applies("auto", _EN_QUERY, param)
    # on: always
    assert _bilingual.bilingual_applies("on", _EN_QUERY, param)


def test_plan_from_model_language_routing():
    model = _bilingual._BilingualPlanModel(
        query_zh="中文问句",
        query_en="english question",
        hl_keywords_zh=["主题"],
        ll_keywords_zh=["实体"],
        hl_keywords_en=["topic"],
        ll_keywords_en=["entity"],
    )
    plan = _bilingual._plan_from_model(model, _ZH_QUERY)
    assert plan is not None
    assert plan.source_language == "zh"
    assert plan.primary_query == _ZH_QUERY  # original text, not the rewrite
    assert plan.secondary_query == "english question"
    assert plan.hl_primary == ["主题"] and plan.ll_secondary == ["entity"]

    plan_en = _bilingual._plan_from_model(model, _EN_QUERY)
    assert plan_en is not None
    assert plan_en.source_language == "en"
    assert plan_en.secondary_query == "中文问句"
    assert plan_en.hl_primary == ["topic"]

    # unusable translation -> None
    empty = _bilingual._BilingualPlanModel(query_zh="中文问句", query_en="")
    assert _bilingual._plan_from_model(empty, _ZH_QUERY) is None
    same = _bilingual._BilingualPlanModel(query_zh="x", query_en=_ZH_QUERY)
    assert _bilingual._plan_from_model(same, _ZH_QUERY) is None


def test_merge_chunk_lists_dedups_and_tags_path():
    primary = [
        {"chunk_id": "c1", "content": "a", "file_path": "f1"},
        {"chunk_id": "c2", "content": "b", "file_path": "f1"},
    ]
    secondary = [
        {"chunk_id": "c2", "content": "b", "file_path": "f1"},  # duplicate
        {"chunk_id": "c3", "content": "c", "file_path": "f2"},
    ]
    merged = _bilingual.merge_chunk_lists(primary, secondary)
    assert [chunk["chunk_id"] for chunk in merged] == ["c1", "c2", "c3"]
    assert merged[0]["retrieval_path"] == "primary"
    assert merged[2]["retrieval_path"] == "secondary"
    # no chunk_id -> content-hash dedup
    merged2 = _bilingual.merge_chunk_lists(
        [{"content": "same", "file_path": "f"}], [{"content": "same", "file_path": "f"}]
    )
    assert len(merged2) == 1


def test_alt_retrieval_param_keyword_seeding():
    param = QueryParam(mode="mix", hl_keywords=["原有"], ll_keywords=[])
    alt = _bilingual.alt_retrieval_param(param, "english alt", [], [])
    # kg mode + empty alt keywords -> alt query seeded as ll keyword so the
    # core never fires a hidden keyword-extraction LLM call
    assert alt.ll_keywords == ["english alt"]
    assert param.hl_keywords == ["原有"]  # original untouched
    alt2 = _bilingual.alt_retrieval_param(param, "english alt", ["topic"], ["entity"])
    assert alt2.hl_keywords == ["topic"] and alt2.ll_keywords == ["entity"]
    naive = QueryParam(mode="naive")
    alt3 = _bilingual.alt_retrieval_param(naive, "english alt", [], [])
    assert alt3.ll_keywords == []  # naive ignores keywords entirely


def test_usable_alt_query():
    assert _bilingual.usable_alt_query("q", None) is None
    assert _bilingual.usable_alt_query("q", "  ") is None
    assert _bilingual.usable_alt_query("query one", "query one") is None
    assert _bilingual.usable_alt_query("query one", " other ") == "other"


def test_prepare_bilingual_queries_success_and_failure(tmp_path):
    rag = BilingualFakeRAG("ws")
    plan = asyncio.run(_bilingual.prepare_bilingual_queries(rag, _ZH_QUERY))
    assert plan is not None
    assert plan.source_language == "zh"
    assert plan.secondary_query.startswith("heat resistant")
    assert plan.hl_primary == ["橡胶配方"]

    failing = BilingualFakeRAG("ws", fail_preprocess=True)
    assert asyncio.run(_bilingual.prepare_bilingual_queries(failing, _ZH_QUERY)) is None


def test_prepare_bilingual_queries_timeout_falls_back():
    rag = BilingualFakeRAG("ws")
    original_config = rag._build_global_config

    def slow_config():
        config = original_config()
        inner = config["role_llm_funcs"]["query"]

        async def slow(*args, **kwargs):
            await asyncio.sleep(0.5)
            return await inner(*args, **kwargs)

        config["role_llm_funcs"]["query"] = slow
        return config

    rag._build_global_config = slow_config
    plan = asyncio.run(
        _bilingual.prepare_bilingual_queries(rag, _ZH_QUERY, timeout=0.05)
    )
    assert plan is None


def test_dual_aquery_data_tolerates_secondary_failure():
    rag = BilingualFakeRAG("ws", fail_secondary=True)
    plan = asyncio.run(_bilingual.prepare_bilingual_queries(rag, _ZH_QUERY))
    assert plan is not None
    dual = asyncio.run(
        _bilingual.dual_aquery_data(rag, _ZH_QUERY, QueryParam(mode="mix"), plan)
    )
    assert dual.secondary_failed is True
    assert len(dual.primary_chunks) == 1
    assert dual.merged_chunks()[0]["chunk_id"] == "ws-zh-c1"
    info = dual.info()
    assert info["secondary_failed"] is True and info["secondary_chunks"] == 0


# ---------------------------------------------------------------------------
# KB config plumbing
# ---------------------------------------------------------------------------


def test_query_config_bilingual_query_validation():
    runtime = _cvs._active_query_runtime_config(
        {"query_config": {"bilingual_query": "ON"}}
    )
    assert runtime["bilingual_query"] == "on"
    with pytest.raises(ValueError):
        _cvs._active_query_runtime_config({"query_config": {"bilingual_query": "yes"}})
    # accepted by the section whitelist (would raise otherwise)
    _cvs._reject_unknown_section_keys(
        {"query_config": {"bilingual_query": "auto", "top_k": 5}},
        "query_config",
        _cvs._ACTIVE_QUERY_CONFIG_KEYS | _cvs._QUERY_CONFIG_EXTRA_KEYS,
    )


def test_bilingual_query_never_leaks_into_query_param_defaults():
    rag = SimpleNamespace(
        kb_active_query_config={"bilingual_query": "on", "top_k": 9}
    )
    defaults = _cvs.active_query_defaults_from_rag(rag)
    assert defaults == {"top_k": 9}
    assert _bilingual.bilingual_mode_from_rag(rag) == "on"


# ---------------------------------------------------------------------------
# Single-KB routes
# ---------------------------------------------------------------------------


def test_kb_query_dual_path_merges_both_languages(tmp_path):
    client, instances, *_ = _build_client(
        tmp_path, active_query_config={"bilingual_query": "on"}
    )
    _create_kb(client, "kb_bi")
    response = client.post(
        "/kbs/kb_bi/query",
        json={"query": _ZH_QUERY, "mode": "mix"},
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    # synthesis saw evidence from BOTH language paths
    assert "中文证据" in body["response"] and "english evidence" in body["response"]
    # references rebuilt over the merged pool (zh + en files)
    ref_paths = {ref["file_path"] for ref in body["references"]}
    workspace = instances["kb_bi"].workspace
    assert ref_paths == {f"{workspace}/zh.pdf", f"{workspace}/en.pdf"}
    info = body["metadata"]["bilingual"]
    assert info["enabled"] is True
    assert info["source_language"] == "zh"
    assert info["primary_chunks"] == 1 and info["secondary_chunks"] == 1
    assert info["merged_chunks"] == 2 and info["final_chunks"] == 2
    assert info["translated_query"].startswith("heat resistant")
    # two retrievals: original zh query + translated en query, keywords seeded
    calls = _data_calls(instances["kb_bi"])
    assert len(calls) == 2
    assert {call[1] for call in calls} == {
        _ZH_QUERY,
        "heat resistant sidewall rubber formula recommendation",
    }
    seeded = [p for p in instances["kb_bi"].query_params if p.hl_keywords]
    assert any(p.hl_keywords == ["橡胶配方"] for p in seeded)
    assert any(p.hl_keywords == ["rubber formulation"] for p in seeded)


def test_kb_query_auto_mode_skips_english_query(tmp_path):
    client, instances, *_ = _build_client(tmp_path)  # env default auto
    _create_kb(client, "kb_auto")
    response = client.post(
        "/kbs/kb_auto/query",
        json={"query": _EN_QUERY, "mode": "mix"},
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["response"].startswith("single-path-answer-from-")
    assert "bilingual" not in body["metadata"]
    assert instances["kb_auto"].queries[0][0] == "llm"  # normal aquery_llm path


def test_kb_query_auto_mode_dual_for_chinese_query(tmp_path):
    client, instances, *_ = _build_client(tmp_path)  # env default auto
    _create_kb(client, "kb_auto_zh")
    response = client.post(
        "/kbs/kb_auto_zh/query",
        json={"query": _ZH_QUERY, "mode": "mix"},
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    assert response.json()["metadata"]["bilingual"]["enabled"] is True
    assert len(_data_calls(instances["kb_auto_zh"])) == 2


def test_kb_query_master_switch_off_stays_single_path(tmp_path, _pin_global_args):
    _pin_global_args.bilingual_query_enabled = False
    client, instances, *_ = _build_client(
        tmp_path, active_query_config={"bilingual_query": "on"}
    )
    _create_kb(client, "kb_off")
    response = client.post(
        "/kbs/kb_off/query",
        json={"query": _ZH_QUERY, "bilingual": True},
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    assert "bilingual" not in response.json()["metadata"]
    assert instances["kb_off"].queries[0][0] == "llm"


def test_kb_query_request_false_overrides_kb_on(tmp_path):
    client, instances, *_ = _build_client(
        tmp_path, active_query_config={"bilingual_query": "on"}
    )
    _create_kb(client, "kb_override")
    response = client.post(
        "/kbs/kb_override/query",
        json={"query": _ZH_QUERY, "bilingual": False},
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    assert "bilingual" not in response.json()["metadata"]
    assert instances["kb_override"].queries[0][0] == "llm"


def test_kb_query_preprocess_failure_falls_back_single_path(tmp_path):
    client, instances, *_ = _build_client(
        tmp_path,
        active_query_config={"bilingual_query": "on"},
        fail_preprocess=True,
    )
    _create_kb(client, "kb_fb")
    response = client.post(
        "/kbs/kb_fb/query",
        json={"query": _ZH_QUERY},
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["response"].startswith("single-path-answer-from-")
    info = body["metadata"]["bilingual"]
    assert info["enabled"] is False
    assert info["reason"] == "preprocess_unavailable"
    # fell back to the normal aquery_llm path
    assert [entry[0] for entry in instances["kb_fb"].queries if entry[0] == "llm"]


def test_kb_query_explicit_keywords_stay_single_path(tmp_path):
    client, instances, *_ = _build_client(
        tmp_path, active_query_config={"bilingual_query": "on"}
    )
    _create_kb(client, "kb_kw")
    response = client.post(
        "/kbs/kb_kw/query",
        json={"query": _ZH_QUERY, "hl_keywords": ["调用方自带"]},
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    assert "bilingual" not in response.json()["metadata"]
    assert instances["kb_kw"].queries[0][0] == "llm"


def test_kb_query_stream_dual_path_first_line_metadata(tmp_path):
    client, instances, *_ = _build_client(
        tmp_path, active_query_config={"bilingual_query": "on"}
    )
    _create_kb(client, "kb_stream")
    with client.stream(
        "POST",
        "/kbs/kb_stream/query/stream",
        json={"query": _ZH_QUERY, "stream": True},
        headers=_HEADERS,
    ) as response:
        assert response.status_code == 200
        body = b"".join(response.iter_bytes()).decode("utf-8")
    lines = [json.loads(line) for line in body.split("\n") if line]
    head = lines[0]
    assert head["kb_id"] == "kb_stream"
    assert head["metadata"]["bilingual"]["enabled"] is True
    assert {ref["file_path"].rsplit("/", 1)[-1] for ref in head["references"]} == {
        "zh.pdf",
        "en.pdf",
    }
    answer = "".join(line.get("response", "") for line in lines[1:])
    assert "中文证据" in answer and "english evidence" in answer
    assert len(_data_calls(instances["kb_stream"])) == 2


def test_kb_query_data_merges_entities_and_rebuilds_references(tmp_path):
    client, instances, *_ = _build_client(
        tmp_path, active_query_config={"bilingual_query": "on"}
    )
    _create_kb(client, "kb_data")
    response = client.post(
        "/kbs/kb_data/query/data",
        json={"query": _ZH_QUERY, "mode": "mix"},
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    data = body["data"]
    entity_names = {entity["entity_name"] for entity in data["entities"]}
    assert entity_names == {"硫化促进剂", "vulcanization accelerator"}
    # per-path reference numbering cannot survive the merge
    assert all(entity["reference_id"] is None for entity in data["entities"])
    assert len(data["chunks"]) == 2
    chunk_refs = {chunk["reference_id"] for chunk in data["chunks"]}
    ref_ids = {ref["reference_id"] for ref in data["references"]}
    assert chunk_refs == ref_ids and len(ref_ids) == 2
    assert body["metadata"]["bilingual"]["merged_chunks"] == 2


def test_kb_query_bypass_mode_skips_bilingual(tmp_path):
    client, instances, *_ = _build_client(
        tmp_path, active_query_config={"bilingual_query": "on"}
    )
    _create_kb(client, "kb_bypass")
    response = client.post(
        "/kbs/kb_bypass/query",
        json={"query": _ZH_QUERY, "mode": "bypass"},
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    assert "bilingual" not in response.json()["metadata"]
    assert instances["kb_bypass"].queries[0][0] == "llm"


# ---------------------------------------------------------------------------
# Multi-KB routes
# ---------------------------------------------------------------------------


def test_multi_kb_query_dual_path_per_kb(tmp_path):
    client, instances, *_ = _build_client(tmp_path)
    _create_kb(client, "kb_a")
    _create_kb(client, "kb_b")
    response = client.post(
        "/kbs:query",
        json={"kb_ids": ["kb_a", "kb_b"], "query": _ZH_QUERY, "bilingual": True},
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "中文证据" in body["response"] and "english evidence" in body["response"]
    info = body["metadata"]["bilingual"]
    assert info["enabled"] is True
    assert info["per_kb_secondary_chunks"] == {"kb_a": 1, "kb_b": 1}
    assert info["secondary_chunks"] == 2
    # every KB retrieved twice (zh + en)
    assert len(_data_calls(instances["kb_a"])) == 2
    assert len(_data_calls(instances["kb_b"])) == 2
    # 2 KBs x 2 languages = 4 distinct reference files
    assert len(body["references"]) == 4


def test_multi_kb_query_without_flag_uses_env_auto(tmp_path):
    client, instances, *_ = _build_client(tmp_path)
    _create_kb(client, "kb_c")
    response = client.post(
        "/kbs:query",
        json={"kb_ids": ["kb_c"], "query": _EN_QUERY},
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    assert "bilingual" not in response.json()["metadata"]
    assert len(_data_calls(instances["kb_c"])) == 1


def test_multi_kb_retrieve_reports_bilingual_metadata(tmp_path):
    client, instances, *_ = _build_client(tmp_path)
    _create_kb(client, "kb_r")
    response = client.post(
        "/kbs:retrieve",
        json={"kb_ids": ["kb_r"], "query": _ZH_QUERY, "bilingual": True},
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["metadata"]["bilingual"]["enabled"] is True
    assert len(body["data"]["chunks"]) == 2


# ---------------------------------------------------------------------------
# Legacy global routes
# ---------------------------------------------------------------------------


def test_legacy_query_dual_path(tmp_path):
    rag = BilingualFakeRAG("legacy-ws")
    app = FastAPI()
    app.include_router(create_query_routes(rag, api_key=_API_KEY))
    client = TestClient(app)
    response = client.post(
        "/query",
        json={"query": _ZH_QUERY, "mode": "mix", "bilingual": True},
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "中文证据" in body["response"] and "english evidence" in body["response"]
    assert len(_data_calls(rag)) == 2


def test_legacy_query_default_single_path_for_english(tmp_path):
    rag = BilingualFakeRAG("legacy-ws")
    app = FastAPI()
    app.include_router(create_query_routes(rag, api_key=_API_KEY))
    client = TestClient(app)
    response = client.post(
        "/query",
        json={"query": _EN_QUERY, "mode": "mix"},
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    assert response.json()["response"].startswith("single-path-answer-from-")
    assert rag.queries[0][0] == "llm"


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------


def test_agent_plan_step_parses_alt_fields():
    step = _agent_service.AgentPlanStep.model_validate(
        {
            "step_index": 1,
            "query": "查中文法规",
            "kb_ids": ["kb_x"],
            "mode": "mix",
            "query_alt": "look up english regulations",
            "hl_keywords_alt": ["regulation"],
            "ll_keywords_alt": "REACH",  # string coerced to list
        }
    )
    assert step.query_alt == "look up english regulations"
    assert step.ll_keywords_alt == ["REACH"]
    # absent alt fields default empty (backwards compatible)
    bare = _agent_service.AgentPlanStep.model_validate(
        {"step_index": 1, "query": "查中文法规", "kb_ids": ["kb_x"], "mode": "mix"}
    )
    assert bare.query_alt == "" and bare.hl_keywords_alt == []


def test_agent_bilingual_enabled_matrix(_pin_global_args):
    body = _agent_service.AgentQueryRequest(query=_ZH_QUERY)
    assert _agent_service.agent_bilingual_enabled(body) is True  # auto + CJK
    body_en = _agent_service.AgentQueryRequest(query=_EN_QUERY)
    assert _agent_service.agent_bilingual_enabled(body_en) is False
    body_en_on = _agent_service.AgentQueryRequest(query=_EN_QUERY, bilingual=True)
    assert _agent_service.agent_bilingual_enabled(body_en_on) is True
    _pin_global_args.bilingual_query_enabled = False
    assert _agent_service.agent_bilingual_enabled(body) is False


def test_query_tool_retrieve_serial_alt_path(tmp_path):
    client, instances, document_service, registry = _build_client(tmp_path)
    _create_kb(client, "kb_tool")
    service = _query_tool.QueryToolService(document_service, registry)
    result = asyncio.run(
        service.retrieve_serial(
            http_request=SimpleNamespace(),  # enterprise off: never touched
            kb_ids=["kb_tool"],
            query=_ZH_QUERY,
            mode="mix",
            hl_keywords=["中文关键词"],
            query_alt="heat resistant sidewall rubber formula",
            hl_keywords_alt=["rubber"],
        )
    )
    assert len(result.chunks) == 2
    assert result.alt_chunk_counts == {"kb_tool": 1}
    assert result.alt_failed_kbs == []
    assert result.per_kb_chunk_counts == {"kb_tool": 1}
    calls = _data_calls(instances["kb_tool"])
    assert len(calls) == 2


def test_query_tool_retrieve_serial_alt_failure_tolerated(tmp_path):
    client, instances, document_service, registry = _build_client(
        tmp_path, fail_secondary=True
    )
    _create_kb(client, "kb_tool_fail")
    service = _query_tool.QueryToolService(document_service, registry)
    result = asyncio.run(
        service.retrieve_serial(
            http_request=SimpleNamespace(),
            kb_ids=["kb_tool_fail"],
            query=_ZH_QUERY,
            mode="mix",
            query_alt="english variant of the question",
        )
    )
    assert len(result.chunks) == 1  # primary kept
    assert result.alt_failed_kbs == ["kb_tool_fail"]


def test_query_tool_retrieve_serial_ignores_useless_alt(tmp_path):
    client, instances, document_service, registry = _build_client(tmp_path)
    _create_kb(client, "kb_tool_same")
    service = _query_tool.QueryToolService(document_service, registry)
    result = asyncio.run(
        service.retrieve_serial(
            http_request=SimpleNamespace(),
            kb_ids=["kb_tool_same"],
            query=_ZH_QUERY,
            mode="mix",
            query_alt=_ZH_QUERY,  # identical -> no second path
        )
    )
    assert len(result.chunks) == 1
    assert result.alt_chunk_counts == {}
    assert len(_data_calls(instances["kb_tool_same"])) == 1


def test_staged_factor_queries_bilingual_pairing():
    skeleton = _agent_staged.SkeletonExtract(
        components=[],
        open_questions=["中文补充问题一号", "中文补充问题二号"],
        open_questions_alt=["english follow-up one", "english follow-up two"],
    )
    requirement = _agent_staged.StagedRequirement(
        application="胎侧胶",
        target_properties=[_agent_staged.TargetProperty(name="耐热性")],
    )
    pairs = _agent_staged.AgentStagedRunner._factor_queries(
        requirement=requirement, skeleton=skeleton, allowance=5, bilingual=True
    )
    assert pairs[0] == ("中文补充问题一号", "english follow-up one")
    assert pairs[1] == ("中文补充问题二号", "english follow-up two")
    # length mismatch -> pairing disabled (no wrong positional matches)
    skeleton_mismatch = _agent_staged.SkeletonExtract(
        components=[],
        open_questions=["中文补充问题一号", "中文补充问题二号"],
        open_questions_alt=["english follow-up one"],
    )
    pairs2 = _agent_staged.AgentStagedRunner._factor_queries(
        requirement=requirement, skeleton=skeleton_mismatch, allowance=5, bilingual=True
    )
    assert pairs2[0][1] == "" and pairs2[1][1] == ""
    # bilingual off -> alt always empty
    pairs3 = _agent_staged.AgentStagedRunner._factor_queries(
        requirement=requirement, skeleton=skeleton, allowance=5, bilingual=False
    )
    assert all(alt == "" for _, alt in pairs3)


def test_staged_target_property_name_alt():
    prop = _agent_staged.TargetProperty.model_validate(
        {"name": "耐热老化", "priority": "P0", "name_alt": "heat aging resistance"}
    )
    assert prop.name_alt == "heat aging resistance"
    bare = _agent_staged.TargetProperty.model_validate({"name": "耐热老化"})
    assert bare.name_alt == ""


# ---------------------------------------------------------------------------
# Dedicated bilingual translation LLM role (BILINGUAL_LLM_*)
# ---------------------------------------------------------------------------


def test_bilingual_role_registered_in_roles_registry():
    from lightrag.llm_roles import ROLES_BY_NAME

    spec = ROLES_BY_NAME.get("bilingual")
    assert spec is not None
    assert spec.env_prefix == "BILINGUAL"


def test_resolve_translation_llm_prefers_bilingual_role():
    async def bilingual_func(*args, **kwargs):
        return ""

    async def query_func(*args, **kwargs):
        return ""

    func, role = _bilingual.resolve_translation_llm(
        {"role_llm_funcs": {"bilingual": bilingual_func, "query": query_func}}
    )
    assert func is bilingual_func and role == "bilingual"
    func, role = _bilingual.resolve_translation_llm(
        {"role_llm_funcs": {"query": query_func}}
    )
    assert func is query_func and role == "query"
    func, role = _bilingual.resolve_translation_llm({"role_llm_funcs": {}})
    assert func is None and role == "query"


def test_prepare_bilingual_queries_uses_dedicated_role_when_present():
    rag = BilingualFakeRAG("ws")
    original_config = rag._build_global_config
    preprocess_calls: list[str] = []

    def config_with_bilingual_role():
        config = original_config()
        query_only = config["role_llm_funcs"]["query"]

        async def dedicated_translator(prompt, *, system_prompt=None, **kwargs):
            preprocess_calls.append("bilingual")
            return await query_only(prompt, system_prompt=system_prompt, **kwargs)

        async def query_must_not_translate(prompt, *, system_prompt=None, **kwargs):
            assert system_prompt != _bilingual.BILINGUAL_PREPROCESS_SYSTEM_PROMPT, (
                "preprocessing must go through the bilingual role"
            )
            return await query_only(prompt, system_prompt=system_prompt, **kwargs)

        config["role_llm_funcs"] = {
            "bilingual": dedicated_translator,
            "query": query_must_not_translate,
        }
        return config

    rag._build_global_config = config_with_bilingual_role
    plan = asyncio.run(_bilingual.prepare_bilingual_queries(rag, _ZH_QUERY))
    assert plan is not None
    assert preprocess_calls == ["bilingual"]


def test_backfill_bilingual_role_args_inherits_query_fields():
    from lightrag.api.config import _backfill_bilingual_role_args

    args = SimpleNamespace(
        # QUERY role fully configured
        query_llm_binding="openai",
        query_llm_model="qwen3-32b",
        query_llm_binding_host="http://192.168.1.66:8000/v1",
        query_llm_binding_api_key="query-key",
        query_llm_max_async=4,
        query_llm_timeout=120,
        query_aws_region=None,
        query_aws_access_key_id=None,
        query_aws_secret_access_key=None,
        query_aws_session_token=None,
        # BILINGUAL: only the model explicitly overridden
        bilingual_llm_binding=None,
        bilingual_llm_model="nllb-translator",
        bilingual_llm_binding_host=None,
        bilingual_llm_binding_api_key=None,
        bilingual_llm_max_async=None,
        bilingual_llm_timeout=None,
        bilingual_aws_region=None,
        bilingual_aws_access_key_id=None,
        bilingual_aws_secret_access_key=None,
        bilingual_aws_session_token=None,
    )
    _backfill_bilingual_role_args(args)
    # explicit value wins
    assert args.bilingual_llm_model == "nllb-translator"
    # unset fields inherit the QUERY role per-field
    assert args.bilingual_llm_binding == "openai"
    assert args.bilingual_llm_binding_host == "http://192.168.1.66:8000/v1"
    assert args.bilingual_llm_binding_api_key == "query-key"
    assert args.bilingual_llm_max_async == 4
    assert args.bilingual_llm_timeout == 120
    # QUERY-unset fields stay None (standard base-LLM fallback applies later)
    assert args.bilingual_aws_region is None


def test_bilingual_role_in_kb_llm_role_config_and_query_hash():
    # KB-level llm_role_config accepts the bilingual role...
    runtime = _cvs._active_llm_role_runtime_config(
        {"llm_role_config": {"bilingual": "dedicated-translator"}}
    )
    assert runtime == {"bilingual": {"model": "dedicated-translator"}}
    # ...and its identity participates in query_hash (not index_hash)
    assert "bilingual" in _cvs._QUERY_AFFECTING_ROLES
    assert "bilingual" not in _cvs._INDEX_AFFECTING_ROLES
    base_hash = _cvs._active_query_runtime_hash({})
    override_hash = _cvs._active_query_runtime_hash(
        {"llm_role_config": {"bilingual": "dedicated-translator"}}
    )
    assert base_hash != override_hash
