"""Bilingual (zh <-> en) dual-path query service.

Implements the query-side half of ``docs/BilingualQuery-zh.md``: one LLM
preprocessing call produces a translated query plus bilingual hl/ll keyword
sets, then retrieval runs twice (original query + same-language keywords,
translated query + other-language keywords) and the chunk pools are merged,
deduplicated, reranked against the original query and truncated by the
existing budgets before a single synthesis call answers in the language of
the original question.

Design invariants:

- Fail-open everywhere: a failed/timed-out preprocessing call or a failed
  secondary retrieval degrades to today's single-path behaviour, never to an
  error the user sees.
- Zero extra LLM calls for kg modes: the preprocessing call *replaces* the
  core keyword-extraction call because both paths pre-seed
  ``QueryParam.hl_keywords`` / ``ll_keywords`` (``get_keywords_from_query``
  skips extraction when keywords are provided).
- No core-module changes: only stable public surfaces are used
  (``aquery_data``, ``QueryParam`` keyword injection,
  ``process_chunks_unified`` / ``generate_reference_list_from_chunks``).
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from pydantic import BaseModel, Field, field_validator

from lightrag.api.llm_json_utils import call_llm_json
from lightrag.base import QueryParam
from lightrag.constants import DEFAULT_QUERY_PRIORITY
from lightrag.prompt import PROMPTS
from lightrag.sensitive_context import (
    SensitiveContext,
    SensitiveContextPayload,
    bind_sensitive_context_endpoint,
    mark_sensitive_context_not_used,
    serialize_sensitive_final_request,
)
from lightrag.utils import (
    CacheData,
    compute_args_hash,
    generate_reference_list_from_chunks,
    get_llm_cache_identity,
    handle_cache,
    logger,
    process_chunks_unified,
    save_to_cache,
    serialize_llm_cache_identity,
)

BILINGUAL_MODES = ("off", "auto", "on")
_KG_MODES = {"local", "global", "hybrid", "mix"}
_PREPROCESS_LLM_ATTEMPTS = 2
_MAX_KEYWORDS = 12
_MAX_KEYWORD_CHARS = 120
_MAX_QUERY_CHARS = 4096
# Version tag participates in the cache hash so a prompt change invalidates
# previously cached preprocessing results.
_PREPROCESS_CACHE_SEED = "bilingual_query_preprocess_v1"

BILINGUAL_PREPROCESS_SYSTEM_PROMPT = """
你是 RAG 系统的双语查询预处理器。给定一个用户查询，输出严格 JSON，且只输出 JSON：
{"query_zh": "...", "query_en": "...", "hl_keywords_zh": [], "ll_keywords_zh": [], "hl_keywords_en": [], "ll_keywords_en": []}
规则：
1. query_zh 与 query_en 分别是该查询的中文与英文完整问句；原句已是该语言时照抄原句，不要改写、不要补充。
2. hl_keywords 是宏观概念、主题或意图层面的关键词；ll_keywords 是具体实体、专有名词、技术术语、产品或牌号名。
3. 关键词只能来源于查询本身，禁止虚构；两种语言的关键词语义一一对应，优先使用领域通用译法。
4. 型号、代号、化学式、标准号、数值等无需翻译的记号在两种语言中原样保留。
5. 不要输出 markdown 代码块，不要输出解释或思维链。
""".strip()


# ---------------------------------------------------------------------------
# Configuration accessors (read lazily so tests can monkeypatch global_args)
# ---------------------------------------------------------------------------


def _global_args() -> Any:
    from lightrag.api import config as api_config

    return api_config.global_args


def bilingual_query_master_enabled() -> bool:
    return bool(getattr(_global_args(), "bilingual_query_enabled", False))


def bilingual_query_default_mode() -> str:
    value = str(
        getattr(_global_args(), "bilingual_query_default_mode", "auto") or "auto"
    ).strip().lower()
    return value if value in BILINGUAL_MODES else "auto"


def bilingual_query_timeout() -> float:
    try:
        value = float(getattr(_global_args(), "bilingual_query_timeout", 12) or 12)
    except (TypeError, ValueError):
        return 12.0
    return max(1.0, value)


# ---------------------------------------------------------------------------
# Language detection and mode resolution
# ---------------------------------------------------------------------------

_CJK_RANGES = (
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # Extension A
    (0xF900, 0xFAFF),  # Compatibility Ideographs
)


def contains_cjk(text: str) -> bool:
    for char in text or "":
        code = ord(char)
        for start, end in _CJK_RANGES:
            if start <= code <= end:
                return True
    return False


def query_language(text: str) -> str:
    """Primary language of a query: ``zh`` when it contains any CJK char."""
    return "zh" if contains_cjk(text) else "en"


def normalize_bilingual_mode(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if normalized in BILINGUAL_MODES else None


def bilingual_mode_from_rag(rag: Any) -> str | None:
    """Read ``query_config.bilingual_query`` from the KB's active config."""
    query_config = getattr(rag, "kb_active_query_config", None)
    if isinstance(query_config, dict):
        return normalize_bilingual_mode(query_config.get("bilingual_query"))
    return None


def resolve_bilingual_mode(
    request_flag: bool | None, kb_mode: Any = None
) -> str:
    """Effective mode: request override > KB config > env default.

    The env master switch is a kill switch: when off, everything is off
    regardless of KB config or request flags.
    """
    if not bilingual_query_master_enabled():
        return "off"
    if request_flag is True:
        return "on"
    if request_flag is False:
        return "off"
    kb_normalized = normalize_bilingual_mode(kb_mode)
    if kb_normalized is not None:
        return kb_normalized
    return bilingual_query_default_mode()


def bilingual_applies(
    mode: str, query: str, param: QueryParam | None = None
) -> bool:
    """Whether dual-path retrieval should run for this specific query."""
    if mode not in ("auto", "on"):
        return False
    if param is not None:
        # bypass has no retrieval; only_need_context/prompt keep core
        # semantics; caller-supplied keywords mean the caller controls
        # retrieval focus — respect all three by staying single-path.
        if getattr(param, "mode", None) == "bypass":
            return False
        if getattr(param, "only_need_context", False) or getattr(
            param, "only_need_prompt", False
        ):
            return False
        if getattr(param, "hl_keywords", None) or getattr(param, "ll_keywords", None):
            return False
    if mode == "on":
        return True
    return contains_cjk(query)


# ---------------------------------------------------------------------------
# Preprocessing (one LLM call -> translated query + bilingual keywords)
# ---------------------------------------------------------------------------


class _BilingualPlanModel(BaseModel):
    query_zh: str = ""
    query_en: str = ""
    hl_keywords_zh: list[str] = Field(default_factory=list)
    ll_keywords_zh: list[str] = Field(default_factory=list)
    hl_keywords_en: list[str] = Field(default_factory=list)
    ll_keywords_en: list[str] = Field(default_factory=list)

    @field_validator("query_zh", "query_en", mode="before")
    @classmethod
    def _clip_query(cls, value: Any) -> str:
        return str(value if value is not None else "").strip()[:_MAX_QUERY_CHARS]

    @field_validator(
        "hl_keywords_zh",
        "ll_keywords_zh",
        "hl_keywords_en",
        "ll_keywords_en",
        mode="before",
    )
    @classmethod
    def _coerce_keywords(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            raw: list[Any] = [value]
        elif isinstance(value, list):
            raw = value
        elif value is None:
            raw = []
        else:
            raw = [value]
        items: list[str] = []
        for entry in raw:
            text = str(entry).strip()[:_MAX_KEYWORD_CHARS]
            if text and text not in items:
                items.append(text)
            if len(items) >= _MAX_KEYWORDS:
                break
        return items


@dataclass(slots=True)
class BilingualQueryPlan:
    """Validated output of the preprocessing call, keyed by language."""

    source_language: str  # "zh" | "en" — language of the original query
    primary_query: str  # always the ORIGINAL query text, never a rewrite
    secondary_query: str
    hl_primary: list[str] = field(default_factory=list)
    ll_primary: list[str] = field(default_factory=list)
    hl_secondary: list[str] = field(default_factory=list)
    ll_secondary: list[str] = field(default_factory=list)
    from_cache: bool = False

    @property
    def secondary_language(self) -> str:
        return "en" if self.source_language == "zh" else "zh"


def _plan_from_model(
    model: _BilingualPlanModel, original_query: str, *, from_cache: bool = False
) -> BilingualQueryPlan | None:
    source = query_language(original_query)
    if source == "zh":
        secondary_query = model.query_en
        hl_primary, ll_primary = model.hl_keywords_zh, model.ll_keywords_zh
        hl_secondary, ll_secondary = model.hl_keywords_en, model.ll_keywords_en
    else:
        secondary_query = model.query_zh
        hl_primary, ll_primary = model.hl_keywords_en, model.ll_keywords_en
        hl_secondary, ll_secondary = model.hl_keywords_zh, model.ll_keywords_zh
    secondary_query = secondary_query.strip()
    # No usable translation -> nothing a second path could add.
    if len(secondary_query) < 2 or secondary_query == original_query.strip():
        return None
    return BilingualQueryPlan(
        source_language=source,
        primary_query=original_query,
        secondary_query=secondary_query,
        hl_primary=hl_primary,
        ll_primary=ll_primary,
        hl_secondary=hl_secondary,
        ll_secondary=ll_secondary,
        from_cache=from_cache,
    )


def _plan_cache_payload(model: _BilingualPlanModel) -> str:
    return json.dumps(model.model_dump(), ensure_ascii=False, sort_keys=True)


def resolve_translation_llm(
    global_config: dict[str, Any],
) -> tuple[Any | None, str]:
    """Pick the LLM func for the preprocessing (translation) call.

    Prefers the dedicated ``bilingual`` role — which the API env layer
    defaults to the QUERY role's model when no BILINGUAL_LLM_* is configured
    — and falls back to the ``query`` role for callers (tests, embedded use)
    that never registered a bilingual role. Returns ``(func, role_name)``.
    """
    funcs = global_config.get("role_llm_funcs") or {}
    func = funcs.get("bilingual")
    if func is not None:
        return func, "bilingual"
    return funcs.get("query"), "query"


async def prepare_bilingual_queries(
    rag: Any,
    query: str,
    *,
    timeout: float | None = None,
) -> BilingualQueryPlan | None:
    """Run (or fetch from cache) the bilingual preprocessing call.

    Returns ``None`` on any failure — the caller must then fall back to the
    standard single-path flow. Never raises.

    The call always goes out with ``enable_cot=False`` (via ``call_llm_json``),
    so chain-of-thought/thinking stays off for translation regardless of the
    model behind the role.
    """
    try:
        global_config = rag._build_global_config()
        query_func, llm_role = resolve_translation_llm(global_config)
        if query_func is None:
            logger.warning(
                "Bilingual preprocess skipped: no bilingual/query role LLM available"
            )
            return None

        hashing_kv = getattr(rag, "llm_response_cache", None)
        llm_identity = get_llm_cache_identity(global_config, llm_role)
        args_hash = compute_args_hash(
            _PREPROCESS_CACHE_SEED,
            query,
            "\n<llm_identity>\n",
            serialize_llm_cache_identity(llm_identity),
        )
        if hashing_kv is not None:
            cached = await handle_cache(
                hashing_kv, args_hash, query, "bilingual", cache_type="query_preprocess"
            )
            if cached is not None:
                try:
                    model = _BilingualPlanModel.model_validate(json.loads(cached[0]))
                    return _plan_from_model(model, query, from_cache=True)
                except Exception as exc:  # noqa: BLE001 — treat as cache miss
                    logger.warning("Bilingual preprocess cache entry invalid: %s", exc)

        payload = json.dumps(
            {
                "query": query,
                "output_schema": {
                    "query_zh": "中文完整问句",
                    "query_en": "English full question",
                    "hl_keywords_zh": [],
                    "ll_keywords_zh": [],
                    "hl_keywords_en": [],
                    "ll_keywords_en": [],
                },
            },
            ensure_ascii=False,
        )
        model = await asyncio.wait_for(
            call_llm_json(
                query_func,
                payload,
                system_prompt=BILINGUAL_PREPROCESS_SYSTEM_PROMPT,
                priority=DEFAULT_QUERY_PRIORITY,
                parse=_BilingualPlanModel.model_validate,
                attempts=_PREPROCESS_LLM_ATTEMPTS,
                label="bilingual_preprocess",
            ),
            timeout=timeout if timeout is not None else bilingual_query_timeout(),
        )
        plan = _plan_from_model(model, query)
        if plan is not None and hashing_kv is not None:
            try:
                await save_to_cache(
                    hashing_kv,
                    CacheData(
                        args_hash=args_hash,
                        content=_plan_cache_payload(model),
                        prompt=query,
                        mode="bilingual",
                        cache_type="query_preprocess",
                    ),
                )
            except Exception as exc:  # noqa: BLE001 — cache write is best-effort
                logger.warning("Bilingual preprocess cache write failed: %s", exc)
        if plan is None:
            logger.info(
                "Bilingual preprocess produced no usable translation; "
                "falling back to single-path"
            )
        return plan
    except Exception as exc:  # noqa: BLE001 — fail-open by contract
        logger.warning(
            "Bilingual preprocess failed, falling back to single-path: %s", exc
        )
        return None


# ---------------------------------------------------------------------------
# Dual-path retrieval
# ---------------------------------------------------------------------------


def _param_with_keywords(
    param: QueryParam, hl_keywords: list[str], ll_keywords: list[str]
) -> QueryParam:
    clone = copy.copy(param)
    clone.hl_keywords = list(hl_keywords)
    clone.ll_keywords = list(ll_keywords)
    return clone


def apply_plan_keywords_to_param(param: QueryParam, plan: BilingualQueryPlan) -> None:
    """Seed the primary path's keywords so core keyword extraction is skipped.

    Only applied when the plan produced keywords; otherwise the core keeps
    its own (cached) extraction behaviour for the primary path.
    """
    if plan.hl_primary or plan.ll_primary:
        param.hl_keywords = list(plan.hl_primary)
        param.ll_keywords = list(plan.ll_primary)


def alt_retrieval_param(
    param: QueryParam,
    query_alt: str,
    hl_keywords_alt: list[str] | None,
    ll_keywords_alt: list[str] | None,
) -> QueryParam:
    """Build the secondary-path QueryParam for an explicit alt query.

    Used by agent step execution where the planner LLM (not the
    preprocessing call) supplies the alternate-language query/keywords.
    """
    hl, ll = list(hl_keywords_alt or []), list(ll_keywords_alt or [])
    # Never let the secondary path trigger a hidden core keyword-extraction
    # LLM call: seed the alt query itself as a low-level keyword.
    if param.mode in _KG_MODES and not hl and not ll:
        ll = [query_alt]
    return _param_with_keywords(param, hl, ll)


def usable_alt_query(query: str, query_alt: Any) -> str | None:
    """Normalize an alternate-language query; None when it adds nothing."""
    if not isinstance(query_alt, str):
        return None
    normalized = query_alt.strip()
    if len(normalized) < 2 or normalized == (query or "").strip():
        return None
    return normalized


def _secondary_param(param: QueryParam, plan: BilingualQueryPlan) -> QueryParam:
    return alt_retrieval_param(
        param, plan.secondary_query, plan.hl_secondary, plan.ll_secondary
    )


def _chunk_key(chunk: dict[str, Any]) -> tuple[Any, ...]:
    chunk_id = chunk.get("chunk_id")
    if chunk_id:
        return ("id", chunk_id)
    return (
        "content",
        chunk.get("file_path"),
        hashlib.sha256(str(chunk.get("content", "")).encode("utf-8")).hexdigest(),
    )


def merge_chunk_lists(
    primary: Iterable[dict[str, Any]], secondary: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Primary-first merge with dedup; tags each chunk's retrieval path."""
    seen: set[tuple[Any, ...]] = set()
    merged: list[dict[str, Any]] = []
    for path, chunks in (("primary", primary), ("secondary", secondary)):
        for chunk in chunks or []:
            key = _chunk_key(chunk)
            if key in seen:
                continue
            seen.add(key)
            tagged = dict(chunk)
            tagged.setdefault("retrieval_path", path)
            merged.append(tagged)
    return merged


def _data_section(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    section = result.get("data")
    return section if isinstance(section, dict) else {}


def _merge_by_key(
    primary: Iterable[dict[str, Any]],
    secondary: Iterable[dict[str, Any]],
    key_func,
) -> list[dict[str, Any]]:
    seen: set[Any] = set()
    merged: list[dict[str, Any]] = []
    for items in (primary, secondary):
        for item in items or []:
            key = key_func(item)
            if key in seen:
                continue
            seen.add(key)
            copied = dict(item)
            # reference_id numbering is per-path and cannot survive a merge;
            # chunk-derived references stay authoritative for citations.
            copied["reference_id"] = None
            merged.append(copied)
    return merged


@dataclass(slots=True)
class DualRetrievalResult:
    plan: BilingualQueryPlan
    primary: dict[str, Any]
    secondary: dict[str, Any] | None
    secondary_failed: bool

    @property
    def primary_chunks(self) -> list[dict[str, Any]]:
        return list(_data_section(self.primary).get("chunks") or [])

    @property
    def secondary_chunks(self) -> list[dict[str, Any]]:
        return list(_data_section(self.secondary).get("chunks") or [])

    def merged_chunks(self) -> list[dict[str, Any]]:
        return merge_chunk_lists(self.primary_chunks, self.secondary_chunks)

    def merged_entities(self) -> list[dict[str, Any]]:
        return _merge_by_key(
            _data_section(self.primary).get("entities") or [],
            _data_section(self.secondary).get("entities") or [],
            lambda item: str(item.get("entity_name", "")),
        )

    def merged_relationships(self) -> list[dict[str, Any]]:
        return _merge_by_key(
            _data_section(self.primary).get("relationships") or [],
            _data_section(self.secondary).get("relationships") or [],
            lambda item: (str(item.get("src_id", "")), str(item.get("tgt_id", ""))),
        )

    def info(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "enabled": True,
            "source_language": self.plan.source_language,
            "translated_query": self.plan.secondary_query,
            "translation_cached": self.plan.from_cache,
            "primary_chunks": len(self.primary_chunks),
            "secondary_chunks": len(self.secondary_chunks),
            "merged_chunks": len(self.merged_chunks()),
        }
        if self.secondary_failed:
            payload["secondary_failed"] = True
        return payload


def bilingual_audit_fields(info: dict[str, Any] | None) -> dict[str, Any]:
    """Audit-safe projection: hashes instead of translated query text."""
    if not info:
        return {"bilingual_enabled": False}
    fields: dict[str, Any] = {"bilingual_enabled": bool(info.get("enabled"))}
    translated = info.get("translated_query")
    if translated:
        fields["bilingual_translated_query_hash"] = hashlib.sha256(
            str(translated).encode("utf-8")
        ).hexdigest()
    for key in ("primary_chunks", "secondary_chunks", "secondary_failed"):
        if key in info:
            fields[f"bilingual_{key}"] = info[key]
    return fields


async def dual_aquery_data(
    rag: Any,
    query: str,
    param: QueryParam,
    plan: BilingualQueryPlan,
) -> DualRetrievalResult:
    """Run both retrieval paths concurrently.

    The primary path failing fails the query exactly like today; a secondary
    failure is tolerated and only recorded.
    """
    primary_param = _param_with_keywords(param, plan.hl_primary, plan.ll_primary) \
        if (plan.hl_primary or plan.ll_primary) else copy.copy(param)
    secondary_param = _secondary_param(param, plan)
    primary_result, secondary_result = await asyncio.gather(
        rag.aquery_data(query, param=primary_param),
        rag.aquery_data(plan.secondary_query, param=secondary_param),
        return_exceptions=True,
    )
    if isinstance(primary_result, BaseException):
        raise primary_result
    if isinstance(secondary_result, BaseException):
        logger.warning(
            "Bilingual secondary retrieval failed (%s path kept): %s",
            plan.source_language,
            secondary_result,
        )
        return DualRetrievalResult(
            plan=plan, primary=primary_result, secondary=None, secondary_failed=True
        )
    return DualRetrievalResult(
        plan=plan,
        primary=primary_result,
        secondary=secondary_result,
        secondary_failed=False,
    )


# ---------------------------------------------------------------------------
# Merged results: /query/data shape and /query (LLM synthesis) shape
# ---------------------------------------------------------------------------


async def _process_merged_chunks(
    rag: Any,
    query: str,
    param: QueryParam,
    merged_chunks: list[dict[str, Any]],
    *,
    chunk_token_limit: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool, dict[str, Any]]:
    """Rerank/truncate the merged pool against the ORIGINAL query and rebuild
    reference ids. Returns (reference_list, chunks_with_ids, reranked, gc)."""
    global_config = rag._build_global_config()
    stripped = []
    for chunk in merged_chunks:
        copied = dict(chunk)
        copied.pop("reference_id", None)
        stripped.append(copied)
    processed = await process_chunks_unified(
        query=query,
        unique_chunks=stripped,
        query_param=param,
        global_config=global_config,
        source_type="bilingual",
        chunk_token_limit=chunk_token_limit,
    )
    reranked = bool(global_config.get("rerank_model_func")) and bool(
        param.enable_rerank
    )
    reference_list, processed_with_ids = generate_reference_list_from_chunks(processed)
    return reference_list, processed_with_ids, reranked, global_config


async def bilingual_query_data(
    rag: Any,
    query: str,
    param: QueryParam,
    plan: BilingualQueryPlan,
) -> dict[str, Any]:
    """Dual-path variant of ``aquery_data``: same response envelope, merged."""
    dual = await dual_aquery_data(rag, query, param, plan)
    merged_chunks = dual.merged_chunks()
    reference_list, processed_with_ids, reranked, _ = await _process_merged_chunks(
        rag, query, param, merged_chunks, chunk_token_limit=None
    )
    info = dual.info()
    info["final_chunks"] = len(processed_with_ids)
    info["reranked"] = reranked
    primary_metadata = (
        dual.primary.get("metadata") if isinstance(dual.primary, dict) else {}
    )
    metadata = dict(primary_metadata or {})
    metadata["bilingual"] = info
    return {
        "status": "success",
        "message": "ok",
        "data": {
            "entities": dual.merged_entities(),
            "relationships": dual.merged_relationships(),
            "chunks": processed_with_ids,
            "references": reference_list,
        },
        "metadata": metadata,
    }


def answer_language_rules(source_language: str) -> str:
    if source_language == "zh":
        return (
            "回答语言要求：无论证据片段是中文还是英文，必须全程使用中文撰写回答；"
            "引用英文证据时，关键技术术语、牌号或代号首次出现可在括号中标注英文原文。"
        )
    return (
        "Answer-language requirement: write the entire answer in English "
        "regardless of the evidence language; when citing non-English "
        "evidence you may add the original term in parentheses on first "
        "mention."
    )


async def bilingual_query_llm(
    rag: Any,
    query: str,
    param: QueryParam,
    plan: BilingualQueryPlan,
    *,
    stream: bool,
    sensitive_context: SensitiveContext | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Dual-path variant of ``aquery_llm``.

    Returns ``(result, bilingual_info)`` where ``result`` matches the
    ``aquery_llm`` envelope (``llm_response`` + ``data``) so route handlers
    keep their existing response assembly.
    """
    dual = await dual_aquery_data(rag, query, param, plan)
    merged_chunks = dual.merged_chunks()
    info = dual.info()

    if not merged_chunks:
        info["final_chunks"] = 0
        mark_sensitive_context_not_used(sensitive_context, "no_kb_evidence")
        return (
            {
                "llm_response": {"content": "", "is_streaming": False},
                "data": {"references": [], "chunks": []},
            },
            info,
        )

    global_config = rag._build_global_config()
    tokenizer = global_config.get("tokenizer")
    response_type = param.response_type or "Multiple Paragraphs"
    language_rules = answer_language_rules(plan.source_language)
    user_prompt_text = (
        f"{language_rules}\n\n{param.user_prompt}" if param.user_prompt else language_rules
    )
    max_total_tokens = (
        getattr(param, "max_total_tokens", None)
        or global_config.get("max_total_tokens")
        or 30000
    )
    chunk_token_limit: int | None = None
    if tokenizer:
        pre_sys = PROMPTS["naive_rag_response"].format(
            response_type=response_type,
            user_prompt=user_prompt_text,
            content_data="",
        )
        chunk_token_limit = max_total_tokens - (
            len(tokenizer.encode(pre_sys)) + len(tokenizer.encode(query)) + 200
        )

    reference_list, processed_with_ids, reranked, _ = await _process_merged_chunks(
        rag, query, param, merged_chunks, chunk_token_limit=chunk_token_limit
    )
    info["final_chunks"] = len(processed_with_ids)
    info["reranked"] = reranked

    if sensitive_context is not None and not processed_with_ids:
        mark_sensitive_context_not_used(sensitive_context, "no_kb_evidence")
        return (
            {
                "llm_response": {"content": "", "is_streaming": False},
                "data": {
                    "references": reference_list,
                    "chunks": processed_with_ids,
                },
            },
            info,
        )

    chunks_context = [
        {"reference_id": chunk["reference_id"], "content": chunk["content"]}
        for chunk in processed_with_ids
        if chunk.get("reference_id")
    ]
    text_units_str = "\n".join(
        json.dumps(unit, ensure_ascii=False) for unit in chunks_context
    )
    reference_list_str = "\n".join(
        f"[{ref['reference_id']}] {ref['file_path']}"
        for ref in reference_list
        if ref["reference_id"]
    )
    content_data = PROMPTS["naive_query_context"].format(
        text_chunks_str=text_units_str, reference_list_str=reference_list_str
    )
    sys_prompt = PROMPTS["naive_rag_response"].format(
        response_type=response_type,
        user_prompt=user_prompt_text,
        content_data=content_data,
    )
    use_model_func = global_config["role_llm_funcs"]["query"]

    if sensitive_context is not None:
        # Resolve against the exact query-role runtime that will perform final
        # synthesis, after authoritative bilingual evidence is fully merged.
        final_global_config = rag._build_global_config()
        final_identity = get_llm_cache_identity(final_global_config, "query")
        bind_sensitive_context_endpoint(
            sensitive_context,
            final_identity.get("host")
            if isinstance(final_identity, dict)
            else None,
        )

        def build_system_prompt(
            payload: SensitiveContextPayload | None,
        ) -> str:
            effective_user_prompt = user_prompt_text
            effective_content_data = content_data
            if payload is not None:
                # Trusted policy is the last server-controlled Additional
                # Instruction; only untrusted JSONL data enters Context.
                effective_user_prompt = (
                    f"{effective_user_prompt}\n\n{payload.trusted_policy}"
                )
                effective_content_data = (
                    f"{effective_content_data}\n\n{payload.context_data}"
                )
            return PROMPTS["naive_rag_response"].format(
                response_type=response_type,
                user_prompt=effective_user_prompt,
                content_data=effective_content_data,
            )

        def build_final_request(
            payload: SensitiveContextPayload | None,
        ) -> str:
            return serialize_sensitive_final_request(
                build_system_prompt(payload),
                query,
                param.conversation_history,
            )

        payload = await sensitive_context.resolve_for_final_request(
            final_global_config.get("tokenizer"),
            param.max_total_tokens,
            build_final_request,
        )
        sys_prompt = build_system_prompt(payload)
        use_model_func = final_global_config["role_llm_funcs"]["query"]
        llm_out = await use_model_func(
            query,
            system_prompt=sys_prompt,
            history_messages=param.conversation_history,
            enable_cot=True,
            stream=stream,
            _sensitive=True,
        )
    else:
        llm_out = await use_model_func(
            query,
            system_prompt=sys_prompt,
            history_messages=param.conversation_history,
            enable_cot=True,
            stream=stream,
        )

    llm_response: dict[str, Any]
    if hasattr(llm_out, "__aiter__"):
        llm_response = {
            "content": "",
            "is_streaming": True,
            "response_iterator": llm_out,
        }
    else:
        llm_response = {
            "content": str(llm_out).strip(),
            "is_streaming": False,
        }
    return (
        {
            "llm_response": llm_response,
            "data": {
                "references": reference_list,
                "chunks": processed_with_ids,
            },
        },
        info,
    )
