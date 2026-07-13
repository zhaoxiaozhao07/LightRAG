"""Staged evidence-chain workflow behind ``workflow="staged"``.

Implements the recommendation-oriented Agent pipeline described in
``docs/AgentStagedRecommendation-zh.md``: a fixed, server-driven stage
sequence (requirement -> skeleton -> factor evidence -> validation ->
gap repair -> synthesis) whose evidence chain is anchored in the knowledge
bases instead of the model's parametric memory.

Design invariants:

- The orchestration LLM only fills small schema-validated JSON decisions;
  the loop, retrieval budget and KB authorization stay on the server.
- Every retrieved chunk gets a stable ``A{n}`` reference id at collection
  time; extraction/verdict calls must cite those ids and citations are
  validated server-side (uncited facts are dropped, unreferenced verdicts
  are downgraded to ``no_data``).
- Evidence sufficiency is checked mechanically (every target property must
  have a verdict), never by model self-assessment.
- Bounded work only: capped steps per stage, one gap-repair round, a global
  retrieval ceiling; anything skipped for budget reasons is reported, never
  silently truncated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Literal

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from lightrag.api.agent_query_service import (
    AGENT_ALLOWED_MODES,
    BILINGUAL_PLAN_PROMPT_SUFFIX,
    AgentPlanStep,
    AgentRunResult,
    agent_bilingual_enabled,
    agent_kb_profile,
)
from lightrag.api.bilingual_query_service import contains_cjk
from lightrag.api.enterprise_auth import (
    agent_staged_max_kbs_per_step,
    agent_staged_max_retrievals,
    append_enterprise_audit_event,
)
from lightrag.api.kb_service import KnowledgeBaseRecord
from lightrag.api.llm_json_utils import LLMJsonError, call_llm_json
from lightrag.api.query_tool_service import QueryToolResult
from lightrag.constants import DEFAULT_QUERY_PRIORITY
from lightrag.utils import logger, truncate_list_by_token_size

if TYPE_CHECKING:
    from lightrag.api.agent_query_service import AgentQueryRequest, AgentQueryService

STAGE_REQUIREMENT = "requirement"
STAGE_SKELETON = "skeleton"
STAGE_FACTOR = "factor_evidence"
STAGE_VALIDATION = "validation"
STAGE_REPAIR = "gap_repair"

KB_EVIDENCE_ROLES = (
    "reference_formula",
    "experimental",
    "literature",
    "application_spec",
    "other",
)

PRIORITY_RANK = {"P0": 0, "P1": 1, "P2": 2}
PRIORITY_ALIASES = {
    "0": "P0",
    "critical": "P0",
    "high": "P0",
    "highest": "P0",
    "important": "P0",
    "must": "P0",
    "required": "P0",
    "urgent": "P0",
    "关键": "P0",
    "高": "P0",
    "高优先级": "P0",
    "重要": "P0",
    "必要": "P0",
    "1": "P1",
    "default": "P1",
    "medium": "P1",
    "normal": "P1",
    "standard": "P1",
    "一般": "P1",
    "中": "P1",
    "普通": "P1",
    "2": "P2",
    "low": "P2",
    "optional": "P2",
    "低": "P2",
    "低优先级": "P2",
    "可选": "P2",
}

STAGED_LLM_ATTEMPTS = 3
SKELETON_MAX_STEPS = 3
FACTOR_MAX_STEPS = 8
VALIDATION_MAX_STEPS = 8
REPAIR_MAX_STEPS = 4
MAX_TARGET_PROPERTIES = 8
MAX_SKELETON_COMPONENTS = 12
MAX_OPEN_QUESTIONS = 8
# Payload bounds for extraction/verdict prompts (local models, small JSON).
_EXTRACT_MAX_CHUNKS = 24
_EXTRACT_CHUNK_CHARS = 700
_VERDICT_CHUNKS_PER_PROPERTY = 3
_VERDICT_OTHER_EVIDENCE_CHUNKS = 8

REQUIREMENT_SYSTEM_PROMPT = """
你是配比/配方推荐 Agent 的需求解析器。把用户问题解析为结构化需求 JSON：
- application：应用对象（如某类制品、部件或产品）
- conditions：环境与工况条件列表
- target_properties：3~8 个目标性能指标，每项含 name/why/priority；
  P0 表示决定方案可用性的硬指标；指标名使用领域通用术语，便于检索。
- constraints：其他约束（成本、工艺、原料范围等）
若用户问题缺少无法从上下文推断的关键信息（如应用对象或环境），设置
clarification_required=true 并给出 clarification_question，此时 target_properties 可为空。
只输出严格 JSON，不要 markdown，不要思维链。
""".strip()

SKELETON_PLAN_SYSTEM_PROMPT = """
你是配比推荐 Agent 的检索规划器。基于结构化需求与 allowed_kbs：
1) 在 kb_roles 中为每个 kb_id 标注证据角色：reference_formula（参考配方/配比案例）、
   experimental（实验与测试数据）、literature（文献与机理）、application_spec（应用规范/规格约束）、other。
2) 规划最多 max_steps 个检索步骤，目标是找到与应用对象和环境条件最接近的参考配方或配比案例，
   优先使用 reference_formula 与 application_spec 角色的知识库。
每步包含完整自洽的 query、kb_ids（仅限 allowed_kbs 中的 kb_id，且每步不超过 max_kbs_per_step 个，
只选与该步最相关的知识库）、mode（local/global/hybrid/naive/mix 之一，禁止 bypass）。
只输出严格 JSON，不要 markdown，不要思维链。
""".strip()

SKELETON_EXTRACT_SYSTEM_PROMPT = """
你是配比推荐 Agent 的骨架提取器。仅根据给定证据片段提取基础配比骨架：
- components：组分列表，每项含 material/ratio/function/source_refs；
  ratio 保留证据中的原始数值与单位；source_refs 必须引用证据片段的 reference_id（如 "A3"）。
  禁止编造证据中不存在的组分、数值或引用编号。
- open_questions：为确定最终配比还需检索证据的关键问题；每条必须完整自洽、可直接用于检索
 （包含应用对象与环境条件），不要使用“上述”“该配方”这类指代。
证据不足时 components 可为空，不要臆造。只输出严格 JSON，不要 markdown，不要思维链。
""".strip()

VERDICT_SYSTEM_PROMPT = """
你是配比推荐 Agent 的证据裁决器。对 target_properties 中每个性能指标，仅基于给定证据片段判定：
- supported：有直接实验/测试证据支持推荐方案满足该指标
- partial：只有间接或部分证据
- unsupported：证据显示不满足该指标
- no_data：没有相关证据
每项输出 property/verdict/evidence_refs/note；evidence_refs 必须引用证据片段的 reference_id，
no_data 时 evidence_refs 为空。宁可判 no_data，不要编造。
只输出严格 JSON，不要 markdown，不要思维链。
""".strip()

REPAIR_PLAN_SYSTEM_PROMPT = """
你是配比推荐 Agent 的补查规划器。给定证据缺口（缺少证据或证据不利的性能指标、检索结果为空的步骤），
规划最多 max_steps 个补充检索步骤：可以换知识库、换检索模式或改写查询语句。
每步包含完整自洽的 query、kb_ids（仅限 allowed_kbs 中的 kb_id，且每步不超过 max_kbs_per_step 个）、
mode（local/global/hybrid/naive/mix 之一）。没有值得补查的内容时 steps 输出空数组。
只输出严格 JSON，不要 markdown，不要思维链。
""".strip()

SYNTHESIS_EXTRA_RULES = """
本次回答是配比/配方推荐，输出必须包含以下结构：
1) 推荐配比表：每个组分一行，含组分、推荐配比（保留证据中的数值与单位）、作用、依据引用编号；
   配比数值只能来自证据（参考配方或实验数据），禁止凭空给出数值。
2) 目标性能指标核对：逐项列出指标、证据结论（支持/部分支持/不支持/无数据）与引用编号。
3) 未覆盖点与风险：明确列出无数据或证据不利的指标、未完成的检索，以及采纳建议前需补充的验证实验。
""".strip()

BILINGUAL_REQUIREMENT_PROMPT_SUFFIX = """
本次启用双语检索（payload 中 bilingual_retrieval=true）：为 target_properties 中每一项额外给出
name_alt（该指标名的另一语言版本：中文指标给英文，英文指标给中文；使用领域通用译法，
型号/代号/标准号原样保留）。
""".strip()

BILINGUAL_SKELETON_EXTRACT_PROMPT_SUFFIX = """
本次启用双语检索（payload 中 bilingual_retrieval=true）：额外输出 open_questions_alt 数组，
与 open_questions 按顺序一一对应，给出每条补充检索问题的另一语言完整版本
（中文问题给英文，英文问题给中文）；无法翻译的条目用空字符串占位。
""".strip()


def _clip_str(value: Any, limit: int) -> str:
    return str(value if value is not None else "").strip()[:limit]


def _clip_str_list(value: Any, *, max_items: int, max_chars: int) -> list[str]:
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
        text = str(entry).strip()[:max_chars]
        if text and text not in items:
            items.append(text)
        if len(items) >= max_items:
            break
    return items


def _json_schema_response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}


def _string_array_schema(
    *,
    min_items: int = 0,
    max_items: int | None = None,
    enum: list[str] | None = None,
) -> dict[str, Any]:
    item_schema: dict[str, Any] = {"type": "string"}
    if enum is not None:
        item_schema["enum"] = enum
    schema: dict[str, Any] = {
        "type": "array",
        "minItems": min_items,
        "items": item_schema,
    }
    if max_items is not None:
        schema["maxItems"] = max_items
    return schema


def _staged_step_schema(
    *, allowed_kb_ids: list[str], max_kbs_per_step: int, bilingual: bool
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "step_index": {"type": "integer", "minimum": 1},
        "title": {"type": "string"},
        "query": {"type": "string", "minLength": 3},
        "kb_ids": _string_array_schema(
            min_items=1, max_items=max_kbs_per_step, enum=allowed_kb_ids
        ),
        "mode": {"type": "string", "enum": sorted(AGENT_ALLOWED_MODES)},
        "priority": {"type": "string", "enum": list(PRIORITY_RANK)},
        "hl_keywords": _string_array_schema(max_items=20),
        "ll_keywords": _string_array_schema(max_items=20),
    }
    if bilingual:
        properties.update(
            {
                "query_alt": {"type": "string"},
                "hl_keywords_alt": _string_array_schema(max_items=20),
                "ll_keywords_alt": _string_array_schema(max_items=20),
            }
        )
    return {
        "type": "object",
        "properties": properties,
        "required": ["step_index", "query", "kb_ids", "mode", "priority"],
    }


def _requirement_response_format(*, bilingual: bool) -> dict[str, Any]:
    property_fields: dict[str, Any] = {
        "name": {"type": "string", "minLength": 1},
        "why": {"type": "string"},
        "priority": {"type": "string", "enum": list(PRIORITY_RANK)},
    }
    if bilingual:
        property_fields["name_alt"] = {"type": "string"}
    schema = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["requirement"]},
            "clarification_required": {"type": "boolean"},
            "clarification_question": {"type": ["string", "null"]},
            "application": {"type": "string"},
            "conditions": _string_array_schema(max_items=10),
            "target_properties": {
                "type": "array",
                "maxItems": MAX_TARGET_PROPERTIES,
                "items": {
                    "type": "object",
                    "properties": property_fields,
                    "required": ["name", "priority"],
                },
            },
            "constraints": _string_array_schema(max_items=10),
        },
        "required": ["type", "clarification_required", "target_properties"],
    }
    return _json_schema_response_format("agent_staged_requirement", schema)


def _skeleton_plan_response_format(
    *, allowed_kb_ids: list[str], max_kbs_per_step: int, bilingual: bool
) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["skeleton_plan"]},
            "kb_roles": {
                "type": "object",
                "additionalProperties": {
                    "type": "string",
                    "enum": list(KB_EVIDENCE_ROLES),
                },
            },
            "steps": {
                "type": "array",
                "minItems": 1,
                "maxItems": SKELETON_MAX_STEPS,
                "items": _staged_step_schema(
                    allowed_kb_ids=allowed_kb_ids,
                    max_kbs_per_step=max_kbs_per_step,
                    bilingual=bilingual,
                ),
            },
        },
        "required": ["type", "steps"],
    }
    return _json_schema_response_format("agent_staged_skeleton_plan", schema)


def _skeleton_extract_response_format(
    *, reference_ids: list[str], bilingual: bool
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "type": {"type": "string", "enum": ["skeleton"]},
        "components": {
            "type": "array",
            "maxItems": MAX_SKELETON_COMPONENTS,
            "items": {
                "type": "object",
                "properties": {
                    "material": {"type": "string", "minLength": 1},
                    "ratio": {"type": "string"},
                    "function": {"type": "string"},
                    "source_refs": _string_array_schema(
                        min_items=1, max_items=6, enum=reference_ids
                    ),
                },
                "required": ["material", "source_refs"],
            },
        },
        "open_questions": _string_array_schema(max_items=MAX_OPEN_QUESTIONS),
        "rationale": {"type": "string"},
    }
    if bilingual:
        properties["open_questions_alt"] = _string_array_schema(
            max_items=MAX_OPEN_QUESTIONS
        )
    schema = {
        "type": "object",
        "properties": properties,
        "required": ["type", "components", "open_questions"],
    }
    return _json_schema_response_format("agent_staged_skeleton_extract", schema)


def _verdicts_response_format(
    *, properties: list[TargetProperty], reference_ids: list[str]
) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["verdicts"]},
            "verdicts": {
                "type": "array",
                "maxItems": len(properties),
                "items": {
                    "type": "object",
                    "properties": {
                        "property": {
                            "type": "string",
                            "enum": [prop.name for prop in properties],
                        },
                        "verdict": {
                            "type": "string",
                            "enum": [
                                "supported",
                                "partial",
                                "unsupported",
                                "no_data",
                            ],
                        },
                        "evidence_refs": _string_array_schema(
                            max_items=6, enum=reference_ids
                        ),
                        "note": {"type": "string"},
                    },
                    "required": ["property", "verdict", "evidence_refs"],
                },
            },
        },
        "required": ["type", "verdicts"],
    }
    return _json_schema_response_format("agent_staged_verdicts", schema)


def _repair_plan_response_format(
    *, allowed_kb_ids: list[str], max_kbs_per_step: int, bilingual: bool
) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["repair_plan"]},
            "steps": {
                "type": "array",
                "maxItems": REPAIR_MAX_STEPS,
                "items": _staged_step_schema(
                    allowed_kb_ids=allowed_kb_ids,
                    max_kbs_per_step=max_kbs_per_step,
                    bilingual=bilingual,
                ),
            },
        },
        "required": ["type", "steps"],
    }
    return _json_schema_response_format("agent_staged_repair_plan", schema)


class TargetProperty(BaseModel):
    name: str = Field(min_length=1)
    why: str = ""
    priority: Literal["P0", "P1", "P2"] = "P1"
    # Alternate-language property name, filled only when bilingual retrieval
    # is on; drives the validation step's secondary retrieval path.
    name_alt: str = ""

    @field_validator("name", "name_alt", mode="before")
    @classmethod
    def _clip_name(cls, value: Any) -> str:
        return _clip_str(value, 120)

    @field_validator("why", mode="before")
    @classmethod
    def _clip_why(cls, value: Any) -> str:
        return _clip_str(value, 300)

    @field_validator("priority", mode="before")
    @classmethod
    def _coerce_priority(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return "P1"
        upper = text.upper()
        if upper in PRIORITY_RANK:
            return upper
        normalized = text.lower().replace("_", "-").strip()
        return PRIORITY_ALIASES.get(normalized, "P1")


class StagedRequirement(BaseModel):
    type: Literal["requirement"] = "requirement"
    clarification_required: bool = False
    clarification_question: str | None = None
    application: str = ""
    conditions: list[str] = Field(default_factory=list)
    target_properties: list[TargetProperty] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)

    @field_validator("application", mode="before")
    @classmethod
    def _clip_application(cls, value: Any) -> str:
        return _clip_str(value, 300)

    @field_validator("conditions", "constraints", mode="before")
    @classmethod
    def _clip_lists(cls, value: Any) -> list[str]:
        return _clip_str_list(value, max_items=10, max_chars=200)

    def limited_properties(self) -> list[TargetProperty]:
        """P0-first, de-duplicated, capped property checklist.

        The cap protects the retrieval budget; P0-first ordering guarantees
        hard requirements survive both the cap and any budget clipping."""
        ordered = sorted(
            enumerate(self.target_properties),
            key=lambda pair: (PRIORITY_RANK.get(pair[1].priority, 1), pair[0]),
        )
        seen: set[str] = set()
        result: list[TargetProperty] = []
        for _, prop in ordered:
            key = prop.name.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(prop)
            if len(result) >= MAX_TARGET_PROPERTIES:
                break
        return result


class SkeletonPlan(BaseModel):
    type: Literal["skeleton_plan"] = "skeleton_plan"
    kb_roles: dict[str, str] = Field(default_factory=dict)
    steps: list[AgentPlanStep] = Field(default_factory=list)

    @field_validator("kb_roles", mode="before")
    @classmethod
    def _coerce_roles(cls, value: Any) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        return {str(key): str(role).strip().lower() for key, role in value.items()}


class SkeletonComponent(BaseModel):
    material: str = Field(min_length=1)
    ratio: str = ""
    function: str = ""
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("material", mode="before")
    @classmethod
    def _clip_material(cls, value: Any) -> str:
        return _clip_str(value, 200)

    @field_validator("ratio", mode="before")
    @classmethod
    def _clip_ratio(cls, value: Any) -> str:
        return _clip_str(value, 100)

    @field_validator("function", mode="before")
    @classmethod
    def _clip_function(cls, value: Any) -> str:
        return _clip_str(value, 200)

    @field_validator("source_refs", mode="before")
    @classmethod
    def _clip_refs(cls, value: Any) -> list[str]:
        return _clip_str_list(value, max_items=6, max_chars=20)


class SkeletonExtract(BaseModel):
    type: Literal["skeleton"] = "skeleton"
    components: list[SkeletonComponent] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    # Positionally paired alternate-language variants of open_questions
    # (bilingual retrieval only); shorter lists simply leave the tail unpaired.
    open_questions_alt: list[str] = Field(default_factory=list)
    rationale: str = ""

    @field_validator("open_questions", mode="before")
    @classmethod
    def _clip_questions(cls, value: Any) -> list[str]:
        return _clip_str_list(value, max_items=MAX_OPEN_QUESTIONS, max_chars=500)

    @field_validator("open_questions_alt", mode="before")
    @classmethod
    def _clip_questions_alt(cls, value: Any) -> list[str]:
        # Positional pairing forbids the dedup/drop-empty normalization used
        # for open_questions; only clip length and item count here.
        if isinstance(value, str):
            raw: list[Any] = [value]
        elif isinstance(value, list):
            raw = value
        elif value is None:
            raw = []
        else:
            raw = [value]
        return [str(entry).strip()[:500] for entry in raw[:MAX_OPEN_QUESTIONS]]

    @field_validator("rationale", mode="before")
    @classmethod
    def _clip_rationale(cls, value: Any) -> str:
        return _clip_str(value, 500)


class PropertyVerdict(BaseModel):
    property: str = Field(min_length=1)
    verdict: Literal["supported", "partial", "unsupported", "no_data"] = "no_data"
    evidence_refs: list[str] = Field(default_factory=list)
    note: str = ""

    @field_validator("property", mode="before")
    @classmethod
    def _clip_property(cls, value: Any) -> str:
        return _clip_str(value, 120)

    @field_validator("verdict", mode="before")
    @classmethod
    def _coerce_verdict(cls, value: Any) -> str:
        # Fail closed: any unknown verdict value counts as "no evidence".
        text = str(value or "").strip().lower()
        return text if text in {"supported", "partial", "unsupported", "no_data"} else "no_data"

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def _clip_refs(cls, value: Any) -> list[str]:
        return _clip_str_list(value, max_items=6, max_chars=20)

    @field_validator("note", mode="before")
    @classmethod
    def _clip_note(cls, value: Any) -> str:
        return _clip_str(value, 300)


class ValidationVerdicts(BaseModel):
    type: Literal["verdicts"] = "verdicts"
    verdicts: list[PropertyVerdict] = Field(default_factory=list)


class RepairPlan(BaseModel):
    type: Literal["repair_plan"] = "repair_plan"
    steps: list[AgentPlanStep] = Field(default_factory=list)


@dataclass(slots=True)
class _EvidenceBoard:
    """Session-wide evidence store assigning stable ``A{n}`` ids on insert.

    Ids are handed out at collection time (not at synthesis time) so that
    extraction and verdict calls can cite evidence before the final context
    selection happens; the final reference list may therefore contain gaps
    in numbering — ids are identifiers, not positions."""

    chunks: list[dict[str, Any]] = field(default_factory=list)
    _seen: set[tuple[Any, ...]] = field(default_factory=set)
    _by_id: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(
        self,
        chunk: dict[str, Any],
        *,
        stage: str,
        evidence_role: str,
        round_index: int,
        mode: str,
    ) -> dict[str, Any] | None:
        kb_id = chunk.get("kb_id")
        chunk_id = chunk.get("chunk_id")
        if kb_id and chunk_id:
            key: tuple[Any, ...] = ("chunk", kb_id, chunk_id)
        else:
            key = (
                "content",
                kb_id,
                chunk.get("file_path") or chunk.get("source"),
                hashlib.sha256(str(chunk.get("content", "")).encode("utf-8")).hexdigest(),
            )
        if key in self._seen:
            return None
        self._seen.add(key)
        reference_id = f"A{len(self.chunks) + 1}"
        tagged = {
            **chunk,
            "reference_id": reference_id,
            "source_reference_id": chunk.get("reference_id"),
            "stage": stage,
            "evidence_role": evidence_role,
            "round_index": round_index,
            "step_index": round_index,
            "mode": mode,
        }
        self.chunks.append(tagged)
        self._by_id[reference_id] = tagged
        return tagged

    def ids(self) -> set[str]:
        return set(self._by_id)

    def for_stage(self, *stages: str) -> list[dict[str, Any]]:
        return [chunk for chunk in self.chunks if chunk["stage"] in stages]

    def for_round(self, round_index: int) -> list[dict[str, Any]]:
        return [chunk for chunk in self.chunks if chunk["round_index"] == round_index]


class AgentStagedRunner:
    """Drives one staged Agent session on behalf of ``AgentQueryService``.

    Reuses the parent service's gatekeeping outcome (effective KB set),
    retrieval executor (with empty-result retry) and synthesis path so plan
    and staged mode share authorization and generation semantics."""

    def __init__(self, service: "AgentQueryService"):
        self._service = service

    async def run_events(
        self,
        *,
        request: Request,
        body: "AgentQueryRequest",
        session_id: str,
        effective_records: list[KnowledgeBaseRecord],
        stream_synthesis: bool,
    ) -> AsyncIterator[dict[str, Any]]:
        allowed_ids = {record.id for record in effective_records}
        effective_kb_ids = [record.id for record in effective_records]
        agent_func, user_prompt = await self._service._agent_llm_context(
            request, body, effective_records
        )
        bilingual = agent_bilingual_enabled(body)
        kb_profiles = [agent_kb_profile(record) for record in effective_records]
        board = _EvidenceBoard()
        steps_summary: list[dict[str, Any]] = []
        clipped_notes: list[str] = []
        max_retrievals = agent_staged_max_retrievals()
        max_kbs_per_step = agent_staged_max_kbs_per_step()
        # agent_priority (manual profile field) breaks ties when a step must
        # be narrowed to a subset of an unknown-size KB fleet.
        priority_by_id = {
            profile["kb_id"]: profile.get("agent_priority") for profile in kb_profiles
        }
        # Mutable per-session counters shared with the step executor.
        state: dict[str, Any] = {"round": 0, "synth_result": None}

        # ---- Stage 0: requirement parsing --------------------------------
        yield {
            "event": "stage_started",
            "session_id": session_id,
            "stage": STAGE_REQUIREMENT,
        }
        requirement = await self._parse_requirement(
            agent_func=agent_func,
            body=body,
            kb_profiles=kb_profiles,
            user_prompt=user_prompt,
            bilingual=bilingual,
        )
        if requirement.clarification_required:
            result = AgentRunResult(
                status="clarification_required",
                session_id=session_id,
                clarification_question=requirement.clarification_question
                or "请补充关键约束。",
                metadata={
                    "workflow": "staged",
                    "effective_kb_ids": effective_kb_ids,
                },
            )
            yield {
                "event": "clarification_required",
                "session_id": session_id,
                "clarification_question": result.clarification_question,
            }
            yield {"event": "done", "session_id": session_id, "_result": result}
            return
        properties = requirement.limited_properties()
        requirement_payload = self._requirement_payload(requirement, properties)
        yield {
            "event": "requirement_parsed",
            "session_id": session_id,
            "requirement": requirement_payload,
        }

        # ---- Stage 1: skeleton recall + extraction ------------------------
        yield {
            "event": "stage_started",
            "session_id": session_id,
            "stage": STAGE_SKELETON,
        }
        plan = await self._plan_skeleton(
            agent_func=agent_func,
            requirement_payload=requirement_payload,
            kb_profiles=kb_profiles,
            user_prompt=user_prompt,
            max_kbs_per_step=max_kbs_per_step,
            bilingual=bilingual,
        )
        kb_roles = {
            kb_id: (role if role in KB_EVIDENCE_ROLES else "other")
            for kb_id, role in plan.kb_roles.items()
            if kb_id in allowed_ids
        }
        for record in effective_records:
            kb_roles.setdefault(record.id, "other")
        yield {
            "event": "kb_roles_assigned",
            "session_id": session_id,
            "kb_roles": kb_roles,
        }
        skeleton_steps = plan.steps[:SKELETON_MAX_STEPS]
        if len(plan.steps) > SKELETON_MAX_STEPS:
            clipped_notes.append(
                f"骨架检索步骤从 {len(plan.steps)} 裁剪到 {SKELETON_MAX_STEPS}"
            )
        if not skeleton_steps:
            raise HTTPException(
                status_code=400,
                detail="Agent skeleton plan contains no executable steps",
            )
        self._service._ensure_steps_allowed(skeleton_steps, allowed_ids)
        for step in skeleton_steps:
            if len(step.kb_ids) > max_kbs_per_step:
                clipped_notes.append(
                    f"步骤“{step.title or step.query[:40]}”的知识库从 "
                    f"{len(step.kb_ids)} 个裁剪到 {max_kbs_per_step} 个"
                )
                step.kb_ids = self._cap_kbs(
                    step.kb_ids, priority_by_id, max_kbs_per_step
                )
        for step in skeleton_steps:
            async for event in self._run_step(
                request=request,
                body=body,
                session_id=session_id,
                board=board,
                state=state,
                steps_summary=steps_summary,
                clipped_notes=clipped_notes,
                max_retrievals=max_retrievals,
                stage=STAGE_SKELETON,
                evidence_role="reference_formula",
                title=step.title or step.query[:80],
                query=step.query,
                kb_ids=step.kb_ids,
                mode=step.mode,
                hl_keywords=step.hl_keywords,
                ll_keywords=step.ll_keywords,
                query_alt=step.query_alt if bilingual else "",
                hl_keywords_alt=step.hl_keywords_alt,
                ll_keywords_alt=step.ll_keywords_alt,
                priority=step.priority,
            ):
                yield event

        skeleton, dropped_components = await self._extract_skeleton(
            agent_func=agent_func,
            requirement_payload=requirement_payload,
            board=board,
            bilingual=bilingual,
        )
        if skeleton is None or not skeleton.components:
            clipped_notes.append("未能从知识库证据中提取骨架配方")
        yield {
            "event": "skeleton_extracted",
            "session_id": session_id,
            "components": self._component_payload(skeleton),
            "open_questions": list(skeleton.open_questions) if skeleton else [],
            "dropped_components": dropped_components,
        }

        # ---- Stage 2: factor evidence (template-instantiated retrieval) ---
        # Reserve one retrieval slot per pending property so mechanism
        # lookups can never starve the P0 validation checklist.
        validation_reserve = min(len(properties), VALIDATION_MAX_STEPS)
        factor_allowance = min(
            FACTOR_MAX_STEPS,
            max(0, max_retrievals - state["round"] - validation_reserve),
        )
        factor_queries = self._factor_queries(
            requirement=requirement,
            skeleton=skeleton,
            allowance=factor_allowance,
            bilingual=bilingual,
        )
        factor_kbs = self._kbs_with_roles(
            effective_records,
            kb_roles,
            ("literature", "experimental", "application_spec"),
        )
        if len(factor_kbs) > max_kbs_per_step:
            factor_kbs = self._cap_kbs(factor_kbs, priority_by_id, max_kbs_per_step)
            clipped_notes.append(
                f"要素证据检索每步只使用优先级最高的 {max_kbs_per_step} 个知识库"
            )
        if factor_queries:
            yield {
                "event": "stage_started",
                "session_id": session_id,
                "stage": STAGE_FACTOR,
            }
            for factor_query, factor_query_alt in factor_queries:
                async for event in self._run_step(
                    request=request,
                    body=body,
                    session_id=session_id,
                    board=board,
                    state=state,
                    steps_summary=steps_summary,
                    clipped_notes=clipped_notes,
                    max_retrievals=max_retrievals,
                    stage=STAGE_FACTOR,
                    evidence_role="mechanism",
                    title=f"要素证据：{factor_query[:60]}",
                    query=factor_query,
                    kb_ids=factor_kbs,
                    mode="mix",
                    query_alt=factor_query_alt,
                    priority="P1",
                ):
                    yield event

        # ---- Stage 3: per-property validation ------------------------------
        yield {
            "event": "stage_started",
            "session_id": session_id,
            "stage": STAGE_VALIDATION,
        }
        validation_kbs = self._validation_kbs(effective_records, kb_roles)
        if len(validation_kbs) > max_kbs_per_step:
            validation_kbs = self._cap_kbs(
                validation_kbs, priority_by_id, max_kbs_per_step
            )
            clipped_notes.append(
                f"指标验证检索每步只使用优先级最高的 {max_kbs_per_step} 个知识库"
            )
        conditions_text = "、".join(requirement.conditions) or "目标环境"
        property_rounds: dict[str, int] = {}
        for prop in properties[:VALIDATION_MAX_STEPS]:
            if state["round"] >= max_retrievals:
                clipped_notes.append(
                    f"性能指标“{prop.name}”的验证检索因预算不足被跳过"
                )
                continue
            query = (
                f"{requirement.application}在{conditions_text}条件下的"
                f"{prop.name}实验数据与测试结果"
            )
            prop_query_alt = ""
            prop_hl_alt: list[str] = []
            if bilingual and prop.name_alt:
                prop_hl_alt = [prop.name_alt]
                if contains_cjk(prop.name):
                    prop_query_alt = (
                        f"{prop.name_alt} experimental data and test results "
                        f"for {requirement.application}"
                    )
                else:
                    prop_query_alt = (
                        f"{requirement.application}的{prop.name_alt}实验数据与测试结果"
                    )
            next_round = state["round"] + 1
            async for event in self._run_step(
                request=request,
                body=body,
                session_id=session_id,
                board=board,
                state=state,
                steps_summary=steps_summary,
                clipped_notes=clipped_notes,
                max_retrievals=max_retrievals,
                stage=STAGE_VALIDATION,
                evidence_role="validation",
                title=f"指标验证：{prop.name}",
                query=query,
                kb_ids=validation_kbs,
                mode="mix",
                hl_keywords=[prop.name],
                query_alt=prop_query_alt,
                hl_keywords_alt=prop_hl_alt,
                priority=prop.priority,
            ):
                yield event
            # Only map the property to its round when the step actually ran
            # (the executor skips silently-noted steps once the budget is gone).
            if state["round"] == next_round:
                property_rounds[prop.name] = next_round
        if len(properties) > VALIDATION_MAX_STEPS:
            for prop in properties[VALIDATION_MAX_STEPS:]:
                clipped_notes.append(
                    f"性能指标“{prop.name}”超出验证步数上限，未单独检索"
                )

        verdicts = await self._extract_verdicts(
            agent_func=agent_func,
            requirement_payload=requirement_payload,
            properties=properties,
            board=board,
            property_rounds=property_rounds,
        )
        yield {
            "event": "validation_verdicts",
            "session_id": session_id,
            "verdicts": verdicts,
            "after_repair": False,
        }

        # ---- Stage 4: bounded gap repair -----------------------------------
        gaps = [v for v in verdicts if v["verdict"] in ("no_data", "unsupported")]
        empty_rounds = [
            summary
            for summary in steps_summary
            if summary.get("status") == "ok" and not summary.get("chunk_count")
        ]
        if (gaps or empty_rounds) and state["round"] < max_retrievals:
            yield {
                "event": "stage_started",
                "session_id": session_id,
                "stage": STAGE_REPAIR,
            }
            repair_steps = await self._plan_repair(
                agent_func=agent_func,
                requirement_payload=requirement_payload,
                gaps=gaps,
                empty_rounds=empty_rounds,
                kb_profiles=kb_profiles,
                allowed_ids=allowed_ids,
                clipped_notes=clipped_notes,
                priority_by_id=priority_by_id,
                max_kbs_per_step=max_kbs_per_step,
                bilingual=bilingual,
            )
            rounds_before_repair = state["round"]
            for step in repair_steps:
                async for event in self._run_step(
                    request=request,
                    body=body,
                    session_id=session_id,
                    board=board,
                    state=state,
                    steps_summary=steps_summary,
                    clipped_notes=clipped_notes,
                    max_retrievals=max_retrievals,
                    stage=STAGE_REPAIR,
                    evidence_role="repair",
                    title=step.title or step.query[:80],
                    query=step.query,
                    kb_ids=step.kb_ids,
                    mode=step.mode,
                    hl_keywords=step.hl_keywords,
                    ll_keywords=step.ll_keywords,
                    query_alt=step.query_alt if bilingual else "",
                    hl_keywords_alt=step.hl_keywords_alt,
                    ll_keywords_alt=step.ll_keywords_alt,
                    priority=step.priority,
                ):
                    yield event
            repair_added = any(
                chunk["round_index"] > rounds_before_repair for chunk in board.chunks
            )
            if gaps and repair_added:
                gap_names = {v["property"] for v in gaps}
                gap_properties = [p for p in properties if p.name in gap_names]
                updated = await self._extract_verdicts(
                    agent_func=agent_func,
                    requirement_payload=requirement_payload,
                    properties=gap_properties,
                    board=board,
                    property_rounds=property_rounds,
                    include_repair=True,
                )
                updated_by_name = {v["property"]: v for v in updated}
                verdicts = [
                    updated_by_name.get(v["property"], v) for v in verdicts
                ]
                yield {
                    "event": "validation_verdicts",
                    "session_id": session_id,
                    "verdicts": verdicts,
                    "after_repair": True,
                }

        # ---- Synthesis ------------------------------------------------------
        failed_rounds = [s for s in steps_summary if s.get("status") == "failed"]
        synth_result: QueryToolResult | None = state["synth_result"]
        if synth_result is None:
            raise HTTPException(
                status_code=502,
                detail={
                    "error_code": "agent_all_steps_failed",
                    "message": "All Agent retrieval steps failed",
                    "steps_summary": steps_summary,
                },
            )
        cited_ids: set[str] = set()
        if skeleton is not None:
            for component in skeleton.components:
                cited_ids.update(component.source_refs)
        for verdict in verdicts:
            cited_ids.update(verdict["evidence_refs"])
        references, context_units = self._build_references(
            body=body, board=board, cited_ids=cited_ids, synth_result=synth_result
        )
        if references and body.include_references:
            yield {
                "event": "references",
                "session_id": session_id,
                "references": references,
            }

        answer_parts: list[str] = []
        if not context_units:
            answer = "未检索到可用于回答的证据。"
            yield {"event": "response", "session_id": session_id, "delta": answer}
        else:
            extra_rules = self._synthesis_rules(
                requirement_payload=requirement_payload,
                skeleton=skeleton,
                verdicts=verdicts,
                clipped_notes=clipped_notes,
            )
            async for delta in self._service._synthesize_answer(
                request=request,
                body=body,
                synth_result=synth_result,
                references=references,
                context_units=context_units,
                steps_summary=steps_summary,
                stream=stream_synthesis,
                extra_rules=extra_rules,
            ):
                if delta:
                    answer_parts.append(delta)
                    yield {
                        "event": "response",
                        "session_id": session_id,
                        "delta": delta,
                    }
            answer = "".join(answer_parts)

        verdict_counts: dict[str, int] = {}
        for verdict in verdicts:
            verdict_counts[verdict["verdict"]] = (
                verdict_counts.get(verdict["verdict"], 0) + 1
            )
        await append_enterprise_audit_event(
            request,
            "agent_query_completed",
            target_type="agent_session",
            target_id=session_id,
            metadata={
                "workflow": "staged",
                "round_count": len(steps_summary),
                "failed_round_count": len(failed_rounds),
                "reference_count": len(references),
                "effective_kb_ids": effective_kb_ids,
                "verdict_counts": verdict_counts,
            },
        )
        result = AgentRunResult(
            status="success",
            session_id=session_id,
            answer=answer,
            references=references if body.include_references else [],
            steps_summary=steps_summary,
            metadata={
                "workflow": "staged",
                "effective_kb_ids": effective_kb_ids,
                "kb_roles": kb_roles,
                "requirement": requirement_payload,
                "skeleton_component_count": len(skeleton.components)
                if skeleton
                else 0,
                "dropped_component_count": dropped_components,
                "property_verdicts": verdicts,
                "round_count": len(steps_summary),
                "failed_round_count": len(failed_rounds),
                "retrieval_budget": {"max": max_retrievals, "used": state["round"]},
                "clipped": clipped_notes,
                "bilingual_retrieval": bilingual,
            },
        )
        yield {"event": "done", "session_id": session_id, "_result": result}

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    async def _run_step(
        self,
        *,
        request: Request,
        body: "AgentQueryRequest",
        session_id: str,
        board: _EvidenceBoard,
        state: dict[str, Any],
        steps_summary: list[dict[str, Any]],
        clipped_notes: list[str],
        max_retrievals: int,
        stage: str,
        evidence_role: str,
        title: str,
        query: str,
        kb_ids: list[str],
        mode: str,
        hl_keywords: list[str] | None = None,
        ll_keywords: list[str] | None = None,
        query_alt: str = "",
        hl_keywords_alt: list[str] | None = None,
        ll_keywords_alt: list[str] | None = None,
        priority: str = "P1",
    ) -> AsyncIterator[dict[str, Any]]:
        if state["round"] >= max_retrievals:
            clipped_notes.append(
                f"检索预算（{max_retrievals}）用尽，跳过步骤：{title}"
            )
            return
        state["round"] += 1
        round_index = state["round"]
        yield {
            "event": "round_started",
            "session_id": session_id,
            "round": round_index,
            "stage": stage,
            "step_index": round_index,
            "title": title,
            "query": query,
            "kb_ids": kb_ids,
            "mode": mode,
            "priority": priority,
        }
        summary: dict[str, Any] = {
            "round": round_index,
            "stage": stage,
            "step_index": round_index,
            "title": title,
            "query": query,
            "kb_ids": kb_ids,
            "mode": mode,
            "priority": priority,
            "status": "ok",
            "chunk_count": 0,
            "per_kb_chunk_counts": {},
            "skipped_kbs": [],
        }
        step = AgentPlanStep(
            step_index=round_index,
            title=title,
            query=query,
            kb_ids=kb_ids,
            mode=mode,  # type: ignore[arg-type]
            priority=priority,  # type: ignore[arg-type]
            hl_keywords=hl_keywords or [],
            ll_keywords=ll_keywords or [],
            query_alt=query_alt or "",
            hl_keywords_alt=hl_keywords_alt or [],
            ll_keywords_alt=ll_keywords_alt or [],
        )
        try:
            tool_result, retried_mode = await self._service._retrieve_with_empty_retry(
                http_request=request, body=body, step=step
            )
        except Exception as exc:  # noqa: BLE001 — tolerate per-step failure
            if isinstance(exc, HTTPException):
                detail = exc.detail
                error_code = (
                    detail.get("error_code", "agent_step_failed")
                    if isinstance(detail, dict)
                    else "agent_step_failed"
                )
            else:
                error_code = "agent_step_failed"
                logger.error(
                    "Agent staged step %d failed: %s", round_index, exc, exc_info=True
                )
            summary["status"] = "failed"
            summary["error_code"] = error_code
            steps_summary.append(summary)
            await append_enterprise_audit_event(
                request,
                "agent_retrieve_round",
                target_type="agent_session",
                target_id=session_id,
                metadata={
                    "workflow": "staged",
                    "stage": stage,
                    "round": round_index,
                    "kb_ids": kb_ids,
                    "mode": mode,
                    "status": "failed",
                    "error_code": error_code,
                    "query_hash": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                },
            )
            yield {"event": "round_result", "session_id": session_id, **summary}
            return

        if state["synth_result"] is None:
            state["synth_result"] = tool_result
        used_mode = retried_mode or mode
        if retried_mode:
            summary["retried_mode"] = retried_mode
        if tool_result.alt_chunk_counts or tool_result.alt_failed_kbs:
            summary["bilingual"] = True
            summary["alt_chunk_counts"] = tool_result.alt_chunk_counts
            if tool_result.alt_failed_kbs:
                summary["alt_failed_kbs"] = tool_result.alt_failed_kbs
        new_chunks = 0
        for chunk in tool_result.chunks:
            if board.add(
                chunk,
                stage=stage,
                evidence_role=evidence_role,
                round_index=round_index,
                mode=used_mode,
            ):
                new_chunks += 1
        summary.update(
            {
                "kb_ids": tool_result.queried_kb_ids,
                "chunk_count": len(tool_result.chunks),
                "new_chunk_count": new_chunks,
                "per_kb_chunk_counts": tool_result.per_kb_chunk_counts,
                "skipped_kbs": tool_result.skipped_kbs,
            }
        )
        steps_summary.append(summary)
        audit_metadata = {
            "workflow": "staged",
            "stage": stage,
            "round": round_index,
            "kb_ids": kb_ids,
            "mode": mode,
            "status": "ok",
            "query_hash": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "chunk_count": len(tool_result.chunks),
            "skipped_kbs": tool_result.skipped_kbs,
        }
        if retried_mode:
            audit_metadata["retried_mode"] = retried_mode
        await append_enterprise_audit_event(
            request,
            "agent_retrieve_round",
            target_type="agent_session",
            target_id=session_id,
            metadata=audit_metadata,
        )
        yield {"event": "round_result", "session_id": session_id, **summary}

    # ------------------------------------------------------------------
    # LLM decision calls
    # ------------------------------------------------------------------

    async def _parse_requirement(
        self,
        *,
        agent_func: Any,
        body: "AgentQueryRequest",
        kb_profiles: list[dict[str, Any]],
        user_prompt: str,
        bilingual: bool = False,
    ) -> StagedRequirement:
        property_schema: dict[str, Any] = {
            "name": "性能指标",
            "why": "为何重要",
            "priority": "P0",
        }
        if bilingual:
            property_schema["name_alt"] = "指标名的另一语言版本"
        payload = {
            "user_question": body.query,
            "allowed_kbs": kb_profiles,
            "bilingual_retrieval": bilingual,
            "user_workflow_prompt": user_prompt,
            "output_schema": {
                "type": "requirement",
                "clarification_required": False,
                "clarification_question": None,
                "application": "应用对象",
                "conditions": ["环境/工况条件"],
                "target_properties": [property_schema],
                "constraints": ["其他约束"],
            },
        }
        system_prompt = REQUIREMENT_SYSTEM_PROMPT
        if bilingual:
            system_prompt = f"{system_prompt}\n{BILINGUAL_REQUIREMENT_PROMPT_SUFFIX}"

        def _parse(data: Any) -> StagedRequirement:
            requirement = StagedRequirement.model_validate(data)
            if (
                not requirement.clarification_required
                and not requirement.target_properties
            ):
                raise ValueError(
                    "requirement contains no target properties and no clarification"
                )
            return requirement

        try:
            return await call_llm_json(
                agent_func,
                json.dumps(payload, ensure_ascii=False),
                system_prompt=system_prompt,
                priority=DEFAULT_QUERY_PRIORITY,
                parse=_parse,
                attempts=STAGED_LLM_ATTEMPTS,
                label="agent_staged_requirement",
                response_format=_requirement_response_format(bilingual=bilingual),
            )
        except LLMJsonError as exc:
            raise HTTPException(
                status_code=502,
                detail={"error_code": "agent_requirement_invalid", "message": str(exc)},
            ) from exc

    async def _plan_skeleton(
        self,
        *,
        agent_func: Any,
        requirement_payload: dict[str, Any],
        kb_profiles: list[dict[str, Any]],
        user_prompt: str,
        max_kbs_per_step: int,
        bilingual: bool = False,
    ) -> SkeletonPlan:
        step_schema: dict[str, Any] = {
            "step_index": 1,
            "title": "短标题",
            "query": "完整自洽的检索子问题",
            "kb_ids": ["kb_xxx"],
            "mode": "mix",
            "priority": "P0",
            "hl_keywords": [],
            "ll_keywords": [],
        }
        if bilingual:
            step_schema.update(
                {
                    "query_alt": "该步子问题的另一语言完整版本",
                    "hl_keywords_alt": [],
                    "ll_keywords_alt": [],
                }
            )
        payload = {
            "requirement": requirement_payload,
            "allowed_kbs": kb_profiles,
            "max_steps": SKELETON_MAX_STEPS,
            "max_kbs_per_step": max_kbs_per_step,
            "kb_role_values": list(KB_EVIDENCE_ROLES),
            "bilingual_retrieval": bilingual,
            "user_workflow_prompt": user_prompt,
            "output_schema": {
                "type": "skeleton_plan",
                "kb_roles": {"kb_xxx": "reference_formula"},
                "steps": [step_schema],
            },
        }
        system_prompt = SKELETON_PLAN_SYSTEM_PROMPT
        if bilingual:
            system_prompt = f"{system_prompt}\n{BILINGUAL_PLAN_PROMPT_SUFFIX}"

        def _parse(data: Any) -> SkeletonPlan:
            plan = SkeletonPlan.model_validate(data)
            if not plan.steps:
                raise ValueError("skeleton plan contains no steps")
            return plan

        try:
            return await call_llm_json(
                agent_func,
                json.dumps(payload, ensure_ascii=False),
                system_prompt=system_prompt,
                priority=DEFAULT_QUERY_PRIORITY,
                parse=_parse,
                attempts=STAGED_LLM_ATTEMPTS,
                label="agent_staged_skeleton_plan",
                response_format=_skeleton_plan_response_format(
                    allowed_kb_ids=[str(profile["kb_id"]) for profile in kb_profiles],
                    max_kbs_per_step=max_kbs_per_step,
                    bilingual=bilingual,
                ),
            )
        except LLMJsonError as exc:
            raise HTTPException(
                status_code=502,
                detail={
                    "error_code": "agent_skeleton_plan_invalid",
                    "message": str(exc),
                },
            ) from exc

    async def _extract_skeleton(
        self,
        *,
        agent_func: Any,
        requirement_payload: dict[str, Any],
        board: _EvidenceBoard,
        bilingual: bool = False,
    ) -> tuple[SkeletonExtract | None, int]:
        chunks = board.for_stage(STAGE_SKELETON)[:_EXTRACT_MAX_CHUNKS]
        if not chunks:
            return None, 0
        output_schema: dict[str, Any] = {
            "type": "skeleton",
            "components": [
                {
                    "material": "组分/原料名",
                    "ratio": "证据中的用量（含单位）",
                    "function": "作用",
                    "source_refs": ["A1"],
                }
            ],
            "open_questions": ["完整自洽的补充检索问题"],
            "rationale": "一句话选择依据",
        }
        if bilingual:
            output_schema["open_questions_alt"] = [
                "与 open_questions 按序对应的另一语言版本"
            ]
        payload = {
            "requirement": requirement_payload,
            "bilingual_retrieval": bilingual,
            "evidence": [
                {
                    "reference_id": chunk["reference_id"],
                    "kb_id": chunk.get("kb_id"),
                    "content": str(chunk.get("content", ""))[:_EXTRACT_CHUNK_CHARS],
                }
                for chunk in chunks
            ],
            "output_schema": output_schema,
        }
        system_prompt = SKELETON_EXTRACT_SYSTEM_PROMPT
        if bilingual:
            system_prompt = (
                f"{system_prompt}\n{BILINGUAL_SKELETON_EXTRACT_PROMPT_SUFFIX}"
            )
        try:
            extract = await call_llm_json(
                agent_func,
                json.dumps(payload, ensure_ascii=False),
                system_prompt=system_prompt,
                priority=DEFAULT_QUERY_PRIORITY,
                parse=SkeletonExtract.model_validate,
                attempts=STAGED_LLM_ATTEMPTS,
                label="agent_staged_skeleton_extract",
                response_format=_skeleton_extract_response_format(
                    reference_ids=sorted(board.ids()), bilingual=bilingual
                ),
            )
        except LLMJsonError as exc:
            # Extraction is best-effort: later stages can still ground an
            # answer, and the missing skeleton is declared as a gap.
            logger.warning("Agent staged skeleton extraction failed: %s", exc)
            return None, 0
        valid_ids = board.ids()
        kept: list[SkeletonComponent] = []
        dropped = max(0, len(extract.components) - MAX_SKELETON_COMPONENTS)
        for component in extract.components[:MAX_SKELETON_COMPONENTS]:
            refs = [ref for ref in component.source_refs if ref in valid_ids]
            if not refs:
                dropped += 1
                continue
            component.source_refs = refs
            kept.append(component)
        extract.components = kept
        return extract, dropped

    async def _extract_verdicts(
        self,
        *,
        agent_func: Any,
        requirement_payload: dict[str, Any],
        properties: list[TargetProperty],
        board: _EvidenceBoard,
        property_rounds: dict[str, int],
        include_repair: bool = False,
    ) -> list[dict[str, Any]]:
        if not properties:
            return []

        def _no_data(prop: TargetProperty, note: str) -> dict[str, Any]:
            return {
                "property": prop.name,
                "priority": prop.priority,
                "verdict": "no_data",
                "evidence_refs": [],
                "note": note,
            }

        evidence_by_property = []
        for prop in properties:
            round_index = property_rounds.get(prop.name)
            prop_chunks = (
                board.for_round(round_index)[:_VERDICT_CHUNKS_PER_PROPERTY]
                if round_index
                else []
            )
            evidence_by_property.append(
                {
                    "property": prop.name,
                    "priority": prop.priority,
                    "chunks": [
                        {
                            "reference_id": chunk["reference_id"],
                            "content": str(chunk.get("content", ""))[
                                :_EXTRACT_CHUNK_CHARS
                            ],
                        }
                        for chunk in prop_chunks
                    ],
                }
            )
        other_stages = [STAGE_SKELETON, STAGE_FACTOR]
        if include_repair:
            other_stages.append(STAGE_REPAIR)
        other_evidence = [
            {
                "reference_id": chunk["reference_id"],
                "content": str(chunk.get("content", ""))[:_EXTRACT_CHUNK_CHARS],
            }
            for chunk in board.for_stage(*other_stages)[
                :_VERDICT_OTHER_EVIDENCE_CHUNKS
            ]
        ]
        if not other_evidence and all(
            not entry["chunks"] for entry in evidence_by_property
        ):
            # No evidence anywhere — fail closed without burning an LLM call.
            return [_no_data(prop, "无相关证据") for prop in properties]

        payload = {
            "requirement": requirement_payload,
            "target_properties": [
                {"name": prop.name, "priority": prop.priority}
                for prop in properties
            ],
            "evidence_by_property": evidence_by_property,
            "other_evidence": other_evidence,
            "output_schema": {
                "type": "verdicts",
                "verdicts": [
                    {
                        "property": "指标名（必须与 target_properties 一致）",
                        "verdict": "supported|partial|unsupported|no_data",
                        "evidence_refs": ["A1"],
                        "note": "一句话依据",
                    }
                ],
            },
        }
        try:
            parsed = await call_llm_json(
                agent_func,
                json.dumps(payload, ensure_ascii=False),
                system_prompt=VERDICT_SYSTEM_PROMPT,
                priority=DEFAULT_QUERY_PRIORITY,
                parse=ValidationVerdicts.model_validate,
                attempts=STAGED_LLM_ATTEMPTS,
                label="agent_staged_verdicts",
                response_format=_verdicts_response_format(
                    properties=properties, reference_ids=sorted(board.ids())
                ),
            )
        except LLMJsonError as exc:
            logger.warning("Agent staged verdict extraction failed: %s", exc)
            return [_no_data(prop, "裁决生成失败，按无数据处理") for prop in properties]

        valid_ids = board.ids()
        requested = {prop.name.lower(): prop for prop in properties}
        by_name: dict[str, dict[str, Any]] = {}
        for verdict in parsed.verdicts:
            prop = requested.get(verdict.property.lower())
            if prop is None or prop.name in by_name:
                continue
            refs = [ref for ref in verdict.evidence_refs if ref in valid_ids]
            outcome = verdict.verdict
            note = verdict.note
            if outcome in ("supported", "partial") and not refs:
                outcome = "no_data"
                note = (
                    f"{note}；" if note else ""
                ) + "结论未提供有效证据引用，已降级为无数据"
            by_name[prop.name] = {
                "property": prop.name,
                "priority": prop.priority,
                "verdict": outcome,
                "evidence_refs": refs,
                "note": note,
            }
        return [
            by_name.get(prop.name, _no_data(prop, "模型未给出该指标的裁决"))
            for prop in properties
        ]

    async def _plan_repair(
        self,
        *,
        agent_func: Any,
        requirement_payload: dict[str, Any],
        gaps: list[dict[str, Any]],
        empty_rounds: list[dict[str, Any]],
        kb_profiles: list[dict[str, Any]],
        allowed_ids: set[str],
        clipped_notes: list[str],
        priority_by_id: dict[str, Any],
        max_kbs_per_step: int,
        bilingual: bool = False,
    ) -> list[AgentPlanStep]:
        step_schema: dict[str, Any] = {
            "step_index": 1,
            "title": "短标题",
            "query": "完整自洽的补查子问题",
            "kb_ids": ["kb_xxx"],
            "mode": "naive",
            "priority": "P0",
            "hl_keywords": [],
            "ll_keywords": [],
        }
        if bilingual:
            step_schema.update(
                {
                    "query_alt": "该步子问题的另一语言完整版本",
                    "hl_keywords_alt": [],
                    "ll_keywords_alt": [],
                }
            )
        payload = {
            "requirement": requirement_payload,
            "gaps": [
                {
                    "property": gap["property"],
                    "verdict": gap["verdict"],
                    "note": gap["note"],
                }
                for gap in gaps
            ],
            "empty_steps": [
                {
                    "title": summary.get("title"),
                    "query": summary.get("query"),
                    "kb_ids": summary.get("kb_ids"),
                    "mode": summary.get("mode"),
                }
                for summary in empty_rounds
            ],
            "allowed_kbs": kb_profiles,
            "max_steps": REPAIR_MAX_STEPS,
            "max_kbs_per_step": max_kbs_per_step,
            "bilingual_retrieval": bilingual,
            "output_schema": {
                "type": "repair_plan",
                "steps": [step_schema],
            },
        }
        system_prompt = REPAIR_PLAN_SYSTEM_PROMPT
        if bilingual:
            system_prompt = f"{system_prompt}\n{BILINGUAL_PLAN_PROMPT_SUFFIX}"
        try:
            repair = await call_llm_json(
                agent_func,
                json.dumps(payload, ensure_ascii=False),
                system_prompt=system_prompt,
                priority=DEFAULT_QUERY_PRIORITY,
                parse=RepairPlan.model_validate,
                attempts=STAGED_LLM_ATTEMPTS,
                label="agent_staged_repair_plan",
                response_format=_repair_plan_response_format(
                    allowed_kb_ids=[str(profile["kb_id"]) for profile in kb_profiles],
                    max_kbs_per_step=max_kbs_per_step,
                    bilingual=bilingual,
                ),
            )
        except LLMJsonError as exc:
            logger.warning("Agent staged repair planning failed: %s", exc)
            clipped_notes.append("补查规划生成失败，本轮未补查")
            return []
        steps: list[AgentPlanStep] = []
        for step in repair.steps[:REPAIR_MAX_STEPS]:
            # Unlike the skeleton plan (fail-closed 403), an invalid repair
            # step is dropped so already-accumulated evidence is not thrown
            # away; the unauthorized retrieval is still never executed.
            if step.mode not in AGENT_ALLOWED_MODES or any(
                kb_id not in allowed_ids for kb_id in step.kb_ids
            ):
                logger.warning(
                    "Agent staged repair step dropped: mode/kb not allowed"
                )
                clipped_notes.append(
                    f"补查步骤“{step.title or step.query[:40]}”包含不可用模式或越权知识库，已丢弃"
                )
                continue
            if len(step.kb_ids) > max_kbs_per_step:
                step.kb_ids = self._cap_kbs(
                    step.kb_ids, priority_by_id, max_kbs_per_step
                )
            steps.append(step)
        return steps

    # ------------------------------------------------------------------
    # Pure helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _requirement_payload(
        requirement: StagedRequirement, properties: list[TargetProperty]
    ) -> dict[str, Any]:
        def _prop_payload(prop: TargetProperty) -> dict[str, Any]:
            payload = {"name": prop.name, "why": prop.why, "priority": prop.priority}
            if prop.name_alt:
                payload["name_alt"] = prop.name_alt
            return payload

        return {
            "application": requirement.application,
            "conditions": requirement.conditions,
            "target_properties": [_prop_payload(prop) for prop in properties],
            "constraints": requirement.constraints,
        }

    @staticmethod
    def _component_payload(skeleton: SkeletonExtract | None) -> list[dict[str, Any]]:
        if skeleton is None:
            return []
        return [
            {
                "material": component.material,
                "ratio": component.ratio,
                "function": component.function,
                "source_refs": component.source_refs,
            }
            for component in skeleton.components
        ]

    @staticmethod
    def _factor_queries(
        *,
        requirement: StagedRequirement,
        skeleton: SkeletonExtract | None,
        allowance: int,
        bilingual: bool = False,
    ) -> list[tuple[str, str]]:
        """Return ``(query, query_alt)`` pairs; ``query_alt`` is empty when
        bilingual is off or no reliable pairing exists."""
        if skeleton is None or allowance <= 0:
            return []
        # Positional pairing is only trusted when the alt list survived
        # validation with the same length (open_questions is deduplicated,
        # which would silently shift indexes otherwise).
        alts: list[str] = []
        if bilingual and len(skeleton.open_questions_alt) == len(
            skeleton.open_questions
        ):
            alts = skeleton.open_questions_alt
        queries: list[tuple[str, str]] = []
        seen: set[str] = set()
        for index, question in enumerate(skeleton.open_questions):
            text = question.strip()
            alt = alts[index].strip() if index < len(alts) else ""
            if len(text) >= 3 and text not in seen:
                seen.add(text)
                queries.append((text, alt))
            if len(queries) >= allowance:
                return queries
        conditions_text = "、".join(requirement.conditions) or "目标环境"
        for component in skeleton.components:
            text = (
                f"{requirement.application}在{conditions_text}条件下，"
                f"{component.material}的用量与影响机理"
            )
            if text not in seen:
                seen.add(text)
                queries.append((text, ""))
            if len(queries) >= allowance:
                break
        return queries

    @staticmethod
    def _cap_kbs(
        kb_ids: list[str], priority_by_id: dict[str, Any], limit: int
    ) -> list[str]:
        """Narrow a KB list to ``limit`` entries, preferring higher manual
        ``agent_priority`` and keeping the original order as tie-breaker."""
        if len(kb_ids) <= limit:
            return kb_ids

        def _rank(kb_id: str) -> tuple[int, int]:
            try:
                priority = int(priority_by_id.get(kb_id) or 0)
            except (TypeError, ValueError):
                priority = 0
            return (-priority, kb_ids.index(kb_id))

        return sorted(kb_ids, key=_rank)[:limit]

    @staticmethod
    def _kbs_with_roles(
        effective_records: list[KnowledgeBaseRecord],
        kb_roles: dict[str, str],
        roles: tuple[str, ...],
    ) -> list[str]:
        ids = [
            record.id
            for record in effective_records
            if kb_roles.get(record.id) in roles
        ]
        return ids or [record.id for record in effective_records]

    @staticmethod
    def _validation_kbs(
        effective_records: list[KnowledgeBaseRecord], kb_roles: dict[str, str]
    ) -> list[str]:
        for roles in (
            ("experimental",),
            ("literature", "application_spec"),
        ):
            ids = [
                record.id
                for record in effective_records
                if kb_roles.get(record.id) in roles
            ]
            if ids:
                return ids
        return [record.id for record in effective_records]

    def _build_references(
        self,
        *,
        body: "AgentQueryRequest",
        board: _EvidenceBoard,
        cited_ids: set[str],
        synth_result: QueryToolResult,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        """Token-budgeted context selection with cited-evidence priority.

        Chunks cited by the skeleton or by property verdicts are placed
        before uncited ones so budget truncation can only drop evidence that
        no structured conclusion depends on."""
        cited = [c for c in board.chunks if c["reference_id"] in cited_ids]
        uncited = [c for c in board.chunks if c["reference_id"] not in cited_ids]
        ordered = cited + uncited
        rag = synth_result.rag
        param = synth_result.param
        tokenizer = None
        if rag is not None:
            tokenizer = rag._build_global_config().get("tokenizer")
        budget = body.max_total_tokens or (
            getattr(param, "max_total_tokens", None) if param is not None else None
        )
        if tokenizer is not None and budget:
            ordered = truncate_list_by_token_size(
                ordered,
                key=lambda chunk: str(chunk.get("content", "")),
                max_token_size=int(budget),
                tokenizer=tokenizer,
            )
        ordered = sorted(ordered, key=lambda chunk: int(chunk["reference_id"][1:]))
        references: list[dict[str, Any]] = []
        context_units: list[dict[str, str]] = []
        for chunk in ordered:
            content = str(chunk.get("content", ""))
            ref = {
                "reference_id": chunk["reference_id"],
                "kb_id": str(chunk.get("kb_id", "")),
                "stage": chunk["stage"],
                "evidence_role": chunk["evidence_role"],
                "round": chunk["round_index"],
                "step_index": chunk["step_index"],
                "mode": chunk["mode"],
                "file_path": str(
                    chunk.get("file_path") or chunk.get("source") or "unknown"
                ),
                "chunk_id": chunk.get("chunk_id"),
                "source_reference_id": chunk.get("source_reference_id"),
            }
            if body.include_chunk_content:
                ref["content"] = [content]
            references.append(ref)
            context_units.append(
                {"reference_id": chunk["reference_id"], "content": content}
            )
        return references, context_units

    @staticmethod
    def _synthesis_rules(
        *,
        requirement_payload: dict[str, Any],
        skeleton: SkeletonExtract | None,
        verdicts: list[dict[str, Any]],
        clipped_notes: list[str],
    ) -> str:
        summary = {
            "requirement": requirement_payload,
            "skeleton_components": AgentStagedRunner._component_payload(skeleton),
            "property_verdicts": verdicts,
            "clipped": clipped_notes,
        }
        return (
            f"{SYNTHESIS_EXTRA_RULES}\n"
            "结构化需求、骨架与指标裁决（引用编号已与证据对应，直接用于组织回答）：\n"
            f"{json.dumps(summary, ensure_ascii=False)}"
        )
