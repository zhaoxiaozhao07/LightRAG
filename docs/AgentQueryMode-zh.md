# LightRAG 企业级 Agent 查询模式 — 需求与设计方案

> 文档版本：2026-07-01（v2）  
> 适用范围：在现有 LightRAG API Server、知识库（KB）多模式检索、企业认证与 KB 级 RBAC/ACL 之上，新增面向生产环境的 **Agent 多轮查询编排** 能力（产品面称 **「Agent 模式」**）。  
> 目标读者：架构师、后端与前端集成者、安全审计、单机/私有化部署运维。  
> 关联文档：`docs/archive/企业级多用户权限管理改造设计方案.md`、`docs/LightRAG-API-Server-zh.md`、`docs/RoleSpecificLLMConfiguration-zh.md`、`docs/API接口.md`、`docs/AgentStagedRecommendation-zh.md`（阶段化配比推荐工作流）。

---

## 1. 文档说明

本文描述 **需求边界、设计理念、部署形态、权限分级、编排 Workflow、内置提示词、用户可配置策略、环境与角色配置、运维治理**。实现阶段应在本设计约束下单独出技术规格（OpenAPI、事件流 schema）与测试计划。

**术语约定**

| 术语 | 含义 |
|------|------|
| **Agent 模式** | 面向用户/API 的一种 **查询交互模式**（独立入口），不是 `QueryParam.mode` 的新枚举值。 |
| **编排层** | 使用 **AGENT** 角色 LLM 做规划、轮次决策与结构化输出；执行层仍调用既有检索能力。 |
| **底层检索 mode** | 每轮子查询实际使用的 `local` / `global` / `hybrid` / `naive` / `mix`；**禁止 `bypass`**（无检索、无引用，对 Agent 无意义）。 |

---

## 2. 背景与动机

### 2.1 现有查询能力

LightRAG 已提供：

- **单知识库**：RAG 问答（含流式）、仅检索（`query/data`、`retrieve`），六种底层 `mode` 及过滤、关键词、`conversation_history`（仅影响生成，不参与检索）。
- **多知识库**：对固定 `kb_ids` 并行检索、合并 chunk 后一次合成（`:query` / `:retrieve`）；多 KB 路径不支持 `bypass`。

上述能力假设 **一次请求** 内已确定问句、KB 范围、检索 mode。复杂开放问题需要 **多轮子查询、换库、换 mode、评估证据是否充分**。

### 2.2 Agent 模式要解决的问题

1. **任务分解**：开放问题拆为可检索子问题，并按业务优先级排序（如法规先于配方）。  
2. **动态选库**：在 **用户被授权且本次会话允许的 KB 集合** 内，由编排模型按轮次选择单库或多库。  
3. **动态选 mode**：按子问题在 `hybrid`、`mix`、`local` 等之间切换。  
4. **可控成本**：多轮默认 **仅检索**；终答基于 **证据板** 合成，避免每轮完整 RAG 生成。  
5. **企业合规**：JWT / 服务密钥、KB 角色、审计、限流与现有体系一致；**权限在服务端强制执行**。

### 2.3 部署前提（本版明确）

- 参与 Agent 的 **所有知识库** 使用 **同一套** 后端 **Embedding、Rerank、Query LLM** 配置（与当前企业单机/局域网典型部署一致）。在此前提下，多 KB 合并检索与分数可比性风险显著降低；仍须在服务端校验 KB 处于可用状态且用户具备 `kb_viewer`。  
- **性能约束**：Agent 会话内 **仅串行** 执行检索步骤（一轮完成后再进入下一轮）；不做并行 retrieve。

---

## 3. 需求定义

### 3.1 功能性需求

| 编号 | 需求 | 说明 |
|------|------|------|
| F1 | Agent 模式入口 | 独立 API（及 WebUI 入口），不与普通 `/query` 的 `mode` 字段混用。 |
| F2 | 多轮检索 | 可配置 `max_rounds`（建议默认 5～6）；每轮调用与现有 **单库 retrieve / 多库 :retrieve** 等价的受权能力，并指定底层 `mode`（非 bypass）。 |
| F3 | 证据累积 | 跨轮结果在服务端 **证据板** 去重、分组，受 token 预算约束后进入终答合成。 |
| F4 | 终答与引用 | 终答可追溯至统一 **Agent 级引用编号**（含 `kb_id`、`round`、文件路径等）；法规/禁忌类结论优先于推荐类（由工作流提示词约束）。 |
| F5 | 单库与多库 | 每轮可为单 KB 或多 KB（同一子问）；多 KB 仅在会话允许集合内且 embedding 配置一致前提下使用。 |
| F6 | 用户指定知识库 | 请求可传 `candidate_kb_ids`（长度 1～N）。有效集合 = **授权 KB ∩ candidate_kb_ids**；若 **未传或为空**，表示候选范围为 **全部授权 KB**，由 Agent 模型在规划时自行决定使用哪些库。不得指定授权外的 `kb_id`。 |
| F7 | 澄清 | 关键约束缺失且无法从 KB 推断时，编排 LLM 可输出 **澄清**（结构化 JSON）；**不调用检索**；客户端可携带同一会话上下文再次请求（产品可配置是否持久 `session_id`）。 |
| F8 | 快速路径 | 简单单库问题可缩短规划与轮次（如 1 次检索 + 证据合成），仍须会话门禁与权限校验。 |
| F9 | 工作流提示词 | 系统内置默认 **Agent 工作流提示词**（见 §8）；支持 **按用户** 持久化自定义工作流提示词，通过企业 API 读写；执行时 **用户自定义覆盖或追加** 于默认策略（实现时二选一：覆盖默认 vs 追加，建议 **追加在系统指令之后**，并限制最大长度）。 |
| F10 | 串行执行 | 所有检索轮次 **严格串行**；规划中的多步按顺序执行，不启用并行 retrieve。 |

### 3.2 非功能性需求

| 编号 | 需求 | 说明 |
|------|------|------|
| NF1 | 企业模式 | 生产多用户场景下 Agent **完整治理** 与企业认证对齐（`LIGHTRAG_ENTERPRISE_AUTH_ENABLED`）。 |
| NF2 | 单机/私有化 | 编排（AGENT 角色）与终答（QUERY 角色）可指向 **同一本地 OpenAI 兼容端点** 的同一模型，或按角色拆分；见 §9。 |
| NF3 | 可观测 | 审计关联用户、KB、底层 mode、查询哈希、轮次、结果摘要；**不记录**完整用户文档正文与 chunk 全文。 |
| NF4 | 限流与配额 | Agent 按 **会话内轮次** 或 **内部 retrieve 次数** 计费扩展，防止单次 HTTP 请求绕过频率限制。 |
| NF5 | 失败语义 | 部分 KB 不可用（404、删除中 409）与现有多 KB 行为一致；禁止静默越权。 |

### 3.3 明确非目标（第一期）

- 不在 `QueryParam.mode` 中增加 `agent` 或 `bypass` 的 Agent 用法。  
- 不在浏览器内执行 Agent tool loop 或存放 AGENT LLM 密钥。  
- 不实现跨租户联合 Agent。  
- 不默认并行检索。  
- 第一期可不实现独立 Agent 进程（默认同进程 API 模块）。

---

## 4. 设计理念

### 4.1 核心原则

1. **Agent 是模式，不是底层检索算法**  
   对用户与集成方呈现为 **Agent 模式**；实现上为 API Server **编排模块** + **AGENT 角色 LLM**；每轮工具执行仍传 `local/global/hybrid/naive/mix`。

2. **编排在上、检索在下**  
   检索与图谱/向量逻辑仍由现有 `LightRAG` 与 **Query Tool Service**（复用 KB 路由的鉴权、filter、生命周期、审计）完成。

3. **Fail Closed**  
   身份不明、KB 未授权、越权 `candidate_kb_ids`、`bypass` 出现在规划中一律拒绝执行。

4. **服务端权威 KB 集合**  
   `effective_kb_ids = authorized_kb_ids ∩ candidate_kb_ids`（未指定 candidate 时 `effective_kb_ids = authorized_kb_ids`）。每轮工具执行前再次校验 `kb_id ⊆ effective_kb_ids`。

5. **检索与生成解耦**  
   多轮阶段默认 **仅检索**；终答优先 **证据板 + QUERY 角色合成**（复用/泛化多 KB synthesis 思路），避免终答阶段再完整 `query` 导致引用与多轮证据不一致。

6. **对话历史不替代检索**  
   `conversation_history` 不参与检索；多轮信息需求须体现为 **显式子 query** 与过滤条件（写入工作流提示词）。

7. **结构化规划，非 Provider 原生 Tool Call**  
   AGENT 角色通过 **OpenAI 兼容 `response_format` / JSON 模式**（或等价约束）输出规划与轮次决策；由服务端 **校验 schema 后** 调用内部工具，不依赖各厂商不一致的 tools API。

8. **串行与成本**  
   串行执行降低峰值 GPU/并发压力；配合 `max_rounds`、超时与配额控制总成本。

### 4.2 产品定位

- **用户**：选择 Agent 模式、可选勾选 KB、查看步骤摘要与引用。  
- **管理员**：高成本查询类型，通过 `can_use_agent_query` 与全局开关治理。  
- **集成方**：`POST /agent/query`（及 stream）会话型 façade；内部等价于已鉴权的 retrieve/synthesis。

---

## 5. 部署架构

### 5.1 推荐形态

**Agent 编排与 `lightrag-server` 同进程**，与 JWT、企业中间件、KB 路由同上下文。

```text
客户端（WebUI / 第三方）
        │
        ▼
LightRAG API Server
├── 企业认证与 KB 访问治理（现有）
├── KB Query Tool Service（建议抽取，供 route + Agent 共用）
├── KB 查询路由（现有）
└── Agent 模块（新增）
        ├── 会话门禁与 effective_kb_ids
        ├── AGENT 角色 LLM（JSON 规划/决策）
        ├── 串行工具执行器（retrieve / multi retrieve）
        ├── 证据板 + 引用重编号
        ├── 内置工作流提示词 + 用户自定义工作流提示词
        └── QUERY 角色终答合成
        │
        ▼
各 KB 对应 LightRAG 实例（registry）
```

### 5.2 前端职责

- Agent 模式 **独立入口**（不放入普通六种 `QueryMode` 下拉框）。  
- KB 多选列表 **仅展示服务端返回的已授权 KB**；提交 `candidate_kb_ids` 可选。  
- 不执行多轮 retrieve；不保存 AGENT API Key。

---

## 6. 权限与治理

### 6.1 与现有企业模型对齐

KB 角色阶梯：`kb_viewer` < `kb_editor` < `kb_admin` < `kb_owner`。  
平台 flags 扩展建议：`can_use_agent_query`（默认关）。

| 能力 | 最低要求 |
|------|----------|
| 开启 Agent 会话 | 已认证 + `can_use_agent_query`（及全局 `LIGHTRAG_AGENT_QUERY_ENABLED`） |
| 会话内 retrieve / 终答合成 | 对每个涉及的 `kb_id` 具备 **kb_viewer** |
| `bypass` | **Agent 不支持**；单次 bypass 仍走现有 `can_use_bypass_query` |
| 服务密钥 | 仅能在密钥 `kb_roles` scope 内使用 Agent |

### 6.2 知识库范围计算

1. **authorized_kb_ids**：当前 principal 下满足 **kb_viewer** 的全部 KB（含 tenant effective role）。  
2. **candidate_kb_ids**（请求可选）：用户本次希望 Agent 考虑的子集。  
3. **effective_kb_ids**：  
   - 若 `candidate_kb_ids` 非空：`authorized_kb_ids ∩ candidate_kb_ids`；交集为空 → **400/403**，不调用 AGENT LLM。  
   - 若未传或为空：`effective_kb_ids = authorized_kb_ids`；规划时仅允许使用其中部分或全部，由模型在 JSON 计划中声明 `kb_ids`。  
4. 注入编排上下文的 KB 列表：**仅 effective 集合** 的 id、名称、描述与 Agent Profile（不泄露未授权 KB）。Agent Profile 由人工覆盖字段 `agent_description` / `agent_tags` / `agent_priority` 与自动字段 `agent_auto_profile` 合并而来；自动字段由 `PROFILE` 角色 LLM 按“文档级 profile → KB 级 profile”后台生成。Profile 只帮助模型判断哪个 KB 更相关，不改变 RBAC。

### 6.3 审计与限流

- 建议事件：`agent_session_started`、`agent_retrieve_round`、`agent_query_completed`、`agent_session_failed`。  
- 关联既有 `retrieve_executed` / `query_executed` 时 metadata 带 `agent_session_id`。  
- 限流：建议 **每轮 retrieve 计 1 单位** query 配额（可配置）。

---

## 7. Agent 工作流（串行）

### 7.1 总览

```text
用户提问 + 可选 candidate_kb_ids
    ▼
[0] 门禁：鉴权 → can_use_agent_query → effective_kb_ids
    ▼
[1] 规划（AGENT LLM，JSON）：子问题、每步 kb_ids、底层 mode、优先级；或 clarification
        （JSON 解析失败自动重试，最多 3 次；仍失败 → 502 + agent_session_failed 审计）
    ▼（若 clarification → 返回用户，结束本轮 HTTP）
[2] 检索循环（串行，最多 max_rounds；步骤按 P0→P1→P2 稳定排序后截断，保证 P0 保留）：
        for step in plan.steps:
            校验 kb_ids ⊆ effective_kb_ids，mode ∈ {local,global,hybrid,naive,mix}
            执行单库或多库 retrieve（Query Tool Service）
            失败 → 该步标记 status=failed，继续后续步骤（全部失败才 502）
            写入证据板
            （可选，后续版本）AGENT LLM 评估是否 CONTINUE / REFINE / FINISH（JSON）
    ▼
[3] 证据整理：去重、按轮次轮转合并（不做二次 rerank）、token 裁剪、Agent 引用编号 A1,A2,...
    ▼
[4] 终答合成（QUERY LLM + 证据包模板，include_references；失败步骤/跳过 KB 的缺口写入合成约束）
    ▼
返回：answer、references、steps_summary（含每步 status）、metadata（含 plan_truncated、failed_round_count）
```

### 7.2 底层 mode 选用（写入默认工作流提示词）

| 子问题类型 | 建议底层 mode |
|------------|----------------|
| 概念、通则、关系网 | `global` 或 `hybrid` |
| 实体、配方、参数、工艺 | `mix` 或 `local` |
| 原文/条文定位 | `naive` |
| 同一子问需多库横向对比 | 多库 `:retrieve`，`kb_ids` 为 effective 子集 |

### 7.3 停止条件

- 评估 FINISH 且关键子问题（P0）在证据板有记录；或达到 `max_rounds`；或超时；或用户取消。  
- 未满足 P0 且已达 `max_rounds`：终答 **声明证据缺口**，禁止编造。

### 7.4 流式输出（已实现）

NDJSON 事件按执行进度 **实时输出**：`session_started`、`plan_created`（规划完成即发）、`round_started`、`round_result`（每轮检索前后）、`references`、`response`（终答增量 delta，可多条）、`clarification_required`、`done`、`error`。
与现有仅 `response` 的 query stream 区分，需 WebUI 独立 parser。

**证据合成语义**：每步检索结果已在检索内按该步子问题 rerank；证据板合并时 **不做二次 rerank**（避免子问题证据被总问题相关性打分整体挤掉），而是按轮次轮转交错合并，再按 `max_total_tokens` 预算截断。失败步骤与跳过的 KB 作为"已知检索缺口"注入终答约束，要求模型明确声明未覆盖内容。

---

## 8. 内置工作流提示词（默认）

以下为 **系统默认** 策略文本（实现时可放入 `lightrag/prompt.py` 或 Agent 模块常量）；**用户自定义工作流提示词**（§10）在运行时拼接。

**角色与目标**

你是 LightRAG Agent 编排器。用户提出复杂问题；你只能在服务端提供的 **允许知识库列表** 内选择 `kb_ids`，并为每个检索步骤指定 **底层检索模式**（`local`、`global`、`hybrid`、`naive`、`mix` 之一）。**禁止使用 bypass。** 你不直接回答最终问题，只输出 **严格 JSON**。

**输入上下文（由服务端注入，勿虚构）**

- `user_question`：用户原始问题  
- `allowed_kbs`：`[{ "kb_id", "name", "description", "agent_description", "agent_tags", "agent_priority", "agent_auto_profile_status", "agent_profile_domains", "agent_profile_sample_questions", "agent_profile_negative_scope" }]`，仅限 effective 集合  
- `max_rounds`：最大检索轮次  
- `default_retrieve_params`：如 `top_k`、`chunk_top_k` 上限（勿超出）  
- `user_workflow_prompt`：用户自定义策略（可为空）

**输出 JSON Schema（规划阶段）**

```json
{
  "type": "plan",
  "clarification_required": false,
  "clarification_question": null,
  "steps": [
    {
      "step_index": 1,
      "title": "短标题，供步骤摘要展示",
      "query": "面向检索的子问题，完整自洽",
      "kb_ids": ["kb_xxx"],
      "mode": "mix",
      "priority": "P0",
      "hl_keywords": [],
      "ll_keywords": []
    }
  ],
  "notes_for_user": "可选，一句说明，非思维链"
}
```

若需澄清：`clarification_required: true`，`steps` 为空数组，`clarification_question` 必填。

**规则**

1. 每个 `kb_id` 必须属于 `allowed_kbs`；可多库仅当同一子问需要横向合并。  
2. `steps` 数量 ≤ `max_rounds`；服务端 **串行** 执行，按 `step_index` 顺序。  
3. 法规、禁忌、合规类子问题标 **P0** 并优先排在前面。  
4. 子 `query` 必须可独立检索；不要把“根据上文”当作检索条件。  
5. 不要输出 markdown 包裹的 JSON 以外的解释；不要输出 chain-of-thought。  
6. 若用户未限定 KB，你应主动选择最相关的库，而非每轮查询全部库。

**轮次评估输出（可选，每轮检索后）**

```json
{
  "type": "evaluate",
  "action": "CONTINUE | REFINE | FINISH_RETRIEVE",
  "next_step": null,
  "reason_summary": "一句，供审计与步骤摘要"
}
```

`REFINE` 时 `next_step` 提供改写后的单步（仍须 JSON 校验）。

**终答阶段**  
不由 AGENT 角色直接生成用户可见长文；由 **QUERY 角色** 在服务端基于证据包与引用模板合成（工作流提示词可要求终答结构：结论、依据、风险、未覆盖点）。

---

## 9. 环境与 AGENT / PROFILE 角色配置

### 9.1 新增角色

在 `lightrag.llm_roles.ROLES` 中增加与 `QUERY` 平行的角色：

```text
RoleSpec("agent", "AGENT", "agent LLM func")
RoleSpec("profile", "PROFILE", "profile LLM func")
```

### 9.2 `.env` 配置（参考 QUERY）

与现有角色级变量一致，规划前缀为 **`AGENT_`**，Profile 生成前缀为 **`PROFILE_`**。单机场景可与 QUERY **共用同一本地 OpenAI 兼容服务**（同一 `host`、同一 `model`），仅角色队列与超时独立配置。

示例（与 `env.enterprise-single-server.example` / `QUERY_LLM_*` 对齐）：

```bash
### Agent orchestration LLM (planning / JSON decisions)
# Available roles: EXTRACT, KEYWORD, QUERY, AGENT, PROFILE, VLM
AGENT_LLM_BINDING=openai
AGENT_LLM_BINDING_HOST=http://127.0.0.1:8000/v1
AGENT_LLM_BINDING_API_KEY=not-needed-or-local-key
AGENT_LLM_MODEL=qwen3.6-36b
AGENT_LLM_TIMEOUT=300

### Profile generation LLM (document profile -> KB profile)
PROFILE_LLM_BINDING=openai
PROFILE_LLM_BINDING_HOST=http://127.0.0.1:8000/v1
PROFILE_LLM_BINDING_API_KEY=not-needed-or-local-key
PROFILE_LLM_MODEL=qwen3.6-36b
PROFILE_LLM_TIMEOUT=300

# 可选：限制 AGENT 调用为 JSON 输出（实现层对 openai binding 设置 response_format）
# AGENT_OPENAI_RESPONSE_FORMAT=json_object
```

说明：

- **绑定**：优先 `openai` 兼容本地 vLLM / SGLang / Ollama OpenAI 路由。  
- **JSON**：规划与评估调用须 **强制结构化输出**（`response_format: json_object` 或项目内等价封装）；解析失败时重试有限次数，仍失败则 `agent_session_failed`。  
- **与 QUERY 分工**：AGENT 负责 plan/evaluate（小步、短输出）；QUERY 负责终答合成（长文本、引用格式）。两角色可 **同一模型、同一 endpoint**，便于运维。  
- **与 PROFILE 分工**：PROFILE 负责文档级/KB 级 profile JSON 生成，默认后台任务执行，不阻塞文档入库主流程。  
- **Embedding / Rerank**：不新增角色；各 KB 共用部署级 embedding/rerank（本设计前提 §2.3）。

### 9.3 功能开关

```bash
LIGHTRAG_AGENT_QUERY_ENABLED=false
LIGHTRAG_AGENT_PROFILE_AUTO_REFRESH=true
AGENT_PROFILE_REFRESH_DOC_DELTA=1
AGENT_PROFILE_REFRESH_MIN_INTERVAL_SECONDS=0
```

`LIGHTRAG_AGENT_QUERY_ENABLED` 与企业模式、`can_use_agent_query` 共同生效。Profile 自动刷新开关与节流参数只影响后台 `agent_profile` job；手动 `POST /kbs/{kb_id}/agent-profile:refresh` 不受节流限制。

---

## 10. KB Agent Profile 自动生成

### 10.1 数据结构

采用“自动字段 + 人工覆盖字段”：

| 层级 | 存储位置 | 说明 |
|------|----------|------|
| 文档级自动 profile | `documents.metadata.agent_doc_profile` | 由 `PROFILE` 角色根据解析后的 `full_docs` 内容抽样生成，包含 `summary`、`tags`、`domains`、`sample_questions`、`negative_scope`、`source_hash`、`index_hash` |
| 组摘要缓存（大库） | `documents.metadata.agent_group_profile`（组首文档） | KB 超过 128 篇时的分层摘要中间结果：文档按创建顺序切成 128 篇/组，每组一份 LLM 组摘要，缓存键 `group_hash` 覆盖成员集合与各成员 profile 版本，成员或内容变化才重算 |
| KB 级自动 profile | `knowledge_bases.metadata.agent_auto_profile` | 聚合文档级 profile（或组摘要）后生成，包含 `description`、`tags`、`domains`、`sample_questions`、`negative_scope`、`status`、`source_doc_count`、`profiled_doc_count`、`pending_document_profiles`、`aggregation_mode`、`group_count`、`updated_at`、`job_id` |
| KB 级脏标记 | `knowledge_bases.metadata.agent_auto_profile_dirty` | 文档事件写入 `{dirty_at, reason, document_id}`；独立于 profile 键，刷新完成只清除早于本次刷新起点的标记，晚于起点的标记触发链式刷新 |
| KB 级人工覆盖 | `knowledge_bases.metadata.agent_description` / `agent_tags` / `agent_priority` / `agent_domains` / `agent_sample_questions` / `agent_negative_scope` | 前端可编辑；人工字段优先于自动字段 |

合并规则：

- `agent_description` 非空时覆盖 `agent_auto_profile.description`。
- `agent_tags` 非空时覆盖 `agent_auto_profile.tags`。
- `agent_domains` / `agent_sample_questions` / `agent_negative_scope` 非空时覆盖自动 profile 的同名字段（纠正自动生成的选库偏差，尤其是错误的 `negative_scope`）。
- `agent_priority` 仅人工控制，默认 `0`。
- 自动 profile 缺失、dirty、failed 时，Agent 查询不失败，回退到 KB `name` / `description` / 人工字段。
- 单篇文档 profile 生成失败不阻塞 KB 级聚合：失败计入 `document_profiles_failed`（明细 `failed_documents`），其余文档照常聚合；全部失败且无缓存时才判刷新失败。

### 10.2 后台生成流程

```text
文档解析 + 构建完成（status=ready）/ 启用停用 / 删除
    ▼
写入独立脏标记 knowledge_bases.metadata.agent_auto_profile_dirty
（独立键：刷新完成写回 profile 不会覆盖并发到达的脏标记）
    ▼
入队 job_type=agent_profile（若该 KB 已有排队/运行中的 profile job 则合并，不重复建 job）
    ▼
PROFILE LLM：为缓存失效（source_hash/index_hash 变化）的文档生成 agent_doc_profile，
每次刷新最多生成 24 篇；其余记入 pending_document_profiles
    ▼
KB 级聚合按已生成 profile 的文档数 N 自动选模式：
  - direct（N ≤ 128）：全部文档 profile 逐条聚合
  - sampled（N > 128 且回填未完成）：全库等距抽样 128 条 + tag/domain 频次统计（过渡模式）
  - grouped（N > 128 且回填收敛）：分层摘要——按创建顺序 128 篇/组，
    每组一份缓存的 LLM 组摘要（组首文档 agent_group_profile，group_hash 失效才重算），
    最终聚合全部组摘要 + 频次统计；追加式入库稳态只重算尾部一组
    ▼
写回 KB metadata；刷新期间若有新脏标记或仍有 pending 文档 → 自动追加链式刷新（chained_refresh）
失败只标记 failed，不影响文档 ready；LLM 超长输出截断而非拒绝，JSON 解析失败重试（最多 3 次）
```

第一期直接在 API Server 内后台执行；启用 `LIGHTRAG_KB_JOB_WORKER=true` 时，`agent_profile` job 也可被 durable worker 认领续跑。Profile 生成只读取当前 KB 的控制面文档与 `full_docs`，不会绕过用户查询时的 RBAC；`allowed_kbs` 注入仍只发生在授权 KB 子集内。

### 10.3 API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/kbs/{kb_id}/agent-profile` | 返回 `manual`、`auto`、`effective` 三段，供前端展示 |
| PUT | `/kbs/{kb_id}/agent-profile` | 写入/清空人工覆盖字段 |
| POST | `/kbs/{kb_id}/agent-profile:refresh` | 手动排队重新调用 `PROFILE` LLM 生成自动 profile |

权限：GET 需 `kb_viewer`，PUT/POST 需 `kb_editor`；企业模式下写审计事件 `kb_agent_profile_manual_updated`、`kb_agent_profile_refresh_queued`。

---

## 11. 用户自定义工作流提示词

### 11.1 需求

- 不同用户可对 Agent **编排策略** 做个性化（如行业术语、输出结构、优先查某类库）。  
- 须 **服务端持久化**，不依赖浏览器 localStorage 作为唯一来源。

### 11.2 API 形态（建议，对齐 KB 级 `user_prompt`）

| 方法 | 路径（示意） | 说明 |
|------|----------------|------|
| GET | `/auth/me/agent-workflow-prompt` | 读取当前用户自定义工作流提示词 |
| PUT | `/auth/me/agent-workflow-prompt` | 写入/清空（`max_length` 建议 8192～16384，与现有 user_prompt 量级一致） |

请求体示例：

```json
{ "workflow_prompt": "你是化妆品配方助手。凡涉及法规限用，必须先检索法规库并 P0 标注。" }
```

执行 Agent 查询时：

- 服务端加载 **默认内置工作流提示词（§8）** + **当前用户 `workflow_prompt`**（非空则按产品规则追加或覆盖段落）。  
- 审计仅记录 `has_custom_workflow_prompt: true/false`，不记录全文（或仅 hash）。

### 11.3 与 KB 级 `user_prompt` 的关系

- KB 级 `user_prompt` 仍作用于该 KB 的 **单次 query/retrieve** 默认参数。  
- Agent **全局工作流提示词** 作用于 **编排与终答结构**；终答合成可将 KB 级提示词按主 KB 合并（实现细则在技术规格中定义）。  
- 避免两处提示词职责重叠：工作流提示词管 **多轮策略**；KB user_prompt 管 **单库生成偏好**。

---

## 12. 工具抽象（逻辑层）

| 逻辑工具 | 映射能力 | 备注 |
|----------|----------|------|
| 列出 effective KB | 门禁结果 | 与 UI 一致 |
| 单库检索 | 单 KB retrieve / `aquery_data` | 串行主路径 |
| 多库检索 | `:retrieve` | 同一子问，`kb_ids ⊆ effective_kb_ids` |
| 终答合成 | 证据包 + QUERY LLM | 非重新全量 query |

**不建议第一期开放**：`bypass`、文档上传、图谱编辑、artifact、未授权 KB 探测。

**实现要求**：工具实现 **必须** 经 **Query Tool Service**，不得裸调 `registry.get(kb).aquery_data()` 以免绕过 filter 与文档生命周期。

---

## 13. 安全、审计与合规

- 会话证据缓存按 user/session 隔离，超时回收；敏感部署可仅内存、不落盘。  
- 错误信息不泄露未授权 KB 是否存在。  
- 配方/医疗/法规场景：默认 P0 法规子问题 + 终答风险提示（可由用户工作流提示词加强）。

---

## 14. 与现有 API 对照

| 用户期望 | 现有 API | Agent 模式 |
|----------|----------|------------|
| 一次问一次答 | 单库 `/query` | 快速路径可近似 |
| 固定多库一次合并 | `/kbs:query` | 规划中锁定库后的某一子步或终答 |
| 只看检索结果 | `/retrieve` | 阶段 2 默认动作 |
| 指定部分 KB 做 Agent | 无 | `candidate_kb_ids` + effective 交集 |
| 多轮换库换 mode | 调用方自循环 | 服务端串行自动完成 |

**不推荐**：`{ "mode": "agent" }` 作为 `QueryParam.mode`。

---

## 15. 实施路线

| 阶段 | 内容 |
|------|------|
| A | 冻结：Agent API、NDJSON 事件、`can_use_agent_query`、AGENT 角色与 `.env`、用户工作流提示词 API |
| B | Query Tool Service 抽取；门禁 + 串行单库多轮 retrieve + 证据合成终答 |
| C | PROFILE 角色；文档级 profile → KB 级自动 profile；查看/人工覆盖/手动刷新 API |
| D | 多库 retrieve、引用重编号、流式事件、WebUI Agent 面板 |
| E | 配额、审计、metrics、runbook；可选独立 Agent worker 进程（loopback + 凭证透传） |

---

## 16. 测试要点（摘要）

- `candidate_kb_ids` 越权、交集为空、未指定时使用全部授权库。  
- 规划含 `bypass` 或非法 `kb_id` 被拒绝。  
- 串行顺序与 `max_rounds`。  
- AGENT JSON 解析失败与重试。  
- 用户工作流提示词读写与执行拼接。  
- PROFILE 角色 JSON 生成、文档级缓存复用、KB 级自动 profile 写回、人工覆盖优先。  
- 引用 A1/A2 跨轮不冲突。  
- 企业限流按轮次计费。  

---

## 17. 总结

**Agent 模式** 是面向用户的一种 **独立查询模式**：用 **AGENT 角色 LLM**（`.env` 中 `AGENT_LLM_*`，可与 QUERY 共用本地同一模型）输出 **JSON 规划**，在 **串行** 工作流中调用既有 **local/global/hybrid/naive/mix** 检索（**不用 bypass**），在 **用户指定或默认全部授权 KB** 范围内选库。KB 选择上下文由人工 Agent Profile 与 **PROFILE 角色**后台生成的自动 profile 合并而来；终答由 **证据板 + QUERY 角色** 生成。推荐与 API Server **同进程** 部署，经 **Query Tool Service** 继承 KB RBAC 与审计，并在统一 embedding/rerank/LLM 前提下安全使用多库能力。

---

## 18. 修订记录

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0 | 2026-07-01 | 初稿 |
| 2.0 | 2026-07-01 | 明确 Agent 为产品模式；AGENT 角色与 `.env`；内置/用户工作流提示词；串行执行；`candidate_kb_ids` 语义；统一 embedding/rerank 前提；证据合成与 Query Tool Service |
| 3.0 | 2026-07-01 | 增加 PROFILE 角色；文档级 profile → KB 级自动 profile；后台 `agent_profile` job；查看/人工覆盖/手动刷新 API |
| 3.1 | 2026-07-02 | 实现修订：真流式事件；规划 JSON 重试与 `agent_session_failed` 审计；P0 优先截断；单步失败容忍；证据合成取消二次 rerank（轮次轮转合并）；profile 脏标记独立键 + job 去重 + 链式刷新；全量文档 profile 聚合（每次生成上限 24、聚合采样 128 + 频次统计）；LLM 输出截断代替拒绝 |
| 3.2 | 2026-07-02 | 大库聚合分层摘要：KB 级聚合按规模自动切换 direct（≤128 全量）/ sampled（回填期全库等距抽样）/ grouped（128 篇/组缓存组摘要 map-reduce，追加式入库稳态 O(1) 组调用），消除 128 篇以上知识库的新旧偏差 |
| 3.3 | 2026-07-02 | 新增 `workflow` 请求参数：`plan`（既有一次性规划，默认）/ `staged`（阶段化配比推荐工作流，设计与实现见 `AgentStagedRecommendation-zh.md`）；两种工作流共用检索空结果自动换 mode 重试（`retried_mode` 上报） |
| 3.4 | 2026-07-03 | Profile 健壮性与可纠偏：单篇文档 profile 失败容忍（`document_profiles_failed` / `failed_documents` 上报，全部失败且无缓存才判失败）；`agent_domains` / `agent_sample_questions` / `agent_negative_scope` 支持人工覆盖 |
