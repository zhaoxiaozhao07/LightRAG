# 双语查询（Bilingual Query）设计方案

> 状态：已实现（v1）。测试结论见文末[第 10 节](#十测试结论)。
>
> 关联文档：`docs/API接口.md`（接口字段与开关）、`docs/AgentQueryMode-zh.md`（Agent 查询模式）。

## 一、背景与问题

知识库中同时存在中文与英文文档；用户以中文提问、期望中文回答。实际使用中出现"中文问题查不到英文文档证据"的情况。原因分三层：

| 层面 | 问题 | 本方案是否解决 |
|---|---|---|
| ① 向量路径 | 中文 query 的 embedding 与英文 chunk 的 embedding 存在跨语言语义 gap（多语言模型也有可测差距，单语模型基本失效） | ✅ 查询侧双路 |
| ② 图谱路径 | 从中文 query 提取的关键词是中文；英文文档抽取出的实体/关系名是英文，实体/关系向量检索命中率低 | ✅ 双语关键词双路 |
| ③ 图谱结构 | 同一概念在图谱中是两个孤立节点（如"硫化促进剂"与 "vulcanization accelerator"互不相连），跨语言图扩展走不通 | ❌ 属入库侧治理，规划为二期 |

## 二、方案总览（四层）

```
用户中文问题
    │
    ├─ 第0层 地基：多语言 embedding（bge-m3）+ 多语言 reranker（bge-reranker-v2-m3）
    │
    ├─ 第1层 预处理：一次 LLM 调用 → {query_zh, query_en, 双语 hl/ll 关键词}
    │         （替代核心层原有的关键词提取调用 → kg 模式净增 LLM 调用数为 0）
    │         （缓存 + 超时 + fail-open：失败即退回现状单路查询）
    │
    ├─ 第2层 双路检索：主路=原句+同语关键词  副路=译句+另一语关键词
    │         两路各自 aquery_data → chunk 去重合并 → 统一 rerank（以原句打分）
    │         → 按 chunk_top_k 与 token 预算截断 → 引用重编号
    │
    └─ 第3层 合成：单次 LLM，naive_rag_response 模板 + 回答语言指令
              （以原句语言回答；引用他语证据时关键术语括注原文）
```

### 2.1 第 0 层：多语言模型地基（前提）

- embedding 必须是多语言模型：vLLM 离线部署默认 `BAAI/bge-m3`（`env.example` 中 `VLLM_EMBED_MODEL`）。存量 KB 若使用中文单语模型，需换模型并全量重建向量（由 `index_hash` / `requires_vector_rebuild` 机制跟踪）。
- reranker 建议 `BAAI/bge-reranker-v2-m3`（多语言）：它是双路合并后统一排序的"裁判"，中文 query 给英文 chunk 打分依赖其跨语言能力。无 reranker 时合并退化为拼接去重 + 截断（见 4.3）。

### 2.2 第 1 层：查询预处理（`prepare_bilingual_queries`）

一次 LLM 调用（**`bilingual` 角色模型**，默认逐字段继承 `query` 角色，见 2.2.1）输出严格 JSON：

```json
{
  "query_zh": "中文完整问句（原句为中文时照抄原句）",
  "query_en": "English full question (verbatim if already English)",
  "hl_keywords_zh": ["宏观概念/主题"],
  "ll_keywords_zh": ["具体实体/术语"],
  "hl_keywords_en": ["high-level concepts"],
  "ll_keywords_en": ["specific entities/jargon"]
}
```

要点：

- **零额外调用（kg 模式）**：产出的关键词经 `QueryParam.hl_keywords / ll_keywords` 注入两路检索，核心层 `get_keywords_from_query` 检测到预置关键词即跳过自身的关键词提取 LLM 调用（`operate.py`），因此 local/global/hybrid/mix 模式总 LLM 调用数不变。naive 模式原本没有该调用，净增 1 次小调用。
- **主路使用原句原文**，不用 LLM 复述的同语版本（避免复述漂移）；副路使用译句。
- **缓存**：结果经 `handle_cache/save_to_cache` 写入 KB 的 `llm_response_cache`，键 `bilingual:query_preprocess:{hash}`，hash 含 LLM 身份（换模型自动失效）。
- **fail-open**：解析失败（`call_llm_json` 最多 2 次尝试后仍失败）、超时（`BILINGUAL_QUERY_TIMEOUT`，默认 12s）、译句为空或与原句相同 → 返回 `None`，调用方退回单路，绝不阻断查询。
- 请求体已显式提供 `hl_keywords/ll_keywords` 时跳过预处理（尊重调用方意图）。

#### 2.2.1 翻译模型独立角色（`BILINGUAL` LLM 角色）

预处理调用走独立的 **`bilingual` LLM 角色**（注册于 `lightrag/llm_roles.py` 的 ROLES 注册表），与 `agent`/`profile` 角色同一套机制（独立并发队列、超时、观测、热更新）：

- **默认继承 QUERY**：env 解析后按字段回填——任何未设置的 `BILINGUAL_LLM_*` 字段逐项继承 `QUERY_LLM_*` 对应字段（`lightrag/api/config.py` 的 `_backfill_bilingual_role_args`），仍未设置的字段走标准"角色 → 基础 LLM"回退。因此默认行为 = 翻译用 query 同款模型，零配置。
- **换专用翻译模型**：设置 `BILINGUAL_LLM_MODEL`（可选 `BILINGUAL_LLM_BINDING/BINDING_HOST/BINDING_API_KEY/MAX_ASYNC_LLM/LLM_TIMEOUT`）即可，只设 model 时服务地址等继承 query 角色。
- **思考默认关闭**：预处理经 `call_llm_json` 调用，固定 `enable_cot=False`；Qwen3 类模型如需在 vLLM 上硬关思考模式，可加角色级 provider options：`BILINGUAL_OPENAI_LLM_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'`。
- **KB 级覆盖**：`llm_role_config.bilingual`（字符串=model，或对象含 binding/host/api_key/provider_options 等）走既有 config 版本机制；其模型身份纳入 `query_hash`（仅影响查询，不触发重建）。
- 缓存键包含所用角色的 LLM 身份（binding/model/host），换翻译模型自动失效旧缓存。
- 服务层解析顺序：`role_llm_funcs["bilingual"]` → 回退 `["query"]`（兼容未注册该角色的嵌入式用法）。
- 注意区分两个超时：`BILINGUAL_LLM_TIMEOUT` 是角色 LLM 请求超时，`BILINGUAL_QUERY_TIMEOUT` 是整个预处理步骤的等待上限（超过即放弃双路）。

### 2.3 第 2 层：双路检索与融合

复用多库合并（`/kbs:query`）已验证的 scatter-gather 骨架，把扇出维度从 KB 换成语言变体：

- 主路：`aquery_data(原句, param + 同语关键词)`
- 副路：`aquery_data(译句, param + 另一语关键词)`（失败仅告警，保留主路结果，metadata 标记 `secondary_failed`）
- 融合：
  - **chunks**：两路拼接 → 按 `chunk_id`（缺失时按内容 hash）去重 → `process_chunks_unified`（以**原句**统一 rerank + `chunk_top_k`/token 预算截断）→ `generate_reference_list_from_chunks` 重建引用编号；
  - **entities / relationships**（仅 `/query/data`、`/retrieve` 返回）：按 `entity_name` / `(src_id, tgt_id)` 去重合并；两路各自的 `reference_id` 编号体系不可跨路复用，合并后置空（chunks 与 references 保持一致编号）。
- 每路 `top_k`/`chunk_top_k` 不减半：让 rerank 在完整合并池裁决，某一语言证据占优是正常结果。

### 2.4 第 3 层：合成（回答语言控制）

单次 LLM 合成，`naive_rag_response` 模板（与多库合并一致的 chunk-only 证据上下文），在 `user_prompt` 槽位前注入固定语言规则：

- 原句为中文 → "必须使用中文回答；引用英文证据时，关键技术术语首次出现可括注英文原文"；
- 原句为英文 → 对称的英文规则。

用户/KB 级 `user_prompt` 依旧生效（拼接在语言规则之后，优先级语义不变）。

> **权衡说明**：双语路径的合成上下文为 chunk-only（不含 kg 模式的实体/关系描述块），与 `/kbs:query` 多库合并的既有取舍一致；实体/关系数据仍完整保留在 `/query/data` 结构化输出中。若后续评估发现图谱描述缺失影响回答质量，可在二期引入"KG 上下文附录"。

## 三、启用控制（三层优先级）

```
请求体 bilingual（显式 true/false） > KB query_config.bilingual_query（off/auto/on） > 环境 BILINGUAL_QUERY_DEFAULT_MODE
                                    └────────── 全部受 BILINGUAL_QUERY_ENABLED 总开关（kill-switch）约束 ──────────┘
```

| 配置 | 位置 | 取值 | 说明 |
|---|---|---|---|
| `BILINGUAL_QUERY_ENABLED` | 环境变量 | `true/false`（默认 `false`） | 总开关；`false` 时无论其他配置如何，全部单路 |
| `BILINGUAL_QUERY_DEFAULT_MODE` | 环境变量 | `off/auto/on`（默认 `auto`） | KB 未配置时的默认模式 |
| `BILINGUAL_QUERY_TIMEOUT` | 环境变量 | 秒（默认 `12`） | 预处理步骤整体等待上限 |
| `BILINGUAL_LLM_*` | 环境变量 | 同其他 LLM 角色 | 翻译模型独立角色（见 2.2.1）；未设字段逐项继承 `QUERY_LLM_*` |
| `query_config.bilingual_query` | KB config 版本（`POST /kbs/{kb_id}/configs` + `:activate`） | `off/auto/on` | 走既有 config 版本机制；参与 `query_hash`，不触发重建 |
| `llm_role_config.bilingual` | KB config 版本 | model 字符串或对象 | KB 级翻译模型覆盖；身份参与 `query_hash` |
| `bilingual` | 查询请求体 | `true/false/null` | 显式覆盖（评测对比用）；`null` 表示跟随上层配置 |

模式语义：

- `on`：始终双路（英文原句也会生成中文副路，反向同样受益）；
- `auto`：仅当 query 含 CJK 字符时双路（对准"中文查英文文档"主诉求，纯英文查询零开销）；
- `off`：从不双路。

以下情形自动跳过双路（即使解析为 on/auto）：`mode=bypass`（无检索）、`only_need_context/only_need_prompt=true`（核心语义保持不变）、请求体显式提供 `hl/ll_keywords`。

**范围**：单库端点读 KB config；多库 `/kbs:query`、`/kbs:retrieve` 与 Agent 与 legacy 全局 `/query` 系列不读 per-KB config（与"多库检索用同一套请求级参数"的既有约定一致），由请求体 `bilingual` + 环境默认值决定。

## 四、稳妥性设计

### 4.1 降级链

```
预处理 LLM 失败/超时/输出无效  → 单路（现状行为），metadata.bilingual.reason 说明
副路检索异常                  → 只用主路，metadata.bilingual.secondary_failed=true
无 reranker                  → 合并池按"主路优先、去重拼接"次序 + 既有 token 截断（无跨语打分时保守保主路）
双路合并后 0 条               → 与现状空结果行为一致（Agent 流沿用既有空结果换 mode 重试）
```

### 4.2 观测

- 单库/多库查询响应 `metadata.bilingual`：`{enabled, mode, source_language, translated_query, primary_chunks, secondary_chunks, merged_chunks, final_chunks, secondary_failed?, reason?}`；
- 审计事件 metadata 增加 `bilingual_enabled` 与 `bilingual_translated_query_hash`（只记 hash，不记原文，与 `query_hash` 口径一致）；
- Agent `round_result` 事件/步骤摘要在双路生效时携带 `bilingual=true`。

运营指标：关注 `secondary_chunks / final_chunks` 占比——长期接近 0 说明语料无英文或英文路故障，均应被看见。

### 4.3 成本

- kg 模式 LLM 调用数不变（预处理替代关键词提取）；naive 模式 +1 次小调用；
- 向量检索次数 ×2（毫秒级）；rerank 候选池约 ×1.5~2；
- 端到端延迟预期增幅 <15%（两路 `aquery_data` 并行执行）。

## 五、各端点行为

| 端点 | 双路支持 | 开关来源 |
|---|---|---|
| `POST /kbs/{kb_id}/query` / `/query/stream` | ✅ 检索双路 + 单次合成（流式支持） | 请求体 > KB config > env |
| `POST /kbs/{kb_id}/query/data` / `/retrieve` | ✅ 检索双路 + 结构化合并 | 同上 |
| `POST /kbs:query` / `:query/stream` / `:retrieve` | ✅ 每 KB 双路后跨库合并 | 请求体 > env |
| `POST /query` / `/query/stream` / `/query/data`（legacy 全局） | ✅ 同单库行为 | 请求体 > env |
| `POST /agent/query` / `/agent/query/stream`（plan / staged） | ✅ 见第六节 | 请求体 > env |
| `mode=bypass` | 跳过（无检索） | — |

## 六、Agent 工作流双语支持

Agent 不走独立预处理调用——规划 LLM 在产出步骤时**直接生成双语检索要素**（一次规划调用顺带完成，零额外成本）：

- `AgentPlanStep` 新增可选字段：`query_en`（该步子问题的另一语言版本）、`hl_keywords_en` / `ll_keywords_en`；
- 规划 payload 增加 `bilingual_retrieval: true/false`；为 true 时 planner 系统提示词要求为每步生成上述字段（plan 工作流的总提示词、staged 的骨架规划 / 补查规划提示词均已扩展）；
- 步骤执行层 `QueryToolService.retrieve_serial` 新增 `query_alt / hl_keywords_alt / ll_keywords_alt`：存在副路时每 KB 双路检索合并（副路失败仅告警），plan 与 staged 两个工作流共用该执行器，天然同时覆盖；
- staged 的指标验证步骤：需求解析新增 `target_properties[].name_en`，双语开启时验证步骤以英文指标名构造副路查询；
- staged 的要素证据步骤：骨架提取新增 `open_questions_en`（与 `open_questions` 按序配对），配对成功的步骤双路检索；
- 空结果换 mode 重试、预算裁剪、审计等既有机制不变。

Agent 的启用判定：`AgentQueryRequest.bilingual`（显式）> 环境默认模式（`auto` 按用户问题是否含 CJK）。

## 七、实现位置（上游合并友好）

全部改动位于 fork 自有文件，未触碰 `lightrag/operate.py`、`lightrag/prompt.py`、`lightrag/lightrag.py`：

| 文件 | 改动 |
|---|---|
| `lightrag/api/bilingual_query_service.py` | **新增**：预处理、语言检测、开关解析、双路检索融合、合成、翻译角色解析 |
| `lightrag/llm_roles.py` | ROLES 注册表新增 `bilingual` 角色（注册表驱动：env 解析/队列/观测/KB 覆盖自动生效） |
| `lightrag/api/config.py` | 3 个环境变量解析 + `_backfill_bilingual_role_args`（BILINGUAL 未设字段逐项继承 QUERY） |
| `lightrag/api/config_version_service.py` | `query_config.bilingual_query` 白名单 + 校验 + 运行时读取；`bilingual` 角色纳入 `_QUERY_AFFECTING_ROLES` |
| `lightrag/api/routers/kb_query_routes.py` | 单库/多库端点接入 + `bilingual` 请求字段 + metadata |
| `lightrag/api/routers/query_routes.py` | legacy 全局端点接入 |
| `lightrag/api/query_tool_service.py` | `retrieve_serial` 副路参数 |
| `lightrag/api/agent_query_service.py` | `AgentPlanStep` 双语字段、规划 prompt、步骤执行透传 |
| `lightrag/api/agent_staged_service.py` | 需求/骨架/补查 schema 与 prompt、验证/要素步骤副路 |

核心依赖均为上游稳定公开接口：`QueryParam.hl_keywords/ll_keywords` 预置跳过关键词提取、`aquery_data` 结构化检索、`process_chunks_unified` / `generate_reference_list_from_chunks`。`llm_roles.py` 的 ROLES 注册表本身是 fork 扩展模块（agent/profile 角色同源），模块文档明确"新增角色 = 注册表加一行"。

## 八、灰度与验收

1. **金标集**：30~50 条中文问题，其中一半答案证据只在英文文档（另设中文-only 与混合对照组），标注目标 chunk；
2. **指标**：目标 chunk 进 top-k 召回率（双语开/关对比）、`metadata.bilingual.secondary_chunks` 贡献占比、端到端延迟 P95；
3. **灰度**：`BILINGUAL_QUERY_ENABLED=true` + 测试 KB `query_config.bilingual_query="on"` 跑金标集 → 达标后生产 KB 逐个设 `auto` → 观察一周审计指标后放开。

## 九、二期方向（本期明确不做）

- **图谱跨语言实体对齐**：同概念双语节点合并或补 alias 边；待 v1 上线后按 `secondary_chunks` 占比与坏例决定；
- **chunk 影子翻译**（入库时翻译存双份）：索引体积翻倍、翻译质量引入静默错误，双路检索已覆盖主要收益；
- **KG 上下文附录**：双语合成上下文引入实体/关系描述块（需要独立评估 prompt 形态）。

## 十、测试结论

**测试于 2026-07-07 全部通过**（Windows 11 / 项目 `.venv`，`PYTHONUTF8=1`）：

| 套件 | 结果 |
|---|---|
| `tests/api/routes/test_bilingual_query.py`（本功能，39 个用例） | **39 passed** |
| 受影响回归：`test_kb_query_routes.py` + `test_query_routes_stream.py` + `test_agent_query_routes.py` + `test_agent_staged_workflow.py` + `test_kb_config_routes.py` + `test_llm_role_runtime.py` + `test_llm_cache_identity.py` | **全部通过** |
| 完整 `tests/api` 套件 | **717 passed, 34 skipped**（跳过项为环境相关既有跳过，与本功能无关） |
| `tests/llm` 套件 | 与改动前基线**逐项一致**（存量失败均为缺 qdrant/voyageai 可选依赖、bedrock 凭据等环境问题） |
| `ruff check`（全部改动文件） | **All checks passed** |

新增用例覆盖：

- **服务层**：CJK 检测与语言判定；`off/auto/on` × 请求/KB config/env 三层解析矩阵（含总开关 kill-switch、非法 KB 值回退）；`bilingual_applies` 守卫（bypass、only_need_context、调用方自带关键词、auto 纯英文跳过）；预处理计划的语言路由（主路恒用原句原文、译句为空/与原句相同判不可用）；chunk 合并去重与 `retrieval_path` 标记；副路参数关键词兜底（kg 模式空关键词时以译句种子化 ll，杜绝隐藏 LLM 调用；naive 不种子化）；预处理失败与超时 fail-open；副路检索异常容忍。
- **KB config**：`query_config.bilingual_query` 大小写规范化、非法值创建期 `400`、白名单接受；`bilingual_query` 不泄漏进 QueryParam 默认值、`bilingual_mode_from_rag` 正确读取。
- **单库端点**：双路合并合成（两种语言证据都进入答案与引用、关键词正确种子化到两路）、auto 模式中文/英文分流、总开关关闭、请求级 `bilingual=false` 覆盖 KB `on`、预处理失败回退单路（含 `reason`）、显式关键词跳过、流式首行 `metadata.bilingual` + 双语引用、`/query/data` 实体/关系合并去重与引用重编号、bypass 跳过。
- **多库端点**：per-KB 双路（每 KB 恰好 2 次检索）、`per_kb_secondary_chunks` 统计、无标志时按 env `auto` 分流、`:retrieve` 的 metadata。
- **legacy 全局端点**：`bilingual=true` 双路、默认英文单路。
- **Agent**：`AgentPlanStep` 双语字段解析与向后兼容；`agent_bilingual_enabled` 矩阵；`retrieve_serial` 副路合并、副路失败容忍、无效 `query_alt`（与主问相同）忽略；staged `_factor_queries` 等长配对/长度不匹配禁用配对/关闭时恒空；`TargetProperty.name_alt`。
- **翻译模型独立角色**：`bilingual` 角色已注册（env 前缀 `BILINGUAL`）；服务层优先用 `bilingual` 角色、缺失时回退 `query`；预处理确实经由专用角色（query 角色断言不接翻译调用）；`_backfill_bilingual_role_args` 逐字段继承语义（显式值优先、未设继承 QUERY、QUERY 也未设则保持 None 走基础回退）；KB `llm_role_config.bilingual` 接受且其身份变更改变 `query_hash`（不在 `_INDEX_AFFECTING_ROLES`）。

**遗留验证项（需真实模型环境，建议按第八节金标集流程执行）**：跨语言召回率提升幅度、`secondary_chunks` 实际贡献占比、端到端延迟增幅、qwen3-rerank 对中文 query × 英文 chunk 的打分质量。
