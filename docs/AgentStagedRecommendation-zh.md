# Agent 阶段化配比推荐工作流（staged）— 设计与实现方案

> 文档版本：2026-07-02（v1）
> 适用范围：在既有 **Agent 查询模式**（`docs/AgentQueryMode-zh.md`，一次性规划 `workflow="plan"`）之上，新增面向 **配比/配方推荐** 场景的 **阶段化证据链工作流**（`workflow="staged"`）。
> 目标读者：后端开发、前端集成、知识库运营。
> 关联文档：`docs/AgentQueryMode-zh.md`、`docs/API接口.md`。

---

## 1. 背景：目标场景与现状缺陷

### 1.1 目标场景

用户按 **证据来源类型** 组织多个知识库，例如：

| 知识库 | 内容 | 证据角色 |
|--------|------|----------|
| 配方知识库 | 历史配方 / 配比案例（组分与用量） | `reference_formula`（参考配方骨架） |
| 实验数据知识库 | 实验与测试数据（性能实测结果） | `experimental`（实验验证） |
| 论文知识库 | 文献、机理研究 | `literature`（机理依据） |
| 胎侧知识库（等应用专项库） | 特定应用的规格、要求与经验 | `application_spec`（应用规范约束） |

> **知识库名称与数量无关**：上表仅为示例。工作流不绑定任何具体库名、数量或领域词——运行时输入是"当前用户可访问的 KB 集合"，各库的证据角色由 AGENT LLM **每会话**根据 KB Agent Profile（含 PROFILE 角色自动生成的内容画像）动态标注（`kb_roles_assigned` 事件），后续新增/改名知识库零配置。角色无法判断的库标为 `other`，各阶段按"目标角色 → 相邻角色 → 全部库"逐级回退，只影响检索聚焦度，不影响正确性；回退到多库时受每步 KB 数上限保护（§3.5）。

典型问题模式：**“帮我推荐一种在某环境（如高寒地区）下使用的 XX 配比”**。合格的回答必须给出配比表，且**每个组分、每个用量、每个性能结论都能追溯到知识库证据**；查不到的内容明确声明，不允许来自模型参数记忆的“常识配方”。

### 1.2 一次性规划（`workflow="plan"`）对该场景的结构性缺陷

1. **骨架来自参数记忆**：规划时模型凭记忆猜组分再去检索，证据链起点断裂——推荐本身不是从知识库里长出来的。
2. **无法表达依赖步**：规划契约要求子 query 完整自洽，而“先找到参考配方，再针对其组分和目标指标逐项验证”天然是依赖链。
3. **对检索结果零反应**：空结果只能变成终答里的缺口声明，不能换库换 mode 重查。
4. **充分性无判据**：证据是否覆盖了决定可用性的性能指标，只能靠终答模型自觉。

### 1.3 设计选型结论（承接 AgentQueryMode 设计 §8 预留的 evaluate 机制）

不采用“模型自由循环直到自认为证据充足”（成本不可预测、终止判断不可靠、P0 保证弱化），而采用：

> **固定阶段模板（服务端驱动）+ 阶段间结构化产物传递 + 清单化充分性判定 + 有界自适应（空结果重试 + 一轮补查）**。

模型每次只输出小的、经 schema 校验的 JSON 决策；循环控制权、KB 越权校验（fail-closed）、预算硬上限全部在服务端。

---

## 2. 工作流总览（六阶段）

```text
用户提问（workflow="staged"，可选 candidate_kb_ids）
    ▼
[门禁] 鉴权 → can_use_agent_query → effective_kbs（与 plan 模式一致）
    ▼
S0 需求解析（AGENT LLM，无检索）
    问题（含截断后的 conversation_history） → {application 应用对象, conditions 环境/工况,
            target_properties 目标性能指标(P0/P1/P2, ≤8), constraints 其他约束,
            assumptions 默认假设}；缺关键信息 → 按领域常识补默认假设继续；
            完全无法判断应用领域时置 clarification（降级执行，不终止会话）
    ▼
S1 骨架召回（AGENT LLM 规划 ≤3 步 → 串行检索 → AGENT LLM 提取）
    a) 模型为每个 effective KB 标注证据角色 kb_roles（reference_formula/experimental/
       literature/application_spec/other）—— 每会话动态标注，无需在 KB 上固化配置
    b) 检索最接近的参考配方/配比案例（优先 reference_formula、application_spec 角色库）
    c) 提取骨架：components[{material, ratio, function, source_refs}] + open_questions
       —— source_refs 必须引用已检回证据的 A 编号，服务端校验，引用不存在的组分整条丢弃并计数上报
    ▼
S2 要素证据（服务端模板实例化查询，无 LLM 规划调用，仅检索）
    open_questions 优先 + 骨架组分逐个补充（“{application}在{conditions}条件下{material}的用量与机理”），
    目标库 = literature/experimental/application_spec 角色库；步数 ≤8 且为 S3 预留预算
    ▼
S3 指标验证（服务端逐指标实例化查询 → 检索 → AGENT LLM 裁决）
    每个 target_property 一步（P0 优先，≤8 步）：“{application}在{conditions}条件下的{指标}实验数据与测试结果”，
    目标库 = experimental 角色库（缺省回退全部）
    裁决：supported / partial / unsupported / no_data，evidence_refs 必须引用 A 编号；
    fail-closed：模型漏答的指标记 no_data；supported/partial 却无有效引用 → 降级 no_data
    ▼
S4 缺口补查（条件触发，至多一轮，AGENT LLM 规划 ≤4 步）
    触发条件：存在 no_data/unsupported 指标，或存在空结果步骤，且预算未用尽；
    补查步可换库、换 mode、改写查询；执行后仅对缺口指标重新裁决并合并
    ▼
S5 终答合成（QUERY LLM，使用专用无编号证据合成模板 + 推荐输出结构约束）
    输出结构强制：① 推荐配比表（组分/配比含单位/作用，不含引用编号列；数值只能来自证据）
                  ② 目标性能指标逐项核对（指标 + 结论 + 简短证据说明，不含引用编号列）
                  ③ 未覆盖点与风险（no_data/unsupported 指标、被裁剪的检索、建议补做的验证实验）
    ▼
返回 answer（正文不显示证据/分块引用编号）、references（结构化单独返回，含 reference_id、
stage/evidence_role、file_path/kb_id 等）、steps_summary、metadata（含指标裁决、预算用量、裁剪记录）
```

### 2.1 示例走查（“推荐一种在高寒地区使用的胎侧胶料配比”）

- S0：application=胎侧胶料；conditions=[高寒/低温环境]；target_properties=[低温屈挚性(P0)、脆性温度(P0)、耐臭氧老化(P1)、撕裂强度(P1)…]。
- S1：kb_roles={配方库: reference_formula, 实验数据库: experimental, 论文库: literature, 胎侧库: application_spec}；从配方库+胎侧库检回相近案例，提取骨架（如 NR/BR 并用比、炭黑品种与份数、防护体系），每项带 A 编号引用。
- S2：对 open_questions（如“低温环境下 BR 并用比例对胎侧屈挠性能的影响”）与关键组分逐项检索论文库/实验数据库。
- S3：对每个指标检索实验数据库并裁决；如“脆性温度”查不到实测数据 → no_data。
- S4：针对 no_data 指标补查一轮（换库/改写）。
- S5：输出配比表 + 指标核对表 + “脆性温度无实测数据，采纳前需补低温脆性实验”之类的明确声明。

---

## 3. 关键机制

### 3.1 证据板与 A 编号先行

- 阶段化会话维护统一证据板：chunk 检回即去重（kb_id+chunk_id，退化为内容哈希）并**立刻分配稳定 A 编号**；后续所有提取/裁决调用引用 A 编号。
- 每条证据带 `stage`（skeleton/factor_evidence/validation/gap_repair）与 `evidence_role`（reference_formula/mechanism/validation/repair）标签；这些内部链接用于服务端核对证据链，最终回答正文不显示 A 编号。
- 终答上下文按 token 预算截断时 **被引用的证据优先保留**（骨架 source_refs + 裁决 evidence_refs 排在最前），落选的只可能是未被任何结构化结论引用的证据。
- 与 plan 模式的差别：A 编号在检回时分配且不重排（编号是 ID 不是序号），仅用于内部骨架/裁决记账和结构化 `references`；终答模型只看到证据正文，不看到这些编号。

### 3.2 提取幻觉防线（fail-closed 引用校验）

- 骨架组分：`source_refs` 全部无效 → 该组分丢弃，`dropped_components` 计数进事件与 metadata。
- 指标裁决：`supported/partial` 且无有效引用 → 降级 `no_data` 并注明；模型漏答的指标补记 `no_data`；无效 verdict 枚举值一律按 `no_data` 处理。
- 裁决/提取阶段 LLM JSON 连续失败（3 次重试后）：**不失败整个会话**——骨架按“未提取到”处理、裁决按全部 `no_data` 处理，缺口如实进入终答声明。仅 S0 需求解析与 S1 骨架规划失败会导致会话失败（502，`agent_requirement_invalid` / `agent_skeleton_plan_invalid`），因为没有它们后续阶段无从谈起。

### 3.3 充分性判定清单化

“证据是否充足”不由模型自评，而由数据结构机械判定：每个 target_property 必须有裁决记录；`no_data/unsupported` 即缺口 → 触发 S4 补查 → 补查后仍缺 → 终答强制声明。模型只判断单条证据的相关性，不判断整体充分性。

### 3.4 有界自适应

| 机制 | 触发 | 上限 |
|------|------|------|
| 空结果换 mode 重试 | 某步检索成功但 0 chunk | 每步 1 次（mix→naive、naive→hybrid、hybrid→mix、local/global→hybrid）；重试自身失败则保留原空结果，不失败该步 |
| 缺口补查（S4） | 存在 no_data/unsupported 指标或空结果步骤 | 每会话 1 轮、≤4 步 |

空结果重试同样应用于 `workflow="plan"` 模式（该模式此前对空结果无任何自救）。

### 3.5 预算与治理

- 会话检索总步数硬上限：`AGENT_STAGED_MAX_RETRIEVALS`（默认 24）；各阶段代码内上限：S1≤3、S2≤8、S3≤8、S4≤4。
- **每步 KB 数上限**：`AGENT_STAGED_MAX_KBS_PER_STEP`（默认 4）。知识库总数未知/较多时，模型选库超限或角色回退到"全部库"会导致单步串行检索所有库；超限时按各库人工 `agent_priority` 降序择优（同分保持原顺序），裁剪必须记入 `metadata.clipped`。规划提示词同步告知该上限。
- **验证预留**：S2 的实际步数 = min(8, 剩余预算 − 待验证指标数)，保证 P0 指标验证不被机理检索挤占。
- 预算不足被跳过的工作**必须上报**（`metadata.clipped` + 终答未覆盖声明），禁止静默截断。
- LLM 调用次数固定上界：S0、S1 规划、S1 提取、S3 裁决、S4 规划、S4 复裁 ≤6 次小 JSON 调用 + 1 次终答合成。
- 权限：完全复用 plan 模式门禁（`can_use_agent_query`、effective_kbs 交集、每步 kb_ids ⊆ effective 校验）。骨架规划越权 → 403 会话失败（与 plan 模式一致）；补查规划越权 → 丢弃该步并记录（已积累的证据不作废，越权步骤同样从未执行）。
- 审计：复用 `agent_session_started` / `agent_retrieve_round` / `agent_query_completed` / `agent_session_failed`，metadata 增加 `workflow`、`stage`、`retried_mode`。

---

## 4. API 变化

### 4.1 请求

`POST /agent/query`、`POST /agent/query/stream` 请求体新增：

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `workflow` | `"plan"` \| `"staged"` | `"plan"` | `plan`=既有一次性规划；`staged`=阶段化配比推荐工作流。`staged` 下 `max_rounds` 不生效（预算由 `AGENT_STAGED_MAX_RETRIEVALS` 与阶段上限控制） |

### 4.2 NDJSON 事件（staged 新增/变化）

| 事件 | 说明 |
|------|------|
| `stage_started` | `{stage}` 阶段开始（requirement/skeleton/factor_evidence/validation/gap_repair） |
| `requirement_parsed` | 结构化需求（application/conditions/target_properties/constraints） |
| `kb_roles_assigned` | 本会话各 KB 的证据角色标注 |
| `skeleton_extracted` | 骨架组分（含 source_refs）、open_questions、`dropped_components` |
| `validation_verdicts` | 指标裁决列表；补查后携带 `after_repair: true` 再发一次 |
| `round_started` / `round_result` | 复用 plan 模式，新增 `stage` 字段；空结果重试成功时 `round_result` 带 `retried_mode` |
| 其余 | `session_started`（metadata 增加 `workflow`）、`references`（每条含 `stage`/`evidence_role`）、`response`、`clarification_downgraded`（澄清降级，不终止会话）、`done`、`error` 复用 |

### 4.3 响应 metadata（staged）

`workflow`、`kb_roles`、`requirement`、`property_verdicts`（逐指标裁决）、`skeleton_component_count` / `dropped_component_count`、`retrieval_budget {max, used}`、`clipped`（被裁剪工作清单）、`round_count` / `failed_round_count`。

### 4.4 错误码

| 错误码 | 状态 | 场景 |
|--------|------|------|
| `agent_requirement_invalid` | 502 | S0 需求解析 JSON 连续失败 |
| `agent_skeleton_plan_invalid` | 502 | S1 骨架规划 JSON 连续失败 |
| `agent_all_steps_failed` | 502 | 所有检索步骤失败（复用） |

### 4.5 配置

```bash
# staged 工作流单会话检索步数硬上限（S1≤3 / S2≤8 / S3≤8 / S4≤4 之和的护栏）
AGENT_STAGED_MAX_RETRIEVALS=24
# staged 工作流每个检索步最多同时查询的 KB 数（超限按 agent_priority 择优并上报）
AGENT_STAGED_MAX_KBS_PER_STEP=4
```

其余复用：`LIGHTRAG_AGENT_QUERY_ENABLED`、`AGENT_LLM_*`（编排小 JSON）、`QUERY_LLM_*`（终答）、`PROFILE_LLM_*`（KB 画像，供选库与角色标注）。

---

## 5. KB 内容侧建议（用户运营侧，影响效果上限）

1. **画像写清角色语义**：各库 `agent_description`/`agent_tags` 明示”参考配方案例””实验测试数据””文献机理””应用规格”，kb_roles 标注准确率直接受益（自动 profile 已就位，人工字段可加强）。自动生成的 `domains` / `sample_questions` / `negative_scope` 若有偏差，也可通过 `PUT /kbs/{kb_id}/agent-profile` 人工覆盖（尤其 `negative_scope` 错误会系统性误导选库）。
2. **实验数据文档结构**：一次实验/一组对比一文档，明确写出配方变量与实测指标数值，S3 才能检到可裁决的证据。
3. **配方案例文档结构**：一配方一文档，组分与用量成表，S1 骨架提取质量的决定因素。
4. **别名/牌号对照**：原料学名、俗称、牌号对照表入库（任意库均可），缓解跨库检索命名不一致。
5. **分块校验**：含配比表/数据表的文档确认表格不被分块截断（chunker 配置按库核对）。

---

## 6. 实施与测试

### 6.1 本次实现范围（随本文档落地）

| # | 内容 | 状态 |
|---|------|------|
| 1 | `workflow` 请求参数与 plan/staged 分流 | ✅ |
| 2 | 空结果换 mode 重试（plan 与 staged 共用） | ✅ |
| 3 | S0–S5 阶段化流水线（本文档 §2–§3 全部机制） | ✅ |
| 4 | 事件流、metadata、审计扩展 | ✅ |
| 5 | `AGENT_STAGED_MAX_RETRIEVALS` 配置 | ✅ |
| 6 | 单元测试（见 §6.2）与 `docs/API接口.md` 更新 | ✅ |

### 6.2 测试矩阵（tests/api/routes/test_agent_staged_workflow.py）

- 全流程 happy path：事件顺序、kb_roles、骨架引用、裁决映射、references 的 stage/evidence_role、终答合成。
- S0 澄清降级（继续检索并在终答附澄清问题，application 缺失时用原始问题回填）；S0 JSON 连续失败（含 clarification 无问题文本）→ 502 `agent_requirement_invalid`。
- 骨架组分无效引用被丢弃并计数；骨架规划选越权 KB → 403 + `agent_session_failed`。
- 裁决 fail-closed：漏答指标 → no_data；supported 无有效引用 → 降级 no_data。
- 缺口补查：no_data 触发补查、补查后裁决更新、`after_repair` 事件。
- 预算硬上限：检索步数不超过上限、被跳过的工作进入 `clipped`。
- 多库规模保护：单步 KB 数超过 `AGENT_STAGED_MAX_KBS_PER_STEP` 时按 `agent_priority` 择优裁剪并上报（模型选库超限与角色回退到全部库两种路径）。
- 空结果重试：plan 模式下 0 chunk → fallback mode 重试成功，`retried_mode` 上报。
- plan 模式回归：默认 `workflow="plan"` 行为不变（既有用例全部通过）。

### 6.3 后续迭代（不在本次范围）

- 金标集评测：20~50 个真实配比需求，指标=指标裁决覆盖率 / 配比表逐行引用率 / 缺口声明准确率。
- 阶段预算 env 化、按用户/租户的工作流模板定制、S2 要素证据的结构化提取（当前为检索+引用，不做逐要素小结）。
- 多轮会话续问（沿用 clarification 后携带上下文重问的既有产品约定）。

---

## 7. 修订记录

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0 | 2026-07-02 | 初稿：阶段化配比推荐工作流设计 + 首期实现范围 |
| 1.1 | 2026-07-03 | 明确知识库名称/数量无关（角色动态标注 + 逐级回退）；新增每步 KB 数上限 `AGENT_STAGED_MAX_KBS_PER_STEP`（按 agent_priority 择优，裁剪上报） |
| 1.2 | 2026-08-17 | staged 终答正文不再输出证据编号；来源改由结构化 `references` 单独返回 |
