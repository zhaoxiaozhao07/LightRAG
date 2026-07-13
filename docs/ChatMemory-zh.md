# 用户项目级对话记忆（Chat Memory / graphiti）设计文档

> 文档版本：2026-07-11
> 状态：已实现（全量：写入 + 检索 + 服务端自动注入 + 补偿/撤销/精排）
> 关联文档：[`docs/API接口.md`](API接口.md) §10.5 / §10.5.1 / §8；对话管理设计见 commit 8480ecde
> 上游库：[getzep/graphiti](https://github.com/getzep/graphiti)（graphiti-core 0.29.x，Apache-2.0）

---

## 1. 目标与非目标

### 1.1 目标

在企业模式"用户 > 项目 > 会话 > 消息"的对话管理（`/chat`，§10.5）之上，给**每个用户的每个项目**引入一张独立的长期记忆图谱：

- 用户在项目内的历史问答被异步提炼为**时序事实图谱**（实体 + 事实边 + bi-temporal 时间戳）；
- 在同一项目**新建会话**提问时，检索出以前会话沉淀的相关事实（需求、约束、结论、偏好）注入本次问答的上下文——"越用越好用"；
- 新旧事实矛盾时自动把旧事实标记失效（`invalid_at`），保留演化历史而非覆盖。

### 1.2 能力范围（全部已实现）

- **写入全自动**：每轮问答落库即异步提炼，无需手动触发；`immediate`（默认）或 `debounced`（按会话缓冲合并）两种模式。
- **服务端自动注入**：`/kbs/{kb_id}/query`(+stream)、`/kbs:query`(+stream)、`/agent/query`(+stream) 请求体带 `memory: {"project_id": ...}` 即可，服务端校验归属→检索→拼进 `user_prompt`。**前端不再需要调 `memory:search` 再拼接**（该独立端点仍保留，供需要显式控制的场景）。
- **幂等 + 补偿**：per-session `seq` 水位去重（`enterprise_chat_memory_episodes` 映射表），启动扫描补摄取崩溃丢失的工作。
- **消息/会话级撤销**：删除消息用 `remove_episode` 移除对应 episode 并重摄取幸存消息；删除会话移除该会话全部 episode；删除项目/用户按 group 清空。
- **精排**：`MEMORY_RERANK_ENABLED=true` 时用部署 reranker（qwen3-rerank）走 cross-encoder 检索配方，否则 RRF。
- `/agent/query` 同时补齐了 `conversation_history` 通道（此前完全没有多轮历史输入）。

### 1.3 仍未做（后续可选）

- 跨项目 / 跨用户共享记忆、graphiti 社区摘要（`build_communities`）。
- 记忆写入走 durable JobWorker（当前用 fire-and-forget + 启动补偿，已足够 best-effort 语义）。

## 2. 选型结论（可行性分析摘要）

| 维度 | 结论 |
|---|---|
| 部署形态 | **库级嵌入**（LightRAG API 进程内 `import graphiti_core`）。graphiti 自带 server/ 不用：embedder 配置未接线、内存队列无持久化、每请求新建 driver、无鉴权 |
| 图存储 | 复用生产 Neo4j 实例（**server 需 ≥ 5.26**，用到原生 `vector.similarity.cosine()` 与全文索引；不需要 APOC/GDS）。graphiti 数据用固定 label（`Entity/Episodic/Community`）+ `group_id` 属性隔离，与 LightRAG 的 workspace-label 图谱同库共存；可用 `MEMORY_NEO4J_DATABASE` 指向独立 database |
| 记忆分区 | graphiti `group_id` = `{user_id}--{project_id}`（两者均为 `usr_<hex>` / `proj_<hex>`，满足 graphiti 校验 `^[a-zA-Z0-9_-]+$`） |
| LLM | `OpenAIGenericClient`（面向 vLLM 等 OpenAI-compatible `/chat/completions`；默认 `json_schema` 约束解码）；与部署共用同一 qwen 模型，`enable_thinking` 通过 extra_body 关闭 |
| Embedding | `OpenAIEmbedder(base_url=...)` 指向部署共用 embedding 服务；**必须显式设置维度**（graphiti 会把返回向量截断到 `embedding_dim`） |
| Reranker | 默认 `graphiti.search()` 走 `EDGE_HYBRID_SEARCH_RRF`（BM25 + cosine + RRF），零 reranker 成本；`MEMORY_RERANK_ENABLED=true` 时用部署 reranker 适配器（`_RerankFnCrossEncoder`）走 `EDGE_HYBRID_SEARCH_CROSS_ENCODER` 配方精排。`Graphiti()` 缺省会构造 OpenAI reranker（logit_bias token id 只对 OpenAI 词表有意义），因此未启用精排时传入 **passthrough CrossEncoderClient** 兜底 |
| 依赖 | graphiti-core 0.29.x 与当前锁定版本（neo4j 6.2 / openai 2.36 / pydantic 2.13 / tenacity 9.1）零冲突；新增传递依赖 `posthog`（遥测，用 `GRAPHITI_TELEMETRY_ENABLED=false` 关闭，服务代码内亦 `setdefault` 兜底） |

## 3. 架构

```text
前端                         LightRAG API（企业模式）                    外部服务
────                        ───────────────────────────                ────────
问答: POST /kbs/{kb}/query ──────────────────────────────────────────▶ LLM/Embedding/Rerank
  │                                                                       ▲
  │ 拿到回答后落库                                                          │
  ├─ POST /chat/.../messages ──▶ ChatConversationService.append ─┐        │
  │                              (metadata store 持久化)          │        │
  │                                     fire-and-forget           ▼        │
  │                              ChatMemoryService.schedule_ingest         │
  │                                per-group 串行 + 全局并发上限            │
  │                                graphiti.add_episode ──────────────────┤ MEMORY_LLM（抽取/去重/失效）
  │                                     │                                  │ MEMORY_EMBEDDING
  │                                     ▼                                  │
  │                                Neo4j（Entity/Episodic, group_id 隔离）  │
  │                                                                        │
  └─ 新会话提问前:                                                          │
     POST /chat/projects/{id}/memory:search ──▶ graphiti.search ───────────┘
        （强制 group_ids=[本人+本项目]，RRF 混合检索）
        facts 由前端拼进 user_prompt / conversation_history
```

### 3.1 新组件

| 组件 | 位置 | 职责 |
|---|---|---|
| `ChatMemoryConfig` | `lightrag/api/chat_memory_service.py` | 从 `global_args` 解析 + 逐项回退（MEMORY_* → QUERY_*/EMBEDDING_*/NEO4J_* → 基础 LLM_*） |
| `ChatMemoryService` | 同上 | graphiti 懒加载与生命周期、group_id 构建/校验、摄取队列、检索、清理、审计 |
| `get_enterprise_chat_memory_service` | `lightrag/api/enterprise_auth.py` | DI helper，从 `app.state` 取服务；未启用时返回 `None`（宽松取用，路由自行判断 503/跳过） |
| `POST /chat/projects/{project_id}/memory:search` | `lightrag/api/routers/chat_routes.py` | 项目记忆检索端点 |

### 3.2 服务生命周期

- 构建：`lightrag_server.py` 中仅当 `enterprise_enabled and LIGHTRAG_CHAT_MEMORY_ENABLED=true` 时创建，挂 `app.state.enterprise_chat_memory_service`。
- 启动：lifespan 内 `await service.initialize()` —— **fail-soft**：graphiti 导入失败 / Neo4j 不可达只记 ERROR 日志，服务标记不可用，不阻塞主服务启动；首次使用时（摄取/检索）持锁**懒重试**初始化，Neo4j 恢复后自动可用。
- 初始化动作：构造 `Neo4jDriver(uri, user, password, database)` + `OpenAIGenericClient` + `OpenAIEmbedder` + passthrough cross-encoder → `Graphiti(graph_driver=..., max_coroutines=MEMORY_MAX_COROUTINES)` → `build_indices_and_constraints()`（全部 `IF NOT EXISTS`，幂等；**绝不**使用 `delete_existing=True`）。
- 关停：lifespan finally 中 `await service.finalize()` —— 等待在途摄取任务短暂收尾（有超时），关闭 graphiti driver。

## 4. 数据与隔离

### 4.1 group_id 约定

```text
group_id = f"{user_id}--{project_id}"      # 例: usr_1f0e...--proj_1a2b3c4d5e6f
```

- 服务端在**每次**摄取/检索/清理前用 `^[a-zA-Z0-9_-]+$` 校验拼接结果，不合法直接拒绝（防御性；现有 ID 生成器只产生 `[a-z0-9_]`）。
- **检索必须显式传 `group_ids=[group_id]`**：graphiti 的 `search(group_ids=None)` 语义是"检索全库"，服务封装层永远不允许 None 穿透——这是本设计最重要的安全不变量。
- 清理只允许**显式非空 group 列表**：`clear_data(driver, group_ids)` 在 `group_ids=None` 时会清空整个 database（与 LightRAG KB 图谱同库时是灾难），封装层用断言拦死。

### 4.2 与 LightRAG 图谱共库

- graphiti 节点/边 label：`Entity` / `Episodic` / `Community` / `RELATES_TO` / `MENTIONS` 等；LightRAG Neo4JStorage 用 `base` + workspace 派生 label（`kb_*`）。两套查询都按各自 label 收敛，互不可见。
- graphiti 建 ~28 个 range index + 4 个全文索引（名字如 `node_name_and_summary`），全 `IF NOT EXISTS`，不影响 LightRAG 索引。
- 禁令（代码层不暴露、运维层不得手工调用）：`build_indices_and_constraints(delete_existing=True)`（会 DROP **整库所有**索引）、`clear_data(driver, None)`（清空整库）。
- 需要物理隔离时设 `MEMORY_NEO4J_DATABASE=<独立库>`（Neo4j 企业版多 database；社区版单库即共存模式）。

### 4.3 记忆数据形态

- **Episode**（原始对话痕迹）：一次 `POST .../messages` 批次 = 一个 episode，`source=message`，`episode_body` 为 `"user: ...\nassistant: ..."` 拼接（graphiti message 类型要求 `actor: content` 格式），`reference_time` = 批次首条消息 `created_at`，name = `{session_id}:{首seq}-{尾seq}`。
- **事实边（EntityEdge）**：graphiti 从 episode 抽取实体与事实，携带 `fact` 文本 + `valid_at/invalid_at`（事实世界时间）+ `created_at/expired_at`（系统时间）；矛盾事实由摄取管线自动失效（旧边 `invalid_at`=新事实生效时刻，不删除）。
- 检索返回**事实边**列表（不返回原始消息，原始历史本来就在 `enterprise_chat_messages`）。

## 5. 写路径（记忆摄取）

触发点：`chat_routes.append_chat_messages` 持久化成功后（拿到带 `seq` 的 saved records），调用 `service.schedule_ingest(...)` fire-and-forget（`asyncio.create_task`，任务集合强引用防 GC，先例 `agent_profile` job runner）。

服务内摄取流程（后台任务，全程 try/except，不影响主请求）：

1. `ensure_ready()`：graphiti 未初始化则懒初始化；不可用则记 debug 日志后放弃本次摄取（best-effort 语义）。
2. 摄取模式：`immediate`（默认，每批立即提炼）或 `debounced`（按会话缓冲，静默 `MEMORY_INGEST_DEBOUNCE_SECONDS` 秒后合并成一个 episode，减少小 LLM 调用）。
3. **幂等水位**：持锁后查 `enterprise_chat_memory_episodes` 该会话 `MAX(last_seq)` 水位，过滤掉 `seq ≤ 水位` 的消息（防补偿扫描与实时摄取重复）；空区间直接返回。
4. 过滤角色（仅 `user`/`assistant`），单条内容截断到 `MEMORY_INGEST_MAX_CHARS`（默认 6000 字符，尾部加 `…[truncated]`），拼 `episode_body`。全空区间写一条 `noop_*` 映射行推进水位（避免补偿反复重试）。
5. **per-group 串行**：以 group_id 为键取 `asyncio.Lock`（graphiti 明确要求同 group 顺序 `add_episode`，并发会破坏边失效逻辑）；跨 group 由全局 `asyncio.Semaphore(MEMORY_INGEST_CONCURRENCY)` 限并发，保护本地 vLLM。
6. `graphiti.add_episode(...)`（内部 ≈ 3~5 + 2×边数 次 LLM 调用，其中去重/时间戳走 `small_model`；embedding 每实体名/事实各一次），成功后写 `enterprise_chat_memory_episodes` 映射行（`episode_uuid ↔ session/first_seq/last_seq`）推进水位。
7. 审计 `chat_memory_ingested`（metadata 只记 `user_id/project_id/session_id/message_count/episode_uuid`，不记正文）。
8. 失败：WARNING 日志（含 group 与会话 id），不重试；由**启动补偿扫描**兜底（见 §5.1）。

**成本预算**：单条问答对典型 5~10 次 LLM 调用，异步执行对问答延迟零影响；吞吐由 `MEMORY_INGEST_CONCURRENCY`（默认 2）+ `MEMORY_MAX_COROUTINES`（默认 4，graphiti 内部并发）钳制。

### 5.1 补偿（重启/崩溃恢复）

`enterprise_chat_memory_episodes` 表按 `(session_id)` 记录已摄取的 `seq` 水位。服务启动（`MEMORY_BACKLOG_SCAN_ON_START=true`）后台执行 `run_backlog_scan`：查 `list_chat_memory_backlog`（消息 `MAX(seq) > 水位` 的会话），按 `MEMORY_BACKLOG_BATCH_MESSAGES` 分批重摄取超出水位的消息。这覆盖 fire-and-forget 任务被崩溃打断、debounce 缓冲未刷、Neo4j/LLM 短时不可用等丢失场景，直至收敛。`finalize()` 也会先刷 debounce 缓冲再关闭。

## 6. 读路径（记忆检索）

```http
POST /chat/projects/{project_id}/memory:search
Content-Type: application/json

{"query": "之前对低温性能有什么结论？", "limit": 10}
```

- 鉴权与 `/chat` 其余端点一致：仅企业模式交互式 JWT 用户。校验顺序：先判功能启用（服务未构建 → `503 {"detail": "Chat memory is not enabled"}`，不泄露任何项目信息），再以 `get_project(user_id, project_id)` 校验归属（不存在/他人项目统一 `404`）；已启用但 graphiti 不可用（Neo4j 断连且懒重试失败）→ `503 {"detail": "Chat memory is temporarily unavailable"}`。
- `query` 1..4096 字符；`limit` 1..50，缺省取 `MEMORY_SEARCH_LIMIT`（默认 10）。
- 服务端调 `graphiti.search(query, group_ids=[group_id], num_results=limit)`（`EDGE_HYBRID_SEARCH_RRF` 配方：BM25 + 向量余弦 + RRF 融合，无 reranker/LLM 调用，毫秒~百毫秒级）。
- 响应（`valid_at`/`invalid_at` 供前端/上游区分"当前有效"与"已被更新的历史事实"）：

```json
{
  "project_id": "proj_1a2b3c4d5e6f",
  "total": 2,
  "facts": [
    {
      "uuid": "…",
      "name": "USES",
      "fact": "该项目胎侧胶料采用 NR/BR 并用 50/50 phr",
      "valid_at": "2026-07-10T08:00:05+00:00",
      "invalid_at": null,
      "created_at": "2026-07-10T08:00:41+00:00",
      "expired_at": null
    }
  ]
}
```

- 审计 `chat_memory_searched`：metadata 记 `user_id/project_id/query_hash(sha256)/fact_count`，**不记查询原文**（对齐 query 审计口径）。
- **两种消费方式**：
  1. **服务端自动注入（推荐，前端零拼接）**——见 §6.1，前端只需在 query/agent 请求体加 `memory: {"project_id": ...}`，服务端检索并拼进 prompt。
  2. **独立检索端点**——`memory:search` 供需要显式控制注入内容的场景；前端拿到 facts 后自行拼进 `user_prompt`。服务端注入用的引导语块由 `ChatMemoryService.format_memory_block` 统一生成：

```text
[项目记忆] 以下是该项目历史对话沉淀的事实：
- 该项目胎侧胶料采用 NR/BR 并用 50/50 phr（自 2026-07-10 起）
- 旧结论（已失效于 2026-07-10，仅供追溯）
请结合以上项目记忆回答本次问题；若记忆与检索到的证据冲突，以检索证据为准。
```

### 6.1 查询/Agent 端点服务端自动注入

覆盖端点：`/kbs/{kb_id}/query`、`/query/stream`、`/kbs:query`、`:query/stream`、`/agent/query`、`/query/stream`。请求体新增可选字段：

```json
{
  "query": "低温性能怎么做？",
  "mode": "mix",
  "memory": {"project_id": "proj_1a2b3c4d5e6f", "limit": 10}
}
```

- 服务端流程：`resolve_memory_injection`（`lightrag/api/chat_memory_routing.py`）→ 校验交互式 JWT 用户 + 项目归属 → `build_memory_block`（fail-open 检索 + 格式化）→ 把事实块**前置**到最终 `user_prompt`（`memory_block\n\n原 user_prompt`）→ 传入检索/合成链路。检索本身不使用记忆（只注入 LLM 上下文，与 `conversation_history` 同性质）。
- `/kbs/{kb_id}/query/data`、`/retrieve` 是纯检索、不调 LLM，因此 `memory` 对它们无效（不注入）。
- `/agent/query` 同时新增 `conversation_history` 字段（`[{role, content}]`），传给规划与终答合成 LLM；记忆块注入终答合成的 `user_prompt`。
- 错误语义（与 `memory:search` 一致）：功能未启用 → `503`；非交互式用户 → `403`；他人/不存在项目 → `404`；后端不可用 → **fail-open**（不注入，`metadata.memory={"enabled":false,"reason":"unavailable"}`，查询照常返回）。
- 响应 `metadata.memory` 上报 `{enabled, project_id, fact_count}`；审计 metadata 增加 `memory_enabled/memory_project_id/memory_fact_count`（不记事实文本）。
- `memory` 省略时请求与响应**逐字节不变**（未启用记忆的部署零影响）。

## 7. 生命周期清理与撤销

| 触发 | 行为 |
|---|---|
| `DELETE /chat/projects/{project_id}` | 级联删除会话/消息成功后，fire-and-forget `schedule_purge(user_id, [project_id])` → `clear_data(driver, [group_id])`（按 group 定向 `DETACH DELETE`）+ 删除该项目 episode 映射行；审计 `chat_memory_purged` |
| `DELETE /admin/users/{user_id}` | 路由在删除前枚举该用户全部 chat 项目 id（分页取全），用户删除成功后 fire-and-forget 清理全部对应 group |
| `DELETE .../sessions/{session_id}` | fire-and-forget `schedule_forget_session` → 对该会话全部 episode 逐个 `remove_episode` 并删除映射行；审计 `chat_memory_forgotten`（scope=session） |
| `DELETE .../messages/{message_id}` | 删除前捕获该消息 `seq`；删除成功后 `schedule_forget_message` → 找到覆盖该 seq 的 episode `remove_episode`，再**重摄取该区间的幸存消息**（force=True 绕过水位），使同轮其余内容仍被记住；审计 `chat_memory_forgotten`（scope=message） |

- `remove_episode`（graphiti）删除该 episode 首创的边与孤立节点；`noop_*` 映射行（空内容占位）跳过 graphiti 只删映射。
- 清理是 best-effort 后台任务：失败记 WARNING，控制面记录已删（合规兜底：项目已不存在则检索必然 404，残留图数据不可达）。

## 8. 配置

### 8.1 环境变量（`.env`）

```bash
# ── 用户项目级对话记忆（graphiti）───────────────────────────────
# 总开关：默认关闭；开启需 pip 安装 memory extra（graphiti-core）
LIGHTRAG_CHAT_MEMORY_ENABLED=true

# 记忆抽取 LLM：未设置的字段逐项继承 QUERY_LLM_* → 基础 LLM_*
# （仅支持 OpenAI-compatible 端点；抽取走 json_schema 约束解码，温度固定低温）
# MEMORY_LLM_BINDING_HOST=http://192.168.1.66:8000/v1
# MEMORY_LLM_BINDING_API_KEY=sk-123456
MEMORY_LLM_MODEL=qwen3.6-36b
# 简单子任务（去重/时间戳）用的小模型，缺省同 MEMORY_LLM_MODEL
# MEMORY_LLM_SMALL_MODEL=qwen3.6-36b
MEMORY_LLM_TIMEOUT=300
MEMORY_LLM_TEMPERATURE=0.0
MEMORY_LLM_MAX_TOKENS=16384
# json_schema（vLLM 约束解码，推荐）/ json_object（后端不支持 json_schema 时）
MEMORY_STRUCTURED_OUTPUT_MODE=json_schema
# Qwen3 类模型硬关思考模式，保证 JSON 输出稳定
MEMORY_OPENAI_LLM_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'

# 记忆 embedding：未设置的字段逐项继承 EMBEDDING_*；维度必须与服务一致（超出会截断）
# MEMORY_EMBEDDING_BINDING_HOST=http://192.168.1.66:8002/v1
# MEMORY_EMBEDDING_BINDING_API_KEY=sk-123456
# MEMORY_EMBEDDING_MODEL=qwen-embed
# MEMORY_EMBEDDING_DIM=4096

# 记忆图存储：未设置逐项继承 NEO4J_*；需要物理隔离时指向独立 database（Neo4j 企业版）
# MEMORY_NEO4J_URI=bolt://192.168.1.66:7687
# MEMORY_NEO4J_USERNAME=neo4j
# MEMORY_NEO4J_PASSWORD=...
# MEMORY_NEO4J_DATABASE=neo4j

# 行为参数
MEMORY_SEARCH_LIMIT=10          # memory:search / 注入 缺省返回条数（1..50）
MEMORY_INGEST_CONCURRENCY=2     # 全局并发摄取任务上限（跨 group）
MEMORY_MAX_COROUTINES=4         # graphiti 单次摄取内部并发（保护本地 vLLM）
MEMORY_INGEST_MAX_CHARS=6000    # 单条消息参与摄取的最大字符数（超长截断）
# 记忆检索用部署 reranker 精排（cross-encoder 配方）；关闭则用 RRF
MEMORY_RERANK_ENABLED=false
# 摄取模式：immediate=每轮即提炼；debounced=按会话缓冲静默 N 秒后合并（省 LLM）
MEMORY_INGEST_MODE=immediate
MEMORY_INGEST_DEBOUNCE_SECONDS=20
# 启动补偿扫描：重启后补摄取 seq 超过记忆水位的会话
MEMORY_BACKLOG_SCAN_ON_START=true
MEMORY_BACKLOG_BATCH_MESSAGES=20
# 每用户在途摄取任务上限（公平性；防单用户刷爆共享 LLM，超限批次由补偿扫描兜底）；0=不限
MEMORY_MAX_INFLIGHT_PER_USER=8

# graphiti 匿名遥测：内网部署保持关闭（服务代码内亦有 setdefault 兜底）
GRAPHITI_TELEMETRY_ENABLED=false
```

> `MEMORY_RERANK_ENABLED=true` 复用部署的 `RERANK_BINDING`/`RERANK_MODEL`/`RERANK_BINDING_HOST` 服务（如 qwen3-rerank）；未配置 reranker 时该开关无效果（回退 RRF）。

### 8.2 回退链

| 最终值 | 优先级（左高右低） |
|---|---|
| LLM host | `MEMORY_LLM_BINDING_HOST` → `QUERY_LLM_BINDING_HOST` → `LLM_BINDING_HOST` |
| LLM api key | `MEMORY_LLM_BINDING_API_KEY` → `QUERY_LLM_BINDING_API_KEY` → `LLM_BINDING_API_KEY` |
| LLM model | `MEMORY_LLM_MODEL` → `QUERY_LLM_MODEL` → `LLM_MODEL` |
| small model | `MEMORY_LLM_SMALL_MODEL` → 最终 LLM model |
| embedding host/key/model/dim | `MEMORY_EMBEDDING_*` → `EMBEDDING_*` |
| Neo4j uri/user/password/database | `MEMORY_NEO4J_*` → `NEO4J_*`（database 最终缺省 `neo4j`） |

### 8.3 依赖安装

```bash
# pyproject 新增 optional extra：memory = ["graphiti-core>=0.29.2,<0.30"]
uv sync --extra api --extra memory
# 或 pip install "lightrag-hku[api,memory]"
```

未安装 graphiti-core 而开启 `LIGHTRAG_CHAT_MEMORY_ENABLED=true` 时：启动打 ERROR 日志、`memory:search` 返回 503，其余功能不受影响（懒导入，不在模块顶层 import）。

## 9. 安全与审计

- RBAC：与 `/chat` 一致——仅交互式 JWT 用户、按 `principal.user_id` 隔离、越权/不存在统一 404、属于企业 anti-bypass 受保护前缀，不能被 `WHITELIST_PATHS` 放行。
- 跨租户/跨用户隔离由 group_id 保证：检索/清理永远强制携带且仅携带本人 group。
- 新增审计事件（metadata 白名单，不含名称/正文/查询原文）：
  - `chat_memory_ingested`：`user_id/project_id/session_id/message_count/episode_uuid`
  - `chat_memory_searched`：`user_id/project_id/query_hash/fact_count/limit`
  - `chat_memory_purged`：`user_id/project_count`（+ 目标 project_id 列于 target_id / metadata）
  - `chat_memory_forgotten`：`user_id/project_id/session_id/scope(message|session)/episode_count`（message 域附 `reingested_messages`）
  - query/agent 端点注入时，既有 `query_executed` / `query_stream_started` / `multi_kb_query_*` / agent 审计增加 `memory_enabled/memory_project_id/memory_fact_count` 白名单字段。
- 遥测：`GRAPHITI_TELEMETRY_ENABLED=false`（env + 代码 setdefault 双保险）；graphiti 遥测仅在实例化时上报 provider 类型，断网时静默失败。

## 9.1 可观测性与运维接口（多用户部署）

- **健康检查**：`GET /health` 增加 `chat_memory: {enabled, available, pending_tasks}`。`available=false` 且 `enabled=true` 表示 graphiti/Neo4j 掉线懒重试中——这是运维最需要的信号（记忆 fail-open，用户端无感）。
- **用户侧记忆概览**：`GET /chat/projects/{project_id}/memory` → `{enabled, available, episode_count, last_ingested_at}`，前端可展示"已沉淀 N 条记忆、上次更新时间"，不触发检索。归属校验同 `/chat`（他人/不存在 404）；`episode_count` 排除 `noop_` 占位行。
- **super admin 全局统计**：`GET /admin/overview` 的 `chat_memory` 块含 `{enabled, available, pending_tasks, episode_count, user_count, project_count}`。
- **super admin 手动运维**：
  - `POST /admin/users/{user_id}/chat-memory:purge`（可选 `project_ids`，省略清全部）——脏数据/模型升级后重置某用户记忆。
  - `POST /admin/chat-memory:backlog-scan`（可选 `limit`）——LLM 恢复后主动补偿，不必等重启。
- **公平性/配额**：`memory:search` 与注入走 `/chat`/query 前缀,受企业中央限流中间件约束。摄取的 LLM 开销**不计入**用户 HTTP 配额（发生在后台任务),因此加了 `MEMORY_MAX_INFLIGHT_PER_USER`（默认 8）per-user 在途上限:单用户狂发消息不会占满全局摄取槽饿死他人;超限批次被 DB 水位记录、由 backlog 扫描兜底重摄，不丢数据。高并发部署可再配合 `MEMORY_INGEST_MODE=debounced` 降低摄取频次。

## 10. 测试策略

服务/路由测试全部离线（`pytest.mark.offline`），不依赖真实 Neo4j / LLM——graphiti 以**注入 fake** 方式替身（服务构造函数接受 `graphiti_factory`/`clear_data_fn`/`metadata_store`/`rerank_fn`）：

- `tests/api/test_chat_memory_service.py`（服务单元）：group_id 校验、episode 格式/截断、同 group 串行与跨 group 并发上限、检索强制 `group_ids`、配置回退链与 clamp、extra_body 包装、**cross-encoder 精排适配与排序/fallback**、**水位幂等**、**空区间推进水位**、**backlog 补偿**、**消息级 forget + 幸存重摄**、**会话级 forget**、**purge 清映射行**、**debounce 合并**、记忆块格式化、`build_memory_block` fail-open。
- `tests/api/routes/test_chat_memory_routes.py`：`memory:search` happy path、他人项目 404 / 未启用 503 / API-key 403 / 未认证；append 触发 ingest；删除消息/会话触发 forget（含 seq 捕获与 no-op 不触发）；项目/用户删除触发 purge。
- `tests/api/routes/test_chat_memory_injection.py`：查询端点服务端注入把记忆块拼进 `param.user_prompt`、`metadata.memory` 上报、无 `memory` 字段零影响、他人项目 404、未启用 503、后端不可用 fail-open。
- `tests/api/test_metadata_store_contract.py`：episode 映射表（水位 / 覆盖查询 / backlog / 各级删除 / 归属隔离）SQLite↔PostgreSQL 行为等价。
- `tests/api/test_chat_memory_server_wiring.py`：`create_app` 按开关构建服务并解析配置回退链。
- 回归：既有 `test_chat_routes.py`、agent 路由/staged、KB query 路由全量通过（`memory` 省略时逐字节兼容）。

## 11. 上线核对清单（运维）

1. Neo4j server 版本 ≥ 5.26：`CALL dbms.components() YIELD versions`；不满足则升级或为记忆单独部署实例并设 `MEMORY_NEO4J_URI`。✅ **已验证（2026-07-11）**：192.168.1.66:7687 为 Neo4j Kernel **5.26.26 community**，达标；社区版单库，按 label+group_id 共存模式运行。
2. 安装 memory extra 依赖；`.env` 打开 `LIGHTRAG_CHAT_MEMORY_ENABLED=true` 并核对 MEMORY_* 回退值。✅ 已完成（graphiti-core 0.29.2 已入 venv，`.env` 已配置同款 qwen3.6-36b / qwen-embed dim=4096）。
3. 用生产 qwen 实测端到端：✅ **真机 PoC 全流程已通过（2026-07-11，一次性分区，测后已清理）**——一期：建索引 5.0s、两轮摄取 11.6s/7.0s（后台异步无感）、检索 0.1s 召回 5 条高质量事实、跨用户隔离 0 泄漏、按 group 清理归零；全量：**幂等水位去重、backlog 补偿扫描、消息级 forget + 幸存重摄、purge 清图与映射行、`build_memory_block` 产出可注入的记忆块**（实测生成"低温屈挠性是首要指标 / NR-BR 并用 50-50 phr / 环烷油替代芳烃油"等事实块）均正确。
4. 观察 vLLM 负载；必要时下调 `MEMORY_INGEST_CONCURRENCY` / `MEMORY_MAX_COROUTINES`，或切 `MEMORY_INGEST_MODE=debounced` 合并摄取；高峰期可临时关总开关（只影响新记忆写入与检索，不丢已存数据）。

## 12. 后续可选增强（备忘）

1. **durable 摄取**：把 fire-and-forget 换成 JobWorker job_type（当前 fire-and-forget + 启动补偿已满足 best-effort；durable 化需解决 jobs 表 `kb_id NOT NULL` 挂靠问题）。
2. **社区摘要**：`update_communities=True` + `build_communities` 为大项目生成主题级摘要层（当前事实边层已足够，成本更低）。
3. **跨项目/全局记忆**：可选让用户在多个项目间共享一层"个人偏好"记忆（需新的 group 维度设计）。

