# 企业版 Chat Memory：当前实现与运维契约

> 校准日期：2026-07-16
>
> **Production gate: GO for the verified configured deployment as of 2026-07-16.**
>
> 此 GO 仅适用于当前 working tree 与本次验证所用的 PostgreSQL 15、Neo4j database/deployment、Graphiti 0.29.2、记忆抽取 LLM/embedding、最终 Query provider、对应模型与 embedding dimension，以及默认 same-endpoint egress policy；它不是跨部署、跨 provider 或未来代码版本的通用认证。working tree 尚未提交，实际部署前必须把证据绑定到不可变 commit/artifact，并取得 release owner 签字。

本文描述当前企业版服务端实现。核心原则只有一条：**PostgreSQL 中的原始聊天记录是真相源；Graphiti/Neo4j 中的记忆图是可删除、可重建的派生索引。**

---

## 1. 范围与前置条件

Chat Memory 只覆盖企业认证与 PostgreSQL 元数据路径。部署前提：

1. `LIGHTRAG_ENTERPRISE_AUTH_ENABLED=true`；
2. `LIGHTRAG_KB_METADATA_BACKEND=postgres`；
3. `LIGHTRAG_CHAT_MEMORY_ENABLED=true`；
4. PostgreSQL 元数据存储可用并已完成 schema 初始化；
5. Graphiti、Neo4j、记忆抽取 LLM 与 embedding 配置可用。

企业认证和 PostgreSQL 是 fail-fast 硬前提：不满足时应用在配置校验阶段失败。Graphiti/Neo4j/provider 是运行就绪前提：初始化失败时进程可启动，但 health 显示未就绪，读路径返回 typed availability，durable worker 留待恢复。

本文不对 Docker 部署或单用户/本地模式作支持性承诺。

安装依赖：

```bash
uv sync --extra api --extra memory
```

仓库当前固定使用 `graphiti-core==0.29.2`，由 Graphiti 访问 Neo4j。

| 数据层 | 当前职责 | 权威性 |
|---|---|---|
| PostgreSQL 聊天表 | 项目、会话、消息原文和顺序 | **权威真相源** |
| PostgreSQL Chat Memory 表 | 逻辑组、generation、outbox、claim、重试、映射、删除意图 | **一致性控制面** |
| Graphiti/Neo4j | 从获准消息抽取事实、实体和关系 | **派生且可重建** |

直接后果：

- 原始消息删除以 PostgreSQL 事务为准；
- 图侧失败不能回滚已提交的聊天 CRUD；
- 图侧通过持久 outbox、generation 前滚、重建或清除最终收敛；
- 不能用 Neo4j 中的 episode/fact 反推完整原始聊天；
- `store_raw_episode_content=false` 不会删除 PostgreSQL 源消息。

---

## 2. 架构

```text
┌──────────────────────────────── Enterprise API process(es) ────────────────────────────────┐
│ JWT principal                                                                               │
│   │                                                                                         │
│   ├─ ChatConversationService / UserService                                                  │
│   │    └─ one PostgreSQL transaction                                                        │
│   │         ├─ chat projects / sessions / messages                                          │
│   │         ├─ enterprise_chat_memory_groups                                                │
│   │         ├─ enterprise_chat_memory_generations                                           │
│   │         ├─ enterprise_chat_memory_outbox                                                │
│   │         └─ enterprise_chat_memory_episodes mapping                                      │
│   │                    └─ COMMIT → post-commit worker nudge                                  │
│   │                                                                                         │
│   └─ Query / Agent authorization → fact-free AuthorizedChatMemoryHandle                     │
└───────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                            ▼
┌──────────────────────────────── PostgreSQL durable log ─────────────────────────────────────┐
│ per (user_id, project_id):                                                                   │
│ monotonic event_seq/reference_time; desired/active generation; FIFO outbox head;             │
│ claim token/owner/retry/dead-letter; source-batch ↔ physical-group/episode mapping            │
└───────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                            ▼
┌────────────────────────────────── ChatMemoryWorker(s) ───────────────────────────────────────┐
│ FIFO claim with SKIP LOCKED                                                                  │
│ claim token + claimed owner + PostgreSQL session advisory group guard                        │
│ known pre-side-effect failure → retry_wait / dead_letter                                     │
│ unknown graph outcome → abandon generation and roll forward                                 │
│ stale recovery → only after non-blocking acquisition of the same group guard                 │
└───────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                            │ Graphiti backend lease
                                            ▼
┌────────────────────────── ChatMemoryService / Graphiti 0.29.2 / Neo4j ───────────────────────┐
│ add_episode; search(explicit active physical group); clear_data(explicit group ids)           │
│ physical groups: cm_<logical_hash>_g1, cm_<logical_hash>_g2, ...                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘

Read: token before search → active physical group → Graphiti current-fact search under lease
      → token after search → return only if both fences match.
```

主要实现文件：

- `lightrag/api/metadata_store.py`：存储协议、数据模型和 SQLite 契约实现；
- `lightrag/api/postgres_metadata_store.py`：生产事务、锁、claim、重建和清除状态机；
- `lightrag/api/chat_memory_worker.py`：FIFO 消费、重试、未知结果恢复和停机；
- `lightrag/api/chat_memory_service.py`：Graphiti 后端、搜索、read fence、物理组和指纹；
- `lightrag/api/enterprise_auth.py`：聊天 CRUD、项目删除和用户删除接线；
- `lightrag/api/chat_memory_routing.py`、`lightrag/sensitive_context.py`：延迟解析、预算和出口策略；
- `lightrag/operate.py`、`bilingual_query_service.py`、`agent_query_service.py`、`agent_staged_service.py`：最终合成；
- `lightrag/llm_roles.py` 与 `lightrag/llm/*`：敏感 LLM 调用生命周期；
- `lightrag/api/routers/*`：HTTP、审计和管理员契约。

---

## 3. 持久模型与一致性

### 3.1 逻辑组和物理 generation
每个 `(user_id, project_id)` 是一个逻辑记忆组。

```text
logical_group_id = cm_<sha256(user_id + NUL + project_id) 前 24 个十六进制字符>
physical_group_id = <logical_group_id>_g<generation>
```

Graphiti 的 ingest、search 和 clear 都使用物理 generation group id。generation 用于隔离 stale writer、未知副作用和 rebuild 结果。

### 3.2 PostgreSQL 对象
| 对象 | 关键内容 |
|---|---|
| Chat messages | `append_batch_id`、`project_event_seq`、`memory_reference_time`、原文 |
| Memory groups | 状态、`next_event_seq`、desired/active generation、desired extraction fingerprint、graph-store fingerprint |
| Memory generations | generation 状态、snapshot cutoff、配置指纹、物理 group id |
| Memory outbox | event id/type/seq、状态、claim、attempt、next attempt、side-effect state、actor/error |
| Memory episode mapping | source batch、session、event seq、physical group、episode uuid、ingestion state |

outbox 不依赖聊天源表外键级联，所以项目或用户源记录删除后，purge intent 仍可恢复执行。

事件类型：

- `ingest`：将一个已提交 append batch 写入目标 generation；
- `rebuild`：从 SQL 中存活且获准的 batches 重建新 generation；
- `purge`：清除逻辑组的全部已知、遗留和孤儿物理组。

主要状态：

```text
event:      pending → running → succeeded
                         ├────→ retry_wait → running
                         ├────→ dead_letter
                         └────→ superseded
generation: building / active / retired / abandoned / purge_pending / purged
group:       active / rebuilding / deleting / failed / deleted
```

### 3.3 事务边界
聊天写入/删除与 durable intent 在同一 PostgreSQL 事务中提交。事务成功后 source rows 与 outbox 同时可见；worker 崩溃不会丢失 intent；HTTP 请求不等待 Graphiti 完成。

PostgreSQL 与 Neo4j 之间没有分布式事务。当前语义是“持久 intent + 可验证状态转换 + generation 前滚”，不是跨数据库 exactly-once。确定性 event/batch id 也不等价于 Graphiti episode exactly-once。

### 3.4 `event_seq` 与 `reference_time`
每个逻辑组在组行锁或创建锁保护下分配单调递增 `event_seq`。同一 append batch 的消息共享：

- 一个 `append_batch_id`；
- 一个 `project_event_seq`；
- 一个持久 `memory_reference_time`。

`reference_time = epoch + event_seq 微秒`。它不使用 worker 当前时间，也不从 session 局部序号推导，因此重放仍保持原项目顺序。

### 3.5 同组串行，不同组并发
同一逻辑组：

1. 只能 claim 未被更低阻塞事件挡住的 FIFO head；
2. claim 使用 PostgreSQL 行锁与 `SKIP LOCKED`；
3. 图副作用前获取 session-level advisory group guard；
4. transition/finalize 必须继续匹配 claim token；
5. stale recovery 也必须先获得相同 group guard。

`claim_token`、`claimed_by` 和 advisory guard 共同构成执行所有权。`claimed_at` 超时只使记录成为 stale 候选，不能单独授权第二个 writer。

不同逻辑组可以并发，受 `MEMORY_INGEST_CONCURRENCY` 限制。

### 3.6 FIFO、重试与 dead letter
`pending`、`running`、`retry_wait` 和 `dead_letter` 都会阻塞同组更高 `event_seq`，防止低序号结果未确定时越序执行。

已知且确定发生在图副作用前的失败进入 `retry_wait`；达到最大 attempt 后进入 `dead_letter`。dead letter 不会被后台隐式当作成功跳过，必须由明确 rebuild、purge 或允许的管理员事件操作恢复。

### 3.7 未知图结果前滚
一旦 `side_effect_started` 已持久化，Graphiti 超时、取消、进程崩溃或断链都可能表示“图已经改变，但 SQL 未 finalize”。

对 `ingest`/`rebuild`：

1. 不在原 generation 上盲重放；
2. 原 target generation 标记 `abandoned`；
3. desired generation 递增；
4. 创建 rebuild event；
5. 在新物理 group 中从 SQL snapshot 重建；
6. 成功后才切换 active generation。

对 `purge`：保留同一 durable intent，回到 `retry_wait`，下次重新计算完整 clear universe 并再清一次；只有确定全部清除后才成功。

### 3.8 graph-store identity 不可变
`graph_store_fingerprint` 包含 Neo4j provider、deployment identity 和 database。deployment identity 优先取 `MEMORY_NEO4J_DEPLOYMENT_ID`，否则使用去凭据、规范化后的 URI。

一旦逻辑组在某 graph-store fingerprint 下存在持久证据，后续 append、delete、rebuild、purge 和 worker claim 必须继续使用同一 fingerprint。改变 Neo4j deployment/database 会产生 `graph_store_migration_required` 类冲突；当前实现不把它当普通配置升级。

### 3.9 extraction upgrade
`extraction_fingerprint` 表示同一图存储中如何抽取与记录事实。LLM/embedding 模型、raw storage、admission、记录 schema、snapshot policy 或单消息截断策略变化会改变它。

只要 graph-store fingerprint 不变，新的 memory-aware mutation 或显式 rebuild 可以推进 generation、固定 snapshot、重放 surviving batches、清除旧组并激活新 extraction fingerprint。图存储搬迁不属于 extraction upgrade。

### 3.10 激活前的权威 clear coverage
rebuild/purge 从 PostgreSQL 构造权威 clear universe，覆盖：

- active/building/retired/abandoned generations；
- generation inventory 中所有 physical group ids；
- episode mappings 和 outbox 中出现过的 groups/generations；
- 旧逻辑 group id；
- 当前 rebuild target group。

Graphiti clear 必须接收显式、非空 group id 列表。finalize 时 SQL 再次计算权威集合；只有 worker 报告的 definite clear coverage 完整覆盖它，才允许激活新 generation 或将 purge group 标记为 deleted。覆盖不足时不得部分激活。

---

## 4. Append、删除、重建与 purge

### 4.1 消息 append
启用 admission 时，`append_messages_with_chat_memory` 在一个事务中：

1. 校验用户、项目和 session 所有权；
2. 锁定聊天与 memory group；
3. 分配 session `seq`、项目 `event_seq` 和 `reference_time`；
4. 写入消息原文和 admission identity；
5. 确保逻辑组与目标 generation 存在；
6. 写入一个 `ingest` outbox event；
7. 提交后 nudge worker。

一个 append 请求是稳定 episode boundary，worker 不按运行时调度重新拼 batch。首次 append 创建 generation 1；第一个确定成功的 ingest（包括 admission 后为空的 no-op）完成后，generation 1 才可 active。

功能关闭时消息仍正常保存，但 admission identity 为 `NULL`，未来打开功能不会自动把这些旧消息纳入 replay。

### 4.2 Admission no-op
outbox 先持久化完整 batch identity，worker 再应用 admission policy。若没有消息获准：

- 不调用 Graphiti add episode；
- 持久化 `no_op` mapping；
- 事件确定成功；
- FIFO 顺序不会卡住。

### 4.3 删除 message
`delete_message_with_chat_memory` 在同一事务中锁定消息，判断是否属于获准 batch/已有 mapping，删除 source message，并在受影响时推进 generation、写入 rebuild event。重建从 SQL 全部 surviving admitted batches 生成图，不做局部图撤销。

### 4.4 删除 session
`delete_session_with_chat_memory` 扫描该 session 的获准 batches/mappings。若派生记忆受影响，事务同时删除 session/messages、推进项目 generation 并写入全项目 rebuild；若从未产生获准 batch 且无 mapping，则不制造无意义 rebuild。

### 4.5 删除 project
项目删除在一个事务中：锁定项目和逻辑组 → 写入 durable purge tombstone/event → group 进入 deleting → 删除 project/session/message source rows → 提交后 nudge。

purge event 不依赖已删除 source rows，进程重启后仍可执行。

### 4.6 删除 user
用户删除事务会收集 source projects 与 memory durable tables 的项目并集，按稳定顺序加锁，为每个项目写入 purge intent，再删除用户及其他源数据。即使项目源记录已缺失，只要 durable memory evidence 存在，它仍进入 purge universe。

### 4.7 Rebuild
claim 时固定 `snapshot_cutoff`，通常为当时 `next_event_seq - 1`。snapshot 只含：

- `project_event_seq <= snapshot_cutoff`；
- admission identity 完整；
- source rows 仍存在；
- batch 中仍有获准消息。

超过 cutoff 的并发 append 保留为更高序号事件，在 rebuild 后继续处理。

Graphiti 调用前先做 aggregate preflight：message 数不得超过 `rebuild_max_messages`，JSON source bytes 不得超过 `rebuild_max_bytes`。超过 hard cap 时不做 partial replay、不激活不完整 generation，事件以 `rebuild_snapshot_hard_cap_exceeded` 进入 `dead_letter`。

正常流程：建立 building generation → 固定 snapshot 与权威 target inventory → 标记 side effect started → 先清除包含 target group 在内的完整集合 → 按原 `reference_time` replay 到 target group（每个 batch 间重检 fence）→ SQL 再核验 snapshot/claim/fingerprints/coverage → 原子切换 active generation → 已清除的其他 generations 标记为 `purged`。

### 4.8 Purge
purge 不需要 source content，只依赖 durable target identity 和权威 group inventory。Graphiti 确定清除完整集合，且 finalize 时 claim、side-effect state、fingerprint 和 coverage 仍匹配后，group 才进入 `deleted`，event 才进入 `succeeded`。

### 4.9 source deletion 后按 event id 重试
```http
POST /admin/chat-memory/events/{event_id}:retry
```

当前只允许重试 `purge`：

- `dead_letter`：重新排为立即可用的 `retry_wait`，清理旧 claim/error；
- `pending`/`retry_wait`：幂等返回原状态并 nudge；
- `running`/`succeeded`/`superseded`：冲突；
- 非 purge event：冲突；
- graph-store fingerprint 不匹配：迁移冲突。

该接口不需要项目或用户源记录仍存在；event row 中的 target identity 会保留。普通 ingest/rebuild 的未知结果必须 generation 前滚，不能在原 generation 上直接重放。

响应：

```json
{"event_id":"...","status":"retry_wait","user_id":"...","project_id":"...","event_type":"purge"}
```

---

## 5. 搜索、read fence、admission 与隐私

### 5.1 专用搜索
```http
POST /chat/projects/{project_id}/memory:search
Authorization: Bearer <JWT>

{"query":"项目当前有哪些关键约束？","limit":10}
```

要求：交互式 JWT 用户、项目 ownership、query 不超过 4096 字符；`limit` 限制为 1..50，缺省使用 `MEMORY_SEARCH_LIMIT`。Graphiti search 总是传显式 active physical group id，绝不做无 group 的全图搜索。

### 5.2 默认 current facts
搜索使用 `invalid_at IS NULL AND expired_at IS NULL`，默认只返回 current facts。响应仍保留 `valid_at`、`invalid_at`、`expired_at` 字段；默认结果的后两者为 `null`。响应中的 Graphiti fact UUID 只在本次 active generation 语境内有效；重建后可能变化，不得作为跨 generation 的永久业务主键。

### 5.3 active-generation 前后 fence
1. SQL 读取 token A；
2. A 必须表明 group 可读，且 active generation 匹配 runtime extraction/graph-store fingerprints；
3. 用 A 的 physical group id 搜索；
4. SQL 读取 token B；
5. A/B 的 generation、fingerprints 和 group state 必须相同；
6. 不同则最多重试一次；再次变化抛 typed availability error。

这样不会把 generation 切换期间的旧组结果作为当前事实返回。

### 5.4 Graphiti backend lease
search、ingest、rebuild、clear 都在 backend lease 下使用 Graphiti instance。后端失败时 slot 被 retired；只有 active calls 降为 0 才 close；新调用创建新 slot，避免长调用或 stream 使用中的 client 被提前关闭。

专用 `memory:search` 的 Graphiti availability failure 返回 503。Query/Agent 注入只对 `ChatMemoryUnavailableError` fail-open 为 `status=unavailable`；授权、ownership、query 长度、final-synthesis、egress、预算/builder 不变量和未分类错误均不能 fail-open。

### 5.5 Admission policy version 1
| 消息 | 进入 episode？ |
|---|---|
| 非空 `user` | 是 |
| 非空 `assistant` 且 `metadata.memory_eligible` 为 JSON 布尔 `true` | 是 |
| assistant 未标记 | 否 |
| 字符串 `"true"`、数字 `1` 等 truthy 值 | 否 |
| system/tool/其他角色或空白内容 | 否 |

assistant 默认不进入长期记忆，必须由可信服务端逻辑显式标记。

获准消息按 `MEMORY_INGEST_MAX_CHARS` 截断，默认 6000，并附确定性 `…[truncated]` 标记；截断策略参与 extraction fingerprint。

### 5.6 Raw episode content
默认 `LIGHTRAG_CHAT_MEMORY_STORE_RAW_EPISODE_CONTENT=false`，Graphiti 初始化收到 `store_raw_episode_content=False`。这表示不要求图中保存完整原始 episode 文本；抽取事实/实体/关系仍在 Neo4j，源消息原文仍在 PostgreSQL。删除和重建依赖 SQL durable identity，而不是图中原文。

---

## 6. Query/Agent 注入契约

### 6.1 请求与早期授权
支持 final synthesis 的单 KB、多 KB、双语 Query 与 Agent 可带：

```json
{"memory":{"project_id":"chat_project_id","limit":10}}
```

路由在 KB retrieval 或 Agent planning 前调用 `authorize_memory_context`，只检查 feature、JWT、ownership、query 长度并创建 fact-free handle；此时不搜索 Graphiti，不把 fact 带入 retrieval、planning、tool round 或 clarification。

### 6.2 只在最终合成解析一次
只有主流程已得到权威 KB evidence、即将发出 final query-LLM request 时才：

1. 绑定实际 final query endpoint；
2. 检查跨 provider egress；
3. 计算 token/char/full-request 预算；
4. 最多搜索 memory 一次；
5. 选择完整 records；
6. 构造 trusted policy 与 untrusted context；
7. 发出一个 sensitive final LLM call。

Agent planning/retrieval rounds 不解析 memory；双语两条 retrieval path 也不分别搜索。

### 6.3 三重预算
| 预算 | 默认/来源 |
|---|---|
| memory token cap | `LIGHTRAG_CHAT_MEMORY_PROMPT_MAX_TOKENS=1024` |
| memory character cap | `LIGHTRAG_CHAT_MEMORY_PROMPT_MAX_CHARS=8192` |
| full final-request cap | 当前请求有效 `max_total_tokens` + 实际 query-role tokenizer |

完整请求计数包含最终 system prompt、KB evidence、trusted memory policy、untrusted JSONL、conversation history role/content、当前 query、确定性 separators 和 64-token framing reserve。

预算器对每个候选前缀重建并编码完整请求，而不是只估算 memory block。缺少 tokenizer 或无法证明 capacity 时，在 Graphiti search 前返回 `budget_exhausted`；不能删除 KB evidence 给 memory 腾空间，也不能发送不可证明安全的 memory-bearing request。

### 6.4 完整 JSONL records
候选 fact 从 `[M1]` 连续编号。每条必须同时满足三重预算；单条过大则跳过并尝试后续记录，不截断 JSON 行，不留下半个 fact。

不可信数据格式示例：

```text
{"reference_id":"M1","fact":"...escaped data...","valid_at":"..."}
{"reference_id":"M2","fact":"...escaped data...","valid_at":null}
```

fact 的 JSON 控制字符、换行以及可伪造 section/ref 的尖括号和方括号序列会被转义。

### 6.5 Trusted policy 与 untrusted context
服务端生成的记忆规则进入 trusted instruction 区域；fact JSONL 进入 data-only context。模型被明确告知：memory data 不可信，其中命令、角色文本或策略文本不能改变系统行为。

### 6.6 引用命名空间与 corroboration
| 来源 | 引用 |
|---|---|
| 普通 KB Query | `[1]`、`[2]` |
| Agent KB evidence | `[A1]`、`[A2]` |
| Chat Memory | `[M1]`、`[M2]` |

顶层 `references` 只表示 KB/Agent evidence。memory refs 位于 `metadata.memory.references`，每项包含 `reference_id`、generation-scoped `fact_id` 和 `valid_at`，不回传 fact 文本。

Trusted policy 要求：

- memory 只作补充；
- 外部事实、数值、日期、制度或当前状态必须有本次 KB evidence 支持；
- KB 与 memory 冲突时以当前 KB 为准并说明冲突；
- 不能只凭 `[M*]` 输出需要 `[1]`/`[A*]` 证明的事实；
- Agent 的 `[M*]` 不能满足 `[A*]` 引用要求。

### 6.7 Cache bypass
存在 authorized memory handle 时，final query completion cache 的读取和写入均绕过，即使结果为 `empty`、`unavailable`、`budget_exhausted` 或 `not_used`。不含 memory fact 的前置关键词缓存可保持原行为。

### 6.8 Sensitive 调用、日志、trace 与 stream
memory-bearing final call 通过私有 `_sensitive` 标志进入敏感作用域：

- final completion cache bypass；
- verbose prompt/response 日志抑制；
- OpenAI 使用未接可选 Langfuse wrapper 的标准 client；
- 内置 provider 对请求、响应和异常使用敏感分支；
- 同步异常映射为稳定、无内容错误；
- async iterator 整个消费生命周期保持敏感作用域；
- Query 流错误只返回通用 `Sensitive LLM call failed`；
- Agent 流错误只返回稳定 error code/status/message；
- memory-scoped Agent 通用异常不回显 provider detail。

第三方自定义 provider 也必须遵守相同契约，不能自行记录完整 prompt/response。

### 6.9 状态与内容安全审计
| status | 含义 |
|---|---|
| `injected` | 至少一个 fact 进入 final request；`injected_count > 0` |
| `empty` | 搜索成功但无 current fact |
| `unavailable` | typed backend availability failure，主查询继续 |
| `budget_exhausted` | 无 fact 可在全部预算内安全注入 |
| `not_used` | 没有 final synthesis，如 clarification/no evidence |

响应 `metadata.memory` 可含 `project_id`、`fact_count`、`injected_count`、`truncated`、`reason`、`references`。

Query/Agent audit 只增加：

```text
memory_enabled
memory_fact_count
memory_injected_count
memory_status
memory_truncated
memory_reason
```

这些 audit 字段不含 query、fact、prompt、response 或 memory reference payload。专用 memory search audit 保存 query hash、limit、计数和授权目标 id，不保存 query/fact 原文。

### 6.10 无 memory 字节兼容
未提供 `memory` 时：不创建 handle；走原非敏感分支；不改变 final cache；不增加 `metadata.memory` 或 memory audit keys；Agent stream 不加 memory wrapper；NDJSON 事件和 done payload 保持原路径字节行为。

---

## 7. 无最终合成矩阵与 Agent 特殊分支

Dedicated `memory:search` 是显式搜索 API，不受本矩阵限制。Query/Agent answer injection 只有在会执行 final query LLM 时才允许解析 memory。

稳定 400 错误码：`chat_memory_requires_final_synthesis`。

| 路径/模式 | 携带 memory 时 |
|---|---|
| `/kbs/{kb_id}/query` 或 `/kbs/{kb_id}/query/stream` + `mode=bypass` | 400 |
| 单 KB query + `only_need_context=true` | 400 |
| 单 KB query + `only_need_prompt=true` | 400 |
| 单 KB query + 两者同时 true | 400 |
| `/kbs/{kb_id}/query/data` | 400 |
| `/kbs/{kb_id}/retrieve` | 400 |
| `/kbs:query` 或 `/kbs:query/stream` + `mode=bypass` | 400 |
| `/kbs:retrieve` | 400 |

显式 bypass body 检查先于企业 bypass capability gate，以返回稳定 memory 契约错误。

Query 没有权威 KB evidence 时：不搜索 memory、不允许只凭 memory 作答，状态 `not_used`，`reason=no_kb_evidence`。该规则覆盖单 KB、多 KB和双语路径。

Agent clarification：planning 可先完成，但不搜索 memory；Agent status 为 `clarification_required`；memory 为 `not_used`，`reason=clarification_required`。

Agent 无 evidence：不搜索 memory，不用 memory 填补证据；memory 为 `not_used/no_kb_evidence`。正常零证据分支返回固定“未检索到可用于回答的证据”语义；全部 retrieval steps 失败时保留 Agent 错误契约，也不得解析 memory。

---

## 8. 跨 provider 出口

默认 `LIGHTRAG_CHAT_MEMORY_ALLOW_CROSS_PROVIDER_QUERY_EGRESS=false`。

Memory extraction provider 与实际 final query provider 分别生成去凭据、去 query、规范化 scheme/host/default port/path 的 endpoint identity。默认只在“两侧都已知且相等”时允许。

endpoint 不同、只有一侧可识别、两侧都不可识别，都会在 Graphiti search 和 final LLM 前拒绝：HTTP 403，错误码 `chat_memory_query_llm_egress_not_allowed`。

只有完成数据出口评审后才可显式设为 `true`。多 KB/Agent 必须绑定实际执行 final synthesis 的 endpoint，不能假设任意首个 KB provider。

---

## 9. 当前配置与默认值

### 9.1 主开关
| 配置 | 默认 | 作用 |
|---|---:|---|
| `LIGHTRAG_ENTERPRISE_AUTH_ENABLED` | false | 必须为 true |
| `LIGHTRAG_KB_METADATA_BACKEND` | `local` | 必须为 `postgres` |
| `LIGHTRAG_CHAT_MEMORY_ENABLED` | false | admission、read/injection、ingest worker |
| `LIGHTRAG_CHAT_MEMORY_MAINTENANCE_ENABLED` | true | 功能关闭时仍处理 rebuild/purge |

`MEMORY_MAINTENANCE_ENABLED` 是兼容别名；规范名称优先。

### 9.2 Graphiti、Neo4j、LLM 与 embedding
| 配置 | 默认/回退 | 指纹类别 |
|---|---|---|
| `MEMORY_NEO4J_URI` | `NEO4J_URI` | graph-store |
| `MEMORY_NEO4J_USERNAME` | `NEO4J_USERNAME` | secret，不进指纹 |
| `MEMORY_NEO4J_PASSWORD` | `NEO4J_PASSWORD` | secret，不进指纹 |
| `MEMORY_NEO4J_DATABASE` | `NEO4J_DATABASE`，再 `neo4j` | graph-store |
| `MEMORY_NEO4J_DEPLOYMENT_ID` | 未设置 | graph-store 稳定部署身份 |
| `MEMORY_LLM_BINDING_HOST` | query LLM host，再主 LLM host | extraction |
| `MEMORY_LLM_BINDING_API_KEY` | query/main key | secret |
| `MEMORY_LLM_MODEL` | query/main model | extraction |
| `MEMORY_LLM_SMALL_MODEL` | 有效 memory/query/main LLM model | extraction |
| `MEMORY_LLM_TIMEOUT` | 300 秒 | runtime，不进 extraction 指纹 |
| `MEMORY_LLM_TEMPERATURE` | 0.0 | extraction |
| `MEMORY_LLM_MAX_TOKENS` | 16384 | extraction |
| `MEMORY_OPENAI_LLM_EXTRA_BODY` | 未设置，JSON object | extraction |
| `MEMORY_STRUCTURED_OUTPUT_MODE` | `json_schema` | extraction |
| `MEMORY_EMBEDDING_BINDING_HOST` | 主 embedding host | extraction |
| `MEMORY_EMBEDDING_BINDING_API_KEY` | 主 embedding key | secret |
| `MEMORY_EMBEDDING_MODEL` | 主 embedding model | extraction |
| `MEMORY_EMBEDDING_DIM` | 主 embedding dim | extraction |

服务在环境未显式设置时令 `GRAPHITI_TELEMETRY_ENABLED=false`。

### 9.3 Admission、read 与 prompt
| 配置 | 默认 | 类别 |
|---|---:|---|
| `LIGHTRAG_CHAT_MEMORY_STORE_RAW_EPISODE_CONTENT` | false | extraction |
| `MEMORY_INGEST_MAX_CHARS` | 6000 | extraction |
| admission/record/snapshot policy versions | `1` | extraction；非环境变量 |
| `MEMORY_SEARCH_LIMIT` | 10 | read runtime，限制 1..50 |
| `MEMORY_RERANK_ENABLED` | false | read runtime |
| `LIGHTRAG_CHAT_MEMORY_PROMPT_MAX_TOKENS` | 1024 | runtime render |
| `LIGHTRAG_CHAT_MEMORY_PROMPT_MAX_CHARS` | 8192 | runtime render |
| `LIGHTRAG_CHAT_MEMORY_ALLOW_CROSS_PROVIDER_QUERY_EGRESS` | false | runtime policy |
| memory query max length | 4096 字符 | API policy，非环境变量 |
| final framing reserve | 64 tokens | request budget，非环境变量 |

### 9.4 Worker、claim 与 hard caps
| 配置 | 默认 | 作用 |
|---|---:|---|
| `MEMORY_INGEST_CONCURRENCY` | 2 | 不同组 worker 并发 |
| `MEMORY_MAX_COROUTINES` | 4 | Graphiti 内部并发 |
| `LIGHTRAG_CHAT_MEMORY_WORKER_POLL_SECONDS` | 1.0 秒 | idle poll；别名 `MEMORY_WORKER_POLL_SECONDS` |
| `LIGHTRAG_CHAT_MEMORY_WORKER_RECOVERY_INTERVAL_SECONDS` | 30.0 秒 | stale recovery；别名 `MEMORY_WORKER_RECOVERY_INTERVAL_SECONDS` |
| `LIGHTRAG_CHAT_MEMORY_WORKER_SIDE_EFFECT_TIMEOUT_SECONDS` | 900.0 秒 | Graphiti timeout；默认 stale 候选阈值 |
| `LIGHTRAG_CHAT_MEMORY_WORKER_SHUTDOWN_TIMEOUT_SECONDS` | 10.0 秒 | drain/cancel 上限 |
| `LIGHTRAG_CHAT_MEMORY_REBUILD_MAX_MESSAGES` | 10000 | rebuild message hard cap |
| `LIGHTRAG_CHAT_MEMORY_REBUILD_MAX_BYTES` | 67108864 | rebuild JSON bytes hard cap（64 MiB） |
| known-failure max attempts | 3 | worker 构造器默认，非环境变量 |
| retry delay | 1.0 秒 | worker 构造器默认，非环境变量 |
| claim ownership | token + owner + session advisory guard | 无“超时即授权第二 writer”的独立 lease 配置 |

side-effect timeout 还接受旧 operation/backend timeout 兼容别名；规范名称优先。

仍被解析但不承担当前 durable 可靠性的兼容参数：

| 配置 | 默认 |
|---|---:|
| `MEMORY_INGEST_MODE` | `immediate` |
| `MEMORY_INGEST_DEBOUNCE_SECONDS` | 20 |
| `MEMORY_BACKLOG_SCAN_ON_START` | true |
| `MEMORY_BACKLOG_BATCH_MESSAGES` | 20 |
| `MEMORY_MAX_INFLIGHT_PER_USER` | 8 |

企业 server 已关闭服务内旧调度入口。当前可靠性来自 PostgreSQL outbox 和 `ChatMemoryWorker`；管理员 backlog scan 也不是扫描全部聊天消息生成 ingest batches。

### 9.5 指纹分类
**Graph-store fingerprint**：fingerprint version、Neo4j provider、deployment identity、database。现有逻辑组内不可变。

**Extraction fingerprint**：`graphiti-core` 版本、record/admission/snapshot policy、raw storage、LLM model/small model/endpoint/temperature/max tokens/structured mode/extra body、embedding model/endpoint/dimension、`MEMORY_INGEST_MAX_CHARS`。不含 API key/password。可在同一 graph store 内通过新 generation rebuild 升级。

**Runtime render/operational**：prompt caps、egress override、search limit/rerank、worker poll/recovery/shutdown/concurrency、provider timeout、rebuild hard caps。它们不代表图内容身份，不能与 extraction fingerprint 混用。代码中的 `runtime_fingerprint()` 只是 `extraction_fingerprint()` 的兼容别名，并不包含这些 prompt/egress/render 设置。

---

## 10. 生命周期、健康和管理 API

### 10.1 启动与停机
应用创建和 lifespan 启动都会校验前置条件。企业 + PostgreSQL 且 `enabled` 或 `maintenance_enabled` 时，服务构建指纹、接入 memory-aware CRUD、创建 maintenance service 和 worker。

`enabled=true`：开放 read/injection，启动时尝试 eager Graphiti initialize；失败记录 warning，后续按 typed availability 处理；worker 处理 ingest/rebuild/purge。

`enabled=false && maintenance_enabled=true`：不开放 read/injection、不创建新 admission ingest，worker 只处理 rebuild/purge，以完成关闭期间的删除收敛。

停机：先 drain/停止 worker，超过 shutdown timeout 后取消剩余任务；再关闭 backend slots；活跃 lease 归零后才 close client。

### 10.2 健康与状态
`GET /health` 的 `chat_memory` 当前包含：`enabled`、`available`、`pending_tasks`、`worker_running`、缩短后的 `extraction_fingerprint` 和 `graph_store_fingerprint`。

`pending_tasks` 是本地辅助任务，不是 durable outbox backlog；后者应读取 backlog endpoint。

`GET /chat/projects/{project_id}/memory` 返回 `project_id`、`enabled`、`available`、`episode_count`、`last_ingested_at`；计数和最后摄取时间来自 PostgreSQL episode mapping/control plane。

`GET /admin/overview` 向超级管理员提供 `enabled`、`available`、`pending_tasks`、全局 episode/user/project 计数。

### 10.3 管理 API
| API | 作用 |
|---|---|
| `POST /admin/users/{user_id}/chat-memory:purge` | 为指定/全部项目 durable enqueue purge 并 nudge |
| `POST /admin/chat-memory/events/{event_id}:retry` | 重试 durable purge event，即使 source 已删除 |
| `POST /admin/chat-memory:backlog-scan` | 恢复 stale claims、nudge worker、返回 outbox stats |

purge body 可省略或传 `{"project_ids":["..."]}`；省略/空列表枚举目标用户当前仍存在的全部 chat 项目；显式列表逐项校验归属。源记录已删除后的既有 purge 不通过该入口重建，而应使用 event-id retry。接口不等待 Neo4j 同步完成。

backlog-scan body 可传 `{"limit":100}`，范围 1..1000。当前行为：查找超过 stale threshold 的 running claims；逐个非阻塞获取 group guard；guard 忙则跳过；获得 guard 后按 side-effect state 安全恢复；nudge worker；返回 recovered count 和 pending/running/retry_wait/dead_letter、oldest available/blocked age 等统计。

worker 也按 `WORKER_RECOVERY_INTERVAL_SECONDS` 周期执行相同 stale recovery。

---

## 11. 测试证据与生产门禁

### 11.1 当前代码门禁证据
当前 remediation 最终本地记录：

```text
Durable Chat Memory suite: 241 passed, 101 skipped
Query/Agent suite:         226 passed
Prior expanded suite:      308 passed, 19 warnings
Oracle Phase 4B:           PASS
```

覆盖：事务 mutation、FIFO claim、跨进程 group guard、claim fencing、retry/dead-letter、stale recovery、未知结果前滚、snapshot/hard caps、clear coverage、graph-store identity、read fence/backend lease、Query/Agent late resolution、完整请求预算、egress、cache bypass、日志/trace/stream sanitization、无 memory 兼容、管理员 purge/retry/backlog。

关键测试文件：

- `tests/api/test_chat_memory_store_phase1.py`；
- `tests/api/test_chat_memory_store_phase2a.py`；
- `tests/api/test_chat_memory_store_phase2b.py`；
- `tests/api/test_chat_memory_worker.py`；
- `tests/api/test_chat_memory_phase4a.py`；
- `tests/api/test_chat_memory_crud_wiring.py`；
- `tests/api/test_chat_memory_service.py`；
- `tests/api/test_chat_memory_server_wiring.py`；
- `tests/api/routes/test_chat_memory_routes.py`；
- `tests/api/routes/test_chat_memory_injection.py`；
- `tests/api/routes/test_agent_memory_routes.py`；
- `tests/api/test_agent_chat_memory_service.py`；
- `tests/api/test_bilingual_chat_memory_service.py`；
- `tests/llm/test_sensitive_llm_scope.py` 及 provider sensitive cleanup tests。

### 11.2 已被取代的历史 NO-GO 门禁
> 本小节保留 live gate 执行前的原始发布判断，作为决策历史；它已被 2026-07-16 的部署级 live evidence 和 11.3 的 scoped GO 明确取代，不是当前 blanket status。

上述是代码门禁，不是真实基础设施验收。生产保持 **NO-GO**，直到三类证据全部完成：

#### A. 真实 PostgreSQL

- CRUD/outbox 同事务回滚；
- 多进程 append/delete/user purge 锁顺序；
- 真实 `SKIP LOCKED` FIFO claim；
- advisory guard 在真实连接池中的持有/释放；
- crash/断链/超时 stale recovery；
- source deletion 后 event-id purge retry；
- 目标 PostgreSQL 版本的 schema migration/index 行为。

#### B. 真实 Graphiti 0.29.2 + Neo4j

- 真实 add episode、抽取和 mapping；
- `store_raw_episode_content=false` 的实际图行为；
- 显式 physical group search/clear 与 current-fact filter；
- generation rebuild、旧组清除和 activation fence；
- timeout/cancel/断链后的未知结果前滚；
- purge 对 active/building/retired/abandoned/legacy/orphan universe 的覆盖；
- backend invalidation 与 active lease drain。

#### C. 真实 provider egress 与敏感 streaming

- endpoint identity/default deny/显式 override；
- memory fact 只发送给获准 final provider；
- prompt/response/fact 不进入应用日志、SDK debug、trace/APM 或 cache；
- 实际 provider 的同步/流式异常无内容泄漏；
- 客户端中断、iterator 异常和取消保持敏感作用域；
- 无 memory 响应与流字节不回归。

三个 gate 任一缺失，都不能解除生产 NO-GO。

### 11.3 2026-07-16 真实部署验证
**Production gate: GO for the verified configured deployment as of 2026-07-16.**

| Gate | 2026-07-16 live evidence | 结论 |
|---|---|---|
| 基础连接 | PostgreSQL 15 与 Neo4j connectivity 均通过 | **PASS** |
| 真实 PostgreSQL | 在隔离临时数据库运行真实 PostgreSQL suite：**223 passed，1 个预期 SQLite-only skip**；临时数据库已 DROP，residue count 为 **0** | **PASS** |
| 真实 Graphiti/Neo4j | `graphiti-core 0.29.2` + 已配置 Neo4j + 已配置 extraction LLM/embedding + 临时 PostgreSQL：post-write unknown outcome 后原事件 superseded、generation 1 abandoned，自动 generation 2 rebuild 并 active；旧 physical group 为空；确定性 canary search 通过 | **PASS** |
| 正常写与检索 | 在 active generation 正常 append 成功；Neo4j Episodic node 增长；canary 可搜索且 `exact-match=true` | **PASS** |
| Durable purge | purge event 成功；logical group state 为 deleted；search 为空；相关 physical Neo4j node/relationship count 均为 0；临时数据库已 DROP | **PASS** |
| 真实 Query provider 敏感流 | same-endpoint egress 被接受；收到真实 stream chunk；可 instrument/trace 的 `AsyncOpenAI` 被 forbidden sentinel 替换且未使用；DEBUG/VERBOSE logs、stdout、stderr 均无 private canary；early close cleanup 在 sensitive scope 内执行且 context 已恢复 | **PASS** |
| Oracle 最终判定 | 对本次配置部署 **GO**，无剩余技术性 Chat Memory blocker | **PASS（scoped）** |

Graphiti live gate 的第一次尝试只因 probe 要求随机第二 token 被逐字保留而失败；该轮 cleanup 成功。修正为确定性判据后通过，后续重跑也观察到 `exact-match=true`。这不是 Graphiti 状态机或清理失败。

### 11.4 GO 的边界与重新验证条件

- **部署范围**：当前 working tree、已验证 PostgreSQL 15 实例、Neo4j database/deployment、Graphiti 0.29.2、extraction/query endpoints 与 models、embedding endpoint/model/dimension、same-endpoint egress policy。
- **不可变制品**：working tree 仍未提交；部署前必须记录 commit/artifact digest、配置/fingerprints 和证据位置，并由 release owner 签字。未完成此发布治理步骤时，不应把本次 live run 归因到某个不可变制品。
- **Langfuse**：本次未配置，状态为 **N/A**，不能写成 live-passed；mocked trace tests 仍通过。未来启用 Langfuse 时必须重跑真实 trace 泄漏 gate。
- **Cross-provider egress**：未获得 live qualification；默认保持 deny。若要启用，必须先取得 data-residency/egress 审批，再用实际两端 provider 重跑 live 验证。
- **故障覆盖**：provider exception/cancellation、cache bypass 和细粒度 race 仍主要由 deterministic mock/fault-injection 证明；真实基础设施本轮覆盖 success、Graphiti post-write unknown outcome 和 stream early close。
- **范围排除**：Docker 与单用户/local mode 仍不在本次 GO 范围。
- **失效条件**：改变代码 artifact，或改变 Neo4j deployment/database、Graphiti 版本、LLM/embedding/query endpoint/model、embedding dimension、egress policy 及相关 graph/extraction/runtime fingerprint 时，必须重跑相应 gate。

---

## 12. 最小运维检查单

- [x] 企业认证开启，真实 PostgreSQL 15 gate 已通过；
- [x] `graphiti-core==0.29.2` 与当前 Neo4j deployment/database live gate 已通过；
- [ ] graph-store 与 extraction fingerprint 在变更流程中分别管理；
- [ ] raw episode storage 保持默认关闭，或已完成隐私评审；
- [ ] assistant admission 只由可信服务端写入布尔 `true`；
- [x] same-endpoint egress 与真实敏感 stream/early-close gate 已通过；
- [ ] cross-provider egress 保持默认 deny；若拟启用，先审批并重新 live 验证；
- [ ] worker poll/recovery/timeout/shutdown/hard caps 已压测；
- [ ] `/health` 正常，backlog 无持续 blocked/dead-letter 增长；
- [ ] event-id purge retry runbook 已演练；
- [x] 当前配置的真实 PostgreSQL、Graphiti/Neo4j、same-endpoint provider/streaming 三个 gate 已通过；
- [x] Oracle 已给出当前配置部署 scoped GO；
- [ ] 将当前未提交 working tree 固化为不可变 commit/artifact，附证据与 fingerprints；
- [ ] release owner 完成部署签字；
- [ ] 若以后启用 Langfuse，重跑 live trace gate（当前为 N/A）；
- [ ] 任一相关代码、provider、基础设施或配置 fingerprint 改变后重跑 gate。

当前结论是：**Production gate: GO for the verified configured deployment as of 2026-07-16.** 该技术 gate 已关闭，但发布前仍须把未提交 working tree 与 live evidence 固定到不可变制品并完成 release-owner sign-off。
