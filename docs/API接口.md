# LightRAG API 接口文档

> 文档版本：2026-06-10
> 文档定位：LightRAG 生产级 KB 后端与企业能力的**单一权威接口契约**（取代 `archive/API接口文档.md`，内容无损延续）。
> 适用范围：当前已经合并到 `main` 分支并通过测试的接口。
> 部署前提：单台服务器、内网使用；所有知识库共享同一组本地部署的模型与解析服务（LLM / VLM / Embedding / Rerank / MinerU / Docling），部署级服务参数统一由 `.env` / 部署编排管理；模型本地部署，无 token/cost 计费。
> 路径前缀：所有路径均为相对路径；部署时通过 FastAPI `root_path` 或 `--api-prefix /api/v1` 暴露为 `/api/v1/...`。
> 鉴权：除 `/health`、`/auth-status`、`/login` 等少数公开接口外，所有接口都受 `combined_auth` 依赖保护，需要在请求头携带 `X-API-Key: <api_key>` 或 JWT。企业模式启用后，`/kbs`、legacy `/documents`/`/query`/`/graph`、Ollama `/api/*` 会额外受企业 RBAC / anti-bypass 策略约束。
> 配套文档：架构与功能状态见 [`docs/设计方案.md`](设计方案.md)；KB 配置字段速查见 [`docs/KB配置项速查表.md`](KB配置项速查表.md)；备份恢复见 [`docs/生产级后端备份恢复Runbook.md`](生产级后端备份恢复Runbook.md)。

---

## 目录

- [一、知识库管理 KB](#一知识库管理-kb)
- [二、知识库文档 Documents](#二知识库文档-documents)
- [三、知识库解析 Parse](#三知识库解析-parse)
- [四、知识库构建 Index / KG](#四知识库构建-index--kg)
- [五、知识库任务 Jobs](#五知识库任务-jobs)
- [六、知识库产物 Artifacts](#六知识库产物-artifacts)
- [七、知识库配置版本 Config Versions](#七知识库配置版本-config-versions)
- [八、知识库问答 Query](#八知识库问答-query)
- [九、兼容旧版 / 全局接口](#九兼容旧版--全局接口)
- [十、企业模式 Auth / Admin](#十企业模式-auth--admin)
- [十一、状态机与字段说明](#十一状态机与字段说明)
- [十二、生产存储配置](#十二生产存储配置)

---

## 一、知识库管理 KB

> 知识库是所有 KB 接口的边界。`kb_id` 派生出 LightRAG 的 `workspace`，并由 `LightRAGInstanceRegistry` 按需懒加载实例。KB 控制面 metadata 默认使用 `WORKING_DIR/metadata/knowledge_bases.json` + `metadata.sqlite3`；设置 `LIGHTRAG_KB_METADATA_BACKEND=postgres` 后改用 PostgreSQL catalog/doc/job/artifact/config-version 表。设置 `LIGHTRAG_OBJECT_STORAGE=minio|s3` 后，上传源文件与解析产物会同步持久化到对象存储，`INPUT_DIR` 仍作为本地 cache。

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/kbs` | 创建知识库 |
| `GET` | `/kbs` | 列出所有知识库 |
| `GET` | `/kbs/{kb_id}` | 获取知识库详情 |
| `PATCH` | `/kbs/{kb_id}` | 局部更新知识库（名称、描述、状态等） |
| `DELETE` | `/kbs/{kb_id}` | 软删除知识库；附加 `?hard=true` 触发硬删除（durable worker 启用时入队 `clear_kb`，否则同步执行） |
| `GET` | `/kbs/{kb_id}/status` | 知识库状态聚合（含运行中任务、pipeline 状态） |
| `POST` | `/kbs/{kb_id}:restore` | 恢复软删除的知识库（`deleted`→`active`）；企业模式允许 super admin 或该 tenant-created KB 所属租户的 `tenant_admin` / `tenant_owner` |
| `GET` | `/kbs/{kb_id}/stats` | 控制面统计：文档状态分布、chunks 合计、entity/relation 数（取自图谱）、job 状态分布、dead-letter、artifact 数、graph 节点/边数 |

### 1.1 创建知识库

```http
POST /kbs
Content-Type: application/json

{
  "id": "kb_research",            // 可选，省略由服务端生成 kb_<12位hex>
  "name": "Research Papers",      // 必填，去首尾空白后非空
  "description": "Optional",       // 可选
  "owner_id": null,                // 默认模式下为兼容 metadata；企业模式会忽略并由当前 principal 派生
  "tenant_id": null,               // 默认模式下为兼容 metadata；企业模式会忽略并由当前 principal 派生
  "visibility": "private",         // 枚举：private / internal / public；企业模式下 internal=同租户隐含只读、public=全员隐含只读（语义见 10.4），写权限仍以 KB ACL 为准；租户用户创建时仅可选 private / internal（public 返回 400）
  "metadata": {"tags": ["legal"]}  // 可选自由 dict（前端标签/分组/扩展字段），序列化 ≤16KB；响应与列表原样返回
}
```

返回 `200 KnowledgeBaseResponse`；冲突 `409`；参数非法 `400`。

### 1.2 列出 / 获取 / 更新 / 删除

- `GET /kbs?include_deleted=false`：默认排除软删除记录。
- `GET /kbs/{kb_id}`：404 表示未找到或已软删除。
- `PATCH /kbs/{kb_id}`：仅更新请求体显式给出的字段；`status` 不允许直接置为 `deleted`；`active_config_version_id` 不能通过 PATCH 修改，若请求体包含该字段返回 `400`，请改用 `POST /kbs/{kb_id}/configs/{version_id}:activate`。`metadata` 为**合并**语义：给出的 key 覆盖现值、value=null 删除该 key、未提及的 key 保留；顶层 `metadata: null` 返回 `400`；合并后序列化超 16KB 返回 `400`。
- Agent 选库可使用 KB `metadata` 中的 profile 字段。人工覆盖字段为 `agent_description`（字符串，面向 Agent 的知识库说明）、`agent_tags`（字符串数组，或逗号分隔字符串）、`agent_priority`（整数，默认 0）；自动字段为 `agent_auto_profile`，由 `PROFILE` 角色 LLM 基于文档级 `metadata.agent_doc_profile` 聚合生成。人工字段优先于自动字段；这些字段不会改变 RBAC，只会随授权 KB 的 `allowed_kbs` 注入 `/agent/query` 的规划上下文。
- `DELETE /kbs/{kb_id}`：默认软删除，同步从 `LightRAGInstanceRegistry` 卸载实例。
- `DELETE /kbs/{kb_id}?hard=true`：触发硬删除。若服务端启用 durable worker 且 `clear_kb` 在 `job_worker.resumable_job_types` 中，路由会先 soft-delete KB，再创建按 `(kb_id, generation)` 唯一的 queued `clear_kb` job；payload 固定保存 `kb_generation` 与 `workspace`，旧 generation 的 job 不能清理同 ID 重建后的新 KB。响应 `KnowledgeBaseDeleteResponse` 包含 `hard_delete_queued=true`、`hard_delete_job_id`、`hard_delete_job_type="clear_kb"`、`hard_delete_job_status="queued"`；后续由 worker 通过 `resume_hard_delete` 幂等执行。job 查询/取消/重试对 soft-deleted KB 使用 `include_deleted=true`，因此 tombstone 后仍可观察和控制。若 durable worker 未启用，则由同一个 durable clear job 同步执行。
- 所有普通 KB 写操作在 generation-scoped shared fence 内执行；hard-delete 持 exclusive fence。PostgreSQL 使用 session advisory lock，local 兼容后端使用 KB 文件锁。删除会等待已开始的写入完成；进入 `deleting` 后拒绝新的 mutation/job，不允许恢复或复用该 KB id。
- `KBDeletionService` 在 exclusive fence 内重新校验 catalog 状态、generation 与 clear job 后，按以下顺序执行：
  1. `force_evict` 在内存中的 LightRAG 实例并调用 `finalize_storages`（关闭存储句柄，不删数据）；
  2. **drop 全部引擎 storage 数据**：用 registry builder 建一个未缓存的瞬时实例并调用 `LightRAG.adrop_all_storages()`，对 full_docs / text_chunks / entities / relations / chunks / vector / graph / doc_status / llm_cache 等全部 storage 调 `drop()`。下一步删 `working_dir` 只能清文件型后端，外部后端（PostgreSQL / Milvus / Neo4j / Qdrant / Redis / Mongo / OpenSearch）数据在远端服务里，必须经此步显式清除，否则会残留并被复用同 workspace 的新 KB 读到；
  3. 删除 `working_dir/<workspace>`（如已配置）；
  4. 删除 `input_dir/<workspace>`（上传文件 + 解析 artifact 的本地 cache）；
  5. 若启用对象存储，删除该 workspace 下的 source/artifact 对象；
  6. 严格清空 metadata scope（documents / artifacts / config versions / ACL / tenant override / service-key KB scope 等），但保留当前 clear job；
  7. 将 lifecycle 从 `deleting` 原子提交为 `deleted`，最后按 generation CAS 删除 catalog 行并把 clear job 置为 `succeeded`。
  `result` 包含 `dropped_storages`（成功 drop 的 storage 数）、`cleared_object_storage`、`deleted_objects` 与 metadata purge 计数。任一物理清理失败都会把同一 clear job 标为 `failed`，但保留 catalog、metadata 与 `deleting` fence，禁止复用 ID；再次 hard-delete 或 `:retry` 会复用原 job/idempotency key。若进程在 catalog purge 后、job 终态提交前退出，orphan recovery 会把该 `clear_kb` 原行重新排队，由 worker 只完成安全的尾部收敛。

企业模式（`LIGHTRAG_ENTERPRISE_AUTH_ENABLED=true`）下：

- `POST /kbs` 需要 super admin、`tenant_admin` / `tenant_owner`，或 `can_create_kb=true`。非 super admin 的 `owner_id`/`tenant_id` 由当前 principal 派生并自动授予创建者 `kb_owner` ACL；租户用户创建的 KB 固定 `origin="tenant"`，`visibility` 可选 `private`（默认，仅创建者与显式授权可见）或 `internal`（共享：同租户成员隐含只读），`public` 返回 `400`；并规范化加入 `tenant:{tenant_id}` 标签。
- `GET /kbs` 对普通用户返回已授权 KB（direct user ACL / tenant ACL）以及 visibility 命中的 KB（`public` / 同租户 `internal`，见 10.4）；`tenant_admin` / `tenant_owner` 额外**始终**可见本租户成员创建（`origin="tenant"`、同 `tenant_id`）的全部 KB——包括 `private`（隐含只读 oversight，见 10.4）；super admin 返回全部；service key 仅按 `kb_roles` scope（可选显式 `inherit_tenant_kb_acl`），不受 visibility 影响。
- `PATCH /kbs/{kb_id}` 忽略非 super admin 请求体中的 `owner_id`/`tenant_id`。visibility 修改规则：super admin 可改任意 KB 为任意值；租户创建的 KB（`origin="tenant"`）允许 effective `kb_owner`（通常为创建者）在 `private` 与 `internal` 之间**随时切换**（改 `public` 返回 `400`，非 owner 返回 `403`）；platform KB 的 visibility 仍仅 super admin 可改。visibility 实际变化时写入 `kb_visibility_changed` 审计事件（metadata 含 `from`/`to`/`origin`）。租户 KB 的 tenant 标签与不可变 `origin` 不能由 metadata 伪造。
- `DELETE /kbs/{kb_id}`、`?hard=true` 与 `POST /kbs/{kb_id}:restore`：super admin 可操作任意 KB；目标 KB 必须为 `origin="tenant"` 且 `tenant_id` 等于当前 canonical tenant 时，该租户的 `tenant_admin` / `tenant_owner` 也可操作。tenant ACL、direct KB admin/owner 或可编辑 metadata 都不能获得 platform KB 的生命周期权限。缺失 `origin` 的历史 catalog 行安全地按 `platform` 处理。

### 1.3 知识库状态

```http
GET /kbs/{kb_id}/status
```

返回字段：

```json
{
  "kb": { /* KnowledgeBaseResponse */ },
  "instance_loaded": true,           // 该 KB 是否已经在内存中加载 LightRAG 实例
  "pipeline_initialized": true,      // 该 workspace 的 pipeline_status 是否已初始化
  "pipeline_status": { /* 运行时状态副本 */ },
  "storage_workspaces": {            // 已加载实例时各 storage 的 workspace
    "full_docs": "kb_research",
    "text_chunks": "kb_research",
    "...": "..."
  },
  "running_jobs": [ /* 状态为 queued/running/retrying/cancelling 的任务 */ ]
}
```

### 1.4 恢复软删除的知识库

```http
POST /kbs/{kb_id}:restore
```

- 仅对 `status="deleted"` 的软删除 KB 生效：恢复为 `active`、清空 `deleted_at`，返回 `KnowledgeBaseResponse`。
- KB 不存在返回 `404`；KB 当前不是 deleted 状态返回 `409`。
- 存在在途（queued/running/retrying/cancelling）`clear_kb` 硬删除任务时返回 `409`，`detail.error_code="kb_hard_delete_in_progress"` 并携带 `job_id`——此时数据即将被硬删 worker 清除，恢复无意义；硬删除完成后控制面已 purge，`:restore` 返回 `404`。
- 企业模式下由同一 catalog provenance 规则授权：super admin 可恢复任意 KB；所属租户的 `tenant_admin` / `tenant_owner` 仅可恢复真正的 `origin="tenant"` KB。restore 在 shared fence 内完成，若 hard-delete 已进入 `deleting` 或存在未完成的 generation-bound clear job 则返回 `409`；成功写入 `kb_restored` 审计事件。

### 1.5 知识库控制面统计

```http
GET /kbs/{kb_id}/stats
```

返回字段：

```json
{
  "kb_id": "kb_research",
  "documents": {"total": 12, "by_status": {"ready": 10, "parse_failed": 2}},
  "counters": {"chunks": 340, "entities": 1200, "relations": 980},
  "jobs": {"total": 25, "by_status": {"succeeded": 23, "failed": 2}, "dead_letter": 1},
  "artifacts": {"total": 96},
  "graph": {"node_count": 1200, "edge_count": 980}
}
```

- 读控制面 metadata store 取文档/job/artifact 统计；实体/关系数取自该 KB 的图谱（按需加载 LightRAG 实例做一次有界全图扫描，与 `GET /kbs/{kb_id}/graph/status` 同源，受 `max_nodes_scanned` 上限保护）。LightRAG 的 `doc_status` 行不保存 entity/relation 计数（表无此列、构建只写 `chunks_count`），所以控制面合计恒为 0，故用图谱规模回填 `counters.entities`/`counters.relations`；`graph` 另外携带原始 `node_count`/`edge_count` 供区分。
- 已删除文档的计数在删除时已清零，不计入 `chunks` 合计。
- `dead_letter` 为 `failed` 且重试耗尽的任务数（与 `/jobs/dead-letter` 口径一致）。
- 企业模式 `kb_viewer`+ 可读；KB 不存在返回 `404`。

---

## 二、知识库文档 Documents

> 文档生命周期由 `DocumentLifecycleService` 管理，元数据落 metadata store（local 模式为 `working_dir/metadata/metadata.sqlite3`，PostgreSQL 模式为 `kb_documents` / `kb_jobs` / `kb_document_artifacts` 等表）。同名文件会写入独立的 `INPUT_DIR/<workspace>/<document_id>/` 子目录，跨进程并发写不会互相覆盖。启用 `LIGHTRAG_OBJECT_STORAGE=minio|s3` 后，源文件会同步持久化到对象存储并在 `metadata.source_object_uri` 记录对象 URI；本地 `source_uri` 保留为 cache path。

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/kbs/{kb_id}/documents:upload` | 多文件上传，可选自动解析 |
| `POST` | `/kbs/{kb_id}/documents:sync` | 按 `source_key` 批量增量同步，可自动解析并构建到可问答状态 |
| `POST` | `/kbs/{kb_id}/documents:texts` | 批量文本导入 |
| `POST` | `/kbs/{kb_id}/documents:urls` | 批量 URL 抓取导入（SSRF/大小受限） |
| `POST` | `/kbs/{kb_id}/documents:import` | 从受控 `INPUT_DIR` staged 文件导入 |
| `POST` | `/kbs/{kb_id}/documents:scan` | 扫描受控 `INPUT_DIR` staged 子目录导入 |
| `GET` | `/kbs/{kb_id}/documents` | 文档列表，支持状态、文件名过滤 |
| `GET` | `/kbs/{kb_id}/documents/{document_id}` | 文档详情 |
| `PATCH` | `/kbs/{kb_id}/documents/{document_id}` | 更新 metadata / enabled / archived |
| `POST` | `/kbs/{kb_id}/documents/{document_id}:disable` | 独立禁用文档（仅控制面 metadata） |
| `POST` | `/kbs/{kb_id}/documents/{document_id}:enable` | 独立启用文档（仅控制面 metadata） |
| `DELETE` | `/kbs/{kb_id}/documents/{document_id}` | 单文档任务化删除 |
| `POST` | `/kbs/{kb_id}/documents/{document_id}:replace` | 单文档任务化替换 |
| `POST` | `/kbs/{kb_id}/documents:batch-delete` | 批量任务化删除 |
| `POST` | `/kbs/{kb_id}/documents:batch-enable` | 批量启用文档（同步 metadata 操作，per-item 结果） |
| `POST` | `/kbs/{kb_id}/documents:batch-disable` | 批量禁用文档（同步 metadata 操作，per-item 结果） |
| `GET` | `/kbs/{kb_id}/documents/{document_id}/chunks` | 查看该文档构建出的引擎 text chunks（分页，检索可解释性） |

### 2.1 多文件上传

```http
POST /kbs/{kb_id}/documents:upload?auto_parse=true&auto_index=false&parser_engine=mineru&process_options=iF&idempotency_key=upload-001
Content-Type: multipart/form-data

files: [a.pdf, b.docx]
```

约束：
- 单请求最多 32 个文件，单文件和单请求总字节数均不得超过 `MAX_UPLOAD_SIZE`，未配置或非正数时 `413`。
- 文件扩展名必须在 `SUPPORTED_DOCUMENT_EXTENSIONS` 列表中。
- `auto_parse=true` 会创建一个 `job_type=parse` 的聚合任务（`document_id=null`、`batch_id` 非空、payload 携带 `document_ids` 列表），并在后台**并发执行解析**（Phase 1 受 `MAX_PARALLEL_PARSE_MINERU` 并发上限约束，每个文档 `parse_queued → parsing → parsed`），结果聚合进该 job 的 `result.items[]`；同时 `auto_index=true` 会在 Phase 2 把全部解析成功的文档**一次性批量入队、单次流水线 drain**（analyze/extract/merge 跨文档重叠）构建到 `ready`（需路由注入 `IndexBuildService`）。`auto_parse=false` 仅落 metadata，job 立即标记 `succeeded`。
  - 行为说明：`result.items[]` 不再保证与输入 `document_ids` 顺序一致（按完成情况聚合，但每个文档恰好出现一次）；单文档失败相互隔离，不影响其它文档继续。`:sync` 的并发模型与此一致。
- 注意：该聚合 parse/build 任务默认由创建请求的 in-process 后台任务立即执行；启用 `LIGHTRAG_KB_JOB_WORKER=true` 后，`queued` 且超过 `LIGHTRAG_KB_JOB_WORKER_GRACE_SECONDS` 的可恢复聚合任务可被 durable worker 认领，并凭 `payload.document_ids` 续跑。若未启用 durable worker，服务在执行中途重启仍需客户端重新触发或重试。`auto_parse=true` 且请求未显式传 `parser_engine/process_options` 时，会把当前 active `parser_config.engine/process_options` snapshot 到文档和 job metadata；`auto_parse=false` 不冻结这些默认值。
- 同名文件会写入独立的 `<workspace>/<document_id>/<filename>` 目录，使用独占创建 (`O_EXCL`)。
- 若启用对象存储，上传成功后每个 document 的 `metadata.source_object_uri` 为 `s3://<bucket>/<prefix>/workspaces/<workspace>/documents/<document_id>/source/<filename>`；本地文件仍保留，用于 parser/build/download。

返回 `DocumentBatchResponse`：

```json
{
  "job_id": "job_parse_xxx",
  "batch_id": "batch_xxx",
  "documents": [
    { "id": "doc_...", "status": "parse_queued", "source_uri": "...", "...": "..." }
  ]
}
```

### 2.2 文本导入

```http
POST /kbs/{kb_id}/documents:texts
Content-Type: application/json

{
  "documents": [
    { "text": "正文内容", "source_name": "note.md", "metadata": {"tag": "unit"} }
  ],
  "auto_parse": false,
  "auto_index": false,
  "parser_engine": null,
  "process_options": null,
  "idempotency_key": "text-import-001"
}
```

约束：
- 单文档文本上限 1 MB，单 metadata JSON 上限 64 KB。
- 单请求最多 100 个文本。
- `idempotency_key` 在 `(kb_id, job_type)` 维度唯一；指纹一致直接返回原 batch；指纹不一致返回 `409`。
- `auto_parse=true` 与多文件 `:upload` 一致：创建 `job_type=parse` 聚合任务并在后台**并发解析**（受 `MAX_PARALLEL_PARSE_MINERU` 并发上限约束）；`auto_index=true` 在 Phase 2 把全部解析成功的文档**一次性批量入队、单次流水线 drain** 构建到 `ready`（需注入 `IndexBuildService`）。该聚合任务默认由 in-process 后台任务执行；启用 `LIGHTRAG_KB_JOB_WORKER=true` 后，`queued` 且超过 grace window 的可恢复聚合 parse job 可由 durable worker 认领并续跑（续跑同样走并发解析 + 单次 drain）。请求未显式传 `parser_engine/process_options` 时，会 snapshot 当前 active `parser_config.engine/process_options` 作为解析默认值；`auto_parse=false` 不冻结这些默认值。

### 2.3 批量增量同步

```http
POST /kbs/{kb_id}/documents:sync?auto_parse=true&auto_index=true&parser_engine=mineru&process_options=iF&force_reparse=false&delete_source_file=true&delete_artifacts=true&delete_llm_cache=false&idempotency_key=sync-001
Content-Type: multipart/form-data

files: [a.pdf, b.pdf]
source_keys: ["manual/a.pdf", "manual/b.pdf"]
```

行为：
- `source_key` 是生产增量同步的稳定业务身份，同一 KB 内用它判断同一份外部文档；建议使用对象存储 key、相对路径或外部系统 document id，不要只用展示文件名。
- `source_key` 在同一 KB 内由 metadata store 原子唯一约束；并发 sync 不会为同一个外部文档创建两个活动 KB 文档。
- 服务端先读取文件内容并计算 `source_hash`，再查找相同 `source_key` 的现有文档。
- 找不到 `source_key`：创建新文档；`source_hash` 相同：跳过 source 替换，但若当前请求的 `parser_engine/process_options` 派生出的 `parser_hash` 与文档上次成功解析的值不同，仍会重解析并继续重建；`source_hash` 不同：复用单文档 replace 语义，保留原 `document_id`，先删除旧 `lightrag_doc_id` 后替换 source。
- `auto_parse=true` 默认继续解析（Phase 1 受 `MAX_PARALLEL_PARSE_MINERU` 约束**并发解析**）；请求未显式传 `parser_engine/process_options` 时，会按当前 active `parser_config.engine/process_options` 作为解析默认值；`auto_index=true` 默认在 Phase 2 把全部解析成功的文档**一次性批量入队、单次流水线 drain** 构建到 `ready` 并可直接走 KB query。
- `force_reparse=true` 会绕过现有 parse cache 重新解析；`delete_source_file` / `delete_artifacts` / `delete_llm_cache` 控制 changed source 走 replace 语义时的旧 source、artifact 和 LLM cache 清理策略。
- 返回单个聚合 `sync` job。每个 item 在 `job.result.items[]` 中记录 `source_key`、`action`（`created` / `replaced` / `skipped` / `reparsed`）、`status`、`parse_result`、`build_result` 等；`result.items[]` 不再保证与输入顺序一致（按完成情况聚合，每个 `source_key` 恰好出现一次）；单个 item 失败不会阻塞其他 item，active parse/build/delete/replace 会保留对应 `*_job_active` 错误码和 `existing_job_id`。
- `idempotency_key` 在 `(kb_id, job_type=sync)` 维度唯一；同 key 同文件和同参数复用原 job，同 key 不同请求返回 `409`。

### 2.4 URL / 本地 staged 文件 / 目录扫描导入

URL 导入：

```http
POST /kbs/{kb_id}/documents:urls
Content-Type: application/json

{
  "documents": [
    {
      "url": "https://example.com/paper.pdf",
      "source_name": "paper.pdf",
      "source_key": "url:https://example.com/paper.pdf",
      "content_type": "application/pdf",
      "metadata": {"tenant": "demo"}
    }
  ],
  "auto_parse": false,
  "auto_index": false,
  "parser_engine": null,
  "process_options": null,
  "idempotency_key": "url-import-001"
}
```

本地 staged 文件导入：

```http
POST /kbs/{kb_id}/documents:import
Content-Type: application/json

{
  "documents": [
    {"path": "staged/manual.pdf", "metadata": {"kind": "manual"}}
  ],
  "auto_parse": false,
  "auto_index": false,
  "idempotency_key": "local-import-001"
}
```

本地 staged 目录扫描：

```http
POST /kbs/{kb_id}/documents:scan
Content-Type: application/json

{
  "directory": "incoming-batch",
  "recursive": true,
  "source_key_prefix": "scan",
  "max_files": 32,
  "auto_parse": false,
  "auto_index": false,
  "idempotency_key": "scan-001"
}
```

约束：
- 三类接口最终都调用 `DocumentLifecycleService.create_source_batch`，落 KB `documents/jobs` metadata，并写入对应 `source_type=url/import/scan`；`auto_parse/auto_index/parser_engine/process_options/idempotency_key` 语义与 `:upload` / `:texts` 一致。`auto_parse=true` 时同一 `idempotency_key` + 同请求指纹会复用原聚合 parse batch/job，指纹不同返回 `409`。
- `:urls` 仅允许 `http/https`，URL 必须有 hostname，禁止 userinfo；请求前解析 hostname 并拒绝 loopback/private/link-local/multicast/reserved/unspecified 地址；`httpx.AsyncClient` 使用 `trust_env=false`、`follow_redirects=false`、显式 timeout，并在读取响应体前复验实际连接的 peer address，防止 DNS rebinding/解析 TOCTOU 读取内网响应。`Content-Length` 会预检，流式读取时仍按 `MAX_UPLOAD_SIZE` 和请求总字节数二次截断；3xx 不自动跟随。
- URL `source_name` 优先级：显式 `source_name` > `Content-Disposition` filename > URL path basename；最终扩展名必须在 `SUPPORTED_DOCUMENT_EXTENSIONS` 内。未显式传 `source_key` 时默认 `url:<normalized-url>`，metadata 会写入 `source_url`。
- `:import` 仅允许读取配置的 `INPUT_DIR` 下文件；绝对路径/相对路径都会规范化并做 containment 校验，逃逸 `INPUT_DIR`、目录、symlink、空文件、不支持扩展名或超限文件均拒绝。未显式传 `source_key` 时默认 `import:<relative-path>`，metadata 写入 `staged_source_path`。
- `:scan` 的 `directory` 必须是 `INPUT_DIR` 下 staged 子目录，不能是 `INPUT_DIR` 根；可递归扫描，跳过 `__parsed__` 与 `.sync-staging` 树，只导入支持扩展名的文件，最多 `max_files`（服务端硬上限 1000）。未显式设置时 `source_key` 为 `<source_key_prefix>:<relative-path>`，metadata 写入 `scanned_source_path`。

### 2.5 文档列表 / 详情

```http
GET /kbs/{kb_id}/documents?status=parsed&source_name=paper&limit=50&offset=0
GET /kbs/{kb_id}/documents/{document_id}
```

`source_name` 使用 SQL `LIKE` 模糊匹配（大小写不敏感）。

### 2.6 文档 PATCH

```http
PATCH /kbs/{kb_id}/documents/{document_id}
Content-Type: application/json

{
  "metadata": {"category": "review"},  // 与现有 metadata 合并
  "enabled": true,
  "archived": false
}
```

约束：
- 至少要给一个字段（空请求体返回 `400`）。
- `metadata` 中**不允许**覆盖内部控制面保留键（`batch_id` / `pending_parse_job_id` / `current_parse_job_id` / `pending_build_job_id` / `current_build_job_id` / `parser_engine` / `process_options` 等）。

### 2.7 独立启用 / 禁用

```http
POST /kbs/{kb_id}/documents/{document_id}:disable
POST /kbs/{kb_id}/documents/{document_id}:enable
```

返回 `DocumentResponse`。这两个动作只更新 metadata 控制面 `enabled` 字段，不删除 source/artifact，也不触发 LightRAG storage 变更。`enabled` 现已接入检索层：禁用文档会被排除出 `QueryParam.ids` 白名单，因此不再参与 KB 级问答检索（无需删除即可临时下线一篇文档）。

批量启停（前端多选场景）：

```http
POST /kbs/{kb_id}/documents:batch-enable
POST /kbs/{kb_id}/documents:batch-disable
Content-Type: application/json

{"document_ids": ["doc_a", "doc_b"]}
```

- 与单文档 `:enable`/`:disable` 同语义的同步控制面操作，不创建 job。
- `document_ids` 非空、≤100、不允许重复（重复返回 `422`）；KB 不存在返回 `404`。
- 响应：`{"enabled": false, "updated": 2, "not_found": 1, "items": [{"document_id": "...", "status": "updated" | "not_found"}]}`；缺失文档按 per-item `not_found` 报告，不阻塞其它文档；重复应用当前状态仍计 `updated`（幂等）。
- 企业模式 `kb_editor`+；审计 `document_batch_enabled` / `document_batch_disabled`。

### 2.7.1 文档 chunks 查看（检索可解释性）

```http
GET /kbs/{kb_id}/documents/{document_id}/chunks?limit=50&offset=0
```

- 经文档 `lightrag_doc_id` → 引擎 doc_status `chunks_list` → text_chunks 批量取回，按 `chunk_order_index` 升序返回；`limit` 默认 50（上限 200）。
- 尚未构建（无 `lightrag_doc_id`）的文档返回 `total=0` 空列表，且**不加载引擎实例**；已构建文档首次调用会按需加载该 KB 实例。
- 引擎中已被清理的 chunk 行自动跳过；`total` 为实际可取回的 chunk 数。
- 响应：`{"kb_id", "document_id", "lightrag_doc_id", "total", "limit", "offset", "chunks": [{"id", "chunk_order_index", "tokens", "content", "file_path"}]}`。
- 用途：核对 `chunk_config` 分块效果、排查"为什么没检索到"；企业模式 `kb_viewer`+。

### 2.8 文档删除

```http
DELETE /kbs/{kb_id}/documents/{document_id}?delete_source_file=false&delete_artifacts=false&delete_llm_cache=false&delete_graph_orphans=true&strategy=safe&idempotency_key=delete-001
```

行为：
- 创建 `delete` job，并将文档原子 claim 到 `deleting`；已有 `parse_queued/parsing`、`build_queued/building`、`deleting` 或 `replacing` 时返回 `409`。
- 若文档已有 `lightrag_doc_id`，后台任务调用 `LightRAG.adelete_by_doc_id`；底层返回 `success` 或 `not_found` 都视为删除成功，适配尚未入库或已被清理的文档。
- **共享图谱删除策略 `strategy`**（`safe` / `rebuild_doc_scope` / `rebuild_kb` / `rebuild_subgraph`，默认 `safe`）：`safe` 与 `rebuild_doc_scope` 复用 `adelete_by_doc_id` 内建的 source-attribution（按剩余来源判定）+ 共享实体保守重建，仅清除失去最后来源的实体/关系；`rebuild_kb` 在删除成功后对 KB 内剩余**全部**可构建文档执行一次保守 force-reindex；`rebuild_subgraph` 是**精确子图局部重建**：在删除前先快照被删文档贡献的实体名/关系对（`full_entities`/`full_relations`），删除成功后**只对与该足迹有交集的幸存文档**做 force-reindex，未触及被删文档子图的文档完全不动。两种 rebuild 的结果都记录在 job `result.rebuild`（`rebuild_subgraph` 额外返回 `affected_documents` / `footprint_entities` / `footprint_relations`）。`rebuild_kb` 与 `rebuild_subgraph` 都需路由注入 `IndexBuildService`，否则返回 `503`。
- **`delete_graph_orphans`**（默认 `true`）：引擎始终修剪失去最后来源的孤立实体/关系；显式传 `false` 暂不支持，返回 `400`。
- `idempotency_key` 在 `(kb_id, job_type=delete)` 维度唯一；请求指纹包含 `delete_source_file`、`delete_artifacts`、`delete_llm_cache`、`delete_graph_orphans` 与 `strategy`。同 key 同策略复用原 job；同 key 不同清理/图谱策略返回 `409`。
- `delete_source_file=true` / `delete_artifacts=true` 时仅允许删除 `INPUT_DIR/<workspace>/<document_id>/...` 内的 source/artifact 文件或目录（source 与 artifact 均锚定到规范化的 `<workspace>/<document_id>` 目录做 containment 校验），路径逃逸会使 job 失败并保留文档为 `delete_failed`。启用对象存储时会同步删除 `metadata.source_object_uri`、artifact `metadata.object_uri` / `metadata.object_prefix_uri`，并在 job result 的 `file_delete_result.deleted_objects[]` 中返回已清理对象 URI/prefix。
- **企业模式删除鉴权**：`kb_editor` 仅可删除本人上传（`metadata.created_by`）的文档（自助撤销误上传，无需额外授予）；删除他人文档需用户级能力 `can_delete_documents` 或 `kb_admin`+/`super_admin`，否则 `403`（`detail="Document delete denied"`）。删除审计事件 `document_delete_queued` 记录 `delete_scope`（`self`/`privileged`）与 `document_owner`（上传者）。`created_by` 是保留 metadata key，用户在 upload/`:texts`/PATCH 中携带会被拒绝；早于本特性、无 `created_by` 的历史文档只能由 `can_delete_documents`/`kb_admin`+/`super_admin` 删除。

批量删除：

```http
POST /kbs/{kb_id}/documents:batch-delete
Content-Type: application/json

{
  "document_ids": ["doc_a", "doc_b"],
  "delete_source_file": false,
  "delete_artifacts": false,
  "delete_llm_cache": false,
  "delete_graph_orphans": true,
  "strategy": "safe",
  "idempotency_key": null
}
```

创建单个聚合 `delete` job（`document_id=null`、`batch_id` 非空）。每个 item 独立 claim 和执行；active job、缺失文档等作为 per-item failure 写入 `job.result.items[]`，不阻塞其他可删除文档。`strategy`（`safe`/`rebuild_doc_scope`/`rebuild_kb`/`rebuild_subgraph`，默认 `safe`）与 `delete_graph_orphans`（默认 `true`，显式 `false` 返回 `400`）语义与单文档删除一致，对整批删除生效；`rebuild_kb`/`rebuild_subgraph` 同样需注入 `IndexBuildService`，否则 `503`。启用 durable worker 时，`documents:batch-delete` 属于可恢复聚合任务：worker 可从 job payload 的 `document_ids` 与删除选项恢复执行，服务重启后 queued batch-delete 不会被孤儿恢复直接标失败。企业模式下对每个文档独立做删除鉴权（规则同单文档删除）：无权删除的文档作为 `error_code=permission_denied` 的 per-item failure 写入 `job.result.items[]`，不阻塞其他文档，且不会被 claim 到 `deleting`；为防 durable worker 崩溃续跑误删，job payload 的 `document_ids` 会被收缩为已授权集合。审计事件 `documents_batch_delete_queued` 记录 `permission_denied_count` 与每文档 `delete_scopes`。

### 2.9 文档替换

```http
POST /kbs/{kb_id}/documents/{document_id}:replace?auto_parse=true&auto_index=false&parser_engine=mineru&process_options=iF&force_reparse=false&delete_source_file=true&delete_artifacts=true&delete_llm_cache=false&idempotency_key=replace-001
Content-Type: multipart/form-data

file: new-paper.pdf
```

行为：
- 创建 `replace` job，并将文档原子 claim 到 `replacing`；已有 `parse_queued/parsing`、`build_queued/building`、`deleting` 或 `replacing` 时返回 `409`。
- 若旧文档已有 `lightrag_doc_id`，后台任务先调用 `LightRAG.adelete_by_doc_id` 清理旧索引；底层返回 `success` 或 `not_found` 都视为可继续替换。
- `delete_source_file=true` / `delete_artifacts=true` 时只允许清理 `INPUT_DIR/<workspace>/<document_id>/...` 内的旧 source/artifact；路径逃逸会使 job 失败，文档进入 `replace_failed`。启用对象存储时会同步删除旧 source/artifact 对象。
- 替换成功后保留原 `document_id`，写入新的 `source_name/source_uri/source_hash/content_type/size_bytes`，清空旧 `parser_hash/index_hash/lightrag_doc_id/chunks_count/entity_count/relation_count` 和解析/索引派生 metadata，并回到 `uploaded`；启用对象存储时新的 `metadata.source_object_uri` 指向新 source 对象。
- `auto_parse=true` 会在同一个 replace job 中继续执行单文档 parse；`auto_index=true` 要求同时 `auto_parse=true` 且路由创建时已注入 `IndexBuildService`，解析成功后继续构建 KG。
- `idempotency_key` 在 `(kb_id, job_type=replace)` 维度唯一；同 key 同文件和同参数复用原 job，同 key 不同请求返回 `409`。

---

## 三、知识库解析 Parse

> 解析阶段独立于索引构建；解析成功后 KB 文档进入 `parsed` 状态，`source_hash` 与 `parser_hash` 同时生效。

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/kbs/{kb_id}/documents/{document_id}:parse` | 单文档解析 |
| `POST` | `/kbs/{kb_id}/documents:batch-parse` | 批量解析（聚合任务） |

### 3.1 单文档解析

```http
POST /kbs/{kb_id}/documents/{document_id}:parse
Content-Type: application/json

{
  "engine": "mineru",            // 可选，覆盖文档默认引擎
  "process_options": "iF",       // 可选，覆盖默认 process options
  "force_reparse": false,         // true 时绕过 MinerU/Docling raw bundle cache
  "auto_index": false,            // parse-only 接口的预留 no-op：始终不触发构建；构建用 :build-kg
  "idempotency_key": null
}
```

行为：
- 解析指令优先级为：请求体 `engine/process_options` > 文档 metadata 中已 snapshot 的 `parser_engine/process_options` > active config 的 `parser_config.engine/process_options` > 文件名/环境变量路由默认值。active `parser_config` 只提供默认值；请求体显式传值始终优先。
- `engine="legacy"` 现已接入 KB 生命周期：文本/数据/代码类文件（如 `txt/md/json/xml/yaml/log/sql/py/js/ts/css/...`）直接按 UTF-8 本地抽取，轻量 Office/PDF 路径支持 `pdf/docx/pptx/xlsx` 本地抽取，不依赖 MinerU/Docling；CSV 仍可显式指定 legacy 本地抽取，但生产默认 `.env` 路由将 `csv:docling-iteP` 放在 `*:legacy-R` 前，表格结构更适合 Docling；传统 Office `doc/ppt/xls` 仍建议按 LibreOffice → Docling/MinerU 预转换链路处理。legacy 解析同样写入 LightRAG sidecar/blocks artifact，可继续走 `:build-kg` 纳入 chunk、实体关系抽取、embedding、图谱和向量管理。
- 解析缓存命中时直接复用 artifacts：缓存有效性由 MinerU/Docling 的 `*.mineru_raw` raw bundle manifest 校验（源文件大小 + 内容 sha256 + options 签名），而非 KB 控制面的 `source_hash`/`parser_hash`（后者用于增量决策与 diff，不作为 raw bundle cache key）。`force_reparse=true` 绕过该 raw bundle cache；legacy 本地解析不依赖外部 raw bundle，会按当前 source 重新生成 sidecar/blocks。
- 同一文档已有 `parse_queued` / `parsing` / `build_queued` / `building` / `deleting` / `replacing` 时返回 `409`，原 active job 保持不变，新建的 job 同步标记 `failed`。
- 成功后写入 `original` / `sidecar` / `blocks` artifact，MinerU/Docling 还会写 `raw_dir`，并从 raw bundle 中记录细粒度文件 artifact：`markdown`、`content_list`、`middle_json`、`model_json`、`image`、`layout_pdf`。解析完成还会生成可缓存的安全预览 artifact（如 `preview_text`、`preview_table_json`；CSV 会生成 `preview_table_json`），metadata 标记 `preview=true`、`source_hash`、`parser_hash`、`truncated`、`preview_schema_version`，供文档级 preview manifest 选择。细粒度 artifact metadata 包含 `parse_engine`、`parser_hash`、`source`、`relative_path`。启用对象存储时，文件 artifact 额外写入 `metadata.object_uri`，目录 artifact 写入 `metadata.object_prefix_uri`；`original` artifact 复用 document 的 `metadata.source_object_uri`。
- **`auto_index` 是 parse-only 预留 no-op**：`:parse` 始终只解析、不构建，持久化 job payload 固定 `auto_index=false`，因此 durable worker 续跑与 in-process 路径行为一致（都不构建）。要在解析后构建知识图谱请调用 `:build-kg`。

### 3.2 批量解析

```http
POST /kbs/{kb_id}/documents:batch-parse
Content-Type: application/json

{
  "document_ids": ["doc_a", "doc_b"],
  "engine": "mineru",
  "process_options": "iF",
  "force_reparse": false,
  "auto_index": false,           // parse-only 预留 no-op：始终不构建；构建用 :batch-build-kg
  "idempotency_key": null
}
```

行为：
- 创建单个聚合 `parse` job（`document_id=null`、`batch_id` 非空）。
- 每个 item 独立成功 / 失败，记录在 `result.items[]`。
- 任一 item 失败时聚合 job 终态为 `failed`，但已成功 item 不回滚。
- 每个 item 使用与单文档解析相同的解析指令优先级；请求级 `engine/process_options` 会覆盖文档 metadata 和 active config 默认值。
- **`auto_index` 是 parse-only 预留 no-op**：`:batch-parse` 始终只解析、不构建，持久化的聚合 job payload 固定 `auto_index=false`，因此 durable worker 续跑（`_run_aggregate` 仅当 `payload["auto_index"]` 为真才构建）与 in-process 路径行为一致。要在解析后构建请调用 `:batch-build-kg`。
- in-process 路径与 durable worker 续跑均按 `MAX_PARALLEL_PARSE_MINERU` 并发解析文档；单个文档失败会作为 per-item failure 写入聚合结果，不阻塞其他文档继续解析。

---

## 四、知识库构建 Index / KG

> 基于解析产物驱动 LightRAG 的 chunk → 实体关系抽取 → embedding → KG merge 流水线。增量入库通过 `index_hash` 三段判断实现。

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/kbs/{kb_id}/documents/{document_id}:build-kg` | 单文档构建知识图谱与索引 |
| `POST` | `/kbs/{kb_id}/documents:batch-build-kg` | 批量构建（聚合任务） |
| `POST` | `/kbs/{kb_id}/documents/{document_id}:reindex` | 单文档强制重建索引（默认所有 force 标志为 true） |
| `POST` | `/kbs/{kb_id}/documents:batch-reindex` | 批量强制重建 |
| `POST` | `/kbs/{kb_id}:rebuild` | 全 KB 重建，枚举可构建文档并复用批量 reindex 路径 |

### 4.1 单文档构建

```http
POST /kbs/{kb_id}/documents/{document_id}:build-kg
Content-Type: application/json

{
  "force_rechunk": false,        // 强制重新分块
  "force_extract": false,        // 强制重新执行实体关系抽取
  "force_embedding": false,      // 强制重新写入向量
  "idempotency_key": null
}
```

增量策略：
- 若 `force_*` 全为 false 且文档已 `ready` 且当前 KB 配置派生的 `index_hash` 与 `documents.index_hash` 相等，job 直接走 skip 分支，**不调用 LightRAG pipeline**，返回 `succeeded`、`result.skipped=true`、`result.skip_reason="index_hash_match"`。
- 否则把 sidecar URI 透传给 `apipeline_enqueue_documents(docs_format="lightrag", lightrag_document_paths=[...])` + `apipeline_process_enqueue_documents()`。
- 成功后从 `doc_status` 回填 `chunks_count` / `entity_count` / `relation_count`，并把新的 `index_hash` 写到 `documents` 表。

错误码：
- `409 document_not_parsed`：文档当前状态不允许构建（必须为 `parsed` / `ready` / `build_failed`）。
- `409 build_job_active`：已有同文档处于 `build_queued` / `building`，返回 `existing_job_id`。
- `409 replace_job_active`：同文档正在替换 source/artifact，返回 `existing_job_id`。
- `409 IdempotencyKeyConflict`：`idempotency_key` 重用但请求指纹不一致。

### 4.2 批量构建

```http
POST /kbs/{kb_id}/documents:batch-build-kg
Content-Type: application/json

{
  "document_ids": ["doc_a", "doc_b"],
  "force_rechunk": false,
  "force_extract": false,
  "force_embedding": false,
  "idempotency_key": null
}
```

行为与批量解析一致：聚合 job、per-item result、active conflict 作为 per-item failure。

### 4.3 重建索引

```http
POST /kbs/{kb_id}/documents/{document_id}:reindex
Content-Type: application/json

{
  "force_rechunk": true,
  "force_extract": true,
  "force_embedding": true,
  "idempotency_key": null
}
```

`:reindex` 与 `:build-kg` 共用同一份后台执行逻辑，区别是默认所有 `force_*` 为 `true`，永远不会触发 KB 层的 `index_hash` skip。当任一 `force_*` 为真且文档已有 `lightrag_doc_id` 时，后台执行会先调用 `LightRAG.adelete_by_doc_id` 清除旧索引再 re-enqueue，从而真正绕过 LightRAG 引擎自身的 id/文件名/内容去重，确保已建文档被重新分块、抽取与嵌入（否则 enqueue 会把同 id 文档当作重复项静默丢弃，使重建变成空操作）。

### 4.4 全 KB 重建

```http
POST /kbs/{kb_id}:rebuild
Content-Type: application/json

{
  "force_rechunk": true,
  "force_extract": true,
  "force_embedding": true,
  "idempotency_key": null
}
```

行为：
- 枚举该 KB 内所有处于可构建状态（`parsed` / `ready` / `build_failed`）的文档，复用 `:batch-reindex` 的批量构建路径对它们整体重建。
- `force_*` 默认全为 `true`（保守全量重建）；可显式放宽以让命中 `index_hash` 的文档走 skip。
- KB 内没有可构建文档时返回 no-op（`job_id=""`、`documents=[]`），不报 400。
- KB 不存在返回 404；注册表/构建服务未配置返回 503。

### 4.5 KB 级图谱查询与编辑

> 通过 `LightRAGInstanceRegistry` 解析到该 KB 的 LightRAG 实例，因此图谱统计/标签/子图/编辑均按 workspace 隔离到单个知识库。企业模式默认禁用全局 `/graph/*` 路由后，写端点是图谱人工纠错（合并重复实体、改名、删错误关系）的唯一入口。

只读端点（企业模式 `kb_viewer`+）：

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/kbs/{kb_id}/graph/status` | 图谱统计：`label_count` / `node_count` / `edge_count` / `is_truncated`（受 `max_nodes_scanned` 上限保护） |
| `GET` | `/kbs/{kb_id}/graph/entities` | 实体标签分页列表，支持 `limit` / `offset` 与可选模糊搜索 `q`（大小写不敏感） |
| `GET` | `/kbs/{kb_id}/graph/relations` | 关系（edge）分页列表，返回 `id/type/source/target/properties`；`source`/`target` 为实体名 |
| `GET` | `/kbs/{kb_id}/graph` | 指定 `label` 的连通子图（`*` 表示整图），支持 `max_depth` / `max_nodes`；`nodes[].id` 与 `edges[].source/target` 为实体名 |

编辑端点（企业模式 `kb_admin`+，包装引擎 curation 方法）：

| 方法 | 路径 | 请求体 | 说明 |
|---|---|---|---|
| `POST` | `/kbs/{kb_id}/graph/entity:edit` | `{entity_name, updated_data, allow_rename=false, allow_merge=false}` | 更新实体属性；`updated_data.entity_name` + `allow_rename=true` 改名；改名撞已有实体且 `allow_merge=true` 时自动合并。响应含 `data` 与 `operation_summary`（`operation_status`/`merged`/`final_entity` 等） |
| `POST` | `/kbs/{kb_id}/graph/entity:create` | `{entity_name, entity_data}` | 新建独立实体（常用字段 `description`/`entity_type`）；同名已存在返回 `400` |
| `POST` | `/kbs/{kb_id}/graph/entity:delete` | `{entity_name}` | 删除实体及其全部关系；不存在返回 `404`，返回 `DeletionResult` |
| `POST` | `/kbs/{kb_id}/graph/entities:merge` | `{source_entities: [...], target_entity}` | 把多个重复/错拼实体合并进目标实体，关系全部转移、来源实体删除 |
| `POST` | `/kbs/{kb_id}/graph/relation:edit` | `{source_entity, target_entity, updated_data}` | 更新关系属性（`description`/`keywords`/`weight` 等） |
| `POST` | `/kbs/{kb_id}/graph/relation:create` | `{source_entity, target_entity, relation_data}` | 在两个**已存在**实体间新建关系（无向边，返回时端点可能交换） |
| `POST` | `/kbs/{kb_id}/graph/relation:delete` | `{source_entity, target_entity}` | 删除一条关系；不存在返回 `404`，返回 `DeletionResult` |

约束：
- 只读端点复用全局 `/graph/*` 同款 LightRAG 方法，但带 KB workspace 边界。
- **子图与关系列表的 id 已归一化为实体名**：部分图存储后端（如 Neo4j）原生返回内部存储 id，KB 路由统一改写为实体名后返回，因此 `nodes[].id`、`edges[].source/target`、`relations[].source/target` 可直接作为编辑端点的 `entity_name` / `source_entity` / `target_entity` 回传（此前直接回传 Neo4j 内部 id 会导致 `entity:delete` 等报 404）。`edges[].id` / `relations[].id` 仍为后端内部边 id，仅作展示用途。
- `entity:delete` / `relation:delete` 的实体名首尾空白会被自动剔除后再匹配。
- `graph/status` 与 `graph/relations` 使用 `"*"` 通配做有界全图扫描（默认上限 100,000 节点）；超限时 `is_truncated=true`。
- 编辑端点在文档 pipeline 忙碌（构建/删除进行中）时返回 `409`，等任务完成后重试；引擎参数校验失败（实体不存在/已存在等）返回 `400`。
- **手工编辑结果存放在引擎存储中，会被该文档的 force `:reindex` / `:rebuild` 重新抽取覆盖**——图谱纠错建议在文档集稳定后进行，或纠错后避免对相关文档强制重建。
- 企业模式下所有编辑动作写审计：`kb_graph_entity_edited/created/deleted`、`kb_graph_entities_merged`、`kb_graph_relation_edited/created/deleted`（metadata 记录实体名/数量，不含正文）。
- KB 不存在返回 404。

---

## 五、知识库任务 Jobs

> 任务持久化在 metadata store（local SQLite 或 PostgreSQL），跨进程可见。所有耗时操作均会创建 job，客户端通过 `job_id` 跟踪进度。

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/kbs/{kb_id}/jobs` | 任务列表，支持状态 / 文档 ID 过滤 |
| `GET` | `/kbs/{kb_id}/jobs/dead-letter` | 死信任务列表（`failed` 且重试已耗尽） |
| `GET` | `/kbs/{kb_id}/jobs/{job_id}` | 任务详情 |
| `POST` | `/kbs/{kb_id}/jobs/{job_id}:wait` | 阻塞等待任务到达终态（succeeded / failed / cancelled） |
| `POST` | `/kbs/{kb_id}/jobs/{job_id}:cancel` | 取消任务 |
| `POST` | `/kbs/{kb_id}/jobs/{job_id}:retry` | 重试 failed / cancelled 任务 |

### 5.1 列表 / 详情

```http
GET /kbs/{kb_id}/jobs?status=running&document_id=doc_xxx&limit=50&offset=0
GET /kbs/{kb_id}/jobs/{job_id}
```

死信列表（dead-letter）：

```http
GET /kbs/{kb_id}/jobs/dead-letter?limit=50&offset=0
```

- 返回 `status=failed` 且 `retry_count >= max_retries` 的任务，即 `:retry` 已被拒绝、不会再自动重跑的终态失败任务。
- 与普通 `/jobs?status=failed` 区分：后者包含仍可重试的失败任务，前者只列出需要人工介入的死信任务。
- `cancelled` 任务不计入死信（属于主动取消，非耗尽重试）。
- 路由注册顺序保证字面量 `dead-letter` 不会被当作 `job_id` 匹配。

任务字段（`JobResponse`）核心列：

| 字段 | 说明 |
|---|---|
| `id` | 任务 ID |
| `job_type` | `upload` / `parse` / `build_kg` / `reindex` / `delete` / `replace` / `sync` / `clear_kb`。`:reindex` / `:batch-reindex` 产出 `job_type=reindex`（与 `:build-kg` 的 `build_kg` 区分）；批量解析/构建/删除复用聚合 `parse` / `build_kg` / `reindex` / `delete` job（`document_id=null` + `batch_id`），`:rebuild` 复用聚合 `build_kg`。 |
| `status` | `queued` / `running` / `succeeded` / `failed` / `cancelling` / `cancelled` / `retrying` |
| `stage` | 当前阶段：`uploading` / `parsing` / `building` / `deleting` 等 |
| `progress` | 0.0 ~ 1.0 |
| `total_items / completed_items / failed_items` | 批量进度 |
| `idempotency_key` | 幂等键 |
| `retry_count / max_retries` | 重试计数 |
| `payload` | 创建任务时的入参（含 `idempotency_fingerprint`） |
| `result` | 成功 / 失败的结构化结果，批量任务包含 `items[]` |

### 5.2 取消任务

```http
POST /kbs/{kb_id}/jobs/{job_id}:cancel
```

状态转换规则：
- `queued` → `cancelled`，`error_code=cancelled_by_user`。
- `running` / `retrying` → `cancelling`。parse / build_kg / reindex 执行器在进入昂贵阶段（解析 / chunk-抽取-嵌入）前会检查 `cancelling` 协作式取消检查点：命中则不调用 parser/pipeline，job 转 `cancelled`，文档释放回 `parse_failed` / `build_failed`（可经 `:retry` 重跑）。
- **parse 阶段额外支持 await 内强制中断**：进入解析后，执行器把单次长 parse await（MinerU/Docling/native）作为独立 `asyncio.Task` 运行，并并发轮询 job 状态；一旦翻转为 `cancelling` 即 `cancel()` 该任务，**无需等待解析跑完**，文档释放回 `parse_failed`（可 `:retry`）。这是安全的，因为解析幂等——重跑只是覆盖 raw bundle/sidecar。**build_kg / 向量写入阶段刻意不做 await 内强制中断**（中途打断可能留下半合并图谱/半写入向量），仍只用阶段边界协作式取消。
- `succeeded` / `failed` / `cancelled` 视为 no-op，原样返回当前 job。
- `cancelling` 视为 no-op。

### 5.3 重试任务

```http
POST /kbs/{kb_id}/jobs/{job_id}:retry
Content-Type: application/json

{
  "idempotency_key": "retry-key-2"   // 可选；不传则保留原 key
}
```

行为：
- 仅允许 `failed` 或 `cancelled` 任务重试；其他状态返回 `409`。
- 任务回到 `queued`，清空 `result` / `error_code` / `error_message` / `started_at` / `finished_at` / `cancelled_at`，并刷新 `queued_at`。
- `retry_count += 1`；超过 `max_retries`（默认 3）返回 `409`。
- 重试后的消费方式取决于是否启用 durable worker：
  - 默认（未启用）：worker 是 in-process 后台任务，重试后需由调用方再次触发同一接口或原始业务动作。
  - 启用 `LIGHTRAG_KB_JOB_WORKER=true` 后：内置 durable worker 会自动消费回到 `queued` 的 `parse` / `build_kg` / `reindex` 单文档任务、单文档 `delete` / `replace` 任务、聚合 `sync` 任务以及 `documents:batch-delete` 聚合任务，无需客户端再次触发；服务重启后这些 `queued` 任务也会被自动续跑（见 5.5）。

### 5.5 Durable job worker（可选）

> 通过环境变量 `LIGHTRAG_KB_JOB_WORKER=true` 启用。企业 PostgreSQL 部署建议开启；关闭时仍使用 in-process 背景任务，但不提供 queued job 的自动重驱动。

启用后：
- 服务启动会拉起一个后台轮询 worker，原子认领（`queued → running` 单赢 CAS）以下可从持久化状态重建的任务类型并执行到终态：单文档 `parse` / `build_kg` / `reindex` / `delete` / `replace`，**聚合** `parse` / `build_kg` / `reindex`（`document_id=null`、payload 携带 `document_ids`，含多文件 `upload` / `texts` 的 auto_parse 聚合 job 与 `batch-parse` / `batch-build-kg` / `batch-reindex` / `:rebuild`），聚合 `sync`（payload 携带 `batch_id` 与 per-item `source_key/source_name/source_hash`，请求字节已落盘到 `.sync-staging/<batch_id>/`），`documents:batch-delete` 聚合 `delete` job，以及 `clear_kb`（KB 硬删除，payload 携带 `kb_id`/`workspace`，幂等清理可重启续跑）。聚合 parse/build 之所以可恢复，是因为其源文件 / 解析产物在 job 运行前已落盘，worker 可凭 `document_ids` 重新规划并逐个 claim 执行；聚合 sync 则凭 staged request bytes 重建 `DocumentSourceInput` 并复用同一 per-item 同步逻辑。
- **单文档 `replace` 现已可恢复**：replace 创建并 claim 时会把替换源字节落盘到 `INPUT_DIR/<workspace>/<document_id>/.replace-staging-<job_id>.bin`，因此 worker 可在重启/`:retry` 后凭 staged 字节重建 `DocumentReplacementSource`，重新 claim 文档进入 `replacing`，复用与同步路径一致的执行逻辑（删旧索引 → 换 source → 可选 auto_parse/auto_index），终态后清理 staging 文件。若 staged 字节缺失（历史 job 未落盘），或 staged 字节内容 hash 与 payload `source_hash` 不匹配（staging 文件被截断/损坏），worker 以 `replace_not_resumable` 明确失败，不会凭错字节续跑（与批量 `sync` 续跑的 hash 校验对齐）。
- **批量 `sync` 现已可恢复**：sync route 在创建聚合 job 前为每个 item 落盘请求字节，并在 payload 中持久化 `batch_id`、`source_key`、`source_name`、`source_hash`、`content_type` 与同步选项；worker 重启/`:retry` 后按 staged bytes 重建每个 source，重新执行 created/replaced/skipped 与可选 parse/build。staging 只在终态 job transition 成功后 best-effort 清理；若 staged bytes 缺失或 hash 不匹配，worker 以 `sync_not_resumable` 明确失败。
- **自动消费 `:retry`**：重试把任务重置回 `queued` 后，worker 在下一轮轮询中认领并重跑，客户端无需再次发起业务请求。
- **跨进程单 owner**：worker claim 后持有 per-job session ownership lock；PostgreSQL 使用独立 operation-lock pool 中的 advisory lock，local 兼容后端使用 job 文件锁。启动恢复与周期恢复只有在 job 超过 grace 且 non-blocking owner-lock 成功时才认定其为 orphan，不会把其他 live worker 正在执行的 job 重置后重复运行。
- **重启续跑**：可恢复类型的 `queued` job 保持 queued。失去 owner 的普通 mid-flight job按既有失败/重试契约处理；`clear_kb` 会原 job 原 payload 自动 requeue，因为其 generation、workspace 与阶段 checkpoint 足以安全恢复，catalog 已 purge 时也只执行 lifecycle/job 尾部收敛。
- **不抢占新任务**：worker 只认领 `queued_at` 早于宽限窗口（`LIGHTRAG_KB_JOB_WORKER_GRACE_SECONDS`，默认 5s）的任务；新建任务由其 in-process 背景任务在毫秒级转入 `running`，因此不会被 worker 抢跑，避免重复执行。
- **需重新发起的类型**：多文件 `upload` 且 `auto_parse=false` 时不产生可重驱动解析工作；其他没有持久化请求上下文的历史/自定义任务仍会在孤儿恢复时标 `failed`，需要重新发起请求。
- **死信**：`failed` 且 `retry_count >= max_retries` 的任务不会再被 `:retry` 或 worker 重跑，可通过 `GET /kbs/{kb_id}/jobs/dead-letter` 单独列出做人工triage。
- 可调环境变量：`LIGHTRAG_KB_JOB_WORKER_POLL_SECONDS`（默认 1.0s）、`LIGHTRAG_KB_JOB_WORKER_GRACE_SECONDS`（默认 5.0s）、`LIGHTRAG_KB_JOB_RECOVERY_INTERVAL_SECONDS`（默认 30.0s）、`LIGHTRAG_KB_JOB_RECOVERY_GRACE_SECONDS`（默认沿用 worker grace，未启用 worker 时为 5.0s）。PostgreSQL 的 `LIGHTRAG_KB_POSTGRES_OPERATION_LOCK_POOL_MAX_SIZE`（默认 10）是 fence/job-owner 的独立连接池，不占用普通 metadata pool，避免业务池 `max_size=1` 时自锁。

### 5.5.1 聚合任务的并发执行模型

> 适用于产生聚合 `parse` job 的入口：多文件 `:upload` / `:texts` 的 `auto_parse=true`、`:sync`、`:batch-parse`，以及它们的 durable worker 续跑。单文档 `:parse` / `:build-kg` / `:reindex` 不受影响（仍是单文档路径）。

聚合任务内部按两阶段执行，让解析阶段（MinerU/VLM 重）与索引阶段（实体抽取 / KG merge / 向量写入）跨文档重叠，缩短整批耗时、提高 GPU/VLM 利用率：

- **Phase 1 并发解析**：批内所有文档**并发**调用解析引擎，并发上限由 `MAX_PARALLEL_PARSE_MINERU` 控制（其余引擎对应 `MAX_PARALLEL_PARSE_NATIVE` / `MAX_PARALLEL_PARSE_DOCLING`）。
- **Phase 2 单次批量构建**（仅 `auto_index=true`）：把全部解析成功的文档**一次性批量入队**到 LightRAG 流水线并**单次 drain**，使 analyze / extract / merge 三层 worker 跨文档流水线化重叠（受 `MAX_PARALLEL_ANALYZE` / `MAX_PARALLEL_INSERT` 约束），而不是每个文档独立启停一次流水线。

行为约定：

- **失败隔离**：单个文档解析或构建失败只标记该文档 `parse_failed` / `build_failed` 并记入 `result.items[]`，不影响同批其它文档继续；聚合 job 终态在全部成功时 `succeeded`，部分失败时 `failed` 且 `result.summary.outcome=partial_failure`。
- **结果顺序**：`result.items[]` 不保证与请求输入顺序一致（按完成情况聚合），但每个输入文档 / `source_key` 恰好出现一次。
- **取消**：批量构建途中触发 `:cancel`，流水线协作式取消会把在途文档标记为 `cancelled`（而非 `build_failed`）。
- **并发 drain 安全**：当同一 KB 上有多个聚合流并发（例如两个 disjoint `:sync`、或 `:sync` 与 `:upload` auto_parse 同时进行），构建结果读取采用对 `doc_status` 终态的轮询等待，避免把仍在其它流水线 drain 中的文档误判为失败。
- 可调环境变量：`MAX_PARALLEL_PARSE_MINERU`（聚合 Phase 1 并发解析上限，默认随部署，生产建议按 MinerU 后端可承受并发设置）、`KB_BUILD_DRAIN_TIMEOUT_SECONDS`（构建结果等待终态超时，默认 3600s）、`KB_BUILD_DRAIN_POLL_SECONDS`（轮询间隔，默认 1.0s）。

### 5.6 等待任务终态

```http
POST /kbs/{kb_id}/jobs/{job_id}:wait?timeout_seconds=60&poll_interval_seconds=0.5
```

服务端持续轮询 metadata store 直到任务进入 `succeeded` / `failed` / `cancelled` 三态之一并返回最终 `JobResponse`；超时未到终态返回 `408 Request Timeout` 携带 `current_status`。

约束：
- `timeout_seconds` 限制在 `[0.1, 600.0]`；客户端可按需调小。
- `poll_interval_seconds` 限制在 `[0.05, 5.0]`，默认 0.5s。
- 该接口存在的目的是让客户端写线性脚本（`upload -> wait -> build -> wait -> query`）时不必自己实现轮询逻辑。

---

## 六、知识库产物 Artifacts

> 产物记录解析阶段产生的文件 / 目录。当前支持 `original` / `sidecar` / `blocks` / `raw_dir`，以及 MinerU/Docling raw bundle 中的 `markdown` / `content_list` / `middle_json` / `model_json` / `image` / `layout_pdf` 等细粒度类型；同时支持预览缓存产物 `preview_text` / `preview_table_json`。`uri` 是本地 cache path；启用对象存储后，metadata 会额外包含 `object_uri` 或 `object_prefix_uri`。

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/kbs/{kb_id}/documents/{document_id}/artifacts` | 产物列表 |
| `GET` | `/kbs/{kb_id}/documents/{document_id}/preview` | 文档级预览 manifest：返回后端已生成的安全预览 variant 与原件下载 fallback |
| `GET` | `/kbs/{kb_id}/documents/{document_id}/artifacts/{artifact_id}` | 产物元数据 |
| `GET` | `/kbs/{kb_id}/documents/{document_id}/artifacts/{artifact_id}:download` | 下载文件型产物；目录型产物以 zip 代理下载 |
| `GET` | `/kbs/{kb_id}/documents/{document_id}/artifacts/{artifact_id}:preview` | 内联预览受支持的小型文件型产物 |
| `GET` | `/kbs/{kb_id}/documents/{document_id}/artifacts/{artifact_id}:download-url` | 为对象存储中的文件型产物生成预签名下载 URL |

列表示例：`GET /kbs/{kb_id}/documents/{document_id}/artifacts?artifact_type=markdown&limit=50&offset=0`。

文档级 preview manifest：
- `GET /kbs/{kb_id}/documents/{document_id}/preview` 面向新前端统一 viewer。返回结构包含 `document_id`、`source_name`、`source_content_type`、`status`、`preferred`、`variants[]`、`fallback`。
- `variants[]` 中每项包含 `kind`（`text` / `table` / `html` 等）、`artifact_id`、`artifact_type`、`media_type`、`size_bytes`、`preview_url`。前端按 `kind` 选择 viewer：文本/代码 viewer、表格 viewer、HTML sandbox viewer、PDF/image 原件 viewer；如果 `variants` 为空则展示 `fallback.download_url` 下载。
- `preferred` 是服务端按安全性和可用性排序的首选 variant；`fallback` 指向 `original` artifact 的 `:download` URL（若原件 artifact 尚不存在或无法恢复则为 `null`）。因此不要让前端直接尝试 inline Excel/Office 原件；Excel 优先消费 `preview_table_json`，否则回退下载。

下载约束：
- 企业模式权限：artifact list/detail 按 `kb_viewer` 或更高角色读取。`:download` 与 `:download-url` 的默认最低角色由 `LIGHTRAG_ENTERPRISE_ARTIFACT_DOWNLOAD_MIN_ROLE` 控制，默认 `kb_viewer` 保持旧行为，可提升为 `kb_editor`、`kb_admin` 或 `kb_owner`；`LIGHTRAG_ENTERPRISE_ARTIFACT_DOWNLOAD_POLICY` 可用 JSON object 按 artifact type 覆盖（如 `{"original":"kb_editor","*":"kb_viewer"}`），并同时作用于显式匹配类型的 `:preview`。更细粒度时可使用 `LIGHTRAG_ENTERPRISE_ARTIFACT_ACTION_POLICY`，按 action 分别设置 artifact type policy，例如 `{"preview":{"*":"kb_editor"},"download":{"original":"kb_editor"},"download-url":{"original":"kb_admin"}}`。action policy 优先于 download policy；低于要求的角色返回 `403`。
- 对交互式 JWT 用户，所有 `:download` / `:download-url` 还必须满足用户级 `can_download_files=true`；`original` 的 `:preview` 同样要求该能力，防止把原文件 inline preview 当作下载绕过。`preview_text` / `preview_table_json` 等派生安全预览只按 KB role/action policy，不要求下载能力。super admin 与 service/scoped API key 不受该用户能力位限制，但 service key 仍受自身 KB scope/role 与 artifact policy 约束。新建用户默认 `can_download_files=false`；迁移前已有用户兼容为 `true`，管理员可显式收紧。
- 文件型产物（`original` / `blocks` / `markdown` / `content_list` / `middle_json` / `model_json` / `image` / `layout_pdf` / `preview_text` / `preview_table_json`）以 `FileResponse` 直接返回。
- 目录型产物（`sidecar` / `raw_dir`）以流式 zip 返回（`Content-Type: application/zip`），单次下载 zip 内未压缩字节上限 512 MB，超限返回 `413`。
- 路径必须位于 `inputs/<workspace>/<document_id>` 内；跨 KB、缺失文件、路径逃逸均返回 `404` / `400`。
- 启用对象存储时，如果本地 cache path 缺失，`:download` 接口会先从 `metadata.object_uri` / `metadata.object_prefix_uri` restore 到原 cache path，再返回文件或 zip，保持旧客户端兼容。
- `:preview` 仅支持文件型 artifact，目录返回 `400`；支持 `text/*`、`application/json`、`application/ld+json`、`application/markdown`、`application/x-ndjson`、普通图片（不含 `image/svg+xml`）和 `application/pdf`，以 `inline` content-disposition 返回；单次 preview 上限 10 MB，超出返回 `413`，不支持的 media type 返回 `415`。响应带 `X-Content-Type-Options: nosniff`；HTML 类预览带保守 CSP。不要把 Excel/Word/PPT MIME 直接加入 inline 白名单，需通过后端生成的 `preview_text` / `preview_table_json` / 未来 `preview_html` 等安全产物预览。本地 cache 缺失时同样按对象存储 metadata restore。
- `:download-url` 仅对 metadata 中存在 `object_uri` 的**文件型** artifact 生效，返回 `{artifact_id,url,object_uri,expires_in_seconds,filename,media_type}`；服务端使用对象存储后端生成 `GET Object` 预签名 URL，不会触发本地 cache restore。`expires_in_seconds` 默认 3600 秒，服务端限制在 `[1, 604800]`。目录型 artifact（`sidecar` / `raw_dir`，metadata 中为 `object_prefix_uri`）仍需走 `:download` 的 zip 代理下载。企业模式且 `LIGHTRAG_ENTERPRISE_MASK_STORAGE_URIS=true`（默认）时，文档 `source_uri`、artifact `uri`、响应中的 storage metadata、job payload/result 中的路径字段与 `download-url.object_uri` 会返回 `"<masked>"` 或被递归移除，不泄露本地路径或对象存储 URI；下载/预览/预签名内部仍使用真实 metadata。

---

## 七、知识库配置版本 Config Versions

> 📌 **完整字段速查见 [`docs/KB配置项速查表.md`](KB配置项速查表.md)**（每个 section 的字段、别名、影响哪个 hash、改动后的最小动作）。

> 不可变的 KB 级配置快照。新建配置不会自动生效，需要显式 `:activate` 才会写入 `KnowledgeBase.active_config_version_id` 并 discard 缓存的 LightRAG 实例。当前实现会让后续实例重建或 parse planning 时读取已支持的 active config 字段；部署级字段会在创建配置版本时直接拒绝，避免单个 KB 修改已经部署好的服务基础设施。创建时会严格校验：各 section（`parser_config`/`chunk_config`/`embedding_config`/`query_config`/`extraction_config`）出现**未知键**（无运行时效果）会直接返回 `400`，避免"存了不生效"。
>
> 已接入运行时的 active config 字段：
> - `parser_config`：`engine`/`parser_engine`、`process_options`/`options`。`engine` 支持 `legacy` / `native` / `mineru` / `docling`，会在创建配置时校验并规范化，作为解析默认值参与 `parser_hash`，并按“请求 > 文档 metadata > active config > 文件路由”的优先级生效。
> - `chunk_config`：`chunk_size`/`chunk_token_size`、`chunk_overlap_size`/`chunk_overlap_token_size`、`tiktoken_model_name`。
> - `embedding_config`：`model`、`dim`/`embedding_dim`、`token_limit`/`max_token_size`（`model` 会触发重建 embedding provider 闭包）。
> - `query_config`：`top_k`/`chunk_top_k`/`max_entity_tokens`/`max_relation_tokens`/`max_total_tokens`/`related_chunk_number`/`cosine_threshold` 等 QueryParam 字段；另支持 `bilingual_query`（`off`/`auto`/`on`，非法值创建时 `400`），控制该 KB 的双语双路检索模式（见 [8.2 双语查询](#82-双语双路检索bilingual)与 `docs/BilingualQuery-zh.md`）。`bilingual_query` 参与 `query_hash`（仅影响查询，不触发重建），且不会作为 QueryParam 默认值下发。
> - `extraction_config`：`language`（摘要/抽取语言）、`entity_types`（列表，自动渲染成 `entity_types_guidance` 并去重保序）或显式 `entity_types_guidance`（优先于 `entity_types`）、`entity_type_prompt_file`、`max_gleaning`/`max_extraction_records`/`max_extraction_entities`/`force_llm_summary_on_merge`。这些会 overlay 到 `addon_params` 与 LightRAG 抽取构造参数，并纳入 `index_hash`，因此变更会被 `:diff` 标为 `requires_reindex`。
> - `llm_role_config`：按角色（`extract`/`keyword`/`query`/`vlm`/`agent`/`profile`/`bilingual`）覆盖运行时 LLM。每个角色可为字符串（等价 `{"model": <str>}`）或对象（`model`/`binding`/`host`/`api_key`/`provider_options`/`model_kwargs`(别名 `kwargs`)/`max_async`/`timeout`）。配置创建时校验角色名与字段名（未知项报错）。实例构建后通过已注册的 role builder 调用 `aupdate_llm_role_config` 应用覆盖，因此 `binding`/`model`/`host`/`api_key` 变更会重建该角色的 LLM func。哈希影响：`extract`/`vlm` 角色的“输出身份”（`binding`/`model`/`host`/`provider_options`/`model_kwargs`，不含 `api_key` 与 `max_async`/`timeout`）纳入 `index_hash`（变更触发 `requires_reindex`）；`query`/`keyword`/`agent`/`profile`/`bilingual` 角色身份纳入 `query_hash`（仅影响查询，不重建）。轮换 `api_key` 或调 `max_async`/`timeout` 不改变任何哈希、不触发重建。
> - 部署级配置不允许写入 KB config：`storage_config`，以及 `parser_config` 中的 parser 服务实例字段（如 endpoint/base_url/api_key/api_mode/token/timeout/workers/max_concurrency 等）。这些字段必须通过 `.env` / 部署编排统一管理；请求中携带会返回 `400`。

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/kbs/{kb_id}/configs` | 创建配置版本（自动派生 `parser_hash` / `index_hash` / `query_hash`） |
| `GET` | `/kbs/{kb_id}/configs` | 列出所有配置版本；支持 `?limit=50&offset=0` 分页 |
| `GET` | `/kbs/{kb_id}/configs/{version_id}` | 获取配置版本详情 |
| `POST` | `/kbs/{kb_id}/configs/{version_id}:activate` | 激活配置版本 |
| `POST` | `/kbs/{kb_id}/configs/{version_id}:diff` | 与当前激活版本做 diff，预测重建影响 |

### 7.1 创建配置版本

```http
POST /kbs/{kb_id}/configs
Content-Type: application/json

{
  "config": {
    "parser_config": {"engine": "mineru", "process_options": "iF"},
    "chunk_config": {"chunk_size": 512},
    "embedding_config": {"model": "bge-large", "dim": 1024},
    "llm_role_config": {"extract": "gpt-4o-mini"},
    "query_config": {"top_k": 60}
  },
  "created_by": "alice"
}
```

返回 `ConfigVersionResponse`，`version` 由服务端按 KB 内单调递增生成。若请求体包含 `storage_config` 或 parser 服务实例级字段（例如 `parser_config.endpoint` / `api_key` / `api_mode`），返回 `400`，不会创建配置版本。

### 7.2 激活配置

```http
POST /kbs/{kb_id}/configs/{version_id}:activate
Content-Type: application/json

{
  "auto_enqueue": true
}
```

行为：
- 更新 KB 的 `active_config_version_id`。
- 写入配置版本的 `activated_at`。
- 调用 `LightRAGInstanceRegistry.discard(kb_id)` 卸载实例，下次请求按已支持的 active config 字段重建。
- 若该 KB 上有 destructive job 在执行（如 `clear_kb`），discard 静默跳过。
- 请求体可省略或传 `{"auto_enqueue": false}`，保持旧行为：只激活配置，不创建后续 job。
- `auto_enqueue=true` 时，服务先按 `:diff` 结果判断影响，再创建至多一个可查询 follow-up job：
  - `requires_reparse=true`：枚举 `uploaded/parsed/parse_failed/build_failed/ready` 且 `enabled=true`、`archived=false` 的文档，创建聚合 `job_type="parse"` job，`payload.force_reparse=true`、`payload.auto_index=true`，由 parse worker 重新规划并在解析成功后批量构建。
  - 否则 `requires_reindex=true` 或 `requires_vector_rebuild=true`：枚举 `parsed/ready/build_failed` 且 `enabled=true`、`archived=false` 的文档，创建聚合 `job_type="reindex"` job，`force_rechunk/force_extract/force_embedding=true`，复用 `:batch-reindex` 语义。
  - query-only 变更不创建 job，返回 `follow_up_noop_reason="no_rebuild_required"`；无 eligible 文档返回 `follow_up_noop_reason="no_eligible_documents"`；未配置 `JobService` 返回 `job_service_unavailable`。
  - follow-up job 使用 `config-activation:{version_id}:{job_type}:{digest}` 幂等 key，同一目标版本、job 类型与文档集合不会重复创建冲突 job。

注意：`auto_enqueue=true` 在激活接口中只负责创建 `queued` follow-up job；实际 parse/reindex 执行依赖 durable job worker 或对应 job consumer 继续消费队列。未配置 `JobService` 时不会创建任务，返回 `follow_up_noop_reason="job_service_unavailable"`；query-only 变更或无 eligible 文档时也不会创建任务。

返回 `ConfigVersionActivationResponse`，兼容包含原 `ConfigVersionResponse` 字段，并额外包含：

```json
{
  "id": "cfg_target",
  "kb_id": "kb_research",
  "version": 2,
  "activated_at": "2026-06-08T...Z",
  "auto_enqueue": true,
  "diff": {
    "target_version_id": "cfg_target",
    "active_version_id": "cfg_previous",
    "requires_reparse": false,
    "requires_reindex": true,
    "requires_vector_rebuild": true,
    "reasons": ["embedding_changed", "index_hash_changed"]
  },
  "follow_up_jobs": [
    {
      "id": "job_reindex_...",
      "job_type": "reindex",
      "status": "queued",
      "total_items": 3,
      "payload": {"document_ids": ["doc_a", "doc_b", "doc_c"], "force_embedding": true}
    }
  ],
  "follow_up_noop_reason": null
}
```

### 7.3 配置 Diff

```http
POST /kbs/{kb_id}/configs/{version_id}:diff
```

返回：

```json
{
  "target_version_id": "cfg_xxx",
  "active_version_id": "cfg_yyy",
  "requires_reparse": false,
  "requires_reindex": true,
  "requires_vector_rebuild": true,
  "reasons": ["embedding_changed", "index_hash_changed"]
}
```

- `requires_reparse`：`parser_hash` 不同。
- `requires_reindex`：`parser_hash` 或 `index_hash` 不同。
- `requires_vector_rebuild`：`embedding_config.model` 或 `embedding_config.dim` 不同。
- 当 KB 没有 active 版本时，三项均为 `true`，`reasons=["no_active_version"]`。

---

## 八、知识库问答 Query

> 在指定知识库上跑 RAG 问答。请求会路由到 `LightRAGInstanceRegistry` 中该 KB 对应的 LightRAG 实例，复用全局 `/query` 同款 `aquery_llm` / `aquery_data` 链路，但带 KB 边界保护。

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/kbs/{kb_id}/query` | 非流式问答，返回 `response + references` |
| `POST` | `/kbs/{kb_id}/query/stream` | 流式问答，返回 NDJSON |
| `POST` | `/kbs/{kb_id}/query/data` | 仅返回结构化检索数据，不调用 LLM |
| `POST` | `/kbs/{kb_id}/retrieve` | `query/data` 的别名，语义等价 |
| `POST` | `/kbs:query` | 跨多个知识库合并问答：扇出检索 → 共享 reranker 合并 → 单次 LLM 合成，references 标注来源 `kb_id` |
| `POST` | `/kbs:query/stream` | 同上，流式返回 NDJSON（首行 `{kb_ids, metadata, references}`，后续 `{response}`） |
| `POST` | `/kbs:retrieve` | 跨库检索-only：返回合并后的 chunks/references（带 `kb_id`），不调用 LLM |

企业模式下，当前用户还可以为单个 KB 保存个人查询提示词：见 [10.2 登录与当前用户](#102-登录与当前用户) 的 `/auth/me/kbs/{kb_id}/query-settings`。KB query 最终 `user_prompt` 优先级为：请求体显式 `user_prompt` > 当前用户在该 KB 下持久化的 `user_prompt` > active KB config 的 `query_config.user_prompt` > 空字符串。

请求体（与全局 `/query` 共用字段，新增 KB scoped `filters.doc_ids` / `filters.metadata`）：

```json
{
  "query": "低共熔溶剂在萃取分离中的应用？",
  "mode": "mix",
  "top_k": 60,
  "chunk_top_k": 20,
  "include_references": true,
  "include_chunk_content": false,
  "stream": false,
  "bilingual": null,
  "filters": {
    "doc_ids": ["doc_xxx"],
    "metadata": {"tenant": "demo", "tag": ["legal", "finance"]}
  },
  "conversation_history": [
    {"role": "user", "content": "上文..."}
  ],
  "user_prompt": "请使用 Markdown 列表呈现"
}
```

响应（非流式）：

```json
{
  "kb_id": "kb_research",
  "mode": "mix",
  "response": "...",
  "references": [
    {"reference_id": "1", "file_path": "paper.pdf", "content": null}
  ],
  "metadata": {
    "config_version_id": "cfg_xxx",
    "parser_hash": "sha256:...",
    "index_hash": "sha256:...",
    "query_hash": "sha256:..."
  }
}
```

约束：
- 同 KB 内的查询不会读取其他 KB 的内容（`workspace` 隔离）；已加测试覆盖。
- 若 KB-wide 查询时 KB 内存在 `deleting` / `replacing` 文档，或显式 `filters.doc_ids` / `filters.metadata` 命中的候选文档中存在此类 active 文档，查询返回 `409`，避免读到删除/替换中的旧内容；metadata filter 未命中的 active 文档不会阻断本次 scoped 查询。
- `mode` 支持 `local / global / hybrid / naive / mix / bypass`；建议默认 `mix`。
- 企业模式下，`mode="bypass"` 按最终解析后的查询模式检查：请求体显式 `mode="bypass"` 或 active query config 默认值解析为 `bypass` 时，均需要 KB read ACL 加 `can_use_bypass_query=true` 或 super admin。
- 企业模式下，用户可按 `user_id + kb_id` 保存个人 `user_prompt`；保存后对 `/kbs/{kb_id}/query`、`/query/stream`、`/query/data` 自动生效，但请求体显式传入的 `user_prompt` 始终优先。
- `filters.doc_ids` 会先校验 ID 必须属于本 KB（不在则 400 + `error_code=doc_ids_not_in_kb`），随后在检索层精确生效：服务端把 `filters.doc_ids` 与"可检索集合"（`enabled=true` 且 `archived=false` 且已建索引、有 `lightrag_doc_id`）取交集，映射成 `QueryParam.ids`（即 `full_doc_id` 白名单）传入 LightRAG。被禁用/归档的文档即使显式出现在 `filters.doc_ids` 里也会被静默剔除，不会进入答案。KB 边界仍由 workspace 双重保证。
- `filters.metadata` 支持非空 key，value 为标量或标量列表（列表为 OR 语义），总 JSON 大小上限 64 KB；它会先在 KB documents metadata 上做精确匹配，再与 `filters.doc_ids`、`enabled/archived/lightrag_doc_id` 可检索集合取交集。无匹配时传入空 `QueryParam.ids=[]`，返回空检索范围而不是退回 KB-wide。
- `include_chunk_content=true` 时 `references[].content` 返回该 reference 命中的 chunk 文本数组，便于评估与排查。
- `bilingual`（可选，`true`/`false`/缺省）：双语双路检索显式覆盖；缺省时跟随 KB `query_config.bilingual_query` 与部署默认值，详见 [8.2 双语双路检索](#82-双语双路检索bilingual)。双路生效时响应 `metadata.bilingual` 返回 `{enabled, source_language, translated_query, primary_chunks, secondary_chunks, merged_chunks, final_chunks, ...}`。
- 非流式、结构化检索和流式首行都会在 `metadata` 中返回 active config 信息（存在时包含 `config_version_id`、`parser_hash`、`index_hash`、`query_hash`）。
- 流式响应 `Content-Type: application/x-ndjson`：第一行是 `{kb_id, metadata}`，若 `include_references=true` 则同一行还包含 `references`；后续每行 `{response: "..."}`，错误时 `{error: "..."}`。当请求体 `stream=false` 或底层返回非流式结果时，`/query/stream` 会返回单行完整 NDJSON，而不是多段 chunk。
- 短查询（< 3 字符）返回 422；KB 不存在 404。

### 8.1 跨知识库合并查询（`/kbs:query`、`/kbs:retrieve`）

在多个知识库上一次问答并合成单一答案，适用于"同一问题需跨多个库检索"的场景。采用 **scatter-gather**：每个 KB 用自身实例检索（保留隔离），chunk 标注来源 `kb_id`，在检索层跨库合并 + 重排，再做单次 LLM 合成。

```http
POST /kbs:query
Content-Type: application/json

{
  "kb_ids": ["kb_research", "kb_legal"],
  "query": "低共熔溶剂在萃取分离中的应用？",
  "mode": "mix",
  "top_k": 60,
  "chunk_top_k": 20,
  "enable_rerank": true,
  "include_references": true,
  "include_chunk_content": false,
  "user_prompt": "请用中文回答并给出引用"
}
```

响应：

```json
{
  "kb_ids": ["kb_research", "kb_legal"],
  "mode": "mix",
  "response": "...",
  "references": [
    {"reference_id": "1", "file_path": "paper.pdf", "kb_id": "kb_research", "content": null}
  ],
  "metadata": {
    "requested_kb_count": 2,
    "per_kb_chunk_counts": {"kb_research": 12, "kb_legal": 8},
    "merged_chunk_count": 20,
    "final_chunk_count": 20,
    "reranked": true,
    "skipped_kbs": [],
    "synthesis_kb_id": "kb_research"
  }
}
```

约束与行为：

- **前提**：所有目标 KB 共用同一套本地模型服务（LLM/VLM/embedding/rerank）；KB 隔离不变（按 KB 扇出，仅在检索层合并，不合并 workspace）。检索对所有 KB 使用请求体中的**同一套查询参数**，不叠加各 KB 的 active `query_config`。
- `kb_ids` 必填，1..10 个；超限 422。`mode` 支持 `local/global/hybrid/naive/mix`，**`bypass` 返回 400**（无检索可合并）。
- **合并排序**：对合并后的 chunk 池用共享 reranker 统一重排（基于文本，与 embedding 无关），再按 `chunk_top_k` 与 token 预算截断。`metadata.reranked` 标识是否重排。
- **引用**：合并后统一重新编号 `reference_id`（跨库冲突自动消解），每条引用标注来源 `kb_id`；`include_chunk_content=true` 时附 chunk 文本。
- **合成所用 LLM/tokenizer/reranker** 为共享部署服务；`metadata.synthesis_kb_id` 仅作溯源标识。
- **企业模式鉴权**：对 `kb_ids` 中**每个** KB 都要求 `kb_viewer`+；任一无权 → **403（fail-closed）**。注意：中央 RBAC 中间件不覆盖 collection 级 `/kbs:query`/`/kbs:retrieve` 路径，鉴权由 handler 自行逐 KB 执行。审计事件 `multi_kb_query_executed` / `multi_kb_retrieve_executed`，仅记 `kb_ids`/`mode`/`query_hash`/计数，不记原文。
- **容错**：单个 KB 检索失败（404/异常）记入 `metadata.skipped_kbs` 并跳过，其余照常；全部失败返回 502；命中 `deleting/replacing` 文档的 KB 触发 409。
- **`/kbs:retrieve`**：同样的扇出+合并+重排，但不调用 LLM，返回 `data.chunks`（带 `kb_id`）+ `data.references`。
- **`/kbs:query/stream`**：与 `/kbs:query` 同入参，流式返回 `application/x-ndjson`：首行 `{kb_ids, metadata, references}`，随后每行 `{response: "..."}`，出错 `{error}`。合成阶段单次 LLM，故先完成检索/合并再流式输出答案。
- **`filters`（可选，per-KB）**：`metadata` 过滤统一作用于每个 KB；`doc_ids` 按所属 KB 拆分（每个 KB 只应用属于它的 id），且每个 `doc_id` 必须至少属于一个目标 KB，否则 `400`（`detail.missing` 列出越界 id）。语义与单库 `filters` 一致，仅 doc_ids 改为跨库软交集。
- **`bilingual`（可选）**：对每个目标 KB 启用双语双路检索（见 8.2）；多库查询不读取 per-KB `query_config`，由请求体标志 + 部署默认值决定，预处理只调用一次并被所有 KB 共享。双路生效时 `metadata.bilingual` 额外含 `per_kb_secondary_chunks` 与（失败时）`secondary_failed_kbs`。
- 查询缓存为后续项。

### 8.2 双语双路检索（bilingual）

> 完整设计、降级链与灰度建议见 `docs/BilingualQuery-zh.md`。前提：多语言 embedding（如 bge-m3 / Qwen-Embedding）+ 多语言 reranker（如 bge-reranker-v2-m3）。

面向中英混合语料的跨语言召回：一次预处理 LLM 调用产出译句 + 双语 hl/ll 关键词（**替代**核心层关键词提取调用，kg 模式 LLM 调用数不变），随后"原句+同语关键词"与"译句+另一语关键词"两路并行检索，chunk 池按 `chunk_id` 去重合并、以原句统一 rerank 截断、引用重编号，最后单次合成并强制以原句语言回答（引用他语证据时关键术语括注原文）。

**三层开关（优先级从高到低）：**

| 层级 | 配置 | 取值 |
|---|---|---|
| 请求体 | `bilingual` | `true`（强制开）/ `false`（强制关）/ 缺省（跟随下层） |
| KB config | `query_config.bilingual_query`（仅单库端点读取） | `off` / `auto` / `on` |
| 部署环境 | `BILINGUAL_QUERY_DEFAULT_MODE`（默认 `auto`） | `off` / `auto` / `on` |

全部受总开关 `BILINGUAL_QUERY_ENABLED`（默认 `false`）约束；`auto` 表示仅含 CJK 字符的查询走双路。预处理超时由 `BILINGUAL_QUERY_TIMEOUT`（默认 12s）控制。

**翻译模型（`bilingual` LLM 角色）**：预处理调用走独立的 `bilingual` 角色，任何未设置的 `BILINGUAL_LLM_*` 字段（model/binding/host/api_key/max_async/timeout）**逐项继承 `QUERY_LLM_*`**，因此默认零配置即用 query 同款模型；后续切专用翻译模型只需设 `BILINGUAL_LLM_MODEL` 等。该调用固定 `enable_cot=false`（不思考）；Qwen3 类模型可再加 `BILINGUAL_OPENAI_LLM_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'` 硬关思考模式。KB 级可用 `llm_role_config.bilingual` 覆盖（身份参与 `query_hash`）。

**行为与约束：**

- 覆盖端点：单库 `/kbs/{kb_id}/query`、`/query/stream`、`/query/data`、`/retrieve`；跨库 `/kbs:query`、`:query/stream`、`:retrieve`；legacy 全局 `/query` 系列（无 KB config 层）；Agent 两种工作流（见 §9.3）。所有底层 mode（`local/global/hybrid/naive/mix`）均支持。
- 自动跳过双路：`mode=bypass`、`only_need_context/only_need_prompt=true`、请求体显式提供 `hl_keywords/ll_keywords`。
- **fail-open 降级链**：预处理失败/超时/译句不可用 → 单路（现状行为），`metadata.bilingual={enabled:false, reason:"preprocess_unavailable"}`；副路检索失败 → 只用主路并标记 `secondary_failed`；主路失败 → 与现状同样报错。
- 预处理结果缓存在 KB 的 LLM cache（`bilingual:query_preprocess:*`），换查询角色模型自动失效。
- `/query/data` 的合并结果中，实体按 `entity_name`、关系按 `(src_id, tgt_id)` 去重；两路各自的 `reference_id` 编号体系无法跨路复用，合并后置空（chunks 与 references 保持一致编号）。
- 审计 metadata 记录 `bilingual_enabled` 与 `bilingual_translated_query_hash`（只记 hash 不记原文，与 `query_hash` 口径一致）。

### 8.3 项目记忆自动注入（仅终答合成）

> 当前公开契约仅适用于企业认证 + PostgreSQL metadata 部署，并要求 `LIGHTRAG_CHAT_MEMORY_ENABLED=true`。客户端只声明项目记忆作用域；原始事实由服务端按需召回，不能由客户端拼入受信任提示词。

支持执行**最终 query LLM 合成**的下列端点：

| 类型 | 非流式 | 流式 | 说明 |
|---|---|---|---|
| 单 KB | `POST /kbs/{kb_id}/query` | `POST /kbs/{kb_id}/query/stream` | 普通与 `bilingual=true` 双语双路均支持 |
| 多 KB | `POST /kbs:query` | `POST /kbs:query/stream` | 普通与 `bilingual=true` 双语双路均支持 |
| Agent | `POST /agent/query` | `POST /agent/query/stream` | `workflow="plan"` 与 `workflow="staged"` 均支持，也可启用双语检索 |

兼容旧版的全局 `/query`、`/query/stream`、`/query/data` 当前不公开 `memory` 请求字段，不属于本契约。

请求体新增可选作用域：

```json
{
  "query": "低温性能怎么做？",
  "mode": "mix",
  "memory": {
    "project_id": "proj_1a2b3c4d5e6f",
    "limit": 10
  }
}
```

- `project_id` 必填，长度 `1..128`；`limit` 可省略，省略时取部署默认 `MEMORY_SEARCH_LIMIT`，合法范围 `1..50`。
- 只要携带 `memory`，`query` 最长为 **4096 字符**；超限在项目记忆检索前返回 `400`。
- **授权早（authorize early）**：在 KB 检索或 Agent 规划前校验功能开关、交互式 JWT principal、项目归属与查询长度，只创建不含事实的进程内授权句柄。不存在和属于他人的项目统一 `404`，不泄露项目是否存在。
- **检索晚（search late）**：先完成规划、KB 选择、双语预处理、检索、合并、rerank 与当前权威证据预算；只有确定将执行最终合成、选定实际 query LLM/tokenizer 并算出完整最终请求的剩余预算后，才校验 egress 并至多检索一次项目记忆。
- **仅终答合成（final-synthesis-only）**：记忆不会进入 Agent 规划、staged 需求解析/骨架召回/指标验证、KB 选择、检索查询、关键词提取、rerank 或结构化 verdict，也不会改变召回结果。Agent 需要澄清时已完成作用域授权，但不搜索记忆；返回 `not_used/clarification_required`。
- **必须有当前 KB 证据**：最终处理后的当前 KB 证据为空时，不搜索记忆、不调用最终 query LLM，也不能仅凭记忆作答；返回 `not_used/no_kb_evidence`。该约束同样适用于单 KB、多 KB、双语、Agent plan 和 Agent staged。

以下请求没有最终合成；携带 `memory` 时会在任何项目记忆检索前直接返回 HTTP `400`，稳定错误码为 `chat_memory_requires_final_synthesis`：

| 不支持组合/端点 | 结果 |
|---|---|
| 单 KB 或多 KB query/query-stream 的 `mode="bypass"` | `400 chat_memory_requires_final_synthesis` |
| 单 KB query/query-stream 的 `only_need_context=true` | `400 chat_memory_requires_final_synthesis` |
| 单 KB query/query-stream 的 `only_need_prompt=true` | `400 chat_memory_requires_final_synthesis` |
| 单 KB query/query-stream 同时设置两个 flag 为 `true` | `400 chat_memory_requires_final_synthesis` |
| `POST /kbs/{kb_id}/query/data` | `400 chat_memory_requires_final_synthesis` |
| `POST /kbs/{kb_id}/retrieve` | `400 chat_memory_requires_final_synthesis` |
| `POST /kbs:retrieve` | `400 chat_memory_requires_final_synthesis` |

#### `metadata.memory` 响应契约

请求了 `memory` 时，非流式响应在原有 `metadata` 下增加 `memory`；单/多 KB stream 在首个 NDJSON 头事件中返回，Agent stream 在最终 `done` 事件中返回：

```json
{
  "metadata": {
    "memory": {
      "enabled": true,
      "project_id": "proj_1a2b3c4d5e6f",
      "status": "injected",
      "fact_count": 5,
      "injected_count": 3,
      "truncated": true,
      "references": [
        {
          "reference_id": "M1",
          "fact_id": "edge-uuid",
          "valid_at": "2026-07-10T08:00:05+00:00"
        }
      ],
      "reason": "可选，仅特定状态出现"
    }
  }
}
```

字段与冻结状态如下：

| 字段/状态 | 契约 |
|---|---|
| `enabled` | 本次记忆作用域是否可用；后端可用但未使用/无结果/预算不足时仍为 `true`，仅 typed backend availability 故障的 fail-open 结果为 `false` |
| `project_id` | 已授权的 chat 项目；只在响应 metadata 中返回，不复制到 query/Agent 审计的记忆投影 |
| `status="injected"` | 至少一条完整事实记录进入最终上下文 |
| `status="empty"` | 已搜索，但没有可用事实 |
| `status="budget_exhausted"` | tokenizer/总容量不可用、固定安全框架放不下，或有事实但没有完整记录能同时满足 token/字符/最终请求预算；若有可用事实因预算被省略，`truncated=true` |
| `status="unavailable"` | 仅限 typed Chat Memory backend availability 故障；fail-open，`enabled=false`、`reason="unavailable"`，主查询继续 |
| `status="not_used"` | 已授权但不搜索；`reason` 固定为 `clarification_required` 或 `no_kb_evidence` |
| `fact_count` | 项目记忆搜索匹配数（保持兼容口径），未搜索时为 `0` |
| `injected_count` | 实际注入的完整 JSONL 记录数 |
| `truncated` | 是否至少有一条可用事实因预算未被注入；空白/格式错误事实不算预算截断 |
| `references` | 仅包含已注入事实的内容无关溯源；`reason` 不适用时省略 |

引用命名空间严格分离：普通/多 KB 当前证据沿用数字 `[1]`、`[2]`；Agent 当前证据沿用 `[A1]`、`[A2]`；项目记忆只使用本次请求内连续分配的 `[M1]`、`[M2]`。`metadata.memory.references[].fact_id` 是 **Graphiti generation-scoped** 标识，重建后可能变化，不得作为永久业务主键。`[M*]` 不进入既有顶层 `references`，也不进入模型生成的 `### References` 段；该段始终只列当前 KB 证据。

#### 信任、预算、缓存与敏感数据边界

- 服务端把记忆拆成两部分：受信任的 **Server Memory Policy** 放在服务端控制的指令区；事实只以带显式 begin/end delimiter 的**不受信任 JSONL 数据**放入最终 Context。事实中的控制符、换行、尖括号和方括号会转义，事实文本不能伪造 delimiter 或 `[M99]` 记录。
- 记忆通过私有 sensitive-context 参数直达最终合成，且不修改 `QueryParam.user_prompt`；它也不会进入查询配置反射、持久化 query param、hash 或缓存 metadata。只有服务端生成的 `reference_id` 字段可建立 `[M*]`。
- `[M*]` 声明必须由本次当前权威 KB 证据独立佐证后才可作为次级 inline provenance；冲突时以 KB 证据为准，未佐证记忆必须丢弃。Agent 的客观结论和 staged verdict 仍必须由 `[A*]` 支撑。
- 预算同时受 `LIGHTRAG_CHAT_MEMORY_PROMPT_MAX_TOKENS`、`LIGHTRAG_CHAT_MEMORY_PROMPT_MAX_CHARS` 和完整最终请求的 `max_total_tokens` 约束；完整请求计入 system prompt、query、全部 conversation history、确定性分隔符和 framing reserve。记忆永不挤掉已选定的 KB 证据。
- 默认 egress 策略要求记忆抽取 LLM 与本次实际最终 query LLM 的 credential-free canonical endpoint identity 均非空且相等；不相等、只解析出一方或双方都未知都会在搜索记忆前拒绝。只有显式设置 `LIGHTRAG_CHAT_MEMORY_ALLOW_CROSS_PROVIDER_QUERY_EGRESS=true` 才可跨 endpoint；原始 endpoint 不进入响应、审计或敏感日志。
- 只要请求携带授权记忆句柄，KG/naive 最终 query-result cache 的 lookup/write 均绕过，即使最终状态为 `empty`、`unavailable` 或 `budget_exhausted`；关键词缓存仍可使用，因为记忆不进入关键词 prompt。
- 最终 LLM 调用在整个非流式/流式生命周期标记为敏感调用：抑制 prompt/response debug 日志与 tracing，避免使用会采集内容的可选 instrumentation，并把 provider/stream 异常清洗为不含 endpoint、请求体、事实或模型输出的错误。
- 省略 `memory` 时不创建敏感句柄，不添加 `metadata.memory`，现有 JSON/NDJSON 字段和字节形态保持兼容；Agent stream 的 `done` 事件也不新增 `metadata`。

#### 错误与 fail-open

| 条件 | HTTP/流契约 | 是否搜索记忆 |
|---|---|---|
| 功能未启用/读取服务未挂载 | `503` | 否 |
| principal 不是交互式 JWT 用户 | `403` | 否 |
| 项目不存在或不属于当前用户 | `404` | 否 |
| `memory` 请求的 query 超过 4096 字符 | `400`；KB 路径使用 `chat_memory_query_too_long`，Agent 当前返回内容无关的长度错误 | 否 |
| 无最终合成（上表全部组合） | `400 chat_memory_requires_final_synthesis` | 否 |
| 最终 query LLM egress 不符合默认同 endpoint 策略 | `403 chat_memory_query_llm_egress_not_allowed` | 否 |
| 最终请求 builder 返回无效内容 | 内容无关 `500 chat_memory_final_request_builder_invalid` | 可能尚未搜索或已在候选预算阶段搜索；均立即硬失败 |
| typed backend availability 故障 | 不返回错误；主查询继续，`metadata.memory={enabled:false,status:"unavailable",fact_count:0,injected_count:0,truncated:false,references:[],reason:"unavailable"}` | 已尝试或在 read fence/backend 阶段失败 |

Agent 流在响应头发出后才发现 egress/builder 等 hard policy error 时，HTTP 状态不能再改写；服务端发出一个终止 NDJSON 事件：`{"event":"error","error_code":"...","status_code":403|400|500,"message":"..."}`，不泄露 provider 或记忆内容。

---

## 九、兼容旧版 / 全局接口

> 这些接口走全局默认 `workspace`，主要给现有 WebUI 与早期客户端使用。生产新接入建议使用 `/kbs/...` 系列。

### 9.1 文档（`/documents`）

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/documents` | 已弃用的文档状态列表（最多 1000 条）；新客户端使用 `POST /documents/paginated` |
| `POST` | `/documents/scan` | 扫描 `input_dir` 并入库 |
| `POST` | `/documents/upload` | 单文件上传（旧版） |
| `POST` | `/documents/text` | 单文本插入 |
| `POST` | `/documents/texts` | 批量文本插入 |
| `DELETE` | `/documents` | 清空所有文档 |
| `GET` | `/documents/pipeline_status` | 全局 pipeline 状态 |
| `DELETE` | `/documents/delete_document` | 按 ID 删除文档 |
| `POST` | `/documents/clear_cache` | 清理 LLM 缓存 |
| `GET` | `/documents/track_status/{track_id}` | 跟踪 ID 状态查询 |
| `POST` | `/documents/paginated` | 分页文档状态 |
| `GET` | `/documents/status_counts` | 状态统计 |
| `POST` | `/documents/reprocess_failed` | 重处理失败文档 |
| `POST` | `/documents/cancel_pipeline` | 取消运行中的 pipeline |

### 9.2 查询（无前缀，挂在根路径）

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/query` | 非流式问答 |
| `POST` | `/query/stream` | 流式问答（NDJSON，`Content-Type: application/x-ndjson`） |
| `POST` | `/query/data` | 仅返回结构化检索数据，不调用 LLM 生成 |

支持的 `mode`：`local` / `global` / `hybrid` / `naive` / `mix` / `bypass`。`/query/stream` 与 KB scoped stream 一样返回 NDJSON；当请求体 `stream=false` 或底层返回非流式结果时，会返回单行完整 NDJSON。请求体支持 `bilingual` 字段启用双语双路检索（无 KB config 层，按"请求体 > 部署默认值"解析，行为同 §8.2）。

### 9.3 Agent 查询模式（`/agent`）

Agent 查询模式是服务端多轮编排入口，不是 `QueryParam.mode=agent`。它使用 `AGENT_LLM_*` 角色模型输出 JSON 规划，随后在当前用户可访问的 KB 范围内串行调用既有 `local` / `global` / `hybrid` / `naive` / `mix` 检索，**不支持 `bypass`**。完整能力要求企业模式、`LIGHTRAG_AGENT_QUERY_ENABLED=true`、当前 principal 具备 `can_use_agent_query`，且每个被选中的 KB 至少具备 `kb_viewer`。AGENT 规划时看到的 `allowed_kbs` 只包含已授权 KB 的 `name`、`description` 与合并后的 Agent Profile：人工 `agent_description` / `agent_tags` / `agent_priority` 优先，缺省时使用 `agent_auto_profile`。

Agent 支持两种工作流（请求体 `workflow` 字段选择）：

- `workflow="plan"`（默认）：**一次性规划** —— AGENT LLM 先输出完整步骤计划，服务端按 P0→P1→P2 排序截断后串行执行，证据合并后一次合成终答。
- `workflow="staged"`：**阶段化配比/配方推荐工作流** —— 面向"推荐一种在某环境下的配比"类问题的证据链流水线（需求解析 → 骨架召回 → 要素证据 → 指标验证 → 缺口补查 → 合成），详见 §9.3.1 与 `docs/AgentStagedRecommendation-zh.md`。

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/agent/query` | 非流式 Agent 问答；返回终答、Agent 级引用、步骤摘要与 metadata |
| `POST` | `/agent/query/stream` | Agent NDJSON 事件流；事件随执行进度**实时输出**（规划完成即发 `plan_created`，每轮检索前后发 `round_started` / `round_result`，终答按增量 `response` delta 流式输出）。`plan` 工作流事件：`session_started`、`plan_created`、`round_started`、`round_result`、`references`、`response`、`clarification_required`、`done`、`error`；`staged` 工作流额外事件见 §9.3.1 |

请求体：

```json
{
  "query": "请结合法规和配方库推荐一个合规方案",
  "workflow": "plan",
  "candidate_kb_ids": ["kb_regulation", "kb_formula"],
  "max_rounds": 5,
  "top_k": 40,
  "chunk_top_k": 20,
  "enable_rerank": true,
  "include_references": true,
  "include_chunk_content": false,
  "bilingual": null,
  "filters": {
    "metadata": {"category": "food"}
  }
}
```

字段说明：

- `workflow` 可省略，默认 `"plan"`；`"staged"` 启用阶段化配比推荐工作流（此时 `max_rounds` 不生效，检索预算由 `AGENT_STAGED_MAX_RETRIEVALS` 与阶段上限控制，见 §9.3.1）。
- `candidate_kb_ids` 可省略或为空：表示候选范围为当前 principal 的全部授权 KB，由 AGENT 模型在规划时自行选择；若提供，则必须全部在授权范围内，否则返回 `403`，且不会调用 AGENT LLM。
- `max_rounds` 会被服务端全局 `AGENT_MAX_ROUNDS` clamp；所有轮次串行执行。规划步骤按 **P0 → P1 → P2 稳定排序** 后再截断到 `max_rounds`，保证 P0（法规/合规类）步骤不会被截断丢弃；发生截断时 `metadata.plan_truncated=true`。
- 规划/阶段化的所有 AGENT LLM 调用均请求 **schema 约束的结构化输出**（OpenAI 兼容 `response_format: json_schema`，其中 `kb_ids`、`mode`、`priority` 等字段为受限枚举，`steps` 要求至少 1 步）；后端不支持 `json_schema` 时自动降级为 `json_object`，再不支持时不传 `response_format`（仅提示词约束）。
- 规划 JSON 解析失败会自动重试（最多 3 次 AGENT LLM 调用）。重试耗尽时：若模型曾返回"合法 JSON 但零步骤"的退化计划，服务端**降级为单步兜底检索**（对全部候选 KB 执行一次 `mix` 检索，`steps_summary` 单步标题为 `直接检索`，`metadata.notes_for_user` 说明降级原因），不再返回 502；其余失败仍返回 `502`（`error_code=agent_plan_invalid`），并写审计事件 `agent_session_failed`。
- 步骤 `priority` 具备别名容错：`high`/`normal`/`low`、`高`/`中`/`低` 等常见别名归一化为 `P0`/`P1`/`P2`，无法识别时默认 `P1`（`staged` 工作流的 `target_properties[].priority` 同样适用）。
- **单步失败容忍**：某一步骤检索失败不会终止会话——该步在 `steps_summary` 中标记 `status="failed"`（附 `error_code`），其余步骤继续执行，终答会明确声明对应证据缺口；所有步骤都失败时返回 `502`（`error_code=agent_all_steps_failed`）。
- **空结果自动重试**：某步检索成功但返回 0 条证据时，服务端自动用一个替代 mode 重试一次（`mix→naive`、`naive→hybrid`、`hybrid→mix`、`local/global→hybrid`）；重试成功时该步 `steps_summary` 与 `round_result` 事件携带 `retried_mode` 字段，审计 metadata 同步记录；重试自身失败不影响该步（保留原空结果）。两种工作流均生效。
- 终答证据合成 **不做二次 rerank**：每步结果已按该步子问题排序（检索内启用 rerank 时），跨轮去重后按轮次轮转合并，再按 `max_total_tokens` 预算截断并编号为 A1、A2…，避免子问题证据被总问题的相关性打分整体挤掉。
- KB 级用户查询设置中的 `user_prompt` 不参与 Agent 终答；需要定制终答风格请使用请求体 `user_prompt` 或用户工作流提示词（`/auth/me/agent-workflow-prompt`）。
- `filters` 为用户请求级过滤，模型不能生成越权 filters；服务端沿用 KB query/retrieve 的文档生命周期、enabled/archived 与 doc-id scope 约束。
- 底层检索 mode 由 AGENT 规划步骤决定，但只能是 `local` / `global` / `hybrid` / `naive` / `mix`。
- **双语双路检索**（`bilingual` 字段，行为基线见 §8.2；Agent 按"请求体 > 部署默认值"解析，不读 per-KB config）：启用时规划 LLM 为每步额外生成 `query_alt` / `hl_keywords_alt` / `ll_keywords_alt`（该步子问题的另一语言版本，一次规划调用顺带完成，零额外 LLM 成本），步骤执行器对每个 KB 双路检索合并（副路失败仅告警不失败）。`staged` 工作流同步生效：需求解析额外产出 `target_properties[].name_alt`（驱动指标验证步骤副路）、骨架提取额外产出 `open_questions_alt`（与 `open_questions` 等长按序配对，驱动要素证据步骤副路）、骨架/补查规划步骤同 plan 工作流。双路生效的轮次在 `steps_summary` 与 `round_result` 事件携带 `bilingual=true` 与 `alt_chunk_counts`；终答 `metadata.bilingual_retrieval` 标识本次会话是否启用。空结果换 mode 重试、检索预算等既有机制不变。

非流式成功响应：

```json
{
  "status": "success",
  "session_id": "agent_...",
  "answer": "根据法规要求... [A1]",
  "clarification_question": null,
  "references": [
    {
      "reference_id": "A1",
      "kb_id": "kb_regulation",
      "round": 1,
      "step_index": 1,
      "mode": "mix",
      "file_path": "regulation.md",
      "chunk_id": "chunk-...",
      "source_reference_id": "1"
    }
  ],
  "steps_summary": [
    {
      "round": 1,
      "step_index": 1,
      "title": "查询法规限制",
      "query": "检索法规限制...",
      "kb_ids": ["kb_regulation"],
      "mode": "mix",
      "priority": "P0",
      "status": "ok",
      "chunk_count": 5,
      "per_kb_chunk_counts": {"kb_regulation": 5},
      "skipped_kbs": []
    }
  ],
  "metadata": {
    "workflow": "plan",
    "effective_kb_ids": ["kb_regulation", "kb_formula"],
    "round_count": 1,
    "failed_round_count": 0,
    "plan_truncated": false,
    "bilingual_retrieval": false,
    "notes_for_user": null
  }
}
```

失败步骤在 `steps_summary` 中的形态（会话不中断）：

```json
{
  "round": 2,
  "step_index": 2,
  "title": "查配方",
  "query": "检索配方建议...",
  "kb_ids": ["kb_formula"],
  "mode": "mix",
  "priority": "P1",
  "status": "failed",
  "error_code": "kb_retrieve_failed",
  "chunk_count": 0,
  "per_kb_chunk_counts": {},
  "skipped_kbs": []
}
```

澄清响应：

```json
{
  "status": "clarification_required",
  "session_id": "agent_...",
  "answer": "",
  "clarification_question": "请补充目标应用场景和限制条件。",
  "references": [],
  "steps_summary": [],
  "metadata": {"effective_kb_ids": ["kb_regulation"]}
}
```

流式响应为 `application/x-ndjson`，事件随执行进度实时输出；`response` 事件为增量 delta，可出现多条。示例：

```json
{"event":"session_started","session_id":"agent_...","metadata":{"workflow":"plan","effective_kb_ids":["kb1"]}}
{"event":"plan_created","session_id":"agent_...","plan_truncated":false,"notes_for_user":null,"steps":[...]}
{"event":"round_started","session_id":"agent_...","round":1,"step_index":1,"title":"查法规","kb_ids":["kb1"],"mode":"mix","priority":"P0"}
{"event":"round_result","session_id":"agent_...","round":1,"status":"ok","kb_ids":["kb1"],"chunk_count":5}
{"event":"references","session_id":"agent_...","references":[...]}
{"event":"response","session_id":"agent_...","delta":"根据法规"}
{"event":"response","session_id":"agent_...","delta":"要求... [A1]"}
{"event":"done","session_id":"agent_..."}
```

#### 9.3.1 阶段化配比推荐工作流（`workflow="staged"`）

面向 **按证据来源分库**（如配方库、实验数据库、论文库、应用专项库）的配比/配方推荐问题，服务端按固定阶段流水线执行，证据链锚定在知识库而非模型记忆（设计详见 `docs/AgentStagedRecommendation-zh.md`）：

```text
S0 需求解析（结构化目标性能指标清单，缺关键信息 → clarification）
S1 骨架召回（模型标注各 KB 证据角色 kb_roles → 检索参考配方 ≤3 步 → 提取骨架组分，引用必须可校验）
S2 要素证据（open_questions + 组分补充查询，服务端模板实例化，≤8 步且为 S3 预留预算）
S3 指标验证（逐指标检索实验数据 → 裁决 supported/partial/unsupported/no_data，无有效引用的结论降级 no_data）
S4 缺口补查（存在 no_data/unsupported 或空结果步骤时补查一轮 ≤4 步，仅对缺口指标重新裁决）
S5 终答合成（推荐配比表 + 指标核对 + 未覆盖点，配比数值只能来自证据）
```

行为要点：

- **检索预算**：单会话检索步数硬上限 `AGENT_STAGED_MAX_RETRIEVALS`（默认 24）；被预算跳过的工作记入 `metadata.clipped` 并进入终答"未覆盖点"，不静默截断。
- **每步 KB 数上限**：`AGENT_STAGED_MAX_KBS_PER_STEP`（默认 4）。知识库数量不定：模型选库超限或角色回退到"全部授权库"时，按各库人工 `agent_priority` 降序择优（同分保持原顺序），裁剪记入 `metadata.clipped`。
- **引用编号**：证据在检回时即分配稳定 `A{n}` 编号（提取/裁决先于终答引用），最终 `references` 编号可能不连续（编号是 ID 不是序号）；每条引用携带 `stage` 与 `evidence_role`（reference_formula/mechanism/validation/repair）。
- **fail-closed**：骨架组分引用无效 → 丢弃并计入 `dropped_components`；裁决缺失/无效引用 → `no_data`；骨架规划选择越权 KB → `403` 会话失败；补查规划越权 → 仅丢弃该步并记录。骨架提取与指标裁决 LLM 失败不终止会话（按缺口处理），仅 S0/S1 规划失败返回 `502`（`agent_requirement_invalid` / `agent_skeleton_plan_invalid`）。
- `max_rounds` 不适用于 staged；其余请求字段（`candidate_kb_ids`、`top_k`、`filters`、`user_prompt`、用户工作流提示词等）语义与 `plan` 一致。

新增 NDJSON 事件（其余复用 `plan` 工作流）：

```json
{"event":"session_started","session_id":"agent_...","metadata":{"workflow":"staged","effective_kb_ids":["kb_formula","kb_exp","kb_paper","kb_side"]}}
{"event":"stage_started","session_id":"agent_...","stage":"requirement"}
{"event":"requirement_parsed","session_id":"agent_...","requirement":{"application":"胎侧胶料","conditions":["高寒环境"],"target_properties":[{"name":"低温屈挠性","why":"低温开裂","priority":"P0"}],"constraints":[]}}
{"event":"stage_started","session_id":"agent_...","stage":"skeleton"}
{"event":"kb_roles_assigned","session_id":"agent_...","kb_roles":{"kb_formula":"reference_formula","kb_exp":"experimental","kb_paper":"literature","kb_side":"application_spec"}}
{"event":"round_started","session_id":"agent_...","round":1,"stage":"skeleton","title":"查参考配方","kb_ids":["kb_formula","kb_side"],"mode":"mix","priority":"P0"}
{"event":"round_result","session_id":"agent_...","round":1,"stage":"skeleton","status":"ok","chunk_count":5,"new_chunk_count":5}
{"event":"skeleton_extracted","session_id":"agent_...","components":[{"material":"NR/BR 并用","ratio":"50/50 phr","function":"低温屈挠性能","source_refs":["A1"]}],"open_questions":["..."],"dropped_components":0}
{"event":"stage_started","session_id":"agent_...","stage":"factor_evidence"}
{"event":"stage_started","session_id":"agent_...","stage":"validation"}
{"event":"validation_verdicts","session_id":"agent_...","verdicts":[{"property":"低温屈挠性","priority":"P0","verdict":"supported","evidence_refs":["A4"],"note":"有实测数据"}],"after_repair":false}
{"event":"stage_started","session_id":"agent_...","stage":"gap_repair"}
{"event":"validation_verdicts","session_id":"agent_...","verdicts":[...],"after_repair":true}
{"event":"references","session_id":"agent_...","references":[{"reference_id":"A1","kb_id":"kb_formula","stage":"skeleton","evidence_role":"reference_formula","round":1,"step_index":1,"mode":"mix","file_path":"formula.md","chunk_id":"chunk-...","source_reference_id":"1"}]}
{"event":"response","session_id":"agent_...","delta":"推荐配比表..."}
{"event":"done","session_id":"agent_..."}
```

非流式响应结构与 `plan` 一致，`metadata` 扩展为：

```json
{
  "workflow": "staged",
  "effective_kb_ids": ["kb_formula", "kb_exp", "kb_paper", "kb_side"],
  "kb_roles": {"kb_formula": "reference_formula", "kb_exp": "experimental"},
  "requirement": {"application": "胎侧胶料", "conditions": ["高寒环境"], "target_properties": [...], "constraints": []},
  "skeleton_component_count": 6,
  "dropped_component_count": 0,
  "property_verdicts": [{"property": "低温屈挠性", "priority": "P0", "verdict": "supported", "evidence_refs": ["A4"], "note": "..."}],
  "round_count": 8,
  "failed_round_count": 0,
  "retrieval_budget": {"max": 24, "used": 8},
  "clipped": []
}
```

配置（`.env`）：

```bash
# staged 工作流单会话检索步数硬上限（阶段内上限：骨架≤3 / 要素≤8 / 验证≤8 / 补查≤4）
AGENT_STAGED_MAX_RETRIEVALS=24
# staged 工作流每个检索步最多同时查询的 KB 数（超限按 agent_priority 择优并上报）
AGENT_STAGED_MAX_KBS_PER_STEP=4
```

### 9.4 KB Agent Profile（`/kbs/{kb_id}/agent-profile`）

KB Agent Profile 用于帮助 `/agent/query` 在多 KB 候选集中选库，不改变 KB RBAC、文档 enabled/archived 生命周期、metadata filters 或 doc-id scope。系统在文档构建到 `ready`、启用/停用或删除后异步触发后台 `agent_profile` job：先用 `PROFILE` 角色 LLM 为文档生成 `documents.metadata.agent_doc_profile`，再聚合为 KB 级 `knowledge_bases.metadata.agent_auto_profile`。生成失败只把 profile 标为 `failed`，不会使文档入库失败；查询时 profile 缺失会回退到 KB `name` + `description` + 人工字段。

后台生成的调度与规模行为：

- **去重合并**：同一 KB 已有排队/运行中的 `agent_profile` job 时，新的自动触发不再重复建 job（批量导入 N 篇文档只产生一条刷新链，而不是 N 个全量刷新）。`force=true` 的手动刷新仅被"运行中的 force job"合并。
- **脏标记与链式刷新**：文档事件写入独立的 `agent_auto_profile_dirty` 标记；刷新完成时若发现运行期间又有新事件（或仍有未生成 profile 的文档），会自动追加一次链式刷新（`reason=chained_refresh`），直到收敛。
- **全库聚合（三种模式，按规模自动切换）**：单次刷新最多新生成 24 篇文档的 profile，超出部分记入 `pending_document_profiles` 并由链式刷新继续处理；文档级 profile 按 `source_hash` + `index_hash` 缓存复用。KB 级聚合按已生成 profile 的文档数 N 自动选择模式（`auto.aggregation_mode`）：
  - `direct`（N ≤ 128）：全部文档 profile 逐条参与聚合；
  - `sampled`（N > 128 且回填未完成）：过渡模式，对全库做**等距抽样**取 128 条逐条参与（覆盖新旧文档，而非只取最新），另附全库 tag/domain 频次统计；
  - `grouped`（N > 128 且回填已收敛）：**分层摘要（map-reduce）**——文档按创建顺序切成 128 篇/组，每组由 PROFILE LLM 生成"组摘要"并缓存在组首文档的 `agent_group_profile` 元数据中（键含成员内容哈希，成员或其 profile 变化才重算），最终聚合全部组摘要 + 频次统计。追加式入库通常只需重算尾部一组，单篇变更的稳态成本为 O(1) 次组调用；组摘要覆盖**全部**已生成 profile 的文档，无新旧偏差。
- **输出韧性**：PROFILE LLM 输出超长字段/超量条目会被**截断**而不是判为失败；JSON 解析失败自动重试（最多 3 次）。
- **单篇失败容忍**：某一篇文档的 profile 生成失败（重试耗尽）不会使整个刷新失败——该篇计入 `auto.document_profiles_failed`（明细在 `auto.failed_documents`，最多 5 条），KB 级 profile 用其余文档照常聚合写回；失败文档在下一次文档事件或手动刷新时重试（**不会**自动链式重试，避免对固定失败的文档形成重试风暴）。仅当**没有任何**可用文档 profile（全部失败且无缓存）时才判定刷新失败（`error_code=agent_profile_documents_failed`），保留醒目的失败信号。

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/kbs/{kb_id}/agent-profile` | 查看人工字段、自动字段和最终 effective profile；需 `kb_viewer`+ |
| `PUT` | `/kbs/{kb_id}/agent-profile` | 更新人工覆盖字段；需 `kb_editor`+ |
| `POST` | `/kbs/{kb_id}/agent-profile:refresh` | 人工触发后台重新生成自动 profile；需 `kb_editor`+ |

`GET /kbs/{kb_id}/agent-profile` 响应：

```json
{
  "kb_id": "kb_regulation",
  "manual": {
    "agent_description": "优先用于法规、合规、禁忌和限量判断",
    "agent_tags": ["法规", "合规"],
    "agent_priority": 10
  },
  "auto": {
    "schema_version": 1,
    "type": "kb_agent_auto_profile",
    "status": "ready",
    "description": "适合回答食品法规、合规限制和原料限量问题。",
    "tags": ["法规", "合规", "限量"],
    "domains": ["食品合规"],
    "sample_questions": ["某原料是否允许使用？"],
    "negative_scope": ["配方工艺细节"],
    "source_doc_count": 12,
    "profiled_doc_count": 12,
    "pending_document_profiles": 0,
    "document_profiles_failed": 0,
    "failed_documents": [],
    "aggregation_mode": "direct",
    "sampled_doc_count": 12,
    "group_count": 0,
    "updated_at": "2026-07-01T...",
    "job_id": "job_agent_profile_..."
  },
  "dirty": null,
  "effective": {
    "kb_id": "kb_regulation",
    "name": "法规库",
    "description": "regulations",
    "agent_description": "优先用于法规、合规、禁忌和限量判断",
    "agent_tags": ["法规", "合规"],
    "agent_priority": 10,
    "agent_auto_profile_status": "ready"
  }
}
```

字段说明：`dirty` 非 null 时表示有文档事件尚未反映到自动 profile（`{"dirty_at", "reason", "document_id"}`），此时 `effective.agent_auto_profile_status` 为 `dirty`；`pending_document_profiles > 0` 表示仍有文档 profile 待链式刷新生成。`aggregation_mode` 为本次聚合模式（`direct` / `sampled` / `grouped`），`grouped` 模式下 `group_count` 为组摘要数量、`sampled_doc_count` 为组摘要覆盖的文档总数（等于全部已生成 profile 的文档数）。

`PUT /kbs/{kb_id}/agent-profile` 请求体支持部分字段更新；显式 `null` 或空列表会清除对应人工覆盖，使 effective profile 回落到自动字段。除 `agent_description` / `agent_tags` / `agent_priority` 外，三个选库关键的自动字段也支持人工覆盖（人工非空则覆盖自动值）：`agent_domains`、`agent_sample_questions`、`agent_negative_scope`——典型用途是纠正自动生成的 `negative_scope` 错误地把本库排除出某类问题。

```json
{
  "agent_description": "优先用于法规和合规判断",
  "agent_tags": ["法规", "合规", "限量"],
  "agent_priority": 10,
  "agent_domains": ["食品合规"],
  "agent_sample_questions": ["某原料在化妆品中的限量是多少？"],
  "agent_negative_scope": ["生产排产", "设备维护"]
}
```

`POST /kbs/{kb_id}/agent-profile:refresh` 请求体：

```json
{
  "force": true,
  "idempotency_key": "manual-profile-refresh-20260701"
}
```

返回：

```json
{
  "job_id": "job_agent_profile_...",
  "job_type": "agent_profile",
  "status": "queued",
  "created": true
}
```

未携带 `idempotency_key` 的刷新请求，在该 KB 已有排队/运行中的 `agent_profile` job 时会合并到现有 job（返回 `created: false`）；`force: true` 仅与运行中的 force job 合并，否则会在当前 job 之后排队一次强制刷新。

### 9.5 图谱（无前缀）

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/graph/label/list` | 全部节点标签 |
| `GET` | `/graph/label/popular` | 高频标签 |
| `GET` | `/graph/label/search` | 标签搜索 |
| `GET` | `/graphs` | 子图查询 |
| `GET` | `/graph/entity/exists` | 实体存在性检查 |
| `POST` | `/graph/entity/edit` | 编辑实体 |
| `POST` | `/graph/entity/create` | 新建实体 |
| `POST` | `/graph/entities/merge` | 合并实体 |
| `POST` | `/graph/relation/edit` | 编辑关系 |
| `POST` | `/graph/relation/create` | 新建关系 |
| `DELETE` | `/graph/entity/delete` | 删除实体及其关系 |
| `DELETE` | `/graph/relation/delete` | 删除关系 |

### 9.6 Ollama 兼容（`/api`）

挂载 `OllamaAPI`，对外提供与 Ollama 接口兼容的端点：

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/version` | Ollama 兼容版本信息 |
| `GET` | `/api/tags` | 模型列表 |
| `GET` | `/api/ps` | 运行中模型列表 |
| `POST` | `/api/generate` | Ollama generate 兼容接口 |
| `POST` | `/api/chat` | Ollama chat 兼容接口 |

默认 `WHITELIST_PATHS` 仅放行 `/health`；如果要让 Ollama 兼容端点免 API Key，需要显式配置 `WHITELIST_PATHS=/health,/api/*`。企业模式下 `/api/*` 属于受保护前缀，不能被 whitelist 或全局 API key 默认绕过。

### 9.7 状态与认证基础接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 系统健康、配置和队列状态；`chat_memory` 当前返回 `{enabled, available, pending_tasks, worker_running, extraction_fingerprint, graph_store_fingerprint}`。其中 `pending_tasks` 仅是进程内兼容任务数，durable outbox backlog 以管理恢复接口返回的状态计数为准；默认 whitelist 放行 |
| `GET` | `/metrics` | Prometheus text format 指标（KB/doc/job/audit gauge + process-local HTTP counter/histogram）；受 `combined_auth` 保护，默认不在 whitelist；单服务器部署配套告警/SLO/dashboard 见 `deploy/monitoring/` |
| `GET` | `/auth-status` | 认证模式状态；非企业模式下可能签发 guest token |
| `POST` | `/login` | 非企业模式下使用 `AUTH_ACCOUNTS`；企业模式下使用企业用户表 |

Chat Memory 运维信号：`enabled` 是新消息 admission/自动召回开关，`available` 表示当前 Graphiti/Neo4j backend slot 是否可用，`worker_running` 表示本进程 durable outbox consumer 是否运行；两个 fingerprint 只返回缩短后的 extraction/graph-store 身份用于部署核对。`/health` 不扫描 PostgreSQL outbox；需要 durable `pending/running/retry_wait/dead_letter` 数量、最老可执行事件和 lag 时，使用 super-admin `POST /admin/chat-memory:backlog-scan`。

---

## 十、企业模式 Auth / Admin

本节接口仅在 `LIGHTRAG_ENTERPRISE_AUTH_ENABLED=true` 时挂载或启用。企业模式默认禁用 guest 对受保护 API 的访问；`LIGHTRAG_API_KEY` 默认不能绕过 RBAC；`WHITELIST_PATHS` 不能放行 `/kbs`、`/documents`、`/query`、`/agent`、`/graph`、`/api`、`/chat` 等受保护前缀。企业 service API key 使用同一 `X-API-Key` 请求头，但与全局 `LIGHTRAG_API_KEY` 分离：只按持久化 hash 查找，默认只拥有创建时授予的 `kb_roles` scope；设置 `tenant_id + inherit_tenant_kb_acl=true` 时可显式继承 tenant-scoped KB ACL。service key 不能成为 super admin，撤销后立即失效。

### 10.1 配置项

```env
LIGHTRAG_ENTERPRISE_AUTH_ENABLED=true
TOKEN_SECRET=<non-default-secret>
LIGHTRAG_SUPER_ADMIN_USERNAME=admin
LIGHTRAG_SUPER_ADMIN_PASSWORD_HASH={bcrypt}$2b$12$...
# 或仅首次开发引导使用：LIGHTRAG_SUPER_ADMIN_PASSWORD=change-me
LIGHTRAG_USER_REGISTRATION_ENABLED=false
# 可选运行时注册模式：disabled / open / invite_only / admin_approval
LIGHTRAG_ENTERPRISE_DISABLE_GLOBAL_ROUTES=true
LIGHTRAG_ENTERPRISE_LEGACY_API_KEY_SUPERADMIN=false
# artifact 下载/预签名 URL 的最低 KB role；默认 kb_viewer 保持旧行为，可设 kb_editor/kb_admin/kb_owner
LIGHTRAG_ENTERPRISE_ARTIFACT_DOWNLOAD_MIN_ROLE=kb_viewer
# artifact type 级别最低角色覆盖，JSON object；key 可为 original/markdown/raw_dir/... 或 *
LIGHTRAG_ENTERPRISE_ARTIFACT_DOWNLOAD_POLICY={"original":"kb_editor"}
# artifact action 级别最低角色覆盖，JSON object；action 为 download/download-url/preview，内层 key 可为 artifact type 或 *
LIGHTRAG_ENTERPRISE_ARTIFACT_ACTION_POLICY={"preview":{"*":"kb_editor"},"download":{"original":"kb_editor"}}
# 企业响应中默认隐藏本地 path / object_uri / object_prefix_uri
LIGHTRAG_ENTERPRISE_MASK_STORAGE_URIS=true

# 企业 KB 控制面（生产使用 PostgreSQL）
LIGHTRAG_KB_METADATA_BACKEND=postgres
LIGHTRAG_KB_POSTGRES_POOL_MIN_SIZE=1
LIGHTRAG_KB_POSTGRES_POOL_MAX_SIZE=10
# generation fence / per-job owner advisory lock 的独立连接池
LIGHTRAG_KB_POSTGRES_OPERATION_LOCK_POOL_MAX_SIZE=10

# durable job worker 与 orphan recovery
LIGHTRAG_KB_JOB_WORKER=true
LIGHTRAG_KB_JOB_WORKER_POLL_SECONDS=1.0
LIGHTRAG_KB_JOB_WORKER_GRACE_SECONDS=5.0
LIGHTRAG_KB_JOB_RECOVERY_INTERVAL_SECONDS=30.0
LIGHTRAG_KB_JOB_RECOVERY_GRACE_SECONDS=5.0

# 默认关闭的企业请求限流/配额；开启后在认证与 RBAC 通过后计数
LIGHTRAG_ENTERPRISE_RATE_LIMIT_ENABLED=false
# user / service key / legacy enterprise API key principal 固定窗口请求限流
LIGHTRAG_ENTERPRISE_RATE_LIMIT_REQUESTS=60
LIGHTRAG_ENTERPRISE_RATE_LIMIT_WINDOW_SECONDS=60
# tenant_id 固定窗口请求限流；0 表示禁用 tenant 维度
LIGHTRAG_ENTERPRISE_TENANT_RATE_LIMIT_REQUESTS=0
LIGHTRAG_ENTERPRISE_TENANT_RATE_LIMIT_WINDOW_SECONDS=60
# principal 固定窗口 quota；0 表示禁用 quota 维度
LIGHTRAG_ENTERPRISE_QUOTA_REQUESTS=0
LIGHTRAG_ENTERPRISE_QUOTA_WINDOW_SECONDS=86400
# tenant_id 固定窗口 quota；0 表示禁用 tenant quota
LIGHTRAG_ENTERPRISE_TENANT_QUOTA_REQUESTS=0
LIGHTRAG_ENTERPRISE_TENANT_QUOTA_WINDOW_SECONDS=86400

# 登录失败锁定（企业 /login）；默认开启，MAX_ATTEMPTS=0 关闭
LIGHTRAG_ENTERPRISE_LOGIN_MAX_ATTEMPTS=10
LIGHTRAG_ENTERPRISE_LOGIN_WINDOW_SECONDS=300
LIGHTRAG_ENTERPRISE_LOGIN_LOCKOUT_SECONDS=900
# 注册失败锁定（企业 /auth/register）；默认开启，MAX_ATTEMPTS=0 关闭
LIGHTRAG_ENTERPRISE_REGISTRATION_MAX_ATTEMPTS=10
LIGHTRAG_ENTERPRISE_REGISTRATION_WINDOW_SECONDS=300
LIGHTRAG_ENTERPRISE_REGISTRATION_LOCKOUT_SECONDS=900
# 每 principal / tenant 的在途 job 并发配额；0 表示禁用
LIGHTRAG_ENTERPRISE_MAX_CONCURRENT_JOBS=0
LIGHTRAG_ENTERPRISE_TENANT_MAX_CONCURRENT_JOBS=0

# Agent 查询模式：默认关闭；开启后仍需用户/服务密钥具备 can_use_agent_query
LIGHTRAG_AGENT_QUERY_ENABLED=false
AGENT_MAX_ROUNDS=5
# staged 工作流（workflow="staged"）单会话检索步数硬上限
AGENT_STAGED_MAX_RETRIEVALS=24
# staged 工作流每个检索步最多同时查询的 KB 数
AGENT_STAGED_MAX_KBS_PER_STEP=4
AGENT_WORKFLOW_PROMPT_MAX_LENGTH=16384
LIGHTRAG_AGENT_PROFILE_AUTO_REFRESH=true
AGENT_PROFILE_REFRESH_DOC_DELTA=1
AGENT_PROFILE_REFRESH_MIN_INTERVAL_SECONDS=0

# Agent 编排角色 LLM：通常与 QUERY 使用同一本地 OpenAI-compatible 模型
AGENT_LLM_BINDING=openai
AGENT_LLM_BINDING_HOST=http://127.0.0.1:8000/v1
AGENT_LLM_BINDING_API_KEY=local-dummy-key
AGENT_LLM_MODEL=qwen3.6-36b
AGENT_MAX_ASYNC_LLM=1
AGENT_LLM_TIMEOUT=300

# Profile 生成角色 LLM：用于文档级 profile → KB 级自动 Agent Profile
PROFILE_LLM_BINDING=openai
PROFILE_LLM_BINDING_HOST=http://127.0.0.1:8000/v1
PROFILE_LLM_BINDING_API_KEY=local-dummy-key
PROFILE_LLM_MODEL=qwen3.6-36b
PROFILE_MAX_ASYNC_LLM=1
PROFILE_LLM_TIMEOUT=300
```

### 10.2 登录与当前用户

当前企业认证支持企业用户表登录、JWT、service/scoped API key。SSO/OIDC/SAML **明确不做**：本系统按单机内网部署，无外部 IdP 接入需求（见 `docs/设计方案.md` §2.2）。

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/auth-status` | 企业模式返回 `auth_mode=enterprise`、当前注册开关与 `chat_memory_enabled`（前端据此决定是否显示记忆 UI）；不签发 guest token |
| `POST` | `/login` | 使用企业用户表认证，返回带 `user_id`、`system_role`、`token_version` metadata 的 JWT；同一用户名连续登录失败达 `LIGHTRAG_ENTERPRISE_LOGIN_MAX_ATTEMPTS`（默认 10）后锁定 `LIGHTRAG_ENTERPRISE_LOGIN_LOCKOUT_SECONDS`（默认 900s），期间返回 `429` + `Retry-After`；成功登录清零计数，`MAX_ATTEMPTS=0` 关闭锁定 |
| `POST` | `/auth/register` | 注册新用户；行为随注册模式而定：`open` 直接创建 active 用户并返回 token；`invite_only` 必须携带有效 `invitation_token`；`admin_approval` 创建 `pending` 用户、待管理员 `:enable` 审批后才能登录（响应不含 token）；`disabled` 返回 `403`。注册失败按 `LIGHTRAG_ENTERPRISE_REGISTRATION_*` 做单进程 per-username 锁定，失败/触发锁定写审计。新用户默认无 KB 权限且不可创建 KB |
| `GET` | `/auth/me` | 返回当前用户与 principal 权限信息；service API key 请求返回 `user:null` 与 service-key principal payload；用户对象含只读 `display_name` / `email` 个人资料字段 |
| `PATCH` | `/auth/me` | 当前用户维护个人资料：`display_name`（≤64 字符）/ `email`（≤254 字符、需含 `@`）。omitted=不变、显式 `null`=清除、空白串 `400`；存入 `enterprise_users.metadata`，**不**递增 `token_version`（当前 token 继续有效）；仅交互式 JWT 用户，API-key principal 返回 `403`；审计 `user_profile_updated` |
| `POST` | `/auth/logout` | 全设备登出：递增本人 `token_version`，使包括当前 token 在内的全部已签发 JWT 立即失效；返回 `{"status":"logged_out","token_version":N}`；仅交互式 JWT 用户，service key 返回 `403`（撤销 key 请用 `:revoke`）；审计 `user_logged_out` |
| `POST` | `/auth/change-password` | 当前用户修改密码；成功后 `token_version` 增加，旧 token 失效 |
| `GET` | `/auth/me/kbs/{kb_id}/query-settings` | 读取当前用户在指定 KB 下的个人查询设置；需 `kb_viewer`+；非交互式用户/API-key principal 返回 `403` |
| `PUT` | `/auth/me/kbs/{kb_id}/query-settings` | 写入/覆盖当前用户在指定 KB 下的个人 `user_prompt`；需 `kb_viewer`+；非交互式用户/API-key principal 返回 `403` |
| `GET` | `/auth/me/agent-workflow-prompt` | 读取当前用户的 Agent 工作流提示词；仅交互式 JWT 用户 |
| `PUT` | `/auth/me/agent-workflow-prompt` | 写入/清空当前用户的 Agent 工作流提示词；空字符串表示清空；最大长度由 `AGENT_WORKFLOW_PROMPT_MAX_LENGTH` 控制；仅交互式 JWT 用户 |

`GET /auth-status` 响应形态：企业模式返回 `auth_mode="enterprise"` 与注册开关且不签发 guest token；禁用认证时返回 guest bearer token；普通认证模式返回认证开关与登录入口信息。

注册模式（由 super admin 通过 `PATCH /admin/settings/registration` 实时切换）：

- `open`：`POST /auth/register {username, password}` 创建 active 用户并返回登录响应（含 `access_token`）。
- `invite_only`：请求体必须带 `invitation_token`（由 `POST /admin/invitations` 颁发，仅创建时返回一次）；token 经原子校验（active + 未过期 + 单次使用）后创建 active 用户并返回 token，无效/过期/已用/已撤销 token 返回 `403`。
- `admin_approval`：创建 `status=pending` 用户，响应为 `{"auth_mode":"enterprise","status":"pending","user":{...},"message":...}`（无 token）；该用户在管理员 `POST /admin/users/{user_id}:enable` 审批前无法登录。
- `disabled`：`POST /auth/register` 返回 `403`。

```json
// POST /auth/register（invite_only 模式）
{"username":"alice","password":"change-me","invitation_token":"lrinv_inv_..."}
```

认证请求示例：

```http
POST /login
Content-Type: application/x-www-form-urlencoded

username=admin&password=change-me
```

```json
// POST /auth/register
{"username":"alice","password":"change-me"}

// POST /auth/change-password
{"current_password":"old-pass","new_password":"new-pass"}

// PUT /auth/me/kbs/kb_research/query-settings
{"user_prompt":"请优先使用中文回答，并给出引用依据"}

// GET/PUT /auth/me/kbs/kb_research/query-settings
{"user_id":"usr_alice","kb_id":"kb_research","user_prompt":"请优先使用中文回答，并给出引用依据"}

// GET/PUT /auth/me/agent-workflow-prompt
{"user_id":"usr_alice","workflow_prompt":"涉及法规时必须先查询法规库，并将该步骤标记为 P0。"}
```

当前用户 KB 查询设置说明：

- 存储维度为 `user_id + kb_id`，不同企业用户、不同知识库互不影响；默认 `user_prompt` 为空字符串。
- 读取或写入前会校验 KB 存在和当前 principal 至少拥有 `kb_viewer` 角色；无权限返回 `403`，KB 不存在返回 `404`。
- service/scoped API key 与 legacy enterprise API key superadmin 不属于交互式用户，不能使用该 self-service 设置接口，也不会在 query 时套用个人 `user_prompt`。
- `PUT` 传空字符串可清空个人提示词。清空后 query 回退到 active KB config 的 `query_config.user_prompt`；若也未配置则为空。

### 10.3 管理接口

以下 `/admin` 接口均需 super admin；唯一兼容例外是精确的 `GET /admin/tenants/{tenant_id}`，目标 tenant 的 `tenant_admin` / `tenant_owner` 也可读取本部门详情。其余 `/admin/*` 不因 tenant role 放宽：

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/admin/settings/registration` | 读取实时注册策略，返回 `enabled` 与 `mode` |
| `GET` | `/admin/overview` | 平台总览 JSON 聚合：KB 状态分布、文档/job/artifact 全局聚合与计数器合计、dead-letter 总数、企业用户/租户/service key/审计事件计数、**项目记忆全局统计**（`chat_memory`：`enabled`、`available`、兼容 `pending_tasks`、`episode_count`、`user_count`、`project_count`）；仅查控制面，不加载引擎实例。durable outbox 状态使用下述 backlog recovery 接口读取 |
| `PATCH` / `PUT` | `/admin/settings/registration` | 更新实时注册策略，body：`{"enabled": true}` 或 `{"mode":"open"}` |
| `GET` | `/admin/users` | 列出企业用户；支持 `status`/`tenant_id`/`q`(用户名子串) 过滤与 `limit`/`offset` 分页 |
| `POST` | `/admin/users` | 创建用户，可设置 `can_create_kb`、`can_use_bypass_query`、`can_use_agent_query`、`can_delete_documents`、`can_download_files`、`tenant_id` |
| `GET` | `/admin/users/{user_id}` | 查询用户详情 |
| `GET` | `/admin/users/{user_id}/access` | 查看用户的访问总览：全局能力 + 租户成员关系(role) + 直接 KB ACL(kb_id/role，不含租户继承的有效角色) |
| `PATCH` | `/admin/users/{user_id}` | 更新用户状态/能力/tenant/password；请求体包含 `status`、`can_create_kb`、`can_use_bypass_query`、`can_use_agent_query`、`can_delete_documents`、`can_download_files` 任一非 null 字段、显式给出 `tenant_id` 或修改 `password`，都会增加 `token_version` 并使旧 token 失效。`tenant_id` 区分 omitted 与显式 null：省略=不变，显式 `null`=清空租户归属，空/空白字符串返回 `400` |
| `POST` | `/admin/users/{user_id}:disable` | 禁用用户并递增 `token_version`，旧 token 失效 |
| `POST` | `/admin/users/{user_id}:enable` | 启用用户并递增 `token_version` |
| `POST` | `/admin/users/{user_id}:reset-password` | 重置用户密码并递增 `token_version` |
| `DELETE` | `/admin/users/{user_id}` | 删除用户；在同一 PostgreSQL 事务中级联清理租户 membership、KB ACL、个人查询设置、对话项目/会话/消息，并为已有记忆组持久化 purge outbox 事件；事件不外键级联，源用户/项目删除后仍可恢复；不允许删除 super admin |
| `POST` | `/admin/users/{user_id}/chat-memory:purge` | **持久化排队用户项目 purge**；使用 maintenance service，不要求 read/ingest 开关仍开启。body 可选 `{"project_ids":[...]}`：显式列表去重并逐项校验属于该用户，省略/空列表枚举该用户当前全部项目；返回 `{queued, noop, project_ids}`。用户/项目不存在 `404`，maintenance 未挂载 `503`，graph-store 不匹配等写冲突 `409` |
| `POST` | `/admin/chat-memory:backlog-scan` | **durable stale-claim recovery + worker wakeup**，不是旧 seq 水位重摄取扫描。body 可选 `{"limit":100}`（`1..1000`）；尝试恢复 stale `running` 事件、nudge worker，并返回 `{recovered_events, outbox:{pending,running,retry_wait,dead_letter,oldest_available_at,oldest_lag_seconds}}`；maintenance/worker 未挂载 `503` |
| `POST` | `/admin/chat-memory/events/{event_id}:retry` | **按 durable event ID 恢复 purge**；可在源用户/项目已删除后使用。仅 `purge` 事件可操作：`dead_letter` 立即重排，`pending/retry_wait` 幂等返回并唤醒 worker；不存在 `404`，非 purge、`running/succeeded/superseded` 或状态竞争返回 `409`。若事件属于另一 graph store，返回 `409 chat_memory_old_graph_store_required`，须恢复原 `MEMORY_NEO4J_DEPLOYMENT_ID`/Neo4j backend 后重试 |
| `POST` | `/admin/users/{user_id}/kb-access:batch-set` | 按用户维度批量 grant/revoke 多个 KB ACL；与按 KB 维度 `/admin/kbs/{kb_id}/acl:batch-set` 互补 |
| `POST` | `/admin/tenants` | 创建租户实体（可指定 `tenant_id`，省略则生成 `tenant_<hex>`；重复 `409`） |
| `GET` | `/admin/tenants` | 列出所有租户 |
| `GET` | `/admin/tenants/{tenant_id}` | 租户详情 + 总览（含 `member_count` / `kb_count`）；`kb_count` 是 active 的 tenant-owned KB 与 super admin 通过 tenant ACL 下发 KB 的去重并集；本 tenant admin/owner 也可读取 |
| `PATCH` | `/admin/tenants/{tenant_id}` | 更新租户 `name`/`description`/`status`（`active`/`disabled`） |
| `DELETE` | `/admin/tenants/{tenant_id}` | 删除租户实体；仅当无任何引用（成员/租户内 KB/归属用户/tenant-KB ACL）时允许，否则 `409`（不级联） |
| `GET` | `/admin/tenants/{tenant_id}/kbs` | 列出该租户下的 KB（`id`/`name`/`status`/`visibility`/`owner_id`） |
| `GET` | `/admin/tenants/{tenant_id}/members` | 列出 tenant 成员与 tenant role；每条 membership 附带解析后的 `username` / `display_name` / `user_status` |
| `PUT` | `/admin/tenants/{tenant_id}/members/{user_id}` | 写入/更新 tenant membership，body：`{"role":"tenant_member"}`；响应同样附带 `username` / `display_name` / `user_status` |
| `DELETE` | `/admin/tenants/{tenant_id}/members/{user_id}` | 删除 tenant membership |
| `GET` | `/admin/kbs/{kb_id}/acl` | 查看 KB ACL，返回 user 与 tenant 两类 principal；KB 不存在时返回 404 |
| `PUT` | `/admin/kbs/{kb_id}/acl` | 授权 KB 角色，body：`{"user_id":"usr_...","role":"kb_viewer"}` 或 `{"tenant_id":"tenant-a","role":"kb_viewer"}`；KB 不存在时返回 404 |
| `POST` | `/admin/kbs/{kb_id}/acl:batch-set` | 批量授权/撤销指定 KB 的 user/tenant ACL |
| `DELETE` | `/admin/kbs/{kb_id}/acl/{user_id}` | 撤销用户对 KB 的 ACL；KB 不存在时返回 404 |
| `DELETE` | `/admin/kbs/{kb_id}/acl/tenants/{tenant_id}` | 撤销 tenant 对 KB 的 ACL；KB 不存在时返回 404 |
| `GET` | `/admin/service-api-keys` | 列出 service/scoped API key；响应不包含 raw key 或 hash |
| `POST` | `/admin/service-api-keys` | 创建 service/scoped API key；raw key 仅在创建响应返回一次；可选 `expires_in_seconds` 设置过期 |
| `POST` | `/admin/service-api-keys/{key_id}:rotate` | 轮换 service/scoped API key；返回新 raw key 一次，可选撤销旧 key |
| `POST` | `/admin/service-api-keys/{key_id}:revoke` | 撤销 service/scoped API key；撤销后下一次请求立即失效 |
| `GET` | `/admin/invitations` | 列出注册邀请（不含 raw token，仅 `token_preview`） |
| `POST` | `/admin/invitations` | 颁发单次注册邀请（`invite_only` 模式用）；raw `invitation_token` 仅创建响应返回一次，可选 `expires_in_seconds` |
| `POST` | `/admin/invitations/{invitation_id}:revoke` | 撤销邀请；已 used/revoked 的邀请保持终态 |
| `GET` | `/admin/audit-events` | 查询审计事件；支持 `limit`/`offset` 分页与 `event_type`/`actor_user_id`/`target_type`/`target_id`/`created_after`/`created_before` 过滤 |

以下 self-service tenant 接口需当前用户是目标 tenant 的 `tenant_admin` 或 `tenant_owner`（super admin 也可用）：

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/tenants/{tenant_id}` | 本 tenant 详情，含 `member_count` 与 active `kb_count` |
| `GET` | `/tenants/{tenant_id}/kbs` | 本 tenant 可管理/已下发的 active KB 摘要列表 |
| `GET` | `/tenants/{tenant_id}/users` | 列出 canonical tenant assignment 与 membership 都属于本 tenant 的非 super 用户；支持 `status`、`q`、`limit`、`offset` |
| `POST` | `/tenants/{tenant_id}/users` | 创建本部门普通用户；可设置五个能力位：`can_create_kb`、`can_delete_documents`、`can_use_bypass_query`、`can_use_agent_query`、`can_download_files`，默认均为 false |
| `GET` | `/tenants/{tenant_id}/users/{user_id}` | 查询本部门用户详情 |
| `PATCH` | `/tenants/{tenant_id}/users/{user_id}` | 更新普通成员状态及五个能力位；不能在此修改 `tenant_id` 或密码 |
| `POST` | `/tenants/{tenant_id}/users/{user_id}:enable` | 启用本部门普通成员并使旧 JWT 失效 |
| `POST` | `/tenants/{tenant_id}/users/{user_id}:disable` | 禁用本部门普通成员并使旧 JWT 失效 |
| `POST` | `/tenants/{tenant_id}/users/{user_id}:reset-password` | 重置本部门普通成员密码并使旧 JWT 失效 |
| `DELETE` | `/tenants/{tenant_id}/users/{user_id}` | 删除本部门普通成员及其 membership/ACL/个人设置/对话数据等关联记录 |
| `GET` | `/tenants/{tenant_id}/members` | 查看本 tenant membership；每条记录附带 `username` / `display_name` / `user_status` |
| `PUT` | `/tenants/{tenant_id}/members/{user_id}` | 仅把当前无租户归属的普通用户授予为 `tenant_member`；不能跨租户搬迁或提升为 admin/owner |
| `DELETE` | `/tenants/{tenant_id}/members/{user_id}` | 仅撤销普通 `tenant_member`；不能撤销自己、tenant admin/owner 或 super admin |
| `GET` | `/tenants/{tenant_id}/audit-events` | 查询事件发生时 `actor_tenant_id` 为本 tenant 的审计事件；过滤和分页参数与 `/admin/audit-events` 一致 |
| `GET` | `/tenants/{tenant_id}/kbs/{kb_id}/members` | 列出本部门普通成员对该 KB 的 override/effective role 与来源 |
| `PUT` | `/tenants/{tenant_id}/kbs/{kb_id}/members/{user_id}` | 设置成员 `viewer` / `editor` / `admin` allow override；对 platform-provisioned KB 不能超过当前 tenant ACL role |
| `DELETE` | `/tenants/{tenant_id}/kbs/{kb_id}/members/{user_id}` | 默认写入 deny，仅撤销 tenant-derived access；加 `?reset=true` 删除 override，恢复当前 tenant ACL 继承 |

Tenant membership 响应对象字段：`tenant_id`、`user_id`、`role`、`granted_by`、`created_at`、`updated_at`，以及读取时从用户记录解析出的 `username`（用户名）、`display_name`（用户自助资料中的显示名，未设置时为 `null`）、`user_status`（`active` / `disabled` / `pending`）；对应用户记录已被删除时这三个解析字段为 `null`。适用于 `/admin/tenants/{tenant_id}/members` 与 `/tenants/{tenant_id}/members` 的 GET 列表响应及 PUT 授予响应。

租户用户管理边界：目标必须在事务提交时仍是本 tenant 的普通 `tenant_member`；tenant admin 不能修改或删除自己、`tenant_admin`、`tenant_owner`、super admin，也不能通过 scoped API 迁移用户到其他 tenant。用户快照、membership、角色与写入采用事务级 CAS；并发迁移/晋升发生时返回 `409`，不会把旧快照写回覆盖新状态。

KB ACL 角色使用规范名称：`kb_viewer`、`kb_editor`、`kb_admin`、`kb_owner`。平台来源角色取 direct user ACL 与 visibility 隐含角色最高值；tenant 来源另按 KB provenance 计算：tenant-owned KB 只有显式 allow override 才给普通成员角色，deny/无 override 均不授予，但 `tenant_admin` / `tenant_owner` 对本租户 tenant-owned KB **始终**保底隐含 `kb_viewer`（source=`tenant_admin_oversight`，deny override 也不能移除；显式 allow override 给出的更高角色优先）；platform-provisioned KB 无 override 时继承当前 tenant ACL，allow 不能超过当前 tenant ACL，deny 只压制 tenant-derived access，`reset=true` 后恢复当前继承。最终 effective role 是 platform 与 tenant contribution 的最高值，因此 tenant admin 的 revoke 不会删除 super admin 的 direct grant 或 public/internal visibility。Service/scoped API key 默认只按自身 `kb_roles` scope；设置 `tenant_id + inherit_tenant_kb_acl=true` 时才显式继承 tenant ACL。

知识库生命周期权限不等同于 KB role：租户管理员只能软删除、硬删除或恢复 `origin="tenant"` 且 `tenant_id` 为本部门的 KB；super admin 下发的 `origin="platform"` KB 即使 tenant ACL 为 `kb_admin` 也不能由租户管理员删除。`origin` 是不可变 catalog 字段，历史缺失值 fail-safe 为 `platform`。

KB ACL 请求/响应约束：

- `PUT /admin/kbs/{kb_id}/acl` 请求体必须且只能包含 `user_id` 或 `tenant_id` 之一，并必须包含 `role`。
- `POST /admin/kbs/{kb_id}/acl:batch-set` 的每个 entry 必须且只能包含 `user_id` 或 `tenant_id` 之一；`action` 默认为 `grant`，取值 `grant` / `revoke`；`grant` 必须提供 `role`，`revoke` 不需要 `role`。
- ACL 响应对象字段为：`kb_id`、`user_id`、`tenant_id`、`principal_type`（`user` / `tenant`）、`role`、`granted_by`、`created_at`、`updated_at`。
- `batch-set` 响应中 `granted` 为 ACL 响应对象数组；`revoked` 为本次实际删除成功的 user id 或 tenant id 字符串数组。

`GET /admin/audit-events` 返回全平台审计事件，按 `created_at DESC, id DESC` 排序；`limit` 默认 `100`，服务端会 clamp 到 `1..500`，`offset` 默认 `0` 用于分页。可选过滤参数（精确匹配，组合为 AND）：`event_type`、`actor_user_id`、`target_type`、`target_id`；时间范围 `created_after` / `created_before` 为 ISO-8601 字符串，按字典序与 `created_at` 比较（`>=` / `<=`）。`GET /tenants/{tenant_id}/audit-events` 使用同一过滤集，但服务端额外强制 `actor_tenant_id=tenant_id`；该字段是事件写入时的 canonical tenant 快照，用户后续调换部门不会改变历史可见性，旧记录若为 null 不会被猜测归属。例：`GET /admin/audit-events?event_type=kb_deleted&actor_user_id=usr_x&created_after=2026-06-01T00:00:00Z&limit=50&offset=50`。

审计事件响应字段：

```json
{
  "id": "audit_...",
  "event_type": "kb_created",
  "actor_user_id": "usr_...",
  "actor_tenant_id": "tenant-a",
  "actor_username": "张三",
  "target_type": "kb",
  "target_id": "kb_...",
  "target_name": "橡胶研究知识库",
  "metadata": {},
  "created_at": "2026-06-08T...Z"
}
```

- `actor_tenant_id`：事件发生时操作者 canonical tenant 的持久化快照；tenant 审计接口按此字段做数据库过滤。
- `actor_username`：由 `actor_user_id` 在读取时动态解析（对齐 `EnterpriseUserResponse.username`），未找到时为 `null`。
- `target_name`：根据 `target_type` 动态解析——`kb` 返回知识库名称、`user` 返回用户名、`tenant` 返回租户名称（对齐各实体的 `name` / `username` 字段）；未找到时为 `null`。

企业模式已实现的审计事件类型包括：

- 登录/注册设置：`login_success`、`login_failed`、`registration_failed`、`registration_locked`、`registration_setting_updated`
- super admin bootstrap/sync：`super_admin_bootstrapped`、`super_admin_synced`
- 用户管理：`user_created`、`user_updated`、`user_deleted`、`user_password_changed`、`user_profile_updated`、`user_logged_out`、`user_agent_workflow_prompt_updated`
- service API key：`service_api_key_created`、`service_api_key_rotated`、`service_api_key_revoked`
- KB ACL / tenant：`kb_acl_granted`、`kb_acl_revoked`、`tenant_created`、`tenant_updated`、`tenant_deleted`、`tenant_membership_granted`、`tenant_membership_revoked`、`tenant_kb_acl_granted`、`tenant_kb_acl_revoked`
- 权限/限流/配额：`permission_denied`、`rate_limited`、`quota_exceeded`
- KB/config/query/Agent：`kb_created`、`kb_deleted`、`kb_hard_deleted`、`kb_restored`、`kb_visibility_changed`、`kb_config_activated`、`query_executed`、`query_stream_started`、`retrieve_executed`、`agent_session_started`、`agent_retrieve_round`、`agent_query_completed`、`agent_session_failed`
- KB 图谱编辑：`kb_graph_entity_edited`、`kb_graph_entity_created`、`kb_graph_entity_deleted`、`kb_graph_entities_merged`、`kb_graph_relation_edited`、`kb_graph_relation_created`、`kb_graph_relation_deleted`
- 用户对话管理：`chat_project_created`、`chat_project_renamed`、`chat_project_deleted`、`chat_session_created`、`chat_session_updated`、`chat_session_deleted`、`chat_messages_appended`、`chat_message_deleted`（metadata 仅记录 id/计数/标志，不记录项目、会话名称或消息正文）
- 项目记忆（chat memory，当前 durable 路径）：成功执行的显式/自动 search 写 `chat_memory_searched`（记录 query hash/计数，不记录 query 或事实正文）；管理员排队 purge 与按事件恢复分别写 `chat_memory_purge_queued`、`chat_memory_purge_retry_queued`。消息追加/删除仍由 `chat_messages_appended`、`chat_message_deleted` 等源数据审计覆盖，durable outbox/worker 负责执行，不再把旧 fire-and-forget `chat_memory_ingested` / `chat_memory_forgotten` 当作可靠性或完成性凭据。
- 自动注入附加到 `query_executed`、`query_stream_started`、`multi_kb_query_executed`、`multi_kb_query_stream_started`、`agent_query_completed` 的记忆投影严格限于 `memory_enabled`、`memory_fact_count`、`memory_injected_count`、`memory_status`、`memory_truncated`、`memory_reason`；不会复制 memory project ID、Graphiti fact ID/UUID、`M*` reference ID/数组或事实文本。`memory_status`/`memory_reason` 只接受 §8.3 的冻结枚举。
- artifact/job/document 类事件：`artifact_downloaded`、`artifact_previewed`、`artifact_download_url_created`、`kb_rebuild_queued`、`job_cancel_requested`、`job_retry_queued`、`document_batch_enabled`、`document_batch_disabled`，以及文档 upload/texts/urls/import/scan/sync/patch/enable/disable/replace/delete/batch-delete/parse/batch-parse/build/reindex/batch-build/batch-reindex/rebuild 相关事件。

审计覆盖：企业模式下，KB 创建/删除、KB Agent Profile 人工更新/刷新排队、config 激活、query/query-stream/retrieve、artifact download/preview/download-url、文档 upload/texts/urls/import/scan/sync/patch/enable/disable/replace/delete/batch-delete/parse/batch-parse/build/reindex/batch-build/batch-reindex/rebuild，以及 job cancel/retry 均写入 audit event。审计 metadata 采用白名单字段：query 仅记录 `query_hash`、mode、过滤摘要；文档与 artifact 事件仅记录 job/batch/document/artifact id、count、flag、hash、size/type 等，不记录 raw query、上传正文、URL、local path、presigned URL、密码/token/API key 明文。

管理请求体示例：

```json
// PATCH /admin/settings/registration
{"enabled": true}

// PATCH /admin/settings/registration
{"mode": "invite_only"}
// 返回：{"enabled": false, "mode": "invite_only"}

// POST /admin/users
{"username":"bob","password":"bob-pass","can_create_kb":true,"can_use_bypass_query":false,"can_use_agent_query":true,"can_delete_documents":false,"can_download_files":false,"tenant_id":null}

// PATCH /admin/users/{user_id}
{"status":"active","can_create_kb":false,"can_use_bypass_query":true,"can_use_agent_query":true,"can_delete_documents":true,"can_download_files":true,"tenant_id":"tenant-a","password":"new-pass"}

// POST /admin/users/{user_id}:reset-password
{"password":"new-pass"}

// PUT /admin/kbs/{kb_id}/acl
{"user_id":"usr_...","role":"kb_viewer"}

// PUT /admin/tenants/{tenant_id}/members/{user_id}
{"role":"tenant_member"}

// POST /tenants/{tenant_id}/users
{"username":"alice","password":"change-me","can_create_kb":true,"can_use_bypass_query":false,"can_use_agent_query":true,"can_delete_documents":false,"can_download_files":true}

// PUT /tenants/{tenant_id}/kbs/{kb_id}/members/{user_id}
{"role":"editor"}

// DELETE /tenants/{tenant_id}/kbs/{kb_id}/members/{user_id}
// 默认写 deny；加 ?reset=true 删除 override 并恢复 tenant ACL 继承

// PUT /admin/kbs/{kb_id}/acl
{"tenant_id":"tenant-a","role":"kb_viewer"}

// POST /admin/kbs/{kb_id}/acl:batch-set
{"entries":[{"user_id":"usr_1","role":"kb_editor"},{"tenant_id":"tenant-a","role":"kb_viewer"},{"user_id":"usr_2","action":"revoke"},{"tenant_id":"tenant-b","action":"revoke"}]}
// 返回：{"granted":[...],"revoked":["usr_2","tenant-b"]}

// POST /admin/users/{user_id}/kb-access:batch-set
{"entries":[{"kb_id":"kb_a","role":"kb_viewer"},{"kb_id":"kb_b","role":"kb_editor"},{"kb_id":"kb_c","action":"revoke"}]}
// 返回：{"granted":[...],"revoked":["kb_c"]}

// POST /admin/service-api-keys
{"name":"ci-reader","kb_roles":{"kb_123":"kb_viewer"},"can_use_bypass_query":false,"can_use_agent_query":true,"inherit_tenant_kb_acl":false,"metadata":{"purpose":"ci"}}
// 返回：{"api_key":"lrsk_svc_key_...","key":{"id":"svc_key_...","key_preview":"...","status":"active", ...}}

// POST /admin/service-api-keys/{key_id}:rotate
{"expires_in_seconds":2592000,"revoke_old":true}
// 返回：{"api_key":"lrsk_svc_key_...","key":{"id":"svc_key_...","metadata":{"rotated_from":"svc_key_old"},"status":"active", ...}}

// POST /admin/service-api-keys/{key_id}:revoke
// 返回的 key.status 为 "revoked"
```

Service API key 行为约束：

创建请求：

```json
{
  "name": "ci-reader",
  "kb_roles": {"kb_123": "kb_viewer"},
  "can_use_bypass_query": false,
  "can_use_agent_query": true,
  "inherit_tenant_kb_acl": false,
  "tenant_id": null,
  "metadata": {"purpose": "ci"},
  "expires_in_seconds": 2592000
}
```

`expires_in_seconds` 可选（正整数）；服务端据此派生 `expires_at`（ISO-8601）。省略则 key 永不过期。

创建响应：

```json
{
  "api_key": "lrsk_svc_key_...",
  "key": {
    "id": "svc_key_...",
    "name": "ci-reader",
    "key_preview": "...",
    "status": "active",
    "created_by": "usr_...",
    "tenant_id": null,
    "scopes": {
      "kb_roles": {"kb_123": "kb_viewer"},
      "can_use_bypass_query": false,
      "can_use_agent_query": true,
      "inherit_tenant_kb_acl": false
    },
    "metadata": {"purpose": "ci"},
    "created_at": "2026-06-08T...Z",
    "updated_at": "2026-06-08T...Z",
    "last_used_at": null,
    "revoked_at": null,
    "revoked_by": null,
    "expires_at": "2026-07-08T...Z"
  }
}
```

- 认证时若 `expires_at` 已过期，key 视为无效（返回 `401`），与撤销同等效果。
- 轮换接口 `POST /admin/service-api-keys/{key_id}:rotate` 会复制旧 key 的 scopes / metadata / tenant 归属创建新 key；`revoke_old` 默认 `true`，会立即撤销旧 key。新 raw key 只在轮换响应中返回一次；新 key metadata 包含 `rotated_from`，审计写 `service_api_key_rotated`。

- `GET /admin/service-api-keys` 与 revoke 响应只返回 `key` 对象形态，不返回 `api_key` 明文；create/rotate 响应会在顶层返回一次新 raw `api_key`。
- `POST /admin/service-api-keys` 会校验 `kb_roles` 中每个 KB 是否存在；任一 KB 不存在返回 `404`。
- Service key 认证成功时会更新 `last_used_at`。
- 只存储 `sha256:<hex>` lookup hash 和 `key_preview`，不存储 raw key；raw key 只在创建响应返回一次。
- 支持 `kb_roles` scope，角色名按 `kb_viewer` / `kb_editor` / `kb_admin` / `kb_owner` 规范化并复用同一角色阶梯。
- `tenant_id` 默认仅作为 service key 的归属/审计/配额维度；service key 不会隐式继承 tenant-scoped KB ACL，必须在 `kb_roles` 中显式列出可访问 KB。若创建时设置 `inherit_tenant_kb_acl=true`，则必须同时提供 `tenant_id`，认证时会显式继承该 tenant 的 tenant-scoped KB ACL，并与 `kb_roles` scope 取最高角色。
- 当前最小闭环**不允许 service key 创建 KB 或成为 super admin**；`POST /kbs` 仍要求 super admin 或普通用户 `can_create_kb=true`。
- `can_use_bypass_query=true` 只授予 bypass 查询能力，仍必须同时拥有目标 KB 的 `kb_viewer` 或更高 role。
- `can_use_agent_query=true` 只授予使用 `/agent/query` 的能力；Agent 内每轮仍逐 KB 校验 `kb_viewer` 或更高 role，并受 `candidate_kb_ids` 子集约束。
- 撤销通过 `status=revoked` 持久化；认证每次从 metadata store 查 hash，因此撤销无需等待 token 过期。

企业请求限流/配额行为约束：

- 默认关闭；只有 `LIGHTRAG_ENTERPRISE_RATE_LIMIT_ENABLED=true` 且对应 request 阈值大于 0 时生效。
- 当前项目 LLM 本地部署，不做 token/cost 预算、计费或成本结算；本节配额只表示单机 request quota 与并发 job 限额。
- 计数发生在企业认证成功且 RBAC/bypass 权限校验通过之后；未认证或无权限请求不会消耗额度。
- principal 维度覆盖普通用户、service API key 和显式启用的 legacy enterprise API key；tenant 维度仅在 principal 带 `tenant_id` 时额外计数。
- 超过 request rate limit 返回 `429`、`Retry-After`，并写 `rate_limited` 审计事件；超过 quota 返回 `429`、`Retry-After`，并写 `quota_exceeded` 审计事件。
- 审计 metadata 只记录 `method`、`path`、`auth_method`、`limit`、`window_seconds`、`subject_type`、`retry_after_seconds` 等安全字段；不记录 query/body/raw API key。

### 10.4 企业模式权限边界

- `mode="bypass"` 查询需要 KB read ACL 加 `can_use_bypass_query=true` 或 super admin；按最终解析后的查询模式判断，包括 active query config 默认值。
- `/agent/query` 需要 `can_use_agent_query=true` 或 super admin；Agent 每轮仍只在当前用户 effective `kb_viewer`+ 的候选 KB 中检索。`can_use_bypass_query` 与 `can_use_agent_query` 是两个独立开关，互不蕴含。
- legacy/global `/documents`、`/query`、`/graph`、Ollama `/api/*` 在 `LIGHTRAG_ENTERPRISE_DISABLE_GLOBAL_ROUTES=true` 时默认拒绝；关闭该开关后仍需 super admin。
- super admin bootstrap 来自 `.env`，启动后同步为 active super admin；企业模式要求非默认 `TOKEN_SECRET`。
- Tenant membership、tenant-scoped KB ACL 与 tenant-user override 已接入统一 effective-role resolver；KB 列表、普通 query/retrieve 与 Agent candidate filtering 复用同一结果。Tenant deny 只屏蔽 tenant-derived access，不能覆盖 direct user ACL 或 visibility。Service key 默认只按显式 `kb_roles` scope；设置 `tenant_id + inherit_tenant_kb_acl=true` 时才继承 tenant ACL。
- **KB 可见性（visibility）语义**：`knowledge_bases.visibility ∈ {private, internal, public}`，默认 `private`。`private` 无隐含权限；`internal` 在 KB `tenant_id` 非空时对该租户用户隐含 `kb_viewer`；`public` 对全部已认证企业交互用户（JWT principal）隐含 `kb_viewer`。隐含角色仅为只读。service/scoped API key 与 legacy enterprise API key 不受 visibility 影响。`GET /admin/users/{id}/access` 仅列显式授权，不枚举 visibility。visibility 修改：super admin 可改任意 KB；tenant-created KB（`origin="tenant"`）的 effective `kb_owner`（通常为创建者）可在 `private` ↔ `internal` 间随时切换实现"私有/共享"（`public` 仍为 super admin 专属）；platform KB 仍仅 super admin 可改。**tenant admin oversight**：`tenant_admin` / `tenant_owner` 对本租户 tenant-owned KB 始终隐含只读 `kb_viewer`，成员私有 KB 对租户管理员始终可见。
- 文档删除为所有权感知模型：`kb_editor` 仅可删除本人上传（`metadata.created_by`）的文档；删除他人文档需用户级能力 `can_delete_documents`（super admin 通过 `/admin/users` 授予）或 `kb_admin`+/`super_admin`。service/scoped API key 不会获得 `can_delete_documents`：只能按 `kb_admin` scope 删任意，或按 `kb_editor` scope 删除该 key 自身上传的文档；无 `created_by` 的历史文档仅 privileged 主体可删。
- 文件导出是 KB role 与用户能力的双重门禁：交互式用户访问 artifact `:download` / `:download-url`，以及 original `:preview`，除满足 artifact role policy 外还需 `can_download_files=true`；问答引用文献的下载链接走相同端点，因此不能绕过。派生安全预览不要求该能力。super admin 与 service key 例外，但 service key 仍受显式 KB scope/role。

KB 路由角色矩阵：

| 范围 | 最低角色 / 能力 |
|---|---|
| `POST /kbs` | super admin、canonical `tenant_admin`/`tenant_owner`，或 `can_create_kb=true`；租户用户创建时强制 tenant provenance/tag，visibility 限 `private`（默认）/`internal` |
| `GET /kbs` | super admin 看全部；普通用户看 direct user ACL / tenant ACL 授权 KB + visibility 命中 KB（`public` / 同租户 `internal`）；`tenant_admin`/`tenant_owner` 额外始终看到本租户 `origin=tenant` 的全部 KB（含 private，oversight 只读）；service key 默认仅看 `kb_roles` scope，显式 `inherit_tenant_kb_acl` 时额外继承 tenant ACL，不受 visibility 影响 |
| `GET /kbs/{kb_id}`、`GET /kbs/{kb_id}/status`、`/stats`、`/documents/{id}/chunks`、graph 读取、artifact/doc/job/config/query 读取 | `kb_viewer` 或更高（可由 visibility 隐含，见上）；artifact action policy 可提升最低角色；`:download` / `:download-url` 与 original `:preview` 对交互式用户额外要求 `can_download_files=true` |
| `/kbs/{kb_id}/query`、`/query/stream`、`/query/data`、`/retrieve` | `kb_viewer` 或更高；最终 `mode="bypass"` 额外需要 `can_use_bypass_query=true` |
| `POST /kbs:query`、`/kbs:query/stream`、`/kbs:retrieve`（跨库合并查询） | `kb_ids` 中每个 KB 均需 `kb_viewer`+（handler 自鉴权，中央中间件不覆盖 collection 级路径）；`bypass` 不支持(400) |
| 文档上传/解析/构建/替换/sync、批量启停（`:batch-enable`/`:batch-disable`）、`:rebuild`、job wait/cancel/retry 等写操作 | `kb_editor` 或更高 |
| 文档删除（`DELETE …/documents/{id}`、`:batch-delete`） | `kb_editor` 仅删本人上传(`metadata.created_by`)的文档；删他人需 `can_delete_documents` 能力或 `kb_admin`+/`super_admin` |
| KB 配置创建/激活/diff、`PATCH /kbs/{kb_id}`、图谱编辑（`/graph` 下全部非 GET 端点） | `kb_admin` 或更高；非 super admin 不能改 owner/tenant/provenance；visibility 例外：tenant-created KB 的 effective `kb_owner` 可切换 `private`/`internal`，其余情况仍 super admin 专属 |
| `DELETE /kbs/{kb_id}`、`?hard=true`、`POST /kbs/{kb_id}:restore` | super admin；或该 KB 为 `origin=tenant` 且当前用户是其 canonical tenant 的 `tenant_admin` / `tenant_owner`。direct/tenant ACL 的 `kb_admin`/`kb_owner` 本身不授予生命周期权限 |
| `/admin/...` | super admin；仅精确 `GET /admin/tenants/{tenant_id}` 允许目标 tenant admin/owner |

### 10.5 用户对话管理（/chat）

> 面向新前端的个人问答历史组织能力，层级严格为 **用户个人对话记录 > 项目 > 会话 > 消息**：一个用户可创建多个项目，一个项目下可创建多个会话（用于细分问答），会话内的问答消息**落库持久化**，用户换浏览器/设备登录后可从服务端拉取同一份历史（跨端同步）。项目/会话/消息是纯控制面记录，不触碰 LightRAG 引擎存储，也不改变 KB RBAC；持久化在 KB 控制面 metadata store（`LIGHTRAG_KB_METADATA_BACKEND=postgres` 时为 PostgreSQL 表 `enterprise_chat_projects` / `enterprise_chat_sessions` / `enterprise_chat_messages`，local 模式为 `metadata.sqlite3` 同名表，两后端行为由契约测试保证一致）。
>
> 仅企业模式挂载该路由；仅**交互式 JWT 用户**可用——service/scoped API key 与 legacy enterprise API key principal 一律 `403`。所有读写都按当前用户隔离：访问不存在或属于他人的项目/会话/消息统一返回 `404`（不泄露存在性）。`/chat` 属于企业 anti-bypass 受保护前缀，不能通过 `WHITELIST_PATHS` 放行。

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/chat/projects` | 创建项目；body `{"name": "..."}`（1..256 字符，去首尾空白后非空，空白串 `400`） |
| `GET` | `/chat/projects?limit=50&offset=0` | 当前用户项目列表，按 `updated_at` 倒序；`limit` 1..200，返回 `total` 供分页 |
| `GET` | `/chat/projects/{project_id}` | 项目详情 |
| `PATCH` | `/chat/projects/{project_id}` | 项目改名；body `{"name": "..."}` |
| `DELETE` | `/chat/projects/{project_id}` | 删除项目并**级联删除其全部会话与消息**，响应含 `deleted_sessions` / `deleted_messages` 计数 |
| `POST` | `/chat/projects/{project_id}/sessions` | 在项目下创建会话；body 可整体省略，`name` 缺省/空白时默认以**服务器当前时间命名**（`YYYY-MM-DD HH:MM:SS`）；`context_rounds` 缺省取部署默认值 `CHAT_SESSION_DEFAULT_CONTEXT_ROUNDS` |
| `GET` | `/chat/projects/{project_id}/sessions?limit=50&offset=0` | 项目下会话列表，按 `updated_at` 倒序；项目不存在返回 `404` |
| `GET` | `/chat/projects/{project_id}/sessions/{session_id}` | 会话详情 |
| `PATCH` | `/chat/projects/{project_id}/sessions/{session_id}` | 更新会话；body 为 `name` / `context_rounds` 的任意组合，至少给一个字段（空 body `400`） |
| `DELETE` | `/chat/projects/{project_id}/sessions/{session_id}` | 删除会话并**级联删除其全部消息**，响应含 `deleted_messages` 计数 |
| `POST` | `/chat/projects/{project_id}/sessions/{session_id}/messages` | 批量追加消息（1..20 条，单事务原子写入并分配连续 `seq`）；同一事务内刷新会话 `updated_at`；启用项目记忆 admission 时，同一 PostgreSQL 事务还会分配项目事件序号/参考时间并持久化 durable ingest outbox，提交后只 nudge worker（见下） |
| `GET` | `/chat/projects/{project_id}/sessions/{session_id}/messages?limit=100&offset=0` | 会话消息列表，按 `seq` 升序（`limit` 1..500，返回 `total`）；会话不存在返回 `404` |
| `DELETE` | `/chat/projects/{project_id}/sessions/{session_id}/messages/{message_id}` | 删除单条消息 |
| `POST` | `/chat/projects/{project_id}/memory:search` | **项目记忆检索**（`LIGHTRAG_CHAT_MEMORY_ENABLED=true` 时可用）：在该项目的长期记忆图谱中混合检索历史会话沉淀的事实，见 [10.5.1 项目记忆](#1051-项目记忆chat-memorygraphiti) |
| `GET` | `/chat/projects/{project_id}/memory` | **项目记忆概览**：返回 `{project_id, enabled, available, episode_count, last_ingested_at}`，供前端展示"已沉淀 N 条记忆、上次更新时间"，不触发检索 |

请求/响应示例：

```http
POST /chat/projects
Content-Type: application/json

{"name": "胎侧配方调研"}
```

```json
{
  "id": "proj_1a2b3c4d5e6f",
  "user_id": "usr_...",
  "name": "胎侧配方调研",
  "created_at": "2026-07-10T08:00:00.000000+00:00",
  "updated_at": "2026-07-10T08:00:00.000000+00:00"
}
```

```http
POST /chat/projects/proj_1a2b3c4d5e6f/sessions
Content-Type: application/json

{}
```

```json
{
  "id": "sess_9f8e7d6c5b4a",
  "project_id": "proj_1a2b3c4d5e6f",
  "user_id": "usr_...",
  "name": "2026-07-10 16:00:05",
  "context_rounds": 1,
  "created_at": "2026-07-10T08:00:05.000000+00:00",
  "updated_at": "2026-07-10T08:00:05.000000+00:00"
}
```

```http
PATCH /chat/projects/proj_1a2b3c4d5e6f/sessions/sess_9f8e7d6c5b4a
Content-Type: application/json

{"name": "低温屈挠专题", "context_rounds": -1}
```

追加消息（前端在拿到问答结果后写入一问一答，也可只写单条）：

```http
POST /chat/projects/proj_1a2b3c4d5e6f/sessions/sess_9f8e7d6c5b4a/messages
Content-Type: application/json

{
  "messages": [
    {"role": "user", "content": "低温屈挠性怎么提升？"},
    {
      "role": "assistant",
      "content": "建议 NR/BR 并用… [A1]",
      "metadata": {
        "memory_eligible": true,
        "mode": "mix",
        "kb_ids": ["kb_formula"],
        "references": [{"reference_id": "A1", "kb_id": "kb_formula", "file_path": "formula.md"}]
      }
    }
  ]
}
```

```json
{
  "session_id": "sess_9f8e7d6c5b4a",
  "project_id": "proj_1a2b3c4d5e6f",
  "messages": [
    {"id": "msg_...", "session_id": "sess_9f8e7d6c5b4a", "project_id": "proj_...", "user_id": "usr_...", "role": "user", "content": "低温屈挠性怎么提升？", "metadata": {}, "seq": 1, "created_at": "..."},
    {"id": "msg_...", "role": "assistant", "content": "建议 NR/BR 并用… [A1]", "metadata": {"memory_eligible": true, "mode": "mix", "kb_ids": ["kb_formula"], "references": [...]}, "seq": 2, "created_at": "...", "session_id": "...", "project_id": "...", "user_id": "..."}
  ]
}
```

列表响应形态：`GET /chat/projects` 返回 `{"total", "limit", "offset", "projects": [...]}`；`GET .../sessions` 返回 `{"total", "limit", "offset", "sessions": [...]}`；`GET .../messages` 返回 `{"total", "limit", "offset", "messages": [...]}`。删除响应：项目 `{"id", "deleted": true, "deleted_sessions": N, "deleted_messages": M}`，会话 `{"id", "project_id", "deleted": true, "deleted_messages": M}`，消息 `{"id", "session_id", "project_id", "deleted": true}`。

行为约束：

- 项目、会话均可改名（`PATCH`）；改名会刷新 `updated_at`，因此最近操作的记录排在列表前面；`created_at` 保持不变。**追加消息也会刷新所属会话的 `updated_at`**（同一事务内），会话列表天然按"最近活跃"排序。
- 会话创建默认名为服务器本地时间（如 `2026-07-10 16:00:05`），显式传非空白 `name` 则使用该名称；名称不要求唯一。
- **`context_rounds`（上下文轮次）**：会话级参数，表示每次问答发送给大模型的最近对话轮数——对话轮数超过 `n` 时只发送最近 `n` 轮，`-1` 表示全部发送。创建时缺省取部署默认值 `CHAT_SESSION_DEFAULT_CONTEXT_ROUNDS`（出厂 `1`），之后可随时 `PATCH` 修改；合法取值为 `-1` 或正整数，`0`、`-2` 等返回 `400`。前端从服务端拉取消息后，按该值取最近 `n` 轮组装 `conversation_history` 再调用 query/agent 端点。
- **消息（跨端同步）**：`role` 仅允许 `user` / `assistant`；`content` 非空且单条 ≤ 1 MB；`metadata` 为自由 dict（推荐存放该条回答的 `references`、`mode`、`kb_ids`、agent `session_id` 等，前端换设备后可原样还原引用面板），序列化 ≤ 64 KB，超限 `400`；单次批量 1..20 条，`422` 校验。消息按会话内 `seq`（服务端在写入事务中分配的连续序号）升序返回，分页用 `total/limit/offset`。消息不可编辑，只能追加或删除单条。
- 删除会话在同一事务内级联删除其全部消息；删除项目级联删除其下全部会话与消息；删除用户（`DELETE /admin/users/{user_id}`）级联清理该用户全部项目、会话与消息。
- 问答本身仍走 `/kbs/{kb_id}/query`、`/kbs:query`、`/agent/query` 等既有端点：前端发起提问 → 拿到回答后把一问一答（含引用 metadata）`POST` 到 `.../messages` 落库；换浏览器登录后 `GET .../messages` 即可还原完整历史。
- 审计：`chat_project_created/renamed/deleted`、`chat_session_created/updated/deleted`、`chat_messages_appended`（记条数）、`chat_message_deleted`，metadata 仅记录 id、计数、`has_custom_name`、`context_rounds` 等安全字段，不记录名称与消息正文。

配置（`.env`）：

```bash
# 新建会话的默认上下文轮次；-1 表示每次把全部历史发给大模型
CHAT_SESSION_DEFAULT_CONTEXT_ROUNDS=1
```

#### 10.5.1 项目记忆（Chat Memory，graphiti）

> 当前部署契约：依赖可选 extra `uv sync --extra memory`（`graphiti-core==0.29.2`）与 Neo4j server ≥ 5.26。自动 admission/召回只支持企业认证 + `LIGHTRAG_KB_METADATA_BACKEND=postgres`；若 `LIGHTRAG_CHAT_MEMORY_ENABLED=true` 但未启用企业认证或 metadata backend 不是 PostgreSQL，服务启动配置校验直接失败。

项目记忆以 PostgreSQL chat 消息为**唯一源数据**，Graphiti/Neo4j 是可重建的派生状态。隔离维度是 `(user_id, project_id)`：逻辑组使用稳定 hash ID，实际 Graphiti group 使用带 generation 的物理 ID（形如 `cm_<hash>_gN`）。客户端不得依赖 group ID、generation 或 Graphiti UUID 的长期稳定性。

##### Durable outbox、FIFO 与 generation fence

- **源写入与工作排队原子提交**：启用 admission 时，`POST .../messages` 在一个 PostgreSQL 事务中写入消息、分配同项目单调 `event_seq/project_event_seq` 与 `memory_reference_time`，并插入 `ingest` outbox；请求提交后只唤醒 worker。不存在“消息已提交但 fire-and-forget 任务丢失”的可靠性窗口。
- outbox 事件类型固定为 `ingest`、`rebuild`、`purge`；状态固定为 `pending`、`running`、`retry_wait`、`succeeded`、`superseded`、`dead_letter`。同一用户×项目按 `event_seq` FIFO，`pending/running/retry_wait/dead_letter` 都会阻塞本组后续普通事件，不能越过 dead-letter gap；不同组可并发，默认并发由 `MEMORY_INGEST_CONCURRENCY` 控制。
- worker 用 PostgreSQL claim token + session advisory group lock 取得所有权，不使用可调时间 lease。`LIGHTRAG_CHAT_MEMORY_WORKER_SIDE_EFFECT_TIMEOUT_SECONDS` 同时界定 Graphiti side effect 总 deadline，并作为 stale-running 判定默认阈值；独立 recovery loop 定期验证 owner 已消失后再恢复事件。
- Graphiti side effect **开始前**的确定失败按当前 worker 默认延迟/次数进入 `retry_wait`，耗尽后进入 `dead_letter`。side effect 已可能开始但结果未知时，ingest/rebuild 不在同一物理 generation 盲重试：旧 generation 标记 abandoned，推进到新 generation 并从 SQL 源数据重建；purge 则保持 `purge_pending`，直到某次 definite clear 成功。
- 每个 generation 有持久化 inventory。读取前后都核对 group state、active generation、state version、extraction fingerprint 与 graph-store fingerprint；只有完整重放且 CAS 激活的 generation 可读。重建超过 `LIGHTRAG_CHAT_MEMORY_REBUILD_MAX_MESSAGES` 或 `...REBUILD_MAX_BYTES` 时 fail closed，绝不激活部分快照。
- `LIGHTRAG_CHAT_MEMORY_ENABLED=false` 关闭**新消息 admission 和自动召回**；该期间追加的消息没有 memory event sequence，不会在以后被静默补录。`LIGHTRAG_CHAT_MEMORY_MAINTENANCE_ENABLED=true` 可让已有组的 rebuild/purge 继续排队和执行，读写开关与维护可靠性相互独立。

##### Admission 与内容策略

- 非空 `user` 消息默认可进入 episode；`assistant` 消息只有在该消息 `metadata.memory_eligible` 为 JSON boolean `true` 时才可进入，避免模型回答自动自我强化。其他 role、空白内容不进入 Graphiti；一个 append batch 仍保留为一个 replay 边界。
- 每条已 admission 消息送入抽取前最多使用 `MEMORY_INGEST_MAX_CHARS` 个字符，超出部分带截断标记；该策略和抽取/embedding/LLM 设置参与 extraction fingerprint，变更会触发新 generation。
- `LIGHTRAG_CHAT_MEMORY_STORE_RAW_EPISODE_CONTENT=false` 是企业隐私默认值：SQL 消息保留为源数据，Neo4j 不额外保存 Graphiti raw episode content。显式改为 `true` 会增加原文副本与隐私面，并改变 extraction fingerprint。
- `MEMORY_INGEST_MODE`、`MEMORY_INGEST_DEBOUNCE_SECONDS`、`MEMORY_BACKLOG_SCAN_ON_START`、`MEMORY_BACKLOG_BATCH_MESSAGES`、`MEMORY_MAX_INFLIGHT_PER_USER` 仍为旧调用方兼容/聚合参数；当前 enterprise server 已禁用 legacy `schedule_*` fire-and-forget 路径，它们不是 durable reliability 机制，也不控制下述管理员 backlog recovery。

##### 删除、重建与 graph-store 绑定

- 删除单条消息或会话：源删除与 generation 推进/`rebuild` outbox 在同一事务提交；worker 从仍存活且曾 admission 的 SQL append batches 完整重放，不调用非可逆的单 episode rollback。
- 删除项目或用户：源数据删除与 durable `purge` 事件原子提交；outbox/generation 记录不随源行级联删除。purge 清理 active、building、retired、abandoned、purge-pending generation 以及旧版 `{user_id}--{project_id}` group，全部 definite clear 后才成功，因此源用户/项目删除后仍可按 event ID 恢复。
- 一个逻辑组绑定一个 graph-store fingerprint。更换 `MEMORY_NEO4J_DEPLOYMENT_ID`、Neo4j endpoint/database 后继续对旧组 append/rebuild/delete/purge 会显式返回 `409 graph_store_migration_required`，不会把同一组分裂写入不同图。管理员重试旧 purge 时若 runtime graph store 不匹配，返回 `409 chat_memory_old_graph_store_required`。

##### 两种读取方式

1. **最终合成自动注入**：在单 KB、多 KB、Agent plan/staged 的 query/query-stream 请求体加 `memory: {"project_id":"...","limit":10}`。服务端授权早、搜索晚，仅在当前 KB 证据完成后把 trusted policy 与 untrusted JSONL 数据加入最终合成，不影响规划/检索，完整状态、引用、错误和 egress 契约见 [8.3](#83-项目记忆自动注入仅终答合成)。
2. **独立检索端点**：`POST /chat/projects/{project_id}/memory:search` 直接返回当前 generation 的有效事实，供管理/展示或由受信任客户端自行处理；它不是自动注入的前置步骤：

```http
POST /chat/projects/{project_id}/memory:search
Content-Type: application/json

{"query": "之前对低温性能有什么结论？", "limit": 10}
```

```json
{
  "project_id": "proj_1a2b3c4d5e6f",
  "total": 1,
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

- 独立检索行为：
  - `query` 为 `1..4096` 字符；`limit` 为 `1..50`，缺省取 `MEMORY_SEARCH_LIMIT`。默认使用 Graphiti hybrid/RRF；`MEMORY_RERANK_ENABLED=true` 且部署 reranker 可用时走 cross-encoder recipe。
  - 只搜索 `invalid_at IS NULL` 且 `expired_at IS NULL` 的当前事实，并在搜索前后执行 active-generation 双重 read fence；fence 改变时丢弃结果或 fail closed 为临时不可用。
  - 仅交互式 JWT 用户可用；他人/不存在项目统一 `404`。功能未启用返回 `503 Chat memory is not enabled`；Graphiti/Neo4j/依赖或 read fence 不可用返回 `503 Chat memory is temporarily unavailable`。这与自动注入的 typed availability **fail-open metadata** 不同。
  - 响应中的 `facts[].uuid` 也是 generation-scoped Graphiti fact ID；重建后可能改变。`invalid_at/expired_at` 字段保留在 schema 中，但正常当前事实搜索结果应为空值。
  - `GET /chat/projects/{project_id}/memory` 返回 SQL mapping 统计 `{project_id,enabled,available,episode_count,last_ingested_at}`，不搜索 Neo4j，也不代表 durable outbox 已清空。

##### 运维与恢复

- `/health.chat_memory` 给出 read/ingest 开关、backend availability、worker 是否运行和缩短 fingerprint；`pending_tasks` 仅为 legacy 进程内任务计数，不是 outbox 深度。
- super-admin `POST /admin/chat-memory:backlog-scan {"limit":100}` 执行 stale `running` claim recovery、唤醒 worker，并返回 durable outbox 的 `pending/running/retry_wait/dead_letter/oldest_available_at/oldest_lag_seconds`。它不扫描会话 seq 水位，也不直接重摄取消息。
- `POST /admin/users/{user_id}/chat-memory:purge` 只**持久化排队** purge，返回 `{queued,noop,project_ids}`；新建事件或已存在 `pending/running/retry_wait` purge 都按幂等已排队计入 `queued`，metadata group 已处于终态 `deleted` 时计入 `noop`。
- `POST /admin/chat-memory/events/{event_id}:retry` 只恢复已有 purge outbox 行，不从请求重建 target，因此可安全用于源删除后的 dead letter。成功返回 `{event_id,status,user_id,project_id,event_type}`；不存在 `404`。非 purge 为 `409 chat_memory_retry_purge_only`，`running/succeeded/superseded` 等终态为 `409 chat_memory_event_not_retryable`，并发状态变化为 `409 chat_memory_event_retry_conflict`，错误 graph store 为 `409 chat_memory_old_graph_store_required`。

当前主要配置（未设置的 provider `MEMORY_*` 字段逐项继承 QUERY LLM / deployment embedding / Neo4j）：

```bash
LIGHTRAG_CHAT_MEMORY_ENABLED=true
# read/ingest 关闭后仍可 drain 已存在的 rebuild/purge
LIGHTRAG_CHAT_MEMORY_MAINTENANCE_ENABLED=true

# durable worker / stale recovery；worker 使用 advisory ownership，不是时间 lease
LIGHTRAG_CHAT_MEMORY_WORKER_POLL_SECONDS=1.0
LIGHTRAG_CHAT_MEMORY_WORKER_RECOVERY_INTERVAL_SECONDS=30.0
LIGHTRAG_CHAT_MEMORY_WORKER_SIDE_EFFECT_TIMEOUT_SECONDS=900.0
LIGHTRAG_CHAT_MEMORY_WORKER_SHUTDOWN_TIMEOUT_SECONDS=10.0
LIGHTRAG_CHAT_MEMORY_REBUILD_MAX_MESSAGES=10000
LIGHTRAG_CHAT_MEMORY_REBUILD_MAX_BYTES=67108864

# privacy/admission/final synthesis
LIGHTRAG_CHAT_MEMORY_STORE_RAW_EPISODE_CONTENT=false
LIGHTRAG_CHAT_MEMORY_PROMPT_MAX_TOKENS=1024
LIGHTRAG_CHAT_MEMORY_PROMPT_MAX_CHARS=8192
LIGHTRAG_CHAT_MEMORY_ALLOW_CROSS_PROVIDER_QUERY_EGRESS=false

MEMORY_LLM_MODEL=qwen3.6-36b                 # 记忆抽取 LLM（OpenAI-compatible）
MEMORY_LLM_TEMPERATURE=0.0
MEMORY_STRUCTURED_OUTPUT_MODE=json_schema    # vLLM 约束解码；不支持时改 json_object
MEMORY_OPENAI_LLM_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'
# MEMORY_EMBEDDING_DIM=4096                  # 必须与 embedding 服务实际维度一致
# MEMORY_NEO4J_DATABASE=neo4j                # 需要物理隔离时指向独立 database（企业版）
# MEMORY_NEO4J_DEPLOYMENT_ID=memory-prod-a   # graph-store 稳定身份；迁移时不可随意改变
MEMORY_SEARCH_LIMIT=10
MEMORY_INGEST_CONCURRENCY=2                  # durable worker 跨项目并发上限
MEMORY_MAX_COROUTINES=4                      # graphiti 单次摄取内部并发
MEMORY_INGEST_MAX_CHARS=6000                 # 单条消息参与摄取的最大字符数
MEMORY_RERANK_ENABLED=false                  # true 时用部署 reranker 精排（cross-encoder 配方）
GRAPHITI_TELEMETRY_ENABLED=false             # 内网部署关闭 graphiti 匿名遥测
```

除主开关 `LIGHTRAG_CHAT_MEMORY_ENABLED` 外，本节 namespaced maintenance/worker/rebuild/raw-content/prompt/egress setting 同时接受相应 `MEMORY_*` alias；新部署应使用上面的 `LIGHTRAG_CHAT_MEMORY_*` canonical key。旧 immediate/debounced/watermark backlog 参数仍被 parser 接受，但当前 enterprise server 不用它们承诺可靠性。

---

## 十一、状态机与字段说明

### 11.1 文档状态

```
created
  -> uploaded
  -> parse_queued -> parsing -> parsed
                              |
                              -> parse_failed
  parsed
  -> build_queued -> building -> ready
                              |
                              -> build_failed
  ready / build_failed
  -> build_queued (重新构建)
  uploaded / parsed / ready / parse_failed / build_failed / replace_failed
  -> replacing -> uploaded
              |
              -> replace_failed
```

辅助状态：`disabled` / `archived` / `deleting` / `delete_failed` / `deleted` / `replacing` / `replace_failed`。

### 11.2 任务状态机（已实现部分）

```
queued ---> running ---> succeeded
   |          |           
   |          +--> cancelling --> cancelled
   |          |
   |          +--> failed
   +-----> cancelled
   +-----> failed
failed   --> retrying --> queued
cancelled --> retrying --> queued
```

允许的转换由 `_allowed_next_job_statuses` 限定；非法转换返回 `409 InvalidJobTransition`。

### 11.3 三段 Hash 含义

| Hash | 派生因子 | 变化时的最小动作 |
|---|---|---|
| `source_hash` | 上传 / 文本 / URL / staged import / scan 内容字节 | 重新解析 + 重新构建 |
| `parser_hash` | 解析引擎 + process options | 重新解析 + 重新构建 |
| `index_hash` | 当前 active runtime 已实际接入的 chunk / embedding / extraction 配置（如 chunk size/overlap、tokenizer、embedding model/dim/token limit、extraction language/entity_types/抽取 caps） | 仅重新构建索引（复用解析产物） |

`:build-kg` 命中 `index_hash` 且文档已 `ready` 时直接 skip；`:reindex` 始终绕过 skip。

### 11.4 幂等键约定

- 幂等键唯一索引：`(kb_id, job_type, idempotency_key)`。
- 文本导入、URL 导入、本地 staged 文件导入、目录扫描、批量增量同步、单文档 parse、批量 parse、单文档 build、批量 build、单文档 replace 都支持幂等键。
- 同 key 同请求指纹返回原 job；同 key 不同请求指纹返回 `409`。

### 11.5 错误码归纳

| HTTP | 业务错误码 | 含义 |
|---|---|---|
| 400 | invalid_parse_request / parser_engine_unsupported | 参数不合法 |
| 404 | KnowledgeBaseNotFoundError / MetadataRecordNotFoundError | KB / 文档 / 任务 / 产物未找到 |
| 409 | parse_job_active | 文档已有运行中的解析任务 |
| 409 | build_job_active | 文档已有运行中的构建任务 |
| 409 | delete_job_active | 文档已有运行中的删除任务 |
| 409 | replace_job_active | 文档已有运行中的替换任务 |
| 409 | document_not_parsed | 文档尚未完成解析，无法触发构建 |
| 409 | IdempotencyKeyConflict | 同幂等键不同请求指纹 |
| 409 | InvalidJobTransitionError | 任务状态不允许该转换 |
| 400 | - | `delete_graph_orphans=false` 暂不支持 |
| 413 | - | 上传体积超出 `MAX_UPLOAD_SIZE` 或文本超限 |
| 503 | - | 注册表 / 构建服务未配置（含 `strategy=rebuild_kb` / `strategy=rebuild_subgraph` 缺少 IndexBuildService） |

---

## 十二、生产存储配置

KB 控制面 metadata 与 LightRAG engine storage 是两套配置：

- `LIGHTRAG_KV_STORAGE` / `LIGHTRAG_VECTOR_STORAGE` / `LIGHTRAG_GRAPH_STORAGE` / `LIGHTRAG_DOC_STATUS_STORAGE` 控制底层 RAG 数据（full docs、chunks、vectors、graph、doc status）。
- `LIGHTRAG_KB_METADATA_BACKEND` 控制 KB catalog、documents、jobs、artifacts、config versions 等业务控制面 metadata。

### 12.1 PostgreSQL 控制面 metadata

```env
LIGHTRAG_KB_METADATA_BACKEND=postgres
LIGHTRAG_KB_POSTGRES_HOST=192.168.1.66
LIGHTRAG_KB_POSTGRES_PORT=5433
LIGHTRAG_KB_POSTGRES_USER=admin
LIGHTRAG_KB_POSTGRES_PASSWORD=123456
LIGHTRAG_KB_POSTGRES_DATABASE=knowledge_base
# 可选：LIGHTRAG_KB_POSTGRES_DSN=postgresql://admin:123456@192.168.1.66:5433/knowledge_base
LIGHTRAG_KB_POSTGRES_POOL_MIN_SIZE=1
LIGHTRAG_KB_POSTGRES_POOL_MAX_SIZE=10
```

启用后服务启动时会创建/迁移所需表：`kb_catalog_schema`、`kb_catalog`、`kb_metadata_schema`、`kb_documents`、`kb_jobs`、`kb_document_artifacts`、`kb_config_versions`。默认不设置或设置为 `local/json/sqlite` 时仍使用 `WORKING_DIR/metadata/knowledge_bases.json` + `metadata.sqlite3`。

从本地 JSON/SQLite 控制面迁移到 PostgreSQL 时，先在旧 `WORKING_DIR` 上运行迁移工具做 dry-run，再执行真实导入：

```bash
lightrag-migrate-kb-metadata --working-dir ./rag_storage --dry-run
lightrag-migrate-kb-metadata --working-dir ./rag_storage --strategy fail --yes
```

可用参数包括 `--postgres-dsn`（覆盖环境变量）、`--kb-id`（只迁移指定 KB，可重复）、`--strategy fail|skip|overwrite`、`--json`。该工具只迁移 KB catalog、documents、jobs、artifacts、config_versions 与 `source_key` projection；不会复制源文件、解析产物、向量库、图存储或 text chunks，这些仍需依赖既有 `INPUT_DIR` / 对象存储 / LightRAG engine storage 运维流程。

> 测试覆盖：`PostgresMetadataStore` 与 `SQLiteMetadataStore` 由 `tests/api/test_metadata_store_contract.py` 用同一组用例参数化校验行为等价。SQLite 参数始终运行；设置 `LIGHTRAG_KB_POSTGRES_TEST_DSN`（或 `POSTGRES_TEST_DSN`）后会对**真实 PostgreSQL** 执行同一契约。**注意：KB 维度记录用唯一 `kb_id` + 结束 `purge`，但企业用户/租户/成员/审计记录用固定标识、不随之清理**——务必把 DSN 指向**一次性测试库**（勿用生产 `knowledge_base`，重复跑会残留并撞唯一用户名）。例：
>
> ```bash
> # 在同台 PG 上建一次性库再跑，跑完可 DROP
> LIGHTRAG_KB_POSTGRES_TEST_DSN=postgresql://admin:123456@<host>:5433/lightrag_contract_test \
>     uv run pytest tests/api/test_metadata_store_contract.py -q
> ```
>
> 注：`source_name` 文档过滤在 Postgres 后端的 `ESCAPE` 子句此前误用两字符转义串（`InvalidEscapeSequenceError`），已修复为单字符并由该 live 契约测试守护。
>
> live 验证记录（2026-06-10，PG 15.17）：在 `192.168.1.66:5433` 上新建一次性库 `lightrag_contract_test` 运行 `tests/api/test_metadata_store_contract.py` **38 passed**，覆盖 KB/文档/任务/配置版本/企业用户·租户实体·成员·KB ACL·审计 全套（含新增的租户实体 CRUD/删除与审计过滤分页契约），用后即 DROP，未污染生产 `knowledge_base`——确认 Postgres 路径与 SQLite 行为一致。

### 12.2 MinIO / S3 source 与 artifact 存储

```env
LIGHTRAG_OBJECT_STORAGE=minio
LIGHTRAG_OBJECT_STORAGE_ENDPOINT=http://192.168.1.66:19000
LIGHTRAG_OBJECT_STORAGE_BUCKET=lightrag-kb
LIGHTRAG_OBJECT_STORAGE_ACCESS_KEY_ID=admin
LIGHTRAG_OBJECT_STORAGE_SECRET_ACCESS_KEY=admin123
LIGHTRAG_OBJECT_STORAGE_USE_SSL=false
LIGHTRAG_OBJECT_STORAGE_REGION=us-east-1
LIGHTRAG_OBJECT_STORAGE_PREFIX=kb
LIGHTRAG_OBJECT_STORAGE_CREATE_BUCKET=true
LIGHTRAG_OBJECT_STORAGE_DISABLE_EXPECT_HEADER=true
```

依赖要求：`aioboto3>=12,<16` 是 MinIO/S3 对象存储后端的运行时依赖。未设置 `LIGHTRAG_OBJECT_STORAGE` 或设置为 `local` 时不会创建 S3 client，也不需要该依赖；设置为 `minio` 或 `s3` 时需使用包含该依赖的 API 安装（源码环境执行 `uv sync --extra api`，包安装使用 `pip install "lightrag-hku[api]"`），老环境升级后也需要重新同步依赖。缺失时服务会在对象存储初始化阶段给出明确错误；若只是本地开发且不需要对象存储，可改回 `LIGHTRAG_OBJECT_STORAGE=local`。

`LIGHTRAG_OBJECT_STORAGE_DISABLE_EXPECT_HEADER` 默认为 `true`：上传 `PutObject` / multipart `UploadPart` 前会移除 botocore 自动添加的 `Expect: 100-continue` 请求头，以兼容部分 MinIO 或反向代理组合在 100-continue 握手上挂起并最终返回 `RequestTimeout` 的场景。若生产 S3 网关明确依赖该头，可显式设置为 `false`。

对象 key 组织在 `<prefix>/workspaces/<workspace>/documents/<document_id>/...` 下。`INPUT_DIR` 仍是本地 cache：parser、build、download 继续使用本地 path；当 cache 缺失时，download/parse planning 会按 metadata 中的对象 URI restore。文件型 artifact 可通过 `:download-url` 直接获取对象存储 `GET Object` 预签名 URL，绕过 API 代理传输大文件；目录型 artifact 仍通过 API zip 代理。硬删除 KB 时会按 workspace prefix 清理对象。

> 测试覆盖：`tests/api/test_object_storage_s3.py` 仅在 boto3 client 边界打桩（`aioboto3` 在 `S3ObjectStorage._new_session` 内惰性 import，可注入 fake session 离线运行），直测出厂 `S3ObjectStorage` 的 key 前缀规范化、URI 构建/解析、bucket 自动创建、upload/download 往返、目录逐文件上传、`list_objects_v2` 分页与续传 token、`delete_uri`/`delete_prefix`/`delete_workspace`、`GET Object` 预签名 URL、`Expect` 请求头兼容处理与 backend 选择。`tests/api/routes/test_kb_document_routes.py` 覆盖 artifact `:preview` 的 inline restore/size/type/directory guard，以及 `:download-url` 对文件型 object artifact 返回 URL、目录型 artifact 拒绝预签名并继续走 zip 下载。该测试路径不需要连接真实 MinIO/S3；生产启用 `LIGHTRAG_OBJECT_STORAGE=minio|s3` 仍需要 `aioboto3`。
