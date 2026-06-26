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
| `POST` | `/kbs/{kb_id}:restore` | 恢复软删除的知识库（`deleted`→`active`）；企业模式仅 super admin |
| `GET` | `/kbs/{kb_id}/stats` | 控制面统计：文档状态分布、chunks/entity/relation 合计、job 状态分布、dead-letter、artifact 数 |

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
  "visibility": "private",         // 枚举：private / internal / public；企业模式下 internal=同租户隐含只读、public=全员隐含只读（语义见 10.4），写权限仍以 KB ACL 为准
  "metadata": {"tags": ["legal"]}  // 可选自由 dict（前端标签/分组/扩展字段），序列化 ≤16KB；响应与列表原样返回
}
```

返回 `200 KnowledgeBaseResponse`；冲突 `409`；参数非法 `400`。

### 1.2 列出 / 获取 / 更新 / 删除

- `GET /kbs?include_deleted=false`：默认排除软删除记录。
- `GET /kbs/{kb_id}`：404 表示未找到或已软删除。
- `PATCH /kbs/{kb_id}`：仅更新请求体显式给出的字段；`status` 不允许直接置为 `deleted`；`active_config_version_id` 不能通过 PATCH 修改，若请求体包含该字段返回 `400`，请改用 `POST /kbs/{kb_id}/configs/{version_id}:activate`。`metadata` 为**合并**语义：给出的 key 覆盖现值、value=null 删除该 key、未提及的 key 保留；顶层 `metadata: null` 返回 `400`；合并后序列化超 16KB 返回 `400`。
- `DELETE /kbs/{kb_id}`：默认软删除，同步从 `LightRAGInstanceRegistry` 卸载实例。
- `DELETE /kbs/{kb_id}?hard=true`：触发硬删除。若服务端启用 durable worker 且 `clear_kb` 在 `job_worker.resumable_job_types` 中，路由会先 soft-delete/tombstone KB，再调用 `KBDeletionService.enqueue_hard_delete()` 创建 queued `clear_kb` job，并返回 `KnowledgeBaseDeleteResponse` 中的 `hard_delete_queued=true`、`hard_delete_job_id`、`hard_delete_job_type="clear_kb"`、`hard_delete_job_status="queued"`；后续由 worker 通过 `resume_hard_delete` 幂等执行，且 job 查询/取消/重试端点对 soft-deleted KB 使用 `include_deleted=true`，因此硬删除 job 在 KB tombstone 后仍可观察和控制。若 durable worker 未启用，保持兼容的同步硬删除流程：`KBDeletionService` 在 destructive lock 下依次执行：
  1. `force_evict` 在内存中的 LightRAG 实例并调用 `finalize_storages`（关闭存储句柄，不删数据）；
  2. **drop 全部引擎 storage 数据**：用 registry builder 建一个未缓存的瞬时实例并调用 `LightRAG.adrop_all_storages()`，对 full_docs / text_chunks / entities / relations / chunks / vector / graph / doc_status / llm_cache 等全部 storage 调 `drop()`。下一步删 `working_dir` 只能清文件型后端，外部后端（PostgreSQL / Milvus / Neo4j / Qdrant / Redis / Mongo / OpenSearch）数据在远端服务里，必须经此步显式清除，否则会残留并被复用同 workspace 的新 KB 读到；
  3. 删除 `working_dir/<workspace>`（如已配置）；
  4. 删除 `input_dir/<workspace>`（上传文件 + 解析 artifact 的本地 cache）；
  5. 若启用对象存储，删除该 workspace 下的 source/artifact 对象；
  6. 清空 metadata store 控制面（documents / jobs / artifacts / config_versions；local 模式为 SQLite，PostgreSQL 模式为对应表）。
  同步分支返回前会创建一条 `clear_kb` 类型的 job 记录最终结果，`result` 包含 `dropped_storages`（成功 drop 的 storage 数）、`cleared_object_storage` 和 `deleted_objects`；任一步失败（含某个 storage `drop()` 失败）HTTP 500 + `clear_kb` job 终态 `failed`，使操作者知道可能有残留并 `:retry`。失败的 `clear_kb` job（`max_retries=3`）可经 `:retry` 重置回 `queued`。

企业模式（`LIGHTRAG_ENTERPRISE_AUTH_ENABLED=true`）下：

- `POST /kbs` 需要 super admin 或 `can_create_kb=true`；服务端忽略请求体中的 `owner_id`/`tenant_id`，改用当前 principal，并自动授予创建者 `kb_owner` ACL。
- `GET /kbs` 对普通用户返回已授权 KB（direct user ACL / tenant ACL）以及 visibility 命中的 KB（`public` / 同租户 `internal`，见 10.4）；super admin 返回全部；service key 仅按 `kb_roles` scope（可选显式 `inherit_tenant_kb_acl`），不受 visibility 影响。
- `PATCH /kbs/{kb_id}` 忽略请求体中的 `owner_id`/`tenant_id`，避免客户端伪造所有权或租户。
- `DELETE /kbs/{kb_id}` 与 `?hard=true` 仅 super admin 可执行，并写入审计事件。

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
- 企业模式仅 super admin 可调用（与 `DELETE /kbs/{kb_id}` 同级），写入 `kb_restored` 审计事件。

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
  "artifacts": {"total": 96}
}
```

- **仅查控制面 metadata store**，不加载 LightRAG 实例，调用廉价且无副作用；图谱规模（节点/边数）继续使用 `GET /kbs/{kb_id}/graph/status`。
- `counters` 为各文档构建回填计数的合计；已删除文档的计数在删除时已清零，不计入。
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
- 解析缓存命中时直接复用 artifacts：缓存有效性由 MinerU/Docling 的 `*.mineru_raw` raw bundle manifest 校验（源文件大小 + 内容 sha256 + options 签名），而非 KB 控制面的 `source_hash`/`parser_hash`（后者用于增量决策与 diff，不作为 raw bundle cache key）。`force_reparse=true` 绕过该 raw bundle cache。
- 同一文档已有 `parse_queued` / `parsing` / `build_queued` / `building` / `deleting` / `replacing` 时返回 `409`，原 active job 保持不变，新建的 job 同步标记 `failed`。
- 成功后写入 `original` / `sidecar` / `blocks` artifact，MinerU/Docling 还会写 `raw_dir`，并从 raw bundle 中记录细粒度文件 artifact：`markdown`、`content_list`、`middle_json`、`model_json`、`image`、`layout_pdf`。细粒度 artifact metadata 包含 `parse_engine`、`parser_hash`、`source`、`relative_path`。启用对象存储时，文件 artifact 额外写入 `metadata.object_uri`，目录 artifact 写入 `metadata.object_prefix_uri`；`original` artifact 复用 document 的 `metadata.source_object_uri`。
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
| `GET` | `/kbs/{kb_id}/graph/relations` | 关系（edge）分页列表，返回 `id/type/source/target/properties` |
| `GET` | `/kbs/{kb_id}/graph` | 指定 `label` 的连通子图（`*` 表示整图），支持 `max_depth` / `max_nodes` |

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

> 通过环境变量 `LIGHTRAG_KB_JOB_WORKER=true` 启用。默认关闭，关闭时行为与历史一致（仅 in-process 背景任务，重启后遗留任务一律标 `failed`）。

启用后：
- 服务启动会拉起一个后台轮询 worker，原子认领（`queued → running` 单赢 CAS）以下可从持久化状态重建的任务类型并执行到终态：单文档 `parse` / `build_kg` / `reindex` / `delete` / `replace`，**聚合** `parse` / `build_kg` / `reindex`（`document_id=null`、payload 携带 `document_ids`，含多文件 `upload` / `texts` 的 auto_parse 聚合 job 与 `batch-parse` / `batch-build-kg` / `batch-reindex` / `:rebuild`），聚合 `sync`（payload 携带 `batch_id` 与 per-item `source_key/source_name/source_hash`，请求字节已落盘到 `.sync-staging/<batch_id>/`），`documents:batch-delete` 聚合 `delete` job，以及 `clear_kb`（KB 硬删除，payload 携带 `kb_id`/`workspace`，幂等清理可重启续跑）。聚合 parse/build 之所以可恢复，是因为其源文件 / 解析产物在 job 运行前已落盘，worker 可凭 `document_ids` 重新规划并逐个 claim 执行；聚合 sync 则凭 staged request bytes 重建 `DocumentSourceInput` 并复用同一 per-item 同步逻辑。
- **单文档 `replace` 现已可恢复**：replace 创建并 claim 时会把替换源字节落盘到 `INPUT_DIR/<workspace>/<document_id>/.replace-staging-<job_id>.bin`，因此 worker 可在重启/`:retry` 后凭 staged 字节重建 `DocumentReplacementSource`，重新 claim 文档进入 `replacing`，复用与同步路径一致的执行逻辑（删旧索引 → 换 source → 可选 auto_parse/auto_index），终态后清理 staging 文件。若 staged 字节缺失（历史 job 未落盘），或 staged 字节内容 hash 与 payload `source_hash` 不匹配（staging 文件被截断/损坏），worker 以 `replace_not_resumable` 明确失败，不会凭错字节续跑（与批量 `sync` 续跑的 hash 校验对齐）。
- **批量 `sync` 现已可恢复**：sync route 在创建聚合 job 前为每个 item 落盘请求字节，并在 payload 中持久化 `batch_id`、`source_key`、`source_name`、`source_hash`、`content_type` 与同步选项；worker 重启/`:retry` 后按 staged bytes 重建每个 source，重新执行 created/replaced/skipped 与可选 parse/build。staging 只在终态 job transition 成功后 best-effort 清理；若 staged bytes 缺失或 hash 不匹配，worker 以 `sync_not_resumable` 明确失败。
- **自动消费 `:retry`**：重试把任务重置回 `queued` 后，worker 在下一轮轮询中认领并重跑，客户端无需再次发起业务请求。
- **重启续跑**：进程重启时，孤儿恢复会保留这些可恢复类型的 `queued` 任务（不再标 `failed`）交给 worker 继续执行；仍处于 `running` 的中途任务无法安全恢复，照旧标 `failed`，其文档同步重置为 `*_failed`，客户端 `:retry` 后即可被 worker 自动重跑。delete 续跑时若孤儿恢复已把文档从 `deleting` 重置为 `delete_failed`，worker 会重新 claim 回 `deleting` 再执行（`_claim_document_deleting` 接受同一 delete job id 的幂等 reclaim）；replace 续跑同理从 `replace_failed` 重新 claim 回 `replacing`。
- **不抢占新任务**：worker 只认领 `queued_at` 早于宽限窗口（`LIGHTRAG_KB_JOB_WORKER_GRACE_SECONDS`，默认 5s）的任务；新建任务由其 in-process 背景任务在毫秒级转入 `running`，因此不会被 worker 抢跑，避免重复执行。
- **需重新发起的类型**：多文件 `upload` 且 `auto_parse=false` 时不产生可重驱动解析工作；其他没有持久化请求上下文的历史/自定义任务仍会在孤儿恢复时标 `failed`，需要重新发起请求。
- **死信**：`failed` 且 `retry_count >= max_retries` 的任务不会再被 `:retry` 或 worker 重跑，可通过 `GET /kbs/{kb_id}/jobs/dead-letter` 单独列出做人工triage。
- 可调环境变量：`LIGHTRAG_KB_JOB_WORKER_POLL_SECONDS`（默认 1.0s）、`LIGHTRAG_KB_JOB_WORKER_GRACE_SECONDS`（默认 5.0s）。

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

> 产物记录解析阶段产生的文件 / 目录。当前支持 `original` / `sidecar` / `blocks` / `raw_dir`，以及 MinerU/Docling raw bundle 中的 `markdown` / `content_list` / `middle_json` / `model_json` / `image` / `layout_pdf` 等细粒度类型。`uri` 是本地 cache path；启用对象存储后，metadata 会额外包含 `object_uri` 或 `object_prefix_uri`。

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/kbs/{kb_id}/documents/{document_id}/artifacts` | 产物列表 |
| `GET` | `/kbs/{kb_id}/documents/{document_id}/artifacts/{artifact_id}` | 产物元数据 |
| `GET` | `/kbs/{kb_id}/documents/{document_id}/artifacts/{artifact_id}:download` | 下载文件型产物；目录型产物以 zip 代理下载 |
| `GET` | `/kbs/{kb_id}/documents/{document_id}/artifacts/{artifact_id}:preview` | 内联预览受支持的小型文件型产物 |
| `GET` | `/kbs/{kb_id}/documents/{document_id}/artifacts/{artifact_id}:download-url` | 为对象存储中的文件型产物生成预签名下载 URL |

列表示例：`GET /kbs/{kb_id}/documents/{document_id}/artifacts?artifact_type=markdown&limit=50&offset=0`。

下载约束：
- 企业模式权限：artifact list/detail 按 `kb_viewer` 或更高角色读取。`:download` 与 `:download-url` 的默认最低角色由 `LIGHTRAG_ENTERPRISE_ARTIFACT_DOWNLOAD_MIN_ROLE` 控制，默认 `kb_viewer` 保持旧行为，可提升为 `kb_editor`、`kb_admin` 或 `kb_owner`；`LIGHTRAG_ENTERPRISE_ARTIFACT_DOWNLOAD_POLICY` 可用 JSON object 按 artifact type 覆盖（如 `{"original":"kb_editor","*":"kb_viewer"}`），并同时作用于显式匹配类型的 `:preview`。更细粒度时可使用 `LIGHTRAG_ENTERPRISE_ARTIFACT_ACTION_POLICY`，按 action 分别设置 artifact type policy，例如 `{"preview":{"*":"kb_editor"},"download":{"original":"kb_editor"},"download-url":{"original":"kb_admin"}}`。action policy 优先于 download policy；低于要求的角色返回 `403`。
- 文件型产物（`original` / `blocks` / `markdown` / `content_list` / `middle_json` / `model_json` / `image` / `layout_pdf`）以 `FileResponse` 直接返回。
- 目录型产物（`sidecar` / `raw_dir`）以流式 zip 返回（`Content-Type: application/zip`），单次下载 zip 内未压缩字节上限 512 MB，超限返回 `413`。
- 路径必须位于 `inputs/<workspace>/<document_id>` 内；跨 KB、缺失文件、路径逃逸均返回 `404` / `400`。
- 启用对象存储时，如果本地 cache path 缺失，`:download` 接口会先从 `metadata.object_uri` / `metadata.object_prefix_uri` restore 到原 cache path，再返回文件或 zip，保持旧客户端兼容。
- `:preview` 仅支持文件型 artifact，目录返回 `400`；支持 `text/*`、`application/json`、`application/ld+json`、`application/markdown`、`application/x-ndjson`、普通图片（不含 `image/svg+xml`）和 `application/pdf`，以 `inline` content-disposition 返回；单次 preview 上限 10 MB，超出返回 `413`，不支持的 media type 返回 `415`。本地 cache 缺失时同样按对象存储 metadata restore。
- `:download-url` 仅对 metadata 中存在 `object_uri` 的**文件型** artifact 生效，返回 `{artifact_id,url,object_uri,expires_in_seconds,filename,media_type}`；服务端使用对象存储后端生成 `GET Object` 预签名 URL，不会触发本地 cache restore。`expires_in_seconds` 默认 3600 秒，服务端限制在 `[1, 604800]`。目录型 artifact（`sidecar` / `raw_dir`，metadata 中为 `object_prefix_uri`）仍需走 `:download` 的 zip 代理下载。企业模式且 `LIGHTRAG_ENTERPRISE_MASK_STORAGE_URIS=true`（默认）时，文档 `source_uri`、artifact `uri`、响应中的 storage metadata、job payload/result 中的路径字段与 `download-url.object_uri` 会返回 `"<masked>"` 或被递归移除，不泄露本地路径或对象存储 URI；下载/预览/预签名内部仍使用真实 metadata。

---

## 七、知识库配置版本 Config Versions

> 📌 **完整字段速查见 [`docs/KB配置项速查表.md`](KB配置项速查表.md)**（每个 section 的字段、别名、影响哪个 hash、改动后的最小动作）。

> 不可变的 KB 级配置快照。新建配置不会自动生效，需要显式 `:activate` 才会写入 `KnowledgeBase.active_config_version_id` 并 discard 缓存的 LightRAG 实例。当前实现会让后续实例重建或 parse planning 时读取已支持的 active config 字段；部署级字段会在创建配置版本时直接拒绝，避免单个 KB 修改已经部署好的服务基础设施。创建时会严格校验：各 section（`parser_config`/`chunk_config`/`embedding_config`/`query_config`/`extraction_config`）出现**未知键**（无运行时效果）会直接返回 `400`，避免"存了不生效"。
>
> 已接入运行时的 active config 字段：
> - `parser_config`：`engine`/`parser_engine`、`process_options`/`options`。这些字段会在创建配置时校验并规范化，作为解析默认值参与 `parser_hash`，并按“请求 > 文档 metadata > active config > 文件路由”的优先级生效。
> - `chunk_config`：`chunk_size`/`chunk_token_size`、`chunk_overlap_size`/`chunk_overlap_token_size`、`tiktoken_model_name`。
> - `embedding_config`：`model`、`dim`/`embedding_dim`、`token_limit`/`max_token_size`（`model` 会触发重建 embedding provider 闭包）。
> - `query_config`：`top_k`/`chunk_top_k`/`max_entity_tokens`/`max_relation_tokens`/`max_total_tokens`/`related_chunk_number`/`cosine_threshold` 等 QueryParam 字段。
> - `extraction_config`：`language`（摘要/抽取语言）、`entity_types`（列表，自动渲染成 `entity_types_guidance` 并去重保序）或显式 `entity_types_guidance`（优先于 `entity_types`）、`entity_type_prompt_file`、`max_gleaning`/`max_extraction_records`/`max_extraction_entities`/`force_llm_summary_on_merge`。这些会 overlay 到 `addon_params` 与 LightRAG 抽取构造参数，并纳入 `index_hash`，因此变更会被 `:diff` 标为 `requires_reindex`。
> - `llm_role_config`：按角色（`extract`/`keyword`/`query`/`vlm`）覆盖运行时 LLM。每个角色可为字符串（等价 `{"model": <str>}`）或对象（`model`/`binding`/`host`/`api_key`/`provider_options`/`model_kwargs`(别名 `kwargs`)/`max_async`/`timeout`）。配置创建时校验角色名与字段名（未知项报错）。实例构建后通过已注册的 role builder 调用 `aupdate_llm_role_config` 应用覆盖，因此 `binding`/`model`/`host`/`api_key` 变更会重建该角色的 LLM func。哈希影响：`extract`/`vlm` 角色的“输出身份”（`binding`/`model`/`host`/`provider_options`/`model_kwargs`，不含 `api_key` 与 `max_async`/`timeout`）纳入 `index_hash`（变更触发 `requires_reindex`）；`query`/`keyword` 角色身份纳入 `query_hash`（仅影响查询，不重建）。轮换 `api_key` 或调 `max_async`/`timeout` 不改变任何哈希、不触发重建。
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
- 查询缓存为后续项。

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

支持的 `mode`：`local` / `global` / `hybrid` / `naive` / `mix` / `bypass`。`/query/stream` 与 KB scoped stream 一样返回 NDJSON；当请求体 `stream=false` 或底层返回非流式结果时，会返回单行完整 NDJSON。

### 9.3 图谱（无前缀）

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

### 9.4 Ollama 兼容（`/api`）

挂载 `OllamaAPI`，对外提供与 Ollama 接口兼容的端点：

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/version` | Ollama 兼容版本信息 |
| `GET` | `/api/tags` | 模型列表 |
| `GET` | `/api/ps` | 运行中模型列表 |
| `POST` | `/api/generate` | Ollama generate 兼容接口 |
| `POST` | `/api/chat` | Ollama chat 兼容接口 |

默认 `WHITELIST_PATHS` 仅放行 `/health`；如果要让 Ollama 兼容端点免 API Key，需要显式配置 `WHITELIST_PATHS=/health,/api/*`。企业模式下 `/api/*` 属于受保护前缀，不能被 whitelist 或全局 API key 默认绕过。

### 9.5 状态与认证基础接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 系统健康、配置和队列状态；默认 whitelist 放行 |
| `GET` | `/metrics` | Prometheus text format 指标（KB/doc/job/audit gauge + process-local HTTP counter/histogram）；受 `combined_auth` 保护，默认不在 whitelist；单服务器部署配套告警/SLO/dashboard 见 `deploy/monitoring/` |
| `GET` | `/auth-status` | 认证模式状态；非企业模式下可能签发 guest token |
| `POST` | `/login` | 非企业模式下使用 `AUTH_ACCOUNTS`；企业模式下使用企业用户表 |

---

## 十、企业模式 Auth / Admin

本节接口仅在 `LIGHTRAG_ENTERPRISE_AUTH_ENABLED=true` 时挂载或启用。企业模式默认禁用 guest 对受保护 API 的访问；`LIGHTRAG_API_KEY` 默认不能绕过 RBAC；`WHITELIST_PATHS` 不能放行 `/kbs`、`/documents`、`/query`、`/graph`、`/api` 等受保护前缀。企业 service API key 使用同一 `X-API-Key` 请求头，但与全局 `LIGHTRAG_API_KEY` 分离：只按持久化 hash 查找，默认只拥有创建时授予的 `kb_roles` scope；设置 `tenant_id + inherit_tenant_kb_acl=true` 时可显式继承 tenant-scoped KB ACL。service key 不能成为 super admin，撤销后立即失效。

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
```

### 10.2 登录与当前用户

当前企业认证支持企业用户表登录、JWT、service/scoped API key。SSO/OIDC/SAML **明确不做**：本系统按单机内网部署，无外部 IdP 接入需求（见 `docs/设计方案.md` §2.2）。

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/auth-status` | 企业模式返回 `auth_mode=enterprise` 和当前注册开关；不签发 guest token |
| `POST` | `/login` | 使用企业用户表认证，返回带 `user_id`、`system_role`、`token_version` metadata 的 JWT；同一用户名连续登录失败达 `LIGHTRAG_ENTERPRISE_LOGIN_MAX_ATTEMPTS`（默认 10）后锁定 `LIGHTRAG_ENTERPRISE_LOGIN_LOCKOUT_SECONDS`（默认 900s），期间返回 `429` + `Retry-After`；成功登录清零计数，`MAX_ATTEMPTS=0` 关闭锁定 |
| `POST` | `/auth/register` | 注册新用户；行为随注册模式而定：`open` 直接创建 active 用户并返回 token；`invite_only` 必须携带有效 `invitation_token`；`admin_approval` 创建 `pending` 用户、待管理员 `:enable` 审批后才能登录（响应不含 token）；`disabled` 返回 `403`。注册失败按 `LIGHTRAG_ENTERPRISE_REGISTRATION_*` 做单进程 per-username 锁定，失败/触发锁定写审计。新用户默认无 KB 权限且不可创建 KB |
| `GET` | `/auth/me` | 返回当前用户与 principal 权限信息；service API key 请求返回 `user:null` 与 service-key principal payload；用户对象含只读 `display_name` / `email` 个人资料字段 |
| `PATCH` | `/auth/me` | 当前用户维护个人资料：`display_name`（≤64 字符）/ `email`（≤254 字符、需含 `@`）。omitted=不变、显式 `null`=清除、空白串 `400`；存入 `enterprise_users.metadata`，**不**递增 `token_version`（当前 token 继续有效）；仅交互式 JWT 用户，API-key principal 返回 `403`；审计 `user_profile_updated` |
| `POST` | `/auth/logout` | 全设备登出：递增本人 `token_version`，使包括当前 token 在内的全部已签发 JWT 立即失效；返回 `{"status":"logged_out","token_version":N}`；仅交互式 JWT 用户，service key 返回 `403`（撤销 key 请用 `:revoke`）；审计 `user_logged_out` |
| `POST` | `/auth/change-password` | 当前用户修改密码；成功后 `token_version` 增加，旧 token 失效 |
| `GET` | `/auth/me/kbs/{kb_id}/query-settings` | 读取当前用户在指定 KB 下的个人查询设置；需 `kb_viewer`+；非交互式用户/API-key principal 返回 `403` |
| `PUT` | `/auth/me/kbs/{kb_id}/query-settings` | 写入/覆盖当前用户在指定 KB 下的个人 `user_prompt`；需 `kb_viewer`+；非交互式用户/API-key principal 返回 `403` |

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
```

当前用户 KB 查询设置说明：

- 存储维度为 `user_id + kb_id`，不同企业用户、不同知识库互不影响；默认 `user_prompt` 为空字符串。
- 读取或写入前会校验 KB 存在和当前 principal 至少拥有 `kb_viewer` 角色；无权限返回 `403`，KB 不存在返回 `404`。
- service/scoped API key 与 legacy enterprise API key superadmin 不属于交互式用户，不能使用该 self-service 设置接口，也不会在 query 时套用个人 `user_prompt`。
- `PUT` 传空字符串可清空个人提示词。清空后 query 回退到 active KB config 的 `query_config.user_prompt`；若也未配置则为空。

### 10.3 管理接口

以下接口均需 super admin：

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/admin/settings/registration` | 读取实时注册策略，返回 `enabled` 与 `mode` |
| `GET` | `/admin/overview` | 平台总览 JSON 聚合：KB 状态分布、文档/job/artifact 全局聚合与计数器合计、dead-letter 总数、企业用户/租户/service key/审计事件计数；仅查控制面，不加载引擎实例（面向管理台 dashboard，替代解析 `/metrics` 文本） |
| `PATCH` / `PUT` | `/admin/settings/registration` | 更新实时注册策略，body：`{"enabled": true}` 或 `{"mode":"open"}` |
| `GET` | `/admin/users` | 列出企业用户；支持 `status`/`tenant_id`/`q`(用户名子串) 过滤与 `limit`/`offset` 分页 |
| `POST` | `/admin/users` | 创建用户，可设置 `can_create_kb`、`can_use_bypass_query`、`can_delete_documents`、`tenant_id` |
| `GET` | `/admin/users/{user_id}` | 查询用户详情 |
| `GET` | `/admin/users/{user_id}/access` | 查看用户的访问总览：全局能力 + 租户成员关系(role) + 直接 KB ACL(kb_id/role，不含租户继承的有效角色) |
| `PATCH` | `/admin/users/{user_id}` | 更新用户状态/能力/tenant/password；请求体包含 `status`、`can_create_kb`、`can_use_bypass_query`、`can_delete_documents` 任一非 null 字段、显式给出 `tenant_id` 或修改 `password`，都会增加 `token_version` 并使旧 token 失效。`tenant_id` 区分 omitted 与显式 null：省略=不变，显式 `null`=清空租户归属，空/空白字符串返回 `400` |
| `POST` | `/admin/users/{user_id}:disable` | 禁用用户并递增 `token_version`，旧 token 失效 |
| `POST` | `/admin/users/{user_id}:enable` | 启用用户并递增 `token_version` |
| `POST` | `/admin/users/{user_id}:reset-password` | 重置用户密码并递增 `token_version` |
| `DELETE` | `/admin/users/{user_id}` | 删除用户；级联清理租户 membership、KB ACL、个人查询设置；不允许删除 super admin |
| `POST` | `/admin/users/{user_id}/kb-access:batch-set` | 按用户维度批量 grant/revoke 多个 KB ACL；与按 KB 维度 `/admin/kbs/{kb_id}/acl:batch-set` 互补 |
| `POST` | `/admin/tenants` | 创建租户实体（可指定 `tenant_id`，省略则生成 `tenant_<hex>`；重复 `409`） |
| `GET` | `/admin/tenants` | 列出所有租户 |
| `GET` | `/admin/tenants/{tenant_id}` | 租户详情 + 总览（含 `member_count` / `kb_count`） |
| `PATCH` | `/admin/tenants/{tenant_id}` | 更新租户 `name`/`description`/`status`（`active`/`disabled`） |
| `DELETE` | `/admin/tenants/{tenant_id}` | 删除租户实体；仅当无任何引用（成员/租户内 KB/归属用户/tenant-KB ACL）时允许，否则 `409`（不级联） |
| `GET` | `/admin/tenants/{tenant_id}/kbs` | 列出该租户下的 KB（`id`/`name`/`status`/`visibility`/`owner_id`） |
| `GET` | `/admin/tenants/{tenant_id}/members` | 列出 tenant 成员与 tenant role |
| `PUT` | `/admin/tenants/{tenant_id}/members/{user_id}` | 写入/更新 tenant membership，body：`{"role":"tenant_member"}` |
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
| `GET` | `/tenants/{tenant_id}/members` | tenant admin 查看本 tenant 成员 |
| `PUT` | `/tenants/{tenant_id}/members/{user_id}` | tenant admin 仅可授予/更新普通 `tenant_member`；不能提升为 admin/owner，也不能修改已有 admin/owner |
| `DELETE` | `/tenants/{tenant_id}/members/{user_id}` | tenant admin 仅可撤销普通 `tenant_member`；不能撤销 tenant admin/owner |

KB ACL 角色使用规范名称：`kb_viewer`、`kb_editor`、`kb_admin`、`kb_owner`。用户 effective KB role 取 direct user ACL 与其 tenant memberships 命中的 tenant-scoped KB ACL 的最高角色；没有 direct/tenant ACL 时跨 tenant 默认拒绝。Service/scoped API key 默认只按自身 `kb_roles` scope 授权；若创建时同时设置 `tenant_id` 与 `inherit_tenant_kb_acl=true`，则显式继承该 tenant 命中的 tenant-scoped KB ACL，并与 `kb_roles` 取最高角色。第一期中 KB ACL 仍由 super admin 统一执行；tenant admin 只具备本 tenant 成员自助管理权限，不具备 KB ACL 管理或删除 KB 的平台权限。

KB ACL 请求/响应约束：

- `PUT /admin/kbs/{kb_id}/acl` 请求体必须且只能包含 `user_id` 或 `tenant_id` 之一，并必须包含 `role`。
- `POST /admin/kbs/{kb_id}/acl:batch-set` 的每个 entry 必须且只能包含 `user_id` 或 `tenant_id` 之一；`action` 默认为 `grant`，取值 `grant` / `revoke`；`grant` 必须提供 `role`，`revoke` 不需要 `role`。
- ACL 响应对象字段为：`kb_id`、`user_id`、`tenant_id`、`principal_type`（`user` / `tenant`）、`role`、`granted_by`、`created_at`、`updated_at`。
- `batch-set` 响应中 `granted` 为 ACL 响应对象数组；`revoked` 为本次实际删除成功的 user id 或 tenant id 字符串数组。

`GET /admin/audit-events` 返回审计事件，按 `created_at DESC, id DESC` 排序；`limit` 默认 `100`，服务端会 clamp 到 `1..500`，`offset` 默认 `0` 用于分页。可选过滤参数（精确匹配，组合为 AND）：`event_type`、`actor_user_id`、`target_type`、`target_id`；时间范围 `created_after` / `created_before` 为 ISO-8601 字符串，按字典序与 `created_at` 比较（`>=` / `<=`）。例：`GET /admin/audit-events?event_type=kb_deleted&actor_user_id=usr_x&created_after=2026-06-01T00:00:00Z&limit=50&offset=50`。

审计事件响应字段：

```json
{
  "id": "audit_...",
  "event_type": "kb_created",
  "actor_user_id": "usr_...",
  "actor_username": "张三",
  "target_type": "kb",
  "target_id": "kb_...",
  "target_name": "橡胶研究知识库",
  "metadata": {},
  "created_at": "2026-06-08T...Z"
}
```

- `actor_username`：由 `actor_user_id` 在读取时动态解析（对齐 `EnterpriseUserResponse.username`），未找到时为 `null`。
- `target_name`：根据 `target_type` 动态解析——`kb` 返回知识库名称、`user` 返回用户名、`tenant` 返回租户名称（对齐各实体的 `name` / `username` 字段）；未找到时为 `null`。

企业模式已实现的审计事件类型包括：

- 登录/注册设置：`login_success`、`login_failed`、`registration_failed`、`registration_locked`、`registration_setting_updated`
- super admin bootstrap/sync：`super_admin_bootstrapped`、`super_admin_synced`
- 用户管理：`user_created`、`user_updated`、`user_deleted`、`user_password_changed`、`user_profile_updated`、`user_logged_out`
- service API key：`service_api_key_created`、`service_api_key_rotated`、`service_api_key_revoked`
- KB ACL / tenant：`kb_acl_granted`、`kb_acl_revoked`、`tenant_created`、`tenant_updated`、`tenant_deleted`、`tenant_membership_granted`、`tenant_membership_revoked`、`tenant_kb_acl_granted`、`tenant_kb_acl_revoked`
- 权限/限流/配额：`permission_denied`、`rate_limited`、`quota_exceeded`
- KB/config/query：`kb_created`、`kb_deleted`、`kb_hard_deleted`、`kb_restored`、`kb_config_activated`、`query_executed`、`query_stream_started`、`retrieve_executed`
- KB 图谱编辑：`kb_graph_entity_edited`、`kb_graph_entity_created`、`kb_graph_entity_deleted`、`kb_graph_entities_merged`、`kb_graph_relation_edited`、`kb_graph_relation_created`、`kb_graph_relation_deleted`
- artifact/job/document 类事件：`artifact_downloaded`、`artifact_previewed`、`artifact_download_url_created`、`kb_rebuild_queued`、`job_cancel_requested`、`job_retry_queued`、`document_batch_enabled`、`document_batch_disabled`，以及文档 upload/texts/urls/import/scan/sync/patch/enable/disable/replace/delete/batch-delete/parse/batch-parse/build/reindex/batch-build/batch-reindex/rebuild 相关事件。

审计覆盖：企业模式下，KB 创建/删除、config 激活、query/query-stream/retrieve、artifact download/preview/download-url、文档 upload/texts/urls/import/scan/sync/patch/enable/disable/replace/delete/batch-delete/parse/batch-parse/build/reindex/batch-build/batch-reindex/rebuild，以及 job cancel/retry 均写入 audit event。审计 metadata 采用白名单字段：query 仅记录 `query_hash`、mode、过滤摘要；文档与 artifact 事件仅记录 job/batch/document/artifact id、count、flag、hash、size/type 等，不记录 raw query、上传正文、URL、local path、presigned URL、密码/token/API key 明文。

管理请求体示例：

```json
// PATCH /admin/settings/registration
{"enabled": true}

// PATCH /admin/settings/registration
{"mode": "invite_only"}
// 返回：{"enabled": false, "mode": "invite_only"}

// POST /admin/users
{"username":"bob","password":"bob-pass","can_create_kb":true,"can_use_bypass_query":false,"can_delete_documents":false,"tenant_id":null}

// PATCH /admin/users/{user_id}
{"status":"active","can_create_kb":false,"can_use_bypass_query":true,"can_delete_documents":true,"tenant_id":"tenant-a","password":"new-pass"}

// POST /admin/users/{user_id}:reset-password
{"password":"new-pass"}

// PUT /admin/kbs/{kb_id}/acl
{"user_id":"usr_...","role":"kb_viewer"}

// PUT /admin/tenants/{tenant_id}/members/{user_id}
{"role":"tenant_member"}

// PUT /admin/kbs/{kb_id}/acl
{"tenant_id":"tenant-a","role":"kb_viewer"}

// POST /admin/kbs/{kb_id}/acl:batch-set
{"entries":[{"user_id":"usr_1","role":"kb_editor"},{"tenant_id":"tenant-a","role":"kb_viewer"},{"user_id":"usr_2","action":"revoke"},{"tenant_id":"tenant-b","action":"revoke"}]}
// 返回：{"granted":[...],"revoked":["usr_2","tenant-b"]}

// POST /admin/users/{user_id}/kb-access:batch-set
{"entries":[{"kb_id":"kb_a","role":"kb_viewer"},{"kb_id":"kb_b","role":"kb_editor"},{"kb_id":"kb_c","action":"revoke"}]}
// 返回：{"granted":[...],"revoked":["kb_c"]}

// POST /admin/service-api-keys
{"name":"ci-reader","kb_roles":{"kb_123":"kb_viewer"},"can_use_bypass_query":false,"inherit_tenant_kb_acl":false,"metadata":{"purpose":"ci"}}
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
- legacy/global `/documents`、`/query`、`/graph`、Ollama `/api/*` 在 `LIGHTRAG_ENTERPRISE_DISABLE_GLOBAL_ROUTES=true` 时默认拒绝；关闭该开关后仍需 super admin。
- super admin bootstrap 来自 `.env`，启动后同步为 active super admin；企业模式要求非默认 `TOKEN_SECRET`。
- Tenant membership 与 tenant-scoped KB ACL 已接入用户 principal hydration；用户请求按 direct user ACL 与 tenant ACL 最高角色授权。Service key 请求默认只按显式 `kb_roles` scope 授权；设置 `tenant_id + inherit_tenant_kb_acl=true` 时显式继承 tenant ACL。
- **KB 可见性（visibility）语义**：`knowledge_bases.visibility ∈ {private, internal, public}`，默认 `private`。`private` 无隐含权限；`internal` 在 KB `tenant_id` 非空时对该租户用户（`user.tenant_id` 相同或拥有该租户 membership）隐含 `kb_viewer`；`public` 对全部已认证企业交互用户（JWT principal）隐含 `kb_viewer`。隐含角色仅为只读（读 + query），写/配置/删除仍需显式 ACL / super admin；effective role 取 direct ACL、tenant ACL 与 visibility 隐含角色的最高者。service/scoped API key 与 legacy enterprise API key 不受 visibility 影响。`GET /admin/users/{id}/access` 仅列显式授权，不枚举 visibility 隐含项。修改 visibility 走 `PATCH /kbs/{kb_id}`（`kb_admin`+）。
- 文档删除为所有权感知模型：`kb_editor` 仅可删除本人上传（`metadata.created_by`）的文档；删除他人文档需用户级能力 `can_delete_documents`（super admin 通过 `/admin/users` 授予）或 `kb_admin`+/`super_admin`。service/scoped API key 不会获得 `can_delete_documents`：只能按 `kb_admin` scope 删任意，或按 `kb_editor` scope 删除该 key 自身上传的文档；无 `created_by` 的历史文档仅 privileged 主体可删。

KB 路由角色矩阵：

| 范围 | 最低角色 / 能力 |
|---|---|
| `POST /kbs` | super admin 或 `can_create_kb=true` |
| `GET /kbs` | super admin 看全部；普通用户看 direct user ACL / tenant ACL 授权 KB + visibility 命中 KB（`public` / 同租户 `internal`）；service key 默认仅看 `kb_roles` scope，显式 `inherit_tenant_kb_acl` 时额外继承 tenant ACL，不受 visibility 影响 |
| `GET /kbs/{kb_id}`、`GET /kbs/{kb_id}/status`、`/stats`、`/documents/{id}/chunks`、graph 读取、artifact/doc/job/config/query 读取 | `kb_viewer` 或更高（可由 visibility 隐含，见上）；artifact `:download` / `:download-url` 可由 `LIGHTRAG_ENTERPRISE_ARTIFACT_DOWNLOAD_MIN_ROLE` 全局提升最低角色，也可由 `LIGHTRAG_ENTERPRISE_ARTIFACT_DOWNLOAD_POLICY` 按 artifact type 覆盖；`:download` / `:download-url` / `:preview` 还可由 `LIGHTRAG_ENTERPRISE_ARTIFACT_ACTION_POLICY` 按 action + artifact type 覆盖 |
| `/kbs/{kb_id}/query`、`/query/stream`、`/query/data`、`/retrieve` | `kb_viewer` 或更高；最终 `mode="bypass"` 额外需要 `can_use_bypass_query=true` |
| `POST /kbs:query`、`/kbs:query/stream`、`/kbs:retrieve`（跨库合并查询） | `kb_ids` 中每个 KB 均需 `kb_viewer`+（handler 自鉴权，中央中间件不覆盖 collection 级路径）；`bypass` 不支持(400) |
| 文档上传/解析/构建/替换/sync、批量启停（`:batch-enable`/`:batch-disable`）、`:rebuild`、job wait/cancel/retry 等写操作 | `kb_editor` 或更高 |
| 文档删除（`DELETE …/documents/{id}`、`:batch-delete`） | `kb_editor` 仅删本人上传(`metadata.created_by`)的文档；删他人需 `can_delete_documents` 能力或 `kb_admin`+/`super_admin` |
| KB 配置创建/激活/diff、`PATCH /kbs/{kb_id}`、图谱编辑（`/graph` 下全部非 GET 端点） | `kb_admin` 或更高 |
| `DELETE /kbs/{kb_id}`、`?hard=true`、`POST /kbs/{kb_id}:restore`、`/admin/...` | super admin |

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
