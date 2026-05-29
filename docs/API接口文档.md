# LightRAG API 接口文档

> 文档版本：2026-05-29
> 适用范围：当前已经合并到 `main` 分支并通过测试的接口。
> 路径前缀：所有路径均为相对路径；部署时通过 FastAPI `root_path` 或 `--api-prefix /api/v1` 暴露为 `/api/v1/...`。
> 鉴权：除 `/health`、`/auth-status`、`/login` 等少数公开接口外，所有接口都受 `combined_auth` 依赖保护，需要在请求头携带 `X-API-Key: <api_key>` 或 JWT。
> 配套文档：`docs/生产级后端改造设计方案.md`。

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
- [十、状态机与字段说明](#十状态机与字段说明)

---

## 一、知识库管理 KB

> 知识库是所有 KB 接口的边界。`kb_id` 派生出 LightRAG 的 `workspace`，并由 `LightRAGInstanceRegistry` 按需懒加载实例。

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/kbs` | 创建知识库 |
| `GET` | `/kbs` | 列出所有知识库 |
| `GET` | `/kbs/{kb_id}` | 获取知识库详情 |
| `PATCH` | `/kbs/{kb_id}` | 局部更新知识库（名称、描述、状态等） |
| `DELETE` | `/kbs/{kb_id}` | 软删除知识库；附加 `?hard=true` 触发异步硬删除 |
| `GET` | `/kbs/{kb_id}/status` | 知识库状态聚合（含运行中任务、pipeline 状态） |

### 1.1 创建知识库

```http
POST /kbs
Content-Type: application/json

{
  "id": "kb_research",            // 可选，省略由服务端生成 kb_<12位hex>
  "name": "Research Papers",      // 必填，去首尾空白后非空
  "description": "Optional",       // 可选
  "owner_id": null,                // 预留多租户字段，暂不强制
  "tenant_id": null,
  "visibility": "private"          // 枚举：private / public / internal
}
```

返回 `200 KnowledgeBaseResponse`；冲突 `409`；参数非法 `400`。

### 1.2 列出 / 获取 / 更新 / 删除

- `GET /kbs?include_deleted=false`：默认排除软删除记录。
- `GET /kbs/{kb_id}`：404 表示未找到或已软删除。
- `PATCH /kbs/{kb_id}`：仅更新请求体显式给出的字段；`status` 不允许直接置为 `deleted`。
- `DELETE /kbs/{kb_id}`：默认软删除，同步从 `LightRAGInstanceRegistry` 卸载实例。
- `DELETE /kbs/{kb_id}?hard=true`：触发同步硬删除流程。`KBDeletionService` 在 destructive lock 下依次执行：
  1. `force_evict` 在内存中的 LightRAG 实例并调用 `finalize_storages`；
  2. 删除 `working_dir/<workspace>`（如已配置）；
  3. 删除 `input_dir/<workspace>`（上传文件 + 解析 artifact）；
  4. 清空 SQLite 控制面（documents / jobs / artifacts / config_versions）。
  返回前会创建一条 `clear_kb` 类型的 job 记录最终结果；任一步失败 HTTP 500 + `clear_kb` job 终态 `failed`。

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

---

## 二、知识库文档 Documents

> 文档生命周期由 `DocumentLifecycleService` 管理，元数据落 SQLite（`working_dir/metadata/metadata.sqlite3`）。同名文件会写入独立的子目录，跨进程并发写不会互相覆盖。

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/kbs/{kb_id}/documents:upload` | 多文件上传，可选自动解析 |
| `POST` | `/kbs/{kb_id}/documents:sync` | 按 `source_key` 批量增量同步，可自动解析并构建到可问答状态 |
| `POST` | `/kbs/{kb_id}/documents:texts` | 批量文本导入 |
| `GET` | `/kbs/{kb_id}/documents` | 文档列表，支持状态、文件名过滤 |
| `GET` | `/kbs/{kb_id}/documents/{document_id}` | 文档详情 |
| `PATCH` | `/kbs/{kb_id}/documents/{document_id}` | 更新 metadata / enabled / archived |
| `POST` | `/kbs/{kb_id}/documents/{document_id}:disable` | 独立禁用文档（仅控制面 metadata） |
| `POST` | `/kbs/{kb_id}/documents/{document_id}:enable` | 独立启用文档（仅控制面 metadata） |
| `DELETE` | `/kbs/{kb_id}/documents/{document_id}` | 单文档任务化删除 |
| `POST` | `/kbs/{kb_id}/documents/{document_id}:replace` | 单文档任务化替换 |
| `POST` | `/kbs/{kb_id}/documents:batch-delete` | 批量任务化删除 |

### 2.1 多文件上传

```http
POST /kbs/{kb_id}/documents:upload?auto_parse=true&auto_index=false&parser_engine=mineru&process_options=iF&idempotency_key=upload-001
Content-Type: multipart/form-data

files: [a.pdf, b.docx]
```

约束：
- 单请求最多 32 个文件，单文件和单请求总字节数均不得超过 `MAX_UPLOAD_SIZE`，未配置或非正数时 `413`。
- 文件扩展名必须在 `SUPPORTED_DOCUMENT_EXTENSIONS` 列表中。
- `auto_parse=true` 会创建 `parse` 队列任务；`auto_parse=false` 仅落 metadata，job 立即标记 `succeeded`。
- 同名文件会写入独立的 `<workspace>/<document_id>/<filename>` 目录，使用独占创建 (`O_EXCL`)。

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

### 2.3 批量增量同步

```http
POST /kbs/{kb_id}/documents:sync?auto_parse=true&auto_index=true&parser_engine=mineru&process_options=iF&idempotency_key=sync-001
Content-Type: multipart/form-data

files: [a.pdf, b.pdf]
source_keys: ["manual/a.pdf", "manual/b.pdf"]
```

行为：
- `source_key` 是生产增量同步的稳定业务身份，同一 KB 内用它判断同一份外部文档；建议使用对象存储 key、相对路径或外部系统 document id，不要只用展示文件名。
- `source_key` 在同一 KB 内由 metadata store 原子唯一约束；并发 sync 不会为同一个外部文档创建两个活动 KB 文档。
- 服务端先读取文件内容并计算 `source_hash`，再查找相同 `source_key` 的现有文档。
- 找不到 `source_key`：创建新文档；`source_hash` 相同：跳过 source 替换，但若当前请求的 `parser_engine/process_options` 派生出的 `parser_hash` 与文档上次成功解析的值不同，仍会重解析并继续重建；`source_hash` 不同：复用单文档 replace 语义，保留原 `document_id`，先删除旧 `lightrag_doc_id` 后替换 source。
- `auto_parse=true` 默认继续解析；`auto_index=true` 默认在解析成功后继续构建 KG/index，使成功 item 到达 `ready` 并可直接走 KB query。
- 返回单个聚合 `sync` job。每个 item 在 `job.result.items[]` 中记录 `source_key`、`action`（`created` / `replaced` / `skipped` / `reparsed`）、`status`、`parse_result`、`build_result` 等；单个 item 失败不会阻塞其他 item，active parse/build/delete/replace 会保留对应 `*_job_active` 错误码和 `existing_job_id`。
- `idempotency_key` 在 `(kb_id, job_type=sync)` 维度唯一；同 key 同文件和同参数复用原 job，同 key 不同请求返回 `409`。

### 2.4 文档列表 / 详情

```http
GET /kbs/{kb_id}/documents?status=parsed&source_name=paper&limit=50&offset=0
GET /kbs/{kb_id}/documents/{document_id}
```

`source_name` 使用 SQL `LIKE` 模糊匹配（大小写不敏感）。

### 2.5 文档 PATCH

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

### 2.6 独立启用 / 禁用

```http
POST /kbs/{kb_id}/documents/{document_id}:disable
POST /kbs/{kb_id}/documents/{document_id}:enable
```

返回 `DocumentResponse`。这两个动作只更新 SQLite 控制面 `enabled` 字段，不删除 source/artifact，也不触发 LightRAG storage 变更。当前 QueryParam 尚未接入按文档过滤，因此它们先作为 metadata control-plane 能力提供。

### 2.7 文档删除

```http
DELETE /kbs/{kb_id}/documents/{document_id}?delete_source_file=false&delete_artifacts=false&delete_llm_cache=false&idempotency_key=delete-001
```

行为：
- 创建 `delete` job，并将文档原子 claim 到 `deleting`；已有 `parse_queued/parsing`、`build_queued/building`、`deleting` 或 `replacing` 时返回 `409`。
- 若文档已有 `lightrag_doc_id`，后台任务调用 `LightRAG.adelete_by_doc_id`；底层返回 `success` 或 `not_found` 都视为删除成功，适配尚未入库或已被清理的文档。
- `delete_source_file=true` / `delete_artifacts=true` 时仅允许删除 `INPUT_DIR/<workspace>/<document_id>/...` 内的 source/artifact 文件或目录，路径逃逸会使 job 失败并保留文档为 `delete_failed`。
- 成功后文档软删除为 `deleted` 并写入 `deleted_at`，列表和详情默认不再返回该文档，artifact metadata 同步清理。

批量删除：

```http
POST /kbs/{kb_id}/documents:batch-delete
Content-Type: application/json

{
  "document_ids": ["doc_a", "doc_b"],
  "delete_source_file": false,
  "delete_artifacts": false,
  "delete_llm_cache": false,
  "idempotency_key": null
}
```

创建单个聚合 `delete` job（`document_id=null`、`batch_id` 非空）。每个 item 独立 claim 和执行；active job、缺失文档等作为 per-item failure 写入 `job.result.items[]`，不阻塞其他可删除文档。

### 2.8 文档替换

```http
POST /kbs/{kb_id}/documents/{document_id}:replace?auto_parse=true&auto_index=false&parser_engine=mineru&process_options=iF&force_reparse=false&delete_source_file=true&delete_artifacts=true&delete_llm_cache=false&idempotency_key=replace-001
Content-Type: multipart/form-data

file: new-paper.pdf
```

行为：
- 创建 `replace` job，并将文档原子 claim 到 `replacing`；已有 `parse_queued/parsing`、`build_queued/building`、`deleting` 或 `replacing` 时返回 `409`。
- 若旧文档已有 `lightrag_doc_id`，后台任务先调用 `LightRAG.adelete_by_doc_id` 清理旧索引；底层返回 `success` 或 `not_found` 都视为可继续替换。
- `delete_source_file=true` / `delete_artifacts=true` 时只允许清理 `INPUT_DIR/<workspace>/<document_id>/...` 内的旧 source/artifact；路径逃逸会使 job 失败，文档进入 `replace_failed`。
- 替换成功后保留原 `document_id`，写入新的 `source_name/source_uri/source_hash/content_type/size_bytes`，清空旧 `parser_hash/index_hash/lightrag_doc_id/chunks_count/entity_count/relation_count` 和解析/索引派生 metadata，并回到 `uploaded`。
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
  "auto_index": false,            // 预留：解析成功后是否触发 build_kg
  "idempotency_key": null
}
```

行为：
- 若 `(source_hash, parser_hash)` 命中 cache，直接复用 artifacts。
- 同一文档已有 `parse_queued` / `parsing` / `build_queued` / `building` / `deleting` / `replacing` 时返回 `409`，原 active job 保持不变，新建的 job 同步标记 `failed`。
- 成功后写入 `original` / `sidecar` / `blocks` artifact，MinerU/Docling 还会写 `raw_dir`。

### 3.2 批量解析

```http
POST /kbs/{kb_id}/documents:batch-parse
Content-Type: application/json

{
  "document_ids": ["doc_a", "doc_b"],
  "engine": "mineru",
  "process_options": "iF",
  "force_reparse": false,
  "auto_index": false,
  "idempotency_key": null
}
```

行为：
- 创建单个聚合 `parse` job（`document_id=null`、`batch_id` 非空）。
- 每个 item 独立成功 / 失败，记录在 `result.items[]`。
- 任一 item 失败时聚合 job 终态为 `failed`，但已成功 item 不回滚。

---

## 四、知识库构建 Index / KG

> 基于解析产物驱动 LightRAG 的 chunk → 实体关系抽取 → embedding → KG merge 流水线。增量入库通过 `index_hash` 三段判断实现。

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/kbs/{kb_id}/documents/{document_id}:build-kg` | 单文档构建知识图谱与索引 |
| `POST` | `/kbs/{kb_id}/documents:batch-build-kg` | 批量构建（聚合任务） |
| `POST` | `/kbs/{kb_id}/documents/{document_id}:reindex` | 单文档强制重建索引（默认所有 force 标志为 true） |
| `POST` | `/kbs/{kb_id}/documents:batch-reindex` | 批量强制重建 |

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

`:reindex` 与 `:build-kg` 共用同一份后台执行逻辑，唯一区别是默认所有 `force_*` 为 `true`，永远不会触发 hash skip。

---

## 五、知识库任务 Jobs

> 任务持久化在 SQLite，跨进程可见。所有耗时操作均会创建 job，客户端通过 `job_id` 跟踪进度。

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/kbs/{kb_id}/jobs` | 任务列表，支持状态 / 文档 ID 过滤 |
| `GET` | `/kbs/{kb_id}/jobs/{job_id}` | 任务详情 |
| `POST` | `/kbs/{kb_id}/jobs/{job_id}:wait` | 阻塞等待任务到达终态（succeeded / failed / cancelled） |
| `POST` | `/kbs/{kb_id}/jobs/{job_id}:cancel` | 取消任务 |
| `POST` | `/kbs/{kb_id}/jobs/{job_id}:retry` | 重试 failed / cancelled 任务 |

### 5.1 列表 / 详情

```http
GET /kbs/{kb_id}/jobs?status=running&document_id=doc_xxx&limit=50&offset=0
GET /kbs/{kb_id}/jobs/{job_id}
```

任务字段（`JobResponse`）核心列：

| 字段 | 说明 |
|---|---|
| `id` | 任务 ID |
| `job_type` | `upload` / `parse` / `build_kg` |
| `status` | `queued` / `running` / `succeeded` / `failed` / `cancelling` / `cancelled` / `retrying` |
| `stage` | 当前阶段：`uploading` / `parsing` / `building` |
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
- `running` / `retrying` → `cancelling`，由后台 worker 在下次检查点终止后转 `cancelled`。
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
- 任务回到 `queued`，清空 `result` / `error_code` / `error_message` / `started_at` / `finished_at` / `cancelled_at`。
- `retry_count += 1`；超过 `max_retries`（默认 3）返回 `409`。
- 注意：当前 worker 是 in-process 后台任务，重试后需要由调用方再次触发同一接口（durable worker 自动恢复仍待实现）。

### 5.4 等待任务终态

```http
POST /kbs/{kb_id}/jobs/{job_id}:wait?timeout_seconds=60&poll_interval_seconds=0.5
```

服务端持续轮询 SQLite 直到任务进入 `succeeded` / `failed` / `cancelled` 三态之一并返回最终 `JobResponse`；超时未到终态返回 `408 Request Timeout` 携带 `current_status`。

约束：
- `timeout_seconds` 限制在 `[0.1, 600.0]`；客户端可按需调小。
- `poll_interval_seconds` 限制在 `[0.05, 5.0]`，默认 0.5s。
- 该接口存在的目的是让客户端写线性脚本（`upload -> wait -> build -> wait -> query`）时不必自己实现轮询逻辑。

---

## 六、知识库产物 Artifacts

> 产物记录解析阶段产生的文件 / 目录。当前支持 `original` / `sidecar` / `blocks` / `raw_dir` 四种类型。

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/kbs/{kb_id}/documents/{document_id}/artifacts` | 产物列表 |
| `GET` | `/kbs/{kb_id}/documents/{document_id}/artifacts/{artifact_id}` | 产物元数据 |
| `GET` | `/kbs/{kb_id}/documents/{document_id}/artifacts/{artifact_id}:download` | 下载文件型产物 |

下载约束：
- 文件型产物（`original` / `blocks`）以 `FileResponse` 直接返回。
- 目录型产物（`sidecar` / `raw_dir`）以流式 zip 返回（`Content-Type: application/zip`），单次下载 zip 内未压缩字节上限 512 MB，超限返回 `413`。
- 路径必须位于 `inputs/<workspace>/<document_id>` 内；跨 KB、缺失文件、路径逃逸均返回 `404` / `400`。

---

## 七、知识库配置版本 Config Versions

> 不可变的 KB 级配置快照。新建配置不会自动生效，需要显式 `:activate` 才会写入 `KnowledgeBase.active_config_version_id` 并 discard 缓存的 LightRAG 实例。

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/kbs/{kb_id}/configs` | 创建配置版本（自动派生 `parser_hash` / `index_hash` / `query_hash`） |
| `GET` | `/kbs/{kb_id}/configs` | 列出所有配置版本 |
| `GET` | `/kbs/{kb_id}/configs/{version_id}` | 获取配置版本详情 |
| `POST` | `/kbs/{kb_id}/configs/{version_id}:activate` | 激活配置版本 |
| `POST` | `/kbs/{kb_id}/configs/{version_id}:diff` | 与当前激活版本做 diff，预测重建影响 |

### 7.1 创建配置版本

```http
POST /kbs/{kb_id}/configs
Content-Type: application/json

{
  "config": {
    "parser_config": {"engine": "mineru"},
    "chunk_config": {"chunk_size": 512},
    "embedding_config": {"model": "bge-large", "dim": 1024},
    "llm_role_config": {"extract": "gpt-4o-mini"},
    "query_config": {"top_k": 60}
  },
  "created_by": "alice"
}
```

返回 `ConfigVersionResponse`，`version` 由服务端按 KB 内单调递增生成。

### 7.2 激活配置

```http
POST /kbs/{kb_id}/configs/{version_id}:activate
```

行为：
- 更新 KB 的 `active_config_version_id`。
- 写入配置版本的 `activated_at`。
- 调用 `LightRAGInstanceRegistry.discard(kb_id)` 卸载实例，下次请求按新配置重建。
- 若该 KB 上有 destructive job 在执行（如 `clear_kb`），discard 静默跳过。

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

请求体（与全局 `/query` 共用字段，新增 `filters.doc_ids`）：

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
    "doc_ids": ["doc_xxx"]
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
  ]
}
```

约束：
- 同 KB 内的查询不会读取其他 KB 的内容（`workspace` 隔离）；已加测试覆盖。
- 若 KB 内存在 `deleting` / `replacing` 文档，或 `filters.doc_ids` 指向此类 active 文档，查询返回 `409`，避免读到删除/替换中的旧内容。
- `mode` 支持 `local / global / hybrid / naive / mix / bypass`；建议默认 `mix`。
- `filters.doc_ids` 当前阶段会校验 ID 必须属于本 KB（不在则 400 + `error_code=doc_ids_not_in_kb`）；retrieval 内部按 doc 精确过滤待 LightRAG QueryParam 增强后接入，KB 边界已由 workspace 保证。
- `include_chunk_content=true` 时 `references[].content` 返回该 reference 命中的 chunk 文本数组，便于评估与排查。
- 流式响应 `Content-Type: application/x-ndjson`：第一行是 `{kb_id, references}`，后续每行 `{response: "..."}`，错误时 `{error: "..."}`。
- 短查询（< 3 字符）返回 422；KB 不存在 404。

---

## 九、兼容旧版 / 全局接口

> 这些接口走全局默认 `workspace`，主要给现有 WebUI 与早期客户端使用。生产新接入建议使用 `/kbs/...` 系列。

### 7.1 文档（`/documents`）

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/documents/scan` | 扫描 `input_dir` 并入库 |
| `POST` | `/documents/upload` | 单文件上传（旧版） |
| `POST` | `/documents/text` | 单文本插入 |
| `POST` | `/documents/texts` | 批量文本插入 |
| `DELETE` | `/documents` | 清空所有文档 |
| `GET` | `/documents/pipeline_status` | 全局 pipeline 状态 |
| `DELETE` | `/documents/delete_document` | 按 ID 删除文档 |
| `POST` | `/documents/clear_cache` | 清理 LLM 缓存 |
| `DELETE` | `/documents/delete_entity` | 删除实体 |
| `DELETE` | `/documents/delete_relation` | 删除关系 |
| `GET` | `/documents/track_status/{track_id}` | 跟踪 ID 状态查询 |
| `GET` | `/documents/paginated` | 分页文档状态 |
| `GET` | `/documents/status_counts` | 状态统计 |
| `POST` | `/documents/reprocess_failed` | 重处理失败文档 |
| `POST` | `/documents/cancel_pipeline` | 取消运行中的 pipeline |

### 7.2 查询（无前缀，挂在根路径）

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/query` | 非流式问答 |
| `POST` | `/query/stream` | 流式问答（SSE） |
| `POST` | `/query/data` | 仅返回结构化检索数据，不调用 LLM 生成 |

支持的 `mode`：`local` / `global` / `hybrid` / `naive` / `mix` / `bypass`。

### 7.3 图谱（无前缀）

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

### 7.4 Ollama 兼容（`/api`）

挂载 `OllamaAPI`，对外提供与 Ollama 接口兼容的端点（`/api/tags`、`/api/chat` 等）。

---

## 十、状态机与字段说明

### 8.1 文档状态

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

### 8.2 任务状态机（已实现部分）

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

### 8.3 三段 Hash 含义

| Hash | 派生因子 | 变化时的最小动作 |
|---|---|---|
| `source_hash` | 上传 / 文本内容字节 | 重新解析 + 重新构建 |
| `parser_hash` | 解析引擎 + process options | 重新解析 + 重新构建 |
| `index_hash` | chunker / embedding / extraction prompt / language / entity_types 等 | 仅重新构建索引（复用解析产物） |

`:build-kg` 命中 `index_hash` 且文档已 `ready` 时直接 skip；`:reindex` 始终绕过 skip。

### 8.4 幂等键约定

- 幂等键唯一索引：`(kb_id, job_type, idempotency_key)`。
- 文本导入、批量增量同步、单文档 parse、批量 parse、单文档 build、批量 build、单文档 replace 都支持幂等键。
- 同 key 同请求指纹返回原 job；同 key 不同请求指纹返回 `409`。

### 8.5 错误码归纳

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
| 413 | - | 上传体积超出 `MAX_UPLOAD_SIZE` 或文本超限 |
| 503 | - | 注册表 / 构建服务未配置 |
