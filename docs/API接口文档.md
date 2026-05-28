# LightRAG API 接口文档

> 文档版本：2026-05-28  
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
- [七、兼容旧版 / 全局接口](#七兼容旧版--全局接口)
- [八、状态机与字段说明](#八状态机与字段说明)

---

## 一、知识库管理 KB

> 知识库是所有 KB 接口的边界。`kb_id` 派生出 LightRAG 的 `workspace`，并由 `LightRAGInstanceRegistry` 按需懒加载实例。

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/kbs` | 创建知识库 |
| `GET` | `/kbs` | 列出所有知识库 |
| `GET` | `/kbs/{kb_id}` | 获取知识库详情 |
| `PATCH` | `/kbs/{kb_id}` | 局部更新知识库（名称、描述、状态等） |
| `DELETE` | `/kbs/{kb_id}` | 软删除知识库（异步硬删除待实现） |
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
- `DELETE /kbs/{kb_id}`：软删除。同步从 `LightRAGInstanceRegistry` 卸载实例。

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
| `POST` | `/kbs/{kb_id}/documents:texts` | 批量文本导入 |
| `GET` | `/kbs/{kb_id}/documents` | 文档列表，支持状态、文件名过滤 |
| `GET` | `/kbs/{kb_id}/documents/{document_id}` | 文档详情 |
| `PATCH` | `/kbs/{kb_id}/documents/{document_id}` | 更新 metadata / enabled / archived |

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

### 2.3 文档列表 / 详情

```http
GET /kbs/{kb_id}/documents?status=parsed&source_name=paper&limit=50&offset=0
GET /kbs/{kb_id}/documents/{document_id}
```

`source_name` 使用 SQL `LIKE` 模糊匹配（大小写不敏感）。

### 2.4 文档 PATCH

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
- 同一文档已有 `parse_queued` / `parsing` 时返回 `409`，原 active job 保持不变，新建的 job 同步标记 `failed`。
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

---

## 六、知识库产物 Artifacts

> 产物记录解析阶段产生的文件 / 目录。当前支持 `original` / `sidecar` / `blocks` / `raw_dir` 四种类型。

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/kbs/{kb_id}/documents/{document_id}/artifacts` | 产物列表 |
| `GET` | `/kbs/{kb_id}/documents/{document_id}/artifacts/{artifact_id}` | 产物元数据 |
| `GET` | `/kbs/{kb_id}/documents/{document_id}/artifacts/{artifact_id}:download` | 下载文件型产物 |

下载约束：
- 仅文件型产物（`original` / `blocks`）可下载；目录型 `sidecar` / `raw_dir` 当前返回 `400 directory artifact cannot be downloaded directly`。
- 路径必须位于 `inputs/<workspace>/<document_id>` 内；跨 KB、缺失文件、路径逃逸均返回 `404` / `400`。

---

## 七、兼容旧版 / 全局接口

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

## 八、状态机与字段说明

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
```

辅助状态：`disabled` / `archived` / `deleting` / `deleted`（部分仍待实现）。

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
- 文本导入、单文档 parse、批量 parse、单文档 build、批量 build 都支持幂等键。
- 同 key 同请求指纹返回原 job；同 key 不同请求指纹返回 `409`。

### 8.5 错误码归纳

| HTTP | 业务错误码 | 含义 |
|---|---|---|
| 400 | invalid_parse_request / parser_engine_unsupported | 参数不合法 |
| 404 | KnowledgeBaseNotFoundError / MetadataRecordNotFoundError | KB / 文档 / 任务 / 产物未找到 |
| 409 | parse_job_active | 文档已有运行中的解析任务 |
| 409 | build_job_active | 文档已有运行中的构建任务 |
| 409 | document_not_parsed | 文档尚未完成解析，无法触发构建 |
| 409 | IdempotencyKeyConflict | 同幂等键不同请求指纹 |
| 409 | InvalidJobTransitionError | 任务状态不允许该转换 |
| 413 | - | 上传体积超出 `MAX_UPLOAD_SIZE` 或文本超限 |
| 503 | - | 注册表 / 构建服务未配置 |
