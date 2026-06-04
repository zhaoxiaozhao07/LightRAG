# Enterprise KB MVP API 演示

此目录包含一个对 Windows 友好的 API 编排脚本，用于在不依赖 WebUI 的情况下模拟企业知识库落地流程。

该脚本会端到端驱动 KB API，完成以下流程：

1. 健康检查与运行时存储摘要；
2. 创建或复用一个隔离的 KB（`/kbs`）；
3. 基于本地 `.env` 中筛选后的安全配置，创建并激活一个 KB 配置快照；
4. 从 `E:\\pycharmprojects\\RAG\\LightRAG\\模拟文件` 导入文件；
5. 在启用时通过 MinIO/S3 持久化源码文件与产物；
6. 使用 MinerU / native 路由解析文件；
7. 构建 KG / 索引，包括实体关系抽取与向量写入；
8. 检查文档、任务、死信、产物、图状态、实体、关系；
9. 可选地运行交互式文档删除测试；
10. 执行 KB 作用域下的 RAG 查询与结构化检索查询；
11. 创建一个空的控制 KB，用于验证工作区隔离。

除上述基线流程（约 26 个端点）外，还可通过可选标志触发更多端点覆盖，包括：重建 / 重建索引、文档替换、文本 / URL 导入、流式 + 结构化检索、配置获取 / diff、KB 元数据 patch、产物元数据 / 下载 URL，以及文档启用 / 禁用。详见下方“扩展端点覆盖”。

在仓库根目录运行：

```powershell
$env:LIGHTRAG_API_KEY = "sk-123456"
uv run python examples/enterprise_kb_mvp/enterprise_kb_mvp_demo.py `
  --server "http://127.0.0.1:9621" `
  --source-dir "E:/pycharmprojects/RAG/LightRAG/模拟文件" `
  --kb-id enterprise_mvp_demo `
  --kb-name "企业知识库 MVP 模拟" `
  --parser-engine mineru `
  --process-options iteP
```

常用的重复运行参数：

- `--skip-ingest`：复用已有 KB，只做检查 / 查询。
- `--skip-query`：仅执行导入 / 构建。
- `--manual-flow`：使用显式的 upload -> parse -> build-kg 调用链，而不是 `documents:sync?auto_parse=true&auto_index=true`。
- `--max-files 1`：仅针对一个文件做快速 smoke 测试。
- `--run-id stable-id`：使用稳定的幂等键 / 报告后缀。仅在对相同文件集合进行完全重试时复用同一个 run id；如果文件新增或变更，请使用新的 run id。
- `--reset-kb ask|yes|no`：可在运行前选择性硬删除主 KB 和隔离控制 KB。默认 `ask` 仅在交互式终端中提示，在非交互 shell 中跳过重置以避免误删。`yes` 无提示直接重置；`no` 始终跳过。重置会调用 `DELETE /kbs/{kb_id}?hard=true`，因此会清除 KB 元数据记录、LightRAG 工作区文件、解析器输入 / 产物缓存，以及与该 KB 工作区相关联的 MinIO / S3 对象。

## 扩展端点覆盖（可选）

基线流程已经覆盖约 26 个 KB 端点。以下标志会启用额外、默认未覆盖的端点。它们默认都处于 **关闭** 状态，因此不会改变基线运行流程；同时它们创建的每个任务都会被持续跟踪，不会因为客户端超时而中断（见下文“长时任务永不超时”）。

- `--demo-extras`：在主查询之后执行以读取为主、可逆的端点：
  - `POST /kbs/{id}/retrieve`：结构化检索，不触发 LLM 生成；
  - `POST /kbs/{id}/query/stream`：NDJSON 流式输出（报告中会记录 token 数）；
  - `GET /kbs/{id}/graph`：导出子图；
  - `GET /kbs/{id}/configs/{version_id}` + `POST .../{version_id}:diff`：查看某个配置版本并与当前激活版本比较差异；
  - `PATCH /kbs/{id}`：描述字段往返测试（patch 后再恢复）；
  - `GET .../artifacts/{artifact_id}` + `:download-url`：获取产物元数据和预签名下载 URL；
  - 文档 `:disable` -> `PATCH` 元数据 -> `:enable` 的往返测试；
  - `POST /kbs/{id}/jobs/{job_id}:retry`：如果存在死信任务，则重试第一个死信任务。
- `--demo-reindex`：重建路径，包括单文档 `:reindex`、`documents:batch-reindex` 以及 `{kb}:rebuild`。这些操作会重新执行 chunk / extract / embedding（可能较慢），并端到端验证：在自研 `_VDBUpsertBatcher` 被上游存储层延迟 embedding 实现替换后，向量重建仍能正常工作。
- `--demo-replace FILE`：通过 `POST /kbs/{id}/documents/{document_id}:replace` 替换第一个 ready 文档的源文件（multipart 上传 + 可恢复续传）。其中 `FILE` 为新的源文件。
- `--demo-ingest-variants`（可搭配 `--demo-url URL`）：测试非文件导入通道 `documents:texts`（合成文本）和 `documents:urls`（仅在提供 `--demo-url` 时启用）。两者都会执行 `auto_parse` + `auto_index`。

可选示例：启用所有以读取为主的额外功能，以及重建路径和文本导入：

```powershell
$env:LIGHTRAG_API_KEY = "sk-123456"
uv run python examples/enterprise_kb_mvp/enterprise_kb_mvp_demo.py `
  --server "http://127.0.0.1:9621" `
  --source-dir "E:/pycharmprojects/RAG/LightRAG/模拟文件" `
  --kb-id enterprise_mvp_demo `
  --reset-kb yes `
  --demo-extras `
  --demo-reindex `
  --demo-ingest-variants
```

若要一并测试文档替换，可额外添加 `--demo-replace path/to/new_file.pdf`。
若要一并抓取 URL，可在 `--demo-ingest-variants` 的同时添加 `--demo-url https://example.com/page`。

## 长时任务永不超时

每个 ingest / parse / build / reindex / rebuild / replace 端点都会返回一个异步任务。客户端会通过 `wait_for_job` 跟踪每个任务：它会发起一个有界的服务端 `:wait`（默认窗口为 120 秒），当收到其 408 心跳响应后，会重新查询进度并再次发起等待。因此，即便是很慢的 MinerU 解析，或者耗时数小时的多 PDF KG 构建，也会持续输出进度，而不会被中途放弃。默认 `--job-timeout 0` 的含义是“持续跟踪直到任务终止”；服务端无论如何都会继续运行，因此若客户端过早放弃，只会遗留一个仍在执行的任务。只有在你明确希望客户端提前退出时，才应传入正值 `--job-timeout SECONDS`。

上述扩展端点也复用了完全相同的跟踪机制，通过 `follow_job_response` 透明处理 API 返回的两种响应形态——`JobResponse`（`id` + `status`）与 `DocumentBatchResponse`（`job_id`）——并将空的 `{kb}:rebuild` 空操作（空白 `job_id`）视为已完成。

## 持久化内容

该 MVP 有意覆盖所有面向生产的存储层：

- PostgreSQL / 元数据后端：KB 记录、激活配置版本、文档元数据、任务记录、产物元数据、源哈希、解析 / 索引哈希以及幂等键。
- MinIO / S3 对象存储：当 `.env` 中启用了对象存储时，用于保存上传的源文件与解析产物。
- Milvus / 向量后端：在 KG 与向量索引构建期间生成的 chunk / entity / relation embedding。
- LightRAG 工作区存储：位于配置的 `WORKING_DIR` 下、按 KB 划分的工作区数据，包括 LightRAG 引擎所使用的 KV / doc-status / graph / cache 结构。
- 本地输入 / 缓存路径：当后端使用本地暂存或解析产物时，位于配置的输入 / 缓存目录中的服务端文件。

## 集成演练覆盖范围与局限

该脚本是一次面向**真实后端**的端到端集成演练：它驱动真实运行的 API server（非测试桩 / FakeRAG），按 `.env` 实际连接的后端落数据。因此它能正面验证的范围取决于服务端 `.env` 的存储后端配置。

当前 `.env` 档位（2026-06-04 起）下，演练实测覆盖的**外部后端**（见运行报告 `env_snapshot`）：

- ✅ PostgreSQL（控制面）—— KB 元数据（`LIGHTRAG_KB_METADATA_BACKEND=postgres`）。
- ✅ Milvus —— chunk / entity / relation 向量（`LIGHTRAG_VECTOR_STORAGE=MilvusVectorDBStorage`）。
- ✅ MinIO / S3 —— 源文件与解析产物对象（`LIGHTRAG_OBJECT_STORAGE=minio`）。
- ✅ 真实 LightRAG 引擎 + 真实 MinerU / LLM / embedding / rerank 全链路。
- 🆕 PostgreSQL（引擎）—— KV + doc_status（`LIGHTRAG_KV_STORAGE=PGKVStorage` / `LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage`），2026-06-04 由 Json 文件后端切换为外部 PG。
- 🆕 Neo4j —— 知识图谱（`LIGHTRAG_GRAPH_STORAGE=Neo4JStorage`），2026-06-04 由 NetworkX 文件后端切换为外部 Neo4j；KB 间靠 workspace 节点 label 隔离。

🆕 标记项的前置条件与确认（首次以新 `.env` 跑前务必处理）：

- **重建数据**：后端切换不迁移旧数据——原 `rag_storage/` 下 NetworkX/Json 文件对新后端不可见，需对每个 KB 走一次完整重建（re-ingest / `:rebuild`）写入 Neo4j + PG。
- **安装驱动**：Neo4j 驱动为可选依赖——先 `uv pip install "neo4j>=5,<7"`（或 `uv sync --extra offline-storage`），否则服务启动失败。
- **确认范围**：「✅」三项（PG 控制面 / Milvus / MinIO）已由历史运行确认；「🆕」两项（引擎 PG、Neo4j）的硬删除清理一致性与隔离需以新 `.env` 完整重建一次本演练确认——届时下方的「硬删除清理一致性」与「隔离」断言会在 Neo4j + PG 上实际生效。

> 若改用其它后端（Redis / MongoDB / Qdrant 等），或改回文件型（`JsonKVStorage` / `NetworkXStorage`），需切到对应档位后另行演练。

演练内置的 pass/fail 断言（任一不满足即非零退出）：

- 隔离：主 KB 与空白对照 KB 的文档 id 无重叠；空白对照 KB 确无文档，且对其查询返回**零 references**（共享向量 / 图后端尊重 workspace 边界的正面证明）。
- 对象持久化：每个 ready 文档的 `metadata.source_object_uri` 必须存在（源文件确已落 MinIO / S3）。
- 硬删除清理一致性：当 `--reset-kb` 实际执行后，重建同名 KB 时复用 workspace 的文档数 / 图节点数 / 图边数必须为 0（正面回归「硬删除残留 + workspace 复用读到旧数据」缺陷）。
- 各 ingest / parse / build / reindex / replace 步骤失败即 `raise` 并终止。

## 按 KB 持久化参数

可以。脚本会通过 `POST /kbs/{kb_id}/configs` 创建一个 KB 配置版本，再通过 `POST /kbs/{kb_id}/configs/{config_id}:activate` 将其激活。该配置快照只包含运行时支持的、按 KB 维度生效的默认项：解析器引擎 / 选项、chunk 大小 / overlap、embedding 模型 / 维度、LLM 角色设置、查询限制，以及 rerank 设置。部署级基础设施设置不会被写入 KB 配置版本，而是记录在运行报告（`env_snapshot` / health 输出）中，用于审计。

重要示例：

- Chunk 设置保存在 `chunk_config.chunk_size` 与 `chunk_config.chunk_overlap_size` 中。
- 解析器默认值保存在 `parser_config.engine` 与 `parser_config.process_options` 中。MinerU / Docling 端点、token、服务模式、worker、timeout 等属于部署级 `.env` 设置，由运行中的服务统一管理，而不是按 KB 配置字段保存。
- Embedding 设置保存在 `embedding_config` 中；如果在数据已经建立索引后更改 embedding 模型或维度，仍然需要重建或清理不兼容的向量数据。
- 查询默认值保存在 `query_config` 中，同时 demo 请求也会显式传入这些参数，以便报告准确记录实际使用值。
- 存储后端、对象存储 endpoint / bucket、向量数据库 URI 和元数据后端都属于部署级设置。它们会出现在报告中的脱敏环境快照 / 健康检查信息中，便于追溯，但若要切换它们，需要通过服务端配置和兼容的数据迁移 / 重建来完成，而不是切换某个 KB 配置版本。

## 来自“模拟文件”的增量更新

默认流程使用 `POST /kbs/{kb_id}/documents:sync`，并启用 `auto_parse=true&auto_index=true`。每次运行时，脚本都会递归扫描源目录，基于“模拟文件”目录下的相对路径生成稳定的 `source_key`：`enterprise-demo/<relative path under 模拟文件>`，然后上传当前文件集合。

增量行为如下：

- 新文件：没有任何现有文档使用该 `source_key`，因此会创建新文档，并执行解析、embedding 与索引。
- 未变化文件：`source_key` 相同且内容哈希相同，因此会跳过源文件替换；如果解析 / 索引哈希也匹配，则 parse / build 也会跳过。
- 同一路径下内容变化的文件：`source_key` 相同，但内容哈希不同，因此会原位替换现有文档并重建。
- 解析器 / chunk / 配置发生变化：即使源字节不变，解析 / 索引哈希也可能改变；服务端会按需重新解析或重建。
- 本地已删除文件：sync 不是镜像删除。新请求中缺失的文档会继续保留在 KB 中，直到你显式调用 delete 或 batch-delete。
- 文件重命名 / 移动：相对路径变化，`source_key` 也会变化；服务端会将其视为新文档，若不希望保留旧文档，则需要显式删除旧文档。

对于重复的增量运行，通常应省略 `--run-id`，让脚本自动生成新的幂等键。如果在文件集合或内容已发生变化后仍复用相同的 `--run-id`，API 会正确返回幂等冲突。

## 交互式删除测试

添加 `--delete-test` 后，脚本会在完成文档 / 产物 / 图检查之后、执行查询之前暂停。此时脚本会列出当前 KB 中的文档并编号，询问是否删除，支持多选；随后会调用删除 API，等待删除任务完成，并在报告中记录 `documents_after_delete` 与 `graph_after_delete`。

示例：

```powershell
$env:LIGHTRAG_API_KEY = "sk-123456"
uv run python examples/enterprise_kb_mvp/enterprise_kb_mvp_demo.py `
  --server "http://127.0.0.1:9621" `
  --source-dir "E:/pycharmprojects/RAG/LightRAG/模拟文件" `
  --kb-id enterprise_mvp_demo `
  --delete-test
```

提示符中的选择语法：

- `1`：删除一个文档。
- `1,3,5`：删除多个文档。
- `2-4`：删除一个范围。
- `all`：删除当前列出的全部文档。
- 空输入 / `none` / `cancel`：取消删除。

删除相关标志：

- 默认删除会在后端允许的情况下保留源对象 / 文件和解析产物；它会将该文档从 KB 元数据和 LightRAG 索引中移除。
- `--delete-source-file`：同时删除原始上传的源对象 / 文件。
- `--delete-artifacts`：同时删除解析产物。
- `--delete-llm-cache`：清除相关 LLM 缓存项。
- `--delete-strategy safe`：默认图清理模式。API 还支持 `rebuild_doc_scope`、`rebuild_kb` 和 `rebuild_subgraph`。

报告会以 UTF-8 JSON 形式写入 `examples/enterprise_kb_mvp/runs/`。这些报告被 git 忽略，但其中会包含源路径、对象 URI、产物元数据以及查询输出，因此应将其视为本地运维产物。
