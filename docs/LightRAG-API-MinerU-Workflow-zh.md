# LightRAG API：MinerU 文档入库、知识图谱构建与问答串联指南

本文面向“源码拉取 + uv/虚拟环境 + 已在 `.env` 配好本地/远端服务”的部署方式，说明如何用 **REST API** 串起完整链路：

```text
PDF/Office/图片上传
  → MinerU 解析为 LightRAG Document/sidecar
  → 可选 VLM 图片/表格/公式分析
  → 分块
  → 实体/关系抽取
  → 图存储 + 向量存储写入
  → 结构化检索 / 图谱查看 / 基于知识库问答
```

对应的可执行 smoke 脚本已放在：

```bash
uv run python scripts/run_lightrag_api_workflow.py --help
```

该脚本会上传一个带 MinerU 文件名 hint 的 PDF，轮询异步处理状态，随后测试文档状态、图谱、结构化检索、普通问答、流式问答和 Ollama 兼容发现接口，并把结果写入 `temp/api_mineru_workflow_report.json`。

---

## 1. 运行前环境

### 1.1 使用 uv 和项目虚拟环境

本项目推荐用 `uv` 管理环境和运行脚本。源码安装方式通常如下：

```bash
# 在项目根目录执行
uv sync --extra api --extra offline --extra test

# 运行脚本时无需手动 activate；uv 会使用项目虚拟环境
uv run python scripts/run_mineru_pdf_pipeline.py
uv run python scripts/run_lightrag_api_workflow.py --help
```

如果只缺少某个 Python 包，优先用：

```bash
uv add <package-name>
```

不要在同一个环境里混用裸 `pip install`，避免 `uv.lock` 和虚拟环境不一致。

### 1.2 `.env` 必须在启动目录

`lightrag-server` 启动时会从当前工作目录加载 `.env`，所以请在项目根目录启动服务。当前部署至少应覆盖这些类别：

| 类别 | 关键变量 |
| --- | --- |
| API 服务 | `HOST`、`PORT`、可选 `LIGHTRAG_API_KEY` / `AUTH_ACCOUNTS` |
| LLM | `LLM_BINDING`、`LLM_MODEL`、`LLM_BINDING_HOST`、`LLM_BINDING_API_KEY` |
| 角色模型 | 可选 `EXTRACT_*`、`KEYWORD_*`、`QUERY_*`、`VLM_*` |
| Embedding | `EMBEDDING_BINDING`、`EMBEDDING_MODEL`、`EMBEDDING_BINDING_HOST`、`EMBEDDING_DIM` |
| Rerank | `RERANK_BINDING`、`RERANK_MODEL`、`RERANK_BINDING_HOST` |
| 存储 | `LIGHTRAG_KV_STORAGE`、`LIGHTRAG_VECTOR_STORAGE`、`LIGHTRAG_GRAPH_STORAGE`、`LIGHTRAG_DOC_STATUS_STORAGE`、如 `MILVUS_URI` |
| MinerU | `MINERU_API_MODE`、`MINERU_LOCAL_ENDPOINT` 或 `MINERU_API_TOKEN`、可选 `MINERU_LOCAL_BACKEND` / `MINERU_VLM_URL` |
| 文件流水线 | `LIGHTRAG_PARSER`、`VLM_PROCESS_ENABLE`、`MAX_PARALLEL_PARSE_MINERU`、`MAX_PARALLEL_INSERT` |

推荐的 MinerU + 多模态 PDF 配置示例：

```bash
LIGHTRAG_PARSER=*:native-iteP,*:mineru-iteP,*:legacy-R
VLM_PROCESS_ENABLE=true
MINERU_API_MODE=local
MINERU_LOCAL_ENDPOINT=http://localhost:8000
MAX_PARALLEL_PARSE_MINERU=1
MAX_PARALLEL_INSERT=2
```

> 注意：`MINERU_LOCAL_ENDPOINT` 必须指向 LightRAG MinerU adapter 期望的 mineru-api/router 服务，即提供 `POST /tasks`、`GET /tasks/{task_id}`、`GET /tasks/{task_id}/result` 的服务；不要直接填 OpenAI-compatible/vLLM 模型接口。

### 1.3 启动服务

已有服务全部就绪后启动 LightRAG API：

```bash
uv run lightrag-server --host 127.0.0.1 --port 9621
```

也可以让 smoke 脚本临时启动并在测试结束后关闭：

```bash
uv run python scripts/run_lightrag_api_workflow.py --start-server
```

---

## 2. 一键端到端 smoke 测试

### 2.1 默认测试

默认会从 `test_pdf/` 选择第一个 PDF，上传时把 HTTP multipart 文件名改成 `api-smoke-时间戳-原文件名.[mineru-iteP].pdf`，从而强制走 MinerU + `iteP` 处理选项，且避免和已有同名文档冲突。

```bash
uv run python scripts/run_lightrag_api_workflow.py
```

如果服务没有启动：

```bash
uv run python scripts/run_lightrag_api_workflow.py --start-server
```

如果已有 Milvus collection 来自旧 embedding 维度，建议给 smoke 测试使用隔离 workspace，避免触碰默认知识库：

```bash
uv run python scripts/run_lightrag_api_workflow.py --start-server --workspace api_smoke
```

在 Windows 中文系统里，若手动启动 `lightrag-server` 遇到 splash 输出 emoji 导致的 `UnicodeEncodeError: 'gbk' codec can't encode character`，先在当前 PowerShell 设置 UTF-8 后再启动：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
uv run lightrag-server --host 127.0.0.1 --port 9621
```

本地 OpenAI-compatible 模型服务如果不校验 key，但 SDK 仍要求 `OPENAI_API_KEY`，可同时设置占位值：

```powershell
$env:OPENAI_API_KEY="not_needed"
```

`--start-server` 模式已自动为子进程设置这些兼容变量。

指定 PDF：

```bash
uv run python scripts/run_lightrag_api_workflow.py \
  --file "test_pdf/低共熔溶剂在有机合成和萃取分离中的应用进展.pdf"
```

只测试已有知识库，不上传新文档：

```bash
uv run python scripts/run_lightrag_api_workflow.py --skip-upload
```

### 2.2 常用参数

| 参数 | 作用 |
| --- | --- |
| `--start-server` | 启动 `lightrag-server`，测试结束后自动关闭 |
| `--workspace api_smoke` | `--start-server` 时传给 server 的 workspace；用于隔离 Milvus collection 和文件存储 |
| `--server-log temp/lightrag_api_smoke_server.log` | `--start-server` 的服务端日志位置 |
| `--base-url http://127.0.0.1:9621` | 指定 API 地址，默认从 `.env` 的 `PORT` 组装 |
| `--api-key xxx` | 用 `X-API-Key` 调受保护接口；默认读取 `LIGHTRAG_API_KEY` |
| `--username/--password` | 没有 API key 时可用账号密码登录 JWT；默认读 `LIGHTRAG_USERNAME` / `LIGHTRAG_PASSWORD` |
| `--process-options iteP` | 写入上传文件名 hint：`[mineru-iteP]` |
| `--query "..."` | 入库后用于 `/query/data`、`/query`、`/query/stream` 的问题 |
| `--mode mix` | 查询模式：`local`、`global`、`hybrid`、`naive`、`mix`、`bypass` |
| `--include-chunk-content` | 让引用返回原始 chunk 内容，便于 RAGAS/调试 |
| `--pipeline-timeout-seconds 1800` | 等待文档处理完成的最长时间 |
| `--report-path temp/report.json` | JSON 报告输出位置 |

### 2.3 报告内容

`temp/api_mineru_workflow_report.json` 会包含：

- `/health` 返回的核心配置快照：LLM、Embedding、Storage、Parser、MinerU、VLM、Rerank、pipeline 状态。
- `/documents/upload` 返回的 `track_id` 和实际上传文件名。
- `/documents/track_status/{track_id}` 最终状态、doc_id、chunks_count、错误信息。
- `/documents/status_counts`、`/documents/pipeline_status`、`/documents/paginated` 的结果。
- `/graph/label/popular`、`/graph/label/list`、`/graph/entity/exists`、`/graph/label/search`、`/graphs` 的图谱探测结果。
- `/query/data` 的实体、关系、chunk、references 数量和首条样例。
- `/query` 的回答预览和引用。
- `/query/stream` 的 NDJSON 前几行。
- `/api/version`、`/api/tags`、`/api/ps` 的 Ollama 兼容发现结果。

退出码约定：

| 退出码 | 含义 |
| --- | --- |
| `0` | 主链路完成 |
| `1` | 脚本/网络/接口错误 |
| `2` | 上传成功但至少一个文档最终不是 `processed` |
| `130` | 用户中断 |

---

## 3. API 端点介绍：输入、输出、作用

下面路径均以默认服务地址 `http://127.0.0.1:9621` 为例。若设置了 `LIGHTRAG_API_PREFIX` 或反向代理前缀，请在路径前加对应前缀。

鉴权规则：

- 配了 `LIGHTRAG_API_KEY`：请求头加 `X-API-Key: <key>`。
- 配了 `AUTH_ACCOUNTS`：先 `POST /login` 获取 `access_token`，之后加 `Authorization: Bearer <token>`。
- 两者都没配：本地开发通常可直接访问；`/auth-status` 会说明当前模式。

### 3.1 健康与鉴权

| 方法 | 路径 | 输入 | 输出 | 作用 |
| --- | --- | --- | --- | --- |
| `GET` | `/auth-status` | 无 | `auth_configured`、可选 guest token、版本、WebUI 标题 | 判断是否启用账号认证 |
| `POST` | `/login` | form: `username`、`password` | `access_token`、`token_type`、`auth_mode` | 登录获取 JWT |
| `GET` | `/health` | 无 | `status`、`working_directory`、`input_directory`、`configuration`、pipeline flags | 总体健康检查和配置核对 |

`/health` 是排障第一入口，重点看：

- `configuration.parser_routing` 是否包含 `mineru`。
- `configuration.mineru.endpoint/api_mode/options` 是否符合 `.env`。
- `configuration.vlm_process_enable` 是否为期望值。
- `configuration.embedding_model/embedding_binding_host` 和向量库维度是否与已有数据一致。
- `pipeline_busy`、`pipeline_scanning`、`pipeline_pending_enqueues` 是否卡住。

### 3.2 文档上传/插入/扫描

| 方法 | 路径 | 输入 | 输出 | 作用 |
| --- | --- | --- | --- | --- |
| `POST` | `/documents/upload` | `multipart/form-data`，字段 `file` | `{status,message,track_id}` | 上传文件到 input dir 并后台入队处理 |
| `POST` | `/documents/text` | JSON: `{text,file_source}` | `{status,message,track_id}` | 插入一段已解析文本；不走 MinerU |
| `POST` | `/documents/texts` | JSON: `{texts:[...],file_sources:[...]}` | `{status,message,track_id}` | 批量插入纯文本；不走 MinerU |
| `POST` | `/documents/scan` | 无 | `{status,message,track_id}` | 扫描 input dir 中已有文件并入队 |
| `POST` | `/documents/reprocess_failed` | 无 | `{status,message,track_id:""}` | 重跑 failed/pending/异常中断文档 |
| `POST` | `/documents/cancel_pipeline` | 无 | `{status,message}` | 请求取消正在运行的 pipeline |

上传 PDF 并强制 MinerU：

```bash
curl.exe -X POST "http://127.0.0.1:9621/documents/upload" ^
  -H "X-API-Key: %LIGHTRAG_API_KEY%" ^
  -F "file=@test_pdf/demo.pdf;filename=demo.[mineru-iteP].pdf"
```

返回：

```json
{
  "status": "success",
  "message": "File 'demo.[mineru-iteP].pdf' uploaded successfully. Processing will continue in background.",
  "track_id": "upload_20260526_101010_ab12cd"
}
```

重要行为：

- `/documents/upload` 和 `/documents/scan` 会读取文件名 hint 与 `LIGHTRAG_PARSER`。
- `/documents/text`、`/documents/texts` 是调用方已经给出纯文本，不触发 MinerU/Docling 文件解析。
- 同 canonical basename 的文件会返回 HTTP `409`；要重传同名文件，应先删除旧 doc。
- 内容重复可能在后台处理阶段才发现，此时最终 `track_status` 里文档会是 `failed` 并带 `error_msg`。

### 3.3 文档状态与流水线观测

| 方法 | 路径 | 输入 | 输出 | 作用 |
| --- | --- | --- | --- | --- |
| `GET` | `/documents/track_status/{track_id}` | path: `track_id` | `{track_id,documents,total_count,status_summary}` | 按上传/插入返回的 track_id 查进度 |
| `GET` | `/documents/pipeline_status` | 无 | `busy`、`job_name`、`docs`、`cur_batch`、`latest_message`、`history_messages`、`update_status` | 查 pipeline 当前状态 |
| `GET` | `/documents/status_counts` | 无 | `{status_counts:{...}}` | 查各状态文档数量 |
| `POST` | `/documents/paginated` | JSON 分页过滤条件 | `{documents,pagination,status_counts}` | 列出文档和 doc_id |
| `GET` | `/documents` | 无 | `{statuses:{...}}` | 旧接口，最多返回 1000 条；建议用 paginated |

状态流转：

```text
pending → parsing → analyzing → processing → processed
                                      └────→ failed
```

`preprocessed` 是兼容旧版本的状态。实际 JSON 中状态通常是小写枚举值；前端或脚本最好大小写不敏感处理。

轮询示例：

```bash
curl.exe "http://127.0.0.1:9621/documents/track_status/upload_..." ^
  -H "X-API-Key: %LIGHTRAG_API_KEY%"
```

### 3.4 查询与问答

| 方法 | 路径 | 输入 | 输出 | 作用 |
| --- | --- | --- | --- | --- |
| `POST` | `/query/data` | `QueryRequest` JSON | `{status,message,data,metadata}` | 只返回结构化检索结果，不生成最终回答 |
| `POST` | `/query` | `QueryRequest` JSON | `{response,references}` | 非流式 RAG 问答 |
| `POST` | `/query/stream` | `QueryRequest` JSON | `application/x-ndjson` | 流式或单行 NDJSON 问答 |

`QueryRequest` 常用字段：

| 字段 | 说明 |
| --- | --- |
| `query` | 问题，至少 3 个字符 |
| `mode` | `local`、`global`、`hybrid`、`naive`、`mix`、`bypass`；推荐 `mix` |
| `top_k` | local/global/hybrid/mix 中实体或关系召回数量 |
| `chunk_top_k` | 初始 chunk 召回和 rerank 后保留数量 |
| `enable_rerank` | 是否为本次查询启用 rerank；未配置 reranker 时会降级 |
| `include_references` | 是否返回引用，默认 true |
| `include_chunk_content` | 是否把引用对应 chunk 原文也返回 |
| `conversation_history` | 只给最终 LLM 作为对话上下文，不参与检索 |
| `hl_keywords` / `ll_keywords` | 可手工传高/低层关键词，跳过关键词 LLM 生成 |

结构化检索示例：

```bash
curl.exe -X POST "http://127.0.0.1:9621/query/data" ^
  -H "Content-Type: application/json" ^
  -H "X-API-Key: %LIGHTRAG_API_KEY%" ^
  -d "{\"query\":\"文档讨论了哪些关键实体和关系？\",\"mode\":\"mix\",\"top_k\":10,\"chunk_top_k\":5}"
```

普通问答示例：

```bash
curl.exe -X POST "http://127.0.0.1:9621/query" ^
  -H "Content-Type: application/json" ^
  -H "X-API-Key: %LIGHTRAG_API_KEY%" ^
  -d "{\"query\":\"请总结知识库中文档的核心观点\",\"mode\":\"mix\",\"include_references\":true}"
```

### 3.5 知识图谱查看与维护

| 方法 | 路径 | 输入 | 输出 | 作用 |
| --- | --- | --- | --- | --- |
| `GET` | `/graph/label/list` | 无 | `string[]` | 全量实体 label 列表 |
| `GET` | `/graph/label/popular?limit=300` | query: `limit` | `string[]` | 按连接度返回热门实体 |
| `GET` | `/graph/label/search?q=...&limit=50` | query: `q`,`limit` | `string[]` | 模糊搜索实体 |
| `GET` | `/graphs?label=...&max_depth=3&max_nodes=1000` | query | 子图数据 | 从实体出发取连通子图 |
| `GET` | `/graph/entity/exists?name=...` | query: `name` | `{exists:bool}` | 判断实体是否存在 |
| `POST` | `/graph/entity/create` | `{entity_name,entity_data}` | `{status,message,data}` | 手工创建实体并写实体向量 |
| `POST` | `/graph/relation/create` | `{source_entity,target_entity,relation_data}` | `{status,message,data}` | 手工创建关系并写关系向量 |
| `POST` | `/graph/entity/edit` | `{entity_name,updated_data,allow_rename,allow_merge}` | `{status,message,data,operation_summary}` | 编辑/重命名/合并实体 |
| `POST` | `/graph/relation/edit` | `{source_id,target_id,updated_data}` | `{status,message,data}` | 编辑关系 |
| `POST` | `/graph/entities/merge` | `{entities_to_change,entity_to_change_into}` | `{status,message,data}` | 合并重复实体 |
| `DELETE` | `/documents/delete_entity` | `{entity_name}` | `DeletionResult` | 删除实体及关系 |
| `DELETE` | `/documents/delete_relation` | `{source_entity,target_entity}` | `DeletionResult` | 删除关系 |

图谱检查顺序建议：

```text
/documents/track_status/{track_id} 确认 processed
  → /graph/label/popular 确认有实体
  → /graph/entity/exists?name=<entity>
  → /graphs?label=<entity>&max_depth=2&max_nodes=80
  → /query/data 查看 entities/relationships/chunks/references 是否命中
```

### 3.6 删除与清理

| 方法 | 路径 | 输入 | 输出 | 作用 |
| --- | --- | --- | --- | --- |
| `DELETE` | `/documents/delete_document` | `{doc_ids:[...],delete_file:false,delete_llm_cache:false}` | `{status,message,doc_id}` | 后台删除指定文档、chunks、向量和相关图谱数据 |
| `DELETE` | `/documents` | 无 | `{status,message}` | 清空所有文档、实体、关系、向量、doc_status，并删除 input dir 文件 |
| `POST` | `/documents/clear_cache` | `{}` | `{status,message}` | 清空 LLM cache |

这些是破坏性接口。正式数据上建议先用 `/documents/paginated` 找准 `doc_id`，再删除。

### 3.7 Ollama 兼容 API

LightRAG 还挂载了 Ollama-compatible 路由，前缀是 `/api`：

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `GET` | `/api/version` | 返回模拟 Ollama 版本 |
| `GET` | `/api/tags` | 返回可用的模拟模型名 |
| `GET` | `/api/ps` | 返回模拟运行模型 |
| `POST` | `/api/generate` | 兼容 Ollama generate；主要直通底层 query LLM |
| `POST` | `/api/chat` | 兼容 Ollama chat；用户问题会按前缀/默认模式走 LightRAG 查询 |

这组接口主要给 Open WebUI 等 Ollama 客户端集成；自研系统优先用 `/query`、`/query/stream`、`/query/data`，因为字段更完整。

---

## 4. 文件名 hint 与处理选项

`LIGHTRAG_PARSER` 是全局默认规则，文件名 hint 是单文件覆盖规则。API 上传时可以通过 multipart 的 filename 指定 hint：

```text
paper.[mineru-iteP].pdf
memo.[native-R!].docx
notes.[-R].md
```

处理选项：

| 选项 | 含义 |
| --- | --- |
| `i` | 对图片/绘图 sidecar 调 VLM 分析 |
| `t` | 对表格 sidecar 调 VLM 分析 |
| `e` | 对公式 sidecar 调 VLM 分析 |
| `!` | 跳过实体/关系抽取和图写入，但仍保存 chunk 向量 |
| `F` | 固定 token 分块，legacy 行为 |
| `R` | 递归字符分块 |
| `V` | 语义向量分块，超长 chunk 再用 `R` |
| `P` | LightRAG Document 段落语义分块，缺结构化内容时回退 `R` |

常见组合：

- `mineru-iteP`：PDF 走 MinerU，图片/表格/公式分析，段落语义分块，构建 KG。
- `mineru-R`：PDF 走 MinerU，递归分块，构建 KG。
- `mineru-P!`：PDF 走 MinerU，段落分块，但跳过 KG，仅做向量检索。

---

## 5. 推荐串联流程

### 5.1 生产 API 调用顺序

```text
1. GET /health
   - 验证 LLM/Embedding/MinerU/Storage/VLM/Rerank 配置。

2. POST /documents/upload
   - 上传文件，filename 带 [mineru-iteP] hint。
   - 保存返回 track_id。

3. GET /documents/track_status/{track_id}
   - 轮询到所有文档进入 processed 或 failed。
   - 如果 failed，读取 error_msg 和 metadata。

4. GET /documents/pipeline_status
   - 若长时间未完成，看 latest_message/history_messages。

5. GET /graph/label/popular
   - 验证实体关系抽取和图写入是否产生实体。

6. POST /query/data
   - 不生成最终回答，只看 entities/relationships/chunks/references。
   - 用它调试召回质量。

7. POST /query 或 POST /query/stream
   - 面向用户返回最终答案。
   - include_references=true 用于溯源。
```

### 5.2 Python 调用示例

完整可运行版本见 `scripts/run_lightrag_api_workflow.py`。核心代码形态如下：

```python
import httpx

base_url = "http://127.0.0.1:9621"
headers = {"X-API-Key": "your-key"}

with open("test_pdf/demo.pdf", "rb") as f:
    files = {"file": ("demo.[mineru-iteP].pdf", f, "application/pdf")}
    upload = httpx.post(f"{base_url}/documents/upload", headers=headers, files=files).json()

track_id = upload["track_id"]
status = httpx.get(f"{base_url}/documents/track_status/{track_id}", headers=headers).json()

query = httpx.post(
    f"{base_url}/query",
    headers={**headers, "Content-Type": "application/json"},
    json={"query": "请总结这批文档", "mode": "mix", "include_references": True},
).json()
```

---

## 6. 常见问题与处理

### 6.1 `GET /health` 连接失败

- 确认服务已启动：`uv run lightrag-server --host 127.0.0.1 --port 9621`。
- Windows 中文环境如果启动日志里出现 `UnicodeEncodeError: 'gbk' codec can't encode character`，设置 `PYTHONUTF8=1` 和 `PYTHONIOENCODING=utf-8` 后重启；或直接用 smoke 脚本的 `--start-server`。
- 如果日志出现 `[Errno 10048] ... address ('127.0.0.1', 9621)`，说明端口已被旧服务占用。先停止旧 `lightrag-server`，或换一个 `--port`。
- 确认端口与 `.env` 的 `PORT` 一致。
- 如果用 Docker/WSL/远程主机，`127.0.0.1` 可能不是同一网络命名空间。

### 6.2 HTTP 403 `API Key required`

- 给请求加 `X-API-Key`。
- 或运行脚本时传 `--api-key`。
- 如果使用账号认证，先 `/login` 拿 JWT，并加 `Authorization: Bearer ...`。

### 6.3 上传返回 HTTP 409

同 canonical basename 的文档已存在。解决方式：

1. 用 `/documents/paginated` 找到旧 `doc_id`。
2. 调 `/documents/delete_document` 删除。
3. 或者上传时换一个唯一 filename。smoke 脚本默认会加时间戳避免这个问题。

### 6.4 一直卡在 `parsing`

优先检查 MinerU：

- `MINERU_API_MODE=local` 时 `MINERU_LOCAL_ENDPOINT` 是否能访问。
- endpoint 是否提供 `/tasks` API，而不是 OpenAI `/v1/chat/completions` 类接口。
- `MAX_PARALLEL_PARSE_MINERU=1` 对单 GPU 更稳。
- 查看服务端日志和 `/documents/pipeline_status.latest_message`。

### 6.5 文档 `processed` 但图谱为空

可能原因：

- 文件 hint 或 `LIGHTRAG_PARSER` 包含 `!`，跳过 KG。
- LLM 抽取失败但被缓存/重试逻辑掩盖，需要查 `history_messages` 和服务端日志。
- 文档内容太少或没有可抽取实体。
- `ENTITY_EXTRACTION_USE_JSON`、实体类型 prompt 或抽取模型能力不匹配。

可用 `/query/data` 判断是否至少召回 chunks；如果 chunks 有、entities/relationships 空，问题在实体关系抽取或 KG 写入。

### 6.6 VLM 分析失败：`VLM call failed: 'OPENAI_API_KEY'`

这说明 MinerU 解析已经完成，失败发生在 `i/t/e` 触发的多模态 VLM 分析阶段。OpenAI-compatible SDK 即使访问本地服务，有时也要求环境里存在 `OPENAI_API_KEY`。

处理方式：

```powershell
$env:OPENAI_API_KEY="not_needed"
uv run lightrag-server --host 127.0.0.1 --port 9621 --workspace api_smoke
```

或把真实/占位 key 写入 `.env` 的 `LLM_BINDING_API_KEY` / `VLM_LLM_BINDING_API_KEY` / `OPENAI_API_KEY`。如果暂时不需要图片、表格、公式 VLM 分析，也可以把上传 hint 从 `mineru-iteP` 改成 `mineru-P` 或把 `VLM_PROCESS_ENABLE=false` 后重启。

### 6.7 更换 Embedding 模型后异常

Embedding 模型、维度、非对称前缀一旦改变，旧向量空间就不兼容。必须清空对应 workspace/向量数据并重新索引；否则查询质量或向量库写入可能异常。

如果启动日志出现 `Vector dimension mismatch for collection 'entities': existing=4096, current=1024`，不要直接删除生产 collection。优先选择：

1. 确认 `.env` 的 `EMBEDDING_MODEL` / `EMBEDDING_DIM` 是否应该回到旧维度；或
2. 用新的 `--workspace` 启动测试实例，例如 `uv run python scripts/run_lightrag_api_workflow.py --start-server --workspace api_smoke`；或
3. 备份后按你的数据策略清理旧 workspace/collection，再重新索引。

### 6.8 流式接口如何解析

`/query/stream` 返回 `application/x-ndjson`，每行一个 JSON：

```text
{"references":[...]}
{"response":"第一段"}
{"response":"第二段"}
{"error":"..."}
```

客户端要逐行解析，不要当成一个整体 JSON 数组。

---

## 7. 与 `scripts/run_mineru_pdf_pipeline.py` 的关系

`uv run python scripts/run_mineru_pdf_pipeline.py` 是 **直接调用 Python Core API** 的端到端验证：它构造 `LightRAG` 实例，调用 `apipeline_enqueue_documents(..., parse_engine=mineru, process_options=...)`，再直接查询 graph/vector/query。

`uv run python scripts/run_lightrag_api_workflow.py` 是 **REST API 层验证**：它通过 HTTP 调 `/documents/upload`、`/documents/track_status`、`/query/data`、`/query`、`/graph/*`，更接近真实业务系统集成方式。

两者互补：

- Core 脚本通过：说明 `.env` 中 LLM/Embedding/MinerU/Storage 基础链路可用。
- API 脚本通过：说明 LightRAG Server、鉴权、文件上传、异步后台任务、REST 查询和图谱接口都能串起来。
