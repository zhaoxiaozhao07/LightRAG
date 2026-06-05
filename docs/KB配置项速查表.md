# LightRAG KB 配置项速查表

> 文档版本：2026-06-05
> 适用范围：每个知识库（KB）可单独设置的运行时配置项。
> 权威来源：`lightrag/api/config_version_service.py`（校验/归一/hash/运行时 overlay）。
> 配套文档：`docs/API接口文档.md` §7（Config Versions）、`docs/生产级后端改造设计方案.md` §12。

---

## 0. 怎么用

每个 KB 的配置是**不可变版本快照**：创建一个版本 → 激活 → 下次请求按新配置重建该 KB 的 LightRAG 实例。

```http
POST   /kbs/{kb_id}/configs                      # 创建配置版本（返回 version_id）
POST   /kbs/{kb_id}/configs/{version_id}:activate # 激活（写入 active_config_version_id 并丢弃缓存实例）
POST   /kbs/{kb_id}/configs/{version_id}:diff     # 预测：与当前激活版本比，需不需要 reparse/reindex/vector rebuild
GET    /kbs/{kb_id}/configs                        # 列表
GET    /kbs/{kb_id}/configs/{version_id}           # 详情
```

创建时会**严格校验**，下列情况直接返回 `400`，不会创建版本：
- 任意 section 出现**未知键**（无运行时效果，避免"存了不生效"）；
- **部署级字段**（存储后端、parser 服务实例参数）；
- **非法值**（如该为正整数处传了 0/负数、空字符串等）；
- `llm_role_config` 出现未知角色或未知字段。

> 优先级：**请求体显式传的 query 参数 > KB `query_config` 默认值 > 系统默认**。`parser_engine`/`process_options` 的优先级为：请求 > 文档 metadata 已 snapshot 值 > KB `parser_config` > 文件名/环境变量路由。

---

## 1. 三段 hash 决定"改了之后要做什么"

| Hash | 派生因子 | 改动后的最小动作 |
|---|---|---|
| `parser_hash` | `parser_config`（engine + process_options） | **重新解析 + 重新构建** |
| `index_hash` | `chunk_config` + `embedding_config` + `extraction_config` + `extract`/`vlm` 角色身份 | **仅重新构建索引（复用解析产物）** |
| `query_hash` | `query_config` + `query`/`keyword` 角色身份 | **仅影响查询，不重建** |
| （额外）vector rebuild | `embedding_config.model` 或 `dim` 变化 | 需**重建向量**（维度/语义空间变了） |

`:diff` 会返回 `requires_reparse` / `requires_reindex` / `requires_vector_rebuild` + `reasons`。

---

## 2. 各配置 section 字段

### 2.1 `parser_config` —— 影响 `parser_hash`（改 → 重解析+重建）

| 字段（别名） | 含义 | 取值 |
|---|---|---|
| `engine` / `parser_engine` | 解析引擎 | `mineru` / `docling` / `native`（拒绝 `legacy`） |
| `process_options` / `options` | 文件级处理选项 | 选项串，创建时校验+规范化 |

> ⛔ parser **服务实例级**字段（endpoint/api_key/api_mode/timeout/workers 等）属部署级，**不能**写进 KB config，必须走 `.env`。详见 §4。

### 2.2 `chunk_config` —— 影响 `index_hash`（改 → 仅重建索引）

| 字段（别名） | 含义 | 取值 |
|---|---|---|
| `chunk_token_size` / `chunk_size` | 每块最大 token 数 | 正整数 |
| `chunk_overlap_token_size` / `chunk_overlap_size` / `overlap` | 相邻块重叠 token 数 | 非负整数 |
| `tiktoken_model_name` | 分块所用分词模型名 | 字符串 |

### 2.3 `embedding_config` —— 影响 `index_hash`；`model`/`dim` 改动额外需向量重建

| 字段（别名） | 含义 | 取值 | 改动后动作 |
|---|---|---|---|
| `model` | embedding 模型名 | 字符串 | 重建索引 + **向量重建** |
| `dim` / `embedding_dim` | 向量维度 | 正整数 | 重建索引 + **向量重建** |
| `token_limit` / `max_token_size` | embedding token 上限 | 正整数 | 重建索引 |

### 2.4 `query_config` —— 影响 `query_hash`（改 → 仅影响查询，不重建）

支持**全部 `QueryParam` 字段** + 两个实例级检索旋钮：

| 字段 | 含义 |
|---|---|
| `mode` | 检索模式 `local`/`global`/`hybrid`/`naive`/`mix`/`bypass` |
| `top_k` | KG 实体/关系检索数 |
| `chunk_top_k` | chunk 检索/保留数 |
| `max_entity_tokens` | 实体上下文 token 上限 |
| `max_relation_tokens` | 关系上下文 token 上限 |
| `max_total_tokens` | 总 token 预算 |
| `enable_rerank` | 是否重排 |
| `response_type` | 回答格式描述 |
| `user_prompt` | 附加给 LLM 的指令 |
| `include_references` | 是否带引用 |
| `only_need_context` / `only_need_prompt` | 只返回上下文 / 只返回 prompt |
| `hl_keywords` / `ll_keywords` | 高/低层关键词 |
| `conversation_history` | 对话历史 |
| `stream` | 流式（⚠️ 运行时由路由按端点强制覆盖，作为 KB 默认意义有限） |
| `ids` | 文档允许列表（⚠️ 运行时由路由按 `filters.doc_ids`/enabled/archived 强制覆盖，作为 KB 默认意义有限） |
| `cosine_threshold` | 向量检索余弦阈值（**实例级**：经构造写入向量库，不是 per-request QueryParam） |
| `related_chunk_number` | 检索的相关 chunk 数（**实例级**默认，非 per-request QueryParam） |

> `cosine_threshold` 与 `related_chunk_number` 不是 `QueryParam` 字段，因此**只能作为 KB/实例级默认**，无法逐请求覆盖；改动它们仍会变 `query_hash`（仅影响查询）。

### 2.5 `extraction_config` —— 影响 `index_hash`（改 → 仅重建索引）

| 字段 | 含义 | 取值 |
|---|---|---|
| `language` | 摘要/抽取语言 | 非空字符串 |
| `entity_types` | 实体类型列表（自动渲染成抽取指引） | 非空字符串列表 |
| `entity_types_guidance` | 直接给抽取指引文本（**优先于** `entity_types`） | 非空字符串 |
| `entity_type_prompt_file` | 实体类型 prompt 文件路径 | 非空字符串 |
| `max_gleaning` | 抽取补漏次数 | 非负整数 |
| `max_extraction_records` | 单次响应实体+关系行数上限 | 正整数 |
| `max_extraction_entities` | 单次响应实体行数上限 | 正整数 |
| `force_llm_summary_on_merge` | 合并时强制 LLM 摘要的阈值 | 非负整数 |

> ⚠️ `entity_types` 与 `entity_types_guidance` **二选一**：若同时提供，`entity_types_guidance` 生效、`entity_types` 被忽略（这是有意的优先级设计）。

### 2.6 `llm_role_config` —— 按角色覆盖运行时 LLM

角色名只能是 `extract` / `keyword` / `query` / `vlm`。每个角色可为**字符串**（等价 `{"model": <str>}`）或**对象**：

| 字段（别名） | 含义 | 是否影响 hash |
|---|---|---|
| `model` | 模型名 | `extract`/`vlm` → `index_hash`；`query`/`keyword` → `query_hash` |
| `binding` | provider binding | 同上 |
| `host` | provider host | 同上 |
| `provider_options` | provider 选项（对象） | 同上 |
| `model_kwargs` / `kwargs` | 模型调用参数（对象） | 同上 |
| `api_key` | 密钥 | **不影响任何 hash**（轮换密钥不触发重建） |
| `max_async` | 并发上限 | **不影响 hash**（性能旋钮） |
| `timeout` | 超时 | **不影响 hash**（性能旋钮） |

> 即：改 `extract`/`vlm` 的模型身份会触发 reindex；改 `query`/`keyword` 只影响查询；轮换 `api_key` 或调 `max_async`/`timeout` 不触发任何重建。

---

## 3. 完整示例

```http
POST /kbs/{kb_id}/configs
Content-Type: application/json

{
  "config": {
    "parser_config":    {"engine": "mineru", "process_options": "iF"},
    "chunk_config":     {"chunk_token_size": 512, "chunk_overlap_token_size": 64, "tiktoken_model_name": "gpt-4o"},
    "embedding_config": {"model": "bge-large", "dim": 1024, "max_token_size": 8192},
    "extraction_config":{"language": "Chinese", "entity_types": ["PERSON", "ORG"], "max_gleaning": 1},
    "query_config":     {"mode": "mix", "top_k": 60, "chunk_top_k": 20, "enable_rerank": true,
                          "cosine_threshold": 0.2, "related_chunk_number": 5},
    "llm_role_config":  {"extract": "gpt-4o-mini",
                          "query":   {"model": "gpt-4o", "max_async": 8}}
  },
  "created_by": "alice"
}
```
随后：`POST /kbs/{kb_id}/configs/{version_id}:activate` 生效。

---

## 4. ⛔ 不允许按 KB 配置的部署级字段（创建时 400）

这些必须通过 `.env` / 部署编排统一管理，不能写进 KB config：

- **整个 `storage_config`**：KV / Vector / Graph / DocStatus 存储后端不能按 KB 切换。
- **`parser_config` 内的服务实例级字段**：`endpoint` / `endpoint_url` / `base_url` / `url` / `host`、`api_key` / `api_token` / `token`、`api_mode`、`timeout` / `poll_interval` / `poll_timeout`、`max_concurrency` / `workers`、`mineru_*` / `docling_*` 等。
- `parser_config` 内任何**非白名单键**（只认 `engine`/`parser_engine`/`process_options`/`options`）。

---

## 5. 校验规则与常见 400

| 情况 | 结果 |
|---|---|
| 任意 section 出现未知键（如 `chunk_config.chunk_strategy`、`embedding_config.base64`、`query_config.bogus`） | `400 ... has unsupported keys (no runtime effect): ...` |
| 写入部署级字段（`storage_config` / parser 服务字段） | `400 ... deployment-level ...` |
| 该为正整数处传 0/负数（如 `chunk_token_size`、`related_chunk_number`、`top_k`） | `400 ... must be positive` |
| `llm_role_config` 未知角色或未知字段 | `400 ... unknown role/key` |
| `entity_types` 不是非空字符串列表 / `language` 为空 | `400` |

---

## 6. 注意事项

1. **`bypass` 模式可用但无权限门控**：`bypass`（直通 LLM、绕过检索）功能正常，但当前任何已认证调用方都能用（RBAC 未做）。生产环境建议自行限制谁能用它（避免绕过知识库直连 LLM 消耗 token）。
2. **`stream` / `ids` 作为 KB 默认意义有限**：查询路由会按端点（流式与否）和文档范围（`filters.doc_ids`/enabled/archived）在运行时强制覆盖这两个值，所以把它们写进 `query_config` 默认值基本不起作用。
3. **`cosine_threshold` / `related_chunk_number` 是实例级**：它们不是 per-request `QueryParam` 字段，只能作为 KB 默认，无法逐请求覆盖。
4. **激活不自动重建数据**：`:activate` 只切换配置并丢弃缓存实例；是否需要 reparse/reindex/vector rebuild 由 `:diff` 提示，需你显式触发 `:reindex` / `:rebuild` 等。
5. **未知键现在会报错**：历史上未知键会被静默忽略（"存了不生效"）；自 2026-06-05 起改为创建时 `400`，避免误以为生效。
