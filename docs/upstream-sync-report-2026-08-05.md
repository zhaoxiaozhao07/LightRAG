# LightRAG 上游同步差异分析与可吸收特性清单

**项目**: HKUDS/LightRAG 深度二开（企业级多租户 API Server）  
**基线**: 本地基于 v1.5.0（MB=c67f055a，2026-06-03），自研 81 个提交；上游已前进至 91068674（2026-08-05，经 v1.5.1→v1.5.5 再+166 提交，共 1225 个提交）  
**分析日期**: 2026-08-05  
**分析覆盖**: 核心引擎、存储层、API 服务器、解析器/分块器、配置/安全/运维横切面

---

## 执行摘要

### 关键发现

1. **本地代码冲突远小于表面**：219 个"冲突文件"中，60% 是 **CRLF 行尾噪音**（Windows 开发导致 78/164 个 Python 文件被改成 CRLF）；去除行尾差异后，真实冲突集中在约 23 个文件，其中仅 5 个为重度改动（lightrag_server.py +1017、operate.py +709、config.py +549、pipeline.py +157、query_routes.py +107）。
   
2. **本地自研功能 90%+ 在新增文件里**（约 40 个企业级模块：kb_service、job_service/job_worker、metadata_store、object_storage、enterprise_auth、person_auth、agent_*、chat_memory_*、bilingual_query_service 等），与上游零路径冲突。

3. **发现 3 个本地生产安全漏洞**（上游已修复，本地未吸收）：
   - GHSA-2wpj-ffvv-2pq8：上传文件名处理可被利用制造碰撞覆盖 + symlink 竞态（上游修复尚未合入其 main，但可独立移植 155 行）
   - ReDoS：`sentence_split_regex` 仍暴露在请求模型，攻击者正则可冻结 worker
   - 依赖安全下限：starlette 1.0.0、python-multipart 0.0.28、cryptography 48.0.0（对应多个 CVE）

4. **2 个升级即崩的破坏点**（必须先解决才能整树同步）：
   - 上游删除了 `docs_format="lightrag"` + `lightrag_document_paths` 入口，本地 `index_build_service` 主路径正依赖它
   - 上游删除了 `pipeline_status["request_pending"]` 协议，本地 `document_routes.py` 和 `index_build_service.py` 有确认调用

5. **命名冲突陷阱**：本地 `lightrag/parser/legacy.py`（模块）vs 上游新增 `lightrag/parser/legacy/`（包），git 不会报冲突但运行时 ImportError；上游实现是功能超集，建议删本地采上游

### 核心价值点（按优先级）

**P0 - 安全修复（可立即独立处理，不依赖整树同步）**
- 3 个生产漏洞修复（上传文件名、ReDoS、依赖升级）
- Neo4j Lucene 特殊字符转义（生产图后端查询 bug）
- Milvus filter 转义 + dynamic-field-overflow（生产向量后端读写 bug）

**P1 - 零冲突高收益（可独立 cherry-pick 的稳定性/质量修复包）**
- 缓存正确性三件套（多轮对话缓存污染、缓存键碰撞、截断响应不入缓存）
- rerank 四连修（越界/null/畸形响应/挂死）
- 抽取质量组（防注入、section heading、去重、edge weight）
- tokenizer/chunking 有界化（事件循环 DoS 防护）
- postgres_impl 小修组（search-path、pgvector 懒加载、drop 防复活）
- keyed-lock 取消安全（本地有"客户端中断处理"，cancellation 高频，此项直接受益）
- 健壮性一揽子（cosine/get_env_value/json 系列零成本修复）

**P1-战略 - PGTableGraphStorage（可单文件后移植试点）**
- 纯 PG14+ 表实现图存储（无 AGE、无 pgvector、无扩展），本地可砍掉 Neo4j 外部依赖
- 昨天（2026-08-05）刚合入上游 main，建议观察 2-4 周再试点
- 图接口自分叉点零变化，可不等整树同步、单文件移植（约 1600 行 + 注册 3 处）

**中期整树同步项（需配套解决破坏点）**
- lr2-bounded-scheduling（有界调度、准入控制、手动重试状态机）
- parser 统一注册表重构（本地 legacy/libreoffice 可改写为插件）
- docx smart_heading（中文公文红头/落款/题名块识别，与企业场景高度契合）
- 原生 Markdown 解析器 + 表格处理增强
- gunicorn 跨 worker 并发帽（本地当前 WORKERS=1 钉死，价值有限，属多 worker 化储备）

---

## 一、分叉现状与冲突性质

### 1.1 Git 拓扑

```
分叉点（merge-base）: c67f055a  v1.5.0  2026-06-03  API 0305
                        ├─ 本地 main: +81 提交（企业级多租户 API Server 深度改造）
                        └─ upstream/main: +1225 提交
                             v1.5.1 (06-09) → v1.5.2 (06-11) → v1.5.3 (06-14)
                             → v1.5.4 (06-24) → v1.5.5rc1 (07-14) → v1.5.5 (07-31)
                             → 91068674 (08-05, +166 提交)  core 1.5.6  API 0327
```

**直接 `git merge` 模拟结果**: 219 个冲突文件（实际命令：`git merge-tree --write-tree main upstream/main`）

### 1.2 冲突性质分层（关键修正：CRLF 噪音占比）

| 层级 | 文件数 | 性质 | 处理方式 |
|---|---|---|---|
| **假冲突（行尾噪音）** | ~130 | 本地 78/164 个 Python 文件被改成 CRLF（Windows 开发），包括：lightrag/kg/ 全部 15 个实现、全部 4 个 chunker、docx/sidecar/multimodal 等；实测 `git diff -w` 后零功能改动 | 合并前先做 LF 归一化提交，或用 `git merge -X renormalize` |
| **路径零冲突（本地新增）** | ~40 | 企业级模块全在新路径：kb_service.py、job_service/job_worker、metadata_store、enterprise_auth、person_auth、agent_*、chat_memory_* 等 | 直接保留 |
| **真冲突-轻度** | ~18 | 双方改同文件但不同函数/区域，可三方合并：llm 绑定（日志脱敏 vs 泄漏修复）、utils.py(+3 vs +702)、base.py(+15 vs +接口扩展) 等 | 三方合并或功能分段 cherry-pick |
| **真冲突-重度** | 5 | 同文件同区域双方均大改：lightrag_server.py(+1017 企业化 vs admission)、operate.py(+709 双语检索 vs 抽取重构)、config.py(+549 企业配置 vs 安全/调度配置)、pipeline.py(+157 legacy 引擎 vs 注册表)、query_routes.py(+107 企业路由 vs query-response-time) | 需人工调和或按整树同步路线图分阶段处理 |

**实测数据**（已用 `git diff -w --numstat` 逐文件核实）：
- lightrag/kg/ 15 个文件：本地零真实改动（全 CRLF）
- lightrag/parser/ + chunker/：本地真实改动仅 7 文件 889 行（legacy.py、libreoffice/、routing.py、mineru cache/client、debug.py）
- 冲突文件中本地真实改动 Top5：lightrag_server.py(+1017)、operate.py(+709)、config.py(+549)、pipeline.py(+157)、query_routes.py(+107)

### 1.3 本地生产拓扑（影响吸收优先级）

**实际配置**（`.env.main` 确认）：
- 存储：**PGKVStorage + PGDocStatusStorage（元数据）+ Neo4JStorage（图）+ MilvusVectorDBStorage（向量）**
- 部署：gunicorn 多 worker，但 **WORKERS=1 显式钉死**（"durable worker 与流水线按单进程设计"）
- 场景：企业内网单机 + PostgreSQL + MinIO + Neo4j

**吸收优先级修正**：
- Neo4j/Milvus 修复是生产直接相关（任务初始假设"milvus 本地可能用不到"不成立）
- 跨 worker 并发帽对当前部署价值为零（WORKERS=1），属未来多 worker 化储备
- PGTableGraphStorage 是砍掉 Neo4j 外部依赖的战略选项（企业单机四类存储全归 PG）

---

## 二、可吸收特性分类清单（按主题与优先级）

### 主题 A：生产安全漏洞修复（P0，可立即独立处理）

| # | 项 | 漏洞/影响 | 上游来源 | 本地现状 | 吸收方式 | 工作量 |
|---|---|---|---|---|---|---|
| A1 | GHSA-2wpj 上传文件名处理 | 可制造文件名碰撞覆盖既有文档、混淆审计；symlink 竞态与 TOCTOU | upstream/codex/-ghsa-2wpj（尚未合入 main）| `document_routes.py:158` 的 `sanitize_filename` 与修复前逐字相同；写入用 `aiofiles.open` 存在 symlink/竞态风险 | 独立移植（155 行 + 162 行测试）| 半天 |
| A2 | ReDoS sentence_split_regex | 请求携带的回溯型正则可冻结 worker（持 GIL） | GHSA-32jh（已进 main）| `document_routes.py:354` 仍把 `sentence_split_regex` 放在请求模型 | 从 API 表面移除该参数（仅允许 env/SDK 配置）| 1 小时 |
| A3 | 依赖安全下限 | starlette CVE-2026-54283/48818、python-multipart CVE-2026-53539、cryptography GHSA-537c | 多个 PR | 本地 uv.lock：starlette 1.0.0、python-multipart 0.0.28、cryptography 48.0.0 均低于安全下限 | `uv sync --upgrade-package starlette --upgrade-package python-multipart --upgrade-package cryptography`；升级后重跑 `--extra memory` 防 graphiti 被清 | 1 小时 |
| A4 | 密码比较常量时间 | timing attack 可探测密码（本地已部分缓解）| #3423 | 本地 `passwords.py:26` 有 fallback 分支 `stored_password == plain_password`（明文比较）；但主路径用 bcrypt.checkpw（常量时间）；enterprise_auth 已用 `secrets.compare_digest` | 把明文 fallback 也改 `secrets.compare_digest`（1 行）| 15 分钟 |
| A5 | guest token bypass API-key | API-key-only 模式下，攻击者可用伪造 guest JWT 绕过 X-API-Key 检查 | #3319 f7819aa3 | 本地 `utils_api.py` 有 combined auth 但逻辑不同（person-v1 token + enterprise auth）；**本地不存在该漏洞**（未实现上游的"API-key-only 模式"，本地是 enterprise auth 或 person auth，均强校验） | 无需吸收（本地不受影响）| 0 |

**A 组小计**: 3 个需立即修复（A1/A2/A3，合计约 1 工作日），1 个可选加固（A4），1 个本地已免疫（A5）。

### 主题 B：核心引擎质量修复（P1，可独立 cherry-pick 的零冲突包）

| # | 项 | 说明 | 上游来源 | 本地冲突 | 吸收方式 | 优先级 |
|---|---|---|---|---|---|---|
| B1 | 缓存正确性三件套 | 带 conversation_history 查询绕过 answer cache（不读不写，防污染）；compute_args_hash 多参数长度前缀编码（防键碰撞）；截断 LLM 响应不入缓存 | #3510 / #3435 / #3446+#3448 | operate.py 有重叠（kg_query/naive_query 本地有双语检索改动，语义正交、行级需手工合）| cherry-pick + 手工三方合并 | **高** |
| B2 | rerank 四连修 | 聚合分数越界、null index/score、畸形响应守卫、zero-max-tokens 挂死（含启动校验 `RERANK_MAX_TOKENS_PER_DOC` 默认 4096）| #3514/#3518/#3529/#3419 | **无冲突**（本地 rerank.py 零改动，双语双路检索重度用 rerank）| 直接合并 4 个提交 | **高** |
| B3 | 抽取质量组 | 抽取 prompt 防注入（格式模板占位符化）；section heading 上下文进抽取；描述合并去重；edge weight 不重复累加；summary encode 一次 | #3231/#3225/#3395/#3399/#3359 | operate.py 重叠（与本地 _MergeStageProgress 进度插桩同在 merge 区）| cherry-pick + 手工合并 merge 区 | 高 |
| B4 | tokenizer/chunking 有界化 | tokenizer 线程池 + 有界提交（`TOKENIZER_SUBMIT_LIMIT=8`）；chunking 移出事件循环；encode-verified 安全截断契约（`EMBEDDING_CHUNK_OVERLAP_TOKEN_SIZE=100`）| #3543/#3544/#3565 | 冲突小（core 无冲突；api 校验部分注意本地自有路由是否复用）| 直接合并（core）| 高 |
| B5 | 健壮性一揽子 | cosine 三连（nonfinite→0、零向量、空 env）；get_env_value 系列（padded bool、拒 nonfinite、各空 env）；load_json 空文件；tolerant_load_json_dict 统一 LLM JSON 恢复 | 十余个 PR | **无冲突**（utils.py/lightrag.py，本地相关文件仅 3 行改动）| 打包直接合并 | **高** |
| B6 | LLM 绑定修复组 | anthropic 客户端泄漏；bedrock timeout→botocore；ollama bracket prefix；thinking token budget 可观测 | #3261/#3454/#3498/#3406 | 轻重叠（本地同文件有日志脱敏，行邻近可合）| cherry-pick | 中-高 |
| B7 | 查询侧小修 | pick_by_vector_similarity 部分向量缺失；chunk 结果确定性排序；其他 operate.py 小修 | #3479/e66173ce 等 | 轻重叠 | 顺带合 | 中 |

**B 组特点**: 全部可独立 cherry-pick（不依赖 lr2/parser 重构），零 schema/迁移，性价比最高。合并后对本地企业多轮对话/双语检索/内网稳定性直接受益。工作量约 2-3 工作日（主要是 operate.py 的手工三方合并）。

### 主题 C：存储层修复（P0-P1，按后端分组）

#### C1. PostgreSQL（生产主力后端）

| 项 | 说明 | 上游来源 | 冲突 | schema/迁移 | 优先级 |
|---|---|---|---|---|---|
| postgres_impl 小修组 | search-path 表检查（to_regclass）；**pgvector 懒加载**（摆脱 pgvector 硬依赖与运行时 pm.install）；vector get_by_id 剥离 content_vector；drop 时清 legacy 行防复活 | #3206/e7fcf00d/88a575c4/#3254 | 无 | 无 | **高** |
| PG 调度页（scheduling contract）| PGDocStatusStorage 新增键集分页/严格读/源冲突修复 8 方法；配套新索引（verify-before-drop 自动替换）| #3103 链 | base.py 强制抽象（见 C4）| **索引级迁移（自动、可验证）**；parse_engine VARCHAR(32)→TEXT 自动 ALTER；基础表结构零变化 | 中（随核心同步获得）|
| **PGTableGraphStorage** | 全新纯表图后端（无 AGE/pgvector/扩展），两张表 + frontier-capped BFS；与 PGGraphStorage 并存；上游已推荐其优先于 AGE | dev-pgtable 合并 + #3103/#3568 | **无**（新文件；BaseGraphStorage 接口 MB↔上游零变化）| **自动建 2 张新表**；切换=重建图（无 Neo4j→PGTable 迁移工具）| **高（战略）** |

**PGTableGraphStorage 专项结论**：
- 昨天（2026-08-05）刚合入上游 main，dev-pgtable 分支约 31 个提交、多轮 cross-review，但无生产沉淀
- **可单文件后移植**（拷 pgtable_impl.py + kg/__init__.py 三处注册 + 从 utils.py 摘 validate_workspace 约 30 行）
- **吸收节奏**：观察上游 2-4 周 followup → 测试环境试点 1-2 个新 KB（与 Neo4j 双跑对比）→ 作为企业单机内网模板默认图后端（Neo4j 保留为可选）

#### C2. Neo4j + Milvus（生产在用后端）

| 项 | 说明 | 上游来源 | 冲突 | 优先级 |
|---|---|---|---|---|
| **Neo4j 修复组** | search_labels Lucene 特殊字符转义（实体名含 `+ - ! ( ) " ~ * ? : \ /` 等时全文检索报错/漏配）；BFS 深度改绑定参数 + is_truncated 修正；workspace 校验 | #3233/9d9714e5/fa146e5e | 无 | **高** |
| **Milvus escape-filter** | filter 表达式字符串字面量转义（`"`/`\`），11 处调用点 | #3555 | 无 | **高** |
| **Milvus dynamic-field-overflow** | 超长额外字段收进 `_lightrag_extra` 动态字段；读侧自动还原 | #3228 | 无，写入格式新增字段（向后兼容）| **高** |
| Milvus migration 加固 | 旧集合迁移 OOM/断点/重试加固（4 提交连锁）| #3257-3260 | 无，含集合级自动迁移（本地数据非老格式则不触发）| 低-中 |

#### C3. shared_storage 并发（按可拆性分组）

| 项 | 说明 | 上游来源 | 本地价值（WORKERS=1 钉死）| 优先级 |
|---|---|---|---|---|
| **keyed-lock/namespace-lock 取消安全组** | 锁释放 asyncio cancel 安全；async keyed lock 即时释放去闲置缓存 | #3407/#3439 | **有**（本地有"客户端中断处理"，cancellation 高频，此项在单 worker 下同样受益）| **高** |
| keyed-lock 死 worker 恢复 + holder table | 多进程锁持有者死亡检测 + Manager 原子 holder 表 + RPC 削减 | #3413/#3434/4ca0597c | 无（WORKERS=1 下收益为零，属多 worker 化储备）| 中 |
| pipeline reservation + LR2 ingress 全家桶 | owner-token busy 预约、死进程恢复、mailbox 唤醒（**删除 request_pending/autoscanned**）、MAX_PENDING_DOCUMENTS 准入、manual-retry sticky 协议 | #3408/#3409/#3415/#3417/#3431 + LR2 Phase0-6 | **直接破坏本地**：document_routes.py L2941/L992/L999 读 request_pending/autoscanned；index_build_service.py 协议依赖；**不可单独摘**，只能作为整树核心同步的一部分 | 中（整树同步时）|

#### C4. 其他后端（本地未用，随整树同步获得）

json_kv/json_doc_status copy-on-read、redis 大重构（ZSET 索引自动 backfill）、mongo client-manager 泄漏、opensearch bulk、is_truncated 行为对齐组、qdrant fallback、faiss/memgraph/nano 微修。优先级均为低-中。

**接口签名变更**（影响本地 api 层）：
- **破坏性**：`pipeline_status["request_pending"]`/`["autoscanned"]` 被删除（600644b1）；本地 document_routes.py/index_build_service.py 有确认调用
- **兼容（已核实）**：shared_storage 六个符号（get_namespace_data/get_namespace_lock 等）签名全部未变；postgres_impl 连接层签名不变；本地调用的全部 doc_status 方法（get_doc_by_file_basename/get_docs_paginated/get_all_status_counts）在上游仍保留
- **新增强制约束**：base.py `DocStatusStorage` 新增 8 个抽象方法（get_docs_by_statuses_page 等核心项强制抽象）；**本地无自定义 DocStatusStorage 子类则无实现负担**

**C 组小计**: 可独立先做的（C1 小修组 + C2 全部 + C3 取消安全组，约 1.5 工作日）；PGTableGraphStorage 按专项节奏推进；LR2 reservation 留待整树同步。

### 主题 D：解析器/分块器（P0-P1，核心是注册表重构）

| # | 项 | 说明 | 上游来源 | 本地冲突 | golden/测试影响 | 优先级 |
|---|---|---|---|---|---|---|
| D1 | **parser 统一注册表重构** | 仿 kg STORAGES 表建 parser 注册表；删除全部 `parse_native/parse_mineru/parse_docling` 方法，dispatch 一律 `get_parser(engine).parse(ctx)`；BaseParser 统一契约 + ExternalParserBase 模板 + 插件发现 | #3235 c8be3406 及 wire 提交 | **大**：本地 pipeline.parse_legacy/parse_docling 定制点消失；lifecycle service 的 4 个 `rag.parse_*` 调用全部失效；debug.py 补丁作废 | 上游 pipeline 测试大量适配；本地 docling/mineru sidecar 测试需改为 Parser 类 + ParseContext | **P0**（其余全部 parser 侧改动的前置）|
| D2 | **legacy 命名冲突处理** | 本地 `legacy.py`（模块）vs 上游 `legacy/`（包），git 不会报冲突但运行时 ImportError；上游 extractors 是功能超集（xlsx 公式处理更好），且本地文档预览不依赖本地版 | #3235 fbd01485 | **隐蔽陷阱**：合并"成功"但运行时炸；必须在同一提交内显式 `git rm lightrag/parser/legacy.py` | 无 | **P0**（与 D1 同批）|
| D3 | docx 增强全家桶 | parse_document.py 去解析期 chunk 组装（重构）；invalid-package-error；**smart_heading**（中文公文红头/落款/题名块识别，spaCy + LLM 题名块判定）+ 5 个后续修复 | #3295/#3210/#3364 + 5 个修复 | **无实质冲突**（本地 docx 两文件均行尾差异）| 上游重生成本地已有 9 组 native_docx golden + 新增 extract_characterization/smart_heading 两套 fixture；**本地无需自行重生成——直接取上游 golden** | **P0**（#3295/#3210 随重构必吸）；smart_heading **P1**（特性开关默认关，可后开；内网需 spaCy 离线分发 `requirements-offline-smart-heading.txt`）|
| D4 | 原生 Markdown 解析器 | 新增 markdown/ 包；`.md`/`.textpack` 归 native 引擎；base64 图片物化进 sidecar；外链图下载（SSRF 多轮加固）；SVG→PNG（cairosvg）| #3280 + #3426 等 | 无（本地 legacy.py 把 .md 当 UTF-8 读，上游是严格升级）| 无 | P1（内网建议 `NATIVE_MD_IMAGE_DOWNLOAD_ENABLED=false`）|
| D5 | 表格处理 | HTML 表格解析抽为共享 `_html_table.py`；超长表格按行组切分、每段保留表头；多模态表格 prompt 格式声明 | #3219/#3216/#3218/#3221 | 无 | chunker 表格测试上游大幅扩展，直接取上游 | P1 |
| D6 | mineru/docling 修复组 | 后缀环境扩展；skip page_number 噪声块；连接错误清晰上报；docling inline JSON；**drawing path/src 语义**（sidecar 契约变化）；equation field fallback；xlsx 公式缓存回退 | #3520/#3288/#3274/#3208/#3501/#3344/#3455/#3515/#3375 | 低（本地 mineru cache/client 的 server_url 补丁 17 行可原位重放）| **#3455 改 drawings.json golden**，须与 docx golden 同批吸收；本地 sidecar 格式文档需同步更新 | P1 |
| D7 | chunker/引用系统 | chunk 注入父标题链；sidecar parent_headings 字段；抽取注入章节上下文（成套"标题上下文"检索质量线）；paragraph_semantic 大改；引用剔除参数；分隔符警告缓存；overlap 防护；sidecar 资产名边界安全 | #3211/#3223/#3225/#3215/#3214/#3285/#3508/#3550/#3396/#3316 | 4 个 chunker 零语义冲突；**唯一真冲突在 operate.py**（与本地双语检索段人工三方合并）| parity 测试与本地 sidecar-URI 用例正交，可拼接 | P1（#3211/#3223/#3225 三件一起吸；#3316 安全必吸）|
| D8 | 多模态稳定性包 | VLM LaTeX 转义修复；多模态 JSON 解析重试；控制字符防 graphml 崩溃；空标题关系防护 | #3238/#3242/#3237/#3224 | 本地相关文件仅行尾差异 | 无 | P1（整批吸收）|

**本地三件套迁移方案**（P0，与 D1 同批执行）：
1. **libreoffice**：改写为 `LibreOfficeDoclingParser(DoclingParser)` 子类（覆盖 `download_into`/`is_bundle_valid`，转换缓存沿用 `*.libreoffice_raw/`），`ENABLE_LIBREOFFICE_CONVERSION=true` 时 `register_parser` 注册扩展后缀 spec——本地 routing.py 的 46 行硬编码门控可整体删除
2. **mineru server_url**：17 行补丁按上游新 `overrides` 结构重放（`MinerUParserOptions` 加字段、签名加键、`_local_form_data` 加 3 行）
3. **lifecycle dispatch**：document_lifecycle_service 的 `rag.parse_*` if 链改为 `get_parser(engine).parse(ParseContext(...))`

**吸收路线结论**：**一步到位**（不建议渐进）。理由：本地实质冲突面只有 889 行/7 文件且有清晰新挂点；行尾差异使任何 cherry-pick 都表现为全文件冲突，渐进路线每步付同样成本；上游后续修复全落在重构后的文件上，绕开 #3235 会让每个修复都要手工回移。

### 主题 E：API 服务器（安全项已在 A 组，此处为其余）

| # | 项 | 说明 | 上游来源 | 本地现状核实结论 | 优先级 |
|---|---|---|---|---|---|
| E1 | token 续期缓存有界化 | 攻击者伪造 guest JWT（任意长 sub）可无限膨胀 `_token_renewal_cache` = 内存耗尽原语；上游三重有界（sub 长度上限 + 表大小上限 + 无条件 cap）| #3531 c8ea584f | **本地存在同样的无界 dict**（utils_api.py:42）；企业模式强制非默认 TOKEN_SECRET 缓解了伪造路径，但纵深防御仍值得吸收 | 中-高 |
| E2 | /health 匿名 liveness + 认证后配置 | 上游：/health 永远可达（liveness），未认证只给 status/版本，配置须认证 | #3329 | **本地无泄漏漏洞**（/health 带 `Depends(combined_auth)`，比上游更严）；但监控探针必须带凭据——上游方案对内网监控更友好，可选吸收 | 低（可选）|
| E3 | whitelist 路径段边界匹配 | `/*` 通配从前缀匹配改段边界匹配，防 `/health-secret` 之类误放行 | #3534 | 本地 whitelist 逻辑继承自 MB（同源），**大概率同样受影响**，吸收成本低（单函数）| 中 |
| E4 | body limit + input limits 中间件 | ASGI 流式 body 上限三层（普通 1MiB/text 50MiB/upload=MAX_UPLOAD_SIZE），查询字段硬上限 | #3544（LR2 Phase5）| 本地无等价机制；新增文件无路径冲突，但**中间件装配点在 lightrag_server.py**（本地 +1017 行企业化），需手工挂载并确认与本地企业中间件（审计/配额）顺序 | 高 |
| E5 | login 限速 | `LOGIN_MAX_FAILED_ATTEMPTS=5` + 锁定窗口（每进程内存计数）| ceac5a21 | 本地有自研工号登录 + person auth 双体系，**需评估把限速逻辑套到本地 /login 全家**（上游实现只覆盖其自带 /login）| 中-高 |
| E6 | sanitize 500 错误 | 500 响应体不再回显内部异常细节 | #3422 | 本地未核实等价机制，吸收成本低 | 中 |
| E7 | strip control chars parsed body | 解析产物剥离控制字符 | #3272 | 本地无等价；随 parser 侧同步 | 中 |
| E8 | worker 迁移 | uvicorn worker 孤儿退出修复、gunicorn_worker.py | #3506/#3494 | **WORKERS=1 钉死，价值有限**；多 worker 化时再取 | 低 |
| E9 | 功能增强杂项 | query-response-time、suggested user prompts、status card server mode、webui dynamic file types、feeder auto-rescan、stream directory scan、custom-chunks 两修复 | 多个 PR | 本地 query_routes/document_routes 已深度分叉：**core 侧改动随大合并取，API 表面改动按需选择性移植**（custom-chunks 两修复 #3401/#3394 值得取）| 低-中 |
| E10 | config.py 撞名检查 | — | — | **已核实无撞名**：本地新增配置全部 `LIGHTRAG_ENTERPRISE_*`/person 前缀，与上游新增（调度/准入/解析）零交集；config.py 合并是纯叠加 | ✅ |

**本地已自行解决、无需吸收**：guest token bypass（#3319，本地 auth 架构不同不受影响）；客户端中断处理（本地 streaming_lifecycle 自研，上游 stop-button-cooldown 是 WebUI 侧功能，不冲突）；/health 配置泄漏（本地更严）。

### 主题 F：配置/依赖/CI/运维/文档

**F1. env.example 新配置组**（+463 行，映射特性）：认证安全（login 限速）、请求体防护、流水线准入、调度有界化、**跨 worker 并发帽**（WORKERS=1 下暂无用）、docx smart_heading（17 项）、MD 远程图片（内网建议关）、解析引擎扩展（per-file hint 参数）、分块（引用剔除）、查询（content_headings）、VLM、存储（**PGTableGraphStorage 选项**、Milvus 迁移重试、Redis eviction 拒启）。

**F2. 重命名/行为变更兼容表**（影响本地 .env——本地 `.env.main:114` 用 `MAX_ASYNC=8`）：

| 项 | 变化 | 本地影响 |
|---|---|---|
| `MAX_ASYNC`→`MAX_ASYNC_LLM` | 旧名仍是 fallback | 暂不改也生效，建议模板更新 |
| `SOURCE_IDS_LIMIT_METHOD` | 默认 FIFO→KEEP | 未显式设置则合并行为变化 |
| `MAX_SOURCE_IDS_PER_ENTITY/RELATION` | 300→200 | 未显式设置则收紧 |
| `MAX_FILE_PATHS` | 100→75 | 本地有 file_path 展示功能（197684a3），需确认截断策略 |
| `LLM_TIMEOUT` 及角色超时 | 180→240；KEYWORD 60/QUERY 240/VLM 300 | 本地 vLLM 场景对照 |
| `MINERU_MAX_POLLS` 180→600、`MINERU_LOCAL_IMAGE_ANALYSIS` true→false | 轮询更久；默认不走 VLM 图像分析 | 用 MinerU 需对照 |
| `LIGHTRAG_PARSER` 分隔符 | 逗号→分号（推荐）| 现有规则不破坏 |
| `VLM_PROCESS_ENABLE` | 语义变化：只管 `i` 项；带 `i` 的文档在 false 下会 FAILED 而非跳过 | 有行为差异 |
| `ENABLE_LLM_CACHE_FOR_EXTRACT` | 从 env.example 移除（代码仍读，默认 true）| "去文档化"非移除 |

**F3. 依赖变更**：核心收紧 json_repair/tiktoken；api extra 新增 cryptography≥48.0.1、python-multipart≥0.0.30、starlette≥1.3.1、uvicorn-worker、cairosvg、spacy；新增 script `lightrag-rebuild-vdb`。**本地 memory extra（graphiti-core）区段上游未触碰，不会被挤掉**；pyproject 三处文本冲突均可平凡解决（依赖列表/scripts/pytest 配置，双方保留）。

**F4. 三个新运维工具**：

| 工具 | 功能 | 本地可用性 |
|---|---|---|
| `rebuild_vdb.py` | 从图/chunks 权威源重建三个 VDB + 一致性检查（换嵌入模型刚需）| **检查模式拷来即跑**；重建模式需约 10 行 embedding 适配器（本地 create_embedding_function 被重构成嵌套函数不可 import）|
| `kg_integrity_repair.py` | 审计 per-doc recovery anchor 缺口 + 孤儿报告 | **不可直接用**（依赖 base.py `strict=True` 参数与 `_union_doc_recovery_anchors`，须先移植 #3400 写前恢复锚框架）|
| `source_conflict_repair.py` | 源冲突显式修复 CLI | **完全不可用**（import 即失败，依赖 LR2 类型系统），随整树同步获得 |

**F5. CI 可借鉴**：pg-smoke.yml（真 PG18 容器跑 PGTable+图存储契约，两层假绿防护——与本地 lightrag_contract_test 实践直接对口）；`uv sync --frozen` + `uv run --no-sync`（记得加 `--extra memory`）；pytest markers 注册进 pyproject + conftest 隔离/spacy gating。

**F6. Docker**：非 root 运行（UID/GID 1000 + gosu 降权 entrypoint，CIS 合规可直接抄）；基镜像钉 `python:3.12-slim-bookworm`；spaCy wheel 在 `uv sync` 之后装（与本地 graphiti 被 uv sync 清掉是同一个坑的上游解法）。

**F7. 文档**：FileProcessingPipeline(-zh).md 全量重构（引擎能力矩阵）；LightRAG-API-Server(-zh).md 新增 **Upgrading 三节**（正是本地同步要处理的行为变更清单）；新 ParserServiceDeployment(-zh).md；OfflineDeployment.md spaCy 离线节。注意：上游 AGENTS.md 要求 commit message 英文，与本地中文提交惯例冲突，同步该文件时留意。

**F8. 上游删除项跟随**：`lightrag/tools/lightrag_visualizer/`（deprecated）与 `prepare_qdrant_legacy_data.py`——本地两者都还在（e1c2915e 恢复 tools 包时带回），**无任何本地引用，建议跟随删除**（与"9ebbd820 误删恢复"不矛盾：那次是整包误删，这次是上游有意废弃）。

### 主题 G：WebUI 与 k8s-deploy（待用户决策项）

- 本地 webui 已被 9ebbd820 误删（126 文件），**本地定制实测仅 3 个文件 +298 行**（KBDocumentPreviewViewer.tsx +231、api/lightrag.ts +64、constants.ts +3，可从 `9ebbd820^` 取回）
- 上游 webui 演进 48 文件 +4683/-1707：ui-speedup、**chat XSS sanitize（安全）**、follow-scroll、动态文件类型、拖拽恢复边、query mode 警告、streaming 修复、React 全家桶升级
- **恢复策略（若决定恢复）**：直接取 `upstream/main` 的 lightrag_webui 整目录（免费获得全部演进）+ 重放本地 3 文件定制，成本约半天
- k8s-deploy：上游仅 README 重命名；本地 30 文件删除待决策，从任一侧恢复均可

---

## 三、破坏点与行为变更总清单（整树同步的前置门槛）

### 3.1 升级即崩的 5 个破坏点

| # | 破坏点 | 本地受影响处 | 解法 |
|---|---|---|---|
| 1 | `docs_format="lightrag"` + `lightrag_document_paths` 入口删除（传参直接 TypeError）| index_build_service.run_build/run_build_batch 主路径 | 迁移到上游 `pending_parse + ReuseParser`（`resolve_stored_document_parser_engine` 可复用已解析行），或本地回补入口作为长期补丁 |
| 2 | `pipeline_status["request_pending"]`/`["autoscanned"]` 协议键删除（改 ingress mailbox）| document_routes.py L2941/L992/L999；index_build_service.py L451/L742 | 改写为 `acquire_enqueue_reservation`/`get_pipeline_ingress` 等预约 API；同步响应模型 |
| 3 | `lightrag/parser/legacy.py` vs `legacy/` 包命名撞车（git 静默通过，运行时 ImportError）| 本地 legacy 引擎全部调用点 | 合并提交内显式 `git rm lightrag/parser/legacy.py`，调用改上游 LegacyParser |
| 4 | `rag.parse_native/parse_mineru/parse_docling/parse_legacy` 方法删除 | document_lifecycle_service.run_parse 的 dispatch 链 | 改 `get_parser(engine).parse(ctx)` |
| 5 | `SUPPORTED_PARSER_ENGINES` 移出 constants → parser/registry | document_lifecycle_service 的 import | 一行 import shim |

### 3.2 语义/行为变更（不崩但需评估）

- **FAILED 文档不再自动重试**（只走 sticky manual retry 通道）——本地强制重建/重试路径（aa228098 等）依赖 re-drain 的地方需复核
- **LLM cache 全量冷启动**：cache policy v2 + compute_args_hash 键变更
- **实体名规范化**（#3566 手工实体编辑统一命名契约 + #3357 写入前 sanitize）——与本地"图谱 ID=实体名"约定（aaaacd4a）直接交互，**需专项评估存量实体 ID 稳定性后再决定**
- source_ids≤200 / file_paths≤75 有界化——本地 file_path 展示功能需确认
- pipeline_status 写路径引入 owner-token fence——本地任何直接置 `busy=True` 的写法需改走预约 API
- Redis eviction 策略不合规直接拒启（本地未用 Redis，无影响）

### 3.3 工程卫生前置

- **行尾归一化**（最重要的单项前置）：本地 78/164 个 py 文件 CRLF → 一次性 LF 化提交 + `.gitattributes` 加 `*.py text eol=lf`；合并用 `git merge -X renormalize`。不做这步，任何合并/cherry-pick 都会淹没在伪冲突里，且未来每次同步重复爆炸

---

## 四、推荐同步策略（四阶段路线图）

### 阶段 0：安全急修（本周内，完全独立于同步）
1. 移植 GHSA-2wpj 上传文件名修复（155 行 + 测试）
2. 从请求模型移除 `sentence_split_regex`（ReDoS）
3. uv.lock 升 starlette/python-multipart/cryptography 至安全下限（记得 `--extra memory`）
4. passwords.py 明文 fallback 改 `secrets.compare_digest`（1 行）
5. 顺手：删除 visualizer + prepare_qdrant_legacy_data（跟随上游）

### 阶段 1：工程卫生 + 零冲突修复包（1-2 周）
1. **行尾归一化提交 + .gitattributes**（后续一切的前提）
2. 独立 cherry-pick 包（全部无 schema/迁移风险）：
   - B5 健壮性一揽子 + B2 rerank 四连修（零冲突）
   - C2 Neo4j 修复组 + Milvus 双修复（生产后端真实 bug）
   - C1 postgres_impl 小修组（含 pgvector 懒加载）
   - C3 keyed-lock/namespace-lock 取消安全组（本地中断处理直接受益）
   - B1 缓存正确性三件套 + B3 抽取质量组（operate.py 手工三方合并，是本阶段主要工作量）
   - B4 tokenizer 有界化 + B6 LLM 绑定修复
   - E3 whitelist 段边界 + E6 sanitize 500 + E1 token 续期有界化

### 阶段 2：战略试点（2-4 周，并行）
1. **PGTableGraphStorage 单文件后移植试点**：观察上游 followup → 测试环境新 KB 双跑对比 → 决定是否作为企业模板默认图后端
2. smart_heading 评估（若公文场景需要）：spaCy 离线分发成本 vs 红头/落款识别收益
3. WebUI 恢复决策（若恢复：取上游整目录 + 重放 3 文件）

### 阶段 3：有计划的整树同步（时机成熟时，预计 1-2 周专注工作）
1. 预改造：先在本地把 5 个破坏点的适配做完（index_build 迁移 ReuseParser、request_pending 改预约 API、git rm legacy.py、lifecycle 改 registry dispatch、import shim）
2. `git merge upstream/main -X renormalize`，按分层规则解冲突：
   - ~130 个纯行尾文件 + 37 个零真实改动代码文件 → 直接取上游
   - ~40 个本地新增文件 → 保留
   - 5 个重度文件（lightrag_server/operate/config/pipeline/query_routes）→ 人工三方合并
3. 重放本地补丁：libreoffice 插件化、mineru server_url、_MergeStageProgress 重排到新 merge 结构、doc_id allowlist、敏感上下文注入、llm 日志脱敏
4. 回归：lightrag_contract_test 合同测试 + golden 直取上游 + 本地企业 E2E（tests/api）+ index_build/job_worker 专项回归
5. 收尾：env 模板对齐 F2 表、docs Upgrading 节参照、CI 搬 pg-smoke/frozen sync

### 整树同步 vs 持续 cherry-pick 的判断
- 阶段 0-2 的产出**不依赖**整树同步决策，先做无悔项
- 整树同步的核心收益：lr2 有界调度全家、parser 注册表生态（含后续所有修复的免手工回移）、source_conflict 工具链、未来同步成本大幅下降（合并祖先接续）
- 建议：完成阶段 0-2 后择机执行阶段 3；越晚做，上游在重构后文件上的新增修复越多，手工回移成本越高

---

## 五、建议放弃/暂缓项汇总

**放弃（明确不做）**
- lr2-bounded-scheduling 任一 phase 的单独 cherry-pick（8 phase 强耦合 + 5 后端 + admission 全家桶，只能整体）
- pipeline_ingress/scan_job_store 单独吸收（本地 PG 持久化作业系统是其能力超集）
- 跨 worker 全局并发闸门（WORKERS=1 钉死，收益为零且触及多个本地已改文件）
- redis/mongo/opensearch 的 scheduling 实现单独吸收（本地未用）
- AGE 相关修复专项投入（战略方向是 PGTable）
- #3387 query-response-time 的 query_routes/webui 部分（本地已重写，只取 core 侧）
- 本地 legacy.py / routing.py 三处门控 / debug.py 挂接（被上游超集/插件机制替代）
- E8 worker 迁移（多 worker 化时再说）

**暂缓评估（先专项评估再决定）**
- graph-sanitize-names #3357 + manual-entity-name-normalization #3566：与本地"图谱 ID=实体名"约定的交互，需评估存量实体 ID 稳定性
- smart_heading：spaCy + 钉版模型的离线分发成本
- /health 匿名 liveness 方案（本地现状更严，是否换取监控便利性）

---

*分析方法：git merge-tree 只读合并模拟 + git diff -w 逐文件真实改动核实 + 5 路并行深度分析（核心引擎/存储层/API/解析器/配置横切面）+ 关键结论人工复核。所有 PR 号与提交哈希均已对实际 diff 核实。*
