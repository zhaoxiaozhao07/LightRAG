# KB 聚合任务流水线并发改造设计方案

> 文档类型：服务端内部架构改造设计
> 状态：草案，待审阅
> 影响面：服务端 `lightrag/api/job_worker.py` + `lightrag/api/index_build_service.py`，HTTP API 契约不变
> 预期收益：4 文档聚合 job 总耗时从 ~140 min 降到 ~55 min（约 60% 加速）

---

## 1. 背景

### 1.1 业务场景

`/kbs/{kb_id}/documents:sync` / `/kbs/{kb_id}/documents:upload?auto_parse=true&auto_index=true` 等聚合接口接收 N 个文档时，会创建一个 `job_type=parse`、`document_ids=[...]` 的聚合任务，由 Job worker 串联跑完：

```
解析（MinerU）→ 多模态分析（analyze_multimodal）→ 实体关系抽取（extract）→ 实体关系合并/写库（merge）
```

企业知识库 MVP demo（`examples/enterprise_kb_mvp/enterprise_kb_mvp_demo.py`）默认走的就是这条 `:sync` 路径，目标用户是"一次扔进一批 PDF，等服务端跑完出问答态"。

### 1.2 用户观察到的现象

跑 4 个 PDF（其中含 116 字符长文件名的 `zheng-et-al-2024-...processing.pdf`）时，服务端日志呈现明显的**逐文档串行**节奏：

```
INFO: Phase 2 relation merge completed for doc-54a99c... in 16m54s
INFO: Phase 3 final index update completed for doc-54a99c... in 0.0s
INFO: Completed merging: ... Total merge=21m53s
INFO: [kb_enterprise_umvp_udemo] Writing graph with 1798 nodes, 3445 edges
INFO: Completed processing file 1/1: doc_2be0df7c4359__zheng-et-al-2024-...
INFO: Enqueued document processing pipeline stopped                    ← 关键标记
INFO: [MinerU] Parsing doc-88b42e... 白炭黑填充胶料的硫变动力学研究.pdf
INFO: Processing 1 document(s)                                          ← 下一个文档独立启一次 pipeline
```

`Enqueued document processing pipeline stopped` 出现在文档之间，说明每个文档都触发一次完整的"启动 pipeline → 排空 → 关闭 pipeline"生命周期。

### 1.3 物理资源拓扑

| 资源 | 部署 | 使用阶段 |
|---|---|---|
| VLLM 多模态模型（如 `MinerU2.5-Pro-2B@192.168.1.66:8001`） | 单一 VLLM server，支持多路 batch | MinerU parse + LightRAG `analyze_multimodal` 共享 |
| 文本 LLM（`LLM_BINDING` 配置的主模型） | 独立绑定 | extract（每个 chunk 一次） + merge（实体描述 ≥ 8 触发 summary） |
| Embedding 模型 | 独立绑定 | merge 阶段实体描述向量化 + chunks_vdb 入库 |
| PostgreSQL + pgvector | KV / vector / graph 后端 | merge 阶段大量 `entity_batch_upsert` / `relation_batch_upsert` |

**关键事实**：用户已验证 MinerU server 能稳定接受多路 `POST /tasks`（vLLM 内部 batch）。

---

## 2. 问题诊断

### 2.1 现象与代码定位

`Enqueued document processing pipeline stopped` 的源头是 `lightrag/pipeline.py:1172`，由 `apipeline_process_enqueue_documents()` 在排空一次 batch 后输出。日志一文档一次，说明这个方法对每个文档被调用一次。

向上追：`lightrag/api/job_worker.py:300-368` 的 `_run_aggregate` 是聚合 parse job 的执行体：

```python
for document_id in document_ids:
    ...
    item = await _execute_parse_plan(...)                # 阻塞等单文档 MinerU 解析
    if item["status"] == "succeeded" and auto_index ...:
        ...
        build_item = await _execute_build_plan(...)      # 阻塞等单文档完整 build_kg
    item_results.append(item)
```

`_execute_build_plan` 最终落到 `lightrag/api/index_build_service.py:273-284`：

```python
await rag.apipeline_enqueue_documents(
    input=[""],
    ids=[plan.document.lightrag_doc_id],                 # ← 一次只塞 1 个 doc
    file_paths=[unique_basename],
    ...
    lightrag_document_paths=[plan.sidecar_uri],
)
await rag.apipeline_process_enqueue_documents()          # ← 等三层流水线排空
return await _collect_doc_status(rag, plan)
```

### 2.2 根本原因

> **三层流水线架构存在但被浪费**：每次 `apipeline_enqueue_documents` 只塞 1 个 doc，三层 worker（parse / analyze / process）虽然在跑，但每层 queue 里始终只有 1 个 item，**跨文档 overlap 无从发生**。

`lightrag/pipeline.py:1199-1264` 的 `_run_pipeline_phase` 设计本意是 multi-doc：

```python
ctx = _BatchRunContext(
    semaphore=asyncio.Semaphore(self.max_parallel_insert),
    q_mineru=asyncio.Queue(maxsize=self.queue_size_default),
    q_analyze=asyncio.Queue(maxsize=self.queue_size_default),
    q_process=asyncio.Queue(maxsize=self.queue_size_insert),
)
for _ in range(max(1, self.max_parallel_parse_mineru)):
    workers.append(asyncio.create_task(self._parse_worker("mineru", ctx.q_mineru, ctx)))
for _ in range(max(1, self.max_parallel_analyze)):
    workers.append(asyncio.create_task(self._analyze_worker(ctx)))
for _ in range(max(1, self.max_parallel_insert)):
    workers.append(asyncio.create_task(self._process_worker(ctx)))
```

三层 worker 是独立 asyncio task，本可以让 doc A 在 process 时、doc B 在 analyze、doc C 在 parse 并行。但 `_run_aggregate` 把它一文档一文档喂，相当于"用八车道高速公路当独木桥"。

### 2.3 现状性能画像（实测 + 估算）

| 阶段 | 单文档耗时（实测/估算） | 4 文档串行总耗时 | VLM 状态 | 文本 LLM 状态 |
|---|---:|---:|---|---|
| MinerU parse | ~3 min | 12 min | 满载 | 闲置 |
| analyze_multimodal | ~2 min | 8 min | 满载 | 闲置 |
| extract（65 chunks × ~7s） | ~8 min | 32 min | 闲置 | 满载 |
| merge（851 entities + 1643 relations） | ~22 min | 88 min | 闲置 | 部分满载（仅 LLMmrg 时） |
| **合计** | ~35 min | **~140 min** | 4 段 × 5 min 满载，其余闲置 | 4 段 × 30 min 满载，其余闲置 |

> **VLM 闲置率 ≈ 86%**，文本 LLM 也有显著闲置（merge 阶段大头是 embedding + DB 写入）。

---

## 3. 需求理由

### 3.1 直接收益

1. **总耗时缩短 60%+**（4 文档场景，详见 §10）
2. **GPU/VLM 利用率提升**：用户购买的 VLLM 部署不再有 80%+ 时间 idle
3. **用户体验改善**：demo 跑 4 文档从 2.5 小时压到 1 小时内，用户能更快验证整条 KB 链路

### 3.2 对生产场景的意义

| 场景 | 当前痛点 | 改造后 |
|---|---|---|
| 一次上传 10-50 篇行业文献 | 串行跑可能 6-10 小时，用户已经下班 | 在用户在线时间内完成 |
| 增量 sync（每天追加几个文档） | 单文档跑还能接受，但同时多个的话会排队 | 自动 batch 处理无感 |
| 灾后 `:rebuild` 批量重建 | N 个文档 × 单文档时间 | N 个文档 / 并发数 × 单文档时间 |
| 多租户共享 VLLM | 文本 LLM 阶段时 VLM 闲置，租户 B 想用也用不上 | VLM 持续满载，提高整集群产出 |

### 3.3 不做这件事的代价

- **demo 体验差**：客户跑 MVP 看到 2.5 小时还没出结果，对项目信心打折
- **VLLM 浪费**：单卡推理资源紧张（你部署的就是 GPU），idle 80% 等于多花 4 倍 GPU 钱
- **后续优化天花板低**：再怎么调 `MAX_PARALLEL_*` env，job worker 这层串行没破，所有参数都失效

### 3.4 为什么是现在做

- 流水线架构（三层 queue）**已经写好**，只是没被聚合 worker 用上 → 改造成本低
- 用户刚踩过这个坑，记忆鲜活，方便联调
- 企业知识库 MVP demo 还在迭代，越早改对历史包袱越少

---

## 4. 设计方案

### 4.1 设计原则

1. **HTTP 契约零变更**：客户端、demo、外部集成无感
2. **状态机零变更**：每个文档仍走 `parse_queued → parsing → parsed → build_queued → building → ready`，只是时序重叠
3. **复用现有 pipeline 架构**：不自己实现 task pool / semaphore，直接用 `pipeline.py` 的三层 queue + `MAX_PARALLEL_*` 旋钮
4. **失败隔离**：单文档失败不影响其他文档（与现状一致）
5. **可观察性优先**：日志要让用户能区分新旧行为；新增字段在 result 里方便排查

### 4.2 推荐方案：A.1（最小改动 + 60% 加速）

**核心思路**：把 `_run_aggregate` 的 for 循环改成两阶段：

```
Phase 1: 并发跑所有文档的 MinerU 解析（受 MAX_PARALLEL_PARSE_MINERU 控）
              ↓ 所有 sidecar 落盘后
Phase 2: 一次性把全部成功文档 enqueue 进 pipeline，单次 process 调用排空
              ↓ pipeline 三层 worker 自然 overlap analyze/extract/merge
Phase 3: 收每个文档的最终状态，聚合到 result.items[]
```

### 4.3 方案 A.1 伪代码

```python
# lightrag/api/job_worker.py:_run_aggregate （改造后）

async def _run_aggregate(job, payload):
    document_ids = list(dict.fromkeys(payload.get("document_ids", [])))
    auto_index = bool(payload.get("auto_index", False))
    rag = await registry.get(kb_id)

    # ───────────── Phase 1: 并发 MinerU parse ─────────────
    parse_sem = asyncio.Semaphore(rag.max_parallel_parse_mineru)

    async def _do_one_parse(doc_id):
        async with parse_sem:
            try:
                plan = await document_service.create_parse_plan(
                    kb_id, doc_id,
                    parser_engine=payload.get("parser_engine"),
                    process_options=payload.get("process_options"),
                    force_reparse=bool(payload.get("force_reparse", False)),
                    auto_index=auto_index,
                )
                await document_service.mark_parse_queued(kb_id, doc_id, job=job, plan=plan)
                item = await _execute_parse_plan(
                    document_service=document_service,
                    kb_id=kb_id, job_id=job.id, plan=plan,
                    rag=rag, job_service=job_service,
                )
                return plan, item
            except Exception as exc:
                return None, _build_failed_item(doc_id, "parse_failed", str(exc))

    parse_results = await asyncio.gather(*[_do_one_parse(d) for d in document_ids])
    item_results = [item for _, item in parse_results]

    # ───────────── Phase 2: 一次性 bulk enqueue ─────────────
    if auto_index and index_service is not None:
        succeeded = [(plan, item) for plan, item in parse_results
                     if plan is not None and item["status"] == "succeeded"]
        if succeeded:
            # 收齐所有 build plans
            build_plans = []
            for plan, item in succeeded:
                try:
                    bp = await index_service.create_build_plan(kb_id, plan.document.id, rag=rag)
                    if not bp.skipped:
                        await index_service.claim_build_queued(kb_id, job_id=job.id, plan=bp)
                    build_plans.append((plan, item, bp))
                except Exception as exc:
                    item["status"] = "failed"
                    item["error_code"] = "build_plan_failed"
                    item["error_message"] = str(exc)

            # 新增：execute_batch 一次性 enqueue 多个 doc → pipeline 自然 overlap
            to_run = [bp for _, _, bp in build_plans if not bp.skipped]
            if to_run:
                await index_service.execute_batch(to_run, rag=rag, job_id=job.id)

            # 收每个 doc 的最终状态
            for plan, item, bp in build_plans:
                try:
                    item["build_result"] = await index_service.collect_doc_status(plan, bp)
                    if item["build_result"]["status"] not in {"succeeded", "cancelled"}:
                        item["status"] = "failed"
                        item["error_code"] = item["build_result"].get("error_code")
                        item["error_message"] = item["build_result"].get("error_message")
                except Exception as exc:
                    item["status"] = "failed"
                    item["error_code"] = "build_failed"
                    item["error_message"] = str(exc)

    # ───────────── Phase 3: 汇总并 transition_job ─────────────
    completed = sum(1 for item in item_results if item["status"] == "succeeded")
    failed = len(item_results) - completed
    final_result = _batch_parse_job_result(
        batch_id=job.batch_id or "",
        total_items=len(document_ids),
        completed_items=completed,
        failed_items=failed,
        items=item_results,
    )
    await job_service.transition_job(
        kb_id, job.id,
        status="succeeded" if failed == 0 else "failed",
        progress=1.0,
        completed_items=completed,
        failed_items=failed,
        result=final_result,
        ...
    )
```

```python
# lightrag/api/index_build_service.py 新增方法

async def execute_batch(
    self,
    plans: list[IndexBuildPlan],
    *,
    rag: Any,
    job_id: str,
) -> None:
    """Bulk-enqueue multiple build plans into the pipeline in a single call,
    so the three-layer worker stack (parse / analyze / process) can overlap
    documents instead of processing them one at a time."""
    if not plans:
        return
    ids = [p.document.lightrag_doc_id for p in plans]
    file_paths = [_kb_unique_basename(p) for p in plans]
    sidecar_uris = [p.sidecar_uri for p in plans]
    # 注意：apipeline_enqueue_documents 已经支持列表入参
    await rag.apipeline_enqueue_documents(
        input=[""] * len(plans),
        ids=ids,
        file_paths=file_paths,
        track_id=generate_track_id(f"build_batch_{job_id}"),
        docs_format="lightrag",
        lightrag_document_paths=sidecar_uris,
        parse_engine=plans[0].document.metadata.get("parse_engine"),
        process_options=plans[0].process_options or None,
    )
    await rag.apipeline_process_enqueue_documents()

async def collect_doc_status(self, plan, build_plan) -> dict:
    """从 _collect_doc_status 抽出来的单 doc 状态读取，给 batch 流程复用。"""
    return await _collect_doc_status(rag, plan)
```

### 4.4 备选方案：A.2（深度 overlap，可后续迭代）

不等所有 MinerU parse 完成，每个文档解析完立刻 enqueue 进 pipeline（利用 `pipeline_status["request_pending"]` 机制）。让 MinerU parse 和 analyze/extract/merge 也能 overlap。

**优势**：在 A.1 基础上再省 10-15 min（4 文档场景）
**代价**：
- 需要协调 `apipeline_process_enqueue_documents` 的生命周期（已经在跑还是要重新启）
- 失败处理更复杂（pipeline 中途有些 doc 还没 enqueue）
- 测试覆盖面更大

**结论**：A.2 留作 A.1 验证稳定后的二期优化，本设计文档以 A.1 为目标。

### 4.5 不采用的方案及理由

| 方案 | 不采用原因 |
|---|---|
| 把 for 循环改成 `asyncio.gather` 全并发跑每个文档的 parse + build_kg | 每个 `_execute_build_plan` 各自调 `apipeline_process_enqueue_documents`，但同 KB 的 `pipeline_status["busy"]` 是互斥的（见 `AGENTS.md`），实际仍会被序列化，且锁竞争更乱 |
| 把 MinerU 解析移到 pipeline 的 `q_mineru` worker（即 A.3） | 需要把 KB 控制面（`parse_queued/parsing/parsed`）和 pipeline 的内部状态机（`PENDING/PARSING/ANALYZING`）双向桥接，改动面巨大且影响 `:parse` 单文档接口 |
| 在客户端层（demo）把一次 `:sync(4 docs)` 拆成 4 次 `:sync(1 doc)` 时间错开 | 不解决服务端问题；不同部署的 demo 客户端都得各自实现错开逻辑；且 `pipeline_status["busy"]` 互斥使得即使错开发，第二个 `:sync` 也会等到第一个的 pipeline 排空 |

---

## 5. 预计改动清单

### 5.1 必改文件

| 文件 | 改动类型 | 大致行数 | 说明 |
|---|---|---:|---|
| `lightrag/api/job_worker.py` | 重构 `_run_aggregate` | ~80 行 | 串行 for 循环 → 两阶段并发结构 |
| `lightrag/api/index_build_service.py` | 新增 `execute_batch` + `collect_doc_status` 公开方法 | ~40 行 | bulk 入队 + 状态收集解耦 |

### 5.2 新增测试

| 测试文件 | 测试用例 | 目的 |
|---|---|---|
| `tests/api/test_kb_aggregate_concurrent.py`（新） | `test_aggregate_parse_runs_multiple_docs_concurrently` | 入 4 mock doc，断言至少有时间窗口内 3 个 doc 处于 `building` 状态 |
| 同上 | `test_aggregate_collects_results_regardless_of_completion_order` | 验证 `result.items[]` 长度等于输入，不要求顺序 |
| 同上 | `test_aggregate_isolates_per_doc_failures` | 4 doc 里 mock 2 doc 失败，断言其他 2 doc 仍走完整路径并 ready |
| 同上 | `test_aggregate_respects_max_parallel_parse_mineru` | 设 `MAX_PARALLEL_PARSE_MINERU=2`，断言同时活跃的 MinerU 调用数 ≤ 2 |
| 同上 | `test_aggregate_emits_single_pipeline_drain` | 断言 `apipeline_process_enqueue_documents` 在整个 batch 期间只被调用 1 次 |

### 5.3 既有测试可能要调整

需要 grep 后逐个 review：

- `tests/api/test_kb_*.py` 里如果有断言 `pipeline_status["busy"]` 在文档间会 False→True→False 的，要松动成"整个 batch 期间持续 True"
- `tests/api/routes/test_kb_*.py` 里如果断言 `result.items[]` 严格按输入顺序的，要改成集合比对

### 5.4 文档改动

| 文档 | 改动 |
|---|---|
| `docs/API接口文档.md` L130 | 把"实际逐个执行解析"修订为"批量并发解析（受 `MAX_PARALLEL_PARSE_MINERU` 约束）" |
| `docs/FileProcessingPipeline-zh.md` | 补一段"聚合 job 与三层流水线的关系"说明 |
| `AGENTS.md` *Pipeline concurrency contract* | 在 `request_pending` 旁边加一句"聚合 job 入队时一次性塞 N doc，pipeline 内部 overlap" |

### 5.5 配置/部署

**无需改 env**。但建议补充推荐值：

```bash
# .env 推荐（在 MinerU server 能扛 2-3 路前提下）
MAX_PARALLEL_PARSE_MINERU=2     # 当前 1，提到 2
MAX_PARALLEL_ANALYZE=8          # 当前 8，保持
MAX_PARALLEL_INSERT=3           # 当前 3，可试 4
QUEUE_SIZE_DEFAULT=100          # 当前 100，保持
QUEUE_SIZE_INSERT=4             # 当前 4，建议提到 8（更大的批量缓冲）
```

---

## 6. 不变的契约（对外承诺）

| 接口/契约 | 是否变化 |
|---|---|
| `POST /kbs/{kb_id}/documents:sync` 请求 schema | **不变** |
| `POST /kbs/{kb_id}/documents:upload` 请求 schema | **不变** |
| `POST /kbs/{kb_id}/documents:batch-parse` 请求 schema | **不变** |
| `POST /kbs/{kb_id}/documents/{id}:build-kg` 请求/响应 | **不变**（仍走单 doc 路径，不受聚合改造影响） |
| 聚合 job 的 `result.items[]` 元素字段 | **不变** |
| 单文档状态机（`parse_queued → parsing → parsed → build_queued → building → ready`） | **不变** |
| `process_options` / `parser_engine` / `auto_parse` / `auto_index` / `idempotency_key` 语义 | **不变** |
| 失败时 `result.error_code` / `error_message` | **不变** |
| `pipeline_status` 字段（`busy` / `destructive_busy` / `pending_enqueues` / `request_pending`） | **语义不变**，但 `busy=True` 的窗口会更长（不会在文档间跳 False） |
| Cancellation（`POST /jobs/{job_id}:cancel`） | **语义不变**，pipeline 自带 `cancellation_requested` 协作式取消会传到所有 in-flight doc |
| 重试（`POST /jobs/{job_id}:retry`） | **语义不变** |
| Crash-resume / orphan recovery | **语义不变**，聚合 job 重新跑会重 plan 全部 doc（plan 是幂等的） |

---

## 7. 行为差异（用户可观察）

以下是预期的非 bug 行为变化，需要事先告知用户/运维：

| 维度 | 旧行为 | 新行为 |
|---|---|---|
| 日志阶段切换 | 一文档跑完一段，pipeline stop，下一个文档启动 | 多文档日志交错；同时刻可能看到 doc A 在 merge、doc B 在 extract、doc C 在 analyze |
| `Enqueued document processing pipeline stopped` 出现次数 | 每文档 1 次（N 次） | 整个 batch 1 次 |
| `result.items[]` 顺序 | 等于输入 `document_ids` 顺序 | 不保证（按完成时间） |
| 单文档失败的影响 | 后续文档完全独立 | 后续文档完全独立（无变化） |
| `pipeline_status["busy"]` 时序 | True→False→True→False...（N 个 doc N 个周期） | 整个 batch 持续 True，结束才 False |
| 单文档 ETA 估算 | 准确（独占资源） | 偏高（资源被并发文档分摊） |
| GPU/CPU 使用率曲线 | 锯齿状（满载-闲置交替） | 平稳满载（直到 batch 末尾） |

---

## 8. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|---|---|---|---|
| MinerU server 在 2-3 路并发下 OOM 或拒服务 | 中 | 高（parse 失败） | A.1 上线前用 `curl` 打 2-3 路并发 stress test；初始 `MAX_PARALLEL_PARSE_MINERU=2` 保守起步 |
| Pipeline `pipeline_status_lock` 在多 doc 入队时争用变热点 | 低 | 中（轻微延迟） | 锁内只做毫秒级写入；如果实测有问题再细化锁粒度 |
| `kb_documents` 表的 `state` 字段在并发 UPDATE 时事务冲突 | 低 | 中（重试可解） | Postgres 行级锁，不同 doc 不冲突 |
| 多 doc 并发触发 LLM provider 限流（OpenAI rate limit） | 中 | 中（部分 chunk 失败重试） | LightRAG 本身有 LLM 重试机制；建议同时审查 `LLM_MAX_ASYNC` 设置 |
| `result.items[]` 顺序变化导致下游消费者出错 | 低 | 低 | 内部 demo 不依赖顺序；外部对接方需在 release notes 中显式告知 |
| Cancel 在并发时未能终止所有 in-flight doc | 中 | 中（资源未释放） | 新增专门测试 `test_aggregate_cancellation_propagates`；pipeline 已有 `cancellation_requested` 协作式机制 |
| 改造引入死锁/资源泄漏 | 低 | 高 | 充分单元测试 + 4 doc 端到端测试 + 24h soak test |

### 8.1 回滚预案

如果生产发现问题：

1. **代码回滚**：A.1 改动集中在 2 个文件，git revert 一次提交即可
2. **配置回滚**：临时把 `MAX_PARALLEL_PARSE_MINERU=1` + `MAX_PARALLEL_INSERT=1` 调成 1，行为退化到接近串行（即使不回滚代码）
3. **降级开关**（可选）：加一个 env `LIGHTRAG_AGGREGATE_PARSE_MODE=serial|parallel`，默认 parallel，出问题切 serial（但代码上仍走新路径，只是并发度=1）

---

## 9. 测试计划

### 9.1 单元测试（pytest）

- `tests/api/test_kb_aggregate_concurrent.py`（新增，详见 §5.2）
- 跑 `uv run pytest tests/api/ -x` 确保现有 200+ 个 KB 相关测试不破

### 9.2 端到端 demo 测试

- 跑 `enterprise_kb_mvp_demo.py` 对 4 个 PDF（模拟文件目录）
- 验收标准（详见 §11）：
  - 总耗时 ≤ 65 min
  - 服务端日志能看到多文档交错
  - 最终 4 个 doc 都到 `ready` 状态
  - query 接口能返回结果

### 9.3 压测

- 同 KB 入 12 个 PDF（用 demo 的 `--max-files 12` 或扩展数据集）
- 监控指标：
  - VLLM GPU 利用率（目标 > 70% 持续）
  - 文本 LLM 调用 QPS（应平稳，不应有"先低后高"的脉冲）
  - Postgres connection pool 利用率（不应饱和）
  - 内存占用（不应随 doc 数线性增长）

### 9.4 异常场景

- 跑 batch 中途 cancel：断言全部 in-flight doc 转 `cancelled`
- 跑 batch 中途 kill server：重启后验证 orphan recovery 把 in-flight doc 转 `parse_failed`，retry job 能重跑
- 1 个 doc 故意构造 parse 失败（用文件大小超限的 PDF）：断言其他 3 个 doc 仍走通

---

## 10. 收益预估

### 10.1 单 batch 总耗时对比（4 文档场景，user 实测数据为基准）

| 参数 | 当前串行 | A.1 改造后 |
|---|---:|---:|
| 单文档 MinerU parse | 3 min | 3 min |
| 单文档 analyze | 2 min | 2 min |
| 单文档 extract | 8 min | 8 min |
| 单文档 merge | 22 min | 22 min |
| `MAX_PARALLEL_PARSE_MINERU` | 1 | 2 |
| `MAX_PARALLEL_ANALYZE` | 8 | 8 |
| `MAX_PARALLEL_INSERT` | 3 | 3 |
| Phase 1: 全部 MinerU parse | 4 × 3 = 12 min | ⌈4/2⌉ × 3 = 6 min |
| Phase 2: 全部 analyze + extract + merge（流水线 overlap，瓶颈在最慢的 merge） | 4 × 32 = 128 min | max(merge × ⌈4/3⌉, analyze + extract + merge) ≈ 22 × 2 ≈ 44 min |
| **总计** | **~140 min** | **~50-55 min** |
| 加速比 | 1× | **2.5-2.8×** |

### 10.2 资源利用率对比

| 资源 | 当前 | A.1 改造后 |
|---|---|---|
| VLM (MinerU + analyze 共享) 满载时间占比 | 14%（20/140 min） | **22%**（12/55 min），且后续工作量可以通过 batch 更大持续提升 |
| 文本 LLM 满载时间占比 | 60%（merge 阶段大部分） | **80%+**（3 个 doc 并发 merge） |
| GPU 闲置成本 | 高 | 显著降低 |

### 10.3 用户体验

- demo 一次跑完时间从 2 小时 20 分 → 50 分钟左右
- 客户在演示场景能在 1 小时内看到完整问答效果
- 生产场景批量入库的反馈时间可预期、可承诺 SLA

---

## 11. 验收标准

| 验收项 | 标准 | 验证方法 |
|---|---|---|
| 总耗时显著缩短 | 4 doc demo ≤ 65 min（基线 140 min） | 跑 demo 卡表 |
| HTTP API 完全无 breaking change | 现有客户端代码 0 修改即可工作 | 不改任何 demo 代码跑 demo |
| 所有现有测试通过 | `uv run pytest tests/` 0 failure | CI |
| 新增并发场景测试通过 | 5 个新测试全 pass | CI |
| 日志能体现 overlap | 出现"doc A 在 merge / doc B 在 extract"的交错日志 | 人工 grep `lightrag.log` |
| 单文档失败不影响其他 doc | 故意一个失败，其他 3 个仍 ready | 测试 + 手动跑 |
| Cancellation 正确传播 | cancel 后所有 in-flight doc 转 cancelled | 测试 + 手动跑 |
| `pipeline_status` 互斥契约保持 | concurrent enqueue 不会丢 doc | 现有 `test_pipeline_*` 测试 |
| VLM 利用率提升 | GPU 监控显示满载时间占比从 14% → 22%+ | nvidia-smi / Prometheus |

---

## 12. 实施步骤建议

| Step | 内容 | 预计工时 |
|---|---|---:|
| 1 | 用 `curl` 对 mineru-api 打 2/3/4 路并发 stress test，确认 server 稳定性 | 0.5 d |
| 2 | 实现 `index_build_service.execute_batch` + `collect_doc_status` | 0.5 d |
| 3 | 重构 `job_worker._run_aggregate` 为两阶段 | 1 d |
| 4 | 新增 5 个并发场景测试 | 1 d |
| 5 | 跑既有测试套件 + 修复破坏的断言 | 0.5 d |
| 6 | 端到端 demo 验证 + 性能对比 | 0.5 d |
| 7 | 更新 `docs/API接口文档.md` / `AGENTS.md` 措辞 | 0.5 d |
| 8 | code review + merge | 0.5 d |
| **合计** | | **~5 人日** |

---

## 13. 后续可能的二期优化

按价值/成本排序：

1. **方案 A.2**：增量 enqueue（每个 doc parse 完立刻入 pipeline，不等全部 parse 完成），让 MinerU parse 和后段 overlap。再省 ~10-15 min（4 doc 场景）
2. **跨 KB 共享 VLLM 调度**：不同 KB 的 batch 错峰提交时，全局调度让 VLLM 始终满载
3. **MinerU 移入 pipeline 的 `q_mineru` worker**：方案 A.3，让 MinerU parse 也享受 pipeline 内部的失败重试、cancel 协作式响应。但需要桥接 KB 控制面与 pipeline 内部状态机，改动面大
4. **merge 阶段拆分批写库**：当前 `entity_batch_upsert` 是 3-13 一批，可调大批次降低 DB roundtrip
5. **embedding 调用合并**：merge 阶段的 entity 描述向量化可以攒批一次性调 embedding API

---

## 附录 A：相关代码索引

| 路径 | 角色 |
|---|---|
| `lightrag/api/job_worker.py:278-389` | 聚合 parse job 的串行 for 循环（**待改造**） |
| `lightrag/api/job_worker.py:180-211` | `build_parse_executor` 入口与文档说明 |
| `lightrag/api/index_build_service.py:273-284` | `execute()` 单 doc 入队（保留给 `:build-kg` 单文档接口用） |
| `lightrag/api/routers/kb_document_routes.py:1211-1300` | `_execute_parse_plan` 单 doc parse 编排（不动） |
| `lightrag/api/routers/kb_document_routes.py:1526+` | `_execute_build_plan` 单 doc build 编排（不动，但聚合 worker 不再调它） |
| `lightrag/pipeline.py:1199-1264` | `_run_pipeline_phase` 三层 queue worker fan-out（不动，**这是改造的依靠对象**） |
| `lightrag/pipeline.py:1465-1610` | Layer 1 `_parse_worker`（不动） |
| `lightrag/pipeline.py:1634-1740` | Layer 2 `_analyze_worker`（不动） |
| `lightrag/pipeline.py:1742-1760` | Layer 3 `_process_worker`（不动） |
| `lightrag/pipeline.py:2755-2871` | `parse_mineru` 实现（不动） |
| `lightrag/lightrag.py:512-543` | `MAX_PARALLEL_*` env 与 dataclass 字段（不动） |
| `lightrag/constants.py:278-285` | 默认并发值常量（不动） |

## 附录 B：相关现有文档

- `docs/API接口文档.md` — KB 接口文档（L130 一句需修订）
- `docs/FileProcessingPipeline-zh.md` — 文档处理流水线说明
- `docs/LightRAG-API-MinerU-Workflow-zh.md` — MinerU 调用工作流
- `AGENTS.md` *Pipeline concurrency contract* 章节 — pipeline 互斥规则

## 附录 C：术语对照

| 术语 | 含义 |
|---|---|
| 聚合 job | `job_type=parse`、`document_id=null`、`payload.document_ids` 非空，由 `:sync` / `:upload?auto_parse=true` / `:batch-parse` 产生 |
| 单文档 job | `document_id` 非空的 `parse` / `build_kg` job，由 `:parse` / `:build-kg` 产生 |
| Layer 1/2/3 worker | `pipeline.py` 中 `_parse_worker` / `_analyze_worker` / `_process_worker` 三层 asyncio task |
| `parse_engine="mineru"` | 文档要走 MinerU 解析（vs `native` / `docling`） |
| sidecar | LightRAG 的解析产物目录，含 `blocks.jsonl` + assets，详见 `docs/LightRAGSidecarFormat-zh.md` |
| extract | 实体关系抽取阶段（每 chunk 调一次文本 LLM） |
| merge | 实体关系合并/写库阶段（含 LLM summary + embedding + DB 写入） |

---

## 附录 D：实施记录（落地与设计的差异）

> 本节记录实际实现，覆盖设计草案 §5/§12 中的预估。设计草案的方案选型（A.1 两阶段）未变，但实施范围与改动文件比草案更大——因为聚合并发不止 §5.1 列的 2 个 call site。

### D.1 实际改动的 call site（4 处，非草案的 2 处）

聚合"逐文档串行 drain"在代码里有 **4 个**入口，全部改为两阶段（并发 parse → 单次批量 drain）：

| # | 位置 | 触发接口 |
|---|---|---|
| 1 | `kb_document_routes.py::_run_auto_parse_batch` | `:upload?auto_parse=true&auto_index=true`（in-process 后台任务） |
| 2 | `kb_document_routes.py::_sync_task`（`create_kb_document_routes` 内） | `:sync`（in-process，**demo 走这条**） |
| 3 | `job_worker.py::build_parse_executor._run_aggregate` | 上述聚合 parse 任务的 durable worker 续跑 |
| 4 | `job_worker.py::build_sync_executor._run` | `:sync` 任务的 durable worker 续跑 |

为支持 #2/#4，给 `_run_sync_followups` / `_execute_sync_item` 加了 `defer_build` 参数：Phase 1 只 claim `build_queued` 并把 `IndexBuildPlan` 暂存到 `item["_deferred_build_plan"]`，Phase 2 统一批量构建。

### D.2 新增的服务层接口

- `IndexBuildService.run_build_batch(rag, plans, *, job_id)`：一次 `apipeline_enqueue_documents([N])` + 一次 `apipeline_process_enqueue_documents()`，然后**轮询** read-back。
- `IndexBuildService.collect_doc_status(rag, plan)`：公开的单 doc 状态读取。
- `kb_document_routes.py::_execute_build_plan_batch`：批量版 `_execute_build_plan`，逐 doc `mark_building → run_build_batch → complete_build/fail_build`。

### D.3 对抗式审查发现并修复的并发缺陷（4 项）

实现后经多 agent 对抗审查，确认并修复：

1. **【HIGH】并发 drain 竞态**：`run_build_batch` 单次 `process_enqueue` 在另一 flow 持有 `busy=True` 时早返回不排空，立即 read-back 会把仍 `pending` 的 doc 误判 `build_failed`。**修复**：read-back 改为轮询 `doc_status` 直到终态（`processed`/`failed`）或超时（`KB_BUILD_DRAIN_TIMEOUT_SECONDS`，默认 3600s；单 flow 首轮即终态、不 sleep）。同时修复了单 doc `run_build` 的同源既存竞态。
2. **【MED】gather 异常导致 claim 泄漏**：4 处 `asyncio.gather` 缺 `return_exceptions=True`，Phase 1 抛 `BaseException` 会丢弃已 claim `build_queued` 的兄弟结果。**修复**：4 处全加 `return_exceptions=True`，异常按位转 failed item，保证 Phase 2 必达以释放 claim。
3. **【MED】批量构建途中取消误报**：批量只在开头查一次 cancel，drain 中途取消被报 `build_failed` 而非 `cancelled`。**修复**：read-back 检测 `doc_status.error_msg` 的 `"User cancelled"` 标记，回报 `cancelled`（经 `_cancel_build_item`）。
4. **【MED·已知/不在本次范围】非强制重建陈旧计数**：非强制重建已 `processed` 的 doc 被引擎 `filter_keys` 跳过、回报陈旧计数为成功。**该缺陷在重构前的单 doc `run_build` 已存在，本次忠实保留**，记为后续单独修复项（replace 路径因显式 `adelete_by_doc_id` 不受影响）。

### D.4 配置

`.env`：`MAX_PARALLEL_PARSE_MINERU` 由 `1` 调到 **`4`**（MinerU 后端单 VLLM 部署、可承受多路 batch，已确认）。新增可选环境变量 `KB_BUILD_DRAIN_TIMEOUT_SECONDS`（默认 3600）/ `KB_BUILD_DRAIN_POLL_SECONDS`（默认 1.0）。

### D.5 测试

- 新增 `tests/api/routes/test_kb_aggregate_concurrent.py`（9 用例）：4-doc 并发跑通 + 单次 drain、结果顺序无关、单 doc 失败隔离、`MAX_PARALLEL_PARSE_MINERU` 限流、`:upload` 单次 drain，外加 3 个直击 race fix 的 `run_build_batch` 单测（等待并发 drain、取消分类、skip 计划）。已用"破坏 Fix A→测试失败→恢复→通过"验证非空。
- 全量 `tests/api/` 回归：**426 passed, 26 skipped, 1 failed**；唯一失败 `test_document_routes_docx_archive.py::test_pipeline_enqueue_md_moves_after_enqueue` 为预存在的 CRLF/LF 行尾问题（在未含本次改动的基线上同样失败），与本改造无关。
