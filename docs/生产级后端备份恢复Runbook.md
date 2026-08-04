# LightRAG 生产级后端备份恢复 Runbook

> 适用范围：单服务器部署下的 KB 控制面 metadata（local SQLite / PostgreSQL）、对象存储（MinIO/S3）、LightRAG 引擎外部后端（PostgreSQL KV/doc_status、Neo4j/Memgraph 图、Milvus/Qdrant 向量、Redis/Mongo/OpenSearch）以及文件型本地缓存。本文是生产演练模板；实际执行前必须替换占位符、确认停机窗口和凭据来源。

## 1. 恢复目标与一致性原则

- **RPO**：以最近一次成功备份时间为准；对象存储和数据库备份必须使用同一演练批次标签（例如 `backup_20260608_120000`）。
- **RTO**：先恢复控制面 metadata 和对象存储，再恢复引擎后端，最后启动 API server 做一致性检查。
- **一致性顺序**：停止写入 -> 备份 metadata -> 备份对象存储 -> 备份外部引擎后端 -> 记录 `.env`/compose/K8s 配置摘要。
- **严禁只备份 `.env` 或 compose**：setup wizard 的配置备份不等同于生产数据备份。

## 2. 备份前检查

```bash
# 1) 确认当前配置，不输出 secret 明文
python - <<'PY'
import os
keys = [
    'LIGHTRAG_KB_METADATA_BACKEND', 'LIGHTRAG_OBJECT_STORAGE',
    'LIGHTRAG_KV_STORAGE', 'LIGHTRAG_VECTOR_STORAGE',
    'LIGHTRAG_GRAPH_STORAGE', 'LIGHTRAG_DOC_STATUS_STORAGE',
]
for key in keys:
    print(f'{key}={os.getenv(key, "<unset>")}')
PY

# 2) 建议进入维护窗口：暂停上传/解析/构建/删除等写操作
# 如果使用 API 前端，先下线入口或临时切只读；如果使用 worker，先停止 durable worker 进程。

# 3) 记录备份批次
export BACKUP_ID=backup_$(date -u +%Y%m%d_%H%M%S)
mkdir -p backups/${BACKUP_ID}
```

## 3. KB 控制面 metadata 备份

### 3.1 local / SQLite 模式

适用：`LIGHTRAG_KB_METADATA_BACKEND=local` 或未设置。

```bash
# 工作目录通常来自 WORKING_DIR，默认 ./rag_storage
export WORKING_DIR=${WORKING_DIR:-./rag_storage}
mkdir -p backups/${BACKUP_ID}/metadata

# KB catalog JSON
cp "${WORKING_DIR}/metadata/knowledge_bases.json" "backups/${BACKUP_ID}/metadata/knowledge_bases.json"

# SQLite metadata 建议用 sqlite3 在线备份命令；没有 sqlite3 时需停写后复制文件。
sqlite3 "${WORKING_DIR}/metadata/metadata.sqlite3" ".backup 'backups/${BACKUP_ID}/metadata/metadata.sqlite3'"
```

恢复：

```bash
# 确认 API server/worker 已停止
export WORKING_DIR=${WORKING_DIR:-./rag_storage}
mkdir -p "${WORKING_DIR}/metadata"
cp "backups/${BACKUP_ID}/metadata/knowledge_bases.json" "${WORKING_DIR}/metadata/knowledge_bases.json"
cp "backups/${BACKUP_ID}/metadata/metadata.sqlite3" "${WORKING_DIR}/metadata/metadata.sqlite3"
```

### 3.2 PostgreSQL 控制面模式

适用：`LIGHTRAG_KB_METADATA_BACKEND=postgres`，DSN 来自 `LIGHTRAG_KB_POSTGRES_DSN` 或拆分变量。

```bash
# 推荐使用 pg_dump custom format，便于选择性恢复
pg_dump "$LIGHTRAG_KB_POSTGRES_DSN" \
  --format=custom \
  --file="backups/${BACKUP_ID}/kb_metadata.dump" \
  --table=kb_catalog \
  --table=kb_documents \
  --table=kb_jobs \
  --table=kb_document_artifacts \
  --table=kb_config_versions \
  --table=kb_document_source_keys \
  --table=enterprise_users \
  --table=enterprise_system_settings \
  --table=enterprise_kb_acl \
  --table=enterprise_audit_events \
  --table=enterprise_api_keys \
  --table=kb_metadata_schema
```

恢复到空库：

```bash
pg_restore --clean --if-exists --no-owner \
  --dbname="$LIGHTRAG_KB_POSTGRES_DSN" \
  "backups/${BACKUP_ID}/kb_metadata.dump"
```

## 4. 对象存储备份与恢复

适用：`LIGHTRAG_OBJECT_STORAGE=minio|s3`。对象 URI 通常记录在 document metadata 或 artifact metadata 中，`INPUT_DIR` 只是本地 cache。

MinIO / S3 兼容备份示例：

```bash
export BUCKET=${LIGHTRAG_OBJECT_STORAGE_BUCKET:-lightrag-kb}
export PREFIX=${LIGHTRAG_OBJECT_STORAGE_PREFIX:-kb}

# MinIO mc：先配置 alias
mc alias set lightrag "$LIGHTRAG_OBJECT_STORAGE_ENDPOINT" \
  "$LIGHTRAG_OBJECT_STORAGE_ACCESS_KEY_ID" \
  "$LIGHTRAG_OBJECT_STORAGE_SECRET_ACCESS_KEY"
mc mirror --overwrite "lightrag/${BUCKET}/${PREFIX}" "backups/${BACKUP_ID}/object-storage/${PREFIX}"

# AWS CLI/S3 网关等价命令
aws s3 sync "s3://${BUCKET}/${PREFIX}" "backups/${BACKUP_ID}/object-storage/${PREFIX}"
```

恢复：

```bash
mc mirror --overwrite "backups/${BACKUP_ID}/object-storage/${PREFIX}" "lightrag/${BUCKET}/${PREFIX}"
# 或
aws s3 sync "backups/${BACKUP_ID}/object-storage/${PREFIX}" "s3://${BUCKET}/${PREFIX}"
```

本地 cache 备份（可选但推荐，能降低恢复后首次下载对象的延迟）：

```bash
export INPUT_DIR=${INPUT_DIR:-./inputs}
tar -C "${INPUT_DIR}" -czf "backups/${BACKUP_ID}/input-cache.tgz" .
```

## 5. LightRAG 引擎外部后端备份

不同 KB 通过 `workspace` 隔离。恢复时必须保证控制面 metadata 中的 workspace 与外部后端数据一致。

### 5.1 PostgreSQL KV / doc_status / vector 后端

适用：`LIGHTRAG_KV_STORAGE=PGKVStorage`、`LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage`、`LIGHTRAG_VECTOR_STORAGE=PGVectorStorage` 等。

```bash
# 如果引擎表与控制面共用同一数据库，可用整库备份；否则替换为引擎 POSTGRES_* DSN。
pg_dump "$POSTGRES_DSN" --format=custom --file="backups/${BACKUP_ID}/lightrag_engine_pg.dump"

# 恢复
pg_restore --clean --if-exists --no-owner --dbname="$POSTGRES_DSN" \
  "backups/${BACKUP_ID}/lightrag_engine_pg.dump"
```

### 5.2 Neo4j / Memgraph 图后端

Neo4j：

```bash
# 停止写入后在 Neo4j 主机执行；database 名通常来自 NEO4J_DATABASE
neo4j-admin database dump "${NEO4J_DATABASE:-neo4j}" --to-path="backups/${BACKUP_ID}/neo4j"

# 恢复到空库或替换库
neo4j-admin database load "${NEO4J_DATABASE:-neo4j}" --from-path="backups/${BACKUP_ID}/neo4j" --overwrite-destination=true
```

Memgraph：按部署方式使用 snapshot 或 `mgconsole` dump，确保 workspace label / property 一并恢复。

### 5.3 Milvus / Qdrant 向量后端

Milvus：优先使用 Milvus Backup 工具或云厂商快照；若使用 standalone + MinIO，要同时备份 Milvus 元数据和其对象存储卷。

```bash
# 示例：Milvus Backup 工具，实际配置见 milvus-backup.yaml
milvus-backup create -n "${BACKUP_ID}"
milvus-backup restore -n "${BACKUP_ID}"
```

Qdrant：使用 collection snapshot。

```bash
# 替换 collection 名；按 workspace/collection 逐个 snapshot
curl -X POST "${QDRANT_URL}/collections/<collection>/snapshots"
curl -X POST "${QDRANT_URL}/collections/<collection>/snapshots/upload" \
  -F "snapshot=@backups/${BACKUP_ID}/qdrant/<snapshot>.snapshot"
```

### 5.4 Redis / Mongo / OpenSearch 后端

- Redis：启用 RDB/AOF 并在维护窗口复制 RDB/AOF；恢复时先停止 Redis，再替换数据文件或使用云备份恢复。
- MongoDB：使用 `mongodump --uri "$MONGO_URI" --archive=... --gzip`；恢复用 `mongorestore --drop --archive=... --gzip`。
- OpenSearch：配置 snapshot repository 后执行 `_snapshot/<repo>/<snapshot>`，恢复前关闭相关 index 写入。

## 6. 恢复后一致性检查

启动 API server 前先恢复 `.env` / compose / K8s Secret / StorageClass 等部署配置，再启动服务。启动后执行：

```bash
# 健康检查
curl -fsS "$LIGHTRAG_BASE_URL/health"

# metadata 与对象存储抽样：列 KB、列文档、列 artifact
curl -H "Authorization: Bearer <admin-token>" "$LIGHTRAG_BASE_URL/kbs"
curl -H "Authorization: Bearer <admin-token>" "$LIGHTRAG_BASE_URL/kbs/<kb_id>/documents"
curl -H "Authorization: Bearer <admin-token>" "$LIGHTRAG_BASE_URL/kbs/<kb_id>/documents/<doc_id>/artifacts"

# 引擎一致性：检查 KB status、graph、query
curl -H "Authorization: Bearer <admin-token>" "$LIGHTRAG_BASE_URL/kbs/<kb_id>/status"
curl -H "Authorization: Bearer <admin-token>" "$LIGHTRAG_BASE_URL/kbs/<kb_id>/graph/status"
curl -H "Authorization: Bearer <admin-token>" \
  -H 'Content-Type: application/json' \
  -d '{"query":"restore smoke test","mode":"mix"}' \
  "$LIGHTRAG_BASE_URL/kbs/<kb_id>/query"
```

也可以直接执行单机恢复 smoke 脚本，脚本会检查 `/health`、`/metrics`、KB/document/graph/query 采样，并把结果写成 JSON 报告：

```bash
uv run python scripts/run_single_server_ops_drill.py \
  --backup-id "$BACKUP_ID" \
  --base-url "$LIGHTRAG_BASE_URL" \
  --api-key "$LIGHTRAG_API_KEY" \
  --kb-id "<kb_id>" \
  --report-path "backups/${BACKUP_ID}/restore-smoke-report.json"
```

如需验证硬删除 workspace 复用清理一致性，可使用一次性 disposable KB（会创建、硬删除、重建并断言文档/图谱为空）：

```bash
uv run python scripts/run_single_server_ops_drill.py \
  --backup-id "$BACKUP_ID" \
  --base-url "$LIGHTRAG_BASE_URL" \
  --api-key "$LIGHTRAG_API_KEY" \
  --skip-query \
  --hard-delete-drill-kb-id "ops_hard_delete_drill_${BACKUP_ID}" \
  --report-path "backups/${BACKUP_ID}/hard-delete-drill-report.json"
```

必须确认：

- KB catalog 中的 `workspace` 未变化。
- 文档列表、job 列表、artifact 列表能打开。
- 对象存储中的 `source_object_uri` / artifact object URI 可恢复或下载。
- 图节点/边数量与恢复前演练记录大体一致。
- service API key 不需要重新明文导入：只要 `enterprise_api_keys` 表恢复，未撤销 key 仍可用；若怀疑泄漏，应恢复后立即撤销并重发。

## 7. 演练流程

建议至少每月做一次非生产恢复演练：

1. 在生产执行只读备份，记录 `BACKUP_ID`。
2. 在隔离环境恢复 metadata、对象存储和外部引擎后端。
3. 使用只读 service key 或管理员 JWT 执行第 6 节 smoke tests；推荐运行 `scripts/run_single_server_ops_drill.py` 并保存 JSON 报告。
4. 随机抽样 3 个 KB：下载 artifact、执行 query、检查 `graph/status` 与必要的 `graph/entities` / `graph/relations`。
5. 记录耗时、失败项、RPO/RTO 和需要自动化的步骤。

演练完成后把结果追加到团队运维日志，至少包含：`BACKUP_ID`、备份时间、恢复开始/结束时间、恢复环境、`restore-smoke-report.json` / `hard-delete-drill-report.json` 路径、通过/失败检查项、需要轮换的凭据。

---

## 8. 对象模式暂存与恢复 (Object-mode staging & resume)

> 适用范围：`LIGHTRAG_ARTIFACT_STORAGE_MODE=object`（对象权威生命周期）。本节描述的是**当前磁盘上已实现并冻结**的 Phase 3.2 object-backed 暂存行为（fix-S / fix-8，parent-accepted）。Phase 3 进度：**Gate 1 / Gate 2 已通过（PASS）**（Phase 3.1 destructive lifecycle + Phase 3.2 暂存/路由/迁移/协调全部 parent-accepted），Phase 3.3 进行中——health/metrics（fix-16）已 parent-accepted（见第 12 节），ops drill 待落地，Gate 3 未开。生产对象模式当前仍被能力常量 `OBJECT_AUTHORITATIVE_LIFECYCLE_IMPLEMENTED=False` 关闭，服务在对象模式下启动即被准入闸门拒绝（见第 11 节）。下列机制在能力常量翻转为 `True`、且 Gate 3 通过后才会实际生效。

### 8.1 为什么需要对象暂存

对象模式下 API 进程与 worker 进程可以运行在不同主机/checkout 上，**请求进程在提交（commit）前崩溃**时，原始请求体（replacement bytes）随进程消失。本地模式遇到这种情况会直接报 `replace_not_resumable`；对象模式则要求 replace/sync 在提交前先把字节落到一个**确定性、不可变、跨进程可见**的对象，使 durable worker 重启后能重新驱动同一状态机。

### 8.2 replace 暂存如何工作

1. 路由层在排队 replace job 之前，调用 `DocumentLifecycleService.stage_replacement_object(...)`：
   - 把 replacement 字节写入 canonical input root 下的 operation-scoped scratch 文件（`<INPUT_DIR>/<workspace>/.replace-staging-<document>-<job>.tmp`）；
   - 通过冻结的 `upload_file_if_absent(..., key=<COW 候选键>, expected_sha256=<source_hash>)` 上传；
   - 上传成功后立即删除 scratch 文件；
   - 返回的对象 URI 即为 `staging_object_uri`。
2. **COW 候选键是确定性的**：`workspaces/<workspace>/documents/<document>/source/generations/<generation>/<name>`——这正是后续 COW commit 要上传的同一个 key。因此 commit 阶段的 `upload_file_if_absent` 是幂等的（对象已存在则返回 `created=False`，校验 sha256 一致即可）。
3. job payload 仅持久化 `staging_object_uri`（对象 URI，**不含任何本地路径**）以及 `source_hash`、`size_bytes`、`source_name` 等元数据。

### 8.3 worker resume 如何从 pre-commit 崩溃恢复

durable worker 取到 replace job 后，在对象模式下按 `document.metadata.replace_phase` 与 payload 中的 `staging_object_uri` 分两种恢复路径：

| 恢复场景 | `replace_phase` | `staging_object_uri` | worker 行为 |
| -------- | --------------- | -------------------- | ----------- |
| 已提交、仅剩引擎清理 | `engine_cleanup_pending` | （可有可无） | 指针+manifest 已落库，源对象已是权威；只需重放引擎删除，**无需任何字节**。 |
| 提交前崩溃、但已暂存 | 非 `engine_cleanup_pending` | **存在** | 调用 `load_staged_replacement_object(...)` 下载暂存对象到 scratch，**校验 SHA-256 与 `source_hash` 一致**，重建 `DocumentReplacementSource`，再重放与路由相同的 `_execute_replace_document`。持久化的 `attempt_token` 使 COW 上传幂等。 |
| 提交前崩溃、未暂存 | 非 `engine_cleanup_pending` | **缺失** | 仍以 `replace_not_resumable` 干净失败——原始请求体已不存在，只能由调用方重新提交。 |

### 8.4 如何解读 job payload 中的 `staging_object_uri`

- 它是一个 `s3://`（或 MinIO endpoint）对象 URI，**不是本地路径**。若在作业详情、错误日志或审计里看到本地路径或 `.lightrag-scratch` 引用，应视为 durable-safe 违规并上报。
- 校验失败（下载到的字节 sha256 与持久化 `source_hash` 不符）会以 `replace_not_resumable` 失败，**绝不重放错误字节**。
- 暂存对象不可用（404）同样以 `replace_not_resumable` 失败；这种情况说明请求进程在暂存完成前就死了，需重新提交 replace。
- 暂存对象**不会被立即删除**：成功 replace 时它就是新 source 指针（保留）；失败/回滚时它由 cleanup-manifest 基础设施回收（见第 10 节）。手动删除暂存对象可能误删当前 source。

### 8.5 sync（聚合批处理）暂存

`sync` 路由走等价机制：每个待同步条目调用 `stage_sync_source_object(...)`，落到**确定性 per-batch、per-item** 暂存键 `workspaces/<workspace>/sync-staging/<batch_id>/<item_index:04d>/<safe_name>`（sync 可能创建全新文档，document_id/generation 在请求时未知，因此不复用 COW 候选键）。job payload 持久化 `staging_object_uris`（复数映射）。worker resume 用 `load_staged_sync_source_object(...)` 逐条下载并校验，再重放 per-item 同步助手按正常流程上传最终 source/artifact 对象。

---

## 9. 制品迁移到对象存储 (Migrate artifacts to object storage)

> 适用范围：把已存在的本地（legacy）制品目录迁移到对象存储权威。本节描述的是当前磁盘上已实现并冻结的迁移 CLI `lightrag-migrate-artifacts-to-object`（fix-M / fix-9，parent-accepted）。生产对象模式整体仍被能力常量关闭，但该迁移工具可独立运行用于准备对象存储内容。

入口（见 `pyproject.toml`）：

```
lightrag-migrate-artifacts-to-object = "lightrag.tools.migrate_artifacts_to_object:main"
```

迁移走冻结的 Phase 3.1-A `ArtifactMaintenanceRunRecord` / `ArtifactMaintenanceItemRecord` 维护运行基础设施，状态机为持久化、可恢复的 `planned → uploaded → applied → verified`。

### 9.1 三段式：plan / apply / resume

```bash
# 1) plan（默认 dry-run；无副作用，仅落库一份可审计的计划）
lightrag-migrate-artifacts-to-object \
  --working-dir ./rag_storage \
  --bucket lightrag-kb \
  --prefix kb \
  legacyA=/srv/rag/legacy-a legacyB=/srv/rag/legacy-b

# 输出形如：Migration plan created: mig-plan-<suffix>
# 记下 plan_id，进入 apply。

# 2) apply（必须显式带上 plan_id 与 --yes）
lightrag-migrate-artifacts-to-object \
  --working-dir ./rag_storage \
  --bucket lightrag-kb \
  --prefix kb \
  --plan-id mig-plan-<suffix> --yes \
  legacyA=/srv/rag/legacy-a legacyB=/srv/rag/legacy-b

# 3) 中断/崩溃后 resume（租约过期后重新认领）
lightrag-migrate-artifacts-to-object \
  --working-dir ./rag_storage \
  --bucket lightrag-kb \
  --prefix kb \
  --plan-id mig-plan-<suffix> --yes --resume \
  legacyA=/srv/rag/legacy-a
```

CLI 参数约束（违反则报错退出）：`--plan-id` 必须搭配 `--yes`；`--yes` 必须搭配 `--plan-id`；`--resume` 必须搭配 `--plan-id`。可加 `--json` 输出机读摘要，`--dry-run` 强制再跑一次 dry-run plan。

### 9.2 显式 `LABEL=/absolute/root`

- 每个根目录必须以 `LABEL=/absolute/root` 形式给出；`LABEL` 是 display-safe 的根标签，`/absolute/root` 是绝对本地路径。
- **绝不从 CWD、env、metadata 父目录推断根**；未显式给出则不迁移。
- apply/resume 时可以重新给出同一组 LABEL（甚至指向不同绝对路径——支持 moved-root 部署），工具按 LABEL 重新解析本地文件。

### 9.3 安全约束（fail-closed）

`_validate_absolute_root` 与 `_open_no_follow_and_validate` 在每个文件上强制：

- 拒绝相对路径、符号链接根、`..` 遍历段、网络/Windows URI、空白/控制字节；
- 拒绝设备文件、FIFO、socket；
- 目录 walk 为 no-follow，跳过 symlink 子树；
- 每个文件先 `lstat`（确认普通文件、非 symlink）→ `O_NOFOLLOW` 打开 → `fstat` 校验 inode/dev 一致后再流式 SHA-256。

所有审计/JSON 输出经过 redaction：scratch 路径、DSN、凭据、presigned URL、绝对本地根都被脱敏（`s3://` 对象 URI 保留）。持久化的 scope 只含根 LABEL 与其绝对路径的 SHA-256 指纹——**不含绝对路径明文**，因此 moved-root 部署可在 apply 时重新解析。

### 9.4 迁移对象键与文档指针

- 迁移目标对象键：`<prefix>/migrate/<root_label>/<relative_path>`。`migrate/` 段把迁移对象与在线写入的 `source/generations/` 名空间隔离开。
- `uploaded → applied` 阶段：可选地把文档的 `source_object_uri` 与 `source_generation_id`（形如 `mig_<sha256>`）通过 `update_document` 提交到 metadata。未绑定文档的条目仍会完成 verified 上传（例如 orphan-only 键）。
- `applied → verified` 阶段：调用**纯 metadata 的 `inspect_object(object_uri)`** 作为在场证明——对象必须存在且 size 匹配；不通过则 `blocked`。

### 9.5 迁移前检查清单

1. 对象存储（S3/MinIO）可达、bucket 存在、凭据来自 `LIGHTRAG_OBJECT_STORAGE_*` env 或 CLI 覆盖参数。
2. `--working-dir` 指向 canonical LightRAG 工作目录（sqlite 后端需包含 `metadata/metadata.sqlite3`）；postgres 后端用 `--metadata-backend postgres` + `--postgres-dsn`（或 env）。
3. **apply 期间禁止在线 KB 写**：工具会扫描 active job（`queued`/`running`/`cancelling`），存在即拒绝 apply（`Online KB mutation is in progress`），稍后重试。
4. 显式确认每个 `LABEL=/absolute/root`；不要把多个 LABEL 指向重叠的目录树（会因重复目标键报错）。
5. plan_id 来自一次 succeeded 的 dry-run；apply 才会真正上传并改写指针。

### 9.6 迁移后验证

- 查看输出摘要：`items_total / items_verified / items_skipped / items_failed / items_blocked`。`items_verified == items_total` 且 `failed=0` 时 apply run 转入 `succeeded`；否则 run 留在 `running`，需 `--resume` 继续推进。
- 抽样用 `inspect_object`（或 `mc stat` / `aws s3api head-object`）核对若干迁移对象的 size 与 sha256。
- 对绑定了文档的条目，确认控制面 metadata 中 `source_object_uri` 已指向 `<prefix>/migrate/...` 对象。
- 如出现 `blocked`（如 checksum_mismatch、object_readback_failed、metadata_pointer_commit_failed），查看 issues 列表里的 error_code，修复后 `--resume`。

---

## 10. 孤儿对象协调 (Orphan object reconciliation)

> 本节覆盖两条互补路径：在线路径在 COW 回滚时自动入队 `orphan_reconcile` manifest；离线路径用 `lightrag-reconcile-orphans` CLI 跨整个 bucket 前缀做计划/应用/恢复。两者最终都把删除交给 `ArtifactCleanupService` 的 verified-absence 流程——**没有任何路径会直接删除对象**。

### 10.1 在线路径：`orphan_reconcile` cleanup manifest

COW replace 在回滚（`fail_document_replace_cow`）一个已上传的候选对象时，会通过 `_enqueue_orphan_reconcile_compensation(...)` 入队一条 `orphan_reconcile / source` cleanup manifest（`document_lifecycle_service.py`）：

- manifest 组 ID 确定性：`orphan-cow-<sha256[:24]>`（基于 kb_id / kb_generation / document_id / job_id / attempt_token / 候选 URI）；
- 目标是被回滚的候选对象 URI，disposition=`delete`，初始 status=`pending`，按 grace window 安排 `delete_after`；
- 由 `ArtifactCleanupService` 在周期清理中处理；**授权要求**：该 manifest 的 `origin_attempt_token` 必须出现在该文档持久的 attempt-token 历史里——否则（无 token 的 orphan_reconcile manifest）cleanup 一律 block，永不自动授权。
- 在 `fail_document_replace_cow` 与本 enqueue 之间崩溃会留下一个**可恢复的不可变孤儿对象**（不是数据丢失路径），由 10.2 的 CLI 兜底回收。

### 10.2 离线路径：`lightrag-reconcile-orphans` CLI

入口（见 `pyproject.toml`）：

```
lightrag-reconcile-orphans = "lightrag.tools.reconcile_orphans:main"
```

该 CLI 是 `OrphanReconcileService`（`lightrag/api/orphan_reconcile_service.py`）的薄壳，复用冻结的 Phase 3.1-A 维护运行基础设施（`ArtifactMaintenanceRunKind="orphan_reconcile"` 的 run/item），同样走 dry-run-first、`--plan-id --apply --yes` 确认语义；输出经过 redaction（无 scratch 路径 / DSN / 凭据 / 绝对本地根）。

#### 10.2.1 三段式：plan / apply / resume

```bash
# 1) plan（默认 dry-run；无副作用，仅扫描+分类+落库一份可审计的计划）
lightrag-reconcile-orphans \
  --working-dir ./rag_storage \
  --bucket lightrag-kb

# 输出形如：Orphan reconcile plan created: or-plan-<suffix>
# 并按分类打印计数（eligible/referenced/retained/malformed/unknown_owner/too_new）。
# 记下 plan_id，进入 apply。

# 2) apply（必须显式带上 plan_id、--apply、--yes）
lightrag-reconcile-orphans \
  --working-dir ./rag_storage \
  --bucket lightrag-kb \
  --plan-id or-plan-<suffix> --apply --yes

# 3) 中断/崩溃后 resume（租约过期后重新认领）
lightrag-reconcile-orphans \
  --working-dir ./rag_storage \
  --bucket lightrag-kb \
  --plan-id or-plan-<suffix> --apply --yes --resume
```

CLI 参数约束（违反则 `parser.error` 退出）：`--plan-id` 必须同时搭配 `--apply` 与 `--yes`；`--apply` / `--yes` 必须搭配 `--plan-id`；`--release-retained` 必须搭配 `--plan-id --apply --yes`；`--resume` 必须搭配 `--plan-id`。`--dry-run` 强制再跑一次 dry-run plan（即使给了 `--plan-id --apply --yes`）。`--json` 输出机读 JSON 摘要。

完整选项（来自 `build_parser()`）：

| 选项 | 默认 | 说明 |
| ---- | ---- | ---- |
| `--working-dir` | （必填） | canonical LightRAG 工作目录，sqlite 后端需含 `metadata/metadata.sqlite3`。 |
| `--object-storage-endpoint` | env | S3/MinIO endpoint URL，覆盖 `LIGHTRAG_OBJECT_STORAGE_ENDPOINT`。 |
| `--bucket` | env / `lightrag-kb` | 目标 bucket，覆盖 `LIGHTRAG_OBJECT_STORAGE_BUCKET`。 |
| `--prefix` | `kb` | 对象键前缀。 |
| `--min-age-hours` | `24` | 最小对象年龄（小时）；小于该窗口的对象分类为 `too_new`，仅上报。设为 `0` 关闭年龄过滤。 |
| `--dry-run` | off | 强制 dry-run plan。 |
| `--plan-id ID` | — | 应用先前创建的计划（需配合 `--apply --yes`）。 |
| `--apply` | off | 确认 apply 意图（需配合 `--plan-id --yes`）。 |
| `--yes` | off | apply 最终确认（需配合 `--plan-id --apply`）。 |
| `--release-retained` | off | 释放指向被协调对象的 retained manifest；默认**不**释放（需配合 `--plan-id --apply --yes`）。 |
| `--resume` | off | 恢复进行中的 apply run（租约过期后重新认领）。 |
| `--metadata-backend` | env | `sqlite` 或 `postgres`；默认从 `LIGHTRAG_KB_METADATA_BACKEND` 推导。 |
| `--postgres-dsn` | env | postgres 后端 DSN。 |
| `--use-ssl` | env | 对 S3 endpoint 启用 TLS。 |
| `--json` | off | 输出机读 JSON 摘要。 |

#### 10.2.2 分类（六类，持久化在 maintenance item payload 上）

`OrphanReconcileService.create_plan()` 用有界 `list_objects_page`（单页 ≤1000，最多 32 页）扫描配置前缀，按下列顺序逐对象分类：

| 分类 | 含义 | apply 行为 |
| ---- | ---- | ---- |
| `eligible` | 孤儿，可回收 | 入队 `orphan_reconcile` cleanup manifest（disposition=`delete`）。 |
| `referenced` | 仍被当前 source/artifact/job/migration-item 引用，或 KB 处于 `deleting`/`deleted` | **跳过**（skip），不上报删除。 |
| `retained` | 当前有 `retained` cleanup manifest 持有该对象 | 仅当 apply 带 `--release-retained` 时释放该 manifest；否则跳过。 |
| `malformed` | 对象键无法解析为已校验的归属（如未归一化、无 namespace） | **仅上报，永不入队**。 |
| `unknown_owner` | workspace 对应的 KB 不存在、workspace 不匹配、或存在 unknown commit-outcome | **仅上报，永不入队**。 |
| `too_new` | 对象 last-modified 在 `--min-age-hours` 窗口内 | **仅上报，永不入队**。 |

apply 阶段还会**再次校验** `eligible` / `retained` 分类（`_revalidate_apply_target`）：若 plan 与 apply 之间出现新的 live reference / pending manifest / unknown commit-outcome，则改为 skip 而不是盲入队。staging namespace 的条目一律 skip（staging 清理需要 job+attempt 授权，orphan reclaim 不具备）。

#### 10.2.3 apply 永不直接删除

`apply_plan(plan_id, *, release_retained, resume)` 对每个 `eligible` 条目调用 `enqueue_artifact_cleanup_manifest(...)`（reason=`orphan_reconcile`，target_namespace 来自对象键解析，`delete_after` = `now + max(min_age_hours,1)h`，`cleanup_deadline_at` = `delete_after + 24h`，`audit_retain_until` = `now + 30d`），把条目置为 `verified`。**真正的对象删除仍由 `ArtifactCleanupService` 在 verified-absence（HEAD / 有界 list 证明缺席）后执行**——CLI 自身没有任何删除 API 调用。`retained` 条目（仅当 `--release-retained`）调用 `release_retained_artifact_cleanup_manifests`，同样不直接删除。

#### 10.2.4 何时运行

- 大版本切换、迁移 apply 异常中断之后；
- 怀疑 `fail_document_replace_cow` 与 manifest enqueue 之间发生过崩溃（留下不可变孤儿）；
- 周期性对账（例如每周边一次 dry-run plan，对比 `eligible` 计数趋势）。

#### 10.2.5 apply 后核对

- 摘要看 `items_total / items_enqueued / items_skipped / items_blocked / items_failed`。`items_enqueued + items_skipped == items_total` 且 `failed=0` 时 apply run 转 `succeeded`；否则 run 留 `running`，需 `--resume` 推进。
- 入队的 manifest 由 cleanup 服务处理；可用 cleanup manifest 视图（metadata store API 或直接查 `artifact_cleanup_manifests` 表）过滤 `reason='orphan_reconcile'` 看 `status` 分布（`pending`/`leased`/`blocked`/`succeeded`）。
- `blocked` 的孤儿 manifest 多半是 attempt-token 历史不匹配——人工核对，**不要绕过 cleanup 服务手工删除对象**。

---

## 11. 路由策略 (Route policy)

> 本节说明对象模式下 destructive 路由的准入行为：能力常量 + 三态策略 + per-KB 路由白名单 + legacy 路由永久禁用。代码载体：`lightrag/api/config.py`（`load_object_route_policy_from_env` / `object_route_policy_allows` / `OBJECT_ROUTE_POLICY_*` 常量）与 `lightrag/api/routers/kb_document_routes.py`（`_require_destructive_lifecycle` 三态、`_reject_legacy_route_in_object_mode`）。

### 11.1 三态准入策略（`_require_destructive_lifecycle`）

对象模式下，所有 per-KB destructive 路由（sync / replace / delete / batch-delete 等）在路由层都经过 `_require_destructive_lifecycle(document_service, operation, *, kb_id, route_operation)`，按**三态**判定：

| 状态 | 条件 | 行为 |
| ---- | ---- | ---- |
| ① 本地模式 | `not document_service.object_authoritative` | 直接放行，与白名单无关。 |
| ② 对象模式 + 能力常量 `False` | `OBJECT_AUTHORITATIVE_LIFECYCLE_IMPLEMENTED is False`（**当前磁盘状态**） | 调用 `assert_destructive_operation_supported(...)`，对象模式下抛 `HTTPException(503)`。白名单**不**在此路径上读取。 |
| ③ 对象模式 + 能力常量 `True` | Gate 3 之后代码审翻转 | 读 `load_object_route_policy_from_env()` + `object_route_policy_allows(policy, kb_id, route_operation)`：白名单允许则放行，否则 `HTTPException(403)`。 |

更外层还有一道**服务启动准入**（`config.py: validate_artifact_storage_server_admission`）：当 `LIGHTRAG_ARTIFACT_STORAGE_MODE=object` 且能力常量 `OBJECT_AUTHORITATIVE_LIFECYCLE_IMPLEMENTED=False` 时，服务**直接拒绝启动**。当前磁盘状态下对象模式根本无法启动服务；destructive 路由的 503 闸门是第二道防线。**即：能力常量为 `False` 时，所有对象模式 destructive 路由一律 503，白名单不会改变这一行为。**

### 11.2 能力常量 vs 路由策略（两件事）

| 层 | 载体 | 粒度 | 何时翻转 |
| ---- | ---- | ---- | -------- |
| 能力常量 | `OBJECT_AUTHORITATIVE_LIFECYCLE_IMPLEMENTED`（`lightrag/api/config.py` 顶部代码常量，**不是配置项**） | 全局开/关 | Gate 3 通过后代码审改为 `True`；翻转后需按规范重跑回归。不接受 env 覆盖。 |
| 路由策略 | `LIGHTRAG_OBJECT_ROUTE_POLICY`（env，JSON per-KB 白名单） | 每个 KB 哪些 object-mode destructive 路由可用 | 代码已交付（fix-10），但在能力常量翻 `True` 之前**运行时不可达**（被三态 ② 截断）。 |

两者**相互独立**：即便能力常量为 `True`，未进白名单的 destructive 路由仍返回 403；能力常量为 `False` 时白名单则根本不参与判定（一律 503）。

### 11.3 `LIGHTRAG_OBJECT_ROUTE_POLICY` 配置形态

JSON 对象，键为 KB ID 或全局键 `"*"`（`OBJECT_ROUTE_POLICY_GLOBAL_KEY`），值为操作 token 列表。`object_route_policy_allows` 把 `*` 与具体 KB 的集合并集后判定。

```json
{"*": ["replace"], "kb_abc": ["sync"]}
```

- 合法 token（`OBJECT_ROUTE_POLICY_OPERATIONS`）：`upload`、`parse`、`build`、`replace`、`delete`、`batch_delete`、`sync`、`hard_delete`。token 大小写不敏感、去空白；**未知 token 不会导致启动失败**，只会被丢弃并记 warning（admission 必须单独 fail-closed）。
- 解析失败（非 JSON、非对象、值非字符串/列表）→ 返回空 dict（**fail closed**：所有 destructive 路由 403）。
- legacy 本地路径变更路由（`documents:texts` / `documents:urls` / `documents:import` / `documents:scan`）**刻意不在 `OBJECT_ROUTE_POLICY_OPERATIONS` 内**：它们在对象模式下由 `_reject_legacy_route_in_object_mode` 永久禁用（能力常量 `False` 时 503，`True` 时永久 403），**永不可进入白名单**。
- 全局键 `"*"` 表示“所有 KB 默认放行这些操作”；某个 KB 想收紧/放开时单独列键，与 `*` 取并集。默认（变量未设/空）即所有对象模式 destructive 路由都不放行。

### 11.4 对象模式前置条件（与路由策略无关）

`validate_artifact_storage_configuration`（任一不满足即拒绝启动）：

- `LIGHTRAG_KB_METADATA_BACKEND=postgres`；
- `LIGHTRAG_OBJECT_STORAGE` ∈ {s3, minio} 且可用；
- `LIGHTRAG_ENTERPRISE_AUTH_ENABLED=true`；
- `LIGHTRAG_ENTERPRISE_DISABLE_GLOBAL_ROUTES=true`（legacy/global 变更路由在对象模式下永久禁用）；
- 存在 canonical `INPUT_DIR` 且支持 POSIX `fcntl` 锁、可写 scratch。

### 11.5 运维含义

- 升级到对象模式前，先确认本批次代码里能力常量是否已翻转（grep `OBJECT_AUTHORITATIVE_LIFECYCLE_IMPLEMENTED`）；未翻转就启动对象模式会直接报错退出。
- 不要试图用 env 绕过能力常量；它不是配置。
- 能力常量翻转后，开放 destructive 路由应按 KB 灰度：先在演练 KB 登记白名单（例如 `LIGHTRAG_OBJECT_ROUTE_POLICY={"*":["replace"],"kb_drill":["sync","delete"]}`），验证 sync/replace/delete/hard-delete 状态机，再逐步放开。
- 配置出错（JSON 解析失败）只会让 admission 收紧到“全 403”，不会让 admission 放行；错误的白名单条目会被丢弃并记 warning，可在日志里搜 `Object route policy`。

---

## 12. 可观测性与健康检查 (Observability & health checks)

> 本节说明 `GET /health` 中对象权威生命周期相关的两个块。代码载体：`lightrag/api/lightrag_server.py`（`_build_artifact_lifecycle_health_block` 与 `/health` 路由）。

### 12.1 两个块：`artifact_lifecycle`（新）与 `artifact_cleanup`（legacy）

`/health` 同时返回两个兄弟块，**互不影响**：

- `artifact_lifecycle`（Phase 3.3，fix-16，parent-accepted）：有界索引化聚合 + 缓存的 HeadBucket 探针。**无条件输出**（本地模式也输出），形状稳定，适合 dashboard 直接采集。
- `artifact_cleanup`（legacy，保留不变）：`{enabled, worker_running, pending_count}`，向已存在的断言向后兼容。

### 12.2 `artifact_lifecycle` 块字段

```json
{
  "artifact_lifecycle": {
    "mode": "local | object",
    "backend": "none | disabled | s3",
    "capability_admitted": {
      "implemented": false,
      "admission_gate_allows_object_mode": false
    },
    "object_store_ready": false,
    "manifests": {
      "total": 0, "retained": 0, "pending": 0, "leased": 0,
      "blocked": 0, "succeeded": 0,
      "due_pending": 0, "expired_leases": 0,
      "cleanup_deadline_overdue": 0,
      "oldest_due_at": null
    },
    "maintenance_runs": 0,
    "migration_blockers": 0,
    "unresolved_commit_unknown": 0,
    "recovery_cursor_stale": 0
  }
}
```

逐字段解读：

| 字段 | 含义 / 如何解读 |
| ---- | --------------- |
| `mode` | 当前 `LIGHTRAG_ARTIFACT_STORAGE_MODE`（`local` 或 `object`）。 |
| `backend` | 对象存储后端稳定标签：`none`（本地模式，未构造对象存储）、`disabled`（`DisabledObjectStorage`）、`s3`（`S3ObjectStorage`）。**不含 endpoint / bucket / 凭据**。 |
| `capability_admitted.implemented` | 代码能力常量 `OBJECT_AUTHORITATIVE_LIFECYCLE_IMPLEMENTED` 的镜像；当前 `false`，Gate 3 翻转后变 `true`。 |
| `capability_admitted.admission_gate_allows_object_mode` | 启动准入闸门（`validate_artifact_storage_server_admission`）是否放行对象模式。当前与 `implemented` 相等（常量 `false` 时对象模式拒绝启动）；Gate 3 解耦后两者可能独立。 |
| `object_store_ready` | 缓存的 HeadBucket 探针（S3 上约 30s TTL，单调时钟缓存）。`true` 表示 bucket 可达；`false` 可能是后端不可达 / 鉴权失败 / 本地模式 / 探针超时。**永不抛异常**；任何传输/鉴权错误都坍缩为 `false`。 |
| `manifests` | `artifact_cleanup_manifests` 表的有界单行聚合。`total/retained/pending/leased/blocked/succeeded` 是状态计数；`due_pending` 是已到 `delete_after` 且 `next_attempt_at` 已到的 pending；`expired_leases` 是租约过期的 leased；`cleanup_deadline_overdue` 是越过 SLO 截止时间的非成功 manifest。 |
| `manifests.oldest_due_at` | `MIN(delete_after)` over `pending/leased`（ISO UTC 字符串或 `null`）。表示“下一次 cleanup 周期最该先处理的对象到期时间”，可据此判断 cleanup 是否积压。 |
| `maintenance_runs` | 非终态维护运行总数（迁移 / orphan_reconcile），跨 `planned / running / waiting_cleanup` 三态求和。`>0` 说明有迁移或孤儿协调在途，此时不应启动新的迁移 apply。 |
| `migration_blockers` | 全局活跃变更 job 计数（状态 `queued / running / retrying / cancelling`）——与迁移 CLI 的在线变更 guard 同一活跃集（fix-15 补了 `retrying`）。`>0` 时迁移 apply 会被拒绝。 |
| `unresolved_commit_unknown` | 见第 13 节。`error_code='metadata_commit_outcome_unknown'` 的 job 数；`>0` 需人工裁定。 |
| `recovery_cursor_stale` | `updated_at` 早于 `now - 6h` 的 artifact recovery cursor 数（`_ARTIFACT_LIFECYCLE_RECOVERY_CURSOR_STALE_SECONDS = 6*60*60`）。`>0` 说明 KB hard-delete drain 或 generation recovery sweep 卡住，需检查 worker / recovery cursor 行。 |

### 12.3 `"not_reported"` 的含义

聚合 COUNT 字段（`manifests / maintenance_runs / migration_blockers / unresolved_commit_unknown / recovery_cursor_stale`）的探针都经 `_bounded_health_value(..., timeout=2.0s)` 包裹。当出现下列任一情况时，该字段返回字符串 `"not_reported"`（而非数字）：

- 查询超过 2 秒（慢存储 / 锁竞争）；
- 查询抛异常（存储不可达、连接断开等）；
- metadata store 缺少对应方法（旧后端 / 测试 double）。

`object_store_ready` 不同：它坍缩为布尔 `false`（不是 `"not_reported"`），因为 dashboard 通常按布尔判定 readiness。

**`"not_reported"` 不等于“零”**：它表示“当前无法快速取到该指标”。告警规则应区分 `==0`（健康）与 `=="not_reported"`（探针超时，需排查存储健康）。`/health` 顶层 `status` 不会因为这些字段变 `"not_reported"` 而变成 unhealthy——这些是诊断信号，不是存活判定。

### 12.4 `/health` 永不列举 bucket 或下载对象

`artifact_lifecycle` 块的设计红线：

- `object_store_ready` 是**单次 `head_bucket`**（metadata-only reachability check），不列举对象键、不下载字节；
- 其余字段都是 metadata store 上的聚合 SQL（COUNT / MIN），不触碰对象存储；
- 因此即便 bucket 极大、对象极多，`/health` 延迟也有界（每个探针 ≤2s，HeadBucket 探针有 TTL 缓存）。

如需查看具体 manifest 或对象，用第 9 / 10 节的 CLI 或直接查 `artifact_cleanup_manifests` 表，不要试图从 `/health` 拉清单。

### 12.5 典型运维判读

- **对象模式 readiness**：`object_store_ready=false` 持续出现 → 检查 MinIO/S3 endpoint、凭据、网络；`backend=s3` 但 `object_store_ready=false` 是后端不可达的强信号。
- **cleanup 积压**：`manifests.due_pending` 或 `manifests.cleanup_deadline_overdue` 持续增长，且 `artifact_cleanup.worker_running=true` → cleanup 跟不上，考虑调大 `LIGHTRAG_ARTIFACT_CLEANUP_CLAIM_LIMIT` / `MAX_CONCURRENT_MANIFESTS`，或排查对象存储延迟。
- **租约泄漏**：`manifests.expired_leases` 长期 `>0` → 上一轮 cleanup worker 崩溃，`ArtifactCleanupService` 会在下个周期通过 `LIGHTRAG_ARTIFACT_CLEANUP_EXPIRED_LEASE_RECOVERY_LIMIT` 回收。
- **迁移阻塞**：`migration_blockers>0` 时不要尝试 `lightrag-migrate-artifacts-to-object --plan-id ... --yes`（会被在线变更 guard 拒绝）；等 `migration_blockers=0` 再 apply。
- **drain 卡住**：`recovery_cursor_stale>0` → 某个 KB hard-delete drain 或 generation recovery 停滞，结合 `/kbs/<kb_id>/status` 与 `artifact_recovery_cursors` 表定位。

---

## 13. 提交未知恢复 (Commit-unknown recovery)

> 对应 `artifact_lifecycle.unresolved_commit_unknown` 字段与 `jobs.error_code='metadata_commit_outcome_unknown'` 持久化哨兵。代码载体：`metadata_store.count_unresolved_commit_unknown_jobs()`、`document_lifecycle_service` 的 commit 分类、`orphan_reconcile_service` / cleanup blocker 的同一分类。

### 13.1 什么是 commit-unknown

对象权威生命周期要求所有 destructive 路径**先提交 metadata（指针 / tombstone / manifest），再删除对象字节**。当一次 replace / sync / delete 的 metadata commit 调用返回**模糊结果**（连接在 ACK 后断开、超时但 commit 可能已落库、网络层重置等），系统无法确认指针是否真正落库。此时 job 的 durable `error_code` 被置为 `metadata_commit_outcome_unknown`，而不是 `succeeded` 或确定的 `failed`。

这是**刻意保留两代**的安全分类（与 COW 回滚中 `ROLLED_BACK` 前失败的补偿语义一致）：在模糊状态下既不补偿、也不失败、也不删除对象，避免推进 cleanup 误删当前 source 或新提交的指针。

### 13.2 `/health` 如何暴露

```json
{ "artifact_lifecycle": { "unresolved_commit_unknown": 2 } }
```

`unresolved_commit_unknown > 0` 表示存在至少一个 `error_code='metadata_commit_outcome_unknown'` 的 job。该字段是 `jobs` 表上 `error_code` 列的索引化 COUNT（单条 `SELECT COUNT(*) FROM jobs WHERE error_code = ?`），不读取 job payload，不列举对象。

### 13.3 系统永不自动恢复

**没有任何自动化路径会改写 `metadata_commit_outcome_unknown` 的终态**：

- `ArtifactCleanupService` 把它视作 cleanup blocker——一个 job 处于 commit-unknown 时，相关 cleanup manifest 不会推进删除；
- `OrphanReconcileService` 把对应对象分类为 `unknown_owner`（仅上报，永不入队）；
- 没有定时 reconciler 会把它翻成 `succeeded` 或 `failed`。

这是因为自动“猜”commit 是否落库会破坏 destructive lifecycle 的安全保证：猜错一次就可能误删当前 source，或留下指向已删对象的指针。**必须由人工裁定**。

### 13.4 运维处置流程

当 `/health` 的 `unresolved_commit_unknown > 0` 时：

1. **定位 job**：直接查 metadata store（不要从 `/health` 找，它只给计数）：

   ```sql
   -- PostgreSQL 控制面
   SELECT id, kb_id, document_id, job_type, status, error_code,
          created_at, updated_at
   FROM kb_jobs
   WHERE error_code = 'metadata_commit_outcome_unknown';
   ```

   SQLite 控制面等价查询 `jobs` 表（`<working_dir>/metadata/metadata.sqlite3`）。

2. **裁定 commit 是否落库**：检查对应 document 的当前指针状态（`source_object_uri` / `source_generation_id` / artifact 列表 / generation fence）。若新指针已存在且 generation 与本次 job 的 attempt 一致，说明 commit 实际已成功——job 应**手动重试**（重新驱动后续步骤）或标记为已成功。若指针仍是旧值，说明 commit 未落库——job 应**手动标记失败**并由调用方重新提交。
3. **不要直接删除对象**：在裁定完成前，相关对象（候选 / 旧 source / staging）都应保留。commit-unknown 的对象会被 orphan reconcile 分类为 `unknown_owner` 而非 `eligible`，不会被自动回收。
4. **处置后**：把 job 的 `error_code` 清除并置为合适的终态（`succeeded` 或 `failed`），`/health` 的 `unresolved_commit_unknown` 计数会在下次聚合（≤探针 TTL）下降。

### 13.5 预防

- 确保 metadata backend（PostgreSQL / SQLite）网络稳定，metadata commit 调用的客户端超时足够长（避免 ACK 后断连的模糊态）。
- 监控 `unresolved_commit_unknown`：任何 `>0` 都应触发告警，因为系统不会自行消化它。
- 升级 / 重启 worker 前先确认 `unresolved_commit_unknown == 0`，避免在模糊态上叠加新的写入。
