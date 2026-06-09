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
