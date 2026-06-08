# Enterprise KB MVP API Demo

This folder contains a Windows-friendly API orchestration script for simulating an
enterprise knowledge-base rollout without the WebUI.

The script drives the KB API end-to-end:

1. health check and runtime storage summary;
2. create or reuse an isolated KB (`/kbs`);
3. create and activate a KB config snapshot with selected sanitized values from
   the local `.env`;
4. ingest files from `E:\pycharmprojects\RAG\LightRAG\模拟文件`;
5. persist source/artifacts through MinIO/S3 when enabled;
6. parse files with MinerU/native routing;
7. build KG/index, including entity/relation extraction and vector writes;
8. inspect documents, jobs, dead letters, artifacts, graph status, entities, relations;
9. optionally run an interactive document deletion test;
10. run KB-scoped RAG query and structured retrieval query;
11. create an empty control KB to verify workspace isolation.

Beyond this baseline (~26 endpoints), opt-in flags exercise many more endpoints —
rebuild/reindex, document replace, text/URL ingest, streaming + structured
retrieval, config get/diff, KB metadata patch, artifact metadata/download-url, and
document enable/disable. See "Extended endpoint coverage" below.

Run from the repository root:

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

Useful repeat-run flags:

- `--skip-ingest`: reuse the existing KB and only inspect/query.
- `--skip-query`: ingest/build only.
- `--manual-flow`: use explicit upload -> parse -> build-kg calls instead of
  `documents:sync?auto_parse=true&auto_index=true`.
- `--max-files 1`: quick smoke run against one file.
- `--run-id stable-id`: use a stable idempotency key/report suffix. Only reuse
  the same run id for an exact retry of the same file set; use a fresh run id
  after adding or changing files.
- `--reset-kb ask|yes|no`: optionally hard-delete the main KB and the isolation
  control KB before the run. The default `ask` prompts only in an interactive
  terminal and skips reset in non-interactive shells to avoid accidental data
  loss. `yes` resets without prompting; `no` always skips. Reset calls
  `DELETE /kbs/{kb_id}?hard=true`, so it clears the KB metadata rows,
  LightRAG workspace files, parser input/artifact cache, and MinIO/S3 objects
  associated with the KB workspace.

## Running in enterprise mode (multi-user / RBAC / ACL / audit)

When `.env` sets `LIGHTRAG_ENTERPRISE_AUTH_ENABLED=true` the server runs in
**enterprise mode**: guests are disabled, the global `LIGHTRAG_API_KEY` can no
longer bypass RBAC by default, `/kbs` routes are guarded by KB roles, and a
super admin is bootstrapped on first startup. The script **auto-detects** the
auth mode and authenticates accordingly; the core command line is unchanged.

### 1) `.env` prerequisites

```env
LIGHTRAG_ENTERPRISE_AUTH_ENABLED=true
# Enterprise mode requires a non-default TOKEN_SECRET (not "please-change-me")
TOKEN_SECRET=<long-random-secret>
# Super admin: created/synced on first startup; prefer PASSWORD_HASH in production
LIGHTRAG_SUPER_ADMIN_USERNAME=admin
LIGHTRAG_SUPER_ADMIN_PASSWORD=Admin@12345
# Production: LIGHTRAG_SUPER_ADMIN_PASSWORD_HASH={bcrypt}$2b$12$... (lightrag-hash-password)
```

Optional (all safe defaults, enable as needed): registration mode
`LIGHTRAG_USER_REGISTRATION_MODE` (`disabled/open/invite_only/admin_approval`),
failed-login lockout `LIGHTRAG_ENTERPRISE_LOGIN_*`, concurrent-job quota
`LIGHTRAG_ENTERPRISE_MAX_CONCURRENT_JOBS` / `..._TENANT_MAX_CONCURRENT_JOBS`,
request rate-limit/quota `LIGHTRAG_ENTERPRISE_RATE_LIMIT_*`, and artifact
download minimum role `LIGHTRAG_ENTERPRISE_ARTIFACT_DOWNLOAD_MIN_ROLE`.

### 2) How the script authenticates

On startup the script calls `GET /auth-status`:

- **Enterprise mode** → it logs in as the super admin via `POST /login`, takes
  the JWT, and switches to `Authorization: Bearer <token>` (dropping the
  `X-API-Key` header). The super admin passes every KB role check, so the
  baseline flow (create KB / sync / parse / build / query / hard delete) is
  **unchanged**.
- **Non-enterprise mode** → it keeps using `X-API-Key` (backward compatible).

Super-admin password resolution order: `--admin-password` > the
`LIGHTRAG_SUPER_ADMIN_PASSWORD` environment variable > the
`LIGHTRAG_SUPER_ADMIN_PASSWORD` value in `--env-file` (default `.env`). So a
configured `.env` works without re-passing the password on the command line. If
the server only has `PASSWORD_HASH` (no plaintext), login needs plaintext —
pass it explicitly via `--admin-password`.

### 3) Run (enterprise mode)

Start the server (reads `.env`, bootstraps the super admin on first startup):

```powershell
lightrag-server
```

Run the drill (auto enterprise login; password falls back to `.env`):

```powershell
uv run python examples/enterprise_kb_mvp/enterprise_kb_mvp_demo.py `
  --server "http://127.0.0.1:9621" `
  --source-dir "E:/pycharmprojects/RAG/LightRAG/模拟文件" `
  --kb-id enterprise_mvp_demo `
  --reset-kb yes
```

To supply the super-admin credentials explicitly instead of via `.env`:

```powershell
uv run python examples/enterprise_kb_mvp/enterprise_kb_mvp_demo.py `
  --admin-username admin --admin-password "Admin@12345" --reset-kb yes
```

### 4) Enterprise control-plane showcase (opt-in `--demo-enterprise-admin`)

Enterprise mode only, off by default. When enabled it adds a final block that
demonstrates the enterprise capabilities (all safe / reversible):

- create a normal user + grant it `kb_viewer` ACL on the demo KB
  (`POST /admin/users`, `PUT /admin/kbs/{kb}/acl`);
- issue a **scoped, expiring service API key** (`POST /admin/service-api-keys`
  with `expires_in_seconds`) and verify with `GET /kbs` that the key only sees
  authorized KBs;
- mint a **single-use registration invitation** (`POST /admin/invitations`, for
  `invite_only` registration); the raw token is returned only once;
- read audit events (`GET /admin/audit-events`).

```powershell
uv run python examples/enterprise_kb_mvp/enterprise_kb_mvp_demo.py `
  --kb-id enterprise_mvp_demo --reset-kb yes --demo-enterprise-admin
```

### 5) Behavior changes and notes

- In enterprise mode the global `LIGHTRAG_API_KEY` **cannot** access `/kbs`;
  legacy scripts/frontends must switch to super-admin login or a service key.
- Hard delete (`--reset-kb yes` → `DELETE /kbs/{id}?hard=true`) is
  **super-admin-only**; the script can do it because it logs in as super admin.
- **Failed-login lockout is on by default**: repeated failures for the same
  username (default 10 within 300s) return `429` and lock the username for the
  lockout window (default 900s). If you get locked out from wrong passwords,
  wait it out or tune/disable `LIGHTRAG_ENTERPRISE_LOGIN_MAX_ATTEMPTS`.
- `.env` is typically git-ignored, so changes are local only; in production
  replace the plaintext super-admin password with
  `LIGHTRAG_SUPER_ADMIN_PASSWORD_HASH` and remove the plaintext line.

## Extended endpoint coverage (opt-in)

The baseline flow already exercises ~26 KB endpoints. These flags turn on
additional, otherwise-uncovered endpoints. They are all **off by default** so the
baseline run is unchanged, and every job they create is followed without a client
timeout (see "Long-running jobs never time out" below).

- `--demo-extras`: read-mostly, reversible endpoints run after the main query —
  - `POST /kbs/{id}/retrieve` — structured retrieval, no LLM generation;
  - `POST /kbs/{id}/query/stream` — NDJSON streaming (the report records token count);
  - `GET /kbs/{id}/graph` — subgraph export;
  - `GET /kbs/{id}/configs/{version_id}` + `POST .../{version_id}:diff` — inspect a
    config version and diff it against the active one;
  - `PATCH /kbs/{id}` — description round-trip (patched, then restored);
  - `GET .../artifacts/{artifact_id}` + `:download-url` — artifact metadata and a
    presigned download URL;
  - document `:disable` -> `PATCH` metadata -> `:enable` round-trip;
  - `POST /kbs/{id}/jobs/{job_id}:retry` — retries the first dead-letter job, if any.
- `--demo-reindex`: rebuild paths — per-document `:reindex`, `documents:batch-reindex`,
  and `{kb}:rebuild`. These re-run chunk/extract/embedding (can be slow) and
  end-to-end verify that vector rebuild still works after the in-house
  `_VDBUpsertBatcher` was replaced by upstream's storage-layer delayed embedding.
- `--demo-replace FILE`: replace the first ready document's source via
  `POST /kbs/{id}/documents/{document_id}:replace` (multipart upload + durable
  resume). `FILE` is the new source file.
- `--demo-ingest-variants` (+ optional `--demo-url URL`): non-file ingest channels
  `documents:texts` (a synthetic text doc) and `documents:urls` (only when
  `--demo-url` is given). Both run `auto_parse` + `auto_index`.

Enable everything read-mostly plus the rebuild paths and text ingest:

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

To also exercise document replacement, add `--demo-replace path/to/new_file.pdf`.
To also fetch a URL, add `--demo-url https://example.com/page` with
`--demo-ingest-variants`.

## Long-running jobs never time out

Every ingest/parse/build/reindex/rebuild/replace endpoint returns an async job.
The client follows each job with `wait_for_job`, which issues a bounded
server-side `:wait` (default 120s window) and, on its 408 heartbeat, re-queries
progress and re-issues — so a slow MinerU parse or a multi-PDF KG build that runs
for hours keeps printing progress instead of being abandoned. The default
`--job-timeout 0` means "follow until the job terminates"; the server keeps
running regardless, so giving up early would only orphan a live job. Pass a
positive `--job-timeout SECONDS` only when you deliberately want the client to
bail out early.

The extended endpoints above reuse the exact same follow mechanism via
`follow_job_response`, which transparently handles both shapes the API returns —
a `JobResponse` (`id` + `status`) and a `DocumentBatchResponse` (`job_id`) — and
treats an empty `{kb}:rebuild` no-op (blank `job_id`) as already done.

## What is persisted

The MVP intentionally exercises all production-facing storage layers:

- PostgreSQL/metadata backend: KB record, active config version, document
  metadata, job records, artifact metadata, source hashes, parser/index hashes,
  and idempotency keys.
- MinIO/S3 object storage: uploaded source files and parser artifacts when object
  storage is enabled in `.env`.
- Milvus/vector backend: chunk/entity/relation embeddings generated during KG and
  vector index build.
- LightRAG workspace storage: per-KB workspace data under the configured
  `WORKING_DIR`, including KV/doc-status/graph/cache structures used by the
  LightRAG engine.
- Local input/cache paths: server-side files under the configured input/cache
  directories when the backend uses local staging or parsed artifacts.

## Drill coverage and limitations

This script is an end-to-end integration drill against **real backends**: it
drives a real running API server (not a test stub / FakeRAG) and writes through
whatever backends `.env` actually points at. What it can positively verify
therefore depends on the server's storage configuration.

External backends exercised under the current `.env` profile (since 2026-06-04;
see the report's `env_snapshot`):

- ✅ PostgreSQL (control plane) — KB metadata (`LIGHTRAG_KB_METADATA_BACKEND=postgres`).
- ✅ Milvus — chunk/entity/relation vectors (`LIGHTRAG_VECTOR_STORAGE=MilvusVectorDBStorage`).
- ✅ MinIO/S3 — source files and parser artifacts (`LIGHTRAG_OBJECT_STORAGE=minio`).
- ✅ Real LightRAG engine + real MinerU / LLM / embedding / rerank end to end.
- 🆕 PostgreSQL (engine) — KV + doc_status (`LIGHTRAG_KV_STORAGE=PGKVStorage` /
  `LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage`), switched from the Json file
  backend on 2026-06-04.
- 🆕 Neo4j — knowledge graph (`LIGHTRAG_GRAPH_STORAGE=Neo4JStorage`), switched from
  the NetworkX file backend on 2026-06-04; KBs are isolated by workspace node label.

Prerequisites and confirmation for the 🆕 items (handle before the first run on the
new `.env`):

- **Rebuild data**: switching backends does not migrate old data — the prior
  NetworkX/Json files under `rag_storage/` are invisible to the new backends, so
  each KB needs one full rebuild (re-ingest / `:rebuild`) to populate Neo4j + PG.
- **Install the driver**: the Neo4j driver is an optional dependency — run
  `uv pip install "neo4j>=5,<7"` (or `uv sync --extra offline-storage`) first, or
  the server fails to start.
- **Confirmation scope**: the three ✅ items (PG control plane / Milvus / MinIO) are
  confirmed by prior runs; the 🆕 items (engine PG, Neo4j) need one full rebuild run
  on the new `.env` to confirm their hard-delete cleanup consistency and isolation —
  at which point the assertions below run against Neo4j + PG for real.

> Other backends (Redis / MongoDB / Qdrant, or reverting to file-based
> `JsonKVStorage` / `NetworkXStorage`) require a separate run under that profile.

Built-in pass/fail assertions (any miss exits non-zero):

- Isolation: the primary KB and the empty control KB share no document ids; the
  control KB has no documents and a query against it returns **zero references**
  (a positive check that the shared vector/graph backends honor the workspace
  boundary).
- Object persistence: every ready document must carry `metadata.source_object_uri`
  (the source really landed in MinIO/S3).
- Hard-delete cleanup consistency: when `--reset-kb` actually runs, the recreated
  KB's reused workspace must report zero documents / graph nodes / graph edges
  (a positive regression for the "hard-delete residue + workspace reuse reads
  stale data" defect).
- Every ingest/parse/build/reindex/replace step raises and aborts on failure.

## Per-KB parameter persistence

Yes. The script creates a KB config version with `POST /kbs/{kb_id}/configs` and
activates it with `POST /kbs/{kb_id}/configs/{config_id}:activate`. The config
snapshot includes only runtime-supported per-KB defaults: parser engine/options,
chunk size/overlap, embedding model/dimension, LLM role settings, query limits,
and rerank settings. Deployment-level infrastructure settings are recorded in the
run report (`env_snapshot` / health output) for audit, but are not posted inside
the KB config version.

Important examples:

- Chunk settings are stored in `chunk_config.chunk_size` and
  `chunk_config.chunk_overlap_size`.
- Parser defaults are stored in `parser_config.engine` and
  `parser_config.process_options`. MinerU/Docling endpoints, tokens, service
  mode, workers, and timeouts are deployment-level `.env` settings managed by
  the running server, not per-KB config fields.
- Embedding settings are stored in `embedding_config`; changing embedding model or
  dimension after data has been indexed still requires rebuilding/clearing
  incompatible vector data.
- Query defaults are stored in `query_config` and are also passed explicitly by
  the demo request so the report captures exactly what was used.
- Storage backends, object storage endpoint/bucket, vector DB URI, and metadata
  backend are deployment-level settings. They remain visible in the report's
  sanitized environment/health snapshots for traceability, but switching them
  requires server-side configuration and compatible data migration/rebuild, not
  activating a different KB config version.

## Incremental updates from 模拟文件

The default flow uses `POST /kbs/{kb_id}/documents:sync` with
`auto_parse=true&auto_index=true`. On every run, the script recursively scans the
source directory, derives a stable `source_key` as
`enterprise-demo/<relative path under 模拟文件>`, and uploads the current files.

Incremental behavior:

- New file: no existing document has that `source_key`, so a new document is
  created, parsed, embedded, and indexed.
- Unchanged file: same `source_key` and same content hash, so source replacement
  is skipped; parse/build are skipped when parser/index hashes still match.
- Changed file at same relative path: same `source_key` but different content
  hash, so the existing document is replaced in place and rebuilt.
- Parser/chunk/config changes: source bytes may be unchanged, but parser/index
  hashes can change; the server reparses or rebuilds as needed.
- Removed local file: sync is not a mirror delete. Documents missing from the new
  request stay in the KB until you call delete or batch-delete explicitly.
- Renamed/moved file: the relative path changes, so the `source_key` changes; the
  server treats it as a new document and the old document must be deleted
  explicitly if you do not want it retained.

For repeat incremental runs, normally omit `--run-id` so the script generates a
fresh idempotency key. If you reuse the same `--run-id` after the file set or
content changes, the API correctly returns an idempotency conflict.

## Interactive delete test

Add `--delete-test` to pause after document/artifact/graph inspection and before
querying. The script lists current KB documents with numbers, asks whether to
delete, accepts multi-selection, calls the delete API, waits for the delete job,
then records `documents_after_delete` and `graph_after_delete` in the report.

Example:

```powershell
$env:LIGHTRAG_API_KEY = "sk-123456"
uv run python examples/enterprise_kb_mvp/enterprise_kb_mvp_demo.py `
  --server "http://127.0.0.1:9621" `
  --source-dir "E:/pycharmprojects/RAG/LightRAG/模拟文件" `
  --kb-id enterprise_mvp_demo `
  --delete-test
```

Selection syntax at the prompt:

- `1`: delete one document.
- `1,3,5`: delete multiple documents.
- `2-4`: delete a range.
- `all`: delete all listed documents.
- empty input / `none` / `cancel`: cancel deletion.

Delete flags:

- Default delete keeps source objects/files and parser artifacts where the backend
  permits that; it removes the document from KB metadata and LightRAG indexes.
- `--delete-source-file`: also remove the original uploaded source object/file.
- `--delete-artifacts`: also remove parser artifacts.
- `--delete-llm-cache`: clear related LLM cache entries.
- `--delete-strategy safe`: default graph cleanup mode. Other supported API
  values are `rebuild_doc_scope`, `rebuild_kb`, and `rebuild_subgraph`.

Reports are written to `examples/enterprise_kb_mvp/runs/` as UTF-8 JSON. Reports
are ignored by git, but they include source paths, object URIs, artifact metadata,
and query output; treat them as local operational artifacts.
