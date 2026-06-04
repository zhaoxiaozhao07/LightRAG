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

External backends exercised in the reference run (see the report's
`env_snapshot`):

- ✅ PostgreSQL — KB control-plane metadata (`LIGHTRAG_KB_METADATA_BACKEND=postgres`).
- ✅ Milvus — chunk/entity/relation vectors (`LIGHTRAG_VECTOR_STORAGE=MilvusVectorDBStorage`).
- ✅ MinIO/S3 — source files and parser artifacts (`LIGHTRAG_OBJECT_STORAGE=minio`).
- ✅ Real LightRAG engine + real MinerU / LLM / embedding / rerank end to end.

External backends NOT yet covered by this drill (file-based in the reference run;
re-run under the matching profile to cover them):

- ⚠️ Graph backend: the reference run uses `NetworkXStorage` (file-based). For a
  production **Neo4j**, re-run with `LIGHTRAG_GRAPH_STORAGE=Neo4JStorage` to verify
  its hard-delete cleanup and workspace isolation.
- ⚠️ KV / doc_status: the reference run uses `JsonKVStorage` / `JsonDocStatusStorage`
  (file-based). For PostgreSQL / Redis / MongoDB, re-run under that profile.

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
