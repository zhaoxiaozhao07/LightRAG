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
