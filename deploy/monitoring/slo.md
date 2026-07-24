# LightRAG single-server SLOs

This file defines the default operational SLOs for the current deployment model:
one LightRAG API server process on one server, with optional local/external
storage services reachable from that server. It intentionally does **not** define
multi-instance or distributed-quota objectives.

## Service availability

- **Objective**: 99.5% monthly availability for authenticated API requests.
- **Good events**: HTTP responses with status `< 500`.
- **Total events**: all HTTP responses recorded by `lightrag_http_requests_total`.
- **PromQL**:

```promql
1 - (
  sum(increase(lightrag_http_requests_total{status_code=~"5.."}[30d]))
  /
  clamp_min(sum(increase(lightrag_http_requests_total[30d])), 1)
)
```

## Interactive API latency

- **Objective**: 95% of non-streaming interactive requests finish within 2s.
- **Scope**: health, auth, KB metadata reads, document/job/config list/detail,
  graph metadata reads, and query metadata endpoints when the backing LLM call
  is not part of the measured route. Long-running parse/build/delete work is
  tracked by job status instead of request latency.
- **PromQL example**:

```promql
histogram_quantile(
  0.95,
  sum by (le, route) (
    rate(lightrag_http_request_duration_seconds_bucket{route!~".*query.*|.*upload.*"}[5m])
  )
) < 2
```

## Job health

- **Objective**: no queued/running/retrying/cancelling KB job remains active for
  more than 45 minutes without operator acknowledgement.
- **Alert**: `LightRAGQueuedOrRunningJobsStuck` in `prometheus-rules.yml`.
- **Runbook**: inspect `GET /kbs/{kb_id}/jobs`, check parser/build/delete logs,
  then either wait, cancel, or retry according to `docs/API接口.md`.

## Metrics freshness

- **Objective**: Prometheus can scrape `/metrics` at least once every 2 minutes.
- **Alert**: `LightRAGExporterDown` in `prometheus-rules.yml`.
- **Note**: `/metrics` is protected by `combined_auth`; configure the scrape job
  with `X-API-Key` or bearer auth rather than whitelisting protected routes in
  enterprise mode.

## Backup/restore drill

- **Objective**: after every backup restore rehearsal, run the single-server ops
  smoke script and archive its JSON report with the backup record.
- **Command**:

```bash
uv run python scripts/run_single_server_ops_drill.py \
  --backup-id "$BACKUP_ID" \
  --base-url "$LIGHTRAG_BASE_URL" \
  --api-key "$LIGHTRAG_API_KEY" \
  --kb-id "$SMOKE_KB_ID"
```
