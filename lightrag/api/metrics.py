from __future__ import annotations

from collections import Counter, defaultdict
import threading
from typing import Any


HTTP_LATENCY_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
)

_HTTP_METRICS_LOCK = threading.Lock()
_HTTP_REQUEST_TOTALS: Counter[tuple[str, str, str]] = Counter()
_HTTP_DURATION_SUMS: defaultdict[tuple[str, str, str], float] = defaultdict(float)
_HTTP_DURATION_BUCKETS: Counter[tuple[str, str, str, str]] = Counter()


def record_http_request(
    method: str,
    route: str | None,
    status_code: int | str,
    duration_seconds: float,
) -> None:
    """Record one HTTP request for the in-process Prometheus exporter.

    Labels intentionally use route templates (for example
    ``/kbs/{kb_id}/query``) instead of raw paths to avoid unbounded
    cardinality. Metrics are process-local, which matches the current
    single-server deployment target.
    """

    normalized_method = (method or "UNKNOWN").upper()
    normalized_route = (route or "__unmatched__").strip() or "__unmatched__"
    normalized_status = str(status_code)
    duration = max(0.0, float(duration_seconds or 0.0))
    key = (normalized_method, normalized_route, normalized_status)
    with _HTTP_METRICS_LOCK:
        _HTTP_REQUEST_TOTALS[key] += 1
        _HTTP_DURATION_SUMS[key] += duration
        for bucket in HTTP_LATENCY_BUCKETS:
            if duration <= bucket:
                _HTTP_DURATION_BUCKETS[
                    (*key, _format_bucket_le(bucket))
                ] += 1
        _HTTP_DURATION_BUCKETS[(*key, "+Inf")] += 1


def reset_http_metrics_for_tests() -> None:
    """Clear process-local HTTP metrics between tests."""

    with _HTTP_METRICS_LOCK:
        _HTTP_REQUEST_TOTALS.clear()
        _HTTP_DURATION_SUMS.clear()
        _HTTP_DURATION_BUCKETS.clear()


DOCUMENT_STATUSES = (
    "uploaded",
    "parse_queued",
    "parsing",
    "parsed",
    "parse_failed",
    "build_queued",
    "building",
    "ready",
    "build_failed",
    "deleting",
    "delete_failed",
    "replacing",
    "replace_failed",
)

JOB_STATUSES = (
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancelling",
    "cancelled",
    "retrying",
)

_AUDIT_SAMPLE_LIMIT = 500


async def build_prometheus_metrics(
    *,
    kb_service: Any,
    metadata_store: Any,
    enterprise_enabled: bool,
    job_worker_enabled: bool,
    object_storage_enabled: bool,
    kb_metadata_backend: str,
) -> str:
    lines: list[str] = []
    _append_help(lines, "lightrag_enterprise_enabled", "Enterprise auth mode enabled.")
    _append_gauge(lines, "lightrag_enterprise_enabled", 1 if enterprise_enabled else 0)
    _append_help(lines, "lightrag_kb_job_worker_enabled", "Durable KB job worker enabled.")
    _append_gauge(lines, "lightrag_kb_job_worker_enabled", 1 if job_worker_enabled else 0)
    _append_help(lines, "lightrag_object_storage_enabled", "Object storage backend enabled.")
    _append_gauge(lines, "lightrag_object_storage_enabled", 1 if object_storage_enabled else 0)
    _append_help(lines, "lightrag_info", "Static LightRAG server configuration info.")
    _append_gauge(
        lines,
        "lightrag_info",
        1,
        labels={"kb_metadata_backend": kb_metadata_backend},
    )
    _append_http_request_metrics(lines)

    try:
        kb_records = await kb_service.list(include_deleted=True)
    except Exception:
        _append_collection_error(lines, "kb_catalog")
        kb_records = []

    _append_help(lines, "lightrag_kb_total", "Knowledge base catalog rows.")
    _append_gauge(lines, "lightrag_kb_total", len(kb_records))
    _append_help(lines, "lightrag_kb_status_total", "Knowledge bases by status.")
    for status, count in sorted(Counter(record.status for record in kb_records).items()):
        _append_gauge(lines, "lightrag_kb_status_total", count, labels={"status": status})

    _append_help(lines, "lightrag_kb_documents_total", "Documents by KB and status.")
    _append_help(lines, "lightrag_kb_jobs_total", "Jobs by KB and status.")
    for record in kb_records:
        if record.status == "deleted":
            continue
        await _append_document_metrics(lines, metadata_store, record.id)
        await _append_job_metrics(lines, metadata_store, record.id)

    if enterprise_enabled:
        await _append_audit_metrics(lines, metadata_store)

    return "\n".join(lines) + "\n"


async def _append_document_metrics(
    lines: list[str], metadata_store: Any, kb_id: str
) -> None:
    try:
        _documents, total = await metadata_store.list_documents(kb_id, limit=1)
        _append_gauge(
            lines,
            "lightrag_kb_documents_total",
            total,
            labels={"kb_id": kb_id, "status": "all"},
        )
        for status in DOCUMENT_STATUSES:
            _documents, status_total = await metadata_store.list_documents(
                kb_id,
                status=status,
                limit=1,
            )
            _append_gauge(
                lines,
                "lightrag_kb_documents_total",
                status_total,
                labels={"kb_id": kb_id, "status": status},
            )
    except Exception:
        _append_collection_error(lines, "documents", {"kb_id": kb_id})


async def _append_job_metrics(
    lines: list[str], metadata_store: Any, kb_id: str) -> None:
    try:
        _jobs, total = await metadata_store.list_jobs(kb_id, limit=1)
        _append_gauge(
            lines,
            "lightrag_kb_jobs_total",
            total,
            labels={"kb_id": kb_id, "status": "all"},
        )
        for status in JOB_STATUSES:
            _jobs, status_total = await metadata_store.list_jobs(
                kb_id,
                statuses=[status],
                limit=1,
            )
            _append_gauge(
                lines,
                "lightrag_kb_jobs_total",
                status_total,
                labels={"kb_id": kb_id, "status": status},
            )
    except Exception:
        _append_collection_error(lines, "jobs", {"kb_id": kb_id})


async def _append_audit_metrics(lines: list[str], metadata_store: Any) -> None:
    try:
        events = await metadata_store.list_audit_events(limit=_AUDIT_SAMPLE_LIMIT)
    except Exception:
        _append_collection_error(lines, "enterprise_audit")
        return

    _append_help(
        lines,
        "lightrag_enterprise_audit_events_sampled_total",
        "Enterprise audit events returned by the bounded metrics sample.",
    )
    _append_gauge(lines, "lightrag_enterprise_audit_events_sampled_total", len(events))
    for event_type, count in sorted(Counter(event.event_type for event in events).items()):
        _append_gauge(
            lines,
            "lightrag_enterprise_audit_events_sampled_total",
            count,
            labels={"event_type": event_type},
        )
    _append_help(
        lines,
        "lightrag_enterprise_audit_events_sample_limit",
        "Maximum audit events sampled for metrics export.",
    )
    _append_gauge(
        lines,
        "lightrag_enterprise_audit_events_sample_limit",
        _AUDIT_SAMPLE_LIMIT,
    )


def _append_http_request_metrics(lines: list[str]) -> None:
    with _HTTP_METRICS_LOCK:
        request_totals = dict(_HTTP_REQUEST_TOTALS)
        duration_sums = dict(_HTTP_DURATION_SUMS)
        duration_buckets = dict(_HTTP_DURATION_BUCKETS)

    _append_help(
        lines,
        "lightrag_http_requests_total",
        "HTTP requests observed by method, route template, and status code.",
        metric_type="counter",
    )
    for (method, route, status_code), count in sorted(request_totals.items()):
        _append_gauge(
            lines,
            "lightrag_http_requests_total",
            count,
            labels={"method": method, "route": route, "status_code": status_code},
        )

    _append_help(
        lines,
        "lightrag_http_request_duration_seconds",
        "HTTP request duration histogram in seconds by method, route template, and status code.",
        metric_type="histogram",
    )
    for (method, route, status_code, le), count in sorted(duration_buckets.items()):
        _append_gauge(
            lines,
            "lightrag_http_request_duration_seconds_bucket",
            count,
            labels={
                "method": method,
                "route": route,
                "status_code": status_code,
                "le": le,
            },
        )
    for (method, route, status_code), total in sorted(duration_sums.items()):
        labels = {"method": method, "route": route, "status_code": status_code}
        _append_gauge(
            lines,
            "lightrag_http_request_duration_seconds_sum",
            total,
            labels=labels,
        )
        _append_gauge(
            lines,
            "lightrag_http_request_duration_seconds_count",
            request_totals.get((method, route, status_code), 0),
            labels=labels,
        )


def _format_bucket_le(bucket: float) -> str:
    return f"{bucket:g}"


def _append_collection_error(
    lines: list[str], scope: str, labels: dict[str, str] | None = None
) -> None:
    merged_labels = {"scope": scope}
    if labels:
        merged_labels.update(labels)
    _append_gauge(lines, "lightrag_metrics_collection_error", 1, labels=merged_labels)


def _append_help(lines: list[str], name: str, help_text: str, metric_type: str = "gauge") -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} {metric_type}")


def _append_gauge(
    lines: list[str], name: str, value: int | float, *, labels: dict[str, str] | None = None
) -> None:
    label_text = ""
    if labels:
        label_text = "{" + ",".join(
            f'{key}="{_escape_label(value)}"' for key, value in sorted(labels.items())
        ) + "}"
    lines.append(f"{name}{label_text} {value}")


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
