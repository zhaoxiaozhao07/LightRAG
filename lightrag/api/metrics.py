from __future__ import annotations

from collections import Counter
from typing import Any


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


def _append_collection_error(
    lines: list[str], scope: str, labels: dict[str, str] | None = None
) -> None:
    merged_labels = {"scope": scope}
    if labels:
        merged_labels.update(labels)
    _append_gauge(lines, "lightrag_metrics_collection_error", 1, labels=merged_labels)


def _append_help(lines: list[str], name: str, help_text: str) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} gauge")


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
