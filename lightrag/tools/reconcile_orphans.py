#!/usr/bin/env python3
"""Orphan reconciliation CLI for object-authoritative artifact storage.

Phase 3.2 Writer O.  Thin wrapper around :class:`OrphanReconcileService` so
operators can plan, apply, and resume orphan reconciliation from a durable
shell entrypoint.  Mirrors the migration CLI's redaction, dry-run-first, and
``--plan-id`` + ``--apply`` + ``--yes`` confirmation semantics.

Usage::

    lightrag-reconcile-orphans \\
        --working-dir ./rag_storage \\
        --bucket lightrag-kb

    lightrag-reconcile-orphans \\
        --working-dir ./rag_storage \\
        --bucket lightrag-kb \\
        --plan-id <plan_id_from_dry_run> --apply --yes

    lightrag-reconcile-orphans \\
        --working-dir ./rag_storage \\
        --bucket lightrag-kb \\
        --plan-id <plan_id_from_dry_run> --apply --yes --release-retained
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Add project root to path for direct ``python lightrag/tools/...`` execution.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from lightrag.api.artifact_lifecycle import ArtifactMaintenanceMetadataBackend
from lightrag.api.metadata_store import SQLiteMetadataStore
from lightrag.api.object_storage import (
    ObjectStorage,
    ObjectStorageConfig,
    S3ObjectStorage,
)
from lightrag.api.orphan_reconcile_service import (
    OrphanReconcileError,
    OrphanReconcileService,
    redact_mapping,
    redact_value,
)

load_dotenv(dotenv_path=".env", override=False)


_DEFAULT_LEASE_SECONDS = 600.0
_DEFAULT_MIN_AGE_HOURS = 24
_DEFAULT_OBJECT_PREFIX = "kb"


# ---------------------------------------------------------------------------
# Backend construction (mirrors migrate_artifacts_to_object.py)
# ---------------------------------------------------------------------------


def _resolve_metadata_backend(
    args: argparse.Namespace,
) -> ArtifactMaintenanceMetadataBackend:
    """Decide the metadata backend label (sqlite vs postgres) from args/env."""

    forced = getattr(args, "metadata_backend", None)
    if forced:
        normalized = forced.strip().lower()
        if normalized in {"sqlite", "postgres"}:
            return normalized  # type: ignore[return-value]
        raise OrphanReconcileError(
            "Unsupported --metadata-backend value; expected sqlite or postgres"
        )
    env_value = os.getenv("LIGHTRAG_KB_METADATA_BACKEND", "local").strip().lower()
    if env_value in {"postgres", "postgresql"}:
        return "postgres"
    return "sqlite"


def _metadata_store_from_args(
    args: argparse.Namespace, backend: ArtifactMaintenanceMetadataBackend
) -> Any:
    if backend == "postgres":
        from lightrag.api.postgres_metadata_store import PostgresMetadataStore

        if getattr(args, "postgres_dsn", None):
            return PostgresMetadataStore(dsn=args.postgres_dsn)
        return PostgresMetadataStore.from_env()
    sqlite_path = Path(args.working_dir) / "metadata" / "metadata.sqlite3"
    if not sqlite_path.exists():
        raise OrphanReconcileError(
            "SQLite metadata store not found under the configured --working-dir"
        )
    return SQLiteMetadataStore(sqlite_path)


def _object_storage_from_args(args: argparse.Namespace) -> ObjectStorage:
    config = ObjectStorageConfig.from_env()
    overrides: dict[str, Any] = {}
    if args.object_storage_endpoint:
        overrides["endpoint_url"] = args.object_storage_endpoint
    if args.bucket:
        overrides["bucket"] = args.bucket
    if args.prefix:
        overrides["prefix"] = args.prefix.strip("/")
    if args.use_ssl is not None:
        overrides["use_ssl"] = args.use_ssl
    if overrides:
        config = type(config)(**{**config.__dict__, **overrides})
    return S3ObjectStorage(config)


def _resolve_bucket_name(
    args: argparse.Namespace, object_storage: ObjectStorage
) -> str:
    if args.bucket:
        return args.bucket
    config_attr = getattr(object_storage, "_config", None)
    if config_attr is not None:
        candidate = getattr(config_attr, "bucket", None)
        if isinstance(candidate, str) and candidate:
            return candidate
    env_bucket = os.getenv("LIGHTRAG_OBJECT_STORAGE_BUCKET")
    if env_bucket:
        return env_bucket.strip()
    return "lightrag-kb"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lightrag-reconcile-orphans",
        description=(
            "Reconcile orphan objects under the configured bucket prefix using "
            "a dry-run-first, resumable plan.  Apply never deletes objects; it "
            "only enqueues cleanup manifests that the cleanup service drains "
            "with verified deletion."
        ),
    )
    parser.add_argument(
        "--working-dir",
        required=True,
        help="Canonical LightRAG working directory containing metadata.sqlite3.",
    )
    parser.add_argument(
        "--object-storage-endpoint",
        default=None,
        help="S3/MinIO endpoint URL (overrides LIGHTRAG_OBJECT_STORAGE_ENDPOINT).",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="Target bucket name (overrides LIGHTRAG_OBJECT_STORAGE_BUCKET).",
    )
    parser.add_argument(
        "--prefix",
        default=_DEFAULT_OBJECT_PREFIX,
        help="Object key prefix (default: 'kb').",
    )
    parser.add_argument(
        "--min-age-hours",
        type=int,
        default=_DEFAULT_MIN_AGE_HOURS,
        help=(
            "Minimum object age (in hours) for reconciliation eligibility. "
            "Objects younger than this window are report-only."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Force dry-run plan creation even when --plan-id/--apply/--yes are "
            "present. Default behaviour: plan creation unless --plan-id, "
            "--apply, and --yes are all set."
        ),
    )
    parser.add_argument(
        "--plan-id",
        default=None,
        help="Apply a previously-created plan (requires --apply and --yes).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Confirm apply intent (required with --plan-id and --yes).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Final confirmation for apply (required with --plan-id and --apply).",
    )
    parser.add_argument(
        "--release-retained",
        action="store_true",
        help=(
            "Release retained manifests targeting reconciled objects. Default: "
            "retained manifests are NOT released."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an in-progress apply run (re-claims after lease expiry).",
    )
    parser.add_argument(
        "--metadata-backend",
        choices=("sqlite", "postgres"),
        default=None,
        help="Force metadata backend (default: derived from LIGHTRAG_KB_METADATA_BACKEND).",
    )
    parser.add_argument(
        "--postgres-dsn",
        default=None,
        help="PostgreSQL DSN when --metadata-backend=postgres (or env equivalent).",
    )
    parser.add_argument(
        "--use-ssl",
        action="store_true",
        default=None,
        help="Use TLS for the S3 endpoint (overrides LIGHTRAG_OBJECT_STORAGE_USE_SSL).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary.",
    )
    return parser


def _print_summary(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    mode = payload.get("mode", "plan")
    if mode == "plan":
        print(f"Orphan reconcile plan created: {payload['plan_id']}")
        print(f"  item_count: {payload['item_count']}")
        print(f"  metadata_backend: {payload['metadata_backend']}")
        classifications = payload.get("classifications", {})
        for label in (
            "eligible",
            "referenced",
            "retained",
            "malformed",
            "unknown_owner",
            "too_new",
        ):
            if classifications.get(label):
                print(f"  {label}: {classifications[label]}")
    else:
        print(f"Orphan reconcile apply summary for plan {payload['plan_id']}")
        print(f"  apply_run_id: {payload['apply_run_id']}")
        print(f"  items_total: {payload['items_total']}")
        print(f"  items_enqueued: {payload['items_enqueued']}")
        print(f"  items_skipped: {payload['items_skipped']}")
        print(f"  items_blocked: {payload['items_blocked']}")
        print(f"  items_failed: {payload['items_failed']}")
        for issue in payload.get("issues", []):
            print(f"  issue: {issue}")


async def _run_plan(args: argparse.Namespace) -> dict[str, Any]:
    backend = _resolve_metadata_backend(args)
    metadata_store = _metadata_store_from_args(args, backend)
    object_storage = _object_storage_from_args(args)
    bucket = _resolve_bucket_name(args, object_storage)
    await metadata_store.initialize()
    try:
        await object_storage.initialize()
        try:
            service = OrphanReconcileService(
                metadata_store=metadata_store,
                object_storage=object_storage,
                metadata_backend=backend,
                bucket=bucket,
                prefix=args.prefix,
                min_age_hours=args.min_age_hours,
            )
            summary = await service.create_plan()
            return {
                "mode": "plan",
                **redact_mapping(summary.to_audit_dict()),
            }
        finally:
            await object_storage.close()
    finally:
        await _close_store(metadata_store)


async def _run_apply(args: argparse.Namespace) -> dict[str, Any]:
    backend = _resolve_metadata_backend(args)
    metadata_store = _metadata_store_from_args(args, backend)
    object_storage = _object_storage_from_args(args)
    bucket = _resolve_bucket_name(args, object_storage)
    await metadata_store.initialize()
    try:
        await object_storage.initialize()
        try:
            service = OrphanReconcileService(
                metadata_store=metadata_store,
                object_storage=object_storage,
                metadata_backend=backend,
                bucket=bucket,
                prefix=args.prefix,
                min_age_hours=args.min_age_hours,
            )
            summary = await service.apply_plan(
                args.plan_id,
                release_retained=args.release_retained,
                resume=args.resume,
            )
            return {
                "mode": "apply",
                **redact_mapping(summary.to_audit_dict()),
            }
        finally:
            await object_storage.close()
    finally:
        await _close_store(metadata_store)


async def _close_store(metadata_store: Any) -> None:
    close = getattr(metadata_store, "close", None)
    if close is None:
        return
    try:
        await close()
    except Exception:
        return


async def _async_main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.plan_id and not args.apply:
        parser.error("--plan-id requires --apply to confirm apply intent")
    if args.plan_id and not args.yes:
        parser.error("--plan-id requires --yes to confirm apply")
    if args.apply and not args.plan_id:
        parser.error("--apply requires --plan-id")
    if args.yes and not args.plan_id:
        parser.error("--yes requires --plan-id")
    if args.release_retained and not (args.plan_id and args.apply and args.yes):
        parser.error("--release-retained requires --plan-id, --apply, and --yes")
    if args.resume and not args.plan_id:
        parser.error("--resume requires --plan-id")
    try:
        if args.plan_id and args.apply and args.yes and not args.dry_run:
            payload = await _run_apply(args)
        else:
            payload = await _run_plan(args)
    except OrphanReconcileError as exc:
        sys.stderr.write(f"orphan reconcile failed: {redact_value(exc)}\n")
        raise SystemExit(2)
    except KeyboardInterrupt:  # pragma: no cover - operator-driven
        sys.stderr.write(
            "orphan reconcile interrupted; resume with "
            "--plan-id --apply --yes --resume\n"
        )
        raise SystemExit(130)
    _print_summary(payload, as_json=args.json)
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
