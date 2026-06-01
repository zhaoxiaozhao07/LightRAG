#!/usr/bin/env python3
"""Migrate LightRAG KB control-plane metadata to PostgreSQL.

This tool copies the API server control plane from the local backend
(``WORKING_DIR/metadata/knowledge_bases.json`` + ``metadata.sqlite3``) into the
PostgreSQL backend used by ``LIGHTRAG_KB_METADATA_BACKEND=postgres``. It is a
metadata/catalog migration only: source files, parsed artifacts, vector stores,
graph stores, and text-chunk stores are not copied.

Usage::

    python -m lightrag.tools.migrate_kb_metadata_to_postgres --dry-run
    python -m lightrag.tools.migrate_kb_metadata_to_postgres --strategy skip
    lightrag-migrate-kb-metadata --working-dir ./rag_storage --strategy fail
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Sequence

from dotenv import load_dotenv

# Add project root to path for direct ``python lightrag/tools/...`` execution.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from lightrag.api.kb_service import KnowledgeBaseRecord, KnowledgeBaseService
from lightrag.api.metadata_store import (
    ArtifactRecord,
    ConfigVersionRecord,
    DocumentRecord,
    JobRecord,
    SQLiteMetadataStore,
    _metadata_source_key,
)
from lightrag.api.postgres_kb_service import PostgresKnowledgeBaseService
from lightrag.api.postgres_metadata_store import PostgresMetadataStore

load_dotenv(dotenv_path=".env", override=False)

ConflictStrategy = Literal["fail", "skip", "overwrite"]


@dataclass(slots=True)
class SourceKeyMapping:
    kb_id: str
    source_key: str
    document_id: str
    created_at: str
    updated_at: str


@dataclass(slots=True)
class ControlPlaneSnapshot:
    knowledge_bases: list[KnowledgeBaseRecord]
    documents: list[DocumentRecord]
    jobs: list[JobRecord]
    artifacts: list[ArtifactRecord]
    config_versions: list[ConfigVersionRecord]
    source_key_mappings: list[SourceKeyMapping]
    issues: list[str]

    def counts(self) -> dict[str, int]:
        return {
            "knowledge_bases": len(self.knowledge_bases),
            "documents": len(self.documents),
            "jobs": len(self.jobs),
            "artifacts": len(self.artifacts),
            "config_versions": len(self.config_versions),
            "source_key_mappings": len(self.source_key_mappings),
        }


@dataclass(slots=True)
class MigrationTableStats:
    inserted: int = 0
    skipped: int = 0
    overwritten: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class MigrationSummary:
    dry_run: bool
    strategy: ConflictStrategy
    source: dict[str, int]
    catalog: MigrationTableStats
    documents: MigrationTableStats
    jobs: MigrationTableStats
    artifacts: MigrationTableStats
    config_versions: MigrationTableStats
    source_keys_projected: int
    issues: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "strategy": self.strategy,
            "source": self.source,
            "catalog": self.catalog.to_dict(),
            "documents": self.documents.to_dict(),
            "jobs": self.jobs.to_dict(),
            "artifacts": self.artifacts.to_dict(),
            "config_versions": self.config_versions.to_dict(),
            "source_keys_projected": self.source_keys_projected,
            "issues": list(self.issues),
        }


def _record_json(
    record: KnowledgeBaseRecord
    | DocumentRecord
    | JobRecord
    | ArtifactRecord
    | ConfigVersionRecord,
) -> str:
    return json.dumps(asdict(record), ensure_ascii=False, sort_keys=True)


def _batch_id(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("batch_id")
    return value if isinstance(value, str) and value else None


def _document_projection_source_key(document: DocumentRecord) -> str | None:
    if document.deleted_at is not None:
        return None
    return _metadata_source_key(document.metadata)


def _source_key_mapping_by_document(
    mappings: Sequence[SourceKeyMapping],
) -> dict[tuple[str, str], SourceKeyMapping]:
    by_document: dict[tuple[str, str], SourceKeyMapping] = {}
    for mapping in mappings:
        by_document[(mapping.kb_id, mapping.document_id)] = mapping
    return by_document


def normalize_documents_for_postgres(
    documents: Sequence[DocumentRecord],
    mappings: Sequence[SourceKeyMapping],
) -> tuple[list[DocumentRecord], list[str]]:
    """Return documents with PostgreSQL-compatible source-key metadata.

    SQLite stores source-key lookup rows in ``document_source_keys``. PostgreSQL
    stores the lookup projection directly on ``kb_documents.source_key`` and
    derives it from ``DocumentRecord.metadata["source_key"]`` during normal
    writes. For migration, keep the full record intact when metadata is already
    canonical; when an old SQLite row has a mapping but missing metadata, inject
    that source key into a copied record so future PostgreSQL writes preserve it.
    """
    mapping_by_document = _source_key_mapping_by_document(mappings)
    normalized: list[DocumentRecord] = []
    issues: list[str] = []
    active_source_keys: dict[tuple[str, str], str] = {}

    for document in documents:
        metadata = dict(document.metadata)
        metadata_source_key = _metadata_source_key(metadata)
        mapping = mapping_by_document.get((document.kb_id, document.id))

        if document.deleted_at is None and metadata_source_key is None and mapping:
            metadata["source_key"] = mapping.source_key
            issues.append(
                "Backfilled missing metadata.source_key for document "
                f"'{document.id}' from SQLite document_source_keys"
            )
        elif (
            document.deleted_at is None
            and metadata_source_key is not None
            and mapping is not None
            and metadata_source_key != mapping.source_key
        ):
            issues.append(
                "Document source-key mismatch for "
                f"'{document.id}': metadata has '{metadata_source_key}', "
                f"document_source_keys has '{mapping.source_key}'; using metadata"
            )

        migrated = replace(document, metadata=metadata)
        projected_source_key = _document_projection_source_key(migrated)
        if projected_source_key is not None:
            key = (migrated.kb_id, projected_source_key)
            existing_document_id = active_source_keys.get(key)
            if existing_document_id is not None and existing_document_id != migrated.id:
                raise ValueError(
                    "Duplicate active source_key in source SQLite metadata: "
                    f"kb_id='{migrated.kb_id}', source_key='{projected_source_key}', "
                    f"documents='{existing_document_id}' and '{migrated.id}'"
                )
            active_source_keys[key] = migrated.id
        normalized.append(migrated)

    mapped_documents = {(mapping.kb_id, mapping.document_id) for mapping in mappings}
    known_documents = {(document.kb_id, document.id) for document in documents}
    for kb_id, document_id in sorted(mapped_documents - known_documents):
        issues.append(
            "Ignoring orphan SQLite document_source_keys row for missing document "
            f"'{document_id}' in KB '{kb_id}'"
        )

    return normalized, issues


def _sqlite_rows(
    conn: sqlite3.Connection,
    table: str,
    kb_ids: Sequence[str] | None,
    order_by: str,
) -> list[sqlite3.Row]:
    if kb_ids is not None and not kb_ids:
        return []
    where = ""
    params: list[str] = []
    if kb_ids:
        placeholders = ", ".join("?" for _ in kb_ids)
        where = f"WHERE kb_id IN ({placeholders})"
        params.extend(kb_ids)
    return conn.execute(
        f"SELECT * FROM {table} {where} ORDER BY {order_by}", params
    ).fetchall()


async def collect_local_snapshot(
    working_dir: str | Path,
    *,
    kb_ids: Sequence[str] | None = None,
) -> ControlPlaneSnapshot:
    """Collect a preservation-oriented snapshot from local JSON/SQLite state."""
    working_path = Path(working_dir)
    metadata_dir = working_path / "metadata"
    catalog_path = metadata_dir / "knowledge_bases.json"
    sqlite_path = metadata_dir / "metadata.sqlite3"
    if not catalog_path.exists():
        raise FileNotFoundError(f"Knowledge base catalog not found: {catalog_path}")
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite metadata store not found: {sqlite_path}")

    kb_filter = set(kb_ids or [])
    kb_service = KnowledgeBaseService(catalog_path)
    metadata_store = SQLiteMetadataStore(sqlite_path)
    await kb_service.initialize()
    await metadata_store.initialize()
    try:
        knowledge_bases = await kb_service.list(include_deleted=True)
        if kb_filter:
            knowledge_bases = [kb for kb in knowledge_bases if kb.id in kb_filter]
        selected_kb_ids = [kb.id for kb in knowledge_bases]
        with metadata_store._connect() as conn:  # noqa: SLF001 - internal migration tool
            documents = [
                DocumentRecord.from_row(row)
                for row in _sqlite_rows(
                    conn,
                    "documents",
                    selected_kb_ids,
                    "created_at ASC, id ASC",
                )
            ]
            jobs = [
                JobRecord.from_row(row)
                for row in _sqlite_rows(
                    conn,
                    "jobs",
                    selected_kb_ids,
                    "created_at ASC, id ASC",
                )
            ]
            artifacts = [
                ArtifactRecord.from_row(row)
                for row in _sqlite_rows(
                    conn,
                    "document_artifacts",
                    selected_kb_ids,
                    "created_at ASC, id ASC",
                )
            ]
            config_versions = [
                ConfigVersionRecord.from_row(row)
                for row in _sqlite_rows(
                    conn,
                    "kb_config_versions",
                    selected_kb_ids,
                    "version ASC, id ASC",
                )
            ]
            source_key_mappings = [
                SourceKeyMapping(
                    kb_id=str(row["kb_id"]),
                    source_key=str(row["source_key"]),
                    document_id=str(row["document_id"]),
                    created_at=str(row["created_at"]),
                    updated_at=str(row["updated_at"]),
                )
                for row in _sqlite_rows(
                    conn,
                    "document_source_keys",
                    selected_kb_ids,
                    "created_at ASC, source_key ASC",
                )
            ]
    finally:
        await metadata_store.close()
        await kb_service.close()

    normalized_documents, issues = normalize_documents_for_postgres(
        documents, source_key_mappings
    )
    return ControlPlaneSnapshot(
        knowledge_bases=knowledge_bases,
        documents=normalized_documents,
        jobs=jobs,
        artifacts=artifacts,
        config_versions=config_versions,
        source_key_mappings=source_key_mappings,
        issues=issues,
    )


async def _target_exists_catalog(conn: Any, record: KnowledgeBaseRecord) -> bool:
    return bool(await _target_catalog_ids(conn, record))


async def _target_catalog_ids(conn: Any, record: KnowledgeBaseRecord) -> set[str]:
    rows = await conn.fetch(
        "SELECT id FROM kb_catalog WHERE id = $1 OR workspace = $2",
        record.id,
        record.workspace,
    )
    return {str(row["id"]) for row in rows}


async def _delete_kb_metadata(conn: Any, kb_ids: set[str]) -> None:
    for kb_id in sorted(kb_ids):
        await conn.execute("DELETE FROM kb_document_artifacts WHERE kb_id = $1", kb_id)
        await conn.execute("DELETE FROM kb_config_versions WHERE kb_id = $1", kb_id)
        await conn.execute("DELETE FROM kb_jobs WHERE kb_id = $1", kb_id)
        await conn.execute("DELETE FROM kb_documents WHERE kb_id = $1", kb_id)


async def _target_document_refs(
    conn: Any, document: DocumentRecord
) -> set[tuple[str, str]]:
    source_key = _document_projection_source_key(document)
    if source_key is None:
        rows = await conn.fetch(
            "SELECT kb_id, id FROM kb_documents WHERE id = $1",
            document.id,
        )
    else:
        rows = await conn.fetch(
            """
            SELECT kb_id, id FROM kb_documents
            WHERE id = $1
               OR (kb_id = $2 AND source_key = $3 AND deleted_at IS NULL)
            """,
            document.id,
            document.kb_id,
            source_key,
        )
    return {(str(row["kb_id"]), str(row["id"])) for row in rows}


async def _delete_document_child_metadata(
    conn: Any, document_refs: set[tuple[str, str]]
) -> None:
    for kb_id, document_id in sorted(document_refs):
        await conn.execute(
            "DELETE FROM kb_document_artifacts WHERE kb_id = $1 AND document_id = $2",
            kb_id,
            document_id,
        )
        await conn.execute(
            "DELETE FROM kb_jobs WHERE kb_id = $1 AND document_id = $2",
            kb_id,
            document_id,
        )


async def _target_exists_document(conn: Any, document: DocumentRecord) -> bool:
    return bool(await _target_document_refs(conn, document))


async def _target_exists_job(conn: Any, job: JobRecord) -> bool:
    if not job.idempotency_key:
        return bool(
            await conn.fetchval("SELECT 1 FROM kb_jobs WHERE id = $1 LIMIT 1", job.id)
        )
    return bool(
        await conn.fetchval(
            """
            SELECT 1 FROM kb_jobs
            WHERE id = $1
               OR (kb_id = $2 AND job_type = $3 AND idempotency_key = $4)
            LIMIT 1
            """,
            job.id,
            job.kb_id,
            job.job_type,
            job.idempotency_key,
        )
    )


async def _target_exists_artifact(conn: Any, artifact: ArtifactRecord) -> bool:
    return bool(
        await conn.fetchval(
            "SELECT 1 FROM kb_document_artifacts WHERE id = $1 LIMIT 1",
            artifact.id,
        )
    )


async def _target_exists_config_version(
    conn: Any, record: ConfigVersionRecord
) -> bool:
    return bool(
        await conn.fetchval(
            """
            SELECT 1 FROM kb_config_versions
            WHERE id = $1 OR (kb_id = $2 AND version = $3)
            LIMIT 1
            """,
            record.id,
            record.kb_id,
            record.version,
        )
    )


async def _insert_catalog(
    conn: Any,
    record: KnowledgeBaseRecord,
    *,
    strategy: ConflictStrategy,
) -> str:
    if strategy == "skip" and await _target_exists_catalog(conn, record):
        return "skipped"
    if strategy == "overwrite":
        await conn.execute(
            "DELETE FROM kb_catalog WHERE id = $1 OR workspace = $2",
            record.id,
            record.workspace,
        )
    await conn.execute(
        """
        INSERT INTO kb_catalog (
            id, name, workspace, status, owner_id, tenant_id, visibility,
            created_at, updated_at, deleted_at, data_json
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb)
        """,
        record.id,
        record.name,
        record.workspace,
        record.status,
        record.owner_id,
        record.tenant_id,
        record.visibility,
        record.created_at,
        record.updated_at,
        record.deleted_at,
        _record_json(record),
    )
    return "overwritten" if strategy == "overwrite" else "inserted"


async def _insert_document(
    conn: Any,
    document: DocumentRecord,
    *,
    strategy: ConflictStrategy,
) -> str:
    if strategy == "skip" and await _target_exists_document(conn, document):
        return "skipped"
    if strategy == "overwrite":
        await _delete_document_child_metadata(
            conn, await _target_document_refs(conn, document)
        )
        source_key = _document_projection_source_key(document)
        if source_key is None:
            await conn.execute("DELETE FROM kb_documents WHERE id = $1", document.id)
        else:
            await conn.execute(
                """
                DELETE FROM kb_documents
                WHERE id = $1
                   OR (kb_id = $2 AND source_key = $3 AND deleted_at IS NULL)
                """,
                document.id,
                document.kb_id,
                source_key,
            )
    await conn.execute(
        """
        INSERT INTO kb_documents (
            id, kb_id, workspace, status, source_name, source_hash, source_key,
            batch_id, deleted_at, created_at, updated_at, data_json
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::jsonb)
        """,
        document.id,
        document.kb_id,
        document.workspace,
        document.status,
        document.source_name,
        document.source_hash,
        _document_projection_source_key(document),
        _batch_id(document.metadata),
        document.deleted_at,
        document.created_at,
        document.updated_at,
        _record_json(document),
    )
    return "overwritten" if strategy == "overwrite" else "inserted"


async def _insert_job(
    conn: Any,
    job: JobRecord,
    *,
    strategy: ConflictStrategy,
) -> str:
    if strategy == "skip" and await _target_exists_job(conn, job):
        return "skipped"
    if strategy == "overwrite":
        if job.idempotency_key:
            await conn.execute(
                """
                DELETE FROM kb_jobs
                WHERE id = $1
                   OR (kb_id = $2 AND job_type = $3 AND idempotency_key = $4)
                """,
                job.id,
                job.kb_id,
                job.job_type,
                job.idempotency_key,
            )
        else:
            await conn.execute("DELETE FROM kb_jobs WHERE id = $1", job.id)
    await conn.execute(
        """
        INSERT INTO kb_jobs (
            id, kb_id, workspace, batch_id, document_id, job_type, status,
            idempotency_key, retry_count, max_retries, queued_at, created_at,
            updated_at, data_json
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::jsonb)
        """,
        job.id,
        job.kb_id,
        job.workspace,
        job.batch_id,
        job.document_id,
        job.job_type,
        job.status,
        job.idempotency_key,
        job.retry_count,
        job.max_retries,
        job.queued_at,
        job.created_at,
        job.updated_at,
        _record_json(job),
    )
    return "overwritten" if strategy == "overwrite" else "inserted"


async def _insert_artifact(
    conn: Any,
    artifact: ArtifactRecord,
    *,
    strategy: ConflictStrategy,
) -> str:
    if strategy == "skip" and await _target_exists_artifact(conn, artifact):
        return "skipped"
    if strategy == "overwrite":
        await conn.execute(
            "DELETE FROM kb_document_artifacts WHERE id = $1",
            artifact.id,
        )
    await conn.execute(
        """
        INSERT INTO kb_document_artifacts (
            id, kb_id, workspace, document_id, artifact_type, created_at, data_json
        ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
        """,
        artifact.id,
        artifact.kb_id,
        artifact.workspace,
        artifact.document_id,
        artifact.artifact_type,
        artifact.created_at,
        _record_json(artifact),
    )
    return "overwritten" if strategy == "overwrite" else "inserted"


async def _insert_config_version(
    conn: Any,
    record: ConfigVersionRecord,
    *,
    strategy: ConflictStrategy,
) -> str:
    if strategy == "skip" and await _target_exists_config_version(conn, record):
        return "skipped"
    if strategy == "overwrite":
        await conn.execute(
            """
            DELETE FROM kb_config_versions
            WHERE id = $1 OR (kb_id = $2 AND version = $3)
            """,
            record.id,
            record.kb_id,
            record.version,
        )
    await conn.execute(
        """
        INSERT INTO kb_config_versions (
            id, kb_id, workspace, version, created_at, data_json
        ) VALUES ($1, $2, $3, $4, $5, $6::jsonb)
        """,
        record.id,
        record.kb_id,
        record.workspace,
        record.version,
        record.created_at,
        _record_json(record),
    )
    return "overwritten" if strategy == "overwrite" else "inserted"


def _bump(stats: MigrationTableStats, outcome: str) -> None:
    if outcome == "inserted":
        stats.inserted += 1
    elif outcome == "skipped":
        stats.skipped += 1
    elif outcome == "overwritten":
        stats.overwritten += 1
    else:  # pragma: no cover - defensive guard for future outcomes
        raise ValueError(f"Unknown migration outcome: {outcome}")


async def apply_snapshot_to_postgres(
    snapshot: ControlPlaneSnapshot,
    *,
    catalog_conn: Any,
    metadata_conn: Any,
    strategy: ConflictStrategy,
) -> MigrationSummary:
    summary = MigrationSummary(
        dry_run=False,
        strategy=strategy,
        source=snapshot.counts(),
        catalog=MigrationTableStats(),
        documents=MigrationTableStats(),
        jobs=MigrationTableStats(),
        artifacts=MigrationTableStats(),
        config_versions=MigrationTableStats(),
        source_keys_projected=0,
        issues=list(snapshot.issues),
    )

    migrated_kb_ids: set[str] = set()
    for record in snapshot.knowledge_bases:
        if strategy == "overwrite":
            conflict_ids = await _target_catalog_ids(catalog_conn, record)
            conflict_ids.add(record.id)
            await _delete_kb_metadata(metadata_conn, conflict_ids)
        outcome = await _insert_catalog(catalog_conn, record, strategy=strategy)
        _bump(summary.catalog, outcome)
        if outcome != "skipped":
            migrated_kb_ids.add(record.id)

    skipped_document_ids: set[tuple[str, str]] = set()
    kbs_with_skipped_documents: set[str] = set()
    for document in snapshot.documents:
        if _document_projection_source_key(document) is not None:
            summary.source_keys_projected += 1
        if document.kb_id not in migrated_kb_ids:
            summary.documents.skipped += 1
            skipped_document_ids.add((document.kb_id, document.id))
            kbs_with_skipped_documents.add(document.kb_id)
            continue
        outcome = await _insert_document(metadata_conn, document, strategy=strategy)
        _bump(summary.documents, outcome)
        if outcome == "skipped":
            skipped_document_ids.add((document.kb_id, document.id))
            kbs_with_skipped_documents.add(document.kb_id)

    for job in snapshot.jobs:
        if (
            job.kb_id not in migrated_kb_ids
            or (
                job.document_id is not None
                and (job.kb_id, job.document_id) in skipped_document_ids
            )
            or (job.document_id is None and job.kb_id in kbs_with_skipped_documents)
        ):
            summary.jobs.skipped += 1
            continue
        _bump(summary.jobs, await _insert_job(metadata_conn, job, strategy=strategy))

    for artifact in snapshot.artifacts:
        if artifact.kb_id not in migrated_kb_ids or (
            artifact.kb_id,
            artifact.document_id,
        ) in skipped_document_ids:
            summary.artifacts.skipped += 1
            continue
        _bump(
            summary.artifacts,
            await _insert_artifact(metadata_conn, artifact, strategy=strategy),
        )

    for record in snapshot.config_versions:
        if record.kb_id not in migrated_kb_ids:
            summary.config_versions.skipped += 1
            continue
        _bump(
            summary.config_versions,
            await _insert_config_version(metadata_conn, record, strategy=strategy),
        )
    return summary


def dry_run_summary(
    snapshot: ControlPlaneSnapshot, *, strategy: ConflictStrategy
) -> MigrationSummary:
    return MigrationSummary(
        dry_run=True,
        strategy=strategy,
        source=snapshot.counts(),
        catalog=MigrationTableStats(),
        documents=MigrationTableStats(),
        jobs=MigrationTableStats(),
        artifacts=MigrationTableStats(),
        config_versions=MigrationTableStats(),
        source_keys_projected=sum(
            1
            for document in snapshot.documents
            if _document_projection_source_key(document) is not None
        ),
        issues=list(snapshot.issues),
    )


def _postgres_services_from_args(args: argparse.Namespace) -> tuple[Any, Any]:
    if args.postgres_dsn:
        return (
            PostgresKnowledgeBaseService(dsn=args.postgres_dsn),
            PostgresMetadataStore(dsn=args.postgres_dsn),
        )
    return PostgresKnowledgeBaseService.from_env(), PostgresMetadataStore.from_env()


async def run_migration(args: argparse.Namespace) -> MigrationSummary:
    snapshot = await collect_local_snapshot(args.working_dir, kb_ids=args.kb_id)
    if args.dry_run:
        return dry_run_summary(snapshot, strategy=args.strategy)

    target_catalog, target_metadata = _postgres_services_from_args(args)
    await target_catalog.initialize()
    await target_metadata.initialize()
    try:
        catalog_pool = target_catalog._pool_or_raise()  # noqa: SLF001 - migration tool
        metadata_pool = target_metadata._pool_or_raise()  # noqa: SLF001 - migration tool
        async with catalog_pool.acquire() as catalog_conn:
            async with metadata_pool.acquire() as metadata_conn:
                async with catalog_conn.transaction():
                    async with metadata_conn.transaction():
                        return await apply_snapshot_to_postgres(
                            snapshot,
                            catalog_conn=catalog_conn,
                            metadata_conn=metadata_conn,
                            strategy=args.strategy,
                        )
    finally:
        await target_metadata.close()
        await target_catalog.close()


def _print_summary(summary: MigrationSummary, *, as_json: bool) -> None:
    payload = summary.to_dict()
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return

    mode = "DRY RUN" if summary.dry_run else "APPLIED"
    print(f"LightRAG KB metadata migration summary ({mode})")
    print(f"Strategy: {summary.strategy}")
    print("Source counts:")
    for key, value in summary.source.items():
        print(f"  - {key}: {value}")
    print(f"  - source_keys_projected: {summary.source_keys_projected}")
    if not summary.dry_run:
        for name in ("catalog", "documents", "jobs", "artifacts", "config_versions"):
            stats = getattr(summary, name)
            print(
                f"{name}: inserted={stats.inserted} "
                f"skipped={stats.skipped} overwritten={stats.overwritten}"
            )
    if summary.issues:
        print("Issues:")
        for issue in summary.issues:
            print(f"  - {issue}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate LightRAG API KB catalog/metadata from local "
            "JSON+SQLite storage to PostgreSQL."
        )
    )
    parser.add_argument(
        "--working-dir",
        default=os.getenv("WORKING_DIR", "./rag_storage"),
        help="Source LightRAG working directory (default: WORKING_DIR or ./rag_storage).",
    )
    parser.add_argument(
        "--postgres-dsn",
        default=None,
        help=(
            "Target PostgreSQL DSN. If omitted, LIGHTRAG_KB_POSTGRES_DSN/"
            "POSTGRES_DSN or split PostgreSQL env vars are used."
        ),
    )
    parser.add_argument(
        "--kb-id",
        action="append",
        help="Limit migration to a specific KB id. May be provided multiple times.",
    )
    parser.add_argument(
        "--strategy",
        choices=("fail", "skip", "overwrite"),
        default="fail",
        help="Target conflict handling strategy (default: fail).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and validate local metadata without connecting to PostgreSQL.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm destructive overwrite mode.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON summary.",
    )
    return parser


async def _async_main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.strategy == "overwrite" and not args.dry_run and not args.yes:
        parser.error("--strategy overwrite requires --yes unless --dry-run is set")
    summary = await run_migration(args)
    _print_summary(summary, as_json=args.json)
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
