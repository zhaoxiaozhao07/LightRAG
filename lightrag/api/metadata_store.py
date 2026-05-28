from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

from lightrag.api.kb_service import _MetadataFileLock, utc_now_iso

MetadataJobStatus = Literal[
    "queued", "running", "succeeded", "failed", "cancelling", "cancelled", "retrying"
]

DocumentStatus = Literal[
    "uploaded",
    "parse_queued",
    "parsing",
    "parsed",
    "parse_failed",
    "build_queued",
    "building",
    "ready",
    "build_failed",
    "deleted",
]

_SCHEMA_VERSION = 1
_T = TypeVar("_T")


class MetadataStoreError(RuntimeError):
    pass


class MetadataRecordNotFoundError(MetadataStoreError):
    pass


class InvalidJobTransitionError(MetadataStoreError):
    pass


class ActiveDocumentParseJobError(MetadataStoreError):
    def __init__(self, document_id: str, existing_job_id: str):
        self.document_id = document_id
        self.existing_job_id = existing_job_id
        super().__init__(
            f"Document '{document_id}' already has an active parse job"
        )


class ActiveDocumentBuildJobError(MetadataStoreError):
    def __init__(self, document_id: str, existing_job_id: str):
        self.document_id = document_id
        self.existing_job_id = existing_job_id
        super().__init__(
            f"Document '{document_id}' already has an active build job"
        )


class DocumentNotParsedError(MetadataStoreError):
    def __init__(self, document_id: str, current_status: str):
        self.document_id = document_id
        self.current_status = current_status
        super().__init__(
            f"Document '{document_id}' must be parsed before build (current status: {current_status})"
        )


class IdempotencyKeyConflictError(MetadataStoreError):
    def __init__(self, idempotency_key: str):
        self.idempotency_key = idempotency_key
        super().__init__(
            f"Idempotency key '{idempotency_key}' is already used for a different request"
        )


@dataclass(slots=True)
class DocumentRecord:
    id: str
    kb_id: str
    workspace: str
    lightrag_doc_id: str | None
    source_type: str
    source_name: str
    source_uri: str
    source_hash: str
    content_type: str | None
    size_bytes: int
    parser_hash: str | None
    index_hash: str | None
    status: str
    enabled: bool
    archived: bool
    chunks_count: int | None
    entity_count: int | None
    relation_count: int | None
    error_code: str | None
    error_message: str | None
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    deleted_at: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "DocumentRecord":
        return cls(
            id=str(row["id"]),
            kb_id=str(row["kb_id"]),
            workspace=str(row["workspace"]),
            lightrag_doc_id=row["lightrag_doc_id"],
            source_type=str(row["source_type"]),
            source_name=str(row["source_name"]),
            source_uri=str(row["source_uri"]),
            source_hash=str(row["source_hash"]),
            content_type=row["content_type"],
            size_bytes=int(row["size_bytes"]),
            parser_hash=row["parser_hash"],
            index_hash=row["index_hash"],
            status=str(row["status"]),
            enabled=bool(row["enabled"]),
            archived=bool(row["archived"]),
            chunks_count=row["chunks_count"],
            entity_count=row["entity_count"],
            relation_count=row["relation_count"],
            error_code=row["error_code"],
            error_message=row["error_message"],
            metadata=_loads_json_object(row["metadata_json"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            deleted_at=row["deleted_at"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class JobRecord:
    id: str
    kb_id: str
    workspace: str
    batch_id: str | None
    document_id: str | None
    job_type: str
    status: str
    stage: str | None
    progress: float
    total_items: int
    completed_items: int
    failed_items: int
    idempotency_key: str | None
    config_version_id: str | None
    config_hash: str | None
    retry_count: int
    max_retries: int
    payload: dict[str, Any]
    result: dict[str, Any] | None
    error_code: str | None
    error_message: str | None
    created_at: str
    updated_at: str
    queued_at: str | None
    started_at: str | None
    finished_at: str | None
    cancelled_at: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "JobRecord":
        result_json = row["result_json"]
        return cls(
            id=str(row["id"]),
            kb_id=str(row["kb_id"]),
            workspace=str(row["workspace"]),
            batch_id=row["batch_id"],
            document_id=row["document_id"],
            job_type=str(row["job_type"]),
            status=str(row["status"]),
            stage=row["stage"],
            progress=float(row["progress"]),
            total_items=int(row["total_items"]),
            completed_items=int(row["completed_items"]),
            failed_items=int(row["failed_items"]),
            idempotency_key=row["idempotency_key"],
            config_version_id=row["config_version_id"],
            config_hash=row["config_hash"],
            retry_count=int(row["retry_count"]),
            max_retries=int(row["max_retries"]),
            payload=_loads_json_object(row["payload_json"]),
            result=_loads_optional_json_object(result_json),
            error_code=row["error_code"],
            error_message=row["error_message"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            queued_at=row["queued_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            cancelled_at=row["cancelled_at"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ArtifactRecord:
    id: str
    kb_id: str
    workspace: str
    document_id: str
    artifact_type: str
    uri: str
    checksum: str | None
    size_bytes: int | None
    metadata: dict[str, Any]
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ArtifactRecord":
        return cls(
            id=str(row["id"]),
            kb_id=str(row["kb_id"]),
            workspace=str(row["workspace"]),
            document_id=str(row["document_id"]),
            artifact_type=str(row["artifact_type"]),
            uri=str(row["uri"]),
            checksum=row["checksum"],
            size_bytes=row["size_bytes"],
            metadata=_loads_json_object(row["metadata_json"]),
            created_at=str(row["created_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ConfigVersionRecord:
    id: str
    kb_id: str
    workspace: str
    version: int
    config: dict[str, Any]
    parser_hash: str | None
    index_hash: str | None
    query_hash: str | None
    created_at: str
    activated_at: str | None
    created_by: str | None


def _dumps_json(value: dict[str, Any] | None) -> str:
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True)


def _loads_json_object(value: str | bytes | None) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise MetadataStoreError("Metadata JSON must be an object")
    return loaded


def _loads_optional_json_object(value: str | bytes | None) -> dict[str, Any] | None:
    if value in (None, ""):
        return None
    return _loads_json_object(value)


class SQLiteMetadataStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.lock_path = Path(f"{self.db_path}.lock")
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        async with self._lock:
            with _MetadataFileLock(self.lock_path):
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                with self._connect() as conn:
                    self._initialize_schema(conn)
                self._initialized = True

    async def create_documents_and_job(
        self,
        documents: Sequence[DocumentRecord],
        job: JobRecord,
    ) -> tuple[list[DocumentRecord], JobRecord, bool]:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> tuple[list[DocumentRecord], JobRecord, bool]:
            existing = self._get_job_by_idempotency_key(
                conn, job.kb_id, job.idempotency_key, job_type=job.job_type
            )
            if existing is not None:
                self._validate_idempotent_job(existing, job)
                return self._documents_for_job(conn, existing), existing, False
            for document in documents:
                self._insert_document(conn, document)
            self._insert_job(conn, job)
            return list(documents), job, True

        return await self._write(write)

    async def list_documents(
        self,
        kb_id: str,
        *,
        status: str | None = None,
        source_name: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[DocumentRecord], int]:
        await self._ensure_initialized()
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        with self._connect() as conn:
            where = "kb_id = ? AND deleted_at IS NULL"
            params: list[Any] = [kb_id]
            if status is not None:
                where += " AND status = ?"
                params.append(status)
            if source_name is not None:
                where += " AND source_name LIKE ? ESCAPE '\\' COLLATE NOCASE"
                params.append(f"%{_escape_like(source_name)}%")
            total = conn.execute(
                f"SELECT COUNT(*) FROM documents WHERE {where}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"""
                SELECT * FROM documents
                WHERE {where}
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()
        return [DocumentRecord.from_row(row) for row in rows], int(total)

    async def get_document(self, kb_id: str, document_id: str) -> DocumentRecord:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM documents
                WHERE kb_id = ? AND id = ? AND deleted_at IS NULL
                """,
                (kb_id, document_id),
            ).fetchone()
        if row is None:
            raise MetadataRecordNotFoundError(f"Document '{document_id}' not found")
        return DocumentRecord.from_row(row)

    async def get_documents_by_ids(
        self, kb_id: str, document_ids: Sequence[str]
    ) -> list[DocumentRecord]:
        await self._ensure_initialized()
        if not document_ids:
            return []
        placeholders = ", ".join("?" for _ in document_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM documents
                WHERE kb_id = ? AND id IN ({placeholders}) AND deleted_at IS NULL
                """,
                [kb_id, *document_ids],
            ).fetchall()
        records_by_id = {row["id"]: DocumentRecord.from_row(row) for row in rows}
        return [records_by_id[document_id] for document_id in document_ids if document_id in records_by_id]

    async def list_documents_by_batch_id(
        self, kb_id: str, batch_id: str
    ) -> list[DocumentRecord]:
        await self._ensure_initialized()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM documents
                WHERE kb_id = ? AND deleted_at IS NULL
                ORDER BY created_at ASC, id ASC
                """,
                (kb_id,),
            ).fetchall()
        documents = [DocumentRecord.from_row(row) for row in rows]
        return [
            document
            for document in documents
            if document.metadata.get("batch_id") == batch_id
        ]

    async def update_document(
        self,
        kb_id: str,
        document_id: str,
        *,
        metadata_patch: dict[str, Any] | None = None,
        enabled: bool | None = None,
        archived: bool | None = None,
    ) -> DocumentRecord:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> DocumentRecord:
            current_row = conn.execute(
                """
                SELECT * FROM documents
                WHERE kb_id = ? AND id = ? AND deleted_at IS NULL
                """,
                (kb_id, document_id),
            ).fetchone()
            if current_row is None:
                raise MetadataRecordNotFoundError(f"Document '{document_id}' not found")
            metadata = _loads_json_object(current_row["metadata_json"])
            if metadata_patch:
                metadata.update(metadata_patch)
            now = utc_now_iso()
            conn.execute(
                """
                UPDATE documents
                SET enabled = ?, archived = ?, metadata_json = ?, updated_at = ?
                WHERE kb_id = ? AND id = ?
                """,
                (
                    int(enabled) if enabled is not None else current_row["enabled"],
                    int(archived) if archived is not None else current_row["archived"],
                    _dumps_json(metadata),
                    now,
                    kb_id,
                    document_id,
                ),
            )
            row = conn.execute(
                "SELECT * FROM documents WHERE kb_id = ? AND id = ?",
                (kb_id, document_id),
            ).fetchone()
            if row is None:
                raise MetadataRecordNotFoundError(f"Document '{document_id}' not found")
            return DocumentRecord.from_row(row)

        return await self._write(write)

    async def mark_document_parse_queued(
        self,
        kb_id: str,
        document_id: str,
        *,
        metadata_patch: dict[str, Any],
    ) -> DocumentRecord:
        await self._ensure_initialized()
        return await self._write(
            lambda conn: self._claim_document_parse_queued(
                conn,
                kb_id,
                document_id,
                metadata_patch=metadata_patch,
                raise_on_active=True,
            )
        )

    async def claim_documents_parse_queued(
        self,
        kb_id: str,
        claims: Sequence[tuple[str, dict[str, Any]]],
    ) -> tuple[list[DocumentRecord], list[dict[str, Any]]]:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> tuple[list[DocumentRecord], list[dict[str, Any]]]:
            documents: list[DocumentRecord] = []
            failures: list[dict[str, Any]] = []
            for document_id, metadata_patch in claims:
                try:
                    documents.append(
                        self._claim_document_parse_queued(
                            conn,
                            kb_id,
                            document_id,
                            metadata_patch=metadata_patch,
                            raise_on_active=True,
                        )
                    )
                except ActiveDocumentParseJobError as exc:
                    failures.append(
                        {
                            "document_id": document_id,
                            "status": "failed",
                            "error_code": "parse_job_active",
                            "error_message": str(exc),
                            "existing_job_id": exc.existing_job_id,
                        }
                    )
                except MetadataRecordNotFoundError as exc:
                    failures.append(
                        {
                            "document_id": document_id,
                            "status": "failed",
                            "error_code": "document_not_found",
                            "error_message": str(exc),
                        }
                    )
            return documents, failures

        return await self._write(write)

    async def mark_document_parsing(
        self,
        kb_id: str,
        document_id: str,
        *,
        metadata_patch: dict[str, Any],
    ) -> DocumentRecord:
        await self._ensure_initialized()
        return await self._write(
            lambda conn: self._update_document_parse_state(
                conn,
                kb_id,
                document_id,
                status="parsing",
                metadata_patch=metadata_patch,
                clear_error=True,
            )
        )

    async def complete_document_parse(
        self,
        kb_id: str,
        document_id: str,
        *,
        parser_hash: str,
        lightrag_doc_id: str,
        metadata_patch: dict[str, Any],
        artifacts: Sequence[ArtifactRecord],
    ) -> tuple[DocumentRecord, list[ArtifactRecord]]:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> tuple[DocumentRecord, list[ArtifactRecord]]:
            document = self._update_document_parse_state(
                conn,
                kb_id,
                document_id,
                status="parsed",
                metadata_patch=metadata_patch,
                parser_hash=parser_hash,
                lightrag_doc_id=lightrag_doc_id,
                clear_error=True,
            )
            conn.execute(
                "DELETE FROM document_artifacts WHERE kb_id = ? AND document_id = ?",
                (kb_id, document_id),
            )
            for artifact in artifacts:
                self._insert_artifact(conn, artifact)
            return document, list(artifacts)

        return await self._write(write)

    async def fail_document_parse(
        self,
        kb_id: str,
        document_id: str,
        *,
        error_code: str,
        error_message: str,
        metadata_patch: dict[str, Any],
    ) -> DocumentRecord:
        await self._ensure_initialized()
        return await self._write(
            lambda conn: self._update_document_parse_state(
                conn,
                kb_id,
                document_id,
                status="parse_failed",
                metadata_patch=metadata_patch,
                error_code=error_code,
                error_message=error_message,
            )
        )

    async def claim_document_build_queued(
        self,
        kb_id: str,
        document_id: str,
        *,
        metadata_patch: dict[str, Any],
        require_parsed: bool = True,
    ) -> DocumentRecord:
        await self._ensure_initialized()
        return await self._write(
            lambda conn: self._claim_document_build_queued(
                conn,
                kb_id,
                document_id,
                metadata_patch=metadata_patch,
                require_parsed=require_parsed,
            )
        )

    async def claim_documents_build_queued(
        self,
        kb_id: str,
        claims: Sequence[tuple[str, dict[str, Any]]],
        *,
        require_parsed: bool = True,
    ) -> tuple[list[DocumentRecord], list[dict[str, Any]]]:
        await self._ensure_initialized()

        def write(
            conn: sqlite3.Connection,
        ) -> tuple[list[DocumentRecord], list[dict[str, Any]]]:
            documents: list[DocumentRecord] = []
            failures: list[dict[str, Any]] = []
            for document_id, metadata_patch in claims:
                try:
                    documents.append(
                        self._claim_document_build_queued(
                            conn,
                            kb_id,
                            document_id,
                            metadata_patch=metadata_patch,
                            require_parsed=require_parsed,
                        )
                    )
                except ActiveDocumentBuildJobError as exc:
                    failures.append(
                        {
                            "document_id": document_id,
                            "status": "failed",
                            "error_code": "build_job_active",
                            "error_message": str(exc),
                            "existing_job_id": exc.existing_job_id,
                        }
                    )
                except DocumentNotParsedError as exc:
                    failures.append(
                        {
                            "document_id": document_id,
                            "status": "failed",
                            "error_code": "document_not_parsed",
                            "error_message": str(exc),
                            "current_status": exc.current_status,
                        }
                    )
                except MetadataRecordNotFoundError as exc:
                    failures.append(
                        {
                            "document_id": document_id,
                            "status": "failed",
                            "error_code": "document_not_found",
                            "error_message": str(exc),
                        }
                    )
            return documents, failures

        return await self._write(write)

    async def mark_document_building(
        self,
        kb_id: str,
        document_id: str,
        *,
        metadata_patch: dict[str, Any],
    ) -> DocumentRecord:
        await self._ensure_initialized()
        return await self._write(
            lambda conn: self._update_document_parse_state(
                conn,
                kb_id,
                document_id,
                status="building",
                metadata_patch=metadata_patch,
                clear_error=True,
            )
        )

    async def complete_document_build(
        self,
        kb_id: str,
        document_id: str,
        *,
        index_hash: str,
        chunks_count: int | None = None,
        entity_count: int | None = None,
        relation_count: int | None = None,
        metadata_patch: dict[str, Any],
    ) -> DocumentRecord:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> DocumentRecord:
            current_row = conn.execute(
                """
                SELECT * FROM documents
                WHERE kb_id = ? AND id = ? AND deleted_at IS NULL
                """,
                (kb_id, document_id),
            ).fetchone()
            if current_row is None:
                raise MetadataRecordNotFoundError(f"Document '{document_id}' not found")
            metadata = _loads_json_object(current_row["metadata_json"])
            metadata.update(metadata_patch)
            now = utc_now_iso()
            conn.execute(
                """
                UPDATE documents
                SET status = ?, index_hash = ?, chunks_count = ?, entity_count = ?,
                    relation_count = ?, error_code = NULL, error_message = NULL,
                    metadata_json = ?, updated_at = ?
                WHERE kb_id = ? AND id = ?
                """,
                (
                    "ready",
                    index_hash,
                    chunks_count
                    if chunks_count is not None
                    else current_row["chunks_count"],
                    entity_count
                    if entity_count is not None
                    else current_row["entity_count"],
                    relation_count
                    if relation_count is not None
                    else current_row["relation_count"],
                    _dumps_json(metadata),
                    now,
                    kb_id,
                    document_id,
                ),
            )
            row = conn.execute(
                "SELECT * FROM documents WHERE kb_id = ? AND id = ?",
                (kb_id, document_id),
            ).fetchone()
            if row is None:
                raise MetadataRecordNotFoundError(f"Document '{document_id}' not found")
            return DocumentRecord.from_row(row)

        return await self._write(write)

    async def fail_document_build(
        self,
        kb_id: str,
        document_id: str,
        *,
        error_code: str,
        error_message: str,
        metadata_patch: dict[str, Any],
    ) -> DocumentRecord:
        await self._ensure_initialized()
        return await self._write(
            lambda conn: self._update_document_parse_state(
                conn,
                kb_id,
                document_id,
                status="build_failed",
                metadata_patch=metadata_patch,
                error_code=error_code,
                error_message=error_message,
            )
        )

    async def list_document_artifacts(
        self,
        kb_id: str,
        document_id: str,
        *,
        artifact_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[ArtifactRecord], int]:
        await self._ensure_initialized()
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        where = "kb_id = ? AND document_id = ?"
        params: list[Any] = [kb_id, document_id]
        if artifact_type is not None:
            where += " AND artifact_type = ?"
            params.append(artifact_type)
        with self._connect() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM document_artifacts WHERE {where}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"""
                SELECT * FROM document_artifacts
                WHERE {where}
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()
        return [ArtifactRecord.from_row(row) for row in rows], int(total)

    async def get_document_artifact(
        self, kb_id: str, document_id: str, artifact_id: str
    ) -> ArtifactRecord:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM document_artifacts
                WHERE kb_id = ? AND document_id = ? AND id = ?
                """,
                (kb_id, document_id, artifact_id),
            ).fetchone()
        if row is None:
            raise MetadataRecordNotFoundError(f"Artifact '{artifact_id}' not found")
        return ArtifactRecord.from_row(row)

    async def create_job(self, job: JobRecord) -> JobRecord:
        await self._ensure_initialized()

        created_job, _created = await self.create_job_once(job)
        return created_job

    async def create_job_once(self, job: JobRecord) -> tuple[JobRecord, bool]:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> tuple[JobRecord, bool]:
            existing = self._get_job_by_idempotency_key(
                conn, job.kb_id, job.idempotency_key, job_type=job.job_type
            )
            if existing is not None:
                self._validate_idempotent_job(existing, job)
                return existing, False
            return self._insert_job(conn, job), True

        return await self._write(write)

    async def get_job_by_idempotency_key(
        self, kb_id: str, idempotency_key: str, *, job_type: str | None = None
    ) -> JobRecord | None:
        await self._ensure_initialized()
        where = "kb_id = ? AND idempotency_key = ?"
        params: list[Any] = [kb_id, idempotency_key]
        if job_type is not None:
            where += " AND job_type = ?"
            params.append(job_type)
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT * FROM jobs
                WHERE {where}
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                params,
            ).fetchone()
        return JobRecord.from_row(row) if row is not None else None

    async def list_jobs(
        self,
        kb_id: str,
        *,
        statuses: Sequence[str] | None = None,
        document_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[JobRecord], int]:
        await self._ensure_initialized()
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        where = "kb_id = ?"
        params: list[Any] = [kb_id]
        if document_id is not None:
            where += " AND document_id = ?"
            params.append(document_id)
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            where += f" AND status IN ({placeholders})"
            params.extend(statuses)
        with self._connect() as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM jobs WHERE {where}", params).fetchone()[0]
            rows = conn.execute(
                f"""
                SELECT * FROM jobs
                WHERE {where}
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()
        return [JobRecord.from_row(row) for row in rows], int(total)

    async def get_job(self, kb_id: str, job_id: str) -> JobRecord:
        await self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE kb_id = ? AND id = ?",
                (kb_id, job_id),
            ).fetchone()
        if row is None:
            raise MetadataRecordNotFoundError(f"Job '{job_id}' not found")
        return JobRecord.from_row(row)

    async def transition_job(
        self,
        kb_id: str,
        job_id: str,
        *,
        status: MetadataJobStatus,
        stage: str | None = None,
        progress: float | None = None,
        completed_items: int | None = None,
        failed_items: int | None = None,
        result: dict[str, Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> JobRecord:
        await self._ensure_initialized()

        def write(conn: sqlite3.Connection) -> JobRecord:
            current_row = conn.execute(
                "SELECT * FROM jobs WHERE kb_id = ? AND id = ?",
                (kb_id, job_id),
            ).fetchone()
            if current_row is None:
                raise MetadataRecordNotFoundError(f"Job '{job_id}' not found")
            current = JobRecord.from_row(current_row)
            if status not in _allowed_next_job_statuses(current.status):
                raise InvalidJobTransitionError(
                    f"Cannot transition job '{job_id}' from {current.status} to {status}"
                )

            now = utc_now_iso()
            started_at = current.started_at
            finished_at = current.finished_at
            cancelled_at = current.cancelled_at
            if status == "running" and started_at is None:
                started_at = now
            if status in {"succeeded", "failed"} and finished_at is None:
                finished_at = now
            if status == "cancelled" and cancelled_at is None:
                cancelled_at = now

            conn.execute(
                """
                UPDATE jobs
                SET status = ?, stage = ?, progress = ?, completed_items = ?,
                    failed_items = ?, result_json = ?, error_code = ?,
                    error_message = ?, updated_at = ?,
                    started_at = ?, finished_at = ?, cancelled_at = ?
                WHERE kb_id = ? AND id = ?
                """,
                (
                    status,
                    stage if stage is not None else current.stage,
                    progress if progress is not None else current.progress,
                    completed_items
                    if completed_items is not None
                    else current.completed_items,
                    failed_items if failed_items is not None else current.failed_items,
                    _dumps_json(result) if result is not None else current_row["result_json"],
                    error_code,
                    error_message,
                    now,
                    started_at,
                    finished_at,
                    cancelled_at,
                    kb_id,
                    job_id,
                ),
            )
            row = conn.execute(
                "SELECT * FROM jobs WHERE kb_id = ? AND id = ?",
                (kb_id, job_id),
            ).fetchone()
            if row is None:
                raise MetadataRecordNotFoundError(f"Job '{job_id}' not found")
            return JobRecord.from_row(row)

        return await self._write(write)

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def _write(self, callback: Callable[[sqlite3.Connection], _T]) -> _T:
        async with self._lock:
            with _MetadataFileLock(self.lock_path):
                with self._connect() as conn:
                    try:
                        conn.execute("BEGIN IMMEDIATE")
                        result = callback(conn)
                        conn.commit()
                        return result
                    except Exception:
                        conn.rollback()
                        raise

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _initialize_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata_schema (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                kb_id TEXT NOT NULL,
                workspace TEXT NOT NULL,
                lightrag_doc_id TEXT,
                source_type TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_uri TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                content_type TEXT,
                size_bytes INTEGER NOT NULL DEFAULT 0,
                parser_hash TEXT,
                index_hash TEXT,
                status TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                archived INTEGER NOT NULL DEFAULT 0,
                chunks_count INTEGER,
                entity_count INTEGER,
                relation_count INTEGER,
                error_code TEXT,
                error_message TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_documents_kb_status
                ON documents (kb_id, status);
            CREATE INDEX IF NOT EXISTS idx_documents_kb_source_hash
                ON documents (kb_id, source_hash);
            CREATE INDEX IF NOT EXISTS idx_documents_workspace
                ON documents (workspace);

            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                kb_id TEXT NOT NULL,
                workspace TEXT NOT NULL,
                batch_id TEXT,
                document_id TEXT,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL,
                stage TEXT,
                progress REAL NOT NULL DEFAULT 0,
                total_items INTEGER NOT NULL DEFAULT 0,
                completed_items INTEGER NOT NULL DEFAULT 0,
                failed_items INTEGER NOT NULL DEFAULT 0,
                idempotency_key TEXT,
                config_version_id TEXT,
                config_hash TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0,
                max_retries INTEGER NOT NULL DEFAULT 3,
                payload_json TEXT NOT NULL DEFAULT '{}',
                result_json TEXT,
                error_code TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                queued_at TEXT,
                started_at TEXT,
                finished_at TEXT,
                cancelled_at TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            );

            CREATE INDEX IF NOT EXISTS idx_jobs_kb_status
                ON jobs (kb_id, status);
            CREATE INDEX IF NOT EXISTS idx_jobs_kb_document
                ON jobs (kb_id, document_id);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_kb_type_idempotency
                ON jobs (kb_id, job_type, idempotency_key)
                WHERE idempotency_key IS NOT NULL;

            CREATE TABLE IF NOT EXISTS document_artifacts (
                id TEXT PRIMARY KEY,
                kb_id TEXT NOT NULL,
                workspace TEXT NOT NULL,
                document_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                uri TEXT NOT NULL,
                checksum TEXT,
                size_bytes INTEGER,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            );

            CREATE INDEX IF NOT EXISTS idx_artifacts_kb_document
                ON document_artifacts (kb_id, document_id);
            CREATE INDEX IF NOT EXISTS idx_artifacts_workspace_type
                ON document_artifacts (workspace, artifact_type);

            CREATE TABLE IF NOT EXISTS kb_config_versions (
                id TEXT PRIMARY KEY,
                kb_id TEXT NOT NULL,
                workspace TEXT NOT NULL,
                version INTEGER NOT NULL,
                config_json TEXT NOT NULL,
                parser_hash TEXT,
                index_hash TEXT,
                query_hash TEXT,
                created_at TEXT NOT NULL,
                activated_at TEXT,
                created_by TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_config_versions_kb_version
                ON kb_config_versions (kb_id, version);
            CREATE INDEX IF NOT EXISTS idx_config_versions_workspace
                ON kb_config_versions (workspace);
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO metadata_schema(version, applied_at) VALUES (?, ?)",
            (_SCHEMA_VERSION, utc_now_iso()),
        )
        conn.commit()

    def _insert_document(
        self, conn: sqlite3.Connection, document: DocumentRecord
    ) -> DocumentRecord:
        conn.execute(
            """
            INSERT INTO documents (
                id, kb_id, workspace, lightrag_doc_id, source_type, source_name,
                source_uri, source_hash, content_type, size_bytes, parser_hash,
                index_hash, status, enabled, archived, chunks_count, entity_count,
                relation_count, error_code, error_message, metadata_json,
                created_at, updated_at, deleted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document.id,
                document.kb_id,
                document.workspace,
                document.lightrag_doc_id,
                document.source_type,
                document.source_name,
                document.source_uri,
                document.source_hash,
                document.content_type,
                document.size_bytes,
                document.parser_hash,
                document.index_hash,
                document.status,
                int(document.enabled),
                int(document.archived),
                document.chunks_count,
                document.entity_count,
                document.relation_count,
                document.error_code,
                document.error_message,
                _dumps_json(document.metadata),
                document.created_at,
                document.updated_at,
                document.deleted_at,
            ),
        )
        return document

    def _insert_job(self, conn: sqlite3.Connection, job: JobRecord) -> JobRecord:
        conn.execute(
            """
            INSERT INTO jobs (
                id, kb_id, workspace, batch_id, document_id, job_type, status,
                stage, progress, total_items, completed_items, failed_items,
                idempotency_key, config_version_id, config_hash, retry_count,
                max_retries, payload_json, result_json, error_code, error_message,
                created_at, updated_at, queued_at, started_at, finished_at,
                cancelled_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job.id,
                job.kb_id,
                job.workspace,
                job.batch_id,
                job.document_id,
                job.job_type,
                job.status,
                job.stage,
                job.progress,
                job.total_items,
                job.completed_items,
                job.failed_items,
                job.idempotency_key,
                job.config_version_id,
                job.config_hash,
                job.retry_count,
                job.max_retries,
                _dumps_json(job.payload),
                _dumps_json(job.result) if job.result is not None else None,
                job.error_code,
                job.error_message,
                job.created_at,
                job.updated_at,
                job.queued_at,
                job.started_at,
                job.finished_at,
                job.cancelled_at,
            ),
        )
        return job

    def _insert_artifact(
        self, conn: sqlite3.Connection, artifact: ArtifactRecord
    ) -> ArtifactRecord:
        conn.execute(
            """
            INSERT INTO document_artifacts (
                id, kb_id, workspace, document_id, artifact_type, uri, checksum,
                size_bytes, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact.id,
                artifact.kb_id,
                artifact.workspace,
                artifact.document_id,
                artifact.artifact_type,
                artifact.uri,
                artifact.checksum,
                artifact.size_bytes,
                _dumps_json(artifact.metadata),
                artifact.created_at,
            ),
        )
        return artifact

    def _get_job_by_idempotency_key(
        self,
        conn: sqlite3.Connection,
        kb_id: str,
        idempotency_key: str | None,
        *,
        job_type: str | None = None,
    ) -> JobRecord | None:
        if not idempotency_key:
            return None
        where = "kb_id = ? AND idempotency_key = ?"
        params: list[Any] = [kb_id, idempotency_key]
        if job_type is not None:
            where += " AND job_type = ?"
            params.append(job_type)
        row = conn.execute(
            f"""
            SELECT * FROM jobs
            WHERE {where}
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            params,
        ).fetchone()
        return JobRecord.from_row(row) if row is not None else None

    def _validate_idempotent_job(
        self, existing: JobRecord, candidate: JobRecord
    ) -> None:
        existing_fingerprint = existing.payload.get("idempotency_fingerprint")
        candidate_fingerprint = candidate.payload.get("idempotency_fingerprint")
        if existing_fingerprint != candidate_fingerprint:
            raise IdempotencyKeyConflictError(candidate.idempotency_key or "")

    def _documents_for_job(
        self, conn: sqlite3.Connection, job: JobRecord
    ) -> list[DocumentRecord]:
        document_ids = job.payload.get("document_ids")
        if not isinstance(document_ids, list) or not all(
            isinstance(document_id, str) for document_id in document_ids
        ):
            if not job.batch_id:
                return []
            rows = conn.execute(
                """
                SELECT * FROM documents
                WHERE kb_id = ? AND deleted_at IS NULL
                ORDER BY created_at ASC, id ASC
                """,
                (job.kb_id,),
            ).fetchall()
            return [
                DocumentRecord.from_row(row)
                for row in rows
                if _loads_json_object(row["metadata_json"]).get("batch_id")
                == job.batch_id
            ]
        if not document_ids:
            return []
        placeholders = ", ".join("?" for _ in document_ids)
        rows = conn.execute(
            f"""
            SELECT * FROM documents
            WHERE kb_id = ? AND id IN ({placeholders}) AND deleted_at IS NULL
            """,
            [job.kb_id, *document_ids],
        ).fetchall()
        documents_by_id = {row["id"]: DocumentRecord.from_row(row) for row in rows}
        return [
            documents_by_id[document_id]
            for document_id in document_ids
            if document_id in documents_by_id
        ]

    def _claim_document_parse_queued(
        self,
        conn: sqlite3.Connection,
        kb_id: str,
        document_id: str,
        *,
        metadata_patch: dict[str, Any],
        raise_on_active: bool,
    ) -> DocumentRecord:
        current_row = conn.execute(
            """
            SELECT * FROM documents
            WHERE kb_id = ? AND id = ? AND deleted_at IS NULL
            """,
            (kb_id, document_id),
        ).fetchone()
        if current_row is None:
            raise MetadataRecordNotFoundError(f"Document '{document_id}' not found")
        if raise_on_active and current_row["status"] in {"parse_queued", "parsing"}:
            raise ActiveDocumentParseJobError(
                document_id,
                _active_parse_job_id_from_row(current_row),
            )
        return self._update_document_parse_state(
            conn,
            kb_id,
            document_id,
            status="parse_queued",
            metadata_patch=metadata_patch,
            clear_error=True,
        )

    def _update_document_parse_state(
        self,
        conn: sqlite3.Connection,
        kb_id: str,
        document_id: str,
        *,
        status: DocumentStatus,
        metadata_patch: dict[str, Any],
        parser_hash: str | None = None,
        lightrag_doc_id: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        clear_error: bool = False,
    ) -> DocumentRecord:
        current_row = conn.execute(
            """
            SELECT * FROM documents
            WHERE kb_id = ? AND id = ? AND deleted_at IS NULL
            """,
            (kb_id, document_id),
        ).fetchone()
        if current_row is None:
            raise MetadataRecordNotFoundError(f"Document '{document_id}' not found")

        metadata = _loads_json_object(current_row["metadata_json"])
        metadata.update(metadata_patch)
        now = utc_now_iso()
        next_parser_hash = parser_hash if parser_hash is not None else current_row["parser_hash"]
        next_lightrag_doc_id = (
            lightrag_doc_id if lightrag_doc_id is not None else current_row["lightrag_doc_id"]
        )
        next_error_code = None if clear_error else error_code
        next_error_message = None if clear_error else error_message
        conn.execute(
            """
            UPDATE documents
            SET status = ?, parser_hash = ?, lightrag_doc_id = ?, error_code = ?,
                error_message = ?, metadata_json = ?, updated_at = ?
            WHERE kb_id = ? AND id = ?
            """,
            (
                status,
                next_parser_hash,
                next_lightrag_doc_id,
                next_error_code,
                next_error_message,
                _dumps_json(metadata),
                now,
                kb_id,
                document_id,
            ),
        )
        row = conn.execute(
            "SELECT * FROM documents WHERE kb_id = ? AND id = ?",
            (kb_id, document_id),
        ).fetchone()
        if row is None:
            raise MetadataRecordNotFoundError(f"Document '{document_id}' not found")
        return DocumentRecord.from_row(row)


def _allowed_next_job_statuses(current: str) -> set[str]:
    transitions = {
        "queued": {"running", "cancelled", "failed"},
        "running": {"succeeded", "failed", "cancelling"},
        "cancelling": {"cancelled", "failed"},
        "retrying": {"queued", "running", "failed"},
        "succeeded": set(),
        "failed": {"retrying"},
        "cancelled": {"retrying"},
    }
    return transitions.get(current, set())


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _active_parse_job_id_from_row(row: sqlite3.Row) -> str:
    metadata = _loads_json_object(row["metadata_json"])
    if row["status"] == "parse_queued":
        job_id = metadata.get("pending_parse_job_id")
        return str(job_id) if job_id else "unknown"
    if row["status"] == "parsing":
        job_id = metadata.get("current_parse_job_id")
        return str(job_id) if job_id else "unknown"
    return "unknown"
