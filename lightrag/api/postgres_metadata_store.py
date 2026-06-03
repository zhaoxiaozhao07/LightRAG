from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Any, Awaitable, Callable, Sequence, TypeVar

from lightrag.api.kb_service import utc_now_iso
from lightrag.api.metadata_store import (
    _AGGREGATE_RESUMABLE_JOB_TYPES,
    _REPLACE_DERIVED_METADATA_KEYS,
    _allowed_next_job_statuses,
    _escape_like,
    _metadata_source_key,
    ActiveDocumentBuildJobError,
    ActiveDocumentDeleteJobError,
    ActiveDocumentParseJobError,
    ActiveDocumentReplaceJobError,
    ArtifactRecord,
    ConfigVersionRecord,
    DocumentNotParsedError,
    DocumentRecord,
    DuplicateDocumentSourceKeyError,
    IdempotencyKeyConflictError,
    InvalidJobTransitionError,
    JobRecord,
    MetadataJobStatus,
    MetadataRecordNotFoundError,
)

_T = TypeVar("_T")


def _load_asyncpg() -> Any:
    try:
        import asyncpg
    except ImportError as exc:  # pragma: no cover - exercised only without optional deps
        raise RuntimeError(
            "PostgreSQL KB metadata backend requires asyncpg. "
            "Install LightRAG with the api/offline-storage extras or install asyncpg."
        ) from exc
    return asyncpg


def _loads_json_object(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return dict(value)
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise RuntimeError("Metadata JSON must be an object")
    return loaded


def _dumps_json(value: dict[str, Any] | None) -> str:
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True)


def _document_from_row(row: Any) -> DocumentRecord:
    data = _loads_json_object(row["data_json"])
    return DocumentRecord(**data)


def _job_from_row(row: Any) -> JobRecord:
    data = _loads_json_object(row["data_json"])
    return JobRecord(**data)


def _artifact_from_row(row: Any) -> ArtifactRecord:
    data = _loads_json_object(row["data_json"])
    return ArtifactRecord(**data)


def _config_from_row(row: Any) -> ConfigVersionRecord:
    data = _loads_json_object(row["data_json"])
    return ConfigVersionRecord(**data)


def _record_json(record: Any) -> str:
    return json.dumps(asdict(record), ensure_ascii=False, sort_keys=True)


def _active_parse_job_id(document: DocumentRecord) -> str:
    if document.status == "parse_queued":
        job_id = document.metadata.get("pending_parse_job_id")
        return str(job_id) if job_id else "unknown"
    if document.status == "parsing":
        job_id = document.metadata.get("current_parse_job_id")
        return str(job_id) if job_id else "unknown"
    return "unknown"


def _active_build_job_id(document: DocumentRecord) -> str:
    if document.status == "build_queued":
        job_id = document.metadata.get("pending_build_job_id")
        return str(job_id) if job_id else "unknown"
    if document.status == "building":
        job_id = document.metadata.get("current_build_job_id")
        return str(job_id) if job_id else "unknown"
    return "unknown"


def _active_delete_job_id(document: DocumentRecord) -> str:
    job_id = document.metadata.get("pending_delete_job_id") or document.metadata.get(
        "current_delete_job_id"
    )
    return str(job_id) if job_id else "unknown"


def _active_replace_job_id(document: DocumentRecord) -> str:
    job_id = document.metadata.get("pending_replace_job_id") or document.metadata.get(
        "current_replace_job_id"
    )
    return str(job_id) if job_id else "unknown"


class PostgresMetadataStore:
    """PostgreSQL-backed KB control-plane metadata store.

    The public methods intentionally mirror :class:`SQLiteMetadataStore` so the
    API services can switch backends without changing route logic. Records are
    persisted as JSONB plus indexed projection columns that are needed for list,
    idempotency, source-key, and worker-claim queries.
    """

    def __init__(
        self,
        dsn: str | None = None,
        *,
        host: str | None = None,
        port: int | str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        min_size: int = 1,
        max_size: int = 10,
    ):
        self._dsn = dsn
        self._connect_kwargs = {
            "host": host,
            "port": int(port) if port is not None else None,
            "user": user,
            "password": password,
            "database": database,
        }
        self._connect_kwargs = {
            key: value for key, value in self._connect_kwargs.items() if value is not None
        }
        self._min_size = min_size
        self._max_size = max_size
        self._pool: Any | None = None
        self._lock = asyncio.Lock()
        self._initialized = False

    @classmethod
    def from_env(cls) -> "PostgresMetadataStore":
        import os

        dsn = os.getenv("LIGHTRAG_KB_POSTGRES_DSN") or os.getenv("POSTGRES_DSN")
        return cls(
            dsn=dsn,
            host=os.getenv("LIGHTRAG_KB_POSTGRES_HOST") or os.getenv("POSTGRES_HOST"),
            port=os.getenv("LIGHTRAG_KB_POSTGRES_PORT") or os.getenv("POSTGRES_PORT"),
            user=os.getenv("LIGHTRAG_KB_POSTGRES_USER") or os.getenv("POSTGRES_USER"),
            password=os.getenv("LIGHTRAG_KB_POSTGRES_PASSWORD")
            or os.getenv("POSTGRES_PASSWORD"),
            database=os.getenv("LIGHTRAG_KB_POSTGRES_DATABASE")
            or os.getenv("POSTGRES_DATABASE"),
            min_size=int(os.getenv("LIGHTRAG_KB_POSTGRES_POOL_MIN_SIZE", "1")),
            max_size=int(os.getenv("LIGHTRAG_KB_POSTGRES_POOL_MAX_SIZE", "10")),
        )

    async def initialize(self) -> None:
        async with self._lock:
            if self._pool is None:
                asyncpg = _load_asyncpg()
                if self._dsn:
                    self._pool = await asyncpg.create_pool(
                        dsn=self._dsn,
                        min_size=self._min_size,
                        max_size=self._max_size,
                    )
                else:
                    self._pool = await asyncpg.create_pool(
                        min_size=self._min_size,
                        max_size=self._max_size,
                        **self._connect_kwargs,
                    )
            async with self._pool_or_raise().acquire() as conn:
                await self._initialize_schema(conn)
            self._initialized = True

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._initialized = False

    async def create_documents_and_job(
        self, documents: Sequence[DocumentRecord], job: JobRecord
    ) -> tuple[list[DocumentRecord], JobRecord, bool]:
        await self._ensure_initialized()

        async def write(conn: Any) -> tuple[list[DocumentRecord], JobRecord, bool]:
            existing = await self._get_job_by_idempotency_key(
                conn, job.kb_id, job.idempotency_key, job_type=job.job_type
            )
            if existing is not None:
                self._validate_idempotent_job(existing, job)
                return await self._documents_for_job(conn, existing), existing, False
            for document in documents:
                await self._insert_document(conn, document)
            await self._insert_job(conn, job)
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
        clauses = ["kb_id = $1", "deleted_at IS NULL"]
        params: list[Any] = [kb_id]
        if status is not None:
            params.append(status)
            clauses.append(f"status = ${len(params)}")
        if source_name is not None:
            params.append(f"%{_escape_like(source_name)}%")
            # ESCAPE must be a single character. ``_escape_like`` escapes with a
            # single backslash, so the SQL escape string is one backslash too:
            # the Python literal ``"\\"`` is exactly one backslash at runtime.
            # (Using ``"\\\\"`` here sends a two-char string to Postgres, which
            # rejects it with InvalidEscapeSequenceError.)
            clauses.append(f"source_name ILIKE ${len(params)} ESCAPE '\\'")
        where = " AND ".join(clauses)
        async with self._pool_or_raise().acquire() as conn:
            total = await conn.fetchval(f"SELECT COUNT(*) FROM kb_documents WHERE {where}", *params)
            rows = await conn.fetch(
                f"""
                SELECT data_json FROM kb_documents
                WHERE {where}
                ORDER BY created_at DESC, id DESC
                LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
                """,
                *params,
                limit,
                offset,
            )
        return [_document_from_row(row) for row in rows], int(total or 0)

    async def get_document(self, kb_id: str, document_id: str) -> DocumentRecord:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            document = await self._get_document(conn, kb_id, document_id)
        return document

    async def get_documents_by_ids(
        self, kb_id: str, document_ids: Sequence[str]
    ) -> list[DocumentRecord]:
        await self._ensure_initialized()
        ordered_ids = list(dict.fromkeys(document_ids))
        if not ordered_ids:
            return []
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM kb_documents
                WHERE kb_id = $1 AND id = ANY($2::text[]) AND deleted_at IS NULL
                """,
                kb_id,
                ordered_ids,
            )
        by_id = {row["data_json"]["id"] if isinstance(row["data_json"], dict) else _loads_json_object(row["data_json"])["id"]: _document_from_row(row) for row in rows}
        return [by_id[document_id] for document_id in document_ids if document_id in by_id]

    async def get_documents_by_source_keys(
        self, kb_id: str, source_keys: Sequence[str]
    ) -> dict[str, DocumentRecord]:
        await self._ensure_initialized()
        ordered_keys = list(dict.fromkeys(source_keys))
        if not ordered_keys:
            return {}
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT source_key, data_json FROM kb_documents
                WHERE kb_id = $1 AND source_key = ANY($2::text[]) AND deleted_at IS NULL
                ORDER BY updated_at DESC, created_at DESC, id DESC
                """,
                kb_id,
                ordered_keys,
            )
        documents: dict[str, DocumentRecord] = {}
        wanted = set(ordered_keys)
        for row in rows:
            source_key = str(row["source_key"])
            if source_key in wanted and source_key not in documents:
                documents[source_key] = _document_from_row(row)
        return documents

    async def list_documents_by_batch_id(
        self, kb_id: str, batch_id: str
    ) -> list[DocumentRecord]:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM kb_documents
                WHERE kb_id = $1 AND batch_id = $2 AND deleted_at IS NULL
                ORDER BY created_at ASC, id ASC
                """,
                kb_id,
                batch_id,
            )
        return [_document_from_row(row) for row in rows]

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

        async def write(conn: Any) -> DocumentRecord:
            document = await self._get_document(conn, kb_id, document_id, for_update=True)
            metadata = dict(document.metadata)
            if metadata_patch:
                metadata.update(metadata_patch)
            document.metadata = metadata
            if enabled is not None:
                document.enabled = enabled
            if archived is not None:
                document.archived = archived
            document.updated_at = utc_now_iso()
            await self._save_document(conn, document)
            return document

        return await self._write(write)

    async def mark_document_parse_queued(
        self, kb_id: str, document_id: str, *, metadata_patch: dict[str, Any]
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
        self, kb_id: str, claims: Sequence[tuple[str, dict[str, Any]]]
    ) -> tuple[list[DocumentRecord], list[dict[str, Any]]]:
        await self._ensure_initialized()

        async def write(conn: Any) -> tuple[list[DocumentRecord], list[dict[str, Any]]]:
            documents: list[DocumentRecord] = []
            failures: list[dict[str, Any]] = []
            for document_id, metadata_patch in claims:
                try:
                    documents.append(
                        await self._claim_document_parse_queued(
                            conn,
                            kb_id,
                            document_id,
                            metadata_patch=metadata_patch,
                            raise_on_active=True,
                        )
                    )
                except ActiveDocumentParseJobError as exc:
                    failures.append(_active_failure(document_id, "parse_job_active", exc))
                except ActiveDocumentBuildJobError as exc:
                    failures.append(_active_failure(document_id, "build_job_active", exc))
                except ActiveDocumentDeleteJobError as exc:
                    failures.append(_active_failure(document_id, "delete_job_active", exc))
                except ActiveDocumentReplaceJobError as exc:
                    failures.append(_active_failure(document_id, "replace_job_active", exc))
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
        self, kb_id: str, document_id: str, *, metadata_patch: dict[str, Any]
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

        async def write(conn: Any) -> tuple[DocumentRecord, list[ArtifactRecord]]:
            document = await self._update_document_parse_state(
                conn,
                kb_id,
                document_id,
                status="parsed",
                metadata_patch=metadata_patch,
                parser_hash=parser_hash,
                lightrag_doc_id=lightrag_doc_id,
                clear_error=True,
            )
            await conn.execute(
                "DELETE FROM kb_document_artifacts WHERE kb_id = $1 AND document_id = $2",
                kb_id,
                document_id,
            )
            for artifact in artifacts:
                await self._insert_artifact(conn, artifact)
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

        async def write(conn: Any) -> tuple[list[DocumentRecord], list[dict[str, Any]]]:
            documents: list[DocumentRecord] = []
            failures: list[dict[str, Any]] = []
            for document_id, metadata_patch in claims:
                try:
                    documents.append(
                        await self._claim_document_build_queued(
                            conn,
                            kb_id,
                            document_id,
                            metadata_patch=metadata_patch,
                            require_parsed=require_parsed,
                        )
                    )
                except ActiveDocumentBuildJobError as exc:
                    failures.append(_active_failure(document_id, "build_job_active", exc))
                except ActiveDocumentDeleteJobError as exc:
                    failures.append(_active_failure(document_id, "delete_job_active", exc))
                except ActiveDocumentReplaceJobError as exc:
                    failures.append(_active_failure(document_id, "replace_job_active", exc))
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
        self, kb_id: str, document_id: str, *, metadata_patch: dict[str, Any]
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

        async def write(conn: Any) -> DocumentRecord:
            document = await self._get_document(conn, kb_id, document_id, for_update=True)
            document.metadata.update(metadata_patch)
            document.status = "ready"
            document.index_hash = index_hash
            if chunks_count is not None:
                document.chunks_count = chunks_count
            if entity_count is not None:
                document.entity_count = entity_count
            if relation_count is not None:
                document.relation_count = relation_count
            document.error_code = None
            document.error_message = None
            document.updated_at = utc_now_iso()
            await self._save_document(conn, document)
            return document

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

    async def claim_document_deleting(
        self, kb_id: str, document_id: str, *, metadata_patch: dict[str, Any]
    ) -> DocumentRecord:
        await self._ensure_initialized()
        return await self._write(
            lambda conn: self._claim_document_deleting(
                conn, kb_id, document_id, metadata_patch=metadata_patch
            )
        )

    async def claim_documents_deleting(
        self, kb_id: str, claims: Sequence[tuple[str, dict[str, Any]]]
    ) -> tuple[list[DocumentRecord], list[dict[str, Any]]]:
        await self._ensure_initialized()

        async def write(conn: Any) -> tuple[list[DocumentRecord], list[dict[str, Any]]]:
            documents: list[DocumentRecord] = []
            failures: list[dict[str, Any]] = []
            for document_id, metadata_patch in claims:
                try:
                    documents.append(
                        await self._claim_document_deleting(
                            conn, kb_id, document_id, metadata_patch=metadata_patch
                        )
                    )
                except ActiveDocumentParseJobError as exc:
                    failures.append(_active_failure(document_id, "parse_job_active", exc))
                except ActiveDocumentBuildJobError as exc:
                    failures.append(_active_failure(document_id, "build_job_active", exc))
                except ActiveDocumentDeleteJobError as exc:
                    failures.append(_active_failure(document_id, "delete_job_active", exc))
                except ActiveDocumentReplaceJobError as exc:
                    failures.append(_active_failure(document_id, "replace_job_active", exc))
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

    async def complete_document_delete(
        self, kb_id: str, document_id: str, *, metadata_patch: dict[str, Any]
    ) -> DocumentRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> DocumentRecord:
            document = await self._get_document(conn, kb_id, document_id, for_update=True)
            document.metadata.update(metadata_patch)
            now = utc_now_iso()
            await conn.execute(
                "DELETE FROM kb_document_artifacts WHERE kb_id = $1 AND document_id = $2",
                kb_id,
                document_id,
            )
            document.status = "deleted"
            document.enabled = False
            document.archived = True
            document.error_code = None
            document.error_message = None
            document.updated_at = now
            document.deleted_at = now
            await self._save_document(conn, document)
            return document

        return await self._write(write)

    async def fail_document_delete(
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
                status="delete_failed",
                metadata_patch=metadata_patch,
                error_code=error_code,
                error_message=error_message,
            )
        )

    async def claim_document_replacing(
        self, kb_id: str, document_id: str, *, metadata_patch: dict[str, Any]
    ) -> DocumentRecord:
        await self._ensure_initialized()
        return await self._write(
            lambda conn: self._claim_document_replacing(
                conn, kb_id, document_id, metadata_patch=metadata_patch
            )
        )

    async def complete_document_replace(
        self,
        kb_id: str,
        document_id: str,
        *,
        source_name: str,
        source_uri: str,
        source_type: str,
        source_hash: str,
        content_type: str | None,
        size_bytes: int,
        metadata_patch: dict[str, Any],
    ) -> DocumentRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> DocumentRecord:
            document = await self._get_document(conn, kb_id, document_id, for_update=True)
            metadata = dict(document.metadata)
            for key in _REPLACE_DERIVED_METADATA_KEYS:
                metadata.pop(key, None)
            metadata.update(metadata_patch)
            document.source_type = source_type
            document.source_name = source_name
            document.source_uri = source_uri
            document.source_hash = source_hash
            document.content_type = content_type
            document.size_bytes = size_bytes
            document.lightrag_doc_id = None
            document.parser_hash = None
            document.index_hash = None
            document.status = "uploaded"
            document.chunks_count = None
            document.entity_count = None
            document.relation_count = None
            document.error_code = None
            document.error_message = None
            document.metadata = metadata
            document.updated_at = utc_now_iso()
            await conn.execute(
                "DELETE FROM kb_document_artifacts WHERE kb_id = $1 AND document_id = $2",
                kb_id,
                document_id,
            )
            await self._save_document(conn, document)
            return document

        return await self._write(write)

    async def fail_document_replace(
        self,
        kb_id: str,
        document_id: str,
        *,
        error_code: str,
        error_message: str,
        metadata_patch: dict[str, Any],
        clear_index_metadata: bool = False,
    ) -> DocumentRecord:
        await self._ensure_initialized()
        return await self._write(
            lambda conn: self._update_document_parse_state(
                conn,
                kb_id,
                document_id,
                status="replace_failed",
                metadata_patch=metadata_patch,
                error_code=error_code,
                error_message=error_message,
                clear_lightrag_doc_id=clear_index_metadata,
                clear_index_state=clear_index_metadata,
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
        params: list[Any] = [kb_id, document_id]
        clauses = ["kb_id = $1", "document_id = $2"]
        if artifact_type is not None:
            params.append(artifact_type)
            clauses.append(f"artifact_type = ${len(params)}")
        where = " AND ".join(clauses)
        async with self._pool_or_raise().acquire() as conn:
            total = await conn.fetchval(
                f"SELECT COUNT(*) FROM kb_document_artifacts WHERE {where}", *params
            )
            rows = await conn.fetch(
                f"""
                SELECT data_json FROM kb_document_artifacts
                WHERE {where}
                ORDER BY created_at DESC, id DESC
                LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
                """,
                *params,
                limit,
                offset,
            )
        return [_artifact_from_row(row) for row in rows], int(total or 0)

    async def get_document_artifact(
        self, kb_id: str, document_id: str, artifact_id: str
    ) -> ArtifactRecord:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT data_json FROM kb_document_artifacts
                WHERE kb_id = $1 AND document_id = $2 AND id = $3
                """,
                kb_id,
                document_id,
                artifact_id,
            )
        if row is None:
            raise MetadataRecordNotFoundError(f"Artifact '{artifact_id}' not found")
        return _artifact_from_row(row)

    async def create_job(self, job: JobRecord) -> JobRecord:
        created_job, _created = await self.create_job_once(job)
        return created_job

    async def create_job_once(self, job: JobRecord) -> tuple[JobRecord, bool]:
        await self._ensure_initialized()

        async def write(conn: Any) -> tuple[JobRecord, bool]:
            existing = await self._get_job_by_idempotency_key(
                conn, job.kb_id, job.idempotency_key, job_type=job.job_type
            )
            if existing is not None:
                self._validate_idempotent_job(existing, job)
                return existing, False
            await self._insert_job(conn, job)
            return job, True

        return await self._write(write)

    async def get_job_by_idempotency_key(
        self, kb_id: str, idempotency_key: str, *, job_type: str | None = None
    ) -> JobRecord | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            return await self._get_job_by_idempotency_key(
                conn, kb_id, idempotency_key, job_type=job_type
            )

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
        params: list[Any] = [kb_id]
        clauses = ["kb_id = $1"]
        if document_id is not None:
            params.append(document_id)
            clauses.append(f"document_id = ${len(params)}")
        if statuses:
            params.append(list(statuses))
            clauses.append(f"status = ANY(${len(params)}::text[])")
        where = " AND ".join(clauses)
        async with self._pool_or_raise().acquire() as conn:
            total = await conn.fetchval(f"SELECT COUNT(*) FROM kb_jobs WHERE {where}", *params)
            rows = await conn.fetch(
                f"""
                SELECT data_json FROM kb_jobs
                WHERE {where}
                ORDER BY created_at DESC, id DESC
                LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
                """,
                *params,
                limit,
                offset,
            )
        return [_job_from_row(row) for row in rows], int(total or 0)

    async def list_dead_letter_jobs(
        self, kb_id: str, *, limit: int = 50, offset: int = 0
    ) -> tuple[list[JobRecord], int]:
        await self._ensure_initialized()
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        async with self._pool_or_raise().acquire() as conn:
            total = await conn.fetchval(
                """
                SELECT COUNT(*) FROM kb_jobs
                WHERE kb_id = $1 AND status = 'failed' AND retry_count >= max_retries
                """,
                kb_id,
            )
            rows = await conn.fetch(
                """
                SELECT data_json FROM kb_jobs
                WHERE kb_id = $1 AND status = 'failed' AND retry_count >= max_retries
                ORDER BY updated_at DESC, id DESC
                LIMIT $2 OFFSET $3
                """,
                kb_id,
                limit,
                offset,
            )
        return [_job_from_row(row) for row in rows], int(total or 0)

    async def get_job(self, kb_id: str, job_id: str) -> JobRecord:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            job = await self._get_job(conn, kb_id, job_id)
        return job

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

        async def write(conn: Any) -> JobRecord:
            current = await self._get_job(conn, kb_id, job_id, for_update=True)
            if status not in _allowed_next_job_statuses(current.status):
                raise InvalidJobTransitionError(
                    f"Cannot transition job '{job_id}' from {current.status} to {status}"
                )
            now = utc_now_iso()
            current.status = status
            if stage is not None:
                current.stage = stage
            if progress is not None:
                current.progress = progress
            if completed_items is not None:
                current.completed_items = completed_items
            if failed_items is not None:
                current.failed_items = failed_items
            if result is not None:
                current.result = result
            current.error_code = error_code
            current.error_message = error_message
            current.updated_at = now
            if status == "running" and current.started_at is None:
                current.started_at = now
            if status in {"succeeded", "failed"} and current.finished_at is None:
                current.finished_at = now
            if status == "cancelled" and current.cancelled_at is None:
                current.cancelled_at = now
            await self._save_job(conn, current)
            return current

        return await self._write(write)

    async def update_job_payload_patch(
        self,
        kb_id: str,
        job_id: str,
        *,
        payload_patch: dict[str, Any],
    ) -> JobRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> JobRecord:
            current = await self._get_job(conn, kb_id, job_id, for_update=True)
            current.payload = {**current.payload, **payload_patch}
            current.updated_at = utc_now_iso()
            await self._save_job(conn, current)
            return current

        return await self._write(write)

    async def reset_job_for_retry(
        self, kb_id: str, job_id: str, *, new_idempotency_key: str | None
    ) -> JobRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> JobRecord:
            current = await self._get_job(conn, kb_id, job_id, for_update=True)
            if current.status not in {"failed", "cancelled"}:
                raise InvalidJobTransitionError(
                    f"Cannot retry job '{job_id}' from {current.status}"
                )
            if current.retry_count >= current.max_retries:
                raise InvalidJobTransitionError(
                    f"Job '{job_id}' has reached max_retries={current.max_retries}"
                )
            now = utc_now_iso()
            current.status = "queued"
            current.progress = 0.0
            current.completed_items = 0
            current.failed_items = 0
            current.result = None
            current.error_code = None
            current.error_message = None
            current.retry_count += 1
            if new_idempotency_key is not None:
                current.idempotency_key = new_idempotency_key
            current.updated_at = now
            current.queued_at = now
            current.started_at = None
            current.finished_at = None
            current.cancelled_at = None
            await self._save_job(conn, current)
            return current

        return await self._write(write)

    async def create_config_version(
        self, record: ConfigVersionRecord
    ) -> ConfigVersionRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> ConfigVersionRecord:
            next_version = (
                await conn.fetchval(
                    "SELECT COALESCE(MAX(version), 0) + 1 FROM kb_config_versions WHERE kb_id = $1",
                    record.kb_id,
                )
                or 1
            )
            persisted = ConfigVersionRecord(
                id=record.id,
                kb_id=record.kb_id,
                workspace=record.workspace,
                version=int(next_version),
                config=record.config,
                parser_hash=record.parser_hash,
                index_hash=record.index_hash,
                query_hash=record.query_hash,
                created_at=record.created_at,
                activated_at=None,
                created_by=record.created_by,
            )
            await self._insert_config_version(conn, persisted)
            return persisted

        return await self._write(write)

    async def list_config_versions(
        self, kb_id: str, *, limit: int = 50, offset: int = 0
    ) -> tuple[list[ConfigVersionRecord], int]:
        await self._ensure_initialized()
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        async with self._pool_or_raise().acquire() as conn:
            total = await conn.fetchval(
                "SELECT COUNT(*) FROM kb_config_versions WHERE kb_id = $1", kb_id
            )
            rows = await conn.fetch(
                """
                SELECT data_json FROM kb_config_versions
                WHERE kb_id = $1
                ORDER BY version DESC
                LIMIT $2 OFFSET $3
                """,
                kb_id,
                limit,
                offset,
            )
        return [_config_from_row(row) for row in rows], int(total or 0)

    async def get_config_version(
        self, kb_id: str, version_id: str
    ) -> ConfigVersionRecord:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT data_json FROM kb_config_versions
                WHERE kb_id = $1 AND id = $2
                """,
                kb_id,
                version_id,
            )
        if row is None:
            raise MetadataRecordNotFoundError(
                f"Config version '{version_id}' not found"
            )
        return _config_from_row(row)

    async def mark_config_version_activated(
        self, kb_id: str, version_id: str
    ) -> ConfigVersionRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> ConfigVersionRecord:
            row = await conn.fetchrow(
                """
                SELECT data_json FROM kb_config_versions
                WHERE kb_id = $1 AND id = $2
                FOR UPDATE
                """,
                kb_id,
                version_id,
            )
            if row is None:
                raise MetadataRecordNotFoundError(
                    f"Config version '{version_id}' not found"
                )
            record = _config_from_row(row)
            record.activated_at = utc_now_iso()
            await self._save_config_version(conn, record)
            return record

        return await self._write(write)

    async def purge_kb_metadata(self, kb_id: str) -> dict[str, int]:
        await self._ensure_initialized()

        async def write(conn: Any) -> dict[str, int]:
            source_key_count = await conn.fetchval(
                """
                SELECT COUNT(*) FROM kb_documents
                WHERE kb_id = $1 AND source_key IS NOT NULL
                """,
                kb_id,
            )
            counts = {"document_source_keys": int(source_key_count or 0)}
            for table, label in (
                ("kb_document_artifacts", "document_artifacts"),
                ("kb_config_versions", "kb_config_versions"),
                ("kb_jobs", "jobs"),
                ("kb_documents", "documents"),
            ):
                status = await conn.execute(f"DELETE FROM {table} WHERE kb_id = $1", kb_id)
                counts[label] = _rowcount(status)
            return counts

        return await self._write(write)

    async def recover_orphan_jobs(
        self,
        *,
        error_code: str = "worker_orphaned",
        error_message: str = "Job worker crashed before completion; please retry",
        resumable_job_types: set[str] | None = None,
    ) -> list[JobRecord]:
        await self._ensure_initialized()
        resumable = set(resumable_job_types or set())

        async def write(conn: Any) -> list[JobRecord]:
            rows = await conn.fetch(
                """
                SELECT data_json FROM kb_jobs
                WHERE status = ANY($1::text[])
                ORDER BY created_at ASC, id ASC
                FOR UPDATE
                """,
                ["queued", "running", "cancelling", "retrying"],
            )
            now = utc_now_iso()
            updated: list[JobRecord] = []
            for row in rows:
                job = _job_from_row(row)
                if (
                    job.status == "queued"
                    and job.job_type in resumable
                    and (
                        job.document_id is not None
                        or job.job_type in _AGGREGATE_RESUMABLE_JOB_TYPES
                    )
                ):
                    continue
                job.status = "failed"
                job.error_code = error_code
                job.error_message = error_message
                job.updated_at = now
                if job.finished_at is None:
                    job.finished_at = now
                await self._save_job(conn, job)
                updated.append(job)
            for source_statuses, target_status in (
                (("parse_queued", "parsing"), "parse_failed"),
                (("build_queued", "building"), "build_failed"),
                (("deleting",), "delete_failed"),
                (("replacing",), "replace_failed"),
            ):
                doc_rows = await conn.fetch(
                    """
                    SELECT data_json FROM kb_documents
                    WHERE status = ANY($1::text[]) AND deleted_at IS NULL
                    FOR UPDATE
                    """,
                    list(source_statuses),
                )
                for doc_row in doc_rows:
                    document = _document_from_row(doc_row)
                    document.status = target_status
                    document.error_code = error_code
                    document.error_message = error_message
                    document.updated_at = now
                    await self._save_document(conn, document)
            return updated

        return await self._write(write)

    async def claim_next_worker_job(
        self, *, job_types: Sequence[str], max_queued_at: str | None = None
    ) -> JobRecord | None:
        await self._ensure_initialized()
        if not job_types:
            return None

        async def write(conn: Any) -> JobRecord | None:
            params: list[Any] = [list(job_types), sorted(_AGGREGATE_RESUMABLE_JOB_TYPES)]
            max_filter = ""
            if max_queued_at is not None:
                params.append(max_queued_at)
                max_filter = f" AND queued_at <= ${len(params)}"
            row = await conn.fetchrow(
                f"""
                SELECT data_json FROM kb_jobs
                WHERE status = 'queued'
                  AND job_type = ANY($1::text[])
                  AND (document_id IS NOT NULL OR job_type = ANY($2::text[]))
                  {max_filter}
                ORDER BY queued_at ASC, id ASC
                LIMIT 1
                FOR UPDATE SKIP LOCKED
                """,
                *params,
            )
            if row is None:
                return None
            job = _job_from_row(row)
            now = utc_now_iso()
            job.status = "running"
            job.progress = max(float(job.progress or 0.0), 0.1)
            job.updated_at = now
            if job.started_at is None:
                job.started_at = now
            await self._save_job(conn, job)
            return job

        return await self._write(write)

    def _pool_or_raise(self) -> Any:
        if self._pool is None:
            raise RuntimeError("PostgreSQL metadata store is not initialized")
        return self._pool

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def _write(self, callback: Callable[[Any], Awaitable[_T]]) -> _T:
        async with self._lock:
            async with self._pool_or_raise().acquire() as conn:
                async with conn.transaction():
                    return await callback(conn)

    async def _initialize_schema(self, conn: Any) -> None:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kb_metadata_schema (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS kb_documents (
                id TEXT PRIMARY KEY,
                kb_id TEXT NOT NULL,
                workspace TEXT NOT NULL,
                status TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                source_key TEXT,
                batch_id TEXT,
                deleted_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                data_json JSONB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_kb_documents_kb_status
                ON kb_documents (kb_id, status);
            CREATE INDEX IF NOT EXISTS idx_kb_documents_kb_source_hash
                ON kb_documents (kb_id, source_hash);
            CREATE INDEX IF NOT EXISTS idx_kb_documents_workspace
                ON kb_documents (workspace);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_kb_documents_source_key
                ON kb_documents (kb_id, source_key)
                WHERE source_key IS NOT NULL AND deleted_at IS NULL;

            CREATE TABLE IF NOT EXISTS kb_jobs (
                id TEXT PRIMARY KEY,
                kb_id TEXT NOT NULL,
                workspace TEXT NOT NULL,
                batch_id TEXT,
                document_id TEXT,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL,
                idempotency_key TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0,
                max_retries INTEGER NOT NULL DEFAULT 3,
                queued_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                data_json JSONB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_kb_jobs_kb_status
                ON kb_jobs (kb_id, status);
            CREATE INDEX IF NOT EXISTS idx_kb_jobs_kb_document
                ON kb_jobs (kb_id, document_id);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_kb_jobs_kb_type_idempotency
                ON kb_jobs (kb_id, job_type, idempotency_key)
                WHERE idempotency_key IS NOT NULL;

            CREATE TABLE IF NOT EXISTS kb_document_artifacts (
                id TEXT PRIMARY KEY,
                kb_id TEXT NOT NULL,
                workspace TEXT NOT NULL,
                document_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                data_json JSONB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_kb_artifacts_kb_document
                ON kb_document_artifacts (kb_id, document_id);
            CREATE INDEX IF NOT EXISTS idx_kb_artifacts_workspace_type
                ON kb_document_artifacts (workspace, artifact_type);

            CREATE TABLE IF NOT EXISTS kb_config_versions (
                id TEXT PRIMARY KEY,
                kb_id TEXT NOT NULL,
                workspace TEXT NOT NULL,
                version INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                data_json JSONB NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_kb_config_versions_kb_version_unique
                ON kb_config_versions (kb_id, version);
            CREATE INDEX IF NOT EXISTS idx_kb_config_versions_workspace
                ON kb_config_versions (workspace);
            """
        )
        await conn.execute(
            """
            INSERT INTO kb_metadata_schema(version, applied_at)
            VALUES (1, $1)
            ON CONFLICT (version) DO NOTHING
            """,
            utc_now_iso(),
        )

    async def _get_document(
        self, conn: Any, kb_id: str, document_id: str, *, for_update: bool = False
    ) -> DocumentRecord:
        suffix = " FOR UPDATE" if for_update else ""
        row = await conn.fetchrow(
            f"""
            SELECT data_json FROM kb_documents
            WHERE kb_id = $1 AND id = $2 AND deleted_at IS NULL
            {suffix}
            """,
            kb_id,
            document_id,
        )
        if row is None:
            raise MetadataRecordNotFoundError(f"Document '{document_id}' not found")
        return _document_from_row(row)

    async def _insert_document(self, conn: Any, document: DocumentRecord) -> None:
        await self._check_source_key_available(conn, document)
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
            _metadata_source_key(document.metadata),
            _batch_id(document.metadata),
            document.deleted_at,
            document.created_at,
            document.updated_at,
            _record_json(document),
        )

    async def _save_document(self, conn: Any, document: DocumentRecord) -> None:
        await self._check_source_key_available(conn, document)
        status = await conn.execute(
            """
            UPDATE kb_documents
            SET status = $1, source_name = $2, source_hash = $3, source_key = $4,
                batch_id = $5, deleted_at = $6, updated_at = $7, data_json = $8::jsonb
            WHERE kb_id = $9 AND id = $10
            """,
            document.status,
            document.source_name,
            document.source_hash,
            _metadata_source_key(document.metadata) if document.deleted_at is None else None,
            _batch_id(document.metadata),
            document.deleted_at,
            document.updated_at,
            _record_json(document),
            document.kb_id,
            document.id,
        )
        if _rowcount(status) == 0:
            raise MetadataRecordNotFoundError(f"Document '{document.id}' not found")

    async def _check_source_key_available(self, conn: Any, document: DocumentRecord) -> None:
        source_key = _metadata_source_key(document.metadata)
        if source_key is None or document.deleted_at is not None:
            return
        existing_id = await conn.fetchval(
            """
            SELECT id FROM kb_documents
            WHERE kb_id = $1 AND source_key = $2 AND deleted_at IS NULL AND id <> $3
            LIMIT 1
            """,
            document.kb_id,
            source_key,
            document.id,
        )
        if existing_id is not None:
            raise DuplicateDocumentSourceKeyError(document.kb_id, source_key, str(existing_id))

    async def _insert_artifact(self, conn: Any, artifact: ArtifactRecord) -> None:
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

    async def _get_job(
        self, conn: Any, kb_id: str, job_id: str, *, for_update: bool = False
    ) -> JobRecord:
        suffix = " FOR UPDATE" if for_update else ""
        row = await conn.fetchrow(
            f"SELECT data_json FROM kb_jobs WHERE kb_id = $1 AND id = $2{suffix}",
            kb_id,
            job_id,
        )
        if row is None:
            raise MetadataRecordNotFoundError(f"Job '{job_id}' not found")
        return _job_from_row(row)

    async def _insert_job(self, conn: Any, job: JobRecord) -> None:
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

    async def _save_job(self, conn: Any, job: JobRecord) -> None:
        status = await conn.execute(
            """
            UPDATE kb_jobs
            SET status = $1, idempotency_key = $2, retry_count = $3,
                max_retries = $4, queued_at = $5, updated_at = $6, data_json = $7::jsonb
            WHERE kb_id = $8 AND id = $9
            """,
            job.status,
            job.idempotency_key,
            job.retry_count,
            job.max_retries,
            job.queued_at,
            job.updated_at,
            _record_json(job),
            job.kb_id,
            job.id,
        )
        if _rowcount(status) == 0:
            raise MetadataRecordNotFoundError(f"Job '{job.id}' not found")

    async def _get_job_by_idempotency_key(
        self,
        conn: Any,
        kb_id: str,
        idempotency_key: str | None,
        *,
        job_type: str | None = None,
    ) -> JobRecord | None:
        if not idempotency_key:
            return None
        params: list[Any] = [kb_id, idempotency_key]
        clause = "kb_id = $1 AND idempotency_key = $2"
        if job_type is not None:
            params.append(job_type)
            clause += f" AND job_type = ${len(params)}"
        row = await conn.fetchrow(
            f"""
            SELECT data_json FROM kb_jobs
            WHERE {clause}
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            *params,
        )
        return _job_from_row(row) if row is not None else None

    def _validate_idempotent_job(self, existing: JobRecord, candidate: JobRecord) -> None:
        if existing.payload.get("idempotency_fingerprint") != candidate.payload.get(
            "idempotency_fingerprint"
        ):
            raise IdempotencyKeyConflictError(candidate.idempotency_key or "")

    async def _documents_for_job(self, conn: Any, job: JobRecord) -> list[DocumentRecord]:
        document_ids = job.payload.get("document_ids")
        if isinstance(document_ids, list) and all(
            isinstance(document_id, str) for document_id in document_ids
        ):
            if not document_ids:
                return []
            rows = await conn.fetch(
                """
                SELECT data_json FROM kb_documents
                WHERE kb_id = $1 AND id = ANY($2::text[]) AND deleted_at IS NULL
                """,
                job.kb_id,
                document_ids,
            )
            by_id = {_loads_json_object(row["data_json"])["id"]: _document_from_row(row) for row in rows}
            return [by_id[document_id] for document_id in document_ids if document_id in by_id]
        if not job.batch_id:
            return []
        rows = await conn.fetch(
            """
            SELECT data_json FROM kb_documents
            WHERE kb_id = $1 AND batch_id = $2 AND deleted_at IS NULL
            ORDER BY created_at ASC, id ASC
            """,
            job.kb_id,
            job.batch_id,
        )
        return [_document_from_row(row) for row in rows]

    async def _claim_document_parse_queued(
        self,
        conn: Any,
        kb_id: str,
        document_id: str,
        *,
        metadata_patch: dict[str, Any],
        raise_on_active: bool,
    ) -> DocumentRecord:
        document = await self._get_document(conn, kb_id, document_id, for_update=True)
        if raise_on_active and document.status in {"parse_queued", "parsing"}:
            raise ActiveDocumentParseJobError(document_id, _active_parse_job_id(document))
        if document.status in {"build_queued", "building"}:
            raise ActiveDocumentBuildJobError(document_id, _active_build_job_id(document))
        if document.status == "deleting":
            raise ActiveDocumentDeleteJobError(document_id, _active_delete_job_id(document))
        if document.status == "replacing":
            raise ActiveDocumentReplaceJobError(document_id, _active_replace_job_id(document))
        return await self._update_document_parse_state(
            conn,
            kb_id,
            document_id,
            status="parse_queued",
            metadata_patch=metadata_patch,
            clear_error=True,
        )

    async def _claim_document_build_queued(
        self,
        conn: Any,
        kb_id: str,
        document_id: str,
        *,
        metadata_patch: dict[str, Any],
        require_parsed: bool,
    ) -> DocumentRecord:
        document = await self._get_document(conn, kb_id, document_id, for_update=True)
        if document.status in {"build_queued", "building"}:
            raise ActiveDocumentBuildJobError(document_id, _active_build_job_id(document))
        if document.status == "deleting":
            raise ActiveDocumentDeleteJobError(document_id, _active_delete_job_id(document))
        if document.status == "replacing":
            raise ActiveDocumentReplaceJobError(document_id, _active_replace_job_id(document))
        if require_parsed and document.status not in {"parsed", "ready", "build_failed"}:
            raise DocumentNotParsedError(document_id, str(document.status))
        return await self._update_document_parse_state(
            conn,
            kb_id,
            document_id,
            status="build_queued",
            metadata_patch=metadata_patch,
            clear_error=True,
        )

    async def _claim_document_deleting(
        self, conn: Any, kb_id: str, document_id: str, *, metadata_patch: dict[str, Any]
    ) -> DocumentRecord:
        document = await self._get_document(conn, kb_id, document_id, for_update=True)
        if document.status in {"parse_queued", "parsing"}:
            raise ActiveDocumentParseJobError(document_id, _active_parse_job_id(document))
        if document.status in {"build_queued", "building"}:
            raise ActiveDocumentBuildJobError(document_id, _active_build_job_id(document))
        if document.status == "deleting":
            existing_job_id = _active_delete_job_id(document)
            requested_job_id = metadata_patch.get("pending_delete_job_id") or metadata_patch.get(
                "current_delete_job_id"
            )
            if requested_job_id is not None and str(requested_job_id) == existing_job_id:
                return await self._update_document_parse_state(
                    conn,
                    kb_id,
                    document_id,
                    status="deleting",
                    metadata_patch=metadata_patch,
                    clear_error=True,
                )
            raise ActiveDocumentDeleteJobError(document_id, existing_job_id)
        if document.status == "replacing":
            raise ActiveDocumentReplaceJobError(document_id, _active_replace_job_id(document))
        return await self._update_document_parse_state(
            conn,
            kb_id,
            document_id,
            status="deleting",
            metadata_patch=metadata_patch,
            clear_error=True,
        )

    async def _claim_document_replacing(
        self, conn: Any, kb_id: str, document_id: str, *, metadata_patch: dict[str, Any]
    ) -> DocumentRecord:
        document = await self._get_document(conn, kb_id, document_id, for_update=True)
        if document.status in {"parse_queued", "parsing"}:
            raise ActiveDocumentParseJobError(document_id, _active_parse_job_id(document))
        if document.status in {"build_queued", "building"}:
            raise ActiveDocumentBuildJobError(document_id, _active_build_job_id(document))
        if document.status == "deleting":
            raise ActiveDocumentDeleteJobError(document_id, _active_delete_job_id(document))
        if document.status == "replacing":
            raise ActiveDocumentReplaceJobError(document_id, _active_replace_job_id(document))
        return await self._update_document_parse_state(
            conn,
            kb_id,
            document_id,
            status="replacing",
            metadata_patch=metadata_patch,
            clear_error=True,
        )

    async def _update_document_parse_state(
        self,
        conn: Any,
        kb_id: str,
        document_id: str,
        *,
        status: str,
        metadata_patch: dict[str, Any],
        parser_hash: str | None = None,
        lightrag_doc_id: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        clear_error: bool = False,
        clear_lightrag_doc_id: bool = False,
        clear_index_state: bool = False,
    ) -> DocumentRecord:
        document = await self._get_document(conn, kb_id, document_id, for_update=True)
        document.metadata.update(metadata_patch)
        document.status = status
        if parser_hash is not None:
            document.parser_hash = parser_hash
        if clear_lightrag_doc_id:
            document.lightrag_doc_id = None
        elif lightrag_doc_id is not None:
            document.lightrag_doc_id = lightrag_doc_id
        if clear_index_state:
            document.index_hash = None
            document.chunks_count = None
            document.entity_count = None
            document.relation_count = None
        if clear_error:
            document.error_code = None
            document.error_message = None
        else:
            document.error_code = error_code
            document.error_message = error_message
        document.updated_at = utc_now_iso()
        await self._save_document(conn, document)
        return document

    async def _insert_config_version(self, conn: Any, record: ConfigVersionRecord) -> None:
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

    async def _save_config_version(self, conn: Any, record: ConfigVersionRecord) -> None:
        await conn.execute(
            """
            UPDATE kb_config_versions
            SET data_json = $1::jsonb
            WHERE kb_id = $2 AND id = $3
            """,
            _record_json(record),
            record.kb_id,
            record.id,
        )


def _batch_id(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("batch_id")
    return value if isinstance(value, str) and value else None


def _rowcount(status: str) -> int:
    try:
        return int(status.rsplit(" ", 1)[-1])
    except (ValueError, IndexError):
        return 0


def _active_failure(document_id: str, error_code: str, exc: Exception) -> dict[str, Any]:
    detail = {
        "document_id": document_id,
        "status": "failed",
        "error_code": error_code,
        "error_message": str(exc),
    }
    existing_job_id = getattr(exc, "existing_job_id", None)
    if existing_job_id is not None:
        detail["existing_job_id"] = existing_job_id
    return detail
