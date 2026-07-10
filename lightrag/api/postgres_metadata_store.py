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
    AuditEventRecord,
    ChatMessageRecord,
    ChatProjectRecord,
    ChatSessionRecord,
    ConfigVersionRecord,
    DocumentNotParsedError,
    DocumentRecord,
    DuplicateDocumentSourceKeyError,
    EnterpriseUserRecord,
    EnterpriseUserKBQuerySettingsRecord,
    EnterpriseAPIKeyRecord,
    EnterpriseInvitationRecord,
    IdempotencyKeyConflictError,
    InvalidJobTransitionError,
    JobRecord,
    KBACLRecord,
    EnterpriseTenantKBACLRecord,
    EnterpriseTenantMembershipRecord,
    EnterpriseTenantRecord,
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


def _enterprise_user_from_row(row: Any) -> EnterpriseUserRecord:
    data = _loads_json_object(row["data_json"])
    # Legacy JSONB rows predate can_delete_documents; default it so the
    # dataclass deserializes without raising on the missing key.
    data.setdefault("can_delete_documents", False)
    data.setdefault("can_use_agent_query", False)
    return EnterpriseUserRecord(**data)


def _enterprise_user_kb_query_settings_from_row(
    row: Any,
) -> EnterpriseUserKBQuerySettingsRecord:
    data = _loads_json_object(row["data_json"])
    return EnterpriseUserKBQuerySettingsRecord(**data)


def _chat_project_from_row(row: Any) -> ChatProjectRecord:
    data = _loads_json_object(row["data_json"])
    return ChatProjectRecord(**data)


def _chat_session_from_row(row: Any) -> ChatSessionRecord:
    data = _loads_json_object(row["data_json"])
    # Legacy JSONB rows predate context_rounds; default it so the dataclass
    # deserializes without raising on the missing key.
    data.setdefault("context_rounds", 1)
    return ChatSessionRecord(**data)


def _chat_message_from_row(row: Any) -> ChatMessageRecord:
    data = _loads_json_object(row["data_json"])
    return ChatMessageRecord(**data)


def _enterprise_api_key_from_row(row: Any) -> EnterpriseAPIKeyRecord:
    data = _loads_json_object(row["data_json"])
    return EnterpriseAPIKeyRecord(**data)


def _enterprise_invitation_from_row(row: Any) -> EnterpriseInvitationRecord:
    data = _loads_json_object(row["data_json"])
    return EnterpriseInvitationRecord(**data)


def _kb_acl_from_row(row: Any) -> KBACLRecord:
    data = _loads_json_object(row["data_json"])
    return KBACLRecord(**data)


def _tenant_membership_from_row(row: Any) -> EnterpriseTenantMembershipRecord:
    data = _loads_json_object(row["data_json"])
    return EnterpriseTenantMembershipRecord(**data)


def _enterprise_tenant_from_row(row: Any) -> EnterpriseTenantRecord:
    return EnterpriseTenantRecord(**_loads_json_object(row["data_json"]))


def _tenant_kb_acl_from_row(row: Any) -> EnterpriseTenantKBACLRecord:
    data = _loads_json_object(row["data_json"])
    return EnterpriseTenantKBACLRecord(**data)


def _audit_event_from_row(row: Any) -> AuditEventRecord:
    data = _loads_json_object(row["data_json"])
    return AuditEventRecord(**data)


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

    async def aggregate_control_plane_stats(
        self, kb_id: str | None = None
    ) -> dict[str, Any]:
        """Control-plane aggregates for the stats endpoints.

        Mirrors ``SQLiteMetadataStore.aggregate_control_plane_stats``; document
        counters are not projected columns in PostgreSQL so they are summed
        from ``data_json``.
        """
        await self._ensure_initialized()
        where = "" if kb_id is None else " WHERE kb_id = $1"
        params: list[Any] = [] if kb_id is None else [kb_id]
        async with self._pool_or_raise().acquire() as conn:
            documents_by_status = {
                str(row["status"]): int(row["count"])
                for row in await conn.fetch(
                    "SELECT status, COUNT(*) AS count FROM kb_documents"
                    f"{where} GROUP BY status",
                    *params,
                )
            }
            counter_row = await conn.fetchrow(
                "SELECT "
                "COALESCE(SUM((data_json->>'chunks_count')::bigint), 0) AS chunks, "
                "COALESCE(SUM((data_json->>'entity_count')::bigint), 0) AS entities, "
                "COALESCE(SUM((data_json->>'relation_count')::bigint), 0) AS relations "
                f"FROM kb_documents{where}",
                *params,
            )
            jobs_by_status = {
                str(row["status"]): int(row["count"])
                for row in await conn.fetch(
                    f"SELECT status, COUNT(*) AS count FROM kb_jobs{where} GROUP BY status",
                    *params,
                )
            }
            dead_letter_where = (
                " WHERE kb_id = $1 AND " if kb_id is not None else " WHERE "
            )
            dead_letter = await conn.fetchval(
                f"SELECT COUNT(*) FROM kb_jobs{dead_letter_where}"
                "status = 'failed' AND retry_count >= max_retries",
                *params,
            )
            artifacts = await conn.fetchval(
                f"SELECT COUNT(*) FROM kb_document_artifacts{where}", *params
            )
        return {
            "documents_by_status": documents_by_status,
            "document_counters": {
                "chunks": int(counter_row["chunks"]),
                "entities": int(counter_row["entities"]),
                "relations": int(counter_row["relations"]),
            },
            "jobs_by_status": jobs_by_status,
            "dead_letter_jobs": int(dead_letter or 0),
            "artifacts": int(artifacts or 0),
        }

    async def aggregate_enterprise_stats(self) -> dict[str, Any]:
        """Platform-wide enterprise aggregates for ``GET /admin/overview``."""
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            users_by_status = {
                str(row["status"]): int(row["count"])
                for row in await conn.fetch(
                    "SELECT status, COUNT(*) AS count FROM enterprise_users GROUP BY status"
                )
            }
            tenants = await conn.fetchval("SELECT COUNT(*) FROM enterprise_tenants")
            api_keys_by_status = {
                str(row["status"]): int(row["count"])
                for row in await conn.fetch(
                    "SELECT status, COUNT(*) AS count FROM enterprise_api_keys GROUP BY status"
                )
            }
            audit_events = await conn.fetchval(
                "SELECT COUNT(*) FROM enterprise_audit_events"
            )
        return {
            "users_by_status": users_by_status,
            "tenants": int(tenants or 0),
            "api_keys_by_status": api_keys_by_status,
            "audit_events": int(audit_events or 0),
        }

    async def count_active_jobs_for_principal(self, subject_id: str) -> int:
        """Count in-flight jobs (queued/running/retrying/cancelling) across all
        KBs attributed to a principal via ``payload._principal.subject_id``."""
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            value = await conn.fetchval(
                """
                SELECT COUNT(*) FROM kb_jobs
                WHERE status = ANY($1::text[])
                  AND data_json->'payload'->'_principal'->>'subject_id' = $2
                """,
                ["queued", "running", "retrying", "cancelling"],
                subject_id,
            )
        return int(value or 0)

    async def count_active_jobs_for_tenant(self, tenant_id: str) -> int:
        """Count in-flight jobs across all KBs attributed to a tenant via
        ``payload._principal.tenant_id``."""
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            value = await conn.fetchval(
                """
                SELECT COUNT(*) FROM kb_jobs
                WHERE status = ANY($1::text[])
                  AND data_json->'payload'->'_principal'->>'tenant_id' = $2
                """,
                ["queued", "running", "retrying", "cancelling"],
                tenant_id,
            )
        return int(value or 0)

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

    async def update_job_progress(
        self,
        kb_id: str,
        job_id: str,
        *,
        progress: float | None = None,
        completed_items: int | None = None,
        stage: str | None = None,
        result_patch: dict[str, Any] | None = None,
    ) -> JobRecord:
        """Patch live progress on a running job WITHOUT changing its status.

        Unlike :meth:`transition_job` this never touches ``status`` (so it
        bypasses the status state-machine and can be called repeatedly while a
        job runs) and never touches ``error_*`` / timestamps. ``result_patch``
        is shallow-merged into the existing ``result``. Patches are silently
        ignored once the job has left an active state, so a late progress poll
        cannot resurrect / overwrite a finished job's terminal snapshot.
        """
        await self._ensure_initialized()

        async def write(conn: Any) -> JobRecord:
            current = await self._get_job(conn, kb_id, job_id, for_update=True)
            if current.status not in {"running", "retrying", "cancelling"}:
                return current
            if progress is not None:
                current.progress = progress
            if completed_items is not None:
                current.completed_items = completed_items
            if stage is not None:
                current.stage = stage
            if result_patch is not None:
                current.result = {**(current.result or {}), **result_patch}
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

    async def get_enterprise_user_by_username(
        self, username: str
    ) -> EnterpriseUserRecord | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_users WHERE username = $1",
                username,
            )
        return _enterprise_user_from_row(row) if row is not None else None

    async def get_enterprise_user_by_id(
        self, user_id: str
    ) -> EnterpriseUserRecord | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_users WHERE id = $1",
                user_id,
            )
        return _enterprise_user_from_row(row) if row is not None else None

    async def list_enterprise_users(self) -> list[EnterpriseUserRecord]:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM enterprise_users
                ORDER BY created_at ASC, id ASC
                """
            )
        return [_enterprise_user_from_row(row) for row in rows]

    async def upsert_enterprise_user(
        self, user: EnterpriseUserRecord
    ) -> EnterpriseUserRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> EnterpriseUserRecord:
            await conn.execute(
                """
                INSERT INTO enterprise_users (
                    id, username, status, tenant_id, created_at, updated_at, data_json
                ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    username = excluded.username,
                    status = excluded.status,
                    tenant_id = excluded.tenant_id,
                    updated_at = excluded.updated_at,
                    data_json = excluded.data_json
                """,
                user.id,
                user.username,
                user.status,
                user.tenant_id,
                user.created_at,
                user.updated_at,
                _record_json(user),
            )
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_users WHERE id = $1", user.id
            )
            if row is None:
                raise MetadataRecordNotFoundError(f"User '{user.id}' not found")
            return _enterprise_user_from_row(row)

        return await self._write(write)

    async def delete_enterprise_user(self, user_id: str) -> bool:
        """Delete a user and cascade-remove related tenant memberships, KB ACLs,
        per-user query settings and chat projects/sessions.  Returns ``True`` if
        the user existed."""
        await self._ensure_initialized()

        async def write(conn: Any) -> bool:
            # Cascade: remove related records first.
            await conn.execute(
                "DELETE FROM enterprise_tenant_memberships WHERE user_id = $1",
                user_id,
            )
            await conn.execute(
                "DELETE FROM enterprise_kb_acl WHERE user_id = $1",
                user_id,
            )
            await conn.execute(
                "DELETE FROM enterprise_user_kb_query_settings WHERE user_id = $1",
                user_id,
            )
            await conn.execute(
                "DELETE FROM enterprise_chat_messages WHERE user_id = $1",
                user_id,
            )
            await conn.execute(
                "DELETE FROM enterprise_chat_sessions WHERE user_id = $1",
                user_id,
            )
            await conn.execute(
                "DELETE FROM enterprise_chat_projects WHERE user_id = $1",
                user_id,
            )
            status = await conn.execute(
                "DELETE FROM enterprise_users WHERE id = $1",
                user_id,
            )
            return _rowcount(status) > 0

        return await self._write(write)

    async def get_enterprise_user_kb_query_settings(
        self, user_id: str, kb_id: str
    ) -> EnterpriseUserKBQuerySettingsRecord | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT data_json FROM enterprise_user_kb_query_settings
                WHERE user_id = $1 AND kb_id = $2
                """,
                user_id,
                kb_id,
            )
        return (
            _enterprise_user_kb_query_settings_from_row(row)
            if row is not None
            else None
        )

    async def upsert_enterprise_user_kb_query_settings(
        self, record: EnterpriseUserKBQuerySettingsRecord
    ) -> EnterpriseUserKBQuerySettingsRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> EnterpriseUserKBQuerySettingsRecord:
            await conn.execute(
                """
                INSERT INTO enterprise_user_kb_query_settings (
                    user_id, kb_id, user_prompt, created_at, updated_at, data_json
                ) VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                ON CONFLICT (user_id, kb_id) DO UPDATE SET
                    user_prompt = excluded.user_prompt,
                    updated_at = excluded.updated_at,
                    data_json = jsonb_set(
                        excluded.data_json,
                        '{created_at}',
                        to_jsonb(enterprise_user_kb_query_settings.created_at)
                    )
                """,
                record.user_id,
                record.kb_id,
                record.user_prompt,
                record.created_at,
                record.updated_at,
                _record_json(record),
            )
            row = await conn.fetchrow(
                """
                SELECT data_json FROM enterprise_user_kb_query_settings
                WHERE user_id = $1 AND kb_id = $2
                """,
                record.user_id,
                record.kb_id,
            )
            if row is None:
                raise MetadataRecordNotFoundError("User KB query settings not found")
            return _enterprise_user_kb_query_settings_from_row(row)

        return await self._write(write)

    async def delete_enterprise_user_kb_query_settings(
        self, user_id: str, kb_id: str
    ) -> bool:
        await self._ensure_initialized()

        async def write(conn: Any) -> bool:
            status = await conn.execute(
                """
                DELETE FROM enterprise_user_kb_query_settings
                WHERE user_id = $1 AND kb_id = $2
                """,
                user_id,
                kb_id,
            )
            return _rowcount(status) > 0

        return await self._write(write)

    async def create_chat_project(self, record: ChatProjectRecord) -> ChatProjectRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> ChatProjectRecord:
            await conn.execute(
                """
                INSERT INTO enterprise_chat_projects (
                    id, user_id, name, created_at, updated_at, data_json
                ) VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                """,
                record.id,
                record.user_id,
                record.name,
                record.created_at,
                record.updated_at,
                _record_json(record),
            )
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_chat_projects WHERE id = $1",
                record.id,
            )
            if row is None:
                raise MetadataRecordNotFoundError("Chat project not found")
            return _chat_project_from_row(row)

        return await self._write(write)

    async def get_chat_project(
        self, user_id: str, project_id: str
    ) -> ChatProjectRecord | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT data_json FROM enterprise_chat_projects
                WHERE id = $1 AND user_id = $2
                """,
                project_id,
                user_id,
            )
        return _chat_project_from_row(row) if row is not None else None

    async def list_chat_projects(
        self, user_id: str, *, limit: int = 50, offset: int = 0
    ) -> tuple[list[ChatProjectRecord], int]:
        await self._ensure_initialized()
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        async with self._pool_or_raise().acquire() as conn:
            total = await conn.fetchval(
                "SELECT COUNT(*) FROM enterprise_chat_projects WHERE user_id = $1",
                user_id,
            )
            rows = await conn.fetch(
                """
                SELECT data_json FROM enterprise_chat_projects
                WHERE user_id = $1
                ORDER BY updated_at DESC, id DESC
                LIMIT $2 OFFSET $3
                """,
                user_id,
                limit,
                offset,
            )
        return [_chat_project_from_row(row) for row in rows], int(total)

    async def rename_chat_project(
        self, user_id: str, project_id: str, *, name: str
    ) -> ChatProjectRecord | None:
        await self._ensure_initialized()

        async def write(conn: Any) -> ChatProjectRecord | None:
            row = await conn.fetchrow(
                """
                SELECT data_json FROM enterprise_chat_projects
                WHERE id = $1 AND user_id = $2
                FOR UPDATE
                """,
                project_id,
                user_id,
            )
            if row is None:
                return None
            record = _chat_project_from_row(row)
            record.name = name
            record.updated_at = utc_now_iso()
            await conn.execute(
                """
                UPDATE enterprise_chat_projects
                SET name = $3, updated_at = $4, data_json = $5::jsonb
                WHERE id = $1 AND user_id = $2
                """,
                project_id,
                user_id,
                record.name,
                record.updated_at,
                _record_json(record),
            )
            return record

        return await self._write(write)

    async def delete_chat_project(
        self, user_id: str, project_id: str
    ) -> tuple[bool, int, int]:
        """Delete a chat project owned by ``user_id`` and cascade-delete its
        sessions and messages. Returns ``(deleted, deleted_sessions,
        deleted_messages)``."""
        await self._ensure_initialized()

        async def write(conn: Any) -> tuple[bool, int, int]:
            owner = await conn.fetchval(
                """
                SELECT id FROM enterprise_chat_projects
                WHERE id = $1 AND user_id = $2
                """,
                project_id,
                user_id,
            )
            if owner is None:
                return False, 0, 0
            messages_status = await conn.execute(
                "DELETE FROM enterprise_chat_messages WHERE project_id = $1",
                project_id,
            )
            sessions_status = await conn.execute(
                "DELETE FROM enterprise_chat_sessions WHERE project_id = $1",
                project_id,
            )
            await conn.execute(
                "DELETE FROM enterprise_chat_projects WHERE id = $1",
                project_id,
            )
            return True, _rowcount(sessions_status), _rowcount(messages_status)

        return await self._write(write)

    async def create_chat_session(self, record: ChatSessionRecord) -> ChatSessionRecord:
        """Insert a chat session. Raises :class:`MetadataRecordNotFoundError`
        when the parent project does not exist or is not owned by the user."""
        await self._ensure_initialized()

        async def write(conn: Any) -> ChatSessionRecord:
            project = await conn.fetchval(
                """
                SELECT id FROM enterprise_chat_projects
                WHERE id = $1 AND user_id = $2
                """,
                record.project_id,
                record.user_id,
            )
            if project is None:
                raise MetadataRecordNotFoundError(
                    f"Chat project '{record.project_id}' not found"
                )
            await conn.execute(
                """
                INSERT INTO enterprise_chat_sessions (
                    id, project_id, user_id, name, created_at, updated_at, data_json
                ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                """,
                record.id,
                record.project_id,
                record.user_id,
                record.name,
                record.created_at,
                record.updated_at,
                _record_json(record),
            )
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_chat_sessions WHERE id = $1",
                record.id,
            )
            if row is None:
                raise MetadataRecordNotFoundError("Chat session not found")
            return _chat_session_from_row(row)

        return await self._write(write)

    async def get_chat_session(
        self, user_id: str, project_id: str, session_id: str
    ) -> ChatSessionRecord | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT data_json FROM enterprise_chat_sessions
                WHERE id = $1 AND project_id = $2 AND user_id = $3
                """,
                session_id,
                project_id,
                user_id,
            )
        return _chat_session_from_row(row) if row is not None else None

    async def list_chat_sessions(
        self, user_id: str, project_id: str, *, limit: int = 50, offset: int = 0
    ) -> tuple[list[ChatSessionRecord], int]:
        await self._ensure_initialized()
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        async with self._pool_or_raise().acquire() as conn:
            total = await conn.fetchval(
                """
                SELECT COUNT(*) FROM enterprise_chat_sessions
                WHERE project_id = $1 AND user_id = $2
                """,
                project_id,
                user_id,
            )
            rows = await conn.fetch(
                """
                SELECT data_json FROM enterprise_chat_sessions
                WHERE project_id = $1 AND user_id = $2
                ORDER BY updated_at DESC, id DESC
                LIMIT $3 OFFSET $4
                """,
                project_id,
                user_id,
                limit,
                offset,
            )
        return [_chat_session_from_row(row) for row in rows], int(total)

    async def update_chat_session(
        self,
        user_id: str,
        project_id: str,
        session_id: str,
        *,
        name: str | None = None,
        context_rounds: int | None = None,
    ) -> ChatSessionRecord | None:
        """Update the provided fields of an owned session; ``None`` leaves a
        field unchanged. Returns ``None`` when the session is not found."""
        await self._ensure_initialized()

        async def write(conn: Any) -> ChatSessionRecord | None:
            row = await conn.fetchrow(
                """
                SELECT data_json FROM enterprise_chat_sessions
                WHERE id = $1 AND project_id = $2 AND user_id = $3
                FOR UPDATE
                """,
                session_id,
                project_id,
                user_id,
            )
            if row is None:
                return None
            record = _chat_session_from_row(row)
            if name is not None:
                record.name = name
            if context_rounds is not None:
                record.context_rounds = context_rounds
            record.updated_at = utc_now_iso()
            await conn.execute(
                """
                UPDATE enterprise_chat_sessions
                SET name = $4, updated_at = $5, data_json = $6::jsonb
                WHERE id = $1 AND project_id = $2 AND user_id = $3
                """,
                session_id,
                project_id,
                user_id,
                record.name,
                record.updated_at,
                _record_json(record),
            )
            return record

        return await self._write(write)

    async def delete_chat_session(
        self, user_id: str, project_id: str, session_id: str
    ) -> tuple[bool, int]:
        """Delete an owned session and cascade-delete its messages. Returns
        ``(deleted, deleted_messages)``."""
        await self._ensure_initialized()

        async def write(conn: Any) -> tuple[bool, int]:
            owner = await conn.fetchval(
                """
                SELECT id FROM enterprise_chat_sessions
                WHERE id = $1 AND project_id = $2 AND user_id = $3
                """,
                session_id,
                project_id,
                user_id,
            )
            if owner is None:
                return False, 0
            messages_status = await conn.execute(
                "DELETE FROM enterprise_chat_messages WHERE session_id = $1",
                session_id,
            )
            await conn.execute(
                "DELETE FROM enterprise_chat_sessions WHERE id = $1",
                session_id,
            )
            return True, _rowcount(messages_status)

        return await self._write(write)

    async def append_chat_messages(
        self, records: Sequence[ChatMessageRecord]
    ) -> list[ChatMessageRecord]:
        """Atomically append messages to a single owned session.

        Assigns consecutive per-session ``seq`` values and bumps the session's
        ``updated_at`` in the same transaction. All records must target the
        same ``(user_id, project_id, session_id)``. Raises
        :class:`MetadataRecordNotFoundError` when the session does not exist
        or is not owned by the user."""
        if not records:
            return []
        head = records[0]
        if any(
            record.session_id != head.session_id
            or record.project_id != head.project_id
            or record.user_id != head.user_id
            for record in records
        ):
            raise ValueError("All appended messages must target the same session")
        await self._ensure_initialized()

        async def write(conn: Any) -> list[ChatMessageRecord]:
            session_row = await conn.fetchrow(
                """
                SELECT data_json FROM enterprise_chat_sessions
                WHERE id = $1 AND project_id = $2 AND user_id = $3
                FOR UPDATE
                """,
                head.session_id,
                head.project_id,
                head.user_id,
            )
            if session_row is None:
                raise MetadataRecordNotFoundError(
                    f"Chat session '{head.session_id}' not found"
                )
            next_seq = (
                int(
                    await conn.fetchval(
                        """
                        SELECT COALESCE(MAX(seq), 0) FROM enterprise_chat_messages
                        WHERE session_id = $1
                        """,
                        head.session_id,
                    )
                )
                + 1
            )
            for index, record in enumerate(records):
                record.seq = next_seq + index
                await conn.execute(
                    """
                    INSERT INTO enterprise_chat_messages (
                        id, session_id, project_id, user_id, seq, created_at,
                        data_json
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                    """,
                    record.id,
                    record.session_id,
                    record.project_id,
                    record.user_id,
                    record.seq,
                    record.created_at,
                    _record_json(record),
                )
            session = _chat_session_from_row(session_row)
            session.updated_at = utc_now_iso()
            await conn.execute(
                """
                UPDATE enterprise_chat_sessions
                SET updated_at = $2, data_json = $3::jsonb
                WHERE id = $1
                """,
                head.session_id,
                session.updated_at,
                _record_json(session),
            )
            rows = await conn.fetch(
                """
                SELECT data_json FROM enterprise_chat_messages
                WHERE session_id = $1 AND seq >= $2
                ORDER BY seq ASC, id ASC
                """,
                head.session_id,
                next_seq,
            )
            return [_chat_message_from_row(row) for row in rows]

        return await self._write(write)

    async def list_chat_messages(
        self,
        user_id: str,
        project_id: str,
        session_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[ChatMessageRecord], int]:
        await self._ensure_initialized()
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        async with self._pool_or_raise().acquire() as conn:
            total = await conn.fetchval(
                """
                SELECT COUNT(*) FROM enterprise_chat_messages
                WHERE session_id = $1 AND project_id = $2 AND user_id = $3
                """,
                session_id,
                project_id,
                user_id,
            )
            rows = await conn.fetch(
                """
                SELECT data_json FROM enterprise_chat_messages
                WHERE session_id = $1 AND project_id = $2 AND user_id = $3
                ORDER BY seq ASC, id ASC
                LIMIT $4 OFFSET $5
                """,
                session_id,
                project_id,
                user_id,
                limit,
                offset,
            )
        return [_chat_message_from_row(row) for row in rows], int(total)

    async def delete_chat_message(
        self, user_id: str, project_id: str, session_id: str, message_id: str
    ) -> bool:
        await self._ensure_initialized()

        async def write(conn: Any) -> bool:
            status = await conn.execute(
                """
                DELETE FROM enterprise_chat_messages
                WHERE id = $1 AND session_id = $2 AND project_id = $3
                    AND user_id = $4
                """,
                message_id,
                session_id,
                project_id,
                user_id,
            )
            return _rowcount(status) > 0

        return await self._write(write)

    async def set_enterprise_system_setting(
        self, key: str, value: str, *, updated_by: str | None = None
    ) -> None:
        await self._ensure_initialized()

        async def write(conn: Any) -> None:
            now = utc_now_iso()
            await conn.execute(
                """
                INSERT INTO enterprise_system_settings (
                    key, value, updated_by, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (key) DO UPDATE SET
                    value = excluded.value,
                    updated_by = excluded.updated_by,
                    updated_at = excluded.updated_at
                """,
                key,
                value,
                updated_by,
                now,
                now,
            )

        await self._write(write)

    async def get_enterprise_system_setting(
        self, key: str, default: str | None = None
    ) -> str | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            value = await conn.fetchval(
                "SELECT value FROM enterprise_system_settings WHERE key = $1",
                key,
            )
        return default if value is None else str(value)

    async def create_enterprise_api_key(
        self, record: EnterpriseAPIKeyRecord
    ) -> EnterpriseAPIKeyRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> EnterpriseAPIKeyRecord:
            await conn.execute(
                """
                INSERT INTO enterprise_api_keys (
                    id, name, key_hash, key_preview, status, created_by, tenant_id,
                    created_at, updated_at, last_used_at, revoked_at, revoked_by,
                    data_json
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13::jsonb)
                """,
                record.id,
                record.name,
                record.key_hash,
                record.key_preview,
                record.status,
                record.created_by,
                record.tenant_id,
                record.created_at,
                record.updated_at,
                record.last_used_at,
                record.revoked_at,
                record.revoked_by,
                _record_json(record),
            )
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_api_keys WHERE id = $1", record.id
            )
            if row is None:
                raise MetadataRecordNotFoundError(f"API key '{record.id}' not found")
            return _enterprise_api_key_from_row(row)

        return await self._write(write)

    async def get_enterprise_api_key_by_hash(
        self, key_hash: str
    ) -> EnterpriseAPIKeyRecord | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_api_keys WHERE key_hash = $1",
                key_hash,
            )
        return _enterprise_api_key_from_row(row) if row is not None else None

    async def get_enterprise_api_key_by_id(
        self, key_id: str
    ) -> EnterpriseAPIKeyRecord | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_api_keys WHERE id = $1",
                key_id,
            )
        return _enterprise_api_key_from_row(row) if row is not None else None

    async def list_enterprise_api_keys(self) -> list[EnterpriseAPIKeyRecord]:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM enterprise_api_keys
                ORDER BY created_at DESC, id DESC
                """
            )
        return [_enterprise_api_key_from_row(row) for row in rows]

    async def revoke_enterprise_api_key(
        self,
        key_id: str,
        *,
        revoked_by: str | None = None,
        revoked_at: str | None = None,
    ) -> EnterpriseAPIKeyRecord | None:
        await self._ensure_initialized()

        async def write(conn: Any) -> EnterpriseAPIKeyRecord | None:
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_api_keys WHERE id = $1 FOR UPDATE",
                key_id,
            )
            if row is None:
                return None
            current = _enterprise_api_key_from_row(row)
            now = revoked_at or utc_now_iso()
            updated = EnterpriseAPIKeyRecord(
                **{
                    **current.to_dict(),
                    "status": "revoked",
                    "updated_at": now,
                    "revoked_at": now,
                    "revoked_by": revoked_by,
                }
            )
            await conn.execute(
                """
                UPDATE enterprise_api_keys
                SET status = $2, updated_at = $3, revoked_at = $4, revoked_by = $5,
                    data_json = $6::jsonb
                WHERE id = $1
                """,
                key_id,
                updated.status,
                updated.updated_at,
                updated.revoked_at,
                updated.revoked_by,
                _record_json(updated),
            )
            return updated

        return await self._write(write)

    async def mark_enterprise_api_key_used(
        self, key_id: str, *, last_used_at: str | None = None
    ) -> EnterpriseAPIKeyRecord | None:
        await self._ensure_initialized()

        async def write(conn: Any) -> EnterpriseAPIKeyRecord | None:
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_api_keys WHERE id = $1 FOR UPDATE",
                key_id,
            )
            if row is None:
                return None
            current = _enterprise_api_key_from_row(row)
            now = last_used_at or utc_now_iso()
            updated = EnterpriseAPIKeyRecord(
                **{
                    **current.to_dict(),
                    "last_used_at": now,
                    "updated_at": now,
                }
            )
            await conn.execute(
                """
                UPDATE enterprise_api_keys
                SET updated_at = $2, last_used_at = $3, data_json = $4::jsonb
                WHERE id = $1
                """,
                key_id,
                updated.updated_at,
                updated.last_used_at,
                _record_json(updated),
            )
            return updated

        return await self._write(write)

    async def create_enterprise_invitation(
        self, record: EnterpriseInvitationRecord
    ) -> EnterpriseInvitationRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> EnterpriseInvitationRecord:
            await conn.execute(
                """
                INSERT INTO enterprise_invitations (
                    id, token_hash, status, created_by, expires_at, used_by,
                    used_at, created_at, updated_at, data_json
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)
                """,
                record.id,
                record.token_hash,
                record.status,
                record.created_by,
                record.expires_at,
                record.used_by,
                record.used_at,
                record.created_at,
                record.updated_at,
                _record_json(record),
            )
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_invitations WHERE id = $1", record.id
            )
            if row is None:
                raise MetadataRecordNotFoundError(
                    f"Invitation '{record.id}' not found"
                )
            return _enterprise_invitation_from_row(row)

        return await self._write(write)

    async def get_enterprise_invitation_by_token_hash(
        self, token_hash: str
    ) -> EnterpriseInvitationRecord | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_invitations WHERE token_hash = $1",
                token_hash,
            )
        return _enterprise_invitation_from_row(row) if row is not None else None

    async def list_enterprise_invitations(self) -> list[EnterpriseInvitationRecord]:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM enterprise_invitations
                ORDER BY created_at DESC, id DESC
                """
            )
        return [_enterprise_invitation_from_row(row) for row in rows]

    async def consume_enterprise_invitation(
        self, token_hash: str, *, used_by: str | None, used_at: str | None = None
    ) -> EnterpriseInvitationRecord | None:
        await self._ensure_initialized()

        async def write(conn: Any) -> EnterpriseInvitationRecord | None:
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_invitations "
                "WHERE token_hash = $1 FOR UPDATE",
                token_hash,
            )
            if row is None:
                return None
            current = _enterprise_invitation_from_row(row)
            if current.status != "active":
                return None
            now = used_at or utc_now_iso()
            updated = EnterpriseInvitationRecord(
                **{
                    **current.to_dict(),
                    "status": "used",
                    "used_by": used_by,
                    "used_at": now,
                    "updated_at": now,
                }
            )
            await conn.execute(
                """
                UPDATE enterprise_invitations
                SET status = $2, used_by = $3, used_at = $4, updated_at = $5,
                    data_json = $6::jsonb
                WHERE id = $1
                """,
                current.id,
                updated.status,
                updated.used_by,
                updated.used_at,
                updated.updated_at,
                _record_json(updated),
            )
            return updated

        return await self._write(write)

    async def revoke_enterprise_invitation(
        self, invitation_id: str, *, revoked_at: str | None = None
    ) -> EnterpriseInvitationRecord | None:
        await self._ensure_initialized()

        async def write(conn: Any) -> EnterpriseInvitationRecord | None:
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_invitations WHERE id = $1 FOR UPDATE",
                invitation_id,
            )
            if row is None:
                return None
            current = _enterprise_invitation_from_row(row)
            if current.status != "active":
                return current
            now = revoked_at or utc_now_iso()
            updated = EnterpriseInvitationRecord(
                **{
                    **current.to_dict(),
                    "status": "revoked",
                    "updated_at": now,
                }
            )
            await conn.execute(
                """
                UPDATE enterprise_invitations
                SET status = $2, updated_at = $3, data_json = $4::jsonb
                WHERE id = $1
                """,
                current.id,
                updated.status,
                updated.updated_at,
                _record_json(updated),
            )
            return updated

        return await self._write(write)

    async def upsert_kb_acl(self, acl: KBACLRecord) -> KBACLRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> KBACLRecord:
            await conn.execute(
                """
                INSERT INTO enterprise_kb_acl (
                    kb_id, user_id, role, granted_by, created_at, updated_at, data_json
                ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                ON CONFLICT (kb_id, user_id) DO UPDATE SET
                    role = excluded.role,
                    granted_by = excluded.granted_by,
                    updated_at = excluded.updated_at,
                    data_json = excluded.data_json
                """,
                acl.kb_id,
                acl.user_id,
                acl.role,
                acl.granted_by,
                acl.created_at,
                acl.updated_at,
                _record_json(acl),
            )
            row = await conn.fetchrow(
                """
                SELECT data_json FROM enterprise_kb_acl
                WHERE kb_id = $1 AND user_id = $2
                """,
                acl.kb_id,
                acl.user_id,
            )
            if row is None:
                raise MetadataRecordNotFoundError("KB ACL grant not found")
            return _kb_acl_from_row(row)

        return await self._write(write)

    async def delete_kb_acl(self, kb_id: str, user_id: str) -> bool:
        await self._ensure_initialized()

        async def write(conn: Any) -> bool:
            status = await conn.execute(
                "DELETE FROM enterprise_kb_acl WHERE kb_id = $1 AND user_id = $2",
                kb_id,
                user_id,
            )
            return _rowcount(status) > 0

        return await self._write(write)

    async def list_kb_acl(self, kb_id: str) -> list[KBACLRecord]:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM enterprise_kb_acl
                WHERE kb_id = $1
                ORDER BY created_at ASC, user_id ASC
                """,
                kb_id,
            )
        return [_kb_acl_from_row(row) for row in rows]

    async def get_kb_acl_role(self, kb_id: str, user_id: str) -> str | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            role = await conn.fetchval(
                """
                SELECT role FROM enterprise_kb_acl
                WHERE kb_id = $1 AND user_id = $2
                """,
                kb_id,
                user_id,
            )
        return None if role is None else str(role)

    async def list_kb_ids_for_user(self, user_id: str) -> list[str]:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT kb_id FROM enterprise_kb_acl
                WHERE user_id = $1
                ORDER BY kb_id ASC
                """,
                user_id,
            )
        return [str(row["kb_id"]) for row in rows]

    async def upsert_enterprise_tenant(
        self, tenant: EnterpriseTenantRecord
    ) -> EnterpriseTenantRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> EnterpriseTenantRecord:
            await conn.execute(
                """
                INSERT INTO enterprise_tenants (id, name, status, created_at, data_json)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    name = excluded.name,
                    status = excluded.status,
                    data_json = excluded.data_json
                """,
                tenant.id,
                tenant.name,
                tenant.status,
                tenant.created_at,
                _record_json(tenant),
            )
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_tenants WHERE id = $1", tenant.id
            )
            if row is None:
                raise MetadataRecordNotFoundError(f"Tenant '{tenant.id}' not found")
            return _enterprise_tenant_from_row(row)

        return await self._write(write)

    async def get_enterprise_tenant_by_id(
        self, tenant_id: str
    ) -> EnterpriseTenantRecord | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_tenants WHERE id = $1", tenant_id
            )
        return _enterprise_tenant_from_row(row) if row is not None else None

    async def list_enterprise_tenants(self) -> list[EnterpriseTenantRecord]:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM enterprise_tenants
                ORDER BY created_at ASC, id ASC
                """
            )
        return [_enterprise_tenant_from_row(row) for row in rows]

    async def delete_enterprise_tenant(self, tenant_id: str) -> bool:
        await self._ensure_initialized()

        async def write(conn: Any) -> bool:
            status = await conn.execute(
                "DELETE FROM enterprise_tenants WHERE id = $1", tenant_id
            )
            return _rowcount(status) > 0

        return await self._write(write)

    async def upsert_tenant_membership(
        self, membership: EnterpriseTenantMembershipRecord
    ) -> EnterpriseTenantMembershipRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> EnterpriseTenantMembershipRecord:
            await conn.execute(
                """
                INSERT INTO enterprise_tenant_memberships (
                    tenant_id, user_id, role, granted_by, created_at, updated_at, data_json
                ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                ON CONFLICT (tenant_id, user_id) DO UPDATE SET
                    role = excluded.role,
                    granted_by = excluded.granted_by,
                    updated_at = excluded.updated_at,
                    data_json = excluded.data_json
                """,
                membership.tenant_id,
                membership.user_id,
                membership.role,
                membership.granted_by,
                membership.created_at,
                membership.updated_at,
                _record_json(membership),
            )
            row = await conn.fetchrow(
                """
                SELECT data_json FROM enterprise_tenant_memberships
                WHERE tenant_id = $1 AND user_id = $2
                """,
                membership.tenant_id,
                membership.user_id,
            )
            if row is None:
                raise MetadataRecordNotFoundError("Tenant membership not found")
            return _tenant_membership_from_row(row)

        return await self._write(write)

    async def delete_tenant_membership(self, tenant_id: str, user_id: str) -> bool:
        await self._ensure_initialized()

        async def write(conn: Any) -> bool:
            status = await conn.execute(
                """
                DELETE FROM enterprise_tenant_memberships
                WHERE tenant_id = $1 AND user_id = $2
                """,
                tenant_id,
                user_id,
            )
            return _rowcount(status) > 0

        return await self._write(write)

    async def list_tenant_memberships(
        self, tenant_id: str
    ) -> list[EnterpriseTenantMembershipRecord]:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM enterprise_tenant_memberships
                WHERE tenant_id = $1
                ORDER BY created_at ASC, user_id ASC
                """,
                tenant_id,
            )
        return [_tenant_membership_from_row(row) for row in rows]

    async def list_user_tenant_memberships(
        self, user_id: str
    ) -> list[EnterpriseTenantMembershipRecord]:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM enterprise_tenant_memberships
                WHERE user_id = $1
                ORDER BY tenant_id ASC
                """,
                user_id,
            )
        return [_tenant_membership_from_row(row) for row in rows]

    async def get_tenant_membership(
        self, tenant_id: str, user_id: str
    ) -> EnterpriseTenantMembershipRecord | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT data_json FROM enterprise_tenant_memberships
                WHERE tenant_id = $1 AND user_id = $2
                """,
                tenant_id,
                user_id,
            )
        return _tenant_membership_from_row(row) if row is not None else None

    async def upsert_tenant_kb_acl(
        self, acl: EnterpriseTenantKBACLRecord
    ) -> EnterpriseTenantKBACLRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> EnterpriseTenantKBACLRecord:
            await conn.execute(
                """
                INSERT INTO enterprise_tenant_kb_acl (
                    tenant_id, kb_id, role, granted_by, created_at, updated_at, data_json
                ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                ON CONFLICT (tenant_id, kb_id) DO UPDATE SET
                    role = excluded.role,
                    granted_by = excluded.granted_by,
                    updated_at = excluded.updated_at,
                    data_json = excluded.data_json
                """,
                acl.tenant_id,
                acl.kb_id,
                acl.role,
                acl.granted_by,
                acl.created_at,
                acl.updated_at,
                _record_json(acl),
            )
            row = await conn.fetchrow(
                """
                SELECT data_json FROM enterprise_tenant_kb_acl
                WHERE tenant_id = $1 AND kb_id = $2
                """,
                acl.tenant_id,
                acl.kb_id,
            )
            if row is None:
                raise MetadataRecordNotFoundError("Tenant KB ACL grant not found")
            return _tenant_kb_acl_from_row(row)

        return await self._write(write)

    async def delete_tenant_kb_acl(self, tenant_id: str, kb_id: str) -> bool:
        await self._ensure_initialized()

        async def write(conn: Any) -> bool:
            status = await conn.execute(
                """
                DELETE FROM enterprise_tenant_kb_acl
                WHERE tenant_id = $1 AND kb_id = $2
                """,
                tenant_id,
                kb_id,
            )
            return _rowcount(status) > 0

        return await self._write(write)

    async def list_kb_tenant_acl(
        self, kb_id: str
    ) -> list[EnterpriseTenantKBACLRecord]:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM enterprise_tenant_kb_acl
                WHERE kb_id = $1
                ORDER BY created_at ASC, tenant_id ASC
                """,
                kb_id,
            )
        return [_tenant_kb_acl_from_row(row) for row in rows]

    async def get_tenant_kb_acl_role(self, tenant_id: str, kb_id: str) -> str | None:
        await self._ensure_initialized()
        async with self._pool_or_raise().acquire() as conn:
            role = await conn.fetchval(
                """
                SELECT role FROM enterprise_tenant_kb_acl
                WHERE tenant_id = $1 AND kb_id = $2
                """,
                tenant_id,
                kb_id,
            )
        return None if role is None else str(role)

    async def list_kb_ids_for_tenants(self, tenant_ids: Sequence[str]) -> list[str]:
        await self._ensure_initialized()
        normalized_ids = sorted({tenant_id for tenant_id in tenant_ids if tenant_id})
        if not normalized_ids:
            return []
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT kb_id FROM enterprise_tenant_kb_acl
                WHERE tenant_id = ANY($1::text[])
                ORDER BY kb_id ASC
                """,
                normalized_ids,
            )
        return [str(row["kb_id"]) for row in rows]

    async def append_audit_event(
        self, event: AuditEventRecord
    ) -> AuditEventRecord:
        await self._ensure_initialized()

        async def write(conn: Any) -> AuditEventRecord:
            await conn.execute(
                """
                INSERT INTO enterprise_audit_events (
                    id, event_type, actor_user_id, target_type, target_id,
                    created_at, data_json
                ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                """,
                event.id,
                event.event_type,
                event.actor_user_id,
                event.target_type,
                event.target_id,
                event.created_at,
                _record_json(event),
            )
            row = await conn.fetchrow(
                "SELECT data_json FROM enterprise_audit_events WHERE id = $1",
                event.id,
            )
            if row is None:
                raise MetadataRecordNotFoundError(f"Audit event '{event.id}' not found")
            return _audit_event_from_row(row)

        return await self._write(write)

    async def list_audit_events(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        event_type: str | None = None,
        actor_user_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
    ) -> list[AuditEventRecord]:
        await self._ensure_initialized()
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        clauses: list[str] = []
        params: list[Any] = []

        def _add(expr_tmpl: str, value: Any) -> None:
            params.append(value)
            clauses.append(expr_tmpl.format(idx=len(params)))

        for column, value in (
            ("event_type", event_type),
            ("actor_user_id", actor_user_id),
            ("target_type", target_type),
            ("target_id", target_id),
        ):
            if value:
                _add(f"{column} = ${{idx}}", value)
        if created_after:
            _add("created_at >= ${idx}", created_after)
        if created_before:
            _add("created_at <= ${idx}", created_before)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        limit_idx = len(params)
        params.append(offset)
        offset_idx = len(params)
        async with self._pool_or_raise().acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT data_json FROM enterprise_audit_events
                {where}
                ORDER BY created_at DESC, id DESC
                LIMIT ${limit_idx} OFFSET ${offset_idx}
                """,
                *params,
            )
        return [_audit_event_from_row(row) for row in rows]

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
                ("enterprise_user_kb_query_settings", "enterprise_user_kb_query_settings"),
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

            CREATE TABLE IF NOT EXISTS enterprise_users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                tenant_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                data_json JSONB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_enterprise_users_status
                ON enterprise_users (status);
            CREATE INDEX IF NOT EXISTS idx_enterprise_users_tenant
                ON enterprise_users (tenant_id);

            CREATE TABLE IF NOT EXISTS enterprise_system_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_by TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS enterprise_user_kb_query_settings (
                user_id TEXT NOT NULL,
                kb_id TEXT NOT NULL,
                user_prompt TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                data_json JSONB NOT NULL,
                PRIMARY KEY (user_id, kb_id),
                FOREIGN KEY (user_id) REFERENCES enterprise_users(id)
            );
            CREATE INDEX IF NOT EXISTS idx_enterprise_user_kb_query_settings_kb
                ON enterprise_user_kb_query_settings (kb_id, user_id);

            CREATE TABLE IF NOT EXISTS enterprise_chat_projects (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                data_json JSONB NOT NULL,
                FOREIGN KEY (user_id) REFERENCES enterprise_users(id)
            );
            CREATE INDEX IF NOT EXISTS idx_enterprise_chat_projects_user
                ON enterprise_chat_projects (user_id, updated_at DESC);

            CREATE TABLE IF NOT EXISTS enterprise_chat_sessions (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                data_json JSONB NOT NULL,
                FOREIGN KEY (project_id) REFERENCES enterprise_chat_projects(id),
                FOREIGN KEY (user_id) REFERENCES enterprise_users(id)
            );
            CREATE INDEX IF NOT EXISTS idx_enterprise_chat_sessions_project
                ON enterprise_chat_sessions (project_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_enterprise_chat_sessions_user
                ON enterprise_chat_sessions (user_id, project_id);

            CREATE TABLE IF NOT EXISTS enterprise_chat_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                seq BIGINT NOT NULL,
                created_at TEXT NOT NULL,
                data_json JSONB NOT NULL,
                FOREIGN KEY (session_id) REFERENCES enterprise_chat_sessions(id),
                FOREIGN KEY (user_id) REFERENCES enterprise_users(id)
            );
            CREATE INDEX IF NOT EXISTS idx_enterprise_chat_messages_session
                ON enterprise_chat_messages (session_id, seq);
            CREATE INDEX IF NOT EXISTS idx_enterprise_chat_messages_project
                ON enterprise_chat_messages (project_id);
            CREATE INDEX IF NOT EXISTS idx_enterprise_chat_messages_user
                ON enterprise_chat_messages (user_id);

            CREATE TABLE IF NOT EXISTS enterprise_api_keys (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                key_hash TEXT NOT NULL UNIQUE,
                key_preview TEXT NOT NULL,
                status TEXT NOT NULL,
                created_by TEXT,
                tenant_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_used_at TEXT,
                revoked_at TEXT,
                revoked_by TEXT,
                data_json JSONB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_enterprise_api_keys_status
                ON enterprise_api_keys (status);
            CREATE INDEX IF NOT EXISTS idx_enterprise_api_keys_created_by
                ON enterprise_api_keys (created_by, status);
            CREATE INDEX IF NOT EXISTS idx_enterprise_api_keys_tenant
                ON enterprise_api_keys (tenant_id, status);

            CREATE TABLE IF NOT EXISTS enterprise_kb_acl (
                kb_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                granted_by TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                data_json JSONB NOT NULL,
                PRIMARY KEY (kb_id, user_id),
                FOREIGN KEY (user_id) REFERENCES enterprise_users(id)
            );
            CREATE INDEX IF NOT EXISTS idx_enterprise_kb_acl_user
                ON enterprise_kb_acl (user_id, kb_id);

            CREATE TABLE IF NOT EXISTS enterprise_tenants (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                data_json JSONB NOT NULL
            );

            CREATE TABLE IF NOT EXISTS enterprise_tenant_memberships (
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                granted_by TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                data_json JSONB NOT NULL,
                PRIMARY KEY (tenant_id, user_id),
                FOREIGN KEY (user_id) REFERENCES enterprise_users(id)
            );
            CREATE INDEX IF NOT EXISTS idx_enterprise_tenant_memberships_user
                ON enterprise_tenant_memberships (user_id, tenant_id);

            CREATE TABLE IF NOT EXISTS enterprise_tenant_kb_acl (
                tenant_id TEXT NOT NULL,
                kb_id TEXT NOT NULL,
                role TEXT NOT NULL,
                granted_by TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                data_json JSONB NOT NULL,
                PRIMARY KEY (tenant_id, kb_id)
            );
            CREATE INDEX IF NOT EXISTS idx_enterprise_tenant_kb_acl_kb
                ON enterprise_tenant_kb_acl (kb_id, tenant_id);

            CREATE TABLE IF NOT EXISTS enterprise_audit_events (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                actor_user_id TEXT,
                target_type TEXT,
                target_id TEXT,
                created_at TEXT NOT NULL,
                data_json JSONB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_enterprise_audit_events_created
                ON enterprise_audit_events (created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_enterprise_audit_events_actor
                ON enterprise_audit_events (actor_user_id);

            CREATE TABLE IF NOT EXISTS enterprise_invitations (
                id TEXT PRIMARY KEY,
                token_hash TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                created_by TEXT,
                expires_at TEXT,
                used_by TEXT,
                used_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                data_json JSONB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_enterprise_invitations_status
                ON enterprise_invitations (status);
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
