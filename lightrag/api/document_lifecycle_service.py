from __future__ import annotations

import hashlib
import json
import mimetypes
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from lightrag.api.kb_service import KnowledgeBaseService, utc_now_iso
from lightrag.api.metadata_store import (
    ActiveDocumentParseJobError,
    ArtifactRecord,
    DocumentRecord,
    JobRecord,
    MetadataRecordNotFoundError,
    SQLiteMetadataStore,
)
from lightrag.constants import (
    DOCLING_RAW_DIR_SUFFIX,
    FULL_DOCS_FORMAT_LIGHTRAG,
    FULL_DOCS_FORMAT_PENDING_PARSE,
    MINERU_RAW_DIR_SUFFIX,
    PARSED_DIR_SUFFIX,
    PARSER_ENGINE_DOCLING,
    PARSER_ENGINE_LEGACY,
    PARSER_ENGINE_MINERU,
    PARSER_ENGINE_NATIVE,
    SUPPORTED_PARSER_ENGINES,
)
from lightrag.parser.routing import (
    normalize_parser_engine,
    parser_engine_supports_suffix,
    parser_suffix,
    resolve_file_parser_directives,
    sanitize_process_options,
    validate_process_options,
)
from lightrag.utils import compute_mdhash_id, generate_track_id

SourceType = Literal["upload", "text"]

# Sanitization rule: drop only path separators, control characters, and
# characters that are unsafe inside a filename on common filesystems
# (``<>:"|?*`` plus ASCII < 0x20). CJK / Latin-extended / accented letters
# stay intact so two PDFs whose names differ only in CJK characters don't
# both collapse to ``_.pdf`` and collide downstream in LightRAG's
# filename-based dedup.
_FILENAME_FORBIDDEN_CHARS = set('<>:"|?*\\/')


def _sanitize_filename_char(char: str) -> str:
    if not char:
        return "_"
    code = ord(char)
    if code < 0x20 or code == 0x7F:
        return "_"
    if char in _FILENAME_FORBIDDEN_CHARS:
        return "_"
    return char


_PARSEABLE_ENGINES = {
    PARSER_ENGINE_NATIVE,
    PARSER_ENGINE_MINERU,
    PARSER_ENGINE_DOCLING,
}


class DocumentLifecycleError(RuntimeError):
    pass


@dataclass(slots=True)
class DocumentSourceInput:
    source_name: str
    content: bytes
    source_type: SourceType
    content_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentBatchResult:
    job: JobRecord
    batch_id: str
    documents: list[DocumentRecord]


@dataclass(slots=True)
class DocumentParsePlan:
    document: DocumentRecord
    parser_engine: str
    process_options: str
    parser_hash: str
    lightrag_doc_id: str
    source_path: Path
    force_reparse: bool
    auto_index: bool


@dataclass(slots=True)
class DocumentBatchParsePlan:
    batch_id: str
    plans: list[DocumentParsePlan]
    failures: list[dict[str, Any]]


@dataclass(slots=True)
class DocumentParseResult:
    document: DocumentRecord
    artifacts: list[ArtifactRecord]


@dataclass(slots=True)
class DocumentDeleteFileResult:
    deleted_source: bool = False
    deleted_artifacts: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DocumentReplacementSource:
    source_name: str
    content: bytes
    source_hash: str
    content_type: str | None
    size_bytes: int


@dataclass(slots=True)
class ArtifactFileResult:
    artifact: ArtifactRecord
    path: Path
    filename: str
    media_type: str
    is_directory: bool = False


class DocumentLifecycleService:
    def __init__(
        self,
        kb_service: KnowledgeBaseService,
        metadata_store: SQLiteMetadataStore,
        source_root: str | Path,
    ):
        self._kb_service = kb_service
        self._metadata_store = metadata_store
        self._source_root = Path(source_root)

    @property
    def metadata_store(self) -> SQLiteMetadataStore:
        return self._metadata_store

    @property
    def kb_service(self) -> KnowledgeBaseService:
        return self._kb_service

    async def create_source_batch(
        self,
        kb_id: str,
        sources: list[DocumentSourceInput],
        *,
        auto_parse: bool = False,
        auto_index: bool = False,
        parser_engine: str | None = None,
        process_options: str | None = None,
        idempotency_key: str | None = None,
    ) -> DocumentBatchResult:
        if not sources:
            raise ValueError("At least one document source is required")

        record = await self._kb_service.get(kb_id)
        job_type = "parse" if auto_parse else "upload"
        workspace_dir = self._source_root / record.workspace
        workspace_dir.mkdir(parents=True, exist_ok=True)
        batch_id = generate_track_id("batch")
        job_id = generate_track_id(f"job_{job_type}")
        document_status = "parse_queued" if auto_parse else "uploaded"
        now = utc_now_iso()
        saved_paths: list[Path] = []
        saved_dirs: list[Path] = []
        documents: list[DocumentRecord] = []
        source_fingerprints: list[dict[str, Any]] = []

        try:
            for source in sources:
                if not source.content:
                    raise ValueError("Document content cannot be empty")
                safe_name = _sanitize_source_name(source.source_name)
                content_hash = _content_hash(source.content)
                source_fingerprints.append(
                    {
                        "source_name": safe_name,
                        "source_type": source.source_type,
                        "content_type": source.content_type,
                        "source_hash": content_hash,
                        "metadata": source.metadata,
                    }
                )
                document_id = f"doc_{uuid4().hex[:12]}"
                target_path = _write_source_file(
                    workspace_dir, document_id, safe_name, source.content
                )
                saved_paths.append(target_path)
                saved_dirs.append(target_path.parent)
                documents.append(
                    DocumentRecord(
                        id=document_id,
                        kb_id=record.id,
                        workspace=record.workspace,
                        lightrag_doc_id=None,
                        source_type=source.source_type,
                        source_name=safe_name,
                        source_uri=str(target_path),
                        source_hash=content_hash,
                        content_type=source.content_type,
                        size_bytes=len(source.content),
                        parser_hash=None,
                        index_hash=None,
                        status=document_status,
                        enabled=True,
                        archived=False,
                        chunks_count=None,
                        entity_count=None,
                        relation_count=None,
                        error_code=None,
                        error_message=None,
                        metadata={
                            **source.metadata,
                            "batch_id": batch_id,
                            "auto_parse": auto_parse,
                            "auto_index": auto_index,
                            "parser_engine": parser_engine,
                            "process_options": process_options,
                            **(
                                {
                                    "pending_parse_job_id": job_id,
                                    "pending_parse_batch_id": batch_id,
                                }
                                if auto_parse
                                else {}
                            ),
                        },
                        created_at=now,
                        updated_at=now,
                        deleted_at=None,
                    )
                )

            job = JobRecord(
                id=job_id,
                kb_id=record.id,
                workspace=record.workspace,
                batch_id=batch_id,
                document_id=None,
                job_type=job_type,
                status="queued" if auto_parse else "succeeded",
                stage="parsing" if auto_parse else "uploading",
                progress=0.0 if auto_parse else 1.0,
                total_items=len(documents),
                completed_items=0 if auto_parse else len(documents),
                failed_items=0,
                idempotency_key=idempotency_key,
                config_version_id=record.active_config_version_id,
                config_hash=None,
                retry_count=0,
                max_retries=3,
                payload={
                    "auto_parse": auto_parse,
                    "auto_index": auto_index,
                    "parser_engine": parser_engine,
                    "process_options": process_options,
                    "source_types": sorted({source.source_type for source in sources}),
                    "document_ids": [document.id for document in documents],
                    "idempotency_fingerprint": _idempotency_fingerprint(
                        {
                            "auto_parse": auto_parse,
                            "auto_index": auto_index,
                            "parser_engine": parser_engine,
                            "process_options": process_options,
                            "sources": source_fingerprints,
                        }
                    ),
                },
                result={"documents_created": len(documents)}
                if not auto_parse
                else None,
                error_code=None,
                error_message=None,
                created_at=now,
                updated_at=now,
                queued_at=now if auto_parse else None,
                started_at=now if not auto_parse else None,
                finished_at=now if not auto_parse else None,
                cancelled_at=None,
            )
            (
                created_documents,
                created_job,
                created,
            ) = await self._metadata_store.create_documents_and_job(documents, job)
            if not created:
                self._cleanup_saved_sources(saved_paths, saved_dirs)
            return DocumentBatchResult(
                job=created_job,
                batch_id=created_job.batch_id or batch_id,
                documents=created_documents,
            )
        except Exception:
            self._cleanup_saved_sources(saved_paths, saved_dirs)
            raise

    async def list_documents(
        self,
        kb_id: str,
        *,
        status: str | None = None,
        source_name: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[DocumentRecord], int]:
        record = await self._kb_service.get(kb_id)
        return await self._metadata_store.list_documents(
            record.id,
            status=status,
            source_name=source_name,
            limit=limit,
            offset=offset,
        )

    async def get_document(self, kb_id: str, document_id: str) -> DocumentRecord:
        record = await self._kb_service.get(kb_id)
        return await self._metadata_store.get_document(record.id, document_id)

    async def get_documents_by_ids(
        self, kb_id: str, document_ids: list[str]
    ) -> list[DocumentRecord]:
        record = await self._kb_service.get(kb_id)
        return await self._metadata_store.get_documents_by_ids(record.id, document_ids)

    async def get_documents_by_source_keys(
        self, kb_id: str, source_keys: list[str]
    ) -> dict[str, DocumentRecord]:
        record = await self._kb_service.get(kb_id)
        return await self._metadata_store.get_documents_by_source_keys(
            record.id, source_keys
        )

    async def update_document(
        self,
        kb_id: str,
        document_id: str,
        *,
        metadata_patch: dict[str, Any] | None = None,
        enabled: bool | None = None,
        archived: bool | None = None,
    ) -> DocumentRecord:
        record = await self._kb_service.get(kb_id)
        return await self._metadata_store.update_document(
            record.id,
            document_id,
            metadata_patch=metadata_patch,
            enabled=enabled,
            archived=archived,
        )

    async def claim_delete(
        self,
        kb_id: str,
        document_id: str,
        *,
        job: JobRecord,
        delete_source_file: bool = False,
        delete_artifacts: bool = False,
    ) -> DocumentRecord:
        record = await self._kb_service.get(kb_id)
        return await self._metadata_store.claim_document_deleting(
            record.id,
            document_id,
            metadata_patch={
                "pending_delete_job_id": job.id,
                "delete_source_file": delete_source_file,
                "delete_artifacts": delete_artifacts,
            },
        )

    async def claim_batch_delete(
        self,
        kb_id: str,
        document_ids: list[str],
        *,
        job: JobRecord,
        delete_source_file: bool = False,
        delete_artifacts: bool = False,
    ) -> tuple[list[DocumentRecord], list[dict[str, Any]]]:
        record = await self._kb_service.get(kb_id)
        claims = [
            (
                document_id,
                {
                    "pending_delete_job_id": job.id,
                    "delete_source_file": delete_source_file,
                    "delete_artifacts": delete_artifacts,
                },
            )
            for document_id in document_ids
        ]
        return await self._metadata_store.claim_documents_deleting(record.id, claims)

    async def complete_delete(
        self,
        kb_id: str,
        document_id: str,
        *,
        job_id: str,
        lightrag_result: dict[str, Any] | None = None,
        file_result: DocumentDeleteFileResult | None = None,
    ) -> DocumentRecord:
        record = await self._kb_service.get(kb_id)
        return await self._metadata_store.complete_document_delete(
            record.id,
            document_id,
            metadata_patch={
                "pending_delete_job_id": None,
                "current_delete_job_id": None,
                "last_delete_job_id": job_id,
                "last_deleted_at": utc_now_iso(),
                "lightrag_delete_result": lightrag_result,
                "file_delete_result": asdict(file_result) if file_result else None,
            },
        )

    async def fail_delete(
        self,
        kb_id: str,
        document_id: str,
        *,
        job_id: str,
        error_code: str,
        error_message: str,
    ) -> DocumentRecord:
        record = await self._kb_service.get(kb_id)
        return await self._metadata_store.fail_document_delete(
            record.id,
            document_id,
            error_code=error_code,
            error_message=error_message,
            metadata_patch={
                "pending_delete_job_id": None,
                "current_delete_job_id": None,
                "last_failed_delete_job_id": job_id,
            },
        )

    def prepare_replacement_source(
        self, source: DocumentSourceInput
    ) -> DocumentReplacementSource:
        if not source.content:
            raise ValueError("Replacement document content cannot be empty")
        safe_name = _sanitize_source_name(source.source_name)
        return DocumentReplacementSource(
            source_name=safe_name,
            content=source.content,
            source_hash=_content_hash(source.content),
            content_type=source.content_type,
            size_bytes=len(source.content),
        )

    async def claim_replace(
        self,
        kb_id: str,
        document_id: str,
        *,
        job: JobRecord,
        replacement: DocumentReplacementSource,
        delete_source_file: bool = True,
        delete_artifacts: bool = True,
        delete_llm_cache: bool = False,
        auto_parse: bool = False,
        auto_index: bool = False,
        parser_engine: str | None = None,
        process_options: str | None = None,
        force_reparse: bool = False,
    ) -> DocumentRecord:
        record = await self._kb_service.get(kb_id)
        return await self._metadata_store.claim_document_replacing(
            record.id,
            document_id,
            metadata_patch={
                "pending_replace_job_id": job.id,
                "replacement_source_name": replacement.source_name,
                "replacement_source_hash": replacement.source_hash,
                "delete_source_file": delete_source_file,
                "delete_artifacts": delete_artifacts,
                "delete_llm_cache": delete_llm_cache,
                "auto_parse": auto_parse,
                "auto_index": auto_index,
                "parser_engine": parser_engine,
                "process_options": process_options,
                "force_reparse": force_reparse,
            },
        )

    async def replace_document_source(
        self,
        kb_id: str,
        document: DocumentRecord,
        *,
        job_id: str,
        replacement: DocumentReplacementSource,
        delete_source_file: bool = True,
        delete_artifacts: bool = True,
        lightrag_delete_result: dict[str, Any] | None = None,
    ) -> tuple[DocumentRecord, DocumentDeleteFileResult]:
        record = await self._kb_service.get(kb_id)
        workspace_dir = (self._source_root / record.workspace).resolve(strict=False)
        document_dir = (workspace_dir / document.id).resolve(strict=False)
        try:
            document_dir.relative_to(workspace_dir)
        except ValueError as exc:
            raise ValueError(
                "Document replacement path escapes workspace directory"
            ) from exc
        document_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        staging_path = document_dir / f".replace-{job_id}.tmp"
        staging_path.unlink(missing_ok=True)
        try:
            with staging_path.open("xb") as output:
                output.write(replacement.content)
                output.flush()

            file_result = await self.cleanup_document_files(
                kb_id,
                document,
                delete_source_file=delete_source_file,
                delete_artifacts=delete_artifacts,
            )
            if file_result.errors:
                raise RuntimeError("; ".join(file_result.errors))

            target_path = _replacement_source_target(
                document_dir, replacement.source_name, job_id
            )
            shutil.move(str(staging_path), str(target_path))
            replaced = await self._metadata_store.complete_document_replace(
                record.id,
                document.id,
                source_name=replacement.source_name,
                source_uri=str(target_path),
                source_hash=replacement.source_hash,
                content_type=replacement.content_type,
                size_bytes=replacement.size_bytes,
                metadata_patch={
                    "pending_replace_job_id": None,
                    "current_replace_job_id": None,
                    "last_replace_job_id": job_id,
                    "last_replaced_at": utc_now_iso(),
                    "previous_lightrag_doc_id": document.lightrag_doc_id,
                    "lightrag_delete_result": lightrag_delete_result,
                    "file_replace_result": asdict(file_result),
                },
            )
            return replaced, file_result
        except Exception:
            staging_path.unlink(missing_ok=True)
            raise

    async def preflight_replace_cleanup(
        self,
        kb_id: str,
        document: DocumentRecord,
        *,
        delete_source_file: bool,
        delete_artifacts: bool,
    ) -> None:
        record = await self._kb_service.get(kb_id)
        workspace_dir = (self._source_root / record.workspace).resolve(strict=False)
        artifacts, _total = await self._metadata_store.list_document_artifacts(
            record.id, document.id, limit=200
        )
        _validate_document_cleanup_paths(
            workspace_dir,
            document,
            artifacts,
            delete_source_file=delete_source_file,
            delete_artifacts=delete_artifacts,
        )

    async def fail_replace(
        self,
        kb_id: str,
        document_id: str,
        *,
        job_id: str,
        error_code: str,
        error_message: str,
        clear_index_metadata: bool = False,
        lightrag_delete_result: dict[str, Any] | None = None,
    ) -> DocumentRecord:
        record = await self._kb_service.get(kb_id)
        return await self._metadata_store.fail_document_replace(
            record.id,
            document_id,
            error_code=error_code,
            error_message=error_message,
            clear_index_metadata=clear_index_metadata,
            metadata_patch={
                "pending_replace_job_id": None,
                "current_replace_job_id": None,
                "last_failed_replace_job_id": job_id,
                "lightrag_delete_result": lightrag_delete_result,
            },
        )

    async def cleanup_document_files(
        self,
        kb_id: str,
        document: DocumentRecord,
        *,
        delete_source_file: bool,
        delete_artifacts: bool,
    ) -> DocumentDeleteFileResult:
        record = await self._kb_service.get(kb_id)
        workspace_dir = (self._source_root / record.workspace).resolve(strict=False)
        # Canonical document directory: <source_root>/<workspace>/<document_id>.
        # Anchoring here (rather than trusting source_uri.parent) ensures both
        # source and artifact cleanup are contained to THIS document's dir, so a
        # crafted source_uri that lives inside the workspace but outside the doc
        # dir cannot escape the per-document boundary.
        document_dir = (workspace_dir / document.id).resolve(strict=False)
        result = DocumentDeleteFileResult()
        artifacts, _total = await self._metadata_store.list_document_artifacts(
            record.id, document.id, limit=200
        )
        source_path: Path | None = None
        if delete_source_file or delete_artifacts:
            try:
                source_path = _safe_document_path(
                    workspace_dir, document_dir, document.source_uri
                )
            except ValueError as exc:
                result.errors.append(f"source: {exc}")
                return result

        if delete_artifacts:
            for artifact in artifacts:
                try:
                    artifact_path = _safe_document_path(
                        workspace_dir,
                        document_dir,
                        artifact.uri,
                    )
                    if not artifact_path.exists():
                        result.skipped.append(artifact.uri)
                        continue
                    _remove_path(artifact_path)
                    result.deleted_artifacts.append(str(artifact_path))
                except (OSError, ValueError) as exc:
                    result.errors.append(f"artifact {artifact.id}: {exc}")

        if delete_source_file and source_path is not None:
            try:
                if source_path.exists():
                    _remove_path(source_path)
                    result.deleted_source = True
                    _remove_empty_parents(source_path.parent, stop_at=workspace_dir)
                else:
                    result.skipped.append(document.source_uri)
            except (OSError, ValueError) as exc:
                result.errors.append(f"source: {exc}")
        return result

    async def create_parse_plan(
        self,
        kb_id: str,
        document_id: str,
        *,
        parser_engine: str | None = None,
        process_options: str | None = None,
        force_reparse: bool = False,
        auto_index: bool = False,
    ) -> DocumentParsePlan:
        document = await self.get_document(kb_id, document_id)
        source_path = Path(document.source_uri)
        if not source_path.is_file():
            raise FileNotFoundError(f"Document source not found: {document.source_uri}")

        engine, options = _resolve_parse_directives(
            source_path,
            document,
            parser_engine=parser_engine,
            process_options=process_options,
        )
        lightrag_doc_id = compute_mdhash_id(str(source_path), prefix="doc-")
        parser_hash = _parser_hash(engine=engine, process_options=options)
        return DocumentParsePlan(
            document=document,
            parser_engine=engine,
            process_options=options,
            parser_hash=parser_hash,
            lightrag_doc_id=lightrag_doc_id,
            source_path=source_path,
            force_reparse=force_reparse,
            auto_index=auto_index,
        )

    async def create_batch_parse_plan(
        self,
        kb_id: str,
        document_ids: list[str],
        *,
        parser_engine: str | None = None,
        process_options: str | None = None,
        force_reparse: bool = False,
        auto_index: bool = False,
    ) -> DocumentBatchParsePlan:
        _validate_parse_request_directives(
            parser_engine=parser_engine, process_options=process_options
        )
        record = await self._kb_service.get(kb_id)
        plans: list[DocumentParsePlan] = []
        failures: list[dict[str, Any]] = []
        for document_id in document_ids:
            try:
                plan = await self.create_parse_plan(
                    record.id,
                    document_id,
                    parser_engine=parser_engine,
                    process_options=process_options,
                    force_reparse=force_reparse,
                    auto_index=auto_index,
                )
                plans.append(plan)
            except MetadataRecordNotFoundError as exc:
                failures.append(
                    _batch_parse_failure(
                        document_id,
                        error_code="document_not_found",
                        error_message=str(exc),
                    )
                )
            except FileNotFoundError as exc:
                failures.append(
                    _batch_parse_failure(
                        document_id,
                        error_code="source_not_found",
                        error_message=str(exc),
                    )
                )
            except ValueError as exc:
                failures.append(
                    _batch_parse_failure(
                        document_id,
                        error_code="invalid_parse_request",
                        error_message=str(exc),
                    )
                )
        return DocumentBatchParsePlan(
            batch_id=generate_track_id("batch"), plans=plans, failures=failures
        )

    async def mark_batch_parse_queued(
        self, kb_id: str, *, job: JobRecord, plans: list[DocumentParsePlan]
    ) -> list[DocumentRecord]:
        queued_documents, failures = await self.claim_batch_parse_queued(
            kb_id, job=job, plans=plans
        )
        if failures:
            failure = failures[0]
            if failure["error_code"] == "parse_job_active":
                raise ActiveDocumentParseJobError(
                    str(failure["document_id"]),
                    str(failure.get("existing_job_id") or "unknown"),
                )
            raise MetadataRecordNotFoundError(str(failure["error_message"]))
        return queued_documents

    async def claim_batch_parse_queued(
        self, kb_id: str, *, job: JobRecord, plans: list[DocumentParsePlan]
    ) -> tuple[list[DocumentRecord], list[dict[str, Any]]]:
        claims = [
            (
                plan.document.id,
                {
                    "pending_parse_job_id": job.id,
                    "pending_parse_batch_id": job.batch_id,
                    "pending_parser_hash": plan.parser_hash,
                    "pending_lightrag_doc_id": plan.lightrag_doc_id,
                    "parser_engine": plan.parser_engine,
                    "process_options": plan.process_options,
                    "force_reparse": plan.force_reparse,
                    "auto_index": plan.auto_index,
                },
            )
            for plan in plans
        ]
        return await self._metadata_store.claim_documents_parse_queued(kb_id, claims)

    async def mark_parse_queued(
        self, kb_id: str, document_id: str, *, job: JobRecord, plan: DocumentParsePlan
    ) -> DocumentRecord:
        return await self._metadata_store.mark_document_parse_queued(
            kb_id,
            document_id,
            metadata_patch={
                "pending_parse_job_id": job.id,
                "pending_parser_hash": plan.parser_hash,
                "pending_lightrag_doc_id": plan.lightrag_doc_id,
                "parser_engine": plan.parser_engine,
                "process_options": plan.process_options,
                "force_reparse": plan.force_reparse,
                "auto_index": plan.auto_index,
            },
        )

    async def mark_parse_running(
        self, kb_id: str, document_id: str, *, job_id: str
    ) -> DocumentRecord:
        return await self._metadata_store.mark_document_parsing(
            kb_id,
            document_id,
            metadata_patch={
                "current_parse_job_id": job_id,
                "parse_started_at": utc_now_iso(),
            },
        )

    async def run_parse(self, rag: Any, plan: DocumentParsePlan) -> dict[str, Any]:
        content_data = {
            "parse_format": FULL_DOCS_FORMAT_PENDING_PARSE,
            "parse_engine": plan.parser_engine,
            "process_options": plan.process_options,
            "force_reparse": plan.force_reparse,
            "archive_source_after_parse": False,
        }
        source_path = str(plan.source_path)
        if plan.parser_engine == PARSER_ENGINE_NATIVE:
            return await rag.parse_native(
                plan.lightrag_doc_id, source_path, content_data
            )
        if plan.parser_engine == PARSER_ENGINE_MINERU:
            return await rag.parse_mineru(
                plan.lightrag_doc_id, source_path, content_data
            )
        if plan.parser_engine == PARSER_ENGINE_DOCLING:
            return await rag.parse_docling(
                plan.lightrag_doc_id, source_path, content_data
            )
        raise ValueError(
            f"Unsupported parser engine for KB parse: {plan.parser_engine}"
        )

    async def complete_parse(
        self,
        kb_id: str,
        document_id: str,
        *,
        job_id: str,
        plan: DocumentParsePlan,
        parsed_data: dict[str, Any],
    ) -> DocumentParseResult:
        artifacts = _build_parse_artifacts(plan, parsed_data)
        (
            document,
            created_artifacts,
        ) = await self._metadata_store.complete_document_parse(
            kb_id,
            document_id,
            parser_hash=plan.parser_hash,
            lightrag_doc_id=plan.lightrag_doc_id,
            artifacts=artifacts,
            metadata_patch={
                "last_parse_job_id": job_id,
                "last_parsed_at": utc_now_iso(),
                "parse_engine": plan.parser_engine,
                "process_options": plan.process_options,
                "parse_format": parsed_data.get(
                    "parse_format", FULL_DOCS_FORMAT_LIGHTRAG
                ),
                "blocks_path": parsed_data.get("blocks_path"),
                "artifact_count": len(artifacts),
                "parse_stage_skipped": bool(parsed_data.get("parse_stage_skipped")),
            },
        )
        return DocumentParseResult(document=document, artifacts=created_artifacts)

    async def fail_parse(
        self,
        kb_id: str,
        document_id: str,
        *,
        job_id: str,
        plan: DocumentParsePlan,
        error_code: str,
        error_message: str,
    ) -> DocumentRecord:
        return await self._metadata_store.fail_document_parse(
            kb_id,
            document_id,
            error_code=error_code,
            error_message=error_message,
            metadata_patch={
                "last_failed_parse_job_id": job_id,
                "last_failed_parser_hash": plan.parser_hash,
            },
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
        record = await self._kb_service.get(kb_id)
        await self._metadata_store.get_document(record.id, document_id)
        return await self._metadata_store.list_document_artifacts(
            record.id,
            document_id,
            artifact_type=artifact_type,
            limit=limit,
            offset=offset,
        )

    async def get_document_artifact(
        self, kb_id: str, document_id: str, artifact_id: str
    ) -> ArtifactRecord:
        record = await self._kb_service.get(kb_id)
        await self._metadata_store.get_document(record.id, document_id)
        return await self._metadata_store.get_document_artifact(
            record.id, document_id, artifact_id
        )

    async def get_document_artifact_file(
        self, kb_id: str, document_id: str, artifact_id: str
    ) -> ArtifactFileResult:
        record = await self._kb_service.get(kb_id)
        document = await self._metadata_store.get_document(record.id, document_id)
        artifact = await self._metadata_store.get_document_artifact(
            record.id, document_id, artifact_id
        )
        artifact_path, is_directory = _resolve_artifact_path(
            self._source_root, document, artifact
        )
        media_type = _artifact_media_type(
            document, artifact, artifact_path, is_directory
        )
        return ArtifactFileResult(
            artifact=artifact,
            path=artifact_path,
            filename=artifact_path.name + (".zip" if is_directory else ""),
            media_type=media_type,
            is_directory=is_directory,
        )

    async def _documents_for_job(
        self, kb_id: str, job: JobRecord
    ) -> list[DocumentRecord]:
        document_ids = job.payload.get("document_ids")
        if isinstance(document_ids, list) and all(
            isinstance(document_id, str) for document_id in document_ids
        ):
            return await self._metadata_store.get_documents_by_ids(kb_id, document_ids)
        if job.batch_id:
            return await self._metadata_store.list_documents_by_batch_id(
                kb_id, job.batch_id
            )
        return []

    @staticmethod
    def _cleanup_saved_sources(saved_paths: list[Path], saved_dirs: list[Path]) -> None:
        for path in saved_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        for directory in saved_dirs:
            try:
                directory.rmdir()
            except OSError:
                pass


def build_text_source(
    *, text: str, source_name: str | None = None, metadata: dict[str, Any] | None = None
) -> DocumentSourceInput:
    normalized_text = text.strip()
    if not normalized_text:
        raise ValueError("Text document cannot be empty")
    name = (
        source_name or f"text_{compute_mdhash_id(normalized_text, prefix='')[:12]}.txt"
    )
    return DocumentSourceInput(
        source_name=name,
        content=normalized_text.encode("utf-8"),
        source_type="text",
        content_type="text/plain; charset=utf-8",
        metadata=metadata or {},
    )


def _content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _idempotency_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _sanitize_source_name(source_name: str) -> str:
    clean_name = source_name.replace("..", "")
    clean_name = "".join(_sanitize_filename_char(char) for char in clean_name)
    clean_name = clean_name.strip().strip(".")
    if not clean_name:
        raise ValueError("Invalid document source name")
    return clean_name


def _write_source_file(
    workspace_dir: Path, document_id: str, filename: str, content: bytes
) -> Path:
    document_dir = workspace_dir / document_id
    document_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    target_path = document_dir / filename
    try:
        resolved_workspace = workspace_dir.resolve()
        resolved_target = target_path.resolve(strict=False)
        if not resolved_target.is_relative_to(resolved_workspace):
            raise ValueError("Document source path escapes workspace directory")
    except OSError as exc:
        raise ValueError("Invalid document source path") from exc

    with target_path.open("xb") as output:
        output.write(content)
        output.flush()
    return target_path


def _replacement_source_target(
    document_dir: Path, source_name: str, job_id: str
) -> Path:
    target_path = document_dir / source_name
    try:
        resolved_document_dir = document_dir.resolve(strict=False)
        resolved_target = target_path.resolve(strict=False)
        if not resolved_target.is_relative_to(resolved_document_dir):
            raise ValueError(
                "Document replacement source path escapes document directory"
            )
    except OSError as exc:
        raise ValueError("Invalid document replacement source path") from exc
    if not target_path.exists():
        return target_path
    return document_dir / f"{job_id}_{source_name}"


def _validate_document_cleanup_paths(
    workspace_dir: Path,
    document: DocumentRecord,
    artifacts: list[ArtifactRecord],
    *,
    delete_source_file: bool,
    delete_artifacts: bool,
) -> None:
    if not (delete_source_file or delete_artifacts):
        return
    source_path = _safe_workspace_path(workspace_dir, document.source_uri)
    document_dir = source_path.parent.resolve(strict=False)
    if delete_artifacts:
        for artifact in artifacts:
            _safe_document_path(workspace_dir, document_dir, artifact.uri)


def _resolve_artifact_path(
    source_root: Path, document: DocumentRecord, artifact: ArtifactRecord
) -> tuple[Path, bool]:
    """Return (path, is_directory) after running containment checks."""
    if not artifact.uri:
        raise ValueError("Artifact URI is empty")

    try:
        artifact_path = Path(artifact.uri).resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Artifact file not found: {artifact.id}") from exc
    except OSError as exc:
        raise ValueError("Invalid artifact path") from exc

    allowed_document_dir = (source_root / document.workspace / document.id).resolve(
        strict=False
    )
    if not artifact_path.is_relative_to(allowed_document_dir):
        raise ValueError("Artifact path escapes document directory")
    is_directory = artifact_path.is_dir()
    if not is_directory and not artifact_path.is_file():
        raise ValueError("Artifact is neither a file nor a directory")
    return artifact_path, is_directory


def _safe_workspace_path(workspace_dir: Path, uri: str) -> Path:
    path = Path(uri)
    if not path.is_absolute():
        path = workspace_dir / path
    try:
        resolved = path.resolve(strict=False)
        resolved.relative_to(workspace_dir)
    except ValueError as exc:
        raise ValueError(f"Path is outside KB workspace: {uri}") from exc
    return resolved


def _safe_document_path(workspace_dir: Path, document_dir: Path, uri: str) -> Path:
    path = _safe_workspace_path(workspace_dir, uri)
    try:
        path.resolve(strict=False).relative_to(document_dir)
    except ValueError as exc:
        raise ValueError(f"Path is outside document directory: {uri}") from exc
    return path


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _remove_empty_parents(path: Path, *, stop_at: Path) -> None:
    current = path
    while current != stop_at and current.is_relative_to(stop_at):
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _resolve_downloadable_artifact_path(
    source_root: Path, document: DocumentRecord, artifact: ArtifactRecord
) -> Path:
    if artifact.metadata.get("is_directory"):
        raise ValueError("Directory artifacts cannot be downloaded directly")
    path, is_directory = _resolve_artifact_path(source_root, document, artifact)
    if is_directory:
        raise ValueError("Artifact is not a downloadable file")
    return path


def _artifact_media_type(
    document: DocumentRecord,
    artifact: ArtifactRecord,
    path: Path,
    is_directory: bool = False,
) -> str:
    if is_directory:
        return "application/zip"
    if artifact.artifact_type == "original" and document.content_type:
        return document.content_type
    if path.suffix.lower() == ".jsonl":
        return "application/x-ndjson"
    guessed_type, _encoding = mimetypes.guess_type(path.name)
    return guessed_type or "application/octet-stream"


def _resolve_parse_directives(
    source_path: Path,
    document: DocumentRecord,
    *,
    parser_engine: str | None,
    process_options: str | None,
) -> tuple[str, str]:
    if parser_engine is not None:
        engine = normalize_parser_engine(parser_engine)
    else:
        metadata_engine = document.metadata.get("parser_engine")
        if metadata_engine:
            engine = normalize_parser_engine(metadata_engine)
        else:
            engine, resolved_options = resolve_file_parser_directives(
                source_path, require_external_endpoint=False
            )
            process_options = (
                process_options
                or document.metadata.get("process_options")
                or resolved_options
            )

    if engine == PARSER_ENGINE_LEGACY:
        raise ValueError("KB parse endpoint does not support legacy parser engine")
    if engine not in _PARSEABLE_ENGINES or engine not in SUPPORTED_PARSER_ENGINES:
        raise ValueError(f"Unsupported parser engine: {parser_engine}")
    suffix = parser_suffix(source_path)
    if not parser_engine_supports_suffix(engine, suffix):
        raise ValueError(f"Parser engine '{engine}' does not support .{suffix} files")

    raw_options = (
        process_options
        if process_options is not None
        else document.metadata.get("process_options")
    )
    raw_options_text = "" if raw_options is None else str(raw_options)
    errors = validate_process_options(raw_options_text)
    if errors:
        raise ValueError("; ".join(errors))
    options = sanitize_process_options(raw_options_text)
    return engine, options


def _validate_parse_request_directives(
    *, parser_engine: str | None, process_options: str | None
) -> None:
    if parser_engine is not None:
        engine = normalize_parser_engine(parser_engine)
        if engine == PARSER_ENGINE_LEGACY:
            raise ValueError("KB parse endpoint does not support legacy parser engine")
        if engine not in _PARSEABLE_ENGINES or engine not in SUPPORTED_PARSER_ENGINES:
            raise ValueError(f"Unsupported parser engine: {parser_engine}")
    raw_options_text = "" if process_options is None else str(process_options)
    errors = validate_process_options(raw_options_text)
    if errors:
        raise ValueError("; ".join(errors))


def _batch_parse_failure(
    document_id: str,
    *,
    error_code: str,
    error_message: str,
    existing_job_id: str | None = None,
) -> dict[str, Any]:
    failure: dict[str, Any] = {
        "document_id": document_id,
        "status": "failed",
        "error_code": error_code,
        "error_message": error_message,
    }
    if existing_job_id is not None:
        failure["existing_job_id"] = existing_job_id
    return failure


def _parser_hash(*, engine: str, process_options: str) -> str:
    payload = {
        "schema": "kb-parser-hash-v1",
        "engine": engine,
        "process_options": process_options,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _build_parse_artifacts(
    plan: DocumentParsePlan, parsed_data: dict[str, Any]
) -> list[ArtifactRecord]:
    now = utc_now_iso()
    artifacts = [
        _artifact_record(
            plan,
            artifact_type="original",
            uri=str(plan.source_path),
            path=plan.source_path,
            created_at=now,
            metadata={
                "source_name": plan.document.source_name,
                "content_type": plan.document.content_type,
                "source_hash": plan.document.source_hash,
            },
        )
    ]

    blocks_path_value = parsed_data.get("blocks_path")
    blocks_path = Path(blocks_path_value) if blocks_path_value else None
    sidecar_dir = blocks_path.parent if blocks_path is not None else None
    if sidecar_dir is not None and sidecar_dir.exists():
        artifacts.append(
            _artifact_record(
                plan,
                artifact_type="sidecar",
                uri=str(sidecar_dir),
                path=sidecar_dir,
                created_at=now,
                metadata={
                    "is_directory": True,
                    "blocks_path": str(blocks_path) if blocks_path else None,
                    "parse_engine": plan.parser_engine,
                    "parser_hash": plan.parser_hash,
                },
            )
        )
    if blocks_path is not None and blocks_path.exists():
        artifacts.append(
            _artifact_record(
                plan,
                artifact_type="blocks",
                uri=str(blocks_path),
                path=blocks_path,
                created_at=now,
                metadata={"parse_engine": plan.parser_engine},
            )
        )

    if sidecar_dir is not None:
        raw_dir = _raw_artifact_dir(sidecar_dir, plan.parser_engine)
        if raw_dir is not None and raw_dir.exists():
            artifacts.append(
                _artifact_record(
                    plan,
                    artifact_type="raw_dir",
                    uri=str(raw_dir),
                    path=raw_dir,
                    created_at=now,
                    metadata={
                        "is_directory": True,
                        "parse_engine": plan.parser_engine,
                    },
                )
            )
    return artifacts


def _artifact_record(
    plan: DocumentParsePlan,
    *,
    artifact_type: str,
    uri: str,
    path: Path,
    created_at: str,
    metadata: dict[str, Any],
) -> ArtifactRecord:
    is_file = path.is_file()
    return ArtifactRecord(
        id=generate_track_id(f"artifact_{artifact_type}"),
        kb_id=plan.document.kb_id,
        workspace=plan.document.workspace,
        document_id=plan.document.id,
        artifact_type=artifact_type,
        uri=uri,
        checksum=_file_checksum(path) if is_file else None,
        size_bytes=path.stat().st_size if is_file else None,
        metadata=metadata,
        created_at=created_at,
    )


def _raw_artifact_dir(sidecar_dir: Path, engine: str) -> Path | None:
    if not sidecar_dir.name.endswith(PARSED_DIR_SUFFIX):
        return None
    base = sidecar_dir.name[: -len(PARSED_DIR_SUFFIX)]
    if engine == PARSER_ENGINE_MINERU:
        return sidecar_dir.parent / f"{base}{MINERU_RAW_DIR_SUFFIX}"
    if engine == PARSER_ENGINE_DOCLING:
        return sidecar_dir.parent / f"{base}{DOCLING_RAW_DIR_SUFFIX}"
    return None


def _file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
