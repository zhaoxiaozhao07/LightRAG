from __future__ import annotations

import asyncio
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Optional, cast

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from lightrag.api.config import global_args
from lightrag.api.document_lifecycle_service import (
    DocumentLifecycleService,
    DocumentReplacementSource,
    DocumentSourceInput,
    build_text_source,
)
from lightrag.api.index_build_service import (
    IndexBuildPlan,
    IndexBuildService,
)
from lightrag.api.job_service import JobService
from lightrag.api.kb_service import KnowledgeBaseNotFoundError, utc_now_iso
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry
from lightrag.api.metadata_store import (
    ActiveDocumentBuildJobError,
    ActiveDocumentDeleteJobError,
    ActiveDocumentParseJobError,
    ActiveDocumentReplaceJobError,
    ArtifactRecord,
    DocumentNotParsedError,
    DocumentRecord,
    DuplicateDocumentSourceKeyError,
    IdempotencyKeyConflictError,
    InvalidJobTransitionError,
    JobRecord,
    MetadataRecordNotFoundError,
)
from lightrag.api.routers.document_routes import SUPPORTED_DOCUMENT_EXTENSIONS
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.utils import generate_track_id, logger

_UPLOAD_CHUNK_SIZE = 1024 * 1024
_MAX_KB_UPLOAD_FILES = 32
_MAX_KB_TEXT_DOCUMENTS = 100
_MAX_KB_BATCH_PARSE_DOCUMENTS = 100
_MAX_SYNC_SOURCE_KEY_BYTES = 1024
_MAX_TEXT_DOCUMENT_BYTES = 1024 * 1024
_MAX_TEXT_METADATA_BYTES = 64 * 1024
_MAX_DIRECTORY_ARTIFACT_BYTES = 512 * 1024 * 1024  # 512 MB cap on directory zip


def _stream_directory_as_zip(artifact_file: Any) -> StreamingResponse:
    """Stream a directory artifact as an in-memory zip.

    The zip is built once and held in memory so we can compute the size
    cap before sending; for parsed sidecar / raw_dir directories this is
    typically a few MB. Anything beyond ``_MAX_DIRECTORY_ARTIFACT_BYTES``
    raises ``413 Payload Too Large`` rather than streaming partially.
    """
    root: Path = artifact_file.path
    buffer = io.BytesIO()
    total_uncompressed = 0
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for entry in sorted(root.rglob("*")):
            if entry.is_dir():
                continue
            try:
                relative = entry.relative_to(root)
            except ValueError:
                continue
            try:
                size = entry.stat().st_size
            except OSError:
                continue
            total_uncompressed += size
            if total_uncompressed > _MAX_DIRECTORY_ARTIFACT_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "Directory artifact exceeds maximum download size of "
                        f"{_MAX_DIRECTORY_ARTIFACT_BYTES // (1024 * 1024)}MB"
                    ),
                )
            archive.write(entry, arcname=str(relative).replace("\\", "/"))
    buffer.seek(0)
    headers = {
        "Content-Disposition": f'attachment; filename="{artifact_file.filename}"',
    }
    return StreamingResponse(buffer, media_type="application/zip", headers=headers)


_RESERVED_DOCUMENT_METADATA_KEYS = {
    "artifact_count",
    "auto_index",
    "auto_parse",
    "batch_id",
    "blocks_path",
    "build_skipped",
    "build_skip_reason",
    "build_started_at",
    "current_build_job_id",
    "current_parse_job_id",
    "current_replace_job_id",
    "force_embedding",
    "force_extract",
    "force_rechunk",
    "force_reparse",
    "last_built_at",
    "last_build_job_id",
    "last_failed_build_job_id",
    "last_failed_parse_job_id",
    "last_failed_parser_hash",
    "last_parse_job_id",
    "last_parsed_at",
    "parse_engine",
    "parse_format",
    "parse_stage_skipped",
    "parse_started_at",
    "parser_engine",
    "pending_build_job_id",
    "pending_index_hash",
    "pending_lightrag_doc_id",
    "pending_parse_batch_id",
    "pending_parse_job_id",
    "pending_parser_hash",
    "pending_replace_job_id",
    "process_options",
    "source_key",
    "last_sync_job_id",
    "last_synced_at",
}


class DocumentResponse(BaseModel):
    id: str
    kb_id: str
    workspace: str
    lightrag_doc_id: Optional[str]
    source_type: str
    source_name: str
    source_uri: str
    source_hash: str
    content_type: Optional[str]
    size_bytes: int
    parser_hash: Optional[str]
    index_hash: Optional[str]
    status: str
    enabled: bool
    archived: bool
    chunks_count: Optional[int]
    entity_count: Optional[int]
    relation_count: Optional[int]
    error_code: Optional[str]
    error_message: Optional[str]
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    deleted_at: Optional[str]

    @classmethod
    def from_record(cls, record: DocumentRecord) -> "DocumentResponse":
        return cls(**record.to_dict())


class DocumentBatchResponse(BaseModel):
    job_id: str
    batch_id: str
    documents: list[DocumentResponse]


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int
    limit: int
    offset: int


class ArtifactResponse(BaseModel):
    id: str
    kb_id: str
    workspace: str
    document_id: str
    artifact_type: str
    uri: str
    checksum: Optional[str]
    size_bytes: Optional[int]
    metadata: dict[str, Any]
    created_at: str

    @classmethod
    def from_record(cls, record: ArtifactRecord) -> "ArtifactResponse":
        return cls(**record.to_dict())


class ArtifactListResponse(BaseModel):
    artifacts: list[ArtifactResponse]
    total: int
    limit: int
    offset: int


class ParseDocumentRequest(BaseModel):
    engine: Optional[str] = None
    process_options: Optional[str] = None
    force_reparse: bool = False
    auto_index: bool = False
    idempotency_key: Optional[str] = None


class BatchParseDocumentsRequest(BaseModel):
    document_ids: list[str] = Field(
        min_length=1, max_length=_MAX_KB_BATCH_PARSE_DOCUMENTS
    )
    engine: Optional[str] = None
    process_options: Optional[str] = None
    force_reparse: bool = False
    auto_index: bool = False
    idempotency_key: Optional[str] = None

    @field_validator("document_ids", mode="after")
    @classmethod
    def reject_duplicate_document_ids(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("Duplicate document_ids are not allowed")
        return value


class BuildKGRequest(BaseModel):
    force_rechunk: bool = False
    force_extract: bool = False
    force_embedding: bool = False
    idempotency_key: Optional[str] = None


class BatchBuildKGRequest(BaseModel):
    document_ids: list[str] = Field(
        min_length=1, max_length=_MAX_KB_BATCH_PARSE_DOCUMENTS
    )
    force_rechunk: bool = False
    force_extract: bool = False
    force_embedding: bool = False
    idempotency_key: Optional[str] = None

    @field_validator("document_ids", mode="after")
    @classmethod
    def reject_duplicate_document_ids(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("Duplicate document_ids are not allowed")
        return value


class ReindexRequest(BaseModel):
    force_rechunk: bool = True
    force_extract: bool = True
    force_embedding: bool = True
    idempotency_key: Optional[str] = None


class BatchReindexRequest(BaseModel):
    document_ids: list[str] = Field(
        min_length=1, max_length=_MAX_KB_BATCH_PARSE_DOCUMENTS
    )
    force_rechunk: bool = True
    force_extract: bool = True
    force_embedding: bool = True
    idempotency_key: Optional[str] = None

    @field_validator("document_ids", mode="after")
    @classmethod
    def reject_duplicate_document_ids(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("Duplicate document_ids are not allowed")
        return value


class RebuildKBRequest(BaseModel):
    """Whole-KB conservative rebuild request.

    Enumerates every buildable document in the KB and force-reindexes it.
    ``force_*`` default to ``True`` (full rebuild); callers may relax them to
    let the ``index_hash`` skip path apply per document.
    """

    force_rechunk: bool = True
    force_extract: bool = True
    force_embedding: bool = True
    idempotency_key: Optional[str] = None


class BatchDeleteDocumentsRequest(BaseModel):
    document_ids: list[str] = Field(
        min_length=1, max_length=_MAX_KB_BATCH_PARSE_DOCUMENTS
    )
    delete_source_file: bool = False
    delete_artifacts: bool = False
    delete_llm_cache: bool = False
    idempotency_key: Optional[str] = None

    @field_validator("document_ids", mode="after")
    @classmethod
    def reject_duplicate_document_ids(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("Duplicate document_ids are not allowed")
        return value


class JobCancelResponse(BaseModel):
    job: "JobResponse"
    cancelled: bool


class JobRetryRequest(BaseModel):
    idempotency_key: Optional[str] = None


class PatchDocumentRequest(BaseModel):
    metadata: Optional[dict[str, Any]] = None
    enabled: Optional[bool] = None
    archived: Optional[bool] = None

    @field_validator("metadata", mode="after")
    @classmethod
    def limit_metadata_size(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return value
        size = len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
        if size > _MAX_TEXT_METADATA_BYTES:
            raise ValueError(
                f"Document metadata too large. Maximum size: {_MAX_TEXT_METADATA_BYTES} bytes"
            )
        reserved_keys = sorted(set(value) & _RESERVED_DOCUMENT_METADATA_KEYS)
        if reserved_keys:
            raise ValueError(
                "Document metadata contains reserved key(s): "
                + ", ".join(reserved_keys)
            )
        return value


class TextDocumentRequest(BaseModel):
    text: str = Field(min_length=1)
    source_name: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text", mode="after")
    @classmethod
    def strip_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Text document cannot be empty")
        return stripped

    @field_validator("metadata", mode="after")
    @classmethod
    def limit_metadata_size(cls, value: dict[str, Any]) -> dict[str, Any]:
        size = len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
        if size > _MAX_TEXT_METADATA_BYTES:
            raise ValueError(
                f"Text document metadata too large. Maximum size: {_MAX_TEXT_METADATA_BYTES} bytes"
            )
        return value


class TextDocumentsRequest(BaseModel):
    documents: list[TextDocumentRequest] = Field(
        min_length=1, max_length=_MAX_KB_TEXT_DOCUMENTS
    )
    auto_parse: bool = False
    auto_index: bool = False
    parser_engine: Optional[str] = None
    process_options: Optional[str] = None
    idempotency_key: Optional[str] = None


class JobResponse(BaseModel):
    id: str
    kb_id: str
    workspace: str
    batch_id: Optional[str]
    document_id: Optional[str]
    job_type: str
    status: str
    stage: Optional[str]
    progress: float
    total_items: int
    completed_items: int
    failed_items: int
    idempotency_key: Optional[str]
    config_version_id: Optional[str]
    config_hash: Optional[str]
    retry_count: int
    max_retries: int
    payload: dict[str, Any]
    result: Optional[dict[str, Any]]
    error_code: Optional[str]
    error_message: Optional[str]
    created_at: str
    updated_at: str
    queued_at: Optional[str]
    started_at: Optional[str]
    finished_at: Optional[str]
    cancelled_at: Optional[str]

    @classmethod
    def from_record(cls, record: JobRecord) -> "JobResponse":
        return cls(**record.to_dict())


class JobListResponse(BaseModel):
    jobs: list[JobResponse]
    total: int
    limit: int
    offset: int


def _required_upload_limit() -> int:
    max_upload_size = getattr(global_args, "max_upload_size", None)
    if max_upload_size is None or max_upload_size <= 0:
        raise HTTPException(
            status_code=413,
            detail="KB document uploads require MAX_UPLOAD_SIZE to be a positive byte limit",
        )
    return int(max_upload_size)


def _file_too_large_detail(max_size: int, uploaded_size: int) -> str:
    return (
        "File too large. "
        f"Maximum size: {max_size / 1024 / 1024:.1f}MB, "
        f"uploaded: {uploaded_size / 1024 / 1024:.1f}MB"
    )


def _batch_too_large_detail(max_size: int, uploaded_size: int) -> str:
    return (
        "Upload batch too large. "
        f"Maximum total size: {max_size / 1024 / 1024:.1f}MB, "
        f"uploaded: {uploaded_size / 1024 / 1024:.1f}MB"
    )


def _text_too_large_detail(max_size: int, uploaded_size: int) -> str:
    return (
        "Text document too large. "
        f"Maximum size: {max_size} bytes, uploaded: {uploaded_size} bytes"
    )


def _is_supported_upload_name(filename: str) -> bool:
    return filename.lower().endswith(SUPPORTED_DOCUMENT_EXTENSIONS)


async def _read_upload_content(
    file: UploadFile, *, max_upload_size: int, remaining_batch_bytes: int
) -> bytes:
    if remaining_batch_bytes <= 0:
        raise HTTPException(
            status_code=413,
            detail=_batch_too_large_detail(max_upload_size, max_upload_size + 1),
        )

    file_size = getattr(file, "size", None)
    if file_size is not None:
        if file_size > max_upload_size:
            raise HTTPException(
                status_code=413,
                detail=_file_too_large_detail(max_upload_size, int(file_size)),
            )
        if file_size > remaining_batch_bytes:
            raise HTTPException(
                status_code=413,
                detail=_batch_too_large_detail(
                    max_upload_size,
                    max_upload_size - remaining_batch_bytes + int(file_size),
                ),
            )

    content = bytearray()
    while True:
        chunk = await file.read(_UPLOAD_CHUNK_SIZE)
        if not chunk:
            break
        content.extend(chunk)
        if len(content) > max_upload_size:
            raise HTTPException(
                status_code=413,
                detail=_file_too_large_detail(max_upload_size, len(content)),
            )
        if len(content) > remaining_batch_bytes:
            raise HTTPException(
                status_code=413,
                detail=_batch_too_large_detail(
                    max_upload_size,
                    max_upload_size - remaining_batch_bytes + len(content),
                ),
            )
    return bytes(content)


def _validate_text_document_sizes(documents: list[TextDocumentRequest]) -> None:
    for document in documents:
        text_size = len(document.text.encode("utf-8"))
        if text_size > _MAX_TEXT_DOCUMENT_BYTES:
            raise HTTPException(
                status_code=413,
                detail=_text_too_large_detail(_MAX_TEXT_DOCUMENT_BYTES, text_size),
            )


def _normalize_sync_source_key(value: str) -> str:
    source_key = value.strip()
    if not source_key:
        raise HTTPException(
            status_code=400, detail="source_keys cannot contain empty values"
        )
    if len(source_key.encode("utf-8")) > _MAX_SYNC_SOURCE_KEY_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                "source_key too large. Maximum size: "
                f"{_MAX_SYNC_SOURCE_KEY_BYTES} bytes"
            ),
        )
    return source_key


def _idempotency_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _sync_job_result(
    *,
    batch_id: str,
    total_items: int,
    completed_items: int,
    failed_items: int,
    skipped_items: int,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    if failed_items == 0:
        outcome = "succeeded"
    elif completed_items == 0:
        outcome = "failed"
    else:
        outcome = "partial_failure"
    return {
        "batch_id": batch_id,
        "total_items": total_items,
        "completed_items": completed_items,
        "failed_items": failed_items,
        "skipped_items": skipped_items,
        "summary": {
            "outcome": outcome,
            "requested_items": total_items,
            "completed_items": completed_items,
            "failed_items": failed_items,
            "skipped_items": skipped_items,
        },
        "items": items,
    }


def _sync_failure_message(failed_items: int, total_items: int) -> str:
    if failed_items == total_items:
        return "No documents synced successfully"
    return f"{failed_items} of {total_items} documents failed to sync"


def _parse_plan_payload(plan: Any) -> dict[str, Any]:
    return {
        "document_id": plan.document.id,
        "source_uri": str(plan.source_path),
        "source_hash": plan.document.source_hash,
        "parser_engine": plan.parser_engine,
        "process_options": plan.process_options,
        "parser_hash": plan.parser_hash,
        "lightrag_doc_id": plan.lightrag_doc_id,
    }


async def _execute_parse_plan(
    *,
    document_service: DocumentLifecycleService,
    kb_id: str,
    job_id: str,
    plan: Any,
    rag: Any,
) -> dict[str, Any]:
    try:
        await document_service.mark_parse_running(
            kb_id, plan.document.id, job_id=job_id
        )
        parsed_data = await document_service.run_parse(rag, plan)
        result = await document_service.complete_parse(
            kb_id,
            plan.document.id,
            job_id=job_id,
            plan=plan,
            parsed_data=parsed_data,
        )
        return {
            "document_id": result.document.id,
            "status": "succeeded",
            "parser_hash": result.document.parser_hash,
            "lightrag_doc_id": result.document.lightrag_doc_id,
            "artifact_count": len(result.artifacts),
        }
    except Exception as exc:
        logger.error(
            "Failed to parse document '%s' for KB '%s': %s",
            plan.document.id,
            kb_id,
            exc,
        )
        try:
            await document_service.fail_parse(
                kb_id,
                plan.document.id,
                job_id=job_id,
                plan=plan,
                error_code="parse_failed",
                error_message=str(exc),
            )
        except Exception as transition_exc:
            logger.error(
                "Failed to mark document '%s' failed for parse job '%s': %s",
                plan.document.id,
                job_id,
                transition_exc,
            )
        return {
            "document_id": plan.document.id,
            "status": "failed",
            "error_code": "parse_failed",
            "error_message": str(exc),
        }


def _batch_parse_job_result(
    *,
    batch_id: str,
    total_items: int,
    completed_items: int,
    failed_items: int,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    if failed_items == 0:
        outcome = "succeeded"
    elif completed_items == 0:
        outcome = "failed"
    else:
        outcome = "partial_failure"
    return {
        "batch_id": batch_id,
        "total_items": total_items,
        "completed_items": completed_items,
        "failed_items": failed_items,
        "summary": {
            "outcome": outcome,
            "requested_items": total_items,
            "completed_items": completed_items,
            "failed_items": failed_items,
        },
        "items": items,
    }


def _batch_parse_failure_message(failed_items: int, total_items: int) -> str:
    if failed_items == total_items:
        return "No documents parsed successfully"
    return f"{failed_items} of {total_items} documents failed to parse"


def _build_plan_payload(plan: IndexBuildPlan) -> dict[str, Any]:
    return {
        "document_id": plan.document.id,
        "lightrag_doc_id": plan.document.lightrag_doc_id,
        "parser_hash": plan.parser_hash,
        "index_hash": plan.index_hash,
        "sidecar_uri": plan.sidecar_uri,
        "blocks_path": plan.blocks_path,
        "process_options": plan.process_options,
        "force_rechunk": plan.force_rechunk,
        "force_extract": plan.force_extract,
        "force_embedding": plan.force_embedding,
        "skipped": plan.skipped,
        "skip_reason": plan.skip_reason,
    }


async def _execute_build_plan(
    *,
    index_service: IndexBuildService,
    kb_id: str,
    job_id: str,
    plan: IndexBuildPlan,
    rag: Any,
) -> dict[str, Any]:
    try:
        if not plan.skipped:
            await index_service.mark_building(kb_id, plan.document.id, job_id=job_id)
        run_result = await index_service.run_build(rag, plan)
        result = await index_service.complete_build(
            kb_id,
            plan.document.id,
            job_id=job_id,
            plan=plan,
            run_result=run_result,
        )
        return {
            "document_id": result.id,
            "status": "succeeded",
            "skipped": bool(plan.skipped or run_result.get("skipped")),
            "skip_reason": plan.skip_reason if plan.skipped else None,
            "index_hash": plan.index_hash,
            "chunks_count": result.chunks_count,
            "entity_count": result.entity_count,
            "relation_count": result.relation_count,
        }
    except Exception as exc:  # noqa: BLE001 — surface and persist
        logger.error(
            "Failed to build KG for document '%s' (KB '%s'): %s",
            plan.document.id,
            kb_id,
            exc,
        )
        try:
            await index_service.fail_build(
                kb_id,
                plan.document.id,
                job_id=job_id,
                error_code="build_failed",
                error_message=str(exc),
            )
        except Exception as transition_exc:
            logger.error(
                "Failed to mark build job '%s' failed: %s",
                job_id,
                transition_exc,
            )
        return {
            "document_id": plan.document.id,
            "status": "failed",
            "error_code": "build_failed",
            "error_message": str(exc),
        }


def _batch_build_job_result(
    *,
    batch_id: str,
    total_items: int,
    completed_items: int,
    failed_items: int,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    if failed_items == 0:
        outcome = "succeeded"
    elif completed_items == 0:
        outcome = "failed"
    else:
        outcome = "partial_failure"
    return {
        "batch_id": batch_id,
        "total_items": total_items,
        "completed_items": completed_items,
        "failed_items": failed_items,
        "summary": {
            "outcome": outcome,
            "requested_items": total_items,
            "completed_items": completed_items,
            "failed_items": failed_items,
        },
        "items": items,
    }


def _batch_build_failure_message(failed_items: int, total_items: int) -> str:
    if failed_items == total_items:
        return "No documents indexed successfully"
    return f"{failed_items} of {total_items} documents failed to build"


def _deletion_result_payload(result: Any) -> dict[str, Any]:
    if result is None:
        return {"status": "skipped", "message": "Document was not indexed"}
    return {
        "status": getattr(result, "status", None),
        "doc_id": getattr(result, "doc_id", None),
        "message": getattr(result, "message", None),
        "status_code": getattr(result, "status_code", None),
        "file_path": getattr(result, "file_path", None),
    }


def _file_result_payload(result: Any) -> dict[str, Any]:
    return {
        "deleted_source": getattr(result, "deleted_source", False),
        "deleted_artifacts": list(getattr(result, "deleted_artifacts", [])),
        "skipped": list(getattr(result, "skipped", [])),
        "errors": list(getattr(result, "errors", [])),
    }


def _delete_job_result(
    *,
    total_items: int,
    completed_items: int,
    failed_items: int,
    items: list[dict[str, Any]],
    batch_id: str | None = None,
) -> dict[str, Any]:
    if failed_items == 0:
        outcome = "succeeded"
    elif completed_items == 0:
        outcome = "failed"
    else:
        outcome = "partial_failure"
    result: dict[str, Any] = {
        "total_items": total_items,
        "completed_items": completed_items,
        "failed_items": failed_items,
        "summary": {
            "outcome": outcome,
            "requested_items": total_items,
            "completed_items": completed_items,
            "failed_items": failed_items,
        },
        "items": items,
    }
    if batch_id is not None:
        result["batch_id"] = batch_id
    return result


def _delete_failure_message(failed_items: int, total_items: int) -> str:
    if failed_items == total_items:
        return "No documents deleted successfully"
    return f"{failed_items} of {total_items} documents failed to delete"


def _active_job_error_code(
    exc: ActiveDocumentParseJobError
    | ActiveDocumentBuildJobError
    | ActiveDocumentDeleteJobError
    | ActiveDocumentReplaceJobError,
) -> str:
    if isinstance(exc, ActiveDocumentParseJobError):
        return "parse_job_active"
    if isinstance(exc, ActiveDocumentBuildJobError):
        return "build_job_active"
    if isinstance(exc, ActiveDocumentDeleteJobError):
        return "delete_job_active"
    return "replace_job_active"


def _active_job_conflict_detail(
    exc: ActiveDocumentParseJobError
    | ActiveDocumentBuildJobError
    | ActiveDocumentDeleteJobError
    | ActiveDocumentReplaceJobError,
) -> dict[str, Any]:
    return {
        "error_code": _active_job_error_code(exc),
        "document_id": exc.document_id,
        "existing_job_id": exc.existing_job_id,
        "message": str(exc),
    }


def create_kb_document_routes(
    document_service: DocumentLifecycleService,
    job_service: JobService,
    api_key: Optional[str] = None,
    registry: LightRAGInstanceRegistry | None = None,
    index_service: IndexBuildService | None = None,
):
    router = APIRouter(prefix="/kbs", tags=["knowledge-base-documents"])
    combined_auth = get_combined_auth_dependency(api_key)

    @router.post(
        "/{kb_id}/documents:upload",
        response_model=DocumentBatchResponse,
        dependencies=[Depends(combined_auth)],
        summary="Upload documents to a knowledge base metadata stage",
    )
    async def upload_documents(
        kb_id: str,
        files: list[UploadFile] = File(...),
        auto_parse: bool = False,
        auto_index: bool = False,
        parser_engine: Optional[str] = None,
        process_options: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ):
        try:
            if len(files) > _MAX_KB_UPLOAD_FILES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Too many files. Maximum files per request: {_MAX_KB_UPLOAD_FILES}",
                )
            max_upload_size = _required_upload_limit()
            total_bytes = 0
            sources: list[DocumentSourceInput] = []
            for file in files:
                source_name = file.filename or "uploaded_document"
                if not _is_supported_upload_name(source_name):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Unsupported file type. Supported types: "
                            f"{SUPPORTED_DOCUMENT_EXTENSIONS}"
                        ),
                    )
                content = await _read_upload_content(
                    file,
                    max_upload_size=max_upload_size,
                    remaining_batch_bytes=max_upload_size - total_bytes,
                )
                total_bytes += len(content)
                sources.append(
                    DocumentSourceInput(
                        source_name=source_name,
                        content=content,
                        source_type="upload",
                        content_type=file.content_type,
                        metadata={},
                    )
                )
            result = await document_service.create_source_batch(
                kb_id,
                sources,
                auto_parse=auto_parse,
                auto_index=auto_index,
                parser_engine=parser_engine,
                process_options=process_options,
                idempotency_key=idempotency_key,
            )
            return DocumentBatchResponse(
                job_id=result.job.id,
                batch_id=result.batch_id,
                documents=[
                    DocumentResponse.from_record(item) for item in result.documents
                ],
            )
        except HTTPException:
            raise
        except IdempotencyKeyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except DuplicateDocumentSourceKeyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("Failed to upload documents for KB '%s': %s", kb_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post(
        "/{kb_id}/documents:texts",
        response_model=DocumentBatchResponse,
        dependencies=[Depends(combined_auth)],
        summary="Import text documents to a knowledge base metadata stage",
    )
    async def import_text_documents(kb_id: str, request: TextDocumentsRequest):
        try:
            _validate_text_document_sizes(request.documents)
            sources = [
                build_text_source(
                    text=document.text,
                    source_name=document.source_name,
                    metadata=document.metadata,
                )
                for document in request.documents
            ]
            result = await document_service.create_source_batch(
                kb_id,
                sources,
                auto_parse=request.auto_parse,
                auto_index=request.auto_index,
                parser_engine=request.parser_engine,
                process_options=request.process_options,
                idempotency_key=request.idempotency_key,
            )
            return DocumentBatchResponse(
                job_id=result.job.id,
                batch_id=result.batch_id,
                documents=[
                    DocumentResponse.from_record(item) for item in result.documents
                ],
            )
        except HTTPException:
            raise
        except IdempotencyKeyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except DuplicateDocumentSourceKeyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("Failed to import texts for KB '%s': %s", kb_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.get(
        "/{kb_id}/documents",
        response_model=DocumentListResponse,
        dependencies=[Depends(combined_auth)],
        summary="List knowledge base documents",
    )
    async def list_documents(
        kb_id: str,
        status: Optional[str] = None,
        source_name: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ):
        try:
            documents, total = await document_service.list_documents(
                kb_id,
                status=status,
                source_name=source_name,
                limit=limit,
                offset=offset,
            )
            return DocumentListResponse(
                documents=[DocumentResponse.from_record(item) for item in documents],
                total=total,
                limit=max(1, min(limit, 200)),
                offset=max(0, offset),
            )
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get(
        "/{kb_id}/documents/{document_id}",
        response_model=DocumentResponse,
        dependencies=[Depends(combined_auth)],
        summary="Get knowledge base document details",
    )
    async def get_document(kb_id: str, document_id: str):
        try:
            return DocumentResponse.from_record(
                await document_service.get_document(kb_id, document_id)
            )
        except (KnowledgeBaseNotFoundError, MetadataRecordNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.patch(
        "/{kb_id}/documents/{document_id}",
        response_model=DocumentResponse,
        dependencies=[Depends(combined_auth)],
        summary="Patch knowledge base document metadata",
    )
    async def patch_document(
        kb_id: str, document_id: str, request: PatchDocumentRequest
    ):
        try:
            if not request.model_fields_set:
                raise HTTPException(
                    status_code=400,
                    detail="At least one document field must be provided",
                )
            if "metadata" in request.model_fields_set and request.metadata is None:
                raise HTTPException(
                    status_code=400, detail="metadata must be an object"
                )
            return DocumentResponse.from_record(
                await document_service.update_document(
                    kb_id,
                    document_id,
                    metadata_patch=request.metadata
                    if "metadata" in request.model_fields_set
                    else None,
                    enabled=request.enabled
                    if "enabled" in request.model_fields_set
                    else None,
                    archived=request.archived
                    if "archived" in request.model_fields_set
                    else None,
                )
            )
        except HTTPException:
            raise
        except (KnowledgeBaseNotFoundError, MetadataRecordNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post(
        "/{kb_id}/documents/{document_id}:disable",
        response_model=DocumentResponse,
        dependencies=[Depends(combined_auth)],
        summary="Disable one knowledge base document",
    )
    async def disable_document(kb_id: str, document_id: str):
        try:
            return DocumentResponse.from_record(
                await document_service.update_document(
                    kb_id, document_id, enabled=False
                )
            )
        except (KnowledgeBaseNotFoundError, MetadataRecordNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post(
        "/{kb_id}/documents/{document_id}:enable",
        response_model=DocumentResponse,
        dependencies=[Depends(combined_auth)],
        summary="Enable one knowledge base document",
    )
    async def enable_document(kb_id: str, document_id: str):
        try:
            return DocumentResponse.from_record(
                await document_service.update_document(kb_id, document_id, enabled=True)
            )
        except (KnowledgeBaseNotFoundError, MetadataRecordNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def _execute_replace_document(
        *,
        kb_id: str,
        job: JobRecord,
        document: DocumentRecord,
        replacement: DocumentReplacementSource,
        active_registry: LightRAGInstanceRegistry,
        active_index_service: IndexBuildService | None,
        delete_source_file: bool,
        delete_artifacts: bool,
        delete_llm_cache: bool,
        auto_parse: bool,
        auto_index: bool,
        parser_engine: str | None,
        process_options: str | None,
        force_reparse: bool,
    ) -> dict[str, Any]:
        replace_completed = False
        old_index_deleted = False
        lightrag_result = None
        try:
            rag: Any | None = None
            await document_service.preflight_replace_cleanup(
                kb_id,
                document,
                delete_source_file=delete_source_file,
                delete_artifacts=delete_artifacts,
            )
            if document.lightrag_doc_id:
                rag_for_delete = cast(Any, await active_registry.get(kb_id))
                if rag_for_delete is None:
                    raise RuntimeError(f"LightRAG instance unavailable for KB {kb_id}")
                rag = rag_for_delete
                lightrag_result = await rag_for_delete.adelete_by_doc_id(
                    document.lightrag_doc_id,
                    delete_llm_cache=delete_llm_cache,
                )
                if getattr(lightrag_result, "status", None) not in {
                    "success",
                    "not_found",
                }:
                    raise RuntimeError(
                        getattr(lightrag_result, "message", None)
                        or f"LightRAG deletion failed for {document.lightrag_doc_id}"
                    )
                old_index_deleted = True

            (
                replaced_document,
                file_result,
            ) = await document_service.replace_document_source(
                kb_id,
                document,
                job_id=job.id,
                replacement=replacement,
                delete_source_file=delete_source_file,
                delete_artifacts=delete_artifacts,
                lightrag_delete_result=_deletion_result_payload(lightrag_result),
            )
            replace_completed = True
            item: dict[str, Any] = {
                "document_id": replaced_document.id,
                "status": "succeeded",
                "source_name": replaced_document.source_name,
                "source_uri": replaced_document.source_uri,
                "source_hash": replaced_document.source_hash,
                "previous_lightrag_doc_id": document.lightrag_doc_id,
                "lightrag_delete_result": _deletion_result_payload(lightrag_result),
                "file_replace_result": _file_result_payload(file_result),
            }

            if auto_parse:
                if rag is None:
                    rag = cast(Any, await active_registry.get(kb_id))
                parse_plan = await document_service.create_parse_plan(
                    kb_id,
                    replaced_document.id,
                    parser_engine=parser_engine,
                    process_options=process_options,
                    force_reparse=force_reparse,
                    auto_index=auto_index,
                )
                await document_service.mark_parse_queued(
                    kb_id,
                    replaced_document.id,
                    job=job,
                    plan=parse_plan,
                )
                parse_item = await _execute_parse_plan(
                    document_service=document_service,
                    kb_id=kb_id,
                    job_id=job.id,
                    plan=parse_plan,
                    rag=rag,
                )
                item["parse_result"] = parse_item
                if parse_item["status"] != "succeeded":
                    item.update(
                        {
                            "status": "failed",
                            "error_code": parse_item.get("error_code", "parse_failed"),
                            "error_message": parse_item.get(
                                "error_message", "Replacement parse failed"
                            ),
                        }
                    )
                    return item

                if auto_index:
                    if active_index_service is None:
                        raise RuntimeError("KB index build service is not configured")
                    build_plan = await active_index_service.create_build_plan(
                        kb_id,
                        replaced_document.id,
                        rag=rag,
                    )
                    await active_index_service.claim_build_queued(
                        kb_id, job_id=job.id, plan=build_plan
                    )
                    build_item = await _execute_build_plan(
                        index_service=active_index_service,
                        kb_id=kb_id,
                        job_id=job.id,
                        plan=build_plan,
                        rag=rag,
                    )
                    item["build_result"] = build_item
                    if build_item["status"] != "succeeded":
                        item.update(
                            {
                                "status": "failed",
                                "error_code": build_item.get(
                                    "error_code", "build_failed"
                                ),
                                "error_message": build_item.get(
                                    "error_message", "Replacement build failed"
                                ),
                            }
                        )
            return item
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to replace document '%s' for KB '%s': %s",
                document.id,
                kb_id,
                exc,
            )
            if not replace_completed:
                try:
                    await document_service.fail_replace(
                        kb_id,
                        document.id,
                        job_id=job.id,
                        error_code="replace_failed",
                        error_message=str(exc),
                        clear_index_metadata=old_index_deleted,
                        lightrag_delete_result=_deletion_result_payload(
                            lightrag_result
                        ),
                    )
                except Exception as transition_exc:
                    logger.error(
                        "Failed to mark document '%s' failed for replace job '%s': %s",
                        document.id,
                        job.id,
                        transition_exc,
                    )
            return {
                "document_id": document.id,
                "status": "failed",
                "error_code": (
                    "replace_failed"
                    if not replace_completed
                    else "replace_followup_failed"
                ),
                "error_message": str(exc),
            }

    async def _run_sync_followups(
        *,
        kb_id: str,
        job: JobRecord,
        document: DocumentRecord,
        item: dict[str, Any],
        active_registry: LightRAGInstanceRegistry,
        active_index_service: IndexBuildService | None,
        rag: Any | None,
        auto_parse: bool,
        auto_index: bool,
        parser_engine: str | None,
        process_options: str | None,
        force_reparse: bool,
    ) -> tuple[dict[str, Any], Any | None]:
        if not auto_parse:
            return item, rag

        parse_plan = await document_service.create_parse_plan(
            kb_id,
            document.id,
            parser_engine=parser_engine,
            process_options=process_options,
            force_reparse=force_reparse,
            auto_index=auto_index,
        )
        status_requires_parse = document.status in {
            "uploaded",
            "parse_queued",
            "parsing",
            "parse_failed",
            "replace_failed",
        }
        parse_needed = (
            force_reparse
            or document.parser_hash != parse_plan.parser_hash
            or status_requires_parse
        )
        if parse_needed:
            if item.get("action") == "skipped":
                item["action"] = "reparsed"
                if document.parser_hash != parse_plan.parser_hash:
                    item["skip_reason"] = "parser_hash_changed"
                elif force_reparse:
                    item["skip_reason"] = "force_reparse"
            if rag is None:
                rag = cast(Any, await active_registry.get(kb_id))
            await document_service.mark_parse_queued(
                kb_id,
                document.id,
                job=job,
                plan=parse_plan,
            )
            parse_item = await _execute_parse_plan(
                document_service=document_service,
                kb_id=kb_id,
                job_id=job.id,
                plan=parse_plan,
                rag=rag,
            )
            item["parse_result"] = parse_item
            if parse_item["status"] != "succeeded":
                item.update(
                    {
                        "status": "failed",
                        "error_code": parse_item.get("error_code", "parse_failed"),
                        "error_message": parse_item.get(
                            "error_message", "Document sync parse failed"
                        ),
                    }
                )
                return item, rag
            item["status"] = "succeeded"
            item.pop("skip_reason", None)

        if auto_index:
            if active_index_service is None:
                raise RuntimeError("KB index build service is not configured")
            if rag is None:
                rag = cast(Any, await active_registry.get(kb_id))
            build_plan = await active_index_service.create_build_plan(
                kb_id,
                document.id,
                rag=rag,
            )
            await active_index_service.claim_build_queued(
                kb_id,
                job_id=job.id,
                plan=build_plan,
            )
            build_item = await _execute_build_plan(
                index_service=active_index_service,
                kb_id=kb_id,
                job_id=job.id,
                plan=build_plan,
                rag=rag,
            )
            item["build_result"] = build_item
            if build_item["status"] != "succeeded":
                item.update(
                    {
                        "status": "failed",
                        "error_code": build_item.get("error_code", "build_failed"),
                        "error_message": build_item.get(
                            "error_message", "Document sync build failed"
                        ),
                    }
                )
                return item, rag
            if not build_item.get("skipped"):
                item["status"] = "succeeded"
                item.pop("skip_reason", None)
        return item, rag

    async def _execute_sync_item(
        *,
        kb_id: str,
        job: JobRecord,
        prepared: dict[str, Any],
        existing_by_source_key: dict[str, DocumentRecord],
        active_registry: LightRAGInstanceRegistry,
        active_index_service: IndexBuildService | None,
        rag: Any | None,
        auto_parse: bool,
        auto_index: bool,
        parser_engine: str | None,
        process_options: str | None,
        force_reparse: bool,
        delete_source_file: bool,
        delete_artifacts: bool,
        delete_llm_cache: bool,
    ) -> tuple[dict[str, Any], Any | None]:
        source_key = str(prepared["source_key"])
        source = cast(DocumentSourceInput, prepared["source"])
        source_hash = str(prepared["source_hash"])
        item: dict[str, Any] = {
            "source_key": source_key,
            "source_name": source.source_name,
            "source_hash": source_hash,
        }
        try:
            existing = existing_by_source_key.get(source_key)
            if existing is None:
                create_result = await document_service.create_source_batch(
                    kb_id,
                    [source],
                    auto_parse=False,
                    auto_index=False,
                )
                document = create_result.documents[0]
                item.update(
                    {
                        "action": "created",
                        "status": "succeeded",
                        "document_id": document.id,
                        "upload_job_id": create_result.job.id,
                    }
                )
                item, rag = await _run_sync_followups(
                    kb_id=kb_id,
                    job=job,
                    document=document,
                    item=item,
                    active_registry=active_registry,
                    active_index_service=active_index_service,
                    rag=rag,
                    auto_parse=auto_parse,
                    auto_index=auto_index,
                    parser_engine=parser_engine,
                    process_options=process_options,
                    force_reparse=force_reparse,
                )
            elif existing.source_hash == source_hash:
                item.update(
                    {
                        "action": "skipped",
                        "status": "skipped",
                        "skip_reason": "source_hash_match",
                        "document_id": existing.id,
                    }
                )
                item, rag = await _run_sync_followups(
                    kb_id=kb_id,
                    job=job,
                    document=existing,
                    item=item,
                    active_registry=active_registry,
                    active_index_service=active_index_service,
                    rag=rag,
                    auto_parse=auto_parse,
                    auto_index=auto_index,
                    parser_engine=parser_engine,
                    process_options=process_options,
                    force_reparse=force_reparse,
                )
            else:
                replacement = document_service.prepare_replacement_source(source)
                claimed = await document_service.claim_replace(
                    kb_id,
                    existing.id,
                    job=job,
                    replacement=replacement,
                    delete_source_file=delete_source_file,
                    delete_artifacts=delete_artifacts,
                    delete_llm_cache=delete_llm_cache,
                    auto_parse=auto_parse,
                    auto_index=auto_index,
                    parser_engine=parser_engine,
                    process_options=process_options,
                    force_reparse=force_reparse,
                )
                replace_item = await _execute_replace_document(
                    kb_id=kb_id,
                    job=job,
                    document=claimed,
                    replacement=replacement,
                    active_registry=active_registry,
                    active_index_service=active_index_service,
                    delete_source_file=delete_source_file,
                    delete_artifacts=delete_artifacts,
                    delete_llm_cache=delete_llm_cache,
                    auto_parse=auto_parse,
                    auto_index=auto_index,
                    parser_engine=parser_engine,
                    process_options=process_options,
                    force_reparse=force_reparse,
                )
                item.update(replace_item)
                item["action"] = "replaced"

            document_id = item.get("document_id")
            if item["status"] in {"succeeded", "skipped"} and isinstance(
                document_id, str
            ):
                await document_service.update_document(
                    kb_id,
                    document_id,
                    metadata_patch={
                        "source_key": source_key,
                        "last_sync_job_id": job.id,
                        "last_synced_at": utc_now_iso(),
                    },
                )
            return item, rag
        except (
            ActiveDocumentParseJobError,
            ActiveDocumentBuildJobError,
            ActiveDocumentDeleteJobError,
            ActiveDocumentReplaceJobError,
        ) as exc:
            item.update(
                {
                    "action": item.get("action", "unknown"),
                    "status": "failed",
                    **_active_job_conflict_detail(exc),
                }
            )
            return item, rag
        except DuplicateDocumentSourceKeyError as exc:
            item.update(
                {
                    "action": item.get("action", "unknown"),
                    "status": "failed",
                    "error_code": "source_key_conflict",
                    "error_message": str(exc),
                    "existing_document_id": exc.existing_document_id,
                }
            )
            return item, rag
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to sync source_key '%s' for KB '%s': %s",
                source_key,
                kb_id,
                exc,
            )
            item.update(
                {
                    "action": item.get("action", "unknown"),
                    "status": "failed",
                    "error_code": "sync_item_failed",
                    "error_message": str(exc),
                }
            )
            return item, rag

    @router.post(
        "/{kb_id}/documents:sync",
        response_model=JobResponse,
        dependencies=[Depends(combined_auth)],
        summary="Synchronize a batch of knowledge base documents by source key",
    )
    async def sync_documents(
        kb_id: str,
        background_tasks: BackgroundTasks,
        files: list[UploadFile] = File(...),
        source_keys: list[str] = Form(...),
        auto_parse: bool = True,
        auto_index: bool = True,
        parser_engine: Optional[str] = None,
        process_options: Optional[str] = None,
        force_reparse: bool = False,
        delete_source_file: bool = True,
        delete_artifacts: bool = True,
        delete_llm_cache: bool = False,
        idempotency_key: Optional[str] = None,
    ):
        if auto_index and not auto_parse:
            raise HTTPException(
                status_code=400,
                detail="auto_index requires auto_parse for document sync",
            )
        if auto_parse and registry is None:
            raise HTTPException(
                status_code=503, detail="KB sync service is not configured"
            )
        if auto_index and index_service is None:
            raise HTTPException(
                status_code=503,
                detail="KB index build service is not configured",
            )
        if len(files) > _MAX_KB_UPLOAD_FILES:
            raise HTTPException(
                status_code=413,
                detail=f"Too many files. Maximum files per request: {_MAX_KB_UPLOAD_FILES}",
            )
        if len(files) != len(source_keys):
            raise HTTPException(
                status_code=400,
                detail="files and source_keys must contain the same number of items",
            )

        active_registry = registry
        active_index_service = index_service
        try:
            normalized_keys = [_normalize_sync_source_key(item) for item in source_keys]
            if len(set(normalized_keys)) != len(normalized_keys):
                raise HTTPException(
                    status_code=400, detail="Duplicate source_keys are not allowed"
                )

            max_upload_size = _required_upload_limit()
            total_bytes = 0
            prepared_sources: list[dict[str, Any]] = []
            for file, source_key in zip(files, normalized_keys, strict=True):
                source_name = file.filename or "uploaded_document"
                if not _is_supported_upload_name(source_name):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Unsupported file type. Supported types: "
                            f"{SUPPORTED_DOCUMENT_EXTENSIONS}"
                        ),
                    )
                content = await _read_upload_content(
                    file,
                    max_upload_size=max_upload_size,
                    remaining_batch_bytes=max_upload_size - total_bytes,
                )
                total_bytes += len(content)
                source_hash = hashlib.sha256(content).hexdigest()
                source = DocumentSourceInput(
                    source_name=source_name,
                    content=content,
                    source_type="upload",
                    content_type=file.content_type,
                    metadata={
                        "source_key": source_key,
                    },
                )
                prepared_sources.append(
                    {
                        "source_key": source_key,
                        "source": source,
                        "source_hash": source_hash,
                        "content_type": file.content_type,
                        "size_bytes": len(content),
                    }
                )

            batch_id = generate_track_id("batch")
            fingerprint_payload = {
                "items": [
                    {
                        "source_key": item["source_key"],
                        "source_name": cast(
                            DocumentSourceInput, item["source"]
                        ).source_name,
                        "source_hash": item["source_hash"],
                        "content_type": item["content_type"],
                        "size_bytes": item["size_bytes"],
                    }
                    for item in prepared_sources
                ],
                "auto_parse": auto_parse,
                "auto_index": auto_index,
                "parser_engine": parser_engine,
                "process_options": process_options,
                "force_reparse": force_reparse,
                "delete_source_file": delete_source_file,
                "delete_artifacts": delete_artifacts,
                "delete_llm_cache": delete_llm_cache,
            }
            payload = {
                **fingerprint_payload,
                "source_keys": normalized_keys,
                "idempotency_fingerprint": _idempotency_fingerprint(
                    fingerprint_payload
                ),
            }
            job, created_job = await job_service.create_job_once(
                kb_id,
                job_type="sync",
                batch_id=batch_id,
                stage="syncing",
                total_items=len(prepared_sources),
                payload=payload,
                idempotency_key=idempotency_key,
            )
            if not created_job:
                return JobResponse.from_record(job)

            async def _sync_task() -> None:
                item_results: list[dict[str, Any]] = []
                completed_items = 0
                failed_items = 0
                skipped_items = 0
                rag: Any | None = None
                try:
                    await job_service.transition_job(
                        kb_id,
                        job.id,
                        status="running",
                        progress=0.0,
                        result=_sync_job_result(
                            batch_id=job.batch_id or batch_id,
                            total_items=len(prepared_sources),
                            completed_items=completed_items,
                            failed_items=failed_items,
                            skipped_items=skipped_items,
                            items=item_results,
                        ),
                    )
                    existing_by_source_key = (
                        await document_service.get_documents_by_source_keys(
                            kb_id, normalized_keys
                        )
                    )
                    for prepared in prepared_sources:
                        item, rag = await _execute_sync_item(
                            kb_id=kb_id,
                            job=job,
                            prepared=prepared,
                            existing_by_source_key=existing_by_source_key,
                            active_registry=cast(
                                LightRAGInstanceRegistry, active_registry
                            ),
                            active_index_service=active_index_service,
                            rag=rag,
                            auto_parse=auto_parse,
                            auto_index=auto_index,
                            parser_engine=parser_engine,
                            process_options=process_options,
                            force_reparse=force_reparse,
                            delete_source_file=delete_source_file,
                            delete_artifacts=delete_artifacts,
                            delete_llm_cache=delete_llm_cache,
                        )
                        item_results.append(item)
                        if item["status"] == "failed":
                            failed_items += 1
                        else:
                            completed_items += 1
                            if item["status"] == "skipped":
                                skipped_items += 1

                    final_result = _sync_job_result(
                        batch_id=job.batch_id or batch_id,
                        total_items=len(prepared_sources),
                        completed_items=completed_items,
                        failed_items=failed_items,
                        skipped_items=skipped_items,
                        items=item_results,
                    )
                    final_status = "succeeded" if failed_items == 0 else "failed"
                    await job_service.transition_job(
                        kb_id,
                        job.id,
                        status=final_status,
                        progress=1.0,
                        completed_items=completed_items,
                        failed_items=failed_items,
                        result=final_result,
                        error_code=None if failed_items == 0 else "partial_sync_failed",
                        error_message=None
                        if failed_items == 0
                        else _sync_failure_message(failed_items, len(prepared_sources)),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Failed to run sync job '%s' for KB '%s': %s",
                        job.id,
                        kb_id,
                        exc,
                    )
                    processed_keys = {item["source_key"] for item in item_results}
                    for prepared in prepared_sources:
                        source_key = str(prepared["source_key"])
                        if source_key in processed_keys:
                            continue
                        item_results.append(
                            {
                                "source_key": source_key,
                                "source_name": cast(
                                    DocumentSourceInput, prepared["source"]
                                ).source_name,
                                "source_hash": prepared["source_hash"],
                                "action": "unknown",
                                "status": "failed",
                                "error_code": "sync_failed",
                                "error_message": str(exc),
                            }
                        )
                    failed_items = len(
                        [item for item in item_results if item["status"] == "failed"]
                    )
                    completed_items = len(item_results) - failed_items
                    skipped_items = len(
                        [item for item in item_results if item["status"] == "skipped"]
                    )
                    await job_service.transition_job(
                        kb_id,
                        job.id,
                        status="failed",
                        progress=1.0,
                        completed_items=completed_items,
                        failed_items=failed_items,
                        result=_sync_job_result(
                            batch_id=job.batch_id or batch_id,
                            total_items=len(prepared_sources),
                            completed_items=completed_items,
                            failed_items=failed_items,
                            skipped_items=skipped_items,
                            items=item_results,
                        ),
                        error_code="sync_failed",
                        error_message=str(exc),
                    )

            background_tasks.add_task(_sync_task)
            return JobResponse.from_record(job)
        except HTTPException:
            raise
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except IdempotencyKeyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("Failed to start document sync for KB '%s': %s", kb_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post(
        "/{kb_id}/documents/{document_id}:replace",
        response_model=JobResponse,
        dependencies=[Depends(combined_auth)],
        summary="Replace one knowledge base document source",
    )
    async def replace_document(
        kb_id: str,
        document_id: str,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        auto_parse: bool = False,
        auto_index: bool = False,
        parser_engine: Optional[str] = None,
        process_options: Optional[str] = None,
        force_reparse: bool = False,
        delete_source_file: bool = True,
        delete_artifacts: bool = True,
        delete_llm_cache: bool = False,
        idempotency_key: Optional[str] = None,
    ):
        if registry is None:
            raise HTTPException(
                status_code=503, detail="KB replace service is not configured"
            )
        if auto_index and not auto_parse:
            raise HTTPException(
                status_code=400,
                detail="auto_index requires auto_parse for document replacement",
            )
        if auto_index and index_service is None:
            raise HTTPException(
                status_code=503,
                detail="KB index build service is not configured",
            )
        active_registry = registry
        active_index_service = index_service
        try:
            source_name = file.filename or "uploaded_document"
            if not _is_supported_upload_name(source_name):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Unsupported file type. Supported types: "
                        f"{SUPPORTED_DOCUMENT_EXTENSIONS}"
                    ),
                )
            max_upload_size = _required_upload_limit()
            content = await _read_upload_content(
                file,
                max_upload_size=max_upload_size,
                remaining_batch_bytes=max_upload_size,
            )
            replacement = document_service.prepare_replacement_source(
                DocumentSourceInput(
                    source_name=source_name,
                    content=content,
                    source_type="upload",
                    content_type=file.content_type,
                    metadata={},
                )
            )
            document = await document_service.get_document(kb_id, document_id)
            job, created_job = await job_service.create_replace_job_once(
                kb_id,
                document_id=document_id,
                previous_lightrag_doc_id=document.lightrag_doc_id,
                source_name=replacement.source_name,
                source_hash=replacement.source_hash,
                content_type=replacement.content_type,
                size_bytes=replacement.size_bytes,
                delete_source_file=delete_source_file,
                delete_artifacts=delete_artifacts,
                delete_llm_cache=delete_llm_cache,
                auto_parse=auto_parse,
                auto_index=auto_index,
                parser_engine=parser_engine,
                process_options=process_options,
                force_reparse=force_reparse,
                idempotency_key=idempotency_key,
            )
            if not created_job:
                return JobResponse.from_record(job)
            try:
                document = await document_service.claim_replace(
                    kb_id,
                    document_id,
                    job=job,
                    replacement=replacement,
                    delete_source_file=delete_source_file,
                    delete_artifacts=delete_artifacts,
                    delete_llm_cache=delete_llm_cache,
                    auto_parse=auto_parse,
                    auto_index=auto_index,
                    parser_engine=parser_engine,
                    process_options=process_options,
                    force_reparse=force_reparse,
                )
            except (
                ActiveDocumentParseJobError,
                ActiveDocumentBuildJobError,
                ActiveDocumentDeleteJobError,
                ActiveDocumentReplaceJobError,
            ) as exc:
                error_code = _active_job_error_code(exc)
                await job_service.transition_job(
                    kb_id,
                    job.id,
                    status="failed",
                    progress=1.0,
                    failed_items=1,
                    error_code=error_code,
                    error_message=str(exc),
                )
                raise HTTPException(
                    status_code=409,
                    detail=_active_job_conflict_detail(exc),
                ) from exc

            async def _replace_task() -> None:
                replace_claim_released = False
                try:
                    await job_service.transition_job(
                        kb_id, job.id, status="running", progress=0.1
                    )
                    item = await _execute_replace_document(
                        kb_id=kb_id,
                        job=job,
                        document=document,
                        replacement=replacement,
                        active_registry=active_registry,
                        active_index_service=active_index_service,
                        delete_source_file=delete_source_file,
                        delete_artifacts=delete_artifacts,
                        delete_llm_cache=delete_llm_cache,
                        auto_parse=auto_parse,
                        auto_index=auto_index,
                        parser_engine=parser_engine,
                        process_options=process_options,
                        force_reparse=force_reparse,
                    )
                    replace_claim_released = True
                    if item["status"] == "succeeded":
                        await job_service.transition_job(
                            kb_id,
                            job.id,
                            status="succeeded",
                            progress=1.0,
                            completed_items=1,
                            result=item,
                        )
                    else:
                        await job_service.transition_job(
                            kb_id,
                            job.id,
                            status="failed",
                            progress=1.0,
                            failed_items=1,
                            result=item,
                            error_code=item.get("error_code", "replace_failed"),
                            error_message=item.get("error_message"),
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Failed to run replace job '%s' for KB '%s': %s",
                        job.id,
                        kb_id,
                        exc,
                    )
                    if not replace_claim_released:
                        try:
                            current_document = await document_service.get_document(
                                kb_id, document.id
                            )
                            if current_document.status == "replacing":
                                await document_service.fail_replace(
                                    kb_id,
                                    document.id,
                                    job_id=job.id,
                                    error_code="replace_failed",
                                    error_message=str(exc),
                                )
                        except Exception as transition_exc:
                            logger.error(
                                "Failed to release replace claim for document '%s': %s",
                                document.id,
                                transition_exc,
                            )
                    try:
                        await job_service.transition_job(
                            kb_id,
                            job.id,
                            status="failed",
                            progress=1.0,
                            failed_items=1,
                            error_code="replace_failed",
                            error_message=str(exc),
                        )
                    except InvalidJobTransitionError:
                        logger.warning(
                            "Replace job '%s' for KB '%s' was already terminal",
                            job.id,
                            kb_id,
                        )

            background_tasks.add_task(_replace_task)
            return JobResponse.from_record(job)
        except HTTPException:
            raise
        except (KnowledgeBaseNotFoundError, MetadataRecordNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except IdempotencyKeyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error(
                "Failed to start replace for KB '%s' doc '%s': %s",
                kb_id,
                document_id,
                exc,
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    async def _execute_delete_document(
        *,
        kb_id: str,
        job_id: str,
        document: DocumentRecord,
        active_registry: LightRAGInstanceRegistry,
        delete_source_file: bool,
        delete_artifacts: bool,
        delete_llm_cache: bool,
    ) -> dict[str, Any]:
        try:
            lightrag_result = None
            if document.lightrag_doc_id:
                rag = cast(Any, await active_registry.get(kb_id))
                lightrag_result = await rag.adelete_by_doc_id(
                    document.lightrag_doc_id,
                    delete_llm_cache=delete_llm_cache,
                )
                if getattr(lightrag_result, "status", None) not in {
                    "success",
                    "not_found",
                }:
                    raise RuntimeError(
                        getattr(lightrag_result, "message", None)
                        or f"LightRAG deletion failed for {document.lightrag_doc_id}"
                    )
            file_result = await document_service.cleanup_document_files(
                kb_id,
                document,
                delete_source_file=delete_source_file,
                delete_artifacts=delete_artifacts,
            )
            if file_result.errors:
                raise RuntimeError("; ".join(file_result.errors))
            await document_service.complete_delete(
                kb_id,
                document.id,
                job_id=job_id,
                lightrag_result=_deletion_result_payload(lightrag_result),
                file_result=file_result,
            )
            return {
                "document_id": document.id,
                "status": "succeeded",
                "lightrag_doc_id": document.lightrag_doc_id,
                "lightrag_delete_result": _deletion_result_payload(lightrag_result),
                "file_delete_result": file_result.__dict__
                if hasattr(file_result, "__dict__")
                else {
                    "deleted_source": file_result.deleted_source,
                    "deleted_artifacts": file_result.deleted_artifacts,
                    "skipped": file_result.skipped,
                    "errors": file_result.errors,
                },
            }
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to delete document '%s' for KB '%s': %s",
                document.id,
                kb_id,
                exc,
            )
            try:
                await document_service.fail_delete(
                    kb_id,
                    document.id,
                    job_id=job_id,
                    error_code="delete_failed",
                    error_message=str(exc),
                )
            except Exception as transition_exc:
                logger.error(
                    "Failed to mark document '%s' failed for delete job '%s': %s",
                    document.id,
                    job_id,
                    transition_exc,
                )
            return {
                "document_id": document.id,
                "status": "failed",
                "error_code": "delete_failed",
                "error_message": str(exc),
            }

    @router.delete(
        "/{kb_id}/documents/{document_id}",
        response_model=JobResponse,
        dependencies=[Depends(combined_auth)],
        summary="Delete one knowledge base document",
    )
    async def delete_document(
        kb_id: str,
        document_id: str,
        background_tasks: BackgroundTasks,
        delete_source_file: bool = False,
        delete_artifacts: bool = False,
        delete_llm_cache: bool = False,
        idempotency_key: Optional[str] = None,
    ):
        if registry is None:
            raise HTTPException(
                status_code=503, detail="KB delete service is not configured"
            )
        active_registry = registry
        try:
            if idempotency_key is not None:
                existing_job = await job_service.get_job_by_idempotency_key(
                    kb_id, idempotency_key, job_type="delete"
                )
                if existing_job is not None:
                    existing_payload = existing_job.payload
                    same_request = (
                        existing_job.document_id == document_id
                        and existing_payload.get("document_id") == document_id
                        and bool(existing_payload.get("delete_source_file"))
                        == delete_source_file
                        and bool(existing_payload.get("delete_artifacts"))
                        == delete_artifacts
                        and bool(existing_payload.get("delete_llm_cache"))
                        == delete_llm_cache
                    )
                    if not same_request:
                        raise IdempotencyKeyConflictError(idempotency_key)
                    return JobResponse.from_record(existing_job)
            document = await document_service.get_document(kb_id, document_id)
            job, created_job = await job_service.create_delete_job_once(
                kb_id,
                document_id=document_id,
                lightrag_doc_id=document.lightrag_doc_id,
                delete_source_file=delete_source_file,
                delete_artifacts=delete_artifacts,
                delete_llm_cache=delete_llm_cache,
                idempotency_key=idempotency_key,
            )
            if not created_job:
                return JobResponse.from_record(job)
            try:
                document = await document_service.claim_delete(
                    kb_id,
                    document_id,
                    job=job,
                    delete_source_file=delete_source_file,
                    delete_artifacts=delete_artifacts,
                )
            except (
                ActiveDocumentParseJobError,
                ActiveDocumentBuildJobError,
                ActiveDocumentDeleteJobError,
                ActiveDocumentReplaceJobError,
            ) as exc:
                error_code = _active_job_error_code(exc)
                await job_service.transition_job(
                    kb_id,
                    job.id,
                    status="failed",
                    progress=1.0,
                    failed_items=1,
                    error_code=error_code,
                    error_message=str(exc),
                )
                raise HTTPException(
                    status_code=409,
                    detail=_active_job_conflict_detail(exc),
                ) from exc

            async def _delete_task() -> None:
                try:
                    await job_service.transition_job(
                        kb_id, job.id, status="running", progress=0.1
                    )
                    item = await _execute_delete_document(
                        kb_id=kb_id,
                        job_id=job.id,
                        document=document,
                        active_registry=active_registry,
                        delete_source_file=delete_source_file,
                        delete_artifacts=delete_artifacts,
                        delete_llm_cache=delete_llm_cache,
                    )
                    if item["status"] == "succeeded":
                        await job_service.transition_job(
                            kb_id,
                            job.id,
                            status="succeeded",
                            progress=1.0,
                            completed_items=1,
                            result=_delete_job_result(
                                total_items=1,
                                completed_items=1,
                                failed_items=0,
                                items=[item],
                            ),
                        )
                    else:
                        await job_service.transition_job(
                            kb_id,
                            job.id,
                            status="failed",
                            progress=1.0,
                            failed_items=1,
                            result=_delete_job_result(
                                total_items=1,
                                completed_items=0,
                                failed_items=1,
                                items=[item],
                            ),
                            error_code=item["error_code"],
                            error_message=item["error_message"],
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Failed to run delete job '%s' for KB '%s': %s",
                        job.id,
                        kb_id,
                        exc,
                    )
                    await job_service.transition_job(
                        kb_id,
                        job.id,
                        status="failed",
                        progress=1.0,
                        failed_items=1,
                        error_code="delete_failed",
                        error_message=str(exc),
                    )

            background_tasks.add_task(_delete_task)
            return JobResponse.from_record(job)
        except HTTPException:
            raise
        except (KnowledgeBaseNotFoundError, MetadataRecordNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except IdempotencyKeyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post(
        "/{kb_id}/documents:batch-delete",
        response_model=JobResponse,
        dependencies=[Depends(combined_auth)],
        summary="Delete multiple knowledge base documents",
    )
    async def batch_delete_documents(
        kb_id: str,
        background_tasks: BackgroundTasks,
        request: BatchDeleteDocumentsRequest,
    ):
        if registry is None:
            raise HTTPException(
                status_code=503, detail="KB delete service is not configured"
            )
        active_registry = registry
        try:
            batch_id = generate_track_id("batch")
            job, created_job = await job_service.create_batch_delete_job_once(
                kb_id,
                batch_id=batch_id,
                document_ids=request.document_ids,
                delete_source_file=request.delete_source_file,
                delete_artifacts=request.delete_artifacts,
                delete_llm_cache=request.delete_llm_cache,
                idempotency_key=request.idempotency_key,
            )
            if not created_job:
                return JobResponse.from_record(job)
            documents, claim_failures = await document_service.claim_batch_delete(
                kb_id,
                request.document_ids,
                job=job,
                delete_source_file=request.delete_source_file,
                delete_artifacts=request.delete_artifacts,
            )

            async def _batch_delete_task() -> None:
                item_results = [*claim_failures]
                completed_items = 0
                failed_items = len(item_results)
                try:
                    await job_service.transition_job(
                        kb_id,
                        job.id,
                        status="running",
                        progress=0.0,
                        failed_items=failed_items,
                        result=_delete_job_result(
                            batch_id=job.batch_id or batch_id,
                            total_items=len(request.document_ids),
                            completed_items=completed_items,
                            failed_items=failed_items,
                            items=item_results,
                        ),
                    )
                    for document in documents:
                        item = await _execute_delete_document(
                            kb_id=kb_id,
                            job_id=job.id,
                            document=document,
                            active_registry=active_registry,
                            delete_source_file=request.delete_source_file,
                            delete_artifacts=request.delete_artifacts,
                            delete_llm_cache=request.delete_llm_cache,
                        )
                        item_results.append(item)
                        if item["status"] == "succeeded":
                            completed_items += 1
                        else:
                            failed_items += 1
                    final_result = _delete_job_result(
                        batch_id=job.batch_id or batch_id,
                        total_items=len(request.document_ids),
                        completed_items=completed_items,
                        failed_items=failed_items,
                        items=item_results,
                    )
                    await job_service.transition_job(
                        kb_id,
                        job.id,
                        status="succeeded" if failed_items == 0 else "failed",
                        progress=1.0,
                        completed_items=completed_items,
                        failed_items=failed_items,
                        result=final_result,
                        error_code=None
                        if failed_items == 0
                        else "partial_delete_failed",
                        error_message=None
                        if failed_items == 0
                        else _delete_failure_message(
                            failed_items, len(request.document_ids)
                        ),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Failed to run batch delete job '%s' for KB '%s': %s",
                        job.id,
                        kb_id,
                        exc,
                    )
                    processed_ids = {item["document_id"] for item in item_results}
                    for document in documents:
                        if document.id in processed_ids:
                            continue
                        item_results.append(
                            {
                                "document_id": document.id,
                                "status": "failed",
                                "error_code": "delete_failed",
                                "error_message": str(exc),
                            }
                        )
                        failed_items += 1
                        await document_service.fail_delete(
                            kb_id,
                            document.id,
                            job_id=job.id,
                            error_code="delete_failed",
                            error_message=str(exc),
                        )
                    await job_service.transition_job(
                        kb_id,
                        job.id,
                        status="failed",
                        progress=1.0,
                        completed_items=completed_items,
                        failed_items=failed_items,
                        result=_delete_job_result(
                            batch_id=job.batch_id or batch_id,
                            total_items=len(request.document_ids),
                            completed_items=completed_items,
                            failed_items=failed_items,
                            items=item_results,
                        ),
                        error_code="batch_delete_failed",
                        error_message=str(exc),
                    )

            background_tasks.add_task(_batch_delete_task)
            return JobResponse.from_record(job)
        except HTTPException:
            raise
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except IdempotencyKeyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post(
        "/{kb_id}/documents:batch-parse",
        response_model=DocumentBatchResponse,
        dependencies=[Depends(combined_auth)],
        summary="Parse multiple knowledge base documents without building the index",
    )
    async def batch_parse_documents(
        kb_id: str,
        background_tasks: BackgroundTasks,
        request: BatchParseDocumentsRequest,
    ):
        if registry is None:
            raise HTTPException(
                status_code=503,
                detail="KB parse service is not configured",
            )
        active_registry = registry
        try:
            batch_plan = await document_service.create_batch_parse_plan(
                kb_id,
                request.document_ids,
                parser_engine=request.engine,
                process_options=request.process_options,
                force_reparse=request.force_reparse,
                auto_index=request.auto_index,
            )
            job, created_job = await job_service.create_batch_parse_job_once(
                kb_id,
                batch_id=batch_plan.batch_id,
                document_ids=request.document_ids,
                total_items=len(request.document_ids),
                plan_items=[_parse_plan_payload(plan) for plan in batch_plan.plans],
                planning_failures=batch_plan.failures,
                force_reparse=request.force_reparse,
                auto_index=request.auto_index,
                idempotency_key=request.idempotency_key,
            )
            if not created_job:
                existing_document_ids = job.payload.get("document_ids")
                existing_document_id_values = (
                    [
                        document_id
                        for document_id in existing_document_ids
                        if isinstance(document_id, str)
                    ]
                    if isinstance(existing_document_ids, list)
                    else []
                )
                existing_documents = await document_service.get_documents_by_ids(
                    kb_id,
                    existing_document_id_values,
                )
                return DocumentBatchResponse(
                    job_id=job.id,
                    batch_id=job.batch_id or "",
                    documents=[
                        DocumentResponse.from_record(item)
                        for item in existing_documents
                    ],
                )
            (
                queued_documents,
                claim_failures,
            ) = await document_service.claim_batch_parse_queued(
                kb_id, job=job, plans=batch_plan.plans
            )
            queued_document_ids = {document.id for document in queued_documents}
            execution_plans = [
                plan
                for plan in batch_plan.plans
                if plan.document.id in queued_document_ids
            ]

            async def _batch_parse_task() -> None:
                item_results = [*batch_plan.failures, *claim_failures]
                completed_items = 0
                failed_items = len(item_results)
                try:
                    await job_service.transition_job(
                        kb_id,
                        job.id,
                        status="running",
                        progress=0.0,
                        failed_items=failed_items,
                        result=_batch_parse_job_result(
                            batch_id=job.batch_id or batch_plan.batch_id,
                            total_items=len(request.document_ids),
                            completed_items=completed_items,
                            failed_items=failed_items,
                            items=item_results,
                        ),
                    )
                    rag = await active_registry.get(kb_id) if execution_plans else None
                    for plan in execution_plans:
                        if rag is None:
                            raise RuntimeError(
                                "KB parse service did not return a LightRAG instance"
                            )
                        item = await _execute_parse_plan(
                            document_service=document_service,
                            kb_id=kb_id,
                            job_id=job.id,
                            plan=plan,
                            rag=rag,
                        )
                        item_results.append(item)
                        if item["status"] == "succeeded":
                            completed_items += 1
                        else:
                            failed_items += 1

                    final_result = _batch_parse_job_result(
                        batch_id=job.batch_id or batch_plan.batch_id,
                        total_items=len(request.document_ids),
                        completed_items=completed_items,
                        failed_items=failed_items,
                        items=item_results,
                    )
                    final_status = "succeeded" if failed_items == 0 else "failed"
                    await job_service.transition_job(
                        kb_id,
                        job.id,
                        status=final_status,
                        progress=1.0,
                        completed_items=completed_items,
                        failed_items=failed_items,
                        result=final_result,
                        error_code=None
                        if failed_items == 0
                        else "partial_parse_failed",
                        error_message=None
                        if failed_items == 0
                        else _batch_parse_failure_message(
                            failed_items, len(request.document_ids)
                        ),
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to run batch parse job '%s' for KB '%s': %s",
                        job.id,
                        kb_id,
                        exc,
                    )
                    processed_ids = {item["document_id"] for item in item_results}
                    for plan in execution_plans:
                        if plan.document.id in processed_ids:
                            continue
                        item_results.append(
                            {
                                "document_id": plan.document.id,
                                "status": "failed",
                                "error_code": "parse_failed",
                                "error_message": str(exc),
                            }
                        )
                        failed_items += 1
                        try:
                            await document_service.fail_parse(
                                kb_id,
                                plan.document.id,
                                job_id=job.id,
                                plan=plan,
                                error_code="parse_failed",
                                error_message=str(exc),
                            )
                        except Exception as transition_exc:
                            logger.error(
                                "Failed to mark document '%s' failed for batch parse job '%s': %s",
                                plan.document.id,
                                job.id,
                                transition_exc,
                            )
                    failed_items = len(request.document_ids) - completed_items
                    await job_service.transition_job(
                        kb_id,
                        job.id,
                        status="failed",
                        progress=1.0,
                        completed_items=completed_items,
                        failed_items=failed_items,
                        result=_batch_parse_job_result(
                            batch_id=job.batch_id or batch_plan.batch_id,
                            total_items=len(request.document_ids),
                            completed_items=completed_items,
                            failed_items=failed_items,
                            items=item_results,
                        ),
                        error_code="batch_parse_failed",
                        error_message=str(exc),
                    )

            background_tasks.add_task(_batch_parse_task)
            return DocumentBatchResponse(
                job_id=job.id,
                batch_id=job.batch_id or batch_plan.batch_id,
                documents=[
                    DocumentResponse.from_record(item) for item in queued_documents
                ],
            )
        except HTTPException:
            raise
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except IdempotencyKeyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("Failed to start batch parse for KB '%s': %s", kb_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post(
        "/{kb_id}/documents/{document_id}:parse",
        response_model=JobResponse,
        dependencies=[Depends(combined_auth)],
        summary="Parse one knowledge base document without building the index",
    )
    async def parse_document(
        kb_id: str,
        document_id: str,
        background_tasks: BackgroundTasks,
        request: ParseDocumentRequest = Body(default_factory=ParseDocumentRequest),
    ):
        if registry is None:
            raise HTTPException(
                status_code=503,
                detail="KB parse service is not configured",
            )
        active_registry = registry
        try:
            plan = await document_service.create_parse_plan(
                kb_id,
                document_id,
                parser_engine=request.engine,
                process_options=request.process_options,
                force_reparse=request.force_reparse,
                auto_index=request.auto_index,
            )
            job, created_job = await job_service.create_parse_job_once(
                kb_id,
                document_id=document_id,
                parser_hash=plan.parser_hash,
                lightrag_doc_id=plan.lightrag_doc_id,
                parser_engine=plan.parser_engine,
                process_options=plan.process_options,
                source_uri=str(plan.source_path),
                source_hash=plan.document.source_hash,
                force_reparse=plan.force_reparse,
                auto_index=plan.auto_index,
                idempotency_key=request.idempotency_key,
            )
            if not created_job:
                return JobResponse.from_record(job)
            try:
                await document_service.mark_parse_queued(
                    kb_id, document_id, job=job, plan=plan
                )
            except (
                ActiveDocumentParseJobError,
                ActiveDocumentBuildJobError,
                ActiveDocumentDeleteJobError,
                ActiveDocumentReplaceJobError,
            ) as exc:
                error_code = _active_job_error_code(exc)
                await job_service.transition_job(
                    kb_id,
                    job.id,
                    status="failed",
                    progress=1.0,
                    failed_items=1,
                    error_code=error_code,
                    error_message=str(exc),
                )
                raise HTTPException(
                    status_code=409,
                    detail=_active_job_conflict_detail(exc),
                ) from exc

            async def _parse_task() -> None:
                try:
                    await job_service.transition_job(
                        kb_id, job.id, status="running", progress=0.1
                    )
                    rag = await active_registry.get(kb_id)
                    item = await _execute_parse_plan(
                        document_service=document_service,
                        kb_id=kb_id,
                        job_id=job.id,
                        plan=plan,
                        rag=rag,
                    )
                    if item["status"] == "succeeded":
                        await job_service.transition_job(
                            kb_id,
                            job.id,
                            status="succeeded",
                            progress=1.0,
                            completed_items=1,
                            result={
                                "document_id": item["document_id"],
                                "parser_hash": item["parser_hash"],
                                "lightrag_doc_id": item["lightrag_doc_id"],
                                "artifact_count": item["artifact_count"],
                            },
                        )
                    else:
                        await job_service.transition_job(
                            kb_id,
                            job.id,
                            status="failed",
                            progress=1.0,
                            failed_items=1,
                            error_code=item["error_code"],
                            error_message=item["error_message"],
                        )
                except Exception as exc:
                    logger.error(
                        "Failed to parse document '%s' for KB '%s': %s",
                        document_id,
                        kb_id,
                        exc,
                    )
                    try:
                        await document_service.fail_parse(
                            kb_id,
                            document_id,
                            job_id=job.id,
                            plan=plan,
                            error_code="parse_failed",
                            error_message=str(exc),
                        )
                        await job_service.transition_job(
                            kb_id,
                            job.id,
                            status="failed",
                            progress=1.0,
                            failed_items=1,
                            error_code="parse_failed",
                            error_message=str(exc),
                        )
                    except Exception as transition_exc:
                        logger.error(
                            "Failed to mark parse job '%s' failed: %s",
                            job.id,
                            transition_exc,
                        )

            background_tasks.add_task(_parse_task)
            return JobResponse.from_record(job)
        except HTTPException:
            raise
        except (KnowledgeBaseNotFoundError, MetadataRecordNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except IdempotencyKeyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("Failed to start parse for KB '%s': %s", kb_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    async def _start_single_build_job(
        *,
        kb_id: str,
        document_id: str,
        force_rechunk: bool,
        force_extract: bool,
        force_embedding: bool,
        idempotency_key: Optional[str],
        background_tasks: BackgroundTasks,
    ) -> JobResponse:
        if registry is None or index_service is None:
            raise HTTPException(
                status_code=503,
                detail="KB index build service is not configured",
            )
        active_registry = registry
        active_index_service = index_service
        rag = await active_registry.get(kb_id)
        plan = await active_index_service.create_build_plan(
            kb_id,
            document_id,
            rag=rag,
            force_rechunk=force_rechunk,
            force_extract=force_extract,
            force_embedding=force_embedding,
        )
        job, created_job = await job_service.create_build_job_once(
            kb_id,
            document_id=document_id,
            parser_hash=plan.parser_hash,
            index_hash=plan.index_hash,
            source_hash=plan.document.source_hash,
            lightrag_doc_id=plan.document.lightrag_doc_id or "",
            sidecar_uri=plan.sidecar_uri,
            blocks_path=plan.blocks_path,
            process_options=plan.process_options,
            force_rechunk=force_rechunk,
            force_extract=force_extract,
            force_embedding=force_embedding,
            idempotency_key=idempotency_key,
        )
        if not created_job:
            return JobResponse.from_record(job)

        try:
            await active_index_service.claim_build_queued(
                kb_id, job_id=job.id, plan=plan
            )
        except (
            ActiveDocumentBuildJobError,
            ActiveDocumentDeleteJobError,
            ActiveDocumentReplaceJobError,
        ) as exc:
            error_code = _active_job_error_code(exc)
            await job_service.transition_job(
                kb_id,
                job.id,
                status="failed",
                progress=1.0,
                failed_items=1,
                error_code=error_code,
                error_message=str(exc),
            )
            raise HTTPException(
                status_code=409,
                detail=_active_job_conflict_detail(exc),
            ) from exc

        if plan.skipped:
            await job_service.transition_job(
                kb_id,
                job.id,
                status="running",
                progress=0.5,
            )
            try:
                run_result = await active_index_service.run_build(rag, plan)
                document = await active_index_service.complete_build(
                    kb_id,
                    document_id,
                    job_id=job.id,
                    plan=plan,
                    run_result=run_result,
                )
            except Exception as exc:  # noqa: BLE001
                await job_service.transition_job(
                    kb_id,
                    job.id,
                    status="failed",
                    progress=1.0,
                    failed_items=1,
                    error_code="build_failed",
                    error_message=str(exc),
                )
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            final_job = await job_service.transition_job(
                kb_id,
                job.id,
                status="succeeded",
                progress=1.0,
                completed_items=1,
                result={
                    "document_id": document.id,
                    "skipped": True,
                    "skip_reason": plan.skip_reason,
                    "index_hash": plan.index_hash,
                    "chunks_count": document.chunks_count,
                    "entity_count": document.entity_count,
                    "relation_count": document.relation_count,
                },
            )
            return JobResponse.from_record(final_job)

        async def _build_task() -> None:
            try:
                await job_service.transition_job(
                    kb_id, job.id, status="running", progress=0.1
                )
                inner_rag = await active_registry.get(kb_id)
                item = await _execute_build_plan(
                    index_service=active_index_service,
                    kb_id=kb_id,
                    job_id=job.id,
                    plan=plan,
                    rag=inner_rag,
                )
                if item["status"] == "succeeded":
                    await job_service.transition_job(
                        kb_id,
                        job.id,
                        status="succeeded",
                        progress=1.0,
                        completed_items=1,
                        result={
                            "document_id": item["document_id"],
                            "skipped": item["skipped"],
                            "skip_reason": item.get("skip_reason"),
                            "index_hash": item["index_hash"],
                            "chunks_count": item.get("chunks_count"),
                            "entity_count": item.get("entity_count"),
                            "relation_count": item.get("relation_count"),
                        },
                    )
                else:
                    await job_service.transition_job(
                        kb_id,
                        job.id,
                        status="failed",
                        progress=1.0,
                        failed_items=1,
                        error_code=item["error_code"],
                        error_message=item["error_message"],
                    )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to build KG for document '%s' (KB '%s'): %s",
                    document_id,
                    kb_id,
                    exc,
                )
                try:
                    await active_index_service.fail_build(
                        kb_id,
                        document_id,
                        job_id=job.id,
                        error_code="build_failed",
                        error_message=str(exc),
                    )
                    await job_service.transition_job(
                        kb_id,
                        job.id,
                        status="failed",
                        progress=1.0,
                        failed_items=1,
                        error_code="build_failed",
                        error_message=str(exc),
                    )
                except Exception as transition_exc:
                    logger.error(
                        "Failed to mark build job '%s' failed: %s",
                        job.id,
                        transition_exc,
                    )

        background_tasks.add_task(_build_task)
        return JobResponse.from_record(job)

    @router.post(
        "/{kb_id}/documents/{document_id}:build-kg",
        response_model=JobResponse,
        dependencies=[Depends(combined_auth)],
        summary="Build the knowledge graph and index for one parsed document",
    )
    async def build_document_kg(
        kb_id: str,
        document_id: str,
        background_tasks: BackgroundTasks,
        request: BuildKGRequest = Body(default_factory=BuildKGRequest),
    ):
        try:
            return await _start_single_build_job(
                kb_id=kb_id,
                document_id=document_id,
                force_rechunk=request.force_rechunk,
                force_extract=request.force_extract,
                force_embedding=request.force_embedding,
                idempotency_key=request.idempotency_key,
                background_tasks=background_tasks,
            )
        except HTTPException:
            raise
        except (KnowledgeBaseNotFoundError, MetadataRecordNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except DocumentNotParsedError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "error_code": "document_not_parsed",
                    "document_id": exc.document_id,
                    "current_status": exc.current_status,
                    "message": str(exc),
                },
            ) from exc
        except IdempotencyKeyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error(
                "Failed to start build_kg for KB '%s' doc '%s': %s",
                kb_id,
                document_id,
                exc,
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post(
        "/{kb_id}/documents/{document_id}:reindex",
        response_model=JobResponse,
        dependencies=[Depends(combined_auth)],
        summary="Reindex one document by forcing chunk/extract/embedding stages",
    )
    async def reindex_document(
        kb_id: str,
        document_id: str,
        background_tasks: BackgroundTasks,
        request: ReindexRequest = Body(default_factory=ReindexRequest),
    ):
        try:
            return await _start_single_build_job(
                kb_id=kb_id,
                document_id=document_id,
                force_rechunk=request.force_rechunk,
                force_extract=request.force_extract,
                force_embedding=request.force_embedding,
                idempotency_key=request.idempotency_key,
                background_tasks=background_tasks,
            )
        except HTTPException:
            raise
        except (KnowledgeBaseNotFoundError, MetadataRecordNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except DocumentNotParsedError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "error_code": "document_not_parsed",
                    "document_id": exc.document_id,
                    "current_status": exc.current_status,
                    "message": str(exc),
                },
            ) from exc
        except IdempotencyKeyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error(
                "Failed to start reindex for KB '%s' doc '%s': %s",
                kb_id,
                document_id,
                exc,
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    async def _start_batch_build_job(
        *,
        kb_id: str,
        request_ids: list[str],
        force_rechunk: bool,
        force_extract: bool,
        force_embedding: bool,
        idempotency_key: Optional[str],
        background_tasks: BackgroundTasks,
    ) -> DocumentBatchResponse:
        if registry is None or index_service is None:
            raise HTTPException(
                status_code=503,
                detail="KB index build service is not configured",
            )
        active_registry = registry
        active_index_service = index_service
        rag = await active_registry.get(kb_id)
        batch_plan = await active_index_service.create_batch_build_plan(
            kb_id,
            request_ids,
            rag=rag,
            force_rechunk=force_rechunk,
            force_extract=force_extract,
            force_embedding=force_embedding,
        )
        job, created_job = await job_service.create_batch_build_job_once(
            kb_id,
            batch_id=batch_plan.batch_id,
            document_ids=request_ids,
            total_items=len(request_ids),
            plan_items=[_build_plan_payload(plan) for plan in batch_plan.plans],
            planning_failures=batch_plan.failures,
            force_rechunk=force_rechunk,
            force_extract=force_extract,
            force_embedding=force_embedding,
            idempotency_key=idempotency_key,
        )
        if not created_job:
            existing_document_ids = job.payload.get("document_ids")
            existing_document_id_values = (
                [
                    document_id
                    for document_id in existing_document_ids
                    if isinstance(document_id, str)
                ]
                if isinstance(existing_document_ids, list)
                else []
            )
            existing_documents = await document_service.get_documents_by_ids(
                kb_id, existing_document_id_values
            )
            return DocumentBatchResponse(
                job_id=job.id,
                batch_id=job.batch_id or "",
                documents=[
                    DocumentResponse.from_record(item) for item in existing_documents
                ],
            )

        skipped_plans = [plan for plan in batch_plan.plans if plan.skipped]
        active_plans = [plan for plan in batch_plan.plans if not plan.skipped]
        (
            queued_documents,
            claim_failures,
        ) = await active_index_service.claim_batch_build_queued(
            kb_id, job_id=job.id, plans=batch_plan.plans
        )
        queued_document_ids = {document.id for document in queued_documents}
        skipped_plans = [
            plan for plan in skipped_plans if plan.document.id in queued_document_ids
        ]
        execution_plans = [
            plan for plan in active_plans if plan.document.id in queued_document_ids
        ]

        async def _batch_build_task() -> None:
            item_results: list[dict[str, Any]] = [
                *batch_plan.failures,
                *claim_failures,
            ]
            completed_items = 0
            failed_items = len(item_results)
            try:
                await job_service.transition_job(
                    kb_id,
                    job.id,
                    status="running",
                    progress=0.0,
                    failed_items=failed_items,
                    result=_batch_build_job_result(
                        batch_id=job.batch_id or batch_plan.batch_id,
                        total_items=len(request_ids),
                        completed_items=completed_items,
                        failed_items=failed_items,
                        items=item_results,
                    ),
                )
                inner_rag = (
                    await active_registry.get(kb_id)
                    if (execution_plans or skipped_plans)
                    else None
                )
                for plan in [*skipped_plans, *execution_plans]:
                    if inner_rag is None:
                        raise RuntimeError(
                            "KB index build service did not return a LightRAG instance"
                        )
                    item = await _execute_build_plan(
                        index_service=active_index_service,
                        kb_id=kb_id,
                        job_id=job.id,
                        plan=plan,
                        rag=inner_rag,
                    )
                    item_results.append(item)
                    if item["status"] == "succeeded":
                        completed_items += 1
                    else:
                        failed_items += 1

                final_result = _batch_build_job_result(
                    batch_id=job.batch_id or batch_plan.batch_id,
                    total_items=len(request_ids),
                    completed_items=completed_items,
                    failed_items=failed_items,
                    items=item_results,
                )
                final_status = "succeeded" if failed_items == 0 else "failed"
                await job_service.transition_job(
                    kb_id,
                    job.id,
                    status=final_status,
                    progress=1.0,
                    completed_items=completed_items,
                    failed_items=failed_items,
                    result=final_result,
                    error_code=None if failed_items == 0 else "partial_build_failed",
                    error_message=None
                    if failed_items == 0
                    else _batch_build_failure_message(failed_items, len(request_ids)),
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to run batch build job '%s' for KB '%s': %s",
                    job.id,
                    kb_id,
                    exc,
                )
                processed_ids = {item["document_id"] for item in item_results}
                for plan in execution_plans:
                    if plan.document.id in processed_ids:
                        continue
                    item_results.append(
                        {
                            "document_id": plan.document.id,
                            "status": "failed",
                            "error_code": "build_failed",
                            "error_message": str(exc),
                        }
                    )
                    failed_items += 1
                    try:
                        await active_index_service.fail_build(
                            kb_id,
                            plan.document.id,
                            job_id=job.id,
                            error_code="build_failed",
                            error_message=str(exc),
                        )
                    except Exception as transition_exc:
                        logger.error(
                            "Failed to mark build job '%s' failed for batch: %s",
                            job.id,
                            transition_exc,
                        )
                await job_service.transition_job(
                    kb_id,
                    job.id,
                    status="failed",
                    progress=1.0,
                    completed_items=completed_items,
                    failed_items=failed_items,
                    result=_batch_build_job_result(
                        batch_id=job.batch_id or batch_plan.batch_id,
                        total_items=len(request_ids),
                        completed_items=completed_items,
                        failed_items=failed_items,
                        items=item_results,
                    ),
                    error_code="batch_build_failed",
                    error_message=str(exc),
                )

        background_tasks.add_task(_batch_build_task)

        # Return queued + skipped (skipped processed within task) + planning-known docs
        all_known_ids = list(queued_document_ids) + [
            plan.document.id for plan in skipped_plans
        ]
        seen: set[str] = set()
        ordered_ids = []
        for document_id in all_known_ids:
            if document_id in seen:
                continue
            seen.add(document_id)
            ordered_ids.append(document_id)
        documents = await document_service.get_documents_by_ids(kb_id, ordered_ids)
        return DocumentBatchResponse(
            job_id=job.id,
            batch_id=job.batch_id or batch_plan.batch_id,
            documents=[DocumentResponse.from_record(item) for item in documents],
        )

    @router.post(
        "/{kb_id}/documents:batch-build-kg",
        response_model=DocumentBatchResponse,
        dependencies=[Depends(combined_auth)],
        summary="Build the knowledge graph and index for multiple parsed documents",
    )
    async def batch_build_documents(
        kb_id: str,
        background_tasks: BackgroundTasks,
        request: BatchBuildKGRequest,
    ):
        try:
            return await _start_batch_build_job(
                kb_id=kb_id,
                request_ids=request.document_ids,
                force_rechunk=request.force_rechunk,
                force_extract=request.force_extract,
                force_embedding=request.force_embedding,
                idempotency_key=request.idempotency_key,
                background_tasks=background_tasks,
            )
        except HTTPException:
            raise
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except IdempotencyKeyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("Failed to start batch build_kg for KB '%s': %s", kb_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post(
        "/{kb_id}/documents:batch-reindex",
        response_model=DocumentBatchResponse,
        dependencies=[Depends(combined_auth)],
        summary="Reindex multiple documents by forcing chunk/extract/embedding stages",
    )
    async def batch_reindex_documents(
        kb_id: str,
        background_tasks: BackgroundTasks,
        request: BatchReindexRequest,
    ):
        try:
            return await _start_batch_build_job(
                kb_id=kb_id,
                request_ids=request.document_ids,
                force_rechunk=request.force_rechunk,
                force_extract=request.force_extract,
                force_embedding=request.force_embedding,
                idempotency_key=request.idempotency_key,
                background_tasks=background_tasks,
            )
        except HTTPException:
            raise
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except IdempotencyKeyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("Failed to start batch reindex for KB '%s': %s", kb_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post(
        "/{kb_id}:rebuild",
        response_model=DocumentBatchResponse,
        dependencies=[Depends(combined_auth)],
        summary="Rebuild the whole KB index by reindexing every buildable document",
    )
    async def rebuild_kb(
        kb_id: str,
        background_tasks: BackgroundTasks,
        request: RebuildKBRequest = Body(default_factory=RebuildKBRequest),
    ):
        """Conservative whole-KB rebuild.

        Enumerates every document currently in a buildable state
        (``parsed`` / ``ready`` / ``build_failed``) and runs the same batch
        build path used by ``:batch-reindex``, defaulting all ``force_*`` flags
        to ``True``. Returns an empty document list (no-op job) when the KB has
        no buildable documents.
        """
        try:
            buildable_statuses = ("parsed", "ready", "build_failed")
            document_ids: list[str] = []
            for status in buildable_statuses:
                offset = 0
                page_size = 200
                while True:
                    documents, total = await document_service.list_documents(
                        kb_id, status=status, limit=page_size, offset=offset
                    )
                    document_ids.extend(doc.id for doc in documents)
                    offset += page_size
                    if offset >= total or not documents:
                        break
            # Preserve discovery order while removing accidental duplicates.
            document_ids = list(dict.fromkeys(document_ids))
            if not document_ids:
                # Nothing to rebuild — surface an explicit no-op rather than a
                # confusing 400 from the batch-plan min-length guard.
                return DocumentBatchResponse(
                    job_id="",
                    batch_id="",
                    documents=[],
                )
            return await _start_batch_build_job(
                kb_id=kb_id,
                request_ids=document_ids,
                force_rechunk=request.force_rechunk,
                force_extract=request.force_extract,
                force_embedding=request.force_embedding,
                idempotency_key=request.idempotency_key,
                background_tasks=background_tasks,
            )
        except HTTPException:
            raise
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except IdempotencyKeyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("Failed to rebuild KB '%s': %s", kb_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post(
        "/{kb_id}/jobs/{job_id}:cancel",
        response_model=JobResponse,
        dependencies=[Depends(combined_auth)],
        summary="Cancel a queued or running job",
    )
    async def cancel_job(kb_id: str, job_id: str):
        try:
            job, _changed = await job_service.cancel_job(kb_id, job_id)
            return JobResponse.from_record(job)
        except (KnowledgeBaseNotFoundError, MetadataRecordNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except InvalidJobTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post(
        "/{kb_id}/jobs/{job_id}:retry",
        response_model=JobResponse,
        dependencies=[Depends(combined_auth)],
        summary="Retry a failed or cancelled job",
    )
    async def retry_job(
        kb_id: str,
        job_id: str,
        request: JobRetryRequest = Body(default_factory=JobRetryRequest),
    ):
        try:
            job = await job_service.retry_job(
                kb_id, job_id, new_idempotency_key=request.idempotency_key
            )
            return JobResponse.from_record(job)
        except (KnowledgeBaseNotFoundError, MetadataRecordNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except InvalidJobTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get(
        "/{kb_id}/documents/{document_id}/artifacts",
        response_model=ArtifactListResponse,
        dependencies=[Depends(combined_auth)],
        summary="List knowledge base document artifacts",
    )
    async def list_document_artifacts(
        kb_id: str,
        document_id: str,
        artifact_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ):
        try:
            artifacts, total = await document_service.list_document_artifacts(
                kb_id,
                document_id,
                artifact_type=artifact_type,
                limit=limit,
                offset=offset,
            )
            return ArtifactListResponse(
                artifacts=[ArtifactResponse.from_record(item) for item in artifacts],
                total=total,
                limit=max(1, min(limit, 200)),
                offset=max(0, offset),
            )
        except (KnowledgeBaseNotFoundError, MetadataRecordNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get(
        "/{kb_id}/documents/{document_id}/artifacts/{artifact_id}:download",
        dependencies=[Depends(combined_auth)],
        summary="Download a knowledge base document artifact (file or directory zip)",
    )
    async def download_document_artifact(
        kb_id: str, document_id: str, artifact_id: str
    ):
        try:
            artifact_file = await document_service.get_document_artifact_file(
                kb_id, document_id, artifact_id
            )
            if artifact_file.is_directory:
                return _stream_directory_as_zip(artifact_file)
            return FileResponse(
                artifact_file.path,
                media_type=artifact_file.media_type,
                filename=artifact_file.filename,
            )
        except (KnowledgeBaseNotFoundError, MetadataRecordNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get(
        "/{kb_id}/documents/{document_id}/artifacts/{artifact_id}",
        response_model=ArtifactResponse,
        dependencies=[Depends(combined_auth)],
        summary="Get knowledge base document artifact details",
    )
    async def get_document_artifact(kb_id: str, document_id: str, artifact_id: str):
        try:
            return ArtifactResponse.from_record(
                await document_service.get_document_artifact(
                    kb_id, document_id, artifact_id
                )
            )
        except (KnowledgeBaseNotFoundError, MetadataRecordNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get(
        "/{kb_id}/jobs",
        response_model=JobListResponse,
        dependencies=[Depends(combined_auth)],
        summary="List knowledge base jobs",
    )
    async def list_jobs(
        kb_id: str,
        status: Optional[str] = None,
        document_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ):
        try:
            statuses = (status,) if status else None
            jobs, total = await job_service.list_jobs(
                kb_id,
                statuses=statuses,
                document_id=document_id,
                limit=limit,
                offset=offset,
            )
            return JobListResponse(
                jobs=[JobResponse.from_record(item) for item in jobs],
                total=total,
                limit=max(1, min(limit, 200)),
                offset=max(0, offset),
            )
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get(
        "/{kb_id}/jobs/{job_id}",
        response_model=JobResponse,
        dependencies=[Depends(combined_auth)],
        summary="Get knowledge base job details",
    )
    async def get_job(kb_id: str, job_id: str):
        try:
            return JobResponse.from_record(await job_service.get_job(kb_id, job_id))
        except (KnowledgeBaseNotFoundError, MetadataRecordNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post(
        "/{kb_id}/jobs/{job_id}:wait",
        response_model=JobResponse,
        dependencies=[Depends(combined_auth)],
        summary="Wait for a job to reach a terminal state",
    )
    async def wait_for_job(
        kb_id: str,
        job_id: str,
        timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 0.5,
    ):
        # Server-side polling helper so clients can write linear scripts
        # (upload -> wait -> build -> wait -> query) without hand-rolling
        # their own retry loop. Returns 408 once the timeout elapses;
        # otherwise returns the final job snapshot.
        terminal_states = {"succeeded", "failed", "cancelled"}
        timeout_seconds = max(0.1, min(timeout_seconds, 600.0))
        poll_interval_seconds = max(0.05, min(poll_interval_seconds, 5.0))
        deadline = asyncio.get_event_loop().time() + timeout_seconds
        try:
            while True:
                job = await job_service.get_job(kb_id, job_id)
                if job.status in terminal_states:
                    return JobResponse.from_record(job)
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    raise HTTPException(
                        status_code=408,
                        detail={
                            "error_code": "wait_timeout",
                            "job_id": job.id,
                            "current_status": job.status,
                            "message": (
                                f"Job '{job_id}' did not reach a terminal state "
                                f"within {timeout_seconds}s (current: {job.status})"
                            ),
                        },
                    )
                await asyncio.sleep(min(poll_interval_seconds, remaining))
        except (KnowledgeBaseNotFoundError, MetadataRecordNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router
