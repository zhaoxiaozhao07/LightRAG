from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Body, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from lightrag.api.config import global_args
from lightrag.api.document_lifecycle_service import (
    DocumentLifecycleService,
    DocumentSourceInput,
    build_text_source,
)
from lightrag.api.index_build_service import (
    IndexBuildPlan,
    IndexBuildService,
)
from lightrag.api.job_service import JobService
from lightrag.api.kb_service import KnowledgeBaseNotFoundError
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry
from lightrag.api.metadata_store import (
    ActiveDocumentBuildJobError,
    ActiveDocumentParseJobError,
    ArtifactRecord,
    DocumentNotParsedError,
    DocumentRecord,
    IdempotencyKeyConflictError,
    InvalidJobTransitionError,
    JobRecord,
    MetadataRecordNotFoundError,
)
from lightrag.api.routers.document_routes import SUPPORTED_DOCUMENT_EXTENSIONS
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.utils import logger

_UPLOAD_CHUNK_SIZE = 1024 * 1024
_MAX_KB_UPLOAD_FILES = 32
_MAX_KB_TEXT_DOCUMENTS = 100
_MAX_KB_BATCH_PARSE_DOCUMENTS = 100
_MAX_TEXT_DOCUMENT_BYTES = 1024 * 1024
_MAX_TEXT_METADATA_BYTES = 64 * 1024
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
    "process_options",
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
    def limit_metadata_size(
        cls, value: dict[str, Any] | None
    ) -> dict[str, Any] | None:
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
                "Document metadata contains reserved key(s): " + ", ".join(reserved_keys)
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
        await document_service.mark_parse_running(kb_id, plan.document.id, job_id=job_id)
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
                documents=[DocumentResponse.from_record(item) for item in result.documents],
            )
        except HTTPException:
            raise
        except IdempotencyKeyConflictError as exc:
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
                documents=[DocumentResponse.from_record(item) for item in result.documents],
            )
        except HTTPException:
            raise
        except IdempotencyKeyConflictError as exc:
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
                raise HTTPException(status_code=400, detail="metadata must be an object")
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
                        DocumentResponse.from_record(item) for item in existing_documents
                    ],
                )
            queued_documents, claim_failures = await document_service.claim_batch_parse_queued(
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
                            raise RuntimeError("KB parse service did not return a LightRAG instance")
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
                        error_code=None if failed_items == 0 else "partial_parse_failed",
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
                documents=[DocumentResponse.from_record(item) for item in queued_documents],
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
            except ActiveDocumentParseJobError as exc:
                await job_service.transition_job(
                    kb_id,
                    job.id,
                    status="failed",
                    progress=1.0,
                    failed_items=1,
                    error_code="parse_job_active",
                    error_message=str(exc),
                )
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error_code": "parse_job_active",
                        "document_id": exc.document_id,
                        "existing_job_id": exc.existing_job_id,
                        "message": str(exc),
                    },
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

        try:
            await active_index_service.claim_build_queued(kb_id, job_id=job.id, plan=plan)
        except ActiveDocumentBuildJobError as exc:
            await job_service.transition_job(
                kb_id,
                job.id,
                status="failed",
                progress=1.0,
                failed_items=1,
                error_code="build_job_active",
                error_message=str(exc),
            )
            raise HTTPException(
                status_code=409,
                detail={
                    "error_code": "build_job_active",
                    "document_id": exc.document_id,
                    "existing_job_id": exc.existing_job_id,
                    "message": str(exc),
                },
            ) from exc

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
        queued_documents, claim_failures = await active_index_service.claim_batch_build_queued(
            kb_id, job_id=job.id, plans=active_plans
        )
        queued_document_ids = {document.id for document in queued_documents}
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
            logger.error(
                "Failed to start batch build_kg for KB '%s': %s", kb_id, exc
            )
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
        response_class=FileResponse,
        dependencies=[Depends(combined_auth)],
        summary="Download a knowledge base document artifact file",
    )
    async def download_document_artifact(
        kb_id: str, document_id: str, artifact_id: str
    ):
        try:
            artifact_file = await document_service.get_document_artifact_file(
                kb_id, document_id, artifact_id
            )
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
    async def get_document_artifact(
        kb_id: str, document_id: str, artifact_id: str
    ):
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

    return router
