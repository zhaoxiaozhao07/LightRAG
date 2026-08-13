from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import ipaddress
import json
import socket
import zipfile
from email.message import Message
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, cast
from urllib.parse import unquote, urlsplit, urlunsplit

import httpx
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
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
    _same_job_execution_identity,
)
from lightrag.api.routers.document_routes import SUPPORTED_DOCUMENT_EXTENSIONS
from lightrag.api.enterprise_auth import (
    append_enterprise_audit_event,
    enterprise_artifact_min_role_for_type,
    enterprise_auth_enabled,
    enterprise_mask_storage_uris,
    get_enterprise_authorization_service,
    get_request_principal,
)
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.utils import generate_track_id, logger

_UPLOAD_CHUNK_SIZE = 1024 * 1024
_MAX_KB_UPLOAD_FILES = 32
_MAX_KB_TEXT_DOCUMENTS = 100
_MAX_KB_BATCH_PARSE_DOCUMENTS = 100
_MAX_KB_SCAN_FILES = 1000
_MAX_SYNC_SOURCE_KEY_BYTES = 1024
_MAX_TEXT_DOCUMENT_BYTES = 1024 * 1024
_MAX_TEXT_METADATA_BYTES = 64 * 1024
_MAX_DIRECTORY_ARTIFACT_BYTES = 512 * 1024 * 1024  # 512 MB cap on directory zip
_MAX_ARTIFACT_PREVIEW_BYTES = 10 * 1024 * 1024
_MAX_PRESIGNED_URL_EXPIRES_SECONDS = 7 * 24 * 60 * 60
_URL_FETCH_TIMEOUT_SECONDS = 20.0
_PREVIEW_TEXT_MEDIA_TYPES = {
    "application/json",
    "application/ld+json",
    "application/markdown",
    "application/x-ndjson",
    "text/html",
    "text/markdown",
    "text/plain",
}
_PREVIEW_BINARY_MEDIA_TYPES = {"application/pdf"}
_PREVIEW_BLOCKED_MEDIA_TYPES = {"image/svg+xml"}


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
    zip_name = artifact_file.filename
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_name}"'},
    )


def _artifact_audit_metadata(artifact: ArtifactRecord, **extra: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "artifact_id": artifact.id,
        "document_id": artifact.document_id,
        "artifact_type": artifact.artifact_type,
        "size_bytes": artifact.size_bytes,
    }
    metadata.update({key: value for key, value in extra.items() if value is not None})
    return metadata


def _masked_storage_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    masked = _mask_storage_uris(metadata)
    return masked if isinstance(masked, dict) else {}


def _mask_storage_uris(value: Any) -> Any:
    if isinstance(value, dict):
        masked: dict[str, Any] = {}
        for key, item in value.items():
            if key in _STORAGE_URI_KEYS:
                continue
            masked[key] = _mask_storage_uris(item)
        return masked
    if isinstance(value, list):
        return [_mask_storage_uris(item) for item in value]
    return value


_STORAGE_URI_KEYS = {
    "source_uri",
    "source_object_uri",
    "object_uri",
    "object_prefix_uri",
    "blocks_path",
    "local_path",
    "path",
}


async def _enforce_artifact_content_policy(
    request: Request,
    kb_id: str,
    artifact: ArtifactRecord,
    *,
    action: str,
) -> None:
    if not enterprise_auth_enabled():
        return
    normalized_action = action.strip().replace("_", "-")
    principal = get_request_principal(request)
    authz = get_enterprise_authorization_service(request)
    min_role = enterprise_artifact_min_role_for_type(
        artifact.artifact_type,
        action=normalized_action,
    )
    await authz.require_kb_role(
        principal,
        kb_id,
        min_role,
    )
    # Download and presign endpoints export bytes and always require the
    # user-global capability. Preview requires it only for the original source,
    # because derived preview artifacts remain safe viewer-level surfaces.
    if normalized_action in {"download", "download-url"} or (
        normalized_action == "preview" and artifact.artifact_type == "original"
    ):
        authz.require_file_download(principal)


def _document_audit_metadata(
    *,
    job: JobRecord | None = None,
    operation: str,
    document_ids: Sequence[str] = (),
    document_count: int | None = None,
    batch_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"operation": operation}
    if job is not None:
        metadata["job_id"] = job.id
        metadata["job_type"] = job.job_type
    if batch_id is not None:
        metadata["batch_id"] = batch_id
    if document_ids:
        ids = list(document_ids)
        metadata["document_ids"] = ids
        metadata["document_count"] = document_count if document_count is not None else len(ids)
    elif document_count is not None:
        metadata["document_count"] = document_count
    metadata.update({key: value for key, value in extra.items() if value is not None})
    return metadata


async def _append_kb_document_audit_event(
    request: Request,
    event_type: str,
    kb_id: str,
    metadata: dict[str, Any],
) -> None:
    await append_enterprise_audit_event(
        request,
        event_type,
        target_type="kb",
        target_id=kb_id,
        metadata=metadata,
    )


def _is_previewable_media_type(media_type: str) -> bool:
    base_media_type = media_type.split(";", 1)[0].lower()
    if base_media_type in _PREVIEW_BLOCKED_MEDIA_TYPES:
        return False
    return (
        base_media_type in _PREVIEW_TEXT_MEDIA_TYPES
        or base_media_type in _PREVIEW_BINARY_MEDIA_TYPES
        or base_media_type.startswith("image/")
    )


def _artifact_preview_response(artifact_file: Any) -> FileResponse:
    if artifact_file.is_directory or artifact_file.path.is_dir():
        raise HTTPException(status_code=400, detail="Directory artifacts cannot be previewed")
    media_type = artifact_file.media_type
    if not _is_previewable_media_type(media_type):
        raise HTTPException(
            status_code=415,
            detail=f"Artifact media type is not supported for preview: {media_type}",
        )
    try:
        size = artifact_file.path.stat().st_size
    except OSError as exc:
        raise FileNotFoundError(f"Artifact file not found: {artifact_file.path}") from exc
    if size > _MAX_ARTIFACT_PREVIEW_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                "Artifact preview exceeds maximum size of "
                f"{_MAX_ARTIFACT_PREVIEW_BYTES // (1024 * 1024)}MB"
            ),
        )
    headers = {"X-Content-Type-Options": "nosniff"}
    if media_type.split(";", 1)[0].lower() == "text/html":
        headers["Content-Security-Policy"] = (
            "default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'"
        )
    return FileResponse(
        artifact_file.path,
        media_type=media_type,
        filename=artifact_file.filename,
        content_disposition_type="inline",
        headers=headers,
    )


_RESERVED_DOCUMENT_METADATA_KEYS = {
    "artifact_count",
    "auto_index",
    "auto_parse",
    "batch_id",
    "blocks_path",
    "build_skipped",
    "build_skip_reason",
    "build_started_at",
    "created_by",
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
        data = record.to_dict()
        if enterprise_mask_storage_uris():
            data["source_uri"] = "<masked>"
            data["metadata"] = _masked_storage_metadata(data.get("metadata") or {})
        return cls(**data)


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
        data = record.to_dict()
        if enterprise_mask_storage_uris():
            data["uri"] = "<masked>"
            data["metadata"] = _masked_storage_metadata(data.get("metadata") or {})
        return cls(**data)


class ArtifactListResponse(BaseModel):
    artifacts: list[ArtifactResponse]
    total: int
    limit: int
    offset: int


class ArtifactDownloadUrlResponse(BaseModel):
    artifact_id: str
    url: str
    object_uri: str
    expires_in_seconds: int
    filename: str
    media_type: str


class DocumentPreviewVariantResponse(BaseModel):
    kind: str
    artifact_id: str
    artifact_type: str
    media_type: str
    size_bytes: Optional[int]
    preview_url: str


class DocumentPreviewFallbackResponse(BaseModel):
    artifact_id: str
    artifact_type: str
    media_type: str
    size_bytes: Optional[int]
    download_url: str


class DocumentPreviewManifestResponse(BaseModel):
    document_id: str
    source_name: str
    source_content_type: Optional[str]
    status: str
    preferred: Optional[DocumentPreviewVariantResponse]
    variants: list[DocumentPreviewVariantResponse]
    fallback: Optional[DocumentPreviewFallbackResponse]


class ParseDocumentRequest(BaseModel):
    engine: Optional[str] = None
    process_options: Optional[str] = None
    force_reparse: bool = False
    # Parse-only endpoint (see route summary "without building the index"):
    # auto_index is a reserved no-op and never triggers a build, so the
    # in-process path and a durable-worker resume behave identically. Use
    # :build-kg / :batch-build-kg to build the index after parsing.
    auto_index: bool = False
    idempotency_key: Optional[str] = None


class BatchParseDocumentsRequest(BaseModel):
    document_ids: list[str] = Field(
        min_length=1, max_length=_MAX_KB_BATCH_PARSE_DOCUMENTS
    )
    engine: Optional[str] = None
    process_options: Optional[str] = None
    force_reparse: bool = False
    # Parse-only endpoint (see route summary "without building the index"):
    # auto_index is a reserved no-op and never triggers a build, so the
    # in-process path and a durable-worker resume behave identically. Use
    # :batch-build-kg to build the index after parsing.
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
    delete_graph_orphans: bool = True
    strategy: Literal["safe", "rebuild_doc_scope", "rebuild_kb", "rebuild_subgraph"] = "safe"
    idempotency_key: Optional[str] = None

    @field_validator("document_ids", mode="after")
    @classmethod
    def reject_duplicate_document_ids(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("Duplicate document_ids are not allowed")
        return value


class BatchSetDocumentsEnabledRequest(BaseModel):
    document_ids: list[str] = Field(
        min_length=1, max_length=_MAX_KB_BATCH_PARSE_DOCUMENTS
    )

    @field_validator("document_ids", mode="after")
    @classmethod
    def reject_duplicate_document_ids(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("Duplicate document_ids are not allowed")
        return value


class BatchSetDocumentsEnabledItem(BaseModel):
    document_id: str
    status: Literal["updated", "not_found"]


class BatchSetDocumentsEnabledResponse(BaseModel):
    enabled: bool
    updated: int
    not_found: int
    items: list[BatchSetDocumentsEnabledItem]


class DocumentChunkItem(BaseModel):
    id: str
    chunk_order_index: Optional[int] = None
    tokens: Optional[int] = None
    content: Optional[str] = None
    file_path: Optional[str] = None


class DocumentChunksResponse(BaseModel):
    kb_id: str
    document_id: str
    lightrag_doc_id: Optional[str] = None
    total: int
    limit: int
    offset: int
    chunks: list[DocumentChunkItem]


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
        _validate_user_metadata_size(value, label="Text document metadata")
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


def _validate_user_metadata_size(value: dict[str, Any], *, label: str) -> None:
    size = len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
    if size > _MAX_TEXT_METADATA_BYTES:
        raise ValueError(
            f"{label} too large. Maximum size: {_MAX_TEXT_METADATA_BYTES} bytes"
        )
    reserved_keys = sorted(set(value) & _RESERVED_DOCUMENT_METADATA_KEYS)
    if reserved_keys:
        raise ValueError(
            f"{label} contains reserved key(s): " + ", ".join(reserved_keys)
        )


class UrlDocumentRequest(BaseModel):
    url: str = Field(min_length=1)
    source_name: Optional[str] = None
    source_key: Optional[str] = None
    content_type: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="after")
    @classmethod
    def limit_metadata_size(cls, value: dict[str, Any]) -> dict[str, Any]:
        _validate_user_metadata_size(value, label="URL document metadata")
        return value


class UrlDocumentsRequest(BaseModel):
    documents: list[UrlDocumentRequest] = Field(
        min_length=1, max_length=_MAX_KB_UPLOAD_FILES
    )
    auto_parse: bool = False
    auto_index: bool = False
    parser_engine: Optional[str] = None
    process_options: Optional[str] = None
    idempotency_key: Optional[str] = None


class LocalImportDocumentRequest(BaseModel):
    path: str = Field(min_length=1)
    source_name: Optional[str] = None
    source_key: Optional[str] = None
    content_type: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="after")
    @classmethod
    def limit_metadata_size(cls, value: dict[str, Any]) -> dict[str, Any]:
        _validate_user_metadata_size(value, label="Import document metadata")
        return value


class LocalImportDocumentsRequest(BaseModel):
    documents: list[LocalImportDocumentRequest] = Field(
        min_length=1, max_length=_MAX_KB_UPLOAD_FILES
    )
    auto_parse: bool = False
    auto_index: bool = False
    parser_engine: Optional[str] = None
    process_options: Optional[str] = None
    idempotency_key: Optional[str] = None


class LocalScanDocumentsRequest(BaseModel):
    directory: str = Field(min_length=1)
    recursive: bool = False
    source_key_prefix: str = "scan"
    max_files: int = Field(default=_MAX_KB_UPLOAD_FILES, ge=1, le=_MAX_KB_SCAN_FILES)
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
        data = record.to_dict()
        if enterprise_mask_storage_uris():
            data["payload"] = _mask_storage_uris(data.get("payload") or {})
            if data.get("result") is not None:
                data["result"] = _mask_storage_uris(data["result"])
        return cls(**data)


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


def _merge_source_metadata(
    user_metadata: dict[str, Any], required_metadata: dict[str, Any]
) -> dict[str, Any]:
    return {**user_metadata, **required_metadata}


def _normalized_url(value: str) -> tuple[str, str]:
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="URL scheme must be http or https")
    if not parsed.hostname:
        raise HTTPException(status_code=400, detail="URL must include a hostname")
    if parsed.username or parsed.password:
        raise HTTPException(status_code=400, detail="URL userinfo is not allowed")
    hostname = parsed.hostname.lower()
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{hostname}:{parsed.port}"
    normalized = urlunsplit(
        (parsed.scheme.lower(), netloc, parsed.path or "/", parsed.query, "")
    )
    return normalized, hostname


def _is_public_ip_address(address: str) -> bool:
    try:
        ip_address = ipaddress.ip_address(address)
    except ValueError:
        return False
    return not (
        ip_address.is_loopback
        or ip_address.is_private
        or ip_address.is_link_local
        or ip_address.is_multicast
        or ip_address.is_reserved
        or ip_address.is_unspecified
    )


async def _validate_public_hostname(hostname: str) -> None:
    try:
        addrinfos = await asyncio.get_running_loop().getaddrinfo(
            hostname, None, type=socket.SOCK_STREAM
        )
    except socket.gaierror as exc:
        raise HTTPException(
            status_code=400, detail="URL hostname could not be resolved"
        ) from exc
    addresses = {item[4][0] for item in addrinfos}
    if not addresses or any(not _is_public_ip_address(address) for address in addresses):
        raise HTTPException(
            status_code=400,
            detail="URL hostname resolves to a disallowed private or local address",
        )


def _validate_url_response_peer(response: httpx.Response) -> None:
    network_stream = response.extensions.get("network_stream")
    get_extra_info = getattr(network_stream, "get_extra_info", None)
    peer_address: str | None = None
    if callable(get_extra_info):
        server_addr = get_extra_info("server_addr")
        if isinstance(server_addr, tuple) and server_addr:
            peer_address = str(server_addr[0])
        elif isinstance(server_addr, str):
            peer_address = server_addr
    if peer_address is None:
        raise HTTPException(
            status_code=502,
            detail="URL fetch peer address could not be verified",
        )
    if not _is_public_ip_address(peer_address):
        raise HTTPException(
            status_code=400,
            detail="URL fetch peer resolves to a disallowed private or local address",
        )


def _filename_from_content_disposition(value: str | None) -> str | None:
    if not value:
        return None
    message = Message()
    message["content-disposition"] = value
    filename = message.get_filename()
    if not filename:
        return None
    return Path(unquote(filename)).name


def _safe_url_source_name(
    *, explicit_source_name: str | None, url: str, content_disposition: str | None
) -> str:
    candidates = [
        explicit_source_name,
        _filename_from_content_disposition(content_disposition),
        Path(unquote(urlsplit(url).path)).name,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        source_name = Path(candidate).name.strip()
        if not source_name:
            continue
        if _is_supported_upload_name(source_name):
            return source_name
        if explicit_source_name == candidate:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported file type. Supported types: "
                    f"{SUPPORTED_DOCUMENT_EXTENSIONS}"
                ),
            )
    raise HTTPException(
        status_code=400,
        detail=(
            "URL source_name is required when the URL or response filename does "
            "not use a supported extension"
        ),
    )


async def _fetch_url_document(
    document: UrlDocumentRequest,
    *,
    max_upload_size: int,
    remaining_batch_bytes: int,
) -> tuple[DocumentSourceInput, int]:
    normalized_url, hostname = _normalized_url(document.url)
    await _validate_public_hostname(hostname)
    if remaining_batch_bytes <= 0:
        raise HTTPException(
            status_code=413,
            detail=_batch_too_large_detail(max_upload_size, max_upload_size + 1),
        )

    content = bytearray()
    try:
        async with httpx.AsyncClient(
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(_URL_FETCH_TIMEOUT_SECONDS),
        ) as client:
            async with client.stream("GET", normalized_url) as response:
                _validate_url_response_peer(response)
                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        declared_size = int(content_length)
                    except ValueError as exc:
                        raise HTTPException(
                            status_code=400, detail="Invalid Content-Length header"
                        ) from exc
                    if declared_size > max_upload_size:
                        raise HTTPException(
                            status_code=413,
                            detail=_file_too_large_detail(max_upload_size, declared_size),
                        )
                    if declared_size > remaining_batch_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=_batch_too_large_detail(
                                max_upload_size,
                                max_upload_size - remaining_batch_bytes + declared_size,
                            ),
                        )
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    status_code = exc.response.status_code
                    raise HTTPException(
                        status_code=400 if 300 <= status_code < 400 else 502,
                        detail=f"URL fetch failed with HTTP status {status_code}",
                    ) from exc
                source_name = _safe_url_source_name(
                    explicit_source_name=document.source_name,
                    url=normalized_url,
                    content_disposition=response.headers.get("content-disposition"),
                )
                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue
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
    except HTTPException:
        raise
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"URL fetch failed: {exc}") from exc

    source_key = _normalize_sync_source_key(document.source_key or f"url:{normalized_url}")
    return (
        DocumentSourceInput(
            source_name=source_name,
            content=bytes(content),
            source_type="url",
            content_type=document.content_type,
            metadata=_merge_source_metadata(
                document.metadata,
                {"source_url": normalized_url, "source_key": source_key},
            ),
        ),
        len(content),
    )


def _source_root_resolved(document_service: DocumentLifecycleService) -> Path:
    return document_service.source_root.resolve(strict=False)


def _resolve_staged_path(source_root: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = source_root / candidate
    if candidate.is_symlink():
        raise HTTPException(status_code=400, detail="Staged symlinks are not allowed")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise HTTPException(status_code=400, detail="Staged path does not exist") from exc
    try:
        resolved.relative_to(source_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Staged path escapes INPUT_DIR") from exc
    return resolved


def _requested_path_resolves_to_root(source_root: Path, value: str) -> bool:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = source_root / candidate
    return candidate.resolve(strict=False) == source_root


def _relative_staged_path(source_root: Path, path: Path) -> str:
    return path.relative_to(source_root).as_posix()


def _validate_staged_file(path: Path, *, source_name: str | None = None) -> None:
    if path.is_symlink():
        raise HTTPException(status_code=400, detail="Staged file symlinks are not allowed")
    if path.is_dir():
        raise HTTPException(status_code=400, detail="Staged path must be a file")
    if not path.is_file():
        raise HTTPException(status_code=400, detail="Staged path must be a file")
    if not _is_supported_upload_name(source_name or path.name):
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. Supported types: "
                f"{SUPPORTED_DOCUMENT_EXTENSIONS}"
            ),
        )


def _read_staged_file_content(
    path: Path,
    *,
    max_upload_size: int,
    remaining_batch_bytes: int,
) -> bytes:
    if remaining_batch_bytes <= 0:
        raise HTTPException(
            status_code=413,
            detail=_batch_too_large_detail(max_upload_size, max_upload_size + 1),
        )
    size = path.stat().st_size
    if size <= 0:
        raise HTTPException(status_code=400, detail="Staged file cannot be empty")
    if size > max_upload_size:
        raise HTTPException(
            status_code=413, detail=_file_too_large_detail(max_upload_size, size)
        )
    if size > remaining_batch_bytes:
        raise HTTPException(
            status_code=413,
            detail=_batch_too_large_detail(
                max_upload_size, max_upload_size - remaining_batch_bytes + size
            ),
        )
    return path.read_bytes()


def _scan_supported_files(
    source_root: Path,
    directory: Path,
    *,
    recursive: bool,
    max_files: int,
) -> list[Path]:
    try:
        directory.relative_to(source_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Scan directory escapes INPUT_DIR") from exc
    if directory == source_root:
        raise HTTPException(status_code=400, detail="Scan directory cannot be INPUT_DIR root")
    if directory.is_symlink():
        raise HTTPException(status_code=400, detail="Scan directory symlinks are not allowed")
    if not directory.is_dir():
        raise HTTPException(status_code=400, detail="Scan directory must be a directory")

    iterator = directory.rglob("*") if recursive else directory.glob("*")
    files: list[Path] = []
    for path in sorted(iterator):
        relative_parts = path.relative_to(source_root).parts
        if "__parsed__" in relative_parts or ".sync-staging" in relative_parts:
            continue
        if path.is_dir():
            if path.is_symlink():
                raise HTTPException(status_code=400, detail="Scan directory symlinks are not allowed")
            continue
        if path.is_symlink() or not path.is_file():
            continue
        if not _is_supported_upload_name(path.name):
            continue
        files.append(path)
        if len(files) > max_files:
            raise HTTPException(
                status_code=413,
                detail=f"Scan matched too many files. Maximum files: {max_files}",
            )
    if not files:
        raise HTTPException(status_code=400, detail="Scan found no supported files")
    return files


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


async def _job_is_cancelling(
    job_service: "JobService | None", kb_id: str, job_id: str
) -> bool:
    """Cooperative-cancellation checkpoint.

    A running job whose status has been flipped to ``cancelling`` by the
    ``:cancel`` endpoint should stop at the next safe boundary. In-process
    background tasks (and the durable worker) call this before starting the
    expensive parse/build stage; if it returns True the executor releases the
    document claim and reports a ``cancelled`` item instead of running.
    """
    if job_service is None:
        return False
    try:
        job = await job_service.get_job(kb_id, job_id)
    except Exception:  # noqa: BLE001 — never let a status probe break execution
        return False
    return job.status == "cancelling"


async def _cancel_parse_item(
    document_service: DocumentLifecycleService,
    *,
    kb_id: str,
    job_id: str,
    plan: Any,
) -> dict[str, Any]:
    """Release a parse claim when cancelled at a checkpoint (doc -> parse_failed,
    recoverable via :retry)."""
    try:
        await document_service.fail_parse(
            kb_id,
            plan.document.id,
            job_id=job_id,
            plan=plan,
            error_code="cancelled_by_user",
            error_message="Parse cancelled before execution",
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to release parse claim for cancelled doc '%s': %s",
            plan.document.id,
            exc,
        )
    return {
        "document_id": plan.document.id,
        "status": "cancelled",
        "error_code": "cancelled_by_user",
        "error_message": "Parse cancelled before execution",
    }


async def _cancel_build_item(
    index_service: IndexBuildService,
    *,
    kb_id: str,
    job_id: str,
    plan: IndexBuildPlan,
) -> dict[str, Any]:
    """Release a build claim when cancelled at a checkpoint (doc -> build_failed,
    recoverable via :retry)."""
    try:
        await index_service.fail_build(
            kb_id,
            plan.document.id,
            job_id=job_id,
            error_code="cancelled_by_user",
            error_message="Build cancelled before execution",
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to release build claim for cancelled doc '%s': %s",
            plan.document.id,
            exc,
        )
    return {
        "document_id": plan.document.id,
        "status": "cancelled",
        "error_code": "cancelled_by_user",
        "error_message": "Build cancelled before execution",
    }


async def _run_parse_with_forced_cancel(
    *,
    document_service: DocumentLifecycleService,
    job_service: "JobService | None",
    kb_id: str,
    job_id: str,
    plan: Any,
    rag: Any,
    poll_interval: float = 0.25,
) -> dict[str, Any]:
    """Run the parse (MinerU/native/docling) await, force-cancellable mid-flight.

    The parse stage is the one long single ``await`` in the document pipeline
    (a MinerU/Docling call can run for minutes). Stage-boundary cooperative
    cancellation cannot interrupt it once entered. Here we run ``run_parse`` as
    its own ``asyncio.Task`` and concurrently poll the job status; when the job
    flips to ``cancelling`` we ``cancel()`` the in-flight task and stop.

    This is **safe for the parse stage specifically** because parse is
    idempotent: it writes a MinerU/Docling raw bundle + sidecar that a re-run
    simply overwrites, and on cancel we reset the document to ``parse_failed``
    (recoverable via ``:retry``). It is deliberately NOT applied to the
    KG-build / vector-upsert stages, where a mid-``await`` interrupt could leave
    a half-merged graph or partially-written vectors. Returns the parsed-data
    dict on success; raises :class:`asyncio.CancelledError` when force-cancelled.

    When ``job_service`` is None (no way to observe cancellation) it degrades to
    a plain ``await`` with no polling overhead.
    """
    if job_service is None:
        return await document_service.run_parse(rag, plan)

    parse_task = asyncio.ensure_future(document_service.run_parse(rag, plan))
    try:
        while True:
            done, _pending = await asyncio.wait({parse_task}, timeout=poll_interval)
            if parse_task in done:
                return parse_task.result()
            if await _job_is_cancelling(job_service, kb_id, job_id):
                parse_task.cancel()
                try:
                    await parse_task
                except asyncio.CancelledError:
                    pass
                raise asyncio.CancelledError()
    except asyncio.CancelledError:
        if not parse_task.done():
            parse_task.cancel()
            try:
                await parse_task
            except asyncio.CancelledError:
                pass
        raise


async def _execute_parse_plan(
    *,
    document_service: DocumentLifecycleService,
    kb_id: str,
    job_id: str,
    plan: Any,
    rag: Any,
    job_service: "JobService | None" = None,
) -> dict[str, Any]:
    try:
        if await _job_is_cancelling(job_service, kb_id, job_id):
            return await _cancel_parse_item(
                document_service, kb_id=kb_id, job_id=job_id, plan=plan
            )
        await document_service.mark_parse_running(
            kb_id, plan.document.id, job_id=job_id
        )
        try:
            parsed_data = await _run_parse_with_forced_cancel(
                document_service=document_service,
                job_service=job_service,
                kb_id=kb_id,
                job_id=job_id,
                plan=plan,
                rag=rag,
            )
        except asyncio.CancelledError:
            # Force-cancelled mid-parse: release the claim as a recoverable
            # parse_failed and report a cancelled item (not a hard failure).
            return await _cancel_parse_item(
                document_service, kb_id=kb_id, job_id=job_id, plan=plan
            )
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


async def _run_auto_parse_batch(
    *,
    document_service: DocumentLifecycleService,
    job_service: JobService,
    registry: LightRAGInstanceRegistry,
    index_service: "IndexBuildService | None",
    kb_id: str,
    job: JobRecord,
    document_ids: list[str],
    auto_index: bool,
) -> None:
    """Background executor for multi-file ``upload`` / ``texts`` ``auto_parse``.

    ``create_source_batch`` has already created the documents in
    ``parse_queued`` and the aggregate ``parse`` job in ``queued``; this task
    parses each document in parallel (bounded by ``MAX_PARALLEL_PARSE_MINERU``
    / per-engine equivalents) and, when ``auto_index`` is set and an index
    service is configured, bulk-enqueues all successfully-parsed documents
    into the LightRAG pipeline in a single drain so the three worker layers
    (parse / analyze / process) overlap docs instead of serializing them.
    Per-item results aggregate into the single aggregate job, mirroring
    ``documents:batch-parse`` but skipping re-claiming (the docs are already
    claimed at creation).
    """
    item_results: list[dict[str, Any]] = []
    completed_items = 0
    failed_items = 0
    try:
        await job_service.transition_job(
            kb_id,
            job.id,
            status="running",
            progress=0.0,
            result=_batch_parse_job_result(
                batch_id=job.batch_id or "",
                total_items=len(document_ids),
                completed_items=0,
                failed_items=0,
                items=[],
            ),
        )
        rag = await registry.get(kb_id) if document_ids else None
        if rag is None or not document_ids:
            final_status = "succeeded" if failed_items == 0 else "failed"
            await job_service.transition_job(
                kb_id,
                job.id,
                status=final_status,
                progress=1.0,
                completed_items=completed_items,
                failed_items=failed_items,
                result=_batch_parse_job_result(
                    batch_id=job.batch_id or "",
                    total_items=len(document_ids),
                    completed_items=completed_items,
                    failed_items=failed_items,
                    items=item_results,
                ),
            )
            return

        item_by_id: dict[str, dict[str, Any]] = {}

        # ── Phase 1: concurrent parse ──────────────────────────────────────
        parse_concurrency = max(
            1, int(getattr(rag, "max_parallel_parse_mineru", 1) or 1)
        )
        parse_sem = asyncio.Semaphore(parse_concurrency)

        async def _do_one_parse(document_id: str) -> tuple[str, Any, dict[str, Any]]:
            async with parse_sem:
                try:
                    plan = await document_service.create_parse_plan(kb_id, document_id)
                except Exception as exc:  # noqa: BLE001 — per-item planning failure
                    return (
                        document_id,
                        None,
                        {
                            "document_id": document_id,
                            "status": "failed",
                            "error_code": "parse_failed",
                            "error_message": str(exc),
                        },
                    )
                item = await _execute_parse_plan(
                    document_service=document_service,
                    kb_id=kb_id,
                    job_id=job.id,
                    plan=plan,
                    rag=rag,
                    job_service=job_service,
                )
                return document_id, plan, item

        raw_outcomes = await asyncio.gather(
            *[_do_one_parse(d) for d in document_ids],
            return_exceptions=True,
        )
        # return_exceptions=True guarantees Phase 2 always runs (and thus
        # terminalizes / releases any doc already claimed into build_queued)
        # even if a task raised an unexpected BaseException; map any such
        # exception back to its doc as a failed item via positional zip.
        parse_outcomes: list[tuple[str, Any, dict[str, Any]]] = []
        for doc_id, outcome in zip(document_ids, raw_outcomes):
            if isinstance(outcome, BaseException):
                parse_outcomes.append(
                    (
                        doc_id,
                        None,
                        {
                            "document_id": doc_id,
                            "status": "failed",
                            "error_code": "parse_failed",
                            "error_message": str(outcome),
                        },
                    )
                )
            else:
                parse_outcomes.append(outcome)
        for doc_id, plan, item in parse_outcomes:
            item_by_id[doc_id] = item

        # ── Phase 2: bulk auto_index build ────────────────────────────────
        build_plans: list[IndexBuildPlan] = []
        build_plan_to_item: dict[str, dict[str, Any]] = {}
        if auto_index and index_service is not None:
            for doc_id, _parse_plan, item in parse_outcomes:
                if item["status"] != "succeeded":
                    continue
                try:
                    build_plan = await index_service.create_build_plan(
                        kb_id, doc_id, rag=rag
                    )
                    if not build_plan.skipped:
                        await index_service.claim_build_queued(
                            kb_id, job_id=job.id, plan=build_plan
                        )
                except Exception as exc:  # noqa: BLE001 — per-item plan failure
                    item["status"] = "failed"
                    item["error_code"] = "build_failed"
                    item["error_message"] = str(exc)
                    continue
                build_plans.append(build_plan)
                build_plan_to_item[build_plan.document.id] = item

            if build_plans:
                build_results = await _execute_build_plan_batch(
                    index_service=index_service,
                    kb_id=kb_id,
                    job_id=job.id,
                    rag=rag,
                    plans=build_plans,
                    job_service=job_service,
                )
                for plan in build_plans:
                    item = build_plan_to_item[plan.document.id]
                    build_item = build_results.get(plan.document.id)
                    if build_item is None:
                        item["status"] = "failed"
                        item["error_code"] = "build_failed"
                        item["error_message"] = "Build result missing from batch"
                        continue
                    item["build_result"] = build_item
                    if build_item["status"] not in {"succeeded", "cancelled"}:
                        item["status"] = "failed"
                        item["error_code"] = build_item.get("error_code")
                        item["error_message"] = build_item.get("error_message")

        # ── Phase 3: finalize ─────────────────────────────────────────────
        for doc_id in document_ids:
            item = item_by_id[doc_id]
            item_results.append(item)
            if item["status"] == "succeeded":
                completed_items += 1
            else:
                failed_items += 1

        final_status = "succeeded" if failed_items == 0 else "failed"
        await job_service.transition_job(
            kb_id,
            job.id,
            status=final_status,
            progress=1.0,
            completed_items=completed_items,
            failed_items=failed_items,
            result=_batch_parse_job_result(
                batch_id=job.batch_id or "",
                total_items=len(document_ids),
                completed_items=completed_items,
                failed_items=failed_items,
                items=item_results,
            ),
            error_code=None if failed_items == 0 else "partial_parse_failed",
            error_message=None
            if failed_items == 0
            else _batch_parse_failure_message(failed_items, len(document_ids)),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to run auto-parse job '%s' for KB '%s': %s",
            job.id,
            kb_id,
            exc,
        )
        processed_ids = {item["document_id"] for item in item_results}
        for document_id in document_ids:
            if document_id in processed_ids:
                continue
            try:
                fallback_plan = await document_service.create_parse_plan(
                    kb_id, document_id
                )
                await document_service.fail_parse(
                    kb_id,
                    document_id,
                    job_id=job.id,
                    plan=fallback_plan,
                    error_code="parse_failed",
                    error_message=str(exc),
                )
            except Exception as transition_exc:  # noqa: BLE001
                logger.error(
                    "Failed to mark document '%s' failed for auto-parse job '%s': %s",
                    document_id,
                    job.id,
                    transition_exc,
                )
            item_results.append(
                {
                    "document_id": document_id,
                    "status": "failed",
                    "error_code": "parse_failed",
                    "error_message": str(exc),
                }
            )
            failed_items += 1
        try:
            await job_service.transition_job(
                kb_id,
                job.id,
                status="failed",
                progress=1.0,
                completed_items=completed_items,
                failed_items=failed_items,
                result=_batch_parse_job_result(
                    batch_id=job.batch_id or "",
                    total_items=len(document_ids),
                    completed_items=completed_items,
                    failed_items=failed_items,
                    items=item_results,
                ),
                error_code="batch_parse_failed",
                error_message=str(exc),
            )
        except InvalidJobTransitionError:
            logger.warning(
                "Auto-parse job '%s' for KB '%s' was already terminal",
                job.id,
                kb_id,
            )


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
    job_service: "JobService | None" = None,
) -> dict[str, Any]:
    try:
        if await _job_is_cancelling(job_service, kb_id, job_id):
            return await _cancel_build_item(
                index_service, kb_id=kb_id, job_id=job_id, plan=plan
            )
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


@contextlib.asynccontextmanager
async def _mirror_pipeline_progress(
    *,
    job_service: "JobService | None",
    kb_id: str,
    job_id: str,
    rag: Any,
    poll_interval: float = 3.0,
):
    """Mirror the KB's live LightRAG ``pipeline_status`` into the job while a
    long parse/build drain runs, so a client polling the job sees the current
    activity ("Extract entities 120/340") instead of a frozen snapshot.

    Spawns a background task that every ``poll_interval`` seconds reads
    ``pipeline_status`` for the KB's workspace and patches the job's
    ``result["pipeline"]`` (``latest_message`` + ``cur_batch`` / ``batchs`` /
    ``docs``). It deliberately does **not** write ``progress`` /
    ``completed_items`` — those stay owned by the per-document counters in the
    job handlers, so this mirror can wrap any drain without fighting them.
    Fully best-effort: any read/write error is swallowed (observability must
    never break a build) and the task is always cancelled on exit.
    """
    if job_service is None or rag is None:
        yield
        return

    workspace = str(getattr(rag, "workspace", "") or "")
    stop = asyncio.Event()

    async def _poll() -> None:
        from lightrag.kg.shared_storage import get_namespace_data

        last_message: str | None = None
        while not stop.is_set():
            try:
                ps = await get_namespace_data("pipeline_status", workspace=workspace)
                latest = ps.get("latest_message")
                message = str(latest) if latest else None
                if message and message != last_message:
                    last_message = message
                    await job_service.update_job_progress(
                        kb_id,
                        job_id,
                        result_patch={
                            "pipeline": {
                                "latest_message": message,
                                "cur_batch": int(ps.get("cur_batch") or 0),
                                "batchs": int(ps.get("batchs") or 0),
                                "docs": int(ps.get("docs") or 0),
                            }
                        },
                    )
            except Exception:  # noqa: BLE001 — observability never breaks the build
                pass
            try:
                await asyncio.wait_for(stop.wait(), timeout=poll_interval)
            except asyncio.TimeoutError:
                pass

    task = asyncio.create_task(_poll())
    try:
        yield
    finally:
        stop.set()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


async def _execute_build_plan_batch(
    *,
    index_service: IndexBuildService,
    kb_id: str,
    job_id: str,
    rag: Any,
    plans: list[IndexBuildPlan],
    job_service: "JobService | None" = None,
) -> dict[str, dict[str, Any]]:
    """Run :meth:`IndexBuildService.run_build_batch` and finalize each doc.

    This is the batched counterpart of :func:`_execute_build_plan`. It:

    1. Honors a job-level ``cancelling`` request by marking every plan as
       cancelled and returning early — same semantics as the single-doc helper.
    2. Marks each non-skipped plan ``building`` (the single-drain pipeline call
       does not do this itself).
    3. Bulk-enqueues every non-skipped plan into the LightRAG pipeline through
       ``IndexBuildService.run_build_batch`` so the three worker layers
       (parse / analyze / process) can overlap documents instead of being
       serialized per-document.
    4. Per doc, calls ``complete_build`` on success or ``fail_build`` on
       per-doc failure, then assembles the same ``build_result`` shape that
       :func:`_execute_build_plan` produces. Returns a
       ``{kb_document_id: build_result}`` mapping.

    Failures during ``run_build_batch`` itself (e.g. a pipeline-level crash)
    fail every plan in the batch; per-doc failures surfaced by
    ``run_build_batch`` (missing sidecar, ``doc_status`` read errors, etc.)
    fail only that doc.
    """
    if not plans:
        return {}

    results: dict[str, dict[str, Any]] = {}
    if await _job_is_cancelling(job_service, kb_id, job_id):
        for plan in plans:
            results[plan.document.id] = await _cancel_build_item(
                index_service, kb_id=kb_id, job_id=job_id, plan=plan
            )
        return results

    # Mark every non-skipped plan ``building``. A failure here is per-doc and
    # excludes that plan from the bulk enqueue.
    runnable: list[IndexBuildPlan] = []
    for plan in plans:
        if plan.skipped:
            runnable.append(plan)
            continue
        try:
            await index_service.mark_building(kb_id, plan.document.id, job_id=job_id)
        except Exception as exc:  # noqa: BLE001 — record per-doc failure
            logger.error(
                "Failed to mark document '%s' building (KB '%s'): %s",
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
            except Exception:  # noqa: BLE001
                pass
            results[plan.document.id] = {
                "document_id": plan.document.id,
                "status": "failed",
                "error_code": "build_failed",
                "error_message": str(exc),
            }
            continue
        runnable.append(plan)

    if runnable:
        try:
            async with _mirror_pipeline_progress(
                job_service=job_service, kb_id=kb_id, job_id=job_id, rag=rag
            ):
                run_results = await index_service.run_build_batch(
                    rag, runnable, job_id=job_id
                )
        except Exception as exc:  # noqa: BLE001 — batch-level failure
            logger.error(
                "KG build batch failed for KB '%s' (job '%s'): %s",
                kb_id,
                job_id,
                exc,
            )
            for plan in runnable:
                try:
                    await index_service.fail_build(
                        kb_id,
                        plan.document.id,
                        job_id=job_id,
                        error_code="build_failed",
                        error_message=str(exc),
                    )
                except Exception:  # noqa: BLE001
                    pass
                results[plan.document.id] = {
                    "document_id": plan.document.id,
                    "status": "failed",
                    "error_code": "build_failed",
                    "error_message": str(exc),
                }
            return results

        for plan in runnable:
            run_result = run_results.get(plan.document.id)
            if run_result is None:
                msg = "Build result missing from pipeline drain"
                try:
                    await index_service.fail_build(
                        kb_id,
                        plan.document.id,
                        job_id=job_id,
                        error_code="build_failed",
                        error_message=msg,
                    )
                except Exception:  # noqa: BLE001
                    pass
                results[plan.document.id] = {
                    "document_id": plan.document.id,
                    "status": "failed",
                    "error_code": "build_failed",
                    "error_message": msg,
                }
                continue
            if run_result.get("cancelled"):
                # The pipeline marked this doc failed with a user-cancellation
                # marker mid-drain. Release the claim and report it as a clean
                # cancellation (status='cancelled') rather than build_failed,
                # matching the per-doc _execute_build_plan checkpoint behavior.
                results[plan.document.id] = await _cancel_build_item(
                    index_service, kb_id=kb_id, job_id=job_id, plan=plan
                )
                continue
            if "error_code" in run_result:
                err_code = str(run_result.get("error_code") or "build_failed")
                err_msg = str(run_result.get("error_message") or "")
                try:
                    await index_service.fail_build(
                        kb_id,
                        plan.document.id,
                        job_id=job_id,
                        error_code=err_code,
                        error_message=err_msg,
                    )
                except Exception:  # noqa: BLE001
                    pass
                results[plan.document.id] = {
                    "document_id": plan.document.id,
                    "status": "failed",
                    "error_code": err_code,
                    "error_message": err_msg,
                }
                continue
            try:
                result_record = await index_service.complete_build(
                    kb_id,
                    plan.document.id,
                    job_id=job_id,
                    plan=plan,
                    run_result=run_result,
                )
                results[plan.document.id] = {
                    "document_id": result_record.id,
                    "status": "succeeded",
                    "skipped": bool(plan.skipped or run_result.get("skipped")),
                    "skip_reason": plan.skip_reason if plan.skipped else None,
                    "index_hash": plan.index_hash,
                    "chunks_count": result_record.chunks_count,
                    "entity_count": result_record.entity_count,
                    "relation_count": result_record.relation_count,
                }
            except Exception as exc:  # noqa: BLE001 — per-doc complete failure
                logger.error(
                    "Failed to finalize build for document '%s' (KB '%s'): %s",
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
                except Exception:  # noqa: BLE001
                    pass
                results[plan.document.id] = {
                    "document_id": plan.document.id,
                    "status": "failed",
                    "error_code": "build_failed",
                    "error_message": str(exc),
                }
    return results


async def _execute_delete_document_impl(
    *,
    document_service: DocumentLifecycleService,
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
            "file_delete_result": _file_result_payload(file_result),
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
        "deleted_objects": list(getattr(result, "deleted_objects", [])),
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


def _validate_delete_strategy(
    *,
    strategy: str,
    delete_graph_orphans: bool,
    index_service: "IndexBuildService | None",
) -> None:
    """Validate shared-graph delete options up front.

    - ``delete_graph_orphans=False`` is rejected: the engine's
      ``adelete_by_doc_id`` always prunes entities/relations that lose their
      last source (source-attribution); opting out is not yet supported, so we
      fail loudly rather than silently ignore the flag.
    - ``strategy="rebuild_kb"`` / ``strategy="rebuild_subgraph"`` require a
      configured ``IndexBuildService`` to run the post-delete rebuild.
    """
    if not delete_graph_orphans:
        raise HTTPException(
            status_code=400,
            detail=(
                "delete_graph_orphans=false is not supported: the engine always "
                "prunes graph entities/relations that lose their last source."
            ),
        )
    if strategy in {"rebuild_kb", "rebuild_subgraph"} and index_service is None:
        raise HTTPException(
            status_code=503,
            detail=f"strategy={strategy} requires the KB index build service",
        )


async def _run_conservative_kb_rebuild(
    *,
    document_service: DocumentLifecycleService,
    index_service: IndexBuildService,
    registry: LightRAGInstanceRegistry,
    kb_id: str,
) -> dict[str, Any]:
    """Force-reindex every remaining buildable document in the KB.

    Used by ``strategy="rebuild_kb"`` after a delete: deleting a document can
    leave shared graph entities/relations whose summaries were partly derived
    from the removed source. A conservative full reindex of the survivors
    re-derives the KG from the remaining chunks. Runs inline within the delete
    job (no separate job) and returns a summary recorded on the delete result.
    """
    rag = await registry.get(kb_id)
    document_ids: list[str] = []
    for status in ("parsed", "ready", "build_failed"):
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
    document_ids = list(dict.fromkeys(document_ids))
    rebuilt = 0
    failed = 0
    for document_id in document_ids:
        try:
            plan = await index_service.create_build_plan(
                kb_id,
                document_id,
                rag=rag,
                force_rechunk=True,
                force_extract=True,
                force_embedding=True,
            )
            if not plan.skipped:
                await index_service.claim_build_queued(
                    kb_id, job_id=f"rebuild_kb::{document_id}", plan=plan
                )
            item = await _execute_build_plan(
                index_service=index_service,
                kb_id=kb_id,
                job_id=f"rebuild_kb::{document_id}",
                plan=plan,
                rag=rag,
            )
            if item["status"] == "succeeded":
                rebuilt += 1
            else:
                failed += 1
        except Exception as exc:  # noqa: BLE001 — isolate per-doc rebuild failures
            failed += 1
            logger.error(
                "Conservative rebuild of doc '%s' (KB '%s') failed: %s",
                document_id,
                kb_id,
                exc,
            )
    return {
        "strategy": "rebuild_kb",
        "rebuilt_documents": rebuilt,
        "failed_documents": failed,
        "total_candidates": len(document_ids),
    }


async def _capture_graph_footprint(
    *, rag: Any, lightrag_doc_id: str | None
) -> dict[str, Any]:
    """Snapshot the entity names + relation pairs a doc contributed to the KG.

    Must be called *before* ``adelete_by_doc_id`` runs, because deletion removes
    the per-doc ``full_entities`` / ``full_relations`` rows this reads. Returns a
    footprint of ``{"entities": set[str], "relations": set[frozenset{src,tgt}]}``
    used by ``strategy="rebuild_subgraph"`` to find which *surviving* documents
    overlap the deleted document's slice of the shared graph.

    Resilient by design: any missing storage / row / key yields an empty
    footprint (→ zero affected survivors), never an exception that would break
    the delete itself.
    """
    entities: set[str] = set()
    relations: set[frozenset] = set()
    if not lightrag_doc_id:
        return {"entities": entities, "relations": relations}
    full_entities = getattr(rag, "full_entities", None)
    full_relations = getattr(rag, "full_relations", None)
    try:
        if full_entities is not None:
            data = await full_entities.get_by_id(lightrag_doc_id)
            if isinstance(data, dict):
                for name in data.get("entity_names", []) or []:
                    if name:
                        entities.add(str(name))
    except Exception as exc:  # noqa: BLE001 — footprint is best-effort
        logger.warning(
            "rebuild_subgraph: failed to read entities for '%s': %s",
            lightrag_doc_id,
            exc,
        )
    try:
        if full_relations is not None:
            data = await full_relations.get_by_id(lightrag_doc_id)
            if isinstance(data, dict):
                for pair in data.get("relation_pairs", []) or []:
                    if isinstance(pair, (list, tuple)) and len(pair) == 2 and all(pair):
                        relations.add(frozenset((str(pair[0]), str(pair[1]))))
    except Exception as exc:  # noqa: BLE001 — footprint is best-effort
        logger.warning(
            "rebuild_subgraph: failed to read relations for '%s': %s",
            lightrag_doc_id,
            exc,
        )
    return {"entities": entities, "relations": relations}


def _merge_footprints(footprints: Sequence[dict[str, Any]]) -> dict[str, Any]:
    entities: set[str] = set()
    relations: set[frozenset] = set()
    for footprint in footprints:
        entities |= footprint.get("entities", set())
        relations |= footprint.get("relations", set())
    return {"entities": entities, "relations": relations}


def _serialize_graph_footprint(
    footprint: dict[str, Any],
    *,
    document_id: str | None = None,
    lightrag_doc_id: str | None = None,
) -> dict[str, Any]:
    entities = sorted(
        {str(entity) for entity in footprint.get("entities", set()) if entity}
    )
    relations: list[list[str]] = []
    for relation in footprint.get("relations", set()):
        if not isinstance(relation, (frozenset, set, tuple, list)) or len(relation) != 2:
            continue
        pair = sorted(str(item) for item in relation if item)
        if len(pair) == 2:
            relations.append(pair)
    payload: dict[str, Any] = {"entities": entities, "relations": sorted(relations)}
    if document_id:
        payload["document_id"] = document_id
    if lightrag_doc_id:
        payload["lightrag_doc_id"] = lightrag_doc_id
    return payload


def _deserialize_graph_footprint(value: Any) -> dict[str, Any]:
    entities: set[str] = set()
    relations: set[frozenset] = set()
    if not isinstance(value, dict):
        return {"entities": entities, "relations": relations}
    for entity in value.get("entities", []) or []:
        if entity:
            entities.add(str(entity))
    for pair in value.get("relations", []) or []:
        if isinstance(pair, (list, tuple)) and len(pair) == 2 and all(pair):
            relations.add(frozenset((str(pair[0]), str(pair[1]))))
    return {"entities": entities, "relations": relations}


async def _document_overlaps_footprint(
    *, rag: Any, lightrag_doc_id: str | None, footprint: dict[str, Any]
) -> bool:
    """True if a surviving document still contributes to any entity/relation in
    the deleted document's footprint (→ its KG slice may need re-derivation)."""
    if not lightrag_doc_id:
        return False
    target_entities: set[str] = footprint.get("entities", set())
    target_relations: set[frozenset] = footprint.get("relations", set())
    if not target_entities and not target_relations:
        return False
    full_entities = getattr(rag, "full_entities", None)
    full_relations = getattr(rag, "full_relations", None)
    try:
        if target_entities and full_entities is not None:
            data = await full_entities.get_by_id(lightrag_doc_id)
            if isinstance(data, dict):
                names = {str(n) for n in (data.get("entity_names", []) or []) if n}
                if names & target_entities:
                    return True
        if target_relations and full_relations is not None:
            data = await full_relations.get_by_id(lightrag_doc_id)
            if isinstance(data, dict):
                pairs = {
                    frozenset((str(p[0]), str(p[1])))
                    for p in (data.get("relation_pairs", []) or [])
                    if isinstance(p, (list, tuple)) and len(p) == 2 and all(p)
                }
                if pairs & target_relations:
                    return True
    except Exception as exc:  # noqa: BLE001 — overlap check is best-effort
        logger.warning(
            "rebuild_subgraph: overlap check failed for '%s': %s",
            lightrag_doc_id,
            exc,
        )
    return False


async def _run_subgraph_rebuild(
    *,
    document_service: DocumentLifecycleService,
    index_service: IndexBuildService,
    registry: LightRAGInstanceRegistry,
    kb_id: str,
    footprint: dict[str, Any],
) -> dict[str, Any]:
    """Precise shared-subgraph local rebuild for ``strategy="rebuild_subgraph"``.

    Unlike ``rebuild_kb`` (which force-reindexes *every* remaining buildable
    document), this re-derives only the *surviving* documents that actually
    shared an entity or relation with the just-deleted document — i.e. the
    documents whose KG slice the deletion could have perturbed. Documents that
    never touched the deleted document's footprint are left untouched.

    The engine's ``adelete_by_doc_id`` already narrows shared entities' source
    attribution; this step rebuilds the overlapping survivors from their
    remaining chunks so descriptions/summaries derived partly from the removed
    source are regenerated. Falls back to reindexing nothing (not the whole KB)
    when the footprint is empty.
    """
    rag = await registry.get(kb_id)
    candidate_ids: list[str] = []
    for status in ("parsed", "ready", "build_failed"):
        offset = 0
        page_size = 200
        while True:
            documents, total = await document_service.list_documents(
                kb_id, status=status, limit=page_size, offset=offset
            )
            candidate_ids.extend(doc.id for doc in documents)
            offset += page_size
            if offset >= total or not documents:
                break
    candidate_ids = list(dict.fromkeys(candidate_ids))

    affected_ids: list[str] = []
    for document_id in candidate_ids:
        try:
            doc = await document_service.get_document(kb_id, document_id)
        except Exception:  # noqa: BLE001 — skip vanished docs
            continue
        if await _document_overlaps_footprint(
            rag=rag, lightrag_doc_id=doc.lightrag_doc_id, footprint=footprint
        ):
            affected_ids.append(document_id)

    rebuilt = 0
    failed = 0
    for document_id in affected_ids:
        try:
            plan = await index_service.create_build_plan(
                kb_id,
                document_id,
                rag=rag,
                force_rechunk=True,
                force_extract=True,
                force_embedding=True,
            )
            if not plan.skipped:
                await index_service.claim_build_queued(
                    kb_id, job_id=f"rebuild_subgraph::{document_id}", plan=plan
                )
            item = await _execute_build_plan(
                index_service=index_service,
                kb_id=kb_id,
                job_id=f"rebuild_subgraph::{document_id}",
                plan=plan,
                rag=rag,
            )
            if item["status"] == "succeeded":
                rebuilt += 1
            else:
                failed += 1
        except Exception as exc:  # noqa: BLE001 — isolate per-doc rebuild failures
            failed += 1
            logger.error(
                "Subgraph rebuild of doc '%s' (KB '%s') failed: %s",
                document_id,
                kb_id,
                exc,
            )
    return {
        "strategy": "rebuild_subgraph",
        "rebuilt_documents": rebuilt,
        "failed_documents": failed,
        "affected_documents": len(affected_ids),
        "total_candidates": len(candidate_ids),
        "footprint_entities": len(footprint.get("entities", set())),
        "footprint_relations": len(footprint.get("relations", set())),
    }


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



async def _run_sync_followups(
    *,
    document_service: DocumentLifecycleService,
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
    defer_build: bool = False,
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
        if defer_build:
            # Caller will batch-build all deferred plans through a single
            # pipeline drain so the three worker layers (parse / analyze /
            # process) overlap documents instead of being serialized.
            # The plan is stashed on the item under a private key the caller
            # pops before finalising the response.
            item["_deferred_build_plan"] = build_plan
            return item, rag
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


async def _execute_replace_document(
    *,
    document_service: DocumentLifecycleService,
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


async def _execute_sync_item(
    *,
    document_service: DocumentLifecycleService,
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
    defer_build: bool = False,
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
                document_service=document_service,
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
                defer_build=defer_build,
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
                document_service=document_service,
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
                defer_build=defer_build,
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
                document_service=document_service,
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


def create_kb_document_routes(
    document_service: DocumentLifecycleService,
    job_service: JobService,
    api_key: Optional[str] = None,
    registry: LightRAGInstanceRegistry | None = None,
    index_service: IndexBuildService | None = None,
):
    router = APIRouter(prefix="/kbs", tags=["knowledge-base-documents"])
    combined_auth = get_combined_auth_dependency(api_key)

    def _schedule_auto_parse(
        background_tasks: BackgroundTasks,
        *,
        kb_id: str,
        job: JobRecord,
        documents: list[DocumentRecord],
        auto_index: bool,
    ) -> None:
        """Schedule the in-process auto-parse executor for a freshly-created
        multi-file upload / texts batch (job_type=parse, document_id=None)."""
        if registry is None:
            return
        document_ids = [document.id for document in documents]
        background_tasks.add_task(
            _run_auto_parse_batch,
            document_service=document_service,
            job_service=job_service,
            registry=registry,
            index_service=index_service,
            kb_id=kb_id,
            job=job,
            document_ids=document_ids,
            auto_index=auto_index,
        )

    @router.post(
        "/{kb_id}/documents:upload",
        response_model=DocumentBatchResponse,
        dependencies=[Depends(combined_auth)],
        summary="Upload documents to a knowledge base metadata stage",
    )
    async def upload_documents(
        kb_id: str,
        background_tasks: BackgroundTasks,
        http_request: Request,
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
            if auto_parse and result.created and registry is not None:
                _schedule_auto_parse(
                    background_tasks,
                    kb_id=kb_id,
                    job=result.job,
                    documents=result.documents,
                    auto_index=auto_index,
                )
            await _append_kb_document_audit_event(
                http_request,
                "documents_uploaded",
                kb_id,
                _document_audit_metadata(
                    job=result.job,
                    operation="upload",
                    document_ids=[document.id for document in result.documents],
                    batch_id=result.batch_id,
                    source_type="upload",
                    auto_parse=auto_parse,
                    auto_index=auto_index,
                    created=result.created,
                    parser_engine=parser_engine,
                ),
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
    async def import_text_documents(
        kb_id: str,
        background_tasks: BackgroundTasks,
        http_request: Request,
        request: TextDocumentsRequest,
    ):
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
            if request.auto_parse and result.created and registry is not None:
                _schedule_auto_parse(
                    background_tasks,
                    kb_id=kb_id,
                    job=result.job,
                    documents=result.documents,
                    auto_index=request.auto_index,
                )
            await _append_kb_document_audit_event(
                http_request,
                "text_documents_imported",
                kb_id,
                _document_audit_metadata(
                    job=result.job,
                    operation="import_texts",
                    document_ids=[document.id for document in result.documents],
                    batch_id=result.batch_id,
                    source_type="text",
                    auto_parse=request.auto_parse,
                    auto_index=request.auto_index,
                    created=result.created,
                    parser_engine=request.parser_engine,
                ),
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

    @router.post(
        "/{kb_id}/documents:urls",
        response_model=DocumentBatchResponse,
        dependencies=[Depends(combined_auth)],
        summary="Ingest URL documents to a knowledge base metadata stage",
    )
    async def ingest_url_documents(
        kb_id: str,
        background_tasks: BackgroundTasks,
        http_request: Request,
        request: UrlDocumentsRequest,
    ):
        try:
            max_upload_size = _required_upload_limit()
            total_bytes = 0
            sources: list[DocumentSourceInput] = []
            seen_source_keys: set[str] = set()
            for document in request.documents:
                source, size = await _fetch_url_document(
                    document,
                    max_upload_size=max_upload_size,
                    remaining_batch_bytes=max_upload_size - total_bytes,
                )
                source_key = str(source.metadata["source_key"])
                if source_key in seen_source_keys:
                    raise HTTPException(
                        status_code=400, detail="Duplicate source_key values are not allowed"
                    )
                seen_source_keys.add(source_key)
                total_bytes += size
                sources.append(source)
            result = await document_service.create_source_batch(
                kb_id,
                sources,
                auto_parse=request.auto_parse,
                auto_index=request.auto_index,
                parser_engine=request.parser_engine,
                process_options=request.process_options,
                idempotency_key=request.idempotency_key,
            )
            if request.auto_parse and result.created and registry is not None:
                _schedule_auto_parse(
                    background_tasks,
                    kb_id=kb_id,
                    job=result.job,
                    documents=result.documents,
                    auto_index=request.auto_index,
                )
            await _append_kb_document_audit_event(
                http_request,
                "url_documents_ingested",
                kb_id,
                _document_audit_metadata(
                    job=result.job,
                    operation="ingest_urls",
                    document_ids=[document.id for document in result.documents],
                    batch_id=result.batch_id,
                    source_type="url",
                    auto_parse=request.auto_parse,
                    auto_index=request.auto_index,
                    created=result.created,
                    parser_engine=request.parser_engine,
                ),
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
            logger.error("Failed to ingest URLs for KB '%s': %s", kb_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post(
        "/{kb_id}/documents:import",
        response_model=DocumentBatchResponse,
        dependencies=[Depends(combined_auth)],
        summary="Import controlled local staged files to a knowledge base metadata stage",
    )
    async def import_local_documents(
        kb_id: str,
        background_tasks: BackgroundTasks,
        http_request: Request,
        request: LocalImportDocumentsRequest,
    ):
        try:
            source_root = _source_root_resolved(document_service)
            max_upload_size = _required_upload_limit()
            total_bytes = 0
            sources: list[DocumentSourceInput] = []
            seen_source_keys: set[str] = set()
            for document in request.documents:
                path = _resolve_staged_path(source_root, document.path)
                source_name = document.source_name or path.name
                _validate_staged_file(path, source_name=source_name)
                content = _read_staged_file_content(
                    path,
                    max_upload_size=max_upload_size,
                    remaining_batch_bytes=max_upload_size - total_bytes,
                )
                total_bytes += len(content)
                relative_path = _relative_staged_path(source_root, path)
                source_key = _normalize_sync_source_key(
                    document.source_key or f"import:{relative_path}"
                )
                if source_key in seen_source_keys:
                    raise HTTPException(
                        status_code=400, detail="Duplicate source_key values are not allowed"
                    )
                seen_source_keys.add(source_key)
                sources.append(
                    DocumentSourceInput(
                        source_name=source_name,
                        content=content,
                        source_type="import",
                        content_type=document.content_type,
                        metadata=_merge_source_metadata(
                            document.metadata,
                            {
                                "source_key": source_key,
                                "staged_source_path": relative_path,
                            },
                        ),
                    )
                )
            result = await document_service.create_source_batch(
                kb_id,
                sources,
                auto_parse=request.auto_parse,
                auto_index=request.auto_index,
                parser_engine=request.parser_engine,
                process_options=request.process_options,
                idempotency_key=request.idempotency_key,
            )
            if request.auto_parse and result.created and registry is not None:
                _schedule_auto_parse(
                    background_tasks,
                    kb_id=kb_id,
                    job=result.job,
                    documents=result.documents,
                    auto_index=request.auto_index,
                )
            await _append_kb_document_audit_event(
                http_request,
                "local_documents_imported",
                kb_id,
                _document_audit_metadata(
                    job=result.job,
                    operation="import_local",
                    document_ids=[document.id for document in result.documents],
                    batch_id=result.batch_id,
                    source_type="import",
                    auto_parse=request.auto_parse,
                    auto_index=request.auto_index,
                    created=result.created,
                    parser_engine=request.parser_engine,
                ),
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
            logger.error("Failed to import local files for KB '%s': %s", kb_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post(
        "/{kb_id}/documents:scan",
        response_model=DocumentBatchResponse,
        dependencies=[Depends(combined_auth)],
        summary="Scan controlled local staged directories into a knowledge base metadata stage",
    )
    async def scan_local_documents(
        kb_id: str,
        background_tasks: BackgroundTasks,
        http_request: Request,
        request: LocalScanDocumentsRequest,
    ):
        try:
            source_root = _source_root_resolved(document_service)
            if _requested_path_resolves_to_root(source_root, request.directory):
                raise HTTPException(
                    status_code=400, detail="Scan directory cannot be INPUT_DIR root"
                )
            directory = _resolve_staged_path(source_root, request.directory)
            files = _scan_supported_files(
                source_root,
                directory,
                recursive=request.recursive,
                max_files=request.max_files,
            )
            max_upload_size = _required_upload_limit()
            total_bytes = 0
            sources: list[DocumentSourceInput] = []
            prefix = _normalize_sync_source_key(request.source_key_prefix)
            for path in files:
                content = _read_staged_file_content(
                    path,
                    max_upload_size=max_upload_size,
                    remaining_batch_bytes=max_upload_size - total_bytes,
                )
                total_bytes += len(content)
                relative_path = _relative_staged_path(source_root, path)
                source_key = _normalize_sync_source_key(f"{prefix}:{relative_path}")
                sources.append(
                    DocumentSourceInput(
                        source_name=path.name,
                        content=content,
                        source_type="scan",
                        content_type=None,
                        metadata={
                            "source_key": source_key,
                            "scanned_source_path": relative_path,
                        },
                    )
                )
            result = await document_service.create_source_batch(
                kb_id,
                sources,
                auto_parse=request.auto_parse,
                auto_index=request.auto_index,
                parser_engine=request.parser_engine,
                process_options=request.process_options,
                idempotency_key=request.idempotency_key,
            )
            if request.auto_parse and result.created and registry is not None:
                _schedule_auto_parse(
                    background_tasks,
                    kb_id=kb_id,
                    job=result.job,
                    documents=result.documents,
                    auto_index=request.auto_index,
                )
            await _append_kb_document_audit_event(
                http_request,
                "local_documents_scanned",
                kb_id,
                _document_audit_metadata(
                    job=result.job,
                    operation="scan_local",
                    document_ids=[document.id for document in result.documents],
                    batch_id=result.batch_id,
                    source_type="scan",
                    auto_parse=request.auto_parse,
                    auto_index=request.auto_index,
                    recursive=request.recursive,
                    created=result.created,
                    parser_engine=request.parser_engine,
                ),
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
            logger.error("Failed to scan local files for KB '%s': %s", kb_id, exc)
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
        kb_id: str, document_id: str, http_request: Request, request: PatchDocumentRequest
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
            document = await document_service.update_document(
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
            await _append_kb_document_audit_event(
                http_request,
                "document_updated",
                kb_id,
                _document_audit_metadata(
                    operation="patch_document",
                    document_ids=[document.id],
                    fields=sorted(request.model_fields_set),
                ),
            )
            return DocumentResponse.from_record(document)
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
    async def disable_document(kb_id: str, document_id: str, http_request: Request):
        try:
            document = await document_service.update_document(
                kb_id, document_id, enabled=False
            )
            await _append_kb_document_audit_event(
                http_request,
                "document_disabled",
                kb_id,
                _document_audit_metadata(
                    operation="disable_document",
                    document_ids=[document.id],
                ),
            )
            return DocumentResponse.from_record(document)
        except (KnowledgeBaseNotFoundError, MetadataRecordNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get(
        "/{kb_id}/documents/{document_id}/chunks",
        response_model=DocumentChunksResponse,
        dependencies=[Depends(combined_auth)],
        summary="List the engine text chunks built from one document",
    )
    async def list_document_chunks(
        kb_id: str,
        document_id: str,
        limit: int = 50,
        offset: int = 0,
    ):
        """Retrieval-explainability view of a built document.

        Resolves ``lightrag_doc_id`` -> engine doc_status ``chunks_list`` ->
        ``text_chunks`` rows, ordered by ``chunk_order_index``. Documents that
        have not been built yet (no ``lightrag_doc_id``) return an empty page
        without loading the engine instance.
        """
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        try:
            document = await document_service.get_document(kb_id, document_id)
        except (KnowledgeBaseNotFoundError, MetadataRecordNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not document.lightrag_doc_id:
            return DocumentChunksResponse(
                kb_id=kb_id,
                document_id=document.id,
                lightrag_doc_id=None,
                total=0,
                limit=limit,
                offset=offset,
                chunks=[],
            )
        if registry is None:
            raise HTTPException(
                status_code=503, detail="LightRAG registry is not configured"
            )
        rag = cast(Any, await registry.get(kb_id))
        doc_status_store = getattr(rag, "doc_status", None)
        text_chunks_store = getattr(rag, "text_chunks", None)
        if doc_status_store is None or text_chunks_store is None:
            raise HTTPException(
                status_code=503, detail="Engine chunk storages are unavailable"
            )
        status_row = await doc_status_store.get_by_id(document.lightrag_doc_id)
        if isinstance(status_row, dict):
            chunk_ids = list(status_row.get("chunks_list") or [])
        else:
            chunk_ids = list(getattr(status_row, "chunks_list", None) or [])
        rows = await text_chunks_store.get_by_ids(chunk_ids) if chunk_ids else []
        items: list[DocumentChunkItem] = []
        for chunk_id, row in zip(chunk_ids, rows):
            if not isinstance(row, dict):
                continue
            items.append(
                DocumentChunkItem(
                    id=chunk_id,
                    chunk_order_index=row.get("chunk_order_index"),
                    tokens=row.get("tokens"),
                    content=row.get("content"),
                    file_path=row.get("file_path"),
                )
            )
        items.sort(
            key=lambda item: (
                item.chunk_order_index is None,
                item.chunk_order_index or 0,
            )
        )
        return DocumentChunksResponse(
            kb_id=kb_id,
            document_id=document.id,
            lightrag_doc_id=document.lightrag_doc_id,
            total=len(items),
            limit=limit,
            offset=offset,
            chunks=items[offset : offset + limit],
        )

    @router.post(
        "/{kb_id}/documents/{document_id}:enable",
        response_model=DocumentResponse,
        dependencies=[Depends(combined_auth)],
        summary="Enable one knowledge base document",
    )
    async def enable_document(kb_id: str, document_id: str, http_request: Request):
        try:
            document = await document_service.update_document(kb_id, document_id, enabled=True)
            await _append_kb_document_audit_event(
                http_request,
                "document_enabled",
                kb_id,
                _document_audit_metadata(
                    operation="enable_document",
                    document_ids=[document.id],
                ),
            )
            return DocumentResponse.from_record(document)
        except (KnowledgeBaseNotFoundError, MetadataRecordNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def _batch_set_documents_enabled(
        kb_id: str,
        http_request: Request,
        request: BatchSetDocumentsEnabledRequest,
        *,
        enabled: bool,
    ) -> BatchSetDocumentsEnabledResponse:
        """Synchronous control-plane bulk toggle (same semantics as the
        single-document ``:enable`` / ``:disable`` actions, no job created).

        Missing documents are reported per item instead of failing the batch;
        re-applying the current state still counts as ``updated`` (idempotent).
        """
        items: list[BatchSetDocumentsEnabledItem] = []
        updated_ids: list[str] = []
        try:
            for document_id in request.document_ids:
                try:
                    document = await document_service.update_document(
                        kb_id, document_id, enabled=enabled
                    )
                except MetadataRecordNotFoundError:
                    items.append(
                        BatchSetDocumentsEnabledItem(
                            document_id=document_id, status="not_found"
                        )
                    )
                    continue
                updated_ids.append(document.id)
                items.append(
                    BatchSetDocumentsEnabledItem(
                        document_id=document.id, status="updated"
                    )
                )
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await _append_kb_document_audit_event(
            http_request,
            "document_batch_enabled" if enabled else "document_batch_disabled",
            kb_id,
            _document_audit_metadata(
                operation="batch_enable_documents"
                if enabled
                else "batch_disable_documents",
                document_ids=updated_ids,
                not_found_count=len(items) - len(updated_ids),
            ),
        )
        return BatchSetDocumentsEnabledResponse(
            enabled=enabled,
            updated=len(updated_ids),
            not_found=len(items) - len(updated_ids),
            items=items,
        )

    @router.post(
        "/{kb_id}/documents:batch-enable",
        response_model=BatchSetDocumentsEnabledResponse,
        dependencies=[Depends(combined_auth)],
        summary="Enable a batch of knowledge base documents",
    )
    async def batch_enable_documents(
        kb_id: str, http_request: Request, request: BatchSetDocumentsEnabledRequest
    ):
        return await _batch_set_documents_enabled(
            kb_id, http_request, request, enabled=True
        )

    @router.post(
        "/{kb_id}/documents:batch-disable",
        response_model=BatchSetDocumentsEnabledResponse,
        dependencies=[Depends(combined_auth)],
        summary="Disable a batch of knowledge base documents",
    )
    async def batch_disable_documents(
        kb_id: str, http_request: Request, request: BatchSetDocumentsEnabledRequest
    ):
        return await _batch_set_documents_enabled(
            kb_id, http_request, request, enabled=False
        )

    @router.post(
        "/{kb_id}/documents:sync",
        response_model=JobResponse,
        dependencies=[Depends(combined_auth)],
        summary="Synchronize a batch of knowledge base documents by source key",
    )
    async def sync_documents(
        kb_id: str,
        background_tasks: BackgroundTasks,
        http_request: Request,
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

        active_registry = cast(LightRAGInstanceRegistry, registry)
        active_index_service = index_service
        batch_id: str | None = None
        sync_staged = False
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

            async with document_service.kb_write_guard(kb_id):
                batch_id = generate_track_id("batch")
                for item_index, prepared in enumerate(prepared_sources):
                    await document_service.stage_sync_source_bytes(
                        kb_id,
                        batch_id=batch_id,
                        item_index=item_index,
                        source=cast(DocumentSourceInput, prepared["source"]),
                    )
                    sync_staged = True
                fingerprint_payload = {
                    "items": [
                        {
                            "source_key": item["source_key"],
                            "source_name": cast(
                                DocumentSourceInput, item["source"]
                            ).source_name,
                            "source_type": cast(
                                DocumentSourceInput, item["source"]
                            ).source_type,
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
                    "batch_id": batch_id,
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
                    await document_service.clear_staged_sync_sources(
                        kb_id, batch_id=batch_id
                    )
                    sync_staged = False
                    return JobResponse.from_record(job)
            await _append_kb_document_audit_event(
                http_request,
                "documents_sync_queued",
                kb_id,
                _document_audit_metadata(
                    job=job,
                    operation="sync_documents",
                    document_count=len(prepared_sources),
                    batch_id=batch_id,
                    source_type="upload",
                    auto_parse=auto_parse,
                    auto_index=auto_index,
                    force_reparse=force_reparse,
                    delete_source_file=delete_source_file,
                    delete_artifacts=delete_artifacts,
                    delete_llm_cache=delete_llm_cache,
                    parser_engine=parser_engine,
                ),
            )

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

                    # Phase 1: per-item sync (create/skip/replace + parse)
                    # runs concurrently bounded by MAX_PARALLEL_PARSE_MINERU,
                    # with auto_index builds deferred so we can drain the
                    # pipeline once for all docs in Phase 2 (so analyze /
                    # extract / merge stages overlap across documents).
                    rag = cast(Any, await active_registry.get(kb_id))
                    parse_concurrency = max(
                        1,
                        int(getattr(rag, "max_parallel_parse_mineru", 1) or 1),
                    )
                    parse_sem = asyncio.Semaphore(parse_concurrency)

                    parsed_count = 0
                    total_sources = len(prepared_sources)

                    async def _do_one_sync_item(
                        prepared_item: dict[str, Any],
                    ) -> dict[str, Any]:
                        nonlocal parsed_count
                        async with parse_sem:
                            item, _ = await _execute_sync_item(
                                document_service=document_service,
                                kb_id=kb_id,
                                job=job,
                                prepared=prepared_item,
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
                                defer_build=True,
                            )
                        parsed_count += 1
                        done = parsed_count
                        try:
                            await job_service.update_job_progress(
                                kb_id,
                                job.id,
                                completed_items=done,
                                progress=0.5 * done / max(total_sources, 1),
                                result_patch={
                                    "pipeline": {
                                        "latest_message": (
                                            f"Parsed {done}/{total_sources} "
                                            "document(s); building knowledge graph…"
                                        )
                                    }
                                },
                            )
                        except Exception:  # noqa: BLE001 — progress is best-effort
                            pass
                        return item

                    raw_sync = await asyncio.gather(
                        *[
                            _do_one_sync_item(prepared)
                            for prepared in prepared_sources
                        ],
                        return_exceptions=True,
                    )
                    # return_exceptions=True guarantees Phase 2 still runs (so
                    # any doc already claimed into build_queued gets built /
                    # released) even if a task raised an unexpected
                    # BaseException; map such exceptions back to a failed item.
                    sync_items: list[dict[str, Any]] = []
                    for prepared, outcome in zip(prepared_sources, raw_sync):
                        if isinstance(outcome, BaseException):
                            source = cast(DocumentSourceInput, prepared["source"])
                            sync_items.append(
                                {
                                    "source_key": str(prepared["source_key"]),
                                    "source_name": source.source_name,
                                    "source_hash": str(prepared["source_hash"]),
                                    "action": "unknown",
                                    "status": "failed",
                                    "error_code": "sync_item_failed",
                                    "error_message": str(outcome),
                                }
                            )
                        else:
                            sync_items.append(outcome)

                    # Phase 2: batch-build for any item whose build was deferred.
                    if active_index_service is not None:
                        deferred_pairs: list[tuple[IndexBuildPlan, dict[str, Any]]] = []
                        for item in sync_items:
                            build_plan = item.pop("_deferred_build_plan", None)
                            if build_plan is not None:
                                deferred_pairs.append((build_plan, item))
                        if deferred_pairs:
                            batch_results = await _execute_build_plan_batch(
                                index_service=active_index_service,
                                kb_id=kb_id,
                                job_id=job.id,
                                rag=rag,
                                plans=[bp for bp, _ in deferred_pairs],
                                job_service=job_service,
                            )
                            for build_plan, item in deferred_pairs:
                                build_item = batch_results.get(build_plan.document.id)
                                if build_item is None:
                                    item.update(
                                        {
                                            "status": "failed",
                                            "error_code": "build_failed",
                                            "error_message": (
                                                "Build result missing from batch"
                                            ),
                                        }
                                    )
                                    continue
                                item["build_result"] = build_item
                                if build_item["status"] not in {
                                    "succeeded",
                                    "cancelled",
                                }:
                                    item.update(
                                        {
                                            "status": "failed",
                                            "error_code": build_item.get(
                                                "error_code", "build_failed"
                                            ),
                                            "error_message": build_item.get(
                                                "error_message",
                                                "Document sync build failed",
                                            ),
                                        }
                                    )

                    for item in sync_items:
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
                    await document_service.clear_staged_sync_sources(
                        kb_id, batch_id=batch_id
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
                    await document_service.clear_staged_sync_sources(
                        kb_id, batch_id=batch_id
                    )

            background_tasks.add_task(_sync_task)
            sync_staged = False
            return JobResponse.from_record(job)
        except HTTPException:
            if sync_staged and batch_id is not None:
                await document_service.clear_staged_sync_sources(kb_id, batch_id=batch_id)
            raise
        except KnowledgeBaseNotFoundError as exc:
            if sync_staged and batch_id is not None:
                await document_service.clear_staged_sync_sources(kb_id, batch_id=batch_id)
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except IdempotencyKeyConflictError as exc:
            if sync_staged and batch_id is not None:
                await document_service.clear_staged_sync_sources(kb_id, batch_id=batch_id)
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            if sync_staged and batch_id is not None:
                await document_service.clear_staged_sync_sources(kb_id, batch_id=batch_id)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            if sync_staged and batch_id is not None:
                await document_service.clear_staged_sync_sources(kb_id, batch_id=batch_id)
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
        http_request: Request,
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
        active_registry = cast(LightRAGInstanceRegistry, registry)
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
                source_type=replacement.source_type,
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
                # Stage the replacement bytes to disk so a durable worker can
                # resume this replace from disk after a crash (orphan recovery
                # → replace_failed → :retry → queued → worker re-drive).
                await document_service.stage_replacement_bytes(
                    kb_id,
                    document_id,
                    job_id=job.id,
                    replacement=replacement,
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

            await _append_kb_document_audit_event(
                http_request,
                "document_replace_queued",
                kb_id,
                _document_audit_metadata(
                    job=job,
                    operation="replace_document",
                    document_ids=[document.id],
                    source_type=replacement.source_type,
                    size_bytes=replacement.size_bytes,
                    auto_parse=auto_parse,
                    auto_index=auto_index,
                    force_reparse=force_reparse,
                    delete_source_file=delete_source_file,
                    delete_artifacts=delete_artifacts,
                    delete_llm_cache=delete_llm_cache,
                    parser_engine=parser_engine,
                ),
            )

            async def _replace_task() -> None:
                replace_claim_released = False
                try:
                    await job_service.transition_job(
                        kb_id, job.id, status="running", progress=0.1
                    )
                    item = await _execute_replace_document(
                        document_service=document_service,
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
                    await document_service.clear_staged_replacement(
                        kb_id,
                        document_id,
                        job_id=job.id,
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
                    else:
                        await document_service.clear_staged_replacement(
                            kb_id,
                            document_id,
                            job_id=job.id,
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
        # Delegates to the module-level executor so the durable job worker can
        # reuse the same delete logic when re-driving a queued delete job.
        return await _execute_delete_document_impl(
            document_service=document_service,
            kb_id=kb_id,
            job_id=job_id,
            document=document,
            active_registry=active_registry,
            delete_source_file=delete_source_file,
            delete_artifacts=delete_artifacts,
            delete_llm_cache=delete_llm_cache,
        )

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
        http_request: Request,
        delete_source_file: bool = False,
        delete_artifacts: bool = False,
        delete_llm_cache: bool = False,
        delete_graph_orphans: bool = True,
        strategy: Literal["safe", "rebuild_doc_scope", "rebuild_kb", "rebuild_subgraph"] = "safe",
        idempotency_key: Optional[str] = None,
    ):
        if registry is None:
            raise HTTPException(
                status_code=503, detail="KB delete service is not configured"
            )
        _validate_delete_strategy(
            strategy=strategy,
            delete_graph_orphans=delete_graph_orphans,
            index_service=index_service,
        )
        active_registry = cast(LightRAGInstanceRegistry, registry)
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
                        and bool(existing_payload.get("delete_graph_orphans", True))
                        == delete_graph_orphans
                        and str(existing_payload.get("strategy", "safe")) == strategy
                    )
                    if not same_request:
                        raise IdempotencyKeyConflictError(idempotency_key)
                    return JobResponse.from_record(existing_job)
            document = await document_service.get_document(kb_id, document_id)
            delete_scope: str | None = None
            if enterprise_auth_enabled():
                delete_scope = await get_enterprise_authorization_service(
                    http_request
                ).authorize_document_delete(
                    get_request_principal(http_request),
                    kb_id,
                    document_owner_id=document.metadata.get("created_by"),
                )
            job, created_job = await job_service.create_delete_job_once(
                kb_id,
                document_id=document_id,
                lightrag_doc_id=document.lightrag_doc_id,
                delete_source_file=delete_source_file,
                delete_artifacts=delete_artifacts,
                delete_llm_cache=delete_llm_cache,
                delete_graph_orphans=delete_graph_orphans,
                strategy=strategy,
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

            await _append_kb_document_audit_event(
                http_request,
                "document_delete_queued",
                kb_id,
                _document_audit_metadata(
                    job=job,
                    operation="delete_document",
                    document_ids=[document.id],
                    delete_source_file=delete_source_file,
                    delete_artifacts=delete_artifacts,
                    delete_llm_cache=delete_llm_cache,
                    delete_graph_orphans=delete_graph_orphans,
                    strategy=strategy,
                    delete_scope=delete_scope,
                    document_owner=document.metadata.get("created_by"),
                ),
            )

            async def _delete_task() -> None:
                try:
                    await job_service.transition_job(
                        kb_id, job.id, status="running", progress=0.1
                    )
                    # Capture the doc's graph footprint BEFORE deletion removes
                    # its full_entities/full_relations rows (rebuild_subgraph).
                    pre_delete_footprint: dict[str, Any] | None = None
                    if strategy == "rebuild_subgraph" and index_service is not None:
                        footprint_rag = cast(Any, await active_registry.get(kb_id))
                        pre_delete_footprint = await _capture_graph_footprint(
                            rag=footprint_rag,
                            lightrag_doc_id=document.lightrag_doc_id,
                        )
                        await job_service.update_job_payload_patch(
                            kb_id,
                            job.id,
                            payload_patch={
                                "rebuild_subgraph_footprints": [
                                    _serialize_graph_footprint(
                                        pre_delete_footprint,
                                        document_id=document.id,
                                        lightrag_doc_id=document.lightrag_doc_id,
                                    )
                                ]
                            },
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
                        rebuild_summary = None
                        if strategy == "rebuild_kb" and index_service is not None:
                            rebuild_summary = await _run_conservative_kb_rebuild(
                                document_service=document_service,
                                index_service=index_service,
                                registry=active_registry,
                                kb_id=kb_id,
                            )
                        elif (
                            strategy == "rebuild_subgraph"
                            and index_service is not None
                        ):
                            rebuild_summary = await _run_subgraph_rebuild(
                                document_service=document_service,
                                index_service=index_service,
                                registry=active_registry,
                                kb_id=kb_id,
                                footprint=pre_delete_footprint
                                or {"entities": set(), "relations": set()},
                            )
                        result_payload = _delete_job_result(
                            total_items=1,
                            completed_items=1,
                            failed_items=0,
                            items=[item],
                        )
                        if rebuild_summary is not None:
                            result_payload["rebuild"] = rebuild_summary
                        await job_service.transition_job(
                            kb_id,
                            job.id,
                            status="succeeded",
                            progress=1.0,
                            completed_items=1,
                            result=result_payload,
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
        http_request: Request,
        request: BatchDeleteDocumentsRequest,
    ):
        if registry is None:
            raise HTTPException(
                status_code=503, detail="KB delete service is not configured"
            )
        _validate_delete_strategy(
            strategy=request.strategy,
            delete_graph_orphans=request.delete_graph_orphans,
            index_service=index_service,
        )
        active_registry = cast(LightRAGInstanceRegistry, registry)
        try:
            batch_id = generate_track_id("batch")
            job, created_job = await job_service.create_batch_delete_job_once(
                kb_id,
                batch_id=batch_id,
                document_ids=request.document_ids,
                delete_source_file=request.delete_source_file,
                delete_artifacts=request.delete_artifacts,
                delete_llm_cache=request.delete_llm_cache,
                delete_graph_orphans=request.delete_graph_orphans,
                strategy=request.strategy,
                idempotency_key=request.idempotency_key,
            )
            if not created_job:
                return JobResponse.from_record(job)

            # Ownership/capability check per requested document (enterprise mode).
            # kb_editor may only delete its own uploads; kb_admin+/super_admin or
            # the can_delete_documents capability may delete any. Denied docs
            # become per-item permission_denied failures and are NOT claimed.
            permission_failures: list[dict[str, Any]] = []
            scope_by_doc: dict[str, str] = {}
            authorized_ids: list[str] = list(request.document_ids)
            if enterprise_auth_enabled():
                authz = get_enterprise_authorization_service(http_request)
                principal = get_request_principal(http_request)
                existing_docs = await document_service.get_documents_by_ids(
                    kb_id, list(request.document_ids)
                )
                owner_by_id = {
                    doc.id: doc.metadata.get("created_by") for doc in existing_docs
                }
                authorized_ids = []
                for doc_id in request.document_ids:
                    # Unknown ids fall through to claim so they surface the
                    # canonical document_not_found failure rather than
                    # permission_denied (avoids existence disclosure).
                    if doc_id not in owner_by_id:
                        authorized_ids.append(doc_id)
                        continue
                    try:
                        scope = await authz.authorize_document_delete(
                            principal,
                            kb_id,
                            document_owner_id=owner_by_id[doc_id],
                        )
                    except HTTPException as exc:
                        if exc.status_code != 403:
                            raise
                        permission_failures.append(
                            {
                                "document_id": doc_id,
                                "status": "failed",
                                "error_code": "permission_denied",
                                "error_message": "Document delete denied",
                            }
                        )
                        continue
                    scope_by_doc[doc_id] = scope
                    authorized_ids.append(doc_id)
                # Shrink the durable payload so a worker resume after a crash can
                # never delete documents the caller was not authorized for.
                if authorized_ids != list(request.document_ids):
                    await job_service.update_job_payload_patch(
                        kb_id, job.id, payload_patch={"document_ids": authorized_ids}
                    )

            documents, store_claim_failures = await document_service.claim_batch_delete(
                kb_id,
                authorized_ids,
                job=job,
                delete_source_file=request.delete_source_file,
                delete_artifacts=request.delete_artifacts,
            )
            claim_failures = [*permission_failures, *store_claim_failures]

            await _append_kb_document_audit_event(
                http_request,
                "documents_batch_delete_queued",
                kb_id,
                _document_audit_metadata(
                    job=job,
                    operation="batch_delete_documents",
                    document_ids=[document.id for document in documents],
                    document_count=len(request.document_ids),
                    batch_id=job.batch_id or batch_id,
                    delete_source_file=request.delete_source_file,
                    delete_artifacts=request.delete_artifacts,
                    delete_llm_cache=request.delete_llm_cache,
                    delete_graph_orphans=request.delete_graph_orphans,
                    strategy=request.strategy,
                    claim_failure_count=len(store_claim_failures),
                    permission_denied_count=len(permission_failures) or None,
                    delete_scopes=scope_by_doc or None,
                ),
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
                    # Capture footprints BEFORE deletion (rebuild_subgraph).
                    pre_delete_footprints: list[dict[str, Any]] = []
                    if request.strategy == "rebuild_subgraph" and index_service is not None:
                        footprint_rag = cast(Any, await active_registry.get(kb_id))
                        serialized_footprints: list[dict[str, Any]] = []
                        for document in documents:
                            footprint = await _capture_graph_footprint(
                                rag=footprint_rag,
                                lightrag_doc_id=document.lightrag_doc_id,
                            )
                            pre_delete_footprints.append(footprint)
                            serialized_footprints.append(
                                _serialize_graph_footprint(
                                    footprint,
                                    document_id=document.id,
                                    lightrag_doc_id=document.lightrag_doc_id,
                                )
                            )
                        await job_service.update_job_payload_patch(
                            kb_id,
                            job.id,
                            payload_patch={"rebuild_subgraph_footprints": serialized_footprints},
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
                    if (
                        request.strategy == "rebuild_kb"
                        and completed_items > 0
                        and index_service is not None
                    ):
                        final_result["rebuild"] = await _run_conservative_kb_rebuild(
                            document_service=document_service,
                            index_service=index_service,
                            registry=active_registry,
                            kb_id=kb_id,
                        )
                    elif (
                        request.strategy == "rebuild_subgraph"
                        and completed_items > 0
                        and index_service is not None
                    ):
                        final_result["rebuild"] = await _run_subgraph_rebuild(
                            document_service=document_service,
                            index_service=index_service,
                            registry=active_registry,
                            kb_id=kb_id,
                            footprint=_merge_footprints(pre_delete_footprints),
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
        http_request: Request,
        request: BatchParseDocumentsRequest,
    ):
        if registry is None:
            raise HTTPException(
                status_code=503,
                detail="KB parse service is not configured",
            )
        active_registry = cast(LightRAGInstanceRegistry, registry)
        try:
            # :batch-parse is parse-only (see route summary). Force auto_index
            # False so the persisted payload stays parse-only: a durable-worker
            # resume (_run_aggregate reads payload["auto_index"]) then behaves
            # identically to the in-process path, which never builds. Clients
            # build with :batch-build-kg.
            batch_plan = await document_service.create_batch_parse_plan(
                kb_id,
                request.document_ids,
                parser_engine=request.engine,
                process_options=request.process_options,
                force_reparse=request.force_reparse,
                auto_index=False,
            )
            job, created_job = await job_service.create_batch_parse_job_once(
                kb_id,
                batch_id=batch_plan.batch_id,
                document_ids=request.document_ids,
                total_items=len(request.document_ids),
                plan_items=[_parse_plan_payload(plan) for plan in batch_plan.plans],
                planning_failures=batch_plan.failures,
                force_reparse=request.force_reparse,
                auto_index=False,
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

            await _append_kb_document_audit_event(
                http_request,
                "documents_batch_parse_queued",
                kb_id,
                _document_audit_metadata(
                    job=job,
                    operation="batch_parse_documents",
                    document_ids=[document.id for document in queued_documents],
                    document_count=len(request.document_ids),
                    batch_id=job.batch_id or batch_plan.batch_id,
                    force_reparse=request.force_reparse,
                    parser_engine=request.engine,
                    claim_failure_count=len(claim_failures),
                    planning_failure_count=len(batch_plan.failures),
                ),
            )

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
                    if execution_plans and rag is None:
                        raise RuntimeError(
                            "KB parse service did not return a LightRAG instance"
                        )
                    parse_concurrency = max(
                        1,
                        int(getattr(rag, "max_parallel_parse_mineru", 1) or 1),
                    ) if rag is not None else 1
                    parse_sem = asyncio.Semaphore(parse_concurrency)

                    async def _do_one_parse(plan: Any) -> dict[str, Any]:
                        async with parse_sem:
                            return await _execute_parse_plan(
                                document_service=document_service,
                                kb_id=kb_id,
                                job_id=job.id,
                                plan=plan,
                                rag=rag,
                                job_service=job_service,
                            )

                    raw_items = await asyncio.gather(
                        *[_do_one_parse(plan) for plan in execution_plans],
                        return_exceptions=True,
                    )
                    for plan, outcome in zip(execution_plans, raw_items):
                        if isinstance(outcome, BaseException):
                            item = {
                                "document_id": plan.document.id,
                                "status": "failed",
                                "error_code": "parse_failed",
                                "error_message": str(outcome),
                            }
                        else:
                            item = outcome
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
            logger.error(
                "Failed to start batch parse for KB '%s': %s",
                kb_id,
                exc,
                exc_info=True,
            )
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
        http_request: Request,
        request: ParseDocumentRequest = Body(default_factory=ParseDocumentRequest),
    ):
        if registry is None:
            raise HTTPException(
                status_code=503,
                detail="KB parse service is not configured",
            )
        active_registry = cast(LightRAGInstanceRegistry, registry)
        try:
            # :parse is parse-only (see route summary). Force auto_index False
            # so the plan/payload stay parse-only and a durable-worker resume
            # matches the in-process path (neither builds). Build with :build-kg.
            plan = await document_service.create_parse_plan(
                kb_id,
                document_id,
                parser_engine=request.engine,
                process_options=request.process_options,
                force_reparse=request.force_reparse,
                auto_index=False,
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

            await _append_kb_document_audit_event(
                http_request,
                "document_parse_queued",
                kb_id,
                _document_audit_metadata(
                    job=job,
                    operation="parse_document",
                    document_ids=[document_id],
                    force_reparse=request.force_reparse,
                    parser_engine=plan.parser_engine,
                    parser_hash=plan.parser_hash,
                ),
            )

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
                        job_service=job_service,
                    )
                    if item["status"] == "cancelled":
                        await job_service.transition_job(
                            kb_id,
                            job.id,
                            status="cancelled",
                            progress=1.0,
                            error_code="cancelled_by_user",
                            error_message=item.get("error_message"),
                        )
                    elif item["status"] == "succeeded":
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
            # An unexpected exception here is by definition undiagnosed, and
            # the message alone rarely says where it came from -- keep the
            # traceback so the next unknown 500 is readable from the logs.
            logger.error(
                "Failed to start parse for KB '%s': %s", kb_id, exc, exc_info=True
            )
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
        http_request: Request,
        job_type: str = "build_kg",
    ) -> JobResponse:
        if registry is None or index_service is None:
            raise HTTPException(
                status_code=503,
                detail="KB index build service is not configured",
            )
        active_registry = cast(LightRAGInstanceRegistry, registry)
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
            job_type=job_type,
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

        await _append_kb_document_audit_event(
            http_request,
            "document_reindex_queued" if job_type == "reindex" else "document_build_queued",
            kb_id,
            _document_audit_metadata(
                job=job,
                operation="reindex_document" if job_type == "reindex" else "build_document_kg",
                document_ids=[document_id],
                parser_hash=plan.parser_hash,
                index_hash=plan.index_hash,
                force_rechunk=force_rechunk,
                force_extract=force_extract,
                force_embedding=force_embedding,
                skipped=plan.skipped,
                skip_reason=plan.skip_reason,
            ),
        )

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

        async def _run_build_task() -> None:
            try:
                await job_service.transition_job(
                    kb_id, job.id, status="running", progress=0.1
                )
                inner_rag = await active_registry.get(kb_id)
                async with _mirror_pipeline_progress(
                    job_service=job_service,
                    kb_id=kb_id,
                    job_id=job.id,
                    rag=inner_rag,
                ):
                    item = await _execute_build_plan(
                        index_service=active_index_service,
                        kb_id=kb_id,
                        job_id=job.id,
                        plan=plan,
                        rag=inner_rag,
                        job_service=job_service,
                    )
                if item["status"] == "cancelled":
                    await job_service.transition_job(
                        kb_id,
                        job.id,
                        status="cancelled",
                        progress=1.0,
                        error_code="cancelled_by_user",
                        error_message=item.get("error_message"),
                    )
                elif item["status"] == "succeeded":
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

        async def _build_task() -> None:
            async with job_service.job_execution_guard(
                job.id, wait=False
            ) as acquired:
                if not acquired:
                    return
                try:
                    current = await job_service.get_persisted_job(job)
                except MetadataRecordNotFoundError:
                    return
                if current.status != "queued" or not _same_job_execution_identity(
                    job, current
                ):
                    return
                await _run_build_task()

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
        http_request: Request,
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
                http_request=http_request,
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
        http_request: Request,
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
                http_request=http_request,
                job_type="reindex",
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
        http_request: Request,
        job_type: str = "build_kg",
        audit_event_type: str | None = None,
        audit_operation: str | None = None,
    ) -> DocumentBatchResponse:
        if registry is None or index_service is None:
            raise HTTPException(
                status_code=503,
                detail="KB index build service is not configured",
            )
        active_registry = cast(LightRAGInstanceRegistry, registry)
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
            job_type=job_type,
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

        default_event_type = (
            "documents_batch_reindex_queued"
            if job_type == "reindex"
            else "documents_batch_build_queued"
        )
        await _append_kb_document_audit_event(
            http_request,
            audit_event_type or default_event_type,
            kb_id,
            _document_audit_metadata(
                job=job,
                operation=audit_operation
                or ("batch_reindex_documents" if job_type == "reindex" else "batch_build_documents"),
                document_ids=[document.id for document in queued_documents],
                document_count=len(request_ids),
                batch_id=job.batch_id or batch_plan.batch_id,
                force_rechunk=force_rechunk,
                force_extract=force_extract,
                force_embedding=force_embedding,
                claim_failure_count=len(claim_failures),
                planning_failure_count=len(batch_plan.failures),
                skipped_count=len(skipped_plans),
            ),
        )

        async def _run_batch_build_task() -> None:
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
                total_build_items = len(request_ids)
                progress_mirror = (
                    _mirror_pipeline_progress(
                        job_service=job_service,
                        kb_id=kb_id,
                        job_id=job.id,
                        rag=inner_rag,
                    )
                    if inner_rag is not None
                    else contextlib.nullcontext()
                )
                async with progress_mirror:
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
                        # Publish per-document progress as each doc finishes so a
                        # multi-doc rebuild/reindex visibly advances instead of
                        # sitting at 0/N until the whole batch completes.
                        try:
                            await job_service.update_job_progress(
                                kb_id,
                                job.id,
                                completed_items=completed_items,
                                progress=(completed_items + failed_items)
                                / max(total_build_items, 1),
                            )
                        except Exception:  # noqa: BLE001 — progress is best-effort
                            pass

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

        async def _batch_build_task() -> None:
            async with job_service.job_execution_guard(
                job.id, wait=False
            ) as acquired:
                if not acquired:
                    return
                try:
                    current = await job_service.get_persisted_job(job)
                except MetadataRecordNotFoundError:
                    return
                if current.status != "queued" or not _same_job_execution_identity(
                    job, current
                ):
                    return
                await _run_batch_build_task()

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
        http_request: Request,
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
                http_request=http_request,
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
        http_request: Request,
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
                http_request=http_request,
                job_type="reindex",
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
        http_request: Request,
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
                http_request=http_request,
                audit_event_type="kb_rebuild_queued",
                audit_operation="rebuild_kb",
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
    async def cancel_job(kb_id: str, job_id: str, http_request: Request):
        try:
            job, changed = await job_service.cancel_job(
                kb_id, job_id, include_deleted=True
            )
            await _append_kb_document_audit_event(
                http_request,
                "job_cancel_requested",
                kb_id,
                _document_audit_metadata(
                    job=job,
                    operation="cancel_job",
                    document_ids=[job.document_id] if job.document_id else (),
                    batch_id=job.batch_id,
                    changed=changed,
                    status=job.status,
                ),
            )
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
        http_request: Request,
        request: JobRetryRequest = Body(default_factory=JobRetryRequest),
    ):
        try:
            job = await job_service.retry_job(
                kb_id,
                job_id,
                new_idempotency_key=request.idempotency_key,
                include_deleted=True,
            )
            await _append_kb_document_audit_event(
                http_request,
                "job_retry_queued",
                kb_id,
                _document_audit_metadata(
                    job=job,
                    operation="retry_job",
                    document_ids=[job.document_id] if job.document_id else (),
                    batch_id=job.batch_id,
                    original_job_id=job_id,
                    status=job.status,
                ),
            )
            return JobResponse.from_record(job)
        except (KnowledgeBaseNotFoundError, MetadataRecordNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except InvalidJobTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get(
        "/{kb_id}/documents/{document_id}/preview",
        response_model=DocumentPreviewManifestResponse,
        dependencies=[Depends(combined_auth)],
        summary="Get a manifest of preview variants for a knowledge base document",
    )
    async def get_document_preview_manifest(kb_id: str, document_id: str):
        try:
            return await document_service.get_document_preview_manifest(
                kb_id, document_id
            )
        except (KnowledgeBaseNotFoundError, MetadataRecordNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
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
        request: Request, kb_id: str, document_id: str, artifact_id: str
    ):
        try:
            artifact = await document_service.get_document_artifact(
                kb_id, document_id, artifact_id
            )
            await _enforce_artifact_content_policy(
                request, kb_id, artifact, action="download"
            )
            artifact_file = await document_service.get_document_artifact_file(
                kb_id, document_id, artifact_id
            )
            await append_enterprise_audit_event(
                request,
                "artifact_downloaded",
                target_type="kb",
                target_id=kb_id,
                metadata=_artifact_audit_metadata(
                    artifact_file.artifact,
                    is_directory=artifact_file.is_directory,
                    filename=artifact_file.filename,
                    media_type=artifact_file.media_type,
                ),
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
        "/{kb_id}/documents/{document_id}/artifacts/{artifact_id}:preview",
        dependencies=[Depends(combined_auth)],
        summary="Preview a supported knowledge base document artifact inline",
    )
    async def preview_document_artifact(
        request: Request, kb_id: str, document_id: str, artifact_id: str
    ):
        try:
            artifact = await document_service.get_document_artifact(
                kb_id, document_id, artifact_id
            )
            await _enforce_artifact_content_policy(
                request, kb_id, artifact, action="preview"
            )
            artifact_file = await document_service.get_document_artifact_file(
                kb_id, document_id, artifact_id
            )
            await append_enterprise_audit_event(
                request,
                "artifact_previewed",
                target_type="kb",
                target_id=kb_id,
                metadata=_artifact_audit_metadata(
                    artifact_file.artifact,
                    filename=artifact_file.filename,
                    media_type=artifact_file.media_type,
                ),
            )
            return _artifact_preview_response(artifact_file)
        except (KnowledgeBaseNotFoundError, MetadataRecordNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get(
        "/{kb_id}/documents/{document_id}/artifacts/{artifact_id}:download-url",
        response_model=ArtifactDownloadUrlResponse,
        dependencies=[Depends(combined_auth)],
        summary="Create a presigned download URL for an object-backed file artifact",
    )
    async def create_document_artifact_download_url(
        request: Request,
        kb_id: str,
        document_id: str,
        artifact_id: str,
        expires_in_seconds: int = 3600,
    ):
        try:
            expires_in_seconds = max(
                1,
                min(expires_in_seconds, _MAX_PRESIGNED_URL_EXPIRES_SECONDS),
            )
            artifact = await document_service.get_document_artifact(
                kb_id, document_id, artifact_id
            )
            await _enforce_artifact_content_policy(
                request, kb_id, artifact, action="download-url"
            )
            result = await document_service.get_document_artifact_download_url(
                kb_id,
                document_id,
                artifact_id,
                expires_in_seconds=expires_in_seconds,
            )
            await append_enterprise_audit_event(
                request,
                "artifact_download_url_created",
                target_type="kb",
                target_id=kb_id,
                metadata=_artifact_audit_metadata(
                    result.artifact,
                    expires_in_seconds=result.expires_in_seconds,
                    filename=result.filename,
                    media_type=result.media_type,
                    object_backed=True,
                ),
            )
            return ArtifactDownloadUrlResponse(
                artifact_id=result.artifact.id,
                url=result.url,
                object_uri="<masked>" if enterprise_mask_storage_uris() else result.object_uri,
                expires_in_seconds=result.expires_in_seconds,
                filename=result.filename,
                media_type=result.media_type,
            )
        except (KnowledgeBaseNotFoundError, MetadataRecordNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
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
                include_deleted=True,
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
        "/{kb_id}/jobs/dead-letter",
        response_model=JobListResponse,
        dependencies=[Depends(combined_auth)],
        summary="List dead-lettered knowledge base jobs",
    )
    async def list_dead_letter_jobs(
        kb_id: str,
        limit: int = 50,
        offset: int = 0,
    ):
        """Jobs that are ``failed`` AND have exhausted ``max_retries`` — they
        will not run again without operator intervention (``:retry`` is
        rejected). Distinct from the general jobs list so operators can triage
        terminal failures separately from still-retryable ones."""
        try:
            jobs, total = await job_service.list_dead_letter_jobs(
                kb_id, limit=limit, offset=offset, include_deleted=True
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
            return JobResponse.from_record(
                await job_service.get_job(kb_id, job_id, include_deleted=True)
            )
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
                job = await job_service.get_job(kb_id, job_id, include_deleted=True)
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
