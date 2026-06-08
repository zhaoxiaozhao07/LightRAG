from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from lightrag.api.config_version_service import ConfigVersionService
from lightrag.api.job_service import JobService
from lightrag.api.kb_deletion_service import KBDeletionService
from lightrag.api.kb_service import (
    KnowledgeBaseConflictError,
    KnowledgeBaseNotFoundError,
    KnowledgeBaseRecord,
    KnowledgeBaseService,
    KnowledgeBaseStatus,
    KnowledgeBaseVisibility,
    validate_kb_id,
)
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry
from lightrag.api.metadata_store import (
    ConfigVersionRecord,
    JobRecord,
    MetadataRecordNotFoundError,
)
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.api.enterprise_auth import (
    KB_ROLE_OWNER,
    append_enterprise_audit_event,
    enterprise_auth_enabled,
    get_enterprise_audit_service,
    get_enterprise_authorization_service,
    get_request_principal,
)
from lightrag.exceptions import PipelineNotInitializedError
from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock
from lightrag.utils import generate_track_id, logger

MutableKnowledgeBaseStatus = Literal[
    "creating", "active", "disabled", "deleting", "error"
]
_CONFIG_REPARSE_STATUSES = ("uploaded", "parsed", "parse_failed", "build_failed", "ready")
_CONFIG_REINDEX_STATUSES = ("parsed", "ready", "build_failed")


class KnowledgeBaseCreateRequest(BaseModel):
    id: Optional[str] = Field(
        default=None,
        description="Optional stable knowledge base id. If omitted, the server generates one.",
    )
    name: str = Field(min_length=1, description="Knowledge base display name")
    description: Optional[str] = Field(default=None, description="Knowledge base description")
    owner_id: Optional[str] = Field(default=None, description="Reserved owner id")
    tenant_id: Optional[str] = Field(default=None, description="Reserved tenant id")
    visibility: KnowledgeBaseVisibility = Field(
        default="private", description="Reserved visibility flag"
    )

    @field_validator("id", mode="after")
    @classmethod
    def validate_id(cls, value: str | None) -> str | None:
        return validate_kb_id(value) if value is not None else None

    @field_validator("name", mode="after")
    @classmethod
    def strip_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Knowledge base name cannot be empty")
        return stripped

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "kb_research",
                "name": "Research Papers",
                "description": "Papers and notes for retrieval",
                "visibility": "private",
            }
        }
    )


class KnowledgeBaseUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1)
    description: Optional[str] = None
    status: Optional[MutableKnowledgeBaseStatus] = None
    owner_id: Optional[str] = None
    tenant_id: Optional[str] = None
    visibility: Optional[KnowledgeBaseVisibility] = None
    active_config_version_id: Optional[str] = None

    @field_validator("name", mode="after")
    @classmethod
    def strip_optional_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("Knowledge base name cannot be empty")
        return stripped


class KnowledgeBaseResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    workspace: str
    status: KnowledgeBaseStatus
    active_config_version_id: Optional[str]
    owner_id: Optional[str]
    tenant_id: Optional[str]
    visibility: KnowledgeBaseVisibility
    created_at: str
    updated_at: str
    deleted_at: Optional[str]

    @classmethod
    def from_record(cls, record: KnowledgeBaseRecord) -> "KnowledgeBaseResponse":
        return cls(**record.to_dict())


class KnowledgeBaseDeleteResponse(KnowledgeBaseResponse):
    hard_delete_queued: bool = False
    hard_delete_job_id: Optional[str] = None
    hard_delete_job_type: Optional[str] = None
    hard_delete_job_status: Optional[str] = None

    @classmethod
    def from_record_and_job(
        cls, record: KnowledgeBaseRecord, job: JobRecord | None = None
    ) -> "KnowledgeBaseDeleteResponse":
        data = record.to_dict()
        if job is not None:
            data.update(
                {
                    "hard_delete_queued": job.status == "queued",
                    "hard_delete_job_id": job.id,
                    "hard_delete_job_type": job.job_type,
                    "hard_delete_job_status": job.status,
                }
            )
        return cls(**data)


class KnowledgeBaseListResponse(BaseModel):
    knowledge_bases: list[KnowledgeBaseResponse]
    total: int


class KnowledgeBaseStatusResponse(BaseModel):
    kb: KnowledgeBaseResponse
    instance_loaded: bool
    pipeline_initialized: bool
    pipeline_status: dict[str, Any]
    storage_workspaces: dict[str, Any]
    running_jobs: list[dict[str, Any]] = Field(default_factory=list)


class ConfigVersionCreateRequest(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)
    created_by: Optional[str] = None


class ConfigVersionActivateRequest(BaseModel):
    auto_enqueue: bool = Field(
        default=False,
        description=(
            "When true, activation creates idempotent parse/reindex follow-up jobs "
            "for documents affected by parser/index/vector changes."
        ),
    )


class ConfigVersionResponse(BaseModel):
    id: str
    kb_id: str
    workspace: str
    version: int
    config: dict[str, Any]
    parser_hash: Optional[str]
    index_hash: Optional[str]
    query_hash: Optional[str]
    created_at: str
    activated_at: Optional[str]
    created_by: Optional[str]

    @classmethod
    def from_record(cls, record: ConfigVersionRecord) -> "ConfigVersionResponse":
        return cls(**record.to_dict())


class ConfigVersionListResponse(BaseModel):
    versions: list[ConfigVersionResponse]
    total: int
    limit: int
    offset: int


class ConfigVersionDiffResponse(BaseModel):
    target_version_id: str
    active_version_id: Optional[str]
    requires_reparse: bool
    requires_reindex: bool
    requires_vector_rebuild: bool
    reasons: list[str]


class ConfigActivationJobResponse(BaseModel):
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
    payload: dict[str, Any]
    created_at: str
    updated_at: str

    @classmethod
    def from_record(cls, record: JobRecord) -> "ConfigActivationJobResponse":
        data = record.to_dict()
        return cls(
            id=data["id"],
            kb_id=data["kb_id"],
            workspace=data["workspace"],
            batch_id=data["batch_id"],
            document_id=data["document_id"],
            job_type=data["job_type"],
            status=data["status"],
            stage=data["stage"],
            progress=data["progress"],
            total_items=data["total_items"],
            completed_items=data["completed_items"],
            failed_items=data["failed_items"],
            idempotency_key=data["idempotency_key"],
            config_version_id=data["config_version_id"],
            config_hash=data["config_hash"],
            payload=data["payload"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )


class ConfigVersionActivationResponse(ConfigVersionResponse):
    auto_enqueue: bool = False
    diff: ConfigVersionDiffResponse
    follow_up_jobs: list[ConfigActivationJobResponse] = Field(default_factory=list)
    follow_up_noop_reason: Optional[str] = None

    @classmethod
    def from_record_and_followups(
        cls,
        record: ConfigVersionRecord,
        *,
        auto_enqueue: bool,
        diff: ConfigVersionDiffResponse,
        follow_up_jobs: list[JobRecord],
        follow_up_noop_reason: str | None,
    ) -> "ConfigVersionActivationResponse":
        data = record.to_dict()
        data.update(
            {
                "auto_enqueue": auto_enqueue,
                "diff": diff,
                "follow_up_jobs": [
                    ConfigActivationJobResponse.from_record(job)
                    for job in follow_up_jobs
                ],
                "follow_up_noop_reason": follow_up_noop_reason,
            }
        )
        return cls(**data)


async def _copy_pipeline_status(workspace: str) -> tuple[bool, dict[str, Any]]:
    try:
        pipeline_status = await get_namespace_data("pipeline_status", workspace=workspace)
    except PipelineNotInitializedError:
        return False, {}

    async with get_namespace_lock("pipeline_status", workspace=workspace):
        copied = dict(pipeline_status)
        history_messages = copied.get("history_messages", [])
        copied["history_messages"] = list(history_messages)[-1000:]
        return True, copied


def _storage_workspaces_for_rag(rag: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for attr in (
        "llm_response_cache",
        "full_docs",
        "text_chunks",
        "entities_vdb",
        "relationships_vdb",
        "chunks_vdb",
        "chunk_entity_relation_graph",
        "doc_status",
    ):
        storage = getattr(rag, attr, None)
        if storage is not None:
            result[attr] = getattr(storage, "workspace", None)
    return result


def _hard_delete_worker_enabled(request: Request) -> bool:
    worker = getattr(request.app.state, "job_worker", None)
    resumable_types = getattr(worker, "resumable_job_types", set())
    return "clear_kb" in set(resumable_types)


def _config_follow_up_idempotency_key(
    kb_id: str, version_id: str, job_type: str, document_ids: list[str]
) -> str:
    payload = {
        "kb_id": kb_id,
        "version_id": version_id,
        "job_type": job_type,
        "document_ids": document_ids,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return f"config-activation:{version_id}:{job_type}:{digest}"


async def _enqueue_config_activation_followups(
    kb_id: str,
    version_id: str,
    diff: ConfigVersionDiffResponse,
    active_job_service: JobService | None,
) -> tuple[list[JobRecord], str | None]:
    if active_job_service is None:
        return [], "job_service_unavailable"
    if diff.requires_reparse:
        document_ids = await active_job_service.list_document_ids_for_config_followup(
            kb_id, statuses=_CONFIG_REPARSE_STATUSES
        )
        if not document_ids:
            return [], "no_eligible_documents"
        job, _created = await active_job_service.create_batch_parse_job_once(
            kb_id,
            batch_id=generate_track_id("batch"),
            document_ids=document_ids,
            total_items=len(document_ids),
            plan_items=[],
            planning_failures=[],
            force_reparse=True,
            auto_index=True,
            idempotency_key=_config_follow_up_idempotency_key(
                kb_id, version_id, "parse", document_ids
            ),
        )
        return [job], None
    if diff.requires_reindex or diff.requires_vector_rebuild:
        document_ids = await active_job_service.list_document_ids_for_config_followup(
            kb_id, statuses=_CONFIG_REINDEX_STATUSES
        )
        if not document_ids:
            return [], "no_eligible_documents"
        job, _created = await active_job_service.create_batch_build_job_once(
            kb_id,
            batch_id=generate_track_id("batch"),
            document_ids=document_ids,
            total_items=len(document_ids),
            plan_items=[],
            planning_failures=[],
            force_rechunk=True,
            force_extract=True,
            force_embedding=True,
            job_type="reindex",
            idempotency_key=_config_follow_up_idempotency_key(
                kb_id, version_id, "reindex", document_ids
            ),
        )
        return [job], None
    return [], "no_rebuild_required"


def create_kb_routes(
    kb_service: KnowledgeBaseService,
    registry: LightRAGInstanceRegistry,
    api_key: Optional[str] = None,
    job_service: JobService | None = None,
    config_service: ConfigVersionService | None = None,
    deletion_service: "KBDeletionService | None" = None,
):
    router = APIRouter(prefix="/kbs", tags=["knowledge-bases"])
    combined_auth = get_combined_auth_dependency(api_key)

    @router.post(
        "",
        response_model=KnowledgeBaseResponse,
        dependencies=[Depends(combined_auth)],
        summary="Create a knowledge base",
    )
    async def create_knowledge_base(request: Request, body: KnowledgeBaseCreateRequest):
        try:
            principal = get_request_principal(request)
            owner_id = body.owner_id
            tenant_id = body.tenant_id
            if enterprise_auth_enabled():
                if principal is None:
                    raise HTTPException(status_code=401, detail="Login required")
                owner_id = principal.user_id
                tenant_id = principal.tenant_id
            record = await kb_service.create(
                kb_id=body.id,
                name=body.name,
                description=body.description,
                owner_id=owner_id,
                tenant_id=tenant_id,
                visibility=body.visibility,
            )
            if enterprise_auth_enabled() and principal is not None:
                try:
                    authz_service = get_enterprise_authorization_service(request)
                    await authz_service.grant_kb_role(
                        record.id,
                        principal.user_id,
                        KB_ROLE_OWNER,
                        granted_by=principal.user_id,
                    )
                    await append_enterprise_audit_event(
                        request,
                        "kb_created",
                        target_type="kb",
                        target_id=record.id,
                        metadata={
                            "workspace": record.workspace,
                            "tenant_id": record.tenant_id,
                            "visibility": record.visibility,
                        },
                    )
                except Exception:
                    await kb_service.delete(record.id)
                    raise
            return KnowledgeBaseResponse.from_record(record)
        except KnowledgeBaseConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("Failed to create knowledge base: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.get(
        "",
        response_model=KnowledgeBaseListResponse,
        dependencies=[Depends(combined_auth)],
        summary="List knowledge bases",
    )
    async def list_knowledge_bases(request: Request, include_deleted: bool = False):
        records = await kb_service.list(include_deleted=include_deleted)
        if enterprise_auth_enabled():
            authz_service = get_enterprise_authorization_service(request)
            records = await authz_service.filter_kbs_for_principal(
                get_request_principal(request), records
            )
        items = [KnowledgeBaseResponse.from_record(record) for record in records]
        return KnowledgeBaseListResponse(knowledge_bases=items, total=len(items))

    @router.get(
        "/{kb_id}",
        response_model=KnowledgeBaseResponse,
        dependencies=[Depends(combined_auth)],
        summary="Get a knowledge base",
    )
    async def get_knowledge_base(kb_id: str):
        try:
            record = await kb_service.get(kb_id)
            return KnowledgeBaseResponse.from_record(record)
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.patch(
        "/{kb_id}",
        response_model=KnowledgeBaseResponse,
        dependencies=[Depends(combined_auth)],
        summary="Update a knowledge base",
    )
    async def update_knowledge_base(kb_id: str, request: KnowledgeBaseUpdateRequest):
        data = request.model_dump(exclude_unset=True)
        if enterprise_auth_enabled():
            data.pop("owner_id", None)
            data.pop("tenant_id", None)
        if "active_config_version_id" in data:
            raise HTTPException(
                status_code=400,
                detail=(
                    "active_config_version_id must be changed via "
                    "POST /kbs/{kb_id}/configs/{version_id}:activate"
                ),
            )
        try:
            record = await kb_service.update(kb_id, **data)
            return KnowledgeBaseResponse.from_record(record)
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("Failed to update knowledge base '%s': %s", kb_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.delete(
        "/{kb_id}",
        response_model=KnowledgeBaseDeleteResponse,
        dependencies=[Depends(combined_auth)],
        summary="Delete a knowledge base",
    )
    async def delete_knowledge_base(kb_id: str, request: Request, hard: bool = False):
        try:
            if hard:
                # Idempotent hard delete: if the KB is already soft-deleted,
                # skip the (now failing) soft-delete step and run the destructive
                # clear directly. Without this, a hard delete that fails midway
                # (or any soft-delete-then-retry-with-hard flow) gets stuck with
                # the catalog row in 'deleted' status forever.
                try:
                    record = await kb_service.get(kb_id, include_deleted=True)
                except KnowledgeBaseNotFoundError as exc:
                    raise HTTPException(status_code=404, detail=str(exc)) from exc
                if record.status != "deleted":
                    record = await kb_service.delete(kb_id)
                if deletion_service is None:
                    raise HTTPException(
                        status_code=503,
                        detail="KB hard-delete service is not configured",
                    )
                if _hard_delete_worker_enabled(request):
                    try:
                        await registry.discard(record.id)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Hard delete queued but registry discard failed for '%s': %s",
                            kb_id,
                            exc,
                        )
                    job = await deletion_service.enqueue_hard_delete(record.id)
                    if enterprise_auth_enabled():
                        principal = get_request_principal(request)
                        audit_service = get_enterprise_audit_service(request)
                        await audit_service.append(
                            "kb_hard_delete_queued",
                            actor_user_id=principal.user_id if principal else None,
                            target_type="kb",
                            target_id=record.id,
                            metadata={"hard": True, "job_id": job.id},
                        )
                    return KnowledgeBaseDeleteResponse.from_record_and_job(record, job)
                # Synchronous (in-process) clear when durable worker is disabled.
                # Failures are surfaced via the clear_kb job and HTTP 500.
                result = await deletion_service.hard_delete(record.id)
                if result.errors:
                    raise HTTPException(
                        status_code=500,
                        detail={
                            "error_code": "kb_hard_delete_failed",
                            "errors": result.errors,
                            "job_id": result.job.id,
                        },
                    )
            else:
                record = await kb_service.delete(kb_id)
                try:
                    await registry.discard(record.id)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Soft delete completed but registry discard failed for '%s': %s",
                        kb_id,
                        exc,
                    )
            if enterprise_auth_enabled():
                principal = get_request_principal(request)
                audit_service = get_enterprise_audit_service(request)
                await audit_service.append(
                    "kb_hard_deleted" if hard else "kb_deleted",
                    actor_user_id=principal.user_id if principal else None,
                    target_type="kb",
                    target_id=record.id,
                    metadata={"hard": hard},
                )
            return KnowledgeBaseDeleteResponse.from_record_and_job(record)
        except HTTPException:
            raise
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("Failed to delete knowledge base '%s': %s", kb_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.get(
        "/{kb_id}/status",
        response_model=KnowledgeBaseStatusResponse,
        dependencies=[Depends(combined_auth)],
        summary="Get knowledge base status",
    )
    async def get_knowledge_base_status(kb_id: str):
        try:
            record = await kb_service.get(kb_id)
            instance_loaded = registry.is_loaded(record.id)
            pipeline_initialized, pipeline_status = await _copy_pipeline_status(
                record.workspace
            )
            storage_workspaces: dict[str, Any] = {}
            if instance_loaded:
                entry = await registry.get_entry(record.id)
                storage_workspaces = _storage_workspaces_for_rag(entry.rag)
            running_jobs = []
            if job_service is not None:
                running_jobs = [
                    job.to_dict()
                    for job in await job_service.list_running_jobs(record.id)
                ]
            return KnowledgeBaseStatusResponse(
                kb=KnowledgeBaseResponse.from_record(record),
                instance_loaded=instance_loaded,
                pipeline_initialized=pipeline_initialized,
                pipeline_status=pipeline_status,
                storage_workspaces=storage_workspaces,
                running_jobs=running_jobs,
            )
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("Failed to get knowledge base status '%s': %s", kb_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    if config_service is not None:
        active_config_service = config_service

        @router.post(
            "/{kb_id}/configs",
            response_model=ConfigVersionResponse,
            dependencies=[Depends(combined_auth)],
            summary="Create a new KB configuration version",
        )
        async def create_config_version(
            kb_id: str, request: Request, body: ConfigVersionCreateRequest
        ):
            try:
                created_by = body.created_by
                if enterprise_auth_enabled():
                    principal = get_request_principal(request)
                    if principal is None:
                        raise HTTPException(status_code=401, detail="Login required")
                    created_by = principal.user_id
                record = await active_config_service.create(
                    kb_id, config=body.config, created_by=created_by
                )
                return ConfigVersionResponse.from_record(record)
            except KnowledgeBaseNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        @router.get(
            "/{kb_id}/configs",
            response_model=ConfigVersionListResponse,
            dependencies=[Depends(combined_auth)],
            summary="List KB configuration versions",
        )
        async def list_config_versions(
            kb_id: str, limit: int = 50, offset: int = 0
        ):
            try:
                versions, total = await active_config_service.list(
                    kb_id, limit=limit, offset=offset
                )
                return ConfigVersionListResponse(
                    versions=[
                        ConfigVersionResponse.from_record(item) for item in versions
                    ],
                    total=total,
                    limit=max(1, min(limit, 200)),
                    offset=max(0, offset),
                )
            except KnowledgeBaseNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        @router.get(
            "/{kb_id}/configs/{version_id}",
            response_model=ConfigVersionResponse,
            dependencies=[Depends(combined_auth)],
            summary="Get a KB configuration version",
        )
        async def get_config_version(kb_id: str, version_id: str):
            try:
                return ConfigVersionResponse.from_record(
                    await active_config_service.get(kb_id, version_id)
                )
            except (KnowledgeBaseNotFoundError, MetadataRecordNotFoundError) as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        @router.post(
            "/{kb_id}/configs/{version_id}:activate",
            response_model=ConfigVersionActivationResponse,
            dependencies=[Depends(combined_auth)],
            summary="Activate a KB configuration version",
        )
        async def activate_config_version(
            request: Request,
            kb_id: str,
            version_id: str,
            body: ConfigVersionActivateRequest | None = None,
        ):
            try:
                diff = ConfigVersionDiffResponse(
                    **await active_config_service.diff(kb_id, version_id)
                )
                auto_enqueue = body.auto_enqueue if body is not None else False
                record = await active_config_service.activate(kb_id, version_id)
                follow_up_jobs: list[JobRecord] = []
                follow_up_noop_reason: str | None = None
                if auto_enqueue:
                    (
                        follow_up_jobs,
                        follow_up_noop_reason,
                    ) = await _enqueue_config_activation_followups(
                        kb_id, version_id, diff, job_service
                    )
                await append_enterprise_audit_event(
                    request,
                    "kb_config_activated",
                    target_type="kb",
                    target_id=kb_id,
                    metadata={
                        "version_id": record.id,
                        "version": record.version,
                        "parser_hash": record.parser_hash,
                        "index_hash": record.index_hash,
                        "query_hash": record.query_hash,
                        "auto_enqueue": auto_enqueue,
                        "follow_up_job_ids": [job.id for job in follow_up_jobs],
                        "follow_up_noop_reason": follow_up_noop_reason,
                    },
                )
                return ConfigVersionActivationResponse.from_record_and_followups(
                    record,
                    auto_enqueue=auto_enqueue,
                    diff=diff,
                    follow_up_jobs=follow_up_jobs,
                    follow_up_noop_reason=follow_up_noop_reason,
                )
            except (KnowledgeBaseNotFoundError, MetadataRecordNotFoundError) as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        @router.post(
            "/{kb_id}/configs/{version_id}:diff",
            response_model=ConfigVersionDiffResponse,
            dependencies=[Depends(combined_auth)],
            summary="Diff a KB configuration version against the active one",
        )
        async def diff_config_version(kb_id: str, version_id: str):
            try:
                return ConfigVersionDiffResponse(
                    **await active_config_service.diff(kb_id, version_id)
                )
            except (KnowledgeBaseNotFoundError, MetadataRecordNotFoundError) as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router
