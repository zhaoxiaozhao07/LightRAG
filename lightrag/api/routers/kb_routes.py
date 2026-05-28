from __future__ import annotations

from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from lightrag.api.job_service import JobService
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
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.exceptions import PipelineNotInitializedError
from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock
from lightrag.utils import logger

MutableKnowledgeBaseStatus = Literal[
    "creating", "active", "disabled", "deleting", "error"
]


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


def create_kb_routes(
    kb_service: KnowledgeBaseService,
    registry: LightRAGInstanceRegistry,
    api_key: Optional[str] = None,
    job_service: JobService | None = None,
):
    router = APIRouter(prefix="/kbs", tags=["knowledge-bases"])
    combined_auth = get_combined_auth_dependency(api_key)

    @router.post(
        "",
        response_model=KnowledgeBaseResponse,
        dependencies=[Depends(combined_auth)],
        summary="Create a knowledge base",
    )
    async def create_knowledge_base(request: KnowledgeBaseCreateRequest):
        try:
            record = await kb_service.create(
                kb_id=request.id,
                name=request.name,
                description=request.description,
                owner_id=request.owner_id,
                tenant_id=request.tenant_id,
                visibility=request.visibility,
            )
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
    async def list_knowledge_bases(include_deleted: bool = False):
        records = await kb_service.list(include_deleted=include_deleted)
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
        response_model=KnowledgeBaseResponse,
        dependencies=[Depends(combined_auth)],
        summary="Delete a knowledge base",
    )
    async def delete_knowledge_base(kb_id: str):
        try:
            record = await kb_service.delete(kb_id)
            await registry.discard(record.id)
            return KnowledgeBaseResponse.from_record(record)
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

    return router
