from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from lightrag.api.config_version_service import ConfigVersionService
from lightrag.api.job_service import JobService
from lightrag.api.kb_deletion_service import (
    KBDeletionService,
    KBHardDeleteInProgressError,
)
from lightrag.api.kb_service import (
    LEGACY_PROVENANCE_METADATA_KEYS,
    KnowledgeBaseConflictError,
    KnowledgeBaseNotFoundError,
    KnowledgeBaseOrigin,
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
    KBLifecycleConflictError,
    MetadataRecordNotFoundError,
)
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.api.enterprise_auth import (
    KB_ROLE_OWNER,
    append_enterprise_audit_event,
    enterprise_auth_enabled,
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
        default="private",
        description=(
            "KB visibility: private (owner/ACL only), internal (implicit "
            "read-only for the KB's tenant members), public (implicit "
            "read-only for all authenticated users; super admin only)"
        ),
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Free-form control-plane metadata (tags, grouping, frontend fields)",
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
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Merged into the stored metadata: provided keys overwrite, "
            "null values delete the key"
        ),
    )

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
    metadata: dict[str, Any] = Field(default_factory=dict)
    origin: KnowledgeBaseOrigin

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


class KnowledgeBaseStatsResponse(BaseModel):
    kb_id: str
    documents: dict[str, Any]
    counters: dict[str, int]
    jobs: dict[str, Any]
    artifacts: dict[str, int]
    graph: dict[str, int] = Field(default_factory=dict)


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


# Bounded full-graph scan so very large graphs cannot OOM the stats read.
# Mirrors ``/kbs/{kb_id}/graph/status``; kept local so the stats endpoint stays
# a single read with no dependency on the graph router module.
_STATS_GRAPH_SCAN_NODES = 100_000


async def _kb_graph_scale(registry: LightRAGInstanceRegistry, kb_id: str) -> dict[str, int]:
    """Return ``{"node_count", "edge_count"}`` from the KB's live graph.

    LightRAG's ``doc_status`` row never carries entity/relation counts (the
    schema has no such columns and the merge path only writes
    ``chunks_count``), so the per-document ``entity_count``/``relation_count``
    projected by the metadata store are always NULL and sum to 0. The control-
    plane aggregate therefore cannot report graph scale; this reads it from the
    KB's knowledge graph instead. The LightRAG instance is loaded on demand
    (registry.get) so the count is correct on a cold server too; this is more
    work than the rest of the stats read but the endpoint is admin-triggered,
    not a hot path. Best-effort: any failure (instance build, storage error)
    returns an empty dict so the stats endpoint stays available rather than
    erroring on graph state.
    """
    try:
        rag = await registry.get(kb_id)
        graph = await rag.get_knowledge_graph(
            node_label="*",
            max_depth=1,
            max_nodes=_STATS_GRAPH_SCAN_NODES,
        )
    except Exception as exc:  # noqa: BLE001 — graph state must not break stats
        logger.warning("KB graph scale read failed for '%s': %s", kb_id, exc)
        return {}
    return {"node_count": len(graph.nodes), "edge_count": len(graph.edges)}


def _hard_delete_worker_enabled(request: Request) -> bool:
    worker = getattr(request.app.state, "job_worker", None)
    resumable_types = getattr(worker, "resumable_job_types", set())
    return "clear_kb" in set(resumable_types)


async def _purge_catalog_generation(
    kb_service: KnowledgeBaseService,
    record: KnowledgeBaseRecord,
) -> bool:
    """Purge only the catalog generation created by the current request."""

    return bool(
        await kb_service.purge(
            record.id,
            expected_generation=record.generation,
            expected_status=record.status,
        )
    )


async def _cleanup_failed_kb_initialization(
    kb_service: KnowledgeBaseService,
    metadata_store: Any,
    record: KnowledgeBaseRecord,
) -> None:
    """Best-effort rollback for a catalog row that was never fully initialized."""

    cleanup_errors: list[str] = []
    try:
        await metadata_store.purge_kb_metadata(
            record.id,
            generation=record.generation,
        )
    except Exception as exc:  # noqa: BLE001 - preserve the initialization error
        cleanup_errors.append(f"metadata: {exc}")
    try:
        await _purge_catalog_generation(kb_service, record)
    except Exception as exc:  # noqa: BLE001 - preserve the initialization error
        cleanup_errors.append(f"catalog: {exc}")
        # Do not leave an active, discoverable half-initialized row if the hard
        # purge itself is temporarily unavailable.
        try:
            current = await kb_service.get(record.id, include_deleted=True)
            if current.generation == record.generation and current.status != "deleted":
                await kb_service.delete(
                    record.id,
                    expected_generation=record.generation,
                )
        except Exception as quarantine_exc:  # noqa: BLE001
            cleanup_errors.append(f"catalog quarantine: {quarantine_exc}")
    if cleanup_errors:
        logger.error(
            "Failed to fully clean up KB initialization for '%s' generation '%s': %s",
            record.id,
            record.generation,
            "; ".join(cleanup_errors),
        )


def _tenant_managed_metadata(
    metadata: dict[str, Any] | None,
    tenant_id: str,
) -> dict[str, Any]:
    """Canonicalize metadata for a non-platform tenant-created KB."""

    normalized = {
        key: value
        for key, value in dict(metadata or {}).items()
        if key not in LEGACY_PROVENANCE_METADATA_KEYS
    }
    raw_tags = normalized.get("tags", [])
    if not isinstance(raw_tags, list) or any(
        not isinstance(tag, str) for tag in raw_tags
    ):
        raise HTTPException(
            status_code=400,
            detail="Knowledge base metadata tags must be a list of strings",
        )
    tenant_tag = f"tenant:{tenant_id}"
    deduplicated: list[str] = []
    seen: set[str] = set()
    for tag in raw_tags:
        if tag.startswith("tenant:") or tag in seen:
            continue
        seen.add(tag)
        deduplicated.append(tag)
    normalized["tags"] = [*deduplicated, tenant_tag]
    return normalized


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
    metadata_store: Any | None = None,
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
            visibility = body.visibility
            metadata = {
                key: value
                for key, value in dict(body.metadata or {}).items()
                if key not in LEGACY_PROVENANCE_METADATA_KEYS
            }
            origin: KnowledgeBaseOrigin = "platform"
            initial_status: KnowledgeBaseStatus = "active"
            if enterprise_auth_enabled():
                initial_status = "creating"
                if principal is None:
                    raise HTTPException(status_code=401, detail="Login required")
                if metadata_store is None:
                    raise HTTPException(
                        status_code=500,
                        detail="Metadata store is required in enterprise mode",
                    )
                if principal.is_super_admin:
                    owner_id = (
                        principal.user_id if body.owner_id is None else body.owner_id
                    )
                    tenant_id = body.tenant_id
                else:
                    owner_id = principal.user_id
                    tenant_id = principal.tenant_id
                    if tenant_id is not None:
                        origin = "tenant"
                        if visibility == "public":
                            raise HTTPException(
                                status_code=400,
                                detail=(
                                    "Tenant-managed knowledge bases must use "
                                    "private or internal visibility"
                                ),
                            )
                        metadata = _tenant_managed_metadata(metadata, tenant_id)
            if metadata_store is not None and body.id is not None:
                # Reject a known id before writing a tentative catalog row when
                # an incomplete hard-delete tail still owns its lifecycle.
                await metadata_store.assert_kb_not_deleting(body.id)
            record = await kb_service.create(
                kb_id=body.id,
                name=body.name,
                description=body.description,
                owner_id=owner_id,
                tenant_id=tenant_id,
                visibility=visibility,
                metadata=metadata,
                origin=origin,
                initial_status=initial_status,
            )
            if metadata_store is not None:
                try:
                    await metadata_store.activate_kb_generation(
                        record.id,
                        record.generation,
                    )
                    if enterprise_auth_enabled() and principal is not None:
                        authz_service = get_enterprise_authorization_service(request)
                        await authz_service.grant_kb_role(
                            record.id,
                            (
                                record.owner_id
                                if record.owner_id is not None
                                else principal.user_id
                            ),
                            KB_ROLE_OWNER,
                            granted_by=principal.user_id,
                            expected_generation=record.generation,
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
                                "origin": record.origin,
                            },
                        )
                        record = await kb_service.update(
                            record.id,
                            status="active",
                            expected_generation=record.generation,
                        )
                except Exception:
                    await _cleanup_failed_kb_initialization(
                        kb_service,
                        metadata_store,
                        record,
                    )
                    raise
            return KnowledgeBaseResponse.from_record(record)
        except HTTPException:
            raise
        except KnowledgeBaseConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except KBLifecycleConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail="Knowledge base changed concurrently; retry the request",
            ) from exc
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
    async def get_knowledge_base(kb_id: str, request: Request):
        try:
            record = await kb_service.get(kb_id)
            return KnowledgeBaseResponse.from_record(record)
        except KnowledgeBaseNotFoundError as exc:
            # Enterprise oversight: a tenant admin/owner may view a soft-deleted
            # KB owned by their tenant (mirrors the list and restore paths) so
            # they can inspect a member's deleted KB before deciding to restore.
            # The catalog row is loaded with include_deleted=True and then gated
            # on the same lifecycle provenance rule as delete/restore.
            if enterprise_auth_enabled():
                try:
                    deleted_record = await kb_service.get(
                        kb_id, include_deleted=True
                    )
                except (KnowledgeBaseNotFoundError, ValueError):
                    deleted_record = None
                if deleted_record is not None and deleted_record.status == "deleted":
                    principal = get_request_principal(request)
                    authz_service = get_enterprise_authorization_service(request)
                    if authz_service.has_tenant_lifecycle_oversight(
                        principal, deleted_record
                    ):
                        return KnowledgeBaseResponse.from_record(deleted_record)
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.patch(
        "/{kb_id}",
        response_model=KnowledgeBaseResponse,
        dependencies=[Depends(combined_auth)],
        summary="Update a knowledge base",
    )
    async def update_knowledge_base(
        kb_id: str,
        request: Request,
        body: KnowledgeBaseUpdateRequest,
    ):
        data = body.model_dump(exclude_unset=True)
        try:
            metadata_patch = data.get("metadata")
            if isinstance(metadata_patch, dict):
                reserved_keys = sorted(
                    LEGACY_PROVENANCE_METADATA_KEYS.intersection(metadata_patch)
                )
                if reserved_keys:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Knowledge base metadata provenance keys are reserved: "
                            + ", ".join(reserved_keys)
                        ),
                    )
            previous_visibility: str | None = None
            if enterprise_auth_enabled():
                principal = get_request_principal(request)
                if principal is None:
                    raise HTTPException(status_code=401, detail="Login required")
                data.pop("owner_id", None)
                data.pop("tenant_id", None)
                if "visibility" in data:
                    current = await kb_service.get(kb_id)
                    previous_visibility = current.visibility
                    if not principal.is_super_admin:
                        if current.origin != "tenant":
                            raise HTTPException(
                                status_code=403,
                                detail=(
                                    "Only super administrators may update knowledge "
                                    "base visibility"
                                ),
                            )
                        if data["visibility"] == "public":
                            raise HTTPException(
                                status_code=400,
                                detail=(
                                    "Tenant-managed knowledge bases must use "
                                    "private or internal visibility"
                                ),
                            )
                        # Only an effective kb_owner (the creator, or an
                        # explicitly granted owner) may toggle sharing.
                        authz_service = get_enterprise_authorization_service(request)
                        await authz_service.require_kb_role(
                            principal, current.id, KB_ROLE_OWNER
                        )
            if "active_config_version_id" in data:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "active_config_version_id must be changed via "
                        "POST /kbs/{kb_id}/configs/{version_id}:activate"
                    ),
                )
            record = await kb_service.update(kb_id, **data)
            if (
                enterprise_auth_enabled()
                and previous_visibility is not None
                and record.visibility != previous_visibility
            ):
                await append_enterprise_audit_event(
                    request,
                    "kb_visibility_changed",
                    target_type="kb",
                    target_id=record.id,
                    metadata={
                        "from": previous_visibility,
                        "to": record.visibility,
                        "origin": record.origin,
                    },
                )
            return KnowledgeBaseResponse.from_record(record)
        except HTTPException:
            raise
        except KnowledgeBaseConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
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
            try:
                authorized_record = await kb_service.get(kb_id, include_deleted=True)
            except KnowledgeBaseNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            if enterprise_auth_enabled():
                principal = get_request_principal(request)
                authz_service = get_enterprise_authorization_service(request)
                await authz_service.authorize_kb_lifecycle(
                    principal,
                    authorized_record,
                    "hard-delete" if hard else "soft-delete",
                )
            if hard:
                # Idempotent hard delete: if the KB is already soft-deleted,
                # skip the (now failing) soft-delete step and run the destructive
                # clear directly. Without this, a hard delete that fails midway
                # (or any soft-delete-then-retry-with-hard flow) gets stuck with
                # the catalog row in 'deleted' status forever.
                record = authorized_record
                if record.status != "deleted":
                    record = await kb_service.delete(
                        kb_id,
                        expected_generation=authorized_record.generation,
                    )
                if deletion_service is None:
                    raise HTTPException(
                        status_code=503,
                        detail="KB hard-delete service is not configured",
                    )
                if _hard_delete_worker_enabled(request):
                    job = await deletion_service.enqueue_hard_delete(
                        authorized_record.id,
                        expected_generation=authorized_record.generation,
                    )
                    if enterprise_auth_enabled():
                        await append_enterprise_audit_event(
                            request,
                            "kb_hard_delete_queued",
                            target_type="kb",
                            target_id=record.id,
                            metadata={
                                "hard": True,
                                "job_id": job.id,
                                "generation": authorized_record.generation,
                            },
                        )
                    return KnowledgeBaseDeleteResponse.from_record_and_job(record, job)
                # Synchronous (in-process) clear when durable worker is disabled.
                # Failures are surfaced via the clear_kb job and HTTP 500.
                result = await deletion_service.hard_delete(
                    authorized_record.id,
                    expected_generation=authorized_record.generation,
                )
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
                record = await kb_service.delete(
                    kb_id,
                    expected_generation=authorized_record.generation,
                )
                try:
                    await registry.discard(record.id)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Soft delete completed but registry discard failed for '%s': %s",
                        kb_id,
                        exc,
                    )
            if enterprise_auth_enabled():
                await append_enterprise_audit_event(
                    request,
                    "kb_hard_deleted" if hard else "kb_deleted",
                    target_type="kb",
                    target_id=record.id,
                    metadata={
                        "hard": hard,
                        "generation": authorized_record.generation,
                    },
                )
            return KnowledgeBaseDeleteResponse.from_record_and_job(record)
        except HTTPException:
            raise
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except KnowledgeBaseConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except KBHardDeleteInProgressError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "error_code": "kb_hard_delete_in_progress",
                    "job_id": exc.job.id,
                    "job_status": exc.job.status,
                },
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("Failed to delete knowledge base '%s': %s", kb_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post(
        "/{kb_id}:restore",
        response_model=KnowledgeBaseResponse,
        dependencies=[Depends(combined_auth)],
        summary="Restore a soft-deleted knowledge base",
    )
    async def restore_knowledge_base(kb_id: str, request: Request):
        try:
            active_metadata_store = metadata_store
            if active_metadata_store is None and deletion_service is not None:
                active_metadata_store = getattr(
                    deletion_service, "_metadata_store", None
                )
            try:
                record = await kb_service.get(kb_id, include_deleted=True)
            except KnowledgeBaseNotFoundError as exc:
                if active_metadata_store is not None:
                    lifecycle = await active_metadata_store.get_kb_lifecycle(kb_id)
                    if lifecycle is not None and lifecycle.state == "deleting":
                        raise HTTPException(
                            status_code=409,
                            detail={
                                "error_code": "kb_hard_delete_in_progress",
                                "job_id": lifecycle.delete_job_id,
                            },
                        ) from exc
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if enterprise_auth_enabled():
                principal = get_request_principal(request)
                authz_service = get_enterprise_authorization_service(request)
                await authz_service.authorize_kb_lifecycle(
                    principal,
                    record,
                    "restore",
                )
            if record.status != "deleted":
                raise HTTPException(
                    status_code=409,
                    detail="Knowledge base is not deleted",
                )

            async def restore_after_guarded_rechecks() -> KnowledgeBaseRecord:
                if active_metadata_store is not None:
                    lifecycle = await active_metadata_store.get_kb_lifecycle(record.id)
                    if lifecycle is not None and lifecycle.state == "deleting":
                        raise HTTPException(
                            status_code=409,
                            detail={
                                "error_code": "kb_hard_delete_in_progress",
                                "job_id": lifecycle.delete_job_id,
                            },
                        )
                if job_service is not None:
                    # Failed clear jobs remain retryable and retain hard-delete
                    # intent, so they block restore just like queued/running jobs.
                    unfinished_jobs, _total = await job_service.list_jobs(
                        record.id,
                        statuses=(
                            "queued",
                            "running",
                            "retrying",
                            "cancelling",
                            "failed",
                        ),
                        limit=200,
                        include_deleted=True,
                    )
                    active_clears = [
                        job
                        for job in unfinished_jobs
                        if job.job_type == "clear_kb"
                        and job.payload.get("kb_generation")
                        in {None, record.generation}
                    ]
                    if active_clears:
                        raise HTTPException(
                            status_code=409,
                            detail={
                                "error_code": "kb_hard_delete_in_progress",
                                "job_id": active_clears[0].id,
                            },
                        )
                return await kb_service.restore(
                    record.id,
                    expected_generation=record.generation,
                )

            if active_metadata_store is not None:
                async with active_metadata_store.kb_write_guard(
                    record.id,
                    record.generation,
                ):
                    restored = await restore_after_guarded_rechecks()
            else:
                restored = await restore_after_guarded_rechecks()
            if enterprise_auth_enabled():
                await append_enterprise_audit_event(
                    request,
                    "kb_restored",
                    target_type="kb",
                    target_id=restored.id,
                    metadata={"generation": record.generation},
                )
            return KnowledgeBaseResponse.from_record(restored)
        except HTTPException:
            raise
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except KnowledgeBaseConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except KBLifecycleConflictError as exc:
            current = exc.current
            if current.get("state") == "deleting":
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error_code": "kb_hard_delete_in_progress",
                        "job_id": current.get("delete_job_id"),
                    },
                ) from exc
            raise HTTPException(
                status_code=409,
                detail="Knowledge base changed concurrently; retry the request",
            ) from exc
        except ValueError as exc:
            # restore() raced with another state change — surface as conflict.
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("Failed to restore knowledge base '%s': %s", kb_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.get(
        "/{kb_id}/stats",
        response_model=KnowledgeBaseStatsResponse,
        dependencies=[Depends(combined_auth)],
        summary="Control-plane statistics for one knowledge base",
    )
    async def get_knowledge_base_stats(kb_id: str):
        """Aggregate document/job/artifact statistics from the metadata store.

        Deliberately control-plane only: no LightRAG instance is loaded, so
        the call is cheap and side-effect free. Graph-scale numbers remain on
        ``GET /kbs/{kb_id}/graph/status``.
        """
        try:
            record = await kb_service.get(kb_id)
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if metadata_store is None:
            raise HTTPException(
                status_code=503, detail="Metadata store is not configured"
            )
        try:
            stats = await metadata_store.aggregate_control_plane_stats(record.id)
        except Exception as exc:
            logger.error("Failed to aggregate stats for '%s': %s", kb_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        documents_by_status = stats["documents_by_status"]
        jobs_by_status = stats["jobs_by_status"]
        # LightRAG's doc_status rows never carry entity/relation counts
        # (the table has no such columns), so the control-plane ``counters``
        # sum them to 0. Read the live graph scale and backfill the
        # ``entities``/``relations`` counters with the real node/edge counts
        # so the overview shows accurate totals; ``graph`` carries the raw
        # counts separately for clients that distinguish them.
        graph_scale = await _kb_graph_scale(registry, record.id)
        counters = dict(stats["document_counters"])
        if graph_scale:
            # The control-plane counter is structurally always 0 (NULL sum);
            # the graph read is the source of truth for entity/relation totals.
            counters["entities"] = graph_scale["node_count"]
            counters["relations"] = graph_scale["edge_count"]
        return KnowledgeBaseStatsResponse(
            kb_id=record.id,
            documents={
                "total": sum(documents_by_status.values()),
                "by_status": documents_by_status,
            },
            counters=counters,
            jobs={
                "total": sum(jobs_by_status.values()),
                "by_status": jobs_by_status,
                "dead_letter": stats["dead_letter_jobs"],
            },
            artifacts={"total": stats["artifacts"]},
            graph=graph_scale,
        )

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
