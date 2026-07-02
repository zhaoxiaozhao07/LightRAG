from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from lightrag.api.agent_profile_service import (
    AgentProfileService,
    AgentProfileUpdateRequest,
)
from lightrag.api.enterprise_auth import append_enterprise_audit_event
from lightrag.api.kb_service import KnowledgeBaseNotFoundError
from lightrag.api.metadata_store import JobRecord
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.utils import logger


class AgentProfileRefreshRequest(BaseModel):
    force: bool = Field(default=False)
    idempotency_key: str | None = Field(default=None, max_length=200)


class AgentProfileRefreshResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    created: bool


def _refresh_response(job: JobRecord, *, created: bool) -> AgentProfileRefreshResponse:
    return AgentProfileRefreshResponse(
        job_id=job.id,
        job_type=job.job_type,
        status=job.status,
        created=created,
    )


def create_agent_profile_routes(
    *,
    profile_service: AgentProfileService,
    api_key: Optional[str] = None,
) -> APIRouter:
    router = APIRouter(prefix="/kbs", tags=["knowledge-base-agent-profile"])
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get(
        "/{kb_id}/agent-profile",
        dependencies=[Depends(combined_auth)],
        summary="Get the manual, auto-generated, and effective Agent profile for a KB",
    )
    async def get_agent_profile(kb_id: str):
        try:
            return await profile_service.get_profile(kb_id)
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.put(
        "/{kb_id}/agent-profile",
        dependencies=[Depends(combined_auth)],
        summary="Update manual Agent profile overrides for a KB",
    )
    async def update_agent_profile(
        kb_id: str, body: AgentProfileUpdateRequest, request: Request
    ):
        try:
            result = await profile_service.update_manual_profile(kb_id, body)
            await append_enterprise_audit_event(
                request,
                "kb_agent_profile_manual_updated",
                target_type="kb",
                target_id=kb_id,
                metadata={"updated_fields": sorted(body.model_fields_set)},
            )
            return result
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post(
        "/{kb_id}/agent-profile:refresh",
        response_model=AgentProfileRefreshResponse,
        dependencies=[Depends(combined_auth)],
        summary="Queue background PROFILE-role regeneration of the KB Agent profile",
    )
    async def refresh_agent_profile(
        kb_id: str,
        body: AgentProfileRefreshRequest,
        request: Request,
    ):
        try:
            # In-process execution (no durable worker) is handled by the
            # service's job runner; with a worker the queued job is claimed by
            # the polling loop. Repeated no-key requests coalesce onto the
            # active job (created=False).
            job, created = await profile_service.enqueue_refresh(
                kb_id,
                force=body.force,
                reason="manual_refresh",
                idempotency_key=body.idempotency_key,
            )
            await append_enterprise_audit_event(
                request,
                "kb_agent_profile_refresh_queued",
                target_type="kb",
                target_id=kb_id,
                metadata={"job_id": job.id, "force": body.force, "created": created},
            )
            return _refresh_response(job, created=created)
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to queue Agent profile refresh for KB '%s': %s", kb_id, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return router
