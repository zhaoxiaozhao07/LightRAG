from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from lightrag.api.auth import auth_handler
from lightrag.api.enterprise_auth import (
    Principal,
    get_enterprise_audit_service,
    get_enterprise_authorization_service,
    get_enterprise_settings_service,
    get_enterprise_user_service,
    get_request_principal,
)
from lightrag.api.metadata_store import (
    AuditEventRecord,
    EnterpriseUserRecord,
    KBACLRecord,
)
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag import __version__ as core_version
from lightrag.api import __api_version__


class EnterpriseUserResponse(BaseModel):
    id: str
    username: str
    system_role: str
    status: str
    tenant_id: str | None
    can_create_kb: bool
    can_use_bypass_query: bool
    token_version: int
    created_at: str
    updated_at: str

    @classmethod
    def from_record(cls, record: EnterpriseUserRecord) -> "EnterpriseUserResponse":
        data = record.to_dict()
        data.pop("password_hash", None)
        data.pop("metadata", None)
        return cls(**data)


class EnterpriseRegistrationRequest(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


class EnterpriseChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1)
    new_password: str = Field(min_length=1)


class EnterpriseRegistrationToggleRequest(BaseModel):
    enabled: bool


class EnterpriseUserCreateRequest(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)
    can_create_kb: bool = False
    can_use_bypass_query: bool = False
    tenant_id: str | None = None


class EnterpriseUserUpdateRequest(BaseModel):
    status: str | None = None
    can_create_kb: bool | None = None
    can_use_bypass_query: bool | None = None
    tenant_id: str | None = None
    password: str | None = None


class EnterpriseACLGrantRequest(BaseModel):
    user_id: str = Field(min_length=1)
    role: str = Field(min_length=1)


class EnterpriseACLResponse(BaseModel):
    kb_id: str
    user_id: str
    role: str
    granted_by: str | None
    created_at: str
    updated_at: str

    @classmethod
    def from_record(cls, record: KBACLRecord) -> "EnterpriseACLResponse":
        return cls(**record.to_dict())


class EnterpriseAuditEventResponse(BaseModel):
    id: str
    event_type: str
    actor_user_id: str | None
    target_type: str | None
    target_id: str | None
    metadata: dict[str, Any]
    created_at: str

    @classmethod
    def from_record(cls, record: AuditEventRecord) -> "EnterpriseAuditEventResponse":
        return cls(**record.to_dict())


class EnterpriseMeResponse(BaseModel):
    user: EnterpriseUserResponse
    principal: dict[str, Any]


def create_enterprise_routes(api_key: str | None = None) -> APIRouter:
    router = APIRouter(tags=["enterprise-auth"])
    combined_auth = get_combined_auth_dependency(api_key)

    def principal_payload(principal: Principal) -> dict[str, Any]:
        return {
            "user_id": principal.user_id,
            "username": principal.username,
            "system_role": principal.system_role,
            "status": principal.status,
            "tenant_id": principal.tenant_id,
            "can_create_kb": principal.can_create_kb,
            "can_use_bypass_query": principal.can_use_bypass_query,
            "token_version": principal.token_version,
            "auth_method": principal.auth_method,
        }

    def require_principal(request: Request) -> Principal:
        principal = get_request_principal(request)
        if principal is None:
            raise HTTPException(status_code=401, detail="Login required")
        return principal

    def login_response(user_service, user: EnterpriseUserRecord) -> dict[str, Any]:
        token = auth_handler.create_token(
            username=user.username,
            role=user.system_role,
            metadata=user_service.token_metadata_for_user(user),
        )
        return {
            "access_token": token,
            "token_type": "bearer",
            "auth_mode": "enterprise",
            "user": EnterpriseUserResponse.from_record(user).model_dump(),
            "core_version": core_version,
            "api_version": __api_version__,
        }

    @router.post("/auth/register")
    async def register_user(request: Request, body: EnterpriseRegistrationRequest):
        settings_service = get_enterprise_settings_service(request)
        if not await settings_service.registration_enabled():
            raise HTTPException(status_code=403, detail="User registration is disabled")
        user_service = get_enterprise_user_service(request)
        user = await user_service.create_user(
            username=body.username,
            password=body.password,
        )
        return login_response(user_service, user)

    @router.get(
        "/auth/me",
        response_model=EnterpriseMeResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_current_enterprise_user(request: Request):
        principal = require_principal(request)
        user_service = get_enterprise_user_service(request)
        user = await user_service.get_user_or_404(principal.user_id)
        return EnterpriseMeResponse(
            user=EnterpriseUserResponse.from_record(user),
            principal=principal_payload(principal),
        )

    @router.post(
        "/auth/change-password",
        response_model=EnterpriseUserResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def change_enterprise_password(
        request: Request, body: EnterpriseChangePasswordRequest
    ):
        principal = require_principal(request)
        user_service = get_enterprise_user_service(request)
        authenticated = await user_service.authenticate(
            principal.username, body.current_password
        )
        if authenticated is None:
            raise HTTPException(status_code=403, detail="Current password is incorrect")
        user = await user_service.change_password(
            principal.user_id,
            body.new_password,
            actor_user_id=principal.user_id,
        )
        return EnterpriseUserResponse.from_record(user)

    @router.get(
        "/admin/settings/registration",
        dependencies=[Depends(combined_auth)],
    )
    async def get_registration_setting(request: Request):
        settings_service = get_enterprise_settings_service(request)
        return {"enabled": await settings_service.registration_enabled()}

    @router.put(
        "/admin/settings/registration",
        dependencies=[Depends(combined_auth)],
    )
    @router.patch(
        "/admin/settings/registration",
        dependencies=[Depends(combined_auth)],
    )
    async def set_registration_setting(
        request: Request, body: EnterpriseRegistrationToggleRequest
    ):
        principal = require_principal(request)
        settings_service = get_enterprise_settings_service(request)
        await settings_service.set_registration_enabled(
            body.enabled, updated_by=principal.user_id
        )
        audit_service = get_enterprise_audit_service(request)
        await audit_service.append(
            "registration_setting_updated",
            actor_user_id=principal.user_id,
            target_type="system_setting",
            target_id="registration_enabled",
            metadata={"enabled": body.enabled},
        )
        return {"enabled": body.enabled}

    @router.get(
        "/admin/users",
        response_model=list[EnterpriseUserResponse],
        dependencies=[Depends(combined_auth)],
    )
    async def list_enterprise_users(request: Request):
        user_service = get_enterprise_user_service(request)
        return [
            EnterpriseUserResponse.from_record(user)
            for user in await user_service.list_users()
        ]

    @router.post(
        "/admin/users",
        response_model=EnterpriseUserResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def create_enterprise_user(
        request: Request, body: EnterpriseUserCreateRequest
    ):
        principal = require_principal(request)
        user_service = get_enterprise_user_service(request)
        user = await user_service.create_user(
            username=body.username,
            password=body.password,
            created_by=principal.user_id,
            can_create_kb=body.can_create_kb,
            can_use_bypass_query=body.can_use_bypass_query,
            tenant_id=body.tenant_id,
        )
        return EnterpriseUserResponse.from_record(user)

    @router.patch(
        "/admin/users/{user_id}",
        response_model=EnterpriseUserResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def update_enterprise_user(
        user_id: str, request: Request, body: EnterpriseUserUpdateRequest
    ):
        principal = require_principal(request)
        user_service = get_enterprise_user_service(request)
        user = await user_service.get_user_or_404(user_id)
        if any(
            value is not None
            for value in (
                body.status,
                body.can_create_kb,
                body.can_use_bypass_query,
                body.tenant_id,
            )
        ):
            user = await user_service.update_user(
                user_id,
                status_value=body.status,
                can_create_kb=body.can_create_kb,
                can_use_bypass_query=body.can_use_bypass_query,
                tenant_id=body.tenant_id,
                actor_user_id=principal.user_id,
            )
        if body.password is not None:
            user = await user_service.change_password(
                user_id, body.password, actor_user_id=principal.user_id
            )
        return EnterpriseUserResponse.from_record(user)

    @router.get(
        "/admin/kbs/{kb_id}/acl",
        response_model=list[EnterpriseACLResponse],
        dependencies=[Depends(combined_auth)],
    )
    async def list_kb_acl(kb_id: str, request: Request):
        authz_service = get_enterprise_authorization_service(request)
        return [
            EnterpriseACLResponse.from_record(item)
            for item in await authz_service.list_kb_acl(kb_id)
        ]

    @router.put(
        "/admin/kbs/{kb_id}/acl",
        response_model=EnterpriseACLResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def grant_kb_acl(
        kb_id: str, request: Request, body: EnterpriseACLGrantRequest
    ):
        principal = require_principal(request)
        authz_service = get_enterprise_authorization_service(request)
        record = await authz_service.grant_kb_role(
            kb_id,
            body.user_id,
            body.role,
            granted_by=principal.user_id,
        )
        return EnterpriseACLResponse.from_record(record)

    @router.delete(
        "/admin/kbs/{kb_id}/acl/{user_id}",
        dependencies=[Depends(combined_auth)],
    )
    async def revoke_kb_acl(kb_id: str, user_id: str, request: Request):
        principal = require_principal(request)
        authz_service = get_enterprise_authorization_service(request)
        deleted = await authz_service.revoke_kb_role(
            kb_id, user_id, actor_user_id=principal.user_id
        )
        return {"deleted": deleted}

    @router.get(
        "/admin/audit-events",
        response_model=list[EnterpriseAuditEventResponse],
        dependencies=[Depends(combined_auth)],
    )
    async def list_audit_events(request: Request, limit: int = 100):
        audit_service = get_enterprise_audit_service(request)
        return [
            EnterpriseAuditEventResponse.from_record(event)
            for event in await audit_service.list(limit=limit)
        ]

    return router
