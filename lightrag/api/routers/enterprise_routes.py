from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from lightrag.api.auth import auth_handler
from lightrag.api.enterprise_auth import (
    Principal,
    USER_STATUS_ACTIVE,
    USER_STATUS_DISABLED,
    get_enterprise_api_key_service,
    get_enterprise_audit_service,
    get_enterprise_authorization_service,
    get_enterprise_settings_service,
    get_enterprise_user_service,
    get_request_principal,
)
from lightrag.api.kb_service import KnowledgeBaseNotFoundError, KnowledgeBaseService
from lightrag.api.metadata_store import (
    AuditEventRecord,
    EnterpriseAPIKeyRecord,
    EnterpriseUserRecord,
    KBACLRecord,
    EnterpriseTenantKBACLRecord,
    EnterpriseTenantMembershipRecord,
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
    enabled: bool | None = None
    mode: str | None = None


class EnterpriseRegistrationSettingResponse(BaseModel):
    enabled: bool
    mode: str


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


class EnterpriseUserResetPasswordRequest(BaseModel):
    password: str = Field(min_length=1)


class EnterpriseACLGrantRequest(BaseModel):
    user_id: str | None = Field(default=None, min_length=1)
    tenant_id: str | None = Field(default=None, min_length=1)
    role: str = Field(min_length=1)


class EnterpriseACLBatchEntry(BaseModel):
    user_id: str | None = Field(default=None, min_length=1)
    tenant_id: str | None = Field(default=None, min_length=1)
    role: str | None = Field(default=None)
    action: Literal["grant", "revoke"] = "grant"


class EnterpriseACLBatchSetRequest(BaseModel):
    entries: list[EnterpriseACLBatchEntry] = Field(min_length=1)


class EnterpriseACLResponse(BaseModel):
    kb_id: str
    user_id: str | None = None
    tenant_id: str | None = None
    principal_type: Literal["user", "tenant"] = "user"
    role: str
    granted_by: str | None
    created_at: str
    updated_at: str

    @classmethod
    def from_record(cls, record: KBACLRecord) -> "EnterpriseACLResponse":
        return cls(**record.to_dict(), principal_type="user")

    @classmethod
    def from_tenant_record(
        cls, record: EnterpriseTenantKBACLRecord
    ) -> "EnterpriseACLResponse":
        return cls(**record.to_dict(), principal_type="tenant")


class EnterpriseACLBatchSetResponse(BaseModel):
    granted: list[EnterpriseACLResponse]
    revoked: list[str]


class EnterpriseTenantMembershipGrantRequest(BaseModel):
    role: str = Field(min_length=1)


class EnterpriseTenantMembershipResponse(BaseModel):
    tenant_id: str
    user_id: str
    role: str
    granted_by: str | None
    created_at: str
    updated_at: str

    @classmethod
    def from_record(
        cls, record: EnterpriseTenantMembershipRecord
    ) -> "EnterpriseTenantMembershipResponse":
        return cls(**record.to_dict())


class EnterpriseServiceAPIKeyCreateRequest(BaseModel):
    name: str = Field(min_length=1)
    kb_roles: dict[str, str] = Field(default_factory=dict)
    can_use_bypass_query: bool = False
    tenant_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EnterpriseServiceAPIKeyResponse(BaseModel):
    id: str
    name: str
    key_preview: str
    status: str
    created_by: str | None
    tenant_id: str | None
    scopes: dict[str, Any]
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    last_used_at: str | None
    revoked_at: str | None
    revoked_by: str | None

    @classmethod
    def from_record(cls, record: EnterpriseAPIKeyRecord) -> "EnterpriseServiceAPIKeyResponse":
        data = record.to_dict()
        data.pop("key_hash", None)
        return cls(**data)


class EnterpriseServiceAPIKeyCreateResponse(BaseModel):
    api_key: str
    key: EnterpriseServiceAPIKeyResponse


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


def create_enterprise_routes(
    api_key: str | None = None,
    kb_service: KnowledgeBaseService | None = None,
) -> APIRouter:
    router = APIRouter(tags=["enterprise-auth"])
    combined_auth = get_combined_auth_dependency(api_key)

    def principal_payload(principal: Principal) -> dict[str, Any]:
        return {
            "user_id": principal.user_id,
            "username": principal.username,
            "system_role": principal.system_role,
            "status": principal.status,
            "tenant_id": principal.tenant_id,
            "tenant_roles": principal.tenant_roles,
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

    async def require_kb_exists(request: Request, kb_id: str) -> None:
        service = kb_service or getattr(request.app.state, "kb_service", None)
        if not isinstance(service, KnowledgeBaseService):
            raise HTTPException(status_code=500, detail="Knowledge base service unavailable")
        try:
            await service.get(kb_id)
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Knowledge base not found") from exc

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
        response_model=EnterpriseRegistrationSettingResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_registration_setting(request: Request):
        settings_service = get_enterprise_settings_service(request)
        mode = await settings_service.registration_mode()
        return EnterpriseRegistrationSettingResponse(
            enabled=mode == "open",
            mode=mode,
        )

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
        if body.mode is not None:
            mode = await settings_service.set_registration_mode(
                body.mode,
                updated_by=principal.user_id,
            )
        elif body.enabled is not None:
            await settings_service.set_registration_enabled(
                body.enabled, updated_by=principal.user_id
            )
            mode = await settings_service.registration_mode()
        else:
            raise HTTPException(status_code=400, detail="Registration mode or enabled flag required")
        audit_service = get_enterprise_audit_service(request)
        await audit_service.append(
            "registration_setting_updated",
            actor_user_id=principal.user_id,
            target_type="system_setting",
            target_id="registration_mode",
            metadata={"enabled": mode == "open", "mode": mode},
        )
        return EnterpriseRegistrationSettingResponse(enabled=mode == "open", mode=mode)

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

    @router.get(
        "/admin/users/{user_id}",
        response_model=EnterpriseUserResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_enterprise_user(user_id: str, request: Request):
        user_service = get_enterprise_user_service(request)
        return EnterpriseUserResponse.from_record(
            await user_service.get_user_or_404(user_id)
        )

    @router.post(
        "/admin/users/{user_id}:disable",
        response_model=EnterpriseUserResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def disable_enterprise_user(user_id: str, request: Request):
        principal = require_principal(request)
        user_service = get_enterprise_user_service(request)
        user = await user_service.update_user(
            user_id,
            status_value=USER_STATUS_DISABLED,
            actor_user_id=principal.user_id,
        )
        return EnterpriseUserResponse.from_record(user)

    @router.post(
        "/admin/users/{user_id}:enable",
        response_model=EnterpriseUserResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def enable_enterprise_user(user_id: str, request: Request):
        principal = require_principal(request)
        user_service = get_enterprise_user_service(request)
        user = await user_service.update_user(
            user_id,
            status_value=USER_STATUS_ACTIVE,
            actor_user_id=principal.user_id,
        )
        return EnterpriseUserResponse.from_record(user)

    @router.post(
        "/admin/users/{user_id}:reset-password",
        response_model=EnterpriseUserResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def reset_enterprise_user_password(
        user_id: str, request: Request, body: EnterpriseUserResetPasswordRequest
    ):
        principal = require_principal(request)
        user_service = get_enterprise_user_service(request)
        user = await user_service.change_password(
            user_id,
            body.password,
            actor_user_id=principal.user_id,
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
        await require_kb_exists(request, kb_id)
        authz_service = get_enterprise_authorization_service(request)
        user_acl = [
            EnterpriseACLResponse.from_record(item)
            for item in await authz_service.list_kb_acl(kb_id)
        ]
        tenant_acl = [
            EnterpriseACLResponse.from_tenant_record(item)
            for item in await authz_service.list_kb_tenant_acl(kb_id)
        ]
        return [*user_acl, *tenant_acl]

    @router.get(
        "/admin/tenants/{tenant_id}/members",
        response_model=list[EnterpriseTenantMembershipResponse],
        dependencies=[Depends(combined_auth)],
    )
    async def list_tenant_members(tenant_id: str, request: Request):
        authz_service = get_enterprise_authorization_service(request)
        return [
            EnterpriseTenantMembershipResponse.from_record(item)
            for item in await authz_service.list_tenant_memberships(tenant_id)
        ]

    @router.put(
        "/admin/tenants/{tenant_id}/members/{user_id}",
        response_model=EnterpriseTenantMembershipResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def grant_tenant_membership(
        tenant_id: str,
        user_id: str,
        request: Request,
        body: EnterpriseTenantMembershipGrantRequest,
    ):
        principal = require_principal(request)
        authz_service = get_enterprise_authorization_service(request)
        record = await authz_service.grant_tenant_membership(
            tenant_id,
            user_id,
            body.role,
            granted_by=principal.user_id,
        )
        return EnterpriseTenantMembershipResponse.from_record(record)

    @router.delete(
        "/admin/tenants/{tenant_id}/members/{user_id}",
        dependencies=[Depends(combined_auth)],
    )
    async def revoke_tenant_membership(tenant_id: str, user_id: str, request: Request):
        principal = require_principal(request)
        authz_service = get_enterprise_authorization_service(request)
        deleted = await authz_service.revoke_tenant_membership(
            tenant_id,
            user_id,
            actor_user_id=principal.user_id,
        )
        return {"deleted": deleted}

    @router.put(
        "/admin/kbs/{kb_id}/acl",
        response_model=EnterpriseACLResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def grant_kb_acl(
        kb_id: str, request: Request, body: EnterpriseACLGrantRequest
    ):
        principal = require_principal(request)
        await require_kb_exists(request, kb_id)
        authz_service = get_enterprise_authorization_service(request)
        if (body.user_id is None) == (body.tenant_id is None):
            raise HTTPException(status_code=400, detail="Exactly one ACL principal is required")
        if body.tenant_id is not None:
            record = await authz_service.grant_tenant_kb_role(
                kb_id,
                body.tenant_id,
                body.role,
                granted_by=principal.user_id,
            )
            return EnterpriseACLResponse.from_tenant_record(record)
        assert body.user_id is not None
        record = await authz_service.grant_kb_role(
            kb_id,
            body.user_id,
            body.role,
            granted_by=principal.user_id,
        )
        return EnterpriseACLResponse.from_record(record)

    @router.post(
        "/admin/kbs/{kb_id}/acl:batch-set",
        response_model=EnterpriseACLBatchSetResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def batch_set_kb_acl(
        kb_id: str, request: Request, body: EnterpriseACLBatchSetRequest
    ):
        principal = require_principal(request)
        await require_kb_exists(request, kb_id)
        authz_service = get_enterprise_authorization_service(request)
        granted: list[EnterpriseACLResponse] = []
        revoked: list[str] = []
        for entry in body.entries:
            if (entry.user_id is None) == (entry.tenant_id is None):
                raise HTTPException(status_code=400, detail="Exactly one ACL principal is required")
            if entry.action == "revoke":
                if entry.tenant_id is not None:
                    if await authz_service.revoke_tenant_kb_role(
                        kb_id, entry.tenant_id, actor_user_id=principal.user_id
                    ):
                        revoked.append(entry.tenant_id)
                    continue
                assert entry.user_id is not None
                if await authz_service.revoke_kb_role(
                    kb_id, entry.user_id, actor_user_id=principal.user_id
                ):
                    revoked.append(entry.user_id)
                continue
            if entry.role is None:
                raise HTTPException(status_code=400, detail="Role is required for ACL grants")
            if entry.tenant_id is not None:
                record = await authz_service.grant_tenant_kb_role(
                    kb_id,
                    entry.tenant_id,
                    entry.role,
                    granted_by=principal.user_id,
                )
                granted.append(EnterpriseACLResponse.from_tenant_record(record))
                continue
            assert entry.user_id is not None
            record = await authz_service.grant_kb_role(
                kb_id,
                entry.user_id,
                entry.role,
                granted_by=principal.user_id,
            )
            granted.append(EnterpriseACLResponse.from_record(record))
        return EnterpriseACLBatchSetResponse(granted=granted, revoked=revoked)

    @router.delete(
        "/admin/kbs/{kb_id}/acl/{user_id}",
        dependencies=[Depends(combined_auth)],
    )
    async def revoke_kb_acl(kb_id: str, user_id: str, request: Request):
        principal = require_principal(request)
        await require_kb_exists(request, kb_id)
        authz_service = get_enterprise_authorization_service(request)
        deleted = await authz_service.revoke_kb_role(
            kb_id, user_id, actor_user_id=principal.user_id
        )
        return {"deleted": deleted}

    @router.delete(
        "/admin/kbs/{kb_id}/acl/tenants/{tenant_id}",
        dependencies=[Depends(combined_auth)],
    )
    async def revoke_tenant_kb_acl(kb_id: str, tenant_id: str, request: Request):
        principal = require_principal(request)
        await require_kb_exists(request, kb_id)
        authz_service = get_enterprise_authorization_service(request)
        deleted = await authz_service.revoke_tenant_kb_role(
            kb_id, tenant_id, actor_user_id=principal.user_id
        )
        return {"deleted": deleted}

    @router.get(
        "/admin/service-api-keys",
        response_model=list[EnterpriseServiceAPIKeyResponse],
        dependencies=[Depends(combined_auth)],
    )
    async def list_service_api_keys(request: Request):
        api_key_service = get_enterprise_api_key_service(request)
        return [
            EnterpriseServiceAPIKeyResponse.from_record(record)
            for record in await api_key_service.list_keys()
        ]

    @router.post(
        "/admin/service-api-keys",
        response_model=EnterpriseServiceAPIKeyCreateResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def create_service_api_key(
        request: Request, body: EnterpriseServiceAPIKeyCreateRequest
    ):
        principal = require_principal(request)
        for kb_id in body.kb_roles:
            await require_kb_exists(request, kb_id)
        api_key_service = get_enterprise_api_key_service(request)
        record, raw_key = await api_key_service.create_key(
            name=body.name,
            scopes={
                "kb_roles": body.kb_roles,
                "can_use_bypass_query": body.can_use_bypass_query,
            },
            metadata=body.metadata,
            created_by=principal.user_id,
            tenant_id=body.tenant_id,
        )
        return EnterpriseServiceAPIKeyCreateResponse(
            api_key=raw_key,
            key=EnterpriseServiceAPIKeyResponse.from_record(record),
        )

    @router.post(
        "/admin/service-api-keys/{key_id}:revoke",
        response_model=EnterpriseServiceAPIKeyResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def revoke_service_api_key(key_id: str, request: Request):
        principal = require_principal(request)
        api_key_service = get_enterprise_api_key_service(request)
        revoked = await api_key_service.revoke_key(key_id, revoked_by=principal.user_id)
        return EnterpriseServiceAPIKeyResponse.from_record(revoked)

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
