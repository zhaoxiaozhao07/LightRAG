from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from lightrag.api.auth import auth_handler
from lightrag.api.enterprise_auth import (
    Principal,
    INTERACTIVE_AUTH_METHODS,
    KB_ROLE_ADMIN,
    KB_ROLE_EDITOR,
    KB_ROLE_OWNER,
    KB_ROLE_VIEWER,
    REGISTRATION_MODE_ADMIN_APPROVAL,
    REGISTRATION_MODE_INVITE_ONLY,
    REGISTRATION_MODE_OPEN,
    SERVICE_API_KEY_AUTH_METHOD,
    SYSTEM_ROLE_SUPER_ADMIN,
    SYSTEM_ROLE_USER,
    TENANT_ROLE_ADMIN,
    TENANT_ROLE_MEMBER,
    UNSET,
    USER_STATUS_ACTIVE,
    USER_STATUS_DISABLED,
    USER_STATUS_PENDING,
    agent_workflow_prompt_max_length,
    chat_memory_write_conflict,
    get_enterprise_api_key_service,
    get_enterprise_audit_service,
    get_enterprise_authorization_service,
    get_enterprise_chat_memory_service,
    get_enterprise_invitation_service,
    get_enterprise_settings_service,
    get_enterprise_user_agent_workflow_prompt_service,
    get_enterprise_user_kb_query_settings_service,
    get_enterprise_user_service,
    get_request_principal,
)
from lightrag.api.chat_memory_service import (
    ChatMemoryEventNotFoundError,
    ChatMemoryRetryConflictError,
    ChatMemoryUnavailableError,
)
from lightrag.api.kb_service import (
    is_tenant_owned_kb,
    KnowledgeBaseNotFoundError,
    KnowledgeBaseService,
)
from lightrag.api.metadata_store import (
    AuditEventRecord,
    EnterpriseAPIKeyRecord,
    EnterpriseInvitationRecord,
    EnterpriseUserRecord,
    KBACLRecord,
    EnterpriseTenantKBACLRecord,
    EnterpriseTenantMembershipRecord,
    EnterpriseTenantRecord,
    MetadataConflictError,
)
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag import __version__ as core_version
from lightrag.api import __api_version__
from lightrag.utils import logger


class EnterpriseUserResponse(BaseModel):
    id: str
    username: str
    system_role: str
    status: str
    tenant_id: str | None
    can_create_kb: bool
    can_use_bypass_query: bool
    can_use_agent_query: bool
    can_delete_documents: bool
    can_download_files: bool
    token_version: int
    created_at: str
    updated_at: str
    display_name: str | None = None
    email: str | None = None

    @classmethod
    def from_record(cls, record: EnterpriseUserRecord) -> "EnterpriseUserResponse":
        data = record.to_dict()
        data.pop("password_hash", None)
        metadata = data.pop("metadata", None) or {}
        # Self-service profile fields live in user metadata; surface them as
        # read-only response fields without exposing the raw metadata dict.
        data.setdefault("display_name", metadata.get("display_name"))
        data.setdefault("email", metadata.get("email"))
        return cls(**data)


class EnterpriseRegistrationRequest(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)
    invitation_token: str | None = None


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
    can_use_agent_query: bool = False
    can_delete_documents: bool = False
    can_download_files: bool = False
    tenant_id: str | None = None


class EnterpriseUserUpdateRequest(BaseModel):
    status: str | None = None
    can_create_kb: bool | None = None
    can_use_bypass_query: bool | None = None
    can_use_agent_query: bool | None = None
    can_delete_documents: bool | None = None
    can_download_files: bool | None = None
    tenant_id: str | None = None
    password: str | None = None


class EnterpriseTenantUserCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=1)
    password: str = Field(min_length=1)
    can_create_kb: bool = False
    can_use_bypass_query: bool = False
    can_use_agent_query: bool = False
    can_delete_documents: bool = False
    can_download_files: bool = False
    tenant_id: str | None = None


class EnterpriseTenantUserUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str | None = None
    can_create_kb: bool | None = None
    can_use_bypass_query: bool | None = None
    can_use_agent_query: bool | None = None
    can_delete_documents: bool | None = None
    can_download_files: bool | None = None
    # Declared only so attempts to use the super-admin migration/password
    # surface receive a deliberate 400 instead of being silently ignored.
    tenant_id: str | None = None
    password: str | None = None


class EnterpriseUserResetPasswordRequest(BaseModel):
    password: str = Field(min_length=1)


class EnterpriseProfileUpdateRequest(BaseModel):
    display_name: str | None = None
    email: str | None = None


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


class EnterpriseUserKBAccessBatchEntry(BaseModel):
    kb_id: str = Field(min_length=1)
    role: str | None = Field(default=None)
    action: Literal["grant", "revoke"] = "grant"


class EnterpriseUserKBAccessBatchSetRequest(BaseModel):
    entries: list[EnterpriseUserKBAccessBatchEntry] = Field(min_length=1)


class EnterpriseUserKBAccessBatchSetResponse(BaseModel):
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
    # Resolved from the user record so tenant admins (who cannot call the
    # super-admin-only /admin/users endpoints) can see who each member is.
    username: str | None = None
    display_name: str | None = None
    user_status: str | None = None

    @classmethod
    def from_record(
        cls,
        record: EnterpriseTenantMembershipRecord,
        *,
        user: EnterpriseUserRecord | None = None,
    ) -> "EnterpriseTenantMembershipResponse":
        data = record.to_dict()
        if user is not None:
            data["username"] = user.username
            data["display_name"] = (user.metadata or {}).get("display_name")
            data["user_status"] = user.status
        return cls(**data)


class EnterpriseTenantCreateRequest(BaseModel):
    name: str = Field(min_length=1)
    description: str | None = None
    tenant_id: str | None = None


class EnterpriseTenantUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    status: str | None = None


class EnterpriseTenantResponse(BaseModel):
    id: str
    name: str
    description: str | None
    status: str
    created_by: str | None
    created_at: str
    updated_at: str

    @classmethod
    def from_record(cls, record: EnterpriseTenantRecord) -> "EnterpriseTenantResponse":
        data = record.to_dict()
        data.pop("metadata", None)
        return cls(**data)


class EnterpriseTenantDetailResponse(BaseModel):
    id: str
    name: str
    description: str | None
    status: str
    created_by: str | None
    created_at: str
    updated_at: str
    member_count: int
    kb_count: int


class EnterpriseTenantKBSummaryResponse(BaseModel):
    id: str
    name: str
    status: str
    visibility: str | None = None
    owner_id: str | None = None


class EnterpriseUserAccessResponse(BaseModel):
    user_id: str
    username: str
    system_role: str
    tenant_id: str | None
    can_create_kb: bool
    can_use_bypass_query: bool
    can_use_agent_query: bool
    can_delete_documents: bool
    can_download_files: bool
    tenant_memberships: list[dict[str, str]]
    kb_acls: list[dict[str, str]]


class EnterpriseTenantKBMemberRoleRequest(BaseModel):
    role: Literal["viewer", "editor", "admin"]


class EnterpriseTenantKBMemberAccessResponse(BaseModel):
    tenant_id: str
    kb_id: str
    user_id: str
    username: str
    status: str
    override_effect: str | None
    override_role: str | None
    effective_role: str | None
    sources: list[str]
    platform_role: str | None
    tenant_acl_role: str | None


_TENANT_KB_MEMBER_ROLES = {
    "viewer": KB_ROLE_VIEWER,
    "editor": KB_ROLE_EDITOR,
    "admin": KB_ROLE_ADMIN,
}
_KB_ROLE_RANK = {
    KB_ROLE_VIEWER: 1,
    KB_ROLE_EDITOR: 2,
    KB_ROLE_ADMIN: 3,
    KB_ROLE_OWNER: 4,
}


class EnterpriseServiceAPIKeyCreateRequest(BaseModel):
    name: str = Field(min_length=1)
    kb_roles: dict[str, str] = Field(default_factory=dict)
    can_use_bypass_query: bool = False
    can_use_agent_query: bool = False
    inherit_tenant_kb_acl: bool = False
    tenant_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    expires_in_seconds: int | None = Field(default=None, ge=1)


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
    expires_at: str | None = None

    @classmethod
    def from_record(cls, record: EnterpriseAPIKeyRecord) -> "EnterpriseServiceAPIKeyResponse":
        data = record.to_dict()
        data.pop("key_hash", None)
        return cls(**data)


class EnterpriseServiceAPIKeyCreateResponse(BaseModel):
    api_key: str
    key: EnterpriseServiceAPIKeyResponse


class EnterpriseServiceAPIKeyRotateRequest(BaseModel):
    expires_in_seconds: int | None = Field(default=None, ge=1)
    revoke_old: bool = True


class EnterpriseInvitationCreateRequest(BaseModel):
    expires_in_seconds: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EnterpriseInvitationResponse(BaseModel):
    id: str
    token_preview: str
    status: str
    created_by: str | None
    expires_at: str | None
    used_by: str | None
    used_at: str | None
    metadata: dict[str, Any]
    created_at: str
    updated_at: str

    @classmethod
    def from_record(
        cls, record: EnterpriseInvitationRecord
    ) -> "EnterpriseInvitationResponse":
        data = record.to_dict()
        data.pop("token_hash", None)
        return cls(**data)


class EnterpriseInvitationCreateResponse(BaseModel):
    invitation_token: str
    invitation: EnterpriseInvitationResponse


class EnterpriseAuditEventResponse(BaseModel):
    id: str
    event_type: str
    actor_user_id: str | None
    actor_tenant_id: str | None
    actor_username: str | None = None
    target_type: str | None
    target_id: str | None
    target_name: str | None = None
    metadata: dict[str, Any]
    created_at: str

    @classmethod
    def from_record(
        cls,
        record: AuditEventRecord,
        *,
        actor_username: str | None = None,
        target_name: str | None = None,
    ) -> "EnterpriseAuditEventResponse":
        data = record.to_dict()
        data["actor_username"] = actor_username
        data["target_name"] = target_name
        return cls(**data)


class EnterpriseMeResponse(BaseModel):
    user: EnterpriseUserResponse | None
    principal: dict[str, Any]


class EnterpriseUserKBQuerySettingsRequest(BaseModel):
    user_prompt: str = Field(default="", max_length=16384)


class EnterpriseUserKBQuerySettingsResponse(BaseModel):
    user_id: str
    kb_id: str
    user_prompt: str


class EnterpriseUserAgentWorkflowPromptRequest(BaseModel):
    workflow_prompt: str = Field(default="", max_length=16384)


class EnterpriseUserAgentWorkflowPromptResponse(BaseModel):
    user_id: str
    workflow_prompt: str


class AdminChatMemoryPurgeRequest(BaseModel):
    # Omitted / empty => purge every project of the target user.
    project_ids: list[str] | None = Field(default=None)


class AdminChatMemoryBacklogScanRequest(BaseModel):
    limit: int | None = Field(default=None, ge=1, le=1000)


class AdminChatMemoryEventResponse(BaseModel):
    event_id: str
    status: str
    user_id: str
    project_id: str
    event_type: str


def create_enterprise_routes(
    api_key: str | None = None,
    kb_service: KnowledgeBaseService | None = None,
) -> APIRouter:
    router = APIRouter(tags=["enterprise-auth"])
    combined_auth = get_combined_auth_dependency(api_key)

    def chat_memory_maintenance_service(request: Request):
        return getattr(
            request.app.state,
            "enterprise_chat_memory_maintenance_service",
            None,
        ) or get_enterprise_chat_memory_service(request)

    def chat_memory_worker(request: Request):
        return getattr(request.app.state, "enterprise_chat_memory_worker", None)

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
            "can_use_agent_query": principal.can_use_agent_query,
            "can_delete_documents": principal.can_delete_documents,
            "can_download_files": principal.can_download_files,
            "token_version": principal.token_version,
            "auth_method": principal.auth_method,
        }

    def require_principal(request: Request) -> Principal:
        principal = get_request_principal(request)
        if principal is None:
            raise HTTPException(status_code=401, detail="Login required")
        return principal

    def require_interactive_user_principal(request: Request) -> Principal:
        principal = require_principal(request)
        if principal.auth_method not in INTERACTIVE_AUTH_METHODS:
            raise HTTPException(
                status_code=403,
                detail="Only available for interactive users",
            )
        return principal

    def request_kb_service(request: Request) -> KnowledgeBaseService:
        service = kb_service or getattr(request.app.state, "kb_service", None)
        if not isinstance(service, KnowledgeBaseService):
            raise HTTPException(status_code=500, detail="Knowledge base service unavailable")
        return service

    async def require_kb_exists(request: Request, kb_id: str) -> None:
        service = request_kb_service(request)
        try:
            await service.get(kb_id)
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Knowledge base not found") from exc

    async def require_tenant_admin(request: Request, tenant_id: str) -> Principal:
        principal = require_principal(request)
        authz_service = get_enterprise_authorization_service(request)
        principal = await authz_service.require_tenant_role(
            principal,
            tenant_id,
            TENANT_ROLE_ADMIN,
        )
        return principal

    async def memberships_with_user_info(
        request: Request,
        records: list[EnterpriseTenantMembershipRecord],
    ) -> list[EnterpriseTenantMembershipResponse]:
        """Attach username/display_name to membership records.

        Batch-resolves the page's user ids in one ``list_users`` pass; a
        membership whose user record has vanished keeps null user fields.
        """
        user_ids = {record.user_id for record in records}
        users: dict[str, EnterpriseUserRecord] = {}
        if user_ids:
            user_service = get_enterprise_user_service(request)
            users = {
                user.id: user
                for user in await user_service.list_users()
                if user.id in user_ids
            }
        return [
            EnterpriseTenantMembershipResponse.from_record(
                record, user=users.get(record.user_id)
            )
            for record in records
        ]

    async def active_tenant_kbs(request: Request, tenant_id: str) -> list[Any]:
        """Return the active catalog union of owned and tenant-ACL KBs."""
        authz_service = get_enterprise_authorization_service(request)
        assigned_ids = set(
            await authz_service.list_kb_ids_for_tenants([tenant_id])
        )
        records = await request_kb_service(request).list(include_deleted=False)
        return [
            record
            for record in records
            if record.status != "deleted"
            and (record.tenant_id == tenant_id or record.id in assigned_ids)
        ]

    async def tenant_user_or_404(
        request: Request,
        tenant_id: str,
        user_id: str,
    ) -> tuple[EnterpriseUserRecord, EnterpriseTenantMembershipRecord]:
        """Resolve only canonical, non-super users in the path tenant.

        Missing and cross-tenant targets deliberately share the same response
        so tenant administrators cannot use this surface for account discovery.
        """
        user_service = get_enterprise_user_service(request)
        user = await user_service.get_user_or_404(user_id)
        membership = await get_enterprise_authorization_service(
            request
        ).get_tenant_membership(tenant_id, user_id)
        if (
            user.system_role == SYSTEM_ROLE_SUPER_ADMIN
            or user.tenant_id != tenant_id
            or membership is None
        ):
            raise HTTPException(status_code=404, detail="User not found")
        return user, membership

    async def mutable_tenant_member_or_404(
        request: Request,
        tenant_id: str,
        user_id: str,
        *,
        actor_user_id: str,
    ) -> tuple[EnterpriseUserRecord, EnterpriseTenantMembershipRecord]:
        user, membership = await tenant_user_or_404(request, tenant_id, user_id)
        if actor_user_id == user.id:
            raise HTTPException(
                status_code=409,
                detail="Tenant administrators cannot mutate themselves",
            )
        if membership.role != TENANT_ROLE_MEMBER:
            raise HTTPException(status_code=404, detail="User not found")
        return user, membership

    async def unassigned_user_or_404(
        request: Request,
        user_id: str,
        *,
        actor_user_id: str,
    ) -> EnterpriseUserRecord:
        user = await get_enterprise_user_service(request).get_user_or_404(user_id)
        if actor_user_id == user.id:
            raise HTTPException(
                status_code=409,
                detail="Tenant administrators cannot mutate themselves",
            )
        memberships = await get_enterprise_authorization_service(
            request
        ).list_user_tenant_memberships(user_id)
        if (
            user.system_role != SYSTEM_ROLE_USER
            or user.tenant_id is not None
            or memberships
        ):
            raise HTTPException(status_code=404, detail="User not found")
        return user

    async def delete_user_with_chat_memory(
        request: Request,
        user_id: str,
        *,
        actor_user_id: str,
        actor_tenant_id: Any = UNSET,
        expected_user: EnterpriseUserRecord | None = None,
        expected_membership: Any = UNSET,
    ) -> bool:
        """Delete through the memory-aware UserService transaction when wired."""
        user_service = get_enterprise_user_service(request)
        delete_kwargs: dict[str, Any] = {
            "actor_user_id": actor_user_id,
            "expected_user": expected_user,
            "expected_membership": expected_membership,
        }
        if actor_tenant_id is not UNSET:
            delete_kwargs["actor_tenant_id"] = actor_tenant_id
        return await user_service.delete_user(user_id, **delete_kwargs)

    async def audit_event_responses(
        request: Request,
        events: list[AuditEventRecord],
    ) -> list[EnterpriseAuditEventResponse]:
        """Attach the same actor/target enrichment to admin and tenant views."""
        user_ids = {event.actor_user_id for event in events if event.actor_user_id}
        user_ids.update(
            event.target_id
            for event in events
            if event.target_type == "user" and event.target_id
        )
        kb_ids = {
            event.target_id
            for event in events
            if event.target_type == "kb" and event.target_id
        }
        user_names: dict[str, str] = {}
        kb_names: dict[str, str] = {}
        if user_ids:
            for user in await get_enterprise_user_service(request).list_users():
                if user.id in user_ids:
                    user_names[user.id] = user.username
        if kb_ids:
            for record in await request_kb_service(request).list(
                include_deleted=True
            ):
                if record.id in kb_ids:
                    kb_names[record.id] = record.name

        return [
            EnterpriseAuditEventResponse.from_record(
                event,
                actor_username=user_names.get(event.actor_user_id)
                if event.actor_user_id
                else None,
                target_name=(
                    user_names.get(event.target_id)
                    if event.target_type == "user" and event.target_id
                    else kb_names.get(event.target_id)
                    if event.target_type == "kb" and event.target_id
                    else None
                ),
            )
            for event in events
        ]

    async def manageable_tenant_kb(
        request: Request,
        tenant_id: str,
        kb_id: str,
    ) -> tuple[Any, str | None, bool]:
        service = request_kb_service(request)
        try:
            record = await service.get(kb_id)
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail="Knowledge base not found"
            ) from exc
        if record.status != "active":
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        tenant_owned = is_tenant_owned_kb(record, tenant_id)
        authz_service = get_enterprise_authorization_service(request)
        tenant_acl_role = next(
            (
                acl.role
                for acl in await authz_service.list_kb_tenant_acl(kb_id)
                if acl.tenant_id == tenant_id
            ),
            None,
        )
        if not tenant_owned and tenant_acl_role is None:
            raise HTTPException(
                status_code=403,
                detail="Knowledge base is not managed by this tenant",
            )
        return record, tenant_acl_role, tenant_owned

    async def tenant_kb_member_access_response(
        request: Request,
        tenant_id: str,
        record: Any,
        user: EnterpriseUserRecord,
        membership: EnterpriseTenantMembershipRecord,
    ) -> EnterpriseTenantKBMemberAccessResponse:
        # Resolve authorization sources independently of account status; the
        # status is returned separately and disabled accounts cannot authenticate.
        target_principal = Principal(
            user_id=user.id,
            username=user.username,
            system_role=user.system_role,
            status=USER_STATUS_ACTIVE,
            tenant_id=tenant_id,
            tenant_roles={tenant_id: membership.role},
            can_create_kb=user.can_create_kb,
            can_use_bypass_query=user.can_use_bypass_query,
            can_use_agent_query=user.can_use_agent_query,
            can_delete_documents=user.can_delete_documents,
            can_download_files=user.can_download_files,
            token_version=user.token_version,
            auth_method="jwt",
            metadata=dict(user.metadata),
        )
        decision = await get_enterprise_authorization_service(
            request
        ).resolve_kb_access(target_principal, record)
        return EnterpriseTenantKBMemberAccessResponse(
            tenant_id=tenant_id,
            kb_id=record.id,
            user_id=user.id,
            username=user.username,
            status=user.status,
            override_effect=decision.tenant_override_effect,
            override_role=decision.tenant_override_role,
            effective_role=decision.effective_role,
            sources=list(decision.sources),
            platform_role=decision.platform_role,
            tenant_acl_role=decision.tenant_acl_role,
        )

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
        mode = await settings_service.registration_mode()
        user_service = get_enterprise_user_service(request)
        audit_service = get_enterprise_audit_service(request)
        tracker = getattr(request.app.state, "enterprise_registration_tracker", None)
        username = body.username.strip()
        if tracker is not None:
            tracker.check(username)

        async def _record_registration_failure(error: str) -> None:
            locked = tracker.record_failure(username) if tracker is not None else False
            await audit_service.append(
                "registration_locked" if locked else "registration_failed",
                target_type="user",
                target_id=username,
                metadata={"mode": mode, "error": error},
            )

        async def _record_registration_success() -> None:
            if tracker is not None:
                tracker.record_success(username)

        if mode == REGISTRATION_MODE_OPEN:
            try:
                user = await user_service.create_user(
                    username=body.username, password=body.password
                )
            except HTTPException as exc:
                await _record_registration_failure(str(exc.detail))
                raise
            await _record_registration_success()
            await audit_service.append(
                "user_registered",
                actor_user_id=user.id,
                target_type="user",
                target_id=user.id,
                metadata={"mode": mode},
            )
            return login_response(user_service, user)

        if mode == REGISTRATION_MODE_INVITE_ONLY:
            invitation_service = get_enterprise_invitation_service(request)
            # Consume the single-use token first so a failed downstream step
            # cannot mint a user without burning the invitation.
            try:
                await invitation_service.consume_invitation(
                    body.invitation_token, used_by=body.username.strip()
                )
                user = await user_service.create_user(
                    username=body.username, password=body.password
                )
            except HTTPException as exc:
                await _record_registration_failure(str(exc.detail))
                raise
            await _record_registration_success()
            await audit_service.append(
                "user_registered",
                actor_user_id=user.id,
                target_type="user",
                target_id=user.id,
                metadata={"mode": mode},
            )
            return login_response(user_service, user)

        if mode == REGISTRATION_MODE_ADMIN_APPROVAL:
            try:
                user = await user_service.create_user(
                    username=body.username,
                    password=body.password,
                    status=USER_STATUS_PENDING,
                )
            except HTTPException as exc:
                await _record_registration_failure(str(exc.detail))
                raise
            await _record_registration_success()
            await audit_service.append(
                "user_registration_pending",
                actor_user_id=user.id,
                target_type="user",
                target_id=user.id,
                metadata={"mode": mode},
            )
            return {
                "auth_mode": "enterprise",
                "status": USER_STATUS_PENDING,
                "user": EnterpriseUserResponse.from_record(user).model_dump(),
                "message": "Registration submitted; awaiting administrator approval.",
            }

        await _record_registration_failure("registration_disabled")
        raise HTTPException(status_code=403, detail="User registration is disabled")

    @router.get(
        "/auth/me",
        response_model=EnterpriseMeResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_current_enterprise_user(request: Request):
        principal = require_principal(request)
        if principal.auth_method == SERVICE_API_KEY_AUTH_METHOD:
            return EnterpriseMeResponse(user=None, principal=principal_payload(principal))
        user_service = get_enterprise_user_service(request)
        user = await user_service.get_user_or_404(principal.user_id)
        return EnterpriseMeResponse(
            user=EnterpriseUserResponse.from_record(user),
            principal=principal_payload(principal),
        )

    @router.patch(
        "/auth/me",
        response_model=EnterpriseMeResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def update_current_enterprise_user(
        request: Request, body: EnterpriseProfileUpdateRequest
    ):
        principal = require_interactive_user_principal(request)
        user_service = get_enterprise_user_service(request)
        # Omitted fields stay unchanged; explicit null clears (KB-PATCH style).
        user = await user_service.update_own_profile(
            principal.user_id,
            display_name=body.display_name
            if "display_name" in body.model_fields_set
            else UNSET,
            email=body.email if "email" in body.model_fields_set else UNSET,
            actor_tenant_id=principal.tenant_id,
        )
        return EnterpriseMeResponse(
            user=EnterpriseUserResponse.from_record(user),
            principal=principal_payload(principal),
        )

    @router.post(
        "/auth/logout",
        dependencies=[Depends(combined_auth)],
    )
    async def logout_current_user(request: Request):
        """Log out everywhere: bump the caller's token_version so every
        outstanding JWT (this one included) stops validating."""
        principal = require_interactive_user_principal(request)
        user_service = get_enterprise_user_service(request)
        user = await user_service.logout_all_sessions(
            principal.user_id,
            actor_tenant_id=principal.tenant_id,
        )
        return {"status": "logged_out", "token_version": user.token_version}

    @router.get(
        "/auth/me/kbs/{kb_id}/query-settings",
        response_model=EnterpriseUserKBQuerySettingsResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_my_kb_query_settings(kb_id: str, request: Request):
        principal = require_interactive_user_principal(request)
        await require_kb_exists(request, kb_id)
        await get_enterprise_authorization_service(request).require_kb_role(
            principal, kb_id, KB_ROLE_VIEWER
        )
        settings = await get_enterprise_user_kb_query_settings_service(
            request
        ).get_settings(principal.user_id, kb_id)
        return EnterpriseUserKBQuerySettingsResponse(
            user_id=principal.user_id,
            kb_id=kb_id,
            user_prompt=settings.user_prompt if settings is not None else "",
        )

    @router.put(
        "/auth/me/kbs/{kb_id}/query-settings",
        response_model=EnterpriseUserKBQuerySettingsResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def set_my_kb_query_settings(
        kb_id: str, request: Request, body: EnterpriseUserKBQuerySettingsRequest
    ):
        principal = require_interactive_user_principal(request)
        await require_kb_exists(request, kb_id)
        await get_enterprise_authorization_service(request).require_kb_role(
            principal, kb_id, KB_ROLE_VIEWER
        )
        settings_service = get_enterprise_user_kb_query_settings_service(request)
        if body.user_prompt == "":
            await settings_service.clear_user_prompt(
                user_id=principal.user_id,
                kb_id=kb_id,
                actor_user_id=principal.user_id,
            )
            return EnterpriseUserKBQuerySettingsResponse(
                user_id=principal.user_id,
                kb_id=kb_id,
                user_prompt="",
            )
        settings = await settings_service.set_user_prompt(
            user_id=principal.user_id,
            kb_id=kb_id,
            user_prompt=body.user_prompt,
            actor_user_id=principal.user_id,
        )
        return EnterpriseUserKBQuerySettingsResponse(
            user_id=settings.user_id,
            kb_id=settings.kb_id,
            user_prompt=settings.user_prompt,
        )

    @router.get(
        "/auth/me/agent-workflow-prompt",
        response_model=EnterpriseUserAgentWorkflowPromptResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_my_agent_workflow_prompt(request: Request):
        principal = require_interactive_user_principal(request)
        prompt = await get_enterprise_user_agent_workflow_prompt_service(
            request
        ).get_prompt(principal.user_id)
        return EnterpriseUserAgentWorkflowPromptResponse(
            user_id=principal.user_id,
            workflow_prompt=prompt,
        )

    @router.put(
        "/auth/me/agent-workflow-prompt",
        response_model=EnterpriseUserAgentWorkflowPromptResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def set_my_agent_workflow_prompt(
        request: Request, body: EnterpriseUserAgentWorkflowPromptRequest
    ):
        principal = require_interactive_user_principal(request)
        max_length = agent_workflow_prompt_max_length()
        if len(body.workflow_prompt) > max_length:
            raise HTTPException(
                status_code=400,
                detail=f"workflow_prompt exceeds maximum length {max_length}",
            )
        service = get_enterprise_user_agent_workflow_prompt_service(request)
        if body.workflow_prompt == "":
            await service.clear_prompt(
                user_id=principal.user_id,
                actor_user_id=principal.user_id,
            )
            prompt = ""
        else:
            prompt = await service.set_prompt(
                user_id=principal.user_id,
                workflow_prompt=body.workflow_prompt,
                actor_user_id=principal.user_id,
            )
        return EnterpriseUserAgentWorkflowPromptResponse(
            user_id=principal.user_id,
            workflow_prompt=prompt,
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
            actor_tenant_id=principal.tenant_id,
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
            actor_tenant_id=principal.tenant_id,
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
    async def list_enterprise_users(
        request: Request,
        status: str | None = None,
        tenant_id: str | None = None,
        q: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ):
        user_service = get_enterprise_user_service(request)
        users = await user_service.list_users()
        if status:
            users = [user for user in users if user.status == status]
        if tenant_id:
            users = [user for user in users if user.tenant_id == tenant_id]
        if q:
            needle = q.strip().lower()
            users = [user for user in users if needle in user.username.lower()]
        offset = max(0, offset)
        users = users[offset:]
        if limit is not None and limit > 0:
            users = users[:limit]
        return [EnterpriseUserResponse.from_record(user) for user in users]

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
            actor_tenant_id=principal.tenant_id,
            can_create_kb=body.can_create_kb,
            can_use_bypass_query=body.can_use_bypass_query,
            can_use_agent_query=body.can_use_agent_query,
            can_delete_documents=body.can_delete_documents,
            can_download_files=body.can_download_files,
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

    @router.get(
        "/admin/users/{user_id}/access",
        response_model=EnterpriseUserAccessResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_enterprise_user_access(user_id: str, request: Request):
        user_service = get_enterprise_user_service(request)
        user = await user_service.get_user_or_404(user_id)
        authz_service = get_enterprise_authorization_service(request)
        memberships = await authz_service.list_user_tenant_memberships(user_id)
        kb_acls = await authz_service.list_user_kb_acls(user_id)
        return EnterpriseUserAccessResponse(
            user_id=user.id,
            username=user.username,
            system_role=user.system_role,
            tenant_id=user.tenant_id,
            can_create_kb=user.can_create_kb,
            can_use_bypass_query=user.can_use_bypass_query,
            can_use_agent_query=user.can_use_agent_query,
            can_delete_documents=user.can_delete_documents,
            can_download_files=user.can_download_files,
            tenant_memberships=[
                {"tenant_id": m.tenant_id, "role": m.role} for m in memberships
            ],
            kb_acls=kb_acls,
        )

    @router.get(
        "/admin/overview",
        dependencies=[Depends(combined_auth)],
    )
    async def admin_platform_overview(request: Request):
        """Platform-wide JSON aggregates for an admin dashboard.

        Super-admin gated via the /admin prefix guard. Control-plane only:
        counts come from the metadata store and KB catalog, no LightRAG
        instance is loaded (graph scale stays on per-KB ``graph/status``).
        """
        require_principal(request)
        store = getattr(request.app.state, "metadata_store", None)
        if store is None:
            raise HTTPException(status_code=500, detail="Metadata store unavailable")
        service = kb_service or getattr(request.app.state, "kb_service", None)
        kbs_by_status: dict[str, int] = {}
        if isinstance(service, KnowledgeBaseService):
            for record in await service.list(include_deleted=True):
                kbs_by_status[record.status] = kbs_by_status.get(record.status, 0) + 1
        control = await store.aggregate_control_plane_stats(None)
        enterprise = await store.aggregate_enterprise_stats()
        documents_by_status = control["documents_by_status"]
        jobs_by_status = control["jobs_by_status"]
        memory_service = get_enterprise_chat_memory_service(request)
        chat_memory = (
            await memory_service.global_stats()
            if memory_service is not None
            else {"enabled": False, "available": False, "episode_count": 0}
        )
        return {
            "kbs": {"total": sum(kbs_by_status.values()), "by_status": kbs_by_status},
            "documents": {
                "total": sum(documents_by_status.values()),
                "by_status": documents_by_status,
            },
            "counters": control["document_counters"],
            "jobs": {
                "total": sum(jobs_by_status.values()),
                "by_status": jobs_by_status,
                "dead_letter": control["dead_letter_jobs"],
            },
            "artifacts": {"total": control["artifacts"]},
            "enterprise": enterprise,
            "chat_memory": chat_memory,
        }

    @router.post(
        "/admin/users/{user_id}/chat-memory:purge",
        dependencies=[Depends(combined_auth)],
    )
    async def admin_purge_user_chat_memory(
        user_id: str, request: Request, body: AdminChatMemoryPurgeRequest | None = None
    ):
        """Super-admin: durably enqueue per-project Chat Memory purges."""
        principal = require_principal(request)
        memory_service = chat_memory_maintenance_service(request)
        if memory_service is None:
            raise HTTPException(
                status_code=503, detail="Chat memory maintenance is not enabled"
            )
        await get_enterprise_user_service(request).get_user_or_404(user_id)
        project_ids = list(body.project_ids) if body and body.project_ids else None
        chat_service = getattr(
            request.app.state, "enterprise_chat_conversation_service", None
        )
        if project_ids is None:
            project_ids = []
            if chat_service is not None:
                offset = 0
                while True:
                    projects, total = await chat_service.list_projects(
                        user_id, limit=200, offset=offset
                    )
                    if not projects:
                        break
                    project_ids.extend(project.id for project in projects)
                    offset += len(projects)
                    if offset >= total:
                        break
        else:
            project_ids = list(dict.fromkeys(project_ids))
            if chat_service is None:
                raise HTTPException(
                    status_code=503, detail="Chat conversation service unavailable"
                )
            for project_id in project_ids:
                if await chat_service.get_project(user_id, project_id) is None:
                    raise HTTPException(status_code=404, detail="Chat project not found")
        if not project_ids:
            return {"queued": 0, "noop": 0, "project_ids": []}
        try:
            result = await memory_service.enqueue_purge_projects(
                user_id,
                project_ids,
                actor_user_id=principal.user_id,
                actor_tenant_id=principal.tenant_id,
            )
        except MetadataConflictError as exc:
            raise chat_memory_write_conflict(exc) from exc
        except ChatMemoryUnavailableError:
            raise HTTPException(
                status_code=503, detail="Chat memory is temporarily unavailable"
            )
        return {
            "queued": int(result["queued"]),
            "noop": int(result["noop"]),
            "project_ids": project_ids,
        }

    @router.post(
        "/admin/chat-memory/events/{event_id}:retry",
        response_model=AdminChatMemoryEventResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def admin_retry_chat_memory_event(event_id: str, request: Request):
        """Super-admin: requeue one durable dead-letter purge by event id."""

        principal = require_principal(request)
        memory_service = chat_memory_maintenance_service(request)
        if memory_service is None:
            raise HTTPException(
                status_code=503, detail="Chat memory maintenance is not enabled"
            )
        try:
            event = await memory_service.retry_purge_event(
                event_id,
                actor_user_id=principal.user_id,
                actor_tenant_id=principal.tenant_id,
            )
        except ChatMemoryEventNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail="Chat memory event not found"
            ) from exc
        except ChatMemoryRetryConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail={"error_code": exc.error_code, "message": exc.message},
            ) from exc
        except ChatMemoryUnavailableError as exc:
            raise HTTPException(
                status_code=503, detail="Chat memory is temporarily unavailable"
            ) from exc
        return AdminChatMemoryEventResponse(
            event_id=event.event_id,
            status=event.status,
            user_id=event.user_id,
            project_id=event.project_id,
            event_type=event.event_type,
        )

    @router.post(
        "/admin/chat-memory:backlog-scan",
        dependencies=[Depends(combined_auth)],
    )
    async def admin_chat_memory_backlog_scan(
        request: Request, body: AdminChatMemoryBacklogScanRequest | None = None
    ):
        """Super-admin: recover stale durable claims and wake the outbox worker."""
        require_principal(request)
        memory_service = chat_memory_maintenance_service(request)
        worker = chat_memory_worker(request)
        if memory_service is None or worker is None:
            raise HTTPException(
                status_code=503, detail="Chat memory maintenance is not enabled"
            )
        limit = body.limit if body and body.limit else 100
        try:
            recovered = await worker.recover_once(limit=limit)
            try:
                worker.nudge()
            except Exception as exc:  # noqa: BLE001 - recovery already committed
                logger.warning("Chat Memory worker nudge failed: %s", exc)
            outbox = await memory_service.outbox_stats()
        except ChatMemoryUnavailableError:
            raise HTTPException(
                status_code=503, detail="Chat memory is temporarily unavailable"
            )
        return {"recovered_events": recovered, "outbox": outbox}

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
            actor_tenant_id=principal.tenant_id,
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
            actor_tenant_id=principal.tenant_id,
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
            actor_tenant_id=principal.tenant_id,
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
        # ``tenant_id`` distinguishes omitted (unchanged) from an explicit
        # ``null`` (clear the tenant assignment), mirroring KB PATCH semantics.
        tenant_provided = "tenant_id" in body.model_fields_set
        if (
            any(
                value is not None
                for value in (
                    body.status,
                    body.can_create_kb,
                    body.can_use_bypass_query,
                    body.can_use_agent_query,
                    body.can_delete_documents,
                    body.can_download_files,
                )
            )
            or tenant_provided
        ):
            user = await user_service.update_user(
                user_id,
                status_value=body.status,
                can_create_kb=body.can_create_kb,
                can_use_bypass_query=body.can_use_bypass_query,
                can_use_agent_query=body.can_use_agent_query,
                can_delete_documents=body.can_delete_documents,
                can_download_files=body.can_download_files,
                tenant_id=body.tenant_id if tenant_provided else UNSET,
                actor_user_id=principal.user_id,
                actor_tenant_id=principal.tenant_id,
            )
        if body.password is not None:
            user = await user_service.change_password(
                user_id,
                body.password,
                actor_user_id=principal.user_id,
                actor_tenant_id=principal.tenant_id,
            )
        return EnterpriseUserResponse.from_record(user)

    @router.delete(
        "/admin/users/{user_id}",
        dependencies=[Depends(combined_auth)],
    )
    async def delete_enterprise_user(user_id: str, request: Request):
        principal = require_principal(request)
        deleted = await delete_user_with_chat_memory(
            request,
            user_id,
            actor_user_id=principal.user_id,
            actor_tenant_id=principal.tenant_id,
        )
        return {"deleted": deleted}

    @router.post(
        "/admin/users/{user_id}/kb-access:batch-set",
        response_model=EnterpriseUserKBAccessBatchSetResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def batch_set_user_kb_access(
        user_id: str, request: Request, body: EnterpriseUserKBAccessBatchSetRequest
    ):
        principal = require_principal(request)
        await get_enterprise_user_service(request).get_user_or_404(user_id)
        authz_service = get_enterprise_authorization_service(request)
        for entry in body.entries:
            await require_kb_exists(request, entry.kb_id)
            if entry.action == "grant" and entry.role is None:
                raise HTTPException(status_code=400, detail="Role is required for ACL grants")

        granted: list[EnterpriseACLResponse] = []
        revoked: list[str] = []
        for entry in body.entries:
            if entry.action == "revoke":
                if await authz_service.revoke_kb_role(
                    entry.kb_id,
                    user_id,
                    actor_user_id=principal.user_id,
                ):
                    revoked.append(entry.kb_id)
                continue
            assert entry.role is not None
            record = await authz_service.grant_kb_role(
                entry.kb_id,
                user_id,
                entry.role,
                granted_by=principal.user_id,
            )
            granted.append(EnterpriseACLResponse.from_record(record))
        return EnterpriseUserKBAccessBatchSetResponse(granted=granted, revoked=revoked)

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

    @router.post(
        "/admin/tenants",
        response_model=EnterpriseTenantResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def create_tenant(request: Request, body: EnterpriseTenantCreateRequest):
        principal = require_principal(request)
        authz_service = get_enterprise_authorization_service(request)
        tenant = await authz_service.create_tenant(
            name=body.name,
            description=body.description,
            tenant_id=body.tenant_id,
            created_by=principal.user_id,
            actor_tenant_id=principal.tenant_id,
        )
        return EnterpriseTenantResponse.from_record(tenant)

    @router.get(
        "/admin/tenants",
        response_model=list[EnterpriseTenantResponse],
        dependencies=[Depends(combined_auth)],
    )
    async def list_tenants(request: Request):
        authz_service = get_enterprise_authorization_service(request)
        return [
            EnterpriseTenantResponse.from_record(tenant)
            for tenant in await authz_service.list_tenants()
        ]

    @router.get(
        "/admin/tenants/{tenant_id}",
        response_model=EnterpriseTenantDetailResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_tenant(tenant_id: str, request: Request):
        principal = require_principal(request)
        if not principal.is_super_admin:
            await require_tenant_admin(request, tenant_id)
        authz_service = get_enterprise_authorization_service(request)
        tenant = await authz_service.get_tenant_or_404(tenant_id)
        members = await authz_service.list_tenant_memberships(tenant_id)
        kb_count = len(await active_tenant_kbs(request, tenant_id))
        data = tenant.to_dict()
        data.pop("metadata", None)
        return EnterpriseTenantDetailResponse(
            **data, member_count=len(members), kb_count=kb_count
        )

    @router.patch(
        "/admin/tenants/{tenant_id}",
        response_model=EnterpriseTenantResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def update_tenant(
        tenant_id: str, request: Request, body: EnterpriseTenantUpdateRequest
    ):
        principal = require_principal(request)
        authz_service = get_enterprise_authorization_service(request)
        tenant = await authz_service.update_tenant(
            tenant_id,
            name=body.name,
            description=body.description,
            status_value=body.status,
            actor_user_id=principal.user_id,
            actor_tenant_id=principal.tenant_id,
        )
        return EnterpriseTenantResponse.from_record(tenant)

    @router.delete(
        "/admin/tenants/{tenant_id}",
        dependencies=[Depends(combined_auth)],
    )
    async def delete_tenant(tenant_id: str, request: Request):
        principal = require_principal(request)
        authz_service = get_enterprise_authorization_service(request)
        await authz_service.get_tenant_or_404(tenant_id)
        # Refuse only while STRUCTURAL references remain (members, users whose
        # canonical tenant is this one, tenant-owned KBs). Revocable grants TO
        # the tenant (tenant KB ACLs, per-user overrides, person-KB share
        # oversight) are not containment: deleting the tenant revokes them in
        # the same store transaction. Stale grant rows — e.g. an ACL whose KB
        # was deleted later — must never wedge tenant deletion.
        member_count = len(await authz_service.list_tenant_memberships(tenant_id))
        acl_kb_ids = await authz_service.list_kb_ids_for_tenants([tenant_id])
        kb_count = sum(
            1
            for kb in await request_kb_service(request).list(
                include_deleted=False
            )
            if kb.tenant_id == tenant_id
        )
        user_service = get_enterprise_user_service(request)
        user_count = sum(
            1 for user in await user_service.list_users() if user.tenant_id == tenant_id
        )
        if member_count or kb_count or user_count:
            raise HTTPException(
                status_code=409,
                detail={
                    "error_code": "tenant_not_empty",
                    "message": (
                        "Tenant still has references; reassign or remove them first"
                    ),
                    "member_count": member_count,
                    "kb_count": kb_count,
                    "user_count": user_count,
                    "tenant_kb_acl_count": len(acl_kb_ids),
                },
            )
        deleted = await authz_service.delete_tenant(
            tenant_id,
            actor_user_id=principal.user_id,
            actor_tenant_id=principal.tenant_id,
        )
        return {"deleted": deleted, "removed_tenant_kb_acls": len(acl_kb_ids)}

    @router.get(
        "/admin/tenants/{tenant_id}/kbs",
        response_model=list[EnterpriseTenantKBSummaryResponse],
        dependencies=[Depends(combined_auth)],
    )
    async def list_tenant_kbs(tenant_id: str, request: Request):
        authz_service = get_enterprise_authorization_service(request)
        await authz_service.get_tenant_or_404(tenant_id)
        return [
            EnterpriseTenantKBSummaryResponse(
                id=kb.id,
                name=kb.name,
                status=kb.status,
                visibility=getattr(kb, "visibility", None),
                owner_id=getattr(kb, "owner_id", None),
            )
            for kb in await active_tenant_kbs(request, tenant_id)
        ]

    @router.get(
        "/admin/tenants/{tenant_id}/members",
        response_model=list[EnterpriseTenantMembershipResponse],
        dependencies=[Depends(combined_auth)],
    )
    async def list_tenant_members(tenant_id: str, request: Request):
        authz_service = get_enterprise_authorization_service(request)
        return await memberships_with_user_info(
            request, await authz_service.list_tenant_memberships(tenant_id)
        )

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
            actor_tenant_id=principal.tenant_id,
        )
        return (await memberships_with_user_info(request, [record]))[0]

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
            actor_tenant_id=principal.tenant_id,
        )
        return {"deleted": deleted}

    @router.get(
        "/tenants/{tenant_id}",
        response_model=EnterpriseTenantDetailResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_scoped_tenant(tenant_id: str, request: Request):
        await require_tenant_admin(request, tenant_id)
        authz_service = get_enterprise_authorization_service(request)
        tenant = await authz_service.get_tenant_or_404(tenant_id)
        members = await authz_service.list_tenant_memberships(tenant_id)
        data = tenant.to_dict()
        data.pop("metadata", None)
        return EnterpriseTenantDetailResponse(
            **data,
            member_count=len(members),
            kb_count=len(await active_tenant_kbs(request, tenant_id)),
        )

    @router.get(
        "/tenants/{tenant_id}/kbs",
        response_model=list[EnterpriseTenantKBSummaryResponse],
        dependencies=[Depends(combined_auth)],
    )
    async def list_scoped_tenant_kbs(tenant_id: str, request: Request):
        await require_tenant_admin(request, tenant_id)
        return [
            EnterpriseTenantKBSummaryResponse(
                id=record.id,
                name=record.name,
                status=record.status,
                visibility=getattr(record, "visibility", None),
                owner_id=getattr(record, "owner_id", None),
            )
            for record in await active_tenant_kbs(request, tenant_id)
        ]

    @router.get(
        "/tenants/{tenant_id}/users",
        response_model=list[EnterpriseUserResponse],
        dependencies=[Depends(combined_auth)],
    )
    async def list_scoped_tenant_users(
        tenant_id: str,
        request: Request,
        status: str | None = None,
        q: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ):
        await require_tenant_admin(request, tenant_id)
        authz_service = get_enterprise_authorization_service(request)
        member_ids = {
            membership.user_id
            for membership in await authz_service.list_tenant_memberships(
                tenant_id
            )
        }
        users = [
            user
            for user in await get_enterprise_user_service(request).list_users()
            if user.id in member_ids
            and user.tenant_id == tenant_id
            and user.system_role != SYSTEM_ROLE_SUPER_ADMIN
        ]
        if status:
            users = [user for user in users if user.status == status]
        if q:
            needle = q.strip().lower()
            users = [user for user in users if needle in user.username.lower()]
        users = users[max(0, offset) :]
        if limit is not None and limit > 0:
            users = users[:limit]
        return [EnterpriseUserResponse.from_record(user) for user in users]

    @router.post(
        "/tenants/{tenant_id}/users",
        response_model=EnterpriseUserResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def create_scoped_tenant_user(
        tenant_id: str,
        request: Request,
        body: EnterpriseTenantUserCreateRequest,
    ):
        principal = await require_tenant_admin(request, tenant_id)
        if "tenant_id" in body.model_fields_set and body.tenant_id != tenant_id:
            raise HTTPException(
                status_code=400,
                detail="User tenant_id must match the path tenant",
            )
        user = await get_enterprise_user_service(request).create_user(
            username=body.username,
            password=body.password,
            created_by=principal.user_id,
            actor_tenant_id=principal.tenant_id,
            can_create_kb=body.can_create_kb,
            can_use_bypass_query=body.can_use_bypass_query,
            can_use_agent_query=body.can_use_agent_query,
            can_delete_documents=body.can_delete_documents,
            can_download_files=body.can_download_files,
            tenant_id=tenant_id,
        )
        return EnterpriseUserResponse.from_record(user)

    @router.get(
        "/tenants/{tenant_id}/users/{user_id}",
        response_model=EnterpriseUserResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_scoped_tenant_user(
        tenant_id: str, user_id: str, request: Request
    ):
        await require_tenant_admin(request, tenant_id)
        user, _membership = await tenant_user_or_404(
            request, tenant_id, user_id
        )
        return EnterpriseUserResponse.from_record(user)

    @router.patch(
        "/tenants/{tenant_id}/users/{user_id}",
        response_model=EnterpriseUserResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def update_scoped_tenant_user(
        tenant_id: str,
        user_id: str,
        request: Request,
        body: EnterpriseTenantUserUpdateRequest,
    ):
        principal = await require_tenant_admin(request, tenant_id)
        user, membership = await mutable_tenant_member_or_404(
            request,
            tenant_id,
            user_id,
            actor_user_id=principal.user_id,
        )
        if "tenant_id" in body.model_fields_set or "password" in body.model_fields_set:
            raise HTTPException(
                status_code=400,
                detail="Tenant assignment and password cannot be changed here",
            )
        if any(
            value is not None
            for value in (
                body.status,
                body.can_create_kb,
                body.can_use_bypass_query,
                body.can_use_agent_query,
                body.can_delete_documents,
                body.can_download_files,
            )
        ):
            user = await get_enterprise_user_service(request).update_user(
                user_id,
                status_value=body.status,
                can_create_kb=body.can_create_kb,
                can_use_bypass_query=body.can_use_bypass_query,
                can_use_agent_query=body.can_use_agent_query,
                can_delete_documents=body.can_delete_documents,
                can_download_files=body.can_download_files,
                actor_user_id=principal.user_id,
                actor_tenant_id=principal.tenant_id,
                expected_user=user,
                expected_membership=membership,
            )
        return EnterpriseUserResponse.from_record(user)

    @router.post(
        "/tenants/{tenant_id}/users/{user_id}:disable",
        response_model=EnterpriseUserResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def disable_scoped_tenant_user(
        tenant_id: str, user_id: str, request: Request
    ):
        principal = await require_tenant_admin(request, tenant_id)
        user, membership = await mutable_tenant_member_or_404(
            request,
            tenant_id,
            user_id,
            actor_user_id=principal.user_id,
        )
        user = await get_enterprise_user_service(request).update_user(
            user_id,
            status_value=USER_STATUS_DISABLED,
            actor_user_id=principal.user_id,
            actor_tenant_id=principal.tenant_id,
            expected_user=user,
            expected_membership=membership,
        )
        return EnterpriseUserResponse.from_record(user)

    @router.post(
        "/tenants/{tenant_id}/users/{user_id}:enable",
        response_model=EnterpriseUserResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def enable_scoped_tenant_user(
        tenant_id: str, user_id: str, request: Request
    ):
        principal = await require_tenant_admin(request, tenant_id)
        user, membership = await mutable_tenant_member_or_404(
            request,
            tenant_id,
            user_id,
            actor_user_id=principal.user_id,
        )
        user = await get_enterprise_user_service(request).update_user(
            user_id,
            status_value=USER_STATUS_ACTIVE,
            actor_user_id=principal.user_id,
            actor_tenant_id=principal.tenant_id,
            expected_user=user,
            expected_membership=membership,
        )
        return EnterpriseUserResponse.from_record(user)

    @router.post(
        "/tenants/{tenant_id}/users/{user_id}:reset-password",
        response_model=EnterpriseUserResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def reset_scoped_tenant_user_password(
        tenant_id: str,
        user_id: str,
        request: Request,
        body: EnterpriseUserResetPasswordRequest,
    ):
        principal = await require_tenant_admin(request, tenant_id)
        user, membership = await mutable_tenant_member_or_404(
            request,
            tenant_id,
            user_id,
            actor_user_id=principal.user_id,
        )
        user = await get_enterprise_user_service(request).change_password(
            user_id,
            body.password,
            actor_user_id=principal.user_id,
            actor_tenant_id=principal.tenant_id,
            expected_user=user,
            expected_membership=membership,
        )
        return EnterpriseUserResponse.from_record(user)

    @router.delete(
        "/tenants/{tenant_id}/users/{user_id}",
        dependencies=[Depends(combined_auth)],
    )
    async def delete_scoped_tenant_user(
        tenant_id: str, user_id: str, request: Request
    ):
        principal = await require_tenant_admin(request, tenant_id)
        user, membership = await mutable_tenant_member_or_404(
            request,
            tenant_id,
            user_id,
            actor_user_id=principal.user_id,
        )
        deleted = await delete_user_with_chat_memory(
            request,
            user_id,
            actor_user_id=principal.user_id,
            actor_tenant_id=principal.tenant_id,
            expected_user=user,
            expected_membership=membership,
        )
        return {"deleted": deleted}

    @router.get(
        "/tenants/{tenant_id}/members",
        response_model=list[EnterpriseTenantMembershipResponse],
        dependencies=[Depends(combined_auth)],
    )
    async def self_service_list_tenant_members(tenant_id: str, request: Request):
        await require_tenant_admin(request, tenant_id)
        authz_service = get_enterprise_authorization_service(request)
        return await memberships_with_user_info(
            request, await authz_service.list_tenant_memberships(tenant_id)
        )

    @router.put(
        "/tenants/{tenant_id}/members/{user_id}",
        response_model=EnterpriseTenantMembershipResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def self_service_grant_tenant_membership(
        tenant_id: str,
        user_id: str,
        request: Request,
        body: EnterpriseTenantMembershipGrantRequest,
    ):
        principal = await require_tenant_admin(request, tenant_id)
        target_user = await unassigned_user_or_404(
            request,
            user_id,
            actor_user_id=principal.user_id,
        )
        if body.role.strip() not in {
            TENANT_ROLE_MEMBER,
            "member",
        }:
            raise HTTPException(
                status_code=403,
                detail="Tenant admins can only grant tenant_member via self-service",
            )
        authz_service = get_enterprise_authorization_service(request)
        record = await authz_service.grant_tenant_membership(
            tenant_id,
            user_id,
            body.role,
            granted_by=principal.user_id,
            actor_tenant_id=principal.tenant_id,
            expected_user=target_user,
            expected_membership=None,
            allow_tenant_move=False,
        )
        return (await memberships_with_user_info(request, [record]))[0]

    @router.delete(
        "/tenants/{tenant_id}/members/{user_id}",
        dependencies=[Depends(combined_auth)],
    )
    async def self_service_revoke_tenant_membership(
        tenant_id: str, user_id: str, request: Request
    ):
        principal = await require_tenant_admin(request, tenant_id)
        authz_service = get_enterprise_authorization_service(request)
        user, existing = await mutable_tenant_member_or_404(
            request,
            tenant_id,
            user_id,
            actor_user_id=principal.user_id,
        )
        deleted = await authz_service.revoke_tenant_membership(
            tenant_id,
            user_id,
            actor_user_id=principal.user_id,
            actor_tenant_id=principal.tenant_id,
            expected_user=user,
            expected_membership=existing,
        )
        return {"deleted": deleted}

    @router.get(
        "/tenants/{tenant_id}/audit-events",
        response_model=list[EnterpriseAuditEventResponse],
        dependencies=[Depends(combined_auth)],
    )
    async def list_scoped_tenant_audit_events(
        tenant_id: str,
        request: Request,
        limit: int = 100,
        offset: int = 0,
        event_type: str | None = None,
        actor_user_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
    ):
        await require_tenant_admin(request, tenant_id)
        events = await get_enterprise_audit_service(request).list(
            limit=limit,
            offset=offset,
            event_type=event_type,
            actor_user_id=actor_user_id,
            actor_tenant_id=tenant_id,
            target_type=target_type,
            target_id=target_id,
            created_after=created_after,
            created_before=created_before,
        )
        return await audit_event_responses(request, events)

    @router.get(
        "/tenants/{tenant_id}/kbs/{kb_id}/members",
        response_model=list[EnterpriseTenantKBMemberAccessResponse],
        dependencies=[Depends(combined_auth)],
    )
    async def list_scoped_tenant_kb_members(
        tenant_id: str, kb_id: str, request: Request
    ):
        await require_tenant_admin(request, tenant_id)
        record, _tenant_acl_role, _tenant_owned = await manageable_tenant_kb(
            request, tenant_id, kb_id
        )
        authz_service = get_enterprise_authorization_service(request)
        memberships = await authz_service.list_tenant_memberships(tenant_id)
        users_by_id = {
            user.id: user
            for user in await get_enterprise_user_service(request).list_users()
            if user.tenant_id == tenant_id
            and user.system_role != SYSTEM_ROLE_SUPER_ADMIN
        }
        results: list[EnterpriseTenantKBMemberAccessResponse] = []
        for membership in memberships:
            user = users_by_id.get(membership.user_id)
            if user is None:
                continue
            results.append(
                await tenant_kb_member_access_response(
                    request,
                    tenant_id,
                    record,
                    user,
                    membership,
                )
            )
        return results

    @router.put(
        "/tenants/{tenant_id}/kbs/{kb_id}/members/{user_id}",
        response_model=EnterpriseTenantKBMemberAccessResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def grant_scoped_tenant_kb_member(
        tenant_id: str,
        kb_id: str,
        user_id: str,
        request: Request,
        body: EnterpriseTenantKBMemberRoleRequest,
    ):
        principal = await require_tenant_admin(request, tenant_id)
        record, tenant_acl_role, tenant_owned = await manageable_tenant_kb(
            request, tenant_id, kb_id
        )
        user, membership = await mutable_tenant_member_or_404(
            request,
            tenant_id,
            user_id,
            actor_user_id=principal.user_id,
        )
        requested_role = _TENANT_KB_MEMBER_ROLES[body.role]
        if (
            not tenant_owned
            and _KB_ROLE_RANK[requested_role]
            > _KB_ROLE_RANK.get(tenant_acl_role or "", 0)
        ):
            raise HTTPException(
                status_code=400,
                detail="Member role cannot exceed the tenant KB ACL role",
            )
        await get_enterprise_authorization_service(
            request
        ).grant_tenant_user_kb_override(
            kb_id,
            tenant_id,
            user_id,
            requested_role,
            granted_by=principal.user_id,
            actor_tenant_id=principal.tenant_id,
            expected_generation=record.generation,
            expected_user=user,
            expected_membership=membership,
        )
        return await tenant_kb_member_access_response(
            request,
            tenant_id,
            record,
            user,
            membership,
        )

    @router.delete(
        "/tenants/{tenant_id}/kbs/{kb_id}/members/{user_id}",
        response_model=EnterpriseTenantKBMemberAccessResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def revoke_scoped_tenant_kb_member(
        tenant_id: str,
        kb_id: str,
        user_id: str,
        request: Request,
        reset: bool = False,
    ):
        principal = await require_tenant_admin(request, tenant_id)
        record, _tenant_acl_role, _tenant_owned = await manageable_tenant_kb(
            request, tenant_id, kb_id
        )
        user, membership = await mutable_tenant_member_or_404(
            request,
            tenant_id,
            user_id,
            actor_user_id=principal.user_id,
        )
        authz_service = get_enterprise_authorization_service(request)
        if reset:
            await authz_service.reset_tenant_user_kb_override(
                kb_id,
                tenant_id,
                user_id,
                actor_user_id=principal.user_id,
                actor_tenant_id=principal.tenant_id,
                expected_generation=record.generation,
                expected_user=user,
                expected_membership=membership,
            )
        else:
            await authz_service.revoke_tenant_user_kb_override(
                kb_id,
                tenant_id,
                user_id,
                granted_by=principal.user_id,
                actor_tenant_id=principal.tenant_id,
                expected_generation=record.generation,
                expected_user=user,
                expected_membership=membership,
            )
        return await tenant_kb_member_access_response(
            request,
            tenant_id,
            record,
            user,
            membership,
        )

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
        if body.inherit_tenant_kb_acl and not body.tenant_id:
            raise HTTPException(
                status_code=400,
                detail="tenant_id is required when inherit_tenant_kb_acl is true",
            )
        api_key_service = get_enterprise_api_key_service(request)
        expires_at = None
        if body.expires_in_seconds is not None:
            expires_at = (
                datetime.now(timezone.utc)
                + timedelta(seconds=body.expires_in_seconds)
            ).isoformat()
        record, raw_key = await api_key_service.create_key(
            name=body.name,
            scopes={
                "kb_roles": body.kb_roles,
                "can_use_bypass_query": body.can_use_bypass_query,
                "can_use_agent_query": body.can_use_agent_query,
                "inherit_tenant_kb_acl": body.inherit_tenant_kb_acl,
            },
            metadata=body.metadata,
            created_by=principal.user_id,
            tenant_id=body.tenant_id,
            expires_at=expires_at,
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

    @router.post(
        "/admin/service-api-keys/{key_id}:rotate",
        response_model=EnterpriseServiceAPIKeyCreateResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def rotate_service_api_key(
        key_id: str, request: Request, body: EnterpriseServiceAPIKeyRotateRequest | None = None
    ):
        principal = require_principal(request)
        api_key_service = get_enterprise_api_key_service(request)
        expires_at = None
        if body is not None and body.expires_in_seconds is not None:
            expires_at = (
                datetime.now(timezone.utc)
                + timedelta(seconds=body.expires_in_seconds)
            ).isoformat()
        record, raw_key = await api_key_service.rotate_key(
            key_id,
            rotated_by=principal.user_id,
            expires_at=expires_at,
            revoke_old=True if body is None else body.revoke_old,
        )
        return EnterpriseServiceAPIKeyCreateResponse(
            api_key=raw_key,
            key=EnterpriseServiceAPIKeyResponse.from_record(record),
        )

    @router.get(
        "/admin/invitations",
        response_model=list[EnterpriseInvitationResponse],
        dependencies=[Depends(combined_auth)],
    )
    async def list_invitations(request: Request):
        invitation_service = get_enterprise_invitation_service(request)
        return [
            EnterpriseInvitationResponse.from_record(record)
            for record in await invitation_service.list_invitations()
        ]

    @router.post(
        "/admin/invitations",
        response_model=EnterpriseInvitationCreateResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def create_invitation(
        request: Request, body: EnterpriseInvitationCreateRequest
    ):
        principal = require_principal(request)
        invitation_service = get_enterprise_invitation_service(request)
        expires_at = None
        if body.expires_in_seconds is not None:
            expires_at = (
                datetime.now(timezone.utc)
                + timedelta(seconds=body.expires_in_seconds)
            ).isoformat()
        record, raw_token = await invitation_service.create_invitation(
            created_by=principal.user_id,
            expires_at=expires_at,
            metadata=body.metadata,
        )
        return EnterpriseInvitationCreateResponse(
            invitation_token=raw_token,
            invitation=EnterpriseInvitationResponse.from_record(record),
        )

    @router.post(
        "/admin/invitations/{invitation_id}:revoke",
        response_model=EnterpriseInvitationResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def revoke_invitation(invitation_id: str, request: Request):
        principal = require_principal(request)
        invitation_service = get_enterprise_invitation_service(request)
        revoked = await invitation_service.revoke_invitation(
            invitation_id, actor_user_id=principal.user_id
        )
        return EnterpriseInvitationResponse.from_record(revoked)

    @router.get(
        "/admin/audit-events",
        response_model=list[EnterpriseAuditEventResponse],
        dependencies=[Depends(combined_auth)],
    )
    async def list_audit_events(
        request: Request,
        limit: int = 100,
        offset: int = 0,
        event_type: str | None = None,
        actor_user_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
    ):
        audit_service = get_enterprise_audit_service(request)
        events = await audit_service.list(
            limit=limit,
            offset=offset,
            event_type=event_type,
            actor_user_id=actor_user_id,
            target_type=target_type,
            target_id=target_id,
            created_after=created_after,
            created_before=created_before,
        )
        return await audit_event_responses(request, events)

    return router
