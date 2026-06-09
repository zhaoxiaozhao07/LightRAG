from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import contextvars
import hashlib
import json
from importlib import import_module
import os
import secrets
import time
from types import SimpleNamespace
from typing import Any, Callable, Protocol
from uuid import uuid4

from fastapi import HTTPException, Request, status

from lightrag.api.kb_service import KnowledgeBaseRecord, utc_now_iso
from lightrag.api.metadata_store import (
    AuditEventRecord,
    EnterpriseAPIKeyRecord,
    EnterpriseInvitationRecord,
    EnterpriseUserKBQuerySettingsRecord,
    EnterpriseUserRecord,
    KBACLRecord,
    EnterpriseTenantKBACLRecord,
    EnterpriseTenantMembershipRecord,
    EnterpriseTenantRecord,
)
from lightrag.api.passwords import hash_password, verify_password

ENTERPRISE_REGISTRATION_ENABLED_KEY = "registration_enabled"
ENTERPRISE_REGISTRATION_MODE_KEY = "registration_mode"
REGISTRATION_MODE_DISABLED = "disabled"
REGISTRATION_MODE_OPEN = "open"
REGISTRATION_MODE_INVITE_ONLY = "invite_only"
REGISTRATION_MODE_ADMIN_APPROVAL = "admin_approval"
REGISTRATION_MODES = {
    REGISTRATION_MODE_DISABLED,
    REGISTRATION_MODE_OPEN,
    REGISTRATION_MODE_INVITE_ONLY,
    REGISTRATION_MODE_ADMIN_APPROVAL,
}
USER_STATUS_ACTIVE = "active"
USER_STATUS_DISABLED = "disabled"
USER_STATUS_PENDING = "pending"
USER_STATUS_VALUES = {USER_STATUS_ACTIVE, USER_STATUS_DISABLED, USER_STATUS_PENDING}
ENTERPRISE_INVITATION_STATUS_ACTIVE = "active"
ENTERPRISE_INVITATION_STATUS_USED = "used"
ENTERPRISE_INVITATION_STATUS_REVOKED = "revoked"
ENTERPRISE_API_KEY_STATUS_ACTIVE = "active"
ENTERPRISE_API_KEY_STATUS_REVOKED = "revoked"
ENTERPRISE_API_KEY_STATUS_VALUES = {
    ENTERPRISE_API_KEY_STATUS_ACTIVE,
    ENTERPRISE_API_KEY_STATUS_REVOKED,
}
SERVICE_API_KEY_AUTH_METHOD = "service_api_key"
SYSTEM_ROLE_SUPER_ADMIN = "super_admin"
SYSTEM_ROLE_USER = "user"
KB_ROLE_VIEWER = "kb_viewer"
KB_ROLE_EDITOR = "kb_editor"
KB_ROLE_ADMIN = "kb_admin"
KB_ROLE_OWNER = "kb_owner"
TENANT_ROLE_MEMBER = "tenant_member"
TENANT_ROLE_ADMIN = "tenant_admin"
TENANT_ROLE_OWNER = "tenant_owner"

_KB_ROLE_RANK = {
    KB_ROLE_VIEWER: 1,
    KB_ROLE_EDITOR: 2,
    KB_ROLE_ADMIN: 3,
    KB_ROLE_OWNER: 4,
}
_KB_ROLE_ALIASES = {
    "viewer": KB_ROLE_VIEWER,
    "editor": KB_ROLE_EDITOR,
    "admin": KB_ROLE_ADMIN,
    "owner": KB_ROLE_OWNER,
}
_TENANT_ROLE_RANK = {
    TENANT_ROLE_MEMBER: 1,
    TENANT_ROLE_ADMIN: 2,
    TENANT_ROLE_OWNER: 3,
}
_TENANT_ROLE_ALIASES = {
    "member": TENANT_ROLE_MEMBER,
    "admin": TENANT_ROLE_ADMIN,
    "owner": TENANT_ROLE_OWNER,
}
_ARTIFACT_POLICY_ACTIONS = {"download", "download-url", "preview"}
_ENTERPRISE_PROTECTED_PREFIXES = (
    "/admin",
    "/kbs",
    "/documents",
    "/query",
    "/graph",
    "/api",
)
_LEGACY_GLOBAL_PREFIXES = ("/documents", "/query", "/graph", "/api")
_ENV_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_ENV_FALSE_VALUES = {"0", "false", "no", "n", "off"}


class EnterpriseMetadataStore(Protocol):
    async def get_enterprise_user_by_username(
        self, username: str
    ) -> EnterpriseUserRecord | None: ...

    async def get_enterprise_user_by_id(
        self, user_id: str
    ) -> EnterpriseUserRecord | None: ...

    async def list_enterprise_users(self) -> list[EnterpriseUserRecord]: ...

    async def upsert_enterprise_user(
        self, user: EnterpriseUserRecord
    ) -> EnterpriseUserRecord: ...

    async def get_enterprise_user_kb_query_settings(
        self, user_id: str, kb_id: str
    ) -> EnterpriseUserKBQuerySettingsRecord | None: ...

    async def upsert_enterprise_user_kb_query_settings(
        self, record: EnterpriseUserKBQuerySettingsRecord
    ) -> EnterpriseUserKBQuerySettingsRecord: ...

    async def delete_enterprise_user_kb_query_settings(
        self, user_id: str, kb_id: str
    ) -> bool: ...

    async def create_enterprise_api_key(
        self, record: EnterpriseAPIKeyRecord
    ) -> EnterpriseAPIKeyRecord: ...

    async def get_enterprise_api_key_by_hash(
        self, key_hash: str
    ) -> EnterpriseAPIKeyRecord | None: ...

    async def get_enterprise_api_key_by_id(
        self, key_id: str
    ) -> EnterpriseAPIKeyRecord | None: ...

    async def list_enterprise_api_keys(self) -> list[EnterpriseAPIKeyRecord]: ...

    async def revoke_enterprise_api_key(
        self,
        key_id: str,
        *,
        revoked_by: str | None = None,
        revoked_at: str | None = None,
    ) -> EnterpriseAPIKeyRecord | None: ...

    async def mark_enterprise_api_key_used(
        self, key_id: str, *, last_used_at: str | None = None
    ) -> EnterpriseAPIKeyRecord | None: ...

    async def set_enterprise_system_setting(
        self, key: str, value: str, *, updated_by: str | None = None
    ) -> None: ...

    async def get_enterprise_system_setting(
        self, key: str, default: str | None = None
    ) -> str | None: ...

    async def upsert_kb_acl(self, acl: KBACLRecord) -> KBACLRecord: ...

    async def delete_kb_acl(self, kb_id: str, user_id: str) -> bool: ...

    async def list_kb_acl(self, kb_id: str) -> list[KBACLRecord]: ...

    async def get_kb_acl_role(self, kb_id: str, user_id: str) -> str | None: ...

    async def list_kb_ids_for_user(self, user_id: str) -> list[str]: ...

    async def upsert_enterprise_tenant(
        self, tenant: EnterpriseTenantRecord
    ) -> EnterpriseTenantRecord: ...

    async def get_enterprise_tenant_by_id(
        self, tenant_id: str
    ) -> EnterpriseTenantRecord | None: ...

    async def list_enterprise_tenants(self) -> list[EnterpriseTenantRecord]: ...

    async def delete_enterprise_tenant(self, tenant_id: str) -> bool: ...

    async def upsert_tenant_membership(
        self, membership: EnterpriseTenantMembershipRecord
    ) -> EnterpriseTenantMembershipRecord: ...

    async def delete_tenant_membership(self, tenant_id: str, user_id: str) -> bool: ...

    async def list_tenant_memberships(
        self, tenant_id: str
    ) -> list[EnterpriseTenantMembershipRecord]: ...

    async def list_user_tenant_memberships(
        self, user_id: str
    ) -> list[EnterpriseTenantMembershipRecord]: ...

    async def get_tenant_membership(
        self, tenant_id: str, user_id: str
    ) -> EnterpriseTenantMembershipRecord | None: ...

    async def upsert_tenant_kb_acl(
        self, acl: EnterpriseTenantKBACLRecord
    ) -> EnterpriseTenantKBACLRecord: ...

    async def delete_tenant_kb_acl(self, tenant_id: str, kb_id: str) -> bool: ...

    async def list_kb_tenant_acl(
        self, kb_id: str
    ) -> list[EnterpriseTenantKBACLRecord]: ...

    async def get_tenant_kb_acl_role(self, tenant_id: str, kb_id: str) -> str | None: ...

    async def list_kb_ids_for_tenants(self, tenant_ids: list[str]) -> list[str]: ...

    async def append_audit_event(
        self, event: AuditEventRecord
    ) -> AuditEventRecord: ...

    async def list_audit_events(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        event_type: str | None = None,
        actor_user_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
    ) -> list[AuditEventRecord]: ...

    async def create_enterprise_invitation(
        self, record: EnterpriseInvitationRecord
    ) -> EnterpriseInvitationRecord: ...

    async def get_enterprise_invitation_by_token_hash(
        self, token_hash: str
    ) -> EnterpriseInvitationRecord | None: ...

    async def list_enterprise_invitations(self) -> list[EnterpriseInvitationRecord]: ...

    async def consume_enterprise_invitation(
        self, token_hash: str, *, used_by: str | None, used_at: str | None = None
    ) -> EnterpriseInvitationRecord | None: ...

    async def revoke_enterprise_invitation(
        self, invitation_id: str, *, revoked_at: str | None = None
    ) -> EnterpriseInvitationRecord | None: ...


@dataclass(frozen=True, slots=True)
class Principal:
    user_id: str
    username: str
    system_role: str
    status: str
    tenant_id: str | None
    tenant_roles: dict[str, str]
    can_create_kb: bool
    can_use_bypass_query: bool
    token_version: int
    auth_method: str
    metadata: dict[str, Any]
    # Capability to delete documents uploaded by other users (delete-any).
    # Declared last with a default so existing keyword constructions and the
    # frozen-dataclass field ordering stay valid.
    can_delete_documents: bool = False

    @property
    def is_super_admin(self) -> bool:
        return self.system_role == SYSTEM_ROLE_SUPER_ADMIN


_current_principal: contextvars.ContextVar[Principal | None] = contextvars.ContextVar(
    "lightrag_current_principal", default=None
)


def set_current_principal(principal: Principal | None) -> None:
    """Bind the authenticated principal to the current async context.

    Set wherever ``request.state.principal`` is set so downstream code without a
    ``Request`` (e.g. ``JobService``) can attribute work to the caller. Tasks
    spawned during the request inherit a copy of this context.
    """
    _current_principal.set(principal)


def get_current_principal() -> Principal | None:
    return _current_principal.get()


def enterprise_auth_enabled() -> bool:
    return bool(getattr(_global_args(), "enterprise_auth_enabled", False))


def enterprise_legacy_api_key_superadmin_enabled() -> bool:
    return bool(getattr(_global_args(), "enterprise_legacy_api_key_superadmin", False))


def enterprise_global_routes_disabled() -> bool:
    return bool(getattr(_global_args(), "enterprise_disable_global_routes", True))


def enterprise_artifact_download_min_role() -> str:
    configured = getattr(
        _global_args(), "enterprise_artifact_download_min_role", KB_ROLE_VIEWER
    )
    normalized = _normalize_kb_role(str(configured))
    if normalized is None:
        return KB_ROLE_VIEWER
    return normalized


def enterprise_artifact_download_policy() -> dict[str, str]:
    configured = getattr(_global_args(), "enterprise_artifact_download_policy", "")
    if not configured:
        return {}
    if isinstance(configured, str):
        try:
            parsed = json.loads(configured)
        except json.JSONDecodeError:
            return {}
    elif isinstance(configured, dict):
        parsed = configured
    else:
        return {}
    return _normalize_artifact_type_policy(parsed)


def enterprise_artifact_action_policy() -> dict[str, dict[str, str]]:
    configured = getattr(_global_args(), "enterprise_artifact_action_policy", "")
    if not configured:
        return {}
    if isinstance(configured, str):
        try:
            parsed = json.loads(configured)
        except json.JSONDecodeError:
            return {}
    elif isinstance(configured, dict):
        parsed = configured
    else:
        return {}
    if not isinstance(parsed, dict):
        return {}
    policy: dict[str, dict[str, str]] = {}
    for action, action_policy in parsed.items():
        if not isinstance(action, str):
            continue
        normalized_action = action.strip()
        if normalized_action not in _ARTIFACT_POLICY_ACTIONS:
            continue
        normalized_policy = _normalize_artifact_type_policy(action_policy)
        if normalized_policy:
            policy[normalized_action] = normalized_policy
    return policy


def enterprise_artifact_min_role_for_type(
    artifact_type: str | None,
    *,
    action: str = "download",
) -> str:
    normalized_action = action.strip()
    normalized_type = (artifact_type or "").strip()
    action_policy = enterprise_artifact_action_policy().get(normalized_action, {})
    action_role = _artifact_policy_role_for_type(action_policy, normalized_type)
    if action_role is not None:
        return action_role

    policy = enterprise_artifact_download_policy()
    legacy_role = _artifact_policy_role_for_type(policy, normalized_type)
    if legacy_role is not None:
        return legacy_role
    if normalized_action == "preview":
        return KB_ROLE_VIEWER
    return enterprise_artifact_download_min_role()


def _normalize_artifact_type_policy(raw_policy: Any) -> dict[str, str]:
    if not isinstance(raw_policy, dict):
        return {}
    policy: dict[str, str] = {}
    for artifact_type, role in raw_policy.items():
        if not isinstance(artifact_type, str) or not artifact_type.strip():
            continue
        normalized = _normalize_kb_role(str(role or ""))
        if normalized is not None:
            policy[artifact_type.strip()] = normalized
    return policy


def _artifact_policy_role_for_type(
    policy: dict[str, str], artifact_type: str
) -> str | None:
    if artifact_type in policy:
        return policy[artifact_type]
    return policy.get("*")


def enterprise_mask_storage_uris() -> bool:
    return enterprise_auth_enabled() and bool(
        getattr(_global_args(), "enterprise_mask_storage_uris", True)
    )


def protected_whitelist_bypass_forbidden(path: str) -> bool:
    if not enterprise_auth_enabled():
        return False
    return path.startswith(_ENTERPRISE_PROTECTED_PREFIXES)


def principal_from_api_key() -> Principal:
    return Principal(
        user_id="api-key-super-admin",
        username="api-key-super-admin",
        system_role=SYSTEM_ROLE_SUPER_ADMIN,
        status=USER_STATUS_ACTIVE,
        tenant_id=None,
        tenant_roles={},
        can_create_kb=True,
        can_use_bypass_query=True,
        token_version=1,
        auth_method="api_key",
        metadata={"auth_mode": "enterprise", "api_key_superadmin": True},
        can_delete_documents=True,
    )


class AuditService:
    def __init__(self, metadata_store: EnterpriseMetadataStore):
        self._metadata_store = metadata_store

    async def append(
        self,
        event_type: str,
        *,
        actor_user_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEventRecord:
        event = AuditEventRecord(
            id=f"audit_{uuid4().hex}",
            event_type=event_type,
            actor_user_id=actor_user_id,
            target_type=target_type,
            target_id=target_id,
            metadata=metadata or {},
            created_at=utc_now_iso(),
        )
        return await self._metadata_store.append_audit_event(event)

    async def list(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        event_type: str | None = None,
        actor_user_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
    ) -> list[AuditEventRecord]:
        return await self._metadata_store.list_audit_events(
            limit=limit,
            offset=offset,
            event_type=event_type,
            actor_user_id=actor_user_id,
            target_type=target_type,
            target_id=target_id,
            created_after=created_after,
            created_before=created_before,
        )


@dataclass(frozen=True, slots=True)
class EnterpriseLimitConfig:
    enabled: bool
    rate_limit_requests: int
    rate_limit_window_seconds: float
    tenant_rate_limit_requests: int
    tenant_rate_limit_window_seconds: float
    quota_requests: int
    quota_window_seconds: float
    tenant_quota_requests: int
    tenant_quota_window_seconds: float

    @property
    def active(self) -> bool:
        return self.enabled and any(
            requests > 0
            for requests in (
                self.rate_limit_requests,
                self.tenant_rate_limit_requests,
                self.quota_requests,
                self.tenant_quota_requests,
            )
        )


@dataclass(frozen=True, slots=True)
class _LimitRule:
    name: str
    event_type: str
    subject_type: str
    subject_id: str
    requests: int
    window_seconds: float


@dataclass(slots=True)
class _LimitBucket:
    count: int
    reset_at: float


class EnterpriseLimitService:
    def __init__(
        self,
        audit_service: AuditService | None = None,
        *,
        time_func: Callable[[], float] = time.monotonic,
    ):
        self._audit_service = audit_service
        self._time_func = time_func
        self._buckets: dict[tuple[str, str, str], _LimitBucket] = {}

    async def enforce(self, request: Request, principal: Principal) -> None:
        config = _enterprise_limit_config()
        if not config.active:
            return

        rules = _limit_rules_for_principal(config, principal)
        if not rules:
            return

        now = self._time_func()
        self._prune_expired(now)
        increments: list[_LimitBucket] = []
        for rule in rules:
            bucket = self._bucket_for(rule, now)
            if bucket.count >= rule.requests:
                retry_after = max(1, int(bucket.reset_at - now))
                await self._audit_limit_exceeded(request, principal, rule, retry_after)
                detail = (
                    "Enterprise quota exceeded"
                    if rule.event_type == "quota_exceeded"
                    else "Enterprise rate limit exceeded"
                )
                raise HTTPException(
                    status_code=429,
                    detail=detail,
                    headers={"Retry-After": str(retry_after)},
                )
            increments.append(bucket)

        for bucket in increments:
            bucket.count += 1

    def _bucket_for(self, rule: _LimitRule, now: float) -> _LimitBucket:
        key = (rule.name, rule.subject_type, rule.subject_id)
        bucket = self._buckets.get(key)
        if bucket is None or now >= bucket.reset_at:
            bucket = _LimitBucket(count=0, reset_at=now + rule.window_seconds)
            self._buckets[key] = bucket
        return bucket

    def _prune_expired(self, now: float) -> None:
        if len(self._buckets) < 1000:
            return
        expired = [key for key, bucket in self._buckets.items() if now >= bucket.reset_at]
        for key in expired:
            self._buckets.pop(key, None)

    async def _audit_limit_exceeded(
        self,
        request: Request,
        principal: Principal,
        rule: _LimitRule,
        retry_after: int,
    ) -> None:
        if self._audit_service is None:
            return
        await self._audit_service.append(
            rule.event_type,
            actor_user_id=principal.user_id,
            target_type=rule.subject_type,
            target_id=rule.subject_id,
            metadata={
                "auth_method": principal.auth_method,
                "limit_name": rule.name,
                "limit": rule.requests,
                "window_seconds": rule.window_seconds,
                "method": request.method.upper(),
                "path": request.url.path,
                "retry_after_seconds": retry_after,
                "subject_type": rule.subject_type,
            },
        )


DEFAULT_LOGIN_MAX_ATTEMPTS = 10
DEFAULT_LOGIN_WINDOW_SECONDS = 300.0
DEFAULT_LOGIN_LOCKOUT_SECONDS = 900.0


@dataclass(slots=True)
class _LoginAttemptState:
    failures: int
    window_started_at: float
    locked_until: float


class LoginAttemptTracker:
    """In-memory failed-login lockout for enterprise ``/login``.

    Keyed by the submitted username (stripped). After ``max_attempts`` failures
    within ``window_seconds`` the username is locked for ``lockout_seconds`` and
    :meth:`check` raises HTTP 429. A successful login clears the counter.
    ``max_attempts <= 0`` disables the tracker entirely (no behavior change).

    Single-process/in-memory only — consistent with ``EnterpriseLimitService``;
    multi-instance shared lockout coordination is a later platform concern.
    """

    def __init__(
        self,
        *,
        max_attempts: int,
        window_seconds: float,
        lockout_seconds: float,
        time_func: Callable[[], float] = time.monotonic,
    ):
        self._max_attempts = max_attempts
        self._window_seconds = window_seconds
        self._lockout_seconds = lockout_seconds
        self._time_func = time_func
        self._states: dict[str, _LoginAttemptState] = {}

    @classmethod
    def from_args(cls, args: Any) -> "LoginAttemptTracker":
        return cls(
            max_attempts=_non_negative_int(
                getattr(args, "enterprise_login_max_attempts", DEFAULT_LOGIN_MAX_ATTEMPTS),
                DEFAULT_LOGIN_MAX_ATTEMPTS,
            ),
            window_seconds=_positive_float(
                getattr(
                    args,
                    "enterprise_login_window_seconds",
                    DEFAULT_LOGIN_WINDOW_SECONDS,
                ),
                DEFAULT_LOGIN_WINDOW_SECONDS,
            ),
            lockout_seconds=_positive_float(
                getattr(
                    args,
                    "enterprise_login_lockout_seconds",
                    DEFAULT_LOGIN_LOCKOUT_SECONDS,
                ),
                DEFAULT_LOGIN_LOCKOUT_SECONDS,
            ),
        )

    @property
    def enabled(self) -> bool:
        return self._max_attempts > 0

    def check(self, username: str) -> None:
        """Raise HTTP 429 when the username is currently locked out."""
        if not self.enabled:
            return
        state = self._states.get(self._key(username))
        if state is None:
            return
        now = self._time_func()
        if state.locked_until > now:
            retry_after = max(1, int(state.locked_until - now))
            raise HTTPException(
                status_code=429,
                detail="Too many failed login attempts. Try again later.",
                headers={"Retry-After": str(retry_after)},
            )

    def record_failure(self, username: str) -> bool:
        """Record a failed attempt; return True when it triggers a new lock."""
        if not self.enabled:
            return False
        key = self._key(username)
        now = self._time_func()
        state = self._states.get(key)
        if state is None or now - state.window_started_at >= self._window_seconds:
            state = _LoginAttemptState(
                failures=0, window_started_at=now, locked_until=0.0
            )
        state.failures += 1
        triggered = False
        if state.failures >= self._max_attempts:
            state.locked_until = now + self._lockout_seconds
            triggered = True
        self._states[key] = state
        return triggered

    def record_success(self, username: str) -> None:
        self._states.pop(self._key(username), None)

    @staticmethod
    def _key(username: str) -> str:
        return username.strip()


class ServiceAPIKeyService:
    def __init__(
        self,
        metadata_store: EnterpriseMetadataStore,
        audit_service: AuditService | None = None,
    ):
        self._metadata_store = metadata_store
        self._audit_service = audit_service

    async def create_key(
        self,
        *,
        name: str,
        scopes: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        created_by: str | None = None,
        tenant_id: str | None = None,
        expires_at: str | None = None,
    ) -> tuple[EnterpriseAPIKeyRecord, str]:
        normalized_name = name.strip()
        if not normalized_name:
            raise HTTPException(status_code=400, detail="Service API key name is required")
        normalized_scopes = _normalize_service_api_key_scopes(scopes or {})
        key_id = f"svc_key_{uuid4().hex}"
        raw_key = f"lrsk_{key_id}_{secrets.token_urlsafe(32)}"
        now = utc_now_iso()
        record = EnterpriseAPIKeyRecord(
            id=key_id,
            name=normalized_name,
            key_hash=_hash_service_api_key(raw_key),
            key_preview=raw_key[-6:],
            status=ENTERPRISE_API_KEY_STATUS_ACTIVE,
            created_by=created_by,
            tenant_id=tenant_id,
            scopes=normalized_scopes,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
            last_used_at=None,
            revoked_at=None,
            revoked_by=None,
            expires_at=expires_at,
        )
        saved = await self._metadata_store.create_enterprise_api_key(record)
        if self._audit_service is not None:
            await self._audit_service.append(
                "service_api_key_created",
                actor_user_id=created_by,
                target_type="service_api_key",
                target_id=saved.id,
                metadata={
                    "name": saved.name,
                    "key_preview": saved.key_preview,
                    "scopes": saved.scopes,
                    "expires_at": saved.expires_at,
                },
            )
        return saved, raw_key

    async def list_keys(self) -> list[EnterpriseAPIKeyRecord]:
        return await self._metadata_store.list_enterprise_api_keys()

    async def revoke_key(
        self, key_id: str, *, revoked_by: str | None = None
    ) -> EnterpriseAPIKeyRecord:
        revoked = await self._metadata_store.revoke_enterprise_api_key(
            key_id,
            revoked_by=revoked_by,
        )
        if revoked is None:
            raise HTTPException(status_code=404, detail="Service API key not found")
        if self._audit_service is not None:
            await self._audit_service.append(
                "service_api_key_revoked",
                actor_user_id=revoked_by,
                target_type="service_api_key",
                target_id=revoked.id,
                metadata={"name": revoked.name, "key_preview": revoked.key_preview},
            )
        return revoked

    async def rotate_key(
        self,
        key_id: str,
        *,
        rotated_by: str | None = None,
        expires_at: str | None = None,
        revoke_old: bool = True,
    ) -> tuple[EnterpriseAPIKeyRecord, str]:
        existing = await self._metadata_store.get_enterprise_api_key_by_id(key_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Service API key not found")
        if existing.status != ENTERPRISE_API_KEY_STATUS_ACTIVE:
            raise HTTPException(status_code=409, detail="Only active service API keys can be rotated")
        metadata = dict(existing.metadata or {})
        metadata["rotated_from"] = existing.id
        new_key, raw_key = await self.create_key(
            name=existing.name,
            scopes=existing.scopes,
            metadata=metadata,
            created_by=rotated_by,
            tenant_id=existing.tenant_id,
            expires_at=expires_at if expires_at is not None else existing.expires_at,
        )
        if revoke_old:
            await self.revoke_key(existing.id, revoked_by=rotated_by)
        if self._audit_service is not None:
            await self._audit_service.append(
                "service_api_key_rotated",
                actor_user_id=rotated_by,
                target_type="service_api_key",
                target_id=existing.id,
                metadata={
                    "new_key_id": new_key.id,
                    "new_key_preview": new_key.key_preview,
                    "old_key_preview": existing.key_preview,
                    "revoke_old": revoke_old,
                    "expires_at": new_key.expires_at,
                },
            )
        return new_key, raw_key

    async def principal_from_api_key(self, raw_key: str) -> Principal | None:
        if not raw_key.strip():
            return None
        key_hash = _hash_service_api_key(raw_key)
        record = await self._metadata_store.get_enterprise_api_key_by_hash(key_hash)
        if record is None or not secrets.compare_digest(record.key_hash, key_hash):
            return None
        if record.status != ENTERPRISE_API_KEY_STATUS_ACTIVE:
            return None
        if service_api_key_is_expired(record.expires_at):
            return None
        await self._metadata_store.mark_enterprise_api_key_used(record.id)
        scopes = _normalize_service_api_key_scopes(record.scopes)
        return Principal(
            user_id=f"service-key:{record.id}",
            username=record.name,
            system_role=SYSTEM_ROLE_USER,
            status=USER_STATUS_ACTIVE,
            tenant_id=record.tenant_id,
            tenant_roles={},
            can_create_kb=False,
            can_use_bypass_query=bool(scopes.get("can_use_bypass_query", False)),
            token_version=1,
            auth_method=SERVICE_API_KEY_AUTH_METHOD,
            metadata={
                "auth_mode": "enterprise",
                "service_api_key_id": record.id,
                "key_preview": record.key_preview,
                "scopes": scopes,
            },
            can_delete_documents=False,
        )


class UserKBQuerySettingsService:
    def __init__(
        self,
        metadata_store: EnterpriseMetadataStore,
        audit_service: AuditService | None = None,
    ):
        self._metadata_store = metadata_store
        self._audit_service = audit_service

    async def get_settings(
        self, user_id: str, kb_id: str
    ) -> EnterpriseUserKBQuerySettingsRecord | None:
        return await self._metadata_store.get_enterprise_user_kb_query_settings(
            user_id, kb_id
        )

    async def set_user_prompt(
        self,
        *,
        user_id: str,
        kb_id: str,
        user_prompt: str,
        actor_user_id: str | None = None,
    ) -> EnterpriseUserKBQuerySettingsRecord:
        existing = await self.get_settings(user_id, kb_id)
        now = utc_now_iso()
        record = EnterpriseUserKBQuerySettingsRecord(
            user_id=user_id,
            kb_id=kb_id,
            user_prompt=user_prompt,
            created_at=existing.created_at if existing is not None else now,
            updated_at=now,
        )
        saved = await self._metadata_store.upsert_enterprise_user_kb_query_settings(
            record
        )
        if self._audit_service is not None:
            await self._audit_service.append(
                "user_kb_query_settings_updated",
                actor_user_id=actor_user_id or user_id,
                target_type="kb",
                target_id=kb_id,
                metadata={
                    "user_id": user_id,
                    "has_user_prompt": bool(user_prompt),
                },
            )
        return saved

    async def clear_user_prompt(
        self,
        *,
        user_id: str,
        kb_id: str,
        actor_user_id: str | None = None,
    ) -> bool:
        deleted = await self._metadata_store.delete_enterprise_user_kb_query_settings(
            user_id, kb_id
        )
        if self._audit_service is not None:
            await self._audit_service.append(
                "user_kb_query_settings_updated",
                actor_user_id=actor_user_id or user_id,
                target_type="kb",
                target_id=kb_id,
                metadata={
                    "user_id": user_id,
                    "has_user_prompt": False,
                    "deleted": deleted,
                },
            )
        return deleted


class InvitationService:
    """Single-use registration invitations for ``invite_only`` mode.

    The raw token is returned once at creation; only a ``sha256:`` hash and a
    short preview are persisted. ``consume_invitation`` atomically transitions
    an active, non-expired invitation to ``used`` so a token cannot be reused.
    """

    def __init__(
        self,
        metadata_store: EnterpriseMetadataStore,
        audit_service: AuditService | None = None,
    ):
        self._metadata_store = metadata_store
        self._audit_service = audit_service

    async def create_invitation(
        self,
        *,
        created_by: str | None = None,
        expires_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[EnterpriseInvitationRecord, str]:
        invitation_id = f"inv_{uuid4().hex}"
        raw_token = f"lrinv_{invitation_id}_{secrets.token_urlsafe(32)}"
        now = utc_now_iso()
        record = EnterpriseInvitationRecord(
            id=invitation_id,
            token_hash=_hash_invitation_token(raw_token),
            token_preview=raw_token[-6:],
            status=ENTERPRISE_INVITATION_STATUS_ACTIVE,
            created_by=created_by,
            expires_at=expires_at,
            used_by=None,
            used_at=None,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )
        saved = await self._metadata_store.create_enterprise_invitation(record)
        if self._audit_service is not None:
            await self._audit_service.append(
                "invitation_created",
                actor_user_id=created_by,
                target_type="invitation",
                target_id=saved.id,
                metadata={
                    "token_preview": saved.token_preview,
                    "expires_at": saved.expires_at,
                },
            )
        return saved, raw_token

    async def list_invitations(self) -> list[EnterpriseInvitationRecord]:
        return await self._metadata_store.list_enterprise_invitations()

    async def revoke_invitation(
        self, invitation_id: str, *, actor_user_id: str | None = None
    ) -> EnterpriseInvitationRecord:
        revoked = await self._metadata_store.revoke_enterprise_invitation(invitation_id)
        if revoked is None:
            raise HTTPException(status_code=404, detail="Invitation not found")
        if self._audit_service is not None:
            await self._audit_service.append(
                "invitation_revoked",
                actor_user_id=actor_user_id,
                target_type="invitation",
                target_id=revoked.id,
            )
        return revoked

    async def consume_invitation(
        self, raw_token: str | None, *, used_by: str | None
    ) -> EnterpriseInvitationRecord:
        token = (raw_token or "").strip()
        if not token:
            raise HTTPException(
                status_code=403, detail="A valid invitation token is required"
            )
        record = await self._metadata_store.get_enterprise_invitation_by_token_hash(
            _hash_invitation_token(token)
        )
        if (
            record is None
            or record.status != ENTERPRISE_INVITATION_STATUS_ACTIVE
            or _iso_timestamp_is_past(record.expires_at)
        ):
            raise HTTPException(
                status_code=403, detail="Invalid or expired invitation token"
            )
        consumed = await self._metadata_store.consume_enterprise_invitation(
            record.token_hash, used_by=used_by
        )
        if consumed is None:
            raise HTTPException(
                status_code=403, detail="Invalid or expired invitation token"
            )
        if self._audit_service is not None:
            await self._audit_service.append(
                "invitation_consumed",
                actor_user_id=None,
                target_type="invitation",
                target_id=consumed.id,
                metadata={"used_by": used_by},
            )
        return consumed


class SystemSettingsService:
    def __init__(self, metadata_store: EnterpriseMetadataStore):
        self._metadata_store = metadata_store
    async def initialize_registration_setting(self, enabled: bool) -> None:
        existing = await self._metadata_store.get_enterprise_system_setting(
            ENTERPRISE_REGISTRATION_ENABLED_KEY
        )
        mode = await self._metadata_store.get_enterprise_system_setting(
            ENTERPRISE_REGISTRATION_MODE_KEY
        )
        if existing is None:
            await self.set_registration_enabled(enabled)
        elif mode is None:
            await self._metadata_store.set_enterprise_system_setting(
                ENTERPRISE_REGISTRATION_MODE_KEY,
                REGISTRATION_MODE_OPEN
                if str(existing).lower() == "true"
                else REGISTRATION_MODE_DISABLED,
            )

    async def registration_mode(self) -> str:
        value = await self._metadata_store.get_enterprise_system_setting(
            ENTERPRISE_REGISTRATION_MODE_KEY
        )
        if value is None:
            enabled = await self._metadata_store.get_enterprise_system_setting(
                ENTERPRISE_REGISTRATION_ENABLED_KEY,
                "false",
            )
            return (
                REGISTRATION_MODE_OPEN
                if str(enabled).lower() == "true"
                else REGISTRATION_MODE_DISABLED
            )
        return _normalize_registration_mode(value)

    async def registration_enabled(self) -> bool:
        return await self.registration_mode() == REGISTRATION_MODE_OPEN

    async def set_registration_enabled(
        self, enabled: bool, *, updated_by: str | None = None
    ) -> None:
        await self.set_registration_mode(
            REGISTRATION_MODE_OPEN if enabled else REGISTRATION_MODE_DISABLED,
            updated_by=updated_by,
        )

    async def set_registration_mode(
        self, mode: str, *, updated_by: str | None = None
    ) -> str:
        normalized = _normalize_registration_mode(mode)
        await self._metadata_store.set_enterprise_system_setting(
            ENTERPRISE_REGISTRATION_MODE_KEY,
            normalized,
            updated_by=updated_by,
        )
        await self._metadata_store.set_enterprise_system_setting(
            ENTERPRISE_REGISTRATION_ENABLED_KEY,
            "true" if normalized == REGISTRATION_MODE_OPEN else "false",
            updated_by=updated_by,
        )
        return normalized


class UserService:
    def __init__(
        self,
        metadata_store: EnterpriseMetadataStore,
        audit_service: AuditService | None = None,
    ):
        self._metadata_store = metadata_store
        self._audit_service = audit_service

    async def bootstrap_super_admin(
        self,
        *,
        username: str,
        password: str | None,
        password_hash: str | None,
    ) -> EnterpriseUserRecord:
        normalized_username = _normalize_username(username)
        existing = await self._metadata_store.get_enterprise_user_by_username(
            normalized_username
        )
        now = utc_now_iso()
        configured_hash = (password_hash or "").strip() or None
        if existing is None:
            if configured_hash is None:
                if not password:
                    raise ValueError("Super admin password is required for bootstrap")
                configured_hash = hash_password(password)
            created = EnterpriseUserRecord(
                id=f"usr_{uuid4().hex}",
                username=normalized_username,
                password_hash=configured_hash,
                system_role=SYSTEM_ROLE_SUPER_ADMIN,
                status=USER_STATUS_ACTIVE,
                tenant_id=None,
                can_create_kb=True,
                can_use_bypass_query=True,
                token_version=1,
                metadata={"bootstrap": True},
                created_at=now,
                updated_at=now,
                can_delete_documents=True,
            )
            user = await self._metadata_store.upsert_enterprise_user(created)
            await self._audit("super_admin_bootstrapped", actor_user_id=user.id)
            return user

        new_hash = configured_hash or existing.password_hash
        updated = replace(
            existing,
            password_hash=new_hash,
            system_role=SYSTEM_ROLE_SUPER_ADMIN,
            status=USER_STATUS_ACTIVE,
            can_create_kb=True,
            can_use_bypass_query=True,
            can_delete_documents=True,
            token_version=existing.token_version
            + (1 if new_hash != existing.password_hash else 0),
            updated_at=now,
        )
        if updated == existing:
            return existing
        user = await self._metadata_store.upsert_enterprise_user(updated)
        await self._audit("super_admin_synced", actor_user_id=user.id)
        return user

    async def authenticate(
        self, username: str, plain_password: str
    ) -> EnterpriseUserRecord | None:
        user = await self._metadata_store.get_enterprise_user_by_username(
            _normalize_username(username)
        )
        if user is None or user.status != USER_STATUS_ACTIVE:
            return None
        if not verify_password(plain_password, user.password_hash):
            return None
        return user

    async def create_user(
        self,
        *,
        username: str,
        password: str,
        created_by: str | None = None,
        can_create_kb: bool = False,
        can_use_bypass_query: bool = False,
        can_delete_documents: bool = False,
        tenant_id: str | None = None,
        status: str = USER_STATUS_ACTIVE,
    ) -> EnterpriseUserRecord:
        normalized_username = _normalize_username(username)
        if await self._metadata_store.get_enterprise_user_by_username(normalized_username):
            raise HTTPException(status_code=409, detail="Username already exists")
        now = utc_now_iso()
        user = EnterpriseUserRecord(
            id=f"usr_{uuid4().hex}",
            username=normalized_username,
            password_hash=hash_password(password),
            system_role=SYSTEM_ROLE_USER,
            status=status,
            tenant_id=tenant_id,
            can_create_kb=can_create_kb,
            can_use_bypass_query=can_use_bypass_query,
            token_version=1,
            metadata={},
            created_at=now,
            updated_at=now,
            can_delete_documents=can_delete_documents,
        )
        created = await self._metadata_store.upsert_enterprise_user(user)
        await self._audit(
            "user_created",
            actor_user_id=created_by,
            target_type="user",
            target_id=created.id,
        )
        return created

    async def list_users(self) -> list[EnterpriseUserRecord]:
        return await self._metadata_store.list_enterprise_users()

    async def get_user_or_404(self, user_id: str) -> EnterpriseUserRecord:
        user = await self._metadata_store.get_enterprise_user_by_id(user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        return user

    async def update_user(
        self,
        user_id: str,
        *,
        status_value: str | None = None,
        can_create_kb: bool | None = None,
        can_use_bypass_query: bool | None = None,
        can_delete_documents: bool | None = None,
        tenant_id: str | None = None,
        actor_user_id: str | None = None,
    ) -> EnterpriseUserRecord:
        user = await self.get_user_or_404(user_id)
        if status_value is not None and status_value not in USER_STATUS_VALUES:
            raise HTTPException(status_code=400, detail="Invalid user status")
        if user.system_role == SYSTEM_ROLE_SUPER_ADMIN and status_value == USER_STATUS_DISABLED:
            raise HTTPException(status_code=400, detail="Cannot disable a super admin")
        updated = replace(
            user,
            status=status_value or user.status,
            can_create_kb=user.can_create_kb
            if can_create_kb is None
            else can_create_kb,
            can_use_bypass_query=user.can_use_bypass_query
            if can_use_bypass_query is None
            else can_use_bypass_query,
            can_delete_documents=user.can_delete_documents
            if can_delete_documents is None
            else can_delete_documents,
            tenant_id=user.tenant_id if tenant_id is None else tenant_id,
            token_version=user.token_version + 1,
            updated_at=utc_now_iso(),
        )
        saved = await self._metadata_store.upsert_enterprise_user(updated)
        await self._audit(
            "user_updated",
            actor_user_id=actor_user_id,
            target_type="user",
            target_id=saved.id,
        )
        return saved

    async def change_password(
        self, user_id: str, password: str, *, actor_user_id: str | None = None
    ) -> EnterpriseUserRecord:
        user = await self.get_user_or_404(user_id)
        updated = replace(
            user,
            password_hash=hash_password(password),
            token_version=user.token_version + 1,
            updated_at=utc_now_iso(),
        )
        saved = await self._metadata_store.upsert_enterprise_user(updated)
        await self._audit(
            "user_password_changed",
            actor_user_id=actor_user_id,
            target_type="user",
            target_id=saved.id,
        )
        return saved

    async def principal_from_token_info(self, token_info: dict[str, Any]) -> Principal:
        metadata = token_info.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        user_id = metadata.get("user_id")
        user = None
        if isinstance(user_id, str) and user_id:
            user = await self._metadata_store.get_enterprise_user_by_id(user_id)
        if user is None:
            user = await self._metadata_store.get_enterprise_user_by_username(
                str(token_info.get("username", ""))
            )
        if user is None or user.status != USER_STATUS_ACTIVE:
            raise HTTPException(status_code=401, detail="Enterprise user is not active")
        token_version = metadata.get("token_version")
        if token_version != user.token_version:
            raise HTTPException(status_code=401, detail="Token has been revoked")
        memberships = await self._metadata_store.list_user_tenant_memberships(user.id)
        return principal_from_user(user, auth_method="jwt", memberships=memberships)

    def token_metadata_for_user(self, user: EnterpriseUserRecord) -> dict[str, Any]:
        return {
            "auth_mode": "enterprise",
            "user_id": user.id,
            "tenant_id": user.tenant_id,
            "system_role": user.system_role,
            "token_version": user.token_version,
            "can_create_kb": user.can_create_kb,
            "can_use_bypass_query": user.can_use_bypass_query,
        }

    async def _audit(
        self,
        event_type: str,
        *,
        actor_user_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
    ) -> None:
        if self._audit_service is not None:
            await self._audit_service.append(
                event_type,
                actor_user_id=actor_user_id,
                target_type=target_type,
                target_id=target_id,
            )


class AuthorizationService:
    def __init__(
        self,
        metadata_store: EnterpriseMetadataStore,
        audit_service: AuditService | None = None,
    ):
        self._metadata_store = metadata_store
        self._audit_service = audit_service

    def require_super_admin(self, principal: Principal | None) -> Principal:
        principal = _require_principal(principal)
        if not principal.is_super_admin:
            raise HTTPException(status_code=403, detail="Super admin permission required")
        return principal

    def require_create_kb(self, principal: Principal | None) -> Principal:
        principal = _require_principal(principal)
        if principal.auth_method == SERVICE_API_KEY_AUTH_METHOD:
            raise HTTPException(status_code=403, detail="Create-KB permission required")
        if not (principal.is_super_admin or principal.can_create_kb):
            raise HTTPException(status_code=403, detail="Create-KB permission required")
        return principal

    def require_bypass_query(self, principal: Principal | None) -> Principal:
        principal = _require_principal(principal)
        if not (principal.is_super_admin or principal.can_use_bypass_query):
            raise HTTPException(status_code=403, detail="Bypass-query permission required")
        return principal

    async def require_kb_role(
        self, principal: Principal | None, kb_id: str, minimum_role: str
    ) -> Principal:
        principal = _require_principal(principal)
        if principal.is_super_admin:
            return principal
        if principal.auth_method == SERVICE_API_KEY_AUTH_METHOD:
            role = await self._effective_service_api_key_kb_role(principal, kb_id)
        else:
            role = await self._effective_user_kb_role(principal, kb_id)
        if role is None or _KB_ROLE_RANK.get(role, 0) < _KB_ROLE_RANK[minimum_role]:
            await self._audit_denied(principal, kb_id, minimum_role)
            raise HTTPException(status_code=403, detail="Knowledge-base access denied")
        return principal

    async def authorize_document_delete(
        self,
        principal: Principal | None,
        kb_id: str,
        *,
        document_owner_id: str | None,
    ) -> str:
        """Authorize a document deletion and return the delete scope.

        The central request middleware already enforced ``kb_editor`` on the
        delete route, so the caller has write access to the KB. This refines
        that into ownership/capability semantics and returns the scope so the
        route can audit it:

          * ``"privileged"`` — may delete ANY document (super admin, effective
            ``kb_admin``+, or the ``can_delete_documents`` capability).
          * ``"self"``       — a ``kb_editor`` deleting a document they uploaded
            (``document_owner_id`` matches the caller).

        Anything else raises HTTP 403. The effective role is computed directly
        (not via :meth:`require_kb_role`) so the allowed ``self``/``privileged``
        outcomes never emit a misleading ``permission_denied`` audit — only the
        genuine denial path audits.
        """
        principal = _require_principal(principal)
        if principal.is_super_admin:
            return "privileged"
        # User-level capability grants delete-any; service keys never carry it.
        if (
            principal.auth_method != SERVICE_API_KEY_AUTH_METHOD
            and principal.can_delete_documents
        ):
            return "privileged"
        if principal.auth_method == SERVICE_API_KEY_AUTH_METHOD:
            role = await self._effective_service_api_key_kb_role(principal, kb_id)
        else:
            role = await self._effective_user_kb_role(principal, kb_id)
        rank = _KB_ROLE_RANK.get(role, 0) if role else 0
        if rank >= _KB_ROLE_RANK[KB_ROLE_ADMIN]:
            return "privileged"
        if rank >= _KB_ROLE_RANK[KB_ROLE_EDITOR]:
            if (
                document_owner_id is not None
                and document_owner_id == principal.user_id
            ):
                return "self"
        await self._audit_denied(principal, kb_id, KB_ROLE_ADMIN)
        raise HTTPException(status_code=403, detail="Document delete denied")

    async def create_tenant(
        self,
        *,
        name: str,
        description: str | None = None,
        tenant_id: str | None = None,
        created_by: str | None = None,
    ) -> EnterpriseTenantRecord:
        normalized_name = name.strip()
        if not normalized_name:
            raise HTTPException(status_code=400, detail="Tenant name is required")
        tid = (tenant_id or f"tenant_{uuid4().hex[:12]}").strip()
        if not tid:
            raise HTTPException(status_code=400, detail="Tenant id cannot be empty")
        if await self._metadata_store.get_enterprise_tenant_by_id(tid) is not None:
            raise HTTPException(status_code=409, detail="Tenant already exists")
        now = utc_now_iso()
        tenant = EnterpriseTenantRecord(
            id=tid,
            name=normalized_name,
            description=description,
            status=USER_STATUS_ACTIVE,
            metadata={},
            created_by=created_by,
            created_at=now,
            updated_at=now,
        )
        saved = await self._metadata_store.upsert_enterprise_tenant(tenant)
        if self._audit_service is not None:
            await self._audit_service.append(
                "tenant_created",
                actor_user_id=created_by,
                target_type="tenant",
                target_id=saved.id,
                metadata={"name": saved.name},
            )
        return saved

    async def list_tenants(self) -> list[EnterpriseTenantRecord]:
        return await self._metadata_store.list_enterprise_tenants()

    async def get_tenant_or_404(self, tenant_id: str) -> EnterpriseTenantRecord:
        tenant = await self._metadata_store.get_enterprise_tenant_by_id(
            _normalize_required_id(tenant_id, "Tenant id")
        )
        if tenant is None:
            raise HTTPException(status_code=404, detail="Tenant not found")
        return tenant

    async def update_tenant(
        self,
        tenant_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        status_value: str | None = None,
        actor_user_id: str | None = None,
    ) -> EnterpriseTenantRecord:
        tenant = await self.get_tenant_or_404(tenant_id)
        if status_value is not None and status_value not in {
            USER_STATUS_ACTIVE,
            USER_STATUS_DISABLED,
        }:
            raise HTTPException(status_code=400, detail="Invalid tenant status")
        if name is not None and not name.strip():
            raise HTTPException(status_code=400, detail="Tenant name cannot be empty")
        updated = replace(
            tenant,
            name=name.strip() if name is not None else tenant.name,
            description=tenant.description if description is None else description,
            status=status_value or tenant.status,
            updated_at=utc_now_iso(),
        )
        saved = await self._metadata_store.upsert_enterprise_tenant(updated)
        if self._audit_service is not None:
            await self._audit_service.append(
                "tenant_updated",
                actor_user_id=actor_user_id,
                target_type="tenant",
                target_id=saved.id,
            )
        return saved

    async def delete_tenant(
        self, tenant_id: str, *, actor_user_id: str | None = None
    ) -> bool:
        tenant_id = _normalize_required_id(tenant_id, "Tenant id")
        deleted = await self._metadata_store.delete_enterprise_tenant(tenant_id)
        if deleted and self._audit_service is not None:
            await self._audit_service.append(
                "tenant_deleted",
                actor_user_id=actor_user_id,
                target_type="tenant",
                target_id=tenant_id,
            )
        return deleted

    async def list_kb_ids_for_tenants(self, tenant_ids: list[str]) -> list[str]:
        return await self._metadata_store.list_kb_ids_for_tenants(tenant_ids)

    async def require_tenant_role(
        self, principal: Principal | None, tenant_id: str, minimum_role: str
    ) -> Principal:
        principal = _require_principal(principal)
        tenant_id = _normalize_required_id(tenant_id, "Tenant id")
        minimum_role = _normalize_tenant_role(minimum_role)
        if principal.is_super_admin:
            return principal
        if principal.auth_method == SERVICE_API_KEY_AUTH_METHOD:
            await self._audit_tenant_denied(principal, tenant_id, minimum_role)
            raise HTTPException(status_code=403, detail="Tenant access denied")
        role = principal.tenant_roles.get(tenant_id)
        if _tenant_role_rank(role) < _tenant_role_rank(minimum_role):
            await self._audit_tenant_denied(principal, tenant_id, minimum_role)
            raise HTTPException(status_code=403, detail="Tenant access denied")
        return principal

    async def filter_kbs_for_principal(
        self, principal: Principal | None, records: list[KnowledgeBaseRecord]
    ) -> list[KnowledgeBaseRecord]:
        principal = _require_principal(principal)
        if principal.is_super_admin:
            return records
        if principal.auth_method == SERVICE_API_KEY_AUTH_METHOD:
            allowed_ids = _service_api_key_kb_ids(principal)
            if _service_api_key_inherits_tenant_kb_acl(principal) and principal.tenant_id:
                allowed_ids.update(
                    await self._metadata_store.list_kb_ids_for_tenants(
                        [principal.tenant_id]
                    )
                )
        else:
            allowed_ids = set(await self._metadata_store.list_kb_ids_for_user(principal.user_id))
            tenant_ids = list(principal.tenant_roles)
            allowed_ids.update(await self._metadata_store.list_kb_ids_for_tenants(tenant_ids))
        return [record for record in records if record.id in allowed_ids]

    async def grant_kb_role(
        self,
        kb_id: str,
        user_id: str,
        role: str,
        *,
        granted_by: str | None = None,
    ) -> KBACLRecord:
        normalized_role = _normalize_kb_role(role)
        if normalized_role not in _KB_ROLE_RANK:
            raise HTTPException(status_code=400, detail="Invalid KB ACL role")
        user = await self._metadata_store.get_enterprise_user_by_id(user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        now = utc_now_iso()
        acl = KBACLRecord(
            kb_id=kb_id,
            user_id=user_id,
            role=normalized_role,
            granted_by=granted_by,
            created_at=now,
            updated_at=now,
        )
        saved = await self._metadata_store.upsert_kb_acl(acl)
        if self._audit_service is not None:
            await self._audit_service.append(
                "kb_acl_granted",
                actor_user_id=granted_by,
                target_type="kb",
                target_id=kb_id,
                metadata={"user_id": user_id, "role": normalized_role},
            )
        return saved

    async def revoke_kb_role(
        self, kb_id: str, user_id: str, *, actor_user_id: str | None = None
    ) -> bool:
        deleted = await self._metadata_store.delete_kb_acl(kb_id, user_id)
        if deleted and self._audit_service is not None:
            await self._audit_service.append(
                "kb_acl_revoked",
                actor_user_id=actor_user_id,
                target_type="kb",
                target_id=kb_id,
                metadata={"user_id": user_id},
            )
        return deleted

    async def list_kb_acl(self, kb_id: str) -> list[KBACLRecord]:
        return await self._metadata_store.list_kb_acl(kb_id)

    async def grant_tenant_membership(
        self,
        tenant_id: str,
        user_id: str,
        role: str,
        *,
        granted_by: str | None = None,
    ) -> EnterpriseTenantMembershipRecord:
        tenant_id = _normalize_required_id(tenant_id, "Tenant id")
        normalized_role = _normalize_tenant_role(role)
        user = await self._metadata_store.get_enterprise_user_by_id(user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        now = utc_now_iso()
        membership = EnterpriseTenantMembershipRecord(
            tenant_id=tenant_id,
            user_id=user_id,
            role=normalized_role,
            granted_by=granted_by,
            created_at=now,
            updated_at=now,
        )
        saved = await self._metadata_store.upsert_tenant_membership(membership)
        if self._audit_service is not None:
            await self._audit_service.append(
                "tenant_membership_granted",
                actor_user_id=granted_by,
                target_type="tenant",
                target_id=tenant_id,
                metadata={"user_id": user_id, "role": normalized_role},
            )
        return saved

    async def revoke_tenant_membership(
        self, tenant_id: str, user_id: str, *, actor_user_id: str | None = None
    ) -> bool:
        tenant_id = _normalize_required_id(tenant_id, "Tenant id")
        deleted = await self._metadata_store.delete_tenant_membership(tenant_id, user_id)
        if deleted and self._audit_service is not None:
            await self._audit_service.append(
                "tenant_membership_revoked",
                actor_user_id=actor_user_id,
                target_type="tenant",
                target_id=tenant_id,
                metadata={"user_id": user_id},
            )
        return deleted

    async def list_tenant_memberships(
        self, tenant_id: str
    ) -> list[EnterpriseTenantMembershipRecord]:
        tenant_id = _normalize_required_id(tenant_id, "Tenant id")
        return await self._metadata_store.list_tenant_memberships(tenant_id)

    async def get_tenant_membership(
        self, tenant_id: str, user_id: str
    ) -> EnterpriseTenantMembershipRecord | None:
        tenant_id = _normalize_required_id(tenant_id, "Tenant id")
        return await self._metadata_store.get_tenant_membership(tenant_id, user_id)

    async def grant_tenant_kb_role(
        self,
        kb_id: str,
        tenant_id: str,
        role: str,
        *,
        granted_by: str | None = None,
    ) -> EnterpriseTenantKBACLRecord:
        tenant_id = _normalize_required_id(tenant_id, "Tenant id")
        normalized_role = _normalize_kb_role(role)
        if normalized_role not in _KB_ROLE_RANK:
            raise HTTPException(status_code=400, detail="Invalid KB ACL role")
        now = utc_now_iso()
        acl = EnterpriseTenantKBACLRecord(
            tenant_id=tenant_id,
            kb_id=kb_id,
            role=normalized_role,
            granted_by=granted_by,
            created_at=now,
            updated_at=now,
        )
        saved = await self._metadata_store.upsert_tenant_kb_acl(acl)
        if self._audit_service is not None:
            await self._audit_service.append(
                "tenant_kb_acl_granted",
                actor_user_id=granted_by,
                target_type="kb",
                target_id=kb_id,
                metadata={"tenant_id": tenant_id, "role": normalized_role},
            )
        return saved

    async def revoke_tenant_kb_role(
        self, kb_id: str, tenant_id: str, *, actor_user_id: str | None = None
    ) -> bool:
        tenant_id = _normalize_required_id(tenant_id, "Tenant id")
        deleted = await self._metadata_store.delete_tenant_kb_acl(tenant_id, kb_id)
        if deleted and self._audit_service is not None:
            await self._audit_service.append(
                "tenant_kb_acl_revoked",
                actor_user_id=actor_user_id,
                target_type="kb",
                target_id=kb_id,
                metadata={"tenant_id": tenant_id},
            )
        return deleted

    async def list_kb_tenant_acl(self, kb_id: str) -> list[EnterpriseTenantKBACLRecord]:
        return await self._metadata_store.list_kb_tenant_acl(kb_id)

    async def _effective_user_kb_role(self, principal: Principal, kb_id: str) -> str | None:
        roles: list[str] = []
        direct_role = _normalize_kb_role(
            await self._metadata_store.get_kb_acl_role(kb_id, principal.user_id)
        )
        if direct_role is not None:
            roles.append(direct_role)
        for tenant_id in principal.tenant_roles:
            tenant_role = _normalize_kb_role(
                await self._metadata_store.get_tenant_kb_acl_role(tenant_id, kb_id)
            )
            if tenant_role is not None:
                roles.append(tenant_role)
        if not roles:
            return None
        return max(roles, key=lambda item: _KB_ROLE_RANK.get(item, 0))

    async def _effective_service_api_key_kb_role(
        self, principal: Principal, kb_id: str
    ) -> str | None:
        roles: list[str] = []
        explicit_role = _service_api_key_kb_role(principal, kb_id)
        if explicit_role is not None:
            roles.append(explicit_role)
        if _service_api_key_inherits_tenant_kb_acl(principal) and principal.tenant_id:
            tenant_role = _normalize_kb_role(
                await self._metadata_store.get_tenant_kb_acl_role(
                    principal.tenant_id,
                    kb_id,
                )
            )
            if tenant_role is not None:
                roles.append(tenant_role)
        if not roles:
            return None
        return max(roles, key=lambda item: _KB_ROLE_RANK.get(item, 0))

    async def _audit_denied(
        self, principal: Principal, kb_id: str, minimum_role: str
    ) -> None:
        if self._audit_service is not None:
            await self._audit_service.append(
                "permission_denied",
                actor_user_id=principal.user_id,
                target_type="kb",
                target_id=kb_id,
                metadata={"minimum_role": minimum_role},
            )

    async def _audit_tenant_denied(
        self, principal: Principal, tenant_id: str, minimum_role: str
    ) -> None:
        if self._audit_service is not None:
            await self._audit_service.append(
                "permission_denied",
                actor_user_id=principal.user_id,
                target_type="tenant",
                target_id=tenant_id,
                metadata={"minimum_role": minimum_role},
            )


def principal_from_user(
    user: EnterpriseUserRecord,
    *,
    auth_method: str,
    memberships: list[EnterpriseTenantMembershipRecord] | None = None,
) -> Principal:
    tenant_roles = {
        membership.tenant_id: membership.role
        for membership in memberships or []
    }
    return Principal(
        user_id=user.id,
        username=user.username,
        system_role=user.system_role,
        status=user.status,
        tenant_id=user.tenant_id,
        tenant_roles=tenant_roles,
        can_create_kb=user.can_create_kb,
        can_use_bypass_query=user.can_use_bypass_query,
        token_version=user.token_version,
        auth_method=auth_method,
        metadata=dict(user.metadata),
        can_delete_documents=user.can_delete_documents,
    )


def get_request_principal(request: Request) -> Principal | None:
    value = getattr(request.state, "principal", None)
    return value if isinstance(value, Principal) else None


def get_enterprise_user_service(request: Request) -> UserService:
    service = getattr(request.app.state, "enterprise_user_service", None)
    if not isinstance(service, UserService):
        raise HTTPException(status_code=500, detail="Enterprise user service unavailable")
    return service


def get_enterprise_settings_service(request: Request) -> SystemSettingsService:
    service = getattr(request.app.state, "enterprise_settings_service", None)
    if not isinstance(service, SystemSettingsService):
        raise HTTPException(status_code=500, detail="Enterprise settings service unavailable")
    return service


def get_enterprise_api_key_service(request: Request) -> ServiceAPIKeyService:
    service = getattr(request.app.state, "enterprise_api_key_service", None)
    if not isinstance(service, ServiceAPIKeyService):
        raise HTTPException(status_code=500, detail="Enterprise API key service unavailable")
    return service


def get_enterprise_user_kb_query_settings_service(
    request: Request,
) -> UserKBQuerySettingsService:
    service = getattr(request.app.state, "enterprise_user_kb_query_settings_service", None)
    if not isinstance(service, UserKBQuerySettingsService):
        raise HTTPException(
            status_code=500,
            detail="Enterprise user KB query settings service unavailable",
        )
    return service


def get_enterprise_invitation_service(request: Request) -> InvitationService:
    service = getattr(request.app.state, "enterprise_invitation_service", None)
    if not isinstance(service, InvitationService):
        raise HTTPException(
            status_code=500, detail="Enterprise invitation service unavailable"
        )
    return service


def get_enterprise_authorization_service(request: Request) -> AuthorizationService:
    service = getattr(request.app.state, "enterprise_authorization_service", None)
    if not isinstance(service, AuthorizationService):
        raise HTTPException(status_code=500, detail="Enterprise authorization service unavailable")
    return service


def get_enterprise_audit_service(request: Request) -> AuditService:
    service = getattr(request.app.state, "enterprise_audit_service", None)
    if not isinstance(service, AuditService):
        raise HTTPException(status_code=500, detail="Enterprise audit service unavailable")
    return service


async def append_enterprise_audit_event(
    request: Request,
    event_type: str,
    *,
    target_type: str | None = None,
    target_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    if not enterprise_auth_enabled():
        return
    principal = get_request_principal(request)
    await get_enterprise_audit_service(request).append(
        event_type,
        actor_user_id=principal.user_id if principal is not None else None,
        target_type=target_type,
        target_id=target_id,
        metadata=metadata,
    )


async def enforce_enterprise_request_limits(
    request: Request, principal: Principal | None
) -> None:
    if not enterprise_auth_enabled():
        return
    principal = _require_principal(principal)
    service = getattr(request.app.state, "enterprise_limit_service", None)
    if isinstance(service, EnterpriseLimitService):
        await service.enforce(request, principal)


async def enforce_enterprise_request_access(
    request: Request, principal: Principal | None
) -> None:
    if not enterprise_auth_enabled():
        return
    principal = _require_principal(principal)
    path = request.url.path
    method = request.method.upper()
    authz = get_enterprise_authorization_service(request)

    if path.startswith(_LEGACY_GLOBAL_PREFIXES):
        if enterprise_global_routes_disabled():
            raise HTTPException(status_code=403, detail="Legacy global route disabled in enterprise mode")
        authz.require_super_admin(principal)
        return

    if path.startswith("/admin"):
        authz.require_super_admin(principal)
        return

    kb_id = _extract_kb_id(path)
    if kb_id is None:
        if path == "/kbs" and method == "POST":
            authz.require_create_kb(principal)
        return

    if method == "DELETE" and path.rstrip("/") == f"/kbs/{kb_id}":
        authz.require_super_admin(principal)
        return

    if method == "PATCH" and path.rstrip("/") == f"/kbs/{kb_id}":
        await authz.require_kb_role(principal, kb_id, KB_ROLE_ADMIN)
        return

    if "/query" in path or path.endswith("/retrieve"):
        await authz.require_kb_role(principal, kb_id, KB_ROLE_VIEWER)
        if await _request_uses_bypass_mode(request):
            authz.require_bypass_query(principal)
        return

    if "/graph" in path or path.endswith("/status"):
        await authz.require_kb_role(principal, kb_id, KB_ROLE_VIEWER)
        return

    if "/configs" in path:
        minimum = KB_ROLE_VIEWER if method == "GET" else KB_ROLE_ADMIN
        await authz.require_kb_role(principal, kb_id, minimum)
        return

    if _is_artifact_download_action(path):
        # Artifact-type policy needs the artifact metadata, so the path-level
        # guard only establishes KB read access. The route performs the
        # stricter per-type check before restore/presign work.
        await authz.require_kb_role(principal, kb_id, KB_ROLE_VIEWER)
        return

    if method == "GET":
        await authz.require_kb_role(principal, kb_id, KB_ROLE_VIEWER)
        return

    await authz.require_kb_role(principal, kb_id, KB_ROLE_EDITOR)


def _require_principal(principal: Principal | None) -> Principal:
    if principal is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Login required")
    if principal.status != USER_STATUS_ACTIVE:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User is disabled")
    return principal


def _normalize_kb_role(role: str | None) -> str | None:
    if role is None:
        return None
    normalized = role.strip()
    if normalized in _KB_ROLE_RANK:
        return normalized
    return _KB_ROLE_ALIASES.get(normalized)


def _normalize_tenant_role(role: str) -> str:
    normalized = _canonical_tenant_role(role)
    if normalized is None:
        raise HTTPException(status_code=400, detail="Invalid tenant role")
    return normalized


def _canonical_tenant_role(role: str | None) -> str | None:
    if role is None:
        return None
    normalized = role.strip()
    if not normalized:
        return None
    return _TENANT_ROLE_ALIASES.get(normalized, normalized)


def _tenant_role_rank(role: str | None) -> int:
    normalized = _canonical_tenant_role(role)
    if normalized is None:
        return 0
    return _TENANT_ROLE_RANK.get(normalized, 0)


def _normalize_required_id(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise HTTPException(status_code=400, detail=f"{label} is required")
    return normalized


def _hash_service_api_key(raw_key: str) -> str:
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _hash_invitation_token(raw_token: str) -> str:
    digest = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _iso_timestamp_is_past(value: str | None, *, now: datetime | None = None) -> bool:
    """Return True when an ISO-8601 timestamp is at or before ``now``.

    A missing/empty value is never past; unparseable values are treated as
    not-past so a corrupt field cannot revoke an otherwise valid credential.
    """
    if not value:
        return False
    try:
        deadline = datetime.fromisoformat(value)
    except ValueError:
        return False
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)
    return (now or datetime.now(timezone.utc)) >= deadline


def service_api_key_is_expired(
    expires_at: str | None, *, now: datetime | None = None
) -> bool:
    """Return True when a service API key ``expires_at`` is at or past now.

    ``expires_at`` is an ISO-8601 timestamp produced by ``utc_now_iso``; a
    missing/empty value means the key never expires.
    """
    return _iso_timestamp_is_past(expires_at, now=now)


def _normalize_service_api_key_scopes(scopes: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(scopes, dict):
        raise HTTPException(status_code=400, detail="Service API key scopes must be an object")
    if bool(scopes.get("can_create_kb", False)):
        raise HTTPException(status_code=400, detail="Service API keys cannot create knowledge bases")
    raw_kb_roles = scopes.get("kb_roles", {})
    if raw_kb_roles is None:
        raw_kb_roles = {}
    if not isinstance(raw_kb_roles, dict):
        raise HTTPException(status_code=400, detail="Service API key kb_roles must be an object")
    kb_roles: dict[str, str] = {}
    for kb_id, role in raw_kb_roles.items():
        if not isinstance(kb_id, str) or not kb_id.strip():
            raise HTTPException(status_code=400, detail="Service API key KB id is invalid")
        if not isinstance(role, str):
            raise HTTPException(status_code=400, detail="Service API key KB role is invalid")
        normalized_role = _normalize_kb_role(role)
        if normalized_role not in _KB_ROLE_RANK:
            raise HTTPException(status_code=400, detail="Service API key KB role is invalid")
        kb_roles[kb_id.strip()] = normalized_role
    return {
        "kb_roles": kb_roles,
        "can_use_bypass_query": bool(scopes.get("can_use_bypass_query", False)),
        "inherit_tenant_kb_acl": bool(scopes.get("inherit_tenant_kb_acl", False)),
    }


def _service_api_key_scopes(principal: Principal) -> dict[str, Any]:
    if principal.auth_method != SERVICE_API_KEY_AUTH_METHOD:
        return {}
    scopes = principal.metadata.get("scopes", {})
    return _normalize_service_api_key_scopes(scopes if isinstance(scopes, dict) else {})


def _service_api_key_kb_role(principal: Principal, kb_id: str) -> str | None:
    scopes = _service_api_key_scopes(principal)
    kb_roles = scopes.get("kb_roles", {})
    if not isinstance(kb_roles, dict):
        return None
    role = kb_roles.get(kb_id)
    return _normalize_kb_role(role) if isinstance(role, str) else None


def _service_api_key_inherits_tenant_kb_acl(principal: Principal) -> bool:
    scopes = _service_api_key_scopes(principal)
    return bool(scopes.get("inherit_tenant_kb_acl", False))


def _service_api_key_kb_ids(principal: Principal) -> set[str]:
    scopes = _service_api_key_scopes(principal)
    kb_roles = scopes.get("kb_roles", {})
    if not isinstance(kb_roles, dict):
        return set()
    return {kb_id for kb_id, role in kb_roles.items() if _normalize_kb_role(role)}


def _enterprise_limit_config() -> EnterpriseLimitConfig:
    args = _global_args()
    return EnterpriseLimitConfig(
        enabled=bool(getattr(args, "enterprise_rate_limit_enabled", False)),
        rate_limit_requests=_non_negative_int(
            getattr(args, "enterprise_rate_limit_requests", 60), 60
        ),
        rate_limit_window_seconds=_positive_float(
            getattr(args, "enterprise_rate_limit_window_seconds", 60.0), 60.0
        ),
        tenant_rate_limit_requests=_non_negative_int(
            getattr(args, "enterprise_tenant_rate_limit_requests", 0), 0
        ),
        tenant_rate_limit_window_seconds=_positive_float(
            getattr(args, "enterprise_tenant_rate_limit_window_seconds", 60.0), 60.0
        ),
        quota_requests=_non_negative_int(
            getattr(args, "enterprise_quota_requests", 0), 0
        ),
        quota_window_seconds=_positive_float(
            getattr(args, "enterprise_quota_window_seconds", 86400.0), 86400.0
        ),
        tenant_quota_requests=_non_negative_int(
            getattr(args, "enterprise_tenant_quota_requests", 0), 0
        ),
        tenant_quota_window_seconds=_positive_float(
            getattr(args, "enterprise_tenant_quota_window_seconds", 86400.0), 86400.0
        ),
    )


def _limit_rules_for_principal(
    config: EnterpriseLimitConfig, principal: Principal
) -> list[_LimitRule]:
    subject_type, subject_id = _principal_limit_subject(principal)
    rules: list[_LimitRule] = []
    if config.rate_limit_requests > 0:
        rules.append(
            _LimitRule(
                name="principal_rate",
                event_type="rate_limited",
                subject_type=subject_type,
                subject_id=subject_id,
                requests=config.rate_limit_requests,
                window_seconds=config.rate_limit_window_seconds,
            )
        )
    if config.quota_requests > 0:
        rules.append(
            _LimitRule(
                name="principal_quota",
                event_type="quota_exceeded",
                subject_type=subject_type,
                subject_id=subject_id,
                requests=config.quota_requests,
                window_seconds=config.quota_window_seconds,
            )
        )
    if principal.tenant_id:
        if config.tenant_rate_limit_requests > 0:
            rules.append(
                _LimitRule(
                    name="tenant_rate",
                    event_type="rate_limited",
                    subject_type="tenant",
                    subject_id=principal.tenant_id,
                    requests=config.tenant_rate_limit_requests,
                    window_seconds=config.tenant_rate_limit_window_seconds,
                )
            )
        if config.tenant_quota_requests > 0:
            rules.append(
                _LimitRule(
                    name="tenant_quota",
                    event_type="quota_exceeded",
                    subject_type="tenant",
                    subject_id=principal.tenant_id,
                    requests=config.tenant_quota_requests,
                    window_seconds=config.tenant_quota_window_seconds,
                )
            )
    return rules


def _principal_limit_subject(principal: Principal) -> tuple[str, str]:
    if principal.auth_method == SERVICE_API_KEY_AUTH_METHOD:
        key_id = principal.metadata.get("service_api_key_id")
        return "service_api_key", str(key_id or principal.user_id)
    if principal.auth_method == "api_key":
        return "api_key", principal.user_id
    return "user", principal.user_id


def principal_job_subject(principal: Principal) -> dict[str, str | None]:
    """The attribution stamp written into a job payload as ``_principal``."""
    _subject_type, subject_id = _principal_limit_subject(principal)
    return {"subject_id": subject_id, "tenant_id": principal.tenant_id}


def _enterprise_concurrent_job_limits() -> tuple[int, int]:
    args = _global_args()
    return (
        _non_negative_int(getattr(args, "enterprise_max_concurrent_jobs", 0), 0),
        _non_negative_int(
            getattr(args, "enterprise_tenant_max_concurrent_jobs", 0), 0
        ),
    )


async def enforce_concurrent_job_quota(
    metadata_store: Any, principal: Principal | None
) -> None:
    """Reject (HTTP 429) new job creation once a principal/tenant already holds
    the configured number of in-flight jobs.

    No-op unless enterprise auth is on and a positive limit is configured
    (default 0 = disabled). Attribution is read from the job payload stamp
    written by :func:`principal_job_subject`.
    """
    if not enterprise_auth_enabled() or principal is None:
        return
    principal_limit, tenant_limit = _enterprise_concurrent_job_limits()
    if principal_limit <= 0 and tenant_limit <= 0:
        return
    _subject_type, subject_id = _principal_limit_subject(principal)
    if principal_limit > 0:
        active = await metadata_store.count_active_jobs_for_principal(subject_id)
        if active >= principal_limit:
            await _audit_job_quota(
                metadata_store, principal, "principal", principal_limit, active
            )
            raise HTTPException(
                status_code=429,
                detail="Concurrent job quota exceeded",
                headers={"Retry-After": "30"},
            )
    if tenant_limit > 0 and principal.tenant_id:
        active = await metadata_store.count_active_jobs_for_tenant(principal.tenant_id)
        if active >= tenant_limit:
            await _audit_job_quota(
                metadata_store, principal, "tenant", tenant_limit, active
            )
            raise HTTPException(
                status_code=429,
                detail="Concurrent job quota exceeded",
                headers={"Retry-After": "30"},
            )


async def _audit_job_quota(
    metadata_store: Any,
    principal: Principal,
    subject_type: str,
    limit: int,
    active: int,
) -> None:
    try:
        await AuditService(metadata_store).append(
            "quota_exceeded",
            actor_user_id=principal.user_id,
            target_type=subject_type,
            metadata={
                "limit_name": "concurrent_jobs",
                "limit": limit,
                "active": active,
                "subject_type": subject_type,
                "auth_method": principal.auth_method,
            },
        )
    except Exception:
        pass


def _non_negative_int(value: Any, default: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _global_args() -> Any:
    config_module = import_module("lightrag.api.config")
    current_args = config_module.global_args
    initialized = bool(getattr(config_module, "_initialized", False))
    if initialized or current_args.__class__.__name__ != "_GlobalArgsProxy":
        return current_args
    return SimpleNamespace(
        enterprise_auth_enabled=_env_bool("LIGHTRAG_ENTERPRISE_AUTH_ENABLED", False),
        enterprise_disable_global_routes=_env_bool(
            "LIGHTRAG_ENTERPRISE_DISABLE_GLOBAL_ROUTES", True
        ),
        enterprise_legacy_api_key_superadmin=_env_bool(
            "LIGHTRAG_ENTERPRISE_LEGACY_API_KEY_SUPERADMIN", False
        ),
        enterprise_artifact_download_min_role=os.getenv(
            "LIGHTRAG_ENTERPRISE_ARTIFACT_DOWNLOAD_MIN_ROLE", KB_ROLE_VIEWER
        ),
        enterprise_artifact_download_policy=os.getenv(
            "LIGHTRAG_ENTERPRISE_ARTIFACT_DOWNLOAD_POLICY", ""
        ),
        enterprise_mask_storage_uris=_env_bool(
            "LIGHTRAG_ENTERPRISE_MASK_STORAGE_URIS", True
        ),
        enterprise_rate_limit_enabled=_env_bool(
            "LIGHTRAG_ENTERPRISE_RATE_LIMIT_ENABLED", False
        ),
        enterprise_rate_limit_requests=_env_int(
            "LIGHTRAG_ENTERPRISE_RATE_LIMIT_REQUESTS", 60
        ),
        enterprise_rate_limit_window_seconds=_env_float(
            "LIGHTRAG_ENTERPRISE_RATE_LIMIT_WINDOW_SECONDS", 60.0
        ),
        enterprise_tenant_rate_limit_requests=_env_int(
            "LIGHTRAG_ENTERPRISE_TENANT_RATE_LIMIT_REQUESTS", 0
        ),
        enterprise_tenant_rate_limit_window_seconds=_env_float(
            "LIGHTRAG_ENTERPRISE_TENANT_RATE_LIMIT_WINDOW_SECONDS", 60.0
        ),
        enterprise_quota_requests=_env_int("LIGHTRAG_ENTERPRISE_QUOTA_REQUESTS", 0),
        enterprise_quota_window_seconds=_env_float(
            "LIGHTRAG_ENTERPRISE_QUOTA_WINDOW_SECONDS", 86400.0
        ),
        enterprise_tenant_quota_requests=_env_int(
            "LIGHTRAG_ENTERPRISE_TENANT_QUOTA_REQUESTS", 0
        ),
        enterprise_tenant_quota_window_seconds=_env_float(
            "LIGHTRAG_ENTERPRISE_TENANT_QUOTA_WINDOW_SECONDS", 86400.0
        ),
    )


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _ENV_TRUE_VALUES:
        return True
    if normalized in _ENV_FALSE_VALUES:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _normalize_username(username: str) -> str:
    normalized = username.strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="Username cannot be empty")
    return normalized


def _normalize_registration_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in REGISTRATION_MODES:
        raise HTTPException(status_code=400, detail="Invalid registration mode")
    return normalized


def _extract_kb_id(path: str) -> str | None:
    parts = [part for part in path.split("/") if part]
    if len(parts) >= 2 and parts[0] == "kbs":
        return parts[1]
    return None


def _is_artifact_download_action(path: str) -> bool:
    return path.endswith(":download") or path.endswith(":download-url")


async def _request_uses_bypass_mode(request: Request) -> bool:
    if request.method.upper() != "POST":
        return False
    try:
        body = await request.json()
    except Exception:
        return False
    return isinstance(body, dict) and body.get("mode") == "bypass"
