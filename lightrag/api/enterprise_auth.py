from __future__ import annotations

from dataclasses import dataclass, replace
from importlib import import_module
import os
from types import SimpleNamespace
from typing import Any, Protocol
from uuid import uuid4

from fastapi import HTTPException, Request, status

from lightrag.api.kb_service import KnowledgeBaseRecord, utc_now_iso
from lightrag.api.metadata_store import (
    AuditEventRecord,
    EnterpriseUserRecord,
    KBACLRecord,
)
from lightrag.api.passwords import hash_password, verify_password

ENTERPRISE_REGISTRATION_ENABLED_KEY = "registration_enabled"
USER_STATUS_ACTIVE = "active"
USER_STATUS_DISABLED = "disabled"
SYSTEM_ROLE_SUPER_ADMIN = "super_admin"
SYSTEM_ROLE_USER = "user"
KB_ROLE_VIEWER = "kb_viewer"
KB_ROLE_EDITOR = "kb_editor"
KB_ROLE_ADMIN = "kb_admin"
KB_ROLE_OWNER = "kb_owner"

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

    async def append_audit_event(
        self, event: AuditEventRecord
    ) -> AuditEventRecord: ...

    async def list_audit_events(self, *, limit: int = 100) -> list[AuditEventRecord]: ...


@dataclass(frozen=True, slots=True)
class Principal:
    user_id: str
    username: str
    system_role: str
    status: str
    tenant_id: str | None
    can_create_kb: bool
    can_use_bypass_query: bool
    token_version: int
    auth_method: str
    metadata: dict[str, Any]

    @property
    def is_super_admin(self) -> bool:
        return self.system_role == SYSTEM_ROLE_SUPER_ADMIN


def enterprise_auth_enabled() -> bool:
    return bool(getattr(_global_args(), "enterprise_auth_enabled", False))


def enterprise_legacy_api_key_superadmin_enabled() -> bool:
    return bool(getattr(_global_args(), "enterprise_legacy_api_key_superadmin", False))


def enterprise_global_routes_disabled() -> bool:
    return bool(getattr(_global_args(), "enterprise_disable_global_routes", True))


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
        can_create_kb=True,
        can_use_bypass_query=True,
        token_version=1,
        auth_method="api_key",
        metadata={"auth_mode": "enterprise", "api_key_superadmin": True},
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

    async def list(self, *, limit: int = 100) -> list[AuditEventRecord]:
        return await self._metadata_store.list_audit_events(limit=limit)


class SystemSettingsService:
    def __init__(self, metadata_store: EnterpriseMetadataStore):
        self._metadata_store = metadata_store

    async def initialize_registration_setting(self, enabled: bool) -> None:
        existing = await self._metadata_store.get_enterprise_system_setting(
            ENTERPRISE_REGISTRATION_ENABLED_KEY
        )
        if existing is None:
            await self.set_registration_enabled(enabled)

    async def registration_enabled(self) -> bool:
        value = await self._metadata_store.get_enterprise_system_setting(
            ENTERPRISE_REGISTRATION_ENABLED_KEY,
            "false",
        )
        return str(value).lower() == "true"

    async def set_registration_enabled(
        self, enabled: bool, *, updated_by: str | None = None
    ) -> None:
        await self._metadata_store.set_enterprise_system_setting(
            ENTERPRISE_REGISTRATION_ENABLED_KEY,
            "true" if enabled else "false",
            updated_by=updated_by,
        )


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
        tenant_id: str | None = None,
        actor_user_id: str | None = None,
    ) -> EnterpriseUserRecord:
        user = await self.get_user_or_404(user_id)
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
        return principal_from_user(user, auth_method="jwt")

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
        role = _normalize_kb_role(
            await self._metadata_store.get_kb_acl_role(kb_id, principal.user_id)
        )
        if role is None or _KB_ROLE_RANK.get(role, 0) < _KB_ROLE_RANK[minimum_role]:
            await self._audit_denied(principal, kb_id, minimum_role)
            raise HTTPException(status_code=403, detail="Knowledge-base access denied")
        return principal

    async def filter_kbs_for_principal(
        self, principal: Principal | None, records: list[KnowledgeBaseRecord]
    ) -> list[KnowledgeBaseRecord]:
        principal = _require_principal(principal)
        if principal.is_super_admin:
            return records
        allowed_ids = set(await self._metadata_store.list_kb_ids_for_user(principal.user_id))
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


def principal_from_user(user: EnterpriseUserRecord, *, auth_method: str) -> Principal:
    return Principal(
        user_id=user.id,
        username=user.username,
        system_role=user.system_role,
        status=user.status,
        tenant_id=user.tenant_id,
        can_create_kb=user.can_create_kb,
        can_use_bypass_query=user.can_use_bypass_query,
        token_version=user.token_version,
        auth_method=auth_method,
        metadata=dict(user.metadata),
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
    return _KB_ROLE_ALIASES.get(normalized, normalized)


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


def _normalize_username(username: str) -> str:
    normalized = username.strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="Username cannot be empty")
    return normalized


def _extract_kb_id(path: str) -> str | None:
    parts = [part for part in path.split("/") if part]
    if len(parts) >= 2 and parts[0] == "kbs":
        return parts[1]
    return None


async def _request_uses_bypass_mode(request: Request) -> bool:
    if request.method.upper() != "POST":
        return False
    try:
        body = await request.json()
    except Exception:
        return False
    return isinstance(body, dict) and body.get("mode") == "bypass"
