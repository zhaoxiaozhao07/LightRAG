from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import contextvars
import hashlib
import inspect
import json
from importlib import import_module
import os
import secrets
import time
from types import SimpleNamespace
from typing import Any, Callable, Protocol
from uuid import uuid4

from fastapi import HTTPException, Request, status

from lightrag.api.kb_service import (
    is_tenant_owned_kb,
    KnowledgeBaseNotFoundError,
    KnowledgeBaseRecord,
    utc_now_iso,
)
from lightrag.api.metadata_store import (
    AuditEventRecord,
    ChatMessageRecord,
    ChatMemoryOutboxEventRecord,
    ChatMemoryOutboxStats,
    ChatProjectRecord,
    ChatSessionRecord,
    EnterpriseAPIKeyRecord,
    EnterpriseInvitationRecord,
    EnterprisePersonAccountLinkRecord,
    EnterprisePersonCredentialRecord,
    EnterprisePersonEnrollmentGrantRecord,
    EnterprisePersonLoginSessionRecord,
    EnterprisePersonRecord,
    EnterpriseUserKBQuerySettingsRecord,
    EnterpriseUserRecord,
    KBACLRecord,
    EnterpriseTenantKBACLRecord,
    EnterpriseTenantMembershipRecord,
    EnterpriseTenantUserKBOverrideRecord,
    EnterpriseTenantRecord,
    KBLifecycleConflictError,
    KBLifecycleRecord,
    MetadataConflictError,
    MetadataRecordNotFoundError,
    TenantUserKBOverrideEffect,
)
from lightrag.api.passwords import hash_password, verify_password
from lightrag.utils import logger

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
# v2 person access JWTs build an account-scoped Principal with this
# auth_method so audit can tell the entry path apart. Every surface that gates
# on "interactive user" must treat it exactly like a legacy login JWT — use
# INTERACTIVE_AUTH_METHODS instead of comparing against "jwt" directly.
PERSON_JWT_AUTH_METHOD = "person_jwt"
INTERACTIVE_AUTH_METHODS = frozenset({"jwt", PERSON_JWT_AUTH_METHOD})
SYSTEM_ROLE_SUPER_ADMIN = "super_admin"
SYSTEM_ROLE_USER = "user"
KB_ROLE_VIEWER = "kb_viewer"
KB_ROLE_EDITOR = "kb_editor"
KB_ROLE_ADMIN = "kb_admin"
KB_ROLE_OWNER = "kb_owner"
TENANT_ROLE_MEMBER = "tenant_member"
TENANT_ROLE_ADMIN = "tenant_admin"
TENANT_ROLE_OWNER = "tenant_owner"
KB_VISIBILITY_INTERNAL = "internal"
KB_VISIBILITY_PUBLIC = "public"

# Sentinel for update calls that need to distinguish "field omitted" from an
# explicit ``None`` (mirrors the ``_UNSET`` idiom in kb_service updates).
UNSET: Any = object()
_ACTOR_TENANT_UNSET: Any = object()

_KB_ROLE_RANK = {
    KB_ROLE_VIEWER: 1,
    KB_ROLE_EDITOR: 2,
    KB_ROLE_ADMIN: 3,
    KB_ROLE_OWNER: 4,
}
# Roles a tenant admin/owner may receive as the oversight floor on
# tenant-owned KBs. kb_owner is intentionally excluded: ownership is
# platform-granted and would unlock visibility changes over members'
# private KBs.
_TENANT_ADMIN_OVERSIGHT_ROLES = frozenset(
    {KB_ROLE_VIEWER, KB_ROLE_EDITOR, KB_ROLE_ADMIN}
)
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
    "/agent",
    "/graph",
    "/api",
    "/chat",
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
        self,
        user: EnterpriseUserRecord,
        *,
        expected_updated_at: str | None = None,
        expected_token_version: int | None = None,
        expected_tenant_id: str | None = None,
    ) -> EnterpriseUserRecord: ...

    async def upsert_enterprise_user_with_membership(
        self,
        user: EnterpriseUserRecord,
        membership: EnterpriseTenantMembershipRecord | None,
        *,
        expected_updated_at: str | None = None,
        expected_token_version: int | None = None,
        expected_tenant_id: str | None = None,
        expected_membership: Any = UNSET,
    ) -> tuple[EnterpriseUserRecord, EnterpriseTenantMembershipRecord | None]: ...

    async def get_enterprise_user_kb_query_settings(
        self, user_id: str, kb_id: str
    ) -> EnterpriseUserKBQuerySettingsRecord | None: ...

    async def upsert_enterprise_user_kb_query_settings(
        self, record: EnterpriseUserKBQuerySettingsRecord
    ) -> EnterpriseUserKBQuerySettingsRecord: ...

    async def delete_enterprise_user_kb_query_settings(
        self, user_id: str, kb_id: str
    ) -> bool: ...

    async def create_chat_project(
        self, record: ChatProjectRecord
    ) -> ChatProjectRecord: ...

    async def get_chat_project(
        self, user_id: str, project_id: str
    ) -> ChatProjectRecord | None: ...

    async def list_chat_projects(
        self, user_id: str, *, limit: int = 50, offset: int = 0
    ) -> tuple[list[ChatProjectRecord], int]: ...

    async def rename_chat_project(
        self, user_id: str, project_id: str, *, name: str
    ) -> ChatProjectRecord | None: ...

    async def delete_chat_project(
        self, user_id: str, project_id: str
    ) -> tuple[bool, int, int]: ...

    async def delete_chat_project_with_memory(
        self,
        user_id: str,
        project_id: str,
        *,
        config_fingerprint: str,
        graph_store_fingerprint: str,
        actor_user_id: str | None = None,
        actor_tenant_id: str | None = None,
    ) -> tuple[bool, int, int]: ...

    async def create_chat_session(
        self, record: ChatSessionRecord
    ) -> ChatSessionRecord: ...

    async def get_chat_session(
        self, user_id: str, project_id: str, session_id: str
    ) -> ChatSessionRecord | None: ...

    async def list_chat_sessions(
        self, user_id: str, project_id: str, *, limit: int = 50, offset: int = 0
    ) -> tuple[list[ChatSessionRecord], int]: ...

    async def update_chat_session(
        self,
        user_id: str,
        project_id: str,
        session_id: str,
        *,
        name: str | None = None,
        context_rounds: int | None = None,
    ) -> ChatSessionRecord | None: ...

    async def delete_chat_session(
        self, user_id: str, project_id: str, session_id: str
    ) -> tuple[bool, int]: ...

    async def delete_chat_session_with_memory(
        self,
        user_id: str,
        project_id: str,
        session_id: str,
        *,
        config_fingerprint: str,
        graph_store_fingerprint: str,
        actor_user_id: str | None = None,
        actor_tenant_id: str | None = None,
    ) -> tuple[bool, int]: ...

    async def append_chat_messages(
        self, records: Sequence[ChatMessageRecord]
    ) -> list[ChatMessageRecord]: ...

    async def append_chat_messages_with_memory(
        self,
        records: Sequence[ChatMessageRecord],
        *,
        config_fingerprint: str,
        graph_store_fingerprint: str,
        actor_user_id: str | None = None,
        actor_tenant_id: str | None = None,
    ) -> list[ChatMessageRecord]: ...

    async def list_chat_messages(
        self,
        user_id: str,
        project_id: str,
        session_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[ChatMessageRecord], int]: ...

    async def get_chat_message(
        self,
        user_id: str,
        project_id: str,
        session_id: str,
        message_id: str,
    ) -> ChatMessageRecord | None: ...

    async def delete_chat_message(
        self, user_id: str, project_id: str, session_id: str, message_id: str
    ) -> bool: ...

    async def delete_chat_message_with_memory(
        self,
        user_id: str,
        project_id: str,
        session_id: str,
        message_id: str,
        *,
        config_fingerprint: str,
        graph_store_fingerprint: str,
        actor_user_id: str | None = None,
        actor_tenant_id: str | None = None,
    ) -> bool: ...

    async def enqueue_chat_memory_purge(
        self,
        user_id: str,
        project_id: str,
        config_fingerprint: str,
        *,
        graph_store_fingerprint: str,
        actor_user_id: str | None = None,
        actor_tenant_id: str | None = None,
    ) -> ChatMemoryOutboxEventRecord | None: ...

    async def get_chat_memory_event(
        self, event_id: str
    ) -> ChatMemoryOutboxEventRecord | None: ...

    async def requeue_chat_memory_purge(
        self,
        event_id: str,
        runtime_fingerprint: str,
        *,
        runtime_graph_store_fingerprint: str | None = None,
        retry_delay_seconds: float = 5.0,
    ) -> ChatMemoryOutboxEventRecord: ...

    async def get_chat_memory_outbox_stats(self) -> ChatMemoryOutboxStats: ...

    async def create_enterprise_api_key(
        self,
        record: EnterpriseAPIKeyRecord,
        *,
        expected_kb_generations: dict[str, str] | None = None,
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

    async def get_kb_lifecycle(self, kb_id: str) -> KBLifecycleRecord | None: ...

    def kb_write_guard(
        self, kb_id: str, expected_generation: str | None
    ) -> Any: ...

    async def register_kb_generation(
        self,
        kb_id: str,
        generation: str,
        *,
        activated_at: str | None = None,
    ) -> KBLifecycleRecord: ...

    async def assert_current_kb_generation(
        self, kb_id: str, expected_generation: str | None
    ) -> KBLifecycleRecord | None: ...

    async def upsert_kb_acl(
        self, acl: KBACLRecord, *, expected_generation: str | None = None
    ) -> KBACLRecord: ...

    async def delete_kb_acl(
        self,
        kb_id: str,
        user_id: str,
        *,
        expected_generation: str | None = None,
    ) -> bool: ...

    async def delete_enterprise_user(
        self,
        user_id: str,
        *,
        expected_updated_at: str | None = None,
        expected_token_version: int | None = None,
        expected_tenant_id: str | None = None,
        expected_membership: Any = UNSET,
    ) -> bool: ...

    async def delete_enterprise_user_with_memory(
        self,
        user_id: str,
        *,
        config_fingerprint: str,
        graph_store_fingerprint: str,
        actor_user_id: str | None = None,
        actor_tenant_id: str | None = None,
        expected_updated_at: str | None = None,
        expected_token_version: int | None = None,
        expected_tenant_id: str | None = None,
        expected_membership: Any = UNSET,
    ) -> bool: ...

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
        self,
        acl: EnterpriseTenantKBACLRecord,
        *,
        expected_generation: str | None = None,
    ) -> EnterpriseTenantKBACLRecord: ...

    async def delete_tenant_kb_acl(
        self,
        tenant_id: str,
        kb_id: str,
        *,
        expected_generation: str | None = None,
    ) -> bool: ...

    async def list_kb_tenant_acl(
        self, kb_id: str
    ) -> list[EnterpriseTenantKBACLRecord]: ...

    async def get_tenant_kb_acl_role(self, tenant_id: str, kb_id: str) -> str | None: ...

    async def list_kb_ids_for_tenants(self, tenant_ids: list[str]) -> list[str]: ...

    async def get_tenant_user_kb_override(
        self, tenant_id: str, kb_id: str, user_id: str
    ) -> EnterpriseTenantUserKBOverrideRecord | None: ...

    async def upsert_tenant_user_kb_override(
        self,
        record: EnterpriseTenantUserKBOverrideRecord,
        *,
        expected_generation: str | None = None,
        expected_user: Any = UNSET,
        expected_membership: Any = UNSET,
    ) -> EnterpriseTenantUserKBOverrideRecord: ...

    async def delete_tenant_user_kb_override(
        self,
        tenant_id: str,
        kb_id: str,
        user_id: str,
        *,
        granted_by: str | None = None,
        updated_at: str | None = None,
        expected_generation: str | None = None,
        expected_user: Any = UNSET,
        expected_membership: Any = UNSET,
    ) -> EnterpriseTenantUserKBOverrideRecord: ...

    async def reset_tenant_user_kb_override(
        self,
        tenant_id: str,
        kb_id: str,
        user_id: str,
        *,
        expected_generation: str | None = None,
        expected_user: Any = UNSET,
        expected_membership: Any = UNSET,
    ) -> bool: ...

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
        actor_tenant_id: str | None = None,
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

    # ------------------------------------------------------------------
    # Multi-account person identity store.
    #
    # Simple reads/writes plus aggregate atomic methods. The atomic methods
    # perform state reads, CAS, multi-table writes and an in-transaction
    # audit-row insert inside a single write transaction; they never nest
    # another write transaction nor call the public ``AuditService.append``.
    # See docs/多账号身份关联与切换执行文档.md sections 4 and 7.2.
    # ------------------------------------------------------------------

    async def get_person_by_id(
        self, person_id: str
    ) -> EnterprisePersonRecord | None: ...

    async def list_person_account_links(
        self, person_id: str, *, only_active: bool = False
    ) -> list[EnterprisePersonAccountLinkRecord]: ...

    async def get_person_account_link(
        self, person_id: str, account_id: str
    ) -> EnterprisePersonAccountLinkRecord | None: ...

    async def get_active_person_link_for_account(
        self, account_id: str
    ) -> EnterprisePersonAccountLinkRecord | None: ...

    async def get_person_credential(
        self, person_id: str
    ) -> EnterprisePersonCredentialRecord | None: ...

    async def record_person_credential_failure_atomic(
        self,
        credential_id: str,
        *,
        max_attempts: int,
        lockout_seconds: float,
        now: str | None = None,
    ) -> EnterprisePersonCredentialRecord: ...

    async def reset_person_credential_failures_atomic(
        self,
        credential_id: str,
        *,
        now: str | None = None,
    ) -> None: ...

    async def get_person_login_session(
        self, session_id: str
    ) -> EnterprisePersonLoginSessionRecord | None: ...

    async def list_person_login_sessions(
        self, person_id: str, *, only_active: bool = False
    ) -> list[EnterprisePersonLoginSessionRecord]: ...

    async def get_person_enrollment_grant_by_token_hash(
        self, token_hash: str
    ) -> EnterprisePersonEnrollmentGrantRecord | None: ...

    async def get_person_enrollment_grant(
        self, grant_id: str
    ) -> EnterprisePersonEnrollmentGrantRecord | None: ...

    async def create_person_enrollment_grant_atomic(
        self,
        grant: EnterprisePersonEnrollmentGrantRecord,
        *,
        actor_user_id: str | None = None,
    ) -> EnterprisePersonEnrollmentGrantRecord: ...

    async def revoke_person_enrollment_grant_atomic(
        self,
        grant_id: str,
        *,
        actor_user_id: str | None = None,
        revoked_at: str | None = None,
        reason: str | None = None,
    ) -> EnterprisePersonEnrollmentGrantRecord | None: ...

    async def consume_enrollment_grant_atomic(
        self,
        token_hash: str,
        *,
        person_id: str,
        actor_user_id: str | None = None,
        consumed_at: str | None = None,
    ) -> EnterprisePersonEnrollmentGrantRecord: ...

    async def enroll_person_atomic(
        self,
        *,
        grant_token_hash: str,
        person: EnterprisePersonRecord,
        credential: EnterprisePersonCredentialRecord,
        link: EnterprisePersonAccountLinkRecord,
        session: EnterprisePersonLoginSessionRecord,
        actor_user_id: str | None = None,
    ) -> tuple[
        EnterprisePersonRecord,
        EnterprisePersonCredentialRecord,
        EnterprisePersonAccountLinkRecord,
        EnterprisePersonLoginSessionRecord,
    ]: ...

    async def create_person_session_atomic(
        self,
        session: EnterprisePersonLoginSessionRecord,
        *,
        expected_person_epoch: int,
        actor_user_id: str | None = None,
    ) -> EnterprisePersonLoginSessionRecord: ...

    async def switch_person_session_atomic(
        self,
        *,
        session_id: str,
        expected_session_epoch: int,
        target_account_id: str,
        actor_user_id: str | None = None,
        switched_at: str | None = None,
    ) -> EnterprisePersonLoginSessionRecord: ...

    async def rotate_person_credential_atomic(
        self,
        *,
        person_id: str,
        new_credential: EnterprisePersonCredentialRecord,
        actor_user_id: str | None = None,
    ) -> tuple[EnterprisePersonRecord, EnterprisePersonCredentialRecord]: ...

    async def disable_person_atomic(
        self,
        *,
        person_id: str,
        actor_user_id: str | None = None,
        reason: str | None = None,
        disabled_at: str | None = None,
    ) -> EnterprisePersonRecord: ...

    async def enable_person_atomic(
        self,
        *,
        person_id: str,
        actor_user_id: str | None = None,
        enabled_at: str | None = None,
    ) -> EnterprisePersonRecord: ...

    async def propose_person_account_link_atomic(
        self,
        link: EnterprisePersonAccountLinkRecord,
        *,
        actor_user_id: str | None = None,
    ) -> EnterprisePersonAccountLinkRecord: ...

    async def confirm_person_account_link_atomic(
        self,
        *,
        person_id: str,
        account_id: str,
        actor_user_id: str | None = None,
        confirmed_at: str | None = None,
    ) -> tuple[EnterprisePersonRecord, EnterprisePersonAccountLinkRecord]: ...

    async def revoke_person_account_link_atomic(
        self,
        *,
        person_id: str,
        account_id: str,
        actor_user_id: str | None = None,
        revoked_at: str | None = None,
        reason: str | None = None,
    ) -> tuple[EnterprisePersonAccountLinkRecord, int]: ...

    async def revoke_person_session_atomic(
        self,
        session_id: str,
        *,
        actor_user_id: str | None = None,
        revoked_at: str | None = None,
    ) -> EnterprisePersonLoginSessionRecord | None: ...

    async def revoke_all_person_sessions_atomic(
        self,
        person_id: str,
        *,
        actor_user_id: str | None = None,
        revoked_at: str | None = None,
    ) -> tuple[EnterprisePersonRecord, int]: ...


def _user_write_conflict(exc: MetadataConflictError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="User was modified concurrently; retry the request",
    )


def chat_memory_write_conflict(exc: MetadataConflictError) -> HTTPException:
    """Map durable Chat Memory metadata conflicts to a stable HTTP contract."""

    error_code = str(exc.current.get("error_code") or "chat_memory_conflict")
    if error_code == "graph_store_migration_required":
        message = "Chat memory graph store migration is required"
    else:
        error_code = "chat_memory_conflict"
        message = "Chat memory state changed concurrently; retry the request"
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={"error_code": error_code, "message": message},
    )


async def _run_post_commit_nudge(
    callback: Callable[[], Any] | None,
    *,
    owner: str,
) -> None:
    """Wake the durable worker without changing an already-committed result."""

    if callback is None:
        return
    try:
        result = callback()
        if inspect.isawaitable(result):
            await result
    except Exception as exc:  # noqa: BLE001 - commit already succeeded
        logger.warning("%s Chat Memory worker nudge failed: %s", owner, exc)


def _kb_lifecycle_write_conflict(exc: KBLifecycleConflictError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="Knowledge base changed concurrently; retry the request",
    )


def _tenant_override_target_write_conflict(
    exc: MetadataConflictError,
) -> HTTPException:
    if (
        exc.entity_type == "tenant_user_kb_override_target"
        and exc.current.get("eligible") is False
    ):
        return HTTPException(status_code=404, detail="User not found")
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="User was modified concurrently; retry the request",
    )


async def _save_user_cas(
    metadata_store: EnterpriseMetadataStore,
    candidate: EnterpriseUserRecord,
    snapshot: EnterpriseUserRecord,
) -> EnterpriseUserRecord:
    try:
        return await metadata_store.upsert_enterprise_user(
            candidate,
            expected_updated_at=snapshot.updated_at,
            expected_token_version=snapshot.token_version,
            expected_tenant_id=snapshot.tenant_id,
        )
    except MetadataConflictError as exc:
        raise _user_write_conflict(exc) from exc


async def _save_user_with_membership_cas(
    metadata_store: EnterpriseMetadataStore,
    candidate: EnterpriseUserRecord,
    membership: EnterpriseTenantMembershipRecord | None,
    snapshot: EnterpriseUserRecord,
    *,
    expected_membership: Any = UNSET,
) -> tuple[EnterpriseUserRecord, EnterpriseTenantMembershipRecord | None]:
    try:
        kwargs: dict[str, Any] = {
            "expected_updated_at": snapshot.updated_at,
            "expected_token_version": snapshot.token_version,
            "expected_tenant_id": snapshot.tenant_id,
        }
        if expected_membership is not UNSET:
            kwargs["expected_membership"] = expected_membership
        return await metadata_store.upsert_enterprise_user_with_membership(
            candidate,
            membership,
            **kwargs,
        )
    except MetadataConflictError as exc:
        raise _user_write_conflict(exc) from exc


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
    can_use_agent_query: bool = False
    # Capability to delete documents uploaded by other users (delete-any).
    # Declared last with a default so existing keyword constructions and the
    # frozen-dataclass field ordering stay valid.
    can_delete_documents: bool = False
    # Interactive users need this capability before source/derived artifacts
    # can leave the server through a download surface. Service keys retain
    # their role/scope semantics and super administrators always pass.
    can_download_files: bool = False

    @property
    def is_super_admin(self) -> bool:
        return self.system_role == SYSTEM_ROLE_SUPER_ADMIN


@dataclass(frozen=True, slots=True)
class KBAccessDecision:
    """Source-aware result of resolving one principal against one KB.

    ``platform_role`` contains only platform-authoritative sources (a direct
    user grant and visibility). ``tenant_role`` is the contribution from the
    principal's one canonical tenant after applying its tenant ACL and optional
    per-user allow/deny override. Keeping those values separate is important:
    a tenant deny must never hide a direct platform grant or public/internal
    visibility.
    """

    kb_id: str
    generation: str | None
    effective_role: str | None
    platform_role: str | None
    direct_role: str | None
    visibility_role: str | None
    tenant_role: str | None
    tenant_id: str | None
    tenant_acl_role: str | None
    tenant_override_effect: str | None
    tenant_override_role: str | None
    tenant_owned: bool
    sources: tuple[str, ...] = ()

    @property
    def role(self) -> str | None:
        """Compatibility shorthand for callers that only need the maximum."""

        return self.effective_role


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


def agent_query_enabled() -> bool:
    return bool(getattr(_global_args(), "agent_query_enabled", False))


def agent_max_rounds() -> int:
    return max(1, int(getattr(_global_args(), "agent_max_rounds", 5) or 5))


def agent_staged_max_retrievals() -> int:
    return max(
        1, int(getattr(_global_args(), "agent_staged_max_retrievals", 24) or 24)
    )


def agent_staged_max_kbs_per_step() -> int:
    return max(
        1, int(getattr(_global_args(), "agent_staged_max_kbs_per_step", 4) or 4)
    )


def agent_workflow_prompt_max_length() -> int:
    return max(
        0,
        int(getattr(_global_args(), "agent_workflow_prompt_max_length", 16384) or 16384),
    )


def chat_session_default_context_rounds() -> int:
    """Default context rounds for newly created chat sessions.

    Configured via ``CHAT_SESSION_DEFAULT_CONTEXT_ROUNDS``; valid values are
    ``-1`` (send the full history to the LLM) or a positive round count. An
    invalid configuration falls back to ``1``.
    """
    try:
        value = int(
            getattr(_global_args(), "chat_session_default_context_rounds", 1)
        )
    except (TypeError, ValueError):
        return 1
    return value if value == -1 or value >= 1 else 1


def enterprise_artifact_download_min_role() -> str:
    configured = getattr(
        _global_args(), "enterprise_artifact_download_min_role", KB_ROLE_VIEWER
    )
    normalized = _normalize_kb_role(str(configured))
    if normalized is None:
        return KB_ROLE_VIEWER
    return normalized


def enterprise_tenant_admin_oversight_role() -> str:
    """Oversight floor role for tenant_admin/tenant_owner on tenant-owned KBs.

    Configurable via LIGHTRAG_ENTERPRISE_TENANT_ADMIN_OVERSIGHT_ROLE; accepts
    kb_viewer/kb_editor/kb_admin. kb_owner is intentionally rejected (ownership
    is platform-granted and would unlock visibility changes over members'
    private KBs). Falls back to kb_viewer on any invalid value.
    """

    configured = getattr(
        _global_args(),
        "enterprise_tenant_admin_oversight_role",
        KB_ROLE_VIEWER,
    )
    normalized = _normalize_kb_role(str(configured))
    if normalized in _TENANT_ADMIN_OVERSIGHT_ROLES:
        return normalized
    return KB_ROLE_VIEWER


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
        can_use_agent_query=True,
        can_delete_documents=True,
        can_download_files=True,
    )


class AuditService:
    def __init__(self, metadata_store: EnterpriseMetadataStore):
        self._metadata_store = metadata_store

    async def append(
        self,
        event_type: str,
        *,
        actor_user_id: str | None = None,
        actor_tenant_id: Any = _ACTOR_TENANT_UNSET,
        target_type: str | None = None,
        target_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEventRecord:
        if actor_tenant_id is _ACTOR_TENANT_UNSET:
            resolved_actor_tenant_id: str | None = None
            if actor_user_id is not None:
                actor = await self._metadata_store.get_enterprise_user_by_id(
                    actor_user_id
                )
                if actor is not None:
                    resolved_actor_tenant_id = actor.tenant_id
        else:
            resolved_actor_tenant_id = actor_tenant_id
        event = AuditEventRecord(
            id=f"audit_{uuid4().hex}",
            event_type=event_type,
            actor_user_id=actor_user_id,
            target_type=target_type,
            target_id=target_id,
            metadata=metadata or {},
            created_at=utc_now_iso(),
            actor_tenant_id=resolved_actor_tenant_id,
        )
        return await self._metadata_store.append_audit_event(event)

    async def list(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        event_type: str | None = None,
        actor_user_id: str | None = None,
        actor_tenant_id: str | None = None,
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
            actor_tenant_id=actor_tenant_id,
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
        kb_service: Any = None,
    ):
        self._metadata_store = metadata_store
        self._audit_service = audit_service
        self._kb_service = kb_service

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
        # Deliberately make the complete generation snapshot the first await.
        expected_generations = await self._service_key_kb_generations(
            normalized_scopes
        )
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
        try:
            saved = await self._metadata_store.create_enterprise_api_key(
                record,
                expected_kb_generations=expected_generations,
            )
        except KBLifecycleConflictError as exc:
            raise _kb_lifecycle_write_conflict(exc) from exc
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

    async def _service_key_kb_generations(
        self, scopes: dict[str, Any]
    ) -> dict[str, str]:
        """Snapshot managed KB generations before persisting key scopes.

        The metadata-store write validates these snapshots transactionally. A
        hard delete/recreate that races this method therefore cannot attach a
        delayed service-key grant to the replacement KB identity.
        """

        kb_roles = scopes.get("kb_roles", {})
        if not isinstance(kb_roles, dict):
            return {}
        generations: dict[str, str] = {}
        for kb_id in sorted(kb_roles):
            if self._kb_service is not None:
                try:
                    record = await self._kb_service.get(kb_id)
                except (KnowledgeBaseNotFoundError, ValueError):
                    # A concurrent hard-delete is still rejected below by its
                    # lifecycle tombstone.  Falling through also preserves
                    # compatibility for keys created directly against legacy
                    # metadata stores without a catalog record.
                    pass
                else:
                    generations[kb_id] = record.generation
                    continue
            lifecycle = await self._metadata_store.get_kb_lifecycle(kb_id)
            if lifecycle is not None:
                generations[kb_id] = lifecycle.generation
        return generations

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
            can_use_agent_query=bool(scopes.get("can_use_agent_query", False)),
            can_delete_documents=False,
            can_download_files=False,
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
        lifecycle = await self._metadata_store.get_kb_lifecycle(kb_id)
        expected_generation = lifecycle.generation if lifecycle is not None else None
        async with self._metadata_store.kb_write_guard(
            kb_id, expected_generation
        ):
            existing = await self.get_settings(user_id, kb_id)
            now = utc_now_iso()
            record = EnterpriseUserKBQuerySettingsRecord(
                user_id=user_id,
                kb_id=kb_id,
                user_prompt=user_prompt,
                created_at=existing.created_at if existing is not None else now,
                updated_at=now,
            )
            saved = (
                await self._metadata_store.upsert_enterprise_user_kb_query_settings(
                    record
                )
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
        lifecycle = await self._metadata_store.get_kb_lifecycle(kb_id)
        expected_generation = lifecycle.generation if lifecycle is not None else None
        async with self._metadata_store.kb_write_guard(
            kb_id, expected_generation
        ):
            deleted = (
                await self._metadata_store.delete_enterprise_user_kb_query_settings(
                    user_id, kb_id
                )
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


class UserAgentWorkflowPromptService:
    """Per-user Agent workflow prompt stored in enterprise system settings.

    The prompt is user-owned policy text used by the Agent planner. Audit events
    intentionally record only presence/absence, not the prompt body.
    """

    _KEY_PREFIX = "user_agent_workflow_prompt:"

    def __init__(
        self,
        metadata_store: EnterpriseMetadataStore,
        audit_service: AuditService | None = None,
    ):
        self._metadata_store = metadata_store
        self._audit_service = audit_service

    @classmethod
    def _key(cls, user_id: str) -> str:
        return f"{cls._KEY_PREFIX}{user_id}"

    async def get_prompt(self, user_id: str) -> str:
        return await self._metadata_store.get_enterprise_system_setting(
            self._key(user_id), ""
        ) or ""

    async def set_prompt(
        self,
        *,
        user_id: str,
        workflow_prompt: str,
        actor_user_id: str | None = None,
    ) -> str:
        await self._metadata_store.set_enterprise_system_setting(
            self._key(user_id), workflow_prompt, updated_by=actor_user_id or user_id
        )
        if self._audit_service is not None:
            await self._audit_service.append(
                "user_agent_workflow_prompt_updated",
                actor_user_id=actor_user_id or user_id,
                target_type="user",
                target_id=user_id,
                metadata={"has_custom_workflow_prompt": bool(workflow_prompt)},
            )
        return workflow_prompt

    async def clear_prompt(
        self,
        *,
        user_id: str,
        actor_user_id: str | None = None,
    ) -> None:
        await self.set_prompt(
            user_id=user_id,
            workflow_prompt="",
            actor_user_id=actor_user_id,
        )


class ChatConversationService:
    """Per-user chat conversation management (projects + sessions).

    Projects and sessions are pure control-plane records that let a client
    organize per-user Q&A history (user > project > session); they never touch
    LightRAG engine storage. Every operation is scoped to the owning
    ``user_id`` so users cannot read or mutate each other's records. Audit
    events record ids and flags only, never user-supplied names.
    """

    def __init__(
        self,
        metadata_store: EnterpriseMetadataStore,
        audit_service: AuditService | None = None,
        *,
        memory_admission_enabled: bool = False,
        memory_extraction_fingerprint: str | None = None,
        memory_graph_store_fingerprint: str | None = None,
        post_commit_nudge: Callable[[], Any] | None = None,
    ):
        self._metadata_store = metadata_store
        self._audit_service = audit_service
        self._memory_admission_enabled = bool(memory_admission_enabled)
        self._memory_extraction_fingerprint = (
            str(memory_extraction_fingerprint).strip()
            if memory_extraction_fingerprint
            else None
        )
        self._memory_graph_store_fingerprint = (
            str(memory_graph_store_fingerprint).strip()
            if memory_graph_store_fingerprint
            else None
        )
        if bool(self._memory_extraction_fingerprint) != bool(
            self._memory_graph_store_fingerprint
        ):
            raise ValueError(
                "Chat Memory extraction and graph-store fingerprints must be "
                "configured together"
            )
        if self._memory_admission_enabled and not self.memory_maintenance_configured:
            raise ValueError(
                "Chat Memory admission requires extraction and graph-store fingerprints"
            )
        self._post_commit_nudge = post_commit_nudge

    @property
    def memory_admission_enabled(self) -> bool:
        return self._memory_admission_enabled

    @property
    def memory_extraction_fingerprint(self) -> str | None:
        return self._memory_extraction_fingerprint

    @property
    def memory_graph_store_fingerprint(self) -> str | None:
        return self._memory_graph_store_fingerprint

    @property
    def memory_maintenance_configured(self) -> bool:
        return bool(
            self._memory_extraction_fingerprint
            and self._memory_graph_store_fingerprint
        )

    def set_post_commit_nudge_callback(
        self, callback: Callable[[], Any] | None
    ) -> None:
        self._post_commit_nudge = callback

    async def _nudge_after_memory_commit(self) -> None:
        await _run_post_commit_nudge(
            self._post_commit_nudge,
            owner="ChatConversationService",
        )

    def _memory_fingerprints(self) -> tuple[str, str]:
        extraction = self._memory_extraction_fingerprint
        graph_store = self._memory_graph_store_fingerprint
        if extraction is None or graph_store is None:
            raise RuntimeError("Chat Memory maintenance runtime is not configured")
        return extraction, graph_store

    @staticmethod
    def default_session_name() -> str:
        # Sessions are named after their creation time (server local time)
        # unless the client supplies an explicit name.
        return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")

    async def _audit(
        self,
        event_type: str,
        *,
        actor_user_id: str,
        target_type: str,
        target_id: str,
        metadata: dict[str, Any],
    ) -> None:
        if self._audit_service is not None:
            await self._audit_service.append(
                event_type,
                actor_user_id=actor_user_id,
                target_type=target_type,
                target_id=target_id,
                metadata=metadata,
            )

    async def create_project(
        self, *, user_id: str, name: str, actor_user_id: str | None = None
    ) -> ChatProjectRecord:
        now = utc_now_iso()
        record = ChatProjectRecord(
            id=f"proj_{uuid4().hex[:12]}",
            user_id=user_id,
            name=name,
            created_at=now,
            updated_at=now,
        )
        saved = await self._metadata_store.create_chat_project(record)
        await self._audit(
            "chat_project_created",
            actor_user_id=actor_user_id or user_id,
            target_type="chat_project",
            target_id=saved.id,
            metadata={"user_id": user_id},
        )
        return saved

    async def get_project(
        self, user_id: str, project_id: str
    ) -> ChatProjectRecord | None:
        return await self._metadata_store.get_chat_project(user_id, project_id)

    async def list_projects(
        self, user_id: str, *, limit: int = 50, offset: int = 0
    ) -> tuple[list[ChatProjectRecord], int]:
        return await self._metadata_store.list_chat_projects(
            user_id, limit=limit, offset=offset
        )

    async def rename_project(
        self,
        *,
        user_id: str,
        project_id: str,
        name: str,
        actor_user_id: str | None = None,
    ) -> ChatProjectRecord | None:
        renamed = await self._metadata_store.rename_chat_project(
            user_id, project_id, name=name
        )
        if renamed is not None:
            await self._audit(
                "chat_project_renamed",
                actor_user_id=actor_user_id or user_id,
                target_type="chat_project",
                target_id=project_id,
                metadata={"user_id": user_id},
            )
        return renamed

    async def delete_project(
        self, *, user_id: str, project_id: str, actor_user_id: str | None = None
    ) -> tuple[bool, int, int]:
        if self.memory_maintenance_configured:
            extraction_fingerprint, graph_store_fingerprint = (
                self._memory_fingerprints()
            )
            try:
                deleted, deleted_sessions, deleted_messages = (
                    await self._metadata_store.delete_chat_project_with_memory(
                        user_id,
                        project_id,
                        config_fingerprint=extraction_fingerprint,
                        graph_store_fingerprint=graph_store_fingerprint,
                        actor_user_id=actor_user_id or user_id,
                    )
                )
            except MetadataConflictError as exc:
                raise chat_memory_write_conflict(exc) from exc
            if deleted:
                await self._nudge_after_memory_commit()
        else:
            deleted, deleted_sessions, deleted_messages = (
                await self._metadata_store.delete_chat_project(user_id, project_id)
            )
        if deleted:
            await self._audit(
                "chat_project_deleted",
                actor_user_id=actor_user_id or user_id,
                target_type="chat_project",
                target_id=project_id,
                metadata={
                    "user_id": user_id,
                    "deleted_sessions": deleted_sessions,
                    "deleted_messages": deleted_messages,
                },
            )
        return deleted, deleted_sessions, deleted_messages

    async def create_session(
        self,
        *,
        user_id: str,
        project_id: str,
        name: str | None = None,
        context_rounds: int | None = None,
        actor_user_id: str | None = None,
    ) -> ChatSessionRecord | None:
        """Create a session under an owned project.

        Returns ``None`` when the project does not exist for this user. A
        blank/omitted ``name`` falls back to the creation-time default; an
        omitted ``context_rounds`` falls back to the deployment default
        (``CHAT_SESSION_DEFAULT_CONTEXT_ROUNDS``).
        """
        effective_name = (name or "").strip() or self.default_session_name()
        effective_rounds = (
            context_rounds
            if context_rounds is not None
            else chat_session_default_context_rounds()
        )
        now = utc_now_iso()
        record = ChatSessionRecord(
            id=f"sess_{uuid4().hex[:12]}",
            project_id=project_id,
            user_id=user_id,
            name=effective_name,
            created_at=now,
            updated_at=now,
            context_rounds=effective_rounds,
        )
        try:
            saved = await self._metadata_store.create_chat_session(record)
        except MetadataRecordNotFoundError:
            return None
        await self._audit(
            "chat_session_created",
            actor_user_id=actor_user_id or user_id,
            target_type="chat_session",
            target_id=saved.id,
            metadata={
                "user_id": user_id,
                "project_id": project_id,
                "has_custom_name": bool((name or "").strip()),
                "context_rounds": saved.context_rounds,
            },
        )
        return saved

    async def get_session(
        self, user_id: str, project_id: str, session_id: str
    ) -> ChatSessionRecord | None:
        return await self._metadata_store.get_chat_session(
            user_id, project_id, session_id
        )

    async def list_sessions(
        self, user_id: str, project_id: str, *, limit: int = 50, offset: int = 0
    ) -> tuple[list[ChatSessionRecord], int]:
        return await self._metadata_store.list_chat_sessions(
            user_id, project_id, limit=limit, offset=offset
        )

    async def update_session(
        self,
        *,
        user_id: str,
        project_id: str,
        session_id: str,
        name: str | None = None,
        context_rounds: int | None = None,
        actor_user_id: str | None = None,
    ) -> ChatSessionRecord | None:
        updated = await self._metadata_store.update_chat_session(
            user_id,
            project_id,
            session_id,
            name=name,
            context_rounds=context_rounds,
        )
        if updated is not None:
            await self._audit(
                "chat_session_updated",
                actor_user_id=actor_user_id or user_id,
                target_type="chat_session",
                target_id=session_id,
                metadata={
                    "user_id": user_id,
                    "project_id": project_id,
                    "renamed": name is not None,
                    "context_rounds_changed": context_rounds is not None,
                    "context_rounds": updated.context_rounds,
                },
            )
        return updated

    async def delete_session(
        self,
        *,
        user_id: str,
        project_id: str,
        session_id: str,
        actor_user_id: str | None = None,
    ) -> tuple[bool, int]:
        if self.memory_maintenance_configured:
            extraction_fingerprint, graph_store_fingerprint = (
                self._memory_fingerprints()
            )
            try:
                deleted, deleted_messages = (
                    await self._metadata_store.delete_chat_session_with_memory(
                        user_id,
                        project_id,
                        session_id,
                        config_fingerprint=extraction_fingerprint,
                        graph_store_fingerprint=graph_store_fingerprint,
                        actor_user_id=actor_user_id or user_id,
                    )
                )
            except MetadataConflictError as exc:
                raise chat_memory_write_conflict(exc) from exc
            if deleted:
                await self._nudge_after_memory_commit()
        else:
            deleted, deleted_messages = await self._metadata_store.delete_chat_session(
                user_id, project_id, session_id
            )
        if deleted:
            await self._audit(
                "chat_session_deleted",
                actor_user_id=actor_user_id or user_id,
                target_type="chat_session",
                target_id=session_id,
                metadata={
                    "user_id": user_id,
                    "project_id": project_id,
                    "deleted_messages": deleted_messages,
                },
            )
        return deleted, deleted_messages

    async def append_messages(
        self,
        *,
        user_id: str,
        project_id: str,
        session_id: str,
        messages: Sequence[dict[str, Any]],
        actor_user_id: str | None = None,
    ) -> list[ChatMessageRecord] | None:
        """Append messages (dicts with ``role``/``content``/``metadata``) to an
        owned session. Returns ``None`` when the session does not exist for
        this user."""
        now = utc_now_iso()
        records = [
            ChatMessageRecord(
                id=f"msg_{uuid4().hex[:12]}",
                session_id=session_id,
                project_id=project_id,
                user_id=user_id,
                role=str(message["role"]),
                content=str(message["content"]),
                metadata=dict(message.get("metadata") or {}),
                seq=0,
                created_at=now,
            )
            for message in messages
        ]
        try:
            if self._memory_admission_enabled:
                extraction_fingerprint, graph_store_fingerprint = (
                    self._memory_fingerprints()
                )
                saved = await self._metadata_store.append_chat_messages_with_memory(
                    records,
                    config_fingerprint=extraction_fingerprint,
                    graph_store_fingerprint=graph_store_fingerprint,
                    actor_user_id=actor_user_id or user_id,
                )
            else:
                saved = await self._metadata_store.append_chat_messages(records)
        except MetadataRecordNotFoundError:
            return None
        except MetadataConflictError as exc:
            raise chat_memory_write_conflict(exc) from exc
        if self._memory_admission_enabled:
            await self._nudge_after_memory_commit()
        await self._audit(
            "chat_messages_appended",
            actor_user_id=actor_user_id or user_id,
            target_type="chat_session",
            target_id=session_id,
            metadata={
                "user_id": user_id,
                "project_id": project_id,
                "message_count": len(saved),
            },
        )
        return saved

    async def list_messages(
        self,
        user_id: str,
        project_id: str,
        session_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[ChatMessageRecord], int]:
        return await self._metadata_store.list_chat_messages(
            user_id, project_id, session_id, limit=limit, offset=offset
        )

    async def get_message(
        self,
        user_id: str,
        project_id: str,
        session_id: str,
        message_id: str,
    ) -> ChatMessageRecord | None:
        return await self._metadata_store.get_chat_message(
            user_id, project_id, session_id, message_id
        )

    async def delete_message(
        self,
        *,
        user_id: str,
        project_id: str,
        session_id: str,
        message_id: str,
        actor_user_id: str | None = None,
    ) -> bool:
        if self.memory_maintenance_configured:
            extraction_fingerprint, graph_store_fingerprint = (
                self._memory_fingerprints()
            )
            try:
                deleted = await self._metadata_store.delete_chat_message_with_memory(
                    user_id,
                    project_id,
                    session_id,
                    message_id,
                    config_fingerprint=extraction_fingerprint,
                    graph_store_fingerprint=graph_store_fingerprint,
                    actor_user_id=actor_user_id or user_id,
                )
            except MetadataConflictError as exc:
                raise chat_memory_write_conflict(exc) from exc
            if deleted:
                await self._nudge_after_memory_commit()
        else:
            deleted = await self._metadata_store.delete_chat_message(
                user_id, project_id, session_id, message_id
            )
        if deleted:
            await self._audit(
                "chat_message_deleted",
                actor_user_id=actor_user_id or user_id,
                target_type="chat_session",
                target_id=session_id,
                metadata={
                    "user_id": user_id,
                    "project_id": project_id,
                    "message_id": message_id,
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
        *,
        memory_admission_enabled: bool = False,
        memory_extraction_fingerprint: str | None = None,
        memory_graph_store_fingerprint: str | None = None,
        post_commit_nudge: Callable[[], Any] | None = None,
    ):
        self._metadata_store = metadata_store
        self._audit_service = audit_service
        self._memory_admission_enabled = bool(memory_admission_enabled)
        self._memory_extraction_fingerprint = (
            str(memory_extraction_fingerprint).strip()
            if memory_extraction_fingerprint
            else None
        )
        self._memory_graph_store_fingerprint = (
            str(memory_graph_store_fingerprint).strip()
            if memory_graph_store_fingerprint
            else None
        )
        if bool(self._memory_extraction_fingerprint) != bool(
            self._memory_graph_store_fingerprint
        ):
            raise ValueError(
                "Chat Memory extraction and graph-store fingerprints must be "
                "configured together"
            )
        if self._memory_admission_enabled and not self.memory_maintenance_configured:
            raise ValueError(
                "Chat Memory admission requires extraction and graph-store fingerprints"
            )
        self._post_commit_nudge = post_commit_nudge

    @property
    def memory_admission_enabled(self) -> bool:
        return self._memory_admission_enabled

    @property
    def memory_extraction_fingerprint(self) -> str | None:
        return self._memory_extraction_fingerprint

    @property
    def memory_graph_store_fingerprint(self) -> str | None:
        return self._memory_graph_store_fingerprint

    @property
    def memory_maintenance_configured(self) -> bool:
        return bool(
            self._memory_extraction_fingerprint
            and self._memory_graph_store_fingerprint
        )

    def set_post_commit_nudge_callback(
        self, callback: Callable[[], Any] | None
    ) -> None:
        self._post_commit_nudge = callback

    async def _nudge_after_memory_commit(self) -> None:
        await _run_post_commit_nudge(
            self._post_commit_nudge,
            owner="UserService",
        )

    def _memory_fingerprints(self) -> tuple[str, str]:
        extraction = self._memory_extraction_fingerprint
        graph_store = self._memory_graph_store_fingerprint
        if extraction is None or graph_store is None:
            raise RuntimeError("Chat Memory maintenance runtime is not configured")
        return extraction, graph_store

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
                can_use_agent_query=True,
                token_version=1,
                metadata={"bootstrap": True},
                created_at=now,
                updated_at=now,
                can_delete_documents=True,
                can_download_files=True,
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
            can_use_agent_query=True,
            can_delete_documents=True,
            can_download_files=True,
            token_version=existing.token_version
            + (1 if new_hash != existing.password_hash else 0),
            updated_at=now,
        )
        if updated == existing:
            return existing
        user = await _save_user_cas(self._metadata_store, updated, existing)
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
        actor_tenant_id: Any = _ACTOR_TENANT_UNSET,
        can_create_kb: bool = False,
        can_use_bypass_query: bool = False,
        can_use_agent_query: bool = False,
        can_delete_documents: bool = False,
        can_download_files: bool = False,
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
            can_use_agent_query=can_use_agent_query,
            token_version=1,
            metadata={},
            created_at=now,
            updated_at=now,
            can_delete_documents=can_delete_documents,
            can_download_files=can_download_files,
        )
        membership = (
            EnterpriseTenantMembershipRecord(
                tenant_id=tenant_id,
                user_id=user.id,
                role=TENANT_ROLE_MEMBER,
                granted_by=created_by,
                created_at=now,
                updated_at=now,
            )
            if tenant_id is not None
            else None
        )
        try:
            created, _saved_membership = (
                await self._metadata_store.upsert_enterprise_user_with_membership(
                    user, membership
                )
            )
        except MetadataConflictError as exc:
            raise _user_write_conflict(exc) from exc
        await self._audit(
            "user_created",
            actor_user_id=created_by,
            actor_tenant_id=actor_tenant_id,
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

    async def delete_user(
        self,
        user_id: str,
        *,
        actor_user_id: str | None = None,
        actor_tenant_id: Any = _ACTOR_TENANT_UNSET,
        expected_user: EnterpriseUserRecord | None = None,
        expected_membership: Any = UNSET,
    ) -> bool:
        user = expected_user or await self.get_user_or_404(user_id)
        if user.id != user_id:
            raise HTTPException(status_code=404, detail="User not found")
        if user.system_role == SYSTEM_ROLE_SUPER_ADMIN:
            raise HTTPException(
                status_code=400, detail="Cannot delete a super admin"
            )
        kwargs: dict[str, Any] = {
            "expected_updated_at": user.updated_at,
            "expected_token_version": user.token_version,
            "expected_tenant_id": user.tenant_id,
        }
        if expected_membership is not UNSET:
            kwargs["expected_membership"] = expected_membership
        try:
            if self.memory_maintenance_configured:
                extraction_fingerprint, graph_store_fingerprint = (
                    self._memory_fingerprints()
                )
                memory_kwargs = {
                    **kwargs,
                    "config_fingerprint": extraction_fingerprint,
                    "graph_store_fingerprint": graph_store_fingerprint,
                    "actor_user_id": actor_user_id,
                }
                if actor_tenant_id is not _ACTOR_TENANT_UNSET:
                    memory_kwargs["actor_tenant_id"] = actor_tenant_id
                deleted = (
                    await self._metadata_store.delete_enterprise_user_with_memory(
                        user_id, **memory_kwargs
                    )
                )
            else:
                deleted = await self._metadata_store.delete_enterprise_user(
                    user_id, **kwargs
                )
        except MetadataConflictError as exc:
            if exc.entity_type.startswith("chat_memory"):
                raise chat_memory_write_conflict(exc) from exc
            raise _user_write_conflict(exc) from exc
        if deleted:
            if self.memory_maintenance_configured:
                await self._nudge_after_memory_commit()
            await self._audit(
                "user_deleted",
                actor_user_id=actor_user_id,
                actor_tenant_id=actor_tenant_id,
                target_type="user",
                target_id=user_id,
            )
        return deleted

    async def update_user(
        self,
        user_id: str,
        *,
        status_value: str | None = None,
        can_create_kb: bool | None = None,
        can_use_bypass_query: bool | None = None,
        can_use_agent_query: bool | None = None,
        can_delete_documents: bool | None = None,
        can_download_files: bool | None = None,
        tenant_id: Any = UNSET,
        actor_user_id: str | None = None,
        actor_tenant_id: Any = _ACTOR_TENANT_UNSET,
        expected_user: EnterpriseUserRecord | None = None,
        expected_membership: Any = UNSET,
    ) -> EnterpriseUserRecord:
        user = expected_user or await self.get_user_or_404(user_id)
        if user.id != user_id:
            raise HTTPException(status_code=404, detail="User not found")
        if status_value is not None and status_value not in USER_STATUS_VALUES:
            raise HTTPException(status_code=400, detail="Invalid user status")
        if user.system_role == SYSTEM_ROLE_SUPER_ADMIN and status_value == USER_STATUS_DISABLED:
            raise HTTPException(status_code=400, detail="Cannot disable a super admin")
        # ``UNSET`` keeps the current tenant, an explicit ``None`` clears it,
        # and a non-empty string reassigns it. Empty strings are rejected so a
        # cleared tenant is always represented as ``None``.
        if tenant_id is UNSET:
            new_tenant_id = user.tenant_id
        elif tenant_id is None:
            new_tenant_id = None
        elif isinstance(tenant_id, str) and tenant_id.strip():
            new_tenant_id = tenant_id.strip()
        else:
            raise HTTPException(
                status_code=400,
                detail="Tenant id cannot be empty; use null to clear it",
            )
        now = utc_now_iso()
        updated = replace(
            user,
            status=status_value or user.status,
            can_create_kb=user.can_create_kb
            if can_create_kb is None
            else can_create_kb,
            can_use_bypass_query=user.can_use_bypass_query
            if can_use_bypass_query is None
            else can_use_bypass_query,
            can_use_agent_query=user.can_use_agent_query
            if can_use_agent_query is None
            else can_use_agent_query,
            can_delete_documents=user.can_delete_documents
            if can_delete_documents is None
            else can_delete_documents,
            can_download_files=user.can_download_files
            if can_download_files is None
            else can_download_files,
            tenant_id=new_tenant_id,
            token_version=user.token_version + 1,
            updated_at=now,
        )
        old_tenant_id = user.tenant_id
        if new_tenant_id != old_tenant_id:
            membership = (
                EnterpriseTenantMembershipRecord(
                    tenant_id=new_tenant_id,
                    user_id=user.id,
                    role=TENANT_ROLE_MEMBER,
                    granted_by=actor_user_id,
                    created_at=now,
                    updated_at=now,
                )
                if new_tenant_id is not None
                else None
            )
            saved, _saved_membership = await _save_user_with_membership_cas(
                self._metadata_store,
                updated,
                membership,
                user,
                expected_membership=expected_membership,
            )
        elif expected_membership is not UNSET:
            saved, _saved_membership = await _save_user_with_membership_cas(
                self._metadata_store,
                updated,
                None,
                user,
                expected_membership=expected_membership,
            )
        else:
            saved = await _save_user_cas(self._metadata_store, updated, user)
        await self._audit(
            "user_updated",
            actor_user_id=actor_user_id,
            actor_tenant_id=actor_tenant_id,
            target_type="user",
            target_id=saved.id,
        )
        return saved

    async def change_password(
        self,
        user_id: str,
        password: str,
        *,
        actor_user_id: str | None = None,
        actor_tenant_id: Any = _ACTOR_TENANT_UNSET,
        expected_user: EnterpriseUserRecord | None = None,
        expected_membership: Any = UNSET,
    ) -> EnterpriseUserRecord:
        user = expected_user or await self.get_user_or_404(user_id)
        if user.id != user_id:
            raise HTTPException(status_code=404, detail="User not found")
        updated = replace(
            user,
            password_hash=hash_password(password),
            token_version=user.token_version + 1,
            updated_at=utc_now_iso(),
        )
        if expected_membership is UNSET:
            saved = await _save_user_cas(self._metadata_store, updated, user)
        else:
            saved, _saved_membership = await _save_user_with_membership_cas(
                self._metadata_store,
                updated,
                None,
                user,
                expected_membership=expected_membership,
            )
        await self._audit(
            "user_password_changed",
            actor_user_id=actor_user_id,
            actor_tenant_id=actor_tenant_id,
            target_type="user",
            target_id=saved.id,
        )
        return saved

    async def logout_all_sessions(
        self,
        user_id: str,
        *,
        actor_tenant_id: Any = _ACTOR_TENANT_UNSET,
    ) -> EnterpriseUserRecord:
        """Invalidate every outstanding token for the user ("log out all
        devices") by bumping ``token_version``."""
        user = await self.get_user_or_404(user_id)
        updated = replace(
            user,
            token_version=user.token_version + 1,
            updated_at=utc_now_iso(),
        )
        saved = await _save_user_cas(self._metadata_store, updated, user)
        await self._audit(
            "user_logged_out",
            actor_user_id=user_id,
            actor_tenant_id=actor_tenant_id,
            target_type="user",
            target_id=saved.id,
        )
        return saved

    async def update_own_profile(
        self,
        user_id: str,
        *,
        display_name: Any = UNSET,
        email: Any = UNSET,
        actor_tenant_id: Any = _ACTOR_TENANT_UNSET,
    ) -> EnterpriseUserRecord:
        """Update self-service profile fields stored in user metadata.

        ``UNSET`` keeps a field, an explicit ``None`` clears it, and a
        non-blank string sets it. Profile changes do NOT bump
        ``token_version`` — they are not security-relevant.
        """
        user = await self.get_user_or_404(user_id)
        metadata = dict(user.metadata)

        def _apply(key: str, value: Any, max_length: int) -> None:
            if value is UNSET:
                return
            if value is None:
                metadata.pop(key, None)
                return
            if not isinstance(value, str) or not value.strip():
                raise HTTPException(
                    status_code=400,
                    detail=f"{key} must be a non-empty string or null to clear it",
                )
            normalized = value.strip()
            if len(normalized) > max_length:
                raise HTTPException(
                    status_code=400,
                    detail=f"{key} must be at most {max_length} characters",
                )
            if key == "email" and "@" not in normalized:
                raise HTTPException(status_code=400, detail="Invalid email address")
            metadata[key] = normalized

        _apply("display_name", display_name, 64)
        _apply("email", email, 254)
        if metadata == user.metadata:
            return user
        updated = replace(user, metadata=metadata, updated_at=utc_now_iso())
        saved = await _save_user_cas(self._metadata_store, updated, user)
        await self._audit(
            "user_profile_updated",
            actor_user_id=user_id,
            actor_tenant_id=actor_tenant_id,
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
            "can_use_agent_query": user.can_use_agent_query,
            "can_download_files": user.can_download_files,
        }

    async def _audit(
        self,
        event_type: str,
        *,
        actor_user_id: str | None = None,
        actor_tenant_id: Any = _ACTOR_TENANT_UNSET,
        target_type: str | None = None,
        target_id: str | None = None,
    ) -> None:
        if self._audit_service is not None:
            kwargs: dict[str, Any] = {
                "actor_user_id": actor_user_id,
                "target_type": target_type,
                "target_id": target_id,
            }
            if actor_tenant_id is not _ACTOR_TENANT_UNSET:
                kwargs["actor_tenant_id"] = actor_tenant_id
            await self._audit_service.append(
                event_type,
                **kwargs,
            )


class AuthorizationService:
    def __init__(
        self,
        metadata_store: EnterpriseMetadataStore,
        audit_service: AuditService | None = None,
        kb_service: Any = None,
    ):
        self._metadata_store = metadata_store
        self._audit_service = audit_service
        # KnowledgeBaseService-like object (``await .get(kb_id)``) used to
        # resolve KB visibility; when absent, visibility implies nothing.
        self._kb_service = kb_service

    async def _append_audit(
        self,
        event_type: str,
        *,
        actor_user_id: str | None = None,
        actor_tenant_id: Any = _ACTOR_TENANT_UNSET,
        target_type: str | None = None,
        target_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._audit_service is None:
            return
        kwargs: dict[str, Any] = {
            "actor_user_id": actor_user_id,
            "target_type": target_type,
            "target_id": target_id,
            "metadata": metadata,
        }
        if actor_tenant_id is not _ACTOR_TENANT_UNSET:
            kwargs["actor_tenant_id"] = actor_tenant_id
        await self._audit_service.append(event_type, **kwargs)

    def require_super_admin(self, principal: Principal | None) -> Principal:
        principal = _require_principal(principal)
        if not principal.is_super_admin:
            raise HTTPException(status_code=403, detail="Super admin permission required")
        return principal

    def require_create_kb(self, principal: Principal | None) -> Principal:
        principal = _require_principal(principal)
        if principal.is_super_admin:
            return principal
        if principal.auth_method == SERVICE_API_KEY_AUTH_METHOD:
            raise HTTPException(status_code=403, detail="Create-KB permission required")
        if principal.tenant_id is None:
            # A tenant administrator without a canonical primary tenant must
            # not fall back to user-global create permission.
            if any(
                _tenant_role_rank(role)
                >= _TENANT_ROLE_RANK[TENANT_ROLE_ADMIN]
                for role in principal.tenant_roles.values()
            ):
                raise HTTPException(
                    status_code=403,
                    detail="Primary tenant is required for tenant KB creation",
                )
            if principal.can_create_kb:
                return principal
            raise HTTPException(status_code=403, detail="Create-KB permission required")
        tenant_role = _canonical_primary_tenant_role(principal)
        tenant_admin = _tenant_role_rank(tenant_role) >= _TENANT_ROLE_RANK[
            TENANT_ROLE_ADMIN
        ]
        if not (principal.can_create_kb or tenant_admin):
            raise HTTPException(status_code=403, detail="Create-KB permission required")
        return principal

    def require_bypass_query(self, principal: Principal | None) -> Principal:
        principal = _require_principal(principal)
        if not (principal.is_super_admin or principal.can_use_bypass_query):
            raise HTTPException(status_code=403, detail="Bypass-query permission required")
        return principal

    def require_agent_query(self, principal: Principal | None) -> Principal:
        principal = _require_principal(principal)
        if not (principal.is_super_admin or principal.can_use_agent_query):
            raise HTTPException(status_code=403, detail="Agent-query permission required")
        return principal

    def require_file_download(self, principal: Principal | None) -> Principal:
        """Require the user-global file export capability.

        Service keys intentionally bypass this user capability and continue to
        rely on their explicit KB scopes/roles. Super administrators also pass
        regardless of the stored bit.
        """

        principal = _require_principal(principal)
        if (
            principal.is_super_admin
            or principal.auth_method == SERVICE_API_KEY_AUTH_METHOD
            or principal.can_download_files
        ):
            return principal
        raise HTTPException(status_code=403, detail="File download permission required")

    # More grammatical alias for callers introduced after the original API.
    require_download_files = require_file_download

    def has_tenant_lifecycle_oversight(
        self, principal: Principal | None, record: KnowledgeBaseRecord
    ) -> bool:
        """Whether ``principal`` holds tenant lifecycle oversight over ``record``.

        Mirrors the provenance rule of :meth:`authorize_kb_lifecycle` but is a
        pure boolean check (no audit, no raise). Used to retain soft-deleted
        tenant-owned KBs in listings/detail for tenant administrators and
        owners so they can inspect and restore their members' deleted KBs.

        Returns ``True`` for a super admin, or for a tenant_admin/tenant_owner
        of the KB's canonical owning tenant. Service API keys never qualify.
        """
        principal = _require_principal(principal)
        if principal.is_super_admin:
            return True
        if principal.auth_method == SERVICE_API_KEY_AUTH_METHOD:
            return False
        tenant_role = _canonical_primary_tenant_role(principal)
        return is_tenant_owned_kb(
            record, principal.tenant_id
        ) and _tenant_role_rank(tenant_role) >= _TENANT_ROLE_RANK[TENANT_ROLE_ADMIN]

    async def kb_is_soft_deleted(self, kb_id: str) -> bool:
        """Whether ``kb_id`` resolves to a soft-deleted catalog row.

        The pre-handler access gate uses this to defer the exact KB detail GET
        (``GET /kbs/{kb_id}``) on a soft-deleted KB to the handler, which then
        renders the deleted record for an oversight-eligible tenant
        admin/owner (or super admin) and returns 404 for everyone else.
        Returns ``False`` for a missing row or any non-deleted row so normal
        active-KB authorization is unchanged.
        """
        if self._kb_service is None:
            return False
        try:
            record = await self._kb_service.get(kb_id, include_deleted=True)
        except (KnowledgeBaseNotFoundError, ValueError):
            return False
        return getattr(record, "status", None) == "deleted"

    async def authorize_kb_lifecycle(
        self,
        principal: Principal | None,
        record: KnowledgeBaseRecord,
        action: str,
    ) -> Principal:
        """Authorize destructive lifecycle operations on a loaded KB record.

        A tenant ACL or direct KB role is deliberately insufficient. Only a
        super administrator, or an administrator/owner of the KB's canonical
        owning tenant, may delete, hard-delete, or restore a genuinely
        tenant-created KB. Callers must load the relevant active/deleted catalog
        row first and pass that exact record.
        """

        normalized_action = action.strip().replace("_", "-")
        if normalized_action not in {"delete", "soft-delete", "hard-delete", "restore"}:
            raise ValueError(f"Unsupported KB lifecycle action: {action}")
        principal = _require_principal(principal)
        if self.has_tenant_lifecycle_oversight(principal, record):
            return principal

        await self._audit_lifecycle_denied(principal, record, normalized_action)
        raise HTTPException(status_code=403, detail="Knowledge-base lifecycle denied")

    async def require_kb_role(
        self, principal: Principal | None, kb_id: str, minimum_role: str
    ) -> Principal:
        principal = _require_principal(principal)
        if principal.is_super_admin:
            return principal
        normalized_minimum = _normalize_kb_role(minimum_role)
        if normalized_minimum is None:
            raise ValueError(f"Invalid minimum KB role: {minimum_role}")
        decision = await self.resolve_kb_access(principal, kb_id)
        role = decision.effective_role
        if (
            role is None
            or _KB_ROLE_RANK.get(role, 0) < _KB_ROLE_RANK[normalized_minimum]
        ):
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
        actor_tenant_id: Any = _ACTOR_TENANT_UNSET,
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
        await self._append_audit(
            "tenant_created",
            actor_user_id=created_by,
            actor_tenant_id=actor_tenant_id,
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
        actor_tenant_id: Any = _ACTOR_TENANT_UNSET,
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
        await self._append_audit(
            "tenant_updated",
            actor_user_id=actor_user_id,
            actor_tenant_id=actor_tenant_id,
            target_type="tenant",
            target_id=saved.id,
        )
        return saved

    async def delete_tenant(
        self,
        tenant_id: str,
        *,
        actor_user_id: str | None = None,
        actor_tenant_id: Any = _ACTOR_TENANT_UNSET,
    ) -> bool:
        tenant_id = _normalize_required_id(tenant_id, "Tenant id")
        deleted = await self._metadata_store.delete_enterprise_tenant(tenant_id)
        if deleted:
            await self._append_audit(
                "tenant_deleted",
                actor_user_id=actor_user_id,
                actor_tenant_id=actor_tenant_id,
                target_type="tenant",
                target_id=tenant_id,
            )
        return deleted

    async def list_kb_ids_for_tenants(self, tenant_ids: list[str]) -> list[str]:
        return await self._metadata_store.list_kb_ids_for_tenants(tenant_ids)

    async def list_user_tenant_memberships(
        self, user_id: str
    ) -> list[EnterpriseTenantMembershipRecord]:
        return await self._metadata_store.list_user_tenant_memberships(user_id)

    async def list_user_kb_acls(self, user_id: str) -> list[dict[str, str]]:
        """Direct (non tenant-inherited) KB ACLs for a user, as {kb_id, role}."""
        acls: list[dict[str, str]] = []
        for kb_id in await self._metadata_store.list_kb_ids_for_user(user_id):
            role = await self._metadata_store.get_kb_acl_role(kb_id, user_id)
            if role:
                acls.append({"kb_id": kb_id, "role": role})
        return acls

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
        role = _canonical_primary_tenant_role(principal)
        if principal.tenant_id != tenant_id:
            role = None
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
        authorized: list[KnowledgeBaseRecord] = []
        for record in records:
            if record.status != "active":
                # Tenant administrators/owners keep lifecycle oversight over
                # their members' soft-deleted tenant KBs so they can inspect
                # and restore them (mirrors authorize_kb_lifecycle provenance).
                # Other non-active lifecycle rows (creating/deleting/...) are
                # never surfaced via listings.
                if record.status == "deleted" and self.has_tenant_lifecycle_oversight(
                    principal, record
                ):
                    authorized.append(record)
                continue
            decision = await self.resolve_kb_access(principal, record)
            if decision.effective_role is not None:
                authorized.append(record)
        return authorized

    async def grant_kb_role(
        self,
        kb_id: str,
        user_id: str,
        role: str,
        *,
        granted_by: str | None = None,
        expected_generation: str | None = None,
    ) -> KBACLRecord:
        normalized_role = _normalize_kb_role(role)
        if normalized_role not in _KB_ROLE_RANK:
            raise HTTPException(status_code=400, detail="Invalid KB ACL role")
        captured_generation = await self._expected_kb_generation(
            kb_id,
            expected_generation=expected_generation,
        )
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
        try:
            saved = await self._metadata_store.upsert_kb_acl(
                acl,
                expected_generation=captured_generation,
            )
        except KBLifecycleConflictError as exc:
            raise _kb_lifecycle_write_conflict(exc) from exc
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
        self,
        kb_id: str,
        user_id: str,
        *,
        actor_user_id: str | None = None,
        expected_generation: str | None = None,
    ) -> bool:
        captured_generation = await self._expected_kb_generation(
            kb_id,
            expected_generation=expected_generation,
        )
        try:
            deleted = await self._metadata_store.delete_kb_acl(
                kb_id,
                user_id,
                expected_generation=captured_generation,
            )
        except KBLifecycleConflictError as exc:
            raise _kb_lifecycle_write_conflict(exc) from exc
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
        actor_tenant_id: Any = _ACTOR_TENANT_UNSET,
        expected_user: EnterpriseUserRecord | None = None,
        expected_membership: Any = UNSET,
        allow_tenant_move: bool = True,
    ) -> EnterpriseTenantMembershipRecord:
        tenant_id = _normalize_required_id(tenant_id, "Tenant id")
        normalized_role = _normalize_tenant_role(role)
        user = expected_user or await self._metadata_store.get_enterprise_user_by_id(
            user_id
        )
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        if user.id != user_id:
            raise HTTPException(status_code=404, detail="User not found")
        if not allow_tenant_move and (
            user.system_role != SYSTEM_ROLE_USER
            or user.tenant_id is not None
            or expected_membership is not None
        ):
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
        updated_user = replace(user, tenant_id=tenant_id, updated_at=now)
        _saved_user, saved = await _save_user_with_membership_cas(
            self._metadata_store,
            updated_user,
            membership,
            user,
            expected_membership=expected_membership,
        )
        if saved is None:
            raise RuntimeError("Canonical tenant membership was not saved")
        await self._append_audit(
            "tenant_membership_granted",
            actor_user_id=granted_by,
            actor_tenant_id=actor_tenant_id,
            target_type="tenant",
            target_id=tenant_id,
            metadata={"user_id": user_id, "role": normalized_role},
        )
        return saved

    async def revoke_tenant_membership(
        self,
        tenant_id: str,
        user_id: str,
        *,
        actor_user_id: str | None = None,
        actor_tenant_id: Any = _ACTOR_TENANT_UNSET,
        expected_user: EnterpriseUserRecord | None = None,
        expected_membership: Any = UNSET,
    ) -> bool:
        tenant_id = _normalize_required_id(tenant_id, "Tenant id")
        membership = (
            expected_membership
            if expected_membership is not UNSET
            else await self._metadata_store.get_tenant_membership(tenant_id, user_id)
        )
        if membership is None:
            return False
        if not isinstance(membership, EnterpriseTenantMembershipRecord):
            raise HTTPException(status_code=409, detail="User membership changed")
        user = expected_user or await self._metadata_store.get_enterprise_user_by_id(
            user_id
        )
        if user is None:
            return False
        if (
            user.id != user_id
            or membership.user_id != user_id
            or membership.tenant_id != tenant_id
            or user.tenant_id != tenant_id
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User tenant membership is inconsistent",
            )
        updated_user = replace(user, tenant_id=None, updated_at=utc_now_iso())
        await _save_user_with_membership_cas(
            self._metadata_store,
            updated_user,
            None,
            user,
            expected_membership=expected_membership,
        )
        await self._append_audit(
            "tenant_membership_revoked",
            actor_user_id=actor_user_id,
            actor_tenant_id=actor_tenant_id,
            target_type="tenant",
            target_id=tenant_id,
            metadata={"user_id": user_id},
        )
        return True

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
        expected_generation: str | None = None,
    ) -> EnterpriseTenantKBACLRecord:
        tenant_id = _normalize_required_id(tenant_id, "Tenant id")
        normalized_role = _normalize_kb_role(role)
        if normalized_role not in _KB_ROLE_RANK:
            raise HTTPException(status_code=400, detail="Invalid KB ACL role")
        captured_generation = await self._expected_kb_generation(
            kb_id,
            expected_generation=expected_generation,
        )
        now = utc_now_iso()
        acl = EnterpriseTenantKBACLRecord(
            tenant_id=tenant_id,
            kb_id=kb_id,
            role=normalized_role,
            granted_by=granted_by,
            created_at=now,
            updated_at=now,
        )
        try:
            saved = await self._metadata_store.upsert_tenant_kb_acl(
                acl,
                expected_generation=captured_generation,
            )
        except KBLifecycleConflictError as exc:
            raise _kb_lifecycle_write_conflict(exc) from exc
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
        self,
        kb_id: str,
        tenant_id: str,
        *,
        actor_user_id: str | None = None,
        expected_generation: str | None = None,
    ) -> bool:
        tenant_id = _normalize_required_id(tenant_id, "Tenant id")
        captured_generation = await self._expected_kb_generation(
            kb_id,
            expected_generation=expected_generation,
        )
        try:
            deleted = await self._metadata_store.delete_tenant_kb_acl(
                tenant_id,
                kb_id,
                expected_generation=captured_generation,
            )
        except KBLifecycleConflictError as exc:
            raise _kb_lifecycle_write_conflict(exc) from exc
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

    async def set_tenant_user_kb_override(
        self,
        kb_id: str,
        tenant_id: str,
        user_id: str,
        *,
        effect: str,
        role: str | None = None,
        granted_by: str | None = None,
        actor_tenant_id: Any = _ACTOR_TENANT_UNSET,
        expected_generation: str | None = None,
        expected_user: Any = UNSET,
        expected_membership: Any = UNSET,
    ) -> EnterpriseTenantUserKBOverrideRecord:
        tenant_id = _normalize_required_id(tenant_id, "Tenant id")
        user_id = _normalize_required_id(user_id, "User id")
        effect_value = effect.strip().lower()
        if effect_value not in {"allow", "deny"}:
            raise HTTPException(status_code=400, detail="Invalid tenant KB override effect")
        normalized_effect: TenantUserKBOverrideEffect = (
            "allow" if effect_value == "allow" else "deny"
        )
        normalized_role = _normalize_kb_role(role)
        if normalized_effect == "allow" and normalized_role not in _KB_ROLE_RANK:
            raise HTTPException(status_code=400, detail="Invalid KB ACL role")
        if normalized_effect == "deny" and role is not None:
            raise HTTPException(
                status_code=400,
                detail="Deny override must not include a role",
            )
        captured_generation = await self._expected_kb_generation(
            kb_id,
            expected_generation=expected_generation,
        )
        now = utc_now_iso()
        override = EnterpriseTenantUserKBOverrideRecord(
            tenant_id=tenant_id,
            kb_id=kb_id,
            user_id=user_id,
            effect=normalized_effect,
            role=normalized_role if normalized_effect == "allow" else None,
            granted_by=granted_by,
            created_at=now,
            updated_at=now,
        )
        try:
            target_cas: dict[str, Any] = {}
            if expected_user is not UNSET:
                target_cas["expected_user"] = expected_user
            if expected_membership is not UNSET:
                target_cas["expected_membership"] = expected_membership
            saved = await self._metadata_store.upsert_tenant_user_kb_override(
                override,
                expected_generation=captured_generation,
                **target_cas,
            )
        except KBLifecycleConflictError as exc:
            raise _kb_lifecycle_write_conflict(exc) from exc
        except MetadataConflictError as exc:
            raise _tenant_override_target_write_conflict(exc) from exc
        await self._append_audit(
            "tenant_user_kb_override_set",
            actor_user_id=granted_by,
            actor_tenant_id=actor_tenant_id,
            target_type="kb",
            target_id=kb_id,
            metadata={
                "tenant_id": tenant_id,
                "user_id": user_id,
                "effect": normalized_effect,
                "role": saved.role,
            },
        )
        return saved

    async def grant_tenant_user_kb_override(
        self,
        kb_id: str,
        tenant_id: str,
        user_id: str,
        role: str,
        *,
        granted_by: str | None = None,
        actor_tenant_id: Any = _ACTOR_TENANT_UNSET,
        expected_generation: str | None = None,
        expected_user: Any = UNSET,
        expected_membership: Any = UNSET,
    ) -> EnterpriseTenantUserKBOverrideRecord:
        return await self.set_tenant_user_kb_override(
            kb_id,
            tenant_id,
            user_id,
            effect="allow",
            role=role,
            granted_by=granted_by,
            actor_tenant_id=actor_tenant_id,
            expected_generation=expected_generation,
            expected_user=expected_user,
            expected_membership=expected_membership,
        )

    async def revoke_tenant_user_kb_override(
        self,
        kb_id: str,
        tenant_id: str,
        user_id: str,
        *,
        granted_by: str | None = None,
        actor_tenant_id: Any = _ACTOR_TENANT_UNSET,
        expected_generation: str | None = None,
        expected_user: Any = UNSET,
        expected_membership: Any = UNSET,
    ) -> EnterpriseTenantUserKBOverrideRecord:
        return await self.set_tenant_user_kb_override(
            kb_id,
            tenant_id,
            user_id,
            effect="deny",
            granted_by=granted_by,
            actor_tenant_id=actor_tenant_id,
            expected_generation=expected_generation,
            expected_user=expected_user,
            expected_membership=expected_membership,
        )

    async def reset_tenant_user_kb_override(
        self,
        kb_id: str,
        tenant_id: str,
        user_id: str,
        *,
        actor_user_id: str | None = None,
        actor_tenant_id: Any = _ACTOR_TENANT_UNSET,
        expected_generation: str | None = None,
        expected_user: Any = UNSET,
        expected_membership: Any = UNSET,
    ) -> bool:
        tenant_id = _normalize_required_id(tenant_id, "Tenant id")
        user_id = _normalize_required_id(user_id, "User id")
        captured_generation = await self._expected_kb_generation(
            kb_id,
            expected_generation=expected_generation,
        )
        try:
            target_cas: dict[str, Any] = {}
            if expected_user is not UNSET:
                target_cas["expected_user"] = expected_user
            if expected_membership is not UNSET:
                target_cas["expected_membership"] = expected_membership
            deleted = await self._metadata_store.reset_tenant_user_kb_override(
                tenant_id,
                kb_id,
                user_id,
                expected_generation=captured_generation,
                **target_cas,
            )
        except KBLifecycleConflictError as exc:
            raise _kb_lifecycle_write_conflict(exc) from exc
        except MetadataConflictError as exc:
            raise _tenant_override_target_write_conflict(exc) from exc
        if deleted:
            await self._append_audit(
                "tenant_user_kb_override_reset",
                actor_user_id=actor_user_id,
                actor_tenant_id=actor_tenant_id,
                target_type="kb",
                target_id=kb_id,
                metadata={"tenant_id": tenant_id, "user_id": user_id},
            )
        return deleted

    async def resolve_kb_access(
        self,
        principal: Principal | None,
        record_or_kb_id: KnowledgeBaseRecord | str,
    ) -> KBAccessDecision:
        """Resolve effective access while preserving each authorization source.

        For interactive users this implements the tenant override formula
        documented by the governance contract. Direct grants and visibility are
        platform sources; tenant allow/deny rows can affect only the one
        canonical tenant contribution. Service keys retain explicit scope plus
        optional tenant-ACL inheritance semantics.
        """

        principal = _require_principal(principal)
        record: KnowledgeBaseRecord | None
        if isinstance(record_or_kb_id, str):
            kb_id = record_or_kb_id
            record = await self._load_kb_record(kb_id)
        else:
            record = record_or_kb_id
            kb_id = record.id

        generation = record.generation if record is not None else None
        if principal.is_super_admin:
            return KBAccessDecision(
                kb_id=kb_id,
                generation=generation,
                effective_role=KB_ROLE_OWNER,
                platform_role=KB_ROLE_OWNER,
                direct_role=None,
                visibility_role=None,
                tenant_role=None,
                tenant_id=principal.tenant_id,
                tenant_acl_role=None,
                tenant_override_effect=None,
                tenant_override_role=None,
                tenant_owned=False,
                sources=("super_admin",),
            )

        # A missing or non-active catalog row is not an ACL-only KB. Fail closed
        # instead of exposing a half-created, disabled, or stale identity.
        if (self._kb_service is not None and record is None) or (
            record is not None and getattr(record, "status", None) != "active"
        ):
            return _empty_kb_access_decision(kb_id, generation, principal.tenant_id)

        if principal.auth_method == SERVICE_API_KEY_AUTH_METHOD:
            explicit_role = _service_api_key_kb_role(principal, kb_id)
            tenant_acl_role: str | None = None
            if (
                _service_api_key_inherits_tenant_kb_acl(principal)
                and principal.tenant_id
            ):
                tenant_acl_role = _normalize_kb_role(
                    await self._metadata_store.get_tenant_kb_acl_role(
                        principal.tenant_id,
                        kb_id,
                    )
                )
            effective_role = _max_kb_role(explicit_role, tenant_acl_role)
            sources = tuple(
                source
                for source, role in (
                    ("service_key", explicit_role),
                    ("service_key_tenant_acl", tenant_acl_role),
                )
                if role is not None
            )
            return KBAccessDecision(
                kb_id=kb_id,
                generation=generation,
                effective_role=effective_role,
                platform_role=explicit_role,
                direct_role=explicit_role,
                visibility_role=None,
                tenant_role=tenant_acl_role,
                tenant_id=principal.tenant_id,
                tenant_acl_role=tenant_acl_role,
                tenant_override_effect=None,
                tenant_override_role=None,
                tenant_owned=False,
                sources=sources,
            )

        primary_tenant_role = _canonical_primary_tenant_role(principal)
        direct_role = _normalize_kb_role(
            await self._metadata_store.get_kb_acl_role(kb_id, principal.user_id)
        )
        visibility_role = (
            KB_ROLE_VIEWER
            if record is not None and self._visibility_grants_view(principal, record)
            else None
        )
        platform_role = _max_kb_role(direct_role, visibility_role)

        # AuthorizationService is also used in a few storage-level contexts
        # without a catalog service. Preserve direct/tenant ACL compatibility
        # there; source-aware owned/provisioned semantics require a KB record.
        if record is None:
            tenant_acl_role: str | None = None
            if principal.tenant_id is not None and primary_tenant_role is not None:
                tenant_acl_role = _normalize_kb_role(
                    await self._metadata_store.get_tenant_kb_acl_role(
                        principal.tenant_id, kb_id
                    )
                )
            effective_role = _max_kb_role(platform_role, tenant_acl_role)
            sources = tuple(
                source
                for source, role in (
                    ("direct", direct_role),
                    ("tenant_acl", tenant_acl_role),
                )
                if role is not None
            )
            return KBAccessDecision(
                kb_id=kb_id,
                generation=None,
                effective_role=effective_role,
                platform_role=platform_role,
                direct_role=direct_role,
                visibility_role=None,
                tenant_role=tenant_acl_role,
                tenant_id=principal.tenant_id,
                tenant_acl_role=tenant_acl_role,
                tenant_override_effect=None,
                tenant_override_role=None,
                tenant_owned=False,
                sources=sources,
            )

        tenant_id = principal.tenant_id
        tenant_owned = is_tenant_owned_kb(record, tenant_id)
        tenant_acl_role: str | None = None
        override: EnterpriseTenantUserKBOverrideRecord | None = None
        if tenant_id is not None and primary_tenant_role is not None:
            tenant_acl_role = _normalize_kb_role(
                await self._metadata_store.get_tenant_kb_acl_role(tenant_id, kb_id)
            )
            override = await self._metadata_store.get_tenant_user_kb_override(
                tenant_id,
                kb_id,
                principal.user_id,
            )

        override_effect = override.effect if override is not None else None
        override_role = (
            _normalize_kb_role(override.role) if override is not None else None
        )
        tenant_role: str | None = None
        tenant_source: str | None = None
        if tenant_owned:
            # Tenant-owned KBs are opt-in per member. The creator's direct owner
            # ACL remains a platform source and is therefore unaffected.
            if override_effect == "allow":
                tenant_role = override_role
                tenant_source = "tenant_owned_override"
            if (
                tenant_role is None
                and _tenant_role_rank(primary_tenant_role)
                >= _TENANT_ROLE_RANK[TENANT_ROLE_ADMIN]
            ):
                # Tenant administrators keep oversight of every KB owned by
                # their tenant, shared or private; a deny override cannot
                # remove it. The oversight floor role is configurable via
                # LIGHTRAG_ENTERPRISE_TENANT_ADMIN_OVERSIGHT_ROLE (default
                # kb_viewer); a direct platform ACL still wins via _max_kb_role.
                tenant_role = enterprise_tenant_admin_oversight_role()
                tenant_source = "tenant_admin_oversight"
        elif tenant_acl_role is not None:
            if override is None:
                tenant_role = tenant_acl_role
                tenant_source = "tenant_acl"
            elif override_effect == "allow" and override_role is not None:
                tenant_role = _min_kb_role(override_role, tenant_acl_role)
                tenant_source = "tenant_override_capped"
            # A deny, malformed override, or override without a current tenant
            # ACL contributes no tenant access.

        effective_role = _max_kb_role(platform_role, tenant_role)
        sources = tuple(
            source
            for source in (
                "direct" if direct_role is not None else None,
                "visibility" if visibility_role is not None else None,
                tenant_source,
            )
            if source is not None
        )
        return KBAccessDecision(
            kb_id=kb_id,
            generation=generation,
            effective_role=effective_role,
            platform_role=platform_role,
            direct_role=direct_role,
            visibility_role=visibility_role,
            tenant_role=tenant_role,
            tenant_id=tenant_id,
            tenant_acl_role=tenant_acl_role,
            tenant_override_effect=override_effect,
            tenant_override_role=override_role,
            tenant_owned=tenant_owned,
            sources=sources,
        )

    async def _load_kb_record(self, kb_id: str) -> KnowledgeBaseRecord | None:
        if self._kb_service is None:
            return None
        try:
            return await self._kb_service.get(kb_id)
        except (KnowledgeBaseNotFoundError, ValueError):
            return None

    async def _expected_kb_generation(
        self,
        kb_id: str,
        *,
        expected_generation: str | None = None,
    ) -> str | None:
        """Resolve and validate the identity used by an ACL mutation."""

        if expected_generation is not None:
            try:
                await self._metadata_store.assert_current_kb_generation(
                    kb_id,
                    expected_generation,
                )
            except KBLifecycleConflictError as exc:
                raise _kb_lifecycle_write_conflict(exc) from exc
            return expected_generation

        captured_generation: str | None = None
        if self._kb_service is not None:
            try:
                record = await self._kb_service.get(kb_id)
            except KnowledgeBaseNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            captured_generation = record.generation
            try:
                await self._metadata_store.register_kb_generation(
                    kb_id,
                    record.generation,
                )
            except KBLifecycleConflictError as exc:
                raise _kb_lifecycle_write_conflict(exc) from exc
        else:
            lifecycle = await self._metadata_store.get_kb_lifecycle(kb_id)
            if lifecycle is not None:
                captured_generation = lifecycle.generation

        try:
            await self._metadata_store.assert_current_kb_generation(
                kb_id,
                captured_generation,
            )
        except KBLifecycleConflictError as exc:
            raise _kb_lifecycle_write_conflict(exc) from exc
        return captured_generation

    async def _effective_user_kb_role(
        self, principal: Principal, kb_id: str
    ) -> str | None:
        return (await self.resolve_kb_access(principal, kb_id)).effective_role

    @staticmethod
    def _visibility_grants_view(principal: Principal, record: Any) -> bool:
        """Return True when KB visibility implies read access for the principal.

        Only interactive (non service-key) principals qualify: ``public``
        grants every authenticated enterprise user, ``internal`` grants users
        belonging to the KB's tenant via direct assignment or membership.
        Service/scoped API keys keep explicit-scope-only semantics.
        """
        if principal.auth_method == SERVICE_API_KEY_AUTH_METHOD:
            return False
        visibility = getattr(record, "visibility", None)
        if visibility == KB_VISIBILITY_PUBLIC:
            return True
        if visibility == KB_VISIBILITY_INTERNAL:
            kb_tenant = getattr(record, "tenant_id", None)
            return bool(kb_tenant) and (
                principal.tenant_id == kb_tenant
                or kb_tenant in principal.tenant_roles
            )
        return False

    async def _effective_service_api_key_kb_role(
        self, principal: Principal, kb_id: str
    ) -> str | None:
        return (await self.resolve_kb_access(principal, kb_id)).effective_role

    async def _audit_denied(
        self, principal: Principal, kb_id: str, minimum_role: str
    ) -> None:
        if self._audit_service is not None:
            await self._audit_service.append(
                "permission_denied",
                actor_user_id=principal.user_id,
                actor_tenant_id=principal.tenant_id,
                target_type="kb",
                target_id=kb_id,
                metadata={"minimum_role": minimum_role},
            )

    async def _audit_lifecycle_denied(
        self,
        principal: Principal,
        record: KnowledgeBaseRecord,
        action: str,
    ) -> None:
        if self._audit_service is not None:
            await self._audit_service.append(
                "permission_denied",
                actor_user_id=principal.user_id,
                actor_tenant_id=principal.tenant_id,
                target_type="kb",
                target_id=record.id,
                metadata={
                    "action": action,
                    "kb_tenant_id": record.tenant_id,
                    "required_tenant_role": TENANT_ROLE_ADMIN,
                },
            )

    async def _audit_tenant_denied(
        self, principal: Principal, tenant_id: str, minimum_role: str
    ) -> None:
        if self._audit_service is not None:
            await self._audit_service.append(
                "permission_denied",
                actor_user_id=principal.user_id,
                actor_tenant_id=principal.tenant_id,
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
    canonical_memberships = memberships or []
    if user.tenant_id is None:
        consistent = not canonical_memberships
    else:
        consistent = (
            len(canonical_memberships) == 1
            and canonical_memberships[0].tenant_id == user.tenant_id
            and canonical_memberships[0].user_id == user.id
            and _canonical_tenant_role(canonical_memberships[0].role)
            in _TENANT_ROLE_RANK
        )
    if not consistent:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tenant membership is inconsistent",
        )
    tenant_roles = {
        membership.tenant_id: _canonical_tenant_role(membership.role) or membership.role
        for membership in canonical_memberships
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
        can_use_agent_query=user.can_use_agent_query,
        token_version=user.token_version,
        auth_method=auth_method,
        metadata=dict(user.metadata),
        can_delete_documents=user.can_delete_documents,
        can_download_files=user.can_download_files,
    )


def get_request_principal(request: Request) -> Principal | None:
    value = getattr(request.state, "principal", None)
    return value if isinstance(value, Principal) else None


def get_enterprise_user_service(request: Request) -> UserService:
    service = getattr(request.app.state, "enterprise_user_service", None)
    if not isinstance(service, UserService):
        raise HTTPException(status_code=500, detail="Enterprise user service unavailable")
    return service


# Module-level registry of the app-bound metadata store / services. This is
# populated by the server lifespan (see lightrag_server) once the app state is
# assembled, so person-session validation (which runs before any request
# service is materialized) can resolve the store without request state.
_active_metadata_store: EnterpriseMetadataStore | None = None


def set_active_metadata_store(store: EnterpriseMetadataStore | None) -> None:
    """Register the running app's metadata store for person-session validation."""

    global _active_metadata_store
    _active_metadata_store = store


def _get_request_app_state_metadata_store() -> EnterpriseMetadataStore:
    """Resolve the app-bound metadata store for person-session validation.

    Person-session validation runs from ``combined_auth`` before any request
    service has been materialized, so it cannot read ``request.app.state``
    via a dependency. Instead it uses the module-level singleton registered by
    the server lifespan. Fallback-free: if nothing is registered, person auth
    is unavailable.
    """

    if _active_metadata_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Person auth metadata store is unavailable",
        )
    return _active_metadata_store


def get_enterprise_settings_service(request: Request) -> SystemSettingsService:
    service = getattr(request.app.state, "enterprise_settings_service", None)
    if not isinstance(service, SystemSettingsService):
        raise HTTPException(status_code=500, detail="Enterprise settings service unavailable")
    return service


def get_person_service(request: Request) -> Any:
    """Return the app-bound :class:`PersonService` (lazy import to avoid cycle).

    Raises 503 when person auth is disabled or the service was not constructed.
    """

    if not enterprise_auth_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Person auth is unavailable",
        )
    service = getattr(request.app.state, "person_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Person auth is unavailable",
        )
    return service


def get_person_token_handler(request: Request) -> Any:
    """Return the app-bound :class:`PersonTokenHandler` (lazy import to avoid cycle)."""

    handler = getattr(request.app.state, "person_token_handler", None)
    if handler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Person auth is unavailable",
        )
    return handler


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


def get_enterprise_user_agent_workflow_prompt_service(
    request: Request,
) -> UserAgentWorkflowPromptService:
    service = getattr(
        request.app.state, "enterprise_user_agent_workflow_prompt_service", None
    )
    if not isinstance(service, UserAgentWorkflowPromptService):
        raise HTTPException(
            status_code=500,
            detail="Enterprise user Agent workflow prompt service unavailable",
        )
    return service


def get_enterprise_chat_conversation_service(
    request: Request,
) -> ChatConversationService:
    service = getattr(request.app.state, "enterprise_chat_conversation_service", None)
    if not isinstance(service, ChatConversationService):
        raise HTTPException(
            status_code=500,
            detail="Enterprise chat conversation service unavailable",
        )
    return service


def get_enterprise_chat_memory_service(request: Request) -> Any:
    """Optional per-user-per-project chat memory service (graphiti).

    Returns ``None`` when the feature is disabled (``LIGHTRAG_CHAT_MEMORY_ENABLED``
    off or non-enterprise mode). Deliberately duck-typed — tests inject fakes
    and the real service lives in ``lightrag.api.chat_memory_service``.
    """
    return getattr(request.app.state, "enterprise_chat_memory_service", None)


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
        actor_tenant_id=principal.tenant_id if principal is not None else None,
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
        # This one compatibility endpoint is intentionally deferred to its
        # handler, which re-authorizes a tenant admin against the exact path
        # tenant.  Do not use prefix matching here: every sibling /admin route
        # and every subpath remains super-admin-only.
        if method == "GET" and _exact_admin_tenant_detail_id(path) is not None:
            return
        # POST /admin/persons/{person_id}/accounts/{account_id} (pending-link
        # proposal) authorizes super admin OR the target account's tenant admin
        # inside its handler (PersonService.authorize_link_proposal). Exact
        # match only: DELETE on the same path (unbind) and every other person
        # admin route remain super-admin-only.
        if method == "POST" and _exact_admin_person_link_proposal(path):
            return
        authz.require_super_admin(principal)
        return

    kb_id = _extract_kb_id(path)
    if kb_id is None:
        if path == "/kbs" and method == "POST":
            authz.require_create_kb(principal)
        return

    if method == "DELETE" and path.rstrip("/") == f"/kbs/{kb_id}":
        return

    if method == "POST" and path.rstrip("/") == f"/kbs/{kb_id}:restore":
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
        # Graph reads stay viewer-level; graph writes (entity/relation
        # edit/create/merge/delete) are knowledge surgery and require admin.
        minimum = KB_ROLE_VIEWER if method == "GET" else KB_ROLE_ADMIN
        await authz.require_kb_role(principal, kb_id, minimum)
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
        # The exact KB detail GET (``GET /kbs/{kb_id}``) may target a
        # soft-deleted KB that a tenant admin/owner can view under lifecycle
        # oversight. resolve_kb_access denies non-active rows, so defer such
        # requests to the handler, which re-checks oversight and returns 404
        # for everyone else. All other GETs (sub-resources, status, etc.) are
        # out of scope here.
        if (
            path.rstrip("/") == f"/kbs/{kb_id}"
            and await authz.kb_is_soft_deleted(kb_id)
        ):
            return
        await authz.require_kb_role(principal, kb_id, KB_ROLE_VIEWER)
        return

    await authz.require_kb_role(principal, kb_id, KB_ROLE_EDITOR)


def _require_principal(principal: Principal | None) -> Principal:
    if principal is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Login required")
    if principal.status != USER_STATUS_ACTIVE:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User is disabled")
    return principal


def _canonical_primary_tenant_role(principal: Principal) -> str | None:
    """Return the sole canonical tenant role or fail closed on divergence."""

    if principal.auth_method == SERVICE_API_KEY_AUTH_METHOD:
        return None
    tenant_id = principal.tenant_id
    if tenant_id is None:
        if principal.tenant_roles:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Tenant membership is inconsistent",
            )
        return None
    if set(principal.tenant_roles) != {tenant_id}:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tenant membership is inconsistent",
        )
    role = _canonical_tenant_role(principal.tenant_roles.get(tenant_id))
    if role not in _TENANT_ROLE_RANK:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tenant membership is inconsistent",
        )
    return role


def _max_kb_role(*roles: str | None) -> str | None:
    normalized = [role for item in roles if (role := _normalize_kb_role(item))]
    if not normalized:
        return None
    return max(normalized, key=lambda item: _KB_ROLE_RANK[item])


def _min_kb_role(*roles: str | None) -> str | None:
    normalized = [role for item in roles if (role := _normalize_kb_role(item))]
    if not normalized:
        return None
    return min(normalized, key=lambda item: _KB_ROLE_RANK[item])


def _empty_kb_access_decision(
    kb_id: str, generation: str | None, tenant_id: str | None
) -> KBAccessDecision:
    return KBAccessDecision(
        kb_id=kb_id,
        generation=generation,
        effective_role=None,
        platform_role=None,
        direct_role=None,
        visibility_role=None,
        tenant_role=None,
        tenant_id=tenant_id,
        tenant_acl_role=None,
        tenant_override_effect=None,
        tenant_override_role=None,
        tenant_owned=False,
    )


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


def tenant_role_is_admin(role: str | None) -> bool:
    """True when ``role`` ranks at or above tenant admin (admin or owner)."""

    return _tenant_role_rank(role) >= _TENANT_ROLE_RANK[TENANT_ROLE_ADMIN]


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
        "can_use_agent_query": bool(scopes.get("can_use_agent_query", False)),
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
        agent_query_enabled=_env_bool("LIGHTRAG_AGENT_QUERY_ENABLED", False),
        agent_max_rounds=_env_int("AGENT_MAX_ROUNDS", 5),
        agent_staged_max_retrievals=_env_int("AGENT_STAGED_MAX_RETRIEVALS", 24),
        agent_staged_max_kbs_per_step=_env_int("AGENT_STAGED_MAX_KBS_PER_STEP", 4),
        agent_workflow_prompt_max_length=_env_int(
            "AGENT_WORKFLOW_PROMPT_MAX_LENGTH", 16384
        ),
        enterprise_artifact_download_min_role=os.getenv(
            "LIGHTRAG_ENTERPRISE_ARTIFACT_DOWNLOAD_MIN_ROLE", KB_ROLE_VIEWER
        ),
        enterprise_tenant_admin_oversight_role=os.getenv(
            "LIGHTRAG_ENTERPRISE_TENANT_ADMIN_OVERSIGHT_ROLE", KB_ROLE_VIEWER
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
        # KB-level custom actions keep the verb in the same path segment
        # ("/kbs/{kb_id}:rebuild", "/kbs/{kb_id}:restore"); kb ids themselves
        # can never contain ":" (validate_kb_id), so strip the suffix instead
        # of treating "kb_x:rebuild" as a (non-existent) kb id.
        kb_segment = parts[1].split(":", 1)[0]
        return kb_segment or None
    return None


def _exact_admin_tenant_detail_id(path: str) -> str | None:
    parts = path.split("/")
    if (
        len(parts) == 4
        and parts[0] == ""
        and parts[1] == "admin"
        and parts[2] == "tenants"
        and parts[3]
    ):
        return parts[3]
    return None


def _exact_admin_person_link_proposal(path: str) -> bool:
    """Exact match for /admin/persons/{person_id}/accounts/{account_id}."""

    parts = path.rstrip("/").split("/")
    return (
        len(parts) == 6
        and parts[0] == ""
        and parts[1] == "admin"
        and parts[2] == "persons"
        and bool(parts[3])
        and parts[4] == "accounts"
        and bool(parts[5])
    )


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
