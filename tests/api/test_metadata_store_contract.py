"""Backend-parity contract tests for the KB control-plane metadata store.

The production backend can be either :class:`SQLiteMetadataStore` (default
``local``) or :class:`PostgresMetadataStore` (``LIGHTRAG_KB_METADATA_BACKEND=
postgres``). Routes/services depend only on the shared public method surface,
so correctness hinges on the two backends behaving identically.

Every test here is parametrized over both backends:

* ``sqlite`` always runs (in CI and locally) — it exercises the real contract
  logic end to end.
* ``postgres`` runs **live** against a real PostgreSQL only when
  ``LIGHTRAG_KB_POSTGRES_TEST_DSN`` (or ``POSTGRES_TEST_DSN``) is set; otherwise
  it is skipped with a clear reason. KB-scoped records use a unique ``kb_id``
  purged at the end, but enterprise records (users/tenants/memberships/audit)
  use fixed identifiers and are NOT purged — so point the DSN at a **disposable
  test database**, never production (re-running against the same DB would leave
  residue and collide on unique usernames).

This closes the gap where the Postgres backend previously had zero behavioral
coverage (all KB tests instantiated SQLite only). Run live coverage against a
throwaway DB on the same server, e.g.::

    LIGHTRAG_KB_POSTGRES_TEST_DSN=postgresql://admin:123456@127.0.0.1:5433/lightrag_contract_test \
        uv run pytest tests/api/test_metadata_store_contract.py -q
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
import uuid
from typing import Any

import pytest

from lightrag.api.enterprise_auth import AuditService
from lightrag.api.kb_service import utc_now_iso
from lightrag.api.metadata_store import (
    ArtifactRecord,
    AuditEventRecord,
    ChatMessageRecord,
    ChatProjectRecord,
    ChatSessionRecord,
    ConfigVersionRecord,
    DocumentRecord,
    EnterpriseAPIKeyRecord,
    EnterpriseInvitationRecord,
    EnterpriseUserRecord,
    EnterpriseUserKBQuerySettingsRecord,
    EnterpriseTenantKBACLRecord,
    EnterpriseTenantMembershipRecord,
    EnterpriseTenantRecord,
    EnterpriseTenantUserKBOverrideRecord,
    IdempotencyKeyConflictError,
    InvalidTenantUserKBOverrideError,
    InvalidJobTransitionError,
    JobRecord,
    KBACLRecord,
    KBLifecycleConflictError,
    MetadataConflictError,
    MetadataRecordNotFoundError,
    SQLiteMetadataStore,
)

pytestmark = pytest.mark.offline

_POSTGRES_DSN = (
    os.getenv("LIGHTRAG_KB_POSTGRES_TEST_DSN")
    or os.getenv("POSTGRES_TEST_DSN")
)


async def _make_store(backend: str, tmp_path) -> Any:
    if backend == "sqlite":
        store = SQLiteMetadataStore(tmp_path / "metadata.sqlite3")
        await store.initialize()
        return store
    # postgres
    from lightrag.api.postgres_metadata_store import PostgresMetadataStore

    store = PostgresMetadataStore(
        dsn=_POSTGRES_DSN,
        min_size=1,
        max_size=1,
        operation_lock_pool_max_size=4,
    )
    await store.initialize()
    return store


@pytest.fixture(params=["sqlite", "postgres"])
async def store(request, tmp_path):
    backend = request.param
    if backend == "postgres" and not _POSTGRES_DSN:
        pytest.skip(
            "live PostgreSQL contract test skipped: set "
            "LIGHTRAG_KB_POSTGRES_TEST_DSN to enable"
        )
    created_kb_ids: list[str] = []
    instance = await _make_store(backend, tmp_path)
    instance._test_created_kb_ids = created_kb_ids  # type: ignore[attr-defined]
    try:
        yield instance
    finally:
        # Postgres is potentially shared: purge everything we created.
        if backend == "postgres":
            for kb_id in created_kb_ids:
                try:
                    await instance.purge_kb_metadata(kb_id)
                except Exception:
                    pass
        await instance.close()


def _unique_kb(store) -> str:
    kb_id = f"kb_ct_{uuid.uuid4().hex[:10]}"
    store._test_created_kb_ids.append(kb_id)
    return kb_id


def _doc(kb_id: str, doc_id: str, *, source_key: str | None = None, status: str = "uploaded") -> DocumentRecord:
    now = utc_now_iso()
    metadata: dict = {}
    if source_key is not None:
        metadata["source_key"] = source_key
    return DocumentRecord(
        id=doc_id,
        kb_id=kb_id,
        workspace=f"ws_{kb_id}",
        lightrag_doc_id=None,
        source_type="upload",
        source_name=f"{doc_id}.pdf",
        source_uri=f"/inputs/{doc_id}.pdf",
        source_hash="sha256:src",
        content_type="application/pdf",
        size_bytes=10,
        parser_hash=None,
        index_hash=None,
        status=status,
        enabled=True,
        archived=False,
        chunks_count=None,
        entity_count=None,
        relation_count=None,
        error_code=None,
        error_message=None,
        metadata=metadata,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )


def _job(
    kb_id: str,
    job_id: str,
    *,
    job_type: str = "parse",
    document_id: str | None = None,
    status: str = "queued",
    idempotency_key: str | None = None,
    max_retries: int = 3,
) -> JobRecord:
    now = utc_now_iso()
    return JobRecord(
        id=job_id,
        kb_id=kb_id,
        workspace=f"ws_{kb_id}",
        batch_id=None,
        document_id=document_id,
        job_type=job_type,
        status=status,
        stage=None,
        progress=0.0,
        total_items=1,
        completed_items=0,
        failed_items=0,
        idempotency_key=idempotency_key,
        config_version_id=None,
        config_hash=None,
        retry_count=0,
        max_retries=max_retries,
        payload={"idempotency_fingerprint": "v1"},
        result=None,
        error_code=None,
        error_message=None,
        created_at=now,
        updated_at=now,
        queued_at=now,
        started_at=None,
        finished_at=None,
        cancelled_at=None,
    )


def _enterprise_user(username: str) -> EnterpriseUserRecord:
    now = utc_now_iso()
    return EnterpriseUserRecord(
        id=f"usr_{uuid.uuid4().hex[:10]}",
        username=username,
        password_hash="{bcrypt}$2b$12$placeholderplaceholderplaceholderplaceholderplace",
        system_role="user",
        status="active",
        tenant_id=None,
        can_create_kb=False,
        can_use_bypass_query=False,
        token_version=1,
        metadata={},
        created_at=now,
        updated_at=now,
    )


def _membership(
    user_id: str, tenant_id: str, *, role: str = "tenant_member"
) -> EnterpriseTenantMembershipRecord:
    now = utc_now_iso()
    return EnterpriseTenantMembershipRecord(
        tenant_id=tenant_id,
        user_id=user_id,
        role=role,
        granted_by="usr_admin",
        created_at=now,
        updated_at=now,
    )


def _user_cas(user: EnterpriseUserRecord) -> dict[str, Any]:
    return {
        "expected_updated_at": user.updated_at,
        "expected_token_version": user.token_version,
        "expected_tenant_id": user.tenant_id,
    }


def _enterprise_api_key(kb_id: str) -> EnterpriseAPIKeyRecord:
    now = utc_now_iso()
    return EnterpriseAPIKeyRecord(
        id=f"svc_key_{uuid.uuid4().hex[:10]}",
        name="contract-key",
        key_hash=f"sha256:{uuid.uuid4().hex}",
        key_preview="abc123",
        status="active",
        created_by=None,
        tenant_id=None,
        scopes={"kb_roles": {kb_id: "kb_viewer"}, "can_use_bypass_query": False},
        metadata={"purpose": "contract"},
        created_at=now,
        updated_at=now,
        last_used_at=None,
        revoked_at=None,
        revoked_by=None,
        expires_at="2099-01-01T00:00:00+00:00",
    )


def _enterprise_invitation(*, expires_at: str | None = None) -> EnterpriseInvitationRecord:
    now = utc_now_iso()
    return EnterpriseInvitationRecord(
        id=f"inv_{uuid.uuid4().hex[:10]}",
        token_hash=f"sha256:{uuid.uuid4().hex}",
        token_preview="inv123",
        status="active",
        created_by="admin",
        expires_at=expires_at,
        used_by=None,
        used_at=None,
        metadata={"purpose": "contract"},
        created_at=now,
        updated_at=now,
    )


async def test_enterprise_metadata_contract(store):
    kb_id = _unique_kb(store)
    user = await store.upsert_enterprise_user(_enterprise_user("alice"))

    by_username = await store.get_enterprise_user_by_username("alice")
    by_id = await store.get_enterprise_user_by_id(user.id)
    assert by_username is not None
    assert by_username.id == user.id
    assert by_id is not None
    assert by_id.username == "alice"

    updated_user = EnterpriseUserRecord(
        **{
            **user.to_dict(),
            "can_create_kb": True,
            "can_use_bypass_query": True,
            "can_use_agent_query": True,
            "token_version": user.token_version + 1,
            "updated_at": utc_now_iso(),
        }
    )
    saved_user = await store.upsert_enterprise_user(updated_user, **_user_cas(user))
    assert saved_user.can_create_kb is True
    assert saved_user.can_use_bypass_query is True
    assert saved_user.can_use_agent_query is True
    assert saved_user.token_version == 2

    await store.set_enterprise_system_setting(
        "registration_enabled", "true", updated_by=user.id
    )
    assert await store.get_enterprise_system_setting("registration_enabled") == "true"
    assert await store.get_enterprise_system_setting("missing", "fallback") == "fallback"

    assert await store.get_enterprise_user_kb_query_settings(user.id, kb_id) is None
    first_settings = await store.upsert_enterprise_user_kb_query_settings(
        EnterpriseUserKBQuerySettingsRecord(
            user_id=user.id,
            kb_id=kb_id,
            user_prompt="answer in Chinese",
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
        )
    )
    assert first_settings.user_prompt == "answer in Chinese"
    fetched_settings = await store.get_enterprise_user_kb_query_settings(
        user.id, kb_id
    )
    assert fetched_settings is not None
    assert fetched_settings.user_prompt == "answer in Chinese"
    updated_settings = await store.upsert_enterprise_user_kb_query_settings(
        EnterpriseUserKBQuerySettingsRecord(
            user_id=user.id,
            kb_id=kb_id,
            user_prompt="",
            created_at=first_settings.created_at,
            updated_at=utc_now_iso(),
        )
    )
    assert updated_settings.user_prompt == ""
    assert await store.delete_enterprise_user_kb_query_settings(user.id, kb_id) is True
    assert await store.get_enterprise_user_kb_query_settings(user.id, kb_id) is None
    assert await store.delete_enterprise_user_kb_query_settings(user.id, kb_id) is False

    now = utc_now_iso()
    acl = await store.upsert_kb_acl(
        KBACLRecord(
            kb_id=kb_id,
            user_id=user.id,
            role="viewer",
            granted_by=user.id,
            created_at=now,
            updated_at=now,
        )
    )
    assert acl.role == "viewer"
    assert await store.get_kb_acl_role(kb_id, user.id) == "viewer"
    assert await store.list_kb_ids_for_user(user.id) == [kb_id]
    assert [item.user_id for item in await store.list_kb_acl(kb_id)] == [user.id]

    updated_acl = await store.upsert_kb_acl(
        KBACLRecord(
            kb_id=kb_id,
            user_id=user.id,
            role="editor",
            granted_by=user.id,
            created_at=acl.created_at,
            updated_at=utc_now_iso(),
        )
    )
    assert updated_acl.role == "editor"
    assert await store.get_kb_acl_role(kb_id, user.id) == "editor"

    event = await store.append_audit_event(
        AuditEventRecord(
            id=f"audit_{uuid.uuid4().hex[:10]}",
            event_type="kb_acl_granted",
            actor_user_id=user.id,
            target_type="kb",
            target_id=kb_id,
            metadata={"role": "editor"},
            created_at=utc_now_iso(),
        )
    )
    events = await store.list_audit_events(limit=10)
    assert events[0].id == event.id
    assert events[0].metadata == {"role": "editor"}

    other = await store.append_audit_event(
        AuditEventRecord(
            id=f"audit_{uuid.uuid4().hex[:10]}",
            event_type="kb_acl_revoked",
            actor_user_id=user.id,
            target_type="kb",
            target_id=kb_id,
            metadata={},
            created_at=utc_now_iso(),
        )
    )
    # Filter by event_type (scoped to this test's unique actor).
    granted = await store.list_audit_events(
        event_type="kb_acl_granted", actor_user_id=user.id
    )
    assert {e.id for e in granted} == {event.id}
    # Filter by actor + target returns both events for this test.
    by_actor = await store.list_audit_events(actor_user_id=user.id, target_id=kb_id)
    assert {e.id for e in by_actor} == {event.id, other.id}
    assert all(
        e.actor_user_id == user.id and e.target_id == kb_id for e in by_actor
    )
    # Pagination over the (newest-first) matches.
    first = await store.list_audit_events(limit=1, actor_user_id=user.id, target_id=kb_id)
    second = await store.list_audit_events(
        limit=1, offset=1, actor_user_id=user.id, target_id=kb_id
    )
    assert len(first) == 1 and len(second) == 1
    assert first[0].id != second[0].id

    assert await store.delete_kb_acl(kb_id, user.id) is True
    assert await store.get_kb_acl_role(kb_id, user.id) is None


async def test_download_default_audit_tenant_filter_and_transaction_contract(store):
    tenant_id = f"tenant-{uuid.uuid4().hex[:10]}"
    raw_user = _enterprise_user(f"governance_{uuid.uuid4().hex[:10]}")
    assert raw_user.can_download_files is False
    user = EnterpriseUserRecord(**{**raw_user.to_dict(), "tenant_id": tenant_id})
    saved, membership = await store.upsert_enterprise_user_with_membership(
        user, _membership(user.id, tenant_id, role="tenant_admin")
    )
    assert saved.can_download_files is False
    assert saved.tenant_id == tenant_id
    assert membership is not None and membership.role == "tenant_admin"
    assert [
        item.tenant_id for item in await store.list_user_tenant_memberships(user.id)
    ] == [tenant_id]

    explicit_null = await store.append_audit_event(
        AuditEventRecord(
            id=f"audit_{uuid.uuid4().hex}",
            event_type="governance_explicit_null",
            actor_user_id=user.id,
            target_type="user",
            target_id=user.id,
            metadata={},
            created_at=utc_now_iso(),
            actor_tenant_id=None,
        )
    )
    inferred = await AuditService(store).append(
        "governance_inferred",
        actor_user_id=user.id,
        target_type="user",
        target_id=user.id,
    )
    explicit = await store.append_audit_event(
        AuditEventRecord(
            id=f"audit_{uuid.uuid4().hex}",
            event_type="governance_explicit",
            actor_user_id=user.id,
            target_type="user",
            target_id=user.id,
            metadata={},
            created_at=utc_now_iso(),
            actor_tenant_id="tenant-other-snapshot",
        )
    )
    assert explicit_null.actor_tenant_id is None
    assert inferred.actor_tenant_id == tenant_id
    assert explicit.actor_tenant_id == "tenant-other-snapshot"
    tenant_events = await store.list_audit_events(
        actor_user_id=user.id, actor_tenant_id=tenant_id
    )
    assert {item.id for item in tenant_events} == {inferred.id}

    # Racing assignments serialize/contend on the user and the database unique
    # invariant guarantees a single canonical membership.
    race_tenants = [
        f"tenant-race-a-{uuid.uuid4().hex[:6]}",
        f"tenant-race-b-{uuid.uuid4().hex[:6]}",
    ]
    await asyncio.gather(
        *(
            store.upsert_tenant_membership(_membership(user.id, race_tenant))
            for race_tenant in race_tenants
        )
    )
    raced_user = await store.get_enterprise_user_by_id(user.id)
    raced_memberships = await store.list_user_tenant_memberships(user.id)
    assert raced_user is not None
    assert len(raced_memberships) == 1
    assert raced_memberships[0].tenant_id == raced_user.tenant_id
    assert raced_user.tenant_id in race_tenants
    # Audit scope is an event-time snapshot, not the actor's current tenant.
    assert {
        item.id
        for item in await store.list_audit_events(actor_tenant_id=tenant_id)
        if item.actor_user_id == user.id
    } == {inferred.id}


async def test_audit_service_resolves_omitted_tenant_once_and_preserves_null(
    store, monkeypatch
):
    tenant_a = f"tenant-audit-a-{uuid.uuid4().hex[:8]}"
    tenant_b = f"tenant-audit-b-{uuid.uuid4().hex[:8]}"
    raw = _enterprise_user(f"audit_once_{uuid.uuid4().hex[:10]}")
    actor, membership = await store.upsert_enterprise_user_with_membership(
        EnterpriseUserRecord(**{**raw.to_dict(), "tenant_id": tenant_a}),
        _membership(raw.id, tenant_a),
    )
    assert membership is not None

    original_append = store.append_audit_event
    moved = False

    async def move_after_event_construction(event):
        nonlocal moved
        if not moved and event.event_type == "background_snapshot":
            moved = True
            assert event.actor_tenant_id == tenant_a
            current = await store.get_enterprise_user_by_id(actor.id)
            current_membership = await store.get_tenant_membership(
                tenant_a, actor.id
            )
            assert current is not None
            assert current_membership is not None
            moved_at = utc_now_iso()
            await store.upsert_enterprise_user_with_membership(
                EnterpriseUserRecord(
                    **{
                        **current.to_dict(),
                        "tenant_id": tenant_b,
                        "updated_at": moved_at,
                    }
                ),
                _membership(actor.id, tenant_b),
                expected_membership=current_membership,
                **_user_cas(current),
            )
        return await original_append(event)

    monkeypatch.setattr(store, "append_audit_event", move_after_event_construction)
    audit_service = AuditService(store)
    inferred = await audit_service.append(
        "background_snapshot",
        actor_user_id=actor.id,
    )
    explicit_null = await audit_service.append(
        "explicit_null_snapshot",
        actor_user_id=actor.id,
        actor_tenant_id=None,
    )

    assert inferred.actor_tenant_id == tenant_a
    assert explicit_null.actor_tenant_id is None
    current = await store.get_enterprise_user_by_id(actor.id)
    assert current is not None and current.tenant_id == tenant_b


async def test_stale_enterprise_user_replay_conflicts_and_preserves_new_state(store):
    tenant_a = f"tenant-stale-a-{uuid.uuid4().hex[:8]}"
    tenant_b = f"tenant-stale-b-{uuid.uuid4().hex[:8]}"
    raw = _enterprise_user(f"stale_{uuid.uuid4().hex[:10]}")
    initial, _ = await store.upsert_enterprise_user_with_membership(
        EnterpriseUserRecord(**{**raw.to_dict(), "tenant_id": tenant_a}),
        _membership(raw.id, tenant_a),
    )
    snapshot_a = await store.get_enterprise_user_by_id(initial.id)
    assert snapshot_a is not None

    writer_b = EnterpriseUserRecord(
        **{
            **snapshot_a.to_dict(),
            "password_hash": "writer-b-password-hash",
            "status": "disabled",
            "tenant_id": tenant_b,
            "can_create_kb": True,
            "can_use_bypass_query": True,
            "can_delete_documents": True,
            "can_use_agent_query": True,
            "can_download_files": True,
            "token_version": snapshot_a.token_version + 1,
            "metadata": {"writer": "b"},
            "updated_at": utc_now_iso(),
        }
    )
    saved_b, membership_b = await store.upsert_enterprise_user_with_membership(
        writer_b,
        _membership(snapshot_a.id, tenant_b, role="tenant_admin"),
        **_user_cas(snapshot_a),
    )
    assert membership_b is not None and membership_b.tenant_id == tenant_b

    stale_profile = EnterpriseUserRecord(
        **{
            **snapshot_a.to_dict(),
            "metadata": {"display_name": "stale writer A"},
            "updated_at": utc_now_iso(),
        }
    )
    with pytest.raises(MetadataConflictError):
        await store.upsert_enterprise_user(
            stale_profile,
            **_user_cas(snapshot_a),
        )

    stale_security = EnterpriseUserRecord(
        **{
            **snapshot_a.to_dict(),
            "password_hash": "stale-security-password-hash",
            "status": "active",
            "token_version": snapshot_a.token_version + 1,
            "updated_at": utc_now_iso(),
        }
    )
    with pytest.raises(MetadataConflictError):
        await store.upsert_enterprise_user(
            stale_security,
            **_user_cas(snapshot_a),
        )

    # Even a current revision cannot use the generic whole-record API to move
    # tenants; assignment is exclusive to the transactional membership API.
    illegal_move = EnterpriseUserRecord(
        **{**saved_b.to_dict(), "tenant_id": tenant_a, "updated_at": utc_now_iso()}
    )
    with pytest.raises(MetadataConflictError):
        await store.upsert_enterprise_user(illegal_move, **_user_cas(saved_b))

    current = await store.get_enterprise_user_by_id(snapshot_a.id)
    assert current == saved_b
    memberships = await store.list_user_tenant_memberships(snapshot_a.id)
    assert [(item.tenant_id, item.role) for item in memberships] == [
        (tenant_b, "tenant_admin")
    ]


async def test_tenant_user_kb_override_and_membership_cleanup_contract(store):
    kb_id = _unique_kb(store)
    tenant_a = f"tenant-a-{uuid.uuid4().hex[:8]}"
    tenant_b = f"tenant-b-{uuid.uuid4().hex[:8]}"
    raw_user = _enterprise_user(f"override_{uuid.uuid4().hex[:10]}")
    user_a = EnterpriseUserRecord(**{**raw_user.to_dict(), "tenant_id": tenant_a})
    saved, _ = await store.upsert_enterprise_user_with_membership(
        user_a, _membership(user_a.id, tenant_a)
    )
    now = utc_now_iso()
    allowed = await store.upsert_tenant_user_kb_override(
        EnterpriseTenantUserKBOverrideRecord(
            tenant_id=tenant_a,
            kb_id=kb_id,
            user_id=saved.id,
            effect="allow",
            role="kb_editor",
            granted_by="usr_admin",
            created_at=now,
            updated_at=now,
        )
    )
    assert allowed.effect == "allow" and allowed.role == "kb_editor"
    fetched = await store.get_tenant_user_kb_override(tenant_a, kb_id, saved.id)
    assert fetched == allowed
    assert await store.list_tenant_user_kb_overrides(tenant_a, kb_id) == [allowed]
    assert await store.list_user_tenant_kb_overrides(
        saved.id, tenant_ids=[tenant_a], kb_id=kb_id
    ) == [allowed]
    assert await store.list_tenant_user_kb_overrides_for_user(
        saved.id, tenant_ids=[tenant_a]
    ) == [allowed]

    # Non-tenant user updates and same-tenant role changes must not erase the
    # canonical tenant's overrides; only leaving the tenant does.
    saved = await store.upsert_enterprise_user(
        EnterpriseUserRecord(
            **{
                **saved.to_dict(),
                "can_create_kb": True,
                "updated_at": utc_now_iso(),
            }
        ),
        **_user_cas(saved),
    )
    await store.upsert_tenant_membership(
        _membership(saved.id, tenant_a, role="tenant_admin")
    )
    assert await store.get_tenant_user_kb_override(tenant_a, kb_id, saved.id) == allowed
    saved = await store.get_enterprise_user_by_id(saved.id)
    assert saved is not None

    with pytest.raises(InvalidTenantUserKBOverrideError):
        await store.upsert_tenant_user_kb_override(
            EnterpriseTenantUserKBOverrideRecord(
                tenant_id=tenant_a,
                kb_id=kb_id,
                user_id=saved.id,
                effect="deny",
                role="kb_viewer",
                granted_by=None,
                created_at=now,
                updated_at=now,
            )
        )

    denied = await store.delete_tenant_user_kb_override(
        tenant_a, kb_id, saved.id, granted_by="usr_admin"
    )
    assert denied.effect == "deny" and denied.role is None
    assert denied.created_at == allowed.created_at
    assert await store.reset_tenant_user_kb_override(
        tenant_a, kb_id, saved.id
    ) is True
    assert await store.reset_tenant_user_kb_override(
        tenant_a, kb_id, saved.id
    ) is False

    await store.upsert_tenant_user_kb_override(allowed)
    user_b = EnterpriseUserRecord(
        **{**saved.to_dict(), "tenant_id": tenant_b, "updated_at": utc_now_iso()}
    )
    moved, moved_membership = await store.upsert_enterprise_user_with_membership(
        user_b,
        _membership(saved.id, tenant_b, role="tenant_member"),
        **_user_cas(saved),
    )
    assert moved.tenant_id == tenant_b
    assert moved_membership is not None
    assert await store.get_tenant_user_kb_override(tenant_a, kb_id, saved.id) is None
    memberships = await store.list_user_tenant_memberships(saved.id)
    assert [(item.tenant_id, item.role) for item in memberships] == [
        (tenant_b, "tenant_member")
    ]

    await store.upsert_tenant_user_kb_override(
        EnterpriseTenantUserKBOverrideRecord(
            **{
                **allowed.to_dict(),
                "tenant_id": tenant_b,
                "updated_at": utc_now_iso(),
            }
        )
    )
    assert await store.delete_tenant_membership(tenant_b, saved.id) is True
    cleared = await store.get_enterprise_user_by_id(saved.id)
    assert cleared is not None and cleared.tenant_id is None
    assert await store.list_user_tenant_kb_overrides(saved.id) == []


async def test_override_target_snapshot_cas_blocks_all_stale_writes(
    store, tmp_path
):
    """A second store cannot use a stale route-admission target snapshot."""

    if isinstance(store, SQLiteMetadataStore):
        peer = SQLiteMetadataStore(store.db_path)
        await peer.initialize()
    else:
        from lightrag.api.postgres_metadata_store import PostgresMetadataStore

        peer = PostgresMetadataStore(dsn=_POSTGRES_DSN)
        await peer.initialize()

    created_user_ids: list[str] = []
    try:
        for race in ("promote", "move", "revision", "token"):
            for operation in ("upsert", "deny", "reset"):
                kb_id = _unique_kb(store)
                tenant_a = f"tenant-cas-a-{uuid.uuid4().hex[:8]}"
                tenant_b = f"tenant-cas-b-{uuid.uuid4().hex[:8]}"
                raw = _enterprise_user(
                    f"override_cas_{race}_{operation}_{uuid.uuid4().hex[:8]}"
                )
                initial_user = EnterpriseUserRecord(
                    **{**raw.to_dict(), "tenant_id": tenant_a}
                )
                saved, saved_membership = (
                    await store.upsert_enterprise_user_with_membership(
                        initial_user,
                        _membership(raw.id, tenant_a),
                    )
                )
                created_user_ids.append(saved.id)
                assert saved_membership is not None
                snapshot_user = await store.get_enterprise_user_by_id(saved.id)
                snapshot_membership = await store.get_tenant_membership(
                    tenant_a, saved.id
                )
                assert snapshot_user is not None
                assert snapshot_membership is not None

                baseline = await store.upsert_tenant_user_kb_override(
                    EnterpriseTenantUserKBOverrideRecord(
                        tenant_id=tenant_a,
                        kb_id=kb_id,
                        user_id=saved.id,
                        effect="allow",
                        role="kb_viewer",
                        granted_by="usr_admin",
                        created_at=utc_now_iso(),
                        updated_at=utc_now_iso(),
                    )
                )

                if race == "promote":
                    await peer.upsert_tenant_membership(
                        _membership(saved.id, tenant_a, role="tenant_admin")
                    )
                elif race == "move":
                    current_user = await peer.get_enterprise_user_by_id(saved.id)
                    current_membership = await peer.get_tenant_membership(
                        tenant_a, saved.id
                    )
                    assert current_user is not None
                    assert current_membership is not None
                    moved_at = utc_now_iso()
                    await peer.upsert_enterprise_user_with_membership(
                        EnterpriseUserRecord(
                            **{
                                **current_user.to_dict(),
                                "tenant_id": tenant_b,
                                "updated_at": moved_at,
                            }
                        ),
                        _membership(saved.id, tenant_b),
                        expected_membership=current_membership,
                        **_user_cas(current_user),
                    )
                else:
                    current_user = await peer.get_enterprise_user_by_id(saved.id)
                    assert current_user is not None
                    changes: dict[str, Any]
                    if race == "revision":
                        changes = {
                            "can_create_kb": not current_user.can_create_kb,
                            "updated_at": utc_now_iso(),
                        }
                    else:
                        # Keep updated_at unchanged so token_version is itself a
                        # required part of the target snapshot.
                        changes = {
                            "token_version": current_user.token_version + 1,
                        }
                    await peer.upsert_enterprise_user(
                        EnterpriseUserRecord(
                            **{**current_user.to_dict(), **changes}
                        ),
                        **_user_cas(current_user),
                    )

                target_cas = {
                    "expected_user": snapshot_user,
                    "expected_membership": snapshot_membership,
                }
                with pytest.raises(MetadataConflictError) as conflict:
                    if operation == "upsert":
                        await store.upsert_tenant_user_kb_override(
                            EnterpriseTenantUserKBOverrideRecord(
                                **{
                                    **baseline.to_dict(),
                                    "role": "kb_editor",
                                    "updated_at": utc_now_iso(),
                                }
                            ),
                            **target_cas,
                        )
                    elif operation == "deny":
                        await store.delete_tenant_user_kb_override(
                            tenant_a,
                            kb_id,
                            saved.id,
                            granted_by="usr_admin",
                            **target_cas,
                        )
                    else:
                        await store.reset_tenant_user_kb_override(
                            tenant_a,
                            kb_id,
                            saved.id,
                            **target_cas,
                        )
                assert (
                    conflict.value.entity_type
                    == "tenant_user_kb_override_target"
                )

                current_override = await store.get_tenant_user_kb_override(
                    tenant_a, kb_id, saved.id
                )
                if race == "move":
                    assert current_override is None
                else:
                    assert current_override == baseline
    finally:
        for user_id in created_user_ids:
            try:
                await store.delete_enterprise_user(user_id)
            except Exception:
                pass
        await peer.close()


async def test_override_cleanup_on_user_and_tenant_delete_contract(store):
    kb_id = _unique_kb(store)
    tenant_id = f"tenant-delete-{uuid.uuid4().hex[:8]}"
    now = utc_now_iso()
    await store.upsert_enterprise_tenant(
        EnterpriseTenantRecord(
            id=tenant_id,
            name="Delete cleanup tenant",
            description=None,
            status="active",
            metadata={},
            created_by="usr_admin",
            created_at=now,
            updated_at=now,
        )
    )
    raw_user = _enterprise_user(f"tenant_delete_{uuid.uuid4().hex[:8]}")
    user, _ = await store.upsert_enterprise_user_with_membership(
        EnterpriseUserRecord(**{**raw_user.to_dict(), "tenant_id": tenant_id}),
        _membership(raw_user.id, tenant_id),
    )

    def override() -> EnterpriseTenantUserKBOverrideRecord:
        return EnterpriseTenantUserKBOverrideRecord(
            tenant_id=tenant_id,
            kb_id=kb_id,
            user_id=user.id,
            effect="allow",
            role="kb_viewer",
            granted_by="usr_admin",
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
        )

    await store.upsert_tenant_user_kb_override(override())
    await store.upsert_tenant_kb_acl(
        EnterpriseTenantKBACLRecord(
            tenant_id=tenant_id,
            kb_id=kb_id,
            role="kb_viewer",
            granted_by="usr_admin",
            created_at=now,
            updated_at=now,
        )
    )
    assert await store.delete_enterprise_tenant(tenant_id) is True
    retained_user = await store.get_enterprise_user_by_id(user.id)
    assert retained_user is not None and retained_user.tenant_id is None
    assert await store.list_user_tenant_memberships(user.id) == []
    assert await store.list_user_tenant_kb_overrides(user.id) == []
    assert await store.get_tenant_kb_acl_role(tenant_id, kb_id) is None

    second_tenant = f"tenant-user-delete-{uuid.uuid4().hex[:8]}"
    moved_user = EnterpriseUserRecord(
        **{
            **retained_user.to_dict(),
            "tenant_id": second_tenant,
            "updated_at": utc_now_iso(),
        }
    )
    await store.upsert_enterprise_user_with_membership(
        moved_user,
        _membership(user.id, second_tenant),
        **_user_cas(retained_user),
    )
    second_override = override()
    second_override.tenant_id = second_tenant
    await store.upsert_tenant_user_kb_override(second_override)
    assert await store.delete_enterprise_user(user.id) is True
    assert await store.list_user_tenant_kb_overrides(user.id) == []


def _chat_project(user_id: str, name: str) -> ChatProjectRecord:
    now = utc_now_iso()
    return ChatProjectRecord(
        id=f"proj_{uuid.uuid4().hex[:12]}",
        user_id=user_id,
        name=name,
        created_at=now,
        updated_at=now,
    )


def _chat_session(
    user_id: str, project_id: str, name: str, *, context_rounds: int = 1
) -> ChatSessionRecord:
    now = utc_now_iso()
    return ChatSessionRecord(
        id=f"sess_{uuid.uuid4().hex[:12]}",
        project_id=project_id,
        user_id=user_id,
        name=name,
        created_at=now,
        updated_at=now,
        context_rounds=context_rounds,
    )


def _chat_message(
    user_id: str,
    project_id: str,
    session_id: str,
    role: str,
    content: str,
    *,
    metadata: dict | None = None,
) -> ChatMessageRecord:
    return ChatMessageRecord(
        id=f"msg_{uuid.uuid4().hex[:12]}",
        session_id=session_id,
        project_id=project_id,
        user_id=user_id,
        role=role,
        content=content,
        metadata=metadata or {},
        seq=0,
        created_at=utc_now_iso(),
    )


async def test_chat_project_and_session_contract(store):
    owner = await store.upsert_enterprise_user(
        _enterprise_user(f"chat_owner_{uuid.uuid4().hex[:8]}")
    )
    stranger = await store.upsert_enterprise_user(
        _enterprise_user(f"chat_stranger_{uuid.uuid4().hex[:8]}")
    )

    project = await store.create_chat_project(_chat_project(owner.id, "我的项目"))
    fetched = await store.get_chat_project(owner.id, project.id)
    assert fetched is not None
    assert fetched.name == "我的项目"
    # Ownership scoping: another user cannot see/rename/delete it.
    assert await store.get_chat_project(stranger.id, project.id) is None
    assert await store.rename_chat_project(stranger.id, project.id, name="hax") is None
    assert await store.delete_chat_project(stranger.id, project.id) == (False, 0, 0)

    second = await store.create_chat_project(_chat_project(owner.id, "second"))
    projects, total = await store.list_chat_projects(owner.id)
    assert total == 2
    assert {p.id for p in projects} == {project.id, second.id}
    page, page_total = await store.list_chat_projects(owner.id, limit=1, offset=1)
    assert page_total == 2
    assert len(page) == 1
    stranger_projects, stranger_total = await store.list_chat_projects(stranger.id)
    assert stranger_total == 0
    assert stranger_projects == []

    renamed = await store.rename_chat_project(owner.id, project.id, name="改名后")
    assert renamed is not None
    assert renamed.name == "改名后"
    assert renamed.created_at == project.created_at
    assert renamed.updated_at > project.updated_at
    # Rename bumps recency: the renamed project sorts first.
    projects, _ = await store.list_chat_projects(owner.id)
    assert projects[0].id == project.id

    # Sessions: parent project must exist and belong to the same user.
    with pytest.raises(MetadataRecordNotFoundError):
        await store.create_chat_session(_chat_session(owner.id, "proj_missing", "s"))
    with pytest.raises(MetadataRecordNotFoundError):
        await store.create_chat_session(_chat_session(stranger.id, project.id, "s"))

    s1 = await store.create_chat_session(
        _chat_session(owner.id, project.id, "2026-07-10 10:00:00")
    )
    s2 = await store.create_chat_session(
        _chat_session(owner.id, project.id, "自定义会话", context_rounds=-1)
    )
    got = await store.get_chat_session(owner.id, project.id, s1.id)
    assert got is not None
    assert got.name == "2026-07-10 10:00:00"
    assert got.context_rounds == 1
    got_s2 = await store.get_chat_session(owner.id, project.id, s2.id)
    assert got_s2 is not None
    assert got_s2.context_rounds == -1
    assert await store.get_chat_session(stranger.id, project.id, s1.id) is None
    # Session lookups are anchored to the parent project id.
    assert await store.get_chat_session(owner.id, second.id, s1.id) is None

    sessions, total = await store.list_chat_sessions(owner.id, project.id)
    assert total == 2
    assert {s.id for s in sessions} == {s1.id, s2.id}
    page, page_total = await store.list_chat_sessions(
        owner.id, project.id, limit=1, offset=1
    )
    assert page_total == 2
    assert len(page) == 1

    # Name-only update keeps context_rounds untouched.
    renamed_session = await store.update_chat_session(
        owner.id, project.id, s2.id, name="改名会话"
    )
    assert renamed_session is not None
    assert renamed_session.name == "改名会话"
    assert renamed_session.context_rounds == -1
    # Rounds-only update keeps the name untouched.
    rounds_updated = await store.update_chat_session(
        owner.id, project.id, s2.id, context_rounds=5
    )
    assert rounds_updated is not None
    assert rounds_updated.name == "改名会话"
    assert rounds_updated.context_rounds == 5
    # Both fields at once.
    both_updated = await store.update_chat_session(
        owner.id, project.id, s2.id, name="再改名", context_rounds=-1
    )
    assert both_updated is not None
    assert both_updated.name == "再改名"
    assert both_updated.context_rounds == -1
    assert (
        await store.update_chat_session(stranger.id, project.id, s2.id, name="hax")
        is None
    )

    # Messages: append is atomic, assigns consecutive per-session seq values
    # and bumps the session's updated_at (recency) in the same transaction.
    session_before = await store.get_chat_session(owner.id, project.id, s1.id)
    with pytest.raises(MetadataRecordNotFoundError):
        await store.append_chat_messages(
            [_chat_message(stranger.id, project.id, s1.id, "user", "hax")]
        )
    first_batch = await store.append_chat_messages(
        [
            _chat_message(owner.id, project.id, s1.id, "user", "问题一"),
            _chat_message(
                owner.id,
                project.id,
                s1.id,
                "assistant",
                "回答一 [A1]",
                metadata={"references": [{"reference_id": "A1", "kb_id": "kb_x"}]},
            ),
        ]
    )
    assert [m.seq for m in first_batch] == [1, 2]
    second_batch = await store.append_chat_messages(
        [_chat_message(owner.id, project.id, s1.id, "user", "问题二")]
    )
    assert [m.seq for m in second_batch] == [3]
    session_after = await store.get_chat_session(owner.id, project.id, s1.id)
    assert session_after is not None
    assert session_after.updated_at > session_before.updated_at

    messages, total = await store.list_chat_messages(owner.id, project.id, s1.id)
    assert total == 3
    assert [m.seq for m in messages] == [1, 2, 3]
    assert [m.role for m in messages] == ["user", "assistant", "user"]
    assert messages[1].metadata == {
        "references": [{"reference_id": "A1", "kb_id": "kb_x"}]
    }
    page, page_total = await store.list_chat_messages(
        owner.id, project.id, s1.id, limit=2, offset=1
    )
    assert page_total == 3
    assert [m.seq for m in page] == [2, 3]
    stranger_msgs, stranger_total = await store.list_chat_messages(
        stranger.id, project.id, s1.id
    )
    assert stranger_total == 0
    assert stranger_msgs == []

    assert (
        await store.delete_chat_message(
            stranger.id, project.id, s1.id, messages[0].id
        )
        is False
    )
    assert (
        await store.delete_chat_message(owner.id, project.id, s1.id, messages[0].id)
        is True
    )
    _, total_after_delete = await store.list_chat_messages(
        owner.id, project.id, s1.id
    )
    assert total_after_delete == 2

    assert await store.delete_chat_session(owner.id, project.id, s2.id) == (True, 0)
    assert await store.delete_chat_session(owner.id, project.id, s2.id) == (False, 0)
    _, remaining = await store.list_chat_sessions(owner.id, project.id)
    assert remaining == 1

    # Project delete cascades its sessions and their messages.
    assert await store.delete_chat_project(owner.id, project.id) == (True, 1, 2)
    assert await store.get_chat_project(owner.id, project.id) is None
    assert await store.get_chat_session(owner.id, project.id, s1.id) is None
    _, orphan_total = await store.list_chat_messages(owner.id, project.id, s1.id)
    assert orphan_total == 0

    # User delete cascades remaining chat projects/sessions/messages.
    s3 = await store.create_chat_session(_chat_session(owner.id, second.id, "left"))
    await store.append_chat_messages(
        [_chat_message(owner.id, second.id, s3.id, "user", "遗留消息")]
    )
    assert await store.delete_enterprise_user(owner.id) is True
    assert await store.get_chat_project(owner.id, second.id) is None
    assert await store.get_chat_session(owner.id, second.id, s3.id) is None
    _, user_cascade_total = await store.list_chat_messages(
        owner.id, second.id, s3.id
    )
    assert user_cascade_total == 0


async def test_chat_memory_episode_contract(store):
    """Episode-mapping surface backing the graphiti chat memory: watermark,
    covering lookups, backlog scan and scoped deletes behave identically on
    both backends (docs/ChatMemory-zh.md)."""
    owner = await store.upsert_enterprise_user(
        _enterprise_user(f"mem_owner_{uuid.uuid4().hex[:8]}")
    )
    project = await store.create_chat_project(_chat_project(owner.id, "记忆项目"))
    session = await store.create_chat_session(
        _chat_session(owner.id, project.id, "记忆会话")
    )
    saved = await store.append_chat_messages(
        [
            _chat_message(owner.id, project.id, session.id, "user", "问题一"),
            _chat_message(owner.id, project.id, session.id, "assistant", "回答一"),
            _chat_message(owner.id, project.id, session.id, "user", "问题二"),
        ]
    )
    assert [m.seq for m in saved] == [1, 2, 3]

    # Point lookup used by the message-delete hook.
    got = await store.get_chat_message(owner.id, project.id, session.id, saved[1].id)
    assert got is not None
    assert (got.seq, got.role) == (2, "assistant")
    assert (
        await store.get_chat_message("usr_other", project.id, session.id, saved[1].id)
        is None
    )

    # Tail fetch used by compensation.
    tail = await store.list_chat_messages_after_seq(
        owner.id, project.id, session.id, after_seq=1
    )
    assert [m.seq for m in tail] == [2, 3]
    assert (
        await store.list_chat_messages_after_seq(
            owner.id, project.id, session.id, after_seq=3
        )
        == []
    )

    # No episodes yet: watermark 0 and the session shows up as backlog.
    assert await store.get_chat_memory_watermark(owner.id, project.id, session.id) == 0
    backlog = await store.list_chat_memory_backlog()
    entry = next(item for item in backlog if item.session_id == session.id)
    assert (entry.ingested_seq, entry.max_seq) == (0, 3)
    assert (entry.user_id, entry.project_id) == (owner.id, project.id)

    def _episode(uuid_str: str, first_seq: int, last_seq: int):
        from lightrag.api.metadata_store import ChatMemoryEpisodeRecord

        return ChatMemoryEpisodeRecord(
            episode_uuid=uuid_str,
            session_id=session.id,
            project_id=project.id,
            user_id=owner.id,
            first_seq=first_seq,
            last_seq=last_seq,
            created_at=utc_now_iso(),
        )

    await store.record_chat_memory_episode(_episode("ep-1", 1, 2))
    # Upsert semantics on the same uuid.
    await store.record_chat_memory_episode(_episode("ep-1", 1, 2))
    assert await store.get_chat_memory_watermark(owner.id, project.id, session.id) == 2
    entry = next(
        item
        for item in await store.list_chat_memory_backlog()
        if item.session_id == session.id
    )
    assert (entry.ingested_seq, entry.max_seq) == (2, 3)

    await store.record_chat_memory_episode(_episode("ep-2", 3, 3))
    assert await store.get_chat_memory_watermark(owner.id, project.id, session.id) == 3
    assert all(
        item.session_id != session.id
        for item in await store.list_chat_memory_backlog()
    )

    covering = await store.find_chat_memory_episodes_covering(
        owner.id, project.id, session.id, 2
    )
    assert [e.episode_uuid for e in covering] == ["ep-1"]
    assert (covering[0].first_seq, covering[0].last_seq) == (1, 2)
    listed = await store.list_chat_memory_episodes_for_session(
        owner.id, project.id, session.id
    )
    assert [e.episode_uuid for e in listed] == ["ep-1", "ep-2"]
    # Ownership scoping mirrors the chat tables.
    assert (
        await store.find_chat_memory_episodes_covering(
            "usr_other", project.id, session.id, 2
        )
        == []
    )

    # Deleting one episode row lowers the watermark and reopens the backlog.
    assert await store.delete_chat_memory_episodes(["ep-2", "ep-missing"]) == 1
    assert await store.delete_chat_memory_episodes([]) == 0
    assert await store.get_chat_memory_watermark(owner.id, project.id, session.id) == 2

    # Scoped bulk deletes for project/user purges.
    await store.record_chat_memory_episode(_episode("ep-3", 3, 3))
    assert await store.delete_chat_memory_episodes_for_project(project.id) == 2
    assert await store.get_chat_memory_watermark(owner.id, project.id, session.id) == 0
    await store.record_chat_memory_episode(_episode("ep-4", 1, 3))
    assert await store.delete_chat_memory_episodes_for_user(owner.id) == 1
    assert (
        await store.list_chat_memory_episodes_for_session(
            owner.id, project.id, session.id
        )
        == []
    )


async def test_chat_memory_episode_counts_contract(store):
    """Episode-count surface backing the project overview + admin dashboard:
    counts exclude ``noop_`` placeholder rows and scope per user/project."""
    from lightrag.api.metadata_store import ChatMemoryEpisodeRecord

    owner = await store.upsert_enterprise_user(
        _enterprise_user(f"cnt_owner_{uuid.uuid4().hex[:8]}")
    )
    project = await store.create_chat_project(_chat_project(owner.id, "计数项目"))
    session = await store.create_chat_session(
        _chat_session(owner.id, project.id, "会话")
    )

    async def _record(uuid_str, first, last, created_at):
        await store.record_chat_memory_episode(
            ChatMemoryEpisodeRecord(
                episode_uuid=uuid_str,
                session_id=session.id,
                project_id=project.id,
                user_id=owner.id,
                first_seq=first,
                last_seq=last,
                created_at=created_at,
            )
        )

    # Empty project => zero count, null last-ingested.
    count, last_at = await store.count_chat_memory_episodes_for_project(
        owner.id, project.id
    )
    assert (count, last_at) == (0, None)

    await _record("ep-1", 1, 2, "2026-07-10T08:00:00+00:00")
    await _record("ep-2", 3, 4, "2026-07-11T09:00:00+00:00")
    # noop rows advance the watermark but must NOT count as real memories.
    await _record("noop_abc", 5, 5, "2026-07-12T10:00:00+00:00")

    count, last_at = await store.count_chat_memory_episodes_for_project(
        owner.id, project.id
    )
    assert count == 2
    assert last_at == "2026-07-11T09:00:00+00:00"

    # Ownership scoping: a foreign user sees nothing.
    other_count, _ = await store.count_chat_memory_episodes_for_project(
        "usr_other", project.id
    )
    assert other_count == 0

    # Global stats (super-admin dashboard): count + distinct users/projects,
    # noop excluded.
    episodes, users, projects = await store.count_chat_memory_episodes()
    assert episodes >= 2
    assert users >= 1
    assert projects >= 1


async def test_enterprise_tenant_entity_contract(store):
    now = utc_now_iso()
    tid = f"tenant-{uuid.uuid4().hex[:8]}"
    saved = await store.upsert_enterprise_tenant(
        EnterpriseTenantRecord(
            id=tid,
            name="Acme",
            description="desc",
            status="active",
            metadata={"k": "v"},
            created_by="admin",
            created_at=now,
            updated_at=now,
        )
    )
    assert saved.id == tid and saved.name == "Acme"

    got = await store.get_enterprise_tenant_by_id(tid)
    assert got is not None
    assert got.metadata == {"k": "v"}
    assert got.description == "desc"

    updated = await store.upsert_enterprise_tenant(
        EnterpriseTenantRecord(
            id=tid,
            name="Acme2",
            description=None,
            status="disabled",
            metadata={},
            created_by="admin",
            created_at=now,
            updated_at=utc_now_iso(),
        )
    )
    assert updated.name == "Acme2" and updated.status == "disabled"

    listed = await store.list_enterprise_tenants()
    assert tid in {t.id for t in listed}
    assert await store.get_enterprise_tenant_by_id("does-not-exist") is None

    assert await store.delete_enterprise_tenant(tid) is True
    assert await store.get_enterprise_tenant_by_id(tid) is None
    assert await store.delete_enterprise_tenant(tid) is False


async def test_enterprise_tenant_membership_and_kb_acl_contract(store):
    kb_id = _unique_kb(store)
    alice = await store.upsert_enterprise_user(_enterprise_user("tenant-alice"))
    bob = await store.upsert_enterprise_user(_enterprise_user("tenant-bob"))
    now = utc_now_iso()

    alice_membership = await store.upsert_tenant_membership(
        EnterpriseTenantMembershipRecord(
            tenant_id="tenant-a",
            user_id=alice.id,
            role="tenant_admin",
            granted_by=bob.id,
            created_at=now,
            updated_at=now,
        )
    )
    bob_membership = await store.upsert_tenant_membership(
        EnterpriseTenantMembershipRecord(
            tenant_id="tenant-a",
            user_id=bob.id,
            role="tenant_member",
            granted_by=alice.id,
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
        )
    )
    assert alice_membership.role == "tenant_admin"
    assert bob_membership.role == "tenant_member"
    assert [item.user_id for item in await store.list_tenant_memberships("tenant-a")] == [
        alice.id,
        bob.id,
    ]
    assert [item.tenant_id for item in await store.list_user_tenant_memberships(alice.id)] == [
        "tenant-a"
    ]
    fetched_membership = await store.get_tenant_membership("tenant-a", alice.id)
    assert fetched_membership is not None
    assert fetched_membership.role == "tenant_admin"

    updated_membership = await store.upsert_tenant_membership(
        EnterpriseTenantMembershipRecord(
            tenant_id="tenant-a",
            user_id=alice.id,
            role="tenant_owner",
            granted_by=bob.id,
            created_at=alice_membership.created_at,
            updated_at=utc_now_iso(),
        )
    )
    assert updated_membership.created_at == alice_membership.created_at
    assert updated_membership.role == "tenant_owner"

    tenant_acl = await store.upsert_tenant_kb_acl(
        EnterpriseTenantKBACLRecord(
            tenant_id="tenant-a",
            kb_id=kb_id,
            role="kb_viewer",
            granted_by=alice.id,
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
        )
    )
    assert tenant_acl.role == "kb_viewer"
    assert await store.get_tenant_kb_acl_role("tenant-a", kb_id) == "kb_viewer"
    assert await store.list_kb_ids_for_tenants(["tenant-a", "tenant-missing"]) == [kb_id]
    assert [item.tenant_id for item in await store.list_kb_tenant_acl(kb_id)] == [
        "tenant-a"
    ]

    updated_acl = await store.upsert_tenant_kb_acl(
        EnterpriseTenantKBACLRecord(
            tenant_id="tenant-a",
            kb_id=kb_id,
            role="kb_editor",
            granted_by=bob.id,
            created_at=tenant_acl.created_at,
            updated_at=utc_now_iso(),
        )
    )
    assert updated_acl.created_at == tenant_acl.created_at
    assert await store.get_tenant_kb_acl_role("tenant-a", kb_id) == "kb_editor"

    assert await store.delete_tenant_kb_acl("tenant-a", kb_id) is True
    assert await store.get_tenant_kb_acl_role("tenant-a", kb_id) is None
    assert await store.delete_tenant_membership("tenant-a", alice.id) is True
    assert await store.get_tenant_membership("tenant-a", alice.id) is None


async def test_enterprise_api_key_metadata_contract(store):
    kb_id = _unique_kb(store)
    record = await store.create_enterprise_api_key(_enterprise_api_key(kb_id))

    by_hash = await store.get_enterprise_api_key_by_hash(record.key_hash)
    by_id = await store.get_enterprise_api_key_by_id(record.id)
    assert by_hash is not None
    assert by_hash.id == record.id
    assert by_id is not None
    assert by_id.key_preview == "abc123"
    assert by_id.scopes["kb_roles"] == {kb_id: "kb_viewer"}
    assert by_id.expires_at == "2099-01-01T00:00:00+00:00"
    assert "api_key" not in by_id.to_dict()

    listed = await store.list_enterprise_api_keys()
    assert any(item.id == record.id for item in listed)

    used_at = utc_now_iso()
    used = await store.mark_enterprise_api_key_used(record.id, last_used_at=used_at)
    assert used is not None
    assert used.last_used_at == used_at

    revoked_at = utc_now_iso()
    revoked = await store.revoke_enterprise_api_key(
        record.id,
        revoked_by="admin",
        revoked_at=revoked_at,
    )
    assert revoked is not None
    assert revoked.status == "revoked"
    assert revoked.revoked_by == "admin"
    assert revoked.revoked_at == revoked_at

    revoked_by_hash = await store.get_enterprise_api_key_by_hash(record.key_hash)
    assert revoked_by_hash is not None
    assert revoked_by_hash.status == "revoked"
    assert await store.revoke_enterprise_api_key("missing") is None
    assert await store.mark_enterprise_api_key_used("missing") is None


async def test_enterprise_invitation_metadata_contract(store):
    record = await store.create_enterprise_invitation(_enterprise_invitation())

    by_hash = await store.get_enterprise_invitation_by_token_hash(record.token_hash)
    assert by_hash is not None
    assert by_hash.id == record.id
    assert by_hash.status == "active"
    assert by_hash.token_preview == "inv123"

    listed = await store.list_enterprise_invitations()
    assert any(item.id == record.id for item in listed)

    consumed = await store.consume_enterprise_invitation(
        record.token_hash, used_by="usr_consumer"
    )
    assert consumed is not None
    assert consumed.status == "used"
    assert consumed.used_by == "usr_consumer"
    assert consumed.used_at is not None

    # Single-use: a second consume of the same token returns None.
    assert (
        await store.consume_enterprise_invitation(
            record.token_hash, used_by="usr_other"
        )
        is None
    )

    # Revoke only an active invitation; a revoked one cannot be consumed.
    fresh = await store.create_enterprise_invitation(_enterprise_invitation())
    revoked = await store.revoke_enterprise_invitation(fresh.id)
    assert revoked is not None
    assert revoked.status == "revoked"
    assert (
        await store.consume_enterprise_invitation(fresh.token_hash, used_by="usr_x")
        is None
    )
    assert await store.revoke_enterprise_invitation("inv_missing") is None


async def test_count_active_jobs_by_principal_and_tenant(store):
    kb_id = _unique_kb(store)

    def _stamped(status: str, subject_id: str, tenant_id: str | None):
        job = _job(kb_id, f"job_{uuid.uuid4().hex[:10]}", status=status)
        job.payload["_principal"] = {"subject_id": subject_id, "tenant_id": tenant_id}
        return job

    await store.create_job(_stamped("queued", "usr_alice", "t1"))
    await store.create_job(_stamped("running", "usr_alice", "t1"))
    # Terminal jobs and unstamped jobs are not counted as in-flight.
    await store.create_job(_stamped("succeeded", "usr_alice", "t1"))
    await store.create_job(_job(kb_id, f"job_{uuid.uuid4().hex[:10]}", status="queued"))
    await store.create_job(_stamped("queued", "usr_bob", "t2"))

    assert await store.count_active_jobs_for_principal("usr_alice") == 2
    assert await store.count_active_jobs_for_principal("usr_bob") == 1
    assert await store.count_active_jobs_for_principal("usr_nobody") == 0
    assert await store.count_active_jobs_for_tenant("t1") == 2
    assert await store.count_active_jobs_for_tenant("t2") == 1
    assert await store.count_active_jobs_for_tenant("t_none") == 0


async def test_create_documents_and_job_then_read_back(store):
    kb_id = _unique_kb(store)
    doc = _doc(kb_id, "doc_a")
    job = _job(kb_id, "job_a", document_id="doc_a")

    docs, created_job, created = await store.create_documents_and_job([doc], job)
    assert created is True
    assert [d.id for d in docs] == ["doc_a"]
    assert created_job.id == "job_a"

    fetched = await store.get_document(kb_id, "doc_a")
    assert fetched.source_name == "doc_a.pdf"
    fetched_job = await store.get_job(kb_id, "job_a")
    assert fetched_job.status == "queued"
    assert fetched_job.job_type == "parse"


async def test_list_documents_status_and_source_name_filter(store):
    kb_id = _unique_kb(store)
    await store.create_documents_and_job(
        [_doc(kb_id, "doc_a", status="parsed"), _doc(kb_id, "doc_b", status="uploaded")],
        _job(kb_id, "job_list", document_id=None),
    )
    parsed, total_parsed = await store.list_documents(kb_id, status="parsed")
    assert total_parsed == 1
    assert [d.id for d in parsed] == ["doc_a"]

    # source_name ILIKE is case-insensitive and substring-based.
    by_name, total_name = await store.list_documents(kb_id, source_name="DOC_B")
    assert total_name == 1
    assert by_name[0].id == "doc_b"


async def test_idempotency_key_reuse_and_conflict(store):
    kb_id = _unique_kb(store)
    job1 = _job(kb_id, "job_idem_1", document_id="doc_a", idempotency_key="key-1")
    doc = _doc(kb_id, "doc_a")
    _docs, first, created1 = await store.create_documents_and_job([doc], job1)
    assert created1 is True

    # Same key + same fingerprint -> returns the original job, no new creation.
    job2 = _job(kb_id, "job_idem_2", document_id="doc_a", idempotency_key="key-1")
    _docs2, second, created2 = await store.create_documents_and_job([], job2)
    assert created2 is False
    assert second.id == first.id

    # Same key + different fingerprint -> conflict.
    job3 = _job(kb_id, "job_idem_3", document_id="doc_a", idempotency_key="key-1")
    job3.payload = {"idempotency_fingerprint": "v2-different"}
    with pytest.raises(IdempotencyKeyConflictError):
        await store.create_documents_and_job([], job3)


async def test_job_transition_and_invalid_transition(store):
    kb_id = _unique_kb(store)
    await store.create_documents_and_job([_doc(kb_id, "doc_a")], _job(kb_id, "job_t", document_id="doc_a"))

    running = await store.transition_job(kb_id, "job_t", status="running", stage="parsing", progress=0.5)
    assert running.status == "running"
    assert running.stage == "parsing"

    done = await store.transition_job(kb_id, "job_t", status="succeeded", progress=1.0, result={"ok": True})
    assert done.status == "succeeded"
    assert done.result == {"ok": True}

    # succeeded is terminal: any further transition is rejected.
    with pytest.raises(InvalidJobTransitionError):
        await store.transition_job(kb_id, "job_t", status="running")


async def test_update_job_progress_patches_live_without_status_change(store):
    kb_id = _unique_kb(store)
    await store.create_documents_and_job(
        [_doc(kb_id, "doc_a")], _job(kb_id, "job_p", document_id="doc_a")
    )
    await store.transition_job(
        kb_id, "job_p", status="running", stage="building", result={"keep": 1}
    )

    # Live patch progress + completed_items + a shallow-merged result patch,
    # all WITHOUT a status change. running -> running is not a legal transition,
    # so this must NOT go through the state machine.
    patched = await store.update_job_progress(
        kb_id,
        "job_p",
        progress=0.5,
        completed_items=2,
        result_patch={"pipeline": {"latest_message": "Extract entities 1/4"}},
    )
    assert patched.status == "running"  # unchanged
    assert patched.stage == "building"  # preserved (not supplied)
    assert patched.progress == 0.5
    assert patched.completed_items == 2
    assert patched.result == {
        "keep": 1,
        "pipeline": {"latest_message": "Extract entities 1/4"},
    }

    # Omitted fields are preserved; result is left intact when no patch is given.
    patched2 = await store.update_job_progress(kb_id, "job_p", progress=0.75)
    assert patched2.completed_items == 2
    assert patched2.progress == 0.75
    assert patched2.result == {
        "keep": 1,
        "pipeline": {"latest_message": "Extract entities 1/4"},
    }

    # Once terminal, a late progress poll is ignored — it cannot resurrect or
    # overwrite the job's authoritative terminal snapshot.
    await store.transition_job(
        kb_id, "job_p", status="succeeded", progress=1.0, completed_items=4
    )
    ignored = await store.update_job_progress(
        kb_id, "job_p", progress=0.1, completed_items=0
    )
    assert ignored.status == "succeeded"
    assert ignored.progress == 1.0
    assert ignored.completed_items == 4


async def test_retry_resets_job_and_enforces_max_retries(store):
    kb_id = _unique_kb(store)
    await store.create_documents_and_job(
        [_doc(kb_id, "doc_a")],
        _job(kb_id, "job_retry", document_id="doc_a", max_retries=1),
    )
    await store.transition_job(kb_id, "job_retry", status="running")
    await store.transition_job(kb_id, "job_retry", status="failed", error_code="boom")

    retried = await store.reset_job_for_retry(kb_id, "job_retry", new_idempotency_key=None)
    assert retried.status == "queued"
    assert retried.retry_count == 1
    assert retried.error_code is None

    # Second failure then retry exceeds max_retries=1 -> rejected.
    await store.transition_job(kb_id, "job_retry", status="running")
    await store.transition_job(kb_id, "job_retry", status="failed", error_code="boom2")
    with pytest.raises(InvalidJobTransitionError):
        await store.reset_job_for_retry(kb_id, "job_retry", new_idempotency_key=None)


async def test_dead_letter_listing(store):
    kb_id = _unique_kb(store)
    # job_dl: failed + retries exhausted -> dead-letter.
    await store.create_documents_and_job(
        [_doc(kb_id, "doc_a")],
        _job(kb_id, "job_dl", document_id="doc_a", max_retries=0),
    )
    await store.transition_job(kb_id, "job_dl", status="running")
    await store.transition_job(kb_id, "job_dl", status="failed", error_code="boom")

    # job_live: failed but still retriable -> NOT dead-letter.
    await store.create_documents_and_job(
        [_doc(kb_id, "doc_b")],
        _job(kb_id, "job_live", document_id="doc_b", max_retries=3),
    )
    await store.transition_job(kb_id, "job_live", status="running")
    await store.transition_job(kb_id, "job_live", status="failed", error_code="boom")

    dead, total = await store.list_dead_letter_jobs(kb_id)
    dead_ids = {j.id for j in dead}
    assert "job_dl" in dead_ids
    assert "job_live" not in dead_ids
    assert total >= 1


async def test_aggregate_stats_contract(store):
    kb_id = _unique_kb(store)
    doc_ready = _doc(kb_id, "doc_stats_a", status="ready")
    doc_ready.chunks_count = 3
    doc_ready.entity_count = 5
    doc_ready.relation_count = 7
    await store.create_documents_and_job(
        [doc_ready, _doc(kb_id, "doc_stats_b", status="uploaded")],
        _job(kb_id, "job_stats_dl", document_id="doc_stats_a", max_retries=0),
    )
    await store.transition_job(kb_id, "job_stats_dl", status="running")
    await store.transition_job(
        kb_id, "job_stats_dl", status="failed", error_code="boom"
    )

    stats = await store.aggregate_control_plane_stats(kb_id)
    assert stats["documents_by_status"] == {"ready": 1, "uploaded": 1}
    assert stats["document_counters"] == {"chunks": 3, "entities": 5, "relations": 7}
    assert stats["jobs_by_status"] == {"failed": 1}
    assert stats["dead_letter_jobs"] == 1
    assert stats["artifacts"] == 0

    # Global aggregation includes (at least) this KB's rows; exact totals are
    # not asserted so a shared live database stays safe.
    global_stats = await store.aggregate_control_plane_stats(None)
    assert global_stats["documents_by_status"].get("ready", 0) >= 1
    assert global_stats["dead_letter_jobs"] >= 1
    assert global_stats["document_counters"]["chunks"] >= 3

    # Enterprise aggregates: delta-based for the same reason.
    before = await store.aggregate_enterprise_stats()
    await store.upsert_enterprise_user(
        _enterprise_user(f"stats_user_{uuid.uuid4().hex[:8]}")
    )
    after = await store.aggregate_enterprise_stats()
    assert (
        after["users_by_status"].get("active", 0)
        == before["users_by_status"].get("active", 0) + 1
    )
    assert after["tenants"] >= before["tenants"]
    assert after["audit_events"] >= before["audit_events"]


async def test_worker_claim_is_single_winner(store):
    kb_id = _unique_kb(store)
    # Single-document parse job is worker-claimable.
    await store.create_documents_and_job(
        [_doc(kb_id, "doc_a")],
        _job(kb_id, "job_claim", job_type="parse", document_id="doc_a"),
    )
    claimed = await store.claim_next_worker_job(job_types=["parse"])
    assert claimed is not None
    assert claimed.id == "job_claim"
    assert claimed.status == "running"
    # Already claimed -> no second winner.
    again = await store.claim_next_worker_job(job_types=["parse"])
    assert again is None


async def test_job_execution_guard_is_cross_store_nonblocking_and_reentrant(store):
    """Both backends expose one crash-stop owner session per durable job."""

    if isinstance(store, SQLiteMetadataStore):
        peer = SQLiteMetadataStore(store.db_path)
    else:
        from lightrag.api.postgres_metadata_store import PostgresMetadataStore

        peer = PostgresMetadataStore(
            dsn=_POSTGRES_DSN,
            min_size=1,
            max_size=1,
            operation_lock_pool_max_size=1,
        )
    await peer.initialize()
    job_id = f"job_guard_{uuid.uuid4().hex}"
    try:
        async with store.job_execution_guard(job_id) as acquired:
            assert acquired is True
            # Same task + same store/job is explicitly re-entrant.
            async with store.job_execution_guard(job_id, wait=False) as nested:
                assert nested is True
            # A separate store/session sees the live owner and never waits in
            # try mode.
            async with peer.job_execution_guard(job_id, wait=False) as contender:
                assert contender is False

        async with peer.job_execution_guard(job_id, wait=False) as after_release:
            assert after_release is True

        entered = asyncio.Event()
        never_release = asyncio.Event()

        async def cancelled_owner():
            async with store.job_execution_guard(job_id) as owned:
                assert owned is True
                entered.set()
                await never_release.wait()

        task = asyncio.create_task(cancelled_owner())
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        async with peer.job_execution_guard(job_id, wait=False) as after_cancel:
            assert after_cancel is True
    finally:
        await peer.close()


async def test_config_version_create_activate_monotonic(store):
    kb_id = _unique_kb(store)
    now = utc_now_iso()

    def _cfg(version_placeholder: int) -> ConfigVersionRecord:
        return ConfigVersionRecord(
            id=f"cfg_{uuid.uuid4().hex[:8]}",
            kb_id=kb_id,
            workspace=f"ws_{kb_id}",
            version=0,  # server assigns monotonic version
            config={"chunk_config": {"chunk_size": 512}},
            parser_hash="sha256:p",
            index_hash="sha256:i",
            query_hash="sha256:q",
            created_at=now,
            activated_at=None,
            created_by="tester",
        )

    v1 = await store.create_config_version(_cfg(1))
    v2 = await store.create_config_version(_cfg(2))
    assert v1.version == 1
    assert v2.version == 2

    activated = await store.mark_config_version_activated(kb_id, v2.id)
    assert activated.activated_at is not None

    versions, total = await store.list_config_versions(kb_id)
    assert total == 2
    assert {v.version for v in versions} == {1, 2}


async def test_complete_parse_persists_artifacts_and_replaces_on_retry(store):
    kb_id = _unique_kb(store)
    await store.create_documents_and_job(
        [_doc(kb_id, "doc_a")], _job(kb_id, "job_p", document_id="doc_a")
    )
    await store.mark_document_parsing(kb_id, "doc_a", metadata_patch={"current_parse_job_id": "job_p"})

    def _artifact(artifact_type: str, suffix: str) -> ArtifactRecord:
        return ArtifactRecord(
            id=f"art_{uuid.uuid4().hex[:8]}",
            kb_id=kb_id,
            workspace=f"ws_{kb_id}",
            document_id="doc_a",
            artifact_type=artifact_type,
            uri=f"/inputs/doc_a/{suffix}",
            checksum="sha256:c",
            size_bytes=5,
            metadata={"source": "mineru"},
            created_at=utc_now_iso(),
        )

    doc, artifacts = await store.complete_document_parse(
        kb_id,
        "doc_a",
        parser_hash="sha256:parser",
        lightrag_doc_id="doc-lr-1",
        metadata_patch={"parsed": True},
        artifacts=[_artifact("original", "a.pdf"), _artifact("blocks", "a.blocks.jsonl")],
    )
    assert doc.status == "parsed"
    assert doc.parser_hash == "sha256:parser"
    listed, total = await store.list_document_artifacts(kb_id, "doc_a")
    assert total == 2
    assert {a.artifact_type for a in listed} == {"original", "blocks"}

    # Re-parse replaces artifacts wholesale (no stale residue).
    await store.mark_document_parsing(kb_id, "doc_a", metadata_patch={"current_parse_job_id": "job_p"})
    _doc2, _arts2 = await store.complete_document_parse(
        kb_id,
        "doc_a",
        parser_hash="sha256:parser2",
        lightrag_doc_id="doc-lr-1",
        metadata_patch={},
        artifacts=[_artifact("original", "a.pdf")],
    )
    listed2, total2 = await store.list_document_artifacts(kb_id, "doc_a")
    assert total2 == 1
    assert listed2[0].artifact_type == "original"


async def test_complete_document_replace_preserves_provided_source_type(store):
    kb_id = _unique_kb(store)
    await store.create_documents_and_job(
        [_doc(kb_id, "doc_replace")],
        _job(kb_id, "job_replace", job_type="replace", document_id="doc_replace"),
    )

    replaced = await store.complete_document_replace(
        kb_id,
        "doc_replace",
        source_name="imported.pdf",
        source_uri="/inputs/doc_replace/imported.pdf",
        source_type="import",
        source_hash="sha256:replacement",
        content_type="application/pdf",
        size_bytes=42,
        metadata_patch={"source_key": "imports/imported.pdf"},
    )

    assert replaced.source_type == "import"
    assert replaced.source_name == "imported.pdf"
    assert replaced.source_hash == "sha256:replacement"
    assert replaced.status == "uploaded"
    fetched = await store.get_document(kb_id, "doc_replace")
    assert fetched.source_type == "import"


async def test_kb_lifecycle_generation_and_tombstone_contract(store):
    kb_id = _unique_kb(store)
    old_generation = f"gen-old-{uuid.uuid4().hex}"
    new_generation = f"gen-new-{uuid.uuid4().hex}"
    tenant_id = f"tenant-lifecycle-{uuid.uuid4().hex[:8]}"
    raw_user = _enterprise_user(f"lifecycle_{uuid.uuid4().hex[:10]}")
    user, _ = await store.upsert_enterprise_user_with_membership(
        EnterpriseUserRecord(**{**raw_user.to_dict(), "tenant_id": tenant_id}),
        _membership(raw_user.id, tenant_id),
    )
    now = utc_now_iso()
    direct_acl = KBACLRecord(
        kb_id=kb_id,
        user_id=user.id,
        role="kb_editor",
        granted_by="usr_admin",
        created_at=now,
        updated_at=now,
    )
    tenant_acl = EnterpriseTenantKBACLRecord(
        tenant_id=tenant_id,
        kb_id=kb_id,
        role="kb_admin",
        granted_by="usr_admin",
        created_at=now,
        updated_at=now,
    )
    override = EnterpriseTenantUserKBOverrideRecord(
        tenant_id=tenant_id,
        kb_id=kb_id,
        user_id=user.id,
        effect="allow",
        role="kb_viewer",
        granted_by="usr_admin",
        created_at=now,
        updated_at=now,
    )

    # Legacy KBs have no lifecycle row and retain the old write contract.
    assert await store.get_kb_lifecycle(kb_id) is None
    assert await store.assert_current_kb_generation(kb_id, None) is None
    await store.upsert_kb_acl(direct_acl)

    activated = await store.activate_kb_generation(
        kb_id, old_generation, activated_at=now
    )
    assert activated.generation == old_generation
    assert activated.state == "active"
    assert activated.deleted_at is None
    # Registering the same active generation is idempotent and does not rewrite
    # its activation timestamp.
    assert await store.register_kb_generation(
        kb_id, old_generation, activated_at="must-not-replace-activated-at"
    ) == activated
    with pytest.raises(KBLifecycleConflictError):
        await store.activate_kb_generation(kb_id, new_generation)

    await store.upsert_kb_acl(direct_acl, expected_generation=old_generation)
    await store.upsert_tenant_kb_acl(
        tenant_acl, expected_generation=old_generation
    )
    await store.upsert_tenant_user_kb_override(
        override, expected_generation=old_generation
    )
    await store.upsert_enterprise_user_kb_query_settings(
        EnterpriseUserKBQuerySettingsRecord(
            user_id=user.id,
            kb_id=kb_id,
            user_prompt="lifecycle prompt",
            created_at=now,
            updated_at=now,
        )
    )
    old_key = await store.create_enterprise_api_key(
        _enterprise_api_key(kb_id),
        expected_kb_generations={kb_id: old_generation},
    )

    purged = await store.purge_kb_metadata(kb_id, generation=old_generation)
    assert purged["enterprise_kb_acl"] >= 1
    assert purged["enterprise_tenant_kb_acl"] >= 1
    assert purged["enterprise_tenant_user_kb_overrides"] >= 1
    assert purged["enterprise_user_kb_query_settings"] >= 1
    deleted = await store.get_kb_lifecycle(kb_id)
    assert deleted is not None
    assert deleted.generation == old_generation
    assert deleted.state == "deleted"
    assert deleted.deleted_at is not None
    assert await store.get_kb_acl_role(kb_id, user.id) is None
    assert await store.get_tenant_kb_acl_role(tenant_id, kb_id) is None
    assert await store.get_tenant_user_kb_override(tenant_id, kb_id, user.id) is None
    assert await store.get_enterprise_user_kb_query_settings(user.id, kb_id) is None
    stripped_old_key = await store.get_enterprise_api_key_by_id(old_key.id)
    assert stripped_old_key is not None
    assert kb_id not in stripped_old_key.scopes["kb_roles"]

    with pytest.raises(KBLifecycleConflictError):
        await store.upsert_kb_acl(direct_acl, expected_generation=old_generation)
    with pytest.raises(KBLifecycleConflictError):
        await store.upsert_kb_acl(direct_acl)
    with pytest.raises(KBLifecycleConflictError):
        await store.upsert_tenant_kb_acl(
            tenant_acl, expected_generation=old_generation
        )
    with pytest.raises(KBLifecycleConflictError):
        await store.upsert_tenant_user_kb_override(
            override, expected_generation=old_generation
        )
    with pytest.raises(KBLifecycleConflictError):
        await store.create_enterprise_api_key(
            _enterprise_api_key(kb_id),
            expected_kb_generations={kb_id: old_generation},
        )
    with pytest.raises(KBLifecycleConflictError):
        await store.activate_kb_generation(kb_id, old_generation)

    reactivated = await store.activate_kb_generation(kb_id, new_generation)
    assert reactivated.generation == new_generation
    assert reactivated.state == "active"
    assert reactivated.deleted_at is None
    with pytest.raises(KBLifecycleConflictError):
        await store.upsert_kb_acl(direct_acl, expected_generation=old_generation)
    with pytest.raises(KBLifecycleConflictError):
        await store.upsert_tenant_kb_acl(tenant_acl)
    with pytest.raises(KBLifecycleConflictError):
        await store.create_enterprise_api_key(
            _enterprise_api_key(kb_id),
            expected_kb_generations={kb_id: old_generation},
        )
    with pytest.raises(KBLifecycleConflictError):
        await store.create_enterprise_api_key(_enterprise_api_key(kb_id))

    await store.upsert_kb_acl(direct_acl, expected_generation=new_generation)
    await store.upsert_tenant_kb_acl(
        tenant_acl, expected_generation=new_generation
    )
    await store.upsert_tenant_user_kb_override(
        override, expected_generation=new_generation
    )
    new_key = await store.create_enterprise_api_key(
        _enterprise_api_key(kb_id),
        expected_kb_generations={kb_id: new_generation},
    )
    assert await store.get_kb_acl_role(kb_id, user.id) == "kb_editor"
    assert await store.get_tenant_kb_acl_role(tenant_id, kb_id) == "kb_admin"
    assert await store.get_tenant_user_kb_override(
        tenant_id, kb_id, user.id
    ) == override
    assert (await store.get_enterprise_api_key_by_id(new_key.id)) is not None
    assert await store.assert_kb_generation(kb_id, new_generation) == reactivated

    # Keep a live PostgreSQL contract database clean without weakening the
    # required-generation rule for managed KBs.
    await store.purge_kb_metadata(kb_id, generation=new_generation)


async def test_kb_exclusive_lock_and_begin_deletion_are_split(store):
    kb_id = _unique_kb(store)
    generation = f"gen-split-{uuid.uuid4().hex}"
    next_generation = f"gen-split-next-{uuid.uuid4().hex}"
    delete_job_id = f"job_split_{uuid.uuid4().hex[:10]}"
    active = await store.activate_kb_generation(kb_id, generation)

    async with store.kb_exclusive_operation_guard(kb_id):
        assert await store.get_kb_lifecycle(kb_id) == active

    still_active = await store.get_kb_lifecycle(kb_id)
    assert still_active is not None
    assert still_active.state == "active"
    assert still_active.delete_job_id is None

    async with store.kb_exclusive_operation_guard(kb_id):
        deleting = await store.begin_kb_deletion(
            kb_id,
            generation,
            delete_job_id,
        )
        assert deleting.state == "deleting"
        assert deleting.delete_job_id == delete_job_id

    await store.complete_kb_deletion(kb_id, generation, delete_job_id)
    await store.activate_kb_generation(kb_id, next_generation)
    await store.purge_kb_metadata(kb_id, generation=next_generation)


async def test_kb_deletion_fence_strict_purge_and_completion_contract(store):
    kb_id = _unique_kb(store)
    generation = f"gen-delete-{uuid.uuid4().hex}"
    next_generation = f"gen-next-{uuid.uuid4().hex}"
    clear_job_id = f"job_clear_{uuid.uuid4().hex[:10]}"
    other_job_id = f"job_other_{uuid.uuid4().hex[:10]}"
    await store.activate_kb_generation(kb_id, generation)

    user = await store.upsert_enterprise_user(
        _enterprise_user(f"delete_fence_{uuid.uuid4().hex[:8]}")
    )
    now = utc_now_iso()
    await store.upsert_kb_acl(
        KBACLRecord(
            kb_id=kb_id,
            user_id=user.id,
            role="kb_admin",
            granted_by="usr_admin",
            created_at=now,
            updated_at=now,
        ),
        expected_generation=generation,
    )
    await store.upsert_enterprise_user_kb_query_settings(
        EnterpriseUserKBQuerySettingsRecord(
            user_id=user.id,
            kb_id=kb_id,
            user_prompt="must be purged",
            created_at=now,
            updated_at=now,
        )
    )
    api_key = await store.create_enterprise_api_key(
        _enterprise_api_key(kb_id),
        expected_kb_generations={kb_id: generation},
    )
    await store.create_config_version(
        ConfigVersionRecord(
            id=f"cfg_{uuid.uuid4().hex[:10]}",
            kb_id=kb_id,
            workspace=f"ws_{kb_id}",
            version=0,
            config={"chunk_config": {"chunk_size": 128}},
            parser_hash=None,
            index_hash=None,
            query_hash=None,
            created_at=now,
            activated_at=None,
            created_by=user.id,
        )
    )
    await store.create_documents_and_job(
        [_doc(kb_id, "doc_delete_fence")],
        _job(
            kb_id,
            other_job_id,
            document_id="doc_delete_fence",
        ),
    )
    await store.create_job(
        _job(kb_id, clear_job_id, job_type="clear_kb", status="running")
    )

    # An exception releases only the exclusive fence. The durable deleting
    # state/job binding remains so restore/create stay blocked and the same job
    # can retry safely.
    with pytest.raises(RuntimeError, match="injected cleanup failure"):
        async with store.kb_deletion_guard(
            kb_id, generation, clear_job_id
        ) as deleting:
            assert deleting.state == "deleting"
            assert deleting.delete_job_id == clear_job_id
            raise RuntimeError("injected cleanup failure")

    deleting = await store.get_kb_lifecycle(kb_id)
    assert deleting is not None
    assert deleting.state == "deleting"
    assert deleting.generation == generation
    assert deleting.delete_job_id == clear_job_id
    with pytest.raises(KBLifecycleConflictError):
        await store.assert_kb_not_deleting(kb_id, generation)
    with pytest.raises(KBLifecycleConflictError):
        await store.activate_kb_generation(kb_id, generation)
    with pytest.raises(KBLifecycleConflictError):
        await store.activate_kb_generation(kb_id, next_generation)
    with pytest.raises(KBLifecycleConflictError):
        async with store.kb_deletion_guard(
            kb_id, generation, f"job_other_{uuid.uuid4().hex[:8]}"
        ):
            pass
    with pytest.raises(KBLifecycleConflictError):
        async with store.kb_deletion_guard(
            kb_id, "old-generation", clear_job_id
        ):
            pass

    # The strict purge is generation/job bound. A stale generation cannot
    # remove any current metadata, and the successful purge retains only the
    # clear job needed for final status/audit updates.
    async with store.kb_deletion_guard(
        kb_id, generation, clear_job_id
    ) as retry:
        assert retry.state == "deleting"
        with pytest.raises(KBLifecycleConflictError):
            await store.purge_kb_metadata(
                kb_id,
                generation="old-generation",
                delete_job_id=clear_job_id,
            )
        assert (await store.get_document(kb_id, "doc_delete_fence")).id == (
            "doc_delete_fence"
        )
        purged = await store.purge_kb_metadata(
            kb_id,
            generation=generation,
            delete_job_id=clear_job_id,
        )
        assert purged["documents"] == 1
        assert purged["jobs"] == 1
        assert (await store.get_job(kb_id, clear_job_id)).id == clear_job_id
        with pytest.raises(MetadataRecordNotFoundError):
            await store.get_job(kb_id, other_job_id)
        with pytest.raises(MetadataRecordNotFoundError):
            await store.get_document(kb_id, "doc_delete_fence")
        assert await store.get_kb_acl_role(kb_id, user.id) is None
        assert await store.get_enterprise_user_kb_query_settings(user.id, kb_id) is None
        versions, total = await store.list_config_versions(kb_id)
        assert versions == [] and total == 0
        stripped_key = await store.get_enterprise_api_key_by_id(api_key.id)
        assert stripped_key is not None
        assert kb_id not in stripped_key.scopes["kb_roles"]
        still_deleting = await store.get_kb_lifecycle(kb_id)
        assert still_deleting is not None and still_deleting.state == "deleting"

    completed = await store.complete_kb_deletion(
        kb_id, generation, clear_job_id
    )
    assert completed.state == "deleted"
    assert completed.deleted_at is not None
    assert completed.delete_job_id == clear_job_id
    assert (
        await store.complete_kb_deletion(kb_id, generation, clear_job_id)
        == completed
    )
    # Exact deleted retries may acquire the fence for tail-only idempotent work;
    # callers inspect the returned state and must not destroy storage again.
    async with store.kb_deletion_guard(
        kb_id, generation, clear_job_id
    ) as deleted_retry:
        assert deleted_retry == completed
    with pytest.raises(KBLifecycleConflictError):
        async with store.kb_write_guard(kb_id, generation):
            pass

    reactivated = await store.activate_kb_generation(kb_id, next_generation)
    assert reactivated.state == "active"
    assert reactivated.delete_job_id is None
    await store.purge_kb_metadata(kb_id, generation=next_generation)


async def test_kb_operation_guards_allow_business_calls_with_single_pool(store):
    """Guard sessions must not consume the pool used by business operations."""

    kb_id = _unique_kb(store)
    generation = f"gen-guard-business-{uuid.uuid4().hex}"
    next_generation = f"gen-guard-business-next-{uuid.uuid4().hex}"
    document_id = f"doc_guard_{uuid.uuid4().hex[:10]}"
    parse_job_id = f"job_guard_parse_{uuid.uuid4().hex[:10]}"
    delete_job_id = f"job_guard_delete_{uuid.uuid4().hex[:10]}"
    await store.activate_kb_generation(kb_id, generation)

    async with store.kb_write_guard(kb_id, generation) as active:
        assert active is not None and active.state == "active"
        documents, parse_job, created = await store.create_documents_and_job(
            [_doc(kb_id, document_id)],
            _job(kb_id, parse_job_id, document_id=document_id),
        )
        assert created is True
        assert documents[0].id == document_id
        assert parse_job.id == parse_job_id
        await store.create_job(
            _job(kb_id, delete_job_id, job_type="clear_kb", status="running")
        )
        assert (await store.get_document(kb_id, document_id)).id == document_id
        assert (await store.get_job(kb_id, delete_job_id)).id == delete_job_id

    async with store.kb_deletion_guard(
        kb_id, generation, delete_job_id
    ) as deleting:
        assert deleting.state == "deleting"
        assert (await store.get_kb_lifecycle(kb_id)).state == "deleting"
        updated = await store.update_job_payload_patch(
            kb_id,
            delete_job_id,
            payload_patch={"guard_business_write": True},
        )
        assert updated.payload["guard_business_write"] is True
        purged = await store.purge_kb_metadata(
            kb_id,
            generation=generation,
            delete_job_id=delete_job_id,
        )
        assert purged["documents"] == 1
        assert purged["jobs"] == 1
        assert (await store.get_job(kb_id, delete_job_id)).id == delete_job_id

    await store.complete_kb_deletion(kb_id, generation, delete_job_id)
    await store.activate_kb_generation(kb_id, next_generation)
    await store.purge_kb_metadata(kb_id, generation=next_generation)


async def test_sqlite_kb_operation_fence_waits_and_rejects_new_writers(tmp_path):
    db_path = tmp_path / "kb-operation-fence.sqlite3"
    writer_store = SQLiteMetadataStore(db_path)
    deletion_store = SQLiteMetadataStore(db_path)
    await writer_store.initialize()
    await deletion_store.initialize()
    kb_id = "kb-operation-fence"
    generation = "gen-operation-fence"
    delete_job_id = "job-operation-fence"
    await writer_store.activate_kb_generation(kb_id, generation)
    exclusive_entered = asyncio.Event()
    release_exclusive = asyncio.Event()

    async def hold_exclusive() -> None:
        async with deletion_store.kb_deletion_guard(
            kb_id, generation, delete_job_id
        ):
            exclusive_entered.set()
            await release_exclusive.wait()

    try:
        async with writer_store.kb_write_guard(kb_id, generation):
            deletion_task = asyncio.create_task(hold_exclusive())
            await asyncio.sleep(0.1)
            assert not exclusive_entered.is_set()

        await asyncio.wait_for(exclusive_entered.wait(), timeout=5)
        # The exclusive owner has already committed active -> deleting. A new
        # shared entrant is rejected rather than deadlocking behind that owner.
        with pytest.raises(KBLifecycleConflictError):
            async with writer_store.kb_write_guard(kb_id, generation):
                pass
        release_exclusive.set()
        await asyncio.wait_for(deletion_task, timeout=5)
        lifecycle = await writer_store.get_kb_lifecycle(kb_id)
        assert lifecycle is not None and lifecycle.state == "deleting"
    finally:
        release_exclusive.set()
        await writer_store.close()
        await deletion_store.close()


async def test_sqlite_kb_write_guard_reenters_with_waiting_writer(tmp_path):
    db_path = tmp_path / "kb-operation-reentrant.sqlite3"
    writer_store = SQLiteMetadataStore(db_path)
    deletion_store = SQLiteMetadataStore(db_path)
    await writer_store.initialize()
    await deletion_store.initialize()
    kb_id = "kb-operation-reentrant"
    generation = "gen-operation-reentrant"
    other_kb_id = "kb-operation-reentrant-other"
    other_generation = "gen-operation-reentrant-other"
    await writer_store.activate_kb_generation(kb_id, generation)
    await writer_store.activate_kb_generation(other_kb_id, other_generation)
    exclusive_entered = asyncio.Event()
    release_exclusive = asyncio.Event()

    async def hold_exclusive() -> None:
        async with deletion_store.kb_deletion_guard(
            kb_id,
            generation,
            "job-operation-reentrant",
        ):
            exclusive_entered.set()
            await release_exclusive.wait()

    try:
        async with writer_store.kb_write_guard(kb_id, generation):
            deletion_task = asyncio.create_task(hold_exclusive())
            await asyncio.sleep(0.1)
            assert not exclusive_entered.is_set()
            # A writer-preferring local lock would deadlock here without exact
            # same-task/store/KB/generation re-entry.
            async with writer_store.kb_write_guard(kb_id, generation):
                pass

            async def child_reentry() -> None:
                async with writer_store.kb_write_guard(kb_id, generation):
                    pass

            await asyncio.wait_for(asyncio.create_task(child_reentry()), timeout=1)
            # Different KBs still acquire their own independent fence.
            async with writer_store.kb_write_guard(other_kb_id, other_generation):
                pass

        await asyncio.wait_for(exclusive_entered.wait(), timeout=2)
        release_exclusive.set()
        await asyncio.wait_for(deletion_task, timeout=2)
    finally:
        release_exclusive.set()
        await writer_store.close()
        await deletion_store.close()


async def test_sqlite_lifecycle_schema_migration_is_idempotent(tmp_path):
    db_path = tmp_path / "legacy-lifecycle.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE enterprise_kb_lifecycle (
                kb_id TEXT PRIMARY KEY,
                generation TEXT NOT NULL,
                state TEXT NOT NULL CHECK (state IN ('active', 'deleted')),
                activated_at TEXT NOT NULL,
                deleted_at TEXT,
                updated_at TEXT NOT NULL,
                CHECK (kb_id <> '' AND kb_id = trim(kb_id)),
                CHECK (generation <> '' AND generation = trim(generation)),
                CHECK (
                    (state = 'active' AND deleted_at IS NULL)
                    OR (state = 'deleted' AND deleted_at IS NOT NULL)
                )
            );
            INSERT INTO enterprise_kb_lifecycle (
                kb_id, generation, state, activated_at, deleted_at, updated_at
            ) VALUES (
                'kb-legacy-lifecycle', 'gen-legacy-lifecycle', 'active',
                '2026-07-14T00:00:00+00:00', NULL,
                '2026-07-14T00:00:00+00:00'
            );
            """
        )

    store = SQLiteMetadataStore(db_path)
    await store.initialize()
    await store.initialize()
    lifecycle = await store.get_kb_lifecycle("kb-legacy-lifecycle")
    assert lifecycle is not None
    assert lifecycle.state == "active"
    assert lifecycle.delete_job_id is None
    with sqlite3.connect(db_path) as conn:
        columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(enterprise_kb_lifecycle)"
            ).fetchall()
        }
        table_sql = conn.execute(
            """
            SELECT sql FROM sqlite_master
            WHERE type = 'table' AND name = 'enterprise_kb_lifecycle'
            """
        ).fetchone()[0]
    assert "delete_job_id" in columns
    assert "'deleting'" in table_sql
    await store.close()


async def test_purge_kb_metadata_removes_everything(store):
    kb_id = _unique_kb(store)
    tenant_id = f"tenant-purge-{uuid.uuid4().hex[:8]}"
    raw_user = _enterprise_user(f"purge_{uuid.uuid4().hex[:10]}")
    user, _ = await store.upsert_enterprise_user_with_membership(
        EnterpriseUserRecord(**{**raw_user.to_dict(), "tenant_id": tenant_id}),
        _membership(raw_user.id, tenant_id),
    )
    await store.upsert_enterprise_user_kb_query_settings(
        EnterpriseUserKBQuerySettingsRecord(
            user_id=user.id,
            kb_id=kb_id,
            user_prompt="temporary prompt",
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
        )
    )
    now = utc_now_iso()
    await store.upsert_kb_acl(
        KBACLRecord(
            kb_id=kb_id,
            user_id=user.id,
            role="kb_viewer",
            granted_by="usr_admin",
            created_at=now,
            updated_at=now,
        )
    )
    await store.upsert_tenant_kb_acl(
        EnterpriseTenantKBACLRecord(
            tenant_id=tenant_id,
            kb_id=kb_id,
            role="kb_editor",
            granted_by="usr_admin",
            created_at=now,
            updated_at=now,
        )
    )
    await store.upsert_tenant_user_kb_override(
        EnterpriseTenantUserKBOverrideRecord(
            tenant_id=tenant_id,
            kb_id=kb_id,
            user_id=user.id,
            effect="allow",
            role="kb_viewer",
            granted_by="usr_admin",
            created_at=now,
            updated_at=now,
        )
    )
    api_key_input = _enterprise_api_key(kb_id)
    api_key_input.scopes["kb_roles"]["kb_keep"] = "kb_admin"
    api_key = await store.create_enterprise_api_key(api_key_input)
    await store.create_documents_and_job(
        [_doc(kb_id, "doc_a")], _job(kb_id, "job_x", document_id="doc_a")
    )
    purged = await store.purge_kb_metadata(kb_id)
    assert purged["documents"] >= 1
    assert purged["jobs"] >= 1
    assert purged["enterprise_kb_acl"] >= 1
    assert purged["enterprise_tenant_kb_acl"] >= 1
    assert purged["enterprise_tenant_user_kb_overrides"] >= 1
    assert purged["enterprise_user_kb_query_settings"] >= 1
    assert purged["enterprise_api_keys"] >= 1
    assert await store.get_enterprise_user_kb_query_settings(user.id, kb_id) is None
    assert await store.get_kb_acl_role(kb_id, user.id) is None
    assert await store.get_tenant_kb_acl_role(tenant_id, kb_id) is None
    assert await store.get_tenant_user_kb_override(tenant_id, kb_id, user.id) is None
    stripped_key = await store.get_enterprise_api_key_by_id(api_key.id)
    assert stripped_key is not None
    assert stripped_key.scopes["kb_roles"] == {"kb_keep": "kb_admin"}
    assert stripped_key.updated_at != api_key.updated_at
    with pytest.raises(MetadataRecordNotFoundError):
        await store.get_document(kb_id, "doc_a")


async def test_recover_orphan_jobs_marks_running_failed(store):
    kb_id = _unique_kb(store)
    await store.create_documents_and_job(
        [_doc(kb_id, "doc_a")], _job(kb_id, "job_orphan", job_type="parse", document_id="doc_a")
    )
    await store.transition_job(kb_id, "job_orphan", status="running")
    recovered = await store.recover_orphan_jobs()
    recovered_ids = {j.id for j in recovered}
    assert "job_orphan" in recovered_ids
    after = await store.get_job(kb_id, "job_orphan")
    assert after.status == "failed"
    assert after.error_code == "worker_orphaned"


async def test_recover_orphan_jobs_requeues_resumable_clear_in_place(store):
    kb_id = _unique_kb(store)
    job_id = f"job_clear_orphan_{uuid.uuid4().hex[:8]}"
    generation = f"gen-clear-orphan-{uuid.uuid4().hex}"
    now = utc_now_iso()
    clear_job = _job(
        kb_id,
        job_id,
        job_type="clear_kb",
        status="running",
        idempotency_key=f"clear_kb:{kb_id}:{generation}",
    )
    clear_job.stage = "finalizing"
    clear_job.progress = 0.9
    clear_job.retry_count = 1
    clear_job.payload = {
        "kb_generation": generation,
        "workspace": clear_job.workspace,
        "idempotency_fingerprint": "pinned",
    }
    clear_job.result = {"purged_rows": {"documents": 1}}
    clear_job.error_code = "stale_error"
    clear_job.error_message = "must be cleared"
    clear_job.started_at = now
    clear_job.finished_at = now
    clear_job.cancelled_at = now
    await store.create_job(clear_job)

    recovered = await store.recover_orphan_jobs(
        resumable_job_types={"clear_kb"},
        grace_seconds=0,
    )

    assert [job.id for job in recovered] == [job_id]
    queued = await store.get_job(kb_id, job_id)
    assert queued.status == "queued"
    assert queued.id == job_id
    assert queued.idempotency_key == clear_job.idempotency_key
    assert queued.payload == clear_job.payload
    assert queued.stage == "finalizing"
    assert queued.result == clear_job.result
    assert queued.progress == 0.9
    assert queued.retry_count == 2
    assert queued.started_at is None
    assert queued.finished_at is None
    assert queued.cancelled_at is None
    assert queued.error_code is None
    assert queued.error_message is None


async def test_recovery_respects_cross_store_job_owner_contract(store):
    kb_id = _unique_kb(store)
    document_id = f"doc_owner_{uuid.uuid4().hex[:8]}"
    job_id = f"job_owner_{uuid.uuid4().hex[:8]}"
    await store.create_documents_and_job(
        [_doc(kb_id, document_id, status="parsing")],
        _job(kb_id, job_id, job_type="parse", document_id=document_id),
    )
    await store.transition_job(kb_id, job_id, status="running")

    if isinstance(store, SQLiteMetadataStore):
        peer = SQLiteMetadataStore(store.db_path)
    else:
        from lightrag.api.postgres_metadata_store import PostgresMetadataStore

        peer = PostgresMetadataStore(
            dsn=_POSTGRES_DSN,
            min_size=1,
            max_size=1,
            operation_lock_pool_max_size=1,
        )
    await peer.initialize()
    try:
        async with store.job_execution_guard(job_id) as owned:
            assert owned is True
            assert await peer.recover_orphan_jobs(grace_seconds=0) == []
            assert (await peer.get_job(kb_id, job_id)).status == "running"
            assert (await peer.get_document(kb_id, document_id)).status == "parsing"

        recovered = await peer.recover_orphan_jobs(grace_seconds=0)
        assert [job.id for job in recovered] == [job_id]
        assert (await store.get_job(kb_id, job_id)).status == "failed"
        assert (await store.get_document(kb_id, document_id)).status == (
            "parse_failed"
        )
    finally:
        await peer.close()


async def test_sqlite_legacy_schema_repair_defaults_and_unique_invariant(tmp_path):
    db_path = tmp_path / "legacy-metadata.sqlite3"
    now = utc_now_iso()
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE enterprise_users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                system_role TEXT NOT NULL,
                status TEXT NOT NULL,
                tenant_id TEXT,
                can_create_kb INTEGER NOT NULL DEFAULT 0,
                can_use_bypass_query INTEGER NOT NULL DEFAULT 0,
                can_delete_documents INTEGER NOT NULL DEFAULT 0,
                can_use_agent_query INTEGER NOT NULL DEFAULT 0,
                token_version INTEGER NOT NULL DEFAULT 1,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE enterprise_tenant_memberships (
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                granted_by TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (tenant_id, user_id)
            );
            CREATE TABLE enterprise_audit_events (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                actor_user_id TEXT,
                target_type TEXT,
                target_id TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            """
        )
        for user_id, username, tenant_id in (
            ("usr_keep", "legacy-keep", "tenant-main"),
            ("usr_null", "legacy-null", None),
            ("usr_foreign_role", "legacy-foreign-role", "tenant-canonical"),
        ):
            conn.execute(
                """
                INSERT INTO enterprise_users (
                    id, username, password_hash, system_role, status, tenant_id,
                    can_create_kb, can_use_bypass_query, can_delete_documents,
                    can_use_agent_query, token_version, metadata_json,
                    created_at, updated_at
                ) VALUES (?, ?, 'hash', 'user', 'active', ?, 0, 0, 0, 0, 1,
                          '{}', ?, ?)
                """,
                (user_id, username, tenant_id, now, now),
            )
        for tenant_id, user_id, role in (
            ("tenant-main", "usr_keep", "tenant_admin"),
            ("tenant-foreign", "usr_keep", "tenant_owner"),
            ("tenant-orphan", "usr_null", "tenant_admin"),
            ("tenant-other", "usr_foreign_role", "tenant_owner"),
        ):
            conn.execute(
                """
                INSERT INTO enterprise_tenant_memberships (
                    tenant_id, user_id, role, granted_by, created_at, updated_at
                ) VALUES (?, ?, ?, NULL, ?, ?)
                """,
                (tenant_id, user_id, role, now, now),
            )
        conn.execute(
            """
            INSERT INTO enterprise_audit_events (
                id, event_type, actor_user_id, target_type, target_id,
                metadata_json, created_at
            ) VALUES ('audit_legacy', 'legacy', 'usr_keep', 'user', 'usr_keep',
                      '{}', ?)
            """,
            (now,),
        )

    store = SQLiteMetadataStore(db_path)
    await store.initialize()
    try:
        legacy = await store.get_enterprise_user_by_id("usr_keep")
        assert legacy is not None and legacy.can_download_files is True
        assert [
            (item.tenant_id, item.role)
            for item in await store.list_user_tenant_memberships("usr_keep")
        ] == [("tenant-main", "tenant_admin")]
        assert await store.list_user_tenant_memberships("usr_null") == []
        # The owner role from another tenant is deleted, never migrated.
        assert [
            (item.tenant_id, item.role)
            for item in await store.list_user_tenant_memberships("usr_foreign_role")
        ] == [("tenant-canonical", "tenant_member")]
        legacy_events = await store.list_audit_events(event_type="legacy")
        assert len(legacy_events) == 1
        assert legacy_events[0].actor_tenant_id is None

        explicit = _enterprise_user("new-explicit-download-default")
        saved = await store.upsert_enterprise_user(explicit)
        assert saved.can_download_files is False

        # Re-running initialization is safe and preserves the repaired role.
        await store.initialize()
        assert [
            item.role for item in await store.list_user_tenant_memberships("usr_keep")
        ] == ["tenant_admin"]
        with sqlite3.connect(db_path) as conn:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    """
                    INSERT INTO enterprise_tenant_memberships (
                        tenant_id, user_id, role, granted_by, created_at, updated_at
                    ) VALUES ('tenant-second', 'usr_keep', 'tenant_member', NULL, ?, ?)
                    """,
                    (now, now),
                )
    finally:
        await store.close()


def test_postgres_legacy_json_defaults():
    from lightrag.api.postgres_metadata_store import (
        _audit_event_from_row,
        _enterprise_user_from_row,
        _tenant_membership_from_row,
    )

    legacy_user = _enterprise_user("pg-legacy")
    legacy_data = legacy_user.to_dict()
    legacy_data.pop("can_download_files")
    assert _enterprise_user_from_row({"data_json": legacy_data}).can_download_files is True
    assert (
        _enterprise_user_from_row(
            {"data_json": {**legacy_data, "can_download_files": False}}
        ).can_download_files
        is False
    )

    legacy_audit = AuditEventRecord(
        id="audit_pg_legacy",
        event_type="legacy",
        actor_user_id="usr_pg",
        target_type=None,
        target_id=None,
        metadata={},
        created_at=utc_now_iso(),
    ).to_dict()
    legacy_audit.pop("actor_tenant_id")
    assert _audit_event_from_row({"data_json": legacy_audit}).actor_tenant_id is None

    # Projection columns are canonical when legacy JSONB diverges. A foreign
    # tenant's owner role must never escape through a principal-facing loader.
    projection_time = utc_now_iso()
    divergent_membership = _tenant_membership_from_row(
        {
            "tenant_id": "tenant-projection-a",
            "user_id": "usr_projection",
            "role": "tenant_member",
            "granted_by": "usr_projection_admin",
            "created_at": projection_time,
            "updated_at": projection_time,
            "data_json": {
                "tenant_id": "tenant-json-b",
                "user_id": "usr_projection",
                "role": "tenant_owner",
                "granted_by": "usr_json_admin",
                "created_at": "stale-created-at",
                "updated_at": "stale-updated-at",
            },
        }
    )
    assert divergent_membership == EnterpriseTenantMembershipRecord(
        tenant_id="tenant-projection-a",
        user_id="usr_projection",
        role="tenant_member",
        granted_by="usr_projection_admin",
        created_at=projection_time,
        updated_at=projection_time,
    )

    # Enterprise-user projection columns are canonical too. Only columns that
    # actually exist in the row are overlaid, so deployments are not forced to
    # have system_role/token_version projection columns.
    json_user = _enterprise_user("pg-divergent-user").to_dict()
    json_user.update(
        {
            "id": "usr-json-b",
            "username": "json-b",
            "status": "disabled",
            "tenant_id": "tenant-json-b",
            "created_at": "stale-created",
            "updated_at": "stale-updated",
        }
    )
    projected_user = _enterprise_user_from_row(
        {
            "id": "usr-projection-a",
            "username": "projection-a",
            "status": "active",
            "tenant_id": "tenant-projection-a",
            "created_at": projection_time,
            "updated_at": projection_time,
            "data_json": json_user,
        }
    )
    assert projected_user.id == "usr-projection-a"
    assert projected_user.username == "projection-a"
    assert projected_user.status == "active"
    assert projected_user.tenant_id == "tenant-projection-a"
    assert projected_user.created_at == projection_time
    assert projected_user.updated_at == projection_time
    assert projected_user.system_role == json_user["system_role"]
    assert projected_user.token_version == json_user["token_version"]


async def test_postgres_user_projection_read_and_cas_sql_contract():
    from lightrag.api.postgres_metadata_store import (
        PostgresMetadataStore,
        _enterprise_user_from_row,
    )

    projection_time = "2026-07-14T01:00:00+00:00"
    candidate_time = "2026-07-14T01:00:01+00:00"
    json_user = _enterprise_user("pg-cas-json").to_dict()
    json_user.update(
        {
            "id": "usr-json-b",
            "username": "json-b",
            "status": "disabled",
            "tenant_id": "tenant-json-b",
            "created_at": "stale-created",
            "updated_at": "stale-updated",
        }
    )
    projection_row = {
        "id": "usr-projection-a",
        "username": "projection-a",
        "status": "active",
        "tenant_id": "tenant-projection-a",
        "created_at": projection_time,
        "updated_at": projection_time,
        "data_json": json_user,
    }

    class ReadConnection:
        def __init__(self):
            self.statements: list[str] = []

        async def fetchrow(self, statement: str, *_args):
            self.statements.append(statement)
            return dict(projection_row)

        async def fetch(self, statement: str, *_args):
            self.statements.append(statement)
            return [dict(projection_row)]

    class AcquireContext:
        def __init__(self, conn):
            self.conn = conn

        async def __aenter__(self):
            return self.conn

        async def __aexit__(self, *_args):
            return None

    class ReadPool:
        def __init__(self, conn):
            self.conn = conn

        def acquire(self):
            return AcquireContext(self.conn)

    read_conn = ReadConnection()
    read_store = PostgresMetadataStore(dsn="postgresql://unused")
    read_store._pool = ReadPool(read_conn)
    read_store._initialized = True
    by_username = await read_store.get_enterprise_user_by_username("projection-a")
    by_id = await read_store.get_enterprise_user_by_id("usr-projection-a")
    listed = await read_store.list_enterprise_users()
    assert by_username is not None and by_username.tenant_id == "tenant-projection-a"
    assert by_id is not None and by_id.status == "active"
    assert [user.id for user in listed] == ["usr-projection-a"]
    assert len(read_conn.statements) == 3
    for statement in read_conn.statements:
        normalized = " ".join(statement.split())
        assert (
            "SELECT id, username, status, tenant_id, created_at, updated_at, data_json"
            in normalized
        )

    class CASConnection:
        def __init__(self):
            self.statements: list[str] = []
            self.saved_row: dict[str, Any] | None = None

        async def fetchrow(self, statement: str, *_args):
            self.statements.append(statement)
            if "enterprise_users" in statement and "FOR UPDATE" in statement:
                return dict(projection_row)
            if "enterprise_tenant_memberships" in statement:
                return None
            if "enterprise_users" in statement:
                return self.saved_row
            return None

        async def fetch(self, statement: str, *_args):
            self.statements.append(statement)
            return []

        async def execute(self, statement: str, *args):
            self.statements.append(statement)
            if "INSERT INTO enterprise_users" in statement:
                self.saved_row = {
                    "id": args[0],
                    "username": args[1],
                    "status": args[2],
                    "tenant_id": args[3],
                    "created_at": args[4],
                    "updated_at": args[5],
                    "data_json": json.loads(args[6]),
                }
            return "INSERT 0 1"

    current = _enterprise_user_from_row(projection_row)
    candidate = EnterpriseUserRecord(
        **{
            **current.to_dict(),
            "can_create_kb": True,
            "token_version": current.token_version + 1,
            "updated_at": candidate_time,
        }
    )
    pg_store = PostgresMetadataStore(dsn="postgresql://unused")
    cas_conn = CASConnection()
    saved, _membership_record = await pg_store._upsert_enterprise_user_with_membership(
        cas_conn,
        candidate,
        membership=None,
        expected_updated_at=current.updated_at,
        expected_token_version=current.token_version,
        expected_tenant_id="tenant-projection-a",
        expected_membership=None,
        allow_tenant_change=False,
    )
    assert saved.tenant_id == "tenant-projection-a"
    assert saved.status == "active"
    assert saved.can_create_kb is True
    assert any(
        "SELECT id, username, status, tenant_id, created_at, updated_at" in statement
        and "FOR UPDATE" in statement
        for statement in cas_conn.statements
    )

    stale_json_conn = CASConnection()
    with pytest.raises(MetadataConflictError):
        await pg_store._upsert_enterprise_user_with_membership(
            stale_json_conn,
            candidate,
            membership=None,
            expected_updated_at=current.updated_at,
            expected_token_version=current.token_version,
            expected_tenant_id="tenant-json-b",
            expected_membership=None,
            allow_tenant_change=False,
        )
    assert not any(
        "INSERT INTO enterprise_users" in statement
        for statement in stale_json_conn.statements
    )


async def test_postgres_override_target_cas_and_audit_sql_contract():
    from lightrag.api.postgres_metadata_store import PostgresMetadataStore

    tenant_id = "tenant-pg-override-cas"
    kb_id = "kb-pg-override-cas"
    expected_user = EnterpriseUserRecord(
        **{
            **_enterprise_user("pg-override-cas").to_dict(),
            "tenant_id": tenant_id,
        }
    )
    expected_membership = _membership(expected_user.id, tenant_id)
    current_user = EnterpriseUserRecord(
        **{
            **expected_user.to_dict(),
            "token_version": expected_user.token_version + 1,
        }
    )
    override = EnterpriseTenantUserKBOverrideRecord(
        tenant_id=tenant_id,
        kb_id=kb_id,
        user_id=expected_user.id,
        effect="allow",
        role="kb_viewer",
        granted_by="usr_admin",
        created_at=utc_now_iso(),
        updated_at=utc_now_iso(),
    )

    class TransactionContext:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *_args):
            return None

    class Connection:
        def __init__(self):
            self.statements: list[str] = []
            self.audit_json: dict[str, Any] | None = None

        def transaction(self):
            return TransactionContext()

        async def execute(self, statement: str, *args):
            self.statements.append(statement)
            if "INSERT INTO enterprise_audit_events" in statement:
                self.audit_json = json.loads(args[7])
            return "INSERT 0 1"

        async def fetchrow(self, statement: str, *_args):
            self.statements.append(statement)
            if "FROM enterprise_kb_lifecycle" in statement:
                return None
            if "FROM enterprise_users" in statement:
                return {
                    "id": current_user.id,
                    "username": current_user.username,
                    "status": current_user.status,
                    "tenant_id": current_user.tenant_id,
                    "created_at": current_user.created_at,
                    "updated_at": current_user.updated_at,
                    "data_json": current_user.to_dict(),
                }
            if "FROM enterprise_tenant_memberships" in statement:
                return {
                    **expected_membership.to_dict(),
                    "data_json": expected_membership.to_dict(),
                }
            if "FROM enterprise_audit_events" in statement:
                assert self.audit_json is not None
                return {"data_json": self.audit_json}
            return None

    class AcquireContext:
        def __init__(self, conn):
            self.conn = conn

        async def __aenter__(self):
            return self.conn

        async def __aexit__(self, *_args):
            return None

    class Pool:
        def __init__(self, conn):
            self.conn = conn

        def acquire(self):
            return AcquireContext(self.conn)

    for operation in ("upsert", "deny", "reset"):
        conn = Connection()
        pg_store = PostgresMetadataStore(dsn="postgresql://unused")
        pg_store._pool = Pool(conn)
        pg_store._initialized = True
        with pytest.raises(MetadataConflictError):
            if operation == "upsert":
                await pg_store.upsert_tenant_user_kb_override(
                    override,
                    expected_user=expected_user,
                    expected_membership=expected_membership,
                )
            elif operation == "deny":
                await pg_store.delete_tenant_user_kb_override(
                    tenant_id,
                    kb_id,
                    expected_user.id,
                    expected_user=expected_user,
                    expected_membership=expected_membership,
                )
            else:
                await pg_store.reset_tenant_user_kb_override(
                    tenant_id,
                    kb_id,
                    expected_user.id,
                    expected_user=expected_user,
                    expected_membership=expected_membership,
                )
        normalized = [" ".join(statement.split()) for statement in conn.statements]
        assert any(
            "FROM enterprise_users" in statement and "FOR UPDATE" in statement
            for statement in normalized
        )
        assert any(
            "FROM enterprise_tenant_memberships" in statement
            and "FOR UPDATE" in statement
            for statement in normalized
        )
        assert not any(
            "INSERT INTO enterprise_tenant_user_kb_overrides" in statement
            or "DELETE FROM enterprise_tenant_user_kb_overrides" in statement
            for statement in normalized
        )

    audit_conn = Connection()
    audit_store = PostgresMetadataStore(dsn="postgresql://unused")
    audit_store._pool = Pool(audit_conn)
    audit_store._initialized = True
    persisted = await audit_store.append_audit_event(
        AuditEventRecord(
            id="audit-pg-explicit-null",
            event_type="explicit_null",
            actor_user_id=expected_user.id,
            actor_tenant_id=None,
            target_type=None,
            target_id=None,
            metadata={},
            created_at=utc_now_iso(),
        )
    )
    assert persisted.actor_tenant_id is None
    assert not any(
        "FROM enterprise_users" in statement for statement in audit_conn.statements
    )


async def test_sqlite_two_store_assignments_keep_one_canonical_membership(tmp_path):
    db_path = tmp_path / "two-store-membership.sqlite3"
    store_a = SQLiteMetadataStore(db_path)
    store_b = SQLiteMetadataStore(db_path)
    await store_a.initialize()
    await store_b.initialize()
    try:
        created = await store_a.upsert_enterprise_user(
            _enterprise_user(f"two_store_{uuid.uuid4().hex[:8]}")
        )
        snapshot_a = await store_a.get_enterprise_user_by_id(created.id)
        snapshot_b = await store_b.get_enterprise_user_by_id(created.id)
        assert snapshot_a is not None
        assert snapshot_b is not None and snapshot_b == snapshot_a

        async def assign(
            target_store: SQLiteMetadataStore,
            snapshot: EnterpriseUserRecord,
            tenant_id: str,
        ):
            now = utc_now_iso()
            candidate = EnterpriseUserRecord(
                **{**snapshot.to_dict(), "tenant_id": tenant_id, "updated_at": now}
            )
            return await target_store.upsert_enterprise_user_with_membership(
                candidate,
                _membership(snapshot.id, tenant_id),
                **_user_cas(snapshot),
            )

        results = await asyncio.gather(
            assign(store_a, snapshot_a, "tenant-concurrent-a"),
            assign(store_b, snapshot_b, "tenant-concurrent-b"),
            return_exceptions=True,
        )
        assert sum(isinstance(item, MetadataConflictError) for item in results) == 1
        assert sum(isinstance(item, tuple) for item in results) == 1

        final_a = await store_a.get_enterprise_user_by_id(created.id)
        final_b = await store_b.get_enterprise_user_by_id(created.id)
        assert final_a is not None and final_b == final_a
        memberships = await store_b.list_user_tenant_memberships(created.id)
        assert len(memberships) == 1
        assert memberships[0].tenant_id == final_a.tenant_id
        assert final_a.tenant_id in {"tenant-concurrent-a", "tenant-concurrent-b"}
    finally:
        await store_a.close()
        await store_b.close()


async def test_sqlite_two_store_scoped_user_and_membership_cas(tmp_path):
    db_path = tmp_path / "two-store-scoped-cas.sqlite3"
    store_a = SQLiteMetadataStore(db_path)
    store_b = SQLiteMetadataStore(db_path)
    await store_a.initialize()
    await store_b.initialize()

    async def create_member(username: str, tenant_id: str):
        raw = _enterprise_user(username)
        return await store_a.upsert_enterprise_user_with_membership(
            EnterpriseUserRecord(**{**raw.to_dict(), "tenant_id": tenant_id}),
            _membership(raw.id, tenant_id),
        )

    try:
        moved_user, moved_membership = await create_member(
            f"scoped_move_{uuid.uuid4().hex[:8]}", "tenant-a"
        )
        assert moved_membership is not None
        moved = EnterpriseUserRecord(
            **{
                **moved_user.to_dict(),
                "tenant_id": "tenant-b",
                "updated_at": "2099-01-01T00:00:01+00:00",
            }
        )
        moved, _ = await store_b.upsert_enterprise_user_with_membership(
            moved,
            _membership(moved.id, "tenant-b"),
            **_user_cas(moved_user),
        )
        stale_update = EnterpriseUserRecord(
            **{
                **moved_user.to_dict(),
                "can_create_kb": True,
                "token_version": moved_user.token_version + 1,
                "updated_at": "2099-01-01T00:00:02+00:00",
            }
        )
        with pytest.raises(MetadataConflictError):
            await store_a.upsert_enterprise_user_with_membership(
                stale_update,
                None,
                **_user_cas(moved_user),
                expected_membership=moved_membership,
            )
        assert await store_a.get_enterprise_user_by_id(moved.id) == moved

        password_user, password_membership = await create_member(
            f"scoped_password_{uuid.uuid4().hex[:8]}", "tenant-a"
        )
        assert password_membership is not None
        promoted = _membership(password_user.id, "tenant-a", role="tenant_admin")
        promoted.updated_at = "2099-01-01T00:00:03+00:00"
        await store_b.upsert_tenant_membership(promoted)
        stale_password = EnterpriseUserRecord(
            **{
                **password_user.to_dict(),
                "password_hash": "stale-password-write",
                "token_version": password_user.token_version + 1,
                "updated_at": "2099-01-01T00:00:04+00:00",
            }
        )
        with pytest.raises(MetadataConflictError):
            await store_a.upsert_enterprise_user_with_membership(
                stale_password,
                None,
                **_user_cas(password_user),
                expected_membership=password_membership,
            )
        current_password_user = await store_a.get_enterprise_user_by_id(
            password_user.id
        )
        current_password_membership = await store_a.get_tenant_membership(
            "tenant-a", password_user.id
        )
        assert current_password_user is not None
        assert current_password_user.password_hash == password_user.password_hash
        assert (
            current_password_membership is not None
            and current_password_membership.role == "tenant_admin"
        )

        delete_user, delete_membership = await create_member(
            f"scoped_delete_{uuid.uuid4().hex[:8]}", "tenant-a"
        )
        assert delete_membership is not None
        revised = EnterpriseUserRecord(
            **{
                **delete_user.to_dict(),
                "can_download_files": True,
                "token_version": delete_user.token_version + 1,
                "updated_at": "2099-01-01T00:00:05+00:00",
            }
        )
        revised = await store_b.upsert_enterprise_user(
            revised, **_user_cas(delete_user)
        )
        with pytest.raises(MetadataConflictError):
            await store_a.delete_enterprise_user(
                delete_user.id,
                **_user_cas(delete_user),
                expected_membership=delete_membership,
            )
        assert await store_a.get_enterprise_user_by_id(delete_user.id) == revised

        grant_user = await store_a.upsert_enterprise_user(
            _enterprise_user(f"scoped_grant_{uuid.uuid4().hex[:8]}")
        )
        moved_grant = EnterpriseUserRecord(
            **{
                **grant_user.to_dict(),
                "tenant_id": "tenant-b",
                "updated_at": "2099-01-01T00:00:06+00:00",
            }
        )
        moved_grant, _ = await store_b.upsert_enterprise_user_with_membership(
            moved_grant,
            _membership(grant_user.id, "tenant-b"),
            **_user_cas(grant_user),
            expected_membership=None,
        )
        stale_grant = EnterpriseUserRecord(
            **{
                **grant_user.to_dict(),
                "tenant_id": "tenant-a",
                "updated_at": "2099-01-01T00:00:07+00:00",
            }
        )
        with pytest.raises(MetadataConflictError):
            await store_a.upsert_enterprise_user_with_membership(
                stale_grant,
                _membership(grant_user.id, "tenant-a"),
                **_user_cas(grant_user),
                expected_membership=None,
            )
        assert await store_a.get_enterprise_user_by_id(grant_user.id) == moved_grant

        revoke_user, revoke_membership = await create_member(
            f"scoped_revoke_{uuid.uuid4().hex[:8]}", "tenant-a"
        )
        assert revoke_membership is not None
        promoted_revoke = _membership(
            revoke_user.id, "tenant-a", role="tenant_admin"
        )
        promoted_revoke.updated_at = "2099-01-01T00:00:08+00:00"
        await store_b.upsert_tenant_membership(promoted_revoke)
        stale_revoke = EnterpriseUserRecord(
            **{
                **revoke_user.to_dict(),
                "tenant_id": None,
                "updated_at": "2099-01-01T00:00:09+00:00",
            }
        )
        with pytest.raises(MetadataConflictError):
            await store_a.upsert_enterprise_user_with_membership(
                stale_revoke,
                None,
                **_user_cas(revoke_user),
                expected_membership=revoke_membership,
            )
        retained_revoke = await store_a.get_enterprise_user_by_id(revoke_user.id)
        retained_membership = await store_a.get_tenant_membership(
            "tenant-a", revoke_user.id
        )
        assert retained_revoke is not None
        assert retained_revoke.tenant_id == "tenant-a"
        assert retained_membership is not None
        assert retained_membership.role == "tenant_admin"
    finally:
        await store_a.close()
        await store_b.close()


async def test_sqlite_membership_write_failure_rolls_back_user_and_override(tmp_path):
    store = SQLiteMetadataStore(tmp_path / "membership-rollback.sqlite3")
    await store.initialize()
    tenant_a = "tenant-rollback-a"
    tenant_b = "tenant-rollback-b"
    kb_id = "kb-membership-rollback"
    raw = _enterprise_user(f"membership_rollback_{uuid.uuid4().hex[:8]}")
    initial, original_membership = await store.upsert_enterprise_user_with_membership(
        EnterpriseUserRecord(**{**raw.to_dict(), "tenant_id": tenant_a}),
        _membership(raw.id, tenant_a, role="tenant_admin"),
    )
    assert original_membership is not None
    override = EnterpriseTenantUserKBOverrideRecord(
        tenant_id=tenant_a,
        kb_id=kb_id,
        user_id=initial.id,
        effect="allow",
        role="kb_editor",
        granted_by="usr_admin",
        created_at=utc_now_iso(),
        updated_at=utc_now_iso(),
    )
    await store.upsert_tenant_user_kb_override(override)

    with sqlite3.connect(store.db_path) as conn:
        conn.executescript(
            f"""
            CREATE TRIGGER fail_tenant_membership_insert
            BEFORE INSERT ON enterprise_tenant_memberships
            WHEN NEW.tenant_id = '{tenant_b}'
            BEGIN
                SELECT RAISE(ABORT, 'injected membership failure');
            END;
            """
        )

    candidate = EnterpriseUserRecord(
        **{
            **initial.to_dict(),
            "tenant_id": tenant_b,
            "password_hash": "must-roll-back",
            "status": "disabled",
            "can_create_kb": True,
            "token_version": initial.token_version + 1,
            "updated_at": utc_now_iso(),
        }
    )
    with pytest.raises(sqlite3.IntegrityError, match="injected membership failure"):
        await store.upsert_enterprise_user_with_membership(
            candidate,
            _membership(initial.id, tenant_b),
            **_user_cas(initial),
        )

    current = await store.get_enterprise_user_by_id(initial.id)
    assert current == initial
    assert await store.list_user_tenant_memberships(initial.id) == [
        original_membership
    ]
    assert (
        await store.get_tenant_user_kb_override(tenant_a, kb_id, initial.id)
        == override
    )
    assert await store.get_tenant_membership(tenant_b, initial.id) is None
    await store.close()


async def test_sqlite_hard_delete_failure_rolls_back_metadata(tmp_path):
    from lightrag.api.metadata_store import MetadataStoreError

    store = SQLiteMetadataStore(tmp_path / "purge-rollback.sqlite3")
    await store.initialize()
    kb_id = "kb_rollback"
    generation = "gen-rollback"
    await store.activate_kb_generation(kb_id, generation)
    user = await store.upsert_enterprise_user(_enterprise_user("rollback-user"))
    now = utc_now_iso()
    await store.upsert_kb_acl(
        KBACLRecord(
            kb_id=kb_id,
            user_id=user.id,
            role="kb_viewer",
            granted_by=None,
            created_at=now,
            updated_at=now,
        ),
        expected_generation=generation,
    )
    valid_key = await store.create_enterprise_api_key(
        _enterprise_api_key(kb_id),
        expected_kb_generations={kb_id: generation},
    )
    malformed_key = await store.create_enterprise_api_key(
        _enterprise_api_key(kb_id),
        expected_kb_generations={kb_id: generation},
    )
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE enterprise_api_keys SET scopes_json = '[]' WHERE id = ?",
            (malformed_key.id,),
        )

    with pytest.raises(MetadataStoreError):
        await store.purge_kb_metadata(kb_id, generation=generation)

    # Neither the tombstone nor any earlier metadata rewrite may survive.
    lifecycle = await store.get_kb_lifecycle(kb_id)
    assert lifecycle is not None
    assert lifecycle.generation == generation
    assert lifecycle.state == "active"
    assert lifecycle.deleted_at is None
    assert await store.get_kb_acl_role(kb_id, user.id) == "kb_viewer"
    unchanged_key = await store.get_enterprise_api_key_by_id(valid_key.id)
    assert unchanged_key is not None
    assert unchanged_key.scopes["kb_roles"] == {kb_id: "kb_viewer"}
    await store.close()


async def test_sqlite_purge_serializes_with_delayed_old_generation_grant(
    tmp_path, monkeypatch
):
    import lightrag.api.metadata_store as metadata_store_module

    db_path = tmp_path / "lifecycle-interleaving.sqlite3"
    purge_store = SQLiteMetadataStore(db_path)
    grant_store = SQLiteMetadataStore(db_path)
    await purge_store.initialize()
    await grant_store.initialize()
    kb_id = "kb-lifecycle-interleaving"
    generation = "gen-lifecycle-interleaving"
    entered_purge = threading.Event()
    release_purge = threading.Event()
    armed = False
    original_loads = metadata_store_module._loads_json_object

    def blocking_loads(value):
        if armed and not entered_purge.is_set():
            entered_purge.set()
            if not release_purge.wait(timeout=10):
                raise RuntimeError("timed out waiting to release lifecycle purge")
        return original_loads(value)

    try:
        await purge_store.activate_kb_generation(kb_id, generation)
        user = await purge_store.upsert_enterprise_user(
            _enterprise_user(f"lifecycle_race_{uuid.uuid4().hex[:8]}")
        )
        await purge_store.create_enterprise_api_key(
            _enterprise_api_key(kb_id),
            expected_kb_generations={kb_id: generation},
        )
        grant = KBACLRecord(
            kb_id=kb_id,
            user_id=user.id,
            role="kb_admin",
            granted_by="usr_admin",
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
        )

        monkeypatch.setattr(
            metadata_store_module, "_loads_json_object", blocking_loads
        )
        armed = True
        purge_task = asyncio.create_task(
            asyncio.to_thread(
                lambda: asyncio.run(
                    purge_store.purge_kb_metadata(kb_id, generation=generation)
                )
            )
        )
        assert await asyncio.to_thread(entered_purge.wait, 5)
        grant_task = asyncio.create_task(
            asyncio.to_thread(
                lambda: asyncio.run(
                    grant_store.upsert_kb_acl(
                        grant, expected_generation=generation
                    )
                )
            )
        )
        await asyncio.sleep(0.1)
        assert not grant_task.done()
        release_purge.set()
        await purge_task
        grant_result = (await asyncio.gather(grant_task, return_exceptions=True))[0]
        assert isinstance(grant_result, KBLifecycleConflictError)
        assert await grant_store.get_kb_acl_role(kb_id, user.id) is None
        lifecycle = await grant_store.get_kb_lifecycle(kb_id)
        assert lifecycle is not None and lifecycle.state == "deleted"
    finally:
        release_purge.set()
        await purge_store.close()
        await grant_store.close()


async def test_postgres_schema_migrations_are_idempotent_contract():
    from lightrag.api.postgres_metadata_store import PostgresMetadataStore

    class RecordingConnection:
        def __init__(self):
            self.statements: list[str] = []

        async def execute(self, statement: str, *_args):
            self.statements.append(statement)
            return "OK"

    conn = RecordingConnection()
    pg_store = PostgresMetadataStore(dsn="postgresql://unused")
    await pg_store._initialize_schema(conn)
    await pg_store._initialize_schema(conn)
    sql = "\n".join(conn.statements)
    assert "CREATE TABLE IF NOT EXISTS enterprise_kb_lifecycle" in sql
    assert "state IN ('active', 'deleting', 'deleted')" in sql
    assert "ADD COLUMN IF NOT EXISTS delete_job_id" in sql
    assert "enterprise_kb_lifecycle_state_payload_v2_check" in sql
    assert "enterprise_tenant_user_kb_overrides" in sql
    assert "ADD COLUMN IF NOT EXISTS actor_tenant_id" in sql
    assert "uq_enterprise_tenant_memberships_user" in sql
    assert "idx_enterprise_audit_events_actor_tenant" in sql
    assert "UPDATE enterprise_tenant_memberships" in sql
    assert "'role', role" in sql
    assert sql.count("ADD COLUMN IF NOT EXISTS actor_tenant_id") == 2
    assert sql.count("ADD COLUMN IF NOT EXISTS delete_job_id") == 2


async def test_postgres_lifecycle_uses_advisory_and_row_locks_sql_contract():
    from lightrag.api.postgres_metadata_store import PostgresMetadataStore

    class RecordingLifecycleConnection:
        def __init__(self):
            self.statements: list[str] = []
            self.lifecycle: dict[str, Any] | None = None

        async def execute(self, statement: str, *args):
            self.statements.append(statement)
            if "INSERT INTO enterprise_kb_lifecycle" in statement:
                self.lifecycle = {
                    "kb_id": args[0],
                    "generation": args[1],
                    "state": "active",
                    "activated_at": args[2],
                    "deleted_at": None,
                    "updated_at": args[2],
                    "delete_job_id": None,
                }
                return "INSERT 0 1"
            if "SET state = 'deleting'" in statement:
                assert self.lifecycle is not None
                self.lifecycle.update(
                    state="deleting",
                    delete_job_id=args[2],
                    deleted_at=None,
                    updated_at=args[3],
                )
                return "UPDATE 1"
            if "SET state = 'deleted'" in statement:
                assert self.lifecycle is not None
                self.lifecycle.update(
                    state="deleted",
                    deleted_at=args[3],
                    updated_at=args[3],
                )
                return "UPDATE 1"
            return "SELECT 1"

        async def fetchrow(self, statement: str, *_args):
            self.statements.append(statement)
            if "FROM enterprise_kb_lifecycle" in statement:
                return self.lifecycle
            return None

    conn = RecordingLifecycleConnection()
    pg_store = PostgresMetadataStore(dsn="postgresql://unused")
    activated = await pg_store._activate_kb_generation(
        conn,
        "kb-pg-lock-contract",
        "gen-pg-lock-contract",
        activated_at="2026-07-14T00:00:00+00:00",
    )
    asserted = await pg_store._assert_kb_generation(
        conn, activated.kb_id, activated.generation
    )
    assert asserted == activated
    with pytest.raises(KBLifecycleConflictError):
        await pg_store._assert_kb_generation(conn, activated.kb_id, None)
    deleting = await pg_store._begin_kb_deletion(
        conn, activated.kb_id, activated.generation, "job-pg-delete-contract"
    )
    assert deleting.state == "deleting"
    assert deleting.delete_job_id == "job-pg-delete-contract"
    completed = await pg_store._complete_kb_deletion(
        conn, activated.kb_id, activated.generation, "job-pg-delete-contract"
    )
    assert completed.state == "deleted"

    lock_indexes = [
        index
        for index, statement in enumerate(conn.statements)
        if "pg_advisory_xact_lock" in statement
    ]
    row_lock_indexes = [
        index
        for index, statement in enumerate(conn.statements)
        if "FROM enterprise_kb_lifecycle" in statement and "FOR UPDATE" in statement
    ]
    assert len(lock_indexes) == len(row_lock_indexes) == 5
    assert all(lock_index < row_index for lock_index, row_index in zip(lock_indexes, row_lock_indexes))
    sql = "\n".join(conn.statements)
    assert "state = 'active'" in sql and "delete_job_id IS NULL" in sql
    assert "state = 'deleting'" in sql and "delete_job_id = $3" in sql


def test_postgres_operation_guards_use_session_locks_and_finally_unlock():
    import inspect

    from lightrag.api.postgres_metadata_store import PostgresMetadataStore

    job_source = inspect.getsource(PostgresMetadataStore.job_execution_guard)
    shared_source = inspect.getsource(PostgresMetadataStore.kb_write_guard)
    exclusive_source = inspect.getsource(
        PostgresMetadataStore.kb_exclusive_operation_guard
    )
    assert "pg_advisory_lock" in job_source
    assert "pg_try_advisory_lock" in job_source
    assert "pg_advisory_unlock" in job_source
    assert "1263295563" in job_source
    assert "_operation_session" in job_source
    assert "pg_advisory_lock_shared" in shared_source
    assert "pg_advisory_unlock_shared" in shared_source
    assert "finally:" in shared_source
    assert "pg_advisory_lock(" in exclusive_source
    assert "pg_advisory_unlock(" in exclusive_source
    assert "finally:" in exclusive_source
    assert "_ensure_operation_lock_pool" in shared_source
    assert "_ensure_operation_lock_pool" in exclusive_source
    assert "_pool_or_raise" not in shared_source + exclusive_source
    # Operation locks use a namespace distinct from lifecycle xact locks.
    assert "1263295562" in shared_source
    assert "1263295562" in exclusive_source
    assert "1263295563" not in shared_source + exclusive_source
    assert "1263295561" not in shared_source + exclusive_source
    assert "hash(" not in shared_source + exclusive_source


async def test_postgres_operation_lock_pool_init_is_bounded_and_concurrent_safe(
    monkeypatch,
):
    import lightrag.api.postgres_metadata_store as postgres_metadata_store_module

    class TransactionContext:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *_args):
            return None

    class Connection:
        def __init__(self):
            self.statements: list[str] = []

        def transaction(self):
            return TransactionContext()

        async def execute(self, statement: str, *_args):
            self.statements.append(statement)
            return "SELECT 1"

    class AcquireContext:
        def __init__(self, pool):
            self.pool = pool

        async def __aenter__(self):
            self.pool.acquired += 1
            return self.pool.connection

        async def __aexit__(self, *_args):
            self.pool.released += 1
            return None

    class Pool:
        def __init__(self, label: str):
            self.label = label
            self.connection = Connection()
            self.acquired = 0
            self.released = 0
            self.closed = 0

        def acquire(self):
            return AcquireContext(self)

        async def close(self):
            self.closed += 1

    class FakeAsyncPG:
        def __init__(self):
            self.calls: list[dict[str, Any]] = []
            self.pools: list[Pool] = []

        async def create_pool(self, **kwargs):
            self.calls.append(dict(kwargs))
            pool = Pool(f"created-{len(self.pools)}")
            self.pools.append(pool)
            await asyncio.sleep(0)
            return pool

    fake_asyncpg = FakeAsyncPG()
    monkeypatch.setattr(
        postgres_metadata_store_module, "_load_asyncpg", lambda: fake_asyncpg
    )
    store = postgres_metadata_store_module.PostgresMetadataStore(
        host="db.internal",
        port=5544,
        user="metadata-user",
        password="metadata-password",
        database="metadata-db",
        min_size=1,
        max_size=1,
        operation_lock_pool_max_size=3,
    )

    async def skip_schema(_conn):
        return None

    monkeypatch.setattr(store, "_initialize_schema", skip_schema)
    await asyncio.gather(store.initialize(), store.initialize())

    assert len(fake_asyncpg.calls) == 2
    common = {
        "host": "db.internal",
        "port": 5544,
        "user": "metadata-user",
        "password": "metadata-password",
        "database": "metadata-db",
    }
    assert fake_asyncpg.calls[0] == {**common, "min_size": 1, "max_size": 1}
    assert fake_asyncpg.calls[1] == {**common, "min_size": 0, "max_size": 3}
    assert store._pool is fake_asyncpg.pools[0]
    assert store._operation_lock_pool is fake_asyncpg.pools[1]

    await store.close()
    assert [pool.closed for pool in fake_asyncpg.pools] == [1, 1]

    # Preserve the existing fake-pool injection pattern while lazily creating
    # exactly one independent operation pool under concurrent first use.
    injected_main_pool = Pool("injected-main")
    lazy_store = postgres_metadata_store_module.PostgresMetadataStore(
        host="db.internal",
        port=5544,
        user="metadata-user",
        password="metadata-password",
        database="metadata-db",
        operation_lock_pool_max_size=2,
    )
    lazy_store._pool = injected_main_pool
    lazy_store._initialized = True
    lazy_pools = await asyncio.gather(
        *(lazy_store._ensure_operation_lock_pool() for _ in range(5))
    )
    assert len(fake_asyncpg.calls) == 3
    assert fake_asyncpg.calls[2] == {**common, "min_size": 0, "max_size": 2}
    assert all(pool is lazy_pools[0] for pool in lazy_pools)
    assert lazy_pools[0] is not injected_main_pool

    await lazy_store.close()
    assert injected_main_pool.closed == 1
    assert fake_asyncpg.pools[2].closed == 1


async def test_postgres_job_and_kb_guards_share_one_operation_session(monkeypatch):
    """Nested job -> KB locking completes with an operation pool of size one."""

    from lightrag.api.postgres_metadata_store import PostgresMetadataStore

    class TransactionContext:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *_args):
            return None

    class Connection:
        def __init__(self):
            self.statements: list[tuple[str, str]] = []

        def transaction(self):
            return TransactionContext()

        async def execute(self, statement: str, value: str):
            self.statements.append((" ".join(statement.split()), value))
            return "SELECT 1"

    class AcquireContext:
        def __init__(self, pool):
            self.pool = pool

        async def __aenter__(self):
            if self.pool.in_use:
                raise AssertionError("single-slot operation pool was re-acquired")
            self.pool.in_use = True
            self.pool.acquired += 1
            return self.pool.connection

        async def __aexit__(self, *_args):
            self.pool.in_use = False
            self.pool.released += 1

    class SingleSlotPool:
        def __init__(self):
            self.connection = Connection()
            self.in_use = False
            self.acquired = 0
            self.released = 0

        def acquire(self):
            return AcquireContext(self)

    class ForbiddenMainPool:
        def acquire(self):
            raise AssertionError("operation guards must not use the main pool")

    operation_pool = SingleSlotPool()
    store = PostgresMetadataStore(
        dsn="postgresql://unused", operation_lock_pool_max_size=1
    )
    store._pool = ForbiddenMainPool()
    store._operation_lock_pool = operation_pool
    store._initialized = True

    async def assert_generation(*_args, **_kwargs):
        return None

    monkeypatch.setattr(store, "assert_kb_generation", assert_generation)
    monkeypatch.setattr(store, "_assert_kb_generation", assert_generation)

    async def nested_guards():
        async with store.job_execution_guard("job-pg-nested") as owned:
            assert owned is True
            async with store.kb_write_guard("kb-pg-nested", "gen-pg-nested"):
                async with store.kb_write_guard(
                    "kb-pg-nested", "gen-pg-nested"
                ):
                    pass

                async def child_reentry():
                    async with store.kb_write_guard(
                        "kb-pg-nested", "gen-pg-nested"
                    ):
                        pass

                await asyncio.wait_for(
                    asyncio.create_task(child_reentry()), timeout=1
                )
                async with store.kb_write_guard(
                    "kb-pg-nested-other", "gen-pg-nested-other"
                ):
                    pass

    await asyncio.wait_for(nested_guards(), timeout=2)
    assert operation_pool.acquired == 1
    assert operation_pool.released == 1
    recorded_statements = operation_pool.connection.statements
    statements = [statement for statement, _ in recorded_statements]
    assert any("1263295563" in statement for statement in statements)
    assert any("1263295562" in statement for statement in statements)
    same_kb_statements = [
        statement
        for statement, lock_id in recorded_statements
        if lock_id == "kb-pg-nested"
    ]
    assert (
        sum("pg_advisory_lock_shared" in statement for statement in same_kb_statements)
        == 1
    )
    assert (
        sum(
            "pg_advisory_unlock_shared" in statement
            for statement in same_kb_statements
        )
        == 1
    )


@pytest.mark.skipif(
    not _POSTGRES_DSN,
    reason="live PostgreSQL nested-lock test requires LIGHTRAG_KB_POSTGRES_TEST_DSN",
)
async def test_postgres_live_nested_job_and_kb_guard_with_single_slot_pool():
    """Optional live hook for asyncpg session identity and max_size=1."""

    from lightrag.api.postgres_metadata_store import PostgresMetadataStore

    store = PostgresMetadataStore(
        dsn=_POSTGRES_DSN,
        min_size=1,
        max_size=1,
        operation_lock_pool_max_size=1,
    )
    await store.initialize()
    kb_id = f"kb_live_nested_{uuid.uuid4().hex}"
    generation = f"gen_live_nested_{uuid.uuid4().hex}"
    try:
        await store.activate_kb_generation(kb_id, generation)

        async def nested():
            async with store.job_execution_guard(
                f"job_live_nested_{uuid.uuid4().hex}"
            ) as owned:
                assert owned is True
                async with store.kb_write_guard(kb_id, generation):
                    pass

        await asyncio.wait_for(nested(), timeout=10)
    finally:
        try:
            await store.purge_kb_metadata(kb_id, generation=generation)
        finally:
            await store.close()


async def test_postgres_operation_guards_unlock_and_release_on_error_and_cancel(
    monkeypatch,
):
    from lightrag.api.postgres_metadata_store import PostgresMetadataStore

    class TransactionContext:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *_args):
            return None

    class Connection:
        def __init__(self):
            self.statements: list[str] = []

        def transaction(self):
            return TransactionContext()

        async def execute(self, statement: str, *_args):
            await asyncio.sleep(0)
            self.statements.append(" ".join(statement.split()))
            return "SELECT 1"

    class AcquireContext:
        def __init__(self, pool):
            self.pool = pool

        async def __aenter__(self):
            self.pool.acquired += 1
            return self.pool.connection

        async def __aexit__(self, *_args):
            self.pool.released += 1
            return None

    class RecordingPool:
        def __init__(self):
            self.connection = Connection()
            self.acquired = 0
            self.released = 0

        def acquire(self):
            return AcquireContext(self)

    class ForbiddenMainPool:
        def acquire(self):
            raise AssertionError("guard must not acquire the main metadata pool")

    store = PostgresMetadataStore(dsn="postgresql://unused")
    operation_pool = RecordingPool()
    store._pool = ForbiddenMainPool()
    store._operation_lock_pool = operation_pool
    store._initialized = True

    async def assert_generation(*_args, **_kwargs):
        return None

    async def begin_deletion(*_args, **_kwargs):
        return object()

    monkeypatch.setattr(store, "assert_kb_generation", assert_generation)
    monkeypatch.setattr(store, "_assert_kb_generation", assert_generation)
    monkeypatch.setattr(store, "_begin_kb_deletion", begin_deletion)

    with pytest.raises(RuntimeError, match="guard body failed"):
        async with store.kb_deletion_guard(
            "kb-pg-operation-cleanup",
            "gen-pg-operation-cleanup",
            "job-pg-operation-cleanup",
        ):
            raise RuntimeError("guard body failed")

    body_entered = asyncio.Event()
    never_release = asyncio.Event()

    async def cancelled_writer():
        async with store.kb_write_guard(
            "kb-pg-operation-cleanup", "gen-pg-operation-cleanup"
        ):
            body_entered.set()
            await never_release.wait()

    task = asyncio.create_task(cancelled_writer())
    await asyncio.wait_for(body_entered.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert operation_pool.acquired == 2
    assert operation_pool.released == 2
    statements = operation_pool.connection.statements
    assert "pg_advisory_lock(hashtextextended($1, 1263295562))" in statements[0]
    assert "pg_advisory_unlock(hashtextextended($1, 1263295562))" in statements[1]
    assert "pg_advisory_lock_shared(hashtextextended($1, 1263295562))" in statements[2]
    assert "pg_advisory_unlock_shared(hashtextextended($1, 1263295562))" in statements[3]


async def test_postgres_operation_guards_preserve_shared_exclusive_order(
    monkeypatch,
):
    from lightrag.api.postgres_metadata_store import PostgresMetadataStore

    class AdvisoryManager:
        def __init__(self):
            self.condition = asyncio.Condition()
            self.readers = 0
            self.writer = False
            self.waiting_writers = 0
            self.exclusive_waiting = asyncio.Event()

        async def acquire_shared(self):
            async with self.condition:
                await self.condition.wait_for(
                    lambda: not self.writer and self.waiting_writers == 0
                )
                self.readers += 1

        async def release_shared(self):
            async with self.condition:
                self.readers -= 1
                self.condition.notify_all()

        async def acquire_exclusive(self):
            async with self.condition:
                self.waiting_writers += 1
                self.exclusive_waiting.set()
                try:
                    await self.condition.wait_for(
                        lambda: not self.writer and self.readers == 0
                    )
                    self.writer = True
                finally:
                    self.waiting_writers -= 1

        async def release_exclusive(self):
            async with self.condition:
                self.writer = False
                self.condition.notify_all()

    class TransactionContext:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *_args):
            return None

    class Connection:
        def __init__(self, manager: AdvisoryManager):
            self.manager = manager

        def transaction(self):
            return TransactionContext()

        async def execute(self, statement: str, kb_id: str):
            assert kb_id == "kb-pg-operation-order"
            normalized = " ".join(statement.split())
            if "pg_advisory_unlock_shared" in normalized:
                await self.manager.release_shared()
            elif "pg_advisory_lock_shared" in normalized:
                await self.manager.acquire_shared()
            elif "pg_advisory_unlock(" in normalized:
                await self.manager.release_exclusive()
            elif "pg_advisory_lock(" in normalized:
                await self.manager.acquire_exclusive()
            return "SELECT 1"

    class AcquireContext:
        def __init__(self, pool):
            self.pool = pool
            self.connection: Connection | None = None

        async def __aenter__(self):
            self.pool.acquired += 1
            self.connection = Connection(self.pool.manager)
            return self.connection

        async def __aexit__(self, *_args):
            self.pool.released += 1
            return None

    class OperationPool:
        def __init__(self, manager: AdvisoryManager):
            self.manager = manager
            self.acquired = 0
            self.released = 0

        def acquire(self):
            return AcquireContext(self)

    class ForbiddenMainPool:
        def acquire(self):
            raise AssertionError("guard must not acquire the main metadata pool")

    manager = AdvisoryManager()
    operation_pool = OperationPool(manager)
    store = PostgresMetadataStore(dsn="postgresql://unused")
    store._pool = ForbiddenMainPool()
    store._operation_lock_pool = operation_pool
    store._initialized = True

    async def assert_generation(*_args, **_kwargs):
        return None

    async def begin_deletion(*_args, **_kwargs):
        return object()

    monkeypatch.setattr(store, "assert_kb_generation", assert_generation)
    monkeypatch.setattr(store, "_assert_kb_generation", assert_generation)
    monkeypatch.setattr(store, "_begin_kb_deletion", begin_deletion)

    order: list[str] = []
    release_shared = asyncio.Event()
    release_exclusive = asyncio.Event()
    shared_a_entered = asyncio.Event()
    shared_b_entered = asyncio.Event()
    exclusive_entered = asyncio.Event()
    shared_c_entered = asyncio.Event()

    async def shared(label: str, entered: asyncio.Event, release: asyncio.Event):
        async with store.kb_write_guard(
            "kb-pg-operation-order", "gen-pg-operation-order"
        ):
            order.append(label)
            entered.set()
            await release.wait()

    async def exclusive():
        async with store.kb_deletion_guard(
            "kb-pg-operation-order",
            "gen-pg-operation-order",
            "job-pg-operation-order",
        ):
            order.append("exclusive")
            exclusive_entered.set()
            await release_exclusive.wait()

    shared_a = asyncio.create_task(
        shared("shared-a", shared_a_entered, release_shared)
    )
    await asyncio.wait_for(shared_a_entered.wait(), timeout=2)
    shared_b = asyncio.create_task(
        shared("shared-b", shared_b_entered, release_shared)
    )
    await asyncio.wait_for(shared_b_entered.wait(), timeout=2)

    exclusive_task = asyncio.create_task(exclusive())
    await asyncio.wait_for(manager.exclusive_waiting.wait(), timeout=2)
    shared_c = asyncio.create_task(
        shared("shared-c", shared_c_entered, asyncio.Event())
    )
    await asyncio.sleep(0.05)
    assert not exclusive_entered.is_set()
    assert not shared_c_entered.is_set()

    release_shared.set()
    await asyncio.wait_for(exclusive_entered.wait(), timeout=2)
    assert not shared_c_entered.is_set()
    release_exclusive.set()
    await asyncio.wait_for(exclusive_task, timeout=2)
    await asyncio.wait_for(shared_c_entered.wait(), timeout=2)
    shared_c.cancel()
    with pytest.raises(asyncio.CancelledError):
        await shared_c
    await asyncio.gather(shared_a, shared_b)

    assert order == ["shared-a", "shared-b", "exclusive", "shared-c"]
    assert operation_pool.acquired == 4
    assert operation_pool.released == 4
