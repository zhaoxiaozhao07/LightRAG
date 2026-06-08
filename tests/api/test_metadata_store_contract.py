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
  it is skipped with a clear reason. Each run uses a unique ``kb_id`` and purges
  it at the end, so it is safe against a shared database.

This closes the gap where the Postgres backend previously had zero behavioral
coverage (all KB tests instantiated SQLite only). Run live coverage with e.g.::

    LIGHTRAG_KB_POSTGRES_TEST_DSN=postgresql://admin:123456@127.0.0.1:5433/knowledge_base \
        uv run pytest tests/api/test_metadata_store_contract.py -q
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest

from lightrag.api.kb_service import utc_now_iso
from lightrag.api.metadata_store import (
    ArtifactRecord,
    AuditEventRecord,
    ConfigVersionRecord,
    DocumentRecord,
    EnterpriseAPIKeyRecord,
    EnterpriseInvitationRecord,
    EnterpriseUserRecord,
    EnterpriseTenantKBACLRecord,
    EnterpriseTenantMembershipRecord,
    IdempotencyKeyConflictError,
    InvalidJobTransitionError,
    JobRecord,
    KBACLRecord,
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

    store = PostgresMetadataStore(dsn=_POSTGRES_DSN)
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
            "token_version": user.token_version + 1,
            "updated_at": utc_now_iso(),
        }
    )
    saved_user = await store.upsert_enterprise_user(updated_user)
    assert saved_user.can_create_kb is True
    assert saved_user.can_use_bypass_query is True
    assert saved_user.token_version == 2

    await store.set_enterprise_system_setting(
        "registration_enabled", "true", updated_by=user.id
    )
    assert await store.get_enterprise_system_setting("registration_enabled") == "true"
    assert await store.get_enterprise_system_setting("missing", "fallback") == "fallback"

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

    assert await store.delete_kb_acl(kb_id, user.id) is True
    assert await store.get_kb_acl_role(kb_id, user.id) is None


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


async def test_purge_kb_metadata_removes_everything(store):
    kb_id = _unique_kb(store)
    await store.create_documents_and_job(
        [_doc(kb_id, "doc_a")], _job(kb_id, "job_x", document_id="doc_a")
    )
    purged = await store.purge_kb_metadata(kb_id)
    assert purged["documents"] >= 1
    assert purged["jobs"] >= 1
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
