"""End-to-end tests for the Phase 4 person-identity routes.

Builds a FastAPI TestClient with the enterprise + person routers mounted, then
exercises enroll/login/accounts/switch/logout/logout-all/change-password and
the super-admin management surface (grant create/revoke, link propose/unbind,
disable/enable), plus the negative cases from docs/多账号身份关联与切换执行文档.md
section 5.11.
"""

from __future__ import annotations

# ruff: noqa: E402 - LightRAG API modules parse sys.argv at import time.

import asyncio
import sys
from types import SimpleNamespace

sys.argv = [sys.argv[0]]

import importlib
from pathlib import Path

import pytest

config_module = importlib.import_module("lightrag.api.config")

pytestmark = pytest.mark.offline

_PERSON_SECRET = "person-route-test-secret-distinct-from-legacy-0123456789"
_LEGACY_SECRET = "legacy-route-test-secret-distinct-from-person-secret"
_API_KEY = "test-api-key-for-person-routes"


def _patch_person_args(monkeypatch):
    """Patch global_args with enterprise + person auth enabled."""

    args = SimpleNamespace(
        # Legacy auth (AuthHandler reads these at module construction; some
        # auth tests reload the auth module, so the values must be present).
        # NOTE: token_secret is intentionally NOT overridden here — the
        # auth_handler singleton was built at import time with whatever secret
        # it has, and create_token/validate_token must use that same singleton.
        # Overriding token_secret in global_args would not affect the already-
        # constructed handler and would break token round-trips.
        jwt_algorithm="HS256",
        token_expire_hours=48,
        guest_token_expire_hours=24,
        auth_accounts="",
        enterprise_auth_enabled=True,
        enterprise_legacy_api_key_superadmin=False,
        enterprise_disable_global_routes=True,
        enterprise_rate_limit_enabled=False,
        enterprise_rate_limit_requests=60,
        enterprise_rate_limit_window_seconds=60.0,
        enterprise_tenant_rate_limit_requests=0,
        enterprise_tenant_rate_limit_window_seconds=60.0,
        enterprise_quota_requests=0,
        enterprise_quota_window_seconds=86400.0,
        enterprise_tenant_quota_requests=0,
        enterprise_tenant_quota_window_seconds=86400.0,
        enterprise_artifact_download_min_role="kb_viewer",
        enterprise_tenant_admin_oversight_role="kb_viewer",
        enterprise_artifact_download_policy="",
        enterprise_artifact_action_policy="",
        enterprise_mask_storage_uris=True,
        enterprise_registration_max_attempts=10,
        enterprise_registration_window_seconds=300.0,
        enterprise_registration_lockout_seconds=900.0,
        token_auto_renew=False,
        token_renew_threshold=0.5,
        # Person auth enabled with a distinct secret.
        person_auth_enabled=True,
        person_token_secret=_PERSON_SECRET,
        person_access_token_ttl=3600,
        person_session_ttl=28800,
        person_login_max_attempts=5,
        person_password_min_length=8,
        whitelist_paths="/health",
    )
    # Re-import the LIVE config module (other auth tests may have popped and
    # recreated it); monkeypatching a stale reference would no-op.
    live_config = importlib.import_module("lightrag.api.config")
    monkeypatch.setattr(live_config, "global_args", args)
    # utils_api reads global_args at request time; also patch its attribute.
    utils_api = importlib.import_module("lightrag.api.utils_api")
    monkeypatch.setattr(utils_api, "global_args", args)
    monkeypatch.setattr(utils_api, "whitelist_patterns", [])
    # Reset the lazy person_token_handler singleton so it picks up the secret.
    person_auth_mod = importlib.import_module("lightrag.api.person_auth")
    person_auth_mod.person_token_handler = None
    return args


def _build_client(monkeypatch, tmp_path: Path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from lightrag.api.enterprise_auth import (
        AuditService,
        AuthorizationService,
        EnterpriseLimitService,
        InvitationService,
        ServiceAPIKeyService,
        SystemSettingsService,
        UserKBQuerySettingsService,
        UserService,
        set_active_metadata_store,
    )
    from lightrag.api.kb_service import KnowledgeBaseService
    from lightrag.api.metadata_store import SQLiteMetadataStore
    from lightrag.api.person_auth import PersonService, PersonTokenHandler
    from lightrag.api.routers.enterprise_routes import create_enterprise_routes
    from lightrag.api.routers.person_routes import create_person_routes

    _patch_person_args(monkeypatch)

    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kb.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    audit_service = AuditService(metadata_store)
    user_service = UserService(metadata_store, audit_service)
    settings_service = SystemSettingsService(metadata_store)
    api_key_service = ServiceAPIKeyService(
        metadata_store, audit_service, kb_service=kb_service
    )
    invitation_service = InvitationService(metadata_store, audit_service)
    limit_service = EnterpriseLimitService(audit_service)
    authz_service = AuthorizationService(
        metadata_store, audit_service, kb_service=kb_service
    )
    user_kb_query_settings_service = UserKBQuerySettingsService(
        metadata_store, audit_service
    )
    person_token_handler = PersonTokenHandler()
    person_service = PersonService(
        metadata_store,
        person_token_handler,
        login_max_attempts=5,
        password_min_length=8,
    )

    def seed():
        async def _seed():
            await kb_service.initialize()
            await metadata_store.initialize()
            await settings_service.initialize_registration_setting(False)
            admin = await user_service.bootstrap_super_admin(
                username="admin", password="admin-pass", password_hash=None
            )
            alice = await user_service.create_user(
                username="alice", password="alice-pass", can_create_kb=True
            )
            bob = await user_service.create_user(
                username="bob", password="bob-pass"
            )
            return admin, alice, bob

        return asyncio.run(_seed())

    admin, alice, bob = seed()
    # Register the metadata store so person-session validation resolves it.
    set_active_metadata_store(metadata_store)

    app = FastAPI()
    app.state.enterprise_enabled = True
    app.state.kb_service = kb_service
    app.state.metadata_store = metadata_store
    app.state.enterprise_user_service = user_service
    app.state.enterprise_settings_service = settings_service
    app.state.enterprise_user_kb_query_settings_service = user_kb_query_settings_service
    app.state.enterprise_api_key_service = api_key_service
    app.state.enterprise_invitation_service = invitation_service
    app.state.enterprise_limit_service = limit_service
    app.state.enterprise_authorization_service = authz_service
    app.state.enterprise_audit_service = audit_service
    app.state.person_service = person_service
    app.state.person_token_handler = person_token_handler

    app.include_router(
        create_enterprise_routes(api_key=_API_KEY, kb_service=kb_service)
    )
    app.include_router(create_person_routes(api_key=_API_KEY, kb_service=kb_service))

    client = TestClient(app)
    return (
        client,
        user_service,
        person_service,
        metadata_store,
        admin,
        alice,
        bob,
    )


def _legacy_token(user_service, user) -> str:
    # Use the SAME auth_handler reference that combined_auth captured at import
    # time. Other auth tests may have reloaded lightrag.api.auth, creating a
    # new auth_handler singleton that combined_auth does not see; a token
    # signed by the reloaded singleton would fail validation.
    import lightrag.api.utils_api as utils_api

    handler = getattr(utils_api, "auth_handler", None)
    if handler is None:
        from lightrag.api.auth import auth_handler as handler  # type: ignore

    return handler.create_token(
        username=user.username,
        role=user.system_role,
        metadata=user_service.token_metadata_for_user(user),
    )


def _admin_headers(user_service, admin) -> dict[str, str]:
    return {"Authorization": f"Bearer {_legacy_token(user_service, admin)}"}


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_enroll_login_accounts_switch_logout(monkeypatch, tmp_path):
    (
        client,
        user_service,
        person_service,
        metadata_store,
        admin,
        alice,
        bob,
    ) = _build_client(monkeypatch, tmp_path)

    # Super admin issues an enrollment grant for alice's account.
    grant_resp = client.post(
        "/admin/persons/enrollment-grants",
        json={"account_id": alice.id, "ttl_seconds": 900},
        headers=_admin_headers(user_service, admin),
    )
    assert grant_resp.status_code == 201, grant_resp.text
    grant = grant_resp.json()
    assert "grant_token" in grant
    grant_token = grant["grant_token"]

    # Public enroll with the grant.
    enroll_resp = client.post(
        "/auth/person/enroll",
        json={"grant_token": grant_token, "person_password": "RightPass-1"},
    )
    assert enroll_resp.status_code == 201, enroll_resp.text
    enrolled = enroll_resp.json()
    person_id = enrolled["person"]["person_id"] if "person_id" in enrolled["person"] else enrolled["person"]["id"]
    access_token = enrolled["access_token"]
    assert enrolled["active_account"]["account_id"] == alice.id
    headers = {"Authorization": f"Bearer {access_token}"}

    # List accounts.
    accounts_resp = client.get("/auth/person/accounts", headers=headers)
    assert accounts_resp.status_code == 200, accounts_resp.text
    accounts = accounts_resp.json()
    assert accounts["active_account_id"] == alice.id
    assert any(a["account_id"] == alice.id for a in accounts["accounts"])

    # login path (separate from enroll): wrong password first.
    bad_login = client.post(
        "/auth/person/login",
        json={
            "person_id": person_id,
            "person_password": "wrong",
            "account_id": alice.id,
        },
    )
    assert bad_login.status_code == 401

    # Correct login.
    login_resp = client.post(
        "/auth/person/login",
        json={
            "person_id": person_id,
            "person_password": "RightPass-1",
            "account_id": alice.id,
        },
    )
    assert login_resp.status_code == 200, login_resp.text
    logged_in = login_resp.json()
    headers = {"Authorization": f"Bearer {logged_in['access_token']}"}

    # Add a second account, propose + confirm link, then switch.
    # Enroll would create a second person; instead propose a link to bob for
    # the existing person and confirm it.
    propose = client.post(
        f"/admin/persons/{person_id}/accounts/{bob.id}",
        json={"reason": "second account"},
        headers=_admin_headers(user_service, admin),
    )
    assert propose.status_code == 201, propose.text
    # Confirm via the person session-control path.
    confirm = client.post(
        f"/auth/person/links/{bob.id}:confirm",
        json={"person_password": "RightPass-1"},
        headers=headers,
    )
    # Confirm bumps auth_epoch and revokes the current session; re-login.
    if confirm.status_code == 200:
        # Re-login to get a fresh session, then switch alice -> bob.
        relogin = client.post(
            "/auth/person/login",
            json={
                "person_id": person_id,
                "person_password": "RightPass-1",
                "account_id": alice.id,
            },
        ).json()
        headers = {"Authorization": f"Bearer {relogin['access_token']}"}
        switch = client.post(
            "/auth/person/switch",
            json={"account_id": bob.id},
            headers=headers,
        )
        assert switch.status_code == 200, switch.text
        switched = switch.json()
        assert switched["active_account"]["account_id"] == bob.id
        headers = {"Authorization": f"Bearer {switched['access_token']}"}

    # Logout (single session).
    logout = client.post("/auth/person/logout", headers=headers)
    assert logout.status_code == 200
    assert logout.json()["status"] == "logged_out"

    # logout-all.
    relogin = client.post(
        "/auth/person/login",
        json={
            "person_id": person_id,
            "person_password": "RightPass-1",
            "account_id": alice.id,
        },
    ).json()
    headers = {"Authorization": f"Bearer {relogin['access_token']}"}
    logout_all = client.post("/auth/person/logout-all", headers=headers)
    assert logout_all.status_code == 200
    assert logout_all.json()["status"] == "logged_out_all"


def test_change_password_requires_relogin(monkeypatch, tmp_path):
    (
        client,
        user_service,
        person_service,
        metadata_store,
        admin,
        alice,
        bob,
    ) = _build_client(monkeypatch, tmp_path)

    grant = client.post(
        "/admin/persons/enrollment-grants",
        json={"account_id": alice.id},
        headers=_admin_headers(user_service, admin),
    ).json()
    enrolled = client.post(
        "/auth/person/enroll",
        json={"grant_token": grant["grant_token"], "person_password": "OldPass-1"},
    ).json()
    person_id = enrolled["person"].get("person_id") or enrolled["person"]["id"]
    headers = {"Authorization": f"Bearer {enrolled['access_token']}"}

    # Wrong current password rejected.
    bad = client.post(
        "/auth/person/change-password",
        json={"current_person_password": "wrong", "new_person_password": "NewPass-2"},
        headers=headers,
    )
    assert bad.status_code == 401

    # Correct change.
    ok = client.post(
        "/auth/person/change-password",
        json={"current_person_password": "OldPass-1", "new_person_password": "NewPass-2"},
        headers=headers,
    )
    assert ok.status_code == 200
    assert ok.json()["status"] == "password_changed"

    # Old password no longer logs in; new password does.
    assert (
        client.post(
            "/auth/person/login",
            json={
                "person_id": person_id,
                "person_password": "OldPass-1",
                "account_id": alice.id,
            },
        ).status_code
        == 401
    )
    new_login = client.post(
        "/auth/person/login",
        json={
            "person_id": person_id,
            "person_password": "NewPass-2",
            "account_id": alice.id,
        },
    )
    assert new_login.status_code == 200


def test_admin_disable_enable_person(monkeypatch, tmp_path):
    (
        client,
        user_service,
        person_service,
        metadata_store,
        admin,
        alice,
        bob,
    ) = _build_client(monkeypatch, tmp_path)

    grant = client.post(
        "/admin/persons/enrollment-grants",
        json={"account_id": alice.id},
        headers=_admin_headers(user_service, admin),
    ).json()
    enrolled = client.post(
        "/auth/person/enroll",
        json={"grant_token": grant["grant_token"], "person_password": "RightPass-1"},
    ).json()
    person_id = enrolled["person"].get("person_id") or enrolled["person"]["id"]

    disable = client.post(
        f"/admin/persons/{person_id}:disable", headers=_admin_headers(user_service, admin)
    )
    assert disable.status_code == 200
    assert disable.json()["status"] == "disabled"

    # Disabled person cannot log in.
    login = client.post(
        "/auth/person/login",
        json={
            "person_id": person_id,
            "person_password": "RightPass-1",
            "account_id": alice.id,
        },
    )
    assert login.status_code in (401, 403)

    enable = client.post(
        f"/admin/persons/{person_id}:enable", headers=_admin_headers(user_service, admin)
    )
    assert enable.status_code == 200
    assert enable.json()["status"] == "active"


def test_admin_unbind_link(monkeypatch, tmp_path):
    (
        client,
        user_service,
        person_service,
        metadata_store,
        admin,
        alice,
        bob,
    ) = _build_client(monkeypatch, tmp_path)

    grant = client.post(
        "/admin/persons/enrollment-grants",
        json={"account_id": alice.id},
        headers=_admin_headers(user_service, admin),
    ).json()
    enrolled = client.post(
        "/auth/person/enroll",
        json={"grant_token": grant["grant_token"], "person_password": "RightPass-1"},
    ).json()
    person_id = enrolled["person"].get("person_id") or enrolled["person"]["id"]

    unbind = client.delete(
        f"/admin/persons/{person_id}/accounts/{alice.id}",
        headers=_admin_headers(user_service, admin),
    )
    assert unbind.status_code == 200
    assert unbind.json()["status"] == "unlinked"
    assert unbind.json()["account_id"] == alice.id


def test_admin_grant_revoke_idempotent(monkeypatch, tmp_path):
    (
        client,
        user_service,
        person_service,
        metadata_store,
        admin,
        alice,
        bob,
    ) = _build_client(monkeypatch, tmp_path)

    grant = client.post(
        "/admin/persons/enrollment-grants",
        json={"account_id": alice.id},
        headers=_admin_headers(user_service, admin),
    ).json()
    grant_id = grant["grant_id"]

    revoke = client.delete(
        f"/admin/persons/enrollment-grants/{grant_id}",
        headers=_admin_headers(user_service, admin),
    )
    assert revoke.status_code == 200
    # Idempotent: revoke again on an already-revoked grant.
    revoke2 = client.delete(
        f"/admin/persons/enrollment-grants/{grant_id}",
        headers=_admin_headers(user_service, admin),
    )
    assert revoke2.status_code == 200


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


def test_non_super_admin_cannot_create_grant(monkeypatch, tmp_path):
    (
        client,
        user_service,
        person_service,
        metadata_store,
        admin,
        alice,
        bob,
    ) = _build_client(monkeypatch, tmp_path)

    # alice is a regular user; she cannot issue a grant.
    alice_headers = {"Authorization": f"Bearer {_legacy_token(user_service, alice)}"}
    resp = client.post(
        "/admin/persons/enrollment-grants",
        json={"account_id": bob.id},
        headers=alice_headers,
    )
    assert resp.status_code == 403


def test_non_super_admin_cannot_propose_link(monkeypatch, tmp_path):
    (
        client,
        user_service,
        person_service,
        metadata_store,
        admin,
        alice,
        bob,
    ) = _build_client(monkeypatch, tmp_path)

    alice_headers = {"Authorization": f"Bearer {_legacy_token(user_service, alice)}"}
    resp = client.post(
        f"/admin/persons/per_foo/accounts/{bob.id}",
        headers=alice_headers,
    )
    assert resp.status_code == 403


def test_enroll_with_consumed_or_invalid_grant(monkeypatch, tmp_path):
    (
        client,
        user_service,
        person_service,
        metadata_store,
        admin,
        alice,
        bob,
    ) = _build_client(monkeypatch, tmp_path)

    grant = client.post(
        "/admin/persons/enrollment-grants",
        json={"account_id": alice.id},
        headers=_admin_headers(user_service, admin),
    ).json()
    grant_token = grant["grant_token"]

    # First enroll succeeds.
    first = client.post(
        "/auth/person/enroll",
        json={"grant_token": grant_token, "person_password": "RightPass-1"},
    )
    assert first.status_code == 201

    # Re-using the consumed grant must fail with 401 invalid_grant.
    second = client.post(
        "/auth/person/enroll",
        json={"grant_token": grant_token, "person_password": "RightPass-1"},
    )
    assert second.status_code == 401

    # A bogus grant token is also rejected.
    bogus = client.post(
        "/auth/person/enroll",
        json={"grant_token": "not-a-real-token", "person_password": "RightPass-1"},
    )
    assert bogus.status_code == 401


def test_switch_to_unlinked_account_404(monkeypatch, tmp_path):
    (
        client,
        user_service,
        person_service,
        metadata_store,
        admin,
        alice,
        bob,
    ) = _build_client(monkeypatch, tmp_path)

    grant = client.post(
        "/admin/persons/enrollment-grants",
        json={"account_id": alice.id},
        headers=_admin_headers(user_service, admin),
    ).json()
    enrolled = client.post(
        "/auth/person/enroll",
        json={"grant_token": grant["grant_token"], "person_password": "RightPass-1"},
    ).json()
    headers = {"Authorization": f"Bearer {enrolled['access_token']}"}

    # bob is NOT linked -> switch must fail with 404 account_not_linked.
    switch = client.post(
        "/auth/person/switch", json={"account_id": bob.id}, headers=headers
    )
    assert switch.status_code == 404


def test_session_control_rejects_legacy_token(monkeypatch, tmp_path):
    (
        client,
        user_service,
        person_service,
        metadata_store,
        admin,
        alice,
        bob,
    ) = _build_client(monkeypatch, tmp_path)

    # A legacy JWT (no kid) cannot reach the session-control endpoints.
    legacy_headers = {"Authorization": f"Bearer {_legacy_token(user_service, alice)}"}
    resp = client.get("/auth/person/accounts", headers=legacy_headers)
    # The v2 validator rejects the unsigned-as-v2 token.
    assert resp.status_code in (401, 403)


def test_enroll_rejects_weak_password(monkeypatch, tmp_path):
    (
        client,
        user_service,
        person_service,
        metadata_store,
        admin,
        alice,
        bob,
    ) = _build_client(monkeypatch, tmp_path)

    grant = client.post(
        "/admin/persons/enrollment-grants",
        json={"account_id": alice.id},
        headers=_admin_headers(user_service, admin),
    ).json()
    resp = client.post(
        "/auth/person/enroll",
        json={"grant_token": grant["grant_token"], "person_password": "short"},
    )
    assert resp.status_code == 400


def test_create_grant_rejects_super_admin_target(monkeypatch, tmp_path):
    (
        client,
        user_service,
        person_service,
        metadata_store,
        admin,
        alice,
        bob,
    ) = _build_client(monkeypatch, tmp_path)

    # The admin account is super_admin; binding it must be rejected.
    resp = client.post(
        "/admin/persons/enrollment-grants",
        json={"account_id": admin.id},
        headers=_admin_headers(user_service, admin),
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# person token on business surfaces (interactive parity)
# ---------------------------------------------------------------------------


def _enroll_person_token(client, user_service, admin, account) -> tuple[str, str]:
    """Enroll a person for ``account`` and return (person_id, access_token)."""

    grant = client.post(
        "/admin/persons/enrollment-grants",
        json={"account_id": account.id, "ttl_seconds": 900},
        headers=_admin_headers(user_service, admin),
    ).json()
    enrolled = client.post(
        "/auth/person/enroll",
        json={"grant_token": grant["grant_token"], "person_password": "RightPass-1"},
    ).json()
    return enrolled["person"]["id"], enrolled["access_token"]


def test_person_token_is_interactive_on_business_endpoints(monkeypatch, tmp_path):
    """A v2 person access token must behave exactly like the account's own
    interactive login JWT on business surfaces gated by
    ``require_interactive_user_principal`` (doc 5.10: the account-access path
    builds a plain account Principal)."""

    (
        client,
        user_service,
        person_service,
        metadata_store,
        admin,
        alice,
        bob,
    ) = _build_client(monkeypatch, tmp_path)

    _person_id, access_token = _enroll_person_token(
        client, user_service, admin, alice
    )
    person_headers = {"Authorization": f"Bearer {access_token}"}

    # PATCH /auth/me is guarded by require_interactive_user_principal and was
    # 403 for person tokens before INTERACTIVE_AUTH_METHODS.
    resp = client.patch(
        "/auth/me",
        json={"display_name": "Alice via person token"},
        headers=person_headers,
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    # The Principal is account-scoped (I-3) and tagged with the person path.
    assert payload["principal"]["user_id"] == alice.id
    assert payload["principal"]["auth_method"] == "person_jwt"

    # Service API keys must still be rejected as non-interactive.
    resp = client.patch(
        "/auth/me",
        json={"display_name": "nope"},
        headers={"X-API-Key": _API_KEY},
    )
    assert resp.status_code in (401, 403)


# ---------------------------------------------------------------------------
# tenant-admin link proposal matrix
# ---------------------------------------------------------------------------


def test_tenant_admin_link_proposal_matrix(monkeypatch, tmp_path):
    """Tenant admins may propose pending links for regular members of their
    own tenant only; tenant-admin targets and cross-tenant targets stay
    super-admin territory; plain members cannot propose at all."""

    (
        client,
        user_service,
        person_service,
        metadata_store,
        admin,
        alice,
        bob,
    ) = _build_client(monkeypatch, tmp_path)
    authz = client.app.state.enterprise_authorization_service

    async def _seed_tenants():
        # alice: tenant-a admin; bob: tenant-a member; carol: tenant-b member;
        # dave: a second tenant-a admin (admin-role target).
        carol = await user_service.create_user(
            username="carol", password="carol-pass"
        )
        dave = await user_service.create_user(username="dave", password="dave-pass")
        await authz.grant_tenant_membership(
            "tenant-a", alice.id, "tenant_admin", granted_by=admin.id
        )
        await authz.grant_tenant_membership(
            "tenant-a", bob.id, "tenant_member", granted_by=admin.id
        )
        await authz.grant_tenant_membership(
            "tenant-b", carol.id, "tenant_member", granted_by=admin.id
        )
        await authz.grant_tenant_membership(
            "tenant-a", dave.id, "tenant_admin", granted_by=admin.id
        )
        # Re-read so legacy tokens embed the post-grant token_version/tenant.
        refreshed = {}
        for user in (alice, bob, carol, dave):
            refreshed[user.username] = await user_service.get_user_or_404(user.id)
        return refreshed

    users = asyncio.run(_seed_tenants())
    alice_r, bob_r, carol_r, dave_r = (
        users["alice"],
        users["bob"],
        users["carol"],
        users["dave"],
    )

    # An enrolled person to attach proposals to (enrolled on carol/tenant-b).
    person_id, _token = _enroll_person_token(client, user_service, admin, carol_r)

    alice_headers = {
        "Authorization": f"Bearer {_legacy_token(user_service, alice_r)}"
    }
    bob_headers = {"Authorization": f"Bearer {_legacy_token(user_service, bob_r)}"}

    # 1) Tenant admin -> own-tenant regular member: allowed, pending link.
    ok = client.post(
        f"/admin/persons/{person_id}/accounts/{bob_r.id}",
        json={"reason": "dept onboarding"},
        headers=alice_headers,
    )
    assert ok.status_code == 201, ok.text
    assert ok.json()["link"]["status"] == "pending"
    assert ok.json()["link"]["bound_by"] == alice_r.id

    # 2) Tenant admin -> cross-tenant target: hidden as 404.
    cross = client.post(
        f"/admin/persons/{person_id}/accounts/{carol_r.id}",
        headers=alice_headers,
    )
    assert cross.status_code == 404, cross.text
    assert cross.json()["detail"]["error_code"] == "account_not_found"

    # 3) Tenant admin -> tenant-admin target: super admin required.
    peer = client.post(
        f"/admin/persons/{person_id}/accounts/{dave_r.id}",
        headers=alice_headers,
    )
    assert peer.status_code == 403, peer.text
    assert peer.json()["detail"]["error_code"] == "super_admin_required"

    # 4) Regular member cannot propose at all.
    member = client.post(
        f"/admin/persons/{person_id}/accounts/{bob_r.id}",
        headers=bob_headers,
    )
    assert member.status_code == 403, member.text
    assert member.json()["detail"]["error_code"] == "admin_required"

    # 5) Super admin retains the superset: may target the tenant-admin account.
    su = client.post(
        f"/admin/persons/{person_id}/accounts/{dave_r.id}",
        json={"reason": "cross-dept binding"},
        headers=_admin_headers(user_service, admin),
    )
    assert su.status_code == 201, su.text
    assert su.json()["link"]["status"] == "pending"


# ---------------------------------------------------------------------------
# cross-tenant switch isolation (the target user scenario)
# ---------------------------------------------------------------------------


def test_cross_tenant_switch_isolates_business_writes(monkeypatch, tmp_path):
    """One natural person, two department accounts in different tenants:
    business writes made under one account never leak to the other, and
    switching flips the whole account context (I-4)."""

    (
        client,
        user_service,
        person_service,
        metadata_store,
        admin,
        alice,
        bob,
    ) = _build_client(monkeypatch, tmp_path)
    authz = client.app.state.enterprise_authorization_service

    async def _seed():
        await authz.grant_tenant_membership(
            "tenant-fin", alice.id, "tenant_member", granted_by=admin.id
        )
        await authz.grant_tenant_membership(
            "tenant-legal", bob.id, "tenant_member", granted_by=admin.id
        )
        return await user_service.get_user_or_404(alice.id)

    alice_r = asyncio.run(_seed())

    person_id, token_a = _enroll_person_token(client, user_service, admin, alice_r)
    headers_a = {"Authorization": f"Bearer {token_a}"}

    # Business write under the finance account.
    patched = client.patch(
        "/auth/me", json={"display_name": "Fin Alice"}, headers=headers_a
    )
    assert patched.status_code == 200, patched.text

    # Bind the legal-department account and confirm (kills all sessions).
    propose = client.post(
        f"/admin/persons/{person_id}/accounts/{bob.id}",
        json={"reason": "second department"},
        headers=_admin_headers(user_service, admin),
    )
    assert propose.status_code == 201, propose.text
    confirm = client.post(
        f"/auth/person/links/{bob.id}:confirm",
        json={"person_password": "RightPass-1"},
        headers=headers_a,
    )
    assert confirm.status_code == 200, confirm.text

    # Fresh person login straight into the legal account.
    login_b = client.post(
        "/auth/person/login",
        json={
            "person_id": person_id,
            "person_password": "RightPass-1",
            "account_id": bob.id,
        },
    )
    assert login_b.status_code == 200, login_b.text
    headers_b = {"Authorization": f"Bearer {login_b.json()['access_token']}"}

    # The legal account context: bob, tenant-legal, display name untouched.
    me_b = client.get("/auth/me", headers=headers_b)
    assert me_b.status_code == 200, me_b.text
    assert me_b.json()["user"]["username"] == "bob"
    assert me_b.json()["user"]["tenant_id"] == "tenant-legal"
    assert me_b.json()["user"]["display_name"] is None

    # Switch back to finance within the same session.
    switch = client.post(
        "/auth/person/switch",
        json={"account_id": alice_r.id},
        headers=headers_b,
    )
    assert switch.status_code == 200, switch.text
    body = switch.json()
    assert body["active_account"]["tenant_id"] == "tenant-fin"
    assert body["expires_in"] > 0
    headers_a2 = {"Authorization": f"Bearer {body['access_token']}"}

    # Old token (pre-switch epoch) is dead immediately.
    stale = client.get("/auth/me", headers=headers_b)
    assert stale.status_code == 401

    # Finance context restored, with the earlier write intact.
    me_a2 = client.get("/auth/me", headers=headers_a2)
    assert me_a2.status_code == 200, me_a2.text
    assert me_a2.json()["user"]["username"] == "alice"
    assert me_a2.json()["user"]["tenant_id"] == "tenant-fin"
    assert me_a2.json()["user"]["display_name"] == "Fin Alice"

    # Both departments visible in the switchable set.
    accounts = client.get("/auth/person/accounts", headers=headers_a2)
    assert accounts.status_code == 200
    tenants = {a["tenant_id"] for a in accounts.json()["accounts"]}
    assert tenants == {"tenant-fin", "tenant-legal"}


# ---------------------------------------------------------------------------
# person KB shares: cross-department personal KB usage
# ---------------------------------------------------------------------------


def test_person_kb_share_flow_and_department_admin_oversight(monkeypatch, tmp_path):
    """The target user scenario end-to-end: alice (tenant-fin) shares her
    personal KB to her own legal-department account (bob). Zero-copy: bob's
    account gets a direct ACL on the SAME kb_id; tenant-legal's admin gains
    the configured oversight floor; other principals gain nothing; revoking
    the share cuts both immediately."""

    (
        client,
        user_service,
        person_service,
        metadata_store,
        admin,
        alice,
        bob,
    ) = _build_client(monkeypatch, tmp_path)
    authz = client.app.state.enterprise_authorization_service
    kb_service = client.app.state.kb_service

    from lightrag.api.enterprise_auth import principal_from_user

    async def _seed():
        legal_admin = await user_service.create_user(
            username="legal_admin", password="legal-admin-pass"
        )
        carol = await user_service.create_user(
            username="carol", password="carol-pass"
        )
        await authz.grant_tenant_membership(
            "tenant-fin", alice.id, "tenant_member", granted_by=admin.id
        )
        await authz.grant_tenant_membership(
            "tenant-legal", bob.id, "tenant_member", granted_by=admin.id
        )
        await authz.grant_tenant_membership(
            "tenant-legal", legal_admin.id, "tenant_admin", granted_by=admin.id
        )
        await authz.grant_tenant_membership(
            "tenant-legal", carol.id, "tenant_member", granted_by=admin.id
        )
        record = await kb_service.create(
            name="alice-personal", owner_id=alice.id, visibility="private"
        )
        # The real KB-create route grants the creator an owner ACL; mirror it.
        await authz.grant_kb_role(
            record.id, alice.id, "kb_owner", granted_by=admin.id
        )
        refreshed = {}
        for user in (alice, bob, legal_admin, carol):
            refreshed[user.username] = await user_service.get_user_or_404(user.id)
        return record, refreshed

    record, users = asyncio.run(_seed())
    alice_r, bob_r = users["alice"], users["bob"]
    legal_admin_r, carol_r = users["legal_admin"], users["carol"]

    async def _resolve(user):
        memberships = await metadata_store.list_user_tenant_memberships(user.id)
        principal = principal_from_user(
            user, auth_method="jwt", memberships=memberships
        )
        return await authz.resolve_kb_access(principal, record)

    # Enroll alice's person and confirm bob as a second link.
    person_id, person_token = _enroll_person_token(
        client, user_service, admin, alice_r
    )
    headers_person = {"Authorization": f"Bearer {person_token}"}
    propose = client.post(
        f"/admin/persons/{person_id}/accounts/{bob_r.id}",
        json={"reason": "legal dept"},
        headers=_admin_headers(user_service, admin),
    )
    assert propose.status_code == 201, propose.text
    confirm = client.post(
        f"/auth/person/links/{bob_r.id}:confirm",
        json={"person_password": "RightPass-1"},
        headers=headers_person,
    )
    assert confirm.status_code == 200, confirm.text

    # Before sharing: bob and legal admin have no access to alice's KB.
    assert asyncio.run(_resolve(bob_r)).effective_role is None
    assert asyncio.run(_resolve(legal_admin_r)).effective_role is None

    # Alice shares her KB to her own legal-department account.
    alice_headers = {
        "Authorization": f"Bearer {_legacy_token(user_service, alice_r)}"
    }
    share = client.post(
        f"/kbs/{record.id}/person-shares",
        json={"target_account_id": bob_r.id, "role": "kb_editor"},
        headers=alice_headers,
    )
    assert share.status_code == 201, share.text
    assert share.json()["share"]["status"] == "active"
    assert share.json()["share"]["target_tenant_id"] == "tenant-legal"

    # Target account: direct editor access to the SAME kb (no rebuild).
    bob_decision = asyncio.run(_resolve(bob_r))
    assert bob_decision.effective_role == "kb_editor"
    assert "direct" in bob_decision.sources

    # Department admin: oversight floor, via the person-share signal.
    admin_decision = asyncio.run(_resolve(legal_admin_r))
    assert admin_decision.effective_role == "kb_viewer"
    assert "person_share_oversight" in admin_decision.sources
    # The KB shows up in the legal admin's listing.
    async def _filtered(user):
        memberships = await metadata_store.list_user_tenant_memberships(user.id)
        principal = principal_from_user(
            user, auth_method="jwt", memberships=memberships
        )
        return await authz.filter_kbs_for_principal(principal, [record])

    assert asyncio.run(_filtered(legal_admin_r)) == [record]

    # Ordinary legal member: nothing (sharing exposes to the admin, not the
    # whole department).
    assert asyncio.run(_resolve(carol_r)).effective_role is None

    # Person-level listing shows the share.
    mine = client.get("/auth/person/kb-shares", headers=headers_person)
    if mine.status_code == 401:
        # confirm revoked the person session; re-login.
        relogin = client.post(
            "/auth/person/login",
            json={
                "person_id": person_id,
                "person_password": "RightPass-1",
                "account_id": alice_r.id,
            },
        ).json()
        headers_person = {"Authorization": f"Bearer {relogin['access_token']}"}
        mine = client.get("/auth/person/kb-shares", headers=headers_person)
    assert mine.status_code == 200, mine.text
    assert any(s["kb_id"] == record.id for s in mine.json()["shares"])

    # Owner listing on the KB.
    listing = client.get(
        f"/kbs/{record.id}/person-shares", headers=alice_headers
    )
    assert listing.status_code == 200, listing.text
    assert len(listing.json()["shares"]) == 1

    # Revoke: both the target account and the department admin lose access.
    revoke = client.delete(
        f"/kbs/{record.id}/person-shares/{bob_r.id}", headers=alice_headers
    )
    assert revoke.status_code == 200, revoke.text
    assert revoke.json()["status"] == "unshared"
    assert asyncio.run(_resolve(bob_r)).effective_role is None
    assert asyncio.run(_resolve(legal_admin_r)).effective_role is None


def test_person_kb_share_rejections(monkeypatch, tmp_path):
    (
        client,
        user_service,
        person_service,
        metadata_store,
        admin,
        alice,
        bob,
    ) = _build_client(monkeypatch, tmp_path)
    kb_service = client.app.state.kb_service

    async def _seed():
        carol = await user_service.create_user(
            username="carol", password="carol-pass"
        )
        personal = await kb_service.create(
            name="alice-personal", owner_id=alice.id, visibility="private"
        )
        tenant_kb = await kb_service.create(
            name="dept-kb",
            owner_id=alice.id,
            tenant_id="tenant-fin",
            origin="tenant",
        )
        authz = client.app.state.enterprise_authorization_service
        await authz.grant_kb_role(
            personal.id, alice.id, "kb_owner", granted_by=admin.id
        )
        await authz.grant_kb_role(
            tenant_kb.id, alice.id, "kb_owner", granted_by=admin.id
        )
        return carol, personal, tenant_kb

    carol, personal, tenant_kb = asyncio.run(_seed())

    person_id, _token = _enroll_person_token(client, user_service, admin, alice)
    alice_headers = {"Authorization": f"Bearer {_legacy_token(user_service, alice)}"}
    bob_headers = {"Authorization": f"Bearer {_legacy_token(user_service, bob)}"}

    # Non-owner cannot share someone else's KB.
    resp = client.post(
        f"/kbs/{personal.id}/person-shares",
        json={"target_account_id": bob.id},
        headers=bob_headers,
    )
    # The KB-write middleware rejects the outsider before the handler's
    # owner check (plain-string detail); either layer is a valid 403 here.
    assert resp.status_code == 403, resp.text

    # Target must be an active link of the SAME person (carol is not).
    resp = client.post(
        f"/kbs/{personal.id}/person-shares",
        json={"target_account_id": carol.id},
        headers=alice_headers,
    )
    assert resp.status_code == 404, resp.text
    assert resp.json()["detail"]["error_code"] == "account_not_linked"

    # Tenant-origin KBs cannot be person-shared.
    resp = client.post(
        f"/kbs/{tenant_kb.id}/person-shares",
        json={"target_account_id": bob.id},
        headers=alice_headers,
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"]["error_code"] == "person_share_requires_personal_kb"

    # Sharing to the owner account itself is a validation error.
    resp = client.post(
        f"/kbs/{personal.id}/person-shares",
        json={"target_account_id": alice.id},
        headers=alice_headers,
    )
    assert resp.status_code == 400, resp.text


# ---------------------------------------------------------------------------
# listing endpoints: person links / admin grants / admin persons
# ---------------------------------------------------------------------------


def test_person_links_listing_shows_pending(monkeypatch, tmp_path):
    """GET /auth/person/links surfaces every link with its status — most
    importantly pending links awaiting the person's confirmation — with the
    non-secret account summary attached."""

    (
        client,
        user_service,
        person_service,
        metadata_store,
        admin,
        alice,
        bob,
    ) = _build_client(monkeypatch, tmp_path)

    person_id, access_token = _enroll_person_token(client, user_service, admin, alice)
    headers = {"Authorization": f"Bearer {access_token}"}

    # No token -> 401.
    assert client.get("/auth/person/links").status_code == 401

    # Only the enrollment link exists initially.
    initial = client.get("/auth/person/links", headers=headers)
    assert initial.status_code == 200, initial.text
    assert initial.json()["total"] == 1
    assert initial.json()["links"][0]["link"]["status"] == "active"
    assert initial.json()["links"][0]["account"]["account_id"] == alice.id

    # Admin proposes a pending link to bob.
    propose = client.post(
        f"/admin/persons/{person_id}/accounts/{bob.id}",
        json={"reason": "second dept"},
        headers=_admin_headers(user_service, admin),
    )
    assert propose.status_code == 201, propose.text

    listing = client.get("/auth/person/links", headers=headers)
    assert listing.status_code == 200, listing.text
    body = listing.json()
    assert body["person_id"] == person_id
    assert body["total"] == 2
    by_account = {item["link"]["account_id"]: item for item in body["links"]}
    assert by_account[alice.id]["link"]["status"] == "active"
    assert by_account[bob.id]["link"]["status"] == "pending"
    assert by_account[bob.id]["link"]["confirmed_by_person_at"] is None
    assert by_account[bob.id]["account"]["username"] == "bob"

    # status filter narrows to the pending link only.
    pending = client.get("/auth/person/links?status=pending", headers=headers)
    assert pending.status_code == 200, pending.text
    assert pending.json()["total"] == 1
    assert pending.json()["links"][0]["link"]["account_id"] == bob.id

    # Unknown status value is a validation error.
    bad = client.get("/auth/person/links?status=bogus", headers=headers)
    assert bad.status_code == 400
    assert bad.json()["detail"]["error_code"] == "validation_error"

    # Confirm the pending link, re-login (confirm revokes sessions), and the
    # listing now shows both links active.
    confirm = client.post(
        f"/auth/person/links/{bob.id}:confirm",
        json={"person_password": "RightPass-1"},
        headers=headers,
    )
    assert confirm.status_code == 200, confirm.text
    relogin = client.post(
        "/auth/person/login",
        json={
            "person_id": person_id,
            "person_password": "RightPass-1",
            "account_id": alice.id,
        },
    ).json()
    headers = {"Authorization": f"Bearer {relogin['access_token']}"}
    after = client.get("/auth/person/links", headers=headers)
    assert after.status_code == 200, after.text
    statuses = {
        item["link"]["account_id"]: item["link"]["status"]
        for item in after.json()["links"]
    }
    assert statuses == {alice.id: "active", bob.id: "active"}
    assert client.get(
        "/auth/person/links?status=pending", headers=headers
    ).json()["total"] == 0


def test_admin_list_enrollment_grants(monkeypatch, tmp_path):
    """GET /admin/persons/enrollment-grants lists issued grants with status,
    consumption and expiry info while never echoing the token or its hash."""

    (
        client,
        user_service,
        person_service,
        metadata_store,
        admin,
        alice,
        bob,
    ) = _build_client(monkeypatch, tmp_path)

    # alice's grant gets consumed by enroll; bob's stays active.
    person_id, _token = _enroll_person_token(client, user_service, admin, alice)
    bob_grant = client.post(
        "/admin/persons/enrollment-grants",
        json={"account_id": bob.id},
        headers=_admin_headers(user_service, admin),
    ).json()

    listing = client.get(
        "/admin/persons/enrollment-grants",
        headers=_admin_headers(user_service, admin),
    )
    assert listing.status_code == 200, listing.text
    body = listing.json()
    assert body["total"] == 2
    for entry in body["grants"]:
        assert "token_hash" not in entry
        assert "grant_token" not in entry
        assert entry["expired"] is False

    by_account = {entry["account_id"]: entry for entry in body["grants"]}
    assert by_account[alice.id]["status"] == "consumed"
    assert by_account[alice.id]["consumed_by_person"] == person_id
    assert by_account[alice.id]["consumed_at"] is not None
    assert by_account[bob.id]["status"] == "active"
    assert by_account[bob.id]["grant_id"] == bob_grant["grant_id"]
    assert by_account[bob.id]["created_by"] == admin.id

    # account_id filter.
    only_bob = client.get(
        f"/admin/persons/enrollment-grants?account_id={bob.id}",
        headers=_admin_headers(user_service, admin),
    )
    assert only_bob.status_code == 200
    assert [g["account_id"] for g in only_bob.json()["grants"]] == [bob.id]

    # status filter follows the revoke transition.
    revoke = client.delete(
        f"/admin/persons/enrollment-grants/{bob_grant['grant_id']}",
        headers=_admin_headers(user_service, admin),
    )
    assert revoke.status_code == 200
    active_left = client.get(
        "/admin/persons/enrollment-grants?status=active",
        headers=_admin_headers(user_service, admin),
    )
    assert active_left.status_code == 200
    assert active_left.json()["total"] == 0
    revoked = client.get(
        "/admin/persons/enrollment-grants?status=revoked",
        headers=_admin_headers(user_service, admin),
    )
    assert [g["grant_id"] for g in revoked.json()["grants"]] == [
        bob_grant["grant_id"]
    ]

    # A stale-but-active grant reads expired=true (inserted directly with a
    # past expires_at; the API clamps ttl_seconds to >=60 so this state only
    # arises with the passage of time).
    from lightrag.api.metadata_store import EnterprisePersonEnrollmentGrantRecord
    import uuid as _uuid

    async def _seed_expired():
        carol = await user_service.create_user(
            username="carol", password="carol-pass"
        )
        now = "2000-01-01T00:00:00+00:00"
        await metadata_store.create_person_enrollment_grant_atomic(
            EnterprisePersonEnrollmentGrantRecord(
                id=f"pgrant_{_uuid.uuid4().hex[:12]}",
                account_id=carol.id,
                token_hash=f"sha256:{_uuid.uuid4().hex}",
                status="active",
                created_by=admin.id,
                consumed_by_person=None,
                expires_at="2000-01-02T00:00:00+00:00",
                created_at=now,
                updated_at=now,
                consumed_at=None,
            ),
            actor_user_id=admin.id,
        )
        return carol

    carol = asyncio.run(_seed_expired())
    stale = client.get(
        f"/admin/persons/enrollment-grants?account_id={carol.id}",
        headers=_admin_headers(user_service, admin),
    )
    assert stale.status_code == 200
    assert stale.json()["grants"][0]["status"] == "active"
    assert stale.json()["grants"][0]["expired"] is True

    # Unknown status value is a validation error.
    bad = client.get(
        "/admin/persons/enrollment-grants?status=bogus",
        headers=_admin_headers(user_service, admin),
    )
    assert bad.status_code == 400

    # Non-super-admin is rejected.
    alice_headers = {"Authorization": f"Bearer {_legacy_token(user_service, alice)}"}
    assert (
        client.get("/admin/persons/enrollment-grants", headers=alice_headers)
        .status_code
        == 403
    )


def test_admin_list_persons(monkeypatch, tmp_path):
    """GET /admin/persons lists natural persons with all their links."""

    (
        client,
        user_service,
        person_service,
        metadata_store,
        admin,
        alice,
        bob,
    ) = _build_client(monkeypatch, tmp_path)

    person_a, _ = _enroll_person_token(client, user_service, admin, alice)
    person_b, _ = _enroll_person_token(client, user_service, admin, bob)

    # A pending link on person_a shows up in its links array.
    carol = asyncio.run(
        user_service.create_user(username="carol", password="carol-pass")
    )
    propose = client.post(
        f"/admin/persons/{person_a}/accounts/{carol.id}",
        json={"reason": "pending link visible in listing"},
        headers=_admin_headers(user_service, admin),
    )
    assert propose.status_code == 201, propose.text

    listing = client.get(
        "/admin/persons", headers=_admin_headers(user_service, admin)
    )
    assert listing.status_code == 200, listing.text
    body = listing.json()
    assert body["total"] == 2
    by_id = {entry["id"]: entry for entry in body["persons"]}
    assert set(by_id) == {person_a, person_b}
    a_links = {
        link["account_id"]: link["status"] for link in by_id[person_a]["links"]
    }
    assert a_links == {alice.id: "active", carol.id: "pending"}
    b_links = {
        link["account_id"]: link["status"] for link in by_id[person_b]["links"]
    }
    assert b_links == {bob.id: "active"}

    # status filter after disabling person_b.
    disable = client.post(
        f"/admin/persons/{person_b}:disable",
        headers=_admin_headers(user_service, admin),
    )
    assert disable.status_code == 200
    disabled = client.get(
        "/admin/persons?status=disabled",
        headers=_admin_headers(user_service, admin),
    )
    assert [p["id"] for p in disabled.json()["persons"]] == [person_b]
    active = client.get(
        "/admin/persons?status=active",
        headers=_admin_headers(user_service, admin),
    )
    assert [p["id"] for p in active.json()["persons"]] == [person_a]

    # Unknown status value is a validation error.
    bad = client.get(
        "/admin/persons?status=bogus", headers=_admin_headers(user_service, admin)
    )
    assert bad.status_code == 400

    # Non-super-admin is rejected; person tokens (non-interactive-legacy) too.
    alice_headers = {"Authorization": f"Bearer {_legacy_token(user_service, alice)}"}
    assert client.get("/admin/persons", headers=alice_headers).status_code == 403
