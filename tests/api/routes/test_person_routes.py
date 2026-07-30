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
    app.include_router(create_person_routes(api_key=_API_KEY))

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
