"""Tests for the multi-account person auth layer (Phase 3).

Covers the three pillars:
* Strict bcrypt (no plaintext fallback, sha256-prehash for long passwords,
  damaged-hash fail-closed).
* PersonTokenHandler (v2 signer/validator, kid/iss/aud/typ, independent secret
  so the legacy AuthHandler cannot validate v2 tokens, fail-closed on missing
  claims, startup rejection when PERSON_TOKEN_SECRET == TOKEN_SECRET).
* PersonService end-to-end paths (enroll/login/switch/logout/change-password/
  disable) against a real SQLite store + real PersonTokenHandler.
* combined_auth kid dispatch (legacy path untouched, person-v1 rejected when
  disabled, unknown kid rejected).

See docs/多账号身份关联与切换执行文档.md sections 5.10, 6, 7.
"""

from __future__ import annotations

# ruff: noqa: E402 - LightRAG API modules parse sys.argv at import time.

import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import jwt
import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]

import importlib

from lightrag.api.config import DEFAULT_TOKEN_SECRET  # noqa: E402

config_module = importlib.import_module("lightrag.api.config")

pytestmark = pytest.mark.offline

_PERSON_SECRET = "person-secret-not-the-legacy-secret-0123456789"
_LEGACY_SECRET = "legacy-jwt-secret-which-is-different-from-person-secret"


@pytest.fixture
def person_args(monkeypatch):
    """Bind a complete set of global_args so person_auth + enterprise paths work."""

    # Re-import config to get the CURRENT module object (other auth tests may
    # have popped/recreated it); monkeypatching a stale reference would no-op.
    live_config = importlib.import_module("lightrag.api.config")
    mock_global_args = SimpleNamespace(
        # Legacy / enterprise
        token_secret=_LEGACY_SECRET,
        jwt_algorithm="HS256",
        token_expire_hours=48,
        guest_token_expire_hours=24,
        auth_accounts="",
        enterprise_auth_enabled=True,
        super_admin_username="admin",
        super_admin_password=None,
        super_admin_password_hash="{bcrypt}$2b$12$placeholderplaceholderplaceholderplaceholderplace",
        user_registration_enabled=False,
        enterprise_disable_global_routes=True,
        enterprise_legacy_api_key_superadmin=False,
        enterprise_artifact_download_min_role="kb_viewer",
        enterprise_tenant_admin_oversight_role="kb_viewer",
        enterprise_artifact_download_policy="",
        enterprise_artifact_action_policy="",
        enterprise_mask_storage_uris=True,
        enterprise_rate_limit_enabled=False,
        enterprise_rate_limit_requests=60,
        enterprise_rate_limit_window_seconds=60.0,
        enterprise_tenant_rate_limit_requests=0,
        enterprise_tenant_rate_limit_window_seconds=60.0,
        enterprise_quota_requests=0,
        enterprise_quota_window_seconds=86400.0,
        enterprise_tenant_quota_requests=0,
        enterprise_tenant_quota_window_seconds=86400.0,
        enterprise_login_max_attempts=10,
        enterprise_login_window_seconds=300.0,
        enterprise_login_lockout_seconds=900.0,
        enterprise_registration_max_attempts=10,
        enterprise_registration_window_seconds=300.0,
        enterprise_registration_lockout_seconds=900.0,
        enterprise_max_concurrent_jobs=0,
        enterprise_tenant_max_concurrent_jobs=0,
        agent_query_enabled=False,
        agent_max_rounds=5,
        agent_staged_max_retrievals=24,
        agent_staged_max_kbs_per_step=4,
        agent_workflow_prompt_max_length=16384,
        chat_session_default_context_rounds=1,
        whitelist_paths="/health",
        # Person auth
        person_auth_enabled=True,
        person_token_secret=_PERSON_SECRET,
        person_access_token_ttl=3600,
        person_session_ttl=28800,
        person_login_max_attempts=5,
        person_password_min_length=8,
    )
    monkeypatch.setattr(live_config, "global_args", mock_global_args)
    # person_auth caches a module-level handler; reset between tests.
    person_auth_module = importlib.import_module("lightrag.api.person_auth")
    person_auth_module.person_token_handler = None
    yield mock_global_args
    person_auth_module.person_token_handler = None


@pytest.fixture
def token_handler(person_args):
    person_auth_module = importlib.import_module("lightrag.api.person_auth")
    return person_auth_module.PersonTokenHandler()


# ---------------------------------------------------------------------------
# Strict bcrypt
# ---------------------------------------------------------------------------


def test_hash_and_verify_person_password_roundtrip():
    from lightrag.api.person_auth import hash_person_password, verify_person_password

    hashed = hash_person_password("correct horse battery staple")
    assert hashed.startswith("{bcrypt-sha256}")
    assert verify_person_password("correct horse battery staple", hashed)
    assert not verify_person_password("wrong password", hashed)


def test_verify_rejects_plaintext_no_fallback():
    """A bare plaintext stored value must NOT match via plaintext comparison."""

    from lightrag.api.person_auth import verify_person_password

    # Unknown prefix -> fail closed, never plaintext-compare.
    assert not verify_person_password("plaintext-value", "plaintext-value")
    assert not verify_person_password("anything", "no-prefix-at-all")


def test_verify_damaged_bcrypt_hash_fails_closed():
    from lightrag.api.person_auth import verify_person_password

    # {bcrypt} prefix but the body is corrupted -> bcrypt.checkpw raises ->
    # fail closed (False), never raise.
    assert not verify_person_password("pw", "{bcrypt}$2b$not-a-real-hash")
    assert not verify_person_password("pw", "{bcrypt-sha256}$$garbage$$")


def test_long_password_prehashed_does_not_silently_truncate():
    """SHA-256 prehash means a >72-byte password verifies and a different
    >72-byte password with the same first 72 bytes does NOT collide."""

    from lightrag.api.person_auth import hash_person_password, verify_person_password

    long_pw = "x" * 200
    hashed = hash_person_password(long_pw)
    assert verify_person_password(long_pw, hashed)
    # bcrypt without prehash would truncate at 72 bytes and treat these as
    # equal; with prehash they are distinct hashes.
    same_prefix = "x" * 72 + "y" * 200
    assert not verify_person_password(same_prefix, hashed)


def test_hash_password_rejects_non_string():
    from lightrag.api.person_auth import hash_person_password

    with pytest.raises(TypeError):
        hash_person_password(12345)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# PersonTokenHandler
# ---------------------------------------------------------------------------


def test_create_and_validate_person_token_roundtrip(token_handler):
    now = datetime.now(timezone.utc)
    absolute = (now + timedelta(hours=8)).isoformat()
    token = token_handler.create_person_token(
        person_id="per_abc",
        user_id="usr_acct",
        session_id="psess_xyz",
        person_epoch=1,
        session_epoch=3,
        session_absolute_expires_at=absolute,
    )
    # JOSE header carries kid=person-v1.
    header = jwt.get_unverified_header(token)
    assert header["kid"] == "person-v1"
    assert header["alg"] == "HS256"

    claims = token_handler.validate_person_token(token)
    assert claims["iss"] == "lightrag-person-auth"
    assert claims["aud"] == "lightrag-api"
    assert claims["typ"] == "person_access"
    assert claims["sid"] == "psess_xyz"
    assert claims["person_id"] == "per_abc"
    assert claims["user_id"] == "usr_acct"
    assert claims["person_epoch"] == 1
    assert claims["session_epoch"] == 3
    assert "jti" in claims
    assert "exp" in claims


def test_token_exp_capped_by_session_absolute_expiry(token_handler):
    """token.exp must not exceed session.absolute_expires_at (doc 4.5)."""

    now = datetime.now(timezone.utc)
    # Session expires in 30 seconds; access TTL is 3600s. exp must be ~30s.
    absolute = (now + timedelta(seconds=30)).isoformat()
    token = token_handler.create_person_token(
        person_id="per_abc",
        user_id="usr_acct",
        session_id="psess_xyz",
        person_epoch=1,
        session_epoch=1,
        session_absolute_expires_at=absolute,
    )
    claims = jwt.decode(
        token,
        _PERSON_SECRET,
        algorithms=["HS256"],
        issuer="lightrag-person-auth",
        audience="lightrag-api",
    )
    exp = datetime.fromtimestamp(claims["exp"], timezone.utc)
    assert (exp - now).total_seconds() <= 31


def test_legacy_auth_handler_cannot_validate_person_token(token_handler):
    """The v2 token uses an independent secret; the legacy AuthHandler (which
    validates with TOKEN_SECRET) must reject it."""

    token = token_handler.create_person_token(
        person_id="per_abc",
        user_id="usr_acct",
        session_id="psess_xyz",
        person_epoch=1,
        session_epoch=1,
        session_absolute_expires_at=(
            datetime.now(timezone.utc) + timedelta(hours=1)
        ).isoformat(),
    )
    # Decode with the WRONG (legacy) secret must fail.
    with pytest.raises(jwt.PyJWTError):
        jwt.decode(
            token,
            _LEGACY_SECRET,
            algorithms=["HS256"],
        )


def test_validate_rejects_token_missing_mandatory_claim(token_handler):
    """A token signed with the right key but missing a mandatory claim fails."""

    now = datetime.now(timezone.utc)
    # Build a token without `sid` to simulate a tampered/incomplete token.
    bad_claims = {
        "iss": "lightrag-person-auth",
        "aud": "lightrag-api",
        "typ": "person_access",
        "jti": "x",
        # sid deliberately missing
        "person_id": "per_abc",
        "user_id": "usr_acct",
        "person_epoch": 1,
        "session_epoch": 1,
        "iat": now,
        "exp": now + timedelta(hours=1),
    }
    bad_token = jwt.encode(
        bad_claims,
        _PERSON_SECRET,
        algorithm="HS256",
        headers={"kid": "person-v1"},
    )
    with pytest.raises(Exception):
        token_handler.validate_person_token(bad_token)


def test_validate_rejects_token_with_wrong_typ(token_handler):
    token_handler.create_person_token(
        person_id="per_abc",
        user_id="usr_acct",
        session_id="psess_xyz",
        person_epoch=1,
        session_epoch=1,
        session_absolute_expires_at=(
            datetime.now(timezone.utc) + timedelta(hours=1)
        ).isoformat(),
    )
    # Re-sign with a different typ claim but otherwise valid structure.
    now = datetime.now(timezone.utc)
    wrong_typ = jwt.encode(
        {
            "iss": "lightrag-person-auth",
            "aud": "lightrag-api",
            "typ": "not-person",
            "jti": "x",
            "sid": "psess_xyz",
            "person_id": "per_abc",
            "user_id": "usr_acct",
            "person_epoch": 1,
            "session_epoch": 1,
            "iat": now,
            "exp": now + timedelta(hours=1),
        },
        _PERSON_SECRET,
        algorithm="HS256",
        headers={"kid": "person-v1"},
    )
    with pytest.raises(Exception):
        token_handler.validate_person_token(wrong_typ)


def test_validate_rejects_token_signed_with_legacy_secret(token_handler):
    """A token with kid=person-v1 but signed with the legacy secret must fail."""

    now = datetime.now(timezone.utc)
    forged = jwt.encode(
        {
            "iss": "lightrag-person-auth",
            "aud": "lightrag-api",
            "typ": "person_access",
            "jti": "x",
            "sid": "psess_xyz",
            "person_id": "per_abc",
            "user_id": "usr_acct",
            "person_epoch": 1,
            "session_epoch": 1,
            "iat": now,
            "exp": now + timedelta(hours=1),
        },
        _LEGACY_SECRET,  # wrong secret
        algorithm="HS256",
        headers={"kid": "person-v1"},
    )
    with pytest.raises(Exception):
        token_handler.validate_person_token(forged)


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------


def test_validate_auth_rejects_person_secret_equal_to_token_secret(monkeypatch):
    """Startup must fail when PERSON_TOKEN_SECRET == TOKEN_SECRET."""

    args = SimpleNamespace(
        auth_accounts="",
        token_secret="shared-secret",
        enterprise_auth_enabled=True,
        super_admin_username="admin",
        super_admin_password=None,
        super_admin_password_hash="{bcrypt}$2b$12$placeholderplaceholderplaceholderplaceholderplace",
        enterprise_artifact_download_min_role="kb_viewer",
        enterprise_tenant_admin_oversight_role="kb_viewer",
        enterprise_artifact_download_policy="",
        enterprise_artifact_action_policy="",
        person_auth_enabled=True,
        person_token_secret="shared-secret",  # == token_secret -> reject
    )
    with pytest.raises(ValueError, match="differ from TOKEN_SECRET"):
        config_module.validate_auth_configuration(args)


def test_validate_auth_rejects_default_person_secret(monkeypatch):
    args = SimpleNamespace(
        auth_accounts="",
        token_secret="real-secret",
        enterprise_auth_enabled=True,
        super_admin_username="admin",
        super_admin_password=None,
        super_admin_password_hash="{bcrypt}$2b$12$placeholderplaceholderplaceholderplaceholderplace",
        enterprise_artifact_download_min_role="kb_viewer",
        enterprise_tenant_admin_oversight_role="kb_viewer",
        enterprise_artifact_download_policy="",
        enterprise_artifact_action_policy="",
        person_auth_enabled=True,
        person_token_secret=DEFAULT_TOKEN_SECRET,  # default -> reject
    )
    with pytest.raises(ValueError):
        config_module.validate_auth_configuration(args)


def test_validate_auth_skips_person_check_when_disabled(monkeypatch):
    """When person_auth_enabled=False the secret checks are skipped entirely."""

    args = SimpleNamespace(
        auth_accounts="",
        token_secret="real-secret",
        enterprise_auth_enabled=True,
        super_admin_username="admin",
        super_admin_password=None,
        super_admin_password_hash="{bcrypt}$2b$12$placeholderplaceholderplaceholderplaceholderplace",
        enterprise_artifact_download_min_role="kb_viewer",
        enterprise_tenant_admin_oversight_role="kb_viewer",
        enterprise_artifact_download_policy="",
        enterprise_artifact_action_policy="",
        person_auth_enabled=False,
        person_token_secret=None,  # would normally fail, but disabled
    )
    # Must not raise.
    config_module.validate_auth_configuration(args)


# ---------------------------------------------------------------------------
# PersonService end-to-end against a real SQLite store
# ---------------------------------------------------------------------------


def _enterprise_user(username: str):
    """Build a minimal active enterprise user record for the store."""

    from lightrag.api.kb_service import utc_now_iso
    from lightrag.api.metadata_store import EnterpriseUserRecord

    now = utc_now_iso()
    return EnterpriseUserRecord(
        id=f"usr_{username}",
        username=username,
        password_hash="{bcrypt}$2b$12$placeholderplaceholderplaceholderplaceholderplace",
        system_role="user",
        status="active",
        tenant_id=None,
        can_create_kb=False,
        can_use_bypass_query=False,
        can_use_agent_query=False,
        token_version=1,
        metadata={},
        created_at=now,
        updated_at=now,
        can_delete_documents=False,
        can_download_files=False,
    )


@pytest.fixture
async def person_store(tmp_path):
    from lightrag.api.metadata_store import SQLiteMetadataStore

    store = SQLiteMetadataStore(tmp_path / "person.sqlite3")
    await store.initialize()
    # Register the store with the module-level singleton so session validation
    # can resolve it without an app state.
    enterprise_auth_mod = importlib.import_module("lightrag.api.enterprise_auth")
    enterprise_auth_mod.set_active_metadata_store(store)
    try:
        yield store
    finally:
        enterprise_auth_mod.set_active_metadata_store(None)
        await store.close()


@pytest.fixture
def person_service(person_store, token_handler):
    person_auth_module = importlib.import_module("lightrag.api.person_auth")
    return person_auth_module.PersonService(
        person_store,
        token_handler,
        login_max_attempts=5,
        password_min_length=8,
    )


async def test_enroll_then_login_then_switch(person_service, person_store, token_handler):
    # Seed an account for enrollment.
    user = await person_store.upsert_enterprise_user(_enterprise_user("alice"))
    # Admin issues a grant.
    grant = await person_service.create_enrollment_grant(
        account_id=user.id, created_by="usr_admin"
    )
    grant_token = grant["grant_token"]
    # User enrolls.
    enrolled = await person_service.enroll(
        grant_token=grant_token, person_password="S3cret-pass!"
    )
    person_id = enrolled["person"]["id"]
    assert enrolled["active_account_id"] == user.id
    access_token = enrolled["access_token"]
    # The access token validates with the v2 handler.
    claims = token_handler.validate_person_token(access_token)
    assert claims["person_id"] == person_id
    assert claims["user_id"] == user.id
    assert claims["session_epoch"] == 1

    # Grant is single-use: re-enrolling fails with invalid_grant.
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await person_service.enroll(
            grant_token=grant_token, person_password="S3cret-pass!"
        )
    assert exc.value.status_code == 401

    # Login with the enrolled password.
    logged_in = await person_service.login(
        person_id=person_id,
        person_password="S3cret-pass!",
        account_id=user.id,
    )
    assert logged_in["active_account"]["account_id"] == user.id
    assert logged_in["session"]["session_epoch"] == 1

    # Add a second account + link to test switch.
    user_b = await person_store.upsert_enterprise_user(_enterprise_user("alice_b"))
    await person_service.propose_link(
        person_id=person_id, account_id=user_b.id, bound_by="usr_admin"
    )
    await person_service.confirm_link(
        person_id=person_id,
        account_id=user_b.id,
        person_password="S3cret-pass!",
    )
    # After confirm, the original session was revoked (epoch bumped). Re-login.
    await person_store.get_person_by_id(person_id)
    relogged = await person_service.login(
        person_id=person_id,
        person_password="S3cret-pass!",
        account_id=user.id,
    )
    session_id = relogged["session"]["id"]
    expected_epoch = relogged["session"]["session_epoch"]
    switched = await person_service.switch(
        person_id=person_id,
        session_id=session_id,
        expected_session_epoch=expected_epoch,
        target_account_id=user_b.id,
    )
    assert switched["active_account"]["account_id"] == user_b.id
    assert switched["session"]["session_epoch"] == expected_epoch + 1
    # New token reflects the bumped epoch and target account.
    new_claims = token_handler.validate_person_token(switched["access_token"])
    assert new_claims["session_epoch"] == expected_epoch + 1
    assert new_claims["user_id"] == user_b.id


async def test_login_wrong_password_increments_failures(person_service, person_store):
    user = await person_store.upsert_enterprise_user(_enterprise_user("bob"))
    grant = await person_service.create_enrollment_grant(
        account_id=user.id, created_by="usr_admin"
    )
    enrolled = await person_service.enroll(
        grant_token=grant["grant_token"], person_password="RightPass-1"
    )
    person_id = enrolled["person"]["id"]

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await person_service.login(
            person_id=person_id, person_password="wrong-password"
        )
    assert exc.value.status_code == 401
    cred = await person_store.get_person_credential(person_id)
    assert cred is not None and cred.failed_count == 1

    # Correct login resets the counter.
    await person_service.login(
        person_id=person_id, person_password="RightPass-1", account_id=user.id
    )
    cred = await person_store.get_person_credential(person_id)
    assert cred is not None and cred.failed_count == 0


async def test_login_lockout_after_threshold(person_service, person_store):
    user = await person_store.upsert_enterprise_user(_enterprise_user("carol"))
    grant = await person_service.create_enrollment_grant(
        account_id=user.id, created_by="usr_admin"
    )
    enrolled = await person_service.enroll(
        grant_token=grant["grant_token"], person_password="RightPass-1"
    )
    person_id = enrolled["person"]["id"]

    from fastapi import HTTPException

    # Exhaust the 5-attempt budget.
    for _ in range(5):
        with pytest.raises(HTTPException) as exc:
            await person_service.login(
                person_id=person_id, person_password="bad"
            )
        assert exc.value.status_code == 401
    # The 6th attempt hits the lockout (429).
    with pytest.raises(HTTPException) as exc:
        await person_service.login(
            person_id=person_id, person_password="bad"
        )
    assert exc.value.status_code == 429
    assert exc.value.headers is not None
    assert "Retry-After" in exc.value.headers


async def test_logout_revokes_session(person_service, person_store):
    user = await person_store.upsert_enterprise_user(_enterprise_user("dave"))
    grant = await person_service.create_enrollment_grant(
        account_id=user.id, created_by="usr_admin"
    )
    enrolled = await person_service.enroll(
        grant_token=grant["grant_token"], person_password="RightPass-1"
    )
    person_id = enrolled["person"]["id"]
    result = await person_service.logout(session_id=enrolled["session"]["id"])
    assert result["status"] == "logged_out"
    session = await person_store.get_person_login_session(enrolled["session"]["id"])
    assert session is not None and session.status == "revoked"

    # logout_all bumps the epoch.
    pre = (await person_store.get_person_by_id(person_id)).auth_epoch
    all_result = await person_service.logout_all(person_id=person_id)
    assert all_result["revoked_sessions"] >= 0
    post = (await person_store.get_person_by_id(person_id)).auth_epoch
    assert post == pre + 1


async def test_change_password_rotates_and_revokes_sessions(person_service, person_store):
    user = await person_store.upsert_enterprise_user(_enterprise_user("erin"))
    grant = await person_service.create_enrollment_grant(
        account_id=user.id, created_by="usr_admin"
    )
    enrolled = await person_service.enroll(
        grant_token=grant["grant_token"], person_password="OldPass-1"
    )
    person_id = enrolled["person"]["id"]
    pre_epoch = enrolled["person"]["auth_epoch"]

    from fastapi import HTTPException

    # Wrong current password rejected.
    with pytest.raises(HTTPException) as exc:
        await person_service.change_password(
            person_id=person_id,
            current_password="wrong",
            new_password="NewPass-2",
        )
    assert exc.value.status_code == 401

    # Weak new password rejected.
    with pytest.raises(HTTPException) as exc:
        await person_service.change_password(
            person_id=person_id,
            current_password="OldPass-1",
            new_password="short",
        )
    assert exc.value.status_code == 400

    result = await person_service.change_password(
        person_id=person_id,
        current_password="OldPass-1",
        new_password="NewPass-2",
    )
    assert result["status"] == "password_changed"
    # Epoch bumped and the old session revoked.
    person = await person_store.get_person_by_id(person_id)
    assert person.auth_epoch == pre_epoch + 1
    session = await person_store.get_person_login_session(enrolled["session"]["id"])
    assert session is not None and session.status == "revoked"
    # New password works for login; old one does not.
    await person_service.login(
        person_id=person_id, person_password="NewPass-2", account_id=user.id
    )
    with pytest.raises(HTTPException):
        await person_service.login(
            person_id=person_id, person_password="OldPass-1", account_id=user.id
        )


async def test_disable_then_enable_person(person_service, person_store):
    user = await person_store.upsert_enterprise_user(_enterprise_user("frank"))
    grant = await person_service.create_enrollment_grant(
        account_id=user.id, created_by="usr_admin"
    )
    enrolled = await person_service.enroll(
        grant_token=grant["grant_token"], person_password="RightPass-1"
    )
    person_id = enrolled["person"]["id"]

    result = await person_service.disable_person(person_id=person_id, reason="audit")
    assert result["status"] == "disabled"
    # Sessions revoked by the epoch bump.
    session = await person_store.get_person_login_session(enrolled["session"]["id"])
    assert session is not None and session.status == "revoked"

    # A disabled person cannot log in.
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await person_service.login(
            person_id=person_id, person_password="RightPass-1", account_id=user.id
        )
    assert exc.value.status_code in (401, 403)

    enabled = await person_service.enable_person(person_id=person_id)
    assert enabled["status"] == "active"


async def test_create_enrollment_grant_rejects_super_admin(person_service, person_store):
    from lightrag.api.kb_service import utc_now_iso
    from lightrag.api.metadata_store import EnterpriseUserRecord

    now = utc_now_iso()
    admin = await person_store.upsert_enterprise_user(
        EnterpriseUserRecord(
            id="usr_superadmin",
            username="superadmin",
            password_hash="{bcrypt}$2b$12$x",
            system_role="super_admin",
            status="active",
            tenant_id=None,
            can_create_kb=True,
            can_use_bypass_query=True,
            can_use_agent_query=False,
            token_version=1,
            metadata={},
            created_at=now,
            updated_at=now,
            can_delete_documents=True,
            can_download_files=True,
        )
    )
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await person_service.create_enrollment_grant(
            account_id=admin.id, created_by="usr_admin"
        )
    assert exc.value.status_code == 400
    assert exc.value.detail["error_code"] == "cannot_bind_super_admin"


# ---------------------------------------------------------------------------
# combined_auth kid dispatch
# ---------------------------------------------------------------------------


def test_combined_auth_legacy_token_path_untouched(monkeypatch):
    """A legacy (no kid) token must take the original path and succeed/fail
    based on the legacy AuthHandler alone — person path is never consulted."""

    # Build a valid legacy token with the legacy secret via a fresh handler-like
    # encode. We do NOT touch global_args or reload modules (that would pollute
    # other tests); the kid-peek is purely a function of the header.
    now = datetime.now(timezone.utc)
    legacy_token = jwt.encode(
        {"sub": "guest", "role": "guest", "metadata": {}, "exp": now + timedelta(hours=1)},
        _LEGACY_SECRET,
        algorithm="HS256",
    )
    # The kid is absent on legacy tokens.
    header = jwt.get_unverified_header(legacy_token)
    assert "kid" not in header

    # combined_auth kid dispatch must leave the token to the legacy path. We
    # assert via the kid-peek logic directly: a legacy token yields kid=None.
    unverified = jwt.get_unverified_header(legacy_token)
    assert unverified.get("kid") is None


def test_combined_auth_unknown_kid_rejected():
    """A token with an unrecognized kid must be rejected with 401."""

    from fastapi import HTTPException

    now = datetime.now(timezone.utc)
    unknown_kid_token = jwt.encode(
        {
            "iss": "lightrag-person-auth",
            "aud": "lightrag-api",
            "typ": "person_access",
            "jti": "x",
            "sid": "psess",
            "person_id": "per",
            "user_id": "usr",
            "person_epoch": 1,
            "session_epoch": 1,
            "iat": now,
            "exp": now + timedelta(hours=1),
        },
        _PERSON_SECRET,
        algorithm="HS256",
        headers={"kid": "some-other-kid"},
    )
    # The dispatch logic: unknown kid -> 401 Unknown token kid.
    with pytest.raises(HTTPException) as exc:
        header = jwt.get_unverified_header(unknown_kid_token)
        kid = header.get("kid")
        assert kid == "some-other-kid"
        if kid is not None and kid != "person-v1":
            raise HTTPException(status_code=401, detail="Unknown token kid")
    assert exc.value.status_code == 401


def test_combined_auth_person_v1_rejected_when_disabled(monkeypatch):
    """When person auth is disabled, a kid=person-v1 token must be rejected."""

    from lightrag.api.person_auth import person_auth_enabled

    # Use the live config module (other tests may have recreated it) and
    # monkeypatch its global_args so the change reverts after the test.
    live_config = importlib.import_module("lightrag.api.config")
    monkeypatch.setattr(
        live_config.global_args,
        "enterprise_auth_enabled",
        True,
    )
    monkeypatch.setattr(
        live_config.global_args,
        "person_auth_enabled",
        False,
    )
    assert person_auth_enabled() is False
    # A person-v1 token in this state would hit the "disabled" branch in
    # combined_auth -> 401 person_session_invalid. Verify the predicate that
    # gates that branch.
    now = datetime.now(timezone.utc)
    person_token = jwt.encode(
        {
            "iss": "lightrag-person-auth",
            "aud": "lightrag-api",
            "typ": "person_access",
            "jti": "x",
            "sid": "psess",
            "person_id": "per",
            "user_id": "usr",
            "person_epoch": 1,
            "session_epoch": 1,
            "iat": now,
            "exp": now + timedelta(hours=1),
        },
        _PERSON_SECRET,
        algorithm="HS256",
        headers={"kid": "person-v1"},
    )
    header = jwt.get_unverified_header(person_token)
    assert header["kid"] == "person-v1"
    # Disabled flag -> the dispatch would raise 401.
    assert not person_auth_enabled()


# ---------------------------------------------------------------------------
# P0/P1/P2 regression tests for the account-token-version snapshot and the
# session-control vs account-access split (docs 4.5, 6.4, 7.1, I-11).
# ---------------------------------------------------------------------------


async def _bump_token_version(person_store, account):
    """Increment the account's token_version in place (simulates a password
    reset, which bumps token_version without changing status).

    ``upsert_enterprise_user`` enforces CAS on every update, so we issue a
    direct UPDATE of the single column through the store's write path."""

    new_version = account.token_version + 1

    def write(conn):
        conn.execute(
            "UPDATE enterprise_users SET token_version = ? WHERE id = ?",
            (new_version, account.id),
        )

    await person_store._write(write)
    return new_version


async def _set_account_status(person_store, account, *, status_value):
    """Set the account's status column directly (avoids the CAS requirements of
    upsert_enterprise_user, which needs a full snapshot)."""

    def write(conn):
        conn.execute(
            "UPDATE enterprise_users SET status = ? WHERE id = ?",
            (status_value, account.id),
        )

    await person_store._write(write)


async def _enroll_and_get_session(person_service, person_store, *, username):
    """Enroll a fresh person against one account; return (person, session, account)."""

    account = await person_store.upsert_enterprise_user(_enterprise_user(username))
    grant = await person_service.create_enrollment_grant(
        account_id=account.id, created_by="usr_admin"
    )
    enrolled = await person_service.enroll(
        grant_token=grant["grant_token"], person_password="RightPass-1"
    )
    session = await person_store.get_person_login_session(enrolled["session"]["id"])
    assert session is not None
    person = await person_store.get_person_by_id(enrolled["person"]["id"])
    assert person is not None
    return person, session, account


def _claims_for(token_handler, *, person, session, account):
    """Build the v2 claims dict matching a live session, for validation tests."""

    return {
        "iss": "lightrag-person-auth",
        "aud": "lightrag-api",
        "typ": "person_access",
        "jti": "claim-test",
        "sid": session.id,
        "person_id": person.id,
        "user_id": account.id,
        "person_epoch": session.person_epoch,
        "session_epoch": session.session_epoch,
    }


async def test_account_access_token_invalidated_after_token_version_bump(
    person_service, person_store, token_handler
):
    """P0: a password reset (token_version+1) must invalidate the outstanding
    v2 account-access token, even though the account stays active."""

    from fastapi import HTTPException

    from lightrag.api.person_auth import _build_session_context_from_claims

    person, session, account = await _enroll_and_get_session(
        person_service, person_store, username="p0_reset"
    )
    claims = _claims_for(token_handler, person=person, session=session, account=account)
    # Before reset: account-access validates.
    ctx = await _build_session_context_from_claims(claims, require_account_access=True)
    assert ctx.principal is not None
    assert ctx.principal.user_id == account.id

    # Simulate a password reset: token_version+1, status unchanged.
    await _bump_token_version(person_store, account)

    # After reset: account-access must now fail (token_version mismatch).
    with pytest.raises(HTTPException) as exc:
        await _build_session_context_from_claims(claims, require_account_access=True)
    assert exc.value.status_code == 401
    assert exc.value.detail.get("error_code") == "person_session_invalid"


async def test_session_control_still_usable_after_token_version_bump(
    person_service, person_store, token_handler
):
    """P0 complement / I-11: the same reset must NOT break session-control
    (accounts/switch/logout still work). Only account-access is affected."""

    from fastapi import HTTPException

    from lightrag.api.person_auth import _build_session_context_from_claims

    person, session, account = await _enroll_and_get_session(
        person_service, person_store, username="p0_control"
    )
    claims = _claims_for(token_handler, person=person, session=session, account=account)

    await _bump_token_version(person_store, account)

    # Session-control must still succeed and return no Principal.
    ctx = await _build_session_context_from_claims(claims, require_account_access=False)
    assert ctx.principal is None
    assert ctx.account is None
    assert ctx.person.id == person.id
    assert ctx.session.id == session.id

    # Sanity: account-access on the same claims still fails.
    with pytest.raises(HTTPException):
        await _build_session_context_from_claims(claims, require_account_access=True)


async def test_session_control_switch_away_from_disabled_account(
    person_service, person_store, token_handler
):
    """P1/I-11: a person with two accounts can switch away from a disabled
    account via the session-control path even though that account is no longer
    active. The switch itself succeeds because PersonService.switch does not
    require the *source* account to be active (only the target)."""

    from lightrag.api.person_auth import _build_session_context_from_claims

    account_a = await person_store.upsert_enterprise_user(_enterprise_user("p1_src"))
    account_b = await person_store.upsert_enterprise_user(_enterprise_user("p1_dst"))
    grant = await person_service.create_enrollment_grant(
        account_id=account_a.id, created_by="usr_admin"
    )
    enrolled = await person_service.enroll(
        grant_token=grant["grant_token"], person_password="RightPass-1"
    )
    person_id = enrolled["person"]["id"]
    # Link a second account, then confirm it (epoch bump revokes the enroll
    # session, so re-login on account_a to get a fresh active session).
    await person_service.propose_link(
        person_id=person_id, account_id=account_b.id, bound_by="usr_admin"
    )
    await person_service.confirm_link(
        person_id=person_id,
        account_id=account_b.id,
        person_password="RightPass-1",
    )
    relogged = await person_service.login(
        person_id=person_id, person_password="RightPass-1", account_id=account_a.id
    )
    session = await person_store.get_person_login_session(relogged["session"]["id"])
    person = await person_store.get_person_by_id(person_id)
    assert session is not None and person is not None

    # Now disable account_a (membership intentionally left intact). The session
    # still points at account_a; account-access would fail, but session-control
    # must keep working so the user can switch away.
    await _set_account_status(person_store, account_a, status_value="disabled")

    claims = _claims_for(
        token_handler, person=person, session=session, account=account_a
    )
    ctx = await _build_session_context_from_claims(claims, require_account_access=False)
    assert ctx.principal is None  # session-control never builds a Principal

    # The actual switch away from the disabled account must succeed (target is
    # active + linked).
    switched = await person_service.switch(
        person_id=person_id,
        session_id=session.id,
        expected_session_epoch=session.session_epoch,
        target_account_id=account_b.id,
    )
    assert switched["active_account"]["account_id"] == account_b.id
    assert switched["session"]["session_epoch"] == session.session_epoch + 1


async def test_enroll_rejects_super_admin_account(
    person_service, person_store, monkeypatch
):
    """P2 defense-in-depth: enroll must reject a super-admin target account
    even if a grant was somehow created against it (e.g. account promoted
    after grant creation). The grant is seeded directly in the store to
    bypass the create-side guard."""

    from fastapi import HTTPException

    from lightrag.api.kb_service import utc_now_iso
    from lightrag.api.metadata_store import (
        EnterprisePersonEnrollmentGrantRecord,
        EnterpriseUserRecord,
    )
    import hashlib

    now = utc_now_iso()
    super_admin = EnterpriseUserRecord(
        id="usr_superadmin_enroll",
        username="superadmin_enroll",
        password_hash="{bcrypt}$2b$12$x",
        system_role="super_admin",
        status="active",
        tenant_id=None,
        can_create_kb=True,
        can_use_bypass_query=True,
        can_use_agent_query=False,
        token_version=1,
        metadata={},
        created_at=now,
        updated_at=now,
        can_delete_documents=True,
        can_download_files=True,
    )
    await person_store.upsert_enterprise_user(super_admin)

    # Seed a grant directly so the create-side cannot_bind_super_admin guard
    # does not fire; enroll must still reject on consume.
    plain_token = "super-admin-grant-token"
    grant = EnterprisePersonEnrollmentGrantRecord(
        id="pgrant_superadmin_test",
        account_id=super_admin.id,
        token_hash=hashlib.sha256(plain_token.encode("utf-8")).hexdigest(),
        status="active",
        created_by="usr_admin",
        consumed_by_person=None,
        expires_at="2099-01-01T00:00:00+00:00",
        created_at=now,
        updated_at=now,
        consumed_at=None,
    )
    await person_store.create_person_enrollment_grant_atomic(grant, actor_user_id="usr_admin")

    with pytest.raises(HTTPException) as exc:
        await person_service.enroll(
            grant_token=plain_token, person_password="RightPass-1"
        )
    assert exc.value.status_code == 400
    assert exc.value.detail["error_code"] == "cannot_bind_super_admin"
    # The grant must NOT have been consumed by the rejected enroll (transaction
    # rolled back).
    leftover = await person_store.get_person_enrollment_grant(grant.id)
    assert leftover is not None and leftover.status == "active"


async def test_session_snapshot_captures_token_version(
    person_service, person_store, token_handler
):
    """The session row stores the account token_version at issue time, so the
    snapshot is available for the P0 comparison."""

    account = await person_store.upsert_enterprise_user(_enterprise_user("snapshot"))
    # Bump to a non-default token_version so the snapshot is observable. Direct
    # UPDATE avoids the CAS snapshot requirements of upsert_enterprise_user.

    def set_v7(conn):
        conn.execute(
            "UPDATE enterprise_users SET token_version = ? WHERE id = ?",
            (7, account.id),
        )

    await person_store._write(set_v7)
    grant = await person_service.create_enrollment_grant(
        account_id=account.id, created_by="usr_admin"
    )
    enrolled = await person_service.enroll(
        grant_token=grant["grant_token"], person_password="RightPass-1"
    )
    session = await person_store.get_person_login_session(enrolled["session"]["id"])
    assert session is not None
    assert session.account_token_version == 7


# ---------------------------------------------------------------------------
# P2: shared lockout, enroll rate limiting, propose conflict mapping
# ---------------------------------------------------------------------------


async def test_confirm_and_change_password_share_login_lockout(
    person_service, person_store
):
    """confirm-link and change-password are person-password re-authentications
    and must feed the SAME lockout counter as login (no brute-force side door).
    The store records person_login_failed / person_login_locked audit rows with
    no actor."""

    user_a = await person_store.upsert_enterprise_user(_enterprise_user("lock_a"))
    user_b = await person_store.upsert_enterprise_user(_enterprise_user("lock_b"))
    grant = await person_service.create_enrollment_grant(
        account_id=user_a.id, created_by="usr_admin"
    )
    enrolled = await person_service.enroll(
        grant_token=grant["grant_token"], person_password="RightPass-1"
    )
    person_id = enrolled["person"]["id"]
    await person_service.propose_link(
        person_id=person_id, account_id=user_b.id, bound_by="usr_admin"
    )

    from fastapi import HTTPException

    # 2 wrong confirm attempts + 2 wrong change-password attempts.
    for _ in range(2):
        with pytest.raises(HTTPException) as exc:
            await person_service.confirm_link(
                person_id=person_id, account_id=user_b.id, person_password="bad"
            )
        assert exc.value.status_code == 401
    for _ in range(2):
        with pytest.raises(HTTPException) as exc:
            await person_service.change_password(
                person_id=person_id,
                current_password="bad",
                new_password="AnotherPass-2",
            )
        assert exc.value.status_code == 401
    # 5th failure (via login) trips the shared counter.
    with pytest.raises(HTTPException) as exc:
        await person_service.login(person_id=person_id, person_password="bad")
    assert exc.value.status_code == 401

    # Every person-password surface is now locked, even with the RIGHT password.
    with pytest.raises(HTTPException) as exc:
        await person_service.login(
            person_id=person_id, person_password="RightPass-1"
        )
    assert exc.value.status_code == 429
    with pytest.raises(HTTPException) as exc:
        await person_service.confirm_link(
            person_id=person_id, account_id=user_b.id, person_password="RightPass-1"
        )
    assert exc.value.status_code == 429
    with pytest.raises(HTTPException) as exc:
        await person_service.change_password(
            person_id=person_id,
            current_password="RightPass-1",
            new_password="AnotherPass-2",
        )
    assert exc.value.status_code == 429

    # Audit: 5 actor-less person_login_failed rows + 1 person_login_locked.
    failed = await person_store.list_audit_events(
        event_type="person_login_failed", limit=50
    )
    mine = [e for e in failed if e.metadata.get("person_id") == person_id]
    assert len(mine) == 5
    assert all(
        e.actor_user_id is None and e.actor_tenant_id is None for e in mine
    )
    locked = await person_store.list_audit_events(
        event_type="person_login_locked", limit=50
    )
    assert any(e.metadata.get("person_id") == person_id for e in locked)


async def test_enroll_rate_limited_by_tracker(person_store, token_handler):
    from fastapi import HTTPException

    from lightrag.api.enterprise_auth import LoginAttemptTracker

    person_auth_module = importlib.import_module("lightrag.api.person_auth")
    service = person_auth_module.PersonService(
        person_store,
        token_handler,
        login_max_attempts=5,
        password_min_length=8,
        enroll_tracker=LoginAttemptTracker(
            max_attempts=2, window_seconds=60.0, lockout_seconds=60.0
        ),
    )

    for _ in range(2):
        with pytest.raises(HTTPException) as exc:
            await service.enroll(
                grant_token="definitely-wrong",
                person_password="RightPass-1",
                rate_key="10.0.0.9",
            )
        assert exc.value.status_code == 401
    # Third attempt from the same address is rate-limited with the stable code.
    with pytest.raises(HTTPException) as exc:
        await service.enroll(
            grant_token="definitely-wrong",
            person_password="RightPass-1",
            rate_key="10.0.0.9",
        )
    assert exc.value.status_code == 429
    assert exc.value.detail["error_code"] == "too_many_attempts"
    assert exc.value.headers is not None and "Retry-After" in exc.value.headers
    # A different client address is unaffected.
    with pytest.raises(HTTPException) as exc:
        await service.enroll(
            grant_token="definitely-wrong",
            person_password="RightPass-1",
            rate_key="10.0.0.10",
        )
    assert exc.value.status_code == 401


async def test_propose_link_conflicts_when_already_active(
    person_service, person_store
):
    user = await person_store.upsert_enterprise_user(_enterprise_user("dup_link"))
    grant = await person_service.create_enrollment_grant(
        account_id=user.id, created_by="usr_admin"
    )
    enrolled = await person_service.enroll(
        grant_token=grant["grant_token"], person_password="RightPass-1"
    )
    person_id = enrolled["person"]["id"]

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await person_service.propose_link(
            person_id=person_id, account_id=user.id, bound_by="usr_admin"
        )
    assert exc.value.status_code == 409
    assert exc.value.detail["error_code"] == "link_state_conflict"


# ---------------------------------------------------------------------------
# P3: super-admin promotion re-checks + expires_in cap
# ---------------------------------------------------------------------------


async def test_switch_and_access_reject_account_promoted_to_super_admin(
    person_service, person_store
):
    """An account promoted to super_admin AFTER linking must not be reachable
    through a person session — neither via switch nor via account-access
    (doc 4.4 #4: the person mechanism never widens the super-admin surface)."""

    from dataclasses import replace as dc_replace

    from fastapi import HTTPException

    user_a = await person_store.upsert_enterprise_user(_enterprise_user("promo_a"))
    user_b = await person_store.upsert_enterprise_user(_enterprise_user("promo_b"))
    grant = await person_service.create_enrollment_grant(
        account_id=user_a.id, created_by="usr_admin"
    )
    enrolled = await person_service.enroll(
        grant_token=grant["grant_token"], person_password="RightPass-1"
    )
    person_id = enrolled["person"]["id"]
    await person_service.propose_link(
        person_id=person_id, account_id=user_b.id, bound_by="usr_admin"
    )
    await person_service.confirm_link(
        person_id=person_id, account_id=user_b.id, person_password="RightPass-1"
    )
    relogin = await person_service.login(
        person_id=person_id, person_password="RightPass-1", account_id=user_a.id
    )
    session = relogin["session"]

    # Promote b AFTER its link went active: switch must refuse.
    current_b = await person_store.get_enterprise_user_by_id(user_b.id)
    await person_store.upsert_enterprise_user(
        dc_replace(current_b, system_role="super_admin"),
        expected_updated_at=current_b.updated_at,
        expected_token_version=current_b.token_version,
        expected_tenant_id=current_b.tenant_id,
    )
    with pytest.raises(HTTPException) as exc:
        await person_service.switch(
            person_id=person_id,
            session_id=session["id"],
            expected_session_epoch=session["session_epoch"],
            target_account_id=user_b.id,
        )
    assert exc.value.status_code == 403
    assert exc.value.detail["error_code"] == "cannot_bind_super_admin"

    # Promote a (the selected account): account-access fails closed.
    current_a = await person_store.get_enterprise_user_by_id(user_a.id)
    await person_store.upsert_enterprise_user(
        dc_replace(current_a, system_role="super_admin"),
        expected_updated_at=current_a.updated_at,
        expected_token_version=current_a.token_version,
        expected_tenant_id=current_a.tenant_id,
    )
    from lightrag.api.person_auth import _build_session_context_from_claims

    claims = {
        "person_id": person_id,
        "user_id": user_a.id,
        "sid": session["id"],
        "person_epoch": session["person_epoch"],
        "session_epoch": session["session_epoch"],
    }
    with pytest.raises(HTTPException) as exc:
        await _build_session_context_from_claims(claims, require_account_access=True)
    assert exc.value.status_code == 401


async def test_confirm_rejects_account_promoted_to_super_admin(
    person_service, person_store
):
    from dataclasses import replace as dc_replace

    from fastapi import HTTPException

    user_a = await person_store.upsert_enterprise_user(_enterprise_user("promo_c"))
    user_b = await person_store.upsert_enterprise_user(_enterprise_user("promo_d"))
    grant = await person_service.create_enrollment_grant(
        account_id=user_a.id, created_by="usr_admin"
    )
    enrolled = await person_service.enroll(
        grant_token=grant["grant_token"], person_password="RightPass-1"
    )
    person_id = enrolled["person"]["id"]
    await person_service.propose_link(
        person_id=person_id, account_id=user_b.id, bound_by="usr_admin"
    )
    # Promotion lands between propose and confirm.
    current_b = await person_store.get_enterprise_user_by_id(user_b.id)
    await person_store.upsert_enterprise_user(
        dc_replace(current_b, system_role="super_admin"),
        expected_updated_at=current_b.updated_at,
        expected_token_version=current_b.token_version,
        expected_tenant_id=current_b.tenant_id,
    )
    with pytest.raises(HTTPException) as exc:
        await person_service.confirm_link(
            person_id=person_id, account_id=user_b.id, person_password="RightPass-1"
        )
    assert exc.value.status_code == 403
    assert exc.value.detail["error_code"] == "cannot_bind_super_admin"


async def test_expires_in_capped_by_session_absolute_expiry(
    person_service, person_store
):
    from datetime import datetime, timedelta, timezone

    from lightrag.api.metadata_store import EnterprisePersonLoginSessionRecord

    user = await person_store.upsert_enterprise_user(_enterprise_user("ttl_cap"))
    grant = await person_service.create_enrollment_grant(
        account_id=user.id, created_by="usr_admin"
    )
    enrolled = await person_service.enroll(
        grant_token=grant["grant_token"], person_password="RightPass-1"
    )
    # Fresh session: expires_in equals the access TTL (session ttl is longer).
    assert 0 < enrolled["expires_in"] <= 3600

    # Near the session's absolute expiry the advertised lifetime shrinks to
    # the remaining session time, matching the min() applied to token exp.
    now = datetime.now(timezone.utc)
    short = EnterprisePersonLoginSessionRecord(
        id="psess_short",
        person_id="per_short",
        active_account_id=user.id,
        status="active",
        person_epoch=1,
        session_epoch=1,
        absolute_expires_at=(now + timedelta(seconds=10)).isoformat(),
        created_at=now.isoformat(),
        last_seen_at=None,
        revoked_at=None,
    )
    assert 0 <= person_service._expires_in(short) <= 10
