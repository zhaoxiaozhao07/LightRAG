"""Multi-account person identity authentication (Phase 3).

This module implements the v2 person access JWT signer/validator, strict bcrypt
credential helpers, the :class:`PersonService` business layer, and the two
FastAPI dependencies documented in
``docs/多账号身份关联与切换执行文档.md`` section 5.10.

Design constraints enforced here and verified by tests:

* Person credentials use **strict bcrypt only**. The legacy
  :mod:`lightrag.api.passwords` module has a plaintext fallback and MUST NOT be
  imported by this module for verification.
* v2 person tokens are signed with an independent ``LIGHTRAG_PERSON_TOKEN_SECRET``
  (distinct from ``TOKEN_SECRET``) so a rollback to an older binary (which lacks
  the person key) cannot verify them. The JOSE header carries ``kid="person-v1"``
  for deterministic dispatch in ``combined_auth``.
* ``person_auth_enabled=false`` (the default) disables every person path; the
  legacy ``combined_auth`` (``kid`` absent) path is untouched.
* Identity is proven by signed ``sid`` + ``session_epoch``; the JWT ``jti`` is
  for uniqueness/audit only and never participates in session verification.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import bcrypt
import jwt
from fastapi import HTTPException, Request, Security, status
from fastapi.security import OAuth2PasswordBearer

from lightrag.api.enterprise_auth import (
    EnterpriseMetadataStore,
    PERSON_JWT_AUTH_METHOD,
    Principal,
    SYSTEM_ROLE_SUPER_ADMIN,
    USER_STATUS_ACTIVE,
    enterprise_auth_enabled,
    principal_from_user,
    tenant_role_is_admin,
)
from lightrag.api.kb_service import utc_now_iso
from lightrag.api.metadata_store import (
    EnterprisePersonAccountLinkRecord,
    EnterprisePersonCredentialRecord,
    EnterprisePersonEnrollmentGrantRecord,
    EnterprisePersonKBShareRecord,
    EnterprisePersonLoginSessionRecord,
    EnterprisePersonRecord,
    EnterpriseUserRecord,
    KBLifecycleConflictError,
    MetadataConflictError,
    MetadataRecordNotFoundError,
)

logger = logging.getLogger("lightrag")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# bcrypt has a 72-byte input limit; longer inputs are silently truncated which
# weakens the hash. To make that impossible we hash the password with SHA-256
# first (fixed 32 bytes) and bcrypt the digest. The stored value carries the
# ``{bcrypt-sha256}`` prefix so verification routes correctly and the decision
# is auditable. Plain ``{bcrypt}`` (no prehash) is also accepted on verify for
# forward compatibility, but the hasher always emits the prehashed form.
PERSON_BCRYPT_SHA256_PREFIX = "{bcrypt-sha256}"
PERSON_BCRYPT_PREFIX = "{bcrypt}"
PERSON_TOKEN_KID = "person-v1"
PERSON_TOKEN_ISS = "lightrag-person-auth"
PERSON_TOKEN_AUD = "lightrag-api"
PERSON_TOKEN_TYP = "person_access"
PERSON_TOKEN_ALG = "HS256"


# ---------------------------------------------------------------------------
# Strict bcrypt helpers (NO plaintext fallback, unlike passwords.py)
# ---------------------------------------------------------------------------


def hash_person_password(password: str) -> str:
    """Hash a person password with SHA-256 then bcrypt.

    Returns a ``{bcrypt-sha256}$2b$...`` value. The SHA-256 prehash is applied
    so bcrypt's 72-byte input limit cannot silently truncate a long password.
    """

    if not isinstance(password, str):
        raise TypeError("person password must be a string")
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(digest, salt).decode("utf-8")
    return f"{PERSON_BCRYPT_SHA256_PREFIX}{hashed}"


def verify_person_password(plain_password: str, stored_hash: str) -> bool:
    """Verify a person password against a stored hash.

    Strict bcrypt only. A damaged/unknown hash fails closed (returns False).
    This function MUST NOT fall back to plaintext comparison; the legacy
    :mod:`lightrag.api.passwords` module is deliberately not used here.
    """

    if not isinstance(plain_password, str) or not isinstance(stored_hash, str):
        return False
    if stored_hash.startswith(PERSON_BCRYPT_SHA256_PREFIX):
        bcrypt_part = stored_hash[len(PERSON_BCRYPT_SHA256_PREFIX) :]
        digest = hashlib.sha256(plain_password.encode("utf-8")).digest()
    elif stored_hash.startswith(PERSON_BCRYPT_PREFIX):
        bcrypt_part = stored_hash[len(PERSON_BCRYPT_PREFIX) :]
        digest = plain_password.encode("utf-8")
    else:
        # Unknown scheme -> fail closed. Never compare as plaintext.
        return False
    if not bcrypt_part:
        return False
    try:
        return bcrypt.checkpw(digest, bcrypt_part.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Feature flag helpers
# ---------------------------------------------------------------------------


def person_auth_enabled() -> bool:
    """Return True only when person auth is explicitly enabled.

    Person auth requires enterprise auth (it builds on enterprise accounts and
    the metadata store); when either flag is off the person path is disabled.
    """

    return bool(
        enterprise_auth_enabled()
        and getattr(_person_global_args(), "person_auth_enabled", False)
    )


def _person_global_args() -> Any:
    """Read ``config.global_args`` directly (no enterprise fallback).

    The enterprise ``_global_args()`` helper returns a fallback SimpleNamespace
    when the config proxy is not initialized; that fallback omits person auth
    fields and would mis-report the flag. Person auth reads the real config
    object so tests that monkeypatch ``config.global_args`` are honored.
    """

    from importlib import import_module

    return import_module("lightrag.api.config").global_args


# ---------------------------------------------------------------------------
# PersonTokenHandler (v2 signer/validator, independent key)
# ---------------------------------------------------------------------------


class PersonTokenHandler:
    """Sign and validate v2 person access JWTs with an independent secret."""

    def __init__(self) -> None:
        args = _person_global_args()
        secret = getattr(args, "person_token_secret", None)
        if not secret:
            raise ValueError(
                "LIGHTRAG_PERSON_TOKEN_SECRET is required to construct a "
                "PersonTokenHandler."
            )
        self._secret = secret
        self._algorithm = PERSON_TOKEN_ALG
        self._access_ttl = int(getattr(args, "person_access_token_ttl", 3600))
        self._session_ttl = int(getattr(args, "person_session_ttl", 28800))
        self._issuer = PERSON_TOKEN_ISS
        self._audience = PERSON_TOKEN_AUD

    @property
    def access_ttl_seconds(self) -> int:
        return self._access_ttl

    @property
    def session_ttl_seconds(self) -> int:
        return self._session_ttl

    def session_absolute_expiry(self, *, now: datetime | None = None) -> str:
        """Return the ISO-8601 absolute session expiry from ``now``."""

        base = now or datetime.now(timezone.utc)
        return (base + timedelta(seconds=self._session_ttl)).isoformat()

    def create_person_token(
        self,
        *,
        person_id: str,
        user_id: str,
        session_id: str,
        person_epoch: int,
        session_epoch: int,
        session_absolute_expires_at: str,
        now: datetime | None = None,
    ) -> str:
        base = now or datetime.now(timezone.utc)
        absolute_expiry = datetime.fromisoformat(session_absolute_expires_at)
        access_expiry = min(
            base + timedelta(seconds=self._access_ttl), absolute_expiry
        )
        # exp as a POSIX timestamp (PyJWT handles datetime or int).
        claims = {
            "iss": self._issuer,
            "aud": self._audience,
            "typ": PERSON_TOKEN_TYP,
            "jti": uuid4().hex,
            "sid": session_id,
            "person_id": person_id,
            "user_id": user_id,
            "person_epoch": int(person_epoch),
            "session_epoch": int(session_epoch),
            "iat": base,
            "exp": access_expiry,
        }
        return jwt.encode(
            claims,
            self._secret,
            algorithm=self._algorithm,
            headers={"kid": PERSON_TOKEN_KID},
        )

    def validate_person_token(self, token: str) -> dict[str, Any]:
        """Validate a v2 person token. Raises HTTPException(401) on any failure.

        Returns the full claims dict on success. Validation is strict: only
        HS256 with the independent person secret, and the mandatory
        iss/aud/typ claims are checked. There is intentionally NO fallback to
        the legacy validator.
        """

        try:
            unverified_header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid person token",
            ) from exc
        if unverified_header.get("kid") != PERSON_TOKEN_KID:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid person token",
            )
        try:
            claims = jwt.decode(
                token,
                self._secret,
                algorithms=[self._algorithm],
                issuer=self._issuer,
                audience=self._audience,
                options={"require": ["exp", "iat", "iss", "aud"]},
            )
        except jwt.MissingRequiredClaimError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid person token",
            ) from exc
        except jwt.PyJWTError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid person token",
            ) from exc
        if claims.get("typ") != PERSON_TOKEN_TYP:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid person token",
            )
        for required in ("sid", "person_id", "user_id", "person_epoch", "session_epoch"):
            if required not in claims:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid person token",
                )
        return claims


# Module-level lazy singleton, mirroring the ``auth_handler`` pattern. It is
# only constructed when person auth is enabled (the secret must be set).
person_token_handler: PersonTokenHandler | None = None


def get_person_token_handler() -> PersonTokenHandler:
    """Return the module-level :class:`PersonTokenHandler`, constructing once.

    Raises if person auth is disabled or the secret is missing — callers
    should gate on :func:`person_auth_enabled` first.
    """

    global person_token_handler
    if person_token_handler is None:
        person_token_handler = PersonTokenHandler()
    return person_token_handler


# ---------------------------------------------------------------------------
# PersonSessionContext
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PersonSessionContext:
    """Result of validating a person session for the current request.

    ``principal``/``account`` are populated only on the account-access path
    (combined_auth for business APIs). On the session-control path
    (accounts/switch/logout/change-password) they are ``None`` because those
    endpoints do not need an account Principal and must keep working even when
    the current account was disabled or its membership stripped (doc I-11).
    Person/session fields are always populated.
    """

    person: EnterprisePersonRecord
    session: EnterprisePersonLoginSessionRecord
    person_epoch: int
    session_epoch: int
    principal: Principal | None = None
    account: EnterpriseUserRecord | None = None


# ---------------------------------------------------------------------------
# PersonService (business layer over the atomic store methods)
# ---------------------------------------------------------------------------


class PersonService:
    """Business-layer facade over the Phase-2 aggregate atomic store methods.

    The service enforces person password policy, login failure/lockout, account
    eligibility (no super-admin binding) and v2 token issuance. All state
    mutations delegate to the store's atomic methods so state+CAS+audit stay in
    one transaction. This service never calls :mod:`lightrag.api.passwords`
    (plaintext fallback) — only :func:`hash_person_password` /
    :func:`verify_person_password`.
    """

    def __init__(
        self,
        metadata_store: EnterpriseMetadataStore,
        token_handler: PersonTokenHandler,
        *,
        login_max_attempts: int = 5,
        password_min_length: int = 8,
        lockout_seconds: float = 900.0,
        enroll_tracker: Any | None = None,
    ) -> None:
        self._store = metadata_store
        self._tokens = token_handler
        self._login_max_attempts = max(1, int(login_max_attempts))
        self._password_min_length = max(1, int(password_min_length))
        self._lockout_seconds = float(lockout_seconds)
        # Optional LoginAttemptTracker keyed by client address; guards the
        # public enroll endpoint against grant-guessing hammering (doc 5.3).
        self._enroll_tracker = enroll_tracker

    # -- password policy ---------------------------------------------------

    def _validate_password(self, password: str) -> None:
        if not isinstance(password, str) or not password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error_code": "person_password_weak", "message": "Password is required"},
            )
        if password.strip() != password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "person_password_weak",
                    "message": "Password must not have leading/trailing whitespace",
                },
            )
        if len(password) < self._password_min_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "person_password_weak",
                    "message": f"Password must be at least {self._password_min_length} characters",
                },
            )
        # UTF-8 byte ceiling; bcrypt prehash handles length, this bounds input.
        if len(password.encode("utf-8")) > 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error_code": "person_password_weak", "message": "Password is too long"},
            )

    def _issue_token(self, *, session: EnterprisePersonLoginSessionRecord) -> str:
        return self._tokens.create_person_token(
            person_id=session.person_id,
            user_id=session.active_account_id or "",
            session_id=session.id,
            person_epoch=session.person_epoch,
            session_epoch=session.session_epoch,
            session_absolute_expires_at=session.absolute_expires_at,
        )

    def _expires_in(self, session: EnterprisePersonLoginSessionRecord) -> int:
        """Actual token lifetime: access TTL capped by the session's absolute
        expiry — mirrors the min() applied to ``exp`` at signing time."""

        remaining = int(
            (
                datetime.fromisoformat(session.absolute_expires_at)
                - datetime.now(timezone.utc)
            ).total_seconds()
        )
        return max(0, min(self._tokens.access_ttl_seconds, remaining))

    # -- enroll ------------------------------------------------------------

    def _check_enroll_rate(self, rate_key: str | None) -> None:
        if self._enroll_tracker is None or not rate_key:
            return
        try:
            self._enroll_tracker.check(rate_key)
        except HTTPException as exc:
            # Re-raise with the stable person error contract, keeping the
            # tracker's Retry-After header.
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error_code": "too_many_attempts",
                    "message": "Too many attempts",
                },
                headers=exc.headers,
            ) from exc

    async def enroll(
        self,
        *,
        grant_token: str,
        person_password: str,
        rate_key: str | None = None,
    ) -> dict[str, Any]:
        self._check_enroll_rate(rate_key)
        self._validate_password(person_password)
        token_hash = hashlib.sha256(grant_token.encode("utf-8")).hexdigest()
        grant = await self._store.get_person_enrollment_grant_by_token_hash(token_hash)
        if grant is None:
            if self._enroll_tracker is not None and rate_key:
                self._enroll_tracker.record_failure(rate_key)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error_code": "invalid_grant", "message": "Invalid enrollment grant"},
            )
        # Verify the target account is eligible and active before consuming.
        account = await self._store.get_enterprise_user_by_id(grant.account_id)
        if account is None or account.status != USER_STATUS_ACTIVE:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error_code": "account_not_active", "message": "Account is not active"},
            )
        # P2 defense-in-depth: re-check super_admin at consume time. The grant
        # creation side already rejects super-admin targets, but a grant could
        # be created against a non-super-admin account that is later promoted,
        # or a future grant path could bypass the check. Enforce here too.
        if account.system_role == SYSTEM_ROLE_SUPER_ADMIN:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "cannot_bind_super_admin",
                    "message": "Cannot bind a super admin account",
                },
            )
        now = utc_now_iso()
        person = EnterprisePersonRecord(
            id=f"per_{secrets.token_hex(12)}",
            status="active",
            auth_epoch=1,
            metadata={"created_via": "enrollment"},
            created_at=now,
            updated_at=now,
            # 工号 pinned on the grant at issue time; the persons unique
            # index is the final arbiter at consume time.
            person_number=grant.person_number,
        )
        credential = EnterprisePersonCredentialRecord(
            id=f"pcred_{secrets.token_hex(12)}",
            person_id=person.id,
            credential_type="password",
            algorithm="bcrypt",
            password_hash=hash_person_password(person_password),
            status="active",
            failed_count=0,
            locked_until=None,
            last_used_at=None,
            created_at=now,
            updated_at=now,
        )
        link = EnterprisePersonAccountLinkRecord(
            id=f"plink_{secrets.token_hex(12)}",
            person_id=person.id,
            account_id=grant.account_id,
            status="active",
            bound_by=grant.created_by,
            bound_at=now,
            confirmed_by_person_at=now,
            revoked_by=None,
            revoked_at=None,
            reason="enrollment",
            created_at=now,
            updated_at=now,
        )
        session = EnterprisePersonLoginSessionRecord(
            id=f"psess_{secrets.token_hex(12)}",
            person_id=person.id,
            active_account_id=grant.account_id,
            status="active",
            person_epoch=person.auth_epoch,
            session_epoch=1,
            absolute_expires_at=self._tokens.session_absolute_expiry(),
            created_at=now,
            last_seen_at=None,
            revoked_at=None,
        )
        try:
            saved_person, _saved_cred, saved_link, saved_session = (
                await self._store.enroll_person_atomic(
                    grant_token_hash=token_hash,
                    person=person,
                    credential=credential,
                    link=link,
                    session=session,
                    actor_user_id=account.id,
                )
            )
        except MetadataConflictError as exc:
            # Map the distinct conflict shapes to stable codes.
            if exc.entity_type == "person_account_link_active":
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail={
                        "error_code": "account_already_linked",
                        "message": "Account is already linked to a person",
                    },
                ) from exc
            if exc.entity_type == "person_number_unique":
                # The grant itself is valid (and stays consumable after the
                # rollback), so this is not an invalid-grant rate signal.
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail={
                        "error_code": "person_number_conflict",
                        "message": "person_number is already bound to another person",
                    },
                ) from exc
            if self._enroll_tracker is not None and rate_key:
                self._enroll_tracker.record_failure(rate_key)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error_code": "invalid_grant", "message": "Invalid enrollment grant"},
            ) from exc
        if self._enroll_tracker is not None and rate_key:
            self._enroll_tracker.record_success(rate_key)
        token = self._issue_token(session=saved_session)
        return {
            "person": saved_person.to_dict(),
            "active_account_id": saved_link.account_id,
            "session": saved_session.to_dict(),
            "access_token": token,
            "token_type": "bearer",
            "expires_in": self._expires_in(saved_session),
        }

    # -- login -------------------------------------------------------------

    _PERSON_NUMBER_MAX_LENGTH = 64

    def _normalize_person_number(self, person_number: str | None) -> str | None:
        """Strip the 工号; empty/whitespace-only collapses to ``None``."""

        if person_number is None:
            return None
        value = person_number.strip()
        if not value:
            return None
        if len(value) > self._PERSON_NUMBER_MAX_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "validation_error",
                    "message": "person_number is too long",
                },
            )
        return value

    def _ensure_credential_not_locked(
        self, credential: EnterprisePersonCredentialRecord
    ) -> None:
        """Raise 429 (with Retry-After) while the credential is locked out."""

        if credential.locked_until is None or credential.locked_until <= utc_now_iso():
            return
        retry_after = max(
            1,
            int(
                (
                    datetime.fromisoformat(credential.locked_until)
                    - datetime.now(timezone.utc)
                ).total_seconds()
            ),
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"error_code": "too_many_attempts", "message": "Too many attempts"},
            headers={"Retry-After": str(retry_after)},
        )

    async def _verify_password_or_record_failure(
        self,
        credential: EnterprisePersonCredentialRecord,
        password: str,
        *,
        error_code: str,
        message: str,
    ) -> None:
        """Shared lockout gate for every person-password re-authentication.

        Applies to login, confirm-link and change-password alike so the
        natural-person password cannot be brute-forced through a side door.
        The failure counter increments atomically in SQL (store method) and
        the store writes the ``person_login_failed``/``person_login_locked``
        audit rows in the same transaction.
        """

        self._ensure_credential_not_locked(credential)
        if verify_person_password(password, credential.password_hash):
            return
        try:
            await self._store.record_person_credential_failure_atomic(
                credential.id,
                max_attempts=self._login_max_attempts,
                lockout_seconds=self._lockout_seconds,
            )
        except MetadataRecordNotFoundError:
            # Credential vanished mid-flight (person deleted); still fail auth.
            pass
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": error_code, "message": message},
        )

    async def _reset_failed_login(
        self, credential: EnterprisePersonCredentialRecord
    ) -> None:
        await self._store.reset_person_credential_failures_atomic(credential.id)

    async def _pick_default_link(
        self,
        person_id: str,
        active_links: list[EnterprisePersonAccountLinkRecord],
    ) -> EnterprisePersonAccountLinkRecord:
        """Choose the login landing account when ``account_id`` is omitted.

        Preference order: the account of the most recent login session (its
        ``active_account_id`` reflects the last switch), then the remaining
        active links in bind order. Ineligible candidates (account missing,
        inactive, or promoted to super_admin) are skipped; if none is
        eligible the first candidate is returned so the shared eligibility
        checks in ``login`` produce the proper 403.
        """

        by_account = {link.account_id: link for link in active_links}
        ordered: list[EnterprisePersonAccountLinkRecord] = []
        sessions = await self._store.list_person_login_sessions(person_id)
        for session in sessions:  # newest first
            link = by_account.get(session.active_account_id or "")
            if link is not None and link not in ordered:
                ordered.append(link)
        for link in active_links:  # bound_at ascending fallback
            if link not in ordered:
                ordered.append(link)
        for link in ordered:
            account = await self._store.get_enterprise_user_by_id(link.account_id)
            if (
                account is not None
                and account.status == USER_STATUS_ACTIVE
                and account.system_role != SYSTEM_ROLE_SUPER_ADMIN
            ):
                return link
        return ordered[0]

    async def login(
        self,
        *,
        person_id: str | None = None,
        person_password: str,
        account_id: str | None = None,
        person_number: str | None = None,
    ) -> dict[str, Any]:
        # Exactly one person identifier: the opaque person_id or the bound
        # 工号. Rejecting both-or-neither happens before any lookup so the
        # 400 cannot become an existence oracle.
        person_number = self._normalize_person_number(person_number)
        if bool(person_id) == bool(person_number):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "validation_error",
                    "message": "Provide exactly one of person_id or person_number",
                },
            )
        if person_id:
            person = await self._store.get_person_by_id(person_id)
        else:
            person = await self._store.get_person_by_number(person_number or "")
        credential = (
            await self._store.get_person_credential(person.id)
            if person is not None
            else None
        )
        if credential is None or person is None:
            # Dummy bcrypt to smooth timing when the person/credential is absent.
            verify_person_password(person_password, hash_person_password("dummy"))
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error_code": "invalid_person_credentials",
                    "message": "Invalid person credentials",
                },
            )
        await self._verify_password_or_record_failure(
            credential,
            person_password,
            error_code="invalid_person_credentials",
            message="Invalid person credentials",
        )
        if person.status != "active":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error_code": "person_disabled", "message": "Person is disabled"},
            )
        active_links = await self._store.list_person_account_links(
            person.id, only_active=True
        )
        if not active_links:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error_code": "account_not_linked", "message": "No linked account"},
            )
        # Resolve target account. When omitted, land on the most recently
        # used eligible account (multi-link persons no longer get a 409; the
        # session can switch afterwards at no extra credential cost).
        if account_id is None:
            if len(active_links) == 1:
                target_link = active_links[0]
            else:
                target_link = await self._pick_default_link(
                    person.id, active_links
                )
        else:
            target_link = next(
                (lnk for lnk in active_links if lnk.account_id == account_id), None
            )
            if target_link is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error_code": "account_not_linked",
                        "message": "Account is not linked",
                    },
                )
        account = await self._store.get_enterprise_user_by_id(target_link.account_id)
        if account is None or account.status != USER_STATUS_ACTIVE:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error_code": "account_not_active", "message": "Account is not active"},
            )
        now = utc_now_iso()
        session = EnterprisePersonLoginSessionRecord(
            id=f"psess_{secrets.token_hex(12)}",
            person_id=person.id,
            active_account_id=account.id,
            status="active",
            person_epoch=person.auth_epoch,
            session_epoch=1,
            absolute_expires_at=self._tokens.session_absolute_expiry(),
            created_at=now,
            last_seen_at=None,
            revoked_at=None,
        )
        try:
            saved_session = await self._store.create_person_session_atomic(
                session,
                expected_person_epoch=person.auth_epoch,
                actor_user_id=account.id,
            )
        except MetadataConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"error_code": "session_epoch_conflict", "message": "Retry the request"},
            ) from exc
        await self._reset_failed_login(credential)
        token = self._issue_token(session=saved_session)
        return {
            "person": person.to_dict(),
            "active_account": _account_summary(account),
            "session": saved_session.to_dict(),
            "access_token": token,
            "token_type": "bearer",
            "expires_in": self._expires_in(saved_session),
        }

    # -- list accounts -----------------------------------------------------

    async def list_accounts(
        self, *, person_id: str, active_account_id: str | None
    ) -> dict[str, Any]:
        person = await self._store.get_person_by_id(person_id)
        if person is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error_code": "person_not_found", "message": "Person not found"},
            )
        links = await self._store.list_person_account_links(person.id, only_active=True)
        accounts = []
        for link in links:
            user = await self._store.get_enterprise_user_by_id(link.account_id)
            if user is not None:
                accounts.append(_account_summary(user))
        return {
            "person": {"person_id": person.id, "status": person.status},
            "active_account_id": active_account_id,
            "accounts": accounts,
        }

    # -- switch ------------------------------------------------------------

    async def switch(
        self,
        *,
        person_id: str,
        session_id: str,
        expected_session_epoch: int,
        target_account_id: str,
    ) -> dict[str, Any]:
        link = await self._store.get_person_account_link(person_id, target_account_id)
        if link is None or link.status != "active":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error_code": "account_not_linked", "message": "Account is not linked"},
            )
        account = await self._store.get_enterprise_user_by_id(target_account_id)
        if account is None or account.status != USER_STATUS_ACTIVE:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error_code": "account_not_active", "message": "Account is not active"},
            )
        # Re-check at switch time: an account promoted to super_admin AFTER the
        # link was activated must not become reachable through a person token
        # (doc 4.4 #4 — the person mechanism never widens the super-admin
        # surface).
        if account.system_role == SYSTEM_ROLE_SUPER_ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error_code": "cannot_bind_super_admin",
                    "message": "Cannot switch into a super admin account",
                },
            )
        try:
            saved_session = await self._store.switch_person_session_atomic(
                session_id=session_id,
                expected_session_epoch=expected_session_epoch,
                target_account_id=target_account_id,
                actor_user_id=account.id,
            )
        except MetadataConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"error_code": "session_epoch_conflict", "message": "Retry the request"},
            ) from exc
        except MetadataRecordNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error_code": "account_not_linked", "message": "Account is not linked"},
            ) from exc
        token = self._issue_token(session=saved_session)
        return {
            "active_account": _account_summary(account),
            "session": saved_session.to_dict(),
            "access_token": token,
            "token_type": "bearer",
            "expires_in": self._expires_in(saved_session),
        }

    # -- logout ------------------------------------------------------------

    async def logout(self, *, session_id: str) -> dict[str, Any]:
        await self._store.revoke_person_session_atomic(session_id)
        return {"status": "logged_out", "session_id": session_id}

    async def logout_all(self, *, person_id: str) -> dict[str, Any]:
        _person, revoked = await self._store.revoke_all_person_sessions_atomic(
            person_id
        )
        return {"status": "logged_out_all", "revoked_sessions": revoked}

    # -- change password ---------------------------------------------------

    async def change_password(
        self,
        *,
        person_id: str,
        current_password: str,
        new_password: str,
    ) -> dict[str, Any]:
        self._validate_password(new_password)
        person = await self._store.get_person_by_id(person_id)
        if person is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error_code": "person_not_found", "message": "Person not found"},
            )
        credential = await self._store.get_person_credential(person_id)
        if credential is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error_code": "invalid_current_password",
                    "message": "Current password is invalid",
                },
            )
        # Same lockout counter as login/confirm (person-password re-auth).
        await self._verify_password_or_record_failure(
            credential,
            current_password,
            error_code="invalid_current_password",
            message="Current password is invalid",
        )
        new_credential = EnterprisePersonCredentialRecord(
            id=credential.id,
            person_id=person_id,
            credential_type="password",
            algorithm="bcrypt",
            password_hash=hash_person_password(new_password),
            status="active",
            failed_count=0,
            locked_until=None,
            last_used_at=None,
            created_at=credential.created_at,
            updated_at=utc_now_iso(),
        )
        try:
            await self._store.rotate_person_credential_atomic(
                person_id=person_id,
                new_credential=new_credential,
                actor_user_id=None,
            )
        except MetadataRecordNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error_code": "person_not_found", "message": "Person not found"},
            ) from exc
        return {"status": "password_changed"}

    # -- enrollment grants -------------------------------------------------

    async def create_enrollment_grant(
        self,
        *,
        account_id: str,
        created_by: str,
        ttl_seconds: int = 900,
        person_number: str | None = None,
    ) -> dict[str, Any]:
        account = await self._store.get_enterprise_user_by_id(account_id)
        if account is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error_code": "account_not_found", "message": "Account not found"},
            )
        if account.system_role == SYSTEM_ROLE_SUPER_ADMIN:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "cannot_bind_super_admin",
                    "message": "Cannot bind a super admin account",
                },
            )
        person_number = self._normalize_person_number(person_number)
        if person_number is not None:
            # Fail-fast duplicate check; the persons unique index at enroll
            # time remains the concurrency arbiter.
            existing = await self._store.get_person_by_number(person_number)
            if existing is not None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail={
                        "error_code": "person_number_conflict",
                        "message": "person_number is already bound to another person",
                    },
                )
        plain_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(plain_token.encode("utf-8")).hexdigest()
        now = datetime.now(timezone.utc)
        grant = EnterprisePersonEnrollmentGrantRecord(
            id=f"pgrant_{secrets.token_hex(12)}",
            account_id=account_id,
            token_hash=token_hash,
            status="active",
            created_by=created_by,
            consumed_by_person=None,
            expires_at=(now + timedelta(seconds=max(60, int(ttl_seconds)))).isoformat(),
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
            consumed_at=None,
            person_number=person_number,
        )
        try:
            saved = await self._store.create_person_enrollment_grant_atomic(
                grant, actor_user_id=created_by
            )
        except MetadataConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error_code": "active_grant_exists",
                    "message": "An active grant already exists for this account",
                },
            ) from exc
        return {
            "grant_id": saved.id,
            # Plaintext token is returned exactly once; only the hash persists.
            "grant_token": plain_token,
            "account_id": saved.account_id,
            "expires_at": saved.expires_at,
            "person_number": saved.person_number,
        }

    async def revoke_enrollment_grant(self, *, grant_id: str) -> dict[str, Any]:
        result = await self._store.revoke_person_enrollment_grant_atomic(grant_id)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error_code": "grant_not_found", "message": "Grant not found"},
            )
        return {"status": "revoked", "grant_id": grant_id, "current_status": result.status}

    _GRANT_STATUSES = frozenset({"active", "consumed", "revoked"})

    async def list_enrollment_grants(
        self,
        *,
        account_id: str | None = None,
        status_filter: str | None = None,
    ) -> dict[str, Any]:
        """List issued enrollment grants for the super-admin surface.

        The stored ``token_hash`` is never exposed — even the hash stays
        server-side (the plaintext token is returned exactly once at create
        time). ``expired`` reports whether ``expires_at`` has passed; an
        ``active`` grant with ``expired=true`` can no longer be consumed.
        """

        if status_filter is not None and status_filter not in self._GRANT_STATUSES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "validation_error",
                    "message": "status must be one of active/consumed/revoked",
                },
            )
        grants = await self._store.list_person_enrollment_grants(
            account_id=account_id, status=status_filter
        )
        now = utc_now_iso()
        items = [
            {
                "grant_id": grant.id,
                "account_id": grant.account_id,
                "status": grant.status,
                "expired": grant.expires_at <= now,
                "created_by": grant.created_by,
                "consumed_by_person": grant.consumed_by_person,
                "person_number": grant.person_number,
                "expires_at": grant.expires_at,
                "created_at": grant.created_at,
                "updated_at": grant.updated_at,
                "consumed_at": grant.consumed_at,
            }
            for grant in grants
        ]
        return {"grants": items, "total": len(items)}

    # -- link management ---------------------------------------------------

    async def authorize_link_proposal(
        self, *, proposer: Principal, account_id: str
    ) -> None:
        """Enforce who may propose a pending link for ``account_id``.

        Matrix (docs/多账号身份关联与切换执行文档.md 5.9):

        * ``super_admin`` may propose for any existing non-super-admin account.
        * A tenant admin (admin/owner role) may propose only for accounts whose
          canonical tenant is one they administer, and only when the target is
          a regular member — binding another tenant-admin account to a person
          stays a super-admin decision.
        * Accounts outside the proposer's admin scope read as 404 so tenant
          admins cannot probe other tenants' account IDs.
        """

        if proposer.is_super_admin:
            return
        admin_tenants = {
            tenant_id
            for tenant_id, role in (proposer.tenant_roles or {}).items()
            if tenant_role_is_admin(role)
        }
        if not admin_tenants:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error_code": "admin_required",
                    "message": "Super admin or tenant admin is required",
                },
            )
        account = await self._store.get_enterprise_user_by_id(account_id)
        if (
            account is None
            or account.tenant_id is None
            or account.tenant_id not in admin_tenants
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error_code": "account_not_found", "message": "Account not found"},
            )
        memberships = await self._store.list_user_tenant_memberships(account.id)
        if any(
            membership.tenant_id == account.tenant_id
            and tenant_role_is_admin(membership.role)
            for membership in memberships
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error_code": "super_admin_required",
                    "message": "Binding a tenant admin account requires super admin",
                },
            )

    async def propose_link(
        self, *, person_id: str, account_id: str, bound_by: str, reason: str | None = None
    ) -> dict[str, Any]:
        account = await self._store.get_enterprise_user_by_id(account_id)
        if account is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error_code": "account_not_found", "message": "Account not found"},
            )
        if account.system_role == SYSTEM_ROLE_SUPER_ADMIN:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "cannot_bind_super_admin",
                    "message": "Cannot bind a super admin account",
                },
            )
        person = await self._store.get_person_by_id(person_id)
        if person is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error_code": "person_not_found", "message": "Person not found"},
            )
        now = utc_now_iso()
        link = EnterprisePersonAccountLinkRecord(
            id=f"plink_{secrets.token_hex(12)}",
            person_id=person_id,
            account_id=account_id,
            status="pending",
            bound_by=bound_by,
            bound_at=now,
            confirmed_by_person_at=None,
            revoked_by=None,
            revoked_at=None,
            reason=reason,
            created_at=now,
            updated_at=now,
        )
        try:
            saved = await self._store.propose_person_account_link_atomic(
                link, actor_user_id=bound_by
            )
        except MetadataConflictError as exc:
            # Same (person, account) pair already active — proposing again is
            # a state conflict, not a server error.
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error_code": "link_state_conflict",
                    "message": "Link is already active",
                },
            ) from exc
        return {"link": saved.to_dict()}

    _LINK_STATUSES = frozenset({"pending", "active", "revoked"})

    async def list_links(
        self, *, person_id: str, status_filter: str | None = None
    ) -> dict[str, Any]:
        """List the person's own account links in every state.

        Unlike :meth:`list_accounts` (active links only, the switchable set)
        this surfaces ``pending`` links awaiting the person's confirmation and
        ``revoked`` history. Each entry carries the non-secret account summary
        so the person can tell which account a pending link points at; the
        summary is ``None`` when the account row no longer exists.
        """

        if status_filter is not None and status_filter not in self._LINK_STATUSES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "validation_error",
                    "message": "status must be one of pending/active/revoked",
                },
            )
        links = await self._store.list_person_account_links(
            person_id, only_active=False
        )
        if status_filter is not None:
            links = [link for link in links if link.status == status_filter]
        items = []
        for link in links:
            account = await self._store.get_enterprise_user_by_id(link.account_id)
            items.append(
                {
                    "link": link.to_dict(),
                    "account": _account_summary(account)
                    if account is not None
                    else None,
                }
            )
        return {"person_id": person_id, "links": items, "total": len(items)}

    async def confirm_link(
        self, *, person_id: str, account_id: str, person_password: str
    ) -> dict[str, Any]:
        credential = await self._store.get_person_credential(person_id)
        if credential is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error_code": "invalid_person_password",
                    "message": "Person password is invalid",
                },
            )
        # Confirm is a person-password re-authentication: it shares the same
        # lockout counter as login so the password cannot be brute-forced here.
        await self._verify_password_or_record_failure(
            credential,
            person_password,
            error_code="invalid_person_password",
            message="Person password is invalid",
        )
        await self._reset_failed_login(credential)
        # Re-check at confirm time: the target may have been promoted to
        # super_admin after the pending link was proposed (doc 5.9 lists
        # cannot_bind_super_admin among confirm errors).
        account = await self._store.get_enterprise_user_by_id(account_id)
        if account is not None and account.system_role == SYSTEM_ROLE_SUPER_ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error_code": "cannot_bind_super_admin",
                    "message": "Cannot bind a super admin account",
                },
            )
        try:
            _person, link = await self._store.confirm_person_account_link_atomic(
                person_id=person_id,
                account_id=account_id,
                actor_user_id=account_id,
            )
        except MetadataConflictError as exc:
            if exc.entity_type == "person_account_link_active":
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail={
                        "error_code": "account_already_linked",
                        "message": "Account is already linked to a person",
                    },
                ) from exc
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error_code": "link_state_conflict",
                    "message": "Link state conflict",
                },
            ) from exc
        except MetadataRecordNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error_code": "link_not_found", "message": "Pending link not found"},
            ) from exc
        return {"link": link.to_dict()}

    async def unbind_link(
        self, *, person_id: str, account_id: str, revoked_by: str, reason: str | None = None
    ) -> dict[str, Any]:
        try:
            link, revoked = await self._store.revoke_person_account_link_atomic(
                person_id=person_id,
                account_id=account_id,
                actor_user_id=revoked_by,
                reason=reason,
            )
        except MetadataRecordNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error_code": "link_not_found", "message": "Link not found"},
            ) from exc
        return {
            "status": "unlinked",
            "person_id": person_id,
            "account_id": account_id,
            "revoked_sessions": revoked,
        }

    # -- person enable/disable --------------------------------------------

    _PERSON_STATUSES = frozenset({"active", "disabled"})

    async def set_person_number(
        self,
        *,
        person_id: str,
        person_number: str | None,
        actor_user_id: str | None = None,
    ) -> dict[str, Any]:
        """Bind or clear the person's 工号 (super-admin 补录/修改).

        ``None``/empty clears the binding; after a bind both the person_id
        and the 工号 log in. Duplicate numbers map to 409
        ``person_number_conflict``.
        """

        normalized = self._normalize_person_number(person_number)
        try:
            person = await self._store.set_person_number_atomic(
                person_id=person_id,
                person_number=normalized,
                actor_user_id=actor_user_id,
            )
        except MetadataConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error_code": "person_number_conflict",
                    "message": "person_number is already bound to another person",
                },
            ) from exc
        except MetadataRecordNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error_code": "person_not_found", "message": "Person not found"},
            ) from exc
        return {"person": person.to_dict()}

    async def list_persons(
        self, *, status_filter: str | None = None
    ) -> dict[str, Any]:
        """List natural persons with their account links (super-admin view)."""

        if status_filter is not None and status_filter not in self._PERSON_STATUSES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "validation_error",
                    "message": "status must be one of active/disabled",
                },
            )
        persons = await self._store.list_persons(status=status_filter)
        items = []
        for person in persons:
            links = await self._store.list_person_account_links(
                person.id, only_active=False
            )
            entry = person.to_dict()
            entry["links"] = [link.to_dict() for link in links]
            items.append(entry)
        return {"persons": items, "total": len(items)}

    async def disable_person(self, *, person_id: str, reason: str | None = None) -> dict[str, Any]:
        try:
            person = await self._store.disable_person_atomic(
                person_id=person_id, actor_user_id=None, reason=reason
            )
        except MetadataRecordNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error_code": "person_not_found", "message": "Person not found"},
            ) from exc
        return {"person_id": person.id, "status": person.status}

    async def enable_person(self, *, person_id: str) -> dict[str, Any]:
        try:
            person = await self._store.enable_person_atomic(person_id=person_id)
        except MetadataRecordNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error_code": "person_not_found", "message": "Person not found"},
            ) from exc
        return {"person_id": person.id, "status": person.status}

    # -- person KB shares (cross-department personal KB usage) -------------

    _KB_SHARE_ROLES = frozenset({"kb_viewer", "kb_editor", "kb_admin"})

    async def _require_share_manager(
        self, *, kb_record: Any, requester: Principal
    ) -> EnterprisePersonRecord:
        """Common gate for share create/list/revoke on one KB.

        Only personal KBs (non-tenant origin, with an owner account) can be
        person-shared; the requester must be that owner (or super admin), and
        the owner account must belong to an enrolled person — the share target
        set is that person's other active-link accounts, never arbitrary users.
        """

        if kb_record.origin == "tenant" or not kb_record.owner_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error_code": "person_share_requires_personal_kb",
                    "message": "Only personal KBs can be person-shared",
                },
            )
        if not requester.is_super_admin and requester.user_id != kb_record.owner_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error_code": "kb_owner_required",
                    "message": "Only the KB owner can manage person shares",
                },
            )
        owner_link = await self._store.get_active_person_link_for_account(
            kb_record.owner_id
        )
        if owner_link is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error_code": "person_identity_required",
                    "message": "The KB owner account is not linked to a person",
                },
            )
        person = await self._store.get_person_by_id(owner_link.person_id)
        if person is None or person.status != "active":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error_code": "person_disabled", "message": "Person is disabled"},
            )
        return person

    async def share_kb(
        self,
        *,
        kb_record: Any,
        requester: Principal,
        target_account_id: str,
        role: str = "kb_editor",
        reason: str | None = None,
    ) -> dict[str, Any]:
        person = await self._require_share_manager(
            kb_record=kb_record, requester=requester
        )
        normalized_role = (role or "kb_editor").strip()
        if normalized_role not in self._KB_SHARE_ROLES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "validation_error",
                    "message": "role must be one of kb_viewer/kb_editor/kb_admin",
                },
            )
        if target_account_id == kb_record.owner_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "validation_error",
                    "message": "Cannot share a KB to its owner account",
                },
            )
        target_link = await self._store.get_person_account_link(
            person.id, target_account_id
        )
        if target_link is None or target_link.status != "active":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error_code": "account_not_linked",
                    "message": "Target account is not an active link of this person",
                },
            )
        target_account = await self._store.get_enterprise_user_by_id(
            target_account_id
        )
        if target_account is None or target_account.status != USER_STATUS_ACTIVE:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error_code": "account_not_active", "message": "Account is not active"},
            )
        now = utc_now_iso()
        share = EnterprisePersonKBShareRecord(
            id=f"pkbs_{secrets.token_hex(12)}",
            kb_id=kb_record.id,
            person_id=person.id,
            owner_account_id=kb_record.owner_id,
            target_account_id=target_account_id,
            target_tenant_id=target_account.tenant_id,
            role=normalized_role,
            status="active",
            created_by=requester.user_id,
            revoked_by=None,
            reason=reason,
            created_at=now,
            updated_at=now,
            revoked_at=None,
        )
        try:
            saved = await self._store.create_person_kb_share_atomic(
                share,
                expected_generation=kb_record.generation,
                actor_user_id=requester.user_id,
            )
        except KBLifecycleConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error_code": "kb_lifecycle_conflict",
                    "message": "Knowledge base changed; retry the request",
                },
            ) from exc
        return {"share": saved.to_dict()}

    async def unshare_kb(
        self,
        *,
        kb_record: Any,
        requester: Principal,
        target_account_id: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        await self._require_share_manager(kb_record=kb_record, requester=requester)
        share, revoked = await self._store.revoke_person_kb_share_atomic(
            kb_record.id,
            target_account_id,
            revoked_by=requester.user_id,
            reason=reason,
        )
        if share is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error_code": "share_not_found", "message": "Share not found"},
            )
        return {
            "status": "unshared",
            "kb_id": kb_record.id,
            "target_account_id": target_account_id,
            "revoked": revoked,
            "share": share.to_dict(),
        }

    async def list_kb_shares(
        self, *, kb_record: Any, requester: Principal
    ) -> dict[str, Any]:
        await self._require_share_manager(kb_record=kb_record, requester=requester)
        shares = await self._store.list_person_kb_shares(kb_id=kb_record.id)
        return {"kb_id": kb_record.id, "shares": [s.to_dict() for s in shares]}

    async def list_my_kb_shares(self, *, person_id: str) -> dict[str, Any]:
        shares = await self._store.list_person_kb_shares(
            person_id=person_id, only_active=True
        )
        return {"shares": [s.to_dict() for s in shares]}


def _account_summary(user: EnterpriseUserRecord) -> dict[str, Any]:
    return {
        "account_id": user.id,
        "username": user.username,
        "status": user.status,
        "system_role": user.system_role,
        "tenant_id": user.tenant_id,
    }


# ---------------------------------------------------------------------------
# Session validation (shared by both dependencies)
# ---------------------------------------------------------------------------


async def _build_session_context_from_claims(
    claims: dict[str, Any], *, require_account_access: bool
) -> PersonSessionContext:
    """Validate person/session state and (optionally) build a Principal.

    When ``require_account_access`` is True (combined_auth for business APIs)
    the selected account must be active, its live token_version must match the
    snapshot stored on the session (so a password reset invalidates outstanding
    v2 tokens — doc 6.4/7.1), and an active (person, account) link must exist.
    A full account Principal is built and the membership consistency invariant
    is enforced.

    When False (session-control: accounts/switch/logout/change-password) the
    account active/token_version/link/membership checks are ALL skipped per
    doc I-11, and no Principal is built (the endpoints operate on
    ``request.state.person_session``). This keeps the path usable when the
    current account is disabled or its canonical membership was stripped.
    """

    store = _active_metadata_store()
    person_id = str(claims["person_id"])
    user_id = str(claims["user_id"])
    session_id = str(claims["sid"])
    person_epoch = int(claims["person_epoch"])
    session_epoch = int(claims["session_epoch"])

    person = await store.get_person_by_id(person_id)
    if person is None or person.status != "active":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": "person_session_invalid", "message": "Person is not active"},
        )
    if person.auth_epoch != person_epoch:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": "person_session_invalid", "message": "Person epoch mismatch"},
        )
    session = await store.get_person_login_session(session_id)
    if session is None or session.status != "active":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": "person_session_invalid", "message": "Session is not active"},
        )
    if session.absolute_expires_at <= utc_now_iso():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": "person_session_invalid", "message": "Session expired"},
        )
    if session.person_epoch != person_epoch or session.session_epoch != session_epoch:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": "person_session_invalid", "message": "Session epoch mismatch"},
        )

    if not require_account_access:
        # Session-control: no account Principal, no account-state checks. The
        # person/session/epoch checks above are sufficient (doc I-11).
        return PersonSessionContext(
            person=person,
            session=session,
            person_epoch=person_epoch,
            session_epoch=session_epoch,
        )

    # --- account-access path ---
    account = await store.get_enterprise_user_by_id(user_id)
    if account is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": "person_session_invalid", "message": "Account not found"},
        )
    if account.status != USER_STATUS_ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_code": "person_session_invalid",
                "message": "Account is not active",
            },
        )
    # Fail closed if the account was promoted to super_admin after linking:
    # person tokens must never resolve to a super-admin Principal (doc 4.4 #4).
    if account.system_role == SYSTEM_ROLE_SUPER_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_code": "person_session_invalid",
                "message": "Account is not reachable via person session",
            },
        )
    # P0: compare the live account token_version against the snapshot captured
    # at session creation/switch time. A password reset bumps token_version
    # without changing status, so this is what invalidates the outstanding v2
    # account-access token (doc 6.4/7.1). session-control skips this.
    if account.token_version != session.account_token_version:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_code": "person_session_invalid",
                "message": "Token has been revoked",
            },
        )
    link = await store.get_person_account_link(person_id, user_id)
    if link is None or link.status != "active":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_code": "person_session_invalid",
                "message": "Account is not linked",
            },
        )
    if session.active_account_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_code": "person_session_invalid",
                "message": "Active account mismatch",
            },
        )

    memberships = await store.list_user_tenant_memberships(user_id)
    principal = principal_from_user(
        account, auth_method=PERSON_JWT_AUTH_METHOD, memberships=memberships
    )
    return PersonSessionContext(
        person=person,
        session=session,
        person_epoch=person_epoch,
        session_epoch=session_epoch,
        principal=principal,
        account=account,
    )


def _active_metadata_store() -> EnterpriseMetadataStore:
    """Return the metadata store bound to the running app state.

    Resolved lazily from the FastAPI app state via the enterprise user service,
    which already holds a reference to the same store. This avoids threading a
    separate store singleton.
    """

    from lightrag.api.enterprise_auth import _get_request_app_state_metadata_store

    return _get_request_app_state_metadata_store()


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

# Shared bearer scheme for person-only routes (session-control). The account
# access path is invoked from combined_auth, which already has the bearer.
_person_oauth2 = OAuth2PasswordBearer(
    tokenUrl="login", auto_error=False, description="Person access token"
)


def _extract_bearer_token(request: Request) -> str | None:
    header = request.headers.get("Authorization") or request.headers.get("authorization")
    if not header:
        return None
    parts = header.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        token = parts[1].strip()
        return token or None
    return None


async def require_person_session_control(
    request: Request,
    token: str | None = Security(_person_oauth2),
) -> PersonSessionContext:
    """Session-control dependency (accounts/switch/logout/change-password).

    Validates the v2 token, person active, session active/expiry and the
    person/session epochs. Deliberately does NOT require the current account to
    be active or the link to still exist, so a user whose current account was
    disabled can still switch or log out.
    """

    if not person_auth_enabled():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error_code": "person_session_required", "message": "Person auth disabled"},
        )
    raw_token = token or _extract_bearer_token(request)
    if not raw_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": "authentication_required", "message": "Token required"},
        )
    handler = get_person_token_handler()
    claims = handler.validate_person_token(raw_token)
    context = await _build_session_context_from_claims(claims, require_account_access=False)
    request.state.principal = context.principal
    request.state.person_session = context
    return context


async def person_account_access_validate(
    request: Request, token: str
) -> PersonSessionContext:
    """Account-access validation invoked by ``combined_auth`` for v2 tokens.

    Performs the full session-control checks PLUS account active, link and
    active-account-id checks, then builds the account-scoped Principal so the
    existing business API keeps working unchanged.
    """

    handler = get_person_token_handler()
    claims = handler.validate_person_token(token)
    context = await _build_session_context_from_claims(claims, require_account_access=True)
    request.state.principal = context.principal
    request.state.person_session = context
    return context
