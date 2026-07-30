"""Multi-account person identity routes (Phase 4).

Factory ``create_person_routes()`` returns an :class:`APIRouter` with the
person self-service and super-admin management endpoints defined in
``docs/多账号身份关联与切换执行文档.md`` section 5.

Two dependency classes, kept strictly separate (doc 5.10):

* **session-control endpoints** (accounts/switch/logout/logout-all/change-
  password/confirm-link) use ``require_person_session_control`` and the v2
  person access JWT. They deliberately do NOT go through ``combined_auth`` —
  combined_auth would re-process the v2 token and the two paths must not
  duplicate each other.
* **super-admin management endpoints** go through ``combined_auth`` (legacy
  interactive JWT) and require an active super-admin principal.

Every endpoint is guarded by ``person_auth_enabled()``; when the feature flag
is off all person routes return 503 ``person_auth_unavailable`` so a disabled
deployment never exposes the surface.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from lightrag.api.enterprise_auth import (
    INTERACTIVE_AUTH_METHODS,
    Principal,
    get_request_principal,
)
from lightrag.api.person_auth import (
    PersonSessionContext,
    person_auth_enabled,
    require_person_session_control,
)
from lightrag.api.utils_api import get_combined_auth_dependency


# ---------------------------------------------------------------------------
# Response / request models
# ---------------------------------------------------------------------------


class PersonAccountSummary(BaseModel):
    """Non-secret projection of an enterprise account for person responses."""

    account_id: str
    username: str
    status: str
    system_role: str
    tenant_id: str | None = None


class PersonTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    # enroll/login return a person summary; switch does not (the person is
    # unchanged), so this field is optional for the switch response.
    person: dict[str, Any] | None = None
    active_account: PersonAccountSummary
    session: dict[str, Any]


class PersonAccountsResponse(BaseModel):
    person: dict[str, Any]
    active_account_id: str | None = None
    accounts: list[PersonAccountSummary]


class PersonEnrollRequest(BaseModel):
    grant_token: str = Field(..., min_length=1)
    person_password: str = Field(..., min_length=1)


class PersonLoginRequest(BaseModel):
    person_id: str = Field(..., min_length=1)
    person_password: str = Field(..., min_length=1)
    account_id: str | None = None


class PersonSwitchRequest(BaseModel):
    account_id: str = Field(..., min_length=1)


class PersonChangePasswordRequest(BaseModel):
    current_person_password: str = Field(..., min_length=1)
    new_person_password: str = Field(..., min_length=1)


class PersonConfirmLinkRequest(BaseModel):
    person_password: str = Field(..., min_length=1)


class PersonLinkSummary(BaseModel):
    link: dict[str, Any]


class CreateEnrollmentGrantRequest(BaseModel):
    account_id: str = Field(..., min_length=1)
    ttl_seconds: int = 900
    reason: str | None = None


class EnrollmentGrantResponse(BaseModel):
    grant_id: str
    grant_token: str
    account_id: str
    expires_at: str


class ProposeLinkRequest(BaseModel):
    reason: str | None = None


class PersonDisableRequest(BaseModel):
    reason: str | None = None


class UnbindLinkResponse(BaseModel):
    status: str
    person_id: str
    account_id: str
    revoked_sessions: int


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_person_routes(api_key: str | None = None) -> APIRouter:
    """Build the person-identity APIRouter.

    ``api_key`` is forwarded to ``combined_auth`` so the super-admin management
    endpoints honor the same API-key/legacy-JWT gating as the enterprise
    routes. The self-service session-control endpoints bypass ``combined_auth``
    entirely (they consume the v2 person token via
    ``require_person_session_control``).
    """

    router = APIRouter(tags=["person-auth"])
    combined_auth = get_combined_auth_dependency(api_key)

    def _require_person_enabled() -> None:
        if not person_auth_enabled():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error_code": "person_auth_unavailable",
                    "message": "Person auth is not enabled",
                },
            )

    def _person_service(request: Request):
        from lightrag.api.enterprise_auth import get_person_service

        return get_person_service(request)

    def _token_handler(request: Request):
        from lightrag.api.enterprise_auth import get_person_token_handler

        return get_person_token_handler(request)

    def _require_interactive_principal(request: Request) -> Principal:
        principal = get_request_principal(request)
        if principal is None:
            raise HTTPException(status_code=401, detail="Login required")
        if principal.auth_method not in INTERACTIVE_AUTH_METHODS:
            raise HTTPException(
                status_code=403,
                detail={
                    "error_code": "interactive_jwt_required",
                    "message": "Only interactive JWTs are accepted",
                },
            )
        return principal

    def _require_interactive_super_admin(request: Request) -> Principal:
        principal = _require_interactive_principal(request)
        if not principal.is_super_admin:
            raise HTTPException(
                status_code=403,
                detail={
                    "error_code": "super_admin_required",
                    "message": "Super admin is required",
                },
            )
        return principal

    def _access_ttl(request: Request) -> int:
        return int(_token_handler(request).access_ttl_seconds)

    def _token_response(
        request: Request, payload: dict[str, Any]
    ) -> PersonTokenResponse:
        # login/switch return an `active_account` dict summary already; map it
        # to the response model. enroll resolves its own summary (see handler).
        # switch does not echo the person (it is unchanged).
        account_summary = PersonAccountSummary(**payload["active_account"])
        return PersonTokenResponse(
            access_token=payload["access_token"],
            token_type=payload.get("token_type", "bearer"),
            expires_in=int(payload.get("expires_in", _access_ttl(request))),
            person=payload.get("person"),
            active_account=account_summary,
            session=payload["session"],
        )

    # ------------------------------------------------------------------
    # Public self-service endpoints (no auth dependency; enroll/login use the
    # grant/person credential directly).
    # ------------------------------------------------------------------

    @router.post(
        "/auth/person/enroll",
        response_model=PersonTokenResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def person_enroll(request: Request, body: PersonEnrollRequest):
        _require_person_enabled()
        service = _person_service(request)
        # enroll returns {person, active_account_id, session, access_token}.
        result = await service.enroll(
            grant_token=body.grant_token,
            person_password=body.person_password,
            rate_key=request.client.host if request.client else None,
        )
        # Resolve the active account summary for the documented response shape.
        from lightrag.api.enterprise_auth import get_enterprise_user_service

        account = await get_enterprise_user_service(request).get_user_or_404(
            result["active_account_id"]
        )
        return PersonTokenResponse(
            access_token=result["access_token"],
            token_type=result.get("token_type", "bearer"),
            expires_in=int(result.get("expires_in", _access_ttl(request))),
            person=result["person"],
            active_account=PersonAccountSummary(
                account_id=account.id,
                username=account.username,
                status=account.status,
                system_role=account.system_role,
                tenant_id=account.tenant_id,
            ),
            session=result["session"],
        )

    @router.post("/auth/person/login", response_model=PersonTokenResponse)
    async def person_login(request: Request, body: PersonLoginRequest):
        _require_person_enabled()
        service = _person_service(request)
        result = await service.login(
            person_id=body.person_id,
            person_password=body.person_password,
            account_id=body.account_id,
        )
        return _token_response(request, result)

    # ------------------------------------------------------------------
    # Person session-control endpoints (v2 token via require_person_session_control)
    # ------------------------------------------------------------------

    @router.get("/auth/person/accounts", response_model=PersonAccountsResponse)
    async def person_accounts(
        request: Request,
        session: PersonSessionContext = Depends(require_person_session_control),
    ):
        _require_person_enabled()
        service = _person_service(request)
        result = await service.list_accounts(
            person_id=session.person.id,
            active_account_id=session.session.active_account_id,
        )
        accounts = [PersonAccountSummary(**item) for item in result["accounts"]]
        return PersonAccountsResponse(
            person=result["person"],
            active_account_id=result["active_account_id"],
            accounts=accounts,
        )

    @router.post("/auth/person/switch", response_model=PersonTokenResponse)
    async def person_switch(
        request: Request,
        body: PersonSwitchRequest,
        session: PersonSessionContext = Depends(require_person_session_control),
    ):
        _require_person_enabled()
        service = _person_service(request)
        result = await service.switch(
            person_id=session.person.id,
            session_id=session.session.id,
            expected_session_epoch=session.session_epoch,
            target_account_id=body.account_id,
        )
        return _token_response(request, result)

    @router.post("/auth/person/logout")
    async def person_logout(
        request: Request,
        session: PersonSessionContext = Depends(require_person_session_control),
    ):
        _require_person_enabled()
        service = _person_service(request)
        return await service.logout(session_id=session.session.id)

    @router.post("/auth/person/logout-all")
    async def person_logout_all(
        request: Request,
        session: PersonSessionContext = Depends(require_person_session_control),
    ):
        _require_person_enabled()
        service = _person_service(request)
        return await service.logout_all(person_id=session.person.id)

    @router.post("/auth/person/change-password")
    async def person_change_password(
        request: Request,
        body: PersonChangePasswordRequest,
        session: PersonSessionContext = Depends(require_person_session_control),
    ):
        _require_person_enabled()
        service = _person_service(request)
        return await service.change_password(
            person_id=session.person.id,
            current_password=body.current_person_password,
            new_password=body.new_person_password,
        )

    @router.post(
        "/auth/person/links/{account_id}:confirm", response_model=PersonLinkSummary
    )
    async def person_confirm_link(
        request: Request,
        account_id: str,
        body: PersonConfirmLinkRequest,
        session: PersonSessionContext = Depends(require_person_session_control),
    ):
        _require_person_enabled()
        service = _person_service(request)
        return await service.confirm_link(
            person_id=session.person.id,
            account_id=account_id,
            person_password=body.person_password,
        )

    # ------------------------------------------------------------------
    # Super-admin management endpoints (combined_auth + interactive super admin)
    # ------------------------------------------------------------------

    @router.post(
        "/admin/persons/enrollment-grants",
        response_model=EnrollmentGrantResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(combined_auth)],
    )
    async def create_enrollment_grant(
        request: Request, body: CreateEnrollmentGrantRequest
    ):
        _require_person_enabled()
        principal = _require_interactive_super_admin(request)
        service = _person_service(request)
        result = await service.create_enrollment_grant(
            account_id=body.account_id,
            created_by=principal.user_id,
            ttl_seconds=body.ttl_seconds,
        )
        return EnrollmentGrantResponse(
            grant_id=result["grant_id"],
            grant_token=result["grant_token"],
            account_id=result["account_id"],
            expires_at=result["expires_at"],
        )

    @router.delete(
        "/admin/persons/enrollment-grants/{grant_id}",
        dependencies=[Depends(combined_auth)],
    )
    async def revoke_enrollment_grant(request: Request, grant_id: str):
        _require_person_enabled()
        _require_interactive_super_admin(request)
        service = _person_service(request)
        return await service.revoke_enrollment_grant(grant_id=grant_id)

    @router.post(
        "/admin/persons/{person_id}/accounts/{account_id}",
        response_model=PersonLinkSummary,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(combined_auth)],
    )
    async def propose_account_link(
        request: Request,
        person_id: str,
        account_id: str,
        body: ProposeLinkRequest | None = None,
    ):
        """Propose a pending link.

        Super admins may target any non-super-admin account; tenant admins may
        target regular member accounts of their own tenant (binding another
        tenant-admin account remains super-admin-only). The link always starts
        as ``pending`` and requires the person's own confirmation.
        """

        _require_person_enabled()
        principal = _require_interactive_principal(request)
        service = _person_service(request)
        await service.authorize_link_proposal(
            proposer=principal, account_id=account_id
        )
        return await service.propose_link(
            person_id=person_id,
            account_id=account_id,
            bound_by=principal.user_id,
            reason=body.reason if body is not None else None,
        )

    @router.delete(
        "/admin/persons/{person_id}/accounts/{account_id}",
        response_model=UnbindLinkResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def unbind_account_link(
        request: Request, person_id: str, account_id: str
    ):
        _require_person_enabled()
        principal = _require_interactive_super_admin(request)
        service = _person_service(request)
        return await service.unbind_link(
            person_id=person_id,
            account_id=account_id,
            revoked_by=principal.user_id,
        )

    @router.post(
        "/admin/persons/{person_id}:disable",
        dependencies=[Depends(combined_auth)],
    )
    async def disable_person(
        request: Request, person_id: str, body: PersonDisableRequest | None = None
    ):
        _require_person_enabled()
        _require_interactive_super_admin(request)
        service = _person_service(request)
        reason = body.reason if body is not None else None
        return await service.disable_person(person_id=person_id, reason=reason)

    @router.post(
        "/admin/persons/{person_id}:enable",
        dependencies=[Depends(combined_auth)],
    )
    async def enable_person(request: Request, person_id: str):
        _require_person_enabled()
        _require_interactive_super_admin(request)
        service = _person_service(request)
        return await service.enable_person(person_id=person_id)

    return router
