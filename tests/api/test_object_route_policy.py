"""Phase 3.2 object route-policy tests.

These tests cover the per-KB object-route allowlist mechanism that lives in
``lightrag.api.routers.kb_document_routes._require_destructive_lifecycle`` plus
its advisory configuration in ``lightrag.api.config``.

Frozen constraints exercised here:

* The capability constant ``OBJECT_AUTHORITATIVE_LIFECYCLE_IMPLEMENTED`` stays
  ``False`` in Phase 3.2, so the allowlist branch is exercised via the
  ``_object_lifecycle_capability_enabled`` injection point rather than by
  flipping the real constant.
* Legacy local-path routes (``documents:import``/``scan``/``texts``/``urls``)
  are permanently blocked in object mode and can never be allowlisted.
"""

from __future__ import annotations

import sys

# Importing the routes module transitively constructs AuthHandler, which parses
# CLI args from sys.argv. Stash and neutralize argv for the import, matching
# the pattern in tests/api/routes/test_kb_document_routes.py.
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]

import pytest  # noqa: E402

pytestmark = pytest.mark.offline

from lightrag.api.config import (  # noqa: E402
    OBJECT_AUTHORITATIVE_LIFECYCLE_IMPLEMENTED,
    OBJECT_ROUTE_POLICY_ENV_VAR,
    OBJECT_ROUTE_POLICY_OPERATIONS,
    load_object_route_policy_from_env,
    object_route_policy_allows,
)
from lightrag.api.document_lifecycle_service import DocumentLifecycleError  # noqa: E402
from lightrag.api.routers import kb_document_routes as routes_module  # noqa: E402
from lightrag.api.routers.kb_document_routes import (  # noqa: E402
    _object_lifecycle_capability_enabled,
    _reject_legacy_route_in_object_mode,
    _require_destructive_lifecycle,
)

import fastapi  # noqa: E402

sys.argv = _original_argv


# ---------------------------------------------------------------------------
# Capability constant must remain False in Phase 3.2.
# ---------------------------------------------------------------------------


def test_capability_constant_is_true_after_phase_3_gates():
    """The capability switch was flipped to True after Phase 3 Gates 1-3 PASS."""

    assert OBJECT_AUTHORITATIVE_LIFECYCLE_IMPLEMENTED is True
    assert _object_lifecycle_capability_enabled() is True


# ---------------------------------------------------------------------------
# Minimal duck-typed stand-in for DocumentLifecycleService.
# ---------------------------------------------------------------------------


class _FakeDocService:
    """Mirrors the real admission surface used by the gate.

    The real ``DocumentLifecycleService.assert_destructive_operation_supported``
    raises ``DocumentLifecycleError`` iff ``self.object_authoritative``; this
    fake reproduces that contract without the heavy constructor dependencies.
    """

    def __init__(self, *, object_authoritative: bool) -> None:
        self._object_authoritative = object_authoritative

    @property
    def object_authoritative(self) -> bool:
        return self._object_authoritative

    def assert_destructive_operation_supported(self, operation: str) -> None:
        if self._object_authoritative:
            raise DocumentLifecycleError(
                f"{operation} is disabled in object artifact mode until Phase 3"
            )


@pytest.fixture
def local_service() -> _FakeDocService:
    return _FakeDocService(object_authoritative=False)


@pytest.fixture
def object_service() -> _FakeDocService:
    return _FakeDocService(object_authoritative=True)


@pytest.fixture
def capability_enabled(monkeypatch):
    """Flip the route-layer capability injection point (constant stays False)."""

    monkeypatch.setattr(
        routes_module,
        "_object_lifecycle_capability_enabled",
        lambda: True,
    )


@pytest.fixture
def capability_disabled(monkeypatch):
    """Ensure the route-layer capability injection reports False (real state)."""

    monkeypatch.setattr(
        routes_module,
        "_object_lifecycle_capability_enabled",
        lambda: False,
    )


@pytest.fixture
def clear_route_policy_env(monkeypatch):
    """Provide a controllable env mapping for the policy loader."""

    env: dict[str, str] = {}
    # ``load_object_route_policy_from_env`` reads ``os.environ`` by default; we
    # monkeypatch the loader so tests can drive the policy without touching the
    # real environment.
    return env


def _wire_policy(monkeypatch, policy: dict[str, set[str]] | None) -> None:
    """Inject a fixed policy result into the routes module."""

    monkeypatch.setattr(
        routes_module,
        "load_object_route_policy_from_env",
        lambda *args, **kwargs: dict(policy or {}),
    )


# ---------------------------------------------------------------------------
# Config: load_object_route_policy_from_env
# ---------------------------------------------------------------------------


class TestLoadObjectRoutePolicyFromEnv:
    def test_missing_env_returns_empty(self):
        assert load_object_route_policy_from_env({}) == {}

    def test_blank_env_returns_empty(self):
        assert (
            load_object_route_policy_from_env({OBJECT_ROUTE_POLICY_ENV_VAR: "  "}) == {}
        )

    def test_invalid_json_returns_empty(self):
        assert (
            load_object_route_policy_from_env({OBJECT_ROUTE_POLICY_ENV_VAR: "not json"})
            == {}
        )

    def test_non_object_json_returns_empty(self):
        assert (
            load_object_route_policy_from_env(
                {OBJECT_ROUTE_POLICY_ENV_VAR: '["replace", "delete"]'}
            )
            == {}
        )

    def test_valid_global_default(self):
        policy = load_object_route_policy_from_env(
            {OBJECT_ROUTE_POLICY_ENV_VAR: '{"*": ["replace", "delete"]}'}
        )
        assert policy == {"*": {"replace", "delete"}}

    def test_per_kb_entry(self):
        policy = load_object_route_policy_from_env(
            {OBJECT_ROUTE_POLICY_ENV_VAR: '{"kb_abc": ["sync"]}'}
        )
        assert policy == {"kb_abc": {"sync"}}

    def test_global_and_per_kb_union(self):
        policy = load_object_route_policy_from_env(
            {
                OBJECT_ROUTE_POLICY_ENV_VAR: (
                    '{"*": ["replace"], "kb_abc": ["sync", "delete"]}'
                )
            }
        )
        assert policy == {
            "*": {"replace"},
            "kb_abc": {"sync", "delete"},
        }

    def test_unknown_operations_dropped(self):
        policy = load_object_route_policy_from_env(
            {OBJECT_ROUTE_POLICY_ENV_VAR: '{"*": ["replace", "bogus"]}'}
        )
        assert policy == {"*": {"replace"}}

    def test_legacy_operations_dropped(self):
        # import/scan/texts/urls are legacy local-path operations and must
        # never appear in an allowlist even if the operator writes them in.
        for legacy in ("import", "scan", "texts", "urls"):
            policy = load_object_route_policy_from_env(
                {OBJECT_ROUTE_POLICY_ENV_VAR: f'{{"*": ["{legacy}"]}}'}
            )
            assert policy == {}, f"legacy op {legacy!r} must be dropped"

    def test_case_normalization(self):
        policy = load_object_route_policy_from_env(
            {OBJECT_ROUTE_POLICY_ENV_VAR: '{"*": ["REPLACE", "Sync"]}'}
        )
        assert policy == {"*": {"replace", "sync"}}

    def test_empty_entry_dropped(self):
        policy = load_object_route_policy_from_env(
            {OBJECT_ROUTE_POLICY_ENV_VAR: '{"kb_empty": [], "kb_real": ["delete"]}'}
        )
        assert policy == {"kb_real": {"delete"}}

    def test_non_string_value_entries_dropped(self):
        policy = load_object_route_policy_from_env(
            {OBJECT_ROUTE_POLICY_ENV_VAR: '{"kb_x": 123, "kb_y": ["delete"]}'}
        )
        assert policy == {"kb_y": {"delete"}}

    def test_returns_fresh_dict(self):
        # Callers must be able to mutate the result without affecting cached
        # state (advisory config).
        first = load_object_route_policy_from_env(
            {OBJECT_ROUTE_POLICY_ENV_VAR: '{"*": ["replace"]}'}
        )
        first["*"].add("delete")
        second = load_object_route_policy_from_env(
            {OBJECT_ROUTE_POLICY_ENV_VAR: '{"*": ["replace"]}'}
        )
        assert second == {"*": {"replace"}}


# ---------------------------------------------------------------------------
# Config: object_route_policy_allows
# ---------------------------------------------------------------------------


class TestObjectRoutePolicyAllows:
    def test_empty_policy_denies(self):
        assert object_route_policy_allows({}, "kb_abc", "replace") is False
        assert object_route_policy_allows(None, "kb_abc", "replace") is False

    def test_global_default_allows(self):
        policy = {"*": {"replace"}}
        assert object_route_policy_allows(policy, "kb_abc", "replace") is True
        assert object_route_policy_allows(policy, None, "replace") is True

    def test_per_kb_allows(self):
        policy = {"kb_abc": {"sync"}}
        assert object_route_policy_allows(policy, "kb_abc", "sync") is True
        assert object_route_policy_allows(policy, "kb_other", "sync") is False

    def test_global_and_per_kb_union(self):
        policy = {"*": {"replace"}, "kb_abc": {"sync"}}
        assert object_route_policy_allows(policy, "kb_abc", "replace") is True
        assert object_route_policy_allows(policy, "kb_abc", "sync") is True
        assert object_route_policy_allows(policy, "kb_abc", "delete") is False
        assert object_route_policy_allows(policy, "kb_other", "sync") is False

    def test_unknown_operation_denied(self):
        policy = {"*": {"replace"}}
        assert object_route_policy_allows(policy, "kb_abc", "bogus") is False

    def test_legacy_operations_always_denied(self):
        # Even if a legacy op somehow appears in the policy, admission must
        # independently reject it because it's not in OBJECT_ROUTE_POLICY_OPERATIONS.
        for legacy in ("import", "scan", "texts", "urls"):
            assert legacy not in OBJECT_ROUTE_POLICY_OPERATIONS
            policy = {"*": {legacy}}
            assert object_route_policy_allows(policy, "kb_abc", legacy) is False, (
                f"legacy op {legacy!r} must never be allowed"
            )


# ---------------------------------------------------------------------------
# _require_destructive_lifecycle
# ---------------------------------------------------------------------------


class TestRequireDestructiveLifecycle:
    def test_local_mode_proceeds_regardless_of_allowlist(
        self, local_service, capability_enabled, monkeypatch
    ):
        # Empty allowlist + local mode → must NOT raise.
        _wire_policy(monkeypatch, None)
        _require_destructive_lifecycle(
            local_service, "Document replace", kb_id="kb_abc", route_operation="replace"
        )

    def test_local_mode_ignores_capability(
        self, local_service, capability_enabled, monkeypatch
    ):
        _wire_policy(monkeypatch, None)
        # Capability True but local mode → proceeds.
        _require_destructive_lifecycle(
            local_service, "Document delete", kb_id="kb_abc", route_operation="delete"
        )

    def test_object_mode_capability_false_preserves_503(
        self, object_service, capability_disabled, monkeypatch
    ):
        # Even with an allowlist that includes the op, capability False must
        # preserve the closed-gate 503 behavior.
        _wire_policy(monkeypatch, {"*": {"replace"}})
        with pytest.raises(fastapi.HTTPException) as exc_info:
            _require_destructive_lifecycle(
                object_service,
                "Document replace",
                kb_id="kb_abc",
                route_operation="replace",
            )
        assert exc_info.value.status_code == 503

    def test_object_mode_capability_false_all_destructive_routes_503(
        self, object_service, capability_disabled, monkeypatch
    ):
        # Regression: every destructive route stays 503 while the capability
        # gate is closed, regardless of allowlist contents.
        _wire_policy(monkeypatch, {"*": {"replace", "delete", "sync", "batch_delete"}})
        for op, route in [
            ("Document sync", "sync"),
            ("Document replace", "replace"),
            ("Document delete", "delete"),
            ("Batch document delete", "batch_delete"),
        ]:
            with pytest.raises(fastapi.HTTPException) as exc_info:
                _require_destructive_lifecycle(
                    object_service, op, kb_id="kb_abc", route_operation=route
                )
            assert exc_info.value.status_code == 503, op

    def test_object_mode_capability_true_allowlist_includes_op_proceeds(
        self, object_service, capability_enabled, monkeypatch
    ):
        _wire_policy(monkeypatch, {"kb_abc": {"replace"}})
        # Must NOT raise.
        _require_destructive_lifecycle(
            object_service,
            "Document replace",
            kb_id="kb_abc",
            route_operation="replace",
        )

    def test_object_mode_capability_true_allowlist_missing_op_returns_403(
        self, object_service, capability_enabled, monkeypatch
    ):
        _wire_policy(monkeypatch, {"kb_abc": {"sync"}})
        with pytest.raises(fastapi.HTTPException) as exc_info:
            _require_destructive_lifecycle(
                object_service,
                "Document replace",
                kb_id="kb_abc",
                route_operation="replace",
            )
        assert exc_info.value.status_code == 403
        assert "not enabled in object mode" in str(exc_info.value.detail)

    def test_object_mode_capability_true_empty_allowlist_returns_403(
        self, object_service, capability_enabled, monkeypatch
    ):
        _wire_policy(monkeypatch, None)
        with pytest.raises(fastapi.HTTPException) as exc_info:
            _require_destructive_lifecycle(
                object_service,
                "Document replace",
                kb_id="kb_abc",
                route_operation="replace",
            )
        assert exc_info.value.status_code == 403

    def test_object_mode_capability_true_no_route_operation_returns_403(
        self, object_service, capability_enabled, monkeypatch
    ):
        # A route with no allowlist token is not allowlistable → 403.
        _wire_policy(monkeypatch, {"*": {"replace"}})
        with pytest.raises(fastapi.HTTPException) as exc_info:
            _require_destructive_lifecycle(
                object_service, "Document replace", kb_id="kb_abc", route_operation=None
            )
        assert exc_info.value.status_code == 403

    def test_object_mode_capability_true_global_default_allows(
        self, object_service, capability_enabled, monkeypatch
    ):
        _wire_policy(monkeypatch, {"*": {"delete"}})
        # Any KB covered by the global default.
        _require_destructive_lifecycle(
            object_service,
            "Document delete",
            kb_id="kb_anything",
            route_operation="delete",
        )

    def test_object_mode_capability_true_per_kb_granularity(
        self, object_service, capability_enabled, monkeypatch
    ):
        _wire_policy(monkeypatch, {"kb_allowed": {"sync"}, "kb_other": {"replace"}})
        # kb_allowed may sync.
        _require_destructive_lifecycle(
            object_service, "Document sync", kb_id="kb_allowed", route_operation="sync"
        )
        # kb_allowed may NOT replace.
        with pytest.raises(fastapi.HTTPException) as exc_info:
            _require_destructive_lifecycle(
                object_service,
                "Document replace",
                kb_id="kb_allowed",
                route_operation="replace",
            )
        assert exc_info.value.status_code == 403
        # kb_other may replace but NOT sync.
        _require_destructive_lifecycle(
            object_service,
            "Document replace",
            kb_id="kb_other",
            route_operation="replace",
        )
        with pytest.raises(fastapi.HTTPException) as exc_info:
            _require_destructive_lifecycle(
                object_service,
                "Document sync",
                kb_id="kb_other",
                route_operation="sync",
            )
        assert exc_info.value.status_code == 403

    def test_legacy_route_operation_token_never_allowed(
        self, object_service, capability_enabled, monkeypatch
    ):
        # Legacy routes do not pass a route_operation token at all (they use
        # the dedicated legacy guard), but even if a caller attempted one of
        # the legacy-named tokens the allowlist helper would reject it. Here
        # we verify the gate itself treats an explicit legacy token as
        # un-allowlistable.
        _wire_policy(monkeypatch, {"*": {"import", "scan", "texts", "urls"}})
        for legacy in ("import", "scan", "texts", "urls"):
            with pytest.raises(fastapi.HTTPException) as exc_info:
                _require_destructive_lifecycle(
                    object_service,
                    f"Document {legacy}",
                    kb_id="kb_abc",
                    route_operation=legacy,
                )
            assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# _reject_legacy_route_in_object_mode
# ---------------------------------------------------------------------------


class TestRejectLegacyRouteInObjectMode:
    def test_local_mode_proceeds(self, local_service, capability_enabled):
        # Must NOT raise even with capability enabled.
        _reject_legacy_route_in_object_mode(local_service, "documents:import")

    def test_object_mode_capability_false_returns_503(
        self, object_service, capability_disabled
    ):
        with pytest.raises(fastapi.HTTPException) as exc_info:
            _reject_legacy_route_in_object_mode(object_service, "documents:import")
        assert exc_info.value.status_code == 503
        assert "legacy local-path route" in str(exc_info.value.detail)

    def test_object_mode_capability_true_returns_403(
        self, object_service, capability_enabled
    ):
        with pytest.raises(fastapi.HTTPException) as exc_info:
            _reject_legacy_route_in_object_mode(object_service, "documents:scan")
        assert exc_info.value.status_code == 403
        assert "permanently disabled" in str(exc_info.value.detail)

    @pytest.mark.parametrize(
        "route_label",
        ["documents:import", "documents:scan", "documents:texts", "documents:urls"],
    )
    def test_all_legacy_routes_blocked_in_object_mode(
        self, object_service, capability_disabled, route_label
    ):
        with pytest.raises(fastapi.HTTPException) as exc_info:
            _reject_legacy_route_in_object_mode(object_service, route_label)
        assert exc_info.value.status_code == 503

    def test_legacy_routes_blocked_even_if_allowlisted(
        self, object_service, capability_enabled, monkeypatch
    ):
        # Even if someone put legacy ops in the policy AND the capability is
        # flipped, the dedicated legacy guard still hard-rejects. Legacy routes
        # have no allowlist escape hatch.
        _wire_policy(monkeypatch, {"*": {"import", "scan", "texts", "urls"}})
        with pytest.raises(fastapi.HTTPException) as exc_info:
            _reject_legacy_route_in_object_mode(object_service, "documents:import")
        assert exc_info.value.status_code == 403
