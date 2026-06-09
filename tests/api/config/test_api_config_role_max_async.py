import sys

import pytest

from lightrag.api.config import parse_args


pytestmark = pytest.mark.offline


ROLE_MAX_ASYNC_ENV_KEYS = (
    "MAX_ASYNC",
    "EXTRACT_MAX_ASYNC_LLM",
    "KEYWORD_MAX_ASYNC_LLM",
    "QUERY_MAX_ASYNC_LLM",
    "VLM_MAX_ASYNC_LLM",
)

ENTERPRISE_LIMIT_ENV_KEYS = (
    "LIGHTRAG_ENTERPRISE_RATE_LIMIT_ENABLED",
    "LIGHTRAG_ENTERPRISE_RATE_LIMIT_REQUESTS",
    "LIGHTRAG_ENTERPRISE_RATE_LIMIT_WINDOW_SECONDS",
    "LIGHTRAG_ENTERPRISE_TENANT_RATE_LIMIT_REQUESTS",
    "LIGHTRAG_ENTERPRISE_TENANT_RATE_LIMIT_WINDOW_SECONDS",
    "LIGHTRAG_ENTERPRISE_QUOTA_REQUESTS",
    "LIGHTRAG_ENTERPRISE_QUOTA_WINDOW_SECONDS",
    "LIGHTRAG_ENTERPRISE_TENANT_QUOTA_REQUESTS",
    "LIGHTRAG_ENTERPRISE_TENANT_QUOTA_WINDOW_SECONDS",
    "LIGHTRAG_ENTERPRISE_ARTIFACT_DOWNLOAD_MIN_ROLE",
    "LIGHTRAG_ENTERPRISE_ARTIFACT_DOWNLOAD_POLICY",
    "LIGHTRAG_ENTERPRISE_ARTIFACT_ACTION_POLICY",
    "LIGHTRAG_ENTERPRISE_MASK_STORAGE_URIS",
    "LIGHTRAG_ENTERPRISE_REGISTRATION_MAX_ATTEMPTS",
    "LIGHTRAG_ENTERPRISE_REGISTRATION_WINDOW_SECONDS",
    "LIGHTRAG_ENTERPRISE_REGISTRATION_LOCKOUT_SECONDS",
)


def _clear_max_async_env(monkeypatch):
    for key in ROLE_MAX_ASYNC_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _clear_enterprise_limit_env(monkeypatch):
    for key in ENTERPRISE_LIMIT_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_role_max_async_defaults_none_when_env_unset(monkeypatch):
    _clear_max_async_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("MAX_ASYNC", "10")

    args = parse_args()

    assert args.max_async == 10
    assert args.extract_llm_max_async is None
    assert args.keyword_llm_max_async is None
    assert args.query_llm_max_async is None
    assert args.vlm_llm_max_async is None


def test_role_max_async_env_override_keeps_other_roles_none(monkeypatch):
    _clear_max_async_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("MAX_ASYNC", "10")
    monkeypatch.setenv("EXTRACT_MAX_ASYNC_LLM", "7")

    args = parse_args()

    assert args.max_async == 10
    assert args.extract_llm_max_async == 7
    assert args.keyword_llm_max_async is None
    assert args.query_llm_max_async is None
    assert args.vlm_llm_max_async is None


def test_role_max_async_literal_none_string_is_preserved(monkeypatch):
    _clear_max_async_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("MAX_ASYNC", "10")
    monkeypatch.setenv("QUERY_MAX_ASYNC_LLM", "None")

    args = parse_args()

    assert args.max_async == 10
    assert args.query_llm_max_async is None


def test_enterprise_limit_config_defaults_disabled(monkeypatch):
    _clear_enterprise_limit_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])

    args = parse_args()

    assert args.enterprise_rate_limit_enabled is False
    assert args.enterprise_rate_limit_requests == 60
    assert args.enterprise_rate_limit_window_seconds == 60.0
    assert args.enterprise_tenant_rate_limit_requests == 0
    assert args.enterprise_quota_requests == 0
    assert args.enterprise_tenant_quota_requests == 0
    assert args.enterprise_artifact_download_min_role == "kb_viewer"
    assert args.enterprise_artifact_download_policy == ""
    assert args.enterprise_artifact_action_policy == ""
    assert args.enterprise_mask_storage_uris is True
    assert args.enterprise_registration_max_attempts == 10
    assert args.enterprise_registration_window_seconds == 300.0
    assert args.enterprise_registration_lockout_seconds == 900.0


def test_enterprise_limit_config_reads_env_overrides(monkeypatch):
    _clear_enterprise_limit_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("LIGHTRAG_ENTERPRISE_RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("LIGHTRAG_ENTERPRISE_RATE_LIMIT_REQUESTS", "7")
    monkeypatch.setenv("LIGHTRAG_ENTERPRISE_RATE_LIMIT_WINDOW_SECONDS", "12.5")
    monkeypatch.setenv("LIGHTRAG_ENTERPRISE_TENANT_RATE_LIMIT_REQUESTS", "19")
    monkeypatch.setenv("LIGHTRAG_ENTERPRISE_TENANT_RATE_LIMIT_WINDOW_SECONDS", "30.5")
    monkeypatch.setenv("LIGHTRAG_ENTERPRISE_QUOTA_REQUESTS", "101")
    monkeypatch.setenv("LIGHTRAG_ENTERPRISE_QUOTA_WINDOW_SECONDS", "3600")
    monkeypatch.setenv("LIGHTRAG_ENTERPRISE_TENANT_QUOTA_REQUESTS", "303")
    monkeypatch.setenv("LIGHTRAG_ENTERPRISE_TENANT_QUOTA_WINDOW_SECONDS", "7200")
    monkeypatch.setenv("LIGHTRAG_ENTERPRISE_ARTIFACT_DOWNLOAD_MIN_ROLE", "kb_admin")
    monkeypatch.setenv(
        "LIGHTRAG_ENTERPRISE_ARTIFACT_DOWNLOAD_POLICY",
        '{"original":"kb_editor"}',
    )
    monkeypatch.setenv(
        "LIGHTRAG_ENTERPRISE_ARTIFACT_ACTION_POLICY",
        '{"preview":{"*":"kb_editor"}}',
    )
    monkeypatch.setenv("LIGHTRAG_ENTERPRISE_MASK_STORAGE_URIS", "false")
    monkeypatch.setenv("LIGHTRAG_ENTERPRISE_REGISTRATION_MAX_ATTEMPTS", "5")
    monkeypatch.setenv("LIGHTRAG_ENTERPRISE_REGISTRATION_WINDOW_SECONDS", "15")
    monkeypatch.setenv("LIGHTRAG_ENTERPRISE_REGISTRATION_LOCKOUT_SECONDS", "30")

    args = parse_args()

    assert args.enterprise_rate_limit_enabled is True
    assert args.enterprise_rate_limit_requests == 7
    assert args.enterprise_rate_limit_window_seconds == 12.5
    assert args.enterprise_tenant_rate_limit_requests == 19
    assert args.enterprise_tenant_rate_limit_window_seconds == 30.5
    assert args.enterprise_quota_requests == 101
    assert args.enterprise_quota_window_seconds == 3600.0
    assert args.enterprise_tenant_quota_requests == 303
    assert args.enterprise_tenant_quota_window_seconds == 7200.0
    assert args.enterprise_artifact_download_min_role == "kb_admin"
    assert args.enterprise_artifact_download_policy == '{"original":"kb_editor"}'
    assert args.enterprise_artifact_action_policy == '{"preview":{"*":"kb_editor"}}'
    assert args.enterprise_mask_storage_uris is False
    assert args.enterprise_registration_max_attempts == 5
    assert args.enterprise_registration_window_seconds == 15.0
    assert args.enterprise_registration_lockout_seconds == 30.0


def test_enterprise_artifact_download_min_role_validation(monkeypatch):
    _clear_enterprise_limit_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("LIGHTRAG_ENTERPRISE_AUTH_ENABLED", "true")
    monkeypatch.setenv("TOKEN_SECRET", "test-token-secret")
    monkeypatch.setenv("LIGHTRAG_SUPER_ADMIN_USERNAME", "admin")
    monkeypatch.setenv("LIGHTRAG_SUPER_ADMIN_PASSWORD", "admin-pass")
    monkeypatch.setenv("LIGHTRAG_ENTERPRISE_ARTIFACT_DOWNLOAD_MIN_ROLE", "viewer")

    with pytest.raises(ValueError, match="ARTIFACT_DOWNLOAD_MIN_ROLE"):
        parse_args()


def test_enterprise_artifact_download_policy_validation(monkeypatch):
    _clear_enterprise_limit_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("LIGHTRAG_ENTERPRISE_AUTH_ENABLED", "true")
    monkeypatch.setenv("TOKEN_SECRET", "test-token-secret")
    monkeypatch.setenv("LIGHTRAG_SUPER_ADMIN_USERNAME", "admin")
    monkeypatch.setenv("LIGHTRAG_SUPER_ADMIN_PASSWORD", "admin-pass")
    monkeypatch.setenv(
        "LIGHTRAG_ENTERPRISE_ARTIFACT_DOWNLOAD_POLICY",
        '{"original":"viewer"}',
    )

    with pytest.raises(ValueError, match="ARTIFACT_DOWNLOAD_POLICY"):
        parse_args()


def test_enterprise_artifact_download_policy_rejects_invalid_json(monkeypatch):
    _clear_enterprise_limit_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("LIGHTRAG_ENTERPRISE_AUTH_ENABLED", "true")
    monkeypatch.setenv("TOKEN_SECRET", "test-token-secret")
    monkeypatch.setenv("LIGHTRAG_SUPER_ADMIN_USERNAME", "admin")
    monkeypatch.setenv("LIGHTRAG_SUPER_ADMIN_PASSWORD", "admin-pass")
    monkeypatch.setenv("LIGHTRAG_ENTERPRISE_ARTIFACT_DOWNLOAD_POLICY", "not-json")

    with pytest.raises(ValueError, match="ARTIFACT_DOWNLOAD_POLICY"):
        parse_args()


def test_enterprise_artifact_action_policy_validation(monkeypatch):
    _clear_enterprise_limit_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("LIGHTRAG_ENTERPRISE_AUTH_ENABLED", "true")
    monkeypatch.setenv("TOKEN_SECRET", "test-token-secret")
    monkeypatch.setenv("LIGHTRAG_SUPER_ADMIN_USERNAME", "admin")
    monkeypatch.setenv("LIGHTRAG_SUPER_ADMIN_PASSWORD", "admin-pass")
    monkeypatch.setenv(
        "LIGHTRAG_ENTERPRISE_ARTIFACT_ACTION_POLICY",
        '{"preview":{"*":"viewer"}}',
    )

    with pytest.raises(ValueError, match="ARTIFACT_ACTION_POLICY"):
        parse_args()


def test_enterprise_artifact_action_policy_rejects_invalid_action(monkeypatch):
    _clear_enterprise_limit_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("LIGHTRAG_ENTERPRISE_AUTH_ENABLED", "true")
    monkeypatch.setenv("TOKEN_SECRET", "test-token-secret")
    monkeypatch.setenv("LIGHTRAG_SUPER_ADMIN_USERNAME", "admin")
    monkeypatch.setenv("LIGHTRAG_SUPER_ADMIN_PASSWORD", "admin-pass")
    monkeypatch.setenv(
        "LIGHTRAG_ENTERPRISE_ARTIFACT_ACTION_POLICY",
        '{"delete":{"*":"kb_admin"}}',
    )

    with pytest.raises(ValueError, match="ARTIFACT_ACTION_POLICY"):
        parse_args()
