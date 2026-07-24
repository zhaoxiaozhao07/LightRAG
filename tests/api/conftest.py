"""Shared fixtures for ``tests/api`` — insulate route tests from a leaked .env.

``lightrag/api/config.py`` parses args (including ``load_dotenv(override=False)``)
at import time, so a developer's local ``.env`` that sets
``LIGHTRAG_ENTERPRISE_AUTH_ENABLED=true`` flips ``global_args.enterprise_auth_enabled``
to ``True`` for the whole test process. That makes the global ``LIGHTRAG_API_KEY``
auth path return ``403 "API key is disabled in enterprise mode"`` on every
protected route, so non-enterprise KB route tests fail wholesale through no fault
of their own (mirrors the ``_hermetic_mineru_env`` fixture in ``tests/conftest.py``).

The enterprise auth gating all funnels through ``enterprise_auth_enabled()`` which
reads ``config.global_args.enterprise_auth_enabled``. Flip just that one attribute
off by default — preserving every other real attribute (no missing-attr risk).
Enterprise tests replace ``global_args`` wholesale inside their own bodies and so
override this default for themselves.

NOTE: ``config.global_args`` is a lazy proxy whose attribute access/assignment
auto-runs ``parse_args()`` on first touch. If config has not been initialized yet
in this test session (e.g. running this directory's tests in isolation), that
parse would consume pytest's own ``sys.argv`` and crash with "unrecognized
arguments". We therefore neutralize ``sys.argv`` while touching the proxy.

A leaked ``LIGHTRAG_CHAT_MEMORY_ENABLED=true`` is nastier than the auth flag:
``parse_args()`` runs ``validate_chat_memory_configuration()``, which *raises*
when the leaked flag meets a test's hermetic env (no enterprise auth / local
metadata backend) — so the very first ``global_args`` touch (including the one
in the autouse fixture below) explodes at collection or fixture-setup time.
The module-level pin below runs when pytest imports this conftest, i.e. before
any ``tests/api`` module import can trigger the lazy parse. Setting the key
explicitly (never deleting it) also blocks every later
``load_dotenv(override=False)`` from re-filling ``true``. Chat-memory tests
that need the feature on set the key inside their own bodies, which happens
after both layers.
"""

import os
import sys

import pytest

os.environ["LIGHTRAG_CHAT_MEMORY_ENABLED"] = "false"


@pytest.fixture(autouse=True)
def _disable_leaked_enterprise_auth(monkeypatch):
    from lightrag.api import config as api_config

    # Re-pin per test (guards against a previous test setting os.environ
    # directly), and do it BEFORE touching global_args: the touch itself can
    # trigger parse_args() -> validate_chat_memory_configuration().
    monkeypatch.setenv("LIGHTRAG_CHAT_MEMORY_ENABLED", "false")

    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        if getattr(api_config.global_args, "enterprise_auth_enabled", False):
            monkeypatch.setattr(
                api_config.global_args,
                "enterprise_auth_enabled",
                False,
                raising=False,
            )
        # A chat-memory test that reloaded config with the feature on can leave
        # the parsed singleton enabled; later create_app() calls re-validate
        # against this attribute, so flip it back like the auth flag above.
        if getattr(api_config.global_args, "chat_memory_enabled", False):
            monkeypatch.setattr(
                api_config.global_args,
                "chat_memory_enabled",
                False,
                raising=False,
            )
    finally:
        sys.argv = saved_argv
