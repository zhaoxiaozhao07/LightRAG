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
"""

import pytest


@pytest.fixture(autouse=True)
def _disable_leaked_enterprise_auth(monkeypatch):
    from lightrag.api import config as api_config

    if getattr(api_config.global_args, "enterprise_auth_enabled", False):
        monkeypatch.setattr(
            api_config.global_args,
            "enterprise_auth_enabled",
            False,
            raising=False,
        )
