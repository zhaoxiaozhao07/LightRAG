from __future__ import annotations

import sys
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

pytestmark = pytest.mark.offline

_MODULES_TO_RELOAD = (
    "lightrag.api.config",
    "lightrag.api.auth",
    "lightrag.api.utils_api",
)


def _drop_module(name: str) -> None:
    sys.modules.pop(name, None)
    package_name, _, child_name = name.rpartition(".")
    package = sys.modules.get(package_name)
    if package is not None and hasattr(package, child_name):
        delattr(package, child_name)


def _fresh_client(monkeypatch, tmp_path, *, whitelist_paths: str | None = None):
    monkeypatch.chdir(tmp_path)
    for name in _MODULES_TO_RELOAD:
        _drop_module(name)
    monkeypatch.setenv("LIGHTRAG_API_KEY", "test-key")
    monkeypatch.setenv("AUTH_ACCOUNTS", "")
    monkeypatch.delenv("TOKEN_SECRET", raising=False)
    # A developer .env may leak LIGHTRAG_ENTERPRISE_AUTH_ENABLED=true into the
    # process env (load_dotenv at config import time); these tests exercise the
    # NON-enterprise whitelist behavior and re-run parse_args, which would
    # otherwise fail the enterprise TOKEN_SECRET startup validation.
    monkeypatch.delenv("LIGHTRAG_ENTERPRISE_AUTH_ENABLED", raising=False)
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("LLM_BINDING_HOST", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_BINDING_API_KEY", "llm-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("EMBEDDING_BINDING", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING_HOST", "https://api.openai.com/v1")
    monkeypatch.setenv("EMBEDDING_BINDING_API_KEY", "embedding-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_DIM", "1536")
    monkeypatch.setenv("RERANK_BINDING", "null")
    monkeypatch.setenv("WORKING_DIR", str(tmp_path / "rag_storage"))
    monkeypatch.setenv("INPUT_DIR", str(tmp_path / "inputs"))
    if whitelist_paths is None:
        monkeypatch.delenv("WHITELIST_PATHS", raising=False)
    else:
        monkeypatch.setenv("WHITELIST_PATHS", whitelist_paths)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])

    from lightrag.api.config import initialize_config, parse_args
    from lightrag.api.utils_api import get_combined_auth_dependency

    initialize_config(parse_args(), force=True)

    combined_auth = get_combined_auth_dependency(api_key="test-key")

    app = FastAPI()

    @app.get("/health", dependencies=[Depends(combined_auth)])
    async def health():
        return {"ok": True}

    @app.get("/api/tags", dependencies=[Depends(combined_auth)])
    async def api_tags():
        return {"models": []}

    return TestClient(app)


def test_api_routes_require_api_key_with_default_whitelist(monkeypatch, tmp_path):
    client = _fresh_client(monkeypatch, tmp_path)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"ok": True}

    response = client.get("/api/tags")
    assert response.status_code == 403
    assert response.json()["detail"] == "API Key required"

    authed = client.get("/api/tags", headers={"X-API-Key": "test-key"})
    assert authed.status_code == 200
    assert authed.json() == {"models": []}


def test_explicit_api_wildcard_whitelist_still_bypasses_api_key(
    monkeypatch, tmp_path
):
    client = _fresh_client(monkeypatch, tmp_path, whitelist_paths="/health,/api/*")

    response = client.get("/api/tags")
    assert response.status_code == 200
    assert response.json() == {"models": []}
