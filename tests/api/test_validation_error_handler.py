"""Regression tests for the app-wide RequestValidationError handler.

``create_app`` registers a custom ``RequestValidationError`` handler
(``lightrag_server.py``). Its non-``/query/data`` branch must JSON-encode
``exc.errors()`` safely: under Pydantic v2 a failed validator (e.g. the
``UploadFile`` validator FastAPI uses for ``files: list[UploadFile]``) leaves the
raw ``ValueError`` object in ``error["ctx"]["error"]``. Passing that straight to
``JSONResponse`` makes the stdlib ``json.dumps`` raise
``TypeError: Object of type ValueError is not JSON serializable`` -> HTTP 500.

This reproduces a production failure: ``curl -F 'files=D:/x.pdf'`` (a string form
field rather than ``-F 'files=@D:/x.pdf'``) sends a string where an upload is
expected, tripping exactly this validation error. The fix wraps ``exc.errors()``
in ``jsonable_encoder`` (mirroring FastAPI's own default handler).

The handler is a closure inside ``create_app``; we retrieve the *real* registered
handler from ``app.exception_handlers`` and invoke it directly, so the test covers
the production code path without a live LightRAG/storage stack.
"""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi import UploadFile
from fastapi.exceptions import RequestValidationError
from pydantic import TypeAdapter, ValidationError
from starlette.requests import Request

# Env vars a developer .env may set that would steer create_app's binding
# validation; clear them and pin minimal OpenAI-compatible defaults so the app
# builds without importing optional local providers (mirrors test_path_prefixes).
_ENV_VARS_TO_ISOLATE = (
    "LLM_BINDING",
    "EMBEDDING_BINDING",
    "LLM_BINDING_HOST",
    "LLM_BINDING_API_KEY",
    "LLM_MODEL",
    "EMBEDDING_BINDING_HOST",
    "EMBEDDING_BINDING_API_KEY",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "RERANK_BINDING",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for var in _ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("LLM_BINDING_HOST", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_BINDING_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("EMBEDDING_BINDING", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING_HOST", "https://api.openai.com/v1")
    monkeypatch.setenv("EMBEDDING_BINDING_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_DIM", "1536")
    monkeypatch.setenv("RERANK_BINDING", "null")


def _validation_handler():
    """Build the app with LightRAG mocked and return the registered handler."""
    from lightrag.api.config import parse_args

    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server"]
        args = parse_args()
    finally:
        sys.argv = original_argv

    with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
        mock_rag.return_value = MagicMock()
        from lightrag.api.lightrag_server import create_app

        app = create_app(args)

    handler = app.exception_handlers[RequestValidationError]
    assert handler is not None
    return handler


def _upload_validation_error() -> RequestValidationError:
    """A RequestValidationError whose ctx holds a raw ValueError.

    Reproduces what FastAPI raises when ``files: list[UploadFile]`` receives a
    plain string (``curl -F 'files=D:/x.pdf'`` with no ``@``).
    """
    try:
        TypeAdapter(list[UploadFile]).validate_python(["D:/LightRAG/x.pdf"])
    except ValidationError as exc:
        errors = exc.errors()
    else:  # pragma: no cover - validation must fail
        pytest.fail("expected UploadFile validation to fail for a string input")

    # Sanity: the offending error really does carry a non-serializable ValueError.
    assert any(
        isinstance((e.get("ctx") or {}).get("error"), ValueError) for e in errors
    )
    return RequestValidationError(errors)


def _request(path: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [],
        }
    )


async def test_valueerror_ctx_is_serialized_not_500():
    """Non-/query/data branch returns a clean, JSON-serializable 422 (not a 500)."""
    handler = _validation_handler()
    response = await handler(
        _request("/kbs/kb_test/documents:upload"), _upload_validation_error()
    )

    assert response.status_code == 422
    # The crux: rendering the body must not raise. ``response.body`` is the
    # already-rendered bytes from JSONResponse; it must be valid JSON.
    payload = json.loads(response.body)
    assert "detail" in payload
    assert isinstance(payload["detail"], list)


async def test_query_data_branch_keeps_failure_shape():
    """The /query/data branch keeps its bespoke failure envelope and 400 status."""
    handler = _validation_handler()
    response = await handler(
        _request("/api/query/data"), _upload_validation_error()
    )

    assert response.status_code == 400
    payload = json.loads(response.body)
    assert payload["status"] == "failure"
    assert "Validation error" in payload["message"]
