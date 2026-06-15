"""Regression tests: file-upload fields must render as a file-picker in /docs.

FastAPI 0.136 emits OpenAPI 3.1, which describes ``UploadFile`` /
``list[UploadFile]`` parameters as ``{"type": "string", "contentMediaType":
"application/octet-stream"}``. The bundled (offline) Swagger UI only switches a
field to its file-picker widget on the OpenAPI 3.0 marker ``format: "binary"`` —
it ignores ``contentMediaType`` — so without post-processing the upload fields
show up as plain text inputs and users are forced to type a path (which can never
upload a file). ``create_app`` installs an ``app.openapi`` wrapper that restores
``format: binary`` (see ``_coerce_binary_string_schemas``). These tests pin both
the pure transform and the assembled app's published schema.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Importing the server module touches config.global_args, which runs parse_args()
# against the live sys.argv. Under pytest that argv carries pytest's own flags and
# argparse would exit(2). Neutralize argv just for this import (mirrors the
# argv-guard idiom in tests/api/conftest.py).
_saved_argv = sys.argv
sys.argv = [_saved_argv[0]]
try:
    from lightrag.api.lightrag_server import _coerce_binary_string_schemas
finally:
    sys.argv = _saved_argv


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


def test_coerce_converts_3_1_binary_to_format_binary():
    """contentMediaType / contentEncoding binary strings become format: binary."""
    schema = {
        "properties": {
            "files": {
                "type": "array",
                "items": {
                    "type": "string",
                    "contentMediaType": "application/octet-stream",
                },
            },
            "single": {"type": "string", "contentMediaType": "application/octet-stream"},
            "encoded": {"type": "string", "contentEncoding": "base64"},
            "name": {"type": "string"},  # ordinary string field, must be untouched
            "count": {"type": "integer"},
        }
    }
    _coerce_binary_string_schemas(schema)

    props = schema["properties"]
    assert props["files"]["items"] == {"type": "string", "format": "binary"}
    assert props["single"] == {"type": "string", "format": "binary"}
    assert props["encoded"] == {"type": "string", "format": "binary"}
    # Non-binary fields are left exactly as-is.
    assert props["name"] == {"type": "string"}
    assert props["count"] == {"type": "integer"}


def _build_app():
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

        return create_app(args)


def _upload_body_schemas(spec):
    """All request-body component schemas that carry a 'files' property."""
    bodies = []
    for name, comp in spec.get("components", {}).get("schemas", {}).items():
        if "files" in comp.get("properties", {}):
            bodies.append((name, comp["properties"]["files"]))
    return bodies


def test_published_openapi_marks_upload_fields_as_binary():
    """The assembled app's /openapi.json renders upload fields as a file-picker."""
    app = _build_app()
    spec = app.openapi()

    bodies = _upload_body_schemas(spec)
    assert bodies, "expected at least one file-upload request body in the schema"

    for name, files_schema in bodies:
        # list[UploadFile] -> array of binary strings; UploadFile -> binary string.
        leaf = files_schema.get("items", files_schema)
        assert leaf.get("format") == "binary", f"{name}: {leaf} missing format:binary"
        assert "contentMediaType" not in leaf, f"{name}: 3.1 contentMediaType leaked"


def test_openapi_schema_is_cached_after_first_build():
    """The wrapper must honour FastAPI's openapi_schema cache (walk runs once)."""
    app = _build_app()
    first = app.openapi()
    second = app.openapi()
    assert first is second
