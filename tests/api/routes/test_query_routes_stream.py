from __future__ import annotations

import importlib
import json
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_query_routes = importlib.import_module("lightrag.api.routers.query_routes")
sys.argv = _original_argv

create_query_routes = _query_routes.create_query_routes

pytestmark = pytest.mark.offline


class _StreamingFakeRAG:
    async def aquery_llm(self, query: str, *, param):
        async def chunks():
            yield "first "
            yield "second"

        return {
            "llm_response": {
                "is_streaming": True,
                "response_iterator": chunks(),
            },
            "data": {
                "references": [
                    {"reference_id": "1", "file_path": "inputs/source.txt"}
                ]
            },
        }


def test_query_stream_returns_ndjson_content_type_and_lines():
    app = FastAPI()
    app.include_router(create_query_routes(_StreamingFakeRAG(), api_key="test-key"))
    client = TestClient(app)

    with client.stream(
        "POST",
        "/query/stream",
        json={"query": "streaming please", "stream": True},
        headers={"X-API-Key": "test-key"},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        body = b"".join(response.iter_bytes()).decode("utf-8")

    parsed = [json.loads(line) for line in body.split("\n") if line]
    assert parsed == [
        {"references": [{"reference_id": "1", "file_path": "inputs/source.txt"}]},
        {"response": "first "},
        {"response": "second"},
    ]
