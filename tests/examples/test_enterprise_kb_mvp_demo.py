from __future__ import annotations

import argparse
import json

import httpx
import pytest

from examples.enterprise_kb_mvp.enterprise_kb_mvp_demo import (
    EnterpriseKBClient,
    SourceFile,
    _hard_reset_demo_kbs,
    confirm_reset_kb,
    follow_job_response,
    make_delete_idempotency_key,
    normalize_run_id,
    parse_document_selection,
    prompt_for_document_selection,
    redact_value,
    run_delete_test,
)


def test_parse_document_selection_accepts_single_multi_range_and_all() -> None:
    assert parse_document_selection("1", 5) == [0]
    assert parse_document_selection("1,3-5", 5) == [0, 2, 3, 4]
    assert parse_document_selection("4-2", 5) == [1, 2, 3]
    assert parse_document_selection("1，3；5", 5) == [0, 2, 4]
    assert parse_document_selection("all", 3) == [0, 1, 2]
    assert parse_document_selection("*", 2) == [0, 1]
    assert parse_document_selection("全部", 2) == [0, 1]


def test_parse_document_selection_accepts_cancel_values() -> None:
    assert parse_document_selection("", 3) == []
    assert parse_document_selection("none", 3) == []
    assert parse_document_selection("cancel", 3) == []
    assert parse_document_selection("q", 3) == []


@pytest.mark.parametrize("value", ["0", "6", "abc", "1-a"])
def test_parse_document_selection_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_document_selection(value, 5)


def test_prompt_for_document_selection_treats_eof_as_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_eof(_prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)

    assert prompt_for_document_selection(3) == []


def test_make_delete_idempotency_key_is_stable_and_flag_sensitive() -> None:
    args = argparse.Namespace(
        kb_id="enterprise_mvp_demo",
        delete_source_file=False,
        delete_artifacts=False,
        delete_llm_cache=False,
        delete_strategy="safe",
    )
    first = make_delete_idempotency_key(args, "run-1", ["doc_a", "doc_b"])
    second = make_delete_idempotency_key(args, "run-1", ["doc_a", "doc_b"])
    args.delete_artifacts = True
    changed = make_delete_idempotency_key(args, "run-1", ["doc_a", "doc_b"])

    assert first == second
    assert first != changed
    assert first.startswith("enterprise-delete-run-1-")


def test_normalize_run_id_keeps_filename_safe_characters_only() -> None:
    assert normalize_run_id("stable-id.01") == "stable-id.01"
    assert normalize_run_id("../bad path/运行") == "bad_path"
    assert normalize_run_id("../") == "run"


def test_redact_value_masks_sensitive_keys() -> None:
    assert redact_value("OPENAI_API_KEY", "sk-secret-value") == "sk***ue"
    assert redact_value("MILVUS_TOKEN", "abcdef") == "ab***ef"
    assert redact_value("WORKSPACE", "demo") == "demo"


def _reset_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "reset_kb": "ask",
        "kb_id": "enterprise_mvp_demo",
        "isolation_kb_id": "enterprise_mvp_isolation",
        "skip_isolation_check": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_confirm_reset_kb_honors_explicit_yes_no_without_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_input(_prompt: str) -> str:
        raise AssertionError("explicit reset choice must not prompt")

    monkeypatch.setattr("builtins.input", fail_input)

    assert confirm_reset_kb(_reset_args(reset_kb="yes")) is True
    assert confirm_reset_kb(_reset_args(reset_kb="no")) is False


def test_confirm_reset_kb_skips_ask_in_non_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NonTTY:
        def isatty(self) -> bool:
            return False

    def fail_input(_prompt: str) -> str:
        raise AssertionError("non-TTY reset confirmation must not prompt")

    monkeypatch.setattr("sys.stdin", NonTTY())
    monkeypatch.setattr("builtins.input", fail_input)

    assert confirm_reset_kb(_reset_args(reset_kb="ask")) is False


def test_confirm_reset_kb_prompts_in_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TTY:
        def isatty(self) -> bool:
            return True

    prompts: list[str] = []

    def answer_yes(prompt: str) -> str:
        prompts.append(prompt)
        return "y"

    monkeypatch.setattr("sys.stdin", TTY())
    monkeypatch.setattr("builtins.input", answer_yes)

    assert confirm_reset_kb(_reset_args(reset_kb="ask")) is True
    assert prompts == ["        输入 yes / y 确认，其他键跳过："]


def test_hard_delete_kb_returns_deleted_record_or_none() -> None:
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, str(request.url)))
        if request.url.path.endswith("/missing"):
            return httpx.Response(404, json={"detail": "not found"})
        return httpx.Response(200, json={"id": "enterprise_mvp_demo"})

    client = EnterpriseKBClient("http://testserver", "", timeout=5.0)
    client._client.close()
    client._client = httpx.Client(
        base_url="http://testserver",
        transport=httpx.MockTransport(handler),
    )

    try:
        assert client.hard_delete_kb("enterprise_mvp_demo") == {
            "id": "enterprise_mvp_demo"
        }
        assert client.hard_delete_kb("missing") is None
    finally:
        client.close()

    assert requests == [
        ("DELETE", "http://testserver/kbs/enterprise_mvp_demo?hard=true"),
        ("DELETE", "http://testserver/kbs/missing?hard=true"),
    ]


def test_hard_reset_demo_kbs_records_main_and_isolation_results() -> None:
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.url.path.endswith("/enterprise_mvp_isolation"):
            return httpx.Response(404, json={"detail": "not found"})
        return httpx.Response(200, json={"id": "enterprise_mvp_demo"})

    client = EnterpriseKBClient("http://testserver", "", timeout=5.0)
    client._client.close()
    client._client = httpx.Client(
        base_url="http://testserver",
        transport=httpx.MockTransport(handler),
    )

    try:
        summary = _hard_reset_demo_kbs(
            client,
            _reset_args(
                kb_id="enterprise_mvp_demo",
                isolation_kb_id="enterprise_mvp_isolation",
                skip_isolation_check=False,
            ),
        )
    finally:
        client.close()

    assert requests == [
        ("DELETE", "/kbs/enterprise_mvp_demo"),
        ("DELETE", "/kbs/enterprise_mvp_isolation"),
    ]
    assert summary == {
        "performed": True,
        "targets": {
            "main": {
                "kb_id": "enterprise_mvp_demo",
                "state": "deleted",
                "record": {"id": "enterprise_mvp_demo"},
            },
            "isolation": {
                "kb_id": "enterprise_mvp_isolation",
                "state": "not_found",
                "record": None,
            },
        },
    }


def test_hard_reset_demo_kbs_honors_skip_isolation_check() -> None:
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        return httpx.Response(200, json={"id": "enterprise_mvp_demo"})

    client = EnterpriseKBClient("http://testserver", "", timeout=5.0)
    client._client.close()
    client._client = httpx.Client(
        base_url="http://testserver",
        transport=httpx.MockTransport(handler),
    )

    try:
        summary = _hard_reset_demo_kbs(
            client,
            _reset_args(
                kb_id="enterprise_mvp_demo",
                isolation_kb_id="enterprise_mvp_isolation",
                skip_isolation_check=True,
            ),
        )
    finally:
        client.close()

    assert requests == [("DELETE", "/kbs/enterprise_mvp_demo")]
    assert set(summary["targets"]) == {"main"}


def test_run_delete_test_records_failed_job_before_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/kbs/enterprise_mvp_demo/documents:batch-delete"
        return httpx.Response(
            200,
            json={
                "id": "job_delete_demo",
                "status": "failed",
                "error_code": "partial_delete_failed",
                "error_message": "boom",
                "completed_items": 1,
            },
        )

    answers = iter(["y", "all", "y"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    args = argparse.Namespace(
        kb_id="enterprise_mvp_demo",
        delete_source_file=False,
        delete_artifacts=False,
        delete_llm_cache=False,
        delete_strategy="safe",
        job_timeout=1.0,
    )
    documents_payload = {
        "total": 2,
        "documents": [
            {"id": "doc_a", "status": "ready", "source_name": "a.txt"},
            {"id": "doc_b", "status": "ready", "source_name": "b.txt"},
        ],
    }
    report: dict[str, object] = {"steps": {}}
    client = EnterpriseKBClient("http://testserver", "", timeout=5.0)
    client._client.close()
    client._client = httpx.Client(
        base_url="http://testserver",
        transport=httpx.MockTransport(handler),
    )

    try:
        with pytest.raises(RuntimeError, match="delete failed"):
            run_delete_test(
                client,
                args,
                documents_payload,
                "run-1",
                report,
            )
    finally:
        client.close()

    steps = report["steps"]
    assert isinstance(steps, dict)
    summary = steps["delete_test"]
    assert isinstance(summary, dict)
    job = summary["job"]
    assert isinstance(job, dict)
    final = job["final"]
    assert isinstance(final, dict)
    assert summary["failed"] is True
    assert summary["requested_count"] == 2
    assert summary["deleted_count"] == 1
    assert final["id"] == "job_delete_demo"


# --------------------------------------------------------------------------- #
# Extended-endpoint client methods (request construction via MockTransport)
# --------------------------------------------------------------------------- #


def _client_with_handler(handler) -> EnterpriseKBClient:
    """Build an EnterpriseKBClient whose transport is a MockTransport handler."""
    client = EnterpriseKBClient("http://testserver", "", timeout=5.0)
    client._client.close()
    client._client = httpx.Client(
        base_url="http://testserver",
        transport=httpx.MockTransport(handler),
    )
    return client


def test_ingest_and_batch_methods_build_expected_requests() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content) if request.content else None
        return httpx.Response(
            200, json={"job_id": "j1", "batch_id": "b1", "documents": []}
        )

    client = _client_with_handler(handler)
    try:
        client.import_texts(
            "kb1",
            [{"text": "hello", "source_name": "a.txt"}],
            parser_engine="mineru",
            process_options="iteP",
            idempotency_key="k1",
        )
        assert captured["method"] == "POST"
        assert captured["path"] == "/kbs/kb1/documents:texts"
        body = captured["body"]
        assert body["documents"][0]["text"] == "hello"
        assert body["auto_parse"] is True and body["auto_index"] is True
        assert body["parser_engine"] == "mineru"
        assert body["idempotency_key"] == "k1"

        # batch-parse uses `engine` (NOT parser_engine)
        client.batch_parse("kb1", ["d1", "d2"], engine="mineru", force_reparse=True)
        assert captured["path"] == "/kbs/kb1/documents:batch-parse"
        body = captured["body"]
        assert body["engine"] == "mineru"
        assert "parser_engine" not in body
        assert body["document_ids"] == ["d1", "d2"]
        assert body["force_reparse"] is True

        client.batch_build_kg("kb1", ["d1"], force_extract=True)
        assert captured["path"] == "/kbs/kb1/documents:batch-build-kg"
        assert captured["body"]["force_extract"] is True

        client.import_urls("kb1", [{"url": "http://x", "source_key": "k"}])
        assert captured["path"] == "/kbs/kb1/documents:urls"
        assert captured["body"]["documents"][0]["url"] == "http://x"
    finally:
        client.close()


def test_document_control_methods_build_expected_requests() -> None:
    captured: list[tuple[str, str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        captured.append((request.method, request.url.path, body))
        return httpx.Response(200, json={"id": "d1", "enabled": True, "metadata": {}})

    client = _client_with_handler(handler)
    try:
        client.disable_document("kb1", "d1")
        client.enable_document("kb1", "d1")
        client.patch_document("kb1", "d1", metadata={"x": 1}, enabled=True)
    finally:
        client.close()

    assert captured[0] == ("POST", "/kbs/kb1/documents/d1:disable", None)
    assert captured[1] == ("POST", "/kbs/kb1/documents/d1:enable", None)
    assert captured[2][0] == "PATCH"
    assert captured[2][1] == "/kbs/kb1/documents/d1"
    # Only explicitly-passed fields are sent (PATCH semantics).
    assert captured[2][2] == {"metadata": {"x": 1}, "enabled": True}


def test_reindex_rebuild_retry_methods_build_expected_requests() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content) if request.content else None
        return httpx.Response(200, json={"id": "j1", "status": "queued"})

    client = _client_with_handler(handler)
    try:
        client.reindex_document("kb1", "d1")
        assert captured["path"] == "/kbs/kb1/documents/d1:reindex"
        # reindex force_* default to True
        assert captured["body"]["force_rechunk"] is True
        assert captured["body"]["force_embedding"] is True

        client.batch_reindex("kb1", ["d1", "d2"])
        assert captured["path"] == "/kbs/kb1/documents:batch-reindex"
        assert captured["body"]["document_ids"] == ["d1", "d2"]

        client.rebuild_kb_index("kb1")
        assert captured["path"] == "/kbs/kb1:rebuild"

        client.retry_job("kb1", "j9", idempotency_key="r1")
        assert captured["path"] == "/kbs/kb1/jobs/j9:retry"
        assert captured["body"]["idempotency_key"] == "r1"
    finally:
        client.close()


def test_replace_document_sends_file_multipart_and_query_params(tmp_path) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["query"] = dict(request.url.params)
        captured["content_type"] = request.headers.get("content-type", "")
        captured["body_bytes"] = request.content
        return httpx.Response(200, json={"id": "jr", "status": "queued"})

    source_path = tmp_path / "doc.txt"
    source_path.write_text("replacement-bytes", encoding="utf-8")
    source = SourceFile(
        path=source_path,
        relative_key="enterprise-demo/replace/doc.txt",
        sha256="abc",
        size_bytes=source_path.stat().st_size,
        content_type="text/plain",
    )

    client = _client_with_handler(handler)
    try:
        client.replace_document(
            "kb1", "d1", source, parser_engine="mineru", idempotency_key="ir"
        )
    finally:
        client.close()

    assert captured["method"] == "POST"
    assert captured["path"] == "/kbs/kb1/documents/d1:replace"
    # Scalar params ride in the QUERY string, not the multipart form body.
    query = captured["query"]
    assert query["auto_parse"] == "true"
    assert query["auto_index"] == "true"
    assert query["delete_source_file"] == "true"
    assert query["delete_artifacts"] == "true"
    assert query["parser_engine"] == "mineru"
    assert query["idempotency_key"] == "ir"
    # The file rides in a multipart/form-data body under field name "file".
    assert str(captured["content_type"]).startswith("multipart/form-data")
    assert b"replacement-bytes" in captured["body_bytes"]
    assert b'name="file"' in captured["body_bytes"]


def test_query_graph_config_artifact_methods_build_expected_requests() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["query"] = dict(request.url.params)
        captured["body"] = json.loads(request.content) if request.content else None
        return httpx.Response(200, json={"status": "success", "nodes": [], "edges": []})

    client = _client_with_handler(handler)
    try:
        client.retrieve("kb1", "q", mode="mix", top_k=10, chunk_top_k=5)
        assert captured["method"] == "POST"
        assert captured["path"] == "/kbs/kb1/retrieve"
        assert captured["body"]["query"] == "q"
        assert captured["body"]["stream"] is False

        client.subgraph("kb1", label="*", max_depth=2, max_nodes=50)
        assert captured["method"] == "GET"
        assert captured["path"] == "/kbs/kb1/graph"
        assert captured["query"]["label"] == "*"
        assert captured["query"]["max_depth"] == "2"
        assert captured["query"]["max_nodes"] == "50"

        client.update_kb("kb1", description="d")
        assert captured["method"] == "PATCH"
        assert captured["path"] == "/kbs/kb1"
        assert captured["body"] == {"description": "d"}

        client.get_config_version("kb1", "v1")
        assert captured["path"] == "/kbs/kb1/configs/v1"

        client.diff_config_version("kb1", "v1")
        assert captured["method"] == "POST"
        assert captured["path"] == "/kbs/kb1/configs/v1:diff"

        client.get_artifact("kb1", "d1", "a1")
        assert captured["path"] == "/kbs/kb1/documents/d1/artifacts/a1"

        client.artifact_download_url("kb1", "d1", "a1", expires_in_seconds=120)
        assert captured["path"] == "/kbs/kb1/documents/d1/artifacts/a1:download-url"
        assert captured["query"]["expires_in_seconds"] == "120"
    finally:
        client.close()


def test_query_stream_parses_ndjson_header_and_tokens() -> None:
    lines = [
        b'{"kb_id": "kb1", "metadata": {"m": 1}, "references": [{"reference_id": "r1"}]}',
        b'{"response": "Hello"}',
        b'{"response": ", world"}',
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/kbs/kb1/query/stream"
        body = json.loads(request.content)
        assert body["stream"] is True
        return httpx.Response(200, content=b"\n".join(lines) + b"\n")

    client = _client_with_handler(handler)
    try:
        result = client.query_stream("kb1", "q", mode="mix", top_k=10, chunk_top_k=5)
    finally:
        client.close()

    assert result["response"] == "Hello, world"
    assert result["token_count"] == 2
    assert result["metadata"] == {"m": 1}
    assert result["references"] == [{"reference_id": "r1"}]
    assert "error" not in result


def test_query_stream_captures_stream_error() -> None:
    lines = [
        b'{"kb_id": "kb1", "metadata": {}}',
        b'{"error": "boom"}',
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"\n".join(lines) + b"\n")

    client = _client_with_handler(handler)
    try:
        result = client.query_stream("kb1", "q", mode="mix", top_k=1, chunk_top_k=1)
    finally:
        client.close()

    assert result.get("error") == "boom"


def test_follow_job_response_skips_http_for_terminal_and_noop() -> None:
    def fail_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("terminal/noop responses must not trigger an HTTP call")

    client = _client_with_handler(fail_handler)
    try:
        terminal = follow_job_response(
            client, "kb1", {"id": "j1", "status": "succeeded"}, 0.0
        )
        assert terminal["status"] == "succeeded"
        # Empty job_id (e.g. {kb}:rebuild no-op) returns unchanged, no polling.
        noop = follow_job_response(client, "kb1", {"job_id": "", "documents": []}, 0.0)
        assert noop["job_id"] == ""
    finally:
        client.close()


def test_follow_job_response_polls_jobresponse_and_batch_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # Both an id-based job and a job_id-based batch resolve via the wait endpoint.
        if request.url.path == "/kbs/kb1/jobs/j2:wait":
            return httpx.Response(200, json={"id": "j2", "status": "succeeded"})
        if request.url.path == "/kbs/kb1/jobs/jb:wait":
            return httpx.Response(200, json={"id": "jb", "status": "succeeded"})
        raise AssertionError(f"unexpected path {request.url.path}")

    client = _client_with_handler(handler)
    try:
        from_job = follow_job_response(
            client, "kb1", {"id": "j2", "status": "queued"}, 5.0
        )
        assert from_job["status"] == "succeeded"
        from_batch = follow_job_response(
            client, "kb1", {"job_id": "jb", "batch_id": "b", "documents": []}, 5.0
        )
        assert from_batch["status"] == "succeeded"
    finally:
        client.close()
