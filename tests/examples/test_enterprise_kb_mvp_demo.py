from __future__ import annotations

import argparse

import httpx
import pytest

from examples.enterprise_kb_mvp.enterprise_kb_mvp_demo import (
    EnterpriseKBClient,
    _hard_reset_demo_kbs,
    confirm_reset_kb,
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
