"""Logical cancellation and shutdown safety for an in-flight parse producer."""

from __future__ import annotations

import asyncio
import importlib
import sys
from typing import Any

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_kb_document_routes = importlib.import_module("lightrag.api.routers.kb_document_routes")
sys.argv = _original_argv

_execute_parse_plan = _kb_document_routes._execute_parse_plan
_run_parse_with_forced_cancel = _kb_document_routes._run_parse_with_forced_cancel

pytestmark = pytest.mark.offline


class _FakeDoc:
    id = "doc_cancel"


class _FakePlan:
    document = _FakeDoc()


class _FakeJob:
    def __init__(self, status: str):
        self.status = status


class _FakeJobService:
    """Job status oracle the forced-cancel poller reads."""

    def __init__(self, status: str = "running"):
        self._status = status
        self.get_calls = 0

    def set_status(self, status: str) -> None:
        self._status = status

    async def get_job(self, kb_id: str, job_id: str) -> _FakeJob:
        self.get_calls += 1
        return _FakeJob(self._status)


class _FakeExecution:
    def __init__(self):
        self.deferred = False
        self.cleaned = False
        self.producer_task: asyncio.Task[dict[str, Any]] | None = None

    def defer_cleanup(self) -> None:
        self.deferred = True

    def cleanup(self) -> None:
        if not self.deferred:
            self.cleaned = True


class _FakeDocumentService:
    def __init__(self, *, parse_gate: asyncio.Event):
        self._parse_gate = parse_gate
        self.run_parse_started = asyncio.Event()
        self.run_parse_cancelled = False
        self.fail_parse_calls: list[dict] = []
        self.completed = False
        self.execution = _FakeExecution()

    async def mark_parse_running(self, kb_id, document_id, *, job_id) -> None:
        return None

    async def materialize_parse_execution(self, plan):
        return self.execution

    async def run_parse(self, rag, plan, execution) -> dict[str, Any]:
        # Model a long MinerU call: block until released OR cancelled.
        self.run_parse_started.set()
        try:
            await self._parse_gate.wait()
            return {"parse_format": "lightrag", "blocks_path": "/tmp/x.jsonl"}
        except asyncio.CancelledError:
            self.run_parse_cancelled = True
            raise

    async def finalize_parse_runtime_references(
        self, rag, plan, execution, parsed_data
    ) -> None:
        return None

    async def complete_parse(
        self, kb_id, document_id, *, job_id, plan, execution, parsed_data
    ):
        self.completed = True
        raise AssertionError("complete_parse must not run after a forced cancel")

    async def commit_parse_artifact_binding(self, rag, plan, result) -> None:
        return None

    async def fail_parse(
        self, kb_id, document_id, *, job_id, plan, error_code, error_message
    ) -> None:
        self.fail_parse_calls.append(
            {"document_id": document_id, "error_code": error_code}
        )


async def test_logical_cancel_waits_for_in_flight_parse_terminal():
    parse_gate = asyncio.Event()
    job_service = _FakeJobService(status="running")
    document_service = _FakeDocumentService(parse_gate=parse_gate)

    async def _flip_to_cancelling_once_started() -> None:
        await document_service.run_parse_started.wait()
        # Parse is now blocked inside its await; request cancellation.
        job_service.set_status("cancelling")

    task = asyncio.create_task(
        _execute_parse_plan(
            document_service=document_service,  # type: ignore[arg-type]
            kb_id="kb_x",
            job_id="job_x",
            plan=_FakePlan(),
            rag=object(),
            job_service=job_service,  # type: ignore[arg-type]
        )
    )
    await document_service.run_parse_started.wait()
    job_service.set_status("cancelling")
    await asyncio.sleep(0.35)

    assert task.done() is False
    assert document_service.run_parse_cancelled is False
    assert document_service.fail_parse_calls == []
    assert document_service.execution.deferred is False
    assert document_service.execution.cleaned is False

    parse_gate.set()
    item = await asyncio.wait_for(task, timeout=5.0)

    assert item["status"] == "cancelled"
    assert item["error_code"] == "cancelled_by_user"
    assert document_service.run_parse_cancelled is False
    assert document_service.fail_parse_calls == [
        {"document_id": "doc_cancel", "error_code": "cancelled_by_user"}
    ]
    assert document_service.completed is False
    assert document_service.execution.deferred is False
    assert document_service.execution.cleaned is True


async def test_outer_task_cancellation_shields_producer_and_defers_cleanup():
    parse_gate = asyncio.Event()
    job_service = _FakeJobService(status="running")
    document_service = _FakeDocumentService(parse_gate=parse_gate)
    task = asyncio.create_task(
        _execute_parse_plan(
            document_service=document_service,  # type: ignore[arg-type]
            kb_id="kb_x",
            job_id="job_x",
            plan=_FakePlan(),
            rag=object(),
            job_service=job_service,  # type: ignore[arg-type]
        )
    )
    await document_service.run_parse_started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert document_service.run_parse_cancelled is False
    assert document_service.execution.deferred is True
    assert document_service.execution.cleaned is False
    producer_task = document_service.execution.producer_task
    assert producer_task is not None and producer_task.done() is False

    parse_gate.set()
    await asyncio.wait_for(asyncio.shield(producer_task), timeout=5.0)
    assert document_service.run_parse_cancelled is False


async def test_no_job_service_runs_plain_parse_without_polling():
    """When no job_service is wired, forced-cancel degrades to a plain await."""
    parse_gate = asyncio.Event()
    parse_gate.set()  # parse returns immediately
    document_service = _FakeDocumentService(parse_gate=parse_gate)
    execution = document_service.execution

    parsed = await _run_parse_with_forced_cancel(
        document_service=document_service,  # type: ignore[arg-type]
        job_service=None,
        kb_id="kb_x",
        job_id="job_x",
        plan=_FakePlan(),
        execution=execution,
        rag=object(),
    )
    assert parsed["parse_format"] == "lightrag"
    assert document_service.run_parse_cancelled is False


async def test_parse_completes_when_not_cancelled():
    """Happy path: parse finishes before any cancel, returns succeeded."""
    parse_gate = asyncio.Event()
    parse_gate.set()
    job_service = _FakeJobService(status="running")
    document_service = _FakeDocumentService(parse_gate=parse_gate)

    # complete_parse asserts-not-run in the fake; override for the happy path.
    class _Result:
        class document:
            id = "doc_cancel"
            parser_hash = "sha256:p"
            lightrag_doc_id = "lr"

        artifacts: list = []

    async def _complete_parse(
        kb_id, document_id, *, job_id, plan, execution, parsed_data
    ):
        return _Result()

    document_service.complete_parse = _complete_parse  # type: ignore[assignment]

    item = await _execute_parse_plan(
        document_service=document_service,  # type: ignore[arg-type]
        kb_id="kb_x",
        job_id="job_x",
        plan=_FakePlan(),
        rag=object(),
        job_service=job_service,
    )
    assert item["status"] == "succeeded"
    assert document_service.run_parse_cancelled is False
