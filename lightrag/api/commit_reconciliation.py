from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar


class MetadataCommitOutcome(str, Enum):
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    UNKNOWN = "unknown"


class MetadataCommitOutcomeUnknownError(RuntimeError):
    """A metadata commit could not be proven committed or rolled back."""

    def __init__(
        self,
        operation: str,
        *,
        candidate_document_ids: Sequence[str],
        candidate_job_id: str,
        candidate_artifact_ids: Sequence[str] = (),
        candidate_artifact_types: Sequence[str] = (),
        reason: str | None = None,
    ) -> None:
        self.operation = operation
        self.candidate_document_ids = tuple(candidate_document_ids)
        self.candidate_job_id = candidate_job_id
        self.candidate_artifact_ids = tuple(candidate_artifact_ids)
        self.candidate_artifact_types = tuple(candidate_artifact_types)
        self.reason = reason
        super().__init__(
            f"Metadata commit outcome is unknown for {operation}; "
            f"candidate_job_id={candidate_job_id}, "
            f"candidate_document_ids={list(self.candidate_document_ids)}, "
            f"candidate_artifact_ids={list(self.candidate_artifact_ids)}, "
            f"candidate_artifact_types={list(self.candidate_artifact_types)}, "
            f"reason={reason or 'unknown'}"
        )


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class MetadataCommitReconciliation(Generic[T]):
    outcome: MetadataCommitOutcome
    value: T | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class CancellationSafeReconciliationResult(Generic[T]):
    value: T
    caller_cancelled: bool


def _observe_reconciliation_task(task: asyncio.Task[object]) -> None:
    def consume_result(done: asyncio.Task[object]) -> None:
        if done.cancelled():
            return
        try:
            done.result()
        except BaseException:
            pass

    task.add_done_callback(consume_result)


async def await_cancellation_safe_reconciliation(
    reconcile: Callable[[], Coroutine[object, object, T]],
    *,
    timeout: float | None = 10.0,
) -> CancellationSafeReconciliationResult[T]:
    """Run read-back independently and never cancel it with the caller task.

    Caller cancellation is remembered and returned only after the reconciliation
    task reaches a result. A timeout leaves the independent read task observable
    in the background and raises ``TimeoutError`` without cancelling it.
    """

    task = asyncio.create_task(reconcile())
    caller_cancelled = False
    loop = asyncio.get_running_loop()
    deadline = None if timeout is None else loop.time() + max(0.0, timeout)

    while True:
        remaining = None if deadline is None else deadline - loop.time()
        if remaining is not None and remaining <= 0:
            _observe_reconciliation_task(task)
            raise TimeoutError("Metadata commit reconciliation timed out")
        try:
            value = await asyncio.wait_for(
                asyncio.shield(task),
                timeout=remaining,
            )
            return CancellationSafeReconciliationResult(
                value=value,
                caller_cancelled=caller_cancelled,
            )
        except TimeoutError:
            _observe_reconciliation_task(task)
            raise
        except asyncio.CancelledError:
            if task.done():
                if task.cancelled():
                    task.result()
                caller_cancelled = True
                return CancellationSafeReconciliationResult(
                    value=task.result(),
                    caller_cancelled=True,
                )
            caller_cancelled = True
