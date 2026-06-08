from __future__ import annotations

# ruff: noqa: E402 - LightRAG API modules parse sys.argv at import time.

import sys

import pytest
from fastapi import HTTPException

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from lightrag.api.enterprise_auth import LoginAttemptTracker
sys.argv = _original_argv

pytestmark = pytest.mark.offline


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _tracker(clock, *, max_attempts=3, window=60.0, lockout=120.0):
    return LoginAttemptTracker(
        max_attempts=max_attempts,
        window_seconds=window,
        lockout_seconds=lockout,
        time_func=clock,
    )


def test_lockout_after_threshold_then_unlocks_after_window():
    clock = _Clock()
    tracker = _tracker(clock, max_attempts=3, window=60.0, lockout=120.0)

    assert tracker.record_failure("alice") is False
    assert tracker.record_failure("alice") is False
    tracker.check("alice")  # below threshold -> still allowed

    assert tracker.record_failure("alice") is True  # threshold reached -> locked
    with pytest.raises(HTTPException) as exc:
        tracker.check("alice")
    assert exc.value.status_code == 429
    assert exc.value.headers["Retry-After"] == "120"

    tracker.check("bob")  # a different username is unaffected

    clock.advance(119.0)
    with pytest.raises(HTTPException):
        tracker.check("alice")  # still within the lockout window

    clock.advance(2.0)
    tracker.check("alice")  # lockout window elapsed -> allowed again


def test_success_resets_failure_counter():
    clock = _Clock()
    tracker = _tracker(clock, max_attempts=3)
    tracker.record_failure("alice")
    tracker.record_failure("alice")
    tracker.record_success("alice")
    assert tracker.record_failure("alice") is False
    assert tracker.record_failure("alice") is False
    tracker.check("alice")


def test_failures_outside_window_do_not_accumulate():
    clock = _Clock()
    tracker = _tracker(clock, max_attempts=3, window=60.0)
    tracker.record_failure("alice")
    tracker.record_failure("alice")
    clock.advance(61.0)  # window elapsed -> counter resets
    assert tracker.record_failure("alice") is False
    tracker.check("alice")


def test_disabled_tracker_never_locks():
    clock = _Clock()
    tracker = _tracker(clock, max_attempts=0)
    assert tracker.enabled is False
    for _ in range(20):
        assert tracker.record_failure("alice") is False
    tracker.check("alice")  # never raises


def test_username_keyed_by_stripped_value():
    clock = _Clock()
    tracker = _tracker(clock, max_attempts=2)
    tracker.record_failure("alice")
    assert tracker.record_failure("  alice  ") is True  # same key after strip
    with pytest.raises(HTTPException):
        tracker.check("alice")
