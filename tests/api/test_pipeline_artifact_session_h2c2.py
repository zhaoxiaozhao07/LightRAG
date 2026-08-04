from __future__ import annotations

import shutil
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from lightrag.artifact_runtime import (
    PipelineArtifactBinding,
    PipelineArtifactCommitOutcome,
    PipelineArtifactFinalizationResult,
    PipelineTerminalOutcome,
)
from lightrag.api.pipeline_artifact_coordinator import (
    CoordinatedPipelineArtifactSession,
)


pytestmark = pytest.mark.offline


class _FakeDocumentLifecycleService:
    def __init__(self) -> None:
        self.releases: list[
            tuple[PipelineArtifactBinding, PipelineTerminalOutcome]
        ] = []

    async def release_pipeline_artifact_attempt_if_owned(
        self,
        binding: PipelineArtifactBinding,
        outcome: PipelineTerminalOutcome,
    ) -> None:
        self.releases.append((binding, outcome))


class _FakeLease:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.mkdir(parents=True)
        self.cleanup_deferred = False
        self.cleanup_calls = 0

    def defer_cleanup(self) -> None:
        self.cleanup_deferred = True

    def cleanup(self) -> None:
        self.cleanup_calls += 1
        if self.path.exists():
            shutil.rmtree(self.path)


def _claimed_binding() -> PipelineArtifactBinding:
    return PipelineArtifactBinding(
        version=1,
        authority="kb_metadata",
        state="claimed",
        operation="parse",
        kb_id="kb-h2c2",
        kb_generation="generation-h2c2",
        workspace="workspace-h2c2",
        document_id="document-h2c2",
        lightrag_doc_id="lightrag-document-h2c2",
        job_id="job-h2c2",
        claim_token="claim-h2c2",
        source_hash="sha256:" + "a" * 64,
        parser_hash="sha256:" + "b" * 64,
        parse_generation_id="claim-h2c2",
        index_hash="sha256:" + "c" * 64,
        sidecar_artifact_id=None,
        blocks_artifact_id=None,
        expected_current_sidecar_artifact_id=None,
        expected_current_blocks_artifact_id=None,
        raw_artifact_ids=(),
    )


_Finalizer = Callable[
    [Mapping[str, Any] | None, int | None],
    Awaitable[PipelineArtifactFinalizationResult],
]


def _make_session(
    tmp_path: Path,
    finalizer: _Finalizer,
    *,
    sensitive_values: tuple[str, ...] = (),
) -> tuple[
    CoordinatedPipelineArtifactSession,
    PipelineArtifactBinding,
    _FakeDocumentLifecycleService,
    _FakeLease,
]:
    binding = _claimed_binding()
    document_service = _FakeDocumentLifecycleService()
    lease = _FakeLease(tmp_path / ".lightrag-scratch" / "lease-h2c2")
    source_path = lease.path / "source.txt"
    source_path.write_text("runtime", encoding="utf-8")
    session = CoordinatedPipelineArtifactSession(
        binding,
        document_service=document_service,
        success_finalizer=finalizer,
        lease=lease,
        source_path=source_path,
        sidecar_dir=None,
        blocks_path=None,
        sensitive_values=sensitive_values,
    )
    return session, binding, document_service, lease


async def test_committed_handoff_is_cached_and_runtime_closes_only_on_aclose(
    tmp_path: Path,
) -> None:
    binding = _claimed_binding()
    expected = PipelineArtifactFinalizationResult(
        outcome=PipelineArtifactCommitOutcome.COMMITTED,
        committed_binding=binding.committed(
            parse_generation_id=binding.claim_token,
            index_hash=binding.index_hash,
            sidecar_artifact_id=None,
            blocks_artifact_id=None,
            raw_artifact_ids=(),
        ),
        chunks_count=3,
    )
    calls: list[tuple[Mapping[str, Any] | None, int | None]] = []

    async def finalize(
        parsed_data: Mapping[str, Any] | None, chunks_count: int | None
    ) -> PipelineArtifactFinalizationResult:
        calls.append((parsed_data, chunks_count))
        return expected

    session, claimed, document_service, lease = _make_session(tmp_path, finalize)
    first_data = {"entity_count": 2}

    first = await session.handoff_success(parsed_data=first_data, chunks_count=3)
    second = await session.handoff_success(
        parsed_data={"entity_count": 999}, chunks_count=999
    )

    assert first is expected
    assert second is first
    assert calls == [(first_data, 3)]
    assert claimed is session.binding
    assert document_service.releases == []
    assert lease.cleanup_calls == 0
    assert lease.path.is_dir()

    await session.aclose()
    await session.aclose()

    assert lease.cleanup_calls == 1
    assert not lease.path.exists()


async def test_unknown_handoff_is_cached_without_committed_binding(
    tmp_path: Path,
) -> None:
    expected = PipelineArtifactFinalizationResult(
        outcome=PipelineArtifactCommitOutcome.UNKNOWN,
        reason="commit outcome is uncertain",
    )
    calls = 0

    async def finalize(
        parsed_data: Mapping[str, Any] | None, chunks_count: int | None
    ) -> PipelineArtifactFinalizationResult:
        nonlocal calls
        del parsed_data, chunks_count
        calls += 1
        return expected

    session, _binding, _document_service, _lease = _make_session(tmp_path, finalize)

    first = await session.handoff_success()
    second = await session.handoff_success(chunks_count=99)

    assert first is expected
    assert second is first
    assert first.outcome is PipelineArtifactCommitOutcome.UNKNOWN
    assert first.committed_binding is None
    assert calls == 1
    await session.aclose()


async def test_finalizer_error_is_cached_and_reraised_without_retry(
    tmp_path: Path,
) -> None:
    expected_error = RuntimeError("finalization failed")
    calls = 0

    async def finalize(
        parsed_data: Mapping[str, Any] | None, chunks_count: int | None
    ) -> PipelineArtifactFinalizationResult:
        nonlocal calls
        del parsed_data, chunks_count
        calls += 1
        raise expected_error

    session, _binding, _document_service, _lease = _make_session(tmp_path, finalize)

    with pytest.raises(RuntimeError) as first:
        await session.handoff_success()
    with pytest.raises(RuntimeError) as second:
        await session.handoff_success(chunks_count=42)

    assert first.value is expected_error
    assert second.value is expected_error
    assert calls == 1
    await session.aclose()


@pytest.mark.parametrize(
    "outcome",
    [PipelineTerminalOutcome.FAILED, PipelineTerminalOutcome.CANCELLED],
)
async def test_failed_or_cancelled_finish_releases_exact_owner_before_close(
    tmp_path: Path,
    outcome: PipelineTerminalOutcome,
) -> None:
    async def unexpected_finalizer(
        parsed_data: Mapping[str, Any] | None, chunks_count: int | None
    ) -> PipelineArtifactFinalizationResult:
        del parsed_data, chunks_count
        raise AssertionError("failure finish must not finalize success")

    session, binding, document_service, lease = _make_session(
        tmp_path, unexpected_finalizer
    )

    await session.finish(outcome)
    await session.finish(outcome)

    assert len(document_service.releases) == 1
    released_binding, released_outcome = document_service.releases[0]
    assert released_binding is binding
    assert released_outcome is outcome
    assert session.terminal_outcome is outcome
    assert lease.cleanup_calls == 0
    assert lease.path.is_dir()

    await session.aclose()

    assert lease.cleanup_calls == 1
    assert not lease.path.exists()


async def test_redact_removes_scratch_runtime_and_object_values(tmp_path: Path) -> None:
    object_uri = "s3://access:secret@example.invalid/private/object"
    runtime_token = "runtime-secret-h2c2"

    async def unexpected_finalizer(
        parsed_data: Mapping[str, Any] | None, chunks_count: int | None
    ) -> PipelineArtifactFinalizationResult:
        del parsed_data, chunks_count
        raise AssertionError("redaction does not finalize")

    session, _binding, _document_service, lease = _make_session(
        tmp_path,
        unexpected_finalizer,
        sensitive_values=(object_uri, runtime_token),
    )
    assert session.source_path is not None
    source_path = session.source_path
    error = RuntimeError(
        f"scratch={lease.path}; scratch_uri={lease.path.as_uri()}; "
        f"source={source_path}; source_uri={source_path.as_uri()}; "
        f"object={object_uri}; token={runtime_token}; "
        "remote=https://user:pass@example.invalid/runtime"
    )

    redacted = session.redact(error)

    for forbidden in (
        str(lease.path),
        lease.path.as_uri(),
        str(source_path),
        source_path.as_uri(),
        object_uri,
        runtime_token,
        ".lightrag-scratch",
        "access:secret",
        "user:pass",
    ):
        assert forbidden not in redacted
    await session.aclose()
