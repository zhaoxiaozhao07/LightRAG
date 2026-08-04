from __future__ import annotations

import multiprocessing
import os
import shutil
import stat
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightrag.api.artifact_materialization import (
    ArtifactMaterializationError,
    ArtifactMaterializer,
    MaterializationLimitError,
    MaterializationLimits,
    ensure_materialization_scratch_root,
    materialization_limits_from_args,
)
from lightrag.api.object_storage import ObjectStat, ObjectStorage, ObjectStorageError
from tests.api.test_object_storage_s3 import _make_storage

pytestmark = pytest.mark.offline


class _LeaseOnlyObjectStorage(ObjectStorage):
    async def stat_object(self, object_uri: str) -> ObjectStat:
        return ObjectStat(size=0)

    def validate_document_file_uri(self, object_uri: str, **kwargs) -> None:
        return None

    def validate_document_prefix_uri(self, prefix_uri: str, **kwargs) -> None:
        return None


def _spawn_and_hold_lease(input_root: str, connection) -> None:
    try:
        materializer = ArtifactMaterializer(
            _LeaseOnlyObjectStorage(),
            input_root=input_root,
            limits=MaterializationLimits(
                max_objects=1,
                max_total_bytes=1,
                stale_ttl_seconds=1,
            ),
        )
        lease = materializer.create_lease()
        connection.send(("ready", str(lease.path)))
        connection.recv()
    except BaseException as exc:
        try:
            connection.send(("error", repr(exc)))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        connection.close()


def _limits(*, max_objects: int = 20, max_bytes: int = 4096, ttl: int = 10):
    return MaterializationLimits(
        max_objects=max_objects,
        max_total_bytes=max_bytes,
        stale_ttl_seconds=ttl,
    )


def test_materializer_rejects_storage_without_ownership_validation(tmp_path: Path):
    with pytest.raises(ArtifactMaterializationError, match="fail-closed"):
        ArtifactMaterializer(
            ObjectStorage(),
            input_root=tmp_path / "inputs",
            limits=_limits(),
        )


def test_materializer_requires_explicit_validated_limits(tmp_path: Path):
    with pytest.raises(TypeError, match="limits"):
        ArtifactMaterializer(  # type: ignore[call-arg]
            _LeaseOnlyObjectStorage(),
            input_root=tmp_path / "inputs",
        )


def test_limits_factory_uses_validated_args_not_runtime_environment(
    tmp_path: Path, monkeypatch
):
    args = SimpleNamespace(
        artifact_materialization_max_objects=7,
        artifact_materialization_max_bytes=1234,
        artifact_materialization_stale_ttl_seconds=0,
    )
    monkeypatch.setenv("LIGHTRAG_ARTIFACT_MATERIALIZATION_MAX_OBJECTS", "999")
    monkeypatch.setenv("LIGHTRAG_ARTIFACT_MATERIALIZATION_MAX_BYTES", "999999")
    monkeypatch.setenv(
        "LIGHTRAG_ARTIFACT_MATERIALIZATION_STALE_TTL_SECONDS", "999"
    )

    limits = materialization_limits_from_args(args)
    materializer = ArtifactMaterializer(
        _LeaseOnlyObjectStorage(),
        input_root=tmp_path / "inputs",
        limits=limits,
    )

    assert materializer.limits == MaterializationLimits(
        max_objects=7,
        max_total_bytes=1234,
        stale_ttl_seconds=0,
    )


def test_materializer_fails_closed_without_fcntl(tmp_path: Path, monkeypatch):
    from lightrag.api import artifact_materialization

    monkeypatch.setattr(artifact_materialization, "fcntl", None)

    with pytest.raises(ArtifactMaterializationError, match="POSIX fcntl"):
        ArtifactMaterializer(
            _LeaseOnlyObjectStorage(),
            input_root=tmp_path / "inputs",
            limits=_limits(),
        )


async def test_materialize_file_uses_0700_operation_lease_and_cleans_success(
    tmp_path: Path,
):
    storage, _, _ = _make_storage()
    await storage.initialize()
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    uri = await storage.upload_file(
        source,
        key="workspaces/ws/documents/doc/source/source.bin",
    )
    materializer = ArtifactMaterializer(
        storage,
        input_root=tmp_path / "inputs",
        limits=_limits(),
    )
    lease = materializer.create_lease()
    lease_path = lease.path

    async with lease:
        assert stat.S_IMODE(lease.path.stat().st_mode) == 0o700
        target = await lease.materialize_file(
            uri,
            workspace="ws",
            document_id="doc",
            namespace="source",
            target_name="source.bin",
        )
        assert target.parent == lease.path
        assert target.read_bytes() == b"payload"
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
        assert lease.object_count == 1
        assert lease.total_bytes == len(b"payload")

    assert not lease_path.exists()


async def test_materialize_prefix_restores_tree_without_using_key_as_target(
    tmp_path: Path,
):
    storage, _, _ = _make_storage()
    await storage.initialize()
    bundle = tmp_path / "bundle"
    (bundle / "nested").mkdir(parents=True)
    (bundle / "a.txt").write_text("a", encoding="utf-8")
    (bundle / "nested" / "b.txt").write_text("bb", encoding="utf-8")
    uri = await storage.upload_directory(
        bundle,
        prefix=(
            "workspaces/ws/documents/doc/artifacts/raw/artifact-1/bundle"
        ),
    )
    materializer = ArtifactMaterializer(
        storage,
        input_root=tmp_path / "inputs",
        limits=_limits(),
    )

    async with materializer.create_lease() as lease:
        target = await lease.materialize_prefix(
            uri,
            workspace="ws",
            document_id="doc",
            namespace="artifacts",
            artifact_id="artifact-1",
            target_name="restored",
        )
        assert target == lease.path / "restored"
        assert (target / "a.txt").read_text(encoding="utf-8") == "a"
        assert (target / "nested" / "b.txt").read_text(encoding="utf-8") == "bb"
        assert lease.object_count == 2
        assert lease.total_bytes == 3


async def test_prefix_materialization_failure_leaves_no_target_or_part_files(
    tmp_path: Path,
):
    storage, state, _ = _make_storage()
    await storage.initialize()
    prefix = "kb/workspaces/ws/documents/doc/artifacts/raw/artifact-1/bundle"
    first_key = f"{prefix}/a.bin"
    second_key = f"{prefix}/b.bin"
    state.objects[("lightrag-kb", first_key)] = b"aa"
    state.objects[("lightrag-kb", second_key)] = b"bb"
    state.download_errors[("lightrag-kb", second_key)] = RuntimeError(
        "injected second download failure"
    )
    materializer = ArtifactMaterializer(
        storage,
        input_root=tmp_path / "inputs",
        limits=_limits(),
    )

    async with materializer.create_lease() as lease:
        with pytest.raises(RuntimeError, match="second download failure"):
            await lease.materialize_prefix(
                f"s3://lightrag-kb/{prefix}/",
                workspace="ws",
                document_id="doc",
                namespace="artifacts",
                artifact_id="artifact-1",
                target_name="restored",
            )

        assert not (lease.path / "restored").exists()
        assert not any(path.name.endswith(".part") for path in lease.path.rglob("*"))
        assert [path.name for path in lease.path.iterdir()] == [".active-lease"]


async def test_prefix_late_byte_limit_failure_is_atomic_through_materializer(
    tmp_path: Path,
):
    storage, state, _ = _make_storage()
    await storage.initialize()
    prefix = "kb/workspaces/ws/documents/doc/artifacts/raw/artifact-1/bundle"
    first_key = f"{prefix}/a.bin"
    second_key = f"{prefix}/b.bin"
    state.objects[("lightrag-kb", first_key)] = b"aa"
    state.objects[("lightrag-kb", second_key)] = b"bb"
    state.listed_sizes[("lightrag-kb", first_key)] = 1
    state.listed_sizes[("lightrag-kb", second_key)] = 1
    materializer = ArtifactMaterializer(
        storage,
        input_root=tmp_path / "inputs",
        limits=_limits(max_bytes=3),
    )

    async with materializer.create_lease() as lease:
        with pytest.raises(ObjectStorageError, match="max_total_bytes"):
            await lease.materialize_prefix(
                f"s3://lightrag-kb/{prefix}/",
                workspace="ws",
                document_id="doc",
                namespace="artifacts",
                artifact_id="artifact-1",
                target_name="restored",
            )

        assert not (lease.path / "restored").exists()
        assert not any(path.name.endswith(".part") for path in lease.path.rglob("*"))
        assert [path.name for path in lease.path.iterdir()] == [".active-lease"]


async def test_materializer_rejects_untrusted_target_name(tmp_path: Path):
    storage, _, _ = _make_storage()
    materializer = ArtifactMaterializer(
        storage,
        input_root=tmp_path / "inputs",
        limits=_limits(),
    )
    uri = "s3://lightrag-kb/kb/workspaces/ws/documents/doc/source/a.bin"

    async with materializer.create_lease() as lease:
        with pytest.raises(ArtifactMaterializationError, match="one safe path segment"):
            await lease.materialize_file(
                uri,
                workspace="ws",
                document_id="doc",
                namespace="source",
                target_name="../escape.bin",
            )

    assert not (tmp_path / "escape.bin").exists()


async def test_artifact_materialization_requires_artifact_id(tmp_path: Path):
    storage, _, _ = _make_storage()
    materializer = ArtifactMaterializer(
        storage,
        input_root=tmp_path / "inputs",
        limits=_limits(),
    )
    uri = (
        "s3://lightrag-kb/kb/workspaces/ws/documents/doc/"
        "artifacts/raw/artifact-1/bundle/"
    )

    async with materializer.create_lease() as lease:
        with pytest.raises(ArtifactMaterializationError, match="requires an artifact id"):
            await lease.materialize_prefix(
                uri,
                workspace="ws",
                document_id="doc",
                namespace="artifacts",
            )


async def test_materializer_cleans_on_exception(tmp_path: Path):
    storage, _, _ = _make_storage()
    await storage.initialize()
    materializer = ArtifactMaterializer(
        storage,
        input_root=tmp_path / "inputs",
        limits=_limits(),
    )
    lease = materializer.create_lease()
    lease_path = lease.path
    missing_uri = (
        "s3://lightrag-kb/kb/workspaces/ws/documents/doc/source/missing.bin"
    )

    with pytest.raises(RuntimeError, match="missing object"):
        async with lease:
            await lease.materialize_file(
                missing_uri,
                workspace="ws",
                document_id="doc",
                namespace="source",
            )

    assert not lease_path.exists()


async def test_materializer_enforces_single_file_byte_limit(tmp_path: Path):
    storage, state, _ = _make_storage()
    await storage.initialize()
    source = tmp_path / "large.bin"
    source.write_bytes(b"12345")
    uri = await storage.upload_file(
        source,
        key="workspaces/ws/documents/doc/source/large.bin",
    )
    materializer = ArtifactMaterializer(
        storage,
        input_root=tmp_path / "inputs",
        limits=_limits(max_bytes=4),
    )

    async with materializer.create_lease() as lease:
        with pytest.raises(MaterializationLimitError, match="byte limit"):
            await lease.materialize_file(
                uri,
                workspace="ws",
                document_id="doc",
                namespace="source",
            )
        assert [path.name for path in lease.path.iterdir()] == [
            ".active-lease"
        ]
        object_key = "kb/workspaces/ws/documents/doc/source/large.bin"
        assert ("head_object", "lightrag-kb", object_key) in state.calls
        assert ("download_file", "lightrag-kb", object_key) not in state.calls


async def test_materializer_rechecks_actual_file_size_after_download(tmp_path: Path):
    storage, state, _ = _make_storage()
    await storage.initialize()
    source = tmp_path / "late-large.bin"
    source.write_bytes(b"12345")
    uri = await storage.upload_file(
        source,
        key="workspaces/ws/documents/doc/source/late-large.bin",
    )
    object_key = "kb/workspaces/ws/documents/doc/source/late-large.bin"
    state.head_object_sizes[("lightrag-kb", object_key)] = 4
    materializer = ArtifactMaterializer(
        storage,
        input_root=tmp_path / "inputs",
        limits=_limits(max_bytes=4),
    )

    async with materializer.create_lease() as lease:
        with pytest.raises(MaterializationLimitError, match="byte limit"):
            await lease.materialize_file(
                uri,
                workspace="ws",
                document_id="doc",
                namespace="source",
                target_name="source.bin",
            )
        assert [path.name for path in lease.path.iterdir()] == [
            ".active-lease"
        ]
        assert ("download_file", "lightrag-kb", object_key) in state.calls


def test_deferred_cleanup_keeps_then_allows_explicit_cleanup(tmp_path: Path):
    storage, _, _ = _make_storage()
    materializer = ArtifactMaterializer(
        storage,
        input_root=tmp_path / "inputs",
        limits=_limits(),
    )
    lease = materializer.create_lease()
    lease_path = lease.path

    with lease:
        assert lease.defer_cleanup() == lease_path

    assert lease_path.is_dir()
    assert (lease_path / ".active-lease").is_file()
    old = time.time() - 60
    os.utime(lease_path, (old, old), follow_symlinks=False)
    assert materializer.cleanup_stale_leases(now=time.time()) == 0
    lease.cleanup()
    assert not lease_path.exists()


def test_materializer_construction_runs_one_stale_cleanup(tmp_path: Path):
    input_root = tmp_path / "inputs"
    scratch_root = ensure_materialization_scratch_root(input_root)
    stale = scratch_root / ("op-" + "c" * 32)
    stale.mkdir(mode=0o700)
    old = time.time() - 60
    os.utime(stale, (old, old), follow_symlinks=False)

    ArtifactMaterializer(
        _LeaseOnlyObjectStorage(),
        input_root=input_root,
        limits=_limits(ttl=10),
    )

    assert not stale.exists()


def test_zero_ttl_disables_materializer_startup_cleanup(tmp_path: Path):
    input_root = tmp_path / "inputs"
    scratch_root = ensure_materialization_scratch_root(input_root)
    stale = scratch_root / ("op-" + "d" * 32)
    stale.mkdir(mode=0o700)
    old = time.time() - 60
    os.utime(stale, (old, old), follow_symlinks=False)

    ArtifactMaterializer(
        _LeaseOnlyObjectStorage(),
        input_root=input_root,
        limits=_limits(ttl=0),
    )

    assert stale.is_dir()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX fcntl locks")
def test_spawned_active_lease_survives_janitor_until_holder_terminates(
    tmp_path: Path,
):
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe()
    input_root = tmp_path / "inputs"
    process = context.Process(
        target=_spawn_and_hold_lease,
        args=(str(input_root), child_connection),
    )
    process.start()
    child_connection.close()
    try:
        assert parent_connection.poll(15), "spawned lease holder did not become ready"
        status, payload = parent_connection.recv()
        assert status == "ready", payload
        lease_path = Path(payload)
        assert lease_path.is_dir()
        old = time.time() - 60
        os.utime(lease_path, (old, old), follow_symlinks=False)

        # This process never registered the path in _active_lease_paths. The
        # cross-process fcntl lock is therefore the only activity proof.
        materializer = ArtifactMaterializer(
            _LeaseOnlyObjectStorage(),
            input_root=input_root,
            limits=_limits(ttl=1),
        )
        assert lease_path.is_dir()
        assert materializer.cleanup_stale_leases(now=time.time()) == 0
        assert lease_path.is_dir()

        process.terminate()
        process.join(timeout=10)
        assert not process.is_alive()

        assert materializer.cleanup_stale_leases(now=time.time()) == 1
        assert not lease_path.exists()
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        parent_connection.close()


def test_stale_janitor_removes_inactive_orphan_lease(tmp_path: Path):
    storage, _, _ = _make_storage()
    materializer = ArtifactMaterializer(
        storage,
        input_root=tmp_path / "inputs",
        limits=_limits(ttl=10),
    )
    lease_path = materializer.scratch_root / ("op-" + "b" * 32)
    lease_path.mkdir(mode=0o700)
    old = time.time() - 60
    os.utime(lease_path, (old, old), follow_symlinks=False)
    assert materializer.cleanup_stale_leases(now=time.time()) == 1
    assert not lease_path.exists()


def test_stale_janitor_never_deletes_active_lease(tmp_path: Path):
    storage, _, _ = _make_storage()
    materializer = ArtifactMaterializer(
        storage,
        input_root=tmp_path / "inputs",
        limits=_limits(ttl=1),
    )
    lease = materializer.create_lease()
    lease_path = lease.path
    old = time.time() - 60
    os.utime(lease_path, (old, old), follow_symlinks=False)

    assert materializer.cleanup_stale_leases(now=time.time()) == 0
    assert lease_path.is_dir()

    lease.cleanup()
    assert not lease_path.exists()


def test_janitor_ignores_unknown_and_symlink_entries(tmp_path: Path):
    storage, _, _ = _make_storage()
    materializer = ArtifactMaterializer(
        storage,
        input_root=tmp_path / "inputs",
        limits=_limits(ttl=1),
    )
    unknown = materializer.scratch_root / "not-a-lease"
    outside = tmp_path / "outside"
    unknown.mkdir()
    outside.mkdir()
    symlink = materializer.scratch_root / ("op-" + "a" * 32)
    symlink.symlink_to(outside, target_is_directory=True)
    old = time.time() - 60
    os.utime(unknown, (old, old), follow_symlinks=False)

    assert materializer.cleanup_stale_leases(now=time.time()) == 0
    assert unknown.is_dir()
    assert symlink.is_symlink()
    assert outside.is_dir()


def test_cleanup_rechecks_containment_before_deleting(tmp_path: Path):
    storage, _, _ = _make_storage()
    materializer = ArtifactMaterializer(
        storage,
        input_root=tmp_path / "inputs",
        limits=_limits(),
    )
    lease = materializer.create_lease()
    lease_path = lease.path
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    shutil.rmtree(lease_path)
    lease_path.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ArtifactMaterializationError, match="symlink|escapes"):
        lease.cleanup()

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_cleanup_unlinks_nested_symlink_without_following_it(tmp_path: Path):
    storage, _, _ = _make_storage()
    materializer = ArtifactMaterializer(
        storage,
        input_root=tmp_path / "inputs",
        limits=_limits(),
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    lease = materializer.create_lease()
    lease_path = lease.path
    (lease.path / "external").symlink_to(outside, target_is_directory=True)
    lease.cleanup()

    assert not lease_path.exists()
    assert sentinel.read_text(encoding="utf-8") == "keep"
