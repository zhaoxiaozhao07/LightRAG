from __future__ import annotations

import argparse
import copy
import stat
import sys
from pathlib import Path

import pytest

from lightrag.api.artifact_materialization import (
    DEFAULT_MATERIALIZATION_MAX_BYTES,
    DEFAULT_MATERIALIZATION_MAX_OBJECTS,
    DEFAULT_MATERIALIZATION_STALE_TTL_SECONDS,
    MATERIALIZATION_MAX_BYTES_ENV,
    MATERIALIZATION_MAX_OBJECTS_ENV,
    MATERIALIZATION_STALE_TTL_ENV,
)
from lightrag.api.config import (
    configure_artifact_storage_args,
    normalize_artifact_storage_mode,
    validate_artifact_storage_configuration,
)
from lightrag.utils_pipeline import (
    configured_input_dir,
    get_canonical_input_root,
    resolve_canonical_input_root_candidate,
    set_canonical_input_root,
)

pytestmark = pytest.mark.offline


def _object_mode_args(input_dir: Path, **overrides) -> argparse.Namespace:
    values = {
        "artifact_storage_mode": "object",
        "kb_metadata_backend": "postgres",
        "object_storage_backend": "minio",
        "enterprise_auth_enabled": True,
        "enterprise_disable_global_routes": True,
        "input_dir": str(input_dir),
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _complete_server_args(tmp_path: Path, monkeypatch) -> argparse.Namespace:
    safe_env = {
        "AUTH_ACCOUNTS": "",
        "LIGHTRAG_ENTERPRISE_AUTH_ENABLED": "false",
        "LIGHTRAG_CHAT_MEMORY_ENABLED": "false",
        "LIGHTRAG_PERSON_AUTH_ENABLED": "false",
        "LIGHTRAG_KB_METADATA_BACKEND": "local",
        "LIGHTRAG_ARTIFACT_STORAGE_MODE": "local",
        "LIGHTRAG_OBJECT_STORAGE": "local",
        "LLM_BINDING": "openai",
        "LLM_BINDING_HOST": "https://api.openai.com/v1",
        "LLM_BINDING_API_KEY": "test-llm-key",
        "LLM_MODEL": "gpt-4o-mini",
        "EMBEDDING_BINDING": "openai",
        "EMBEDDING_BINDING_HOST": "https://api.openai.com/v1",
        "EMBEDDING_BINDING_API_KEY": "test-embedding-key",
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "EMBEDDING_DIM": "1536",
        "RERANK_BINDING": "null",
        "TOKEN_SECRET": "test-server-secret",
    }
    for name, value in safe_env.items():
        monkeypatch.setenv(name, value)

    from lightrag.api.config import parse_args

    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    args = parse_args()
    args.input_dir = str(tmp_path / "inputs")
    args.working_dir = str(tmp_path / "working")
    args.ssl = False
    return args


def test_local_mode_is_backward_compatible_default(monkeypatch):
    monkeypatch.setenv("LIGHTRAG_ARTIFACT_STORAGE_MODE", "local")
    for name in (
        MATERIALIZATION_MAX_OBJECTS_ENV,
        MATERIALIZATION_MAX_BYTES_ENV,
        MATERIALIZATION_STALE_TTL_ENV,
    ):
        monkeypatch.delenv(name, raising=False)

    args = argparse.Namespace()
    mode = configure_artifact_storage_args(args)
    validate_artifact_storage_configuration(args)

    assert mode == "local"
    assert args.artifact_storage_mode == "local"
    assert (
        args.artifact_materialization_max_objects
        == DEFAULT_MATERIALIZATION_MAX_OBJECTS
    )
    assert (
        args.artifact_materialization_max_bytes
        == DEFAULT_MATERIALIZATION_MAX_BYTES
    )
    assert (
        args.artifact_materialization_stale_ttl_seconds
        == DEFAULT_MATERIALIZATION_STALE_TTL_SECONDS
    )


@pytest.mark.parametrize("mode", ["remote", ""])
def test_invalid_artifact_storage_mode_fails_closed(mode: str):
    with pytest.raises(ValueError, match="expected local or object"):
        normalize_artifact_storage_mode(mode)


@pytest.mark.parametrize(
    ("overrides", "backend", "available", "message"),
    [
        ({"kb_metadata_backend": "local"}, "minio", True, "postgres"),
        ({}, "local", False, "S3/MinIO"),
        ({}, "minio", False, "S3/MinIO"),
        ({"enterprise_auth_enabled": False}, "minio", True, "ENTERPRISE_AUTH"),
        (
            {"enterprise_disable_global_routes": False},
            "minio",
            True,
            "DISABLE_GLOBAL_ROUTES",
        ),
    ],
)
def test_object_mode_rejects_missing_startup_dependencies(
    tmp_path: Path,
    overrides: dict,
    backend: str,
    available: bool,
    message: str,
):
    args = _object_mode_args(tmp_path / "inputs", **overrides)
    with pytest.raises(ValueError, match=message):
        validate_artifact_storage_configuration(
            args,
            object_storage_backend=backend,
            object_storage_available=available,
            canonical_input_root=tmp_path / "inputs",
        )


def test_object_mode_validates_writable_0700_scratch(tmp_path: Path):
    input_root = tmp_path / "inputs"
    args = _object_mode_args(input_root)

    validate_artifact_storage_configuration(
        args,
        object_storage_backend="minio",
        object_storage_available=True,
        canonical_input_root=input_root,
    )

    scratch = input_root / ".lightrag-scratch"
    assert scratch.is_dir()
    assert stat.S_IMODE(scratch.stat().st_mode) == 0o700


def test_object_mode_rejects_scratch_symlink(tmp_path: Path):
    input_root = tmp_path / "inputs"
    outside = tmp_path / "outside"
    input_root.mkdir()
    outside.mkdir()
    (input_root / ".lightrag-scratch").symlink_to(
        outside, target_is_directory=True
    )

    with pytest.raises(ValueError, match="writable canonical INPUT_DIR"):
        validate_artifact_storage_configuration(
            _object_mode_args(input_root),
            object_storage_backend="minio",
            object_storage_available=True,
            canonical_input_root=input_root,
        )


def test_object_mode_requires_posix_fcntl_but_local_mode_is_unaffected(
    tmp_path: Path, monkeypatch
):
    from lightrag.api import artifact_materialization

    monkeypatch.setattr(artifact_materialization, "fcntl", None)

    with pytest.raises(ValueError, match="POSIX fcntl"):
        validate_artifact_storage_configuration(
            _object_mode_args(tmp_path / "object-inputs"),
            object_storage_backend="minio",
            object_storage_available=True,
            canonical_input_root=tmp_path / "object-inputs",
        )

    local_args = argparse.Namespace(
        artifact_storage_mode="local",
        input_dir=str(tmp_path / "local-inputs"),
    )
    validate_artifact_storage_configuration(local_args)
    assert not (tmp_path / "local-inputs" / ".lightrag-scratch").exists()


def test_create_app_rejects_object_mode_after_all_prerequisites_pass(
    tmp_path: Path, monkeypatch
):
    from lightrag.api import lightrag_server

    args = _complete_server_args(tmp_path, monkeypatch)
    args.artifact_storage_mode = "object"
    args.kb_metadata_backend = "postgres"
    args.object_storage_backend = "minio"
    args.enterprise_auth_enabled = True
    args.enterprise_disable_global_routes = True
    monkeypatch.setenv("LIGHTRAG_OBJECT_STORAGE", "minio")
    monkeypatch.setenv("LIGHTRAG_OBJECT_STORAGE_BUCKET", "phase1-test-bucket")
    monkeypatch.setattr(lightrag_server, "check_frontend_build", lambda: (True, False))

    with pytest.raises(
        ValueError, match="object-authoritative document lifecycle is not implemented"
    ):
        lightrag_server.create_app(args)

    assert (Path(args.input_dir) / ".lightrag-scratch").is_dir()
    assert get_canonical_input_root() is None


def test_failed_create_app_validation_does_not_pin_root_and_retry_can_rebase(
    tmp_path: Path, monkeypatch
):
    from lightrag.api import lightrag_server

    first_args = _complete_server_args(tmp_path / "first", monkeypatch)
    first_args.ssl = True
    first_args.ssl_certfile = None
    first_args.ssl_keyfile = None
    monkeypatch.setattr(lightrag_server, "check_frontend_build", lambda: (True, False))

    first_candidate = resolve_canonical_input_root_candidate(first_args.input_dir)
    assert get_canonical_input_root() is None
    with pytest.raises(Exception, match="SSL certificate and key"):
        lightrag_server.create_app(first_args)
    assert get_canonical_input_root() is None
    assert resolve_canonical_input_root_candidate(first_args.input_dir) == first_candidate

    second_args = copy.deepcopy(first_args)
    second_args.input_dir = str(tmp_path / "second" / "inputs")
    second_args.working_dir = str(tmp_path / "second" / "working")
    second_args.ssl = False

    def stop_at_first_service(*args, **kwargs):
        raise RuntimeError("document manager construction reached")

    monkeypatch.setattr(lightrag_server, "DocumentManager", stop_at_first_service)
    with pytest.raises(RuntimeError, match="document manager construction reached"):
        lightrag_server.create_app(second_args)

    assert get_canonical_input_root() == Path(second_args.input_dir).resolve()


@pytest.mark.parametrize(
    ("env_name", "value", "message"),
    [
        (MATERIALIZATION_MAX_OBJECTS_ENV, "0", "max_objects.*positive"),
        (MATERIALIZATION_MAX_BYTES_ENV, "-1", "max_total_bytes.*positive"),
        (MATERIALIZATION_STALE_TTL_ENV, "-1", "stale_ttl_seconds.*non-negative"),
        (MATERIALIZATION_MAX_OBJECTS_ENV, "not-an-int", "must be an integer"),
    ],
)
def test_materialization_limit_env_validation(
    monkeypatch, env_name: str, value: str, message: str
):
    for name in (
        MATERIALIZATION_MAX_OBJECTS_ENV,
        MATERIALIZATION_MAX_BYTES_ENV,
        MATERIALIZATION_STALE_TTL_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(env_name, value)

    with pytest.raises(ValueError, match=message):
        configure_artifact_storage_args(argparse.Namespace())


def test_zero_stale_ttl_is_allowed_as_janitor_disable(monkeypatch):
    monkeypatch.setenv(MATERIALIZATION_STALE_TTL_ENV, "0")
    args = argparse.Namespace()

    configure_artifact_storage_args(args)

    assert args.artifact_materialization_stale_ttl_seconds == 0


def test_canonical_root_does_not_drift_with_env_or_cwd(
    tmp_path: Path, monkeypatch
):
    first_root = tmp_path / "first" / "inputs"
    second_root = tmp_path / "second" / "inputs"
    first_root.mkdir(parents=True)
    second_root.mkdir(parents=True)
    monkeypatch.setenv("INPUT_DIR", str(first_root))

    canonical = set_canonical_input_root(configured_input_dir())
    monkeypatch.setenv("INPUT_DIR", str(second_root))
    monkeypatch.chdir(tmp_path / "second")

    assert configured_input_dir() == first_root.resolve()
    assert set_canonical_input_root(first_root / ".") == canonical
    with pytest.raises(RuntimeError, match="refusing conflicting root"):
        set_canonical_input_root(second_root)
