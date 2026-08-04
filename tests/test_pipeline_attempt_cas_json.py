import asyncio
import json
import threading
from copy import deepcopy
from pathlib import Path

import pytest

from lightrag.artifact_runtime import PipelineAttemptCommitOutcomeUnknownError
from lightrag.kg import json_kv_impl
from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage
from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data


pytestmark = pytest.mark.offline


class _DummyEmbeddingFunc:
    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


@pytest.fixture(autouse=True)
def _shared_storage_lifecycle():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


def _full_doc(token: str, marker: str) -> dict:
    return {
        "content": marker,
        "file_path": "document.txt",
        "artifact_binding": {
            "claim_token": token,
            "state": marker,
        },
    }


def _doc_status(token: str, marker: str) -> dict:
    return {
        "status": marker,
        "file_path": "document.txt",
        "metadata": {
            "pipeline_attempt_token": token,
            "marker": marker,
        },
    }


def _kv_storage(tmp_path: Path) -> JsonKVStorage:
    return JsonKVStorage(
        namespace="full_docs",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="cas",
    )


def _status_storage(tmp_path: Path) -> JsonDocStatusStorage:
    return JsonDocStatusStorage(
        namespace="doc_status",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="cas",
    )


async def _seed_full_doc(
    storage: JsonKVStorage,
    key: str,
    row: dict,
) -> None:
    await storage.upsert({key: deepcopy(row)})
    await storage.index_done_callback()


@pytest.mark.asyncio
async def test_newer_shared_full_doc_blocks_stale_cas_before_deferred_flush(
    tmp_path: Path,
):
    storage = _kv_storage(tmp_path)
    await storage.initialize()
    await _seed_full_doc(storage, "doc-1", _full_doc("old-attempt", "claimed"))
    durable_old = Path(storage._file_name).read_bytes()

    await storage.upsert(
        {"doc-1": _full_doc("new-attempt", "newer-claimed-before-flush")}
    )
    shared_newer = deepcopy(storage._data["doc-1"])
    memory_before = deepcopy(dict(storage._data))
    assert storage.storage_updated.value is True
    assert Path(storage._file_name).read_bytes() == durable_old

    assert not await storage.compare_and_commit_pipeline_attempt(
        "doc-1",
        _full_doc("old-attempt", "stale-terminal"),
        expected_attempt_token="old-attempt",
        row_kind="full_docs",
    )
    assert Path(storage._file_name).read_bytes() == durable_old
    assert dict(storage._data) == memory_before
    assert storage.storage_updated.value is True

    await storage.index_done_callback()
    assert json.loads(Path(storage._file_name).read_text())["doc-1"] == shared_newer
    assert storage._data["doc-1"] == shared_newer
    assert storage.storage_updated.value is False


@pytest.mark.asyncio
async def test_newer_shared_doc_status_blocks_stale_cas_during_auto_flush_gap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    storage = _status_storage(tmp_path)
    await storage.initialize()
    await storage.upsert(
        {"doc-1": _doc_status("old-attempt", "processing-old-attempt")}
    )
    durable_old = Path(storage._file_name).read_bytes()

    callback_entered = asyncio.Event()
    release_callback = asyncio.Event()
    original_callback = storage.index_done_callback

    async def controlled_callback() -> None:
        callback_entered.set()
        await release_callback.wait()
        await original_callback()

    monkeypatch.setattr(storage, "index_done_callback", controlled_callback)
    newer_upsert = asyncio.create_task(
        storage.upsert({"doc-1": _doc_status("new-attempt", "processing-new-attempt")})
    )
    await asyncio.wait_for(callback_entered.wait(), timeout=5)

    try:
        shared_newer = deepcopy(storage._data["doc-1"])
        memory_before = deepcopy(dict(storage._data))
        assert storage.storage_updated.value is True
        assert Path(storage._file_name).read_bytes() == durable_old

        assert not await storage.compare_and_commit_pipeline_attempt(
            "doc-1",
            _doc_status("old-attempt", "failed-old-attempt"),
            expected_attempt_token="old-attempt",
            row_kind="doc_status",
        )
        assert Path(storage._file_name).read_bytes() == durable_old
        assert dict(storage._data) == memory_before
        assert storage.storage_updated.value is True
    finally:
        release_callback.set()
        await newer_upsert

    assert json.loads(Path(storage._file_name).read_text())["doc-1"] == shared_newer
    assert storage._data["doc-1"] == shared_newer
    assert storage.storage_updated.value is False


@pytest.mark.asyncio
async def test_pending_shared_delete_blocks_cas_over_durable_old_row(tmp_path: Path):
    storage = _kv_storage(tmp_path)
    await storage.initialize()
    await _seed_full_doc(storage, "doc-1", _full_doc("attempt-a", "claimed"))
    durable_old = Path(storage._file_name).read_bytes()

    await storage.delete(["doc-1"])
    assert "doc-1" not in storage._data
    assert storage.storage_updated.value is True

    assert not await storage.compare_and_commit_pipeline_attempt(
        "doc-1",
        _full_doc("attempt-a", "stale-terminal"),
        expected_attempt_token="attempt-a",
        row_kind="full_docs",
    )
    assert Path(storage._file_name).read_bytes() == durable_old
    assert "doc-1" not in storage._data
    assert storage.storage_updated.value is True

    await storage.index_done_callback()
    assert "doc-1" not in json.loads(Path(storage._file_name).read_text())
    assert storage.storage_updated.value is False


@pytest.mark.asyncio
async def test_full_docs_cas_success_mismatch_and_missing(tmp_path: Path):
    storage = _kv_storage(tmp_path)
    await storage.initialize()
    await _seed_full_doc(storage, "doc-1", _full_doc("attempt-a", "claimed"))

    committed = _full_doc("attempt-a", "committed")
    assert await storage.compare_and_commit_pipeline_attempt(
        "doc-1",
        committed,
        expected_attempt_token="attempt-a",
        row_kind="full_docs",
    )
    assert json.loads(Path(storage._file_name).read_text())["doc-1"] == committed
    assert storage._data["doc-1"] == committed

    durable_before = Path(storage._file_name).read_bytes()
    memory_before = deepcopy(dict(storage._data))
    assert not await storage.compare_and_commit_pipeline_attempt(
        "doc-1",
        _full_doc("stale-attempt", "stale"),
        expected_attempt_token="stale-attempt",
        row_kind="full_docs",
    )
    assert Path(storage._file_name).read_bytes() == durable_before
    assert dict(storage._data) == memory_before

    assert not await storage.compare_and_commit_pipeline_attempt(
        "missing-doc",
        _full_doc("attempt-a", "missing"),
        expected_attempt_token="attempt-a",
        row_kind="full_docs",
    )
    assert Path(storage._file_name).read_bytes() == durable_before
    assert dict(storage._data) == memory_before


@pytest.mark.asyncio
async def test_doc_status_cas_success_and_mismatch_are_durable(tmp_path: Path):
    storage = _status_storage(tmp_path)
    await storage.initialize()
    await storage.upsert({"doc-1": _doc_status("attempt-a", "processing")})

    committed = _doc_status("attempt-a", "processed")
    assert await storage.compare_and_commit_pipeline_attempt(
        "doc-1",
        committed,
        expected_attempt_token="attempt-a",
        row_kind="doc_status",
    )
    assert json.loads(Path(storage._file_name).read_text())["doc-1"] == committed
    assert storage._data["doc-1"] == committed

    durable_before = Path(storage._file_name).read_bytes()
    memory_before = deepcopy(dict(storage._data))
    assert not await storage.compare_and_commit_pipeline_attempt(
        "doc-1",
        _doc_status("attempt-b", "failed"),
        expected_attempt_token="attempt-b",
        row_kind="doc_status",
    )
    assert Path(storage._file_name).read_bytes() == durable_before
    assert dict(storage._data) == memory_before


@pytest.mark.asyncio
async def test_two_process_like_instances_reread_durable_row_inside_file_lock(
    tmp_path: Path,
):
    seed = _kv_storage(tmp_path)
    await seed.initialize()
    durable_row = _full_doc("new-attempt", "claimed")
    await _seed_full_doc(seed, "doc-1", durable_row)

    stale_writer = _kv_storage(tmp_path)
    current_writer = _kv_storage(tmp_path)
    await stale_writer.initialize()
    await current_writer.initialize()

    # Detach the instances from the normal shared dict/namespace lock to model
    # independently started writers that share only the durable JSON file.
    stale_writer._data = {"doc-1": _full_doc("old-attempt", "stale-local")}
    current_writer._data = {"doc-1": deepcopy(durable_row)}
    stale_writer._storage_lock = asyncio.Lock()
    current_writer._storage_lock = asyncio.Lock()

    barrier = threading.Barrier(2)

    def commit(storage, payload, expected_token):
        barrier.wait()
        return asyncio.run(
            storage.compare_and_commit_pipeline_attempt(
                "doc-1",
                payload,
                expected_attempt_token=expected_token,
                row_kind="full_docs",
            )
        )

    stale_result, current_result = await asyncio.gather(
        asyncio.to_thread(
            commit,
            stale_writer,
            _full_doc("old-attempt", "stale-commit"),
            "old-attempt",
        ),
        asyncio.to_thread(
            commit,
            current_writer,
            _full_doc("new-attempt", "committed"),
            "new-attempt",
        ),
    )

    assert stale_result is False
    assert current_result is True
    durable = json.loads(Path(seed._file_name).read_text())
    assert durable["doc-1"] == _full_doc("new-attempt", "committed")
    assert stale_writer._data["doc-1"] == _full_doc("old-attempt", "stale-local")


@pytest.mark.asyncio
async def test_deferred_callback_cannot_overwrite_cas_and_third_instance_reopens(
    tmp_path: Path,
):
    cas_writer = _kv_storage(tmp_path)
    callback_writer = _kv_storage(tmp_path)
    await cas_writer.initialize()
    await callback_writer.initialize()
    await _seed_full_doc(cas_writer, "doc-1", _full_doc("attempt-a", "claimed"))

    await callback_writer.upsert({"doc-1": _full_doc("attempt-a", "queued-before-cas")})
    committed = _full_doc("attempt-a", "committed")
    assert await cas_writer.compare_and_commit_pipeline_attempt(
        "doc-1",
        committed,
        expected_attempt_token="attempt-a",
        row_kind="full_docs",
    )
    assert callback_writer.storage_updated.value is True
    assert callback_writer._data["doc-1"] == committed

    await callback_writer.index_done_callback()
    assert json.loads(Path(cas_writer._file_name).read_text())["doc-1"] == committed
    assert callback_writer.storage_updated.value is False

    finalize_share_data()
    initialize_share_data()
    reopened = _kv_storage(tmp_path)
    await reopened.initialize()
    reopened_row = await reopened.get_by_id("doc-1")
    assert reopened_row is not None
    assert reopened_row["content"] == "committed"
    assert reopened_row["artifact_binding"]["claim_token"] == "attempt-a"


@pytest.mark.asyncio
async def test_same_token_same_payload_is_idempotent(tmp_path: Path):
    storage = _kv_storage(tmp_path)
    await storage.initialize()
    payload = _full_doc("attempt-a", "committed")
    await _seed_full_doc(storage, "doc-1", payload)

    assert await storage.compare_and_commit_pipeline_attempt(
        "doc-1",
        payload,
        expected_attempt_token="attempt-a",
        row_kind="full_docs",
    )
    first_bytes = Path(storage._file_name).read_bytes()
    assert await storage.compare_and_commit_pipeline_attempt(
        "doc-1",
        payload,
        expected_attempt_token="attempt-a",
        row_kind="full_docs",
    )
    assert Path(storage._file_name).read_bytes() == first_bytes
    assert storage._data["doc-1"] == payload


@pytest.mark.asyncio
async def test_known_atomic_write_failure_preserves_original_error_and_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    storage = _kv_storage(tmp_path)
    await storage.initialize()
    await _seed_full_doc(storage, "doc-1", _full_doc("attempt-a", "claimed"))
    durable_before = Path(storage._file_name).read_bytes()
    memory_before = deepcopy(dict(storage._data))
    write_error = OSError("atomic replace failed before publication")

    def fail_write(_candidate, _file_name):
        raise write_error

    monkeypatch.setattr(json_kv_impl, "write_json", fail_write)
    with pytest.raises(OSError) as exc_info:
        await storage.compare_and_commit_pipeline_attempt(
            "doc-1",
            _full_doc("attempt-a", "committed"),
            expected_attempt_token="attempt-a",
            row_kind="full_docs",
        )

    assert exc_info.value is write_error
    assert Path(storage._file_name).read_bytes() == durable_before
    assert dict(storage._data) == memory_before


@pytest.mark.parametrize(
    ("readback_outcome", "expected_result", "raises_unknown"),
    [
        ("exact", True, False),
        ("different-token", False, False),
        ("same-token-different-payload", None, True),
        ("readback-error", None, True),
    ],
)
@pytest.mark.asyncio
async def test_ambiguous_json_write_reconciles_exact_durable_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    readback_outcome: str,
    expected_result: bool | None,
    raises_unknown: bool,
):
    storage = _kv_storage(tmp_path)
    await storage.initialize()
    await _seed_full_doc(storage, "doc-1", _full_doc("attempt-a", "claimed"))
    candidate = _full_doc("attempt-a", "committed")
    write_error = OSError("write acknowledgement was lost")
    readback_error = OSError("durable read-back failed")
    real_write_json = json_kv_impl.write_json
    real_load_json = json_kv_impl.load_json

    def ambiguous_write(snapshot, file_name):
        real_write_json(snapshot, file_name)
        if readback_outcome == "different-token":
            winner = deepcopy(snapshot)
            winner["doc-1"] = _full_doc("attempt-b", "winner")
            real_write_json(winner, file_name)
        elif readback_outcome == "same-token-different-payload":
            winner = deepcopy(snapshot)
            winner["doc-1"] = _full_doc("attempt-a", "other-payload")
            real_write_json(winner, file_name)
        raise write_error

    monkeypatch.setattr(json_kv_impl, "write_json", ambiguous_write)
    if readback_outcome == "readback-error":
        load_count = 0

        def fail_second_load(file_name):
            nonlocal load_count
            load_count += 1
            if load_count == 1:
                return real_load_json(file_name)
            raise readback_error

        monkeypatch.setattr(json_kv_impl, "load_json", fail_second_load)

    call = storage.compare_and_commit_pipeline_attempt(
        "doc-1",
        candidate,
        expected_attempt_token="attempt-a",
        row_kind="full_docs",
    )
    if raises_unknown:
        with pytest.raises(PipelineAttemptCommitOutcomeUnknownError) as exc_info:
            await call
        assert exc_info.value.key == "doc-1"
        assert exc_info.value.row_kind == "full_docs"
        assert exc_info.value.reason == "OSError"
        assert exc_info.value.__cause__ is (
            readback_error if readback_outcome == "readback-error" else write_error
        )
    else:
        assert await call is expected_result

    if readback_outcome == "exact":
        assert storage._data["doc-1"] == candidate
