import asyncio
from contextlib import suppress
from typing import Any, cast

import pytest

from lightrag.constants import SOURCE_IDS_LIMIT_METHOD_KEEP
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.operate import _MergeStageProgress, _VDBUpsertBatcher, merge_nodes_and_edges


class FakeGraphStorage:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    async def get_node(self, node_id):
        node = self.nodes.get(node_id)
        return dict(node) if node is not None else None

    async def upsert_node(self, node_id, node_data):
        self.nodes[node_id] = dict(node_data)

    async def has_edge(self, src_id, tgt_id):
        return tuple(sorted((src_id, tgt_id))) in self.edges

    async def get_edge(self, src_id, tgt_id):
        edge = self.edges.get(tuple(sorted((src_id, tgt_id))))
        return dict(edge) if edge is not None else None

    async def upsert_edge(self, src_id, tgt_id, edge_data):
        self.edges[tuple(sorted((src_id, tgt_id)))] = dict(edge_data)


class CaptureVectorStorage:
    def __init__(self):
        self.upserts = []
        self.deletes = []

    async def upsert(self, data):
        self.upserts.append(dict(data))

    async def delete(self, ids):
        self.deletes.append(list(ids))


class FailingVectorStorage(CaptureVectorStorage):
    async def upsert(self, data):
        await super().upsert(data)
        raise RuntimeError("upsert boom")


class BlockingStatusLock:
    def __init__(self):
        self.block_next = False
        self.blocked = asyncio.Event()
        self.release = asyncio.Event()

    async def __aenter__(self):
        if self.block_next:
            self.block_next = False
            self.blocked.set()
            await self.release.wait()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _global_config(workspace: str) -> dict:
    return {
        "workspace": workspace,
        "llm_model_max_async": 4,
        "embedding_batch_num": 2,
        "default_embedding_timeout": 120,
        "source_ids_limit_method": SOURCE_IDS_LIMIT_METHOD_KEEP,
        "max_source_ids_per_entity": 10,
        "max_source_ids_per_relation": 10,
        "max_file_paths": 10,
    }


def _pipeline_status() -> dict:
    return {"latest_message": "", "history_messages": []}


def _entity(name: str) -> dict:
    return {
        "entity_name": name,
        "entity_type": "CONCEPT",
        "description": f"{name} description",
        "source_id": f"chunk-{name}",
        "file_path": "doc.md",
        "timestamp": 1,
    }


def _relation(src: str, tgt: str) -> dict:
    return {
        "src_id": src,
        "tgt_id": tgt,
        "description": f"{src} relates to {tgt}",
        "keywords": "related",
        "weight": 1.0,
        "source_id": f"chunk-{src}-{tgt}",
        "file_path": "doc.md",
        "timestamp": 1,
    }


@pytest.mark.asyncio
async def test_merge_nodes_and_edges_batches_phase1_entity_vdb_upserts():
    initialize_share_data()
    graph = FakeGraphStorage()
    entities_vdb = CaptureVectorStorage()
    relationships_vdb = CaptureVectorStorage()
    nodes = {name: [_entity(name)] for name in ("A", "B", "C")}
    pipeline_status = _pipeline_status()

    await merge_nodes_and_edges(
        [(nodes, {})],
        cast(Any, graph),
        cast(Any, entities_vdb),
        cast(Any, relationships_vdb),
        _global_config("test_entity_vdb_batching"),
        doc_id="doc-1",
        pipeline_status=pipeline_status,
        pipeline_status_lock=asyncio.Lock(),
    )

    assert [len(payload) for payload in entities_vdb.upserts] == [2, 1]
    entity_names = {
        item["entity_name"]
        for payload in entities_vdb.upserts
        for item in payload.values()
    }
    assert entity_names == {"A", "B", "C"}
    assert relationships_vdb.upserts == []
    assert pipeline_status["merge_stage_timings"]["phase1_entities"]["total"] == 3
    assert pipeline_status["merge_stage_timings"]["phase2_relations"]["skipped"] is True
    assert "timings:" in pipeline_status["latest_message"]


@pytest.mark.asyncio
async def test_merge_nodes_and_edges_batches_relationship_vdb_upserts_and_deletes():
    initialize_share_data()
    graph = FakeGraphStorage()
    entities_vdb = CaptureVectorStorage()
    relationships_vdb = CaptureVectorStorage()
    pipeline_status = _pipeline_status()
    edges = {
        ("A", "B"): [_relation("A", "B")],
        ("C", "D"): [_relation("C", "D")],
        ("E", "F"): [_relation("E", "F")],
    }

    await merge_nodes_and_edges(
        [({}, edges)],
        cast(Any, graph),
        cast(Any, entities_vdb),
        cast(Any, relationships_vdb),
        _global_config("test_relationship_vdb_batching"),
        doc_id="doc-2",
        pipeline_status=pipeline_status,
        pipeline_status_lock=asyncio.Lock(),
    )

    assert [len(payload) for payload in relationships_vdb.upserts] == [2, 1]
    assert sum(len(ids) for ids in relationships_vdb.deletes) == 6
    relation_pairs = {
        (item["src_id"], item["tgt_id"])
        for payload in relationships_vdb.upserts
        for item in payload.values()
    }
    assert relation_pairs == {("A", "B"), ("C", "D"), ("E", "F")}
    assert pipeline_status["merge_progress"]["stage"] == "Phase 2 relation merge"
    assert pipeline_status["merge_progress"]["finished"] == 3
    assert pipeline_status["merge_progress"]["percent"] == 100.0
    assert pipeline_status["merge_stage_timings"]["phase2_relations"]["total"] == 3
    assert pipeline_status["merge_stage_timings"]["total"]["seconds"] >= 0
    assert "Phase 2 relation merge" in "\n".join(
        pipeline_status["history_messages"]
    )


@pytest.mark.asyncio
async def test_vdb_upsert_batcher_reschedules_after_empty_flush():
    storage = CaptureVectorStorage()
    batcher = _VDBUpsertBatcher(
        cast(Any, storage),
        "test_upsert",
        "records",
        {"embedding_batch_num": 2},
        flush_interval_seconds=0,
    )
    keep_alive = asyncio.Event()
    stale_flush_task = asyncio.create_task(keep_alive.wait())
    batcher._flush_task = stale_flush_task

    try:
        await batcher._flush_pending()
        assert batcher._flush_task is None

        await asyncio.wait_for(
            batcher.submit({"id-1": {"content": "one"}}),
            timeout=1,
        )
    finally:
        stale_flush_task.cancel()
        with suppress(asyncio.CancelledError):
            await stale_flush_task

    assert [list(payload) for payload in storage.upserts] == [["id-1"]]


@pytest.mark.asyncio
async def test_vdb_upsert_batcher_propagates_batch_upsert_failure_to_submitters():
    storage = FailingVectorStorage()
    batcher = _VDBUpsertBatcher(
        cast(Any, storage),
        "failing_upsert",
        "records",
        {"embedding_batch_num": 2},
        retry_delay=0,
        flush_interval_seconds=0.01,
    )

    submitters = [
        asyncio.create_task(batcher.submit({"id-1": {"content": "one"}})),
        asyncio.create_task(batcher.submit({"id-2": {"content": "two"}})),
    ]
    results = await asyncio.wait_for(
        asyncio.gather(*submitters, return_exceptions=True),
        timeout=1,
    )

    assert len(storage.upserts) == 3
    assert all(isinstance(result, Exception) for result in results)
    assert all("failing_upsert" in str(result) for result in results)
    assert all("upsert boom" in str(result) for result in results)


@pytest.mark.asyncio
async def test_merge_stage_progress_serializes_pipeline_status_publication():
    pipeline_status = _pipeline_status()
    pipeline_status_lock = BlockingStatusLock()
    progress = _MergeStageProgress(
        "Phase 2 relation merge",
        "relations",
        2,
        "doc-progress",
        pipeline_status,
        pipeline_status_lock,
    )
    await progress.start("first")
    await progress.start("second")

    pipeline_status_lock.block_next = True
    first_finish = asyncio.create_task(progress.finish("first"))
    await asyncio.wait_for(pipeline_status_lock.blocked.wait(), timeout=1)

    second_finish = asyncio.create_task(progress.finish("second"))
    await asyncio.sleep(0.05)
    assert not second_finish.done()

    pipeline_status_lock.release.set()
    await asyncio.wait_for(asyncio.gather(first_finish, second_finish), timeout=1)

    snapshot = pipeline_status["merge_progress"]
    assert snapshot["completed"] == 2
    assert snapshot["finished"] == 2
    assert snapshot["active"] == 0
    assert snapshot["detail"] == "second"


@pytest.mark.asyncio
async def test_merge_stage_progress_cancel_counts_as_finished_work():
    pipeline_status = _pipeline_status()
    progress = _MergeStageProgress(
        "Phase 2 relation merge",
        "relations",
        1,
        "doc-cancel",
        pipeline_status,
        asyncio.Lock(),
    )

    await progress.start("A->B")
    await progress.cancel("A->B")

    snapshot = pipeline_status["merge_progress"]
    assert snapshot["started"] == 1
    assert snapshot["cancelled"] == 1
    assert snapshot["finished"] == 1
    assert snapshot["active"] == 0
    assert snapshot["state"] == "cancelled"


@pytest.mark.asyncio
async def test_merge_stage_progress_start_cancellation_marks_cancelled():
    pipeline_status = _pipeline_status()
    pipeline_status_lock = BlockingStatusLock()
    progress = _MergeStageProgress(
        "Phase 2 relation merge",
        "relations",
        1,
        "doc-start-cancel",
        pipeline_status,
        pipeline_status_lock,
    )

    pipeline_status_lock.block_next = True
    start_task = asyncio.create_task(progress.start("A->B"))
    await asyncio.wait_for(pipeline_status_lock.blocked.wait(), timeout=1)
    start_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(start_task, timeout=1)

    snapshot = pipeline_status["merge_progress"]
    assert snapshot["started"] == 1
    assert snapshot["cancelled"] == 1
    assert snapshot["finished"] == 1
    assert snapshot["active"] == 0
    assert snapshot["state"] == "cancelled"


@pytest.mark.asyncio
async def test_merge_stage_progress_finish_cancellation_does_not_double_count():
    pipeline_status = _pipeline_status()
    pipeline_status_lock = BlockingStatusLock()
    progress = _MergeStageProgress(
        "Phase 2 relation merge",
        "relations",
        1,
        "doc-finish-cancel",
        pipeline_status,
        pipeline_status_lock,
    )
    await progress.start("A->B")

    pipeline_status_lock.blocked.clear()
    pipeline_status_lock.release.clear()
    pipeline_status_lock.block_next = True
    finish_task = asyncio.create_task(progress.finish("A->B"))
    await asyncio.wait_for(pipeline_status_lock.blocked.wait(), timeout=1)
    finish_task.cancel()

    await asyncio.wait_for(finish_task, timeout=1)

    snapshot = pipeline_status["merge_progress"]
    assert snapshot["started"] == 1
    assert snapshot["completed"] == 1
    assert snapshot["cancelled"] == 0
    assert snapshot["finished"] == 1
    assert snapshot["active"] == 0
    assert snapshot["state"] == "completed"
