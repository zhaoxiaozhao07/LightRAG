from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from lightrag.api.kb_service import KnowledgeBaseService
from lightrag.api.lightrag_registry import (
    DestructiveLockBusyError,
    LightRAGInstanceRegistry,
)

pytestmark = pytest.mark.offline


class FakeRAG:
    def __init__(self, workspace: str, generation: str | None = None):
        self.workspace = workspace
        self.generation = generation
        self.finalized = False

    async def finalize_storages(self) -> None:
        self.finalized = True

    async def adrop_all_storages(self) -> dict:
        return {"dropped": 0, "failed": 0, "errors": []}


class CountingBuilder:
    def __init__(self):
        self.calls: list[str] = []

    async def build(self, record) -> FakeRAG:
        self.calls.append(record.id)
        return FakeRAG(record.workspace, record.generation)

    async def finalize(self, rag) -> None:
        await rag.finalize_storages()


async def _seed_kbs(service: KnowledgeBaseService, ids: list[str]) -> None:
    await service.initialize()
    for kb_id in ids:
        await service.create(kb_id=kb_id, name=kb_id)


@pytest.mark.asyncio
async def test_registry_lru_evicts_least_recently_used(tmp_path: Path):
    service = KnowledgeBaseService(tmp_path / "metadata" / "kb.json")
    await _seed_kbs(service, ["kb_a", "kb_b", "kb_c"])
    builder = CountingBuilder()
    registry = LightRAGInstanceRegistry(
        service, builder.build, builder.finalize, max_entries=2
    )

    rag_a = await registry.get("kb_a")
    rag_b = await registry.get("kb_b")
    assert isinstance(rag_a, FakeRAG)
    assert isinstance(rag_b, FakeRAG)
    assert registry.is_loaded("kb_a")
    assert registry.is_loaded("kb_b")

    # Access kb_a so kb_b becomes the LRU candidate
    await registry.get("kb_a")
    rag_c = await registry.get("kb_c")
    assert isinstance(rag_c, FakeRAG)
    assert registry.is_loaded("kb_a")
    assert registry.is_loaded("kb_c")
    assert not registry.is_loaded("kb_b")
    # kb_b's instance was finalized
    assert rag_b.finalized is True
    assert rag_a.finalized is False
    assert rag_c.finalized is False


@pytest.mark.asyncio
async def test_registry_idle_ttl_reaper_evicts(tmp_path: Path, monkeypatch):
    service = KnowledgeBaseService(tmp_path / "metadata" / "kb.json")
    await _seed_kbs(service, ["kb_one"])
    builder = CountingBuilder()
    registry = LightRAGInstanceRegistry(
        service, builder.build, builder.finalize, idle_ttl_seconds=0.05
    )
    rag = await registry.get("kb_one")
    assert isinstance(rag, FakeRAG)
    assert registry.is_loaded("kb_one")

    # Wait past the TTL and reap
    await asyncio.sleep(0.08)
    evicted = await registry.reap_idle()
    assert evicted == 1
    assert not registry.is_loaded("kb_one")
    assert rag.finalized is True


@pytest.mark.asyncio
async def test_destructive_lock_blocks_discard_and_eviction(tmp_path: Path):
    service = KnowledgeBaseService(tmp_path / "metadata" / "kb.json")
    await _seed_kbs(service, ["kb_lock"])
    builder = CountingBuilder()
    registry = LightRAGInstanceRegistry(
        service, builder.build, builder.finalize, idle_ttl_seconds=0.01
    )
    await registry.get("kb_lock")

    async with registry.destructive_lock("kb_lock"):
        # Discard should refuse while destructive lock is held
        with pytest.raises(DestructiveLockBusyError):
            await registry.discard("kb_lock")
        # Idle reaper must skip protected entries
        await asyncio.sleep(0.03)
        assert await registry.reap_idle() == 0
        assert registry.is_loaded("kb_lock")
        # The destructive worker itself can force_evict while holding the lock
        assert await registry.force_evict("kb_lock") is True
        assert not registry.is_loaded("kb_lock")
    # Lock is released; subsequent discard / get works
    assert await registry.discard("kb_lock") is False
    await registry.get("kb_lock")
    assert registry.is_loaded("kb_lock")


@pytest.mark.asyncio
async def test_force_evict_requires_destructive_lock(tmp_path: Path):
    service = KnowledgeBaseService(tmp_path / "metadata" / "kb.json")
    await _seed_kbs(service, ["kb_force"])
    builder = CountingBuilder()
    registry = LightRAGInstanceRegistry(
        service, builder.build, builder.finalize
    )
    await registry.get("kb_force")
    with pytest.raises(DestructiveLockBusyError):
        await registry.force_evict("kb_force")


@pytest.mark.asyncio
async def test_single_flight_initialization(tmp_path: Path):
    service = KnowledgeBaseService(tmp_path / "metadata" / "kb.json")
    await _seed_kbs(service, ["kb_singleflight"])
    builder = CountingBuilder()

    barrier = asyncio.Event()

    async def slow_build(record):
        await barrier.wait()
        builder.calls.append(record.id)
        return FakeRAG(record.workspace)

    registry = LightRAGInstanceRegistry(
        service, slow_build, builder.finalize
    )

    task1 = asyncio.create_task(registry.get("kb_singleflight"))
    task2 = asyncio.create_task(registry.get("kb_singleflight"))
    await asyncio.sleep(0)
    barrier.set()
    rag1, rag2 = await asyncio.gather(task1, task2)
    assert rag1 is rag2
    assert builder.calls == ["kb_singleflight"]


@pytest.mark.asyncio
async def test_registry_replaces_cached_instance_when_generation_changes(
    tmp_path: Path,
):
    service = KnowledgeBaseService(tmp_path / "metadata" / "kb.json")
    await service.initialize()
    first_record = await service.create(kb_id="kb_recreated", name="First")
    builder = CountingBuilder()
    registry = LightRAGInstanceRegistry(service, builder.build, builder.finalize)

    first_rag = await registry.get(first_record.id)
    assert isinstance(first_rag, FakeRAG)
    await service.delete(
        first_record.id,
        expected_generation=first_record.generation,
    )
    assert await service.purge(
        first_record.id,
        expected_generation=first_record.generation,
    )
    second_record = await service.create(kb_id=first_record.id, name="Second")

    assert second_record.workspace == first_record.workspace
    assert second_record.active_config_version_id == first_record.active_config_version_id
    assert second_record.generation != first_record.generation

    replacements = await asyncio.gather(
        *(registry.get(second_record.id) for _ in range(8))
    )
    assert isinstance(replacements[0], FakeRAG)
    assert all(rag is replacements[0] for rag in replacements)
    assert replacements[0] is not first_rag
    assert replacements[0].generation == second_record.generation
    assert first_rag.finalized is True
    assert builder.calls == [first_record.id, second_record.id]


@pytest.mark.asyncio
async def test_registry_discards_build_that_races_catalog_recreation(tmp_path: Path):
    service = KnowledgeBaseService(tmp_path / "metadata" / "kb.json")
    await service.initialize()
    first_record = await service.create(kb_id="kb_build_race", name="First")
    build_started = asyncio.Event()
    release_build = asyncio.Event()
    built: list[FakeRAG] = []

    async def racing_build(record) -> FakeRAG:
        rag = FakeRAG(record.workspace, record.generation)
        built.append(rag)
        if len(built) == 1:
            build_started.set()
            await release_build.wait()
        return rag

    async def finalize(rag) -> None:
        await rag.finalize_storages()

    registry = LightRAGInstanceRegistry(service, racing_build, finalize)
    get_task = asyncio.create_task(registry.get(first_record.id))
    await asyncio.wait_for(build_started.wait(), timeout=5)

    await service.delete(
        first_record.id,
        expected_generation=first_record.generation,
    )
    assert await service.purge(
        first_record.id,
        expected_generation=first_record.generation,
    )
    second_record = await service.create(kb_id=first_record.id, name="Second")
    release_build.set()

    current_rag = await asyncio.wait_for(get_task, timeout=5)
    assert isinstance(current_rag, FakeRAG)
    assert current_rag.generation == second_record.generation
    assert [rag.generation for rag in built] == [
        first_record.generation,
        second_record.generation,
    ]
    assert built[0].finalized is True
    assert built[1] is current_rag
