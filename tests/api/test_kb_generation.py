from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from lightrag.api.kb_service import (
    KnowledgeBaseConflictError,
    KnowledgeBaseRecord,
    KnowledgeBaseService,
)
from lightrag.api.postgres_kb_service import PostgresKnowledgeBaseService

pytestmark = pytest.mark.offline


class _AsyncContext:
    def __init__(self, value: Any):
        self._value = value

    async def __aenter__(self) -> Any:
        return self._value

    async def __aexit__(self, *_args: object) -> None:
        return None


class _PostgresConnectionProbe:
    def __init__(self):
        self.inserted_record: dict[str, Any] | None = None

    def transaction(self) -> _AsyncContext:
        return _AsyncContext(self)

    async def fetchrow(self, query: str, *_args: object) -> Any:
        if "SELECT id FROM kb_catalog" in query:
            return None if self.inserted_record is None else {"id": "existing"}
        if "SELECT data_json FROM kb_catalog" in query:
            if self.inserted_record is None:
                return None
            return {"data_json": dict(self.inserted_record)}
        raise AssertionError(f"Unexpected query: {query}")

    async def execute(self, query: str, *args: object) -> str:
        if "INSERT INTO kb_catalog" in query:
            self.inserted_record = json.loads(str(args[-1]))
            return "INSERT 0 1"
        if "UPDATE kb_catalog" in query:
            self.inserted_record = json.loads(str(args[-2]))
            return "UPDATE 1"
        if "DELETE FROM kb_catalog" in query:
            self.inserted_record = None
            return "DELETE 1"
        raise AssertionError(f"Unexpected query: {query}")


class _PostgresPoolProbe:
    def __init__(self, connection: _PostgresConnectionProbe):
        self._connection = connection

    def acquire(self) -> _AsyncContext:
        return _AsyncContext(self._connection)


@pytest.mark.asyncio
async def test_file_catalog_generation_is_created_and_immutable(tmp_path: Path):
    service = KnowledgeBaseService(tmp_path / "knowledge_bases.json")
    await service.initialize()

    created = await service.create(kb_id="kb_generation", name="Generation")
    generation = created.generation

    assert generation
    assert UUID(generation).version == 4
    assert created.to_dict()["generation"] == generation
    assert created.origin == "platform"

    updated = await service.update(created.id, name="Updated")
    deleted = await service.delete(created.id)
    restored = await service.restore(created.id)

    assert updated.generation == generation
    assert deleted.generation == generation
    assert restored.generation == generation
    assert {updated.origin, deleted.origin, restored.origin} == {"platform"}
    with pytest.raises(TypeError):
        await service.update(created.id, generation="replacement")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        await service.update(created.id, origin="tenant")  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_legacy_file_catalog_generation_is_stable_and_saved(tmp_path: Path):
    metadata_path = tmp_path / "knowledge_bases.json"
    legacy_record = {
        "id": "kb_legacy_generation",
        "name": "Legacy",
        "description": None,
        "workspace": "kb_kb_legacy_generation",
        "status": "active",
        "active_config_version_id": None,
        "owner_id": None,
        "tenant_id": "tenant-a",
        "visibility": "private",
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-01T00:00:00+00:00",
        "deleted_at": None,
        "metadata": {"tenant_managed": True},
    }
    metadata_path.write_text(
        json.dumps(
            {
                "version": 1,
                "knowledge_bases": {legacy_record["id"]: legacy_record},
            }
        ),
        encoding="utf-8",
    )

    first_service = KnowledgeBaseService(metadata_path)
    second_service = KnowledgeBaseService(metadata_path)
    await first_service.initialize()
    await second_service.initialize()

    first = await first_service.get(legacy_record["id"])
    second = await second_service.get(legacy_record["id"])
    assert first.generation
    assert first.generation == second.generation
    assert UUID(first.generation).version == 5
    assert first.origin == "platform"

    await first_service.update(first.id, name="Legacy Updated")
    persisted = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert (
        persisted["knowledge_bases"][first.id]["generation"]
        == first.generation
    )


@pytest.mark.asyncio
async def test_postgres_create_persists_a_new_generation():
    connection = _PostgresConnectionProbe()
    service = PostgresKnowledgeBaseService()
    service._pool = _PostgresPoolProbe(connection)
    service._initialized = True

    created = await service.create(
        kb_id="kb_pg_generation",
        name="Postgres",
        tenant_id="tenant-a",
        origin="tenant",
    )

    assert created.generation
    assert UUID(created.generation).version == 4
    assert connection.inserted_record is not None
    assert connection.inserted_record["generation"] == created.generation
    assert created.origin == "tenant"
    assert connection.inserted_record["origin"] == "tenant"

    updated = await service.update(created.id, name="Postgres Updated")
    deleted = await service.delete(created.id)
    restored = await service.restore(created.id)
    assert updated.generation == created.generation
    assert deleted.generation == created.generation
    assert restored.generation == created.generation
    assert {updated.origin, deleted.origin, restored.origin} == {"tenant"}
    assert connection.inserted_record["generation"] == created.generation


@pytest.mark.asyncio
async def test_legacy_postgres_generation_is_stable_and_saved():
    connection = _PostgresConnectionProbe()
    connection.inserted_record = {
        "id": "kb_pg_legacy",
        "name": "Legacy Postgres",
        "description": None,
        "workspace": "kb_kb_pg_legacy",
        "status": "active",
        "active_config_version_id": None,
        "owner_id": None,
        "tenant_id": "tenant-a",
        "visibility": "private",
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-01T00:00:00+00:00",
        "deleted_at": None,
        "metadata": {"tenant_managed": True},
    }
    service = PostgresKnowledgeBaseService()
    service._pool = _PostgresPoolProbe(connection)
    service._initialized = True

    first = await service.get("kb_pg_legacy")
    second = await service.get("kb_pg_legacy")
    assert first.generation == second.generation
    assert UUID(first.generation).version == 5
    assert first.origin == "platform"

    updated = await service.update(first.id, name="Legacy Postgres Updated")
    assert updated.generation == first.generation
    assert connection.inserted_record["generation"] == first.generation


def _legacy_record(
    *, tenant_id: str | None, metadata: dict[str, Any], origin: str | None = None
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": "kb_legacy_origin",
        "name": "Legacy Origin",
        "description": None,
        "workspace": "kb_kb_legacy_origin",
        "status": "active",
        "active_config_version_id": None,
        "owner_id": None,
        "tenant_id": tenant_id,
        "visibility": "private",
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-01T00:00:00+00:00",
        "deleted_at": None,
        "metadata": metadata,
    }
    if origin is not None:
        record["origin"] = origin
    return record


@pytest.mark.parametrize(
    ("tenant_id", "metadata"),
    [
        (None, {}),
        ("tenant-a", {}),
        (None, {"tenant_managed": True}),
        ("tenant-a", {"tenant_managed": True}),
        (None, {"platform_provisioned": True}),
        ("tenant-a", {"platform_provisioned": True}),
        (
            None,
            {"tenant_managed": True, "platform_provisioned": True},
        ),
        (
            "tenant-a",
            {"tenant_managed": True, "platform_provisioned": True},
        ),
        ("", {"tenant_managed": True}),
    ],
)
def test_legacy_origin_defaults_to_platform_without_trusting_provenance(
    tenant_id: str | None,
    metadata: dict[str, Any],
):
    record = KnowledgeBaseRecord.from_dict(
        _legacy_record(tenant_id=tenant_id, metadata=metadata)
    )
    assert record.origin == "platform"
    assert record.tenant_id == tenant_id
    assert record.metadata == metadata


def test_explicit_origin_wins_over_legacy_metadata_markers():
    record = KnowledgeBaseRecord.from_dict(
        _legacy_record(
            tenant_id="tenant-a",
            metadata={"platform_provisioned": True},
            origin="tenant",
        )
    )
    assert record.origin == "tenant"


@pytest.mark.asyncio
async def test_file_catalog_initial_status_origin_and_generation_cas(tmp_path: Path):
    service = KnowledgeBaseService(tmp_path / "knowledge_bases.json")
    created = await service.create(
        kb_id="kb_file_cas",
        name="File CAS",
        tenant_id="tenant-a",
        origin="tenant",
        initial_status="creating",
    )
    assert created.status == "creating"
    assert created.origin == "tenant"

    with pytest.raises(KnowledgeBaseConflictError):
        await service.update(
            created.id,
            status="active",
            expected_generation="stale-generation",
        )
    active = await service.update(
        created.id,
        status="active",
        expected_generation=created.generation,
    )
    assert active.status == "active"
    assert active.origin == "tenant"

    with pytest.raises(KnowledgeBaseConflictError):
        await service.delete(created.id, expected_generation="stale-generation")
    deleted = await service.delete(
        created.id, expected_generation=created.generation
    )
    with pytest.raises(KnowledgeBaseConflictError):
        await service.restore(created.id, expected_generation="stale-generation")
    restored = await service.restore(
        created.id, expected_generation=created.generation
    )
    assert deleted.origin == restored.origin == "tenant"
    with pytest.raises(KnowledgeBaseConflictError):
        await service.purge(created.id, expected_generation="stale-generation")
    # Generation equality alone is insufficient: a restored/active catalog row
    # must never be purged by a delayed hard-delete tail.
    with pytest.raises(KnowledgeBaseConflictError):
        await service.purge(
            created.id,
            expected_generation=created.generation,
        )
    await service.delete(created.id, expected_generation=created.generation)
    assert await service.purge(
        created.id, expected_generation=created.generation
    )


@pytest.mark.asyncio
async def test_postgres_catalog_initial_status_origin_and_generation_cas():
    connection = _PostgresConnectionProbe()
    service = PostgresKnowledgeBaseService()
    service._pool = _PostgresPoolProbe(connection)
    service._initialized = True

    created = await service.create(
        kb_id="kb_pg_cas",
        name="Postgres CAS",
        tenant_id="tenant-a",
        origin="tenant",
        initial_status="creating",
    )
    assert created.status == "creating"
    assert connection.inserted_record is not None
    assert connection.inserted_record["origin"] == "tenant"

    with pytest.raises(KnowledgeBaseConflictError):
        await service.update(
            created.id,
            status="active",
            expected_generation="stale-generation",
        )
    active = await service.update(
        created.id,
        status="active",
        expected_generation=created.generation,
    )
    assert active.status == "active"
    with pytest.raises(KnowledgeBaseConflictError):
        await service.delete(created.id, expected_generation="stale-generation")
    await service.delete(created.id, expected_generation=created.generation)
    with pytest.raises(KnowledgeBaseConflictError):
        await service.restore(created.id, expected_generation="stale-generation")
    restored = await service.restore(
        created.id, expected_generation=created.generation
    )
    assert restored.origin == "tenant"
    with pytest.raises(KnowledgeBaseConflictError):
        await service.purge(created.id, expected_generation="stale-generation")
    with pytest.raises(KnowledgeBaseConflictError):
        await service.purge(
            created.id,
            expected_generation=created.generation,
        )
    await service.delete(created.id, expected_generation=created.generation)
    assert await service.purge(
        created.id, expected_generation=created.generation
    )
    assert connection.inserted_record is None
