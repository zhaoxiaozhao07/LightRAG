from __future__ import annotations

import json
from pathlib import Path

import pytest

from lightrag.api.kb_service import KnowledgeBaseRecord, utc_now_iso
from lightrag.api.metadata_store import (
    ArtifactRecord,
    ConfigVersionRecord,
    DocumentRecord,
    JobRecord,
    SQLiteMetadataStore,
)
from lightrag.tools.migrate_kb_metadata_to_postgres import (
    ControlPlaneSnapshot,
    SourceKeyMapping,
    apply_snapshot_to_postgres,
    collect_local_snapshot,
    dry_run_summary,
    normalize_documents_for_postgres,
)

pytestmark = pytest.mark.offline


def _kb(kb_id: str) -> KnowledgeBaseRecord:
    now = utc_now_iso()
    return KnowledgeBaseRecord(
        id=kb_id,
        name=f"KB {kb_id}",
        description="migration fixture",
        workspace=f"kb_{kb_id}",
        status="active",
        active_config_version_id="cfg_a",
        owner_id="owner",
        tenant_id="tenant",
        visibility="private",
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )


def _doc(kb_id: str, doc_id: str, *, source_key: str | None = None) -> DocumentRecord:
    now = utc_now_iso()
    metadata: dict[str, object] = {"batch_id": "batch_a"}
    if source_key is not None:
        metadata["source_key"] = source_key
    return DocumentRecord(
        id=doc_id,
        kb_id=kb_id,
        workspace=f"kb_{kb_id}",
        lightrag_doc_id="lr-doc",
        source_type="upload",
        source_name=f"{doc_id}.pdf",
        source_uri=f"/inputs/{doc_id}.pdf",
        source_hash="sha256:src",
        content_type="application/pdf",
        size_bytes=123,
        parser_hash="sha256:parser",
        index_hash="sha256:index",
        status="ready",
        enabled=True,
        archived=False,
        chunks_count=3,
        entity_count=2,
        relation_count=1,
        error_code=None,
        error_message=None,
        metadata=metadata,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )


def _job(kb_id: str, job_id: str, document_id: str) -> JobRecord:
    now = utc_now_iso()
    return JobRecord(
        id=job_id,
        kb_id=kb_id,
        workspace=f"kb_{kb_id}",
        batch_id="batch_a",
        document_id=document_id,
        job_type="parse",
        status="succeeded",
        stage="parsing",
        progress=1.0,
        total_items=1,
        completed_items=1,
        failed_items=0,
        idempotency_key="idem-a",
        config_version_id="cfg_a",
        config_hash="sha256:cfg",
        retry_count=0,
        max_retries=3,
        payload={"idempotency_fingerprint": "v1"},
        result={"ok": True},
        error_code=None,
        error_message=None,
        created_at=now,
        updated_at=now,
        queued_at=now,
        started_at=now,
        finished_at=now,
        cancelled_at=None,
    )


def _artifact(kb_id: str, document_id: str) -> ArtifactRecord:
    return ArtifactRecord(
        id="art_a",
        kb_id=kb_id,
        workspace=f"kb_{kb_id}",
        document_id=document_id,
        artifact_type="original",
        uri=f"/inputs/{document_id}/original.pdf",
        checksum="sha256:art",
        size_bytes=42,
        metadata={"source": "mineru"},
        created_at=utc_now_iso(),
    )


def _config(kb_id: str) -> ConfigVersionRecord:
    return ConfigVersionRecord(
        id="cfg_a",
        kb_id=kb_id,
        workspace=f"kb_{kb_id}",
        version=7,
        config={"chunk_config": {"chunk_size": 512}},
        parser_hash="sha256:parser",
        index_hash="sha256:index",
        query_hash="sha256:query",
        created_at=utc_now_iso(),
        activated_at=utc_now_iso(),
        created_by="tester",
    )


async def _seed_local_working_dir(tmp_path: Path) -> Path:
    working_dir = tmp_path / "rag_storage"
    metadata_dir = working_dir / "metadata"
    metadata_dir.mkdir(parents=True)
    kb = _kb("kb_migrate")
    (metadata_dir / "knowledge_bases.json").write_text(
        json.dumps({"version": 1, "knowledge_bases": {kb.id: kb.to_dict()}}),
        encoding="utf-8",
    )
    store = SQLiteMetadataStore(metadata_dir / "metadata.sqlite3")
    await store.initialize()
    doc = _doc(kb.id, "doc_a", source_key="manual/doc_a.pdf")
    await store.create_documents_and_job([doc], _job(kb.id, "job_a", doc.id))
    await store.complete_document_parse(
        kb.id,
        doc.id,
        parser_hash="sha256:parser",
        lightrag_doc_id="lr-doc",
        metadata_patch={"parsed": True},
        artifacts=[_artifact(kb.id, doc.id)],
    )
    await store.create_config_version(_config(kb.id))
    await store.close()
    return working_dir


def test_normalize_documents_backfills_source_key_from_sqlite_mapping():
    doc = _doc("kb_migrate", "doc_a", source_key=None)
    mapping = SourceKeyMapping(
        kb_id="kb_migrate",
        source_key="manual/doc_a.pdf",
        document_id="doc_a",
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )

    normalized, issues = normalize_documents_for_postgres([doc], [mapping])

    assert normalized[0].metadata["source_key"] == "manual/doc_a.pdf"
    assert "Backfilled missing metadata.source_key" in issues[0]


def test_normalize_documents_rejects_duplicate_active_source_keys():
    docs = [
        _doc("kb_migrate", "doc_a", source_key="duplicate.pdf"),
        _doc("kb_migrate", "doc_b", source_key="duplicate.pdf"),
    ]

    with pytest.raises(ValueError, match="Duplicate active source_key"):
        normalize_documents_for_postgres(docs, [])


async def test_collect_local_snapshot_preserves_catalog_and_metadata(tmp_path: Path):
    working_dir = await _seed_local_working_dir(tmp_path)

    snapshot = await collect_local_snapshot(working_dir)

    assert snapshot.counts() == {
        "knowledge_bases": 1,
        "documents": 1,
        "jobs": 1,
        "artifacts": 1,
        "config_versions": 1,
        "source_key_mappings": 1,
    }
    assert snapshot.knowledge_bases[0].active_config_version_id == "cfg_a"
    assert snapshot.documents[0].metadata["source_key"] == "manual/doc_a.pdf"
    # create_config_version assigns version 1 in SQLite; the migration snapshot
    # must preserve the stored value rather than recomputing it later.
    assert snapshot.config_versions[0].version == 1
    assert dry_run_summary(snapshot, strategy="fail").source_keys_projected == 1


async def test_collect_local_snapshot_empty_kb_filter_does_not_read_all_metadata(
    tmp_path: Path,
):
    working_dir = await _seed_local_working_dir(tmp_path)

    snapshot = await collect_local_snapshot(working_dir, kb_ids=["missing_kb"])

    assert snapshot.counts() == {
        "knowledge_bases": 0,
        "documents": 0,
        "jobs": 0,
        "artifacts": 0,
        "config_versions": 0,
        "source_key_mappings": 0,
    }


class _FakePostgresConnection:
    def __init__(self, existing: set[tuple[str, object]] | None = None):
        self.existing = existing or set()
        self.executed: list[tuple[str, tuple[object, ...]]] = []

    async def fetch(self, query: str, *args):
        if "FROM kb_catalog" in query:
            rows: list[dict[str, object]] = []
            if ("kb_catalog", args[0]) in self.existing:
                rows.append({"id": args[0]})
            if len(args) > 1 and ("kb_catalog_workspace", args[1]) in self.existing:
                rows.append({"id": args[1]})
            return rows
        if "FROM kb_documents" in query:
            rows = []
            if ("kb_documents", args[0]) in self.existing:
                rows.append({"kb_id": "kb_migrate", "id": args[0]})
            return rows
        return []

    async def fetchval(self, query: str, *args):
        if "kb_catalog" in query:
            return 1 if ("kb_catalog", args[0]) in self.existing else None
        if "kb_documents" in query:
            return 1 if ("kb_documents", args[0]) in self.existing else None
        if "kb_jobs" in query:
            return 1 if ("kb_jobs", args[0]) in self.existing else None
        if "kb_document_artifacts" in query:
            return 1 if ("kb_document_artifacts", args[0]) in self.existing else None
        if "kb_config_versions" in query:
            return 1 if ("kb_config_versions", args[0]) in self.existing else None
        return None

    async def execute(self, query: str, *args):
        self.executed.append((" ".join(query.split()), args))
        return "INSERT 0 1"


async def test_apply_snapshot_to_postgres_uses_skip_strategy():
    kb = _kb("kb_migrate")
    doc = _doc(kb.id, "doc_a", source_key="manual/doc_a.pdf")
    snapshot = ControlPlaneSnapshot(
        knowledge_bases=[kb],
        documents=[doc],
        jobs=[_job(kb.id, "job_a", doc.id)],
        artifacts=[_artifact(kb.id, doc.id)],
        config_versions=[_config(kb.id)],
        source_key_mappings=[],
        issues=[],
    )
    catalog_conn = _FakePostgresConnection(existing={("kb_catalog", kb.id)})
    metadata_conn = _FakePostgresConnection()

    summary = await apply_snapshot_to_postgres(
        snapshot,
        catalog_conn=catalog_conn,
        metadata_conn=metadata_conn,
        strategy="skip",
    )

    assert summary.catalog.skipped == 1
    assert summary.documents.skipped == 1
    assert summary.jobs.skipped == 1
    assert summary.artifacts.skipped == 1
    assert summary.config_versions.skipped == 1
    assert summary.source_keys_projected == 1
    assert not any("INSERT INTO kb_catalog" in sql for sql, _ in catalog_conn.executed)
    assert not any("INSERT INTO kb_documents" in sql for sql, _ in metadata_conn.executed)


async def test_apply_snapshot_to_postgres_overwrite_deletes_related_metadata():
    kb = _kb("kb_migrate")
    doc = _doc(kb.id, "doc_a", source_key="manual/doc_a.pdf")
    snapshot = ControlPlaneSnapshot(
        knowledge_bases=[kb],
        documents=[doc],
        jobs=[_job(kb.id, "job_a", doc.id)],
        artifacts=[_artifact(kb.id, doc.id)],
        config_versions=[_config(kb.id)],
        source_key_mappings=[],
        issues=[],
    )
    catalog_conn = _FakePostgresConnection(existing={("kb_catalog", kb.id)})
    metadata_conn = _FakePostgresConnection(existing={("kb_documents", doc.id)})

    summary = await apply_snapshot_to_postgres(
        snapshot,
        catalog_conn=catalog_conn,
        metadata_conn=metadata_conn,
        strategy="overwrite",
    )

    assert summary.catalog.overwritten == 1
    assert summary.documents.overwritten == 1
    metadata_sql = [sql for sql, _ in metadata_conn.executed]
    assert any("DELETE FROM kb_document_artifacts WHERE kb_id = $1" in sql for sql in metadata_sql)
    assert any("DELETE FROM kb_documents WHERE kb_id = $1" in sql for sql in metadata_sql)
    assert any(
        "DELETE FROM kb_document_artifacts WHERE kb_id = $1 AND document_id = $2" in sql
        for sql in metadata_sql
    )
    assert any("DELETE FROM kb_jobs WHERE kb_id = $1 AND document_id = $2" in sql for sql in metadata_sql)


def test_cli_requires_confirmation_for_overwrite():
    from lightrag.tools.migrate_kb_metadata_to_postgres import main

    with pytest.raises(SystemExit):
        main(["--strategy", "overwrite"])
