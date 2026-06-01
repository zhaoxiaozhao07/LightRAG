"""Durable resume of single-document ``replace`` jobs.

Replace jobs were previously NOT worker-resumable: their uploaded bytes lived
only in request memory, so a crash mid-replace lost them. This test proves the
new durability path:

1. a document is driven to ``ready`` through the normal API,
2. a ``replace`` job is created ``queued`` and its replacement bytes are staged
   to disk (``stage_replacement_bytes``) — modelling the state right after the
   route claims the job but before/instead of the in-process task finishing
   (i.e. a crash, then orphan-recovery → ``replace_failed`` → ``:retry`` →
   ``queued``),
3. the durable ``JobWorker`` claims the queued replace job and re-drives it to
   ``succeeded`` entirely from the staged bytes — deleting the old index,
   swapping the source, and (here) re-parsing — with no request context.

It also covers the non-resumable fallback: a queued replace job with no staged
bytes fails cleanly with ``replace_not_resumable`` instead of guessing.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightrag.api.document_lifecycle_service import (
    DocumentLifecycleService,
    DocumentSourceInput,
)
from lightrag.api.index_build_service import IndexBuildService
from lightrag.api.job_service import JobService
from lightrag.api.job_worker import JobWorker, build_replace_executor
from lightrag.api.kb_service import KnowledgeBaseService
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry
from lightrag.api.metadata_store import SQLiteMetadataStore

# Reuse the fully-wired FakeRAG/BuilderProbe from the build-kg route tests.
from tests.api.routes.test_kb_build_kg_routes import BuilderProbe, FakeRAG  # noqa: F401

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_kb_routes = importlib.import_module("lightrag.api.routers.kb_routes")
_kb_document_routes = importlib.import_module("lightrag.api.routers.kb_document_routes")
sys.argv = _original_argv

create_kb_routes = _kb_routes.create_kb_routes
create_kb_document_routes = _kb_document_routes.create_kb_document_routes

pytestmark = pytest.mark.offline

_API_KEY = "test-key"
_HEADERS = {"X-API-Key": _API_KEY}


def _wire(tmp_path: Path):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "knowledge_bases.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    document_service = DocumentLifecycleService(
        kb_service, metadata_store, tmp_path / "inputs"
    )
    job_service = JobService(kb_service, metadata_store)
    index_service = IndexBuildService(document_service)
    probe = BuilderProbe()
    registry = LightRAGInstanceRegistry(kb_service, probe.build, probe.finalize)
    app = FastAPI()
    app.include_router(
        create_kb_routes(kb_service, registry, api_key=_API_KEY, job_service=job_service)
    )
    app.include_router(
        create_kb_document_routes(
            document_service,
            job_service,
            api_key=_API_KEY,
            registry=registry,
            index_service=index_service,
        )
    )
    client = TestClient(app)
    return {
        "client": client,
        "kb_service": kb_service,
        "metadata_store": metadata_store,
        "document_service": document_service,
        "job_service": job_service,
        "index_service": index_service,
        "registry": registry,
        "probe": probe,
    }


def _ready_document(client: TestClient, kb_id: str) -> str:
    assert (
        client.post("/kbs", json={"id": kb_id, "name": kb_id}, headers=_HEADERS).status_code
        == 200
    )
    upload = client.post(
        f"/kbs/{kb_id}/documents:upload",
        files=[("files", ("paper.pdf", b"original-bytes", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200, upload.text
    document_id = upload.json()["documents"][0]["id"]
    parse = client.post(
        f"/kbs/{kb_id}/documents/{document_id}:parse",
        json={"engine": "mineru", "process_options": "iF"},
        headers=_HEADERS,
    )
    assert parse.status_code == 200, parse.text
    build = client.post(
        f"/kbs/{kb_id}/documents/{document_id}:build-kg", json={}, headers=_HEADERS
    )
    assert build.status_code == 200, build.text
    build_job = client.get(
        f"/kbs/{kb_id}/jobs/{build.json()['id']}", headers=_HEADERS
    ).json()
    assert build_job["status"] == "succeeded", build_job
    detail = client.get(f"/kbs/{kb_id}/documents/{document_id}", headers=_HEADERS).json()
    assert detail["status"] == "ready"
    return document_id


async def test_worker_resumes_queued_replace_from_staged_bytes(tmp_path):
    env = _wire(tmp_path)
    client = env["client"]
    document_service = env["document_service"]
    job_service = env["job_service"]
    kb_id = "kb_replace_resume"
    document_id = _ready_document(client, kb_id)

    before = client.get(f"/kbs/{kb_id}/documents/{document_id}", headers=_HEADERS).json()
    old_hash = before["source_hash"]
    old_lightrag_id = before["lightrag_doc_id"]

    # Simulate the post-crash, post-:retry state: a QUEUED replace job whose
    # bytes were staged at claim time, but which was never run in-process.
    replacement = document_service.prepare_replacement_source(
        DocumentSourceInput(
            source_name="paper-v2.pdf",
            content=b"replacement-bytes-v2",
            source_type="upload",
            content_type="application/pdf",
            metadata={},
        )
    )
    job, created = await job_service.create_replace_job_once(
        kb_id,
        document_id=document_id,
        previous_lightrag_doc_id=old_lightrag_id,
        source_name=replacement.source_name,
        source_hash=replacement.source_hash,
        content_type=replacement.content_type,
        size_bytes=replacement.size_bytes,
        auto_parse=True,
        auto_index=False,
        parser_engine="mineru",
        process_options="iF",
    )
    assert created is True
    assert job.status == "queued"
    staged_path = await document_service.stage_replacement_bytes(
        kb_id, document_id, job_id=job.id, replacement=replacement
    )
    assert Path(staged_path).is_file()

    # The durable worker claims the queued replace job and drives it to done.
    worker = JobWorker(
        job_service,
        executors={
            "replace": build_replace_executor(
                document_service=document_service,
                registry=env["registry"],
                job_service=job_service,
                index_service=env["index_service"],
            )
        },
        claim_grace_seconds=0.0,  # no grace: claim the freshly-queued job now
    )
    claimed = await worker.poll_once()
    assert claimed is not None
    assert claimed.id == job.id

    final = client.get(f"/kbs/{kb_id}/jobs/{job.id}", headers=_HEADERS).json()
    assert final["status"] == "succeeded", final
    assert final["result"]["resumed_by_worker"] is True

    # The source was actually swapped (new hash) and re-parsed to parsed/ready.
    after = client.get(f"/kbs/{kb_id}/documents/{document_id}", headers=_HEADERS).json()
    assert after["source_hash"] != old_hash
    assert after["source_name"] == "paper-v2.pdf"
    assert after["status"] in ("parsed", "ready")

    # Old index was deleted on the shared FakeRAG instance.
    rag = env["probe"].instances[0]
    assert (old_lightrag_id, False) in rag.delete_calls

    # Staged bytes were cleaned up after the terminal transition.
    assert not Path(staged_path).is_file()


async def test_worker_fails_replace_without_staged_bytes(tmp_path):
    env = _wire(tmp_path)
    client = env["client"]
    job_service = env["job_service"]
    kb_id = "kb_replace_nostage"
    document_id = _ready_document(client, kb_id)

    # Queued replace job but NO staged bytes (older job / never staged).
    job, created = await job_service.create_replace_job_once(
        kb_id,
        document_id=document_id,
        previous_lightrag_doc_id=None,
        source_name="paper-v2.pdf",
        source_hash="sha256:does-not-matter",
        content_type="application/pdf",
        size_bytes=10,
        auto_parse=False,
        auto_index=False,
    )
    assert created is True

    worker = JobWorker(
        job_service,
        executors={
            "replace": build_replace_executor(
                document_service=env["document_service"],
                registry=env["registry"],
                job_service=job_service,
                index_service=env["index_service"],
            )
        },
        claim_grace_seconds=0.0,
    )
    claimed = await worker.poll_once()
    assert claimed is not None

    final = client.get(f"/kbs/{kb_id}/jobs/{job.id}", headers=_HEADERS).json()
    assert final["status"] == "failed"
    assert final["error_code"] == "replace_not_resumable"


async def test_orphan_recovery_preserves_queued_replace_job(tmp_path):
    """A queued ``replace`` job must survive restart orphan-recovery when the
    durable worker lists ``replace`` as resumable (so the worker can re-drive
    it), instead of being failed like non-resumable types."""
    env = _wire(tmp_path)
    client = env["client"]
    job_service = env["job_service"]
    metadata_store = env["metadata_store"]
    kb_id = "kb_replace_orphan"
    document_id = _ready_document(client, kb_id)

    job, created = await job_service.create_replace_job_once(
        kb_id,
        document_id=document_id,
        previous_lightrag_doc_id=None,
        source_name="paper-v2.pdf",
        source_hash="sha256:x",
        content_type="application/pdf",
        size_bytes=10,
        auto_parse=False,
        auto_index=False,
    )
    assert created and job.status == "queued"

    # Worker lists replace as resumable -> recovery leaves the queued job alone.
    recovered = await metadata_store.recover_orphan_jobs(
        resumable_job_types={"parse", "build_kg", "reindex", "delete", "replace"}
    )
    assert all(r.id != job.id for r in recovered), "queued replace must be preserved"
    after = client.get(f"/kbs/{kb_id}/jobs/{job.id}", headers=_HEADERS).json()
    assert after["status"] == "queued"

    # Without replace in the resumable set, the same job WOULD be failed.
    job2, _ = await job_service.create_replace_job_once(
        kb_id,
        document_id=document_id,
        previous_lightrag_doc_id=None,
        source_name="paper-v3.pdf",
        source_hash="sha256:y",
        content_type="application/pdf",
        size_bytes=10,
        auto_parse=False,
        auto_index=False,
        idempotency_key="other-key",
    )
    recovered2 = await metadata_store.recover_orphan_jobs(
        resumable_job_types={"parse"}
    )
    assert any(r.id == job2.id for r in recovered2)
