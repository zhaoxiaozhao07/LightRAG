"""Concurrency tests for the KB aggregate parse/sync refactor.

The aggregate flows (``documents:sync`` and ``documents:upload?auto_parse=
true&auto_index=true``) used to process documents strictly one-at-a-time:
parse doc A fully, drive its build through its own
``apipeline_process_enqueue_documents`` drain, THEN doc B. The refactor makes
them two-phase:

* **Phase 1** parses every document concurrently, bounded by
  ``asyncio.Semaphore(MAX_PARALLEL_PARSE_MINERU)``.
* **Phase 2** bulk-enqueues all parsed docs in ONE ``apipeline_enqueue_documents``
  call followed by ONE ``apipeline_process_enqueue_documents`` drain so the
  three pipeline worker layers overlap documents.

These tests use a FakeRAG that actually MODELS the concurrency-relevant
behaviors the plain full-pipeline fake glosses over:

* parse concurrency tracking (max simultaneous parse calls), to prove the
  semaphore bound is honored;
* a single-drain ``busy`` contract (enqueue stamps docs *pending*; a drain
  stamps pending→processed), and a ``process_enqueue`` call counter, to prove
  the batch path drains exactly once;
* an optional "another flow owns the drain" mode (enqueue stamps pending, the
  drain returns WITHOUT processing and an out-of-band owner finishes the docs
  later) — this reproduces the busy-mutex race the read-back poll fixes.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightrag.api.document_lifecycle_service import DocumentLifecycleService
from lightrag.api.index_build_service import (
    BuildArtifactReference,
    IndexBuildExecution,
    IndexBuildPlan,
    IndexBuildService,
)
from lightrag.api.job_service import JobService
from lightrag.api.kb_service import KnowledgeBaseService
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry, LightRAGLike
from lightrag.api.metadata_store import DocumentRecord, SQLiteMetadataStore

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


@pytest.fixture(autouse=True)
def _disable_enterprise_auth_for_non_enterprise_route_tests(monkeypatch):
    from lightrag.api import config as api_config

    monkeypatch.setattr(
        api_config,
        "global_args",
        SimpleNamespace(
            enterprise_auth_enabled=False,
            token_auto_renew=False,
            token_renew_threshold=0.5,
        ),
    )


class FakeDocStatus:
    def __init__(self):
        self.rows: dict[str, dict] = {}

    async def get_by_ids(self, ids):
        return [self.rows.get(item_id) for item_id in ids]

    def stamp_pending(self, doc_id: str) -> None:
        self.rows.setdefault(doc_id, {"status": "pending"})

    def stamp_processed(self, doc_id: str) -> None:
        self.rows[doc_id] = {
            "status": "processed",
            "chunks_count": 4,
            "entity_count": 9,
            "relation_count": 6,
        }

    def stamp_failed(self, doc_id: str, error_msg: str) -> None:
        self.rows[doc_id] = {"status": "failed", "error_msg": error_msg}


class FakeDeletionResult:
    def __init__(self, doc_id: str, delete_llm_cache: bool):
        self.status = "success"
        self.doc_id = doc_id
        self.delete_llm_cache = delete_llm_cache
        self.message = "deleted"
        self.status_code = 200
        self.file_path = None


class ConcurrentFakeRAG:
    """LightRAG stand-in that models parse concurrency + a single-drain pipeline.

    drain_mode:
      * "self"  — ``apipeline_process_enqueue_documents`` drains pending→processed
                   itself (normal single-flow).
      * "owner" — the drain returns WITHOUT processing (models "another flow
                   holds busy") and schedules an out-of-band task to finish the
                   docs after ``owner_delay`` seconds. Exercises the read-back
                   poll that waits the concurrent drain out.
      * "cancel" — the drain marks docs failed with a 'User cancelled' marker,
                   modeling a cancel landing mid-drain.
    """

    def __init__(
        self,
        workspace: str,
        *,
        max_parallel_parse_mineru: int = 4,
        parse_delay: float = 0.03,
        parse_should_fail_for: set[str] | None = None,
        drain_mode: str = "self",
        owner_delay: float = 0.05,
    ):
        self.workspace = workspace
        self.embedding_dim = 768
        self.chunk_token_size = 512
        self.chunk_overlap_token_size = 64
        self.tiktoken_model_name = "gpt-4o-mini"
        self.summary_max_tokens = 800
        self.force_llm_summary_on_merge = False
        self.addon_params = {
            "chunker": {"strategy": "F"},
            "entity_types": ["concept"],
            "language": "en",
            "extraction": {"prompt_version": "v1"},
        }

        class _Embed:
            __name__ = "fake_embed"
            func_name = "fake_embed"

        self.embedding_func = _Embed()
        self.doc_status = FakeDocStatus()

        # Concurrency knobs / observability.
        self.max_parallel_parse_mineru = max_parallel_parse_mineru
        self._parse_delay = parse_delay
        self._parse_should_fail_for = parse_should_fail_for or set()
        self._drain_mode = drain_mode
        self._owner_delay = owner_delay

        self._active_parses = 0
        self.max_concurrent_parses = 0
        self.parse_calls: list[str] = []
        self.enqueue_calls: list[dict] = []
        self.process_enqueue_calls = 0
        self._owner_tasks: list[asyncio.Task] = []

    async def finalize_storages(self) -> None:
        for task in self._owner_tasks:
            task.cancel()
        return None

    async def adelete_by_doc_id(self, doc_id: str, *, delete_llm_cache: bool = False):
        self.doc_status.rows.pop(doc_id, None)
        return FakeDeletionResult(doc_id, delete_llm_cache)

    async def parse_native(self, doc_id, file_path, content_data):
        return await self._parse(doc_id, file_path)

    async def parse_mineru(self, doc_id, file_path, content_data):
        return await self._parse(doc_id, file_path)

    async def parse_docling(self, doc_id, file_path, content_data):
        return await self._parse(doc_id, file_path)

    async def _parse(self, doc_id, file_path):
        self._active_parses += 1
        self.max_concurrent_parses = max(
            self.max_concurrent_parses, self._active_parses
        )
        try:
            # Sleep to create an overlap window so concurrent tasks actually
            # coincide (and so the semaphore bound is observable).
            await asyncio.sleep(self._parse_delay)
            self.parse_calls.append(doc_id)
            name = Path(file_path).name
            if any(token in name for token in self._parse_should_fail_for):
                raise RuntimeError(f"parse exploded for {name}")
            source = Path(file_path)
            parsed_dir = source.parent / "__parsed__" / f"{source.name}.parsed"
            parsed_dir.mkdir(parents=True, exist_ok=True)
            blocks_path = parsed_dir / f"{source.stem}.blocks.jsonl"
            blocks_path.write_text(
                '{"type":"meta"}\n{"type":"content","content":"hello"}\n',
                encoding="utf-8",
            )
            return {
                "doc_id": doc_id,
                "file_path": str(source),
                "parse_format": "lightrag",
                "content": "hello",
                "blocks_path": str(blocks_path),
                "parse_stage_skipped": False,
            }
        finally:
            self._active_parses -= 1

    async def apipeline_enqueue_documents(
        self,
        input,
        *,
        ids,
        file_paths,
        track_id,
        docs_format,
        lightrag_document_paths,
        parse_engine=None,
        process_options=None,
    ):
        self.enqueue_calls.append(
            {
                "ids": list(ids),
                "file_paths": list(file_paths),
                "docs_format": docs_format,
                "lightrag_document_paths": list(lightrag_document_paths),
                "parse_engine": parse_engine,
                "process_options": process_options,
            }
        )
        for doc_id in ids:
            self.doc_status.stamp_pending(doc_id)
        return track_id

    async def apipeline_process_enqueue_documents(self):
        self.process_enqueue_calls += 1
        if self._drain_mode == "self":
            for doc_id, row in list(self.doc_status.rows.items()):
                if row.get("status") == "pending":
                    self.doc_status.stamp_processed(doc_id)
        elif self._drain_mode == "owner":
            # Model "another flow holds busy": return without draining, but an
            # out-of-band owner loop finishes the docs shortly. The read-back
            # poll must wait this out rather than failing the docs.
            pending = [
                d
                for d, r in self.doc_status.rows.items()
                if r.get("status") == "pending"
            ]

            async def _owner(pending_ids):
                await asyncio.sleep(self._owner_delay)
                for d in pending_ids:
                    self.doc_status.stamp_processed(d)

            self._owner_tasks.append(asyncio.create_task(_owner(pending)))
        elif self._drain_mode == "cancel":
            for doc_id, row in list(self.doc_status.rows.items()):
                if row.get("status") == "pending":
                    self.doc_status.stamp_failed(
                        doc_id, "User cancelled during merge 1/1: x.pdf"
                    )
        return None

    async def aquery_llm(self, query: str, *, param):
        return {"llm_response": {"content": "ok", "is_streaming": False}, "data": {}}

    async def aquery_data(self, query: str, *, param):
        return {"status": "success", "message": "ok", "data": {}, "metadata": {}}


class BuilderProbe:
    def __init__(self, **rag_kwargs):
        self.instances: dict[str, ConcurrentFakeRAG] = {}
        self._rag_kwargs = rag_kwargs

    async def build(self, record) -> ConcurrentFakeRAG:
        rag = ConcurrentFakeRAG(record.workspace, **self._rag_kwargs)
        self.instances[record.id] = rag
        return rag

    async def finalize(self, rag: LightRAGLike) -> None:
        return None


def _build_client(tmp_path: Path, **rag_kwargs):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kb.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    document_service = DocumentLifecycleService(
        kb_service, metadata_store, tmp_path / "inputs"
    )
    job_service = JobService(kb_service, metadata_store)
    index_service = IndexBuildService(document_service)
    probe = BuilderProbe(**rag_kwargs)
    registry = LightRAGInstanceRegistry(kb_service, probe.build, probe.finalize)
    app = FastAPI()
    app.state.document_service = document_service
    app.state.metadata_store = metadata_store
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
    return TestClient(app), probe


def _create_kb(client: TestClient, kb_id: str) -> dict:
    response = client.post("/kbs", json={"id": kb_id, "name": kb_id}, headers=_HEADERS)
    assert response.status_code == 200, response.text
    return response.json()


def _sync(client: TestClient, kb_id: str, items, *, idempotency_key=None) -> dict:
    files = [
        ("files", (filename, content, "application/pdf"))
        for _source_key, filename, content in items
    ]
    data = {"source_keys": [source_key for source_key, _f, _c in items]}
    params = {"parser_engine": "mineru", "process_options": "iF"}
    if idempotency_key is not None:
        params["idempotency_key"] = idempotency_key
    response = client.post(
        f"/kbs/{kb_id}/documents:sync",
        params=params,
        data=data,
        files=files,
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    return response.json()


def _wait(client: TestClient, kb_id: str, job_id: str) -> dict:
    response = client.post(
        f"/kbs/{kb_id}/jobs/{job_id}:wait?timeout_seconds=30", headers=_HEADERS
    )
    assert response.status_code == 200, response.text
    return response.json()


def _upload_auto(client: TestClient, kb_id: str, items) -> dict:
    files = [("files", (name, content, "application/pdf")) for name, content in items]
    response = client.post(
        f"/kbs/{kb_id}/documents:upload"
        "?auto_parse=true&auto_index=true&parser_engine=mineru&process_options=iF",
        files=files,
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# Route-level integration: the two-phase aggregate flows
# ---------------------------------------------------------------------------


def test_sync_four_docs_concurrent_run_to_ready(tmp_path):
    """Scenario 1: a 4-doc :sync drives every doc to ready, parsing them
    concurrently (max concurrent parses > 1 proves overlap), and indexing
    them through a SINGLE pipeline drain."""
    client, probe = _build_client(tmp_path, max_parallel_parse_mineru=4)
    _create_kb(client, "kb_c")
    items = [
        (f"manual/doc_{i}.pdf", f"doc_{i}.pdf", f"content {i}".encode())
        for i in range(4)
    ]
    job = _sync(client, "kb_c", items, idempotency_key="r1")
    final = _wait(client, "kb_c", job["id"])

    assert final["status"] == "succeeded"
    assert final["completed_items"] == 4
    assert {item["action"] for item in final["result"]["items"]} == {"created"}

    listing = client.get("/kbs/kb_c/documents", headers=_HEADERS).json()["documents"]
    assert {doc["status"] for doc in listing} == {"ready"}

    rag = probe.instances["kb_c"]
    assert len(rag.parse_calls) == 4
    # Concurrency actually happened (semaphore=4, 4 docs with an overlap delay).
    assert rag.max_concurrent_parses > 1
    # Scenario 5: the whole batch was indexed through ONE bulk enqueue + ONE drain.
    assert len(rag.enqueue_calls) == 1
    assert len(rag.enqueue_calls[0]["ids"]) == 4
    assert rag.process_enqueue_calls == 1


def test_sync_results_complete_regardless_of_completion_order(tmp_path):
    """Scenario 2: with per-doc parse delays staggering completion order, every
    input source_key still appears exactly once in the aggregated result."""
    client, probe = _build_client(tmp_path, max_parallel_parse_mineru=4, parse_delay=0.02)
    _create_kb(client, "kb_order")
    items = [
        (f"k/{i}.pdf", f"{i}.pdf", f"c{i}".encode()) for i in range(5)
    ]
    job = _sync(client, "kb_order", items, idempotency_key="r1")
    final = _wait(client, "kb_order", job["id"])

    assert final["status"] == "succeeded"
    result_keys = [item["source_key"] for item in final["result"]["items"]]
    assert sorted(result_keys) == sorted(k for k, _f, _c in items)
    assert len(result_keys) == len(set(result_keys)) == 5


def test_sync_single_doc_failure_isolated(tmp_path):
    """Scenario 3: one doc whose parse fails does not prevent the others from
    reaching ready; the job is partial-failure with exactly one failed item."""
    client, probe = _build_client(
        tmp_path, max_parallel_parse_mineru=4, parse_should_fail_for={"fail"}
    )
    _create_kb(client, "kb_iso")
    items = [
        ("ok/a.pdf", "a.pdf", b"a"),
        ("bad/fail.pdf", "fail.pdf", b"b"),
        ("ok/c.pdf", "c.pdf", b"c"),
        ("ok/d.pdf", "d.pdf", b"d"),
    ]
    job = _sync(client, "kb_iso", items, idempotency_key="r1")
    final = _wait(client, "kb_iso", job["id"])

    assert final["status"] == "failed"  # aggregate reports partial failure
    by_key = {item["source_key"]: item for item in final["result"]["items"]}
    assert by_key["bad/fail.pdf"]["status"] == "failed"
    assert {k for k in by_key if k != "bad/fail.pdf"} == {
        "ok/a.pdf",
        "ok/c.pdf",
        "ok/d.pdf",
    }
    # The three good docs still reached ready.
    listing = client.get("/kbs/kb_iso/documents", headers=_HEADERS).json()["documents"]
    ready = {doc["metadata"].get("source_key") for doc in listing if doc["status"] == "ready"}
    assert ready == {"ok/a.pdf", "ok/c.pdf", "ok/d.pdf"}


@pytest.mark.parametrize("limit", [1, 2])
def test_sync_respects_max_parallel_parse_mineru(tmp_path, limit):
    """Scenario 4: the Phase-1 semaphore caps concurrent parses at
    MAX_PARALLEL_PARSE_MINERU."""
    client, probe = _build_client(
        tmp_path, max_parallel_parse_mineru=limit, parse_delay=0.03
    )
    kb_id = f"kb_lim{limit}"
    _create_kb(client, kb_id)
    items = [(f"k/{i}.pdf", f"{i}.pdf", f"c{i}".encode()) for i in range(5)]
    job = _sync(client, kb_id, items, idempotency_key="r1")
    final = _wait(client, kb_id, job["id"])

    assert final["status"] == "succeeded"
    rag = probe.instances[kb_id]
    assert len(rag.parse_calls) == 5
    assert rag.max_concurrent_parses <= limit


def test_upload_auto_parse_single_drain(tmp_path):
    """Scenario 5 (other route): :upload?auto_parse&auto_index indexes all docs
    through exactly one pipeline drain."""
    client, probe = _build_client(tmp_path, max_parallel_parse_mineru=4)
    _create_kb(client, "kb_up")
    body = _upload_auto(
        client,
        "kb_up",
        [("a.pdf", b"a"), ("b.pdf", b"b"), ("c.pdf", b"c")],
    )
    final = _wait(client, "kb_up", body["job_id"])
    assert final["status"] == "succeeded"

    rag = probe.instances["kb_up"]
    assert len(rag.parse_calls) == 3
    assert len(rag.enqueue_calls) == 1
    assert len(rag.enqueue_calls[0]["ids"]) == 3
    assert rag.process_enqueue_calls == 1


def test_batch_parse_route_parses_documents_concurrently_and_respects_limit(tmp_path):
    client, probe = _build_client(
        tmp_path, max_parallel_parse_mineru=2, parse_delay=0.03
    )
    _create_kb(client, "kb_batch_parse_concurrent")
    upload = client.post(
        "/kbs/kb_batch_parse_concurrent/documents:upload",
        files=[
            ("files", (f"doc_{i}.pdf", f"content {i}".encode(), "application/pdf"))
            for i in range(4)
        ],
        headers=_HEADERS,
    )
    assert upload.status_code == 200, upload.text
    document_ids = [item["id"] for item in upload.json()["documents"]]

    response = client.post(
        "/kbs/kb_batch_parse_concurrent/documents:batch-parse",
        json={
            "document_ids": document_ids,
            "engine": "mineru",
            "process_options": "iF",
        },
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    final = _wait(client, "kb_batch_parse_concurrent", response.json()["job_id"])

    assert final["status"] == "succeeded"
    rag = probe.instances["kb_batch_parse_concurrent"]
    assert len(rag.parse_calls) == 4
    assert rag.max_concurrent_parses > 1
    assert rag.max_concurrent_parses <= 2


# ---------------------------------------------------------------------------
# Direct unit tests on IndexBuildService.run_build_batch (the race fixes)
# ---------------------------------------------------------------------------


def _document_record(doc_id: str, lr_id: str) -> DocumentRecord:
    return DocumentRecord(
        id=doc_id,
        kb_id="kb_unit",
        workspace="kb_unit",
        lightrag_doc_id=lr_id,
        source_type="file",
        source_name=f"{doc_id}.pdf",
        source_uri=f"{doc_id}.pdf",
        source_hash="sha256:src",
        content_type="application/pdf",
        size_bytes=10,
        parser_hash="sha256:parser",
        index_hash=None,
        status="parsed",
        enabled=True,
        archived=False,
        chunks_count=None,
        entity_count=None,
        relation_count=None,
        error_code=None,
        error_message=None,
        metadata={"parse_engine": "mineru"},
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        deleted_at=None,
    )


def _plan(doc_id: str, lr_id: str) -> IndexBuildPlan:
    return IndexBuildPlan(
        document=_document_record(doc_id, lr_id),
        sidecar_artifact=BuildArtifactReference(
            id=f"artifact-{doc_id}",
            artifact_type="sidecar",
            checksum=None,
            size_bytes=None,
            object_uri=None,
            object_prefix_uri=None,
            compatibility_locator="file:///sidecar/",
        ),
        blocks_artifact=None,
        expected_current_sidecar_artifact_id=None,
        expected_current_blocks_artifact_id=None,
        parser_hash="sha256:parser",
        index_hash="sha256:index",
        process_options="iF",
        force_rechunk=False,
        force_extract=False,
        force_embedding=False,
    )


def _execution(plan: IndexBuildPlan) -> IndexBuildExecution:
    return IndexBuildExecution(
        lease=None,
        runtime_sidecar_dir=Path("/sidecar"),
        runtime_sidecar_uri="file:///sidecar/",
        runtime_blocks_path=Path("/sidecar/paper.blocks.jsonl"),
        canonical_sidecar_locator=Path("/sidecar"),
        canonical_blocks_locator=Path("/sidecar/paper.blocks.jsonl"),
        expected_current_sidecar_artifact_id=None,
        expected_current_blocks_artifact_id=None,
        initial_sidecar_checksum="sha256:sidecar",
        initial_blocks_checksum="sha256:blocks",
    )


def _executions(plans: list[IndexBuildPlan]) -> dict[str, IndexBuildExecution]:
    return {
        plan.document.id: _execution(plan) for plan in plans if not plan.skipped
    }


def _service_with_fast_poll() -> IndexBuildService:
    # document_service is unused by run_build_batch; a sentinel keeps __init__ happy.
    svc = IndexBuildService(document_service=object())  # type: ignore[arg-type]
    svc._build_drain_poll = 0.01
    svc._build_drain_timeout = 5.0
    return svc


@pytest.mark.asyncio
async def test_run_build_batch_waits_out_concurrent_drain(tmp_path):
    """Fix A: when apipeline_process_enqueue_documents returns WITHOUT draining
    (another flow holds the pipeline busy flag), the read-back must POLL until
    the owning loop finishes the docs — not immediately mark them build_failed."""
    svc = _service_with_fast_poll()
    rag = ConcurrentFakeRAG("kb_unit", drain_mode="owner", owner_delay=0.05)
    plans = [_plan(f"doc_{i}", f"doc-{i}") for i in range(3)]

    results = await svc.run_build_batch(
        rag, plans, _executions(plans), job_id="job_x"
    )

    assert set(results) == {"doc_0", "doc_1", "doc_2"}
    for doc_id, run_result in results.items():
        assert "error_code" not in run_result, (doc_id, run_result)
        assert run_result["chunks_count"] == 4
    # Exactly one enqueue + one drain for the whole batch.
    assert len(rag.enqueue_calls) == 1
    assert rag.process_enqueue_calls == 1


@pytest.mark.asyncio
async def test_run_build_batch_classifies_mid_drain_cancel(tmp_path):
    """Fix D: a doc the pipeline marked failed with a 'User cancelled' marker is
    reported as cancelled (not a generic build failure)."""
    svc = _service_with_fast_poll()
    rag = ConcurrentFakeRAG("kb_unit", drain_mode="cancel")
    plans = [_plan("doc_0", "doc-0")]

    results = await svc.run_build_batch(
        rag, plans, _executions(plans), job_id="job_x"
    )

    assert results["doc_0"].get("cancelled") is True
    assert "error_code" not in results["doc_0"]


@pytest.mark.asyncio
async def test_run_build_batch_skips_skipped_plans(tmp_path):
    """A skipped plan (index_hash match) is finalized from the plan, never
    enqueued; mixed batches still enqueue only the runnable docs."""
    svc = _service_with_fast_poll()
    rag = ConcurrentFakeRAG("kb_unit", drain_mode="self")
    runnable = _plan("doc_0", "doc-0")
    skipped = _plan("doc_1", "doc-1")
    skipped.skipped = True
    skipped.skip_reason = "index_hash_match"

    plans = [runnable, skipped]
    results = await svc.run_build_batch(
        rag, plans, _executions(plans), job_id="job_x"
    )

    assert results["doc_1"]["skipped"] is True
    assert results["doc_0"]["chunks_count"] == 4
    # Only the runnable doc was enqueued.
    assert len(rag.enqueue_calls) == 1
    assert rag.enqueue_calls[0]["ids"] == ["doc-0"]
