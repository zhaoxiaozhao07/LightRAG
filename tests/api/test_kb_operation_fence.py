from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

from lightrag.api.kb_operation_fence import (
    ConflictingKBWriteTargetsError,
    KBWriteAdmissionMiddleware,
    kb_write_target_from_scope,
)
from lightrag.api.kb_service import KnowledgeBaseService
from lightrag.api.metadata_store import SQLiteMetadataStore


pytestmark = pytest.mark.offline


def _scope(method: str, path: str, *, root_path: str = "") -> dict:
    return {
        "type": "http",
        "method": method,
        "path": path,
        "root_path": root_path,
    }


@pytest.mark.parametrize(
    ("scope", "expected_kb_id"),
    [
        (_scope("POST", "/kbs/kb_query/query"), "kb_query"),
        (_scope("POST", "/kbs/kb_retrieve/retrieve"), "kb_retrieve"),
        (
            _scope(
                "PATCH",
                "/site/root/kbs/kb_root/documents/doc_1",
                root_path="/site/root",
            ),
            "kb_root",
        ),
        (
            _scope(
                "POST",
                "/query",
                root_path="/tenant/kbs/kb_root_path_identity",
            ),
            None,
        ),
        (
            _scope(
                "POST",
                "/kbs/kb_route_identity/query",
                root_path="/tenant/kbs/kb_root_path_identity",
            ),
            "kb_route_identity",
        ),
        (
            _scope(
                "POST",
                "/tenant/kbs/kb_root_path_identity/kbs/kb_route_identity/query",
                root_path="/tenant/kbs/kb_root_path_identity",
            ),
            "kb_route_identity",
        ),
        (
            _scope(
                "POST",
                "/tenant/kbs/kb_root_path_identity",
                root_path="/tenant/kbs/kb_root_path_identity",
            ),
            None,
        ),
        (
            _scope("PUT", "/auth/me/kbs/kb_settings/query-settings"),
            "kb_settings",
        ),
        # Encoded separators are decoded before matching so they cannot evade
        # a route that the ASGI server/router sees as /kbs/{kb_id}/....
        (_scope("POST", "/api/kbs%2Fkb_encoded/query", root_path="/api"), "kb_encoded"),
        (_scope("POST", "/kbs/kb_rebuild:rebuild"), "kb_rebuild"),
        (_scope("POST", "/kbs/kb_reparse%3Areparse"), "kb_reparse"),
        (_scope("POST", "/kbs/kb_restore:restore/extra"), "kb_restore"),
        (_scope("POST", "/kbsX/kb_false/query"), None),
        (_scope("POST", "/kbsX%2Fkb_false/query"), None),
        (_scope("POST", "/kbs"), None),
        (_scope("DELETE", "/kbs/kb_lifecycle"), None),
        (_scope("POST", "/kbs/kb_lifecycle:restore"), None),
        (_scope("POST", "/kbs/kb_lifecycle%3Arestore"), None),
        (_scope("GET", "/kbs/kb_read/query"), None),
        (_scope("HEAD", "/kbs/kb_read/query"), None),
    ],
)
def test_kb_write_target_exact_path_root_prefix_and_exclusions(
    scope: dict,
    expected_kb_id: str | None,
):
    target = kb_write_target_from_scope(scope)  # type: ignore[arg-type]
    assert (target.kb_id if target is not None else None) == expected_kb_id


def test_kb_write_target_rejects_conflicting_route_identities():
    with pytest.raises(ConflictingKBWriteTargetsError) as exc_info:
        kb_write_target_from_scope(
            _scope("POST", "/kbs/kb_alpha/proxy/kbs/kb_beta/query")  # type: ignore[arg-type]
        )

    assert exc_info.value.kb_ids == ("kb_alpha", "kb_beta")
    repeated = kb_write_target_from_scope(
        _scope("POST", "/kbs/kb_alpha/proxy/kbs/kb_alpha/query")  # type: ignore[arg-type]
    )
    assert repeated is not None and repeated.kb_id == "kb_alpha"

    with pytest.raises(ConflictingKBWriteTargetsError) as action_conflict:
        kb_write_target_from_scope(
            _scope(
                "POST",
                "/kbs/kb_alpha:rebuild/proxy/kbs/kb_beta:reparse",
            )  # type: ignore[arg-type]
        )
    assert action_conflict.value.kb_ids == ("kb_alpha", "kb_beta")


@pytest.mark.asyncio
async def test_middleware_rejects_conflicting_route_identities_before_dispatch():
    class UnusedKBService:
        async def get(self, *_args, **_kwargs):
            raise AssertionError("conflicting target must not reach KB lookup")

    downstream_called = False
    app = FastAPI()
    app.add_middleware(
        KBWriteAdmissionMiddleware,
        kb_service=UnusedKBService(),
        metadata_store=object(),
    )

    @app.post("/{route:path}")
    async def catch_all(route: str):
        nonlocal downstream_called
        downstream_called = True
        return {"route": route}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post("/kbs/kb_alpha/proxy/kbs/kb_beta/query")

    assert response.status_code == 400
    assert response.json() == {
        "detail": {
            "error_code": "conflicting_kb_write_targets",
            "kb_ids": ["kb_alpha", "kb_beta"],
            "message": "Request path contains conflicting KB identities",
        }
    }
    assert downstream_called is False


async def _make_environment(tmp_path: Path, kb_id: str):
    kb_service = KnowledgeBaseService(tmp_path / f"{kb_id}.json")
    await kb_service.initialize()
    record = await kb_service.create(kb_id=kb_id, name=kb_id)
    db_path = tmp_path / f"{kb_id}.sqlite3"
    request_store = SQLiteMetadataStore(db_path)
    delete_store = SQLiteMetadataStore(db_path)
    await request_store.initialize()
    await delete_store.initialize()
    await request_store.activate_kb_generation(record.id, record.generation)
    return kb_service, request_store, delete_store, record


@pytest.mark.asyncio
async def test_middleware_stamps_scope_state_and_preserves_missing_kb_404(
    tmp_path: Path,
):
    kb_service, store, _delete_store, record = await _make_environment(
        tmp_path, "kb_scope_state"
    )
    app = FastAPI()
    app.add_middleware(
        KBWriteAdmissionMiddleware,
        kb_service=kb_service,
        metadata_store=store,
    )

    @app.post("/kbs/{kb_id}/query")
    async def query(kb_id: str, request: Request):
        if kb_id != record.id:
            raise HTTPException(status_code=404, detail="missing")
        return {
            "kb_id": request.state.kb_id,
            "kb_generation": request.state.kb_generation,
        }

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        guarded = await client.post(f"/kbs/{record.id}/query")
        missing = await client.post("/kbs/kb_missing/query")

    assert guarded.status_code == 200
    assert guarded.json() == {
        "kb_id": record.id,
        "kb_generation": record.generation,
    }
    assert missing.status_code == 404


@pytest.mark.asyncio
async def test_response_background_task_holds_shared_guard_until_complete(
    tmp_path: Path,
):
    kb_service, request_store, delete_store, record = await _make_environment(
        tmp_path, "kb_background_guard"
    )
    app = FastAPI()
    app.add_middleware(
        KBWriteAdmissionMiddleware,
        kb_service=kb_service,
        metadata_store=request_store,
    )
    background_entered = asyncio.Event()
    release_background = asyncio.Event()
    exclusive_entered = asyncio.Event()

    @app.post("/kbs/{kb_id}/documents:upload")
    async def upload(kb_id: str, background_tasks: BackgroundTasks):
        assert kb_id == record.id

        async def blocking_stage() -> None:
            background_entered.set()
            await release_background.wait()

        background_tasks.add_task(blocking_stage)
        return {"status": "staging"}

    async def delete_attempt() -> None:
        async with delete_store.kb_deletion_guard(
            record.id,
            record.generation,
            "job_delete_background_guard",
        ):
            exclusive_entered.set()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        request_task = asyncio.create_task(
            client.post(f"/kbs/{record.id}/documents:upload")
        )
        await asyncio.wait_for(background_entered.wait(), timeout=2.0)
        delete_task = asyncio.create_task(delete_attempt())
        await asyncio.sleep(0.1)
        assert not exclusive_entered.is_set()

        release_background.set()
        response = await asyncio.wait_for(request_task, timeout=2.0)
        await asyncio.wait_for(exclusive_entered.wait(), timeout=2.0)
        await asyncio.wait_for(delete_task, timeout=2.0)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_rebuild_action_background_task_holds_shared_guard_until_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    kb_service, request_store, delete_store, record = await _make_environment(
        tmp_path, "kb_rebuild_background_guard"
    )
    from lightrag.api.index_build_service import BatchIndexBuildPlan, IndexBuildPlan
    from lightrag.api.kb_service import utc_now_iso
    from lightrag.api.metadata_store import DocumentRecord, JobRecord
    from lightrag.api.routers import kb_document_routes

    now = utc_now_iso()
    document = DocumentRecord(
        id="doc_rebuild_guard",
        kb_id=record.id,
        workspace=record.workspace,
        lightrag_doc_id="doc-rag-rebuild-guard",
        source_type="upload",
        source_name="rebuild.pdf",
        source_uri="/inputs/rebuild.pdf",
        source_hash="sha256:rebuild",
        content_type="application/pdf",
        size_bytes=1,
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
        metadata={},
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )
    plan = IndexBuildPlan(
        document=document,
        sidecar_uri=None,
        blocks_path=None,
        parser_hash="sha256:parser",
        index_hash="sha256:index",
        process_options="",
        force_rechunk=True,
        force_extract=True,
        force_embedding=True,
    )
    job = JobRecord(
        id="job_rebuild_guard",
        kb_id=record.id,
        workspace=record.workspace,
        batch_id="batch_rebuild_guard",
        document_id=None,
        job_type="build_kg",
        status="queued",
        stage="building",
        progress=0.0,
        total_items=1,
        completed_items=0,
        failed_items=0,
        idempotency_key=None,
        config_version_id=None,
        config_hash=None,
        retry_count=0,
        max_retries=3,
        payload={"kb_generation": record.generation},
        result=None,
        error_code=None,
        error_message=None,
        created_at=now,
        updated_at=now,
        queued_at=now,
        started_at=None,
        finished_at=None,
        cancelled_at=None,
    )

    class FakeDocumentService:
        async def list_documents(self, kb_id: str, *, status: str, **_kwargs):
            assert kb_id == record.id
            return ([document], 1) if status == "parsed" else ([], 0)

        async def get_documents_by_ids(self, kb_id: str, document_ids: list[str]):
            assert kb_id == record.id
            return [document] if document.id in document_ids else []

    class FakeJobService:
        async def create_batch_build_job_once(self, *_args, **_kwargs):
            return job, True

        async def transition_job(self, _kb_id: str, _job_id: str, **values):
            for key, value in values.items():
                if hasattr(job, key) and value is not None:
                    setattr(job, key, value)
            return job

        async def update_job_progress(self, *_args, **_kwargs):
            return job

    class FakeRegistry:
        async def get(self, _kb_id: str):
            return type("FakeRAG", (), {"workspace": record.workspace})()

    class FakeIndexService:
        async def create_batch_build_plan(self, *_args, **_kwargs):
            return BatchIndexBuildPlan(
                batch_id="batch_rebuild_guard",
                plans=[plan],
            )

        async def claim_batch_build_queued(self, *_args, **_kwargs):
            return [document], []

        async def fail_build(self, *_args, **_kwargs):
            return None

    app = FastAPI()
    app.add_middleware(
        KBWriteAdmissionMiddleware,
        kb_service=kb_service,
        metadata_store=request_store,
    )
    app.include_router(
        kb_document_routes.create_kb_document_routes(
            FakeDocumentService(),  # type: ignore[arg-type]
            FakeJobService(),  # type: ignore[arg-type]
            registry=FakeRegistry(),  # type: ignore[arg-type]
            index_service=FakeIndexService(),  # type: ignore[arg-type]
        )
    )
    background_entered = asyncio.Event()
    release_background = asyncio.Event()
    exclusive_entered = asyncio.Event()

    async def blocking_execute_build_plan(**_kwargs):
        background_entered.set()
        await release_background.wait()
        return {
            "document_id": document.id,
            "status": "succeeded",
            "skipped": False,
            "index_hash": plan.index_hash,
        }

    monkeypatch.setattr(
        kb_document_routes,
        "_execute_build_plan",
        blocking_execute_build_plan,
    )

    async def exclusive_attempt() -> None:
        async with delete_store.kb_exclusive_operation_guard(record.id):
            exclusive_entered.set()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        request_task = asyncio.create_task(
            client.post(f"/kbs/{record.id}:rebuild", json={})
        )
        await asyncio.wait_for(background_entered.wait(), timeout=2.0)
        exclusive_task = asyncio.create_task(exclusive_attempt())
        await asyncio.sleep(0.1)
        assert not exclusive_entered.is_set()

        release_background.set()
        response = await asyncio.wait_for(request_task, timeout=2.0)
        await asyncio.wait_for(exclusive_entered.wait(), timeout=2.0)
        await asyncio.wait_for(exclusive_task, timeout=2.0)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_downstream_asgi_exception_releases_shared_guard(tmp_path: Path):
    kb_service, request_store, delete_store, record = await _make_environment(
        tmp_path, "kb_exception_guard"
    )
    app = FastAPI()
    app.add_middleware(
        KBWriteAdmissionMiddleware,
        kb_service=kb_service,
        metadata_store=request_store,
    )

    @app.post("/kbs/{kb_id}/explode")
    async def explode(kb_id: str):
        assert kb_id == record.id
        raise RuntimeError("response exploded")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://test",
    ) as client:
        with pytest.raises(RuntimeError, match="response exploded"):
            await client.post(f"/kbs/{record.id}/explode")

    async def acquire_exclusive() -> None:
        async with delete_store.kb_deletion_guard(
            record.id,
            record.generation,
            "job_delete_exception_guard",
        ):
            return None

    await asyncio.wait_for(acquire_exclusive(), timeout=2.0)
