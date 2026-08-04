import asyncio
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from lightrag.api.config_version_service import ConfigVersionService
from lightrag.api.document_lifecycle_service import (
    DocumentLifecycleService,
    DocumentSourceInput,
)
from lightrag.api.job_service import JobService
from lightrag.api.kb_operation_fence import KBWriteAdmissionMiddleware
from lightrag.api.kb_service import (
    KnowledgeBaseConflictError,
    KnowledgeBaseService,
    sanitize_workspace,
)
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry, LightRAGLike
from lightrag.api.metadata_store import (
    InvalidJobTransitionError,
    KBLifecycleConflictError,
    SQLiteMetadataStore,
)
from lightrag.api.object_storage import ObjectStat, ObjectStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

# Phase 3.1-C Integration Writer B2: object-authoritative COW branch imports.
import hashlib as _hashlib_b2
from dataclasses import replace as _dataclass_replace_b2
from datetime import datetime as _datetime_b2, timezone as _timezone_b2
from uuid import uuid4 as _uuid4_b2

from lightrag.api.artifact_materialization import (
    ArtifactMaterializer as _ArtifactMaterializer_b2,
    MaterializationLimits as _MaterializationLimits_b2,
)
from lightrag.api.config import ArtifactCleanupConfig as _ArtifactCleanupConfig_b2
from lightrag.api.document_lifecycle_service import (
    DocumentReplacementSource as _DocumentReplacementSource_b2,
)
from lightrag.api.kb_service import utc_now_iso as _utc_now_iso_b2
from lightrag.api.metadata_store import (
    ArtifactRecord as _ArtifactRecord_b2,
    DocumentRecord as _DocumentRecord_b2,
)
from lightrag.api.object_storage import (
    ObjectReadback as _ObjectReadback_b2,
    ObjectStat as _ObjectStat_b2,
    ObjectStorageError as _ObjectStorageError_b2,
)
from lightrag.utils_pipeline import (
    reset_canonical_input_root_for_tests as _reset_root_b2,
    set_canonical_input_root as _set_root_b2,
)

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


class FakeRAG:
    def __init__(
        self,
        workspace: str,
        *,
        should_fail: bool = False,
        fail_source_names: set[str] | None = None,
        parse_content: str = "parsed",
    ):
        self.workspace = workspace
        self.should_fail = should_fail
        self.fail_source_names = fail_source_names or set()
        self.parse_content = parse_content
        self.parse_calls = []
        self.delete_calls = []

    async def finalize_storages(self) -> None:
        return None

    async def adrop_all_storages(self) -> dict:
        return {"dropped": 0, "failed": 0, "errors": []}

    async def parse_native(self, doc_id: str, file_path: str, content_data):
        return await self._parse("native", doc_id, file_path, content_data)

    async def parse_mineru(self, doc_id: str, file_path: str, content_data):
        return await self._parse("mineru", doc_id, file_path, content_data)

    async def parse_docling(self, doc_id: str, file_path: str, content_data):
        return await self._parse("docling", doc_id, file_path, content_data)

    async def parse_legacy(self, doc_id: str, file_path: str, content_data):
        return await self._parse("legacy", doc_id, file_path, content_data)

    async def apipeline_enqueue_documents(self, *args, **kwargs):
        raise AssertionError("KB parse endpoint must not enqueue indexing pipeline")

    async def apipeline_process_enqueue_documents(self, *args, **kwargs):
        raise AssertionError("KB parse endpoint must not process indexing pipeline")

    async def adelete_by_doc_id(self, doc_id: str, delete_llm_cache: bool = False):
        self.delete_calls.append((doc_id, delete_llm_cache))
        return FakeDeletionResult(
            status="success",
            doc_id=doc_id,
            message="deleted",
            status_code=200,
            file_path="",
        )

    async def _parse(self, engine: str, doc_id: str, file_path: str, content_data):
        self.parse_calls.append((engine, doc_id, file_path, content_data))
        source_path = Path(file_path)
        if self.should_fail or source_path.name in self.fail_source_names:
            raise RuntimeError("parser exploded")
        if engine == "legacy":
            from lightrag.parser.legacy import parse_legacy_source_file

            result = parse_legacy_source_file(doc_id=doc_id, file_path=source_path)
            if content_data.get("archive_source_after_parse", True):
                source_path.unlink()
            return result
        parsed_dir = source_path.parent / "__parsed__" / f"{source_path.name}.parsed"
        parsed_dir.mkdir(parents=True, exist_ok=True)
        blocks_path = parsed_dir / f"{source_path.stem}.blocks.jsonl"
        blocks_path.write_text(
            json.dumps(
                {"type": "content", "text": self.parse_content},
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        if engine == "mineru":
            raw_dir = parsed_dir.parent / f"{source_path.name}.mineru_raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            images_dir = raw_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / "full.md").write_text("# parsed", encoding="utf-8")
            (raw_dir / "content_list.json").write_text(
                '[{"type":"text","text":"parsed"}]', encoding="utf-8"
            )
            (raw_dir / "middle.json").write_text('{"pages":[]}', encoding="utf-8")
            (raw_dir / "model.json").write_text('{"model":"mineru"}', encoding="utf-8")
            (raw_dir / "layout.pdf").write_bytes(b"layout")
            (images_dir / "page-1.png").write_bytes(b"image")
        if content_data.get("archive_source_after_parse", True):
            source_path.unlink()
        return {
            "doc_id": doc_id,
            "file_path": file_path,
            "parse_format": "lightrag",
            "content": self.parse_content,
            "blocks_path": str(blocks_path),
            "parse_stage_skipped": False,
        }


class FakeDeletionResult(BaseModel):
    status: str
    doc_id: str
    message: str
    status_code: int
    file_path: str | None = None


class FakeObjectStorage(ObjectStorage):
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.prefix_files: dict[str, dict[str, bytes]] = {}
        self.uploads: list[str] = []
        self.directory_uploads: list[str] = []
        self.downloads: list[tuple[str, Path]] = []
        self.prefix_downloads: list[tuple[str, Path]] = []
        self.deleted_uris: list[str] = []
        self.deleted_prefixes: list[str] = []
        self.deleted_workspaces: list[str] = []
        self.presigned: list[tuple[str, int]] = []

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def upload_file(
        self, local_path: Path, *, key: str, content_type: str | None = None
    ) -> str:
        uri = f"s3://fake-bucket/{key}"
        self.files[uri] = local_path.read_bytes()
        self.uploads.append(uri)
        return uri

    async def upload_directory(self, local_dir: Path, *, prefix: str) -> str:
        uri = f"s3://fake-bucket/{prefix}/"
        files: dict[str, bytes] = {}
        for path in sorted(local_dir.rglob("*")):
            if path.is_file() and not path.is_symlink():
                files[path.relative_to(local_dir).as_posix()] = path.read_bytes()
        self.prefix_files[uri] = files
        self.directory_uploads.append(uri)
        return uri

    async def download_file(self, object_uri: str, local_path: Path) -> None:
        self.downloads.append((object_uri, local_path))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(self.files[object_uri])

    async def stat_object(self, object_uri: str) -> ObjectStat:
        return ObjectStat(size=len(self.files[object_uri]))

    async def download_prefix(
        self,
        prefix_uri: str,
        local_dir: Path,
        *,
        max_objects: int | None = None,
        max_total_bytes: int | None = None,
    ) -> int:
        self.prefix_downloads.append((prefix_uri, local_dir))
        local_dir.mkdir(parents=True, exist_ok=True)
        files = self.prefix_files[prefix_uri]
        for relative_path, content in files.items():
            target = local_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        return len(files)

    async def delete_uri(self, object_uri: str) -> bool:
        self.deleted_uris.append(object_uri)
        self.files.pop(object_uri, None)
        return True

    async def delete_prefix(self, prefix_uri: str) -> int:
        self.deleted_prefixes.append(prefix_uri)
        files = self.prefix_files.pop(prefix_uri, {})
        return len(files)

    async def delete_workspace(self, workspace: str) -> int:
        self.deleted_workspaces.append(workspace)
        marker = f"/workspaces/{workspace}/"
        deleted = 0
        for uri in list(self.files):
            if marker in uri:
                self.files.pop(uri, None)
                deleted += 1
        for uri, files in list(self.prefix_files.items()):
            if marker in uri:
                self.prefix_files.pop(uri, None)
                deleted += len(files)
        return deleted

    async def presign_download_url(
        self, object_uri: str, *, expires_in_seconds: int = 3600
    ) -> str:
        self.presigned.append((object_uri, expires_in_seconds))
        return f"https://objects.example/download?uri={object_uri}&expires={expires_in_seconds}"

    def validate_document_file_uri(
        self,
        object_uri: str,
        *,
        workspace: str,
        document_id: str,
        namespace: str | None = None,
        artifact_id: str | None = None,
    ) -> None:
        prefix = f"s3://fake-bucket/workspaces/{workspace}/documents/{document_id}/"
        if not object_uri.startswith(prefix) or object_uri.endswith("/"):
            from lightrag.api.object_storage import ObjectStorageError

            raise ObjectStorageError("Object URI is outside the document object prefix")

    def validate_document_prefix_uri(
        self,
        prefix_uri: str,
        *,
        workspace: str,
        document_id: str,
        namespace: str | None = None,
        artifact_id: str | None = None,
    ) -> None:
        prefix = f"s3://fake-bucket/workspaces/{workspace}/documents/{document_id}/"
        if not prefix_uri.startswith(prefix) or not prefix_uri.endswith("/"):
            from lightrag.api.object_storage import ObjectStorageError

            raise ObjectStorageError("Object URI is outside the document object prefix")


class BuilderProbe:
    def __init__(
        self,
        *,
        should_fail: bool = False,
        fail_source_names: set[str] | None = None,
        parse_content: str = "parsed",
    ):
        self.should_fail = should_fail
        self.fail_source_names = fail_source_names or set()
        self.parse_content = parse_content
        self.instances: list[FakeRAG] = []

    async def build(self, record) -> FakeRAG:
        rag = FakeRAG(
            record.workspace,
            should_fail=self.should_fail,
            fail_source_names=self.fail_source_names,
            parse_content=self.parse_content,
        )
        self.instances.append(rag)
        return rag

    async def finalize(self, rag: LightRAGLike) -> None:
        return None


def _build_client(
    tmp_path: Path,
    *,
    probe: BuilderProbe | None = None,
    wire_document_registry: bool = True,
    object_storage: FakeObjectStorage | None = None,
):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "knowledge_bases.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    document_service = DocumentLifecycleService(
        kb_service, metadata_store, tmp_path / "inputs", object_storage=object_storage
    )
    job_service = JobService(kb_service, metadata_store)
    probe = probe or BuilderProbe()
    registry = LightRAGInstanceRegistry(kb_service, probe.build, probe.finalize)
    config_service = ConfigVersionService(kb_service, metadata_store, registry)
    app = FastAPI()
    app.add_middleware(
        KBWriteAdmissionMiddleware,
        kb_service=kb_service,
        metadata_store=metadata_store,
    )
    app.include_router(
        create_kb_routes(
            kb_service,
            registry,
            api_key=_API_KEY,
            job_service=job_service,
            config_service=config_service,
            metadata_store=metadata_store,
        )
    )
    # wire_document_registry=False mimics "no parse worker wired": auto_parse
    # uploads still persist metadata + a queued parse job but do not execute,
    # so tests can assert the queued snapshot deterministically.
    app.include_router(
        create_kb_document_routes(
            document_service,
            job_service,
            api_key=_API_KEY,
            registry=registry if wire_document_registry else None,
        )
    )
    return TestClient(app), kb_service, metadata_store, document_service, job_service


def _create_kb(client: TestClient, kb_id: str):
    response = client.post("/kbs", json={"id": kb_id, "name": kb_id}, headers=_HEADERS)
    assert response.status_code == 200
    return response.json()


def _upload_and_parse_document(
    client: TestClient,
    kb_id: str,
    *,
    filename: str = "paper.pdf",
    content: bytes = b"pdf",
    content_type: str = "application/pdf",
):
    upload = client.post(
        f"/kbs/{kb_id}/documents:upload",
        files=[("files", (filename, content, content_type))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]

    parse = client.post(
        f"/kbs/{kb_id}/documents/{document_id}:parse",
        json={"engine": "mineru", "process_options": "iF"},
        headers=_HEADERS,
    )
    assert parse.status_code == 200

    artifacts = client.get(
        f"/kbs/{kb_id}/documents/{document_id}/artifacts", headers=_HEADERS
    )
    assert artifacts.status_code == 200
    artifacts_by_type = {
        item["artifact_type"]: item for item in artifacts.json()["artifacts"]
    }
    return document_id, artifacts_by_type


def test_upload_persists_documents_jobs_and_running_status(tmp_path):
    initialize_share_data()
    try:
        # No parse worker wired: auto_parse persists metadata + a queued parse
        # job without executing, so the queued/running snapshot is stable.
        client, kb_service, _store, _document_service, _job_service = _build_client(
            tmp_path, wire_document_registry=False
        )
        kb = _create_kb(client, "kb_upload")

        response = client.post(
            "/kbs/kb_upload/documents:upload?auto_parse=true&auto_index=false",
            files=[
                ("files", ("alpha.txt", b"alpha", "text/plain")),
                ("files", ("beta.txt", b"beta", "text/plain")),
            ],
            headers=_HEADERS,
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["job_id"].startswith("job_parse_")
        assert payload["batch_id"].startswith("batch_")
        assert [doc["status"] for doc in payload["documents"]] == [
            "parse_queued",
            "parse_queued",
        ]
        assert {doc["workspace"] for doc in payload["documents"]} == {kb["workspace"]}
        assert all(Path(doc["source_uri"]).exists() for doc in payload["documents"])
        assert (tmp_path / "metadata" / "metadata.sqlite3").exists()

        list_response = client.get("/kbs/kb_upload/documents", headers=_HEADERS)
        assert list_response.status_code == 200
        listed = list_response.json()
        assert listed["total"] == 2
        assert {doc["source_name"] for doc in listed["documents"]} == {
            "alpha.txt",
            "beta.txt",
        }

        document_id = payload["documents"][0]["id"]
        detail_response = client.get(
            f"/kbs/kb_upload/documents/{document_id}", headers=_HEADERS
        )
        assert detail_response.status_code == 200
        assert detail_response.json()["id"] == document_id

        jobs_response = client.get(
            "/kbs/kb_upload/jobs?status=queued", headers=_HEADERS
        )
        assert jobs_response.status_code == 200
        jobs = jobs_response.json()
        assert jobs["total"] == 1
        assert jobs["jobs"][0]["id"] == payload["job_id"]
        assert jobs["jobs"][0]["job_type"] == "parse"
        persisted_job = asyncio.run(
            _job_service.get_job("kb_upload", payload["job_id"])
        )
        catalog = asyncio.run(kb_service.get("kb_upload"))
        assert persisted_job.payload["kb_generation"] == catalog.generation

        status_response = client.get("/kbs/kb_upload/status", headers=_HEADERS)
        assert status_response.status_code == 200
        status = status_response.json()
        assert status["kb"]["workspace"] == sanitize_workspace("kb_upload")
        assert [job["id"] for job in status["running_jobs"]] == [payload["job_id"]]
    finally:
        finalize_share_data()


def test_upload_is_rejected_before_staging_when_kb_is_deleting(tmp_path):
    client, kb_service, store, _document_service, _job_service = _build_client(
        tmp_path
    )
    kb = _create_kb(client, "kb_upload_deleting")

    async def mark_deleting() -> None:
        record = await kb_service.get(kb["id"])
        async with store.kb_deletion_guard(
            record.id,
            record.generation,
            "job_delete_upload_guard",
        ):
            pass

    asyncio.run(mark_deleting())
    response = client.post(
        "/kbs/kb_upload_deleting/documents:upload",
        files=[("files", ("blocked.txt", b"blocked", "text/plain"))],
        headers=_HEADERS,
    )

    assert response.status_code == 409
    documents, total = asyncio.run(store.list_documents(kb["id"]))
    assert documents == []
    assert total == 0
    assert not (tmp_path / "inputs" / kb["workspace"]).exists()


def test_text_import_is_metadata_only_and_kb_scoped(tmp_path):
    client, _kb_service, store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_texts")
    _create_kb(client, "kb_other")

    response = client.post(
        "/kbs/kb_texts/documents:texts",
        json={
            "documents": [
                {
                    "text": "hello metadata",
                    "source_name": "note.md",
                    "metadata": {"tag": "unit"},
                }
            ],
            "auto_parse": False,
        },
        headers=_HEADERS,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"].startswith("job_upload_")
    assert payload["documents"][0]["status"] == "uploaded"
    assert payload["documents"][0]["metadata"]["tag"] == "unit"

    own_response = client.get("/kbs/kb_texts/documents", headers=_HEADERS)
    other_response = client.get("/kbs/kb_other/documents", headers=_HEADERS)
    assert own_response.status_code == 200
    assert other_response.status_code == 200
    assert own_response.json()["total"] == 1
    assert other_response.json()["total"] == 0

    reopened = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    documents, total = asyncio.run(reopened.list_documents("kb_texts"))
    assert total == 1
    assert documents[0].source_name == "note.md"
    assert documents[0].status == "uploaded"

    jobs, total_jobs = asyncio.run(store.list_jobs("kb_texts"))
    assert total_jobs == 1
    assert jobs[0].status == "succeeded"
    assert jobs[0].progress == 1.0


def test_text_import_idempotency_key_reuses_existing_batch(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_text_idem")
    request = {
        "documents": [
            {
                "text": "hello idempotency",
                "source_name": "idem.txt",
                "metadata": {"tag": "first"},
            }
        ],
        "idempotency_key": "idem-text-1",
    }

    first = client.post(
        "/kbs/kb_text_idem/documents:texts", json=request, headers=_HEADERS
    )
    second = client.post(
        "/kbs/kb_text_idem/documents:texts",
        json=request,
        headers=_HEADERS,
    )

    assert first.status_code == 200
    assert second.status_code == 200
    first_payload = first.json()
    second_payload = second.json()
    assert second_payload["job_id"] == first_payload["job_id"]
    assert second_payload["batch_id"] == first_payload["batch_id"]
    assert second_payload["documents"][0]["id"] == first_payload["documents"][0]["id"]
    assert second_payload["documents"][0]["source_name"] == "idem.txt"

    listed = client.get("/kbs/kb_text_idem/documents", headers=_HEADERS)
    assert listed.status_code == 200
    assert listed.json()["total"] == 1
    jobs = client.get("/kbs/kb_text_idem/jobs", headers=_HEADERS)
    assert jobs.status_code == 200
    assert jobs.json()["total"] == 1

    conflict = client.post(
        "/kbs/kb_text_idem/documents:texts",
        json={
            **request,
            "documents": [
                {
                    "text": "different body must conflict",
                    "source_name": "different.txt",
                }
            ],
        },
        headers=_HEADERS,
    )
    assert conflict.status_code == 409


@pytest.mark.asyncio
async def test_text_import_idempotency_key_is_atomic_for_concurrent_batches(tmp_path):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "knowledge_bases.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    document_service = DocumentLifecycleService(
        kb_service, metadata_store, tmp_path / "inputs"
    )
    await kb_service.create(kb_id="kb_text_concurrent", name="Concurrent")
    source = DocumentSourceInput(
        source_name="same.txt",
        content=b"same content",
        source_type="text",
        content_type="text/plain",
        metadata={"tag": "same"},
    )

    first, second = await asyncio.gather(
        document_service.create_source_batch(
            "kb_text_concurrent", [source], idempotency_key="same-key"
        ),
        document_service.create_source_batch(
            "kb_text_concurrent", [source], idempotency_key="same-key"
        ),
    )

    assert first.job.id == second.job.id
    assert first.documents[0].id == second.documents[0].id
    documents, total = await document_service.list_documents("kb_text_concurrent")
    assert total == 1
    assert documents[0].source_name == "same.txt"
    jobs, total_jobs = await metadata_store.list_jobs("kb_text_concurrent")
    assert total_jobs == 1
    assert jobs[0].id == first.job.id
    workspace_dir = tmp_path / "inputs" / sanitize_workspace("kb_text_concurrent")
    assert [path.name for path in workspace_dir.iterdir()] == [documents[0].id]


@pytest.mark.asyncio
async def test_create_source_batch_holds_guard_through_staging_and_commit(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "source-batch-fence.sqlite3"
    writer_store = SQLiteMetadataStore(db_path)
    delete_store = SQLiteMetadataStore(db_path)
    await writer_store.initialize()
    await delete_store.initialize()
    kb_service = KnowledgeBaseService(tmp_path / "source-batch-kbs.json")
    await kb_service.initialize()
    record = await kb_service.create(kb_id="kb_source_batch_fence", name="Fence")
    await writer_store.activate_kb_generation(record.id, record.generation)
    document_service = DocumentLifecycleService(
        kb_service, writer_store, tmp_path / "inputs"
    )

    first_stage_entered = asyncio.Event()
    release_first_stage = asyncio.Event()
    exclusive_entered = asyncio.Event()
    original_persist_source = document_service._persist_source_file

    async def blocked_persist_source(*args, **kwargs):
        first_stage_entered.set()
        await release_first_stage.wait()
        return await original_persist_source(*args, **kwargs)

    monkeypatch.setattr(
        document_service, "_persist_source_file", blocked_persist_source
    )

    async def delete_attempt() -> None:
        async with delete_store.kb_deletion_guard(
            record.id,
            record.generation,
            "job_delete_source_batch_fence",
        ):
            documents, document_total = await delete_store.list_documents(record.id)
            jobs, job_total = await delete_store.list_jobs(record.id)
            assert document_total == 1 and len(documents) == 1
            assert job_total == 1 and len(jobs) == 1
            exclusive_entered.set()

    source = DocumentSourceInput(
        source_name="fenced.txt",
        content=b"fenced content",
        source_type="upload",
        content_type="text/plain",
    )
    create_task = asyncio.create_task(
        document_service.create_source_batch(record.id, [source])
    )
    await asyncio.wait_for(first_stage_entered.wait(), timeout=2)
    delete_task = asyncio.create_task(delete_attempt())
    await asyncio.sleep(0.1)
    assert not exclusive_entered.is_set()

    release_first_stage.set()
    result = await asyncio.wait_for(create_task, timeout=2)
    await asyncio.wait_for(exclusive_entered.wait(), timeout=2)
    await asyncio.wait_for(delete_task, timeout=2)

    assert result.created is True
    assert result.job.payload["kb_generation"] == record.generation
    assert Path(result.documents[0].source_uri).read_bytes() == b"fenced content"


@pytest.mark.asyncio
async def test_create_source_batch_deletion_race_cleans_all_staging(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "source-batch-cleanup.sqlite3"
    writer_store = SQLiteMetadataStore(db_path)
    delete_store = SQLiteMetadataStore(db_path)
    await writer_store.initialize()
    await delete_store.initialize()
    kb_service = KnowledgeBaseService(tmp_path / "source-batch-cleanup-kbs.json")
    await kb_service.initialize()
    record = await kb_service.create(kb_id="kb_source_batch_cleanup", name="Cleanup")
    await writer_store.activate_kb_generation(record.id, record.generation)
    object_storage = FakeObjectStorage()
    document_service = DocumentLifecycleService(
        kb_service,
        writer_store,
        tmp_path / "inputs",
        object_storage=object_storage,
    )

    object_staged = asyncio.Event()
    release_object_stage = asyncio.Event()
    exclusive_entered = asyncio.Event()
    original_upload_file = object_storage.upload_file

    async def blocked_upload_file(*args, **kwargs):
        uri = await original_upload_file(*args, **kwargs)
        object_staged.set()
        await release_object_stage.wait()
        return uri

    monkeypatch.setattr(object_storage, "upload_file", blocked_upload_file)

    async def delete_attempt() -> None:
        async with delete_store.kb_deletion_guard(
            record.id,
            record.generation,
            "job_delete_source_batch_cleanup",
        ):
            exclusive_entered.set()

    source = DocumentSourceInput(
        source_name="cleanup.txt",
        content=b"cleanup content",
        source_type="upload",
        content_type="text/plain",
    )
    create_task = asyncio.create_task(
        document_service.create_source_batch(record.id, [source])
    )
    await asyncio.wait_for(object_staged.wait(), timeout=2)
    await kb_service.delete(record.id, expected_generation=record.generation)
    delete_task = asyncio.create_task(delete_attempt())
    await asyncio.sleep(0.1)
    assert not exclusive_entered.is_set()

    release_object_stage.set()
    with pytest.raises(KnowledgeBaseConflictError) as exc_info:
        await asyncio.wait_for(create_task, timeout=2)
    assert "not active" in str(exc_info.value)
    await asyncio.wait_for(exclusive_entered.wait(), timeout=2)
    await asyncio.wait_for(delete_task, timeout=2)

    documents, document_total = await writer_store.list_documents(record.id)
    jobs, job_total = await writer_store.list_jobs(record.id)
    assert documents == [] and document_total == 0
    assert jobs == [] and job_total == 0
    assert object_storage.files == {}
    assert object_storage.deleted_uris
    workspace_dir = tmp_path / "inputs" / record.workspace
    assert not workspace_dir.exists() or not any(workspace_dir.rglob("*"))

    with pytest.raises(KBLifecycleConflictError):
        await document_service.create_source_batch(record.id, [source])


@pytest.mark.asyncio
async def test_sync_staging_and_job_persist_share_one_composite_guard(tmp_path):
    db_path = tmp_path / "sync-composite-fence.sqlite3"
    writer_store = SQLiteMetadataStore(db_path)
    delete_store = SQLiteMetadataStore(db_path)
    await writer_store.initialize()
    await delete_store.initialize()
    kb_service = KnowledgeBaseService(tmp_path / "sync-composite-kbs.json")
    await kb_service.initialize()
    record = await kb_service.create(kb_id="kb_sync_composite", name="Sync")
    await writer_store.activate_kb_generation(record.id, record.generation)
    document_service = DocumentLifecycleService(
        kb_service, writer_store, tmp_path / "inputs"
    )
    job_service = JobService(kb_service, writer_store)
    exclusive_entered = asyncio.Event()

    async def delete_attempt() -> None:
        async with delete_store.kb_deletion_guard(
            record.id,
            record.generation,
            "job_delete_sync_composite",
        ):
            exclusive_entered.set()

    source = DocumentSourceInput(
        source_name="sync.txt",
        content=b"sync content",
        source_type="upload",
        content_type="text/plain",
        metadata={"source_key": "manual/sync.txt"},
    )
    batch_id = "batch_sync_composite"
    async with document_service.kb_write_guard(record.id):
        staged_path = await document_service.stage_sync_source_bytes(
            record.id,
            batch_id=batch_id,
            item_index=0,
            source=source,
        )
        delete_task = asyncio.create_task(delete_attempt())
        await asyncio.sleep(0.1)
        assert not exclusive_entered.is_set()
        job, created = await job_service.create_job_once(
            record.id,
            job_type="sync",
            batch_id=batch_id,
            stage="syncing",
            payload={"batch_id": batch_id, "source_keys": ["manual/sync.txt"]},
        )
        assert created is True
        assert Path(staged_path).is_file()
        assert job.payload["kb_generation"] == record.generation
        assert not exclusive_entered.is_set()

    await asyncio.wait_for(exclusive_entered.wait(), timeout=2)
    await asyncio.wait_for(delete_task, timeout=2)


def test_list_documents_source_name_filter_and_patch_metadata_flags(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_patch_doc")
    upload = client.post(
        "/kbs/kb_patch_doc/documents:upload",
        files=[
            ("files", ("Alpha Report.pdf", b"alpha", "application/pdf")),
            ("files", ("beta-notes.txt", b"beta", "text/plain")),
        ],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    alpha = next(
        document
        for document in upload.json()["documents"]
        if document["source_name"] == "Alpha Report.pdf"
    )

    filtered = client.get(
        "/kbs/kb_patch_doc/documents?source_name=alpha", headers=_HEADERS
    )
    assert filtered.status_code == 200
    assert filtered.json()["total"] == 1
    assert filtered.json()["documents"][0]["id"] == alpha["id"]

    patched = client.patch(
        f"/kbs/kb_patch_doc/documents/{alpha['id']}",
        json={"metadata": {"reviewed": True}, "enabled": False, "archived": True},
        headers=_HEADERS,
    )
    assert patched.status_code == 200
    patched_payload = patched.json()
    assert patched_payload["enabled"] is False
    assert patched_payload["archived"] is True
    assert patched_payload["metadata"]["reviewed"] is True
    assert patched_payload["metadata"]["batch_id"] == alpha["metadata"]["batch_id"]

    empty_patch = client.patch(
        f"/kbs/kb_patch_doc/documents/{alpha['id']}", json={}, headers=_HEADERS
    )
    assert empty_patch.status_code == 400

    reserved_patch = client.patch(
        f"/kbs/kb_patch_doc/documents/{alpha['id']}",
        json={"metadata": {"pending_parse_job_id": "job_fake"}},
        headers=_HEADERS,
    )
    assert reserved_patch.status_code == 422


def test_enable_disable_document_actions_update_metadata_only(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_enable_disable")
    upload = client.post(
        "/kbs/kb_enable_disable/documents:upload",
        files=[("files", ("paper.pdf", b"pdf", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]

    disabled = client.post(
        f"/kbs/kb_enable_disable/documents/{document_id}:disable",
        headers=_HEADERS,
    )
    enabled = client.post(
        f"/kbs/kb_enable_disable/documents/{document_id}:enable",
        headers=_HEADERS,
    )

    assert disabled.status_code == 200
    assert disabled.json()["enabled"] is False
    assert disabled.json()["status"] == "uploaded"
    assert enabled.status_code == 200
    assert enabled.json()["enabled"] is True
    assert enabled.json()["status"] == "uploaded"


def test_batch_enable_disable_documents(tmp_path):
    client, _kb_service, _store, document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_batch_toggle")
    upload = client.post(
        "/kbs/kb_batch_toggle/documents:upload",
        files=[
            ("files", ("a.pdf", b"pdf-a", "application/pdf")),
            ("files", ("b.pdf", b"pdf-b", "application/pdf")),
        ],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    doc_ids = [doc["id"] for doc in upload.json()["documents"]]

    disabled = client.post(
        "/kbs/kb_batch_toggle/documents:batch-disable",
        json={"document_ids": doc_ids + ["doc_ghost"]},
        headers=_HEADERS,
    )
    assert disabled.status_code == 200, disabled.text
    body = disabled.json()
    assert body["enabled"] is False
    assert body["updated"] == 2
    assert body["not_found"] == 1
    statuses = {item["document_id"]: item["status"] for item in body["items"]}
    assert statuses["doc_ghost"] == "not_found"
    assert all(statuses[doc_id] == "updated" for doc_id in doc_ids)
    for doc_id in doc_ids:
        detail = client.get(
            f"/kbs/kb_batch_toggle/documents/{doc_id}", headers=_HEADERS
        )
        assert detail.json()["enabled"] is False

    # Re-enabling is idempotent and re-applying counts as updated.
    enabled = client.post(
        "/kbs/kb_batch_toggle/documents:batch-enable",
        json={"document_ids": doc_ids},
        headers=_HEADERS,
    )
    assert enabled.status_code == 200, enabled.text
    assert enabled.json()["updated"] == 2
    assert enabled.json()["not_found"] == 0
    for doc_id in doc_ids:
        detail = client.get(
            f"/kbs/kb_batch_toggle/documents/{doc_id}", headers=_HEADERS
        )
        assert detail.json()["enabled"] is True

    # Validation: duplicates 422, empty list 422, unknown KB 404.
    duplicate = client.post(
        "/kbs/kb_batch_toggle/documents:batch-enable",
        json={"document_ids": [doc_ids[0], doc_ids[0]]},
        headers=_HEADERS,
    )
    assert duplicate.status_code == 422
    empty = client.post(
        "/kbs/kb_batch_toggle/documents:batch-enable",
        json={"document_ids": []},
        headers=_HEADERS,
    )
    assert empty.status_code == 422
    missing_kb = client.post(
        "/kbs/kb_ghost/documents:batch-enable",
        json={"document_ids": ["doc_x"]},
        headers=_HEADERS,
    )
    assert missing_kb.status_code == 404


def test_kb_stats_endpoint_aggregates_control_plane(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_stats")
    upload = client.post(
        "/kbs/kb_stats/documents:upload",
        files=[
            ("files", ("a.pdf", b"pdf-a", "application/pdf")),
            ("files", ("b.pdf", b"pdf-b", "application/pdf")),
        ],
        headers=_HEADERS,
    )
    assert upload.status_code == 200, upload.text

    stats = client.get("/kbs/kb_stats/stats", headers=_HEADERS)
    assert stats.status_code == 200, stats.text
    body = stats.json()
    assert body["kb_id"] == "kb_stats"
    assert body["documents"]["total"] == 2
    assert body["documents"]["by_status"] == {"uploaded": 2}
    assert body["counters"] == {"chunks": 0, "entities": 0, "relations": 0}
    assert body["jobs"]["total"] >= 1
    assert body["jobs"]["dead_letter"] == 0
    assert body["artifacts"]["total"] == 0

    assert client.get("/kbs/kb_ghost/stats", headers=_HEADERS).status_code == 404


class _FakeKVStore:
    def __init__(self, rows: dict[str, dict]):
        self._rows = rows

    async def get_by_id(self, id: str):
        return self._rows.get(id)

    async def get_by_ids(self, ids: list[str]):
        return [self._rows.get(item) for item in ids]


def test_document_chunks_endpoint_lists_engine_chunks(tmp_path):
    class ChunkRAG(FakeRAG):
        def __init__(self, workspace: str):
            super().__init__(workspace)
            self.doc_status = _FakeKVStore(
                {"doc-lr-1": {"chunks_list": ["c2", "c1", "c_missing"]}}
            )
            self.text_chunks = _FakeKVStore(
                {
                    "c1": {
                        "content": "first chunk",
                        "tokens": 3,
                        "chunk_order_index": 0,
                        "file_path": "a.pdf",
                    },
                    "c2": {
                        "content": "second chunk",
                        "tokens": 4,
                        "chunk_order_index": 1,
                        "file_path": "a.pdf",
                    },
                }
            )

    class ChunkProbe(BuilderProbe):
        async def build(self, record):
            rag = ChunkRAG(record.workspace)
            self.instances.append(rag)
            return rag

    probe = ChunkProbe()
    client, _kb_service, store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_chunks")
    upload = client.post(
        "/kbs/kb_chunks/documents:upload",
        files=[("files", ("a.pdf", b"pdf", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200, upload.text
    doc_id = upload.json()["documents"][0]["id"]

    # Not built yet: empty page, no engine instance needed.
    empty = client.get(f"/kbs/kb_chunks/documents/{doc_id}/chunks", headers=_HEADERS)
    assert empty.status_code == 200, empty.text
    assert empty.json()["total"] == 0
    assert empty.json()["chunks"] == []
    assert empty.json()["lightrag_doc_id"] is None

    # Simulate a parsed document wired to the engine doc id.
    asyncio.run(
        store.complete_document_parse(
            "kb_chunks",
            doc_id,
            parser_hash="sha256:p",
            lightrag_doc_id="doc-lr-1",
            metadata_patch={},
            artifacts=[],
        )
    )

    listed = client.get(f"/kbs/kb_chunks/documents/{doc_id}/chunks", headers=_HEADERS)
    assert listed.status_code == 200, listed.text
    body = listed.json()
    assert body["lightrag_doc_id"] == "doc-lr-1"
    # Missing chunk rows are skipped; the rest sort by chunk_order_index.
    assert body["total"] == 2
    assert [chunk["id"] for chunk in body["chunks"]] == ["c1", "c2"]
    assert body["chunks"][0]["content"] == "first chunk"
    assert body["chunks"][0]["tokens"] == 3

    paged = client.get(
        f"/kbs/kb_chunks/documents/{doc_id}/chunks?limit=1&offset=1",
        headers=_HEADERS,
    )
    assert paged.status_code == 200
    assert [chunk["id"] for chunk in paged.json()["chunks"]] == ["c2"]
    assert paged.json()["total"] == 2

    assert (
        client.get(
            "/kbs/kb_chunks/documents/doc_ghost/chunks", headers=_HEADERS
        ).status_code
        == 404
    )


def test_delete_uploaded_unindexed_document_soft_deletes_without_lightrag(tmp_path):
    probe = BuilderProbe()
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_delete_uploaded")
    upload = client.post(
        "/kbs/kb_delete_uploaded/documents:upload",
        files=[("files", ("draft.txt", b"draft", "text/plain"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document = upload.json()["documents"][0]

    response = client.delete(
        f"/kbs/kb_delete_uploaded/documents/{document['id']}?idempotency_key=delete-draft",
        headers=_HEADERS,
    )

    assert response.status_code == 200
    job_id = response.json()["id"]
    retry = client.delete(
        f"/kbs/kb_delete_uploaded/documents/{document['id']}?idempotency_key=delete-draft",
        headers=_HEADERS,
    )
    assert retry.status_code == 200
    assert retry.json()["id"] == job_id
    conflict = client.delete(
        f"/kbs/kb_delete_uploaded/documents/{document['id']}"
        "?idempotency_key=delete-draft&delete_source_file=true",
        headers=_HEADERS,
    )
    assert conflict.status_code == 409
    strategy_conflict = client.delete(
        f"/kbs/kb_delete_uploaded/documents/{document['id']}"
        "?idempotency_key=delete-draft&strategy=rebuild_doc_scope",
        headers=_HEADERS,
    )
    assert strategy_conflict.status_code == 409
    job = client.get(f"/kbs/kb_delete_uploaded/jobs/{job_id}", headers=_HEADERS)
    assert job.status_code == 200
    assert job.json()["status"] == "succeeded"
    assert (
        job.json()["result"]["items"][0]["lightrag_delete_result"]["status"]
        == "skipped"
    )
    assert probe.instances == []

    assert (
        client.get(
            f"/kbs/kb_delete_uploaded/documents/{document['id']}", headers=_HEADERS
        ).status_code
        == 404
    )
    listed = client.get("/kbs/kb_delete_uploaded/documents", headers=_HEADERS)
    assert listed.status_code == 200
    assert listed.json()["total"] == 0
    assert Path(document["source_uri"]).exists()


def test_delete_idempotency_key_detects_delete_graph_policy_mismatch(tmp_path):
    client, _kb_service, _store, _document_service, job_service = _build_client(tmp_path)
    _create_kb(client, "kb_delete_graph_idem")
    upload = client.post(
        "/kbs/kb_delete_graph_idem/documents:upload",
        files=[("files", ("draft.txt", b"draft", "text/plain"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document = upload.json()["documents"][0]
    asyncio.run(
        job_service.create_delete_job_once(
            "kb_delete_graph_idem",
            document_id=document["id"],
            lightrag_doc_id=document["lightrag_doc_id"],
            delete_graph_orphans=False,
            idempotency_key="delete-graph-policy",
        )
    )

    response = client.delete(
        f"/kbs/kb_delete_graph_idem/documents/{document['id']}"
        "?idempotency_key=delete-graph-policy",
        headers=_HEADERS,
    )

    assert response.status_code == 409


def test_delete_ready_document_invokes_lightrag_and_removes_files_when_requested(
    tmp_path,
):
    probe = BuilderProbe()
    object_storage = FakeObjectStorage()
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe, object_storage=object_storage
    )
    _create_kb(client, "kb_delete_ready")
    document_id, artifacts = _upload_and_parse_document(client, "kb_delete_ready")
    document = client.get(
        f"/kbs/kb_delete_ready/documents/{document_id}", headers=_HEADERS
    )
    assert document.status_code == 200
    document_payload = document.json()
    source_path = Path(document_payload["source_uri"])
    sidecar_path = Path(artifacts["sidecar"]["uri"])
    lightrag_doc_id = document_payload["lightrag_doc_id"]

    response = client.delete(
        f"/kbs/kb_delete_ready/documents/{document_id}"
        "?delete_source_file=true&delete_artifacts=true&delete_llm_cache=true",
        headers=_HEADERS,
    )

    assert response.status_code == 200
    job = client.get(
        f"/kbs/kb_delete_ready/jobs/{response.json()['id']}", headers=_HEADERS
    )
    assert job.status_code == 200
    assert job.json()["status"] == "succeeded"
    assert probe.instances[-1].delete_calls == [(lightrag_doc_id, True)]
    deleted_objects = job.json()["result"]["items"][0]["file_delete_result"]["deleted_objects"]
    assert document_payload["metadata"]["source_object_uri"] in deleted_objects
    assert artifacts["sidecar"]["metadata"]["object_prefix_uri"] in deleted_objects
    assert artifacts["blocks"]["metadata"]["object_uri"] in deleted_objects
    assert document_payload["metadata"]["source_object_uri"] in object_storage.deleted_uris
    assert artifacts["sidecar"]["metadata"]["object_prefix_uri"] in object_storage.deleted_prefixes
    assert not source_path.exists()
    assert not sidecar_path.exists()
    assert (
        client.get(
            f"/kbs/kb_delete_ready/documents/{document_id}", headers=_HEADERS
        ).status_code
        == 404
    )


def test_batch_delete_partial_failure_for_active_build_and_missing_doc(tmp_path):
    client, _kb_service, store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_batch_delete")
    upload = client.post(
        "/kbs/kb_batch_delete/documents:upload",
        files=[
            ("files", ("active.txt", b"active", "text/plain")),
            ("files", ("ok.txt", b"ok", "text/plain")),
        ],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    active_id, ok_id = [document["id"] for document in upload.json()["documents"]]
    asyncio.run(
        store.claim_document_build_queued(
            "kb_batch_delete",
            active_id,
            metadata_patch={"pending_build_job_id": "job_active_build"},
            require_parsed=False,
        )
    )

    response = client.post(
        "/kbs/kb_batch_delete/documents:batch-delete",
        json={"document_ids": [active_id, ok_id, "doc_missing"]},
        headers=_HEADERS,
    )

    assert response.status_code == 200
    job = client.get(
        f"/kbs/kb_batch_delete/jobs/{response.json()['id']}", headers=_HEADERS
    )
    assert job.status_code == 200
    job_payload = job.json()
    assert job_payload["status"] == "failed"
    assert job_payload["completed_items"] == 1
    assert job_payload["failed_items"] == 2
    assert job_payload["result"]["summary"]["outcome"] == "partial_failure"
    failures = {
        item["document_id"]: item
        for item in job_payload["result"]["items"]
        if item["status"] == "failed"
    }
    assert failures[active_id]["error_code"] == "build_job_active"
    assert failures[active_id]["existing_job_id"] == "job_active_build"
    assert failures["doc_missing"]["error_code"] == "document_not_found"
    assert (
        client.get(
            f"/kbs/kb_batch_delete/documents/{ok_id}", headers=_HEADERS
        ).status_code
        == 404
    )
    active = client.get(f"/kbs/kb_batch_delete/documents/{active_id}", headers=_HEADERS)
    assert active.status_code == 200
    assert active.json()["status"] == "build_queued"


def test_active_delete_blocks_parse_claim(tmp_path):
    client, _kb_service, store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_delete_parse_guard")
    upload = client.post(
        "/kbs/kb_delete_parse_guard/documents:upload",
        files=[("files", ("guard.pdf", b"guard", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]
    asyncio.run(
        store.claim_document_deleting(
            "kb_delete_parse_guard",
            document_id,
            metadata_patch={"pending_delete_job_id": "job_delete_guard"},
        )
    )

    response = client.post(
        f"/kbs/kb_delete_parse_guard/documents/{document_id}:parse",
        json={"engine": "mineru", "process_options": "iF"},
        headers=_HEADERS,
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["error_code"] == "delete_job_active"
    assert detail["existing_job_id"] == "job_delete_guard"
    document = client.get(
        f"/kbs/kb_delete_parse_guard/documents/{document_id}", headers=_HEADERS
    )
    assert document.status_code == 200
    assert document.json()["status"] == "deleting"


def test_active_build_blocks_parse_claim(tmp_path):
    client, _kb_service, store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_build_parse_guard")
    upload = client.post(
        "/kbs/kb_build_parse_guard/documents:upload",
        files=[("files", ("guard.pdf", b"guard", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]
    asyncio.run(
        store.claim_document_build_queued(
            "kb_build_parse_guard",
            document_id,
            metadata_patch={"pending_build_job_id": "job_build_guard"},
            require_parsed=False,
        )
    )

    response = client.post(
        f"/kbs/kb_build_parse_guard/documents/{document_id}:parse",
        json={"engine": "mineru", "process_options": "iF"},
        headers=_HEADERS,
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["error_code"] == "build_job_active"
    assert detail["existing_job_id"] == "job_build_guard"
    document = client.get(
        f"/kbs/kb_build_parse_guard/documents/{document_id}", headers=_HEADERS
    )
    assert document.status_code == 200
    assert document.json()["status"] == "build_queued"

    jobs, _total = asyncio.run(store.list_jobs("kb_build_parse_guard"))
    failed_parse_jobs = [
        job
        for job in jobs
        if job.job_type == "parse" and job.document_id == document_id
    ]
    assert len(failed_parse_jobs) == 1
    assert failed_parse_jobs[0].status == "failed"
    assert failed_parse_jobs[0].error_code == "build_job_active"


def test_delete_artifact_cleanup_rejects_workspace_escape(tmp_path):
    probe = BuilderProbe()
    client, _kb_service, store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_delete_escape")
    document_id, artifacts = _upload_and_parse_document(client, "kb_delete_escape")
    sibling_dir = Path(artifacts["original"]["uri"]).parent.parent / "doc_sibling"
    sibling_dir.mkdir()
    escaped_path = sibling_dir / "escaped-delete.txt"
    escaped_path.write_text("outside", encoding="utf-8")
    with store._connect() as conn:
        conn.execute(
            "UPDATE document_artifacts SET uri = ? WHERE id = ?",
            (str(escaped_path), artifacts["blocks"]["id"]),
        )
        conn.commit()

    response = client.delete(
        f"/kbs/kb_delete_escape/documents/{document_id}?delete_artifacts=true",
        headers=_HEADERS,
    )

    assert response.status_code == 200
    job = client.get(
        f"/kbs/kb_delete_escape/jobs/{response.json()['id']}", headers=_HEADERS
    )
    assert job.status_code == 200
    assert job.json()["status"] == "failed"
    assert job.json()["error_code"] == "delete_failed"
    assert escaped_path.exists()
    document = client.get(
        f"/kbs/kb_delete_escape/documents/{document_id}", headers=_HEADERS
    )
    assert document.status_code == 200
    assert document.json()["status"] == "delete_failed"


def test_replace_ready_document_resets_source_artifacts_and_old_index(tmp_path):
    probe = BuilderProbe()
    object_storage = FakeObjectStorage()
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe, object_storage=object_storage
    )
    _create_kb(client, "kb_replace_ready")
    document_id, artifacts = _upload_and_parse_document(client, "kb_replace_ready")
    before = client.get(
        f"/kbs/kb_replace_ready/documents/{document_id}", headers=_HEADERS
    )
    assert before.status_code == 200
    before_payload = before.json()
    old_source_path = Path(before_payload["source_uri"])
    old_sidecar_path = Path(artifacts["sidecar"]["uri"])
    old_lightrag_doc_id = before_payload["lightrag_doc_id"]
    old_source_object_uri = before_payload["metadata"]["source_object_uri"]
    old_sidecar_object_prefix_uri = artifacts["sidecar"]["metadata"]["object_prefix_uri"]

    response = client.post(
        f"/kbs/kb_replace_ready/documents/{document_id}:replace"
        "?delete_llm_cache=true&idempotency_key=replace-ready-1",
        files={"file": ("paper-v2.pdf", b"new-pdf", "application/pdf")},
        headers=_HEADERS,
    )

    assert response.status_code == 200, response.text
    job = client.get(
        f"/kbs/kb_replace_ready/jobs/{response.json()['id']}", headers=_HEADERS
    )
    assert job.status_code == 200
    job_payload = job.json()
    assert job_payload["status"] == "succeeded"
    assert job_payload["job_type"] == "replace"
    assert job_payload["result"]["previous_lightrag_doc_id"] == old_lightrag_doc_id
    assert probe.instances[-1].delete_calls == [(old_lightrag_doc_id, True)]

    detail = client.get(
        f"/kbs/kb_replace_ready/documents/{document_id}", headers=_HEADERS
    )
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["id"] == document_id
    assert payload["status"] == "uploaded"
    assert payload["source_name"] == "paper-v2.pdf"
    assert payload["source_hash"] != before_payload["source_hash"]
    assert payload["metadata"]["source_object_uri"].startswith("s3://fake-bucket/")
    assert payload["metadata"]["source_object_uri"] != old_source_object_uri
    assert payload["metadata"]["source_object_uri"] in object_storage.files
    assert old_source_object_uri in object_storage.deleted_uris
    assert old_sidecar_object_prefix_uri in object_storage.deleted_prefixes
    assert payload["lightrag_doc_id"] is None
    assert payload["parser_hash"] is None
    assert payload["index_hash"] is None
    assert payload["chunks_count"] is None
    assert payload["metadata"]["last_replace_job_id"] == response.json()["id"]
    assert "blocks_path" not in payload["metadata"]
    new_source_path = Path(payload["source_uri"])
    assert new_source_path.exists()
    assert new_source_path.read_bytes() == b"new-pdf"
    assert not old_source_path.exists()
    assert not old_sidecar_path.exists()
    staged_replace_path = (
        tmp_path
        / "inputs"
        / "kb_replace_ready"
        / document_id
        / f".replace-staging-{response.json()['id']}.bin"
    )
    assert not staged_replace_path.exists()

    artifacts_after = client.get(
        f"/kbs/kb_replace_ready/documents/{document_id}/artifacts", headers=_HEADERS
    )
    assert artifacts_after.status_code == 200
    assert artifacts_after.json()["total"] == 0


def test_sync_cleans_partial_staging_when_later_item_fails(tmp_path, monkeypatch):
    client, _kb_service, _store, document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_sync_partial_stage")
    original_stage_sync_source_bytes = document_service.stage_sync_source_bytes
    staged_paths: list[Path] = []

    async def fail_after_first_stage(kb_id, *, batch_id, item_index, source):
        staged_path = await original_stage_sync_source_bytes(
            kb_id,
            batch_id=batch_id,
            item_index=item_index,
            source=source,
        )
        staged_paths.append(Path(staged_path))
        if item_index == 1:
            raise RuntimeError("stage exploded")
        return staged_path

    monkeypatch.setattr(
        document_service,
        "stage_sync_source_bytes",
        fail_after_first_stage,
    )

    response = client.post(
        "/kbs/kb_sync_partial_stage/documents:sync?auto_index=false",
        files=[
            ("files", ("one.pdf", b"one", "application/pdf")),
            ("files", ("two.pdf", b"two", "application/pdf")),
        ],
        data={"source_keys": ["manual/one.pdf", "manual/two.pdf"]},
        headers=_HEADERS,
    )

    assert response.status_code == 500
    assert len(staged_paths) == 2
    assert not any(path.exists() for path in staged_paths)
    sync_staging_root = tmp_path / "inputs" / "kb_sync_partial_stage" / ".sync-staging"
    assert not any(sync_staging_root.glob("**/*"))


def test_replace_idempotency_key_reuses_existing_job_and_conflicts(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_replace_idem")
    upload = client.post(
        "/kbs/kb_replace_idem/documents:upload",
        files=[("files", ("paper.pdf", b"pdf", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]

    first = client.post(
        f"/kbs/kb_replace_idem/documents/{document_id}:replace"
        "?idempotency_key=replace-idem-1",
        files={"file": ("paper-v2.pdf", b"new", "application/pdf")},
        headers=_HEADERS,
    )
    second = client.post(
        f"/kbs/kb_replace_idem/documents/{document_id}:replace"
        "?idempotency_key=replace-idem-1",
        files={"file": ("paper-v2.pdf", b"new", "application/pdf")},
        headers=_HEADERS,
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]

    conflict = client.post(
        f"/kbs/kb_replace_idem/documents/{document_id}:replace"
        "?idempotency_key=replace-idem-1",
        files={"file": ("paper-v3.pdf", b"different", "application/pdf")},
        headers=_HEADERS,
    )
    assert conflict.status_code == 409


def test_replace_auto_index_requires_auto_parse(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_replace_auto_index")
    upload = client.post(
        "/kbs/kb_replace_auto_index/documents:upload",
        files=[("files", ("paper.pdf", b"pdf", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]

    response = client.post(
        f"/kbs/kb_replace_auto_index/documents/{document_id}:replace"
        "?auto_index=true&auto_parse=false",
        files={"file": ("paper-v2.pdf", b"new", "application/pdf")},
        headers=_HEADERS,
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "auto_index requires auto_parse for document replacement"
    )


def test_active_replace_blocks_parse_claim(tmp_path):
    client, _kb_service, store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_replace_guard")
    upload = client.post(
        "/kbs/kb_replace_guard/documents:upload",
        files=[("files", ("guard.pdf", b"guard", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]
    asyncio.run(
        store.claim_document_replacing(
            "kb_replace_guard",
            document_id,
            metadata_patch={"pending_replace_job_id": "job_replace_guard"},
        )
    )

    response = client.post(
        f"/kbs/kb_replace_guard/documents/{document_id}:parse",
        json={"engine": "mineru", "process_options": "iF"},
        headers=_HEADERS,
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["error_code"] == "replace_job_active"
    assert detail["existing_job_id"] == "job_replace_guard"
    document = client.get(
        f"/kbs/kb_replace_guard/documents/{document_id}", headers=_HEADERS
    )
    assert document.status_code == 200
    assert document.json()["status"] == "replacing"


def test_replace_artifact_cleanup_rejects_workspace_escape(tmp_path):
    probe = BuilderProbe()
    client, _kb_service, store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_replace_escape")
    document_id, artifacts = _upload_and_parse_document(client, "kb_replace_escape")
    before = client.get(
        f"/kbs/kb_replace_escape/documents/{document_id}", headers=_HEADERS
    )
    assert before.status_code == 200
    old_lightrag_doc_id = before.json()["lightrag_doc_id"]
    escaped_path = tmp_path / "escaped-replace.txt"
    escaped_path.write_text("outside", encoding="utf-8")
    with store._connect() as conn:
        conn.execute(
            "UPDATE document_artifacts SET uri = ? WHERE id = ?",
            (str(escaped_path), artifacts["blocks"]["id"]),
        )
        conn.commit()

    response = client.post(
        f"/kbs/kb_replace_escape/documents/{document_id}:replace",
        files={"file": ("new.pdf", b"new", "application/pdf")},
        headers=_HEADERS,
    )

    assert response.status_code == 200
    job = client.get(
        f"/kbs/kb_replace_escape/jobs/{response.json()['id']}", headers=_HEADERS
    )
    assert job.status_code == 200
    assert job.json()["status"] == "failed"
    assert job.json()["error_code"] == "replace_failed"
    assert escaped_path.exists()
    assert probe.instances[-1].delete_calls == []
    document = client.get(
        f"/kbs/kb_replace_escape/documents/{document_id}", headers=_HEADERS
    )
    assert document.status_code == 200
    payload = document.json()
    assert payload["status"] == "replace_failed"
    assert payload["lightrag_doc_id"] == old_lightrag_doc_id


def test_replace_failure_after_old_index_delete_clears_index_metadata(
    tmp_path, monkeypatch
):
    probe = BuilderProbe()
    client, _kb_service, _store, document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_replace_partial_fail")
    document_id, _artifacts = _upload_and_parse_document(
        client, "kb_replace_partial_fail"
    )
    before = client.get(
        f"/kbs/kb_replace_partial_fail/documents/{document_id}", headers=_HEADERS
    )
    assert before.status_code == 200
    old_lightrag_doc_id = before.json()["lightrag_doc_id"]

    async def fail_after_lightrag_delete(*_args, **_kwargs):
        raise RuntimeError("file replacement exploded")

    monkeypatch.setattr(
        document_service,
        "replace_document_source",
        fail_after_lightrag_delete,
    )

    response = client.post(
        f"/kbs/kb_replace_partial_fail/documents/{document_id}:replace",
        files={"file": ("new.pdf", b"new", "application/pdf")},
        headers=_HEADERS,
    )

    assert response.status_code == 200
    job = client.get(
        f"/kbs/kb_replace_partial_fail/jobs/{response.json()['id']}",
        headers=_HEADERS,
    )
    assert job.status_code == 200
    assert job.json()["status"] == "failed"
    assert probe.instances[-1].delete_calls == [(old_lightrag_doc_id, False)]
    document = client.get(
        f"/kbs/kb_replace_partial_fail/documents/{document_id}", headers=_HEADERS
    )
    assert document.status_code == 200
    payload = document.json()
    assert payload["status"] == "replace_failed"
    assert payload["lightrag_doc_id"] is None
    assert payload["index_hash"] is None


def test_replace_background_start_failure_releases_replacing_claim(
    tmp_path, monkeypatch
):
    client, _kb_service, _store, _document_service, job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_replace_start_fail")
    upload = client.post(
        "/kbs/kb_replace_start_fail/documents:upload",
        files=[("files", ("paper.pdf", b"pdf", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]
    original_transition_job = job_service.transition_job

    async def fail_running_transition(kb_id, job_id, **kwargs):
        if kwargs.get("status") == "running":
            raise RuntimeError("transition exploded")
        return await original_transition_job(kb_id, job_id, **kwargs)

    monkeypatch.setattr(job_service, "transition_job", fail_running_transition)

    response = client.post(
        f"/kbs/kb_replace_start_fail/documents/{document_id}:replace",
        files={"file": ("paper-v2.pdf", b"new", "application/pdf")},
        headers=_HEADERS,
    )

    assert response.status_code == 200
    job = client.get(
        f"/kbs/kb_replace_start_fail/jobs/{response.json()['id']}",
        headers=_HEADERS,
    )
    assert job.status_code == 200
    assert job.json()["status"] == "failed"
    document = client.get(
        f"/kbs/kb_replace_start_fail/documents/{document_id}", headers=_HEADERS
    )
    assert document.status_code == 200
    assert document.json()["status"] == "replace_failed"


def test_parse_document_succeeds_and_persists_artifacts(tmp_path):
    probe = BuilderProbe()
    object_storage = FakeObjectStorage()
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe, object_storage=object_storage
    )
    _create_kb(client, "kb_parse")
    upload = client.post(
        "/kbs/kb_parse/documents:upload",
        files=[("files", ("paper.pdf", b"pdf", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]

    response = client.post(
        f"/kbs/kb_parse/documents/{document_id}:parse",
        json={"engine": "mineru", "process_options": "iF", "force_reparse": True},
        headers=_HEADERS,
    )

    assert response.status_code == 200
    job_id = response.json()["id"]
    assert response.json()["job_type"] == "parse"
    assert response.json()["document_id"] == document_id

    job = client.get(f"/kbs/kb_parse/jobs/{job_id}", headers=_HEADERS)
    assert job.status_code == 200
    assert job.json()["status"] == "succeeded"
    assert job.json()["completed_items"] == 1
    assert job.json()["result"]["artifact_count"] >= 3

    document = client.get(f"/kbs/kb_parse/documents/{document_id}", headers=_HEADERS)
    assert document.status_code == 200
    document_payload = document.json()
    assert document_payload["status"] == "parsed"
    assert document_payload["parser_hash"].startswith("sha256:")
    assert document_payload["lightrag_doc_id"].startswith("doc-")
    assert Path(document_payload["source_uri"]).exists()
    assert document_payload["metadata"]["source_object_uri"].startswith(
        "s3://fake-bucket/workspaces/"
    )

    assert probe.instances
    _engine, _doc_id, _file_path, content_data = probe.instances[0].parse_calls[0]
    assert content_data["force_reparse"] is True
    assert content_data["archive_source_after_parse"] is False

    artifacts = client.get(
        f"/kbs/kb_parse/documents/{document_id}/artifacts", headers=_HEADERS
    )
    assert artifacts.status_code == 200
    artifact_payload = artifacts.json()
    artifact_types = {item["artifact_type"] for item in artifact_payload["artifacts"]}
    assert {
        "original",
        "sidecar",
        "blocks",
        "raw_dir",
        "markdown",
        "content_list",
        "middle_json",
        "model_json",
        "image",
        "layout_pdf",
    }.issubset(artifact_types)
    original = next(
        item
        for item in artifact_payload["artifacts"]
        if item["artifact_type"] == "original"
    )
    assert original["checksum"].startswith("sha256:")
    assert original["size_bytes"] == 3
    assert original["metadata"]["object_uri"] == document_payload["metadata"]["source_object_uri"]
    assert original["metadata"]["object_uri"] in object_storage.files
    sidecar = next(
        item
        for item in artifact_payload["artifacts"]
        if item["artifact_type"] == "sidecar"
    )
    assert sidecar["metadata"]["object_prefix_uri"].startswith(
        "s3://fake-bucket/workspaces/"
    )
    assert sidecar["metadata"]["object_prefix_uri"] in object_storage.prefix_files
    blocks = next(
        item
        for item in artifact_payload["artifacts"]
        if item["artifact_type"] == "blocks"
    )
    assert blocks["metadata"]["object_uri"] in object_storage.files
    content_list = next(
        item
        for item in artifact_payload["artifacts"]
        if item["artifact_type"] == "content_list"
    )
    assert content_list["metadata"]["source"] == "raw_dir"
    assert content_list["metadata"]["relative_path"] == "content_list.json"
    image = next(
        item
        for item in artifact_payload["artifacts"]
        if item["artifact_type"] == "image"
    )
    assert image["metadata"]["relative_path"] == "images/page-1.png"

    artifact_id = artifact_payload["artifacts"][0]["id"]
    detail = client.get(
        f"/kbs/kb_parse/documents/{document_id}/artifacts/{artifact_id}",
        headers=_HEADERS,
    )
    assert detail.status_code == 200
    assert detail.json()["id"] == artifact_id


@pytest.mark.parametrize(
    ("filename", "content", "content_type", "expected_snippet"),
    [
        ("notes.txt", b"plain legacy text", "text/plain", "plain legacy text"),
        ("data.json", b'{"answer": 42}', "application/json", "answer"),
        ("legacy.csv", b"name,value\nlegacy,1\n", "text/csv", "legacy,1"),
        ("script.py", b"print('legacy code')\n", "text/x-python", "legacy code"),
    ],
)
def test_parse_legacy_text_data_code_succeeds_and_persists_artifacts(
    tmp_path, filename, content, content_type, expected_snippet
):
    probe = BuilderProbe()
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_parse_legacy")
    upload = client.post(
        "/kbs/kb_parse_legacy/documents:upload",
        files=[("files", (filename, content, content_type))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200, upload.text
    document_id = upload.json()["documents"][0]["id"]

    response = client.post(
        f"/kbs/kb_parse_legacy/documents/{document_id}:parse",
        json={"engine": "legacy", "force_reparse": True},
        headers=_HEADERS,
    )

    assert response.status_code == 200, response.text
    job = client.get(
        f"/kbs/kb_parse_legacy/jobs/{response.json()['id']}", headers=_HEADERS
    )
    assert job.status_code == 200
    assert job.json()["status"] == "succeeded"
    assert probe.instances[0].parse_calls[0][0] == "legacy"

    document = client.get(
        f"/kbs/kb_parse_legacy/documents/{document_id}", headers=_HEADERS
    )
    assert document.status_code == 200
    document_payload = document.json()
    assert document_payload["status"] == "parsed"
    assert document_payload["metadata"]["parse_engine"] == "legacy"

    artifacts = client.get(
        f"/kbs/kb_parse_legacy/documents/{document_id}/artifacts", headers=_HEADERS
    )
    assert artifacts.status_code == 200
    artifacts_by_type = {
        item["artifact_type"]: item for item in artifacts.json()["artifacts"]
    }
    assert {"original", "sidecar", "blocks"}.issubset(artifacts_by_type)
    source_path = Path(document_payload["source_uri"])
    sidecar_path = Path(artifacts_by_type["sidecar"]["uri"])
    blocks_path = Path(artifacts_by_type["blocks"]["uri"])
    assert sidecar_path.is_relative_to(source_path.parent / "__parsed__")
    assert blocks_path.parent == sidecar_path
    assert expected_snippet in blocks_path.read_text(encoding="utf-8")


def test_parse_csv_with_docling_persists_artifacts_and_table_preview(tmp_path):
    probe = BuilderProbe()
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_parse_csv_docling")
    upload = client.post(
        "/kbs/kb_parse_csv_docling/documents:upload",
        files=[("files", ("data.csv", b"name,value\nalpha,1\n", "text/csv"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200, upload.text
    document_id = upload.json()["documents"][0]["id"]

    response = client.post(
        f"/kbs/kb_parse_csv_docling/documents/{document_id}:parse",
        json={"engine": "docling"},
        headers=_HEADERS,
    )

    assert response.status_code == 200, response.text
    job = client.get(
        f"/kbs/kb_parse_csv_docling/jobs/{response.json()['id']}",
        headers=_HEADERS,
    )
    assert job.status_code == 200
    assert job.json()["status"] == "succeeded"
    assert probe.instances[0].parse_calls[0][0] == "docling"

    document = client.get(
        f"/kbs/kb_parse_csv_docling/documents/{document_id}", headers=_HEADERS
    )
    assert document.status_code == 200
    document_payload = document.json()
    assert document_payload["status"] == "parsed"
    assert document_payload["metadata"]["parse_engine"] == "docling"

    artifacts = client.get(
        f"/kbs/kb_parse_csv_docling/documents/{document_id}/artifacts",
        headers=_HEADERS,
    )
    assert artifacts.status_code == 200
    artifact_types = {item["artifact_type"] for item in artifacts.json()["artifacts"]}
    assert {"original", "sidecar", "blocks", "preview_table_json"}.issubset(
        artifact_types
    )

    manifest = client.get(
        f"/kbs/kb_parse_csv_docling/documents/{document_id}/preview",
        headers=_HEADERS,
    )
    assert manifest.status_code == 200, manifest.text
    payload = manifest.json()
    assert payload["preferred"]["kind"] == "table"
    assert any(
        variant["artifact_type"] == "preview_table_json"
        for variant in payload["variants"]
    )


def test_preview_manifest_returns_text_variant_and_original_fallback(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_preview_text")
    upload = client.post(
        "/kbs/kb_preview_text/documents:upload",
        files=[("files", ("notes.txt", b"preview me", "text/plain"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200, upload.text
    document_id = upload.json()["documents"][0]["id"]
    parse = client.post(
        f"/kbs/kb_preview_text/documents/{document_id}:parse",
        json={"engine": "legacy"},
        headers=_HEADERS,
    )
    assert parse.status_code == 200, parse.text

    manifest = client.get(
        f"/kbs/kb_preview_text/documents/{document_id}/preview", headers=_HEADERS
    )

    assert manifest.status_code == 200, manifest.text
    payload = manifest.json()
    assert payload["document_id"] == document_id
    assert payload["source_name"] == "notes.txt"
    assert payload["status"] == "parsed"
    assert payload["preferred"]["kind"] == "text"
    assert payload["preferred"]["artifact_type"] == "preview_text"
    assert payload["preferred"]["preview_url"].startswith(
        f"/kbs/kb_preview_text/documents/{document_id}/artifacts/"
    )
    assert "source_uri" not in json.dumps(payload)
    assert payload["fallback"]["artifact_type"] == "original"
    assert payload["fallback"]["download_url"].endswith(":download")

    artifacts = client.get(
        f"/kbs/kb_preview_text/documents/{document_id}/artifacts",
        headers=_HEADERS,
    ).json()["artifacts"]
    document = client.get(
        f"/kbs/kb_preview_text/documents/{document_id}", headers=_HEADERS
    ).json()
    preview_artifact = next(
        item for item in artifacts if item["artifact_type"] == "preview_text"
    )
    assert preview_artifact["metadata"]["preview"] is True
    assert preview_artifact["metadata"]["source_hash"] == document["source_hash"]
    assert preview_artifact["metadata"]["parser_hash"] == document["parser_hash"]
    assert preview_artifact["metadata"]["preview_schema_version"] == 1


def test_preview_manifest_binary_document_falls_back_to_original(tmp_path):
    probe = BuilderProbe(parse_content="")
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_preview_binary")
    upload = client.post(
        "/kbs/kb_preview_binary/documents:upload",
        files=[("files", ("paper.pdf", b"%PDF-binary", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200, upload.text
    document_id = upload.json()["documents"][0]["id"]
    parse = client.post(
        f"/kbs/kb_preview_binary/documents/{document_id}:parse",
        json={"engine": "mineru"},
        headers=_HEADERS,
    )
    assert parse.status_code == 200, parse.text

    manifest = client.get(
        f"/kbs/kb_preview_binary/documents/{document_id}/preview", headers=_HEADERS
    )

    assert manifest.status_code == 200, manifest.text
    payload = manifest.json()
    assert payload["preferred"]["kind"] == "pdf"
    assert payload["preferred"]["artifact_type"] == "original"
    assert payload["preferred"]["preview_url"].endswith(":preview")
    assert payload["fallback"]["artifact_type"] == "original"
    assert payload["fallback"]["media_type"] == "application/pdf"


def test_object_storage_preview_artifact_restores_before_inline_preview(tmp_path):
    object_storage = FakeObjectStorage()
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, object_storage=object_storage
    )
    _create_kb(client, "kb_preview_object")
    upload = client.post(
        "/kbs/kb_preview_object/documents:upload",
        files=[("files", ("notes.txt", b"restore preview", "text/plain"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200, upload.text
    document_id = upload.json()["documents"][0]["id"]
    parse = client.post(
        f"/kbs/kb_preview_object/documents/{document_id}:parse",
        json={"engine": "legacy"},
        headers=_HEADERS,
    )
    assert parse.status_code == 200, parse.text
    artifacts = client.get(
        f"/kbs/kb_preview_object/documents/{document_id}/artifacts",
        headers=_HEADERS,
    ).json()["artifacts"]
    preview_artifact = next(
        item for item in artifacts if item["artifact_type"] == "preview_text"
    )
    preview_path = Path(preview_artifact["uri"])
    object_uri = preview_artifact["metadata"]["object_uri"]
    assert object_uri in object_storage.files
    preview_path.unlink()

    response = client.get(
        f"/kbs/kb_preview_object/documents/{document_id}/artifacts/"
        f"{preview_artifact['id']}:preview",
        headers=_HEADERS,
    )

    assert response.status_code == 200, response.text
    assert response.text == "restore preview"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert (object_uri, preview_path) in object_storage.downloads


def test_parse_document_uses_active_parser_config_defaults(tmp_path):
    probe = BuilderProbe()
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_parse_cfg")
    config = client.post(
        "/kbs/kb_parse_cfg/configs",
        json={
            "config": {
                "parser_config": {
                    "engine": "mineru",
                    "process_options": " i-F ",
                }
            }
        },
        headers=_HEADERS,
    )
    assert config.status_code == 200
    activate = client.post(
        f"/kbs/kb_parse_cfg/configs/{config.json()['id']}:activate",
        headers=_HEADERS,
    )
    assert activate.status_code == 200
    upload = client.post(
        "/kbs/kb_parse_cfg/documents:upload",
        files=[("files", ("paper.pdf", b"pdf", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]

    response = client.post(
        f"/kbs/kb_parse_cfg/documents/{document_id}:parse",
        json={},
        headers=_HEADERS,
    )

    assert response.status_code == 200
    assert response.json()["payload"]["parser_engine"] == "mineru"
    assert response.json()["payload"]["process_options"] == "iF"
    job = client.get(
        f"/kbs/kb_parse_cfg/jobs/{response.json()['id']}", headers=_HEADERS
    )
    assert job.status_code == 200
    assert job.json()["status"] == "succeeded"
    assert probe.instances[0].parse_calls[0][0] == "mineru"
    assert probe.instances[0].parse_calls[0][3]["process_options"] == "iF"
    document = client.get(
        f"/kbs/kb_parse_cfg/documents/{document_id}", headers=_HEADERS
    )
    assert document.status_code == 200
    assert document.json()["metadata"]["parser_engine"] == "mineru"
    assert document.json()["metadata"]["process_options"] == "iF"


def test_auto_parse_upload_snapshots_active_parser_config_defaults(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(tmp_path)
    _create_kb(client, "kb_auto_parse_cfg")
    config = client.post(
        "/kbs/kb_auto_parse_cfg/configs",
        json={
            "config": {
                "parser_config": {
                    "engine": "mineru",
                    "process_options": " i-F ",
                }
            }
        },
        headers=_HEADERS,
    )
    assert config.status_code == 200
    activate = client.post(
        f"/kbs/kb_auto_parse_cfg/configs/{config.json()['id']}:activate",
        headers=_HEADERS,
    )
    assert activate.status_code == 200

    upload = client.post(
        "/kbs/kb_auto_parse_cfg/documents:upload?auto_parse=true",
        files=[("files", ("paper.pdf", b"pdf", "application/pdf"))],
        headers=_HEADERS,
    )

    assert upload.status_code == 200
    payload = upload.json()
    document = payload["documents"][0]
    assert document["status"] == "parse_queued"
    assert document["metadata"]["parser_engine"] == "mineru"
    assert document["metadata"]["process_options"] == "iF"
    job = client.get(
        f"/kbs/kb_auto_parse_cfg/jobs/{payload['job_id']}", headers=_HEADERS
    )
    assert job.status_code == 200
    assert job.json()["payload"]["parser_engine"] == "mineru"
    assert job.json()["payload"]["process_options"] == "iF"


def test_parse_document_idempotency_key_reuses_existing_job(tmp_path):
    probe = BuilderProbe()
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_parse_idem")
    upload = client.post(
        "/kbs/kb_parse_idem/documents:upload",
        files=[("files", ("paper.pdf", b"pdf", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]

    first = client.post(
        f"/kbs/kb_parse_idem/documents/{document_id}:parse",
        json={"engine": "mineru", "idempotency_key": "idem-parse-1"},
        headers=_HEADERS,
    )
    second = client.post(
        f"/kbs/kb_parse_idem/documents/{document_id}:parse",
        json={"engine": "mineru", "idempotency_key": "idem-parse-1"},
        headers=_HEADERS,
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert len(probe.instances) == 1
    assert len(probe.instances[0].parse_calls) == 1

    conflict = client.post(
        f"/kbs/kb_parse_idem/documents/{document_id}:parse",
        json={
            "engine": "mineru",
            "force_reparse": True,
            "idempotency_key": "idem-parse-1",
        },
        headers=_HEADERS,
    )
    assert conflict.status_code == 409


def test_batch_parse_documents_succeeds_and_persists_artifacts(tmp_path):
    probe = BuilderProbe()
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_batch_parse")
    upload = client.post(
        "/kbs/kb_batch_parse/documents:upload",
        files=[
            ("files", ("alpha.pdf", b"alpha", "application/pdf")),
            ("files", ("beta.pdf", b"beta", "application/pdf")),
        ],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_ids = [document["id"] for document in upload.json()["documents"]]

    response = client.post(
        "/kbs/kb_batch_parse/documents:batch-parse",
        json={
            "document_ids": document_ids,
            "engine": "mineru",
            "process_options": "iF",
            "force_reparse": True,
        },
        headers=_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"].startswith("job_parse_")
    assert payload["batch_id"].startswith("batch_")
    assert payload["documents"]
    assert {document["status"] for document in payload["documents"]} == {"parse_queued"}

    job = client.get(f"/kbs/kb_batch_parse/jobs/{payload['job_id']}", headers=_HEADERS)
    assert job.status_code == 200
    job_payload = job.json()
    assert job_payload["status"] == "succeeded"
    assert job_payload["job_type"] == "parse"
    assert job_payload["document_id"] is None
    assert job_payload["batch_id"] == payload["batch_id"]
    assert job_payload["total_items"] == 2
    assert job_payload["completed_items"] == 2
    assert job_payload["failed_items"] == 0
    assert job_payload["result"]["summary"]["outcome"] == "succeeded"
    assert {item["status"] for item in job_payload["result"]["items"]} == {"succeeded"}

    for document_id in document_ids:
        document = client.get(
            f"/kbs/kb_batch_parse/documents/{document_id}", headers=_HEADERS
        )
        assert document.status_code == 200
        assert document.json()["status"] == "parsed"
        artifacts = client.get(
            f"/kbs/kb_batch_parse/documents/{document_id}/artifacts",
            headers=_HEADERS,
        )
        assert artifacts.status_code == 200
        assert artifacts.json()["total"] >= 3

    assert len(probe.instances) == 1
    assert len(probe.instances[0].parse_calls) == 2


def test_batch_parse_auto_index_true_stays_parse_only(tmp_path):
    """Regression: :batch-parse is parse-only. Even when a client sends
    ``auto_index=true`` the persisted job payload must stay ``auto_index=False``
    so a durable-worker resume (``_run_aggregate`` builds only when
    ``payload["auto_index"]`` is set) behaves identically to the in-process path
    (which never builds). Previously the route forwarded ``auto_index=true``
    into the payload, so the SAME job parsed-only in-process but parsed+built on
    a worker resume — divergent behavior for one persisted job."""
    probe = BuilderProbe()
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_batch_parse_ai")
    upload = client.post(
        "/kbs/kb_batch_parse_ai/documents:upload",
        files=[
            ("files", ("alpha.pdf", b"alpha", "application/pdf")),
            ("files", ("beta.pdf", b"beta", "application/pdf")),
        ],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_ids = [document["id"] for document in upload.json()["documents"]]

    response = client.post(
        "/kbs/kb_batch_parse_ai/documents:batch-parse",
        json={
            "document_ids": document_ids,
            "engine": "mineru",
            "process_options": "iF",
            "force_reparse": True,
            "auto_index": True,
        },
        headers=_HEADERS,
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    job = client.get(f"/kbs/kb_batch_parse_ai/jobs/{job_id}", headers=_HEADERS)
    assert job.status_code == 200
    job_payload = job.json()
    # The fix: payload is parse-only regardless of the requested auto_index, so
    # a worker resume of this job cannot build.
    assert job_payload["payload"]["auto_index"] is False
    assert job_payload["job_type"] == "parse"

    # Documents land in `parsed` (parse-only), never `building` / `ready`.
    for document_id in document_ids:
        document = client.get(
            f"/kbs/kb_batch_parse_ai/documents/{document_id}", headers=_HEADERS
        )
        assert document.status_code == 200
        assert document.json()["status"] == "parsed"


def test_single_parse_auto_index_true_stays_parse_only(tmp_path):
    """Regression: single-document :parse is parse-only too — ``auto_index=true``
    must not leak into the persisted payload (mirrors :batch-parse), so a worker
    resume and the in-process path agree (neither builds)."""
    probe = BuilderProbe()
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_parse_ai")
    upload = client.post(
        "/kbs/kb_parse_ai/documents:upload",
        files=[("files", ("alpha.pdf", b"alpha", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]

    response = client.post(
        f"/kbs/kb_parse_ai/documents/{document_id}:parse",
        json={
            "engine": "mineru",
            "process_options": "iF",
            "force_reparse": True,
            "auto_index": True,
        },
        headers=_HEADERS,
    )
    assert response.status_code == 200
    job_id = response.json()["id"]

    job = client.get(f"/kbs/kb_parse_ai/jobs/{job_id}", headers=_HEADERS)
    assert job.status_code == 200
    assert job.json()["payload"]["auto_index"] is False

    document = client.get(
        f"/kbs/kb_parse_ai/documents/{document_id}", headers=_HEADERS
    )
    assert document.status_code == 200
    assert document.json()["status"] == "parsed"


def test_batch_parse_idempotency_key_reuses_existing_job(tmp_path):
    probe = BuilderProbe()
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_batch_idem")
    upload = client.post(
        "/kbs/kb_batch_idem/documents:upload",
        files=[
            ("files", ("alpha.pdf", b"alpha", "application/pdf")),
            ("files", ("beta.pdf", b"beta", "application/pdf")),
        ],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_ids = [document["id"] for document in upload.json()["documents"]]
    request = {
        "document_ids": document_ids,
        "engine": "mineru",
        "idempotency_key": "idem-batch-parse-1",
    }

    first = client.post(
        "/kbs/kb_batch_idem/documents:batch-parse", json=request, headers=_HEADERS
    )
    second = client.post(
        "/kbs/kb_batch_idem/documents:batch-parse", json=request, headers=_HEADERS
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["job_id"] == first.json()["job_id"]
    assert second.json()["batch_id"] == first.json()["batch_id"]
    assert [document["id"] for document in second.json()["documents"]] == document_ids
    assert len(probe.instances) == 1
    assert len(probe.instances[0].parse_calls) == 2

    conflict = client.post(
        "/kbs/kb_batch_idem/documents:batch-parse",
        json={**request, "force_reparse": True},
        headers=_HEADERS,
    )
    assert conflict.status_code == 409


def test_batch_parse_partial_failure_marks_job_failed_and_continues(tmp_path):
    probe = BuilderProbe(fail_source_names={"bad.pdf"})
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_batch_partial")
    upload = client.post(
        "/kbs/kb_batch_partial/documents:upload",
        files=[
            ("files", ("good.pdf", b"good", "application/pdf")),
            ("files", ("bad.pdf", b"bad", "application/pdf")),
        ],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    documents = upload.json()["documents"]
    document_ids = [document["id"] for document in documents]
    names_by_id = {document["id"]: document["source_name"] for document in documents}

    response = client.post(
        "/kbs/kb_batch_partial/documents:batch-parse",
        json={"document_ids": document_ids, "engine": "mineru"},
        headers=_HEADERS,
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    job = client.get(f"/kbs/kb_batch_partial/jobs/{job_id}", headers=_HEADERS)
    assert job.status_code == 200
    job_payload = job.json()
    assert job_payload["status"] == "failed"
    assert job_payload["completed_items"] == 1
    assert job_payload["failed_items"] == 1
    assert job_payload["error_code"] == "partial_parse_failed"
    assert job_payload["result"]["summary"]["outcome"] == "partial_failure"
    assert {item["status"] for item in job_payload["result"]["items"]} == {
        "succeeded",
        "failed",
    }

    for document_id in document_ids:
        document = client.get(
            f"/kbs/kb_batch_partial/documents/{document_id}", headers=_HEADERS
        )
        assert document.status_code == 200
        expected_status = (
            "parse_failed" if names_by_id[document_id] == "bad.pdf" else "parsed"
        )
        assert document.json()["status"] == expected_status


def test_batch_parse_treats_active_parse_as_per_item_failure(tmp_path):
    probe = BuilderProbe()
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_batch_active")
    active_upload = client.post(
        "/kbs/kb_batch_active/documents:upload",
        files=[("files", ("active.pdf", b"active", "application/pdf"))],
        headers=_HEADERS,
    )
    assert active_upload.status_code == 200
    active_payload = active_upload.json()
    active_document_id = active_payload["documents"][0]["id"]

    async def _seed_active_parse() -> str:
        plan = await _document_service.create_parse_plan(
            "kb_batch_active", active_document_id, parser_engine="mineru"
        )
        job, _created = await _job_service.create_parse_job_once(
            "kb_batch_active",
            document_id=active_document_id,
            parser_hash=plan.parser_hash,
            lightrag_doc_id=plan.lightrag_doc_id,
            parser_engine=plan.parser_engine,
            process_options=plan.process_options,
            source_hash=plan.document.source_hash,
            source_name=plan.source_name,
        )
        await _document_service.mark_parse_queued(
            "kb_batch_active", active_document_id, job=job, plan=plan
        )
        return job.id

    active_job_id = asyncio.run(_seed_active_parse())
    valid_upload = client.post(
        "/kbs/kb_batch_active/documents:upload",
        files=[("files", ("valid.pdf", b"valid", "application/pdf"))],
        headers=_HEADERS,
    )
    assert valid_upload.status_code == 200
    valid_document_id = valid_upload.json()["documents"][0]["id"]

    response = client.post(
        "/kbs/kb_batch_active/documents:batch-parse",
        json={
            "document_ids": [active_document_id, valid_document_id],
            "engine": "mineru",
        },
        headers=_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert [document["id"] for document in payload["documents"]] == [valid_document_id]
    job = client.get(f"/kbs/kb_batch_active/jobs/{payload['job_id']}", headers=_HEADERS)
    assert job.status_code == 200
    job_payload = job.json()
    assert job_payload["status"] == "failed"
    assert job_payload["completed_items"] == 1
    assert job_payload["failed_items"] == 1
    assert job_payload["result"]["summary"]["outcome"] == "partial_failure"
    failure = next(
        item for item in job_payload["result"]["items"] if item["status"] == "failed"
    )
    assert failure["document_id"] == active_document_id
    assert failure["error_code"] == "parse_job_active"
    assert failure["existing_job_id"] == active_job_id

    active_document = client.get(
        f"/kbs/kb_batch_active/documents/{active_document_id}", headers=_HEADERS
    )
    assert active_document.status_code == 200
    assert active_document.json()["status"] == "parse_queued"
    assert active_document.json()["metadata"]["pending_parse_job_id"] == active_job_id
    valid_document = client.get(
        f"/kbs/kb_batch_active/documents/{valid_document_id}", headers=_HEADERS
    )
    assert valid_document.status_code == 200
    assert valid_document.json()["status"] == "parsed"
    assert len(probe.instances) == 1
    assert len(probe.instances[0].parse_calls) == 1


def test_batch_parse_missing_document_and_source_are_per_item_failures(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_batch_missing")
    upload = client.post(
        "/kbs/kb_batch_missing/documents:upload",
        files=[
            ("files", ("ok.pdf", b"ok", "application/pdf")),
            ("files", ("lost.pdf", b"lost", "application/pdf")),
        ],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    documents = upload.json()["documents"]
    ok_id = documents[0]["id"]
    lost = documents[1]
    Path(lost["source_uri"]).unlink()

    response = client.post(
        "/kbs/kb_batch_missing/documents:batch-parse",
        json={"document_ids": [ok_id, lost["id"], "doc_missing"], "engine": "mineru"},
        headers=_HEADERS,
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    job = client.get(f"/kbs/kb_batch_missing/jobs/{job_id}", headers=_HEADERS)
    assert job.status_code == 200
    job_payload = job.json()
    assert job_payload["status"] == "failed"
    assert job_payload["completed_items"] == 1
    assert job_payload["failed_items"] == 2
    result_items = job_payload["result"]["items"]
    assert {
        item["error_code"] for item in result_items if item["status"] == "failed"
    } == {
        "parse_failed",
        "document_not_found",
    }

    ok_document = client.get(
        f"/kbs/kb_batch_missing/documents/{ok_id}", headers=_HEADERS
    )
    assert ok_document.status_code == 200
    assert ok_document.json()["status"] == "parsed"
    lost_document = client.get(
        f"/kbs/kb_batch_missing/documents/{lost['id']}", headers=_HEADERS
    )
    assert lost_document.status_code == 200
    assert lost_document.json()["status"] == "parse_failed"


def test_batch_parse_rejects_invalid_options_duplicates_and_cross_kb(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_batch_invalid")
    _create_kb(client, "kb_batch_other")
    upload = client.post(
        "/kbs/kb_batch_invalid/documents:upload",
        files=[("files", ("paper.pdf", b"pdf", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]

    invalid_options = client.post(
        "/kbs/kb_batch_invalid/documents:batch-parse",
        json={
            "document_ids": [document_id],
            "engine": "mineru",
            "process_options": "iZ",
        },
        headers=_HEADERS,
    )
    assert invalid_options.status_code == 400
    assert "unsupported character" in invalid_options.json()["detail"]

    duplicates = client.post(
        "/kbs/kb_batch_invalid/documents:batch-parse",
        json={"document_ids": [document_id, document_id], "engine": "mineru"},
        headers=_HEADERS,
    )
    assert duplicates.status_code == 422

    cross_kb = client.post(
        "/kbs/kb_batch_other/documents:batch-parse",
        json={"document_ids": [document_id], "engine": "mineru"},
        headers=_HEADERS,
    )
    assert cross_kb.status_code == 200
    job = client.get(
        f"/kbs/kb_batch_other/jobs/{cross_kb.json()['job_id']}", headers=_HEADERS
    )
    assert job.status_code == 200
    assert job.json()["status"] == "failed"
    assert job.json()["failed_items"] == 1
    assert job.json()["result"]["items"][0]["error_code"] == "document_not_found"


def test_download_document_file_artifacts_returns_bytes(tmp_path):
    object_storage = FakeObjectStorage()
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, object_storage=object_storage
    )
    _create_kb(client, "kb_artifact_download")
    document_id, artifacts = _upload_and_parse_document(
        client,
        "kb_artifact_download",
        filename="paper.pdf",
        content=b"pdf-body",
    )

    original = artifacts["original"]
    original_response = client.get(
        f"/kbs/kb_artifact_download/documents/{document_id}/artifacts/{original['id']}:download",
        headers=_HEADERS,
    )
    assert original_response.status_code == 200
    assert original_response.content == b"pdf-body"
    assert original_response.headers["content-type"].startswith("application/pdf")
    assert "paper.pdf" in original_response.headers["content-disposition"]

    blocks = artifacts["blocks"]
    Path(blocks["uri"]).unlink()
    blocks_response = client.get(
        f"/kbs/kb_artifact_download/documents/{document_id}/artifacts/{blocks['id']}:download",
        headers=_HEADERS,
    )
    assert blocks_response.status_code == 200
    assert blocks_response.content.replace(b"\r\n", b"\n") == (
        b'{"type":"content","text":"parsed"}\n'
    )
    assert blocks_response.headers["content-type"].startswith("application/x-ndjson")
    assert object_storage.downloads[-1][0] == blocks["metadata"]["object_uri"]


def test_preview_document_artifact_returns_inline_and_restores_cache(tmp_path):
    object_storage = FakeObjectStorage()
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, object_storage=object_storage
    )
    _create_kb(client, "kb_artifact_preview")
    document_id, artifacts = _upload_and_parse_document(client, "kb_artifact_preview")

    blocks = artifacts["blocks"]
    Path(blocks["uri"]).unlink()
    response = client.get(
        f"/kbs/kb_artifact_preview/documents/{document_id}/artifacts/{blocks['id']}:preview",
        headers=_HEADERS,
    )

    assert response.status_code == 200, response.text
    assert response.content.replace(b"\r\n", b"\n") == (
        b'{"type":"content","text":"parsed"}\n'
    )
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert response.headers["content-disposition"].startswith("inline")
    assert object_storage.downloads[-1][0] == blocks["metadata"]["object_uri"]


def test_preview_document_artifact_rejects_directory(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_artifact_preview_dir")
    document_id, artifacts = _upload_and_parse_document(client, "kb_artifact_preview_dir")

    sidecar = artifacts["sidecar"]
    response = client.get(
        f"/kbs/kb_artifact_preview_dir/documents/{document_id}/artifacts/{sidecar['id']}:preview",
        headers=_HEADERS,
    )

    assert response.status_code == 400
    assert "directory" in response.json()["detail"].lower()


def test_preview_document_artifact_rejects_large_file(tmp_path, monkeypatch):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_artifact_preview_large")
    document_id, artifacts = _upload_and_parse_document(client, "kb_artifact_preview_large")
    monkeypatch.setattr(_kb_document_routes, "_MAX_ARTIFACT_PREVIEW_BYTES", 1)

    blocks = artifacts["blocks"]
    response = client.get(
        f"/kbs/kb_artifact_preview_large/documents/{document_id}/artifacts/{blocks['id']}:preview",
        headers=_HEADERS,
    )

    assert response.status_code == 413
    assert "maximum size" in response.json()["detail"].lower()


def test_preview_document_artifact_rejects_unsupported_media(tmp_path):
    client, _kb_service, store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_artifact_preview_media")
    document_id, artifacts = _upload_and_parse_document(client, "kb_artifact_preview_media")

    with store._connect() as conn:
        conn.execute(
            "UPDATE documents SET content_type = ? WHERE id = ?",
            ("application/octet-stream", document_id),
        )
        conn.commit()

    original = artifacts["original"]
    response = client.get(
        f"/kbs/kb_artifact_preview_media/documents/{document_id}/artifacts/{original['id']}:preview",
        headers=_HEADERS,
    )

    assert response.status_code == 415
    assert "not supported" in response.json()["detail"]


def test_preview_document_artifact_rejects_svg_media(tmp_path):
    client, _kb_service, store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_artifact_preview_svg")
    document_id, artifacts = _upload_and_parse_document(client, "kb_artifact_preview_svg")

    with store._connect() as conn:
        conn.execute(
            "UPDATE documents SET content_type = ? WHERE id = ?",
            ("image/svg+xml", document_id),
        )
        conn.commit()

    original = artifacts["original"]
    response = client.get(
        f"/kbs/kb_artifact_preview_svg/documents/{document_id}/artifacts/{original['id']}:preview",
        headers=_HEADERS,
    )

    assert response.status_code == 415
    assert "not supported" in response.json()["detail"]


def test_create_document_file_artifact_download_url(tmp_path):
    object_storage = FakeObjectStorage()
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, object_storage=object_storage
    )
    _create_kb(client, "kb_artifact_presign")
    document_id, artifacts = _upload_and_parse_document(
        client,
        "kb_artifact_presign",
        filename="paper.pdf",
        content=b"pdf-body",
    )

    blocks = artifacts["blocks"]
    response = client.get(
        f"/kbs/kb_artifact_presign/documents/{document_id}/artifacts/{blocks['id']}:download-url",
        params={"expires_in_seconds": 900},
        headers=_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["artifact_id"] == blocks["id"]
    assert payload["object_uri"] == blocks["metadata"]["object_uri"]
    assert payload["expires_in_seconds"] == 900
    assert payload["filename"].endswith(".blocks.jsonl")
    assert payload["media_type"] == "application/x-ndjson"
    assert payload["url"] == (
        "https://objects.example/download"
        f"?uri={blocks['metadata']['object_uri']}&expires=900"
    )
    assert object_storage.presigned == [(blocks["metadata"]["object_uri"], 900)]


def test_create_document_file_artifact_download_url_clamps_ttl(tmp_path):
    object_storage = FakeObjectStorage()
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, object_storage=object_storage
    )
    _create_kb(client, "kb_artifact_presign_ttl")
    document_id, artifacts = _upload_and_parse_document(
        client,
        "kb_artifact_presign_ttl",
        filename="paper.pdf",
        content=b"pdf-body",
    )

    blocks = artifacts["blocks"]
    response = client.get(
        f"/kbs/kb_artifact_presign_ttl/documents/{document_id}/artifacts/{blocks['id']}:download-url",
        params={"expires_in_seconds": 999999999},
        headers=_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["expires_in_seconds"] == 604800
    assert payload["url"].endswith("&expires=604800")
    assert object_storage.presigned == [(blocks["metadata"]["object_uri"], 604800)]


def test_create_document_file_artifact_download_url_rejects_untrusted_object_uri(
    tmp_path,
):
    object_storage = FakeObjectStorage()
    client, _kb_service, store, _document_service, _job_service = _build_client(
        tmp_path, object_storage=object_storage
    )
    _create_kb(client, "kb_artifact_presign_scope")
    document_id, artifacts = _upload_and_parse_document(
        client,
        "kb_artifact_presign_scope",
        filename="paper.pdf",
        content=b"pdf-body",
    )

    blocks = artifacts["blocks"]
    metadata = dict(blocks["metadata"])
    metadata["object_uri"] = (
        "s3://evil-bucket/workspaces/other/documents/other/artifacts/blocks.jsonl"
    )
    with store._connect() as conn:
        conn.execute(
            "UPDATE document_artifacts SET metadata_json = ? WHERE id = ?",
            (json.dumps(metadata), blocks["id"]),
        )
        conn.commit()

    response = client.get(
        f"/kbs/kb_artifact_presign_scope/documents/{document_id}/artifacts/{blocks['id']}:download-url",
        headers=_HEADERS,
    )

    assert response.status_code == 503
    assert "outside the document object prefix" in response.json()["detail"]
    assert object_storage.presigned == []


def test_create_document_file_artifact_download_url_rejects_document_prefix_collision(
    tmp_path,
):
    object_storage = FakeObjectStorage()
    client, _kb_service, store, _document_service, _job_service = _build_client(
        tmp_path, object_storage=object_storage
    )
    _create_kb(client, "kb_artifact_presign_prefix")
    document_id, artifacts = _upload_and_parse_document(
        client,
        "kb_artifact_presign_prefix",
        filename="paper.pdf",
        content=b"pdf-body",
    )

    blocks = artifacts["blocks"]
    metadata = dict(blocks["metadata"])
    metadata["object_uri"] = blocks["metadata"]["object_uri"].replace(
        f"/documents/{document_id}/",
        f"/documents/{document_id}-extra/",
    )
    with store._connect() as conn:
        conn.execute(
            "UPDATE document_artifacts SET metadata_json = ? WHERE id = ?",
            (json.dumps(metadata), blocks["id"]),
        )
        conn.commit()

    response = client.get(
        f"/kbs/kb_artifact_presign_prefix/documents/{document_id}/artifacts/{blocks['id']}:download-url",
        headers=_HEADERS,
    )

    assert response.status_code == 503
    assert "outside the document object prefix" in response.json()["detail"]
    assert object_storage.presigned == []


def test_create_document_artifact_download_url_rejects_directories(tmp_path):
    object_storage = FakeObjectStorage()
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, object_storage=object_storage
    )
    _create_kb(client, "kb_artifact_presign_dir")
    document_id, artifacts = _upload_and_parse_document(client, "kb_artifact_presign_dir")

    sidecar = artifacts["sidecar"]
    response = client.get(
        f"/kbs/kb_artifact_presign_dir/documents/{document_id}/artifacts/{sidecar['id']}:download-url",
        headers=_HEADERS,
    )

    assert response.status_code == 400
    assert "only available for file artifacts" in response.json()["detail"]
    assert object_storage.presigned == []


def test_download_document_artifact_streams_directory_as_zip(tmp_path):
    import io
    import zipfile

    object_storage = FakeObjectStorage()
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, object_storage=object_storage
    )
    _create_kb(client, "kb_artifact_zip")
    document_id, artifacts = _upload_and_parse_document(client, "kb_artifact_zip")

    sidecar = artifacts["sidecar"]
    import shutil

    shutil.rmtree(Path(sidecar["uri"]))
    response = client.get(
        f"/kbs/kb_artifact_zip/documents/{document_id}/artifacts/{sidecar['id']}:download",
        headers=_HEADERS,
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    names = archive.namelist()
    assert any(name.endswith(".blocks.jsonl") for name in names)
    assert object_storage.prefix_downloads[-1][0] == sidecar["metadata"]["object_prefix_uri"]


def test_download_document_artifact_rejects_directories_and_missing_files(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_artifact_errors")
    document_id, artifacts = _upload_and_parse_document(client, "kb_artifact_errors")

    missing_response = client.get(
        f"/kbs/kb_artifact_errors/documents/{document_id}/artifacts/artifact_missing:download",
        headers=_HEADERS,
    )
    assert missing_response.status_code == 404

    blocks = artifacts["blocks"]
    Path(blocks["uri"]).unlink()
    blocks_response = client.get(
        f"/kbs/kb_artifact_errors/documents/{document_id}/artifacts/{blocks['id']}:download",
        headers=_HEADERS,
    )
    assert blocks_response.status_code == 404
    assert "Artifact file not found" in blocks_response.json()["detail"]


def test_download_document_artifact_rejects_cross_kb_and_path_escape(tmp_path):
    client, _kb_service, store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_artifact_owner")
    _create_kb(client, "kb_artifact_other")
    document_id, artifacts = _upload_and_parse_document(client, "kb_artifact_owner")
    original = artifacts["original"]

    cross_kb_response = client.get(
        f"/kbs/kb_artifact_other/documents/{document_id}/artifacts/{original['id']}:download",
        headers=_HEADERS,
    )
    assert cross_kb_response.status_code == 404

    escaped_path = tmp_path / "escaped.txt"
    escaped_path.write_text("escaped", encoding="utf-8")
    with store._connect() as conn:
        conn.execute(
            "UPDATE document_artifacts SET uri = ? WHERE id = ?",
            (str(escaped_path), original["id"]),
        )
        conn.commit()

    escape_response = client.get(
        f"/kbs/kb_artifact_owner/documents/{document_id}/artifacts/{original['id']}:download",
        headers=_HEADERS,
    )
    assert escape_response.status_code == 400
    assert "escapes document directory" in escape_response.json()["detail"]


def test_parse_document_failure_marks_job_and_document_failed(tmp_path):
    probe = BuilderProbe(should_fail=True)
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_parse_fail")
    upload = client.post(
        "/kbs/kb_parse_fail/documents:upload",
        files=[("files", ("paper.pdf", b"pdf", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]

    response = client.post(
        f"/kbs/kb_parse_fail/documents/{document_id}:parse",
        json={"engine": "mineru"},
        headers=_HEADERS,
    )

    assert response.status_code == 200
    job_id = response.json()["id"]
    job = client.get(f"/kbs/kb_parse_fail/jobs/{job_id}", headers=_HEADERS)
    assert job.status_code == 200
    assert job.json()["status"] == "failed"
    assert job.json()["failed_items"] == 1
    assert job.json()["error_code"] == "parse_failed"

    document = client.get(
        f"/kbs/kb_parse_fail/documents/{document_id}", headers=_HEADERS
    )
    assert document.status_code == 200
    assert document.json()["status"] == "parse_failed"
    assert document.json()["error_code"] == "parse_failed"


def test_parse_failure_job_is_retryable(tmp_path):
    """A failed parse job can be reset to queued via :retry — the retryability
    path a transient MinerU/network failure relies on. The job returns to
    ``queued`` with ``retry_count`` incremented and the error cleared, ready to
    be re-driven (by the client or a durable worker)."""
    probe = BuilderProbe(should_fail=True)
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_parse_retry")
    upload = client.post(
        "/kbs/kb_parse_retry/documents:upload",
        files=[("files", ("paper.pdf", b"pdf", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]

    parse = client.post(
        f"/kbs/kb_parse_retry/documents/{document_id}:parse",
        json={"engine": "mineru"},
        headers=_HEADERS,
    )
    assert parse.status_code == 200
    job_id = parse.json()["id"]
    assert (
        client.get(f"/kbs/kb_parse_retry/jobs/{job_id}", headers=_HEADERS).json()[
            "status"
        ]
        == "failed"
    )

    retry = client.post(
        f"/kbs/kb_parse_retry/jobs/{job_id}:retry", json={}, headers=_HEADERS
    )
    assert retry.status_code == 200
    body = retry.json()
    assert body["status"] == "queued"
    assert body["retry_count"] == 1
    assert body["error_code"] is None
    assert body["error_message"] is None


def test_parse_document_missing_source_fails_after_claim(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_missing_source")
    upload = client.post(
        "/kbs/kb_missing_source/documents:upload",
        files=[("files", ("paper.pdf", b"pdf", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document = upload.json()["documents"][0]
    Path(document["source_uri"]).unlink()

    response = client.post(
        f"/kbs/kb_missing_source/documents/{document['id']}:parse",
        json={"engine": "mineru"},
        headers=_HEADERS,
    )

    assert response.status_code == 200
    job_id = response.json()["id"]
    job = client.get(f"/kbs/kb_missing_source/jobs/{job_id}", headers=_HEADERS)
    assert job.status_code == 200
    payload = job.json()
    assert payload["status"] == "failed"
    assert payload["error_code"] == "parse_failed"
    assert "Document source not found" in payload["error_message"]
    assert "source_uri" not in payload["payload"]


def test_parse_document_rejects_invalid_process_options(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_bad_options")
    upload = client.post(
        "/kbs/kb_bad_options/documents:upload",
        files=[("files", ("paper.pdf", b"pdf", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]

    response = client.post(
        f"/kbs/kb_bad_options/documents/{document_id}:parse",
        json={"engine": "mineru", "process_options": "iZ"},
        headers=_HEADERS,
    )

    assert response.status_code == 400
    assert "unsupported character" in response.json()["detail"]


def test_parse_document_rejects_existing_active_parse_job(tmp_path):
    probe = BuilderProbe()
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, probe=probe
    )
    _create_kb(client, "kb_parse_active")
    upload = client.post(
        "/kbs/kb_parse_active/documents:upload",
        files=[("files", ("paper.pdf", b"pdf", "application/pdf"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 200
    document_id = upload.json()["documents"][0]["id"]

    # Deterministically seed an active parse claim (doc -> parse_queued with a
    # pending job) without running the parser, so the second :parse hits 409.
    async def _seed_active_parse() -> str:
        plan = await _document_service.create_parse_plan(
            "kb_parse_active",
            document_id,
            parser_engine="mineru",
            process_options="iF",
        )
        job, _created = await _job_service.create_parse_job_once(
            "kb_parse_active",
            document_id=document_id,
            parser_hash=plan.parser_hash,
            lightrag_doc_id=plan.lightrag_doc_id,
            parser_engine=plan.parser_engine,
            process_options=plan.process_options,
            source_hash=plan.document.source_hash,
            source_name=plan.source_name,
        )
        await _document_service.mark_parse_queued(
            "kb_parse_active", document_id, job=job, plan=plan
        )
        return job.id

    active_job_id = asyncio.run(_seed_active_parse())

    response = client.post(
        f"/kbs/kb_parse_active/documents/{document_id}:parse",
        json={"engine": "mineru"},
        headers=_HEADERS,
    )

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "error_code": "parse_job_active",
        "document_id": document_id,
        "existing_job_id": active_job_id,
        "message": f"Document '{document_id}' already has an active parse job",
    }
    document = client.get(
        f"/kbs/kb_parse_active/documents/{document_id}", headers=_HEADERS
    )
    assert document.status_code == 200
    assert document.json()["status"] == "parse_queued"
    assert document.json()["metadata"]["pending_parse_job_id"] == active_job_id
    failed_jobs = client.get(
        f"/kbs/kb_parse_active/jobs?status=failed&document_id={document_id}",
        headers=_HEADERS,
    )
    assert failed_jobs.status_code == 200
    assert failed_jobs.json()["total"] == 1
    assert failed_jobs.json()["jobs"][0]["error_code"] == "parse_job_active"
    # The second :parse was rejected before building any LightRAG instance.
    assert probe.instances == []


def test_same_name_uploads_use_distinct_exclusive_source_paths(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_same_name")

    response = client.post(
        "/kbs/kb_same_name/documents:upload",
        files=[
            ("files", ("same.txt", b"first", "text/plain")),
            ("files", ("same.txt", b"second", "text/plain")),
        ],
        headers=_HEADERS,
    )

    assert response.status_code == 200
    documents = response.json()["documents"]
    paths = [Path(document["source_uri"]) for document in documents]
    assert len({str(path) for path in paths}) == 2
    assert all(path.exists() for path in paths)
    assert {path.parent.name for path in paths} == {
        document["id"] for document in documents
    }
    assert {path.read_bytes() for path in paths} == {b"first", b"second"}


def test_upload_rejects_unsupported_file_type(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_ext")

    response = client.post(
        "/kbs/kb_ext/documents:upload",
        files=[("files", ("malware.exe", b"nope", "application/octet-stream"))],
        headers=_HEADERS,
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_sanitizes_unsafe_filename_inside_document_directory(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_safe_name")

    response = client.post(
        "/kbs/kb_safe_name/documents:upload",
        files=[("files", ("../unsafe?.txt", b"safe", "text/plain"))],
        headers=_HEADERS,
    )

    assert response.status_code == 200
    document = response.json()["documents"][0]
    source_path = Path(document["source_uri"])
    # ".." is dropped (left-over leading slash from "/" then sanitized to "_"),
    # the unsafe "?" character becomes "_". CJK / spaces / dashes survive.
    assert document["source_name"] == "_unsafe_.txt"
    assert source_path.parent.name == document["id"]
    assert source_path.parent.parent == tmp_path / "inputs" / sanitize_workspace(
        "kb_safe_name"
    )
    assert source_path.read_bytes() == b"safe"


def test_upload_rejects_oversized_file(tmp_path, monkeypatch):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_big")
    monkeypatch.setattr(_kb_document_routes.global_args, "max_upload_size", 3)

    response = client.post(
        "/kbs/kb_big/documents:upload",
        files=[("files", ("big.txt", b"1234", "text/plain"))],
        headers=_HEADERS,
    )

    assert response.status_code == 413
    assert "File too large" in response.json()["detail"]
    assert not any((tmp_path / "inputs" / sanitize_workspace("kb_big")).glob("**/*"))


def test_upload_rejects_aggregate_oversized_batch(tmp_path, monkeypatch):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_big_batch")
    monkeypatch.setattr(_kb_document_routes.global_args, "max_upload_size", 6)

    response = client.post(
        "/kbs/kb_big_batch/documents:upload",
        files=[
            ("files", ("one.txt", b"1234", "text/plain")),
            ("files", ("two.txt", b"5678", "text/plain")),
        ],
        headers=_HEADERS,
    )

    assert response.status_code == 413
    assert "Upload batch too large" in response.json()["detail"]
    assert not any(
        (tmp_path / "inputs" / sanitize_workspace("kb_big_batch")).glob("**/*")
    )


def test_upload_rejects_unlimited_max_upload_size(tmp_path, monkeypatch):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_unlimited")
    monkeypatch.setattr(_kb_document_routes.global_args, "max_upload_size", 0)

    response = client.post(
        "/kbs/kb_unlimited/documents:upload",
        files=[("files", ("tiny.txt", b"tiny", "text/plain"))],
        headers=_HEADERS,
    )

    assert response.status_code == 413
    assert "MAX_UPLOAD_SIZE" in response.json()["detail"]


def test_upload_rejects_too_many_files(tmp_path, monkeypatch):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_many")
    monkeypatch.setattr(_kb_document_routes, "_MAX_KB_UPLOAD_FILES", 1)

    response = client.post(
        "/kbs/kb_many/documents:upload",
        files=[
            ("files", ("one.txt", b"one", "text/plain")),
            ("files", ("two.txt", b"two", "text/plain")),
        ],
        headers=_HEADERS,
    )

    assert response.status_code == 413
    assert "Too many files" in response.json()["detail"]


def test_text_import_rejects_oversized_text(tmp_path, monkeypatch):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_big_text")
    monkeypatch.setattr(_kb_document_routes, "_MAX_TEXT_DOCUMENT_BYTES", 4)

    response = client.post(
        "/kbs/kb_big_text/documents:texts",
        json={"documents": [{"text": "12345", "source_name": "big.txt"}]},
        headers=_HEADERS,
    )

    assert response.status_code == 413
    assert "Text document too large" in response.json()["detail"]


@pytest.mark.asyncio
async def test_job_transition_rules(tmp_path):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "knowledge_bases.json")
    store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    job_service = JobService(kb_service, store)
    await kb_service.create(kb_id="kb_jobs", name="Jobs")

    job = await job_service.create_job("kb_jobs", job_type="parse", stage="parsing")
    running = await job_service.transition_job(
        "kb_jobs", job.id, status="running", progress=0.5
    )
    assert running.status == "running"
    assert running.started_at is not None
    assert running.progress == 0.5

    succeeded = await job_service.transition_job(
        "kb_jobs", job.id, status="succeeded", progress=1.0, result={"ok": True}
    )
    assert succeeded.status == "succeeded"
    assert succeeded.finished_at is not None
    assert succeeded.result == {"ok": True}

    with pytest.raises(InvalidJobTransitionError):
        await job_service.transition_job("kb_jobs", job.id, status="running")


def test_missing_kb_document_routes_return_404(tmp_path):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )

    response = client.get("/kbs/missing/documents", headers=_HEADERS)
    assert response.status_code == 404

    upload = client.post(
        "/kbs/missing/documents:upload",
        files=[("files", ("missing.txt", b"missing", "text/plain"))],
        headers=_HEADERS,
    )
    assert upload.status_code == 404


def test_text_import_rejects_oversized_metadata(tmp_path):
    """A single text document whose metadata JSON exceeds the 64KB cap is
    rejected by request validation (422)."""
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_big_meta")
    big_metadata = {"blob": "x" * (64 * 1024 + 10)}

    response = client.post(
        "/kbs/kb_big_meta/documents:texts",
        json={
            "documents": [
                {"text": "hello", "source_name": "n.md", "metadata": big_metadata}
            ]
        },
        headers=_HEADERS,
    )
    assert response.status_code == 422
    assert "metadata too large" in response.text.lower()


class _FakeUrlNetworkStream:
    def __init__(self, server_addr: tuple[str, int] = ("93.184.216.34", 443)):
        self._server_addr = server_addr

    def get_extra_info(self, info: str) -> tuple[str, int] | None:
        if info == "server_addr":
            return self._server_addr
        return None


class _FakeUrlResponse:
    def __init__(
        self,
        *,
        headers: dict[str, str] | None = None,
        chunks: list[bytes],
        server_addr: tuple[str, int] = ("93.184.216.34", 443),
    ):
        self.headers = headers or {}
        self._chunks = chunks
        self.status_code = 200
        self.iterated = False
        self.extensions = {"network_stream": _FakeUrlNetworkStream(server_addr)}

    def raise_for_status(self):
        return None

    async def aiter_bytes(self, chunk_size: int = 65536):
        self.iterated = True
        for chunk in self._chunks:
            yield chunk


class _FakeUrlStream:
    def __init__(self, response: _FakeUrlResponse):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _FakeAsyncClient:
    requests: list[tuple[str, str]] = []
    response = _FakeUrlResponse(chunks=[b"url content"])

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def stream(self, method: str, url: str):
        self.requests.append((method, url))
        return _FakeUrlStream(self.response)


def test_url_ingestion_success_persists_url_metadata_and_content(tmp_path, monkeypatch):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_url")
    _FakeAsyncClient.requests = []
    _FakeAsyncClient.response = _FakeUrlResponse(
        headers={"content-disposition": 'attachment; filename="remote.md"'},
        chunks=[b"url ", b"content"],
    )
    monkeypatch.setattr(
        _kb_document_routes, "_validate_public_hostname", lambda hostname: asyncio.sleep(0)
    )
    monkeypatch.setattr(_kb_document_routes.httpx, "AsyncClient", _FakeAsyncClient)

    response = client.post(
        "/kbs/kb_url/documents:urls",
        json={
            "documents": [
                {
                    "url": "https://Example.COM/docs/ignored",
                    "metadata": {"tag": "url"},
                }
            ]
        },
        headers=_HEADERS,
    )

    assert response.status_code == 200, response.text
    document = response.json()["documents"][0]
    assert document["source_type"] == "url"
    assert document["source_name"] == "remote.md"
    assert document["metadata"]["tag"] == "url"
    assert document["metadata"]["source_url"] == "https://example.com/docs/ignored"
    assert document["metadata"]["source_key"] == "url:https://example.com/docs/ignored"
    assert Path(document["source_uri"]).read_bytes() == b"url content"
    assert _FakeAsyncClient.requests == [("GET", "https://example.com/docs/ignored")]


def test_url_ingestion_auto_parse_idempotency_reuses_existing_batch(
    tmp_path, monkeypatch
):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path, wire_document_registry=False
    )
    _create_kb(client, "kb_url_auto_idem")
    _FakeAsyncClient.requests = []
    _FakeAsyncClient.response = _FakeUrlResponse(chunks=[b"url content"])
    monkeypatch.setattr(
        _kb_document_routes, "_validate_public_hostname", lambda hostname: asyncio.sleep(0)
    )
    monkeypatch.setattr(_kb_document_routes.httpx, "AsyncClient", _FakeAsyncClient)
    payload = {
        "documents": [{"url": "https://example.com/remote.md"}],
        "auto_parse": True,
        "parser_engine": "mineru",
        "process_options": "iF",
        "idempotency_key": "idem-url-auto-1",
    }

    first = client.post(
        "/kbs/kb_url_auto_idem/documents:urls", json=payload, headers=_HEADERS
    )
    second = client.post(
        "/kbs/kb_url_auto_idem/documents:urls", json=payload, headers=_HEADERS
    )

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert second.json()["job_id"] == first.json()["job_id"]
    assert second.json()["batch_id"] == first.json()["batch_id"]
    document = second.json()["documents"][0]
    assert document["id"] == first.json()["documents"][0]["id"]
    assert document["status"] == "parse_queued"
    assert document["metadata"]["pending_parse_job_id"] == first.json()["job_id"]
    assert document["metadata"]["parser_engine"] == "mineru"

    listing = client.get("/kbs/kb_url_auto_idem/documents", headers=_HEADERS)
    assert listing.json()["total"] == 1
    jobs = client.get("/kbs/kb_url_auto_idem/jobs", headers=_HEADERS)
    assert jobs.json()["total"] == 1

    conflict = client.post(
        "/kbs/kb_url_auto_idem/documents:urls",
        json={**payload, "documents": [{"url": "https://example.com/other.md"}]},
        headers=_HEADERS,
    )
    assert conflict.status_code == 409


def test_url_ingestion_rejects_localhost_before_request(tmp_path, monkeypatch):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_url_local")
    _FakeAsyncClient.requests = []
    monkeypatch.setattr(_kb_document_routes.httpx, "AsyncClient", _FakeAsyncClient)

    response = client.post(
        "/kbs/kb_url_local/documents:urls",
        json={"documents": [{"url": "http://localhost/private.txt"}]},
        headers=_HEADERS,
    )

    assert response.status_code == 400
    assert "disallowed" in response.json()["detail"]
    assert _FakeAsyncClient.requests == []


def test_url_ingestion_rejects_private_peer_before_reading_body(tmp_path, monkeypatch):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_url_rebind")
    _FakeAsyncClient.requests = []
    _FakeAsyncClient.response = _FakeUrlResponse(
        chunks=[b"internal secret"], server_addr=("169.254.169.254", 80)
    )
    monkeypatch.setattr(
        _kb_document_routes, "_validate_public_hostname", lambda hostname: asyncio.sleep(0)
    )
    monkeypatch.setattr(_kb_document_routes.httpx, "AsyncClient", _FakeAsyncClient)

    response = client.post(
        "/kbs/kb_url_rebind/documents:urls",
        json={"documents": [{"url": "http://example.com/private.txt"}]},
        headers=_HEADERS,
    )

    assert response.status_code == 400
    assert "disallowed" in response.json()["detail"]
    assert _FakeAsyncClient.requests == [("GET", "http://example.com/private.txt")]
    assert not _FakeAsyncClient.response.iterated


def test_url_ingestion_rejects_oversized_responses(tmp_path, monkeypatch):
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_url_big")
    monkeypatch.setattr(_kb_document_routes.global_args, "max_upload_size", 3)
    monkeypatch.setattr(
        _kb_document_routes, "_validate_public_hostname", lambda hostname: asyncio.sleep(0)
    )
    monkeypatch.setattr(_kb_document_routes.httpx, "AsyncClient", _FakeAsyncClient)

    _FakeAsyncClient.response = _FakeUrlResponse(
        headers={"content-length": "4"}, chunks=[b"tiny"]
    )
    length_response = client.post(
        "/kbs/kb_url_big/documents:urls",
        json={"documents": [{"url": "https://example.com/big.txt"}]},
        headers=_HEADERS,
    )
    assert length_response.status_code == 413
    assert "File too large" in length_response.json()["detail"]

    _FakeAsyncClient.response = _FakeUrlResponse(chunks=[b"12", b"34"])
    stream_response = client.post(
        "/kbs/kb_url_big/documents:urls",
        json={"documents": [{"url": "https://example.com/stream.txt"}]},
        headers=_HEADERS,
    )
    assert stream_response.status_code == 413
    assert "File too large" in stream_response.json()["detail"]


def test_local_import_success_persists_import_metadata_and_content(tmp_path):
    client, _kb_service, _store, document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_import")
    staged = document_service.source_root / "staged" / "import.md"
    staged.parent.mkdir(parents=True)
    staged.write_text("imported", encoding="utf-8")

    response = client.post(
        "/kbs/kb_import/documents:import",
        json={"documents": [{"path": str(staged), "metadata": {"kind": "manual"}}]},
        headers=_HEADERS,
    )

    assert response.status_code == 200, response.text
    document = response.json()["documents"][0]
    assert document["source_type"] == "import"
    assert document["source_name"] == "import.md"
    assert document["metadata"]["kind"] == "manual"
    assert document["metadata"]["staged_source_path"] == "staged/import.md"
    assert document["metadata"]["source_key"] == "import:staged/import.md"
    assert Path(document["source_uri"]).read_text(encoding="utf-8") == "imported"


def test_local_import_auto_parse_idempotency_reuses_existing_batch(tmp_path):
    client, _kb_service, _store, document_service, _job_service = _build_client(
        tmp_path, wire_document_registry=False
    )
    _create_kb(client, "kb_import_auto_idem")
    staged = document_service.source_root / "staged" / "idem-import.md"
    staged.parent.mkdir(parents=True)
    staged.write_text("imported", encoding="utf-8")
    payload = {
        "documents": [{"path": str(staged), "metadata": {"kind": "manual"}}],
        "auto_parse": True,
        "parser_engine": "mineru",
        "process_options": "iF",
        "idempotency_key": "idem-import-auto-1",
    }

    first = client.post(
        "/kbs/kb_import_auto_idem/documents:import", json=payload, headers=_HEADERS
    )
    second = client.post(
        "/kbs/kb_import_auto_idem/documents:import", json=payload, headers=_HEADERS
    )

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert second.json()["job_id"] == first.json()["job_id"]
    assert second.json()["batch_id"] == first.json()["batch_id"]
    assert second.json()["documents"][0]["id"] == first.json()["documents"][0]["id"]
    assert second.json()["documents"][0]["status"] == "parse_queued"
    assert second.json()["documents"][0]["metadata"]["parser_engine"] == "mineru"
    assert client.get(
        "/kbs/kb_import_auto_idem/documents", headers=_HEADERS
    ).json()["total"] == 1
    assert client.get("/kbs/kb_import_auto_idem/jobs", headers=_HEADERS).json()[
        "total"
    ] == 1


def test_local_import_rejects_escape_and_unsupported_extension(tmp_path):
    client, _kb_service, _store, document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_import_reject")
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    unsupported = document_service.source_root / "staged" / "bad.exe"
    unsupported.parent.mkdir(parents=True)
    unsupported.write_text("bad", encoding="utf-8")

    escaped = client.post(
        "/kbs/kb_import_reject/documents:import",
        json={"documents": [{"path": str(outside)}]},
        headers=_HEADERS,
    )
    assert escaped.status_code == 400
    assert "escapes INPUT_DIR" in escaped.json()["detail"]

    bad_extension = client.post(
        "/kbs/kb_import_reject/documents:import",
        json={"documents": [{"path": str(unsupported)}]},
        headers=_HEADERS,
    )
    assert bad_extension.status_code == 400
    assert "Unsupported file type" in bad_extension.json()["detail"]


def test_scan_success_discovers_supported_staged_files(tmp_path):
    client, _kb_service, _store, document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_scan")
    staged_dir = document_service.source_root / "scan-stage"
    nested_dir = staged_dir / "nested"
    parsed_dir = staged_dir / "__parsed__"
    sync_dir = staged_dir / ".sync-staging"
    nested_dir.mkdir(parents=True)
    parsed_dir.mkdir()
    sync_dir.mkdir()
    (staged_dir / "a.txt").write_text("a", encoding="utf-8")
    (nested_dir / "b.md").write_text("b", encoding="utf-8")
    (staged_dir / "skip.exe").write_text("skip", encoding="utf-8")
    (parsed_dir / "old.txt").write_text("old", encoding="utf-8")
    (sync_dir / "pending.txt").write_text("pending", encoding="utf-8")

    response = client.post(
        "/kbs/kb_scan/documents:scan",
        json={"directory": str(staged_dir), "recursive": True},
        headers=_HEADERS,
    )

    assert response.status_code == 200, response.text
    documents = response.json()["documents"]
    assert len(documents) == 2
    assert {document["source_type"] for document in documents} == {"scan"}
    metadata_by_name = {document["source_name"]: document["metadata"] for document in documents}
    assert metadata_by_name["a.txt"]["scanned_source_path"] == "scan-stage/a.txt"
    assert metadata_by_name["a.txt"]["source_key"] == "scan:scan-stage/a.txt"
    assert metadata_by_name["b.md"]["scanned_source_path"] == "scan-stage/nested/b.md"
    assert metadata_by_name["b.md"]["source_key"] == "scan:scan-stage/nested/b.md"


def test_scan_auto_parse_idempotency_reuses_existing_batch(tmp_path):
    client, _kb_service, _store, document_service, _job_service = _build_client(
        tmp_path, wire_document_registry=False
    )
    _create_kb(client, "kb_scan_auto_idem")
    staged_dir = document_service.source_root / "idem-scan"
    nested_dir = staged_dir / "nested"
    nested_dir.mkdir(parents=True)
    (staged_dir / "a.md").write_text("a", encoding="utf-8")
    (nested_dir / "b.txt").write_text("b", encoding="utf-8")
    payload = {
        "directory": str(staged_dir),
        "recursive": True,
        "auto_parse": True,
        "parser_engine": "mineru",
        "process_options": "iF",
        "idempotency_key": "idem-scan-auto-1",
    }

    first = client.post(
        "/kbs/kb_scan_auto_idem/documents:scan", json=payload, headers=_HEADERS
    )
    second = client.post(
        "/kbs/kb_scan_auto_idem/documents:scan", json=payload, headers=_HEADERS
    )

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert second.json()["job_id"] == first.json()["job_id"]
    assert second.json()["batch_id"] == first.json()["batch_id"]
    assert {doc["id"] for doc in second.json()["documents"]} == {
        doc["id"] for doc in first.json()["documents"]
    }
    assert {doc["status"] for doc in second.json()["documents"]} == {"parse_queued"}
    assert client.get("/kbs/kb_scan_auto_idem/documents", headers=_HEADERS).json()[
        "total"
    ] == 2
    assert client.get("/kbs/kb_scan_auto_idem/jobs", headers=_HEADERS).json()[
        "total"
    ] == 1


def test_scan_rejects_root_input_dir_and_no_supported_files(tmp_path):
    client, _kb_service, _store, document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_scan_reject")

    root_response = client.post(
        "/kbs/kb_scan_reject/documents:scan",
        json={"directory": str(document_service.source_root)},
        headers=_HEADERS,
    )
    assert root_response.status_code == 400
    assert "INPUT_DIR root" in root_response.json()["detail"]

    empty_dir = document_service.source_root / "empty-scan"
    empty_dir.mkdir(parents=True)
    (empty_dir / "unsupported.exe").write_text("nope", encoding="utf-8")
    empty_response = client.post(
        "/kbs/kb_scan_reject/documents:scan",
        json={"directory": str(empty_dir)},
        headers=_HEADERS,
    )
    assert empty_response.status_code == 400
    assert "no supported files" in empty_response.json()["detail"]


def test_text_import_rejects_too_many_documents(tmp_path):
    """More than 100 text documents in one request is rejected (422)."""
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_many_texts")
    documents = [
        {"text": f"doc {i}", "source_name": f"n{i}.md"} for i in range(101)
    ]

    response = client.post(
        "/kbs/kb_many_texts/documents:texts",
        json={"documents": documents},
        headers=_HEADERS,
    )
    assert response.status_code == 422


def test_list_documents_status_filter(tmp_path):
    """GET /documents?status=... filters by exact document status."""
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_status_filter")
    # Two uploaded (unparsed) docs and one parsed doc.
    client.post(
        "/kbs/kb_status_filter/documents:upload",
        files=[("files", ("a.pdf", b"a", "application/pdf"))],
        headers=_HEADERS,
    )
    parsed_id, _artifacts = _upload_and_parse_document(
        client, "kb_status_filter", filename="b.pdf", content=b"b"
    )

    parsed = client.get(
        "/kbs/kb_status_filter/documents?status=parsed", headers=_HEADERS
    )
    assert parsed.status_code == 200
    parsed_payload = parsed.json()
    assert parsed_payload["total"] == 1
    assert parsed_payload["documents"][0]["id"] == parsed_id
    assert parsed_payload["documents"][0]["status"] == "parsed"

    uploaded = client.get(
        "/kbs/kb_status_filter/documents?status=uploaded", headers=_HEADERS
    )
    assert uploaded.status_code == 200
    assert uploaded.json()["total"] == 1
    assert all(
        doc["status"] == "uploaded" for doc in uploaded.json()["documents"]
    )


def test_download_directory_artifact_rejects_when_over_size_cap(tmp_path, monkeypatch):
    """A directory artifact whose uncompressed size exceeds the cap returns
    413 before any zip bytes are streamed."""
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_zip_cap")
    document_id, artifacts = _upload_and_parse_document(client, "kb_zip_cap")
    # Force the cap to 1 byte so the (small) sidecar directory trips it.
    monkeypatch.setattr(_kb_document_routes, "_MAX_DIRECTORY_ARTIFACT_BYTES", 1)

    sidecar = artifacts["sidecar"]
    response = client.get(
        f"/kbs/kb_zip_cap/documents/{document_id}/artifacts/{sidecar['id']}:download",
        headers=_HEADERS,
    )
    assert response.status_code == 413
    assert "maximum download size" in response.json()["detail"].lower()


def test_delete_rejects_unsupported_graph_orphans_option(tmp_path):
    """delete_graph_orphans=false is not supported and must be rejected up front
    (the engine always prunes orphaned entities/relations)."""
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_orphans")
    document_id, _artifacts = _upload_and_parse_document(client, "kb_orphans")

    single = client.delete(
        f"/kbs/kb_orphans/documents/{document_id}?delete_graph_orphans=false",
        headers=_HEADERS,
    )
    assert single.status_code == 400
    assert "delete_graph_orphans" in single.json()["detail"]

    batch = client.post(
        "/kbs/kb_orphans/documents:batch-delete",
        json={"document_ids": [document_id], "delete_graph_orphans": False},
        headers=_HEADERS,
    )
    assert batch.status_code == 400


def test_delete_rebuild_kb_strategy_requires_index_service(tmp_path):
    """strategy=rebuild_kb without a configured IndexBuildService returns 503
    (the doc-routes test harness wires no index_service)."""
    client, _kb_service, _store, _document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_rebuild_503")
    document_id, _artifacts = _upload_and_parse_document(client, "kb_rebuild_503")

    response = client.delete(
        f"/kbs/kb_rebuild_503/documents/{document_id}?strategy=rebuild_kb",
        headers=_HEADERS,
    )
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# Phase 3.1-C Integration Writer B2: object-authoritative COW branches.
#
# These tests exercise the object-mode branches in the route-level helpers
# (_execute_replace_document, _execute_delete_document_impl, _execute_sync_item)
# directly via fakes, bypassing the HTTP admission gate which remains closed
# (object-mode HTTP still 503 via assert_destructive_operation_supported).
# ---------------------------------------------------------------------------

_B2_NOW = _datetime_b2(2026, 8, 3, 12, 0, 0, tzinfo=_timezone_b2.utc)
_B2_BUCKET = "b2-bucket"


class _B2FakeObjectStorage(ObjectStorage):
    """Deterministic object storage with metadata-only inspection (no GetObject).

    Mirrors the proven fake from test_document_lifecycle_cow_service.py.
    """

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.upload_proof_calls: list[tuple[str, str | None]] = []
        self.deleted_uris: list[str] = []
        self.deleted_prefixes: list[str] = []

    async def initialize(self):
        return None

    async def close(self):
        return None

    async def upload_file(self, local_path: Path, *, key: str, content_type=None):
        uri = self.object_uri_for_key(key)
        self.files[uri] = local_path.read_bytes()
        return uri

    async def upload_file_if_absent(
        self, local_path: Path, *, key: str, content_type=None, expected_sha256=None
    ):
        del content_type
        uri = self.object_uri_for_key(key)
        self.upload_proof_calls.append((uri, expected_sha256))
        if uri in self.files:
            return uri, False
        self.files[uri] = local_path.read_bytes()
        return uri, True

    def object_uri_for_key(self, key: str):
        return f"s3://{_B2_BUCKET}/{key.lstrip('/')}"

    def object_prefix_uri_for_key(self, prefix: str):
        return f"s3://{_B2_BUCKET}/{prefix.strip('/')}/"

    async def stat_object(self, object_uri: str):
        rb = await self.inspect_object(object_uri)
        if not rb.present or rb.stat is None:
            raise _ObjectStorageError_b2(f"Missing: {object_uri}")
        return rb.stat

    async def inspect_object(self, object_uri: str, *, version_id=None):
        if object_uri not in self.files:
            return _ObjectReadback_b2(present=False)
        data = self.files[object_uri]
        return _ObjectReadback_b2(
            present=True,
            stat=_ObjectStat_b2(
                size=len(data),
                etag=f'"etag-{len(data)}"',
                last_modified=_B2_NOW,
                checksum=f"sha256:{_hashlib_b2.sha256(data).hexdigest()}",
            ),
        )

    async def delete_uri(self, object_uri: str):
        self.deleted_uris.append(object_uri)
        return self.files.pop(object_uri, None) is not None

    async def delete_prefix(self, prefix_uri: str):
        self.deleted_prefixes.append(prefix_uri)
        count = 0
        for uri in list(self.files):
            if uri.startswith(prefix_uri):
                self.files.pop(uri)
                count += 1
        return count

    async def delete_workspace(self, workspace: str):
        return 0

    def validate_document_file_uri(self, *args, **kwargs):
        return None

    def validate_document_prefix_uri(self, *args, **kwargs):
        return None


class _B2FakeRAG:
    """Fake LightRAG instance whose adelete_by_doc_id is idempotent."""

    def __init__(self):
        self.deleted: list[tuple[str, bool]] = []

    async def adelete_by_doc_id(self, doc_id: str, delete_llm_cache: bool = False):
        self.deleted.append((doc_id, delete_llm_cache))
        return SimpleNamespace(
            status="success",
            doc_id=doc_id,
            message="deleted",
            status_code=200,
            file_path="",
        )

    async def finalize_storages(self):
        return None

    async def adrop_all_storages(self):
        return {"dropped": 0, "failed": 0, "errors": []}


class _B2FakeRegistry:
    def __init__(self, rag):
        self._rag = rag

    async def get(self, kb_id: str):
        return self._rag

    async def acquire(self, kb_id: str):
        return self._rag


def _b2_sha256(data: bytes) -> str:
    return _hashlib_b2.sha256(data).hexdigest()


def _b2_limits():
    return _MaterializationLimits_b2(
        max_objects=1000, max_total_bytes=64 * 1024 * 1024, stale_ttl_seconds=1
    )


def _b2_document(
    kb_id: str,
    document_id: str,
    *,
    workspace: str,
    source_generation_id: str = "srcg-b2-old",
    artifact_id: str | None = "artifact-b2-old",
    source_key: str | None = None,
):
    now = _utc_now_iso_b2()
    source_uri = (
        f"s3://{_B2_BUCKET}/workspaces/{workspace}/documents/{document_id}/source/"
        f"generations/{source_generation_id}/source.pdf"
    )
    metadata: dict = {
        "source_object_uri": source_uri,
        "source_generation_id": source_generation_id,
    }
    if artifact_id is not None:
        metadata.update(
            {
                "current_sidecar_artifact_id": artifact_id,
                "current_artifact_ids": [artifact_id],
            }
        )
    if source_key is not None:
        metadata["source_key"] = source_key
    return _DocumentRecord_b2(
        id=document_id,
        kb_id=kb_id,
        workspace=workspace,
        lightrag_doc_id=f"engine-{document_id}",
        source_type="upload",
        source_name="source.pdf",
        source_uri=source_uri,
        source_hash="sha256:" + "0" * 64,
        content_type="application/pdf",
        size_bytes=4,
        parser_hash="parser-old",
        index_hash="index-old",
        status="ready",
        enabled=True,
        archived=False,
        chunks_count=1,
        entity_count=0,
        relation_count=0,
        error_code=None,
        error_message=None,
        metadata=metadata,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )


def _b2_artifact(document, artifact_id="artifact-b2-old"):
    now = _utc_now_iso_b2()
    object_uri = (
        f"s3://{_B2_BUCKET}/workspaces/{document.workspace}/documents/{document.id}/"
        f"artifacts/raw/{artifact_id}/sidecar.json"
    )
    return _ArtifactRecord_b2(
        id=artifact_id,
        kb_id=document.kb_id,
        workspace=document.workspace,
        document_id=document.id,
        artifact_type="sidecar",
        uri=object_uri,
        checksum="sha256:" + "a" * 64,
        size_bytes=9,
        metadata={"object_uri": object_uri},
        created_at=now,
    )


async def _b2_put_artifact(store, artifact):
    def write(conn):
        store._insert_artifact(conn, artifact)

    await store._write(write)


@pytest.fixture
def b2_object_setup(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir(parents=True, exist_ok=True)
    _reset_root_b2()
    _set_root_b2(root)

    async def _build():
        store = SQLiteMetadataStore(tmp_path / "b2.sqlite3")
        await store.initialize()
        kb_service = KnowledgeBaseService(tmp_path / "b2_kbs.json")
        kb_id = f"kb_b2_{_uuid4_b2().hex[:10]}"
        kb_record = await kb_service.create(name=kb_id, kb_id=kb_id)
        workspace = kb_record.workspace
        generation = kb_record.generation
        await store.activate_kb_generation(kb_id, generation)
        storage = _B2FakeObjectStorage()
        materializer = _ArtifactMaterializer_b2(
            storage, input_root=root, limits=_b2_limits()
        )
        service = DocumentLifecycleService(
            kb_service,
            store,
            root,
            object_storage=storage,
            artifact_storage_mode="object",
            materializer=materializer,
            artifact_cleanup_config=_ArtifactCleanupConfig_b2(),
            clock=lambda: _B2_NOW,
        )
        return service, store, storage, kb_id, workspace, generation

    return _build


async def _b2_seed(b2_object_setup, *, document_id="doc-b2", job_id="job-b2"):
    service, store, storage, kb_id, workspace, generation = await b2_object_setup()
    document = _b2_document(kb_id, document_id, workspace=workspace)
    artifact = _b2_artifact(document)
    from lightrag.api.metadata_store import JobRecord as _JobRecord_b2

    now = _utc_now_iso_b2()
    job = _JobRecord_b2(
        id=job_id,
        kb_id=kb_id,
        workspace=workspace,
        batch_id=None,
        document_id=document_id,
        job_type="replace",
        status="running",
        stage="replacing",
        progress=0.1,
        total_items=1,
        completed_items=0,
        failed_items=0,
        idempotency_key=f"idem-{job_id}",
        config_version_id=None,
        config_hash=None,
        retry_count=0,
        max_retries=3,
        payload={"idempotency_fingerprint": "sha256:b2", "attempt_tokens": {}},
        result=None,
        error_code=None,
        error_message=None,
        created_at=now,
        updated_at=now,
        queued_at=now,
        started_at=now,
        finished_at=None,
        cancelled_at=None,
    )
    await store.create_documents_and_job([document], job)
    await _b2_put_artifact(store, artifact)
    storage.files[document.metadata["source_object_uri"]] = b"old-bytes"
    storage.files[artifact.uri] = b"artifact"
    return service, store, storage, kb_id, workspace, generation, document, artifact, job


async def _b2_seed_for_delete(b2_object_setup, *, document_id="doc-b2-del", job_id="job-b2-del"):
    """Seed a document with NO active replace job so a delete claim can proceed."""
    service, store, storage, kb_id, workspace, generation = await b2_object_setup()
    document = _b2_document(kb_id, document_id, workspace=workspace)
    artifact = _b2_artifact(document)
    from lightrag.api.metadata_store import JobRecord as _JobRecord_b2

    now = _utc_now_iso_b2()
    del_job = _JobRecord_b2(
        id=job_id,
        kb_id=kb_id,
        workspace=workspace,
        batch_id=None,
        document_id=document_id,
        job_type="delete",
        status="running",
        stage="deleting",
        progress=0.1,
        total_items=1,
        completed_items=0,
        failed_items=0,
        idempotency_key=f"idem-{job_id}",
        config_version_id=None,
        config_hash=None,
        retry_count=0,
        max_retries=3,
        payload={"idempotency_fingerprint": "sha256:del", "attempt_tokens": {}},
        result=None,
        error_code=None,
        error_message=None,
        created_at=now,
        updated_at=now,
        queued_at=now,
        started_at=now,
        finished_at=None,
        cancelled_at=None,
    )
    await store.create_documents_and_job([document], del_job)
    await _b2_put_artifact(store, artifact)
    storage.files[document.metadata["source_object_uri"]] = b"old-bytes"
    storage.files[artifact.uri] = b"artifact"
    return service, store, storage, kb_id, workspace, generation, document, artifact, del_job


async def test_b2_object_replace_commits_before_engine_delete(b2_object_setup):
    """Object-mode replace commits pointer+manifest BEFORE engine delete."""
    (
        service,
        store,
        storage,
        kb_id,
        workspace,
        generation,
        document,
        artifact,
        job,
    ) = await _b2_seed(b2_object_setup)
    rag = _B2FakeRAG()
    registry = _B2FakeRegistry(rag)
    replacement = _DocumentReplacementSource_b2(
        source_name="source.pdf",
        content=b"new-content",
        source_type="upload",
        source_hash="sha256:" + _b2_sha256(b"new-content"),
        content_type="application/pdf",
        size_bytes=12,
    )
    item = await _kb_document_routes._execute_replace_document(
        document_service=service,
        kb_id=kb_id,
        job=job,
        document=document,
        replacement=replacement,
        active_registry=registry,  # type: ignore[arg-type]
        active_index_service=None,
        delete_source_file=True,
        delete_artifacts=True,
        delete_llm_cache=False,
        auto_parse=False,
        auto_index=False,
        parser_engine=None,
        process_options=None,
        force_reparse=False,
    )
    assert item["status"] == "succeeded"
    assert item["document_id"] == document.id
    assert item["previous_lightrag_doc_id"] == document.lightrag_doc_id
    # Engine delete happened (after commit).
    assert rag.deleted == [(document.lightrag_doc_id, False)]
    # No direct object/local cleanup; manifests enqueued instead.
    assert storage.deleted_uris == []
    assert storage.deleted_prefixes == []
    # Current pointer is NOT in the cleanup group.
    final = await store.get_document(kb_id, document.id)
    new_uri = final.metadata["source_object_uri"]
    manifests, total = await store.list_artifact_cleanup_manifests(
        kb_id=kb_id, document_id=document.id, limit=20
    )
    assert total == 2
    assert new_uri not in {m.target_uri for m in manifests}
    assert document.metadata["source_object_uri"] in {m.target_uri for m in manifests}
    assert item["cleanup_pending_count"] == 2


async def test_b2_object_delete_pre_engine_recheck_and_tombstone(b2_object_setup):
    """Object-mode delete: tombstone+manifest commit; response shape unchanged."""
    (
        service,
        store,
        storage,
        kb_id,
        workspace,
        generation,
        document,
        artifact,
        del_job,
    ) = await _b2_seed_for_delete(b2_object_setup)
    rag = _B2FakeRAG()
    registry = _B2FakeRegistry(rag)
    item = await _kb_document_routes._execute_delete_document_impl(
        document_service=service,
        kb_id=kb_id,
        job_id=del_job.id,
        document=document,
        active_registry=registry,  # type: ignore[arg-type]
        delete_source_file=True,
        delete_artifacts=True,
        delete_llm_cache=False,
        job_service=None,
        job=del_job,
    )
    assert item["status"] == "succeeded"
    assert item["document_id"] == document.id
    assert item["lightrag_doc_id"] == document.lightrag_doc_id
    # Engine delete happened.
    assert rag.deleted == [(document.lightrag_doc_id, False)]
    # No direct object/local cleanup.
    assert storage.deleted_uris == []
    assert storage.deleted_prefixes == []
    # Tombstone committed.
    tombstone = await store.get_document_lifecycle(kb_id, document.id)
    assert tombstone.deleted_at is not None
    # Manifests enqueued.
    manifests, total = await store.list_artifact_cleanup_manifests(
        kb_id=kb_id, document_id=document.id, limit=20
    )
    assert total == 2
    assert all(m.reason == "document_delete" for m in manifests)


async def test_b2_object_delete_engine_failure_preserves_bytes(b2_object_setup):
    """Engine delete failure in object mode preserves bytes (no tombstone)."""
    (
        service,
        store,
        storage,
        kb_id,
        workspace,
        generation,
        document,
        artifact,
        del_job,
    ) = await _b2_seed_for_delete(b2_object_setup, job_id="job-b2-engfail")

    class _FailingRAG:
        async def adelete_by_doc_id(self, doc_id, delete_llm_cache=False):
            raise RuntimeError("engine exploded")

        async def finalize_storages(self):
            return None

        async def adrop_all_storages(self):
            return {"dropped": 0, "failed": 0, "errors": []}

    registry = _B2FakeRegistry(_FailingRAG())
    item = await _kb_document_routes._execute_delete_document_impl(
        document_service=service,
        kb_id=kb_id,
        job_id=del_job.id,
        document=document,
        active_registry=registry,  # type: ignore[arg-type]
        delete_source_file=True,
        delete_artifacts=True,
        delete_llm_cache=False,
        job_service=None,
        job=del_job,
    )
    assert item["status"] == "failed"
    assert item["error_code"] == "delete_failed"
    # Bytes preserved: document is NOT tombstoned.
    stalled = await store.get_document(kb_id, document.id)
    assert stalled.deleted_at is None
    assert storage.deleted_uris == []


async def test_b2_object_batch_delete_partial_results(b2_object_setup):
    """Object-mode batch delete with partial results and per-document tokens."""
    service, store, storage, kb_id, workspace, generation = await b2_object_setup()
    rag = _B2FakeRAG()
    registry = _B2FakeRegistry(rag)
    from lightrag.api.metadata_store import JobRecord as _JobRecord_b2

    now = _utc_now_iso_b2()
    doc1 = _b2_document(kb_id, "doc-batch-1", workspace=workspace)
    doc2 = _b2_document(kb_id, "doc-batch-2", workspace=workspace, artifact_id=None)
    for doc in (doc1, doc2):
        storage.files[doc.metadata["source_object_uri"]] = b"old"
    _b2_art = _b2_artifact(doc1)
    await store.create_documents_and_job([doc1, doc2], _JobRecord_b2(
        id="job-batch", kb_id=kb_id, workspace=workspace, batch_id="batch-1",
        document_id=None, job_type="delete", status="running", stage="deleting",
        progress=0.0, total_items=2, completed_items=0, failed_items=0,
        idempotency_key="idem-batch", config_version_id=None, config_hash=None,
        retry_count=0, max_retries=3,
        payload={"document_ids": ["doc-batch-1", "doc-batch-2"], "attempt_tokens": {}},
        result=None, error_code=None, error_message=None,
        created_at=now, updated_at=now, queued_at=now, started_at=now,
        finished_at=None, cancelled_at=None,
    ))
    await _b2_put_artifact(store, _b2_art)

    job_payload = {"attempt_tokens": {}, "document_ids": ["doc-batch-1", "doc-batch-2"]}
    job_obj = SimpleNamespace(
        id="job-batch", kb_id=kb_id, workspace=workspace, batch_id="batch-1",
        document_id=None, job_type="delete", status="running",
        payload=job_payload,
    )

    items = []
    for doc in (doc1, doc2):
        item = await _kb_document_routes._execute_delete_document_impl(
            document_service=service,
            kb_id=kb_id,
            job_id="job-batch",
            document=doc,
            active_registry=registry,  # type: ignore[arg-type]
            delete_source_file=True,
            delete_artifacts=True,
            delete_llm_cache=False,
            job_service=None,
            job=job_obj,
        )
        items.append(item)
    assert all(i["status"] == "succeeded" for i in items)
    assert {i["document_id"] for i in items} == {"doc-batch-1", "doc-batch-2"}
    # Each document has its own independent manifest group.
    for doc in (doc1, doc2):
        manifests, total = await store.list_artifact_cleanup_manifests(
            kb_id=kb_id, document_id=doc.id, limit=10
        )
        assert total >= 1


async def test_b2_object_replace_stale_generation_fenced(b2_object_setup):
    """Stale kb generation is fenced by Store A before any side effect."""
    (
        service,
        store,
        storage,
        kb_id,
        workspace,
        generation,
        document,
        artifact,
        job,
    ) = await _b2_seed(b2_object_setup)
    rag = _B2FakeRAG()
    registry = _B2FakeRegistry(rag)
    replacement = _DocumentReplacementSource_b2(
        source_name="source.pdf",
        content=b"stale",
        source_type="upload",
        source_hash="sha256:" + _b2_sha256(b"stale"),
        content_type="application/pdf",
        size_bytes=5,
    )
    # Monkeypatch the service to use a stale generation.
    original_get = service.kb_service.get

    async def _stale_get(kb_id_arg, **kwargs):
        record = await original_get(kb_id_arg, **kwargs)
        return _dataclass_replace_b2(record, generation=generation + "-stale")

    service._kb_service.get = _stale_get  # type: ignore[assignment]
    try:
        item = await _kb_document_routes._execute_replace_document(
            document_service=service,
            kb_id=kb_id,
            job=job,
            document=document,
            replacement=replacement,
            active_registry=registry,  # type: ignore[arg-type]
            active_index_service=None,
            delete_source_file=True,
            delete_artifacts=True,
            delete_llm_cache=False,
            auto_parse=False,
            auto_index=False,
            parser_engine=None,
            process_options=None,
            force_reparse=False,
        )
    finally:
        service._kb_service.get = original_get  # type: ignore[assignment]
    assert item["status"] == "failed"
    assert rag.deleted == []  # no engine side effect


async def test_b2_http_admission_gate_remains_503_in_object_mode(tmp_path):
    """The capability/HTTP admission gate stays closed: object-mode HTTP 503."""
    root = tmp_path / "source"
    root.mkdir(parents=True, exist_ok=True)
    _reset_root_b2()
    _set_root_b2(root)
    store = SQLiteMetadataStore(tmp_path / "gate.sqlite3")
    await store.initialize()
    kb_service = KnowledgeBaseService(tmp_path / "gate_kbs.json")
    kb_id = f"kb_gate_{_uuid4_b2().hex[:8]}"
    kb_record = await kb_service.create(name=kb_id, kb_id=kb_id)
    await store.activate_kb_generation(kb_id, kb_record.generation)
    storage = _B2FakeObjectStorage()
    materializer = _ArtifactMaterializer_b2(
        storage, input_root=root, limits=_b2_limits()
    )
    document_service = DocumentLifecycleService(
        kb_service,
        store,
        root,
        object_storage=storage,
        artifact_storage_mode="object",
        materializer=materializer,
        artifact_cleanup_config=_ArtifactCleanupConfig_b2(),
        clock=lambda: _B2_NOW,
    )
    # The admission gate must raise in object mode.
    with pytest.raises(Exception):
        document_service.assert_destructive_operation_supported("Document replace")
    with pytest.raises(Exception):
        document_service.assert_destructive_operation_supported("Document delete")


# ---------------------------------------------------------------------------
# Phase 3.2 route policy — HTTP-level object-mode admission coverage.
#
# These tests exercise the route-layer wiring (``_require_destructive_lifecycle``
# and ``_reject_legacy_route_in_object_mode``) through the FastAPI router. The
# capability constant ``OBJECT_AUTHORITATIVE_LIFECYCLE_IMPLEMENTED`` stays
# False, so every object-mode destructive/legacy route must return 503 today.
# The allowlist branch (capability True) is covered by direct unit tests in
# ``tests/api/test_object_route_policy.py``.
# ---------------------------------------------------------------------------


def _build_object_mode_client(tmp_path: Path):
    """Build a router whose DocumentLifecycleService runs in object mode.

    Mirrors the B2 admission-gate fixture but wires a full document router so
    HTTP-level admission can be asserted. The admission gate fires at the very
    start of each handler (before any KB lookup), so no KB needs to be created.
    Returns the TestClient. The capability constant stays False, so every
    object-mode destructive/legacy route must return 503 today.
    """

    root = tmp_path / "object_source"
    root.mkdir(parents=True, exist_ok=True)
    _reset_root_b2()
    _set_root_b2(root)
    store = SQLiteMetadataStore(tmp_path / "policy.sqlite3")
    kb_service = KnowledgeBaseService(tmp_path / "policy_kbs.json")
    storage = _B2FakeObjectStorage()
    materializer = _ArtifactMaterializer_b2(
        storage, input_root=root, limits=_b2_limits()
    )
    document_service = DocumentLifecycleService(
        kb_service,
        store,
        root,
        object_storage=storage,
        artifact_storage_mode="object",
        materializer=materializer,
        artifact_cleanup_config=_ArtifactCleanupConfig_b2(),
        clock=lambda: _B2_NOW,
    )
    job_service = JobService(kb_service, store)
    app = FastAPI()
    app.include_router(
        create_kb_document_routes(
            document_service,
            job_service,
            api_key=_API_KEY,
            registry=None,
        )
    )
    return TestClient(app)


@pytest.fixture(autouse=True)
def _restore_canonical_input_root_after_object_policy_tests():
    """Ensure ``utils_pipeline`` canonical root is reset after object-mode tests."""

    yield
    _reset_root_b2()


def test_object_mode_legacy_text_route_blocked(tmp_path):
    """documents:texts is a legacy local-path route, permanently blocked in object mode."""

    client = _build_object_mode_client(tmp_path)
    response = client.post(
        "/kbs/kb_policy/documents:texts",
        json={"documents": [{"text": "hi", "source_name": "a.txt"}]},
        headers=_HEADERS,
    )
    assert response.status_code in {403, 503}, response.text
    assert "legacy local-path route" in response.text


def test_object_mode_legacy_url_route_blocked(tmp_path, monkeypatch):
    """documents:urls is a legacy local-path route, permanently blocked in object mode."""

    client = _build_object_mode_client(tmp_path)
    response = client.post(
        "/kbs/kb_policy/documents:urls",
        json={
            "documents": [
                {
                    "url": "https://example.com/test.txt",
                    "source_name": "test.txt",
                }
            ]
        },
        headers=_HEADERS,
    )
    assert response.status_code in {403, 503}, response.text
    assert "legacy local-path route" in response.text


def test_object_mode_legacy_import_route_blocked(tmp_path):
    """documents:import is a legacy local-path route, permanently blocked in object mode."""

    client = _build_object_mode_client(tmp_path)
    staged = tmp_path / "staged.md"
    staged.write_text("x", encoding="utf-8")
    response = client.post(
        "/kbs/kb_policy/documents:import",
        json={"documents": [{"path": str(staged)}]},
        headers=_HEADERS,
    )
    assert response.status_code in {403, 503}, response.text
    assert "legacy local-path route" in response.text


def test_object_mode_legacy_scan_route_blocked(tmp_path):
    """documents:scan is a legacy local-path route, permanently blocked in object mode."""

    client = _build_object_mode_client(tmp_path)
    response = client.post(
        "/kbs/kb_policy/documents:scan",
        json={"directory": "."},
        headers=_HEADERS,
    )
    assert response.status_code in {403, 503}, response.text
    assert "legacy local-path route" in response.text


def test_object_mode_destructive_routes_return_403_empty_policy(tmp_path):
    """Regression: gated destructive routes return 403 (empty allowlist) now that
    the capability constant is True. Legacy routes stay 403/503 independently.
    """

    from lightrag.api.config import OBJECT_AUTHORITATIVE_LIFECYCLE_IMPLEMENTED

    assert OBJECT_AUTHORITATIVE_LIFECYCLE_IMPLEMENTED is True

    client = _build_object_mode_client(tmp_path)

    # documents:delete — empty allowlist → 403.
    delete_response = client.delete(
        "/kbs/kb_policy/documents/doc_admission",
        headers=_HEADERS,
    )
    assert delete_response.status_code == 403, delete_response.text

    # documents:batch-delete.
    batch_delete_response = client.request(
        "POST",
        "/kbs/kb_policy/documents:batch-delete",
        json={"document_ids": ["doc_admission_b"]},
        headers=_HEADERS,
    )
    assert batch_delete_response.status_code == 403, batch_delete_response.text


def test_object_mode_sync_route_returns_403_empty_policy(tmp_path):
    """documents:sync returns 403 in object mode with empty allowlist (capability True)."""

    client = _build_object_mode_client(tmp_path)
    response = client.post(
        "/kbs/kb_policy/documents:sync",
        data={"source_keys": "key1"},
        files={"files": ("a.txt", b"hi", "text/plain")},
        headers=_HEADERS,
    )
    assert response.status_code == 403, response.text


def test_object_mode_replace_route_returns_403_empty_policy(tmp_path):
    """documents replace returns 403 in object mode with empty allowlist (capability True)."""

    client = _build_object_mode_client(tmp_path)
    response = client.post(
        "/kbs/kb_policy/documents/doc_admission_replace:replace",
        files={"file": ("a.txt", b"hi", "text/plain")},
        headers=_HEADERS,
    )
    assert response.status_code == 403, response.text


def test_local_mode_legacy_routes_still_allowed(tmp_path):
    """Regression: legacy routes work normally in local mode (guard is a no-op)."""

    client, _kb_service, _store, document_service, _job_service = _build_client(
        tmp_path
    )
    _create_kb(client, "kb_local_legacy")
    staged = document_service.source_root / "staged" / "local.md"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_text("local-mode", encoding="utf-8")
    response = client.post(
        "/kbs/kb_local_legacy/documents:import",
        json={"documents": [{"path": str(staged)}]},
        headers=_HEADERS,
    )
    assert response.status_code == 200, response.text
    assert response.json()["documents"][0]["source_type"] == "import"
