from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import pytest
from fastapi import HTTPException

from lightrag.artifact_runtime import (
    PipelineArtifactBinding,
    PipelineAttemptRowKind,
    extract_pipeline_attempt_token,
)
from lightrag.api.artifact_materialization import (
    ArtifactMaterializer,
    MaterializationLimits,
)
from lightrag.api.document_lifecycle_service import (
    DocumentLifecycleError,
    DocumentLifecycleService,
    DocumentSourceInput,
    DocumentSourceChecksumError,
)
from lightrag.api.job_service import JobService
from lightrag.api.kb_deletion_service import (
    KBDeletionService,
    KBHardDeleteUnsupportedError,
)
from lightrag.api.kb_service import KnowledgeBaseConflictError, KnowledgeBaseService
from lightrag.api.metadata_store import (
    ActiveDocumentParseJobError,
    SQLiteMetadataStore,
)
from lightrag.api.object_storage import ObjectStat, ObjectStorage, ObjectStorageError
from lightrag.utils import compute_mdhash_id
from lightrag.utils_pipeline import (
    reset_canonical_input_root_for_tests,
    set_canonical_input_root,
    sidecar_uri_for,
)

pytestmark = pytest.mark.offline

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_kb_document_routes = importlib.import_module("lightrag.api.routers.kb_document_routes")
sys.argv = _original_argv
_execute_parse_plan = _kb_document_routes._execute_parse_plan
_parse_plan_payload = _kb_document_routes._parse_plan_payload


class _FakeObjectStorage(ObjectStorage):
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.prefixes: dict[str, dict[str, bytes]] = {}
        self.file_uploads: list[str] = []
        self.prefix_uploads: list[str] = []
        self.file_downloads: list[tuple[str, Path]] = []
        self.prefix_downloads: list[tuple[str, Path]] = []
        self.deleted_files: list[str] = []
        self.deleted_prefixes: list[str] = []

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def upload_file(
        self, local_path: Path, *, key: str, content_type: str | None = None
    ) -> str:
        del content_type
        uri = f"s3://phase2a/{key}"
        self.files[uri] = local_path.read_bytes()
        self.file_uploads.append(uri)
        return uri

    async def upload_directory(self, local_dir: Path, *, prefix: str) -> str:
        uri = f"s3://phase2a/{prefix.rstrip('/')}/"
        payload: dict[str, bytes] = {}
        for path in sorted(local_dir.rglob("*")):
            if path.is_file() and not path.is_symlink():
                payload[path.relative_to(local_dir).as_posix()] = path.read_bytes()
        self.prefixes[uri] = payload
        self.prefix_uploads.append(uri)
        return uri

    async def upload_file_if_absent(
        self, local_path: Path, *, key: str, content_type: str | None = None
    ) -> tuple[str, bool]:
        del content_type
        uri = self.object_uri_for_key(key)
        if uri in self.files:
            return uri, False
        self.files[uri] = local_path.read_bytes()
        self.file_uploads.append(uri)
        return uri, True

    def object_uri_for_key(self, key: str) -> str:
        return f"s3://phase2a/{key.lstrip('/')}"

    def object_prefix_uri_for_key(self, prefix: str) -> str:
        return f"s3://phase2a/{prefix.strip('/')}/"

    async def download_file(self, object_uri: str, local_path: Path) -> None:
        self.file_downloads.append((object_uri, local_path))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(self.files[object_uri])

    async def stat_object(self, object_uri: str) -> ObjectStat:
        try:
            return ObjectStat(size=len(self.files[object_uri]))
        except KeyError as exc:
            raise ObjectStorageError(f"Missing fake object: {object_uri}") from exc

    async def download_prefix(
        self,
        prefix_uri: str,
        local_dir: Path,
        *,
        max_objects: int | None = None,
        max_total_bytes: int | None = None,
    ) -> int:
        payload = dict(self.prefixes.get(prefix_uri, {}))
        for uri, content in self.files.items():
            if uri.startswith(prefix_uri):
                relative_name = uri[len(prefix_uri) :]
                if relative_name:
                    payload[relative_name] = content
        if not payload and prefix_uri not in self.prefixes:
            raise ObjectStorageError(f"Missing fake prefix: {prefix_uri}")
        if max_objects is not None and len(payload) > max_objects:
            raise ObjectStorageError("fake prefix object limit exceeded")
        total_bytes = sum(len(content) for content in payload.values())
        if max_total_bytes is not None and total_bytes > max_total_bytes:
            raise ObjectStorageError("fake prefix byte limit exceeded")
        self.prefix_downloads.append((prefix_uri, local_dir))
        local_dir.mkdir(parents=True, exist_ok=True)
        for relative_name, content in payload.items():
            target = local_dir / relative_name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        return len(payload)

    async def delete_uri(self, object_uri: str) -> bool:
        self.deleted_files.append(object_uri)
        return self.files.pop(object_uri, None) is not None

    async def delete_prefix(self, prefix_uri: str) -> int:
        self.deleted_prefixes.append(prefix_uri)
        payload = self.prefixes.pop(prefix_uri, {})
        return len(payload)

    async def delete_workspace(self, workspace: str) -> int:
        marker = f"/workspaces/{workspace}/"
        deleted = 0
        for uri in list(self.files):
            if marker in uri:
                self.files.pop(uri)
                deleted += 1
        for uri, payload in list(self.prefixes.items()):
            if marker in uri:
                self.prefixes.pop(uri)
                deleted += len(payload)
        return deleted

    async def presign_download_url(
        self, object_uri: str, *, expires_in_seconds: int = 3600
    ) -> str:
        return f"https://objects.invalid/{expires_in_seconds}?uri={object_uri}"

    def validate_document_file_uri(
        self,
        object_uri: str,
        *,
        workspace: str,
        document_id: str,
        namespace: str | None = None,
        artifact_id: str | None = None,
    ) -> None:
        self._validate_document_uri(
            object_uri,
            workspace=workspace,
            document_id=document_id,
            namespace=namespace,
            artifact_id=artifact_id,
            expect_prefix=False,
        )

    def validate_document_prefix_uri(
        self,
        prefix_uri: str,
        *,
        workspace: str,
        document_id: str,
        namespace: str | None = None,
        artifact_id: str | None = None,
    ) -> None:
        self._validate_document_uri(
            prefix_uri,
            workspace=workspace,
            document_id=document_id,
            namespace=namespace,
            artifact_id=artifact_id,
            expect_prefix=True,
        )

    @staticmethod
    def _validate_document_uri(
        uri: str,
        *,
        workspace: str,
        document_id: str,
        namespace: str | None,
        artifact_id: str | None,
        expect_prefix: bool,
    ) -> None:
        document_prefix = (
            f"s3://phase2a/workspaces/{workspace}/documents/{document_id}/"
        )
        if not uri.startswith(document_prefix) or uri.endswith("/") != expect_prefix:
            raise ObjectStorageError("Object URI is outside the owned document scope")
        relative = uri[len(document_prefix) :].rstrip("/")
        parts = relative.split("/")
        if namespace == "source":
            if artifact_id is not None or len(parts) != 2 or parts[0] != "source":
                raise ObjectStorageError("Invalid source object scope")
        elif namespace == "artifacts":
            if len(parts) < 4 or parts[0] != "artifacts" or parts[2] != artifact_id:
                raise ObjectStorageError("Invalid artifact object scope")


class _MemoryKV:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def get_by_id(self, key: str) -> dict[str, Any] | None:
        async with self._lock:
            value = self.records.get(key)
            return dict(value) if value is not None else None

    async def upsert(self, values: dict[str, dict[str, Any]]) -> None:
        async with self._lock:
            self.records.update({key: dict(value) for key, value in values.items()})

    async def compare_and_commit_pipeline_attempt(
        self,
        key: str,
        payload: Mapping[str, Any],
        *,
        expected_attempt_token: str,
        row_kind: PipelineAttemptRowKind,
    ) -> bool:
        async with self._lock:
            current = self.records.get(key)
            if (
                extract_pipeline_attempt_token(current, row_kind=row_kind)
                != expected_attempt_token
            ):
                return False
            self.records[key] = dict(payload)
            return True

    async def index_done_callback(self) -> None:
        return None


class _ParserRAG:
    def __init__(self) -> None:
        self.full_docs = _MemoryKV()
        self.source_paths: list[Path] = []
        self.raw_present_at_start: list[bool] = []

    async def parse_mineru(
        self, doc_id: str, file_path: str, content_data: dict[str, Any]
    ) -> dict[str, Any]:
        source = Path(file_path)
        self.source_paths.append(source)
        raw_dir = source.parent / "__parsed__" / f"{source.name}.mineru_raw"
        self.raw_present_at_start.append(raw_dir.is_dir())
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "full.md").write_text("# parsed\n", encoding="utf-8")
        (raw_dir / "content_list.json").write_text("[]\n", encoding="utf-8")
        images = raw_dir / "images"
        images.mkdir(exist_ok=True)
        (images / "page-1.png").write_bytes(b"image")

        sidecar = source.parent / "__parsed__" / f"{source.name}.parsed"
        sidecar.mkdir(parents=True, exist_ok=True)
        blocks = sidecar / f"{source.stem}.blocks.jsonl"
        blocks.write_text('{"type":"content","text":"parsed"}\n', encoding="utf-8")
        raw_binding = content_data.get("artifact_binding")
        binding = (
            PipelineArtifactBinding.from_mapping(raw_binding)
            if isinstance(raw_binding, dict)
            else None
        )
        if binding is not None:
            assert binding.lightrag_doc_id == doc_id
            durable_fields = {
                "file_path": content_data["durable_file_path"],
                "process_options": content_data.get("process_options", ""),
                "artifact_binding": binding.to_dict(),
            }
        else:
            durable_fields = {
                "file_path": str(source),
                "sidecar_location": sidecar_uri_for(sidecar),
            }
        await self.full_docs.upsert(
            {
                doc_id: {
                    "content": "{{LRdoc}}parsed",
                    "parse_format": "lightrag",
                    "parse_engine": "mineru",
                    **durable_fields,
                }
            }
        )
        return {
            "doc_id": doc_id,
            "file_path": str(source),
            "parse_format": "lightrag",
            "parse_engine": "mineru",
            "content": "parsed",
            "blocks_path": str(blocks),
            "parse_stage_skipped": self.raw_present_at_start[-1],
        }


class _BlockingParserRAG:
    def __init__(self) -> None:
        self.full_docs = _MemoryKV()
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.source_path: Path | None = None

    async def parse_mineru(
        self, doc_id: str, file_path: str, content_data: dict[str, Any]
    ) -> dict[str, Any]:
        del doc_id, content_data

        def producer() -> dict[str, Any]:
            source = Path(file_path)
            self.source_path = source
            self.started.set()
            if not self.release.wait(timeout=10):
                raise TimeoutError("test producer was not released")
            assert source.is_file()
            sidecar = source.parent / "__parsed__" / f"{source.name}.parsed"
            sidecar.mkdir(parents=True, exist_ok=True)
            blocks = sidecar / f"{source.stem}.blocks.jsonl"
            blocks.write_text('{"text":"late"}\n', encoding="utf-8")
            self.finished.set()
            return {
                "parse_format": "lightrag",
                "content": "late",
                "blocks_path": str(blocks),
            }

        return await asyncio.to_thread(producer)


class _PathFailingParserRAG:
    def __init__(self) -> None:
        self.full_docs = _MemoryKV()

    async def parse_mineru(
        self, doc_id: str, file_path: str, content_data: dict[str, Any]
    ) -> dict[str, Any]:
        del doc_id, content_data
        raise RuntimeError(f"parser failed while reading {file_path}")


def _limits(*, stale_ttl_seconds: int = 1) -> MaterializationLimits:
    return MaterializationLimits(
        max_objects=1_000,
        max_total_bytes=64 * 1024 * 1024,
        stale_ttl_seconds=stale_ttl_seconds,
    )


def _build_object_service(
    *,
    root: Path,
    kb_service: KnowledgeBaseService,
    metadata_store: SQLiteMetadataStore,
    storage: _FakeObjectStorage,
) -> tuple[DocumentLifecycleService, ArtifactMaterializer]:
    root.mkdir(parents=True, exist_ok=True)
    reset_canonical_input_root_for_tests()
    set_canonical_input_root(root)
    materializer = ArtifactMaterializer(
        storage,
        input_root=root,
        limits=_limits(),
    )
    service = DocumentLifecycleService(
        kb_service,
        metadata_store,
        root,
        object_storage=storage,
        artifact_storage_mode="object",
        materializer=materializer,
    )
    return service, materializer


async def _create_document(
    service: DocumentLifecycleService,
    kb_id: str,
    *,
    source_name: str = "report.pdf",
    content: bytes = b"pdf-bytes",
):
    result = await service.create_source_batch(
        kb_id,
        [
            DocumentSourceInput(
                source_name=source_name,
                content=content,
                source_type="upload",
                content_type="application/pdf",
                metadata={},
            )
        ],
    )
    return result.documents[0], result.job


async def _execute_one_parse(
    service: DocumentLifecycleService,
    job_service: JobService,
    *,
    kb_id: str,
    document_id: str,
    rag: Any,
    force_reparse: bool = False,
):
    plan = await service.create_parse_plan(
        kb_id,
        document_id,
        parser_engine="mineru",
        force_reparse=force_reparse,
    )
    job, _created = await job_service.create_parse_job_once(
        kb_id,
        document_id=document_id,
        parser_hash=plan.parser_hash,
        lightrag_doc_id=plan.lightrag_doc_id,
        parser_engine=plan.parser_engine,
        process_options=plan.process_options,
        source_hash=plan.document.source_hash,
        source_name=plan.source_name,
        source_object_uri=plan.source_object_uri,
        raw_object_refs=_parse_plan_payload(plan)["raw_object_refs"],
        force_reparse=force_reparse,
    )
    await service.mark_parse_queued(kb_id, document_id, job=job, plan=plan)
    await job_service.transition_job(kb_id, job.id, status="running", progress=0.0)
    item = await _execute_parse_plan(
        document_service=service,
        kb_id=kb_id,
        job_id=job.id,
        plan=plan,
        rag=rag,
        job_service=job_service,
    )
    if item["status"] == "succeeded":
        await job_service.transition_job(
            kb_id,
            job.id,
            status="succeeded",
            progress=1.0,
            completed_items=1,
            result={"items": [item]},
        )
    elif item["status"] == "failed":
        await job_service.transition_job(
            kb_id,
            job.id,
            status="failed",
            progress=1.0,
            failed_items=1,
            result={"items": [item]},
            error_code=item["error_code"],
            error_message=item["error_message"],
        )
    return plan, job, item


def _assert_no_scratch(value: Any) -> None:
    assert ".lightrag-scratch" not in json.dumps(value, ensure_ascii=False, default=str)


async def test_moved_root_object_restore_upload_cleanup_and_stable_identity(tmp_path):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kbs.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    await kb_service.create(name="phase2a", kb_id="kb_phase2a")
    storage = _FakeObjectStorage()

    root_a = tmp_path / "checkout-a" / "inputs"
    service_a, _materializer_a = _build_object_service(
        root=root_a,
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
    )
    document, upload_job = await _create_document(service_a, "kb_phase2a")
    historical_source_uri = document.source_uri
    source_object_uri = document.metadata["source_object_uri"]
    assert historical_source_uri.startswith(str(root_a))
    assert not Path(historical_source_uri).exists()
    assert source_object_uri in storage.files
    _assert_no_scratch(asdict(document))
    _assert_no_scratch(asdict(upload_job))

    root_b = tmp_path / "checkout-b" / "inputs"
    service_b, materializer_b = _build_object_service(
        root=root_b,
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
    )
    job_service = JobService(kb_service, metadata_store)
    scratch_before = list(materializer_b.scratch_root.iterdir())
    plan = await service_b.create_parse_plan(
        "kb_phase2a", document.id, parser_engine="mineru"
    )
    assert list(materializer_b.scratch_root.iterdir()) == scratch_before
    assert plan.lightrag_doc_id == compute_mdhash_id(
        historical_source_uri, prefix="doc-"
    )
    assert plan.source_object_uri == source_object_uri

    rag = _ParserRAG()
    first_plan, parse_job, item = await _execute_one_parse(
        service_b,
        job_service,
        kb_id="kb_phase2a",
        document_id=document.id,
        rag=rag,
    )
    assert item["status"] == "succeeded"
    assert rag.source_paths[0].is_relative_to(materializer_b.scratch_root)
    assert not rag.source_paths[0].exists()
    assert not list(materializer_b.scratch_root.iterdir())

    parsed_document = await service_b.get_document("kb_phase2a", document.id)
    assert first_plan.claim_token
    assert first_plan.claim_token != parse_job.id
    assert (
        parsed_document.metadata["current_parse_generation_id"]
        == first_plan.claim_token
    )
    artifacts, _total = await service_b.list_document_artifacts(
        "kb_phase2a", document.id, limit=200
    )
    assert parsed_document.lightrag_doc_id == first_plan.lightrag_doc_id
    assert parsed_document.metadata["blocks_path"].startswith(str(root_b))
    assert all(artifact.uri.startswith(str(root_b)) for artifact in artifacts)
    assert all(
        artifact.metadata.get("blocks_path", str(root_b)).startswith(str(root_b))
        for artifact in artifacts
        if artifact.artifact_type == "sidecar"
    )
    pipeline_record = rag.full_docs.records[first_plan.lightrag_doc_id]
    assert pipeline_record["file_path"] == first_plan.source_name
    assert "sidecar_location" not in pipeline_record
    assert "blocks_path" not in pipeline_record
    committed_binding = PipelineArtifactBinding.from_mapping(
        pipeline_record["artifact_binding"],
        expected_workspace=first_plan.document.workspace,
    )
    assert committed_binding.state == "committed"
    assert committed_binding.parse_generation_id == first_plan.claim_token
    _assert_no_scratch(asdict(parsed_document))
    _assert_no_scratch([asdict(artifact) for artifact in artifacts])
    _assert_no_scratch(asdict(parse_job))
    _assert_no_scratch(rag.full_docs.records)

    second_plan = await service_b.create_parse_plan(
        "kb_phase2a", document.id, parser_engine="mineru", force_reparse=True
    )
    assert second_plan.lightrag_doc_id == first_plan.lightrag_doc_id
    assert second_plan.lightrag_doc_id not in {
        compute_mdhash_id(str(path), prefix="doc-") for path in rag.source_paths
    }


async def test_object_source_checksum_accepts_bare_and_prefixed_sha256(tmp_path):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kbs.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    await kb_service.create(name="source-ok", kb_id="kb_source_ok")
    storage = _FakeObjectStorage()
    service, materializer = _build_object_service(
        root=tmp_path / "inputs",
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
    )
    document, _job = await _create_document(service, "kb_source_ok")

    bare_plan = await service.create_parse_plan(
        "kb_source_ok", document.id, parser_engine="mineru"
    )
    bare_plan.claim_token = "checksum-test-bare"
    bare_execution = await service.materialize_parse_execution(bare_plan)
    assert bare_execution.source_path.read_bytes() == b"pdf-bytes"
    bare_rag = _ParserRAG()
    assert (await service.run_parse(bare_rag, bare_plan, bare_execution))[
        "parse_format"
    ] == "lightrag"
    bare_execution.cleanup()

    prefixed_plan = await service.create_parse_plan(
        "kb_source_ok", document.id, parser_engine="mineru"
    )
    prefixed_plan.document.source_hash = f"sha256:{document.source_hash}"
    prefixed_plan.claim_token = "checksum-test-prefixed"
    prefixed_execution = await service.materialize_parse_execution(prefixed_plan)
    assert prefixed_execution.source_path.read_bytes() == b"pdf-bytes"
    prefixed_rag = _ParserRAG()
    assert (await service.run_parse(prefixed_rag, prefixed_plan, prefixed_execution))[
        "parse_format"
    ] == "lightrag"
    prefixed_execution.cleanup()
    assert not list(materializer.scratch_root.iterdir())


async def test_corrupt_object_source_fails_before_parser_and_redacts_scratch(tmp_path):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kbs.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    await kb_service.create(name="source-corrupt", kb_id="kb_source_corrupt")
    storage = _FakeObjectStorage()
    service, materializer = _build_object_service(
        root=tmp_path / "inputs",
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
    )
    document, _job = await _create_document(service, "kb_source_corrupt")
    source_object_uri = document.metadata["source_object_uri"]
    storage.files[source_object_uri] = b"corrupted-object-bytes"
    direct_plan = await service.create_parse_plan(
        "kb_source_corrupt", document.id, parser_engine="mineru"
    )
    direct_plan.claim_token = "checksum-mismatch-direct"
    direct_execution = await service.materialize_parse_execution(direct_plan)
    direct_rag = _ParserRAG()
    with pytest.raises(DocumentSourceChecksumError) as checksum_error:
        await service.run_parse(direct_rag, direct_plan, direct_execution)
    direct_execution.cleanup()
    assert ".lightrag-scratch" not in str(checksum_error.value)
    assert str(materializer.scratch_root) not in str(checksum_error.value)
    assert direct_rag.source_paths == []
    rag = _ParserRAG()

    _plan, _parse_job, item = await _execute_one_parse(
        service,
        JobService(kb_service, metadata_store),
        kb_id="kb_source_corrupt",
        document_id=document.id,
        rag=rag,
    )
    assert item["status"] == "failed"
    assert "source checksum mismatch" in item["error_message"].lower()
    assert str(materializer.scratch_root) not in item["error_message"]
    assert ".lightrag-scratch" not in item["error_message"]
    assert rag.source_paths == []
    assert not list(materializer.scratch_root.iterdir())


async def test_corrupt_canonical_local_fallback_fails_before_parser(tmp_path):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kbs.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    await kb_service.create(name="fallback-corrupt", kb_id="kb_fallback_corrupt")
    storage = _FakeObjectStorage()
    service, materializer = _build_object_service(
        root=tmp_path / "inputs",
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
    )
    document, _job = await _create_document(service, "kb_fallback_corrupt")
    source_object_uri = document.metadata["source_object_uri"]
    await metadata_store.update_document(
        "kb_fallback_corrupt",
        document.id,
        metadata_patch={"source_object_uri": None},
    )
    fallback_path = Path(document.source_uri)
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    fallback_path.write_bytes(b"corrupted-local-fallback")
    rag = _ParserRAG()

    plan, _parse_job, item = await _execute_one_parse(
        service,
        JobService(kb_service, metadata_store),
        kb_id="kb_fallback_corrupt",
        document_id=document.id,
        rag=rag,
    )
    assert plan.source_object_uri is None
    assert item["status"] == "failed"
    assert "source checksum mismatch" in item["error_message"].lower()
    assert rag.source_paths == []
    assert source_object_uri in storage.files
    assert source_object_uri not in storage.deleted_files
    assert not list(materializer.scratch_root.iterdir())


async def test_empty_legacy_source_hash_skips_validation(tmp_path):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kbs.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    await kb_service.create(name="legacy-hash", kb_id="kb_legacy_hash")
    storage = _FakeObjectStorage()
    service, materializer = _build_object_service(
        root=tmp_path / "inputs",
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
    )
    document, _job = await _create_document(service, "kb_legacy_hash")
    plan = await service.create_parse_plan(
        "kb_legacy_hash", document.id, parser_engine="mineru"
    )
    plan.document.source_hash = ""
    plan.claim_token = "legacy-empty-checksum"
    execution = await service.materialize_parse_execution(plan)
    rag = _ParserRAG()
    parsed = await service.run_parse(rag, plan, execution)
    assert parsed["parse_format"] == "lightrag"
    assert len(rag.source_paths) == 1
    execution.cleanup()
    assert not list(materializer.scratch_root.iterdir())


async def test_non_force_raw_restore_and_force_skips_restore(tmp_path):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kbs.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    await kb_service.create(name="raw", kb_id="kb_raw")
    storage = _FakeObjectStorage()
    service, materializer = _build_object_service(
        root=tmp_path / "inputs",
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
    )
    document, _job = await _create_document(service, "kb_raw")
    job_service = JobService(kb_service, metadata_store)
    rag = _ParserRAG()

    _plan1, _job1, item1 = await _execute_one_parse(
        service,
        job_service,
        kb_id="kb_raw",
        document_id=document.id,
        rag=rag,
    )
    assert item1["status"] == "succeeded"
    assert rag.raw_present_at_start == [False]
    first_generation, _total = await service.list_document_artifacts(
        "kb_raw", document.id, limit=200
    )
    first_generation_ids = {artifact.id for artifact in first_generation}
    first_generation_files = set(storage.files)
    first_generation_prefixes = set(storage.prefixes)

    downloads_before = len(storage.prefix_downloads)
    plan2, _job2, item2 = await _execute_one_parse(
        service,
        job_service,
        kb_id="kb_raw",
        document_id=document.id,
        rag=rag,
    )
    assert item2["status"] == "succeeded"
    assert plan2.raw_object_refs
    assert all(
        raw_ref.checksum and raw_ref.checksum.startswith("sha256:")
        for raw_ref in plan2.raw_object_refs
    )
    assert rag.raw_present_at_start[-1] is True
    assert len(storage.prefix_downloads) > downloads_before
    after_reparse, _total = await service.list_document_artifacts(
        "kb_raw", document.id, limit=200
    )
    assert first_generation_ids < {artifact.id for artifact in after_reparse}
    assert first_generation_files <= set(storage.files)
    assert first_generation_prefixes <= set(storage.prefixes)

    downloads_before_force = len(storage.prefix_downloads)
    plan3, _job3, item3 = await _execute_one_parse(
        service,
        job_service,
        kb_id="kb_raw",
        document_id=document.id,
        rag=rag,
        force_reparse=True,
    )
    assert item3["status"] == "succeeded"
    assert plan3.raw_object_refs
    assert rag.raw_present_at_start[-1] is False
    assert len(storage.prefix_downloads) == downloads_before_force
    assert not list(materializer.scratch_root.iterdir())


async def test_corrupt_raw_cache_is_discarded_and_remote_generation_is_retained(
    tmp_path,
):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kbs.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    await kb_service.create(name="raw-checksum", kb_id="kb_raw_checksum")
    storage = _FakeObjectStorage()
    service, materializer = _build_object_service(
        root=tmp_path / "inputs",
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
    )
    document, _job = await _create_document(service, "kb_raw_checksum")
    job_service = JobService(kb_service, metadata_store)
    rag = _ParserRAG()
    _plan1, _job1, item1 = await _execute_one_parse(
        service,
        job_service,
        kb_id="kb_raw_checksum",
        document_id=document.id,
        rag=rag,
    )
    assert item1["status"] == "succeeded"

    corrupt_plan = await service.create_parse_plan(
        "kb_raw_checksum", document.id, parser_engine="mineru"
    )
    assert corrupt_plan.raw_object_refs
    corrupt_ref = corrupt_plan.raw_object_refs[0]
    assert corrupt_ref.checksum and corrupt_ref.checksum.startswith("sha256:")
    remote_payload = storage.prefixes[corrupt_ref.object_prefix_uri]
    corrupt_key = sorted(remote_payload)[0]
    remote_payload[corrupt_key] += b"-corrupt"
    corrupted_remote_snapshot = dict(remote_payload)
    deleted_before = list(storage.deleted_prefixes)

    plan2, _job2, item2 = await _execute_one_parse(
        service,
        job_service,
        kb_id="kb_raw_checksum",
        document_id=document.id,
        rag=rag,
    )
    assert item2["status"] == "succeeded"
    assert plan2.raw_object_refs[0].checksum == corrupt_ref.checksum
    assert rag.raw_present_at_start[-1] is False
    assert corrupt_ref.object_prefix_uri in storage.prefixes
    assert storage.prefixes[corrupt_ref.object_prefix_uri] == corrupted_remote_snapshot
    assert storage.deleted_prefixes == deleted_before
    assert not list(materializer.scratch_root.iterdir())


async def test_single_and_batch_parse_payloads_are_metadata_only(tmp_path):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kbs.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    await kb_service.create(name="payloads", kb_id="kb_payloads")
    storage = _FakeObjectStorage()
    service, materializer = _build_object_service(
        root=tmp_path / "inputs",
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
    )
    document1, _job1 = await _create_document(service, "kb_payloads")
    document2, _job2 = await _create_document(
        service, "kb_payloads", source_name="second.pdf", content=b"second"
    )
    job_service = JobService(kb_service, metadata_store)

    plan = await service.create_parse_plan(
        "kb_payloads", document1.id, parser_engine="mineru"
    )
    payload = _parse_plan_payload(plan)
    single_job, _created = await job_service.create_parse_job_once(
        "kb_payloads",
        document_id=document1.id,
        parser_hash=plan.parser_hash,
        lightrag_doc_id=plan.lightrag_doc_id,
        parser_engine=plan.parser_engine,
        process_options=plan.process_options,
        source_hash=plan.document.source_hash,
        source_name=plan.source_name,
        source_object_uri=plan.source_object_uri,
        raw_object_refs=payload["raw_object_refs"],
    )
    batch_plan = await service.create_batch_parse_plan(
        "kb_payloads", [document1.id, document2.id], parser_engine="mineru"
    )
    batch_job = await job_service.create_batch_parse_job(
        "kb_payloads",
        batch_id=batch_plan.batch_id,
        document_ids=[item.document.id for item in batch_plan.plans],
        total_items=2,
        plan_items=[_parse_plan_payload(item) for item in batch_plan.plans],
        planning_failures=batch_plan.failures,
    )

    for durable_payload in (payload, single_job.payload, batch_job.payload):
        serialized = json.dumps(durable_payload, default=str)
        assert "source_uri" not in serialized
        assert "sidecar" not in serialized
        assert "blocks_path" not in serialized
        assert ".lightrag-scratch" not in serialized
    assert not list(materializer.scratch_root.iterdir())


async def test_object_mode_allows_only_present_canonical_local_fallback(tmp_path):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kbs.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    await kb_service.create(name="fallback", kb_id="kb_fallback")
    input_root = tmp_path / "inputs"
    local_service = DocumentLifecycleService(kb_service, metadata_store, input_root)
    present_document, _present_upload = await _create_document(
        local_service, "kb_fallback", source_name="present.pdf"
    )
    missing_document, _missing_upload = await _create_document(
        local_service, "kb_fallback", source_name="missing.pdf"
    )
    Path(missing_document.source_uri).unlink()

    storage = _FakeObjectStorage()
    object_service, materializer = _build_object_service(
        root=input_root,
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
    )
    job_service = JobService(kb_service, metadata_store)
    rag = _ParserRAG()
    present_plan, _job, present_item = await _execute_one_parse(
        object_service,
        job_service,
        kb_id="kb_fallback",
        document_id=present_document.id,
        rag=rag,
    )
    assert present_plan.source_object_uri is None
    assert present_item["status"] == "succeeded"
    assert Path(present_document.source_uri).is_file()

    _missing_plan, _job, missing_item = await _execute_one_parse(
        object_service,
        job_service,
        kb_id="kb_fallback",
        document_id=missing_document.id,
        rag=rag,
    )
    assert missing_item["status"] == "failed"
    assert "migration required" in missing_item["error_message"].lower()
    assert not list(materializer.scratch_root.iterdir())


async def test_parse_commit_failure_compensates_only_new_generation(
    tmp_path, monkeypatch
):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kbs.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    await kb_service.create(name="compensation", kb_id="kb_compensation")
    storage = _FakeObjectStorage()
    service, materializer = _build_object_service(
        root=tmp_path / "inputs",
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
    )
    document, _upload_job = await _create_document(service, "kb_compensation")
    job_service = JobService(kb_service, metadata_store)
    rag = _ParserRAG()
    _plan1, _job1, item1 = await _execute_one_parse(
        service,
        job_service,
        kb_id="kb_compensation",
        document_id=document.id,
        rag=rag,
    )
    assert item1["status"] == "succeeded"
    old_files = set(storage.files)
    old_prefixes = set(storage.prefixes)
    old_artifacts, _total = await service.list_document_artifacts(
        "kb_compensation", document.id, limit=200
    )
    old_artifact_ids = {artifact.id for artifact in old_artifacts}
    file_upload_index = len(storage.file_uploads)
    prefix_upload_index = len(storage.prefix_uploads)

    async def fail_commit(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("injected complete_document_parse failure")

    monkeypatch.setattr(metadata_store, "complete_document_parse", fail_commit)
    _plan2, _job2, item2 = await _execute_one_parse(
        service,
        job_service,
        kb_id="kb_compensation",
        document_id=document.id,
        rag=rag,
        force_reparse=True,
    )
    assert item2["status"] == "failed"
    assert "injected complete_document_parse failure" in item2["error_message"]

    new_file_uris = storage.file_uploads[file_upload_index:]
    new_prefix_uris = storage.prefix_uploads[prefix_upload_index:]
    assert new_file_uris
    assert new_prefix_uris
    assert all(uri in storage.deleted_files for uri in new_file_uris)
    assert all(uri in storage.deleted_prefixes for uri in new_prefix_uris)
    assert all(uri not in storage.files for uri in new_file_uris)
    assert all(uri not in storage.prefixes for uri in new_prefix_uris)
    assert old_files <= set(storage.files)
    assert old_prefixes <= set(storage.prefixes)
    current_artifacts, _total = await service.list_document_artifacts(
        "kb_compensation", document.id, limit=200
    )
    assert {artifact.id for artifact in current_artifacts} == old_artifact_ids
    assert not list(materializer.scratch_root.iterdir())


async def test_parser_failure_redacts_runtime_paths_from_durable_errors(tmp_path):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kbs.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    await kb_service.create(name="errors", kb_id="kb_errors")
    storage = _FakeObjectStorage()
    service, materializer = _build_object_service(
        root=tmp_path / "inputs",
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
    )
    document, _upload_job = await _create_document(service, "kb_errors")
    job_service = JobService(kb_service, metadata_store)
    _plan, job, item = await _execute_one_parse(
        service,
        job_service,
        kb_id="kb_errors",
        document_id=document.id,
        rag=_PathFailingParserRAG(),
    )
    assert item["status"] == "failed"
    failed_job = await job_service.get_job("kb_errors", job.id)
    failed_document = await service.get_document("kb_errors", document.id)
    _assert_no_scratch(item)
    _assert_no_scratch(asdict(failed_job))
    _assert_no_scratch(asdict(failed_document))
    assert str(service.source_root) in item["error_message"]
    assert not list(materializer.scratch_root.iterdir())


async def test_logical_forced_cancel_waits_for_producer_then_cleans_lease(tmp_path):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kbs.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    await kb_service.create(name="cancel", kb_id="kb_cancel")
    storage = _FakeObjectStorage()
    service, materializer = _build_object_service(
        root=tmp_path / "inputs",
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
    )
    document, _upload_job = await _create_document(service, "kb_cancel")
    job_service = JobService(kb_service, metadata_store)
    plan = await service.create_parse_plan(
        "kb_cancel", document.id, parser_engine="mineru"
    )
    job, _created = await job_service.create_parse_job_once(
        "kb_cancel",
        document_id=document.id,
        parser_hash=plan.parser_hash,
        lightrag_doc_id=plan.lightrag_doc_id,
        parser_engine=plan.parser_engine,
        process_options=plan.process_options,
        source_hash=plan.document.source_hash,
        source_name=plan.source_name,
        source_object_uri=plan.source_object_uri,
    )
    await service.mark_parse_queued("kb_cancel", document.id, job=job, plan=plan)
    await job_service.transition_job("kb_cancel", job.id, status="running")
    rag = _BlockingParserRAG()
    task = asyncio.create_task(
        _execute_parse_plan(
            document_service=service,
            kb_id="kb_cancel",
            job_id=job.id,
            plan=plan,
            rag=rag,
            job_service=job_service,
        )
    )
    assert await asyncio.wait_for(asyncio.to_thread(rag.started.wait, 5), timeout=6)
    await job_service.transition_job("kb_cancel", job.id, status="cancelling")
    await asyncio.sleep(0.35)

    assert task.done() is False
    assert rag.source_path is not None
    lease_path = rag.source_path.parent.parent
    assert lease_path.is_dir()
    assert rag.source_path.is_file()
    assert len(materializer._deferred_leases) == 0
    in_flight_document = await metadata_store.get_document("kb_cancel", document.id)
    assert in_flight_document.status == "parsing"
    assert in_flight_document.metadata["current_parse_job_id"] == job.id
    assert in_flight_document.metadata["current_parse_claim_token"] == plan.claim_token
    retry_plan = await service.create_parse_plan(
        "kb_cancel", document.id, parser_engine="mineru"
    )
    with pytest.raises(ActiveDocumentParseJobError):
        await metadata_store.mark_document_parse_queued(
            "kb_cancel",
            document.id,
            metadata_patch={"pending_parse_job_id": "job-retry-too-early"},
            expected_snapshot=retry_plan.expected_snapshot,
        )

    os.utime(lease_path, (0, 0))
    assert materializer.cleanup_stale_leases(now=time.time() + 10_000) == 0
    other_worker = ArtifactMaterializer(
        storage,
        input_root=materializer.input_root,
        limits=_limits(),
    )
    assert other_worker.cleanup_stale_leases(now=time.time() + 10_000) == 0
    assert lease_path.is_dir()

    rag.release.set()
    item = await asyncio.wait_for(task, timeout=6)
    assert item["status"] == "cancelled"
    await job_service.transition_job(
        "kb_cancel", job.id, status="cancelled", progress=1.0
    )
    assert rag.finished.is_set()
    cancelled_document = await metadata_store.get_document("kb_cancel", document.id)
    assert cancelled_document.status == "parse_failed"
    assert cancelled_document.metadata.get("pending_parse_job_id") is None
    assert cancelled_document.metadata.get("pending_parse_claim_token") is None
    assert cancelled_document.metadata.get("current_parse_job_id") is None
    assert cancelled_document.metadata.get("current_parse_claim_token") is None
    assert len(materializer._deferred_leases) == 0
    assert not lease_path.exists()
    assert not list(materializer.scratch_root.iterdir())


async def test_outer_parse_task_cancellation_defers_locked_lease(tmp_path):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kbs.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    await kb_service.create(name="shutdown", kb_id="kb_shutdown")
    storage = _FakeObjectStorage()
    service, materializer = _build_object_service(
        root=tmp_path / "inputs",
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
    )
    document, _upload_job = await _create_document(service, "kb_shutdown")
    job_service = JobService(kb_service, metadata_store)
    plan = await service.create_parse_plan(
        "kb_shutdown", document.id, parser_engine="mineru"
    )
    job, _created = await job_service.create_parse_job_once(
        "kb_shutdown",
        document_id=document.id,
        parser_hash=plan.parser_hash,
        lightrag_doc_id=plan.lightrag_doc_id,
        parser_engine=plan.parser_engine,
        process_options=plan.process_options,
        source_hash=plan.document.source_hash,
        source_name=plan.source_name,
        source_object_uri=plan.source_object_uri,
    )
    await service.mark_parse_queued("kb_shutdown", document.id, job=job, plan=plan)
    await job_service.transition_job("kb_shutdown", job.id, status="running")
    rag = _BlockingParserRAG()
    task = asyncio.create_task(
        _execute_parse_plan(
            document_service=service,
            kb_id="kb_shutdown",
            job_id=job.id,
            plan=plan,
            rag=rag,
            job_service=job_service,
        )
    )
    assert await asyncio.wait_for(asyncio.to_thread(rag.started.wait, 5), timeout=6)
    assert rag.source_path is not None
    lease_path = rag.source_path.parent.parent

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(materializer._deferred_leases) == 1
    os.utime(lease_path, (0, 0))
    other_worker = ArtifactMaterializer(
        storage,
        input_root=materializer.input_root,
        limits=_limits(),
    )
    assert other_worker.cleanup_stale_leases(now=time.time() + 10_000) == 0
    assert lease_path.is_dir()

    rag.release.set()
    assert await asyncio.wait_for(asyncio.to_thread(rag.finished.wait, 5), timeout=6)
    lease = next(iter(materializer._deferred_leases))
    stale_path = lease.release_deferred_cleanup_for_janitor()
    os.utime(stale_path, (0, 0))
    assert other_worker.cleanup_stale_leases(now=time.time() + 10_000) == 1
    assert not stale_path.exists()


async def test_object_mode_destructive_gates_and_local_default_regression(tmp_path):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kbs.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    await kb_service.create(name="gates", kb_id="kb_gates")
    storage = _FakeObjectStorage()
    object_service, _materializer = _build_object_service(
        root=tmp_path / "object-inputs",
        kb_service=kb_service,
        metadata_store=metadata_store,
        storage=storage,
    )
    document, upload_job = await _create_document(object_service, "kb_gates")

    with pytest.raises(DocumentLifecycleError, match="Phase 3"):
        object_service.assert_destructive_operation_supported("Document sync")
    for operation in (
        "Document sync",
        "Document replace",
        "Document delete",
        "Batch document delete",
    ):
        with pytest.raises(HTTPException) as route_error:
            _kb_document_routes._require_destructive_lifecycle(
                object_service, operation
            )
        # With capability True + empty allowlist, the route policy returns 403.
        # (The service-layer gate below still raises independently.)
        assert route_error.value.status_code in {403, 503}
    with pytest.raises(DocumentLifecycleError, match="Phase 3"):
        await object_service.claim_delete("kb_gates", document.id, job=upload_job)
    with pytest.raises(DocumentLifecycleError, match="Phase 3"):
        await object_service.claim_batch_delete(
            "kb_gates", [document.id], job=upload_job
        )
    replacement = object_service.prepare_replacement_source(
        DocumentSourceInput(
            source_name="replacement.pdf",
            content=b"replacement",
            source_type="upload",
            content_type="application/pdf",
            metadata={},
        )
    )
    with pytest.raises(DocumentLifecycleError, match="Phase 3"):
        await object_service.claim_replace(
            "kb_gates",
            document.id,
            job=upload_job,
            replacement=replacement,
        )
    with pytest.raises(DocumentLifecycleError, match="Phase 3"):
        await object_service.complete_delete(
            "kb_gates", document.id, job_id=upload_job.id
        )
    with pytest.raises(DocumentLifecycleError, match="Phase 3"):
        await object_service.stage_sync_source_bytes(
            "kb_gates",
            batch_id="batch",
            item_index=0,
            source=DocumentSourceInput(
                source_name="sync.pdf",
                content=b"sync",
                source_type="upload",
                content_type="application/pdf",
                metadata={},
            ),
        )
    deletion_service = KBDeletionService(
        kb_service,
        metadata_store,
        object(),  # type: ignore[arg-type]
        input_root=tmp_path / "object-inputs",
        artifact_storage_mode="object",
    )
    # With capability True (flipped after Phase 3 Gates 1-3 PASS), the hard-delete
    # gate is OPEN in object mode. The service proceeds to the real deletion flow,
    # which fails here because the test passes a bogus expected_generation
    # (the KB was never soft-deleted in this gate-regression fixture).
    with pytest.raises(KnowledgeBaseConflictError):
        await deletion_service.hard_delete(
            "kb_gates", expected_generation="unused-by-object-mode-gate"
        )

    reset_canonical_input_root_for_tests()
    local_root = tmp_path / "local-inputs"
    local_service = DocumentLifecycleService(kb_service, metadata_store, local_root)
    local_document, _local_job = await _create_document(local_service, "kb_gates")
    assert Path(local_document.source_uri).is_file()
    local_service.assert_destructive_operation_supported("Document sync")
    _kb_document_routes._require_destructive_lifecycle(local_service, "Document sync")
    plan = await local_service.create_parse_plan(
        "kb_gates", local_document.id, parser_engine="mineru"
    )
    assert plan.source_name == local_document.source_name
    assert plan.lightrag_doc_id == compute_mdhash_id(
        local_document.source_uri, prefix="doc-"
    )


def test_object_mode_construction_fails_closed_without_explicit_materializer(tmp_path):
    kb_service = KnowledgeBaseService(tmp_path / "metadata" / "kbs.json")
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata" / "metadata.sqlite3")
    storage = _FakeObjectStorage()
    with pytest.raises(DocumentLifecycleError, match="remote object storage"):
        DocumentLifecycleService(
            kb_service,
            metadata_store,
            tmp_path / "inputs",
            artifact_storage_mode="object",
        )
    with pytest.raises(DocumentLifecycleError, match="ArtifactMaterializer"):
        DocumentLifecycleService(
            kb_service,
            metadata_store,
            tmp_path / "inputs",
            object_storage=storage,
            artifact_storage_mode="object",
        )
