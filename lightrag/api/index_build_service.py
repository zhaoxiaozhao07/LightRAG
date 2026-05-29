from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lightrag.api.document_lifecycle_service import DocumentLifecycleService
from lightrag.api.kb_service import utc_now_iso
from lightrag.api.metadata_store import (
    DocumentNotParsedError,
    DocumentRecord,
    MetadataRecordNotFoundError,
)
from lightrag.utils import generate_track_id, logger
from lightrag.utils_pipeline import sidecar_uri_for


@dataclass(slots=True)
class IndexBuildPlan:
    document: DocumentRecord
    sidecar_uri: str | None
    blocks_path: str | None
    parser_hash: str
    index_hash: str
    process_options: str
    force_rechunk: bool
    force_extract: bool
    force_embedding: bool
    skipped: bool = False
    skip_reason: str | None = None

    @property
    def force(self) -> bool:
        """Whether any force flag requires bypassing incremental reuse."""
        return self.force_rechunk or self.force_extract or self.force_embedding


@dataclass(slots=True)
class BatchIndexBuildPlan:
    batch_id: str
    plans: list[IndexBuildPlan]
    failures: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class IndexBuildResult:
    document: DocumentRecord
    skipped: bool
    chunks_count: int | None
    entity_count: int | None
    relation_count: int | None
    index_hash: str


class IndexBuildService:
    """Drive KG / index construction on top of parsed artifacts.

    Reuses LightRAG's pipeline (``apipeline_enqueue_documents`` +
    ``apipeline_process_enqueue_documents``) for chunking, entity/relation
    extraction, embedding, and KG merge. The service is responsible for
    incremental ingestion semantics: hash-based skip when source_hash +
    parser_hash + index_hash all match, else feed the document into the
    pipeline and stamp the new index_hash on success.
    """

    def __init__(
        self,
        document_service: DocumentLifecycleService,
    ):
        self._document_service = document_service

    async def create_build_plan(
        self,
        kb_id: str,
        document_id: str,
        *,
        rag: Any,
        force_rechunk: bool = False,
        force_extract: bool = False,
        force_embedding: bool = False,
    ) -> IndexBuildPlan:
        document = await self._document_service.get_document(kb_id, document_id)
        if document.status not in {
            "parsed",
            "ready",
            "build_failed",
            "build_queued",
            "building",
        }:
            raise DocumentNotParsedError(document_id, document.status)
        if not document.parser_hash:
            raise DocumentNotParsedError(document_id, document.status)

        sidecar_uri, blocks_path = await self._resolve_artifacts(kb_id, document)
        index_hash = compute_index_hash(rag)
        process_options_value = document.metadata.get("process_options")
        process_options = (
            str(process_options_value) if process_options_value is not None else ""
        )
        force = force_rechunk or force_extract or force_embedding
        skipped = False
        skip_reason: str | None = None
        if (
            not force
            and document.status == "ready"
            and document.index_hash == index_hash
        ):
            skipped = True
            skip_reason = "index_hash_match"
        return IndexBuildPlan(
            document=document,
            sidecar_uri=sidecar_uri,
            blocks_path=blocks_path,
            parser_hash=document.parser_hash,
            index_hash=index_hash,
            process_options=process_options,
            force_rechunk=force_rechunk,
            force_extract=force_extract,
            force_embedding=force_embedding,
            skipped=skipped,
            skip_reason=skip_reason,
        )

    async def create_batch_build_plan(
        self,
        kb_id: str,
        document_ids: list[str],
        *,
        rag: Any,
        force_rechunk: bool = False,
        force_extract: bool = False,
        force_embedding: bool = False,
    ) -> BatchIndexBuildPlan:
        plans: list[IndexBuildPlan] = []
        failures: list[dict[str, Any]] = []
        for document_id in document_ids:
            try:
                plan = await self.create_build_plan(
                    kb_id,
                    document_id,
                    rag=rag,
                    force_rechunk=force_rechunk,
                    force_extract=force_extract,
                    force_embedding=force_embedding,
                )
                plans.append(plan)
            except MetadataRecordNotFoundError as exc:
                failures.append(
                    _build_failure_item(
                        document_id,
                        error_code="document_not_found",
                        error_message=str(exc),
                    )
                )
            except DocumentNotParsedError as exc:
                failures.append(
                    _build_failure_item(
                        document_id,
                        error_code="document_not_parsed",
                        error_message=str(exc),
                        current_status=exc.current_status,
                    )
                )
            except FileNotFoundError as exc:
                failures.append(
                    _build_failure_item(
                        document_id,
                        error_code="parse_artifact_missing",
                        error_message=str(exc),
                    )
                )
        return BatchIndexBuildPlan(
            batch_id=generate_track_id("batch"),
            plans=plans,
            failures=failures,
        )

    async def claim_build_queued(
        self, kb_id: str, *, job_id: str, plan: IndexBuildPlan
    ) -> DocumentRecord:
        return await self._document_service.metadata_store.claim_document_build_queued(
            kb_id,
            plan.document.id,
            metadata_patch={
                "pending_build_job_id": job_id,
                "pending_index_hash": plan.index_hash,
                "force_rechunk": plan.force_rechunk,
                "force_extract": plan.force_extract,
                "force_embedding": plan.force_embedding,
            },
        )

    async def claim_batch_build_queued(
        self, kb_id: str, *, job_id: str, plans: list[IndexBuildPlan]
    ) -> tuple[list[DocumentRecord], list[dict[str, Any]]]:
        claims = [
            (
                plan.document.id,
                {
                    "pending_build_job_id": job_id,
                    "pending_index_hash": plan.index_hash,
                    "force_rechunk": plan.force_rechunk,
                    "force_extract": plan.force_extract,
                    "force_embedding": plan.force_embedding,
                },
            )
            for plan in plans
        ]
        return await self._document_service.metadata_store.claim_documents_build_queued(
            kb_id, claims
        )

    async def mark_building(
        self, kb_id: str, document_id: str, *, job_id: str
    ) -> DocumentRecord:
        return await self._document_service.metadata_store.mark_document_building(
            kb_id,
            document_id,
            metadata_patch={
                "current_build_job_id": job_id,
                "build_started_at": utc_now_iso(),
            },
        )

    async def run_build(
        self,
        rag: Any,
        plan: IndexBuildPlan,
    ) -> dict[str, Any]:
        """Push the parsed artifacts through LightRAG's index pipeline."""
        if plan.skipped:
            return {
                "skipped": True,
                "skip_reason": plan.skip_reason,
                "chunks_count": plan.document.chunks_count,
                "entity_count": plan.document.entity_count,
                "relation_count": plan.document.relation_count,
            }

        if not plan.sidecar_uri:
            raise FileNotFoundError(
                f"Document '{plan.document.id}' has no sidecar artifact for build"
            )

        track_id = generate_track_id(f"build_{plan.document.id}")
        # A forced rebuild (``:reindex``, or any explicit ``force_*`` on
        # ``:build-kg``) must actually re-run the LightRAG pipeline. LightRAG's
        # enqueue silently drops a document whose id is already present in
        # ``doc_status`` (``filter_keys``) or whose basename / content-hash
        # matches an existing row. Because a KB document keeps the SAME
        # ``lightrag_doc_id`` across rebuilds, re-enqueuing without first
        # removing the old entry would be a no-op — the force flags would only
        # bypass the KB-layer skip gate, not the engine-layer dedup. Delete the
        # old LightRAG document first so the re-enqueue is processed afresh.
        if plan.force and plan.document.lightrag_doc_id:
            deletion_result = await rag.adelete_by_doc_id(plan.document.lightrag_doc_id)
            status = getattr(deletion_result, "status", None)
            if status not in {"success", "not_found"}:
                raise RuntimeError(
                    getattr(deletion_result, "message", None)
                    or f"Forced reindex could not clear existing LightRAG doc "
                    f"'{plan.document.lightrag_doc_id}' (status={status})"
                )
        # LightRAG's enqueue performs filename-based dedup against doc_status
        # using the basename of ``file_path``. Two KB documents that share
        # the same ``source_name`` (e.g. both files sanitised to ``_.pdf``)
        # would otherwise collide and the second build would silently drop.
        # Prefix the basename with the KB document id so each KB doc gets a
        # globally unique key inside the LightRAG workspace.
        unique_basename = _kb_unique_basename(plan)
        await rag.apipeline_enqueue_documents(
            input=[""],
            ids=[plan.document.lightrag_doc_id],
            file_paths=[unique_basename],
            track_id=track_id,
            docs_format="lightrag",
            lightrag_document_paths=[plan.sidecar_uri],
            parse_engine=plan.document.metadata.get("parse_engine"),
            process_options=plan.process_options or None,
        )
        await rag.apipeline_process_enqueue_documents()
        return await _collect_doc_status(rag, plan)

    async def complete_build(
        self,
        kb_id: str,
        document_id: str,
        *,
        job_id: str,
        plan: IndexBuildPlan,
        run_result: dict[str, Any],
    ) -> DocumentRecord:
        if plan.skipped or run_result.get("skipped"):
            metadata_patch = {
                "last_build_job_id": job_id,
                "last_built_at": utc_now_iso(),
                "build_skipped": True,
                "build_skip_reason": plan.skip_reason or "index_hash_match",
                "pending_build_job_id": None,
                "current_build_job_id": None,
                "pending_index_hash": None,
            }
            return await self._document_service.metadata_store.complete_document_build(
                kb_id,
                document_id,
                index_hash=plan.index_hash,
                metadata_patch=metadata_patch,
            )
        chunks_count = run_result.get("chunks_count")
        entity_count = run_result.get("entity_count")
        relation_count = run_result.get("relation_count")
        metadata_patch = {
            "last_build_job_id": job_id,
            "last_built_at": utc_now_iso(),
            "build_skipped": False,
            "pending_build_job_id": None,
            "current_build_job_id": None,
            "pending_index_hash": None,
        }
        return await self._document_service.metadata_store.complete_document_build(
            kb_id,
            document_id,
            index_hash=plan.index_hash,
            chunks_count=chunks_count,
            entity_count=entity_count,
            relation_count=relation_count,
            metadata_patch=metadata_patch,
        )

    async def fail_build(
        self,
        kb_id: str,
        document_id: str,
        *,
        job_id: str,
        error_code: str,
        error_message: str,
    ) -> DocumentRecord:
        return await self._document_service.metadata_store.fail_document_build(
            kb_id,
            document_id,
            error_code=error_code,
            error_message=error_message,
            metadata_patch={
                "last_failed_build_job_id": job_id,
                "pending_build_job_id": None,
                "current_build_job_id": None,
            },
        )

    async def _resolve_artifacts(
        self, kb_id: str, document: DocumentRecord
    ) -> tuple[str | None, str | None]:
        artifacts, _total = await self._document_service.list_document_artifacts(
            kb_id, document.id, limit=200
        )
        sidecar_uri: str | None = None
        blocks_path: str | None = None
        for artifact in artifacts:
            if artifact.artifact_type == "blocks" and not blocks_path:
                blocks_path = artifact.uri
            if artifact.artifact_type == "sidecar" and not sidecar_uri:
                sidecar_uri = _to_sidecar_uri(artifact.uri)
        if sidecar_uri is None and blocks_path:
            sidecar_uri = _to_sidecar_uri(str(Path(blocks_path).parent))
        return sidecar_uri, blocks_path


def compute_index_hash(rag: Any) -> str:
    """Build a hash that captures chunk/embedding/extraction config.

    Anything that, when changed, would invalidate previously-built chunks,
    vectors, or KG content for a document. Query-time-only knobs (top_k etc.)
    are intentionally excluded.
    """
    active_index_hash = getattr(rag, "kb_active_index_hash", None)
    if active_index_hash:
        return str(active_index_hash)

    addon = getattr(rag, "addon_params", {}) or {}
    payload = {
        "schema": "kb-index-hash-v1",
        "embedding_func": getattr(
            getattr(rag, "embedding_func", None), "func_name", None
        )
        or getattr(getattr(rag, "embedding_func", None), "__name__", None),
        "embedding_dim": getattr(rag, "embedding_dim", None),
        "chunk_token_size": getattr(rag, "chunk_token_size", None),
        "chunk_overlap_token_size": getattr(rag, "chunk_overlap_token_size", None),
        "tiktoken_model_name": getattr(rag, "tiktoken_model_name", None),
        "summary_max_tokens": getattr(rag, "summary_max_tokens", None),
        "force_llm_summary_on_merge": getattr(
            rag, "force_llm_summary_on_merge", None
        ),
        "addon_chunker": addon.get("chunker"),
        "addon_entity_types": addon.get("entity_types"),
        "addon_language": addon.get("language"),
        "addon_extraction": addon.get("extraction"),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode(
        "utf-8"
    )
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _kb_unique_basename(plan: IndexBuildPlan) -> str:
    """Build a basename that is globally unique inside the KB workspace.

    LightRAG's filename-based dedup keys off the basename of the supplied
    ``file_path``. KB-layer source names can collide (e.g. two CJK PDFs
    that both sanitise to ``_.pdf``); prefixing with the KB document id
    keeps each entry distinct without losing the original suffix used for
    filetype detection downstream.
    """
    raw_name = (plan.document.source_name or "").strip() or "document"
    safe_name = raw_name.replace("/", "_").replace("\\", "_")
    return f"{plan.document.id}__{safe_name}"


async def _collect_doc_status(rag: Any, plan: IndexBuildPlan) -> dict[str, Any]:
    doc_status_storage = getattr(rag, "doc_status", None)
    if doc_status_storage is None:
        return {
            "skipped": False,
            "chunks_count": None,
            "entity_count": None,
            "relation_count": None,
        }
    try:
        rows = await doc_status_storage.get_by_ids([plan.document.lightrag_doc_id])
    except Exception as exc:  # noqa: BLE001 — fallback when storage probe fails
        logger.warning(
            "Failed to read doc_status for build result of '%s': %s",
            plan.document.id,
            exc,
        )
        rows = []
    row = rows[0] if rows else None
    if row is None:
        raise RuntimeError(
            f"Document '{plan.document.id}' build did not create doc_status row "
            f"for LightRAG doc '{plan.document.lightrag_doc_id}'"
        )
    if row.get("status") != "processed":
        raise RuntimeError(
            f"Document '{plan.document.id}' build did not reach processed (status={row.get('status')}: {row.get('error_msg')})"
        )
    return {
        "skipped": False,
        "chunks_count": row.get("chunks_count"),
        "entity_count": row.get("entity_count"),
        "relation_count": row.get("relation_count"),
    }


def _to_sidecar_uri(directory: str) -> str:
    return sidecar_uri_for(directory)


def _build_failure_item(
    document_id: str,
    *,
    error_code: str,
    error_message: str,
    **extra: Any,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "document_id": document_id,
        "status": "failed",
        "error_code": error_code,
        "error_message": error_message,
    }
    item.update({key: value for key, value in extra.items() if value is not None})
    return item
