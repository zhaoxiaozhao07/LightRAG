from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from lightrag.api.document_lifecycle_service import DocumentLifecycleService
from lightrag.api.kb_service import utc_now_iso
from lightrag.api.metadata_store import (
    DocumentNotParsedError,
    DocumentRecord,
    MetadataRecordNotFoundError,
)
from lightrag.utils import generate_track_id, logger
from lightrag.utils_pipeline import rebase_under_input_dir, sidecar_uri_for

# doc_status status values that mean "still being processed by the pipeline".
# A document showing one of these has NOT reached a terminal build state yet —
# the read-back must wait it out rather than treating it as a failure (a
# concurrent same-KB pipeline drain may be the one that will finish it; see
# the concurrency notes on ``run_build_batch``).
_INFLIGHT_BUILD_STATUSES = frozenset(
    {"pending", "parsing", "analyzing", "processing", "preprocessed"}
)
# Substring the pipeline writes into a doc's ``error_msg`` when a build is
# cancelled mid-drain (see pipeline.py ``_mark_doc_cancelled_in_stage`` /
# merge cancellation). Used to reclassify such docs as cancelled rather than
# a hard build failure.
_CANCEL_ERROR_MARKER = "User cancelled"
# How long the post-drain read-back will wait for each enqueued doc to reach a
# terminal doc_status (processed/failed) before giving up. Generous because a
# concurrent drain that owns the pipeline ``busy`` flag may be processing a
# large merge for many minutes; the common single-flow case returns on the
# first poll (our own drain already finished) so this never sleeps.
DEFAULT_BUILD_DRAIN_TIMEOUT_SECONDS = float(
    os.getenv("KB_BUILD_DRAIN_TIMEOUT_SECONDS", "3600")
)
DEFAULT_BUILD_DRAIN_POLL_SECONDS = float(
    os.getenv("KB_BUILD_DRAIN_POLL_SECONDS", "1.0")
)

AgentProfileDirtyCallback = Callable[[str, str], Awaitable[None]]


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
        agent_profile_dirty_callback: AgentProfileDirtyCallback | None = None,
    ):
        self._document_service = document_service
        self._build_drain_timeout = DEFAULT_BUILD_DRAIN_TIMEOUT_SECONDS
        self._build_drain_poll = DEFAULT_BUILD_DRAIN_POLL_SECONDS
        self._agent_profile_dirty_callback = agent_profile_dirty_callback

    def set_agent_profile_dirty_callback(
        self, callback: AgentProfileDirtyCallback | None
    ) -> None:
        self._agent_profile_dirty_callback = callback

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
        # Clear any live LightRAG row before (re-)enqueue. LightRAG's enqueue
        # silently drops a document whose id is already present in ``doc_status``
        # (``filter_keys``) or whose basename / content-hash matches an existing
        # row. Because a KB document keeps the SAME ``lightrag_doc_id`` across
        # rebuilds, re-enqueuing an already-built doc without first removing the
        # old entry is a no-op — the pipeline does nothing, yet the read-back
        # stamps the OLD doc_status counts as a fresh success and the new
        # ``index_hash`` is recorded. That silently re-reports stale counts for a
        # NON-forced rebuild (e.g. after an index-config change or a reparse).
        # See ``_build_needs_engine_clear`` for exactly when a row is cleared.
        if _build_needs_engine_clear(plan):
            deletion_result = await rag.adelete_by_doc_id(plan.document.lightrag_doc_id)
            status = getattr(deletion_result, "status", None)
            if status not in {"success", "not_found"}:
                raise RuntimeError(
                    getattr(deletion_result, "message", None)
                    or f"Build could not clear existing LightRAG doc "
                    f"'{plan.document.lightrag_doc_id}' before re-enqueue "
                    f"(status={status})"
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

    async def run_build_batch(
        self,
        rag: Any,
        plans: list[IndexBuildPlan],
        *,
        job_id: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Bulk-build multiple documents through a single pipeline drain.

        Counterpart of :meth:`run_build` for the parallel-aggregate path. The
        three-layer LightRAG pipeline (``parse → analyze → process``) is
        designed for cross-document overlap, but ``run_build`` only ever feeds
        it one document and waits for ``apipeline_process_enqueue_documents``
        to drain before returning. That serializes all aggregate flows
        (``:sync`` / ``:upload?auto_parse=true&auto_index=true`` / batch parse
        / their resume paths). This method instead enqueues every non-skipped
        plan in one call so the worker layers naturally overlap docs, then
        drains the pipeline exactly once and reads back ``doc_status`` for
        each document.

        Pre-conditions (mirror :meth:`run_build`):
          * Caller has already marked each document ``building`` (or the plan
            is ``skipped``).
          * ``plan.sidecar_uri`` is present for every non-skipped plan.
          * Forced rebuilds (``plan.force``) have an existing
            ``lightrag_doc_id`` — they are deleted from the engine first so
            the re-enqueue is not de-duplicated.

        Returns a ``{kb_document_id: run_result}`` mapping. ``run_result``
        matches :meth:`run_build`'s return shape so callers can feed it
        straight into :meth:`complete_build`. Failure to read back a doc's
        status surfaces as a per-doc dict ``{"error_code": ..., "error_message": ...}``;
        the caller decides whether to fail that doc only or the whole batch.
        """
        results: dict[str, dict[str, Any]] = {}
        runnable: list[IndexBuildPlan] = []
        for plan in plans:
            if plan.skipped:
                results[plan.document.id] = {
                    "skipped": True,
                    "skip_reason": plan.skip_reason,
                    "chunks_count": plan.document.chunks_count,
                    "entity_count": plan.document.entity_count,
                    "relation_count": plan.document.relation_count,
                }
                continue
            if not plan.sidecar_uri:
                results[plan.document.id] = {
                    "error_code": "build_failed",
                    "error_message": (
                        f"Document '{plan.document.id}' has no sidecar artifact for build"
                    ),
                }
                continue
            runnable.append(plan)

        if not runnable:
            return results

        # Clear any live LightRAG row so re-enqueue is not de-duplicated — for
        # forced rebuilds AND non-forced rebuilds of already-indexed docs (see
        # ``_build_needs_engine_clear``). Skipping this for a non-forced rebuild
        # lets the engine dedup the re-enqueue into a no-op and re-report stale
        # counts as success.
        for plan in runnable:
            if _build_needs_engine_clear(plan):
                try:
                    deletion_result = await rag.adelete_by_doc_id(
                        plan.document.lightrag_doc_id
                    )
                    status = getattr(deletion_result, "status", None)
                    if status not in {"success", "not_found"}:
                        raise RuntimeError(
                            getattr(deletion_result, "message", None)
                            or (
                                "Build could not clear existing LightRAG doc "
                                f"'{plan.document.lightrag_doc_id}' before "
                                f"re-enqueue (status={status})"
                            )
                        )
                except Exception as exc:  # noqa: BLE001 — record + drop from batch
                    logger.error(
                        "Pre-build delete failed for doc '%s' (KB build batch): %s",
                        plan.document.id,
                        exc,
                    )
                    results[plan.document.id] = {
                        "error_code": "build_failed",
                        "error_message": str(exc),
                    }
        # Drop docs whose forced-pre-delete failed from the actual enqueue list.
        runnable = [p for p in runnable if p.document.id not in results]
        if not runnable:
            return results

        # Bulk-enqueue every doc in one call so the pipeline sees the whole
        # batch up front and can overlap parse / analyze / process across
        # docs. The track id ties the batch together for log correlation.
        batch_track_id = generate_track_id(
            f"build_batch_{job_id or runnable[0].document.id}"
        )
        ids = [p.document.lightrag_doc_id for p in runnable]
        file_paths = [_kb_unique_basename(p) for p in runnable]
        sidecar_uris = [p.sidecar_uri for p in runnable]
        # Per-doc parse_engine / process_options (these drive the per-doc
        # chunker selection F/R/V/P and the engine label). Aligned with the
        # ``ids`` list rather than broadcasting one doc's value to the batch.
        parse_engines = [
            p.document.metadata.get("parse_engine") or "" for p in runnable
        ]
        process_options = [p.process_options or "" for p in runnable]
        await rag.apipeline_enqueue_documents(
            input=[""] * len(runnable),
            ids=ids,
            file_paths=file_paths,
            track_id=batch_track_id,
            docs_format="lightrag",
            lightrag_document_paths=sidecar_uris,
            parse_engine=parse_engines,
            process_options=process_options,
        )
        await rag.apipeline_process_enqueue_documents()

        # Read back each doc's outcome. apipeline_process_enqueue_documents()
        # returns WITHOUT draining when another flow already holds the
        # per-KB pipeline ``busy`` flag (it just sets request_pending and
        # returns) — in that case our freshly-enqueued docs are still inflight
        # and the owning loop will finish them shortly. So we POLL each doc's
        # doc_status until it reaches a terminal state instead of reading once
        # (which would spuriously mark still-PENDING docs build_failed). The
        # common single-flow case — our own call drained everything before
        # returning — sees terminal status on the first poll and never sleeps.
        for plan in runnable:
            results[plan.document.id] = await self._resolve_build_result(rag, plan)
        return results

    async def _resolve_build_result(
        self, rag: Any, plan: IndexBuildPlan
    ) -> dict[str, Any]:
        """Poll doc_status for a single enqueued plan and classify its outcome.

        Returns one of:
          * success counts dict (``{"skipped": False, "chunks_count": ...}``)
          * ``{"cancelled": True, "error_message": ...}`` when the pipeline
            marked the doc failed with a user-cancellation marker
          * ``{"error_code": "build_failed", "error_message": ...}`` for a
            genuine failure, a missing row, or a drain timeout.
        """
        doc_status_storage = getattr(rag, "doc_status", None)
        if doc_status_storage is None:
            return {
                "skipped": False,
                "chunks_count": None,
                "entity_count": None,
                "relation_count": None,
            }
        row = await self._await_doc_terminal(rag, plan.document.lightrag_doc_id)
        if row is None:
            return {
                "error_code": "build_failed",
                "error_message": (
                    f"Document '{plan.document.id}' build did not create a "
                    f"doc_status row for LightRAG doc "
                    f"'{plan.document.lightrag_doc_id}'"
                ),
            }
        status = row.get("status")
        if status == "processed":
            return {
                "skipped": False,
                "chunks_count": row.get("chunks_count"),
                "entity_count": row.get("entity_count"),
                "relation_count": row.get("relation_count"),
            }
        error_msg = str(row.get("error_msg") or "")
        if _CANCEL_ERROR_MARKER in error_msg:
            return {"cancelled": True, "error_message": error_msg}
        if status in _INFLIGHT_BUILD_STATUSES:
            return {
                "error_code": "build_failed",
                "error_message": (
                    f"Document '{plan.document.id}' build timed out waiting for "
                    f"the pipeline drain (status={status})"
                ),
            }
        return {
            "error_code": "build_failed",
            "error_message": (
                f"Document '{plan.document.id}' build did not reach processed "
                f"(status={status}: {error_msg})"
            ),
        }

    async def _await_doc_terminal(
        self, rag: Any, lightrag_doc_id: str
    ) -> dict[str, Any] | None:
        """Poll a doc's doc_status row until it leaves the inflight set.

        Returns the final row (processed/failed), the last inflight row on
        timeout, or None when the row is absent. Never sleeps when the row is
        already terminal on the first read (the single-flow fast path).
        """
        return await _await_doc_status_terminal(
            rag,
            lightrag_doc_id,
            timeout=self._build_drain_timeout,
            poll_interval=self._build_drain_poll,
        )

    async def collect_doc_status(
        self, rag: Any, plan: IndexBuildPlan
    ) -> dict[str, Any]:
        """Public read-back of LightRAG ``doc_status`` for a single plan.

        Exposed for callers that drive their own pipeline (e.g., the batch
        aggregate flow) and need to fetch the final ``chunks/entities/
        relations`` counts after their own drain.
        """
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
        document = await self._document_service.metadata_store.complete_document_build(
            kb_id,
            document_id,
            index_hash=plan.index_hash,
            chunks_count=chunks_count,
            entity_count=entity_count,
            relation_count=relation_count,
            metadata_patch=metadata_patch,
        )
        await self._notify_agent_profile_dirty(kb_id, document_id)
        return document

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
        # Artifact URIs are absolute paths captured at parse time, so they only
        # resolve under the INPUT_DIR that was configured back then. A KB
        # document always lives at ``<INPUT_DIR>/<workspace>/<document_id>/``,
        # which makes the relocation exact for a deployment that moved its
        # INPUT_DIR (bare metal -> container, volume remount) — no re-parse.
        anchor = (document.workspace, document.id)

        def _local(uri: str | None) -> str | None:
            if not uri:
                return None
            rebased = rebase_under_input_dir(uri, anchor=anchor)
            return str(rebased) if rebased is not None else uri

        for artifact in artifacts:
            if artifact.artifact_type == "blocks" and not blocks_path:
                blocks_path = _local(artifact.uri)
            if artifact.artifact_type == "sidecar" and not sidecar_uri:
                sidecar_uri = _to_sidecar_uri(_local(artifact.uri) or artifact.uri)
        if sidecar_uri is None and blocks_path:
            sidecar_uri = _to_sidecar_uri(str(Path(blocks_path).parent))
        return sidecar_uri, blocks_path

    async def _notify_agent_profile_dirty(self, kb_id: str, document_id: str) -> None:
        if self._agent_profile_dirty_callback is None:
            return
        try:
            await self._agent_profile_dirty_callback(kb_id, document_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Agent profile dirty callback failed for KB '%s' doc '%s': %s",
                kb_id,
                document_id,
                exc,
            )


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


def _build_needs_engine_clear(plan: IndexBuildPlan) -> bool:
    """Whether the old LightRAG row must be deleted before a (re-)enqueue.

    LightRAG's enqueue de-duplicates a document whose id is already present
    (``full_docs`` / ``doc_status`` ``filter_keys``). A KB document keeps the
    SAME ``lightrag_doc_id`` across rebuilds, so re-enqueuing an already-built
    doc without deleting it first is a silent no-op that re-reports the stale
    doc_status counts as success. We clear the old row whenever the document
    already has a live engine row:

    * ``plan.force`` — an explicit ``:reindex`` / ``force_*`` on ``:build-kg``
      (unchanged behavior; a never-built id resolves to a cheap ``not_found``).
    * ``index_hash is not None`` — the document was successfully indexed before
      and has not since been reset. ``complete_document_build`` stamps
      ``index_hash``; a ``replace`` clears it (and deletes the engine row
      itself), so a post-replace first build sees ``None`` and skips the
      delete; a ``reparse`` preserves it, so a post-reparse rebuild correctly
      clears the stale row. This is what makes a NON-forced ``:build-kg`` of an
      already-built document actually rebuild instead of re-reporting stale
      counts.

    A genuine first build (``index_hash is None``, not forced) has no engine
    row to clear and is left untouched.
    """
    if not plan.document.lightrag_doc_id:
        return False
    return plan.force or plan.document.index_hash is not None


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


async def _await_doc_status_terminal(
    rag: Any,
    lightrag_doc_id: str,
    *,
    timeout: float,
    poll_interval: float,
) -> dict[str, Any] | None:
    """Poll ``rag.doc_status`` for ``lightrag_doc_id`` until it leaves the
    inflight set (pending/parsing/analyzing/processing/preprocessed).

    Returns the terminal row (processed/failed), the last inflight row on
    timeout, or None when the row is absent / storage is unavailable. Returns
    on the first read when already terminal (single-flow fast path: no sleep).

    The wait is what makes the read-back safe under concurrent same-KB drains:
    ``apipeline_process_enqueue_documents`` returns immediately (setting
    ``request_pending``) when another flow holds the pipeline ``busy`` flag, so
    our freshly-enqueued docs may still be inflight on return; the owning loop
    will drive them to a terminal state shortly.
    """
    doc_status_storage = getattr(rag, "doc_status", None)
    if doc_status_storage is None:
        return None
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0.0, timeout)
    consecutive_missing = 0
    while True:
        try:
            rows = await doc_status_storage.get_by_ids([lightrag_doc_id])
        except Exception as exc:  # noqa: BLE001 — treat probe failure as terminal
            logger.warning(
                "Failed to read doc_status while awaiting build of '%s': %s",
                lightrag_doc_id,
                exc,
            )
            return None
        row = rows[0] if rows else None
        if row is None:
            consecutive_missing += 1
            if consecutive_missing >= 2 or loop.time() >= deadline:
                return None
            await asyncio.sleep(poll_interval)
            continue
        consecutive_missing = 0
        status = row.get("status")
        if status not in _INFLIGHT_BUILD_STATUSES:
            # processed / failed → terminal enough to classify.
            return row
        if loop.time() >= deadline:
            return row
        await asyncio.sleep(poll_interval)


async def _collect_doc_status(
    rag: Any,
    plan: IndexBuildPlan,
    *,
    timeout: float = DEFAULT_BUILD_DRAIN_TIMEOUT_SECONDS,
    poll_interval: float = DEFAULT_BUILD_DRAIN_POLL_SECONDS,
) -> dict[str, Any]:
    doc_status_storage = getattr(rag, "doc_status", None)
    if doc_status_storage is None:
        return {
            "skipped": False,
            "chunks_count": None,
            "entity_count": None,
            "relation_count": None,
        }
    # Wait out a concurrent drain rather than reading once and mis-reporting a
    # still-inflight doc as a failure (see _await_doc_status_terminal).
    row = await _await_doc_status_terminal(
        rag, plan.document.lightrag_doc_id, timeout=timeout, poll_interval=poll_interval
    )
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
