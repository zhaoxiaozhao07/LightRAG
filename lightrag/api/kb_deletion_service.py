from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from lightrag.api.kb_service import KnowledgeBaseService, utc_now_iso
from lightrag.api.lightrag_registry import LightRAGInstanceRegistry
from lightrag.api.metadata_store import (
    JobRecord,
    SQLiteMetadataStore,
)
from lightrag.utils import generate_track_id, logger


@dataclass(slots=True)
class KBHardDeleteResult:
    job: JobRecord
    purged_rows: dict[str, int] = field(default_factory=dict)
    cleared_input_dir: bool = False
    finalized_storages: bool = False
    errors: list[str] = field(default_factory=list)


class KBDeletionService:
    """Drive the asynchronous hard delete of a knowledge base.

    The flow:

    1. Acquire ``destructive_lock(kb_id)`` on the registry so no other
       request can rebuild the instance underneath us.
    2. Force-evict the in-memory LightRAG instance (calls
       ``finalize_storages``).
    3. If the KB workspace has on-disk LightRAG storage in
       ``working_dir/<workspace>``, drop the directory.
    4. Remove ``input_dir/<workspace>`` (uploaded sources + parse
       artifacts).
    5. Purge SQLite control-plane state (documents / jobs / artifacts /
       config versions) for the KB.

    The clear_kb job records progress; failures are surfaced both in the
    job's ``error_message`` and in ``KBHardDeleteResult.errors`` so the
    caller can decide whether to retry.
    """

    def __init__(
        self,
        kb_service: KnowledgeBaseService,
        metadata_store: SQLiteMetadataStore,
        registry: LightRAGInstanceRegistry,
        *,
        input_root: Path,
        working_dir: Path | None = None,
    ):
        self._kb_service = kb_service
        self._metadata_store = metadata_store
        self._registry = registry
        self._input_root = Path(input_root)
        self._working_dir = Path(working_dir) if working_dir else None

    async def hard_delete(self, kb_id: str) -> KBHardDeleteResult:
        record = await self._kb_service.get(kb_id, include_deleted=True)
        now = utc_now_iso()
        job = JobRecord(
            id=generate_track_id("job_clear_kb"),
            kb_id=record.id,
            workspace=record.workspace,
            batch_id=None,
            document_id=None,
            job_type="clear_kb",
            status="running",
            stage="deleting",
            progress=0.0,
            total_items=1,
            completed_items=0,
            failed_items=0,
            idempotency_key=None,
            config_version_id=record.active_config_version_id,
            config_hash=None,
            retry_count=0,
            max_retries=0,
            payload={"kb_id": record.id, "workspace": record.workspace},
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
        created = await self._metadata_store.create_job(job)
        result = KBHardDeleteResult(job=created)
        try:
            async with self._registry.destructive_lock(record.id):
                # Drop in-memory instance (also closes storage handles)
                try:
                    await self._registry.force_evict(record.id)
                    result.finalized_storages = True
                except Exception as exc:  # noqa: BLE001
                    result.errors.append(f"force_evict: {exc}")

                if self._working_dir is not None:
                    workspace_dir = (self._working_dir / record.workspace).resolve()
                    self._safe_rmtree(workspace_dir, result, label="working_dir")

                input_workspace = (self._input_root / record.workspace).resolve()
                if input_workspace.exists():
                    self._safe_rmtree(input_workspace, result, label="input_dir")
                    result.cleared_input_dir = True

                result.purged_rows = await self._metadata_store.purge_kb_metadata(
                    record.id
                )
                # The purge wiped the just-created clear_kb job alongside
                # the rest of the KB's job history; re-insert it so we can
                # transition it to the final status below.
                created = await self._metadata_store.create_job(created)

            final_status = "succeeded" if not result.errors else "failed"
            result.job = await self._metadata_store.transition_job(
                record.id,
                created.id,
                status=final_status,
                progress=1.0,
                completed_items=1 if not result.errors else 0,
                failed_items=0 if not result.errors else 1,
                result={
                    "purged_rows": result.purged_rows,
                    "cleared_input_dir": result.cleared_input_dir,
                    "finalized_storages": result.finalized_storages,
                    "errors": result.errors,
                },
                error_code=None if not result.errors else "kb_hard_delete_failed",
                error_message=None if not result.errors else "; ".join(result.errors),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Hard delete failed for KB '%s': %s", record.id, exc)
            result.errors.append(str(exc))
            result.job = await self._metadata_store.transition_job(
                record.id,
                created.id,
                status="failed",
                progress=1.0,
                failed_items=1,
                error_code="kb_hard_delete_failed",
                error_message=str(exc),
            )
        return result

    @staticmethod
    def _safe_rmtree(
        path: Path, result: KBHardDeleteResult, *, label: str
    ) -> None:
        try:
            if path.exists():
                shutil.rmtree(path)
        except OSError as exc:
            result.errors.append(f"{label}: {exc}")
