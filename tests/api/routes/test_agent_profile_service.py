from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import HTTPException

from lightrag.api.agent_profile_service import (
    DOC_PROFILE_METADATA_KEY,
    GROUP_PROFILE_METADATA_KEY,
    KB_AUTO_PROFILE_DIRTY_METADATA_KEY,
    KB_AUTO_PROFILE_METADATA_KEY,
    PROFILE_SCHEMA_VERSION,
    AgentProfileService,
    AgentProfileUpdateRequest,
    KBAgentAutoProfile,
    _compact_kb_profile_for_metadata,
    _stride_sample,
    effective_agent_profile,
)
from lightrag.api.kb_service import KnowledgeBaseRecord, utc_now_iso
from lightrag.api.metadata_store import DocumentRecord
from lightrag.api.metadata_store import JobRecord


pytestmark = pytest.mark.offline


def _make_document(index: int, *, id_width: int = 0) -> DocumentRecord:
    now = utc_now_iso()
    doc_id = f"doc{index:0{id_width}d}" if id_width else f"doc{index}"
    return DocumentRecord(
        id=doc_id,
        kb_id="kb1",
        workspace="kb_kb1",
        lightrag_doc_id=f"lr_{doc_id}",
        source_type="upload",
        source_name=f"file{index}.md",
        source_uri=f"file{index}.md",
        source_hash=f"source-hash-{index}",
        content_type="text/markdown",
        size_bytes=123,
        parser_hash="parser-hash",
        index_hash="index-hash",
        status="ready",
        enabled=True,
        archived=False,
        chunks_count=3,
        entity_count=1,
        relation_count=1,
        error_code=None,
        error_message=None,
        metadata={},
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )


def _seed_doc_profile(document: DocumentRecord) -> None:
    """Attach a cache-valid document profile, as a previous refresh would."""
    document.metadata[DOC_PROFILE_METADATA_KEY] = {
        "type": "document_agent_profile",
        "schema_version": PROFILE_SCHEMA_VERSION,
        "status": "ready",
        "document_id": document.id,
        "source_name": document.source_name,
        "source_hash": document.source_hash,
        "index_hash": document.index_hash,
        "summary": f"{document.id} 摘要",
        "tags": ["法规"],
        "domains": ["合规"],
        "sample_questions": ["某原料是否允许使用？"],
        "negative_scope": [],
        "updated_at": utc_now_iso(),
    }


class _KBService:
    def __init__(self):
        now = utc_now_iso()
        self.record = KnowledgeBaseRecord(
            id="kb1",
            name="法规库",
            description="合规法规资料",
            workspace="kb_kb1",
            status="active",
            active_config_version_id=None,
            owner_id=None,
            tenant_id=None,
            visibility="private",
            created_at=now,
            updated_at=now,
            metadata={},
        )

    async def get(self, kb_id: str, *, include_deleted: bool = False):
        assert kb_id == self.record.id
        return self.record

    async def update(self, kb_id: str, **kwargs):
        assert kb_id == self.record.id
        metadata_patch = kwargs.get("metadata")
        if metadata_patch:
            for key, value in metadata_patch.items():
                if value is None:
                    self.record.metadata.pop(key, None)
                else:
                    self.record.metadata[key] = value
        return self.record


class _DocumentService:
    def __init__(self, documents: list[DocumentRecord] | None = None):
        self.documents = documents if documents is not None else [_make_document(1)]

    async def list_documents(self, kb_id: str, **kwargs):
        assert kb_id == "kb1"
        status = kwargs.get("status")
        docs = [d for d in self.documents if status is None or d.status == status]
        offset = kwargs.get("offset", 0)
        limit = kwargs.get("limit", 50)
        return docs[offset : offset + limit], len(docs)

    async def update_document(self, kb_id: str, document_id: str, *, metadata_patch=None, **_kwargs):
        assert kb_id == "kb1"
        document = next(d for d in self.documents if d.id == document_id)
        if metadata_patch:
            document.metadata.update(metadata_patch)
        return document


class _AnyFullDocs:
    async def get_by_id(self, doc_id: str):
        return {"content": f"{doc_id} 的法规内容示例。", "parse_format": "raw"}


class _FakeRAG:
    """Configurable PROFILE-role fake.

    ``doc_responses`` (optional) supplies raw strings returned for successive
    document-profile calls before falling back to the default JSON.
    """

    def __init__(self, doc_responses: list[str] | None = None):
        self.full_docs = _AnyFullDocs()
        self.doc_calls = 0
        self.kb_calls = 0
        self.group_calls = 0
        self.kb_payloads = []
        self.group_payloads = []
        self.doc_responses = list(doc_responses or [])
        self.on_kb_call = None  # optional async hook run before answering

    def _build_global_config(self):
        async def profile_func(prompt, **_kwargs):
            payload = json.loads(prompt)
            schema_type = payload["output_schema"]["type"]
            if schema_type == "document_agent_profile":
                self.doc_calls += 1
                if self.doc_responses:
                    return self.doc_responses.pop(0)
                doc_id = payload["document"]["document_id"]
                return json.dumps(
                    {
                        "type": "document_agent_profile",
                        "summary": f"{doc_id} 食品合规限制与用量判断。",
                        "tags": ["法规", "限量"],
                        "domains": ["食品合规"],
                        "sample_questions": ["A 原料能否使用？"],
                        "negative_scope": ["生产工艺"],
                    },
                    ensure_ascii=False,
                )
            if schema_type == "document_group_profile":
                self.group_calls += 1
                self.group_payloads.append(payload)
                return json.dumps(
                    {
                        "type": "document_group_profile",
                        "description": f"第 {payload['group_index']} 组文档的合规主题摘要。",
                        "tags": ["法规", "限量"],
                        "domains": ["食品合规"],
                        "sample_questions": ["某原料是否允许使用？"],
                        "negative_scope": ["生产工艺"],
                    },
                    ensure_ascii=False,
                )
            self.kb_calls += 1
            self.kb_payloads.append(payload)
            if self.on_kb_call is not None:
                await self.on_kb_call()
            return json.dumps(
                {
                    "type": "kb_agent_auto_profile",
                    "description": "适合回答食品法规、合规限制和原料限量问题。",
                    "tags": ["法规", "合规", "限量"],
                    "domains": ["食品合规"],
                    "sample_questions": ["某原料是否允许使用？"],
                    "negative_scope": ["配方工艺细节"],
                },
                ensure_ascii=False,
            )

        return {"role_llm_funcs": {"profile": profile_func}}


class _Registry:
    def __init__(self, rag):
        self.rag = rag

    async def get(self, kb_id: str):
        assert kb_id == "kb1"
        return self.rag


class _JobService:
    def __init__(self):
        self.statuses = []

    async def transition_job(self, kb_id: str, job_id: str, *, status: str, **_kwargs):
        self.statuses.append(status)
        return None

    async def list_jobs(self, kb_id: str, *, statuses=None, limit=50, offset=0, **_kwargs):
        return [], 0


class _RecordingJobService:
    """Fake JobService persisting jobs in memory with active-status queries."""

    def __init__(self):
        self.jobs: list[JobRecord] = []

    def _make_job(self, kb_id: str, job_type: str, payload: dict) -> JobRecord:
        now = utc_now_iso()
        return JobRecord(
            id=f"job{len(self.jobs) + 1}",
            kb_id=kb_id,
            workspace=f"kb_{kb_id}",
            batch_id=None,
            document_id=None,
            job_type=job_type,
            status="queued",
            stage="profiling",
            progress=0.0,
            total_items=1,
            completed_items=0,
            failed_items=0,
            idempotency_key=None,
            config_version_id=None,
            config_hash=None,
            retry_count=0,
            max_retries=3,
            payload=payload,
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

    async def create_job_once(
        self,
        kb_id: str,
        *,
        job_type: str,
        stage=None,
        total_items=1,
        payload=None,
        idempotency_key=None,
        document_id=None,
        batch_id=None,
    ):
        job = self._make_job(kb_id, job_type, payload or {})
        self.jobs.append(job)
        return job, True

    async def list_jobs(self, kb_id: str, *, statuses=None, limit=50, offset=0, **_kwargs):
        rows = [
            job
            for job in self.jobs
            if job.kb_id == kb_id and (statuses is None or job.status in statuses)
        ]
        return rows[offset : offset + limit], len(rows)

    async def transition_job(self, kb_id: str, job_id: str, *, status: str, **_kwargs):
        for job in self.jobs:
            if job.id == job_id:
                job.status = status
                job.updated_at = utc_now_iso()
                return job
        return None


def _service(
    kb_service=None,
    document_service=None,
    rag=None,
    job_service=None,
) -> tuple[AgentProfileService, _KBService, _DocumentService, _FakeRAG]:
    kb_service = kb_service or _KBService()
    document_service = document_service or _DocumentService()
    rag = rag or _FakeRAG()
    service = AgentProfileService(
        kb_service=kb_service,
        document_service=document_service,
        registry=_Registry(rag),
        job_service=job_service,
    )
    return service, kb_service, document_service, rag


@pytest.mark.asyncio
async def test_refresh_kb_profile_generates_document_and_kb_profiles():
    service, kb_service, document_service, _rag = _service()

    result = await service.refresh_kb_profile("kb1", force=True, job_id="job1")

    assert result.status == "ready"
    assert result.document_profiles_generated == 1
    assert result.pending_document_profiles == 0
    assert (
        document_service.documents[0].metadata["agent_doc_profile"]["summary"]
        == "doc1 食品合规限制与用量判断。"
    )
    auto = kb_service.record.metadata["agent_auto_profile"]
    assert auto["status"] == "ready"
    assert auto["description"] == "适合回答食品法规、合规限制和原料限量问题。"
    assert auto["tags"] == ["法规", "合规", "限量"]
    assert auto["job_id"] == "job1"
    assert auto["profiled_doc_count"] == 1
    assert auto["pending_document_profiles"] == 0


@pytest.mark.asyncio
async def test_manual_profile_overrides_auto_profile():
    kb_service = _KBService()
    kb_service.record.metadata["agent_auto_profile"] = {
        "status": "ready",
        "description": "自动说明",
        "tags": ["自动"],
    }
    service, _, _, _ = _service(kb_service=kb_service)

    await service.update_manual_profile(
        "kb1",
        AgentProfileUpdateRequest(
            agent_description="人工说明",
            agent_tags=["人工", "覆盖"],
            agent_priority=9,
        ),
    )

    effective = effective_agent_profile(kb_service.record)
    assert effective["agent_description"] == "人工说明"
    assert effective["agent_tags"] == ["人工", "覆盖"]
    assert effective["agent_priority"] == 9


@pytest.mark.asyncio
async def test_manual_scope_fields_override_auto_and_clear():
    kb_service = _KBService()
    kb_service.record.metadata["agent_auto_profile"] = {
        "status": "ready",
        "description": "自动说明",
        "domains": ["自动领域"],
        "sample_questions": ["自动示例问题？"],
        "negative_scope": ["自动排除项"],
    }
    service, _, _, _ = _service(kb_service=kb_service)

    await service.update_manual_profile(
        "kb1",
        AgentProfileUpdateRequest(
            agent_domains=["橡胶配方"],
            agent_negative_scope=["生产排产"],
        ),
    )

    effective = effective_agent_profile(kb_service.record)
    # Manual values win where set; untouched fields keep the auto values.
    assert effective["agent_profile_domains"] == ["橡胶配方"]
    assert effective["agent_profile_negative_scope"] == ["生产排产"]
    assert effective["agent_profile_sample_questions"] == ["自动示例问题？"]

    # Clearing with an empty list falls back to the auto value.
    await service.update_manual_profile(
        "kb1", AgentProfileUpdateRequest(agent_domains=[])
    )
    effective = effective_agent_profile(kb_service.record)
    assert effective["agent_profile_domains"] == ["自动领域"]
    assert effective["agent_profile_negative_scope"] == ["生产排产"]


@pytest.mark.asyncio
async def test_document_profile_failure_is_tolerated():
    kb_service = _KBService()
    documents = [_make_document(1), _make_document(2)]
    document_service = _DocumentService(documents)
    # doc1 fails all 3 JSON attempts; doc2 falls back to the default response.
    rag = _FakeRAG(doc_responses=["坏输出 1", "坏输出 2", "坏输出 3"])
    service, _, _, _ = _service(
        kb_service=kb_service, document_service=document_service, rag=rag
    )

    result = await service.refresh_kb_profile("kb1", force=True, job_id="job1")

    assert result.status == "ready"
    assert result.document_profiles_generated == 1
    assert result.document_profiles_failed == 1
    auto = kb_service.record.metadata[KB_AUTO_PROFILE_METADATA_KEY]
    assert auto["status"] == "ready"
    assert auto["document_profiles_failed"] == 1
    assert auto["failed_documents"][0]["document_id"] == "doc1"
    assert auto["profiled_doc_count"] == 1
    assert DOC_PROFILE_METADATA_KEY not in documents[0].metadata
    assert DOC_PROFILE_METADATA_KEY in documents[1].metadata


@pytest.mark.asyncio
async def test_all_document_profiles_failed_without_cache_fails_loudly():
    rag = _FakeRAG(doc_responses=["坏输出 1", "坏输出 2", "坏输出 3"])
    service, kb_service, _, _ = _service(rag=rag)

    with pytest.raises(HTTPException) as exc:
        await service.refresh_kb_profile("kb1", force=True)

    assert exc.value.status_code == 502
    assert exc.value.detail["error_code"] == "agent_profile_documents_failed"
    assert KB_AUTO_PROFILE_METADATA_KEY not in kb_service.record.metadata


@pytest.mark.asyncio
async def test_run_job_accepts_worker_claimed_running_job():
    job_service = _JobService()
    service = AgentProfileService(
        kb_service=_KBService(),
        document_service=_DocumentService(),
        registry=_Registry(_FakeRAG()),
        job_service=job_service,
    )
    now = utc_now_iso()
    job = JobRecord(
        id="job1",
        kb_id="kb1",
        workspace="kb_kb1",
        batch_id=None,
        document_id=None,
        job_type="agent_profile",
        status="running",
        stage="profiling",
        progress=0.1,
        total_items=1,
        completed_items=0,
        failed_items=0,
        idempotency_key=None,
        config_version_id=None,
        config_hash=None,
        retry_count=0,
        max_retries=3,
        payload={"force": True, "reason": "worker_test"},
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

    await service.run_job(job)

    assert job_service.statuses == ["succeeded"]


@pytest.mark.asyncio
async def test_mark_dirty_dedupes_onto_active_refresh_job():
    job_service = _RecordingJobService()
    service, kb_service, _, _ = _service(job_service=job_service)

    first = await service.mark_dirty("kb1", reason="document_build_completed", document_id="doc1")
    second = await service.mark_dirty("kb1", reason="document_build_completed", document_id="doc2")

    assert first is not None
    assert second is None  # coalesced onto the queued job
    assert len(job_service.jobs) == 1
    assert kb_service.record.metadata[KB_AUTO_PROFILE_DIRTY_METADATA_KEY]["document_id"] == "doc2"


@pytest.mark.asyncio
async def test_force_refresh_supersedes_active_nonforce_job():
    job_service = _RecordingJobService()
    service, _, _, _ = _service(job_service=job_service)

    await service.mark_dirty("kb1", reason="document_build_completed", document_id="doc1")
    job, created = await service.enqueue_refresh("kb1", force=True, reason="manual_refresh")

    assert created is True
    assert job.payload["force"] is True
    assert len(job_service.jobs) == 2

    # A second force enqueue coalesces onto the already-active force job.
    job2, created2 = await service.enqueue_refresh("kb1", force=True, reason="manual_refresh")
    assert created2 is False
    assert job2.id == job.id


@pytest.mark.asyncio
async def test_run_job_chains_refresh_when_dirty_lands_during_run():
    job_service = _RecordingJobService()
    kb_service = _KBService()
    rag = _FakeRAG()

    async def _dirty_mid_run():
        await asyncio.sleep(0.02)
        kb_service.record.metadata[KB_AUTO_PROFILE_DIRTY_METADATA_KEY] = {
            "dirty_at": utc_now_iso(),
            "reason": "document_build_completed",
            "document_id": "doc-late",
        }

    rag.on_kb_call = _dirty_mid_run
    service, _, _, _ = _service(kb_service=kb_service, rag=rag, job_service=job_service)

    job, created = await service.enqueue_refresh("kb1", reason="manual_refresh")
    assert created is True
    await service.run_job(job)

    assert [j.status for j in job_service.jobs][0] == "succeeded"
    assert len(job_service.jobs) == 2
    assert job_service.jobs[1].payload["reason"] == "chained_refresh"
    # The newer dirty flag survives until the chained refresh covers it.
    assert KB_AUTO_PROFILE_DIRTY_METADATA_KEY in kb_service.record.metadata


@pytest.mark.asyncio
async def test_run_job_clears_stale_dirty_flag_without_chaining():
    job_service = _RecordingJobService()
    kb_service = _KBService()
    kb_service.record.metadata[KB_AUTO_PROFILE_DIRTY_METADATA_KEY] = {
        "dirty_at": utc_now_iso(),
        "reason": "document_build_completed",
        "document_id": "doc1",
    }
    service, _, _, _ = _service(kb_service=kb_service, job_service=job_service)

    await asyncio.sleep(0.02)  # ensure refresh start is newer than the flag
    job, _ = await service.enqueue_refresh("kb1", reason="manual_refresh")
    await service.run_job(job)

    assert KB_AUTO_PROFILE_DIRTY_METADATA_KEY not in kb_service.record.metadata
    assert len(job_service.jobs) == 1


@pytest.mark.asyncio
async def test_refresh_caps_generation_per_run_and_chains_pending():
    job_service = _RecordingJobService()
    kb_service = _KBService()
    documents = [_make_document(i) for i in range(1, 31)]
    document_service = _DocumentService(documents)
    rag = _FakeRAG()
    service, _, _, _ = _service(
        kb_service=kb_service,
        document_service=document_service,
        rag=rag,
        job_service=job_service,
    )

    job1, _ = await service.enqueue_refresh("kb1", reason="manual_refresh")
    await service.run_job(job1)

    auto = kb_service.record.metadata[KB_AUTO_PROFILE_METADATA_KEY]
    assert rag.doc_calls == 24  # per-run generation cap
    assert auto["pending_document_profiles"] == 6
    assert auto["profiled_doc_count"] == 24
    assert len(job_service.jobs) == 2  # chained refresh queued for the rest
    assert job_service.jobs[1].payload["reason"] == "chained_refresh"

    await service.run_job(job_service.jobs[1])

    auto = kb_service.record.metadata[KB_AUTO_PROFILE_METADATA_KEY]
    assert rag.doc_calls == 30  # only the 6 pending documents were generated
    assert auto["pending_document_profiles"] == 0
    assert auto["profiled_doc_count"] == 30
    assert auto["source_doc_count"] == 30
    assert len(job_service.jobs) == 2  # nothing left to chain

    # Aggregation saw every cached profile plus whole-KB tag stats.
    last_kb_payload = rag.kb_payloads[-1]
    assert len(last_kb_payload["document_profiles"]) == 30
    assert last_kb_payload["profiled_document_count"] == 30
    assert last_kb_payload["tag_stats"][0]["count"] == 30


@pytest.mark.asyncio
async def test_oversized_llm_output_is_truncated_not_rejected():
    oversized = json.dumps(
        {
            "type": "document_agent_profile",
            "summary": "长" * 5000,
            "tags": [f"标签{i}" for i in range(50)],
            "domains": [f"领域{i}" for i in range(50)],
            "sample_questions": [f"问题{i}" for i in range(30)],
            "negative_scope": [f"排除{i}" for i in range(30)],
        },
        ensure_ascii=False,
    )
    rag = _FakeRAG(doc_responses=[oversized])
    service, _, document_service, _ = _service(rag=rag)

    result = await service.refresh_kb_profile("kb1", force=True)

    assert result.status == "ready"
    profile = document_service.documents[0].metadata[DOC_PROFILE_METADATA_KEY]
    assert len(profile["summary"]) <= 1000
    assert len(profile["tags"]) <= 20
    assert len(profile["sample_questions"]) <= 12


@pytest.mark.asyncio
async def test_profile_llm_invalid_json_is_retried():
    rag = _FakeRAG(doc_responses=["这不是 JSON"])
    service, kb_service, _, _ = _service(rag=rag)

    result = await service.refresh_kb_profile("kb1", force=True)

    assert result.status == "ready"
    assert rag.doc_calls == 2  # first attempt invalid, second succeeded
    assert kb_service.record.metadata[KB_AUTO_PROFILE_METADATA_KEY]["status"] == "ready"


def test_compact_kb_profile_respects_metadata_budget():
    profile = KBAgentAutoProfile(
        description="配" * 5000,
        tags=[f"标签{i}" + "补充说明" * 20 for i in range(60)],
        domains=[f"领域{i}" + "补充说明" * 20 for i in range(60)],
        sample_questions=[f"问题{i}" + "上下文" * 40 for i in range(30)],
        negative_scope=[f"排除{i}" + "上下文" * 40 for i in range(30)],
    )
    documents = [_make_document(i) for i in range(1, 9)]
    profiles_by_doc = {
        document.id: {"summary": "摘" * 240} for document in documents
    }

    result = _compact_kb_profile_for_metadata(
        profile,
        job_id="job1",
        total_ready_documents=8,
        profiled_documents=8,
        pending_documents=0,
        aggregated_documents=documents,
        profiles_by_doc=profiles_by_doc,
        generated=8,
        reused=0,
    )

    encoded = len(
        json.dumps({KB_AUTO_PROFILE_METADATA_KEY: result}, ensure_ascii=False).encode("utf-8")
    )
    assert encoded <= 12 * 1024
    assert result["description"]
    assert result["status"] == "ready"


def test_stride_sample_covers_whole_range():
    items = list(range(300))

    sampled = _stride_sample(items, 128)

    assert len(sampled) == 128
    assert sampled[0] == 0  # head included
    assert sampled == sorted(sampled)  # order preserved
    assert len(set(sampled)) == 128  # no duplicates
    assert sampled[-1] >= 290  # tail region reached
    assert _stride_sample(items[:50], 128) == items[:50]  # small list untouched


@pytest.mark.asyncio
async def test_large_kb_uses_grouped_aggregation_with_cached_digests():
    kb_service = _KBService()
    documents = [_make_document(i, id_width=4) for i in range(1, 261)]
    for document in documents:
        _seed_doc_profile(document)
    document_service = _DocumentService(documents)
    rag = _FakeRAG()
    service, _, _, _ = _service(
        kb_service=kb_service, document_service=document_service, rag=rag
    )

    result = await service.refresh_kb_profile("kb1")

    # 260 profiled docs -> buckets of 128: [1..128], [129..256], [257..260]
    assert result.aggregation_mode == "grouped"
    assert result.pending_document_profiles == 0
    assert rag.doc_calls == 0  # every document profile came from cache
    assert rag.group_calls == 3
    auto = kb_service.record.metadata[KB_AUTO_PROFILE_METADATA_KEY]
    assert auto["aggregation_mode"] == "grouped"
    assert auto["group_count"] == 3
    assert auto["sampled_doc_count"] == 260  # groups cover every profiled doc
    kb_payload = rag.kb_payloads[-1]
    assert kb_payload["document_profiles"] == []
    assert len(kb_payload["group_profiles"]) == 3
    assert kb_payload["group_profiles"][2]["doc_count"] == 4
    # Digests cached on each bucket's first document (created-order anchors).
    for anchor_id in ("doc0001", "doc0129", "doc0257"):
        anchor = next(d for d in documents if d.id == anchor_id)
        assert anchor.metadata[GROUP_PROFILE_METADATA_KEY]["status"] == "ready"

    # Second refresh: nothing changed, every group digest is reused.
    result2 = await service.refresh_kb_profile("kb1")
    assert rag.group_calls == 3
    assert result2.group_profiles_reused == 3
    assert result2.group_profiles_generated == 0

    # Touch one document's profile in the LAST bucket: only that bucket's
    # digest regenerates.
    tail = next(d for d in documents if d.id == "doc0260")
    tail.metadata[DOC_PROFILE_METADATA_KEY]["updated_at"] = "2099-01-01T00:00:00+00:00"
    result3 = await service.refresh_kb_profile("kb1")
    assert rag.group_calls == 4
    assert result3.group_profiles_generated == 1
    assert result3.group_profiles_reused == 2


@pytest.mark.asyncio
async def test_sampled_mode_spans_whole_kb_while_backfill_pending():
    kb_service = _KBService()
    documents = [_make_document(i, id_width=4) for i in range(1, 201)]
    # Leave the first 40 documents without cached profiles: one refresh
    # generates 24 of them, the rest stay pending -> transitional sampling.
    for document in documents[40:]:
        _seed_doc_profile(document)
    document_service = _DocumentService(documents)
    rag = _FakeRAG()
    service, _, _, _ = _service(
        kb_service=kb_service, document_service=document_service, rag=rag
    )

    result = await service.refresh_kb_profile("kb1")

    assert result.aggregation_mode == "sampled"
    assert result.document_profiles_generated == 24
    assert result.pending_document_profiles == 16
    assert rag.group_calls == 0  # no group digests while buckets still shift
    kb_payload = rag.kb_payloads[-1]
    assert len(kb_payload["document_profiles"]) == 128
    sampled_ids = {item["document_id"] for item in kb_payload["document_profiles"]}
    # The stride covers both ends of the KB instead of only the newest slice.
    assert any(doc_id <= "doc0100" for doc_id in sampled_ids)
    assert any(doc_id >= "doc0150" for doc_id in sampled_ids)
    assert kb_payload["aggregation_mode"] == "sampled"
