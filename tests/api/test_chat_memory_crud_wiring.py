from __future__ import annotations

from typing import Any, cast

import pytest
from fastapi import HTTPException

from lightrag.api.enterprise_auth import ChatConversationService, UserService
from lightrag.api.metadata_store import (
    ChatMessageRecord,
    EnterpriseUserRecord,
    MetadataConflictError,
)

pytestmark = pytest.mark.offline

_EXTRACTION_FINGERPRINT = "chat-memory-extraction:v1:sha256:" + "1" * 64
_GRAPH_FINGERPRINT = "chat-memory-graph-store:v1:sha256:" + "2" * 64


class RecordingStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.order: list[str] = []
        self.append_conflict: MetadataConflictError | None = None
        self.delete_user_conflict: MetadataConflictError | None = None

    @staticmethod
    def _saved(records: list[ChatMessageRecord]) -> list[ChatMessageRecord]:
        for index, record in enumerate(records, start=1):
            record.seq = index
        return records

    async def append_chat_messages(self, records):
        saved = self._saved(list(records))
        self.calls.append(("append", saved))
        self.order.append("append_commit")
        return saved

    async def append_chat_messages_with_memory(self, records, **kwargs):
        if self.append_conflict is not None:
            raise self.append_conflict
        saved = self._saved(list(records))
        for record in saved:
            record.append_batch_id = "batch-1"
            record.project_event_seq = 1
            record.memory_reference_time = "2026-07-16T00:00:00+00:00"
        self.calls.append(("append_with_memory", kwargs))
        self.order.append("append_with_memory_commit")
        return saved

    async def delete_chat_message(self, *_args, **_kwargs):
        self.calls.append(("delete_message", None))
        return True

    async def delete_chat_message_with_memory(self, *_args, **kwargs):
        self.calls.append(("delete_message_with_memory", kwargs))
        self.order.append("delete_message_with_memory_commit")
        return True

    async def delete_chat_session(self, *_args, **_kwargs):
        self.calls.append(("delete_session", None))
        return True, 2

    async def delete_chat_session_with_memory(self, *_args, **kwargs):
        self.calls.append(("delete_session_with_memory", kwargs))
        self.order.append("delete_session_with_memory_commit")
        return True, 2

    async def delete_chat_project(self, *_args, **_kwargs):
        self.calls.append(("delete_project", None))
        return True, 1, 2

    async def delete_chat_project_with_memory(self, *_args, **kwargs):
        self.calls.append(("delete_project_with_memory", kwargs))
        self.order.append("delete_project_with_memory_commit")
        return True, 1, 2

    async def delete_enterprise_user(self, *_args, **_kwargs):
        self.calls.append(("delete_user", None))
        return True

    async def delete_enterprise_user_with_memory(self, _user_id, **kwargs):
        if self.delete_user_conflict is not None:
            raise self.delete_user_conflict
        self.calls.append(("delete_user_with_memory", kwargs))
        self.order.append("delete_user_with_memory_commit")
        return True


def _conversation_service(
    store: RecordingStore,
    *,
    admission_enabled: bool,
    nudge=None,
) -> ChatConversationService:
    return ChatConversationService(
        store,  # type: ignore[arg-type]
        memory_admission_enabled=admission_enabled,
        memory_extraction_fingerprint=_EXTRACTION_FINGERPRINT,
        memory_graph_store_fingerprint=_GRAPH_FINGERPRINT,
        post_commit_nudge=nudge,
    )


def _user() -> EnterpriseUserRecord:
    return EnterpriseUserRecord(
        id="usr-target",
        username="target",
        password_hash="hash",
        system_role="user",
        status="active",
        tenant_id="tenant-a",
        can_create_kb=False,
        can_use_bypass_query=False,
        token_version=7,
        metadata={},
        created_at="2026-07-15T00:00:00+00:00",
        updated_at="2026-07-16T00:00:00+00:00",
    )


async def test_enabled_append_uses_single_durable_store_call_and_post_commit_nudge():
    store = RecordingStore()

    def nudge() -> None:
        assert store.order == ["append_with_memory_commit"]
        store.order.append("nudge")

    service = _conversation_service(store, admission_enabled=True, nudge=nudge)
    saved = await service.append_messages(
        user_id="usr-a",
        project_id="proj-a",
        session_id="sess-a",
        messages=[{"role": "user", "content": "hello", "metadata": {}}],
        actor_user_id="usr-a",
    )

    assert saved is not None
    assert [call[0] for call in store.calls] == ["append_with_memory"]
    kwargs = store.calls[0][1]
    assert kwargs == {
        "config_fingerprint": _EXTRACTION_FINGERPRINT,
        "graph_store_fingerprint": _GRAPH_FINGERPRINT,
        "actor_user_id": "usr-a",
    }
    assert saved[0].append_batch_id == "batch-1"
    assert saved[0].project_event_seq == 1
    assert saved[0].memory_reference_time is not None
    assert store.order == ["append_with_memory_commit", "nudge"]


async def test_feature_off_append_is_ordinary_and_leaves_admission_null():
    store = RecordingStore()
    nudges: list[str] = []
    service = _conversation_service(
        store,
        admission_enabled=False,
        nudge=lambda: nudges.append("nudge"),
    )

    saved = await service.append_messages(
        user_id="usr-a",
        project_id="proj-a",
        session_id="sess-a",
        messages=[{"role": "user", "content": "feature off"}],
    )

    assert saved is not None
    assert [call[0] for call in store.calls] == ["append"]
    assert saved[0].append_batch_id is None
    assert saved[0].project_event_seq is None
    assert saved[0].memory_reference_time is None
    assert nudges == []


async def test_feature_off_deletes_still_use_memory_maintenance_transactions():
    store = RecordingStore()
    nudges: list[str] = []
    service = _conversation_service(
        store,
        admission_enabled=False,
        nudge=lambda: nudges.append("nudge"),
    )

    assert await service.delete_message(
        user_id="usr-a",
        project_id="proj-a",
        session_id="sess-a",
        message_id="msg-a",
    )
    assert await service.delete_session(
        user_id="usr-a", project_id="proj-a", session_id="sess-a"
    ) == (True, 2)
    assert await service.delete_project(
        user_id="usr-a", project_id="proj-a"
    ) == (True, 1, 2)

    assert [call[0] for call in store.calls] == [
        "delete_message_with_memory",
        "delete_session_with_memory",
        "delete_project_with_memory",
    ]
    for _name, kwargs in store.calls:
        assert kwargs["config_fingerprint"] == _EXTRACTION_FINGERPRINT
        assert kwargs["graph_store_fingerprint"] == _GRAPH_FINGERPRINT
    assert nudges == ["nudge", "nudge", "nudge"]


async def test_user_delete_enqueues_durable_purge_with_actor_cas_and_nudges():
    store = RecordingStore()
    nudges: list[str] = []
    service = UserService(
        store,  # type: ignore[arg-type]
        memory_admission_enabled=False,
        memory_extraction_fingerprint=_EXTRACTION_FINGERPRINT,
        memory_graph_store_fingerprint=_GRAPH_FINGERPRINT,
        post_commit_nudge=lambda: nudges.append("nudge"),
    )
    user = _user()
    membership_snapshot = {"tenant_id": "tenant-a", "role": "tenant_member"}

    assert await service.delete_user(
        user.id,
        actor_user_id="usr-admin",
        actor_tenant_id="tenant-admin",
        expected_user=user,
        expected_membership=membership_snapshot,
    )

    assert [call[0] for call in store.calls] == ["delete_user_with_memory"]
    kwargs = store.calls[0][1]
    assert kwargs["actor_user_id"] == "usr-admin"
    assert kwargs["actor_tenant_id"] == "tenant-admin"
    assert kwargs["expected_updated_at"] == user.updated_at
    assert kwargs["expected_token_version"] == user.token_version
    assert kwargs["expected_tenant_id"] == user.tenant_id
    assert kwargs["expected_membership"] == membership_snapshot
    assert kwargs["config_fingerprint"] == _EXTRACTION_FINGERPRINT
    assert kwargs["graph_store_fingerprint"] == _GRAPH_FINGERPRINT
    assert nudges == ["nudge"]


async def test_failed_commit_does_not_nudge_and_graph_migration_is_http_409():
    store = RecordingStore()
    store.append_conflict = MetadataConflictError(
        "chat_memory_graph_store",
        "logical-group",
        expected={"graph_store_fingerprints": (_GRAPH_FINGERPRINT,)},
        current={
            "error_code": "graph_store_migration_required",
            "graph_store_fingerprints": ("old-store",),
        },
    )
    nudges: list[str] = []
    service = _conversation_service(
        store,
        admission_enabled=True,
        nudge=lambda: nudges.append("nudge"),
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.append_messages(
            user_id="usr-a",
            project_id="proj-a",
            session_id="sess-a",
            messages=[{"role": "user", "content": "hello"}],
        )

    assert exc_info.value.status_code == 409
    assert isinstance(exc_info.value.detail, dict)
    detail = cast(dict[str, Any], exc_info.value.detail)
    assert detail["error_code"] == "graph_store_migration_required"
    assert nudges == []
    assert store.calls == []


async def test_nudge_failure_does_not_roll_back_committed_append():
    store = RecordingStore()

    def broken_nudge() -> None:
        raise RuntimeError("worker unavailable")

    service = _conversation_service(
        store,
        admission_enabled=True,
        nudge=broken_nudge,
    )

    saved = await service.append_messages(
        user_id="usr-a",
        project_id="proj-a",
        session_id="sess-a",
        messages=[{"role": "user", "content": "committed"}],
    )

    assert saved is not None
    assert [call[0] for call in store.calls] == ["append_with_memory"]
