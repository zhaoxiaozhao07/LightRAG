from unittest.mock import MagicMock, patch
from typing import Any, cast

import numpy as np
import pytest

from lightrag.kg.milvus_impl import MilvusVectorDBStorage


class FakeEmbeddingFunc:
    embedding_dim = 3

    async def __call__(self, texts, context="document"):
        return np.array(
            [[float(index), float(index + 1), float(index + 2)] for index, _ in enumerate(texts)],
            dtype=np.float32,
        )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_milvus_upsert_logs_detailed_timing_segments():
    storage = MilvusVectorDBStorage(
        namespace="test_entities",
        workspace="test_workspace",
        global_config={
            "embedding_batch_num": 2,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.3,
                "index_type": "AUTOINDEX",
            },
        },
        embedding_func=cast(Any, FakeEmbeddingFunc()),
        meta_fields={"entity_name"},
    )
    storage._client = MagicMock()
    storage._ensure_collection_loaded = MagicMock()

    with patch("lightrag.kg.milvus_impl.performance_timing_log") as timing_log:
        await storage.upsert(
            {
                "ent-a": {"content": "alpha", "entity_name": "A"},
                "ent-b": {"content": "beta", "entity_name": "B"},
                "ent-c": {"content": "gamma", "entity_name": "C"},
            }
        )

    storage._ensure_collection_loaded.assert_called_once()
    storage._client.upsert.assert_called_once()
    _, kwargs = storage._client.upsert.call_args
    assert kwargs["collection_name"] == "test_workspace_test_entities"
    assert len(kwargs["data"]) == 3
    assert all("vector" in row for row in kwargs["data"])

    messages = [call.args[0] for call in timing_log.call_args_list]
    assert any("collection load completed" in message for message in messages)
    assert any("payload build completed" in message for message in messages)
    assert any("embedding generation completed" in message for message in messages)
    assert any("vector attach completed" in message for message in messages)
    assert any("client upsert completed" in message for message in messages)
    assert any("completed" in message for message in messages)
