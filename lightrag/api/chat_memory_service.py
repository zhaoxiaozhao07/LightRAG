"""Per-user-per-project chat memory backed by graphiti (docs/ChatMemory-zh.md).

Every (user, project) pair from the enterprise chat conversation management
(``/chat``) owns an isolated temporal knowledge-graph partition (graphiti
``group_id``). Persisted Q&A messages are asynchronously distilled into
entity/fact edges with bi-temporal validity; the facts flow back either through
``POST /chat/projects/{id}/memory:search`` or through server-side injection on
the query/agent endpoints (``memory: {"project_id": ...}``).

Design invariants:

- graphiti is imported lazily; the module (and the whole API server) works
  without ``graphiti-core`` installed as long as the feature stays disabled.
- Reads and purges always carry an explicit non-empty ``group_ids`` list —
  ``None`` means "whole database" to graphiti and must never leak through.
- Ingestion is best-effort and fully asynchronous: same-group episodes are
  serialized (required by graphiti's edge-invalidation logic), cross-group
  concurrency is capped, and failures are logged without affecting the
  originating chat request.
- With a metadata store attached, ingestion is idempotent (per-session seq
  watermark in ``enterprise_chat_memory_episodes``) and crash-lost work is
  re-ingested by the startup backlog scan; deleting a message/session removes
  the graphiti episodes distilled from it.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable, Sequence
from uuid import uuid4

from lightrag.api.metadata_store import ChatMemoryEpisodeRecord
from lightrag.utils import logger

GROUP_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
MEMORY_SEARCH_MAX_LIMIT = 50
MEMORY_QUERY_MAX_LENGTH = 4096
MEMORY_INGEST_MODES = ("immediate", "debounced")
_INGEST_ROLES = ("user", "assistant")
_TRUNCATION_MARKER = "…[truncated]"
_FINALIZE_DRAIN_TIMEOUT_SECONDS = 10.0
# Episode rows with this uuid prefix advance the ingestion watermark without a
# graphiti episode behind them (e.g. a range whose messages were all blank).
_NOOP_EPISODE_PREFIX = "noop_"


class ChatMemoryUnavailableError(RuntimeError):
    """Raised when the memory backend cannot be used (missing dependency,
    unreachable Neo4j, incomplete configuration)."""


def _first_set(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return default


@dataclass(frozen=True)
class ChatMemoryConfig:
    """Resolved chat-memory settings.

    ``from_args`` applies the documented fallback chains so a deployment that
    already configures QUERY/EMBEDDING/NEO4J/RERANK needs nothing beyond the
    enable flag (see docs/ChatMemory-zh.md §8.2).
    """

    enabled: bool = False
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_small_model: str | None = None
    llm_temperature: float = 0.0
    llm_max_tokens: int = 16384
    llm_timeout: int = 300
    llm_extra_body: dict[str, Any] | None = None
    structured_output_mode: str = "json_schema"
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_model: str | None = None
    embedding_dim: int | None = None
    neo4j_uri: str | None = None
    neo4j_username: str | None = None
    neo4j_password: str | None = None
    neo4j_database: str = "neo4j"
    search_limit: int = 10
    ingest_concurrency: int = 2
    max_coroutines: int = 4
    ingest_max_chars: int = 6000
    # Rerank facts with the deployment reranker (cross-encoder recipe) instead
    # of plain RRF; requires a server rerank function to be wired in.
    rerank_enabled: bool = False
    # "immediate" distills every persisted batch right away; "debounced"
    # buffers per session and flushes after a quiet window (fewer, larger
    # episodes => fewer LLM calls).
    ingest_mode: str = "immediate"
    ingest_debounce_seconds: float = 20.0
    # Startup compensation: re-ingest messages whose seq ran ahead of the
    # per-session watermark (fire-and-forget losses, debounce buffers lost to
    # a crash, LLM outages).
    backlog_scan_on_start: bool = True
    backlog_batch_messages: int = 20
    # Fairness cap: max concurrent in-flight ingest tasks per user. A user
    # spamming messages cannot starve the shared LLM / other users' ingestion;
    # skipped batches are recovered by the watermark backlog scan. 0 disables.
    max_inflight_per_user: int = 8

    @classmethod
    def from_args(cls, args: Any) -> "ChatMemoryConfig":
        def arg(name: str) -> Any:
            return getattr(args, name, None)

        extra_body: dict[str, Any] | None = None
        raw_extra_body = arg("memory_openai_llm_extra_body")
        if raw_extra_body:
            try:
                parsed = json.loads(raw_extra_body)
                if isinstance(parsed, dict) and parsed:
                    extra_body = parsed
                else:
                    logger.warning(
                        "MEMORY_OPENAI_LLM_EXTRA_BODY must be a JSON object; ignoring"
                    )
            except (TypeError, ValueError) as exc:
                logger.warning(
                    f"Invalid MEMORY_OPENAI_LLM_EXTRA_BODY, ignoring: {exc}"
                )

        structured_mode = str(
            _first_set(arg("memory_structured_output_mode"), default="json_schema")
        ).strip().lower()
        if structured_mode not in ("json_schema", "json_object"):
            logger.warning(
                "MEMORY_STRUCTURED_OUTPUT_MODE must be json_schema or json_object; "
                f"got {structured_mode!r}, using json_schema"
            )
            structured_mode = "json_schema"

        ingest_mode = str(
            _first_set(arg("memory_ingest_mode"), default="immediate")
        ).strip().lower()
        if ingest_mode not in MEMORY_INGEST_MODES:
            logger.warning(
                "MEMORY_INGEST_MODE must be immediate or debounced; "
                f"got {ingest_mode!r}, using immediate"
            )
            ingest_mode = "immediate"

        llm_model = _first_set(
            arg("memory_llm_model"), arg("query_llm_model"), arg("llm_model")
        )
        embedding_dim = _first_set(
            arg("memory_embedding_dim"), arg("embedding_dim")
        )

        def clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
            try:
                number = int(value)
            except (TypeError, ValueError):
                return default
            return max(minimum, min(maximum, number))

        def clamp_float(
            value: Any, default: float, minimum: float, maximum: float
        ) -> float:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return default
            return max(minimum, min(maximum, number))

        return cls(
            enabled=bool(arg("chat_memory_enabled")),
            llm_base_url=_first_set(
                arg("memory_llm_binding_host"),
                arg("query_llm_binding_host"),
                arg("llm_binding_host"),
            ),
            llm_api_key=_first_set(
                arg("memory_llm_binding_api_key"),
                arg("query_llm_binding_api_key"),
                arg("llm_binding_api_key"),
            ),
            llm_model=llm_model,
            llm_small_model=_first_set(arg("memory_llm_small_model"), llm_model),
            llm_temperature=float(
                _first_set(arg("memory_llm_temperature"), default=0.0)
            ),
            llm_max_tokens=clamp_int(
                arg("memory_llm_max_tokens"), 16384, 1024, 131072
            ),
            llm_timeout=clamp_int(arg("memory_llm_timeout"), 300, 1, 3600),
            llm_extra_body=extra_body,
            structured_output_mode=structured_mode,
            embedding_base_url=_first_set(
                arg("memory_embedding_binding_host"), arg("embedding_binding_host")
            ),
            embedding_api_key=_first_set(
                arg("memory_embedding_binding_api_key"),
                arg("embedding_binding_api_key"),
            ),
            embedding_model=_first_set(
                arg("memory_embedding_model"), arg("embedding_model")
            ),
            embedding_dim=int(embedding_dim) if embedding_dim else None,
            neo4j_uri=_first_set(arg("memory_neo4j_uri"), os.getenv("NEO4J_URI")),
            neo4j_username=_first_set(
                arg("memory_neo4j_username"), os.getenv("NEO4J_USERNAME")
            ),
            neo4j_password=_first_set(
                arg("memory_neo4j_password"), os.getenv("NEO4J_PASSWORD")
            ),
            neo4j_database=str(
                _first_set(
                    arg("memory_neo4j_database"),
                    os.getenv("NEO4J_DATABASE"),
                    default="neo4j",
                )
            ),
            search_limit=clamp_int(
                arg("memory_search_limit"), 10, 1, MEMORY_SEARCH_MAX_LIMIT
            ),
            ingest_concurrency=clamp_int(arg("memory_ingest_concurrency"), 2, 1, 64),
            max_coroutines=clamp_int(arg("memory_max_coroutines"), 4, 1, 64),
            ingest_max_chars=clamp_int(
                arg("memory_ingest_max_chars"), 6000, 200, 200_000
            ),
            rerank_enabled=bool(arg("memory_rerank_enabled")),
            ingest_mode=ingest_mode,
            ingest_debounce_seconds=clamp_float(
                arg("memory_ingest_debounce_seconds"), 20.0, 1.0, 3600.0
            ),
            backlog_scan_on_start=bool(
                _first_set(arg("memory_backlog_scan_on_start"), default=True)
            ),
            backlog_batch_messages=clamp_int(
                arg("memory_backlog_batch_messages"), 20, 2, 100
            ),
            max_inflight_per_user=clamp_int(
                arg("memory_max_inflight_per_user"), 8, 0, 1000
            ),
        )


class _ExtraBodyAsyncOpenAI:
    """Delegating AsyncOpenAI wrapper that injects a fixed ``extra_body``.

    graphiti's ``OpenAIGenericClient`` only calls
    ``client.chat.completions.create``; wrapping at that call site lets
    deployments pass vLLM/Qwen switches such as
    ``{"chat_template_kwargs": {"enable_thinking": false}}`` without patching
    graphiti. Per-call ``extra_body`` keys win over the configured ones.
    """

    def __init__(self, inner: Any, extra_body: dict[str, Any]):
        self._inner = inner
        self._extra_body = dict(extra_body)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create_with_extra_body)
        )

    async def _create_with_extra_body(self, **kwargs: Any) -> Any:
        merged = dict(self._extra_body)
        merged.update(kwargs.pop("extra_body", None) or {})
        return await self._inner.chat.completions.create(extra_body=merged, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _RerankFnCrossEncoder:
    """graphiti CrossEncoderClient adapter over the deployment rerank service.

    ``rerank_fn`` is the server rerank callable built in ``create_app`` from
    RERANK_BINDING/RERANK_MODEL/RERANK_BINDING_HOST (signature
    ``(query, documents, top_n=None, extra_body=None) -> [{"index",
    "relevance_score"}]``). Failures fall back to the original passage order so
    a flaky reranker degrades search quality, never availability.
    """

    def __init__(self, rerank_fn: Callable[..., Any]):
        self._rerank_fn = rerank_fn

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        if not passages:
            return []
        try:
            results = await self._rerank_fn(
                query=query, documents=list(passages), top_n=len(passages)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Chat memory rerank failed; keeping RRF order: {exc}")
            return [(passage, 1.0) for passage in passages]
        scores: dict[int, float] = {}
        for item in results or []:
            try:
                index = int(item.get("index", -1))
                if 0 <= index < len(passages):
                    scores[index] = float(item.get("relevance_score", 0.0))
            except (TypeError, ValueError, AttributeError):
                continue
        pairs = [
            (passage, scores.get(position, 0.0))
            for position, passage in enumerate(passages)
        ]
        pairs.sort(key=lambda pair: pair[1], reverse=True)
        return pairs


def _default_graphiti_factory(
    config: ChatMemoryConfig, cross_encoder: Any = None
) -> Any:
    """Build a real graphiti instance from the resolved configuration.

    Imports are local so the API server never needs ``graphiti-core`` unless
    the feature is actually enabled.
    """
    # Anonymous telemetry defaults ON upstream; this deployment is intranet.
    os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")
    # Mirror lightrag.kg.neo4j_impl: the driver's notification logger warns on
    # every first query against an empty memory partition ("property key does
    # not exist" server hints). Suppress it here too so deployments whose
    # engine graph backend is not Neo4j stay quiet as well.
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
    try:
        from graphiti_core import Graphiti
        from graphiti_core.cross_encoder.client import CrossEncoderClient
        from graphiti_core.driver.neo4j_driver import Neo4jDriver
        from graphiti_core.embedder.openai import (
            OpenAIEmbedder,
            OpenAIEmbedderConfig,
        )
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import (
            OpenAIGenericClient,
        )
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ChatMemoryUnavailableError(
            "graphiti-core is not installed; install the 'memory' extra "
            '(pip install "lightrag-hku[memory]")'
        ) from exc

    if not config.neo4j_uri:
        raise ChatMemoryUnavailableError(
            "Chat memory requires MEMORY_NEO4J_URI (or NEO4J_URI)"
        )
    if not config.llm_base_url or not config.llm_model:
        raise ChatMemoryUnavailableError(
            "Chat memory requires an OpenAI-compatible LLM endpoint "
            "(MEMORY_LLM_BINDING_HOST/MEMORY_LLM_MODEL or the QUERY/base "
            "LLM fallbacks)"
        )

    class _PassthroughCrossEncoder(CrossEncoderClient):
        # The RRF search recipe used by ``Graphiti.search`` never invokes the
        # cross encoder; this placeholder only prevents graphiti from
        # constructing its OpenAI reranker (whose logit_bias token ids assume
        # the OpenAI tokenizer and which requires OPENAI_API_KEY).
        async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
            return [(passage, 1.0) for passage in passages]

    llm_api_key = config.llm_api_key or "dummy-key"
    raw_llm_client = AsyncOpenAI(
        api_key=llm_api_key,
        base_url=config.llm_base_url,
        timeout=float(config.llm_timeout),
    )
    llm_client: Any = raw_llm_client
    if config.llm_extra_body:
        llm_client = _ExtraBodyAsyncOpenAI(raw_llm_client, config.llm_extra_body)
    llm = OpenAIGenericClient(
        config=LLMConfig(
            api_key=llm_api_key,
            model=config.llm_model,
            base_url=config.llm_base_url,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            small_model=config.llm_small_model or config.llm_model,
        ),
        client=llm_client,
        max_tokens=config.llm_max_tokens,
        structured_output_mode=config.structured_output_mode,  # type: ignore[arg-type]
    )

    embedder_kwargs: dict[str, Any] = {
        "api_key": config.embedding_api_key or "dummy-key",
        "base_url": config.embedding_base_url,
    }
    if config.embedding_model:
        embedder_kwargs["embedding_model"] = config.embedding_model
    if config.embedding_dim:
        embedder_kwargs["embedding_dim"] = config.embedding_dim
    embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(**embedder_kwargs))

    driver = Neo4jDriver(
        uri=config.neo4j_uri,
        user=config.neo4j_username,
        password=config.neo4j_password,
        database=config.neo4j_database or "neo4j",
    )
    return Graphiti(
        graph_driver=driver,
        llm_client=llm,
        embedder=embedder,
        cross_encoder=cross_encoder or _PassthroughCrossEncoder(),
        max_coroutines=config.max_coroutines,
    )


async def _default_clear_data(graphiti: Any, group_ids: list[str]) -> None:
    from graphiti_core.utils.maintenance.graph_data_operations import clear_data

    await clear_data(graphiti.driver, group_ids)


def _iso_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _seq_of(message: Any) -> int:
    try:
        return int(getattr(message, "seq", 0) or 0)
    except (TypeError, ValueError):
        return 0


class ChatMemoryService:
    """graphiti-backed project memory for the enterprise chat feature.

    ``graphiti_factory``/``clear_data_fn`` are injection points for tests; the
    defaults build a real graphiti instance lazily on first use (and retry on
    the next use if e.g. Neo4j was down at startup). ``metadata_store``
    enables the durability features (idempotent watermarks, backlog
    compensation, message/session-level forget); without it the service
    degrades to fire-and-forget ingestion. ``rerank_fn`` (the server rerank
    callable) upgrades search to the cross-encoder recipe when
    ``MEMORY_RERANK_ENABLED`` is on.
    """

    def __init__(
        self,
        config: ChatMemoryConfig,
        audit_service: Any = None,
        graphiti_factory: Callable[..., Any] | None = None,
        clear_data_fn: Callable[[Any, list[str]], Any] | None = None,
        metadata_store: Any = None,
        rerank_fn: Callable[..., Any] | None = None,
    ):
        self._config = config
        self._audit_service = audit_service
        self._metadata_store = metadata_store
        self._use_cross_encoder = bool(config.rerank_enabled and rerank_fn)
        cross_encoder = (
            _RerankFnCrossEncoder(rerank_fn) if self._use_cross_encoder else None
        )
        if graphiti_factory is not None:
            self._graphiti_factory = graphiti_factory
        else:
            self._graphiti_factory = (
                lambda cfg: _default_graphiti_factory(cfg, cross_encoder=cross_encoder)
            )
        self._clear_data_fn = clear_data_fn or _default_clear_data
        self._graphiti: Any = None
        self._init_lock = asyncio.Lock()
        self._group_locks: dict[str, asyncio.Lock] = {}
        self._ingest_semaphore = asyncio.Semaphore(max(1, config.ingest_concurrency))
        self._background_tasks: set[asyncio.Task] = set()
        self._inflight_per_user: dict[str, int] = {}
        self._debounce_buffers: dict[tuple[str, str, str], list[Any]] = {}
        self._debounce_timers: dict[tuple[str, str, str], asyncio.Task] = {}
        self._closed = False

    # ------------------------------------------------------------------ state

    @property
    def config(self) -> ChatMemoryConfig:
        return self._config

    @property
    def available(self) -> bool:
        return self._graphiti is not None

    @property
    def pending_background_tasks(self) -> int:
        return len([task for task in self._background_tasks if not task.done()])

    @staticmethod
    def build_group_id(user_id: str, project_id: str) -> str:
        """Compose the graphiti partition key for one user's project.

        Raises ``ValueError`` on ids that would break graphiti's group-id
        charset (defense in depth; generated ids are hex-based).
        """
        group_id = f"{user_id}--{project_id}"
        if not user_id or not project_id or not GROUP_ID_PATTERN.match(group_id):
            raise ValueError(f"Invalid chat memory group id: {group_id!r}")
        return group_id

    async def initialize(self) -> bool:
        """Best-effort startup initialization; never raises.

        Returns availability. On failure the service stays constructed and
        retries lazily on the first ingest/search. On success the startup
        backlog scan (compensation for work lost to a previous crash) runs in
        the background.
        """
        try:
            await self._ensure_ready()
        except Exception as exc:
            logger.error(
                "Chat memory initialization failed (will retry on first use): "
                f"{exc}"
            )
            return False
        logger.info("Chat memory service ready (graphiti initialized)")
        if self._config.backlog_scan_on_start and self._metadata_store is not None:
            self._track(asyncio.create_task(self._startup_backlog_scan()))
        return True

    async def _startup_backlog_scan(self) -> None:
        try:
            ingested = await self.run_backlog_scan()
            if ingested:
                logger.info(
                    f"Chat memory backlog scan re-ingested {ingested} episode(s)"
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Chat memory backlog scan failed: {exc}")

    async def finalize(self) -> None:
        self._closed = True
        # Flush debounce buffers so a graceful shutdown does not rely on the
        # next startup's backlog scan.
        for key in list(self._debounce_buffers.keys()):
            timer = self._debounce_timers.pop(key, None)
            if timer is not None and not timer.done():
                timer.cancel()
            messages = self._debounce_buffers.pop(key, [])
            if messages:
                user_id, project_id, session_id = key
                self._track(
                    asyncio.create_task(
                        self._ingest(
                            user_id=user_id,
                            project_id=project_id,
                            session_id=session_id,
                            messages=messages,
                        )
                    )
                )
        pending = [task for task in self._background_tasks if not task.done()]
        if pending:
            done, still_pending = await asyncio.wait(
                pending, timeout=_FINALIZE_DRAIN_TIMEOUT_SECONDS
            )
            for task in still_pending:
                task.cancel()
            if still_pending:
                logger.warning(
                    f"Chat memory shutdown cancelled {len(still_pending)} "
                    "in-flight background task(s)"
                )
        graphiti, self._graphiti = self._graphiti, None
        if graphiti is not None:
            try:
                close = getattr(graphiti, "close", None)
                if close is not None:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
            except Exception as exc:
                logger.warning(f"Chat memory close failed: {exc}")

    async def _ensure_ready(self) -> Any:
        if self._closed:
            raise ChatMemoryUnavailableError("Chat memory service is shut down")
        if self._graphiti is not None:
            return self._graphiti
        async with self._init_lock:
            if self._graphiti is not None:
                return self._graphiti
            try:
                graphiti = self._graphiti_factory(self._config)
                if inspect.isawaitable(graphiti):
                    graphiti = await graphiti
                build_indices = getattr(
                    graphiti, "build_indices_and_constraints", None
                )
                if build_indices is not None:
                    # Idempotent (IF NOT EXISTS); never pass delete_existing.
                    await build_indices()
            except ChatMemoryUnavailableError:
                raise
            except Exception as exc:
                raise ChatMemoryUnavailableError(
                    f"Chat memory backend unavailable: {exc}"
                ) from exc
            self._graphiti = graphiti
            return graphiti

    # ------------------------------------------------------------------ audit

    async def _audit(
        self,
        event_type: str,
        *,
        actor_user_id: str,
        target_type: str,
        target_id: str,
        metadata: dict[str, Any],
    ) -> None:
        if self._audit_service is not None:
            await self._audit_service.append(
                event_type,
                actor_user_id=actor_user_id,
                target_type=target_type,
                target_id=target_id,
                metadata=metadata,
            )

    # ----------------------------------------------------------------- ingest

    def schedule_ingest(
        self,
        *,
        user_id: str,
        project_id: str,
        session_id: str,
        messages: Sequence[Any],
    ) -> asyncio.Task | None:
        """Fire-and-forget distillation of freshly persisted chat messages.

        ``messages`` are ``ChatMessageRecord``-shaped objects (``role``,
        ``content``, ``seq``, ``created_at``). In ``debounced`` mode batches
        are buffered per session and flushed after a quiet window. Returns the
        created task (kept strongly referenced until done) or ``None`` when
        there is nothing to ingest or the service is shutting down.
        """
        if self._closed:
            return None
        payload = [
            message
            for message in messages
            if getattr(message, "role", None) in _INGEST_ROLES
            and str(getattr(message, "content", "") or "").strip()
        ]
        if not payload:
            return None
        if self._config.ingest_mode == "debounced":
            key = (user_id, project_id, session_id)
            self._debounce_buffers.setdefault(key, []).extend(payload)
            previous = self._debounce_timers.get(key)
            if previous is not None and not previous.done():
                previous.cancel()
            task = asyncio.create_task(self._debounce_flush(key))
            self._debounce_timers[key] = task
            self._track(task)
            return task
        # Per-user fairness: cap concurrent in-flight ingests so one user can't
        # monopolize the shared LLM. Skipped batches persist in chat storage and
        # are recovered by the watermark backlog scan, so nothing is lost.
        cap = self._config.max_inflight_per_user
        if cap and self._inflight_per_user.get(user_id, 0) >= cap:
            logger.debug(
                f"Chat memory ingest deferred for user {user_id} "
                f"(in-flight cap {cap} reached; backlog scan will catch up)"
            )
            return None
        self._inflight_per_user[user_id] = (
            self._inflight_per_user.get(user_id, 0) + 1
        )
        task = asyncio.create_task(
            self._ingest(
                user_id=user_id,
                project_id=project_id,
                session_id=session_id,
                messages=payload,
            )
        )
        task.add_done_callback(
            lambda _t, uid=user_id: self._release_inflight(uid)
        )
        self._track(task)
        return task

    def _release_inflight(self, user_id: str) -> None:
        remaining = self._inflight_per_user.get(user_id, 0) - 1
        if remaining > 0:
            self._inflight_per_user[user_id] = remaining
        else:
            self._inflight_per_user.pop(user_id, None)

    async def _debounce_flush(self, key: tuple[str, str, str]) -> None:
        try:
            await asyncio.sleep(self._config.ingest_debounce_seconds)
        except asyncio.CancelledError:
            # Superseded by a newer batch in the same session; the restarted
            # timer owns the buffer now.
            return
        self._debounce_timers.pop(key, None)
        messages = self._debounce_buffers.pop(key, [])
        if not messages:
            return
        user_id, project_id, session_id = key
        await self._ingest(
            user_id=user_id,
            project_id=project_id,
            session_id=session_id,
            messages=messages,
        )

    def _track(self, task: asyncio.Task) -> None:
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _ingestible(self, messages: Sequence[Any]) -> list[Any]:
        return [
            message
            for message in messages
            if getattr(message, "role", None) in _INGEST_ROLES
            and str(getattr(message, "content", "") or "").strip()
        ]

    def _episode_body(self, messages: Sequence[Any]) -> str:
        max_chars = self._config.ingest_max_chars
        lines: list[str] = []
        for message in messages:
            content = str(getattr(message, "content", "") or "").strip()
            if len(content) > max_chars:
                content = content[:max_chars] + _TRUNCATION_MARKER
            lines.append(f"{getattr(message, 'role', 'user')}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _reference_time(messages: Sequence[Any]) -> datetime:
        raw = getattr(messages[0], "created_at", None)
        if isinstance(raw, datetime):
            return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
        if isinstance(raw, str):
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                return (
                    parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
                )
            except ValueError:
                pass
        return datetime.now(timezone.utc)

    async def _record_episode(
        self,
        *,
        episode_uuid: str,
        user_id: str,
        project_id: str,
        session_id: str,
        first_seq: int,
        last_seq: int,
    ) -> None:
        if self._metadata_store is None:
            return
        from lightrag.api.kb_service import utc_now_iso

        await self._metadata_store.record_chat_memory_episode(
            ChatMemoryEpisodeRecord(
                episode_uuid=episode_uuid,
                session_id=session_id,
                project_id=project_id,
                user_id=user_id,
                first_seq=first_seq,
                last_seq=last_seq,
                created_at=utc_now_iso(),
            )
        )

    async def _ingest(
        self,
        *,
        user_id: str,
        project_id: str,
        session_id: str,
        messages: Sequence[Any],
        force: bool = False,
    ) -> None:
        """Distill one batch into the project's memory graph.

        With a metadata store, ingestion is idempotent: messages at or below
        the session watermark are skipped (protects against the backlog scan
        racing a live append), and a successful episode advances the
        watermark. ``force=True`` bypasses the watermark for survivor
        re-ingestion after a message delete.
        """
        try:
            group_id = self.build_group_id(user_id, project_id)
        except ValueError as exc:
            logger.warning(f"Chat memory ingest skipped: {exc}")
            return
        try:
            graphiti = await self._ensure_ready()
        except ChatMemoryUnavailableError as exc:
            logger.warning(
                f"Chat memory ingest skipped for {group_id} (unavailable): {exc}"
            )
            return
        episode_name = ""
        try:
            # graphiti requires sequential add_episode per group (edge
            # invalidation reads the partition it is about to update); the
            # semaphore only bounds cross-group LLM pressure.
            lock = self._group_locks.setdefault(group_id, asyncio.Lock())
            async with lock:
                effective = list(messages)
                if self._metadata_store is not None and not force:
                    watermark = await self._metadata_store.get_chat_memory_watermark(
                        user_id, project_id, session_id
                    )
                    effective = [
                        message
                        for message in effective
                        if _seq_of(message) > watermark
                    ]
                if not effective:
                    return
                seqs = [_seq_of(message) for message in effective]
                first_seq, last_seq = min(seqs), max(seqs)
                episode_name = f"{session_id}:{first_seq}-{last_seq}"
                ingestible = self._ingestible(effective)
                if not ingestible:
                    # Nothing distillable in the range (e.g. blank content);
                    # still advance the watermark so the backlog scan
                    # converges instead of retrying the range forever.
                    await self._record_episode(
                        episode_uuid=f"{_NOOP_EPISODE_PREFIX}{uuid4().hex}",
                        user_id=user_id,
                        project_id=project_id,
                        session_id=session_id,
                        first_seq=first_seq,
                        last_seq=last_seq,
                    )
                    return
                async with self._ingest_semaphore:
                    result = await graphiti.add_episode(
                        name=episode_name,
                        episode_body=self._episode_body(ingestible),
                        source_description="enterprise chat",
                        reference_time=self._reference_time(ingestible),
                        group_id=group_id,
                    )
                episode_uuid = (
                    getattr(getattr(result, "episode", None), "uuid", None)
                    or f"ep_{uuid4().hex}"
                )
                await self._record_episode(
                    episode_uuid=str(episode_uuid),
                    user_id=user_id,
                    project_id=project_id,
                    session_id=session_id,
                    first_seq=first_seq,
                    last_seq=last_seq,
                )
            await self._audit(
                "chat_memory_ingested",
                actor_user_id=user_id,
                target_type="chat_session",
                target_id=session_id,
                metadata={
                    "user_id": user_id,
                    "project_id": project_id,
                    "session_id": session_id,
                    "message_count": len(ingestible),
                    "episode_uuid": str(episode_uuid),
                },
            )
            logger.debug(
                f"Chat memory episode ingested: group={group_id} "
                f"episode={episode_name}"
            )
        except Exception as exc:
            logger.warning(
                f"Chat memory ingest failed for {group_id} "
                f"({episode_name or session_id}): {exc}"
            )

    # ----------------------------------------------------- backlog (recovery)

    async def run_backlog_scan(self, *, limit: int = 100) -> int:
        """Re-ingest messages whose ``seq`` ran ahead of the session watermark.

        Covers fire-and-forget tasks lost to a crash/restart, debounce buffers
        that never flushed, and appends made while the memory backend was
        down. Returns the number of ingested batches. No-op without a
        metadata store.
        """
        if self._metadata_store is None:
            return 0
        await self._ensure_ready()
        items = await self._metadata_store.list_chat_memory_backlog(limit=limit)
        batches = 0
        batch_size = max(2, self._config.backlog_batch_messages)
        for item in items:
            messages = await self._metadata_store.list_chat_messages_after_seq(
                item.user_id,
                item.project_id,
                item.session_id,
                after_seq=item.ingested_seq,
                limit=1000,
            )
            for start in range(0, len(messages), batch_size):
                await self._ingest(
                    user_id=item.user_id,
                    project_id=item.project_id,
                    session_id=item.session_id,
                    messages=messages[start : start + batch_size],
                )
                batches += 1
        return batches

    # ----------------------------------------------------------------- search

    async def search(
        self,
        *,
        user_id: str,
        project_id: str,
        query: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search over one project's memory partition.

        Uses the cross-encoder recipe when the deployment reranker is wired
        in (``MEMORY_RERANK_ENABLED``), plain RRF otherwise. Raises
        ``ChatMemoryUnavailableError`` when the backend cannot be reached;
        ownership checks belong to the caller (route layer).
        """
        group_id = self.build_group_id(user_id, project_id)
        graphiti = await self._ensure_ready()
        effective_limit = limit if limit is not None else self._config.search_limit
        effective_limit = max(1, min(MEMORY_SEARCH_MAX_LIMIT, int(effective_limit)))
        edges: list[Any] | None = None
        if self._use_cross_encoder and hasattr(graphiti, "search_"):
            try:
                from graphiti_core.search.search_config_recipes import (
                    EDGE_HYBRID_SEARCH_CROSS_ENCODER,
                )

                search_config = EDGE_HYBRID_SEARCH_CROSS_ENCODER.model_copy(deep=True)
                search_config.limit = effective_limit
                results = await graphiti.search_(
                    query=query,
                    config=search_config,
                    group_ids=[group_id],
                )
                edges = list(getattr(results, "edges", None) or [])
            except ImportError:
                edges = None
        if edges is None:
            edges = await graphiti.search(
                query=query,
                group_ids=[group_id],
                num_results=effective_limit,
            )
        facts = [
            {
                "uuid": str(getattr(edge, "uuid", "") or ""),
                "name": str(getattr(edge, "name", "") or ""),
                "fact": str(getattr(edge, "fact", "") or ""),
                "valid_at": _iso_or_none(getattr(edge, "valid_at", None)),
                "invalid_at": _iso_or_none(getattr(edge, "invalid_at", None)),
                "created_at": _iso_or_none(getattr(edge, "created_at", None)),
                "expired_at": _iso_or_none(getattr(edge, "expired_at", None)),
            }
            for edge in edges
        ]
        await self._audit(
            "chat_memory_searched",
            actor_user_id=user_id,
            target_type="chat_project",
            target_id=project_id,
            metadata={
                "user_id": user_id,
                "project_id": project_id,
                "query_hash": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                "fact_count": len(facts),
                "limit": effective_limit,
            },
        )
        return facts

    # -------------------------------------------------------- prompt building

    @staticmethod
    def format_memory_block(facts: Sequence[dict[str, Any]]) -> str:
        """Render facts as the context block injected into query prompts."""
        lines: list[str] = []
        for fact in facts:
            text = str(fact.get("fact", "") or "").strip()
            if not text:
                continue
            if fact.get("invalid_at"):
                suffix = f"（已失效于 {str(fact['invalid_at'])[:10]}，仅供追溯）"
            elif fact.get("valid_at"):
                suffix = f"（自 {str(fact['valid_at'])[:10]} 起）"
            else:
                suffix = ""
            lines.append(f"- {text}{suffix}")
        if not lines:
            return ""
        return "\n".join(
            [
                "[项目记忆] 以下是该项目历史对话沉淀的事实：",
                *lines,
                "请结合以上项目记忆回答本次问题；若记忆与检索到的证据冲突，以检索证据为准。",
            ]
        )

    async def build_memory_block(
        self,
        *,
        user_id: str,
        project_id: str,
        query: str,
        limit: int | None = None,
    ) -> tuple[str | None, dict[str, Any]]:
        """Search + format for server-side prompt injection.

        Fail-open: an unavailable backend degrades to no injection with an
        explanatory info dict instead of failing the query.
        """
        try:
            facts = await self.search(
                user_id=user_id, project_id=project_id, query=query, limit=limit
            )
        except ChatMemoryUnavailableError as exc:
            logger.warning(
                f"Chat memory injection skipped for {project_id} (unavailable): {exc}"
            )
            return None, {"enabled": False, "reason": "unavailable"}
        info = {
            "enabled": True,
            "project_id": project_id,
            "fact_count": len(facts),
        }
        block = self.format_memory_block(facts)
        return (block or None), info

    # ---------------------------------------------------------------- overview

    async def project_overview(
        self, user_id: str, project_id: str
    ) -> dict[str, Any]:
        """Per-project memory overview (episode count + last ingest time).

        ``available`` reflects whether the backend is reachable; the counts come
        from the metadata mapping table so they work even when graphiti/Neo4j
        is momentarily down.
        """
        count, last_at = 0, None
        if self._metadata_store is not None:
            count, last_at = (
                await self._metadata_store.count_chat_memory_episodes_for_project(
                    user_id, project_id
                )
            )
        return {
            "project_id": project_id,
            "enabled": self._config.enabled,
            "available": self.available,
            "episode_count": count,
            "last_ingested_at": last_at,
        }

    async def global_stats(self) -> dict[str, Any]:
        """Global memory stats for admin observability (super admin)."""
        episodes = users = projects = 0
        if self._metadata_store is not None:
            episodes, users, projects = (
                await self._metadata_store.count_chat_memory_episodes()
            )
        return {
            "enabled": self._config.enabled,
            "available": self.available,
            "pending_tasks": self.pending_background_tasks,
            "episode_count": episodes,
            "user_count": users,
            "project_count": projects,
        }

    # ----------------------------------------------------------------- forget

    def schedule_forget_message(
        self, *, user_id: str, project_id: str, session_id: str, seq: int
    ) -> asyncio.Task | None:
        """Remove the episode(s) distilled from a deleted message and re-ingest
        the surviving messages of the covered range."""
        if self._closed or self._metadata_store is None:
            return None
        task = asyncio.create_task(
            self._forget_message(
                user_id=user_id,
                project_id=project_id,
                session_id=session_id,
                seq=seq,
            )
        )
        self._track(task)
        return task

    def schedule_forget_session(
        self, *, user_id: str, project_id: str, session_id: str
    ) -> asyncio.Task | None:
        """Remove every episode distilled from a deleted session."""
        if self._closed or self._metadata_store is None:
            return None
        task = asyncio.create_task(
            self._forget_session(
                user_id=user_id, project_id=project_id, session_id=session_id
            )
        )
        self._track(task)
        return task

    async def _remove_episodes(self, episodes: Sequence[Any]) -> int:
        """remove_episode for each mapping row (noop rows skip graphiti)."""
        if not episodes:
            return 0
        graphiti = await self._ensure_ready()
        removed = 0
        for episode in episodes:
            uuid_str = str(getattr(episode, "episode_uuid", "") or "")
            if not uuid_str or uuid_str.startswith(_NOOP_EPISODE_PREFIX):
                continue
            try:
                await graphiti.remove_episode(uuid_str)
                removed += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"Chat memory remove_episode failed for {uuid_str}: {exc}"
                )
        await self._metadata_store.delete_chat_memory_episodes(
            [str(getattr(episode, "episode_uuid", "") or "") for episode in episodes]
        )
        return removed

    async def _forget_message(
        self, *, user_id: str, project_id: str, session_id: str, seq: int
    ) -> None:
        try:
            episodes = await self._metadata_store.find_chat_memory_episodes_covering(
                user_id, project_id, session_id, int(seq)
            )
            if not episodes:
                return
            removed = await self._remove_episodes(episodes)
            # Re-ingest the surviving messages of the covered ranges so the
            # rest of those turns stays remembered.
            first = min(int(episode.first_seq) for episode in episodes)
            last = max(int(episode.last_seq) for episode in episodes)
            survivors = await self._metadata_store.list_chat_messages_after_seq(
                user_id,
                project_id,
                session_id,
                after_seq=first - 1,
                limit=1000,
            )
            survivors = [
                message for message in survivors if _seq_of(message) <= last
            ]
            if survivors:
                await self._ingest(
                    user_id=user_id,
                    project_id=project_id,
                    session_id=session_id,
                    messages=survivors,
                    force=True,
                )
            await self._audit(
                "chat_memory_forgotten",
                actor_user_id=user_id,
                target_type="chat_session",
                target_id=session_id,
                metadata={
                    "user_id": user_id,
                    "project_id": project_id,
                    "session_id": session_id,
                    "scope": "message",
                    "episode_count": removed,
                    "reingested_messages": len(survivors),
                },
            )
        except ChatMemoryUnavailableError as exc:
            logger.warning(
                f"Chat memory forget skipped for {session_id} (unavailable): {exc}"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Chat memory forget failed for {session_id}: {exc}")

    async def _forget_session(
        self, *, user_id: str, project_id: str, session_id: str
    ) -> None:
        try:
            episodes = (
                await self._metadata_store.list_chat_memory_episodes_for_session(
                    user_id, project_id, session_id
                )
            )
            if not episodes:
                return
            removed = await self._remove_episodes(episodes)
            await self._audit(
                "chat_memory_forgotten",
                actor_user_id=user_id,
                target_type="chat_session",
                target_id=session_id,
                metadata={
                    "user_id": user_id,
                    "project_id": project_id,
                    "session_id": session_id,
                    "scope": "session",
                    "episode_count": removed,
                },
            )
        except ChatMemoryUnavailableError as exc:
            logger.warning(
                f"Chat memory forget skipped for {session_id} (unavailable): {exc}"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Chat memory forget failed for {session_id}: {exc}")

    # ------------------------------------------------------------------ purge

    def schedule_purge(
        self, user_id: str, project_ids: Sequence[str]
    ) -> asyncio.Task | None:
        """Fire-and-forget removal of the memory partitions for deleted
        projects (or all projects of a deleted user)."""
        ids = [project_id for project_id in project_ids if project_id]
        if self._closed or not ids:
            return None
        task = asyncio.create_task(self._purge(user_id=user_id, project_ids=ids))
        self._track(task)
        return task

    async def purge_projects(self, user_id: str, project_ids: Sequence[str]) -> int:
        """Synchronous variant of the purge path; returns purged group count."""
        group_ids: list[str] = []
        valid_project_ids: list[str] = []
        for project_id in project_ids:
            try:
                group_ids.append(self.build_group_id(user_id, project_id))
                valid_project_ids.append(project_id)
            except ValueError as exc:
                logger.warning(f"Chat memory purge skipped invalid group: {exc}")
        if not group_ids:
            return 0
        graphiti = await self._ensure_ready()
        # Guard the graphiti clear_data(None) == "wipe the whole database"
        # footgun: only ever pass an explicit non-empty list.
        assert group_ids, "chat memory purge requires explicit group ids"
        result = self._clear_data_fn(graphiti, group_ids)
        if inspect.isawaitable(result):
            await result
        if self._metadata_store is not None:
            for project_id in valid_project_ids:
                await self._metadata_store.delete_chat_memory_episodes_for_project(
                    project_id
                )
        await self._audit(
            "chat_memory_purged",
            actor_user_id=user_id,
            target_type="chat_project",
            target_id=valid_project_ids[0]
            if len(valid_project_ids) == 1
            else user_id,
            metadata={
                "user_id": user_id,
                "project_count": len(group_ids),
            },
        )
        return len(group_ids)

    async def _purge(self, *, user_id: str, project_ids: list[str]) -> None:
        try:
            purged = await self.purge_projects(user_id, project_ids)
            logger.debug(
                f"Chat memory purged {purged} group(s) for user {user_id}"
            )
        except ChatMemoryUnavailableError as exc:
            logger.warning(
                f"Chat memory purge skipped for user {user_id} (unavailable): {exc}"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Chat memory purge failed for user {user_id}: {exc}")

    # ------------------------------------------------------------------ tests

    async def wait_for_background_tasks(self, timeout: float = 5.0) -> None:
        pending = [task for task in self._background_tasks if not task.done()]
        if pending:
            await asyncio.wait(pending, timeout=timeout)
