from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from typing import Any

from lightrag.api.kb_service import KnowledgeBaseService, utc_now_iso
from lightrag.api.lightrag_registry import (
    DestructiveLockBusyError,
    LightRAGInstanceRegistry,
)
from lightrag.api.metadata_store import (
    ConfigVersionRecord,
    SQLiteMetadataStore,
)
from lightrag.base import QueryParam
from lightrag.constants import (
    PARSER_ENGINE_LEGACY,
    SUPPORTED_PARSER_ENGINES,
)
from lightrag.llm_roles import ROLE_NAMES
from lightrag.parser.routing import (
    normalize_parser_engine,
    sanitize_process_options,
    validate_process_options,
)
from lightrag.utils import EmbeddingFunc, generate_track_id

_ACTIVE_QUERY_CONFIG_KEYS = set(QueryParam.__dataclass_fields__)

# Which roles, when their model/binding identity changes, invalidate built
# index content (chunks / KG / vectors) vs only query-time behavior. The
# extraction and VLM roles produce persisted KG/chunk content, so a change
# there must trigger reindex; query/keyword roles only affect retrieval.
_INDEX_AFFECTING_ROLES = frozenset({"extract", "vlm"})
_QUERY_AFFECTING_ROLES = frozenset({"query", "keyword"})

# Keys accepted inside an ``llm_role_config[<role>]`` mapping. ``model_kwargs``
# and ``kwargs`` are aliases. Secret material (``api_key``) is accepted for
# runtime wiring but excluded from hashing identity.
_LLM_ROLE_CONFIG_KEYS = frozenset(
    {
        "model",
        "binding",
        "host",
        "api_key",
        "provider_options",
        "model_kwargs",
        "kwargs",
        "max_async",
        "timeout",
    }
)
# Fields that change LLM *output* (and therefore built content / answers).
# Excludes ``api_key`` (secret, not output-affecting) and perf-only knobs
# (``max_async`` / ``timeout``).
_LLM_ROLE_IDENTITY_KEYS = ("binding", "model", "host", "provider_options", "model_kwargs")


def _section_hash(section_name: str, payload: dict[str, Any] | None) -> str:
    blob = {
        "schema": f"kb-{section_name}-hash-v1",
        "payload": payload or {},
    }
    encoded = json.dumps(blob, ensure_ascii=False, sort_keys=True, default=str).encode(
        "utf-8"
    )
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _config_section(
    active_config_version: ConfigVersionRecord | None, section_name: str
) -> dict[str, Any]:
    if active_config_version is None:
        return {}
    return _raw_config_section(active_config_version.config, section_name)


def _raw_config_section(config: dict[str, Any] | None, section_name: str) -> dict[str, Any]:
    section = (config or {}).get(section_name)
    return dict(section) if isinstance(section, dict) else {}


def _first_present(config: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return None


def _positive_int(value: Any) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("Active config integer values must be positive")
    return parsed


def _non_negative_int(value: Any) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError("Active config integer values must be non-negative")
    return parsed


def _active_chunk_runtime_config(config: dict[str, Any] | None) -> dict[str, Any]:
    chunk_config = _raw_config_section(config, "chunk_config")
    runtime: dict[str, Any] = {}
    chunk_size = _positive_int(
        _first_present(chunk_config, "chunk_token_size", "chunk_size")
    )
    if chunk_size is not None:
        runtime["chunk_token_size"] = chunk_size
    chunk_overlap = _non_negative_int(
        _first_present(
            chunk_config,
            "chunk_overlap_token_size",
            "chunk_overlap_size",
            "overlap",
        )
    )
    if chunk_overlap is not None:
        runtime["chunk_overlap_token_size"] = chunk_overlap
    if chunk_config.get("tiktoken_model_name") is not None:
        runtime["tiktoken_model_name"] = str(chunk_config["tiktoken_model_name"])
    return runtime


def _active_embedding_runtime_config(config: dict[str, Any] | None) -> dict[str, Any]:
    embedding_config = _raw_config_section(config, "embedding_config")
    runtime: dict[str, Any] = {}
    embedding_dim = _positive_int(
        _first_present(embedding_config, "embedding_dim", "dim")
    )
    if embedding_dim is not None:
        runtime["embedding_dim"] = embedding_dim
    max_token_size = _positive_int(
        _first_present(embedding_config, "max_token_size", "token_limit")
    )
    if max_token_size is not None:
        runtime["max_token_size"] = max_token_size
    if embedding_config.get("model") is not None:
        runtime["model_name"] = str(embedding_config["model"])
    return runtime


def _active_parser_runtime_config(config: dict[str, Any] | None) -> dict[str, str]:
    parser_config = _raw_config_section(config, "parser_config")
    runtime: dict[str, str] = {}
    engine_value = _first_present(parser_config, "engine", "parser_engine")
    if engine_value is not None:
        engine = normalize_parser_engine(str(engine_value))
        if engine == PARSER_ENGINE_LEGACY:
            raise ValueError("parser_config.engine does not support legacy")
        if engine not in SUPPORTED_PARSER_ENGINES:
            raise ValueError(f"Unsupported parser_config.engine: {engine_value}")
        runtime["parser_engine"] = engine

    options_value = _first_present(parser_config, "process_options", "options")
    if options_value is not None:
        options_text = str(options_value)
        errors = validate_process_options(options_text)
        if errors:
            raise ValueError("; ".join(errors))
        runtime["process_options"] = sanitize_process_options(options_text)
    return runtime


def active_parser_runtime_config_from_version(
    active_config_version: ConfigVersionRecord | None,
) -> dict[str, str]:
    if active_config_version is None:
        return {}
    return _active_parser_runtime_config(active_config_version.config)


def active_embedding_runtime_config_from_version(
    active_config_version: ConfigVersionRecord | None,
) -> dict[str, Any]:
    if active_config_version is None:
        return {}
    return _active_embedding_runtime_config(active_config_version.config)


def _active_query_runtime_config(config: dict[str, Any] | None) -> dict[str, Any]:
    query_config = _raw_config_section(config, "query_config")
    runtime = {
        key: value
        for key, value in query_config.items()
        if key in _ACTIVE_QUERY_CONFIG_KEYS and value is not None
    }
    if query_config.get("cosine_threshold") is not None:
        runtime["cosine_threshold"] = float(query_config["cosine_threshold"])
    return runtime


def _entity_types_to_guidance(entity_types: list[str]) -> str:
    """Render a list of entity types into extraction-prompt guidance text.

    Mirrors the shape of ``PROMPTS['default_entity_types_guidance']`` so a KB
    can constrain extraction to a domain-specific type set without authoring a
    full prompt file.
    """
    lines = [
        "Classify each entity using one of the following types. "
        "If no type fits, use `Other`.",
        "",
    ]
    lines.extend(f"- {entity_type}" for entity_type in entity_types)
    return "\n".join(lines)


def _active_extraction_runtime_config(
    config: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Resolve the ``extraction_config`` section into runtime-applicable parts.

    Returns a dict with two sub-maps:

    - ``addon``: keys overlaid onto ``LightRAG.addon_params`` (``language``,
      ``entity_types_guidance``, ``entity_type_prompt_file``).
    - ``kwargs``: direct LightRAG constructor kwargs for extraction caps
      (``entity_extract_max_gleaning`` etc.).

    Only fields with stable LightRAG runtime equivalents are mapped; unknown
    keys stay persisted for diff/audit but do not affect the built instance.
    """
    extraction_config = _raw_config_section(config, "extraction_config")
    addon: dict[str, Any] = {}
    kwargs: dict[str, Any] = {}
    if not extraction_config:
        return {"addon": addon, "kwargs": kwargs}

    language = extraction_config.get("language")
    if language is not None:
        if not isinstance(language, str) or not language.strip():
            raise ValueError("extraction_config.language must be a non-empty string")
        addon["language"] = language.strip()

    guidance = extraction_config.get("entity_types_guidance")
    entity_types = extraction_config.get("entity_types")
    if guidance is not None:
        if not isinstance(guidance, str) or not guidance.strip():
            raise ValueError(
                "extraction_config.entity_types_guidance must be a non-empty string"
            )
        addon["entity_types_guidance"] = guidance
    elif entity_types is not None:
        if not isinstance(entity_types, (list, tuple)) or not all(
            isinstance(item, str) and item.strip() for item in entity_types
        ):
            raise ValueError(
                "extraction_config.entity_types must be a list of non-empty strings"
            )
        if entity_types:
            normalized_types = list(dict.fromkeys(t.strip() for t in entity_types))
            addon["entity_types"] = normalized_types
            addon["entity_types_guidance"] = _entity_types_to_guidance(
                normalized_types
            )

    prompt_file = extraction_config.get("entity_type_prompt_file")
    if prompt_file is not None:
        if not isinstance(prompt_file, str) or not prompt_file.strip():
            raise ValueError(
                "extraction_config.entity_type_prompt_file must be a non-empty string"
            )
        addon["entity_type_prompt_file"] = prompt_file.strip()

    for source_key, target_key, allow_zero in (
        ("max_gleaning", "entity_extract_max_gleaning", True),
        ("max_extraction_records", "entity_extract_max_records", False),
        ("max_extraction_entities", "entity_extract_max_entities", False),
        ("force_llm_summary_on_merge", "force_llm_summary_on_merge", True),
    ):
        raw = extraction_config.get(source_key)
        if raw is None:
            continue
        value = _non_negative_int(raw) if allow_zero else _positive_int(raw)
        if value is not None:
            kwargs[target_key] = value

    return {"addon": addon, "kwargs": kwargs}


def _active_llm_role_runtime_config(
    config: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Resolve ``llm_role_config`` into per-role runtime override maps.

    Accepts two shapes per role for convenience:

    - a bare string, treated as ``{"model": <str>}``;
    - a mapping with any of ``model`` / ``binding`` / ``host`` / ``api_key`` /
      ``provider_options`` / ``model_kwargs`` (alias ``kwargs``) /
      ``max_async`` / ``timeout``.

    Returns ``{role: {normalized override kwargs}}`` for recognized roles
    only. Unknown roles or keys raise ``ValueError`` so a bad config is
    rejected at create time rather than silently ignored at runtime.
    """
    role_config = _raw_config_section(config, "llm_role_config")
    runtime: dict[str, dict[str, Any]] = {}
    if not role_config:
        return runtime
    for role, raw in role_config.items():
        if not isinstance(role, str) or role.strip().lower() not in ROLE_NAMES:
            raise ValueError(f"llm_role_config has unknown role: {role!r}")
        normalized_role = role.strip().lower()
        if isinstance(raw, str):
            if not raw.strip():
                raise ValueError(
                    f"llm_role_config[{normalized_role!r}] model must be non-empty"
                )
            runtime[normalized_role] = {"model": raw.strip()}
            continue
        if not isinstance(raw, dict):
            raise ValueError(
                f"llm_role_config[{normalized_role!r}] must be a string or object"
            )
        override: dict[str, Any] = {}
        for key, value in raw.items():
            if key not in _LLM_ROLE_CONFIG_KEYS:
                raise ValueError(
                    f"llm_role_config[{normalized_role!r}] has unknown key: {key!r}"
                )
            if value is None:
                continue
            if key in {"model", "binding", "host"}:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"llm_role_config[{normalized_role!r}].{key} must be a non-empty string"
                    )
                override[key] = value.strip()
            elif key == "api_key":
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"llm_role_config[{normalized_role!r}].api_key must be a string"
                    )
                override["api_key"] = value
            elif key in {"provider_options", "model_kwargs", "kwargs"}:
                if not isinstance(value, dict):
                    raise ValueError(
                        f"llm_role_config[{normalized_role!r}].{key} must be an object"
                    )
                target_key = "model_kwargs" if key in {"model_kwargs", "kwargs"} else key
                override[target_key] = dict(value)
            else:  # max_async / timeout
                parsed = _positive_int(value)
                if parsed is not None:
                    override[key] = parsed
        if override:
            runtime[normalized_role] = override
    return runtime


def active_llm_role_runtime_config_from_version(
    active_config_version: ConfigVersionRecord | None,
) -> dict[str, dict[str, Any]]:
    """Public accessor: per-role LLM overrides for a config version (or ``{}``)."""
    if active_config_version is None:
        return {}
    return _active_llm_role_runtime_config(active_config_version.config)


def _llm_role_identity(
    role_runtime: dict[str, dict[str, Any]], roles
) -> dict[str, Any]:
    """Output-affecting identity for the given roles, used in hashing.

    Excludes ``api_key`` (secret) and ``max_async`` / ``timeout`` (perf only)
    so rotating a key or tuning concurrency never forces a needless rebuild.
    """
    identity: dict[str, Any] = {}
    for role in roles:
        override = role_runtime.get(role)
        if not override:
            continue
        role_identity = {
            key: override[key]
            for key in _LLM_ROLE_IDENTITY_KEYS
            if key in override
        }
        if role_identity:
            identity[role] = role_identity
    return identity


def _active_index_runtime_hash(config: dict[str, Any] | None) -> str:
    extraction = _active_extraction_runtime_config(config)
    role_runtime = _active_llm_role_runtime_config(config)
    return _section_hash(
        "index",
        {
            "chunk": _active_chunk_runtime_config(config),
            "embedding": _active_embedding_runtime_config(config),
            "extraction": {**extraction["addon"], **extraction["kwargs"]},
            "llm_roles": _llm_role_identity(role_runtime, _INDEX_AFFECTING_ROLES),
        },
    )


def _active_query_runtime_hash(config: dict[str, Any] | None) -> str:
    role_runtime = _active_llm_role_runtime_config(config)
    return _section_hash(
        "query",
        {
            "query": _active_query_runtime_config(config),
            "llm_roles": _llm_role_identity(role_runtime, _QUERY_AFFECTING_ROLES),
        },
    )


def apply_active_config_to_lightrag_kwargs(
    kwargs: dict[str, Any], active_config_version: ConfigVersionRecord | None
) -> dict[str, Any]:
    """Overlay supported active KB config fields onto LightRAG constructor kwargs.

    This helper intentionally maps only fields that already have stable
    LightRAG constructor/runtime equivalents. Unsupported config sections stay
    persisted for diff/audit purposes until their runtime integration is added.
    """
    if active_config_version is None:
        return kwargs

    chunk_runtime = _active_chunk_runtime_config(active_config_version.config)
    kwargs.update(chunk_runtime)

    query_runtime = _active_query_runtime_config(active_config_version.config)
    for key in (
        "top_k",
        "chunk_top_k",
        "max_entity_tokens",
        "max_relation_tokens",
        "max_total_tokens",
        "related_chunk_number",
    ):
        value = _positive_int(query_runtime.get(key))
        if value is not None:
            kwargs[key] = value
    if query_runtime.get("cosine_threshold") is not None:
        cosine_threshold = float(query_runtime["cosine_threshold"])
        kwargs["cosine_threshold"] = cosine_threshold
        kwargs["cosine_better_than_threshold"] = cosine_threshold
        vector_kwargs = dict(kwargs.get("vector_db_storage_cls_kwargs") or {})
        vector_kwargs["cosine_better_than_threshold"] = cosine_threshold
        kwargs["vector_db_storage_cls_kwargs"] = vector_kwargs

    embedding_runtime = _active_embedding_runtime_config(active_config_version.config)
    embedding_func = kwargs.get("embedding_func")
    if isinstance(embedding_func, EmbeddingFunc) and embedding_runtime:
        embedding_updates: dict[str, Any] = {}
        for source_key, target_key in (
            ("embedding_dim", "embedding_dim"),
            ("max_token_size", "max_token_size"),
            ("model_name", "model_name"),
        ):
            if source_key in embedding_runtime:
                embedding_updates[target_key] = embedding_runtime[source_key]
        if embedding_updates:
            kwargs["embedding_func"] = replace(embedding_func, **embedding_updates)

    extraction_runtime = _active_extraction_runtime_config(active_config_version.config)
    if extraction_runtime["addon"]:
        merged_addon = dict(kwargs.get("addon_params") or {})
        merged_addon.update(extraction_runtime["addon"])
        kwargs["addon_params"] = merged_addon
    kwargs.update(extraction_runtime["kwargs"])

    return kwargs


def attach_active_config_metadata(
    rag: Any, active_config_version: ConfigVersionRecord | None
) -> None:
    if active_config_version is None:
        return
    setattr(rag, "kb_active_config_version_id", active_config_version.id)
    setattr(rag, "kb_active_config", active_config_version.config)
    setattr(rag, "kb_active_parser_hash", active_config_version.parser_hash)
    setattr(rag, "kb_active_index_hash", _active_index_runtime_hash(active_config_version.config))
    setattr(rag, "kb_active_query_hash", _active_query_runtime_hash(active_config_version.config))
    setattr(
        rag,
        "kb_active_parser_config",
        _active_parser_runtime_config(active_config_version.config),
    )
    setattr(
        rag,
        "kb_active_query_config",
        _active_query_runtime_config(active_config_version.config),
    )


def active_query_defaults_from_rag(rag: Any) -> dict[str, Any]:
    query_config = getattr(rag, "kb_active_query_config", None)
    if not isinstance(query_config, dict):
        return {}
    return {
        key: value
        for key, value in query_config.items()
        if key in _ACTIVE_QUERY_CONFIG_KEYS and value is not None
    }


def active_query_metadata_from_rag(rag: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for attr, key in (
        ("kb_active_config_version_id", "config_version_id"),
        ("kb_active_parser_hash", "parser_hash"),
        ("kb_active_index_hash", "index_hash"),
        ("kb_active_query_hash", "query_hash"),
    ):
        value = getattr(rag, attr, None)
        if value is not None:
            metadata[key] = value
    return metadata


class ConfigVersionService:
    """Persist and activate KB-scoped configuration versions.

    A config version is a frozen snapshot of parser / chunk / embedding /
    extraction / query / storage settings, plus three derived hashes
    (``parser_hash``, ``index_hash``, ``query_hash``). Activation simply
    points the KB record at the new ``active_config_version_id`` and
    discards the cached LightRAG instance so the next request rebuilds it.
    """

    def __init__(
        self,
        kb_service: KnowledgeBaseService,
        metadata_store: SQLiteMetadataStore,
        registry: LightRAGInstanceRegistry,
    ):
        self._kb_service = kb_service
        self._metadata_store = metadata_store
        self._registry = registry

    async def create(
        self,
        kb_id: str,
        *,
        config: dict[str, Any],
        created_by: str | None = None,
    ) -> ConfigVersionRecord:
        record = await self._kb_service.get(kb_id)
        derived = self._derive_hashes(config)
        version_record = ConfigVersionRecord(
            id=generate_track_id("cfg"),
            kb_id=record.id,
            workspace=record.workspace,
            version=0,
            config=config,
            parser_hash=derived["parser_hash"],
            index_hash=derived["index_hash"],
            query_hash=derived["query_hash"],
            created_at=utc_now_iso(),
            activated_at=None,
            created_by=created_by,
        )
        return await self._metadata_store.create_config_version(version_record)

    async def list(
        self, kb_id: str, *, limit: int = 50, offset: int = 0
    ) -> tuple[list[ConfigVersionRecord], int]:
        record = await self._kb_service.get(kb_id)
        return await self._metadata_store.list_config_versions(
            record.id, limit=limit, offset=offset
        )

    async def get(self, kb_id: str, version_id: str) -> ConfigVersionRecord:
        record = await self._kb_service.get(kb_id)
        return await self._metadata_store.get_config_version(record.id, version_id)

    async def activate(self, kb_id: str, version_id: str) -> ConfigVersionRecord:
        record = await self._kb_service.get(kb_id)
        version = await self._metadata_store.get_config_version(record.id, version_id)
        await self._kb_service.update(
            record.id, active_config_version_id=version.id
        )
        marked = await self._metadata_store.mark_config_version_activated(
            record.id, version.id
        )
        # Drop cached instance so the next request rebuilds with the new
        # config version. If a destructive job is in flight, leave it alone.
        try:
            await self._registry.discard(record.id)
        except DestructiveLockBusyError:
            pass
        return marked

    async def diff(
        self, kb_id: str, version_id: str
    ) -> dict[str, Any]:
        """Compare the target version against the currently active one.

        Returns rebuild requirements (``requires_reparse`` /
        ``requires_reindex`` / ``requires_vector_rebuild``) plus a list of
        changed config sections — enough for clients to preview the impact
        before activation.
        """
        record = await self._kb_service.get(kb_id)
        target = await self._metadata_store.get_config_version(record.id, version_id)
        if not record.active_config_version_id:
            return {
                "target_version_id": target.id,
                "active_version_id": None,
                "requires_reparse": True,
                "requires_reindex": True,
                "requires_vector_rebuild": True,
                "reasons": ["no_active_version"],
            }
        active = await self._metadata_store.get_config_version(
            record.id, record.active_config_version_id
        )
        reasons: list[str] = []
        requires_reparse = active.parser_hash != target.parser_hash
        requires_reindex = (
            requires_reparse or active.index_hash != target.index_hash
        )
        requires_vector_rebuild = self._embedding_changed(
            active.config, target.config
        )
        if requires_reparse:
            reasons.append("parser_hash_changed")
        if active.index_hash != target.index_hash:
            reasons.append("index_hash_changed")
        if requires_vector_rebuild:
            reasons.append("embedding_changed")
        if active.query_hash != target.query_hash:
            reasons.append("query_hash_changed")
        return {
            "target_version_id": target.id,
            "active_version_id": active.id,
            "requires_reparse": requires_reparse,
            "requires_reindex": requires_reindex,
            "requires_vector_rebuild": requires_vector_rebuild,
            "reasons": reasons,
        }

    @staticmethod
    def _derive_hashes(config: dict[str, Any]) -> dict[str, str]:
        return {
            "parser_hash": _section_hash(
                "parser", _active_parser_runtime_config(config)
            ),
            "index_hash": _active_index_runtime_hash(config),
            "query_hash": _active_query_runtime_hash(config),
        }

    @staticmethod
    def _embedding_changed(
        active: dict[str, Any], target: dict[str, Any]
    ) -> bool:
        active_embed = _active_embedding_runtime_config(active)
        target_embed = _active_embedding_runtime_config(target)
        for key in ("model_name", "embedding_dim"):
            if active_embed.get(key) != target_embed.get(key):
                return True
        return False
