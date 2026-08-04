import asyncio
import os
import threading
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import IO, Any, Literal, final

from lightrag.base import (
    BaseKVStorage,
)
from lightrag.file_atomic import reap_orphan_tmp_files
from lightrag.utils import (
    _cooperative_yield,
    load_json,
    logger,
    write_json,
)
from lightrag.exceptions import StorageNotInitializedError

try:
    from lightrag.artifact_runtime import (
        PipelineAttemptCommitOutcomeUnknownError,
        PipelineAttemptRowKind,
        extract_pipeline_attempt_token,
    )
except ImportError:  # pragma: no cover - compatibility until core capability lands
    PipelineAttemptRowKind = Literal["full_docs", "doc_status"]

    class PipelineAttemptCommitOutcomeUnknownError(RuntimeError):
        """A JSON attempt-fenced commit may already be durable."""

        error_code = "pipeline_attempt_commit_outcome_unknown"

        def __init__(
            self,
            key: str,
            *,
            row_kind: PipelineAttemptRowKind,
            reason: str | None = None,
        ) -> None:
            self.key = key
            self.row_kind = row_kind
            self.reason = reason
            super().__init__(
                f"{self.error_code}: {row_kind} row {key!r} commit outcome is "
                f"unknown ({reason or 'json_durability_ambiguity'})"
            )

    def extract_pipeline_attempt_token(
        row: Mapping[str, Any] | None,
        *,
        row_kind: PipelineAttemptRowKind,
    ) -> str | None:
        if row_kind not in {"full_docs", "doc_status"}:
            raise ValueError(f"Unsupported pipeline attempt row kind: {row_kind!r}")
        if not isinstance(row, Mapping):
            return None
        if row_kind == "full_docs":
            container = row.get("artifact_binding")
            token_key = "claim_token"
        else:
            container = row.get("metadata")
            token_key = "pipeline_attempt_token"
        if not isinstance(container, Mapping):
            return None
        token = container.get(token_key)
        return token if isinstance(token, str) and token else None


from .shared_storage import (
    get_namespace_data,
    get_namespace_lock,
    get_data_init_lock,
    get_update_flag,
    set_all_update_flags,
    clear_all_update_flags,
    try_initialize_namespace,
)


_JSON_PROCESS_LOCKS: dict[str, threading.Lock] = {}
_JSON_PROCESS_LOCKS_GUARD = threading.Lock()


def _get_json_process_lock(lock_path: str) -> threading.Lock:
    with _JSON_PROCESS_LOCKS_GUARD:
        lock = _JSON_PROCESS_LOCKS.get(lock_path)
        if lock is None:
            lock = threading.Lock()
            _JSON_PROCESS_LOCKS[lock_path] = lock
        return lock


class _JsonWriterFileLock:
    """Serialize atomic JSON snapshot writers across processes and instances."""

    def __init__(self, file_name: str) -> None:
        self.lock_path = os.path.abspath(f"{file_name}.lock")
        self._process_lock = _get_json_process_lock(self.lock_path)
        self._file: IO[bytes] | None = None

    def __enter__(self) -> "_JsonWriterFileLock":
        self._process_lock.acquire()
        try:
            file = open(self.lock_path, "a+b")
            self._file = file
            if file.seek(0, os.SEEK_END) == 0:
                file.write(b"0")
                file.flush()
                os.fsync(file.fileno())
            file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(file.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        except BaseException:
            if self._file is not None:
                self._file.close()
                self._file = None
            self._process_lock.release()
            raise
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._file is None:
            self._process_lock.release()
            return
        try:
            self._file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None
            self._process_lock.release()


def _pipeline_attempt_unknown_error(
    key: str,
    *,
    row_kind: PipelineAttemptRowKind,
    error: BaseException,
) -> PipelineAttemptCommitOutcomeUnknownError:
    return PipelineAttemptCommitOutcomeUnknownError(
        key,
        row_kind=row_kind,
        reason=type(error).__name__,
    )


async def _compare_and_commit_json_pipeline_attempt(
    storage: Any,
    key: str,
    payload: Mapping[str, Any],
    *,
    expected_attempt_token: str,
    row_kind: PipelineAttemptRowKind,
    load_json_fn: Callable[[str], Any],
    write_json_fn: Callable[[Any, str], bool],
) -> bool:
    """Replace an attempt-owned row only when memory and disk still agree."""

    if not isinstance(key, str) or not key:
        raise ValueError("Pipeline attempt commit key must be a non-empty string")
    if row_kind not in {"full_docs", "doc_status"}:
        raise ValueError(f"Unsupported pipeline attempt row kind: {row_kind!r}")
    if not isinstance(expected_attempt_token, str) or not expected_attempt_token:
        raise ValueError("expected_attempt_token must be a non-empty string")
    if not isinstance(payload, Mapping):
        raise TypeError("Pipeline attempt commit payload must be a mapping")

    replacement = deepcopy(dict(payload))
    if "_id" in replacement and replacement["_id"] != key:
        raise ValueError("Pipeline attempt commit payload _id does not match key")
    if (
        extract_pipeline_attempt_token(replacement, row_kind=row_kind)
        != expected_attempt_token
    ):
        raise ValueError(
            f"{row_kind} payload does not carry the expected pipeline attempt token"
        )
    if storage._storage_lock is None or storage._data is None:
        raise StorageNotInitializedError(type(storage).__name__)

    durable_commit_visible = False
    try:
        async with storage._storage_lock:
            shared_current = storage._data.get(key)
            if (
                extract_pipeline_attempt_token(shared_current, row_kind=row_kind)
                != expected_attempt_token
            ):
                return False

            with _JsonWriterFileLock(storage._file_name):
                persisted = load_json_fn(storage._file_name)
                if persisted is None:
                    return False
                if not isinstance(persisted, dict):
                    raise TypeError(
                        f"JSON storage snapshot must be an object: {storage._file_name}"
                    )

                durable_current = persisted.get(key)
                if (
                    extract_pipeline_attempt_token(durable_current, row_kind=row_kind)
                    != expected_attempt_token
                ):
                    return False

                candidate = deepcopy(persisted)
                candidate[key] = replacement
                committed_snapshot = candidate
                try:
                    write_json_fn(candidate, storage._file_name)
                except asyncio.CancelledError:
                    raise
                except Exception as write_error:
                    try:
                        observed = load_json_fn(storage._file_name)
                    except asyncio.CancelledError:
                        raise
                    except Exception as read_error:
                        raise _pipeline_attempt_unknown_error(
                            key,
                            row_kind=row_kind,
                            error=write_error,
                        ) from read_error

                    if observed == candidate:
                        committed_snapshot = observed
                    elif observed == persisted:
                        raise
                    elif (
                        extract_pipeline_attempt_token(
                            observed.get(key)
                            if isinstance(observed, Mapping)
                            else None,
                            row_kind=row_kind,
                        )
                        != expected_attempt_token
                    ):
                        return False
                    else:
                        raise _pipeline_attempt_unknown_error(
                            key,
                            row_kind=row_kind,
                            error=write_error,
                        ) from write_error
                else:
                    durable_commit_visible = True
                    try:
                        committed_snapshot = load_json_fn(storage._file_name)
                    except asyncio.CancelledError:
                        raise
                    except Exception as read_error:
                        raise _pipeline_attempt_unknown_error(
                            key,
                            row_kind=row_kind,
                            error=read_error,
                        ) from read_error

                durable_commit_visible = True
                if committed_snapshot != candidate:
                    error = RuntimeError(
                        "durable JSON snapshot differs from the committed candidate"
                    )
                    raise _pipeline_attempt_unknown_error(
                        key,
                        row_kind=row_kind,
                        error=error,
                    ) from error
                committed_row = committed_snapshot.get(key)
                if not isinstance(committed_row, dict):
                    error = RuntimeError(
                        "durable JSON snapshot omitted the committed row"
                    )
                    raise _pipeline_attempt_unknown_error(
                        key,
                        row_kind=row_kind,
                        error=error,
                    ) from error
                storage._data[key] = deepcopy(committed_row)
        return True
    except PipelineAttemptCommitOutcomeUnknownError:
        raise
    except asyncio.CancelledError:
        raise
    except Exception as error:
        if durable_commit_visible:
            raise _pipeline_attempt_unknown_error(
                key,
                row_kind=row_kind,
                error=error,
            ) from error
        raise


@final
@dataclass
class JsonKVStorage(BaseKVStorage):
    """JSON-file-backed KV storage with **shared in-memory state across processes**.

    This class uses a *fundamentally different* cross-process model from
    ``NanoVectorDBStorage`` / ``FaissVectorDBStorage`` / ``NetworkXStorage``
    (which keep one in-memory copy per process and reconcile via file
    reloads). Compare carefully before changing either side.

    Storage model:
        ``self._data`` is **not** a per-process dict — it is the value
        returned by ``get_namespace_data(namespace, workspace=...)``, i.e.
        a reference into ``shared_storage._shared_dicts``. In multi-
        process mode this is a ``multiprocessing.Manager().dict()`` proxy
        that every worker sees the **same instance** of; in single-
        process mode it degrades to a plain ``dict``. Either way, a
        mutation in any process is *immediately* visible to every other
        process — there is no reload needed.

        The on-disk file at
        ``working_dir/[workspace/]kv_store_<namespace>.json`` exists for
        durability only. It is the source of truth at startup and the
        target of ``index_done_callback`` flushes, but is **not** part of
        the steady-state read/write path.

    First-time load (``initialize``):
        ``try_initialize_namespace`` is a global init lock that returns
        ``True`` to exactly one process per ``(namespace, workspace)``.
        That process reads the JSON file and populates ``self._data``
        under ``_storage_lock``. Other processes skip the load — they
        will see the data through the same shared ``self._data`` proxy.

    Cross-process sync protocol (note: reversed semantics vs file-backed
    classes):
        Anyone writing (``upsert`` / ``delete`` / ``drop``):
            1. Mutate ``self._data`` under ``_storage_lock`` (same lock,
               same dict, all processes see the change immediately).
            2. Call ``set_all_update_flags`` to mark **every** process's
               ``storage_updated`` flag ``True``. Here ``True`` means
               *"there is dirty data that still needs to be flushed"*,
               not *"there is fresher data on disk that I need to
               reload"* as in the file-backed implementations.
        Commit (``index_done_callback``):
            1. Under ``_storage_lock``, if ``storage_updated.value`` is
               ``True``, snapshot ``self._data`` and write it to disk
               via ``write_json`` (atomic).
            2. ``clear_all_update_flags`` — wipe every process's flag
               back to ``False``. Because the in-memory state is already
               consistent across processes, there is nothing for the
               *other* processes to do; the clear is just a
               "the dirty data has been persisted" signal.

    Lock scope:
        ``_storage_lock`` is a per-``(namespace, workspace)`` keyed lock
        spanning intra-process coroutines **and** inter-process workers.
        Unlike the file-backed classes (which only lock reload/commit
        critical sections), this class **holds the lock over every
        ``self._data`` access** — read or write — because the underlying
        ``Manager().dict()`` is not free-threaded across processes.

        Two places intentionally do work outside the lock for latency
        reasons:
            * ``upsert`` performs its per-key timestamp prep loop inside
              the lock but yields to the event loop via
              ``_cooperative_yield`` between keys (safe: ``NamespaceLock``
              is non-reentrant, so siblings blocked on it stay blocked).
            * ``JsonDocStatusStorage.upsert`` prepares its caller-supplied
              dict outside the lock (it only mutates the input, not the
              shared store).

    Who can write:
        Pipeline ``busy`` still serializes the document ingest / purge
        flows, but the *file-flush trigger* is symmetric: any process
        whose ``storage_updated.value`` is ``True`` when
        ``index_done_callback`` fires will perform the write. In a
        single-writer pipeline this is always the same process; if you
        ever permit multiple writers, two processes may race to flush
        the same in-memory state — that race is safe (both flush the
        same shared dict, ``write_json`` is atomic per file) but
        wasteful, and the ``clear_all_update_flags`` after each flush
        means subsequent re-flushes are no-ops.

    Caveats vs file-backed implementations:
        * **No reload path.** If something writes to the on-disk file
          out of band, this class will not pick it up until restart.
          The file is only ever written by ``index_done_callback`` and
          read once in ``initialize``.
        * **No ``_get_*`` entry method.** Adding one would be wrong —
          there's nothing to "get fresher than" since the in-memory
          state is already the shared, authoritative view.
        * **``write_json`` may sanitize.** If sanitization happens, the
          on-disk JSON differs from what was in memory; the callback
          re-reads the cleaned file back into ``self._data`` under the
          same lock so the shared view stays consistent with disk.

    Non-pipeline write paths:
        * ``drop`` — destructive, **not** serialized by this storage
          class. Currently gated by the API layer
          (``/documents/clear``); any new caller must hold the pipeline
          ``busy`` reservation.
        * ``upsert`` / ``delete`` invoked from non-pipeline admin flows
          (cache management, etc.) — safe under the shared-lock model,
          but consumers should still respect the pipeline gate to avoid
          interleaving with batched ingest work.
    """

    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        if self.workspace:
            # Include workspace in the file path for data isolation
            workspace_dir = os.path.join(working_dir, self.workspace)
        else:
            # Default behavior when workspace is empty
            workspace_dir = working_dir
            self.workspace = ""

        os.makedirs(workspace_dir, exist_ok=True)
        self._file_name = os.path.join(workspace_dir, f"kv_store_{self.namespace}.json")
        self._data = None
        self._storage_lock = None
        self.storage_updated = None

        reap_orphan_tmp_files(self._file_name, self.workspace or "_")

    async def initialize(self):
        """Bind to the shared namespace dict and load from disk on first init.

        ``try_initialize_namespace`` is a global init lock that returns
        ``True`` for exactly one process per ``(namespace, workspace)``;
        that process reads the JSON file and populates the shared
        ``self._data`` under ``_storage_lock``. Subsequent processes
        skip the file read — they will see the same shared dict via
        ``get_namespace_data``.

        For ``*_cache`` namespaces an extra
        ``_migrate_legacy_cache_structure`` pass runs against the loaded
        data and may rewrite the on-disk file if a migration was applied.
        """
        self._storage_lock = get_namespace_lock(
            self.namespace, workspace=self.workspace
        )
        self.storage_updated = await get_update_flag(
            self.namespace, workspace=self.workspace
        )
        async with get_data_init_lock():
            # check need_init must before get_namespace_data
            need_init = await try_initialize_namespace(
                self.namespace, workspace=self.workspace
            )
            self._data = await get_namespace_data(
                self.namespace, workspace=self.workspace
            )
            if need_init:
                loaded_data = load_json(self._file_name) or {}
                async with self._storage_lock:
                    # Migrate legacy cache structure if needed
                    if self.namespace.endswith("_cache"):
                        loaded_data = await self._migrate_legacy_cache_structure(
                            loaded_data
                        )

                    self._data.update(loaded_data)
                    data_count = len(loaded_data)

                    logger.info(
                        f"[{self.workspace}] Process {os.getpid()} KV load {self.namespace} with {data_count} records"
                    )

    async def index_done_callback(self) -> None:
        """Flush dirty in-memory state to disk and clear all dirty flags.

        Commit point in the shared-memory protocol (see class docstring,
        *Cross-process sync protocol*). Steps:
            1. Under ``_storage_lock``, check this process's
               ``storage_updated.value``. If ``False``, nothing to do —
               return.
            2. Snapshot ``self._data`` (converting from ``Manager.dict``
               proxy to a plain ``dict`` so the JSON encoder doesn't trip
               over the proxy) and write it via ``write_json``.
            3. If ``write_json`` reports sanitization was applied, the
               on-disk file no longer matches what was in memory — reload
               the cleaned data back into ``self._data`` under the same
               lock so the shared view stays consistent.
            4. ``clear_all_update_flags`` — wipe every process's
               ``storage_updated`` flag back to ``False``, signaling
               that the dirty data has been persisted.

        Note the **semantic difference** from the file-backed classes'
        commit: there is no ``set_all_update_flags`` here. The shared
        dict is already consistent across processes; the only thing
        ``index_done_callback`` does globally is *clear* the dirty
        flags.
        """
        async with self._storage_lock:
            if self.storage_updated.value:
                data_dict = (
                    dict(self._data) if hasattr(self._data, "_getvalue") else self._data
                )

                # Calculate data count - all data is now flattened
                data_count = len(data_dict)

                logger.debug(
                    f"[{self.workspace}] Process {os.getpid()} KV writting {data_count} records to {self.namespace}"
                )

                with _JsonWriterFileLock(self._file_name):
                    # Write JSON and check if sanitization was applied
                    needs_reload = write_json(data_dict, self._file_name)

                    # If data was sanitized, reload cleaned data to update shared memory
                    if needs_reload:
                        logger.info(
                            f"[{self.workspace}] Reloading sanitized data into shared memory for {self.namespace}"
                        )
                        cleaned_data = load_json(self._file_name)
                        if cleaned_data is not None:
                            self._data.clear()
                            self._data.update(cleaned_data)

                await clear_all_update_flags(self.namespace, workspace=self.workspace)

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        async with self._storage_lock:
            result = self._data.get(id)
            if result:
                # Create a copy to avoid modifying the original data
                result = dict(result)
                # Ensure time fields are present, provide default values for old data
                result.setdefault("create_time", 0)
                result.setdefault("update_time", 0)
                # Ensure _id field contains the clean ID
                result["_id"] = id
            return result

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        async with self._storage_lock:
            results = []
            for id in ids:
                data = self._data.get(id, None)
                if data:
                    # Create a copy to avoid modifying the original data
                    result = {k: v for k, v in data.items()}
                    # Ensure time fields are present, provide default values for old data
                    result.setdefault("create_time", 0)
                    result.setdefault("update_time", 0)
                    # Ensure _id field contains the clean ID
                    result["_id"] = id
                    results.append(result)
                else:
                    results.append(None)
            return results

    async def filter_keys(self, keys: set[str]) -> set[str]:
        async with self._storage_lock:
            return set(keys) - set(self._data.keys())

    async def compare_and_commit_pipeline_attempt(
        self,
        key: str,
        payload: Mapping[str, Any],
        *,
        expected_attempt_token: str,
        row_kind: PipelineAttemptRowKind,
    ) -> bool:
        """Atomically replace and durably publish an attempt-owned JSON row."""

        return await _compare_and_commit_json_pipeline_attempt(
            self,
            key,
            payload,
            expected_attempt_token=expected_attempt_token,
            row_kind=row_kind,
            load_json_fn=load_json,
            write_json_fn=write_json,
        )

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Insert or update KV records in shared memory; mark all processes dirty.

        Two side effects under ``_storage_lock``:
            1. Stamp ``create_time`` / ``update_time`` / ``_id`` on each
               value, then ``self._data.update(data)``. Because
               ``self._data`` is the shared ``Manager.dict()`` proxy, the
               update is visible to all processes immediately — no
               reload needed.
            2. ``set_all_update_flags`` — flip every process's
               ``storage_updated.value`` to ``True``. Here ``True``
               means *"there is dirty data that still needs to be
               flushed to disk"*, **not** *"there is fresher data on
               disk"* as in the file-backed classes (see class docstring
               for the contrast).

        Persistence is deferred to the next ``index_done_callback`` (the
        pipeline calls this via ``_insert_done()`` after each batch).

        Note: the per-key prep loop calls ``_cooperative_yield`` inside
        the lock. That is safe because ``NamespaceLock`` is non-
        reentrant — siblings waiting on this lock stay blocked across
        the yield; only unrelated coroutines benefit from the yield.
        """
        if not data:
            return

        import time

        current_time = int(time.time())  # Get current Unix timestamp

        logger.debug(
            f"[{self.workspace}] Inserting {len(data)} records to {self.namespace}"
        )
        if self._storage_lock is None:
            raise StorageNotInitializedError("JsonKVStorage")
        async with self._storage_lock:
            # Add timestamps to data based on whether key exists.
            # The loop reads self._data (k in self._data) so it must stay inside
            # the lock. _cooperative_yield is safe here: NamespaceLock is
            # non-reentrant, so other coroutines waiting on this lock will block
            # until we release it; the yield only benefits unrelated coroutines.
            for i, (k, v) in enumerate(data.items(), start=1):
                # For text_chunks namespace, ensure llm_cache_list field exists
                if self.namespace.endswith("text_chunks"):
                    if "llm_cache_list" not in v:
                        v["llm_cache_list"] = []

                # Add timestamps based on whether key exists
                if k in self._data:  # Key exists, only update update_time
                    v["update_time"] = current_time
                else:  # New key, set both create_time and update_time
                    v["create_time"] = current_time
                    v["update_time"] = current_time

                v["_id"] = k
                await _cooperative_yield(i)

            self._data.update(data)
            await set_all_update_flags(self.namespace, workspace=self.workspace)

    async def delete(self, ids: list[str]) -> None:
        """Remove records from shared memory; mark all processes dirty if any deleted.

        Under ``_storage_lock``: ``self._data.pop(doc_id, None)`` for
        each id. Only calls ``set_all_update_flags`` if at least one key
        was actually present (avoids creating spurious dirty state for
        no-op deletes).

        See class docstring for the shared-memory + dirty-flag protocol
        and the semantic contrast vs file-backed classes.

        Args:
            ids: List of document IDs to be deleted from storage
        """
        async with self._storage_lock:
            any_deleted = False
            for doc_id in ids:
                result = self._data.pop(doc_id, None)
                if result is not None:
                    any_deleted = True

            if any_deleted:
                await set_all_update_flags(self.namespace, workspace=self.workspace)

    async def is_empty(self) -> bool:
        """Check if the storage is empty

        Returns:
            bool: True if storage contains no data, False otherwise
        """
        async with self._storage_lock:
            return len(self._data) == 0

    async def drop(self) -> dict[str, str]:
        """Clear shared memory and immediately persist the empty state.

        This method will:
            1. Clear the shared ``self._data`` dict under
               ``_storage_lock`` (visible to all processes immediately).
            2. ``set_all_update_flags`` so every process knows there is
               dirty state pending persistence.
            3. Call ``index_done_callback`` synchronously to flush the
               empty state to disk and clear the dirty flags.

        Caller contract:
            ``drop`` is destructive and **not** serialized by this
            storage class. The caller must hold the pipeline ``busy``
            reservation (the ``/documents/clear`` endpoint does this)
            before invoking it — running ``drop`` concurrently with an
            active document pipeline will wipe out in-flight work and
            silently lose data. See class docstring,
            *Non-pipeline write paths*.

        Returns:
            dict[str, str]: Operation status and message
            - On success: {"status": "success", "message": "data dropped"}
            - On failure: {"status": "error", "message": "<error details>"}
        """
        try:
            async with self._storage_lock:
                self._data.clear()
                await set_all_update_flags(self.namespace, workspace=self.workspace)

            await self.index_done_callback()
            logger.info(
                f"[{self.workspace}] Process {os.getpid()} drop {self.namespace}"
            )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error dropping {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}

    async def _migrate_legacy_cache_structure(self, data: dict) -> dict:
        """Migrate legacy nested cache structure to flattened structure

        Args:
            data: Original data dictionary that may contain legacy structure

        Returns:
            Migrated data dictionary with flattened cache keys (sanitized if needed)
        """
        from lightrag.utils import generate_cache_key

        # Early return if data is empty
        if not data:
            return data

        # Check first entry to see if it's already in new format
        first_key = next(iter(data.keys()))
        if ":" in first_key and len(first_key.split(":")) == 3:
            # Already in flattened format, return as-is
            return data

        migrated_data = {}
        migration_count = 0

        for key, value in data.items():
            # Check if this is a legacy nested cache structure
            if isinstance(value, dict) and all(
                isinstance(v, dict) and "return" in v for v in value.values()
            ):
                # This looks like a legacy cache mode with nested structure
                mode = key
                for cache_hash, cache_entry in value.items():
                    cache_type = cache_entry.get("cache_type", "extract")
                    flattened_key = generate_cache_key(mode, cache_type, cache_hash)
                    migrated_data[flattened_key] = cache_entry
                    migration_count += 1
            else:
                # Keep non-cache data or already flattened cache data as-is
                migrated_data[key] = value

        if migration_count > 0:
            logger.info(
                f"[{self.workspace}] Migrated {migration_count} legacy cache entries to flattened structure"
            )
            # Persist migrated data immediately and check if sanitization was applied
            needs_reload = write_json(migrated_data, self._file_name)

            # If data was sanitized during write, reload cleaned data
            if needs_reload:
                logger.info(
                    f"[{self.workspace}] Reloading sanitized migration data for {self.namespace}"
                )
                cleaned_data = load_json(self._file_name)
                if cleaned_data is not None:
                    return cleaned_data  # Return cleaned data to update shared memory

        return migrated_data

    async def finalize(self):
        """On shutdown, flush ``*_cache`` namespaces to disk.

        Cache namespaces are routinely written to during query/extract
        without triggering an immediate ``index_done_callback`` (caches
        churn fast and the pipeline doesn't always end at a natural
        commit point). This hook ensures whatever dirty cache state is
        in shared memory at process exit gets persisted, so the next
        run can pick it up.

        Non-cache namespaces don't need this — their writes already
        flow through pipeline-driven ``_insert_done()`` commits.
        """
        if self.namespace.endswith("_cache"):
            await self.index_done_callback()
