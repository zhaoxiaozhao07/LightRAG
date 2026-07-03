"""LibreOffice conversion adapter for legacy Microsoft Office files.

This adapter intentionally **only converts** documents.  Parsing remains the
responsibility of the downstream Docling adapter: ``.doc`` / ``.ppt`` /
``.xls`` files are converted to their OOXML equivalents, cached under a
``*.libreoffice_raw/`` sibling directory, and the converted artifact is then
uploaded to docling-serve.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from lightrag.constants import LIBREOFFICE_RAW_DIR_SUFFIX
from lightrag.parser.external._common import (
    clear_dir_contents,
    compute_size_and_hash,
    env_bool,
    env_int,
    raw_dir_for_parsed_dir as _raw_dir_for_parsed_dir,
)
from lightrag.parser.external._manifest import (
    Manifest,
    ManifestFile,
    load_manifest,
    write_manifest,
)
from lightrag.utils import logger

MANIFEST_ENGINE = "libreoffice"

DEFAULT_LIBREOFFICE_EXECUTABLE = "soffice"
DEFAULT_LIBREOFFICE_TIMEOUT_SECONDS = 120

LEGACY_OFFICE_SUFFIX_MAP: dict[str, str] = {
    ".doc": ".docx",
    ".ppt": ".pptx",
    ".xls": ".xlsx",
}

BASE_CONVERT_ARGS: tuple[str, ...] = (
    "--headless",
    "--norestore",
    "--nofirststartwizard",
    "--nolockcheck",
)


@dataclass(frozen=True)
class LibreOfficeConfig:
    """Effective LibreOffice conversion configuration from environment."""

    enabled: bool = False
    executable: str = DEFAULT_LIBREOFFICE_EXECUTABLE
    timeout_seconds: int = DEFAULT_LIBREOFFICE_TIMEOUT_SECONDS
    force_reconvert: bool = False
    extra_args: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "LibreOfficeConfig":
        timeout = env_int(
            "LIBREOFFICE_TIMEOUT_SECONDS", DEFAULT_LIBREOFFICE_TIMEOUT_SECONDS
        )
        if timeout <= 0:
            timeout = DEFAULT_LIBREOFFICE_TIMEOUT_SECONDS
        return cls(
            enabled=env_bool("ENABLE_LIBREOFFICE_CONVERSION", False),
            executable=(
                os.getenv("LIBREOFFICE_EXECUTABLE", DEFAULT_LIBREOFFICE_EXECUTABLE)
                .strip()
                or DEFAULT_LIBREOFFICE_EXECUTABLE
            ),
            timeout_seconds=timeout,
            force_reconvert=env_bool("LIBREOFFICE_FORCE_RECONVERT", False),
            extra_args=_parse_extra_args(os.getenv("LIBREOFFICE_EXTRA_ARGS", "")),
        )


@dataclass(frozen=True)
class LibreOfficeConversionResult:
    """Converted artifact information consumed by ``parse_docling``."""

    converted_path: Path
    upload_filename: str
    cache_hit: bool
    manifest: Manifest | None = None


def raw_dir_for_parsed_dir(parsed_dir: Path) -> Path:
    """``foo.parsed/`` → ``foo.libreoffice_raw/``."""

    return _raw_dir_for_parsed_dir(parsed_dir, suffix=LIBREOFFICE_RAW_DIR_SUFFIX)


def is_legacy_office_suffix(suffix: str) -> bool:
    """Return True for the legacy Office formats LibreOffice can bridge."""

    suffix = str(suffix or "").strip().lower()
    if suffix and not suffix.startswith("."):
        suffix = f".{suffix}"
    return suffix in LEGACY_OFFICE_SUFFIX_MAP


def is_legacy_office_file(path_or_name: str | Path) -> bool:
    return is_legacy_office_suffix(Path(path_or_name).suffix)


def target_suffix_for_source_suffix(source_suffix: str) -> str:
    suffix = str(source_suffix or "").strip().lower()
    if suffix and not suffix.startswith("."):
        suffix = f".{suffix}"
    try:
        return LEGACY_OFFICE_SUFFIX_MAP[suffix]
    except KeyError as exc:
        supported = ", ".join(sorted(LEGACY_OFFICE_SUFFIX_MAP))
        raise ValueError(
            f"LibreOffice conversion only supports {supported}; got {source_suffix!r}"
        ) from exc


def converted_upload_filename(document_name: str) -> str:
    """Return the canonical converted upload name (``demo.doc`` → ``demo.docx``)."""

    name = Path(document_name).name
    suffix = Path(name).suffix.lower()
    target_suffix = target_suffix_for_source_suffix(suffix)
    return Path(name).with_suffix(target_suffix).name


def compute_options_signature(
    config: LibreOfficeConfig,
    *,
    source_suffix: str,
    target_suffix: str,
) -> str:
    """Stable cache key for conversion options that affect output bytes."""

    payload = json.dumps(
        {
            "executable": config.executable,
            "timeout_seconds": int(config.timeout_seconds),
            "extra_args": list(config.extra_args),
            "source_suffix": source_suffix.lower(),
            "target_suffix": target_suffix.lower(),
            "fixed_args": list(BASE_CONVERT_ARGS),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_conversion_cache_valid(
    raw_dir: Path,
    source_file_path: Path,
    *,
    engine_version: str,
    options_signature: str,
    source_suffix: str,
    target_suffix: str,
    target_name: str,
) -> bool:
    """Return True iff the cached converted artifact matches source + options."""

    if not raw_dir.is_dir():
        return False

    manifest = load_manifest(raw_dir, expected_engine=MANIFEST_ENGINE)
    if manifest is None:
        return False

    try:
        cur_size = source_file_path.stat().st_size
    except OSError:
        return False
    if int(manifest.source_size_bytes) != cur_size:
        return False

    _, cur_hash = compute_size_and_hash(source_file_path)
    if manifest.source_content_hash != cur_hash:
        return False

    if manifest.engine_version != engine_version:
        return False
    if manifest.options_signature != options_signature:
        return False

    extras = manifest.extras or {}
    if extras.get("source_suffix") != source_suffix:
        return False
    if extras.get("target_suffix") != target_suffix:
        return False
    if extras.get("target_name") != target_name:
        return False

    crit = manifest.critical_file
    crit_path = raw_dir / crit.path
    try:
        if crit_path.stat().st_size != int(crit.size):
            return False
    except OSError:
        return False
    if crit.sha256:
        _, actual = compute_size_and_hash(crit_path)
        if actual != crit.sha256:
            return False

    for entry in manifest.files:
        p = raw_dir / entry.path
        try:
            if p.stat().st_size != int(entry.size):
                return False
        except OSError:
            return False

    return True


class LibreOfficeConverter:
    """Convert legacy Office files to OOXML using ``soffice``."""

    def __init__(self, config: LibreOfficeConfig | None = None) -> None:
        self.config = config or LibreOfficeConfig.from_env()

    async def convert_for_docling(
        self,
        raw_dir: Path,
        source_file_path: Path,
        *,
        document_name: str,
    ) -> LibreOfficeConversionResult:
        """Convert ``source_file_path`` for Docling and return the cached artifact.

        ``document_name`` must be the canonical, hint-stripped basename used by
        the pipeline.  It is copied into the temporary input directory before
        conversion so LibreOffice's output stem is stable and independent of
        filename parser hints or storage-specific source names.
        """

        source_file_path = Path(source_file_path)
        if not source_file_path.is_file():
            raise FileNotFoundError(
                f"LibreOffice conversion source file not found: {source_file_path}"
            )

        source_suffix = _resolve_source_suffix(document_name, source_file_path)
        target_suffix = target_suffix_for_source_suffix(source_suffix)
        target_name = Path(Path(document_name).name).with_suffix(target_suffix).name

        if not self.config.enabled:
            raise RuntimeError(
                f"LibreOffice conversion is required before Docling can parse "
                f"legacy Office file {document_name!r} ({source_suffix}). Set "
                f"ENABLE_LIBREOFFICE_CONVERSION=true, install LibreOffice on the "
                f"server, and configure LIBREOFFICE_EXECUTABLE if 'soffice' is not "
                f"on PATH."
            )

        return await asyncio.to_thread(
            self._convert_sync,
            Path(raw_dir),
            source_file_path,
            Path(document_name).name,
            source_suffix,
            target_suffix,
            target_name,
        )

    # ------------------------------------------------------------------
    # Synchronous implementation (always called via ``asyncio.to_thread``)
    # ------------------------------------------------------------------

    def _convert_sync(
        self,
        raw_dir: Path,
        source_file_path: Path,
        document_name: str,
        source_suffix: str,
        target_suffix: str,
        target_name: str,
    ) -> LibreOfficeConversionResult:
        engine_version = self._detect_version()
        options_signature = compute_options_signature(
            self.config,
            source_suffix=source_suffix,
            target_suffix=target_suffix,
        )

        converted_path = raw_dir / "output" / target_name
        if not self.config.force_reconvert and is_conversion_cache_valid(
            raw_dir,
            source_file_path,
            engine_version=engine_version,
            options_signature=options_signature,
            source_suffix=source_suffix,
            target_suffix=target_suffix,
            target_name=target_name,
        ):
            logger.info("[libreoffice] conversion cache hit: %s", converted_path)
            manifest = load_manifest(raw_dir, expected_engine=MANIFEST_ENGINE)
            return LibreOfficeConversionResult(
                converted_path=converted_path,
                upload_filename=target_name,
                cache_hit=True,
                manifest=manifest,
            )

        if self.config.force_reconvert and raw_dir.exists():
            logger.info(
                "[libreoffice] force reconvert requested; discarding cache at %s",
                raw_dir,
            )
        raw_dir.mkdir(parents=True, exist_ok=True)
        clear_dir_contents(raw_dir)
        output_dir = raw_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="lightrag-libreoffice-") as tmp:
            tmp_root = Path(tmp)
            input_dir = tmp_root / "input"
            tmp_output_dir = tmp_root / "output"
            profile_dir = tmp_root / "profile"
            input_dir.mkdir(parents=True, exist_ok=True)
            tmp_output_dir.mkdir(parents=True, exist_ok=True)
            profile_dir.mkdir(parents=True, exist_ok=True)

            input_copy = input_dir / document_name
            shutil.copyfile(source_file_path, input_copy)

            command = self._build_convert_command(
                input_copy=input_copy,
                outdir=tmp_output_dir,
                profile_dir=profile_dir,
                target_suffix=target_suffix,
            )
            try:
                proc = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_seconds,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"LibreOffice executable not found: {self.config.executable!r}. "
                    f"Set LIBREOFFICE_EXECUTABLE to the soffice binary."
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"LibreOffice conversion timed out after "
                    f"{self.config.timeout_seconds}s for {source_file_path}"
                ) from exc

            if proc.returncode != 0:
                raise RuntimeError(
                    "LibreOffice conversion failed for "
                    f"{source_file_path} -> {target_name} "
                    f"(exit {proc.returncode}). {_format_process_output(proc)}"
                )

            produced = tmp_output_dir / f"{Path(document_name).stem}{target_suffix}"
            if not produced.is_file():
                candidates = sorted(
                    p for p in tmp_output_dir.glob(f"*{target_suffix}") if p.is_file()
                )
                if len(candidates) == 1:
                    produced = candidates[0]
                else:
                    names = ", ".join(p.name for p in candidates) or "<none>"
                    raise RuntimeError(
                        "LibreOffice conversion did not produce expected output "
                        f"{produced.name!r}; candidates: {names}. "
                        f"{_format_process_output(proc)}"
                    )

            staging = output_dir / f".{target_name}.tmp"
            shutil.copyfile(produced, staging)
            os.replace(staging, converted_path)

        manifest = _build_and_write_manifest(
            raw_dir,
            source_file_path=source_file_path,
            document_name=document_name,
            converted_path=converted_path,
            engine_version=engine_version,
            options_signature=options_signature,
            source_suffix=source_suffix,
            target_suffix=target_suffix,
            target_name=target_name,
        )
        return LibreOfficeConversionResult(
            converted_path=converted_path,
            upload_filename=target_name,
            cache_hit=False,
            manifest=manifest,
        )

    def _detect_version(self) -> str:
        command = [self.config.executable, "--version"]
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=min(self.config.timeout_seconds, 30),
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"LibreOffice executable not found: {self.config.executable!r}. "
                f"Set LIBREOFFICE_EXECUTABLE to the soffice binary."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("LibreOffice version probe timed out") from exc

        if proc.returncode != 0:
            raise RuntimeError(
                "LibreOffice version probe failed "
                f"(exit {proc.returncode}). {_format_process_output(proc)}"
            )
        return _first_nonempty_line(proc.stdout, proc.stderr) or "unknown"

    def _build_convert_command(
        self,
        *,
        input_copy: Path,
        outdir: Path,
        profile_dir: Path,
        target_suffix: str,
    ) -> list[str]:
        return [
            self.config.executable,
            *BASE_CONVERT_ARGS,
            f"-env:UserInstallation={profile_dir.resolve().as_uri()}",
            *self.config.extra_args,
            "--convert-to",
            target_suffix.lstrip("."),
            "--outdir",
            str(outdir),
            str(input_copy),
        ]


def _build_and_write_manifest(
    raw_dir: Path,
    *,
    source_file_path: Path,
    document_name: str,
    converted_path: Path,
    engine_version: str,
    options_signature: str,
    source_suffix: str,
    target_suffix: str,
    target_name: str,
) -> Manifest:
    source_size, source_hash = compute_size_and_hash(source_file_path)
    crit_size, crit_hash = compute_size_and_hash(converted_path)
    critical = ManifestFile(
        path=converted_path.relative_to(raw_dir).as_posix(),
        size=crit_size,
        sha256=crit_hash,
    )
    manifest = Manifest(
        engine=MANIFEST_ENGINE,
        source_content_hash=source_hash,
        source_size_bytes=source_size,
        source_filename_at_parse=document_name,
        critical_file=critical,
        files=[],
        total_size_bytes=crit_size,
        engine_version=engine_version,
        options_signature=options_signature,
        downloaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        extras={
            "source_suffix": source_suffix,
            "target_suffix": target_suffix,
            "target_name": target_name,
        },
    )
    write_manifest(raw_dir, manifest)
    return manifest


def _resolve_source_suffix(document_name: str, source_file_path: Path) -> str:
    suffix = Path(document_name).suffix.lower() or source_file_path.suffix.lower()
    if suffix not in LEGACY_OFFICE_SUFFIX_MAP:
        supported = ", ".join(sorted(LEGACY_OFFICE_SUFFIX_MAP))
        raise ValueError(
            f"LibreOffice conversion supports only {supported}; got "
            f"document_name={document_name!r}, source={source_file_path}"
        )
    return suffix


def _parse_extra_args(raw: str) -> tuple[str, ...]:
    raw = (raw or "").strip()
    if not raw:
        return ()
    try:
        return tuple(shlex.split(raw, posix=os.name != "nt"))
    except ValueError as exc:
        raise ValueError(f"LIBREOFFICE_EXTRA_ARGS could not be parsed: {exc}") from exc


def _first_nonempty_line(*parts: str | None) -> str:
    for part in parts:
        for line in str(part or "").splitlines():
            line = line.strip()
            if line:
                return line
    return ""


def _format_process_output(proc: subprocess.CompletedProcess) -> str:
    stdout = " ".join(str(proc.stdout or "").split())
    stderr = " ".join(str(proc.stderr or "").split())
    if len(stdout) > 1000:
        stdout = f"{stdout[:1000]}...<truncated>"
    if len(stderr) > 1000:
        stderr = f"{stderr[:1000]}...<truncated>"
    return f"stdout={stdout!r} stderr={stderr!r}"


__all__ = [
    "BASE_CONVERT_ARGS",
    "DEFAULT_LIBREOFFICE_EXECUTABLE",
    "DEFAULT_LIBREOFFICE_TIMEOUT_SECONDS",
    "LEGACY_OFFICE_SUFFIX_MAP",
    "MANIFEST_ENGINE",
    "LibreOfficeConfig",
    "LibreOfficeConversionResult",
    "LibreOfficeConverter",
    "compute_options_signature",
    "converted_upload_filename",
    "is_conversion_cache_valid",
    "is_legacy_office_file",
    "is_legacy_office_suffix",
    "raw_dir_for_parsed_dir",
    "target_suffix_for_source_suffix",
]
