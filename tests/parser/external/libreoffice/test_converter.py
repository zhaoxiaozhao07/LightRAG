from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from lightrag.parser.external.libreoffice import LibreOfficeConverter


class _SubprocessRecorder:
    def __init__(self, *, fail_convert: bool = False) -> None:
        self.fail_convert = fail_convert
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str], **_: Any) -> subprocess.CompletedProcess:
        self.calls.append(list(args))
        if args[1:] == ["--version"]:
            return subprocess.CompletedProcess(
                args, 0, stdout="LibreOffice 24.2.0\n", stderr=""
            )

        if "--convert-to" not in args:
            raise AssertionError(f"unexpected subprocess command: {args!r}")
        if self.fail_convert:
            return subprocess.CompletedProcess(
                args, 7, stdout="", stderr="synthetic conversion failure"
            )

        outdir = Path(args[args.index("--outdir") + 1])
        target_ext = args[args.index("--convert-to") + 1]
        source_copy = Path(args[-1])
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / f"{source_copy.stem}.{target_ext}").write_bytes(
            b"converted:" + source_copy.read_bytes()
        )
        return subprocess.CompletedProcess(args, 0, stdout="convert ok", stderr="")

    @property
    def convert_calls(self) -> list[list[str]]:
        return [c for c in self.calls if "--convert-to" in c]


@pytest.fixture(autouse=True)
def _clear_libreoffice_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ENABLE_LIBREOFFICE_CONVERSION",
        "LIBREOFFICE_EXECUTABLE",
        "LIBREOFFICE_TIMEOUT_SECONDS",
        "LIBREOFFICE_FORCE_RECONVERT",
        "LIBREOFFICE_EXTRA_ARGS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_disabled_conversion_raises_clear_error(tmp_path: Path) -> None:
    source = tmp_path / "demo.doc"
    source.write_bytes(b"legacy doc")

    with pytest.raises(RuntimeError, match="ENABLE_LIBREOFFICE_CONVERSION=true"):
        asyncio.run(
            LibreOfficeConverter().convert_for_docling(
                tmp_path / "demo.doc.libreoffice_raw",
                source,
                document_name="demo.doc",
            )
        )


def test_conversion_command_manifest_cache_and_force_reconvert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENABLE_LIBREOFFICE_CONVERSION", "true")
    monkeypatch.setenv("LIBREOFFICE_EXECUTABLE", "soffice-test")
    monkeypatch.setenv("LIBREOFFICE_TIMEOUT_SECONDS", "9")
    monkeypatch.setenv("LIBREOFFICE_EXTRA_ARGS", "--safe-mode --foo=bar")

    recorder = _SubprocessRecorder()
    monkeypatch.setattr(
        "lightrag.parser.external.libreoffice.converter.subprocess.run", recorder
    )

    # Source filename contains a parser hint; the converter must copy it into
    # the temp input dir as the canonical document_name before invoking soffice.
    source = tmp_path / "demo.[docling].doc"
    source.write_bytes(b"legacy doc payload")
    raw_dir = tmp_path / "demo.doc.libreoffice_raw"

    result = asyncio.run(
        LibreOfficeConverter().convert_for_docling(
            raw_dir, source, document_name="demo.doc"
        )
    )

    assert result.upload_filename == "demo.docx"
    assert result.converted_path == raw_dir / "output" / "demo.docx"
    assert result.converted_path.read_bytes() == b"converted:legacy doc payload"
    assert result.cache_hit is False

    assert len(recorder.convert_calls) == 1
    command = recorder.convert_calls[0]
    assert command[0] == "soffice-test"
    assert "--headless" in command
    assert "--norestore" in command
    assert "--nofirststartwizard" in command
    assert "--nolockcheck" in command
    assert any(arg.startswith("-env:UserInstallation=file://") for arg in command)
    assert "--safe-mode" in command
    assert "--foo=bar" in command
    assert command[command.index("--convert-to") + 1] == "docx"
    assert Path(command[-1]).name == "demo.doc"

    manifest = json.loads((raw_dir / "_manifest.json").read_text(encoding="utf-8"))
    assert manifest["engine"] == "libreoffice"
    assert manifest["engine_version"] == "LibreOffice 24.2.0"
    assert manifest["critical_file"]["path"] == "output/demo.docx"
    assert manifest["extras"] == {
        "source_suffix": ".doc",
        "target_suffix": ".docx",
        "target_name": "demo.docx",
    }

    cached = asyncio.run(
        LibreOfficeConverter().convert_for_docling(
            raw_dir, source, document_name="demo.doc"
        )
    )
    assert cached.cache_hit is True
    assert len(recorder.convert_calls) == 1, "cache hit must skip soffice conversion"

    monkeypatch.setenv("LIBREOFFICE_FORCE_RECONVERT", "true")
    forced = asyncio.run(
        LibreOfficeConverter().convert_for_docling(
            raw_dir, source, document_name="demo.doc"
        )
    )
    assert forced.cache_hit is False
    assert len(recorder.convert_calls) == 2


def test_conversion_failure_reports_subprocess_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENABLE_LIBREOFFICE_CONVERSION", "true")
    recorder = _SubprocessRecorder(fail_convert=True)
    monkeypatch.setattr(
        "lightrag.parser.external.libreoffice.converter.subprocess.run", recorder
    )

    source = tmp_path / "slides.ppt"
    source.write_bytes(b"legacy ppt")

    with pytest.raises(RuntimeError, match="LibreOffice conversion failed") as exc:
        asyncio.run(
            LibreOfficeConverter().convert_for_docling(
                tmp_path / "slides.ppt.libreoffice_raw",
                source,
                document_name="slides.ppt",
            )
        )
    assert "synthetic conversion failure" in str(exc.value)
