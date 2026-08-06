"""Regression test for single-process server log directory creation.

``configure_logging()`` must create the directory that actually holds
``lightrag.log`` (LOG_DIR), not the parent of LOG_DIR. When LOG_DIR points at a
directory that does not exist yet (fresh clone, bind-mounted volume leaf), the
RotatingFileHandler would otherwise fail to open the log file at startup.
"""

import logging
import logging.handlers
from pathlib import Path

from lightrag.constants import DEFAULT_LOG_FILENAME


def _reset_configured_loggers() -> None:
    for name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.filters = []
        logger.propagate = True


def test_configure_logging_creates_log_dir(tmp_path, monkeypatch) -> None:
    from lightrag.api.lightrag_server import configure_logging

    log_dir = tmp_path / "logs" / "nested"
    monkeypatch.setenv("LOG_DIR", str(log_dir))
    try:
        configure_logging()

        log_file = log_dir / DEFAULT_LOG_FILENAME
        assert log_dir.is_dir()
        assert log_file.is_file()

        handlers = logging.getLogger("lightrag").handlers
        assert any(
            isinstance(h, logging.handlers.RotatingFileHandler)
            and Path(h.baseFilename).resolve() == log_file.resolve()
            for h in handlers
        )
    finally:
        _reset_configured_loggers()
