"""Logging with timestamps in the configured local timezone (IST by default), not UTC."""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from .timeutil import get_zone


class LocalTimeFormatter(logging.Formatter):
    def __init__(self, fmt: str, tz_name: str):
        super().__init__(fmt)
        self._zone = get_zone(tz_name)

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:  # noqa: N802
        return datetime.fromtimestamp(record.created, self._zone).isoformat(timespec="seconds")


def configure_logging(tz_name: str = "Asia/Kolkata", level: int = logging.INFO, log_file: Path | None = None) -> None:
    fmt = LocalTimeFormatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", tz_name)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    root = logging.getLogger()
    root.handlers.clear()
    for h in handlers:
        h.setFormatter(fmt)
        root.addHandler(h)
    root.setLevel(level)
