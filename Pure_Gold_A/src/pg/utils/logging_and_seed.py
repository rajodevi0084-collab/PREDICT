"""Utility helpers for deterministic execution and structured logging."""
from __future__ import annotations

import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np


_LOG_DIR = Path("Pure_Gold_A/logs/labels")
_LOG_FILE = _LOG_DIR / "labels.log"


class _DualTimeFormatter(logging.Formatter):
    """Formatter that embeds both UTC and local timestamps."""

    default_time_format = "%Y-%m-%dT%H:%M:%S.%f%z"

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        created_utc = datetime.fromtimestamp(record.created, tz=timezone.utc)
        created_local = created_utc.astimezone()
        utc_str = created_utc.strftime(self.default_time_format)
        local_str = created_local.strftime(self.default_time_format)
        return f"UTC {utc_str} | LOCAL {local_str}"


def set_seed(seed: int) -> None:
    """Seed random number generators for reproducibility."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that writes to the shared labels log file."""

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.INFO)
    formatter = _DualTimeFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = logging.FileHandler(_LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger
