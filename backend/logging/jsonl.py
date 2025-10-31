"""Structured JSONL logging utilities."""
from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, Iterator, Union

REQUIRED_FIELDS = {
    "run_id",
    "ts",
    "symbol",
    "made_at",
    "valid_at",
    "horizon",
    "action",
    "qty",
    "price",
    "cost_breakdown",
    "pnl",
    "cum_pnl",
    "position",
}


def _serialise(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (list, dict, str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def append_event(path: Union[str, Path], payload: Dict) -> None:
    missing = REQUIRED_FIELDS - payload.keys()
    if missing:
        raise ValueError(f"Missing required fields for JSONL event: {sorted(missing)}")
    serialised = {k: _serialise(v) for k, v in payload.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(serialised, ensure_ascii=False) + "\n")


@contextmanager
def open_run_logger(run_id: str) -> Iterator["RunLogger"]:
    logs_dir = Path("artifacts") / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"{run_id}.jsonl"
    logger = RunLogger(run_id=run_id, path=path)
    try:
        yield logger
    finally:
        logger.close()


class RunLogger:
    def __init__(self, run_id: str, path: Path) -> None:
        self.run_id = run_id
        self.path = path
        self.cum_pnl = 0.0

    def write(self, payload: Dict) -> None:
        payload = dict(payload)
        payload.setdefault("run_id", self.run_id)
        if "pnl" in payload:
            self.cum_pnl += float(payload["pnl"])
        payload.setdefault("cum_pnl", self.cum_pnl)
        append_event(self.path, payload)

    def close(self) -> None:
        pass
