"""Run registry management utilities.

Provides a thread-safe interface for storing and promoting model training runs.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


class RunRegistry:
    """Persist and manage information about training runs.

    Parameters
    ----------
    registry_path:
        Path to the JSON registry file (defaults to ``artifacts/registry.json``).
    models_dir:
        Directory containing model artifacts. The ``active`` pointer will be
        created inside this directory when a run is promoted.
    """

    def __init__(
        self,
        registry_path: str | os.PathLike[str] | None = None,
        models_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self.registry_path = Path(registry_path or Path("artifacts") / "registry.json")
        self.models_dir = Path(models_dir or Path("artifacts") / "models")
        self._lock = RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_run(
        self,
        run_id: str,
        checkpoint_path: str | os.PathLike[str],
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Insert a new run entry and persist it."""

        entry = self._build_entry(run_id, checkpoint_path, metadata)
        with self._lock:
            state = self._load_state()
            if any(run["id"] == run_id for run in state["runs"]):
                raise ValueError(f"Run '{run_id}' already exists in registry")
            state["runs"].append(entry)
            self._write_state(state)
        return entry.copy()

    def update_run(self, run_id: str, **updates: Any) -> dict[str, Any]:
        """Update an existing run entry.

        Parameters
        ----------
        run_id:
            Identifier of the run to update.
        updates:
            Arbitrary key/value pairs to merge into the stored entry. ``metadata``
            keys are merged shallowly when provided as a mapping.
        """

        with self._lock:
            state = self._load_state()
            run = self._get_run(state["runs"], run_id)
            if run is None:
                raise KeyError(f"Run '{run_id}' does not exist")

            payload = updates.copy()
            metadata_updates = payload.pop("metadata", None)
            if metadata_updates is not None:
                if not isinstance(metadata_updates, dict):
                    raise TypeError("metadata updates must be a mapping")
                current_metadata = run.get("metadata", {})
                if not isinstance(current_metadata, dict):
                    current_metadata = {}
                run["metadata"] = {**current_metadata, **metadata_updates}

            for key, value in payload.items():
                run[key] = value

            run["updated_at"] = self._timestamp()
            self._write_state(state)
            return run.copy()

    def list_runs(self) -> list[dict[str, Any]]:
        """Return a copy of the stored runs."""

        with self._lock:
            state = self._load_state()
            return [run.copy() for run in state["runs"]]

    def promote(self, run_id: str) -> dict[str, Any]:
        """Mark ``run_id`` as active and update the active model pointer."""

        with self._lock:
            state = self._load_state()
            run = self._get_run(state["runs"], run_id)
            if run is None:
                raise KeyError(f"Run '{run_id}' does not exist")

            checkpoint = Path(run["checkpoint_path"]).expanduser().resolve()
            if not checkpoint.exists():
                raise FileNotFoundError(f"Checkpoint for run '{run_id}' not found: {checkpoint}")

            timestamp = self._timestamp()
            state["active"] = run_id
            run["promoted_at"] = timestamp
            run["updated_at"] = timestamp
            self._write_state(state)

            self._update_active_pointer(checkpoint)
            return run.copy()

    def resolve_active(self) -> Optional[dict[str, Any]]:
        """Return the active run entry if present."""

        with self._lock:
            state = self._load_state()
            active_id = state.get("active")
            if not isinstance(active_id, str):
                return None
            run = self._get_run(state["runs"], active_id)
            return run.copy() if run else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_state(self) -> dict[str, Any]:
        try:
            raw = self.registry_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return self._default_state()
        except OSError as exc:
            logger.warning("Failed to read registry: %s", exc)
            return self._default_state()

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("Registry file is corrupt; resetting to defaults")
            return self._default_state()

        return self._validate_state(payload)

    def _write_state(self, state: dict[str, Any]) -> None:
        directory = self.registry_path.parent
        directory.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(state, indent=2, sort_keys=True)

        with NamedTemporaryFile("w", dir=directory, delete=False, encoding="utf-8") as tmp:
            tmp.write(serialized)
            tmp.flush()
            os.fsync(tmp.fileno())
            temp_path = Path(tmp.name)
        temp_path.replace(self.registry_path)

    def _validate_state(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return self._default_state()

        runs = payload.get("runs")
        if not isinstance(runs, list):
            runs = []

        normalized_runs: list[dict[str, Any]] = []
        for item in runs:
            normalized = self._normalize_run(item)
            if normalized is not None:
                normalized_runs.append(normalized)

        active_id = payload.get("active")
        if not isinstance(active_id, str) or not any(run["id"] == active_id for run in normalized_runs):
            active_id = None

        return {"runs": normalized_runs, "active": active_id}

    def _normalize_run(self, item: Any) -> Optional[dict[str, Any]]:
        if not isinstance(item, dict):
            return None

        run_id = item.get("id")
        checkpoint_path = item.get("checkpoint_path")
        if not isinstance(run_id, str) or not run_id:
            return None
        if not isinstance(checkpoint_path, str) or not checkpoint_path:
            return None

        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        created_at = item.get("created_at")
        if not isinstance(created_at, str):
            created_at = self._timestamp()

        updated_at = item.get("updated_at")
        if not isinstance(updated_at, str):
            updated_at = created_at

        promoted_at = item.get("promoted_at")
        if promoted_at is not None and not isinstance(promoted_at, str):
            promoted_at = None

        run = {
            "id": run_id,
            "checkpoint_path": checkpoint_path,
            "metadata": metadata,
            "created_at": created_at,
            "updated_at": updated_at,
        }
        if promoted_at is not None:
            run["promoted_at"] = promoted_at
        for key in ("status", "metrics"):
            value = item.get(key)
            if isinstance(value, (str, dict)):
                run[key] = value
        return run

    def _build_entry(
        self,
        run_id: str,
        checkpoint_path: str | os.PathLike[str],
        metadata: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        timestamp = self._timestamp()
        entry = {
            "id": run_id,
            "checkpoint_path": str(Path(checkpoint_path)),
            "metadata": metadata.copy() if isinstance(metadata, dict) else {},
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        return entry

    def _get_run(self, runs: Iterable[dict[str, Any]], run_id: str) -> Optional[dict[str, Any]]:
        for run in runs:
            if run.get("id") == run_id:
                return run
        return None

    def _update_active_pointer(self, checkpoint: Path) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        pointer = self.models_dir / "active"

        if pointer.exists() or pointer.is_symlink():
            if pointer.is_dir() and not pointer.is_symlink():
                raise RuntimeError(
                    "Existing active pointer is a directory; manual intervention required",
                )
            self._remove_path(pointer)

        try:
            pointer.symlink_to(checkpoint, target_is_directory=checkpoint.is_dir())
        except (OSError, NotImplementedError) as exc:
            logger.warning(
                "Failed to create active symlink, falling back to pointer file: %s",
                exc,
            )
            pointer.write_text(str(checkpoint), encoding="utf-8")

    def _remove_path(self, path: Path) -> None:
        try:
            if path.is_dir() and not path.is_symlink():
                for child in path.iterdir():
                    self._remove_path(child)
                path.rmdir()
            else:
                try:
                    path.unlink()
                except FileNotFoundError:
                    return
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.warning("Failed to remove existing active pointer: %s", exc)

    @staticmethod
    def _default_state() -> dict[str, Any]:
        return {"runs": [], "active": None}

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()


class Registry(RunRegistry):
    """Backward-compatible alias matching the project specification."""


__all__ = ["RunRegistry", "Registry"]
