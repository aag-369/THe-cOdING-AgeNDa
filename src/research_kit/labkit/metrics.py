"""Crash-safe append-only metrics: one JSON object per line in ``metrics.jsonl``.

Every :meth:`MetricsWriter.log` call writes a complete line and ``flush()``es
it, so a process that is killed mid-run loses at most nothing already returned
from ``log``. ``fsync`` (surviving an OS crash, not just a process crash) is
opt-in because it is slow.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, TextIO

from .errors import MetricsError

__all__ = ["RESERVED_METRIC_KEYS", "STEP_KEY", "WALL_TIME_KEY", "MetricsWriter"]

STEP_KEY = "_step"
WALL_TIME_KEY = "wall_time"
RESERVED_METRIC_KEYS = frozenset({STEP_KEY, WALL_TIME_KEY})


class MetricsWriter:
    """Owns the open ``metrics.jsonl`` file handle for one run."""

    def __init__(self, path: str | os.PathLike[str], *, fsync: bool = False) -> None:
        """Open ``path`` for appending (created if absent). Does not truncate."""
        self._path = Path(path)
        self._handle: TextIO = self._path.open("a", encoding="utf-8")
        self._fsync = fsync
        self._auto_step = 0
        self._closed = False

    def log(
        self,
        step: int | None = None,
        *,
        allow_nonscalar: bool = False,
        **values: object,
    ) -> None:
        """Append one JSON object: ``{_step, wall_time, **values}``.

        ``_step`` is ``step`` if given, else an auto-incrementing counter.
        ``wall_time`` is the current epoch seconds as a float. The line is
        flushed immediately (and ``fsync``'d if configured).

        Raises:
            MetricsError: a value is non-scalar and ``allow_nonscalar`` is
                False; a value is not JSON-serializable; or a key is reserved
                or starts with ``_``.
        """
        if self._closed:
            raise MetricsError("metrics writer is closed")

        for key in values:
            if key in RESERVED_METRIC_KEYS or key.startswith("_"):
                raise MetricsError(f"metric key {key!r} is reserved")

        if step is not None and (isinstance(step, bool) or not isinstance(step, int)):
            raise MetricsError(f"step must be an int, got {step!r}")

        row: dict[str, Any] = {}
        row[STEP_KEY] = self._auto_step if step is None else step
        self._auto_step = int(row[STEP_KEY]) + 1
        row[WALL_TIME_KEY] = time.time()

        for key, value in values.items():
            _check_value(key, value, allow_nonscalar=allow_nonscalar)
            row[key] = value

        try:
            line = json.dumps(row, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise MetricsError(f"metric row is not JSON-serializable: {exc}") from exc

        self._handle.write(line + "\n")
        self._handle.flush()
        if self._fsync:
            os.fsync(self._handle.fileno())

    def close(self) -> None:
        """Flush and close the file handle. Idempotent."""
        if not self._closed:
            self._handle.flush()
            self._handle.close()
            self._closed = True


def _check_value(key: str, value: object, *, allow_nonscalar: bool) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MetricsError(f"metric {key!r} is a non-finite float ({value!r})")
        return
    if allow_nonscalar:
        return
    raise MetricsError(
        f"metric {key!r} has non-scalar value of type {type(value).__name__}; "
        f"pass allow_nonscalar=True to log it anyway"
    )
