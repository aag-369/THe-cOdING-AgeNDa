"""Lightweight wall-clock instrumentation."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

__all__ = ["Stopwatch"]


class Stopwatch:
    """Accumulates named wall-clock durations (seconds)."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = defaultdict(float)

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.totals[name] += time.perf_counter() - t0

    def as_dict(self) -> dict[str, float]:
        return dict(self.totals)
