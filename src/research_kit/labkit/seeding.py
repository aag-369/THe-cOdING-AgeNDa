"""One call to seed every RNG in play, with an exact record of what was seeded."""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import random
from dataclasses import dataclass

__all__ = ["SEED_MAX", "SeedReport", "seed_everything"]

SEED_MAX = 2**32
"""numpy's legacy global seeder requires ``0 <= seed < 2**32``."""


@dataclass(frozen=True)
class SeedReport:
    """Exactly what :func:`seed_everything` did, for the run manifest."""

    seed: int
    seeded: dict[str, bool]
    """Keys attempted (``random``, ``numpy``, ``torch``, ``torch.cuda``) -> whether set."""
    absent: list[str]
    """Libraries that were not importable."""
    deterministic_torch: bool
    pythonhashseed: str | None
    """``os.environ['PYTHONHASHSEED']`` at call time (cannot be changed now)."""

    def as_dict(self) -> dict[str, object]:
        """Plain-dict form for JSON serialization into the manifest."""
        return dataclasses.asdict(self)


def _importable(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def seed_everything(seed: int, *, deterministic_torch: bool = False) -> SeedReport:
    """Seed ``random``; ``numpy`` (if importable); ``torch`` and ``torch.cuda``
    (if importable).

    With ``deterministic_torch=True`` also call
    ``torch.use_deterministic_algorithms(True)``, set cuDNN deterministic, and
    set the cuBLAS workspace environment variable.

    Does not set ``PYTHONHASHSEED`` (impossible after interpreter start) but
    records its current value in the report.

    Raises:
        ValueError: ``seed`` is not in ``[0, 2**32)``.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an int, got {seed!r}")
    if not 0 <= seed < SEED_MAX:
        raise ValueError(f"seed {seed} is outside the valid range [0, 2**32)")

    seeded: dict[str, bool] = {}
    absent: list[str] = []

    random.seed(seed)
    seeded["random"] = True

    if _importable("numpy"):
        import numpy as np

        np.random.seed(seed)
        seeded["numpy"] = True
    else:
        seeded["numpy"] = False
        absent.append("numpy")

    if _importable("torch"):
        import torch

        torch.manual_seed(seed)
        seeded["torch"] = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            seeded["torch.cuda"] = True
        else:
            seeded["torch.cuda"] = False
        if deterministic_torch:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        seeded["torch"] = False
        seeded["torch.cuda"] = False
        absent.append("torch")

    return SeedReport(
        seed=seed,
        seeded=seeded,
        absent=absent,
        deterministic_torch=deterministic_torch,
        pythonhashseed=os.environ.get("PYTHONHASHSEED"),
    )
