"""Invariant 6: public functions have docstrings stating contracts; mypy --strict
passes on the module's own code."""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest

import research_kit.core.spine as spine
import research_kit.labkit as labkit

_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("module", [spine, labkit], ids=["spine", "labkit"])
def test_public_callables_have_docstrings(module: object) -> None:
    missing = []
    for name in module.__all__:  # type: ignore[attr-defined]
        obj = getattr(module, name)
        if inspect.isclass(obj) or inspect.isfunction(obj):
            if not (obj.__doc__ and obj.__doc__.strip()):
                missing.append(name)
    assert not missing, f"missing docstrings: {missing}"


@pytest.mark.slow
def test_mypy_strict_clean() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "mypy"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
