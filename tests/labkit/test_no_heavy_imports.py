"""Invariant 5: zero heavyweight imports at module import time."""

from __future__ import annotations

import subprocess
import sys

HEAVY = ["numpy", "torch", "qiskit", "tenseal"]


def test_importing_research_kit_pulls_no_heavy_libs() -> None:
    code = (
        "import sys;"
        "import research_kit, research_kit.core.spine, research_kit.labkit;"
        f"bad=[m for m in {HEAVY!r} if m in sys.modules];"
        "print(bad);"
        "sys.exit(1 if bad else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
