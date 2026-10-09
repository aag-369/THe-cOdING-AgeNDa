"""The examples/ script must keep working: run it as a subprocess (it manages
its own temp runs directory) and check it completes end to end."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "end_to_end.py"


def test_end_to_end_example_runs_clean(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(_EXAMPLE)],
        cwd=tmp_path,  # so the "runs/" it creates lands in a throwaway dir
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "config hash:" in proc.stdout
    assert "final loss:" in proc.stdout


def test_end_to_end_example_accepts_overrides(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(_EXAMPLE), "base_lr=0.5", "n_steps=3"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "config hash:" in proc.stdout
