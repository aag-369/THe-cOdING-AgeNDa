"""metrics.jsonl: append-only, crash-safe (invariant 3)."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from research_kit.labkit import Run
from research_kit.labkit.errors import MetricsError, RunStateError


def _rows(run_dir: Path) -> list[dict[str, object]]:
    text = (run_dir / "metrics.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_one_line_per_call(runs_dir: Path) -> None:
    with Run({"seed": 1}, runs_dir=runs_dir, require_git=False) as run:
        run.log(step=0, loss=1.0)
        run.log(step=1, loss=0.5)
        d = run.dir
    rows = _rows(d)
    assert [r["loss"] for r in rows] == [1.0, 0.5]


def test_wall_time_and_step_added(runs_dir: Path) -> None:
    with Run({"seed": 1}, runs_dir=runs_dir, require_git=False) as run:
        run.log(step=7, acc=0.9)
        d = run.dir
    row = _rows(d)[0]
    assert row["_step"] == 7
    assert isinstance(row["wall_time"], float)


def test_auto_step_increments(runs_dir: Path) -> None:
    with Run({"seed": 1}, runs_dir=runs_dir, require_git=False) as run:
        run.log(x=1)
        run.log(x=2)
        d = run.dir
    assert [r["_step"] for r in _rows(d)] == [0, 1]


def test_data_survives_hard_crash(tmp_path: Path) -> None:
    script = tmp_path / "crasher.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import os
            from research_kit.labkit import Run
            run = Run({{"seed": 1}}, runs_dir=r"{tmp_path / "runs"}", require_git=False)
            run.__enter__()
            for i in range(5):
                run.log(step=i, v=i)
            os._exit(1)
            """
        ),
        encoding="utf-8",
    )
    subprocess.run([sys.executable, str(script)], check=False)
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    rows = _rows(run_dirs[0])
    assert [r["v"] for r in rows] == [0, 1, 2, 3, 4]


def test_nonscalar_value_raises_unless_opted_in(runs_dir: Path) -> None:
    with Run({"seed": 1}, runs_dir=runs_dir, require_git=False) as run:
        with pytest.raises(MetricsError):
            run.log(step=0, hist=[1, 2, 3])
        run.log(step=0, hist=[1, 2, 3], allow_nonscalar=True)


def test_non_serializable_value_raises(runs_dir: Path) -> None:
    with Run({"seed": 1}, runs_dir=runs_dir, require_git=False) as run:
        with pytest.raises(MetricsError):
            run.log(step=0, obj=object())


def test_reserved_key_raises(runs_dir: Path) -> None:
    with Run({"seed": 1}, runs_dir=runs_dir, require_git=False) as run:
        with pytest.raises(MetricsError):
            run.log(step=0, _step=5)
        with pytest.raises(MetricsError):
            run.log(step=0, wall_time=1.0)


def test_log_after_exit_raises(runs_dir: Path) -> None:
    with Run({"seed": 1}, runs_dir=runs_dir, require_git=False) as run:
        pass
    with pytest.raises(RunStateError):
        run.log(step=0, x=1)
