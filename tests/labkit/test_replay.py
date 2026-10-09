"""replay: re-invoke a recorded run's original command."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from research_kit.labkit import replay
from research_kit.labkit.errors import ReplayError, ReplayMismatchError, RunNotFoundError

_SCRIPT = """
import sys
from pathlib import Path
from research_kit.labkit import Run

def main() -> int:
    sentinel = Path(sys.argv[1])
    cfg = {"seed": 7, "note": "hello"}
    with Run(cfg, runs_dir=sys.argv[2], require_git=(sys.argv[3] == "git")) as run:
        run.log(step=0, v=1)
        prev = int(sentinel.read_text()) if sentinel.exists() else 0
        sentinel.write_text(str(prev + 1), encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
"""


def _record(repo: Path, *, git: bool) -> tuple[Path, Path]:
    script = repo / "job.py"
    script.write_text(textwrap.dedent(_SCRIPT), encoding="utf-8")
    sentinel = repo / "sentinel.txt"
    runs = repo / "runs"
    subprocess.run(
        [sys.executable, str(script), str(sentinel), str(runs), "git" if git else "nogit"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    (run_dir,) = list(runs.iterdir())
    return run_dir, sentinel


def test_replay_reinvokes_recorded_argv(tmp_path: Path) -> None:
    repo = tmp_path
    run_dir, sentinel = _record(repo, git=False)
    assert sentinel.read_text() == "1"
    result = replay(run_dir.name, runs_dir=run_dir.parent)
    assert result.returncode == 0
    assert result.replay_dir.exists()
    assert (run_dir / "replays") in result.replay_dir.parents
    assert sentinel.read_text() == "2"


def test_replay_detects_corrupted_stored_config(tmp_path: Path) -> None:
    run_dir, _ = _record(tmp_path, git=False)
    (run_dir / "resolved_config.yaml").write_text("seed: 999\n", encoding="utf-8")
    with pytest.raises(ReplayError):
        replay(run_dir.name, runs_dir=run_dir.parent)


def test_replay_warns_on_git_mismatch_but_continues(
    tmp_git_repo: Path, caplog: pytest.LogCaptureFixture
) -> None:
    run_dir, _ = _record(tmp_git_repo, git=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-qm", "moved on"],
        cwd=tmp_git_repo,
        check=True,
        capture_output=True,
    )
    result = replay(run_dir.name, runs_dir=run_dir.parent)
    assert result.returncode == 0
    assert result.mismatches


def test_replay_strict_raises_on_mismatch(tmp_git_repo: Path) -> None:
    run_dir, _ = _record(tmp_git_repo, git=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-qm", "moved on"],
        cwd=tmp_git_repo,
        check=True,
        capture_output=True,
    )
    with pytest.raises(ReplayMismatchError):
        replay(run_dir.name, runs_dir=run_dir.parent, strict=True)


def test_replay_unknown_hash_raises(tmp_path: Path) -> None:
    with pytest.raises(RunNotFoundError):
        replay("0" * 64, runs_dir=tmp_path)
