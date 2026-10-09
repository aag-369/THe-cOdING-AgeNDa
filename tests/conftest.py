"""Shared fixtures.

Every test that touches the process-global registry is isolated by the
autouse ``_isolate_registry`` fixture: it snapshots the registry before the
test and restores it after, so registrations never leak between tests.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_registry() -> Iterator[None]:
    from research_kit.core.spine.registry import get_registry

    reg = get_registry()
    snap = reg.snapshot()
    try:
        yield
    finally:
        reg.restore(snap)


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    d.mkdir()
    return d


@pytest.fixture
def synthetic_config() -> dict[str, object]:
    return {
        "seed": 123,
        "model": {"_target": "model:mlp", "width": 16, "depth": 2},
        "optim": {"lr": 0.01, "name": "adam"},
        "tags": ["a", "b"],
    }


@pytest.fixture
def tmp_git_repo(tmp_path: Path) -> Path:
    """A throwaway git repo with one commit."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    git("config", "commit.gpgsign", "false")
    (repo / "tracked.txt").write_text("original\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "initial")
    return repo


@pytest.fixture
def entrypoint_script(tmp_path: Path) -> Path:
    """A minimal runnable module: registers a target, opens a Run, logs a
    metric, and writes a sentinel file. Used by replay/CLI tests."""
    script = tmp_path / "entry.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys
            from pathlib import Path
            from research_kit.core.spine import register, build
            from research_kit.labkit import Run


            @register("thing", "widget")
            class Widget:
                def __init__(self, n: int) -> None:
                    self.n = n


            def run_main(argv: list[str]) -> int:
                sentinel = Path(argv[1]) if len(argv) > 1 else Path("sentinel.txt")
                cfg = {"seed": 7, "obj": {"_target": "thing:widget", "n": 3}}
                with Run(cfg, require_git=False) as run:
                    obj = build(cfg["obj"])
                    run.log(step=0, value=obj.n)
                    sentinel.write_text(str(obj.n), encoding="utf-8")
                return 0


            if __name__ == "__main__":
                raise SystemExit(run_main(sys.argv))
            """
        ),
        encoding="utf-8",
    )
    return script


@pytest.fixture
def run_python() -> callable:
    """Return a helper that runs a python snippet in a subprocess and returns
    the CompletedProcess."""

    def _run(code: str, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code), *args],
            cwd=cwd,
            capture_output=True,
            text=True,
        )

    return _run
