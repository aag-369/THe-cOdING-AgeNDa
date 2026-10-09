"""Run lifecycle + manifest (invariants 3 and 4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_kit.core.spine import config_hash
from research_kit.labkit import GitError, InputError, Run, RunExistsError
from research_kit.labkit.errors import RunStateError


def _cfg() -> dict[str, object]:
    return {"seed": 1, "optim": {"lr": 0.1}}


def test_run_creates_directory_layout(runs_dir: Path) -> None:
    with Run(_cfg(), runs_dir=runs_dir, require_git=False) as run:
        d = run.dir
    assert (d / "resolved_config.yaml").is_file()
    assert (d / "manifest.json").is_file()
    assert (d / "metrics.jsonl").is_file()
    assert (d / "logs").is_dir()
    assert (d / "artifacts").is_dir()


def test_run_dir_name_is_config_hash(runs_dir: Path) -> None:
    cfg = _cfg()
    with Run(cfg, runs_dir=runs_dir, require_git=False) as run:
        assert run.dir.name == config_hash(cfg)


def test_resolved_config_bytes_are_deterministic(tmp_path: Path) -> None:
    cfg = {"b": 1, "a": {"y": 2, "x": 1}}
    out = []
    for name in ("r1", "r2"):
        rd = tmp_path / name
        rd.mkdir()
        with Run(cfg, runs_dir=rd, require_git=False) as run:
            out.append((run.dir / "resolved_config.yaml").read_bytes())
    assert out[0] == out[1]


def test_manifest_running_then_completed(runs_dir: Path) -> None:
    with Run(_cfg(), runs_dir=runs_dir, require_git=False) as run:
        mid = json.loads((run.dir / "manifest.json").read_text())
        assert mid["status"] == "running"
        d = run.dir
    final = json.loads((d / "manifest.json").read_text())
    assert final["status"] == "completed"
    assert final["end_time"] is not None


def test_exception_marks_failed_and_reraises(runs_dir: Path) -> None:
    with pytest.raises(ValueError, match="boom"):
        with Run(_cfg(), runs_dir=runs_dir, require_git=False) as run:
            d = run.dir
            raise ValueError("boom")
    final = json.loads((d / "manifest.json").read_text())
    assert final["status"] == "failed"
    assert "boom" in final["error"]["traceback"]


def test_rerun_same_config_raises(runs_dir: Path) -> None:
    with Run(_cfg(), runs_dir=runs_dir, require_git=False):
        pass
    with pytest.raises(RunExistsError), Run(_cfg(), runs_dir=runs_dir, require_git=False):
        pass


def test_on_exist_new_attempt(runs_dir: Path) -> None:
    with Run(_cfg(), runs_dir=runs_dir, require_git=False):
        pass
    with Run(_cfg(), runs_dir=runs_dir, require_git=False, on_exist="new_attempt") as run:
        assert "attempt" in run.dir.name


def test_require_git_outside_repo_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(GitError), Run(_cfg(), runs_dir=tmp_path / "runs", require_git=True):
        pass


def test_dirty_tree_records_diff_and_untracked(
    tmp_git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_git_repo)
    (tmp_git_repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    (tmp_git_repo / "new.txt").write_text("x\n", encoding="utf-8")
    with Run(_cfg(), runs_dir=tmp_git_repo / "runs", require_git=True) as run:
        d = run.dir
    git = json.loads((d / "manifest.json").read_text())["git"]
    assert git["dirty"] is True
    assert "changed" in git["diff"]
    assert "new.txt" in git["untracked"]
    assert "x\n" not in json.dumps(git["untracked"])  # paths only


def test_manifest_records_python_and_packages(runs_dir: Path) -> None:
    with Run(_cfg(), runs_dir=runs_dir, require_git=False) as run:
        d = run.dir
    m = json.loads((d / "manifest.json").read_text())
    assert m["python"]["version"]
    assert "pyyaml" in {k.lower() for k in m["packages"]}
    assert m["platform"]["hostname"]


def test_declare_input_records_checksum(runs_dir: Path, tmp_path: Path) -> None:
    f = tmp_path / "data.bin"
    f.write_bytes(b"abc")
    with Run(_cfg(), runs_dir=runs_dir, require_git=False) as run:
        cs = run.declare_input(f, name="data")
        d = run.dir
    rec = json.loads((d / "manifest.json").read_text())["inputs"][0]
    assert rec["name"] == "data"
    assert rec["checksum"] == cs


def test_missing_declared_input_raises(runs_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(InputError):
        with Run(_cfg(), runs_dir=runs_dir, require_git=False, inputs=[tmp_path / "nope"]):
            pass


def test_artifact_path_inside_and_rejects_escape(runs_dir: Path) -> None:
    with Run(_cfg(), runs_dir=runs_dir, require_git=False) as run:
        p = run.artifact_path("sub/thing.pt")
        assert p.parent.is_dir()
        assert (run.dir / "artifacts") in p.parents
        with pytest.raises(ValueError):
            run.artifact_path("../escape")


def test_use_after_exit_and_double_enter(runs_dir: Path) -> None:
    run = Run(_cfg(), runs_dir=runs_dir, require_git=False)
    with run, pytest.raises(RunStateError):
        run.__enter__()
    with pytest.raises(RunStateError):
        run.log(step=0, x=1)


# --- PHASE 4 hardening: adversarial misuse ---------------------------------


def test_inputs_as_bare_string_raises_helpful_error(runs_dir: Path) -> None:
    """``inputs="data.csv"`` is a classic Python footgun: a plain string is a
    sequence of its own characters, so ``list(inputs)`` would silently turn it
    into ['d', 'a', 't', 'a', ...] and later fail with a baffling
    "declared input not found: d". Reject it immediately with a clear error."""
    with pytest.raises(InputError, match="single string"):
        Run(_cfg(), runs_dir=runs_dir, require_git=False, inputs="data.csv")


def test_concurrent_run_directory_race_raises_run_exists_error(
    runs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two processes launched with the identical config (e.g. concurrent
    SLURM array-job workers) can both pass the `exists()` check before either
    calls `mkdir`. The loser must see a typed RunExistsError, never a raw
    FileExistsError."""
    cfg = _cfg()
    target_dir = runs_dir / config_hash(cfg)
    original_mkdir = Path.mkdir
    already_raced = False

    def racy_mkdir(self: Path, *args: object, **kwargs: object) -> None:
        nonlocal already_raced
        if not already_raced and self == target_dir and kwargs.get("parents"):
            already_raced = True
            original_mkdir(self, parents=True, exist_ok=True)  # the "other" process wins
            raise FileExistsError(f"[simulated race] {self}")
        original_mkdir(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "mkdir", racy_mkdir)
    with pytest.raises(RunExistsError):
        with Run(cfg, runs_dir=runs_dir, require_git=False):
            pass
