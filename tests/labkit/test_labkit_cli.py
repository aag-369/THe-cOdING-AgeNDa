"""labkit CLI: thin wrappers (invariant 2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_kit.labkit import Run, diff_runs
from research_kit.labkit import cli as labkit_cli
from research_kit.labkit.errors import AmbiguousRunError


def _make_run(runs_dir: Path, cfg: dict[str, object]) -> str:
    with Run(cfg, runs_dir=runs_dir, require_git=False) as run:
        return run.dir.name


def test_show_prints_manifest_and_config(
    runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    h = _make_run(runs_dir, {"seed": 3, "lr": 0.1})
    rc = labkit_cli.main(["show", h, "--runs-dir", str(runs_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "lr" in out and "completed" in out


def test_show_json(runs_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    h = _make_run(runs_dir, {"seed": 3})
    labkit_cli.main(["show", h, "--json", "--runs-dir", str(runs_dir)])
    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest"]["status"] == "completed"


def test_diff_cli_matches_library(runs_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a = _make_run(runs_dir, {"seed": 1, "lr": 0.1})
    b = _make_run(runs_dir, {"seed": 1, "lr": 0.2})
    labkit_cli.main(["diff", a, b, "--json", "--runs-dir", str(runs_dir)])
    payload = json.loads(capsys.readouterr().out)
    assert payload == diff_runs(a, b, runs_dir=runs_dir).as_dict()


def test_unknown_subcommand_exits_2() -> None:
    with pytest.raises(SystemExit) as ei:
        labkit_cli.main(["frobnicate"])
    assert ei.value.code == 2


def test_ambiguous_short_hash_raises(runs_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Two run dirs sharing a prefix -> AmbiguousRunError from resolution.
    (runs_dir / ("ab" + "c" * 62)).mkdir()
    (runs_dir / ("ab" + "d" * 62)).mkdir()
    with pytest.raises(AmbiguousRunError):
        labkit_cli.main(["show", "ab", "--runs-dir", str(runs_dir)])
