"""diff_runs: structured config + manifest comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_kit.labkit import Run, diff_runs
from research_kit.labkit.errors import RunNotFoundError


def _make_run(runs_dir: Path, cfg: dict[str, object]) -> str:
    with Run(cfg, runs_dir=runs_dir, require_git=False) as run:
        return run.dir.name


def test_config_diff_added_removed_changed(runs_dir: Path) -> None:
    a = _make_run(runs_dir, {"seed": 1, "lr": 0.1, "gone": 1})
    b = _make_run(runs_dir, {"seed": 1, "lr": 0.2, "added": 5})
    d = diff_runs(a, b, runs_dir=runs_dir)
    kinds = {(c.path, c.kind) for c in d.config}
    assert ("lr", "changed") in kinds
    assert ("gone", "removed") in kinds
    assert ("added", "added") in kinds


def test_manifest_diff_curated_keys(runs_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a = _make_run(runs_dir, {"seed": 1})
    b_dir = runs_dir / "fake"
    b_dir.mkdir()
    # clone run a's outputs, then perturb the manifest's python version
    src = runs_dir / a
    (b_dir / "resolved_config.yaml").write_text(
        (src / "resolved_config.yaml").read_text(), encoding="utf-8"
    )
    m = json.loads((src / "manifest.json").read_text())
    m["python"]["version"] = "9.9.9"
    (b_dir / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    d = diff_runs(a, str(b_dir), runs_dir=runs_dir)
    assert any(c.key == "python.version" for c in d.manifest)


def test_package_diff_hidden_unless_flag(runs_dir: Path) -> None:
    a = _make_run(runs_dir, {"seed": 1})
    b = _make_run(runs_dir, {"seed": 2})
    assert not any("packages" in c.key for c in diff_runs(a, b, runs_dir=runs_dir).manifest)
    _ = diff_runs(a, b, runs_dir=runs_dir, include_packages=True)


def test_diff_accepts_bare_config_path(runs_dir: Path, tmp_path: Path) -> None:
    a = _make_run(runs_dir, {"seed": 1, "lr": 0.1})
    cfg_path = tmp_path / "other.yaml"
    cfg_path.write_text("seed: 1\nlr: 0.5\n", encoding="utf-8")
    d = diff_runs(a, str(cfg_path), runs_dir=runs_dir)
    assert any(c.path == "lr" for c in d.config)
    assert d.manifest == []  # no manifest on the bare-config side


def test_render_text_and_as_dict_consistent(runs_dir: Path) -> None:
    a = _make_run(runs_dir, {"seed": 1, "lr": 0.1})
    b = _make_run(runs_dir, {"seed": 1, "lr": 0.2})
    d = diff_runs(a, b, runs_dir=runs_dir)
    assert isinstance(d.render_text(), str)
    assert "lr" in json.dumps(d.as_dict())


def test_unknown_run_raises(runs_dir: Path) -> None:
    with pytest.raises(RunNotFoundError):
        diff_runs("0" * 64, "1" * 64, runs_dir=runs_dir)
