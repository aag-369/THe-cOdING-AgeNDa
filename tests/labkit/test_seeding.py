"""seed_everything + Run seeding."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from research_kit.labkit import Run, seed_everything


def test_seeds_random_reproducibly() -> None:
    seed_everything(42)
    a = [random.random() for _ in range(5)]
    seed_everything(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_report_lists_seeded_and_absent() -> None:
    rep = seed_everything(0)
    assert rep.seeded["random"] is True
    assert isinstance(rep.absent, list)


def test_torch_absent_recorded_not_error() -> None:
    rep = seed_everything(0)
    if "torch" in rep.absent:
        assert rep.seeded.get("torch") in (False, None)


def test_seed_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        seed_everything(2**32)
    with pytest.raises(ValueError):
        seed_everything(-1)


def test_pythonhashseed_recorded() -> None:
    rep = seed_everything(0)
    assert hasattr(rep, "pythonhashseed")


def test_run_auto_seeds_from_cfg(runs_dir: Path) -> None:
    with Run({"seed": 99}, runs_dir=runs_dir, require_git=False) as run:
        d = run.dir
    seeds = json.loads((d / "manifest.json").read_text())["seeds"]
    assert seeds and seeds[0]["seed"] == 99


def test_explicit_run_seed_appended_in_order(runs_dir: Path) -> None:
    with Run({"seed": 1}, runs_dir=runs_dir, require_git=False) as run:
        run.seed(2)
        run.seed(3)
        d = run.dir
    seeds = [s["seed"] for s in json.loads((d / "manifest.json").read_text())["seeds"]]
    assert seeds == [1, 2, 3]


def test_numpy_reproducible_if_present() -> None:
    np = pytest.importorskip("numpy")
    seed_everything(7)
    a = np.random.rand(4).tolist()
    seed_everything(7)
    assert np.random.rand(4).tolist() == a


def test_torch_seeded_when_present() -> None:
    torch = pytest.importorskip("torch")
    rep = seed_everything(5, deterministic_torch=True)
    assert rep.seeded["torch"] is True
    assert rep.deterministic_torch is True
    first = torch.rand(3).tolist()
    seed_everything(5, deterministic_torch=True)
    assert torch.rand(3).tolist() == first
