#!/usr/bin/env python
"""End-to-end example: core.spine builds a synthetic model + optimizer from a
YAML spec (with ``${...}`` interpolation and CLI-style overrides); labkit runs
it with full provenance -- deterministic run directory, manifest, seeding, a
declared input, an artifact, and a metrics log.

Everything here is generated in-process. No network access, no external data.

Run it (from the repo root, so the "runs/" it creates lands there)::

    python examples/end_to_end.py
    python examples/end_to_end.py base_lr=0.05 n_steps=50

Then inspect the run it created::

    research-spine resolve examples/synthetic_config.yaml
    labkit show <hash-printed-below> --metrics

The run directory follows labkit's own default resolution: ``$RESEARCH_RUNS_DIR``
if set, else ``./runs`` relative to the current working directory -- this
script does not hardcode a location.
"""

from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

from research_kit.core.spine import build, load_resolved, register
from research_kit.labkit import Run

EXAMPLE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXAMPLE_DIR / "synthetic_config.yaml"


# --------------------------------------------------------------------------- #
# 1. Register the constructible pieces. In a real project each of these lives #
#    in its own module and is registered as an import side effect; build()   #
#    never needs to know they exist ahead of time.                           #
# --------------------------------------------------------------------------- #
@register("model", "linear")
class LinearModel:
    """A toy model: one weight per input feature, nudged toward -x each step."""

    def __init__(self, width: int) -> None:
        self.width = width
        self.weights = [0.0] * width

    def step(self, inputs: list[float], lr: float) -> float:
        loss = 0.0
        for i, x in enumerate(inputs):
            error = self.weights[i] - x
            self.weights[i] -= lr * error
            loss += error * error
        return loss / max(len(inputs), 1)


@register("optim", "sgd")
class SGD:
    def __init__(self, lr: float) -> None:
        self.lr = lr


def make_synthetic_dataset(seed: int, n_steps: int, width: int, path: Path) -> None:
    """Write a small synthetic dataset -- stands in for the kind of file a
    real run would `declare_input` for checksum-based provenance."""
    rng = random.Random(seed)
    rows = (",".join(f"{rng.uniform(-1, 1):.6f}" for _ in range(width)) for _ in range(n_steps))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    overrides = argv[1:]
    cfg, digest = load_resolved(CONFIG_PATH, overrides)
    print(f"resolved config hash: {digest}")

    with tempfile.TemporaryDirectory() as scratch:
        dataset_path = Path(scratch) / "synthetic_data.csv"
        make_synthetic_dataset(
            seed=cfg["seed"],
            n_steps=cfg["n_steps"],
            width=cfg["model"]["width"],
            path=dataset_path,
        )

        # require_git=False so this example runs the same inside or outside a
        # git checkout; a real project leaves it at its default (True).
        with Run(cfg, require_git=False, on_exist="new_attempt") as run:
            model = build(cfg["model"])
            optim = build(cfg["optim"])
            run.declare_input(dataset_path, name="synthetic_dataset")

            loss = 0.0
            for step, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines()):
                row = [float(x) for x in line.split(",")]
                loss = model.step(row, optim.lr)
                run.log(step=step, loss=loss)

            weights_path = run.artifact_path("final_weights.txt")
            weights_path.write_text(",".join(f"{w:.6f}" for w in model.weights), encoding="utf-8")

            print(f"run dir:     {run.dir}")
            print(f"config hash: {run.config_hash}")
            print(f"final loss:  {loss:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
