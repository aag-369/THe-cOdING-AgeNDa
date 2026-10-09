# research-kit

Foundation for the `research-kit` personal research monorepo: spec-driven
construction (`research_kit.core.spine`) and run provenance (`research_kit.labkit`).
Everything else in the monorepo — quantum, deep learning, agentic AI,
privacy-preserving computation — is built as a `spine` spec and run inside a
`labkit.Run`.

## Quickstart

```bash
pip install -e ".[dev]"

# 1. Register a constructible piece (normally done as an import side effect)
python - <<'PY'
from research_kit.core.spine import register, build

@register("model", "linear")
class LinearModel:
    def __init__(self, width: int) -> None:
        self.width = width

# 2. Build it from a plain dict spec
model = build({"_target": "model:linear", "width": 4})
print(model.width)  # 4
PY

# 3. Run something with full provenance
python - <<'PY'
from research_kit.labkit import Run

# require_git=True is the default and the recommended setting inside a real
# checkout; turned off here only so this snippet works from any directory.
with Run({"seed": 0, "lr": 0.1}, require_git=False) as run:
    run.seed(0)
    for step in range(3):
        run.log(step=step, loss=1.0 / (step + 1))
print(run.dir)  # runs/<config_hash>/
PY

# 4. Inspect it from the CLI
research-spine resolve path/to/config.yaml lr=0.05 --json
labkit show <hash-from-above> --metrics
labkit diff <hash-a> <hash-b>
```

`Run(...)` writes `resolved_config.yaml`, `manifest.json`, `metrics.jsonl`,
`logs/`, and `artifacts/` under `runs/<config_hash>/` — same config, same
directory, always. See [examples/end_to_end.py](examples/end_to_end.py) for a
complete, synthetic, no-network walkthrough (spine build + interpolation +
overrides, feeding a `labkit.Run` with a declared input, logged metrics, and
a saved artifact).

## Modules

- **`research_kit.core.spine`** — `register`/`build` over a process-global
  registry; `load_config`/`apply_overrides`/`resolve_config` (`${a.b}`,
  `${env:VAR}` interpolation); `config_hash` (order-independent, blake2b).
  CLI: `research-spine resolve`.
- **`research_kit.labkit`** — the `Run` context manager (deterministic run
  dir, manifest lifecycle, git/seed/input/metric provenance); `replay` (re-run
  a recorded invocation with mismatch detection); `diff_runs`. CLI:
  `labkit replay|diff|show`.

## Development

```bash
pytest                    # 131 tests; -m "not slow" skips the mypy subprocess check
mypy                      # --strict over src/research_kit
ruff check . && ruff format --check .
```

See [CHANGELOG.md](CHANGELOG.md) for release notes.
