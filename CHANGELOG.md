# Changelog

All notable changes to `research-kit` are recorded here.

## 0.1.0 — initial release: `core.spine` + `labkit`

The foundation module. Every other module in the monorepo builds on this one.

### `research_kit.core.spine`

- `Registry` / `register` / `build`: a process-global (and swappable)
  `(kind, name) -> target` registry, and a recursive `build(spec)` that
  constructs an object graph from a plain dict/YAML spec (`_target`,
  `_partial_`, `_imports_`).
- `load_config` / `apply_overrides` / `resolve_config` / `load_resolved`: the
  fixed YAML → overrides → `${...}` interpolation pipeline. Interpolation
  supports path references (`${a.b[0]}`) and environment variables
  (`${env:VAR}`, `${env:VAR,default}`), with cycle detection.
  Dotted-path overrides support `PATH=VALUE` (replace) and `+PATH=VALUE`
  (create).
- `canonicalize` / `config_hash` / `short_hash`: order-independent config
  hashing (blake2b-256) that is the basis for every deterministic run
  directory in `labkit`.
- CLI: `research-spine resolve CONFIG [OVERRIDE ...] [--json] [--hash-only]`.

### `research_kit.labkit`

- `Run`: a context manager that materializes `runs/<config_hash>/` with
  `resolved_config.yaml`, a full `manifest.json` (git state, seeds,
  interpreter/package/platform info, accelerators, declared inputs, the exact
  invocation), `metrics.jsonl`, `logs/`, and `artifacts/`. Handles the
  `running` → `completed`/`failed` manifest lifecycle, including on exception.
- `seed_everything` / `SeedReport`: one call seeds `random`, `numpy`, and
  `torch` (whichever are installed) and records exactly what happened.
- Crash-safe metrics: `run.log(step=..., **values)` appends one flushed JSON
  line per call.
- `replay`: re-invokes a recorded run's exact command line after verifying
  the stored config still hashes correctly and that git/input state hasn't
  drifted (warn by default, `--strict` to abort on drift).
- `diff_runs`: structured config + manifest diff between two runs (or a run
  and a bare config file).
- CLI: `labkit replay|diff|show`.

### Hardening

Five adversarial misuse cases converted from crashes/confusing errors into
typed exceptions (see the module's test suite for the exact scenarios):
circular YAML anchors (`RecursionError` → `ConfigTypeError`/`SpecError`/
`InterpolationError`), a non-mapping top-level config passed directly to
`resolve_config`, a non-string `kind`/`name` passed to `register`, a bare
string passed as `Run(inputs=...)`, and a directory-creation race between
concurrent `Run`s targeting the same config hash.

### Non-goals

No training loop, no model code, no quantum, no agents — by design, this
module has zero knowledge of what it is running.
