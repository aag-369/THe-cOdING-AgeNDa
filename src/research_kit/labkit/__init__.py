"""labkit: run provenance -- deterministic run directories, manifests, seeding,
crash-safe metrics, replay, and diff.

labkit has **zero knowledge** of what it is running: no training loop, no model
code, no domain concepts.
"""

from __future__ import annotations

from .checksums import file_checksum, tree_checksum
from .diff import ConfigChange, ManifestChange, RunDiff, diff_runs
from .errors import (
    AmbiguousRunError,
    GitError,
    InputError,
    LabkitError,
    ManifestError,
    MetricsError,
    ReplayError,
    ReplayMismatchError,
    RunExistsError,
    RunNotFoundError,
    RunStateError,
)
from .handle import RunHandle, open_run
from .manifest import MANIFEST_SCHEMA_VERSION, Manifest, read_manifest
from .replay import ReplayResult, replay
from .run import FORCE_RUN_DIR_ENV, RUNS_DIR_ENV, Run
from .seeding import SeedReport, seed_everything

__all__ = [
    "FORCE_RUN_DIR_ENV",
    "MANIFEST_SCHEMA_VERSION",
    "RUNS_DIR_ENV",
    "AmbiguousRunError",
    "ConfigChange",
    "GitError",
    "InputError",
    "LabkitError",
    "Manifest",
    "ManifestChange",
    "ManifestError",
    "MetricsError",
    "ReplayError",
    "ReplayMismatchError",
    "ReplayResult",
    "Run",
    "RunDiff",
    "RunExistsError",
    "RunHandle",
    "RunNotFoundError",
    "RunStateError",
    "SeedReport",
    "diff_runs",
    "file_checksum",
    "open_run",
    "read_manifest",
    "replay",
    "seed_everything",
    "tree_checksum",
]
