"""``replay``: re-run a recorded run by re-invoking its original command.

labkit has zero knowledge of what a run *does*, so replay cannot "call" the
work. Instead every run records its ``sys.argv``, interpreter, and working
directory in the manifest, and replay re-invokes exactly that -- after
verifying the stored config still hashes to its directory name and that the
git SHA and declared input checksums still match.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from research_kit.core.spine.hashing import config_hash

from .checksums import file_checksum, tree_checksum
from .errors import InputError, ReplayError, ReplayMismatchError
from .gitinfo import collect_git_info
from .handle import RunHandle, resolve_run_dir
from .manifest import Manifest
from .run import FORCE_RUN_DIR_ENV

__all__ = ["ReplayResult", "replay"]

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReplayResult:
    """Outcome of a replay."""

    returncode: int
    replay_dir: Path
    mismatches: list[str]
    """Human-readable mismatch descriptions (empty on a clean replay)."""


def replay(
    hash_or_dir: str,
    *,
    runs_dir: str | os.PathLike[str] | None = None,
    strict: bool = False,
    extra_overrides: Sequence[str] = (),
) -> ReplayResult:
    """Re-execute a recorded run.

    Steps:
        1. Locate the run; reload ``resolved_config.yaml`` and re-check that it
           hashes to the run directory name / recorded hash.
        2. Compare the current git SHA to ``manifest["git"]["sha"]`` and every
           declared input's checksum to its recorded value.
        3. On any mismatch: log a loud warning and continue, unless ``strict``,
           in which case raise :class:`ReplayMismatchError`.
        4. Re-invoke ``manifest["invocation"]["argv"]`` with the recorded
           interpreter and cwd, with ``$RESEARCH_FORCE_RUN_DIR`` pointing at
           ``<run>/replays/<utc-timestamp>/`` so the child does not collide.

    Returns:
        :class:`ReplayResult` with the child return code and any mismatches.

    Raises:
        RunNotFoundError / AmbiguousRunError: run cannot be located.
        ReplayError: stored config no longer hashes to its name; recorded
            invocation is unusable; recorded interpreter is missing.
        ReplayMismatchError: ``strict`` and a git/input mismatch was found.
    """
    run_dir = resolve_run_dir(hash_or_dir, runs_dir=runs_dir)
    handle = RunHandle(run_dir)
    manifest = handle.manifest

    actual_hash = config_hash(handle.resolved_config)
    recorded_hash = manifest["config_hash"]
    if actual_hash != recorded_hash:
        raise ReplayError(
            f"stored resolved_config.yaml hashes to {actual_hash[:16]} but the "
            f"manifest records {recorded_hash[:16]}; the run is not reproducible"
        )

    mismatches: list[str] = []
    _check_git(manifest, mismatches)
    _check_inputs(manifest, mismatches)

    if mismatches and strict:
        raise ReplayMismatchError("; ".join(mismatches))
    if mismatches:
        _log.warning(
            "replay of %s proceeding despite mismatches: %s",
            run_dir.name,
            "; ".join(mismatches),
        )

    invocation = manifest["invocation"]
    argv = list(invocation["argv"])
    if not argv:
        raise ReplayError("recorded invocation has an empty argv; cannot replay")
    executable = invocation["executable"] or sys.executable
    if not Path(executable).exists():
        raise ReplayError(f"recorded interpreter no longer exists: {executable}")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%f")
    replay_dir = run_dir / "replays" / stamp
    # The child's Run creates this directory (its FORCE_RUN_DIR); pre-creating it
    # here would trip the child's on_exist="error" guard.
    replay_dir.parent.mkdir(parents=True, exist_ok=True)

    child_env = dict(os.environ)
    child_env[FORCE_RUN_DIR_ENV] = str(replay_dir)

    command = [executable, *argv, *extra_overrides]
    completed = subprocess.run(
        command,
        cwd=invocation["cwd"] or None,
        env=child_env,
        check=False,
    )
    return ReplayResult(
        returncode=completed.returncode,
        replay_dir=replay_dir,
        mismatches=mismatches,
    )


def _check_git(manifest: Manifest, mismatches: list[str]) -> None:
    recorded_sha = manifest["git"]["sha"]
    if not recorded_sha:
        return
    current = collect_git_info(Path(manifest["invocation"]["cwd"]))
    if current["available"] and current["sha"] and current["sha"] != recorded_sha:
        mismatches.append(
            f"git SHA changed: recorded {recorded_sha[:10]}, now {current['sha'][:10]}"
        )


def _check_inputs(manifest: Manifest, mismatches: list[str]) -> None:
    for record in manifest["inputs"]:
        path = Path(record["path"])
        try:
            if record["kind"] == "dir":
                current, _ = tree_checksum(path)
            else:
                current = file_checksum(path)
        except InputError:
            mismatches.append(f"declared input is missing: {record['path']}")
            continue
        if current != record["checksum"]:
            mismatches.append(f"declared input changed: {record['path']}")
