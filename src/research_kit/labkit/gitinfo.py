"""Git provenance capture via the ``git`` binary.

This layer **never raises** -- it reports what it could and could not learn.
Policy decisions (``require_git``, ``require_clean``) live in
:class:`~research_kit.labkit.run.Run`, which turns an unavailable/dirty result
into a :class:`~research_kit.labkit.errors.GitError` when the caller asked for
that.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import TypedDict

__all__ = ["LARGE_DIFF_BYTES", "GitInfo", "collect_git_info"]

_log = logging.getLogger(__name__)

LARGE_DIFF_BYTES = 1_048_576
"""A captured diff larger than this triggers a logged warning (still recorded)."""


class GitInfo(TypedDict):
    """Recorded under ``manifest["git"]``."""

    available: bool
    sha: str | None
    dirty: bool | None
    diff: str | None
    untracked: list[str] | None
    reason: str | None


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


def collect_git_info(cwd: str | os.PathLike[str]) -> GitInfo:
    """Inspect the git repository containing ``cwd``.

    Returns a fully-populated :class:`GitInfo`. When git is unavailable
    (no binary, not a repo) ``available`` is False and ``reason`` explains
    why; the remaining fields are ``None``. When available, ``sha`` and
    ``dirty`` are set; if ``dirty`` then ``diff`` holds ``git diff HEAD`` and
    ``untracked`` lists untracked paths (paths only, never their contents).
    """
    info: GitInfo = {
        "available": False,
        "sha": None,
        "dirty": None,
        "diff": None,
        "untracked": None,
        "reason": None,
    }
    directory = Path(cwd)

    try:
        toplevel = _git(directory, "rev-parse", "--show-toplevel")
    except FileNotFoundError as exc:
        info["reason"] = f"git executable not found: {exc}"
        return info
    except OSError as exc:
        info["reason"] = f"cannot invoke git: {exc}"
        return info

    if toplevel.returncode != 0:
        info["reason"] = (toplevel.stderr or "not a git repository").strip()
        return info

    head = _git(directory, "rev-parse", "HEAD")
    if head.returncode != 0:
        info["available"] = True
        info["reason"] = (head.stderr or "repository has no commits").strip()
        return info

    info["available"] = True
    info["sha"] = head.stdout.strip()

    status = _git(directory, "status", "--porcelain")
    dirty = bool(status.stdout.strip())
    info["dirty"] = dirty
    if dirty:
        diff = _git(directory, "diff", "HEAD").stdout
        if len(diff.encode("utf-8", "replace")) > LARGE_DIFF_BYTES:
            _log.warning("captured git diff for %s is %d bytes", directory, len(diff))
        info["diff"] = diff
        untracked = _git(directory, "ls-files", "--others", "--exclude-standard")
        info["untracked"] = [line for line in untracked.stdout.splitlines() if line]
    return info
