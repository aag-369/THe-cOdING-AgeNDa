"""The ``manifest.json`` schema and its read/write helpers.

The manifest is a plain ``dict`` (a ``TypedDict`` for static checking) written
as JSON -- one representation, JSON-native, no dataclass mirror. It is the
declared home of every nondeterministic fact about a run (timestamps, hostname,
diff, package set, accelerators).
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from .env import (
    ENV_ALLOWLIST,
    PlatformInfo,
    PythonInfo,
    accelerator_info,
    package_versions,
    platform_info,
    python_info,
)
from .errors import ManifestError
from .gitinfo import GitInfo, collect_git_info

__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "InputRecord",
    "InvocationInfo",
    "Manifest",
    "new_manifest",
    "read_manifest",
    "utc_now",
    "write_manifest",
]

MANIFEST_FILENAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1


class InputRecord(TypedDict):
    name: str
    path: str
    kind: Literal["file", "dir"]
    checksum: str
    size: int
    mtime: float
    files: list[tuple[str, str]] | None


class InvocationInfo(TypedDict):
    argv: list[str]
    executable: str
    cwd: str
    main: str | None
    env: dict[str, str]


class Manifest(TypedDict):
    """The full contents of ``manifest.json`` (schema version 1).

    Groups every recorded fact about a run: its config hash and status, start
    and end times, any failure traceback, git state, seeds, interpreter and
    package versions, platform, accelerators, declared inputs, the exact
    invocation, and determinism notes.
    """

    schema_version: int
    config_hash: str
    config_hash_short: str
    status: Literal["running", "completed", "failed"]
    start_time: str
    end_time: str | None
    error: dict[str, str] | None
    git: GitInfo
    seeds: list[dict[str, object]]
    python: PythonInfo
    packages: dict[str, str]
    platform: PlatformInfo
    accelerators: list[dict[str, object]]
    inputs: list[InputRecord]
    invocation: InvocationInfo
    determinism: dict[str, object]
    metrics_file: str


def utc_now() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def _invocation() -> InvocationInfo:
    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", None)
    return {
        "argv": list(sys.argv),
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "main": main_file,
        "env": {name: os.environ[name] for name in ENV_ALLOWLIST if name in os.environ},
    }


def new_manifest(config_hash: str) -> Manifest:
    """Build a ``status="running"`` manifest with environment/git/invocation
    fields populated and ``end_time``/``error`` unset."""
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "config_hash": config_hash,
        "config_hash_short": config_hash[:16],
        "status": "running",
        "start_time": utc_now(),
        "end_time": None,
        "error": None,
        "git": collect_git_info(Path.cwd()),
        "seeds": [],
        "python": python_info(),
        "packages": package_versions(),
        "platform": platform_info(),
        "accelerators": accelerator_info(),
        "inputs": [],
        "invocation": _invocation(),
        "determinism": {
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "notes": [],
        },
        "metrics_file": "metrics.jsonl",
    }


def write_manifest(run_dir: str | os.PathLike[str], manifest: Manifest) -> None:
    """Serialize ``manifest`` to ``<run_dir>/manifest.json`` atomically
    (write to a temp file in the same directory, then replace)."""
    directory = Path(run_dir)
    target = directory / MANIFEST_FILENAME
    handle_fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=".manifest-", suffix=".json")
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        os.replace(tmp_name, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def read_manifest(path: str | os.PathLike[str]) -> Manifest:
    """Load a manifest from a run directory or a direct file path.

    Raises:
        ManifestError: missing file, invalid JSON, not an object, or
            ``schema_version`` absent/unknown.
    """
    file_path = Path(path)
    if file_path.is_dir():
        file_path = file_path / MANIFEST_FILENAME
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"cannot read manifest at {file_path}: {exc}") from exc
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid JSON in manifest {file_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestError(f"manifest {file_path} is not a JSON object")
    version = data.get("schema_version")
    if version != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(f"manifest {file_path} has unsupported schema_version {version!r}")
    return cast(Manifest, data)
