"""Read-only access to a finished (or running) run directory."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from research_kit.core.spine._types import Config
from research_kit.core.spine.errors import ConfigError

from .errors import AmbiguousRunError, MetricsError, RunNotFoundError
from .manifest import Manifest, read_manifest
from .run import DEFAULT_RUNS_DIR, RUNS_DIR_ENV

__all__ = ["RESOLVED_CONFIG_FILENAME", "RunHandle", "open_run", "resolve_run_dir"]

RESOLVED_CONFIG_FILENAME = "resolved_config.yaml"
METRICS_FILENAME = "metrics.jsonl"


class RunHandle:
    """A thin reader over one run directory. Downstream modules use this to
    consume provenance without importing :class:`~research_kit.labkit.run.Run`.
    """

    def __init__(self, run_dir: str | os.PathLike[str]) -> None:
        directory = Path(run_dir)
        if not directory.is_dir():
            raise RunNotFoundError(f"not a run directory: {directory}")
        self._dir = directory

    @property
    def dir(self) -> Path:
        return self._dir

    @property
    def manifest(self) -> Manifest:
        """Parsed ``manifest.json``. Raises ManifestError if unreadable."""
        return read_manifest(self._dir)

    @property
    def resolved_config(self) -> Config:
        """Parsed ``resolved_config.yaml``. Raises ConfigError if unreadable."""
        path = self._dir / RESOLVED_CONFIG_FILENAME
        try:
            raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ConfigError(f"cannot read {path}: {exc}") from exc
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"{path} does not contain a mapping")
        return raw

    def iter_metrics(self) -> Iterator[dict[str, object]]:
        """Yield each ``metrics.jsonl`` row in file order.

        Raises:
            MetricsError: a line is not a JSON object.
        """
        path = self._dir / METRICS_FILENAME
        if not path.exists():
            return
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MetricsError(f"{path}:{number} is not valid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise MetricsError(f"{path}:{number} is not a JSON object")
            yield obj


def _runs_root(runs_dir: str | os.PathLike[str] | None) -> Path:
    if runs_dir is not None:
        return Path(runs_dir)
    return Path(os.environ.get(RUNS_DIR_ENV, DEFAULT_RUNS_DIR))


def resolve_run_dir(hash_or_dir: str, *, runs_dir: str | os.PathLike[str] | None = None) -> Path:
    """Resolve a full/short config hash or a path to a run directory.

    Raises:
        RunNotFoundError: nothing matches.
        AmbiguousRunError: a short hash matches multiple run directories.
    """
    as_path = Path(hash_or_dir)
    if as_path.is_dir():
        return as_path

    root = _runs_root(runs_dir)
    exact = root / hash_or_dir
    if exact.is_dir():
        return exact

    if root.is_dir():
        matches = sorted(
            child
            for child in root.iterdir()
            if child.is_dir() and child.name.startswith(hash_or_dir)
        )
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise AmbiguousRunError(f"{hash_or_dir!r} matches {len(matches)} runs under {root}")
    raise RunNotFoundError(f"no run found for {hash_or_dir!r} under {root}")


def open_run(hash_or_dir: str, *, runs_dir: str | os.PathLike[str] | None = None) -> RunHandle:
    """:func:`resolve_run_dir` then wrap it in a :class:`RunHandle`."""
    return RunHandle(resolve_run_dir(hash_or_dir, runs_dir=runs_dir))
