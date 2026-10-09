"""``diff_runs``: structured config + manifest comparison."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from research_kit.core.spine._types import Config
from research_kit.core.spine.config import load_config

from .handle import RunHandle, resolve_run_dir
from .manifest import Manifest

__all__ = ["MANIFEST_DIFF_KEYS", "ConfigChange", "ManifestChange", "RunDiff", "diff_runs"]

MANIFEST_DIFF_KEYS = (
    "python.version",
    "git.sha",
    "git.dirty",
    "seeds",
    "accelerators",
    "platform.system",
    "platform.machine",
)
"""Curated manifest paths compared by default (full package delta needs a flag)."""


@dataclass(frozen=True)
class ConfigChange:
    path: str
    kind: Literal["added", "removed", "changed"]
    old: object | None
    new: object | None


@dataclass(frozen=True)
class ManifestChange:
    key: str
    old: object | None
    new: object | None


@dataclass(frozen=True)
class RunDiff:
    config: list[ConfigChange]
    manifest: list[ManifestChange]

    def render_text(self) -> str:
        """Human-readable rendering (used by ``labkit diff``)."""
        lines = ["# config"]
        if not self.config:
            lines.append("  (identical)")
        for change in self.config:
            if change.kind == "changed":
                lines.append(f"  ~ {change.path}: {change.old!r} -> {change.new!r}")
            elif change.kind == "added":
                lines.append(f"  + {change.path}: {change.new!r}")
            else:
                lines.append(f"  - {change.path}: {change.old!r}")
        lines.append("# manifest")
        if not self.manifest:
            lines.append("  (identical or not comparable)")
        for entry in self.manifest:
            lines.append(f"  ~ {entry.key}: {entry.old!r} -> {entry.new!r}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, object]:
        """Machine-readable rendering (used by ``labkit diff --json``)."""
        return {
            "config": [
                {"path": c.path, "kind": c.kind, "old": c.old, "new": c.new} for c in self.config
            ],
            "manifest": [{"key": c.key, "old": c.old, "new": c.new} for c in self.manifest],
        }


def diff_runs(
    a: str,
    b: str,
    *,
    runs_dir: str | os.PathLike[str] | None = None,
    include_packages: bool = False,
) -> RunDiff:
    """Diff two runs.

    Each of ``a`` / ``b`` may be a config hash, a run directory, or a path to a
    config YAML. A bare config path has no manifest, so the manifest diff is
    skipped for that side (no error).

    Args:
        include_packages: also emit per-package version deltas.

    Raises:
        RunNotFoundError: a hash/dir side cannot be located.
        ConfigError: a config file cannot be read.
    """
    cfg_a, man_a = _load_side(a, runs_dir)
    cfg_b, man_b = _load_side(b, runs_dir)

    config_changes = _diff_config(cfg_a, cfg_b)
    manifest_changes: list[ManifestChange] = []
    if man_a is not None and man_b is not None:
        manifest_changes = _diff_manifest(man_a, man_b, include_packages=include_packages)
    return RunDiff(config=config_changes, manifest=manifest_changes)


def _load_side(ref: str, runs_dir: str | os.PathLike[str] | None) -> tuple[Config, Manifest | None]:
    as_path = Path(ref)
    if as_path.is_file():
        return load_config(as_path), None
    handle = RunHandle(resolve_run_dir(ref, runs_dir=runs_dir))
    return handle.resolved_config, handle.manifest


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten(value, child))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            flat.update(_flatten(value, f"{prefix}[{index}]"))
    else:
        flat[prefix] = node
    return flat


def _diff_config(a: Config, b: Config) -> list[ConfigChange]:
    flat_a, flat_b = _flatten(a), _flatten(b)
    changes: list[ConfigChange] = []
    for key in sorted(set(flat_a) | set(flat_b)):
        if key not in flat_b:
            changes.append(ConfigChange(key, "removed", flat_a[key], None))
        elif key not in flat_a:
            changes.append(ConfigChange(key, "added", None, flat_b[key]))
        elif flat_a[key] != flat_b[key]:
            changes.append(ConfigChange(key, "changed", flat_a[key], flat_b[key]))
    return changes


def _dotted(manifest: Manifest, dotted_key: str) -> Any:
    current: Any = manifest
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _diff_manifest(a: Manifest, b: Manifest, *, include_packages: bool) -> list[ManifestChange]:
    changes: list[ManifestChange] = []
    for key in MANIFEST_DIFF_KEYS:
        value_a, value_b = _dotted(a, key), _dotted(b, key)
        if value_a != value_b:
            changes.append(ManifestChange(key, value_a, value_b))
    if include_packages:
        packages_a, packages_b = a["packages"], b["packages"]
        for name in sorted(set(packages_a) | set(packages_b)):
            if packages_a.get(name) != packages_b.get(name):
                changes.append(
                    ManifestChange(
                        f"packages.{name}",
                        packages_a.get(name),
                        packages_b.get(name),
                    )
                )
    return changes
