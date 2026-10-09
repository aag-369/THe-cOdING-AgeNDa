"""load_config: YAML -> plain dict, with strict typing."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_kit.core.spine import ConfigError, ConfigTypeError, load_config


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "c.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_load_returns_plain_dict(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, "a: 1\nb:\n  c: [1, 2]\n"))
    assert cfg == {"a": 1, "b": {"c": [1, 2]}}
    assert type(cfg) is dict


def test_missing_file_raises_configerror(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")


def test_non_mapping_top_level_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, "- 1\n- 2\n"))


def test_yaml_parse_error_raises_configerror(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, "a: [1, 2\n"))


def test_date_value_rejected_as_configtypeerror(tmp_path: Path) -> None:
    with pytest.raises(ConfigTypeError):
        load_config(_write(tmp_path, "when: 2021-01-01\n"))


def test_non_finite_float_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigTypeError):
        load_config(_write(tmp_path, "x: .nan\n"))


def test_circular_yaml_anchor_raises_configtypeerror_not_recursionerror(
    tmp_path: Path,
) -> None:
    """PHASE 4 hardening: PyYAML's safe_load legitimately constructs circular
    references for a self-referencing anchor (``a: &x {b: *x}``). Loading such
    a file must raise a typed ConfigTypeError, never crash the process with a
    RecursionError."""
    with pytest.raises(ConfigTypeError, match="circular"):
        load_config(_write(tmp_path, "a: &anchor\n  b: *anchor\n"))
