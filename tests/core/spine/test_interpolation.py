"""resolve_config: ${...} grammar (invariant 3: no silent failure)."""

from __future__ import annotations

import pytest

from research_kit.core.spine import ConfigTypeError, InterpolationError, resolve_config


def test_whole_value_reference_copies_node() -> None:
    src = {"base": {"k": [1, 2]}, "alias": "${base}"}
    out = resolve_config(src)
    assert out["alias"] == {"k": [1, 2]}
    out["alias"]["k"].append(3)
    assert out["base"]["k"] == [1, 2]  # copy, not shared


def test_embedded_reference_stringifies_scalar() -> None:
    out = resolve_config({"n": 3, "msg": "count=${n}"})
    assert out["msg"] == "count=3"


def test_embedded_reference_to_nonscalar_raises() -> None:
    with pytest.raises(InterpolationError):
        resolve_config({"lst": [1], "msg": "x=${lst}"})


def test_env_required_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RESEARCH_TEST_VAR", raising=False)
    with pytest.raises(InterpolationError):
        resolve_config({"x": "${env:RESEARCH_TEST_VAR}"})


def test_env_with_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RESEARCH_TEST_VAR", raising=False)
    assert resolve_config({"x": "${env:RESEARCH_TEST_VAR,fallback}"})["x"] == "fallback"


def test_escape_double_dollar_is_literal() -> None:
    assert resolve_config({"x": "$${keep}"})["x"] == "${keep}"


def test_reference_cycle_raises() -> None:
    with pytest.raises(InterpolationError):
        resolve_config({"a": "${b}", "b": "${a}"})


def test_unknown_path_raises_not_empty_string() -> None:
    with pytest.raises(InterpolationError):
        resolve_config({"x": "${does.not.exist}"})


def test_list_index_in_path() -> None:
    out = resolve_config({"items": [10, 20], "first": "${items[0]}"})
    assert out["first"] == 10


def test_multi_hop_reference_chain() -> None:
    out = resolve_config({"a": 1, "b": "${a}", "c": "${b}"})
    assert out["c"] == 1


def test_idempotent_on_plain_config() -> None:
    cfg = {"a": 1, "b": {"c": [1, 2]}}
    assert resolve_config(cfg) == cfg


# --- PHASE 4 hardening: adversarial misuse ---------------------------------


def test_non_mapping_top_level_raises_configtypeerror_immediately() -> None:
    """A caller passing a list/str/None straight to resolve_config (bypassing
    load_config's own check) must get an immediate, correctly-typed error
    instead of a late, confusing InterpolationError."""
    with pytest.raises(ConfigTypeError):
        resolve_config(["a", "b"])  # type: ignore[arg-type]
    with pytest.raises(ConfigTypeError):
        resolve_config("not-a-mapping")  # type: ignore[arg-type]


def test_circular_yaml_anchor_raises_interpolationerror_not_recursionerror() -> None:
    """A self-referencing structure (what PyYAML produces for an anchor like
    ``a: &x {b: *x}``) must raise a typed InterpolationError instead of
    crashing the process with a RecursionError."""
    cyclic: dict[str, object] = {"seed": 1}
    cyclic["self"] = cyclic
    with pytest.raises(InterpolationError, match="circular"):
        resolve_config(cyclic)
