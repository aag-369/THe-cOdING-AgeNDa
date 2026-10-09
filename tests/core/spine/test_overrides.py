"""apply_overrides: CLI-style dotted-path overrides."""

from __future__ import annotations

import pytest

from research_kit.core.spine import (
    ConfigTypeError,
    OverrideError,
    apply_overrides,
    resolve_config,
)


def _cfg() -> dict[str, object]:
    return {"optim": {"lr": 0.1, "name": "sgd"}, "layers": [{"w": 1}, {"w": 2}], "seed": 0}


def test_replace_leaf_with_yaml_typed_value() -> None:
    out = apply_overrides(_cfg(), ["optim.lr=0.5"])
    assert out["optim"]["lr"] == 0.5
    assert isinstance(out["optim"]["lr"], float)


def test_replace_subtree_with_mapping() -> None:
    out = apply_overrides(_cfg(), ["optim={name: adam, lr: 0.2}"])
    assert out["optim"] == {"name": "adam", "lr": 0.2}


def test_create_requires_plus() -> None:
    with pytest.raises(OverrideError):
        apply_overrides(_cfg(), ["optim.momentum=0.9"])
    out = apply_overrides(_cfg(), ["+optim.momentum=0.9"])
    assert out["optim"]["momentum"] == 0.9


def test_plus_on_existing_path_raises() -> None:
    with pytest.raises(OverrideError):
        apply_overrides(_cfg(), ["+optim.lr=0.9"])


def test_list_index_override_and_out_of_range() -> None:
    out = apply_overrides(_cfg(), ["layers[0].w=99"])
    assert out["layers"][0]["w"] == 99
    with pytest.raises(OverrideError):
        apply_overrides(_cfg(), ["layers[5].w=1"])


def test_traverse_through_scalar_raises() -> None:
    with pytest.raises(OverrideError):
        apply_overrides(_cfg(), ["seed.nested=1"])


def test_malformed_override_string_raises() -> None:
    with pytest.raises(OverrideError):
        apply_overrides(_cfg(), ["optim.lr"])


def test_value_type_validated() -> None:
    with pytest.raises(ConfigTypeError):
        apply_overrides(_cfg(), ["optim.lr=2021-01-01"])


def test_overrides_applied_before_interpolation() -> None:
    cfg = {"a": 1, "b": 2}
    overridden = apply_overrides(cfg, ["b=${a}"])
    assert resolve_config(overridden)["b"] == 1


def test_original_config_not_mutated() -> None:
    cfg = _cfg()
    apply_overrides(cfg, ["optim.lr=0.9", "+optim.x=1"])
    assert cfg["optim"] == {"lr": 0.1, "name": "sgd"}
