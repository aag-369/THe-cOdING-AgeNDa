"""config_hash / canonicalize (invariant 4: same config => same bytes)."""

from __future__ import annotations

import pytest

from research_kit.core.spine import ConfigTypeError, canonicalize, config_hash, short_hash


def test_dict_key_order_does_not_change_hash() -> None:
    a = {"x": 1, "y": {"p": 2, "q": 3}}
    b = {"y": {"q": 3, "p": 2}, "x": 1}
    assert config_hash(a) == config_hash(b)


def test_list_order_changes_hash() -> None:
    assert config_hash({"v": [1, 2, 3]}) != config_hash({"v": [3, 2, 1]})


def test_int_and_float_hash_differently() -> None:
    assert config_hash({"v": 1}) != config_hash({"v": 1.0})


def test_semantically_equal_nested_configs_match() -> None:
    a: dict[str, object] = {}
    a["b"] = {"c": 1}
    a["a"] = [1, {"z": 2, "y": 1}]
    b: dict[str, object] = {"a": [1, {"y": 1, "z": 2}], "b": {"c": 1}}
    assert config_hash(a) == config_hash(b)


def test_unhashable_type_raises() -> None:
    with pytest.raises(ConfigTypeError):
        config_hash({"v": {1, 2}})  # type: ignore[dict-item]


def test_non_finite_float_raises() -> None:
    with pytest.raises(ConfigTypeError):
        config_hash({"v": float("inf")})


def test_non_str_key_raises() -> None:
    with pytest.raises(ConfigTypeError):
        config_hash({1: "x"})  # type: ignore[dict-item]


def test_canonicalize_is_deterministic_bytes() -> None:
    cfg = {"b": 1, "a": 2}
    first = canonicalize(cfg)
    assert isinstance(first, bytes)
    assert first == canonicalize(cfg)


def test_config_hash_is_64_hex() -> None:
    h = config_hash({"a": 1})
    assert len(h) == 64
    int(h, 16)


def test_short_hash_prefix_and_validation() -> None:
    h = config_hash({"a": 1})
    assert short_hash(h, 16) == h[:16]
    with pytest.raises(ValueError):
        short_hash("abc", 16)
