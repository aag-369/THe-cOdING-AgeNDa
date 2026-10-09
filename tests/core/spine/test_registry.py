"""Registry behaviour (invariant 1: everything is registered, no clobber)."""

from __future__ import annotations

import pytest

from research_kit.core.spine import (
    Registry,
    RegistryError,
    SpecError,
    UnknownTargetError,
    build,
    register,
    registered,
)


def test_register_decorator_then_build_roundtrip() -> None:
    @register("widget", "adder")
    class Adder:
        def __init__(self, base: int) -> None:
            self.base = base

    obj = build({"_target": "widget:adder", "base": 5})
    assert isinstance(obj, Adder)
    assert obj.base == 5


def test_register_returns_object_unchanged() -> None:
    def factory() -> int:
        return 1

    assert register("fn", "factory")(factory) is factory


def test_duplicate_registration_different_object_raises() -> None:
    @register("k", "n")
    class A:
        pass

    with pytest.raises(RegistryError):

        @register("k", "n")
        class B:
            pass


def test_reregister_identical_object_is_idempotent() -> None:
    class A:
        pass

    register("k", "n")(A)
    register("k", "n")(A)  # must not raise
    assert ("k", "n") in registered()


def test_override_flag_replaces_binding() -> None:
    @register("k", "n")
    class A:
        pass

    @register("k", "n", override=True)
    class B:
        pass

    assert build({"_target": "k:n"}).__class__ is B


@pytest.mark.parametrize("kind,name", [("", "n"), ("k", ""), ("k:x", "n"), ("k", "n:x")])
def test_invalid_kind_or_name_raises_specerror(kind: str, name: str) -> None:
    with pytest.raises(SpecError):
        register(kind, name)(lambda: None)


def test_non_string_kind_or_name_raises_specerror_not_typeerror() -> None:
    """PHASE 4 hardening: a caller ignoring the ``str`` type hint (e.g. an int
    kind from a dynamically-built spec) must get a typed SpecError, not a raw
    TypeError from the internal ':' membership check."""
    with pytest.raises(SpecError):
        register(123, "name")(lambda: None)  # type: ignore[arg-type]
    with pytest.raises(SpecError):
        register("kind", None)(lambda: None)  # type: ignore[arg-type]


def test_registered_is_sorted_pairs() -> None:
    register("b", "2")(lambda: None)
    register("a", "1")(lambda: None)
    pairs = registered()
    assert pairs == sorted(pairs)
    assert ("a", "1") in pairs and ("b", "2") in pairs


def test_isolated_registry_does_not_touch_global() -> None:
    local = Registry()
    local.register("k", "n", lambda: "local")
    assert ("k", "n") not in registered()
    with pytest.raises(UnknownTargetError):
        build({"_target": "k:n"})
    assert build({"_target": "k:n"}, registry=local) == "local"
