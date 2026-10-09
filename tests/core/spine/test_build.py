"""build() behaviour: recursive spec construction from plain dicts."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from research_kit.core.spine import (
    BuildError,
    SpecError,
    UnknownTargetError,
    build,
    register,
)


@pytest.fixture
def _targets() -> None:
    @register("box", "leaf")
    class Leaf:
        def __init__(self, v: int = 0) -> None:
            self.v = v

    @register("box", "node")
    class Node:
        def __init__(self, child: object, tag: str = "") -> None:
            self.child = child
            self.tag = tag

    @register("box", "boom")
    class Boom:
        def __init__(self) -> None:
            raise RuntimeError("cannot construct")


def test_nested_specs_built_recursively(_targets: None) -> None:
    obj = build(
        {
            "_target": "box:node",
            "tag": "root",
            "child": {"_target": "box:leaf", "v": 9},
        }
    )
    assert obj.tag == "root"
    assert obj.child.v == 9


def test_list_of_specs_built_elementwise(_targets: None) -> None:
    out = build([{"_target": "box:leaf", "v": 1}, {"_target": "box:leaf", "v": 2}])
    assert [o.v for o in out] == [1, 2]


def test_dict_without_target_passes_through_children_built(_targets: None) -> None:
    out = build({"a": {"_target": "box:leaf", "v": 3}, "b": 4})
    assert out["b"] == 4
    assert out["a"].v == 3


def test_unknown_target_raises(_targets: None) -> None:
    with pytest.raises(UnknownTargetError):
        build({"_target": "box:missing"})


@pytest.mark.parametrize("bad", ["nocolon", "", ":name", "kind:", 123])
def test_malformed_target_raises_specerror(bad: object) -> None:
    with pytest.raises(SpecError):
        build({"_target": bad})


def test_constructor_failure_raises_builderror_with_path(_targets: None) -> None:
    with pytest.raises(BuildError) as ei:
        build({"_target": "box:node", "child": {"_target": "box:boom"}})
    assert "child" in str(ei.value)
    assert isinstance(ei.value.__cause__, RuntimeError)


def test_partial_returns_callable_not_instance(_targets: None) -> None:
    made = build({"_target": "box:leaf", "_partial_": True})
    assert callable(made)
    assert made(v=7).v == 7


def test_reserved_keys_not_forwarded(_targets: None) -> None:
    # box:leaf.__init__ has no **kwargs; a leaked _partial_ kwarg would raise.
    obj = build({"_target": "box:leaf", "_partial_": False, "v": 1})
    assert obj.v == 1


def test_imports_key_triggers_registration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = tmp_path / "late_mod.py"
    mod.write_text(
        textwrap.dedent(
            """
            from research_kit.core.spine import register

            @register("late", "thing")
            class Thing:
                def __init__(self, x: int) -> None:
                    self.x = x
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("late_mod", None)

    obj = build({"_imports_": ["late_mod"], "_target": "late:thing", "x": 2})
    assert obj.x == 2


def test_bad_imports_value_raises_specerror() -> None:
    with pytest.raises(SpecError):
        build({"_imports_": "not-a-list", "_target": "x:y"})


def test_circular_spec_raises_specerror_not_recursionerror(_targets: None) -> None:
    """PHASE 4 hardening: a self-referencing spec (the same shape produced by
    a YAML anchor like ``a: &x {b: *x}``) must raise a typed SpecError instead
    of crashing the process with a RecursionError."""
    cyclic: dict[str, object] = {"_target": "box:node", "tag": "root"}
    cyclic["child"] = cyclic
    with pytest.raises(SpecError, match="circular"):
        build(cyclic)
