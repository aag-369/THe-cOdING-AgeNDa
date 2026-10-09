"""``build``: turn a plain spec tree into live objects.

The spec language, in full:

* A ``dict`` with a ``_target`` key of the form ``"kind:name"`` is *built*:
  every other value is recursively built and passed as a keyword argument to
  the resolved target. If ``_partial_`` is truthy, ``functools.partial`` is
  returned instead of calling the target.
* A ``dict`` with an ``_imports_`` key (``list[str]``) imports each module
  first (populating the registry as a side effect), then drops the key.
* A ``dict`` without ``_target`` is returned as a new ``dict`` with each value
  recursively built.
* A ``list`` is built elementwise.
* Anything else is returned unchanged.

There are no positional arguments and no attribute/${} interpolation here --
interpolation happens earlier, in :mod:`research_kit.core.spine.config`.
"""

from __future__ import annotations

import functools
import importlib
from typing import Any

from .errors import BuildError, SpecError
from .registry import Registry, get_registry

__all__ = ["IMPORTS_KEY", "PARTIAL_KEY", "RESERVED_KEYS", "TARGET_KEY", "build"]

TARGET_KEY = "_target"
PARTIAL_KEY = "_partial_"
IMPORTS_KEY = "_imports_"
RESERVED_KEYS = frozenset({TARGET_KEY, PARTIAL_KEY, IMPORTS_KEY})


def build(spec: object, *, registry: Registry | None = None) -> Any:
    """Recursively construct objects described by ``spec``.

    Args:
        spec: a spec tree (see module docstring). Not mutated.
        registry: registry to resolve ``_target`` against; defaults to the
            process-global registry.

    Returns:
        The constructed object graph. A pure function of ``spec`` and the
        registry's current contents.

    Raises:
        SpecError: malformed ``_target``; a non-string mapping key; a
            non-bool ``_partial_``; an ``_imports_`` that is not a list of
            importable module paths.
        UnknownTargetError: ``"kind:name"`` not registered.
        BuildError: the target raised while being called; the failing spec
            path is in the message and the original exception is chained.
    """
    reg = registry if registry is not None else get_registry()
    return _build(spec, reg, "$", frozenset())


def _build(node: object, reg: Registry, path: str, ancestors: frozenset[int]) -> Any:
    if isinstance(node, (dict, list)):
        if id(node) in ancestors:
            raise SpecError(
                f"{path}: circular reference detected (a YAML anchor referencing "
                f"one of its own ancestors); specs must be finite trees"
            )
        ancestors = ancestors | {id(node)}
    if isinstance(node, dict):
        working = dict(node)
        if IMPORTS_KEY in working:
            _run_imports(working.pop(IMPORTS_KEY), path)
        if TARGET_KEY in working:
            return _build_target(working, reg, path, ancestors)
        result: dict[str, Any] = {}
        for key, value in working.items():
            if not isinstance(key, str):
                raise SpecError(f"{path}: mapping key {key!r} is not a string")
            result[key] = _build(value, reg, f"{path}.{key}", ancestors)
        return result
    if isinstance(node, list):
        return [_build(item, reg, f"{path}[{index}]", ancestors) for index, item in enumerate(node)]
    return node


def _run_imports(value: object, path: str) -> None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SpecError(
            f"{path}: {IMPORTS_KEY} must be a list of module-path strings, got {value!r}"
        )
    for module_path in value:
        try:
            importlib.import_module(module_path)
        except ImportError as exc:
            raise SpecError(
                f"{path}: cannot import {module_path!r} listed in {IMPORTS_KEY}"
            ) from exc


def _build_target(
    working: dict[str, Any], reg: Registry, path: str, ancestors: frozenset[int]
) -> Any:
    target_ref = working.pop(TARGET_KEY)
    if not isinstance(target_ref, str) or target_ref.count(":") != 1:
        raise SpecError(f"{path}: malformed {TARGET_KEY} {target_ref!r}; expected 'kind:name'")
    kind, name = target_ref.split(":")
    if not kind or not name:
        raise SpecError(f"{path}: malformed {TARGET_KEY} {target_ref!r}; expected 'kind:name'")

    partial = working.pop(PARTIAL_KEY, False)
    if not isinstance(partial, bool):
        raise SpecError(f"{path}: {PARTIAL_KEY} must be a bool, got {partial!r}")

    target = reg.get(kind, name)

    kwargs: dict[str, Any] = {}
    for key, value in working.items():
        if not isinstance(key, str):
            raise SpecError(f"{path}: mapping key {key!r} is not a string")
        kwargs[key] = _build(value, reg, f"{path}.{key}", ancestors)

    if partial:
        return functools.partial(target, **kwargs)
    try:
        return target(**kwargs)
    except (SpecError, BuildError):
        raise
    except Exception as exc:
        raise BuildError(f"{path}: constructing {target_ref!r} failed: {exc}") from exc
