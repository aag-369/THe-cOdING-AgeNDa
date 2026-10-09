"""The spec registry: an explicit, process-global name -> target table.

There is one sanctioned global (``_REGISTRY``). It is explicit, introspectable,
and overridable -- :func:`build` accepts an alternate :class:`Registry` for
tests and embedding. No hidden state, no ``if name == ...`` factory chains.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from .errors import RegistryError, SpecError, UnknownTargetError

__all__ = ["Registry", "Target", "get_registry", "register", "registered"]

Target = type | Callable[..., Any]
"""A registerable: a class (constructed with ``**kwargs``) or a factory callable."""

T = TypeVar("T", bound=Target)

KindName = tuple[str, str]


def _validate_kind_name(kind: object, name: object) -> None:
    # Typed `str` in the public signatures below; validated as `object` here
    # because these are decorator/call arguments any caller can violate at
    # runtime regardless of what mypy sees at the call site.
    for label, value in (("kind", kind), ("name", name)):
        if not isinstance(value, str) or not value or ":" in value:
            raise SpecError(f"invalid {label} {value!r}: expected a non-empty string without ':'")


class Registry:
    """A mapping from ``(kind, name)`` to a target object."""

    def __init__(self) -> None:
        self._table: dict[KindName, Target] = {}

    def register(self, kind: str, name: str, target: Target, *, override: bool = False) -> None:
        """Bind ``(kind, name)`` to ``target``.

        Re-binding the *identical* object is a no-op (so re-importing a module
        is safe).

        Raises:
            SpecError: ``kind`` or ``name`` is empty or contains ``":"``.
            RegistryError: ``(kind, name)`` is already bound to a different
                object and ``override`` is False.
        """
        _validate_kind_name(kind, name)
        key = (kind, name)
        existing = self._table.get(key)
        if existing is not None and existing is not target and not override:
            raise RegistryError(
                f"{kind}:{name} is already registered to {existing!r}; "
                f"pass override=True to replace it"
            )
        self._table[key] = target

    def get(self, kind: str, name: str) -> Target:
        """Return the target bound to ``(kind, name)``.

        Raises:
            UnknownTargetError: no such binding.
        """
        try:
            return self._table[(kind, name)]
        except KeyError:
            raise UnknownTargetError(f"no target registered for {kind!r}:{name!r}") from None

    def contains(self, kind: str, name: str) -> bool:
        """True if ``(kind, name)`` is bound."""
        return (kind, name) in self._table

    def items(self, kind: str | None = None) -> list[tuple[KindName, Target]]:
        """All bindings, optionally filtered to one ``kind``, sorted by key."""
        pairs = sorted(self._table.items(), key=lambda item: item[0])
        if kind is not None:
            pairs = [pair for pair in pairs if pair[0][0] == kind]
        return pairs

    def clear(self) -> None:
        """Remove every binding."""
        self._table.clear()

    def snapshot(self) -> dict[KindName, Target]:
        """A shallow copy of the current bindings (for test save/restore)."""
        return dict(self._table)

    def restore(self, snap: dict[KindName, Target]) -> None:
        """Replace all bindings with ``snap``."""
        self._table = dict(snap)


_REGISTRY = Registry()


def get_registry() -> Registry:
    """Return the process-global registry."""
    return _REGISTRY


def register(kind: str, name: str, *, override: bool = False) -> Callable[[T], T]:
    """Decorator: register the decorated class/function in the global registry.

    Returns the object unchanged (no wrapping).

    Raises:
        SpecError: invalid ``kind``/``name`` (see :meth:`Registry.register`).
        RegistryError: conflicting binding (see :meth:`Registry.register`).
    """

    def decorator(obj: T) -> T:
        _REGISTRY.register(kind, name, obj, override=override)
        return obj

    return decorator


def registered(kind: str | None = None) -> list[KindName]:
    """Sorted ``(kind, name)`` pairs currently in the global registry."""
    return [key for key, _ in _REGISTRY.items(kind)]
