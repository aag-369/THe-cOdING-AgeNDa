"""Dotted-path traversal for config trees.

A *path* is a string like ``a.b[0].c``: dot-separated keys with optional
bracketed non-negative integer indices for list positions. Used by both the
override machinery and the interpolation resolver so the two share one grammar.
"""

from __future__ import annotations

import re
from typing import Any

from ._types import Config

__all__ = ["PathToken", "get_path", "has_path", "parse_path", "set_path"]

PathToken = str | int
"""A single step: a ``str`` mapping key or an ``int`` list index."""

_SEGMENT_RE = re.compile(r"([^.\[\]]*)((?:\[\d+\])*)\Z")
_INDEX_RE = re.compile(r"\[(\d+)\]")


def parse_path(path: str) -> list[PathToken]:
    """Split ``path`` into an ordered list of string keys and integer indices.

    ``"a.b[0].c"`` -> ``["a", "b", 0, "c"]``.

    Raises:
        ValueError: empty path, empty segment, malformed bracket, or a
            non-integer / negative index.
    """
    if not path or not path.strip():
        raise ValueError("empty path")
    tokens: list[PathToken] = []
    for segment in path.split("."):
        if segment == "":
            raise ValueError(f"empty segment in path {path!r}")
        match = _SEGMENT_RE.fullmatch(segment)
        if match is None:
            raise ValueError(f"malformed path segment {segment!r} in {path!r}")
        key, index_part = match.group(1), match.group(2)
        if key:
            tokens.append(key)
        elif not index_part:
            raise ValueError(f"empty segment in path {path!r}")
        for index_match in _INDEX_RE.finditer(index_part):
            tokens.append(int(index_match.group(1)))
    if not tokens:
        raise ValueError(f"path {path!r} resolved to no tokens")
    return tokens


def get_path(root: Config, tokens: list[PathToken]) -> Any:
    """Return the value at ``tokens`` within ``root``.

    Raises:
        KeyError: a mapping key is absent.
        IndexError: a list index is out of range.
        TypeError: traversal passes through a non-container.
    """
    current: Any = root
    for depth, token in enumerate(tokens):
        if isinstance(token, bool):  # defensive: bool is an int subclass
            raise TypeError(f"path token {token!r} at position {depth} is a bool")
        if isinstance(token, int):
            if not isinstance(current, list):
                raise TypeError(f"cannot index non-list with [{token}] at position {depth}")
            current = current[token]
        else:
            if not isinstance(current, dict):
                raise TypeError(f"cannot look up key {token!r} in non-mapping at position {depth}")
            current = current[token]
    return current


def set_path(root: Config, tokens: list[PathToken], value: Any, *, create: bool) -> None:
    """Set ``value`` at ``tokens`` within ``root`` in place.

    With ``create=False`` every intermediate and the final node must already
    exist (replace semantics). With ``create=True`` missing intermediate
    mappings are created and the final key must **not** already exist.

    Raises:
        KeyError: intermediate missing and ``create`` is False; or final key
            already present and ``create`` is True.
        IndexError: a list index is out of range (indices are never created).
        TypeError: traversal passes through a scalar.
    """
    if not tokens:
        raise ValueError("empty path")
    *parents, last = tokens
    current: Any = root
    for depth, token in enumerate(parents):
        if isinstance(token, int) and not isinstance(token, bool):
            if not isinstance(current, list):
                raise TypeError(f"cannot index non-list at position {depth}")
            current = current[token]
        else:
            if not isinstance(current, dict):
                raise TypeError(f"cannot descend into non-mapping at position {depth}")
            if token not in current:
                if not create:
                    raise KeyError(f"intermediate key {token!r} does not exist")
                current[token] = {}
            current = current[token]

    if isinstance(last, int) and not isinstance(last, bool):
        if not isinstance(current, list):
            raise TypeError("cannot assign by index into a non-list")
        current[last] = value  # raises IndexError if out of range; never extends
        return
    if not isinstance(current, dict):
        raise TypeError("cannot assign a key into a non-mapping")
    exists = last in current
    if create and exists:
        raise KeyError(f"{last!r} already exists")
    if not create and not exists:
        raise KeyError(f"{last!r} does not exist")
    current[last] = value


def has_path(root: Config, tokens: list[PathToken]) -> bool:
    """True if ``tokens`` resolves to an existing node in ``root``."""
    try:
        get_path(root, tokens)
    except (KeyError, IndexError, TypeError):
        return False
    return True
