"""Shared type aliases for spine.

A *config* is a plain JSON-compatible tree: nested ``dict`` (str keys) and
``list``, with leaves drawn from ``str | int | float | bool | None``. There is
exactly one representation everywhere in the kit -- a ``dict``. No dataclass
mirrors, no config objects.
"""

from __future__ import annotations

from typing import TypeAlias

ConfigValue: TypeAlias = (
    "dict[str, ConfigValue] | list[ConfigValue] | str | int | float | bool | None"
)
"""Any value that may legally appear inside a config tree."""

Config: TypeAlias = dict[str, "ConfigValue"]
"""The top level of a config is always a string-keyed mapping."""
