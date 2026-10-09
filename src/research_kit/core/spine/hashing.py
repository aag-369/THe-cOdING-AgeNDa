"""Stable, order-independent hashing of a resolved config.

The contract that the whole kit's determinism guarantee rests on: two configs
that are equal as trees -- regardless of dict key insertion order -- produce
the same hash. List order **is** significant. ``int`` ``1`` and ``float``
``1.0`` are distinct and hash differently.
"""

from __future__ import annotations

import json
import math
from hashlib import blake2b

from ._types import Config
from .errors import ConfigTypeError

__all__ = [
    "DIGEST_SIZE",
    "SHORT_LEN",
    "canonicalize",
    "config_hash",
    "ensure_config_types",
    "ensure_config_value",
    "short_hash",
]

DIGEST_SIZE = 32
"""blake2b digest size in bytes -> 64 hex characters."""

SHORT_LEN = 16
"""Default prefix length for run-directory names."""


def ensure_config_value(
    value: object, path: str = "$", *, _ancestors: frozenset[int] = frozenset()
) -> None:
    """Validate a single value (of any type) against the config type rules.

    Raises:
        ConfigTypeError: disallowed type, non-finite float, non-string
            mapping key, or a circular reference (typically from a
            self-referencing YAML anchor) anywhere in ``value``.
    """
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigTypeError(f"{path}: non-finite float {value!r}")
        return
    if value is None or isinstance(value, str):
        return
    if isinstance(value, (dict, list)):
        if id(value) in _ancestors:
            raise ConfigTypeError(
                f"{path}: circular reference detected (a YAML anchor referencing "
                f"one of its own ancestors); configs must be finite trees"
            )
        nested = _ancestors | {id(value)}
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ConfigTypeError(f"{path}: mapping key {key!r} is not a string")
                ensure_config_value(item, f"{path}.{key}", _ancestors=nested)
        else:
            for index, item in enumerate(value):
                ensure_config_value(item, f"{path}[{index}]", _ancestors=nested)
        return
    raise ConfigTypeError(
        f"{path}: value of type {type(value).__name__} is not allowed in a config"
    )


def ensure_config_types(cfg: Config) -> None:
    """Validate a whole config tree. Raises ConfigTypeError (see
    :func:`ensure_config_value`)."""
    if not isinstance(cfg, dict):
        raise ConfigTypeError(f"config must be a mapping, got {type(cfg).__name__}")
    ensure_config_value(cfg)


def canonicalize(cfg: Config) -> bytes:
    """Return the canonical UTF-8 JSON encoding used as hash input.

    Rules: dict keys recursively sorted; list order preserved; separators
    ``(",", ":")``; floats via ``repr`` (round-trippable and platform-stable
    on CPython 3.11).

    Raises:
        ConfigTypeError: a value whose type cannot be canonicalized, a
            non-finite float, or a non-string dict key.
    """
    ensure_config_types(cfg)
    return json.dumps(
        cfg,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def config_hash(cfg: Config) -> str:
    """Return the 64-character blake2b hex digest of ``canonicalize(cfg)``.

    ``cfg`` is expected to be fully resolved (no ``${...}``); this function
    does not resolve it.

    Raises:
        ConfigTypeError: see :func:`canonicalize`.
    """
    return blake2b(canonicalize(cfg), digest_size=DIGEST_SIZE).hexdigest()


def short_hash(digest: str, length: int = SHORT_LEN) -> str:
    """Return the first ``length`` characters of ``digest``.

    Raises:
        ValueError: ``digest`` is shorter than ``length`` or ``length`` <= 0.
    """
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if len(digest) < length:
        raise ValueError(f"digest {digest!r} is shorter than {length} characters")
    return digest[:length]
