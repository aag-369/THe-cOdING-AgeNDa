"""core.spine: spec-driven construction and config handling.

Everything constructible in the kit is registered here and built from a plain
dict/YAML spec. No heavyweight imports occur at import time.
"""

from __future__ import annotations

from .build import IMPORTS_KEY, PARTIAL_KEY, RESERVED_KEYS, TARGET_KEY, build
from .config import apply_overrides, load_config, load_resolved, resolve_config
from .errors import (
    BuildError,
    ConfigError,
    ConfigTypeError,
    InterpolationError,
    OverrideError,
    RegistryError,
    SpecError,
    SpineError,
    UnknownTargetError,
)
from .hashing import canonicalize, config_hash, short_hash
from .registry import Registry, Target, get_registry, register, registered

__all__ = [
    "IMPORTS_KEY",
    "PARTIAL_KEY",
    "RESERVED_KEYS",
    "TARGET_KEY",
    "BuildError",
    "ConfigError",
    "ConfigTypeError",
    "InterpolationError",
    "OverrideError",
    # construction
    "Registry",
    "RegistryError",
    "SpecError",
    # errors
    "SpineError",
    "Target",
    "UnknownTargetError",
    "apply_overrides",
    "build",
    "canonicalize",
    "config_hash",
    "get_registry",
    # config
    "load_config",
    "load_resolved",
    "register",
    "registered",
    "resolve_config",
    "short_hash",
]
