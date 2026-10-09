"""Exception taxonomy for :mod:`research_kit.core.spine`.

All are subclasses of :class:`SpineError`, itself a
:class:`research_kit.errors.ResearchError`. Every raise site in spine uses one of
these -- never a bare ``ValueError``/``KeyError`` for an anticipated failure.
"""

from __future__ import annotations

from research_kit.errors import ResearchError


class SpineError(ResearchError):
    """Base for every spine failure."""


class SpecError(SpineError):
    """A spec is structurally invalid.

    Raised for: a malformed ``_target`` (missing/empty ``kind`` or ``name``, not
    a ``str``), misuse of a reserved key (``_target``, ``_partial_``,
    ``_imports_``), a non-string mapping key, or an ``_imports_`` value that is
    not a list of importable module paths.
    """


class RegistryError(SpineError):
    """A registration conflicts with an existing, different binding."""


class UnknownTargetError(SpineError):
    """``"kind:name"`` in a spec is not present in the registry."""


class BuildError(SpineError):
    """A registered target raised while being constructed.

    The message carries the structural path to the failing node (e.g.
    ``$.model.encoder.layers[0]``); the original exception is chained via
    ``__cause__``.
    """


class ConfigError(SpineError):
    """A config file cannot be loaded (missing, not a mapping, YAML parse error)."""


class ConfigTypeError(ConfigError):
    """A config contains a value that cannot appear in a config tree.

    Allowed: ``dict`` (str keys), ``list``, ``str``, ``int``, ``float``,
    ``bool``, ``None``. Also raised for non-finite floats (``nan``/``inf``) and
    non-string mapping keys.
    """


class OverrideError(ConfigError):
    """A CLI-style override string is malformed or violates the path rules.

    Raised for: no ``=``; ``PATH`` does not exist and the override is not
    prefixed with ``+``; ``+PATH`` already exists; a list index is out of range;
    traversal passes through a scalar.
    """


class InterpolationError(ConfigError):
    """A ``${...}`` expression cannot be resolved.

    Raised for: unknown reference path, missing required environment variable,
    reference cycle, a non-scalar value used in a string context, or malformed
    expression syntax. Never silently substitutes an empty string.
    """
