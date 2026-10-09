"""Config loading pipeline: YAML -> overrides -> ${} interpolation.

Order matters and is fixed: :func:`load_config` then :func:`apply_overrides`
then :func:`resolve_config`. Overrides are applied to raw values (so an
override may itself introduce a ``${...}`` expression that is resolved
afterwards). :func:`load_resolved` runs all three and also returns the hash.

Interpolation grammar (exhaustive)::

    ${a.b[0].c}     reference another node by path from the config root
    ${env:VAR}      environment variable, required
    ${env:VAR,def}  environment variable with a literal default
    $${...}         a literal "${...}"

A whole-value reference (``x: ${a.b}``) copies the referenced node, whatever
its type. A reference embedded in a larger string (``x: "n=${a.b}"``) requires
the referenced value to be a scalar.
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from ._types import Config
from .errors import ConfigError, ConfigTypeError, InterpolationError, OverrideError
from .hashing import config_hash, ensure_config_types, ensure_config_value
from .paths import PathToken, get_path, has_path, parse_path, set_path

__all__ = [
    "apply_overrides",
    "load_config",
    "load_resolved",
    "resolve_config",
]

_EXPR_RE = re.compile(r"\$\{([^{}]*)\}")
_ESCAPE_SENTINEL = "\x00RESEARCH_DOLLAR_BRACE\x00"


# --------------------------------------------------------------------------- #
# loading                                                                     #
# --------------------------------------------------------------------------- #
def load_config(path: str | os.PathLike[str]) -> Config:
    """Read a YAML file into a plain ``dict`` (``yaml.safe_load``).

    Does not apply overrides or resolve interpolation.

    Raises:
        ConfigError: the file is missing, its top level is not a mapping, or
            YAML parsing fails.
        ConfigTypeError: a value has a type that cannot appear in a config
            tree, or is a non-finite float, or a mapping key is not a string.
    """
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {file_path}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read config file {file_path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {file_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(
            f"config top level must be a mapping, got {type(data).__name__} in {file_path}"
        )
    ensure_config_types(data)
    return data


# --------------------------------------------------------------------------- #
# overrides                                                                   #
# --------------------------------------------------------------------------- #
def apply_overrides(cfg: Config, overrides: Sequence[str]) -> Config:
    """Return a deep copy of ``cfg`` with dotted-path overrides applied.

    Each override is ``PATH=VALUE`` (replace an existing node) or
    ``+PATH=VALUE`` (create a new node). ``VALUE`` is parsed as YAML, so
    ``lr=0.1`` yields a float and ``tags=[a,b]`` a list. ``PATH`` supports
    ``a.b[0].c``.

    Raises:
        OverrideError: no ``=``; ``PATH`` missing without ``+``; ``+PATH``
            already exists; list index out of range; traversal through a
            scalar.
        ConfigTypeError: ``VALUE`` parses to a disallowed type.
    """
    result = copy.deepcopy(cfg)
    for raw in overrides:
        create = raw.startswith("+")
        body = raw[1:] if create else raw
        if "=" not in body:
            raise OverrideError(f"malformed override {raw!r}: expected PATH=VALUE")
        path_str, _, value_str = body.partition("=")
        path_str = path_str.strip()
        if not path_str:
            raise OverrideError(f"malformed override {raw!r}: empty path")
        try:
            tokens: list[PathToken] = parse_path(path_str)
        except ValueError as exc:
            raise OverrideError(f"malformed override path in {raw!r}: {exc}") from exc

        value = _parse_override_value(value_str, raw)
        exists = has_path(result, tokens)
        if create and exists:
            raise OverrideError(f"+{path_str} already exists; drop the '+' to replace it")
        if not create and not exists:
            raise OverrideError(f"{path_str} does not exist; prefix with '+' to create it")
        try:
            set_path(result, tokens, value, create=create)
        except (KeyError, IndexError, TypeError) as exc:
            raise OverrideError(f"cannot apply override {raw!r}: {exc}") from exc
    return result


def _parse_override_value(value_str: str, raw: str) -> Any:
    try:
        value = yaml.safe_load(value_str)
    except yaml.YAMLError as exc:
        raise OverrideError(f"cannot parse value in override {raw!r}: {exc}") from exc
    try:
        ensure_config_value(value)
    except ConfigTypeError as exc:
        raise ConfigTypeError(f"override {raw!r}: {exc}") from exc
    return value


# --------------------------------------------------------------------------- #
# interpolation                                                               #
# --------------------------------------------------------------------------- #
def resolve_config(cfg: Config) -> Config:
    """Return a deep copy of ``cfg`` with every ``${...}`` expression expanded.

    Idempotent on a config that contains no expressions.

    Raises:
        ConfigTypeError: ``cfg`` is not a mapping at the top level.
        InterpolationError: unknown reference path, missing required env var,
            reference cycle, non-scalar used in a string context, malformed
            expression, or a circular reference (e.g. a self-referencing YAML
            anchor).
    """
    if not isinstance(cfg, dict):
        raise ConfigTypeError(f"config must be a mapping, got {type(cfg).__name__}")
    resolver = _Resolver(copy.deepcopy(cfg))
    resolved = resolver.resolve_node(resolver.root, ())
    assert isinstance(resolved, dict)  # a dict node always resolves to a dict
    return resolved


class _Resolver:
    def __init__(self, root: Config) -> None:
        self.root = root

    def resolve_node(
        self,
        node: Any,
        trail: tuple[tuple[PathToken, ...], ...],
        ancestors: frozenset[int] = frozenset(),
    ) -> Any:
        if isinstance(node, (dict, list)):
            if id(node) in ancestors:
                raise InterpolationError(
                    "circular reference detected (a YAML anchor referencing "
                    "one of its own ancestors); configs must be finite trees"
                )
            ancestors = ancestors | {id(node)}
        if isinstance(node, dict):
            return {key: self.resolve_node(value, trail, ancestors) for key, value in node.items()}
        if isinstance(node, list):
            return [self.resolve_node(item, trail, ancestors) for item in node]
        if isinstance(node, str):
            return self._resolve_string(node, trail)
        return node

    def _resolve_string(self, text: str, trail: tuple[tuple[PathToken, ...], ...]) -> Any:
        if "${" not in text:
            return text

        whole = _EXPR_RE.fullmatch(text)
        if whole is not None and "$${" not in text:
            value = self._resolve_expr(whole.group(1), trail)
            return self.resolve_node(value, trail)

        def replace(match: re.Match[str]) -> str:
            resolved = self.resolve_node(self._resolve_expr(match.group(1), trail), trail)
            if isinstance(resolved, (dict, list)):
                raise InterpolationError(
                    f"cannot interpolate non-scalar value for ${{{match.group(1)}}} into a string"
                )
            return _scalar_to_str(resolved)

        protected = text.replace("$${", _ESCAPE_SENTINEL)
        substituted = _EXPR_RE.sub(replace, protected)
        return substituted.replace(_ESCAPE_SENTINEL, "${")

    def _resolve_expr(self, expr: str, trail: tuple[tuple[PathToken, ...], ...]) -> Any:
        expr = expr.strip()
        if expr.startswith("env:"):
            return _resolve_env(expr[len("env:") :])

        try:
            tokens = parse_path(expr)
        except ValueError as exc:
            raise InterpolationError(
                f"malformed interpolation expression ${{{expr}}}: {exc}"
            ) from exc

        key = tuple(tokens)
        if key in trail:
            chain = " -> ".join(_format_key(item) for item in (*trail, key))
            raise InterpolationError(f"interpolation reference cycle: {chain}")
        try:
            referenced = get_path(self.root, tokens)
        except (KeyError, IndexError, TypeError) as exc:
            raise InterpolationError(
                f"interpolation ${{{expr}}} refers to a path that does not exist"
            ) from exc
        return self.resolve_node(referenced, (*trail, key))


def _resolve_env(rest: str) -> str:
    if "," in rest:
        name, default = rest.split(",", 1)
        return os.environ.get(name.strip(), default)
    name = rest.strip()
    try:
        return os.environ[name]
    except KeyError:
        raise InterpolationError(
            f"environment variable {name!r} is not set and no default was given"
        ) from None


def _scalar_to_str(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    return str(value)


def _format_key(key: tuple[PathToken, ...]) -> str:
    out = ""
    for token in key:
        if isinstance(token, int):
            out += f"[{token}]"
        else:
            out = f"{out}.{token}" if out else str(token)
    return out


# --------------------------------------------------------------------------- #
# combined                                                                    #
# --------------------------------------------------------------------------- #
def load_resolved(
    path: str | os.PathLike[str], overrides: Sequence[str] = ()
) -> tuple[Config, str]:
    """``load_config`` -> ``apply_overrides`` -> ``resolve_config``.

    Returns:
        ``(resolved_config, config_hash)``.

    Raises:
        Whatever the three stages raise (``ConfigError`` and subclasses).
    """
    cfg = load_config(path)
    cfg = apply_overrides(cfg, overrides)
    cfg = resolve_config(cfg)
    return cfg, config_hash(cfg)
