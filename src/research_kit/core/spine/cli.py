"""``research-spine`` command line -- a thin wrapper over the library.

Subcommands:

``resolve CONFIG [OVERRIDE ...]``
    Load, override, and interpolate a config, then print it (YAML by default,
    ``--json`` for JSON) followed by ``hash:`` and ``short:`` lines.
    ``--hash-only`` prints just the full hash.
"""

from __future__ import annotations

import argparse
import json
import sys

import yaml

from .config import load_resolved
from .errors import SpineError
from .hashing import short_hash

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser (exposed for testing/inspection)."""
    parser = argparse.ArgumentParser(
        prog="research-spine",
        description="Inspect research-kit configs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    resolve = sub.add_parser("resolve", help="resolve a config and print it with its hash")
    resolve.add_argument("config", help="path to a YAML config file")
    resolve.add_argument(
        "overrides",
        nargs="*",
        metavar="PATH=VALUE",
        help="dotted-path overrides applied before interpolation",
    )
    resolve.add_argument("--json", action="store_true", dest="as_json")
    resolve.add_argument(
        "--hash-only", action="store_true", dest="hash_only", help="print only the hash"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code (0 success, 1 on a handled
    ``SpineError``, 2 on argument errors via argparse).

    All real work is delegated to
    :func:`research_kit.core.spine.config.load_resolved` and
    :func:`research_kit.core.spine.hashing.config_hash`.
    """
    args = build_parser().parse_args(argv)
    if args.command == "resolve":
        return _cmd_resolve(args)
    return 2  # pragma: no cover - argparse enforces a valid subcommand


def _cmd_resolve(args: argparse.Namespace) -> int:
    try:
        cfg, digest = load_resolved(args.config, list(args.overrides))
    except SpineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.hash_only:
        print(digest)
        return 0

    if args.as_json:
        print(json.dumps(cfg, indent=2, sort_keys=True))
    else:
        print(yaml.safe_dump(cfg, sort_keys=True, default_flow_style=False).rstrip())
    print(f"hash: {digest}")
    print(f"short: {short_hash(digest)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
