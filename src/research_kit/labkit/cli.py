"""``labkit`` command line -- thin wrappers over :mod:`research_kit.labkit`.

::

    labkit replay HASH [--strict] [--runs-dir DIR] [-o OVERRIDE ]...
    labkit diff   A B  [--json] [--packages] [--runs-dir DIR]
    labkit show   HASH [--json] [--metrics] [--runs-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys

import yaml

from .diff import diff_runs
from .errors import LabkitError, RunNotFoundError
from .handle import open_run
from .replay import replay

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser (exposed for testing/inspection)."""
    parser = argparse.ArgumentParser(prog="labkit", description="Inspect labkit runs.")
    sub = parser.add_subparsers(dest="command", required=True)

    replay_parser = sub.add_parser("replay", help="re-run a recorded run")
    replay_parser.add_argument("hash", help="config hash (short or full) or run dir")
    replay_parser.add_argument("--strict", action="store_true")
    replay_parser.add_argument("--runs-dir", default=None, dest="runs_dir")
    replay_parser.add_argument("-o", "--override", action="append", default=[], dest="overrides")

    diff_parser = sub.add_parser("diff", help="diff two runs")
    diff_parser.add_argument("a")
    diff_parser.add_argument("b")
    diff_parser.add_argument("--json", action="store_true", dest="as_json")
    diff_parser.add_argument("--packages", action="store_true", dest="packages")
    diff_parser.add_argument("--runs-dir", default=None, dest="runs_dir")

    show_parser = sub.add_parser("show", help="print a run's manifest and config")
    show_parser.add_argument("hash", help="config hash (short or full) or run dir")
    show_parser.add_argument("--json", action="store_true", dest="as_json")
    show_parser.add_argument("--metrics", action="store_true")
    show_parser.add_argument("--runs-dir", default=None, dest="runs_dir")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code:

    * ``replay``: the child process return code.
    * ``diff`` / ``show``: 0 on success, 1 on a handled ``LabkitError``.
    * argparse errors exit 2.

    ``RunNotFoundError`` (and its ``AmbiguousRunError`` subclass) propagate --
    an unresolvable run reference is a caller error, not a clean exit.
    """
    args = build_parser().parse_args(argv)
    try:
        if args.command == "replay":
            return _cmd_replay(args)
        if args.command == "diff":
            return _cmd_diff(args)
        if args.command == "show":
            return _cmd_show(args)
    except RunNotFoundError:
        raise
    except LabkitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2  # pragma: no cover - argparse enforces a valid subcommand


def _cmd_replay(args: argparse.Namespace) -> int:
    result = replay(
        args.hash,
        runs_dir=args.runs_dir,
        strict=args.strict,
        extra_overrides=list(args.overrides),
    )
    if result.mismatches:
        for line in result.mismatches:
            print(f"warning: {line}", file=sys.stderr)
    print(f"replay dir: {result.replay_dir}")
    return result.returncode


def _cmd_diff(args: argparse.Namespace) -> int:
    run_diff = diff_runs(args.a, args.b, runs_dir=args.runs_dir, include_packages=args.packages)
    if args.as_json:
        print(json.dumps(run_diff.as_dict(), indent=2, sort_keys=True))
    else:
        print(run_diff.render_text())
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    handle = open_run(args.hash, runs_dir=args.runs_dir)
    manifest = handle.manifest
    config = handle.resolved_config

    if args.as_json:
        payload: dict[str, object] = {
            "dir": str(handle.dir),
            "manifest": manifest,
            "config": config,
        }
        if args.metrics:
            payload["metrics"] = list(handle.iter_metrics())
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0

    print(f"dir: {handle.dir}")
    print(f"status: {manifest['status']}")
    print(f"config_hash: {manifest['config_hash']}")
    print(f"start_time: {manifest['start_time']}")
    print("config:")
    print(yaml.safe_dump(config, sort_keys=True, default_flow_style=False).rstrip())
    if args.metrics:
        print("metrics:")
        for row in handle.iter_metrics():
            print(f"  {json.dumps(row, sort_keys=True)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
