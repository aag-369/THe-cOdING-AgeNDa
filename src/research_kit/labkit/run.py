"""``Run``: the context manager that turns a config into a provenance-tracked
run directory.

::

    with Run(cfg) as run:
        run.seed(0)
        run.declare_input("data/train.npz")
        for step in range(10):
            run.log(step=step, loss=...)

On ``__enter__`` it materializes ``runs/<config_hash>/`` containing
``resolved_config.yaml``, ``manifest.json`` (status ``running``),
``metrics.jsonl``, ``logs/``, and ``artifacts/``. On ``__exit__`` it finalizes
the manifest (``completed`` or ``failed`` with a traceback) and flushes
metrics. It never suppresses an exception.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import os
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, TextIO

import yaml

from research_kit.core.spine._types import Config
from research_kit.core.spine.config import resolve_config
from research_kit.core.spine.hashing import config_hash

from .checksums import file_checksum, tree_checksum
from .errors import GitError, InputError, RunExistsError, RunStateError
from .manifest import InputRecord, Manifest, new_manifest, utc_now, write_manifest
from .metrics import MetricsWriter
from .seeding import SeedReport, seed_everything

__all__ = ["DEFAULT_RUNS_DIR", "FORCE_RUN_DIR_ENV", "RUNS_DIR_ENV", "Run"]

RUNS_DIR_ENV = "RESEARCH_RUNS_DIR"
"""If set and no ``runs_dir`` argument is given, used as the runs root."""

FORCE_RUN_DIR_ENV = "RESEARCH_FORCE_RUN_DIR"
"""If set, used verbatim as the run directory (skips the ``<hash>`` join).
Set by :func:`research_kit.labkit.replay.replay` for its child process."""

DEFAULT_RUNS_DIR = "runs"
RESOLVED_CONFIG_FILENAME = "resolved_config.yaml"
METRICS_FILENAME = "metrics.jsonl"

OnExist = Literal["error", "resume", "new_attempt"]
_State = Literal["created", "active", "closed"]


class _Tee:
    """Write-through stream: everything goes to the original and a sink file."""

    def __init__(self, original: TextIO, sink: TextIO) -> None:
        self._original = original
        self._sink = sink

    def write(self, data: str) -> int:
        self._original.write(data)
        with contextlib.suppress(ValueError):  # sink closed early
            self._sink.write(data)
        return len(data)

    def flush(self) -> None:
        self._original.flush()
        with contextlib.suppress(ValueError):
            self._sink.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class Run:
    """Provenance-tracked run rooted at ``<runs_dir>/<config_hash>/``."""

    def __init__(
        self,
        cfg: Config,
        *,
        runs_dir: str | os.PathLike[str] | None = None,
        on_exist: OnExist = "error",
        require_git: bool = True,
        require_clean: bool = False,
        capture_streams: bool = True,
        fsync_metrics: bool = False,
        seed: int | None = None,
        inputs: Sequence[str | os.PathLike[str]] = (),
        log_level: int = logging.INFO,
    ) -> None:
        """Resolve and hash ``cfg`` and compute the target directory. Performs
        no filesystem writes.

        Raises:
            ConfigTypeError: ``cfg`` is not a mapping, or cannot be hashed as
                a config.
            InputError: ``inputs`` is a single string/bytes path rather than a
                sequence of paths (a plain string is iterable character by
                character, which is almost never what was intended).
        """
        if isinstance(inputs, (str, bytes)):
            raise InputError(
                f"inputs must be a list of paths, not a single string/bytes "
                f"({inputs!r}); did you mean inputs=[{inputs!r}]?"
            )
        self._resolved: Config = resolve_config(cfg)
        self._hash: str = config_hash(self._resolved)
        self._runs_dir_arg = runs_dir
        self._on_exist: OnExist = on_exist
        self._require_git = require_git
        self._require_clean = require_clean
        self._capture_streams = capture_streams
        self._fsync_metrics = fsync_metrics
        self._explicit_seed = seed
        self._inputs = list(inputs)
        self._log_level = log_level

        self._state: _State = "created"
        self._dir: Path | None = None
        self._manifest: Manifest | None = None
        self._metrics: MetricsWriter | None = None
        self._file_handler: logging.Handler | None = None
        self._saved_stdout: TextIO | None = None
        self._saved_stderr: TextIO | None = None
        self._stream_files: list[TextIO] = []

    # -- lifecycle ------------------------------------------------------------
    def __enter__(self) -> Run:
        if self._state != "created":
            raise RunStateError(f"Run has already been {self._state}")

        run_dir = self._prepare_dir(self._compute_dir())
        self._dir = run_dir

        manifest = new_manifest(self._hash)
        self._enforce_git(manifest)
        self._manifest = manifest

        (run_dir / RESOLVED_CONFIG_FILENAME).write_text(
            yaml.safe_dump(
                self._resolved,
                sort_keys=True,
                default_flow_style=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        metrics_path = run_dir / METRICS_FILENAME
        metrics_path.touch(exist_ok=True)
        self._metrics = MetricsWriter(metrics_path, fsync=self._fsync_metrics)
        write_manifest(run_dir, manifest)

        self._install_logging(run_dir)
        if self._capture_streams:
            self._install_stream_capture(run_dir)

        self._state = "active"
        try:
            for item in self._inputs:
                self.declare_input(item)
            resolved_seed = self._resolve_seed()
            if resolved_seed is not None:
                self.seed(resolved_seed)
        except BaseException as exc:
            self._finalize(type(exc), exc, exc.__traceback__)
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        self._finalize(exc_type, exc, tb)
        return False

    # -- properties ---------------------------------------------------------
    @property
    def dir(self) -> Path:
        """The run directory. Raises RunStateError if not entered."""
        if self._dir is None:
            raise RunStateError("run has not been entered")
        return self._dir

    @property
    def config_hash(self) -> str:
        """The full 64-char config hash."""
        return self._hash

    @property
    def config(self) -> Config:
        """A deep copy of the resolved config."""
        return copy.deepcopy(self._resolved)

    # -- during the run ---------------------------------------------------
    def artifact_path(self, name: str) -> Path:
        """Return ``<run>/artifacts/<name>`` (parent dirs created).

        Raises:
            RunStateError: run not active.
            ValueError: ``name`` escapes the artifacts directory.
        """
        self._require_active()
        base = (self.dir / "artifacts").resolve()
        target = (base / name).resolve()
        if target != base and base not in target.parents:
            raise ValueError(f"artifact name {name!r} escapes the artifacts directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def declare_input(self, path: str | os.PathLike[str], *, name: str | None = None) -> str:
        """Checksum a file or directory now and record it in the manifest.

        Returns:
            The checksum (rolled-up hash for a directory).

        Raises:
            InputError: the path is missing or unreadable.
            RunStateError: run not active.
        """
        self._require_active()
        assert self._manifest is not None
        source = Path(path)
        if source.is_dir():
            checksum, files = tree_checksum(source)
            size = sum(child.stat().st_size for child in source.rglob("*") if child.is_file())
            record: InputRecord = {
                "name": name or source.name,
                "path": str(source),
                "kind": "dir",
                "checksum": checksum,
                "size": size,
                "mtime": source.stat().st_mtime,
                "files": files,
            }
        elif source.is_file():
            checksum = file_checksum(source)
            record = {
                "name": name or source.name,
                "path": str(source),
                "kind": "file",
                "checksum": checksum,
                "size": source.stat().st_size,
                "mtime": source.stat().st_mtime,
                "files": None,
            }
        else:
            raise InputError(f"declared input not found: {source}")
        self._manifest["inputs"].append(record)
        write_manifest(self.dir, self._manifest)
        return checksum

    def seed(self, seed: int, *, deterministic_torch: bool = False) -> SeedReport:
        """Seed all RNGs and append the report to the manifest.

        Raises:
            RunStateError: run not active.
            ValueError: ``seed`` outside ``[0, 2**32)``.
        """
        self._require_active()
        assert self._manifest is not None
        report = seed_everything(seed, deterministic_torch=deterministic_torch)
        self._manifest["seeds"].append(report.as_dict())
        write_manifest(self.dir, self._manifest)
        return report

    def log(
        self,
        step: int | None = None,
        *,
        allow_nonscalar: bool = False,
        **values: object,
    ) -> None:
        """Append one metrics row (see
        :meth:`research_kit.labkit.metrics.MetricsWriter.log`).

        Raises:
            MetricsError: bad value or reserved key.
            RunStateError: run not active.
        """
        self._require_active()
        assert self._metrics is not None
        self._metrics.log(step, allow_nonscalar=allow_nonscalar, **values)

    def get_logger(self, name: str | None = None) -> logging.Logger:
        """A logger whose records reach ``<run>/logs/run.log``."""
        return logging.getLogger(name if name is not None else "research_kit.run")

    # -- internals ------------------------------------------------------
    def _require_active(self) -> None:
        if self._state != "active":
            raise RunStateError(f"run is {self._state!r}; this operation requires an active run")

    def _compute_dir(self) -> Path:
        forced = os.environ.get(FORCE_RUN_DIR_ENV)
        if forced:
            return Path(forced)
        if self._runs_dir_arg is not None:
            root = Path(self._runs_dir_arg)
        else:
            root = Path(os.environ.get(RUNS_DIR_ENV, DEFAULT_RUNS_DIR))
        return root / self._hash

    def _prepare_dir(self, run_dir: Path) -> Path:
        # Two processes launched with the identical config (e.g. concurrent
        # SLURM array-job workers) can race between this existence check and
        # the mkdir below; the mkdir is re-checked so the race still ends in a
        # typed RunExistsError instead of a raw FileExistsError.
        while True:
            if run_dir.exists():
                if self._on_exist == "error":
                    raise RunExistsError(f"run directory already exists: {run_dir}")
                if self._on_exist == "new_attempt":
                    run_dir = _next_attempt(run_dir)
                    continue
            try:
                run_dir.mkdir(parents=True, exist_ok=self._on_exist == "resume")
            except FileExistsError as exc:
                if self._on_exist != "new_attempt":
                    raise RunExistsError(f"run directory already exists: {run_dir}") from exc
                run_dir = _next_attempt(run_dir)
                continue
            break
        (run_dir / "logs").mkdir(exist_ok=True)
        (run_dir / "artifacts").mkdir(exist_ok=True)
        return run_dir

    def _enforce_git(self, manifest: Manifest) -> None:
        git = manifest["git"]
        if self._require_git and not git["available"]:
            raise GitError(f"require_git=True but git is unavailable: {git['reason']}")
        if self._require_clean and git["dirty"]:
            raise GitError("require_clean=True but the git working tree is dirty")

    def _resolve_seed(self) -> int | None:
        if self._explicit_seed is not None:
            return self._explicit_seed
        candidate = self._resolved.get("seed")
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return candidate
        return None

    def _install_logging(self, run_dir: Path) -> None:
        handler = logging.FileHandler(run_dir / "logs" / "run.log", encoding="utf-8")
        handler.setLevel(self._log_level)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger().addHandler(handler)
        self._file_handler = handler

    def _install_stream_capture(self, run_dir: Path) -> None:
        self._saved_stdout = sys.stdout
        self._saved_stderr = sys.stderr
        out_file = (run_dir / "logs" / "stdout.log").open("w", encoding="utf-8")
        err_file = (run_dir / "logs" / "stderr.log").open("w", encoding="utf-8")
        self._stream_files = [out_file, err_file]
        sys.stdout = _Tee(self._saved_stdout, out_file)
        sys.stderr = _Tee(self._saved_stderr, err_file)

    def _remove_stream_capture(self) -> None:
        if self._saved_stdout is not None:
            sys.stdout = self._saved_stdout
            self._saved_stdout = None
        if self._saved_stderr is not None:
            sys.stderr = self._saved_stderr
            self._saved_stderr = None
        for handle in self._stream_files:
            with contextlib.suppress(ValueError):
                handle.flush()
                handle.close()
        self._stream_files = []

    def _remove_logging(self) -> None:
        if self._file_handler is not None:
            logging.getLogger().removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None

    def _finalize(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._state == "closed":
            return
        assert self._manifest is not None and self._dir is not None

        if exc is not None:
            self._manifest["status"] = "failed"
            self._manifest["error"] = {
                "type": _exc_name(exc_type, exc),
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(exc_type, exc, tb)),
            }
        else:
            self._manifest["status"] = "completed"
        self._manifest["end_time"] = utc_now()

        if self._metrics is not None:
            self._metrics.close()
        self._remove_stream_capture()
        self._remove_logging()
        write_manifest(self._dir, self._manifest)
        self._state = "closed"


def _exc_name(exc_type: type[BaseException] | None, exc: BaseException) -> str:
    cls = exc_type if exc_type is not None else type(exc)
    module = getattr(cls, "__module__", "builtins")
    return f"{module}.{cls.__qualname__}" if module != "builtins" else cls.__qualname__


def _next_attempt(run_dir: Path) -> Path:
    number = 2
    while True:
        candidate = run_dir / f"attempt-{number:03d}"
        if not candidate.exists():
            return candidate
        number += 1
