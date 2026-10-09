"""Content checksums for declared input files and directories."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .errors import InputError

__all__ = ["ALGO", "file_checksum", "tree_checksum"]

ALGO = "blake2b"
"""Hash algorithm; digests are 64 hex chars (blake2b, digest_size=32)."""

_DIGEST_SIZE = 32
_CHUNK = 1 << 20


def file_checksum(path: str | os.PathLike[str]) -> str:
    """Return the blake2b-256 hex digest of a file's bytes (streamed).

    Raises:
        InputError: the path is missing, is not a regular file, or is
            unreadable.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise InputError(f"not a file: {file_path}")
    digest = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    try:
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK), b""):
                digest.update(chunk)
    except OSError as exc:
        raise InputError(f"cannot read {file_path}: {exc}") from exc
    return digest.hexdigest()


def tree_checksum(path: str | os.PathLike[str]) -> tuple[str, list[tuple[str, str]]]:
    """Checksum every regular file under a directory.

    Returns:
        ``(rolled_up_hash, files)`` where ``files`` is a list of
        ``(posix_relpath, file_hash)`` sorted by relpath, and
        ``rolled_up_hash`` is a blake2b digest over that sorted listing
        (independent of filesystem walk order).

    Raises:
        InputError: the path is missing or is not a directory.
    """
    root = Path(path)
    if not root.is_dir():
        raise InputError(f"not a directory: {root}")
    files: list[tuple[str, str]] = []
    for child in root.rglob("*"):
        if child.is_file():
            files.append((child.relative_to(root).as_posix(), file_checksum(child)))
    files.sort()
    rolled = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    for relpath, file_hash in files:
        rolled.update(relpath.encode("utf-8"))
        rolled.update(b"\0")
        rolled.update(file_hash.encode("ascii"))
        rolled.update(b"\n")
    return rolled.hexdigest(), files
