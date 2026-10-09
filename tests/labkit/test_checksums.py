"""file_checksum / tree_checksum."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from research_kit.labkit import file_checksum, tree_checksum
from research_kit.labkit.errors import InputError


def test_file_checksum_matches_blake2b(tmp_path: Path) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello world")
    assert file_checksum(f) == hashlib.blake2b(b"hello world", digest_size=32).hexdigest()


def test_tree_checksum_walk_order_independent(tmp_path: Path) -> None:
    for name in ["c.txt", "a.txt", "b/d.txt"]:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(name, encoding="utf-8")
    rolled, files = tree_checksum(tmp_path)
    assert [rel for rel, _ in files] == sorted(rel for rel, _ in files)
    assert "b/d.txt" in {rel for rel, _ in files}
    rolled2, _ = tree_checksum(tmp_path)
    assert rolled == rolled2


def test_tree_checksum_changes_on_edit(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("one", encoding="utf-8")
    before, _ = tree_checksum(tmp_path)
    (tmp_path / "x.txt").write_text("two", encoding="utf-8")
    after, _ = tree_checksum(tmp_path)
    assert before != after


def test_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(InputError):
        file_checksum(tmp_path / "nope")
    with pytest.raises(InputError):
        tree_checksum(tmp_path / "nope")
