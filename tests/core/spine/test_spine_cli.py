"""research-spine resolve (invariant 2: CLI is a thin wrapper)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_kit.core.spine import cli as spine_cli
from research_kit.core.spine import config_hash


@pytest.fixture
def cfg_file(tmp_path: Path) -> Path:
    p = tmp_path / "c.yaml"
    p.write_text("a: 1\nb: ${a}\n", encoding="utf-8")
    return p


def test_resolve_prints_resolved_and_hash(
    cfg_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = spine_cli.main(["resolve", str(cfg_file), "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out.split("hash:")[0])
    assert payload == {"a": 1, "b": 1}
    assert config_hash({"a": 1, "b": 1})[:16] in out


def test_resolve_applies_overrides(cfg_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    spine_cli.main(["resolve", str(cfg_file), "a=5", "--json"])
    payload = json.loads(capsys.readouterr().out.split("hash:")[0])
    assert payload == {"a": 5, "b": 5}


def test_resolve_hash_only(cfg_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    spine_cli.main(["resolve", str(cfg_file), "--hash-only"])
    assert capsys.readouterr().out.strip() == config_hash({"a": 1, "b": 1})


def test_main_returns_int_on_bad_config(tmp_path: Path) -> None:
    rc = spine_cli.main(["resolve", str(tmp_path / "missing.yaml")])
    assert rc == 1
