"""collect_git_info: never raises, reports what it learned."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_kit.labkit.gitinfo import collect_git_info


def test_clean_repo_reports_sha(tmp_git_repo: Path) -> None:
    info = collect_git_info(tmp_git_repo)
    assert info["available"] is True
    assert info["dirty"] is False
    assert len(info["sha"] or "") == 40


def test_dirty_repo_reports_diff(tmp_git_repo: Path) -> None:
    (tmp_git_repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    info = collect_git_info(tmp_git_repo)
    assert info["dirty"] is True
    assert "changed" in (info["diff"] or "")


def test_untracked_listed_not_dumped(tmp_git_repo: Path) -> None:
    (tmp_git_repo / "secret.txt").write_text("PASSWORD\n", encoding="utf-8")
    info = collect_git_info(tmp_git_repo)
    assert "secret.txt" in (info["untracked"] or [])
    assert "PASSWORD" not in (info["diff"] or "")


def test_non_repo_returns_unavailable(tmp_path: Path) -> None:
    info = collect_git_info(tmp_path)
    assert info["available"] is False
    assert info["reason"]


def test_missing_git_binary_handled(tmp_git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")
    info = collect_git_info(tmp_git_repo)
    assert info["available"] is False
    assert info["reason"]
