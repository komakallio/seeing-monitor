"""Tests for `tools/scan_secrets.py`.

The trigger string comes from two pieces, so this file never holds a literal that the scan
itself would report.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tools.scan_secrets import BASELINE, batches, list_files, main

REPO_BASELINE = Path(__file__).resolve().parents[1] / BASELINE
SECRET_LINE = "key = '" + "AKIA" + "IOSFODNN7EXAMPLE" + "'"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    shutil.copy(REPO_BASELINE, tmp_path / BASELINE)
    return tmp_path


def test_batches_split_a_list_without_losing_names() -> None:
    names = [f"f{number}" for number in range(7)]
    parts = list(batches(names, 3))
    assert [len(part) for part in parts] == [3, 3, 1]
    assert [name for part in parts for name in part] == names


def test_list_files_reads_the_tracked_files_or_only_the_staged_ones(repo: Path) -> None:
    (repo / "a.txt").write_text("one\n", encoding="utf-8")
    (repo / "b.txt").write_text("two\n", encoding="utf-8")
    git(repo, "add", "a.txt")
    assert list_files(repo, staged=True) == ["a.txt"]
    assert list_files(repo, staged=False) == ["a.txt"]
    git(repo, "add", "b.txt")
    assert sorted(list_files(repo, staged=False)) == ["a.txt", "b.txt"]


def test_a_clean_repository_passes(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (repo / "a.txt").write_text("nothing to see\n", encoding="utf-8")
    git(repo, "add", "a.txt", BASELINE)
    assert main(["--repo", str(repo)]) == 0
    assert "clean" in capsys.readouterr().out


def test_a_secret_fails_and_the_pragma_allows_it(repo: Path) -> None:
    target = repo / "settings.py"
    target.write_text(SECRET_LINE + "\n", encoding="utf-8")
    git(repo, "add", "settings.py", BASELINE)
    assert main(["--repo", str(repo)]) != 0
    assert main(["--repo", str(repo), "--staged"]) != 0
    target.write_text(SECRET_LINE + "  # pragma: allowlist secret\n", encoding="utf-8")
    git(repo, "add", "settings.py")
    assert main(["--repo", str(repo)]) == 0
