"""Helpers for the deploy tests."""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path

from tools.lint_deploy import Finding


def rules(findings: Iterable[Finding]) -> set[str]:
    """The rule names of some findings."""
    return {finding.rule for finding in findings}


def mutate(text: str, old: str, new: str) -> str:
    """Replace `old` with `new` once. Fails when `old` is missing, so a stale test is loud."""
    assert old in text, f"the text no longer contains {old!r}"
    return text.replace(old, new, 1)


def read_deploy(repo_root: Path, relative: str) -> str:
    """The text of a file under `deploy/`."""
    return (repo_root / "deploy" / relative).read_text(encoding="utf-8")


def copy_deploy(repo_root: Path, destination: Path) -> Path:
    """Copy the `deploy/` tree into a fake repository, and return the root of that repository."""
    shutil.copytree(repo_root / "deploy", destination / "deploy")
    return destination
