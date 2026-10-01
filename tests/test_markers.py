"""The skip rules in `conftest.py`. Opt-in tests assert their own opt-in, so they pass or skip."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.mark.hardware
def test_hardware_tests_need_an_opt_in(request: pytest.FixtureRequest) -> None:
    """Real-hardware checks look like this. The body runs only after an explicit opt-in."""
    opted_in = request.config.getoption("--hardware") or os.environ.get("SEEINGMON_HARDWARE") == "1"
    assert opted_in


@pytest.mark.slow
def test_slow_tests_need_an_opt_in(request: pytest.FixtureRequest) -> None:
    opted_in = request.config.getoption("--slow") or os.environ.get("SEEINGMON_SLOW") == "1"
    assert opted_in


@pytest.mark.recordings
def test_recordings_fixture_skips_or_points_at_a_folder(recordings_dir: Path) -> None:
    """With no recordings configured, the fixture skips this test."""
    assert recordings_dir.is_dir()


def test_repo_root_fixture_finds_the_project(repo_root: Path) -> None:
    assert (repo_root / "pyproject.toml").is_file()
