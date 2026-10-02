"""Shared pytest configuration: options, skip rules, and fixtures.

Markers (declared in `pyproject.toml`):

- `hardware`: needs a real camera, Raspberry Pi, or other hardware. Skipped unless you pass
  `--hardware` or set `SEEINGMON_HARDWARE=1`.
- `recordings`: needs the owner's recordings. The `recordings_dir` fixture skips the test
  when the location is not configured.
- `slow`: runs for a long time. Skipped unless you pass `--slow` or set `SEEINGMON_SLOW=1`.
"""

from __future__ import annotations

import os
import tempfile
import tomllib
from pathlib import Path

import pytest
from hypothesis import HealthCheck, settings

from seeingmon import paths

REPO_ROOT = Path(__file__).resolve().parents[1]

settings.register_profile("default", deadline=None, suppress_health_check=[HealthCheck.too_slow])
settings.register_profile("thorough", max_examples=1000, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("seeingmon")
    group.addoption("--hardware", action="store_true", help="run tests that need hardware")
    group.addoption("--slow", action="store_true", help="run slow tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_hardware = (
        bool(config.getoption("--hardware")) or os.environ.get("SEEINGMON_HARDWARE") == "1"
    )
    run_slow = bool(config.getoption("--slow")) or os.environ.get("SEEINGMON_SLOW") == "1"
    for item in items:
        if item.get_closest_marker("hardware") is not None and not run_hardware:
            reason = "needs hardware: pass --hardware or set SEEINGMON_HARDWARE=1"
            item.add_marker(pytest.mark.skip(reason=reason))
        if item.get_closest_marker("slow") is not None and not run_slow:
            reason = "slow: pass --slow or set SEEINGMON_SLOW=1"
            item.add_marker(pytest.mark.skip(reason=reason))


# A location where no local configuration file exists.
NO_LOCAL_CONFIG = Path(tempfile.gettempdir()) / "seeingmon-tests-no-local-config" / "config.toml"


@pytest.fixture(autouse=True)
def _keep_the_local_config_out_of_the_tests(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hide the `local/config.toml` of the machine from every test.

    A test that loads the configuration would otherwise see the settings of the person who runs
    it, and a command that writes where the configuration says would write into their data. A
    test of `dark` once added simulated dark sets to the owner's real dark library that way. The
    hardware tests use the real file on purpose, and the tests of `seeingmon.paths` call the real
    function.
    """
    if request.node.get_closest_marker("hardware") is not None:
        return
    if request.module.__name__.rsplit(".", 1)[-1] == "test_paths":
        return
    monkeypatch.setattr(paths, "local_config_file", lambda **_: NO_LOCAL_CONFIG)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The root of the source tree. Tests locate repository files through this, never the cwd."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def recordings_dir() -> Path:
    """The folder with the owner's recordings. Skips the test when it is not configured.

    The location comes from `SEEINGMON_REPLAY__RECORDINGS_DIR` (the configuration scheme for
    `recordings_dir` in the `[replay]` table), from the older `SEEINGMON_RECORDINGS_DIR`, or
    from `recordings_dir` in the `[replay]` table of `local/config.toml`. None of them belongs in
    the repository.
    """
    configured = os.environ.get("SEEINGMON_REPLAY__RECORDINGS_DIR") or os.environ.get(
        "SEEINGMON_RECORDINGS_DIR", ""
    )
    config_file = REPO_ROOT / "local" / "config.toml"
    if not configured and config_file.is_file():
        with config_file.open("rb") as handle:
            configured = str(tomllib.load(handle).get("replay", {}).get("recordings_dir", ""))
    if not configured:
        pytest.skip("recordings are not configured (see docs/development.md)")
    path = Path(configured)
    if not path.is_dir():
        pytest.skip("the configured recordings folder does not exist")
    return path
