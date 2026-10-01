"""Fixtures for the real-hardware checks.

The checks carry the `hardware` marker, so they skip unless you pass `--hardware` or set
`SEEINGMON_HARDWARE=1`. With the opt-in, a check that needs a device that is not there skips with
the reason, and a check that finds a device that misbehaves fails. The fixtures read the same local
configuration as the system (`local/config.toml` and the `SEEINGMON_*` variables). See
`docs/hardware-checks.md`.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from seeingmon.clock import SystemClock
from seeingmon.config import Config, load_config
from seeingmon.drivers.asi import AsiDriver
from seeingmon.hardware.asi.api import AsiApi, AsiLibraryError


@pytest.fixture(scope="session")
def local_config() -> Config:
    """The configuration of this machine: the defaults, `local/config.toml`, and the environment."""
    return load_config()


def driver_options(config: Config) -> dict[str, Any]:
    """The options of the camera driver, from `[services.acquire.driver_options]`."""
    values = config.effective(redact=False)
    options = values.get("services", {}).get("acquire", {}).get("driver_options", {})
    return dict(options) if isinstance(options, dict) else {}


@pytest.fixture
def asi_api(local_config: Config) -> AsiApi:
    """The vendor library through `ctypes`. Skips when the library is not installed."""
    from seeingmon.hardware.asi.ctypes_api import load_asi_api

    try:
        api = load_asi_api(driver_options(local_config).get("library_path"))
    except AsiLibraryError as error:
        pytest.skip(f"the ASI library is not available: {error}")
    if api.get_connected_camera_count() == 0:
        pytest.skip("no ASI camera is connected")
    return api


@pytest.fixture
def asi_driver(local_config: Config, asi_api: AsiApi) -> Iterator[AsiDriver]:
    """The production driver, built by `create` with its watchdog, on the system clock."""
    from seeingmon.drivers import asi

    driver = asi.create(
        profile=local_config.profile, clock=SystemClock(), options=driver_options(local_config)
    )
    try:
        yield driver
    finally:
        driver.close()
