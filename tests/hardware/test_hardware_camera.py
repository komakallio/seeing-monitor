"""Checks against a real ASI camera. They skip unless you opt in with `--hardware`.

Each test calls a function of `camera_checks` and prints its report, so `pytest -s` shows what the
camera did. A check needs the vendor library, and it skips when no camera is connected. The USB
reset check also needs `SEEINGMON_HARDWARE_USB_RESET=1`, because the reset interrupts the camera.
"""

from __future__ import annotations

import os
import sys

import pytest

from seeingmon.config import Config
from seeingmon.drivers.asi import AsiDriver
from seeingmon.hardware.asi.api import AsiApi
from tests.hardware import camera_checks as checks

pytestmark = pytest.mark.hardware


def test_enumerate_and_open_the_camera(asi_api: AsiApi, asi_driver: AsiDriver) -> None:
    print(checks.check_enumerate_and_open(asi_api, asi_driver))


def test_the_reported_capabilities_match_the_profile(
    asi_driver: AsiDriver, local_config: Config
) -> None:
    asi_driver.open()
    print(checks.check_capabilities_match_the_profile(asi_driver, local_config.profile))


def test_a_roi_round_trips(asi_driver: AsiDriver, local_config: Config) -> None:
    asi_driver.open()
    print(checks.check_roi_round_trip(asi_driver, local_config.profile))


def test_a_stream_of_100_frames_has_steady_timing_and_counted_drops(
    asi_driver: AsiDriver, local_config: Config
) -> None:
    asi_driver.open()
    print(checks.check_stream(asi_driver, local_config.profile, frames=100))


def test_the_temperature_reads(asi_driver: AsiDriver, local_config: Config) -> None:
    asi_driver.open()
    print(checks.check_temperature(asi_driver, local_config.profile))


def test_recovery_step_one_restarts_capture(asi_driver: AsiDriver, local_config: Config) -> None:
    asi_driver.open()
    print(checks.check_recovery_restart(asi_driver, local_config.profile))


def not_linux() -> bool:
    """Whether this is not Linux. A function, so mypy does not fold the platform test."""
    return not sys.platform.startswith("linux")


@pytest.mark.skipif(not_linux(), reason="the USB reset step is Linux-only")
@pytest.mark.skipif(
    os.environ.get("SEEINGMON_HARDWARE_USB_RESET") != "1",
    reason="the USB reset interrupts the camera: set SEEINGMON_HARDWARE_USB_RESET=1",
)
def test_recovery_step_three_resets_the_usb_device(
    asi_driver: AsiDriver, local_config: Config
) -> None:
    asi_driver.open()
    print(checks.check_usb_reset(asi_driver, local_config.profile))
