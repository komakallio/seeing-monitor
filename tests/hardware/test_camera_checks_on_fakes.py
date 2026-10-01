"""The camera checks of `camera_checks`, run against the fake SDK so that their code is tested."""

from __future__ import annotations

import pytest

from seeingmon.drivers.base import CameraDriver
from tests.hardware import camera_checks as checks
from tests.hardware.asi_support import Rig, make_rig, reference_profile


@pytest.fixture
def rig() -> Rig:
    return make_rig()


def test_enumerate_and_open(rig: Rig) -> None:
    report = checks.check_enumerate_and_open(rig.sdk, rig.driver)
    assert "ZWO ASI294MM" in report
    assert "serial" not in report.lower()


def test_the_fake_camera_matches_the_reference_profile(rig: Rig) -> None:
    checks.check_capabilities_match_the_profile(rig.driver, reference_profile())


def test_a_camera_that_differs_from_the_profile_fails_the_check() -> None:
    rig = make_rig(sdk={"max_width": 4144, "max_height": 2822})
    with pytest.raises(AssertionError, match="sensor 4144 x 2822"):
        checks.check_capabilities_match_the_profile(rig.driver, reference_profile())


def test_roi_round_trip(rig: Rig) -> None:
    rig.driver.open()
    report = checks.check_roi_round_trip(rig.driver, reference_profile())
    assert "asked (1001, 801)" in report


def test_the_roi_check_notices_a_camera_that_aligns_the_start_position(rig: Rig) -> None:
    aligned = make_rig(sdk={"start_alignment": 4})
    aligned.driver.open()
    report = checks.check_roi_round_trip(aligned.driver, reference_profile())
    assert "asked (1001, 801), the camera applied (1000, 800)" in report


def test_stream(rig: Rig) -> None:
    rig.driver.open()
    report = checks.check_stream(rig.driver, reference_profile())
    assert "100 frames" in report
    assert "0 dropped" in report


def test_the_stream_check_fails_when_the_camera_drops_frames(rig: Rig) -> None:
    rig.driver.open()
    original = rig.sdk.get_video_data
    state = {"reads": 0}

    def lossy(camera_id: int, buffer: bytearray, wait_ms: int) -> None:
        state["reads"] += 1
        if state["reads"] % 3 == 0:
            rig.sdk.lose_frames(1)
        original(camera_id, buffer, wait_ms)

    rig.sdk.get_video_data = lossy  # type: ignore[method-assign]
    with pytest.raises(AssertionError, match="dropped"):
        checks.check_stream(rig.driver, reference_profile())


def test_temperature(rig: Rig) -> None:
    rig.driver.open()
    report = checks.check_temperature(rig.driver, reference_profile())
    assert "18.3" in report


def test_recovery_step_one(rig: Rig) -> None:
    rig.driver.open()
    checks.check_recovery_restart(rig.driver, reference_profile())


def test_recovery_step_three(rig: Rig) -> None:
    rig.driver.open()
    checks.check_usb_reset(rig.driver, reference_profile())
    assert rig.resetter.count == 1


def test_every_check_works_through_the_driver_protocol(rig: Rig) -> None:
    driver: CameraDriver = rig.driver
    rig.driver.open()
    checks.check_recovery_restart(driver, reference_profile())
