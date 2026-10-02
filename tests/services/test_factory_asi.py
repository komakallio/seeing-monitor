"""The asi driver of `acquire` on the fake SDK: the option for tests of the real-camera setup.

`seeingmon dev --driver asi` runs `acquire` with the `asi` driver, which needs a vendor library and
a camera. The option `fake_sdk` swaps both for `FakeAsiSdk`, so that the tests can run the plan of
that launcher as processes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from seeingmon.clock import VirtualClock
from seeingmon.drivers.asi import AsiDriver
from seeingmon.drivers.base import CameraConfigError
from seeingmon.frames import Roi, StreamConfig, StreamKind
from seeingmon.hardware.asi.api import AsiLibraryError
from seeingmon.profile import load_profile
from seeingmon.services.acquire.factory import create_camera_driver

FAST = StreamConfig("bin1", 2000, 0, roi=Roi(100, 200, 128, 128), kind=StreamKind.VIDEO)


def make(**options: object) -> AsiDriver:
    driver = create_camera_driver(
        "asi",
        profile=load_profile("asi294mm-gs250"),
        clock=VirtualClock(),
        options={"fake_sdk": True, **options},
    )
    assert isinstance(driver, AsiDriver)
    return driver


class TestTheFakeSdk:
    def test_the_option_builds_the_asi_driver_without_a_library(self) -> None:
        driver = make()
        info = driver.open()
        try:
            assert "ASI294MM" in info.model  # the fake SDK names itself as the reference camera
            assert driver.configure(FAST).config.roi == FAST.roi
            driver.start()
            frame = driver.read_frame(timeout_s=5.0)
            driver.stop()
            assert (frame.roi.width, frame.roi.height) == (128, 128)
        finally:
            driver.close()

    def test_the_other_options_are_those_of_the_asi_driver(self) -> None:
        driver = make(bandwidth_pct=50, discard_frames=0)
        driver.open()
        driver.close()

    def test_an_option_that_the_driver_does_not_know_is_a_configuration_error(self) -> None:
        with pytest.raises(CameraConfigError, match="no_such_option"):
            make(no_such_option=1)

    def test_a_real_run_without_the_option_asks_for_the_library(self, tmp_path: Path) -> None:
        with pytest.raises(AsiLibraryError, match="does not exist"):
            create_camera_driver(
                "asi",
                profile=load_profile("asi294mm-gs250"),
                clock=VirtualClock(),
                options={"library_path": str(tmp_path / "absent-library")},
            )
