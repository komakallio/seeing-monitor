"""The `asi` driver on the `ctypes` binding, with a Python stand-in for the vendor library.

`FakeCLibrary` answers the C calls from a `FakeAsiSdk`, so the driver runs through the same
argument handling, structure fields, and status-code mapping that it uses on a real library. These
tests repeat the main flows of the conformance tests on that path.
"""

from __future__ import annotations

import numpy as np
import pytest

from seeingmon.drivers import CameraConfigError, CameraTimeoutError, RecoveryLevel
from seeingmon.frames import FrameFlag, PixelFormat, Roi, StreamConfig, StreamKind
from seeingmon.hardware.asi.api import AsiControl, AsiImageType
from seeingmon.hardware.asi.ctypes_api import CtypesAsiApi
from seeingmon.hardware.asi.fake import FakeAsiSdk, FakeFrameInfo, default_pixels
from tests.hardware.asi_clib import FakeCLibrary
from tests.hardware.asi_support import FAST, TINY, Rig, make_rig


def through_ctypes(sdk: FakeAsiSdk) -> CtypesAsiApi:
    return CtypesAsiApi(FakeCLibrary(sdk))


@pytest.fixture
def rig() -> Rig:
    return make_rig(api=through_ctypes).opened()


def test_the_driver_reports_the_camera_through_the_binding(rig: Rig) -> None:
    info = rig.driver.open()
    assert (info.model, info.sdk_version, info.has_temperature) == (
        "ZWO ASI294MM (fake)",
        "1, 41, 0, 0",
        True,
    )
    caps = rig.driver.capabilities()
    assert (caps.gain_range, caps.bins) == ((0, 570), (1, 2))
    assert PixelFormat.RAW16 in caps.pixel_formats


def test_frames_carry_the_pixels_of_the_fake_camera(rig: Rig) -> None:
    rig.driver.configure(TINY)
    rig.driver.start()
    frame = rig.driver.read_frame(1.0)
    counts = default_pixels(FakeFrameInfo(1, 16, 8, 1, AsiImageType.RAW16, 8, 4, 2000, 120, 12))
    np.testing.assert_array_equal(frame.data, counts << 4)  # frame 0 went at the start
    assert frame.roi == Roi(8, 4, 16, 8)
    assert rig.driver.read_frame(1.0).seq == 1


def test_a_stream_moves_its_roi_and_counts_drops(rig: Rig) -> None:
    rig.driver.configure(FAST)
    rig.driver.start()
    assert rig.driver.move_roi(-5, 10_000) == Roi(0, 5644 - 128, 128, 128)
    assert rig.driver.read_frame(1.0).roi == Roi(0, 5644 - 128, 128, 128)
    rig.sdk.lose_frames(4)
    assert rig.driver.read_frame(1.0).dropped_before == 4


def test_the_controls_round_trip(rig: Rig) -> None:
    active = rig.driver.configure(
        StreamConfig(
            mode="bin1",
            exposure_us=2000,
            gain=150,
            offset=30,
            roi=Roi(0, 0, 16, 8),
            high_speed=True,
        )
    )
    assert (active.config.gain, active.config.offset, active.adc_bits) == (150, 30, 10)
    assert rig.sdk.control(AsiControl.GAIN) == 150
    assert rig.sdk.control(AsiControl.HIGH_SPEED_MODE) == 1
    with pytest.raises(CameraConfigError, match="gain"):
        rig.driver.configure(StreamConfig(mode="bin1", exposure_us=2000, gain=9999))


def test_a_stalled_read_times_out_and_the_ladder_recovers(rig: Rig) -> None:
    rig.driver.configure(TINY)
    rig.driver.start()
    rig.sdk.stall_reads(1)
    with pytest.raises(CameraTimeoutError):
        rig.driver.read_frame(1.0)
    rig.driver.recover(RecoveryLevel.RESTART_CAPTURE)
    assert rig.driver.read_frame(1.0).flags & FrameFlag.RECOVERED
    rig.driver.recover(RecoveryLevel.REOPEN)
    assert rig.driver.read_frame(1.0).flags & FrameFlag.RECOVERED
    rig.driver.recover(RecoveryLevel.USB_RESET)
    assert rig.driver.read_frame(1.0).flags & FrameFlag.RECOVERED


def test_a_silent_geometry_change_is_corrected_through_the_binding(rig: Rig) -> None:
    rig.driver.configure(TINY)  # the first configure also sets the other format (the latch)
    rig.sdk.corrupt_next_roi(24, 8)
    active = rig.driver.configure(TINY)
    assert active.config.roi == Roi(8, 4, 16, 8)
    assert rig.event_kinds() == ["camera.geometry_corrected"]


def test_a_snapshot_takes_one_exposure(rig: Rig) -> None:
    rig.driver.configure(
        StreamConfig(
            mode="bin2",
            exposure_us=2_000_000,
            gain=120,
            kind=StreamKind.SNAPSHOT,
            roi=Roi(0, 0, 64, 64),
        )
    )
    rig.driver.start()
    frame = rig.driver.read_frame(10.0)
    assert (frame.exposure_us, frame.adc_bits, frame.data.shape) == (2_000_000, 14, (64, 64))
    assert frame.temperature_c == pytest.approx(18.3)
