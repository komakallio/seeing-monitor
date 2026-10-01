"""The lifecycle conformance tests of `FakeCameraDriver`, run against the `asi` driver.

`tests/test_fakes.py` checks the lifecycle that every `CameraDriver` follows: the stream
lifecycle, the ROI rules, a new stream per `configure`, `move_roi`, snapshots, bounded waits,
failures, drops, and the pixel rules. Each test here repeats one of those checks on `AsiDriver`
running on `FakeAsiSdk` and a `VirtualClock`. Where the real driver differs from the fake driver
on purpose, a comment says so (the time quality is `ESTIMATED`, and a read fails through an SDK
error code).
"""

from __future__ import annotations

import numpy as np
import pytest

from seeingmon.clock import NS_PER_S, VirtualClock
from seeingmon.drivers import (
    CameraConfigError,
    CameraDisconnectedError,
    CameraDriver,
    CameraError,
    CameraStateError,
    CameraTimeoutError,
    RecoveryLevel,
)
from seeingmon.frames import (
    FrameFlag,
    PixelFormat,
    Roi,
    StreamConfig,
    StreamKind,
    TimeQuality,
    decode_frame,
    encode_frame,
    frames_equal,
)
from seeingmon.hardware.asi.api import AsiErrorCode, AsiImageType
from seeingmon.hardware.asi.fake import FakeFrameInfo, default_pixels
from tests.hardware.asi_support import FAST, Rig, make_rig

RAW16 = PixelFormat.RAW16
ASI_RAW16 = AsiImageType.RAW16


@pytest.fixture
def rig() -> Rig:
    return make_rig().opened()


def test_the_driver_satisfies_the_protocol(rig: Rig) -> None:
    assert isinstance(rig.driver, CameraDriver)
    assert rig.driver.name == "asi"


class TestStreaming:
    def test_streams_frames_and_advances_virtual_time(self, rig: Rig) -> None:
        driver, clock = rig.driver, rig.clock
        active = driver.configure(FAST)
        driver.start()
        started = clock.utc_ns()
        frames = [driver.read_frame(timeout_s=1.0) for _ in range(5)]
        assert [f.seq for f in frames] == [0, 1, 2, 3, 4]
        assert all(f.stream_id == active.stream_id for f in frames)
        assert active.frame_period_s is not None
        period_ns = round(active.frame_period_s * NS_PER_S)
        assert clock.utc_ns() - started == 5 * period_ns
        assert frames[1].t_arrival_ns - frames[0].t_arrival_ns == period_ns
        # The real driver estimates the time, where the fake driver reports it exactly.
        assert all(f.t_quality is TimeQuality.ESTIMATED for f in frames)
        assert not any(f.flags & FrameFlag.SIMULATED for f in frames)
        assert frames[0].data.shape == (128, 128)
        assert frames[0].t_utc_ns < frames[0].t_arrival_ns

    def test_frames_survive_the_wire_format(self, rig: Rig) -> None:
        rig.driver.configure(FAST)
        rig.driver.start()
        frame = rig.driver.read_frame(timeout_s=1.0)
        assert frames_equal(decode_frame(encode_frame(frame)), frame)

    def test_the_roi_follows_the_vendor_rules_and_stays_inside_the_frame(self, rig: Rig) -> None:
        odd = StreamConfig(mode="bin1", exposure_us=2000, gain=1, roi=Roi(8200, 5600, 131, 125))
        roi = rig.driver.configure(odd).config.roi
        assert roi is not None
        assert (roi.width, roi.height) == (128, 124)
        assert roi.x_end <= 8288
        assert roi.y_end <= 5644

    def test_the_full_frame_is_the_default_roi(self, rig: Rig) -> None:
        active = rig.driver.configure(StreamConfig(mode="bin2", exposure_us=1000, gain=1))
        assert active.frame_shape == (2822, 4144)

    def test_each_configure_starts_a_new_stream(self, rig: Rig) -> None:
        driver = rig.driver
        first = driver.configure(FAST)
        driver.start()
        driver.read_frame(1.0)
        second = driver.configure(FAST)
        assert second.stream_id == first.stream_id + 1
        with pytest.raises(CameraStateError):  # configure stops capture
            driver.read_frame(1.0)
        driver.start()
        assert driver.read_frame(1.0).seq == 0

    def test_move_roi_keeps_the_stream_and_clamps(self, rig: Rig) -> None:
        driver = rig.driver
        active = driver.configure(FAST)
        driver.start()
        moved = driver.move_roi(-50, 10_000)
        assert moved == Roi(0, 5644 - 128, 128, 128)
        frame = driver.read_frame(1.0)
        assert frame.roi == moved
        assert frame.stream_id == active.stream_id

    def test_snapshot_returns_one_frame_per_start(self, rig: Rig) -> None:
        driver = rig.driver
        driver.configure(
            StreamConfig(
                mode="bin2",
                exposure_us=30_000_000,
                gain=120,
                kind=StreamKind.SNAPSHOT,
                roi=Roi(0, 0, 64, 64),
            )
        )
        driver.start()
        frame = driver.read_frame(timeout_s=60.0)
        assert frame.exposure_us == 30_000_000
        with pytest.raises(CameraStateError):
            driver.read_frame(1.0)
        driver.start()
        assert driver.read_frame(60.0).seq == 1

    def test_a_short_timeout_expires_after_waiting_the_timeout(self, rig: Rig) -> None:
        rig.driver.configure(
            StreamConfig(mode="bin1", exposure_us=500_000, gain=1, roi=Roi(0, 0, 8, 2))
        )
        rig.driver.start()
        started = rig.clock.monotonic_ns()
        with pytest.raises(CameraTimeoutError):
            rig.driver.read_frame(timeout_s=0.1)
        assert rig.clock.monotonic_ns() - started == round(0.1 * NS_PER_S)

    def test_scripted_read_failures_come_one_per_read(self, rig: Rig) -> None:
        driver = rig.driver
        driver.configure(FAST)
        driver.start()
        rig.sdk.fail_next("get_video_data", AsiErrorCode.TIMEOUT)
        rig.sdk.fail_next("get_video_data", AsiErrorCode.CAMERA_REMOVED)
        with pytest.raises(CameraTimeoutError):
            driver.read_frame(1.0)
        with pytest.raises(CameraDisconnectedError):
            driver.read_frame(1.0)
        assert driver.read_frame(1.0).seq == 0

    def test_dropped_frames_are_reported_once(self, rig: Rig) -> None:
        driver = rig.driver
        driver.configure(FAST)
        driver.start()
        rig.sdk.lose_frames(3)
        assert driver.dropped_frames() == 3
        assert driver.read_frame(1.0).dropped_before == 3
        assert driver.read_frame(1.0).dropped_before == 0
        assert driver.dropped_frames() == 3


class TestLifecycleErrors:
    def test_the_calls_need_the_right_state(self) -> None:
        driver = make_rig().driver
        with pytest.raises(CameraStateError):
            driver.configure(FAST)
        driver.open()
        with pytest.raises(CameraStateError):
            driver.start()
        with pytest.raises(CameraConfigError):
            driver.configure(StreamConfig(mode="bin9", exposure_us=1, gain=0))
        with pytest.raises(CameraConfigError):
            driver.configure(
                StreamConfig(mode="bin1", exposure_us=64, gain=0, roi=Roi(0, 0, 9000, 8))
            )

    def test_open_fails_when_no_camera_answers(self) -> None:
        rig = make_rig()
        rig.sdk.disconnect()
        with pytest.raises(CameraDisconnectedError):
            rig.driver.open()

    def test_open_reports_the_camera_without_a_serial_number(self) -> None:
        info = make_rig().driver.open()
        assert (info.driver, info.has_temperature, info.max_width) == ("asi", True, 8288)
        assert info.sdk_version == "1, 41, 0, 0"
        assert "serial" not in repr(info).lower()

    def test_recover_works_and_a_failing_step_raises_a_camera_error(self) -> None:
        rig = make_rig().streaming()
        rig.driver.recover(RecoveryLevel.RESTART_CAPTURE)
        rig.sdk.fail_next("set_roi_format", AsiErrorCode.GENERAL_ERROR)
        with pytest.raises(CameraError):
            rig.driver.recover(RecoveryLevel.USB_RESET)

    def test_close_is_safe_to_call_twice(self) -> None:
        rig = make_rig().streaming()
        rig.driver.close()
        rig.driver.close()
        assert not rig.sdk.video_active
        with pytest.raises(CameraStateError):
            rig.driver.read_frame(1.0)


class TestPixels:
    def test_a_16_bit_frame_holds_the_adc_value_in_the_high_bits(self, rig: Rig) -> None:
        driver = rig.driver
        driver.configure(StreamConfig(mode="bin1", exposure_us=1000, gain=0, roi=Roi(0, 0, 16, 8)))
        driver.start()
        frame = driver.read_frame(1.0)
        # The first frame of a stream is frame 1, because the driver discards frame 0 on start.
        counts = default_pixels(FakeFrameInfo(1, 16, 8, 1, ASI_RAW16, 0, 0, 1000, 0, 12))
        assert frame.data.dtype == np.uint16
        np.testing.assert_array_equal(frame.data, counts << 4)  # 12 bit: shifted left by 4
        assert frame.adc_bits == 12

    def test_the_shift_follows_the_adc_depth_of_the_mode(self, rig: Rig) -> None:
        driver = rig.driver
        driver.configure(StreamConfig(mode="bin2", exposure_us=1000, gain=0, roi=Roi(0, 0, 16, 8)))
        driver.start()
        frame = driver.read_frame(1.0)
        counts = default_pixels(FakeFrameInfo(1, 16, 8, 2, ASI_RAW16, 0, 0, 1000, 0, 14))
        np.testing.assert_array_equal(frame.data, counts << 2)  # 14 bit: shifted left by 2
        assert frame.adc_bits == 14

    def test_an_8_bit_frame_is_an_8_bit_array(self, rig: Rig) -> None:
        driver = rig.driver
        driver.configure(
            StreamConfig(
                mode="bin1",
                exposure_us=1000,
                gain=0,
                roi=Roi(0, 0, 16, 8),
                pixel_format=PixelFormat.RAW8,
            )
        )
        driver.start()
        frame = driver.read_frame(1.0)
        assert frame.data.dtype == np.uint8
        assert frame.pixel_format is PixelFormat.RAW8


def test_a_simulated_night_of_frames_runs_in_seconds(rig: Rig) -> None:
    clock: VirtualClock = rig.clock
    rig.driver.configure(
        StreamConfig(mode="bin1", exposure_us=1_000_000, gain=1, roi=Roi(0, 0, 8, 2))
    )
    rig.driver.start()
    started = clock.utc_ns()
    count = 0
    while clock.utc_ns() - started < 2 * 3600 * NS_PER_S:
        rig.driver.read_frame(timeout_s=5.0)
        count += 1
    assert 2 * 3600 // 2 < count <= 2 * 3600
