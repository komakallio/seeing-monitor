"""The `asi` driver on a fake SDK: time stamps, bounded waits, controls, geometry, and options."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from seeingmon.clock import NS_PER_S, ClockStatus, VirtualClock
from seeingmon.drivers import (
    CameraConfigError,
    CameraDisconnectedError,
    CameraStateError,
    CameraTimeoutError,
)
from seeingmon.drivers.asi import AsiDriver, AsiOptions, create
from seeingmon.frames import (
    FrameFlag,
    PixelFormat,
    Roi,
    StreamConfig,
    StreamKind,
    TimeQuality,
)
from seeingmon.hardware.asi.api import AsiControl, AsiImageType, AsiLibraryError
from seeingmon.hardware.asi.fake import FakeAsiSdk, FakeFrameInfo, default_pixels
from tests.hardware.asi_support import FAST, TINY, Rig, make_rig, reference_profile

RAW16, RAW8 = PixelFormat.RAW16, PixelFormat.RAW8
ASI_RAW16 = AsiImageType.RAW16


class AdvancingApi:
    """Wraps the fake SDK and advances the clock by 1 ms after every call but the frame read.

    It records the clock reading at the moment the frame read returned, so a test can tell
    whether the driver stamped the frame there or later.
    """

    def __init__(self, sdk: FakeAsiSdk) -> None:
        self._sdk = sdk
        self._clock = cast(VirtualClock, sdk.clock)
        self.read_returned_utc_ns: list[int] = []

    def __getattr__(self, name: str) -> Any:
        method = getattr(self._sdk, name)
        clock = self._clock

        def call(*args: Any, **kwargs: Any) -> Any:
            result = method(*args, **kwargs)
            if name == "get_video_data":
                self.read_returned_utc_ns.append(clock.utc_ns())
            else:
                clock.advance(0.001)
            return result

        return call


class TestTimeStamps:
    def test_arrival_is_stamped_right_after_the_read_returns(self) -> None:
        proxies: list[AdvancingApi] = []

        def wrap(sdk: FakeAsiSdk) -> AdvancingApi:
            proxies.append(AdvancingApi(sdk))
            return proxies[0]

        rig = make_rig(api=wrap, geometry_check_interval=1).streaming(TINY)
        proxies[0].read_returned_utc_ns.clear()
        frame = rig.driver.read_frame(1.0)
        assert frame.t_arrival_ns == proxies[0].read_returned_utc_ns[-1]
        assert rig.clock.utc_ns() > frame.t_arrival_ns  # the later SDK calls took time

    def test_the_utc_time_is_the_arrival_minus_the_period_plus_half_the_exposure(
        self,
    ) -> None:
        rig = make_rig().streaming(FAST)
        active_period_s = reference_profile().frame_period_s("bin1", 128, 2000)
        frame = rig.driver.read_frame(1.0)
        expected = frame.t_arrival_ns - round(active_period_s * NS_PER_S) + 1_000_000
        assert frame.t_utc_ns == expected

    def test_a_synchronized_clock_gives_an_estimated_time_with_its_error_bound(self) -> None:
        rig = make_rig(time_error_ms=5.0).streaming(FAST)
        rig.clock.set_status(ClockStatus(synchronized=True, error_bound_ns=2_000_000, source="t"))
        rig.driver.configure(FAST)
        rig.driver.start()
        frame = rig.driver.read_frame(1.0)
        assert frame.t_quality is TimeQuality.ESTIMATED
        assert frame.t_err_ns == 2_000_000 + 5_000_000
        assert not frame.flags & FrameFlag.TIME_INVALID

    def test_an_unsynchronized_clock_marks_the_time_invalid(self) -> None:
        rig = make_rig().opened()
        rig.clock.set_status(ClockStatus(synchronized=False, error_bound_ns=None, source="t"))
        rig.driver.configure(FAST)
        rig.driver.start()
        frame = rig.driver.read_frame(1.0)
        assert frame.t_quality is TimeQuality.INVALID
        assert frame.flags & FrameFlag.TIME_INVALID

    def test_the_clock_status_is_read_at_the_interval_and_not_for_every_frame(self) -> None:
        rig = make_rig(status_interval_s=5.0).opened()
        calls = []
        original = rig.clock.status

        def counting() -> ClockStatus:
            calls.append(1)
            return original()

        rig.clock.status = counting  # type: ignore[method-assign]
        rig.driver.configure(FAST)
        rig.driver.start()
        for _ in range(50):
            rig.driver.read_frame(1.0)
        assert len(calls) == 1
        rig.clock.advance(6.0)
        rig.driver.read_frame(1.0)
        assert len(calls) == 2

    def test_a_snapshot_time_is_the_middle_of_the_exposure(self) -> None:
        rig = make_rig().opened()
        rig.driver.configure(
            StreamConfig(
                mode="bin2",
                exposure_us=10_000_000,
                gain=120,
                kind=StreamKind.SNAPSHOT,
                roi=Roi(0, 0, 64, 64),
            )
        )
        rig.driver.start()
        started = rig.clock.utc_ns()
        frame = rig.driver.read_frame(60.0)
        assert frame.t_utc_ns == started + 5_000_000_000
        assert frame.t_arrival_ns > started + 10_000_000_000


class TestBoundedWaits:
    def wait_ms(self, rig: Rig) -> int:
        return int(rig.sdk.calls_named("get_video_data")[-1][-1])

    def test_a_read_waits_twice_the_period_plus_500_ms(self) -> None:
        rig = make_rig().streaming(FAST)
        rig.driver.read_frame(10.0)
        period_s = reference_profile().frame_period_s("bin1", 128, 2000)  # 11.3 ms
        assert self.wait_ms(rig) == math.ceil((2 * period_s + 0.5) * 1000)

    def test_a_long_exposure_sets_the_bound(self) -> None:
        rig = make_rig().streaming(
            StreamConfig(mode="bin1", exposure_us=400_000, gain=1, roi=Roi(0, 0, 8, 2))
        )
        rig.driver.read_frame(10.0)
        assert self.wait_ms(rig) == 1300  # 2 x 0.4 s + 0.5 s

    def test_the_callers_timeout_caps_the_wait(self) -> None:
        rig = make_rig().streaming(FAST)
        rig.driver.read_frame(0.2)
        assert self.wait_ms(rig) == 200

    def test_a_stalled_stream_times_out_after_the_bound_and_not_the_callers_timeout(self) -> None:
        rig = make_rig().streaming(TINY)
        rig.sdk.stall_reads(1)
        started = rig.clock.monotonic_ns()
        with pytest.raises(CameraTimeoutError):
            rig.driver.read_frame(60.0)
        waited_s = (rig.clock.monotonic_ns() - started) / NS_PER_S
        assert 0.5 < waited_s < 0.6

    def test_the_slower_of_the_exposure_and_the_readout_sets_the_bound(self) -> None:
        # A full bin1 frame takes 212 ms to read, which is longer than the 2 ms exposure.
        rig = make_rig().opened()
        rig.driver.configure(StreamConfig(mode="bin1", exposure_us=2000, gain=1))
        period_s = reference_profile().frame_period_s("bin1", 5644, 2000)
        assert period_s > 0.2
        rig.driver.start()
        rig.driver.read_frame(10.0)
        assert self.wait_ms(rig) == math.ceil((2 * period_s + 0.5) * 1000)


class TestControls:
    def test_the_applied_values_come_from_the_camera(self) -> None:
        rig = make_rig().opened()
        active = rig.driver.configure(
            StreamConfig(
                mode="bin1",
                exposure_us=2000,
                gain=120,
                offset=30,
                bandwidth_pct=80,
                roi=Roi(0, 0, 16, 8),
            )
        )
        assert (active.config.gain, active.config.exposure_us) == (120, 2000)
        assert (active.config.offset, active.config.bandwidth_pct) == (30, 80)
        assert rig.sdk.control(AsiControl.OFFSET) == 30
        assert rig.sdk.control(AsiControl.BANDWIDTH_OVERLOAD) == 80

    def test_a_value_that_the_request_leaves_out_is_read_back(self) -> None:
        rig = make_rig().opened()
        active = rig.driver.configure(StreamConfig(mode="bin1", exposure_us=2000, gain=1))
        assert active.config.offset == 10  # the camera default
        assert active.config.bandwidth_pct == 50

    @pytest.mark.parametrize(
        ("field", "value"),
        [("gain", 9999), ("exposure_us", 31), ("offset", 300), ("bandwidth_pct", 10)],
    )
    def test_a_value_outside_the_camera_range_is_refused_before_the_sdk_sees_it(
        self, field: str, value: int
    ) -> None:
        rig = make_rig().opened()
        values: dict[str, Any] = {
            "mode": "bin1",
            "exposure_us": 2000,
            "gain": 1,
            "roi": Roi(0, 0, 8, 2),
        }
        values[field] = value
        with pytest.raises(CameraConfigError, match=field):
            rig.driver.configure(StreamConfig(**values))
        assert rig.sdk.calls_named("set_roi_format") == []

    def test_a_value_that_the_camera_changes_silently_is_an_error(self) -> None:
        rig = make_rig(sdk={"silent_gain_limit": 300}).opened()
        with pytest.raises(CameraConfigError, match="gain 300 for the request 450"):
            rig.driver.configure(StreamConfig(mode="bin1", exposure_us=2000, gain=450))

    def test_high_speed_mode_changes_the_adc_depth_and_the_frame_period(self) -> None:
        rig = make_rig().opened()
        active = rig.driver.configure(
            StreamConfig(
                mode="bin1", exposure_us=2000, gain=1, roi=Roi(0, 0, 16, 8), high_speed=True
            )
        )
        assert rig.sdk.control(AsiControl.HIGH_SPEED_MODE) == 1
        assert active.adc_bits == 10
        assert active.frame_period_s == pytest.approx(5.0e-3 + 8 * 30.1e-6)
        rig.driver.start()
        assert rig.driver.read_frame(1.0).adc_bits == 10

    def test_high_speed_mode_is_switched_off_again(self) -> None:
        rig = make_rig().opened()
        rig.driver.configure(
            StreamConfig(
                mode="bin1", exposure_us=2000, gain=1, roi=Roi(0, 0, 16, 8), high_speed=True
            )
        )
        active = rig.driver.configure(
            StreamConfig(mode="bin1", exposure_us=2000, gain=1, roi=Roi(0, 0, 16, 8))
        )
        assert rig.sdk.control(AsiControl.HIGH_SPEED_MODE) == 0
        assert active.adc_bits == 12

    def test_a_mode_without_a_high_speed_variant_is_refused(self) -> None:
        profile = reference_profile()
        assert profile.mode("bin1").has_high_speed  # the reference modes have one
        rig = make_rig().opened()
        with pytest.raises(CameraConfigError, match="unknown readout mode"):
            rig.driver.configure(
                StreamConfig(mode="bin7", exposure_us=2000, gain=1, high_speed=True)
            )

    def test_a_pixel_format_that_the_camera_lacks_is_refused(self) -> None:
        rig = make_rig().opened()
        assert RAW16 in rig.driver.capabilities().pixel_formats
        rig.driver.configure(
            StreamConfig(
                mode="bin1", exposure_us=2000, gain=1, roi=Roi(0, 0, 8, 2), pixel_format=RAW8
            )
        )


class TestGeometry:
    def test_the_mode_change_function_runs_in_order(self) -> None:
        rig = make_rig().opened()
        rig.driver.configure(FAST)
        rig.driver.start()
        names = [name for name, _ in rig.sdk.calls]
        order = [
            "set_control_value",
            "set_roi_format",
            "set_start_position",
            "get_roi_format",
            "get_start_position",
            "start_video_capture",
        ]
        positions = [names.index(name) for name in order]
        assert positions == sorted(positions)
        assert rig.sdk.calls_named("set_roi_format")[0][1:4] == (128, 128, 1)

    def test_a_silent_change_is_corrected_by_applying_the_geometry_again(self) -> None:
        rig = make_rig().opened()
        rig.sdk.corrupt_next_roi(24, 8)
        active = rig.driver.configure(TINY)
        assert active.config.roi == Roi(8, 4, 16, 8)
        assert rig.sdk.roi == (8, 4, 16, 8)
        assert len(rig.sdk.calls_named("set_roi_format")) == 2
        assert rig.event_kinds() == ["camera.geometry_corrected"]

    def test_a_silent_change_that_persists_is_an_error(self) -> None:
        rig = make_rig().opened()
        rig.sdk.corrupt_next_roi(24, 8, always=True)
        with pytest.raises(CameraConfigError, match="different geometry"):
            rig.driver.configure(TINY)
        with pytest.raises(CameraStateError):  # a failed configure leaves no stream
            rig.driver.start()

    def test_a_change_in_the_middle_of_a_stream_is_detected_and_stops_capture(self) -> None:
        rig = make_rig(geometry_check_interval=1).streaming(TINY)
        rig.driver.read_frame(1.0)
        rig.sdk.change_geometry(8, 8)  # smaller, so the frame still fits the buffer
        with pytest.raises(CameraConfigError, match="different ROI"):
            rig.driver.read_frame(1.0)
        assert not rig.sdk.video_active
        assert "camera.geometry_changed" in rig.event_kinds()

    def test_a_larger_frame_than_the_buffer_is_a_geometry_change_too(self) -> None:
        rig = make_rig().streaming(TINY)
        rig.sdk.change_geometry(32, 8)
        with pytest.raises(CameraConfigError, match="buffer"):
            rig.driver.read_frame(1.0)
        assert not rig.sdk.video_active

    def test_the_check_interval_limits_how_often_the_geometry_is_read(self) -> None:
        rig = make_rig(geometry_check_interval=4).streaming(TINY)
        before = len(rig.sdk.calls_named("get_roi_format"))
        for _ in range(8):
            rig.driver.read_frame(1.0)
        assert len(rig.sdk.calls_named("get_roi_format")) - before == 2

    def test_the_check_can_be_turned_off(self) -> None:
        rig = make_rig(geometry_check_interval=0).streaming(TINY)
        before = len(rig.sdk.calls_named("get_roi_format"))
        for _ in range(8):
            rig.driver.read_frame(1.0)
        assert len(rig.sdk.calls_named("get_roi_format")) == before

    def test_a_camera_that_aligns_the_start_position_reports_where_the_roi_is(self) -> None:
        rig = make_rig(sdk={"start_alignment": 4}).opened()
        active = rig.driver.configure(
            StreamConfig(mode="bin1", exposure_us=2000, gain=1, roi=Roi(103, 202, 16, 8))
        )
        assert active.config.roi == Roi(100, 200, 16, 8)
        rig.driver.start()
        assert rig.driver.move_roi(203, 306) == Roi(200, 304, 16, 8)
        assert rig.driver.read_frame(1.0).roi == Roi(200, 304, 16, 8)


class TestDiscards:
    def test_start_drops_the_first_frames_of_a_video_stream(self) -> None:
        rig = make_rig(discard_frames=2).opened()
        rig.driver.configure(TINY)
        rig.driver.start()
        frame = rig.driver.read_frame(1.0)
        counts = default_pixels(FakeFrameInfo(2, 16, 8, 1, ASI_RAW16, 8, 4, 2000, 120, 12))
        np.testing.assert_array_equal(frame.data, counts << 4)  # frames 0 and 1 are gone
        assert frame.seq == 0

    def test_no_discard_keeps_the_first_frame(self) -> None:
        rig = make_rig(discard_frames=0).opened()
        rig.driver.configure(TINY)
        rig.driver.start()
        counts = default_pixels(FakeFrameInfo(0, 16, 8, 1, ASI_RAW16, 8, 4, 2000, 120, 12))
        np.testing.assert_array_equal(rig.driver.read_frame(1.0).data, counts << 4)

    def test_frames_from_the_old_position_never_carry_the_new_roi(self) -> None:
        rig = make_rig(discard_frames=0, roi_move_discard_frames=2).opened()
        rig.driver.configure(TINY)
        rig.driver.start()
        rig.clock.advance(2.5 * rig.sdk.frame_period_s())  # frames 0 and 1 wait in the buffer
        moved = rig.driver.move_roi(40, 20)
        frame = rig.driver.read_frame(1.0)
        assert frame.roi == moved == Roi(40, 20, 16, 8)
        counts = default_pixels(FakeFrameInfo(2, 16, 8, 1, ASI_RAW16, 40, 20, 2000, 120, 12))
        np.testing.assert_array_equal(frame.data, counts << 4)

    def test_without_the_discard_a_buffered_frame_would_carry_the_wrong_roi(self) -> None:
        rig = make_rig(discard_frames=0, roi_move_discard_frames=0).opened()
        rig.driver.configure(TINY)
        rig.driver.start()
        rig.clock.advance(2.5 * rig.sdk.frame_period_s())
        moved = rig.driver.move_roi(40, 20)
        frame = rig.driver.read_frame(1.0)
        counts = default_pixels(FakeFrameInfo(0, 16, 8, 1, ASI_RAW16, 40, 20, 2000, 120, 12))
        assert frame.roi == moved
        assert not np.array_equal(frame.data, counts << 4)  # the pixels came from (8, 4)

    def test_a_late_read_reports_the_frames_that_the_camera_lost(self) -> None:
        rig = make_rig(discard_frames=0).opened()
        rig.driver.configure(TINY)
        rig.driver.start()
        rig.clock.advance(10.5 * rig.sdk.frame_period_s())  # 10 frames complete, 3 fit the buffer
        frame = rig.driver.read_frame(1.0)
        assert frame.dropped_before == 7
        assert rig.driver.read_frame(1.0).dropped_before == 0
        assert rig.driver.dropped_frames() == 7

    def test_drops_during_startup_do_not_count_against_the_first_frame(self) -> None:
        rig = make_rig(discard_frames=1).opened()
        rig.driver.configure(TINY)
        rig.sdk.lose_frames(0)
        rig.driver.start()
        assert rig.driver.read_frame(1.0).dropped_before == 0


class TestTemperature:
    def test_the_first_reading_waits_out_the_zero_of_the_first_250_ms(self) -> None:
        rig = make_rig().opened()
        started = rig.clock.monotonic_ns()
        assert rig.driver.read_temperature_c() == pytest.approx(18.3)
        assert (rig.clock.monotonic_ns() - started) / NS_PER_S == pytest.approx(0.3, abs=0.01)

    def test_a_later_reading_of_exactly_zero_is_a_real_temperature(self) -> None:
        rig = make_rig(sdk={"temperature_c": 0.0}).opened()
        rig.clock.advance(1.0)
        assert rig.driver.read_temperature_c() == 0.0

    def test_frames_carry_no_temperature_until_the_sensor_is_valid(self) -> None:
        rig = make_rig().opened()
        rig.driver.configure(TINY)
        rig.driver.start()
        first = rig.driver.read_frame(1.0)
        assert first.temperature_c is None  # inside the warm-up window
        for _ in range(40):
            last = rig.driver.read_frame(1.0)
        assert last.temperature_c == pytest.approx(18.3)

    def test_a_running_stream_refreshes_the_temperature_at_the_interval(self) -> None:
        rig = make_rig(temperature_interval_s=5.0).opened()
        rig.clock.advance(1.0)
        rig.driver.configure(TINY)
        rig.driver.start()
        for _ in range(100):
            rig.driver.read_frame(1.0)
        reads = len(rig.sdk.calls_named("get_control_value"))
        rig.sdk.temperature_c = 19.0
        rig.clock.advance(6.0)
        frame = rig.driver.read_frame(1.0)
        assert frame.temperature_c == pytest.approx(19.0)
        assert len(rig.sdk.calls_named("get_control_value")) > reads

    def test_a_camera_without_a_sensor_reports_none(self) -> None:
        rig = make_rig(sdk={"temperature_c": None}).opened()
        assert rig.driver.read_temperature_c() is None
        rig.driver.configure(TINY)
        rig.driver.start()
        assert rig.driver.read_frame(1.0).temperature_c is None
        assert not rig.driver.open().has_temperature
        assert "camera.profile_mismatch" in rig.event_kinds()


class TestCapabilities:
    def test_the_caps_come_from_the_camera_and_the_profile(self) -> None:
        caps = make_rig().opened().driver.capabilities()
        assert caps.gain_range == (0, 570)
        assert caps.exposure_us_range == (32, 2_000_000_000)
        assert caps.bins == (1, 2)
        assert caps.pixel_formats == (RAW8, RAW16)
        assert caps.offset_range == (0, 255)
        assert (caps.roi_width_multiple, caps.roi_height_multiple) == (8, 2)

    def test_the_caps_need_an_open_camera(self) -> None:
        with pytest.raises(CameraStateError):
            make_rig().driver.capabilities()

    def test_a_camera_that_differs_from_the_profile_is_reported(self) -> None:
        rig = make_rig(sdk={"max_width": 4144, "max_height": 2822}).opened()
        assert rig.event_kinds() == ["camera.profile_mismatch"]
        assert rig.events[0].detail == {"camera_size": [4144, 2822], "profile_size": [8288, 5644]}


class TestSnapshots:
    def test_a_snapshot_frame_carries_a_fresh_temperature_and_no_drops(self) -> None:
        rig = make_rig().opened()
        rig.clock.advance(1.0)
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
        assert frame.temperature_c == pytest.approx(18.3)
        assert frame.dropped_before == 0
        assert frame.adc_bits == 14

    def test_a_snapshot_period_includes_the_readout(self) -> None:
        rig = make_rig().opened()
        active = rig.driver.configure(
            StreamConfig(
                mode="bin2",
                exposure_us=2_000_000,
                gain=1,
                kind=StreamKind.SNAPSHOT,
                roi=Roi(0, 0, 64, 64),
            )
        )
        assert active.frame_period_s == pytest.approx(2.0 + 1.4e-3 + 64 * 21.3e-6)

    def test_an_exposure_that_does_not_finish_in_time_times_out_and_can_be_awaited_again(
        self,
    ) -> None:
        rig = make_rig().opened()
        rig.driver.configure(
            StreamConfig(
                mode="bin2",
                exposure_us=10_000_000,
                gain=1,
                kind=StreamKind.SNAPSHOT,
                roi=Roi(0, 0, 64, 64),
            )
        )
        rig.driver.start()
        with pytest.raises(CameraTimeoutError):
            rig.driver.read_frame(1.0)
        assert rig.driver.read_frame(30.0).exposure_us == 10_000_000

    def test_a_failed_exposure_is_a_camera_error_and_ends_the_snapshot(self) -> None:
        rig = make_rig().opened()
        rig.driver.configure(
            StreamConfig(
                mode="bin2",
                exposure_us=1_000_000,
                gain=1,
                kind=StreamKind.SNAPSHOT,
                roi=Roi(0, 0, 64, 64),
            )
        )
        rig.sdk.fail_next_exposure()
        rig.driver.start()
        with pytest.raises(Exception, match="exposure failed"):
            rig.driver.read_frame(5.0)
        with pytest.raises(CameraStateError):
            rig.driver.read_frame(1.0)

    def test_stop_aborts_the_exposure(self) -> None:
        rig = make_rig().opened()
        rig.driver.configure(
            StreamConfig(
                mode="bin2",
                exposure_us=60_000_000,
                gain=1,
                kind=StreamKind.SNAPSHOT,
                roi=Roi(0, 0, 64, 64),
            )
        )
        rig.driver.start()
        before = len(rig.sdk.calls_named("stop_exposure"))
        rig.driver.stop()
        assert len(rig.sdk.calls_named("stop_exposure")) == before + 1
        with pytest.raises(CameraStateError):
            rig.driver.read_frame(1.0)

    def test_start_twice_takes_one_exposure(self) -> None:
        rig = make_rig().opened()
        rig.driver.configure(
            StreamConfig(
                mode="bin2",
                exposure_us=1_000_000,
                gain=1,
                kind=StreamKind.SNAPSHOT,
                roi=Roi(0, 0, 64, 64),
            )
        )
        rig.driver.start()
        rig.driver.start()
        assert len(rig.sdk.calls_named("start_exposure")) == 1


class TestOptions:
    def test_every_option_has_a_default(self) -> None:
        options = AsiOptions()
        assert options.camera_index == 0
        assert options.usb_vendor_id == 0x03C3
        assert options.watchdog is True

    def test_a_mapping_validates_and_names_the_bad_key_without_its_value(self) -> None:
        with pytest.raises(CameraConfigError) as raised:
            AsiOptions.from_mapping({"camera_index": -1, "library_path": 7})
        message = str(raised.value)
        assert "camera_index" in message
        assert "library_path" in message
        assert "-1" not in message

    def test_an_unknown_key_is_an_error(self) -> None:
        with pytest.raises(CameraConfigError, match="gaen"):
            AsiOptions.from_mapping({"gaen": 1})

    def test_the_usb_reset_mode_is_one_of_four(self) -> None:
        with pytest.raises(CameraConfigError, match="usb_reset"):
            AsiOptions.from_mapping({"usb_reset": "power"})


class TestCreate:
    def test_a_missing_profile_is_refused(self) -> None:
        from seeingmon.clock import VirtualClock

        with pytest.raises(CameraConfigError, match="profile"):
            create(profile=None, clock=VirtualClock(), options={})

    def test_a_missing_library_is_reported_with_the_two_places_to_set_it(
        self, tmp_path: Path
    ) -> None:
        from seeingmon.clock import VirtualClock

        with pytest.raises(AsiLibraryError, match="library_path"):
            create(
                profile=reference_profile(),
                clock=VirtualClock(),
                options={"library_path": str(tmp_path / "missing")},
            )

    def test_create_builds_a_driver_on_the_loaded_library(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from seeingmon.clock import VirtualClock

        clock = VirtualClock()
        sdk = FakeAsiSdk(clock)
        monkeypatch.setattr("seeingmon.drivers.asi.load_asi_api", lambda path: sdk)
        driver = create(
            profile=reference_profile(),
            clock=clock,
            options={"watchdog": False, "usb_reset": "off"},
        )
        assert isinstance(driver, AsiDriver)
        assert driver.open().model == "ZWO ASI294MM (fake)"

    def test_the_options_reach_the_driver(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from seeingmon.clock import VirtualClock

        clock = VirtualClock()
        sdk = FakeAsiSdk(clock)
        seen: list[str | None] = []

        def loader(path: str | None) -> FakeAsiSdk:
            seen.append(path)
            return sdk

        monkeypatch.setattr("seeingmon.drivers.asi.load_asi_api", loader)
        driver = create(
            profile=reference_profile(),
            clock=clock,
            options={"library_path": "a-library", "watchdog": False, "discard_frames": 0},
        )
        driver.open()
        driver.configure(TINY)
        driver.start()
        counts = default_pixels(FakeFrameInfo(0, 16, 8, 1, ASI_RAW16, 8, 4, 2000, 120, 12))
        np.testing.assert_array_equal(driver.read_frame(1.0).data, counts << 4)
        assert seen == ["a-library"]


class TestOpen:
    def test_open_stops_a_capture_that_a_previous_process_left_running(self) -> None:
        rig = make_rig()
        rig.sdk.open_camera(0)
        rig.sdk.init_camera(0)
        rig.sdk.start_video_capture(0)
        assert rig.sdk.video_active
        rig.driver.open()
        assert not rig.sdk.video_active

    def test_open_is_idempotent(self) -> None:
        rig = make_rig()
        first = rig.driver.open()
        assert rig.driver.open() == first
        assert len(rig.sdk.calls_named("open_camera")) == 1

    def test_open_finds_the_camera_by_index(self) -> None:
        rig = make_rig(camera_index=1)
        with pytest.raises(CameraDisconnectedError, match="index 1"):
            rig.driver.open()
