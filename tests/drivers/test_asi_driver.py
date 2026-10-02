"""The `asi` driver on a fake SDK: time stamps, bounded waits, controls, geometry, and options."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from seeingmon.clock import NS_PER_S, ClockStatus, VirtualClock
from seeingmon.drivers import (
    CameraConfigError,
    CameraDisconnectedError,
    CameraError,
    CameraStateError,
    CameraTimeoutError,
    RecoveryLevel,
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
from seeingmon.hardware.asi.api import AsiControl, AsiErrorCode, AsiImageType, AsiLibraryError
from seeingmon.hardware.asi.fake import (
    DEFAULT_ADC_BITS,
    FakeAsiSdk,
    FakeCameraState,
    FakeFrameInfo,
    default_pixels,
)
from seeingmon.profile import derived
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

    def test_a_value_that_the_request_leaves_out_comes_from_the_options(self) -> None:
        rig = make_rig().opened()
        active = rig.driver.configure(StreamConfig(mode="bin1", exposure_us=2000, gain=1))
        assert active.config.offset == 10  # the camera's default, because the option is None
        assert active.config.bandwidth_pct == 100  # the default of the option
        assert rig.sdk.control(AsiControl.BANDWIDTH_OVERLOAD) == 100

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
        fast = reference_profile().mode("bin1", high_speed=True)
        assert active.frame_period_s == pytest.approx(derived.frame_period_s(fast, 8, 2000))
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


PERSISTENT = (
    AsiControl.HIGH_SPEED_MODE,
    AsiControl.FLIP,
    AsiControl.BANDWIDTH_OVERLOAD,
    AsiControl.OFFSET,
    AsiControl.GAIN,
    AsiControl.EXPOSURE,
)


def stale_state() -> FakeCameraState:
    """What another program can leave in a camera: a half bandwidth, a flip, an offset, high-speed
    mode, a gain, an exposure, and the automatic mode of three controls."""
    state = FakeCameraState(
        {
            AsiControl.BANDWIDTH_OVERLOAD: 50,
            AsiControl.FLIP: 3,
            AsiControl.OFFSET: 20,
            AsiControl.HIGH_SPEED_MODE: 1,
            AsiControl.GAIN: 99,
            AsiControl.EXPOSURE: 123_456,
        }
    )
    state.automatic.update({AsiControl.GAIN, AsiControl.EXPOSURE, AsiControl.BANDWIDTH_OVERLOAD})
    return state


def applied(rig: Rig) -> dict[str, int]:
    return {control.name.lower(): rig.sdk.control(control) for control in PERSISTENT}


class TestPersistentControls:
    """The camera keeps its controls until it loses power. A driver that leaves one as it finds it
    inherits what the last process or program set: half the frame rate, a mirrored frame, or
    another bias level."""

    def test_a_stream_applies_every_persistent_control_over_a_stale_camera(self) -> None:
        state = stale_state()
        rig = make_rig(sdk={"state": state}).opened()
        active = rig.driver.configure(
            StreamConfig(mode="bin1", exposure_us=2000, gain=120, roi=Roi(0, 0, 16, 8))
        )
        assert applied(rig) == {
            "high_speed_mode": 0,
            "flip": 0,
            "bandwidth_overload": 100,
            "offset": 10,  # the default of the camera, because the option is None
            "gain": 120,
            "exposure": 2000,
        }
        assert state.automatic == set()  # every control is back in manual mode
        config = active.config
        assert (config.bandwidth_pct, config.offset, config.gain, config.exposure_us) == (
            100,
            10,
            120,
            2000,
        )
        assert config.high_speed is False

    def test_the_bandwidth_option_none_leaves_the_control_and_reports_it(self) -> None:
        state = stale_state()
        rig = make_rig(sdk={"state": state}, bandwidth_pct=None).opened()
        active = rig.driver.configure(
            StreamConfig(mode="bin1", exposure_us=2000, gain=120, roi=Roi(0, 0, 16, 8))
        )
        assert rig.sdk.control(AsiControl.BANDWIDTH_OVERLOAD) == 50
        assert active.config.bandwidth_pct == 50
        assert rig.sdk.control(AsiControl.FLIP) == 0  # the other controls still apply
        assert rig.sdk.control(AsiControl.OFFSET) == 10

    def test_the_options_apply_when_the_stream_asks_for_nothing(self) -> None:
        rig = make_rig(bandwidth_pct=70, offset=15).opened()
        active = rig.driver.configure(
            StreamConfig(mode="bin1", exposure_us=2000, gain=1, roi=Roi(0, 0, 16, 8))
        )
        assert (active.config.bandwidth_pct, active.config.offset) == (70, 15)
        assert rig.sdk.control(AsiControl.BANDWIDTH_OVERLOAD) == 70
        assert rig.sdk.control(AsiControl.OFFSET) == 15

    def test_a_stream_overrides_the_options_for_that_stream_only(self) -> None:
        rig = make_rig(bandwidth_pct=70, offset=15).opened()
        override = StreamConfig(
            mode="bin1",
            exposure_us=2000,
            gain=1,
            roi=Roi(0, 0, 16, 8),
            bandwidth_pct=60,
            offset=25,
        )
        active = rig.driver.configure(override)
        assert (active.config.bandwidth_pct, active.config.offset) == (60, 25)
        again = rig.driver.configure(replace(override, bandwidth_pct=None, offset=None))
        assert (again.config.bandwidth_pct, again.config.offset) == (70, 15)

    def test_every_configure_applies_the_controls_again(self) -> None:
        rig = make_rig().opened()
        config = StreamConfig(mode="bin1", exposure_us=2000, gain=120, roi=Roi(0, 0, 16, 8))
        rig.driver.configure(config)
        rig.sdk.state.controls.update({AsiControl.BANDWIDTH_OVERLOAD: 40, AsiControl.FLIP: 2})
        rig.driver.configure(config)  # another program changed the camera in between
        assert rig.sdk.control(AsiControl.BANDWIDTH_OVERLOAD) == 100
        assert rig.sdk.control(AsiControl.FLIP) == 0

    def test_a_new_process_finds_the_camera_as_the_last_one_left_it_and_sets_it_again(
        self,
    ) -> None:
        state = FakeCameraState()
        first = make_rig(sdk={"state": state}, bandwidth_pct=60).opened()
        first.driver.configure(TINY)
        first.driver.close()
        assert state.controls[AsiControl.BANDWIDTH_OVERLOAD] == 60  # the camera keeps it
        second = make_rig(sdk={"state": state}).opened()  # a process with the default option
        assert second.sdk.control(AsiControl.BANDWIDTH_OVERLOAD) == 60  # until it configures
        second.driver.configure(TINY)
        assert second.sdk.control(AsiControl.BANDWIDTH_OVERLOAD) == 100

    @pytest.mark.parametrize("level", [RecoveryLevel.REOPEN, RecoveryLevel.USB_RESET])
    def test_recovery_applies_the_controls_again(self, level: RecoveryLevel) -> None:
        rig = make_rig().streaming(FAST)
        rig.sdk.state.controls.update({AsiControl.BANDWIDTH_OVERLOAD: 50, AsiControl.FLIP: 3})
        rig.driver.recover(level)
        assert rig.sdk.control(AsiControl.BANDWIDTH_OVERLOAD) == 100
        assert rig.sdk.control(AsiControl.FLIP) == 0
        assert rig.driver.read_frame(1.0).flags & FrameFlag.RECOVERED

    def test_a_control_that_stays_automatic_is_an_error(self) -> None:
        rig = make_rig().opened()
        rig.sdk.keep_auto(AsiControl.EXPOSURE)
        with pytest.raises(CameraConfigError, match="exposure_us in automatic mode"):
            rig.driver.configure(TINY)

    def test_the_driver_asks_for_manual_mode_in_every_write(self) -> None:
        rig = make_rig().opened()
        rig.driver.configure(TINY)
        writes = rig.sdk.calls_named("set_control_value")
        assert len(writes) >= len(PERSISTENT)
        assert all(call[3] is False for call in writes)  # the `auto` argument

    def test_a_camera_without_flip_and_bandwidth_controls_is_not_an_error(self) -> None:
        rig = make_rig(sdk={"without": (AsiControl.FLIP, AsiControl.BANDWIDTH_OVERLOAD)}).opened()
        active = rig.driver.configure(TINY)
        assert active.config.bandwidth_pct is None  # nothing to apply and nothing to report
        written = {call[1] for call in rig.sdk.calls_named("set_control_value")}
        assert written == {
            int(control)
            for control in (
                AsiControl.HIGH_SPEED_MODE,
                AsiControl.OFFSET,
                AsiControl.GAIN,
                AsiControl.EXPOSURE,
            )
        }

    def test_a_stream_that_asks_for_a_control_the_camera_lacks_is_refused(self) -> None:
        rig = make_rig(sdk={"without": (AsiControl.BANDWIDTH_OVERLOAD,)}).opened()
        with pytest.raises(CameraConfigError, match="no bandwidth_pct control"):
            rig.driver.configure(replace(TINY, bandwidth_pct=80))

    def test_the_new_options_have_defaults_and_limits(self) -> None:
        options = AsiOptions()
        assert (options.bandwidth_pct, options.offset) == (100, None)
        assert AsiOptions(bandwidth_pct=None).bandwidth_pct is None
        for bad in ({"bandwidth_pct": 0}, {"bandwidth_pct": 101}, {"offset": -1}):
            with pytest.raises(CameraConfigError):
                AsiOptions.from_mapping(bad)


class TestSaveAndRestoreSettings:
    """A tool that shares the camera with another program saves the settings, and puts them back."""

    @staticmethod
    def owner_state() -> FakeCameraState:
        """The state that the other program left: not the defaults, and one control automatic."""
        state = FakeCameraState(
            {
                AsiControl.BANDWIDTH_OVERLOAD: 60,
                AsiControl.FLIP: 2,
                AsiControl.OFFSET: 33,
                AsiControl.GAIN: 77,
                AsiControl.EXPOSURE: 12_345,
                AsiControl.HIGH_SPEED_MODE: 1,
            }
        )
        state.automatic.add(AsiControl.EXPOSURE)
        return state

    def test_the_saved_controls_are_the_writable_ones_with_their_names(self) -> None:
        rig = make_rig(sdk={"state": self.owner_state()}).opened()
        saved = rig.driver.save_settings()
        names = {control.name for control in saved.controls.values()}
        assert names == {
            "Gain",
            "Exposure",
            "Gamma",
            "Offset",
            "BandWidth",
            "HighSpeedMode",
            "Flip",
        }
        assert saved.controls[int(AsiControl.BANDWIDTH_OVERLOAD)].value == 60
        assert saved.controls[int(AsiControl.EXPOSURE)].automatic is True
        assert saved.roi.width == 8288  # the fake starts with the full frame
        assert saved.start == (0, 0)

    def test_restore_puts_back_what_a_stream_changed(self) -> None:
        state = self.owner_state()
        rig = make_rig(sdk={"state": state}).opened()
        before = (dict(state.controls), set(state.automatic))  # with the defaults filled in
        saved = rig.driver.save_settings()
        rig.driver.configure(FAST)
        rig.driver.start()
        rig.driver.read_frame(1.0)
        assert rig.sdk.control(AsiControl.BANDWIDTH_OVERLOAD) == 100  # the stream changed it
        assert rig.driver.restore_settings(saved) == []
        assert (dict(state.controls), set(state.automatic)) == before
        assert rig.sdk.roi == (0, 0, 8288, 5644)  # the geometry came back too
        assert not rig.sdk.video_active

    def test_restore_drops_the_stream(self) -> None:
        rig = make_rig().opened()
        saved = rig.driver.save_settings()
        rig.driver.configure(TINY)
        rig.driver.restore_settings(saved)
        with pytest.raises(CameraStateError, match="start before configure"):
            rig.driver.start()

    def test_restore_writes_only_what_differs(self) -> None:
        rig = make_rig().opened()
        saved = rig.driver.save_settings()
        rig.sdk.calls.clear()
        assert rig.driver.restore_settings(saved) == []
        assert rig.sdk.calls_named("set_control_value") == []
        assert rig.sdk.calls_named("set_roi_format") == []

    def test_restore_tries_every_setting_when_one_write_fails(self) -> None:
        state = self.owner_state()
        rig = make_rig(sdk={"state": state}).opened()
        saved = rig.driver.save_settings()
        rig.driver.configure(FAST)
        rig.sdk.fail_next("set_control_value", AsiErrorCode.GENERAL_ERROR)
        problems = rig.driver.restore_settings(saved)
        assert len(problems) == 1
        assert state.controls[AsiControl.GAIN] != 120 or problems  # the others were still written
        changed_back = [
            control
            for control, value in saved.controls.items()
            if rig.sdk.control(AsiControl(control)) == value.value
        ]
        assert len(changed_back) >= len(saved.controls) - 1

    def test_restore_reports_a_failed_geometry_and_still_returns(self) -> None:
        rig = make_rig().opened()
        saved = rig.driver.save_settings()
        rig.driver.configure(TINY)
        rig.sdk.fail_next("set_roi_format", AsiErrorCode.INVALID_SIZE)
        assert rig.driver.restore_settings(saved) == ["roi"]

    def test_saving_needs_an_open_camera(self) -> None:
        with pytest.raises(CameraStateError):
            make_rig().driver.save_settings()


class TestControlNumbers:
    """The driver finds a control by its number and checks its name, and the name wins."""

    def test_a_control_that_the_camera_numbers_differently_is_found_by_its_name(self) -> None:
        numbers = {AsiControl.GAIN: 41, AsiControl.HIGH_SPEED_MODE: 40}
        rig = make_rig(sdk={"control_numbers": numbers}).opened()
        assert rig.event_kinds().count("camera.control_renumbered") == 2
        rig.driver.configure(
            StreamConfig(
                mode="bin1", exposure_us=2000, gain=120, roi=Roi(0, 0, 16, 8), high_speed=True
            )
        )
        assert rig.sdk.control(AsiControl.GAIN) == 120
        assert rig.sdk.control(AsiControl.HIGH_SPEED_MODE) == 1
        used = {args[1] for args in rig.sdk.calls_named("set_control_value")}
        assert {40, 41} <= used  # the calls carry the numbers that the camera reports
        assert rig.driver.capabilities().gain_range == (0, 570)

    def test_two_controls_with_swapped_numbers_are_still_told_apart(self) -> None:
        rig = make_rig(
            sdk={"control_numbers": {AsiControl.GAIN: 1, AsiControl.EXPOSURE: 0}}
        ).opened()
        active = rig.driver.configure(
            StreamConfig(mode="bin1", exposure_us=2000, gain=120, roi=Roi(0, 0, 16, 8))
        )
        assert (active.config.gain, active.config.exposure_us) == (120, 2000)

    def test_a_control_with_an_unexpected_name_is_found_by_its_number(self) -> None:
        rig = make_rig(sdk={"control_names": {AsiControl.GAIN: "Verstaerkung"}}).opened()
        assert rig.event_kinds() == []
        active = rig.driver.configure(
            StreamConfig(mode="bin1", exposure_us=2000, gain=120, roi=Roi(0, 0, 16, 8))
        )
        assert active.config.gain == 120

    def test_a_camera_without_a_gain_control_is_refused_at_open(self) -> None:
        rig = make_rig(
            sdk={
                "control_numbers": {AsiControl.GAIN: 77},
                "control_names": {AsiControl.GAIN: "Mystery"},
            }
        )
        with pytest.raises(CameraError, match="no gain control"):
            rig.driver.open()

    def test_the_temperature_follows_the_name_too(self) -> None:
        rig = make_rig(sdk={"control_numbers": {AsiControl.TEMPERATURE: 55}}).opened()
        assert rig.driver.read_temperature_c() == pytest.approx(18.3)


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
        rig.driver.configure(TINY)  # the first configure sets the other format too (the latch)
        sets_before = len(rig.sdk.calls_named("set_roi_format"))
        rig.sdk.corrupt_next_roi(24, 8)
        active = rig.driver.configure(TINY)
        assert active.config.roi == Roi(8, 4, 16, 8)
        assert rig.sdk.roi == (8, 4, 16, 8)
        assert len(rig.sdk.calls_named("set_roi_format")) - sets_before == 2
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

    # The real camera takes up the high-speed flag only when the image format changes, so the
    # driver sets the other format first. These tests check the calls; the camera check in
    # `tests/hardware` checks the effect on a camera.

    @staticmethod
    def format_sets(rig: Rig) -> list[AsiImageType]:
        return [AsiImageType(call[4]) for call in rig.sdk.calls_named("set_roi_format")]

    def test_the_first_configure_sets_the_other_format_and_then_the_requested_one(self) -> None:
        rig = make_rig().opened()
        rig.driver.configure(FAST)
        assert self.format_sets(rig) == [AsiImageType.RAW8, AsiImageType.RAW16]

    def test_a_request_for_raw8_sets_raw16_first(self) -> None:
        rig = make_rig().opened()
        rig.driver.configure(replace(FAST, pixel_format=RAW8))
        assert self.format_sets(rig) == [AsiImageType.RAW16, AsiImageType.RAW8]

    def test_an_unchanged_flag_needs_no_second_format_set(self) -> None:
        rig = make_rig().opened()
        rig.driver.configure(FAST)
        rig.driver.configure(replace(FAST, exposure_us=1000))
        assert self.format_sets(rig) == [AsiImageType.RAW8, ASI_RAW16, ASI_RAW16]

    def test_a_changed_flag_sets_the_other_format_again(self) -> None:
        rig = make_rig().opened()
        rig.driver.configure(FAST)
        rig.driver.configure(replace(FAST, high_speed=True))
        rig.driver.configure(replace(FAST, high_speed=False))
        assert self.format_sets(rig) == [AsiImageType.RAW8, ASI_RAW16] * 3

    def test_a_reopen_forgets_the_flag_so_the_recovery_step_sets_the_format_again(self) -> None:
        rig = make_rig().opened()
        rig.driver.configure(FAST)
        rig.driver.start()
        before = len(rig.sdk.calls_named("set_roi_format"))
        rig.driver.recover(RecoveryLevel.REOPEN)
        assert self.format_sets(rig)[before:] == [AsiImageType.RAW8, ASI_RAW16]

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


class TestHighSpeedLatch:
    """The effect of the latch. The camera takes up the high-speed flag only when the image format
    changes, and the first time after `init`, and the fake SDK does the same. What the driver
    reports (the ADC depth and the frame period) must be what the camera runs. `TestGeometry`
    checks the calls that the driver makes for it."""

    # The sequence of the bench: n128 / h128 / h8_128 / h128 / n128 / n8_128 / n128, with the flag
    # and the format of each step. The second and the fifth step change the flag alone.
    BENCH = (
        (False, RAW16),
        (True, RAW16),
        (True, RAW8),
        (True, RAW16),
        (False, RAW16),
        (False, RAW8),
        (False, RAW16),
    )

    @staticmethod
    def request(high_speed: bool, pixel_format: PixelFormat = RAW16) -> StreamConfig:
        return replace(FAST, high_speed=high_speed, pixel_format=pixel_format)

    @staticmethod
    def depth_of(data: npt.NDArray[Any]) -> int:
        """The ADC depth of a RAW16 frame: the values sit in the high bits, and the low bits of a
        frame that holds odd counts show where they start."""
        ored = int(np.bitwise_or.reduce(data, axis=None))
        return 16 - ((ored & -ored).bit_length() - 1)

    def test_a_change_of_the_flag_alone_gives_the_requested_regime(self) -> None:
        rig = make_rig().opened()
        for high_speed in (False, True, False, True, True, False):
            rig.driver.configure(self.request(high_speed))
            assert rig.sdk.high_speed_regime is high_speed

    @pytest.mark.parametrize("high_speed", [False, True])
    def test_the_first_configure_gives_the_requested_regime_over_a_stale_flag(
        self, high_speed: bool
    ) -> None:
        state = FakeCameraState({AsiControl.HIGH_SPEED_MODE: int(not high_speed)})
        rig = make_rig(sdk={"state": state}).opened()  # another program left the other flag
        rig.driver.configure(self.request(high_speed))
        assert rig.sdk.high_speed_regime is high_speed

    @pytest.mark.parametrize("level", [RecoveryLevel.REOPEN, RecoveryLevel.USB_RESET])
    def test_a_recovery_that_reopens_the_camera_restores_the_regime(
        self, level: RecoveryLevel
    ) -> None:
        rig = make_rig().opened()
        rig.driver.configure(self.request(True))
        rig.driver.start()
        rig.driver.recover(level)  # a new open starts in the normal regime
        assert rig.sdk.high_speed_regime is True
        assert rig.driver.read_frame(1.0).adc_bits == 10

    def test_the_driver_forgets_the_regime_when_it_restores_the_settings(self) -> None:
        """The restore writes the saved flag and the saved format back, and the format change makes
        the camera take that flag up. The next stream asks for the other flag in the format that the
        camera has now (RAW8, as it starts), so only a driver that forgot the regime sets the
        other format first."""
        rig = make_rig().opened()
        saved = rig.driver.save_settings()
        rig.driver.configure(self.request(True))
        rig.driver.restore_settings(saved)
        assert rig.sdk.high_speed_regime is False  # the camera took up the flag that went back
        rig.driver.configure(self.request(True, RAW8))
        assert rig.sdk.high_speed_regime is True

    def test_what_the_driver_reports_is_what_the_camera_runs(self) -> None:
        rig = make_rig(discard_frames=0).opened()
        for step, (high_speed, pixel_format) in enumerate(self.BENCH, start=1):
            active = rig.driver.configure(self.request(high_speed, pixel_format))
            rig.driver.start()
            first, second = rig.driver.read_frame(1.0), rig.driver.read_frame(1.0)
            rig.driver.stop()
            camera_bits = DEFAULT_ADC_BITS[(1, rig.sdk.high_speed_regime)]
            context = f"step {step}, flag {high_speed}, {pixel_format.name}"
            assert active.adc_bits == first.adc_bits == camera_bits, context
            period_s = active.frame_period_s
            assert period_s == pytest.approx(rig.sdk.frame_period_s()), context
            assert period_s is not None
            assert second.t_arrival_ns - first.t_arrival_ns == round(period_s * NS_PER_S), context
            if pixel_format is RAW16:  # the data itself holds the depth that the camera runs
                assert self.depth_of(first.data) == first.adc_bits, context

    def test_the_bench_sequence_runs_every_step_in_its_regime(self) -> None:
        """The camera gave 82.1, 102.9, 102.9, 102.9, 82.1, 82.1, and 82.1 fps with the fix. A
        driver that sets the flag alone would give 82.1, 82.1, 102.9, 102.9, 102.9, 82.1, 82.1."""
        rig = make_rig().opened()
        periods_s = []
        for high_speed, pixel_format in self.BENCH:
            rig.driver.configure(self.request(high_speed, pixel_format))
            periods_s.append(rig.sdk.frame_period_s())
        normal, fast = periods_s[0], periods_s[2]
        assert fast < 0.9 * normal  # the regimes differ by more than the noise of a measurement
        assert periods_s == [normal, fast, fast, fast, normal, normal, normal]


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
        survey = reference_profile().mode("bin2")
        assert active.frame_period_s == pytest.approx(2.0 + derived.readout_time_s(survey, 64))

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

    def test_no_options_at_all_use_the_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clock = VirtualClock()
        monkeypatch.setattr("seeingmon.drivers.asi.load_asi_api", lambda path: FakeAsiSdk(clock))
        driver = create(profile=reference_profile(), clock=clock, options=None)
        assert driver.name == "asi"

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
