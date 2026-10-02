"""The fake SDK: the behavior that the research notes describe, and the faults it scripts."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from seeingmon.clock import NS_PER_S, VirtualClock
from seeingmon.hardware.asi.api import (
    AsiApi,
    AsiConfigError,
    AsiControl,
    AsiDisconnectedError,
    AsiError,
    AsiErrorCode,
    AsiExposureStatus,
    AsiImageType,
    AsiStateError,
    AsiTimeoutError,
)
from seeingmon.hardware.asi.fake import (
    DEFAULT_TIMING,
    FakeAsiSdk,
    FakeCameraState,
    FakeFrameInfo,
    FakeTiming,
    default_pixels,
    pixel_bytes,
)

RAW8, RAW16 = AsiImageType.RAW8, AsiImageType.RAW16

# What the bench measured on the ASI294MM in bin1 at bandwidth 100 (SDK 1.41): the frame period is
# 7.37 ms plus 37.6 us a row in the normal regime, and 5.88 ms plus 30.0 us a row in the high-speed
# regime.
BENCH_TIMING = {
    **DEFAULT_TIMING,
    (1, False): FakeTiming(37.6e-6, 7.37e-3),
    (1, True): FakeTiming(30.0e-6, 5.88e-3),
}


def bench_period_s(high_speed: bool, rows: int = 128) -> float:
    timing = BENCH_TIMING[(1, high_speed)]
    return timing.overhead_s + rows * timing.row_time_s


@pytest.fixture
def clock() -> VirtualClock:
    return VirtualClock()


@pytest.fixture
def sdk(clock: VirtualClock) -> FakeAsiSdk:
    return FakeAsiSdk(clock)


@pytest.fixture
def camera(sdk: FakeAsiSdk, clock: VirtualClock) -> int:
    """The ID of a camera that is open and initialized, past the temperature warm-up."""
    camera_id = sdk.get_camera_property(0).camera_id
    sdk.open_camera(camera_id)
    sdk.init_camera(camera_id)
    clock.advance(1.0)
    return camera_id


def frame_bytes(width: int, height: int, image_type: AsiImageType = RAW16) -> bytearray:
    return bytearray(width * height * (2 if image_type is RAW16 else 1))


def stream(
    sdk: FakeAsiSdk,
    camera: int,
    *,
    width: int = 16,
    height: int = 8,
    binning: int = 1,
    exposure_us: int = 2000,
    image_type: AsiImageType = RAW16,
) -> None:
    sdk.set_control_value(camera, AsiControl.EXPOSURE, exposure_us)
    sdk.set_roi_format(camera, width, height, binning, image_type)
    sdk.set_start_position(camera, 8, 4)
    sdk.start_video_capture(camera)


def test_the_fake_satisfies_the_protocol(sdk: FakeAsiSdk) -> None:
    api: AsiApi = sdk
    assert api.get_sdk_version()


class TestEnumeration:
    def test_one_camera_with_its_properties(self, sdk: FakeAsiSdk) -> None:
        assert sdk.get_connected_camera_count() == 1
        info = sdk.get_camera_property(0)
        assert (info.max_width, info.max_height) == (8288, 5644)
        assert info.supported_bins == (1, 2)
        assert AsiImageType.RAW16 in info.supported_formats
        assert not info.is_color

    def test_an_index_past_the_last_camera_is_invalid(self, sdk: FakeAsiSdk) -> None:
        with pytest.raises(AsiConfigError) as raised:
            sdk.get_camera_property(1)
        assert raised.value.code == AsiErrorCode.INVALID_INDEX

    def test_a_disconnected_camera_leaves_the_list_and_fails_calls(
        self, sdk: FakeAsiSdk, camera: int
    ) -> None:
        sdk.disconnect()
        assert sdk.get_connected_camera_count() == 0
        with pytest.raises(AsiDisconnectedError):
            sdk.get_control_value(camera, AsiControl.GAIN)
        with pytest.raises(AsiConfigError):
            sdk.get_camera_property(0)

    def test_a_reconnected_camera_returns_after_the_delay_with_power_on_state(
        self, sdk: FakeAsiSdk, camera: int, clock: VirtualClock
    ) -> None:
        sdk.set_control_value(camera, AsiControl.GAIN, 200)
        sdk.disconnect()
        sdk.reconnect(after_s=2.0)
        assert sdk.get_connected_camera_count() == 0
        clock.advance(2.0)
        assert sdk.get_connected_camera_count() == 1
        with pytest.raises(AsiStateError):  # closed until you open it again
            sdk.get_control_value(camera, AsiControl.GAIN)
        sdk.open_camera(camera)
        sdk.init_camera(camera)
        assert sdk.get_control_value(camera, AsiControl.GAIN) == (0, False)


class TestLifecycle:
    def test_calls_need_an_open_and_initialized_camera(self, sdk: FakeAsiSdk) -> None:
        with pytest.raises(AsiStateError):
            sdk.get_control_count(0)
        sdk.open_camera(0)
        with pytest.raises(AsiStateError):
            sdk.get_control_count(0)
        sdk.init_camera(0)
        assert sdk.get_control_count(0) > 0

    def test_a_wrong_camera_id_is_invalid(self, sdk: FakeAsiSdk) -> None:
        with pytest.raises(AsiDisconnectedError):
            sdk.open_camera(7)

    def test_control_caps_list_gain_exposure_and_temperature(
        self, sdk: FakeAsiSdk, camera: int
    ) -> None:
        count = sdk.get_control_count(camera)
        caps = {sdk.get_control_caps(camera, i).control: sdk.get_control_caps(camera, i)
                for i in range(count)}  # fmt: skip
        assert caps[AsiControl.GAIN].max_value == 570
        assert caps[AsiControl.EXPOSURE].min_value == 32
        assert not caps[AsiControl.TEMPERATURE].is_writable
        with pytest.raises(AsiConfigError):
            sdk.get_control_caps(camera, count)

    def test_a_camera_without_a_sensor_lists_no_temperature_control(
        self, clock: VirtualClock
    ) -> None:
        plain = FakeAsiSdk(clock, temperature_c=None)
        plain.open_camera(0)
        plain.init_camera(0)
        controls = {plain.get_control_caps(0, i).control for i in range(plain.get_control_count(0))}
        assert AsiControl.TEMPERATURE not in controls
        with pytest.raises(AsiConfigError):
            plain.get_control_value(0, AsiControl.TEMPERATURE)


class TestControls:
    def test_the_sdk_clamps_a_value_to_the_range(self, sdk: FakeAsiSdk, camera: int) -> None:
        sdk.set_control_value(camera, AsiControl.GAIN, 10_000)
        assert sdk.get_control_value(camera, AsiControl.GAIN) == (570, False)

    def test_a_read_only_control_rejects_a_write(self, sdk: FakeAsiSdk, camera: int) -> None:
        with pytest.raises(AsiConfigError):
            sdk.set_control_value(camera, AsiControl.TEMPERATURE, 100)

    def test_the_first_temperature_read_is_zero_for_250_ms(
        self, sdk: FakeAsiSdk, clock: VirtualClock
    ) -> None:
        sdk.open_camera(0)
        sdk.init_camera(0)
        assert sdk.get_control_value(0, AsiControl.TEMPERATURE)[0] == 0
        clock.advance(0.249)
        assert sdk.get_control_value(0, AsiControl.TEMPERATURE)[0] == 0
        clock.advance(0.002)
        assert sdk.get_control_value(0, AsiControl.TEMPERATURE) == (183, False)

    def test_a_scripted_silent_gain_limit_clamps_without_an_error(
        self, clock: VirtualClock
    ) -> None:
        limited = FakeAsiSdk(clock, silent_gain_limit=300)
        limited.open_camera(0)
        limited.init_camera(0)
        limited.set_control_value(0, AsiControl.GAIN, 450)
        assert limited.get_control_value(0, AsiControl.GAIN)[0] == 300


def open_camera(sdk: FakeAsiSdk) -> int:
    """Open and initialize the camera, as a new process does."""
    camera_id = sdk.get_camera_property(0).camera_id
    sdk.open_camera(camera_id)
    sdk.init_camera(camera_id)
    return camera_id


class TestPersistentControls:
    """The camera keeps its controls until it loses power, as the real one does. A process that
    does not set a control finds the value that the last process or program left."""

    def test_controls_survive_a_close_and_an_open(self, sdk: FakeAsiSdk, camera: int) -> None:
        sdk.set_control_value(camera, AsiControl.BANDWIDTH_OVERLOAD, 80)
        sdk.set_control_value(camera, AsiControl.FLIP, 2)
        sdk.close_camera(camera)
        camera = open_camera(sdk)
        assert sdk.get_control_value(camera, AsiControl.BANDWIDTH_OVERLOAD) == (80, False)
        assert sdk.get_control_value(camera, AsiControl.FLIP) == (2, False)

    def test_a_new_instance_that_shares_the_state_finds_what_the_last_one_left(
        self, clock: VirtualClock
    ) -> None:
        state = FakeCameraState()
        first = FakeAsiSdk(clock, state=state)
        camera = open_camera(first)
        first.set_control_value(camera, AsiControl.BANDWIDTH_OVERLOAD, 90)
        first.set_control_value(camera, AsiControl.OFFSET, 33)
        first.close_camera(camera)
        second = FakeAsiSdk(clock, state=state)  # a new process opens the same camera
        camera = open_camera(second)
        assert second.get_control_value(camera, AsiControl.BANDWIDTH_OVERLOAD) == (90, False)
        assert second.get_control_value(camera, AsiControl.OFFSET) == (33, False)
        assert second.state is first.state

    def test_an_instance_without_a_shared_state_starts_from_the_defaults(
        self, clock: VirtualClock
    ) -> None:
        first = FakeAsiSdk(clock)
        camera = open_camera(first)
        first.set_control_value(camera, AsiControl.BANDWIDTH_OVERLOAD, 90)
        other = FakeAsiSdk(clock)
        camera = open_camera(other)
        assert other.get_control_value(camera, AsiControl.BANDWIDTH_OVERLOAD) == (50, False)

    def test_a_stale_state_keeps_the_values_that_it_lists_and_defaults_the_rest(
        self, clock: VirtualClock
    ) -> None:
        stale = FakeCameraState(
            {AsiControl.BANDWIDTH_OVERLOAD: 50, AsiControl.FLIP: 3, AsiControl.OFFSET: 20}
        )
        sdk = FakeAsiSdk(clock, state=stale)
        camera = open_camera(sdk)
        assert sdk.get_control_value(camera, AsiControl.FLIP) == (3, False)
        assert sdk.get_control_value(camera, AsiControl.OFFSET) == (20, False)
        assert sdk.get_control_value(camera, AsiControl.GAIN) == (0, False)  # not listed: default
        assert sdk.control(AsiControl.FLIP) == 3

    def test_a_power_cycle_resets_the_shared_state(self, clock: VirtualClock) -> None:
        state = FakeCameraState({AsiControl.FLIP: 3})
        sdk = FakeAsiSdk(clock, state=state)
        camera = open_camera(sdk)
        sdk.set_control_value(camera, AsiControl.EXPOSURE, 2000, auto=True)
        sdk.disconnect()
        sdk.reconnect()
        camera = open_camera(sdk)
        assert sdk.get_control_value(camera, AsiControl.FLIP) == (0, False)
        assert sdk.get_control_value(camera, AsiControl.EXPOSURE) == (10_000, False)
        assert state.automatic == set()

    def test_the_automatic_flag_persists_until_a_manual_write_clears_it(
        self, sdk: FakeAsiSdk, camera: int
    ) -> None:
        sdk.set_control_value(camera, AsiControl.EXPOSURE, 5000, auto=True)
        assert sdk.get_control_value(camera, AsiControl.EXPOSURE) == (5000, True)
        sdk.close_camera(camera)
        camera = open_camera(sdk)
        assert sdk.get_control_value(camera, AsiControl.EXPOSURE)[1] is True
        sdk.set_control_value(camera, AsiControl.EXPOSURE, 5000)
        assert sdk.get_control_value(camera, AsiControl.EXPOSURE) == (5000, False)

    def test_a_control_without_automatic_support_ignores_the_flag(
        self, sdk: FakeAsiSdk, camera: int
    ) -> None:
        sdk.set_control_value(camera, AsiControl.FLIP, 1, auto=True)
        assert sdk.get_control_value(camera, AsiControl.FLIP) == (1, False)

    def test_a_camera_can_keep_a_control_automatic_whatever_a_write_asks(
        self, sdk: FakeAsiSdk, camera: int
    ) -> None:
        sdk.keep_auto(AsiControl.GAIN)
        sdk.set_control_value(camera, AsiControl.GAIN, 100, auto=False)
        assert sdk.get_control_value(camera, AsiControl.GAIN) == (100, True)


class TestHighSpeedLatch:
    """The camera takes the high-speed flag into its regime only at the first ROI format after it
    opens and when the image type changes. The ASI294MM does this with SDK 1.41, and it reports no
    error for a change of the flag alone."""

    @staticmethod
    def apply(
        sdk: FakeAsiSdk,
        camera: int,
        *,
        high_speed: bool,
        image_type: AsiImageType = RAW16,
        size: int = 128,
    ) -> None:
        """What a mode change does: set the flag, and then the ROI format."""
        sdk.set_control_value(camera, AsiControl.HIGH_SPEED_MODE, int(high_speed))
        sdk.set_roi_format(camera, size, size, 1, image_type)

    @pytest.fixture
    def bench(self, clock: VirtualClock) -> tuple[FakeAsiSdk, int]:
        sdk = FakeAsiSdk(clock, timing=BENCH_TIMING)
        camera = open_camera(sdk)
        sdk.set_control_value(camera, AsiControl.EXPOSURE, 2000)  # the readout sets the period
        return sdk, camera

    def test_a_new_open_starts_in_the_normal_regime_whatever_the_control_holds(
        self, clock: VirtualClock
    ) -> None:
        sdk = FakeAsiSdk(clock, state=FakeCameraState({AsiControl.HIGH_SPEED_MODE: 1}))
        open_camera(sdk)
        assert sdk.control(AsiControl.HIGH_SPEED_MODE) == 1
        assert sdk.high_speed_regime is False

    def test_the_first_roi_format_takes_the_flag_even_without_a_change_of_type(
        self, bench: tuple[FakeAsiSdk, int]
    ) -> None:
        sdk, camera = bench
        self.apply(sdk, camera, high_speed=True, image_type=RAW8)  # RAW8 is the type at the start
        assert sdk.high_speed_regime is True
        assert sdk.frame_period_s() == pytest.approx(bench_period_s(True))

    def test_a_change_of_the_flag_alone_leaves_the_regime(
        self, bench: tuple[FakeAsiSdk, int]
    ) -> None:
        sdk, camera = bench
        self.apply(sdk, camera, high_speed=False)  # the first call takes the flag
        sdk.set_control_value(camera, AsiControl.HIGH_SPEED_MODE, 1)
        sdk.set_roi_format(camera, 128, 128, 1, RAW16)  # the same type and size
        sdk.set_roi_format(camera, 64, 64, 1, RAW16)  # a smaller size
        sdk.set_roi_format(camera, 64, 64, 2, RAW16)  # another binning
        assert sdk.high_speed_regime is False
        assert sdk.control(AsiControl.HIGH_SPEED_MODE) == 1  # the control holds the request
        assert sdk.get_control_value(camera, AsiControl.HIGH_SPEED_MODE) == (1, False)

    def test_a_change_of_the_image_type_takes_the_flag_in_both_directions(
        self, bench: tuple[FakeAsiSdk, int]
    ) -> None:
        sdk, camera = bench
        self.apply(sdk, camera, high_speed=False)
        self.apply(sdk, camera, high_speed=True, image_type=RAW8)
        assert sdk.high_speed_regime is True
        self.apply(sdk, camera, high_speed=False)  # RAW8 to RAW16 takes the flag again
        assert sdk.high_speed_regime is False

    def test_the_period_and_the_adc_depth_follow_the_regime_and_not_the_control(
        self, bench: tuple[FakeAsiSdk, int]
    ) -> None:
        sdk, camera = bench
        self.apply(sdk, camera, high_speed=False)
        sdk.set_control_value(camera, AsiControl.HIGH_SPEED_MODE, 1)  # no latch
        assert sdk.frame_period_s() == pytest.approx(bench_period_s(False))
        sdk.start_video_capture(camera)
        buffer = frame_bytes(128, 128)
        sdk.get_video_data(camera, buffer, wait_ms=500)
        x, y, _, _ = sdk.roi
        counts = default_pixels(FakeFrameInfo(0, 128, 128, 1, RAW16, x, y, 2000, 0, 12))
        words = np.frombuffer(buffer, dtype="<u2").reshape(128, 128)
        np.testing.assert_array_equal(
            words, counts << 4
        )  # 12 bits, not the 10 that the control asks

    def test_the_sequence_that_the_bench_ran(self, bench: tuple[FakeAsiSdk, int]) -> None:
        """n128 / h128 / h8_128 / h128 / n128 / n8_128 / n128 gave 82.1, 82.1, 102.9, 102.9, 102.9,
        82.1, and 82.1 frames a second (n is the normal flag, h is high-speed, and 8 is RAW8)."""
        sdk, camera = bench
        steps = [
            (False, RAW16),
            (True, RAW16),
            (True, RAW8),
            (True, RAW16),
            (False, RAW16),
            (False, RAW8),
            (False, RAW16),
        ]
        rates = []
        for high_speed, image_type in steps:
            self.apply(sdk, camera, high_speed=high_speed, image_type=image_type)
            rates.append(round(1 / sdk.frame_period_s(), 1))
        assert rates == [82.1, 82.1, 102.9, 102.9, 102.9, 82.1, 82.1]

    def test_a_new_open_forgets_the_regime(self, bench: tuple[FakeAsiSdk, int]) -> None:
        sdk, camera = bench
        self.apply(sdk, camera, high_speed=True)
        regimes = [sdk.high_speed_regime]
        sdk.close_camera(camera)
        open_camera(sdk)
        regimes.append(sdk.high_speed_regime)
        assert regimes == [True, False]
        assert sdk.control(AsiControl.HIGH_SPEED_MODE) == 1  # the camera keeps the control

    def test_a_power_cycle_resets_the_regime_and_the_control(
        self, bench: tuple[FakeAsiSdk, int]
    ) -> None:
        sdk, camera = bench
        self.apply(sdk, camera, high_speed=True)
        sdk.disconnect()
        sdk.reconnect()
        open_camera(sdk)
        assert (sdk.high_speed_regime, sdk.control(AsiControl.HIGH_SPEED_MODE)) == (False, 0)


class TestRoiRules:
    @pytest.mark.parametrize(
        ("width", "height", "binning"),
        [(12, 8, 1), (16, 7, 1), (0, 8, 1), (16, 0, 1), (8296, 8, 1), (16, 5646, 1), (4152, 8, 2)],
    )
    def test_a_size_that_breaks_the_rules_is_invalid(
        self, sdk: FakeAsiSdk, camera: int, width: int, height: int, binning: int
    ) -> None:
        with pytest.raises(AsiConfigError) as raised:
            sdk.set_roi_format(camera, width, height, binning, RAW16)
        assert raised.value.code == AsiErrorCode.INVALID_SIZE

    def test_an_unsupported_binning_or_format_is_invalid(
        self, sdk: FakeAsiSdk, camera: int
    ) -> None:
        with pytest.raises(AsiConfigError):
            sdk.set_roi_format(camera, 16, 8, 3, RAW16)
        with pytest.raises(AsiConfigError) as raised:
            sdk.set_roi_format(camera, 16, 8, 1, AsiImageType.RGB24)
        assert raised.value.code == AsiErrorCode.INVALID_IMAGE_TYPE

    def test_set_roi_format_recenters_the_roi(self, sdk: FakeAsiSdk, camera: int) -> None:
        sdk.set_roi_format(camera, 128, 128, 1, RAW16)
        assert sdk.get_start_position(camera) == ((8288 - 128) // 2, (5644 - 128) // 2)
        sdk.set_start_position(camera, 100, 200)
        assert sdk.get_start_position(camera) == (100, 200)
        sdk.set_roi_format(camera, 128, 128, 1, RAW16)
        assert sdk.get_start_position(camera) == ((8288 - 128) // 2, (5644 - 128) // 2)

    def test_the_start_position_stays_inside_the_frame(self, sdk: FakeAsiSdk, camera: int) -> None:
        sdk.set_roi_format(camera, 128, 128, 1, RAW16)
        sdk.set_start_position(camera, 8288 - 128, 5644 - 128)
        for x, y in [(8288 - 127, 0), (0, 5644 - 127), (-1, 0)]:
            with pytest.raises(AsiConfigError) as raised:
                sdk.set_start_position(camera, x, y)
            assert raised.value.code == AsiErrorCode.OUT_OF_BOUNDARY

    def test_the_binned_frame_is_smaller(self, sdk: FakeAsiSdk, camera: int) -> None:
        sdk.set_roi_format(camera, 4144, 2822, 2, RAW16)
        assert sdk.get_roi_format(camera).binning == 2
        assert sdk.get_start_position(camera) == (0, 0)

    def test_the_start_position_aligns_when_the_camera_requires_it(
        self, clock: VirtualClock
    ) -> None:
        coarse = FakeAsiSdk(clock, start_alignment=4)
        coarse.open_camera(0)
        coarse.init_camera(0)
        coarse.set_roi_format(0, 16, 8, 1, RAW16)
        coarse.set_start_position(0, 103, 202)
        assert coarse.get_start_position(0) == (100, 200)

    def test_a_scripted_roi_format_is_applied_once(self, sdk: FakeAsiSdk, camera: int) -> None:
        sdk.corrupt_next_roi(24, 8)
        sdk.set_roi_format(camera, 16, 8, 1, RAW16)
        assert sdk.get_roi_format(camera).width == 24
        sdk.set_roi_format(camera, 16, 8, 1, RAW16)
        assert sdk.get_roi_format(camera).width == 16

    def test_a_persistent_scripted_roi_format_applies_every_time(
        self, sdk: FakeAsiSdk, camera: int
    ) -> None:
        sdk.corrupt_next_roi(24, 8, always=True)
        for _ in range(3):
            sdk.set_roi_format(camera, 16, 8, 1, RAW16)
            assert sdk.get_roi_format(camera).width == 24

    def test_set_roi_format_during_capture_succeeds_without_effect(
        self, sdk: FakeAsiSdk, camera: int
    ) -> None:
        stream(sdk, camera, width=16, height=8)
        sdk.set_roi_format(camera, 64, 32, 1, RAW16)  # no error
        fmt = sdk.get_roi_format(camera)
        assert (fmt.width, fmt.height) == (16, 8)


class TestVideo:
    def test_frames_complete_one_period_apart(
        self, sdk: FakeAsiSdk, camera: int, clock: VirtualClock
    ) -> None:
        stream(sdk, camera, width=128, height=128)
        period_ns = round(sdk.frame_period_s() * NS_PER_S)
        assert period_ns == 6_500_000 + 128 * 37_600  # readout is longer than the 2 ms exposure
        buffer = frame_bytes(128, 128)
        started = clock.monotonic_ns()
        for count in range(1, 5):
            sdk.get_video_data(camera, buffer, wait_ms=500)
            assert clock.monotonic_ns() - started == count * period_ns

    def test_a_slow_exposure_sets_the_period(
        self, sdk: FakeAsiSdk, camera: int, clock: VirtualClock
    ) -> None:
        stream(sdk, camera, exposure_us=50_000)
        buffer = frame_bytes(16, 8)
        started = clock.monotonic_ns()
        sdk.get_video_data(camera, buffer, wait_ms=500)
        assert clock.monotonic_ns() - started == 50_000_000

    def test_pixels_follow_the_sensor_position_and_hold_the_adc_value_in_the_high_bits(
        self, sdk: FakeAsiSdk, camera: int
    ) -> None:
        stream(sdk, camera, width=16, height=8)  # the ROI starts at (8, 4) in bin1, 12 bit
        buffer = frame_bytes(16, 8)
        sdk.get_video_data(camera, buffer, wait_ms=500)
        info = FakeFrameInfo(0, 16, 8, 1, RAW16, 8, 4, 2000, 0, 12)
        counts = default_pixels(info)
        words = np.frombuffer(buffer, dtype="<u2").reshape(8, 16)
        np.testing.assert_array_equal(words, counts << 4)
        assert not np.any(words & 0xF)  # the low 4 bits of a 12-bit value are zero

    def test_raw8_frames_carry_the_top_eight_bits(self, sdk: FakeAsiSdk, camera: int) -> None:
        stream(sdk, camera, width=16, height=8, image_type=RAW8)
        buffer = frame_bytes(16, 8, RAW8)
        sdk.get_video_data(camera, buffer, wait_ms=500)
        counts = default_pixels(FakeFrameInfo(0, 16, 8, 1, RAW8, 8, 4, 2000, 0, 12))
        np.testing.assert_array_equal(
            np.frombuffer(buffer, dtype=np.uint8).reshape(8, 16), counts >> 4
        )

    def test_the_adc_depth_follows_the_binning(self, sdk: FakeAsiSdk, camera: int) -> None:
        stream(sdk, camera, width=16, height=8, binning=2)
        buffer = frame_bytes(16, 8)
        sdk.get_video_data(camera, buffer, wait_ms=500)
        counts = default_pixels(FakeFrameInfo(0, 16, 8, 2, RAW16, 8, 4, 2000, 0, 14))
        np.testing.assert_array_equal(
            np.frombuffer(buffer, dtype="<u2").reshape(8, 16), counts << 2
        )

    def test_high_speed_mode_changes_the_adc_depth_and_the_timing(
        self, sdk: FakeAsiSdk, camera: int
    ) -> None:
        sdk.set_control_value(camera, AsiControl.HIGH_SPEED_MODE, 1)
        stream(sdk, camera, width=16, height=8)
        timing = DEFAULT_TIMING[(1, True)]
        assert sdk.frame_period_s() == pytest.approx(timing.overhead_s + 8 * timing.row_time_s)
        buffer = frame_bytes(16, 8)
        sdk.get_video_data(camera, buffer, wait_ms=500)
        counts = default_pixels(FakeFrameInfo(0, 16, 8, 1, RAW16, 8, 4, 2000, 0, 10))
        np.testing.assert_array_equal(
            np.frombuffer(buffer, dtype="<u2").reshape(8, 16), counts << 6
        )

    def test_a_short_wait_expires_after_exactly_the_wait(
        self, sdk: FakeAsiSdk, camera: int, clock: VirtualClock
    ) -> None:
        stream(sdk, camera, exposure_us=500_000)
        started = clock.monotonic_ns()
        with pytest.raises(AsiTimeoutError):
            sdk.get_video_data(camera, frame_bytes(16, 8), wait_ms=100)
        assert clock.monotonic_ns() - started == 100 * 1_000_000
        sdk.get_video_data(camera, frame_bytes(16, 8), wait_ms=500)  # the frame is still coming

    def test_a_buffer_that_is_too_small_fails(self, sdk: FakeAsiSdk, camera: int) -> None:
        stream(sdk, camera, width=16, height=8)
        with pytest.raises(AsiConfigError) as raised:
            sdk.get_video_data(camera, bytearray(10), wait_ms=500)
        assert raised.value.code == AsiErrorCode.BUFFER_TOO_SMALL

    def test_a_read_without_capture_is_out_of_sequence(self, sdk: FakeAsiSdk, camera: int) -> None:
        with pytest.raises(AsiStateError):
            sdk.get_video_data(camera, frame_bytes(16, 8), wait_ms=10)

    def test_a_late_reader_loses_the_oldest_frames_and_the_counter_counts_them(
        self, sdk: FakeAsiSdk, camera: int, clock: VirtualClock
    ) -> None:
        stream(sdk, camera, width=16, height=8)
        period = sdk.frame_period_s()
        clock.advance(10.5 * period)  # frames 0 to 9 complete, and the buffer holds three
        assert sdk.get_dropped_frames(camera) == 7
        buffer = frame_bytes(16, 8)
        indices = []
        for _ in range(3):
            sdk.get_video_data(camera, buffer, wait_ms=500)
            words = np.frombuffer(buffer, dtype="<u2").reshape(8, 16)
            expected = [
                default_pixels(FakeFrameInfo(i, 16, 8, 1, RAW16, 8, 4, 2000, 0, 12)) << 4
                for i in (7, 8, 9)
            ]
            indices.append(next(i for i, e in zip((7, 8, 9), expected, strict=True)
                                if np.array_equal(words, e)))  # fmt: skip
        assert indices == [7, 8, 9]
        assert sdk.get_dropped_frames(camera) == 7

    def test_stopping_capture_resets_the_drop_counter(
        self, sdk: FakeAsiSdk, camera: int, clock: VirtualClock
    ) -> None:
        stream(sdk, camera)
        clock.advance(1.0)
        assert sdk.get_dropped_frames(camera) > 0
        sdk.stop_video_capture(camera)
        assert sdk.get_dropped_frames(camera) == 0
        sdk.start_video_capture(camera)
        assert sdk.get_dropped_frames(camera) == 0

    def test_lose_frames_raises_the_counter_and_delays_the_next_frame(
        self, sdk: FakeAsiSdk, camera: int, clock: VirtualClock
    ) -> None:
        stream(sdk, camera)
        period_ns = round(sdk.frame_period_s() * NS_PER_S)
        sdk.lose_frames(3)
        assert sdk.get_dropped_frames(camera) == 3
        started = clock.monotonic_ns()
        sdk.get_video_data(camera, frame_bytes(16, 8), wait_ms=500)
        assert clock.monotonic_ns() - started == 4 * period_ns

    def test_stalled_reads_time_out_and_then_the_stream_resumes(
        self, sdk: FakeAsiSdk, camera: int, clock: VirtualClock
    ) -> None:
        stream(sdk, camera)
        sdk.stall_reads(2)
        started = clock.monotonic_ns()
        for _ in range(2):
            with pytest.raises(AsiTimeoutError):
                sdk.get_video_data(camera, frame_bytes(16, 8), wait_ms=200)
        assert clock.monotonic_ns() - started == 400 * 1_000_000
        sdk.get_video_data(camera, frame_bytes(16, 8), wait_ms=200)

    def test_a_start_position_change_applies_to_new_frames_and_not_to_the_buffered_ones(
        self, sdk: FakeAsiSdk, camera: int, clock: VirtualClock
    ) -> None:
        stream(sdk, camera, width=16, height=8)  # starts at (8, 4)
        clock.advance(2.5 * sdk.frame_period_s())  # frames 0 and 1 wait in the buffer
        sdk.set_start_position(camera, 40, 20)
        buffer = frame_bytes(16, 8)
        sdk.get_video_data(camera, buffer, wait_ms=500)
        old = default_pixels(FakeFrameInfo(0, 16, 8, 1, RAW16, 8, 4, 2000, 0, 12)) << 4
        np.testing.assert_array_equal(np.frombuffer(buffer, dtype="<u2").reshape(8, 16), old)
        sdk.get_video_data(camera, buffer, wait_ms=500)
        sdk.get_video_data(camera, buffer, wait_ms=500)  # frame 2 completed after the move
        new = default_pixels(FakeFrameInfo(2, 16, 8, 1, RAW16, 40, 20, 2000, 0, 12)) << 4
        np.testing.assert_array_equal(np.frombuffer(buffer, dtype="<u2").reshape(8, 16), new)

    def test_a_silent_geometry_change_shows_in_the_read_back(
        self, sdk: FakeAsiSdk, camera: int
    ) -> None:
        stream(sdk, camera, width=16, height=8)
        sdk.change_geometry(24, 8)
        assert sdk.get_roi_format(camera).width == 24
        with pytest.raises(AsiConfigError):  # the caller's buffer is too small for a frame now
            sdk.get_video_data(camera, frame_bytes(16, 8), wait_ms=500)
        sdk.get_video_data(camera, frame_bytes(24, 8), wait_ms=500)


class TestExposures:
    def test_an_exposure_completes_after_the_exposure_and_the_readout(
        self, sdk: FakeAsiSdk, camera: int, clock: VirtualClock
    ) -> None:
        sdk.set_control_value(camera, AsiControl.EXPOSURE, 1_000_000)
        sdk.set_roi_format(camera, 16, 8, 2, RAW16)
        sdk.start_exposure(camera)
        assert sdk.get_exposure_status(camera) is AsiExposureStatus.WORKING
        clock.advance(1.0)  # the readout of 8 rows takes about 1.6 ms more
        assert sdk.get_exposure_status(camera) is AsiExposureStatus.WORKING
        clock.advance(0.002)
        assert sdk.get_exposure_status(camera) is AsiExposureStatus.SUCCESS
        buffer = frame_bytes(16, 8)
        sdk.get_data_after_exposure(camera, buffer)
        assert sdk.get_exposure_status(camera) is AsiExposureStatus.IDLE
        with pytest.raises(AsiStateError):  # the data came out once
            sdk.get_data_after_exposure(camera, buffer)

    def test_video_and_exposure_exclude_each_other(self, sdk: FakeAsiSdk, camera: int) -> None:
        stream(sdk, camera)
        with pytest.raises(AsiStateError) as raised:
            sdk.start_exposure(camera)
        assert raised.value.code == AsiErrorCode.VIDEO_MODE_ACTIVE
        sdk.stop_video_capture(camera)
        sdk.start_exposure(camera)
        with pytest.raises(AsiStateError) as busy:
            sdk.start_video_capture(camera)
        assert busy.value.code == AsiErrorCode.EXPOSURE_IN_PROGRESS
        with pytest.raises(AsiStateError):
            sdk.set_roi_format(camera, 16, 8, 1, RAW16)

    def test_a_stopped_exposure_goes_idle(self, sdk: FakeAsiSdk, camera: int) -> None:
        sdk.start_exposure(camera)
        sdk.stop_exposure(camera)
        assert sdk.get_exposure_status(camera) is AsiExposureStatus.IDLE

    def test_a_scripted_failure_ends_the_exposure(
        self, sdk: FakeAsiSdk, camera: int, clock: VirtualClock
    ) -> None:
        sdk.fail_next_exposure()
        sdk.start_exposure(camera)
        clock.advance(100.0)
        assert sdk.get_exposure_status(camera) is AsiExposureStatus.FAILED


class TestScriptedFaults:
    def test_fail_next_raises_the_error_for_that_call_only(
        self, sdk: FakeAsiSdk, camera: int
    ) -> None:
        sdk.fail_next("get_roi_format", AsiErrorCode.GENERAL_ERROR)
        with pytest.raises(AsiError) as raised:
            sdk.get_roi_format(camera)
        assert raised.value.code == AsiErrorCode.GENERAL_ERROR
        sdk.get_roi_format(camera)

    def test_a_hung_call_blocks_until_you_release_it(self, sdk: FakeAsiSdk, camera: int) -> None:
        stream(sdk, camera)
        release = sdk.hang_next("get_video_data")
        outcome: list[BaseException | None] = []
        entered = threading.Event()

        def read() -> None:
            entered.set()
            try:
                sdk.get_video_data(camera, frame_bytes(16, 8), wait_ms=500)
                outcome.append(None)
            except AsiError as error:
                outcome.append(error)

        thread = threading.Thread(target=read)
        thread.start()
        assert entered.wait(5.0)
        thread.join(0.2)
        assert thread.is_alive()  # still blocked
        release.set()
        thread.join(5.0)
        assert not thread.is_alive()
        assert isinstance(outcome[0], AsiTimeoutError)

    def test_the_call_log_records_each_call_and_each_read_return(
        self, sdk: FakeAsiSdk, camera: int
    ) -> None:
        stream(sdk, camera)
        sdk.get_video_data(camera, frame_bytes(16, 8), wait_ms=500)
        sdk.stop_video_capture(camera)
        names = [name for name, _ in sdk.calls]
        assert names.index("get_video_data") < names.index("get_video_data_returned")
        assert names.index("get_video_data_returned") < names.index("stop_video_capture")
        assert sdk.calls_named("get_video_data")[0][-1] == 500


def test_pixel_bytes_rejects_formats_it_does_not_render() -> None:
    with pytest.raises(ValueError, match="RAW8 and RAW16"):
        pixel_bytes(np.zeros((2, 2), dtype=np.uint32), AsiImageType.RGB24, 12)
