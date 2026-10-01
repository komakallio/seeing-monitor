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
from seeingmon.hardware.asi.fake import FakeAsiSdk, FakeFrameInfo, default_pixels, pixel_bytes

RAW8, RAW16 = AsiImageType.RAW8, AsiImageType.RAW16


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
        assert sdk.frame_period_s() == pytest.approx(5.0e-3 + 8 * 30.1e-6)
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
