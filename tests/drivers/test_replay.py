"""The `replay` driver, on small synthetic recordings and a virtual clock."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from seeingmon.clock import NS_PER_S, SystemClock, VirtualClock
from seeingmon.drivers import (
    CameraConfigError,
    CameraDisconnectedError,
    CameraDriver,
    CameraError,
    CameraStateError,
    CameraTimeoutError,
    RecoveryLevel,
)
from seeingmon.drivers.replay import (
    ReplayDriver,
    ReplayFinishedError,
    ReplayOptions,
    create,
)
from seeingmon.frames import (
    ActiveStream,
    Frame,
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
from seeingmon.recordings.ser import ColorId, SerWriter
from seeingmon.recordings.sidecar import (
    BurstSidecar,
    burst_sidecar_path,
    sharpcap_sidecar_path,
    write_burst_sidecar,
)
from tests.recordings.synthetic import (
    PERIOD_NS,
    SHARPCAP_SIDECAR,
    START_UTC_NS,
    Recording,
    make_ser,
    regular_timestamps,
)

EXPOSURE_US = 5_000
GAIN = 120
WIDTH, HEIGHT = 32, 16
FULL = StreamConfig(mode="bin2", exposure_us=2_000, gain=1, pixel_format=PixelFormat.RAW16)


def make_driver(
    path: Path, clock: VirtualClock | SystemClock | None = None, **options: object
) -> ReplayDriver:
    settings: dict[str, object] = {
        "path": path,
        "exposure_us": EXPOSURE_US,
        "gain": GAIN,
        **options,
    }
    return create(profile=None, clock=clock or VirtualClock(START_UTC_NS), options=settings)


def started(
    tmp_path: Path, config: StreamConfig = FULL, *, count: int = 16, **options: object
) -> tuple[ReplayDriver, VirtualClock, Recording, ActiveStream]:
    """A recording, and a driver that is open, configured, and started on a virtual clock."""
    recording = make_ser(tmp_path / "a.ser", count=count, width=WIDTH, height=HEIGHT)
    clock = VirtualClock(START_UTC_NS)
    driver = make_driver(recording.path, clock, **options)
    driver.open()
    active = driver.configure(config)
    driver.start()
    return driver, clock, recording, active


def drain(driver: ReplayDriver) -> list[Frame]:
    """Read frames until the recording ends."""
    frames: list[Frame] = []
    try:
        while True:
            frames.append(driver.read_frame(1.0))
    except ReplayFinishedError:
        return frames


class TestProtocol:
    def test_it_satisfies_the_camera_driver_protocol(self, tmp_path: Path) -> None:
        driver = make_driver(make_ser(tmp_path / "a.ser").path)
        typed: CameraDriver = driver  # a static check for mypy
        assert isinstance(typed, CameraDriver)
        assert driver.name == "replay"

    def test_the_lifecycle_errors(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", width=WIDTH, height=HEIGHT)
        driver = make_driver(recording.path)
        with pytest.raises(CameraStateError):
            driver.configure(FULL)
        with pytest.raises(CameraStateError):
            driver.capabilities()
        driver.open()
        with pytest.raises(CameraStateError):
            driver.start()
        with pytest.raises(CameraStateError):
            driver.move_roi(0, 0)
        driver.configure(FULL)
        with pytest.raises(CameraStateError):
            driver.read_frame(1.0)  # configured but not started
        driver.start()
        driver.read_frame(1.0)
        driver.stop()
        driver.stop()
        with pytest.raises(CameraStateError):
            driver.read_frame(1.0)
        driver.close()
        driver.close()
        with pytest.raises(CameraStateError):
            driver.configure(FULL)

    def test_open_describes_the_recording(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", width=WIDTH, height=HEIGHT)
        driver = make_driver(recording.path)
        info = driver.open()
        assert (info.driver, info.sdk_version, info.is_color, info.has_temperature) == (
            "replay",
            None,
            False,
            False,
        )
        assert (info.max_width, info.max_height) == (2 * WIDTH, 2 * HEIGHT)  # bin2 doubles it
        caps = driver.capabilities()
        assert caps.bins == (2,)
        assert caps.pixel_formats == (PixelFormat.RAW8,)
        assert driver.read_temperature_c() is None
        assert driver.dropped_frames() == 0

    def test_recover_stops_the_capture(self, tmp_path: Path) -> None:
        driver, _, _, _ = started(tmp_path)
        driver.recover(RecoveryLevel.RESTART_CAPTURE)
        with pytest.raises(CameraStateError):
            driver.read_frame(1.0)
        driver.start()
        assert driver.read_frame(1.0).seq == 0


class TestFrames:
    def test_original_rate_reproduces_the_recorded_timestamps(self, tmp_path: Path) -> None:
        driver, clock, recording, active = started(tmp_path)
        assert recording.timestamps_ns is not None
        for i, expected in enumerate(recording.frames):
            frame = driver.read_frame(timeout_s=1.0)
            assert frame.t_arrival_ns == recording.timestamps_ns[i]
            assert clock.utc_ns() == frame.t_arrival_ns
            assert frame.seq == i
            assert frame.stream_id == active.stream_id
            assert np.array_equal(frame.data, expected)
        assert clock.utc_ns() - START_UTC_NS == 15 * PERIOD_NS

    def test_frames_carry_the_recorded_settings(self, tmp_path: Path) -> None:
        driver, _, _, active = started(tmp_path)
        frame = driver.read_frame(1.0)
        assert (frame.exposure_us, frame.gain, frame.mode) == (EXPOSURE_US, GAIN, "bin2")
        assert frame.pixel_format is PixelFormat.RAW8
        assert frame.data.dtype == np.uint8
        assert frame.adc_bits == 8
        assert frame.roi == Roi(0, 0, WIDTH, HEIGHT)
        assert frame.flags == FrameFlag.REPLAYED
        assert frame.t_quality is TimeQuality.ESTIMATED
        assert frame.dropped_before == 0
        assert frame.temperature_c is None
        assert not frame.data.flags.writeable
        # The request asked for 2 ms, gain 1, and 16 bits. The stream shows what the frames carry.
        assert active.config.exposure_us == EXPOSURE_US
        assert active.config.gain == GAIN
        assert active.config.pixel_format is PixelFormat.RAW8
        assert active.frame_shape == (HEIGHT, WIDTH)
        assert active.adc_bits == 8
        assert active.frame_period_s == pytest.approx(0.01)

    def test_utc_is_the_middle_of_the_first_row_exposure(self, tmp_path: Path) -> None:
        driver, _, recording, _ = started(tmp_path)
        frame = driver.read_frame(1.0)
        assert recording.timestamps_ns is not None
        # arrival - (frame period - exposure / 2) = arrival - (10 ms - 2.5 ms)
        assert frame.t_utc_ns == recording.timestamps_ns[0] - 7_500_000
        assert frame.t_err_ns > 0

    def test_an_exposure_longer_than_the_period_cannot_push_utc_past_arrival(
        self, tmp_path: Path
    ) -> None:
        driver, _, recording, _ = started(tmp_path, exposure_us=40_000)
        frame = driver.read_frame(1.0)
        assert recording.timestamps_ns is not None
        assert frame.t_utc_ns == recording.timestamps_ns[0] - 20_000_000  # the exposure sets it

    def test_frames_survive_the_wire_codec(self, tmp_path: Path) -> None:
        driver, _, _, _ = started(tmp_path)
        frame = driver.read_frame(1.0)
        assert frames_equal(decode_frame(encode_frame(frame)), frame)

    def test_16_bit_recordings(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", depth=16, width=WIDTH, height=HEIGHT)
        driver = make_driver(recording.path)
        driver.open()
        active = driver.configure(FULL)
        driver.start()
        frame = driver.read_frame(1.0)
        assert frame.pixel_format is PixelFormat.RAW16
        assert frame.adc_bits == active.adc_bits == 16
        assert np.array_equal(frame.data, recording.frames[0])
        assert driver.capabilities().pixel_formats == (PixelFormat.RAW16,)

    def test_the_adc_depth_and_byte_order_can_be_set(self, tmp_path: Path) -> None:
        recording = make_ser(
            tmp_path / "a.ser", depth=16, byte_order="big", width=WIDTH, height=HEIGHT
        )
        driver = make_driver(recording.path, adc_bits=14, byte_order="big")
        driver.open()
        assert driver.configure(FULL).adc_bits == 14
        driver.start()
        assert np.array_equal(driver.read_frame(1.0).data, recording.frames[0])

    def test_a_raw_mosaic_is_flagged_as_color(self, tmp_path: Path) -> None:
        path = tmp_path / "a.ser"
        with SerWriter(path, width=8, height=4, color=ColorId.BAYER_RGGB, timestamps=False) as w:
            w.write_frame(np.zeros((4, 8), dtype=np.uint8))
        assert make_driver(path).open().is_color


class TestRate:
    def test_a_speed_factor_shortens_the_waits(self, tmp_path: Path) -> None:
        driver, clock, _, _ = started(tmp_path, rate=2.0)
        for _ in range(11):
            driver.read_frame(1.0)
        assert clock.utc_ns() - START_UTC_NS == 5 * PERIOD_NS  # 10 intervals at twice the speed

    def test_a_slow_factor_stretches_them(self, tmp_path: Path) -> None:
        driver, clock, _, _ = started(tmp_path, rate="0.5x")
        for _ in range(3):
            driver.read_frame(1.0)
        assert clock.utc_ns() - START_UTC_NS == 4 * PERIOD_NS

    def test_max_never_waits(self, tmp_path: Path) -> None:
        driver, clock, recording, _ = started(tmp_path, rate="max")
        frames = [driver.read_frame(0.0) for _ in range(16)]
        assert clock.utc_ns() == START_UTC_NS
        assert recording.timestamps_ns is not None
        assert [
            f.t_arrival_ns for f in frames
        ] == recording.timestamps_ns  # still the recorded times

    def test_original_is_the_default_and_a_name(self, tmp_path: Path) -> None:
        driver, clock, _, _ = started(tmp_path, rate="original")
        driver.read_frame(1.0)
        driver.read_frame(1.0)
        assert clock.utc_ns() - START_UTC_NS == PERIOD_NS

    def test_a_consumer_that_falls_behind_gets_frames_at_once_until_it_catches_up(
        self, tmp_path: Path
    ) -> None:
        driver, clock, _, _ = started(tmp_path)
        driver.read_frame(1.0)
        clock.advance_ns(5 * PERIOD_NS)  # the consumer was busy for five periods
        before = clock.utc_ns()
        for _ in range(5):
            driver.read_frame(1.0)
        assert clock.utc_ns() == before  # frames 1 to 5 were already due
        driver.read_frame(1.0)
        assert clock.utc_ns() - before == PERIOD_NS  # then the pace resumes

    def test_a_short_timeout_waits_the_timeout_and_keeps_the_schedule(self, tmp_path: Path) -> None:
        driver, clock, _, _ = started(tmp_path)
        driver.read_frame(1.0)
        started_at = clock.monotonic_ns()
        with pytest.raises(CameraTimeoutError):
            driver.read_frame(timeout_s=0.004)  # the next frame is due in 10 ms
        assert clock.monotonic_ns() - started_at == 4_000_000
        frame = driver.read_frame(timeout_s=0.1)
        assert frame.seq == 1  # the failed read took no frame
        assert clock.monotonic_ns() - started_at == PERIOD_NS

    def test_the_first_frame_is_due_when_the_capture_starts(self, tmp_path: Path) -> None:
        driver, clock, _, _ = started(tmp_path)
        clock.advance(3600)  # the consumer is slow to ask for the first frame
        before = clock.utc_ns()
        driver.read_frame(0.0)
        assert clock.utc_ns() == before

    def test_the_real_clock_paces_by_sleeping(self, tmp_path: Path) -> None:
        stamps = regular_timestamps(6, period_ns=2_000_000)
        recording = make_ser(
            tmp_path / "a.ser", count=6, width=WIDTH, height=HEIGHT, timestamps=stamps
        )
        driver = make_driver(recording.path, SystemClock())
        driver.open()
        driver.configure(FULL)
        driver.start()
        started_at = time.monotonic()
        for _ in range(6):
            driver.read_frame(1.0)
        assert 0.009 <= time.monotonic() - started_at < 2.0  # five intervals of 2 ms

    @pytest.mark.parametrize(
        "rate", [0, -1, 0.0, "fast", True, [1], float("inf"), float("nan"), ""]
    )
    def test_a_bad_rate_is_refused(self, tmp_path: Path, rate: object) -> None:
        with pytest.raises(CameraConfigError, match="rate"):
            make_driver(make_ser(tmp_path / "a.ser").path, rate=rate)


class TestGaps:
    def gappy(self, tmp_path: Path) -> ReplayDriver:
        """1000 frames at 10 ms with one frame lost before 250, two before 500, and a gap of
        exactly 1.5 periods before 750."""
        shifts = {250: PERIOD_NS, 500: 2 * PERIOD_NS, 750: PERIOD_NS // 2}
        stamps: list[int] = []
        shift = 0
        for i in range(1000):
            shift += shifts.get(i, 0)
            stamps.append(START_UTC_NS + i * PERIOD_NS + shift)
        recording = make_ser(
            tmp_path / "a.ser", count=1000, width=WIDTH, height=HEIGHT, timestamps=stamps
        )
        driver = make_driver(recording.path, rate="max")
        driver.open()
        driver.configure(FULL)
        driver.start()
        return driver

    def test_a_gap_over_one_and_a_half_periods_becomes_dropped_before(self, tmp_path: Path) -> None:
        driver = self.gappy(tmp_path)
        dropped = {frame.seq: frame.dropped_before for frame in drain(driver)}
        assert {seq: count for seq, count in dropped.items() if count} == {250: 1, 500: 2}
        assert dropped[750] == 0  # exactly 1.5 periods is not over the limit

    def test_the_period_is_not_thrown_off_by_a_few_gaps(self, tmp_path: Path) -> None:
        driver = self.gappy(tmp_path)
        active = driver.configure(FULL)
        assert active.frame_period_s == pytest.approx(0.01, rel=1e-9)

    def test_the_counter_adds_up_and_restarts_with_the_stream(self, tmp_path: Path) -> None:
        driver = self.gappy(tmp_path)
        for _ in range(300):
            driver.read_frame(0)
        assert driver.dropped_frames() == 1
        for _ in range(300):
            driver.read_frame(0)
        assert driver.dropped_frames() == 3
        driver.configure(FULL)
        assert driver.dropped_frames() == 0

    def test_a_gap_just_over_the_threshold_counts_one_frame(self, tmp_path: Path) -> None:
        stamps = regular_timestamps(400)
        stamps = stamps[:200] + [s + PERIOD_NS // 2 + 1_000 for s in stamps[200:]]
        recording = make_ser(
            tmp_path / "a.ser", count=400, width=WIDTH, height=HEIGHT, timestamps=stamps
        )
        driver = make_driver(recording.path, rate="max")
        driver.open()
        driver.configure(FULL)
        driver.start()
        counts = [frame.dropped_before for frame in drain(driver)]
        assert counts[200] == 1  # a gap of 1.5 periods and 1 microsecond rounds to 2, minus 1
        assert sum(counts) == 1

    def test_the_first_frame_of_a_stream_never_reports_a_gap(self, tmp_path: Path) -> None:
        driver = self.gappy(tmp_path)
        for _ in range(249):
            driver.read_frame(0)
        driver.configure(FULL)  # a new stream; the recording continues at frame 249
        driver.start()
        assert driver.read_frame(0).dropped_before == 0  # frame 249 follows a normal interval
        assert driver.read_frame(0).dropped_before == 1  # frame 250 follows the gap

    def test_a_start_frame_after_a_gap_ignores_the_gap(self, tmp_path: Path) -> None:
        self.gappy(tmp_path).close()
        late = make_driver(tmp_path / "a.ser", rate="max", start_frame=250)
        late.open()
        late.configure(FULL)
        late.start()
        assert late.read_frame(0).dropped_before == 0  # the gap precedes the first frame read
        assert late.read_frame(0).dropped_before == 0


class TestRoi:
    def test_a_smaller_roi_is_cropped_in_software(self, tmp_path: Path) -> None:
        crop = StreamConfig(mode="bin2", exposure_us=1, gain=0, roi=Roi(8, 4, 16, 8))
        driver, _, recording, active = started(tmp_path, crop)
        assert active.config.roi == Roi(8, 4, 16, 8)
        assert active.frame_shape == (8, 16)
        frame = driver.read_frame(1.0)
        assert frame.roi == Roi(8, 4, 16, 8)
        assert np.array_equal(frame.data, recording.frames[0][4:12, 8:24])
        assert frame.data.flags["C_CONTIGUOUS"]
        assert frames_equal(decode_frame(encode_frame(frame)), frame)

    def test_the_full_frame_is_the_default(self, tmp_path: Path) -> None:
        _, _, _, active = started(tmp_path)
        assert active.config.roi == Roi(0, 0, WIDTH, HEIGHT)

    def test_the_roi_follows_the_vendor_rules_and_stays_inside_the_frame(
        self, tmp_path: Path
    ) -> None:
        odd = StreamConfig(mode="bin2", exposure_us=1, gain=0, roi=Roi(30, 15, 13, 7))
        _, _, _, active = started(tmp_path, odd)
        assert active.config.roi == Roi(24, 10, 8, 6)  # 13 x 7 became 8 x 6, then moved inside

    def test_move_roi_moves_the_window_and_keeps_the_stream(self, tmp_path: Path) -> None:
        crop = StreamConfig(mode="bin2", exposure_us=1, gain=0, roi=Roi(0, 0, 16, 8))
        driver, _, recording, active = started(tmp_path, crop)
        driver.read_frame(1.0)
        moved = driver.move_roi(10, 6)
        assert moved == Roi(10, 6, 16, 8)
        frame = driver.read_frame(1.0)
        assert frame.roi == moved
        assert frame.stream_id == active.stream_id
        assert np.array_equal(frame.data, recording.frames[1][6:14, 10:26])
        assert driver.move_roi(-50, 10_000) == Roi(0, HEIGHT - 8, 16, 8)
        assert driver.move_roi(10_000, -50) == Roi(WIDTH - 16, 0, 16, 8)

    def test_a_request_that_does_not_fit_is_refused(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", width=WIDTH, height=HEIGHT)
        driver = make_driver(recording.path)
        driver.open()
        for roi in (Roi(0, 0, WIDTH + 8, 8), Roi(0, 0, 8, HEIGHT + 2), Roi(0, 0, 1000, 1000)):
            with pytest.raises(CameraConfigError, match="larger than the recording"):
                driver.configure(StreamConfig(mode="bin2", exposure_us=1, gain=0, roi=roi))

    def test_another_readout_mode_is_refused(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", width=WIDTH, height=HEIGHT)
        driver = make_driver(recording.path)
        driver.open()
        with pytest.raises(CameraConfigError, match="another mode"):
            driver.configure(StreamConfig(mode="bin1", exposure_us=1, gain=0))

    def test_a_refused_request_stops_the_capture(self, tmp_path: Path) -> None:
        driver, _, _, _ = started(tmp_path)
        with pytest.raises(CameraConfigError):
            driver.configure(StreamConfig(mode="bin1", exposure_us=1, gain=0))
        with pytest.raises(CameraStateError):
            driver.read_frame(1.0)


class TestStreams:
    def test_each_configure_starts_a_new_stream_and_continues_the_recording(
        self, tmp_path: Path
    ) -> None:
        driver, _, recording, first = started(tmp_path)
        driver.read_frame(1.0)
        driver.read_frame(1.0)
        second = driver.configure(FULL)
        assert second.stream_id == first.stream_id + 1
        with pytest.raises(CameraStateError):  # configure stops the capture
            driver.read_frame(1.0)
        driver.start()
        frame = driver.read_frame(1.0)
        assert frame.seq == 0
        assert frame.stream_id == second.stream_id
        assert np.array_equal(frame.data, recording.frames[2])  # the recording moved on

    def test_open_rewinds(self, tmp_path: Path) -> None:
        driver, _, recording, _ = started(tmp_path)
        for _ in range(5):
            driver.read_frame(1.0)
        driver.open()
        driver.configure(FULL)
        driver.start()
        assert np.array_equal(driver.read_frame(1.0).data, recording.frames[0])

    def test_stop_and_start_resume_where_they_paused(self, tmp_path: Path) -> None:
        driver, clock, recording, _ = started(tmp_path)
        driver.read_frame(1.0)
        driver.read_frame(1.0)
        driver.stop()
        clock.advance(10)
        driver.start()
        frame = driver.read_frame(1.0)
        assert np.array_equal(frame.data, recording.frames[2])
        assert frame.seq == 2

    def test_a_snapshot_stream_returns_one_frame_per_start(self, tmp_path: Path) -> None:
        config = StreamConfig(mode="bin2", exposure_us=1, gain=0, kind=StreamKind.SNAPSHOT)
        driver, _, recording, _ = started(tmp_path, config)
        assert np.array_equal(driver.read_frame(1.0).data, recording.frames[0])
        with pytest.raises(CameraStateError):
            driver.read_frame(1.0)
        driver.start()
        assert driver.read_frame(1.0).seq == 1


class TestEndOfRecording:
    def test_it_raises_a_dedicated_error_after_the_last_frame(self, tmp_path: Path) -> None:
        driver, _, _, _ = started(tmp_path, count=3)
        for _ in range(3):
            driver.read_frame(1.0)
        for _ in range(2):  # and it keeps raising
            with pytest.raises(ReplayFinishedError, match="no more frames"):
                driver.read_frame(1.0)
        assert issubclass(ReplayFinishedError, CameraError)

    def test_a_window_of_the_recording(self, tmp_path: Path) -> None:
        driver, _, recording, _ = started(tmp_path, start_frame=4, max_frames=5)
        frames = drain(driver)
        assert len(frames) == 5
        assert [f.seq for f in frames] == [0, 1, 2, 3, 4]
        for offset, frame in enumerate(frames):
            assert np.array_equal(frame.data, recording.frames[4 + offset])

    def test_max_frames_past_the_end_stops_at_the_end(self, tmp_path: Path) -> None:
        driver, _, _, _ = started(tmp_path, count=6, start_frame=4, max_frames=100)
        assert [driver.read_frame(1.0).seq for _ in range(2)] == [0, 1]
        with pytest.raises(ReplayFinishedError):
            driver.read_frame(1.0)

    def test_loop_wraps_with_a_continuous_timeline(self, tmp_path: Path) -> None:
        driver, clock, recording, _ = started(tmp_path, count=4, loop=True)
        frames = [driver.read_frame(1.0) for _ in range(10)]
        assert [f.seq for f in frames] == list(range(10))
        for i, frame in enumerate(frames):
            assert np.array_equal(frame.data, recording.frames[i % 4])
            assert frame.t_arrival_ns == START_UTC_NS + i * PERIOD_NS
            assert frame.dropped_before == 0
        assert clock.utc_ns() == START_UTC_NS + 9 * PERIOD_NS

    def test_loop_wraps_to_the_start_frame(self, tmp_path: Path) -> None:
        driver, _, recording, _ = started(tmp_path, start_frame=2, max_frames=3, loop=True)
        frames = [driver.read_frame(1.0) for _ in range(7)]
        expected = [2, 3, 4, 2, 3, 4, 2]
        for frame, index in zip(frames, expected, strict=True):
            assert np.array_equal(frame.data, recording.frames[index])
        stamps = [f.t_arrival_ns for f in frames]
        assert stamps == sorted(stamps)
        assert len(set(stamps)) == 7

    def test_a_looped_recording_with_a_gap_at_the_wrap_reports_no_drop(
        self, tmp_path: Path
    ) -> None:
        stamps = [START_UTC_NS + i * PERIOD_NS for i in (0, 1, 2, 5)]
        recording = make_ser(
            tmp_path / "a.ser", count=4, width=WIDTH, height=HEIGHT, timestamps=stamps
        )
        driver = make_driver(recording.path, rate="max", loop=True)
        driver.open()
        driver.configure(FULL)
        driver.start()
        counts = [driver.read_frame(0).dropped_before for _ in range(8)]
        assert counts[3] == 2  # the recorded gap
        assert counts[4] == 0  # the wrap is not a gap


class TestSynthesizedTimestamps:
    def test_a_file_without_a_trailer_gets_timestamps_from_its_header(self, tmp_path: Path) -> None:
        recording = make_ser(
            tmp_path / "a.ser",
            width=WIDTH,
            height=HEIGHT,
            timestamps=False,
            header_start_utc_ns=START_UTC_NS + 123_000,
        )
        clock = VirtualClock(START_UTC_NS)
        driver = make_driver(recording.path, clock)
        driver.open()
        active = driver.configure(FULL)
        driver.start()
        frames = [driver.read_frame(1.0) for _ in range(4)]
        period_ns = EXPOSURE_US * 1000  # with nothing else to go on, the stream is exposure-limited
        assert [f.t_arrival_ns for f in frames] == [
            START_UTC_NS + 123_000 + i * period_ns for i in range(4)
        ]
        assert active.frame_period_s == pytest.approx(EXPOSURE_US / 1e6)

    def test_without_any_start_time_the_clock_gives_one(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", width=WIDTH, height=HEIGHT, timestamps=False)
        clock = VirtualClock(START_UTC_NS + 5 * NS_PER_S)
        driver = make_driver(recording.path, clock)
        driver.open()
        driver.configure(FULL)
        driver.start()
        assert driver.read_frame(1.0).t_arrival_ns == START_UTC_NS + 5 * NS_PER_S

    def test_a_sharpcap_sidecar_gives_the_start_time_and_the_frame_rate(
        self, tmp_path: Path
    ) -> None:
        recording = make_ser(
            tmp_path / "a.ser",
            width=WIDTH,
            height=HEIGHT,
            timestamps=False,
            header_start_utc_ns=START_UTC_NS
            + 42_000_000,  # the header start differs from StartCapture
        )
        sharpcap_sidecar_path(recording.path).write_text(SHARPCAP_SIDECAR, encoding="utf-8")
        driver = make_driver(recording.path, rate="max")
        driver.open()
        active = driver.configure(FULL)
        driver.start()
        first, second = driver.read_frame(0), driver.read_frame(0)
        assert first.t_arrival_ns == 1_767_268_800_123_456_700  # StartCapture, not the header
        assert second.t_arrival_ns - first.t_arrival_ns == 10_000_000  # ActualFrameRate=100
        assert active.frame_period_s == pytest.approx(0.01)

    def test_a_burst_sidecar_gives_the_start_time_and_the_period(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", width=WIDTH, height=HEIGHT, timestamps=False)
        write_burst_sidecar(
            burst_sidecar_path(recording.path),
            BurstSidecar(
                profile_id="p1",
                stream=StreamConfig(mode="bin2", exposure_us=1000, gain=3),
                adc_bits=14,
                time_quality=TimeQuality.FITTED,
                frame_period_s=0.004,
                start_utc_ns=START_UTC_NS + 7,
            ),
        )
        driver = make_driver(recording.path, rate="max", exposure_us=None, gain=None)
        driver.open()
        driver.configure(FULL)
        driver.start()
        first, second = driver.read_frame(0), driver.read_frame(0)
        assert first.t_arrival_ns == START_UTC_NS + 7
        assert second.t_arrival_ns - first.t_arrival_ns == 4_000_000

    def test_all_zero_timestamps_count_as_missing(self, tmp_path: Path) -> None:
        recording = make_ser(
            tmp_path / "a.ser",
            count=3,
            width=WIDTH,
            height=HEIGHT,
            header_start_utc_ns=START_UTC_NS + 500,
        )
        with recording.path.open("r+b") as handle:
            handle.seek(178 + 3 * WIDTH * HEIGHT)
            handle.write(bytes(24))
        driver = make_driver(recording.path, rate="max")
        driver.open()
        driver.configure(FULL)
        driver.start()
        frames = [driver.read_frame(0) for _ in range(3)]
        period_ns = EXPOSURE_US * 1000
        assert [f.t_arrival_ns for f in frames] == [
            START_UTC_NS + 500 + i * period_ns for i in range(3)
        ]

    def test_a_recording_with_one_frame_still_gets_a_period(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", count=1, width=WIDTH, height=HEIGHT)
        driver = make_driver(recording.path, rate="max")
        driver.open()
        active = driver.configure(FULL)
        driver.start()
        assert active.frame_period_s == pytest.approx(EXPOSURE_US / 1e6)
        driver.read_frame(0)
        with pytest.raises(ReplayFinishedError):
            driver.read_frame(0)


class TestSidecars:
    def test_a_sharpcap_sidecar_supplies_the_settings(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", width=WIDTH, height=HEIGHT)
        sharpcap_sidecar_path(recording.path).write_text(SHARPCAP_SIDECAR, encoding="utf-8")
        driver = create(
            profile=None, clock=VirtualClock(START_UTC_NS), options={"path": recording.path}
        )
        info = driver.open()
        assert info.has_temperature
        assert driver.read_temperature_c() == 20.5
        active = driver.configure(StreamConfig(mode="bin2", exposure_us=1, gain=0))
        driver.start()
        frame = driver.read_frame(1.0)
        assert (frame.exposure_us, frame.gain, frame.mode) == (10_000, 100, "bin2")
        assert frame.temperature_c == 20.5
        assert active.config.exposure_us == 10_000

    def test_the_mode_comes_from_the_read_mode_and_options_override_it(
        self, tmp_path: Path
    ) -> None:
        recording = make_ser(tmp_path / "a.ser", width=WIDTH, height=HEIGHT)
        native = SHARPCAP_SIDECAR.replace("Read Mode=11 Megapixel", "Read Mode=47 Megapixel")
        sharpcap_sidecar_path(recording.path).write_text(native, encoding="utf-8")
        driver = make_driver(recording.path)
        driver.open()
        assert driver.capabilities().bins == (1,)
        driver.configure(StreamConfig(mode="bin1", exposure_us=1, gain=0))
        forced = make_driver(recording.path, mode="bin2")
        forced.open()
        assert forced.capabilities().bins == (2,)

    def test_options_override_the_sidecar(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", width=WIDTH, height=HEIGHT)
        sharpcap_sidecar_path(recording.path).write_text(SHARPCAP_SIDECAR, encoding="utf-8")
        driver = make_driver(recording.path)  # exposure_us=5000 and gain=120
        driver.open()
        driver.configure(FULL)
        driver.start()
        frame = driver.read_frame(1.0)
        assert (frame.exposure_us, frame.gain) == (EXPOSURE_US, GAIN)

    def test_a_gain_of_zero_is_a_value_not_a_gap(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", width=WIDTH, height=HEIGHT)
        sharpcap_sidecar_path(recording.path).write_text(SHARPCAP_SIDECAR, encoding="utf-8")
        driver = make_driver(recording.path, gain=0, exposure_us=None)
        driver.open()
        driver.configure(FULL)
        driver.start()
        frame = driver.read_frame(1.0)
        assert (frame.exposure_us, frame.gain) == (10_000, 0)

    def test_a_burst_sidecar_supplies_the_settings_and_the_position(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", width=WIDTH, height=HEIGHT)
        write_burst_sidecar(
            burst_sidecar_path(recording.path),
            BurstSidecar(
                profile_id="p1",
                stream=StreamConfig(
                    mode="bin1",
                    exposure_us=2_000,
                    gain=77,
                    pixel_format=PixelFormat.RAW8,
                    roi=Roi(104, 52, WIDTH, HEIGHT),
                ),
                adc_bits=12,
                time_quality=TimeQuality.FITTED,
                temperature_c=9.5,
            ),
        )
        driver = create(
            profile=None, clock=VirtualClock(START_UTC_NS), options={"path": str(recording.path)}
        )
        info = driver.open()
        assert (info.max_width, info.max_height) == (104 + WIDTH, 52 + HEIGHT)  # bin1
        active = driver.configure(StreamConfig(mode="bin1", exposure_us=9, gain=9))
        assert active.config.roi == Roi(104, 52, WIDTH, HEIGHT)  # sensor coordinates
        assert active.adc_bits == 12
        driver.start()
        frame = driver.read_frame(1.0)
        assert (frame.exposure_us, frame.gain, frame.temperature_c) == (2_000, 77, 9.5)
        crop = driver.configure(
            StreamConfig(mode="bin1", exposure_us=9, gain=9, roi=Roi(112, 56, 16, 8))
        )
        assert crop.config.roi == Roi(112, 56, 16, 8)
        driver.start()
        assert np.array_equal(driver.read_frame(1.0).data, recording.frames[1][4:12, 8:24])
        assert driver.move_roi(0, 0) == Roi(104, 52, 16, 8)

    def test_an_explicit_sidecar_path(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", width=WIDTH, height=HEIGHT)
        elsewhere = tmp_path / "notes.txt"
        elsewhere.write_text(SHARPCAP_SIDECAR, encoding="utf-8")
        driver = create(
            profile=None,
            clock=VirtualClock(START_UTC_NS),
            options={"path": recording.path, "sidecar": elsewhere},
        )
        driver.open()
        driver.configure(FULL)
        driver.start()
        assert driver.read_frame(1.0).exposure_us == 10_000

    def test_sidecar_false_reads_none(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", width=WIDTH, height=HEIGHT)
        sharpcap_sidecar_path(recording.path).write_text(SHARPCAP_SIDECAR, encoding="utf-8")
        driver = create(
            profile=None,
            clock=VirtualClock(START_UTC_NS),
            options={"path": recording.path, "sidecar": False},
        )
        with pytest.raises(CameraConfigError, match="exposure_us and gain"):
            driver.open()

    def test_a_burst_sidecar_wins_over_a_sharpcap_one(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", width=WIDTH, height=HEIGHT)
        sharpcap_sidecar_path(recording.path).write_text(SHARPCAP_SIDECAR, encoding="utf-8")
        write_burst_sidecar(
            burst_sidecar_path(recording.path),
            BurstSidecar(
                "p1", StreamConfig(mode="bin2", exposure_us=3_000, gain=5), 8, TimeQuality.EXACT
            ),
        )
        driver = create(
            profile=None, clock=VirtualClock(START_UTC_NS), options={"path": recording.path}
        )
        driver.open()
        driver.configure(FULL)
        driver.start()
        assert driver.read_frame(1.0).exposure_us == 3_000

    @pytest.mark.parametrize(
        "options",
        [{}, {"exposure_us": 100}, {"gain": 1}],
        ids=["neither", "gain missing", "exposure missing"],
    )
    def test_settings_that_nothing_supplies_are_an_error(
        self, tmp_path: Path, options: Mapping[str, object]
    ) -> None:
        recording = make_ser(tmp_path / "a.ser", width=WIDTH, height=HEIGHT)
        driver = create(
            profile=None,
            clock=VirtualClock(START_UTC_NS),
            options={"path": recording.path, **options},
        )
        with pytest.raises(CameraConfigError, match="exposure_us and gain"):
            driver.open()

    def test_a_broken_sidecar_is_a_config_error_without_the_path(self, tmp_path: Path) -> None:
        folder = tmp_path / "distinctive-folder-name"
        folder.mkdir()
        recording = make_ser(folder / "a.ser", width=WIDTH, height=HEIGHT)
        burst_sidecar_path(recording.path).write_text("{ not json", encoding="utf-8")
        driver = make_driver(recording.path)
        with pytest.raises(CameraConfigError, match="sidecar cannot be used") as raised:
            driver.open()
        assert "distinctive-folder-name" not in str(raised.value)


class TestOpening:
    def test_a_missing_file(self, tmp_path: Path) -> None:
        folder = tmp_path / "distinctive-folder-name"
        folder.mkdir()
        driver = make_driver(folder / "missing.ser")
        with pytest.raises(CameraDisconnectedError, match="cannot open the recording") as raised:
            driver.open()
        assert "distinctive-folder-name" not in str(raised.value)

    def test_a_file_that_is_not_a_ser_recording(self, tmp_path: Path) -> None:
        folder = tmp_path / "distinctive-folder-name"
        folder.mkdir()
        path = folder / "bad.ser"
        path.write_bytes(b"x" * 500)
        with pytest.raises(CameraConfigError, match="not a valid SER file") as raised:
            make_driver(path).open()
        assert "distinctive-folder-name" not in str(raised.value)

    def test_a_truncated_recording(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", width=WIDTH, height=HEIGHT)
        recording.path.write_bytes(recording.path.read_bytes()[:-1000])
        with pytest.raises(CameraConfigError, match="truncated"):
            make_driver(recording.path).open()

    def test_a_color_recording(self, tmp_path: Path) -> None:
        path = tmp_path / "a.ser"
        with SerWriter(path, width=8, height=4, color=ColorId.RGB, timestamps=False) as writer:
            writer.write_frame(np.zeros((4, 8, 3), dtype=np.uint8))
        with pytest.raises(CameraConfigError, match="mono and raw mosaic"):
            make_driver(path).open()

    def test_a_recording_without_frames(self, tmp_path: Path) -> None:
        path = tmp_path / "a.ser"
        with SerWriter(path, width=8, height=4):
            pass
        with pytest.raises(CameraConfigError, match="no frames"):
            make_driver(path).open()

    def test_a_start_frame_past_the_end(self, tmp_path: Path) -> None:
        recording = make_ser(tmp_path / "a.ser", count=4, width=WIDTH, height=HEIGHT)
        with pytest.raises(CameraConfigError, match="start_frame"):
            make_driver(recording.path, start_frame=4).open()

    def test_a_failed_open_leaves_the_driver_closed(self, tmp_path: Path) -> None:
        driver = make_driver(tmp_path / "missing.ser")
        with pytest.raises(CameraDisconnectedError):
            driver.open()
        with pytest.raises(CameraStateError):
            driver.configure(FULL)


class TestOptions:
    def test_the_defaults(self, tmp_path: Path) -> None:
        options = ReplayOptions.from_mapping({"path": tmp_path / "a.ser"})
        assert options == ReplayOptions(path=tmp_path / "a.ser")
        assert (options.rate, options.loop, options.start_frame, options.max_frames) == (
            1.0,
            False,
            0,
            None,
        )

    def test_every_option(self, tmp_path: Path) -> None:
        options = ReplayOptions.from_mapping(
            {
                "path": str(tmp_path / "a.ser"),
                "rate": "max",
                "loop": True,
                "start_frame": 3,
                "max_frames": 9,
                "mode": "bin1",
                "exposure_us": 2000,
                "gain": 0,
                "sidecar": False,
                "byte_order": "big",
                "adc_bits": 12,
            }
        )
        assert options == ReplayOptions(
            path=tmp_path / "a.ser",
            rate=None,
            loop=True,
            start_frame=3,
            max_frames=9,
            mode="bin1",
            exposure_us=2000,
            gain=0,
            sidecar=False,
            byte_order="big",
            adc_bits=12,
        )

    @pytest.mark.parametrize(
        ("options", "message"),
        [
            ({}, "path"),
            ({"path": ""}, "path"),
            ({"path": 5}, "path"),
            ({"path": "a.ser", "bogus": 1}, "unknown replay options: bogus"),
            ({"path": "a.ser", "loop": "yes"}, "loop"),
            ({"path": "a.ser", "start_frame": -1}, "start_frame"),
            ({"path": "a.ser", "start_frame": 1.5}, "start_frame"),
            ({"path": "a.ser", "start_frame": True}, "start_frame"),
            ({"path": "a.ser", "max_frames": 0}, "max_frames"),
            ({"path": "a.ser", "mode": ""}, "mode"),
            ({"path": "a.ser", "mode": "a" * 17}, "mode"),
            ({"path": "a.ser", "mode": "bin 2"}, "mode"),
            ({"path": "a.ser", "mode": 2}, "mode"),
            ({"path": "a.ser", "exposure_us": 0}, "exposure_us"),
            ({"path": "a.ser", "gain": -1}, "gain"),
            ({"path": "a.ser", "adc_bits": 0}, "adc_bits"),
            ({"path": "a.ser", "adc_bits": 17}, "adc_bits"),
            ({"path": "a.ser", "byte_order": "middle"}, "byte_order"),
            ({"path": "a.ser", "sidecar": 3}, "sidecar"),
            ({"path": "a.ser", "sidecar": ""}, "sidecar"),
        ],
    )
    def test_bad_options_are_refused(self, options: Mapping[str, object], message: str) -> None:
        with pytest.raises(CameraConfigError, match=message):
            create(profile=None, clock=VirtualClock(), options=options)

    def test_the_sidecar_true_means_look_next_to_the_file(self, tmp_path: Path) -> None:
        options = ReplayOptions.from_mapping({"path": tmp_path / "a.ser", "sidecar": True})
        assert options.sidecar is None
