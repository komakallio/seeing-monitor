"""The fakes in `seeingmon.testing`, and a check that each satisfies its protocol."""

from __future__ import annotations

import numpy as np
import pytest

from seeingmon.clock import NS_PER_S, Clock, VirtualClock
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
from seeingmon.sinks import Sink, SinkError, StoredRow
from seeingmon.solvers import PlateSolver, SolveRequest, SolverError, SolveResult, StarList
from seeingmon.testing import FakeCameraDriver, FakeSink, FakeSolver

FAST = StreamConfig(mode="bin1", exposure_us=2000, gain=120, roi=Roi(100, 200, 128, 128))


def test_fakes_satisfy_the_protocols() -> None:
    clock: Clock = VirtualClock()
    driver: CameraDriver = FakeCameraDriver(clock)
    sink: Sink = FakeSink()
    solver: PlateSolver = FakeSolver()
    assert isinstance(driver, CameraDriver)
    assert isinstance(sink, Sink)
    assert isinstance(solver, PlateSolver)


@pytest.fixture
def clock() -> VirtualClock:
    return VirtualClock()


@pytest.fixture
def driver(clock: VirtualClock) -> FakeCameraDriver:
    fake = FakeCameraDriver(clock)
    fake.open()
    return fake


class TestFakeCameraDriver:
    def test_streams_frames_and_advances_virtual_time(
        self, driver: FakeCameraDriver, clock: VirtualClock
    ) -> None:
        active = driver.configure(FAST)
        driver.start()
        started = clock.utc_ns()
        frames = [driver.read_frame(timeout_s=1.0) for _ in range(5)]
        assert [f.seq for f in frames] == [0, 1, 2, 3, 4]
        assert all(f.stream_id == active.stream_id for f in frames)
        period_ns = round(active.frame_period_s * NS_PER_S)  # type: ignore[operator]
        assert clock.utc_ns() - started == 5 * period_ns
        assert frames[1].t_arrival_ns - frames[0].t_arrival_ns == period_ns
        assert all(
            f.t_quality is TimeQuality.EXACT and f.flags & FrameFlag.SIMULATED for f in frames
        )
        assert frames[0].data.shape == (128, 128)
        assert frames[0].t_utc_ns < frames[0].t_arrival_ns

    def test_frames_survive_the_wire_format(self, driver: FakeCameraDriver) -> None:
        driver.configure(FAST)
        driver.start()
        frame = driver.read_frame(timeout_s=1.0)
        assert frames_equal(decode_frame(encode_frame(frame)), frame)

    def test_roi_follows_the_vendor_rules_and_stays_inside_the_frame(
        self, driver: FakeCameraDriver
    ) -> None:
        odd = StreamConfig(mode="bin1", exposure_us=2000, gain=1, roi=Roi(8200, 5600, 131, 125))
        active = driver.configure(odd)
        roi = active.config.roi
        assert roi is not None
        assert (roi.width, roi.height) == (128, 124)
        assert roi.x_end <= 8288
        assert roi.y_end <= 5644

    def test_the_full_frame_is_the_default_roi(self, driver: FakeCameraDriver) -> None:
        active = driver.configure(StreamConfig(mode="bin2", exposure_us=1000, gain=1))
        assert active.frame_shape == (2822, 4144)

    def test_each_configure_starts_a_new_stream(self, driver: FakeCameraDriver) -> None:
        first = driver.configure(FAST)
        driver.start()
        driver.read_frame(1.0)
        second = driver.configure(FAST)
        assert second.stream_id == first.stream_id + 1
        with pytest.raises(CameraStateError):  # configure stops capture
            driver.read_frame(1.0)
        driver.start()
        assert driver.read_frame(1.0).seq == 0

    def test_move_roi_keeps_the_stream_and_clamps(self, driver: FakeCameraDriver) -> None:
        active = driver.configure(FAST)
        driver.start()
        moved = driver.move_roi(-50, 10_000)
        assert moved == Roi(0, 5644 - 128, 128, 128)
        frame = driver.read_frame(1.0)
        assert frame.roi == moved
        assert frame.stream_id == active.stream_id

    def test_snapshot_returns_one_frame_per_start(self, driver: FakeCameraDriver) -> None:
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

    def test_a_short_timeout_expires_after_waiting_the_timeout(
        self, driver: FakeCameraDriver, clock: VirtualClock
    ) -> None:
        driver.configure(
            StreamConfig(mode="bin1", exposure_us=500_000, gain=1, roi=Roi(0, 0, 8, 2))
        )
        driver.start()
        started = clock.monotonic_ns()
        with pytest.raises(CameraTimeoutError):
            driver.read_frame(timeout_s=0.1)
        assert clock.monotonic_ns() - started == round(0.1 * NS_PER_S)

    def test_scripted_read_failures_come_one_per_read(self, driver: FakeCameraDriver) -> None:
        driver.configure(FAST)
        driver.start()
        driver.fail_reads(CameraTimeoutError("stall"), CameraDisconnectedError("unplugged"))
        with pytest.raises(CameraTimeoutError):
            driver.read_frame(1.0)
        with pytest.raises(CameraDisconnectedError):
            driver.read_frame(1.0)
        assert driver.read_frame(1.0).seq == 0

    def test_dropped_frames_are_reported_once(self, driver: FakeCameraDriver) -> None:
        driver.configure(FAST)
        driver.start()
        driver.drop_frames(3)
        assert driver.dropped_frames() == 3
        assert driver.read_frame(1.0).dropped_before == 3
        assert driver.read_frame(1.0).dropped_before == 0
        assert driver.dropped_frames() == 3

    def test_lifecycle_errors(self, clock: VirtualClock) -> None:
        fake = FakeCameraDriver(clock)
        with pytest.raises(CameraStateError):
            fake.configure(FAST)
        fake.open()
        with pytest.raises(CameraStateError):
            fake.start()
        with pytest.raises(CameraConfigError):
            fake.configure(StreamConfig(mode="bin9", exposure_us=1, gain=0))
        with pytest.raises(CameraConfigError):
            fake.configure(StreamConfig(mode="bin1", exposure_us=1, gain=0, roi=Roi(0, 0, 9000, 8)))

    def test_open_failure_recovery_and_close(self, clock: VirtualClock) -> None:
        fake = FakeCameraDriver(clock)
        fake.fail_open()
        with pytest.raises(CameraDisconnectedError):
            fake.open()
        working = FakeCameraDriver(clock)
        info = working.open()
        assert (info.driver, info.has_temperature) == ("fake", True)
        working.recover(RecoveryLevel.RESTART_CAPTURE)
        working.fail_recover()
        with pytest.raises(CameraError):
            working.recover(RecoveryLevel.USB_RESET)
        working.close()
        working.close()
        assert [call for call, _ in working.calls if call == "recover"] == ["recover", "recover"]

    def test_custom_frame_factory_and_pixel_scaling(self, clock: VirtualClock) -> None:
        def star(config: StreamConfig, roi: Roi, seq: int) -> np.ndarray:
            data = np.zeros((roi.height, roi.width), dtype=config.pixel_format.dtype)
            data[roi.height // 2, roi.width // 2] = 1000 + seq
            return data

        fake = FakeCameraDriver(clock, frame_factory=star)
        fake.open()
        fake.configure(StreamConfig(mode="bin1", exposure_us=1000, gain=0, roi=Roi(0, 0, 16, 16)))
        fake.start()
        assert fake.read_frame(1.0).data[8, 8] == 1000
        assert fake.read_frame(1.0).data[8, 8] == 1001
        plain = FakeCameraDriver(clock, adc_bits=12)
        plain.open()
        plain.configure(
            StreamConfig(
                mode="bin1",
                exposure_us=1000,
                gain=0,
                roi=Roi(0, 0, 8, 2),
                pixel_format=PixelFormat.RAW16,
            )
        )
        plain.start()
        assert plain.read_frame(1.0).data[0, 0] == 100 << 4

    def test_a_simulated_night_of_frames_runs_in_seconds(
        self, driver: FakeCameraDriver, clock: VirtualClock
    ) -> None:
        driver.configure(
            StreamConfig(mode="bin1", exposure_us=1_000_000, gain=1, roi=Roi(0, 0, 8, 2))
        )
        driver.start()
        started = clock.utc_ns()
        count = 0
        while clock.utc_ns() - started < 12 * 3600 * NS_PER_S:
            driver.read_frame(timeout_s=5.0)
            count += 1
        assert 12 * 3600 // 2 < count <= 12 * 3600


class TestFakeSink:
    def row(self, row_id: int) -> StoredRow:
        return StoredRow(row_id=row_id, values={"t_utc_ns": row_id})

    def test_keeps_what_it_receives_in_order(self) -> None:
        sink = FakeSink(record_types={"seeing_window"})
        assert sink.accepts("seeing_window")
        assert not sink.accepts("event")
        sink.send("seeing_window", [self.row(1), self.row(2)])
        sink.send("seeing_window", [self.row(3)])
        assert [row.row_id for row in sink.rows("seeing_window")] == [1, 2, 3]
        assert sink.rows("event") == []

    def test_scripted_failures_keep_the_batch_out(self) -> None:
        sink = FakeSink()
        sink.fail_next(2, retryable=False)
        for _ in range(2):
            with pytest.raises(SinkError) as raised:
                sink.send("event", [self.row(1)])
            assert raised.value.retryable is False
        sink.send("event", [self.row(1)])
        assert len(sink.rows("event")) == 1

    def test_rejects_empty_and_oversized_batches(self) -> None:
        sink = FakeSink(max_batch_rows=2)
        with pytest.raises(ValueError, match="batch"):
            sink.send("event", [])
        with pytest.raises(ValueError, match="batch"):
            sink.send("event", [self.row(1), self.row(2), self.row(3)])


class TestFakeSolver:
    request = SolveRequest(
        stars=StarList(x=np.zeros(3), y=np.zeros(3), flux=np.ones(3)),
        width_px=100,
        height_px=100,
        scale_low_arcsec_px=3.0,
        scale_high_arcsec_px=4.5,
    )

    def test_returns_the_scripted_result_and_remembers_requests(self) -> None:
        result = SolveResult(solved=True, solver="fake", elapsed_s=0.5, n_matched=12)
        solver = FakeSolver(result=result)
        assert solver.solve(self.request) is result
        assert solver.requests == [self.request]

    def test_defaults_to_no_solution_and_can_raise(self) -> None:
        assert FakeSolver().solve(self.request).solved is False
        with pytest.raises(SolverError):
            FakeSolver(error=SolverError("binary missing")).solve(self.request)

    def test_star_list_rejects_mismatched_arrays(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            StarList(x=np.zeros(3), y=np.zeros(2), flux=np.zeros(3))
        assert len(self.request.stars) == 3
