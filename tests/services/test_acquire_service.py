"""The `acquire` service and the remote driver, in one process, on real sockets and threads."""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from seeingmon.clock import DEFAULT_START_UTC_NS, ClockStatus, VirtualClock
from seeingmon.drivers.base import (
    CameraCaps,
    CameraConfigError,
    CameraDisconnectedError,
    CameraError,
    CameraLinkError,
    CameraStateError,
    CameraTimeoutError,
    RecoveryLevel,
)
from seeingmon.frames import (
    Frame,
    FrameFlag,
    PixelFormat,
    Roi,
    StreamConfig,
    StreamKind,
    TimeQuality,
)
from seeingmon.services.acquire.events import HardwareEventLog
from seeingmon.services.acquire.service import AcquireService
from seeingmon.services.config import AcquireSettings, CallTimeouts, ServicesConfig
from seeingmon.services.ipc.endpoint import Endpoint
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.remote import RemoteCameraDriver

from .conftest import wait_until
from .rig import (
    FAST,
    SMALL,
    OpenCountingFake,
    ParkingFake,
    PassiveRecoverFake,
    Rig,
    TracingFake,
    ZeroArrivalFake,
    make_rig,
)

RigFactory = Callable[..., Rig]


@pytest.fixture
def build(native: Endpoint, key: ConnectionKey) -> Iterator[RigFactory]:
    rigs: list[Rig] = []

    def factory(**options: Any) -> Rig:
        rig = make_rig(options.pop("endpoint", native), options.pop("key", key), **options)
        rigs.append(rig)
        return rig

    yield factory
    for rig in rigs:
        rig.close()


def virtual_rig(build: RigFactory, parked_at: int, **options: Any) -> Rig:
    """A rig on a virtual clock whose fake parks after `parked_at` frames. The counts are exact."""
    clock = VirtualClock(start_utc_ns=DEFAULT_START_UTC_NS)
    fake_options = {**options.pop("fake_options", {}), "park_after": parked_at}
    return build(clock=clock, fake_class=ParkingFake, fake_options=fake_options, **options)


def pending(driver: RemoteCameraDriver) -> int:
    """The frames that have reached the receiver of the driver and that nobody has read."""
    link = driver._link
    return 0 if link is None else link.frames.pending


def read_until_seq(driver: RemoteCameraDriver, last_seq: int) -> list[Frame]:
    """Read frames until the one with sequence number `last_seq`, the newest, which never drops."""
    frames = [driver.read_frame(10.0)]
    while frames[-1].seq < last_seq:
        frames.append(driver.read_frame(10.0))
    return frames


def read_frames(driver: RemoteCameraDriver, count: int, timeout_s: float = 10.0) -> list[Frame]:
    return [driver.read_frame(timeout_s) for _ in range(count)]


def streaming(rig: Rig, config: StreamConfig = FAST) -> RemoteCameraDriver:
    driver = rig.driver()
    driver.open()
    driver.configure(config)
    driver.start()
    return driver


def read_until_it_raises(driver: RemoteCameraDriver, timeout_s: float = 10.0) -> None:
    """Read frames until the driver raises, which ends the test that expects it. Never returns."""
    while True:
        driver.read_frame(timeout_s)


class DeadThread:
    """Stands in for a thread that has died."""

    def is_alive(self) -> bool:
        return False

    def join(self, timeout: float | None = None) -> None:
        return None


def assert_every_frame_is_accounted_for(frames: list[Frame]) -> None:
    """The sequence number is the delivered count before a frame plus the drops reported so far."""
    reported = 0
    for position, frame in enumerate(frames):
        reported += frame.dropped_before
        assert frame.seq == position + reported, (position, frame.seq, reported)


class TestStreaming:
    def test_frames_arrive_in_order_and_the_driver_time_is_kept(self, build: RigFactory) -> None:
        rig = build()
        driver = rig.driver()
        info = driver.open()
        assert (info.driver, info.model) == ("fake", "Fake camera")
        active = driver.configure(FAST)
        assert active.frame_shape == (128, 128)
        driver.start()
        frames = read_frames(driver, 25)
        assert [f.seq for f in frames] == list(range(25))
        assert all(f.stream_id == active.stream_id for f in frames)
        assert all(f.t_quality is TimeQuality.EXACT for f in frames)  # time_source is auto
        assert all(f.flags & FrameFlag.SIMULATED for f in frames)
        assert all(f.data.shape == (128, 128) for f in frames)
        assert_every_frame_is_accounted_for(frames)
        health = driver.health()
        assert health["state"] == "streaming"
        assert health["frames_captured"] >= 25
        assert health["client_connected"]
        assert health["stream_connected"]
        assert driver.ping() == rig.service.instance

    def test_stamped_frames_get_a_fitted_time_and_an_honest_error(self, build: RigFactory) -> None:
        rig = virtual_rig(build, 60, acquire={"time_source": "stamp"})
        assert isinstance(rig.fake, ParkingFake)
        driver = streaming(rig)
        assert rig.fake.parked.wait(30.0)  # the fake made its 60 frames, and the queue holds them
        frames = read_frames(driver, 60)
        period_ns = 11_312_800  # 6.5 ms of overhead and 128 rows of 37.6 us
        offset_ns = period_ns - 1_000_000  # the period minus half the 2 ms exposure
        # Virtual arrivals are exact, so the line fits them exactly: the first 19 frames fill the
        # window, and every frame after that is fitted.
        assert [f.t_quality for f in frames[:19]] == [TimeQuality.ESTIMATED] * 19
        assert [f.t_quality for f in frames[19:]] == [TimeQuality.FITTED] * 41
        assert all(f.t_utc_ns == f.t_arrival_ns - offset_ns for f in frames)
        # The error is the clock bound (0), the latency sigma (5 ms), and the jitter or the fit.
        assert all(f.t_err_ns == 7_000_000 for f in frames[:19])  # 5 ms and a 2 ms jitter
        assert all(f.t_err_ns == 5_000_000 for f in frames[19:])  # a perfect line has no error

    def test_a_clock_that_is_not_synchronized_marks_the_time_invalid(
        self, build: RigFactory
    ) -> None:
        status = ClockStatus(synchronized=False, error_bound_ns=None, source="test")
        rig = build(acquire={"time_source": "stamp"}, status=status)
        frames = read_frames(streaming(rig), 5)
        assert all(f.t_quality is TimeQuality.INVALID for f in frames)
        assert all(f.flags & FrameFlag.TIME_INVALID for f in frames)

    def test_the_capture_thread_stamps_a_frame_that_the_driver_left_unstamped(
        self, build: RigFactory
    ) -> None:
        class ZeroArrivalParking(ZeroArrivalFake, ParkingFake):
            pass

        rig = build(
            clock=VirtualClock(start_utc_ns=DEFAULT_START_UTC_NS),
            fake_class=ZeroArrivalParking,
            fake_options={"park_after": 30},
            acquire={"time_source": "stamp"},
        )
        assert isinstance(rig.fake, ParkingFake)
        driver = streaming(rig)
        assert rig.fake.parked.wait(30.0)
        frames = read_frames(driver, 30)
        # The driver sent no arrival time, so the capture thread read the clock after each read.
        arrivals = [f.t_arrival_ns for f in frames]
        assert all(arrival > DEFAULT_START_UTC_NS for arrival in arrivals)
        assert arrivals == sorted(set(arrivals))
        assert frames[-1].t_quality is TimeQuality.FITTED

    def test_a_snapshot_stream_delivers_one_frame_per_start(self, build: RigFactory) -> None:
        rig = build()
        driver = rig.driver()
        driver.open()
        driver.configure(
            StreamConfig(
                mode="bin1",
                exposure_us=200_000,
                gain=1,
                kind=StreamKind.SNAPSHOT,
                roi=Roi(0, 0, 16, 16),
            )
        )
        driver.start()
        assert driver.read_frame(10.0).seq == 0
        with pytest.raises(CameraStateError):
            driver.read_frame(1.0)
        driver.start()
        assert driver.read_frame(10.0).seq == 1

    def test_a_pixel_format_survives_the_trip(self, build: RigFactory) -> None:
        rig = build()
        driver = rig.driver()
        driver.open()
        driver.configure(
            StreamConfig(
                mode="bin1",
                exposure_us=2000,
                gain=1,
                roi=Roi(0, 0, 16, 16),
                pixel_format=PixelFormat.RAW8,
            )
        )
        driver.start()
        frame = driver.read_frame(10.0)
        assert frame.data.dtype.itemsize == 1

    def test_the_service_runs_on_the_loopback_socket_too(self, key: ConnectionKey) -> None:
        rig = make_rig(Endpoint.loopback(0), key)
        try:
            frames = read_frames(streaming(rig), 5)
            assert [f.seq for f in frames] == [0, 1, 2, 3, 4]
        finally:
            rig.close()


class TestDrops:
    def test_the_driver_counter_reaches_the_next_frame_and_the_health(
        self, build: RigFactory
    ) -> None:
        rig = build()
        driver = rig.driver()
        driver.open()
        driver.configure(FAST)
        rig.fake.drop_frames(3)
        driver.start()
        first, second = read_frames(driver, 2)
        assert (first.dropped_before, second.dropped_before) == (3, 0)
        assert driver.dropped_frames() == 3
        health = driver.health()
        assert health["dropped_driver"] == 3
        assert health["dropped_gap"] == 0

    def test_a_slow_consumer_makes_the_queue_drop_and_count(self, build: RigFactory) -> None:
        rig = virtual_rig(
            build,
            200,
            acquire={"queue_depth": 4},
            services={"stream_window_messages": 2, "stream_batch_frames": 1},
        )
        assert isinstance(rig.fake, ParkingFake)
        driver = streaming(rig)
        assert rig.fake.parked.wait(30.0)  # the camera made 200 frames while nobody read
        # Nobody reads, so the sender fills the window of two and then waits for credit.
        assert wait_until(lambda: driver.health()["flow_stalls"] > 0, 10.0)
        frames = read_until_seq(driver, 199)
        assert_every_frame_is_accounted_for(frames)
        reported = sum(f.dropped_before for f in frames)
        assert len(frames) + reported == 200  # every frame arrived, or was counted as dropped
        health = driver.health()
        assert health["dropped_queue"] == reported
        assert health["dropped_queue"] >= 190  # the window and the queue hold ten frames at most
        assert health["queue_peak_frames"] <= 4

    def test_the_byte_bound_protects_the_memory_of_large_frames(self, build: RigFactory) -> None:
        frame_bytes = 64 * 64 * 2
        rig = virtual_rig(
            build,
            100,
            acquire={"queue_depth": 1000, "queue_max_bytes": 3 * (frame_bytes + 96)},
            services={"stream_window_messages": 1},
        )
        assert isinstance(rig.fake, ParkingFake)
        driver = rig.driver()
        driver.open()
        driver.configure(StreamConfig(mode="bin1", exposure_us=2000, gain=1, roi=Roi(0, 0, 64, 64)))
        driver.start()
        assert rig.fake.parked.wait(30.0)
        frames = read_until_seq(driver, 99)
        assert_every_frame_is_accounted_for(frames)
        assert len(frames) + sum(f.dropped_before for f in frames) == 100
        assert driver.health()["queue_peak_frames"] <= 3

    def test_a_gap_that_no_counter_explains_is_counted(self, build: RigFactory) -> None:
        rig = virtual_rig(
            build,
            60,
            fake_options={"stall_at": 30, "stall_periods": 5},  # frame 30 comes five periods late
            acquire={"gap_factor": 1.5},
        )
        assert isinstance(rig.fake, ParkingFake)
        driver = streaming(rig)
        assert rig.fake.parked.wait(30.0)
        frames = read_frames(driver, 60)
        # The interval before frame 30 is six periods, so five frames are missing. Frame 31
        # follows after a regular interval, which confirms the gap, so it carries the count. Every
        # other frame follows its predecessor by one period.
        assert [f.dropped_before for f in frames] == [0] * 31 + [5] + [0] * 28
        health = driver.health()
        assert (health["dropped_gap"], health["dropped_driver"], health["dropped_queue"]) == (
            5,
            0,
            0,
        )


class TestStaleFrames:
    def test_frames_of_an_old_stream_never_reach_the_caller_after_configure(
        self, build: RigFactory
    ) -> None:
        rig = build()
        driver = streaming(rig, SMALL)
        first = driver.read_frame(10.0)
        assert wait_until(lambda: pending(driver) >= 3, 10.0)  # old frames wait in the receiver
        second = driver.configure(
            StreamConfig(mode="bin1", exposure_us=2000, gain=1, roi=Roi(0, 0, 32, 32))
        )
        assert second.stream_id == first.stream_id + 1
        driver.start()
        frames = read_frames(driver, 15)
        assert [f.seq for f in frames] == list(range(15))
        assert all(f.stream_id == second.stream_id and f.roi.width == 32 for f in frames)
        assert driver.stale_discarded > 0

    def test_a_stop_and_start_discard_what_was_in_flight(self, build: RigFactory) -> None:
        rig = build()
        driver = streaming(rig, SMALL)
        read_frames(driver, 3)
        assert wait_until(lambda: pending(driver) >= 3, 10.0)  # frames wait in the receiver
        driver.stop()
        driver.start()  # the same stream, a new epoch
        frames = read_frames(driver, 10)
        assert driver.stale_discarded > 0
        assert all(f.t_arrival_ns >= frames[0].t_arrival_ns for f in frames)
        # `acquire` flushes at `start`, so no frame of the old epoch is among these. Frames of
        # the new epoch carry the driver's own sequence, which continues from the old one.
        assert [b.seq - a.seq for a, b in itertools.pairwise(frames)] == [1] * 9

    def test_reading_after_a_stop_is_a_state_error(self, build: RigFactory) -> None:
        driver = streaming(build())
        driver.read_frame(10.0)
        driver.stop()
        with pytest.raises(CameraStateError):
            driver.read_frame(0.5)


class TestErrorsFromTheDriver:
    def test_scripted_errors_arrive_one_per_read_and_in_order(self, build: RigFactory) -> None:
        rig = build(speed=5.0)
        driver = rig.driver()
        driver.open()
        driver.configure(FAST)
        rig.fake.fail_reads(CameraTimeoutError("stall"), CameraDisconnectedError("unplugged"))
        driver.start()
        with pytest.raises(CameraTimeoutError, match="stall"):
            driver.read_frame(10.0)
        with pytest.raises(CameraDisconnectedError, match="unplugged"):
            driver.read_frame(10.0)
        assert driver.read_frame(10.0).seq == 0
        health = driver.health()
        assert (health["read_timeouts"], health["read_errors"]) == (1, 1)
        assert health["last_error"] == "CameraDisconnectedError: unplugged"

    def test_an_error_that_repeats_takes_one_place_in_the_queue(self, build: RigFactory) -> None:
        rig = virtual_rig(
            build,
            10,
            acquire={"error_backoff_s": 0.0},
            services={"stream_window_messages": 1},  # the sender sends one message and then waits
        )
        assert isinstance(rig.fake, ParkingFake)
        driver = rig.driver()
        driver.open()
        driver.configure(FAST)
        rig.fake.fail_reads(*[CameraDisconnectedError("unplugged") for _ in range(40)])
        driver.start()
        assert rig.fake.parked.wait(30.0)  # the 40 errors came first, and then ten frames
        assert driver.health()["read_errors"] == 40
        stats = rig.service._queue.stats
        assert (stats.events_in, stats.events_dropped) == (40, 0)  # nothing was lost
        # The queue held at most one event at a time, and the sender may have sent the first of
        # them before the rest came, so the caller sees the error once or twice, and then frames.
        errors = 0
        frames: list[Frame] = []
        while len(frames) < 10:
            try:
                frames.append(driver.read_frame(10.0))
            except CameraDisconnectedError:
                errors += 1
                assert errors <= 2
        assert errors >= 1
        assert [f.seq for f in frames] == list(range(10))

    def test_the_driver_stopping_by_itself_is_a_state_error_for_the_caller(
        self, build: RigFactory
    ) -> None:
        rig = build()
        driver = streaming(rig, SMALL)
        driver.read_frame(10.0)
        rig.fake.stop()  # the driver stops without being asked, as a driver may after a fault
        with pytest.raises(CameraStateError):
            read_until_it_raises(driver)
        with pytest.raises(CameraStateError):
            driver.read_frame(0.5)  # and it stays stopped until the next start

    def test_errors_of_the_camera_on_control_calls_keep_their_class(
        self, build: RigFactory
    ) -> None:
        driver = build().driver()
        with pytest.raises(CameraStateError):
            driver.configure(FAST)  # before open: refused here, without a call
        driver.open()
        with pytest.raises(CameraStateError):
            driver.start()  # the driver in acquire says so
        with pytest.raises(CameraConfigError):
            driver.configure(StreamConfig(mode="bin9", exposure_us=1, gain=0))
        with pytest.raises(CameraConfigError):
            driver.configure(
                StreamConfig(mode="bin1", exposure_us=1, gain=0, roi=Roi(0, 0, 9000, 8))
            )
        assert driver.configure(FAST).stream_id >= 1  # and a good call still works

    def test_a_failing_open_comes_back_as_the_camera_error(self, build: RigFactory) -> None:
        rig = build()
        rig.fake.fail_open()
        driver = rig.driver()
        with pytest.raises(CameraDisconnectedError, match="no camera"):
            driver.open()
        assert driver.connected  # the session is fine, and the caller may try again


class TestControlCalls:
    def test_each_call_reaches_the_driver(self, build: RigFactory) -> None:
        rig = build()
        driver = rig.driver()
        driver.open()
        caps = driver.capabilities()
        assert isinstance(caps, CameraCaps)
        assert caps == rig.fake.capabilities()
        assert driver.read_temperature_c() == 18.0
        driver.configure(FAST)
        driver.start()
        assert driver.move_roi(-50, 10_000) == Roi(0, 5644 - 128, 128, 128)
        driver.recover(RecoveryLevel.RESTART_CAPTURE)
        driver.stop()
        driver.stop()
        driver.close()
        driver.close()
        assert ("recover", RecoveryLevel.RESTART_CAPTURE) in rig.fake.calls
        assert [c for c, _ in rig.fake.calls].count("close") >= 1

    def test_a_camera_without_a_sensor_reports_none(self, build: RigFactory) -> None:
        rig = build(fake_options={"temperature_c": None})
        driver = rig.driver()
        driver.open()
        assert driver.read_temperature_c() is None

    def test_a_failing_recovery_comes_back_as_a_camera_error(self, build: RigFactory) -> None:
        rig = build()
        driver = rig.driver()
        driver.open()
        rig.fake.fail_recover()
        with pytest.raises(CameraError, match="recovery did not work"):
            driver.recover(RecoveryLevel.USB_RESET)

    def test_moved_roi_shows_in_later_frames(self, build: RigFactory) -> None:
        driver = streaming(build(), SMALL)
        moved = driver.move_roi(100, 200)
        deadline = time.monotonic() + 10.0
        frame = driver.read_frame(10.0)
        while frame.roi != moved and time.monotonic() < deadline:
            frame = driver.read_frame(10.0)
        assert frame.roi == moved

    def test_the_recovered_flag_marks_the_first_frame_after_a_recovery(
        self, build: RigFactory
    ) -> None:
        rig = build(fake_class=PassiveRecoverFake)
        driver = streaming(rig, SMALL)
        read_frames(driver, 3)
        driver.recover(RecoveryLevel.RESTART_CAPTURE)
        frames = read_frames(driver, 60)
        flagged = [f for f in frames if f.flags & FrameFlag.RECOVERED]
        assert len(flagged) == 1

    def test_open_in_a_new_session_does_not_open_the_camera_again(self, build: RigFactory) -> None:
        rig = build(fake_class=OpenCountingFake)
        first = rig.driver()
        first.open()
        second = rig.driver()
        info = second.open()
        assert info.driver == "fake"
        assert isinstance(rig.fake, OpenCountingFake)
        assert rig.fake.opens == 1


class TestSessions:
    def test_a_new_client_replaces_the_old_one_and_the_capture_stops(
        self, build: RigFactory
    ) -> None:
        rig = build()
        first = streaming(rig, SMALL)
        read_frames(first, 3)
        second = rig.driver()
        second.open()
        with pytest.raises(CameraDisconnectedError):
            read_until_it_raises(first)
        with pytest.raises(CameraDisconnectedError):
            first.configure(SMALL)
        assert wait_until(lambda: ("stop", None) in rig.fake.calls)
        second.configure(SMALL)
        second.start()
        assert read_frames(second, 3)[0].seq >= 0
        assert second.health()["client_connected"]

    def test_closing_the_driver_stops_the_capture_and_a_new_client_takes_over(
        self, build: RigFactory
    ) -> None:
        rig = build()
        driver = streaming(rig, SMALL)
        read_frames(driver, 3)
        driver.close()
        assert wait_until(lambda: rig.service.health().state == "closed")
        assert not rig.service.health().capturing
        again = rig.driver()
        again.open()
        again.configure(SMALL)
        again.start()
        assert again.read_frame(10.0).seq == 0

    def test_a_client_that_vanishes_stops_the_capture_and_keeps_the_camera_open(
        self, build: RigFactory
    ) -> None:
        rig = build()
        driver = streaming(rig, SMALL)
        read_frames(driver, 3)
        # Drop the connections without a `close` call, as a crashed `core` does.
        link = driver._link
        assert link is not None
        link.close("test")
        assert wait_until(lambda: not rig.service.health().client_connected)
        assert wait_until(lambda: not rig.service.health().capturing)
        assert rig.service.health().opened

    def test_a_frame_stream_without_a_session_is_refused(
        self, build: RigFactory, key: ConnectionKey
    ) -> None:
        from seeingmon.services.ipc.errors import IpcProtocolError
        from seeingmon.services.ipc.stream import connect_stream

        rig = build()
        with pytest.raises(IpcProtocolError, match="session"):
            connect_stream(rig.endpoint, key, {"session": "nope"}, channel="frames")


class TestConnecting:
    def test_a_wrong_key_is_a_config_error(
        self, build: RigFactory, other_key: ConnectionKey
    ) -> None:
        rig = build()
        driver = RemoteCameraDriver(rig.endpoint, other_key, connect_timeout_s=2.0)
        with pytest.raises(CameraConfigError, match="connection key"):
            driver.open()
        assert rig.service.health().client_connected is False

    def test_nobody_listening_is_a_disconnect_after_the_timeout(
        self, native: Endpoint, key: ConnectionKey
    ) -> None:
        driver = RemoteCameraDriver(native, key, connect_timeout_s=0.3)
        started = time.monotonic()
        with pytest.raises(CameraDisconnectedError, match="cannot reach acquire") as raised:
            driver.open()
        assert 0.15 <= time.monotonic() - started < 20.0
        assert isinstance(raised.value, CameraLinkError)  # the camera is out of reach, not gone

    def test_calls_before_open_are_state_errors(self, native: Endpoint, key: ConnectionKey) -> None:
        driver = RemoteCameraDriver(native, key)
        for call in (
            lambda: driver.capabilities(),
            lambda: driver.configure(FAST),
            lambda: driver.start(),
            lambda: driver.read_frame(0.1),
            lambda: driver.move_roi(1, 1),
        ):
            with pytest.raises(CameraStateError):
                call()
        driver.stop()  # safe before open
        driver.close()
        driver.close()

    def test_the_driver_is_a_camera_driver(self, native: Endpoint, key: ConnectionKey) -> None:
        from seeingmon.drivers import CameraDriver

        driver = RemoteCameraDriver(native, key)
        assert isinstance(driver, CameraDriver)
        assert driver.name == "remote"
        assert not driver.connected
        assert driver.instance is None

    def test_a_service_that_restarts_is_a_disconnect_and_open_reconnects(
        self, native: Endpoint, key: ConnectionKey
    ) -> None:
        first = make_rig(native, key)
        driver = first.driver()
        try:
            driver.open()
            driver.configure(SMALL)
            driver.start()
            read_frames(driver, 3)
            old_instance = driver.instance
            first.service.stop()
            with pytest.raises(CameraLinkError):  # a restart of acquire breaks the link
                read_until_it_raises(driver)
            with pytest.raises(CameraLinkError):
                driver.configure(SMALL)
            second = make_rig(native, key)
            try:
                driver.open()  # the usual ladder: reopen, reconfigure, restart
                assert driver.instance != old_instance
                assert driver.connections == 2
                driver.configure(SMALL)
                driver.start()
                assert driver.read_frame(10.0).seq == 0
            finally:
                driver.close()
                second.close()
        finally:
            first.close()


class TestRecoveryWithoutASession:
    """A restart of acquire, or an acquire that is not up yet, is the first step of the ladder."""

    def test_the_mildest_step_reconnects_after_acquire_restarted(
        self, native: Endpoint, key: ConnectionKey
    ) -> None:
        first = make_rig(native, key)
        driver = first.driver()
        try:
            driver.open()
            driver.configure(SMALL)
            driver.start()
            read_frames(driver, 3)
            old_instance = driver.instance
            first.service.stop()
            with pytest.raises(CameraDisconnectedError):
                read_until_it_raises(driver)
            with pytest.raises(CameraDisconnectedError):
                driver.recover(RecoveryLevel.RESTART_CAPTURE)  # nobody answers yet
            second = make_rig(native, key)
            try:
                driver.recover(RecoveryLevel.RESTART_CAPTURE)
                assert driver.connected
                assert driver.instance != old_instance
                driver.configure(SMALL)  # the scheduler configures a new stream after a fault
                driver.start()
                assert driver.read_frame(10.0).seq == 0
            finally:
                driver.close()
                second.close()
        finally:
            first.close()

    def test_a_driver_that_never_opened_opens_on_the_first_step(
        self, native: Endpoint, key: ConnectionKey
    ) -> None:
        driver = RemoteCameraDriver(native, key, connect_timeout_s=0.2)
        with pytest.raises(CameraDisconnectedError):
            driver.recover(RecoveryLevel.REOPEN)
        rig = make_rig(native, key)
        try:
            driver.recover(RecoveryLevel.REOPEN)
            assert driver.open().driver == "fake"  # the camera is open, and open is idempotent
        finally:
            driver.close()
            rig.close()


class TestStatusCalls:
    """The calls that report on `acquire` come from the supervisor of `core`: they are short."""

    def test_health_events_and_ping_use_the_short_timeout(
        self, native: Endpoint, key: ConnectionKey
    ) -> None:
        rig = make_rig(native, key)
        driver = rig.driver()
        try:
            driver.open()
            timeouts: dict[str, float | None] = {}
            original = driver._call

            def spy(
                method: str,
                params: Any = None,
                *,
                slow: bool = False,
                timeout_s: float | None = None,
            ) -> Any:
                timeouts[method] = timeout_s
                return original(method, params, slow=slow, timeout_s=timeout_s)

            driver._call = spy  # type: ignore[method-assign]
            driver.health()
            driver.events()
            driver.ping()
            assert timeouts == {"health": 5.0, "events": 5.0, "ping": 5.0}
        finally:
            driver.close()
            rig.close()


class TestGuardAndWatchdog:
    def test_a_driver_call_that_hangs_is_reported_by_the_guard(self, build: RigFactory) -> None:
        timeouts = CallTimeouts(configure_s=0.3)
        rig = build(acquire={"call_timeouts": timeouts})
        block = threading.Event()
        rig.fake.block = block
        driver = rig.driver()
        driver.open()
        outcome: list[Any] = []
        thread = threading.Thread(target=lambda: outcome.append(driver.configure(FAST)))
        thread.start()
        try:
            assert rig.fake.entered.wait(10.0)
            assert wait_until(lambda: len(rig.hangs) == 1, 10.0)
            assert rig.hangs[0].name == "configure"
            # While the call hangs, the quick calls still answer.
            assert driver.ping() == rig.service.instance
            assert driver.health()["state"] in {"ready", "streaming"}
            time.sleep(0.4)
            assert len(rig.hangs) == 1  # reported once
        finally:
            block.set()
            thread.join(20.0)
        assert not thread.is_alive()
        assert len(outcome) == 1
        assert outcome[0].stream_id >= 1

    def test_the_watchdog_sends_ready_heartbeats_and_status(self, build: RigFactory) -> None:
        rig = build(notify=True, acquire={"heartbeat_interval_s": 0.1})
        assert wait_until(lambda: sum(m == b"WATCHDOG=1" for m in rig.notifications) >= 3, 10.0)
        assert rig.notifications[0].startswith(b"READY=1")
        assert any(m.startswith(b"STATUS=") for m in rig.notifications)

    def test_a_thread_that_dies_is_fatal(self, build: RigFactory) -> None:
        rig = build()
        rig.service._threads["acquire-sender"] = DeadThread()  # type: ignore[assignment]
        assert wait_until(lambda: bool(rig.fatals), 10.0)
        assert "thread died" in rig.fatals[0]

    def test_the_priority_hook_runs_on_the_capture_thread(self, build: RigFactory) -> None:
        threads: list[str] = []

        def hook() -> str:
            threads.append(threading.current_thread().name)
            return "ran"

        rig = build(priority_hook=hook)
        assert wait_until(lambda: threads == ["acquire-capture"], 10.0)
        assert wait_until(lambda: rig.service.health().priority == "ran", 10.0)

    def test_stopping_ends_every_thread_and_releases_the_driver(
        self, native: Endpoint, key: ConnectionKey
    ) -> None:
        before = set(threading.enumerate())
        rig = make_rig(native, key)
        try:
            driver = streaming(rig, SMALL)
            read_frames(driver, 3)
            rig.service.stop()
            rig.service.stop()  # twice is fine
            assert wait_until(
                lambda: (
                    not any(
                        thread.name.startswith("acquire-") and thread not in before
                        for thread in threading.enumerate()
                    )
                ),
                10.0,
            )
            assert ("close", None) in rig.fake.calls
            assert not rig.service._server.running
        finally:
            rig.close()


class TestRestartRequest:
    def test_a_restart_request_stops_the_service_with_a_reason(self, build: RigFactory) -> None:
        rig = build()
        driver = rig.driver()
        driver.open()
        assert not rig.service.wait(0)
        driver.request_restart("the camera hangs")
        assert rig.service.wait(10.0)
        assert rig.service.exit_reason == "restart requested: the camera hangs"

    def test_a_long_reason_is_cut_and_an_empty_one_gets_a_default(self, build: RigFactory) -> None:
        rig = build()
        driver = rig.driver()
        driver.open()
        driver.request_restart("x" * 1000)
        assert rig.service.wait(10.0)
        assert rig.service.exit_reason == "restart requested: " + "x" * 200


class TestThreadingOfTheDriver:
    def test_the_driver_is_never_called_by_two_threads_at_once(self, build: RigFactory) -> None:
        rig = build()
        driver = streaming(rig, SMALL)
        for _ in range(60):
            assert driver.read_temperature_c() == 18.0
            driver.move_roi(10, 10)
        read_frames(driver, 5)
        assert rig.fake.max_active == 1

    @pytest.mark.parametrize(
        ("mode", "thread_safe", "name", "gated"),
        [
            ("auto", False, "fake", True),
            ("auto", True, "fake", False),
            ("auto", False, "asi", True),  # the name decides nothing: the declaration does
            ("serialize", True, "fake", True),
            ("concurrent", False, "fake", False),
        ],
    )
    def test_the_gate_follows_the_setting_and_what_the_driver_declares(
        self,
        build: RigFactory,
        native: Endpoint,
        key: ConnectionKey,
        mode: str,
        thread_safe: bool,
        name: str,
        gated: bool,
    ) -> None:
        rig = build()

        class Declared(TracingFake):
            @property
            def name(self) -> str:
                return name

        fake = Declared(rig.clock)
        fake.thread_safe = thread_safe  # type: ignore[attr-defined]
        service = AcquireService(
            fake,
            rig.clock,
            native,
            key,
            ServicesConfig(acquire=AcquireSettings(driver_threads=mode)),
            guard=rig.service._guard,
        )
        assert (service._gate is not None) is gated


class TestWithTheAsiDriver:
    """`acquire` runs the real `asi` driver on the fake SDK, which is thread-safe by design."""

    def test_the_driver_streams_is_called_concurrently_and_recovers(
        self, native: Endpoint, key: ConnectionKey
    ) -> None:
        from seeingmon.clock import DEFAULT_START_UTC_NS, ScaledClock
        from seeingmon.drivers.asi import AsiDriver, AsiOptions
        from seeingmon.hardware.asi.fake import FakeAsiSdk, FakeUsbResetter
        from seeingmon.hardware.asi.watchdog import CallWatchdog
        from seeingmon.profile import load_profile

        clock = ScaledClock(
            start_utc_ns=DEFAULT_START_UTC_NS, origin_real_ns=time.time_ns(), speed=1.0
        )
        sdk = FakeAsiSdk(clock)
        hangs: list[Any] = []
        log = HardwareEventLog()
        driver = AsiDriver(
            api=sdk,
            profile=load_profile("asi294mm-gs250"),
            clock=clock,
            options=AsiOptions(),
            usb_resetter=FakeUsbResetter(sdk, reappear_after_s=2.0),
            watchdog=CallWatchdog(clock, hangs.append),
            on_event=log.record,
        )
        service = AcquireService(
            driver,
            clock,
            native,
            key,
            ServicesConfig(acquire=AcquireSettings(raise_priority=False, outlier_floor_s=0.03)),
            guard=CallWatchdog(clock, hangs.append),
            events=log,
            priority_hook=lambda: "test",
            on_fatal=lambda reason: None,
        )
        endpoint = service.start()
        remote = RemoteCameraDriver(endpoint, key, connect_timeout_s=10.0)
        try:
            assert remote.open().driver == "asi"
            remote.configure(FAST)
            remote.start()
            assert AsiDriver.thread_safe  # the driver declares it, so no gate stands in the way
            assert service._gate is None
            frames = read_frames(remote, 60)
            assert frames[0].t_quality is TimeQuality.ESTIMATED  # the driver's own estimate
            assert frames[-1].t_quality is TimeQuality.FITTED  # acquire fitted the arrival times
            for _ in range(20):  # control calls while a read waits
                assert remote.read_temperature_c() is not None
                remote.dropped_frames()
            remote.recover(RecoveryLevel.RESTART_CAPTURE)
            after = read_frames(remote, 60)
            assert any(f.flags & FrameFlag.RECOVERED for f in after)
            assert hangs == []
            # The driver reported the recovery step, and core can collect the event.
            batch = remote.events()
            recoveries = [
                item.event for item in batch.events if item.event.kind == "camera.recovery"
            ]
            assert len(recoveries) == 1
            assert recoveries[0].detail == {"step": int(RecoveryLevel.RESTART_CAPTURE)}
            assert remote.events(after=batch.last).events == ()
        finally:
            remote.close()
            service.stop()


@pytest.mark.slow
def test_a_long_run_with_constant_overflow_accounts_for_every_frame(build: RigFactory) -> None:
    """The camera outruns the consumer for 8,000 frames, and the counts stay exact.

    A scaled clock at 300 times real time turns every stall of a thread into minutes of camera
    time, and the gap rule would count them as lost frames that the synchronous fake never lost.
    The gap rule is off here, so the test checks the queue and the flow control alone.
    """
    rig = build(
        speed=300.0,
        acquire={"queue_depth": 32, "gap_factor": 1e12},
        services={"stream_window_messages": 16},
    )
    driver = streaming(rig, SMALL)
    frames = read_frames(driver, 8_000, timeout_s=30.0)
    assert_every_frame_is_accounted_for(frames)
    health = driver.health()
    assert health["dropped_queue"] >= sum(f.dropped_before for f in frames)
    assert health["queue_peak_frames"] <= 32
    assert health["internal_errors"] == 0
