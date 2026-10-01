"""`Liveness`, `BeatClock`, and the calls of `InfoDriver`: what the watchdog counts as progress."""

from __future__ import annotations

import threading

import pytest

from seeingmon.clock import VirtualClock
from seeingmon.drivers.base import CameraInfo
from seeingmon.frames import Frame, StreamConfig
from seeingmon.services.core.driver_proxy import InfoDriver
from seeingmon.services.core.liveness import BeatClock, Liveness
from seeingmon.testing import FakeCameraDriver


def make() -> tuple[VirtualClock, Liveness, BeatClock]:
    clock = VirtualClock(1_000_000_000_000_000_000)
    liveness = Liveness(clock, stall_s=30.0)
    return clock, liveness, BeatClock(clock, liveness)


class TestLiveness:
    def test_a_thread_that_makes_progress_is_alive(self) -> None:
        clock, liveness, _ = make()
        assert liveness.alive()
        clock.advance(29.0)
        assert liveness.alive()
        liveness.beat()
        clock.advance(29.0)
        assert liveness.alive()

    def test_a_thread_that_goes_quiet_for_the_stall_limit_is_not_alive(self) -> None:
        clock, liveness, _ = make()
        clock.advance(31.0)
        assert not liveness.alive()
        assert liveness.idle_s == pytest.approx(31.0)
        liveness.beat()
        assert liveness.alive()

    def test_a_call_in_bounds_counts_as_alive_however_long_it_takes(self) -> None:
        clock, liveness, _ = make()
        liveness.expect(200.0)
        clock.advance(150.0)  # far beyond the stall limit, but within the bound of the call
        assert liveness.alive()
        clock.advance(51.0)  # the call outlasts its bound: a hang
        assert not liveness.alive()

    def test_the_end_of_a_call_is_progress_and_ends_the_grace(self) -> None:
        clock, liveness, _ = make()
        liveness.expect(200.0)
        clock.advance(100.0)
        liveness.leave()
        assert liveness.alive()
        clock.advance(31.0)
        assert not liveness.alive()  # the grace of the call is gone

    def test_the_stall_limit_is_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            Liveness(VirtualClock(0), stall_s=0)


class TestBeatClock:
    def test_a_read_of_the_time_and_a_sleep_are_progress(self) -> None:
        clock, liveness, beat = make()
        clock.advance(31.0)
        assert not liveness.alive()
        beat.monotonic_ns()
        assert liveness.alive()
        clock.advance(31.0)
        beat.utc_ns()
        assert liveness.alive()
        clock.advance(31.0)
        beat.sleep(0.01)
        assert liveness.alive()

    def test_it_forwards_the_time_and_the_status(self) -> None:
        clock, _, beat = make()
        assert beat.utc_ns() == clock.utc_ns()
        assert beat.monotonic_ns() == clock.monotonic_ns()
        assert beat.status() == clock.status()
        before = clock.utc_ns()
        beat.sleep(2.0)  # a virtual clock advances
        assert clock.utc_ns() - before == 2_000_000_000

    def test_only_the_bound_thread_counts(self) -> None:
        clock, liveness, beat = make()
        liveness.bind_thread()  # this thread is the scheduler
        clock.advance(31.0)
        assert not liveness.alive()

        def read_all_day() -> None:
            beat.monotonic_ns()
            beat.utc_ns()
            beat.sleep(0.0)

        worker = threading.Thread(target=read_all_day)
        worker.start()
        worker.join()
        assert not liveness.alive()  # another thread reads the clock all day, and proves nothing
        beat.monotonic_ns()
        assert liveness.alive()


class SlowCamera(FakeCameraDriver):
    """A fake camera whose calls take `inside_s` of virtual time, and note the liveness there."""

    def __init__(self, clock: VirtualClock, liveness: Liveness) -> None:
        super().__init__(clock)
        self.virtual = clock
        self.liveness = liveness
        self.inside_s = 0.0
        self.alive_inside: list[bool] = []

    def spend(self) -> None:
        self.virtual.advance(self.inside_s)
        self.alive_inside.append(self.liveness.alive())

    def open(self) -> CameraInfo:
        self.spend()
        return super().open()

    def read_frame(self, timeout_s: float) -> Frame:
        self.spend()
        return super().read_frame(timeout_s)


class TestInfoDriver:
    def build(self) -> tuple[InfoDriver, SlowCamera, VirtualClock, Liveness]:
        clock, liveness, _ = make()
        camera = SlowCamera(clock, liveness)
        driver = InfoDriver(
            camera,
            lambda info: None,
            liveness=liveness,
            call_limit_s=100.0,
            read_margin_s=5.0,
        )
        return driver, camera, clock, liveness

    @pytest.mark.parametrize(("inside_s", "alive"), [(90.0, True), (110.0, False)])
    def test_a_slow_call_is_alive_within_its_limit_and_a_longer_one_is_a_hang(
        self, inside_s: float, alive: bool
    ) -> None:
        driver, camera, _, liveness = self.build()
        camera.inside_s = inside_s
        driver.open()
        assert camera.alive_inside == [alive]
        assert liveness.alive()  # the call is over, and its end is progress

    @pytest.mark.parametrize(("inside_s", "alive"), [(34.0, True), (36.0, False)])
    def test_a_read_may_block_for_its_timeout_and_the_margin(
        self, inside_s: float, alive: bool
    ) -> None:
        driver, camera, _, _ = self.build()
        driver.open()
        driver.configure(StreamConfig("bin2", 1000, 0))
        driver.start()
        camera.alive_inside.clear()
        camera.inside_s = inside_s
        driver.read_frame(30.0)  # the timeout is 30 s, and the margin 5 s
        assert camera.alive_inside == [alive]

    def test_a_call_of_another_thread_gives_no_grace(self) -> None:
        driver, _, clock, liveness = self.build()
        liveness.bind_thread()
        clock.advance(31.0)
        other = threading.Thread(target=driver.open)
        other.start()
        other.join()
        assert not liveness.alive()  # a call from another thread proves nothing about the scheduler

    def test_a_driver_without_liveness_just_forwards(self) -> None:
        clock = VirtualClock(0)
        camera = FakeCameraDriver(clock)
        driver = InfoDriver(camera, lambda info: None)
        assert driver.open().model == "Fake camera"
        assert driver.wrapped is camera
        assert driver.name == "fake"
