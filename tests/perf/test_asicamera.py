"""The production ASI driver on a stub SDK: it reads stored frames, one frame period apart."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from seeingmon.clock import VirtualClock
from seeingmon.drivers.asi.driver import AsiDriver
from seeingmon.frames import Roi, StreamConfig
from seeingmon.perf._asicamera import create_asi_driver

EXPOSURE_US = 10_000
SIZE = 128


def pool_of(count: int = 4) -> npt.NDArray[np.uint16]:
    values = np.arange(count, dtype=np.uint16)[:, None, None] * np.uint16(100)
    return np.broadcast_to(values, (count, SIZE, SIZE)).copy()


def started_driver(
    count: int = 4,
) -> tuple[VirtualClock, AsiDriver, npt.NDArray[np.uint16]]:
    clock = VirtualClock()
    pool = pool_of(count)
    driver = create_asi_driver(clock, {f"f{SIZE}x{SIZE}": pool}, watchdog_thread=False)
    driver.open()
    driver.configure(StreamConfig("bin1", EXPOSURE_US, 120, roi=Roi(100, 200, SIZE, SIZE)))
    driver.start()
    return clock, driver, pool


class TestStubSdk:
    def test_the_driver_returns_the_stored_frames_in_turn(self) -> None:
        _, driver, pool = started_driver()
        frames = [driver.read_frame(1.0) for _ in range(6)]
        first = next(i for i in range(len(pool)) if np.array_equal(frames[0].data, pool[i]))
        for offset, frame in enumerate(frames):
            np.testing.assert_array_equal(frame.data, pool[(first + offset) % len(pool)])
        assert [frame.seq for frame in frames] == list(range(6))

    def test_a_read_waits_for_one_frame_period_because_the_sdk_has_no_readout_time(self) -> None:
        clock, driver, _ = started_driver()
        driver.read_frame(1.0)
        before = clock.monotonic_ns()
        driver.read_frame(1.0)
        assert clock.monotonic_ns() - before == EXPOSURE_US * 1_000

    def test_the_frames_describe_the_stream_and_carry_an_arrival_time(self) -> None:
        _, driver, _ = started_driver()
        frame = driver.read_frame(1.0)
        assert (frame.mode, frame.gain, frame.exposure_us) == ("bin1", 120, EXPOSURE_US)
        assert (frame.roi.width, frame.roi.height) == (SIZE, SIZE)
        assert frame.t_arrival_ns > 0

    def test_the_drop_counter_stays_at_zero(self) -> None:
        _, driver, _ = started_driver()
        assert all(driver.read_frame(1.0).dropped_before == 0 for _ in range(5))
