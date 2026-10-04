"""Late reads at the capture thread: what the drop accounting makes of them.

The tests hand frames to an `AcquireService` that nobody started, at times that they choose, and
read the frames that reach the queue. The service sees the same calls that its capture thread makes.
"""

from __future__ import annotations

import secrets
from dataclasses import replace

import pytest

from seeingmon.clock import DEFAULT_START_UTC_NS, VirtualClock
from seeingmon.frames import ActiveStream, Frame
from seeingmon.services.acquire.service import AcquireService, timing_config
from seeingmon.services.acquire.timing import TimeStamper
from seeingmon.services.config import AcquireSettings, ServicesConfig
from seeingmon.services.ipc.endpoint import Endpoint
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.testing import FakeCameraDriver

from .rig import FAST


class Bench:
    """An `AcquireService` with a stream that it believes runs, and frames that arrive on demand."""

    def __init__(self, **acquire: object) -> None:
        self.clock = VirtualClock(start_utc_ns=DEFAULT_START_UTC_NS)
        driver = FakeCameraDriver(self.clock)
        options: dict[str, object] = {
            "raise_priority": False,
            "time_source": "stamp",
            "queue_depth": 2048,  # nobody reads the queue
            **acquire,
        }
        settings = ServicesConfig(acquire=AcquireSettings(**options))
        self.service = AcquireService(
            driver,
            self.clock,
            Endpoint.loopback(0),
            ConnectionKey.from_text(secrets.token_urlsafe(24)),
            settings,
            priority_hook=lambda: "test",
        )
        driver.open()
        self.active: ActiveStream = driver.configure(FAST)
        service = self.service
        service._stamper.configure(service._stream_timing(self.active))
        service._drops.reset(self.active.frame_period_s)
        service._timing_stream_id = self.active.stream_id
        service._active = self.active
        service._capturing = True
        self.epoch = service._new_epoch()
        driver.start()
        self.template = driver.read_frame(1.0)
        assert self.active.frame_period_s
        self.period_ns = round(self.active.frame_period_s * 1e9)
        self.frames: list[Frame] = []
        original = service._queue.put_frame

        def record(frame: Frame, epoch: int) -> int:
            self.frames.append(frame)
            return original(frame, epoch)

        service._queue.put_frame = record  # type: ignore[method-assign]
        self.at_ns = 0
        self.seq = 0

    def arrive(self, after_periods: float, *, counted: int = 0) -> None:
        """A frame comes `after_periods` frame periods after the previous one.

        `counted` is what the driver reports in `dropped_before`, as the SDK counter would.
        """
        self.at_ns += round(after_periods * self.period_ns)
        frame = replace(self.template, seq=self.seq, t_arrival_ns=0, dropped_before=counted)
        self.seq += 1
        self.service._on_frame(
            frame, DEFAULT_START_UTC_NS + self.at_ns, self.at_ns, self.active, self.epoch
        )

    @property
    def lost(self) -> list[int]:
        """The `dropped_before` of each frame that reached the queue."""
        return [frame.dropped_before for frame in self.frames]


def test_the_gap_rule_ignores_the_period_of_the_time_fit() -> None:
    bench = Bench()

    class ShrunkenFit(TimeStamper):
        """What the fit reads after phantom drops have counted into its frame numbers."""

        @property
        def period_s(self) -> float | None:
            return 0.001

    shrunken = ShrunkenFit(bench.clock, timing_config(bench.service._cfg))
    shrunken.configure(bench.service._stream_timing(bench.active))
    bench.service._stamper = shrunken
    for _ in range(60):
        bench.arrive(1.0)
    assert bench.lost == [0] * 60
    assert bench.service.health().dropped_gap == 0


def test_late_reads_followed_by_catch_ups_lose_no_frame_and_leave_the_fit_alone() -> None:
    bench = Bench()
    bench.arrive(1.0)
    for _ in range(100):  # three frames in three periods: one read that is 1.6 periods late
        for step in (2.6, 0.2, 0.2):
            bench.arrive(step)
    assert bench.lost == [0] * 301
    health = bench.service.health()
    assert (health.dropped_driver, health.dropped_gap, health.dropped_queue) == (0, 0, 0)
    assert health.late_reads == 100
    # The fit counts frames as they came, so its period stays the period of the camera.
    assert bench.service._stamper.period_s == pytest.approx(bench.period_ns / 1e9, rel=0.02)
    assert health.time_resets == 0


def test_a_real_gap_reaches_the_frame_after_it_and_the_fit_follows() -> None:
    bench = Bench()
    for _ in range(40):
        bench.arrive(1.0)
    bench.arrive(3.0)  # two frames never came
    for _ in range(40):
        bench.arrive(1.0)
    assert bench.lost == [0] * 41 + [2] + [0] * 39
    health = bench.service.health()
    assert (health.dropped_driver, health.dropped_gap) == (0, 2)
    assert health.time_resets == 0


def test_a_loss_that_the_driver_counted_reaches_the_frame_at_once() -> None:
    bench = Bench()
    for _ in range(40):
        bench.arrive(1.0)
    bench.arrive(2.0, counted=1)  # one frame lost, and the SDK counted it
    bench.arrive(1.0)
    assert bench.lost == [0] * 40 + [1, 0]
    assert bench.service.health().dropped_driver == 1
