"""A test rig: an `AcquireService` with a fake driver, and the remote drivers that talk to it.

The service runs in this process, on real sockets and real threads. By default it runs on a
`StatusClock`, so that frames come at a steady pace in real time. A test that counts frames passes
a `VirtualClock` instead: the fake advances virtual time by exactly one period per frame, so the
arrival times, the gaps, and the counts are exact, and no timer of the operating system matters.
The rig records what the guard reports (hangs), what the watchdog thread decides (fatal errors),
and what the notifier would send to systemd.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from typing import Any

from seeingmon.clock import DEFAULT_START_UTC_NS, Clock, ClockStatus, ScaledClock
from seeingmon.drivers.base import CameraInfo, CameraStateError, RecoveryLevel
from seeingmon.frames import ActiveStream, Frame, Roi, StreamConfig
from seeingmon.hardware.asi.watchdog import CallWatchdog, HangReport
from seeingmon.services.acquire.service import AcquireService
from seeingmon.services.config import AcquireSettings, ServicesConfig
from seeingmon.services.ipc.endpoint import Endpoint
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.notify import SystemdNotifier
from seeingmon.services.remote import RemoteCameraDriver
from seeingmon.testing import FakeCameraDriver

FAST = StreamConfig(mode="bin1", exposure_us=2000, gain=120, roi=Roi(100, 200, 128, 128))
SMALL = StreamConfig(mode="bin1", exposure_us=2000, gain=120, roi=Roi(0, 0, 16, 16))


class TracingFake(FakeCameraDriver):
    """A fake that notes how many threads are inside it, and that can block `configure`."""

    def __init__(self, clock: Any, **options: Any) -> None:
        super().__init__(clock, **options)
        self._trace_lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.block: threading.Event | None = None
        self.entered = threading.Event()

    @contextlib.contextmanager
    def _trace(self) -> Iterator[None]:
        with self._trace_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            yield
        finally:
            with self._trace_lock:
                self.active -= 1

    def configure(self, config: StreamConfig) -> ActiveStream:
        with self._trace():
            if self.block is not None:
                self.entered.set()
                self.block.wait(30.0)
            return super().configure(config)

    def read_frame(self, timeout_s: float) -> Frame:
        with self._trace():
            return super().read_frame(timeout_s)

    def move_roi(self, x: int, y: int) -> Roi:
        with self._trace():
            return super().move_roi(x, y)

    def read_temperature_c(self) -> float | None:
        with self._trace():
            return super().read_temperature_c()


class PassiveRecoverFake(TracingFake):
    """A fake whose recovery does not stop the stream, as `RESTART_CAPTURE` does not."""

    def recover(self, level: RecoveryLevel) -> None:
        self.calls.append(("recover", level))


class ZeroArrivalFake(TracingFake):
    """A fake that leaves `t_arrival_ns` at zero, so that `acquire` has to stamp the arrival."""

    def read_frame(self, timeout_s: float) -> Frame:
        return replace(super().read_frame(timeout_s), t_arrival_ns=0)


class ParkingFake(TracingFake):
    """A fake with a script: stall before one frame, and park after a number of frames.

    `stall_at` is the number of the frame that comes late, `stall_periods` is how many frame
    periods it is late, and `park_after` is the number of frames after which `read_frame` waits
    until `release` is set. On a `VirtualClock`, a parked fake keeps the capture thread idle, so
    the test sees a fixed number of frames and no more.
    """

    def __init__(
        self,
        clock: Any,
        *,
        stall_at: int | None = None,
        stall_periods: int = 0,
        park_after: int | None = None,
        **options: Any,
    ) -> None:
        super().__init__(clock, **options)
        self.stall_at = stall_at
        self.stall_periods = stall_periods
        self.park_after = park_after
        self.frames_read = 0
        self.parked = threading.Event()
        self.release = threading.Event()

    def read_frame(self, timeout_s: float) -> Frame:
        if self.park_after is not None and self.frames_read >= self.park_after:
            self.parked.set()
            self.release.wait(120.0)
            raise CameraStateError("the fake is parked")
        if self._active is not None and self.frames_read == self.stall_at:
            period_s = self._active.frame_period_s or 0.01
            self._clock.sleep(self.stall_periods * period_s)
        frame = super().read_frame(timeout_s)
        self.frames_read += 1
        return frame


class OpenCountingFake(TracingFake):
    """A fake that counts how often `open` reaches the driver."""

    def __init__(self, clock: Any, **options: Any) -> None:
        super().__init__(clock, **options)
        self.opens = 0

    def open(self) -> CameraInfo:
        self.opens += 1
        return super().open()


class StatusClock(ScaledClock):
    """A scaled clock for the rig.

    Its status can be set by a test. Its `sleep` keeps a schedule: each sleep ends one period
    after the end of the previous one, not one period after the call. The fake driver sleeps once
    per frame, so frames arrive on a grid with independent jitter, as frames from a camera do.
    Chained sleeps would let the jitter accumulate, which no camera does.
    """

    def __init__(self, status: ClockStatus | None = None, **options: Any) -> None:
        super().__init__(**options)
        self.status_value = status or ClockStatus(synchronized=True, error_bound_ns=0, source="rig")
        self._pace_lock = threading.Lock()
        self._due_ns: int | None = None

    def status(self) -> ClockStatus:
        return self.status_value

    def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            return
        with self._pace_lock:
            now_ns = self.monotonic_ns()
            if self._due_ns is None or self._due_ns < now_ns - 500_000_000:
                self._due_ns = now_ns  # after a long idle, the schedule starts again
            self._due_ns += round(seconds * 1e9)
            wait_s = (self._due_ns - now_ns) / 1e9
        if wait_s > 0:
            time.sleep(wait_s / self.speed)


@dataclass
class Rig:
    """A running service with its fake, and the drivers that connect to it."""

    service: AcquireService
    fake: TracingFake
    clock: Clock
    key: ConnectionKey
    endpoint: Endpoint
    hangs: list[HangReport] = field(default_factory=list)
    fatals: list[str] = field(default_factory=list)
    notifications: list[bytes] = field(default_factory=list)
    drivers: list[RemoteCameraDriver] = field(default_factory=list)

    def driver(self, **options: Any) -> RemoteCameraDriver:
        """A new remote driver for the service. It is closed when the rig closes."""
        options.setdefault("connect_timeout_s", 10.0)
        options.setdefault("rpc_timeout_s", 20.0)
        driver = RemoteCameraDriver(self.endpoint, self.key, clock=None, **options)
        self.drivers.append(driver)
        return driver

    def close(self) -> None:
        release = getattr(self.fake, "release", None)
        if release is not None:
            release.set()  # let a parked fake go, so that the service can stop
        for driver in self.drivers:
            driver.close()
        self.service.stop()


def make_rig(
    endpoint: Endpoint,
    key: ConnectionKey,
    *,
    speed: float = 1.0,
    fake_class: type[TracingFake] = TracingFake,
    fake_options: dict[str, Any] | None = None,
    acquire: dict[str, Any] | None = None,
    services: dict[str, Any] | None = None,
    notify: bool = False,
    on_fatal: Callable[[str], None] | None = None,
    status: ClockStatus | None = None,
    priority_hook: Callable[[], str] | None = None,
    clock: Clock | None = None,
) -> Rig:
    """Start a service on `endpoint` with a fake driver. `acquire` overrides its settings."""
    clock = clock or StatusClock(
        status, start_utc_ns=DEFAULT_START_UTC_NS, origin_real_ns=time.time_ns(), speed=speed
    )
    fake = fake_class(clock, **(fake_options or {}))
    settings = ServicesConfig(
        acquire=AcquireSettings(
            **{
                "gap_factor": 1000.0,
                "raise_priority": False,
                "watchdog_tick_s": 0.05,
                "queue_depth": 64,
                **(acquire or {}),
            }
        ),
        handshake_timeout_s=3.0,
        **(services or {}),
    )
    hangs: list[HangReport] = []
    fatals: list[str] = []
    notifications: list[bytes] = []
    service = AcquireService(
        fake,
        clock,
        endpoint,
        key,
        settings,
        guard=CallWatchdog(clock, hangs.append),
        notifier=SystemdNotifier(env={}, send=notifications.append) if notify else None,
        priority_hook=priority_hook or (lambda: "test"),
        on_fatal=on_fatal or fatals.append,
    )
    started = service.start()
    return Rig(service, fake, clock, key, started, hangs, fatals, notifications)
