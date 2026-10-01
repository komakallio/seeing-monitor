"""A stand-in for the scheduler's context, for the tests of the commissioning handlers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from seeingmon.clock import VirtualClock
from seeingmon.frames import ActiveStream, Frame, Roi, StreamConfig
from seeingmon.profile import Profile, load_profile
from seeingmon.scheduler.commission import FastWindowSample
from seeingmon.scheduler.config import SchedulerConfig
from seeingmon.testing import FakeCameraDriver

FAST = StreamConfig("bin1", 2000, 0, roi=Roi(100, 100, 32, 32))


class FakeContext:
    """The scheduler's context for a handler, on a fake camera."""

    def __init__(
        self,
        clock: VirtualClock,
        *,
        stop_after: int | None = None,
        drop_at: int | None = None,
    ) -> None:
        self._clock = clock
        self._profile = load_profile("asi294mm-gs250")
        self.camera = FakeCameraDriver(clock)
        self.camera.open()
        self.stop_after = stop_after
        self.drop_at = drop_at
        self.reads = 0
        self.configs: list[StreamConfig] = []
        self.pointing: StreamConfig | None = FAST
        self.events: list[tuple[str, str]] = []

    @property
    def clock(self) -> VirtualClock:
        return self._clock

    @property
    def profile(self) -> Profile:
        return self._profile

    @property
    def config(self) -> SchedulerConfig:
        return SchedulerConfig()

    def should_stop(self) -> bool:
        return self.stop_after is not None and self.reads >= self.stop_after

    def emit_event(
        self, level: str, kind: str, message: str, detail: Mapping[str, Any] | None = None
    ) -> None:
        self.events.append((level, kind))

    def fast_stream_config(
        self,
        *,
        mode: str | None = None,
        exposure_us: int | None = None,
        gain: int | None = None,
        roi_arcmin: float | None = None,
    ) -> StreamConfig | None:
        return self.pointing

    def configure(self, config: StreamConfig) -> ActiveStream:
        self.configs.append(config)
        return self.camera.configure(config)

    def start(self) -> None:
        self.camera.start()

    def stop(self) -> None:
        self.camera.stop()

    def read_frame(self, timeout_s: float | None = None) -> Frame:
        if self.drop_at is not None and self.reads == self.drop_at:
            self.camera.drop_frames(7)
        frame = self.camera.read_frame(timeout_s or 5.0)
        self.reads += 1
        return frame

    def run_fast_window(self, config: StreamConfig, duration_s: float) -> FastWindowSample:
        raise NotImplementedError("a burst runs no fast window")
