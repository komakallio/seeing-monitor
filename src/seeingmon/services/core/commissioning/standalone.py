"""Run one commissioning task without `core`, against the configured driver. For bench work.

`seeingmon burst --standalone` (and `sweep`, and `replay`) build a small scheduler, queue the one
task, and step the scheduler until the task finishes. The scheduler is the production one, so the
task runs the same code as in `core`: the same handlers, the same stream rules, and the same fast
analysis. What the standalone mode leaves out is everything around it:

- **No `acquire`.** The command opens the driver of `[services.acquire] driver` in its own process,
  so no other process may hold the camera: stop `acquire` first. Nothing guards the driver calls
  against a hang, as `acquire` does.
- **No store.** The scheduler's events go to the log. A burst still writes its SER file and sidecar
  into the `bursts` folder of the data layout (`[paths] data_dir`) and pins it, and a replay still
  writes its separate store under the data directory.
- **No survey analysis.** There is no pointing solution, so the pointing provider reports the
  center of the sensor, and you aim the camera at Polaris, which sits in the middle of the frame.
  A burst without stream settings records a fast stream with the ROI at the sensor center.
- **A replay needs no camera.** The scheduler opens a fake camera that does nothing, because the
  replay handler reads the recording itself.

The scheduler in `safe` runs a queued task at its next boundary, so a task starts within moments.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy.typing as npt

from seeingmon.analysis import SurveyOutput
from seeingmon.clock import NS_PER_S, Clock
from seeingmon.config import Config
from seeingmon.drivers.base import CameraDriver
from seeingmon.fastpath import FastPathConfig, create_fast_analyzer
from seeingmon.frames import Frame
from seeingmon.profile import Profile
from seeingmon.records import EventRecord, Record
from seeingmon.scheduler import CommissionResult, build_scheduler
from seeingmon.scheduler.commands import Command, CommandResult, QueueReplay
from seeingmon.services.config import ServicesConfig
from seeingmon.services.core.commissioning.burst import BurstHandler
from seeingmon.services.core.commissioning.replay import ReplayHandler
from seeingmon.services.core.settings import ReplaySettings
from seeingmon.store.config import StoreConfig
from seeingmon.store.layout import DataLayout

_log = logging.getLogger(__name__)

STEP_LIMIT = 10_000_000


class NullSurvey:
    """A survey analysis that does nothing: no frame is analyzed and no result ever arrives."""

    def submit(self, frame: Frame) -> None:
        return None

    def poll(self) -> tuple[SurveyOutput, ...]:
        return ()

    def pending(self) -> int:
        return 0


class CenterPointing:
    """A pointing provider that puts Polaris at the center of the sensor in every readout mode."""

    def __init__(self, profile: Profile) -> None:
        self._profile = profile

    def polaris_position(self, t_utc_ns: int, mode: str) -> tuple[float, float] | None:
        readout = self._profile.mode(mode)
        return (readout.width_px - 1) / 2.0, (readout.height_px - 1) / 2.0


class LogWriter:
    """A record writer that logs the events of the scheduler and drops everything else."""

    def __init__(self, show: Callable[[str], None]) -> None:
        self._show = show
        self.events: list[EventRecord] = []

    def write(self, record: Record) -> None:
        if isinstance(record, EventRecord):
            self.events.append(record)
            self._show(f"{record.level}: {record.message}")


class NullMetrics:
    """A metrics writer that drops the per-frame rows."""

    def write_metrics(self, stream_id: int, rows: npt.NDArray[Any]) -> None:
        return None


@dataclass(frozen=True, slots=True)
class StandaloneOutcome:
    """What a standalone run produced: the answer to the command, and the result if it ran."""

    answer: CommandResult
    result: CommissionResult | None


def run_standalone(
    config: Config,
    services: ServicesConfig,
    command: Command,
    *,
    clock: Clock,
    driver: CameraDriver | None = None,
    show: Callable[[str], None] = print,
    timeout_s: float = 3600.0,
    should_stop: Callable[[], bool] = lambda: False,
) -> StandaloneOutcome:
    """Queue `command` in a private scheduler and run it. `timeout_s` counts in clock seconds.

    Pass `driver` to use a driver that you built, such as the simulator of a test. Otherwise the
    function builds the driver of `[services.acquire]`, or a fake camera for a replay.
    """
    profile = config.profile
    station_id = config.station_id
    layout = DataLayout.from_config(config)
    layout.create()
    fast_config = config.section("fastpath", FastPathConfig)
    replay = config.section("replay", ReplaySettings)
    commissioning = services.core.commissioning
    camera = driver or _build_driver(config, services, clock, command)
    writer = LogWriter(show)
    scheduler = build_scheduler(
        config,
        driver=camera,
        fast=create_fast_analyzer(profile, fast_config, station_id),
        survey=NullSurvey(),
        pointing=CenterPointing(profile),
        records=writer,
        metrics=NullMetrics(),
        clock=clock,
    )
    scheduler.register_handler(
        "burst",
        BurstHandler(
            layout=layout,
            profile=profile,
            clock=clock,
            capture_allowed=lambda: True,
            max_duration_s=commissioning.burst_max_duration_s,
        ),
    )
    scheduler.register_handler(
        "replay",
        ReplayHandler(
            layout=layout,
            profile=profile,
            station_id=station_id,
            clock=clock,
            fast_config=fast_config,
            store_config=config.section("store", StoreConfig),
            recordings_dir=Path(replay.recordings_dir) if replay.recordings_dir else None,
            replays_dir=commissioning.replays_dir,
        ),
    )
    answer = scheduler.submit(command)
    if not answer.accepted or answer.task_id is None:
        scheduler.close()
        return StandaloneOutcome(answer, None)
    deadline_ns = clock.monotonic_ns() + round(timeout_s * NS_PER_S)
    result: CommissionResult | None = None
    try:
        for _ in range(STEP_LIMIT):
            scheduler.step()
            result = next((r for r in scheduler.results() if r.task_id == answer.task_id), None)
            if result is not None or should_stop() or clock.monotonic_ns() >= deadline_ns:
                break
    finally:
        scheduler.close()
    return StandaloneOutcome(answer, result)


def _build_driver(
    config: Config, services: ServicesConfig, clock: Clock, command: Command
) -> CameraDriver:
    if isinstance(command, QueueReplay):
        from seeingmon.testing import FakeCameraDriver

        return FakeCameraDriver(clock)  # the replay handler reads the recording itself
    from seeingmon.services.acquire.factory import create_camera_driver

    return create_camera_driver(
        services.acquire.driver,
        profile=config.profile,
        clock=clock,
        options=services.acquire.driver_options,
    )
