"""The burst handler: record raw frames to a SER file with a JSON sidecar, and pin the result.

A burst answers a question that the windowed statistics cannot: what do the raw frames look like?
The scheduler runs the handler at a cycle boundary (`seeingmon.scheduler.commission`), with the
camera to itself. The handler:

1. refuses when raw capture is stopped (`capture_allowed()` of the storage is false, which means
   that the free space fell below the limit), and when the duration is out of range;
2. takes the stream settings of the command, or the fast stream with the ROI on Polaris;
3. makes the folder of the burst in the `bursts` folder of the data layout, and writes the frames
   to one SER file in it (`seeingmon.recordings.ser.SerWriter`). Each frame stores its arrival time
   in the timestamp trailer of the file;
4. writes the JSON sidecar next to the file (`seeingmon.recordings.sidecar.BurstSidecar`): the
   schema version, the stream settings, the profile ID, the time quality of the frames, and the
   frame count. The sidecar holds no host name, path, or serial number;
5. pins the burst, so that retention never deletes it.

The handler stops early, with the status `aborted`, when a preempting command arrives
(`context.should_stop()`) or when raw capture stops in the middle of the burst. It keeps what it
wrote in both cases. A camera error propagates to the scheduler after the handler closed the file,
because the scheduler owns the recovery.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from pathlib import Path

from seeingmon.clock import NS_PER_S, Clock
from seeingmon.drivers.base import CameraConfigError
from seeingmon.frames import ActiveStream, Frame, StreamConfig, TimeQuality
from seeingmon.profile import Profile
from seeingmon.recordings.ser import HEADER_SIZE, SerError, SerWriter
from seeingmon.recordings.sidecar import (
    BurstSidecar,
    SidecarError,
    burst_sidecar_path,
    write_burst_sidecar,
)
from seeingmon.scheduler.commands import QueueBurst
from seeingmon.scheduler.commission import (
    CommissionContext,
    CommissionResult,
    CommissionTask,
)
from seeingmon.store.layout import DataLayout

_log = logging.getLogger(__name__)


def stream_summary(config: StreamConfig) -> dict[str, object]:
    """The stream settings as a JSON-able detail for the result."""
    roi = config.roi
    return {
        "mode": config.mode,
        "exposure_us": config.exposure_us,
        "gain": config.gain,
        "roi": None if roi is None else [roi.x, roi.y, roi.width, roi.height],
    }


class BurstHandler:
    """Runs `burst` tasks. Register it with `Scheduler.register_handler("burst", handler)`."""

    def __init__(
        self,
        *,
        layout: DataLayout,
        profile: Profile,
        clock: Clock,
        capture_allowed: Callable[[], bool],
        max_duration_s: float = 600.0,
    ) -> None:
        self._layout = layout
        self._profile = profile
        self._clock = clock
        self._capture_allowed = capture_allowed
        self._max_duration_s = max_duration_s

    def _failed(self, task: CommissionTask, started_ns: int, summary: str) -> CommissionResult:
        return CommissionResult(
            task_id=task.task_id,
            kind=task.kind,
            status="failed",
            summary=summary,
            started_utc_ns=started_ns,
            finished_utc_ns=self._clock.utc_ns(),
            pinned=False,
        )

    def run(self, task: CommissionTask, context: CommissionContext) -> CommissionResult:
        started_ns = self._clock.utc_ns()
        command = task.command
        if not isinstance(command, QueueBurst):
            return self._failed(task, started_ns, "the task holds no burst command")
        if not 0 < command.duration_s <= self._max_duration_s:
            return self._failed(
                task,
                started_ns,
                f"a burst lasts more than 0 and at most {self._max_duration_s:g} s",
            )
        if not self._capture_allowed():
            return self._failed(
                task, started_ns, "raw capture is stopped because the free space is low"
            )
        stream = command.stream or context.fast_stream_config()
        if stream is None:
            return self._failed(
                task,
                started_ns,
                "there is no pointing solution yet, so the ROI cannot center on Polaris",
            )
        try:
            active = context.configure(stream)
        except (CameraConfigError, ValueError) as error:
            return self._failed(task, started_ns, f"the camera refused the stream: {error}")
        context.start()
        burst = self._layout.new_burst(started_ns, command.label or None)
        ser_path = burst / f"{burst.name}.ser"
        try:
            outcome = self._record(command, context, active, ser_path)
        except BaseException:
            self._discard_if_empty(burst, ser_path)
            raise
        frames, dropped, quality, first, stopped_for = outcome
        if frames == 0:
            self._discard_if_empty(burst, ser_path)
            return self._failed(
                task, started_ns, f"the burst recorded no frame ({stopped_for or 'no frames came'})"
            )
        self._write_sidecar(ser_path, active, frames, quality, first)
        self._layout.pin_burst(burst)
        finished_ns = self._clock.utc_ns()
        aborted = stopped_for is not None
        summary = f"Recorded {frames} frames, {dropped} dropped, in {burst.name}."
        if aborted:
            summary = f"Stopped early ({stopped_for}). {summary}"
        return CommissionResult(
            task_id=task.task_id,
            kind=task.kind,
            status="aborted" if aborted else "ok",
            summary=summary,
            started_utc_ns=started_ns,
            finished_utc_ns=finished_ns,
            data={
                "frames": frames,
                "dropped": dropped,
                "duration_s": (finished_ns - started_ns) / NS_PER_S,
                "stream": stream_summary(active.config),
                "time_quality": quality.name,
                "burst": burst.name,
            },
            artifacts=(
                self._layout.relative(ser_path),
                self._layout.relative(burst_sidecar_path(ser_path)),
            ),
            pinned=True,
        )

    def _record(
        self, command: QueueBurst, context: CommissionContext, active: ActiveStream, ser_path: Path
    ) -> tuple[int, int, TimeQuality, Frame | None, str | None]:
        """Write frames until the time is up. Returns the counts and why it stopped early."""
        height, width = active.frame_shape
        depth = active.config.pixel_format.value
        deadline_ns = self._clock.monotonic_ns() + round(command.duration_s * NS_PER_S)
        frames = dropped = 0
        quality = TimeQuality.EXACT
        first: Frame | None = None
        stopped_for: str | None = None
        writer = SerWriter(ser_path, width=width, height=height, pixel_depth=depth)
        try:
            while self._clock.monotonic_ns() < deadline_ns:
                if context.should_stop():
                    stopped_for = "another command took the camera"
                    break
                if not self._capture_allowed():
                    stopped_for = "the free space fell below the limit"
                    break
                frame = context.read_frame()
                writer.write_frame(frame.data, frame.t_arrival_ns)
                first = first or frame
                frames += 1
                dropped += frame.dropped_before
                quality = min(quality, frame.t_quality)
        except OSError as error:  # a full disk: keep what the file holds
            _log.warning("the burst stopped on a write error: %s", type(error).__name__)
            stopped_for = "the disk refused a write"
        finally:
            try:
                writer.close()
            except (OSError, SerError):
                _log.warning("the burst file could not be closed cleanly")
        return frames, dropped, quality, first, stopped_for

    def _write_sidecar(
        self,
        ser_path: Path,
        active: ActiveStream,
        frames: int,
        quality: TimeQuality,
        first: Frame | None,
    ) -> None:
        try:
            write_burst_sidecar(
                burst_sidecar_path(ser_path),
                BurstSidecar(
                    profile_id=self._profile.id,
                    stream=active.config,
                    adc_bits=active.adc_bits,
                    time_quality=quality,
                    frame_period_s=active.frame_period_s,
                    frame_count=frames,
                    start_utc_ns=None if first is None else first.t_utc_ns,
                    temperature_c=None if first is None else first.temperature_c,
                ),
            )
        except (OSError, SidecarError):
            _log.warning("the sidecar of the burst could not be written", exc_info=True)

    @staticmethod
    def _discard_if_empty(burst: Path, ser_path: Path) -> None:
        """Remove the folder of a burst that holds no frames, so that no empty burst lingers."""
        try:
            if (
                not ser_path.exists() or ser_path.stat().st_size <= HEADER_SIZE
            ):  # a header and no frame
                shutil.rmtree(burst, ignore_errors=True)
        except OSError:
            _log.debug("could not inspect the empty burst", exc_info=True)
