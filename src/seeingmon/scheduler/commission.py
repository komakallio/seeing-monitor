"""Commissioning: the task queue, the handler protocol, and the sweep.

A burst, a sweep, or a replay is a task in a priority queue. The scheduler runs the tasks at the
next cycle boundary, one handler call for each, and it holds the camera for the whole call. A
handler is any object with a `run(task, context)` method (`CommissionHandler`). The scheduler
owns the sweep handler (`SweepHandler`), because a sweep is a scheduling job: it runs a short
fast window for each setting. The services lane registers the burst handler, which records raw
frames to a SER file, and the replay handler. Register a handler with
`Scheduler.register_handler(kind, handler)`.

**Context.** A handler gets a `CommissionContext` with exclusive use of the camera. Change the
stream through `context.configure`, never through the driver, so that the fast analyzer sees every
reconfiguration. A long handler must check `context.should_stop()` between frames: a command that
preempts commissioning (an alignment or a pause) or a shutdown sets it, and the handler then
returns early with the status `aborted`.

**Results.** Every task ends in a `CommissionResult` with `pinned` set, which exempts what the
task stored from retention. The scheduler writes an event with the result and keeps the latest
results in memory (`Scheduler.results`). A sweep's windows stay out of the `seeing_window` series,
so a window that was taken for a test can never pass for seeing data. They live in the result.

**Sweep.** For each cell of a grid of readout mode, ROI size, gain, and exposure, the sweep runs
a fast window and reports: the share of frames that saturate, the star's peak against the
saturation level, the background, the signal-to-noise ratio of the peak over the noise of the ROI
border, the frame and drop rates, and the noise of the estimator from the analyzer's own window
records. A cell that the camera rejects, or that has no pointing solution, reports why and does
not stop the sweep.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from seeingmon.clock import Clock
from seeingmon.drivers.base import CameraConfigError
from seeingmon.frames import ActiveStream, Frame, StreamConfig
from seeingmon.profile import Profile
from seeingmon.records import SeeingWindowRecord
from seeingmon.scheduler.commands import Command, QueueSweep
from seeingmon.scheduler.config import SchedulerConfig, SweepConfig

TaskStatus = Literal["ok", "failed", "aborted"]

# A frame saturates when its brightest pixel reaches this share of the saturation level, as the
# architecture flags it.
SATURATED_PEAK_FRACTION = 0.98
_MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True, slots=True)
class CommissionTask:
    """A queued task. `command` is the command that created it, with the settings of the task."""

    task_id: int
    kind: str
    command: Command
    submitted_utc_ns: int
    priority: int = 0


@dataclass(frozen=True, slots=True)
class CommissionResult:
    """What a handler reports. `pinned` exempts what the task stored from retention.

    `data` holds the JSON-compatible detail, such as the table of a sweep. `artifacts` names the
    files that the task wrote, as paths relative to the data directory.
    """

    task_id: int
    kind: str
    status: TaskStatus
    summary: str
    started_utc_ns: int
    finished_utc_ns: int
    data: Mapping[str, Any] = field(default_factory=dict)
    artifacts: tuple[str, ...] = ()
    pinned: bool = True

    def to_detail(self) -> dict[str, Any]:
        """The result as the `detail` object of an event record."""
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "status": self.status,
            "summary": self.summary,
            "started_utc_ns": self.started_utc_ns,
            "finished_utc_ns": self.finished_utc_ns,
            "pinned": self.pinned,
            "artifacts": list(self.artifacts),
            "data": dict(self.data),
        }


class TaskQueue:
    """A priority queue of tasks. A higher priority runs first, and equal priorities keep order.

    The queue is not thread-safe. The scheduler guards it with its lock.
    """

    def __init__(self, max_queued: int) -> None:
        self._max_queued = max_queued
        self._heap: list[tuple[int, int, CommissionTask]] = []

    def __len__(self) -> int:
        return len(self._heap)

    @property
    def full(self) -> bool:
        return len(self._heap) >= self._max_queued

    def push(self, task: CommissionTask) -> bool:
        """Add a task. Returns `False` without adding it when the queue is full."""
        if self.full:
            return False
        heapq.heappush(self._heap, (-task.priority, task.task_id, task))
        return True

    def pop(self) -> CommissionTask | None:
        """Remove and return the next task, or `None` when the queue is empty."""
        if not self._heap:
            return None
        return heapq.heappop(self._heap)[2]

    def tasks(self) -> tuple[CommissionTask, ...]:
        """The waiting tasks in the order that they will run."""
        return tuple(entry[2] for entry in sorted(self._heap))

    def has(self, predicate: Callable[[CommissionTask], bool]) -> bool:
        """Whether a waiting task satisfies `predicate`. It does not sort, so asking is cheap."""
        return any(predicate(entry[2]) for entry in self._heap)

    def remove(self, predicate: Callable[[CommissionTask], bool]) -> tuple[CommissionTask, ...]:
        """Remove the waiting tasks that satisfy `predicate`, in the order that they would run."""
        gone = sorted(entry for entry in self._heap if predicate(entry[2]))
        if gone:
            self._heap = [entry for entry in self._heap if not predicate(entry[2])]
            heapq.heapify(self._heap)
        return tuple(entry[2] for entry in gone)


# --- Frame statistics ----------------------------------------------------------------------------


def border_pixels(data: npt.NDArray[np.uint8] | npt.NDArray[np.uint16]) -> npt.NDArray[np.float64]:
    """The pixels of the outer ring of a frame, which hold the background and no star."""
    height, width = data.shape
    ring = max(1, min(height, width) // 16)
    if height <= 2 * ring or width <= 2 * ring:
        return data.astype(np.float64).ravel()
    parts = (
        data[:ring].ravel(),
        data[-ring:].ravel(),
        data[ring:-ring, :ring].ravel(),
        data[ring:-ring, -ring:].ravel(),
    )
    return np.concatenate(parts).astype(np.float64)


@dataclass(frozen=True, slots=True)
class FrameStatsSummary:
    """Statistics of the frames in a window. A value is `None` when no frame supports it."""

    n_frames: int
    saturated_fraction: float | None
    peak_fraction_mean: float | None
    peak_fraction_max: float | None
    background_fraction: float | None
    snr_median: float | None
    star_found_fraction: float | None


class FrameStatsAccumulator:
    """Collect per-frame statistics for the sweep: peak, background, and signal-to-noise ratio."""

    def __init__(self) -> None:
        self._peaks: list[float] = []
        self._backgrounds: list[float] = []
        self._snr: list[float] = []
        self._saturated = 0
        self._found = 0

    def add(self, frame: Frame, saturation_dn: float, *, star_found: bool) -> None:
        """Add one frame. `saturation_dn` is the saturation level in the counts of the frame."""
        data = frame.data
        peak = float(data.max())
        border = border_pixels(data)
        background = float(np.median(border))
        sigma = _MAD_TO_SIGMA * float(np.median(np.abs(border - background)))
        self._peaks.append(peak / saturation_dn)
        self._backgrounds.append(background / saturation_dn)
        if peak >= SATURATED_PEAK_FRACTION * saturation_dn:
            self._saturated += 1
        if sigma > 0.0:
            self._snr.append((peak - background) / sigma)
        if star_found:
            self._found += 1

    def summary(self) -> FrameStatsSummary:
        count = len(self._peaks)
        if count == 0:
            return FrameStatsSummary(0, None, None, None, None, None, None)
        return FrameStatsSummary(
            n_frames=count,
            saturated_fraction=self._saturated / count,
            peak_fraction_mean=float(np.mean(self._peaks)),
            peak_fraction_max=float(np.max(self._peaks)),
            background_fraction=float(np.median(self._backgrounds)),
            snr_median=float(np.median(self._snr)) if self._snr else None,
            star_found_fraction=self._found / count,
        )


@dataclass(frozen=True, slots=True)
class FastWindowSample:
    """What `CommissionContext.run_fast_window` measured.

    `windows` are the windows that the fast analyzer closed. The scheduler did not write them to
    the store. `duration_s` is the time from the first frame to the last, plus one frame interval.
    """

    config: StreamConfig
    stream_id: int
    duration_s: float
    n_frames: int
    n_dropped: int
    windows: tuple[SeeingWindowRecord, ...]
    stats: FrameStatsSummary
    aborted: bool = False

    @property
    def frame_rate_hz(self) -> float | None:
        return self.n_frames / self.duration_s if self.n_frames and self.duration_s > 0 else None

    @property
    def drop_rate(self) -> float | None:
        expected = self.n_frames + self.n_dropped
        return self.n_dropped / expected if expected else None


# --- The protocols -------------------------------------------------------------------------


@runtime_checkable
class CommissionContext(Protocol):
    """What the scheduler gives a handler. The handler runs on the scheduler's thread."""

    @property
    def clock(self) -> Clock: ...

    @property
    def profile(self) -> Profile: ...

    @property
    def config(self) -> SchedulerConfig: ...

    def should_stop(self) -> bool:
        """Whether the handler must end early: a preempting command or a shutdown arrived."""
        ...

    def emit_event(
        self, level: str, kind: str, message: str, detail: Mapping[str, Any] | None = None
    ) -> None:
        """Write an event record. `kind` is a dotted code such as `scheduler.burst_progress`."""
        ...

    def fast_stream_config(
        self,
        *,
        mode: str | None = None,
        exposure_us: int | None = None,
        gain: int | None = None,
        roi_arcmin: float | None = None,
    ) -> StreamConfig | None:
        """The fast stream settings with the ROI centered on Polaris, or `None` without a solution.

        A missing argument takes the configured fast value. The ROI sits at the position that
        the pointing provider predicts for the readout mode.
        """
        ...

    def configure(self, config: StreamConfig) -> ActiveStream:
        """Reconfigure the camera and tell the fast analyzer. Use this, never the driver."""
        ...

    def start(self) -> None:
        """Start the configured stream."""
        ...

    def read_frame(self, timeout_s: float | None = None) -> Frame:
        """Read the next frame. The default timeout follows the stream's frame period.

        A camera error propagates to the caller. Let it go: the scheduler runs its recovery.
        """
        ...

    def stop(self) -> None:
        """Stop the stream."""
        ...

    def run_fast_window(self, config: StreamConfig, duration_s: float) -> FastWindowSample:
        """Run the fast analysis on a stream for `duration_s` seconds, and return what it saw."""
        ...


@runtime_checkable
class CommissionHandler(Protocol):
    """Runs one kind of task. Register it with `Scheduler.register_handler`."""

    def run(self, task: CommissionTask, context: CommissionContext) -> CommissionResult: ...


# --- The sweep -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SweepCell:
    """One combination of settings."""

    mode: str
    exposure_us: int
    gain: int
    roi_arcmin: float


@dataclass(frozen=True, slots=True)
class SweepPlan:
    """The cells of a sweep and the length of each cell's window."""

    cells: tuple[SweepCell, ...]
    window_s: float

    @classmethod
    def resolve(cls, command: QueueSweep, sweep: SweepConfig, profile: Profile) -> SweepPlan:
        """Fill the empty axes of a command from the configuration and check every value.

        Raises `ValueError` with a plain message when a value is out of range for the profile,
        or when the grid has more cells than `max_cells`.
        """
        exposures = command.exposure_us or sweep.exposure_us
        gains = command.gain or sweep.gain
        rois = command.roi_arcmin or sweep.roi_arcmin
        modes = command.modes or sweep.modes or (profile.fast_mode.mode,)
        window_s = sweep.window_s if command.window_s is None else command.window_s
        if not (math.isfinite(window_s) and window_s > 0):
            raise ValueError("window_s must be a positive number of seconds")
        low_us, high_us = profile.limits.exposure_us_range
        low_gain, high_gain = profile.limits.gain_range
        for exposure in exposures:
            if not low_us <= exposure <= high_us:
                raise ValueError(f"exposure {exposure} us is outside {low_us} to {high_us} us")
        for gain in gains:
            if not low_gain <= gain <= high_gain:
                raise ValueError(f"gain {gain} is outside {low_gain} to {high_gain}")
        for arcmin in rois:
            if not (math.isfinite(arcmin) and arcmin > 0):
                raise ValueError("every ROI size must be a positive angle in arcminutes")
        known = {mode.name for mode in profile.readout_modes}
        for mode in modes:
            if mode not in known:
                raise ValueError(f"unknown readout mode {mode!r}; the profile has {sorted(known)}")
        count = len(modes) * len(rois) * len(gains) * len(exposures)
        if count > sweep.max_cells:
            raise ValueError(f"the grid has {count} cells, and the limit is {sweep.max_cells}")
        cells = tuple(
            SweepCell(mode, exposure, gain, arcmin)
            for mode, arcmin, gain, exposure in itertools.product(modes, rois, gains, exposures)
        )
        return cls(cells=cells, window_s=window_s)


@dataclass(frozen=True, slots=True)
class SweepCellResult:
    """What one cell measured. A value is `None` when the cell could not support it.

    `status` is `ok`, `skipped` (no pointing solution), or `failed` (the camera rejected the
    settings). `note` says why when the status is not `ok`.
    """

    cell: SweepCell
    status: str
    note: str | None = None
    roi_px: tuple[int, int] | None = None
    n_frames: int = 0
    n_dropped: int = 0
    frame_rate_hz: float | None = None
    drop_rate: float | None = None
    saturated_fraction: float | None = None
    peak_fraction_mean: float | None = None
    background_fraction: float | None = None
    snr_median: float | None = None
    star_found_fraction: float | None = None
    n_windows: int = 0
    centroid_noise_px: float | None = None
    image_motion_rms_arcsec: float | None = None
    seeing_fwhm_arcsec: float | None = None


def _mean_of(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(present)) if present else None


def summarize_cell(cell: SweepCell, sample: FastWindowSample) -> SweepCellResult:
    """Combine the frame statistics and the analyzer's window records into one cell result."""
    windows = sample.windows
    motion = [
        value
        for window in windows
        for value in (window.image_motion_rms_x_arcsec, window.image_motion_rms_y_arcsec)
    ]
    roi = sample.config.roi
    return SweepCellResult(
        cell=cell,
        status="ok",
        roi_px=None if roi is None else (roi.width, roi.height),
        n_frames=sample.n_frames,
        n_dropped=sample.n_dropped,
        frame_rate_hz=sample.frame_rate_hz,
        drop_rate=sample.drop_rate,
        saturated_fraction=sample.stats.saturated_fraction,
        peak_fraction_mean=sample.stats.peak_fraction_mean,
        background_fraction=sample.stats.background_fraction,
        snr_median=sample.stats.snr_median,
        star_found_fraction=sample.stats.star_found_fraction,
        n_windows=len(windows),
        centroid_noise_px=_mean_of([window.centroid_noise_px for window in windows]),
        image_motion_rms_arcsec=_mean_of(motion),
        seeing_fwhm_arcsec=_mean_of([window.seeing_fwhm_arcsec for window in windows]),
    )


class SweepHandler:
    """Run a sweep: a short fast window for each cell of the grid.

    The scheduler registers one for the `sweep` kind. A cell that the camera rejects
    (`CameraConfigError`) or that has no pointing solution is reported and skipped. Any other
    camera error propagates, and the scheduler runs its recovery.
    """

    def sweep(
        self, task: CommissionTask, context: CommissionContext
    ) -> tuple[tuple[SweepCellResult, ...], SweepPlan, bool]:
        """Run the cells. Returns the results, the plan, and whether the sweep was aborted."""
        command = task.command
        if not isinstance(command, QueueSweep):
            raise TypeError("the sweep handler takes a QueueSweep command")
        plan = SweepPlan.resolve(command, context.config.sweep, context.profile)
        results: list[SweepCellResult] = []
        aborted = False
        for cell in plan.cells:
            if context.should_stop():
                aborted = True
                break
            results.append(self._run_cell(cell, plan, context))
        return tuple(results), plan, aborted

    def run(self, task: CommissionTask, context: CommissionContext) -> CommissionResult:
        started = context.clock.utc_ns()
        results, plan, aborted = self.sweep(task, context)
        done = sum(1 for result in results if result.status == "ok")
        status: TaskStatus = "aborted" if aborted else "ok"
        summary = f"{done} of {len(plan.cells)} cells measured"
        if aborted:
            summary += ", then the sweep stopped early"
        return CommissionResult(
            task_id=task.task_id,
            kind=task.kind,
            status=status,
            summary=summary,
            started_utc_ns=started,
            finished_utc_ns=context.clock.utc_ns(),
            data={
                "window_s": plan.window_s,
                "cells": [_cell_data(result) for result in results],
            },
        )

    @staticmethod
    def _run_cell(cell: SweepCell, plan: SweepPlan, context: CommissionContext) -> SweepCellResult:
        config = context.fast_stream_config(
            mode=cell.mode,
            exposure_us=cell.exposure_us,
            gain=cell.gain,
            roi_arcmin=cell.roi_arcmin,
        )
        if config is None:
            return SweepCellResult(
                cell=cell, status="skipped", note="no pointing solution for this readout mode"
            )
        try:
            sample = context.run_fast_window(config, plan.window_s)
        except CameraConfigError as error:
            return SweepCellResult(cell=cell, status="failed", note=f"camera rejected: {error}")
        return summarize_cell(cell, sample)


def _cell_data(result: SweepCellResult) -> dict[str, Any]:
    """A cell result as JSON values: tuples become lists, and no value is NaN or infinite."""
    data = asdict(result)
    data["roi_px"] = list(result.roi_px) if result.roi_px is not None else None
    for key, value in data.items():
        if isinstance(value, float) and not math.isfinite(value):
            data[key] = None
    return data


def format_sweep_table(results: Sequence[SweepCellResult]) -> str:
    """Render the cell results as a plain-text table, for `seeingmon sweep` and for logs."""

    def percent(value: float | None) -> str:
        return "-" if value is None else f"{100.0 * value:.1f}"

    def number(value: float | None, digits: int = 1) -> str:
        return "-" if value is None else f"{value:.{digits}f}"

    header = (
        f"{'mode':<6} {'exp_us':>8} {'gain':>5} {'roi_px':>9} {'fps':>7} {'drop%':>6} "
        f"{'sat%':>6} {'peak%':>6} {'bg%':>6} {'snr':>7} {'noise_px':>9}  note"
    )
    lines = [header]
    for result in results:
        cell = result.cell
        roi = "-" if result.roi_px is None else f"{result.roi_px[0]}x{result.roi_px[1]}"
        lines.append(
            f"{cell.mode:<6} {cell.exposure_us:>8} {cell.gain:>5} {roi:>9} "
            f"{number(result.frame_rate_hz):>7} {percent(result.drop_rate):>6} "
            f"{percent(result.saturated_fraction):>6} {percent(result.peak_fraction_mean):>6} "
            f"{percent(result.background_fraction):>6} {number(result.snr_median):>7} "
            f"{number(result.centroid_noise_px, 3):>9}  {result.note or ''}".rstrip()
        )
    return "\n".join(lines)
