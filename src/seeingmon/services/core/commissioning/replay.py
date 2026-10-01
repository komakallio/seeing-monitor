"""The replay handler: feed a recording through the production analysis into a separate store.

A replay shows what the system would have measured on a recording, such as a burst from the
camera or a capture of another program. The handler runs the same code as a live night: the
`replay` driver (`seeingmon.drivers.replay`) delivers the frames of a SER file, and a fast
analyzer built from the same configuration turns them into windows and per-frame metrics.

**A separate store.** The results go to their own store under the data directory, in
`<replays_dir>/<time>-<name>/` with the layout of the data directory (`db/results.sqlite` and
`segments/`). The main store never sees a replayed window, so a replay cannot pass for a night of
seeing data, and the sinks never forward one. The scheduler writes the usual event for the task.

**The source.** The command names a recording without a directory part. The handler looks for it in
the recordings folder of `[replay] recordings_dir`, with or without the `.ser` suffix. A name
that matches a burst folder under the data directory also works (the SER file inside it). No other
path is read, so a command can never make `core` open an arbitrary file, and no message holds a
path.

**The speed.** `speed` is a factor of the recorded rate (1 is the original pace, 0 is as fast as
possible). The handler paces through the clock of the scheduler, and it checks `should_stop()` for
every frame, so a preempting command ends the replay early with the status `aborted`.

**Options.** The `options` of the command may hold `start_frame`, `max_frames`, `mode`,
`exposure_us`, `gain`, and `adc_bits`, which the replay driver understands. The options `path` and
`sidecar` are refused.
"""

from __future__ import annotations

import itertools
import logging
import math
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from seeingmon.clock import NS_PER_S, Clock, utc_ns_to_datetime
from seeingmon.drivers.base import CameraConfigError, CameraError
from seeingmon.drivers.replay import ReplayFinishedError
from seeingmon.frames import StreamConfig, StreamKind
from seeingmon.profile import Profile
from seeingmon.records import SeeingWindowRecord
from seeingmon.scheduler.commands import QueueReplay
from seeingmon.scheduler.commission import (
    CommissionContext,
    CommissionResult,
    CommissionTask,
)
from seeingmon.store.config import StoreConfig
from seeingmon.store.db import Store
from seeingmon.store.layout import DataLayout
from seeingmon.store.segments import SegmentWriter

_log = logging.getLogger(__name__)

ALLOWED_OPTIONS = frozenset(
    {"start_frame", "max_frames", "mode", "exposure_us", "gain", "adc_bits"}
)
METRICS_EVERY_FRAMES = 256
SER_SUFFIX = ".ser"


def resolve_source(
    name: str, recordings_dir: Path | None, layout: DataLayout
) -> tuple[Path | None, str]:
    """The SER file that `name` selects, or `None` with the reason.

    `name` has no directory part. The recordings folder wins over a burst of the same name.
    """
    if not name or Path(name).name != name or name in (".", "..") or "\\" in name or ":" in name:
        return None, "the source must be a recording name without a directory part"
    candidates: list[Path] = []
    if recordings_dir is not None:
        candidates += [recordings_dir / name]
        if not name.lower().endswith(SER_SUFFIX):
            candidates.append(recordings_dir / f"{name}{SER_SUFFIX}")
    burst = layout.bursts_dir / name
    if burst.is_dir():
        candidates += sorted(burst.glob(f"*{SER_SUFFIX}"))
    for path in candidates:
        if path.is_file():
            return path, ""
    return None, "no recording has that name"


def clean_options(options: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """The options for the replay driver, or an empty mapping with the reason for a refusal."""
    unknown = sorted(set(options) - ALLOWED_OPTIONS)
    if unknown:
        return {}, f"the replay does not take the option {unknown[0]!r}"
    return dict(options), ""


def new_replay_dir(parent: Path, name: str) -> Path:
    """Create a folder for a replay that no earlier replay used, and return it.

    A second replay that starts in the same second as another one of the same recording gets the
    suffix `-2`, `-3`, and so on, so two replays never share a store.
    """
    parent.mkdir(parents=True, exist_ok=True)
    for number in itertools.count(1):
        path = parent / (name if number == 1 else f"{name}-{number}")
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return path
    raise AssertionError("unreachable")  # itertools.count never ends


def median_of(windows: list[SeeingWindowRecord], name: str) -> float | None:
    """The median of a window field over the windows that have it."""
    values = [v for w in windows if isinstance((v := getattr(w, name)), float) and math.isfinite(v)]
    return statistics.median(values) if values else None


class ReplayHandler:
    """Runs `replay` tasks. Register it with `Scheduler.register_handler("replay", handler)`."""

    def __init__(
        self,
        *,
        layout: DataLayout,
        profile: Profile,
        station_id: str,
        clock: Clock,
        fast_config: Any,
        store_config: StoreConfig,
        recordings_dir: Path | None,
        replays_dir: str = "replays",
    ) -> None:
        self._layout = layout
        self._profile = profile
        self._station_id = station_id
        self._clock = clock
        self._fast_config = fast_config
        self._store_config = store_config
        self._recordings_dir = recordings_dir
        self._replays_dir = replays_dir

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
        if not isinstance(command, QueueReplay):
            return self._failed(task, started_ns, "the task holds no replay command")
        if not (math.isfinite(command.speed) and command.speed >= 0):
            return self._failed(task, started_ns, "the speed is a number from 0 up")
        path, reason = resolve_source(command.source, self._recordings_dir, self._layout)
        if path is None:
            return self._failed(task, started_ns, f"cannot replay: {reason}")
        options, reason = clean_options(command.options)
        if reason:
            return self._failed(task, started_ns, f"cannot replay: {reason}")
        options["path"] = str(path)
        options["rate"] = command.speed if command.speed > 0 else "max"

        from seeingmon.drivers import create_driver
        from seeingmon.fastpath import create_fast_analyzer

        try:
            driver = create_driver(
                "replay", profile=self._profile, clock=context.clock, options=options
            )
        except CameraError as error:
            return self._failed(task, started_ns, f"cannot replay: {error}")
        stem = Path(command.source).stem or "replay"
        root = new_replay_dir(
            self._layout.root / self._replays_dir,
            f"{utc_ns_to_datetime(started_ns).strftime('%Y%m%dT%H%M%SZ')}-{stem[:40]}",
        )
        replay_layout = DataLayout(root)
        replay_layout.create()
        store = Store.open(replay_layout.db_path)
        segments = SegmentWriter(
            replay_layout.segments_dir,
            self._clock,
            station_id=self._station_id,
            profile_id=self._profile.id,
            config=self._store_config.segments,
        )
        analyzer = create_fast_analyzer(self._profile, self._fast_config, self._station_id)
        windows: list[SeeingWindowRecord] = []
        frames = dropped = 0
        stopped_for: str | None = None
        try:
            try:
                driver.open()
                caps = driver.capabilities()
                modes = [m.name for m in self._profile.readout_modes if m.sdk_bin in caps.bins]
                mode = options.get("mode") or (modes[0] if modes else None)
                if mode is None:
                    return self._failed(
                        task, started_ns, "cannot replay: the profile has no mode for the recording"
                    )
                active = driver.configure(
                    StreamConfig(
                        mode=str(mode),
                        exposure_us=1,
                        gain=0,
                        pixel_format=caps.pixel_formats[0],
                        roi=None,
                        kind=StreamKind.VIDEO,
                    )
                )
                windows += analyzer.begin_stream(active)
                driver.start()
                while True:
                    if context.should_stop():
                        stopped_for = "another command took over"
                        break
                    try:
                        frame = driver.read_frame(30.0)
                    except ReplayFinishedError:
                        break
                    update = analyzer.push(frame)
                    windows += update.windows
                    frames += 1
                    dropped += frame.dropped_before
                    if frames % METRICS_EVERY_FRAMES == 0:
                        self._drain(analyzer, segments, active.stream_id)
                windows += analyzer.flush("end")
                self._drain(analyzer, segments, active.stream_id)
                for window in windows:
                    store.write(window)
            except CameraConfigError as error:
                return self._failed(task, started_ns, f"cannot replay: {error}")
            except CameraError as error:
                return self._failed(task, started_ns, f"the replay failed: {error}")
        finally:
            driver.close()
            segments.close()
            store.close()
        finished_ns = self._clock.utc_ns()
        r0 = median_of(windows, "r0_cm")
        seeing = median_of(windows, "seeing_fwhm_arcsec")
        summary = f"Replayed {frames} frames into {len(windows)} windows."
        if r0 is not None:
            summary += f" The median r0 is {r0:.1f} cm."
        if stopped_for:
            summary = f"Stopped early ({stopped_for}). {summary}"
        return CommissionResult(
            task_id=task.task_id,
            kind=task.kind,
            status="aborted" if stopped_for else "ok",
            summary=summary,
            started_utc_ns=started_ns,
            finished_utc_ns=finished_ns,
            data={
                "frames": frames,
                "windows": len(windows),
                "dropped": dropped,
                "r0_cm_median": r0,
                "seeing_fwhm_arcsec_median": seeing,
                "duration_s": (finished_ns - started_ns) / NS_PER_S,
                "store": self._layout.relative(root),
            },
            artifacts=(self._layout.relative(replay_layout.db_path),),
            pinned=True,
        )

    @staticmethod
    def _drain(analyzer: Any, segments: SegmentWriter, stream_id: int) -> None:
        rows = analyzer.drain_metrics()
        if rows is not None and len(rows):
            segments.write_metrics(stream_id, rows)
