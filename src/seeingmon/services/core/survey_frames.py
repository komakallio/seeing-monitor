"""Survey frames on disk: the previews, the FITS files, and the newest frames in RAM.

The survey path turns each frame into records and then forgets the pixels. `SurveyFrames` wraps
the survey analyzer of `core` (it is a `SurveyAnalyzer` itself, as `NightlySummary` is) and keeps
what an operator and a later reanalysis want to see:

- **RAM.** `submit` keeps the frame in a ring of `ram_frames` frames (one by default, 23 MB at
  bin2). The ring holds a frame until the analysis has returned its result and the files are
  written, and it keeps the newest frames after that. It never copies a frame, and it never holds
  more than `ram_frames` of them, so the memory of `core` stays within its budget. A frame that
  falls out of the ring first gets no files. Nothing reads the frames that stay after their files
  are written: `web` is another process and cannot see them. A larger ring needs a reader first,
  such as a tmpfs file in the runtime directory that `core` writes and `web` serves.
- **Preview.** Every long frame (an exposure of at least `[survey.sky] min_exposure_s`, the frames
  that carry the sky quality) gets a JPEG of at most 1 megapixel under `previews/`, because the
  Images page of the web UI lists previews. A short frame (the 1 ms frame of the bright stars)
  gets one only when it is also kept as a FITS file.
- **FITS.** Every `keep_every`-th long frame (the first one included), and every event frame (long
  or short), goes to disk as a Rice-compressed FITS file under `survey/`
  (`seeingmon.survey.framefile`). An *event frame* is the first frame after one of these
  conditions starts: no pointing solution (`unsolved`, which covers a solver that failed and an
  analysis that failed), a pointing that moved (`moved`, the flag of the pointing record), clouds
  (the cloud fraction of the result reaches `event_cloud_fraction`), or a sky so bright that the
  background reaches `event_background_fraction` of the saturation level (`bright_sky`). The
  pointing and cloud conditions come from the long frame of a step, and the brightness from either
  frame. Each kind of frame gets at most one event frame in `event_min_interval_s`, so flickering
  clouds cannot fill the card. Writing a FITS file respects the capture gate of retention: below
  the free-space limit, the frame keeps its preview only. The header carries the cloud fraction
  and the transparency of the result when it has them (`CLOUDFRC` and `TRANSP`), so that
  `seeingmon flat build` can pick the clear frames from the files alone.
- **Reference.** The `survey_frame` record gets its `image_ref`: the FITS file when the frame was
  kept, else the preview, else `null`. The reference is a path under the data directory
  (`DataLayout.relative`), which the web process turns back into the image
  (`ImageStore.for_ref`).

**Names.** A preview is `previews/YYYY/MM/DD/<kind>-<stamp>.jpg` and a FITS file is
`survey/YYYY/MM/DD/<stamp>.fits`, with the stamp of the frame time, as `DataLayout` names them and
as the web process expects. The kind says what the frame is: `event` for an event frame, `short`
for a short frame, and `survey` for the rest. Retention ages the files by their modification time,
so the hourly task covers them (7 days of previews, and 7 days of FITS files and then one a night).

**Threads.** `submit` and `poll` run on the scheduler thread and do no I/O: `poll` decides, sets
the reference, and queues the job. One writer thread (`run`) writes the files, the preview first.
A write goes to a hidden temporary name, reaches the disk, and gets renamed
(`DataLayout.write_atomic`), so a reader never sees a partial file. A preview takes tens of
milliseconds and a FITS file a few hundred on a development machine. A failed write is logged and
counted, and the loop goes on. In a stepped run that has no thread, `drain` writes the queued
files on the calling thread.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from seeingmon.analysis import SurveyAnalyzer, SurveyOutput
from seeingmon.clock import NS_PER_S, Clock
from seeingmon.frames import Frame
from seeingmon.profile import Profile
from seeingmon.profile.errors import ProfileError
from seeingmon.records import PointingRecord, Record, SkyQualityRecord, SurveyFrameRecord
from seeingmon.services.core.alignment.preview import make_preview
from seeingmon.services.core.settings import SurveyFrameSettings
from seeingmon.store.layout import DataLayout
from seeingmon.survey.framefile import (
    frame_cards,
    frame_comments,
    native_pixels,
    write_frame_fits,
)

_log = logging.getLogger(__name__)

LONG = "long"
SHORT = "short"

KIND_SURVEY = "survey"
KIND_SHORT = "short"
KIND_EVENT = "event"

# The conditions that make an event frame.
UNSOLVED = "unsolved"
MOVED = "moved"
CLOUD = "cloud"
BRIGHT_SKY = "bright_sky"

FAILURE_EVENT_INTERVAL_S = 600.0

EventCallback = Callable[[str, str, str, Mapping[str, Any] | None], object]


@dataclass(frozen=True, slots=True)
class Decision:
    """What one frame gets.

    `kind` is the part of the preview name that says what the frame is. `reasons` says why the
    frame is a FITS file (`every_10`, `event:cloud`), and it stays filled when the capture gate
    stopped the file. `skipped` is `low_space` in that case.
    """

    frame_class: str
    kind: str
    preview: bool
    fits: bool
    reasons: tuple[str, ...] = ()
    skipped: str | None = None


class KeepPolicy:
    """Decides which frames get a preview and which a FITS file. See the module text.

    The policy keeps a count of the frames of each class and the conditions of the last one, so call
    `decide` once for each result, in the order of the results, from one thread.
    """

    def __init__(
        self,
        settings: SurveyFrameSettings,
        profile: Profile | None,
        *,
        long_min_exposure_s: float,
    ) -> None:
        self._settings = settings
        self._profile = profile
        self._long_min_us = round(long_min_exposure_s * 1e6)
        self._counts: dict[str, int] = {LONG: 0, SHORT: 0}
        self._active: dict[str, frozenset[str]] = {LONG: frozenset(), SHORT: frozenset()}
        self._last_event_ns: dict[str, int] = {}

    def classify(self, frame: Frame) -> str:
        """`long` for an exposure of at least the minimum of the sky quality, else `short`."""
        return LONG if frame.exposure_us >= self._long_min_us else SHORT

    def decide(
        self,
        frame: Frame,
        output: SurveyOutput,
        *,
        capture_allowed: Callable[[], bool] | None = None,
    ) -> Decision:
        """The decision for a frame whose analysis has returned `output`."""
        frame_class = self.classify(frame)
        count = self._counts[frame_class]
        self._counts[frame_class] = count + 1
        reasons: list[str] = []
        every = self._settings.keep_every
        if frame_class == LONG and count % every == 0:
            reasons.append(f"every_{every}")
        conditions = self._conditions(frame, output, frame_class)
        started = conditions - self._active[frame_class]
        self._active[frame_class] = conditions
        event = bool(started) and self._event_due(frame_class, frame.t_utc_ns)
        if event:
            self._last_event_ns[frame_class] = frame.t_utc_ns
            reasons.append("event:" + "+".join(sorted(started)))
        fits = bool(reasons)
        skipped: str | None = None
        if fits and not _gate_open(capture_allowed):
            fits, skipped = False, "low_space"
        kind = KIND_EVENT if event else (KIND_SHORT if frame_class == SHORT else KIND_SURVEY)
        return Decision(
            frame_class,
            kind,
            preview=frame_class == LONG or bool(reasons),
            fits=fits,
            reasons=tuple(reasons),
            skipped=skipped,
        )

    def _event_due(self, frame_class: str, t_utc_ns: int) -> bool:
        last = self._last_event_ns.get(frame_class)
        if last is None:
            return True
        elapsed_ns = t_utc_ns - last
        # A clock that stepped back makes the elapsed time negative: the limit does not hold then.
        limit_ns = round(self._settings.event_min_interval_s * NS_PER_S)
        return elapsed_ns < 0 or elapsed_ns >= limit_ns

    def _conditions(self, frame: Frame, output: SurveyOutput, frame_class: str) -> frozenset[str]:
        found: set[str] = set()
        fraction = self._background_fraction(frame, output)
        if fraction is not None and fraction >= self._settings.event_background_fraction:
            found.add(BRIGHT_SKY)
        if frame_class == LONG:
            if not output.solved:
                found.add(UNSOLVED)
            if any(isinstance(r, PointingRecord) and MOVED in r.flags for r in output.records):
                found.add(MOVED)
            cloud = output.cloud_fraction
            if cloud is not None and cloud >= self._settings.event_cloud_fraction:
                found.add(CLOUD)
        return frozenset(found)

    def _background_fraction(self, frame: Frame, output: SurveyOutput) -> float | None:
        """The background of the frame as a share of the saturation level, or `None`."""
        if self._profile is None:
            return None
        background = next(
            (
                r.background_dn
                for r in output.records
                if isinstance(r, SurveyFrameRecord) and r.background_dn is not None
            ),
            None,
        )
        if background is None:
            return None
        try:
            level = self._profile.saturation(frame.mode, frame.gain).native_dn
        except (ProfileError, ValueError):
            return None
        return float(background) / level if level > 0 else None


@dataclass(slots=True)
class FrameStats:
    """What the writer did, for the log and for a test."""

    frames: int = 0  # frames that went through `submit`
    results: int = 0  # results that `poll` matched to a frame in the ring
    previews: int = 0
    fits_files: int = 0
    fits_compressed: int = 0
    event_frames: int = 0
    bytes_written: int = 0
    failures: int = 0
    dropped: int = 0  # left the ring before the analysis returned
    lost: int = 0  # left the ring before the files were written
    skipped_low_space: int = 0
    last_error: str | None = None


@dataclass(slots=True)
class _Held:
    """A frame in the ring. `frame` becomes `None` when the ring lets it go."""

    frame: Frame | None
    analyzed: bool = False


@dataclass(frozen=True, slots=True)
class _Job:
    entry: _Held
    decision: Decision
    t_utc_ns: int
    preview_path: Path | None
    fits_path: Path | None
    cloud_fraction: float | None = None
    transparency: float | None = None


class SurveyFrames:
    """A `SurveyAnalyzer` that wraps another one and keeps its frames. See the module text."""

    def __init__(
        self,
        analyzer: SurveyAnalyzer,
        *,
        layout: DataLayout,
        profile: Profile | None,
        station_id: str,
        clock: Clock,
        settings: SurveyFrameSettings,
        long_min_exposure_s: float,
        capture_allowed: Callable[[], bool] | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self._analyzer = analyzer
        self._layout = layout
        self._profile = profile
        self._station_id = station_id
        self._clock = clock
        self._settings = settings
        self._capture_allowed = capture_allowed
        self._on_event = on_event
        self._policy = KeepPolicy(settings, profile, long_min_exposure_s=long_min_exposure_s)
        self._wake = threading.Condition()
        self._held: list[_Held] = []
        self._jobs: deque[_Job] = deque()
        self._closing = False
        self._previews_enabled = True
        self._last_failure_event_ns: int | None = None
        self.stats = FrameStats()

    # --- The SurveyAnalyzer interface ------------------------------------------------------

    def submit(self, frame: Frame) -> None:
        """Keep the frame in the ring, and hand it to the analyzer."""
        with self._wake:
            self._held.append(_Held(frame))
            self.stats.frames += 1
            self._evict()
        self._analyzer.submit(frame)

    def poll(self) -> tuple[SurveyOutput, ...]:
        """The results of the analyzer, with `image_ref` set where the frame has a file."""
        return tuple(self._handle(output) for output in self._analyzer.poll())

    def pending(self) -> int:
        return self._analyzer.pending()

    def __getattr__(self, name: str) -> Any:
        """Everything else (the tracker, the history, `close`) is the analyzer's."""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._analyzer, name)

    # --- The ring --------------------------------------------------------------------------

    @property
    def ram_frames(self) -> int:
        """The number of frames that the ring holds now."""
        with self._wake:
            return sum(1 for entry in self._held if entry.frame is not None)

    def recent_frames(self) -> list[Frame]:
        """The frames in the ring, newest first. The frames are the ones that the driver gave."""
        with self._wake:
            return [e.frame for e in reversed(self._held) if e.frame is not None]

    def _evict(self) -> None:
        """Let the oldest frames go until the ring is within its size. Call it with the lock."""
        while len(self._held) > self._settings.ram_frames:
            entry = self._held.pop(0)
            frame, entry.frame = entry.frame, None
            if frame is None:
                continue
            if not entry.analyzed:
                self.stats.dropped += 1
                _log.warning(
                    "the survey frame at %d left the RAM ring before its analysis returned, "
                    "so it gets no files",
                    frame.t_utc_ns,
                )

    def _take(self, t_utc_ns: int) -> _Held | None:
        """The oldest frame of that time whose result has not come yet. Call it with the lock."""
        for entry in self._held:
            if entry.frame is not None and not entry.analyzed and entry.frame.t_utc_ns == t_utc_ns:
                entry.analyzed = True
                return entry
        return None

    # --- Deciding --------------------------------------------------------------------------

    def _handle(self, output: SurveyOutput) -> SurveyOutput:
        with self._wake:
            entry = self._take(output.t_utc_ns)
            frame = None if entry is None else entry.frame
        if entry is None or frame is None:
            _log.info("the survey result at %d has no frame in the ring", output.t_utc_ns)
            return output
        try:
            decision = self._policy.decide(frame, output, capture_allowed=self._capture_allowed)
            job = self._job(entry, frame, decision, output)
            ref = self._reference(job)
        except Exception:
            _log.exception(
                "could not decide what to keep of the survey frame at %d", output.t_utc_ns
            )
            return output
        with self._wake:
            self.stats.results += 1
            if decision.skipped is not None:
                self.stats.skipped_low_space += 1
            if decision.kind == KIND_EVENT:
                self.stats.event_frames += 1
        if decision.skipped is not None:
            _log.warning(
                "the survey frame at %d is not kept as a FITS file: the free space is low (%s)",
                frame.t_utc_ns,
                ", ".join(decision.reasons),
            )
        if job.preview_path is None and job.fits_path is None:
            return output
        with self._wake:
            self._jobs.append(job)
            self._wake.notify()
        if ref is None:
            return output
        records = tuple(_with_image_ref(record, ref) for record in output.records)
        return dataclasses.replace(output, records=records)

    def _job(self, entry: _Held, frame: Frame, decision: Decision, output: SurveyOutput) -> _Job:
        sky = next((r for r in output.records if isinstance(r, SkyQualityRecord)), None)
        return _Job(
            entry,
            decision,
            frame.t_utc_ns,
            self._layout.preview_path(frame.t_utc_ns, kind=decision.kind)
            if decision.preview
            else None,
            self._layout.survey_path(frame.t_utc_ns) if decision.fits else None,
            cloud_fraction=output.cloud_fraction,
            transparency=None if sky is None else sky.transparency,
        )

    def _reference(self, job: _Job) -> str | None:
        chosen = job.fits_path or job.preview_path
        return None if chosen is None else self._layout.relative(chosen)

    # --- Writing ---------------------------------------------------------------------------

    def run(self) -> None:
        """The writer thread: write the queued files until `finish`, then the rest, and return."""
        while True:
            with self._wake:
                while not self._jobs and not self._closing:
                    self._wake.wait()
                if not self._jobs:
                    return
                job = self._jobs.popleft()
            self._execute(job)

    def drain(self) -> int:
        """Write every queued file on the calling thread. Returns the number of jobs it ran.

        A stepped run (`CoreApp` with `threads=False`) calls it instead of running the thread.
        """
        count = 0
        while True:
            with self._wake:
                if not self._jobs:
                    return count
                job = self._jobs.popleft()
            self._execute(job)
            count += 1

    def finish(self) -> None:
        """Tell the writer that no more frames come. It writes what is queued and `run` returns."""
        with self._wake:
            self._closing = True
            self._wake.notify_all()

    @property
    def queued(self) -> int:
        """The number of jobs that wait for the writer."""
        with self._wake:
            return len(self._jobs)

    def _execute(self, job: _Job) -> None:
        with self._wake:
            frame = job.entry.frame
        if frame is None:
            with self._wake:
                self.stats.lost += 1
            _log.warning(
                "the survey frame at %d left the RAM ring before its files were written",
                job.t_utc_ns,
            )
            return
        if job.preview_path is not None:
            self._guard("a preview", job, lambda: self._write_preview(frame, job.preview_path))
        if job.fits_path is not None:
            self._guard("a FITS file", job, lambda: self._write_fits(frame, job))

    def _guard(self, what: str, job: _Job, write: Callable[[], None]) -> None:
        try:
            write()
        except Exception as error:
            with self._wake:
                self.stats.failures += 1
                self.stats.last_error = f"{type(error).__name__}: {error}"
            _log.exception("could not write %s of the survey frame at %d", what, job.t_utc_ns)
            self._report_failure(what, job, error)

    def _write_preview(self, frame: Frame, path: Path | None) -> None:
        assert path is not None
        if not self._previews_enabled:
            return
        try:
            preview = make_preview(
                frame.data,
                max_pixels=self._settings.preview_max_pixels,
                quality=self._settings.jpeg_quality,
            )
        except ImportError:  # Pillow comes with the web extra, and this install lacks it
            self._previews_enabled = False
            _log.warning("the survey frames get no previews: Pillow is not installed")
            raise
        self._layout.write_atomic(path, preview.jpeg)
        with self._wake:
            self.stats.previews += 1
            self.stats.bytes_written += len(preview.jpeg)

    def _write_fits(self, frame: Frame, job: _Job) -> None:
        path, decision = job.fits_path, job.decision
        assert path is not None
        if not _gate_open(self._capture_allowed):
            with self._wake:
                self.stats.skipped_low_space += 1
            _log.warning(
                "the survey frame at %d is not written as a FITS file: the free space is low",
                job.t_utc_ns,
            )
            return
        pixels = native_pixels(frame)
        cards = frame_cards(
            frame,
            profile=self._profile,
            station_id=self._station_id,
            reasons=decision.reasons,
            cloud_fraction=job.cloud_fraction,
            transparency=job.transparency,
        )
        comments = frame_comments(frame)
        compressed = False

        def write(handle: BinaryIO) -> None:
            nonlocal compressed
            compressed = write_frame_fits(
                handle,
                pixels,
                cards,
                comments,
                compress=self._settings.fits_compression == "rice",
            )

        written = self._layout.write_atomic(path, write)
        size = written.stat().st_size
        with self._wake:
            self.stats.fits_files += 1
            self.stats.fits_compressed += int(compressed)
            self.stats.bytes_written += size

    def _report_failure(self, what: str, job: _Job, error: Exception) -> None:
        """Write one event for a failure, and at most one in `FAILURE_EVENT_INTERVAL_S`."""
        if self._on_event is None:
            return
        now = self._clock.monotonic_ns()
        last = self._last_failure_event_ns
        if last is not None and now - last < round(FAILURE_EVENT_INTERVAL_S * NS_PER_S):
            return
        self._last_failure_event_ns = now
        try:
            self._on_event(
                "warning",
                "survey_images.write_failed",
                f"Could not write {what} of a survey frame: {type(error).__name__}.",
                {"t_utc_ns": job.t_utc_ns, "kind": job.decision.kind},
            )
        except Exception:
            _log.exception("could not report the failed write")


def _gate_open(capture_allowed: Callable[[], bool] | None) -> bool:
    """Whether the capture gate lets a FITS file go to disk. A gate that fails counts as open."""
    if capture_allowed is None:
        return True
    try:
        return bool(capture_allowed())
    except Exception:
        _log.exception("could not read the free space, so the gate counts as open")
        return True


def _with_image_ref(record: Record, ref: str) -> Record:
    """The record with `image_ref` set, when it is a `survey_frame` that has none."""
    if isinstance(record, SurveyFrameRecord) and record.image_ref is None:
        return record.model_copy(update={"image_ref": ref})
    return record
