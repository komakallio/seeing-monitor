"""Interfaces between the scheduler and the analysis code.

The scheduler owns the camera and decides what runs next. The analysis code turns frames
into records. These protocols keep the two apart, so each lane builds and tests its side
against fakes of the other (see `seeingmon.testing`).

**Threading.** One consumer thread calls a `FastAnalyzer`, so it needs no locking, and
`push` is on the hot path: it runs for every frame (up to a few hundred times a second)
and must neither block nor do I/O. A `SurveyAnalyzer` takes a frame in `submit`, which
returns at once, and hands back results through `poll`, so an implementation can run the
heavy work (detection, solving, photometry) in a worker process. The scheduler never waits
for a survey result.

**Time.** All times are UTC nanoseconds, as in `seeingmon.frames`. Windows follow the frame
time (`Frame.t_utc_ns`), never the wall clock, so replays and simulations behave the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy.typing as npt

from seeingmon.frames import ActiveStream, Frame
from seeingmon.records import Record, SeeingWindowRecord


@dataclass(frozen=True, slots=True)
class StarState:
    """Where the target star is, as the fast analysis last saw it.

    Coordinates are sensor pixels of the readout mode (not ROI pixels), so they stay valid
    when the ROI moves. `edge_distance_px` is the distance from the star to the nearest ROI
    edge, which the scheduler uses to recenter the ROI before the star leaves it.
    """

    found: bool
    x_px: float | None = None
    y_px: float | None = None
    peak_fraction: float | None = None  # brightest pixel as a share of the saturation level
    edge_distance_px: float | None = None


NO_STAR = StarState(found=False)


@dataclass(frozen=True, slots=True)
class FastContext:
    """What the frames cannot tell the fast analysis. The scheduler or the core supplies it.

    `flags` are window flags (see `SEEING_WINDOW_FLAGS`) to add to every window that closes
    while they apply, such as `cloud`, `twilight`, or `heater_on`.
    """

    flags: frozenset[str] = frozenset()
    heater_duty: float | None = None
    zenith_angle_deg: float | None = None


@dataclass(frozen=True, slots=True)
class FastUpdate:
    """The result of one frame: the star state, and any windows that this frame closed."""

    star: StarState
    windows: tuple[SeeingWindowRecord, ...] = ()


@runtime_checkable
class FastAnalyzer(Protocol):
    """Per-frame metrics and windowed seeing statistics for the fast stream."""

    def begin_stream(self, stream: ActiveStream) -> tuple[SeeingWindowRecord, ...]:
        """Start analyzing a new stream, and return the windows that closing the old one produced.

        Call it after every `CameraDriver.configure`. A window never spans two streams, so
        an open window of the previous stream closes here, with the `partial` flag when it is
        shorter than the configured window length.
        """
        ...

    def set_context(self, context: FastContext) -> None:
        """Replace the context that applies to windows closing from now on."""
        ...

    def push(self, frame: Frame) -> FastUpdate:
        """Analyze one frame. A frame from a new stream starts that stream."""
        ...

    def flush(self, reason: str = "end") -> tuple[SeeingWindowRecord, ...]:
        """Close the open window early and return it (with the `partial` flag if short)."""
        ...

    def drain_metrics(self) -> npt.NDArray[Any] | None:
        """Return the per-frame metrics since the last call, or `None` when there are none.

        The rows use the dtype of the `frame` record, `segment_dtype("frame")` in
        `seeingmon.records.segments`.
        """
        ...


@dataclass(frozen=True, slots=True)
class SurveyOutput:
    """What the survey analysis made of one frame.

    `records` holds the survey records (`survey_frame`, `sky_quality`, `pointing`,
    `star_list`) for the frame. `solved` says whether the pointing solved, and
    `cloud_fraction` is the share of the expected catalog stars that detection missed (the
    scheduler adapts to it), or `None` when it cannot tell.
    """

    t_utc_ns: int
    records: tuple[Record, ...]
    solved: bool
    cloud_fraction: float | None = None


@runtime_checkable
class SurveyAnalyzer(Protocol):
    """Detection, solving, and photometry for survey frames, off the critical path."""

    def submit(self, frame: Frame) -> None:
        """Queue a survey frame. Returns at once."""
        ...

    def poll(self) -> tuple[SurveyOutput, ...]:
        """Return the results that finished since the last call, in submission order."""
        ...

    def pending(self) -> int:
        """The number of submitted frames whose results `poll` has not returned yet."""
        ...


@runtime_checkable
class PointingProvider(Protocol):
    """Where Polaris is on the sensor, from the latest pointing solution and the ephemeris."""

    def polaris_position(self, t_utc_ns: int, mode: str) -> tuple[float, float] | None:
        """The predicted Polaris position `(x, y)` in sensor pixels of the readout mode.

        Returns `None` when there is no solution yet, so the scheduler runs a survey frame to
        get one.
        """
        ...


@runtime_checkable
class RecordWriter(Protocol):
    """Where the scheduler and the analysis send finished records. The store implements it."""

    def write(self, record: Record) -> None:
        """Store one record. The record is immutable, and the store assigns its row ID."""
        ...


@runtime_checkable
class MetricsWriter(Protocol):
    """Where per-frame metric rows go. The store writes them to segment files."""

    def write_metrics(self, stream_id: int, rows: npt.NDArray[Any]) -> None:
        """Append rows with the `frame` record dtype from one stream."""
        ...
