"""The quick solve of the alignment helper: where is Polaris, how is the camera rolled, how sharp.

`QuickSolver` runs the survey pipeline on one alignment frame and keeps what the live view needs.
The pipeline detects the stars, tries the pointing tracker first (it predicts the star field from
the latest solution, so it takes a few tens of milliseconds after the first solve), and falls back
to the plate solver when the mount moved too far for the tracker. A solution that passes the same
trust rule as the survey analysis updates the shared `PointingTracker`, so the scheduler follows
Polaris to its new place when the alignment ends.

The result is a `QuickSolution`: the position of Polaris at the time of the frame, the roll, the
number of matched stars and the residual, and the **focus value**, which is the median FWHM of the
detected stars that are neither saturated nor otherwise unreliable. The pipeline measures the FWHM
across the trail, so the focus value stays free of trailing.

**The brightest star.** A frame that the solver cannot solve still shows Polaris, which is far
brighter than any other star of the field (magnitude 2, against about 4 for the next one in 3
degrees). The solution keeps the position of the brightest detection and how many times brighter it
is than the next one (`brightest_ratio`), so that the rapid focus mode can find Polaris without a
solution (`seeingmon.services.core.alignment.rapid_availability`). A detection that carries the flag
of a hot pixel or of a streak does not count. A saturated star does, because Polaris saturates.

A frame that cannot be solved is a normal result (`solved` is false). The solver never raises: an
unexpected error becomes an unsolved result with a note, because a bad frame must not stop the live
view.

**Two steps.** `analyze_frame` is the analysis: it runs the pipeline and returns the solution of the
frame with the pointing solution that the tracker may adopt. It reads and changes nothing outside
its arguments, so it can run in this process (`QuickSolver`) or in a worker process
(`seeingmon.services.core.alignment.worker`), which keeps the detector away from the live view. The
second step, `adopt`, judges the pointing solution against the trust rule, updates the tracker, and
stays in the process of `core`, where the tracker lives. `QuickSolution` and `PointingSolution` turn
into plain data (`to_dict`) for the trip between the processes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Protocol

import numpy as np

from seeingmon.clock import NS_PER_S, Clock
from seeingmon.frames import Frame
from seeingmon.survey.detect import Detections, StarFlag
from seeingmon.survey.geometry import ARCSEC_PER_RAD
from seeingmon.survey.pipeline import FrameAnalysis, frame_time_invalid
from seeingmon.survey.pointing import PointingSolution, ReferenceSolution
from seeingmon.survey.tracker import PointingTracker
from seeingmon.survey.wcs_fit import CameraAttitude

_log = logging.getLogger(__name__)

MIN_FOCUS_SNR = 10.0
MIN_FOCUS_STARS = 3
LONE_STAR_RATIO = 1000.0  # the ratio of a brightest star that has no second star to compare with


@dataclass(frozen=True, slots=True)
class QuickSolution:
    """What one quick solve found. A field is `None` when the frame does not support it.

    `x_px` and `y_px` locate Polaris in the pixels of the frame's readout mode at the time of the
    frame, and `roll_deg` is the position angle of the direction to the pole (undefined when the
    pole sits on the field center). `focus_fwhm_px` is the median FWHM of the usable stars.

    `attitude` is the camera model at the time of the frame (CIRS to camera, in the pixels of the
    frame), and `polaris_colatitude_deg` is the angle between Polaris and the pole at that time.
    The live view builds its sky view, the pole, and the orbit of Polaris from the two. `elapsed_s`
    is the time that the analysis took.

    `brightest_x_px` and `brightest_y_px` give the position of the brightest detection in the
    pixels of the frame, and `brightest_ratio` says how many times brighter it is than the next
    one (`LONE_STAR_RATIO` when it has no next one). They are `None` when the frame has no
    detection that counts (see `brightest_star`), and they hold for an unsolved frame too.
    """

    t_utc_ns: int
    seq: int
    solved: bool
    x_px: float | None = None
    y_px: float | None = None
    roll_deg: float | None = None
    n_matched: int = 0
    rms_arcsec: float | None = None
    scale_arcsec_px: float | None = None
    solver: str = ""
    n_detected: int = 0
    focus_fwhm_px: float | None = None
    n_focus_stars: int = 0
    elapsed_s: float = 0.0
    note: str = ""
    attitude: CameraAttitude | None = None
    polaris_colatitude_deg: float | None = None
    brightest_x_px: float | None = None
    brightest_y_px: float | None = None
    brightest_ratio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """The solution as plain data (numbers, text, and lists), for another process."""
        data: dict[str, Any] = {
            "t_utc_ns": self.t_utc_ns,
            "seq": self.seq,
            "solved": self.solved,
            "x_px": self.x_px,
            "y_px": self.y_px,
            "roll_deg": self.roll_deg,
            "n_matched": self.n_matched,
            "rms_arcsec": self.rms_arcsec,
            "scale_arcsec_px": self.scale_arcsec_px,
            "solver": self.solver,
            "n_detected": self.n_detected,
            "focus_fwhm_px": self.focus_fwhm_px,
            "n_focus_stars": self.n_focus_stars,
            "elapsed_s": self.elapsed_s,
            "note": self.note,
            "attitude": None,
            "polaris_colatitude_deg": self.polaris_colatitude_deg,
            "brightest_x_px": self.brightest_x_px,
            "brightest_y_px": self.brightest_y_px,
            "brightest_ratio": self.brightest_ratio,
        }
        if self.attitude is not None:
            data["attitude"] = {
                "rotation": [float(v) for v in self.attitude.rotation.reshape(-1)],
                "scale_rad_px": self.attitude.scale_rad_px,
                "parity": self.attitude.parity,
                "center_px": [float(self.attitude.center_px[0]), float(self.attitude.center_px[1])],
            }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuickSolution:
        """The inverse of `to_dict`. Raises `KeyError`, `TypeError`, or `ValueError` on bad data."""
        attitude = None
        raw = data["attitude"]
        if raw is not None:
            rotation = np.array(raw["rotation"], dtype=np.float64).reshape(3, 3)
            centre = raw["center_px"]
            attitude = CameraAttitude(
                rotation=rotation,
                scale_rad_px=float(raw["scale_rad_px"]),
                parity=int(raw["parity"]),
                center_px=(float(centre[0]), float(centre[1])),
            )
        return cls(
            t_utc_ns=int(data["t_utc_ns"]),
            seq=int(data["seq"]),
            solved=bool(data["solved"]),
            x_px=_optional_float(data["x_px"]),
            y_px=_optional_float(data["y_px"]),
            roll_deg=_optional_float(data["roll_deg"]),
            n_matched=int(data["n_matched"]),
            rms_arcsec=_optional_float(data["rms_arcsec"]),
            scale_arcsec_px=_optional_float(data["scale_arcsec_px"]),
            solver=str(data["solver"]),
            n_detected=int(data["n_detected"]),
            focus_fwhm_px=_optional_float(data["focus_fwhm_px"]),
            n_focus_stars=int(data["n_focus_stars"]),
            elapsed_s=float(data["elapsed_s"]),
            note=str(data["note"]),
            attitude=attitude,
            polaris_colatitude_deg=_optional_float(data["polaris_colatitude_deg"]),
            brightest_x_px=_optional_float(data.get("brightest_x_px")),
            brightest_y_px=_optional_float(data.get("brightest_y_px")),
            brightest_ratio=_optional_float(data.get("brightest_ratio")),
        )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


@dataclass(frozen=True, slots=True, eq=False)
class QuickAnalysis:
    """The outcome of `analyze_frame`: the solution, and the pointing solution to offer the tracker.

    `pointing` is `None` when the frame was not solved. The tracker has not seen it yet: `adopt`
    judges it.
    """

    solution: QuickSolution
    pointing: PointingSolution | None = None


class Analyzer(Protocol):
    """`seeingmon.survey.pipeline.SurveyPipeline` fits."""

    def analyze(
        self,
        frame: Frame,
        *,
        previous: PointingSolution | None = None,
        reference: ReferenceSolution | None = None,
        index: int = 0,
        sky_quality: bool | None = None,
    ) -> FrameAnalysis: ...


def focus_value(detections: Detections | None) -> tuple[float | None, int]:
    """The median FWHM of the usable stars, and how many stars it rests on.

    A star is usable when it is reliable (not saturated, not at the edge, not blended, not a
    streak), has a finite width, and stands out with a signal-to-noise ratio of at least
    `MIN_FOCUS_SNR`. With fewer than `MIN_FOCUS_STARS` such stars the value is `None`.
    """
    if detections is None or len(detections) == 0:
        return None, 0
    usable = detections.reliable()
    usable &= np.isfinite(detections.fwhm_px) & (detections.fwhm_px > 0)
    usable &= detections.snr >= MIN_FOCUS_SNR
    count = int(np.count_nonzero(usable))
    if count < MIN_FOCUS_STARS:
        return None, count
    return float(np.median(detections.fwhm_px[usable])), count


def brightest_star(
    detections: Detections | None,
) -> tuple[float | None, float | None, float | None]:
    """The position of the brightest star, and how many times brighter it is than the next one.

    A star that carries the flag of a hot pixel or of a streak, or that has no positive flux, does
    not count. The ratio is `LONE_STAR_RATIO` for a brightest star that has no next one. Without a
    star that counts, the three values are `None`.
    """
    if detections is None or len(detections) == 0:
        return None, None, None
    counts = ~detections.has(StarFlag.HOT_PIXEL | StarFlag.STREAK)
    counts &= np.isfinite(detections.flux) & (detections.flux > 0.0)
    flux = detections.flux[counts]
    if len(flux) == 0:
        return None, None, None
    order = np.argsort(flux)
    first = int(order[-1])
    ratio = LONE_STAR_RATIO if len(flux) == 1 else float(flux[first] / flux[int(order[-2])])
    return (
        float(detections.x[counts][first]),
        float(detections.y[counts][first]),
        min(ratio, LONE_STAR_RATIO),
    )


def analyze_frame(
    pipeline: Analyzer,
    frame: Frame,
    *,
    previous: PointingSolution | None,
    reference: ReferenceSolution | None,
    index: int,
    clock: Clock,
) -> QuickAnalysis:
    """Analyze one alignment frame with the pipeline. Never raises.

    `previous` is the latest pointing solution (the tracker predicts the field from it) and
    `reference` the saved reference. Nothing here touches the tracker, so the function runs in any
    process that has a pipeline.
    """
    started = clock.monotonic_ns()
    try:
        analysis = pipeline.analyze(
            frame,
            previous=previous,
            reference=reference,
            index=index,
            sky_quality=False,  # the live view needs no zero point, and the step costs a second
        )
    except Exception as error:
        _log.exception("the quick solve of frame %d failed", frame.seq)
        return QuickAnalysis(
            QuickSolution(
                frame.t_utc_ns,
                frame.seq,
                False,
                elapsed_s=(clock.monotonic_ns() - started) / NS_PER_S,
                note=f"analysis error: {type(error).__name__}",
            )
        )
    elapsed_s = (clock.monotonic_ns() - started) / NS_PER_S
    focus, n_focus = focus_value(analysis.detections)
    bright_x, bright_y, bright_ratio = brightest_star(analysis.detections)
    n_detected = 0 if analysis.detections is None else len(analysis.detections)
    note = analysis.notes[-1] if analysis.notes else ""
    solution = analysis.solution
    if solution is None or not analysis.solved:
        return QuickAnalysis(
            QuickSolution(
                frame.t_utc_ns,
                frame.seq,
                False,
                n_detected=n_detected,
                focus_fwhm_px=focus,
                n_focus_stars=n_focus,
                elapsed_s=elapsed_s,
                note=note or "the frame could not be solved",
                brightest_x_px=bright_x,
                brightest_y_px=bright_y,
                brightest_ratio=bright_ratio,
            )
        )
    position = solution.polaris_pixel(frame.t_utc_ns)
    attitude = solution.attitude_at(frame.t_utc_ns)
    return QuickAnalysis(
        QuickSolution(
            frame.t_utc_ns,
            frame.seq,
            True,
            x_px=None if position is None else position[0],
            y_px=None if position is None else position[1],
            roll_deg=attitude.roll_deg(),
            n_matched=solution.n_matched,
            rms_arcsec=solution.rms_arcsec,
            scale_arcsec_px=solution.scale_rad_px * ARCSEC_PER_RAD,
            solver=solution.solver,
            n_detected=n_detected,
            focus_fwhm_px=focus,
            n_focus_stars=n_focus,
            elapsed_s=elapsed_s,
            note=note,
            attitude=attitude,
            polaris_colatitude_deg=solution.polaris_colatitude_deg(frame.t_utc_ns),
            brightest_x_px=bright_x,
            brightest_y_px=bright_y,
            brightest_ratio=bright_ratio,
        ),
        solution,
    )


def is_trusted(solution: PointingSolution, *, min_stars: int, max_rms_px: float) -> bool:
    """Whether a pointing solution may update the tracker: enough stars and a small residual."""
    rms_px = (
        None
        if solution.rms_arcsec is None
        else solution.rms_arcsec / (solution.scale_rad_px * ARCSEC_PER_RAD)
    )
    return solution.n_matched >= min_stars and (rms_px is None or rms_px <= max_rms_px)


def adopt(
    analysis: QuickAnalysis,
    tracker: PointingTracker,
    *,
    min_stars: int,
    max_rms_px: float,
    time_invalid: bool = False,
) -> QuickSolution:
    """Judge the pointing solution of an analysis, update the tracker, and return the solution.

    Only a solve that succeeds can replace the solution of the tracker: a frame that was not
    solved leaves it alone. So does a solution with fewer matched stars than `min_stars` or a
    residual above `max_rms_px` pixels, and its note says so. The solution of a frame that the
    clock did not time (`time_invalid`) may lie in the future, so it fills only a tracker that holds
    no solution or another untimed one, and the first timed solve replaces it, whatever its time
    (`PointingTracker.update`). When the tracker refuses it, its note says so, and the live view
    still shows that frame's solution.
    """
    pointing = analysis.pointing
    if pointing is None:
        return analysis.solution
    if not is_trusted(pointing, min_stars=min_stars, max_rms_px=max_rms_px):
        note = analysis.solution.note or "the fit is too weak to move the tracker"
        return replace(analysis.solution, note=note)
    adopted = tracker.update(pointing, timed=not time_invalid)
    if time_invalid and not adopted:
        note = "the clock is not synchronized, so the solution does not move the tracker"
        return replace(analysis.solution, note=note)
    return analysis.solution


class QuickSolver:
    """Solves alignment frames with the survey pipeline and the shared tracker.

    `min_stars` and `max_rms_px` are the trust rule for updating the tracker: a solution with
    fewer matched stars or a larger residual (in pixels) leaves the tracker alone. Call `solve`
    from one thread. The detector holds the GIL for seconds, so a live view in the same process
    stalls while a solve runs: `ProcessQuickSolver` runs the same analysis in a worker process.
    """

    def __init__(
        self,
        pipeline: Analyzer,
        tracker: PointingTracker,
        clock: Clock,
        *,
        min_stars: int = 8,
        max_rms_px: float = 1.5,
    ) -> None:
        self._pipeline = pipeline
        self._tracker = tracker
        self._clock = clock
        self._min_stars = min_stars
        self._max_rms_px = max_rms_px
        self._index = 0
        self.solves = 0
        self.failures = 0

    @property
    def tracker(self) -> PointingTracker:
        return self._tracker

    def solve(self, frame: Frame) -> QuickSolution:
        """Solve one frame. Never raises."""
        index, self._index = self._index, self._index + 1
        analysis = analyze_frame(
            self._pipeline,
            frame,
            previous=self._tracker.solution,
            reference=self._tracker.reference,
            index=index,
            clock=self._clock,
        )
        solution = adopt(
            analysis,
            self._tracker,
            min_stars=self._min_stars,
            max_rms_px=self._max_rms_px,
            time_invalid=frame_time_invalid(frame),
        )
        if analysis.pointing is None:
            self.failures += 1
        else:
            self.solves += 1
        return solution
