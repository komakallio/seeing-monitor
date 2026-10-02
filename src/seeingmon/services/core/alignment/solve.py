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

A frame that cannot be solved is a normal result (`solved` is false). The solver never raises: an
unexpected error becomes an unsolved result with a note, because a bad frame must not stop the live
view.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from seeingmon.clock import NS_PER_S, Clock
from seeingmon.frames import Frame
from seeingmon.survey.detect import Detections
from seeingmon.survey.geometry import ARCSEC_PER_RAD
from seeingmon.survey.pipeline import FrameAnalysis
from seeingmon.survey.pointing import PointingSolution, ReferenceSolution
from seeingmon.survey.tracker import PointingTracker
from seeingmon.survey.wcs_fit import CameraAttitude

_log = logging.getLogger(__name__)

MIN_FOCUS_SNR = 10.0
MIN_FOCUS_STARS = 3


@dataclass(frozen=True, slots=True)
class QuickSolution:
    """What one quick solve found. A field is `None` when the frame does not support it.

    `x_px` and `y_px` locate Polaris in the pixels of the frame's readout mode at the time of the
    frame, and `roll_deg` is the position angle of the direction to the pole (undefined when the
    pole sits on the field center). `focus_fwhm_px` is the median FWHM of the usable stars.

    `attitude` is the camera model at the time of the frame (CIRS to camera, in the pixels of the
    frame), and `polaris_colatitude_deg` is the angle between Polaris and the pole at that time.
    The live view builds its sky view, the pole, and the orbit of Polaris from the two.
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


class QuickSolver:
    """Solves alignment frames with the survey pipeline and the shared tracker.

    `min_stars` and `max_rms_px` are the trust rule for updating the tracker: a solution with
    fewer matched stars or a larger residual (in pixels) leaves the tracker alone. Call `solve`
    from one thread.
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

    def _trusted(self, solution: PointingSolution) -> bool:
        rms_px = (
            None
            if solution.rms_arcsec is None
            else solution.rms_arcsec / (solution.scale_rad_px * ARCSEC_PER_RAD)
        )
        return solution.n_matched >= self._min_stars and (
            rms_px is None or rms_px <= self._max_rms_px
        )

    def solve(self, frame: Frame) -> QuickSolution:
        """Solve one frame. Never raises."""
        started = self._clock.monotonic_ns()
        index, self._index = self._index, self._index + 1
        try:
            analysis = self._pipeline.analyze(
                frame,
                previous=self._tracker.solution,
                reference=self._tracker.reference,
                index=index,
                sky_quality=False,  # the live view needs no zero point, and the step costs a second
            )
        except Exception as error:
            self.failures += 1
            _log.exception("the quick solve of frame %d failed", frame.seq)
            return QuickSolution(
                frame.t_utc_ns,
                frame.seq,
                False,
                elapsed_s=(self._clock.monotonic_ns() - started) / NS_PER_S,
                note=f"analysis error: {type(error).__name__}",
            )
        elapsed_s = (self._clock.monotonic_ns() - started) / NS_PER_S
        focus, n_focus = focus_value(analysis.detections)
        n_detected = 0 if analysis.detections is None else len(analysis.detections)
        note = analysis.notes[-1] if analysis.notes else ""
        solution = analysis.solution
        if solution is None or not analysis.solved:
            self.failures += 1
            return QuickSolution(
                frame.t_utc_ns,
                frame.seq,
                False,
                n_detected=n_detected,
                focus_fwhm_px=focus,
                n_focus_stars=n_focus,
                elapsed_s=elapsed_s,
                note=note or "the frame could not be solved",
            )
        if self._trusted(solution):
            self._tracker.update(solution)
        else:
            note = note or "the fit is too weak to move the tracker"
        position = solution.polaris_pixel(frame.t_utc_ns)
        attitude = solution.attitude_at(frame.t_utc_ns)
        self.solves += 1
        return QuickSolution(
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
        )
