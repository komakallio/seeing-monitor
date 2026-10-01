"""The pointing tracker: Polaris position for the scheduler, and a predict-match-fit step.

`PointingTracker` implements `PointingProvider`. It keeps the latest `PointingSolution` and
answers two questions without a solver:

- **Where is Polaris?** `polaris_position(t, mode)` projects the apparent place of Polaris at
  time `t` through the attitude that the latest solution predicts for `t`. The Earth turns the
  field about the pole, and the apparent place follows precession, nutation, and aberration,
  so the position stays right between solves. The answer is `None` when there is no solution
  or the solution is older than the validity limit, which makes the scheduler take a survey
  frame to solve again.
- **Which stars are where?** `track` predicts the pixel position of every catalog star in the
  field from the latest solution, matches them to the detections of a new frame, and fits the
  attitude again. It takes a few tens of milliseconds, so the alignment helper uses it between
  full solves, and the survey analysis tries it before it calls a solver.

The tracker converts pixel positions between readout modes by the ratio of their frame sizes,
with the center of the first pixel at 0 in both: `x_new = (x + 0.5) * f - 0.5`.

The tracker is thread-safe. The scheduler reads from one thread while the analysis updates the
solution from another.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from seeingmon.profile import Profile
from seeingmon.survey import apparent
from seeingmon.survey.apparent import ObservationEpoch
from seeingmon.survey.catalog import CapCatalog
from seeingmon.survey.geometry import FloatArray
from seeingmon.survey.pointing import (
    PointingOffset,
    PointingSolution,
    ReferenceSolution,
    offset_between,
)
from seeingmon.survey.trail import TrailModel
from seeingmon.survey.wcs_fit import CameraAttitude, FitOptions, FitResult, fit_attitude

NS_PER_S = 1_000_000_000
DEFAULT_VALIDITY_S = 12 * 3600.0
# Catalog stars this much farther than the half diagonal of the frame can still match.
FIELD_MARGIN_DEG = 0.15


@dataclass(frozen=True, slots=True, eq=False)
class TrackResult:
    """The outcome of a predict-match-fit step.

    `rows` are the catalog rows that the predicted field covers, and `vectors` their apparent
    unit vectors at the time of the frame. `fit.catalog_index` indexes into `rows`.
    """

    fit: FitResult
    solution: PointingSolution
    epoch: ObservationEpoch
    rows: npt.NDArray[np.intp]
    vectors: FloatArray


class PointingTracker:
    """Keeps the latest pointing solution and predicts Polaris and the star field from it."""

    def __init__(
        self,
        profile: Profile,
        *,
        validity_s: float = DEFAULT_VALIDITY_S,
        reference: ReferenceSolution | None = None,
    ) -> None:
        if validity_s <= 0:
            raise ValueError("validity_s must be positive")
        self._profile = profile
        self._validity_ns = round(validity_s * NS_PER_S)
        self._lock = threading.Lock()
        self._solution: PointingSolution | None = None
        self._reference = reference

    # --- State ---------------------------------------------------------------------------

    @property
    def solution(self) -> PointingSolution | None:
        """The latest solution, or `None`."""
        with self._lock:
            return self._solution

    @property
    def reference(self) -> ReferenceSolution | None:
        with self._lock:
            return self._reference

    def set_reference(self, reference: ReferenceSolution | None) -> None:
        with self._lock:
            self._reference = reference

    def update(self, solution: PointingSolution) -> bool:
        """Adopt a solution, unless it is older than the one that the tracker holds.

        Returns whether the tracker adopted it. Results can arrive out of order when a worker
        process finishes frames late, and an old frame must not replace a newer solution.
        """
        with self._lock:
            current = self._solution
            if current is not None and solution.t_utc_ns < current.t_utc_ns:
                return False
            self._solution = solution
            return True

    def clear(self) -> None:
        """Forget the solution, for example after a camera fault that may have moved it."""
        with self._lock:
            self._solution = None

    def age_s(self, t_utc_ns: int) -> float | None:
        """The seconds from the latest solution to `t_utc_ns`, or `None` without a solution."""
        solution = self.solution
        return None if solution is None else (t_utc_ns - solution.t_utc_ns) / NS_PER_S

    def valid_at(self, t_utc_ns: int) -> bool:
        """Whether the latest solution is recent enough to predict the attitude at a time."""
        solution = self.solution
        return solution is not None and abs(t_utc_ns - solution.t_utc_ns) <= self._validity_ns

    # --- Predictions ----------------------------------------------------------------------

    def attitude_at(self, t_utc_ns: int) -> CameraAttitude | None:
        """The camera model at a time, in the readout mode of the solution, or `None`."""
        solution = self.solution
        if solution is None or not self.valid_at(t_utc_ns):
            return None
        return solution.attitude_at(t_utc_ns)

    def trail_model(self, t_utc_ns: int, exposure_s: float) -> TrailModel | None:
        """The trail model for an exposure that is centered on a time, or `None`."""
        attitude = self.attitude_at(t_utc_ns)
        if attitude is None:
            return None
        pole = attitude.pole_pixel()
        if pole is None:
            return None
        return TrailModel.for_exposure(pole[0], pole[1], exposure_s)

    def polaris_position(self, t_utc_ns: int, mode: str) -> tuple[float, float] | None:
        """The predicted Polaris position in sensor pixels of the readout mode `mode`.

        Returns `None` when there is no valid solution, so the scheduler takes a survey frame.
        """
        solution = self.solution
        if solution is None or not self.valid_at(t_utc_ns):
            return None
        position = solution.polaris_pixel(t_utc_ns)
        if position is None:
            return None
        return self.convert(position[0], position[1], solution.mode, mode)

    def convert(self, x: float, y: float, from_mode: str, to_mode: str) -> tuple[float, float]:
        """Convert a pixel position from one readout mode to another.

        The modes cover the same sensor, so the factor is the ratio of the frame widths (which
        the profile checks against the heights).
        """
        if from_mode == to_mode:
            return x, y
        source = self._profile.mode(from_mode)
        target = self._profile.mode(to_mode)
        factor = target.width_px / source.width_px
        height_factor = target.height_px / source.height_px
        if abs(factor - height_factor) > 1e-6 * factor:
            raise ValueError(f"readout modes {from_mode!r} and {to_mode!r} differ in aspect ratio")
        return (x + 0.5) * factor - 0.5, (y + 0.5) * factor - 0.5

    def offset_from_reference(self, solution: PointingSolution) -> PointingOffset | None:
        """The offset of a solution from the reference, or `None` without a reference."""
        reference = self.reference
        return None if reference is None else offset_between(solution, reference.solution)

    # --- Predict, match, fit --------------------------------------------------------------

    def track(
        self,
        catalog: CapCatalog,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        error_px: npt.ArrayLike,
        *,
        t_utc_ns: int,
        mode: str,
        shape: tuple[int, int],
        options: FitOptions | None = None,
        max_g_mag: float | None = None,
        commit: bool = False,
    ) -> TrackResult | None:
        """Predict the star field from the latest solution, match it to detections, and fit.

        `x`, `y`, and `error_px` are the detections in pixels of the readout mode `mode`, which
        must be the mode of the solution. `shape` is `(height, width)` of that mode. The
        function returns `None` when there is no valid solution, when too few stars match, or
        when the mode differs. With `commit` set, a successful fit becomes the new solution.
        """
        solution = self.solution
        if solution is None or solution.mode != mode or not self.valid_at(t_utc_ns):
            return None
        predicted = solution.attitude_at(t_utc_ns)
        epoch = apparent.epoch_from_utc_ns(t_utc_ns, solution.dut1_s)
        half_diagonal_deg = (
            0.5
            * float(np.hypot(solution.width_px, solution.height_px))
            * solution.scale_rad_px
            * 180.0
            / np.pi
        )
        center = apparent.astrometric_from_apparent(predicted.boresight(), epoch)
        rows = catalog.cone(center, half_diagonal_deg + FIELD_MARGIN_DEG, max_g_mag=max_g_mag)
        if rows.size == 0:
            return None
        vectors = apparent.apparent_vectors(
            catalog.ra_deg[rows],
            catalog.dec_deg[rows],
            catalog.pm_ra_mas_yr[rows],
            catalog.pm_dec_mas_yr[rows],
            catalog.parallax_mas[rows],
            epoch,
            catalog_epoch_jyear=catalog.epoch_jyear,
        )
        fit = fit_attitude(
            predicted,
            vectors,
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(error_px, dtype=np.float64),
            shape=shape,
            options=options or FitOptions(match_radius_px=(6.0, 3.0, 1.5)),
        )
        if fit is None:
            return None
        updated = PointingSolution.from_attitude(
            fit.attitude,
            epoch,
            mode=solution.mode,
            width_px=solution.width_px,
            height_px=solution.height_px,
            n_matched=fit.n_matched,
            rms_arcsec=fit.rms_arcsec,
            solver="tracker",
        )
        if commit:
            self.update(updated)
        return TrackResult(fit=fit, solution=updated, epoch=epoch, rows=rows, vectors=vectors)
