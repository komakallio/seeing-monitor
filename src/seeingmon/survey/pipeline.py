"""One survey frame, from pixels to records.

`SurveyPipeline.analyze` runs the steps that turn a frame into the survey records:

1. **Detect.** `seeingmon.survey.detect` finds and measures the stars. When the tracker holds a
   recent solution, its trail model sets the trails.
2. **Solve.** The tracker comes first: it predicts the star field from the latest solution,
   matches it to the detections, and fits (`PointingTracker.track`). If that fails, the
   solver adapters run in the configured order, each on the star list, and the first solution
   starts the fit. The adapter's solution lives in the catalog frame, so the pipeline moves
   it to the apparent frame with the catalog stars of the field (`attitude_from_solver_solution`),
   and then fits against the apparent places (`fit_attitude`). See "Solver hints" for where the
   solvers search.
3. **Match.** With the final attitude, every detection is matched to the catalog, which gives
   the star list, the cloud fraction, and the matched stars for photometry.
4. **Sky quality.** `seeingmon.survey.quality` measures the stars (photometry and the zero
   point), the sky (the dark level, the flat, and a clipped median), the transparency, and the
   limiting magnitude. A frame with an exposure under `SkyConfig.min_exposure_s` (1 s) skips this
   step: a shorter frame, such as the 1 ms frame of a survey step, shows too few stars and too
   little sky, and the adaptive long frames of 1 s and longer get the record. Pass
   `sky_quality=True` to `analyze` to force the step, or `sky_quality=False` to skip it, as the
   alignment helper does for its frames of 0.5 s.
5. **Records.** The pipeline builds the `survey_frame`, `sky_quality`, `pointing`, and
   `star_list` records.

The pipeline holds no state between frames. The caller passes the previous solution and the
reference, and receives the new solution, so a worker process can run it and the parent can keep
the tracker. `PipelineSpec` describes a pipeline in plain data, so a worker process can build
its own copy.

**Failure.** A frame that cannot be solved is a normal result. The pointing record gets the
`unsolved` flag, and `quality` says why. Only an unexpected error propagates.

**No attempt.** A short frame with fewer than `MIN_SOLVER_STARS` stars gets no pointing record.
The 1 ms frame of each survey step shows Polaris alone by design, so no solver can use it, and an
`unsolved` record for it would say that a solve failed when none was tried. The frame keeps its
`survey_frame` record, and the latest `pointing` record stays the latest frame that a solver or
the tracker could try. A long frame (see `SkyConfig.min_exposure_s`) always gets one, so clouds
that hide the stars show as `unsolved`, and so does any frame with enough stars that fails. The
rule holds when detection fails too: a 1 ms frame that a sunny sky saturates whole keeps its
`survey_frame` record alone, and only a long frame gets the empty `sky_quality` record and the
`unsolved` pointing record of a failure.

**Solver hints.** With a solution, the solvers search `[survey.solve] hint_radius_deg` (2 degrees)
around the field center that the solution predicts. A solution has no age limit, so after the
mount moves more than that, the prediction points the solvers at the wrong place. When every
solver fails near the prediction, the pipeline therefore runs them again around the pole
(`pole_hint_radius_deg`, 15 degrees), where the camera looks. The trail model of the old solution
misshapes every star after such a move, so the retry detects the stars again without it. Without a
solution, the first search is the one around the pole. The retry needs a solver that ran and found
nothing (an error that keeps a solver from running, such as a missing program, would come back
with any hint), and a frame with at least `[survey.fit] confident_stars` stars, so that a thin
frame under clouds does not pay for a second detection.

**Solve attempts.** Each run of a plate solver leaves a `SolveAttempt` in `FrameAnalysis.attempts`:
the solver, the hint that it searched (`prediction`, `pole`, or `none`), the number of stars that
went to it, its time, and the outcome with the reason for a failure. The pipeline does not log
them, because it often runs in a worker process that has no log setup, where an info line would
vanish. `SurveyPipelineAnalyzer` logs one line for each attempt in the process of `core`.

**Cloud fraction.** The expected stars are the catalog stars in the field that the profile's
photometric prior says a clear sky would show at a signal-to-noise ratio of
`CloudConfig.expected_snr` or better, and that the search that ran finds with a chance of at least
`CloudConfig.min_completeness` (0.9), by a model of that search for the noise and the star image of
the frame (`seeingmon.survey.completeness`). Each star takes the trail of its own position, because
near the pole the trails are short and the binned search loses those stars, and the model sees the
share of a star's light that the fitted image holds (`light_in_core`), because the wings of a real
image hold light below the threshold. So a frame never
expects a star that its detector cannot find. The cloud fraction is the share of the expected stars
that no detection matches. Stars that a saturated star covers do not count. The `sky_quality`
record keeps both counts (`n_expected` and `n_expected_found`), also when too few stars are
expected for a fraction, and its provenance names the search (`search`: `full` or `binned2`).

**The search of a long frame.** The binned search (`[survey.detect] coarse_bin`) loses sharp stars
when nothing spreads them across its blocks: in a short frame at dusk it found 10 of the 13 stars
that the cloud fraction expected, so a clear frame read 0.23. A frame that gets a cloud fraction
(a long frame, see `SkyConfig.min_exposure_s`, unless the caller turns the sky quality step off,
and not a saturated sky) therefore takes the full search when the model says that the binned
search would find a typical star of the field at the faintest flux that the cloud fraction can
expect of the full search with a smaller chance than `min_completeness`. The model reads the noise
and the star image that the binned search measured, so the choice follows the sky and the
exposure: a dark frame of 30 s keeps the binned search, because its trail spreads the stars across
the blocks, and the long frames of a bright sky take the full search. The 1 ms frame of a step and
the frames of the alignment helper never pay for it.

**A saturated sky.** In a bright sky the background of a long frame can reach the saturation
level. A clipped background looks quiet, so the noise of the frame promises many stars that
detection then misses, and the frame would report clouds. A frame with more saturated pixels than
`[survey.twilight] max_saturated_fraction` (1%), or with a sky background above
`max_background_fraction` (80%) of saturation, therefore gets the flag `saturated_sky` on its
`sky_quality` record. The ring background of every star is clipped too, so the record holds no
photometry: no zero point, transparency, cloud fraction, counts of the expected stars, limiting
magnitude, or sky brightness. The frame keeps its pointing record, because its stars can still
solve, and the scheduler skips such frames in daylight (`seeingmon.scheduler.exposure`). The share
of saturated pixels comes from a regular sample of about 65,000 pixels, at the detector's
threshold.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import sep

from seeingmon.clock import NS_PER_S, Clock, SystemClock
from seeingmon.frames import Frame, FrameFlag, TimeQuality
from seeingmon.profile import Profile
from seeingmon.profile.errors import ProfileError
from seeingmon.records import Record
from seeingmon.records.survey import (
    SkyQualityRecord,
    StarListRecord,
    SurveyFrameRecord,
    pack_star_rows,
)
from seeingmon.solvers.base import PlateSolver, SolveRequest, SolverError, StarList
from seeingmon.survey import apparent
from seeingmon.survey.catalog import CapCatalog, load_catalog
from seeingmon.survey.completeness import fold_angle, star_completeness, star_limit_e
from seeingmon.survey.config import CloudConfig, SurveyConfig
from seeingmon.survey.dark import DARKS_DIRNAME, DarkLibrary, DarkModel, DarkStatus, dark_status
from seeingmon.survey.detect import (
    UNRELIABLE,
    DetectionError,
    Detections,
    DetectOptions,
    SearchSpec,
    StarFlag,
    detect_stars,
)
from seeingmon.survey.field import catalog_field
from seeingmon.survey.flat_library import ActiveFlat
from seeingmon.survey.geometry import FloatArray
from seeingmon.survey.photometry import (
    PhotometryOptions,
    aperture_correction,
    aperture_photometry,
    isolated,
)
from seeingmon.survey.pointing import (
    POINTING_ALGORITHM,
    PointingLimits,
    PointingSolution,
    ReferenceSolution,
    build_pointing_record,
)
from seeingmon.survey.quality import (
    FieldStars,
    QualityOptions,
    SkyQualityResult,
    assess_frame,
    field_stars,
)
from seeingmon.survey.rawdata import native_counts
from seeingmon.survey.sky import FlatModel, UnitFlat
from seeingmon.survey.star_epoch import FrameStars
from seeingmon.survey.tracker import PointingTracker
from seeingmon.survey.trail import TrailModel
from seeingmon.survey.transparency import ZeroPointReference
from seeingmon.survey.wcs_fit import (
    CameraAttitude,
    FitOptions,
    FitResult,
    attitude_from_solver_solution,
    fit_attitude,
    match_stars,
    pixel_center,
)

log = logging.getLogger("seeingmon.survey")

MIN_SOLVER_STARS = 4  # a plate solver needs at least this many stars
STAR_LIST_COLUMNS = ["x_px", "y_px", "flux_dn", "fwhm_px", "flags", "cat_row"]
STAR_LIST_CATALOG = "gaia-dr3+tycho-2"
STAR_LIST_G_LIMIT = 11.0  # matched stars brighter than this go to the star list
_UNMATCHED_ROW = -1.0
# The saturation guard counts the saturated pixels on a regular stride of at most about this many
# pixels, so a share of 1% rests on some 650 of them.
_GUARD_SAMPLE_PIXELS = 65_536
# `light_in_core` measures the share of the light in the fitted image on at most this many of the
# brightest isolated stars, each with at least this SNR, and needs at least `_CORE_MIN_STARS`.
_CORE_STARS = 20
_CORE_MIN_SNR = 30.0
_CORE_MIN_STARS = 3

# The outcomes of `SolveAttempt`.
SOLVED = "solved"  # the solver found a field, and the fit confirmed it
NO_SOLUTION = "no_solution"  # the solver ran and found no field
REJECTED = "rejected"  # the solver found a field that the catalog or the fit could not confirm
ERROR = "error"  # the solver could not run: a missing program, a crash, or a timeout

# The hints of `SolveAttempt`: where the solver searched.
HINT_PREDICTION = "prediction"  # around the field center that the latest solution predicts
HINT_POLE = "pole"  # around the celestial pole
HINT_NONE = "none"  # the whole sky


@dataclass(frozen=True, slots=True)
class SolveAttempt:
    """One run of a plate solver on the stars of a frame, as the log of `core` reports it.

    `stars` is the number of stars that went to the solver and `elapsed_s` its time. `matched` is
    the number of stars that the fit paired with the catalog, for a solved attempt. `reason` says
    why any other attempt failed, in the words of the matching note of the analysis. `hint` says
    where the solver searched (`HINT_PREDICTION`, `HINT_POLE`, or `HINT_NONE`), and the pipeline
    always sets it. An attempt holds no coordinate. It crosses the boundary of the worker process
    as a dictionary.
    """

    solver: str
    outcome: str
    stars: int
    elapsed_s: float
    matched: int = 0
    reason: str = ""
    hint: str = ""

    def describe(self) -> str:
        """The attempt as one structured text, such as `solver=astap hint=pole result=solved`."""
        text = f"solver={self.solver}"
        if self.hint:
            text += f" hint={self.hint}"
        text += f" result={self.outcome} stars={self.stars} time_s={self.elapsed_s:.2f}"
        if self.outcome == SOLVED:
            return f"{text} matched={self.matched}"
        return f"{text} reason={json.dumps(' '.join(self.reason.split()))}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "solver": self.solver,
            "outcome": self.outcome,
            "stars": self.stars,
            "elapsed_s": self.elapsed_s,
            "matched": self.matched,
            "reason": self.reason,
            "hint": self.hint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SolveAttempt:
        return cls(
            solver=str(data["solver"]),
            outcome=str(data["outcome"]),
            stars=int(data["stars"]),
            elapsed_s=float(data["elapsed_s"]),
            matched=int(data.get("matched", 0)),
            reason=str(data.get("reason", "")),
            hint=str(data.get("hint", "")),
        )


@dataclass(frozen=True, slots=True)
class SolverSpec:
    """A plate solver in plain data. `kind` is `triangles`, `astrometry.net`, or `astap`."""

    kind: str
    command: str
    index_dir: str = ""
    database_dir: str = ""


@dataclass(frozen=True, slots=True)
class PipelineSpec:
    """Everything a worker process needs to build its own `SurveyPipeline`, as plain data."""

    station_id: str
    profile: dict[str, Any]
    config: dict[str, Any]
    catalog_path: str
    solvers: tuple[SolverSpec, ...] = ()
    hot_pixel_file: str = ""


def solver_specs_from_config(config: SurveyConfig) -> tuple[SolverSpec, ...]:
    """The solver specifications that a configuration names, in order."""
    specs: list[SolverSpec] = []
    for name in config.solvers:
        if name == "triangles":
            specs.append(SolverSpec(name, ""))
        elif name == "astrometry.net":
            specs.append(SolverSpec(name, config.solve_field_command, index_dir=config.index_dir))
        elif name == "astap":
            specs.append(
                SolverSpec(name, config.astap_command, database_dir=config.astap_database_dir)
            )
        else:
            raise ValueError(f"unknown solver {name!r}; use triangles, astrometry.net, or astap")
    return tuple(specs)


def build_solvers(
    specs: tuple[SolverSpec, ...], clock: Clock | None = None, catalog: CapCatalog | None = None
) -> list[PlateSolver]:
    """Make the solver adapters that the specifications describe.

    The `triangles` solver works in this process on the catalog, so it needs `catalog`.
    """
    from seeingmon.solvers.astap import AstapSolver
    from seeingmon.solvers.astrometry_net import AstrometryNetSolver
    from seeingmon.solvers.triangles import TriangleSolver

    solvers: list[PlateSolver] = []
    for spec in specs:
        if spec.kind == "triangles":
            if catalog is None:
                raise ValueError("the triangles solver needs the catalog")
            triangles = TriangleSolver(catalog, clock=clock)
            triangles.prepare()  # the table builds here, not at the first solve
            solvers.append(triangles)
        elif spec.kind == "astrometry.net":
            solvers.append(AstrometryNetSolver(spec.index_dir, command=spec.command, clock=clock))
        elif spec.kind == "astap":
            solvers.append(
                AstapSolver(
                    command=spec.command, database_dir=spec.database_dir or None, clock=clock
                )
            )
        else:
            raise ValueError(f"unknown solver kind {spec.kind!r}")
    return solvers


def build_pipeline(spec: PipelineSpec, clock: Clock | None = None) -> SurveyPipeline:
    """Make a pipeline from its specification. This loads the catalog and the hot-pixel mask."""
    profile = Profile.model_validate(spec.profile)
    config = SurveyConfig.model_validate(spec.config)
    hot: npt.NDArray[np.bool_] | None = None
    if spec.hot_pixel_file:
        hot = np.asarray(np.load(spec.hot_pixel_file), dtype=np.bool_)
    library = (
        DarkLibrary(Path(config.calibration_dir) / DARKS_DIRNAME)
        if config.calibration_dir
        else None
    )
    # The flat is the active one of the flat library, then `flat_file`, then a unit flat. The
    # pipeline asks again before each frame, so an activation needs no restart.
    active = ActiveFlat.from_config(config)
    catalog = load_catalog(spec.catalog_path)
    return SurveyPipeline(
        station_id=spec.station_id,
        profile=profile,
        catalog=catalog,
        solvers=build_solvers(spec.solvers, clock, catalog),
        config=config,
        hot_pixels=hot,
        dark_library=library,
        flat=active.current(),
        flat_source=active.current,
        clock=clock,
    )


@dataclass(frozen=True, slots=True, eq=False)
class FrameAnalysis:
    """The result of one frame.

    `solution` is the new pointing solution (`None` for an unsolved frame), which the caller
    gives to the tracker. `detections` and the fields after it stay in the process that ran the
    pipeline, for the steps that build on the pointing (the sky quality uses them):
    `cat_row` holds the catalog row that each detection matched (-1 for none), `attitude` is
    the final camera model, `epoch` the time parameters, and `field_rows` and `field_vectors`
    the catalog stars of the field with their apparent places. `epoch_stars` holds what the frame
    adds to the nightly star summary, and `quality` the intermediate results of the sky quality.
    `timings` maps each step to its time in seconds, and `notes` explain a failure or a
    disagreement. `attempts` lists the runs of the plate solvers, in order, and it stays empty when
    the tracker solved the frame.
    """

    records: tuple[Record, ...]
    solved: bool
    cloud_fraction: float | None
    solution: PointingSolution | None = None
    detections: Detections | None = None
    fit: FitResult | None = None
    timings: dict[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    cat_row: npt.NDArray[np.intp] | None = None
    attitude: CameraAttitude | None = None
    epoch: apparent.ObservationEpoch | None = None
    field_rows: npt.NDArray[np.intp] | None = None
    field_vectors: FloatArray | None = None
    epoch_stars: FrameStars | None = None
    quality: SkyQualityResult | None = None
    attempts: tuple[SolveAttempt, ...] = ()


@dataclass(slots=True)
class _Solved:
    fit: FitResult
    rows: npt.NDArray[np.intp]  # catalog rows of the field
    vectors: FloatArray  # their apparent unit vectors
    solver: str
    solve_time_s: float
    cross_check: str | None = None


@dataclass(frozen=True, slots=True)
class CloudCount:
    """The expected stars of a frame and the cloud fraction that they give.

    `n_expected` counts the catalog stars that a clear sky shows at `CloudConfig.expected_snr`,
    and `n_found` those of them that detection found. Both are `None` when no catalog star of the
    field lies in the frame (or the frame has no pointing) and when no clear-sky signal exists.
    `fraction` is `None` then too, and also when fewer than `CloudConfig.min_expected` stars are
    expected.
    """

    fraction: float | None = None
    n_expected: int | None = None
    n_found: int | None = None


@dataclass(frozen=True, slots=True)
class _StarImage:
    """The star image of a frame, which sets the stars that the cloud fraction expects.

    `fwhm_px` is the median width of the reliable stars (1.1 px without one). `trail` is the trail
    model of the frame in the pixels of the data, and `origin` the position of the data in the
    sensor, so each star gets the trail of its own position. Without a model (the first frame after
    a start, which takes the full search), every star takes the typical trail: `trail_px`, the
    median length of the trails of the detections, and `trail_angle_rad`, the median of their
    directions folded to 0 to 45 degrees from a pixel axis (`completeness.fold_angle`).
    `core_fraction` is the share of a star's light in the fitted image (see `light_in_core`), and
    the models of the search see only that share.
    """

    fwhm_px: float
    trail_px: float
    trail_angle_rad: float
    trail: TrailModel | None = None
    origin: tuple[float, float] = (0.0, 0.0)
    core_fraction: float = 1.0

    @classmethod
    def of(
        cls,
        detections: Detections,
        trail: TrailModel | None = None,
        origin: tuple[float, float] = (0.0, 0.0),
        core_fraction: float | None = None,
    ) -> _StarImage:
        widths = detections.fwhm_px[detections.reliable()]
        widths = widths[np.isfinite(widths)]  # a NaN size must not spoil the frame
        fwhm = float(np.median(widths)) if widths.size else 1.1
        core = 1.0 if core_fraction is None else core_fraction
        usable = np.isfinite(detections.trail_length_px) & np.isfinite(detections.trail_angle_rad)
        if not np.any(usable):
            return cls(fwhm, 0.0, 0.0, trail, origin, core)
        length = float(np.median(detections.trail_length_px[usable]))
        angle = float(np.median(fold_angle(detections.trail_angle_rad[usable])))
        return cls(fwhm, length, angle, trail, origin, core)

    def trails(self, x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
        """The trail length and direction of a star at each sensor position."""
        if self.trail is None:
            return np.full(x.shape, self.trail_px), np.full(x.shape, self.trail_angle_rad)
        data_x, data_y = x - self.origin[0], y - self.origin[1]
        return self.trail.length(data_x, data_y), self.trail.angle(data_x, data_y)

    def aperture_minimum_e(
        self, cloud: CloudConfig, noise_e: float, trail_px: npt.ArrayLike
    ) -> FloatArray:
        """The clear-sky signal in electrons that reaches `expected_snr` in an aperture."""
        width = max(self.fwhm_px, 0.8)
        aperture = math.pi * (1.5 * width) ** 2 + 2.0 * np.asarray(trail_px) * width
        snr2 = cloud.expected_snr**2
        minimum = 0.5 * (snr2 + np.sqrt(snr2**2 + 4.0 * snr2 * aperture * noise_e**2))
        return np.asarray(minimum, dtype=np.float64)

    def completeness(
        self,
        search: SearchSpec,
        flux_e: npt.ArrayLike,
        noise_e: npt.ArrayLike,
        trail_px: npt.ArrayLike,
        trail_angle_rad: npt.ArrayLike,
    ) -> FloatArray:
        """The completeness of `search` for stars of these fluxes, noises, and trails.

        `flux_e` is the whole light of each star, and the model sees the share in the core.
        """
        return star_completeness(
            search,
            self.core_fraction * np.asarray(flux_e, dtype=np.float64),
            noise_e,
            fwhm_px=self.fwhm_px,
            trail_px=trail_px,
            trail_angle_rad=trail_angle_rad,
        )

    def limit_e(
        self,
        search: SearchSpec,
        completeness: float,
        noise_e: float,
        trail_px: npt.ArrayLike,
        trail_angle_rad: npt.ArrayLike,
    ) -> FloatArray:
        """The whole light of a star that `search` finds with the chance `completeness`."""
        core = star_limit_e(
            search,
            completeness,
            noise_e,
            fwhm_px=self.fwhm_px,
            trail_px=trail_px,
            trail_angle_rad=trail_angle_rad,
        )
        return np.asarray(core / self.core_fraction, dtype=np.float64)


def light_in_core(
    data: npt.NDArray[np.float32],
    detections: Detections,
    *,
    origin: tuple[float, float],
    e_per_adu: float,
    saturation_dn: float,
    options: PhotometryOptions,
    bad: npt.NDArray[np.bool_] | None = None,
) -> float | None:
    """The share of a star's light that the fitted image holds, or `None` with too few stars.

    The model of a search draws a star as a Gaussian of the fitted width
    (`seeingmon.survey.completeness`). A real image has wings that a Gaussian lacks: the Airy rings
    of the aperture hold about 16% of the light of a sharp image, and the simulator's image puts 9%
    of its light outside the fitted Gaussian. Light in the wings lies below the threshold, so a
    model that put all the light in the Gaussian would expect stars that the full search misses:
    for the simulator's image, it overstates the share found by 0.14 where it says 0.9.
    The share is the median ratio of the fitted flux to the flux in the aperture of the photometry
    (`PhotometryOptions.aperture_px`), over the brightest isolated stars that the fit measured
    (`_CORE_STARS`, with an SNR of `_CORE_MIN_SNR` or more), divided by the aperture correction of
    the growth curve (`seeingmon.survey.photometry.aperture_correction`), so it refers to the light
    within 12 pixels, as the zero point does. A frame with too few bright stars for the correction
    (8 with an SNR of 50 in the aperture) keeps the aperture's light, and the share then errs high
    by the light that the aperture misses: 1.5% for the simulator's image, and 2 to 3% for the
    reference optics. It stays between 0.5 and 1. `origin` is the position of `data` in the
    sensor.
    """
    bright = np.flatnonzero(
        detections.reliable()
        & (detections.snr >= _CORE_MIN_SNR)
        & np.isfinite(detections.flux)
        & (detections.flux > 0.0)
    )
    picked = detections.select(bright)
    alone = isolated(
        picked.x,
        picked.y,
        picked.flux,
        picked.trail_length_px,
        detections.x,
        detections.y,
        detections.flux,
        options,
    )
    picked = picked.select(np.flatnonzero(alone)[:_CORE_STARS])  # the detections are by flux
    if len(picked) < _CORE_MIN_STARS:
        return None
    measured = aperture_photometry(
        data,
        picked.x - origin[0],
        picked.y - origin[1],
        picked.trail_length_px,
        picked.trail_angle_rad,
        e_per_adu=e_per_adu,
        saturation_dn=saturation_dn,
        options=options,
        bad=bad,
    )
    usable = measured.ok & (measured.flux_e > 0.0) & (measured.error_e > 0.0)
    if int(usable.sum()) < _CORE_MIN_STARS:
        return None
    narrow = np.where(usable, measured.flux_e, 0.0)
    correction, _ = aperture_correction(
        data,
        picked,
        detections,
        narrow,
        np.where(usable, narrow / np.where(usable, measured.error_e, 1.0), 0.0),
        e_per_adu=e_per_adu,
        saturation_dn=saturation_dn,
        origin_px=origin,
        options=options,
        bad=bad,
        flat_scale=np.ones(len(picked)),
    )
    ratio = picked.flux[usable] * e_per_adu / narrow[usable]
    return float(np.clip(np.median(ratio) / correction, 0.5, 1.0))


def saturated_share(data: npt.NDArray[np.float32], threshold_dn: float) -> float:
    """The share of the pixels of a frame at or above `threshold_dn`.

    A large frame is measured on a regular stride of at most about 65,000 pixels, which keeps the
    cost far below that of detection.
    """
    stride = max(1, math.isqrt(data.size // _GUARD_SAMPLE_PIXELS))
    sample = data[::stride, ::stride]
    return float(np.count_nonzero(sample >= threshold_dn)) / max(1, sample.size)


def frame_time_invalid(frame: Frame) -> bool:
    """Whether the clock was not synchronized when the frame was taken."""
    return bool(frame.flags & FrameFlag.TIME_INVALID) or frame.t_quality == TimeQuality.INVALID


class SurveyPipeline:
    """Analyzes one survey frame at a time. See the module documentation."""

    def __init__(
        self,
        *,
        station_id: str,
        profile: Profile,
        catalog: CapCatalog,
        solvers: list[PlateSolver],
        config: SurveyConfig | None = None,
        hot_pixels: npt.NDArray[np.bool_] | None = None,
        dark_library: DarkLibrary | None = None,
        flat: FlatModel | None = None,
        flat_source: Callable[[], FlatModel] | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._station_id = station_id
        self._profile = profile
        self._catalog = catalog
        self._solvers = list(solvers)
        self._config = config or SurveyConfig()
        self._hot = hot_pixels
        self._library = dark_library
        self._flat: FlatModel = flat or UnitFlat()
        self._flat_source = flat_source  # asked before each frame: one `stat` when nothing changed
        self._hot_cache: dict[str, npt.NDArray[np.bool_]] = {}
        self._clock = clock or SystemClock()
        cfg = self._config
        self._quality = QualityOptions.from_config(cfg)
        self._detect_options = DetectOptions.from_config(cfg.detect)
        self._full_options = replace(self._detect_options, coarse_bin=1)
        self._fit_options = FitOptions(
            match_radius_px=cfg.fit.match_radius_px,
            clip_sigma=cfg.fit.clip_sigma,
            min_stars=cfg.fit.min_stars,
            sigma_floor_px=cfg.fit.sigma_floor_px,
        )
        self._limits = PointingLimits(
            few_stars=cfg.pointing.few_stars,
            moved_arcmin=cfg.pointing.moved_arcmin,
            moved_roll_deg=cfg.pointing.moved_roll_deg,
        )

    @property
    def catalog(self) -> CapCatalog:
        return self._catalog

    # --- The frame -----------------------------------------------------------------------

    def analyze(
        self,
        frame: Frame,
        *,
        previous: PointingSolution | None = None,
        reference: ReferenceSolution | None = None,
        index: int = 0,
        zp_reference: ZeroPointReference | None = None,
        sky_quality: bool | None = None,
    ) -> FrameAnalysis:
        """Analyze a frame. `previous` is the latest solution, `reference` the saved one.

        `index` counts the frames that the caller has analyzed, and it picks the frames for the
        second-solver check. `zp_reference` is the reference zero point of the clearest
        conditions (see `seeingmon.survey.transparency`), which gives the transparency. While the
        history is too short for a reference, it can be the provisional zero point of the last few
        hours instead. That one calibrates the sky brightness of a frame without a zero point of
        its own, and it sets neither a transparency nor the expected signal of the cloud
        fraction. `sky_quality` forces the sky quality step on or off. The default (`None`) runs it
        for frames with an exposure of at least `SkyConfig.min_exposure_s`.
        """
        if self._flat_source is not None:
            self._flat = self._flat_source()
        timings: dict[str, float] = {}
        started = self._clock.monotonic_ns()
        notes: list[str] = []
        attempts: list[SolveAttempt] = []
        try:
            readout = self._profile.mode(frame.mode)
        except ProfileError:
            return self._failure(
                frame,
                reference,
                f"unknown readout mode {frame.mode!r}",
                timings,
                sky_quality=sky_quality,
            )
        mode_shape = (readout.height_px, readout.width_px)

        def lap(name: str) -> None:
            nonlocal started
            now = self._clock.monotonic_ns()
            timings[name] = (now - started) / NS_PER_S
            started = now

        tracker = PointingTracker(self._profile, validity_s=self._config.pointing.validity_s)
        if previous is not None and previous.mode == frame.mode:
            tracker.update(previous)
        exposure_s = frame.exposure_us / 1e6
        trail = tracker.trail_model(frame.t_utc_ns, exposure_s)
        if trail is not None:  # the model works in the pixels of the data, not of the sensor
            trail = TrailModel(
                trail.pole_x - frame.roi.x, trail.pole_y - frame.roi.y, trail.rotation_rad
            )

        saturation = self._profile.saturation(frame.mode, frame.gain)
        native = native_counts(frame)
        hot = self._hot_mask_for(frame)
        guard = self._config.twilight
        saturated = saturated_share(
            native, self._detect_options.saturation_fraction * saturation.native_dn
        )

        e_per_adu = self._profile.e_per_adu(frame.mode, frame.gain)

        def detect(model: TrailModel | None, *, full: bool = False) -> Detections:
            raw = detect_stars(
                native,
                saturation_dn=saturation.native_dn,
                options=self._full_options if full else self._detect_options,
                e_per_adu=e_per_adu,
                hot_pixels=hot,
                trail=model,
            )
            return raw.shifted(frame.roi.x, frame.roi.y)

        cores: list[float] = []

        def core_of(found: Detections) -> float | None:
            """The share of a star's light in the fitted image (see `light_in_core`).

            The bright stars of a frame are the same in every search of it, so a measurement
            holds for the frame. A search with too few bright stars, such as the binned search of
            a short frame at dusk, leaves the measurement to the next one.
            """
            if not cores:
                share = light_in_core(
                    native,
                    found,
                    origin=(float(frame.roi.x), float(frame.roi.y)),
                    e_per_adu=e_per_adu,
                    saturation_dn=saturation.native_dn,
                    options=self._quality.photometry,
                    bad=hot,
                )
                if share is None:
                    return None
                cores.append(share)
            return cores[0]

        def sky_of(found: Detections) -> tuple[float, bool]:
            """The background as a share of saturation, and whether the sky counts as saturated."""
            level = float(found.background_level) / saturation.native_dn
            clipped = saturated > guard.max_saturated_fraction
            return level, clipped or level > guard.max_background_fraction

        long_frame = exposure_s >= self._config.sky.min_exposure_s
        # The sky quality step, and with it the cloud fraction, runs on the long frames unless the
        # caller says otherwise, as the quick solve of the alignment helper does.
        wanted = long_frame if sky_quality is None else sky_quality
        # A provisional zero point is no clear-sky level, so the cloud fraction keeps the
        # photometric prior of the profile, as it does while no reference exists.
        cloud_reference = None if zp_reference is None or zp_reference.provisional else zp_reference
        try:
            detections = detect(trail)
        except DetectionError as error:
            return self._failure(
                frame,
                reference,
                f"detection failed: {error}",
                timings,
                saturated_sky=saturated > guard.max_saturated_fraction,
                sky_quality=sky_quality,
            )
        # Only a frame that gets a cloud fraction pays for the full search: not the 1 ms frame of
        # a step, not the alignment frames, and not a saturated sky, which gets no cloud fraction.
        if (
            wanted
            and long_frame
            and not sky_of(detections)[1]
            and self._full_search_due(frame, detections, cloud_reference, core_of)
        ):
            notes.append(
                "the binned search would miss stars that the cloud fraction expects, so the frame "
                "takes the full search"
            )
            try:
                detections = detect(trail, full=True)
            except DetectionError as error:
                notes.append(f"the full search failed, so the binned search stays: {error}")
        lap("detect")
        detected_trail = trail  # the trail model behind the detections, if any
        background, saturated_sky = sky_of(detections)
        if saturated_sky:
            notes.append(
                f"the sky is saturated: {saturated:.1%} of the pixels saturate, and the "
                f"background reads {background:.0%} of saturation"
            )

        epoch = apparent.epoch_from_utc_ns(frame.t_utc_ns, self._config.dut1_s)
        solved = self._solve(frame, detections, tracker, epoch, mode_shape, index, notes, attempts)
        if solved is None and self._pole_retry_due(detections, attempts):
            # The mount may have moved further than the hint around the prediction reaches. The
            # trail model of the old solution then misshapes every star, so detect them again
            # without it, as in the first frame after a start.
            notes.append("no solver found the field near the prediction, so they try the pole")
            if trail is not None:
                try:
                    detections = detect(None)
                    detected_trail = None
                except DetectionError as error:
                    notes.append(f"the detection without a trail model failed: {error}")
            solved = self._solve(
                frame, detections, tracker, epoch, mode_shape, index, notes, attempts, pole=True
            )
        lap("solve")
        fit = None if solved is None else solved.fit

        # Match every detection to the catalog with the final attitude.
        attitude = None if fit is None else fit.attitude
        field_rows = np.zeros(0, dtype=np.intp) if solved is None else solved.rows
        field_vectors: FloatArray = np.zeros((0, 3)) if solved is None else solved.vectors
        if attitude is None and tracker.valid_at(frame.t_utc_ns):
            predicted = tracker.attitude_at(frame.t_utc_ns)
            if predicted is not None:
                field_rows, field_vectors = catalog_field(
                    self._catalog,
                    predicted,
                    epoch,
                    width_px=readout.width_px,
                    height_px=readout.height_px,
                )
        match_attitude = attitude or tracker.attitude_at(frame.t_utc_ns)
        cat_row = np.full(len(detections), -1, dtype=np.intp)
        if match_attitude is not None and field_rows.size and len(detections):
            cat_row = self._match_all(
                match_attitude,
                detections,
                field_rows,
                field_vectors,
                mode_shape,
                fitted=attitude is not None,
            )
        lap("match")

        roi = frame.roi
        coverage: FieldStars | None = None
        if match_attitude is not None and field_rows.size:
            coverage = field_stars(
                self._catalog,
                match_attitude,
                field_rows,
                field_vectors,
                detections,
                cat_row,
                roi_bounds=(float(roi.x), float(roi.y), float(roi.x_end), float(roi.y_end)),
                edge_px=self._config.cloud.edge_px,
                match_radius_px=self._config.cloud.match_radius_px,
            )
        # The cloud fraction gives each catalog star the trail of its position. Detections without
        # the trail model (the first frame after a start, or the retry at the pole) have none. Only
        # a frame with the sky quality step measures the share of the light in the fitted image.
        core = core_of(detections) if wanted and not saturated_sky else None
        image = _StarImage.of(detections, detected_trail, (float(roi.x), float(roi.y)), core)
        # A clipped background promises stars that the frame cannot show, so a saturated sky gives
        # no counts and no cloud fraction.
        count = (
            CloudCount()
            if saturated_sky
            else self._cloud_count(frame, detections, coverage, cloud_reference, image)
        )
        cloud = count.fraction
        focus = self._focus(detections)
        solution: PointingSolution | None = None
        if attitude is not None and fit is not None:
            solution = PointingSolution.from_attitude(
                attitude,
                epoch,
                mode=frame.mode,
                width_px=readout.width_px,
                height_px=readout.height_px,
                n_matched=fit.n_matched,
                rms_arcsec=fit.rms_arcsec,
                solver="" if solved is None else solved.solver,
            )
        quality: SkyQualityResult | None = None
        if wanted:
            dark_model, status = self._dark_for(frame)
            extra = self._quality_provenance(dark_model)
            if detections.search is not None:
                extra["search"] = detections.search.label  # the search behind n_expected
            if core is not None:
                extra["core"] = f"{core:.3f}"  # the share of the light that the model of it sees
            quality = assess_frame(
                station_id=self._station_id,
                profile=self._profile,
                frame=frame,
                data=native,
                detections=detections,
                cat_row=cat_row,
                catalog=self._catalog,
                attitude=attitude,
                field_rows=field_rows,
                field_vectors=field_vectors,
                field=coverage,
                cloud_fraction=cloud,
                n_expected=count.n_expected,
                n_expected_found=count.n_found,
                dark_model=dark_model,
                dark_status=status,
                flat=self._flat,
                hot_pixels=hot,
                zp_reference=zp_reference,
                options=self._quality,
                provenance=self._provenance(extra),
                time_invalid=frame_time_invalid(frame),
                saturated_sky=saturated_sky,
            )
            lap("quality")
        # A short frame that no solver could use is no failed solve, so it gets no pointing record.
        attempted = (
            solved is not None or long_frame or len(detections.star_list()) >= MIN_SOLVER_STARS
        )
        records = self._records(
            frame,
            detections,
            solved,
            solution,
            epoch,
            reference,
            cat_row,
            focus,
            None if quality is None else quality.record,
            pointing=attempted,
        )
        lap("records")
        return FrameAnalysis(
            records=records,
            solved=solved is not None,
            cloud_fraction=cloud,
            solution=solution,
            detections=detections,
            fit=fit,
            timings=timings,
            notes=tuple(notes),
            cat_row=cat_row,
            attitude=attitude,
            epoch=epoch,
            field_rows=field_rows,
            field_vectors=field_vectors,
            epoch_stars=FrameStars.empty() if quality is None else quality.stars,
            quality=quality,
            attempts=tuple(attempts),
        )

    def _window(self, mask: npt.NDArray[np.bool_], frame: Frame) -> npt.NDArray[np.bool_] | None:
        """The part of a full-sensor mask that a frame covers, or `None` if the sizes differ."""
        if mask.shape == frame.shape:
            return mask
        roi = frame.roi
        window = mask[roi.y : roi.y_end, roi.x : roi.x_end]
        return np.ascontiguousarray(window) if window.shape == frame.shape else None

    def _hot_mask_for(self, frame: Frame) -> npt.NDArray[np.bool_] | None:
        """The hot pixels of a frame: the configured mask and the pixels of the dark library."""
        masks = []
        if self._hot is not None:
            window = self._window(self._hot, frame)
            if window is not None:
                masks.append(window)
        library = self._library_hot_mask(frame)
        if library is not None:
            masks.append(library)
        if not masks:
            return None
        return masks[0] if len(masks) == 1 else masks[0] | masks[1]

    def _library_hot_mask(self, frame: Frame) -> npt.NDArray[np.bool_] | None:
        if self._library is None or frame.temperature_c is None:
            return None
        chosen = self._library.nearest(frame.mode, frame.gain, frame.temperature_c)
        if chosen is None:
            return None
        mask = self._hot_cache.get(chosen.name)
        if mask is None:
            mask = np.zeros((chosen.height_px, chosen.width_px), dtype=np.bool_)
            columns, rows, _ = self._library.hot_pixels(chosen)
            mask[rows, columns] = True
            self._hot_cache = {chosen.name: mask}  # one set at a time is enough
        return self._window(mask, frame)

    def _dark_for(self, frame: Frame) -> tuple[DarkModel | None, DarkStatus | None]:
        """The dark model of the frame's readout setting, and whether the library is due."""
        if self._library is None:
            return None, None
        cfg = self._config.dark
        model = self._library.model(frame.mode, frame.gain, prior_doubling_c=cfg.doubling_c)
        status = dark_status(
            self._library,
            frame.temperature_c,
            frame.t_utc_ns,
            mode=frame.mode,
            gain=frame.gain,
            tolerance_c=cfg.temperature_tolerance_c,
            max_age_days=cfg.max_age_days,
        )
        return model, status

    def _quality_provenance(self, dark_model: DarkModel | None) -> dict[str, str]:
        return {
            "dark": "none" if dark_model is None else dark_model.version,
            "flat": self._flat.version,
        }

    # --- Solving -------------------------------------------------------------------------

    def _solve(
        self,
        frame: Frame,
        detections: Detections,
        tracker: PointingTracker,
        epoch: apparent.ObservationEpoch,
        shape: tuple[int, int],
        index: int,
        notes: list[str],
        attempts: list[SolveAttempt],
        *,
        pole: bool = False,
    ) -> _Solved | None:
        """The tracker first, then each solver, until one solves. Returns `None` if none does.

        With `pole` set, the tracker stays out, and the solvers search around the pole.
        """
        reliable = detections.reliable()
        x, y = detections.x[reliable], detections.y[reliable]
        error = 0.5 * (detections.x_error_px[reliable] + detections.y_error_px[reliable])
        t = frame.t_utc_ns
        if not pole and tracker.valid_at(t) and x.size:
            started = self._clock.monotonic_ns()
            tracked = tracker.track(
                self._catalog,
                x,
                y,
                error,
                t_utc_ns=t,
                mode=frame.mode,
                shape=shape,
                options=self._fit_options,
            )
            if tracked is not None and self._acceptable(tracked.fit, int(x.size)):
                elapsed = (self._clock.monotonic_ns() - started) / NS_PER_S
                return _Solved(tracked.fit, tracked.rows, tracked.vectors, "tracker", elapsed)
            notes.append("the tracker could not match the frame")

        stars = detections.star_list()
        if len(stars) < MIN_SOLVER_STARS:
            notes.append(f"only {len(stars)} stars for a solver")
            return None
        request, hint = self._solve_request(frame, stars, tracker, epoch, pole=pole)
        for solver in self._solvers:
            outcome = self._run_solver(
                solver, request, hint, frame, detections, x, y, error, epoch, shape, notes, attempts
            )
            if outcome is not None:
                every = self._config.solve.cross_check_every
                if every > 0 and index % every == 0:
                    outcome.cross_check = self._cross_check(
                        solver, request, outcome, frame, epoch, shape, notes
                    )
                return outcome
        return None

    def _pole_retry_due(self, detections: Detections, attempts: list[SolveAttempt]) -> bool:
        """Whether the solvers search around the pole after they failed near the prediction.

        A solver must have run near the prediction and found nothing: an error that keeps a
        solver from running, such as a missing program, comes back with any hint. The frame needs
        `[survey.fit] confident_stars` stars or more. A thinner frame, as under clouds, would
        rarely confirm a field in a search that wide, and the retry costs a second detection.
        """
        return (
            self._config.solve.pole_hint_radius_deg > 0.0
            and len(detections.star_list()) >= self._config.fit.confident_stars
            and any(a.hint == HINT_PREDICTION and a.outcome != ERROR for a in attempts)
        )

    def _acceptable(self, fit: FitResult, n_reliable: int) -> bool:
        """Whether a fit is a solution: enough pairs, a small residual, and not by chance."""
        cfg = self._config.fit
        if fit.n_matched < cfg.min_stars or fit.rms_px > cfg.max_rms_px:
            return False
        return (
            fit.n_matched >= cfg.confident_stars
            or fit.n_matched >= cfg.min_match_fraction * n_reliable
        )

    def _solve_request(
        self,
        frame: Frame,
        stars: StarList,
        tracker: PointingTracker,
        epoch: apparent.ObservationEpoch,
        *,
        pole: bool = False,
    ) -> tuple[SolveRequest, str]:
        """The request for the solvers, and its hint (`HINT_PREDICTION`, `HINT_POLE`, `HINT_NONE`).

        The hint surrounds the field center that the tracker predicts, or the pole without a
        prediction. With `pole` set, it surrounds the pole in any case.
        """
        cfg = self._config.solve
        readout = self._profile.mode(frame.mode)
        scale = self._profile.plate_scale_arcsec_per_px(readout)
        hint_ra = hint_dec = radius = None
        hint = HINT_NONE
        predicted = None if pole else tracker.attitude_at(frame.t_utc_ns)
        if predicted is not None:
            ra, dec = predicted.center_icrs(epoch)
            hint_ra, hint_dec, radius = ra, dec, cfg.hint_radius_deg
            hint = HINT_PREDICTION
        elif cfg.pole_hint_radius_deg > 0.0:
            # No pointing is known, but the camera looks at Polaris, so the field is near the
            # pole. A hint makes ASTAP solve in 0.2 s, where a blind search takes seconds and
            # sometimes fails.
            hint_ra, hint_dec, radius = 0.0, 90.0, cfg.pole_hint_radius_deg
            hint = HINT_POLE
        order = np.argsort(-stars.flux, kind="stable")[: cfg.max_stars]
        request = SolveRequest(
            stars=StarList(x=stars.x[order], y=stars.y[order], flux=stars.flux[order]),
            width_px=readout.width_px,
            height_px=readout.height_px,
            scale_low_arcsec_px=scale * (1.0 - cfg.scale_tolerance),
            scale_high_arcsec_px=scale * (1.0 + cfg.scale_tolerance),
            center_ra_deg=hint_ra,
            center_dec_deg=hint_dec,
            radius_deg=radius,
            timeout_s=cfg.timeout_s,
        )
        return request, hint

    def _attitude_from_result(
        self,
        result_center: tuple[float, float],
        cd_matrix: tuple[float, float, float, float],
        frame: Frame,
        epoch: apparent.ObservationEpoch,
    ) -> CameraAttitude | None:
        """The apparent-frame attitude that a solver's catalog-frame solution implies."""
        readout = self._profile.mode(frame.mode)
        center_px = pixel_center(readout.width_px, readout.height_px)
        determinant = abs(cd_matrix[0] * cd_matrix[3] - cd_matrix[1] * cd_matrix[2])
        scale_deg = float(np.sqrt(determinant))  # degrees per pixel
        radius = 0.5 * float(np.hypot(readout.width_px, readout.height_px)) * scale_deg + 0.1
        rows = self._catalog.cone(result_center, radius, max_g_mag=STAR_LIST_G_LIMIT)
        if rows.size < 4:
            return None
        vectors = apparent.apparent_vectors(
            self._catalog.ra_deg[rows],
            self._catalog.dec_deg[rows],
            self._catalog.pm_ra_mas_yr[rows],
            self._catalog.pm_dec_mas_yr[rows],
            self._catalog.parallax_mas[rows],
            epoch,
            catalog_epoch_jyear=self._catalog.epoch_jyear,
        )
        return attitude_from_solver_solution(
            result_center[0],
            result_center[1],
            cd_matrix,
            center_px=center_px,
            catalog_vectors_icrs=self._catalog.vectors[rows],
            apparent_vectors=vectors,
        )

    def _run_solver(
        self,
        solver: PlateSolver,
        request: SolveRequest,
        hint: str,
        frame: Frame,
        detections: Detections,
        x: FloatArray,
        y: FloatArray,
        error: FloatArray,
        epoch: apparent.ObservationEpoch,
        shape: tuple[int, int],
        notes: list[str],
        attempts: list[SolveAttempt],
    ) -> _Solved | None:
        stars = len(request.stars)
        started = self._clock.monotonic_ns()

        def failed(outcome: str, reason: str, elapsed_s: float) -> None:
            """Record a failed attempt. The reason is also the note of the analysis."""
            notes.append(reason)
            attempts.append(
                SolveAttempt(solver.name, outcome, stars, elapsed_s, reason=reason, hint=hint)
            )

        try:
            result = solver.solve(request)
        except SolverError as exc:
            log.warning("the %s solver failed: %s", solver.name, exc)
            waited_s = (self._clock.monotonic_ns() - started) / NS_PER_S
            failed(ERROR, f"{solver.name} failed: {exc}", waited_s)
            return None
        if not result.solved or result.cd_matrix is None or result.center_ra_deg is None:
            failed(NO_SOLUTION, f"{solver.name} found no solution", result.elapsed_s)
            return None
        assert result.center_dec_deg is not None
        initial = self._attitude_from_result(
            (result.center_ra_deg, result.center_dec_deg), result.cd_matrix, frame, epoch
        )
        if initial is None:
            failed(
                REJECTED,
                f"{solver.name} solved a field that the catalog does not cover",
                result.elapsed_s,
            )
            return None
        readout = self._profile.mode(frame.mode)
        rows, vectors = catalog_field(
            self._catalog, initial, epoch, width_px=readout.width_px, height_px=readout.height_px
        )
        fit = fit_attitude(initial, vectors, x, y, error, shape=shape, options=self._fit_options)
        if fit is None or not self._acceptable(fit, int(x.size)):
            failed(REJECTED, f"the fit failed after {solver.name} solved", result.elapsed_s)
            return None
        attempts.append(
            SolveAttempt(
                solver.name, SOLVED, stars, result.elapsed_s, matched=fit.n_matched, hint=hint
            )
        )
        return _Solved(fit, rows, vectors, solver.name, result.elapsed_s)

    def _cross_check(
        self,
        solved_with: PlateSolver,
        request: SolveRequest,
        outcome: _Solved,
        frame: Frame,
        epoch: apparent.ObservationEpoch,
        shape: tuple[int, int],
        notes: list[str],
    ) -> str | None:
        """Run another solver on the same stars and compare its solution with the fit."""
        others = [solver for solver in self._solvers if solver is not solved_with]
        if not others:
            return None
        other = others[0]
        try:
            result = other.solve(request)
        except SolverError as exc:
            notes.append(f"the check with {other.name} failed: {exc}")
            return f"{other.name} failed"
        if not result.solved or result.cd_matrix is None or result.center_ra_deg is None:
            return f"{other.name} found no solution"
        assert result.center_dec_deg is not None
        attitude = self._attitude_from_result(
            (result.center_ra_deg, result.center_dec_deg), result.cd_matrix, frame, epoch
        )
        if attitude is None:
            return f"{other.name} solved another field"
        disagreement = field_disagreement_px(attitude, outcome.fit.attitude, shape)
        if disagreement > self._config.solve.cross_check_max_px:
            notes.append(f"{other.name} disagrees with the fit by {disagreement:.2f} px")
        return f"{other.name} {disagreement:.2f} px"

    # --- Matching, focus, clouds ---------------------------------------------------------

    def _match_all(
        self,
        attitude: CameraAttitude,
        detections: Detections,
        rows: npt.NDArray[np.intp],
        vectors: FloatArray,
        shape: tuple[int, int],
        *,
        fitted: bool,
    ) -> npt.NDArray[np.intp]:
        """Match all detections to the catalog. Returns the catalog row of each (-1: none)."""
        cfg = self._config.fit
        wide = 3.0 if fitted else 8.0
        det_index, cat_index, _ = match_stars(
            attitude,
            vectors,
            np.column_stack([detections.x, detections.y]),
            wide,
            shape,
            8.0,
            np.ones(vectors.shape[0], dtype=bool),
        )
        cat_row = np.full(len(detections), -1, dtype=np.intp)
        if det_index.size:
            px, py, _ = attitude.project(vectors[cat_index])
            distance = np.hypot(detections.x[det_index] - px, detections.y[det_index] - py)
            unreliable = detections.has(UNRELIABLE)[det_index]
            limit = np.where(unreliable, wide, min(cfg.final_match_radius_px, wide))
            good = distance <= limit
            cat_row[det_index[good]] = rows[cat_index[good]]
        return cat_row

    @staticmethod
    def _focus(detections: Detections) -> float | None:
        """The median FWHM of the stars that are neither saturated nor otherwise unreliable."""
        usable = detections.reliable() & (detections.snr > 10.0)
        if int(usable.sum()) < 3:
            return None
        return float(np.median(detections.fwhm_px[usable]))

    def _clear_sky_electrons(
        self, frame: Frame, g_mag: FloatArray, zp_reference: ZeroPointReference | None
    ) -> FloatArray | None:
        """The electrons that a clear sky gives a star of each G magnitude in the frame.

        The signal comes from the reference zero point when the history has one, and from the
        profile's photometric prior otherwise. Without either, the result is `None`.
        """
        exposure_s = frame.exposure_us / 1e6
        if zp_reference is not None:
            rate = 10.0 ** (0.4 * (zp_reference.zero_point_mag - g_mag))
        elif self._profile.photometry is not None:
            rate = np.array([self._profile.star_electron_rate_e_per_s(float(g)) for g in g_mag])
        else:
            return None
        return np.asarray(rate * exposure_s, dtype=np.float64)

    def _cloud_count(
        self,
        frame: Frame,
        detections: Detections,
        coverage: FieldStars | None,
        zp_reference: ZeroPointReference | None,
        image: _StarImage,
    ) -> CloudCount:
        """The expected catalog stars, how many detection found, and the share that it missed.

        The expected stars are those of the field that a clear sky would show at the
        signal-to-noise ratio of `CloudConfig.expected_snr` in an aperture, and that the search
        that ran finds with a chance of at least `CloudConfig.min_completeness`
        (`seeingmon.survey.completeness`), each star with the trail of its position and the share
        of its light in the fitted image (`_StarImage.core_fraction`). When SEP took the noise map
        (`SearchSpec.relative`), each star also takes the noise at its position, because SEP then
        thresholds each pixel by its own noise. So a frame never expects a star that its detector
        cannot find.
        """
        cfg = self._config.cloud
        if coverage is None or coverage.rows.size == 0:
            return CloudCount()
        g_mag = coverage.g_mag
        electrons = self._clear_sky_electrons(frame, g_mag, zp_reference)
        if electrons is None:
            return CloudCount()
        e_per_adu = self._profile.e_per_adu(frame.mode, frame.gain)
        noise_e = detections.background_rms * e_per_adu
        length, angle = image.trails(coverage.x, coverage.y)
        expected = (g_mag < cfg.mag_limit) & (
            electrons >= image.aperture_minimum_e(cfg, noise_e, length)
        )
        search = detections.search
        candidates = np.flatnonzero(expected)
        if search is not None and candidates.size:
            local: FloatArray | float = noise_e
            if search.relative and detections.noise_map is not None:
                x, y = coverage.x[candidates], coverage.y[candidates]
                local = detections.noise_map.at(x, y) * e_per_adu
            found = image.completeness(
                search, electrons[candidates], local, length[candidates], angle[candidates]
            )
            expected[candidates[found < cfg.min_completeness]] = False
        n_expected = int(expected.sum())
        n_found = int(coverage.found[expected].sum())
        if n_expected < cfg.min_expected:
            return CloudCount(None, n_expected, n_found)
        return CloudCount(1.0 - n_found / n_expected, n_expected, n_found)

    def _full_search_due(
        self,
        frame: Frame,
        detections: Detections,
        zp_reference: ZeroPointReference | None,
        core_of: Callable[[Detections], float | None],
    ) -> bool:
        """Whether the binned search misses stars that the cloud fraction would expect.

        The rule looks at a typical star of the field, with the median trail and direction of the
        detections (`_StarImage`). The faintest such star that the cloud fraction can expect has
        the highest of three fluxes: that of `CloudConfig.mag_limit`, the clear-sky signal of
        `expected_snr`, and the flux at which the full search finds a star with the chance
        `min_completeness`. When the binned search finds a star of that flux with a smaller
        chance, the stars between the limits of the two searches would drop out of the cloud
        fraction, so the frame takes the full search. The model takes the noise, the star image,
        and the share of the light in the fitted image (`core_of`, see `light_in_core`) that the
        binned search measured, so the rule follows the sky and the exposure, and never the Sun. A
        dark frame of 30 s keeps the binned search, because its trail spreads a sharp star across
        the blocks and the stars down to `mag_limit` are bright. A short frame in a bright sky takes
        the full search. The stars near the pole, whose trails are short, can still lie beyond the
        binned search when a typical star does not, and the cloud fraction then leaves them out
        (`_cloud_count`).
        """
        search = detections.search
        if search is None or not search.binned:
            return False
        cfg = self._config.cloud
        faintest = self._clear_sky_electrons(frame, np.array([cfg.mag_limit]), zp_reference)
        if faintest is None:
            return False  # no clear-sky signal, so no cloud fraction
        noise_e = detections.background_rms * self._profile.e_per_adu(frame.mode, frame.gain)
        image = _StarImage.of(detections, core_fraction=core_of(detections))
        typical = image.trail_px, image.trail_angle_rad
        full_limit = image.limit_e(
            replace(search, factor=1), cfg.min_completeness, noise_e, *typical
        )
        flux = max(
            float(faintest[0]),
            float(image.aperture_minimum_e(cfg, noise_e, image.trail_px)),
            float(full_limit),
        )
        if not math.isfinite(flux):
            return False  # the full search cannot reach the limit either
        found = image.completeness(search, flux, noise_e, *typical)
        return float(found) < cfg.min_completeness

    # --- Records -------------------------------------------------------------------------

    def _provenance(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        values = {
            "algo": POINTING_ALGORITHM,
            "catalog": self._catalog.content_id,
            "detector": f"sep-{sep.__version__}",
        }
        values.update(extra or {})
        return values

    def _records(
        self,
        frame: Frame,
        detections: Detections,
        solved: _Solved | None,
        solution: PointingSolution | None,
        epoch: apparent.ObservationEpoch,
        reference: ReferenceSolution | None,
        cat_row: npt.NDArray[np.intp],
        focus: float | None,
        sky_quality: SkyQualityRecord | None,
        *,
        pointing: bool = True,
    ) -> tuple[Record, ...]:
        base: dict[str, Any] = {
            "station_id": self._station_id,
            "t_utc_ns": frame.t_utc_ns,
            "profile_id": self._profile.id,
        }
        survey_frame = SurveyFrameRecord(
            **base,
            provenance=self._provenance(),
            exposure_s=frame.exposure_us / 1e6,
            gain=frame.gain,
            readout_mode=frame.mode,
            sensor_temperature_c=frame.temperature_c,
            n_detected=len(detections),
            n_saturated=int(detections.has(StarFlag.SATURATED).sum()),
            background_dn=float(detections.background_level),
        )
        time_invalid = frame_time_invalid(frame)
        polaris = None if solution is None else solution.polaris_pixel(frame.t_utc_ns)
        provenance = (
            {}
            if solved is None or solved.cross_check is None
            else {"cross_check": solved.cross_check}
        )
        pointing_record = build_pointing_record(
            station_id=self._station_id,
            profile_id=self._profile.id,
            t_utc_ns=frame.t_utc_ns,
            mode=frame.mode,
            solver="none" if solved is None else solved.solver,
            provenance=self._provenance(provenance),
            attitude=None if solved is None else solved.fit.attitude,
            epoch=None if solved is None else epoch,
            solution=solution,
            n_matched=0 if solved is None else solved.fit.n_matched,
            rms_arcsec=None if solved is None else solved.fit.rms_arcsec,
            focus_fwhm_px=focus,
            solve_time_s=None if solved is None else solved.solve_time_s,
            polaris_xy=polaris,
            reference=reference,
            time_invalid=time_invalid,
            limits=self._limits,
        )
        records: list[Record] = [survey_frame]
        if sky_quality is not None:
            records.append(sky_quality)
        if pointing:
            records.append(pointing_record)
        if solved is not None:
            records.append(self._star_list(frame, detections, cat_row, base))
        return tuple(records)

    def _star_list(
        self,
        frame: Frame,
        detections: Detections,
        cat_row: npt.NDArray[np.intp],
        base: dict[str, Any],
    ) -> StarListRecord:
        """The matched stars brighter than G = 11 and all unmatched detections."""
        matched = cat_row >= 0
        bright = np.zeros(len(detections), dtype=bool)
        bright[matched] = self._catalog.g_mag[cat_row[matched]] < STAR_LIST_G_LIMIT
        keep = bright | ~matched
        rows = np.column_stack(
            [
                detections.x[keep],
                detections.y[keep],
                detections.flux[keep],
                detections.fwhm_px[keep],
                detections.flags[keep].astype(np.float64),
                np.where(matched[keep], cat_row[keep], _UNMATCHED_ROW),
            ]
        )
        return StarListRecord(
            **base,
            provenance=self._provenance(),
            n_stars=int(keep.sum()),
            columns=list(STAR_LIST_COLUMNS),
            data=pack_star_rows(rows),
            catalog=STAR_LIST_CATALOG,
        )

    def _failure(
        self,
        frame: Frame,
        reference: ReferenceSolution | None,
        reason: str,
        timings: dict[str, float],
        *,
        saturated_sky: bool = False,
        sky_quality: bool | None = None,
    ) -> FrameAnalysis:
        """The records of a frame that could not be processed at all.

        A frame that saturates whole, which detection cannot process, carries `saturated_sky`. A
        short frame gets neither the `sky_quality` record (unless `sky_quality` forces it) nor the
        `unsolved` pointing record: no solve was tried (see "No attempt" in the module text).
        """
        log.warning("survey frame at %d: %s", frame.t_utc_ns, reason)
        long_frame = frame.exposure_us / 1e6 >= self._config.sky.min_exposure_s
        return FrameAnalysis(
            records=failure_records(
                station_id=self._station_id,
                profile_id=self._profile.id,
                t_utc_ns=frame.t_utc_ns,
                exposure_us=frame.exposure_us,
                gain=frame.gain,
                mode=frame.mode,
                temperature_c=frame.temperature_c,
                time_invalid=frame_time_invalid(frame),
                reference=reference,
                provenance=self._provenance(),
                reason=reason,
                saturated_sky=saturated_sky,
                sky_quality=long_frame if sky_quality is None else sky_quality,
                pointing=long_frame,
            ),
            solved=False,
            cloud_fraction=None,
            timings=timings,
            notes=(reason,),
        )


def failure_records(
    *,
    station_id: str,
    profile_id: str,
    t_utc_ns: int,
    exposure_us: int,
    gain: int,
    mode: str,
    temperature_c: float | None,
    time_invalid: bool,
    reference: ReferenceSolution | None,
    provenance: dict[str, str],
    reason: str,
    saturated_sky: bool = False,
    sky_quality: bool = True,
    pointing: bool = True,
) -> tuple[Record, ...]:
    """The survey_frame, empty sky_quality, and unsolved pointing records of a failed frame.

    `saturated_sky` adds that flag to the `sky_quality` record, for a frame that saturates whole.
    `sky_quality` and `pointing` false leave out those records, for a short frame.
    """
    flags = ["time_invalid"] if time_invalid else []
    if saturated_sky:
        flags.append("saturated_sky")
    records: list[Record] = [
        SurveyFrameRecord(
            station_id=station_id,
            t_utc_ns=t_utc_ns,
            profile_id=profile_id,
            provenance=provenance,
            quality={"n_detected": reason},
            exposure_s=exposure_us / 1e6,
            gain=gain,
            readout_mode=mode,
            sensor_temperature_c=temperature_c,
        )
    ]
    if sky_quality:
        records.append(
            SkyQualityRecord(
                station_id=station_id,
                t_utc_ns=t_utc_ns,
                profile_id=profile_id,
                provenance=provenance,
                quality={"sky_mag_arcsec2": reason, "zero_point_mag": reason},
                n_stars_used=0,
                flags=flags,
            )
        )
    if pointing:
        records.append(
            build_pointing_record(
                station_id=station_id,
                profile_id=profile_id,
                t_utc_ns=t_utc_ns,
                mode=mode,
                solver="none",
                provenance=provenance,
                attitude=None,
                epoch=None,
                solution=None,
                reference=reference,
                time_invalid=time_invalid,
            )
        )
    return tuple(records)


def field_disagreement_px(
    first: CameraAttitude, second: CameraAttitude, shape: tuple[int, int]
) -> float:
    """The largest pixel displacement between two attitudes over a grid across the frame."""
    height, width = shape
    gx, gy = np.meshgrid(np.linspace(0.0, width - 1.0, 7), np.linspace(0.0, height - 1.0, 5))
    vectors = first.unproject(gx.ravel(), gy.ravel())
    x, y, _ = second.project(vectors)
    return float(np.max(np.hypot(x - gx.ravel(), y - gy.ravel())))


def load_hot_pixels(path: str | Path) -> npt.NDArray[np.bool_]:
    """Read a hot-pixel mask from a NumPy file."""
    return np.asarray(np.load(path), dtype=np.bool_)
