"""One survey frame, from pixels to records.

`SurveyPipeline.analyze` runs the steps that turn a frame into the survey records:

1. **Detect.** `seeingmon.survey.detect` finds and measures the stars. When the tracker holds a
   recent solution, its trail model sets the trails.
2. **Solve.** The tracker comes first: it predicts the star field from the latest solution,
   matches it to the detections, and fits (`PointingTracker.track`). If that fails, the
   solver adapters run in the configured order, each on the star list, and the first solution
   starts the fit. The adapter's solution lives in the catalog frame, so the pipeline moves
   it to the apparent frame with the catalog stars of the field (`attitude_from_solver_solution`),
   and then fits against the apparent places (`fit_attitude`).
3. **Match.** With the final attitude, every detection is matched to the catalog, which gives
   the star list, the cloud fraction, and the matched stars for photometry later.
4. **Records.** The pipeline builds the `survey_frame`, `pointing`, and `star_list` records.

The pipeline holds no state between frames. The caller passes the previous solution and the
reference, and receives the new solution, so a worker process can run it and the parent can keep
the tracker. `PipelineSpec` describes a pipeline in plain data, so a worker process can build
its own copy.

**Failure.** A frame that cannot be solved is a normal result. The pointing record gets the
`unsolved` flag, and `quality` says why. Only an unexpected error propagates.

**Cloud fraction.** The expected stars are the catalog stars in the field that the profile's
photometric prior says a clear sky would show at a signal-to-noise ratio of
`CloudConfig.expected_snr` or better. The cloud fraction is the share of them that no detection
matches. Stars that a saturated star covers do not count.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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
from seeingmon.records.survey import StarListRecord, SurveyFrameRecord, pack_star_rows
from seeingmon.solvers.base import PlateSolver, SolveRequest, SolverError, StarList
from seeingmon.survey import apparent
from seeingmon.survey.catalog import CapCatalog, load_catalog
from seeingmon.survey.config import SurveyConfig
from seeingmon.survey.detect import (
    UNRELIABLE,
    DetectionError,
    Detections,
    DetectOptions,
    StarFlag,
    detect_stars,
)
from seeingmon.survey.field import catalog_field
from seeingmon.survey.geometry import FloatArray
from seeingmon.survey.pointing import (
    POINTING_ALGORITHM,
    PointingLimits,
    PointingSolution,
    ReferenceSolution,
    build_pointing_record,
)
from seeingmon.survey.tracker import PointingTracker
from seeingmon.survey.trail import TrailModel
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

STAR_LIST_COLUMNS = ["x_px", "y_px", "flux_dn", "fwhm_px", "flags", "cat_row"]
STAR_LIST_CATALOG = "gaia-dr3+tycho-2"
STAR_LIST_G_LIMIT = 11.0  # matched stars brighter than this go to the star list
_UNMATCHED_ROW = -1.0


@dataclass(frozen=True, slots=True)
class SolverSpec:
    """A plate solver in plain data. `kind` is `astrometry.net` or `astap`."""

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
        if name == "astrometry.net":
            specs.append(SolverSpec(name, config.solve_field_command, index_dir=config.index_dir))
        elif name == "astap":
            specs.append(
                SolverSpec(name, config.astap_command, database_dir=config.astap_database_dir)
            )
        else:
            raise ValueError(f"unknown solver {name!r}; use astrometry.net or astap")
    return tuple(specs)


def build_solvers(specs: tuple[SolverSpec, ...], clock: Clock | None = None) -> list[PlateSolver]:
    """Make the solver adapters that the specifications describe."""
    from seeingmon.solvers.astap import AstapSolver
    from seeingmon.solvers.astrometry_net import AstrometryNetSolver

    solvers: list[PlateSolver] = []
    for spec in specs:
        if spec.kind == "astrometry.net":
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
    return SurveyPipeline(
        station_id=spec.station_id,
        profile=profile,
        catalog=load_catalog(spec.catalog_path),
        solvers=build_solvers(spec.solvers, clock),
        config=config,
        hot_pixels=hot,
        clock=clock,
    )


@dataclass(frozen=True, slots=True, eq=False)
class FrameAnalysis:
    """The result of one frame.

    `solution` is the new pointing solution (`None` for an unsolved frame), which the caller
    gives to the tracker. `detections` stays in the process that ran the pipeline. `timings`
    maps each step to its time in seconds. `notes` explain a failure or a disagreement.
    """

    records: tuple[Record, ...]
    solved: bool
    cloud_fraction: float | None
    solution: PointingSolution | None = None
    detections: Detections | None = None
    fit: FitResult | None = None
    timings: dict[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class _Solved:
    fit: FitResult
    rows: npt.NDArray[np.intp]  # catalog rows of the field
    vectors: FloatArray  # their apparent unit vectors
    solver: str
    solve_time_s: float
    cross_check: str | None = None


def frame_time_invalid(frame: Frame) -> bool:
    """Whether the clock was not synchronized when the frame was taken."""
    return bool(frame.flags & FrameFlag.TIME_INVALID) or frame.t_quality == TimeQuality.INVALID


def native_counts(frame: Frame) -> npt.NDArray[np.float32]:
    """The frame in ADC counts: the high bits of a 16-bit container, or an 8-bit value scaled up."""
    scale = np.float32(2.0 ** (frame.adc_bits - frame.pixel_format.value))
    return np.asarray(frame.data.astype(np.float32) * scale, dtype=np.float32)


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
        clock: Clock | None = None,
    ) -> None:
        self._station_id = station_id
        self._profile = profile
        self._catalog = catalog
        self._solvers = list(solvers)
        self._config = config or SurveyConfig()
        self._hot = hot_pixels
        self._clock = clock or SystemClock()
        cfg = self._config
        self._detect_options = DetectOptions(
            threshold_sigma=cfg.detect.threshold_sigma,
            min_pixels=cfg.detect.min_pixels,
            mesh_px=cfg.detect.mesh_px,
            edge_margin_px=cfg.detect.edge_margin_px,
            max_stars=cfg.detect.max_stars,
            max_saturated_pixels=cfg.detect.max_saturated_pixels,
            trail_flag_px=cfg.detect.trail_flag_px,
        )
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
    ) -> FrameAnalysis:
        """Analyze a frame. `previous` is the latest solution, `reference` the saved one.

        `index` counts the frames that the caller has analyzed, and it picks the frames for the
        second-solver check.
        """
        timings: dict[str, float] = {}
        started = self._clock.monotonic_ns()
        notes: list[str] = []
        try:
            readout = self._profile.mode(frame.mode)
        except ProfileError:
            return self._failure(frame, reference, f"unknown readout mode {frame.mode!r}", timings)
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
        try:
            raw = detect_stars(
                native_counts(frame),
                saturation_dn=saturation.native_dn,
                options=self._detect_options,
                e_per_adu=self._profile.e_per_adu(frame.mode, frame.gain),
                hot_pixels=self._hot_mask_for(frame),
                trail=trail,
            )
        except DetectionError as error:
            return self._failure(frame, reference, f"detection failed: {error}", timings)
        detections = raw.shifted(frame.roi.x, frame.roi.y)
        lap("detect")

        epoch = apparent.epoch_from_utc_ns(frame.t_utc_ns, self._config.dut1_s)
        solved = self._solve(frame, detections, tracker, epoch, mode_shape, index, notes)
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

        cloud = self._cloud_fraction(
            frame,
            detections,
            match_attitude,
            field_rows,
            field_vectors,
            cat_row,
            readout.width_px,
            readout.height_px,
        )
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
        records = self._records(
            frame, detections, solved, solution, epoch, reference, cat_row, focus, notes
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
        )

    def _hot_mask_for(self, frame: Frame) -> npt.NDArray[np.bool_] | None:
        mask = self._hot
        if mask is None:
            return None
        if mask.shape == frame.shape:
            return mask
        roi = frame.roi
        window = mask[roi.y : roi.y_end, roi.x : roi.x_end]
        return window if window.shape == frame.shape else None

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
    ) -> _Solved | None:
        reliable = detections.reliable()
        x, y = detections.x[reliable], detections.y[reliable]
        error = 0.5 * (detections.x_error_px[reliable] + detections.y_error_px[reliable])
        t = frame.t_utc_ns
        if tracker.valid_at(t) and x.size:
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
        if len(stars) < 4:
            notes.append(f"only {len(stars)} stars for a solver")
            return None
        request = self._solve_request(frame, stars, tracker, epoch)
        for solver in self._solvers:
            outcome = self._run_solver(
                solver, request, frame, detections, x, y, error, epoch, shape, notes
            )
            if outcome is not None:
                every = self._config.solve.cross_check_every
                if every > 0 and index % every == 0:
                    outcome.cross_check = self._cross_check(
                        solver, request, outcome, frame, epoch, shape, notes
                    )
                return outcome
        return None

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
    ) -> SolveRequest:
        cfg = self._config.solve
        readout = self._profile.mode(frame.mode)
        scale = self._profile.plate_scale_arcsec_per_px(readout)
        hint_ra = hint_dec = radius = None
        predicted = tracker.attitude_at(frame.t_utc_ns)
        if predicted is not None:
            ra, dec = predicted.center_icrs(epoch)
            hint_ra, hint_dec, radius = ra, dec, cfg.hint_radius_deg
        order = np.argsort(-stars.flux, kind="stable")[: cfg.max_stars]
        return SolveRequest(
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
        frame: Frame,
        detections: Detections,
        x: FloatArray,
        y: FloatArray,
        error: FloatArray,
        epoch: apparent.ObservationEpoch,
        shape: tuple[int, int],
        notes: list[str],
    ) -> _Solved | None:
        try:
            result = solver.solve(request)
        except SolverError as exc:
            notes.append(f"{solver.name} failed: {exc}")
            log.warning("the %s solver failed: %s", solver.name, exc)
            return None
        if not result.solved or result.cd_matrix is None or result.center_ra_deg is None:
            notes.append(f"{solver.name} found no solution")
            return None
        assert result.center_dec_deg is not None
        initial = self._attitude_from_result(
            (result.center_ra_deg, result.center_dec_deg), result.cd_matrix, frame, epoch
        )
        if initial is None:
            notes.append(f"{solver.name} solved a field that the catalog does not cover")
            return None
        readout = self._profile.mode(frame.mode)
        rows, vectors = catalog_field(
            self._catalog, initial, epoch, width_px=readout.width_px, height_px=readout.height_px
        )
        fit = fit_attitude(initial, vectors, x, y, error, shape=shape, options=self._fit_options)
        if fit is None or not self._acceptable(fit, int(x.size)):
            notes.append(f"the fit failed after {solver.name} solved")
            return None
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

    def _cloud_fraction(
        self,
        frame: Frame,
        detections: Detections,
        attitude: CameraAttitude | None,
        rows: npt.NDArray[np.intp],
        vectors: FloatArray,
        cat_row: npt.NDArray[np.intp],
        width: int,
        height: int,
    ) -> float | None:
        """The share of expected catalog stars that detection missed, or `None`."""
        cfg = self._config.cloud
        photometry = self._profile.photometry
        if attitude is None or photometry is None or rows.size == 0:
            return None
        exposure_s = frame.exposure_us / 1e6
        x, y, front = attitude.project(vectors)
        roi = frame.roi
        edge = cfg.edge_px
        inside = (
            front
            & (x > roi.x + edge)
            & (x < roi.x_end - 1 - edge)
            & (y > roi.y + edge)
            & (y < roi.y_end - 1 - edge)
        )
        g_mag = self._catalog.g_mag[rows]
        rate = np.array([self._profile.star_electron_rate_e_per_s(float(g)) for g in g_mag])
        electrons = rate * exposure_s
        noise_e = detections.background_rms * self._profile.e_per_adu(frame.mode, frame.gain)
        fwhm = (
            float(np.median(detections.fwhm_px[detections.reliable()]))
            if np.any(detections.reliable())
            else 1.1
        )
        trail = float(np.median(detections.trail_length_px)) if len(detections) else 0.0
        aperture = np.pi * (1.5 * max(fwhm, 0.8)) ** 2 + 2.0 * trail * max(fwhm, 0.8)
        snr2 = cfg.expected_snr**2
        minimum = 0.5 * (snr2 + np.sqrt(snr2**2 + 4.0 * snr2 * aperture * noise_e**2))
        expected = inside & (g_mag < cfg.mag_limit) & (electrons >= minimum)
        # A saturated star hides the stars inside its blob.
        saturated = detections.has(StarFlag.SATURATED)
        if saturated.any():
            reach = 2.0 * np.sqrt(np.maximum(detections.n_pixels[saturated], 1))
            for sx, sy, r in zip(
                detections.x[saturated], detections.y[saturated], reach, strict=True
            ):
                expected &= np.hypot(x - sx, y - sy) > r
        if int(expected.sum()) < cfg.min_expected:
            return None
        detected_rows = {int(row) for row in cat_row[cat_row >= 0]}
        found = np.array([int(row) in detected_rows for row in rows[expected]])
        # A catalog star near a detection that matched nothing else counts as seen, too.
        if (~found).any() and len(detections):
            distance = np.full(int((~found).sum()), np.inf)
            px, py = x[expected][~found], y[expected][~found]
            for i, (cx, cy) in enumerate(zip(px, py, strict=True)):
                distance[i] = float(np.min(np.hypot(detections.x - cx, detections.y - cy)))
            found[~found] = distance <= cfg.match_radius_px
        return float(np.clip(1.0 - found.mean(), 0.0, 1.0))

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
        notes: list[str],
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
        pointing = build_pointing_record(
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
        records: list[Record] = [survey_frame, pointing]
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
    ) -> FrameAnalysis:
        """The records of a frame that could not be processed at all."""
        log.warning("survey frame at %d: %s", frame.t_utc_ns, reason)
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
) -> tuple[Record, ...]:
    """The `survey_frame` and unsolved `pointing` records for a frame that failed entirely."""
    survey_frame = SurveyFrameRecord(
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
    pointing = build_pointing_record(
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
    return survey_frame, pointing


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
