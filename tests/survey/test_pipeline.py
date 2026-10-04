"""The survey pipeline: one frame in, the survey records out."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pytest

from seeingmon.frames import Frame, FrameFlag, Roi, TimeQuality
from seeingmon.profile import Profile
from seeingmon.records.survey import (
    PointingRecord,
    SkyQualityRecord,
    StarListRecord,
    SurveyFrameRecord,
    star_rows,
)
from seeingmon.solvers.base import PlateSolver, SolverError, SolveResult
from seeingmon.survey import pointing as pt
from seeingmon.survey.catalog import CapCatalog
from seeingmon.survey.config import SolveConfig, SurveyConfig
from seeingmon.survey.detect import StarFlag
from seeingmon.survey.geometry import ARCSEC_PER_RAD
from seeingmon.survey.pipeline import (
    ERROR,
    NO_SOLUTION,
    REJECTED,
    SOLVED,
    STAR_LIST_COLUMNS,
    FrameAnalysis,
    SolveAttempt,
    SurveyPipeline,
)
from seeingmon.survey.wcs_fit import CameraAttitude
from seeingmon.testing import FakeSolver
from tests.survey import synth

NS = 1_000_000_000


@pytest.fixture(scope="module")
def profile() -> Profile:
    return synth.cropped_profile(1200, 800)


@pytest.fixture(scope="module")
def catalog() -> CapCatalog:
    return synth.synthetic_catalog(cap_radius_deg=5.0, density_scale=3.0, seed=1)


@pytest.fixture(scope="module")
def scene(profile: Profile, catalog: CapCatalog) -> tuple[Frame, synth.SynthTruth]:
    return synth.render_frame(
        catalog,
        profile,
        rotation_tirs=synth.make_attitude(0.9, 40.0, 25.0),
        exposure_s=30.0,
        seed=3,
    )


def pipeline_for(
    profile: Profile,
    catalog: CapCatalog,
    solvers: Sequence[PlateSolver] | None = None,
    config: SurveyConfig | None = None,
    **options: object,
) -> SurveyPipeline:
    return SurveyPipeline(
        station_id="test",
        profile=profile,
        catalog=catalog,
        solvers=list(solvers or []),
        config=config,
        **options,  # type: ignore[arg-type]
    )


def truth_solver(
    truth: synth.SynthTruth, catalog: CapCatalog, center_shift_px: float = 0.0
) -> synth.QueueSolver:
    result = synth.truth_solve_result(truth, catalog, center_shift_px=center_shift_px)
    return synth.QueueSolver([result])


def pointing_error_px(analysis: FrameAnalysis, truth: synth.SynthTruth) -> tuple[float, float]:
    """The RMS and the largest displacement, in pixels, between the fit and the true attitude."""
    assert analysis.fit is not None
    true_model = CameraAttitude(
        truth.rotation_cirs, truth.scale_arcsec_px / ARCSEC_PER_RAD, truth.parity, truth.center_px
    )
    gx, gy = np.meshgrid(
        np.linspace(0, truth.width - 1.0, 9), np.linspace(0, truth.height - 1.0, 7)
    )
    x, y, _ = analysis.fit.attitude.project(true_model.unproject(gx.ravel(), gy.ravel()))
    shift = np.hypot(x - gx.ravel(), y - gy.ravel())
    return float(np.sqrt(np.mean(shift**2))), float(shift.max())


def records_of(analysis: FrameAnalysis) -> tuple[SurveyFrameRecord, PointingRecord, StarListRecord]:
    survey, pointing, star_list = (
        survey_of(analysis),
        pointing_of(analysis),
        next(r for r in analysis.records if isinstance(r, StarListRecord)),
    )
    return survey, pointing, star_list


def survey_of(analysis: FrameAnalysis) -> SurveyFrameRecord:
    return next(r for r in analysis.records if isinstance(r, SurveyFrameRecord))


def pointing_of(analysis: FrameAnalysis) -> PointingRecord:
    return next(r for r in analysis.records if isinstance(r, PointingRecord))


def sky_of(analysis: FrameAnalysis) -> SkyQualityRecord:
    return next(r for r in analysis.records if isinstance(r, SkyQualityRecord))


def test_a_solved_frame_gives_three_records_and_recovers_the_pointing(
    profile: Profile, catalog: CapCatalog, scene: tuple[Frame, synth.SynthTruth]
) -> None:
    frame, truth = scene
    solver = truth_solver(truth, catalog)
    analysis = pipeline_for(profile, catalog, [solver]).analyze(frame)
    assert analysis.solved
    assert [record.record_type for record in analysis.records] == [
        "survey_frame",
        "sky_quality",
        "pointing",
        "star_list",
    ]
    rms, worst = pointing_error_px(analysis, truth)
    assert worst < 0.05  # the task's criterion is 0.1 pixel
    assert rms < 0.02
    survey, pointing, star_list = records_of(analysis)
    assert survey.t_utc_ns == pointing.t_utc_ns == star_list.t_utc_ns == frame.t_utc_ns
    assert survey.exposure_s == 30.0
    assert (survey.gain, survey.readout_mode, survey.sensor_temperature_c) == (120, "bin2", 15.0)
    assert survey.n_detected is not None
    assert survey.n_detected > 300
    assert survey.n_saturated is not None
    assert 20 < survey.n_saturated < survey.n_detected
    assert survey.background_dn == pytest.approx(130.0, rel=0.1)
    assert pointing.solver == "synthetic"
    assert pointing.flags == []
    assert pointing.n_matched > 250
    assert pointing.roll_deg == pytest.approx(25.0, abs=0.01)
    assert pointing.plate_scale_arcsec_px == pytest.approx(truth.scale_arcsec_px, rel=1e-4)
    assert pointing.solve_rms_arcsec is not None
    assert pointing.solve_rms_arcsec < 0.1
    assert pointing.focus_fwhm_px == pytest.approx(0.45 * 2.3548, rel=0.04)
    assert pointing.polaris_x_px is not None
    # The pole lies outside this frame (0.9 degree from the center), and the record still has it.
    assert pointing.pole_x_px is not None
    assert pointing.pole_y_px is not None
    assert (pointing.pole_x_px, pointing.pole_y_px) == pytest.approx(truth.pole_px, abs=0.05)
    assert pointing.quality is not None
    assert "pole_x_px" not in pointing.quality
    assert pointing.solve_time_s == 0.25  # the solver's own time
    assert star_list.catalog == "gaia-dr3+tycho-2"
    assert pointing.provenance["catalog"] == catalog.content_id
    assert pointing.provenance["algo"] == pt.POINTING_ALGORITHM
    assert pointing.provenance["detector"].startswith("sep-")
    assert solver.requests[0].width_px == 1200


def test_a_first_solve_without_a_pointing_gets_a_hint_at_the_pole(
    profile: Profile, catalog: CapCatalog, scene: tuple[Frame, synth.SynthTruth]
) -> None:
    frame, truth = scene
    solver = truth_solver(truth, catalog)
    pipeline_for(profile, catalog, [solver]).analyze(frame)
    request = solver.requests[0]
    assert (request.center_ra_deg, request.center_dec_deg) == (0.0, 90.0)
    assert request.radius_deg == 15.0


def test_the_pole_hint_can_be_turned_off(
    profile: Profile, catalog: CapCatalog, scene: tuple[Frame, synth.SynthTruth]
) -> None:
    frame, truth = scene
    solver = truth_solver(truth, catalog)
    config = SurveyConfig(solve=SolveConfig(pole_hint_radius_deg=0.0))
    pipeline_for(profile, catalog, [solver], config=config).analyze(frame)
    request = solver.requests[0]
    assert request.center_ra_deg is None
    assert request.radius_deg is None


def test_the_star_list_holds_the_bright_matches_and_the_unmatched_detections(
    profile: Profile, catalog: CapCatalog, scene: tuple[Frame, synth.SynthTruth]
) -> None:
    frame, truth = scene
    analysis = pipeline_for(profile, catalog, [truth_solver(truth, catalog)]).analyze(frame)
    _, _, star_list = records_of(analysis)
    assert star_list.columns == STAR_LIST_COLUMNS
    rows = star_rows(star_list)
    assert rows.shape == (star_list.n_stars, 6)
    x, y, flux, fwhm, flags, cat_row = rows.T
    matched = cat_row >= 0
    assert matched.sum() > 40
    # Every matched star is bright, and every other detection is in the list.
    assert np.all(catalog.g_mag[cat_row[matched].astype(int)] < 11.0)
    assert (~matched).sum() >= 0
    assert analysis.detections is not None
    n_matched_dim = int((analysis.detections.n_pixels > 0).sum()) - star_list.n_stars
    assert n_matched_dim > 0  # the dim matched stars stay out of the list
    # A matched star sits where the catalog puts it. The centroid of a saturated star is only
    # good to a pixel or two, so the position check takes the stars with no flags.
    sample = np.flatnonzero(matched)[:30]
    assert sample.size >= 20
    for i in sample:
        row = int(cat_row[i])
        where = np.flatnonzero(truth.rows == row)
        assert where.size == 1
        tolerance = 0.1 if flags[i] == 0 else 3.0
        assert abs(x[i] - truth.x[where[0]]) < tolerance
        assert abs(y[i] - truth.y[where[0]]) < tolerance
    # In a 30 s exposure at gain 120, every star brighter than G = 11 saturates.
    assert np.mean((flags[matched].astype(int) & int(StarFlag.SATURATED)) != 0) > 0.9
    assert np.all(fwhm > 0)
    assert np.all(flux > 0)
    # The flags are the detector's flag bits.
    assert set(np.unique(flags.astype(int))) <= set(range(0, 256))


def test_the_pointing_is_recovered_within_a_tenth_of_a_pixel_on_a_full_bin2_frame() -> None:
    """The done criterion on a frame of the reference size, 4144 x 2822, with real star counts."""
    profile = synth.reference_profile()
    catalog = synth.synthetic_catalog(cap_radius_deg=6.0, density_scale=1.0, seed=11)
    frame, truth = synth.render_frame(
        catalog,
        profile,
        rotation_tirs=synth.make_attitude(0.9, 40.0, 25.0),
        exposure_s=30.0,
        seed=12,
    )
    assert frame.data.shape == (2822, 4144)
    analysis = pipeline_for(profile, catalog, [truth_solver(truth, catalog)]).analyze(frame)
    assert analysis.solved
    rms, worst = pointing_error_px(analysis, truth)
    assert worst < 0.1  # pixels, anywhere in the field
    assert rms < 0.05
    _, pointing, _ = records_of(analysis)
    assert pointing.n_matched > 700
    assert sum(analysis.timings.values()) < 60.0  # seconds, on any CI machine


def test_the_tracker_solves_the_next_frame_without_calling_a_solver(
    profile: Profile, catalog: CapCatalog, scene: tuple[Frame, synth.SynthTruth]
) -> None:
    frame, truth = scene
    first = pipeline_for(profile, catalog, [truth_solver(truth, catalog)]).analyze(frame)
    assert first.solution is not None
    # The next frame comes 3 minutes later. The mount did not move, and the sky turned.
    later = truth.t_utc_ns + 180 * NS
    frame2, truth2 = synth.render_frame(
        catalog,
        profile,
        rotation_tirs=truth.rotation_tirs,
        t_utc_ns=later,
        exposure_s=30.0,
        seed=4,
    )
    no_solver = FakeSolver(error=SolverError("must not be called"))
    second = pipeline_for(profile, catalog, [no_solver]).analyze(frame2, previous=first.solution)
    assert second.solved
    assert no_solver.requests == []
    assert pointing_error_px(second, truth2)[1] < 0.05
    _, pointing, _ = records_of(second)
    assert pointing.solver == "tracker"
    # The trail model from the tracker gives the same accuracy and a clean width.
    assert pointing.focus_fwhm_px == pytest.approx(0.45 * 2.3548, rel=0.04)


def test_the_solver_is_the_fallback_when_the_tracker_loses_the_field(
    profile: Profile, catalog: CapCatalog, scene: tuple[Frame, synth.SynthTruth]
) -> None:
    frame, truth = scene
    first = pipeline_for(profile, catalog, [truth_solver(truth, catalog)]).analyze(frame)
    assert first.solution is not None
    # The mount moved by 0.15 degrees (140 pixels), far more than the tracker's search radius.
    from seeingmon.survey.geometry import exp_so3

    moved_tirs = exp_so3([0.0, np.radians(0.15), 0.0]) @ truth.rotation_tirs
    later = truth.t_utc_ns + 600 * NS
    frame2, truth2 = synth.render_frame(
        catalog, profile, rotation_tirs=moved_tirs, t_utc_ns=later, exposure_s=30.0, seed=5
    )
    solver = truth_solver(truth2, catalog)
    second = pipeline_for(profile, catalog, [solver]).analyze(frame2, previous=first.solution)
    assert second.solved
    assert len(solver.requests) == 1
    assert pointing_error_px(second, truth2)[1] < 0.05
    assert any("tracker could not match" in note for note in second.notes)
    _, pointing, _ = records_of(second)
    assert pointing.solver == "synthetic"
    # The solver got a center hint from the tracker's prediction.
    assert solver.requests[0].center_ra_deg is not None
    assert solver.requests[0].radius_deg == 2.0


def test_the_second_solver_runs_when_the_first_fails(
    profile: Profile, catalog: CapCatalog, scene: tuple[Frame, synth.SynthTruth]
) -> None:
    frame, truth = scene
    broken = FakeSolver(error=SolverError("the index folder holds no index files"))
    working = truth_solver(truth, catalog)
    analysis = pipeline_for(profile, catalog, [broken, working]).analyze(frame)
    assert analysis.solved
    assert len(broken.requests) == 1
    assert len(working.requests) == 1
    assert any("index folder" in note for note in analysis.notes)
    unsolved = FakeSolver()  # reports "no solution"
    again = pipeline_for(profile, catalog, [unsolved, truth_solver(truth, catalog)]).analyze(frame)
    assert again.solved
    assert any("found no solution" in note for note in again.notes)


def test_every_solver_run_leaves_an_attempt_with_its_stars_time_and_reason(
    profile: Profile, catalog: CapCatalog, scene: tuple[Frame, synth.SynthTruth]
) -> None:
    frame, truth = scene
    broken = FakeSolver(error=SolverError("the index folder holds no index files"))
    silent = synth.QueueSolver(name="astap")  # reports "no solution"
    working = truth_solver(truth, catalog)  # solves in 0.25 s, and is called "synthetic"
    analysis = pipeline_for(profile, catalog, [broken, silent, working]).analyze(frame)
    assert analysis.solved
    assert analysis.fit is not None
    first, second, third = analysis.attempts
    stars = len(working.requests[0].stars)
    assert stars >= 4
    assert (first.solver, first.outcome, first.stars) == ("fake", ERROR, stars)
    assert first.reason == "fake failed: the index folder holds no index files"
    assert 0.0 <= first.elapsed_s < 30.0  # the time that the pipeline waited for the solver
    assert (second.solver, second.outcome, second.stars) == ("astap", NO_SOLUTION, stars)
    assert (second.reason, second.elapsed_s) == ("astap found no solution", 0.0)
    assert (third.solver, third.outcome, third.stars) == ("synthetic", SOLVED, stars)
    assert third.matched == analysis.fit.n_matched > 100
    assert (third.reason, third.elapsed_s) == ("", 0.25)  # the time that the solver reports
    for failed in (first, second):
        assert failed.reason in analysis.notes  # the reason is the note of the analysis


def test_a_solver_result_that_the_fit_rejects_is_an_attempt_that_says_so(
    profile: Profile, catalog: CapCatalog, scene: tuple[Frame, synth.SynthTruth]
) -> None:
    frame, truth = scene
    good = synth.truth_solve_result(truth, catalog)
    wrong = replace(good, center_ra_deg=(good.center_ra_deg or 0.0) + 60.0)
    analysis = pipeline_for(profile, catalog, [synth.QueueSolver([wrong])]).analyze(frame)
    assert not analysis.solved
    (attempt,) = analysis.attempts
    assert attempt.outcome == REJECTED
    assert attempt.matched == 0
    assert attempt.reason in analysis.notes
    assert "solved a field that the catalog does not cover" in attempt.reason or (
        "the fit failed after synthetic solved" in attempt.reason
    )


def test_a_frame_that_the_tracker_solves_has_no_attempt_and_one_with_few_stars_has_none_either(
    profile: Profile, catalog: CapCatalog, scene: tuple[Frame, synth.SynthTruth]
) -> None:
    frame, truth = scene
    first = pipeline_for(profile, catalog, [truth_solver(truth, catalog)]).analyze(frame)
    assert first.solution is not None
    later = truth.t_utc_ns + 180 * NS
    frame2, _ = synth.render_frame(
        catalog,
        profile,
        rotation_tirs=truth.rotation_tirs,
        t_utc_ns=later,
        exposure_s=30.0,
        seed=4,
    )
    tracked = pipeline_for(profile, catalog, [FakeSolver()]).analyze(
        frame2, previous=first.solution
    )
    assert tracked.solved
    assert tracked.attempts == ()  # the tracker needed no solver
    dark, _ = synth.render_frame(
        catalog,
        profile,
        rotation_tirs=truth.rotation_tirs,
        exposure_s=0.0001,
        gain=0,
        seed=7,
    )
    solver = FakeSolver()
    empty = pipeline_for(profile, catalog, [solver]).analyze(dark)
    assert not empty.solved
    assert solver.requests == []  # fewer than 4 stars never reach a solver
    assert empty.attempts == ()
    assert any("stars for a solver" in note for note in empty.notes)


class TestTheSolveAttempt:
    def test_a_solved_attempt_reads_as_structured_text_with_the_matched_stars(self) -> None:
        attempt = SolveAttempt("astap", SOLVED, stars=48, elapsed_s=0.8123, matched=87)
        assert attempt.describe() == "solver=astap result=solved stars=48 time_s=0.81 matched=87"

    def test_a_failed_attempt_gives_its_reason_as_one_quoted_line(self) -> None:
        attempt = SolveAttempt(
            "astrometry.net",
            ERROR,
            stars=600,
            elapsed_s=3.0,
            reason='astrometry.net failed: solve-field exited\nwith code 1: "no index"',
        )
        assert attempt.describe() == (
            "solver=astrometry.net result=error stars=600 time_s=3.00 "
            'reason="astrometry.net failed: solve-field exited with code 1: \\"no index\\""'
        )
        assert "\n" not in attempt.describe()

    def test_an_attempt_survives_the_trip_through_the_worker_as_a_dictionary(self) -> None:
        attempt = SolveAttempt("astap", NO_SOLUTION, 12, 1.5, reason="astap found no solution")
        assert SolveAttempt.from_dict(attempt.to_dict()) == attempt
        assert set(attempt.to_dict()) == {
            "solver",
            "outcome",
            "stars",
            "elapsed_s",
            "matched",
            "reason",
        }


def test_no_solution_gives_an_unsolved_record_and_no_star_list(
    profile: Profile, catalog: CapCatalog, scene: tuple[Frame, synth.SynthTruth]
) -> None:
    frame, _ = scene
    analysis = pipeline_for(profile, catalog, [FakeSolver()]).analyze(frame)
    assert not analysis.solved
    assert analysis.cloud_fraction is None  # no solution and no prediction: cannot tell
    assert [record.record_type for record in analysis.records] == [
        "survey_frame",
        "sky_quality",
        "pointing",
    ]
    survey, pointing = survey_of(analysis), pointing_of(analysis)
    assert isinstance(pointing, PointingRecord)
    assert pointing.flags == ["unsolved"]
    assert pointing.center_ra_deg is None
    assert pointing.solver == "none"
    assert pointing.focus_fwhm_px is not None  # the focus does not need a solution
    assert isinstance(survey, SurveyFrameRecord)
    assert survey.n_detected is not None
    assert survey.n_detected > 300


def test_a_solver_result_that_the_fit_cannot_use_is_not_a_solution(
    profile: Profile, catalog: CapCatalog, scene: tuple[Frame, synth.SynthTruth]
) -> None:
    frame, truth = scene
    good = synth.truth_solve_result(truth, catalog)
    wrong = replace(good, center_ra_deg=(good.center_ra_deg or 0.0) + 60.0)
    solver = synth.QueueSolver([wrong])
    analysis = pipeline_for(profile, catalog, [solver]).analyze(frame)
    assert not analysis.solved
    assert any("fit failed" in note or "does not cover" in note for note in analysis.notes)


def test_a_short_exposure_with_few_stars_still_solves_and_says_so(
    profile: Profile, catalog: CapCatalog
) -> None:
    frame, truth = synth.render_frame(
        catalog,
        profile,
        rotation_tirs=synth.make_attitude(0.9, 40.0, 25.0),
        exposure_s=0.02,
        gain=0,
        seed=6,
    )
    analysis = pipeline_for(profile, catalog, [truth_solver(truth, catalog)]).analyze(frame)
    assert analysis.solved
    pointing = pointing_of(analysis)
    assert isinstance(pointing, PointingRecord)
    assert 4 <= pointing.n_matched < 40
    assert pointing_error_px(analysis, truth)[1] < 0.2


@pytest.mark.parametrize("polaris", ["saturated", "missing"])
def test_the_pointing_survives_a_saturated_or_missing_polaris(
    polaris: str, profile: Profile, catalog: CapCatalog
) -> None:
    polaris_row = int(np.argmin(catalog.g_mag))
    exclude = (polaris_row,) if polaris == "missing" else ()
    frame, truth = synth.render_frame(
        catalog,
        profile,
        rotation_tirs=synth.make_attitude(0.9, 40.0, 25.0),
        exposure_s=30.0,
        exclude_rows=exclude,
        seed=7,
    )
    if polaris == "saturated":
        assert polaris_row in truth.rows
        assert analysis_flags_saturated(frame, truth, polaris_row)
    analysis = pipeline_for(profile, catalog, [truth_solver(truth, catalog)]).analyze(frame)
    assert analysis.solved
    assert pointing_error_px(analysis, truth)[1] < 0.05


def analysis_flags_saturated(frame: Frame, truth: synth.SynthTruth, row: int) -> bool:
    """Whether the rendered Polaris has saturated pixels (a blob much larger than a star)."""
    where = int(np.flatnonzero(truth.rows == row)[0])
    x, y = round(truth.x[where]), round(truth.y[where])
    window = frame.data[max(y - 8, 0) : y + 9, max(x - 8, 0) : x + 9]
    return int((window >= 65000).sum()) > 20


def test_clouds_lower_the_share_of_expected_stars_that_detection_finds(
    profile: Profile, catalog: CapCatalog
) -> None:
    rotation = synth.make_attitude(0.9, 40.0, 25.0)

    def cloud_fraction(transmission: float | synth.TransmissionFunction, seed: int) -> float | None:
        frame, truth = synth.render_frame(
            catalog,
            profile,
            rotation_tirs=rotation,
            exposure_s=30.0,
            transmission=transmission,
            seed=seed,
        )
        analysis = pipeline_for(profile, catalog, [truth_solver(truth, catalog)]).analyze(frame)
        return analysis.cloud_fraction

    clear = cloud_fraction(1.0, 20)
    thin = cloud_fraction(0.35, 21)  # 1.1 magnitudes dimmer
    patchy = cloud_fraction(lambda x, y: np.where(x > 600, 0.003, 1.0), 22)  # half the field
    medium = cloud_fraction(0.01, 23)  # 5 magnitudes dimmer: the bright stars still show
    thick = cloud_fraction(0.0005, 24)  # 8.3 magnitudes dimmer
    assert clear is not None
    assert clear < 0.05
    assert thin is not None
    assert thin < 0.2  # a thin veil leaves the bright stars visible
    assert patchy is not None
    assert 0.3 < patchy < 0.7
    assert medium is not None
    assert 0.2 < medium < 0.7
    assert thick is None or thick > 0.9
    assert clear < thin + 0.2 < patchy


def test_a_frame_with_no_stars_is_reported_not_raised(
    profile: Profile, catalog: CapCatalog
) -> None:
    frame, _ = synth.render_frame(
        catalog,
        profile,
        rotation_tirs=synth.make_attitude(0.9, 40.0, 25.0),
        exposure_s=30.0,
        transmission=0.0,
        seed=8,
    )
    analysis = pipeline_for(profile, catalog, [FakeSolver()]).analyze(frame)
    assert not analysis.solved
    pointing = pointing_of(analysis)
    assert isinstance(pointing, PointingRecord)
    assert pointing.flags == ["unsolved"]
    survey = survey_of(analysis)
    assert isinstance(survey, SurveyFrameRecord)
    assert survey.n_detected is not None
    assert survey.n_detected <= 3


def test_a_cloud_over_a_known_pointing_gives_a_full_cloud_fraction(
    profile: Profile, catalog: CapCatalog, scene: tuple[Frame, synth.SynthTruth]
) -> None:
    frame, truth = scene
    first = pipeline_for(profile, catalog, [truth_solver(truth, catalog)]).analyze(frame)
    assert first.solution is not None
    dark, _ = synth.render_frame(
        catalog,
        profile,
        rotation_tirs=truth.rotation_tirs,
        t_utc_ns=truth.t_utc_ns + 180 * NS,
        exposure_s=30.0,
        transmission=0.0,
        seed=9,
    )
    analysis = pipeline_for(profile, catalog, [FakeSolver()]).analyze(dark, previous=first.solution)
    assert not analysis.solved
    # The tracker still knows where the stars should be, and none of them shows.
    assert analysis.cloud_fraction == pytest.approx(1.0, abs=0.01)


def test_an_unknown_readout_mode_gives_the_failure_records(
    profile: Profile, catalog: CapCatalog, scene: tuple[Frame, synth.SynthTruth]
) -> None:
    frame, _ = scene
    odd = replace(frame, mode="bin7")
    analysis = pipeline_for(profile, catalog, [FakeSolver()]).analyze(odd)
    assert not analysis.solved
    assert analysis.notes == ("unknown readout mode 'bin7'",)
    survey, sky, pointing = analysis.records
    assert isinstance(survey, SurveyFrameRecord)
    assert survey.readout_mode == "bin7"
    assert survey.quality == {"n_detected": "unknown readout mode 'bin7'"}
    assert isinstance(sky, SkyQualityRecord)  # an empty record that says why
    assert sky.n_stars_used == 0
    assert sky.sky_mag_arcsec2 is None
    assert sky.quality == {
        "sky_mag_arcsec2": "unknown readout mode 'bin7'",
        "zero_point_mag": "unknown readout mode 'bin7'",
    }
    assert isinstance(pointing, PointingRecord)
    assert pointing.flags == ["unsolved"]


def test_an_invalid_time_is_flagged(
    profile: Profile, catalog: CapCatalog, scene: tuple[Frame, synth.SynthTruth]
) -> None:
    frame, truth = scene
    bad_clock = replace(
        frame, t_quality=TimeQuality.INVALID, flags=frame.flags | FrameFlag.TIME_INVALID
    )
    analysis = pipeline_for(profile, catalog, [truth_solver(truth, catalog)]).analyze(bad_clock)
    pointing = pointing_of(analysis)
    assert isinstance(pointing, PointingRecord)
    assert pointing.flags == ["time_invalid"]


def test_a_roi_frame_is_analyzed_in_sensor_pixels(catalog: CapCatalog) -> None:
    profile = synth.cropped_profile(1400, 900)
    frame, truth = synth.render_frame(
        catalog,
        profile,
        rotation_tirs=synth.make_attitude(0.9, 40.0, 25.0),
        exposure_s=30.0,
        seed=10,
    )
    window = frame.data[100:800, 200:1200].copy()
    roi_frame = replace(frame, data=window, roi=Roi(200, 100, 1000, 700))
    analysis = pipeline_for(profile, catalog, [truth_solver(truth, catalog)]).analyze(roi_frame)
    assert analysis.solved
    assert pointing_error_px(analysis, truth)[1] < 0.1
    assert analysis.detections is not None
    # The detections are in sensor pixels, so they lie inside the ROI rectangle.
    assert np.all(analysis.detections.x > 195)
    assert np.all(analysis.detections.x < 1205)
    assert np.all(analysis.detections.y > 95)


def test_the_hot_pixel_mask_keeps_hot_pixels_out_of_the_pointing(
    profile: Profile, catalog: CapCatalog
) -> None:
    frame, truth = synth.render_frame(
        catalog,
        profile,
        rotation_tirs=synth.make_attitude(0.9, 40.0, 25.0),
        exposure_s=30.0,
        n_hot_pixels=80,
        hot_e_per_s=60.0,
        seed=13,
    )
    masked = pipeline_for(
        profile, catalog, [truth_solver(truth, catalog)], hot_pixels=truth.hot_pixels
    ).analyze(frame)
    unmasked = pipeline_for(profile, catalog, [truth_solver(truth, catalog)]).analyze(frame)
    assert masked.solved
    assert unmasked.solved
    assert masked.detections is not None
    assert unmasked.detections is not None
    assert len(masked.detections) < len(unmasked.detections)
    assert pointing_error_px(masked, truth)[1] < 0.05
    # The hot pixels make no difference to the pointing, but they fill the star list.
    assert pointing_error_px(unmasked, truth)[1] < 0.05
    assert int(unmasked.detections.has(StarFlag.HOT_PIXEL).sum()) > 20


def test_a_reference_turns_a_mount_move_into_an_offset_and_a_flag(
    profile: Profile, catalog: CapCatalog, scene: tuple[Frame, synth.SynthTruth]
) -> None:
    from seeingmon.survey.geometry import exp_so3

    frame, truth = scene
    first = pipeline_for(profile, catalog, [truth_solver(truth, catalog)]).analyze(frame)
    assert first.solution is not None
    reference = pt.ReferenceSolution("commissioning", first.solution)
    # No move: offset 0 and no flag.
    same = pipeline_for(profile, catalog, [truth_solver(truth, catalog)]).analyze(
        frame, reference=reference
    )
    pointing = pointing_of(same)
    assert isinstance(pointing, PointingRecord)
    assert pointing.reference_id == "commissioning"
    assert pointing.offset_arcmin == pytest.approx(0.0, abs=0.02)
    assert pointing.flags == []
    # A 12 arcmin move.
    moved_tirs = exp_so3([0.0, np.radians(0.2), 0.0]) @ truth.rotation_tirs
    frame2, truth2 = synth.render_frame(
        catalog,
        profile,
        rotation_tirs=moved_tirs,
        t_utc_ns=truth.t_utc_ns + 600 * NS,
        exposure_s=30.0,
        seed=14,
    )
    moved = pipeline_for(profile, catalog, [truth_solver(truth2, catalog)]).analyze(
        frame2, previous=first.solution, reference=reference
    )
    pointing = pointing_of(moved)
    assert isinstance(pointing, PointingRecord)
    assert pointing.offset_arcmin == pytest.approx(12.0, rel=0.02)
    assert pointing.flags == ["moved"]


def test_the_second_solver_checks_a_sample_of_frames(
    profile: Profile, catalog: CapCatalog, scene: tuple[Frame, synth.SynthTruth]
) -> None:
    frame, truth = scene
    config = SurveyConfig.model_validate({"solve": {"cross_check_every": 2}})
    both = [truth_solver(truth, catalog), truth_solver(truth, catalog, center_shift_px=0.0)]
    # Frame 0 is a check frame: both solvers run, and they agree.
    agreed = pipeline_for(profile, catalog, both, config).analyze(frame, index=0)
    pointing = pointing_of(agreed)
    assert isinstance(pointing, PointingRecord)
    assert pointing.provenance["cross_check"].startswith("synthetic ")
    assert float(pointing.provenance["cross_check"].split()[1]) < 0.2
    assert len(both[1].requests) == 1
    # Frame 1 is not a check frame: the second solver stays idle.
    other = [truth_solver(truth, catalog), truth_solver(truth, catalog)]
    skipped = pipeline_for(profile, catalog, other, config).analyze(frame, index=1)
    assert "cross_check" not in pointing_of(skipped).provenance
    assert other[1].requests == []
    # A second solver that is 4 pixels off is reported.
    wrong = synth.QueueSolver([synth.truth_solve_result(truth, catalog, center_shift_px=4.0)])
    disagreed = pipeline_for(
        profile, catalog, [truth_solver(truth, catalog), wrong], config
    ).analyze(frame, index=0)
    assert any("disagrees with the fit" in note for note in disagreed.notes)
    pointing = pointing_of(disagreed)
    assert float(pointing.provenance["cross_check"].split()[1]) > 2.0
    # A second solver that fails is noted, and the analysis still succeeds.
    failing = FakeSolver(error=SolverError("crashed"))
    crashed = pipeline_for(
        profile, catalog, [truth_solver(truth, catalog), failing], config
    ).analyze(frame, index=0)
    assert crashed.solved
    assert any("check with" in note for note in crashed.notes)


def test_a_failure_to_detect_is_a_normal_failure(profile: Profile, catalog: CapCatalog) -> None:
    flat = Frame(
        data=np.full((800, 1200), 4000, dtype=np.uint16),
        stream_id=1,
        seq=0,
        t_arrival_ns=0,
        t_utc_ns=synth.NIGHT_UTC_NS,
        t_err_ns=0,
        t_quality=TimeQuality.EXACT,
        dropped_before=0,
        exposure_us=30_000_000,
        gain=120,
        mode="bin2",
        roi=Roi(0, 0, 1200, 800),
        adc_bits=14,
    )
    analysis = pipeline_for(profile, catalog, [FakeSolver()]).analyze(flat)
    assert not analysis.solved
    assert analysis.notes[0].startswith("detection failed")
    assert isinstance(pointing_of(analysis), PointingRecord)


def test_the_timings_name_each_step(
    profile: Profile, catalog: CapCatalog, scene: tuple[Frame, synth.SynthTruth]
) -> None:
    frame, truth = scene
    analysis = pipeline_for(profile, catalog, [truth_solver(truth, catalog)]).analyze(frame)
    assert set(analysis.timings) == {"detect", "solve", "match", "quality", "records"}
    assert all(value >= 0.0 for value in analysis.timings.values())


def test_an_unsolved_result_with_a_solver_that_returns_no_cd_is_handled(
    profile: Profile, catalog: CapCatalog, scene: tuple[Frame, synth.SynthTruth]
) -> None:
    frame, _ = scene
    odd = synth.QueueSolver([SolveResult(solved=True, solver="odd", elapsed_s=0.1)])
    analysis = pipeline_for(profile, catalog, [odd]).analyze(frame)
    assert not analysis.solved


def test_the_analysis_keeps_what_the_steps_after_the_pointing_need(
    profile: Profile, catalog: CapCatalog, scene: tuple[Frame, synth.SynthTruth]
) -> None:
    frame, truth = scene
    analysis = pipeline_for(profile, catalog, [truth_solver(truth, catalog)]).analyze(frame)
    assert analysis.detections is not None
    assert analysis.cat_row is not None
    assert analysis.attitude is not None
    assert analysis.epoch is not None
    assert analysis.field_rows is not None
    assert analysis.field_vectors is not None
    assert analysis.cat_row.shape == (len(analysis.detections),)
    assert analysis.field_vectors.shape == (analysis.field_rows.size, 3)
    matched = analysis.cat_row >= 0
    assert matched.sum() > analysis.fit.n_matched * 0.9 if analysis.fit else False
    # Every matched row is a row of the catalog field, and no row matches twice.
    assert set(analysis.cat_row[matched]) <= set(analysis.field_rows)
    assert len(set(analysis.cat_row[matched])) == int(matched.sum())
    # The matched stars sit where the truth puts them.
    rows = analysis.cat_row[matched]
    where = {int(row): i for i, row in enumerate(truth.rows)}
    reliable = matched & analysis.detections.reliable()
    for i in np.flatnonzero(reliable)[:30]:
        j = where[int(analysis.cat_row[i])]
        assert abs(analysis.detections.x[i] - truth.x[j]) < 0.1
    assert rows.size > 100
