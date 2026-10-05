"""The quick solve: synthetic frames, a mount that moves, and a frame that cannot be solved."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace

import numpy as np
import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")

from seeingmon.clock import VirtualClock
from seeingmon.frames import Frame, TimeQuality
from seeingmon.profile import Profile
from seeingmon.services.core.alignment.solve import (
    QuickSolution,
    QuickSolver,
    adopt,
    analyze_frame,
    focus_value,
    is_trusted,
)
from seeingmon.survey.catalog import CapCatalog
from seeingmon.survey.detect import Detections, StarFlag
from seeingmon.survey.geometry import ARCSEC_PER_RAD, exp_so3
from seeingmon.survey.pipeline import FrameAnalysis, SurveyPipeline
from seeingmon.survey.tracker import PointingTracker
from tests.survey import synth


@dataclass
class Scene:
    profile: Profile
    catalog: CapCatalog
    first: tuple[Frame, synth.SynthTruth]
    moved: tuple[Frame, synth.SynthTruth]
    shift_px: tuple[float, float]


def displacement(a: synth.SynthTruth, b: synth.SynthTruth) -> tuple[float, float]:
    """How far the stars that both frames show moved from the first frame to the second."""
    _, ia, ib = np.intersect1d(a.rows, b.rows, return_indices=True)
    return float(np.median(b.x[ib] - a.x[ia])), float(np.median(b.y[ib] - a.y[ia]))


@pytest.fixture(scope="module")
def scene() -> Scene:
    profile = synth.cropped_profile(1200, 800)
    catalog = synth.synthetic_catalog(cap_radius_deg=5.0, density_scale=3.0, seed=1)
    rotation = synth.make_attitude(0.9, 40.0, 25.0)
    first = synth.render_frame(
        catalog, profile, rotation_tirs=rotation, exposure_s=0.5, gain=60, seed=3, seq=1
    )
    scale_rad = first[1].scale_arcsec_px / ARCSEC_PER_RAD
    moved_rotation = exp_so3(np.array([3.5, -2.1, 0.0]) * scale_rad) @ rotation
    moved = synth.render_frame(
        catalog, profile, rotation_tirs=moved_rotation, exposure_s=0.5, gain=60, seed=4, seq=2
    )
    return Scene(profile, catalog, first, moved, displacement(first[1], moved[1]))


def quick_solver(
    scene: Scene, *, with_solver: bool = True
) -> tuple[QuickSolver, synth.QueueSolver]:
    truth = scene.first[1]
    queue = synth.QueueSolver(
        [synth.truth_solve_result(truth, scene.catalog)] if with_solver else []
    )
    pipeline = SurveyPipeline(
        station_id="test", profile=scene.profile, catalog=scene.catalog, solvers=[queue]
    )
    tracker = PointingTracker(scene.profile)
    return QuickSolver(pipeline, tracker, VirtualClock(synth.NIGHT_UTC_NS)), queue


def holds_untimed(solver: QuickSolver) -> bool:
    """Whether the tracker holds an untimed solution (a function, so mypy does not narrow it)."""
    return solver.tracker.untimed


class TestSolving:
    def test_the_first_frame_goes_to_the_solver_and_starts_the_tracker(self, scene: Scene) -> None:
        solver, queue = quick_solver(scene)
        solution = solver.solve(scene.first[0])
        assert solution.solved
        assert solution.solver == "synthetic"
        assert solution.n_matched >= 20
        assert solution.rms_arcsec is not None
        assert solution.rms_arcsec < 1.0
        assert solution.x_px is not None
        assert solution.y_px is not None
        assert solution.scale_arcsec_px == pytest.approx(scene.first[1].scale_arcsec_px, rel=1e-3)
        assert len(queue.requests) == 1
        assert solver.tracker.solution is not None  # the next frame can skip the solver
        assert solver.solves == 1

    def test_the_solution_carries_the_attitude_and_the_colatitude_of_polaris(
        self, scene: Scene
    ) -> None:
        solver, _ = quick_solver(scene)
        frame = scene.first[0]
        solution = solver.solve(frame)
        assert solution.attitude is not None
        assert solution.x_px is not None
        assert solution.y_px is not None
        adopted = solver.tracker.solution
        assert adopted is not None
        # The attitude is the one that the tracker holds for the time of the frame.
        expected = adopted.attitude_at(frame.t_utc_ns)
        assert solution.attitude.rotation == pytest.approx(expected.rotation, abs=1e-12)
        assert solution.attitude.center_px == expected.center_px
        # The distance from the pole to Polaris in the frame is the colatitude, to the
        # difference between a tangent and an angle.
        pole = solution.attitude.pole_pixel()
        assert pole is not None
        assert solution.polaris_colatitude_deg is not None
        assert solution.scale_arcsec_px is not None
        distance_px = math.hypot(solution.x_px - pole[0], solution.y_px - pole[1])
        distance_deg = distance_px * solution.scale_arcsec_px / 3600.0
        assert solution.polaris_colatitude_deg == pytest.approx(distance_deg, rel=2e-3)
        assert 0.55 < solution.polaris_colatitude_deg < 0.70

    def test_a_moved_mount_shows_as_the_displacement_of_the_field_through_the_tracker(
        self, scene: Scene
    ) -> None:
        solver, queue = quick_solver(scene)
        before = solver.solve(scene.first[0])
        after = solver.solve(scene.moved[0])
        assert after.solved
        assert after.solver == "tracker"  # a few tens of milliseconds, with no solver
        assert len(queue.requests) == 1  # the plate solver ran for the first frame only
        assert before.x_px is not None and after.x_px is not None  # noqa: PT018
        assert before.y_px is not None and after.y_px is not None  # noqa: PT018
        dx, dy = scene.shift_px
        assert math.hypot(dx, dy) > 4.0  # the scenario moves the mount by a few pixels
        assert after.x_px - before.x_px == pytest.approx(dx, abs=0.15)
        assert after.y_px - before.y_px == pytest.approx(dy, abs=0.15)

    def test_the_focus_value_is_the_width_of_the_stars(self, scene: Scene) -> None:
        solver, _ = quick_solver(scene)
        solution = solver.solve(scene.first[0])
        assert solution.n_focus_stars >= 10
        assert solution.focus_fwhm_px is not None
        # A Gaussian with sigma 0.45 pixel has a FWHM of 1.06 pixels. The model fit measures the
        # width of the PSF, and the pixel integration widens it a little.
        assert 0.8 < solution.focus_fwhm_px < 2.0

    def test_a_frame_without_stars_is_a_normal_result_that_leaves_the_tracker_alone(
        self, scene: Scene
    ) -> None:
        solver, _ = quick_solver(scene, with_solver=False)
        blank = synth.render_frame(
            scene.catalog,
            scene.profile,
            rotation_tirs=synth.make_attitude(0.9, 40.0, 25.0),
            exposure_s=0.5,
            gain=60,
            seed=9,
            transmission=0.0,
        )[0]
        solution = solver.solve(blank)
        assert not solution.solved
        assert solution.note
        assert solution.x_px is None
        assert solution.attitude is None
        assert solution.polaris_colatitude_deg is None
        assert solver.tracker.solution is None
        assert solver.failures == 1

    def test_a_failed_solve_after_a_good_one_leaves_the_good_solution_in_the_tracker(
        self, scene: Scene
    ) -> None:
        solver, _ = quick_solver(scene)
        assert solver.solve(scene.first[0]).solved
        good = solver.tracker.solution
        assert good is not None
        blank = synth.render_frame(
            scene.catalog,
            scene.profile,
            rotation_tirs=synth.make_attitude(0.9, 40.0, 25.0),
            t_utc_ns=synth.NIGHT_UTC_NS + 60_000_000_000,
            exposure_s=0.5,
            gain=60,
            seed=9,
            transmission=0.0,
        )[0]
        assert not solver.solve(blank).solved
        assert solver.tracker.solution is good
        assert (solver.solves, solver.failures) == (1, 1)

    def test_a_frame_without_a_valid_time_fills_an_empty_tracker_and_never_replaces_a_timed_one(
        self, scene: Scene
    ) -> None:
        """Its solution could lie in the future, so only a timed solution can replace it."""
        solver, _ = quick_solver(scene)
        untimed = replace(scene.first[0], t_quality=TimeQuality.INVALID)
        solution = solver.solve(untimed)
        assert solution.solved
        assert solution.note == ""
        assert holds_untimed(solver)
        assert solver.solve(scene.first[0]).solved  # the same frame with a valid time
        timed = solver.tracker.solution
        assert timed is not None
        assert not holds_untimed(solver)
        solution = solver.solve(untimed)
        assert solution.solved  # the live view still shows where Polaris is
        assert solution.note == (
            "the clock is not synchronized, so the solution does not move the tracker"
        )
        assert solver.tracker.solution is timed

    def test_an_alignment_frame_never_asks_for_the_sky_quality(self, scene: Scene) -> None:
        asked: list[object] = []

        class Spy:
            def __init__(self, inner: object) -> None:
                self._inner = inner

            def analyze(self, frame: Frame, **options: object) -> FrameAnalysis:
                asked.append(options.get("sky_quality"))
                return self._inner.analyze(frame, **options)  # type: ignore[attr-defined, no-any-return]

        solver, _ = quick_solver(scene)
        solver._pipeline = Spy(solver._pipeline)
        solver.solve(scene.first[0])
        assert asked == [False]  # even a long frame would get no zero point and no sky brightness

    def test_the_analysis_that_raises_becomes_an_unsolved_result(self, scene: Scene) -> None:
        class Broken:
            def analyze(self, frame: Frame, **options: object) -> FrameAnalysis:
                raise MemoryError("not enough memory")

        solver = QuickSolver(
            Broken(), PointingTracker(scene.profile), VirtualClock(synth.NIGHT_UTC_NS)
        )
        solution = solver.solve(scene.first[0])
        assert not solution.solved
        assert solution.note == "analysis error: MemoryError"
        assert (solution.t_utc_ns, solution.seq) == (scene.first[0].t_utc_ns, 1)


class TestPlainData:
    """The solution crosses to another process as plain data, and the trust rule stays in `core`."""

    def test_a_solved_frame_survives_the_trip_as_plain_data(self, scene: Scene) -> None:
        solver, _ = quick_solver(scene)
        solution = solver.solve(scene.first[0])
        assert solution.solved
        data = solution.to_dict()
        assert json.loads(json.dumps(data)) == data  # numbers, text, and lists only
        again = QuickSolution.from_dict(data)
        assert again.attitude is not None
        assert solution.attitude is not None
        assert again.attitude.rotation == pytest.approx(solution.attitude.rotation, abs=1e-15)
        assert (again.attitude.scale_rad_px, again.attitude.parity, again.attitude.center_px) == (
            solution.attitude.scale_rad_px,
            solution.attitude.parity,
            solution.attitude.center_px,
        )
        assert again.to_dict() == data

    def test_an_unsolved_frame_survives_it_without_an_attitude(self) -> None:
        unsolved = QuickSolution(
            123, 4, False, n_detected=2, focus_fwhm_px=2.5, elapsed_s=0.4, note="too few stars"
        )
        again = QuickSolution.from_dict(json.loads(json.dumps(unsolved.to_dict())))
        assert again == unsolved

    def test_data_that_lacks_a_field_is_refused(self) -> None:
        data = QuickSolution(1, 2, False).to_dict()
        del data["note"]
        with pytest.raises(KeyError):
            QuickSolution.from_dict(data)

    def test_the_analysis_hands_the_pointing_solution_to_the_caller_and_not_to_the_tracker(
        self, scene: Scene
    ) -> None:
        solver, _ = quick_solver(scene)
        analysis = analyze_frame(
            solver._pipeline,
            scene.first[0],
            previous=None,
            reference=None,
            index=0,
            clock=VirtualClock(synth.NIGHT_UTC_NS),
        )
        assert analysis.solution.solved
        assert analysis.pointing is not None
        assert solver.tracker.solution is None  # `adopt` decides, and nothing adopted it yet
        solution = adopt(analysis, solver.tracker, min_stars=8, max_rms_px=1.5)
        assert solution == analysis.solution
        assert solver.tracker.solution is analysis.pointing

    def test_a_weak_fit_does_not_move_the_tracker(self, scene: Scene) -> None:
        solver, _ = quick_solver(scene)
        analysis = analyze_frame(
            solver._pipeline,
            scene.first[0],
            previous=None,
            reference=None,
            index=0,
            clock=VirtualClock(synth.NIGHT_UTC_NS),
        )
        assert analysis.pointing is not None
        weak = adopt(analysis, solver.tracker, min_stars=100_000, max_rms_px=1.5)
        assert weak.note == "the fit is too weak to move the tracker"
        assert solver.tracker.solution is None
        loose = adopt(analysis, solver.tracker, min_stars=8, max_rms_px=1e-6)
        assert loose.note == "the fit is too weak to move the tracker"
        assert solver.tracker.solution is None

    def test_the_trust_rule_wants_enough_stars_and_a_small_residual(self, scene: Scene) -> None:
        solver, _ = quick_solver(scene)
        analysis = analyze_frame(
            solver._pipeline,
            scene.first[0],
            previous=None,
            reference=None,
            index=0,
            clock=VirtualClock(synth.NIGHT_UTC_NS),
        )
        pointing = analysis.pointing
        assert pointing is not None
        assert is_trusted(pointing, min_stars=pointing.n_matched, max_rms_px=10.0)
        assert not is_trusted(pointing, min_stars=pointing.n_matched + 1, max_rms_px=10.0)
        assert not is_trusted(pointing, min_stars=1, max_rms_px=1e-9)


class TestFocusValue:
    def detections(self, fwhm: list[float], flags: list[int], snr: list[float]) -> Detections:
        n = len(fwhm)
        zeros = np.zeros(n)
        return Detections(
            shape=(10, 10),
            x=zeros.copy(),
            y=zeros.copy(),
            flux=np.ones(n),
            peak=np.ones(n),
            fwhm_px=np.array(fwhm),
            x_error_px=zeros.copy(),
            y_error_px=zeros.copy(),
            elongation=np.ones(n),
            trail_length_px=zeros.copy(),
            trail_angle_rad=zeros.copy(),
            flags=np.array(flags, dtype=np.uint16),
            snr=np.array(snr),
            n_pixels=np.ones(n, dtype=np.int32),
            background_level=0.0,
            background_rms=1.0,
        )

    def test_the_median_of_the_usable_stars(self) -> None:
        saturated = int(StarFlag.SATURATED)
        det = self.detections(
            [2.0, 2.2, 2.4, 9.0, 8.0, 2.1],
            [0, 0, 0, saturated, 0, int(StarFlag.TRAILED)],
            [50, 50, 50, 50, 5, 50],  # the fifth star is too faint, and the fourth saturated
        )
        value, count = focus_value(det)
        assert count == 4  # the trailed star counts: the width is measured across the trail
        assert value == pytest.approx(np.median([2.0, 2.2, 2.4, 2.1]))

    def test_too_few_usable_stars_give_no_value(self) -> None:
        det = self.detections([2.0, 2.0], [0, 0], [50, 50])
        assert focus_value(det) == (None, 2)
        assert focus_value(None) == (None, 0)
