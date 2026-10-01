"""The quick solve: synthetic frames, a mount that moves, and a frame that cannot be solved."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")

from seeingmon.clock import VirtualClock
from seeingmon.frames import Frame
from seeingmon.profile import Profile
from seeingmon.services.core.alignment.solve import QuickSolver, focus_value
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
        assert solver.tracker.solution is None
        assert solver.failures == 1

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
