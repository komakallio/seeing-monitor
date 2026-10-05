"""The quick solve in a worker process: the round trip, the failures, and the life of the worker."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from concurrent.futures import BrokenExecutor, Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")

from seeingmon.clock import SystemClock, VirtualClock
from seeingmon.frames import Frame, FrameFlag
from seeingmon.profile import Profile
from seeingmon.services.core.alignment.solve import QuickSolver
from seeingmon.services.core.alignment.worker import (
    ExecutorFactory,
    ProcessQuickSolver,
    install_pipeline,
    make_quick_executor,
    run_quick_job,
    warm_up,
)
from seeingmon.survey.catalog import CapCatalog, write_catalog
from seeingmon.survey.config import SurveyConfig
from seeingmon.survey.geometry import ARCSEC_PER_RAD, exp_so3
from seeingmon.survey.pipeline import PipelineSpec, SurveyPipeline
from seeingmon.survey.tracker import PointingTracker
from tests.survey import synth


@dataclass
class Scene:
    profile: Profile
    catalog: CapCatalog
    first: Frame
    second: Frame
    truth: synth.SynthTruth


@pytest.fixture(scope="module")
def scene() -> Scene:
    profile = synth.cropped_profile(1200, 800)
    catalog = synth.synthetic_catalog(cap_radius_deg=5.0, density_scale=3.0, seed=1)
    rotation = synth.make_attitude(0.9, 40.0, 25.0)
    first, truth = synth.render_frame(
        catalog, profile, rotation_tirs=rotation, exposure_s=0.5, gain=60, seed=3, seq=1
    )
    shift = np.array([3.5, -2.1, 0.0]) * truth.scale_arcsec_px / ARCSEC_PER_RAD
    second, _ = synth.render_frame(
        catalog,
        profile,
        rotation_tirs=exp_so3(shift) @ rotation,
        exposure_s=0.5,
        gain=60,
        seed=4,
        seq=2,
    )
    return Scene(profile, catalog, first, second, truth)


@pytest.fixture(autouse=True)
def clean_worker_state() -> Iterator[None]:
    """A thread executor installs its pipeline in this process: take it away afterwards."""
    yield
    install_pipeline(None)


def pipeline_without_solver(scene: Scene) -> SurveyPipeline:
    return SurveyPipeline(
        station_id="test", profile=scene.profile, catalog=scene.catalog, solvers=[]
    )


def seeded_tracker(scene: Scene) -> PointingTracker:
    """A tracker with the solution of the first frame, so the next solve needs no plate solver."""
    queue = synth.QueueSolver([synth.truth_solve_result(scene.truth, scene.catalog)])
    pipeline = SurveyPipeline(
        station_id="test", profile=scene.profile, catalog=scene.catalog, solvers=[queue]
    )
    tracker = PointingTracker(scene.profile)
    first = QuickSolver(pipeline, tracker, VirtualClock(synth.NIGHT_UTC_NS)).solve(scene.first)
    assert first.solved
    assert tracker.solution is not None
    return tracker


def copy_of(tracker: PointingTracker, scene: Scene) -> PointingTracker:
    other = PointingTracker(scene.profile)
    assert tracker.solution is not None
    other.update(tracker.solution)
    return other


def thread_factory(pipeline: SurveyPipeline) -> ExecutorFactory:
    """A worker that runs on a thread, so that a test needs no second process."""

    def make() -> Executor:
        return ThreadPoolExecutor(max_workers=1, initializer=install_pipeline, initargs=(pipeline,))

    return make


class TestTheWorkerStart:
    def test_the_worker_names_itself_lowers_its_priority_and_builds_its_pipeline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from seeingmon.services.core.alignment import worker

        calls: list[tuple[str, Any]] = []
        built = object()

        def record(label: str, result: Any = "x") -> Callable[..., Any]:
            def call(*args: Any) -> Any:
                calls.append((label, args[0]))
                return result

            return call

        monkeypatch.setattr(worker, "name_process", record("name"))
        monkeypatch.setattr(worker, "lower_process_priority", record("nice"))
        monkeypatch.setattr(worker, "raise_oom_score", record("oom"))
        monkeypatch.setattr(worker, "build_pipeline", record("build", built))
        spec = PipelineSpec(
            station_id="test", profile={}, config={}, catalog_path="none", solvers=()
        )
        worker.init_quick_worker(spec, 12, 600)
        # The name tells this worker from the survey worker, which names itself `smon-survey`.
        assert calls == [("name", "smon-align"), ("nice", 12), ("oom", 600), ("build", spec)]
        assert worker._PIPELINE is built


class TestTheRoundTrip:
    def test_the_worker_finds_what_the_same_solve_in_this_process_finds(self, scene: Scene) -> None:
        pipeline = pipeline_without_solver(scene)
        clock = VirtualClock(synth.NIGHT_UTC_NS)
        seeded = seeded_tracker(scene)
        expected = QuickSolver(pipeline, copy_of(seeded, scene), clock).solve(scene.second)
        tracker = copy_of(seeded, scene)
        solver = ProcessQuickSolver(tracker, clock, executor_factory=thread_factory(pipeline))
        try:
            got = solver.solve(scene.second)
        finally:
            solver.close()
        assert got.solved
        assert got.solver == "tracker"
        for field in ("x_px", "y_px", "roll_deg", "rms_arcsec", "focus_fwhm_px"):
            assert getattr(got, field) == pytest.approx(getattr(expected, field), abs=1e-9), field
        assert (got.seq, got.t_utc_ns, got.n_matched, got.n_detected) == (
            expected.seq,
            expected.t_utc_ns,
            expected.n_matched,
            expected.n_detected,
        )
        assert got.attitude is not None
        assert expected.attitude is not None
        assert got.attitude.rotation == pytest.approx(expected.attitude.rotation, abs=1e-12)
        assert got.polaris_colatitude_deg == expected.polaris_colatitude_deg
        assert (solver.solves, solver.failures) == (1, 0)

    def test_a_trusted_solution_updates_the_tracker_in_this_process(self, scene: Scene) -> None:
        pipeline = pipeline_without_solver(scene)
        tracker = copy_of(seeded_tracker(scene), scene)
        solver = ProcessQuickSolver(
            tracker, VirtualClock(synth.NIGHT_UTC_NS), executor_factory=thread_factory(pipeline)
        )
        try:
            solver.solve(scene.second)
        finally:
            solver.close()
        assert tracker.solution is not None
        assert tracker.solution.t_utc_ns == scene.second.t_utc_ns
        assert tracker.solution.solver == "tracker"

    def test_a_weak_solution_leaves_the_tracker_alone_and_says_so(self, scene: Scene) -> None:
        pipeline = pipeline_without_solver(scene)
        tracker = copy_of(seeded_tracker(scene), scene)
        before = tracker.solution
        solver = ProcessQuickSolver(
            tracker,
            VirtualClock(synth.NIGHT_UTC_NS),
            executor_factory=thread_factory(pipeline),
            min_stars=100_000,
        )
        try:
            solution = solver.solve(scene.second)
        finally:
            solver.close()
        assert solution.solved
        assert solution.note == "the fit is too weak to move the tracker"
        assert tracker.solution is before

    def test_a_frame_without_a_valid_time_leaves_the_tracker_alone_and_says_so(
        self, scene: Scene
    ) -> None:
        pipeline = pipeline_without_solver(scene)
        tracker = copy_of(seeded_tracker(scene), scene)
        before = tracker.solution
        solver = ProcessQuickSolver(
            tracker, VirtualClock(synth.NIGHT_UTC_NS), executor_factory=thread_factory(pipeline)
        )
        try:
            solution = solver.solve(replace(scene.second, flags=FrameFlag.TIME_INVALID))
        finally:
            solver.close()
        assert solution.solved
        assert solution.note == (
            "the clock is not synchronized, so the solution does not move the tracker"
        )
        assert tracker.solution is before

    def test_a_frame_without_stars_is_an_unsolved_result_and_the_worker_stays(
        self, scene: Scene
    ) -> None:
        blank = synth.render_frame(
            scene.catalog,
            scene.profile,
            rotation_tirs=synth.make_attitude(0.9, 40.0, 25.0),
            exposure_s=0.5,
            gain=60,
            seed=9,
            transmission=0.0,
        )[0]
        pipeline = pipeline_without_solver(scene)
        solver = ProcessQuickSolver(
            PointingTracker(scene.profile),
            VirtualClock(synth.NIGHT_UTC_NS),
            executor_factory=thread_factory(pipeline),
        )
        try:
            solution = solver.solve(blank)
            assert solver.running
        finally:
            solver.close()
        assert not solution.solved
        assert solution.note
        assert (solver.solves, solver.failures) == (0, 1)

    def test_the_job_refuses_to_run_without_a_pipeline(self, scene: Scene) -> None:
        with pytest.raises(RuntimeError, match="no pipeline"):
            run_quick_job(b"", None, None, 0)


class FakeExecutor(Executor):
    """An executor that stands in for a worker process, and records how it was stopped."""

    def __init__(self, behavior: Callable[[], Future[Any]]) -> None:
        self.behavior = behavior
        self.shutdowns: list[tuple[bool, bool]] = []

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        return self.behavior()

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self.shutdowns.append((wait, cancel_futures))


def failed_future(error: BaseException) -> Future[Any]:
    future: Future[Any] = Future()
    future.set_exception(error)
    return future


class Factory:
    """Makes `FakeExecutor`s with one behavior, and keeps them."""

    def __init__(self, behavior: Callable[[], Future[Any]]) -> None:
        self.behavior = behavior
        self.made: list[FakeExecutor] = []

    def __call__(self) -> Executor:
        executor = FakeExecutor(self.behavior)
        self.made.append(executor)
        return executor


class TestFailures:
    def test_a_pool_that_breaks_gives_an_unsolved_result_and_waits_before_the_next_try(
        self, scene: Scene
    ) -> None:
        clock = VirtualClock(synth.NIGHT_UTC_NS)
        factory = Factory(lambda: failed_future(BrokenExecutor("the worker died")))
        solver = ProcessQuickSolver(
            PointingTracker(scene.profile), clock, executor_factory=factory, retry_s=5.0
        )
        first = solver.solve(scene.first)
        assert not first.solved
        assert "stopped" in first.note
        assert factory.made[0].shutdowns == [(False, True)]  # ended at once
        assert not solver.running
        again = solver.solve(scene.first)  # too soon: no new worker
        assert again.note == first.note
        assert solver.workers_started == 1
        clock.advance(6.0)
        solver.solve(scene.first)
        assert solver.workers_started == 2
        assert solver.failures == 3

    def test_a_solve_that_takes_too_long_ends_the_worker(self, scene: Scene) -> None:
        factory = Factory(lambda: Future())  # a job that never finishes
        solver = ProcessQuickSolver(
            PointingTracker(scene.profile),
            VirtualClock(synth.NIGHT_UTC_NS),
            executor_factory=factory,
            timeout_s=0.05,
        )
        solution = solver.solve(scene.first)
        assert not solution.solved
        assert "took more than 0.05 s" in solution.note
        assert factory.made[0].shutdowns == [(False, True)]
        assert not solver.running

    def test_a_job_that_raises_is_an_unsolved_result_and_the_worker_stays(
        self, scene: Scene
    ) -> None:
        factory = Factory(lambda: failed_future(ValueError("no good")))
        solver = ProcessQuickSolver(
            PointingTracker(scene.profile),
            VirtualClock(synth.NIGHT_UTC_NS),
            executor_factory=factory,
        )
        solution = solver.solve(scene.first)
        assert solution.note == "analysis error: ValueError"
        solver.solve(scene.first)
        assert solver.workers_started == 1
        assert factory.made[0].shutdowns == []

    def test_an_answer_that_cannot_be_read_is_an_unsolved_result(self, scene: Scene) -> None:
        answer: Future[Any] = Future()
        answer.set_result({"solution": {"seq": 1}, "pointing": None})  # lacks most fields
        solver = ProcessQuickSolver(
            PointingTracker(scene.profile),
            VirtualClock(synth.NIGHT_UTC_NS),
            executor_factory=Factory(lambda: answer),
        )
        solution = solver.solve(scene.first)
        assert not solution.solved
        assert solution.note == "analysis error: KeyError"

    def test_a_job_that_the_release_cancelled_is_an_unsolved_result(self, scene: Scene) -> None:
        cancelled: Future[Any] = Future()
        cancelled.cancel()
        solver = ProcessQuickSolver(
            PointingTracker(scene.profile),
            VirtualClock(synth.NIGHT_UTC_NS),
            executor_factory=Factory(lambda: cancelled),
        )
        assert solver.solve(scene.first).note == "the solver stopped"


class TestLife:
    def test_the_worker_starts_at_the_first_solve_and_the_release_stops_it(
        self, scene: Scene
    ) -> None:
        made: list[Executor] = []
        pipeline = pipeline_without_solver(scene)

        def make() -> Executor:
            made.append(thread_factory(pipeline)())
            return made[-1]

        solver = ProcessQuickSolver(
            PointingTracker(scene.profile), VirtualClock(synth.NIGHT_UTC_NS), executor_factory=make
        )
        assert not solver.running
        assert made == []  # nothing starts before the first frame
        try:
            solver.solve(scene.first)
            states = [solver.running]
            solver.release()
            states.append(solver.running)
            solver.solve(scene.first)  # the next alignment starts a new worker
            assert states == [True, False]
            assert (len(made), solver.workers_started) == (2, 2)
        finally:
            solver.close()

    def test_the_prepare_starts_the_worker_before_the_first_solve(self, scene: Scene) -> None:
        made: list[Executor] = []
        pipeline = pipeline_without_solver(scene)

        def make() -> Executor:
            made.append(thread_factory(pipeline)())
            return made[-1]

        solver = ProcessQuickSolver(
            PointingTracker(scene.profile), VirtualClock(synth.NIGHT_UTC_NS), executor_factory=make
        )
        try:
            solver.prepare()
            running_after_prepare = solver.running
            solver.prepare()  # a second call finds the worker
            solver.solve(scene.first)
        finally:
            solver.close()
        assert running_after_prepare is True
        assert (len(made), solver.workers_started) == (1, 1)  # the solve used the same worker

    def test_the_warm_up_answers_once_the_pipeline_exists(self, scene: Scene) -> None:
        assert warm_up() is False
        install_pipeline(pipeline_without_solver(scene))
        assert warm_up() is True

    def test_the_prepare_does_nothing_after_the_close_or_during_the_wait_after_a_failure(
        self, scene: Scene
    ) -> None:
        clock = VirtualClock(synth.NIGHT_UTC_NS)
        factory = Factory(lambda: failed_future(BrokenExecutor("the worker died")))
        solver = ProcessQuickSolver(
            PointingTracker(scene.profile), clock, executor_factory=factory, retry_s=5.0
        )
        solver.solve(scene.first)  # the worker breaks, and the solver waits
        assert factory.made != []
        started = solver.workers_started
        solver.prepare()
        assert solver.workers_started == started  # nothing starts in the wait
        clock.advance(6.0)
        solver.close()
        solver.prepare()
        assert solver.workers_started == started  # and nothing starts after the close

    def test_a_pool_that_cannot_start_does_not_raise_in_the_prepare(self, scene: Scene) -> None:
        def cannot_start() -> Executor:
            raise OSError("no process")

        solver = ProcessQuickSolver(
            PointingTracker(scene.profile),
            VirtualClock(synth.NIGHT_UTC_NS),
            executor_factory=cannot_start,
        )
        solver.prepare()
        assert solver.running is False

    def test_a_worker_that_cannot_start_gives_an_unsolved_result_and_a_pause(
        self, scene: Scene
    ) -> None:
        clock = VirtualClock(synth.NIGHT_UTC_NS)
        calls: list[int] = []

        def cannot_start() -> Executor:
            calls.append(1)
            raise OSError("no process")

        solver = ProcessQuickSolver(
            PointingTracker(scene.profile), clock, executor_factory=cannot_start, retry_s=5.0
        )
        first = solver.solve(scene.first)
        assert not first.solved
        assert first.note == "the solver process could not start"
        solver.solve(scene.first)
        assert len(calls) == 1  # the pause: no second try at once
        clock.advance(6.0)
        solver.solve(scene.first)
        assert len(calls) == 2

    def test_a_closed_solver_starts_no_worker(self, scene: Scene) -> None:
        factory = Factory(lambda: Future())
        solver = ProcessQuickSolver(
            PointingTracker(scene.profile),
            VirtualClock(synth.NIGHT_UTC_NS),
            executor_factory=factory,
        )
        solver.close()
        solver.close()  # twice is safe
        solution = solver.solve(scene.first)
        assert solution.note == "the solver is closed"
        assert factory.made == []

    def test_the_release_while_a_solve_runs_lets_that_solve_finish(self, scene: Scene) -> None:
        pipeline = pipeline_without_solver(scene)
        entered = threading.Event()
        proceed = threading.Event()

        class Slow:
            def analyze(self, frame: Frame, **options: Any) -> Any:
                entered.set()
                assert proceed.wait(30.0)
                return pipeline.analyze(frame, **options)

        solver = ProcessQuickSolver(
            PointingTracker(scene.profile),
            VirtualClock(synth.NIGHT_UTC_NS),
            executor_factory=thread_factory(Slow()),  # type: ignore[arg-type]
        )
        results: list[Any] = []
        thread = threading.Thread(target=lambda: results.append(solver.solve(scene.first)))
        thread.start()
        try:
            assert entered.wait(30.0)
            solver.release()  # the alignment ended while the solve runs
            assert not solver.running
        finally:
            proceed.set()
            thread.join(60.0)
            solver.close()
        assert len(results) == 1  # the solve ended, solved or not, and raised nothing

    def test_the_arguments_are_checked(self, scene: Scene) -> None:
        tracker = PointingTracker(scene.profile)
        clock = VirtualClock(synth.NIGHT_UTC_NS)
        with pytest.raises(ValueError, match="either spec or executor_factory"):
            ProcessQuickSolver(tracker, clock)
        with pytest.raises(ValueError, match="either spec or executor_factory"):
            ProcessQuickSolver(
                tracker,
                clock,
                spec=PipelineSpec("t", {}, {}, "none"),
                executor_factory=Factory(lambda: Future()),
            )
        with pytest.raises(ValueError, match="timeout_s"):
            ProcessQuickSolver(tracker, clock, executor_factory=Factory(Future), timeout_s=0.0)


def test_the_default_executor_is_one_process_pool_that_starts_late(scene: Scene) -> None:
    from concurrent.futures import ProcessPoolExecutor

    spec = PipelineSpec("test", {}, {}, "none")
    executor = make_quick_executor(spec)
    try:
        assert isinstance(executor, ProcessPoolExecutor)  # no process exists before a job
    finally:
        executor.shutdown()


def test_a_real_worker_process_solves_a_frame_and_the_tracker_follows(
    scene: Scene, tmp_path: Path
) -> None:
    """The whole path: spawn, a pipeline built from plain data, one frame as bytes, one answer."""
    catalog_path = tmp_path / "cap.smcat"
    write_catalog(catalog_path, scene.catalog)
    spec = PipelineSpec(
        station_id="test",
        profile=scene.profile.model_dump(mode="python"),
        config=SurveyConfig().model_dump(mode="python"),
        catalog_path=str(catalog_path),
        solvers=(),
    )
    seeded = seeded_tracker(scene)
    expected = QuickSolver(
        pipeline_without_solver(scene), copy_of(seeded, scene), VirtualClock(synth.NIGHT_UTC_NS)
    ).solve(scene.second)
    tracker = copy_of(seeded, scene)
    solver = ProcessQuickSolver(tracker, SystemClock(), spec=spec, timeout_s=240.0)
    try:
        solver.prepare()  # the helper does this when the alignment starts: the worker loads early
        got = solver.solve(scene.second)
        states = [solver.running]
    finally:
        solver.close()
    states.append(solver.running)
    assert states == [True, False]
    assert solver.workers_started == 1  # the prepare and the solve share one worker
    assert got.solved, got.note
    assert got.solver == "tracker"
    assert got.x_px == pytest.approx(expected.x_px, abs=1e-6)
    assert got.y_px == pytest.approx(expected.y_px, abs=1e-6)
    assert got.n_matched == expected.n_matched
    assert tracker.solution is not None
    assert tracker.solution.t_utc_ns == scene.second.t_utc_ns
