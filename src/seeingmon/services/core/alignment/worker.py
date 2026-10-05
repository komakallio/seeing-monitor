"""The quick solve in a worker process, so that the detector never stalls the live view.

**Why a process.** The detector (SEP) holds the Python GIL for the whole of its background
estimate and its extraction, which takes seconds on a full bin2 frame (11.7 megapixels). A solver
thread in the process of `core` therefore freezes every other thread of `core` for that long: the
JPEG encoder, the scheduler thread that reads the camera frames, and the connection layer. The
first light measured the result as a live view that froze for 3 to 4 seconds at a time, frames that
reached `core` late, and read timeouts that the scheduler took for camera faults. A worker process
has its own GIL, so the live view keeps its pace whatever the detector does.

**How it works.** `ProcessQuickSolver` is a `Solver` for the helper. It starts one worker process
(with the `spawn` method, like the survey worker) when the helper calls `prepare` or, failing that,
at the first solve, and the worker builds its own `SurveyPipeline` from a `PipelineSpec`. Each solve
sends the frame as the bytes of `encode_frame`, the latest pointing solution as a dictionary, and
the reference as JSON, and it gets back a dictionary with the quick solution
(`QuickSolution.to_dict`) and the new pointing solution (`PointingSolution.to_dict`). Only plain
data crosses the boundary. The tracker stays in `core`: `adopt` applies the trust rule and updates
it there.

**Life of the worker.** The helper calls `prepare` when the alignment starts, so that the worker
loads while the first frames arrive, `release` when the alignment ends, which stops the worker and
gives its memory back (a Raspberry Pi 4 has 2 GB), and `close` when `core` stops. A worker takes
about 100 MB while it waits, and it peaks at about 390 MB on a full bin2 frame. The next
alignment starts a new worker, which takes a few seconds to import its modules and read the
catalog. A worker that dies, or a solve that takes longer than `timeout_s`, ends the worker, and the
solve returns an unsolved result with a note. After such a failure the solver waits `retry_s`
before it starts another worker, so a worker that cannot start does not spin.

**Priority.** The worker lowers its own priority and raises its `oom_score_adj`, like the survey
worker, so the camera, the live view, and the kernel's out-of-memory killer treat it as the
expendable part of `core`.

**Name.** On Linux the worker names itself `smon-align`, and the survey worker names itself
`smon-survey`, so that `ps`, `top`, and the performance tooling tell the two processes apart
(see `seeingmon.services.core.process_names`). Both start the same way and run the same interpreter,
so nothing else tells them apart.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable
from concurrent.futures import BrokenExecutor, CancelledError, Executor, ProcessPoolExecutor
from typing import Any

from seeingmon.clock import NS_PER_S, Clock, SystemClock
from seeingmon.frames import Frame, decode_frame, encode_frame
from seeingmon.services.core.alignment.solve import (
    Analyzer,
    QuickAnalysis,
    QuickSolution,
    adopt,
    analyze_frame,
)
from seeingmon.services.core.process_names import ALIGNMENT_WORKER_NAME, name_process
from seeingmon.services.core.survey_worker import (
    lower_process_priority,
    make_worker_pool,
    raise_oom_score,
)
from seeingmon.survey.pipeline import PipelineSpec, build_pipeline, frame_time_invalid
from seeingmon.survey.pointing import PointingSolution, ReferenceSolution
from seeingmon.survey.tracker import PointingTracker

_log = logging.getLogger(__name__)

ExecutorFactory = Callable[[], Executor]

# --- The worker side ---------------------------------------------------------------------------

_PIPELINE: Analyzer | None = None
_CLOCK: Clock = SystemClock()


def install_pipeline(pipeline: Analyzer | None) -> None:
    """Set the pipeline that `run_quick_job` uses in this process.

    The worker calls it through `init_quick_worker`. A test that runs the job on a thread calls it
    directly, and calls it again with `None` to remove the pipeline.
    """
    global _PIPELINE
    _PIPELINE = pipeline


def init_quick_worker(spec: PipelineSpec, nice: int, oom_score_adj: int) -> None:
    """The initializer of the worker: its name, its priority, its OOM score, and then the pipeline.

    The priority applies before the catalog loads, so the slow start does not compete with `core`.
    The name (`smon-align` on Linux) tells this worker from the survey worker in `ps`, in `top`,
    and in the performance tooling. The function lives at the top level of the module, so that the
    `spawn` method can import it.
    """
    name_process(ALIGNMENT_WORKER_NAME)
    lower_process_priority(nice)
    raise_oom_score(oom_score_adj)
    install_pipeline(build_pipeline(spec))


def warm_up() -> bool:
    """A job that does nothing. The worker answers it after it has built its pipeline."""
    return _PIPELINE is not None


def run_quick_job(
    frame_bytes: bytes, previous: dict[str, Any] | None, reference_json: str | None, index: int
) -> dict[str, Any]:
    """Analyze one frame in the worker. The function that the pool calls.

    The answer holds `solution` (a `QuickSolution` as a dictionary) and `pointing` (the new
    `PointingSolution` as a dictionary, or `None` for a frame that no solver solved).
    """
    pipeline = _PIPELINE
    if pipeline is None:
        raise RuntimeError("the worker has no pipeline: pass init_quick_worker as the initializer")
    frame = decode_frame(frame_bytes)
    analysis = analyze_frame(
        pipeline,
        frame,
        previous=None if previous is None else PointingSolution.from_dict(previous),
        reference=None if reference_json is None else ReferenceSolution.from_json(reference_json),
        index=index,
        clock=_CLOCK,
    )
    return {
        "solution": analysis.solution.to_dict(),
        "pointing": None if analysis.pointing is None else analysis.pointing.to_dict(),
    }


def decode_analysis(answer: dict[str, Any]) -> QuickAnalysis:
    """The inverse of the answer of `run_quick_job`. Raises for anything malformed."""
    pointing = answer["pointing"]
    return QuickAnalysis(
        QuickSolution.from_dict(answer["solution"]),
        None if pointing is None else PointingSolution.from_dict(pointing),
    )


def make_quick_executor(
    spec: PipelineSpec, nice: int = 10, oom_score_adj: int = 500
) -> ProcessPoolExecutor:
    """One worker process that builds its own pipeline from `spec`. It starts at the first job."""
    return make_worker_pool(init_quick_worker, (spec, nice, oom_score_adj))


# --- The side of `core` ------------------------------------------------------------------------


def _stop(executor: Executor, *, kill: bool) -> None:
    """Shut an executor down without waiting. With `kill`, end its worker processes at once."""
    # `shutdown` forgets the processes, so take them first.
    workers = list((getattr(executor, "_processes", None) or {}).values()) if kill else []
    with contextlib.suppress(Exception):
        executor.shutdown(wait=False, cancel_futures=True)
    for worker in workers:
        with contextlib.suppress(Exception):
            worker.terminate()


class ProcessQuickSolver:
    """A `Solver` that runs the quick solve in a worker process. See the module text.

    Call `solve` from one thread. `release` and `close` may come from any thread.
    """

    def __init__(
        self,
        tracker: PointingTracker,
        clock: Clock,
        *,
        spec: PipelineSpec | None = None,
        executor_factory: ExecutorFactory | None = None,
        min_stars: int = 8,
        max_rms_px: float = 1.5,
        nice: int = 10,
        oom_score_adj: int = 500,
        timeout_s: float = 60.0,
        retry_s: float = 10.0,
    ) -> None:
        if (spec is None) == (executor_factory is None):
            raise ValueError("pass either spec or executor_factory")
        if timeout_s <= 0 or retry_s < 0:
            raise ValueError("timeout_s must be positive and retry_s must not be negative")
        if executor_factory is None:
            assert spec is not None
            frozen = spec

            def default_factory() -> Executor:
                return make_quick_executor(frozen, nice, oom_score_adj)

            executor_factory = default_factory
        self._tracker = tracker
        self._clock = clock
        self._spec = spec
        self._factory = executor_factory
        self._min_stars = min_stars
        self._max_rms_px = max_rms_px
        self._timeout_s = timeout_s
        self._retry_ns = round(retry_s * NS_PER_S)
        self._lock = threading.Lock()
        self._executor: Executor | None = None
        self._closed = False
        self._retry_after_ns = 0
        self._last_failure = ""
        self._index = 0
        self.solves = 0
        self.failures = 0
        self.workers_started = 0

    @property
    def tracker(self) -> PointingTracker:
        return self._tracker

    @property
    def spec(self) -> PipelineSpec | None:
        """What the worker builds its pipeline from, or `None` when a test gave a factory."""
        return self._spec

    @property
    def running(self) -> bool:
        """Whether a worker exists.

        It starts with `prepare` or at the first solve, and it ends with `release`.
        """
        with self._lock:
            return self._executor is not None

    # --- Solving ---------------------------------------------------------------------------

    def prepare(self) -> None:
        """Start the worker now, so that the first solve does not wait for it to load.

        The call returns at once. The worker needs a few seconds to import its modules and read
        the catalog, and the helper calls this when the alignment starts. The call does nothing
        after `close` or during the wait after a failure. A worker that cannot start shows in the
        next solve.
        """
        if self._closed or self._clock.monotonic_ns() < self._retry_after_ns:
            return
        try:
            self._executor_for_job().submit(warm_up)
        except Exception:
            _log.warning("the alignment worker could not start", exc_info=True)

    def solve(self, frame: Frame) -> QuickSolution:
        """Solve one frame in the worker. Never raises."""
        started = self._clock.monotonic_ns()
        if self._closed:
            return self._unsolved(frame, started, "the solver is closed")
        if started < self._retry_after_ns:
            return self._unsolved(frame, started, self._last_failure)
        index, self._index = self._index, self._index + 1
        try:
            analysis = self._run(frame, index)
        except BrokenExecutor:
            return self._fail(frame, started, "the solver process stopped, so it restarts", True)
        except TimeoutError:
            note = f"the quick solve took more than {self._timeout_s:g} s, so the solver restarts"
            return self._fail(frame, started, note, True)
        except CancelledError:
            return self._unsolved(frame, started, "the solver stopped")  # `release` cancelled it
        except OSError:  # the system could not start the process, or its pipe broke
            return self._fail(frame, started, "the solver process could not start", True)
        except Exception as error:
            _log.exception("the quick solve of frame %d failed", frame.seq)
            return self._unsolved(frame, started, f"analysis error: {type(error).__name__}")
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

    def _run(self, frame: Frame, index: int) -> QuickAnalysis:
        previous = self._tracker.solution
        reference = self._tracker.reference
        args = (
            encode_frame(frame),
            None if previous is None else previous.to_dict(),
            None if reference is None else reference.to_json(),
            index,
        )
        executor = self._executor_for_job()
        try:
            try:
                future = executor.submit(run_quick_job, *args)
            except BrokenExecutor:
                raise
            except RuntimeError:  # `release` shut it down since we took it: start a new worker
                executor = self._executor_for_job()
                future = executor.submit(run_quick_job, *args)
            answer = future.result(timeout=self._timeout_s)
        except (BrokenExecutor, TimeoutError):
            self._discard(executor)
            raise
        return decode_analysis(answer)

    def _executor_for_job(self) -> Executor:
        with self._lock:
            if self._closed:
                raise CancelledError("the solver is closed")
            if self._executor is None:
                self._executor = self._factory()
                self.workers_started += 1
            return self._executor

    def _discard(self, executor: Executor) -> None:
        """End a worker that failed, and forget it, unless a newer worker replaced it."""
        with self._lock:
            if self._executor is executor:
                self._executor = None
        _stop(executor, kill=True)

    def _fail(self, frame: Frame, started_ns: int, note: str, restart: bool) -> QuickSolution:
        _log.warning("the alignment solver failed: %s", note)
        if restart:
            self._retry_after_ns = self._clock.monotonic_ns() + self._retry_ns
            self._last_failure = note
        return self._unsolved(frame, started_ns, note)

    def _unsolved(self, frame: Frame, started_ns: int, note: str) -> QuickSolution:
        self.failures += 1
        return QuickSolution(
            frame.t_utc_ns,
            frame.seq,
            False,
            elapsed_s=(self._clock.monotonic_ns() - started_ns) / NS_PER_S,
            note=note,
        )

    # --- Life ------------------------------------------------------------------------------

    def release(self) -> None:
        """Stop the worker, so that it gives its memory back. The next solve starts a new one.

        The worker finishes the solve that it runs, and then it exits. The call returns at once.
        """
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            _stop(executor, kill=False)

    def close(self) -> None:
        """End the worker now, whatever it does, and start no other. Safe to call twice."""
        with self._lock:
            self._closed = True
            executor, self._executor = self._executor, None
        if executor is not None:
            _stop(executor, kill=True)
