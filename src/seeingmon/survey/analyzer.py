"""The survey analyzer: the `SurveyAnalyzer` interface over the pipeline and a worker.

`SurveyPipelineAnalyzer` implements `seeingmon.analysis.SurveyAnalyzer`. The scheduler calls
`submit` with a survey frame, which returns at once, and later collects the results with
`poll`. The heavy work (detection, solving, the fit) runs on an `Executor`:

- With no executor, a private single-thread pool runs the work in this process. NumPy and SEP
  release the GIL for most of it, but a worker process isolates the survey path better.
- Pass a `ProcessPoolExecutor` made by `make_process_executor(spec)` to run the work in a
  worker process, as the architecture describes. The worker builds its own pipeline from a
  `PipelineSpec` (plain data), loads the catalog once, and never shares memory with this process.
- Pass `InlineExecutor()` to run each job in `submit`, which keeps tests deterministic.

**What crosses the process boundary.** The job takes the frame as the bytes of
`encode_frame`, the previous solution as a dictionary, and the reference as a JSON string. It
returns a dictionary of record rows, the new solution, and numbers. No object crosses by pickle
beyond these plain types, so the rule that no pickle of arbitrary objects crosses a process
boundary holds here too.

**State.** The analyzer keeps the `PointingTracker`. `poll` gives each solved result to the
tracker, so `tracker.polaris_position` (a `PointingProvider`) always reflects the latest
accepted solution. A result with too few stars or a large residual does not update the tracker.
Each `submit` hands the worker the solution at that moment, so a frame that follows quickly
uses the previous solution even if the earlier result has not come back.

A job that fails with an unexpected error does not stop the analyzer. `poll` returns an
unsolved result with a `survey_frame` and an unsolved `pointing` record, and the error goes to
the log.
"""

from __future__ import annotations

import logging
import multiprocessing
from collections import deque
from collections.abc import Callable
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from seeingmon.analysis import SurveyOutput
from seeingmon.clock import Clock
from seeingmon.frames import Frame, decode_frame, encode_frame
from seeingmon.profile import Profile
from seeingmon.records import Record, get_record_type
from seeingmon.survey.config import SurveyConfig
from seeingmon.survey.geometry import ARCSEC_PER_RAD
from seeingmon.survey.pipeline import (
    FrameAnalysis,
    PipelineSpec,
    SurveyPipeline,
    build_pipeline,
    failure_records,
    frame_time_invalid,
    solver_specs_from_config,
)
from seeingmon.survey.pointing import (
    POINTING_ALGORITHM,
    PointingSolution,
    ReferenceSolution,
    load_reference,
)
from seeingmon.survey.tracker import PointingTracker

log = logging.getLogger("seeingmon.survey")


class InlineExecutor(Executor):
    """An executor that runs each job when you submit it, in the calling thread.

    The future is complete when `submit` returns. Use it in tests and in virtual-time runs,
    where a job must finish before the next step.
    """

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        future: Future[Any] = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future


# --- The worker side -----------------------------------------------------------------------

_WORKER_PIPELINE: SurveyPipeline | None = None


def init_worker(spec: PipelineSpec) -> None:
    """Build the pipeline of a worker process. Pass it as the `initializer` of the pool."""
    global _WORKER_PIPELINE
    _WORKER_PIPELINE = build_pipeline(spec)


def make_process_executor(spec: PipelineSpec) -> ProcessPoolExecutor:
    """A single worker process that builds its own pipeline from `spec`.

    The pool uses the `spawn` start method, so the worker holds no inherited state, such as
    threads or open files from the parent.
    """
    return ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker,
        initargs=(spec,),
    )


def run_job(
    frame_bytes: bytes,
    previous: dict[str, Any] | None,
    reference_json: str | None,
    index: int,
) -> dict[str, Any]:
    """Run one frame in a worker process (the function that the pool calls)."""
    if _WORKER_PIPELINE is None:
        raise RuntimeError("the worker has no pipeline: pass init_worker as the pool initializer")
    return run_encoded(_WORKER_PIPELINE, frame_bytes, previous, reference_json, index)


def run_encoded(
    pipeline: SurveyPipeline,
    frame_bytes: bytes,
    previous: dict[str, Any] | None,
    reference_json: str | None,
    index: int,
) -> dict[str, Any]:
    """Decode a job, run the pipeline, and encode the result as plain data."""
    frame = decode_frame(frame_bytes)
    solution = None if previous is None else PointingSolution.from_dict(previous)
    reference = None if reference_json is None else ReferenceSolution.from_json(reference_json)
    analysis = pipeline.analyze(frame, previous=solution, reference=reference, index=index)
    return encode_analysis(analysis)


def encode_analysis(analysis: FrameAnalysis) -> dict[str, Any]:
    """The plain-data form of a `FrameAnalysis`, without the detections."""
    return {
        "records": [
            {"record_type": record.record_type, "row": record.to_row()}
            for record in analysis.records
        ],
        "solved": analysis.solved,
        "cloud_fraction": analysis.cloud_fraction,
        "solution": None if analysis.solution is None else analysis.solution.to_dict(),
        "timings": dict(analysis.timings),
        "notes": list(analysis.notes),
    }


def decode_records(encoded: list[dict[str, Any]]) -> tuple[Record, ...]:
    """Rebuild records from `encode_analysis` rows."""
    return tuple(get_record_type(item["record_type"]).from_row(item["row"]) for item in encoded)


# --- The scheduler side --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _FrameInfo:
    """What a failed job still needs to report about its frame."""

    t_utc_ns: int
    exposure_us: int
    gain: int
    mode: str
    temperature_c: float | None
    time_invalid: bool


@dataclass(slots=True)
class _Pending:
    future: Future[dict[str, Any]]
    info: _FrameInfo
    reference: ReferenceSolution | None


class SurveyPipelineAnalyzer:
    """A `SurveyAnalyzer` that runs the survey pipeline on an executor. See the module text."""

    def __init__(
        self,
        *,
        profile: Profile,
        station_id: str,
        config: SurveyConfig | None = None,
        spec: PipelineSpec | None = None,
        pipeline: SurveyPipeline | None = None,
        executor: Executor | None = None,
        tracker: PointingTracker | None = None,
        reference: ReferenceSolution | None = None,
        clock: Clock | None = None,
    ) -> None:
        if (spec is None) == (pipeline is None):
            raise ValueError("pass either spec or pipeline")
        if isinstance(executor, ProcessPoolExecutor) and spec is None:
            raise ValueError(
                "a process pool needs a spec, which the worker builds its pipeline from"
            )
        self._profile = profile
        self._station_id = station_id
        self._config = config or SurveyConfig()
        self._spec = spec
        self._pipeline = pipeline
        self._clock = clock
        self._owned_executor: Executor | None = None
        if executor is None:
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="survey")
            self._owned_executor = executor
        self._executor = executor
        self._use_worker_function = isinstance(executor, ProcessPoolExecutor)
        self._tracker = tracker or PointingTracker(
            profile, validity_s=self._config.pointing.validity_s, reference=reference
        )
        if reference is not None:
            self._tracker.set_reference(reference)
        elif self._tracker.reference is None and self._config.pointing.reference_file:
            self._tracker.set_reference(load_reference(self._config.pointing.reference_file))
        self._jobs: deque[_Pending] = deque()
        self._submitted = 0

    # --- The SurveyAnalyzer interface -----------------------------------------------------

    def submit(self, frame: Frame) -> None:
        """Queue a survey frame. Returns at once (the job runs on the executor)."""
        previous = self._tracker.solution
        reference = self._tracker.reference
        args = (
            encode_frame(frame),
            None if previous is None else previous.to_dict(),
            None if reference is None else reference.to_json(),
            self._submitted,
        )
        self._submitted += 1
        if self._use_worker_function:
            future = self._executor.submit(run_job, *args)
        else:
            future = self._executor.submit(self._run_in_process, *args)
        info = _FrameInfo(
            t_utc_ns=frame.t_utc_ns,
            exposure_us=frame.exposure_us,
            gain=frame.gain,
            mode=frame.mode,
            temperature_c=frame.temperature_c,
            time_invalid=frame_time_invalid(frame),
        )
        self._jobs.append(_Pending(future, info, reference))

    def poll(self) -> tuple[SurveyOutput, ...]:
        """Return the finished results, in submission order, and update the tracker."""
        ready: list[SurveyOutput] = []
        while self._jobs and self._jobs[0].future.done():
            ready.append(self._finish(self._jobs.popleft()))
        return tuple(ready)

    def pending(self) -> int:
        return len(self._jobs)

    # --- The rest -------------------------------------------------------------------------

    @property
    def tracker(self) -> PointingTracker:
        """The pointing tracker, which is also the `PointingProvider` for the scheduler."""
        return self._tracker

    def close(self, *, wait: bool = False) -> None:
        """Shut down the executor that the analyzer created. An executor you passed stays up."""
        if self._owned_executor is not None:
            self._owned_executor.shutdown(wait=wait, cancel_futures=not wait)

    def _run_in_process(
        self,
        frame_bytes: bytes,
        previous: dict[str, Any] | None,
        reference_json: str | None,
        index: int,
    ) -> dict[str, Any]:
        if self._pipeline is None:
            assert self._spec is not None
            self._pipeline = build_pipeline(self._spec, self._clock)
        return run_encoded(self._pipeline, frame_bytes, previous, reference_json, index)

    def _finish(self, job: _Pending) -> SurveyOutput:
        info = job.info
        try:
            result = job.future.result()
            records = decode_records(result["records"])
            solved = bool(result["solved"])
            cloud = result["cloud_fraction"]
            solution = (
                None
                if result["solution"] is None
                else PointingSolution.from_dict(result["solution"])
            )
            for note in result["notes"]:
                log.info("survey frame at %d: %s", info.t_utc_ns, note)
        except Exception as error:
            log.exception("the survey analysis of the frame at %d failed", info.t_utc_ns)
            return self._failure_output(job, f"analysis error: {type(error).__name__}")
        if solution is not None and self._trusted(solution):
            self._tracker.update(solution)
        return SurveyOutput(
            t_utc_ns=info.t_utc_ns,
            records=records,
            solved=solved,
            cloud_fraction=None if cloud is None else float(cloud),
        )

    def _trusted(self, solution: PointingSolution) -> bool:
        """Whether a solution may update the tracker: enough stars and a small residual."""
        cfg = self._config.pointing
        rms_px = (
            None
            if solution.rms_arcsec is None
            else solution.rms_arcsec / (solution.scale_rad_px * ARCSEC_PER_RAD)
        )
        return solution.n_matched >= cfg.tracker_min_stars and (
            rms_px is None or rms_px <= cfg.tracker_max_rms_px
        )

    def _failure_output(self, job: _Pending, reason: str) -> SurveyOutput:
        info = job.info
        records = failure_records(
            station_id=self._station_id,
            profile_id=self._profile.id,
            t_utc_ns=info.t_utc_ns,
            exposure_us=info.exposure_us,
            gain=info.gain,
            mode=info.mode,
            temperature_c=info.temperature_c,
            time_invalid=info.time_invalid,
            reference=job.reference,
            provenance={"algo": POINTING_ALGORITHM},
            reason=reason,
        )
        return SurveyOutput(
            t_utc_ns=info.t_utc_ns, records=records, solved=False, cloud_fraction=None
        )


def create_survey_analyzer(
    *,
    profile: Profile,
    station_id: str,
    config: SurveyConfig,
    executor: Executor | None = None,
    clock: Clock | None = None,
) -> SurveyPipelineAnalyzer:
    """Build an analyzer from the configuration, which names the catalog and the solvers.

    Raises `ValueError` when `catalog_path` is empty. With no executor the analyzer uses a
    private thread. For a worker process, pass `make_process_executor(spec)` with the same spec
    that this function builds: `analyzer_spec(...)`.
    """
    spec = analyzer_spec(profile=profile, station_id=station_id, config=config)
    return SurveyPipelineAnalyzer(
        profile=profile,
        station_id=station_id,
        config=config,
        spec=spec,
        executor=executor,
        clock=clock,
    )


def analyzer_spec(*, profile: Profile, station_id: str, config: SurveyConfig) -> PipelineSpec:
    """The pipeline specification that a configuration describes."""
    if not config.catalog_path:
        raise ValueError(
            "catalog_path is not set in the [survey] configuration: run `seeingmon catalog build`"
        )
    return PipelineSpec(
        station_id=station_id,
        profile=profile.model_dump(mode="python"),
        config=config.model_dump(mode="python"),
        catalog_path=config.catalog_path,
        solvers=solver_specs_from_config(config),
        hot_pixel_file=config.hot_pixel_file,
    )
