"""The survey analyzer: the SurveyAnalyzer interface over the pipeline and an executor."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from seeingmon.analysis import PointingProvider, SurveyAnalyzer, SurveyOutput
from seeingmon.frames import Frame
from seeingmon.profile import Profile
from seeingmon.records.survey import PointingRecord, SkyQualityRecord, SurveyFrameRecord
from seeingmon.survey import pointing as pt
from seeingmon.survey.analyzer import (
    InlineExecutor,
    SurveyPipelineAnalyzer,
    analyzer_spec,
    create_survey_analyzer,
    decode_records,
    encode_analysis,
    make_process_executor,
)
from seeingmon.survey.catalog import CapCatalog, write_catalog
from seeingmon.survey.config import SurveyConfig
from seeingmon.survey.dark import DarkLibrary
from seeingmon.survey.geometry import ARCSEC_PER_RAD, exp_so3
from seeingmon.survey.pipeline import (
    NO_SOLUTION,
    SOLVED,
    FrameAnalysis,
    PipelineSpec,
    SolveAttempt,
    SurveyPipeline,
)
from seeingmon.survey.transparency import ZeroPointReference
from seeingmon.survey.wcs_fit import CameraAttitude
from tests.survey import synth

NS = 1_000_000_000


@pytest.fixture(scope="module")
def profile() -> Profile:
    return synth.cropped_profile(1200, 800)


@pytest.fixture(scope="module")
def catalog() -> CapCatalog:
    return synth.synthetic_catalog(cap_radius_deg=5.0, density_scale=3.0, seed=1)


class ScriptedPipeline(SurveyPipeline):
    """A pipeline that returns scripted analyses, so tests control timing and failure."""

    def __init__(self, profile: Profile, catalog: CapCatalog) -> None:
        self._profile = profile
        self._catalog = catalog
        self.calls: list[tuple[int, int]] = []  # (t_utc_ns, index) of each call
        self.gates: dict[int, threading.Event] = {}  # a call waits for the gate of its index
        self.done = threading.Event()
        self.solution_for: dict[int, pt.PointingSolution | None] = {}
        self.fail_at: set[int] = set()
        self.references: list[ZeroPointReference | None] = []  # the zp_reference of each call
        self.notes_for: dict[int, tuple[str, ...]] = {}
        self.attempts_for: dict[int, tuple[SolveAttempt, ...]] = {}
        self.timings_for: dict[int, dict[str, float]] = {}

    def analyze(
        self,
        frame: Frame,
        *,
        previous: pt.PointingSolution | None = None,
        reference: pt.ReferenceSolution | None = None,
        index: int = 0,
        zp_reference: ZeroPointReference | None = None,
        sky_quality: bool | None = None,
    ) -> FrameAnalysis:
        self.calls.append((frame.t_utc_ns, index))
        self.references.append(zp_reference)
        gate = self.gates.get(index)
        if gate is not None:
            assert gate.wait(timeout=30.0)
        if index in self.fail_at:
            raise RuntimeError("the pipeline failed on purpose")
        solution = self.solution_for.get(index)
        survey = SurveyFrameRecord(
            station_id="test",
            t_utc_ns=frame.t_utc_ns,
            profile_id=self._profile.id,
            provenance={"algo": "scripted"},
            exposure_s=frame.exposure_us / 1e6,
            gain=frame.gain,
            readout_mode=frame.mode,
            n_detected=index,
        )
        self.done.set()
        return FrameAnalysis(
            records=(survey,),
            solved=solution is not None,
            cloud_fraction=0.1 * index,
            solution=solution,
            notes=self.notes_for.get(index, ()),
            attempts=self.attempts_for.get(index, ()),
            timings=self.timings_for.get(index, {}),
        )


def small_frame(t_utc_ns: int, mode: str = "bin2") -> Frame:
    from seeingmon.frames import Roi, TimeQuality

    return Frame(
        data=np.zeros((4, 4), dtype=np.uint16),
        stream_id=1,
        seq=0,
        t_arrival_ns=t_utc_ns,
        t_utc_ns=t_utc_ns,
        t_err_ns=0,
        t_quality=TimeQuality.EXACT,
        dropped_before=0,
        exposure_us=1_000_000,
        gain=100,
        mode=mode,
        roi=Roi(0, 0, 4, 4),
        adc_bits=14,
    )


def collect(analyzer: SurveyPipelineAnalyzer, count: int, *, timeout_s: float = 30.0):  # type: ignore[no-untyped-def]
    """Poll until `count` results have come back (a real wait, for a thread or a process)."""
    outputs: list[SurveyOutput] = []
    waiter = threading.Event()
    for _ in range(int(timeout_s / 0.02)):
        outputs.extend(analyzer.poll())
        if len(outputs) >= count:
            return outputs
        waiter.wait(0.02)
    raise AssertionError(f"only {len(outputs)} of {count} results arrived")


def held(analyzer: SurveyPipelineAnalyzer) -> pt.PointingSolution | None:
    """The solution that the tracker holds now (a function, so mypy does not narrow it)."""
    return analyzer.tracker.solution


def solution_with(
    n_matched: int, rms_arcsec: float, t_utc_ns: int = 1_000_000_000
) -> pt.PointingSolution:
    return pt.PointingSolution(
        rotation_earth_fixed=synth.make_attitude(0.9, 40.0, 25.0),
        scale_rad_px=3.82 / ARCSEC_PER_RAD,
        parity=1,
        mode="bin2",
        width_px=1200,
        height_px=800,
        center_px=(599.5, 399.5),
        t_utc_ns=t_utc_ns,
        n_matched=n_matched,
        rms_arcsec=rms_arcsec,
    )


def scripted(
    profile: Profile, catalog: CapCatalog, **options: object
) -> tuple[SurveyPipelineAnalyzer, ScriptedPipeline]:
    pipeline = ScriptedPipeline(profile, catalog)
    analyzer = SurveyPipelineAnalyzer(
        profile=profile,
        station_id="test",
        pipeline=pipeline,
        **options,  # type: ignore[arg-type]
    )
    return analyzer, pipeline


# --- The interface ---------------------------------------------------------------------------


def test_the_analyzer_satisfies_the_interfaces(profile: Profile, catalog: CapCatalog) -> None:
    analyzer, _ = scripted(profile, catalog, executor=InlineExecutor())
    assert isinstance(analyzer, SurveyAnalyzer)
    assert isinstance(analyzer.tracker, PointingProvider)
    assert analyzer.pending() == 0
    assert analyzer.poll() == ()


def test_submit_returns_at_once_and_poll_waits_for_the_job(
    profile: Profile, catalog: CapCatalog
) -> None:
    analyzer, pipeline = scripted(profile, catalog)  # a private thread
    try:
        pipeline.gates[0] = threading.Event()
        analyzer.submit(small_frame(10 * NS))
        analyzer.submit(small_frame(20 * NS))
        assert analyzer.pending() == 2
        assert analyzer.poll() == ()  # the first job is blocked
        pipeline.gates[0].set()
        outputs = collect(analyzer, 2)
        assert [output.t_utc_ns for output in outputs] == [10 * NS, 20 * NS]
        assert analyzer.pending() == 0
    finally:
        analyzer.close()


def test_results_keep_the_submission_order_even_when_a_later_job_finishes_first(
    profile: Profile, catalog: CapCatalog
) -> None:
    executor = ThreadPoolExecutor(max_workers=2)
    analyzer, pipeline = scripted(profile, catalog, executor=executor)
    try:
        pipeline.gates[0] = threading.Event()
        analyzer.submit(small_frame(10 * NS))
        analyzer.submit(small_frame(20 * NS))
        analyzer.submit(small_frame(30 * NS))
        # Wait until the second job has finished, while the first is still blocked.
        pipeline.done.wait(10.0)
        waiter = threading.Event()
        for _ in range(200):
            if len(pipeline.calls) == 3:
                break
            waiter.wait(0.01)
        waiter.wait(0.1)
        assert analyzer.poll() == ()  # nothing comes back before the first job
        pipeline.gates[0].set()
        outputs = collect(analyzer, 3)
        assert [output.t_utc_ns for output in outputs] == [10 * NS, 20 * NS, 30 * NS]
    finally:
        executor.shutdown(wait=True)


def test_each_job_gets_the_previous_solution_that_the_tracker_held_at_submit(
    profile: Profile, catalog: CapCatalog
) -> None:
    analyzer, pipeline = scripted(profile, catalog, executor=InlineExecutor())
    pipeline.solution_for[0] = solution_with(300, 0.1, t_utc_ns=10 * NS)
    seen: list[pt.PointingSolution | None] = []
    original = pipeline.analyze

    def spy(frame: Frame, **kwargs: object) -> FrameAnalysis:
        seen.append(kwargs.get("previous"))  # type: ignore[arg-type]
        return original(frame, **kwargs)  # type: ignore[arg-type]

    pipeline.analyze = spy  # type: ignore[method-assign]
    analyzer.submit(small_frame(10 * NS))
    first = analyzer.poll()
    analyzer.submit(small_frame(20 * NS))
    analyzer.poll()
    assert seen[0] is None
    assert seen[1] is not None
    assert seen[1].t_utc_ns == 10 * NS
    assert first[0].solved


def test_the_output_carries_the_records_the_solved_flag_and_the_cloud_fraction(
    profile: Profile, catalog: CapCatalog
) -> None:
    analyzer, pipeline = scripted(profile, catalog, executor=InlineExecutor())
    pipeline.solution_for[2] = solution_with(300, 0.1)
    for i in range(3):
        analyzer.submit(small_frame((i + 1) * NS))
    outputs = analyzer.poll()
    assert [output.solved for output in outputs] == [False, False, True]
    assert [output.cloud_fraction for output in outputs] == pytest.approx([0.0, 0.1, 0.2])
    record = next(r for r in outputs[1].records if isinstance(r, SurveyFrameRecord))
    assert isinstance(record, SurveyFrameRecord)
    assert record.n_detected == 1
    assert record.t_utc_ns == outputs[1].t_utc_ns == 2 * NS


# --- The tracker -----------------------------------------------------------------------------


def test_only_a_trusted_solution_updates_the_tracker(profile: Profile, catalog: CapCatalog) -> None:
    analyzer, pipeline = scripted(profile, catalog, executor=InlineExecutor())
    pipeline.solution_for[0] = solution_with(3, 0.1, t_utc_ns=1 * NS)  # too few stars
    pipeline.solution_for[1] = solution_with(300, 9.0, t_utc_ns=2 * NS)  # 2.4 px residual
    pipeline.solution_for[2] = solution_with(300, 0.1, t_utc_ns=3 * NS)
    analyzer.submit(small_frame(1 * NS))
    analyzer.poll()
    assert held(analyzer) is None
    analyzer.submit(small_frame(2 * NS))
    analyzer.poll()
    assert held(analyzer) is None
    analyzer.submit(small_frame(3 * NS))
    analyzer.poll()
    adopted = held(analyzer)
    assert adopted is not None
    assert adopted.t_utc_ns == 3 * NS


def test_a_failed_job_gives_an_unsolved_output_and_the_analyzer_goes_on(
    profile: Profile, catalog: CapCatalog, caplog: pytest.LogCaptureFixture
) -> None:
    analyzer, pipeline = scripted(profile, catalog, executor=InlineExecutor())
    pipeline.fail_at = {0}
    pipeline.solution_for[1] = solution_with(300, 0.1, t_utc_ns=2 * NS)
    analyzer.submit(small_frame(1 * NS))
    analyzer.submit(small_frame(2 * NS))
    with caplog.at_level("ERROR", logger="seeingmon.survey"):
        outputs = analyzer.poll()
    assert [output.solved for output in outputs] == [False, True]
    failed = outputs[0]
    assert [record.record_type for record in failed.records] == [
        "survey_frame",
        "sky_quality",
        "pointing",
    ]
    pointing = next(r for r in failed.records if isinstance(r, PointingRecord))
    assert isinstance(pointing, PointingRecord)
    assert pointing.flags == ["unsolved"]
    assert failed.cloud_fraction is None
    assert any("failed" in message for message in caplog.messages)
    assert analyzer.tracker.solution is not None  # the second frame still updated it


def survey_log(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The messages of the survey logger, without the time that the logging module adds."""
    return [r.getMessage() for r in caplog.records if r.name == "seeingmon.survey"]


def test_the_log_follows_each_solver_run_and_the_outcome_of_a_frame(
    profile: Profile, catalog: CapCatalog, caplog: pytest.LogCaptureFixture
) -> None:
    analyzer, pipeline = scripted(profile, catalog, executor=InlineExecutor())
    pipeline.solution_for[0] = replace(solution_with(87, 0.1, t_utc_ns=10 * NS), solver="astap")
    pipeline.notes_for[0] = (
        "the tracker could not match the frame",
        "astrometry.net found no solution",
    )
    pipeline.attempts_for[0] = (
        SolveAttempt(
            "astrometry.net", NO_SOLUTION, 48, 2.41, reason="astrometry.net found no solution"
        ),
        SolveAttempt("astap", SOLVED, 48, 0.81, matched=87),
    )
    pipeline.timings_for[0] = {"detect": 1.5, "solve": 0.9}
    analyzer.submit(small_frame(10 * NS))
    with caplog.at_level("INFO", logger="seeingmon.survey"):
        analyzer.poll()
    stamp = "survey frame 1970-01-01T00:00:10Z"
    assert survey_log(caplog) == [
        f"{stamp}: the tracker could not match the frame",
        f"{stamp}: solver=astrometry.net result=no_solution stars=48 time_s=2.41 "
        'reason="astrometry.net found no solution"',
        f"{stamp}: solver=astap result=solved stars=48 time_s=0.81 matched=87",
        f"{stamp}: 1 s bin2: 0 stars detected, solved by astap (87 matched), analysis took 2.4 s",
    ]


def test_a_frame_without_a_solution_says_so_and_a_frame_that_the_tracker_solved_has_one_line(
    profile: Profile, catalog: CapCatalog, caplog: pytest.LogCaptureFixture
) -> None:
    analyzer, pipeline = scripted(profile, catalog, executor=InlineExecutor())
    pipeline.notes_for[0] = ("only 3 stars for a solver",)
    pipeline.solution_for[1] = replace(solution_with(214, 0.1, t_utc_ns=20 * NS), solver="tracker")
    pipeline.timings_for[1] = {"detect": 0.5}
    analyzer.submit(small_frame(10 * NS))
    analyzer.submit(small_frame(20 * NS))
    with caplog.at_level("INFO", logger="seeingmon.survey"):
        analyzer.poll()
    assert survey_log(caplog) == [
        "survey frame 1970-01-01T00:00:10Z: only 3 stars for a solver",
        "survey frame 1970-01-01T00:00:10Z: 1 s bin2: 0 stars detected, not solved, "
        "analysis took 0.0 s",
        "survey frame 1970-01-01T00:00:20Z: 1 s bin2: 1 stars detected, "
        "solved by tracker (214 matched), analysis took 0.5 s",
    ]


def test_an_attempt_reaches_core_through_the_dictionary_of_the_worker(
    profile: Profile, catalog: CapCatalog
) -> None:
    attempt = SolveAttempt("astap", SOLVED, 48, 0.81, matched=87)
    analysis = FrameAnalysis(records=(), solved=True, cloud_fraction=None, attempts=(attempt,))
    encoded = encode_analysis(analysis)
    assert encoded["attempts"] == [attempt.to_dict()]
    assert all(isinstance(value, str | int | float) for value in encoded["attempts"][0].values())
    assert (
        encode_analysis(FrameAnalysis(records=(), solved=False, cloud_fraction=None))["attempts"]
        == []
    )


def test_the_reference_file_in_the_configuration_is_loaded(
    profile: Profile, catalog: CapCatalog, tmp_path: Path
) -> None:
    path = tmp_path / "reference.json"
    pt.save_reference(path, pt.ReferenceSolution("ref-7", solution_with(300, 0.1)))
    config = SurveyConfig.model_validate({"pointing": {"reference_file": str(path)}})
    analyzer, _ = scripted(profile, catalog, executor=InlineExecutor(), config=config)
    assert analyzer.tracker.reference is not None
    assert analyzer.tracker.reference.reference_id == "ref-7"
    explicit = pt.ReferenceSolution("explicit", solution_with(300, 0.1))
    other, _ = scripted(
        profile, catalog, executor=InlineExecutor(), config=config, reference=explicit
    )
    assert other.tracker.reference is not None
    assert other.tracker.reference.reference_id == "explicit"


# --- Construction ----------------------------------------------------------------------------


def test_the_analyzer_needs_exactly_one_of_a_spec_and_a_pipeline(
    profile: Profile, catalog: CapCatalog
) -> None:
    with pytest.raises(ValueError, match="either spec or pipeline"):
        SurveyPipelineAnalyzer(profile=profile, station_id="test")
    pipeline = ScriptedPipeline(profile, catalog)
    spec = PipelineSpec("test", {}, {}, "")
    with pytest.raises(ValueError, match="either spec or pipeline"):
        SurveyPipelineAnalyzer(profile=profile, station_id="test", pipeline=pipeline, spec=spec)


def test_a_process_pool_needs_a_spec(profile: Profile, catalog: CapCatalog, tmp_path: Path) -> None:
    spec = analyzer_spec(
        profile=profile,
        station_id="test",
        config=SurveyConfig(catalog_path=str(tmp_path / "x.smcat"), solvers=()),
    )
    executor = make_process_executor(spec)
    try:
        with pytest.raises(ValueError, match="needs a spec"):
            SurveyPipelineAnalyzer(
                profile=profile,
                station_id="test",
                pipeline=ScriptedPipeline(profile, catalog),
                executor=executor,
            )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def test_the_spec_describes_the_pipeline_in_plain_data(profile: Profile, tmp_path: Path) -> None:
    config = SurveyConfig(
        catalog_path=str(tmp_path / "cap.smcat"),
        index_dir=str(tmp_path / "idx"),
        solve_field_command="solve-field --verbose",
        astap_database_dir=str(tmp_path / "db"),
    )
    spec = analyzer_spec(profile=profile, station_id="station-9", config=config)
    assert spec.station_id == "station-9"
    assert Profile.model_validate(spec.profile) == profile
    assert SurveyConfig.model_validate(spec.config) == config
    assert [s.kind for s in spec.solvers] == ["astrometry.net", "astap"]
    assert spec.solvers[0].command == "solve-field --verbose"
    assert spec.solvers[0].index_dir == str(tmp_path / "idx")
    assert spec.solvers[1].database_dir == str(tmp_path / "db")
    with pytest.raises(ValueError, match="catalog_path"):
        analyzer_spec(profile=profile, station_id="s", config=SurveyConfig())
    with pytest.raises(ValueError, match="unknown solver"):
        analyzer_spec(
            profile=profile,
            station_id="s",
            config=SurveyConfig(catalog_path="x", solvers=("nonsense",)),
        )


def test_the_records_survive_the_encoding_that_crosses_a_process_boundary(
    profile: Profile, catalog: CapCatalog
) -> None:
    frame, truth = synth.render_frame(
        catalog, profile, rotation_tirs=synth.make_attitude(0.9, 40.0, 25.0), seed=3
    )
    pipeline = SurveyPipeline(
        station_id="test",
        profile=profile,
        catalog=catalog,
        solvers=[synth.QueueSolver([synth.truth_solve_result(truth, catalog)])],
    )
    analysis = pipeline.analyze(frame)
    encoded = encode_analysis(analysis)

    # Only plain types cross: dictionaries, lists, strings, bytes, numbers, and None.
    def is_plain(value: object) -> bool:
        if isinstance(value, dict):
            return all(isinstance(k, str) and is_plain(v) for k, v in value.items())
        if isinstance(value, list):
            return all(is_plain(v) for v in value)
        return value is None or isinstance(value, str | bytes | int | float | bool)

    assert is_plain(encoded)
    again = decode_records(encoded["records"])
    assert again == analysis.records  # records compare field by field


# --- The whole analyzer on a short night ------------------------------------------------------

SLIP_PX = 3.0  # the mount slips by this many pixels at frame 5


@dataclass
class Night:
    analyzer: SurveyPipelineAnalyzer
    truths: list[synth.SynthTruth]
    outputs: list[SurveyOutput]


def pointing_of(output: SurveyOutput) -> PointingRecord:
    return next(record for record in output.records if isinstance(record, PointingRecord))


def run_night(profile: Profile, catalog: CapCatalog) -> Night:
    """Eight frames, 3 minutes apart, from one fixed mount.

    Frame 0 needs the solver. Frame 3 is under heavy cloud. At frame 5 the mount slips by 3
    pixels. After frame 0 the test sets that solution as the reference, as commissioning would.
    """
    rotation = synth.make_attitude(0.9, 40.0, 25.0)
    slipped = exp_so3([0.0, SLIP_PX * 3.82 / ARCSEC_PER_RAD, 0.0]) @ rotation
    frames = [
        synth.render_frame(
            catalog,
            profile,
            rotation_tirs=slipped if i >= 5 else rotation,
            t_utc_ns=synth.NIGHT_UTC_NS + i * 180 * NS,
            exposure_s=30.0,
            transmission=0.0005 if i == 3 else 1.0,
            seed=40 + i,
        )
        for i in range(8)
    ]
    solver = synth.QueueSolver([synth.truth_solve_result(frames[0][1], catalog)])
    pipeline = SurveyPipeline(station_id="test", profile=profile, catalog=catalog, solvers=[solver])
    analyzer = SurveyPipelineAnalyzer(
        profile=profile, station_id="test", pipeline=pipeline, executor=InlineExecutor()
    )
    outputs: list[SurveyOutput] = []
    for index, (frame, _) in enumerate(frames):
        analyzer.submit(frame)
        outputs.extend(analyzer.poll())
        if index == 0:
            first = analyzer.tracker.solution
            assert first is not None
            analyzer.tracker.set_reference(pt.ReferenceSolution("night-0", first))
    return Night(analyzer, [truth for _, truth in frames], outputs)


@pytest.fixture(scope="module")
def night(profile: Profile, catalog: CapCatalog) -> Night:
    return run_night(profile, catalog)


def test_the_night_produces_one_result_for_each_frame(night: Night) -> None:
    assert len(night.outputs) == 8
    assert night.analyzer.pending() == 0


def test_the_first_frame_needs_the_solver_and_the_others_use_the_tracker(night: Night) -> None:
    solvers = [pointing_of(output).solver for output in night.outputs]
    assert solvers[0] == "synthetic"
    assert solvers[1] == solvers[2] == "tracker"
    assert solvers[3] == "tracker"  # a few bright stars still shine through the cloud
    assert solvers[4:] == ["tracker"] * 4


def test_the_cloudy_frame_solves_with_a_few_stars_and_reports_the_clouds(night: Night) -> None:
    cloudy = night.outputs[3]
    pointing = pointing_of(cloudy)
    # Only the brightest stars show, so the fit has few stars and says so.
    assert cloudy.solved
    assert 4 <= pointing.n_matched < 12
    assert pointing.flags == ["few_stars"]
    assert cloudy.cloud_fraction is not None
    assert cloudy.cloud_fraction > 0.9
    clear = [output for i, output in enumerate(night.outputs) if i != 3]
    assert all(output.solved for output in clear)
    assert all(
        output.cloud_fraction is not None and output.cloud_fraction < 0.1 for output in clear
    )


def test_every_solved_frame_recovers_the_pointing(night: Night) -> None:
    for index, (output, truth) in enumerate(zip(night.outputs, night.truths, strict=True)):
        pointing = pointing_of(output)
        assert pointing.attitude is not None
        fitted = CameraAttitude(
            np.array(pointing.attitude).reshape(3, 3),
            (pointing.plate_scale_arcsec_px or 0.0) / ARCSEC_PER_RAD,
            1,
            truth.center_px,
        )
        true_model = CameraAttitude(
            truth.rotation_cirs, truth.scale_arcsec_px / ARCSEC_PER_RAD, 1, truth.center_px
        )
        gx, gy = np.meshgrid(np.linspace(0, 1199, 7), np.linspace(0, 799, 5))
        x, y, _ = fitted.project(true_model.unproject(gx.ravel(), gy.ravel()))
        # The cloudy frame has only a few stars, so it is less precise.
        limit = 0.5 if index == 3 else 0.05
        assert float(np.max(np.hypot(x - gx.ravel(), y - gy.ravel()))) < limit, index
        assert pointing.roll_deg == pytest.approx(true_model.roll_deg(), abs=0.05)


def test_the_slip_of_the_mount_shows_in_the_offset_from_the_reference(night: Night) -> None:
    offsets = {
        index: pointing_of(output).offset_arcmin
        for index, output in enumerate(night.outputs)
        if index != 0
    }
    assert all(offset is not None for offset in offsets.values())
    slip_arcmin = SLIP_PX * 3.82 / 60.0
    for index, offset in offsets.items():
        assert offset is not None
        expected = slip_arcmin if index >= 5 else 0.0
        assert offset == pytest.approx(expected, abs=0.05 if index == 3 else 0.02), index
    # A slip of 0.19 arcmin is far below the limit, so no frame carries the flag.
    assert all(pointing_of(output).flags == [] for i, output in enumerate(night.outputs) if i != 3)
    assert offsets[3] is not None


def test_the_tracker_predicts_polaris_after_the_night(night: Night, catalog: CapCatalog) -> None:
    last_truth = night.truths[-1]
    later = last_truth.t_utc_ns + 120 * NS
    expected_truth, _, _ = synth.star_truth(
        catalog,
        synth.cropped_profile(1200, 800),
        rotation_tirs=last_truth.rotation_tirs,
        t_utc_ns=later,
        exposure_s=0.001,
    )
    polaris_row = int(np.argmin(catalog.g_mag))
    where = int(np.flatnonzero(expected_truth.rows == polaris_row)[0])
    predicted = night.analyzer.tracker.polaris_position(later, "bin2")
    assert predicted is not None
    assert predicted == pytest.approx(
        (float(expected_truth.x[where]), float(expected_truth.y[where])), abs=0.05
    )


# --- A worker process ------------------------------------------------------------------------


def test_a_worker_process_gives_the_same_records_as_this_process(
    profile: Profile, catalog: CapCatalog, tmp_path: Path
) -> None:
    rotation = synth.make_attitude(0.9, 40.0, 25.0)
    truth0, _, _ = synth.star_truth(
        catalog, profile, rotation_tirs=rotation, t_utc_ns=synth.NIGHT_UTC_NS
    )
    frame1, _ = synth.render_frame(
        catalog,
        profile,
        rotation_tirs=rotation,
        t_utc_ns=synth.NIGHT_UTC_NS + 180 * NS,
        seed=51,
    )
    write_catalog(tmp_path / "cap.smcat", catalog)
    library = DarkLibrary(tmp_path / "calibration" / "darks")
    library.add_set(  # the renderer's bias (40 counts) and no dark current: the worker reads this
        np.full((800, 1200), 40, dtype=np.uint16),
        mode="bin2",
        gain=120,
        exposure_s=30.0,
        temperature_c=15.0,
        temperature_spread_c=0.1,
        t_utc_ns=synth.NIGHT_UTC_NS - 3600 * NS,
        n_frames=9,
        n_bias_frames=9,
        bias_dn=40.0,
        read_noise_dn=2.1,
        adc_bits=14,
        dark_dn=40.0,
    )
    config = SurveyConfig(
        catalog_path=str(tmp_path / "cap.smcat"),
        calibration_dir=str(tmp_path / "calibration"),
        solvers=(),
    )
    spec = analyzer_spec(profile=profile, station_id="test", config=config)
    # The tracker already holds a solution, so the worker needs no plate solver.
    truth_model = CameraAttitude(
        truth0.rotation_cirs, truth0.scale_arcsec_px / ARCSEC_PER_RAD, 1, truth0.center_px
    )
    seed_solution = pt.PointingSolution.from_attitude(
        truth_model, truth0.epoch, mode="bin2", width_px=1200, height_px=800, n_matched=300,
        rms_arcsec=0.05, solver="seed",
    )  # fmt: skip
    in_process = SurveyPipelineAnalyzer(
        profile=profile, station_id="test", spec=spec, executor=InlineExecutor()
    )
    in_process.tracker.update(seed_solution)
    in_process.submit(frame1)
    expected = in_process.poll()
    assert expected[0].solved

    executor = make_process_executor(spec)
    worker = SurveyPipelineAnalyzer(
        profile=profile, station_id="test", spec=spec, executor=executor
    )
    try:
        worker.tracker.update(seed_solution)
        worker.submit(frame1)
        outputs = collect(worker, 1, timeout_s=120.0)
    finally:
        executor.shutdown(wait=True)
    assert outputs[0].solved
    assert outputs[0].t_utc_ns == frame1.t_utc_ns
    assert [r.record_type for r in outputs[0].records] == [
        r.record_type for r in expected[0].records
    ]
    got = next(r for r in outputs[0].records if isinstance(r, PointingRecord))
    want = next(r for r in expected[0].records if isinstance(r, PointingRecord))
    assert got.solver == "tracker"
    assert got.n_matched == want.n_matched
    assert got.attitude is not None
    assert want.attitude is not None
    np.testing.assert_allclose(got.attitude, want.attitude, atol=1e-12)
    # The worker reads the dark library and makes the same sky quality.
    got_sky = next(r for r in outputs[0].records if isinstance(r, SkyQualityRecord))
    want_sky = next(r for r in expected[0].records if isinstance(r, SkyQualityRecord))
    assert want_sky.sky_mag_arcsec2 is not None
    assert want_sky.zero_point_mag is not None
    assert got_sky.zero_point_mag == pytest.approx(want_sky.zero_point_mag, abs=1e-9)
    assert got_sky.sky_mag_arcsec2 == pytest.approx(want_sky.sky_mag_arcsec2, abs=1e-9)
    assert got_sky.n_stars_used == want_sky.n_stars_used
    assert got_sky.dark_model_version == want_sky.dark_model_version
    assert got_sky.dark_model_version is not None


def test_create_survey_analyzer_builds_from_the_configuration(
    profile: Profile, catalog: CapCatalog, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="catalog_path"):
        create_survey_analyzer(profile=profile, station_id="s", config=SurveyConfig())
    write_catalog(tmp_path / "cap.smcat", catalog)
    config = SurveyConfig(catalog_path=str(tmp_path / "cap.smcat"), solvers=())
    analyzer = create_survey_analyzer(
        profile=profile, station_id="s", config=config, executor=InlineExecutor()
    )
    frame, _ = synth.render_frame(
        catalog, profile, rotation_tirs=synth.make_attitude(0.9, 40.0, 25.0), seed=60
    )
    analyzer.submit(frame)
    outputs = analyzer.poll()
    assert len(outputs) == 1
    assert not outputs[0].solved  # no solver and no previous solution
    pointing = next(r for r in outputs[0].records if isinstance(r, PointingRecord))
    assert isinstance(pointing, PointingRecord)
    assert pointing.flags == ["unsolved"]
    analyzer.close()
    assert replace(frame, seq=1).seq == 1


def test_a_built_pipeline_loads_the_catalog_and_the_hot_pixel_file(
    profile: Profile, catalog: CapCatalog, tmp_path: Path
) -> None:
    from seeingmon.survey.pipeline import build_pipeline

    write_catalog(tmp_path / "cap.smcat", catalog)
    mask = np.zeros((800, 1200), dtype=bool)
    mask[10, 10] = True
    np.save(tmp_path / "hot.npy", mask)
    config = SurveyConfig(
        catalog_path=str(tmp_path / "cap.smcat"),
        hot_pixel_file=str(tmp_path / "hot.npy"),
        solvers=(),
    )
    spec = analyzer_spec(profile=profile, station_id="s", config=config)
    assert spec.hot_pixel_file == str(tmp_path / "hot.npy")
    pipeline = build_pipeline(spec)
    assert pipeline.catalog.content_id == catalog.content_id
    frame, truth = synth.render_frame(
        catalog, profile, rotation_tirs=synth.make_attitude(0.9, 40.0, 25.0), seed=61
    )
    analysis = pipeline.analyze(frame)
    assert analysis.detections is not None
    assert analysis.cat_row is not None
    assert analysis.cat_row.shape == (len(analysis.detections),)
    assert not analysis.solved  # no solver, no previous solution
    assert truth.rows.size > 0
