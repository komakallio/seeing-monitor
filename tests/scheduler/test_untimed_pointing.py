"""A start with neither a synchronized clock nor a pointing solution, through the real analyzer.

A Pi 4 has no real-time clock. After a boot without a time source, its clock can be hours off, and
`core` starts the tracker empty, because a stored solution would place Polaris wrong by the error
of the clock. The first survey frame that solves must then fill the tracker: the error cancels
between the solve and the predictions, so the fast stream finds Polaris, and the windows carry
`time_invalid`. When chrony steps the clock back, the next survey solve has a valid time, and it
replaces the untimed solution, although the untimed one lies hours later.

The scheduler here gets a `SurveyPipelineAnalyzer` in place of the fake survey analysis, and the
tracker of the analyzer is its pointing provider. A scripted pipeline solves each long frame as
the real one does: the fit finds the attitude that the stars show at the true time, and the
solution stores it with the time of the frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")

from seeingmon.clock import NS_PER_S, ClockStatus, VirtualClock, iso_to_utc_ns
from seeingmon.frames import Frame
from seeingmon.records import SeeingWindowRecord
from seeingmon.records.survey import SurveyFrameRecord
from seeingmon.scheduler import Scheduler
from seeingmon.survey import apparent
from seeingmon.survey import pointing as pt
from seeingmon.survey.analyzer import InlineExecutor, SurveyPipelineAnalyzer
from seeingmon.survey.pipeline import FrameAnalysis, SurveyPipeline
from seeingmon.survey.transparency import ZeroPointReference
from tests.scheduler.scenario import PROFILE, SITE, ConfigureCall, World
from tests.survey import synth
from tests.survey.pointfx import made

TRUE_START = iso_to_utc_ns("2026-01-01T22:00:00Z")  # a dark sky at the synthetic site
CLOCK_AHEAD_NS = 6 * 3600 * NS_PER_S  # the clock after the boot runs 6 hours ahead
TRUE_POINTING = made(TRUE_START).solution  # the rigid mount, as a solution with a valid time
BIN2 = PROFILE.mode("bin2")
SYNCHRONIZED = ClockStatus(synchronized=True, error_bound_ns=1_000_000, source="test")
UNSYNCHRONIZED = ClockStatus(synchronized=False, error_bound_ns=None, source="test")


class BootWorld(World):
    """The scenario world, with Polaris where the rigid mount sees it at the true time."""

    clock_error_ns = CLOCK_AHEAD_NS

    def star_position(self, t_utc_ns: int, mode: str = "bin1") -> tuple[float, float]:
        position = TRUE_POINTING.polaris_pixel(t_utc_ns - self.clock_error_ns)
        assert position is not None
        factor = PROFILE.mode(mode).width_px / BIN2.width_px  # as the tracker converts
        return (position[0] + 0.5) * factor - 0.5, (position[1] + 0.5) * factor - 0.5


class MountPipeline(SurveyPipeline):
    """Solves each long frame of the world as the survey pipeline would, without pixels.

    A frame shorter than 1 s shows too few stars, as the 1 ms frame of a survey step does.
    """

    def __init__(self, world: BootWorld) -> None:  # nothing else of the pipeline runs
        self._profile = PROFILE
        self._catalog = synth.synthetic_catalog(cap_radius_deg=1.0, density_scale=0.01)
        self._world = world

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
        record = SurveyFrameRecord(
            station_id="test",
            t_utc_ns=frame.t_utc_ns,
            profile_id=PROFILE.id,
            provenance={"algo": "scripted"},
            exposure_s=frame.exposure_us / 1e6,
            gain=frame.gain,
            readout_mode=frame.mode,
        )
        if frame.exposure_us < 1_000_000:
            return FrameAnalysis(records=(record,), solved=False, cloud_fraction=None)
        seen = TRUE_POINTING.attitude_at(frame.t_utc_ns - self._world.clock_error_ns)
        solution = pt.PointingSolution.from_attitude(
            seen,
            apparent.epoch_from_utc_ns(frame.t_utc_ns),
            mode="bin2",
            width_px=BIN2.width_px,
            height_px=BIN2.height_px,
            n_matched=300,
            rms_arcsec=0.4,
            solver="fake",
        )
        return FrameAnalysis(records=(record,), solved=True, cloud_fraction=0.0, solution=solution)


@dataclass(frozen=True)
class Boot:
    """The world after the boot and the synchronization, with what held before the step."""

    world: BootWorld
    analyzer: SurveyPipelineAnalyzer
    untimed: pt.PointingSolution
    windows_before: list[SeeingWindowRecord]
    periods_before: list[ConfigureCall]
    star_in_roi_before: list[bool]  # for each of those fast periods
    synchronized_at_s: float  # the clock time of the step, in seconds after the true start


def fast_periods(world: World) -> list[ConfigureCall]:
    return world.configures(mode="bin1", video=True)


def star_in_roi(world: World, call: ConfigureCall) -> bool:
    roi = call.config.roi
    assert roi is not None
    return roi.contains(*world.star_position(call.t_utc_ns))


@pytest.fixture(scope="module")
def boot() -> Boot:
    clock = VirtualClock(TRUE_START + CLOCK_AHEAD_NS, status=UNSYNCHRONIZED)
    world = BootWorld(start_utc_ns=TRUE_START + CLOCK_AHEAD_NS, clock=clock, solved_at_start=False)
    analyzer = SurveyPipelineAnalyzer(
        profile=PROFILE,
        station_id="test",
        pipeline=MountPipeline(world),
        executor=InlineExecutor(),
    )
    # The scheduler that the world built never runs: this one takes the analyzer instead.
    world.scheduler = Scheduler(
        driver=world.camera,
        fast=world.fast,
        survey=analyzer,
        pointing=analyzer.tracker,
        records=world.writer,
        metrics=world.writer,
        clock=clock,
        profile=PROFILE,
        station_id="test",
        config=world.config,
        site=SITE,
    )
    world.run_until(900)
    untimed = analyzer.tracker.solution
    assert untimed is not None
    assert analyzer.tracker.untimed
    windows_before = world.windows()
    periods_before = fast_periods(world)
    in_roi = [star_in_roi(world, call) for call in periods_before]  # with the clock error
    # chrony synchronizes, and steps the clock back to the true time.
    clock.step_utc_ns(-CLOCK_AHEAD_NS)
    world.clock_error_ns = 0
    clock.set_status(SYNCHRONIZED)
    synchronized_at_s = (clock.utc_ns() - TRUE_START) / NS_PER_S
    world.run_for(900)
    world.close()
    return Boot(world, analyzer, untimed, windows_before, periods_before, in_roi, synchronized_at_s)


class TestBeforeTheClockSynchronizes:
    def test_the_first_untimed_solve_fills_the_empty_tracker_and_starts_the_fast_stream(
        self, boot: Boot
    ) -> None:
        requested = boot.world.events("scheduler.solve_requested")
        assert len(requested) == 1  # the first cycle only: its survey step fills the tracker
        assert boot.untimed.t_utc_ns > TRUE_START + CLOCK_AHEAD_NS  # stamped by the clock
        # A fast period every 180 s from the first solve, each with its ROI on the star, because
        # the error of the clock cancels between the solve and the prediction.
        assert len(boot.periods_before) >= 4
        assert all(boot.star_in_roi_before)

    def test_the_windows_carry_time_invalid(self, boot: Boot) -> None:
        assert len(boot.windows_before) >= 8  # 60 s windows of the fast periods
        assert all("time_invalid" in window.flags for window in boot.windows_before)
        assert not [w for w in boot.windows_before if "partial" in w.flags]  # the star stayed


class TestAfterTheClockSynchronizes:
    def test_the_first_timed_solve_replaces_the_untimed_solution_hours_before_it(
        self, boot: Boot
    ) -> None:
        tracker = boot.analyzer.tracker
        solution = tracker.solution
        assert solution is not None
        assert not tracker.untimed
        assert solution.t_utc_ns < boot.untimed.t_utc_ns  # hours before, by the true clock
        assert solution.t_utc_ns >= TRUE_START + round(boot.synchronized_at_s * NS_PER_S)

    def test_after_the_first_timed_solve_every_fast_period_finds_the_star(self, boot: Boot) -> None:
        synchronized_ns = TRUE_START + round(boot.synchronized_at_s * NS_PER_S)
        first_solve_ns = min(
            record.t_utc_ns
            for record in boot.world.records("survey_frame")
            if isinstance(record, SurveyFrameRecord)
            and record.exposure_s >= 1.0
            and synchronized_ns <= record.t_utc_ns < TRUE_START + CLOCK_AHEAD_NS
        )
        later = fast_periods(boot.world)[len(boot.periods_before) :]
        # After the step, the untimed solution points wrong by 6 hours of the Earth's rotation, so
        # a fast period that starts before the next survey step misses the star and ends early.
        stale = [call for call in later if call.t_utc_ns < first_solve_ns]
        assert len(stale) <= 1
        assert not any(star_in_roi(boot.world, call) for call in stale)
        fresh = [call for call in later if call.t_utc_ns > first_solve_ns]
        assert len(fresh) >= 4
        assert all(star_in_roi(boot.world, call) for call in fresh)
        assert boot.world.events("scheduler.solve_requested")[1:] == []

    def test_the_windows_after_the_synchronization_are_timed(self, boot: Boot) -> None:
        # The scheduler reads the status of the clock every few seconds.
        after = [
            w
            for w in boot.world.windows()
            if TRUE_START + round((boot.synchronized_at_s + 10) * NS_PER_S)
            <= w.t_utc_ns
            < TRUE_START + CLOCK_AHEAD_NS
        ]
        assert len(after) >= 8
        assert all("time_invalid" not in window.flags for window in after)
