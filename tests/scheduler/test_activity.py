"""The activity: what the scheduler says that it does, and what it then does.

The scenarios run a whole evening on the virtual clock, step by step, and take the status after each
step. Every state and every phase of the activity must appear, and the times that the activity
announces (`ends_utc_ns` and `next_utc_ns`) must agree with what the scheduler then does: the next
phase begins when the activity said that it would.
"""

from __future__ import annotations

import dataclasses
import itertools
import re
import threading
import time
from collections.abc import Callable
from typing import Any, ClassVar

import pytest

from seeingmon.clock import NS_PER_S, iso_to_utc_ns
from seeingmon.scheduler import (
    Command,
    CommissionContext,
    CommissionResult,
    CommissionTask,
    Pause,
    QueueBurst,
    QueueDark,
    QueueFlat,
    QueueSweep,
    Resume,
    SchedulerConfig,
    StartAlignment,
    StartRapidFocus,
    StopAlignment,
    StopRapidFocus,
)
from seeingmon.scheduler import activity as words
from seeingmon.scheduler.config import AlignConfig
from seeingmon.scheduler.status import ActivityPhase, ActivityStatus, SchedulerStatus
from tests.scheduler.scenario import START, TEST_CONFIG, World

NIGHT = iso_to_utc_ns("2026-01-01T22:00:00Z")
STATES = {"safe", "auto", "align", "commission", "paused"}
PHASES = {phase.value for phase in ActivityPhase}
# How far an announced time may differ from the time of what then happens. One fast frame takes 2
# seconds in these scenarios, and the first survey exposure uses the default for its overhead.
TOLERANCE_S = 2.5


# --- The words -------------------------------------------------------------------------------


class TestTheWords:
    @pytest.mark.parametrize(
        ("seconds", "text"),
        [
            (0.0, "0 s"),
            (0.00025, "250 us"),
            (0.001, "1 ms"),
            (0.0005, "500 us"),
            (0.25, "250 ms"),
            (0.5, "500 ms"),
            (1.0, "1 s"),
            (2.5, "2.5 s"),
            (30.0, "30 s"),
            (60.0, "1 min"),
            (140.0, "2 min 20 s"),
            (180.0, "3 min"),
            (1800.0, "30 min"),
            (3900.0, "1 h 5 min"),
            (7200.0, "2 h"),
        ],
    )
    def test_a_duration_reads_in_its_plainest_unit(self, seconds: float, text: str) -> None:
        assert words.duration_text(seconds) == text

    def test_the_survey_step_names_both_exposures(self) -> None:
        assert words.survey_step_label(0.001, 30.0) == "Survey step: a 1 ms and a 30 s frame"
        assert words.survey_frame_label(30.0) == "Survey step: the 30 s frame"

    def test_the_windows_of_a_fast_period_count_up_to_the_total(self) -> None:
        assert words.fast_detail(20.0, 4, 7, clouds=False) == "Windows of 20 s: 4 of 7 closed"
        assert words.fast_detail(60.0, 3, 1, clouds=True) == (
            "Windows of 1 min: 1 of 1 closed; clouds shorten the period"
        )

    def test_the_daylight_gate_names_the_limit(self) -> None:
        assert words.daylight_label(-4.0) == "Daylight gate: the Sun is above -4 degrees"

    def test_a_reason_that_is_a_code_reads_as_words(self) -> None:
        assert words.state_reason_text("startup") == (
            "the scheduler started, and it checks the sky first"
        )
        assert words.state_reason_text("the sky is dark enough") == "the sky is dark enough"
        assert words.state_reason_text("") is None
        assert words.state_reason_text(None) is None

    def test_a_dark_session_names_its_phase(self) -> None:
        assert words.task_label("dark", "cover") == "Dark session: waiting for the cover"
        assert words.task_label("dark", None) == "Dark session"
        assert words.task_label("dark", "capture") == "Dark session"  # a phase of another kind
        assert words.task_label("burst") == "Burst: recording raw frames"
        assert words.task_label(None) == words.TASK_STARTING_LABEL

    def test_a_flat_session_names_its_phase(self) -> None:
        assert words.task_label("flat", None) == "Flat session"
        assert [
            words.task_label("flat", phase) for phase in ("setup", "exposure", "capture", "build")
        ] == [
            "Flat session: setting up the camera",
            "Flat session: finding the exposure",
            "Flat session: taking frames",
            "Flat session: combining the frames",
        ]
        assert words.task_label("flat", "cover") == "Flat session"  # a phase of another kind
        assert words.state_reason_text("a flat session starts at once") == (
            "a flat session starts at once"
        )

    def test_a_task_detail_joins_the_message_and_the_queue(self) -> None:
        assert (
            words.task_detail(2, "Bias frame 3 of 9.") == "Bias frame 3 of 9; 2 more tasks queued"
        )
        assert words.task_detail(1, None) == "1 more task queued"
        assert words.task_detail(0, None) is None

    def test_a_ladder_step_reads_as_an_action(self) -> None:
        assert words.recovery_label("usb_reset") == "Recovery step: reset the USB device"
        assert words.step_text("a_new_step") == "a new step"


# --- Sampling an evening ---------------------------------------------------------------------


def send(command: Command) -> Callable[[World], None]:
    """A scripted action that hands a command to the scheduler."""

    def action(world: World) -> None:
        world.scheduler.submit(command)

    return action


def one_step_samples(world: World, until_s: float) -> list[SchedulerStatus]:
    """Step the scheduler until `until_s` seconds after the start, and take the status after each
    step, so that no change hides between two samples. The scripted actions fire on time."""
    target = world.t(until_s)
    statuses: list[SchedulerStatus] = []
    while world.clock.utc_ns() < target:
        world._fire_due()
        world.scheduler.step()
        statuses.append(world.scheduler.status())
    return statuses


@dataclasses.dataclass(slots=True)
class Episode:
    """A run of samples with one state and one phase."""

    state: str
    phase: str
    first: SchedulerStatus
    last: SchedulerStatus

    @property
    def announced(self) -> ActivityStatus:
        activity = self.last.activity
        assert activity is not None
        return activity

    @property
    def began_ns(self) -> int:
        activity = self.first.activity
        assert activity is not None
        return activity.since_utc_ns


def episodes(statuses: list[SchedulerStatus]) -> list[Episode]:
    found: list[Episode] = []
    for status in statuses:
        activity = status.activity
        assert activity is not None
        if found and (found[-1].state, found[-1].phase) == (activity.state, activity.phase):
            found[-1].last = status
        else:
            found.append(Episode(activity.state, activity.phase, status, status))
    return found


# --- The evening -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def evening() -> tuple[World, list[SchedulerStatus]]:
    """An afternoon that turns into a night with every kind of activity.

    The Sun sets, and the daylight gate opens at about 6000 s. The first survey step finds no
    pointing and the analysis fails, so the scheduler waits and tries again. Clouds shorten the
    cycle. Alignment runs from 8000 s to 8100 s, a sweep follows, a pause lasts from 9300 s to
    9400 s, the camera fails from 10000 s to 10100 s, and a floodlight brightens the sky from
    11000 s to 11400 s. A second alignment runs from 12000 s to 12200 s, with rapid focus from
    12040 s to 12060 s.
    """
    world = World(start_utc_ns=START, solved_at_start=False, survey_polls=2)
    world.no_solution(6000, 6400)
    world.cloud(6600, 7400, 0.8)
    world.at(8000, send(StartAlignment()))
    world.at(8100, send(StopAlignment()))
    world.at(8400, send(QueueSweep(exposure_us=(500, 1000), gain=(0,), window_s=4.0)))
    world.at(9300, send(Pause()))
    world.at(9400, send(Resume()))
    world.camera_fault(10000, 10100)
    world.light(11000, 11400, 0.8)
    world.at(12000, send(StartAlignment()))
    star_x, star_y = world.star_position(world.t(12040))
    world.at(12040, send(StartRapidFocus(star_x, star_y, exposure_us=2000)))
    world.at(12060, send(StopRapidFocus()))
    world.at(12200, send(StopAlignment()))
    statuses = one_step_samples(world, 13000)
    return world, statuses


def phases_seen(statuses: list[SchedulerStatus]) -> set[tuple[str, str]]:
    return {(s.activity.state, s.activity.phase) for s in statuses if s.activity is not None}


class TestEveryStateAndPhase:
    def test_every_state_and_phase_appears_in_the_evening(
        self, evening: tuple[World, list[SchedulerStatus]]
    ) -> None:
        _, statuses = evening
        seen = phases_seen(statuses)
        assert {
            ("safe", "watch"),
            ("auto", "survey_short"),
            ("auto", "survey_long"),
            ("auto", "solve_wait"),
            ("auto", "idle"),
            ("auto", "fast"),
            ("align", "align"),
            ("align", "rapid_focus"),
            ("commission", "commission"),
            ("paused", "paused"),
            ("auto", "camera_fault"),
            ("safe", "camera_fault"),
        } <= seen
        assert {phase for _, phase in seen} == PHASES

    def test_every_status_of_the_evening_has_an_activity_that_agrees_with_the_state(
        self, evening: tuple[World, list[SchedulerStatus]]
    ) -> None:
        _, statuses = evening
        for status in statuses:
            activity = status.activity
            assert activity is not None
            assert activity.state == status.state
            assert activity.state in STATES
            assert activity.phase in PHASES
            assert activity.label

    def test_an_activity_never_begins_in_the_future_or_ends_before_it_begins(
        self, evening: tuple[World, list[SchedulerStatus]]
    ) -> None:
        _, statuses = evening
        for status in statuses:
            activity = status.activity
            assert activity is not None
            assert activity.since_utc_ns <= status.t_utc_ns
            if activity.ends_utc_ns is not None:
                assert activity.ends_utc_ns >= activity.since_utc_ns

    def test_the_cadence_belongs_to_the_cycle_of_auto(
        self, evening: tuple[World, list[SchedulerStatus]]
    ) -> None:
        _, statuses = evening
        cycle = {"fast", "survey_short", "survey_long", "solve_wait", "idle"}
        for status in statuses:
            activity = status.activity
            assert activity is not None
            if activity.phase in cycle:
                assert activity.cadence_s in (180.0, 100.0)
            else:
                assert activity.cadence_s is None

    def test_clouds_shorten_the_cadence_and_the_fast_period(
        self, evening: tuple[World, list[SchedulerStatus]]
    ) -> None:
        _, statuses = evening
        cloudy = [
            s.activity for s in statuses if s.activity is not None and s.activity.cadence_s == 100.0
        ]
        assert cloudy
        fast = [a for a in cloudy if a.phase == "fast"]
        assert fast
        for activity in fast:
            assert activity.ends_utc_ns is not None
            assert (activity.ends_utc_ns - activity.since_utc_ns) / NS_PER_S == pytest.approx(60.0)
            assert activity.detail is not None
            assert "clouds shorten the period" in activity.detail


class TestWhatTheScheduleAnnounces:
    """The end of a phase and the start of the next one match what the scheduler then does."""

    # The phase that follows each phase of the cycle when nothing interrupts it.
    SUCCESSORS: ClassVar[dict[str, set[str]]] = {
        "fast": {"survey_short"},
        "survey_short": {"survey_long"},
        "survey_long": {"idle", "fast", "solve_wait"},
        "idle": {"fast", "survey_short"},
    }

    @classmethod
    def uninterrupted(cls, runs: list[Episode]) -> list[tuple[Episode, Episode]]:
        """The pairs of neighbors in which one phase of the cycle gives way to its successor.

        A cloud result that arrives while the camera idles changes the cadence, a camera fault ends
        a period early, and a command ends the cycle. None of them is a mistake of the activity, so
        the pairs leave them out.
        """
        pairs = []
        for current, following in itertools.pairwise(runs):
            if current.state != "auto" or current.phase not in cls.SUCCESSORS:
                continue
            if (following.state, following.phase) not in {
                ("auto", successor) for successor in cls.SUCCESSORS[current.phase]
            }:
                continue
            if current.announced.cadence_s != following.last.activity.cadence_s:  # type: ignore[union-attr]
                continue
            pairs.append((current, following))
        return pairs

    def test_a_phase_ends_when_it_said_that_it_would(
        self, evening: tuple[World, list[SchedulerStatus]]
    ) -> None:
        _, statuses = evening
        checked = dict.fromkeys(self.SUCCESSORS, 0)
        for current, following in self.uninterrupted(episodes(statuses)):
            ends = current.announced.ends_utc_ns
            assert ends is not None, current.phase
            gap = abs(following.began_ns - ends) / NS_PER_S
            assert gap <= TOLERANCE_S, (current.phase, gap)
            checked[current.phase] += 1
        assert all(count >= 10 for count in checked.values()), checked

    def test_the_wait_for_a_pointing_solution_ends_by_its_deadline(
        self, evening: tuple[World, list[SchedulerStatus]]
    ) -> None:
        _, statuses = evening
        runs = episodes(statuses)
        waits = [
            (current, following)
            for current, following in itertools.pairwise(runs)
            if current.phase == "solve_wait"
        ]
        assert len(waits) >= 2
        for current, following in waits:
            ends = current.announced.ends_utc_ns
            assert ends is not None
            assert following.began_ns <= ends + NS_PER_S // 2
            assert current.announced.next_utc_ns is None  # the solution can come at any time

    def test_the_next_activity_starts_when_the_activity_said_that_it_would(
        self, evening: tuple[World, list[SchedulerStatus]]
    ) -> None:
        _, statuses = evening
        runs = episodes(statuses)
        survey_step = words.survey_step_label(
            TEST_CONFIG.survey.short_exposure_s, TEST_CONFIG.survey.long_exposure_s
        )
        starts = {
            words.FAST_LABEL: "fast",
            survey_step: "survey_short",
            words.survey_frame_label(TEST_CONFIG.survey.long_exposure_s): "survey_long",
            words.SOLVE_WAIT_LABEL: "solve_wait",
        }
        checked = 0
        for index, current in enumerate(runs):
            activity = current.announced
            if current.state != "auto" or activity.next_utc_ns is None:
                continue
            phase = starts.get(activity.next_label or "")
            if phase is None:
                continue
            following = None
            for later in runs[index + 1 : index + 4]:
                if later.state != "auto" or later.phase in {"camera_fault"}:
                    break  # a command or a fault came in between
                if later.phase == phase:
                    following = later
                    break
            if following is None:
                continue
            if following.first.activity.cadence_s != activity.cadence_s:  # type: ignore[union-attr]
                continue  # a cloud result changed the cadence in the meantime
            assert abs(following.began_ns - activity.next_utc_ns) / NS_PER_S <= TOLERANCE_S, (
                current.phase,
                activity.next_label,
            )
            checked += 1
        assert checked >= 30

    def test_the_survey_exposures_take_what_the_activity_expects_after_the_first_cycle(
        self, evening: tuple[World, list[SchedulerStatus]]
    ) -> None:
        """The scheduler measures the overhead of an exposure, so later estimates are tight."""
        _, statuses = evening
        runs = [r for r in episodes(statuses) if r.phase == "survey_long"]
        runs = runs[3:]
        assert len(runs) >= 10
        for run in runs:
            ends = run.announced.ends_utc_ns
            assert ends is not None
            length = (ends - run.announced.since_utc_ns) / NS_PER_S
            assert length == pytest.approx(30.03, abs=0.2)

    def test_the_idle_time_is_the_slack_of_the_cadence(
        self, evening: tuple[World, list[SchedulerStatus]]
    ) -> None:
        _, statuses = evening
        for run in episodes(statuses):
            if run.phase != "idle":
                continue
            activity = run.announced
            assert activity.ends_utc_ns is not None
            assert activity.next_utc_ns == activity.ends_utc_ns
            assert activity.detail is not None
            assert activity.detail.startswith("The camera rests for") or activity.detail.startswith(
                "The scheduler waits"
            )

    def test_the_fast_period_counts_its_windows(
        self, evening: tuple[World, list[SchedulerStatus]]
    ) -> None:
        world, statuses = evening
        counts: list[tuple[int, int]] = []
        for status in statuses:
            activity = status.activity
            assert activity is not None
            if activity.phase != "fast" or activity.ends_utc_ns is None:
                continue
            if (activity.ends_utc_ns - activity.since_utc_ns) / NS_PER_S == pytest.approx(120.0):
                assert activity.detail is not None
                found = re.fullmatch(
                    r"Windows of 1 min: (\d+) of (\d+) closed(;.*)?", activity.detail
                )
                assert found, activity.detail
                counts.append((int(found.group(1)), int(found.group(2))))
        assert {total for _, total in counts} == {2}  # a period of 120 s holds two windows of 60 s
        # The last window closes with the period itself, so a status of the period never shows it.
        assert {closed for closed, _ in counts} == {0, 1}
        assert world.windows()  # the windows that the count refers to were written


class TestTheStatesThatTheCommandsChange:
    def test_the_daylight_gate_holds_the_scheduler_in_safe_at_the_start(
        self, evening: tuple[World, list[SchedulerStatus]]
    ) -> None:
        _, statuses = evening
        first = statuses[0].activity
        assert first is not None
        assert (first.state, first.phase) == ("safe", "watch")
        assert first.label == "Daylight gate: the Sun is above -4 degrees"
        assert first.reason == "the scheduler started, and it checks the sky first"
        assert first.ends_utc_ns is None
        assert first.next_label == words.WATCH_NEXT_LABEL
        assert first.next_utc_ns is not None
        assert first.since_utc_ns == START
        assert first.detail == "The camera takes a 1 ms frame every 1 min"

    def test_the_next_brightness_frame_comes_when_the_watch_said(
        self, evening: tuple[World, list[SchedulerStatus]]
    ) -> None:
        world, statuses = evening
        daylight = [s for s in statuses if s.state == "safe" and s.t_utc_ns < world.t(5000)]
        announced = {s.activity.next_utc_ns for s in daylight if s.activity is not None}
        # The brightness frame is the only snapshot with an ROI: a survey frame has the full frame.
        taken = [
            call.t_utc_ns
            for call in world.configures(mode="bin2", video=False)
            if call.config.roi is not None
        ]
        assert len(announced) >= 50
        for when in announced:
            assert when is not None
            assert min(abs(when - t) for t in taken) < NS_PER_S, world.seconds(when)

    def test_alignment_shows_the_idle_timeout(
        self, evening: tuple[World, list[SchedulerStatus]]
    ) -> None:
        _, statuses = evening
        run = next(r for r in episodes(statuses) if r.phase == "align")
        activity = run.first.activity
        assert activity is not None
        assert activity.label == "Aligning (the live view)"
        assert activity.reason == "you started the alignment"
        assert activity.ends_utc_ns is not None
        timeout_s = (activity.ends_utc_ns - activity.since_utc_ns) / NS_PER_S
        assert timeout_s == pytest.approx(TEST_CONFIG.align.idle_timeout_s, abs=1.0)
        assert activity.next_label == words.ALIGN_NEXT_LABEL
        assert activity.next_utc_ns == activity.ends_utc_ns
        assert activity.cadence_s is None

    def test_a_stopped_alignment_goes_back_to_the_brightness_check(
        self, evening: tuple[World, list[SchedulerStatus]]
    ) -> None:
        _, statuses = evening
        runs = episodes(statuses)
        index = next(i for i, r in enumerate(runs) if r.phase == "align")
        following = runs[index + 1]
        assert (following.state, following.phase) == ("safe", "watch")
        reason = following.first.activity.reason  # type: ignore[union-attr]
        assert reason == "you stopped the alignment, so the scheduler checks the sky first"

    def test_a_queued_task_moves_the_state_to_commission_and_back_to_the_cycle(
        self, evening: tuple[World, list[SchedulerStatus]]
    ) -> None:
        """The sweep runs inside one step, so the status of the evening sees only its boundaries."""
        _, statuses = evening
        runs = episodes(statuses)
        index = next(i for i, r in enumerate(runs) if r.phase == "commission")
        activity = runs[index].first.activity
        assert activity is not None
        assert activity.label == words.TASK_STARTING_LABEL
        assert activity.next_label == words.AFTER_TASK_AUTO
        assert activity.reason == "a commissioning task is queued"
        assert activity.ends_utc_ns is None
        assert (runs[index + 1].state, runs[index + 1].phase) == ("auto", "fast")
        assert runs[index + 1].first.activity.reason == "the commissioning tasks are done"  # type: ignore[union-attr]

    def test_a_pause_shows_why_and_what_resumes(
        self, evening: tuple[World, list[SchedulerStatus]]
    ) -> None:
        _, statuses = evening
        run = next(r for r in episodes(statuses) if r.phase == "paused")
        activity = run.first.activity
        assert activity is not None
        assert activity.label == words.PAUSED_LABEL
        assert activity.reason == "you paused the scheduler"
        assert activity.next_label == words.PAUSED_NEXT_LABEL
        assert activity.ends_utc_ns is None

    def test_a_bright_sky_holds_the_gate_in_words(
        self, evening: tuple[World, list[SchedulerStatus]]
    ) -> None:
        world, statuses = evening
        bright = [
            s.activity
            for s in statuses
            if s.activity is not None
            and s.activity.phase == "watch"
            and s.activity.label == words.BRIGHT_SKY_LABEL
        ]
        assert bright
        assert bright[0].detail is not None
        assert "of saturation; the cycle resumes below 35%" in bright[0].detail
        assert world.scheduler.status().state == "auto"  # the gate opened again


class TestACameraFault:
    def test_the_fault_names_the_next_try_and_matches_the_status(
        self, evening: tuple[World, list[SchedulerStatus]]
    ) -> None:
        _, statuses = evening
        faults = [s for s in statuses if s.activity and s.activity.phase == "camera_fault"]
        assert faults
        for status in faults:
            activity = status.activity
            assert activity is not None
            assert activity.next_utc_ns == status.fault.next_attempt_utc_ns
            assert activity.next_label == words.recovery_label(status.fault.next_step or "")
            assert activity.since_utc_ns <= status.t_utc_ns
            assert activity.ends_utc_ns is None
            assert activity.cadence_s is None

    def test_the_episode_has_one_start_while_the_failures_climb(
        self, evening: tuple[World, list[SchedulerStatus]]
    ) -> None:
        _, statuses = evening
        faults = [s for s in statuses if s.activity and s.activity.phase == "camera_fault"]
        assert {s.activity.since_utc_ns for s in faults} == {  # type: ignore[union-attr]
            faults[0].activity.since_utc_ns  # type: ignore[union-attr]
        }
        assert faults[0].fault.failures == 1
        assert max(s.fault.failures for s in faults) >= 5

    def test_the_state_follows_the_failures(
        self, evening: tuple[World, list[SchedulerStatus]]
    ) -> None:
        _, statuses = evening
        faults = [s for s in statuses if s.activity and s.activity.phase == "camera_fault"]
        for status in faults:
            assert status.state == ("safe" if status.degraded else "auto")
        degraded = [s for s in faults if s.degraded]
        assert degraded
        # The evening's fault is a timeout, and the label of a degraded status names its cause.
        assert degraded[0].activity.label == "Camera fault: no frame arrives"  # type: ignore[union-attr]
        assert degraded[0].activity.reason == (  # type: ignore[union-attr]
            "no frame arrived; the camera may be disconnected"
        )
        assert faults[0].activity.label == "Camera fault: recovering"  # type: ignore[union-attr]

    def test_a_camera_that_works_again_says_that_it_recovers(
        self, evening: tuple[World, list[SchedulerStatus]]
    ) -> None:
        _, statuses = evening
        notes = [
            s.activity
            for s in statuses
            if s.activity is not None
            and s.activity.phase != "camera_fault"
            and s.fault.failures
            and s.activity.detail is not None
            and "the camera recovers" in s.activity.detail
        ]
        assert notes


# --- A task that reports while it runs --------------------------------------------------------


class RecordingHandler:
    """A commissioning handler that takes the activity from inside the loop while it runs.

    A real handler holds the camera for the whole call, so no other step of the loop sees it. The
    handler asks for the status itself, which is what a thread of the web process would see.
    """

    def __init__(
        self,
        world: World,
        phases: tuple[tuple[Any, ...], ...] = (),
        *,
        run_s: float = 20.0,
        event: str = "scheduler.dark_phase",
    ) -> None:
        self._world = world
        self._phases = phases  # (phase, message), and optionally the rest of the detail
        self._run_s = run_s
        self._event = event
        self.seen: list[ActivityStatus] = []
        self.times: list[int] = []  # the UTC time of each look

    def _look(self) -> None:
        activity = self._world.scheduler.status().activity
        assert activity is not None
        self.seen.append(activity)
        self.times.append(self._world.clock.utc_ns())

    def run(self, task: CommissionTask, context: CommissionContext) -> CommissionResult:
        started = self._world.clock.utc_ns()
        self._look()
        for phase, message, *more in self._phases:
            context.emit_event(
                "info",
                self._event,
                message,
                {"task_id": task.task_id, "phase": phase, "steps": 3, **(more[0] if more else {})},
            )
            self._look()
            context.clock.sleep(5.0)
        context.clock.sleep(self._run_s)
        return CommissionResult(
            task_id=task.task_id,
            kind=task.kind,
            status="ok",
            summary="done",
            started_utc_ns=started,
            finished_utc_ns=self._world.clock.utc_ns(),
        )


class TestACommissioningTask:
    def test_a_burst_knows_when_it_ends(self) -> None:
        world = World(start_utc_ns=NIGHT)
        handler = RecordingHandler(world)
        world.scheduler.register_handler("burst", handler)
        world.at(200, send(QueueBurst(duration_s=20.0)))
        world.run_until(600)
        (activity,) = handler.seen
        assert (activity.state, activity.phase) == ("commission", "commission")
        assert activity.label == "Burst: recording raw frames"
        assert activity.ends_utc_ns is not None
        assert (activity.ends_utc_ns - activity.since_utc_ns) / NS_PER_S == pytest.approx(20.0)
        assert activity.next_label == words.AFTER_TASK_AUTO
        assert activity.next_utc_ns == activity.ends_utc_ns
        assert activity.cadence_s is None
        world.close()

    def test_a_dark_session_names_each_phase_as_it_reports_it(self) -> None:
        world = World(start_utc_ns=NIGHT)
        handler = RecordingHandler(
            world,
            (
                ("bias", "Bias frame 1 of 3."),
                ("cover", "Cover the camera now. Waiting for a dark frame."),
                ("dark", "Dark frame 1 of 3."),
                ("build", "Building the master dark and the library."),
            ),
        )
        world.scheduler.register_handler("dark", handler)
        world.at(200, send(QueueDark(frames=3, bias_frames=3)))
        world.run_until(600)
        assert [a.label for a in handler.seen] == [
            "Dark session",
            "Dark session: taking bias frames",
            "Dark session: waiting for the cover",
            "Dark session: taking dark frames",
            "Dark session: building the master dark",
        ]
        waiting = handler.seen[2]
        assert waiting.detail == "Cover the camera now. Waiting for a dark frame"
        assert waiting.reason == "a dark session starts at once"
        assert waiting.next_label == words.AFTER_TASK_PAUSED  # `pause_after` is the default
        assert waiting.ends_utc_ns is None  # the wait for the cover has no known end
        after = world.scheduler.status().activity
        assert after is not None
        assert (after.state, after.phase) == ("paused", "paused")
        assert after.reason == "the dark session is done, and the camera may still be covered"
        world.close()

    def test_a_flat_session_names_each_phase_and_announces_the_end_of_its_frames(self) -> None:
        world = World(start_utc_ns=NIGHT)
        handler = RecordingHandler(
            world,
            (
                ("setup", "Setting up the camera and the library."),
                ("exposure", "Finding the exposure that reaches 50 % of full scale."),
                ("capture", "Taking 32 frames of 39 ms.", {"expected_s": 40.0}),
                ("build", "Combining the frames into a flat."),
            ),
            event="scheduler.flat_phase",
        )
        world.scheduler.register_handler("flat", handler)
        world.at(200, send(QueueFlat()))
        world.run_until(600)
        assert [a.label for a in handler.seen] == [
            "Flat session",
            "Flat session: setting up the camera",
            "Flat session: finding the exposure",
            "Flat session: taking frames",
            "Flat session: combining the frames",
        ]
        capture = handler.seen[3]
        assert capture.detail == "Taking 32 frames of 39 ms"
        assert capture.reason == "a flat session starts at once"
        assert capture.next_label == words.AFTER_TASK_PAUSED  # `pause_after` is the default
        assert capture.next_utc_ns is None  # the next phase is of the same session
        assert capture.ends_utc_ns is not None
        assert (capture.ends_utc_ns - handler.times[3]) / NS_PER_S == pytest.approx(40.0)
        # The other phases do not know their length, and the first look is before any phase.
        assert [a.ends_utc_ns for a in handler.seen if a is not capture] == [None] * 4
        assert len({a.since_utc_ns for a in handler.seen}) == 1  # the session began once
        after = world.scheduler.status().activity
        assert after is not None
        assert (after.state, after.phase) == ("paused", "paused")
        assert (
            after.reason
            == "the flat session is done, and the light source may still cover the camera"
        )
        world.close()

    @pytest.mark.parametrize("expected", [0, -5.0, float("nan"), float("inf"), "40", True, None])
    def test_a_phase_with_a_length_that_makes_no_sense_announces_no_end(
        self, expected: Any
    ) -> None:
        world = World(start_utc_ns=NIGHT)
        handler = RecordingHandler(
            world,
            (("capture", "Taking 32 frames.", {"expected_s": expected}),),
            event="scheduler.flat_phase",
        )
        world.scheduler.register_handler("flat", handler)
        world.at(200, send(QueueFlat()))
        world.run_until(600)
        assert handler.seen[1].label == "Flat session: taking frames"
        assert handler.seen[1].ends_utc_ns is None
        world.close()

    def test_the_tasks_that_wait_behind_the_running_one_are_counted(self) -> None:
        world = World(start_utc_ns=NIGHT)
        handler = RecordingHandler(world)
        world.scheduler.register_handler("burst", handler)

        def queue_three(w: World) -> None:
            for _ in range(3):
                w.scheduler.submit(QueueBurst(duration_s=5.0))

        world.at(200, queue_three)
        world.run_until(800)
        assert [a.detail for a in handler.seen] == [
            "2 more tasks queued",
            "1 more task queued",
            None,
        ]
        world.close()


# --- A status that is read while the loop runs -----------------------------------------------


def test_a_status_read_from_another_thread_always_has_an_activity() -> None:
    """The activity reads the state of the loop without its lock, so it must never fail."""
    world = World(start_utc_ns=NIGHT, solved_at_start=False, survey_polls=2)
    world.cloud(1000, 2000, 0.8)
    world.camera_fault(3000, 3100)
    world.at(1500, send(StartAlignment()))
    world.at(1700, send(StopAlignment()))
    stop = threading.Event()
    problems: list[str] = []
    reads = 0

    def poll() -> None:
        nonlocal reads
        while not stop.is_set():
            activity = world.scheduler.status().activity
            reads += 1
            if activity is None:
                problems.append("no activity")
            elif activity.phase not in PHASES:
                problems.append(activity.phase)
            time.sleep(0)  # let the loop run

    reader = threading.Thread(target=poll, daemon=True)
    reader.start()
    try:
        world.run_until(3600)
    finally:
        stop.set()
        reader.join(5.0)
    assert reads > 100
    assert problems == []


# --- Small things ----------------------------------------------------------------------------


def test_the_alignment_timeout_moves_when_someone_uses_the_helper() -> None:
    config = SchedulerConfig(
        fast=TEST_CONFIG.fast, loop=TEST_CONFIG.loop, align=AlignConfig(idle_timeout_s=600.0)
    )
    world = World(start_utc_ns=NIGHT, config=config)
    world.run_until(100)
    world.scheduler.submit(StartAlignment())
    world.run_until(400)
    before = world.scheduler.status().activity
    assert before is not None
    assert before.ends_utc_ns is not None
    world.scheduler.touch_alignment()
    after = world.scheduler.status().activity
    assert after is not None
    assert after.ends_utc_ns is not None
    assert after.ends_utc_ns - before.ends_utc_ns == pytest.approx(300 * NS_PER_S, abs=NS_PER_S)
    assert before.since_utc_ns == after.since_utc_ns  # the activity began once
    assert after.detail == "The alignment ends after 10 min without use; it has been idle for 0 s"
    world.close()


def test_a_scheduler_that_has_not_started_reports_the_first_brightness_frame() -> None:
    world = World(start_utc_ns=NIGHT)
    activity = world.scheduler.status().activity
    assert activity is not None
    assert (activity.state, activity.phase) == ("safe", "watch")
    assert activity.label == words.FIRST_FRAME_LABEL
    assert activity.next_utc_ns == NIGHT
    world.close()


def test_the_activity_survives_the_json_of_the_status() -> None:
    world = World(start_utc_ns=NIGHT)
    world.run_until(10)
    status = world.scheduler.status()
    data: dict[str, Any] = dataclasses.asdict(status)
    assert set(data["activity"]) == {
        "state",
        "phase",
        "label",
        "since_utc_ns",
        "ends_utc_ns",
        "next_label",
        "next_utc_ns",
        "cadence_s",
        "detail",
        "reason",
    }
    world.close()


def test_a_scheduler_without_a_site_has_no_daylight_gate_to_name() -> None:
    """Without a site the scheduler judges the sky by its brightness alone, and says so."""
    world = World(start_utc_ns=NIGHT, site=None)
    first = world.scheduler.status().activity
    assert first is not None
    assert first.label == words.FIRST_FRAME_LABEL
    world.run_until(5)
    activity = world.scheduler.status().activity
    assert activity is not None
    assert activity.state in {"safe", "auto"}
    assert "Daylight gate" not in activity.label
    world.close()
