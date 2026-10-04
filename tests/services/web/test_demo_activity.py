"""The activity of the demo: a scripted evening that moves with the clock."""

from __future__ import annotations

import itertools

import pytest

from seeingmon.clock import NS_PER_S, VirtualClock
from seeingmon.scheduler.commands import Pause, QueueDark, Resume, StartAlignment, StopAlignment
from seeingmon.scheduler.status import ActivityPhase
from seeingmon.services.web import demo_activity
from seeingmon.services.web.contract import ActivityView
from seeingmon.services.web.demo import DEMO_DARK_SCRIPT, DEMO_NOW_NS, DemoCore

AUTO_PHASES = {"fast", "survey_short", "survey_long", "solve_wait", "idle", "camera_fault"}


class StandingClock(VirtualClock):
    """The clock of the real demo: UTC stands still, and the monotonic clock runs."""

    def utc_ns(self) -> int:
        return DEMO_NOW_NS


def activity(core: DemoCore) -> ActivityView:
    view = core.status().scheduler.activity
    assert view is not None
    return view


def play(core: DemoCore, clock: VirtualClock, seconds: float) -> ActivityView:
    clock.advance(seconds)
    return activity(core)


@pytest.fixture
def clock() -> VirtualClock:
    return VirtualClock(DEMO_NOW_NS)


@pytest.fixture
def core(clock: VirtualClock) -> DemoCore:
    return DemoCore(clock)


class TestThePlaylist:
    def test_the_segments_follow_each_other_without_a_gap(self) -> None:
        playlist = demo_activity.PLAYLIST
        assert playlist[0].start_s == 0.0
        for earlier, later in itertools.pairwise(playlist):
            assert later.start_s == earlier.end_s
        assert playlist[-1].end_s == demo_activity.PLAYLIST_S

    def test_the_playlist_shows_every_phase_of_auto(self) -> None:
        assert {segment.phase.value for segment in demo_activity.PLAYLIST} == AUTO_PHASES

    def test_a_cycle_is_shorter_than_a_real_one_and_adds_up(self) -> None:
        assert demo_activity.CYCLE_S == 45.0
        fast = demo_activity.FAST_WINDOW_S * demo_activity.FAST_WINDOWS
        assert fast + 2.0 + 8.0 + 5.0 == demo_activity.CYCLE_S
        solve, cycles, fault = 16.0, 3 * 45.0, 25.0
        length = demo_activity.PLAYLIST_S
        assert length == pytest.approx(solve + cycles + fault)

    def test_the_camera_fault_comes_with_two_steps_of_the_ladder(self) -> None:
        faults = [s for s in demo_activity.PLAYLIST if s.phase is ActivityPhase.CAMERA_FAULT]
        assert [s.next_step for s in faults] == ["restart_capture", "reopen"]
        assert {s.episode_start_s for s in faults} == {faults[0].start_s}


class TestAuto:
    def test_the_demo_starts_with_the_survey_step_that_finds_the_pointing(
        self, core: DemoCore, clock: VirtualClock
    ) -> None:
        first = activity(core)
        assert (first.state, first.phase) == ("auto", "survey_short")
        assert first.label == "Survey step: the 1 ms frame"
        assert first.reason == "the sky is dark enough"
        second = play(core, clock, 5.0)
        assert second.phase == "survey_long"
        assert second.label == "Survey step: the 6 s frame"
        third = play(core, clock, 6.0)
        assert third.phase == "solve_wait"
        assert third.label == "Waiting for a pointing solution"
        assert third.next_utc_ns is None  # the solution can arrive at any time

    def test_the_phases_move_with_the_clock_through_every_phase_and_repeat(
        self, core: DemoCore, clock: VirtualClock
    ) -> None:
        seen: list[str] = []
        for _ in range(int(2 * demo_activity.PLAYLIST_S)):
            phase = activity(core).phase
            if not seen or seen[-1] != phase:
                seen.append(phase)
            clock.advance(1.0)
        assert set(seen) == AUTO_PHASES
        # The cycle runs fast, survey_short, survey_long, idle, three times before the fault.
        start = seen.index("fast")
        assert seen[start : start + 4] == ["fast", "survey_short", "survey_long", "idle"]
        # The two steps of the fault share a phase, so each pass of the playlist shows it once.
        assert seen.count("camera_fault") == 2
        assert seen.count("solve_wait") == 2  # the playlist starts again with a solve

    def test_the_times_are_relative_to_now_so_that_a_standing_clock_works(self) -> None:
        clock = StandingClock(DEMO_NOW_NS)
        core = DemoCore(clock)
        clock.advance(20.0)  # the fast period began at 16 s
        first = activity(core)
        clock.advance(10.0)
        second = activity(core)
        assert first.phase == second.phase == "fast"
        now = DEMO_NOW_NS
        assert now - first.since_utc_ns == pytest.approx(4.0 * NS_PER_S)
        assert now - second.since_utc_ns == pytest.approx(14.0 * NS_PER_S)
        assert first.ends_utc_ns is not None
        assert second.ends_utc_ns is not None
        assert first.ends_utc_ns - now == pytest.approx(26.0 * NS_PER_S)
        assert second.ends_utc_ns - now == pytest.approx(16.0 * NS_PER_S)
        assert first.next_utc_ns == first.ends_utc_ns

    def test_the_windows_of_the_fast_period_close_one_by_one(
        self, core: DemoCore, clock: VirtualClock
    ) -> None:
        clock.advance(16.0)
        counts = []
        for _ in range(30):
            view = activity(core)
            assert view.detail is not None
            counts.append(int(view.detail.split(": ")[1].split(" ")[0]))
            clock.advance(1.0)
        assert counts[0] == 0
        assert counts[-1] == 5
        assert counts == sorted(counts)
        assert view.detail == "Windows of 5 s: 5 of 6 closed"

    def test_the_cadence_is_the_cycle_of_the_demo_and_absent_in_a_fault(
        self, core: DemoCore, clock: VirtualClock
    ) -> None:
        assert activity(core).cadence_s == 45.0
        fault = play(core, clock, 16.0 + 135.0 + 3.0)
        assert fault.phase == "camera_fault"
        assert fault.cadence_s is None

    def test_a_camera_fault_names_the_next_try_and_matches_the_fault_of_the_status(
        self, core: DemoCore, clock: VirtualClock
    ) -> None:
        clock.advance(16.0 + 135.0 + 3.0)
        scheduler = core.status().scheduler
        view = scheduler.activity
        assert view is not None
        assert view.phase == "camera_fault"
        assert view.next_label == "Recovery step: restart the capture"
        assert view.next_utc_ns == scheduler.fault.next_attempt_utc_ns
        assert scheduler.fault.failures == 2
        assert scheduler.fault.next_step == "restart_capture"
        assert (scheduler.t_utc_ns - view.since_utc_ns) / NS_PER_S == pytest.approx(3.0)
        # The fault says why, in the words of the real scheduler, for the activity and the status.
        assert view.reason == "no frame arrived; the camera may be disconnected"
        assert scheduler.fault.cause == "timeout"
        assert scheduler.fault.reason == view.reason
        assert scheduler.fault.since_utc_ns == view.since_utc_ns
        later = play(core, clock, 15.0)
        assert later.next_label == "Recovery step: reopen the camera"
        assert core.status().scheduler.fault.failures == 3
        assert later.since_utc_ns == pytest.approx(view.since_utc_ns, abs=10)  # one episode

    def test_the_fault_of_the_status_is_empty_outside_the_fault(
        self, core: DemoCore, clock: VirtualClock
    ) -> None:
        assert core.status().scheduler.fault.failures == 0
        clock.advance(40.0)
        assert core.status().scheduler.fault.next_step is None


class TestTheOtherStates:
    def test_the_demo_gate_holds_safe_and_then_opens(
        self, core: DemoCore, clock: VirtualClock
    ) -> None:
        core.submit(Pause())
        core.submit(Resume())
        view = activity(core)
        assert (view.state, view.phase) == ("safe", "watch")
        assert view.label == "Daylight gate: the Sun is above -4 degrees"
        assert view.reason == "the scheduler checks the sky first"
        assert view.next_label == "Brightness frame: the cycle starts when the sky is dark enough"
        assert view.next_utc_ns is not None
        assert view.next_utc_ns - DEMO_NOW_NS == pytest.approx(10.0 * NS_PER_S)
        clock.advance(demo_activity.GATE_OPEN_S - 1.0)
        assert activity(core).state == "safe"
        clock.advance(2.0)
        opened = activity(core)
        assert opened.state == "auto"
        assert opened.reason == "the sky is dark enough"
        assert opened.phase == "survey_short"  # the cycle starts again

    def test_the_brightness_frame_comes_within_a_few_seconds(
        self, core: DemoCore, clock: VirtualClock
    ) -> None:
        core.submit(StartAlignment())
        core.submit(StopAlignment())
        for _ in range(3):
            clock.advance(4.0)
            scheduler = core.status().scheduler
            view = scheduler.activity
            assert view is not None
            assert view.next_utc_ns is not None
            assert 0 < view.next_utc_ns - scheduler.t_utc_ns <= 10 * NS_PER_S

    def test_alignment_shows_the_live_view_and_its_idle_timeout(
        self, core: DemoCore, clock: VirtualClock
    ) -> None:
        core.submit(StartAlignment())
        clock.advance(30.0)
        scheduler = core.status().scheduler
        view = scheduler.activity
        assert view is not None
        assert (view.state, view.phase) == ("align", "align")
        assert view.label == "Aligning (the live view)"
        assert view.reason == "you started the alignment"
        assert view.ends_utc_ns is not None
        # The demo's live view is always in use, so the idle timer starts again at every look.
        assert view.ends_utc_ns - scheduler.t_utc_ns == pytest.approx(1800 * NS_PER_S)
        assert (scheduler.t_utc_ns - view.since_utc_ns) / NS_PER_S == pytest.approx(30.0)
        assert view.next_utc_ns == view.ends_utc_ns
        assert view.cadence_s is None

    def test_a_pause_shows_why_and_what_resumes(self, core: DemoCore) -> None:
        core.submit(Pause())
        view = activity(core)
        assert (view.state, view.phase) == ("paused", "paused")
        assert view.reason == "you paused the scheduler"
        assert view.next_label == "After you resume: a brightness check, then the normal cycle"
        assert view.ends_utc_ns is None

    def test_a_dark_session_names_its_phase_as_the_simulator_reports_it(
        self, core: DemoCore, clock: VirtualClock
    ) -> None:
        core.submit(QueueDark())
        script = DEMO_DARK_SCRIPT
        labels = []
        for offset in (
            script.queued_s + 1.0,
            script.queued_s + script.bias_s + 1.0,
            script.queued_s + script.bias_s + script.cover_s * 0.8,
            script.queued_s + script.bias_s + script.cover_s + 1.0,
            script.queued_s + script.bias_s + script.cover_s + script.dark_s + 0.5,
        ):
            clock.advance(offset - clock.monotonic_ns() / NS_PER_S)
            view = activity(core)
            assert (view.state, view.phase) == ("commission", "commission")
            labels.append(view.label)
        assert labels == [
            "Dark session: taking bias frames",
            "Dark session: waiting for the cover",
            "Dark session: waiting for the cover",
            "Dark session: taking dark frames",
            "Dark session: building the master dark",
        ]
        view = activity(core)
        assert view.next_label == "Paused: nothing records until you resume"
        assert view.reason == "a dark session runs"
        clock.advance(5.0)
        done = activity(core)
        assert (done.state, done.phase) == ("paused", "paused")
        assert done.reason == "the dark session ended, and the camera may still be covered"

    def test_a_status_always_has_an_activity_that_agrees_with_the_state(
        self, core: DemoCore, clock: VirtualClock
    ) -> None:
        for command in (None, StartAlignment(), StopAlignment(), Pause(), Resume()):
            if command is not None:
                core.submit(command)
            for _ in range(3):
                scheduler = core.status().scheduler
                assert scheduler.activity is not None
                assert scheduler.activity.state == scheduler.state
                clock.advance(7.0)
