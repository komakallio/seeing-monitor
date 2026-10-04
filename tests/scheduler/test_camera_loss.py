"""A camera that is lost: how fast the scheduler says so, how it climbs, and what it never does.

The first night of the real camera taught the lesson. The camera stopped answering with a timeout
and then said "no ASI camera is connected" to every reopen, and the status showed a camera that was
fine for nine minutes. These scenarios replay that loss on the virtual clock with the default
settings of the fault response, and they hold the system to what a person at the telescope expects:

- the camera component of the health reads `degraded` within a minute, and it says why in words,
- the steps of the ladder follow each other within a minute while quicker steps remain,
- a camera that the reopen cannot find counts as lost at once, with its own reason in the events,
  and the state goes to `safe`, and
- the scheduler keeps trying, slowly, and it never asks for a reboot or a power cycle more often
  than the limit allows.
"""

from __future__ import annotations

import itertools
import sys
from collections.abc import Callable

import pytest

from seeingmon.clock import NS_PER_S, VirtualClock, iso_to_utc_ns
from seeingmon.drivers import base as drivers
from seeingmon.drivers.base import CameraDisconnectedError, CameraLinkError, CameraStateError
from seeingmon.records import EventRecord
from seeingmon.scheduler import Command, Pause, StartAlignment
from seeingmon.scheduler import activity as words
from seeingmon.scheduler.config import FaultConfig, LadderConfig, SchedulerConfig
from seeingmon.scheduler.faults import FaultCause, classify
from seeingmon.scheduler.levels import EscalationLevel
from seeingmon.scheduler.status import SchedulerStatus
from tests.scheduler.scenario import TEST_CONFIG, World

NIGHT = iso_to_utc_ns("2026-01-01T22:00:00Z")
LOST_AT = 1000.0
RB, PC = EscalationLevel.REBOOT, EscalationLevel.POWER_CYCLE
QUICK_STEPS = ["restart_capture", "reopen", "usb_reset", "restart_acquire"]


def send(command: Command) -> Callable[[World], None]:
    """A scripted action that hands a command to the scheduler."""

    def action(world: World) -> None:
        world.scheduler.submit(command)

    return action


def detail(event: EventRecord) -> dict[str, object]:
    assert event.detail is not None
    return event.detail


def times(world: World, events: list[EventRecord]) -> list[float]:
    return [world.seconds(event.t_utc_ns) for event in events]


def gone_world(*, back_at: float | None = None, config: SchedulerConfig = TEST_CONFIG) -> World:
    """A night in which the camera disappears at `LOST_AT`, and comes back at `back_at`."""
    world = World(start_utc_ns=NIGHT, config=config)
    world.camera_gone(LOST_AT, back_at)
    return world


def step_until(world: World, done: object, *, limit_s: float = 4000.0) -> list[SchedulerStatus]:
    """Step the scheduler and take the status after each step, until `done(status)` holds."""
    assert callable(done)
    statuses: list[SchedulerStatus] = []
    deadline = world.t(limit_s)
    while world.clock.utc_ns() < deadline:
        world.scheduler.step()
        status = world.scheduler.status()
        statuses.append(status)
        if done(status):
            return statuses
    raise AssertionError("the condition never held")


@pytest.fixture(scope="module")
def lost() -> World:
    """The camera is gone for good. Four hours of the loss, with the default settings."""
    world = gone_world()
    world.run_until(LOST_AT + 4 * 3600)
    return world


class TestTheFirstMinute:
    def test_the_first_failed_read_is_a_timeout_and_the_first_step_fails(self, lost: World) -> None:
        first_fault, second_fault = lost.events("scheduler.fault")[:2]
        assert detail(first_fault)["cause"] == "timeout"
        assert detail(first_fault)["where"] == "reading a fast frame"
        assert "no frame arrived" in str(detail(first_fault)["reason"])
        assert detail(second_fault)["cause"] == "error"
        assert "GENERAL_ERROR" in second_fault.message
        step = lost.events("scheduler.recovery_step")[0]
        assert detail(step)["step"] == "restart_capture"
        assert detail(step)["ok"] is False

    def test_the_camera_reads_degraded_within_a_minute_and_says_why(self) -> None:
        world = gone_world()
        # Step to the first failed recovery step, and then to a minute after the first failed read.
        seen = step_until(world, lambda s: s.fault.failures >= 2 and s.fault.last_error is not None)
        status = seen[-1]
        failed_read = world.seconds(world.events("scheduler.fault")[0].t_utc_ns)
        assert world.seconds(status.t_utc_ns) - failed_read < 60.0
        assert status.camera_component == "degraded"
        assert status.health_fields()["components"]["camera"] == "degraded"
        assert status.fault.reason == "no frame arrived; the camera may be disconnected"
        assert status.fault.cause == "timeout"
        assert status.camera_reason == "no frame arrived; the camera may be disconnected"
        world.close()

    def test_the_cause_of_the_timeout_survives_the_failing_steps_that_follow(
        self, lost: World
    ) -> None:
        """A failing restart of the capture does not replace the first and better explanation."""
        faults = lost.events("scheduler.fault")
        causes = [detail(event)["cause"] for event in faults[:4]]
        assert causes == ["timeout", "error", "error", "disconnected"]
        reasons = [detail(event)["reason"] for event in faults[:4]]
        assert (
            reasons[0]
            == reasons[1]
            == reasons[2]
            == ("no frame arrived; the camera may be disconnected")
        )

    def test_the_first_steps_follow_within_seconds(self, lost: World) -> None:
        steps = lost.events("scheduler.recovery_step")
        names = [detail(event)["step"] for event in steps[:8]]
        assert names == [
            "restart_capture",
            "restart_capture",
            "reopen",
            "reopen",
            "usb_reset",
            "usb_reset",
            "restart_acquire",
            "restart_acquire",
        ]
        when = times(lost, steps[:8])
        assert when[0] - LOST_AT < 10.0  # the first step comes seconds after the failed read
        gaps = [later - earlier for earlier, later in itertools.pairwise(when)]
        assert gaps[0] == pytest.approx(4.0, abs=0.5)  # the backoff doubles: 2 s, 4 s, 8 s ...
        assert gaps[1] == pytest.approx(8.0, abs=0.5)
        assert gaps[2] == pytest.approx(16.0, abs=0.5)
        assert all(gap <= 61.0 for gap in gaps)  # and it stops at a minute


class TestALostCamera:
    def test_the_reopen_that_finds_no_camera_marks_it_lost_at_once(self, lost: World) -> None:
        reopen = next(
            event
            for event in lost.events("scheduler.fault")
            if detail(event)["cause"] == "disconnected"
        )
        assert detail(reopen)["where"] == "the recovery step reopen"
        assert reopen.message.endswith("no ASI camera is connected")
        assert detail(reopen)["failures"] == 4  # it did not wait for the fifth failure
        (degraded,) = lost.events("scheduler.degraded")
        assert degraded.t_utc_ns == reopen.t_utc_ns
        assert degraded.level == "error"
        assert degraded.message.startswith(
            "The camera is not connected, so the status is degraded."
        )
        assert detail(degraded)["cause"] == "disconnected"
        assert detail(degraded)["failures"] == 4
        assert world_seconds(lost, degraded) - LOST_AT < 30.0  # the loss is known within seconds

    def test_the_state_goes_to_safe_in_the_same_step_and_says_that_the_camera_failed(
        self, lost: World
    ) -> None:
        (degraded,) = lost.events("scheduler.degraded")
        change = next(
            event
            for event in lost.events("scheduler.state_change")
            if detail(event)["reason"] == "camera fault"
        )
        assert detail(change)["to"] == "safe"
        assert abs(change.t_utc_ns - degraded.t_utc_ns) <= NS_PER_S

    def test_the_camera_fault_shows_the_reason_and_the_time_of_the_next_try(self) -> None:
        world = gone_world()
        seen = step_until(world, lambda s: s.degraded)
        status = seen[-1]
        activity = status.activity
        assert activity is not None
        assert (status.state, activity.state, activity.phase) == ("safe", "safe", "camera_fault")
        assert activity.label == "Camera fault: the camera is not connected"
        assert activity.reason == "the camera is not connected"
        assert activity.next_utc_ns == status.fault.next_attempt_utc_ns
        assert activity.next_utc_ns is not None
        assert activity.next_utc_ns > status.t_utc_ns
        assert activity.next_label == words.recovery_label(status.fault.next_step or "")
        assert activity.since_utc_ns <= status.t_utc_ns
        assert activity.ends_utc_ns is None  # nobody knows when the camera comes back
        assert status.camera_component == "failed"
        assert status.fault.cause == "disconnected"
        assert status.camera_reason == "the camera is not connected"
        world.close()

    def test_the_scheduler_keeps_the_quick_steps_after_the_loss_and_then_slows_down(
        self, lost: World
    ) -> None:
        steps = lost.events("scheduler.recovery_step")
        when = times(lost, steps)
        names = [str(detail(event)["step"]) for event in steps]
        # The quick steps (up to the restart of `acquire`) keep the quick backoff, a minute at most.
        quick = [i for i, name in enumerate(names) if name in QUICK_STEPS][:8]
        quick_gaps = [when[b] - when[a] for a, b in itertools.pairwise(quick)]
        assert max(quick_gaps) <= 61.0
        assert when[quick[-1]] - LOST_AT < 5 * 60  # the whole climb takes about four minutes
        # After that every attempt waits the slow period, which is 10 minutes by default.
        slow = [later - earlier for earlier, later in itertools.pairwise(when[quick[-1] :])]
        assert len(slow) > 20
        assert all(gap == pytest.approx(600.0, abs=1.0) for gap in slow)

    def test_the_scheduler_still_tries_hours_later(self, lost: World) -> None:
        last = times(lost, lost.events("scheduler.recovery_step"))[-1]
        assert last > LOST_AT + 3.5 * 3600
        assert lost.scheduler.status().state == "safe"
        assert lost.scheduler.status().degraded is True

    def test_the_destructive_steps_keep_their_limits(self, lost: World) -> None:
        destructive = [(t, level) for t, level in lost.escalations if level in (RB, PC)]
        quick_end = times(lost, lost.events("scheduler.recovery_step"))[7]
        assert destructive, "the ladder reaches the top"
        # Never before every quicker step had its attempts, and at most one per limit.
        assert lost.seconds(destructive[0][0]) > quick_end
        gaps = [
            (later - earlier) / NS_PER_S
            for (earlier, _), (later, _) in itertools.pairwise(destructive)
        ]
        assert all(gap >= TEST_CONFIG.ladder.destructive_interval_s for gap in gaps)
        assert len(destructive) <= 4 * 3600 // int(TEST_CONFIG.ladder.destructive_interval_s) + 1
        # Between the destructive steps the scheduler restarts `acquire`, which costs less.
        restarts = [
            level for _, level in lost.escalations if level is EscalationLevel.RESTART_ACQUIRE
        ]
        assert len(restarts) > len(destructive)

    def test_a_dead_camera_writes_two_events_for_each_attempt_and_no_more(
        self, lost: World
    ) -> None:
        counters = lost.scheduler.status().counters
        assert counters.faults == len(lost.events("scheduler.fault"))
        assert counters.recovery_steps == len(lost.events("scheduler.recovery_step"))
        assert len(lost.events("scheduler.degraded")) == 1  # one event for the whole episode


class TestACameraThatComesBack:
    @pytest.mark.parametrize("back_at", [1100.0, 1300.0, 2500.0])
    def test_the_scheduler_recovers_by_itself_and_the_cycle_resumes(self, back_at: float) -> None:
        world = gone_world(back_at=back_at)
        world.run_until(back_at + 3 * 3600)
        (recovered,) = world.events("scheduler.recovered")
        assert world.scheduler.state.value == "auto"
        status = world.scheduler.status()
        assert status.degraded is False
        assert status.camera_component == "ok"
        assert status.fault.cause is None
        assert status.camera_reason is None
        assert world.seconds(recovered.t_utc_ns) > back_at
        assert world.windows()[-1].t_utc_ns > world.t(back_at + 3600)
        world.close()

    def test_a_camera_that_returns_during_the_quick_climb_is_found_within_a_minute(self) -> None:
        world = gone_world(back_at=1100.0)
        world.run_until(1100 + 61)
        fixing = [
            event
            for event in world.events("scheduler.recovery_step")
            if detail(event)["ok"] is True
        ]
        assert fixing
        assert world.seconds(fixing[0].t_utc_ns) - 1100 <= 61.0
        world.close()

    def test_the_reason_goes_away_with_the_failures(self) -> None:
        world = gone_world(back_at=1100.0)
        world.run_until(2500)
        status = world.scheduler.status()
        assert status.fault.failures == 0
        assert status.fault.reason is None
        assert status.fault.last_error is not None  # the last error stays for the operator
        world.close()


class TestWhatTheFaultDoesToTheOtherStates:
    def test_alignment_ends_with_the_camera(self) -> None:
        world = World(start_utc_ns=NIGHT)
        world.at(
            900, lambda w: setattr(w, "_aligned", w.scheduler.submit(StartAlignment()).accepted)
        )
        world.camera_gone(1000, None)
        world.run_until(1300)
        assert world.scheduler.state.value == "safe"
        assert world.events("scheduler.degraded")
        assert world.scheduler.status().activity.phase == "camera_fault"  # type: ignore[union-attr]
        world.close()

    def test_a_paused_scheduler_stays_paused_and_makes_no_attempt(self) -> None:
        world = World(start_utc_ns=NIGHT)
        world.at(900, send(Pause()))
        world.camera_gone(950, None)
        world.run_until(2000)
        activity = world.scheduler.status().activity
        assert activity is not None
        assert activity.phase == "paused"
        assert world.events("scheduler.fault") == []
        world.close()


class TestACameraThatNeverOpened:
    def test_a_step_of_the_ladder_opens_the_camera_when_it_appears(self) -> None:
        """The driver steps need an open camera, as the ASI driver does. Opening is the step."""
        world = World(start_utc_ns=NIGHT, config=TEST_CONFIG)
        world.camera.fail_open(CameraDisconnectedError("no ASI camera is connected"))
        world.camera.fail_recover(CameraStateError("the camera is not open"))
        world.at(20, lambda w: setattr(w.camera, "_open_failure", None))
        world.run_until(300)
        opening = [
            event
            for event in world.events("scheduler.recovery_step")
            if detail(event).get("opened") is True
        ]
        assert opening, "a step of the ladder opened the camera"
        assert 20.0 < world.seconds(opening[0].t_utc_ns) <= 31.0  # the next step after it appeared
        assert world.escalations == []  # no restart of `acquire` was needed
        assert world.scheduler.state.value == "auto"
        assert world.scheduler.status().degraded is False
        world.close()


class TestAnUnreachableAcquire:
    def test_a_broken_link_is_not_a_lost_camera(self) -> None:
        """A restart of `acquire` breaks the link for seconds, and the camera is fine."""
        world = World(start_utc_ns=NIGHT, config=TEST_CONFIG)
        world.camera.fail_reads(CameraLinkError("the connection to acquire was lost"))
        world.run_until(1500)
        (fault,) = world.events("scheduler.fault")
        assert detail(fault)["cause"] == "link"
        assert world.events("scheduler.degraded") == []
        assert world.scheduler.state.value == "auto"
        world.close()


class TestTheCauses:
    @pytest.mark.parametrize(
        ("name", "cause"),
        [
            ("CameraTimeoutError", FaultCause.TIMEOUT),
            ("CameraDisconnectedError", FaultCause.DISCONNECTED),
            ("CameraLinkError", FaultCause.LINK),
            ("CameraStateError", FaultCause.ERROR),
            ("CameraConfigError", FaultCause.ERROR),
            ("CameraError", FaultCause.ERROR),
        ],
    )
    def test_each_camera_error_has_a_cause(self, name: str, cause: FaultCause) -> None:
        assert classify(getattr(drivers, name)("x")) is cause

    def test_an_error_that_is_not_a_camera_error_is_a_plain_error(self) -> None:
        assert classify(RuntimeError("the supervisor failed")) is FaultCause.ERROR


# --- A machine that sleeps -------------------------------------------------------------------


class SleepingClock(VirtualClock):
    """A clock for a machine that suspends once: one `sleep` returns minutes late.

    The suspension hits the first sleep after `suspend_at` that comes from the function `where`
    names: the loop's own sleep (`_sleep_until`), or the wait of the fake camera for a frame.
    """

    def __init__(
        self, start_utc_ns: int, *, suspend_at: float, suspend_s: float, where: str
    ) -> None:
        super().__init__(start_utc_ns)
        self._suspend_at_ns: int | None = start_utc_ns + round(suspend_at * NS_PER_S)
        self._suspend_s = suspend_s
        self._where = where

    def sleep(self, seconds: float) -> None:
        super().sleep(seconds)
        caller = sys._getframe(1).f_code.co_name
        if (
            self._suspend_at_ns is not None
            and self.utc_ns() >= self._suspend_at_ns
            and caller == self._where
        ):
            self._suspend_at_ns = None
            self.advance(self._suspend_s)


class TestAStalledLoop:
    def test_a_sleep_that_returns_minutes_late_is_an_event_that_says_why(self) -> None:
        clock = SleepingClock(NIGHT, suspend_at=1000, suspend_s=559.0, where="_sleep_until")
        world = World(start_utc_ns=NIGHT, clock=clock)
        world.run_until(1800)
        (stalled,) = world.events("scheduler.stalled")
        assert stalled.level == "warning"
        assert stalled.message == (
            "The scheduler did not run for 9 min 19 s, so the machine may have been suspended."
        )
        assert detail(stalled)["stalled_s"] == pytest.approx(559.0, abs=0.01)
        assert world.scheduler.status().counters.stalls == 1
        assert world.scheduler.state.value == "auto"  # it goes on
        world.close()

    def test_a_read_that_returns_minutes_after_its_timeout_is_a_stall_too(self) -> None:
        """The machine can sleep while a frame is on its way, as it does during a long exposure."""
        clock = SleepingClock(NIGHT, suspend_at=1000, suspend_s=559.0, where="read_frame")
        world = World(start_utc_ns=NIGHT, clock=clock)
        world.run_until(1800)
        (stalled,) = world.events("scheduler.stalled")
        # The read waits for its timeout of 4.5 s at most, so what exceeds it is the suspension.
        assert detail(stalled)["stalled_s"] == pytest.approx(559.0 - 4.5 + 2.0, abs=3.0)
        assert world.scheduler.state.value == "auto"
        world.close()

    def test_a_normal_evening_has_no_stall(self) -> None:
        world = World(start_utc_ns=NIGHT)
        world.run_until(3 * 3600)
        assert world.events("scheduler.stalled") == []
        world.close()

    def test_a_long_camera_call_is_no_stall(self) -> None:
        """A read that waits for its timeout is a call into the camera, and it is not a sleep."""
        config = SchedulerConfig(
            fast=TEST_CONFIG.fast,
            loop=TEST_CONFIG.loop,
            faults=FaultConfig(backoff_initial_s=1.0, backoff_max_s=2.0),
            ladder=LadderConfig(attempts_per_level=1),
        )
        world = World(start_utc_ns=NIGHT, config=config)
        world.camera_fault(1000, 1600)
        world.run_until(2500)
        assert world.events("scheduler.fault")
        assert world.events("scheduler.stalled") == []
        world.close()


def world_seconds(world: World, event: EventRecord) -> float:
    return world.seconds(event.t_utc_ns)
