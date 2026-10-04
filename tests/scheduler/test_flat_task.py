"""The flat task in the scheduler: its limits, the single slot, the start at once, the pause after
it, and the cancel.

The session itself belongs to `seeingmon.services.core.commissioning.flat`. These tests register a
stand-in handler that reads a few frames, so they show what the scheduler does around a flat task:
it checks the command, holds one flat task at a time, starts the task at the next step when it
asks to (`immediate`), ends the episode in `paused` when it asks for that, and takes a task out of
the queue (or stops the running one) when `CancelTask` arrives.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from seeingmon.clock import iso_to_utc_ns
from seeingmon.scheduler import (
    CancelTask,
    CommandResult,
    CommissionContext,
    CommissionResult,
    CommissionTask,
    Pause,
    QueueBurst,
    QueueDark,
    QueueFlat,
    RejectReason,
    Resume,
    Scheduler,
    StartAlignment,
)
from tests.scheduler.scenario import World

NIGHT = iso_to_utc_ns("2026-01-01T22:00:00Z")
FRAME_PERIOD_S = 2.0  # the fast stream of the scenario gives one frame every 2 s


class StandInFlat:
    """A flat handler that reads some frames. `during` runs after the first frame."""

    def __init__(self, scheduler: Scheduler, *, frames: int = 3, during: Any = None) -> None:
        self.scheduler = scheduler
        self.frames = frames
        self.during = during
        self.tasks: list[CommissionTask] = []

    def run(self, task: CommissionTask, context: CommissionContext) -> CommissionResult:
        self.tasks.append(task)
        started = context.clock.utc_ns()
        config = context.fast_stream_config()
        assert config is not None
        context.configure(config)
        context.start()
        stopped = False
        for index in range(self.frames):
            if context.should_stop():
                stopped = True
                break
            context.read_frame()
            if index == 0 and self.during is not None:
                self.during(self.scheduler)
        context.stop()
        return CommissionResult(
            task_id=task.task_id,
            kind=task.kind,
            status="aborted" if stopped else "ok",
            summary="a stand-in session",
            started_utc_ns=started,
            finished_utc_ns=context.clock.utc_ns(),
        )


class Plain:
    """A burst or dark handler that reads a few frames."""

    def run(self, task: CommissionTask, context: CommissionContext) -> CommissionResult:
        started = context.clock.utc_ns()
        config = context.fast_stream_config()
        assert config is not None
        context.configure(config)
        context.start()
        for _ in range(2):
            context.read_frame()
        context.stop()
        return CommissionResult(
            task_id=task.task_id,
            kind=task.kind,
            status="ok",
            summary="a stand-in task",
            started_utc_ns=started,
            finished_utc_ns=context.clock.utc_ns(),
        )


def night_world(*, handler: bool = True, **options: Any) -> tuple[World, StandInFlat | None]:
    world = World(start_utc_ns=NIGHT)
    stand_in = StandInFlat(world.scheduler, **options) if handler else None
    if stand_in is not None:
        world.scheduler.register_handler("flat", stand_in)
    return world, stand_in


def submit_at(world: World, seconds: float, command: Any) -> list[CommandResult]:
    holder: list[CommandResult] = []
    world.at(seconds, lambda w: holder.append(w.scheduler.submit(command)))
    return holder


def started_at(world: World, kind: str = "flat") -> float:
    events = [e for e in world.events("scheduler.task_started") if (e.detail or {})["kind"] == kind]
    return world.seconds(events[0].t_utc_ns)


def commission_reason(world: World) -> str:
    for event in world.events("scheduler.state_change"):
        detail = event.detail or {}
        if detail["to"] == "commission":
            return str(detail["reason"])
    raise AssertionError("the scheduler never entered commission")


def last_reason(world: World) -> str:
    detail = world.events("scheduler.state_change")[-1].detail
    assert detail is not None
    return str(detail["reason"])


class TestTheCommand:
    def test_a_flat_task_needs_a_handler(self) -> None:
        world, _ = night_world(handler=False)
        result = world.scheduler.submit(QueueFlat())
        assert (result.accepted, result.reason) == (False, RejectReason.NO_HANDLER)
        assert "flat" in result.message

    def test_a_flat_task_is_queued_with_a_task_id(self) -> None:
        world, _ = night_world()
        result = world.scheduler.submit(QueueFlat(frames=16, target_fraction=0.4))
        assert result.accepted
        assert result.task_id == 1
        assert "flat is queued" in result.message
        assert world.scheduler.status().queued_tasks == 1

    @pytest.mark.parametrize(
        ("command", "word"),
        [
            (QueueFlat(frames=7), "frames"),
            (QueueFlat(frames=65), "frames"),
            (QueueFlat(frames=0), "frames"),
            (QueueFlat(target_fraction=0.29), "target_fraction"),
            (QueueFlat(target_fraction=0.71), "target_fraction"),
            (QueueFlat(target_fraction=math.nan), "target_fraction"),
            (QueueFlat(target_fraction=math.inf), "target_fraction"),
            (QueueFlat(set_number=0), "set_number"),
            (QueueFlat(set_number=3), "set_number"),
        ],
    )
    def test_a_setting_out_of_range_is_invalid_and_names_the_field(
        self, command: QueueFlat, word: str
    ) -> None:
        world, _ = night_world()
        result = world.scheduler.submit(command)
        assert (result.accepted, result.reason) == (False, RejectReason.INVALID)
        assert word in result.message
        assert world.scheduler.status().queued_tasks == 0

    @pytest.mark.parametrize(
        "command",
        [
            QueueFlat(frames=8),
            QueueFlat(frames=64),
            QueueFlat(target_fraction=0.3),
            QueueFlat(target_fraction=0.7),
            QueueFlat(set_number=2),
            QueueFlat(pause_after=False, immediate=False, priority=5),
            QueueFlat(),
        ],
    )
    def test_the_edges_of_every_range_are_valid(self, command: QueueFlat) -> None:
        world, _ = night_world()
        assert world.scheduler.submit(command).accepted

    def test_the_message_names_the_limits(self) -> None:
        world, _ = night_world()
        assert world.scheduler.submit(QueueFlat(frames=7)).message == (
            "frames must be between 8 and 64"
        )
        assert world.scheduler.submit(QueueFlat(target_fraction=0.9)).message == (
            "target_fraction must be between 0.3 and 0.7"
        )


class TestWhenTheTaskStarts:
    def test_the_answer_says_the_next_step_for_a_session_that_starts_at_once(self) -> None:
        world, _ = night_world()
        queued = submit_at(world, 50, QueueFlat())
        world.run_until(60)
        assert queued[0].state == "auto"
        assert "starts at the next step" in queued[0].message

    def test_the_answer_says_the_next_cycle_boundary_for_a_session_that_waits(self) -> None:
        world, _ = night_world()
        queued = submit_at(world, 50, QueueFlat(immediate=False))
        world.run_until(60)
        assert "runs at the next cycle boundary" in queued[0].message

    def test_a_session_queued_while_paused_waits_for_the_resume(self) -> None:
        world, stand_in = night_world()
        submit_at(world, 10, Pause())
        queued = submit_at(world, 20, QueueFlat(pause_after=False))
        submit_at(world, 300, Resume())
        world.run_until(100)
        assert queued[0].accepted
        assert "the scheduler is paused" in queued[0].message
        assert stand_in is not None
        assert stand_in.tasks == []
        world.run_until(900)
        assert len(stand_in.tasks) == 1

    def test_a_session_queued_during_the_alignment_runs_after_it(self) -> None:
        world, stand_in = night_world()
        submit_at(world, 100, StartAlignment())
        queued = submit_at(world, 150, QueueFlat(pause_after=False))
        world.run_until(200)
        assert "runs after the alignment helper ends" in queued[0].message
        assert stand_in is not None
        assert stand_in.tasks == []


class TestStartingAtOnce:
    """A flat session does not wait for the cycle boundary, because someone holds the light source.

    The first fast period of the scenario runs from 0 to 120 s, and the boundary follows it.
    """

    def test_a_session_queued_in_a_fast_period_starts_within_one_frame_period(self) -> None:
        world, stand_in = night_world()
        submit_at(world, 20, QueueFlat(pause_after=False))
        world.run_until(400)
        queued = world.seconds(world.events("scheduler.command")[0].t_utc_ns)
        assert 19.0 <= queued <= 23.0
        assert 0.0 <= started_at(world) - queued <= FRAME_PERIOD_S
        assert commission_reason(world) == "a flat session starts at once"
        assert stand_in is not None
        assert len(stand_in.tasks) == 1

    def test_a_session_without_the_flag_waits_for_the_boundary(self) -> None:
        world, _ = night_world()
        submit_at(world, 20, QueueFlat(immediate=False, pause_after=False))
        world.run_until(400)
        assert started_at(world) >= 120.0
        assert commission_reason(world) == "a task is queued"

    def test_a_dark_session_still_gives_its_own_reason(self) -> None:
        world, _ = night_world()
        world.scheduler.register_handler("dark", Plain())
        submit_at(world, 20, QueueDark(pause_after=False))
        world.run_until(400)
        assert commission_reason(world) == "a dark session starts at once"

    def test_the_other_kinds_of_task_still_wait_for_the_boundary(self) -> None:
        world, _ = night_world()
        world.scheduler.register_handler("burst", Plain())
        submit_at(world, 20, QueueBurst())
        world.run_until(400)
        assert started_at(world, "burst") >= 120.0


class TestOneAtATime:
    def test_a_second_flat_task_is_busy_while_the_first_waits(self) -> None:
        world, _ = night_world()
        first = world.scheduler.submit(QueueFlat())
        second = world.scheduler.submit(QueueFlat(frames=16))
        assert first.accepted
        assert (second.accepted, second.reason) == (False, RejectReason.BUSY)
        assert "a flat session is already queued or running" in second.message
        assert world.scheduler.status().queued_tasks == 1

    def test_a_second_flat_task_is_busy_while_the_first_runs(self) -> None:
        answers: list[CommandResult] = []

        def ask_again(scheduler: Scheduler) -> None:
            answers.append(scheduler.submit(QueueFlat()))

        world, stand_in = night_world(during=ask_again)
        world.scheduler.submit(QueueFlat())
        world.run_until(300)
        assert [(a.accepted, a.reason) for a in answers] == [(False, RejectReason.BUSY)]
        assert stand_in is not None
        assert len(stand_in.tasks) == 1

    def test_another_flat_task_is_fine_after_the_first_ended(self) -> None:
        world, stand_in = night_world()
        world.scheduler.submit(QueueFlat(pause_after=False))
        world.run_until(300)
        assert world.scheduler.submit(QueueFlat(pause_after=False)).accepted
        world.run_until(600)
        assert stand_in is not None
        assert len(stand_in.tasks) == 2

    def test_a_flat_and_a_dark_session_do_not_block_each_other(self) -> None:
        world, _ = night_world()
        world.scheduler.register_handler("dark", Plain())
        assert world.scheduler.submit(QueueFlat()).accepted
        assert world.scheduler.submit(QueueDark()).accepted
        assert world.scheduler.status().queued_tasks == 2

    def test_a_dark_session_is_busy_only_for_a_dark_session(self) -> None:
        world, _ = night_world()
        world.scheduler.register_handler("dark", Plain())
        assert world.scheduler.submit(QueueDark()).accepted
        busy = world.scheduler.submit(QueueDark())
        assert (busy.accepted, busy.reason) == (False, RejectReason.BUSY)
        assert "a dark session is already queued or running" in busy.message


class TestPauseAfter:
    def test_the_episode_ends_in_paused_and_the_event_says_why(self) -> None:
        world, stand_in = night_world()
        submit_at(world, 50, QueueFlat(pause_after=True))
        world.run_until(600)
        assert stand_in is not None
        assert len(stand_in.tasks) == 1
        assert world.states_visited() == ["safe", "auto", "commission", "paused"]
        assert "flat session" in last_reason(world)
        assert "light source" in last_reason(world)

    def test_resume_brings_the_system_back_to_auto(self) -> None:
        world, _ = night_world()
        submit_at(world, 50, QueueFlat(pause_after=True))
        submit_at(world, 500, Resume())
        world.run_until(1500)
        assert world.states_visited() == ["safe", "auto", "commission", "paused", "safe", "auto"]

    def test_without_the_flag_the_scheduler_returns_where_it_was(self) -> None:
        world, _ = night_world()
        submit_at(world, 50, QueueFlat(pause_after=False))
        world.run_until(600)
        assert world.states_visited() == ["safe", "auto", "commission", "auto"]

    def test_the_result_of_the_task_is_an_event_of_its_kind(self) -> None:
        world, _ = night_world()
        submit_at(world, 50, QueueFlat())
        world.run_until(600)
        (result,) = world.events("scheduler.flat_result")
        assert result.detail is not None
        assert result.detail["status"] == "ok"
        assert [r.kind for r in world.results] == ["flat"]

    def test_a_pause_cuts_the_session_short(self) -> None:
        world, _ = night_world(frames=4, during=lambda s: s.submit(Pause()))
        submit_at(world, 20, QueueFlat(pause_after=False))
        world.run_until(400)
        assert [r.status for r in world.results if r.kind == "flat"] == ["aborted"]
        assert world.states_visited()[-2:] == ["commission", "paused"]


class TestCancel:
    def test_a_waiting_task_is_removed_and_never_starts(self) -> None:
        world, stand_in = night_world()
        submit_at(world, 10, Pause())
        submit_at(world, 20, QueueFlat())
        answers = submit_at(world, 30, CancelTask(kind="flat"))
        submit_at(world, 40, Resume())
        world.run_until(900)
        (answer,) = answers
        assert answer.accepted
        assert answer.task_id == 1
        assert "removed" in answer.message
        assert stand_in is not None
        assert stand_in.tasks == []
        # the scheduler keeps the result, and the sink hears of tasks that ran only
        (result,) = [r for r in world.scheduler.results() if r.kind == "flat"]
        assert result.status == "aborted"
        assert result.summary == "The flat was cancelled before it started."
        assert result.pinned is False
        assert world.results == []
        (event,) = world.events("scheduler.flat_result")
        assert event.detail is not None
        assert event.detail["status"] == "aborted"
        assert world.scheduler.status().queued_tasks == 0

    def test_a_new_session_may_follow_a_cancel(self) -> None:
        world, _ = night_world()
        world.scheduler.submit(Pause())
        assert world.scheduler.submit(QueueFlat()).accepted
        assert world.scheduler.submit(CancelTask(kind="flat")).accepted
        assert world.scheduler.submit(QueueFlat()).accepted

    def test_nothing_to_cancel_is_a_rejection_with_its_own_reason(self) -> None:
        world, _ = night_world()
        answer = world.scheduler.submit(CancelTask(kind="flat"))
        assert (answer.accepted, answer.reason) == (False, RejectReason.NO_TASK)
        assert answer.message == "no flat task waits or runs"

    def test_a_kind_that_does_not_exist_is_invalid(self) -> None:
        world, _ = night_world()
        for kind in ("", "nonsense", "FLAT"):
            answer = world.scheduler.submit(CancelTask(kind=kind))
            assert (answer.accepted, answer.reason) == (False, RejectReason.INVALID)

    def test_the_cancel_touches_only_its_own_kind(self) -> None:
        world, _ = night_world()
        world.scheduler.register_handler("burst", Plain())
        world.scheduler.submit(Pause())
        world.scheduler.submit(QueueBurst())
        world.scheduler.submit(QueueFlat())
        assert world.scheduler.submit(CancelTask(kind="flat")).accepted
        assert world.scheduler.status().queued_tasks == 1
        refused = world.scheduler.submit(CancelTask(kind="dark"))
        assert refused.reason is RejectReason.NO_TASK

    def test_a_running_task_stops_at_its_next_check_and_the_episode_still_pauses(self) -> None:
        world, stand_in = night_world(frames=6, during=lambda s: s.submit(CancelTask(kind="flat")))
        submit_at(world, 20, QueueFlat(pause_after=True))
        world.run_until(600)
        assert stand_in is not None
        (result,) = [r for r in world.results if r.kind == "flat"]
        assert result.status == "aborted"
        # the handler read the first frame, and the cancel took effect at the next check
        assert world.states_visited() == ["safe", "auto", "commission", "paused"]
        (event,) = world.events("scheduler.command")[-1:]
        assert event.detail is not None

    def test_the_answer_for_a_running_task_names_its_id(self) -> None:
        holder: list[CommandResult] = []
        world, _ = night_world(
            frames=6, during=lambda s: holder.append(s.submit(CancelTask(kind="flat")))
        )
        world.scheduler.submit(QueueFlat())
        world.run_until(300)
        assert holder[0].accepted
        assert holder[0].task_id == 1
        assert "stops at its next check" in holder[0].message

    def test_a_cancel_does_not_leak_into_the_next_task(self) -> None:
        world, stand_in = night_world(frames=3, during=lambda s: s.submit(CancelTask(kind="flat")))
        world.scheduler.submit(QueueFlat(pause_after=False))
        world.run_until(300)
        assert stand_in is not None
        stand_in.during = None  # the second session is left alone
        world.scheduler.submit(QueueFlat(pause_after=False))
        world.run_until(700)
        statuses = [r.status for r in world.results if r.kind == "flat"]
        assert statuses == ["aborted", "ok"]
