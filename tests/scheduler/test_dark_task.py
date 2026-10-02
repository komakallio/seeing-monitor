"""The dark task in the scheduler: its limits, the single slot, and the pause after it.

The session itself belongs to the services lane (`seeingmon.services.core.commissioning.dark`).
These tests register a stand-in handler that reads a few frames, so they show what the scheduler
does around a dark task: it checks the command, holds one dark task at a time, starts the task at
the next step when it asks to (`immediate`), and, when the task asks for it, ends the commission
episode in `paused`, so that nothing records data while the camera is still covered.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from seeingmon.clock import iso_to_utc_ns
from seeingmon.scheduler import (
    CommandResult,
    CommissionContext,
    CommissionResult,
    CommissionTask,
    Pause,
    QueueBurst,
    QueueDark,
    RejectReason,
    Resume,
    Scheduler,
    StartAlignment,
    StopAlignment,
)
from seeingmon.scheduler.config import SchedulerConfig
from tests.scheduler.scenario import PROFILE, World

NIGHT = iso_to_utc_ns("2026-01-01T22:00:00Z")
FRAME_PERIOD_S = 2.0  # the fast stream of the scenario gives one frame every 2 s
LONG_EXPOSURE_US = 30_000_000  # the long exposure of the survey step


class StandInDark:
    """A dark handler that reads some frames. `during` runs in the middle of the task."""

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
    """A burst handler that reads a few frames."""

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
            summary="a stand-in burst",
            started_utc_ns=started,
            finished_utc_ns=context.clock.utc_ns(),
        )


def night_world(*, handler: bool = True, **options: Any) -> tuple[World, StandInDark | None]:
    world = World(start_utc_ns=NIGHT)
    stand_in = StandInDark(world.scheduler, **options) if handler else None
    if stand_in is not None:
        world.scheduler.register_handler("dark", stand_in)
    return world, stand_in


def submit_at(world: World, seconds: float, command: Any) -> list[CommandResult]:
    holder: list[CommandResult] = []
    world.at(seconds, lambda w: holder.append(w.scheduler.submit(command)))
    return holder


def last_reason(world: World) -> str:
    detail = world.events("scheduler.state_change")[-1].detail
    assert detail is not None
    return str(detail["reason"])


class TestTheCommand:
    def test_a_dark_task_needs_a_handler(self) -> None:
        world, _ = night_world(handler=False)
        result = world.scheduler.submit(QueueDark())
        assert (result.accepted, result.reason) == (False, RejectReason.NO_HANDLER)
        assert "dark" in result.message

    def test_a_dark_task_is_queued_with_a_task_id(self) -> None:
        world, _ = night_world()
        result = world.scheduler.submit(QueueDark(frames=5, label="after the clean"))
        assert result.accepted
        assert result.task_id == 1
        assert "dark is queued" in result.message
        assert world.scheduler.status().queued_tasks == 1

    @pytest.mark.parametrize(
        ("command", "word"),
        [
            (QueueDark(frames=2), "frames"),
            (QueueDark(frames=61), "frames"),
            (QueueDark(bias_frames=2), "bias_frames"),
            (QueueDark(bias_frames=61), "bias_frames"),
            (QueueDark(exposure_s=0.0), "exposure_s"),
            (QueueDark(exposure_s=-3.0), "exposure_s"),
            (QueueDark(exposure_s=math.nan), "exposure_s"),
            (QueueDark(exposure_s=math.inf), "exposure_s"),
            (QueueDark(exposure_s=1e-9), "exposure_s"),
            (QueueDark(exposure_s=1e7), "exposure_s"),
            (QueueDark(label="x" * 81), "label"),
            (QueueDark(wait_for_cover_timeout_s=0.0), "wait_for_cover_timeout_s"),
            (QueueDark(wait_for_cover_timeout_s=-1.0), "wait_for_cover_timeout_s"),
            (QueueDark(wait_for_cover_timeout_s=math.nan), "wait_for_cover_timeout_s"),
            (QueueDark(wait_for_cover_timeout_s=math.inf), "wait_for_cover_timeout_s"),
            (QueueDark(wait_for_cover_timeout_s=7200.5), "wait_for_cover_timeout_s"),
        ],
    )
    def test_a_setting_out_of_range_is_invalid_and_names_the_field(
        self, command: QueueDark, word: str
    ) -> None:
        world, _ = night_world()
        result = world.scheduler.submit(command)
        assert (result.accepted, result.reason) == (False, RejectReason.INVALID)
        assert word in result.message
        assert world.scheduler.status().queued_tasks == 0

    @pytest.mark.parametrize(
        "command",
        [
            QueueDark(frames=3, bias_frames=3),
            QueueDark(frames=60, bias_frames=60),
            QueueDark(exposure_s=30.0),
            QueueDark(exposure_s=0.0001),
            QueueDark(label="x" * 80),
            QueueDark(wait_for_cover_timeout_s=0.001),
            QueueDark(wait_for_cover_timeout_s=7200.0),
            QueueDark(),
        ],
    )
    def test_the_edges_of_every_range_are_valid(self, command: QueueDark) -> None:
        world, _ = night_world()
        assert world.scheduler.submit(command).accepted

    def test_the_limits_come_from_the_configuration(self) -> None:
        config = SchedulerConfig.model_validate(
            {"dark": {"max_frames": 5, "max_label_chars": 4, "max_cover_wait_s": 60.0}}
        )
        world = World(start_utc_ns=NIGHT, config=config)
        world.scheduler.register_handler("dark", StandInDark(world.scheduler))
        assert not world.scheduler.submit(QueueDark(frames=6)).accepted
        assert not world.scheduler.submit(QueueDark(label="12345")).accepted
        assert not world.scheduler.submit(QueueDark(wait_for_cover_timeout_s=61.0)).accepted
        assert world.scheduler.submit(
            QueueDark(frames=5, label="1234", wait_for_cover_timeout_s=60.0)
        ).accepted

    def test_the_exposure_range_is_the_one_of_the_profile(self) -> None:
        low_us, high_us = PROFILE.limits.exposure_us_range
        world, _ = night_world()
        assert world.scheduler.submit(QueueDark(exposure_s=high_us / 1e6)).accepted
        world2, _ = night_world()
        assert not world2.scheduler.submit(QueueDark(exposure_s=(high_us + 1e6) / 1e6)).accepted
        assert low_us > 0


class TestWhenTheTaskStarts:
    def test_the_answer_says_the_next_step_for_a_session_that_starts_at_once(self) -> None:
        world, _ = night_world()
        queued = submit_at(world, 50, QueueDark())
        world.run_until(60)
        assert queued[0].state == "auto"
        assert "starts at the next step" in queued[0].message

    def test_the_answer_says_the_next_cycle_boundary_for_a_session_that_waits(self) -> None:
        world, _ = night_world()
        queued = submit_at(world, 50, QueueDark(immediate=False))
        world.run_until(60)
        assert queued[0].state == "auto"
        assert "runs at the next cycle boundary" in queued[0].message

    def test_a_session_queued_while_paused_waits_for_the_resume(self) -> None:
        world, stand_in = night_world()
        submit_at(world, 10, Pause())
        queued = submit_at(world, 20, QueueDark(pause_after=False))
        submit_at(world, 300, Resume())
        world.run_until(100)
        assert queued[0].accepted
        assert queued[0].state == "paused"
        assert "the scheduler is paused" in queued[0].message
        assert "runs after you resume" in queued[0].message
        assert stand_in is not None
        assert stand_in.tasks == []  # nothing runs while the scheduler is paused
        world.run_until(900)
        assert len(stand_in.tasks) == 1  # the resume starts it
        assert "commission" in world.states_visited()

    def test_a_session_queued_during_the_alignment_runs_after_it(self) -> None:
        world, stand_in = night_world()
        submit_at(world, 100, StartAlignment())
        queued = submit_at(world, 150, QueueDark(pause_after=False))
        submit_at(world, 300, StopAlignment())
        world.run_until(200)
        assert queued[0].state == "align"
        assert "runs after the alignment helper ends" in queued[0].message
        assert stand_in is not None
        assert stand_in.tasks == []
        world.run_until(900)
        assert len(stand_in.tasks) == 1


def started_at(world: World, kind: str = "dark") -> float:
    """The seconds since the start of the world at which the first task of a kind started."""
    events = [e for e in world.events("scheduler.task_started") if (e.detail or {})["kind"] == kind]
    return world.seconds(events[0].t_utc_ns)


def commission_reason(world: World) -> str:
    """The reason that the scheduler gave when it entered `commission`."""
    for event in world.events("scheduler.state_change"):
        detail = event.detail or {}
        if detail["to"] == "commission":
            return str(detail["reason"])
    raise AssertionError("the scheduler never entered commission")


def queued_at(world: World, command: str) -> float:
    """The seconds since the start of the world at which a command reached the scheduler."""
    events = [
        e for e in world.events("scheduler.command") if (e.detail or {})["command"] == command
    ]
    return world.seconds(events[0].t_utc_ns)


class TestStartingAtOnce:
    """A dark session does not wait for the cycle boundary, because someone stands at the camera.

    The first fast period of the scenario runs from 0 to 120 s, the survey step follows to about
    150 s, and the boundary is there.
    """

    def test_a_session_queued_in_a_fast_period_starts_within_one_frame_period(self) -> None:
        world, stand_in = night_world()
        submit_at(world, 20, QueueDark(pause_after=False))
        world.run_until(400)
        queued = queued_at(world, "QueueDark")
        assert 19.0 <= queued <= 23.0  # in the middle of the first fast period
        assert 0.0 <= started_at(world) - queued <= FRAME_PERIOD_S
        assert commission_reason(world) == "a dark session starts at once"
        assert stand_in is not None
        assert len(stand_in.tasks) == 1

    def test_the_fast_window_ends_as_a_partial_one_and_no_frame_is_lost(self) -> None:
        world, _ = night_world()
        submit_at(world, 20, QueueDark(pause_after=False))
        world.run_until(400)
        world.close()  # the window of the new cycle is still open, so close it
        queued = queued_at(world, "QueueDark")
        before = [w for w in world.windows() if world.seconds(w.t_utc_ns) < queued]
        assert before
        last = before[-1]
        assert world.seconds(last.t_utc_ns) + last.duration_s == pytest.approx(queued, abs=2.5)
        assert "partial" in last.flags
        assert sum(w.n_frames for w in world.windows()) == world.fast.frames_pushed
        assert sum(w.n_dropped for w in world.windows()) == 0

    def test_the_scheduler_starts_a_new_cycle_when_the_session_is_done(self) -> None:
        world, _ = night_world()
        submit_at(world, 20, QueueDark(pause_after=False))
        world.run_until(400)
        assert world.states_visited() == ["safe", "auto", "commission", "auto"]
        end = next(t for t, _, to in world.state_changes() if to == "auto" and t > 20)
        assert any(world.seconds(w.t_utc_ns) >= end for w in world.windows())

    def test_what_is_left_of_the_survey_step_is_skipped(self) -> None:
        world, _ = night_world()
        queued: list[CommandResult] = []
        original = world.survey.submit

        def submit_then_queue(frame: Any) -> None:
            original(frame)
            if len(world.survey.submitted) == 1:  # the short exposure is in, the long one is next
                queued.append(world.scheduler.submit(QueueDark(pause_after=False)))

        world.survey.submit = submit_then_queue  # type: ignore[method-assign]
        world.run_until(400)
        assert queued[0].accepted
        started = started_at(world)
        long_exposures = [
            c
            for c in world.camera.configure_log
            if c.config.exposure_us == LONG_EXPOSURE_US and world.seconds(c.t_utc_ns) < started
        ]
        assert long_exposures == []  # the exposure that would have taken 30 s did not start
        assert started - queued_at(world, "QueueDark") < 1.0

    def test_an_exposure_in_progress_finishes_first(self) -> None:
        world, _ = night_world()
        queued: list[CommandResult] = []
        original = world.survey.submit

        def queue_during_the_long_exposure(frame: Any) -> None:
            original(frame)
            if len(world.survey.submitted) == 2:  # the long frame is in: its exposure is over
                queued.append(world.scheduler.submit(QueueDark(pause_after=False)))

        world.survey.submit = queue_during_the_long_exposure  # type: ignore[method-assign]
        world.run_until(400)
        assert queued[0].accepted
        assert [f.exposure_us for f in world.survey.submitted[:2]] == [1000, LONG_EXPOSURE_US]
        assert started_at(world) - queued_at(world, "QueueDark") < 1.0

    def test_a_session_without_the_flag_waits_for_the_boundary(self) -> None:
        world, _ = night_world()
        submit_at(world, 20, QueueDark(immediate=False, pause_after=False))
        world.run_until(400)
        assert started_at(world) >= 120.0  # after the fast period
        assert commission_reason(world) == "a task is queued"

    def test_the_other_kinds_of_task_still_wait_for_the_boundary(self) -> None:
        world, _ = night_world()
        world.scheduler.register_handler("burst", Plain())
        submit_at(world, 20, QueueBurst())
        world.run_until(400)
        assert started_at(world, "burst") >= 120.0

    def test_a_dark_session_takes_the_waiting_tasks_with_it(self) -> None:
        world, _ = night_world()
        world.scheduler.register_handler("burst", Plain())
        submit_at(world, 20, QueueBurst(priority=-1))
        submit_at(world, 30, QueueDark(pause_after=False))
        world.run_until(400)
        assert [r.kind for r in world.results] == ["dark", "burst"]
        assert started_at(world, "burst") < 120.0  # the episode ran both

    def test_a_command_that_preempts_commissioning_still_ends_the_session(self) -> None:
        world, stand_in = night_world(frames=4, during=lambda s: s.submit(StartAlignment()))
        submit_at(world, 20, QueueDark(pause_after=True))
        world.run_until(400)
        assert stand_in is not None
        assert [r.status for r in world.results if r.kind == "dark"] == ["aborted"]
        assert "align" in world.states_visited()
        assert "paused" not in world.states_visited()  # the cut-short episode does not pause

    def test_a_pause_cuts_the_session_short_too(self) -> None:
        world, _ = night_world(frames=4, during=lambda s: s.submit(Pause()))
        submit_at(world, 20, QueueDark(pause_after=False))
        world.run_until(400)
        assert [r.status for r in world.results if r.kind == "dark"] == ["aborted"]
        assert world.states_visited()[-2:] == ["commission", "paused"]


class TestOneAtATime:
    def test_a_second_dark_task_is_busy_while_the_first_waits(self) -> None:
        world, _ = night_world()
        first = world.scheduler.submit(QueueDark())
        second = world.scheduler.submit(QueueDark(frames=5))
        assert first.accepted
        assert (second.accepted, second.reason) == (False, RejectReason.BUSY)
        assert "already queued or running" in second.message
        assert world.scheduler.status().queued_tasks == 1

    def test_a_second_dark_task_is_busy_while_the_first_runs(self) -> None:
        answers: list[CommandResult] = []

        def ask_again(scheduler: Scheduler) -> None:
            answers.append(scheduler.submit(QueueDark()))

        world, stand_in = night_world(during=ask_again)
        world.scheduler.submit(QueueDark())
        world.run_until(300)
        assert [(a.accepted, a.reason) for a in answers] == [(False, RejectReason.BUSY)]
        assert stand_in is not None
        assert len(stand_in.tasks) == 1

    def test_another_dark_task_is_fine_after_the_first_ended(self) -> None:
        world, stand_in = night_world()
        world.scheduler.submit(QueueDark(pause_after=False))
        world.run_until(300)
        assert stand_in is not None
        assert len(stand_in.tasks) == 1
        again = world.scheduler.submit(QueueDark(pause_after=False))
        assert again.accepted
        world.run_until(600)
        assert len(stand_in.tasks) == 2

    def test_a_burst_does_not_count_as_a_dark_task(self) -> None:
        world, _ = night_world()
        world.scheduler.register_handler("burst", Plain())
        assert world.scheduler.submit(QueueBurst()).accepted
        assert world.scheduler.submit(QueueDark()).accepted


class TestPauseAfter:
    def test_the_episode_ends_in_paused_and_the_event_says_why(self) -> None:
        world, stand_in = night_world()
        queued = submit_at(world, 50, QueueDark(pause_after=True))
        world.run_until(600)
        assert queued[0].accepted
        assert stand_in is not None
        assert len(stand_in.tasks) == 1
        assert world.states_visited() == ["safe", "auto", "commission", "paused"]
        assert world.scheduler.state.value == "paused"
        assert "dark session" in last_reason(world)
        assert "covered" in last_reason(world)

    def test_nothing_records_data_while_the_scheduler_is_paused(self) -> None:
        world, _ = night_world()
        submit_at(world, 50, QueueDark(pause_after=True))
        world.run_until(400)
        paused_at = next(t for t, _, to in world.state_changes() if to == "paused")
        before = len(world.windows())
        world.run_until(2000)
        assert len(world.windows()) == before
        assert paused_at < 400

    def test_resume_brings_the_system_back_to_auto(self) -> None:
        world, _ = night_world()
        submit_at(world, 50, QueueDark(pause_after=True))
        submit_at(world, 500, Resume())
        world.run_until(1500)
        assert world.states_visited() == ["safe", "auto", "commission", "paused", "safe", "auto"]
        assert any(world.seconds(w.t_utc_ns) > 500 for w in world.windows())

    def test_without_the_flag_the_scheduler_returns_where_it_was(self) -> None:
        world, _ = night_world()
        submit_at(world, 50, QueueDark(pause_after=False))
        world.run_until(600)
        assert world.states_visited() == ["safe", "auto", "commission", "auto"]

    def test_from_safe_the_episode_also_ends_in_paused(self) -> None:
        world = World()  # 14:30 UTC: the Sun is up, so the scheduler waits in `safe`
        world.scheduler.register_handler("dark", StandInDark(world.scheduler))
        submit_at(world, 20, QueueDark(pause_after=True))
        world.run_until(300)
        assert world.states_visited() == ["safe", "commission", "paused"]

    def test_the_pause_waits_for_the_other_tasks_of_the_episode(self) -> None:
        world, _ = night_world()
        world.scheduler.register_handler("burst", Plain())
        submit_at(world, 50, QueueDark(pause_after=True))
        submit_at(world, 50, QueueBurst(priority=-1))
        world.run_until(800)
        assert world.states_visited() == ["safe", "auto", "commission", "paused"]
        assert [r.kind for r in world.results] == ["dark", "burst"]

    def test_the_result_of_the_task_is_an_event_of_its_kind(self) -> None:
        world, _ = night_world()
        submit_at(world, 50, QueueDark())
        world.run_until(600)
        (result,) = world.events("scheduler.dark_result")
        assert result.detail is not None
        assert result.detail["status"] == "ok"
        assert [r.kind for r in world.results] == ["dark"]

    def test_a_command_that_cuts_the_task_short_leaves_nothing_behind(self) -> None:
        """A pause in the middle of the task ends the episode. A later task must not inherit it."""
        world, stand_in = night_world(frames=4, during=lambda s: s.submit(Pause()))
        world.scheduler.register_handler("burst", Plain())
        submit_at(world, 50, QueueDark(pause_after=True))
        submit_at(world, 800, Resume())
        submit_at(world, 1000, QueueBurst())
        world.run_until(2000)
        assert stand_in is not None
        assert [r.status for r in world.results if r.kind == "dark"] == ["aborted"]
        # The burst ran and the scheduler went on to `auto`: it did not pause a second time.
        assert [r.kind for r in world.results] == ["dark", "burst"]
        assert world.states_visited()[-3:] == ["auto", "commission", "auto"]
        assert world.scheduler.state.value == "auto"
