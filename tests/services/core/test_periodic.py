"""The periodic tasks of core: intervals, the first run, a delay from the task, and a trigger."""

from __future__ import annotations

import pytest

from seeingmon.clock import VirtualClock
from seeingmon.services.core.periodic import PeriodicTasks


def make() -> tuple[VirtualClock, PeriodicTasks, list[str]]:
    clock = VirtualClock(1_000_000_000_000_000_000)
    return clock, PeriodicTasks(clock), []


class TestRunning:
    def test_a_task_runs_at_once_and_then_every_interval(self) -> None:
        clock, tasks, calls = make()
        tasks.add("a", 10.0, lambda: calls.append("a"))
        assert tasks.run_due() == 1
        clock.advance(9.0)
        assert tasks.run_due() == 0
        clock.advance(1.0)
        assert tasks.run_due() == 1
        assert calls == ["a", "a"]
        assert tasks.runs("a") == 2

    def test_a_task_can_wait_for_its_first_interval(self) -> None:
        clock, tasks, calls = make()
        tasks.add("a", 10.0, lambda: calls.append("a"), immediate=False)
        assert tasks.run_due() == 0
        clock.advance(10.0)
        assert tasks.run_due() == 1

    def test_the_tasks_run_in_the_order_that_they_were_added(self) -> None:
        _, tasks, calls = make()
        tasks.add("first", 1.0, lambda: calls.append("first"))
        tasks.add("second", 1.0, lambda: calls.append("second"))
        tasks.run_due()
        assert calls == ["first", "second"]

    def test_a_task_that_raises_runs_again_at_its_turn_and_does_not_stop_the_rest(self) -> None:
        clock, tasks, calls = make()

        def broken() -> None:
            raise RuntimeError("a bad job")

        tasks.add("broken", 1.0, broken)
        tasks.add("good", 1.0, lambda: calls.append("good"))
        tasks.run_due()
        clock.advance(1.0)
        tasks.run_due()
        assert calls == ["good", "good"]
        assert tasks.failures("broken") == 2
        assert tasks.failures("good") == 0

    def test_a_task_can_name_its_next_delay(self) -> None:
        clock, tasks, calls = make()

        def poll() -> float:
            calls.append("poll")
            return 2.5

        tasks.add("poll", 60.0, poll)
        tasks.run_due()
        clock.advance(2.4)
        assert tasks.run_due() == 0
        clock.advance(0.1)
        assert tasks.run_due() == 1

    def test_the_wait_until_the_next_task_is_known(self) -> None:
        clock, tasks, _ = make()
        assert tasks.seconds_until_next() == 3600.0  # nothing to do: sleep long
        tasks.add("a", 10.0, lambda: None, immediate=False)
        tasks.add("b", 4.0, lambda: None, immediate=False)
        assert tasks.seconds_until_next() == pytest.approx(4.0)
        clock.advance(5.0)
        assert tasks.seconds_until_next() == pytest.approx(-1.0)  # a task is overdue

    def test_an_interval_is_positive(self) -> None:
        _, tasks, _ = make()
        with pytest.raises(ValueError, match="positive"):
            tasks.add("a", 0.0, lambda: None)

    def test_a_task_can_be_removed(self) -> None:
        _, tasks, calls = make()
        tasks.add("a", 1.0, lambda: calls.append("a"))
        tasks.remove("a")
        tasks.run_due()
        assert calls == []


class TestTrigger:
    def test_a_trigger_runs_the_task_before_it_is_due_and_keeps_the_interval(self) -> None:
        clock, tasks, calls = make()
        tasks.add("health", 60.0, lambda: calls.append("health"))
        tasks.run_due()
        clock.advance(5.0)
        tasks.trigger("health")
        assert tasks.seconds_until_next() == 0.0
        assert tasks.run_due() == 1
        assert calls == ["health", "health"]
        assert tasks.run_due() == 0  # a trigger runs the task once
        clock.advance(59.0)
        assert tasks.run_due() == 0
        clock.advance(1.0)
        assert tasks.run_due() == 1  # the interval counts from the triggered run

    def test_a_trigger_during_a_run_is_not_lost(self) -> None:
        _, tasks, calls = make()

        def slow() -> None:
            calls.append("slow")
            if len(calls) == 1:
                tasks.trigger("slow")  # another thread asks while the task runs

        tasks.add("slow", 60.0, slow)
        tasks.run_due()
        assert tasks.run_due() == 1  # the second run is the triggered one
        assert calls == ["slow", "slow"]

    def test_a_trigger_for_an_unknown_name_does_nothing(self) -> None:
        _, tasks, _ = make()
        tasks.trigger("nothing")
        assert tasks.run_due() == 0
