"""Periodic work: tasks that run when they are due, on one thread or on the caller's.

`core` has a few small jobs that repeat: the `health` record every minute, the collection of the
events of `acquire`, the heartbeat for systemd. `PeriodicTasks` holds them. Each task has an
interval on the monotonic clock, and `run_due` runs the tasks that are due. The supervisor thread
of `core` calls `run_due` in a loop and sleeps until the next task is due. A test with a
`VirtualClock` calls `run_due` between scheduler steps, so a whole night runs on one thread, in
order, with no sleep to wait for.

A task that raises is logged and runs again at its next turn, so one bad job never stops the rest.
A task may return a number of seconds, which replaces its interval for the next turn (the heater
controller says how long to wait that way).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from seeingmon.clock import NS_PER_S, Clock

_log = logging.getLogger(__name__)

Task = Callable[[], float | None]


@dataclass(slots=True)
class _Entry:
    name: str
    interval_ns: int
    task: Task
    due_ns: int
    runs: int = 0
    failures: int = 0
    triggered: bool = False  # `trigger` asked for a run before the due time


class PeriodicTasks:
    """A list of tasks with intervals. Call `run_due` from one thread."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._entries: list[_Entry] = []

    def add(self, name: str, interval_s: float, task: Task, *, immediate: bool = True) -> None:
        """Add a task. It is due at once when `immediate` is true, and after one interval if not."""
        if interval_s <= 0:
            raise ValueError("an interval is positive")
        interval_ns = round(interval_s * NS_PER_S)
        now = self._clock.monotonic_ns()
        self._entries.append(
            _Entry(name, interval_ns, task, now if immediate else now + interval_ns)
        )

    def remove(self, name: str) -> None:
        """Remove every task of that name."""
        self._entries = [e for e in self._entries if e.name != name]

    def trigger(self, name: str) -> None:
        """Ask for the tasks of that name to run at the next `run_due`, whatever their due time.

        Any thread may call it. The task then runs once and keeps its interval afterward.
        """
        for entry in self._entries:
            if entry.name == name:
                entry.triggered = True

    def run_due(self) -> int:
        """Run the tasks that are due, in the order that they were added. Returns how many ran."""
        ran = 0
        for entry in list(self._entries):
            now = self._clock.monotonic_ns()
            if now < entry.due_ns and not entry.triggered:
                continue
            entry.triggered = False
            delay_s: float | None = None
            try:
                delay_s = entry.task()
            except Exception:
                entry.failures += 1
                _log.exception("the periodic task %s failed", entry.name)
            entry.runs += 1
            ran += 1
            interval_ns = entry.interval_ns
            if isinstance(delay_s, int | float) and delay_s > 0:
                interval_ns = round(delay_s * NS_PER_S)
            entry.due_ns = self._clock.monotonic_ns() + interval_ns
        return ran

    def seconds_until_next(self) -> float:
        """The time until the next task is due, in clock seconds. Zero or less means now."""
        if not self._entries:
            return 3600.0
        now = self._clock.monotonic_ns()
        if any(e.triggered for e in self._entries):
            return 0.0
        return (min(e.due_ns for e in self._entries) - now) / NS_PER_S

    def runs(self, name: str) -> int:
        """How many times the tasks of that name ran."""
        return sum(e.runs for e in self._entries if e.name == name)

    def failures(self, name: str) -> int:
        """How many times the tasks of that name raised."""
        return sum(e.failures for e in self._entries if e.name == name)
