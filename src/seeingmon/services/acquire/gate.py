"""Keep the capture thread and the control thread from using a driver at the same time.

`acquire` reads frames on one thread and serves driver calls on another. A driver that does not
promise to be thread-safe must not see both at once, so `DriverGate` lets one side in at a time.
The control side has priority: when a control call waits, the capture thread does not start a
new read, so a call waits at most for the read in progress. (Without that rule, a capture thread
that releases and takes the lock in a tight loop starves the other thread.)

A driver that is safe to call from several threads, such as the `asi` driver, does not need the
gate, and `acquire` skips it (see `AcquireSettings.driver_threads`).

A read takes the gate for every frame, so the read side is a small class and not a generator: it
takes the lock once to enter and once to leave, and it wakes a control call only when one waits.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager


def _never() -> bool:
    return False


class _Read:
    """The hold of one read. `__enter__` returns whether the gate admitted the read."""

    __slots__ = ("_abort", "_admitted", "_gate")

    def __init__(self, gate: DriverGate, abort: Callable[[], bool]) -> None:
        self._gate = gate
        self._abort = abort
        self._admitted = False

    def __enter__(self) -> bool:
        gate = self._gate
        with gate._cond:
            if gate._controlling or gate._waiting:
                while gate._controlling or gate._waiting:
                    if self._abort():
                        break
                    gate._cond.wait(0.05)
                self._admitted = not (gate._controlling or gate._waiting)
            else:
                self._admitted = True
            if self._admitted:
                gate._reading = True
        return self._admitted

    def __exit__(self, *exc_info: object) -> None:
        if self._admitted:
            gate = self._gate
            with gate._cond:
                gate._reading = False
                if gate._waiting:
                    gate._cond.notify_all()


class DriverGate:
    """A lock with two sides: reads, which share nothing, and control calls, which come first."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._reading = False
        self._controlling = False
        self._waiting = 0

    def read(self, abort: Callable[[], bool] = _never) -> AbstractContextManager[bool]:
        """Hold the gate for one read. Yields `False` without holding it when `abort` is true.

        The block waits while a control call runs or waits. It checks `abort` every 50 ms.
        """
        return _Read(self, abort)

    @contextlib.contextmanager
    def control(self) -> Iterator[None]:
        """Hold the gate for one control call. Waits for a read in progress to end."""
        with self._cond:
            self._waiting += 1
            try:
                while self._reading or self._controlling:
                    self._cond.wait(0.05)
            finally:
                self._waiting -= 1
            self._controlling = True
        try:
            yield
        finally:
            with self._cond:
                self._controlling = False
                self._cond.notify_all()
