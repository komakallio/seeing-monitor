"""Keep the capture thread and the control thread from using a driver at the same time.

`acquire` reads frames on one thread and serves driver calls on another. A driver that does not
promise to be thread-safe must not see both at once, so `DriverGate` lets one side in at a time.
The control side has priority: when a control call waits, the capture thread does not start a
new read, so a call waits at most for the read in progress. (Without that rule, a capture thread
that releases and takes the lock in a tight loop starves the other thread.)

A driver that is safe to call from several threads, such as the `asi` driver, does not need the
gate, and `acquire` skips it (see `AcquireSettings.driver_threads`).
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Iterator


class DriverGate:
    """A lock with two sides: reads, which share nothing, and control calls, which come first."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._reading = False
        self._controlling = False
        self._waiting = 0

    @contextlib.contextmanager
    def read(self, abort: Callable[[], bool] = lambda: False) -> Iterator[bool]:
        """Hold the gate for one read. Yields `False` without holding it when `abort` is true.

        The block waits while a control call runs or waits. It checks `abort` every 50 ms.
        """
        with self._cond:
            while self._controlling or self._waiting:
                if abort():
                    break
                self._cond.wait(0.05)
            admitted = not (self._controlling or self._waiting)
            if admitted:
                self._reading = True
        try:
            yield admitted
        finally:
            if admitted:
                with self._cond:
                    self._reading = False
                    self._cond.notify_all()

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
