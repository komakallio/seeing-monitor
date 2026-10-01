"""Proof that the scheduler thread makes progress, for the watchdog of systemd.

The unit of `core` has `WatchdogSec`, and `core` sends `WATCHDOG=1` at half of that interval. A
heartbeat on a timer of its own proves only that the timer thread runs. It would go on while the
scheduler hangs, and systemd would never restart the process. So the supervisor asks `Liveness`
before it sends the heartbeat, and `Liveness` answers from what the scheduler thread does:

- **The clock.** The scheduler gets a `BeatClock`. Every read of the time and every sleep that the
  scheduler thread makes counts as progress. A loop that waits for a frame, an exposure, or the next
  slot reads the clock every few hundred milliseconds, so a healthy scheduler never goes quiet.
- **The driver calls.** The scheduler thread blocks inside a call of the camera driver for a long
  time when the camera recovers (a call to `recover` may take minutes). `InfoDriver` tells
  `Liveness` that it enters a call (`expect`) with the longest time that the call may take, and the
  scheduler counts as alive until then. A call that outlasts its limit is a hang.

The scheduler is alive when it made progress within `stall_s`, or when a driver call that is within
its limit is in flight. Other threads that read the same `BeatClock` do not count: the clock beats
only for the thread that called `bind_thread`.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

from seeingmon.clock import NS_PER_S, Clock, ClockStatus


class Liveness:
    """Tracks the progress of one thread. `alive` is true while the thread makes progress."""

    def __init__(self, clock: Clock, stall_s: float) -> None:
        if stall_s <= 0:
            raise ValueError("the stall limit is positive")
        self._clock = clock
        self._stall_ns = round(stall_s * NS_PER_S)
        self._lock = threading.Lock()
        self._last_ns = clock.monotonic_ns()
        self._grace_until_ns = 0
        self._thread_id: int | None = None
        self._local = threading.local()

    def bind_thread(self) -> None:
        """Count only the progress of the calling thread from now on."""
        with self._lock:
            self._thread_id = threading.get_ident()
            self._last_ns = self._clock.monotonic_ns()

    def counts_here(self) -> bool:
        """Whether a beat from the calling thread counts: no thread is bound, or it is this one."""
        if getattr(self._local, "quiet", False):
            return False
        thread_id = self._thread_id
        return thread_id is None or thread_id == threading.get_ident()

    @contextmanager
    def quiet(self) -> Iterator[None]:
        """Do not count what the calling thread does inside the block.

        A run that a test drives on one thread plays the part of two threads, the scheduler and the
        supervisor, and the supervisor must not prove that the scheduler lives.
        """
        previous = getattr(self._local, "quiet", False)
        self._local.quiet = True
        try:
            yield
        finally:
            self._local.quiet = previous

    def note(self, mono_ns: int) -> None:
        """Record progress at the monotonic time `mono_ns`. This is the hot path, so it takes no
        lock: a single store, and a reader that sees the old value only errs toward `alive`."""
        self._last_ns = mono_ns

    def beat(self) -> None:
        """Record progress now."""
        self.note(self._clock.monotonic_ns())

    def expect(self, seconds: float) -> None:
        """Say that the thread enters a call that may block for up to `seconds`."""
        now = self._clock.monotonic_ns()
        with self._lock:
            self._last_ns = now
            self._grace_until_ns = max(self._grace_until_ns, now + round(seconds * NS_PER_S))

    def leave(self) -> None:
        """Say that the call ended. This is progress, and it ends the grace of the call."""
        now = self._clock.monotonic_ns()
        with self._lock:
            self._last_ns = now
            self._grace_until_ns = 0

    @property
    def idle_s(self) -> float:
        """The seconds since the last progress."""
        return (self._clock.monotonic_ns() - self._last_ns) / NS_PER_S

    def alive(self) -> bool:
        """Whether the thread made progress recently, or waits inside a call that is in bounds."""
        now = self._clock.monotonic_ns()
        with self._lock:
            return now - self._last_ns <= self._stall_ns or now <= self._grace_until_ns


class BeatClock:
    """A clock that tells `Liveness` about the calls of the bound thread, and forwards the rest."""

    def __init__(self, clock: Clock, liveness: Liveness) -> None:
        self._clock = clock
        self._liveness = liveness

    def utc_ns(self) -> int:
        if self._liveness.counts_here():
            self._liveness.beat()
        return self._clock.utc_ns()

    def monotonic_ns(self) -> int:
        now = self._clock.monotonic_ns()
        if self._liveness.counts_here():
            self._liveness.note(now)
        return now

    def sleep(self, seconds: float) -> None:
        counts = self._liveness.counts_here()
        if counts:
            self._liveness.beat()
        self._clock.sleep(seconds)
        if counts:
            self._liveness.beat()

    def status(self) -> ClockStatus:
        return self._clock.status()
