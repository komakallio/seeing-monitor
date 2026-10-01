"""A deadline around each SDK call, and what happens when a call outlives it.

The vendor SDK is a closed binary, and a call into it can block for good after a USB fault. A
blocked call cannot be cancelled, and a Python thread cannot interrupt it. The only recovery is to
end the process and let the supervisor (systemd) start a new one. `CallWatchdog` detects the hang.

Arm a deadline around a call with `guard`:

    with watchdog.guard("get_video_data", timeout_s=0.6):
        api.get_video_data(...)

A thread (`start`) calls `check` every `poll_interval_s`. When a guarded call is still running
after its deadline, `check` calls `on_hang` once for that call. In production `on_hang` is
`exit_process_on_hang`, which dumps the stacks of all threads and ends the process at once. Tests
pass a function that records the report.

**Time.** The watchdog reads `Clock.monotonic_ns`. With a `VirtualClock`, no thread polls (the
clock would advance as fast as the thread can loop), so a test advances the clock and calls
`check` itself. Leaving a guard also checks its deadline, so a call that returns late still
reports its overrun when nothing polled in between.

**Threads.** Any number of threads can guard calls at the same time. `on_hang` runs on the thread
that noticed the hang, without the watchdog's lock held. A failing `on_hang` is logged and never
disturbs the guarded call.
"""

from __future__ import annotations

import faulthandler
import io
import logging
import os
import sys
import threading
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import IO

from seeingmon.clock import NS_PER_S, Clock, VirtualClock

HANG_EXIT_CODE = 70  # EX_SOFTWARE: an internal fault that a restart can clear
_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HangReport:
    """A guarded call that outlived its deadline."""

    name: str
    timeout_s: float
    elapsed_s: float


@dataclass(slots=True)
class _Guard:
    name: str
    timeout_s: float
    started_ns: int
    deadline_ns: int
    reported: bool = False


class CallWatchdog:
    """Arm deadlines around calls, and report a call that does not return in time.

    Args:
        clock: The time source. The watchdog reads `monotonic_ns` and sleeps through `sleep`.
        on_hang: Called once for each call that outlives its deadline.
        poll_interval_s: How often the thread started by `start` checks the deadlines.
    """

    def __init__(
        self,
        clock: Clock,
        on_hang: Callable[[HangReport], None],
        *,
        poll_interval_s: float = 0.1,
    ) -> None:
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        self._clock = clock
        self._on_hang = on_hang
        self._poll_interval_s = poll_interval_s
        self._lock = threading.Lock()
        self._guards: dict[int, _Guard] = {}
        self._next_token = 0
        self._hang_count = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def armed(self) -> int:
        """The number of calls that are running under a guard now."""
        with self._lock:
            return len(self._guards)

    @property
    def hang_count(self) -> int:
        """The number of hangs reported so far."""
        with self._lock:
            return self._hang_count

    def arm(self, name: str, timeout_s: float) -> int:
        """Start the deadline of a call. Returns a token for `disarm`. Prefer `guard`."""
        if timeout_s < 0:
            raise ValueError("timeout_s must not be negative")
        now = self._clock.monotonic_ns()
        with self._lock:
            self._next_token += 1
            token = self._next_token
            self._guards[token] = _Guard(name, timeout_s, now, now + round(timeout_s * NS_PER_S))
            return token

    def disarm(self, token: int) -> None:
        """End the deadline of a call. A call that overran and has not been reported reports now."""
        report = self._take_overdue(token)
        with self._lock:
            self._guards.pop(token, None)
        if report is not None:
            self._deliver(report)

    @contextmanager
    def guard(self, name: str, timeout_s: float) -> Iterator[None]:
        """Run a block under a deadline of `timeout_s` seconds."""
        token = self.arm(name, timeout_s)
        try:
            yield
        finally:
            self.disarm(token)

    def check(self) -> list[HangReport]:
        """Report each armed call that is past its deadline and not yet reported.

        Returns the new reports. A call that stays blocked reports once.
        """
        now = self._clock.monotonic_ns()
        reports: list[HangReport] = []
        with self._lock:
            for guard in self._guards.values():
                if not guard.reported and now > guard.deadline_ns:
                    guard.reported = True
                    self._hang_count += 1
                    reports.append(
                        HangReport(guard.name, guard.timeout_s, (now - guard.started_ns) / NS_PER_S)
                    )
        for report in reports:
            self._deliver(report)
        return reports

    def _take_overdue(self, token: int) -> HangReport | None:
        now = self._clock.monotonic_ns()
        with self._lock:
            guard = self._guards.get(token)
            if guard is None or guard.reported or now <= guard.deadline_ns:
                return None
            guard.reported = True
            self._hang_count += 1
            return HangReport(guard.name, guard.timeout_s, (now - guard.started_ns) / NS_PER_S)

    def _deliver(self, report: HangReport) -> None:
        try:
            self._on_hang(report)
        except Exception:
            _log.exception("the hang handler failed for %s", report.name)

    # --- The polling thread ---

    def start(self) -> None:
        """Start the thread that calls `check`. Does nothing if it already runs.

        Raises `ValueError` for a `VirtualClock`, because the thread would advance virtual time
        without limit. Drive `check` from the test instead.
        """
        if isinstance(self._clock, VirtualClock):
            raise ValueError("a virtual clock cannot drive a polling thread; call check() instead")
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="call-watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the thread. Does nothing if it does not run."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(self._poll_interval_s * 5 + 1.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._clock.sleep(self._poll_interval_s)
            try:
                self.check()
            except Exception:  # keep the thread alive: it is the safety net
                _log.exception("the watchdog check failed")


def dump_thread_stacks(target: IO[str]) -> None:
    """Write the stack of every thread to `target`.

    The function uses `faulthandler` when `target` has a file descriptor, because it also works
    while the interpreter is unhealthy. Otherwise it formats the stacks itself.
    """
    try:
        target.fileno()
    except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
        names = {thread.ident: thread.name for thread in threading.enumerate()}
        for ident, frame in sys._current_frames().items():
            target.write(f"Thread {names.get(ident, ident)}:\n")
            target.write("".join(traceback.format_stack(frame)))
        return
    faulthandler.dump_traceback(file=target, all_threads=True)


def make_exit_handler(
    *,
    exit_process: Callable[[int], object] = os._exit,
    dump_stacks: Callable[[IO[str]], object] = dump_thread_stacks,
    stream: IO[str] | None = None,
) -> Callable[[HangReport], None]:
    """Build the handler that ends the process when a call hangs.

    The handler writes one line and the stack of every thread to `stream` (standard error by
    default), so the journal shows where the SDK call blocked. Then it ends the process at once with
    `HANG_EXIT_CODE`, without cleanup, because a thread blocked inside the SDK would block a clean
    shutdown too. The supervisor starts a new process.
    """

    def handler(report: HangReport) -> None:
        target = stream if stream is not None else sys.stderr
        try:
            target.write(
                f"The SDK call {report.name} ran for {report.elapsed_s:.1f} s, over its limit of "
                f"{report.timeout_s:.1f} s. Ending the process.\n"
            )
            target.flush()
            dump_stacks(target)
        except Exception:  # the report is a courtesy and must not block the exit
            _log.exception("the hang report failed")
        exit_process(HANG_EXIT_CODE)

    return handler


exit_process_on_hang = make_exit_handler()
