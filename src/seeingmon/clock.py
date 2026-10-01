"""The time source for the whole system.

Library code never reads the time or sleeps on its own (`time.time`, `time.sleep`,
`datetime.now`). It takes a `Clock`. A real run uses `SystemClock`, and tests use
`VirtualClock`, which advances only when told to, so a simulated night takes seconds.
`ScaledClock` runs faster than real time, and processes that share its parameters agree
on the time, which suits end-to-end tests with several processes.

All times are integers: nanoseconds since the Unix epoch in UTC (`utc_ns`), or nanoseconds
on a monotonic scale with an arbitrary origin (`monotonic_ns`). The wall clock can step
(NTP, a manual change), and the monotonic clock cannot. Measure durations with the
monotonic clock.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable

NS_PER_S = 1_000_000_000
NS_PER_US = 1_000

# 2026-01-01T00:00:00Z, a fixed default start for virtual time.
DEFAULT_START_UTC_NS = 1_767_225_600 * NS_PER_S


@dataclass(frozen=True, slots=True)
class ClockStatus:
    """What a clock knows about its own accuracy.

    `synchronized` is `None` when the clock cannot tell. `error_bound_ns` is the bound on
    the absolute error of `utc_ns` (for example, the chrony error bound), or `None` when it
    is unknown.
    """

    synchronized: bool | None
    error_bound_ns: int | None
    source: str


@runtime_checkable
class Clock(Protocol):
    """A source of time that tests can replace."""

    def utc_ns(self) -> int:
        """Nanoseconds since the Unix epoch, UTC. The value can step."""
        ...

    def monotonic_ns(self) -> int:
        """Nanoseconds on a clock that never steps back. The origin is arbitrary."""
        ...

    def sleep(self, seconds: float) -> None:
        """Block for `seconds` of this clock's time. A virtual clock advances instead."""
        ...

    def status(self) -> ClockStatus:
        """The accuracy of `utc_ns`."""
        ...


def sleep_until_utc_ns(clock: Clock, t_utc_ns: int) -> None:
    """Sleep until the clock's UTC time reaches `t_utc_ns`. Returns at once if it has."""
    remaining_ns = t_utc_ns - clock.utc_ns()
    if remaining_ns > 0:
        clock.sleep(remaining_ns / NS_PER_S)


class SystemClock:
    """The operating system clock.

    Pass `status_probe` to report the synchronization state, for example from chrony. The
    default reports an unknown state.
    """

    def __init__(self, status_probe: Callable[[], ClockStatus] | None = None) -> None:
        self._status_probe = status_probe

    def utc_ns(self) -> int:
        return time.time_ns()

    def monotonic_ns(self) -> int:
        return time.monotonic_ns()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)

    def status(self) -> ClockStatus:
        if self._status_probe is not None:
            return self._status_probe()
        return ClockStatus(synchronized=None, error_bound_ns=None, source="system")


class VirtualClock:
    """A clock that advances only when you call `advance` or `sleep`.

    `sleep` advances the clock and returns at once, so single-threaded code that sleeps
    runs as fast as the CPU allows. `step_utc_ns` moves the wall clock without moving the
    monotonic clock, which simulates an NTP step. The clock is thread-safe, but threads
    that sleep on it do not wait for each other. Drive it from one thread.
    """

    def __init__(
        self,
        start_utc_ns: int = DEFAULT_START_UTC_NS,
        *,
        monotonic_start_ns: int = 0,
        status: ClockStatus | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._monotonic_ns = monotonic_start_ns
        self._wall_offset_ns = start_utc_ns - monotonic_start_ns
        self._status = status or ClockStatus(synchronized=True, error_bound_ns=0, source="virtual")

    def utc_ns(self) -> int:
        with self._lock:
            return self._monotonic_ns + self._wall_offset_ns

    def monotonic_ns(self) -> int:
        with self._lock:
            return self._monotonic_ns

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self.advance_ns(round(seconds * NS_PER_S))

    def status(self) -> ClockStatus:
        with self._lock:
            return self._status

    def advance(self, seconds: float) -> None:
        """Move both clocks forward. A negative value raises `ValueError`."""
        self.advance_ns(round(seconds * NS_PER_S))

    def advance_ns(self, delta_ns: int) -> None:
        if delta_ns < 0:
            raise ValueError("a clock cannot move backward; use step_utc_ns to step the wall clock")
        with self._lock:
            self._monotonic_ns += delta_ns

    def advance_to_utc_ns(self, t_utc_ns: int) -> None:
        """Advance until `utc_ns()` equals `t_utc_ns`. A time in the past raises `ValueError`."""
        with self._lock:
            delta_ns = t_utc_ns - (self._monotonic_ns + self._wall_offset_ns)
            if delta_ns < 0:
                raise ValueError("t_utc_ns is in the past")
            self._monotonic_ns += delta_ns

    def step_utc_ns(self, delta_ns: int) -> None:
        """Step the wall clock by `delta_ns` (either sign). The monotonic clock stays put."""
        with self._lock:
            self._wall_offset_ns += delta_ns

    def set_status(self, status: ClockStatus) -> None:
        with self._lock:
            self._status = status


class ScaledClock:
    """A clock that runs `speed` times faster than real time.

    Construct every process with the same `start_utc_ns`, `origin_real_ns`, and `speed`, and
    they agree on the time. Take `origin_real_ns` from `time.time_ns()` once, and pass it to
    the other processes.
    """

    def __init__(self, *, start_utc_ns: int, origin_real_ns: int, speed: float) -> None:
        if speed <= 0:
            raise ValueError("speed must be positive")
        self._start_utc_ns = start_utc_ns
        self._origin_real_ns = origin_real_ns
        self._origin_monotonic_ns = time.monotonic_ns() - (time.time_ns() - origin_real_ns)
        self._speed = speed

    @property
    def speed(self) -> float:
        return self._speed

    def utc_ns(self) -> int:
        return self._start_utc_ns + round((time.time_ns() - self._origin_real_ns) * self._speed)

    def monotonic_ns(self) -> int:
        return round((time.monotonic_ns() - self._origin_monotonic_ns) * self._speed)

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds / self._speed)

    def status(self) -> ClockStatus:
        return ClockStatus(synchronized=True, error_bound_ns=0, source="scaled")


def utc_ns_to_datetime(t_utc_ns: int) -> datetime:
    """Convert to an aware `datetime` in UTC. The conversion truncates to microseconds."""
    return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=t_utc_ns // NS_PER_US)


def utc_ns_to_iso(t_utc_ns: int, *, digits: int = 6) -> str:
    """Format as an ISO 8601 UTC string, such as `2026-10-01T21:28:53.123456Z`.

    `digits` is the number of fractional-second digits (0 to 9). The API uses 6.
    """
    if not 0 <= digits <= 9:
        raise ValueError("digits must be between 0 and 9")
    seconds, nanoseconds = divmod(t_utc_ns, NS_PER_S)
    stamp = (datetime(1970, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    if digits == 0:
        return f"{stamp}Z"
    return f"{stamp}.{nanoseconds:09d}"[: len(stamp) + 1 + digits] + "Z"


def iso_to_utc_ns(text: str) -> int:
    """Parse an ISO 8601 UTC string with up to nine fractional digits.

    The string must end in `Z` or `+00:00`. Any other offset raises `ValueError`.
    """
    body = text.strip()
    if body.endswith("Z"):
        body = body[:-1]
    elif body.endswith("+00:00"):
        body = body[: -len("+00:00")]
    else:
        raise ValueError(f"not a UTC time: {text!r}")
    whole, dot, fraction = body.partition(".")
    if dot and not (fraction.isdigit() and 1 <= len(fraction) <= 9):
        raise ValueError(f"bad fractional seconds: {text!r}")
    moment = datetime.strptime(whole, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC)
    seconds = (moment - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(seconds=1)
    nanoseconds = int(fraction.ljust(9, "0")) if dot else 0
    return seconds * NS_PER_S + nanoseconds
