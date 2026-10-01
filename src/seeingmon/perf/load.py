"""How busy the machine is, so that a report can say whether other work disturbed a run.

`system_busy_percent` samples the processor time of the whole machine over a short interval and
returns the share that was not idle, in percent of all logical processors. On a quiet machine
the value stays in the low single digits. A run on a machine that is 30% busy shows larger
medians and maxima, and the figure explains why.

- **Linux:** the first line of `/proc/stat`.
- **Windows:** `GetSystemTimes` (through `ctypes`).
- **Other systems:** the function returns `None`.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable

Sleep = Callable[[float], None]


def busy_percent_between(first: tuple[int, int], second: tuple[int, int]) -> float | None:
    """The busy share between two readings of `(busy ticks, total ticks)`, in percent.

    Returns `None` when the total did not advance, because a share needs a duration.
    """
    total = second[1] - first[1]
    if total <= 0:
        return None
    busy = min(max(second[0] - first[0], 0), total)
    return 100.0 * busy / total


def parse_proc_stat(line: str) -> tuple[int, int] | None:
    """`(busy, total)` clock ticks from the first line of `/proc/stat`, or `None` if it is not one.

    The line reads `cpu user nice system idle iowait irq softirq steal ...`. The idle time is the
    `idle` and `iowait` columns. The total is the first eight columns, because the guest columns
    are already part of `user` and `nice`.
    """
    parts = line.split()
    if len(parts) < 5 or parts[0] != "cpu":
        return None
    values = [int(item) for item in parts[1:] if item.isdigit()]
    if len(values) < 4:
        return None
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    total = sum(values[:8])
    return total - idle, total


if sys.platform == "linux":

    def _read_ticks() -> tuple[int, int] | None:
        """`(busy, total)` clock ticks of all processors since the start of the system."""
        try:
            with open("/proc/stat", encoding="ascii", errors="replace") as handle:
                return parse_proc_stat(handle.readline())
        except OSError:
            return None

elif sys.platform == "win32":
    import ctypes
    import functools
    from ctypes import wintypes

    @functools.cache
    def _system_times() -> Callable[[], tuple[int, int] | None]:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetSystemTimes.argtypes = (
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
        )
        kernel32.GetSystemTimes.restype = wintypes.BOOL

        def ticks(value: wintypes.FILETIME) -> int:
            return (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)

        def read() -> tuple[int, int] | None:
            idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
            if not kernel32.GetSystemTimes(
                ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
            ):
                return None
            # The kernel time includes the idle time.
            total = ticks(kernel) + ticks(user)
            return total - ticks(idle), total

        return read

    def _read_ticks() -> tuple[int, int] | None:
        """`(busy, total)` ticks of all processors since the start of the system."""
        return _system_times()()

else:

    def _read_ticks() -> tuple[int, int] | None:
        """`(busy, total)` ticks of all processors. This system gives none."""
        return None


def system_busy_percent(sample_s: float = 0.2, *, sleep: Sleep = time.sleep) -> float | None:
    """The share of processor time, over all logical processors, that was not idle during
    the next `sample_s` seconds, in percent. Returns `None` when the system gives no answer.
    """
    first = _read_ticks()
    if first is None:
        return None
    sleep(sample_s)
    second = _read_ticks()
    if second is None:
        return None
    return busy_percent_between(first, second)
