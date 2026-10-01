"""A probe of the clock's synchronization state, for the `SystemClock` of `acquire` and `core`.

A Raspberry Pi 4 has no real-time clock, so after a boot its UTC time can be days off until chrony
reaches a time server. Every frame, window, and record then needs the `time_invalid` mark. The probe
tells the clock whether it is synchronized, and how large the error can be, so that `acquire` marks
frames and `core` marks windows and the `health` record.

**The kernel knows.** chrony steers the kernel clock, and the kernel keeps the state of the
synchronization: the `STA_UNSYNC` status bit, and `maxerror`, the bound on the absolute error of the
clock in microseconds. The `adjtimex` call with `modes = 0` reads both and changes nothing, so it
needs no privilege. The systemd units of the project allow the call (`SystemCallFilter=adjtimex
clock_adjtime`), and nothing else would answer the question without a subprocess.

**The rules.** The clock is synchronized when the state that the call returns is not `TIME_ERROR`
and `STA_UNSYNC` is clear. The error bound is `maxerror`. A system that cannot answer (another
operating system, a filtered call) gives an unknown state, `ClockStatus(None, None, ...)`, and the
callers treat unknown as usable, as before.

`make_probe("auto")` picks `adjtimex` on Linux and no probe elsewhere.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from collections.abc import Callable

from seeingmon.clock import ClockStatus

_log = logging.getLogger(__name__)

STA_UNSYNC = 0x0040
TIME_ERROR = 5
NS_PER_US = 1000

Probe = Callable[[], ClockStatus]


class _Timeval(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class Timex(ctypes.Structure):
    """`struct timex` of Linux (`<sys/timex.h>`). It is 208 bytes where `long` has 64 bits."""

    _fields_ = [
        ("modes", ctypes.c_uint),
        ("offset", ctypes.c_long),
        ("freq", ctypes.c_long),
        ("maxerror", ctypes.c_long),
        ("esterror", ctypes.c_long),
        ("status", ctypes.c_int),
        ("constant", ctypes.c_long),
        ("precision", ctypes.c_long),
        ("tolerance", ctypes.c_long),
        ("time", _Timeval),
        ("tick", ctypes.c_long),
        ("ppsfreq", ctypes.c_long),
        ("jitter", ctypes.c_long),
        ("shift", ctypes.c_int),
        ("stabil", ctypes.c_long),
        ("jitcnt", ctypes.c_long),
        ("calcnt", ctypes.c_long),
        ("errcnt", ctypes.c_long),
        ("stbcnt", ctypes.c_long),
        ("tai", ctypes.c_int),
        ("padding", ctypes.c_int * 11),
    ]


def status_from_kernel(state: int, status: int, maxerror_us: int) -> ClockStatus:
    """The clock status that the values of `adjtimex` mean. See the module text for the rules."""
    synchronized = state != TIME_ERROR and not status & STA_UNSYNC
    return ClockStatus(
        synchronized=synchronized,
        error_bound_ns=max(0, maxerror_us) * NS_PER_US,
        source="adjtimex",
    )


UNKNOWN = ClockStatus(synchronized=None, error_bound_ns=None, source="adjtimex")


class AdjtimexProbe:
    """Reads the synchronization state of the kernel clock with `adjtimex`. Linux only.

    `call` replaces the system call for a test: it returns `(state, status, maxerror_us)`, or
    `None` when the call failed.
    """

    def __init__(self, call: Callable[[], tuple[int, int, int] | None] | None = None) -> None:
        self._call = call or self._system_call
        self._failed_once = False

    @staticmethod
    def _system_call() -> tuple[int, int, int] | None:
        libc = ctypes.CDLL(None, use_errno=True)
        adjtimex = libc.adjtimex
        adjtimex.argtypes = [ctypes.POINTER(Timex)]
        adjtimex.restype = ctypes.c_int
        timex = Timex()
        state = adjtimex(ctypes.byref(timex))  # modes = 0: read only
        if state < 0:
            return None
        return int(state), int(timex.status), int(timex.maxerror)

    def __call__(self) -> ClockStatus:
        try:
            answer = self._call()
        except (OSError, AttributeError):  # no libc, or no adjtimex in it
            answer = None
        if answer is None:
            if not self._failed_once:
                self._failed_once = True
                _log.warning("the clock probe cannot read the kernel clock state, so it is unknown")
            return UNKNOWN
        return status_from_kernel(*answer)


def make_probe(kind: str) -> Probe | None:
    """The probe for a setting: `adjtimex`, `none`, or `auto` (`adjtimex` on Linux only)."""
    if kind == "none":
        return None
    if kind == "adjtimex" or (kind == "auto" and sys.platform == "linux"):
        return AdjtimexProbe()
    return None
