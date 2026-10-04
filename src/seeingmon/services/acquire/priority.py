"""Raise the priority of the capture thread, and on Windows the resolution of the timer.

The capture thread waits in the SDK most of the time, and it must wake up promptly when a frame
arrives, because the arrival time is the basis of the frame time. A higher priority keeps other
work, such as the sender and the analysis in `core`, from delaying the wake-up.

`raise_current_thread_priority` tries, and it never raises: without the privilege, the thread
keeps its normal priority, and the function says so. A systemd unit can grant the privilege
(`AmbientCapabilities=CAP_SYS_NICE` on Linux).

**Windows priority.** The thread gets `THREAD_PRIORITY_HIGHEST`, two steps above normal inside the
normal priority class. It needs no privilege, and it leaves the other threads and the class of the
process alone. The choice rests on a measurement on the dev machine: a fake camera at 82 frames per
second, read by the real capture loop in 30 s runs. At normal priority, 4 to 10% of the read
intervals exceeded 1.5 frame periods, and the 99th percentile was 23 to 29 ms against a period of
12.2 ms. At `THREAD_PRIORITY_ABOVE_NORMAL`, the 99th percentile fell to 16 to 18 ms and the late
reads to under 1%. At `THREAD_PRIORITY_HIGHEST`, it fell to 14 to 15 ms and the late reads to 0.04%
or fewer. A higher level (time-critical) or a higher class for the process would take more from
the machine and add nothing here.

**Windows timer.** Windows runs its system timer at 15.6 ms unless a process asks for a finer
period, and a wait with a timeout (`Event.wait`, `Sleep`, the polling loop of a vendor library)
ends on a timer tick. On the dev machine, a wait of 1 ms took 14 ms at the median, and 1.2 ms after
`timeBeginPeriod(1)`. `TimerResolution` asks for a period of 1 ms for as long as `acquire` runs.
The request is for the process and costs a little power, which a dev machine can spare. Whether
the vendor library waits on timer ticks is unknown, so the request is a precaution, and the
health summary shows whether it applied.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections.abc import Callable
from typing import Any

_log = logging.getLogger(__name__)

THREAD_PRIORITY_HIGHEST = 2  # two steps above normal, inside the normal priority class
TIMER_RESOLUTION_MS = 1  # the period that `TimerResolution` asks Windows for
TIMERR_NOERROR = 0  # what `timeBeginPeriod` returns when it grants the period


def windows_priority(kernel32: Any, last_error: Callable[[], int]) -> str:
    """Raise the calling thread to the highest priority of its class, on a `kernel32` object.

    `kernel32` offers `GetCurrentThread` and `SetThreadPriority` as the Windows library does.
    Returns what happened, in words, and never raises for a refused call.
    """
    if kernel32.SetThreadPriority(kernel32.GetCurrentThread(), THREAD_PRIORITY_HIGHEST):
        return "highest thread priority"
    return f"the call failed with error {last_error()}, so the thread keeps its normal priority"


if sys.platform == "linux":

    def raise_current_thread_priority() -> str:
        """Raise the priority of the calling thread. Returns what happened, in words."""
        try:
            os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(10))  # 0 is the calling thread
            return "real-time round-robin priority 10"
        except OSError:
            pass
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), -10)
            return "nice -10"
        except OSError:
            return "not permitted, so the thread keeps its normal priority"

elif sys.platform == "win32":

    def raise_current_thread_priority() -> str:
        """Raise the priority of the calling thread. Returns what happened, in words."""
        import ctypes

        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # a private copy, so the
            kernel32.GetCurrentThread.restype = ctypes.c_void_p  # argument types stay ours
            kernel32.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
            kernel32.SetThreadPriority.restype = ctypes.c_int
            return windows_priority(kernel32, ctypes.get_last_error)
        except (OSError, AttributeError) as error:
            return (
                f"the call raised {type(error).__name__}, so the thread keeps its normal priority"
            )

else:

    def raise_current_thread_priority() -> str:
        """Raise the priority of the calling thread. Returns what happened, in words."""
        return "not supported on this platform"


if sys.platform == "win32":

    def _load_winmm() -> Any:
        """The Windows multimedia library, which holds the timer calls."""
        import ctypes

        return ctypes.WinDLL("winmm")

else:

    def _load_winmm() -> Any:
        """Another platform has no such library, and needs no timer request."""
        return None


class TimerResolution:
    """A request for a finer system timer, held from `request` to `release`. Windows only.

    On another platform, `request` does nothing and returns an empty string, because the platform
    needs no request. `release` undoes a request that Windows granted, and nothing else.
    """

    def __init__(self, milliseconds: int = TIMER_RESOLUTION_MS, *, winmm: Any = None) -> None:
        self._milliseconds = milliseconds
        self._winmm = winmm  # a test passes a stand-in for the Windows library
        self._granted = False

    @property
    def granted(self) -> bool:
        """Whether Windows granted the period, and `release` has not given it back."""
        return self._granted

    def request(self) -> str:
        """Ask for the period. Returns what happened, in words, or `""` where it does not apply."""
        winmm = self._winmm
        if winmm is None:
            try:
                winmm = _load_winmm()
            except OSError as error:
                kind = type(error).__name__
                return f"the library raised {kind}, so the timer keeps its default resolution"
            if winmm is None:
                return ""
            self._winmm = winmm
        code = winmm.timeBeginPeriod(self._milliseconds)
        self._granted = code == TIMERR_NOERROR
        if self._granted:
            return f"{self._milliseconds} ms resolution"
        return f"the call failed with code {code}, so the timer keeps its default resolution"

    def release(self) -> None:
        """Give the period back, if Windows granted it."""
        if not self._granted:
            return
        self._granted = False
        try:
            self._winmm.timeEndPeriod(self._milliseconds)
        except Exception:  # the process is ending, and Windows takes the request back anyway
            _log.debug("could not release the timer resolution", exc_info=True)
