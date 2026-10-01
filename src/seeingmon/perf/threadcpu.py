"""The CPU time of a thread, read from another thread, with the standard library only.

`time.thread_time_ns` reads the calling thread only. A process that wants to know where its threads
spend their time must read the clocks of the others, and this module does that:

- **Linux:** `time.pthread_getcpuclockid` gives the CPU clock of a thread, and
  `time.clock_gettime_ns` reads it with nanosecond resolution.
- **Windows:** `GetThreadTimes` (through `ctypes`) gives the kernel and user time in units of
  100 ns. The kernel charges the time in clock ticks of about 15.6 ms, so read a thread that runs
  for seconds, and never one that runs for milliseconds.
- **Other systems:** the function returns `None`.

`threads_cpu_ns` reads every live thread of the process, and returns the time by thread name. A
thread name is not unique, so the function adds the thread ID to a name that repeats.
"""

from __future__ import annotations

import sys
import threading
import time

if sys.platform == "linux":

    def thread_cpu_ns(thread: threading.Thread) -> int | None:
        """The CPU time that a thread has used, in nanoseconds, or `None` when it is unknown."""
        if thread.ident is None:
            return None
        try:
            return time.clock_gettime_ns(time.pthread_getcpuclockid(thread.ident))
        except (OSError, ProcessLookupError):  # the thread ended
            return None

elif sys.platform == "win32":
    import ctypes
    import functools
    from collections.abc import Callable
    from ctypes import wintypes

    _THREAD_QUERY_LIMITED_INFORMATION = 0x0800

    @functools.cache
    def _thread_times() -> Callable[[int], int | None]:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # private copy, our own types
        kernel32.OpenThread.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenThread.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.GetThreadTimes.argtypes = (
            wintypes.HANDLE,
            *([ctypes.POINTER(wintypes.FILETIME)] * 4),
        )
        kernel32.GetThreadTimes.restype = wintypes.BOOL

        def ticks(value: wintypes.FILETIME) -> int:
            return (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)

        def read(native_id: int) -> int | None:
            handle = kernel32.OpenThread(_THREAD_QUERY_LIMITED_INFORMATION, False, native_id)
            if not handle:
                return None
            try:
                created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
                if not kernel32.GetThreadTimes(
                    handle,
                    ctypes.byref(created),
                    ctypes.byref(exited),
                    ctypes.byref(kernel),
                    ctypes.byref(user),
                ):
                    return None
                return (ticks(kernel) + ticks(user)) * 100  # units of 100 ns
            finally:
                kernel32.CloseHandle(handle)

        return read

    def thread_cpu_ns(thread: threading.Thread) -> int | None:
        """The CPU time that a thread has used, in nanoseconds, or `None` when it is unknown."""
        if thread.native_id is None:
            return None
        return _thread_times()(thread.native_id)

else:

    def thread_cpu_ns(thread: threading.Thread) -> int | None:
        """The CPU time that a thread has used, in nanoseconds, or `None` when it is unknown."""
        return None


def threads_cpu_ns() -> dict[str, int]:
    """The CPU time of every live thread of this process, by thread name, in nanoseconds.

    A thread that the system gives no reading for is left out. A name that repeats gets the
    thread ID added.
    """
    found: dict[str, int] = {}
    for thread in threading.enumerate():
        used = thread_cpu_ns(thread)
        if used is None:
            continue
        name = thread.name
        if name in found:
            name = f"{name}-{thread.ident}"
        found[name] = used
    return found
