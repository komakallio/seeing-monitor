"""The memory and CPU time of the current process, with the standard library only.

- **Linux:** `/proc/self/status` gives the resident set (`VmRSS`) and its peak (`VmHWM`, the
  high-water mark).
- **Windows:** `GetProcessMemoryInfo` (through `ctypes`) gives the working set (`WorkingSetSize`)
  and its peak (`PeakWorkingSetSize`).
- **Other systems:** `resource.getrusage` gives the peak, and the current size stays unknown.

Both operating systems count the resident pages of a process, including the pages of shared
libraries that the process touched. The numbers are therefore comparable across the two systems
within some tens of megabytes, and they stay an upper bound for the private memory of the process.
Every reader returns `None` when the system gives no answer, so a caller never fails on a platform
that the harness does not know.

`process_cpu_ns` reads the CPU time of the whole process (`time.process_time_ns`). Its granularity
is coarse on Windows (about 15.6 ms), so use it over a loop of at least a second, never per frame.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

_KIB = 1024


@dataclass(frozen=True, slots=True)
class MemoryReading:
    """The resident size of the process now (`None` when unknown) and its peak, in bytes."""

    rss_bytes: int | None
    peak_rss_bytes: int


def parse_proc_status(text: str) -> MemoryReading | None:
    """Read `VmRSS` and `VmHWM` from the text of `/proc/<pid>/status`.

    The kernel reports both in kilobytes. Returns `None` when the text has no `VmHWM` line.
    """
    fields: dict[str, int] = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        if name in ("VmRSS", "VmHWM"):
            parts = rest.split()
            if parts and parts[0].isdigit():
                fields[name] = int(parts[0]) * _KIB
    if "VmHWM" not in fields:
        return None
    return MemoryReading(fields.get("VmRSS"), fields["VmHWM"])


if sys.platform == "linux":

    def read_memory() -> MemoryReading | None:
        """Read the resident size and its peak. Returns `None` when the system gives no answer."""
        try:
            with open("/proc/self/status", encoding="ascii", errors="replace") as handle:
                return parse_proc_status(handle.read())
        except OSError:
            return None

elif sys.platform == "win32":
    import ctypes
    import functools
    from collections.abc import Callable
    from ctypes import wintypes

    class _ProcessMemoryCounters(ctypes.Structure):
        """`PROCESS_MEMORY_COUNTERS` of `psapi.h`."""

        _fields_ = (
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        )
        cb: int  # these annotations tell the type checker about the fields above
        PeakWorkingSetSize: int
        WorkingSetSize: int

    @functools.cache
    def _memory_info() -> Callable[[], MemoryReading | None]:
        # Private copies of the libraries, so that the argument types stay ours.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(_ProcessMemoryCounters),
            wintypes.DWORD,
        )
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        def read() -> MemoryReading | None:
            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if not psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
            ):
                return None
            return MemoryReading(int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize))

        return read

    def read_memory() -> MemoryReading | None:
        """Read the resident size and its peak. Returns `None` when the system gives no answer."""
        return _memory_info()()

else:

    def read_memory() -> MemoryReading | None:
        """Read the resident size and its peak. Returns `None` when the system gives no answer."""
        try:
            import resource
        except ImportError:
            return None
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS counts bytes, and the other Unix systems count kilobytes.
        return MemoryReading(None, int(peak if sys.platform == "darwin" else peak * _KIB))


def peak_rss_bytes() -> int | None:
    """The peak resident size of the current process, in bytes, or `None` when it is unknown."""
    reading = read_memory()
    return None if reading is None else reading.peak_rss_bytes


def current_rss_bytes() -> int | None:
    """The resident size of the current process now, in bytes, or `None` when it is unknown."""
    reading = read_memory()
    return None if reading is None else reading.rss_bytes


def process_cpu_ns() -> int:
    """The CPU time that the whole process has used (user and kernel), in nanoseconds."""
    return time.process_time_ns()
