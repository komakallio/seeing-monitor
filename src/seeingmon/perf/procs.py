"""The memory and CPU time of other processes, read from outside, with the standard library only.

The `memory` module reads the process that runs the code. A run of the whole system needs the same
figures for the processes that it started, and for their children, without a change to the code
that those processes run. This module reads them:

- **Linux:** `/proc/<pid>/status` gives the resident set and its peak (`VmHWM`), `/proc/<pid>/stat`
  gives the CPU time in clock ticks (10 ms), and `/proc/<pid>/task/<tid>/stat` gives the same for
  each thread. The parent of a process comes from the same file, the command line from
  `/proc/<pid>/cmdline`, and the name that the process gives itself from `/proc/<pid>/comm`.
- **Windows:** `GetProcessMemoryInfo` gives the peak working set, `GetProcessTimes` gives the CPU
  time in units of 100 ns (the kernel charges it in clock ticks of about 15.6 ms), and a snapshot of
  the process list (`CreateToolhelp32Snapshot`) gives the parents. A thread reading needs a thread
  handle for each thread, so `thread_cpu_ns` returns nothing on Windows.
- **Other systems:** every reader returns `None` or an empty result.

**The launcher of a virtual environment.** On Windows, the `python.exe` of a virtual environment
starts the interpreter of the base installation as its only child, and the child does the work.
`is_venv_launcher` finds such a process, so that a caller can read the child instead.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from seeingmon.perf.memory import parse_proc_status


@dataclass(frozen=True, slots=True)
class ProcessReading:
    """The CPU time and the memory of one process. A field is `None` when the system gives none."""

    pid: int
    cpu_ns: int | None
    peak_rss_bytes: int | None
    rss_bytes: int | None


def parse_stat(text: str) -> tuple[int, int, int] | None:
    """The parent, and the user and kernel time in clock ticks, from the text of a `stat` file.

    The name of the process sits in parentheses and may hold spaces and parentheses itself, so the
    fields count from the last closing parenthesis. Returns `None` for text that has no such fields.
    """
    end = text.rfind(")")
    if end < 0:
        return None
    fields = text[end + 2 :].split()
    if len(fields) < 13:
        return None
    try:
        return int(fields[1]), int(fields[11]), int(fields[12])
    except ValueError:
        return None


if sys.platform == "linux":
    _TICK_NS = 1_000_000_000 // os.sysconf("SC_CLK_TCK")

    def _read(path: str) -> str | None:
        try:
            with open(path, encoding="ascii", errors="replace") as handle:
                return handle.read()
        except OSError:
            return None

    def read_process(pid: int) -> ProcessReading | None:
        """Read the CPU time and the memory of a process. Returns `None` when it is gone."""
        stat = _read(f"/proc/{pid}/stat")
        parsed = None if stat is None else parse_stat(stat)
        if parsed is None:
            return None
        status = _read(f"/proc/{pid}/status")
        memory = None if status is None else parse_proc_status(status)
        return ProcessReading(
            pid,
            (parsed[1] + parsed[2]) * _TICK_NS,
            None if memory is None else memory.peak_rss_bytes,
            None if memory is None else memory.rss_bytes,
        )

    def thread_cpu_ns(pid: int) -> dict[int, int]:
        """The CPU time of each thread of a process, by thread ID, in nanoseconds."""
        found: dict[int, int] = {}
        try:
            names = os.listdir(f"/proc/{pid}/task")
        except OSError:
            return found
        for name in names:
            if not name.isdigit():
                continue
            stat = _read(f"/proc/{pid}/task/{name}/stat")
            parsed = None if stat is None else parse_stat(stat)
            if parsed is not None:
                found[int(name)] = (parsed[1] + parsed[2]) * _TICK_NS
        return found

    def parent_map() -> dict[int, int]:
        """The parent of every process, by process ID."""
        parents: dict[int, int] = {}
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            stat = _read(f"/proc/{name}/stat")
            parsed = None if stat is None else parse_stat(stat)
            if parsed is not None:
                parents[int(name)] = parsed[0]
        return parents

    def command_line(pid: int) -> str | None:
        """The command line of a process, with a space between the words, or `None`."""
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                raw = handle.read()
        except OSError:
            return None
        return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()

    def image_path(pid: int) -> str | None:
        """The path of the program that a process runs, or `None`."""
        try:
            return os.readlink(f"/proc/{pid}/exe")
        except OSError:
            return None

    def process_name(pid: int) -> str | None:
        """The name that a process carries in the process table, or `None`.

        A process names itself with `prctl`, as the workers of `core` do, and the interpreter's own
        name is the default.
        """
        text = _read(f"/proc/{pid}/comm")
        return None if text is None else text.strip()

elif sys.platform == "win32":
    import ctypes
    import functools
    from ctypes import wintypes

    _PROCESS_QUERY_INFORMATION = 0x0400
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _PROCESS_VM_READ = 0x0010
    _TH32CS_SNAPPROCESS = 0x00000002
    _INVALID_HANDLE = ctypes.c_void_p(-1).value

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

    class _ProcessEntry(ctypes.Structure):
        """`PROCESSENTRY32W` of `tlhelp32.h`."""

        _fields_ = (
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        )

    @functools.cache
    def _api() -> tuple[ctypes.WinDLL, ctypes.WinDLL]:
        # Private copies of the libraries, so that the argument types stay ours.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.GetProcessTimes.argtypes = (
            wintypes.HANDLE,
            *([ctypes.POINTER(wintypes.FILETIME)] * 4),
        )
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.QueryFullProcessImageNameW.argtypes = (
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        )
        kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        kernel32.CreateToolhelp32Snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
        kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        for name in ("Process32FirstW", "Process32NextW"):
            function = getattr(kernel32, name)
            function.argtypes = (wintypes.HANDLE, ctypes.POINTER(_ProcessEntry))
            function.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(_ProcessMemoryCounters),
            wintypes.DWORD,
        )
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        return kernel32, psapi

    def _ticks(value: wintypes.FILETIME) -> int:
        return (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)

    def _open(pid: int, access: int) -> int | None:
        kernel32, _ = _api()
        handle = kernel32.OpenProcess(access, False, pid)
        return int(handle) if handle else None

    def read_process(pid: int) -> ProcessReading | None:
        """Read the CPU time and the memory of a process. Returns `None` when it is gone."""
        kernel32, psapi = _api()
        handle = _open(pid, _PROCESS_QUERY_INFORMATION | _PROCESS_VM_READ)
        if handle is None:
            handle = _open(pid, _PROCESS_QUERY_LIMITED_INFORMATION)
        if handle is None:
            return None
        try:
            created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
            cpu: int | None = None
            if kernel32.GetProcessTimes(
                handle,
                ctypes.byref(created),
                ctypes.byref(exited),
                ctypes.byref(kernel),
                ctypes.byref(user),
            ):
                cpu = (_ticks(kernel) + _ticks(user)) * 100  # units of 100 ns
            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            peak: int | None = None
            current: int | None = None
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                peak, current = int(counters.PeakWorkingSetSize), int(counters.WorkingSetSize)
            if cpu is None and peak is None:
                return None
            return ProcessReading(pid, cpu, peak, current)
        finally:
            kernel32.CloseHandle(handle)

    def thread_cpu_ns(pid: int) -> dict[int, int]:
        """The CPU time of each thread of a process. Windows gives no such reading here."""
        return {}

    def parent_map() -> dict[int, int]:
        """The parent of every process, by process ID."""
        kernel32, _ = _api()
        snapshot = kernel32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
        if not snapshot or int(snapshot) == _INVALID_HANDLE:
            return {}
        parents: dict[int, int] = {}
        try:
            entry = _ProcessEntry()
            entry.dwSize = ctypes.sizeof(entry)
            more = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
            while more:
                parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
                more = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
        finally:
            kernel32.CloseHandle(snapshot)
        return parents

    def command_line(pid: int) -> str | None:
        """The command line of a process. Windows gives none without a private structure."""
        return None

    def image_path(pid: int) -> str | None:
        """The path of the program that a process runs, or `None`."""
        kernel32, _ = _api()
        handle = _open(pid, _PROCESS_QUERY_LIMITED_INFORMATION)
        if handle is None:
            return None
        try:
            size = wintypes.DWORD(1024)
            buffer = ctypes.create_unicode_buffer(size.value)
            if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
                return None
            return buffer.value
        finally:
            kernel32.CloseHandle(handle)

    def process_name(pid: int) -> str | None:
        """The name that a process gives itself. Windows has no such name."""
        return None

else:

    def read_process(pid: int) -> ProcessReading | None:
        """Read the CPU time and the memory of a process. This system gives none."""
        return None

    def thread_cpu_ns(pid: int) -> dict[int, int]:
        """The CPU time of each thread of a process. This system gives none."""
        return {}

    def parent_map() -> dict[int, int]:
        """The parent of every process. This system gives none."""
        return {}

    def command_line(pid: int) -> str | None:
        """The command line of a process. This system gives none."""
        return None

    def image_path(pid: int) -> str | None:
        """The path of the program that a process runs. This system gives none."""
        return None

    def process_name(pid: int) -> str | None:
        """The name that a process gives itself. This system gives none."""
        return None


def children_of(pid: int, parents: Mapping[int, int]) -> list[int]:
    """The processes whose parent is `pid`, in the order of their IDs."""
    return sorted(child for child, parent in parents.items() if parent == pid and child != pid)


def descendants_of(pid: int, parents: Mapping[int, int]) -> list[int]:
    """Every process below `pid`, at any depth, in the order of their IDs."""
    found: list[int] = []
    todo = [pid]
    while todo:
        for child in children_of(todo.pop(), parents):
            if child not in found:
                found.append(child)
                todo.append(child)
    return sorted(found)


def is_venv_launcher(pid: int) -> bool:
    """Whether a process is the `python.exe` of a virtual environment on Windows.

    That program starts the interpreter of the base installation as a child and waits for it. It
    sits in the `Scripts` folder of an environment, next to a `pyvenv.cfg` one level up. Other
    systems start the interpreter in the same process, and they keep it in a `bin` folder, so the
    answer is no there.
    """
    image = image_path(pid)
    if image is None:
        return False
    folder = Path(image).parent
    return folder.name.lower() == "scripts" and (folder.parent / "pyvenv.cfg").is_file()


def real_process(pid: int, parents: Mapping[int, int]) -> int:
    """The process that does the work of `pid`: the child of a virtual-environment launcher.

    A launcher has one child, which may be a launcher itself. Any other process is its own answer.
    """
    current = pid
    while is_venv_launcher(current):
        below = children_of(current, parents)
        if len(below) != 1:
            break
        current = below[0]
    return current
