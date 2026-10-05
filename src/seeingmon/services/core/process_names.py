"""The names that the worker processes of `core` give themselves.

`core` runs two worker processes: the survey worker, and the alignment worker for the quick solve.
Both start with the `spawn` method of `multiprocessing`, so they have the same command line, and
the same interpreter runs them. Nothing outside tells them apart, and in `ps` and `top` both show
the name of the interpreter.

On Linux a process can name itself (`prctl` with `PR_SET_NAME`), and each worker does so when it
starts. The name shows in `ps -o comm`, in `top`, and in `/proc/<pid>/comm`, and the performance
tooling (`seeingmon.perf.sysrun`) reads it to give each worker its own role. The kernel keeps 15
bytes of a name. Windows has no such name, so a worker keeps its own name there, and the tooling
takes every worker for the survey worker.

The module imports nothing heavy, so that the performance tooling and the worker initializers can
use it freely.
"""

from __future__ import annotations

import sys
from typing import Any

SURVEY_WORKER_NAME = "smon-survey"
ALIGNMENT_WORKER_NAME = "smon-align"

PR_SET_NAME = 15  # the option of `prctl` that names the calling thread
MAX_NAME_BYTES = 15  # what the kernel keeps of a name, without the terminator


def prctl_name(libc: Any, name: str) -> str:
    """Name the calling thread on a `libc` object that offers `prctl`.

    The main thread of a process carries the name of the process, so call it from there. Returns
    what happened, in words, and never raises for a refused call.
    """
    encoded = name.encode("ascii", "replace")[:MAX_NAME_BYTES]
    if libc.prctl(PR_SET_NAME, encoded, 0, 0, 0) == 0:
        return f"process name {encoded.decode('ascii')}"
    return "the call failed, so the process keeps its name"


if sys.platform == "linux":

    def name_process(name: str) -> str:
        """Name this process, as `ps` and `top` show it. Returns what happened, in words."""
        import ctypes

        try:
            libc = ctypes.CDLL(None, use_errno=True)  # a private handle, so the types stay ours
            libc.prctl.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_ulong,
                ctypes.c_ulong,
                ctypes.c_ulong,
            ]
            libc.prctl.restype = ctypes.c_int
            return prctl_name(libc, name)
        except (OSError, AttributeError) as error:
            return f"the call raised {type(error).__name__}, so the process keeps its name"

else:

    def name_process(name: str) -> str:
        """Name this process. This platform has no such name, so nothing changes."""
        return "not supported on this platform"
