"""Raise the priority of the capture thread, where the platform allows it.

The capture thread waits in the SDK most of the time, and it must wake up promptly when a frame
arrives, because the arrival time is the basis of the frame time. A higher priority keeps other
work, such as the sender and the analysis in `core`, from delaying the wake-up.

`raise_current_thread_priority` tries, and it never raises: without the privilege, the thread
keeps its normal priority, and the function says so. A systemd unit can grant the privilege
(`AmbientCapabilities=CAP_SYS_NICE` on Linux). On Windows, the highest priority within the normal
class needs no privilege.
"""

from __future__ import annotations

import logging
import os
import sys
import threading

_log = logging.getLogger(__name__)

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

        kernel32 = ctypes.WinDLL("kernel32")  # a private copy, so the argument types stay ours
        kernel32.GetCurrentThread.restype = ctypes.c_void_p
        kernel32.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
        kernel32.SetThreadPriority.restype = ctypes.c_int
        highest = 2  # THREAD_PRIORITY_HIGHEST
        if kernel32.SetThreadPriority(kernel32.GetCurrentThread(), highest):
            return "highest thread priority"
        return "not permitted, so the thread keeps its normal priority"

else:

    def raise_current_thread_priority() -> str:
        """Raise the priority of the calling thread. Returns what happened, in words."""
        return "not supported on this platform"
