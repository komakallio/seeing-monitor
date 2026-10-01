"""The survey analysis in a worker process of its own, at a low priority.

Detection, plate solving, and photometry need hundreds of megabytes and a whole core for a few
seconds. They run in one worker process (`make_survey_executor`), so that a spike in that process
cannot disturb the scheduler, and so that the kernel can take it first:

- **Priority.** The worker lowers its own priority before it builds the pipeline. On Linux it sets
  the niceness of `[services.core.survey_worker] nice`. On Windows it selects the below-normal
  priority class.
- **Out-of-memory killer.** On Linux the worker raises its `oom_score_adj`, so that the kernel takes
  the worker before it takes `core` or `acquire`. Raising the score needs no privilege.

The worker process starts with the `spawn` method (see `seeingmon.survey.analyzer`), so it holds no
state of `core`. Only plain data crosses the boundary: the pipeline specification, and the jobs and
results of the analyzer.

The functions here never raise for a platform that cannot do a step. They return a sentence that
says what happened, and the worker logs it.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import sys
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from seeingmon.services.core.settings import SurveyWorkerSettings
from seeingmon.survey.analyzer import InlineExecutor, init_worker
from seeingmon.survey.pipeline import PipelineSpec

_log = logging.getLogger(__name__)

BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
IDLE_PRIORITY_CLASS = 0x00000040

if sys.platform == "win32":

    def lower_process_priority(nice: int) -> str:
        """Lower the priority of this process. Returns what happened, in words."""
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32")  # a private copy, so the argument types stay ours
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.SetPriorityClass.restype = ctypes.c_int
        wanted = IDLE_PRIORITY_CLASS if nice >= 15 else BELOW_NORMAL_PRIORITY_CLASS
        if kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), wanted):
            return "idle priority class" if wanted == IDLE_PRIORITY_CLASS else "below-normal class"
        return "not permitted, so the process keeps its priority"

elif hasattr(os, "setpriority"):

    def lower_process_priority(nice: int) -> str:
        """Lower the priority of this process. Returns what happened, in words."""
        try:
            os.setpriority(os.PRIO_PROCESS, 0, nice)
        except OSError:
            return "not permitted, so the process keeps its priority"
        return f"nice {nice}"

else:

    def lower_process_priority(nice: int) -> str:
        """Lower the priority of this process. Returns what happened, in words."""
        return "not supported on this platform"


def raise_oom_score(value: int, proc_dir: Path | str = "/proc/self") -> str:
    """Raise `oom_score_adj`, so that the kernel takes this process first when memory runs out.

    The file exists on Linux only. Raising the value (making the process more likely to be killed)
    needs no privilege.
    """
    path = Path(proc_dir) / "oom_score_adj"
    try:
        path.write_text(str(value), encoding="ascii")
    except OSError:
        return "not supported on this platform"
    return f"oom_score_adj {value}"


def init_survey_worker(spec: PipelineSpec, nice: int, oom_score_adj: int) -> None:
    """The initializer of the worker: lower the priority, raise the OOM score, build the pipeline.

    The order matters: the priority applies before the catalog loads, so the slow start does not
    compete with `core`. The function must live at the top level of a module, so that the `spawn`
    method can import it in the new process.
    """
    _log.info("the survey worker runs at %s", lower_process_priority(nice))
    _log.info("the survey worker has %s", raise_oom_score(oom_score_adj))
    init_worker(spec)


def make_survey_executor(spec: PipelineSpec, settings: SurveyWorkerSettings) -> Executor:
    """The executor for the survey analyzer, as the settings choose it.

    `process` makes one worker process (it starts at the first job). `thread` makes one thread in
    this process, and `inline` runs each job in the thread that submits it.
    """
    if settings.mode == "inline":
        return InlineExecutor()
    if settings.mode == "thread":
        return ThreadPoolExecutor(max_workers=1, thread_name_prefix="survey")
    return ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_survey_worker,
        initargs=(spec, settings.nice, settings.oom_score_adj),
    )
