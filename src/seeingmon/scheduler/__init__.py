"""The scheduler: one owner of the camera that shares its time between the measurement modes.

See the "Scheduler" section of `docs/architecture.md`. The modules are:

- `scheduler`: the `Scheduler` class and `build_scheduler`. Start here.
- `commands`: `submit` takes these commands and returns a `CommandResult`.
- `machine`: the states and the legal transitions.
- `config`: the `[scheduler]` table and the observing site.
- `commission`: the task queue, the handler protocol, and the sweep.
- `faults` and `levels`: the response to camera errors and the recovery ladder.
- `gates`: the daylight gate, the twilight flag, and the cloud tracker.
- `ephemeris`: the Sun's elevation, for the search limit and the twilight flag.
- `status`: the snapshot for `/status` and the `health` record, with the activity.
- `activity`: the words of the activity: its labels, details, and reasons.
- `events`: the codes of the events that the scheduler writes.
"""

from __future__ import annotations

from seeingmon.scheduler.commands import (
    CancelTask,
    Command,
    CommandResult,
    Pause,
    QueueBurst,
    QueueDark,
    QueueFlat,
    QueueReplay,
    QueueSweep,
    RejectReason,
    Resume,
    StartAlignment,
    StartRapidFocus,
    StopAlignment,
    StopRapidFocus,
)
from seeingmon.scheduler.commission import (
    CommissionContext,
    CommissionHandler,
    CommissionResult,
    CommissionTask,
    FastWindowSample,
    SweepHandler,
    format_sweep_table,
)
from seeingmon.scheduler.config import SchedulerConfig, SiteConfig, load_site
from seeingmon.scheduler.ephemeris import sun_elevation_deg
from seeingmon.scheduler.levels import EscalationLevel
from seeingmon.scheduler.machine import State
from seeingmon.scheduler.scheduler import Scheduler, StepKind, build_scheduler
from seeingmon.scheduler.status import SchedulerStatus

__all__ = [
    "CancelTask",
    "Command",
    "CommandResult",
    "CommissionContext",
    "CommissionHandler",
    "CommissionResult",
    "CommissionTask",
    "EscalationLevel",
    "FastWindowSample",
    "Pause",
    "QueueBurst",
    "QueueDark",
    "QueueFlat",
    "QueueReplay",
    "QueueSweep",
    "RejectReason",
    "Resume",
    "Scheduler",
    "SchedulerConfig",
    "SchedulerStatus",
    "SiteConfig",
    "StartAlignment",
    "StartRapidFocus",
    "State",
    "StepKind",
    "StopAlignment",
    "StopRapidFocus",
    "SweepHandler",
    "build_scheduler",
    "format_sweep_table",
    "load_site",
    "sun_elevation_deg",
]
