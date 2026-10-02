"""The commands that the REST API, the UI, and the CLI send to the scheduler.

Call `Scheduler.submit(command)` from any thread. It returns a `CommandResult` at once: the
scheduler accepted the command, or it names a reason for the rejection. The caller never waits
for the camera. The scheduler writes an `event` record for every command, accepted or not.

A command is a frozen dataclass with plain fields, so a web handler can build one from a
request body. The queue commands (`QueueBurst`, `QueueSweep`, `QueueReplay`, and `QueueDark`)
share a `priority`: a higher number runs first, and equal priorities run in the order of arrival.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from seeingmon.frames import StreamConfig


class RejectReason(StrEnum):
    """Why the scheduler rejected a command. The web layer can map each code to a response."""

    PAUSED = "paused"  # the scheduler is paused, so it cannot start the alignment
    ALREADY_PAUSED = "already_paused"
    NOT_PAUSED = "not_paused"
    NOT_ALIGNING = "not_aligning"
    DEGRADED = "degraded"  # the camera has failed repeatedly, so alignment cannot start
    NO_HANDLER = "no_handler"  # nothing is registered to run this kind of task
    QUEUE_FULL = "queue_full"
    BUSY = "busy"  # a task of this kind waits or runs, and only one may (a dark session)
    INVALID = "invalid"  # a field of the command is out of range
    CLOSED = "closed"  # the scheduler has shut down


class Command:
    """The base class of every command. It carries no fields."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class StartAlignment(Command):
    """Start the alignment stream, or keep it alive when it already runs.

    The command preempts every other mode. Send it again while the stream runs to restart the
    idle timer and to change the settings. Leave a field `None` to use the configured value.
    """

    exposure_s: float | None = None
    gain: int | None = None


@dataclass(frozen=True, slots=True)
class StopAlignment(Command):
    """End the alignment stream. The scheduler goes back to `safe` and re-checks the sky."""


@dataclass(frozen=True, slots=True)
class Pause(Command):
    """Stop everything. The camera stays open, and nothing runs until `Resume`."""


@dataclass(frozen=True, slots=True)
class Resume(Command):
    """Leave `paused`. The scheduler enters `safe` and goes to `auto` when the sky allows it."""


@dataclass(frozen=True, slots=True)
class QueueBurst(Command):
    """Queue a burst: record raw frames for `duration_s` seconds, using the registered handler.

    `stream` holds the settings. `None` means the fast stream with the ROI on Polaris.
    """

    duration_s: float = 10.0
    stream: StreamConfig | None = None
    label: str = ""
    priority: int = 0


@dataclass(frozen=True, slots=True)
class QueueSweep(Command):
    """Queue a sweep: a short fast window for every cell of a grid of settings.

    An empty axis uses the default of `[scheduler.sweep]`. The grid is the product of the four
    axes. `window_s` is the length of each cell. `None` uses the configured length.
    """

    exposure_us: tuple[int, ...] = ()
    gain: tuple[int, ...] = ()
    roi_arcmin: tuple[float, ...] = ()
    modes: tuple[str, ...] = ()
    window_s: float | None = None
    priority: int = 0


@dataclass(frozen=True, slots=True)
class QueueReplay(Command):
    """Queue a replay of a recording through the production analysis, using the handler.

    `source` names the recording, and `speed` is the replay speed factor (0 means as fast as
    possible). `options` carries anything else that the handler understands.
    """

    source: str = ""
    speed: float = 1.0
    options: Mapping[str, Any] = field(default_factory=dict)
    priority: int = 0


@dataclass(frozen=True, slots=True)
class QueueDark(Command):
    """Queue a dark session: record bias and dark frames with the camera covered, and add a set
    to the dark library, using the registered handler.

    Leave a field `None` to use the configured value (`[survey.dark]`). With `wait_for_cover`, the
    session waits until short test frames are dark, which means that you covered the camera.
    Without it, the first frame that is not dark ends the task as failed. `pause_after` pauses
    the scheduler when the queue is empty again, so that nothing records data while the camera is
    still covered, and `Resume` continues. `label` is a note for the event.
    """

    exposure_s: float | None = None
    frames: int | None = None
    bias_frames: int | None = None
    wait_for_cover: bool = True
    pause_after: bool = True
    label: str = ""
    priority: int = 0


QUEUE_COMMANDS = (QueueBurst, QueueSweep, QueueReplay, QueueDark)
TASK_KINDS: Mapping[type[Command], str] = {
    QueueBurst: "burst",
    QueueSweep: "sweep",
    QueueReplay: "replay",
    QueueDark: "dark",
}


@dataclass(frozen=True, slots=True)
class CommandResult:
    """The outcome of `Scheduler.submit`.

    `accepted` says whether the scheduler took the command. A rejection sets `reason`. `message`
    is one plain sentence for the operator. `task_id` identifies a queued task, and `state` is
    the scheduler's state right after the command.
    """

    accepted: bool
    message: str
    state: str
    reason: RejectReason | None = None
    task_id: int | None = None
