"""The commands that the REST API, the UI, and the CLI send to the scheduler.

Call `Scheduler.submit(command)` from any thread. It returns a `CommandResult` at once: the
scheduler accepted the command, or it names a reason for the rejection. The caller never waits
for the camera. The scheduler writes an `event` record for every command, accepted or not.

A command is a frozen dataclass with plain fields, so a web handler can build one from a
request body. The queue commands (`QueueBurst`, `QueueSweep`, `QueueReplay`, `QueueDark`, and
`QueueFlat`) share a `priority`: a higher number runs first, and equal priorities run in the order
of arrival. `CancelTask` removes the queued tasks of one kind and stops the running one.
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
    BUSY = "busy"  # a task of this kind waits or runs, and only one may (a dark or flat session)
    INVALID = "invalid"  # a field of the command is out of range
    CLOSED = "closed"  # the scheduler has shut down
    NO_TASK = "no_task"  # no task of the kind waits or runs, so there is nothing to cancel


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
    session waits until short test frames are dark, which means that you covered the camera, and
    it gives up after `wait_for_cover_timeout_s` seconds. Without `wait_for_cover`, the first
    frame that is not dark ends the task as failed. `pause_after` pauses the scheduler when the
    queue is empty again, so that nothing records data while the camera is still covered, and
    `Resume` continues. `label` is a note for the event.

    With `immediate`, the task does not wait for the cycle boundary, because someone stands at the
    camera with the cover on: the next step of the scheduler ends the fast stream, which flushes
    its window as a partial one, or skips what is left of the survey step, and the task starts.
    An exposure in progress finishes first. Without `immediate`, the task waits for the boundary,
    as the other kinds of task do.
    """

    exposure_s: float | None = None
    frames: int | None = None
    bias_frames: int | None = None
    wait_for_cover: bool = True
    pause_after: bool = True
    label: str = ""
    priority: int = 0
    wait_for_cover_timeout_s: float | None = None
    immediate: bool = True


# The limits of a flat session. The numbers are the ones of the web UI, and the scheduler, the
# web layer, and the page all hold a command to them.
MIN_FLAT_FRAMES = 8
MAX_FLAT_FRAMES = 64
MIN_FLAT_TARGET = 0.3
MAX_FLAT_TARGET = 0.7


@dataclass(frozen=True, slots=True)
class QueueFlat(Command):
    """Queue a flat session: take frames of a light source over the aperture, combine them into a
    flat, and add the flat to the flat library as a pending flat, using the registered handler.

    `frames` is the number of frames of the set (8 to 64), and `target_fraction` is the level that
    the session aims at, as a fraction of the full scale (0.3 to 0.7). With `set_number` 2, the
    source has turned by 180 degrees since the first set, and the session combines the new frames
    with the first set of the same session (the handler keeps it for 24 hours). `pause_after`
    pauses the scheduler when the queue is empty again, so that nothing records data while the
    light source still covers the camera, and `Resume` continues.

    With `immediate`, the task does not wait for the cycle boundary, because someone holds the
    light source at the camera. The next step of the scheduler ends the fast stream or skips what
    is left of the survey step, and the task starts. An exposure in progress finishes first.
    """

    frames: int = 32
    target_fraction: float = 0.5
    set_number: int = 1
    pause_after: bool = True
    priority: int = 0
    immediate: bool = True


@dataclass(frozen=True, slots=True)
class CancelTask(Command):
    """Cancel the tasks of one kind: remove the ones that wait, and stop the one that runs.

    `kind` is the kind of a task, such as `flat`. A task that waits never starts. A running task
    ends early with the status `aborted` when its handler next asks `should_stop`. The scheduler
    rejects the command with `no_task` when no task of the kind waits or runs.
    """

    kind: str = ""


QUEUE_COMMANDS = (QueueBurst, QueueSweep, QueueReplay, QueueDark, QueueFlat)
TASK_KINDS: Mapping[type[Command], str] = {
    QueueBurst: "burst",
    QueueSweep: "sweep",
    QueueReplay: "replay",
    QueueDark: "dark",
    QueueFlat: "flat",
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
