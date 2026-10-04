"""The dark handler: record a dark set with the camera that the scheduler lends, from the web UI.

The owner covers the camera by hand, opens the Dark page, and presses Start. The page sends a
`QueueDark` command, and the scheduler runs this handler in its `commission` state, with the
camera to itself. The session is the one of `seeingmon dark` (`seeingmon.survey.dark_session`),
run on a borrowed camera:

1. take bias frames at the shortest exposure,
2. wait until short test frames are dark, which proves that the camera is covered (with
   `wait_for_cover` off, the first frame that is not dark ends the task as failed),
3. take dark frames at the survey exposure, and
4. build the master dark, find the hot pixels, and add the set to the dark library.

The command's own fields (exposure, frame counts, `wait_for_cover`) win over the defaults of
`[survey.dark]`, and the mode, the gain, and the check limits always come from that section.
`pause_after` belongs to the scheduler: when the queue is empty again, it pauses, so that nothing
records data while the camera is still covered, and `Resume` continues.

**What each part does.**

- `ContextCamera` lends the scheduler's camera to the session. A frame is a `start`, a
  `read_frame`, and a `stop` of the context, and the sensor temperature comes from the frames
  (the context has no other way to read it, so a camera without one fails the task).
- `DarkHandler` runs the task and returns a `CommissionResult`: `ok`, `failed` (with one plain
  sentence), or `aborted` (a command took the camera, and the library stays as it was). A
  `CameraError` goes on to the scheduler, which runs its recovery.
- `DarkTaskState` is the small, thread-safe record that the RPC `dark_library` reads: `queued`
  from the moment that `core` accepts the command, `running` with the phase, the step, and the
  latest check of the cover, then `ok`, `failed`, or `aborted`.
- `DarkLibraryReader` builds the answer of the RPC (`DarkLibraryView`): the sets of the library,
  its status, the model, the sensor temperature, and the state of the task.

A set that the task adds serves the very next survey frame. The pipeline reads the folder of the
library for every frame (it keeps no list of sets), so no restart is needed.

**Events.** The handler writes one `scheduler.dark_phase` event at each change of phase, and none
for each frame. The scheduler writes the start and the result (`scheduler.dark_result`).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from seeingmon.clock import Clock, utc_ns_to_iso
from seeingmon.drivers.base import CameraConfigError, CameraError
from seeingmon.frames import Frame, StreamConfig
from seeingmon.profile import Profile, ProfileError
from seeingmon.scheduler.commands import QueueDark
from seeingmon.scheduler.commission import (
    CommissionContext,
    CommissionResult,
    CommissionTask,
)
from seeingmon.scheduler.events import DARK_PHASE_EVENT
from seeingmon.services.web.contract import (
    MAX_LIST_ITEMS,
    DarkLibraryView,
    DarkModelView,
    DarkSetView,
    DarkStatusView,
    DarkTaskView,
)
from seeingmon.store.layout import DataLayout
from seeingmon.survey.config import DarkConfig
from seeingmon.survey.dark import (
    SECONDS_PER_DAY,
    DarkError,
    DarkLibrary,
    DarkSet,
    dark_status,
)
from seeingmon.survey.dark_session import (
    DarkAborted,
    DarkProgress,
    DarkSessionOptions,
    DarkSessionResult,
    record_dark_set,
)

_log = logging.getLogger(__name__)

PHASE_EVENT = DARK_PHASE_EVENT
READ_MARGIN_S = 30.0  # a read waits this long beyond the exposure
TaskState = Literal["idle", "queued", "running", "ok", "failed", "aborted"]
WAITING_MESSAGE = "The dark session waits for the scheduler to start it."
# A task starts in `safe` or `auto` only, so a session that you queue in another state waits.
WAITING_MESSAGES = {
    "paused": "The scheduler is paused. The dark session starts after you resume it.",
    "align": "The alignment helper runs. The dark session starts after it ends.",
}


def _sentence(text: str) -> str:
    """The text as one sentence: it starts with a capital letter and ends with a period."""
    text = text.strip()
    if not text:
        return text
    return (text[0].upper() + text[1:]).rstrip(".") + "."


class ContextCamera:
    """The camera of a `CommissionContext` as a `DarkCamera`.

    Every frame is a snapshot: the context starts the stream, reads the frame, and stops it. The
    context never gives the sensor temperature apart from a frame, so `read_temperature_c` has
    nothing to answer, and a camera whose frames carry none cannot make a set.
    """

    def __init__(self, context: CommissionContext) -> None:
        self._context = context

    def configure(self, config: StreamConfig) -> None:
        self._context.configure(config)

    def take(self, exposure_s: float) -> Frame:
        self._context.start()
        try:
            return self._context.read_frame(exposure_s + READ_MARGIN_S)
        finally:
            self._context.stop()

    def read_temperature_c(self) -> float | None:
        return None


# --- The state that the RPC reads ---------------------------------------------------------------


class DarkTaskState:
    """The latest dark session of this process, for the RPC `dark_library`. Thread-safe.

    The state starts `idle`. `core` calls `queued` when the scheduler accepts a `QueueDark`, the
    handler calls `started`, `progress`, and `finished`, and the RPC calls `snapshot`. A call to
    `queued` for a task that has begun already changes nothing, because the scheduler can start
    the task before the call that accepted it returns to `core`.
    """

    def __init__(self, clock: Clock, defaults: DarkConfig) -> None:
        self._clock = clock
        self._defaults = defaults
        self._lock = threading.Lock()
        self._view = DarkTaskView()

    def _now_iso(self) -> str:
        return utc_ns_to_iso(self._clock.utc_ns(), digits=0)

    def queued(self, task_id: int, command: QueueDark, scheduler_state: str = "auto") -> None:
        """The scheduler accepted a dark task. `scheduler_state` is its state at that moment."""
        with self._lock:
            if self._view.task_id == task_id and self._view.state != "idle":
                return
            self._view = DarkTaskView(
                state="queued",
                task_id=task_id,
                message=WAITING_MESSAGES.get(scheduler_state, WAITING_MESSAGE),
                exposure_s=(
                    self._defaults.exposure_s if command.exposure_s is None else command.exposure_s
                ),
                frames=self._defaults.frames if command.frames is None else command.frames,
                bias_frames=(
                    self._defaults.bias_frames
                    if command.bias_frames is None
                    else command.bias_frames
                ),
                wait_for_cover=command.wait_for_cover,
                pause_after=command.pause_after,
            )

    def started(self, task_id: int, command: QueueDark, options: DarkSessionOptions | None) -> None:
        """The handler began the task. `options` are the settings in effect, or `None` when the
        command asked for settings that the session refuses."""
        with self._lock:
            changes: dict[str, object] = {
                "state": "running",
                "task_id": task_id,
                "message": "The dark session started.",
                "wait_for_cover": command.wait_for_cover,
                "pause_after": command.pause_after,
                "started_utc": self._now_iso(),
            }
            if options is not None:
                changes.update(
                    exposure_s=options.exposure_s,
                    frames=options.frames,
                    bias_frames=options.bias_frames,
                    wait_for_cover=options.wait,
                )
            self._view = DarkTaskView().model_copy(update=changes)

    def progress(self, progress: DarkProgress) -> None:
        """The session moved on. A frame's check updates `covered`, `level_dn`, and `reason`."""
        with self._lock:
            changes: dict[str, object] = {
                "phase": progress.phase,
                "step": progress.step,
                "steps": progress.steps,
                "message": progress.message,
            }
            if progress.check is not None:
                changes.update(
                    covered=progress.check.ok,
                    level_dn=float(progress.check.level_dn),
                    reason=progress.check.reason,
                )
            self._view = self._view.model_copy(update=changes)

    def finished(
        self,
        status: Literal["ok", "failed", "aborted"],
        summary: str,
        set_name: str | None = None,
    ) -> None:
        """The task ended."""
        with self._lock:
            self._view = self._view.model_copy(
                update={
                    "state": status,
                    "phase": None,
                    "message": "",
                    "summary": summary,
                    "set_name": set_name,
                    "finished_utc": self._now_iso(),
                }
            )

    def snapshot(self) -> DarkTaskView:
        """The state now, as the view that the RPC sends."""
        with self._lock:
            return self._view


class _PhaseReporter:
    """The progress callback of one task: it keeps the state and writes an event for each phase."""

    def __init__(self, context: CommissionContext, state: DarkTaskState, task_id: int) -> None:
        self._context = context
        self._state = state
        self._task_id = task_id
        self._phase: str | None = None

    def __call__(self, progress: DarkProgress) -> None:
        self._state.progress(progress)
        if progress.phase == self._phase or progress.phase == "done":
            return
        self._phase = progress.phase
        self._context.emit_event(
            "info",
            PHASE_EVENT,
            progress.message,
            {"task_id": self._task_id, "phase": progress.phase, "steps": progress.steps},
        )


# --- The handler --------------------------------------------------------------------------------


class DarkHandler:
    """Runs `dark` tasks. Register it with `Scheduler.register_handler("dark", handler)`."""

    def __init__(
        self,
        *,
        library: DarkLibrary,
        layout: DataLayout,
        profile: Profile,
        clock: Clock,
        config: DarkConfig,
        state: DarkTaskState,
    ) -> None:
        self._library = library
        self._layout = layout
        self._profile = profile
        self._clock = clock
        self._config = config
        self._state = state

    def _result(
        self,
        task: CommissionTask,
        started_ns: int,
        status: Literal["ok", "failed", "aborted"],
        summary: str,
        data: dict[str, object] | None = None,
        artifacts: tuple[str, ...] = (),
    ) -> CommissionResult:
        return CommissionResult(
            task_id=task.task_id,
            kind=task.kind,
            status=status,
            summary=summary,
            started_utc_ns=started_ns,
            finished_utc_ns=self._clock.utc_ns(),
            data=data or {},
            artifacts=artifacts,
            pinned=status == "ok",
        )

    def run(self, task: CommissionTask, context: CommissionContext) -> CommissionResult:
        started_ns = self._clock.utc_ns()
        command = task.command
        if not isinstance(command, QueueDark):
            return self._result(task, started_ns, "failed", "The task holds no dark command.")
        try:
            options = DarkSessionOptions.from_config(
                self._config,
                exposure_s=command.exposure_s,
                frames=command.frames,
                bias_frames=command.bias_frames,
                wait=command.wait_for_cover,
                wait_timeout_s=command.wait_for_cover_timeout_s,
            )
            self._profile.mode(options.mode)
        except (ValueError, ProfileError) as error:
            summary = _sentence(f"The dark settings are not valid: {error}")
            self._state.started(task.task_id, command, None)
            self._state.finished("failed", summary)
            return self._result(task, started_ns, "failed", summary)
        self._state.started(task.task_id, command, options)
        finished = False
        try:
            outcome = self._record(task, command, options, context)
            finished = True
        except CameraError as error:
            # The scheduler runs the recovery. The state says what happened to the task.
            self._state.finished("failed", _sentence(f"The camera failed: {error}"))
            finished = True
            raise
        finally:
            if not finished:
                self._state.finished("failed", "The dark session ended unexpectedly.")
        status, summary, data, artifacts, set_name = outcome
        self._state.finished(status, summary, set_name)
        return self._result(task, started_ns, status, summary, data, artifacts)

    def _record(
        self,
        task: CommissionTask,
        command: QueueDark,
        options: DarkSessionOptions,
        context: CommissionContext,
    ) -> tuple[
        Literal["ok", "failed", "aborted"],
        str,
        dict[str, object],
        tuple[str, ...],
        str | None,
    ]:
        """Run the session. Returns the status, the summary, the data, the files, and the set."""
        uncover: dict[str, object] = {"remove_cover": True}  # the UI tells the owner to uncover
        try:
            session = record_dark_set(
                ContextCamera(context),
                self._library,
                self._profile,
                context.clock,
                options,
                say=_log.debug,
                progress=_PhaseReporter(context, self._state, task.task_id),
                should_stop=context.should_stop,
            )
        except DarkAborted:
            summary = (
                "The dark session stopped early because another command took the camera, "
                "and the library is unchanged."
            )
            return "aborted", summary, uncover, (), None
        except DarkError as error:
            return "failed", _sentence(f"{error}. The library is unchanged"), uncover, (), None
        except CameraConfigError as error:
            summary = _sentence(f"The camera refused the dark settings: {error}")
            return "failed", summary, uncover, (), None
        except OSError as error:
            summary = f"The dark set could not be written ({type(error).__name__})."
            return "failed", summary, uncover, (), None
        dark_set = session.dark_set
        rate = dark_set.rate_dn_per_s * self._profile.e_per_adu(options.mode, options.gain)
        summary = (
            f"Added {dark_set.name} to the dark library: {rate:.2f} e-/s per pixel at "
            f"{dark_set.temperature_c:.1f} C, and {dark_set.n_hot_pixels} hot pixels."
        )
        data = self._data(session, options, rate)
        data.update(uncover)
        return "ok", summary, data, (self._artifact(dark_set),), dark_set.name

    def _artifact(self, dark_set: DarkSet) -> str:
        """The file of the set, relative to the data directory when the library lives there."""
        path = self._library.directory / dark_set.name
        try:
            return self._layout.relative(path)
        except ValueError:
            return dark_set.name

    @staticmethod
    def _data(
        session: DarkSessionResult, options: DarkSessionOptions, rate_e_per_s: float
    ) -> dict[str, object]:
        dark_set, model, status = session.dark_set, session.model, session.status
        return {
            "set_name": dark_set.name,
            "temperature_c": dark_set.temperature_c,
            "rate_e_per_s": rate_e_per_s,
            "hot_pixels": dark_set.n_hot_pixels,
            "exposure_s": options.exposure_s,
            "frames": dark_set.n_frames,
            "bias_frames": dark_set.n_bias_frames,
            "waited_s": session.waited_s,
            "n_sets": session.n_sets,
            "model": None
            if model is None
            else {
                "reference_c": model.reference_c,
                "doubling_c": model.doubling_c,
                "doubling_fitted": model.doubling_fitted,
                "rms_log2": model.rms_log2,
                "n_sets": model.n_sets,
            },
            "due": status.due,
            "due_reason": status.reason,
        }


# --- The view for the RPC -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DarkLibraryReader:
    """Builds the answer of the RPC `dark_library`: the library, its status, and the latest task.

    `temperature_c` gives the sensor temperature from the scheduler's status, and `None` when no
    frame has come yet. The sets are all that the library holds, newest first. The status and the
    model use the survey's own settings (`[survey.dark]`: mode, gain, and the limits).
    """

    library: DarkLibrary
    profile: Profile
    clock: Clock
    config: DarkConfig
    state: DarkTaskState
    temperature_c: Callable[[], float | None]

    def view(self) -> DarkLibraryView:
        cfg = self.config
        now_ns = self.clock.utc_ns()
        temperature = self.temperature_c()
        status = dark_status(
            self.library,
            temperature,
            now_ns,
            mode=cfg.mode,
            gain=cfg.gain,
            tolerance_c=cfg.temperature_tolerance_c,
            max_age_days=cfg.max_age_days,
        )
        model = self.library.model(cfg.mode, cfg.gain, prior_doubling_c=cfg.doubling_c)
        model_view: DarkModelView | None = None
        if model is not None:
            e_per_adu = self.profile.e_per_adu(cfg.mode, cfg.gain)
            model_view = DarkModelView(
                reference_c=model.reference_c,
                rate_ref_e_per_s=model.rate_ref_dn_per_s * e_per_adu,
                doubling_c=model.doubling_c,
                doubling_fitted=model.doubling_fitted,
                rms_log2=model.rms_log2,
                n_sets=model.n_sets,
            )
        return DarkLibraryView(
            mode=cfg.mode,
            gain=cfg.gain,
            exposure_s=cfg.exposure_s,
            sensor_temperature_c=temperature,
            status=DarkStatusView(
                due=status.due,
                reason=status.reason,
                tolerance_c=cfg.temperature_tolerance_c,
                max_age_days=cfg.max_age_days,
                gap_c=status.gap_c,
                nearest_name=None if status.nearest is None else status.nearest.name,
                newest_age_days=status.newest_age_days,
            ),
            model=model_view,
            sets=self._sets(now_ns),
            task=self.state.snapshot(),
        )

    def _sets(self, now_ns: int) -> list[DarkSetView]:
        views: list[DarkSetView] = []
        newest_first = sorted(self.library.sets(), key=lambda s: (s.t_utc_ns, s.name), reverse=True)
        for item in newest_first[:MAX_LIST_ITEMS]:
            try:
                e_per_adu = self.profile.e_per_adu(item.mode, item.gain)
            except ProfileError:
                continue  # a set of a readout mode that this profile does not know
            views.append(
                DarkSetView(
                    name=item.name,
                    t_utc=utc_ns_to_iso(item.t_utc_ns, digits=0),
                    age_days=item.age_s(now_ns) / SECONDS_PER_DAY,
                    temperature_c=item.temperature_c,
                    temperature_spread_c=item.temperature_spread_c,
                    exposure_s=item.exposure_s,
                    n_frames=item.n_frames,
                    n_bias_frames=item.n_bias_frames,
                    rate_e_per_s=item.rate_dn_per_s * e_per_adu,
                    hot_pixels=item.n_hot_pixels,
                )
            )
        return views


__all__ = [
    "PHASE_EVENT",
    "ContextCamera",
    "DarkHandler",
    "DarkLibraryReader",
    "DarkTaskState",
]
