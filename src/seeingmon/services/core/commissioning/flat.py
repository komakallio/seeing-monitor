"""The flat handler: take a flat with the camera that the scheduler lends, from the web UI.

The owner covers the front of the guide scope with a uniform light source, opens the Flat page, and
presses Take flat. The page sends a `QueueFlat` command, and the scheduler runs this handler in its
`commission` state, with the camera to itself. The session is the one of
`seeingmon.survey.flat_session` on a borrowed camera: it finds the exposure, takes the frames into
a temporary SER file, combines them with `make_flat` and the bias of the dark library, and adds the
flat to the flat library as a pending flat. Nothing changes for the survey until the owner
activates the flat.

The command's `frames`, `target_fraction`, and `set_number` win over the defaults, and the readout
mode, the gain, and the limits of the search come from `[survey.dark]` and `[survey.flat]`.
`pause_after` belongs to the scheduler: when the queue is empty again, it pauses, so that nothing
records data while the light source still covers the camera, and `Resume` continues. A
`CancelTask` for the kind `flat` removes a queued task, and it stops a running one at its next
frame.

**What each part does.**

- `FlatHandler` runs the task and returns a `CommissionResult`: `ok`, `failed` (with one plain
  sentence), or `aborted` (a command took the camera, and the library stays as it was). A
  `CameraError` goes on to the scheduler, which runs its recovery.
- `FlatTaskState` is the small, thread-safe record that the RPC `flat_library` reads: `queued` from
  the moment that `core` accepts the command, `running` with the phase, the exposure, the level, and
  the warnings, then `ok`, `failed`, or `aborted`.
- `FlatLibraryReader` builds the answer of the RPC (`FlatLibraryView`), and it carries out the
  activation and the deletion of a flat, the preview, and the check of a command that the scheduler
  cannot make (a second set without a first one, no dark set).

**Events.** The handler writes one `scheduler.flat_phase` event at each change of phase, and none
for each frame. The scheduler writes the start and the result (`scheduler.flat_result`).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from seeingmon.clock import Clock, utc_ns_to_iso
from seeingmon.drivers.base import CameraConfigError, CameraError
from seeingmon.profile import Profile, ProfileError
from seeingmon.scheduler.commands import QueueFlat
from seeingmon.scheduler.commission import CommissionContext, CommissionResult, CommissionTask
from seeingmon.services.core.commissioning.dark import ContextCamera
from seeingmon.services.web.contract import (
    MAX_FLAT_JPEG_BYTES,
    MAX_LIST_ITEMS,
    FlatActionView,
    FlatAgreementView,
    FlatLibraryView,
    FlatPointView,
    FlatSessionView,
    FlatSetView,
    FlatShadowView,
    FlatTaskView,
    FlatTiltView,
    FlatView,
)
from seeingmon.store.layout import DataLayout
from seeingmon.survey.config import SurveyConfig
from seeingmon.survey.dark import DarkLibrary
from seeingmon.survey.flat_library import (
    SESSION_TTL_S,
    FlatEntry,
    FlatLibrary,
    FlatLibraryError,
    flat_pins,
)
from seeingmon.survey.flat_make import MakeOptions
from seeingmon.survey.flat_session import (
    DARK_FIRST,
    NO_FIRST_SET,
    FlatAborted,
    FlatProgress,
    FlatSessionError,
    FlatSessionOptions,
    FlatSessionResult,
    disk_free_bytes,
    format_exposure,
    memory_available_bytes,
    record_flat,
)

_log = logging.getLogger(__name__)

PHASE_EVENT = "scheduler.flat_phase"
TaskState = Literal["idle", "queued", "running", "ok", "failed", "aborted"]
WAITING_MESSAGE = "The flat session waits for the scheduler to start it."
# A task starts in `safe` or `auto` only, so a session that you queue in another state waits.
WAITING_MESSAGES = {
    "paused": "The scheduler is paused. The flat session starts after you resume it.",
    "align": "The alignment helper runs. The flat session starts after it ends.",
}
BUSY_MESSAGE = "A flat session is queued or running. Wait until it ends, or stop it."
ABORTED_SUMMARY = "The flat session stopped before it added a flat, and the library is unchanged."
CANCELLED_SUMMARY = "The flat session was cancelled before it started."


def _sentence(text: str) -> str:
    """The text as one sentence: it starts with a capital letter and ends with a period."""
    text = text.strip()
    if not text:
        return text
    return (text[0].upper() + text[1:]).rstrip(".") + "."


# --- The state that the RPC reads ---------------------------------------------------------------


class FlatTaskState:
    """The latest flat session of this process, for the RPC `flat_library`. Thread-safe.

    The state starts `idle`. `core` calls `queued` when the scheduler accepts a `QueueFlat`, the
    handler calls `started`, `progress`, and `finished`, `core` calls `cancelled` when a
    `CancelTask` removed the queued task, and the RPC calls `snapshot`. A call to `queued` for a
    task that has begun already changes nothing, because the scheduler can start the task before
    the call that accepted it returns to `core`.
    """

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._view = FlatTaskView()

    def _now_iso(self) -> str:
        return utc_ns_to_iso(self._clock.utc_ns(), digits=0)

    def queued(self, task_id: int, command: QueueFlat, scheduler_state: str = "auto") -> None:
        """The scheduler accepted a flat task. `scheduler_state` is its state at that moment."""
        with self._lock:
            if self._view.task_id == task_id and self._view.state != "idle":
                return
            self._view = FlatTaskView(
                state="queued",
                task_id=task_id,
                message=WAITING_MESSAGES.get(scheduler_state, WAITING_MESSAGE),
                set_number=command.set_number,
                frames=command.frames,
                target_fraction=command.target_fraction,
                pause_after=command.pause_after,
            )

    def started(self, task_id: int, command: QueueFlat) -> None:
        """The handler began the task."""
        with self._lock:
            self._view = FlatTaskView(
                state="running",
                task_id=task_id,
                message="The flat session started.",
                set_number=command.set_number,
                frames=command.frames,
                target_fraction=command.target_fraction,
                pause_after=command.pause_after,
                started_utc=self._now_iso(),
            )

    def progress(self, progress: FlatProgress) -> None:
        """The session moved on. A value that the step does not know keeps its last value."""
        with self._lock:
            current = self._view
            self._view = current.model_copy(
                update={
                    "phase": progress.phase,
                    "step": progress.step,
                    "steps": progress.steps,
                    "message": progress.message,
                    "exposure_s": (
                        current.exposure_s if progress.exposure_s is None else progress.exposure_s
                    ),
                    "level_fraction": (
                        current.level_fraction
                        if progress.level_fraction is None
                        else progress.level_fraction
                    ),
                    "saturated_fraction": (
                        current.saturated_fraction
                        if progress.saturated_fraction is None
                        else progress.saturated_fraction
                    ),
                    "warnings": list(progress.warnings),
                }
            )

    def finished(
        self,
        status: Literal["ok", "failed", "aborted"],
        summary: str,
        version: str | None = None,
    ) -> None:
        """The task ended."""
        with self._lock:
            self._view = self._view.model_copy(
                update={
                    "state": status,
                    "phase": None,
                    "message": "",
                    "summary": summary,
                    "version": version,
                    "finished_utc": self._now_iso(),
                }
            )

    def cancelled(self, task_id: int) -> None:
        """A `CancelTask` removed the queued task. A task that runs ends through its handler."""
        with self._lock:
            view = self._view
            if view.task_id != task_id or view.state != "queued":
                return
            self._view = view.model_copy(
                update={
                    "state": "aborted",
                    "message": "",
                    "summary": CANCELLED_SUMMARY,
                    "finished_utc": self._now_iso(),
                }
            )

    def snapshot(self) -> FlatTaskView:
        """The state now, as the view that the RPC sends."""
        with self._lock:
            return self._view


class _PhaseReporter:
    """The progress callback of one task: it keeps the state and writes an event for each phase."""

    def __init__(self, context: CommissionContext, state: FlatTaskState, task_id: int) -> None:
        self._context = context
        self._state = state
        self._task_id = task_id
        self._phase: str | None = None

    def __call__(self, progress: FlatProgress) -> None:
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


class FlatHandler:
    """Runs `flat` tasks. Register it with `Scheduler.register_handler("flat", handler)`.

    `capture_allowed` is the gate of the storage, and `reserve_bytes` is the free space that the
    frames must leave. `free_bytes` tells the free space of the partition that holds a path (the
    default asks the operating system). `make_options` replaces the options of `make_flat`, which
    a test on a small sensor needs. `available_memory` tells the free memory (`None`: not known),
    and the default asks the operating system.
    """

    def __init__(
        self,
        *,
        flats: FlatLibrary,
        darks: DarkLibrary,
        layout: DataLayout,
        profile: Profile,
        clock: Clock,
        survey: SurveyConfig,
        state: FlatTaskState,
        capture_allowed: Callable[[], bool] | None = None,
        reserve_bytes: int = 0,
        free_bytes: Callable[[Path], int] | None = None,
        make_options: MakeOptions | None = None,
        available_memory: Callable[[], int | None] | None = None,
    ) -> None:
        self._flats = flats
        self._darks = darks
        self._layout = layout
        self._profile = profile
        self._clock = clock
        self._survey = survey
        self._state = state
        self._capture_allowed = capture_allowed
        self._reserve_bytes = reserve_bytes
        self._free_bytes = free_bytes
        self._make_options = make_options
        self._available_memory = available_memory

    def _result(
        self,
        task: CommissionTask,
        started_ns: int,
        status: Literal["ok", "failed", "aborted"],
        summary: str,
        data: dict[str, Any] | None = None,
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
        if not isinstance(command, QueueFlat):
            return self._result(task, started_ns, "failed", "The task holds no flat command.")
        try:
            options = FlatSessionOptions.from_config(
                self._survey,
                frames=command.frames,
                target_fraction=command.target_fraction,
                set_number=command.set_number,
                make=self._make_options,
            )
        except ValueError as error:
            summary = _sentence(f"The flat settings are not valid: {error}")
            self._state.started(task.task_id, command)
            self._state.finished("failed", summary)
            return self._result(task, started_ns, "failed", summary)
        self._state.started(task.task_id, command)
        finished = False
        try:
            status, summary, data, artifacts, version = self._record(task, options, context)
            finished = True
        except CameraError as error:
            # The scheduler runs the recovery. The state says what happened to the task.
            self._state.finished("failed", _sentence(f"The camera failed: {error}"))
            finished = True
            raise
        finally:
            if not finished:
                self._state.finished("failed", "The flat session ended unexpectedly.")
        self._state.finished(status, summary, version)
        return self._result(task, started_ns, status, summary, data, artifacts)

    def _record(
        self, task: CommissionTask, options: FlatSessionOptions, context: CommissionContext
    ) -> tuple[
        Literal["ok", "failed", "aborted"], str, dict[str, Any], tuple[str, ...], str | None
    ]:
        """Run the session. Returns the status, the summary, the data, the files, and the flat."""
        remove_light: dict[str, Any] = {"remove_light": True}  # the page tells you to uncover
        try:
            with self._flats.session.capturing():  # the periodic sweep leaves the folder alone
                session = record_flat(
                    ContextCamera(context),
                    self._flats,
                    self._darks,
                    self._profile,
                    context.clock,
                    options,
                    say=_log.debug,
                    progress=_PhaseReporter(context, self._state, task.task_id),
                    should_stop=context.should_stop,
                    free_bytes=self._free_bytes or disk_free_bytes,
                    reserve_bytes=self._reserve_bytes,
                    capture_allowed=self._capture_allowed,
                    available_memory=self._available_memory or memory_available_bytes,
                )
        except FlatAborted:
            return "aborted", ABORTED_SUMMARY, remove_light, (), None
        except FlatSessionError as error:
            return "failed", _sentence(f"{error} The library is unchanged"), remove_light, (), None
        except CameraConfigError as error:
            summary = _sentence(f"The camera refused the flat settings: {error}")
            return "failed", summary, remove_light, (), None
        except OSError as error:
            summary = f"The flat could not be written ({type(error).__name__})."
            return "failed", summary, remove_light, (), None
        data = {**self._data(session), **remove_light}
        return (
            "ok",
            describe_session(session),
            data,
            (self._artifact(session),),
            session.entry.version,
        )

    def _artifact(self, session: FlatSessionResult) -> str:
        """The file of the flat, relative to the data directory when the library lives there."""
        path = self._flats.directory / f"{session.entry.version}.npy"
        try:
            return self._layout.relative(path)
        except ValueError:
            return path.name

    @staticmethod
    def _data(session: FlatSessionResult) -> dict[str, Any]:
        report = session.entry.report
        shadows = report.get("shadows") or {}
        return {
            "version": session.entry.version,
            "set_number": session.set_number,
            "second_set": bool(report.get("second_set")),
            "exposure_s": session.exposure_s,
            "level_fraction": session.level_fraction,
            "frames_taken": session.frames_taken,
            "frames_used": sum(int(item.get("used", 0)) for item in report.get("sets", ())),
            "temperature_c": session.temperature_c,
            "corner_percent": (report.get("vignetting") or {}).get("corner_percent"),
            "shadows": shadows.get("count", 0),
            "warnings": list(session.warnings),
        }


def describe_session(session: FlatSessionResult) -> str:
    """The sentence that says what a session made, and what to do with it."""
    report = session.entry.report
    corner = (report.get("vignetting") or {}).get("corner_percent")
    if report.get("second_set"):
        text = f"Made the flat {session.entry.version} from two sets of frames"
    else:
        text = (
            f"Made the flat {session.entry.version} from {session.frames_taken} frames of "
            f"{format_exposure(session.exposure_s)}"
        )
    if isinstance(corner, int | float):
        more = "more" if corner > 0 else "less"
        text += f". The corners get {abs(corner):.0f} % {more} light than the center"
    return text + ". It waits on the Flat page for you to use it or discard it."


# --- The view for the RPC -----------------------------------------------------------------------


def _float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _int(value: Any, default: int = 0) -> int:
    return default if isinstance(value, bool) or not isinstance(value, int) else value


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _tilt(value: Any) -> FlatTiltView | None:
    if not isinstance(value, dict):
        return None
    return FlatTiltView(
        width_percent=_float(value.get("width_percent")),
        height_percent=_float(value.get("height_percent")),
    )


def flat_view(entry: FlatEntry, *, active: bool, now_ns: int, has_image: bool) -> FlatView:
    """A flat of the library as the view that the RPC sends. A value that the report lacks is
    `None` or empty, so a report from another version still shows."""
    report = entry.report
    vignetting = report.get("vignetting") or {}
    shadows = report.get("shadows") or {}
    bias = report.get("bias") or {}
    split = report.get("split") or {}
    agreement = report.get("agreement")
    sets = [
        FlatSetView(
            number=_int(item.get("number")),
            exposure_s=_float(item.get("exposure_s")),
            level_fraction=_float(item.get("level_fraction")),
            frames=_int(item.get("frames")),
            used=_int(item.get("used")),
            dropped={str(k): _int(v) for k, v in (item.get("dropped") or {}).items()},
            noise_percent=_float(item.get("noise_percent")),
            tilt=_tilt(item.get("tilt")),
        )
        for item in report.get("sets", ())
        if isinstance(item, dict)
    ]
    return FlatView(
        version=entry.version,
        t_utc=str(report.get("t_utc") or utc_ns_to_iso(entry.t_utc_ns, digits=0)),
        age_days=max(0.0, (now_ns - entry.t_utc_ns) / 86_400e9),
        state=entry.state,
        active=active,
        pending=entry.pending,
        mode=str(report.get("mode", "")),
        gain=_int(report.get("gain")),
        width_px=_int(report.get("width_px")),
        height_px=_int(report.get("height_px")),
        sensor_temperature_c=_float(report.get("sensor_temperature_c")),
        exposure_s=_float(report.get("exposure_s")),
        target_fraction=_float(report.get("target_fraction")),
        second_set=bool(report.get("second_set")),
        source_turned=bool(report.get("source_turned")),
        frames_taken=sum(item.frames for item in sets),
        frames_used=sum(item.used for item in sets),
        noise_percent=_float(report.get("noise_percent")),
        bias_source=str(bias.get("source", "")),
        bias_note=str(bias.get("note", "")),
        corner_percent=_float(vignetting.get("corner_percent")),
        vignetting=[
            FlatPointView(
                radius_deg=float(point.get("radius_deg", 0.0)),
                change_percent=_float(point.get("change_percent")),
                corner=bool(point.get("corner")),
            )
            for point in vignetting.get("points", ())
            if isinstance(point, dict)
        ],
        tilt=_tilt(report.get("tilt")) or FlatTiltView(),
        optics_tilt=_tilt(split.get("optics")),
        source_tilt=_tilt(split.get("source")),
        shadows=_int(shadows.get("count")),
        shadow_min_depth_percent=_float(shadows.get("min_depth_percent")),
        shadow_items=[
            FlatShadowView(
                x_px=_int(item.get("x_px")),
                y_px=_int(item.get("y_px")),
                depth_percent=float(item.get("depth_percent", 0.0)),
                width_px=float(item.get("width_px", 0.0)),
            )
            for item in shadows.get("items", ())
            if isinstance(item, dict)
        ],
        edge_artifacts=_int(report.get("edge_artifacts")),
        agreement=(
            None
            if not isinstance(agreement, dict)
            else FlatAgreementView(
                smooth_rms_percent=_float(agreement.get("smooth_rms_percent")),
                fine_rms_percent=_float(agreement.get("fine_rms_percent")),
                expected_fine_rms_percent=_float(agreement.get("expected_fine_rms_percent")),
                plane=_tilt(agreement.get("plane")),
            )
        ),
        sets=sets,
        warnings=[str(text) for text in report.get("warnings", ())][:32],
        has_image=has_image,
        activated_utc=_text(report.get("activated_utc")),
    )


@dataclass(frozen=True, slots=True)
class FlatLibraryReader:
    """The flat library as `core` serves it: the view, the actions, the preview, and the check.

    `temperature_c` gives the sensor temperature from the scheduler's status, and `None` when no
    frame has come yet. The mode and the gain are the ones of the survey (`[survey.dark]`).
    """

    flats: FlatLibrary
    darks: DarkLibrary
    profile: Profile
    clock: Clock
    survey: SurveyConfig
    state: FlatTaskState
    temperature_c: Callable[[], float | None]

    # --- The view -----------------------------------------------------------------------------

    def view(self) -> FlatLibraryView:
        cfg = self.survey.dark
        now_ns = self.clock.utc_ns()
        entries = self.flats.entries()
        pins = flat_pins(self.survey)
        active = self.flats.active_version()
        session = self.flats.session.load(now_ns)
        return FlatLibraryView(
            mode=cfg.mode,
            gain=cfg.gain,
            sensor_temperature_c=self.temperature_c(),
            active_version=active,
            pending_version=next((e.version for e in entries if e.pending), None),
            flat_file_pinned=pins.flat_file_pinned,
            library_overrides=pins.flat_file_pinned and active is not None,
            blocker=None if self._has_dark_model() else DARK_FIRST,
            flats=[
                flat_view(
                    entry,
                    active=entry.version == active,
                    now_ns=now_ns,
                    has_image=(self.flats.directory / f"{entry.version}.jpg").is_file(),
                )
                for entry in entries[:MAX_LIST_ITEMS]
            ],
            session=(
                None
                if session is None
                else FlatSessionView(
                    version=session.version,
                    t_utc=utc_ns_to_iso(session.t_utc_ns, digits=0),
                    expires_utc=utc_ns_to_iso(
                        session.t_utc_ns + SESSION_TTL_S * 1_000_000_000, digits=0
                    ),
                    frames=session.frames,
                    exposure_s=session.exposure_s,
                )
            ),
            task=self.state.snapshot(),
        )

    def _has_dark_model(self) -> bool:
        cfg = self.survey.dark
        return bool(self.darks.sets_for(cfg.mode, cfg.gain))

    # --- The actions --------------------------------------------------------------------------

    def _session_runs(self) -> bool:
        return self.state.snapshot().state in ("queued", "running")

    def activate(self, version: str) -> FlatActionView:
        """Make a flat the active one. The survey divides by it from its next frame."""
        if self._session_runs():
            return FlatActionView(ok=False, reason="busy", message=BUSY_MESSAGE, version=version)
        readout = self._readout()
        try:
            self.flats.activate(
                version,
                now_utc_ns=self.clock.utc_ns(),
                expect_shape=None if readout is None else (readout[0], readout[1]),
            )
        except FlatLibraryError as error:
            return FlatActionView(
                ok=False, reason=error.reason, message=error.message, version=version
            )
        self._end_session_of(version)
        message = f"The flat {version} is in use. The survey divides by it from its next frame."
        if self.survey.flat_file:
            message += " It replaces the flat of the setting flat_file."
        return FlatActionView(ok=True, message=message, version=version)

    def delete(self, version: str) -> FlatActionView:
        """Delete a flat that is not in use."""
        if self._session_runs():
            return FlatActionView(ok=False, reason="busy", message=BUSY_MESSAGE, version=version)
        try:
            self.flats.delete(version)
        except FlatLibraryError as error:
            return FlatActionView(
                ok=False, reason=error.reason, message=error.message, version=version
            )
        self._end_session_of(version)
        return FlatActionView(ok=True, message=f"The flat {version} is deleted.", version=version)

    def _end_session_of(self, version: str) -> None:
        """A session ends when you decide about its flat: its first set has no use any more."""
        session = self.flats.session.load(self.clock.utc_ns())
        if session is not None and session.version == version:
            self.flats.session.clear()

    def _readout(self) -> tuple[int, int] | None:
        try:
            mode = self.profile.mode(self.survey.dark.mode)
        except ProfileError:
            return None
        return (mode.height_px, mode.width_px)

    def image(self, version: str) -> bytes | None:
        """The JPEG preview of a flat, or `None` when it has none or is too large to send."""
        jpeg = self.flats.jpeg(version)
        if jpeg is not None and len(jpeg) > MAX_FLAT_JPEG_BYTES:
            _log.warning("a flat preview is too large for the RPC, so the page shows none")
            return None
        return jpeg

    # --- The check of a command ---------------------------------------------------------------

    def check(self, command: QueueFlat) -> str | None:
        """A sentence that says why the scheduler cannot take this command, or `None`.

        The scheduler checks the numbers. This check knows the libraries: a session needs a dark
        model for the bias, and a second set needs the first set of a session.
        """
        if not self._has_dark_model():
            return DARK_FIRST
        if command.set_number == 2 and self.flats.session.load(self.clock.utc_ns()) is None:
            return NO_FIRST_SET
        return None


__all__ = [
    "PHASE_EVENT",
    "FlatHandler",
    "FlatLibraryReader",
    "FlatTaskState",
    "describe_session",
    "flat_view",
]
