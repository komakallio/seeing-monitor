"""A flat library and a scripted flat task, for `FakeCoreClient` and the demo.

The real `core` records a flat in its `commission` state: it finds the exposure for a light source
that covers the camera, takes frames into a file, combines them with the bias of the dark library,
and adds the flat to the flat library as a pending flat. The owner then looks at the flat and uses
it or discards it. `FlatSimulator` plays the same story on a clock, without a camera, so that the
tests of the API and the demo of the UI can show every state.

The task follows the clock of the fake core. It is `queued` for `FlatScript.queued_s`, and then it
`runs` through the phases `setup`, `exposure` (the search for the exposure), `capture` (the frames),
and `build` (the combination), each one as long as the script says. A finished task adds a pending
flat to the library, and a first set keeps a session open for a second set (24 hours). A second set
(`set_number` 2) combines with it, so the flat of the first set gives way. With `pause_after`, the
fake scheduler ends in `paused`, and `Resume` works as in the real one. A `Pause` ends a running
task as `aborted`, and `CancelTask` ends a queued or a running one.

`FlatScript.outcome` decides how a session ends: `ok`, `dim` (not enough light), or `bright` (too
much light). `FlatScript.warnings` are notes that the capture shows from its middle on, and that the
flat keeps in its report.

The simulator checks a command like `core` does: the dark library must hold a set, a second set
needs a first set, 8 to 64 frames, a target of 0.3 to 0.7, and one task at a time
(`RejectReason.BUSY`).
"""

from __future__ import annotations

import hashlib
import io
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from seeingmon.clock import NS_PER_S, Clock, utc_ns_to_iso
from seeingmon.scheduler.commands import QueueFlat, RejectReason
from seeingmon.services.web.contract import (
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
from seeingmon.services.web.fake_dark import Transition

MIN_FRAMES = 8
MAX_FRAMES = 64
MIN_TARGET = 0.3
MAX_TARGET = 0.7
KEEP_FLATS = 10
MAX_EXPOSURE_S = 1.0
MIN_EXPOSURE_S = 32e-6
START_EXPOSURE_S = 0.02
MIN_LEVEL_FRACTION = 0.25
DIM_RATE_PER_S = 0.1  # the level of a dim light: 10 % of the full scale after a second
BRIGHT_RATE_PER_S = 40_000.0  # a light that saturates the frame even at the shortest exposure
SESSION_TTL_NS = 24 * 3600 * NS_PER_S
DAY_NS = 86_400 * NS_PER_S
# These two sentences are the ones that `seeingmon.survey.flat_session` gives. The web process does
# not import the survey code, so a test compares the copies.
DARK_FIRST = "Record a dark set first (Dark page)."
NO_FIRST_SET = (
    "There is no first set to combine with. Take the first set, and take the second within "
    "24 hours."
)
BUSY_MESSAGE = "A flat session is queued or running. Wait until it ends, or stop it."
ABORTED_SUMMARY = "The flat session stopped before it added a flat, and the library is unchanged."
CANCELLED_SUMMARY = "The flat session was cancelled before it started."
ACTIVE_MESSAGE = "The flat is in use. Activate another flat first."
UNKNOWN_MESSAGE = "There is no flat with that name."
SOURCE_WARNING = (
    "The tilt may include the gradient of your light source, up to about 1% for a phone screen. "
    "A second set with the source turned by 180 degrees separates the two."
)
DIM_SUMMARY = (
    "Not enough light: the frame reaches 10 % of full scale at the longest exposure of 1 s. Use a "
    "brighter source, or hold it closer to the lens. The library is unchanged."
)
BRIGHT_SUMMARY = (
    "Too much light: the frame saturates at the shortest exposure of 32 µs. Dim the source, or put "
    "a layer of cloth between it and the lens. The library is unchanged."
)
# A task runs only in `safe` or `auto`. In these states of the scheduler, it waits.
HOLD_MESSAGES = {
    "paused": "The scheduler is paused. The flat session starts after you resume it.",
    "align": "The alignment helper runs. The flat session starts after it ends.",
}
WAIT_MESSAGE = "The flat session waits for the scheduler to start it."
PHASES = ("setup", "exposure", "capture", "build")
SENSOR_SHAPE = (2822, 4144)  # rows and columns of the survey mode in the demo
PLATE_SCALE_ARCSEC_PX = 3.82
PREVIEW_SHAPE = (353, 518)


@dataclass(frozen=True, slots=True)
class FlatScript:
    """How long each part of a scripted task lasts, in seconds of the clock, and how it ends.

    `rate_per_s` is the level of the light source in the middle of the frame, as a fraction of the
    full scale for each second of exposure. The exposure that reaches a target level follows from
    it. `warnings` appear while the frames come in, from the middle of the capture on.
    """

    queued_s: float = 0.0
    setup_s: float = 1.0
    exposure_s: float = 3.0
    capture_s: float = 8.0
    build_s: float = 2.0
    outcome: str = "ok"
    rate_per_s: float = 12.8
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FlatLook:
    """What a scripted flat looks like: the loss in the corners, the tilt, the dust, and the noise.

    A shadow is `(x_px, y_px, depth_percent, width_px)` on the sensor. `tilt` is the tilt of the
    optics alone, and `source_tilt` is what a light source adds to the first set (the second set,
    with the source turned by 180 degrees, gets the opposite).
    """

    corner_percent: float = -9.6
    tilt: tuple[float, float] = (-0.40, 0.30)
    source_tilt: tuple[float, float] = (-0.22, 0.11)
    shadows: tuple[tuple[int, int, float, float], ...] = (
        (1210, 802, 2.4, 38.0),
        (2874, 1905, 1.6, 26.0),
        (3420, 420, 1.1, 21.0),
    )
    edge_artifacts: int = 2
    noise_percent: float = 0.11


@dataclass(slots=True)
class _Run:
    task_id: int
    frames: int
    target_fraction: float
    set_number: int
    pause_after: bool
    submitted_ns: int
    submitted_utc_ns: int
    started: bool = False
    outcome: str | None = None  # `ok`, `failed`, or `aborted`
    summary: str = ""
    version: str | None = None
    finished_utc_ns: int | None = None


@dataclass(slots=True)
class _Entry:
    view: FlatView
    t_utc_ns: int
    frames: int
    exposure_s: float
    level_fraction: float


@dataclass(frozen=True, slots=True)
class CancelOutcome:
    """What a `CancelTask` did: whether the fake took it, and the changes of the scheduler."""

    accepted: bool
    reason: RejectReason | None
    message: str
    task_id: int | None = None
    changes: tuple[Transition, ...] = ()


def exposure_text(seconds: float) -> str:
    """An exposure in words: `1.5 s`, `40 ms`, or `32 µs`."""
    if seconds >= 1.0:
        return f"{seconds:.3g} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.3g} ms"
    return f"{seconds * 1e6:.3g} µs"


def _step(seconds: float, length_s: float, steps: int) -> int:
    """The frame (counted from 1) that a phase of `length_s` is at, `seconds` after its start."""
    if length_s <= 0:
        return steps
    return min(steps, int(seconds / length_s * steps) + 1)


def _tilt(width: float, height: float) -> FlatTiltView:
    return FlatTiltView(width_percent=round(width, 3), height_percent=round(height, 3))


def corner_radius_deg() -> float:
    """The distance of a corner from the center of the sensor, in degrees."""
    half_height, half_width = SENSOR_SHAPE[0] / 2, SENSOR_SHAPE[1] / 2
    return math.hypot(half_width, half_height) * PLATE_SCALE_ARCSEC_PX / 3600.0


def vignetting_points(corner_percent: float) -> list[FlatPointView]:
    """The loss of light at a few radii, from 0.5 degrees to the corner, falling with the radius."""
    corner = corner_radius_deg()
    radii = [0.5, 1.0, 1.5, 2.0]
    points = [
        FlatPointView(
            radius_deg=radius,
            change_percent=round(corner_percent * (radius / corner) ** 2.3, 2),
            corner=False,
        )
        for radius in radii
        if radius < corner - 0.1
    ]
    points.append(
        FlatPointView(radius_deg=round(corner, 3), change_percent=corner_percent, corner=True)
    )
    return points


def render_flat_jpeg(view: FlatView) -> bytes:
    """A JPEG of a flat as its numbers describe it, stretched to plus and minus 10 % around 1.

    The preview is a made-up picture of the loss in the corners, the tilt, and the dust. Pillow
    loads when the function runs, because it belongs to the web extra.
    """
    from PIL import Image

    rows, columns = PREVIEW_SHAPE
    y, x = np.mgrid[0:rows, 0:columns].astype(np.float32)
    cx, cy = (columns - 1) / 2.0, (rows - 1) / 2.0
    radius = np.hypot(x - cx, y - cy) / np.hypot(cx, cy)
    corner = 0.0 if view.corner_percent is None else view.corner_percent / 100.0
    image = 1.0 + np.float32(corner) * radius**2.3
    tilt = view.tilt
    image += np.float32((tilt.width_percent or 0.0) / 100.0) * (x - cx) / columns
    image += np.float32((tilt.height_percent or 0.0) / 100.0) * (y - cy) / rows
    scale = columns / SENSOR_SHAPE[1]
    for shadow in view.shadow_items:
        sigma = max(1.5, shadow.width_px * scale / 2.0)
        distance = ((x - shadow.x_px * scale) ** 2 + (y - shadow.y_px * scale) ** 2) / sigma**2
        image *= 1.0 - np.float32(shadow.depth_percent / 100.0) * np.exp(-distance)
    seed = int(view.version[-8:], 16)
    noise = np.random.default_rng(seed).standard_normal(image.shape).astype(np.float32)
    image += np.float32(view.noise_percent or 0.0) / 100.0 * noise
    scaled = np.clip((image - 0.9) / 0.2, 0, 1)
    pixels = np.asarray(scaled * 255.0 + 0.5, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


class FlatSimulator:
    """The flat library of a fake `core`, and the scripted task that adds a flat to it.

    Set `flats` (the newest first) and `active_version` to give the library a content, or call
    `seed` to add flats with a look. `dark_ready` says whether the dark library holds a set: without
    one, a session cannot start (`blocker`), as in the real `core`.
    """

    def __init__(
        self,
        clock: Clock,
        *,
        script: FlatScript | None = None,
        mode: str = "bin2",
        gain: int = 120,
        sensor_temperature_c: float | None = 12.3,
        dark_ready: Callable[[], bool] | None = None,
        look: FlatLook | None = None,
    ) -> None:
        self._clock = clock
        self.script = script or FlatScript()
        self.mode = mode
        self.gain = gain
        self.sensor_temperature_c = sensor_temperature_c
        self.dark_ready: Callable[[], bool] = dark_ready or (lambda: True)
        self.look = look or FlatLook()
        self.flat_file_pinned = False
        self.active_version: str | None = None
        self._entries: list[_Entry] = []  # the newest first
        self._session: tuple[str, int, int, float, float] | None = None
        self._jpegs: dict[str, bytes] = {}
        self._run: _Run | None = None
        self._held_by: str | None = None
        self._made = 0

    # --- The library -----------------------------------------------------------------------

    @property
    def flats(self) -> list[FlatView]:
        """The flats of the library, the newest first, as the library shows them now."""
        return [self._shown(entry) for entry in self._entries]

    def _shown(self, entry: _Entry) -> FlatView:
        now_ns = self._clock.utc_ns()
        return entry.view.model_copy(
            update={
                "age_days": round(max(0.0, (now_ns - entry.t_utc_ns) / DAY_NS), 3),
                "active": entry.view.version == self.active_version,
                "pending": entry.view.state == "pending",
            }
        )

    def _new_version(self, t_utc_ns: int) -> str:
        self._made += 1
        text = f"{t_utc_ns}:{self._made}:{self.mode}:{self.gain}"
        return "flat-" + hashlib.sha256(text.encode()).hexdigest()[:8]

    def seed(
        self,
        *,
        age_days: float,
        look: FlatLook | None = None,
        state: str = "approved",
        frames: int = 32,
        active: bool = False,
        second_set: bool = False,
    ) -> FlatView:
        """Add a flat that is `age_days` old, and return it as the library shows it."""
        t_utc_ns = self._clock.utc_ns() - round(age_days * DAY_NS)
        entry = self._add(
            t_utc_ns,
            look or self.look,
            state=state,
            frames=frames,
            exposure_s=round(0.5 / self.script.rate_per_s, 4),
            target_fraction=0.5,
            second_set=second_set,
        )
        if active:
            self.active_version = entry.view.version
        return self._shown(entry)

    def _add(
        self,
        t_utc_ns: int,
        look: FlatLook,
        *,
        state: str,
        frames: int,
        exposure_s: float,
        target_fraction: float,
        second_set: bool,
        warnings: tuple[str, ...] = (),
    ) -> _Entry:
        version = self._new_version(t_utc_ns)
        view = self._view(
            version,
            t_utc_ns,
            look,
            state=state,
            frames=frames,
            exposure_s=exposure_s,
            target_fraction=target_fraction,
            second_set=second_set,
            warnings=warnings,
        )
        entry = _Entry(view, t_utc_ns, frames, exposure_s, target_fraction)
        # A clock that stands still gives two flats the same time, so the new one goes first.
        self._entries.insert(0, entry)
        self._entries.sort(key=lambda item: item.t_utc_ns, reverse=True)
        self._prune()
        return entry

    def _view(
        self,
        version: str,
        t_utc_ns: int,
        look: FlatLook,
        *,
        state: str,
        frames: int,
        exposure_s: float,
        target_fraction: float,
        second_set: bool,
        warnings: tuple[str, ...],
    ) -> FlatView:
        optics, source = look.tilt, look.source_tilt
        first = _tilt(optics[0] + source[0], optics[1] + source[1])
        second = _tilt(optics[0] - source[0], optics[1] - source[1])
        flickered = 1 if frames > 16 else 0  # a long set of a real light loses a frame or two
        used = [frames - flickered, frames]
        sets = [
            FlatSetView(
                number=number,
                exposure_s=exposure_s,
                level_fraction=round(target_fraction, 3),
                frames=frames,
                used=used[number - 1],
                dropped={"flicker": flickered} if flickered and number == 1 else {},
                noise_percent=round(look.noise_percent * 1.4, 3),
                tilt=(first, second)[number - 1],
            )
            for number in range(1, 3 if second_set else 2)
        ]
        taken = sum(item.frames for item in sets)
        counted = sum(item.used for item in sets)
        temperature = 12.3 if self.sensor_temperature_c is None else self.sensor_temperature_c
        stamp = utc_ns_to_iso(t_utc_ns, digits=0)
        return FlatView(
            version=version,
            t_utc=stamp,
            state=state,
            mode=self.mode,
            gain=self.gain,
            width_px=SENSOR_SHAPE[1],
            height_px=SENSOR_SHAPE[0],
            sensor_temperature_c=self.sensor_temperature_c,
            exposure_s=exposure_s,
            target_fraction=round(target_fraction, 3),
            second_set=second_set,
            source_turned=second_set,
            frames_taken=taken,
            frames_used=counted,
            noise_percent=round(look.noise_percent * math.sqrt(32 / counted), 3),
            bias_source="dark library",
            bias_note=(
                f"from the dark library (6 sets of {self.mode} at gain {self.gain}, "
                f"interpolated to {temperature:.1f} C)"
            ),
            corner_percent=look.corner_percent,
            vignetting=vignetting_points(look.corner_percent),
            tilt=_tilt(*optics) if second_set else first,
            optics_tilt=_tilt(*optics) if second_set else None,
            source_tilt=_tilt(*source) if second_set else None,
            shadows=len(look.shadows),
            shadow_min_depth_percent=min((item[2] for item in look.shadows), default=None),
            shadow_items=[
                FlatShadowView(x_px=x, y_px=y, depth_percent=depth, width_px=width)
                for x, y, depth, width in look.shadows
            ],
            edge_artifacts=look.edge_artifacts,
            agreement=(
                FlatAgreementView(
                    smooth_rms_percent=0.09,
                    fine_rms_percent=0.13,
                    expected_fine_rms_percent=0.12,
                    plane=_tilt(
                        (first.width_percent or 0.0) - (second.width_percent or 0.0),
                        (first.height_percent or 0.0) - (second.height_percent or 0.0),
                    ),
                )
                if second_set
                else None
            ),
            sets=sets,
            warnings=[*warnings, *([] if second_set else [SOURCE_WARNING])],
            has_image=True,
            activated_utc=stamp if state == "approved" else None,
        )

    def _prune(self) -> None:
        """Keep the newest flats, and the flat in use whatever its age."""
        keep: list[_Entry] = []
        for index, entry in enumerate(self._entries):
            if index < KEEP_FLATS or entry.view.version == self.active_version:
                keep.append(entry)
            else:
                self._jpegs.pop(entry.view.version, None)
        self._entries = keep

    def _entry(self, version: str) -> _Entry | None:
        return next((e for e in self._entries if e.view.version == version), None)

    @property
    def blocker(self) -> str | None:
        """Why a session cannot start now, or `None`."""
        return None if self.dark_ready() else DARK_FIRST

    def library(self) -> FlatLibraryView:
        """The answer of `flat_library`."""
        flats = self.flats
        session = None
        if self._session is not None:
            version, t_utc_ns, frames, exposure_s, _ = self._session
            session = FlatSessionView(
                version=version,
                t_utc=utc_ns_to_iso(t_utc_ns, digits=0),
                expires_utc=utc_ns_to_iso(t_utc_ns + SESSION_TTL_NS, digits=0),
                frames=frames,
                exposure_s=exposure_s,
            )
        return FlatLibraryView(
            mode=self.mode,
            gain=self.gain,
            sensor_temperature_c=self.sensor_temperature_c,
            active_version=self.active_version,
            pending_version=next((item.version for item in flats if item.pending), None),
            flat_file_pinned=self.flat_file_pinned,
            library_overrides=self.flat_file_pinned and self.active_version is not None,
            blocker=self.blocker,
            flats=flats,
            session=session,
            task=self.task(),
        )

    def image(self, version: str) -> bytes | None:
        """The JPEG of a flat, or `None` when the library has no such flat."""
        entry = self._entry(version)
        if entry is None or not entry.view.has_image:
            return None
        if version not in self._jpegs:
            self._jpegs[version] = render_flat_jpeg(entry.view)
        return self._jpegs[version]

    # --- Decisions -------------------------------------------------------------------------

    def _refuse(self, version: str) -> FlatActionView | None:
        if self.active:
            return FlatActionView(ok=False, reason="busy", message=BUSY_MESSAGE, version=version)
        if self._entry(version) is None:
            return FlatActionView(
                ok=False, reason="unknown", message=UNKNOWN_MESSAGE, version=version
            )
        return None

    def _end_session_of(self, version: str) -> None:
        if self._session is not None and self._session[0] == version:
            self._session = None

    def activate(self, version: str) -> FlatActionView:
        """Make a flat the active one."""
        refused = self._refuse(version)
        if refused is not None:
            return refused
        entry = self._entry(version)
        assert entry is not None
        stamp = utc_ns_to_iso(self._clock.utc_ns(), digits=0)
        entry.view = entry.view.model_copy(update={"state": "approved", "activated_utc": stamp})
        self.active_version = version
        self._end_session_of(version)
        message = f"The flat {version} is in use. The survey divides by it from its next frame."
        if self.flat_file_pinned:
            message += " It replaces the flat of the setting flat_file."
        return FlatActionView(ok=True, message=message, version=version)

    def delete(self, version: str) -> FlatActionView:
        """Delete a flat that is not in use."""
        refused = self._refuse(version)
        if refused is not None:
            return refused
        if self.active_version == version:
            return FlatActionView(
                ok=False, reason="active", message=ACTIVE_MESSAGE, version=version
            )
        self._entries = [e for e in self._entries if e.view.version != version]
        self._jpegs.pop(version, None)
        self._end_session_of(version)
        return FlatActionView(ok=True, message=f"The flat {version} is deleted.", version=version)

    # --- The task --------------------------------------------------------------------------

    @property
    def active(self) -> bool:
        """Whether a task is queued or running."""
        return self._run is not None and self._run.outcome is None

    @property
    def queued(self) -> bool:
        """Whether a task waits for its turn, which counts as a queued task of the scheduler."""
        return self.active and self._run is not None and not self._run.started

    @property
    def running(self) -> bool:
        """Whether a task runs now."""
        return self.active and self._run is not None and self._run.started

    def submit(
        self, command: QueueFlat, task_id: int, state: str = "auto"
    ) -> tuple[bool, RejectReason | None, str]:
        """Check a command, and queue the task. Returns whether it was accepted, and why not."""
        if self.blocker is not None:
            return False, RejectReason.INVALID, self.blocker
        if command.set_number == 2 and self._session is None:
            return False, RejectReason.INVALID, NO_FIRST_SET
        if not MIN_FRAMES <= command.frames <= MAX_FRAMES:
            return (
                False,
                RejectReason.INVALID,
                f"frames must be between {MIN_FRAMES} and {MAX_FRAMES}",
            )
        if not MIN_TARGET <= command.target_fraction <= MAX_TARGET:
            return (
                False,
                RejectReason.INVALID,
                f"target_fraction must be between {MIN_TARGET:g} and {MAX_TARGET:g}",
            )
        if command.set_number not in (1, 2):
            return False, RejectReason.INVALID, "set_number must be 1 or 2"
        if self.active:
            return (
                False,
                RejectReason.BUSY,
                "a flat session is already queued or running; wait until it ends",
            )
        self._run = _Run(
            task_id=task_id,
            frames=command.frames,
            target_fraction=command.target_fraction,
            set_number=command.set_number,
            pause_after=command.pause_after,
            submitted_ns=self._clock.monotonic_ns(),
            submitted_utc_ns=self._clock.utc_ns(),
        )
        return (
            True,
            None,
            HOLD_MESSAGES.get(state, "the flat session is queued and starts at the next step"),
        )

    def abort(self) -> bool:
        """End a running task as `aborted`, because the scheduler paused. A queued task waits."""
        run = self._run
        if run is None or run.outcome is not None or not run.started:
            return False
        self._finish(run, "aborted", ABORTED_SUMMARY)
        return True

    def cancel(self, state: str = "auto") -> CancelOutcome:
        """The `CancelTask` command: remove a queued task, or stop the running one."""
        run = self._run
        if run is None or run.outcome is not None:
            return CancelOutcome(False, RejectReason.NO_TASK, "no flat task waits or runs")
        if not run.started:
            self._finish(run, "aborted", CANCELLED_SUMMARY)
            return CancelOutcome(
                True, None, "the waiting flat is removed, and it never starts", run.task_id
            )
        self._finish(run, "aborted", ABORTED_SUMMARY)
        return CancelOutcome(
            True,
            None,
            "the running flat stops at its next check",
            run.task_id,
            (self._after(run),),
        )

    def settle(self, *, state: str = "auto") -> list[Transition]:
        """Move the task to where the clock puts it. Returns the changes of the scheduler.

        `state` is the state of the scheduler. A paused scheduler and a running alignment hold a
        task that waits for its turn, and its queue time starts again when they let go.
        """
        run = self._run
        if run is None or run.outcome is not None:
            return []
        self._held_by = state if state in HOLD_MESSAGES else None
        if self._held_by is not None and not run.started:
            run.submitted_ns = self._clock.monotonic_ns()
            return []
        changes: list[Transition] = []
        seconds = self._seconds_running(run)
        if seconds >= 0 and not run.started:
            run.started = True
            changes.append(Transition("commission", "a flat session runs"))
        outcome = self._outcome(seconds)
        if outcome is not None:
            self._complete(run, outcome)
            changes.append(self._after(run))
        return changes

    @staticmethod
    def _after(run: _Run) -> Transition:
        """The change of the scheduler when the task has ended."""
        if run.pause_after:
            return Transition(
                "paused",
                "the flat session is done, and the light source may still cover the camera",
            )
        return Transition("safe", "the flat session ended")

    def task(self) -> FlatTaskView:
        """The progress of the latest task, as `flat_library` reports it."""
        run = self._run
        if run is None:
            return FlatTaskView()
        if run.outcome is not None:
            finished = run.finished_utc_ns
            return self._view_of(
                run,
                state=run.outcome,
                summary=run.summary,
                version=run.version,
                finished_utc=None if finished is None else utc_ns_to_iso(finished, digits=0),
                **self._last_reading(run),
            )
        if not run.started:
            message = HOLD_MESSAGES.get(self._held_by or "", WAIT_MESSAGE)
            return self._view_of(run, state="queued", message=message)
        return self._running(run, self._seconds_running(run))

    # --- Inside ----------------------------------------------------------------------------

    def _seconds_running(self, run: _Run) -> float:
        elapsed = (self._clock.monotonic_ns() - run.submitted_ns) / NS_PER_S
        return elapsed - self.script.queued_s

    def capture_remaining_s(self) -> float | None:
        """The seconds until the end of the frames while the task takes them, or `None`.

        The real scheduler announces this moment as the end of the activity (the session tells the
        length of its frames when it starts them), and the demo does the same.
        """
        run = self._run
        if run is None or run.outcome is not None or not run.started:
            return None
        seconds = self._seconds_running(run)
        _, exposure, capture, _ = self._bounds()
        return capture - seconds if exposure <= seconds < capture else None

    def _bounds(self) -> tuple[float, float, float, float]:
        """The ends of the phases `setup`, `exposure`, `capture`, and `build`, in seconds."""
        script = self.script
        setup = script.setup_s
        exposure = setup + script.exposure_s
        capture = exposure + script.capture_s
        return setup, exposure, capture, capture + script.build_s

    def _outcome(self, seconds: float) -> str | None:
        _, exposure, _, end = self._bounds()
        if self.script.outcome in ("dim", "bright"):
            return "failed" if seconds >= exposure else None
        return "ok" if seconds >= end else None

    def _view_of(self, run: _Run, **fields: Any) -> FlatTaskView:
        """A view of the task with what every state shares: the numbers that the command set."""
        return FlatTaskView(
            task_id=run.task_id,
            set_number=run.set_number,
            frames=run.frames,
            target_fraction=run.target_fraction,
            pause_after=run.pause_after,
            started_utc=utc_ns_to_iso(run.submitted_utc_ns, digits=0),
            **fields,
        )

    @property
    def _rate(self) -> float:
        """The level of the light in the middle of the frame, for each second of exposure."""
        return {"dim": DIM_RATE_PER_S, "bright": BRIGHT_RATE_PER_S}.get(
            self.script.outcome, self.script.rate_per_s
        )

    def _final_exposure_s(self, run: _Run) -> float:
        return max(MIN_EXPOSURE_S, min(MAX_EXPOSURE_S, run.target_fraction / self._rate))

    def _last_reading(self, run: _Run) -> dict[str, Any]:
        """The exposure and the level that the task showed last, which a finished task keeps."""
        if run.outcome == "ok":
            return {
                "exposure_s": self._final_exposure_s(run),
                "level_fraction": run.target_fraction,
            }
        if run.outcome == "failed":
            exposure_s = MAX_EXPOSURE_S if self.script.outcome == "dim" else MIN_EXPOSURE_S
            level = min(1.0, self._rate * exposure_s)
            return {
                "exposure_s": exposure_s,
                "level_fraction": round(level, 3),
                "saturated_fraction": 0.02 if level >= 0.98 else 0.0,
            }
        return {}

    def _running(self, run: _Run, seconds: float) -> FlatTaskView:
        setup, exposure, capture, _ = self._bounds()
        script = self.script
        if seconds < setup:
            step = 0 if seconds < setup / 2 else 1
            message = (
                "Setting up the camera and the library."
                if step == 0
                else "The camera and the library are ready."
            )
            return self._view_of(
                run, state="running", phase="setup", step=step, steps=1, message=message
            )
        if seconds < exposure:
            return self._searching(run, seconds - setup)
        if seconds < capture:
            return self._capturing(run, seconds - exposure)
        return self._view_of(
            run,
            state="running",
            phase="build",
            step=0,
            steps=1,
            message="Combining the frames into a flat.",
            exposure_s=self._final_exposure_s(run),
            level_fraction=run.target_fraction,
            warnings=list(script.warnings),
        )

    def _searching(self, run: _Run, seconds: float) -> FlatTaskView:
        """The search for the exposure: its start, a first try that misses, and tries that close in.

        As in the real session, the phase begins with a report at step 0, before the first try has
        ended, and the tries follow in equal parts of the rest.
        """
        rate = self._rate
        tries = 3 if self.script.outcome in ("dim", "bright") else 2
        step = _step(seconds, self.script.exposure_s, tries + 1) - 1
        if step == 0:
            return self._view_of(
                run,
                state="running",
                phase="exposure",
                step=0,
                steps=8,
                message=(
                    f"Finding the exposure that reaches {run.target_fraction * 100:.0f} % of full "
                    "scale."
                ),
                exposure_s=START_EXPOSURE_S,
            )
        if self.script.outcome == "dim":
            exposures = [START_EXPOSURE_S, 0.4, MAX_EXPOSURE_S]
        elif self.script.outcome == "bright":
            exposures = [START_EXPOSURE_S, 0.0025, MIN_EXPOSURE_S]
        else:
            exposures = [START_EXPOSURE_S, self._final_exposure_s(run)]
        exposure_s = exposures[step - 1]
        level = min(1.0, rate * exposure_s)
        return self._view_of(
            run,
            state="running",
            phase="exposure",
            step=step,
            steps=8,
            message=(
                f"Try {step} of 8: {exposure_text(exposure_s)} gives {level * 100:.0f} % of full "
                f"scale (target {run.target_fraction * 100:.0f} %)."
            ),
            exposure_s=exposure_s,
            level_fraction=round(level, 3),
            saturated_fraction=0.02 if level >= 0.98 else 0.0,
        )

    def _capturing(self, run: _Run, seconds: float) -> FlatTaskView:
        script = self.script
        step = _step(seconds, script.capture_s, run.frames)
        wobble = ((step * 37) % 11 - 5) / 500.0  # a steady light that the noise moves a little
        level = run.target_fraction + wobble
        warnings = list(script.warnings) if seconds >= script.capture_s / 2 else []
        return self._view_of(
            run,
            state="running",
            phase="capture",
            step=step,
            steps=run.frames,
            message=f"Frame {step} of {run.frames}: {level * 100:.0f} % of full scale.",
            exposure_s=self._final_exposure_s(run),
            level_fraction=round(level, 3),
            saturated_fraction=0.0,
            warnings=warnings,
        )

    def _finish(self, run: _Run, outcome: str, summary: str) -> None:
        run.outcome = outcome
        run.summary = summary
        run.finished_utc_ns = self._clock.utc_ns()

    def _complete(self, run: _Run, outcome: str) -> None:
        """End the task as the clock says: a failed light, or a flat for the library."""
        if outcome == "failed":
            summary = DIM_SUMMARY if self.script.outcome == "dim" else BRIGHT_SUMMARY
            self._finish(run, "failed", summary)
            return
        added = self._add_flat(run)
        corner = abs(added.view.corner_percent or 0.0)
        if added.view.second_set:
            text = f"Made the flat {added.view.version} from two sets of frames"
        else:
            text = (
                f"Made the flat {added.view.version} from {run.frames} frames of "
                f"{exposure_text(added.exposure_s)}"
            )
        text += f". The corners get {corner:.0f} % less light than the center"
        text += ". It waits on the Flat page for you to use it or discard it."
        run.version = added.view.version
        self._finish(run, "ok", text)

    def _add_flat(self, run: _Run) -> _Entry:
        now_ns = self._clock.utc_ns()
        exposure_s = self._final_exposure_s(run)
        second = run.set_number == 2 and self._session is not None
        warnings = tuple(self.script.warnings)
        if second:
            assert self._session is not None
            first_version = self._session[0]
            self._entries = [e for e in self._entries if e.view.version != first_version]
            self._jpegs.pop(first_version, None)
            self._session = None
        entry = self._add(
            now_ns,
            self.look,
            state="pending",
            frames=run.frames,
            exposure_s=round(exposure_s, 6),
            target_fraction=run.target_fraction,
            second_set=second,
            warnings=warnings,
        )
        if not second:
            self._session = (
                entry.view.version,
                now_ns,
                run.frames,
                round(exposure_s, 6),
                run.target_fraction,
            )
        return entry


__all__ = [
    "CancelOutcome",
    "FlatLook",
    "FlatScript",
    "FlatSimulator",
    "corner_radius_deg",
    "exposure_text",
    "render_flat_jpeg",
    "vignetting_points",
]
