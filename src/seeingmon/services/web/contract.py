"""What `web` and `core` agree on: the RPC methods, the JSON shapes, and the live-view frames.

`core` serves three channels on its `IpcServer`, and `web` is the client of all of them. This module
holds the names and the codecs, so that both sides build and read the same bytes. It needs no
FastAPI, so `core` can import it.

**Channel `rpc`** (an `RpcService`). Every request is a JSON object, and every answer is a JSON
value. A call that fails raises an exception that the connection layer sends back as an error.

- `ping` takes no parameters and answers `{"instance": "<ID of this core process>"}`. The ID
  changes at a restart.
- `status` takes no parameters and answers `{"instance": ..., "scheduler": {...}}`, where the
  scheduler object holds the fields of `SchedulerStatus` (`encode_status`), including its
  `activity`: what the scheduler does now, and what comes next.
- `submit` takes `{"command": <a command>}` (`encode_command`) and answers with the
  `CommandResult` as JSON (`encode_result`).
- `alignment_state` takes no parameters and answers with the `AlignmentState` as JSON, with
  `"active": false` outside alignment.
- `dark_library` takes no parameters and answers with the `DarkLibraryView` as JSON: the dark
  sets of the library, whether it is due for a new set, the dark model, the sensor temperature,
  and the progress of the latest dark session (`DarkTaskView`).
- `alignment_reset_focus` takes no parameters, restarts the best focus value of the alignment
  (the history of the values stays), and answers `{"reset": true}`.
- `live_seeing` takes no parameters and answers with the `LiveSeeingView` as JSON: the rolling
  seeing value of the fast stream. The answer is `null` while `core` has no value, which is the
  case before the first fast period has gathered `live_min_span_s` seconds of frames.
- `flat_library` takes no parameters and answers with the `FlatLibraryView` as JSON: the flats of
  the library with the numbers of their reports, the flat in use, the flat that waits for a
  decision, the first set that waits for a second set, and the progress of the latest flat
  session (`FlatTaskView`).
- `flat_activate` and `flat_delete` take `{"version": "flat-<8 hex digits>"}` and answer with a
  `FlatActionView`: `ok`, or a `reason` (`unknown`, `active`, `invalid`, or `busy`) and a sentence.
- `flat_image` takes `{"version": ...}` and answers `{"found": bool, "jpeg": "<base64>"}`, the
  preview of a flat as a JPEG of at most `MAX_FLAT_JPEG_BYTES`.

`submit` hands the command to `Scheduler.submit` and answers at once. A rejected command is a
normal answer with `"accepted": false`, and not an error. A command that `decode_command` refuses
is a malformed request: raise the `CodecError` (the connection layer turns it into an
`InvalidParams` error).

**Channel `alignment`** (a `StreamService`). `web` opens one stream at a time, and only while a
person watches the live view. While the scheduler is in `align`, `core` sends one data message for
each frame that it encodes: `pack_frame(state, jpeg)`. A data message holds the magic `SMAF`, the
length of the JSON state (4 bytes, little endian), the JSON state, and then the JPEG to the end of
the message. The state describes the same frame as the JPEG. `core` skips frames when the window is
full, so a slow consumer never makes `core` buffer. While the stream is open, `core` treats the
person as present and calls `Scheduler.touch_alignment` now and then, so the idle timer does not end
`align`. Outside alignment, the stream stays open and sends nothing.

**Channel `polaris`** (a `StreamService`). `web` opens one stream at a time, and only while a person
watches the live video of Polaris. While the fast stream runs, `core` sends one data message for
each frame that it keeps after it thins the stream to `max_fps` frames per second of frame time:
`pack_polaris_frame(state, image)`. A data message holds the magic `SMPF`, the length of the JSON
state (4 bytes, little endian), the JSON state, and then the PNG image to the end of the message.
The image is a lossless 8-bit grayscale PNG with the size of the ROI, so a page can magnify it
without interpolation and show the pixels as they are. The state describes the same frame. `core`
skips frames when the window is full, so a slow consumer never makes `core` buffer, and while no
stream is open `core` copies and encodes nothing. `core` sends `"seq": 0` in the state, because the
hub of `web` numbers the frames that it receives and stamps the number. Only the frames that the
fast analyzer receives go out, so the frames of the survey, the alignment, and the bursts never
reach this channel.

**Commands.** `encode_command` writes a command as `{"type": <name>, ...fields}`. The names are
`start_alignment`, `stop_alignment`, `pause`, `resume`, `queue_burst`, `queue_sweep`,
`queue_replay`, `queue_dark`, `queue_flat`, and `cancel_task`. A field that the command lacks takes
the default of the dataclass. `queue_replay` names its source (`source`) as a recording name
without a directory part, and `core` resolves it under the configured recordings folder.
"""

from __future__ import annotations

import base64
import dataclasses
import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

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
    StopAlignment,
)
from seeingmon.scheduler.status import SchedulerStatus
from seeingmon.services.ipc.codec import (
    CodecError,
    as_mapping,
    decode_stream_config,
    encode_stream_config,
    get_bool,
    get_float,
    get_int,
    get_opt_float,
    get_opt_int,
    get_str,
)

RPC_CHANNEL = "rpc"
ALIGNMENT_CHANNEL = "alignment"
POLARIS_CHANNEL = "polaris"

METHOD_PING = "ping"
METHOD_STATUS = "status"
METHOD_SUBMIT = "submit"
METHOD_ALIGNMENT_STATE = "alignment_state"
METHOD_DARK_LIBRARY = "dark_library"
METHOD_ALIGNMENT_RESET_FOCUS = "alignment_reset_focus"
METHOD_LIVE_SEEING = "live_seeing"
METHOD_FLAT_LIBRARY = "flat_library"
METHOD_FLAT_ACTIVATE = "flat_activate"
METHOD_FLAT_DELETE = "flat_delete"
METHOD_FLAT_IMAGE = "flat_image"
METHODS = (
    METHOD_PING,
    METHOD_STATUS,
    METHOD_SUBMIT,
    METHOD_ALIGNMENT_STATE,
    METHOD_DARK_LIBRARY,
    METHOD_ALIGNMENT_RESET_FOCUS,
    METHOD_LIVE_SEEING,
    METHOD_FLAT_LIBRARY,
    METHOD_FLAT_ACTIVATE,
    METHOD_FLAT_DELETE,
    METHOD_FLAT_IMAGE,
)

FRAME_MAGIC = b"SMAF"
POLARIS_MAGIC = b"SMPF"
MAX_STATE_BYTES = 64 * 1024
JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
POLARIS_IMAGE_TYPE = "image/png"
MAX_LIST_ITEMS = 256
MAX_RAPID_READINGS = 600
MAX_TEXT_CHARS = 4000
MAX_FLAT_JPEG_BYTES = 600_000


# --- Commands --------------------------------------------------------------------------------


def _expect_keys(
    data: Mapping[str, Any], what: str, required: Iterable[str], optional: Iterable[str] = ()
) -> None:
    required_set, optional_set = set(required), set(optional)
    missing = required_set - data.keys()
    if missing:
        raise CodecError(f"{what} lacks {', '.join(sorted(missing))}")
    unknown = data.keys() - required_set - optional_set
    if unknown:
        raise CodecError(f"{what} has unknown fields: {', '.join(sorted(unknown))}")


def _number_list(data: Mapping[str, Any], key: str, what: str, *, integer: bool) -> tuple[Any, ...]:
    value = data.get(key)
    if not isinstance(value, list) or len(value) > MAX_LIST_ITEMS:
        raise CodecError(f"{what}.{key} must be a list of at most {MAX_LIST_ITEMS} numbers")
    read = get_int if integer else get_float
    return tuple(read({"value": item}, "value", f"{what}.{key}") for item in value)


def _text_list(data: Mapping[str, Any], key: str, what: str) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or len(value) > MAX_LIST_ITEMS:
        raise CodecError(f"{what}.{key} must be a list of at most {MAX_LIST_ITEMS} strings")
    if not all(isinstance(item, str) for item in value):
        raise CodecError(f"{what}.{key} must be a list of strings")
    return tuple(value)


def encode_command(command: Command) -> dict[str, Any]:
    """A scheduler command as the JSON object that `submit` carries. Raises `TypeError`."""
    if isinstance(command, StartAlignment):
        return {"type": "start_alignment", "exposure_s": command.exposure_s, "gain": command.gain}
    if isinstance(command, StopAlignment):
        return {"type": "stop_alignment"}
    if isinstance(command, Pause):
        return {"type": "pause"}
    if isinstance(command, Resume):
        return {"type": "resume"}
    if isinstance(command, QueueBurst):
        return {
            "type": "queue_burst",
            "duration_s": command.duration_s,
            "stream": None if command.stream is None else encode_stream_config(command.stream),
            "label": command.label,
            "priority": command.priority,
        }
    if isinstance(command, QueueSweep):
        return {
            "type": "queue_sweep",
            "exposure_us": list(command.exposure_us),
            "gain": list(command.gain),
            "roi_arcmin": list(command.roi_arcmin),
            "modes": list(command.modes),
            "window_s": command.window_s,
            "priority": command.priority,
        }
    if isinstance(command, QueueReplay):
        return {
            "type": "queue_replay",
            "source": command.source,
            "speed": command.speed,
            "options": dict(command.options),
            "priority": command.priority,
        }
    if isinstance(command, QueueDark):
        return {
            "type": "queue_dark",
            "exposure_s": command.exposure_s,
            "frames": command.frames,
            "bias_frames": command.bias_frames,
            "wait_for_cover": command.wait_for_cover,
            "pause_after": command.pause_after,
            "label": command.label,
            "priority": command.priority,
            "wait_for_cover_timeout_s": command.wait_for_cover_timeout_s,
            "immediate": command.immediate,
        }
    if isinstance(command, QueueFlat):
        return {
            "type": "queue_flat",
            "frames": command.frames,
            "target_fraction": command.target_fraction,
            "set_number": command.set_number,
            "pause_after": command.pause_after,
            "priority": command.priority,
            "immediate": command.immediate,
        }
    if isinstance(command, CancelTask):
        return {"type": "cancel_task", "kind": command.kind}
    raise TypeError(f"cannot send {type(command).__name__} to core")


def decode_command(value: Any) -> Command:
    """The inverse of `encode_command`. Raises `CodecError` for anything malformed."""
    data = as_mapping(value, "command")
    kind = get_str(data, "type", "command")
    what = f"command {kind}"
    body = {key: item for key, item in data.items() if key != "type"}
    if kind == "start_alignment":
        _expect_keys(body, what, (), ("exposure_s", "gain"))
        return StartAlignment(
            exposure_s=get_opt_float(body, "exposure_s", what),
            gain=get_opt_int(body, "gain", what),
        )
    if kind in ("stop_alignment", "pause", "resume"):
        _expect_keys(body, what, ())
        return {"stop_alignment": StopAlignment, "pause": Pause, "resume": Resume}[kind]()
    if kind == "queue_burst":
        _expect_keys(body, what, (), ("duration_s", "stream", "label", "priority"))
        stream = body.get("stream")
        return QueueBurst(
            duration_s=get_float(body, "duration_s", what) if "duration_s" in body else 10.0,
            stream=None if stream is None else decode_stream_config(stream),
            label=get_str(body, "label", what) if "label" in body else "",
            priority=get_int(body, "priority", what) if "priority" in body else 0,
        )
    if kind == "queue_sweep":
        _expect_keys(
            body, what, (), ("exposure_us", "gain", "roi_arcmin", "modes", "window_s", "priority")
        )
        return QueueSweep(
            exposure_us=(
                _number_list(body, "exposure_us", what, integer=True)
                if "exposure_us" in body
                else ()
            ),
            gain=_number_list(body, "gain", what, integer=True) if "gain" in body else (),
            roi_arcmin=(
                _number_list(body, "roi_arcmin", what, integer=False)
                if "roi_arcmin" in body
                else ()
            ),
            modes=_text_list(body, "modes", what) if "modes" in body else (),
            window_s=get_opt_float(body, "window_s", what),
            priority=get_int(body, "priority", what) if "priority" in body else 0,
        )
    if kind == "queue_replay":
        _expect_keys(body, what, (), ("source", "speed", "options", "priority"))
        options = body.get("options")
        return QueueReplay(
            source=get_str(body, "source", what) if "source" in body else "",
            speed=get_float(body, "speed", what) if "speed" in body else 1.0,
            options={} if options is None else dict(as_mapping(options, f"{what}.options")),
            priority=get_int(body, "priority", what) if "priority" in body else 0,
        )
    if kind == "queue_dark":
        _expect_keys(
            body,
            what,
            (),
            (
                "exposure_s",
                "frames",
                "bias_frames",
                "wait_for_cover",
                "pause_after",
                "label",
                "priority",
                "wait_for_cover_timeout_s",
                "immediate",
            ),
        )
        return QueueDark(
            exposure_s=get_opt_float(body, "exposure_s", what),
            frames=get_opt_int(body, "frames", what),
            bias_frames=get_opt_int(body, "bias_frames", what),
            wait_for_cover=(
                get_bool(body, "wait_for_cover", what) if "wait_for_cover" in body else True
            ),
            pause_after=get_bool(body, "pause_after", what) if "pause_after" in body else True,
            label=get_str(body, "label", what) if "label" in body else "",
            priority=get_int(body, "priority", what) if "priority" in body else 0,
            wait_for_cover_timeout_s=get_opt_float(body, "wait_for_cover_timeout_s", what),
            immediate=get_bool(body, "immediate", what) if "immediate" in body else True,
        )
    if kind == "queue_flat":
        _expect_keys(
            body,
            what,
            (),
            ("frames", "target_fraction", "set_number", "pause_after", "priority", "immediate"),
        )
        defaults = QueueFlat()
        return QueueFlat(
            frames=get_int(body, "frames", what) if "frames" in body else defaults.frames,
            target_fraction=(
                get_float(body, "target_fraction", what)
                if "target_fraction" in body
                else defaults.target_fraction
            ),
            set_number=(
                get_int(body, "set_number", what) if "set_number" in body else defaults.set_number
            ),
            pause_after=(get_bool(body, "pause_after", what) if "pause_after" in body else True),
            priority=get_int(body, "priority", what) if "priority" in body else 0,
            immediate=get_bool(body, "immediate", what) if "immediate" in body else True,
        )
    if kind == "cancel_task":
        _expect_keys(body, what, ("kind",))
        return CancelTask(kind=get_str(body, "kind", what))
    raise CodecError("command.type is not a command that core accepts")


def encode_result(result: CommandResult) -> dict[str, Any]:
    """A `CommandResult` as the JSON object that `submit` answers with."""
    return {
        "accepted": result.accepted,
        "message": result.message,
        "state": result.state,
        "reason": None if result.reason is None else result.reason.value,
        "task_id": result.task_id,
    }


def decode_result(value: Any) -> CommandResult:
    """The inverse of `encode_result`. Raises `CodecError` for anything malformed."""
    data = as_mapping(value, "command result")
    _expect_keys(data, "command result", ("accepted", "message", "state"), ("reason", "task_id"))
    accepted = data.get("accepted")
    if not isinstance(accepted, bool):
        raise CodecError("command result.accepted must be true or false")
    reason = data.get("reason")
    try:
        parsed = None if reason is None else RejectReason(reason)
    except ValueError:
        raise CodecError("command result.reason is not a known reason") from None
    return CommandResult(
        accepted=accepted,
        message=get_str(data, "message", "command result")[:MAX_TEXT_CHARS],
        state=get_str(data, "state", "command result"),
        reason=parsed,
        task_id=get_opt_int(data, "task_id", "command result"),
    )


# --- Status ----------------------------------------------------------------------------------


class _View(BaseModel):
    """A frozen model that ignores unknown fields, so a newer `core` can serve an older `web`.

    The model is strict: it rejects a string where a number belongs and a number where a boolean
    belongs, because the bytes come from another process.
    """

    model_config = ConfigDict(frozen=True, extra="ignore", strict=True, allow_inf_nan=False)


class RoiView(_View):
    x: int
    y: int
    width: int
    height: int


class StreamView(_View):
    """The stream that the camera runs or ran last."""

    stream_id: int
    purpose: str
    mode: str
    exposure_us: int
    gain: int
    roi: RoiView | None = None


class FaultView(_View):
    """Where the scheduler stands in a fault episode (`FaultStatus`).

    `cause` is `timeout`, `disconnected`, `link`, or `error`, and `reason` says it in words. Both
    are `None` without an episode. `since_utc_ns` is when the episode began.
    """

    failures: int = 0
    good_frames: int = 0
    last_error: str | None = None
    next_attempt_utc_ns: int | None = None
    next_step: str | None = None
    cause: str | None = None
    reason: str | None = None
    since_utc_ns: int | None = None


class ActivityView(_View):
    """What the scheduler does now, for how long, and what comes next (`ActivityStatus`).

    `phase` is one of `fast`, `survey_short`, `survey_long`, `solve_wait`, `idle`, `watch`,
    `align`, `commission`, `paused`, and `camera_fault`. Times are nanoseconds since the Unix
    epoch, in UTC. The text fields are plain words, and a value that the scheduler does not know
    is `None`.
    """

    state: str
    phase: str
    label: str
    since_utc_ns: int
    ends_utc_ns: int | None = None
    next_label: str | None = None
    next_utc_ns: int | None = None
    cadence_s: float | None = None
    detail: str | None = None
    reason: str | None = None


class SchedulerView(_View):
    """The fields of `SchedulerStatus` as JSON. Every field is plain, so `asdict` fills it."""

    t_utc_ns: int
    state: str
    state_reason: str = ""
    state_since_utc_ns: int
    last_transition_utc_ns: int | None = None
    degraded: bool = False
    stream: StreamView | None = None
    cloud: bool = False
    cloud_fraction: float | None = None
    twilight: bool = False
    sun_elevation_deg: float | None = None
    background_fraction: float | None = None
    sensor_temperature_c: float | None = None
    counters: dict[str, int] = Field(default_factory=dict)
    fault: FaultView = Field(default_factory=FaultView)
    queued_tasks: int = 0
    survey_pending: int = 0
    alignment_idle_s: float | None = None
    activity: ActivityView | None = None


class CoreStatus(_View):
    """The answer of the `status` method."""

    instance: str
    scheduler: SchedulerView


def encode_status(status: SchedulerStatus, instance: str) -> dict[str, Any]:
    """The answer of the `status` method: the process identity and the scheduler status."""
    return {"instance": instance, "scheduler": dataclasses.asdict(status)}


def decode_status(value: Any) -> CoreStatus:
    """The inverse of `encode_status`. Raises `CodecError` for anything malformed."""
    try:
        return CoreStatus.model_validate(value)
    except ValidationError as error:
        fields = ", ".join(
            sorted({".".join(str(part) for part in e["loc"]) for e in error.errors()})
        )
        raise CodecError(f"the status is not valid: {fields}") from None


# --- Alignment -------------------------------------------------------------------------------


class AlignmentFrameInfo(_View):
    """The frame that the state describes. Coordinates elsewhere are pixels of this frame."""

    seq: int = Field(ge=0)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    readout_mode: str = Field("", max_length=64)
    exposure_s: float | None = Field(None, gt=0)
    gain: int | None = Field(None, ge=0)
    plate_scale_arcsec_px: float | None = Field(None, gt=0)


class TargetView(_View):
    """Where the star belongs: the position of Polaris in the reference solution."""

    x_px: float
    y_px: float
    roll_deg: float | None = None


class SolvedView(_View):
    """Where the latest solution puts the star, and how good the solution is."""

    x_px: float
    y_px: float
    roll_deg: float | None = None
    n_matched: int = Field(0, ge=0)
    rms_arcsec: float | None = Field(None, ge=0)
    age_s: float | None = Field(None, ge=0)


class OffsetView(_View):
    """The solved position minus the target, in pixels and in arcseconds."""

    dx_px: float
    dy_px: float
    distance_px: float = Field(ge=0)
    dx_arcsec: float | None = None
    dy_arcsec: float | None = None
    distance_arcsec: float | None = Field(None, ge=0)
    roll_deg: float | None = None


class FocusHistoryView(_View):
    """The last values of the focus measure, oldest first, as parallel lists of one length.

    `core` keeps the last 120 values with the capture times of their frames, so a page that
    reloads finds the whole curve at once. Parallel lists keep the message small: 120 values take
    about 5 kilobytes in the state of `GET /alignment/state`. The live view sends the whole history
    in the first message of a viewer (`reset` is `true`), and after that only the points that this
    viewer lacks (`reset` is `false`). One rule serves every message: if `reset` is `true`, throw
    away the points that you hold, then append the points of the message, then keep the newest 120.
    `session` changes when the history restarts (a new alignment), and `index` counts the points of
    a session from 1, so a reader can tell new points from old ones.
    """

    session: int = Field(
        ge=0, description="Changes when the history restarts, for example in a new alignment."
    )
    reset: bool = Field(
        True,
        description="`true`: discard the points that you hold, then append these. `false`: append "
        "these to the points that you hold.",
    )
    index: list[int] = Field(
        default_factory=list,
        max_length=MAX_LIST_ITEMS,
        description="The number of each point in its session, from 1. It grows by one for each "
        "point and never repeats.",
    )
    seq: list[int] = Field(
        default_factory=list,
        max_length=MAX_LIST_ITEMS,
        description="The sequence number of the frame that each value was measured in.",
    )
    t_utc_ms: list[int] = Field(
        default_factory=list,
        max_length=MAX_LIST_ITEMS,
        description="The capture time of that frame, in milliseconds since the Unix epoch (UTC).",
    )
    fwhm_px: list[float] = Field(
        default_factory=list,
        max_length=MAX_LIST_ITEMS,
        description="The median FWHM of the usable stars of that frame, in pixels of the readout "
        "mode.",
    )
    fwhm_arcsec: list[float | None] = Field(
        default_factory=list,
        max_length=MAX_LIST_ITEMS,
        description="The same value in arcseconds, through the plate scale of the frame (3.82 "
        "arcseconds per pixel in bin2). An entry is `null` when the plate scale is not known.",
    )
    n_stars: list[int] = Field(
        default_factory=list,
        max_length=MAX_LIST_ITEMS,
        description="The number of stars that each value rests on.",
    )
    spike: list[bool] = Field(
        default_factory=list,
        max_length=MAX_LIST_ITEMS,
        description="`true` for a value that exceeds twice the median of the preceding ten "
        "values. Touching the telescope inflates the width of the stars for a moment, and a page "
        "leaves such a value out of its scale. A spike never sets the best value.",
    )

    @model_validator(mode="after")
    def _lists_have_one_length(self) -> FocusHistoryView:
        lengths = {
            len(self.index),
            len(self.seq),
            len(self.t_utc_ms),
            len(self.fwhm_px),
            len(self.fwhm_arcsec),
            len(self.n_stars),
            len(self.spike),
        }
        if len(lengths) != 1:
            raise ValueError("the lists of the focus history differ in length")
        return self


class FocusView(_View):
    """The focus measure: the median FWHM of the usable stars of one frame, and a short history.

    A star is usable when it is not saturated, not at the edge, not blended, and bright enough to
    measure. The value is the median of their widths across the trail, in pixels of the readout
    mode (3.82 arcseconds per pixel in bin2), and it has no smoothing. `fwhm_px`, `n_stars`, and
    `spike` describe the frame of the latest finished solve (`frame_seq`). They are `null` when
    that frame had too few usable stars, while the best value and the history remain.

    `best_fwhm_px` is the smallest value of the session that is not a spike. A reset
    (`POST /alignment/focus/reset`) restarts it, for example after a refocus, and the history stays.
    """

    fwhm_px: float | None = Field(
        None, ge=0, description="The focus value of the latest solve, in pixels."
    )
    best_fwhm_px: float | None = Field(
        None,
        ge=0,
        description="The smallest value since the session began or since the last reset, in "
        "pixels. A spike never sets it.",
    )
    n_stars: int | None = Field(
        None, ge=0, description="The number of stars that `fwhm_px` rests on."
    )
    fwhm_arcsec: float | None = Field(
        None, ge=0, description="`fwhm_px` in arcseconds, through the plate scale of the frame."
    )
    best_fwhm_arcsec: float | None = Field(None, ge=0, description="`best_fwhm_px` in arcseconds.")
    spike: bool = Field(
        False,
        description="`true` when `fwhm_px` exceeds twice the median of the preceding ten values: "
        "a spike, such as the moment after you touch the telescope. A page leaves a spike out of "
        "its scale.",
    )
    frame_seq: int | None = Field(
        None, ge=0, description="The sequence number of the frame that `fwhm_px` was measured in."
    )
    history: FocusHistoryView | None = Field(
        None,
        description="The last 120 values, oldest first. It is `null` until the first value of "
        "the session.",
    )


class HistogramView(_View):
    """Counts of pixels in equal bins from `min_dn` to `max_dn`. The UI draws them on a log axis."""

    counts: list[int] = Field(max_length=MAX_LIST_ITEMS)
    min_dn: float = 0.0
    max_dn: float = Field(gt=0)


class SaturationView(_View):
    """The share of saturated pixels, and whether it is enough to warn about."""

    fraction: float = Field(ge=0, le=1)
    warning: bool = False


class CameraView(_View):
    """The camera model of the frame, so that a page projects sky points like the solver does.

    `rotation` holds nine numbers, row by row: the rotation from CIRS (the apparent frame of
    date) to the camera frame. A sky point `u` projects to `w = R u`, `x = center_x_px +
    (w_x / w_z) / s`, and `y = center_y_px + parity * (w_y / w_z) / s`, where `s` is
    `scale_arcsec_px` in radians per pixel. A point with `w_z` of 0.05 or less is not in front of
    the camera. Pixels follow the frame of `AlignmentFrameInfo`, with the center of the first
    pixel at 0.
    """

    rotation: list[float] = Field(min_length=9, max_length=9)
    scale_arcsec_px: float = Field(gt=0)
    parity: Literal[1, -1]
    center_x_px: float
    center_y_px: float


class PoleView(_View):
    """Where the celestial pole of date falls. Offsets are the pole minus the frame center.

    `x_px`, `y_px`, `dx_px`, `dy_px`, and `distance_px` are `null` when the pole lies behind the
    camera. `distance_arcmin` is the exact angle between the frame center and the pole. `roll_deg`
    is the position angle of the direction to the pole, from image up toward image left, and it
    is `null` within one pixel of the center.
    """

    in_front: bool
    x_px: float | None = None
    y_px: float | None = None
    inside_frame: bool = False
    dx_px: float | None = None
    dy_px: float | None = None
    distance_px: float | None = Field(None, ge=0)
    distance_arcmin: float | None = Field(None, ge=0)
    roll_deg: float | None = None


class OrbitView(_View):
    """How the circle that Polaris follows around the pole sits in the frame.

    `margin_px` is the distance from the circle to the nearest frame edge. It is negative when
    the circle leaves the frame, and `fits` says the same in one flag.
    """

    fits: bool
    margin_px: float
    margin_arcmin: float | None = None


class PointView(_View):
    """A pixel of the frame."""

    x_px: float
    y_px: float


class ReticleView(_View):
    """The fixed reticle: a circle that does not move with the mount.

    It is centered on the aim (the center of the frame, unless `[alignment]` names another pixel),
    and its radius is the radius of the orbit of Polaris in pixels of the frame:
    `tan(colatitude) / scale_rad_px`. It depends on the frame size, the plate scale, and the
    colatitude of Polaris only, so `core` serves it without a solution too.
    """

    x_px: float
    y_px: float
    radius_px: float = Field(gt=0)
    polaris_colatitude_deg: float | None = Field(None, ge=0, lt=90)


class AimView(_View):
    """Where the pole should go, and how far it is from there.

    `x_px` and `y_px` are the aim, which is the center of the reticle. `dx_px` and `dy_px` are the
    pole minus the aim, and they are `null` when the pole lies behind the camera.
    `distance_arcmin` is the exact angle between the pole and the sky direction at the aim pixel.
    """

    x_px: float
    y_px: float
    dx_px: float | None = None
    dy_px: float | None = None
    distance_arcmin: float | None = Field(None, ge=0)


class AxesView(_View):
    """The image directions of the two moves of the mount, as unit vectors (x right, y down).

    `altitude_*` is where the camera looks when you raise it, and `azimuth_*` is where it looks
    when you turn it toward the east. The pole lies on the side of the aim in which the camera has
    to move, so the two vectors read like the arrows of a compass drawn on the picture.
    """

    altitude_dx: float
    altitude_dy: float
    azimuth_dx: float
    azimuth_dy: float


class SkyView(_View):
    """The sky layer of the live view: the camera, the pole, the aim, and the move to make.

    Everything here is fixed to the stars and comes from the solution of the very frame, so it
    moves in the picture as the mount moves. Coordinates are of date. The whole view takes about a
    kilobyte, because the state travels with every frame.

    `polaris_colatitude_deg` is the angle between Polaris and the pole at the time of the frame,
    which is the radius of the orbit, and `orbit` says whether that circle fits in the frame
    (`null` without the colatitude). `aim` says where the pole should go and how far it is from
    there. `aim_ring` is where Polaris belongs now: the pixel on the circle of the reticle that the
    real Polaris would take if the pole sat at the aim, with the same roll and at the same time.
    It is the detected Polaris pixel plus the aim minus the detected pole, because the moves of an
    altitude-azimuth mount translate the picture.

    `altitude_arcmin` and `azimuth_arcmin` are the move of the camera's pointing that brings the
    pole to the aim: positive altitude means raise the camera (toward the zenith), and positive
    azimuth means turn it toward the east. They are arcs on the sky, so a turn of the azimuth axis
    is the azimuth arc divided by the cosine of the altitude of the camera. Both are `null` when
    the site is not configured, and when the zenith lies within about a degree of the aim, where
    the directions of the two moves are not defined. `axes` holds the image directions of the two
    moves, under the same conditions.
    """

    camera: CameraView
    pole: PoleView
    polaris_colatitude_deg: float | None = Field(None, ge=0, lt=90)
    orbit: OrbitView | None = None
    aim: AimView | None = None
    aim_ring: PointView | None = None
    altitude_arcmin: float | None = None
    azimuth_arcmin: float | None = None
    axes: AxesView | None = None

    @classmethod
    def from_geometry(cls, geometry: Any) -> SkyView:
        """Wrap the `SkyGeometry` that `seeingmon.survey.skyview.build_sky_view` returns."""
        return cls.model_validate(dataclasses.asdict(geometry))


class AimRingView(_View):
    """The aim ring: where Polaris belongs on the circle of the reticle for the time of this frame.

    It is the pixel that the real Polaris would take if the pole sat at the aim, with the same
    orientation of the picture and at the time of this frame. The orientation of the picture
    (the twist about the optical axis) and the time fix it. The moves of an altitude-azimuth mount
    translate the picture and leave the twist alone, so the ring stays right while you adjust the
    mount, even when the solver finds no star field. `x_px` and `y_px` are pixels of the frame in
    `frame`, and the ring lies on the circle of `reticle`.

    `source` says where the ring comes from. `current frame` means the solution of this very state
    (it equals `sky.aim_ring`). `last solution` means that the latest solve failed or is too old, so
    `core` turned the last good solution to the time of this frame (the Earth turns the picture
    about the pole by 15 degrees an hour) and took the ring from that. The pole, the polar grid, and
    the move in altitude and azimuth need the pointing of this frame, so they stay out of a state
    that has no current solution.
    """

    x_px: float = Field(description="The x position of the ring, in pixels of the frame.")
    y_px: float = Field(description="The y position of the ring, in pixels of the frame.")
    source: Literal["current frame", "last solution"] = Field(
        description="Whether the ring comes from the solution of this state or from the last "
        "good solution."
    )
    age_s: float | None = Field(
        None,
        ge=0,
        description="The time from the frame of the solution to the frame of this state, in "
        "seconds. It is small for `current frame` and grows while the solver fails.",
    )
    solution_frame_seq: int | None = Field(
        None, ge=0, description="The sequence number of the frame that the solution came from."
    )


class LastSolutionView(_View):
    """The last good solution of this alignment: the latest solve that found the star field.

    It stays while later solves fail, so the page can say how old the overlay is. `core` forgets it
    when the alignment ends. A good solution may be the current one: then `age_s` is small and
    `solved` holds the same numbers.
    """

    frame_seq: int = Field(ge=0, description="The sequence number of the frame that was solved.")
    t_utc: str = Field(max_length=40, description="The capture time of that frame, ISO 8601 UTC.")
    age_s: float = Field(
        ge=0,
        description="The time from that frame to the frame of this state, in seconds.",
    )
    roll_deg: float | None = Field(
        None,
        description="The position angle of the direction to the pole in that solution, from "
        "image up toward image left. It is `null` when the pole sat on the center.",
    )
    polaris_colatitude_deg: float | None = Field(
        None,
        ge=0,
        lt=90,
        description="The angle between Polaris and the pole at the time of that frame, in "
        "degrees: the radius of the orbit.",
    )
    n_matched: int = Field(0, ge=0, description="The number of catalog stars that the fit matched.")
    rms_arcsec: float | None = Field(
        None, ge=0, description="The residual of the fit, in arcseconds."
    )
    solver: str = Field(
        "", max_length=32, description="What solved the frame: `tracker`, or the plate solver."
    )


class TimingView(_View):
    """Where the time goes between the camera and the page, in seconds, for one state.

    The state describes one frame, and the quick solve works on its own frame, which is usually
    an older one: the solver takes the newest frame when it is free, and a solve runs while newer
    frames arrive. `frame_seq` and `solution_frame_seq` say which two frames the state joins, and
    the ages say how stale each part is. A time that `core` cannot know is `null`: a frame with
    no valid time (the clock was not synchronized) has no age.
    """

    frame_seq: int = Field(
        ge=0, description="The sequence number of the frame that the picture and `frame` show."
    )
    frame_t_utc: str | None = Field(
        None,
        max_length=40,
        description="The capture time of that frame (the middle of its exposure), ISO 8601 UTC. "
        "It equals `t_utc` of the state.",
    )
    frame_age_s: float | None = Field(
        None,
        ge=0,
        description="The age of the frame when `core` sent this state: the time from its capture "
        "to the send, which covers the transfer from the camera process, the wait for the "
        "encoder, and the encoding. The state of `GET /alignment/state` has the age at the "
        "request. It is `null` for a frame without a valid time.",
    )
    receive_lag_s: float | None = Field(
        None,
        ge=0,
        description="The time from the capture of the frame to its arrival in `core`: the "
        "transfer from the camera process and the wait for the scheduler thread. It is `null` "
        "when `core` did not time the arrival, and for a frame without a valid time.",
    )
    preview_s: float | None = Field(
        None,
        ge=0,
        description="The time that `core` needed for this frame, from its arrival to the finished "
        "preview and state: the wait for the encoder plus the stretch, the histogram, and the "
        "JPEG. It is `null` when `core` did not time the arrival.",
    )
    solution_frame_seq: int | None = Field(
        None,
        ge=0,
        description="The sequence number of the frame that the latest finished solve ran on, "
        "solved or not. `solved`, `offset`, `sky`, and `focus` come from that frame, and "
        "`frame_seq` minus this number says how many frames the solve lags. It is `null` until a "
        "solve has finished.",
    )
    solve_elapsed_s: float | None = Field(
        None,
        ge=0,
        description="How long the latest finished solve took, from the start of its work to its "
        "result, including the transfer of the frame to the solver process. It is `null` until a "
        "solve has finished.",
    )
    solving_frame_seq: int | None = Field(
        None,
        ge=0,
        description="The sequence number of the frame that the solver works on now, or `null` "
        "while it idles. It is `null` too when the solver is not available.",
    )
    solving_s: float | None = Field(
        None,
        ge=0,
        description="How long the solve in progress has run, or `null` while the solver idles.",
    )


class RapidReadingsView(_View):
    """The last readings of the rapid focus mode, oldest first, as parallel lists of one length.

    A reading is the median width of the star over the frames of an interval of 50 milliseconds
    (about four frames at 82 frames a second), so there are 20 readings a second. `core` keeps the
    last 600 of them, which is 30 seconds. One rule serves every message: if `reset` is `true`,
    throw away the readings that you hold, then append the readings of the message, then keep the
    newest 600. `web` sends the whole history in the first message of a viewer (`reset` is `true`),
    and after that only the readings that this viewer lacks (`reset` is `false`). `session`
    changes when a new rapid focus session starts, and `index` counts the readings of a session
    from 1, so a reader can tell new readings from old ones.
    """

    session: int = Field(
        ge=0, description="Changes when a new rapid focus session starts, with an empty history."
    )
    reset: bool = Field(
        True,
        description="`true`: discard the readings that you hold, then append these. `false`: "
        "append these to the readings that you hold.",
    )
    index: list[int] = Field(
        default_factory=list,
        max_length=MAX_RAPID_READINGS,
        description="The number of each reading in its session, from 1. It grows by one for each "
        "reading and never repeats.",
    )
    t_utc_ms: list[int] = Field(
        default_factory=list,
        max_length=MAX_RAPID_READINGS,
        description="The middle of the interval that the reading covers, in milliseconds since "
        "the Unix epoch (UTC), in the time of the frames.",
    )
    fwhm_arcsec: list[float] = Field(
        default_factory=list,
        max_length=MAX_RAPID_READINGS,
        description="The width of the star in arcseconds: the median over the frames of the "
        "interval of the mean of the two second-moment widths, times 2.355, through the plate "
        "scale of the readout mode (1.91 arcseconds per pixel in bin1). It is the width that "
        "the stored windows call `width_fwhm_arcsec`.",
    )
    peak_fraction: list[float] = Field(
        default_factory=list,
        max_length=MAX_RAPID_READINGS,
        description="The brightest pixel of the star in the interval, as a share of the full "
        "scale of the ADC.",
    )
    n_frames: list[int] = Field(
        default_factory=list,
        max_length=MAX_RAPID_READINGS,
        description="The number of frames that the reading rests on.",
    )
    spike: list[bool] = Field(
        default_factory=list,
        max_length=MAX_RAPID_READINGS,
        description="`true` for a reading that exceeds twice the median of the preceding ten "
        "readings, such as the moment after you touch the telescope. A page leaves a spike out "
        "of its scale, and a spike never sets the best value.",
    )
    saturated: list[bool] = Field(
        default_factory=list,
        max_length=MAX_RAPID_READINGS,
        description="`true` when the star reached the saturation level in a frame of the "
        "interval. A saturated star reads too narrow, so such a reading never sets the best "
        "value.",
    )

    @model_validator(mode="after")
    def _lists_have_one_length(self) -> RapidReadingsView:
        lengths = {
            len(self.index),
            len(self.t_utc_ms),
            len(self.fwhm_arcsec),
            len(self.peak_fraction),
            len(self.n_frames),
            len(self.spike),
            len(self.saturated),
        }
        if len(lengths) != 1:
            raise ValueError("the lists of the rapid focus readings differ in length")
        return self


RapidLocatedBy = Literal["current solution", "last solution", "brightest star"]


class RapidFocusView(_View):
    """The rapid focus mode: whether it is offered, and, while it runs, what it measures.

    In the rapid focus mode the camera streams the fast readout mode on a small ROI around Polaris
    (128 by 128 pixels in bin1, 1.91 arcseconds per pixel) at the camera rate, and `core` reports
    the width of the star in arcseconds 20 times a second, with the live video of the star on the
    `polaris` channel. Only one star counts: Polaris.

    **Offered.** `available` is `true` while the alignment runs, the coarse focus is good enough
    (the stars of the normal view are at most `max_fwhm_arcsec` wide), and `core` knows where
    Polaris is (`located_by` says how). Otherwise `reason` says in words what is missing. While the
    mode runs, `available` stays `true`, because a start request then restarts its idle timer.
    `POST /alignment/rapid-focus/start` takes no position: `core` places the ROI by itself.

    **Running.** `active` is `true` from the start to the end of the mode. `since_utc` is the
    start, and `mode`, `exposure_us`, `gain`, `roi`, and `scale_arcsec_px` describe the stream. The
    value of the newest reading is `fwhm_arcsec`, and `best_fwhm_arcsec` is the lowest smoothed
    value of the session (the lowest median of ten to twenty consecutive readings that are not
    spikes and not saturated). `n_stars` is 1 while the star shows in the frames, and 0 while it
    does not. `readings` holds the history. A mode that ended on its own says why in
    `ended_reason` until the next session starts.
    """

    available: bool
    reason: str | None = Field(
        None, max_length=MAX_TEXT_CHARS, description="What is missing, in words, or `null`."
    )
    located_by: RapidLocatedBy | None = Field(
        None, description="What tells `core` where Polaris is. It is `null` while it does not know."
    )
    coarse_fwhm_arcsec: float | None = Field(
        None,
        ge=0,
        description="The coarse focus: the median of the last five focus values of the normal "
        "view, in arcseconds.",
    )
    max_fwhm_arcsec: float | None = Field(
        None, gt=0, description="The widest coarse focus that offers the mode, in arcseconds."
    )
    active: bool = False
    ended_reason: str | None = Field(
        None, max_length=MAX_TEXT_CHARS, description="Why the last session ended on its own."
    )
    since_utc: str | None = Field(None, max_length=40, description="When the session started.")
    mode: str | None = Field(None, max_length=64)
    exposure_us: int | None = Field(None, ge=0)
    gain: int | None = Field(None, ge=0)
    roi: RoiView | None = None
    scale_arcsec_px: float | None = Field(None, gt=0)
    n_stars: int | None = Field(None, ge=0, description="The number of stars that count.")
    fwhm_arcsec: float | None = Field(None, ge=0, description="The newest reading.")
    best_fwhm_arcsec: float | None = Field(None, ge=0, description="The best value of the session.")
    peak_fraction: float | None = Field(None, ge=0, description="The peak of the newest reading.")
    spike: bool = Field(False, description="`true` when the newest reading is a spike.")
    saturated: bool = Field(
        False, description="`true` when the newest reading rests on a saturated star."
    )
    readings: RapidReadingsView | None = Field(
        None, description="The last 600 readings. `null` while the mode does not run."
    )
    quality: dict[str, str] = Field(default_factory=dict, max_length=32)


class AlignmentState(_View):
    """What the Align page shows next to the live view. The answer of `alignment_state`.

    Every part is `null` when `core` does not know it yet, and `quality` says why. `t_utc` is an
    ISO 8601 UTC time. Positions are in pixels of the frame in `frame`, so the UI scales them to
    the size of the image that it shows. `reticle` is the fixed circle of the first layer, and it
    exists without a solution. `sky` is the layer that is fixed to the stars: the pole, the aim, the
    orbit of Polaris, and the camera model of the latest current solution. It exists whether or not
    a target is set. `timing` says how old the frame and the solution are. `aim_ring` is where
    Polaris belongs on the circle of the reticle, and it exists while the state has no current
    solution, when it comes from `last_solution`. `rapid_focus` says whether the rapid focus mode
    is offered, and what it measures while it runs.
    """

    active: bool = False
    t_utc: str | None = Field(None, max_length=40)
    frame: AlignmentFrameInfo | None = None
    target: TargetView | None = None
    solved: SolvedView | None = None
    offset: OffsetView | None = None
    focus: FocusView | None = None
    histogram: HistogramView | None = None
    saturation: SaturationView | None = None
    reticle: ReticleView | None = None
    sky: SkyView | None = None
    aim_ring: AimRingView | None = Field(
        None,
        description="Where Polaris belongs on the circle of the reticle for the time of this "
        "frame, from the current solution or, without one, from the last good solution. It is "
        "`null` when no solution has found the star field in this alignment.",
    )
    last_solution: LastSolutionView | None = Field(
        None,
        description="The latest solve that found the star field, whether or not it is current. "
        "It is `null` until a solve succeeds, and after the alignment ends.",
    )
    timing: TimingView | None = Field(
        None,
        description="The age of the frame, and the frame and the time of the quick solve. It is "
        "`null` until a frame has arrived.",
    )
    rapid_focus: RapidFocusView | None = Field(
        None,
        description="Whether the rapid focus mode is offered, and what it measures while it runs. "
        "It is `null` outside the alignment. While the mode runs, the camera streams the fast "
        "readout mode, so no new live-view frame arrives, and the readings travel with the video "
        "of Polaris (`polaris` channel).",
    )
    quality: dict[str, str] = Field(default_factory=dict, max_length=32)


def decode_alignment_state(value: Any) -> AlignmentState:
    """Check the JSON that `alignment_state` answered with. Raises `CodecError`."""
    try:
        return AlignmentState.model_validate(value)
    except ValidationError as error:
        fields = ", ".join(
            sorted({".".join(str(part) for part in e["loc"]) for e in error.errors()})
        )
        raise CodecError(f"the alignment state is not valid: {fields}") from None


@dataclass(frozen=True, slots=True)
class AlignmentFrame:
    """One live-view frame: the JPEG of the image and the state that describes it."""

    state: AlignmentState
    jpeg: bytes


def _pack_message(
    magic: bytes, state: BaseModel, image: bytes, image_magic: bytes, label: str, image_name: str
) -> bytes:
    """The payload of a live-view message: the magic, the state length, the state, the image."""
    body = state.model_dump_json().encode("utf-8")
    if len(body) > MAX_STATE_BYTES:
        raise ValueError(f"the {label} state is too large")
    if not image.startswith(image_magic):
        raise ValueError(f"the frame is not a {image_name} image")
    return magic + struct.pack("<I", len(body)) + body + image


def _split_message(
    payload: bytes | bytearray | memoryview,
    magic: bytes,
    image_magic: bytes,
    noun: str,
    label: str,
    image_name: str,
) -> tuple[bytes, bytes]:
    """The JSON state and the image of a live-view message. Raises `CodecError`."""
    raw = bytes(payload)
    if len(raw) < 8 or raw[:4] != magic:
        raise CodecError(f"the message is not {noun}")
    (length,) = struct.unpack_from("<I", raw, 4)
    if length > MAX_STATE_BYTES or 8 + length >= len(raw):
        raise CodecError(f"the {label} frame has a bad state length")
    image = raw[8 + length :]
    if not image.startswith(image_magic):
        raise CodecError(f"the {label} frame holds no {image_name} image")
    return raw[8 : 8 + length], image


def pack_frame(state: AlignmentState, jpeg: bytes) -> bytes:
    """The payload of one data message of the `alignment` channel."""
    return _pack_message(FRAME_MAGIC, state, jpeg, JPEG_MAGIC, "alignment", "JPEG")


def unpack_frame(payload: bytes | bytearray | memoryview) -> AlignmentFrame:
    """The inverse of `pack_frame`. Raises `CodecError` for a message that is not a frame."""
    body, jpeg = _split_message(
        payload, FRAME_MAGIC, JPEG_MAGIC, "an alignment frame", "alignment", "JPEG"
    )
    try:
        state = AlignmentState.model_validate_json(body)
    except ValidationError:
        raise CodecError("the alignment frame has an unreadable state") from None
    return AlignmentFrame(state, jpeg)


# --- The live video of Polaris ---------------------------------------------------------------


class PolarisStar(_View):
    """Where the star is in this frame, and how wide it is.

    `x` and `y` are pixels of the image, and the center of the first pixel is 0. A page that
    magnifies the image by `k` draws the star at `((x + 0.5) * k, (y + 0.5) * k)`. `peak_fraction`
    is the brightest pixel of the aperture as a share of the full scale of the ADC. `fwhm_arcsec`
    is the width of this frame alone, from its second moments inside the aperture of the fast
    analysis, so it jitters more than the width of a stored window, which averages it. When `found`
    is false, the other fields are `null`.
    """

    found: bool
    x: float | None = None
    y: float | None = None
    peak_fraction: float | None = Field(None, ge=0)
    fwhm_arcsec: float | None = Field(None, ge=0)


class PolarisStretch(_View):
    """The levels of the stretch, in the counts that the camera delivers.

    A frame of a 12-bit ADC in a 16-bit container reads 16 times its ADU. The image is black at
    `black_dn` and below, and white at `white_dn` and above, with an `asinh` curve between them.
    """

    black_dn: float
    white_dn: float


class LiveSeeingView(_View):
    """The rolling seeing value of the fast stream: a provisional number that `core` does not store.

    `core` estimates the seeing from the newest `span_s` seconds of frames with the estimator of the
    stored windows, and it repeats the estimate every few seconds. `t_utc_ns` is the end of the
    span in frame time. `n_frames` counts the frames of the span, and `n_usable` those with a
    usable centroid, and `valid_fraction` is `n_usable` over the frames that the camera produced
    (the frames lost in between included). The `seeing_fwhm_*` and `r0_*` values follow the
    records of the `seeing` series: the first of each pair comes from the variance of the motion,
    and the second from its structure function. `flags` holds the window flags that apply
    (`degraded`, `saturated`, and the context of the scheduler, such as `cloud`). A value that
    `core` cannot give is `null`, and `quality` says why.
    """

    t_utc_ns: int
    span_s: float = Field(gt=0)
    n_frames: int = Field(ge=0)
    n_usable: int = Field(ge=0)
    valid_fraction: float = Field(ge=0, le=1)
    seeing_fwhm_arcsec: float | None = None
    seeing_fwhm_structure_arcsec: float | None = None
    r0_cm: float | None = None
    r0_structure_cm: float | None = None
    image_motion_rms_x_arcsec: float | None = None
    image_motion_rms_y_arcsec: float | None = None
    width_fwhm_arcsec: float | None = None
    stream_id: int = Field(ge=0)
    readout_mode: str = Field(max_length=64)
    exposure_us: int = Field(ge=0)
    flags: list[str] = Field(default_factory=list, max_length=MAX_LIST_ITEMS)
    quality: dict[str, str] = Field(default_factory=dict, max_length=MAX_LIST_ITEMS)


class PolarisState(_View):
    """What a page shows next to one frame of the live video of Polaris.

    `seq` is the number that the hub of `web` gives the frame (`core` sends 0), and a client that
    polls passes the newest one as `after`. `t_utc` and `t_utc_ns` are the time of the frame, and
    `stream_id`, `mode`, `exposure_us`, and `gain` name the stream that it belongs to. `roi` is the
    rectangle of the sensor that the image shows, in pixels of the readout mode, and
    `scale_arcsec_px` is the plate scale of that mode. `fast_fps` is the frame rate of the camera,
    which is higher than the rate of the video.

    The image has the type `image_type` and the size `image_width` by `image_height`, which is the
    size of the ROI: the page magnifies it. The video is lossless (a PNG), so that the page can
    magnify it with nearest-neighbor scaling and show the pixels of the star. `stretch` holds the
    levels of the stretch of the image, `star` the star of this frame, and `live_seeing` the newest
    rolling seeing value, or `null` while `core` has none. A value that `core` cannot give is
    `null`, and `quality` says why.

    While the rapid focus mode runs, the frames come from that mode (`stream_id` is the stream of
    the mode), `live_seeing` is `null` (the mode makes no seeing value), and `rapid_focus` holds the
    readings of the star width, so the page draws the video and the curve from one message.
    `rapid_focus` is `null` in every other frame.
    """

    seq: int = Field(0, ge=0)
    t_utc: str = Field(max_length=40)
    t_utc_ns: int
    stream_id: int = Field(ge=0)
    mode: str = Field(max_length=64)
    exposure_us: int = Field(ge=0)
    gain: int = Field(ge=0)
    roi: RoiView
    scale_arcsec_px: float | None = Field(None, gt=0)
    fast_fps: float | None = Field(None, gt=0)
    image_type: Literal["image/png"] = "image/png"
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    star: PolarisStar
    stretch: PolarisStretch
    live_seeing: LiveSeeingView | None = None
    rapid_focus: RapidFocusView | None = None
    quality: dict[str, str] = Field(default_factory=dict, max_length=32)


def decode_live_seeing(value: Any) -> LiveSeeingView | None:
    """Check the JSON that `live_seeing` answered with. `None` means that `core` has no value."""
    if value is None:
        return None
    try:
        return LiveSeeingView.model_validate(value)
    except ValidationError as error:
        fields = ", ".join(
            sorted({".".join(str(part) for part in e["loc"]) for e in error.errors()})
        )
        raise CodecError(f"the live seeing is not valid: {fields}") from None


@dataclass(frozen=True, slots=True)
class PolarisFrame:
    """One frame of the live video of Polaris: the PNG image and the state that describes it."""

    state: PolarisState
    image: bytes


def pack_polaris_frame(state: PolarisState, image: bytes) -> bytes:
    """The payload of one data message of the `polaris` channel."""
    return _pack_message(POLARIS_MAGIC, state, image, PNG_MAGIC, "Polaris", "PNG")


def unpack_polaris_frame(payload: bytes | bytearray | memoryview) -> PolarisFrame:
    """The inverse of `pack_polaris_frame`. Raises `CodecError` for a message that is no frame."""
    body, image = _split_message(
        payload, POLARIS_MAGIC, PNG_MAGIC, "a Polaris frame", "Polaris", "PNG"
    )
    try:
        state = PolarisState.model_validate_json(body)
    except ValidationError:
        raise CodecError("the Polaris frame has an unreadable state") from None
    return PolarisFrame(state, image)


# --- The dark library ------------------------------------------------------------------------


class DarkSetView(_View):
    """One dark set of the library. The rate is in electrons per second per pixel."""

    name: str
    t_utc: str
    age_days: float
    temperature_c: float
    temperature_spread_c: float
    exposure_s: float
    n_frames: int
    n_bias_frames: int
    rate_e_per_s: float
    hot_pixels: int


class DarkModelView(_View):
    """The dark current as a function of temperature: the rate at the reference and the doubling."""

    reference_c: float
    rate_ref_e_per_s: float
    doubling_c: float
    doubling_fitted: bool
    rms_log2: float | None = None
    n_sets: int


class DarkStatusView(_View):
    """Whether the library needs a new set, and why (`reason` is a sentence)."""

    due: bool
    reason: str
    tolerance_c: float
    max_age_days: float
    gap_c: float | None = None
    nearest_name: str | None = None
    newest_age_days: float | None = None


class DarkTaskView(_View):
    """The latest dark session of this `core` process.

    `state` is `idle` (none yet), `queued`, `running`, `ok`, `failed`, or `aborted`. While it runs,
    `phase` is `bias`, `cover` (waiting for dark frames), `dark`, or `build` (the master dark and
    the library), with `step` of `steps` in that phase. `covered` and `level_dn` describe the
    latest check of a frame, and `reason` says why a frame was not dark. A finished session keeps
    its `summary` (one sentence) and the `set_name` that it added.
    """

    state: str = "idle"
    task_id: int | None = None
    phase: str | None = None
    step: int = 0
    steps: int = 0
    message: str = ""
    covered: bool | None = None
    level_dn: float | None = None
    reason: str = ""
    exposure_s: float | None = None
    frames: int | None = None
    bias_frames: int | None = None
    wait_for_cover: bool = True
    pause_after: bool = True
    started_utc: str | None = None
    finished_utc: str | None = None
    summary: str = ""
    set_name: str | None = None


class DarkLibraryView(_View):
    """The answer of the `dark_library` method: the library, its status, and the latest session.

    `mode`, `gain`, and `exposure_s` are the settings of the survey, which a new set should match.
    `sets` holds the newest sets first, at most `MAX_LIST_ITEMS`.
    """

    mode: str
    gain: int
    exposure_s: float
    sensor_temperature_c: float | None = None
    status: DarkStatusView
    model: DarkModelView | None = None
    sets: list[DarkSetView] = Field(default_factory=list, max_length=MAX_LIST_ITEMS)
    task: DarkTaskView = Field(default_factory=DarkTaskView)


def decode_dark_library(value: Any) -> DarkLibraryView:
    """The answer of `dark_library` as a model. Raises `CodecError` for anything malformed."""
    try:
        return DarkLibraryView.model_validate(value)
    except ValidationError as error:
        fields = ", ".join(
            sorted({".".join(str(part) for part in e["loc"]) for e in error.errors()})
        )
        raise CodecError(f"the dark library is not valid: {fields}") from None


# --- The flat library ------------------------------------------------------------------------


class FlatPointView(_View):
    """The flat at one radius from the optical center, against the center, in percent."""

    radius_deg: float
    change_percent: float | None = None
    corner: bool = False


class FlatTiltView(_View):
    """A plane across the frame: the change from one edge to the other, in percent."""

    width_percent: float | None = None
    height_percent: float | None = None


class FlatShadowView(_View):
    """A dust shadow: its place in sensor pixels, its depth in percent, and its width."""

    x_px: int
    y_px: int
    depth_percent: float
    width_px: float


class FlatSetView(_View):
    """One set of frames of a flat: its exposure, its level, and the frames that counted."""

    number: int
    exposure_s: float | None = None
    level_fraction: float | None = None
    frames: int = 0
    used: int = 0
    dropped: dict[str, int] = Field(default_factory=dict, max_length=16)
    noise_percent: float | None = None
    tilt: FlatTiltView | None = None


class FlatAgreementView(_View):
    """How well two sets of frames agree, which shows what the light source adds."""

    smooth_rms_percent: float | None = None
    fine_rms_percent: float | None = None
    expected_fine_rms_percent: float | None = None
    plane: FlatTiltView | None = None


class FlatView(_View):
    """One flat of the library, with the numbers of its report.

    `state` is `pending` (a session made it, and nothing uses it yet) or `approved` (you activated
    it once). `active` says that the survey divides by it now, and `pending` says that it waits for
    your decision. `corner_percent` is the change in the corners against the center (negative: the
    corners get less light). `optics_tilt` and `source_tilt` exist for a flat of two sets. The
    image of the flat is at `GET /flat/{version}/image` when `has_image` is true.
    """

    version: str
    t_utc: str
    age_days: float = 0.0
    state: str
    active: bool = False
    pending: bool = False
    mode: str
    gain: int
    width_px: int
    height_px: int
    sensor_temperature_c: float | None = None
    exposure_s: float | None = None
    target_fraction: float | None = None
    second_set: bool = False
    source_turned: bool = False
    frames_taken: int = 0
    frames_used: int = 0
    noise_percent: float | None = None
    bias_source: str = ""
    bias_note: str = ""
    corner_percent: float | None = None
    vignetting: list[FlatPointView] = Field(default_factory=list, max_length=16)
    tilt: FlatTiltView = Field(default_factory=FlatTiltView)
    optics_tilt: FlatTiltView | None = None
    source_tilt: FlatTiltView | None = None
    shadows: int = 0
    shadow_min_depth_percent: float | None = None
    shadow_items: list[FlatShadowView] = Field(default_factory=list, max_length=32)
    edge_artifacts: int = 0
    agreement: FlatAgreementView | None = None
    sets: list[FlatSetView] = Field(default_factory=list, max_length=4)
    warnings: list[str] = Field(default_factory=list, max_length=32)
    has_image: bool = False
    activated_utc: str | None = None


class FlatSessionView(_View):
    """The first set of a session, which waits for a second set with the source turned.

    `version` is the pending flat that the first set made. `expires_utc` is when the frames of the
    first set go (24 hours after the first set).
    """

    version: str
    t_utc: str
    expires_utc: str
    frames: int
    exposure_s: float


class FlatTaskView(_View):
    """The latest flat session of this `core` process.

    `state` is `idle` (none yet), `queued`, `running`, `ok`, `failed`, or `aborted`. While it runs,
    `phase` is `setup`, `exposure` (the search for the exposure), `capture` (the frames), or `build`
    (the combination), with `step` of `steps` in that phase. `exposure_s` is the exposure in use or
    found, `level_fraction` is the latest level above the bias as a fraction of the full scale (the
    page draws it against `target_fraction`), and `saturated_fraction` is the share of saturated
    pixels of the latest frame. `warnings` list what the session noticed: a drifting light, a
    saturating one, a light that does not cover the corners. A finished session keeps its
    `summary` (one sentence) and the `version` of the flat that it added.
    """

    state: str = "idle"
    task_id: int | None = None
    phase: str | None = None
    step: int = 0
    steps: int = 0
    message: str = ""
    set_number: int = 1
    frames: int | None = None
    target_fraction: float | None = None
    exposure_s: float | None = None
    level_fraction: float | None = None
    saturated_fraction: float | None = None
    warnings: list[str] = Field(default_factory=list, max_length=16)
    pause_after: bool = True
    started_utc: str | None = None
    finished_utc: str | None = None
    summary: str = ""
    version: str | None = None


class FlatLibraryView(_View):
    """The answer of the `flat_library` method: the flats, the one in use, and the latest session.

    `mode` and `gain` are the settings of the flat session (the ones of the survey). `blocker` is a
    sentence that says why a session cannot start now (no dark set), or `null`. `pending_version`
    names the newest flat that waits for a decision, `active_version` the flat in use. A
    configuration that names `[survey] flat_file` has `flat_file_pinned` true, and
    `library_overrides` is true when the active flat of the library wins over that file. `session`
    exists while the first set of a session waits for a second set. `flats` holds the newest flats
    first.
    """

    mode: str
    gain: int
    sensor_temperature_c: float | None = None
    active_version: str | None = None
    pending_version: str | None = None
    flat_file_pinned: bool = False
    library_overrides: bool = False
    blocker: str | None = None
    flats: list[FlatView] = Field(default_factory=list, max_length=MAX_LIST_ITEMS)
    session: FlatSessionView | None = None
    task: FlatTaskView = Field(default_factory=FlatTaskView)


class FlatActionView(_View):
    """The answer of `flat_activate` and `flat_delete`.

    `reason` is `null` when the action worked. Otherwise it is `unknown` (no such flat), `active`
    (the flat is in use), `invalid` (the flat cannot be used), or `busy` (a session runs).
    """

    ok: bool
    reason: str | None = None
    message: str = ""
    version: str | None = None


class FlatImageView(_View):
    """The answer of `flat_image`: the JPEG preview of a flat as base64 text, or `found` false."""

    found: bool
    jpeg: str = Field("", max_length=MAX_FLAT_JPEG_BYTES * 2)


ModelT = TypeVar("ModelT", bound=_View)


def _decode_view(model: type[ModelT], value: Any, what: str) -> ModelT:
    try:
        return model.model_validate(value)
    except ValidationError as error:
        fields = ", ".join(
            sorted({".".join(str(part) for part in e["loc"]) for e in error.errors()})
        )
        raise CodecError(f"{what} is not valid: {fields}") from None


def decode_flat_library(value: Any) -> FlatLibraryView:
    """The answer of `flat_library` as a model. Raises `CodecError` for anything malformed."""
    return _decode_view(FlatLibraryView, value, "the flat library")


def decode_flat_action(value: Any) -> FlatActionView:
    """The answer of `flat_activate` or `flat_delete`. Raises `CodecError`."""
    return _decode_view(FlatActionView, value, "the flat answer")


def encode_flat_image(jpeg: bytes | None) -> dict[str, Any]:
    """The answer of `flat_image` for a JPEG, or for none."""
    if jpeg is None:
        return {"found": False, "jpeg": ""}
    if len(jpeg) > MAX_FLAT_JPEG_BYTES or not jpeg.startswith(JPEG_MAGIC):
        raise ValueError("the flat preview is not a JPEG image of a size that the RPC carries")
    return {"found": True, "jpeg": base64.b64encode(jpeg).decode("ascii")}


def decode_flat_image(value: Any) -> bytes | None:
    """The JPEG of a `flat_image` answer, or `None` when `core` has none. Raises `CodecError`."""
    view = _decode_view(FlatImageView, value, "the flat image")
    if not view.found:
        return None
    try:
        jpeg = base64.b64decode(view.jpeg, validate=True)
    except ValueError:
        raise CodecError("the flat image is not valid: jpeg") from None
    if len(jpeg) > MAX_FLAT_JPEG_BYTES or not jpeg.startswith(JPEG_MAGIC):
        raise CodecError("the flat image is not a JPEG image")
    return jpeg
