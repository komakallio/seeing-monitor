"""What `web` and `core` agree on: the RPC methods, the JSON shapes, and the alignment frame.

`core` serves two channels on its `IpcServer`, and `web` is the client of both. This module holds
the names and the codecs, so that both sides build and read the same bytes. It needs no FastAPI, so
`core` can import it.

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

**Commands.** `encode_command` writes a command as `{"type": <name>, ...fields}`. The names are
`start_alignment`, `stop_alignment`, `pause`, `resume`, `queue_burst`, `queue_sweep`,
`queue_replay`, and `queue_dark`. A field that the command lacks takes the default of the
dataclass. `queue_replay` names its source (`source`) as a recording name without a directory
part, and `core` resolves it under the configured recordings folder.
"""

from __future__ import annotations

import dataclasses
import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from seeingmon.scheduler.commands import (
    Command,
    CommandResult,
    Pause,
    QueueBurst,
    QueueDark,
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

METHOD_PING = "ping"
METHOD_STATUS = "status"
METHOD_SUBMIT = "submit"
METHOD_ALIGNMENT_STATE = "alignment_state"
METHOD_DARK_LIBRARY = "dark_library"
METHODS = (
    METHOD_PING,
    METHOD_STATUS,
    METHOD_SUBMIT,
    METHOD_ALIGNMENT_STATE,
    METHOD_DARK_LIBRARY,
)

FRAME_MAGIC = b"SMAF"
MAX_STATE_BYTES = 64 * 1024
JPEG_MAGIC = b"\xff\xd8\xff"
MAX_LIST_ITEMS = 256
MAX_TEXT_CHARS = 4000


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


class FocusView(_View):
    """The focus measure: the median FWHM of the unsaturated stars, and the best of the session."""

    fwhm_px: float | None = Field(None, ge=0)
    best_fwhm_px: float | None = Field(None, ge=0)
    n_stars: int | None = Field(None, ge=0)


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


class AlignmentState(_View):
    """What the Align page shows next to the live view. The answer of `alignment_state`.

    Every part is `null` when `core` does not know it yet, and `quality` says why. `t_utc` is an
    ISO 8601 UTC time. Positions are in pixels of the frame in `frame`, so the UI scales them to
    the size of the image that it shows. `reticle` is the fixed circle of the first layer, and it
    exists without a solution. `sky` is the layer that is fixed to the stars: the pole, the aim, the
    orbit of Polaris, and the camera model of the latest current solution. It exists whether or not
    a target is set. `timing` says how old the frame and the solution are. `aim_ring` is where
    Polaris belongs on the circle of the reticle, and it exists while the state has no current
    solution, when it comes from `last_solution`.
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


def pack_frame(state: AlignmentState, jpeg: bytes) -> bytes:
    """The payload of one data message of the `alignment` channel."""
    body = state.model_dump_json().encode("utf-8")
    if len(body) > MAX_STATE_BYTES:
        raise ValueError("the alignment state is too large")
    if not jpeg.startswith(JPEG_MAGIC):
        raise ValueError("the frame is not a JPEG image")
    return FRAME_MAGIC + struct.pack("<I", len(body)) + body + jpeg


def unpack_frame(payload: bytes | bytearray | memoryview) -> AlignmentFrame:
    """The inverse of `pack_frame`. Raises `CodecError` for a message that is not a frame."""
    raw = bytes(payload)
    if len(raw) < 8 or raw[:4] != FRAME_MAGIC:
        raise CodecError("the message is not an alignment frame")
    (length,) = struct.unpack_from("<I", raw, 4)
    if length > MAX_STATE_BYTES or 8 + length >= len(raw):
        raise CodecError("the alignment frame has a bad state length")
    jpeg = raw[8 + length :]
    if not jpeg.startswith(JPEG_MAGIC):
        raise CodecError("the alignment frame holds no JPEG image")
    try:
        state = AlignmentState.model_validate_json(raw[8 : 8 + length])
    except ValidationError:
        raise CodecError("the alignment frame has an unreadable state") from None
    return AlignmentFrame(state, jpeg)


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
