"""The request and response models of the REST API, except the records.

The request models validate every field that a client sends: the bounds, the enums, and the sizes.
A model is strict (a string never becomes a number), rejects unknown fields, and rejects `NaN` and
infinity. `to_command` and `to_config` turn a validated request into the scheduler objects. The
scheduler checks the values against the profile again, because only it knows the camera limits.

The record schemas come from `seeingmon.records.api_schema`, and `seeingmon.services.web.schemas`
adds them to the OpenAPI document.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from seeingmon.frames import PixelFormat, Roi, StreamConfig
from seeingmon.scheduler.commands import (
    Command,
    Pause,
    QueueBurst,
    QueueDark,
    QueueReplay,
    QueueSweep,
    Resume,
    StartAlignment,
)

MODE_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,15}$"
RECORDING_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
LABEL_PATTERN = r"^[A-Za-z0-9 _.-]*$"
OPTION_KEY_PATTERN = r"^[a-z][a-z0-9_]{0,31}$"
MAX_EXPOSURE_US = 2_000_000_000
MAX_GAIN = 1000
MAX_BURST_S = 600.0
MAX_WINDOW_S = 600.0
MAX_ALIGNMENT_EXPOSURE_S = 10.0
MAX_SWEEP_VALUES = 16
MAX_ROI_PX = 100_000
MAX_ROI_ARCMIN = 60.0
MAX_OPTIONS = 8
MAX_PRIORITY = 10
MIN_DARK_FRAMES = 3
MAX_DARK_FRAMES = 50
MAX_DARK_EXPOSURE_S = 600.0
MAX_LABEL_CHARS = 80

_Gain = Annotated[int, Field(ge=0, le=MAX_GAIN)]
_ExposureUs = Annotated[int, Field(ge=1, le=MAX_EXPOSURE_US)]
_RoiArcmin = Annotated[float, Field(gt=0, le=MAX_ROI_ARCMIN)]
_Mode = Annotated[str, Field(pattern=MODE_PATTERN)]
_OptionKey = Annotated[str, Field(pattern=OPTION_KEY_PATTERN)]
_OptionValue = Annotated[str, Field(max_length=64)] | int | float | bool


class EventLevel(StrEnum):
    """The lowest severity that an event query returns."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Order(StrEnum):
    """The order of a page of events."""

    ASC = "asc"
    DESC = "desc"


class ImageFormat(StrEnum):
    """What `GET /images/{id}` returns."""

    JPEG = "jpeg"
    FITS = "fits"
    JSON = "json"


class _Request(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)


class _Response(BaseModel):
    model_config = ConfigDict(frozen=True)


# --- Commands --------------------------------------------------------------------------------


class RoiRequest(_Request):
    """A rectangle of the sensor, in pixels of the readout mode."""

    x: int = Field(ge=0, le=MAX_ROI_PX)
    y: int = Field(ge=0, le=MAX_ROI_PX)
    width: int = Field(gt=0, le=MAX_ROI_PX)
    height: int = Field(gt=0, le=MAX_ROI_PX)


class StreamRequest(_Request):
    """The stream settings of a burst. Leave the whole object out for the fast stream."""

    mode: str = Field(pattern=MODE_PATTERN, description="The name of a readout mode.")
    exposure_us: _ExposureUs
    gain: _Gain
    pixel_format: Literal["RAW8", "RAW16"] = "RAW16"
    roi: RoiRequest | None = Field(None, description="Leave out for the full frame.")
    high_speed: bool = False

    def to_config(self) -> StreamConfig:
        roi = None
        if self.roi is not None:
            roi = Roi(self.roi.x, self.roi.y, self.roi.width, self.roi.height)
        return StreamConfig(
            mode=self.mode,
            exposure_us=self.exposure_us,
            gain=self.gain,
            pixel_format=PixelFormat[self.pixel_format],
            roi=roi,
            high_speed=self.high_speed,
        )


class BurstRequest(_Request):
    """Record raw frames to a SER file for `duration_s` seconds."""

    duration_s: float = Field(10.0, gt=0, le=MAX_BURST_S)
    label: str = Field("", max_length=40, pattern=LABEL_PATTERN)
    priority: int = Field(0, ge=-MAX_PRIORITY, le=MAX_PRIORITY)
    stream: StreamRequest | None = None

    def to_command(self) -> QueueBurst:
        return QueueBurst(
            duration_s=self.duration_s,
            stream=None if self.stream is None else self.stream.to_config(),
            label=self.label,
            priority=self.priority,
        )


class SweepRequest(_Request):
    """Run a short fast window for every cell of a grid. An empty axis uses the default."""

    exposure_us: list[_ExposureUs] = Field(default_factory=list, max_length=MAX_SWEEP_VALUES)
    gain: list[_Gain] = Field(default_factory=list, max_length=MAX_SWEEP_VALUES)
    roi_arcmin: list[_RoiArcmin] = Field(default_factory=list, max_length=MAX_SWEEP_VALUES)
    modes: list[_Mode] = Field(default_factory=list, max_length=MAX_SWEEP_VALUES)
    window_s: float | None = Field(None, gt=0, le=MAX_WINDOW_S)
    priority: int = Field(0, ge=-MAX_PRIORITY, le=MAX_PRIORITY)

    def to_command(self) -> QueueSweep:
        return QueueSweep(
            exposure_us=tuple(self.exposure_us),
            gain=tuple(self.gain),
            roi_arcmin=tuple(self.roi_arcmin),
            modes=tuple(self.modes),
            window_s=self.window_s,
            priority=self.priority,
        )


class ReplayRequest(_Request):
    """Replay a recording through the production analysis."""

    source: str = Field(
        pattern=RECORDING_PATTERN,
        description="The name of a recording in the configured folder, without a directory part.",
    )
    speed: float = Field(1.0, ge=0, le=1000, description="0 means as fast as possible.")
    options: dict[_OptionKey, _OptionValue] = Field(default_factory=dict, max_length=MAX_OPTIONS)
    priority: int = Field(0, ge=-MAX_PRIORITY, le=MAX_PRIORITY)

    def to_command(self) -> QueueReplay:
        return QueueReplay(
            source=self.source,
            speed=self.speed,
            options=dict(self.options),
            priority=self.priority,
        )


class DarkRequest(_Request):
    """Record a dark set with the camera covered. Leave a field out to use the configured value.

    The server also holds `exposure_s` to `[web.requests] max_dark_exposure_s` and `label` to
    `max_label_chars`, which are settings of the installation.
    """

    exposure_s: float | None = Field(
        None,
        gt=0,
        le=MAX_DARK_EXPOSURE_S,
        description="The exposure of a dark frame, in seconds. `null` uses the survey exposure.",
    )
    frames: int | None = Field(
        None,
        ge=MIN_DARK_FRAMES,
        le=MAX_DARK_FRAMES,
        description="The number of dark frames. `null` uses the configured number.",
    )
    bias_frames: int | None = Field(
        None,
        ge=MIN_DARK_FRAMES,
        le=MAX_DARK_FRAMES,
        description="The number of bias frames. `null` uses the configured number.",
    )
    wait_for_cover: bool = Field(
        True,
        description="Wait until a test frame is dark, which means that the camera is covered. "
        "Without it, the first frame that is not dark ends the task as failed.",
    )
    pause_after: bool = Field(
        True,
        description="Pause the scheduler when the task ends, so that nothing records data while "
        "the camera may still be covered. `Resume` continues.",
    )
    label: str = Field("", max_length=MAX_LABEL_CHARS, pattern=LABEL_PATTERN)

    def to_command(self) -> QueueDark:
        return QueueDark(
            exposure_s=self.exposure_s,
            frames=self.frames,
            bias_frames=self.bias_frames,
            wait_for_cover=self.wait_for_cover,
            pause_after=self.pause_after,
            label=self.label,
        )


class ModeRequest(_Request):
    """Pause the scheduler, or resume it."""

    mode: Literal["paused", "auto"] = Field(
        description="`paused` stops everything. `auto` resumes, through `safe`."
    )

    def to_command(self) -> Command:
        return Pause() if self.mode == "paused" else Resume()


class AlignmentStartRequest(_Request):
    """Start the alignment stream. Leave a field out to use the configured value."""

    exposure_s: float | None = Field(None, gt=0, le=MAX_ALIGNMENT_EXPOSURE_S)
    gain: int | None = Field(None, ge=0, le=MAX_GAIN)

    def to_command(self) -> StartAlignment:
        return StartAlignment(exposure_s=self.exposure_s, gain=self.gain)


class CommandResponse(_Response):
    """The scheduler's answer to a command. A rejection has `accepted` false and a `reason`."""

    accepted: bool
    message: str
    state: str = Field(description="The state of the scheduler right after the command.")
    reason: str | None = Field(None, description="The code of a rejection, or `null`.")
    task_id: int | None = Field(None, description="The ID of a queued task, or `null`.")


# --- Dark ------------------------------------------------------------------------------------


class DarkSetResponse(_Response):
    """One dark set of the library."""

    name: str
    t_utc: str
    age_days: float
    temperature_c: float = Field(description="The mean sensor temperature during the set.")
    temperature_spread_c: float
    exposure_s: float
    n_frames: int
    n_bias_frames: int
    rate_e_per_s: float = Field(description="The dark current, in electrons per second per pixel.")
    hot_pixels: int


class DarkModelResponse(_Response):
    """The dark current as a function of temperature: the rate at the reference and the doubling.

    `doubling_fitted` is false while the library holds too few sets for a fit, and the doubling
    is then a default.
    """

    reference_c: float
    rate_ref_e_per_s: float
    doubling_c: float = Field(description="The temperature step that doubles the dark current.")
    doubling_fitted: bool
    rms_log2: float | None = None
    n_sets: int


class DarkStatusResponse(_Response):
    """Whether the library needs a new set, and why. `reason` is a sentence."""

    due: bool
    reason: str
    tolerance_c: float
    max_age_days: float
    gap_c: float | None = None
    nearest_name: str | None = None
    newest_age_days: float | None = None


class DarkTaskResponse(_Response):
    """The latest dark session of this `core` process.

    `state` is `idle` (none yet), `queued`, `running`, `ok`, `failed`, or `aborted`. While it
    runs, `phase` is `bias`, `cover` (waiting for dark frames), `dark`, or `build` (the master
    dark and the library), with `step` of `steps` in that phase. `covered` and `level_dn` describe
    the latest check of a frame, and `reason` says why a frame was not dark. A finished session
    keeps its `summary` (one sentence) and the `set_name` that it added.
    """

    state: str
    task_id: int | None = None
    phase: str | None = None
    step: int
    steps: int
    message: str
    covered: bool | None = None
    level_dn: float | None = None
    reason: str
    exposure_s: float | None = None
    frames: int | None = None
    bias_frames: int | None = None
    wait_for_cover: bool
    pause_after: bool
    started_utc: str | None = None
    finished_utc: str | None = None
    summary: str
    set_name: str | None = None


class DarkLibraryResponse(_Response):
    """The dark library, whether it is due for a new set, and the latest dark session.

    `mode`, `gain`, and `exposure_s` are the settings of the survey, which a new set should match.
    `sets` holds the newest sets first. A value that `core` does not report is `null`, and
    `quality` says why.
    """

    mode: str
    gain: int
    exposure_s: float
    sensor_temperature_c: float | None
    status: DarkStatusResponse
    model: DarkModelResponse | None
    sets: list[DarkSetResponse]
    task: DarkTaskResponse
    quality: dict[str, str] | None = None


# --- Errors ----------------------------------------------------------------------------------


class ErrorDetail(_Response):
    """What went wrong. `code` is a short word, and `message` is one sentence."""

    code: str
    message: str
    details: list[dict[str, str]] | None = Field(
        None, description="One entry for each invalid field of a request."
    )


class ErrorResponse(_Response):
    """The body of every error response."""

    error: ErrorDetail


# --- Health and status -----------------------------------------------------------------------


class HealthResponse(_Response):
    """The verdict of `GET /health`. The HTTP status is 200 unless `status` is `failed`."""

    status: Literal["healthy", "degraded", "failed"]
    now: str
    age_s: float | None = Field(description="The age of the newest health record, in seconds.")
    reasons: list[str]
    components: dict[str, str]
    flags: list[str]
    quality: dict[str, str] | None = None


class CoreLink(_Response):
    """Whether the web process reaches `core`."""

    reachable: bool
    instance: str | None = Field(None, description="The ID of the `core` process, when known.")


class StreamStatus(_Response):
    stream_id: int
    purpose: str
    mode: str
    exposure_us: int
    gain: int
    roi: dict[str, int] | None = None


class FaultStatusView(_Response):
    failures: int
    good_frames: int
    last_error: str | None = None
    next_attempt: str | None = None
    next_step: str | None = None


class SchedulerStatusResponse(_Response):
    """The scheduler as `core` reports it.

    The status leaves out the Sun's elevation, because a series of elevations shows the site.
    """

    t_utc: str
    state: str
    state_reason: str
    state_since: str
    last_transition: str | None
    degraded: bool
    stream: StreamStatus | None
    cloud: bool
    cloud_fraction: float | None
    twilight: bool
    background_fraction: float | None
    sensor_temperature_c: float | None
    counters: dict[str, int]
    fault: FaultStatusView
    queued_tasks: int
    survey_pending: int
    alignment_idle_s: float | None


class RecordTime(_Response):
    """The newest record of a type."""

    t_utc: str | None
    age_s: float | None


class UiSettings(_Response):
    """What the UI needs to know about the server."""

    refresh_s: float
    token_required_for_reads: bool
    commands_enabled: bool = Field(description="False when no token hash is configured.")
    alignment_max_fps: float
    alignment_stall_s: float


class StatusResponse(_Response):
    """The state of every component."""

    now: str
    api_version: str
    software_version: str
    station_id: str | None
    demo: bool
    health: HealthResponse
    core: CoreLink
    scheduler: SchedulerStatusResponse | None
    data: dict[str, RecordTime]
    ui: UiSettings
    quality: dict[str, str] | None = None


# --- Images ----------------------------------------------------------------------------------


class ImageItem(_Response):
    """One image."""

    id: str = Field(description="The ID. Pass it to `GET /images/{id}`.")
    kind: str
    t_utc: str
    t_utc_ns: int
    size_bytes: int
    has_fits: bool
    fits_bytes: int | None
    preview_url: str
    fits_url: str | None


class ImageList(_Response):
    """A page of images, newest first."""

    now: str
    items: list[ImageItem]
    next_cursor: str | None = None


class ApiInfo(_Response):
    """The root of the API."""

    api_version: str
    software_version: str
    openapi: str
