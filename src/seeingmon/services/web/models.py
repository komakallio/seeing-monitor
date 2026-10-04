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
    MAX_FLAT_FRAMES,
    MAX_FLAT_TARGET,
    MIN_FLAT_FRAMES,
    MIN_FLAT_TARGET,
    Command,
    Pause,
    QueueBurst,
    QueueDark,
    QueueFlat,
    QueueReplay,
    QueueSweep,
    Resume,
    StartAlignment,
)
from seeingmon.services.web.contract import LiveSeeingView

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
MAX_COVER_WAIT_S = 7200.0
FLAT_VERSION_PATTERN = r"^flat-[0-9a-f]{8}$"

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
    wait_for_cover_timeout_s: float | None = Field(
        None,
        gt=0,
        le=MAX_COVER_WAIT_S,
        description="How long the session waits for the cover before it gives up, in seconds. "
        "`null` takes the setting `wait_timeout_s` of `[survey.dark]`.",
    )
    immediate: bool = Field(
        True,
        description="Start at the next step of the scheduler, because someone stands at the "
        "camera with the cover on. A survey exposure in progress finishes first. Without it, the "
        "session waits for the next cycle boundary, as the other tasks do.",
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
            wait_for_cover_timeout_s=self.wait_for_cover_timeout_s,
            immediate=self.immediate,
        )


class FlatSessionRequest(_Request):
    """Record a flat with a light source over the aperture. Leave a field out to use the default.

    The session finds the exposure for the light, takes `frames` frames, combines them with the
    bias of the dark library, and adds a pending flat to the library. The survey divides by it
    only after `POST /flat/{version}/activate`.
    """

    frames: int = Field(
        32,
        ge=MIN_FLAT_FRAMES,
        le=MAX_FLAT_FRAMES,
        description="The number of frames of the set. More frames make a quieter flat.",
    )
    target_fraction: float = Field(
        0.5,
        ge=MIN_FLAT_TARGET,
        le=MAX_FLAT_TARGET,
        description="The level that the session aims at in the middle of the frame, as a fraction "
        "of the full scale of the sensor.",
    )
    set_number: Literal[1, 2] = Field(
        1,
        description="`1` starts a session. `2` is the second set, with the light source turned by "
        "180 degrees since the first set, and it combines with the first set of the session. "
        "Take it within 24 hours of the first set.",
    )
    pause_after: bool = Field(
        True,
        description="Pause the scheduler when the task ends, so that nothing records data while "
        "the light source may still cover the camera. `Resume` continues.",
    )
    immediate: bool = Field(
        True,
        description="Start at the next step of the scheduler, because someone holds the light "
        "source at the camera. A survey exposure in progress finishes first. Without it, the "
        "session waits for the next cycle boundary, as the other tasks do.",
    )
    priority: int = Field(0, ge=-MAX_PRIORITY, le=MAX_PRIORITY)

    def to_command(self) -> QueueFlat:
        return QueueFlat(
            frames=self.frames,
            target_fraction=self.target_fraction,
            set_number=self.set_number,
            pause_after=self.pause_after,
            priority=self.priority,
            immediate=self.immediate,
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


class FocusResetResponse(_Response):
    """The answer to the reset of the best focus value."""

    reset: bool = Field(description="`true` when `core` restarted the best focus value.")
    message: str


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


# --- Flat ------------------------------------------------------------------------------------


class FlatPointResponse(_Response):
    """The flat at one radius from the optical center, against the center, in percent."""

    radius_deg: float
    change_percent: float | None = Field(
        description="Negative: this radius gets less light than the center."
    )
    corner: bool = Field(description="True for the point at the corner of the frame.")


class FlatTiltResponse(_Response):
    """A plane across the frame: the change from one edge to the opposite edge, in percent."""

    width_percent: float | None
    height_percent: float | None


class FlatShadowResponse(_Response):
    """A dust shadow: its place in pixels of the sensor, its depth in percent, and its width."""

    x_px: int
    y_px: int
    depth_percent: float
    width_px: float


class FlatSetResponse(_Response):
    """One set of frames of a flat: its exposure, its level, and the frames that counted."""

    number: int
    exposure_s: float | None
    level_fraction: float | None = Field(description="Of the full scale, above the bias.")
    frames: int
    used: int = Field(description="The frames that passed the checks of the flat.")
    dropped: dict[str, int] = Field(
        description="The frames that failed a check, counted for each reason."
    )
    noise_percent: float | None
    tilt: FlatTiltResponse | None


class FlatAgreementResponse(_Response):
    """How well two sets agree, which shows what the light source adds to the flat."""

    smooth_rms_percent: float | None
    fine_rms_percent: float | None
    expected_fine_rms_percent: float | None
    plane: FlatTiltResponse | None = Field(description="The difference in tilt between the sets.")


class FlatResponse(_Response):
    """One flat of the library, with the numbers of its report.

    `state` is `pending` (a session made it, and nothing uses it yet) or `approved` (you activated
    it once). `active` says that the survey divides by it now, and `pending` that it waits for your
    decision. `corner_percent` is the change in the corners against the center (negative: the
    corners get less light). `optics_tilt` and `source_tilt` exist for a flat of two sets: the tilt
    that stays with the optics, and the one that turned with the light source. `image_url` is the
    preview, stretched to plus and minus 10 percent around 1, or `null` without one.
    """

    version: str = Field(
        description="The name of the flat. It is also its ID in `/flat/{version}`."
    )
    t_utc: str
    age_days: float
    state: str
    active: bool
    pending: bool
    mode: str
    gain: int
    width_px: int
    height_px: int
    sensor_temperature_c: float | None
    exposure_s: float | None
    target_fraction: float | None
    second_set: bool
    source_turned: bool
    frames_taken: int
    frames_used: int
    noise_percent: float | None = Field(description="The noise of the flat, in percent.")
    bias_source: str
    bias_note: str
    corner_percent: float | None
    vignetting: list[FlatPointResponse]
    tilt: FlatTiltResponse
    optics_tilt: FlatTiltResponse | None
    source_tilt: FlatTiltResponse | None
    shadows: int = Field(description="The number of dust shadows that the report found.")
    shadow_min_depth_percent: float | None
    shadow_items: list[FlatShadowResponse]
    edge_artifacts: int
    agreement: FlatAgreementResponse | None
    sets: list[FlatSetResponse]
    warnings: list[str] = Field(description="What the session and the combination noticed.")
    has_image: bool
    image_url: str | None
    activated_utc: str | None


class FlatSessionResponse(_Response):
    """The first set of a session, which waits for a second set with the source turned."""

    version: str = Field(description="The pending flat that the first set made.")
    t_utc: str
    expires_utc: str = Field(description="When the frames of the first set go.")
    frames: int
    exposure_s: float


class FlatTaskResponse(_Response):
    """The latest flat session of this `core` process.

    `state` is `idle` (none yet), `queued`, `running`, `ok`, `failed`, or `aborted`. While it
    runs, `phase` is `setup`, `exposure` (the search for the exposure, at most `steps` tries),
    `capture` (the frames), or `build` (the combination), with `step` of `steps` in that phase.
    `exposure_s` is the exposure in use, `level_fraction` the latest level above the bias as a
    fraction of the full scale (aim: `target_fraction`), and `saturated_fraction` the share of
    saturated pixels of the latest frame. `warnings` list what the session noticed. A finished
    session keeps its `summary` (one sentence) and the `version` of the flat that it added.
    """

    state: str
    task_id: int | None
    phase: str | None
    step: int
    steps: int
    message: str
    set_number: int
    frames: int | None
    target_fraction: float | None
    exposure_s: float | None
    level_fraction: float | None
    saturated_fraction: float | None
    warnings: list[str]
    pause_after: bool
    started_utc: str | None
    finished_utc: str | None
    summary: str
    version: str | None


class FlatLibraryResponse(_Response):
    """The flat library, the flat in use, and the latest flat session.

    `mode` and `gain` are the settings of the survey, which the session uses. `blocker` is a
    sentence that says why no session can start (the dark library holds no set), or `null`.
    `active_version` is the flat in use, and `pending_version` the newest flat that waits for a
    decision. `flat_file_pinned` is true when the configuration names `[survey] flat_file`, and
    `library_overrides` says that the active flat of the library wins over that file. `session`
    exists while the first set of a session waits for a second set. `flats` holds the newest flats
    first. A value that `core` does not report is `null`, and `quality` says why.
    """

    mode: str
    gain: int
    sensor_temperature_c: float | None
    active_version: str | None
    pending_version: str | None
    flat_file_pinned: bool
    library_overrides: bool
    blocker: str | None
    flats: list[FlatResponse]
    session: FlatSessionResponse | None
    task: FlatTaskResponse
    quality: dict[str, str] | None = None


class FlatActionResponse(_Response):
    """The answer to the activation or the deletion of a flat."""

    message: str
    version: str


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
    """Where the scheduler stands in a fault episode of the camera."""

    failures: int = Field(description="The failures in a row.")
    good_frames: int = Field(description="The good frames since the last failure.")
    last_error: str | None = Field(None, description="The latest error, as the driver reported it.")
    next_attempt: str | None = Field(None, description="When the scheduler tries the next step.")
    next_step: str | None = Field(
        None, description="The next step of the recovery ladder, such as `reopen`."
    )
    cause: str | None = Field(
        None,
        description=(
            "The best explanation of the episode: `timeout` (no frame arrived), `disconnected` "
            "(the driver finds no camera), `link` (the scheduler cannot reach `acquire`), or "
            "`error`. It is `null` without an episode."
        ),
    )
    reason: str | None = Field(
        None,
        description=(
            "The cause in words, such as `no frame arrived; the camera may be disconnected`."
        ),
    )
    since: str | None = Field(None, description="When the episode began.")


class ActivityResponse(_Response):
    """What the scheduler does now, for how long, and what comes next, in plain words.

    A value that the scheduler does not know is `null`. The two times that look ahead are
    expectations: a fast period can end early, and a wait for a pointing solution ends when the
    solution arrives.
    """

    state: str = Field(description="The state: `safe`, `auto`, `align`, `commission`, or `paused`.")
    phase: str = Field(
        description=(
            "What the scheduler does within the state. `auto` has `fast` (the fast stream), "
            "`survey_short` and `survey_long` (the two exposures of the survey step), "
            "`solve_wait` (it waits for a pointing solution), and `idle` (the camera rests "
            "until the next slot of the cycle). The other states have `watch` (the brightness "
            "watch of `safe`), `align`, `commission`, and `paused`. `camera_fault` replaces the "
            "phase while the scheduler waits to try a recovery step of the camera."
        )
    )
    label: str = Field(description="The activity in words, such as `Fast stream: seeing windows`.")
    since_utc: str = Field(description="When the activity began.")
    ends_utc: str | None = Field(
        None,
        description=(
            "When the activity ends, if the scheduler knows: the end of the fast period or of an "
            "exposure, the wait for the next slot, or the idle timeout of the alignment."
        ),
    )
    next_label: str | None = Field(None, description="The activity that follows.")
    next_utc: str | None = Field(None, description="When the next activity starts.")
    cadence_s: float | None = Field(
        None,
        description=(
            "The length of the cycle in force, in seconds. It is shorter under clouds, and it is "
            "`null` outside `auto`."
        ),
    )
    detail: str | None = Field(
        None, description="A phrase that adds to the label, such as the windows that have closed."
    )
    reason: str | None = Field(None, description="Why the state holds, in words.")


class SchedulerStatusResponse(_Response):
    """The scheduler as `core` reports it.

    The status leaves out the Sun's elevation, because a series of elevations shows the site.
    `activity` says what the scheduler does now and what comes next.
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
    activity: ActivityResponse | None = None


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
    polaris_max_fps: float = Field(
        description="The most frames per second that the server sends of the video of Polaris."
    )
    polaris_stall_s: float = Field(
        description="The seconds without a frame after which the video counts as stalled."
    )


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


# --- The rolling seeing value ----------------------------------------------------------------


class LiveSeeingResponse(LiveSeeingView):
    """The rolling seeing value of the fast stream: a provisional number that `core` does not store.

    `core` estimates the seeing from the newest `span_s` seconds of frames with the estimator of the
    stored windows, and it repeats the estimate every few seconds while the fast stream runs, which
    is about three quarters of each cycle. Between fast periods the value keeps its time, so
    `age_s` shows how old it is. The `seeing_fwhm_*` and `r0_*` values follow the records of the
    `seeing` series: the first of each pair comes from the variance of the motion, and the second
    from its structure function. `valid_fraction` is the share of the frames of the span (the frames
    that the camera lost included) that have a usable centroid, and `flags` holds the window flags
    that apply. A value that `core` cannot give is `null`, and `quality` says why.
    """

    t_utc: str = Field(description="The end of the span, as an ISO 8601 UTC string.")
    age_s: float = Field(
        ge=0, description="The seconds from the end of the span to the answer of the server."
    )


# --- Images ----------------------------------------------------------------------------------


class ImageItem(_Response):
    """One image."""

    id: str = Field(description="The ID. Pass it to `GET /images/{id}`.")
    kind: str = Field(
        description=(
            "What the frame is: `survey` (a long exposure), `short` (the short exposure of a "
            "step), or `event` (a frame that the station kept because of an event)."
        )
    )
    t_utc: str
    t_utc_ns: int
    size_bytes: int = Field(description="The size of the JPEG preview, in bytes.")
    has_fits: bool = Field(description="Whether the station kept the frame as a FITS file.")
    fits_bytes: int | None = Field(description="The size of the FITS file, or `null` without one.")
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
