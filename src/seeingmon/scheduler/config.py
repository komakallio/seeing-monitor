"""Configuration of the scheduler: the `[scheduler]` table and the observing site.

Read the table with `config.section("scheduler", SchedulerConfig)`. The defaults live in
`config/default.d/scheduler.toml`, and the models below carry the same values, so
`SchedulerConfig()` works without a file. A test keeps the two in step.

Every default is provisional. Phase 3 (commissioning) finds the final exposure, gain, ROI, and
cadence values under a real sky. Keep the values in the models and the file aligned with the
architecture until then, and override them in `local/config.toml` or with environment variables
such as `SEEINGMON_SCHEDULER__FAST__WINDOW_S=60`.

Hardware values do not appear here. The readout modes, the saturation levels, and the ROI sizes
come from the profile.
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from seeingmon.config import Config, SectionModel
from seeingmon.scheduler.levels import STEP_NAMES

Seconds = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Finite = Annotated[float, Field(allow_inf_nan=False)]
Fraction = Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]

_US_PER_S = 1_000_000
MIN_DARK_FRAMES = 3  # a dark set needs at least this many dark frames and bias frames


def seconds_to_us(seconds: float) -> int:
    """Convert seconds to whole microseconds, with a floor of one microsecond."""
    return max(1, round(seconds * _US_PER_S))


class FastConfig(SectionModel):
    """The fast stream: short exposures of a small ROI that follows Polaris."""

    window_s: Seconds = 120.0
    """The length of one fast period, in seconds. Use a multiple of `analysis_window_s`."""

    analysis_window_s: Seconds = 60.0
    """The window length that the fast analyzer uses, in seconds.

    The analyzer owns its window length (`window_s` in `[fastpath]`), and the scheduler
    never reads that section. Set this key to the same value, so that a fast period holds
    whole windows."""

    exposure_us: PositiveInt = 2000
    gain: NonNegativeInt = 0
    high_speed: bool = False
    """Read the fast stream in the high-speed mode of the camera. The frames come faster and the
    ADC depth falls (bin1: 10 bits, not 12). The profile's fast readout mode must have high-speed
    values."""

    roi_arcmin: Seconds = 4.1
    """The ROI width and height as an angle. The profile turns it into pixels."""

    roi_edge_margin_px: NonNegative = 16.0
    """A star closer than this to the ROI edge ends the window and recenters the ROI."""

    edge_cooldown_s: NonNegative = 5.0
    """The shortest stream time between two edge recenters. It stops a loop at the sensor edge."""

    missing_star_frames: PositiveInt = 450
    """The number of frames in a row without a star that triggers a survey solve."""

    resolve_interval_s: NonNegative = 60.0
    """The shortest time between two solves that a missing star triggers."""

    @model_validator(mode="after")
    def _window_holds_an_analysis_window(self) -> Self:
        if self.analysis_window_s > self.window_s:
            raise ValueError("analysis_window_s must not exceed window_s")
        return self


class SurveyConfig(SectionModel):
    """The survey step: a short exposure for bright stars, then a long one for the faint stars."""

    cadence_s: Seconds = 180.0
    """The time from the start of one cycle to the start of the next, in seconds."""

    short_exposure_s: Seconds = 0.001
    short_gain: NonNegativeInt = 0
    long_exposure_s: Seconds = 30.0
    long_gain: NonNegativeInt = 120

    max_pending: PositiveInt = 4
    """The step is skipped while this many survey frames wait for analysis."""

    solve_wait_s: Seconds = 120.0
    """How long to wait for a pointing solution after a survey step, in seconds."""

    solve_retry_s: Seconds = 60.0
    """The pause between two solve attempts when the pointing stays unsolved."""

    @property
    def short_exposure_us(self) -> int:
        return seconds_to_us(self.short_exposure_s)

    @property
    def long_exposure_us(self) -> int:
        return seconds_to_us(self.long_exposure_s)


class WatchConfig(SectionModel):
    """The brightness watch: one short frame at a fixed interval while the camera is idle."""

    exposure_us: PositiveInt = 1000
    gain: NonNegativeInt = 0
    interval_s: Seconds = 60.0
    roi_arcmin: NonNegative = 20.0
    """The width and height of the central ROI, or 0 for the full frame."""


class DaylightConfig(SectionModel):
    """The daylight gate. The Sun's elevation gates the attempt, and the measured sky overrides."""

    sun_elevation_limit_deg: Finite = -3.0
    """Above this elevation the scheduler stays in `safe`."""

    sun_resume_margin_deg: NonNegative = 1.0
    """After `safe`, the Sun must sink this far below the limit before `auto` resumes."""

    twilight_elevation_deg: Finite = -18.0
    """While the Sun is above this elevation, windows and survey results carry `twilight`."""

    saturation_limit: Fraction = 0.5
    """A background above this share of saturation at the shortest exposure forces `safe`."""

    resume_saturation: Fraction = 0.35
    """After `safe`, the background must fall below this share before `auto` resumes."""

    @model_validator(mode="after")
    def _thresholds_are_ordered(self) -> Self:
        if self.resume_saturation >= self.saturation_limit:
            raise ValueError("resume_saturation must be below saturation_limit")
        if self.twilight_elevation_deg >= self.sun_elevation_limit_deg:
            raise ValueError("twilight_elevation_deg must be below sun_elevation_limit_deg")
        return self


class CloudConfig(SectionModel):
    """The response to clouds. The survey analysis reports the cloud fraction."""

    threshold: Fraction = 0.5
    """A cloud fraction at or above this value starts the cloud response."""

    clear_threshold: Fraction = 0.3
    """A cloud fraction at or below this value ends it."""

    fast_window_s: Seconds = 60.0
    """The length of the fast period under cloud, in seconds."""

    survey_cadence_s: Seconds = 100.0
    """The survey cadence under cloud, in seconds."""

    @model_validator(mode="after")
    def _thresholds_are_ordered(self) -> Self:
        if self.clear_threshold >= self.threshold:
            raise ValueError("clear_threshold must be below threshold")
        return self


class AlignConfig(SectionModel):
    """The alignment stream: a low-latency live view for adjusting the mount."""

    idle_timeout_s: Seconds = 1800.0
    """`align` ends after this long without a command or a call to `touch_alignment`."""

    exposure_s: Seconds = 0.5
    gain: NonNegativeInt = 120


class FaultConfig(SectionModel):
    """The response to camera errors: back off, climb the ladder, and finally degrade."""

    backoff_initial_s: Seconds = 2.0
    backoff_factor: Annotated[float, Field(ge=1, allow_inf_nan=False)] = 2.0
    backoff_max_s: Seconds = 60.0

    degraded_after: PositiveInt = 5
    """The number of failures in a row after which the status turns `degraded`. A camera that the
    driver reports as not connected turns it at once."""

    slow_retry_s: Seconds = 600.0
    """The pause between attempts once the quick steps of the ladder (up to the restart of
    `acquire`) have had their attempts, in seconds. A reboot or a power cycle waits this long too.
    Until then, the pause is the backoff."""

    clear_after_frames: PositiveInt = 10
    """The number of good frames in a row that clear the failure count and `degraded`."""

    @model_validator(mode="after")
    def _backoff_is_ordered(self) -> Self:
        if self.backoff_max_s < self.backoff_initial_s:
            raise ValueError("backoff_max_s must not be below backoff_initial_s")
        return self


class LadderConfig(SectionModel):
    """The limits of the recovery ladder (see `seeingmon.scheduler.levels`)."""

    attempts_per_level: PositiveInt = 2
    """The number of failed cycles at one step before the scheduler moves to the next."""

    max_level: str = "power_cycle"
    """The highest step to request: one of the lowercase step names."""

    destructive_interval_s: NonNegative = 21_600.0
    """The shortest time between two reboots or power cycles, in seconds."""

    @field_validator("max_level")
    @classmethod
    def _known_step(cls, value: str) -> str:
        if value not in STEP_NAMES:
            raise ValueError(f"max_level must be one of {', '.join(STEP_NAMES)}")
        return value


class CommissionConfig(SectionModel):
    """The commissioning queue."""

    max_queued: PositiveInt = 8
    """The scheduler rejects a new task while this many tasks wait."""

    max_results: PositiveInt = 50
    """The number of finished results that the scheduler keeps in memory."""


class DarkTaskConfig(SectionModel):
    """The limits of a dark task (`QueueDark`). The defaults of the session are `[survey.dark]`."""

    max_frames: Annotated[int, Field(ge=MIN_DARK_FRAMES)] = 60
    """The most dark frames, and the most bias frames, that one task may ask for."""

    max_label_chars: PositiveInt = 80
    """The longest label that a dark task may carry, in characters."""

    max_cover_wait_s: Seconds = 7200.0
    """The longest wait for the cover that a dark task may ask for, in seconds."""


class SweepConfig(SectionModel):
    """The default grid of a sweep. The values explore a range and choose nothing."""

    window_s: Seconds = 10.0
    """The length of the fast window in each cell, in seconds."""

    exposure_us: tuple[PositiveInt, ...] = (500, 1000, 2000, 5000, 10000)
    gain: tuple[NonNegativeInt, ...] = (0, 60, 120)
    roi_arcmin: tuple[Seconds, ...] = (4.1,)

    modes: tuple[str, ...] = ()
    """The readout modes to sweep. An empty list means the profile's fast mode."""

    max_cells: PositiveInt = 200

    @field_validator("exposure_us", "gain", "roi_arcmin")
    @classmethod
    def _axis_is_not_empty(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value:
            raise ValueError("a sweep axis needs at least one value")
        return value


class LoopConfig(SectionModel):
    """Timing of the run loop."""

    max_sleep_s: Seconds = 0.25
    """The longest single sleep, so commands and the stop event get attention in time."""

    poll_interval_s: Seconds = 0.1
    """The shortest time between two polls of the survey analyzer."""

    read_timeout_factor: Annotated[float, Field(ge=1, allow_inf_nan=False)] = 2.0
    read_timeout_margin_s: NonNegative = 0.5
    """A frame read waits `factor` times the frame period plus this margin, in seconds.

    The driver reports the frame period. For a single exposure it follows the snapshot model of
    the profile, which is much longer than the video model of the same ROI. Raise the margin only
    for a camera or host that is slower than the profile says.
    """

    context_refresh_s: Seconds = 30.0
    """How often the scheduler refreshes the context that it gives the fast analyzer."""

    stall_s: Seconds = 10.0
    """A sleep that returns this much later than it should writes the event `scheduler.stalled`.

    The loop sleeps for fractions of a second, so a sleep that takes much longer means that the
    process did not run: the machine was suspended, or something held it. The value is far above the
    jitter of a loaded machine."""


class SchedulerConfig(SectionModel):
    """The `[scheduler]` table."""

    fast: FastConfig = Field(default_factory=FastConfig)
    survey: SurveyConfig = Field(default_factory=SurveyConfig)
    watch: WatchConfig = Field(default_factory=WatchConfig)
    daylight: DaylightConfig = Field(default_factory=DaylightConfig)
    cloud: CloudConfig = Field(default_factory=CloudConfig)
    align: AlignConfig = Field(default_factory=AlignConfig)
    faults: FaultConfig = Field(default_factory=FaultConfig)
    ladder: LadderConfig = Field(default_factory=LadderConfig)
    commission: CommissionConfig = Field(default_factory=CommissionConfig)
    dark: DarkTaskConfig = Field(default_factory=DarkTaskConfig)
    sweep: SweepConfig = Field(default_factory=SweepConfig)
    loop: LoopConfig = Field(default_factory=LoopConfig)


class SiteConfig(SectionModel):
    """The observing site, from the `[site]` table of `local/config.toml`.

    The scheduler reads the latitude and the longitude to compute the Sun's elevation. Other
    parts of the system may add keys to the table, so this model ignores keys that it does not
    know. Latitude is north-positive and longitude is east-positive.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    latitude_deg: Annotated[float, Field(ge=-90, le=90, allow_inf_nan=False)]
    longitude_deg: Annotated[float, Field(ge=-180, le=180, allow_inf_nan=False)]
    elevation_m: Finite = 0.0


def load_site(config: Config) -> SiteConfig | None:
    """Read the `[site]` table. Returns `None` when the configuration has no such table.

    Raises `ConfigError` when the table exists and is not valid. Without a site, the scheduler
    relies on the measured sky background only and sets no `twilight` flag.
    """
    if "site" not in config.effective(redact=False):
        return None
    return config.section("site", SiteConfig)
