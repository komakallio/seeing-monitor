"""The settings of `core`: the `[services.core]` tables and the `[alignment]` section.

The defaults live in `config/default.d/services.toml`. Override a value on one station in
`local/config.toml`, or with an environment variable such as
`SEEINGMON_SERVICES__CORE__HEALTH_INTERVAL_S`. A value that belongs to one installation (the
target position of the alignment, the command that reboots the machine) has no default that works:
set it in the untracked local file.

    config = load_config()
    services = config.section("services", ServicesConfig)  # services.core holds CoreSettings
    alignment = config.section("alignment", AlignmentSettings)
"""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from seeingmon.config import SectionModel


class SurveyWorkerSettings(SectionModel):
    """How `core` runs the survey analysis.

    `process` runs it in a worker process at a lower priority, and on Linux with a raised
    `oom_score_adj`, so that the out-of-memory killer takes the worker before it takes `core`.
    `thread` runs it in a thread of `core`, and `inline` runs each job in the scheduler thread,
    which keeps a test deterministic. `nice` is the niceness on Linux and selects a below-normal
    priority class on Windows. `oom_score_adj` applies on Linux only.
    """

    mode: Literal["process", "thread", "inline"] = "process"
    nice: int = Field(10, ge=0, le=19)
    oom_score_adj: int = Field(500, ge=0, le=1000)


class EscalationSettings(SectionModel):
    """The steps of the recovery ladder that `core` performs for the scheduler.

    `reboot_command` is an argument list that `core` runs without a shell to reboot the machine.
    It is empty by default, so no reboot happens until you name the command in the local
    configuration. The power cycle has its own `[power]` section (`seeingmon.hardware.power`).
    """

    restart_acquire_wait_s: float = Field(15.0, gt=0)
    reboot_command: list[str] = Field(default_factory=list)
    command_timeout_s: float = Field(30.0, gt=0)


class CommissioningSettings(SectionModel):
    """The limits of the burst and replay handlers.

    A burst records raw frames to the `bursts` folder, so its length has a limit. A replay writes
    its results to a separate store in `replays_dir`, a folder under the data directory.
    """

    burst_max_duration_s: float = Field(600.0, gt=0)
    replays_dir: str = Field("replays", min_length=1, pattern=r"^[A-Za-z0-9._-]+$")


class ReplaySettings(SectionModel):
    """The `[replay]` section: where the recordings that a replay may read live.

    The section also holds keys that other parts read, so this model ignores them.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    recordings_dir: str = ""


class SkyFlagSettings(SectionModel):
    """When `core` sets the `moon` and `dew` flags of a `sky_quality` record.

    The scheduler sets `twilight`, `cloud`, and `time_invalid` on a survey record. `core` adds the
    two flags that need more than the scheduler knows. The Moon counts when its center is above
    `moon_min_elevation_deg` and at least `moon_min_illumination` of its disk is lit, because a thin
    crescent adds little to the sky. The optics count as dewy when their temperature (the optics
    sensor, or the air when there is none) is within `dew_margin_c` of the dew point.
    """

    moon_min_elevation_deg: float = Field(0.0, ge=-90.0, le=90.0)
    moon_min_illumination: float = Field(0.1, ge=0.0, le=1.0)
    dew_margin_c: float = Field(1.0, ge=0.0, le=20.0)


class SurveyFrameSettings(SectionModel):
    """What `core` keeps of the survey frames: previews, FITS files, and the frames in RAM.

    The newest `ram_frames` frames stay in the memory of `core`, and two are needed: a frame stays
    until its result comes back and its files are written, and the result of the short frame of a
    survey step comes back only after the long exposure that follows it has arrived. Nothing reads
    the frames beyond that, because `web` is another process and cannot see the memory of `core`.
    A frame that is a long exposure (at least `[survey.sky] min_exposure_s`) gets a preview. Every
    `keep_every`-th long frame, and every event frame, also goes to disk as a FITS file. An event
    frame is a frame at the start of one of these conditions: no pointing solution (`unsolved`), a
    pointing that moved (`moved`), clouds (a cloud fraction of at least `event_cloud_fraction`), or
    a sky so bright that the background reaches `event_background_fraction` of the saturation
    level. Each kind of frame (long and short) gets at most one event frame in
    `event_min_interval_s`, so that a night of flickering clouds cannot fill the card.

    A preview is a JPEG of at most `preview_max_pixels` pixels. `fits_compression` is `rice`
    (the files are about 40% of the frame, and the writer falls back to `none` with a warning when
    the astropy compressor is missing) or `none`.
    """

    enabled: bool = True
    ram_frames: int = Field(2, ge=1, le=16)
    keep_every: int = Field(10, ge=1)
    jpeg_quality: int = Field(80, ge=10, le=95)
    preview_max_pixels: int = Field(1_000_000, ge=10_000)
    fits_compression: Literal["rice", "none"] = "rice"
    event_min_interval_s: float = Field(3600.0, ge=0)
    event_cloud_fraction: float = Field(0.5, ge=0, le=1)
    event_background_fraction: float = Field(0.5, gt=0, le=1)


class PolarisSettings(SectionModel):
    """The live video of Polaris: how `core` thins, stretches, and encodes the fast stream.

    While a person watches, `core` keeps about `max_fps` frames per second of frame time from the
    fast stream (the camera runs at about 80), stretches each one, and encodes it as a PNG. The
    stretch is black at the median of the frame, and white at `headroom` times the peak of the star
    above the black level, which `core` follows with an exponential average over `time_constant_s`
    seconds. The slow average keeps the flicker of the star visible, because a stretch that
    followed every frame would normalize it away. The brightest pixel of a star that the pixels
    undersample varies by tens of percent as the star moves across them, so `headroom` is 1.5:
    with 1.15, a third of the frames of a simulated star clip at white. The white level never falls
    below `floor_sigmas` times the noise of the frame, so a frame with no star shows noise and not
    a stretched speck. An `asinh` curve with the gain `asinh_gain` lifts the faint wings of the
    star.

    After an error in the code that runs on the scheduler thread, `core` stops offering frames to
    the video for `disable_s` seconds, so that a fault of the video never reaches the scheduler.
    """

    max_fps: float = Field(20.0, gt=0, le=60)
    time_constant_s: float = Field(3.0, gt=0, le=60)
    headroom: float = Field(1.5, ge=1.0, le=4.0)
    floor_sigmas: float = Field(8.0, gt=0, le=1000)
    asinh_gain: float = Field(30.0, gt=0, le=1000)
    disable_s: float = Field(60.0, ge=0)


class CoreSettings(SectionModel):
    """The `[services.core]` table."""

    # How often `core` writes a `health` record, and how often it collects the hardware events of
    # the driver in `acquire`.
    health_interval_s: float = Field(60.0, gt=0)
    events_interval_s: float = Field(5.0, gt=0)

    # The `run` record names the camera, which `core` learns when the scheduler opens it. After
    # this wait, `core` writes the record without the camera.
    run_record_wait_s: float = Field(30.0, ge=0)

    # How long the shutdown may take before `core` gives up on a thread.
    shutdown_timeout_s: float = Field(30.0, gt=0)

    # The watchdog of systemd. `core` stops sending `WATCHDOG=1` when the scheduler thread makes no
    # progress for `scheduler_stall_s`, outside a call of the camera driver. A driver call has its
    # own limit, `driver_call_limit_s`, which exceeds the longest timeout of the remote driver. A
    # call that outlasts the limit counts as a hang.
    scheduler_stall_s: float = Field(60.0, gt=0)
    driver_call_limit_s: float = Field(240.0, gt=0)

    # The most RPC clients at once: `web`, and the commands `burst`, `sweep`, and `replay`.
    max_rpc_connections: int = Field(8, ge=1, le=64)

    # A JSON file with a pointing solution (see `seeingmon.services.simsky.write_seed`) that
    # starts the pointing tracker. A real installation gets its first solution from a plate
    # solver, and a development run on a simulated sky needs no solver with this file.
    seed_solution_file: str = ""

    # At start, with no seed file, `core` gives the pointing tracker the newest solved `pointing`
    # record of the store that the tracker could still use (`[survey.pointing] validity_s`), so
    # that a restart goes on with the solution of the last run and skips the blind first solve.
    seed_from_store: bool = True

    survey_worker: SurveyWorkerSettings = Field(default_factory=SurveyWorkerSettings)
    survey_frames: SurveyFrameSettings = Field(default_factory=SurveyFrameSettings)
    polaris: PolarisSettings = Field(default_factory=PolarisSettings)
    escalation: EscalationSettings = Field(default_factory=EscalationSettings)
    commissioning: CommissioningSettings = Field(default_factory=CommissioningSettings)
    sky_flags: SkyFlagSettings = Field(default_factory=SkyFlagSettings)


class AlignmentSettings(SectionModel):
    """The `[alignment]` section: the live view and the quick solve of the alignment helper.

    The target is where Polaris belongs in the frame of the alignment stream (the survey readout
    mode), and the roll that the camera should have. Take both from a reference solution at
    commissioning. Without a target, the helper still shows the live view and the solved position,
    and it leaves the offset out. The live view needs no target: it aims the pole at the center of
    the frame.

    The aim is where the pole should go, in pixels of the same frame. The default is the center of
    the frame, and `aim_x_px` and `aim_y_px` (set together) name another pixel. The reticle of the
    live view, the dashed circle with the radius of the orbit of Polaris, is centered on the aim.

    The preview holds at most `max_preview_pixels` pixels, and the helper sends at most one frame
    each `min_interval_s`. A frame has a saturation warning when more than
    `saturation_warn_fraction` of its pixels reach `saturation_level` of the saturation level of
    the readout mode.

    **The quick solve.** The solver takes the newest frame as soon as it has finished the previous
    solve. `solve_interval_s` spaces the starts of the solves (0, the default, means back to back),
    for a machine that has no CPU to spare. A solved position older than `solution_max_age_s` no
    longer counts as current. `solver_mode` says where the solve runs: `process` (the default) in
    a worker process, so that the detector, which holds the GIL for seconds, never stalls the live
    view, and `thread` in a thread of `core` (for a machine that cannot spare the memory of a
    second process). A solve that takes longer than `solve_timeout_s` ends the worker. The quick
    solve detects only the bright stars: a star must stand out by `detect_threshold_sigma` times
    the noise, the detector keeps the `detect_max_stars` brightest, and it searches a copy of the
    frame that sums each `detect_coarse_bin` x `detect_coarse_bin` block (1 searches the frame
    itself) and fits the brightest `detect_refine_stars` stars at full resolution (see
    `seeingmon.survey.detect`). A frame without a trail model, such as the first one after a
    start, takes the full search. The survey analysis keeps its own settings (`[survey.detect]`),
    and a lower threshold, more stars, and a smaller bin only make the quick solve slower.

    **Rapid focus.** The mode (a ROI of the fast readout mode around Polaris, with the star width in
    arcseconds 20 times a second) is offered when the coarse focus is good enough and `core` knows
    where Polaris is. The coarse focus is the median of the last five focus values of the normal
    view, and it must be at most `rapid_focus_max_fwhm_arcsec` (12 arcseconds): the aperture of the
    fast analysis (16 pixels in bin1) cuts off a wider star, so its width reads too small and the
    curve flattens. A solution that is older than `rapid_focus_max_solution_age_s` (600 seconds) no
    longer places Polaris for the mode.
    """

    target_x_px: float | None = None
    target_y_px: float | None = None
    target_roll_deg: float | None = None

    aim_x_px: float | None = None
    aim_y_px: float | None = None

    jpeg_quality: int = Field(80, ge=10, le=95)
    max_preview_pixels: int = Field(1_000_000, ge=10_000)
    min_interval_s: float = Field(0.1, ge=0)

    solver_mode: Literal["process", "thread"] = "process"
    solve_interval_s: float = Field(0.0, ge=0)
    solve_timeout_s: float = Field(60.0, gt=0)
    solution_max_age_s: float = Field(10.0, gt=0)
    detect_threshold_sigma: float = Field(8.0, gt=0)
    detect_max_stars: int = Field(300, ge=8)
    detect_coarse_bin: int = Field(2, ge=1, le=16)
    detect_refine_stars: int = Field(300, ge=1)

    histogram_bins: int = Field(64, ge=8, le=256)
    saturation_level: float = Field(0.98, gt=0, le=1)
    saturation_warn_fraction: float = Field(0.0005, ge=0, le=1)

    # How often `core` tells the scheduler that someone watches, while a live view is open.
    touch_interval_s: float = Field(5.0, gt=0)

    # The offer of the rapid focus mode.
    rapid_focus_max_fwhm_arcsec: float = Field(12.0, gt=0)
    rapid_focus_max_solution_age_s: float = Field(600.0, gt=0)

    @model_validator(mode="after")
    def _target_is_whole(self) -> AlignmentSettings:
        if (self.target_x_px is None) != (self.target_y_px is None):
            raise ValueError("set target_x_px and target_y_px together")
        if (self.aim_x_px is None) != (self.aim_y_px is None):
            raise ValueError("set aim_x_px and aim_y_px together")
        return self

    @property
    def has_target(self) -> bool:
        """Whether the configuration gives a target position."""
        return self.target_x_px is not None and self.target_y_px is not None

    @property
    def aim_xy(self) -> tuple[float, float] | None:
        """The configured aim, or `None` for the center of the frame."""
        if self.aim_x_px is None or self.aim_y_px is None:
            return None
        return self.aim_x_px, self.aim_y_px
