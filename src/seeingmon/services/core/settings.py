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

    survey_worker: SurveyWorkerSettings = Field(default_factory=SurveyWorkerSettings)
    escalation: EscalationSettings = Field(default_factory=EscalationSettings)
    commissioning: CommissioningSettings = Field(default_factory=CommissioningSettings)


class AlignmentSettings(SectionModel):
    """The `[alignment]` section: the live view and the quick solve of the alignment helper.

    The target is where Polaris belongs in the frame of the alignment stream (the survey readout
    mode), and the roll that the camera should have. Take both from a reference solution at
    commissioning. Without a target, the helper still shows the live view and the solved position,
    and it leaves the offset out.

    The preview holds at most `max_preview_pixels` pixels, and the helper sends at most one frame
    each `min_interval_s`. A solve starts at most each `solve_interval_s`, and a solved position
    older than `solution_max_age_s` no longer counts as current. A frame has a saturation warning
    when more than `saturation_warn_fraction` of its pixels reach `saturation_level` of the
    saturation level of the readout mode.
    """

    target_x_px: float | None = None
    target_y_px: float | None = None
    target_roll_deg: float | None = None

    jpeg_quality: int = Field(80, ge=10, le=95)
    max_preview_pixels: int = Field(1_000_000, ge=10_000)
    min_interval_s: float = Field(0.1, ge=0)

    solve_interval_s: float = Field(1.0, ge=0)
    solution_max_age_s: float = Field(10.0, gt=0)

    histogram_bins: int = Field(64, ge=8, le=256)
    saturation_level: float = Field(0.98, gt=0, le=1)
    saturation_warn_fraction: float = Field(0.0005, ge=0, le=1)

    # How often `core` tells the scheduler that someone watches, while a live view is open.
    touch_interval_s: float = Field(5.0, gt=0)

    @model_validator(mode="after")
    def _target_is_whole(self) -> AlignmentSettings:
        if (self.target_x_px is None) != (self.target_y_px is None):
            raise ValueError("set target_x_px and target_y_px together")
        return self

    @property
    def has_target(self) -> bool:
        """Whether the configuration gives a target position."""
        return self.target_x_px is not None and self.target_y_px is not None
