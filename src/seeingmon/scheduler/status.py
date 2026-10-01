"""The scheduler's status: a plain snapshot for `/status` and for the `health` record.

`Scheduler.status()` returns a `SchedulerStatus`. It is a frozen dataclass of plain values, so
`dataclasses.asdict` turns it into JSON. The snapshot is best effort: the scheduler thread keeps
running while another thread reads it, so two fields can differ in age by one step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from seeingmon.frames import Roi


@dataclass(frozen=True, slots=True)
class StreamInfo:
    """The stream that the camera runs or ran last.

    `purpose` is one of `fast`, `survey`, `watch`, `align`, and `commission`.
    """

    stream_id: int
    purpose: str
    mode: str
    exposure_us: int
    gain: int
    roi: Roi | None


@dataclass(slots=True)
class Counters:
    """Counts since the scheduler started. A status holds a copy that no later step changes."""

    frames: int = 0  # frames read from the camera in every activity
    dropped: int = 0  # frames lost, summed from `dropped_before`
    reconfigurations: int = 0
    fast_periods: int = 0
    cadence_overruns: int = 0  # cycles that started more than a second after their slot
    windows: int = 0  # seeing windows written
    early_window_ends: int = 0  # periods that ended early, at an edge or for a missing star
    roi_recenters: int = 0
    solves_requested: int = 0
    survey_steps: int = 0
    survey_frames: int = 0
    survey_results: int = 0
    survey_unsolved: int = 0
    survey_skipped: int = 0
    watch_frames: int = 0
    faults: int = 0
    recovery_steps: int = 0
    escalations: int = 0
    commands_accepted: int = 0
    commands_rejected: int = 0
    tasks_run: int = 0
    transitions: int = 0


@dataclass(frozen=True, slots=True)
class FaultStatus:
    """Where the scheduler stands in a fault episode."""

    failures: int = 0  # failures in a row
    good_frames: int = 0  # good frames since the last failure
    last_error: str | None = None
    next_attempt_utc_ns: int | None = None  # when the scheduler will try the recovery step
    next_step: str | None = None  # the name of that step, such as `reopen`


@dataclass(frozen=True, slots=True)
class SchedulerStatus:
    """The state of the scheduler at one moment.

    `degraded` means that the camera failed repeatedly. The scheduler keeps retrying slowly, and
    the store and `web` stay up. `queued_tasks` counts the commissioning tasks that wait, and
    `survey_pending` counts the survey frames that await analysis.
    """

    t_utc_ns: int
    state: str
    state_reason: str
    state_since_utc_ns: int
    last_transition_utc_ns: int | None
    degraded: bool
    stream: StreamInfo | None
    cloud: bool
    cloud_fraction: float | None
    twilight: bool
    sun_elevation_deg: float | None
    background_fraction: float | None
    sensor_temperature_c: float | None
    counters: Counters
    fault: FaultStatus
    queued_tasks: int
    survey_pending: int
    alignment_idle_s: float | None = None

    @property
    def camera_component(self) -> str:
        """The state of the camera for the `components` field of a health record."""
        if self.degraded:
            return "failed"
        return "degraded" if self.fault.failures else "ok"

    def health_fields(self) -> dict[str, Any]:
        """The fields of a `health` record that the scheduler knows.

        Pass them to `HealthRecord` together with the fields that only the core knows, such as
        `dark_due`, the free space, and the sink backlog. The core adds its own components to
        `components`.
        """
        return {
            "state": self.state,
            "degraded": self.degraded,
            "components": {
                "scheduler": "degraded" if self.degraded else "ok",
                "camera": self.camera_component,
            },
            "sensor_temperature_c": self.sensor_temperature_c,
            "dropped_total": self.counters.dropped,
        }
