"""The scheduler's status: a plain snapshot for `/status` and for the `health` record.

`Scheduler.status()` returns a `SchedulerStatus`. It is a frozen dataclass of plain values, so
`dataclasses.asdict` turns it into JSON. The snapshot is best effort: the scheduler thread keeps
running while another thread reads it, so two fields can differ in age by one step.

**Time.** Every time in the status is an integer of nanoseconds since the Unix epoch, in UTC, and
its name ends in `_utc_ns`. The scheduler measures durations on the monotonic clock and converts
the moments that it reports. The REST API turns each one into an ISO 8601 string.

**Activity.** `SchedulerStatus.activity` says what the scheduler does now, for how long, and what
comes next, in words that a person at the telescope understands (see `ActivityStatus`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from seeingmon.frames import Roi


class ActivityPhase(StrEnum):
    """What the scheduler does within its state. `ActivityStatus.phase` holds one of these.

    The phases of `auto` are `search`, `fast`, `survey_short`, `survey_long`, `solve_wait`, and
    `idle`. The other states have one phase each (`watch` for `safe`, then `align`, `commission`,
    and `paused`), and `align` has a second one, `rapid_focus`, while the person focuses with the
    fast stream. `camera_fault` replaces the phase while the scheduler waits for a recovery step
    of the camera, in whatever state it was.
    """

    SEARCH = "search"  # the fast stream looks for Polaris in short bursts
    FAST = "fast"  # the fast stream measures seeing, one analysis window after another
    SURVEY_SHORT = "survey_short"  # the short exposure of the survey step
    SURVEY_LONG = "survey_long"  # the long exposure of the survey step
    SOLVE_WAIT = "solve_wait"  # the survey step ran, and the pointing solution has not arrived
    IDLE = "idle"  # the camera rests until the next slot of the cycle
    WATCH = "watch"  # the brightness watch of `safe`
    ALIGN = "align"  # the alignment helper: the live view
    RAPID_FOCUS = "rapid_focus"  # the alignment helper: the fast stream on a ROI around Polaris
    COMMISSION = "commission"  # a burst, a sweep, a replay, or a dark session
    PAUSED = "paused"  # nothing runs
    CAMERA_FAULT = "camera_fault"  # a camera error: the scheduler waits to try a recovery step


@dataclass(frozen=True, slots=True)
class ActivityStatus:
    """What the scheduler does now, for how long, and what comes next.

    `state` is the state of the machine and `phase` is an `ActivityPhase` value. `label` names
    the activity in words, and `since_utc_ns` is when it began. `ends_utc_ns` is when the activity
    ends, when the scheduler knows: the end of the fast period, of a survey exposure, of the wait
    for a slot, or the idle timeout of the alignment (of the rapid focus mode, while it runs). It is
    an expectation. A fast period can end early, and a wait for a pointing solution ends when the
    solution arrives.

    `next_label` and `next_utc_ns` name the activity that follows and when it starts. `cadence_s`
    is the length of the cycle in force in `auto` (shorter under clouds) and `None` elsewhere.
    `detail` adds a sentence, such as the windows of the fast period that have closed. `reason`
    says why the state holds, in words. A value that the scheduler does not know is `None`.
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


@dataclass(frozen=True, slots=True)
class StreamInfo:
    """The stream that the camera runs or ran last.

    `purpose` is one of `fast`, `search`, `survey`, `watch`, `align`, `rapid_focus`, and
    `commission`. A `search` stream is one burst of the search for Polaris.
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
    solves_requested: int = 0  # survey steps forced because no pointing solution existed
    survey_steps: int = 0
    survey_frames: int = 0
    survey_results: int = 0
    survey_unsolved: int = 0  # results with an unsolved pointing record (a 1 ms frame has none)
    survey_skipped: int = 0
    survey_long_skips: int = 0  # steps whose long exposure the bright sky skipped
    survey_probes: int = 0  # steps whose first long frame measured the exposure of a second one
    watch_frames: int = 0  # brightness frames of the watch, in `safe` and for the gate in `auto`
    faults: int = 0
    recovery_steps: int = 0
    escalations: int = 0
    commands_accepted: int = 0
    commands_rejected: int = 0
    tasks_run: int = 0
    discarded_frames: int = 0  # frames of a sweep cell that a camera error cut short
    transitions: int = 0
    stalls: int = 0  # sleeps of the loop that returned much too late
    search_periods: int = 0  # periods of the cycle that searched to their end
    slack_periods: int = 0  # fast or search periods that filled the slack of a cycle
    search_bursts: int = 0  # bursts that looked for Polaris, probes included
    probe_bursts: int = 0  # bursts while the Sun was above the search limit
    search_frames: int = 0  # frames of the bursts, which reach no window
    detections: int = 0  # bursts that detected Polaris
    measure_starts: int = 0  # switches from search to measure


@dataclass(frozen=True, slots=True)
class SearchStatus:
    """Whether the fast stream of `auto` searches for Polaris or measures it.

    `mode` is `search` or `measure`. While the stream searches, `next_burst_utc_ns` is when the
    next burst may start (when the cycle reaches its fast slot, if that comes later), `probe` says
    that the Sun is above the search limit so that the next burst is a probe, `detections` counts
    the detecting bursts in a row, and `snr` is the median SNR of the star in the last burst.
    While it measures, `since_utc_ns` is when measure began.
    """

    mode: str
    next_burst_utc_ns: int | None = None
    probe: bool = False
    detections: int = 0
    snr: float | None = None
    since_utc_ns: int | None = None


@dataclass(frozen=True, slots=True)
class FaultStatus:
    """Where the scheduler stands in a fault episode.

    `cause` is the best explanation of the episode: `timeout` (no frame arrived), `disconnected`
    (the driver finds no camera), `link` (the scheduler cannot reach `acquire`), or `error`.
    `reason` says it in words, such as `no frame arrived; the camera may be disconnected`. Both
    are `None` without an episode. `since_utc_ns` is when the episode began.
    """

    failures: int = 0  # failures in a row
    good_frames: int = 0  # good frames since the last failure
    last_error: str | None = None
    next_attempt_utc_ns: int | None = None  # when the scheduler will try the recovery step
    next_step: str | None = None  # the name of that step, such as `reopen`
    cause: str | None = None
    reason: str | None = None
    since_utc_ns: int | None = None


@dataclass(frozen=True, slots=True)
class SchedulerStatus:
    """The state of the scheduler at one moment.

    `degraded` means that the camera failed repeatedly. The scheduler keeps retrying slowly, and
    the store and `web` stay up. `queued_tasks` counts the commissioning tasks that wait, and
    `survey_pending` counts the survey frames that await analysis. `activity` says what the
    scheduler does now and what comes next. `background_fraction` is the estimate that the
    daylight gate judges: the background that the fast stream would have at its shortest
    exposure, as a share of saturation, the larger of the last brightness frame and, in `auto`,
    the last burst or window scaled to that exposure. When the frame that decides clipped, the
    value is only a lower bound. `search` says whether the fast stream searches or measures, in
    `auto` with a pointing solution, and is `None` elsewhere. `twilight` says that the Sun is above
    the twilight limit, in twilight or in daylight: the flags of the windows and the survey results
    tell the two apart (`twilight` and `daylight`).
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
    activity: ActivityStatus | None = None
    search: SearchStatus | None = None

    @property
    def camera_component(self) -> str:
        """The state of the camera for the `components` field of a health record."""
        if self.degraded:
            return "failed"
        return "degraded" if self.fault.failures else "ok"

    @property
    def camera_reason(self) -> str | None:
        """Why the camera component is not `ok`, in words, or `None` while the camera works."""
        return self.fault.reason if self.camera_component != "ok" else None

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
