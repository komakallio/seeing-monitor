"""The records of the system itself: health snapshots, events, and runs.

The scheduler and store lanes own this module. Add a field at the end of a class, and keep the
rules in the module documentation of `seeingmon.records.base`.
"""

from __future__ import annotations

from typing import Any, ClassVar

from seeingmon.records.base import Record, quantity

HEALTH_STATES: dict[str, str] = {
    "safe": (
        "Daylight, a sky too bright for any mode, or a persistent fault keeps the camera idle."
    ),
    "auto": "The sky is dark enough and the system is healthy, so the scheduler runs its cycle.",
    "align": "The alignment helper runs and preempts everything else.",
    "commission": "A burst, a sweep, or a replay runs.",
    "paused": "You paused the scheduler, so nothing runs.",
}

HEALTH_FLAGS: dict[str, str] = {
    "time_invalid": "The clock is not synchronized, so `t_utc_ns` is not trustworthy.",
    "low_space": "The free space on the data partition is below the retention threshold.",
    "sink_backlog": "A sink has more unsent rows than the configured limit.",
}

EVENT_LEVELS: dict[str, str] = {
    "info": "A normal occurrence that is worth recording.",
    "warning": "Something unexpected that the system handled.",
    "error": "A failure that needs attention.",
}

# A dotted code: two or more lowercase words, such as `scheduler.state_change`.
EVENT_KIND_PATTERN = r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$"


class HealthRecord(Record):
    """A snapshot of the state of the system, every 60 seconds.

    `t_utc_ns` is the time of the snapshot. A measurement that the system cannot read is `None`.
    """

    record_type: ClassVar[str] = "health"

    state: str = quantity(
        codes=HEALTH_STATES, definition="The state of the scheduler, as documented codes."
    )
    degraded: bool = quantity(
        definition="Whether the system runs in the degraded state, with a component that failed."
    )
    components: dict[str, str] = quantity(
        definition=(
            "The state of each component, as a map from the component name to a state such as "
            "`ok`, `degraded`, or `failed`."
        ),
    )
    sensor_temperature_c: float | None = quantity(
        unit="degC",
        default=None,
        definition="The sensor temperature, in degrees Celsius.",
    )
    heater_duty: float | None = quantity(
        ge=0,
        le=1,
        default=None,
        definition="The duty cycle of the dew heater, from 0 (off) to 1 (always on).",
    )
    free_space_gb: float | None = quantity(
        unit="GB",
        ge=0,
        default=None,
        definition="The free space on the data partition, in gigabytes.",
    )
    data_used_gb: float | None = quantity(
        unit="GB",
        ge=0,
        default=None,
        definition="The space that the data directory uses, in gigabytes.",
    )
    dropped_total: int | None = quantity(
        ge=0,
        default=None,
        definition="The number of frames that the system has dropped since the run started.",
    )
    queue_depth: int | None = quantity(
        ge=0,
        default=None,
        definition="The number of frames that wait in the acquisition queue.",
    )
    sink_backlog: dict[str, int] = quantity(
        default_factory=dict,
        definition=(
            "The number of rows that each sink has not acknowledged, as a map from the sink name "
            "to a count."
        ),
    )
    time_synchronized: bool | None = quantity(
        default=None,
        definition="Whether the clock is synchronized, or `null` when the clock cannot tell.",
    )
    time_error_bound_ms: float | None = quantity(
        unit="ms",
        ge=0,
        default=None,
        definition="The bound on the absolute error of the clock, in milliseconds.",
    )
    uptime_s: float | None = quantity(
        unit="s",
        ge=0,
        default=None,
        definition="The time since the system started, in seconds.",
    )
    cpu_load_1m: float | None = quantity(
        ge=0, default=None, definition="The load average of the CPU over the last minute."
    )
    memory_used_mb: float | None = quantity(
        unit="MB",
        ge=0,
        default=None,
        definition="The memory that the system uses, in megabytes.",
    )
    dark_due: bool = quantity(
        definition=(
            "Whether the dark library needs a new set, because it misses the current temperature "
            "or is older than 6 months."
        ),
    )
    flags: list[str] = quantity(
        default_factory=list,
        codes=HEALTH_FLAGS,
        definition="The conditions that apply to the system, as documented codes.",
    )


class EventRecord(Record):
    """An occurrence that the operator or the analysis may need to see.

    `t_utc_ns` is the time of the occurrence. Two events can share a timestamp, so the writer
    gives the later event the next `revision`.
    """

    record_type: ClassVar[str] = "event"

    level: str = quantity(codes=EVENT_LEVELS, definition="The severity of the event.")
    kind: str = quantity(
        pattern=EVENT_KIND_PATTERN,
        definition="A dotted code that names the kind of event, such as `scheduler.state_change`.",
    )
    message: str = quantity(
        min_length=1, definition="A short description of the event in plain language."
    )
    detail: dict[str, Any] | None = quantity(
        default=None, definition="Structured data about the event, as a JSON object."
    )


class RunRecord(Record):
    """One start of the software.

    `t_utc_ns` is the time of the start. The record keeps the versions and the effective
    configuration, so a reader can reproduce the run. The configuration never holds secrets.
    """

    record_type: ClassVar[str] = "run"

    run_id: str = quantity(min_length=1, definition="The unique ID of the run.")
    software_version: str = quantity(
        min_length=1, definition="The version of the `seeingmon` package."
    )
    versions: dict[str, str] = quantity(
        definition=(
            "The versions of the components, as a map from the component name to a version "
            "string, such as the vendor SDK, the plate solver, and the star catalog."
        ),
    )
    effective_config: dict[str, Any] = quantity(
        definition=(
            "The merged configuration of the run, as a JSON object that the caller has cleaned "
            "of secrets."
        ),
    )
    profile: dict[str, Any] = quantity(
        definition="The hardware profile of the run, with its derived values, as a JSON object."
    )
