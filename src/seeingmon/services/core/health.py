"""The `run` record and the `health` record of `core`.

**Run.** `build_run_record` describes one start of the software: the version of the package and of
the libraries (from `importlib.metadata`), the camera that the scheduler opened, the effective
configuration, and the profile with its derived values. The configuration passes through
`Config.effective`, which replaces every secret and every deployment value (host, URL, address,
command, path) with a marker, and `omit_site` leaves the site coordinates out, because the record
goes to remote sinks and to the API. The camera has no serial number in its info.

**Health.** `HealthReporter` assembles a `health` record every minute from the parts that know
their own state: the scheduler (state, camera, drops), the storage (free space, the backlog of each
sink), the driver in `acquire` (queue depth, thread state), the heater and the SQM-LE reader, the
clock (synchronization and error bound), the dark library (`dark_due`), and the machine (load and
memory, from `/proc` on Linux and unknown elsewhere). A value that no part can give is `None`, and
`quality` says why. A component state is `ok`, `degraded`, or `failed`, and `acquire` is `starting`
while `core` has not heard from it yet and the first seconds of the run last (the scheduler opens
the camera a moment after the start), so that a restart never shows a failure that is not one.
"""

from __future__ import annotations

import importlib.metadata
import logging
import platform
import secrets
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import seeingmon
from seeingmon.clock import NS_PER_S, Clock, utc_ns_to_datetime
from seeingmon.config import Config
from seeingmon.drivers.base import CameraError, CameraInfo
from seeingmon.profile import Profile
from seeingmon.profile.summary import profile_summary
from seeingmon.records import HealthRecord, RunRecord
from seeingmon.scheduler.status import SchedulerStatus
from seeingmon.services.core.system import SystemStats, read_system_stats

_log = logging.getLogger(__name__)

LIBRARIES = (
    "numpy",
    "scipy",
    "astropy",
    "pyerfa",
    "sep",
    "pillow",
    "pydantic",
    "fastapi",
    "uvicorn",
    "psycopg",
)


def library_versions(
    names: tuple[str, ...] = LIBRARIES,
    lookup: Callable[[str], str] = importlib.metadata.version,
) -> dict[str, str]:
    """The versions of the libraries that are installed, and of Python and the platform.

    A library that is not installed is left out. The platform entries name the operating system and
    the CPU family only, never the machine.
    """
    versions = {
        "python": platform.python_version(),
        "system": platform.system() or "unknown",
        "machine": platform.machine() or "unknown",
    }
    for name in names:
        try:
            versions[name] = lookup(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def camera_versions(info: CameraInfo | None) -> dict[str, str]:
    """The camera in the form of the `versions` field. `info` is `None` before the first open."""
    if info is None:
        return {}
    versions = {"camera_model": info.model, "camera_driver": info.driver}
    if info.sdk_version:
        versions["camera_sdk"] = info.sdk_version
    return versions


def new_run_id(clock: Clock) -> str:
    """A unique ID for this start: the UTC time to the second and three random bytes."""
    stamp = utc_ns_to_datetime(clock.utc_ns()).strftime("%Y%m%dT%H%M%SZ")
    return f"run-{stamp}-{secrets.token_hex(3)}"


def build_run_record(
    config: Config,
    profile: Profile,
    clock: Clock,
    *,
    camera: CameraInfo | None = None,
    extra_versions: Mapping[str, str] | None = None,
    run_id: str | None = None,
) -> RunRecord:
    """The `run` record of this start. The configuration is redacted and has no site."""
    versions = {**library_versions(), **camera_versions(camera), **(extra_versions or {})}
    return RunRecord(
        station_id=config.station_id,
        t_utc_ns=clock.utc_ns(),
        profile_id=profile.id,
        provenance={"software": seeingmon.__version__},
        run_id=run_id or new_run_id(clock),
        software_version=seeingmon.__version__,
        versions=versions,
        effective_config=config.effective(omit_site=True),
        profile=profile_summary(profile),
    )


# --- Health ------------------------------------------------------------------------------------


class SchedulerView(Protocol):
    def status(self) -> SchedulerStatus: ...


class StorageView(Protocol):
    def health_fields(self) -> dict[str, Any]: ...


class AcquireView(Protocol):
    def health(self) -> dict[str, Any]: ...


class HeaterStatusView(Protocol):
    @property
    def state(self) -> str: ...


class HeaterView(Protocol):
    def status(self) -> HeaterStatusView: ...

    def recent_duty(self, window_s: float) -> float | None: ...


class SqmView(Protocol):
    @property
    def failures(self) -> int: ...


def acquire_component(health: Mapping[str, Any] | None) -> str:
    """The state of `acquire` for the `components` field, from its health summary."""
    if health is None:
        return "failed"
    if health.get("threads_alive") is False:
        return "failed"
    return "degraded" if health.get("state") in ("stalled", "stopping") else "ok"


def heater_component(state: str) -> str | None:
    """The state of the heater, or `None` for a heater that is not configured."""
    if state == "disabled":
        return None
    return "failed" if state == "fault" else "ok"


def sqm_component(failures: int) -> str:
    """The state of the SQM-LE reader from the polls that failed in a row.

    A failure of the reader is never `failed`. The unit, or the program that feeds InfluxDB with its
    readings, is not part of the station, and a power cycle of the Pi cannot repair it. A `failed`
    component makes `/api/v1/health` answer 503, which an external watchdog may turn into a power
    cycle, so a lost reference instrument must stop at `degraded`.
    """
    return "ok" if failures == 0 else "degraded"


class HealthReporter:
    """Builds the `health` record from the parts of `core`. Call `build` from one thread."""

    def __init__(
        self,
        *,
        clock: Clock,
        station_id: str,
        profile_id: str,
        scheduler: SchedulerView,
        storage: StorageView,
        interval_s: float = 60.0,
        acquire: AcquireView | None = None,
        heater: HeaterView | None = None,
        sqm: SqmView | None = None,
        dark_due: Callable[[float | None, int], bool] | None = None,
        stats: Callable[[], SystemStats] = read_system_stats,
        web_connected: Callable[[], bool] | None = None,
        startup_grace_s: float = 30.0,
    ) -> None:
        self._clock = clock
        self._station_id = station_id
        self._profile_id = profile_id
        self._scheduler = scheduler
        self._storage = storage
        self._interval_s = interval_s
        self._acquire = acquire
        self._heater = heater
        self._sqm = sqm
        self._dark_due = dark_due
        self._stats = stats
        self._web_connected = web_connected
        self._started_ns = clock.monotonic_ns()
        self._startup_grace_s = startup_grace_s
        self._acquire_seen = False  # whether `acquire` has answered since the start

    def _acquire_health(self) -> dict[str, Any] | None:
        if self._acquire is None:
            return None
        try:
            return dict(self._acquire.health())
        except CameraError:
            _log.debug("acquire gave no health summary", exc_info=True)
            return None

    def build(self) -> HealthRecord:
        """Assemble the record for this moment."""
        now_ns = self._clock.utc_ns()
        status = self._scheduler.status()
        fields = status.health_fields()
        quality: dict[str, str] = {}

        components: dict[str, str] = {"core": "ok", **fields["components"]}
        acquire = self._acquire_health()
        uptime_s = max(0.0, (self._clock.monotonic_ns() - self._started_ns) / NS_PER_S)
        queue_depth: int | None = None
        if self._acquire is not None:
            self._acquire_seen = self._acquire_seen or acquire is not None
            # The scheduler opens the camera a moment after the start, and until then `acquire` has
            # not answered. That is a start, not a failure, for as long as the grace lasts.
            starting = (
                acquire is None and not self._acquire_seen and uptime_s < self._startup_grace_s
            )
            components["acquire"] = "starting" if starting else acquire_component(acquire)
            if acquire is None:
                quality["queue_depth"] = "acquire did not answer"
            else:
                queue_depth = int(acquire.get("queue_frames", 0))

        heater_duty: float | None = None
        if self._heater is not None:
            heater = heater_component(self._heater.status().state)
            if heater is not None:
                components["heater"] = heater
                heater_duty = self._heater.recent_duty(self._interval_s)
                if heater_duty is None:
                    quality["heater_duty"] = "the heater log does not reach back a whole interval"
            else:
                quality["heater_duty"] = "the heater is not configured"
        else:
            quality["heater_duty"] = "the heater is not configured"
        if self._sqm is not None:
            components["sqm"] = sqm_component(self._sqm.failures)
        if self._web_connected is not None and self._web_connected():
            components["web"] = "ok"

        storage = self._storage.health_fields()
        flags = list(storage["flags"])
        clock_status = self._clock.status()
        if clock_status.synchronized is False:
            flags.append("time_invalid")
        if clock_status.synchronized is None:
            quality["time_synchronized"] = "the clock cannot tell"
        error_ms = (
            None if clock_status.error_bound_ns is None else clock_status.error_bound_ns / 1e6
        )
        if error_ms is None:
            quality["time_error_bound_ms"] = "the clock gives no error bound"

        if storage["data_used_gb"] is None:
            quality["data_used_gb"] = "retention has not measured it yet"
        if fields["sensor_temperature_c"] is None:
            quality["sensor_temperature_c"] = "the camera gave no reading yet"
        machine = self._stats()
        if machine.load_1m is None:
            quality["cpu_load_1m"] = "this platform does not report it"
        if machine.memory_used_mb is None:
            quality["memory_used_mb"] = "this platform does not report it"

        dark_due = (
            False
            if self._dark_due is None
            else self._dark_due(fields["sensor_temperature_c"], now_ns)
        )
        return HealthRecord(
            station_id=self._station_id,
            t_utc_ns=now_ns,
            profile_id=self._profile_id,
            provenance={"software": seeingmon.__version__},
            quality=quality or None,
            state=fields["state"],
            degraded=bool(fields["degraded"]) or "failed" in components.values(),
            components=components,
            sensor_temperature_c=fields["sensor_temperature_c"],
            heater_duty=heater_duty,
            free_space_gb=storage["free_space_gb"],
            data_used_gb=storage["data_used_gb"],
            dropped_total=fields["dropped_total"],
            queue_depth=queue_depth,
            sink_backlog=storage["sink_backlog"],
            time_synchronized=clock_status.synchronized,
            time_error_bound_ms=error_ms,
            uptime_s=uptime_s,
            cpu_load_1m=machine.load_1m,
            memory_used_mb=machine.memory_used_mb,
            dark_due=dark_due,
            flags=flags,
        )
