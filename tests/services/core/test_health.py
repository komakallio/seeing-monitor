"""The `run` record and the `health` record of `core`."""

from __future__ import annotations

import importlib.metadata
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from seeingmon.clock import ClockStatus, VirtualClock
from seeingmon.config import REDACTED, load_config
from seeingmon.drivers.base import CameraDisconnectedError, CameraInfo
from seeingmon.records import HealthRecord
from seeingmon.scheduler.status import Counters, FaultStatus, SchedulerStatus
from seeingmon.services.core.health import (
    HealthReporter,
    acquire_component,
    build_run_record,
    camera_versions,
    heater_component,
    library_versions,
    new_run_id,
    sqm_component,
)
from seeingmon.services.core.system import SystemStats

START = 1_800_000_000_000_000_000


def status(**changes: Any) -> SchedulerStatus:
    fields: dict[str, Any] = {
        "t_utc_ns": START,
        "state": "auto",
        "state_reason": "dark",
        "state_since_utc_ns": START,
        "last_transition_utc_ns": None,
        "degraded": False,
        "stream": None,
        "cloud": False,
        "cloud_fraction": None,
        "twilight": False,
        "sun_elevation_deg": -30.0,
        "background_fraction": None,
        "sensor_temperature_c": 14.5,
        "counters": Counters(dropped=42),
        "fault": FaultStatus(),
        "queued_tasks": 0,
        "survey_pending": 0,
    }
    fields.update(changes)
    return SchedulerStatus(**fields)


class FakeScheduler:
    def __init__(self, **changes: Any) -> None:
        self.changes = changes

    def status(self) -> SchedulerStatus:
        return status(**self.changes)


class FakeStorage:
    def __init__(self, **changes: Any) -> None:
        self.fields: dict[str, Any] = {
            "free_space_gb": 20.0,
            "data_used_gb": 1.5,
            "sink_backlog": {"influx": 12},
            "flags": [],
        }
        self.fields.update(changes)

    def health_fields(self) -> dict[str, Any]:
        return dict(self.fields)


class FakeAcquire:
    def __init__(self, summary: dict[str, Any] | None) -> None:
        self.summary = summary

    def health(self) -> dict[str, Any]:
        if self.summary is None:
            raise CameraDisconnectedError("no answer")
        return dict(self.summary)


@dataclass
class FakeHeaterStatus:
    state: str


class FakeHeater:
    def __init__(self, state: str = "off", duty: float | None = 0.25) -> None:
        self._state, self._duty = state, duty
        self.windows: list[float] = []

    def status(self) -> FakeHeaterStatus:
        return FakeHeaterStatus(self._state)

    def recent_duty(self, window_s: float) -> float | None:
        self.windows.append(window_s)
        return self._duty


@dataclass
class FakeSqm:
    failures: int = 0


def quality_of(record: HealthRecord) -> dict[str, str]:
    return record.quality or {}


GOOD_ACQUIRE = {"state": "streaming", "threads_alive": True, "queue_frames": 7}
STATS = SystemStats(load_1m=0.5, memory_used_mb=900.0)


def reporter(**parts: Any) -> HealthReporter:
    defaults: dict[str, Any] = {
        "clock": VirtualClock(START),
        "station_id": "test",
        "profile_id": "asi294mm-gs250",
        "scheduler": FakeScheduler(),
        "storage": FakeStorage(),
        "stats": lambda: STATS,
    }
    defaults.update(parts)
    return HealthReporter(**defaults)


class TestComponents:
    @pytest.mark.parametrize(
        ("summary", "expected"),
        [
            (None, "failed"),
            ({"state": "streaming", "threads_alive": True}, "ok"),
            ({"state": "ready", "threads_alive": True}, "ok"),
            ({"state": "stalled", "threads_alive": True}, "degraded"),
            ({"state": "stopping", "threads_alive": True}, "degraded"),
            ({"state": "streaming", "threads_alive": False}, "failed"),
        ],
    )
    def test_acquire(self, summary: dict[str, Any] | None, expected: str) -> None:
        assert acquire_component(summary) == expected

    @pytest.mark.parametrize(
        ("state", "expected"),
        [("disabled", None), ("fault", "failed"), ("off", "ok"), ("heating", "ok")],
    )
    def test_heater(self, state: str, expected: str | None) -> None:
        assert heater_component(state) == expected

    @pytest.mark.parametrize(
        ("failures", "expected"), [(0, "ok"), (1, "degraded"), (5, "degraded"), (500, "degraded")]
    )
    def test_sqm_never_reads_failed(self, failures: int, expected: str) -> None:
        # A lost reference instrument must not turn the health answer into a 503, which an external
        # watchdog could answer with a power cycle of the Pi.
        assert sqm_component(failures) == expected


class TestHealthRecord:
    def test_a_healthy_system_fills_every_field(self) -> None:
        record = reporter(
            acquire=FakeAcquire(GOOD_ACQUIRE),
            heater=FakeHeater("off", 0.25),
            sqm=FakeSqm(0),
            dark_due=lambda temperature, now: temperature == 14.5,
            web_connected=lambda: True,
        ).build()
        assert isinstance(record, HealthRecord)
        assert (record.state, record.degraded) == ("auto", False)
        assert record.components == {
            "core": "ok",
            "scheduler": "ok",
            "camera": "ok",
            "acquire": "ok",
            "heater": "ok",
            "sqm": "ok",
            "web": "ok",
        }
        assert record.sensor_temperature_c == 14.5
        assert record.heater_duty == 0.25
        assert (record.free_space_gb, record.data_used_gb) == (20.0, 1.5)
        assert (record.dropped_total, record.queue_depth) == (42, 7)
        assert record.sink_backlog == {"influx": 12}
        assert record.time_synchronized is True
        assert record.time_error_bound_ms == 0.0
        assert record.uptime_s == 0.0
        assert (record.cpu_load_1m, record.memory_used_mb) == (0.5, 900.0)
        assert record.dark_due is True
        assert record.flags == []
        assert record.quality is None

    def test_uptime_follows_the_monotonic_clock(self) -> None:
        clock = VirtualClock(START)
        health = reporter(clock=clock)
        clock.advance(90.0)
        assert health.build().uptime_s == 90.0

    def test_an_acquire_that_does_not_answer_is_a_failed_component(self) -> None:
        record = reporter(acquire=FakeAcquire(None), startup_grace_s=0.0).build()
        assert record.components["acquire"] == "failed"
        assert record.degraded is True
        assert record.queue_depth is None
        assert quality_of(record)["queue_depth"]

    def test_an_acquire_that_has_not_answered_yet_is_starting_and_not_a_failure(self) -> None:
        clock = VirtualClock(START)
        health = reporter(clock=clock, acquire=FakeAcquire(None), startup_grace_s=30.0)
        record = health.build()
        assert record.components["acquire"] == "starting"
        assert record.degraded is False
        assert quality_of(record)["queue_depth"]  # the missing value still says why
        clock.advance(31.0)  # the grace is over, and acquire is still silent
        assert health.build().components["acquire"] == "failed"

    def test_an_acquire_that_answered_and_then_goes_silent_fails_at_once(self) -> None:
        acquire = FakeAcquire(GOOD_ACQUIRE)
        health = reporter(acquire=acquire, startup_grace_s=30.0)
        assert health.build().components["acquire"] == "ok"
        acquire.summary = None  # the process died a second after the start
        assert health.build().components["acquire"] == "failed"

    def test_the_degraded_scheduler_makes_the_system_degraded(self) -> None:
        record = reporter(scheduler=FakeScheduler(degraded=True, state="safe")).build()
        assert (record.state, record.degraded) == ("safe", True)
        assert record.components["camera"] == "failed"

    def test_the_flags_of_the_storage_pass_through_and_time_invalid_joins_them(self) -> None:
        clock = VirtualClock(START, status=ClockStatus(False, None, "test"))
        record = reporter(
            clock=clock, storage=FakeStorage(flags=["low_space", "sink_backlog"])
        ).build()
        assert record.flags == ["low_space", "sink_backlog", "time_invalid"]
        assert record.time_synchronized is False
        assert record.time_error_bound_ms is None

    def test_a_clock_that_cannot_tell_gets_a_quality_note_and_no_flag(self) -> None:
        clock = VirtualClock(START, status=ClockStatus(None, None, "system"))
        record = reporter(clock=clock).build()
        assert "time_invalid" not in record.flags
        assert record.time_synchronized is None
        assert {"time_synchronized", "time_error_bound_ms"} <= set(quality_of(record))

    def test_a_machine_that_reports_nothing_leaves_load_and_memory_unknown(self) -> None:
        record = reporter(stats=lambda: SystemStats(None, None)).build()
        assert (record.cpu_load_1m, record.memory_used_mb) == (None, None)
        assert {"cpu_load_1m", "memory_used_mb"} <= set(quality_of(record))

    def test_a_missing_heater_is_said_and_a_heater_fault_is_a_failure(self) -> None:
        none = reporter().build()
        assert none.heater_duty is None
        assert "heater" in quality_of(none)["heater_duty"]
        failing = reporter(heater=FakeHeater("fault", 0.0)).build()
        assert failing.components["heater"] == "failed"
        assert failing.degraded is True

    def test_the_heater_duty_covers_the_health_interval(self) -> None:
        heater = FakeHeater("heating", 0.4)
        reporter(heater=heater, interval_s=60.0).build()
        assert heater.windows == [60.0]

    def test_a_heater_log_that_does_not_reach_back_leaves_the_duty_unknown(self) -> None:
        record = reporter(heater=FakeHeater("off", None)).build()
        assert record.heater_duty is None
        assert "heater_duty" in quality_of(record)

    def test_a_storage_that_has_not_measured_its_use_says_so(self) -> None:
        record = reporter(storage=FakeStorage(data_used_gb=None)).build()
        assert record.data_used_gb is None
        assert "data_used_gb" in quality_of(record)

    def test_the_dark_library_is_not_due_when_no_probe_is_given(self) -> None:
        assert reporter().build().dark_due is False


class TestRunRecord:
    @pytest.fixture
    def config(self, tmp_path: Path) -> Any:
        local = tmp_path / "local.toml"
        local.write_text(
            'station_id = "bench"\n'
            "[site]\nlatitude_deg = 12.5\nlongitude_deg = 34.5\nelevation_m = 7.0\n"
            '[paths]\ndata_dir = "somewhere"\n'
            '[services]\nconnection_key = "a-secret-key-of-more-than-32-characters"\n'
        )
        return load_config(local_file=local, env={})

    def test_the_record_describes_the_start_without_secrets_or_the_site(self, config: Any) -> None:
        clock = VirtualClock(START)
        info = CameraInfo("Test camera", "sim", "1.2.3", 8288, 5644)
        record = build_run_record(config, config.profile, clock, camera=info, run_id="run-test")
        assert record.run_id == "run-test"
        assert record.station_id == "bench"
        assert record.t_utc_ns == START
        assert record.profile_id == config.profile.id
        assert record.software_version
        assert record.versions["camera_model"] == "Test camera"
        assert record.versions["camera_sdk"] == "1.2.3"
        assert record.versions["python"].count(".") >= 1
        text = str(record.effective_config)
        assert "a-secret-key" not in text
        assert record.effective_config["services"]["connection_key"] == REDACTED
        assert record.effective_config["paths"] == REDACTED
        assert "site" not in record.effective_config
        assert "12.5" not in text
        assert "optics" in record.profile  # the summary of the profile

    def test_a_start_without_a_camera_still_has_the_libraries(self, config: Any) -> None:
        record = build_run_record(config, config.profile, VirtualClock(START))
        assert "camera_model" not in record.versions
        assert "numpy" in record.versions
        assert re.fullmatch(r"run-\d{8}T\d{6}Z-[0-9a-f]{6}", record.run_id)

    def test_the_run_ids_of_two_starts_differ(self) -> None:
        clock = VirtualClock(START)
        assert new_run_id(clock) != new_run_id(clock)


class TestVersions:
    def test_a_missing_library_is_left_out(self) -> None:
        def lookup(name: str) -> str:
            if name == "scipy":
                return "9.9"
            raise importlib.metadata.PackageNotFoundError(name)

        versions = library_versions(("scipy", "nothing"), lookup)
        assert versions["scipy"] == "9.9"
        assert "nothing" not in versions
        assert {"python", "system", "machine"} <= set(versions)

    def test_a_camera_without_an_sdk_has_no_sdk_entry(self) -> None:
        info = CameraInfo("Test camera", "fake", None, 100, 100)
        assert camera_versions(info) == {"camera_model": "Test camera", "camera_driver": "fake"}
        assert camera_versions(None) == {}
