"""The status snapshot and its mapping to the health record."""

from __future__ import annotations

import dataclasses
import json

import pytest

from seeingmon.frames import Roi
from seeingmon.records import HealthRecord
from seeingmon.scheduler.status import (
    ActivityPhase,
    ActivityStatus,
    Counters,
    FaultStatus,
    SchedulerStatus,
    StreamInfo,
)


def make_status(
    *,
    degraded: bool = False,
    failures: int = 0,
    state: str = "auto",
    dropped: int = 7,
    activity: ActivityStatus | None = None,
) -> SchedulerStatus:
    return SchedulerStatus(
        t_utc_ns=1_000,
        state=state,
        state_reason="the sky is dark enough",
        state_since_utc_ns=500,
        last_transition_utc_ns=500,
        degraded=degraded,
        stream=StreamInfo(3, "fast", "bin1", 2000, 0, Roi(10, 20, 128, 128)),
        cloud=False,
        cloud_fraction=0.1,
        twilight=True,
        sun_elevation_deg=-12.5,
        background_fraction=0.01,
        sensor_temperature_c=18.5,
        counters=Counters(dropped=dropped),
        fault=FaultStatus(failures=failures),
        queued_tasks=2,
        survey_pending=1,
        activity=activity,
    )


def test_the_status_converts_to_json() -> None:
    """`/status` serves `dataclasses.asdict`, so every value must be plain."""
    text = json.dumps(dataclasses.asdict(make_status()))
    data = json.loads(text)
    assert data["state"] == "auto"
    assert data["stream"]["roi"] == {"x": 10, "y": 20, "width": 128, "height": 128}
    assert data["counters"]["dropped"] == 7


def test_the_status_is_frozen_and_holds_its_own_counters() -> None:
    status = make_status()
    with pytest.raises(dataclasses.FrozenInstanceError):
        status.state = "safe"  # type: ignore[misc]
    counters = Counters()
    copy = dataclasses.replace(counters)
    counters.frames += 5
    assert copy.frames == 0  # a snapshot does not follow the live counters


@pytest.mark.parametrize(
    ("degraded", "failures", "camera"),
    [(False, 0, "ok"), (False, 2, "degraded"), (True, 5, "failed")],
)
def test_the_camera_component_follows_the_fault_state(
    degraded: bool, failures: int, camera: str
) -> None:
    status = make_status(degraded=degraded, failures=failures)
    assert status.camera_component == camera
    assert status.health_fields()["components"] == {
        "scheduler": "degraded" if degraded else "ok",
        "camera": camera,
    }


@pytest.mark.parametrize("state", ["safe", "auto", "align", "commission", "paused"])
def test_the_health_fields_build_a_valid_health_record(state: str) -> None:
    fields = make_status(state=state).health_fields()
    record = HealthRecord(
        station_id="test",
        t_utc_ns=1_000,
        profile_id="test",
        provenance={"scheduler": "test"},
        dark_due=False,
        **fields,
    )
    assert record.state == state
    assert record.dropped_total == 7
    assert record.sensor_temperature_c == 18.5
    assert record.components["camera"] == "ok"


def test_a_status_has_no_activity_until_the_scheduler_gives_one() -> None:
    assert make_status().activity is None
    assert dataclasses.asdict(make_status())["activity"] is None


def test_the_activity_converts_to_json_with_plain_values() -> None:
    activity = ActivityStatus(
        state="auto",
        phase=ActivityPhase.FAST.value,
        label="Fast stream: seeing windows",
        since_utc_ns=1_000,
        ends_utc_ns=2_000,
        next_label="Survey step: a 1 ms and a 30 s frame",
        next_utc_ns=2_000,
        cadence_s=180.0,
        detail="Windows of 20 s: 4 of 7 closed",
        reason="the sky is dark enough",
    )
    data = json.loads(json.dumps(dataclasses.asdict(make_status(activity=activity))))
    assert data["activity"] == {
        "state": "auto",
        "phase": "fast",
        "label": "Fast stream: seeing windows",
        "since_utc_ns": 1_000,
        "ends_utc_ns": 2_000,
        "next_label": "Survey step: a 1 ms and a 30 s frame",
        "next_utc_ns": 2_000,
        "cadence_s": 180.0,
        "detail": "Windows of 20 s: 4 of 7 closed",
        "reason": "the sky is dark enough",
    }


def test_the_activity_has_a_phase_for_every_state_of_the_scheduler_and_the_fault() -> None:
    assert {phase.value for phase in ActivityPhase} == {
        "fast",
        "survey_short",
        "survey_long",
        "solve_wait",
        "idle",
        "watch",
        "align",
        "commission",
        "paused",
        "camera_fault",
    }


def test_the_activity_is_frozen() -> None:
    activity = ActivityStatus(state="safe", phase="watch", label="x", since_utc_ns=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        activity.label = "y"  # type: ignore[misc]
