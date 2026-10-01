from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from seeingmon.records.system import (
    EVENT_LEVELS,
    HEALTH_FLAGS,
    HEALTH_STATES,
    EventRecord,
    HealthRecord,
    RunRecord,
)

BASE: dict[str, Any] = {
    "station_id": "station-a",
    "t_utc_ns": 1_800_000_000_000_000_000,
    "profile_id": "profile-a",
    "provenance": {"software": "0.1.0"},
}


class TestHealthRecord:
    def make(self, **overrides: Any) -> HealthRecord:
        values: dict[str, Any] = {
            "state": "auto",
            "degraded": False,
            "components": {"acquire": "ok", "core": "ok", "web": "ok"},
            "dark_due": False,
        }
        values.update(overrides)
        return HealthRecord(**BASE, **values)

    def test_the_states_are_the_scheduler_states(self) -> None:
        assert set(HEALTH_STATES) == {"safe", "auto", "align", "commission", "paused"}
        for state in HEALTH_STATES:
            assert self.make(state=state).state == state
        with pytest.raises(ValidationError, match="unknown code"):
            self.make(state="running")

    def test_a_snapshot_with_every_measurement(self) -> None:
        record = self.make(
            sensor_temperature_c=-4.5,
            heater_duty=0.25,
            free_space_gb=21.5,
            data_used_gb=3.2,
            dropped_total=12,
            queue_depth=0,
            sink_backlog={"influx": 0, "timescale": 140},
            time_synchronized=True,
            time_error_bound_ms=2.5,
            uptime_s=86_400.0,
            cpu_load_1m=0.6,
            memory_used_mb=900.0,
            flags=["sink_backlog"],
        )
        assert record.sink_backlog["timescale"] == 140
        assert set(HEALTH_FLAGS) == {"time_invalid", "low_space", "sink_backlog"}

    def test_the_clock_state_can_be_unknown(self) -> None:
        assert self.make().time_synchronized is None
        assert self.make(time_synchronized=False).time_synchronized is False

    @pytest.mark.parametrize(
        "overrides",
        [
            {"heater_duty": 1.2},
            {"free_space_gb": -1.0},
            {"queue_depth": -1},
            {"sink_backlog": {"influx": 1.5}},
            {"components": {"web": 1}},
            {"degraded": "no"},
        ],
    )
    def test_the_values_must_be_valid(self, overrides: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            self.make(**overrides)


class TestEventRecord:
    def make(self, **overrides: Any) -> EventRecord:
        values: dict[str, Any] = {
            "level": "info",
            "kind": "scheduler.state_change",
            "message": "The scheduler entered the auto state.",
        }
        values.update(overrides)
        return EventRecord(**BASE, **values)

    def test_the_levels(self) -> None:
        assert set(EVENT_LEVELS) == {"info", "warning", "error"}
        with pytest.raises(ValidationError, match="unknown code"):
            self.make(level="critical")

    def test_the_detail_is_any_json_object(self) -> None:
        record = self.make(
            detail={"from": "safe", "to": "auto", "counts": [1, 2], "nested": {"a": None}}
        )
        assert record.detail is not None
        assert self.make().detail is None
        with pytest.raises(ValidationError, match="JSON"):
            self.make(detail={"when": object()})

    @pytest.mark.parametrize(
        "kind", ["scheduler.state_change", "store.retention.deleted", "a.b", "sink.influx_down"]
    )
    def test_a_kind_is_a_dotted_code(self, kind: str) -> None:
        assert self.make(kind=kind).kind == kind

    @pytest.mark.parametrize(
        "kind",
        ["scheduler", "Scheduler.state", "scheduler.", ".state", "a..b", "a b.c", "1a.b", ""],
    )
    def test_other_kinds_are_rejected(self, kind: str) -> None:
        with pytest.raises(ValidationError):
            self.make(kind=kind)

    def test_two_events_at_one_time_differ_by_revision(self) -> None:
        first, second = self.make(), self.make(revision=1)
        assert first.record_key != second.record_key
        assert first.record_key[:3] == second.record_key[:3]


class TestRunRecord:
    def test_a_run_keeps_its_versions_and_configuration(self) -> None:
        record = RunRecord(
            **BASE,
            run_id="run-0001",
            software_version="0.1.0",
            versions={"python": "3.13", "solver": "fake"},
            effective_config={"scheduler": {"window_s": 60}, "sinks": []},
            profile={"name": "asi294mm-gs250", "modes": {"bin1": {"pixel_um": 2.315}}},
        )
        assert RunRecord.from_row(record.to_row()) == record

    @pytest.mark.parametrize("field", ["run_id", "software_version"])
    def test_the_identifiers_are_not_empty(self, field: str) -> None:
        values: dict[str, Any] = {
            "run_id": "run-0001",
            "software_version": "0.1.0",
            "versions": {},
            "effective_config": {},
            "profile": {},
        }
        values[field] = ""
        with pytest.raises(ValidationError):
            RunRecord(**BASE, **values)
