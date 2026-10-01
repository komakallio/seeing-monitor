"""The event type that the hardware components report."""

from __future__ import annotations

import logging

import pytest

from seeingmon.hardware.events import HardwareEvent, emit
from seeingmon.records.system import EventRecord


def event(**overrides: object) -> HardwareEvent:
    values: dict[str, object] = {
        "level": "warning",
        "kind": "power.cycle_done",
        "message": "The power cycle ran.",
        "t_utc_ns": 1_767_225_600_000_000_000,
    }
    values.update(overrides)
    return HardwareEvent(**values)  # type: ignore[arg-type]


def test_an_event_maps_onto_an_event_record() -> None:
    hardware = event(detail={"route": "command"})
    record = EventRecord(
        station_id="station-1",
        t_utc_ns=hardware.t_utc_ns,
        profile_id="profile-1",
        provenance={"algo": "hardware-1"},
        level=hardware.level,
        kind=hardware.kind,
        message=hardware.message,
        detail=dict(hardware.detail or {}),
    )
    assert record.kind == "power.cycle_done"


@pytest.mark.parametrize(
    "overrides",
    [{"level": "fatal"}, {"kind": "powercycle"}, {"kind": "Power.cycle"}, {"message": ""}],
)
def test_an_invalid_event_is_refused(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        event(**overrides)


def test_emit_delivers_the_event() -> None:
    received: list[HardwareEvent] = []
    emit(received.append, event())
    assert [item.kind for item in received] == ["power.cycle_done"]


def test_emit_without_a_callback_does_nothing() -> None:
    emit(None, event())


def test_a_failing_callback_never_disturbs_the_caller(caplog: pytest.LogCaptureFixture) -> None:
    def broken(_: HardwareEvent) -> None:
        raise RuntimeError("the store is down")

    with caplog.at_level(logging.ERROR):
        emit(broken, event())
    assert "power.cycle_done" in caplog.text
