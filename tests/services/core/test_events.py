"""`event` records from the hardware, and the collection of the events of `acquire`."""

from __future__ import annotations

from typing import Any

import pytest

from seeingmon.clock import VirtualClock
from seeingmon.drivers.base import CameraDisconnectedError
from seeingmon.hardware.events import HardwareEvent
from seeingmon.records import EventRecord
from seeingmon.services.acquire.events import EventBatch, LoggedEvent
from seeingmon.services.core.events import EventPump, EventWriter
from seeingmon.testing import ListRecordWriter


def hardware_event(number: int, level: str = "info") -> HardwareEvent:
    return HardwareEvent(level, "camera.recovery", f"step {number}", 1000 + number, {"n": number})


def writer_for(sink: ListRecordWriter) -> EventWriter:
    return EventWriter(
        sink.write,
        station_id="test",
        profile_id="profile-1",
        clock=VirtualClock(5_000),
        source="acquire",
    )


def events_of(sink: ListRecordWriter) -> list[EventRecord]:
    return [r for r in sink.of_type("event") if isinstance(r, EventRecord)]


class TestWriter:
    def test_a_hardware_event_keeps_its_own_time(self) -> None:
        sink = ListRecordWriter()
        writer_for(sink)(hardware_event(7, "warning"))
        (record,) = events_of(sink)
        assert (record.level, record.kind, record.message) == (
            "warning",
            "camera.recovery",
            "step 7",
        )
        assert record.t_utc_ns == 1007
        assert record.detail == {"n": 7}
        assert record.station_id == "test"
        assert record.provenance["source"] == "acquire"

    def test_an_event_of_core_takes_the_time_of_the_clock(self) -> None:
        sink = ListRecordWriter()
        writer_for(sink).emit("info", "core.started", "Core started.", {"pid_free": True})
        (record,) = events_of(sink)
        assert record.t_utc_ns == 5_000
        assert record.detail == {"pid_free": True}

    def test_a_failing_store_never_raises_into_the_caller(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        def broken(record: Any) -> None:
            raise OSError("the disk is full")

        writer = EventWriter(
            broken, station_id="test", profile_id="p", clock=VirtualClock(0), source="local"
        )
        writer(hardware_event(1))
        writer.emit("info", "core.started", "x")
        assert (writer.written, writer.failed) == (0, 2)
        assert "could not store" in caplog.text

    def test_a_bad_kind_is_dropped_and_counted(self, caplog: pytest.LogCaptureFixture) -> None:
        sink = ListRecordWriter()
        writer = writer_for(sink)
        writer.emit("info", "nodots", "x")
        assert writer.failed == 1
        assert events_of(sink) == []


class FakeSource:
    """The part of `RemoteCameraDriver` that the pump uses, with a log per process."""

    def __init__(self) -> None:
        self.instance: str | None = "one"
        self.log: list[HardwareEvent] = []
        self.lost = 0
        self.fail = False
        self.calls: list[int] = []

    def events(self, after: int = 0) -> EventBatch:
        self.calls.append(after)
        if self.fail:
            raise CameraDisconnectedError("acquire went away")
        items = tuple(LoggedEvent(i + 1, e) for i, e in enumerate(self.log) if i + 1 > after)
        return EventBatch(items, len(self.log), self.lost)

    def restart(self, instance: str) -> None:
        self.instance = instance
        self.log = []


class TestPump:
    def test_new_events_are_written_once(self) -> None:
        sink, source = ListRecordWriter(), FakeSource()
        pump = EventPump(source, writer_for(sink))
        source.log = [hardware_event(1), hardware_event(2)]
        assert pump.poll() == 2
        assert pump.poll() == 0
        source.log.append(hardware_event(3))
        assert pump.poll() == 1
        assert [e.message for e in events_of(sink)] == ["step 1", "step 2", "step 3"]
        assert (pump.after, pump.collected) == (3, 3)

    def test_no_session_means_no_call(self) -> None:
        sink, source = ListRecordWriter(), FakeSource()
        source.instance = None
        assert EventPump(source, writer_for(sink)).poll() == 0
        assert source.calls == []

    def test_an_acquire_that_does_not_answer_gives_nothing_and_tries_again(self) -> None:
        sink, source = ListRecordWriter(), FakeSource()
        pump = EventPump(source, writer_for(sink))
        source.log = [hardware_event(1)]
        source.fail = True
        assert pump.poll() == 0
        source.fail = False
        assert pump.poll() == 1

    def test_a_new_process_starts_the_numbers_again_and_says_so(self) -> None:
        sink, source = ListRecordWriter(), FakeSource()
        pump = EventPump(source, writer_for(sink))
        source.log = [hardware_event(i) for i in range(1, 6)]
        pump.poll()
        source.restart("two")
        source.log = [hardware_event(11), hardware_event(12)]  # fewer than the 5 that we saw
        assert pump.poll() == 2
        kinds = [e.kind for e in events_of(sink)]
        assert kinds.count("acquire.restarted") == 1
        assert [e.message for e in events_of(sink)][-2:] == ["step 11", "step 12"]

    def test_a_new_process_is_found_even_when_its_log_is_longer_than_ours(self) -> None:
        sink, source = ListRecordWriter(), FakeSource()
        pump = EventPump(source, writer_for(sink))
        source.log = [hardware_event(1), hardware_event(2)]
        pump.poll()
        source.restart("two")
        source.log = [hardware_event(i) for i in range(21, 26)]
        assert pump.poll() == 5  # all five, not only the ones after number 2
        assert source.calls[-1] == 0

    def test_a_log_that_restarted_without_a_new_instance_is_read_from_the_start(self) -> None:
        sink, source = ListRecordWriter(), FakeSource()
        pump = EventPump(source, writer_for(sink))
        source.log = [hardware_event(i) for i in range(1, 6)]
        pump.poll()
        source.log = [hardware_event(31)]  # the same instance name, and a shorter log
        assert pump.poll() == 1

    def test_lost_events_are_reported(self) -> None:
        sink, source = ListRecordWriter(), FakeSource()
        pump = EventPump(source, writer_for(sink))
        source.log = [hardware_event(1)]
        source.lost = 4
        pump.poll()
        lost = [e for e in events_of(sink) if e.kind == "acquire.events_lost"]
        assert [e.detail for e in lost] == [{"lost": 4}]
