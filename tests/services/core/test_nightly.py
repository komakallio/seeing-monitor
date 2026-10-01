"""The nightly star summary: closed at the end of the night, at shutdown, and never twice."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence

import numpy as np
import pytest

from seeingmon.analysis import SurveyOutput
from seeingmon.clock import VirtualClock, iso_to_utc_ns
from seeingmon.frames import Frame
from seeingmon.records import EventRecord, Record
from seeingmon.services.core.nightly import NightlySummary
from tests.scheduler.helpers import make_frame

EVENING = iso_to_utc_ns("2026-01-10T20:00:00Z")  # the night of 2026-01-10
MORNING_OF_THE_SAME_NIGHT = iso_to_utc_ns("2026-01-11T06:00:00Z")
NEXT_NOON = iso_to_utc_ns("2026-01-11T12:00:30Z")  # the first moment of the next night
NEXT_EVENING = iso_to_utc_ns("2026-01-11T19:00:00Z")


def marker(name: str) -> Record:
    return EventRecord(
        station_id="test",
        t_utc_ns=EVENING,
        profile_id="profile",
        provenance={"source": "local"},
        level="info",
        kind=name,
        message="A stand-in for the star summary.",
    )


class FakeAnalyzer:
    """A survey analyzer with a nightly summary: it holds a record for the open night."""

    def __init__(self) -> None:
        self.waiting = 0
        self.open_night: list[Record] = [marker("night.one")]
        self.flushes = 0
        self.closed = False
        self.tracker = object()
        self.poll_gate: threading.Event | None = None
        self.polling = threading.Event()

    def submit(self, frame: Frame) -> None:
        self.waiting += 1

    def poll(self) -> tuple[SurveyOutput, ...]:
        if self.poll_gate is not None:
            self.polling.set()
            self.poll_gate.wait(10.0)
        return ()

    def pending(self) -> int:
        return self.waiting

    def flush_night(self) -> Sequence[Record]:
        self.flushes += 1
        records, self.open_night = self.open_night, []
        return records

    def close(self) -> None:
        self.closed = True


def frame_at(t_utc_ns: int) -> Frame:
    return make_frame(np.zeros((8, 8), dtype=np.uint16), mode="bin2", t_utc_ns=t_utc_ns)


class Rig:
    def __init__(self, start: int = EVENING) -> None:
        self.clock = VirtualClock(start)
        self.analyzer = FakeAnalyzer()
        self.written: list[Record] = []
        self.summary = NightlySummary(
            self.analyzer, write=self.written.append, clock=self.clock, split_utc_hour=12.0
        )


class TestTheEndOfTheNight:
    def test_nothing_happens_while_the_night_lasts(self) -> None:
        rig = Rig()
        rig.clock.advance_to_utc_ns(MORNING_OF_THE_SAME_NIGHT)
        assert rig.summary.flush_due() == 0
        assert rig.analyzer.flushes == 0

    def test_the_record_is_written_once_the_clock_passes_the_split_hour(self) -> None:
        rig = Rig()
        rig.clock.advance_to_utc_ns(NEXT_NOON)
        assert rig.summary.flush_due() == 1
        assert [r.kind for r in rig.written] == ["night.one"]  # type: ignore[attr-defined]
        assert rig.summary.written == 1
        assert rig.summary.flush_due() == 0  # asked again in the same night: nothing more
        assert rig.analyzer.flushes == 1

    def test_the_next_night_closes_at_the_next_noon_too(self) -> None:
        rig = Rig()
        rig.clock.advance_to_utc_ns(NEXT_NOON)
        rig.summary.flush_due()
        rig.analyzer.open_night = [marker("night.two")]
        rig.clock.advance_to_utc_ns(iso_to_utc_ns("2026-01-12T12:00:30Z"))
        assert rig.summary.flush_due() == 1
        assert [r.kind for r in rig.written] == ["night.one", "night.two"]  # type: ignore[attr-defined]

    def test_a_frame_in_flight_delays_the_close_until_it_is_done(self) -> None:
        rig = Rig()
        rig.summary.submit(frame_at(EVENING + 3600 * 10**9))
        rig.clock.advance_to_utc_ns(NEXT_NOON)
        assert rig.summary.flush_due() == 0  # the analysis still holds a frame of the old night
        assert rig.analyzer.flushes == 0
        rig.analyzer.waiting = 0
        assert rig.summary.flush_due() == 1

    def test_a_frame_of_the_new_night_has_closed_the_old_one_already(self) -> None:
        rig = Rig()
        rig.clock.advance_to_utc_ns(NEXT_NOON)
        rig.summary.submit(frame_at(NEXT_NOON))  # a high-latitude station: it is dark at noon
        rig.analyzer.waiting = 0
        assert rig.summary.flush_due() == 0  # closing again would cut the new night short
        assert rig.analyzer.flushes == 0
        rig.clock.advance_to_utc_ns(NEXT_EVENING)
        assert rig.summary.flush_due() == 0  # the new night goes on

    def test_a_restart_in_the_middle_of_a_night_closes_that_night_at_its_end(self) -> None:
        rig = Rig(start=MORNING_OF_THE_SAME_NIGHT)
        assert rig.summary.flush_due() == 0
        rig.clock.advance_to_utc_ns(NEXT_NOON)
        assert rig.summary.flush_due() == 1


class TestShutdown:
    def test_the_open_night_is_written_whatever_the_time(self) -> None:
        rig = Rig()
        assert rig.summary.flush() == 1
        assert [r.kind for r in rig.written] == ["night.one"]  # type: ignore[attr-defined]
        assert rig.summary.flush() == 0  # nothing is open any more

    def test_a_record_that_cannot_be_stored_is_logged_and_the_others_still_go(self) -> None:
        rig = Rig()
        rig.analyzer.open_night = [marker("night.bad"), marker("night.good")]
        accepted: list[Record] = []

        def write(record: Record) -> None:
            if record.kind == "night.bad":  # type: ignore[attr-defined]
                raise OSError("the disk is full")
            accepted.append(record)

        summary = NightlySummary(rig.analyzer, write=write, clock=rig.clock)
        assert summary.flush() == 1
        assert [r.kind for r in accepted] == ["night.good"]  # type: ignore[attr-defined]


class TestAsAnalyzer:
    def test_the_rest_of_the_analyzer_shows_through(self) -> None:
        rig = Rig()
        assert rig.summary.tracker is rig.analyzer.tracker
        rig.summary.close()
        assert rig.analyzer.closed
        with pytest.raises(AttributeError):
            rig.summary._private  # noqa: B018

    def test_submit_and_pending_go_to_the_analyzer(self) -> None:
        rig = Rig()
        rig.summary.submit(frame_at(EVENING))
        assert rig.summary.pending() == 1

    def test_a_close_waits_for_a_poll_that_is_running(self) -> None:
        rig = Rig()
        rig.analyzer.poll_gate = threading.Event()
        poller = threading.Thread(target=rig.summary.poll)
        poller.start()
        assert rig.analyzer.polling.wait(10.0)
        flushed: list[int] = []
        closer = threading.Thread(target=lambda: flushed.append(rig.summary.flush()))
        closer.start()
        time.sleep(0.2)
        assert flushed == []  # the poll holds the accumulator, so the close waits
        rig.analyzer.poll_gate.set()
        poller.join(10.0)
        closer.join(10.0)
        assert flushed == [1]
