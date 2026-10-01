"""The log of hardware events that `acquire` keeps for `core`."""

from __future__ import annotations

import json
import logging
import threading

import pytest

from seeingmon.drivers.base import CameraError
from seeingmon.hardware.events import HardwareEvent
from seeingmon.services.acquire.events import (
    EventBatch,
    HardwareEventLog,
    after_of,
    decode_batch,
    encode_batch,
)
from seeingmon.services.acquire.factory import accepts_events
from seeingmon.services.ipc.codec import CodecError
from seeingmon.services.ipc.endpoint import Endpoint
from seeingmon.services.ipc.keys import ConnectionKey

from .rig import make_rig


def event(number: int, level: str = "info") -> HardwareEvent:
    return HardwareEvent(level, "camera.recovery", f"event {number}", 1000 + number, {"n": number})


class TestLog:
    def test_events_get_consecutive_numbers(self) -> None:
        log = HardwareEventLog()
        for number in range(1, 4):
            log.record(event(number))
        batch = log.since()
        assert [item.seq for item in batch.events] == [1, 2, 3]
        assert [item.event.message for item in batch.events] == ["event 1", "event 2", "event 3"]
        assert (batch.last, batch.lost) == (3, 0)
        assert log.last == 3

    def test_since_returns_only_what_came_after(self) -> None:
        log = HardwareEventLog()
        for number in range(1, 6):
            log.record(event(number))
        assert [item.seq for item in log.since(3).events] == [4, 5]
        assert log.since(5).events == ()
        assert log.since(99).events == ()
        assert log.since(99).lost == 0

    def test_an_empty_log_has_nothing_and_loses_nothing(self) -> None:
        batch = HardwareEventLog().since(0)
        assert batch == EventBatch((), 0, 0)

    def test_a_full_log_forgets_the_oldest_and_says_how_many(self) -> None:
        log = HardwareEventLog(size=3)
        for number in range(1, 6):
            log.record(event(number))
        everything = log.since(0)
        assert [item.seq for item in everything.events] == [3, 4, 5]
        assert everything.lost == 2  # events 1 and 2 are gone
        assert log.since(1).lost == 1
        assert log.since(2).lost == 0
        assert log.since(2).events[0].seq == 3

    def test_the_size_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            HardwareEventLog(size=0)

    def test_an_event_is_also_written_to_the_log_of_the_process(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        log = HardwareEventLog()
        with caplog.at_level(logging.INFO):
            log.record(event(1, "warning"))
        record = caplog.records[-1]
        assert record.levelno == logging.WARNING
        assert "camera.recovery" in record.getMessage()

    def test_many_threads_can_record_at_once(self) -> None:
        log = HardwareEventLog(size=1000)

        def work() -> None:
            for number in range(100):
                log.record(event(number))

        threads = [threading.Thread(target=work) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10.0)
        assert log.last == 800
        assert [item.seq for item in log.since(0).events] == list(range(1, 801))


class TestJson:
    def test_a_batch_survives_the_trip(self) -> None:
        log = HardwareEventLog()
        log.record(event(1))
        log.record(HardwareEvent("error", "power.cycle_failed", "no route", 5, None))
        batch = log.since(0)
        again = decode_batch(json.loads(json.dumps(encode_batch(batch))))
        assert again == batch

    @pytest.mark.parametrize(
        "bad",
        [
            None,
            {"events": "x", "last": 0, "lost": 0},
            {"events": [], "last": "0", "lost": 0},
            {"events": [{"seq": 1}], "last": 1, "lost": 0},
            {
                "events": [
                    {"seq": 1, "level": "loud", "kind": "a.b", "message": "m", "t_utc_ns": 1}
                ],
                "last": 1,
                "lost": 0,
            },
            {
                "events": [
                    {"seq": 1, "level": "info", "kind": "nodots", "message": "m", "t_utc_ns": 1}
                ],
                "last": 1,
                "lost": 0,
            },
        ],
    )
    def test_a_malformed_batch_is_a_codec_error(self, bad: object) -> None:
        with pytest.raises(CodecError):
            decode_batch(bad)

    def test_the_after_parameter_defaults_and_refuses_negatives(self) -> None:
        assert after_of({}) == 0
        assert after_of({"after": 7}) == 7
        with pytest.raises(CodecError, match="negative"):
            after_of({"after": -1})
        with pytest.raises(CodecError):
            after_of({"after": "7"})


class TestFactoryHook:
    def test_the_asi_driver_takes_events_and_the_others_do_not(self) -> None:
        assert accepts_events("asi")
        assert not accepts_events("sim")
        assert not accepts_events("replay")
        assert not accepts_events("fake")
        assert not accepts_events("does_not_exist")
        assert not accepts_events("base")
        assert not accepts_events("_private")


class TestThroughTheService:
    def test_events_reach_the_remote_driver_with_or_without_a_stream(
        self, native: Endpoint, key: ConnectionKey
    ) -> None:
        rig = make_rig(native, key)
        try:
            rig.service.events.record(event(1))
            rig.service.events.record(event(2, "warning"))
            driver = rig.driver()
            driver.open()
            batch = driver.events()
            assert [item.event.message for item in batch.events] == ["event 1", "event 2"]
            assert driver.events(after=batch.last).events == ()
            rig.service.events.record(event(3))
            assert [item.seq for item in driver.events(after=batch.last).events] == [3]
            assert driver.health()["events_recorded"] == 3
            with pytest.raises(CameraError):
                driver.events(after=-1)
        finally:
            rig.close()
