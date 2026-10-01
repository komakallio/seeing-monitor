"""The forwarder: outage and resume, restarts, independent sinks, backoff, and the backlog."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

from seeingmon.clock import VirtualClock
from seeingmon.records.base import Record
from seeingmon.records.system import HealthRecord
from seeingmon.sinks.base import Sink, StoredRow
from seeingmon.sinks.forwarder import (
    BACKOFF,
    FAILED,
    OK,
    PARKED,
    SENT,
    STORE_ERROR,
    Forwarder,
)
from seeingmon.store.config import ForwarderConfig
from seeingmon.store.db import Store, StoreBusyError
from seeingmon.store.events import EventEmitter
from seeingmon.testing.fakes import FakeSink
from tests.sinks.helpers import OutageSink
from tests.store.builders import NS_PER_S, T0, make_event, make_health, make_window

CONFIG = ForwarderConfig(batch_rows=100, max_batches_per_pass=20)
TYPES = ("health", "seeing_window", "event")


class Feed:
    """Writes records as a station would: a health record a minute, a window every ten."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self.minute = 0

    def tick(self, minutes: int = 1) -> int:
        records: list[Record] = []
        for _ in range(minutes):
            t_utc_ns = T0 + self.minute * 60 * NS_PER_S
            records.append(make_health(t_utc_ns))
            if self.minute % 10 == 0:
                records.append(make_window(t_utc_ns))
            self.minute += 1
        self.store.write_many(records)
        return len(records)


def drain(
    forwarder: Forwarder, clock: VirtualClock, *, step_s: float = 60, passes: int = 5000
) -> None:
    """Run passes until every sink has no backlog, moving the clock on between them."""
    for _ in range(passes):
        report = forwarder.run_once()
        if not report.more_pending and not any(forwarder.sink_backlog().values()):
            return
        clock.advance(step_s)
    raise AssertionError(f"the backlog did not drain: {forwarder.sink_backlog()}")


def event_kinds(store: Store) -> list[str]:
    return [row.values["kind"] for row in store.after("event", 0, 100_000)]


def assert_complete(store: Store, sink: OutageSink, *, types: tuple[str, ...] = TYPES) -> None:
    """The sink holds every row of the store once, and received them in order."""
    for record_type in types:
        expected = list(range(1, store.last_row_id(record_type) + 1))
        assert sink.row_ids(record_type) == expected, record_type  # nothing is lost
        assert sink.received[record_type] == expected, record_type  # in order, none twice
        if expected:
            assert store.cursor(sink.name, record_type) == expected[-1], record_type


class TestOutageAndResume:
    def test_a_sink_that_is_down_for_three_days_loses_nothing_and_keeps_order(
        self, store: Store, clock: VirtualClock, emitter: EventEmitter
    ) -> None:
        sink = OutageSink("influx", clock)
        forwarder = Forwarder(store, [sink], clock, CONFIG, events=emitter, rng=random.Random(7))
        feed = Feed(store)
        for _ in range(24):  # two healthy hours, in five-minute ticks
            feed.tick(5)
            forwarder.run_once()
            clock.advance(300)
        assert forwarder.sink_backlog() == {"influx": 0}
        assert sink.repeated == 0

        sink.up = False
        attempts_before = sink.attempts
        backlog_curve: list[int] = []
        for tick in range(864):  # three days in five-minute ticks
            feed.tick(5)
            forwarder.run_once()
            clock.advance(300)
            if tick % 24 == 23:
                backlog_curve.append(forwarder.sink_backlog()["influx"])
        assert backlog_curve == sorted(backlog_curve)  # the backlog only grows
        assert backlog_curve[-1] > 4000
        for record_type in TYPES:  # and the backlog is exact, for each record type
            held = len(sink.stored[record_type])
            assert forwarder.backlog()["influx"][record_type] == store.count(record_type) - held

        # The backoff bounds the retries: about one an hour at first, then every 10 minutes.
        outage_attempts = sink.attempts - attempts_before
        assert 8 < outage_attempts < 600
        gaps = [
            b - a for a, b in zip(sink.attempt_times_ns, sink.attempt_times_ns[1:], strict=False)
        ]
        steady = gaps[-50:]
        assert min(steady) >= 450 * NS_PER_S  # the longest wait is 600 s, less up to 25% jitter
        assert max(steady) <= 900 * NS_PER_S  # and a tick is 300 s

        sink.up = True
        drain(forwarder, clock)
        assert_complete(store, sink)
        assert sink.repeated == 0
        assert forwarder.sink_backlog() == {"influx": 0}
        kinds = event_kinds(store)
        assert kinds.count("sink.unavailable") == 1
        assert kinds.count("sink.recovered") == 1
        assert kinds.index("sink.unavailable") < kinds.index("sink.recovered")

    def test_a_restart_in_the_middle_resumes_from_the_stored_cursors(
        self, store: Store, clock: VirtualClock
    ) -> None:
        Feed(store).tick(500)
        sink = OutageSink("influx", clock, max_batch_rows=50)
        config = ForwarderConfig(batch_rows=50, max_batches_per_pass=3)
        first = Forwarder(store, [sink], clock, config, rng=random.Random(1))
        first.run_once()
        assert len(sink.batches) == 3
        cursors = {t: store.cursor("influx", t) for t in TYPES}
        assert sum(cursors.values()) > 0
        del first  # the process dies here

        second = Forwarder(store, [sink], clock, config, rng=random.Random(2))
        drain(second, clock)
        assert_complete(store, sink)
        assert sink.repeated == 0
        batch_type, first_row, _ = sink.batches[3]  # the first batch after the restart
        assert first_row == cursors[batch_type] + 1

    def test_a_restart_during_an_outage_retries_at_once_and_loses_nothing(
        self, store: Store, clock: VirtualClock
    ) -> None:
        feed = Feed(store)
        sink = OutageSink("influx", clock)
        first = Forwarder(store, [sink], clock, CONFIG, rng=random.Random(1))
        feed.tick(30)
        sink.up = False
        first.run_once()
        assert first.status()["influx"].state == BACKOFF
        second = Forwarder(store, [sink], clock, CONFIG, rng=random.Random(1))  # the wait is lost
        assert second.run_once().passes[0].state == FAILED  # it tries right away
        sink.up = True
        feed.tick(30)
        drain(second, clock)
        assert_complete(store, sink)
        assert sink.repeated == 0

    def test_a_crash_between_send_and_cursor_repeats_only_that_batch(
        self, store: Store, clock: VirtualClock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        Feed(store).tick(300)
        sink = OutageSink("influx", clock, max_batch_rows=40)
        config = ForwarderConfig(batch_rows=40, max_batches_per_pass=100)
        forwarder = Forwarder(store, [sink], clock, config)
        real = store.advance_cursor
        calls: list[int] = []

        def lose_one(sink_name: str, record_type: str, row_id: int, **kwargs: Any) -> int:
            calls.append(row_id)
            if len(calls) == 2:
                raise StoreBusyError("the database is locked")  # the sink took the batch
            return real(sink_name, record_type, row_id, **kwargs)

        monkeypatch.setattr(store, "advance_cursor", lose_one)
        first = forwarder.run_once()
        assert first.passes[0].state == STORE_ERROR
        monkeypatch.undo()
        drain(forwarder, clock)
        assert sink.repeated == 40  # delivery is at least once: that one batch came twice
        for record_type in TYPES:
            assert sink.row_ids(record_type) == list(range(1, store.last_row_id(record_type) + 1))
            assert store.cursor("influx", record_type) == store.last_row_id(record_type)


class TestIndependentSinks:
    def test_a_down_sink_never_blocks_a_healthy_one(
        self, store: Store, clock: VirtualClock, emitter: EventEmitter
    ) -> None:
        flaky = OutageSink("flaky", clock)
        steady = OutageSink("steady", clock)
        forwarder = Forwarder(store, [flaky, steady], clock, CONFIG, events=emitter)
        feed = Feed(store)
        flaky.up = False
        for _ in range(100):
            feed.tick(5)
            forwarder.run_once()
            clock.advance(300)
            assert forwarder.sink_backlog()["steady"] == 0  # the healthy sink is always current
        assert forwarder.sink_backlog()["flaky"] > 500
        flaky.up = True
        drain(forwarder, clock)
        assert_complete(store, flaky)
        assert_complete(store, steady)
        assert steady.repeated == 0
        assert flaky.repeated == 0

    def test_a_parked_sink_stops_while_the_other_goes_on(
        self, store: Store, clock: VirtualClock, emitter: EventEmitter
    ) -> None:
        broken = OutageSink("broken", clock)
        steady = OutageSink("steady", clock)
        broken.permanent = True
        forwarder = Forwarder(store, [broken, steady], clock, CONFIG, events=emitter)
        feed = Feed(store)
        feed.tick(50)
        report = forwarder.run_once()
        assert [p.state for p in report.passes] == [FAILED, SENT]
        for _ in range(20):
            feed.tick(5)
            forwarder.run_once()
            clock.advance(600)
        assert broken.attempts == 1  # a parked sink gets no more attempts
        assert forwarder.status()["broken"].state == PARKED
        assert store.cursor("broken", "health") == 0  # the cursor stays where it was
        assert forwarder.sink_backlog()["steady"] == 0
        assert event_kinds(store).count("sink.parked") == 1

    def test_resume_lets_a_parked_sink_try_again(
        self, store: Store, clock: VirtualClock, emitter: EventEmitter
    ) -> None:
        sink = OutageSink("influx", clock)
        sink.permanent = True
        forwarder = Forwarder(store, [sink], clock, CONFIG, events=emitter)
        Feed(store).tick(20)
        forwarder.run_once()
        assert forwarder.status()["influx"].state == PARKED
        sink.permanent = False  # the operator fixed the credentials
        assert forwarder.resume("influx") is True
        assert forwarder.resume("influx") is False  # nothing was waiting the second time
        drain(forwarder, clock)
        assert_complete(store, sink)
        with pytest.raises(KeyError):
            forwarder.resume("no-such-sink")

    def test_a_new_sink_backfills_from_row_zero_and_the_old_one_gets_no_repeats(
        self, store: Store, clock: VirtualClock
    ) -> None:
        feed = Feed(store)
        old = OutageSink("old", clock)
        feed.tick(400)
        drain(Forwarder(store, [old], clock, CONFIG), clock)
        feed.tick(100)
        new = OutageSink("new", clock)  # added months later, with the full history waiting
        forwarder = Forwarder(store, [old, new], clock, CONFIG)
        assert forwarder.sink_backlog()["new"] == store.count("health") + store.count(
            "seeing_window"
        )
        drain(forwarder, clock)
        assert_complete(store, old)
        assert_complete(store, new)
        assert old.repeated == 0
        assert new.repeated == 0

    def test_each_sink_keeps_its_own_cursors(self, store: Store, clock: VirtualClock) -> None:
        Feed(store).tick(30)
        fast = OutageSink("fast", clock, max_batch_rows=100)
        slow = OutageSink("slow", clock, max_batch_rows=5)
        forwarder = Forwarder(store, [fast, slow], clock, ForwarderConfig(max_batches_per_pass=3))
        forwarder.run_once()
        assert store.cursor("fast", "health") == 30  # the windows and then all the health rows
        assert store.cursor("slow", "seeing_window") == 3
        assert store.cursor("slow", "health") == 10  # two batches of 5, and then the pass ended


class TestFailures:
    def test_an_unexpected_exception_is_a_retryable_failure_without_its_message(
        self, store: Store, clock: VirtualClock, emitter: EventEmitter
    ) -> None:
        class Buggy(OutageSink):
            def send(self, record_type: str, rows: Any) -> None:
                raise RuntimeError("the adapter holds PRIVATE-VALUE")

        buggy = Buggy("buggy", clock)
        steady = OutageSink("steady", clock)
        forwarder = Forwarder(store, [buggy, steady], clock, CONFIG, events=emitter)
        Feed(store).tick(10)
        forwarder.run_once()
        status = forwarder.status()["buggy"]
        assert status.state == BACKOFF
        assert status.last_error == "unexpected RuntimeError"
        assert forwarder.sink_backlog()["steady"] == 0
        (unavailable,) = [
            r for r in store.after("event", 0) if r.values["kind"] == "sink.unavailable"
        ]
        assert "RuntimeError" in unavailable.values["message"]
        assert "PRIVATE-VALUE" not in unavailable.values["message"]
        assert "PRIVATE-VALUE" not in str(unavailable.values["detail"])

    def test_the_events_of_an_outage_say_which_sink_and_what_happened(
        self, store: Store, clock: VirtualClock, emitter: EventEmitter
    ) -> None:
        sink = OutageSink("influx", clock)
        forwarder = Forwarder(store, [sink], clock, CONFIG, events=emitter, rng=random.Random(3))
        Feed(store).tick(10)
        sink.up = False
        forwarder.run_once()
        clock.advance(10_000)
        forwarder.run_once()  # a second failure writes no second event
        sink.up = True
        clock.advance(10_000)
        forwarder.run_once()
        rows = [r for r in store.after("event", 0) if r.values["kind"].startswith("sink.")]
        assert [r.values["kind"] for r in rows] == ["sink.unavailable", "sink.recovered"]
        unavailable, recovered = (r.values for r in rows)
        assert (unavailable["level"], recovered["level"]) == ("warning", "info")
        assert unavailable["detail"]["sink"] == "influx"
        assert unavailable["detail"]["failures"] == 1
        assert unavailable["detail"]["retry_in_s"] > 0
        assert "the endpoint is down" in unavailable["message"]
        assert recovered["detail"]["failures"] == 2

    def test_a_second_outage_writes_a_second_pair_of_events(
        self, store: Store, clock: VirtualClock, emitter: EventEmitter
    ) -> None:
        sink = OutageSink("influx", clock)
        forwarder = Forwarder(store, [sink], clock, CONFIG, events=emitter)
        feed = Feed(store)
        for _ in range(2):
            feed.tick(5)
            sink.up = False
            forwarder.run_once()
            sink.up = True
            drain(forwarder, clock)
        assert [k for k in event_kinds(store) if k.startswith("sink.")] == [
            "sink.unavailable",
            "sink.recovered",
            "sink.unavailable",
            "sink.recovered",
        ]

    def test_a_permanent_error_writes_one_error_event_and_leaves_the_cursor(
        self, store: Store, clock: VirtualClock, emitter: EventEmitter
    ) -> None:
        sink = OutageSink("influx", clock)
        forwarder = Forwarder(store, [sink], clock, CONFIG, events=emitter)
        feed = Feed(store)
        feed.tick(10)
        forwarder.run_once()  # the first batches go through
        cursors = {t: store.cursor("influx", t) for t in TYPES}
        sink.permanent = True
        feed.tick(10)
        report = forwarder.run_once()
        assert report.passes[0].state == FAILED
        assert {t: store.cursor("influx", t) for t in TYPES} == cursors
        (parked,) = [r for r in store.after("event", 0) if r.values["kind"] == "sink.parked"]
        assert parked.values["level"] == "error"
        assert parked.values["detail"]["sink"] == "influx"
        assert "refused the credentials" in parked.values["message"]
        assert forwarder.run_once().passes[0].state == PARKED

    def test_a_failing_store_is_not_a_failing_sink(
        self, store: Store, clock: VirtualClock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sink = OutageSink("influx", clock)
        forwarder = Forwarder(store, [sink], clock, CONFIG)
        Feed(store).tick(10)

        def locked(*args: Any, **kwargs: Any) -> Any:
            raise StoreBusyError("the database is locked")

        monkeypatch.setattr(store, "after", locked)
        report = forwarder.run_once()
        assert report.passes[0].state == STORE_ERROR
        assert sink.attempts == 0
        assert forwarder.status()["influx"].state == OK  # no backoff, no failure count
        monkeypatch.undo()
        assert forwarder.run_once().passes[0].state == SENT


class TestBackoff:
    def waits(self, forwarder: Forwarder, clock: VirtualClock, count: int) -> list[float]:
        """Fail `count` times in a row, and return the wait that each failure set."""
        waits = []
        for _ in range(count):
            forwarder.run_once()
            wait = forwarder.status()["influx"].retry_in_s
            assert wait is not None
            waits.append(wait)
            clock.advance(wait)
        return waits

    def test_the_wait_doubles_to_the_longest_wait_without_jitter(
        self, store: Store, clock: VirtualClock
    ) -> None:
        sink = OutageSink("influx", clock)
        sink.up = False
        config = ForwarderConfig(jitter=0)
        forwarder = Forwarder(store, [sink], clock, config)
        Feed(store).tick(3)
        assert self.waits(forwarder, clock, 11) == [5, 10, 20, 40, 80, 160, 320, 600, 600, 600, 600]

    def test_jitter_varies_the_wait_within_its_bounds_and_follows_the_seed(
        self, store: Store, clock: VirtualClock
    ) -> None:
        Feed(store).tick(3)
        runs = []
        for seed in (11, 11, 12):
            local = VirtualClock(T0)
            sink = OutageSink("influx", local)
            sink.up = False
            forwarder = Forwarder(store, [sink], local, ForwarderConfig(), rng=random.Random(seed))
            runs.append(self.waits(forwarder, local, 10))
        assert runs[0] == runs[1]  # the same seed gives the same waits
        assert runs[0] != runs[2]
        for waits in runs:
            for failure, wait in enumerate(waits, start=1):
                base = min(5 * 2 ** (failure - 1), 600)
                assert base * 0.75 <= wait <= min(base * 1.25, 600) + 1e-6

    def test_a_success_resets_the_wait(self, store: Store, clock: VirtualClock) -> None:
        sink = OutageSink("influx", clock)
        forwarder = Forwarder(store, [sink], clock, ForwarderConfig(jitter=0))
        feed = Feed(store)
        feed.tick(3)
        sink.up = False
        assert self.waits(forwarder, clock, 4) == [5, 10, 20, 40]
        sink.up = True
        forwarder.run_once()
        assert forwarder.status()["influx"].failures == 0
        feed.tick(3)
        sink.up = False
        assert self.waits(forwarder, clock, 1) == [5]  # back to the first wait

    def test_a_sink_in_backoff_is_not_called_before_the_wait_ends(
        self, store: Store, clock: VirtualClock
    ) -> None:
        sink = OutageSink("influx", clock)
        sink.up = False
        forwarder = Forwarder(store, [sink], clock, ForwarderConfig(jitter=0))
        Feed(store).tick(3)
        forwarder.run_once()
        clock.advance(4.9)
        assert forwarder.run_once().passes[0].state == BACKOFF
        assert sink.attempts == 1
        clock.advance(0.1)
        assert forwarder.run_once().passes[0].state == FAILED
        assert sink.attempts == 2

    def test_a_clock_step_of_the_wall_clock_does_not_change_the_wait(
        self, store: Store, clock: VirtualClock
    ) -> None:
        sink = OutageSink("influx", clock)
        sink.up = False
        forwarder = Forwarder(store, [sink], clock, ForwarderConfig(jitter=0))
        Feed(store).tick(3)
        forwarder.run_once()
        clock.step_utc_ns(-86_400 * NS_PER_S)  # NTP steps the wall clock back a day
        assert forwarder.run_once().passes[0].state == BACKOFF
        clock.advance(5)
        assert forwarder.run_once().passes[0].state == FAILED  # the monotonic wait still ended


class TestBacklog:
    def test_the_backlog_counts_the_rows_after_each_cursor_by_record_type(
        self, store: Store, clock: VirtualClock
    ) -> None:
        Feed(store).tick(25)  # 25 health rows and 3 windows
        health_only = OutageSink("health_only", clock, record_types={"health"})
        everything = OutageSink("everything", clock)
        forwarder = Forwarder(store, [health_only, everything], clock, CONFIG)
        backlog = forwarder.backlog()
        assert backlog["health_only"] == {"health": 25}
        assert backlog["everything"]["health"] == 25
        assert backlog["everything"]["seeing_window"] == 3
        assert backlog["everything"]["event"] == 0
        assert "frame" not in backlog["everything"]  # segment records have no rows to forward
        assert forwarder.sink_backlog() == {"health_only": 25, "everything": 28}
        forwarder.run_once()
        assert forwarder.sink_backlog() == {"health_only": 0, "everything": 0}

    def test_the_sink_backlog_fits_the_health_record(
        self, store: Store, clock: VirtualClock
    ) -> None:
        Feed(store).tick(12)
        sink = OutageSink("influx", clock)
        forwarder = Forwarder(store, [sink], clock, CONFIG)
        health = make_health(T0 + 10**12, sink_backlog=forwarder.sink_backlog())
        assert isinstance(health, HealthRecord)
        assert health.sink_backlog == {"influx": 14}

    def test_a_sink_over_the_limit_is_named(self, store: Store, clock: VirtualClock) -> None:
        Feed(store).tick(60)
        quiet = OutageSink("quiet", clock, record_types={"event"})
        loud = OutageSink("loud", clock)
        forwarder = Forwarder(store, [quiet, loud], clock, ForwarderConfig(backlog_flag_rows=30))
        assert forwarder.backlog_exceeded() == ["loud"]
        forwarder.run_once()
        assert forwarder.backlog_exceeded() == []


class TestFiltersAndContract:
    def test_a_sink_gets_only_the_record_types_it_accepts(
        self, store: Store, clock: VirtualClock
    ) -> None:
        Feed(store).tick(15)
        sink = OutageSink("influx", clock, record_types={"health"})
        Forwarder(store, [sink], clock, CONFIG).run_once()
        assert set(sink.stored) == {"health"}
        assert store.cursor("influx", "seeing_window") == 0

    def test_a_sink_that_accepts_everything_never_gets_the_segment_type(
        self, store: Store, clock: VirtualClock
    ) -> None:
        Feed(store).tick(5)
        sink = FakeSink("everything")
        assert sink.accepts("frame")
        Forwarder(store, [sink], clock, CONFIG).run_once()
        assert {kind for kind, _ in sink.sent} == {"health", "seeing_window"}

    def test_batches_never_exceed_the_batch_limits(self, store: Store, clock: VirtualClock) -> None:
        Feed(store).tick(100)
        sink = FakeSink(max_batch_rows=7)  # the fake itself rejects a larger batch
        forwarder = Forwarder(store, [sink], clock, ForwarderConfig(batch_rows=5))
        drain(forwarder, clock)
        sizes = [len(batch) for _, batch in sink.sent]
        assert max(sizes) == 5  # the smaller of the sink limit and the configured limit
        assert [row.row_id for row in sink.rows("health")] == list(range(1, 101))

    def test_the_fake_sink_keeps_a_failed_batch_for_the_retry(
        self, store: Store, clock: VirtualClock
    ) -> None:
        Feed(store).tick(4)
        sink = FakeSink(record_types={"health"}, max_batch_rows=3)
        sink.fail_next(retryable=True)
        forwarder = Forwarder(store, [sink], clock, ForwarderConfig(jitter=0))
        assert forwarder.run_once().passes[0].state == FAILED
        assert sink.sent == []
        clock.advance(5)
        drain(forwarder, clock)
        assert [row.row_id for row in sink.rows("health")] == [1, 2, 3, 4]

    def test_rows_reach_a_sink_as_stored_rows_with_the_record_values(
        self, store: Store, clock: VirtualClock
    ) -> None:
        record = make_health(T0, state="paused")
        store.write(record)
        sink = FakeSink(record_types={"health"})
        Forwarder(store, [sink], clock, CONFIG).run_once()
        (row,) = sink.rows("health")
        assert isinstance(row, StoredRow)
        assert row.values == record.to_row()

    def test_the_sink_names_must_be_unique_and_the_batch_size_positive(
        self, store: Store, clock: VirtualClock
    ) -> None:
        with pytest.raises(ValueError, match="unique name"):
            Forwarder(store, [FakeSink("a"), FakeSink("a")], clock)
        with pytest.raises(ValueError, match="max_batch_rows"):
            Forwarder(store, [FakeSink("a", max_batch_rows=0)], clock)

    def test_close_closes_the_sinks_that_can_close(self, store: Store, clock: VirtualClock) -> None:
        closed: list[str] = []

        class Closing(OutageSink):
            def close(self) -> None:
                closed.append(self.name)

        Forwarder(store, [Closing("a", clock), FakeSink("b")], clock).close()
        assert closed == ["a"]

    def test_the_outage_sink_double_satisfies_the_sink_protocol(self, clock: VirtualClock) -> None:
        sink: Sink = OutageSink("a", clock)
        assert isinstance(sink, Sink)


class TestLoop:
    def test_the_loop_passes_until_it_is_told_to_stop_and_sleeps_on_the_clock(
        self, store: Store, clock: VirtualClock
    ) -> None:
        Feed(store).tick(10)
        sink = OutageSink("influx", clock)
        forwarder = Forwarder(store, [sink], clock, ForwarderConfig(poll_interval_s=2))
        stop_at = clock.utc_ns() + 10 * NS_PER_S
        passes = forwarder.run_loop(lambda: clock.utc_ns() >= stop_at)
        assert passes == 5  # the first pass sends, and each pass sleeps for 2 s
        assert sink.row_ids("health") == list(range(1, 11))

    def test_the_loop_ends_after_max_passes(self, store: Store, clock: VirtualClock) -> None:
        forwarder = Forwarder(store, [OutageSink("influx", clock)], clock)
        assert forwarder.run_loop(lambda: False, max_passes=4) == 4

    def test_the_loop_does_not_sleep_while_a_healthy_sink_has_rows_waiting(
        self, store: Store, clock: VirtualClock
    ) -> None:
        Feed(store).tick(100)
        sink = OutageSink("influx", clock, max_batch_rows=10)
        config = ForwarderConfig(batch_rows=10, max_batches_per_pass=1, poll_interval_s=1)
        forwarder = Forwarder(store, [sink], clock, config)
        start = clock.monotonic_ns()
        forwarder.run_loop(lambda: not any(forwarder.sink_backlog().values()))
        assert clock.monotonic_ns() == start  # it drained 110 rows without one sleep
        assert len(sink.row_ids("health")) == 100

    def test_the_loop_sleeps_until_a_retry_is_due_when_that_is_sooner_than_the_poll(
        self, store: Store, clock: VirtualClock
    ) -> None:
        Feed(store).tick(3)
        sink = OutageSink("influx", clock)
        sink.up = False
        config = ForwarderConfig(jitter=0, backoff_initial_s=0.25, poll_interval_s=10)
        forwarder = Forwarder(store, [sink], clock, config)
        forwarder.run_loop(lambda: False, max_passes=2)
        # Pass 1 fails and sets a wait of 0.25 s, so the loop sleeps 0.25 s, not the 10 s poll.
        assert clock.monotonic_ns() == 250_000_000
        assert sink.attempts == 2  # and pass 2 finds the retry due


class TestEventsReachSinksOnce:
    """Events that share a timestamp land one nanosecond apart, and every sink gets each once."""

    def test_five_events_at_one_time_reach_a_fake_sink_exactly_once(
        self, store: Store, clock: VirtualClock
    ) -> None:
        for n in range(5):
            store.write(make_event(T0, message=f"event {n}"))
        sink = FakeSink(record_types={"event"}, max_batch_rows=2)
        forwarder = Forwarder(store, [sink], clock, ForwarderConfig(batch_rows=2))
        drain(forwarder, clock)
        forwarder.run_once()  # a further pass sends nothing again
        rows = sink.rows("event")
        assert [row.row_id for row in rows] == [1, 2, 3, 4, 5]
        assert [row.values["t_utc_ns"] for row in rows] == [T0 + n for n in range(5)]
        assert [row.values["message"] for row in rows] == [f"event {n}" for n in range(5)]
        # InfluxDB identifies a point by its measurement, tags, and time: five different points.
        points = {
            (r.values["station_id"], r.values["profile_id"], r.values["t_utc_ns"]) for r in rows
        }
        assert len(points) == 5
        assert [len(batch) for _, batch in sink.sent] == [2, 2, 1]

    def test_events_that_collide_with_an_earlier_run_reach_a_fake_sink_exactly_once(
        self, tmp_path: Path, clock: VirtualClock
    ) -> None:
        path = tmp_path / "results.sqlite"
        with Store.open(path) as first_run:
            first_run.write_many([make_event(T0, message=f"old {n}") for n in range(3)])
        sink = FakeSink(record_types={"event"})
        with Store.open(path) as second_run:  # a restart: the new events share the old times
            second_run.write_many([make_event(T0 + n, message=f"new {n}") for n in range(4)])
            forwarder = Forwarder(second_run, [sink], clock, CONFIG)
            drain(forwarder, clock)
        rows = sink.rows("event")
        assert [row.row_id for row in rows] == list(range(1, 8))
        times = [row.values["t_utc_ns"] for row in rows]
        assert len(set(times)) == 7  # seven distinct points
        assert times == sorted(times)
        assert [row.values["message"] for row in rows] == [
            "old 0",
            "old 1",
            "old 2",
            "new 0",
            "new 1",
            "new 2",
            "new 3",
        ]
