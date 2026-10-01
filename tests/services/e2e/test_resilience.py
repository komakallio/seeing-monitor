"""What core does when the world fails: a sink outage of days, a full disk, and a wrong clock.

Each test runs the whole composition root (`CoreApp`) on a virtual clock with a real store, so that
days pass in seconds and the outcome does not depend on how fast the machine is. The camera and the
analyzers are the fakes of `seeingmon.testing`: these tests are about records, events, and flags,
and not about the numbers of the analysis (see `test_night.py` for those).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")

from seeingmon.clock import NS_PER_S, ClockStatus
from seeingmon.scheduler.commands import QueueBurst
from seeingmon.sinks.base import SinkError, StoredRow
from seeingmon.store.db import StoreReader
from seeingmon.store.retention import DiskUsage
from seeingmon.testing import FakeSink

from ..core.rig import NIGHT, CoreRig, build_rig, events_of, read_all

DAY_S = 86_400
GB = 1024**3


class OutageSink(FakeSink):
    """A sink that refuses every batch while it is down, as a remote database does in an outage."""

    def __init__(self) -> None:
        super().__init__("remote", max_batch_rows=500)
        self.down = False
        self.refused = 0

    def send(self, record_type: str, rows: Sequence[StoredRow]) -> None:
        if self.down:
            self.refused += 1
            raise SinkError("the database is unreachable", retryable=True)
        super().send(record_type, rows)


def advance(rig: CoreRig, seconds: float, *, every_s: float = 600.0) -> None:
    """Move the clock forward in steps and do the periodic work at each, as a quiet core does."""
    steps = max(1, round(seconds / every_s))
    for _ in range(steps):
        rig.clock.advance(seconds / steps)
        rig.app.tick()


def run_until(rig: CoreRig, condition: Callable[[], bool], *, limit_s: float = 900.0) -> None:
    """Step the scheduler in slices of 30 s until the condition holds, or fail after `limit_s`."""
    waited = 0.0
    while not condition():
        assert waited < limit_s, "the condition did not hold in time"
        rig.run_for(30.0)
        waited += 30.0


def stored(rig: CoreRig, record_type: str) -> list[int]:
    assert rig.app.storage is not None
    reader: StoreReader = rig.app.storage.store
    rows: list[int] = []
    after = 0
    while True:
        batch = reader.after(record_type, after, 1000)
        if not batch:
            return rows
        rows += [row.row_id for row in batch]
        after = batch[-1].row_id


class TestASinkOutageOfDays:
    CONFIG = (
        "[services.core]\nhealth_interval_s = 600.0\n"
        "[store.forwarder]\nbacklog_flag_rows = 100\n"
        "backoff_initial_s = 5.0\nbackoff_max_s = 600.0\n"
    )

    def test_the_backlog_grows_in_the_store_and_drains_in_order_when_the_sink_returns(
        self, tmp_path: Path
    ) -> None:
        sink = OutageSink()
        rig = build_rig(tmp_path, config_extra=self.CONFIG, parts={"sinks": [sink]})
        rig.app.start()
        try:
            rig.run_for(30.0)  # some results first, which the sink takes
            assert sink.rows("health")  # the sink works at the start
            sink.down = True
            before = len(stored(rig, "health"))
            advance(rig, 3 * DAY_S)  # three days without a sink
            during = stored(rig, "health")
            assert len(during) - before >= 3 * DAY_S // 600 - 2  # a record each ten minutes
            backlog = rig.app.storage.health_fields()["sink_backlog"]["remote"]  # type: ignore[union-attr]
            assert backlog > 400
            assert sink.refused > 0
            flagged = [r for r in rig.records("health") if "sink_backlog" in r.flags]  # type: ignore[attr-defined]
            assert flagged  # the health record says so
            assert len(events_of(rig.records("event"), "sink.unavailable")) == 1  # once
            sink.down = False
            advance(rig, 1800.0, every_s=60.0)  # the next attempt comes within ten minutes
            assert rig.app.storage.health_fields()["sink_backlog"]["remote"] == 0  # type: ignore[union-attr]
            assert len(events_of(rig.records("event"), "sink.recovered")) == 1
            # Every row of every table reached the sink once and in order: nothing was lost.
            for record_type in ("health", "event", "run"):
                delivered = [row.row_id for row in sink.rows(record_type)]
                assert delivered == stored(rig, record_type), record_type
        finally:
            rig.app.stop()

    def test_a_restart_during_the_outage_keeps_the_backlog(self, tmp_path: Path) -> None:
        sink = OutageSink()
        first = build_rig(tmp_path, config_extra=self.CONFIG, parts={"sinks": [sink]})
        first.app.start()
        sink.down = True
        advance(first, DAY_S)
        written = stored(first, "health")
        first.app.stop("restart")
        # The next start finds the cursors in the store, and the sink returns.
        sink.down = False
        second = build_rig(tmp_path, config_extra=self.CONFIG, parts={"sinks": [sink]})
        second.app.start()
        try:
            advance(second, 1800.0, every_s=60.0)
            delivered = [row.row_id for row in sink.rows("health")]
            assert delivered[: len(written)] == written  # the backlog of the first run, in order
            assert delivered == stored(second, "health")
            assert len(delivered) == len(set(delivered))  # no row went twice
        finally:
            second.app.stop()


class FakeDisk:
    """A disk probe whose free space the test sets."""

    def __init__(self, free_gb: float) -> None:
        self.free_gb = free_gb

    def __call__(self, path: Path) -> DiskUsage:
        return DiskUsage(total_bytes=500 * GB, free_bytes=round(self.free_gb * GB))


class TestAFullDisk:
    def burst(self, rig: CoreRig, *, seconds: float = 2.0) -> Any:
        before = len(rig.app.scheduler.results())
        assert rig.app.scheduler.submit(QueueBurst(duration_s=seconds, label="t")).accepted
        for _ in range(2000):
            rig.app.scheduler.step()
            if len(rig.app.scheduler.results()) > before:
                break
        return rig.app.scheduler.results()[-1]

    def test_capture_stops_while_the_results_go_on_and_the_events_say_so(
        self, tmp_path: Path
    ) -> None:
        disk = FakeDisk(free_gb=300.0)
        rig = build_rig(tmp_path, parts={"disk_usage": disk})
        rig.app.start()
        try:
            rig.run_for(20.0)
            assert rig.app.storage.capture_allowed()  # type: ignore[union-attr]
            assert self.burst(rig).status == "ok"
            windows_before = len(rig.records("seeing_window"))

            disk.free_gb = 0.2  # the partition fills up (the limit is 1 GB)
            advance(rig, 3700.0, every_s=60.0)  # the hourly retention pass notices
            assert not rig.app.storage.capture_allowed()  # type: ignore[union-attr]
            assert events_of(rig.records("event"), "retention.capture_stopped")
            refused = self.burst(rig)
            assert refused.status != "ok"  # no raw frames go to a full disk
            assert "capture" in refused.summary.lower() or "space" in refused.summary.lower()
            rig.run_for(30.0)
            assert len(rig.records("seeing_window")) > windows_before  # the results go on
            health = rig.records("health")[-1]
            assert health.free_space_gb is not None  # type: ignore[attr-defined]
            assert health.free_space_gb < 1.0  # type: ignore[attr-defined]
            assert "low_space" in health.flags  # type: ignore[attr-defined]

            disk.free_gb = 300.0  # an operator makes room
            advance(rig, 3700.0, every_s=60.0)
            assert rig.app.storage.capture_allowed()  # type: ignore[union-attr]
            assert events_of(rig.records("event"), "retention.capture_resumed")
            assert self.burst(rig).status == "ok"
        finally:
            rig.app.stop()

    def test_the_capture_does_not_flap_inside_the_resume_margin(self, tmp_path: Path) -> None:
        disk = FakeDisk(free_gb=0.2)
        rig = build_rig(tmp_path, parts={"disk_usage": disk})
        rig.app.start()
        try:
            advance(rig, 3700.0, every_s=60.0)
            assert not rig.app.storage.capture_allowed()  # type: ignore[union-attr]
            disk.free_gb = 1.2  # above the limit (1 GB) but inside the margin (0.5 GB)
            advance(rig, 3700.0, every_s=60.0)
            assert not rig.app.storage.capture_allowed()  # type: ignore[union-attr]
            disk.free_gb = 1.6
            advance(rig, 3700.0, every_s=60.0)
            assert rig.app.storage.capture_allowed()  # type: ignore[union-attr]
            stops = events_of(rig.records("event"), "retention.capture_stopped")
            resumes = events_of(rig.records("event"), "retention.capture_resumed")
            assert (len(stops), len(resumes)) == (1, 1)
        finally:
            rig.app.stop()


class TestAWrongClock:
    def test_windows_and_health_say_time_invalid_while_the_clock_is_not_synchronized(
        self, tmp_path: Path
    ) -> None:
        rig = build_rig(tmp_path)
        rig.app.start()
        try:
            rig.run_for(45.0)
            good = rig.records("seeing_window")
            assert good
            assert all("time_invalid" not in w.flags for w in good)  # type: ignore[attr-defined]

            rig.clock.set_status(ClockStatus(False, 86_400 * NS_PER_S, "test"))  # sync is lost
            run_until(rig, lambda: len(rig.records("seeing_window")) >= len(good) + 2)
            marked = [
                w
                for w in rig.records("seeing_window")[len(good) :]
                if "time_invalid" in w.flags  # type: ignore[attr-defined]
            ]
            assert marked  # the windows that close after the loss carry the flag
            assert "time_invalid" in rig.records("health")[-1].flags  # type: ignore[attr-defined]
            assert rig.records("health")[-1].time_synchronized is False  # type: ignore[attr-defined]

            rig.clock.set_status(ClockStatus(True, 1_000_000, "test"))  # synchronized again
            seen = len(rig.records("seeing_window"))
            run_until(rig, lambda: len(rig.records("seeing_window")) >= seen + 2)
            later = rig.records("seeing_window")[seen:]
            assert "time_invalid" not in later[-1].flags  # type: ignore[attr-defined]
            assert "time_invalid" not in rig.records("health")[-1].flags  # type: ignore[attr-defined]
        finally:
            rig.app.stop()

    def test_a_step_of_the_utc_clock_never_makes_the_windows_go_back_in_time(
        self, tmp_path: Path
    ) -> None:
        rig = build_rig(tmp_path)
        rig.app.start()
        try:
            rig.run_for(30.0)
            rig.clock.step_utc_ns(-5 * NS_PER_S)  # the time source corrects the clock backward
            rig.run_for(45.0)
            times = [w.t_utc_ns for w in rig.records("seeing_window")]
            assert len(times) >= 2
            # Windows come in the order of their start. A step leaves them in that order, or the
            # store would show a window that began before the one that came earlier.
            assert times == sorted(times)
        finally:
            rig.app.stop()

    def test_a_core_that_starts_with_an_unsynchronized_clock_says_so_from_the_first_record(
        self, tmp_path: Path
    ) -> None:
        rig = build_rig(tmp_path)
        rig.clock.set_status(ClockStatus(False, 86_400 * NS_PER_S, "test"))
        rig.app.start()
        try:
            rig.app.tick()
            first = rig.records("health")[0]
            assert "time_invalid" in first.flags  # type: ignore[attr-defined]
            assert first.t_utc_ns >= NIGHT
        finally:
            rig.app.stop()


def reader_of(rig: CoreRig) -> list[Any]:
    """Every `event` record that the rig stored, for the failure message of a test."""
    assert rig.app.storage is not None
    return read_all(rig.app.storage.store, "event")
