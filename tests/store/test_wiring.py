"""`open_storage` wires the layout, the store, the segments, the forwarder, and retention."""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from seeingmon.clock import VirtualClock
from seeingmon.config import Config, load_config
from seeingmon.records.samples import sample_record
from seeingmon.records.system import HealthRecord
from seeingmon.sinks.config import SinksSection
from seeingmon.store.db import Store, StoreReader
from seeingmon.store.segments import SegmentReader
from seeingmon.store.wiring import Storage, open_storage
from seeingmon.testing.fakes import FakeSink
from tests.sinks.helpers import OutageSink
from tests.store.builders import NS_PER_S, T0, make_health, make_rows, make_window
from tests.store.disk import FakeDisk

DAY_S = 86_400


def make_config(tmp_path: Path, extra: str = "") -> Config:
    local = tmp_path / "config.toml"
    local.write_text(
        'station_id = "test-station"\n'
        f'[paths]\ndata_dir = "{(tmp_path / "data").as_posix()}"\n' + extra,
        encoding="utf-8",
    )
    return load_config(local_file=local, env={})


@pytest.fixture
def clock() -> VirtualClock:
    return VirtualClock(T0)


@pytest.fixture
def storage(tmp_path: Path, clock: VirtualClock) -> Iterator[Storage]:
    opened = open_storage(make_config(tmp_path), clock, sinks=[])
    try:
        yield opened
    finally:
        opened.close()


class TestOpening:
    def test_it_creates_the_folders_and_opens_every_part(
        self, storage: Storage, tmp_path: Path
    ) -> None:
        data = tmp_path / "data"
        assert (data / "db" / "results.sqlite").is_file()
        for folder in ("segments", "survey", "previews", "bursts"):
            assert (data / folder).is_dir()
        assert storage.layout.root == data
        assert storage.store.count("health") == 0
        assert storage.config.retention.quota_fraction == 0.25

    def test_the_parts_share_the_identity_of_the_station(
        self, storage: Storage, clock: VirtualClock
    ) -> None:
        storage.events.emit("info", "test.started", "The test started.")
        (event,) = storage.store.after("event", 0)
        assert event.values["station_id"] == "test-station"
        assert event.values["profile_id"] == "asi294mm-gs250"

    def test_the_store_serves_as_the_record_writer_and_the_segments_as_the_metrics_writer(
        self, storage: Storage
    ) -> None:
        storage.store.as_record_writer().write(make_window(T0))
        storage.segments.write_metrics(1, make_rows(T0, 5))
        storage.segments.close()
        assert storage.store.count("seeing_window") == 1
        reader = SegmentReader(storage.layout.segments_dir)
        assert len(reader.read_range(T0, T0 + 10 * NS_PER_S)) == 5

    def test_the_sinks_come_from_the_configuration_by_default(
        self, tmp_path: Path, clock: VirtualClock
    ) -> None:
        extra = (
            '[sinks.lab]\nkind = "influx"\nendpoint = "https://influx.example.org"\n'
            'org = "o"\nbucket = "b"\nenabled = false\n'
            '[sinks.archive]\nkind = "influx"\nendpoint = "https://influx.example.org"\n'
            'org = "o"\nbucket = "b"\n'
        )
        config = make_config(tmp_path, extra)
        assert set(config.section("sinks", SinksSection).root) == {"lab", "archive"}
        opened = open_storage(config, clock)
        try:
            assert list(opened.forwarder.sink_backlog()) == ["archive"]  # `lab` is disabled
        finally:
            opened.close()

    def test_a_failure_while_opening_closes_the_store(
        self, tmp_path: Path, clock: VirtualClock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("no segments today")

        monkeypatch.setattr("seeingmon.store.wiring.SegmentWriter", explode)
        with pytest.raises(RuntimeError, match="no segments"):
            open_storage(make_config(tmp_path), clock, sinks=[])
        monkeypatch.undo()
        with Store.open(tmp_path / "data" / "db" / "results.sqlite") as store:  # the file is free
            assert store.count("event") == 0

    def test_closing_twice_is_safe(self, tmp_path: Path, clock: VirtualClock) -> None:
        opened = open_storage(make_config(tmp_path), clock, sinks=[])
        opened.close()
        opened.close()


class TestRecovery:
    def test_segment_files_that_a_crash_left_are_repaired_and_reported(
        self, tmp_path: Path, clock: VirtualClock
    ) -> None:
        first = open_storage(make_config(tmp_path), clock, sinks=[])
        first.segments.write_metrics(1, make_rows(T0, 30))
        first.segments.flush()
        part = first.segments.open_path
        assert part is not None
        copy = tmp_path / "saved.part"
        shutil.copyfile(part, copy)
        first.close()
        # Put the unfinished file back, as a crash would leave it, with a torn last row.
        final = part.with_name(part.name.removesuffix(".part"))
        final.unlink()
        shutil.copyfile(copy, part)
        with part.open("r+b") as handle:
            handle.truncate(part.stat().st_size - 10)

        second = open_storage(make_config(tmp_path), clock, sinks=[])
        try:
            assert final.is_file()
            assert not part.exists()
            (event,) = [
                r for r in second.store.after("event", 0) if r.values["kind"] == "storage.recovered"
            ]
            assert event.values["level"] == "warning"
            assert event.values["detail"]["files"] == 1
            assert event.values["detail"]["rows"] == 29
            assert (
                len(SegmentReader(second.layout.segments_dir).read_range(T0, T0 + 99 * NS_PER_S))
                == 29
            )
        finally:
            second.close()

    def test_a_clean_start_writes_no_recovery_event(self, storage: Storage) -> None:
        assert storage.store.count("event") == 0


class TestHealth:
    def test_the_storage_fields_fit_a_health_record(
        self, tmp_path: Path, clock: VirtualClock
    ) -> None:
        sink = OutageSink("influx", clock)
        opened = open_storage(make_config(tmp_path), clock, sinks=[sink])
        try:
            opened.store.write_many([make_health(T0 + n * NS_PER_S) for n in range(4)])
            fields = opened.health_fields()
            assert fields["sink_backlog"] == {"influx": 4}
            assert fields["flags"] == []
            assert fields["data_used_gb"] is None  # no retention pass has run yet
            assert fields["free_space_gb"] > 0
            record = sample_record(HealthRecord, **fields)
            assert record.sink_backlog == {"influx": 4}
            opened.retention.run_once()
            assert opened.health_fields()["data_used_gb"] is not None
        finally:
            opened.close()

    def test_the_flags_name_low_space_and_a_long_backlog(
        self, tmp_path: Path, clock: VirtualClock
    ) -> None:
        extra = "[store.forwarder]\nbacklog_flag_rows = 3\n"
        config = make_config(tmp_path, extra)
        disk = FakeDisk(tmp_path / "data", 10 * 10**9, other_used=int(9.5 * 10**9))
        opened = open_storage(config, clock, sinks=[OutageSink("influx", clock)], disk_usage=disk)
        try:
            opened.store.write_many([make_health(T0 + n * NS_PER_S) for n in range(5)])
            fields = opened.health_fields()
            assert fields["flags"] == ["low_space", "sink_backlog"]
            assert opened.capture_allowed() is False
            record = sample_record(HealthRecord, **fields)
            assert record.flags == ["low_space", "sink_backlog"]
        finally:
            opened.close()


class TestHousekeeping:
    def test_the_loop_forwards_closes_idle_segments_and_runs_retention_on_schedule(
        self, tmp_path: Path, clock: VirtualClock
    ) -> None:
        extra = "[store.retention]\ninterval_s = 100\n[store.segments]\nidle_close_s = 30\n"
        sink = FakeSink("everything")
        opened = open_storage(make_config(tmp_path, extra), clock, sinks=[sink])
        try:
            opened.store.write_many([make_health(T0 + n * NS_PER_S) for n in range(10)])
            opened.segments.write_metrics(1, make_rows(T0, 5))
            passes = 0
            runs = {"retention": 0}
            real_run = opened.retention.run_once

            def counted() -> Any:
                runs["retention"] += 1
                return real_run()

            opened.retention.run_once = counted  # type: ignore[method-assign]
            passes = opened.run_housekeeping(lambda: clock.utc_ns() >= T0 + 250 * NS_PER_S)
            assert passes > 100
            assert [row.row_id for row in sink.rows("health")] == list(range(1, 11))
            assert opened.segments.open_path is None  # the idle segment closed
            assert runs["retention"] == 3  # at the start, at 100 s, and at 200 s
        finally:
            opened.close()

    def test_a_failing_step_does_not_stop_the_loop(
        self, tmp_path: Path, clock: VirtualClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        opened = open_storage(make_config(tmp_path), clock, sinks=[FakeSink("a")])
        try:

            def fail() -> Any:
                raise OSError("the disk is gone")

            opened.retention.run_once = fail  # type: ignore[method-assign]
            with caplog.at_level("ERROR"):
                assert opened.run_housekeeping(lambda: False, max_passes=3) == 3
            assert "the retention pass failed" in caplog.text
        finally:
            opened.close()


def restamp(root: Path, now_ns: int) -> None:
    """Give the files that a test just wrote the virtual time, as retention reads file times."""
    for path in root.rglob("*"):
        if path.is_file() and path.stat().st_mtime_ns > now_ns:
            os.utime(path, ns=(now_ns, now_ns))


class TestFourDaysInVirtualTime:
    """A component test: the storage side of `core` through an outage, a full disk, and a step."""

    def test_nothing_is_lost_or_repeated_and_the_limits_hold(
        self, tmp_path: Path, clock: VirtualClock
    ) -> None:
        extra = (
            "[store.retention]\nmetrics_days = 1\nmin_free_gb = 0.001\nresume_margin_gb = 0.0005\n"
            "[store.segments]\nsegment_s = 3600\n"  # one file for each hour keeps the test fast
            "[store.forwarder]\nbackoff_initial_s = 30\n"
        )
        config = make_config(tmp_path, extra)
        disk = FakeDisk(tmp_path / "data", 10**9)  # a partition of 1 GB, almost all free
        sink = OutageSink("influx", clock, max_batch_rows=500)
        opened = open_storage(config, clock, sinks=[sink], disk_usage=disk)
        written = {"health": 0, "seeing_window": 0}
        try:
            for hour in range(4 * 24):
                t_ns = T0 + hour * 3600 * NS_PER_S
                opened.store.write_many([make_health(t_ns + m * 60 * NS_PER_S) for m in range(60)])
                written["health"] += 60
                opened.store.write(make_window(t_ns))
                written["seeing_window"] += 1
                opened.segments.write_metrics(1, make_rows(t_ns, 360, step_ns=10 * NS_PER_S))
                restamp(opened.layout.segments_dir, clock.utc_ns())
                sink.up = not (24 <= hour < 24 * 2)  # the sink is down for a day
                disk.other_used = 10**9 if 24 * 3 <= hour < 24 * 3 + 6 else 0  # a full disk
                if hour == 24 * 3 + 12:
                    clock.step_utc_ns(-90 * NS_PER_S)  # NTP steps the wall clock back
                clock.advance(3600)
                opened.run_housekeeping(lambda: False, max_passes=1)
            sink.up = True
            for _ in range(2000):  # let the forwarder finish its backlog
                clock.advance(60)
                opened.run_housekeeping(lambda: False, max_passes=1)
                if not any(opened.forwarder.sink_backlog().values()):
                    break

            # Nothing lost, nothing repeated, and in order, whatever the outage did.
            for record_type in ("health", "seeing_window"):
                assert sink.received[record_type] == list(range(1, written[record_type] + 1))
            assert sink.repeated == 0
            assert opened.forwarder.sink_backlog() == {"influx": 0}

            # The events tell the story.
            kinds = [row.values["kind"] for row in opened.store.after("event", 0, 10_000)]
            assert kinds.count("sink.unavailable") == 1
            assert kinds.count("sink.recovered") == 1
            assert kinds.index("sink.unavailable") < kinds.index("sink.recovered")
            assert "retention.early_delete" in kinds  # the full disk forced deletions
            assert kinds.count("retention.capture_stopped") == 1
            assert kinds.count("retention.capture_resumed") == 1
            stopped = kinds.index("retention.capture_stopped")
            assert stopped < kinds.index("retention.capture_resumed")
            assert sink.received["event"] == list(range(1, len(kinds) + 1))

            # The age limit held: a day of hourly segments at most, and not the whole four.
            files = list(opened.layout.segments_dir.rglob("*.seg*"))
            assert 0 < len(files) <= 24 + 3
            assert opened.capture_allowed() is True
        finally:
            opened.close()
        with StoreReader.open(tmp_path / "data" / "db" / "results.sqlite") as reader:
            assert reader.count("health") == written["health"]
            assert reader.count("seeing_window") == written["seeing_window"]
