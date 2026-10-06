"""The nightly visibility summary of `core`: read back from a real store, and written once.

The tests write the rows of a night into a store, as the scheduler and `core` would, and move a
virtual clock past the split hour. `tests/visibility/test_summary.py` checks the rules of the
summary itself.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from seeingmon.clock import NS_PER_S, VirtualClock, iso_to_utc_ns
from seeingmon.records import EventRecord, HealthRecord, SeeingWindowRecord, sample_record
from seeingmon.records.visibility import VisibilitySummaryRecord
from seeingmon.scheduler.config import SiteConfig
from seeingmon.scheduler.ephemeris import sun_elevation_deg
from seeingmon.services.core import visibility as module
from seeingmon.services.core.settings import SkyFlagSettings
from seeingmon.services.core.skyflags import moon_flag
from seeingmon.services.core.visibility import VisibilitySummary
from seeingmon.store.db import Store
from seeingmon.survey.config import VisibilityConfig
from seeingmon.visibility.summary import HIDDEN_EVENT, VISIBLE_EVENT

from .rig import read_all

SITE = SiteConfig(latitude_deg=55.0, longitude_deg=0.0)  # the synthetic site of the simulator
START = iso_to_utc_ns("2026-01-10T12:00:00Z")  # the night 2026-01-10, with the split at 12:00
END = iso_to_utc_ns("2026-01-11T12:00:00Z")
MINUTE = 60 * NS_PER_S
DAY = 24 * 60 * MINUTE
DELAY_S = 60.0  # one analysis window
# A health record every 10 minutes keeps the store small. It vouches for three intervals.
HEALTH_INTERVAL_S = 600.0
HEALTH_STEP = round(HEALTH_INTERVAL_S * NS_PER_S)


def at(text: str) -> int:
    return iso_to_utc_ns(f"2026-01-{text}:00Z")


def event(t_ns: int, kind: str, **detail: Any) -> EventRecord:
    return sample_record(
        EventRecord,
        station_id="test",
        t_utc_ns=t_ns,
        provenance={"scheduler": "test"},
        level="info",
        kind=kind,
        message=kind,
        detail=detail,
    )


def health(
    t_ns: int,
    state: str = "auto",
    *,
    synchronized: bool = True,
    components: dict[str, str] | None = None,
) -> HealthRecord:
    """A health record with the components that `core` writes, all `ok` unless `components`
    says otherwise, and `degraded` as `HealthReporter` sets it."""
    parts = {"core": "ok", "scheduler": "ok", "camera": "ok", **(components or {})}
    return sample_record(
        HealthRecord,
        station_id="test",
        t_utc_ns=t_ns,
        provenance={"software": "test"},
        state=state,
        degraded=parts["scheduler"] == "degraded" or "failed" in parts.values(),
        components=parts,
        time_synchronized=synchronized,
    )


def window(
    t_ns: int, r0_cm: float | None = 10.0, flags: list[str] | None = None
) -> SeeingWindowRecord:
    return sample_record(
        SeeingWindowRecord,
        station_id="test",
        t_utc_ns=t_ns,
        provenance={"fastpath": "test"},
        duration_s=60.0,
        r0_cm=r0_cm,
        flags=flags or [],
    )


class Station:
    """A store, a clock, and the summary of `core` on them."""

    def __init__(self, folder: Path) -> None:
        self.path = folder / "db.sqlite"
        self.store = Store.open(self.path)
        self.clock = VirtualClock(START)

    def summary(
        self, *, site: SiteConfig | None = SITE, settings: VisibilityConfig | None = None
    ) -> VisibilitySummary:
        return VisibilitySummary(
            self.store,
            clock=self.clock,
            station_id="test",
            profile_id="test",
            split_utc_hour=12.0,
            settings=settings or VisibilityConfig(),
            health_interval_s=HEALTH_INTERVAL_S,
            delay_s=DELAY_S,
            site=site,
            sky_flags=SkyFlagSettings(),
        )

    def write_health(self, start_ns: int, end_ns: int, state: str = "auto", **more: Any) -> None:
        self.store.write_many(
            health(t, state, **more) for t in range(start_ns, end_ns, HEALTH_STEP)
        )

    def write_clear_night(self, start_ns: int = START, *, with_health: bool = True) -> None:
        """A station that ran all day and night: Polaris from 16:00 to 07:00, seeing for an hour.

        `start_ns` moves the night by whole days. The health records start one day before it.
        """
        shift = start_ns - START
        if with_health:
            self.write_health(start_ns - DAY, start_ns + DAY + 10 * MINUTE)
        self.store.write(event(at("10T16:00") + shift, VISIBLE_EVENT, sun_elevation_deg=-4.8))
        self.store.write(event(at("11T07:00") + shift, HIDDEN_EVENT, sun_elevation_deg=-5.2))
        self.store.write_many(
            window(t + shift) for t in range(at("10T20:00"), at("10T21:00"), MINUTE)
        )
        self.store.write(window(at("10T21:00") + shift, r0_cm=None))  # a window without a value

    def summaries(self) -> list[VisibilitySummaryRecord]:
        found = read_all(self.store, "visibility_summary")
        return [r for r in found if isinstance(r, VisibilitySummaryRecord)]


def drain(summary: VisibilitySummary) -> list[VisibilitySummaryRecord]:
    """Call `write_due` until no night waits, as the task does every 30 s. Returns the records."""
    written: list[VisibilitySummaryRecord] = []
    while True:
        record = summary.write_due()
        if record is not None:
            written.append(record)
        if not summary.pending:
            return written


def newest(summary: VisibilitySummary) -> VisibilitySummaryRecord | None:
    """The newest record that the calls of `drain` write, or `None`."""
    written = drain(summary)
    return written[-1] if written else None


@pytest.fixture
def station(tmp_path: Path) -> Iterator[Station]:
    opened = Station(tmp_path)
    try:
        yield opened
    finally:
        opened.store.close()


class TestTheNightEnds:
    def test_the_summary_waits_for_the_split_hour_and_one_window(self, station: Station) -> None:
        station.write_clear_night()
        summary = station.summary()
        station.clock.advance_to_utc_ns(END + 30 * NS_PER_S)
        summary.write_due()  # the window that began before the split may still be open
        assert [r.night for r in station.summaries()] == ["2026-01-09"]  # the night before
        station.clock.advance_to_utc_ns(END + 61 * NS_PER_S)
        record = summary.write_due()
        assert record is not None
        assert station.summaries()[-1] == record
        assert record.night == "2026-01-10"
        assert record.t_utc_ns == START  # the fixed key of the night
        assert record.first_visible_utc_ns == at("10T16:00")
        assert record.first_visible_sun_deg == -4.8
        assert record.last_visible_utc_ns == at("11T07:00")
        assert record.last_visible_sun_deg == -5.2
        assert record.visible_hours == pytest.approx(15.0, abs=1e-6)  # rounded to 1e-6 h
        assert record.seeing_hours == pytest.approx(1.0, abs=1e-6)  # the window without r0 adds 0
        assert (record.first_censored, record.last_censored) == (False, False)
        assert record.provenance["software"]
        assert summary.written == 2

    def test_a_night_is_written_once_also_across_a_restart(
        self, station: Station, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        station.write_clear_night()
        station.clock.advance_to_utc_ns(END + 2 * MINUTE)
        first = station.summary()
        assert [r.night for r in drain(first)] == ["2026-01-09", "2026-01-10"]
        assert first.write_due() is None  # the same process asks again
        again = station.summary()  # a restart of core
        reads: list[str] = []
        rows = module.VisibilitySummary._rows

        def counted(self: VisibilitySummary, record_type: str, *args: Any) -> Any:
            reads.append(record_type)
            return rows(self, record_type, *args)

        monkeypatch.setattr(module.VisibilitySummary, "_rows", counted)
        with caplog.at_level(logging.WARNING, logger=module.__name__):
            assert again.write_due() is None
        assert again.pending == ()  # one call checked the whole week
        assert caplog.records == []  # a night that has its summary is no error
        assert reads == []  # and it costs no read of the night's rows
        assert [r.night for r in station.summaries()] == ["2026-01-09", "2026-01-10"]

    def test_a_record_that_the_store_refuses_as_a_duplicate_is_no_error(
        self, station: Station, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        station.write_clear_night()
        station.clock.advance_to_utc_ns(END + 2 * MINUTE)
        assert len(drain(station.summary())) == 2
        again = station.summary()
        # Another writer stored the records between the check and the write.
        monkeypatch.setattr(again, "_has_summary", lambda start_ns: False)
        with caplog.at_level(logging.WARNING, logger=module.__name__):
            assert drain(again) == []  # the store raised DuplicateRecordError twice
        assert caplog.records == []
        assert again.written == 0
        assert len(station.summaries()) == 2

    def test_a_start_after_a_missed_split_hour_writes_the_night_that_ended(
        self, station: Station
    ) -> None:
        station.write_clear_night()
        station.clock.advance_to_utc_ns(at("11T18:00"))  # core was down from noon to the evening
        record = newest(station.summary())
        assert record is not None
        assert record.night == "2026-01-10"

    def test_a_downtime_across_two_split_hours_still_writes_the_night_with_data(
        self, station: Station
    ) -> None:
        # core stopped five minutes before the split hour of 11 January and started again an hour
        # after the split hour of 12 January, so the night 2026-01-11 has no health record.
        station.write_clear_night(with_health=False)
        station.write_health(START - DAY, at("11T11:55"))
        station.clock.advance_to_utc_ns(at("12T13:00"))
        record = newest(station.summary())
        assert record is not None
        assert record.night == "2026-01-10"
        assert record.last_visible_utc_ns == at("11T07:00")
        assert record.last_censored is False
        assert [r.night for r in station.summaries()] == ["2026-01-09", "2026-01-10"]

    def test_the_catch_up_goes_oldest_first_so_each_night_finds_the_one_before(
        self, station: Station
    ) -> None:
        # Polaris showed at 16:00 on 9 January and stayed visible until 20:00 on 11 January, so the
        # day before the night 2026-01-11 holds no polaris event, and the summary of 2026-01-10
        # must tell that Polaris was visible when that night began.
        station.write_health(at("08T12:00"), at("12T12:10"))
        station.store.write(event(at("09T16:00"), VISIBLE_EVENT, sun_elevation_deg=-4.8))
        station.store.write(event(at("11T20:00"), HIDDEN_EVENT, sun_elevation_deg=-30.0))
        station.clock.advance_to_utc_ns(at("12T13:00"))
        summary = station.summary()
        record = summary.write_due()
        assert record is not None
        assert record.night == "2026-01-08"  # one night a call, the oldest first
        assert summary.pending == ("2026-01-09", "2026-01-10", "2026-01-11")
        written = [record, *drain(summary)]
        assert [r.night for r in written] == [
            "2026-01-08",
            "2026-01-09",
            "2026-01-10",
            "2026-01-11",
        ]
        assert written[2].last_visible_utc_ns == END  # still visible at the end of 2026-01-10
        last = written[-1]
        assert last.first_visible_utc_ns == END  # the start of the night: a bound
        assert last.first_censored is True
        assert last.visible_hours == pytest.approx(8.0, abs=1e-6)  # 12:00 to 20:00

    def test_the_nights_due_are_the_last_week_at_first_and_then_the_ones_that_ended(
        self, station: Station
    ) -> None:
        station.clock.advance_to_utc_ns(END + 2 * MINUTE)
        summary = station.summary()
        due = summary.due_nights()
        assert due == [f"2026-01-{day:02d}" for day in range(4, 11)]  # catch_up_nights = 7
        summary.write_due()
        assert summary.due_nights() == []
        station.clock.advance_to_utc_ns(END + 3 * DAY + 2 * MINUTE)  # no call for three days
        assert summary.due_nights() == ["2026-01-11", "2026-01-12", "2026-01-13"]
        one = station.summary(settings=VisibilityConfig(catch_up_nights=1))
        assert one.due_nights() == ["2026-01-13"]

    def test_a_night_in_which_core_never_ran_gets_no_record(self, station: Station) -> None:
        station.clock.advance_to_utc_ns(END + 2 * MINUTE)
        assert station.summary().write_due() is None
        assert station.summaries() == []

    def test_a_failure_is_logged_once_for_each_night_and_never_reaches_the_task(
        self, station: Station, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        station.write_clear_night()
        station.clock.advance_to_utc_ns(END + 2 * MINUTE)
        summary = station.summary()

        def broken(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("the disk is gone")

        monkeypatch.setattr(station.store, "range", broken)
        with caplog.at_level(logging.ERROR, logger=module.__name__):
            for _ in range(9):
                assert summary.write_due() is None
        nights = [record.getMessage().rsplit(" ", 1)[-1] for record in caplog.records]
        # One entry a call for each night of the week (catch_up_nights = 7), and none after.
        assert nights == [f"2026-01-{day:02d}" for day in range(4, 11)]
        assert summary.pending == ()


class TestTheStoreReads:
    def test_the_reads_page_through_more_rows_than_one_batch(
        self, station: Station, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(module, "_BATCH", 7)
        station.write_clear_night()
        station.clock.advance_to_utc_ns(END + 2 * MINUTE)
        record = newest(station.summary())
        assert record is not None
        assert record.seeing_hours == pytest.approx(1.0, abs=1e-6)
        assert record.last_visible_utc_ns == at("11T07:00")

    def test_the_previous_summary_tells_that_polaris_was_visible_at_the_start(
        self, station: Station
    ) -> None:
        station.write_health(START - 24 * 60 * MINUTE, END + 10 * MINUTE)
        station.store.write(
            sample_record(
                VisibilitySummaryRecord,
                station_id="test",
                t_utc_ns=START - 24 * 60 * MINUTE,
                provenance={"software": "test"},
                night="2026-01-09",
                last_visible_utc_ns=START,  # Polaris was visible when that night ended
                last_censored=True,
            )
        )
        station.store.write(event(at("10T18:00"), HIDDEN_EVENT, sun_elevation_deg=-12.0))
        station.clock.advance_to_utc_ns(END + 2 * MINUTE)
        record = newest(station.summary())
        assert record is not None
        assert record.first_visible_utc_ns == START
        assert record.first_censored is True
        assert record.first_visible_sun_deg == pytest.approx(
            sun_elevation_deg(START, 55.0, 0.0), abs=0.005
        )  # the Sun at the bound, from the site, rounded to 0.01 degree
        assert record.visible_hours == pytest.approx(6.0, abs=1e-6)

    def test_the_moon_flag_follows_the_moon_at_the_detections(self, station: Station) -> None:
        station.write_clear_night()
        station.clock.advance_to_utc_ns(END + 2 * MINUTE)
        record = newest(station.summary())
        assert record is not None
        settings = SkyFlagSettings()
        moon = moon_flag(at("10T16:00"), SITE, settings) or moon_flag(
            at("11T07:00"), SITE, settings
        )
        assert ("moon" in record.flags) == moon

    def test_without_a_site_the_record_explains_the_moon_flag(self, station: Station) -> None:
        station.write_clear_night()
        station.clock.advance_to_utc_ns(END + 2 * MINUTE)
        record = newest(station.summary(site=None))
        assert record is not None
        assert record.quality is not None
        assert record.quality["flags"].startswith("no site")
        assert record.first_visible_sun_deg == -4.8  # the event still carries its own

    @pytest.mark.parametrize(
        ("state", "components", "censored"),
        [
            ("auto", None, False),
            ("safe", None, False),  # too bright to measure, but the station watches
            ("paused", None, True),
            ("align", None, True),
            ("commission", None, True),
            ("auto", {"scheduler": "degraded", "camera": "failed"}, True),  # the camera failed
            ("auto", {"heater": "failed"}, False),  # the search goes on
            ("auto", {"acquire": "failed"}, False),  # one health reply that did not come
        ],
    )
    def test_the_health_rows_after_the_last_detection_decide_its_censoring(
        self,
        station: Station,
        state: str,
        components: dict[str, str] | None,
        censored: bool,
    ) -> None:
        station.write_health(START - DAY, at("11T07:00"))
        station.write_health(at("11T07:00"), END + 10 * MINUTE, state, components=components)
        station.store.write(event(at("10T16:00"), VISIBLE_EVENT, sun_elevation_deg=-4.8))
        station.store.write(event(at("11T07:00"), HIDDEN_EVENT, sun_elevation_deg=-5.2))
        station.clock.advance_to_utc_ns(END + 2 * MINUTE)
        record = newest(station.summary())
        assert record is not None
        assert record.last_visible_utc_ns == at("11T07:00")
        assert record.last_censored is censored  # five hours without a watch, against 300 s
        assert record.first_censored is False

    def test_a_health_row_with_an_unsynchronized_clock_sets_time_invalid(
        self, station: Station
    ) -> None:
        station.write_clear_night()
        station.store.write(health(at("10T20:05"), synchronized=False))
        station.clock.advance_to_utc_ns(END + 2 * MINUTE)
        record = newest(station.summary())
        assert record is not None
        assert "time_invalid" in record.flags

    def test_a_window_flagged_time_invalid_sets_time_invalid(self, station: Station) -> None:
        station.write_clear_night()
        station.store.write(window(at("10T22:00"), flags=["time_invalid"]))
        station.clock.advance_to_utc_ns(END + 2 * MINUTE)
        record = newest(station.summary())
        assert record is not None
        assert "time_invalid" in record.flags

    def test_a_clear_night_with_a_synchronized_clock_has_no_time_invalid(
        self, station: Station
    ) -> None:
        station.write_clear_night()
        station.clock.advance_to_utc_ns(END + 2 * MINUTE)
        record = newest(station.summary())
        assert record is not None
        assert "time_invalid" not in record.flags

    def test_another_station_in_the_store_stays_out(self, station: Station) -> None:
        station.write_clear_night()
        other = event(at("10T13:00"), VISIBLE_EVENT, sun_elevation_deg=20.0)
        station.store.write(other.model_copy(update={"station_id": "other"}))
        station.clock.advance_to_utc_ns(END + 2 * MINUTE)
        record = newest(station.summary())
        assert record is not None
        assert record.first_visible_utc_ns == at("10T16:00")
