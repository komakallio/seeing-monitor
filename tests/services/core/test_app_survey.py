"""`CoreApp` and the survey analysis: the sky flags, the nightly summary, and the dark library."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")

from seeingmon.analysis import SurveyOutput
from seeingmon.clock import iso_to_utc_ns
from seeingmon.records import EventRecord, Record
from seeingmon.records.survey import SkyQualityRecord
from seeingmon.store.db import Store
from seeingmon.testing import FakeSurveyAnalyzer

from .rig import CoreRig, build_rig, events_of, read_all

FULL_MOON_EVENING = iso_to_utc_ns("2026-01-02T20:00:00Z")  # a waxing gibbous Moon is up
NEW_MOON_EVENING = iso_to_utc_ns("2026-01-18T20:00:00Z")  # the Moon is new, and near the Sun
NIGHT_START = iso_to_utc_ns("2026-01-10T20:00:00Z")
NEXT_NOON = iso_to_utc_ns("2026-01-11T12:00:30Z")


class SkyQualityFake(FakeSurveyAnalyzer):
    """A survey analyzer whose results carry a `sky_quality` record, like the real one."""

    def _output(self, item: Any) -> SurveyOutput:
        output = super()._output(item)
        record = SkyQualityRecord(
            station_id="test",
            t_utc_ns=item.frame.t_utc_ns,
            profile_id="test",
            provenance={"algo": "fake"},
            n_stars_used=20,
            zero_point_mag=19.0,
        )
        return SurveyOutput(
            t_utc_ns=output.t_utc_ns,
            records=(*output.records, record),
            solved=output.solved,
            cloud_fraction=output.cloud_fraction,
        )


class NightFake(SkyQualityFake):
    """The same, with a nightly summary: a stand-in record for the open night."""

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self.open_night: list[Record] = []
        self.flushes = 0

    def flush_night(self) -> Sequence[Record]:
        self.flushes += 1
        records, self.open_night = self.open_night, []
        return records


class FakeHeater:
    """A heater controller that reports a dew point and the temperature of the air."""

    def __init__(self, dew_point_c: float | None, ambient_c: float | None) -> None:
        self.dew_point_c = dew_point_c
        self.ambient_c = ambient_c

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...

    def step(self) -> float:
        return 1.0

    def run(self, should_stop: Any) -> None: ...

    def status(self) -> Any:
        return SimpleNamespace(
            state="off",
            enabled=True,
            dew_point_c=self.dew_point_c,
            ambient_c=self.ambient_c,
            optics_c=None,
        )

    def recent_duty(self, window_s: float) -> float | None:
        return 0.0


def stored_sky_quality(rig: CoreRig) -> list[SkyQualityRecord]:
    found = rig.records("sky_quality")
    assert all(isinstance(record, SkyQualityRecord) for record in found)
    return found  # type: ignore[return-value]


def run_until_a_survey_frame_is_stored(rig: CoreRig) -> None:
    rig.app.start()
    for _ in range(40):
        rig.run_for(60.0)
        if rig.records("sky_quality"):
            return
    raise AssertionError("the scheduler took no survey frame in 40 minutes of virtual time")


class TestTheSkyFlags:
    def test_a_moon_above_the_horizon_flags_the_stored_record(self, tmp_path: Path) -> None:
        rig = build_rig(
            tmp_path,
            start_utc_ns=FULL_MOON_EVENING,
            parts={"survey": SkyQualityFake(station_id="test", profile_id="test")},
        )
        try:
            run_until_a_survey_frame_is_stored(rig)
            assert "moon" in stored_sky_quality(rig)[0].flags
        finally:
            rig.app.stop()

    def test_a_new_moon_leaves_the_record_without_the_flag(self, tmp_path: Path) -> None:
        rig = build_rig(
            tmp_path,
            start_utc_ns=NEW_MOON_EVENING,
            parts={"survey": SkyQualityFake(station_id="test", profile_id="test")},
        )
        try:
            run_until_a_survey_frame_is_stored(rig)
            assert "moon" not in stored_sky_quality(rig)[0].flags
        finally:
            rig.app.stop()

    def test_optics_at_the_dew_point_flag_the_record(self, tmp_path: Path) -> None:
        rig = build_rig(
            tmp_path,
            start_utc_ns=NEW_MOON_EVENING,
            parts={
                "survey": SkyQualityFake(station_id="test", profile_id="test"),
                "heater": FakeHeater(dew_point_c=5.4, ambient_c=6.0),
            },
        )
        try:
            run_until_a_survey_frame_is_stored(rig)
            assert "dew" in stored_sky_quality(rig)[0].flags
        finally:
            rig.app.stop()

    def test_dry_air_leaves_the_record_without_the_flag(self, tmp_path: Path) -> None:
        rig = build_rig(
            tmp_path,
            start_utc_ns=NEW_MOON_EVENING,
            parts={
                "survey": SkyQualityFake(station_id="test", profile_id="test"),
                "heater": FakeHeater(dew_point_c=-8.0, ambient_c=6.0),
            },
        )
        try:
            run_until_a_survey_frame_is_stored(rig)
            assert "dew" not in stored_sky_quality(rig)[0].flags
        finally:
            rig.app.stop()


def marker(kind: str, t_utc_ns: int) -> Record:
    return EventRecord(
        station_id="test",
        t_utc_ns=t_utc_ns,
        profile_id="test",
        provenance={"source": "local"},
        level="info",
        kind=kind,
        message="A stand-in for the star summary of a night.",
    )


class TestTheNightlySummary:
    def test_the_analyzer_is_wrapped_and_the_tracker_stays_reachable(self, tmp_path: Path) -> None:
        analyzer = NightFake(station_id="test", profile_id="test")
        rig = build_rig(tmp_path, start_utc_ns=NIGHT_START, parts={"survey": analyzer})
        try:
            assert rig.app.nightly is not None
            assert rig.app.survey is rig.app.nightly
        finally:
            rig.app.stop()

    def test_an_analyzer_without_a_summary_is_not_wrapped(self, tmp_path: Path) -> None:
        rig = build_rig(tmp_path, start_utc_ns=NIGHT_START)
        try:
            assert rig.app.nightly is None
        finally:
            rig.app.stop()

    def test_the_night_closes_at_the_split_hour_without_a_new_frame(self, tmp_path: Path) -> None:
        analyzer = NightFake(station_id="test", profile_id="test")
        analyzer.open_night = [marker("night.first", NIGHT_START)]
        rig = build_rig(tmp_path, start_utc_ns=NIGHT_START, parts={"survey": analyzer})
        try:
            rig.app.start()
            rig.app.tick()
            assert analyzer.flushes == 0  # the night is on
            rig.clock.advance_to_utc_ns(NEXT_NOON)  # no frame comes in the daytime
            rig.app.tick()
            assert analyzer.flushes == 1
            assert [e.kind for e in rig.events()].count("night.first") == 1
            rig.app.tick()
            assert analyzer.flushes == 1  # once
        finally:
            rig.app.stop()

    def test_the_open_night_is_written_at_shutdown_before_the_store_closes(
        self, tmp_path: Path
    ) -> None:
        analyzer = NightFake(station_id="test", profile_id="test")
        analyzer.open_night = [marker("night.open", NIGHT_START)]
        rig = build_rig(tmp_path, start_utc_ns=NIGHT_START, parts={"survey": analyzer})
        rig.app.start()
        rig.run_for(5.0)
        path = rig.app.storage.layout.db_path  # type: ignore[union-attr]
        rig.app.stop("a test")
        assert analyzer.flushes == 1
        with Store.open(path) as store:
            kinds = [r.kind for r in read_all(store, "event")]  # type: ignore[attr-defined]
        assert "night.open" in kinds
        assert kinds.index("night.open") < kinds.index("core.stopped")  # before the last event

    def test_a_shutdown_of_a_core_that_never_started_writes_nothing(self, tmp_path: Path) -> None:
        analyzer = NightFake(station_id="test", profile_id="test")
        analyzer.open_night = [marker("night.open", NIGHT_START)]
        rig = build_rig(tmp_path, start_utc_ns=NIGHT_START, parts={"survey": analyzer})
        rig.app.stop()
        assert analyzer.flushes == 0
        assert events_of([], "night.open") == []


class TestTheDarkLibrary:
    def test_the_health_record_asks_the_library_of_the_data_directory(self, tmp_path: Path) -> None:
        rig = build_rig(tmp_path, start_utc_ns=NIGHT_START)
        try:
            rig.app.start()
            library = Path(rig.app.survey_config.calibration_dir)
            assert library == rig.app.storage.layout.root / "calibration"  # type: ignore[union-attr]
            rig.app.tick()
            (record,) = rig.records("health")[:1]
            assert record.dark_due is True  # type: ignore[attr-defined]  # nothing is in the library
        finally:
            rig.app.stop()
