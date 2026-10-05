"""`CoreApp` and the survey analysis: the sky flags, the nightly summary, and the dark library."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")

from seeingmon.analysis import SurveyOutput
from seeingmon.clock import Clock, ClockStatus, VirtualClock, iso_to_utc_ns
from seeingmon.records import EventRecord, Record
from seeingmon.records.survey import SkyQualityRecord
from seeingmon.services.core.app import CoreApp
from seeingmon.services.simsky import write_seed
from seeingmon.store.db import Store
from seeingmon.testing import FakeSurveyAnalyzer
from tests.survey.pointfx import HOUR_NS, MINUTE_NS, made, mount

from .rig import CoreRig, build_rig, events_of, read_all

DAY_NS = 24 * HOUR_NS

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


class TestTheStartWithoutAPointing:
    """A start with no pointing solution says in the log which solvers find the first one."""

    LOGGER = "seeingmon.services.core.app"

    @staticmethod
    def rig(tmp_path: Path, solvers: str) -> CoreRig:
        """A core with the real survey analysis (inline) and a small catalog, and no seed."""
        from seeingmon.survey.analyzer import InlineExecutor
        from seeingmon.survey.catalog import write_catalog
        from tests.survey import synth

        catalog = tmp_path / "catalog.bin"
        write_catalog(catalog, synth.synthetic_catalog(cap_radius_deg=5.0, density_scale=0.05))
        return build_rig(
            tmp_path,
            config_extra=f'[survey]\ncatalog_path = "{catalog.as_posix()}"\nsolvers = {solvers}\n',
            parts={"survey": None, "pointing": None, "survey_executor": InlineExecutor()},
        )

    def test_the_solvers_are_named_in_the_order_that_they_run(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("INFO", logger=self.LOGGER):
            rig = self.rig(tmp_path, '["astrometry.net", "astap"]')
        rig.app.stop()
        assert rig.app.tracker is not None
        assert rig.app.tracker.solution is None
        lines = [r for r in caplog.records if r.name == self.LOGGER]
        assert [(r.levelname, r.getMessage()) for r in lines] == [
            (
                "INFO",
                "the pointing tracker has no stored solution to start with: "
                "the store holds no pointing record",
            ),
            (
                "INFO",
                "no pointing solution yet: the survey frames go to the plate solvers "
                "astrometry.net, astap, in this order, until one solves",
            ),
        ]

    def test_a_start_without_a_solver_warns_that_nothing_can_find_polaris(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("INFO", logger=self.LOGGER):
            rig = self.rig(tmp_path, "[]")
        rig.app.stop()
        (record,) = [
            r for r in caplog.records if r.name == self.LOGGER and r.levelname == "WARNING"
        ]
        assert "names no plate solver" in record.getMessage()

    def test_a_tracker_that_holds_a_solution_adds_no_line(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        seeded = SimpleNamespace(solution=object())
        with caplog.at_level("INFO", logger=self.LOGGER):
            CoreApp._log_pointing_start(seeded, ["astap"])
        assert not [r for r in caplog.records if r.name == self.LOGGER]


class TestTheStartWithAStoredPointing:
    """A restart starts the tracker with the newest solved pointing record of the store."""

    LOGGER = "seeingmon.services.core.app"
    START = iso_to_utc_ns("2026-01-01T22:00:00Z")

    @classmethod
    def restart(
        cls,
        tmp_path: Path,
        records: Sequence[Record],
        *,
        settings: str = "",
        clock: Clock | None = None,
        caplog: pytest.LogCaptureFixture | None = None,
    ) -> CoreRig:
        """Start `core`, put `records` in its store, stop it, and start it again.

        With `caplog`, the log holds the second start only.
        """
        from seeingmon.survey.analyzer import InlineExecutor
        from seeingmon.survey.catalog import write_catalog
        from tests.survey import synth

        catalog = tmp_path / "catalog.bin"
        write_catalog(catalog, synth.synthetic_catalog(cap_radius_deg=5.0, density_scale=0.05))
        extra = f'[survey]\ncatalog_path = "{catalog.as_posix()}"\nsolvers = ["astap"]\n{settings}'

        def parts() -> dict[str, Any]:
            return {"survey": None, "pointing": None, "survey_executor": InlineExecutor()}

        first = build_rig(tmp_path, start_utc_ns=cls.START, config_extra=extra, parts=parts())
        assert first.app.storage is not None
        for record in records:
            first.app.storage.store.write(record)
        first.app.stop()
        if caplog is not None:
            caplog.clear()
        return build_rig(
            tmp_path, start_utc_ns=cls.START, config_extra=extra, parts=parts(), clock=clock
        )

    def test_the_tracker_starts_with_the_newest_good_solution_and_the_log_says_so(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        newest = made(self.START - 10 * MINUTE_NS, n_matched=853, solver="tracker")
        records: list[Record] = [
            made(self.START - 40 * MINUTE_NS).record,
            newest.record,
            made(self.START - 5 * MINUTE_NS, n_matched=7).record,  # too thin
            made(self.START - 3 * MINUTE_NS, solved=False).record,
        ]
        with caplog.at_level("INFO", logger=self.LOGGER):
            rig = self.restart(tmp_path, records, caplog=caplog)
        rig.app.stop()
        assert rig.app.tracker is not None
        solution = rig.app.tracker.solution
        assert solution is not None
        assert solution.t_utc_ns == newest.solution.t_utc_ns
        assert (solution.solver, solution.n_matched) == ("tracker", 853)
        np.testing.assert_allclose(
            solution.rotation_earth_fixed, newest.solution.rotation_earth_fixed, atol=1e-12
        )
        # The scheduler can place Polaris at once, so the first cycle needs no survey step.
        assert rig.app.tracker.polaris_position(self.START, "bin2") is not None
        messages = [r.getMessage() for r in caplog.records if r.name == self.LOGGER]
        assert messages == [
            "the pointing tracker starts with the stored solution of 2026-01-01T21:50:00Z "
            "(10 min old, solved by tracker with 853 matched stars)"
        ]

    def test_a_solution_that_is_30_days_old_seeds_the_tracker_without_a_limit(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The default `validity_s` 0 sets no age limit: a rigid mount keeps its attitude."""
        old = made(self.START - 30 * DAY_NS, n_matched=512)
        with caplog.at_level("INFO", logger=self.LOGGER):
            rig = self.restart(tmp_path, [old.record], caplog=caplog)
        rig.app.stop()
        assert rig.app.tracker is not None
        solution = rig.app.tracker.solution
        assert solution is not None
        assert solution.t_utc_ns == old.solution.t_utc_ns
        predicted = rig.app.tracker.polaris_position(self.START, "bin2")
        expected = old.solution.polaris_pixel(self.START)
        assert predicted is not None
        assert expected is not None
        assert predicted == pytest.approx(expected, abs=1e-6)  # the rebuild agrees to 1e-9
        messages = [r.getMessage() for r in caplog.records if r.name == self.LOGGER]
        assert messages == [
            "the pointing tracker starts with the stored solution of 2025-12-02T22:00:00Z "
            "(30.0 days old, solved by astrometry.net with 512 matched stars)"
        ]

    def test_a_month_of_unsolved_records_does_not_hide_the_last_good_solution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cloudy nights write unsolved records, more than one page of the search reads."""
        from seeingmon.survey import reference

        monkeypatch.setattr(reference, "SCAN_LIMIT", 10)  # the search reads three pages
        good = made(self.START - 30 * DAY_NS)
        cloudy = [made(self.START - n * DAY_NS, solved=False).record for n in range(29, 0, -1)]
        rig = self.restart(tmp_path, [good.record, *cloudy])
        rig.app.stop()
        assert rig.app.tracker is not None
        solution = rig.app.tracker.solution
        assert solution is not None
        assert solution.t_utc_ns == good.solution.t_utc_ns

    def test_with_a_positive_limit_an_older_solution_does_not_seed_the_tracker(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        limit = "[survey.pointing]\nvalidity_s = 43200.0\n"  # 12 h, for a mount that is not rigid
        with caplog.at_level("INFO", logger=self.LOGGER):
            rig = self.restart(
                tmp_path, [made(self.START - 13 * HOUR_NS).record], settings=limit, caplog=caplog
            )
        rig.app.stop()
        assert rig.app.tracker is not None
        assert rig.app.tracker.solution is None
        messages = [r.getMessage() for r in caplog.records if r.name == self.LOGGER]
        assert messages[0] == (
            "the pointing tracker has no stored solution to start with: "
            "the newest pointing record is 13.0 h old, and the limit is 720 minutes"
        )
        assert messages[1].startswith("no pointing solution yet")

    def test_a_record_that_a_bad_clock_wrote_is_skipped_for_the_one_before_it(
        self, tmp_path: Path
    ) -> None:
        good = made(self.START - 30 * MINUTE_NS)
        records: list[Record] = [
            good.record,
            made(self.START - 5 * MINUTE_NS, time_invalid=True).record,
        ]
        rig = self.restart(tmp_path, records)
        rig.app.stop()
        assert rig.app.tracker is not None
        solution = rig.app.tracker.solution
        assert solution is not None
        assert solution.t_utc_ns == good.solution.t_utc_ns

    def test_a_setting_turns_the_seed_off(self, tmp_path: Path) -> None:
        records: list[Record] = [made(self.START - 10 * MINUTE_NS).record]
        rig = self.restart(tmp_path, records, settings="[services.core]\nseed_from_store = false\n")
        rig.app.stop()
        assert rig.app.tracker is not None
        assert rig.app.tracker.solution is None

    def test_a_seed_file_goes_before_the_store(self, tmp_path: Path) -> None:
        stored = made(self.START - 10 * MINUTE_NS)
        filed = made(self.START - 2 * HOUR_NS, rotation_tirs=mount(0.7))
        seed_file = tmp_path / "seed.json"
        write_seed(seed_file, filed.solution)
        settings = f'[services.core]\nseed_solution_file = "{seed_file.as_posix()}"\n'
        rig = self.restart(tmp_path, [stored.record], settings=settings)
        rig.app.stop()
        assert rig.app.tracker is not None
        solution = rig.app.tracker.solution
        assert solution is not None
        assert solution.t_utc_ns == filed.solution.t_utc_ns

    def test_a_clock_that_is_not_synchronized_starts_the_tracker_empty(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        unsynchronized = VirtualClock(
            self.START, status=ClockStatus(synchronized=False, error_bound_ns=None, source="test")
        )
        records: list[Record] = [made(self.START - 10 * MINUTE_NS).record]
        with caplog.at_level("INFO", logger=self.LOGGER):
            rig = self.restart(tmp_path, records, clock=unsynchronized, caplog=caplog)
        rig.app.stop()
        assert rig.app.tracker is not None
        assert rig.app.tracker.solution is None
        messages = [r.getMessage() for r in caplog.records if r.name == self.LOGGER]
        assert "the pointing tracker starts empty: the clock is not synchronized" in messages
