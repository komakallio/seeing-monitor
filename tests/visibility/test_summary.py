"""The visibility summary of one night, from synthetic events, health records, and windows."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from seeingmon.records.system import HEALTH_STATES
from seeingmon.records.visibility import VisibilitySummaryRecord
from seeingmon.scheduler.events import EVENT_KINDS
from seeingmon.visibility import summary as sm
from seeingmon.visibility.summary import (
    CLEAR_VERDICT_EVENT,
    CLOCK_UNSYNCHRONIZED_EVENT,
    DARK_EVENT,
    SOLVE_REQUESTED_EVENT,
    Event,
    HealthMark,
    NightSummary,
    WindowMark,
    longest_gap_ns,
    summarize_night,
    watched_stretches,
)

from .nightfx import (
    END,
    HOUR,
    MINUTE,
    RULES,
    START,
    at,
    health,
    health_values,
    hidden,
    night,
    started,
    visible,
    windows,
)

SUN_AT_BOUNDS = 30.0  # what `sun_at` answers: the split hour lies in the daytime
HOURS_TOLERANCE = 1e-6  # the summary rounds the hours to 1e-6 h (3.6 ms)


def summarize(data: sm.NightData, *, moon: bool | None = False) -> NightSummary:
    return summarize_night(data, RULES, sun_at=lambda t: SUN_AT_BOUNDS, moon_at=lambda t: moon)


def as_record(result: NightSummary) -> VisibilitySummaryRecord:
    """The record that `core` builds from the summary: the constructor checks every value."""
    return VisibilitySummaryRecord(
        station_id="test",
        t_utc_ns=START,
        profile_id="test",
        provenance={"software": "test"},
        **result.fields(),
    )


class TestTheEventKinds:
    def test_the_kinds_that_the_summary_reads_are_the_ones_that_core_and_the_scheduler_write(
        self,
    ) -> None:
        from seeingmon.services.core.darkness import CLEAR_VERDICT_EVENT as VERDICT
        from seeingmon.services.core.darkness import DARK_EVENT as DARK

        assert (DARK_EVENT, CLEAR_VERDICT_EVENT) == (DARK, VERDICT)
        scheduler_kinds = sm.SUMMARY_EVENT_KINDS - {sm.CORE_STARTED_EVENT}
        assert scheduler_kinds <= set(EVENT_KINDS)


class TestTheRowsOfTheStore:
    @pytest.mark.parametrize(
        ("state", "components", "watching"),
        [
            ("auto", None, True),
            ("safe", None, True),  # a sky too bright to measure is a limit of the detection
            ("paused", None, False),
            ("align", None, False),
            ("commission", None, False),
            ("auto", {"heater": "failed"}, True),  # the search goes on
            ("auto", {"acquire": "failed"}, True),  # one health reply that did not come
            ("auto", {"sqm": "degraded"}, True),
            ("auto", {"scheduler": "degraded", "camera": "failed"}, False),  # the camera failed
            ("safe", {"scheduler": "degraded", "camera": "failed"}, False),
        ],
    )
    def test_the_station_watches_in_auto_and_safe_while_the_camera_works(
        self, state: str, components: dict[str, str] | None, watching: bool
    ) -> None:
        values = health_values(0, state=state, components=components)
        assert HealthMark.from_values(values).watching is watching

    def test_the_cases_above_cover_every_state_of_the_health_record(self) -> None:
        assert set(HEALTH_STATES) == {"auto", "safe", "paused", "align", "commission"}
        assert set(HEALTH_STATES) >= sm.WATCHING_STATES

    def test_a_row_without_the_schedulers_component_falls_back_to_degraded(self) -> None:
        bare: dict[str, Any] = {"t_utc_ns": 0, "state": "auto", "components": {"core": "ok"}}
        assert HealthMark.from_values({**bare, "degraded": True}).watching is False
        assert HealthMark.from_values({**bare, "degraded": False}).watching is True

    @pytest.mark.parametrize("synchronized", [True, False, None])
    def test_the_state_of_the_clock_carries_over(self, synchronized: bool | None) -> None:
        mark = HealthMark.from_values(health_values(7, synchronized=synchronized))
        assert (mark.t_utc_ns, mark.synchronized) == (7, synchronized)

    def test_a_window_has_seeing_with_r0_or_a_fwhm_and_carries_time_invalid(self) -> None:
        base: dict[str, Any] = {"t_utc_ns": 5, "duration_s": 60.0, "flags": []}
        assert WindowMark.from_values({**base, "r0_cm": 9.0}) == WindowMark(5, 60.0, True)
        assert WindowMark.from_values({**base, "seeing_fwhm_arcsec": 2.0}).has_seeing
        assert not WindowMark.from_values({**base, "r0_cm": None}).has_seeing
        marked = WindowMark.from_values({**base, "flags": ["daylight", "time_invalid"]})
        assert marked.time_invalid
        assert not WindowMark.from_values({**base, "flags": None}).time_invalid


class TestTheWatchedStretches:
    def test_a_record_vouches_until_the_next_one_and_for_at_most_the_span(self) -> None:
        marks = [
            HealthMark(0, True),
            HealthMark(60, True),
            HealthMark(1000, True),  # after a gap: the record before vouches for 100 at most
            HealthMark(1060, False),  # a pause
            HealthMark(1120, True),
        ]
        assert watched_stretches(marks, 100) == [(0, 160), (1000, 1060), (1120, 1220)]

    def test_the_longest_gap_counts_the_edges_of_the_span(self) -> None:
        stretches = [(100, 200), (250, 300)]
        assert longest_gap_ns(stretches, 0, 400) == 100  # 0 to 100 and 300 to 400
        assert longest_gap_ns(stretches, 120, 280) == 50  # 200 to 250
        assert longest_gap_ns(stretches, 120, 180) == 0
        assert longest_gap_ns([], 0, 400) == 400


class TestAWatchedNight:
    def test_a_clear_night_from_dusk_to_dawn_gives_measured_detections(self) -> None:
        data = night(
            [visible(at("16:00"), sun=-4.5), hidden(at("07:00", day=2), sun=-5.5)],
            window_marks=windows(at("16:00"), at("16:30")),
        )
        result = summarize(data)
        assert result.first_visible_utc_ns == at("16:00")
        assert result.first_visible_sun_deg == -4.5  # from the event
        assert result.last_visible_utc_ns == at("07:00", day=2)
        assert result.last_visible_sun_deg == -5.5
        assert (result.first_censored, result.last_censored) == (False, False)
        assert result.visible_hours == pytest.approx(15.0, abs=HOURS_TOLERANCE)
        assert result.seeing_hours == pytest.approx(0.5, abs=HOURS_TOLERANCE)
        assert result.flags == ()
        record = as_record(result)
        assert record.night == "2026-01-01"
        assert set(record.quality or {}) == {
            "dark_utc_ns",
            "dark_sun_deg",
            "dark_sky_mag_arcsec2",
            "clear_share",
            "transparency_median",
        }  # every value that is missing says why

    def test_clouds_in_the_night_split_the_measure_and_keep_the_first_and_the_last(self) -> None:
        data = night(
            [
                visible(at("16:00")),
                hidden(at("20:00")),
                visible(at("22:00")),
                hidden(at("07:00", day=2)),
            ]
        )
        result = summarize(data)
        assert result.first_visible_utc_ns == at("16:00")
        assert result.last_visible_utc_ns == at("07:00", day=2)
        assert result.visible_hours == pytest.approx(4.0 + 9.0, abs=HOURS_TOLERANCE)

    def test_a_night_without_a_detection_is_unseen_and_explains_the_missing_times(self) -> None:
        result = summarize(night())
        assert result.first_visible_utc_ns is None
        assert result.last_visible_utc_ns is None
        assert (result.first_censored, result.last_censored) == (False, False)
        assert result.visible_hours == 0.0
        assert result.quality["first_visible_utc_ns"] == sm.NO_DETECTION
        assert result.quality["last_visible_sun_deg"] == sm.NO_DETECTION
        as_record(result)

    def test_a_restart_shorter_than_the_limit_leaves_the_detections_measured(self) -> None:
        marks = health(START - 24 * HOUR, at("14:00")) + health(at("14:04"), END)
        data = night(
            [started(at("14:04")), visible(at("16:00")), hidden(at("07:00", day=2))], marks
        )
        result = summarize(data)
        assert (result.first_censored, result.last_censored) == (False, False)

    @pytest.mark.parametrize("hole", ["records that come late", "a record that does not watch"])
    def test_a_hole_in_the_watch_keeps_a_measure_that_polaris_hidden_ends(self, hole: str) -> None:
        # The supervisor thread was busy for ten minutes, or one record does not watch. Measure
        # went on, and the scheduler ended it with polaris.hidden, which marks the end.
        marks = health(START - 24 * HOUR, END)
        if hole == "records that come late":
            marks = [m for m in marks if not at("21:00") <= m.t_utc_ns < at("21:10")]
        else:
            marks = [
                dataclasses.replace(m, watching=False) if m.t_utc_ns == at("21:00") else m
                for m in marks
            ]
        data = night([visible(at("16:00"), sun=-4.5), hidden(at("07:00", day=2), sun=-5.5)], marks)
        result = summarize(data)
        assert result.visible_hours == pytest.approx(15.0, abs=HOURS_TOLERANCE)
        assert result.last_visible_utc_ns == at("07:00", day=2)
        assert result.last_visible_sun_deg == -5.5  # from the event, not from the site
        assert (result.first_censored, result.last_censored) == (False, False)


class TestCensoredDetections:
    def test_polaris_visible_across_the_split_hours_gives_bounds_at_both_ends(self) -> None:
        data = night([visible(START - 2 * HOUR), hidden(at("20:00")), visible(at("21:00"))])
        result = summarize(data)
        assert result.first_visible_utc_ns == START
        assert result.first_visible_sun_deg == SUN_AT_BOUNDS  # no event: from the site
        assert result.last_visible_utc_ns == END
        assert result.last_visible_sun_deg == SUN_AT_BOUNDS
        assert (result.first_censored, result.last_censored) == (True, True)
        assert result.visible_hours == pytest.approx(8.0 + 15.0, abs=HOURS_TOLERANCE)

    def test_a_start_of_core_in_the_night_censors_a_detection_that_follows_at_once(self) -> None:
        # The station did not run before 18:10, and found Polaris at once: it may have shown hours
        # before. The last detection comes after a watched stretch, so it is measured.
        marks = health(at("18:10"), END)
        data = night([started(at("18:10")), visible(at("18:11")), hidden(at("23:00"))], marks)
        result = summarize(data)
        assert result.first_visible_utc_ns == at("18:11")
        assert result.first_censored is True
        assert result.last_censored is False

    def test_a_pause_after_the_last_detection_censors_it(self) -> None:
        marks = health(START - 24 * HOUR, at("02:00", day=2)) + health(
            at("02:00", day=2), END, state="paused"
        )
        data = night(
            [visible(at("16:00")), hidden(at("02:00", day=2), reason="state_change")], marks
        )
        result = summarize(data)
        assert result.last_visible_utc_ns == at("02:00", day=2)
        assert result.last_censored is True  # the station did not watch the rest of the night
        assert result.first_censored is False

    def test_a_night_without_a_detection_and_with_a_gap_is_censored_at_both_ends(self) -> None:
        marks = health(START - 24 * HOUR, at("20:00")) + health(at("23:00"), END)
        result = summarize(night([started(at("23:00"))], marks))
        assert (result.first_censored, result.last_censored) == (True, True)

    def test_a_crash_ends_the_measure_where_the_health_records_stop(self) -> None:
        marks = health(START - 24 * HOUR, at("22:00")) + health(at("23:00"), END)
        data = night(
            [
                visible(at("18:00")),
                started(at("23:00")),  # no polaris.hidden: core crashed after 22:00
                visible(at("23:01")),
                hidden(at("05:00", day=2)),
            ],
            marks,
        )
        result = summarize(data)
        # The last record before the crash came at 21:59 and vouches for two minutes.
        crash_ns = at("21:59") + 2 * MINUTE
        expected_ns = (crash_ns - at("18:00")) + (at("05:00", day=2) - at("23:01"))
        assert result.visible_hours == pytest.approx(expected_ns / HOUR, abs=HOURS_TOLERANCE)
        assert result.first_visible_utc_ns == at("18:00")
        assert result.first_censored is False
        assert result.last_visible_utc_ns == at("05:00", day=2)
        assert result.last_censored is False

    def test_a_crash_without_a_restart_censors_the_end_of_the_measure(self) -> None:
        marks = health(START - 24 * HOUR, at("02:00", day=2))
        result = summarize(night([visible(at("16:00"))], marks))
        assert result.last_visible_utc_ns == at("01:59", day=2) + 2 * MINUTE
        assert result.last_visible_sun_deg == SUN_AT_BOUNDS  # no event marks the end
        assert result.last_censored is True

    def test_the_previous_summary_says_whether_polaris_was_visible_at_the_start(self) -> None:
        # No polaris event in the day before: Polaris was visible for more than a day, or not.
        events = [hidden(at("18:00"))]
        carried = summarize(night(events, visible_before=True))
        assert carried.first_visible_utc_ns == START
        assert carried.first_censored is True
        assert carried.visible_hours == pytest.approx(6.0, abs=HOURS_TOLERANCE)
        not_carried = summarize(night(events, visible_before=False))
        assert not_carried.first_visible_utc_ns is None
        assert not_carried.visible_hours == 0.0

    def test_an_event_in_the_day_before_wins_over_the_previous_summary(self) -> None:
        events = [visible(START - 3 * HOUR), hidden(START - HOUR)]
        result = summarize(night(events, visible_before=True))
        assert result.first_visible_utc_ns is None
        assert result.first_censored is False


class TestTheSkyAndTheFlags:
    def test_the_darkness_and_the_verdict_come_from_their_events(self) -> None:
        data = night(
            [
                visible(at("16:00")),
                Event(
                    at("18:30"),
                    DARK_EVENT,
                    {"sky_mag_arcsec2": 20.81, "sun_elevation_deg": -17.2, "frames": 5},
                ),
                Event(
                    at("18:45"),
                    CLEAR_VERDICT_EVENT,
                    {"clear_share": 0.8, "transparency_median": 0.93, "frames": 5},
                ),
                hidden(at("07:00", day=2)),
            ]
        )
        result = summarize(data)
        assert result.dark_utc_ns == at("18:30")
        assert (result.dark_sun_deg, result.dark_sky_mag_arcsec2) == (-17.2, 20.81)
        assert (result.clear_share, result.transparency_median) == (0.8, 0.93)
        assert result.quality == {}
        assert as_record(result).quality is None

    @pytest.mark.parametrize(
        ("event", "name"),
        [
            (Event(at("18:30"), DARK_EVENT, {"sun_elevation_deg": -17.0}), "dark_sky_mag_arcsec2"),
            (
                Event(at("18:30"), DARK_EVENT, {"sky_mag_arcsec2": float("nan")}),
                "dark_sky_mag_arcsec2",
            ),
            (Event(at("18:45"), CLEAR_VERDICT_EVENT, {"transparency_median": 0.9}), "clear_share"),
            (
                Event(at("18:45"), CLEAR_VERDICT_EVENT, {"clear_share": 1.5}),
                "clear_share",
            ),  # out of [0, 1]
            (Event(at("18:45"), CLEAR_VERDICT_EVENT, {"clear_share": True}), "clear_share"),
        ],
    )
    def test_a_value_that_an_event_lacks_is_none_and_says_why(
        self, event: Event, name: str
    ) -> None:
        data = night([visible(at("16:00")), event, hidden(at("07:00", day=2))])
        result = summarize(data)
        assert result.fields()[name] is None
        record = as_record(result)
        missing = {field for field, value in record.model_dump().items() if value is None}
        assert name in missing
        assert set(record.quality or {}) == missing  # every missing value says why

    def test_a_verdict_without_a_transparency_says_why(self) -> None:
        verdict = Event(at("18:45"), CLEAR_VERDICT_EVENT, {"clear_share": 0.4})
        result = summarize(night([verdict]))
        assert result.clear_share == 0.4
        assert result.transparency_median is None
        assert result.quality["transparency_median"] == sm.NO_TRANSPARENCY

    def test_an_event_without_the_suns_elevation_says_why(self) -> None:
        data = night([visible(at("16:00"), sun=None), hidden(at("07:00", day=2), sun=None)])
        result = summarize(data)
        assert result.first_visible_sun_deg is None
        assert result.quality["first_visible_sun_deg"] == sm.NO_EVENT_SUN
        assert result.quality["last_visible_sun_deg"] == sm.NO_EVENT_SUN

    def test_a_bound_without_a_site_says_why(self) -> None:
        data = night([visible(START - HOUR)])
        result = summarize_night(data, RULES, sun_at=lambda t: None, moon_at=lambda t: None)
        assert result.first_visible_sun_deg is None
        assert result.quality["first_visible_sun_deg"] == sm.NO_SITE

    def test_the_moon_counts_at_the_detections_and_at_sky_dark(self) -> None:
        dark = Event(at("18:30"), DARK_EVENT, {"sky_mag_arcsec2": 20.8})
        events = [visible(at("16:00")), dark, hidden(at("07:00", day=2))]
        seen: list[int] = []

        def moon(t: int) -> bool:
            seen.append(t)
            return t == at("18:30")

        result = summarize_night(night(events), RULES, sun_at=lambda t: 0.0, moon_at=moon)
        assert "moon" in result.flags
        assert sorted(seen) == [at("16:00"), at("18:30"), at("07:00", day=2)]

    def test_without_a_site_the_moon_flag_is_unknown_and_says_so(self) -> None:
        result = summarize(night([visible(at("16:00"))]), moon=None)
        assert "moon" not in result.flags
        assert result.quality["flags"] == sm.MOON_UNKNOWN
        as_record(result)

    @pytest.mark.parametrize(
        "change",
        ["event", "health", "window"],
    )
    def test_an_unsynchronized_clock_in_the_night_sets_time_invalid(self, change: str) -> None:
        events = [visible(at("16:00"))]
        marks = health(START - 24 * HOUR, END)
        window_marks: list[WindowMark] = []
        if change == "event":
            events.append(Event(at("20:00"), CLOCK_UNSYNCHRONIZED_EVENT, {"synchronized": False}))
        elif change == "health":
            marks = [HealthMark(m.t_utc_ns, m.watching, m.t_utc_ns != at("20:00")) for m in marks]
        else:
            window_marks = [WindowMark(at("20:00"), 60.0, True, time_invalid=True)]
        result = summarize(night(events, marks, window_marks))
        assert "time_invalid" in result.flags

    def test_a_request_to_solve_sets_no_pointing(self) -> None:
        event = Event(at("20:00"), SOLVE_REQUESTED_EVENT, {"reason": "no_solution"})
        result = summarize(night([event]))
        assert result.flags == ("no_pointing",)

    def test_rows_outside_the_night_set_no_flag_and_add_no_hours(self) -> None:
        before = START - HOUR
        data = night(
            [Event(before, SOLVE_REQUESTED_EVENT, {}), Event(END, CLOCK_UNSYNCHRONIZED_EVENT, {})],
            window_marks=[WindowMark(before, 60.0, True), WindowMark(END, 60.0, True)],
        )
        result = summarize(data)
        assert result.flags == ()
        assert result.seeing_hours == 0.0

    def test_windows_without_a_seeing_value_add_no_seeing_hours(self) -> None:
        marks = [WindowMark(at("20:00"), 60.0, False), WindowMark(at("20:01"), 60.0, True)]
        result = summarize(night(window_marks=marks))
        assert result.seeing_hours == pytest.approx(60.0 / 3600.0, abs=HOURS_TOLERANCE)


def test_the_rules_refuse_a_negative_gap() -> None:
    with pytest.raises(ValueError, match="max_gap_s"):
        sm.SummaryRules(max_gap_s=-1.0)
    with pytest.raises(ValueError, match="health_span_s"):
        sm.SummaryRules(health_span_s=0.0)


def test_the_night_ends_one_day_after_it_starts() -> None:
    assert END - START == sm.NIGHT_NS
