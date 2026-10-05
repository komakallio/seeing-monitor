"""Record JSON, history, aggregation, and paging in `StoreData`."""

from __future__ import annotations

from typing import Any

import pytest

from seeingmon.clock import NS_PER_S, VirtualClock
from seeingmon.config import ConfigError
from seeingmon.records.samples import sample_record
from seeingmon.records.system import HealthRecord
from seeingmon.services.web.config import WebSettings
from seeingmon.services.web.data import (
    InvalidQueryError,
    Step,
    StoreData,
    StoreUnavailableError,
    TimeRange,
    aggregate_rows,
    array_fields,
    decode_cursor,
    encode_cursor,
    parse_time,
)
from seeingmon.store.db import Store, StoreReader
from tests.services.web.seed import BASE_NS, MINUTE_NS, NOW_NS, STATION, at, common

PRIVATE_TEXT = "Cannot write C:\\Users\\someone\\data\\db.sqlite to 192.0.2.7."  # repo-check: allow


@pytest.fixture
def data(
    seeded: Store, reader: StoreReader, settings: WebSettings, clock: VirtualClock
) -> StoreData:
    return StoreData(reader, settings, clock, station_id=STATION)


def window(start_min: float, end_min: float) -> TimeRange:
    return TimeRange(at(start_min), at(end_min))


def history(
    data: StoreData,
    record_type: str = "seeing_window",
    *,
    start: float = 0,
    end: float = 60,
    step: Step = Step.RAW,
    limit: int = 100,
    cursor: str | None = None,
    wanted: frozenset[str] | None = None,
) -> Any:
    return data.history(
        record_type,
        time_range=window(start, end),
        step=step,
        limit=limit,
        cursor=cursor,
        wanted=wanted,
    )


def read_all(data: StoreData, **options: Any) -> list[dict[str, Any]]:
    """Follow the cursors of a history query to its end."""
    items: list[dict[str, Any]] = []
    cursor = None
    for _ in range(100):
        page = history(data, cursor=cursor, **options)
        items.extend(page.items)
        cursor = page.next_cursor
        if cursor is None:
            return items
    raise AssertionError("the cursors never ended")


# --- Record JSON -----------------------------------------------------------------------------


def test_the_latest_record_has_every_field_and_an_iso_time(data: StoreData) -> None:
    record = data.latest("seeing_window")
    assert record is not None
    assert record["t_utc_ns"] == at(21)
    assert record["t_utc"] == "2026-10-01T02:21:00.000000Z"
    assert record["station_id"] == STATION
    assert "motion_psd_freq_hz" in record  # a record carries its arrays, and a history row does not


def test_the_latest_record_of_an_empty_store_is_none(
    writer: Store, reader: StoreReader, settings: WebSettings, clock: VirtualClock
) -> None:
    empty = StoreData(reader, settings, clock, station_id=STATION)
    for record_type in ("seeing_window", "sky_quality", "pointing", "health", "event"):
        assert empty.latest(record_type) is None


def test_a_withheld_field_is_null_with_a_quality_note(data: StoreData) -> None:
    record = data.latest("seeing_window")
    assert record is not None
    assert record["zenith_angle_deg"] is None
    assert record["quality"] == {"zenith_angle_deg": "withheld"}


def test_a_withheld_field_keeps_the_reasons_that_the_record_gave(data: StoreData) -> None:
    rows = history(data, start=2, end=3).items
    assert rows[0]["quality"] == {
        "seeing_fwhm_arcsec": "too few usable frames",
        "zenith_angle_deg": "withheld",
    }


def test_a_field_is_served_when_the_settings_withhold_nothing(
    reader: StoreReader, seeded: Store, clock: VirtualClock
) -> None:
    open_data = StoreData(reader, WebSettings(withhold_fields=()), clock, station_id=STATION)
    record = open_data.latest("seeing_window")
    assert record is not None
    assert record["zenith_angle_deg"] == 33.0
    assert record["quality"] is None


def test_the_settings_must_withhold_a_field_that_a_record_has(
    reader: StoreReader, clock: VirtualClock
) -> None:
    with pytest.raises(ConfigError, match="no_such_field"):
        StoreData(reader, WebSettings(withhold_fields=("no_such_field",)), clock)


def test_the_settings_cannot_withhold_a_field_that_must_have_a_value(
    reader: StoreReader, clock: VirtualClock
) -> None:
    with pytest.raises(ConfigError, match="cannot be null"):
        StoreData(reader, WebSettings(withhold_fields=("n_frames",)), clock)


def test_the_text_of_an_event_loses_its_paths_and_addresses(
    writer: Store, reader: StoreReader, settings: WebSettings, clock: VirtualClock
) -> None:
    writer.write(
        sample_record(
            "event",
            **common(
                NOW_NS,
                level="error",
                kind="storage.failed",
                message=PRIVATE_TEXT,
                detail={"file": "/home/someone/data/x.sqlite", "count": 3},  # repo-check: allow
            ),
        )
    )
    data = StoreData(reader, settings, clock, station_id=STATION)
    event = data.latest("event")
    assert event is not None
    assert "someone" not in str(event)
    assert "192.0.2.7" not in str(event)
    assert event["detail"]["count"] == 3


def test_an_event_serves_no_suns_elevation(
    writer: Store, reader: StoreReader, settings: WebSettings, clock: VirtualClock
) -> None:
    """The elevation at the time of the event shows the site, so the API withholds it."""
    detail = {"sky_mag_arcsec2": 20.5, "slope_mag_per_hour": 0.1, "sun_elevation_deg": -18.73}
    writer.write(
        sample_record(
            "event",
            **common(NOW_NS, level="info", kind="sky.dark", message="Dark.", detail=detail),
        )
    )
    data = StoreData(reader, settings, clock, station_id=STATION)
    event = data.latest("event")
    assert event is not None
    assert event["detail"] == {
        "sky_mag_arcsec2": 20.5,
        "slope_mag_per_hour": 0.1,
        "sun_elevation_deg": None,
    }
    (listed,) = data.events(
        time_range=TimeRange(NOW_NS, NOW_NS + 1),
        limit=10,
        cursor=None,
        min_level="info",
        kind="sky.",
        descending=True,
    ).items
    assert listed["detail"]["sun_elevation_deg"] is None


def test_only_the_records_of_the_station_are_served(
    seeded: Store, reader: StoreReader, settings: WebSettings, clock: VirtualClock
) -> None:
    seeded.write(
        sample_record(
            "seeing_window",
            station_id="another-station",
            profile_id="p",
            t_utc_ns=at(30),
            seeing_fwhm_arcsec=9.9,
        )
    )
    mine = StoreData(reader, settings, clock, station_id=STATION)
    everyone = StoreData(reader, settings, clock)
    assert mine.latest("seeing_window")["t_utc_ns"] == at(21)  # type: ignore[index]
    assert everyone.latest("seeing_window")["t_utc_ns"] == at(30)  # type: ignore[index]
    assert len(history(mine).items) == 10
    assert len(history(everyone).items) == 11


# --- Raw history -----------------------------------------------------------------------------


def test_raw_history_returns_the_records_oldest_first_without_the_arrays(data: StoreData) -> None:
    page = history(data)
    assert [row["t_utc_ns"] for row in page.items] == [
        at(m) for m in (0, 1, 2, 3, 4, 5, 10, 11, 20, 21)
    ]
    assert page.next_cursor is None
    assert page.step is Step.RAW
    assert "motion_psd_freq_hz" not in page.items[0]
    assert "attitude" not in history(data, "pointing").items[0]
    assert page.items[0]["flags"] == []


def test_the_range_includes_its_start_and_excludes_its_end(data: StoreData) -> None:
    rows = history(data, start=1, end=4).items
    assert [row["t_utc_ns"] for row in rows] == [at(1), at(2), at(3)]


@pytest.mark.parametrize("limit", [1, 2, 3, 4, 9, 10, 11])
def test_pages_follow_one_another_without_gaps_or_repeats(data: StoreData, limit: int) -> None:
    whole = [row["t_utc_ns"] for row in history(data).items]
    paged = [row["t_utc_ns"] for row in read_all(data, limit=limit)]
    assert paged == whole


def test_a_page_that_ends_the_data_has_no_cursor(data: StoreData) -> None:
    assert history(data, limit=10).next_cursor is None
    assert history(data, limit=9).next_cursor is not None


def test_a_cursor_before_the_range_cannot_widen_it(data: StoreData) -> None:
    cursor = encode_cursor(at(-100))
    rows = history(data, start=4, end=6, cursor=cursor).items
    assert [row["t_utc_ns"] for row in rows] == [at(4), at(5)]


def test_a_cursor_after_the_range_returns_nothing(data: StoreData) -> None:
    page = history(data, start=0, end=6, cursor=encode_cursor(at(50)))
    assert page.items == []
    assert page.next_cursor is None


# --- Aggregation, checked by hand ------------------------------------------------------------


def test_ten_minute_buckets_hold_the_mean_and_the_union_of_the_flags(data: StoreData) -> None:
    items = history(data, step=Step.TEN_MINUTES).items
    assert [item["t_utc_ns"] for item in items] == [at(0), at(10), at(20)]
    first = items[0]
    assert first["t_utc"] == "2026-10-01T02:00:00.000000Z"
    # Minutes 0 to 5: seeing 1.0, 1.2, (missing), 1.4, 1.6, 1.8, so the mean is 7.0 / 5.
    assert first["seeing_fwhm_arcsec"] == 1.4
    # r0: 10.1 + 8.4 + 7.2 + 6.3 + 5.6 = 37.6, and 37.6 / 5 = 7.52.
    assert first["r0_cm"] == 7.52
    assert first["flags"] == ["cloud", "twilight", "vibration"]
    assert first["n_samples"] == 6
    # n_dropped is minute % 3, so 0, 1, 2, 0, 1, 2 with the mean 1.0.
    assert first["n_dropped"] == 1.0
    # scintillation_index is 0.002 * (minute + 1), so the mean of 0.002 ... 0.012 is 0.007.
    assert first["scintillation_index"] == 0.007
    # The settings of the stream keep the last value.
    assert (first["stream_id"], first["gain"], first["exposure_us"]) == (7, 0, 2000)
    assert first["readout_mode"] == "bin1"
    assert first["station_id"] == STATION


def test_a_bucket_ignores_the_missing_values_in_its_mean(data: StoreData) -> None:
    second = history(data, step=Step.TEN_MINUTES).items[1]
    assert second["seeing_fwhm_arcsec"] == 2.0  # minute 10 has 2.0, and minute 11 has none
    assert second["n_samples"] == 2


def test_a_bucket_without_any_value_is_null_with_the_reason_that_a_record_gave(
    data: StoreData,
) -> None:
    third = history(data, step=Step.TEN_MINUTES).items[2]
    assert third["seeing_fwhm_arcsec"] is None
    assert third["quality"] == {
        "seeing_fwhm_arcsec": "the star is not detected",
        "zenith_angle_deg": "withheld",
    }
    assert third["r0_cm"] is None  # no record explained it, so the bucket does not either
    assert "r0_cm" not in third["quality"]
    assert third["n_samples"] == 2


def test_one_minute_buckets_equal_the_records(data: StoreData) -> None:
    items = history(data, step=Step.MINUTE).items
    assert len(items) == 10
    assert all(item["n_samples"] == 1 for item in items)
    assert items[2]["seeing_fwhm_arcsec"] is None
    assert items[2]["quality"]["seeing_fwhm_arcsec"] == "too few usable frames"
    assert items[2]["flags"] == ["cloud", "vibration"]
    assert items[5]["seeing_fwhm_arcsec"] == 1.8


def test_the_hour_bucket_combines_every_record(data: StoreData) -> None:
    items = history(data, step=Step.HOUR).items
    assert len(items) == 1
    # (1.0 + 1.2 + 1.4 + 1.6 + 1.8 + 2.0) / 6 = 9.0 / 6 = 1.5
    assert items[0]["seeing_fwhm_arcsec"] == 1.5
    assert items[0]["n_samples"] == 10
    assert items[0]["t_utc_ns"] == at(0)


def test_buckets_start_at_a_multiple_of_the_step_even_when_the_range_does_not(
    data: StoreData,
) -> None:
    items = history(data, start=3, end=7, step=Step.TEN_MINUTES).items
    assert len(items) == 1
    assert items[0]["t_utc_ns"] == at(0)
    assert items[0]["n_samples"] == 3  # minutes 3, 4, and 5 only
    assert items[0]["seeing_fwhm_arcsec"] == 1.6  # (1.4 + 1.6 + 1.8) / 3


def test_the_end_of_the_range_cuts_the_last_bucket(data: StoreData) -> None:
    items = history(data, start=0, end=11, step=Step.TEN_MINUTES).items
    assert [item["n_samples"] for item in items] == [6, 1]  # minute 11 is excluded


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_aggregated_pages_join_up(data: StoreData, limit: int) -> None:
    whole = history(data, step=Step.TEN_MINUTES).items
    paged = read_all(data, step=Step.TEN_MINUTES, limit=limit)
    assert paged == whole


def test_the_cursor_of_an_aggregated_page_is_the_start_of_the_next_bucket(data: StoreData) -> None:
    page = history(data, step=Step.TEN_MINUTES, limit=1)
    assert len(page.items) == 1
    assert page.next_cursor is not None
    assert decode_cursor(page.next_cursor) == at(10)


def test_the_scan_budget_ends_a_page_at_a_bucket_boundary(
    writer: Store, reader: StoreReader, clock: VirtualClock
) -> None:
    for minute in range(120):
        writer.write(
            sample_record(
                "seeing_window", **common(at(minute), seeing_fwhm_arcsec=1.0 + minute / 100)
            )
        )
    settings = WebSettings.model_validate({"paging": {"max_scan_rows": 100}})
    data = StoreData(reader, settings, clock, station_id=STATION)
    data.aggregate_chunk = 30  # a small chunk, so the budget of 100 rows runs out inside the data
    whole = read_all(data, step=Step.TEN_MINUTES, limit=100, end=130)
    assert [item["n_samples"] for item in whole] == [10] * 12
    first = history(data, step=Step.TEN_MINUTES, limit=100, end=130)
    assert first.next_cursor is not None  # the budget ended the page before the data did
    assert len(first.items) < 12


def test_a_bucket_larger_than_the_scan_budget_is_still_returned(
    writer: Store, reader: StoreReader, clock: VirtualClock
) -> None:
    for second in range(150):
        writer.write(
            sample_record(
                "seeing_window",
                **common(BASE_NS + second * NS_PER_S, seeing_fwhm_arcsec=1.0),
            )
        )
    settings = WebSettings.model_validate({"paging": {"max_scan_rows": 100}})
    data = StoreData(reader, settings, clock, station_id=STATION)
    data.aggregate_chunk = 40
    items = read_all(data, step=Step.TEN_MINUTES)
    assert items  # the page made progress, so the cursors ended
    assert items[0]["seeing_fwhm_arcsec"] == 1.0


# --- Fields ----------------------------------------------------------------------------------


def test_fields_limits_the_items_to_the_named_fields_and_the_time(data: StoreData) -> None:
    cls = data.record_class("seeing_window")
    wanted = data.parse_fields(cls, "seeing_fwhm_arcsec, r0_cm")
    items = history(data, wanted=wanted).items
    assert set(items[0]) == {"t_utc_ns", "t_utc", "seeing_fwhm_arcsec", "r0_cm", "quality"}
    assert items[0]["quality"] is None  # the first record has no reason, and zenith is not named


def test_fields_filters_the_quality_note_to_the_named_fields(data: StoreData) -> None:
    cls = data.record_class("seeing_window")
    wanted = data.parse_fields(cls, "seeing_fwhm_arcsec,quality")
    row = history(data, start=2, end=3, wanted=wanted).items[0]
    assert row["quality"] == {"seeing_fwhm_arcsec": "too few usable frames"}


def test_fields_can_name_an_array_in_a_raw_history(data: StoreData) -> None:
    cls = data.record_class("seeing_window")
    wanted = data.parse_fields(cls, "motion_psd_freq_hz")
    assert history(data, wanted=wanted).items[0]["motion_psd_freq_hz"] == [1.0, 2.0, 3.0]


def test_an_array_is_null_in_a_bucket_and_the_quality_says_why(data: StoreData) -> None:
    cls = data.record_class("seeing_window")
    wanted = data.parse_fields(cls, "motion_psd_freq_hz")
    item = history(data, step=Step.TEN_MINUTES, wanted=wanted).items[0]
    assert item["motion_psd_freq_hz"] is None
    assert item["quality"] == {"motion_psd_freq_hz": "arrays are not aggregated"}


@pytest.mark.parametrize(
    "text", ["", " ,", "Bad Name", "a" * 70, "no_such_field", "seeing_fwhm_arcsec,,r0_cm"]
)
def test_fields_refuses_unknown_or_malformed_names(data: StoreData, text: str) -> None:
    with pytest.raises(InvalidQueryError, match="fields"):
        data.parse_fields(data.record_class("seeing_window"), text)


def test_the_arrays_of_each_record_are_known() -> None:
    from seeingmon.records.base import get_record_type

    assert array_fields(get_record_type("pointing")) == {"attitude"}
    assert array_fields(get_record_type("seeing_window")) == {
        "motion_psd_freq_hz",
        "motion_psd_x_arcsec2_per_hz",
        "motion_psd_y_arcsec2_per_hz",
        "vibration_lines_hz",
    }
    assert array_fields(get_record_type("sky_quality")) == set()


# --- The aggregation function ----------------------------------------------------------------


def health_row(t: int, **fields: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "state": "auto",
        "degraded": False,
        "components": {"core": "ok"},
        "dark_due": False,
    }
    values.update(fields)
    return sample_record("health", **common(t, **values)).to_row()


def test_a_flag_list_is_a_union_and_a_boolean_is_true_when_any_record_is(data: StoreData) -> None:
    rows = [
        health_row(at(0), flags=["low_space"], degraded=False, free_space_gb=10.0),
        health_row(at(1), flags=["sink_backlog"], degraded=True, free_space_gb=8.0),
        health_row(at(2), flags=[], degraded=False, free_space_gb=None),
    ]
    out = aggregate_rows(HealthRecord, rows, at(0))
    assert out["flags"] == ["low_space", "sink_backlog"]
    assert out["degraded"] is True
    assert out["free_space_gb"] == 9.0
    assert out["components"] == {"core": "ok"}
    assert out["n_samples"] == 3
    assert out["t_utc_ns"] == at(0)


def test_a_boolean_that_no_record_sets_stays_null() -> None:
    rows = [health_row(at(0)), health_row(at(1))]
    out = aggregate_rows(HealthRecord, rows, at(0))
    assert out["time_synchronized"] is None
    assert out["quality"] is None


def test_the_mean_has_no_floating_point_noise() -> None:
    rows = [
        health_row(at(minute), cpu_load_1m=value) for minute, value in enumerate([0.1, 0.2, 0.3])
    ]
    assert aggregate_rows(HealthRecord, rows, at(0))["cpu_load_1m"] == 0.2


# --- Events ----------------------------------------------------------------------------------


def events(data: StoreData, **options: Any) -> Any:
    options.setdefault("time_range", window(0, 60))
    options.setdefault("limit", 100)
    return data.events(**options)


def test_events_come_newest_first_by_default(data: StoreData) -> None:
    page = events(data)
    assert [e["t_utc_ns"] for e in page.items] == [at(m) for m in (4, 3, 2, 1, 0)]
    assert page.next_cursor is None


def test_events_can_come_oldest_first(data: StoreData) -> None:
    page = events(data, descending=False)
    assert [e["t_utc_ns"] for e in page.items] == [at(m) for m in (0, 1, 2, 3, 4)]


def test_a_minimum_level_hides_the_milder_events(data: StoreData) -> None:
    warnings = events(data, min_level="warning").items
    assert [e["t_utc_ns"] for e in warnings] == [at(4), at(2), at(1)]
    assert [e["t_utc_ns"] for e in events(data, min_level="error").items] == [at(2)]


def test_a_kind_prefix_selects_events(data: StoreData) -> None:
    items = events(data, kind="scheduler.cloud").items
    assert [e["t_utc_ns"] for e in items] == [at(4), at(1)]
    assert events(data, kind="storage.").items[0]["kind"] == "storage.recovered"
    assert events(data, kind="nothing.like.this").items == []


@pytest.mark.parametrize("descending", [True, False])
@pytest.mark.parametrize("limit", [1, 2, 3, 5])
def test_event_pages_follow_one_another(data: StoreData, descending: bool, limit: int) -> None:
    whole = [e["t_utc_ns"] for e in events(data, descending=descending).items]
    seen: list[int] = []
    cursor = None
    for _ in range(20):
        page = events(data, descending=descending, limit=limit, cursor=cursor)
        seen.extend(e["t_utc_ns"] for e in page.items)
        cursor = page.next_cursor
        if cursor is None:
            break
    assert seen == whole


def test_a_filter_pages_over_the_events_that_it_hides(data: StoreData) -> None:
    pages = []
    cursor = None
    while True:
        page = events(data, min_level="warning", limit=1, cursor=cursor)
        pages.append([e["t_utc_ns"] for e in page.items])
        cursor = page.next_cursor
        if cursor is None:
            break
    assert pages == [[at(4)], [at(2)], [at(1)]]


def test_the_scan_budget_gives_a_cursor_even_when_nothing_matched(
    writer: Store, reader: StoreReader, clock: VirtualClock
) -> None:
    for minute in range(300):
        writer.write(
            sample_record("event", **common(at(minute), level="info", kind="a.b", message="m"))
        )
    settings = WebSettings.model_validate({"paging": {"max_scan_rows": 100}})
    data = StoreData(reader, settings, clock, station_id=STATION)
    page = data.events(time_range=window(0, 400), limit=5, kind="x.", descending=False)
    assert page.items == []
    assert page.next_cursor is not None
    walked = 0
    cursor: str | None = page.next_cursor
    while cursor is not None:
        walked += 1
        cursor = data.events(
            time_range=window(0, 400), limit=5, kind="x.", descending=False, cursor=cursor
        ).next_cursor
    assert walked < 10  # the cursors advanced through the whole range


# --- Cursors, times, and limits --------------------------------------------------------------


def test_a_cursor_round_trips() -> None:
    for t in (0, 1, at(7), 4_102_444_800 * NS_PER_S):
        assert decode_cursor(encode_cursor(t)) == t


@pytest.mark.parametrize(
    "text",
    [
        "",
        "!!!",
        "a" * 200,
        "bm90LWpzb24",  # base64url of "not-json"
        "e30",  # {}
        "W10",  # []
        "eyJ0IjoiMSJ9",  # {"t": "1"}
        "eyJ0IjotMX0",  # {"t": -1}
        "eyJ0IjoxLCJ4IjoyfQ",  # {"t": 1, "x": 2}
        "eyJ0Ijp0cnVlfQ",  # {"t": true}
        "eyJ0IjoxZTMwfQ",  # {"t": 1e30}
    ],
)
def test_a_cursor_that_this_server_did_not_give_is_refused(text: str) -> None:
    with pytest.raises(InvalidQueryError, match="cursor"):
        decode_cursor(text)


def test_a_time_parses_with_a_z_or_a_zero_offset() -> None:
    assert parse_time("2026-10-01T02:00:00Z", "from") == BASE_NS
    assert parse_time("2026-10-01T02:00:00+00:00", "from") == BASE_NS
    assert parse_time("2026-10-01T02:00:00.5Z", "from") == BASE_NS + NS_PER_S // 2


@pytest.mark.parametrize(
    "text",
    [
        "",
        "yesterday",
        "2026-10-01",
        "2026-10-01T02:00:00",
        "2026-10-01T02:00:00+02:00",
        "2026-13-01T02:00:00Z",
        "1969-12-31T23:59:59Z",
        "2101-01-01T00:00:00Z",
    ],
)
def test_a_time_that_is_not_utc_iso_8601_is_refused(text: str) -> None:
    with pytest.raises(InvalidQueryError, match="from"):
        parse_time(text, "from")


def test_the_default_range_is_the_last_day(data: StoreData) -> None:
    now = data.now_ns()
    time_range = data.resolve_range(None, None)
    assert time_range.end_ns == now + 1
    assert time_range.start_ns == now + 1 - 24 * 3600 * NS_PER_S


def test_the_default_start_is_a_day_before_the_end_that_the_client_gives(data: StoreData) -> None:
    time_range = data.resolve_range(None, "2026-10-01T02:00:00Z")
    assert time_range == TimeRange(BASE_NS - 24 * 3600 * NS_PER_S, BASE_NS)


def test_a_range_must_run_forward(data: StoreData) -> None:
    with pytest.raises(InvalidQueryError, match="earlier"):
        data.resolve_range("2026-10-01T02:00:00Z", "2026-10-01T02:00:00Z")
    with pytest.raises(InvalidQueryError, match="earlier"):
        data.resolve_range("2026-10-01T03:00:00Z", "2026-10-01T02:00:00Z")


def test_the_page_size_has_a_default_and_a_maximum(data: StoreData) -> None:
    assert data.resolve_limit(None) == 500
    assert data.resolve_limit(2000) == 2000
    for bad in (0, -1, 2001):
        with pytest.raises(InvalidQueryError, match="between 1 and 2000"):
            data.resolve_limit(bad)


def test_the_steps_have_their_widths() -> None:
    assert [step.seconds for step in Step] == [0, 60, 600, 3600]
    assert [step.value for step in Step] == ["raw", "1m", "10m", "1h"]


# --- Failures --------------------------------------------------------------------------------


def test_a_store_that_cannot_be_read_raises_one_error(data: StoreData, reader: StoreReader) -> None:
    reader.close()
    with pytest.raises(StoreUnavailableError):
        data.latest("seeing_window")
    with pytest.raises(StoreUnavailableError):
        history(data)
    with pytest.raises(StoreUnavailableError):
        events(data)


def test_a_minute_is_a_minute() -> None:
    assert MINUTE_NS == 60 * NS_PER_S
