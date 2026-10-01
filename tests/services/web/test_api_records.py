"""`/seeing`, `/sky`, `/pointing`, and `/events`: the latest record, the history, and the errors."""

from __future__ import annotations

from typing import Any

import pytest

from seeingmon.records.base import field_specs
from tests.services.web.client import TestClient
from tests.services.web.seed import at

API = "/api/v1"
SERIES = [
    ("seeing", "seeing_window"),
    ("sky", "sky_quality"),
    ("pointing", "pointing"),
]


def times(response: Any) -> list[int]:
    return [item["t_utc_ns"] for item in response.json()["items"]]


# --- The latest record -----------------------------------------------------------------------


@pytest.mark.parametrize(("path", "record_type"), SERIES)
def test_the_latest_record_has_every_declared_field_and_an_iso_time(
    client: TestClient, path: str, record_type: str
) -> None:
    response = client.get(f"{API}/{path}/latest")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    record = response.json()
    declared = {spec.name for spec in field_specs(record_type)}
    assert set(record) == declared | {"t_utc"}
    assert record["t_utc"].endswith("Z")
    assert record["station_id"] == "test-station"


def test_the_latest_seeing_record_is_the_newest(client: TestClient) -> None:
    record = client.get(f"{API}/seeing/latest").json()
    assert record["t_utc_ns"] == at(21)
    assert record["t_utc"] == "2026-10-01T02:21:00.000000Z"
    assert record["seeing_fwhm_arcsec"] is None


def test_the_latest_record_keeps_its_arrays(client: TestClient) -> None:
    pointing = client.get(f"{API}/pointing/latest").json()
    assert pointing["attitude"] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    seeing = client.get(f"{API}/seeing/latest").json()
    assert seeing["motion_psd_freq_hz"] == [1.0, 2.0, 3.0]


def test_a_missing_value_is_null_and_the_quality_object_says_why(client: TestClient) -> None:
    record = client.get(f"{API}/seeing/latest").json()
    assert record["heater_duty"] is None
    assert isinstance(record["quality"], dict)
    # the value that the server withholds is null with a note
    assert record["zenith_angle_deg"] is None
    assert record["quality"]["zenith_angle_deg"] == "withheld"


@pytest.mark.parametrize(("path", "record_type"), SERIES)
def test_the_latest_record_of_an_empty_store_is_a_404_with_the_error_shape(
    empty_client: TestClient, path: str, record_type: str
) -> None:
    response = empty_client.get(f"{API}/{path}/latest")
    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "no_data",
            "message": f"The store holds no {record_type} record yet.",
            "details": None,
        }
    }


# --- The history -----------------------------------------------------------------------------


@pytest.mark.parametrize(("path", "record_type"), SERIES)
def test_the_history_has_the_documented_shape(
    client: TestClient, path: str, record_type: str
) -> None:
    response = client.get(f"{API}/{path}")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"record_type", "step", "from", "to", "now", "items", "next_cursor"}
    assert body["record_type"] == record_type
    assert body["step"] == "raw"
    assert body["next_cursor"] is None
    assert body["now"] == "2026-10-01T03:00:00.000000Z"
    assert body["items"]


def test_the_default_range_is_the_last_24_hours(client: TestClient) -> None:
    body = client.get(f"{API}/seeing").json()
    assert body["to"] == "2026-10-01T03:00:00.000000Z"  # now, with one nanosecond so now is in
    assert body["from"] == "2026-09-30T03:00:00.000000Z"


def test_the_range_includes_from_and_excludes_to(client: TestClient) -> None:
    response = client.get(
        f"{API}/seeing", params={"from": "2026-10-01T02:03:00Z", "to": "2026-10-01T02:05:00Z"}
    )
    assert times(response) == [at(3), at(4)]
    assert response.json()["from"] == "2026-10-01T02:03:00.000000Z"
    assert response.json()["to"] == "2026-10-01T02:05:00.000000Z"


def test_a_raw_history_row_has_no_arrays_but_has_the_iso_time(client: TestClient) -> None:
    row = client.get(f"{API}/seeing").json()["items"][0]
    assert "motion_psd_freq_hz" not in row
    assert row["t_utc"] == "2026-10-01T02:00:00.000000Z"
    assert row["flags"] == []


def test_a_ten_minute_step_gives_the_means_that_were_computed_by_hand(client: TestClient) -> None:
    body = client.get(f"{API}/seeing", params={"step": "10m"}).json()
    assert body["step"] == "10m"
    first, second, third = body["items"]
    assert first["t_utc"] == "2026-10-01T02:00:00.000000Z"
    assert first["seeing_fwhm_arcsec"] == 1.4  # (1.0 + 1.2 + 1.4 + 1.6 + 1.8) / 5
    assert first["r0_cm"] == 7.52  # (10.1 + 8.4 + 7.2 + 6.3 + 5.6) / 5
    assert first["flags"] == ["cloud", "twilight", "vibration"]  # the union
    assert first["n_samples"] == 6
    assert second["seeing_fwhm_arcsec"] == 2.0
    assert third["seeing_fwhm_arcsec"] is None
    assert third["quality"]["seeing_fwhm_arcsec"] == "the star is not detected"


def test_an_hour_step_gives_one_bucket(client: TestClient) -> None:
    items = client.get(f"{API}/seeing", params={"step": "1h"}).json()["items"]
    assert len(items) == 1
    assert items[0]["seeing_fwhm_arcsec"] == 1.5  # (1.0 + 1.2 + 1.4 + 1.6 + 1.8 + 2.0) / 6
    assert items[0]["n_samples"] == 10


def test_a_one_minute_step_gives_one_bucket_for_each_record(client: TestClient) -> None:
    items = client.get(f"{API}/seeing", params={"step": "1m"}).json()["items"]
    assert len(items) == 10
    assert {item["n_samples"] for item in items} == {1}


def test_pages_follow_the_cursor_to_the_end(client: TestClient) -> None:
    seen: list[int] = []
    params: dict[str, Any] = {"limit": 3}
    for _ in range(10):
        body = client.get(f"{API}/seeing", params=params).json()
        seen.extend(item["t_utc_ns"] for item in body["items"])
        if body["next_cursor"] is None:
            break
        params = {"limit": 3, "cursor": body["next_cursor"]}
    assert seen == [at(m) for m in (0, 1, 2, 3, 4, 5, 10, 11, 20, 21)]


def test_fields_limits_each_item_to_the_named_fields(client: TestClient) -> None:
    body = client.get(f"{API}/pointing", params={"fields": "offset_arcmin,roll_deg"}).json()
    assert set(body["items"][0]) == {"t_utc_ns", "t_utc", "offset_arcmin", "roll_deg", "quality"}
    assert [item["offset_arcmin"] for item in body["items"]] == [0.1, 0.3, 0.5]


def test_fields_may_name_an_array_of_a_raw_history(client: TestClient) -> None:
    body = client.get(f"{API}/pointing", params={"fields": "attitude"}).json()
    assert body["items"][0]["attitude"] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


@pytest.mark.parametrize(
    ("params", "field"),
    [
        ({"from": "yesterday"}, "from"),
        ({"to": "2026-10-01"}, "to"),
        ({"from": "2026-10-01T03:00:00Z", "to": "2026-10-01T02:00:00Z"}, "earlier"),
        ({"from": "2026-10-01T02:00:00+02:00"}, "from"),
        ({"step": "2h"}, "step"),
        ({"step": "RAW"}, "step"),
        ({"limit": 0}, "limit"),
        ({"limit": -3}, "limit"),
        ({"limit": 20000}, "limit"),
        ({"limit": 2001}, "limit"),
        ({"limit": "many"}, "limit"),
        ({"cursor": "not-a-cursor!"}, "cursor"),
        ({"cursor": "e30"}, "cursor"),
        ({"fields": "no_such_field"}, "fields"),
        ({"fields": "x" * 2000}, "fields"),
    ],
)
def test_a_bad_parameter_is_a_422_with_the_error_shape_and_no_trace(
    client: TestClient, params: dict[str, Any], field: str
) -> None:
    response = client.get(f"{API}/seeing", params=params)
    assert response.status_code == 422
    body = response.json()
    assert set(body) == {"error"}
    assert body["error"]["code"] == "invalid_request"
    assert field in str(body["error"])
    assert "Traceback" not in response.text
    assert ".py" not in response.text


@pytest.mark.parametrize("name", ["from", "to", "cursor", "fields", "step", "limit"])
def test_a_bad_parameter_is_not_echoed_in_the_answer(client: TestClient, name: str) -> None:
    sent = "<script>alert(1)</script>"
    response = client.get(f"{API}/seeing", params={name: sent})
    assert response.status_code == 422
    assert sent not in response.text
    assert "script" not in response.text


def test_a_bad_value_names_the_field_in_the_details(client: TestClient) -> None:
    body = client.get(f"{API}/seeing", params={"limit": 0}).json()
    assert body["error"]["details"] == [
        {
            "field": "query.limit",
            "message": "Input should be greater than or equal to 1",
            "type": "greater_than_equal",
        }
    ]


# --- Events ----------------------------------------------------------------------------------


def test_events_have_the_documented_shape_and_come_newest_first(client: TestClient) -> None:
    body = client.get(f"{API}/events").json()
    assert set(body) == {"record_type", "step", "from", "to", "now", "items", "next_cursor"}
    assert body["record_type"] == "event"
    assert [e["t_utc_ns"] for e in body["items"]] == [at(m) for m in (4, 3, 2, 1, 0)]
    declared = {spec.name for spec in field_specs("event")}
    assert set(body["items"][0]) == declared | {"t_utc"}


def test_events_can_come_oldest_first(client: TestClient) -> None:
    body = client.get(f"{API}/events", params={"order": "asc"}).json()
    assert [e["t_utc_ns"] for e in body["items"]] == [at(m) for m in (0, 1, 2, 3, 4)]


def test_events_filter_by_level_and_kind(client: TestClient) -> None:
    warnings = client.get(f"{API}/events", params={"level": "warning"}).json()["items"]
    assert {e["level"] for e in warnings} == {"warning", "error"}
    errors = client.get(f"{API}/events", params={"level": "error"}).json()["items"]
    assert [e["kind"] for e in errors] == ["scheduler.fault"]
    cloud = client.get(f"{API}/events", params={"kind": "scheduler.cloud"}).json()["items"]
    assert [e["message"] for e in cloud] == ["The clouds went away.", "Clouds crossed the star."]


def test_events_page_through_the_cursor(client: TestClient) -> None:
    first = client.get(f"{API}/events", params={"limit": 2}).json()
    assert len(first["items"]) == 2
    assert first["next_cursor"] is not None
    second = client.get(f"{API}/events", params={"limit": 2, "cursor": first["next_cursor"]}).json()
    assert [e["t_utc_ns"] for e in second["items"]] == [at(2), at(1)]


@pytest.mark.parametrize(
    "params",
    [
        {"level": "fatal"},
        {"order": "up"},
        {"kind": "Bad Kind"},
        {"kind": "x" * 100},
        {"limit": 0},
        {"cursor": "!"},
        {"from": "never"},
    ],
)
def test_a_bad_event_parameter_is_a_422(client: TestClient, params: dict[str, Any]) -> None:
    response = client.get(f"{API}/events", params=params)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
