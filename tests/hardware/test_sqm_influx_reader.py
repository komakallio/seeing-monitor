"""The InfluxDB source of the SQM-LE on a fake server: requests, readings, failures, and secrecy.

The server is `tests.sinks.fake_influx.FakeInfluxServer`, on the loopback interface. The tests use
a `VirtualClock`, so a wait between polls costs no real time. Only a request that must time out
waits in real time, for a fraction of a second.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from collections.abc import Iterator
from typing import Any

import pytest

from seeingmon.clock import NS_PER_S, VirtualClock
from seeingmon.config import ConfigError
from seeingmon.hardware import sqm_influx
from seeingmon.hardware.events import HardwareEvent
from seeingmon.hardware.sqm import SqmConfig, SqmInfluxConfig, SqmReader, backoff_delay_s
from seeingmon.hardware.sqm_influx import (
    InfluxBadRequestError,
    InfluxNoDataError,
    InfluxParseError,
    InfluxQueryError,
    InfluxStaleError,
    InfluxUnauthorizedError,
    InfluxUnreachableError,
    SqmInfluxReader,
)
from seeingmon.records.reference import ReferenceRecord
from seeingmon.sinks.influx import make_opener
from tests.hardware.influx_replies import (
    MAGNITUDE,
    TEMPERATURE,
    V1_EMPTY,
    V2_EMPTY,
    ago,
    v1_reply,
    v2_csv,
)
from tests.sinks.fake_influx import Drop, FakeInfluxServer, Hang, Reply, Respond

TOKEN = "example-token-value"
USER = "example-user"
CODE = "example-code-value"  # the password of a version 1 server


@pytest.fixture
def server() -> Iterator[FakeInfluxServer]:
    with FakeInfluxServer() as running:
        yield running


def table(server: FakeInfluxServer, *, version: int = 2, **overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "endpoint": server.url,
        "timeout_s": 5.0,
        "measurement": "sqm",
        "field": "mag",
        "temperature_field": "temp",
        "tags": {"unit": "roof"},
    }
    if version == 2:
        values.update(org="example-org", bucket="example-bucket")
    else:
        values.update(version=1, database="example-db", username=USER)
    values.update(overrides)
    return values


def make_reader(
    server: FakeInfluxServer,
    *,
    version: int = 2,
    events: list[HardwareEvent] | None = None,
    influx: dict[str, Any] | None = None,
    clock: VirtualClock | None = None,
    **sqm: Any,
) -> tuple[SqmInfluxReader, VirtualClock]:
    used = clock or VirtualClock()
    config = SqmConfig(
        enabled=True,
        source="influx",
        influx=SqmInfluxConfig(**table(server, version=version, **(influx or {}))),
        **sqm,
    )
    reader = SqmInfluxReader(
        config,
        clock=used,
        station_id="station-1",
        profile_id="profile-1",
        token=TOKEN if version == 2 else None,
        password=CODE if version == 1 else None,
        opener=make_opener(use_environment_proxies=False),
        on_event=None if events is None else events.append,
    )
    return reader, used


def v2_reply(clock: VirtualClock, age_s: float = 5.0, *, temperature: bool = True) -> Reply:
    """A reply with a point that is `age_s` old on `clock`, as version 2 sends it."""
    t_utc_ns = ago(clock.utc_ns(), age_s)
    rows = [("mag", t_utc_ns, MAGNITUDE)]
    if temperature:
        rows.append(("temp", t_utc_ns, TEMPERATURE))
    return Reply(200, v2_csv(rows))


def v1_point(clock: VirtualClock, age_s: float = 5.0) -> Reply:
    """A reply with a point that is `age_s` old on `clock`, as version 1 sends it."""
    return Reply(200, v1_reply([[ago(clock.utc_ns(), age_s), MAGNITUDE, TEMPERATURE]]))


class TestRequests:
    def test_a_version_2_poll_posts_a_flux_program_with_the_token_header(
        self, server: FakeInfluxServer
    ) -> None:
        reader, clock = make_reader(server)
        server.script(v2_reply(clock))
        assert reader.poll() is not None
        (request,) = server.requests
        assert request.method == "POST"
        assert request.path == "/api/v2/query"
        assert request.query == {"org": ["example-org"]}
        assert request.headers["authorization"] == f"Token {TOKEN}"
        assert request.headers["content-type"] == "application/vnd.flux"
        assert request.headers["accept"] == "application/csv"
        assert request.body == (
            'from(bucket: "example-bucket")\n'
            "  |> range(start: -3600s)\n"
            '  |> filter(fn: (r) => r._measurement == "sqm")\n'
            '  |> filter(fn: (r) => r._field == "mag" or r._field == "temp")\n'
            '  |> filter(fn: (r) => r["unit"] == "roof")\n'
            "  |> last()\n"
        )

    def test_a_version_1_poll_gets_an_influxql_query_with_basic_authentication(
        self, server: FakeInfluxServer
    ) -> None:
        reader, clock = make_reader(server, version=1)
        server.script(v1_point(clock))
        assert reader.poll() is not None
        (request,) = server.requests
        assert request.method == "GET"
        assert request.path == "/query"
        assert set(request.query) == {"db", "epoch", "q"}  # never `u` and `p`
        assert request.query["db"] == ["example-db"]
        assert request.query["epoch"] == ["ns"]
        assert request.query["q"] == [
            'SELECT "mag", "temp" FROM "sqm" WHERE time > now() - 3600s AND "mag" > -1000 '
            "AND \"unit\" = 'roof' ORDER BY time DESC LIMIT 1"
        ]
        expected = base64.b64encode(f"{USER}:{CODE}".encode()).decode()
        assert request.headers["authorization"] == f"Basic {expected}"
        assert request.headers["accept"] == "application/json"
        assert request.body == ""

    def test_the_credentials_never_travel_in_the_address(self, server: FakeInfluxServer) -> None:
        for version, make_reply in ((1, v1_point), (2, v2_reply)):
            reader, clock = make_reader(server, version=version)
            server.script(make_reply(clock))
            reader.poll()
        for request in server.requests:
            assert TOKEN not in request.path + str(request.query)
            assert CODE not in request.path + str(request.query)
            assert USER not in request.path + str(request.query)

    def test_version_1_without_a_user_sends_no_authorization_header(
        self, server: FakeInfluxServer
    ) -> None:
        reader, clock = make_reader(server, version=1, influx={"username": None})
        server.script(v1_point(clock))
        reader.poll()
        assert "authorization" not in server.requests[0].headers

    def test_version_2_without_a_token_sends_no_authorization_header(
        self, server: FakeInfluxServer
    ) -> None:
        config = SqmConfig(enabled=True, source="influx", influx=SqmInfluxConfig(**table(server)))
        clock = VirtualClock()
        reader = SqmInfluxReader(
            config,
            clock=clock,
            station_id="s",
            profile_id="p",
            opener=make_opener(use_environment_proxies=False),
        )
        server.script(v2_reply(clock))
        assert reader.poll() is not None
        assert "authorization" not in server.requests[0].headers

    def test_a_retention_policy_qualifies_the_measurement_in_the_query(
        self, server: FakeInfluxServer
    ) -> None:
        reader, clock = make_reader(server, version=1, influx={"retention_policy": "autogen"})
        server.script(v1_point(clock))
        reader.poll()
        assert 'FROM "autogen"."sqm" WHERE' in server.requests[0].query["q"][0]

    def test_a_trailing_slash_in_the_endpoint_does_not_double(
        self, server: FakeInfluxServer
    ) -> None:
        reader, clock = make_reader(server, influx={"endpoint": server.url + "/"})
        server.script(v2_reply(clock))
        reader.poll()
        assert server.requests[0].path == "/api/v2/query"

    def test_each_poll_sends_one_request(self, server: FakeInfluxServer) -> None:
        reader, clock = make_reader(server)
        server.script(v2_reply(clock), v2_reply(clock, 4.0))
        reader.poll()
        reader.poll()
        assert len(server.requests) == 2

    def test_the_lookback_sets_the_window_of_the_query(self, server: FakeInfluxServer) -> None:
        reader, clock = make_reader(server, influx={"lookback_s": 900.0, "max_age_s": 300.0})
        server.script(v2_reply(clock))
        reader.poll()
        assert "range(start: -900s)" in server.requests[0].body


class TestReadings:
    def test_a_version_2_point_becomes_a_reference_record(self, server: FakeInfluxServer) -> None:
        reader, clock = make_reader(server, altitude_deg=45.0, azimuth_deg=0.0)
        point_ns = ago(clock.utc_ns(), 5.0)
        server.script(v2_reply(clock, 5.0))
        record = reader.poll()
        assert isinstance(record, ReferenceRecord)
        assert record.t_utc_ns == point_ns  # the time of the point, not the time of the poll
        assert record.t_utc_ns != clock.utc_ns()
        assert record.instrument == "sqm_le"
        assert record.source == "fixed"
        assert record.value_mag_arcsec2 == MAGNITUDE
        assert record.temperature_c == TEMPERATURE
        assert (record.altitude_deg, record.azimuth_deg) == (45.0, 0.0)
        assert (record.station_id, record.profile_id) == ("station-1", "profile-1")
        assert record.provenance == {"reader": "sqm-influx-1", "api": "2"}

    def test_a_version_1_point_becomes_a_reference_record(self, server: FakeInfluxServer) -> None:
        reader, clock = make_reader(server, version=1)
        point_ns = ago(clock.utc_ns(), 7.0)
        server.script(v1_point(clock, 7.0))
        record = reader.poll()
        assert record is not None
        assert record.t_utc_ns == point_ns
        assert (record.value_mag_arcsec2, record.temperature_c) == (MAGNITUDE, TEMPERATURE)
        assert record.provenance == {"reader": "sqm-influx-1", "api": "1"}

    def test_the_pointing_and_the_temperature_are_optional(self, server: FakeInfluxServer) -> None:
        reader, clock = make_reader(server, influx={"temperature_field": None})
        server.script(v2_reply(clock, temperature=False))
        record = reader.poll()
        assert record is not None
        assert (record.temperature_c, record.altitude_deg, record.azimuth_deg) == (None, None, None)

    def test_the_instrument_name_follows_the_configuration(self, server: FakeInfluxServer) -> None:
        reader, clock = make_reader(server, instrument="sqm-le")
        server.script(v2_reply(clock))
        record = reader.poll()
        assert record is not None
        assert record.instrument == "sqm-le"

    def test_a_reply_with_one_field_gives_a_record_without_a_temperature(
        self, server: FakeInfluxServer
    ) -> None:
        reader, clock = make_reader(server)  # the reader asks for two fields
        server.script(v2_reply(clock, temperature=False))
        record = reader.poll()
        assert record is not None
        assert record.temperature_c is None

    def test_a_temperature_from_another_time_is_not_the_temperature_of_the_point(
        self, server: FakeInfluxServer
    ) -> None:
        reader, clock = make_reader(server)
        now = clock.utc_ns()
        server.script(Reply(200, v2_csv([("mag", ago(now, 5), 21.4), ("temp", ago(now, 9), 3.4)])))
        record = reader.poll()
        assert record is not None
        assert record.temperature_c is None

    def test_two_tables_with_two_fields_in_separate_blocks_read_the_same(
        self, server: FakeInfluxServer
    ) -> None:
        reader, clock = make_reader(server)
        now = clock.utc_ns()
        rows = [("mag", ago(now, 5), 21.4), ("temp", ago(now, 5), 3.4)]
        server.script(Reply(200, v2_csv(rows, separate_tables=True)))
        record = reader.poll()
        assert record is not None
        assert (record.value_mag_arcsec2, record.temperature_c) == (21.4, 3.4)

    def test_when_the_tags_match_two_series_the_newest_point_wins(
        self, server: FakeInfluxServer
    ) -> None:
        reader, clock = make_reader(server)
        now = clock.utc_ns()
        server.script(Reply(200, v2_csv([("mag", ago(now, 60), 20.0), ("mag", ago(now, 5), 21.0)])))
        record = reader.poll()
        assert record is not None
        assert (record.t_utc_ns, record.value_mag_arcsec2) == (ago(now, 5), 21.0)

    def test_both_readers_have_the_interface_of_core(self, server: FakeInfluxServer) -> None:
        reader, _ = make_reader(server)
        interface: SqmReader = reader  # the type check is the test
        assert interface.failures == 0
        assert interface.delay_s == 60.0

    def test_the_time_stamp_of_a_version_1_point_keeps_its_nanoseconds(
        self, server: FakeInfluxServer
    ) -> None:
        reader, clock = make_reader(server, version=1)
        stamp = clock.utc_ns() - 5 * NS_PER_S - 123_456_789
        server.script(Reply(200, v1_reply([[stamp, 21.4, None]])))
        record = reader.poll()
        assert record is not None
        assert record.t_utc_ns == stamp


class TestFreshness:
    def test_a_point_that_repeats_the_last_time_stamp_gives_no_record_and_no_failure(
        self, server: FakeInfluxServer
    ) -> None:
        events: list[HardwareEvent] = []
        reader, clock = make_reader(server, events=events)
        point = Reply(200, v2_csv([("mag", ago(clock.utc_ns(), 5), 21.4)]))
        server.script(point, point, point)
        assert reader.poll() is not None
        clock.advance(60.0)
        assert reader.poll() is None
        assert reader.failures == 0
        assert reader.delay_s == 60.0
        clock.advance(60.0)
        assert reader.poll() is None
        assert events == []

    def test_a_newer_point_gives_a_new_record(self, server: FakeInfluxServer) -> None:
        reader, clock = make_reader(server)
        server.script(v2_reply(clock, 30.0))
        first = reader.poll()
        clock.advance(60.0)
        server.script(v2_reply(clock, 5.0))
        second = reader.poll()
        assert first is not None
        assert second is not None
        assert second.t_utc_ns > first.t_utc_ns

    def test_only_the_last_time_stamp_counts_as_a_repeat_so_an_earlier_point_is_a_new_reading(
        self, server: FakeInfluxServer
    ) -> None:
        reader, clock = make_reader(server)
        server.script(v2_reply(clock, 5.0), v2_reply(clock, 20.0))
        assert reader.poll() is not None
        earlier = reader.poll()
        assert earlier is not None
        assert earlier.t_utc_ns == ago(clock.utc_ns(), 20.0)
        assert reader.failures == 0

    def test_a_point_with_a_wrong_time_stamp_in_the_future_does_not_hide_the_next_readings(
        self, server: FakeInfluxServer
    ) -> None:
        reader, clock = make_reader(server)
        server.script(v2_reply(clock, -86_400.0), v2_reply(clock, 5.0))  # a day ahead, then now
        first, second = reader.poll(), reader.poll()
        assert first is not None
        assert second is not None
        assert second.t_utc_ns < first.t_utc_ns

    def test_a_point_older_than_max_age_is_stale_and_a_failure(
        self, server: FakeInfluxServer
    ) -> None:
        events: list[HardwareEvent] = []
        reader, clock = make_reader(server, events=events)
        server.script(v2_reply(clock, 601.0))
        assert reader.poll() is None
        assert reader.failures == 1
        assert [(e.kind, e.detail) for e in events] == [("sqm.read_failed", {"cause": "Stale"})]

    def test_the_age_is_measured_on_the_clock_of_the_reader_and_the_limit_is_inclusive(
        self, server: FakeInfluxServer
    ) -> None:
        reader, clock = make_reader(server)
        point = Reply(200, v2_csv([("mag", ago(clock.utc_ns(), 599.0), 21.4)]))
        server.script(point, point)
        assert reader.poll() is not None  # 599 s old, and the limit is 600 s
        clock.advance(1.0)
        assert reader.poll() is None  # 600 s old: the same point, so no record, and no failure
        assert reader.failures == 0
        clock.advance(0.5)
        server.script(point)
        assert reader.poll() is None  # 600.5 s old
        assert reader.failures == 1

    def test_max_age_follows_the_configuration(self, server: FakeInfluxServer) -> None:
        reader, clock = make_reader(server, influx={"max_age_s": 30.0, "lookback_s": 300.0})
        server.script(v2_reply(clock, 29.0), v2_reply(clock, 31.0))
        assert reader.poll() is not None
        clock.advance(1.0)
        assert reader.poll() is None
        assert reader.failures == 1

    def test_the_same_point_that_stays_for_too_long_goes_stale(
        self, server: FakeInfluxServer
    ) -> None:
        events: list[HardwareEvent] = []
        reader, clock = make_reader(server, events=events)
        point = Reply(200, v2_csv([("mag", ago(clock.utc_ns(), 5), 21.4)]))
        server.default = point
        assert reader.poll() is not None
        for _ in range(11):
            clock.advance(60.0)
            reader.poll()
        assert reader.failures >= 1
        assert events[0].detail == {"cause": "Stale"}

    def test_a_reply_with_no_point_in_the_window_is_no_data(self, server: FakeInfluxServer) -> None:
        events: list[HardwareEvent] = []
        reader, _ = make_reader(server, events=events)
        server.script(Reply(200, V2_EMPTY))
        assert reader.poll() is None
        assert reader.failures == 1
        assert events[0].detail == {"cause": "NoData"}

    @pytest.mark.parametrize(
        ("version", "reply"),
        [
            (1, Reply(200, V1_EMPTY)),
            (1, Reply(200, v1_reply([]))),
            (2, Reply(200, V2_EMPTY)),
            (2, Reply(200, v2_csv([]))),
            (2, Reply(200, v2_csv([("temp", 1_767_261_912_123_456_789, 3.4)]))),
            (2, Reply(204, "")),
        ],
    )
    def test_each_kind_of_empty_reply_is_no_data(
        self, server: FakeInfluxServer, version: int, reply: Reply
    ) -> None:
        reader, _ = make_reader(server, version=version)
        server.script(reply)
        with pytest.raises(InfluxNoDataError, match="holds no reading"):
            reader.read()

    def test_the_stale_error_says_how_old_the_point_is(self, server: FakeInfluxServer) -> None:
        reader, clock = make_reader(server)
        server.script(v2_reply(clock, 1800.0))
        with pytest.raises(InfluxStaleError, match=r"1800 s old, and max_age_s is 600 s"):
            reader.read()

    def test_a_good_poll_after_a_failure_streak_recovers_even_when_the_point_is_not_new(
        self, server: FakeInfluxServer
    ) -> None:
        events: list[HardwareEvent] = []
        reader, clock = make_reader(server, events=events)
        point = Reply(200, v2_csv([("mag", ago(clock.utc_ns(), 5), 21.4)]))
        server.script(point, Reply(503), Reply(503), point)
        assert reader.poll() is not None
        assert reader.poll() is None
        assert reader.poll() is None
        assert reader.failures == 2
        assert reader.poll() is None  # the same point: no record, but the server answers again
        assert reader.failures == 0
        assert [e.kind for e in events] == ["sqm.read_failed", "sqm.recovered"]
        assert events[1].detail == {"failures": 2}


class TestFailures:
    @pytest.mark.parametrize("status", [401, 403])
    @pytest.mark.parametrize("version", [1, 2])
    def test_a_refused_credential_is_unauthorized(
        self, server: FakeInfluxServer, version: int, status: int
    ) -> None:
        reader, _ = make_reader(server, version=version)
        server.script(Reply(status, '{"code": "unauthorized", "message": "unauthorized access"}'))
        with pytest.raises(InfluxUnauthorizedError) as caught:
            reader.read()
        assert caught.value.cause == "Unauthorized"
        assert f"HTTP {status}" in str(caught.value)
        assert "unauthorized access" in str(caught.value)
        assert ("check the token" if status == 401 else "may not read") in str(caught.value)

    @pytest.mark.parametrize("status", [500, 502, 503, 504, 408, 429])
    def test_a_server_error_a_timeout_status_and_throttling_are_unreachable(
        self, server: FakeInfluxServer, status: int
    ) -> None:
        reader, _ = make_reader(server)
        server.script(Reply(status, '{"message": "try later"}'))
        with pytest.raises(InfluxUnreachableError) as caught:
            reader.read()
        assert caught.value.cause == "Unreachable"
        assert f"HTTP {status}" in str(caught.value)
        assert "try later" in str(caught.value)

    @pytest.mark.parametrize("status", [400, 404, 405, 413, 422])
    def test_any_other_client_error_is_a_bad_request_with_the_server_message(
        self, server: FakeInfluxServer, status: int
    ) -> None:
        reader, _ = make_reader(server)
        server.script(Reply(status, '{"code": "invalid", "message": "the query is wrong"}'))
        with pytest.raises(InfluxBadRequestError) as caught:
            reader.read()
        assert caught.value.cause == "BadRequest"
        assert f"HTTP {status}" in str(caught.value)
        assert "the query is wrong" in str(caught.value)

    def test_a_404_says_what_to_check(self, server: FakeInfluxServer) -> None:
        reader, _ = make_reader(server)
        server.script(Reply(404, '{"message": "not found"}'))
        with pytest.raises(InfluxBadRequestError, match="check the endpoint, the organization"):
            reader.read()

    @pytest.mark.parametrize("status", [301, 302, 307, 308])
    def test_a_redirect_is_a_bad_request_and_the_reader_never_follows_it(
        self, server: FakeInfluxServer, status: int
    ) -> None:
        reader, _ = make_reader(server)
        server.script(Reply(status, headers={"Location": server.url + "/elsewhere"}))
        with pytest.raises(InfluxBadRequestError, match="set the endpoint"):
            reader.read()
        assert len(server.requests) == 1

    def test_a_long_error_body_gives_a_short_one_line_message(
        self, server: FakeInfluxServer
    ) -> None:
        reader, _ = make_reader(server)
        server.script(Reply(400, ("a line\n" + "x" * 300 + "\n") * 40))
        with pytest.raises(InfluxBadRequestError) as caught:
            reader.read()
        assert "\n" not in str(caught.value)
        assert len(str(caught.value)) < 400

    def test_a_server_that_is_gone_is_unreachable(self) -> None:
        with FakeInfluxServer() as gone:
            reader, _ = make_reader(gone, influx={"timeout_s": 1.0})
        with pytest.raises(InfluxUnreachableError, match="did not answer"):
            reader.read()  # nothing listens on the port any more

    def test_a_connection_closed_without_a_reply_is_unreachable(
        self, server: FakeInfluxServer
    ) -> None:
        reader, _ = make_reader(server)
        server.script(Drop())
        with pytest.raises(InfluxUnreachableError):
            reader.read()

    def test_a_reply_slower_than_the_timeout_is_unreachable_and_ends_at_the_timeout(
        self, server: FakeInfluxServer
    ) -> None:
        events: list[HardwareEvent] = []
        reader, _ = make_reader(server, events=events, influx={"timeout_s": 0.3})
        server.script(Hang(10))
        started = time.monotonic()
        assert reader.poll() is None
        assert 0.2 < time.monotonic() - started < 5
        assert reader.failures == 1
        assert events[0].detail == {"cause": "Unreachable"}

    def test_a_slow_reply_inside_the_timeout_is_read(self, server: FakeInfluxServer) -> None:
        reader, _ = make_reader(server, influx={"timeout_s": 5.0})
        server.script(Hang(0.2))  # the server answers 204 after 0.2 s, which is no data
        with pytest.raises(InfluxNoDataError):
            reader.read()

    @pytest.mark.parametrize(
        ("version", "body"),
        [
            (1, "not json"),
            (1, "{}"),
            (1, "[1, 2]"),
            (1, '{"results":[{"series":[{"columns":["time","mag"],"values":[[1,"x"]]}]}]}'),
            (2, "garbage that is not csv"),
            (2, "<html>502 Bad Gateway</html>"),
            (2, "#datatype,string,long\r\n"),
            (2, ",result,table,_time,_value,_field\r\n,,0,not-a-time,1,mag\r\n"),
        ],
    )
    def test_a_malformed_reply_is_a_parse_error(
        self, server: FakeInfluxServer, version: int, body: str
    ) -> None:
        events: list[HardwareEvent] = []
        reader, _ = make_reader(server, version=version, events=events)
        server.script(Reply(200, body))
        assert reader.poll() is None
        assert events[0].detail == {"cause": "Parse"}
        server.script(Reply(200, body))
        with pytest.raises(InfluxParseError):
            reader.read()

    def test_a_magnitude_outside_the_range_of_a_sky_meter_is_a_parse_error(
        self, server: FakeInfluxServer
    ) -> None:
        reader, clock = make_reader(server)
        server.script(Reply(200, v2_csv([("mag", ago(clock.utc_ns(), 5), 99.99)])))
        with pytest.raises(InfluxParseError, match="magnitude"):
            reader.read()

    def test_an_error_of_the_server_in_a_reply_of_200_is_a_bad_request(
        self, server: FakeInfluxServer
    ) -> None:
        reader, _ = make_reader(server, version=1)
        server.script(Reply(200, '{"results":[{"statement_id":0,"error":"database not found"}]}'))
        with pytest.raises(InfluxBadRequestError, match="database not found"):
            reader.read()
        reader2, _ = make_reader(server)
        server.script(
            Reply(
                200,
                "#datatype,string,string\r\n#group,true,true\r\n#default,,\r\n"
                ",error,reference\r\n,failed to run the program,\r\n",
            )
        )
        with pytest.raises(InfluxBadRequestError, match="failed to run the program"):
            reader2.read()

    def test_a_reply_larger_than_the_limit_is_a_parse_error(
        self, server: FakeInfluxServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sqm_influx, "MAX_REPLY_BYTES", 200)
        reader, clock = make_reader(server)
        server.script(Reply(200, v2_csv([("mag", ago(clock.utc_ns(), 5), 21.4)] * 10)))
        with pytest.raises(InfluxParseError, match="too large"):
            reader.read()

    def test_a_reply_that_is_not_text_is_a_parse_error(self, server: FakeInfluxServer) -> None:
        reader, _ = make_reader(server)
        server.script(Reply(200, b"\xff\xfe is not UTF-8"))
        with pytest.raises(InfluxParseError, match="not text"):
            reader.read()

    def test_a_poll_never_raises(self, server: FakeInfluxServer) -> None:
        reader, clock = make_reader(server)
        server.script(Reply(401), Reply(500), Drop(), Reply(200, "junk"), Reply(200, V2_EMPTY))
        server.script(v2_reply(clock, 9999.0))
        for _ in range(6):
            assert reader.poll() is None
        assert reader.failures == 6

    def test_a_bug_counts_as_a_failure_names_only_its_type_and_never_escapes(
        self, server: FakeInfluxServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events: list[HardwareEvent] = []
        reader, _ = make_reader(server, events=events)

        def broken() -> None:
            raise RuntimeError("the words of a bug can hold anything")

        monkeypatch.setattr(reader, "read", broken)
        assert reader.poll() is None
        assert reader.failures == 1  # the health shows the dead reader
        assert events[0].detail == {"cause": "Error"}
        assert "RuntimeError" in events[0].message
        assert "anything" not in events[0].message

    def test_every_error_is_an_sqm_error_with_a_cause(self, server: FakeInfluxServer) -> None:
        causes = {
            InfluxUnreachableError: "Unreachable",
            InfluxUnauthorizedError: "Unauthorized",
            InfluxBadRequestError: "BadRequest",
            InfluxParseError: "Parse",
            InfluxNoDataError: "NoData",
            InfluxStaleError: "Stale",
        }
        for error_type, cause in causes.items():
            assert issubclass(error_type, InfluxQueryError)
            assert error_type.cause == cause
        assert InfluxQueryError.cause == "Error"


class TestBackoffAndEvents:
    def test_failures_back_off_exponentially_up_to_the_cap_and_a_success_resets(
        self, server: FakeInfluxServer
    ) -> None:
        reader, clock = make_reader(server, backoff_initial_s=5.0, backoff_max_s=300.0)
        server.script(*[Reply(503)] * 9)
        delays = []
        for _ in range(9):
            assert reader.poll() is None
            delays.append(reader.delay_s)
        assert delays == [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 300.0, 300.0, 300.0]
        assert reader.failures == 9
        server.script(v2_reply(clock))
        assert reader.poll() is not None
        assert reader.delay_s == 60.0
        assert reader.failures == 0

    def test_the_poll_interval_sets_the_delay_after_a_success(
        self, server: FakeInfluxServer
    ) -> None:
        reader, clock = make_reader(server, poll_interval_s=45.0)
        server.script(v2_reply(clock))
        reader.poll()
        assert reader.delay_s == 45.0

    def test_a_long_outage_never_overflows_the_backoff(self) -> None:
        config = SqmConfig()
        for failures in (1500, 10**9):
            assert backoff_delay_s(config, failures) == 300.0

    def test_the_first_failure_of_a_streak_reports_once_and_the_recovery_reports_too(
        self, server: FakeInfluxServer
    ) -> None:
        events: list[HardwareEvent] = []
        reader, clock = make_reader(server, events=events)
        server.script(Reply(503), Reply(401), Reply(500), v2_reply(clock))
        for _ in range(4):
            reader.poll()
        assert [event.kind for event in events] == ["sqm.read_failed", "sqm.recovered"]
        failed, recovered = events
        assert (failed.level, failed.detail) == ("warning", {"cause": "Unreachable"})
        assert failed.message.startswith("The SQM-LE did not give a usable reading")
        assert (recovered.level, recovered.detail) == ("info", {"failures": 3})
        assert recovered.t_utc_ns == clock.utc_ns()

    def test_a_second_streak_reports_again(self, server: FakeInfluxServer) -> None:
        events: list[HardwareEvent] = []
        reader, clock = make_reader(server, events=events)
        server.script(Reply(503), v2_reply(clock, 5.0), Reply(401))
        for _ in range(3):
            reader.poll()
        assert [(e.kind, e.detail) for e in events] == [
            ("sqm.read_failed", {"cause": "Unreachable"}),
            ("sqm.recovered", {"failures": 1}),
            ("sqm.read_failed", {"cause": "Unauthorized"}),
        ]

    def test_a_callback_that_fails_does_not_disturb_the_reader(
        self, server: FakeInfluxServer
    ) -> None:
        def broken(event: HardwareEvent) -> None:
            raise RuntimeError("the event store is down")

        config = SqmConfig(enabled=True, source="influx", influx=SqmInfluxConfig(**table(server)))
        reader = SqmInfluxReader(
            config,
            clock=VirtualClock(),
            station_id="s",
            profile_id="p",
            opener=make_opener(use_environment_proxies=False),
            on_event=broken,
        )
        server.script(Reply(503))
        assert reader.poll() is None
        assert reader.failures == 1

    def test_the_recovery_after_each_kind_of_failure_on_the_wire(
        self, server: FakeInfluxServer
    ) -> None:
        events: list[HardwareEvent] = []
        reader, clock = make_reader(
            server, events=events, backoff_initial_s=5.0, influx={"timeout_s": 0.3}
        )
        server.script(Drop(), Hang(5), Reply(200, "junk"), Reply(401), v2_reply(clock))
        started = clock.monotonic_ns()
        records: list[ReferenceRecord] = []
        delays: list[float] = []
        while not records:
            record = reader.poll()
            delays.append(reader.delay_s)
            if record is not None:
                records.append(record)
            clock.sleep(reader.delay_s)
        assert delays == [5.0, 10.0, 20.0, 40.0, 60.0]
        assert [event.kind for event in events] == ["sqm.read_failed", "sqm.recovered"]
        assert (clock.monotonic_ns() - started) / NS_PER_S == pytest.approx(5 + 10 + 20 + 40 + 60)


class TestRun:
    def test_run_polls_at_the_interval_and_sleeps_in_slices_of_a_second(
        self, server: FakeInfluxServer
    ) -> None:
        reader, clock = make_reader(server, poll_interval_s=2.5)
        server.default = Respond(lambda request: v2_reply(clock, 1.0))
        sleeps: list[float] = []
        original = clock.sleep

        def recording(seconds: float) -> None:
            sleeps.append(seconds)
            original(seconds)

        clock.sleep = recording  # type: ignore[method-assign]
        records: list[ReferenceRecord] = []
        started = clock.monotonic_ns()
        reader.run(lambda: len(records) >= 3, records.append)
        assert len(records) == 3
        assert sleeps[:3] == [1.0, 1.0, 0.5]
        assert (clock.monotonic_ns() - started) / NS_PER_S == pytest.approx(5.0)

    def test_run_hands_over_only_new_points(self, server: FakeInfluxServer) -> None:
        reader, clock = make_reader(server, poll_interval_s=1.0)
        fixed = Reply(200, v2_csv([("mag", ago(clock.utc_ns(), 1), 21.4)]))
        server.default = fixed
        records: list[ReferenceRecord] = []
        reader.run(lambda: len(server.requests) >= 5, records.append)
        assert len(server.requests) == 5
        assert len(records) == 1

    def test_run_stops_within_a_second_of_the_request(self, server: FakeInfluxServer) -> None:
        reader, clock = make_reader(server, poll_interval_s=60.0)
        server.default = Respond(lambda request: v2_reply(clock, 1.0))
        records: list[ReferenceRecord] = []
        calls: list[int] = []

        def should_stop() -> bool:
            calls.append(1)
            return len(calls) > 6

        started = clock.monotonic_ns()
        reader.run(should_stop, records.append)
        assert len(records) == 1
        assert (clock.monotonic_ns() - started) / NS_PER_S <= 5.0

    def test_close_is_safe_to_call_twice(self, server: FakeInfluxServer) -> None:
        reader, _ = make_reader(server)
        reader.close()
        reader.close()


# --- What never enters a record, an event, a log line, or an exception ---------------------

SENTINELS: dict[str, str] = {
    "org": "SENTINEL-org-4c1d",
    "bucket": "SENTINEL-bucket-8e2f",
    "database": "SENTINEL-database-91ab",
    "retention_policy": "SENTINEL-policy-5d70",
    "username": "SENTINEL-user-13c9",
    "measurement": "SENTINEL-measurement-6b3e",
    "field": "SENTINEL-field-27d4",
    "temperature_field": "SENTINEL-temperature-f08a",
    "tag_name": "SENTINEL-tag-name-b5e6",
    "tag_value": "SENTINEL-tag-value-3a97",
    "token": "SENTINEL-token-d1c2",
    "password": "SENTINEL-password-7f48",  # pragma: allowlist secret
}


SETTINGS = {
    1: (
        "database",
        "retention_policy",
        "username",
        "measurement",
        "field",
        "temperature_field",
        "tag_name",
        "tag_value",
        "password",
    ),
    2: (
        "org",
        "bucket",
        "measurement",
        "field",
        "temperature_field",
        "tag_name",
        "tag_value",
        "token",
    ),
}


def sentinel_reader(
    server: FakeInfluxServer, version: int, events: list[HardwareEvent] | None = None
) -> tuple[SqmInfluxReader, VirtualClock]:
    clock = VirtualClock()
    values: dict[str, Any] = {
        "endpoint": server.url,
        "timeout_s": 0.4,
        "measurement": SENTINELS["measurement"],
        "field": SENTINELS["field"],
        "temperature_field": SENTINELS["temperature_field"],
        "tags": {SENTINELS["tag_name"]: SENTINELS["tag_value"]},
    }
    if version == 2:
        values.update(org=SENTINELS["org"], bucket=SENTINELS["bucket"])
    else:
        values.update(
            version=1,
            database=SENTINELS["database"],
            retention_policy=SENTINELS["retention_policy"],
            username=SENTINELS["username"],
        )
    reader = SqmInfluxReader(
        SqmConfig(enabled=True, source="influx", influx=SqmInfluxConfig(**values)),
        clock=clock,
        station_id="station-1",
        profile_id="profile-1",
        token=SENTINELS["token"] if version == 2 else None,
        password=SENTINELS["password"] if version == 1 else None,
        opener=make_opener(use_environment_proxies=False),
        on_event=None if events is None else events.append,
    )
    return reader, clock


def echo_everything(server: FakeInfluxServer) -> str:
    """A message that repeats every setting, as a server does when it cannot find a name."""
    names = " ".join(SENTINELS.values())
    return f"could not find {names} at {server.url} ({server.url.split('//')[1]})"


class TestSecrecy:
    @staticmethod
    def failures(server: FakeInfluxServer) -> list[Any]:
        echo = echo_everything(server)
        error_table = (
            "#datatype,string,string\r\n#group,true,true\r\n#default,,\r\n"
            f",error,reference\r\n,{echo},\r\n"
        )
        return [
            Reply(400, json.dumps({"code": "invalid", "message": echo})),
            Reply(401, json.dumps({"message": echo})),
            Reply(403, json.dumps({"error": echo})),
            Reply(404, echo),
            Reply(500, echo),
            Reply(503, json.dumps({"message": echo})),
            Reply(302, echo, headers={"Location": server.url + "/" + SENTINELS["bucket"]}),
            Reply(200, json.dumps({"results": [{"statement_id": 0, "error": echo}]})),
            Reply(200, error_table),
            Reply(200, echo),
            Reply(200, f"{echo}\r\n,x\r\n"),
            Reply(200, V2_EMPTY),
            Reply(200, V1_EMPTY),
            Drop(),
            Hang(5),
        ]

    @staticmethod
    def check(text: str, server: FakeInfluxServer, version: int) -> None:
        """No setting that the reader of this version holds appears in `text`."""
        for name in SETTINGS[version]:
            assert SENTINELS[name] not in text, f"the {name} appears in: {text}"
        assert server.url not in text
        assert server.url.split("//")[1] not in text  # the host and the port

    @pytest.mark.parametrize("version", [1, 2])
    def test_no_failure_puts_a_setting_into_an_event_an_exception_or_a_log_line(
        self, server: FakeInfluxServer, version: int, caplog: pytest.LogCaptureFixture
    ) -> None:
        events: list[HardwareEvent] = []
        for behavior in self.failures(server):
            reader, _ = sentinel_reader(server, version, events)
            server.script(behavior)
            with caplog.at_level(logging.DEBUG):
                assert reader.poll() is None
            server.script(behavior)
            with pytest.raises(InfluxQueryError) as caught:
                reader.read()
            text = str(caught.value) + repr(caught.value) + repr(caught.value.args)
            self.check(text, server, version)
            assert repr(reader).startswith("<seeingmon.hardware.sqm_influx.SqmInfluxReader object")
            self.check(repr(reader), server, version)
        assert events, "each failure reported its first event"
        for event in events:
            self.check(repr(event) + event.message + repr(event.detail), server, version)
        self.check(caplog.text, server, version)

    @pytest.mark.parametrize("version", [1, 2])
    def test_a_stale_point_and_a_missing_point_name_nothing_either(
        self, server: FakeInfluxServer, version: int
    ) -> None:
        events: list[HardwareEvent] = []
        reader, clock = sentinel_reader(server, version, events)
        old = ago(clock.utc_ns(), 5000.0)
        if version == 1:
            columns = ("time", SENTINELS["field"], SENTINELS["temperature_field"])
            server.script(
                Reply(200, v1_reply([[old, 21.4, None]], columns=columns)), Reply(200, V1_EMPTY)
            )
        else:
            server.script(
                Reply(200, v2_csv([(SENTINELS["field"], old, 21.4)])), Reply(200, V2_EMPTY)
            )
        reader.poll()
        reader.poll()
        for event in events:
            self.check(repr(event), server, version)
        assert events[0].detail == {"cause": "Stale"}

    @pytest.mark.parametrize("version", [1, 2])
    def test_a_good_poll_puts_no_setting_into_the_record(
        self, server: FakeInfluxServer, version: int
    ) -> None:
        reader, clock = sentinel_reader(server, version)
        point_ns = ago(clock.utc_ns(), 5)
        field, temperature = SENTINELS["field"], SENTINELS["temperature_field"]
        if version == 1:
            columns = ("time", field, temperature)
            server.script(Reply(200, v1_reply([[point_ns, 21.4, 3.4]], columns=columns)))
        else:
            server.script(
                Reply(200, v2_csv([(field, point_ns, 21.4), (temperature, point_ns, 3.4)]))
            )
        record = reader.poll()
        assert record is not None
        self.check(repr(record) + record.model_dump_json(), server, version)

    def test_an_error_message_replaces_a_name_that_the_server_repeats(
        self, server: FakeInfluxServer
    ) -> None:
        reader, _ = sentinel_reader(server, 2)
        server.script(
            Reply(404, json.dumps({"message": f"could not find bucket {SENTINELS['bucket']}"}))
        )
        with pytest.raises(InfluxBadRequestError) as caught:
            reader.read()
        assert "could not find bucket <redacted>" in str(caught.value)

    def test_a_server_that_repeats_the_escaped_form_of_a_name_is_covered_too(
        self, server: FakeInfluxServer
    ) -> None:
        name = 'odd"name\\with\nstuff'
        reader, _ = make_reader(server, influx={"measurement": name})
        server.script(
            Reply(
                400,
                json.dumps({"message": 'bad query: FROM "odd\\"name\\\\with\\nstuff" is wrong'}),
            )
        )
        with pytest.raises(InfluxBadRequestError) as caught:
            reader.read()
        assert "odd" not in str(caught.value)
        assert "stuff" not in str(caught.value)

    def test_a_short_name_is_replaced_as_a_whole_word_only(self, server: FakeInfluxServer) -> None:
        reader, _ = make_reader(server, influx={"field": "m", "temperature_field": None})
        server.script(Reply(400, '{"message": "unknown field m in the measurement"}'))
        with pytest.raises(InfluxBadRequestError) as caught:
            reader.read()
        assert "unknown field <redacted> in the measurement" in str(caught.value)

    def test_a_setting_that_the_hint_names_does_not_change_the_hint(
        self, server: FakeInfluxServer
    ) -> None:
        """The hint says `[sqm.influx]`, and a measurement can be called `sqm` too."""
        reader, _ = make_reader(server, influx={"measurement": "sqm"})
        server.script(Reply(400, '{"message": "no measurement sqm"}'))
        with pytest.raises(InfluxBadRequestError) as caught:
            reader.read()
        assert "no measurement <redacted>" in str(caught.value)
        assert "(check the settings in [sqm.influx])" in str(caught.value)

    def test_the_secrets_of_the_environment_are_replaced_too(
        self, server: FakeInfluxServer
    ) -> None:
        reader, _ = sentinel_reader(server, 2)
        server.script(Reply(401, json.dumps({"message": f"bad token {SENTINELS['token']}"})))
        with pytest.raises(InfluxUnauthorizedError) as caught:
            reader.read()
        assert SENTINELS["token"] not in str(caught.value)


class TestConstruction:
    def test_a_config_without_the_influx_table_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"needs the \[sqm.influx\] table"):
            SqmInfluxReader(SqmConfig(), clock=VirtualClock(), station_id="s", profile_id="p")

    @pytest.mark.parametrize(
        "token", ["abc\ndef", "abc\r", "caf\N{LATIN SMALL LETTER E WITH ACUTE}"]
    )
    def test_a_token_that_a_header_cannot_carry_is_a_config_error_without_the_value(
        self, server: FakeInfluxServer, token: str
    ) -> None:
        config = SqmConfig(enabled=True, source="influx", influx=SqmInfluxConfig(**table(server)))
        with pytest.raises(ConfigError) as caught:
            SqmInfluxReader(
                config, clock=VirtualClock(), station_id="s", profile_id="p", token=token
            )
        assert "abc" not in str(caught.value)

    def test_the_default_opener_checks_certificates_unless_told_otherwise(
        self, server: FakeInfluxServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[bool] = []

        def fake_opener(*, verify_tls: bool = True, use_environment_proxies: bool = True) -> Any:
            seen.append(verify_tls)
            return make_opener(use_environment_proxies=False)

        monkeypatch.setattr(sqm_influx, "make_opener", fake_opener)
        for verify in (True, False):
            config = SqmConfig(
                enabled=True,
                source="influx",
                influx=SqmInfluxConfig(**table(server, verify_tls=verify)),
            )
            SqmInfluxReader(config, clock=VirtualClock(), station_id="s", profile_id="p")
        assert seen == [True, False]


class TestReadRange:
    def test_a_version_1_range_gives_the_records_of_its_rows_oldest_first(
        self, server: FakeInfluxServer
    ) -> None:
        reader, _ = make_reader(server, version=1, altitude_deg=45.0, azimuth_deg=0.0)
        start, end = 1_000 * NS_PER_S, 2_000 * NS_PER_S
        rows: list[list[object]] = [
            [start + 3, 21.3, None],
            [start + 2, 21.2, 3.0],
            [start + 1, 21.1, 3.1],
        ]
        server.script(Reply(200, v1_reply(rows)))
        records = reader.read_range(start, end)
        assert [r.t_utc_ns for r in records] == [start + 1, start + 2, start + 3]
        assert [r.value_mag_arcsec2 for r in records] == [21.1, 21.2, 21.3]
        assert [r.temperature_c for r in records] == [3.1, 3.0, None]
        assert all(r.provenance == {"reader": "sqm-influx-1", "api": "1"} for r in records)
        assert all((r.altitude_deg, r.azimuth_deg) == (45.0, 0.0) for r in records)
        (request,) = server.requests
        assert request.query["q"][0].startswith('SELECT "mag", "temp" FROM "sqm" WHERE time >= ')
        assert f"time >= {start}ns AND time < {end}ns" in request.query["q"][0]
        assert "ORDER BY time ASC LIMIT 100001" in request.query["q"][0]

    def test_a_version_2_range_pairs_the_fields_by_time_stamp(
        self, server: FakeInfluxServer
    ) -> None:
        reader, _ = make_reader(server)
        start, end = 1_000 * NS_PER_S, 2_000 * NS_PER_S
        rows = [
            ("mag", start + 1, 21.1),
            ("mag", start + 2, 21.2),
            ("mag", start + 3, 21.3),
            ("temp", start + 1, 3.1),
            ("temp", start + 3, 3.3),
        ]
        server.script(Reply(200, v2_csv(rows)))
        records = reader.read_range(start, end)
        assert [r.t_utc_ns for r in records] == [start + 1, start + 2, start + 3]
        assert [r.temperature_c for r in records] == [3.1, None, 3.3]
        (request,) = server.requests
        assert "range(start: time(v: 1000000000000), stop: time(v: 2000000000000))" in request.body
        assert "last()" not in request.body

    def test_a_range_ignores_the_age_and_changes_no_state(self, server: FakeInfluxServer) -> None:
        events: list[HardwareEvent] = []
        reader, clock = make_reader(server, events=events)
        old = ago(clock.utc_ns(), 100_000.0)
        server.script(Reply(200, v2_csv([("mag", old, 21.4)])), Reply(503))
        assert len(reader.read_range(old - NS_PER_S, old + NS_PER_S)) == 1
        with pytest.raises(InfluxUnreachableError):
            reader.read_range(old - NS_PER_S, old + NS_PER_S)
        assert reader.failures == 0
        assert events == []
        server.script(v2_reply(clock))
        assert reader.poll() is not None  # the range read did not set the last time stamp

    def test_an_empty_range_gives_no_records(self, server: FakeInfluxServer) -> None:
        reader, _ = make_reader(server)
        server.script(Reply(200, V2_EMPTY))
        assert reader.read_range(1, 2) == []

    def test_a_range_must_not_be_empty_or_reversed(self, server: FakeInfluxServer) -> None:
        reader, _ = make_reader(server)
        for start, end in ((5, 5), (6, 5)):
            with pytest.raises(ValueError, match="greater than start_ns"):
                reader.read_range(start, end)
        assert server.requests == []

    def test_a_range_with_too_many_points_is_an_error_that_asks_for_a_shorter_range(
        self, server: FakeInfluxServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sqm_influx, "MAX_RANGE_POINTS", 2)
        reader, _ = make_reader(server, version=1)
        server.script(Reply(200, v1_reply([[10, 21.1, None], [11, 21.2, None], [12, 21.3, None]])))
        with pytest.raises(InfluxParseError, match="shorter range"):
            reader.read_range(1, 100)
        assert "LIMIT 3" in server.requests[0].query["q"][0]

    def test_a_failure_of_a_range_read_names_nothing_of_the_installation(
        self, server: FakeInfluxServer
    ) -> None:
        reader, _ = sentinel_reader(server, 2)
        server.script(Reply(400, json.dumps({"message": echo_everything(server)})))
        with pytest.raises(InfluxBadRequestError) as caught:
            reader.read_range(1, 2)
        TestSecrecy.check(str(caught.value), server, 2)
