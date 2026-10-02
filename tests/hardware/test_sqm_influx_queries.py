"""The queries of the InfluxDB source of the SQM-LE: escaping, the query text, and the parsers.

The escaping tests do not trust the escape functions. A small scanner reads each query the way
InfluxDB reads it (InfluxQL for version 1, Flux for version 2), and the tests check two things: the
literals decode to the settings, and the text around the literals does not depend on the settings,
so no setting can end its literal or add to the query.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from seeingmon.clock import NS_PER_S
from seeingmon.hardware.sqm import SqmInfluxConfig
from seeingmon.hardware.sqm_influx import (
    InfluxBadRequestError,
    InfluxParseError,
    InfluxPoint,
    build_flux,
    build_influxql,
    duration_literal,
    flux_string,
    influxql_identifier,
    influxql_string,
    parse_flux_points,
    parse_flux_rows,
    parse_influxql_points,
)
from tests.hardware.influx_replies import V1_EMPTY, V2_EMPTY, go_time, v1_reply, v2_csv

T = 1_767_261_912_123_456_789  # 2026-01-01T10:05:12.123456789Z


def settings(**overrides: Any) -> SqmInfluxConfig:
    values: dict[str, Any] = {
        "endpoint": "https://influx.example.org:8086",
        "org": "example-org",
        "bucket": "example-bucket",
        "measurement": "sqm",
        "field": "mag",
        "temperature_field": "temp",
        "tags": {"unit": "roof"},
    }
    values.update(overrides)
    return SqmInfluxConfig(**values)


def version_1(**overrides: Any) -> SqmInfluxConfig:
    values: dict[str, Any] = {"version": 1, "org": None, "bucket": None, "database": "example-db"}
    values.update(overrides)
    return settings(**values)


# --- Scanners that read a query as the server does -----------------------------------------


def scan_influxql(query: str) -> tuple[list[str], list[str]]:
    """Split a query like the scanner of InfluxQL: the text between literals, and the literals.

    A literal is in double quotes (an identifier) or single quotes (a string). Inside it, the
    scanner reads a backslash followed by `n`, a backslash, or a quote as an escape, and it refuses
    a raw line break and any other escape. Each text between literals ends with the quote that
    opens the next literal, so the kind of every literal is part of the skeleton.
    """
    escapes = {"n": "\n", "\\": "\\", '"': '"', "'": "'"}
    raws: list[str] = []
    literals: list[str] = []
    position = start = 0
    while position < len(query):
        quote = query[position]
        if quote not in "\"'":
            position += 1
            continue
        raws.append(query[start : position + 1])
        position += 1
        decoded: list[str] = []
        while True:
            assert position < len(query), "the literal is not closed"
            char = query[position]
            if char == quote:
                position += 1
                break
            assert char != "\n", "a raw line break inside a literal"
            if char == "\\":
                assert query[position + 1] in escapes, "an escape that InfluxQL refuses"
                decoded.append(escapes[query[position + 1]])
                position += 2
            else:
                decoded.append(char)
                position += 1
        literals.append("".join(decoded))
        start = position
    raws.append(query[start:])
    return raws, literals


def scan_flux(program: str) -> tuple[list[str], list[str]]:
    """Split a Flux program like its scanner: the text between string literals, and the literals.

    Inside a literal, the scanner reads `\\n`, `\\r`, `\\t`, `\\\\`, `\\"`, `\\$`, and `\\xHH` as
    escapes. A raw `${` starts an interpolation, which the test refuses.
    """
    simple = {"n": "\n", "r": "\r", "t": "\t", "\\": "\\", '"': '"', "$": "$"}
    raws: list[str] = []
    literals: list[str] = []
    position = start = 0
    while position < len(program):
        if program[position] != '"':
            position += 1
            continue
        raws.append(program[start:position])
        position += 1
        decoded: list[str] = []
        while True:
            assert position < len(program), "the string is not closed"
            char = program[position]
            if char == '"':
                position += 1
                break
            assert program[position : position + 2] != "${", "an interpolation"
            assert char != "\n", "a raw line break inside a string"
            if char == "\\":
                kind = program[position + 1]
                if kind == "x":
                    decoded.append(chr(int(program[position + 2 : position + 4], 16)))
                    position += 4
                else:
                    assert kind in simple, "an escape that Flux refuses"
                    decoded.append(simple[kind])
                    position += 2
            else:
                decoded.append(char)
                position += 1
        literals.append("".join(decoded))
        start = position
    raws.append(program[start:])
    return raws, literals


NASTY = [
    '"',
    "'",
    "\\",
    "\n",
    "\r\n",
    "a\"b'c\\d\ne",
    '" OR 1=1 --',
    "'; DROP MEASUREMENT sqm; --",
    '") |> yield(name: "x") //',
    "${1 + 1}",
    "\\${x}",
    "$",
    "${",
    "\t\x00\x1f\x7f",
    "\N{LINE SEPARATOR} unicode \U0001f30c",
    "/*",
]


class TestInfluxqlEscaping:
    def test_the_quotes_and_the_escapes_follow_the_language(self) -> None:
        assert influxql_identifier("mag") == '"mag"'
        assert influxql_identifier('a"b') == '"a\\"b"'
        assert influxql_identifier("a\\b") == '"a\\\\b"'
        assert influxql_identifier("a\nb") == '"a\\nb"'
        assert influxql_string("roof") == "'roof'"
        assert influxql_string("it's") == "'it\\'s'"
        assert influxql_string('say "x"') == "'say \"x\"'"  # a double quote needs no escape here

    @pytest.mark.parametrize("text", NASTY)
    def test_a_literal_decodes_to_the_text_and_ends_where_it_should(self, text: str) -> None:
        assert scan_influxql(influxql_identifier(text) + " AND x")[1] == [text]
        assert scan_influxql(influxql_string(text) + " AND x")[1] == [text]

    @given(st.text())
    def test_any_text_round_trips_through_both_literals(self, text: str) -> None:
        assert scan_influxql(influxql_identifier(text) + " tail")[1] == [text]
        assert scan_influxql(influxql_string(text) + " tail")[1] == [text]

    @pytest.mark.parametrize("text", NASTY)
    def test_no_setting_changes_the_text_around_the_literals_of_the_query(self, text: str) -> None:
        benign, _ = scan_influxql(build_influxql(settings()))
        for changed in (
            settings(measurement=text),
            settings(field=text),
            settings(temperature_field=text),
            settings(tags={text: "roof"}),
            settings(tags={"unit": text}),
        ):
            raws, literals = scan_influxql(build_influxql(changed))
            assert raws == benign
            assert text in literals  # the setting arrives as one literal, unchanged
        policy_benign, _ = scan_influxql(build_influxql(version_1(retention_policy="p")))
        raws, literals = scan_influxql(build_influxql(version_1(retention_policy=text)))
        assert raws == policy_benign
        assert text in literals

    @given(
        st.text(min_size=1, max_size=30),
        st.text(min_size=1, max_size=30),
        st.text(min_size=1, max_size=30),
        st.text(min_size=1, max_size=30),
    )
    def test_the_skeleton_of_the_query_does_not_depend_on_the_settings(
        self, measurement: str, field: str, tag_name: str, tag_value: str
    ) -> None:
        benign, _ = scan_influxql(build_influxql(settings()))
        raws, literals = scan_influxql(
            build_influxql(
                settings(
                    measurement=measurement,
                    field=field,
                    temperature_field=field + "_t",
                    tags={tag_name: tag_value},
                )
            )
        )
        assert raws == benign
        assert literals == [field, field + "_t", measurement, field, tag_name, tag_value]

    def test_the_query_of_a_range_has_the_same_guarantee(self) -> None:
        query = build_influxql(settings(measurement='x"y\n', tags={"u": "it's"}), window=(1, 2))
        raws, literals = scan_influxql(query)
        assert literals == ["mag", "temp", 'x"y\n', "mag", "u", "it's"]
        assert any("time >= 1ns AND time < 2ns" in raw for raw in raws)


class TestFluxEscaping:
    def test_the_escapes_follow_the_language(self) -> None:
        assert flux_string("mag") == '"mag"'
        assert flux_string('a"b') == '"a\\"b"'
        assert flux_string("a\\b") == '"a\\\\b"'
        assert flux_string("a\nb\rc\td") == '"a\\nb\\rc\\td"'
        assert flux_string("\x00\x1f\x7f") == '"\\x00\\x1f\\x7f"'

    def test_an_interpolation_is_escaped_and_a_lone_dollar_is_not(self) -> None:
        assert flux_string("${x}") == '"\\${x}"'
        assert flux_string("a$b$") == '"a$b$"'
        assert flux_string("\\${x}") == '"\\\\\\${x}"'  # a backslash, then the escaped dollar

    @pytest.mark.parametrize("text", NASTY)
    def test_a_string_decodes_to_the_text_and_ends_where_it_should(self, text: str) -> None:
        raws, literals = scan_flux(flux_string(text) + " tail")
        assert literals == [text]
        assert raws == ["", " tail"]

    @given(st.text())
    def test_any_text_round_trips(self, text: str) -> None:
        assert scan_flux(flux_string(text) + " tail")[1] == [text]

    @given(
        st.text(min_size=1, max_size=30),
        st.text(min_size=1, max_size=30),
        st.text(min_size=1, max_size=30),
        st.text(min_size=1, max_size=30),
        st.text(min_size=1, max_size=30),
    )
    def test_the_skeleton_of_the_program_does_not_depend_on_the_settings(
        self, bucket: str, measurement: str, field: str, tag_name: str, tag_value: str
    ) -> None:
        benign, _ = scan_flux(build_flux(settings()))
        raws, literals = scan_flux(
            build_flux(
                settings(
                    bucket=bucket,
                    measurement=measurement,
                    field=field,
                    temperature_field=field + "_t",
                    tags={tag_name: tag_value},
                )
            )
        )
        assert raws == benign
        assert literals == [bucket, measurement, field, field + "_t", tag_name, tag_value]

    @pytest.mark.parametrize("text", NASTY)
    def test_a_nasty_setting_arrives_as_one_literal_in_each_place(self, text: str) -> None:
        _, literals = scan_flux(
            build_flux(settings(bucket=text, measurement=text, field=text + "1", tags={text: text}))
        )
        assert literals == [text, text, text + "1", "temp", text, text]


# --- The text of the queries ---------------------------------------------------------------


class TestQueryText:
    def test_the_influxql_query_of_a_poll(self) -> None:
        assert build_influxql(settings()) == (
            'SELECT "mag", "temp" FROM "sqm" WHERE time > now() - 3600s AND "mag" > -1000 '
            "AND \"unit\" = 'roof' ORDER BY time DESC LIMIT 1"
        )

    def test_the_numeric_guard_appears_only_with_a_second_field(self) -> None:
        query = build_influxql(settings(temperature_field=None, tags={}))
        assert query == (
            'SELECT "mag" FROM "sqm" WHERE time > now() - 3600s ORDER BY time DESC LIMIT 1'
        )

    def test_a_retention_policy_qualifies_the_measurement(self) -> None:
        query = build_influxql(version_1(retention_policy="autogen"))
        assert 'FROM "autogen"."sqm" WHERE' in query

    def test_the_tags_keep_their_order_and_join_with_and(self) -> None:
        query = build_influxql(settings(tags={"a": "1", "b": "2"}))
        assert "\"a\" = '1' AND \"b\" = '2' ORDER BY" in query

    def test_the_influxql_query_of_a_range(self) -> None:
        query = build_influxql(settings(temperature_field=None, tags={}), window=(10, 20))
        assert query == (
            'SELECT "mag" FROM "sqm" WHERE time >= 10ns AND time < 20ns '
            "ORDER BY time ASC LIMIT 100001"
        )

    def test_the_flux_program_of_a_poll(self) -> None:
        assert build_flux(settings()) == (
            'from(bucket: "example-bucket")\n'
            "  |> range(start: -3600s)\n"
            '  |> filter(fn: (r) => r._measurement == "sqm")\n'
            '  |> filter(fn: (r) => r._field == "mag" or r._field == "temp")\n'
            '  |> filter(fn: (r) => r["unit"] == "roof")\n'
            "  |> last()\n"
        )

    def test_the_flux_program_with_one_field_and_no_tag(self) -> None:
        lines = build_flux(settings(temperature_field=None, tags={})).splitlines()
        assert lines[3] == '  |> filter(fn: (r) => r._field == "mag")'
        assert len(lines) == 5

    def test_the_flux_program_of_a_range_has_no_last(self) -> None:
        program = build_flux(settings(), window=(1_000_000_000, 2_000_000_000))
        assert "  |> range(start: time(v: 1000000000), stop: time(v: 2000000000))" in program
        assert "last()" not in program

    @pytest.mark.parametrize(
        ("seconds", "text"),
        [(600.0, "600s"), (3600, "3600s"), (0.5, "500ms"), (90.25, "90250ms"), (0.0001, "1ms")],
    )
    def test_a_duration_is_whole_seconds_or_milliseconds(self, seconds: float, text: str) -> None:
        assert duration_literal(seconds) == text

    def test_the_lookback_sets_the_window_of_both_queries(self) -> None:
        table = settings(lookback_s=1800.0, max_age_s=300.0)
        assert "now() - 1800s" in build_influxql(table)
        assert "range(start: -1800s)" in build_flux(table)


# --- The parser of version 1 ---------------------------------------------------------------


def v1_error_cases() -> list[str]:
    time_ns = 1_767_261_912_123_456_789
    return [
        "",
        "not json",
        "[]",
        "null",
        "{}",
        '{"results":[]}',
        '{"results":{}}',
        '{"results":[1]}',
        '{"results":[{"series":{}}]}',
        '{"results":[{"series":[1]}]}',
        '{"results":[{"series":[{"columns":"time","values":[]}]}]}',
        '{"results":[{"series":[{"columns":["time","mag"],"values":{}}]}]}',
        '{"results":[{"series":[{"columns":["time","other"],"values":[[1,2]]}]}]}',
        '{"results":[{"series":[{"columns":["mag"],"values":[[21.4]]}]}]}',
        '{"results":[{"series":[{"columns":["time","mag"],"values":[[1,21.4,5]]}]}]}',
        '{"results":[{"series":[{"columns":["time","mag"],"values":[5]}]}]}',
        '{"results":[{"series":[{"columns":[1,"mag"],"values":[]}]}]}',
        f'{{"results":[{{"series":[{{"columns":["time","mag"],"values":[[{time_ns},"x"]]}}]}}]}}',
        f'{{"results":[{{"series":[{{"columns":["time","mag"],"values":[[{time_ns},true]]}}]}}]}}',
        f'{{"results":[{{"series":[{{"columns":["time","mag"],"values":[[{time_ns},NaN]]}}]}}]}}',
        f'{{"results":[{{"series":[{{"columns":["time","mag"],"values":[[{time_ns},1e999]]}}]}}]}}',
        '{"results":[{"series":[{"columns":["time","mag"],"values":[[true,21.4]]}]}]}',
        '{"results":[{"series":[{"columns":["time","mag"],"values":[[1.5,21.4]]}]}]}',
        '{"results":[{"series":[{"columns":["time","mag"],"values":[["last night",21.4]]}]}]}',
        '{"results":[{"series":[{"columns":["time","mag"],"values":[[-5,21.4]]}]}]}',
        '{"results":[{"series":[{"columns":["time","mag"],"values":[[0,21.4]]}]}]}',
        '{"results":[{"series":[{"columns":["time","mag"],"values":[[99999999999999999999,21.4]]}]}]}',
    ]


class TestInfluxqlParser:
    def test_the_reply_of_a_server(self) -> None:
        points = parse_influxql_points(v1_reply([[T, 21.43, 3.4]]), "mag", "temp")
        assert points == [InfluxPoint(T, 21.43, 3.4)]

    def test_an_integer_magnitude_and_a_null_temperature(self) -> None:
        points = parse_influxql_points(v1_reply([[T, 21, None]]), "mag", "temp")
        assert points == [InfluxPoint(T, 21.0, None)]

    def test_the_temperature_is_none_when_no_field_is_named(self) -> None:
        reply = v1_reply([[T, 21.43]], columns=("time", "mag"))
        assert parse_influxql_points(reply, "mag", None) == [InfluxPoint(T, 21.43, None)]

    def test_columns_are_found_by_name_and_not_by_position(self) -> None:
        reply = v1_reply([[3.4, 21.43, T]], columns=("temp", "mag", "time"))
        assert parse_influxql_points(reply, "mag", "temp") == [InfluxPoint(T, 21.43, 3.4)]

    def test_rows_come_back_in_reply_order_and_a_null_magnitude_is_skipped(self) -> None:
        reply = v1_reply([[T + 2, 21.5, None], [T + 1, None, 3.0], [T, 21.4, None]])
        assert [p.t_utc_ns for p in parse_influxql_points(reply, "mag", "temp")] == [T + 2, T]

    def test_a_time_as_text_is_read_too(self) -> None:
        reply = v1_reply([["2026-01-01T10:05:12.123456789Z", 21.43, None]])
        assert parse_influxql_points(reply, "mag", "temp")[0].t_utc_ns == T

    @pytest.mark.parametrize("reply", [V1_EMPTY, '{"results":[{"statement_id":0,"series":[]}]}'])
    def test_a_reply_with_no_series_holds_no_point(self, reply: str) -> None:
        assert parse_influxql_points(reply, "mag", "temp") == []

    def test_a_series_with_no_rows_holds_no_point(self) -> None:
        assert parse_influxql_points(v1_reply([]), "mag", "temp") == []

    @pytest.mark.parametrize("reply", v1_error_cases())
    def test_anything_else_is_a_parse_error(self, reply: str) -> None:
        with pytest.raises(InfluxParseError):
            parse_influxql_points(reply, "mag", "temp")

    @pytest.mark.parametrize("magnitude", [-5.01, 30.01, 99.99, -1000, 1e9])
    def test_a_magnitude_that_no_sky_meter_gives_is_a_parse_error(self, magnitude: float) -> None:
        with pytest.raises(InfluxParseError, match="magnitude"):
            parse_influxql_points(v1_reply([[T, magnitude, None]]), "mag", "temp")

    def test_the_range_of_the_magnitude_includes_its_limits(self) -> None:
        reply = v1_reply([[T, -5.0, None], [T + 1, 30.0, None]])
        assert [p.magnitude for p in parse_influxql_points(reply, "mag", None)] == [-5.0, 30.0]

    def test_a_temperature_that_is_not_a_number_is_a_parse_error(self) -> None:
        with pytest.raises(InfluxParseError, match="temperature"):
            parse_influxql_points(v1_reply([[T, 21.4, "warm"]]), "mag", "temp")
        with pytest.raises(InfluxParseError, match="temperature"):
            parse_influxql_points(v1_reply([[T, 21.4, 5000.0]]), "mag", "temp")

    @pytest.mark.parametrize(
        "reply",
        [
            '{"results":[{"statement_id":0,"error":"database not found: x"}]}',
            '{"error":"error parsing query: found x"}',
        ],
    )
    def test_an_error_of_the_server_in_a_reply_of_200_is_a_bad_request(self, reply: str) -> None:
        with pytest.raises(InfluxBadRequestError, match="InfluxDB reported an error"):
            parse_influxql_points(reply, "mag", "temp")

    def test_the_message_of_the_server_is_shortened_to_one_line(self) -> None:
        reply = '{"error":"' + "word\\n " * 100 + '"}'
        with pytest.raises(InfluxBadRequestError) as caught:
            parse_influxql_points(reply, "mag", "temp")
        assert "\n" not in str(caught.value)
        assert len(str(caught.value)) < 260


# --- The parser of version 2 ---------------------------------------------------------------


class TestFluxParser:
    def test_the_reply_of_a_server_with_one_field(self) -> None:
        points = parse_flux_points(v2_csv([("mag", T, 21.43)]), "mag", None)
        assert points == [InfluxPoint(T, 21.43, None)]

    def test_two_fields_that_share_a_header(self) -> None:
        reply = v2_csv([("mag", T, 21.43), ("temp", T, 3.4)])
        assert reply.count("#datatype") == 1
        assert parse_flux_points(reply, "mag", "temp") == [InfluxPoint(T, 21.43, 3.4)]

    def test_two_fields_with_a_header_for_each_table(self) -> None:
        reply = v2_csv([("mag", T, 21.43), ("temp", T, 3.4)], separate_tables=True)
        assert reply.count("#datatype") == 2
        assert parse_flux_points(reply, "mag", "temp") == [InfluxPoint(T, 21.43, 3.4)]

    def test_the_line_ends_may_be_bare_line_feeds(self) -> None:
        reply = v2_csv([("mag", T, 21.43), ("temp", T, 3.4)], line_end="\n")
        assert parse_flux_points(reply, "mag", "temp") == [InfluxPoint(T, 21.43, 3.4)]

    def test_a_temperature_at_another_time_is_not_the_temperature_of_the_point(self) -> None:
        reply = v2_csv([("mag", T, 21.43), ("temp", T + NS_PER_S, 3.4)])
        assert parse_flux_points(reply, "mag", "temp") == [InfluxPoint(T, 21.43, None)]

    def test_a_field_that_nobody_asked_for_is_ignored(self) -> None:
        reply = v2_csv([("mag", T, 21.43), ("other", T, 7.0)])
        assert parse_flux_points(reply, "mag", "temp") == [InfluxPoint(T, 21.43, None)]

    def test_a_temperature_alone_gives_no_point(self) -> None:
        assert parse_flux_points(v2_csv([("temp", T, 3.4)]), "mag", "temp") == []

    def test_two_series_of_one_field_give_two_points_oldest_first(self) -> None:
        reply = v2_csv([("mag", T + NS_PER_S, 21.5), ("mag", T, 21.4)])
        assert [p.t_utc_ns for p in parse_flux_points(reply, "mag", None)] == [T, T + NS_PER_S]

    @pytest.mark.parametrize(
        ("t_utc_ns", "text"),
        [
            (1_767_261_912_000_000_000, "2026-01-01T10:05:12Z"),  # Go writes no fraction
            (1_767_261_912_500_000_000, "2026-01-01T10:05:12.5Z"),
            (1_767_261_912_123_456_789, "2026-01-01T10:05:12.123456789Z"),
            (1_767_261_912_000_000_001, "2026-01-01T10:05:12.000000001Z"),
        ],
    )
    def test_the_time_has_up_to_nine_fractional_digits(self, t_utc_ns: int, text: str) -> None:
        assert go_time(t_utc_ns) == text
        points = parse_flux_points(v2_csv([("mag", t_utc_ns, 21.4)]), "mag", None)
        assert points[0].t_utc_ns == t_utc_ns

    def test_columns_are_found_by_name_and_a_value_may_hold_a_comma(self) -> None:
        text = (
            "#datatype,string,long,dateTime:RFC3339,double,string,string\r\n"
            "#group,false,false,false,false,true,true\r\n"
            "#default,_result,,,,,\r\n"
            ",result,table,_time,_value,_field,unit\r\n"
            f',,0,{go_time(T)},21.43,mag,"roof, north"\r\n'
            "\r\n"
        )
        assert parse_flux_points(text, "mag", None) == [InfluxPoint(T, 21.43, None)]

    def test_the_annotation_rows_may_come_in_any_order(self) -> None:
        text = (
            "#default,_result,,,,\r\n"
            "#datatype,string,long,dateTime:RFC3339,double,string\r\n"
            "#group,false,false,false,false,true\r\n"
            ",result,table,_time,_value,_field\r\n"
            f",,0,{go_time(T)},21.43,mag\r\n"
        )
        assert parse_flux_points(text, "mag", None) == [InfluxPoint(T, 21.43, None)]

    def test_a_second_table_may_have_another_header_after_an_empty_line(self) -> None:
        first = v2_csv([("mag", T, 21.43)])
        second = (
            "#datatype,string,long,dateTime:RFC3339,double,string,string\r\n"
            "#group,false,false,false,false,true,true\r\n"
            "#default,_result,,,,,\r\n"
            ",result,table,_time,_value,_field,site\r\n"
            f",,1,{go_time(T)},3.4,temp,north\r\n"
        )
        assert parse_flux_points(first + second, "mag", "temp") == [InfluxPoint(T, 21.43, 3.4)]

    def test_a_long_integer_value_is_a_number(self) -> None:
        text = v2_csv([("mag", T, 21)])
        text = text.replace("double,string,string,string", "long,string,string,string")
        assert parse_flux_points(text, "mag", None) == [InfluxPoint(T, 21.0, None)]

    @pytest.mark.parametrize("reply", [V2_EMPTY, "\r\n", "\r\n\r\n", "\n"])
    def test_an_empty_reply_holds_no_point(self, reply: str) -> None:
        assert parse_flux_points(reply, "mag", "temp") == []
        assert parse_flux_rows(reply) == []

    def test_a_reply_with_a_header_and_no_row_holds_no_point(self) -> None:
        text = v2_csv([])
        assert text.startswith("#group")
        assert parse_flux_points(text, "mag", "temp") == []

    def test_the_error_table_of_a_failed_query_is_a_bad_request(self) -> None:
        text = (
            "#datatype,string,string\r\n#group,true,true\r\n#default,,\r\n"
            ",error,reference\r\n"
            ',"failed to execute: could not find bucket ""x""",\r\n'
        )
        with pytest.raises(InfluxBadRequestError) as caught:
            parse_flux_points(text, "mag", None)
        assert "could not find bucket" in str(caught.value)

    @pytest.mark.parametrize(
        "reply",
        [
            "this is not csv\r\nat all\r\n",
            "a single line of text",
            "<html><body>502</body></html>",
            '{"code":"unauthorized","message":"x"}',
            "#datatype,string,long\r\n",  # the annotations stop before the header
            "#group,false\r\n#datatype,string\r\n#default,\r\n",
            ",result,table,_time,_value,_field\r\n,,0,2026-01-01T00:00:00Z,1,mag\r\n",
        ],
    )
    def test_a_reply_that_is_not_annotated_csv_is_a_parse_error(self, reply: str) -> None:
        with pytest.raises(InfluxParseError):
            parse_flux_points(reply, "mag", None)

    @pytest.mark.parametrize(
        ("old", "new", "message"),
        [
            (go_time(T), "yesterday", "no valid time"),
            (go_time(T), "2026-01-01T10:05:12+02:00", "no valid time"),
            (go_time(T), "2026-01-01T10:05:12.1234567890Z", "no valid time"),
            (go_time(T), "", "no valid time"),
            ("21.43", "abc", "no valid time or value"),
            ("21.43", "", "no valid time or value"),
            ("21.43", "NaN", "no valid time or value"),
            ("21.43", "+Inf", "no valid time or value"),
            ("21.43", "-inf", "no valid time or value"),
            ("21.43", "31", "magnitude"),
            ("21.43", "-6", "magnitude"),
            (",mag,sqm,roof", ",mag,sqm", "does not match the header"),
            (",mag,sqm,roof", ",mag,sqm,roof,extra", "does not match the header"),
            ("double", "string", "not a number"),
            ("double", "boolean", "not a number"),
            ("_value", "_val", "lacks _time, _value, or _field"),
            ("_time", "_when", "lacks _time, _value, or _field"),
            ("_field", "_name", "lacks _time, _value, or _field"),
        ],
    )
    def test_a_row_that_it_cannot_read_is_a_parse_error(
        self, old: str, new: str, message: str
    ) -> None:
        text = v2_csv([("mag", T, 21.43)])
        assert old in text
        with pytest.raises(InfluxParseError, match=message):
            parse_flux_points(text.replace(old, new, 1), "mag", None)

    def test_a_quote_that_never_closes_is_a_parse_error(self) -> None:
        text = v2_csv([("mag", T, 21.43)]).replace(",roof", ',"roof')
        with pytest.raises(InfluxParseError, match="not valid CSV"):
            parse_flux_points(text, "mag", None)

    def test_a_field_larger_than_the_csv_limit_is_a_parse_error(self) -> None:
        text = v2_csv([("mag", T, 21.43)]).replace(",roof", "," + "x" * 200_000)
        with pytest.raises(InfluxParseError, match="not valid CSV"):
            parse_flux_points(text, "mag", None)
