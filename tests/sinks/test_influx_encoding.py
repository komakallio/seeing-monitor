"""The line protocol of the InfluxDB sink: golden lines, every field kind, and escaping."""

from __future__ import annotations

import json
from typing import Any, ClassVar

import pytest
from hypothesis import given
from hypothesis import strategies as st

from seeingmon.records.base import RECORD_TYPES, Record, quantity
from seeingmon.records.samples import sample_record
from seeingmon.records.sink_mapping import sink_mapping
from seeingmon.sinks.base import StoredRow
from seeingmon.sinks.influx import (
    encode_batch,
    encode_field,
    escape_key,
    escape_measurement,
    format_point,
    quote_string,
    sanitize_tag_value,
)
from tests.sinks.line_protocol import ParseError, parse_line, split_lines

T_NS = 1_800_000_000_123_456_789


class ExampleRecord(Record, register=False):
    """An example with one field of every kind, so that the tests need no real record type."""

    record_type: ClassVar[str] = "example"

    count: int = quantity(definition="A count.")
    ratio: float | None = quantity(default=None, definition="A ratio.")
    ok: bool = quantity(definition="A flag.")
    note: str | None = quantity(default=None, definition="Some text.")
    blob: bytes | None = quantity(default=None, definition="Some bytes.")
    names: list[str] = quantity(default_factory=list, definition="A list.")
    extra: dict[str, Any] | None = quantity(default=None, definition="A dict.")


def example(**overrides: Any) -> ExampleRecord:
    values: dict[str, Any] = {
        "station_id": "s1",
        "profile_id": "p1",
        "t_utc_ns": T_NS,
        "provenance": {"algo": "v1"},
        "count": 7,
        "ok": True,
    }
    values.update(overrides)
    return ExampleRecord(**values)


def line_of(record: ExampleRecord) -> str:
    return format_point(ExampleRecord, record.to_row())


class TestGoldenLines:
    def test_a_point_has_the_measurement_sorted_tags_typed_fields_and_a_nanosecond_time(
        self,
    ) -> None:
        record = example(ratio=0.5, note="hello")
        assert line_of(record) == (
            "example,profile=p1,station=s1 "
            'revision=0i,provenance="{\\"algo\\":\\"v1\\"}",count=7i,ratio=0.5,ok=true,'
            'note="hello",names="[]" 1800000000123456789'
        )

    def test_a_missing_value_leaves_its_field_out(self) -> None:
        line = line_of(example())
        assert "ratio" not in line
        assert "note" not in line
        assert "blob" not in line
        assert "extra" not in line
        assert "quality" not in line

    def test_a_field_name_never_collides_with_a_tag_or_the_time(self) -> None:
        mapping = sink_mapping(ExampleRecord).influx
        names = [field.name for field in mapping.fields]
        assert "station_id" not in names
        assert "profile_id" not in names
        assert "t_utc_ns" not in names
        assert "time" not in names


class TestFieldKinds:
    @pytest.mark.parametrize(
        ("kind", "value", "expected"),
        [
            ("int", 0, "0i"),
            ("int", -5, "-5i"),
            ("int", 2**63 - 1, "9223372036854775807i"),
            ("int", -(2**63), "-9223372036854775808i"),
            ("float", 1.5, "1.5"),
            ("float", 3.0, "3.0"),  # a float never looks like an integer, so a field keeps its type
            ("float", -0.0, "-0.0"),
            ("float", 1e22, "1e+22"),
            ("float", 5e-324, "5e-324"),
            ("float", 3, "3.0"),  # an integer in a float field is still a float
            ("bool", True, "true"),
            ("bool", False, "false"),
            ("str", "text", '"text"'),
            ("str", 'say "hi"', '"say \\"hi\\""'),
            ("str", "a\\b", '"a\\\\b"'),
            ("bytes", "AAE=", '"AAE="'),
            ("json", {"a": [1, 2]}, '"{\\"a\\":[1,2]}"'),
            ("json", [1.5, "x"], '"[1.5,\\"x\\"]"'),
        ],
    )
    def test_each_kind_has_its_own_form(self, kind: Any, value: Any, expected: str) -> None:
        assert encode_field(kind, value) == expected

    @pytest.mark.parametrize("kind", ["int", "float", "bool", "str", "bytes", "json"])
    def test_none_leaves_the_field_out(self, kind: Any) -> None:
        assert encode_field(kind, None) is None

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_a_float_that_is_not_finite_leaves_the_field_out(self, value: float) -> None:
        assert encode_field("float", value) is None
        row = {**example(ratio=1.0).to_row(), "ratio": value}
        assert "ratio" not in format_point(ExampleRecord, row)

    def test_the_time_keeps_every_nanosecond(self) -> None:
        for t_utc_ns in (0, 1, T_NS, 2**62 + 1):
            assert parse_line(line_of(example(t_utc_ns=t_utc_ns))).timestamp == t_utc_ns

    def test_a_json_field_keeps_non_ascii_text_readable(self) -> None:
        assert encode_field("json", {"name": "Pohjantähti"}) == '"{\\"name\\":\\"Pohjantähti\\"}"'


class TestEscaping:
    def test_the_measurement_escapes_a_comma_and_a_space(self) -> None:
        assert escape_measurement("a b,c=d") == "a\\ b\\,c=d"

    def test_keys_and_tag_values_escape_a_comma_an_equals_sign_and_a_space(self) -> None:
        assert escape_key("a b,c=d") == "a\\ b\\,c\\=d"

    def test_a_string_escapes_a_quote_and_a_backslash(self) -> None:
        assert quote_string('a"b\\c') == '"a\\"b\\\\c"'

    def test_a_tag_value_cannot_carry_a_backslash_a_newline_or_a_nul(self) -> None:
        assert sanitize_tag_value("a\\b\nc\rd\x00e") == "a_b_c_d_e"
        assert sanitize_tag_value("plain-value.1") == "plain-value.1"

    def test_a_tag_with_special_characters_survives_in_the_line(self) -> None:
        row = example(station_id="st 1,a=b", profile_id="p x").to_row()
        point = parse_line(format_point(ExampleRecord, row))
        assert point.tags == {"profile": "p x", "station": "st 1,a=b"}

    def test_an_empty_tag_value_is_left_out(self) -> None:
        row = {**example().to_row(), "profile_id": ""}
        assert parse_line(format_point(ExampleRecord, row)).tags == {"station": "s1"}


TEXT = st.text(alphabet=st.characters(codec="utf-8", exclude_characters="\x00"))
NAMES = TEXT.filter(bool)


class TestEscapingProperties:
    @given(
        station=NAMES,
        profile=NAMES,
        note=TEXT,
        extra=st.dictionaries(TEXT, TEXT, max_size=3),
        count=st.integers(-(2**63), 2**63 - 1),
        ratio=st.floats(allow_nan=False, allow_infinity=False),
        ok=st.booleans(),
        t_utc_ns=st.integers(0, 2**62),
    )
    def test_any_text_and_number_survives_the_line(
        self,
        station: str,
        profile: str,
        note: str,
        extra: dict[str, str],
        count: int,
        ratio: float,
        ok: bool,
        t_utc_ns: int,
    ) -> None:
        row = {
            **example().to_row(),
            "station_id": station,
            "profile_id": profile,
            "t_utc_ns": t_utc_ns,
            "provenance": {"note": note},
            "count": count,
            "ratio": ratio,
            "ok": ok,
            "note": note,
            "extra": extra,
        }
        line = format_point(ExampleRecord, row)
        point = parse_line(line)
        assert point.measurement == "example"
        assert point.tags == {
            "profile": sanitize_tag_value(profile),
            "station": sanitize_tag_value(station),
        }
        assert point.fields["note"] == note
        assert json.loads(point.fields["provenance"]) == {"note": note}
        assert json.loads(point.fields["extra"]) == extra
        assert point.fields["count"] == count
        assert point.fields["ratio"] == ratio
        assert point.fields["ok"] is ok
        assert point.timestamp == t_utc_ns
        # The body of a batch splits back into the same line, even with a newline in a string.
        body = encode_batch(ExampleRecord, [StoredRow(1, row)]).decode("utf-8")
        assert split_lines(body) == [line]

    @given(text=NAMES)
    def test_a_sanitized_tag_value_survives_the_escaping(self, text: str) -> None:
        value = sanitize_tag_value(text)
        point = parse_line(f"m,k={escape_key(value)} f=1")
        assert point.tags == {"k": value}

    @given(text=NAMES)
    def test_a_sanitized_measurement_survives_the_escaping(self, text: str) -> None:
        value = sanitize_tag_value(text)
        assert parse_line(f"{escape_measurement(value)} f=1").measurement == value

    @given(text=TEXT)
    def test_a_quoted_string_closes_only_at_its_last_quote(self, text: str) -> None:
        quoted = quote_string(text)
        point = parse_line(f"m f={quoted}")
        assert point.fields == {"f": text}


class TestBatches:
    def rows(self, count: int) -> list[StoredRow]:
        return [
            StoredRow(n + 1, example(t_utc_ns=T_NS + n, note=f"row {n}\nsecond line").to_row())
            for n in range(count)
        ]

    def test_a_batch_has_one_line_for_each_row_and_a_final_newline(self) -> None:
        body = encode_batch(ExampleRecord, self.rows(3)).decode("utf-8")
        assert body.endswith("\n")
        lines = split_lines(body)
        assert len(lines) == 3
        assert [parse_line(line).timestamp for line in lines] == [T_NS, T_NS + 1, T_NS + 2]
        assert [parse_line(line).fields["note"] for line in lines] == [
            f"row {n}\nsecond line" for n in range(3)
        ]

    def test_the_same_batch_encodes_to_the_same_bytes(self) -> None:
        rows = self.rows(5)
        assert encode_batch(ExampleRecord, rows) == encode_batch(ExampleRecord, rows)

    def test_an_unknown_record_type_is_an_error(self) -> None:
        with pytest.raises(KeyError):
            format_point("no_such_type", {})

    def test_a_row_without_a_time_is_an_error(self) -> None:
        row = example().to_row()
        del row["t_utc_ns"]
        with pytest.raises(KeyError):
            format_point(ExampleRecord, row)

    def test_a_row_from_older_software_with_missing_columns_still_encodes(self) -> None:
        row = {key: value for key, value in example(note="x").to_row().items() if key != "note"}
        assert "note" not in parse_line(format_point(ExampleRecord, row)).fields


TABLE_TYPES = [cls for cls in RECORD_TYPES.values() if cls.storage == "table"]


@pytest.mark.parametrize("cls", TABLE_TYPES, ids=lambda cls: cls.record_type)
def test_every_table_record_type_encodes_and_parses(cls: type[Record]) -> None:
    record = sample_record(cls)
    row = record.to_row()
    point = parse_line(format_point(cls.record_type, row))
    mapping = sink_mapping(cls).influx
    assert point.measurement == cls.record_type
    assert point.timestamp == record.t_utc_ns
    assert set(point.tags) <= {"station", "profile"}
    expected = {f.name for f in mapping.fields if row[f.name] is not None}
    assert set(point.fields) == expected
    for field in mapping.fields:
        value = row[field.name]
        if value is None:
            continue
        parsed = point.fields[field.name]
        if field.kind == "int":
            assert parsed == value
        elif field.kind == "float":
            assert isinstance(parsed, float)
            assert parsed == value
        elif field.kind == "bool":
            assert parsed is value
        elif field.kind == "json":
            assert json.loads(parsed) == value
        else:
            assert parsed == value


def test_the_parser_rejects_a_malformed_line() -> None:
    with pytest.raises(ParseError):
        parse_line("measurement_without_fields")
    with pytest.raises(ParseError):
        parse_line('m f="unclosed')
