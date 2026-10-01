from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from typing import Any, ClassVar

import numpy as np
import pytest
from pydantic import ValidationError

import seeingmon.records as records_package
from seeingmon.frames import FrameFlag
from seeingmon.records.base import (
    KEY_FIELDS,
    RECORD_TYPES,
    Record,
    Storage,
    field_specs,
    get_record_type,
    quantity,
    resolve_record_type,
)


class SampleRecord(Record, register=False):
    """A record that uses every supported kind of field."""

    record_type: ClassVar[str] = "sample"

    count: int = quantity(ge=0, definition="The number of samples.")
    size_arcsec: float | None = quantity(
        unit="arcsec", default=None, gt=0, definition="The size of a sample, in arcseconds."
    )
    ready: bool = quantity(default=False, definition="Whether the sample is ready.")
    label: str | None = quantity(default=None, definition="A label.")
    blob: bytes | None = quantity(default=None, definition="Some bytes.")
    values: list[float] = quantity(default_factory=list, definition="Some numbers.")
    flags: list[str] = quantity(
        default_factory=list,
        codes={"first": "The first code.", "second": "The second code."},
        definition="Facts about the sample.",
    )
    counts: dict[str, int] | None = quantity(default=None, definition="Counts by name.")
    detail: dict[str, Any] | None = quantity(default=None, definition="Any JSON object.")


def make(**overrides: Any) -> SampleRecord:
    fields: dict[str, Any] = {
        "station_id": "station-a",
        "t_utc_ns": 1_800_000_000_000_000_000,
        "profile_id": "profile-a",
        "provenance": {"algo": "test-1"},
        "count": 3,
    }
    fields.update(overrides)
    return SampleRecord(**fields)


@pytest.fixture
def scratch_registry() -> Iterator[None]:
    """Let a test register record types and remove them afterwards."""
    saved = dict(RECORD_TYPES._types)
    yield
    RECORD_TYPES._types.clear()
    RECORD_TYPES._types.update(saved)


class TestRecord:
    def test_defaults_and_base_fields(self) -> None:
        record = make()
        assert record.revision == 0
        assert record.quality is None
        assert record.ready is False
        assert record.values == []
        assert record.record_key == ("station-a", "sample", 1_800_000_000_000_000_000, 0)
        assert KEY_FIELDS == ("station_id", "t_utc_ns", "revision")

    def test_a_record_with_lists_and_dicts_is_hashable_by_its_key(self) -> None:
        first = make(values=[1.0], counts={"a": 1})
        same = make(values=[1.0], counts={"a": 1})
        other = make(values=[1.0], counts={"a": 1}, revision=1)
        assert first == same
        assert hash(first) == hash(same) == hash(first.record_key)
        assert len({first, same, other}) == 2

    def test_a_record_is_immutable(self) -> None:
        record = make()
        with pytest.raises(ValidationError, match="frozen"):
            record.count = 4  # type: ignore[misc]

    def test_the_constructor_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs"):
            make(surprise=1)

    def test_the_base_class_is_not_a_record_type(self) -> None:
        with pytest.raises(TypeError, match="base class"):
            Record(station_id="s", t_utc_ns=1, profile_id="p", provenance={})

    @pytest.mark.parametrize(
        "overrides",
        [
            {"count": 1.0},  # a float for an int field
            {"count": "3"},
            {"count": True},
            {"count": -1},
            {"count": 2**63},  # SQLite stores 8-byte integers
            {"size_arcsec": 0.0},
            {"size_arcsec": float("nan")},
            {"size_arcsec": float("inf")},
            {"values": [1.0, float("nan")]},
            {"station_id": ""},
            {"revision": -1},
            {"ready": 1},
            {"label": b"bytes"},
            {"blob": "text"},
            {"counts": {"a": 1.5}},
            {"detail": {"bad": object()}},
            {"detail": {"bad": float("nan")}},
        ],
    )
    def test_the_constructor_rejects_bad_values(self, overrides: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            make(**overrides)

    def test_an_integer_is_a_valid_float(self) -> None:
        assert make(size_arcsec=2).size_arcsec == 2.0

    def test_numpy_values_become_python_values(self) -> None:
        record = make(
            t_utc_ns=np.int64(5),
            count=np.uint16(7),
            size_arcsec=np.float32(1.5),
            ready=np.bool_(True),
            values=np.array([1.0, 2.5], dtype=np.float32),
            counts={"a": np.int32(4)},
            detail={"x": np.float64(2.0), "y": [np.int8(1)], "z": np.array([1, 2])},
        )
        assert type(record.t_utc_ns) is int
        assert type(record.count) is int
        assert type(record.size_arcsec) is float
        assert record.ready is True
        assert record.values == [1.0, 2.5]
        assert record.detail == {"x": 2.0, "y": [1], "z": [1, 2]}
        json.dumps(record.to_row())

    def test_enum_members_become_plain_values(self) -> None:
        assert type(make(count=FrameFlag.RECOVERED).count) is int

    def test_a_tuple_becomes_a_list(self) -> None:
        assert make(values=(1.0, 2.0)).values == [1.0, 2.0]

    def test_bytes_like_values_become_bytes(self) -> None:
        assert make(blob=bytearray(b"ab")).blob == b"ab"
        assert make(blob=memoryview(b"ab")).blob == b"ab"

    def test_lists_and_dicts_are_copies(self) -> None:
        values = [1.0]
        record = make(values=values)
        values.append(2.0)
        assert record.values == [1.0]

    def test_documented_codes_are_checked(self) -> None:
        assert make(flags=["first", "second"]).flags == ["first", "second"]
        with pytest.raises(ValidationError, match=r"unknown code\(s\) \['third'\]"):
            make(flags=["first", "third"])

    def test_quality_may_name_only_declared_fields(self) -> None:
        assert make(quality={"size_arcsec": "too few frames"}).quality is not None
        with pytest.raises(ValidationError, match="quality names fields"):
            make(quality={"size": "typo"})


class TestRows:
    def test_row_has_the_declared_names_in_order_and_is_json_compatible(self) -> None:
        row = make(
            size_arcsec=1.25,
            blob=b"\x00\xff\xfe",
            values=[1.0, 2.0],
            flags=["first"],
            counts={"a": 1},
            detail={"nested": {"k": [1, "x", None]}},
        ).to_row()
        assert list(row) == [spec.name for spec in field_specs(SampleRecord)]
        assert row["blob"] == base64.b64encode(b"\x00\xff\xfe").decode("ascii")
        assert json.loads(json.dumps(row, allow_nan=False)) == row

    def test_round_trip(self) -> None:
        record = make(
            size_arcsec=1.25,
            ready=True,
            label="x",
            blob=b"\x00\xff\xfe",
            values=[0.1, 0.2],
            flags=["second"],
            counts={"a": 1},
            detail={"nested": {"k": [1, "x", None]}},
            quality={"size_arcsec": "uncertain"},
            revision=2,
        )
        assert SampleRecord.from_row(json.loads(json.dumps(record.to_row()))) == record

    def test_changing_a_row_leaves_the_record_alone(self) -> None:
        record = make(values=[1.0], detail={"k": [1]})
        row = record.to_row()
        row["values"].append(2.0)
        row["detail"]["k"].append(2)
        assert record.values == [1.0]
        assert record.detail == {"k": [1]}

    def test_from_row_ignores_what_newer_software_adds(self) -> None:
        row = make(flags=["first"]).to_row()
        row.update({"added_later": 1, "row_id": 7})
        assert SampleRecord.from_row(row).flags == ["first"]
        row["flags"] = ["added_code"]
        row["quality"] = {"added_field": "reason"}
        assert SampleRecord.from_row(row).flags == ["added_code"]

    def test_from_row_in_strict_mode_rejects_what_the_declaration_does_not_know(self) -> None:
        row = make().to_row()
        row["added_later"] = 1
        with pytest.raises(ValueError, match="added_later"):
            SampleRecord.from_row(row, strict=True)
        del row["added_later"]
        row["flags"] = ["added_code"]
        with pytest.raises(ValidationError, match="unknown code"):
            SampleRecord.from_row(row, strict=True)

    def test_from_row_still_checks_types(self) -> None:
        row = make().to_row()
        row["count"] = "3"
        with pytest.raises(ValidationError):
            SampleRecord.from_row(row)

    def test_from_row_rejects_invalid_base64(self) -> None:
        row = make(blob=b"abc").to_row()
        row["blob"] = "not base64!"
        with pytest.raises(ValueError, match="base64"):
            SampleRecord.from_row(row)


class TestFieldSpecs:
    def test_specs_follow_the_declaration(self) -> None:
        specs = {spec.name: spec for spec in field_specs(SampleRecord)}
        assert list(specs)[:6] == [
            "station_id",
            "t_utc_ns",
            "revision",
            "profile_id",
            "provenance",
            "quality",
        ]
        size = specs["size_arcsec"]
        assert (size.kind, size.nullable, size.required, size.unit) == (
            "float",
            True,
            False,
            "arcsec",
        )
        assert size.type_name == "float"
        assert size.constraints == {"gt": 0}
        assert specs["count"].required
        assert specs["count"].constraints == {"ge": 0}
        assert specs["blob"].kind == "bytes"
        assert specs["values"].kind == "json"
        assert specs["values"].type_name == "list[float]"
        assert specs["values"].default == []
        assert specs["counts"].type_name == "dict[str, int]"
        assert specs["detail"].type_name == "dict[str, Any]"
        assert specs["flags"].codes == {"first": "The first code.", "second": "The second code."}
        assert specs["station_id"].base
        assert not size.base

    def test_the_base_time_field_is_stored_as_a_64_bit_integer(self) -> None:
        spec = next(s for s in field_specs(SampleRecord) if s.name == "t_utc_ns")
        assert (spec.unit, spec.dtype) == ("ns", "i8")

    def test_definitions_are_normalized_to_one_line(self) -> None:
        field = quantity(definition="One\n    sentence.")
        assert field.description == "One sentence."


class TestDeclarationErrors:
    def test_a_field_that_can_be_none_needs_a_default(self) -> None:
        with pytest.raises(ValueError, match="needs a default"):

            class Bad(Record, register=False):
                record_type: ClassVar[str] = "bad"
                value: float | None = quantity(definition="A value.")

    def test_an_unsupported_type_is_rejected(self) -> None:
        with pytest.raises(TypeError, match=r"Bad.value: unsupported type"):

            class Bad(Record, register=False):
                record_type: ClassVar[str] = "bad"
                value: tuple[int, ...] = quantity(default=(), definition="A value.")

    def test_a_union_of_several_types_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="one type"):

            class Bad(Record, register=False):
                record_type: ClassVar[str] = "bad"
                value: int | str = quantity(default=0, definition="A value.")

    def test_a_field_needs_a_definition_from_quantity(self) -> None:
        with pytest.raises(ValueError, match=r"Bad.value: declare the field with quantity"):

            class Bad(Record, register=False):
                record_type: ClassVar[str] = "bad"
                value: int = 0

    @pytest.mark.parametrize("record_type", ["Bad", "bad-name", "1bad", "", "x" * 64])
    def test_the_record_type_must_be_a_safe_identifier(self, record_type: str) -> None:
        namespace = {"record_type": record_type, "__annotations__": {}}
        with pytest.raises(ValueError, match="record_type"):
            type("Bad", (Record,), namespace, register=False)

    def test_the_storage_kind_must_be_known(self) -> None:
        with pytest.raises(ValueError, match="storage"):

            class Bad(Record, register=False):
                record_type: ClassVar[str] = "bad"
                storage: ClassVar[Storage] = "file"  # type: ignore[assignment]

    def test_retention_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="retention_days"):

            class Bad(Record, register=False):
                record_type: ClassVar[str] = "bad"
                retention_days: ClassVar[int | None] = 0

    def test_a_unit_does_not_apply_to_text(self) -> None:
        with pytest.raises(ValueError, match="no unit"):

            class Bad(Record, register=False):
                record_type: ClassVar[str] = "bad"
                value: str = quantity(unit="px", default="", definition="A value.")

    def test_codes_apply_to_text_only(self) -> None:
        with pytest.raises(ValueError, match="codes apply to"):

            class Bad(Record, register=False):
                record_type: ClassVar[str] = "bad"
                value: int = quantity(default=0, codes={"a": "A."}, definition="A value.")

    def test_a_table_field_takes_no_dtype(self) -> None:
        with pytest.raises(ValueError, match="segment record take a dtype"):

            class Bad(Record, register=False):
                record_type: ClassVar[str] = "bad"
                value: int = quantity(default=0, dtype="u2", definition="A value.")

    def test_a_dtype_must_match_the_kind_of_the_field(self) -> None:
        with pytest.raises(ValueError, match="needs an int field"):

            class BadInt(Record, register=False):
                record_type: ClassVar[str] = "bad"
                storage: ClassVar[Storage] = "segment"
                value: float = quantity(default=0.0, dtype="u2", definition="A value.")

        with pytest.raises(ValueError, match="needs a float field"):

            class BadFloat(Record, register=False):
                record_type: ClassVar[str] = "bad"
                storage: ClassVar[Storage] = "segment"
                value: int = quantity(default=0, dtype="f4", definition="A value.")

    def test_a_segment_record_needs_row_fields(self) -> None:
        with pytest.raises(ValueError, match="per-row fields"):

            class Bad(Record, register=False):
                record_type: ClassVar[str] = "bad"
                storage: ClassVar[Storage] = "segment"

    def test_a_dtype_bounds_an_integer_field(self) -> None:
        class Narrow(Record, register=False):
            record_type: ClassVar[str] = "narrow"
            storage: ClassVar[Storage] = "segment"
            small: int = quantity(dtype="u2", definition="A small number.")

        specs = {spec.name: spec for spec in field_specs(Narrow)}
        assert specs["small"].constraints == {"ge": 0, "le": 65_535}
        with pytest.raises(ValidationError):
            Narrow(station_id="s", t_utc_ns=1, profile_id="p", provenance={}, small=65_536)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"definition": "  "}, "needs a definition"),
            ({"definition": "A.", "default": 1, "default_factory": list}, "not both"),
            ({"definition": "A.", "dtype": "u3"}, "unknown dtype"),
            ({"definition": "A.", "codes": {}}, "codes must not be empty"),
        ],
    )
    def test_quantity_rejects_bad_arguments(self, kwargs: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            quantity(**kwargs)


class TestRegistry:
    def test_a_record_class_registers_itself(self, scratch_registry: None) -> None:
        class Registered(Record):
            record_type: ClassVar[str] = "registered_for_test"
            value: int = quantity(default=0, definition="A value.")

        assert RECORD_TYPES["registered_for_test"] is Registered
        assert get_record_type("registered_for_test") is Registered
        assert resolve_record_type(Registered) is Registered
        assert resolve_record_type("registered_for_test") is Registered
        assert "registered_for_test" in RECORD_TYPES
        assert "registered_for_test" in list(RECORD_TYPES)
        assert dict(RECORD_TYPES.items())["registered_for_test"] is Registered
        assert len(RECORD_TYPES) == len(list(RECORD_TYPES))

    def test_a_second_class_cannot_take_a_name(self, scratch_registry: None) -> None:
        class First(Record):
            record_type: ClassVar[str] = "taken_for_test"
            value: int = quantity(default=0, definition="A value.")

        with pytest.raises(ValueError, match="already declared by First"):

            class Second(Record):
                record_type: ClassVar[str] = "taken_for_test"
                value: int = quantity(default=0, definition="A value.")

        assert RECORD_TYPES["taken_for_test"] is First

    def test_register_false_keeps_a_variant_out_of_the_registry(
        self, scratch_registry: None
    ) -> None:
        class Original(Record):
            record_type: ClassVar[str] = "original_for_test"
            value: int = quantity(default=0, definition="A value.")

        class Variant(Original, register=False):
            extra_field: int | None = quantity(default=None, definition="Another value.")

        assert RECORD_TYPES["original_for_test"] is Original
        assert Variant.record_type == "original_for_test"
        assert [spec.name for spec in field_specs(Variant)][-1] == "extra_field"

    def test_a_subclass_without_its_own_type_is_a_helper(self, scratch_registry: None) -> None:
        class Original(Record):
            record_type: ClassVar[str] = "helper_base_for_test"
            value: int = quantity(default=0, definition="A value.")

        class Helper(Original):
            pass

        assert RECORD_TYPES["helper_base_for_test"] is Original
        assert Helper.record_type == "helper_base_for_test"

    def test_an_unknown_name_lists_the_known_types(self, scratch_registry: None) -> None:
        class Known(Record):
            record_type: ClassVar[str] = "known_for_test"
            value: int = quantity(default=0, definition="A value.")

        with pytest.raises(KeyError, match="known_for_test"):
            get_record_type("no_such_record")

    def test_resolve_rejects_other_objects(self) -> None:
        with pytest.raises(TypeError, match="record type name or a Record subclass"):
            resolve_record_type(3)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            resolve_record_type(dict)  # type: ignore[arg-type]


class TestPackageExports:
    def test_names_load_on_first_use(self) -> None:
        assert records_package.Record is Record
        assert records_package.quantity is quantity
        assert records_package.RECORD_TYPES is RECORD_TYPES
        assert "get_record_type" in dir(records_package)
        assert set(records_package.__all__) == set(records_package._EXPORTS)
        assert set(records_package.__all__) <= set(dir(records_package))

    def test_every_exported_name_resolves_to_the_object_that_it_names(self) -> None:
        for name in records_package.__all__:
            value = getattr(records_package, name)
            assert value is not None, name
            if name.endswith("Record") and name != "Record":
                assert issubclass(value, Record), name
                assert RECORD_TYPES[value.record_type] is value, name

    def test_an_unknown_name_is_an_attribute_error(self) -> None:
        with pytest.raises(AttributeError, match="no attribute"):
            _ = records_package.nothing_here
