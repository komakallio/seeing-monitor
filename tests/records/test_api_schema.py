from __future__ import annotations

import json
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from seeingmon.records.api_schema import (
    QUALITY_REF,
    api_schema,
    field_schema,
    quality_schema,
    record_schema,
    schema_name,
)
from seeingmon.records.base import RECORD_TYPES, Record, field_specs, quantity
from seeingmon.records.system import HealthRecord
from tests.records.jsonschema_lite import InvalidError, unknown_keywords, validate
from tests.records.strategies import ALL_RECORD_TYPES, minimal_record, records, type_id

DOCUMENT = api_schema()
SCHEMAS = DOCUMENT["components"]["schemas"]


class TestComponents:
    def test_there_is_one_component_for_each_record_type_and_one_for_quality(self) -> None:
        assert list(SCHEMAS) == [
            "Quality",
            "Frame",
            "SeeingWindow",
            "SurveyFrame",
            "SkyQuality",
            "Pointing",
            "StarList",
            "StarEpoch",
            "Reference",
            "Health",
            "Event",
            "Run",
        ]

    def test_names_are_camel_case_versions_of_the_record_types(self) -> None:
        assert schema_name("seeing_window") == "SeeingWindow"
        assert schema_name("run") == "Run"
        assert schema_name(RECORD_TYPES["star_epoch"]) == "StarEpoch"

    def test_a_subset_can_be_selected(self) -> None:
        subset = api_schema(["event", RECORD_TYPES["run"]])
        assert list(subset["components"]["schemas"]) == ["Quality", "Event", "Run"]

    def test_the_document_is_plain_json_and_does_not_change_between_calls(self) -> None:
        text = json.dumps(DOCUMENT, indent=2)
        assert json.loads(text) == DOCUMENT
        assert json.dumps(api_schema(), indent=2) == text

    def test_the_generator_emits_only_keywords_that_the_tests_check(self) -> None:
        for name, schema in SCHEMAS.items():
            assert unknown_keywords(schema) == set(), name
        assert unknown_keywords({"properties": {"x": {"oneOf": []}}}) == {"oneOf"}


@pytest.mark.parametrize("cls", ALL_RECORD_TYPES, ids=type_id)
class TestRecordSchema:
    def test_it_contains_every_declared_field_with_its_unit(self, cls: type[Record]) -> None:
        schema = record_schema(cls)
        properties = schema["properties"]
        specs = field_specs(cls)
        assert list(properties) == [spec.name for spec in specs]
        assert schema["required"] == [spec.name for spec in specs]
        for spec in specs:
            assert properties[spec.name]["description"] == spec.definition
            if spec.unit is None:
                assert "x-unit" not in properties[spec.name], spec.name
            else:
                assert properties[spec.name]["x-unit"] == spec.unit, spec.name

    def test_it_describes_the_record(self, cls: type[Record]) -> None:
        schema = record_schema(cls)
        assert schema["type"] == "object"
        assert schema["title"] == schema_name(cls)
        assert schema["x-record-type"] == cls.record_type
        assert schema["x-storage"] == cls.storage
        assert schema["x-key"] == ["station_id", "t_utc_ns", "revision"]
        assert schema.get("x-retention-days") == cls.retention_days
        first_line = (cls.__doc__ or "").strip().split("\n")[0]
        assert schema["description"].startswith(first_line)

    def test_a_nullable_field_accepts_null_and_a_required_one_does_not(
        self, cls: type[Record]
    ) -> None:
        for spec in field_specs(cls):
            schema = field_schema(spec)
            if spec.name == "quality":
                assert {"type": "null"} in schema["anyOf"]
                continue
            types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
            assert ("null" in types) == spec.nullable, f"{cls.record_type}.{spec.name}"

    def test_the_minimal_record_validates(self, cls: type[Record]) -> None:
        validate(minimal_record(cls).to_row(), SCHEMAS[schema_name(cls)], DOCUMENT)

    def test_a_row_with_a_missing_field_does_not_validate(self, cls: type[Record]) -> None:
        row = minimal_record(cls).to_row()
        del row["station_id"]
        with pytest.raises(InvalidError, match="missing station_id"):
            validate(row, SCHEMAS[schema_name(cls)], DOCUMENT)


@pytest.mark.parametrize("cls", ALL_RECORD_TYPES, ids=type_id)
@given(data=st.data())
def test_every_row_validates_against_the_schema(cls: type[Record], data: st.DataObject) -> None:
    row = json.loads(json.dumps(data.draw(records(cls)).to_row()))
    validate(row, SCHEMAS[schema_name(cls)], DOCUMENT)


class TestFields:
    def spec(self, record: str, name: str) -> dict[str, Any]:
        properties: dict[str, dict[str, Any]] = record_schema(record)["properties"]
        return properties[name]

    def test_types(self) -> None:
        assert self.spec("seeing_window", "n_frames")["type"] == "integer"
        assert self.spec("seeing_window", "valid_fraction")["type"] == "number"
        assert self.spec("seeing_window", "readout_mode")["type"] == "string"
        assert self.spec("health", "degraded")["type"] == "boolean"
        assert self.spec("health", "time_synchronized")["type"] == ["boolean", "null"]
        data = next(s for s in field_specs("star_list") if s.name == "data")
        assert self.spec("star_list", "data") == {
            "type": "string",
            "contentEncoding": "base64",
            "description": data.definition,
        }
        assert self.spec("run", "profile")["type"] == "object"
        assert "additionalProperties" not in self.spec("run", "profile")
        assert self.spec("health", "components")["additionalProperties"] == {"type": "string"}
        assert self.spec("health", "sink_backlog")["additionalProperties"]["type"] == "integer"
        assert self.spec("event", "detail")["type"] == ["object", "null"]

    def test_units_are_published_as_extensions(self) -> None:
        assert self.spec("seeing_window", "seeing_fwhm_arcsec")["x-unit"] == "arcsec"
        assert self.spec("seeing_window", "motion_psd_x_arcsec2_per_hz")["x-unit"] == "arcsec^2/Hz"
        assert self.spec("sky_quality", "sky_mag_arcsec2")["x-unit"] == "mag/arcsec^2"
        assert self.spec("pointing", "plate_scale_arcsec_px")["x-unit"] == "arcsec/px"
        assert "x-unit" not in self.spec("seeing_window", "n_frames")

    def test_bounds(self) -> None:
        fraction = self.spec("seeing_window", "valid_fraction")
        assert (fraction["minimum"], fraction["maximum"]) == (0, 1)
        assert self.spec("seeing_window", "exposure_us")["exclusiveMinimum"] == 0
        assert self.spec("frame", "peak_dn")["maximum"] == 65_535
        assert self.spec("pointing", "attitude")["minItems"] == 9
        assert self.spec("pointing", "attitude")["maxItems"] == 9
        assert self.spec("star_epoch", "night")["pattern"] == r"^\d{4}-\d{2}-\d{2}$"
        assert self.spec("run", "run_id")["minLength"] == 1

    def test_codes_become_an_enum_and_a_documented_extension(self) -> None:
        level = self.spec("event", "level")
        assert level["enum"] == ["error", "info", "warning"]
        assert set(level["x-codes"]) == {"error", "info", "warning"}
        assert all(level["x-codes"].values())
        flags = self.spec("seeing_window", "flags")
        assert flags["type"] == "array"
        assert "twilight" in flags["items"]["enum"]
        assert flags["x-codes"]["heater_on"].startswith("The dew heater")

    def test_a_nullable_field_with_codes_allows_null_in_the_enum(self) -> None:
        schema = field_schema(next(s for s in field_specs("health") if s.name == "state"))
        assert None not in schema["enum"]  # state is required
        # No declared record has a nullable field with codes, so check the rule on a variant.

        class Variant(HealthRecord, register=False):
            mode: str | None = quantity(default=None, codes={"a": "A."}, definition="A mode.")

        variant = field_schema(next(s for s in field_specs(Variant) if s.name == "mode"))
        assert variant["type"] == ["string", "null"]
        assert variant["enum"] == ["a", None]


class TestQuality:
    def test_the_quality_object_is_documented_once_and_referenced_by_every_record(self) -> None:
        quality = SCHEMAS["Quality"]
        assert quality == quality_schema()
        assert quality["type"] == "object"
        assert quality["additionalProperties"] == {"type": "string"}
        assert "missing or uncertain" in quality["description"]
        assert "`null` value" in quality["description"]
        for cls in ALL_RECORD_TYPES:
            property_ = record_schema(cls)["properties"]["quality"]
            assert {"$ref": QUALITY_REF} in property_["anyOf"]
            assert property_["description"] == field_specs(cls)[5].definition

    def test_a_missing_value_with_a_reason_validates(self) -> None:
        record = minimal_record("seeing_window")
        row = {**record.to_row(), "quality": {"seeing_fwhm_arcsec": "too_few_frames"}}
        validate(row, SCHEMAS["SeeingWindow"], DOCUMENT)
        row["quality"] = {"seeing_fwhm_arcsec": 3}
        with pytest.raises(InvalidError):
            validate(row, SCHEMAS["SeeingWindow"], DOCUMENT)


class TestTheValidatorCatchesMistakes:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("n_frames", "5400"),
            ("n_frames", True),
            ("n_frames", 5400.5),
            ("valid_fraction", 1.5),
            ("exposure_us", 0),
            ("flags", ["not_a_flag"]),
            ("readout_mode", ""),
            ("heater_duty", "off"),
            ("motion_psd_freq_hz", "1,2"),
            ("provenance", {"algo": 1}),
        ],
    )
    def test_a_wrong_value_does_not_validate(self, field: str, value: Any) -> None:
        row = {**minimal_record("seeing_window").to_row(), field: value}
        with pytest.raises(InvalidError):
            validate(row, SCHEMAS["SeeingWindow"], DOCUMENT)

    def test_bytes_must_be_base64(self) -> None:
        row = {**minimal_record("star_list").to_row(), "data": "not base64!"}
        with pytest.raises(InvalidError, match="base64"):
            validate(row, SCHEMAS["StarList"], DOCUMENT)
