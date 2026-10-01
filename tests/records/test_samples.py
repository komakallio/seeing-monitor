from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic import ValidationError

from seeingmon.records.base import Record, field_specs, quantity
from seeingmon.records.samples import sample_record, sample_values
from seeingmon.records.seeing import SeeingWindowRecord
from seeingmon.records.system import EventRecord, HealthRecord
from tests.records.strategies import ALL_RECORD_TYPES, type_id


@pytest.mark.parametrize("cls", ALL_RECORD_TYPES, ids=type_id)
class TestEveryRecordType:
    def test_a_sample_is_valid_and_round_trips(self, cls: type[Record]) -> None:
        record = sample_record(cls)
        assert type(record) is cls
        assert cls.from_row(record.to_row(), strict=True) == record

    def test_a_sample_sets_the_required_fields_only(self, cls: type[Record]) -> None:
        values = sample_values(cls)
        required = {spec.name for spec in field_specs(cls) if spec.required}
        assert set(values) == required
        row = sample_record(cls).to_row()
        for spec in field_specs(cls):
            if spec.nullable:
                assert row[spec.name] is None, spec.name

    def test_every_declared_example_is_a_valid_value(self, cls: type[Record]) -> None:
        examples = {spec.name: spec.examples[0] for spec in field_specs(cls) if spec.examples}
        record = sample_record(cls, **examples)
        row = record.to_row()
        for name, example in examples.items():
            assert row[name] == example, name

    def test_a_sample_does_not_change_between_calls(self, cls: type[Record]) -> None:
        assert sample_record(cls) == sample_record(cls.record_type)
        assert sample_values(cls) == sample_values(cls)


class TestSamples:
    def test_a_field_takes_its_declared_example(self) -> None:
        values = sample_values("event")
        assert values["station_id"] == "station-1"
        assert values["profile_id"] == "profile-1"
        assert values["t_utc_ns"] == 1_767_225_600_000_000_000
        assert values["provenance"] == {"algo": "fast-1"}
        assert values["kind"] == "scheduler.state_change"
        assert values["level"] == "info"  # the first documented code

    def test_a_field_without_an_example_takes_the_simplest_valid_value(self) -> None:
        values = sample_values("seeing_window")
        assert values["n_frames"] == 0
        assert values["exposure_us"] == 1  # the number must exceed 0
        assert values["valid_fraction"] == 0.0
        assert values["duration_s"] == 1.0
        assert values["readout_mode"] == "bin1"
        health = sample_values("health")
        assert health["degraded"] is False
        assert health["components"] == {"acquire": "ok", "core": "ok", "web": "ok"}

    def test_keyword_arguments_set_and_replace_fields(self) -> None:
        window = sample_record(
            "seeing_window", n_frames=5400, seeing_fwhm_arcsec=1.2, station_id="other"
        )
        assert isinstance(window, SeeingWindowRecord)
        assert (window.n_frames, window.seeing_fwhm_arcsec, window.station_id) == (
            5400,
            1.2,
            "other",
        )

    def test_a_class_gives_a_record_of_that_class(self) -> None:
        event = sample_record(EventRecord, message="Something happened.")
        assert event.message == "Something happened."
        assert sample_record(HealthRecord).state == "safe"

    def test_an_invalid_override_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            sample_record("seeing_window", n_frames=-1)
        with pytest.raises(ValidationError, match="Extra inputs"):
            sample_record("event", surprise=1)

    def test_an_unknown_record_type_is_an_error(self) -> None:
        with pytest.raises(KeyError, match="no_such_record"):
            sample_record("no_such_record")


class TestDeclaredExamples:
    def test_an_example_that_does_not_fit_the_type_is_rejected_when_you_declare_it(self) -> None:
        with pytest.raises(ValueError, match=r"Bad.count: the example 'many' is not valid"):

            class Bad(Record, register=False):
                record_type: ClassVar[str] = "bad"
                count: int = quantity(example="many", definition="A count.")

    def test_an_example_must_respect_the_bounds_and_the_pattern(self) -> None:
        with pytest.raises(ValueError, match="not valid"):

            class Bounded(Record, register=False):
                record_type: ClassVar[str] = "bounded"
                count: int = quantity(ge=0, example=-1, definition="A count.")

        with pytest.raises(ValueError, match="not valid"):

            class Patterned(Record, register=False):
                record_type: ClassVar[str] = "patterned"
                night: str = quantity(pattern=r"^\d{4}$", example="26", definition="A year.")

    def test_an_example_must_be_a_declared_code(self) -> None:
        with pytest.raises(ValueError, match="not a declared code"):

            class Coded(Record, register=False):
                record_type: ClassVar[str] = "coded"
                level: str = quantity(
                    codes={"info": "A level."}, example="loud", definition="A level."
                )

    def test_a_nullable_field_accepts_an_example_and_so_does_a_list(self) -> None:
        class Fine(Record, register=False):
            record_type: ClassVar[str] = "fine"
            maybe: float | None = quantity(default=None, example=1.5, definition="A value.")
            values: list[float] = quantity(default_factory=list, example=[1.0], definition="Some.")

        specs = {spec.name: spec for spec in field_specs(Fine)}
        assert specs["maybe"].examples == (1.5,)
        assert specs["values"].examples == ([1.0],)
