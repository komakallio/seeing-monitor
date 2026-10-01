"""Checks that hold for every declared record type, and checks of the seeing records."""

from __future__ import annotations

import re
from typing import Any

import pytest
from pydantic import ValidationError

from seeingmon.records.base import RECORD_TYPES, Record, field_specs
from seeingmon.records.samples import sample_record
from seeingmon.records.seeing import SEEING_WINDOW_FLAGS, FrameRecord, SeeingWindowRecord
from tests.records.strategies import ALL_RECORD_TYPES, type_id

# The name of a quantity that has a unit ends with the unit, so the API field name carries
# it (`seeing_fwhm_arcsec`). This table maps each unit to the suffix. A new unit needs a line.
UNIT_SUFFIXES: dict[str, str] = {
    "ns": "_ns",
    "us": "_us",
    "ms": "_ms",  # milliseconds
    "s": "_s",
    "Hz": "_hz",
    "arcsec": "_arcsec",
    "arcmin": "_arcmin",
    "deg": "_deg",
    "px": "_px",
    "DN": "_dn",
    "e-": "_e",
    "cm": "_cm",
    "m": "_m",
    "m/s": "_ms",  # meters per second. It shares the suffix with milliseconds.
    "degC": "_c",
    "mag": "_mag",
    "mag/arcsec^2": "_mag_arcsec2",
    "arcsec/px": "_arcsec_px",
    "arcsec^2/Hz": "_arcsec2_per_hz",
    "e-/s/arcsec^2": "_e_per_s_arcsec2",
    "GB": "_gb",
    "MB": "_mb",
}

# Fields whose name continues past the unit suffix: the V-band variant of the sky brightness.
SUFFIX_QUALIFIERS: dict[str, str] = {"sky_mag_arcsec2_v": "mag/arcsec^2"}

# Fields without a unit whose name ends like a unit suffix. None exist yet.
UNITLESS_EXCEPTIONS: frozenset[str] = frozenset()

FILLER = re.compile(r"\b(?:simply|just|easy|easily|obviously|please)\b|!", re.IGNORECASE)


@pytest.mark.parametrize("cls", ALL_RECORD_TYPES, ids=type_id)
class TestEveryRecordType:
    def test_every_field_has_a_definition_in_one_sentence(self, cls: type[Record]) -> None:
        for spec in field_specs(cls):
            where = f"{cls.record_type}.{spec.name}"
            assert spec.definition, where
            assert spec.definition.endswith("."), f"{where}: end the definition with a period"
            assert not FILLER.search(spec.definition), f"{where}: cut the filler words"
            sentences = re.split(r"(?<=[a-z0-9`)])\.\s+(?=[A-Z`])", spec.definition)
            assert len(sentences) == 1, f"{where}: write one sentence"

    def test_names_carry_their_units(self, cls: type[Record]) -> None:
        for spec in field_specs(cls):
            where = f"{cls.record_type}.{spec.name}"
            if spec.unit is None:
                if spec.name in UNITLESS_EXCEPTIONS:
                    continue
                for suffix in UNIT_SUFFIXES.values():
                    assert not spec.name.endswith(suffix), (
                        f"{where} ends with {suffix} but declares no unit"
                    )
                continue
            assert spec.unit in UNIT_SUFFIXES, (
                f"{where}: unknown unit {spec.unit!r}. Add it to UNIT_SUFFIXES in this test."
            )
            if SUFFIX_QUALIFIERS.get(spec.name) == spec.unit:
                continue
            suffix = UNIT_SUFFIXES[spec.unit]
            assert spec.name.endswith(suffix), f"{where}: end the name with {suffix}"

    def test_the_record_documents_itself(self, cls: type[Record]) -> None:
        doc = cls.__doc__ or ""
        assert doc.strip().endswith(".")
        assert not FILLER.search(doc)

    def test_optional_fields_default_to_none(self, cls: type[Record]) -> None:
        for spec in field_specs(cls):
            if spec.nullable:
                assert spec.default is None, f"{cls.record_type}.{spec.name}"

    def test_the_sample_record_is_valid_and_survives_a_round_trip(self, cls: type[Record]) -> None:
        record = sample_record(cls)
        assert cls.from_row(record.to_row(), strict=True) == record


class TestFrameRecord:
    def test_declaration(self) -> None:
        assert FrameRecord.record_type == "frame"
        assert FrameRecord.storage == "segment"
        assert FrameRecord.retention_days == 7
        assert RECORD_TYPES["frame"] is FrameRecord

    def test_per_row_fields_have_a_storage_type(self) -> None:
        row = {s.name: s.dtype for s in field_specs(FrameRecord) if s.dtype is not None}
        assert row == {
            "t_utc_ns": "i8",
            "seq": "u4",
            "t_err_us": "u2",
            "cx_px": "f4",
            "cy_px": "f4",
            "width_x_px": "f4",
            "width_y_px": "f4",
            "peak_dn": "u2",
            "flux_e": "f4",
            "bg_dn": "f4",
            "flags": "u2",
            "dropped_before": "u2",
        }

    def test_the_stream_belongs_to_the_segment(self) -> None:
        stream = next(s for s in field_specs(FrameRecord) if s.name == "stream_id")
        assert stream.dtype is None

    @pytest.mark.parametrize(
        ("field", "value"),
        [("seq", 2**32), ("t_err_us", 65_536), ("peak_dn", -1), ("flags", 2**16)],
    )
    def test_integers_must_fit_their_storage_type(self, field: str, value: int) -> None:
        values: dict[str, Any] = {"seq": 1, "t_err_us": 2, "peak_dn": 3, field: value}
        with pytest.raises(ValidationError):
            FrameRecord(
                station_id="s", t_utc_ns=1, profile_id="p", provenance={}, stream_id=0, **values
            )


def make_window(**overrides: Any) -> SeeingWindowRecord:
    values: dict[str, Any] = {
        "station_id": "s",
        "t_utc_ns": 1_800_000_000_000_000_000,
        "profile_id": "p",
        "provenance": {"algo": "fast-1"},
        "duration_s": 60.0,
        "stream_id": 4,
        "readout_mode": "bin1",
        "exposure_us": 2000,
        "gain": 120,
        "n_frames": 5400,
        "n_dropped": 0,
        "valid_fraction": 1.0,
    }
    values.update(overrides)
    return SeeingWindowRecord(**values)


class TestSeeingWindowRecord:
    def test_declaration(self) -> None:
        assert SeeingWindowRecord.record_type == "seeing_window"
        assert SeeingWindowRecord.storage == "table"
        assert SeeingWindowRecord.retention_days is None

    def test_a_window_with_no_statistics_is_valid(self) -> None:
        window = make_window(quality={"seeing_fwhm_arcsec": "too few frames"})
        assert window.seeing_fwhm_arcsec is None
        assert window.flags == []

    def test_the_documented_flag_codes(self) -> None:
        assert set(SEEING_WINDOW_FLAGS) == {
            "degraded",
            "cloud",
            "twilight",
            "vibration",
            "saturated",
            "partial",
            "time_invalid",
            "heater_on",
        }
        assert make_window(flags=sorted(SEEING_WINDOW_FLAGS)).flags == sorted(SEEING_WINDOW_FLAGS)
        with pytest.raises(ValidationError, match="unknown code"):
            make_window(flags=["cloudy"])

    @pytest.mark.parametrize(
        "overrides",
        [
            {"valid_fraction": 1.01},
            {"saturated_fraction": -0.1},
            {"heater_duty": 1.5},
            {"duration_s": 0},
            {"n_frames": -1},
            {"seeing_fwhm_arcsec": 0.0},
            {"zenith_angle_deg": 181},
        ],
    )
    def test_values_stay_in_their_physical_range(self, overrides: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            make_window(**overrides)

    def test_the_spectrum_lists_have_the_same_length(self) -> None:
        window = make_window(
            motion_psd_freq_hz=[1.0, 2.0],
            motion_psd_x_arcsec2_per_hz=[0.5, 0.4],
            motion_psd_y_arcsec2_per_hz=[0.3, 0.2],
        )
        assert window.motion_psd_freq_hz == [1.0, 2.0]
        with pytest.raises(ValidationError, match="one value for each item"):
            make_window(motion_psd_freq_hz=[1.0, 2.0], motion_psd_x_arcsec2_per_hz=[0.5])
        with pytest.raises(ValidationError, match="one value for each item"):
            make_window(motion_psd_y_arcsec2_per_hz=[0.5])

    def test_an_empty_vibration_list_differs_from_a_missing_one(self) -> None:
        assert make_window(vibration_lines_hz=[]).to_row()["vibration_lines_hz"] == []
        assert make_window().to_row()["vibration_lines_hz"] is None
