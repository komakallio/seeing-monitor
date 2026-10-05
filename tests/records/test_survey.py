from __future__ import annotations

import struct
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from seeingmon.records.base import field_specs
from seeingmon.records.reference import REFERENCE_SOURCES, ReferenceRecord
from seeingmon.records.survey import (
    POINTING_FLAGS,
    SKY_QUALITY_FLAGS,
    PointingRecord,
    SkyQualityRecord,
    StarEpochRecord,
    StarListRecord,
    SurveyFrameRecord,
    pack_star_rows,
    star_rows,
)

BASE: dict[str, Any] = {
    "station_id": "station-a",
    "t_utc_ns": 1_800_000_000_000_000_000,
    "profile_id": "profile-a",
    "provenance": {"algo": "survey-1", "dark": "2026-09"},
}


def floats(*values: float) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


class TestSurveyFrameRecord:
    def test_a_frame_without_analysis_results_is_valid(self) -> None:
        frame = SurveyFrameRecord(**BASE, exposure_s=5.0, gain=120, readout_mode="bin2")
        assert frame.image_ref is None
        assert frame.n_detected is None

    @pytest.mark.parametrize("ref", ["survey/2026-10-01/frame-0001.fits", "a.fits", "x/y/z"])
    def test_an_image_reference_is_a_relative_path(self, ref: str) -> None:
        frame = SurveyFrameRecord(
            **BASE, exposure_s=5.0, gain=120, readout_mode="bin2", image_ref=ref
        )
        assert frame.image_ref == ref

    @pytest.mark.parametrize(
        "ref",
        ["/data/frame.fits", "../frame.fits", "a/../b.fits", "C:\\f.fits", "C:/f.fits", "C:f", ""],
    )
    def test_an_image_reference_never_points_outside_the_data_directory(self, ref: str) -> None:
        with pytest.raises(ValidationError):
            SurveyFrameRecord(**BASE, exposure_s=5.0, gain=120, readout_mode="bin2", image_ref=ref)

    @pytest.mark.parametrize("overrides", [{"exposure_s": 0}, {"gain": -1}, {"n_detected": -1}])
    def test_values_stay_in_their_physical_range(self, overrides: dict[str, Any]) -> None:
        values: dict[str, Any] = {"exposure_s": 5.0, "gain": 120, "readout_mode": "bin2"}
        values.update(overrides)
        with pytest.raises(ValidationError):
            SurveyFrameRecord(**BASE, **values)


class TestSkyQualityRecord:
    def test_the_documented_flag_codes(self) -> None:
        assert set(SKY_QUALITY_FLAGS) == {
            "cloud",
            "twilight",
            "moon",
            "dew",
            "dark_due",
            "time_invalid",
            "saturated_sky",
        }
        record = SkyQualityRecord(**BASE, n_stars_used=0, flags=["cloud", "moon"])
        assert record.flags == ["cloud", "moon"]
        with pytest.raises(ValidationError, match="unknown code"):
            SkyQualityRecord(**BASE, n_stars_used=0, flags=["clouds"])

    def test_a_missing_value_is_none_and_the_quality_says_why(self) -> None:
        record = SkyQualityRecord(
            **BASE, n_stars_used=0, quality={"zero_point_mag": "no stars matched"}
        )
        assert record.zero_point_mag is None
        assert record.to_row()["zero_point_mag"] is None
        assert record.to_row()["quality"] == {"zero_point_mag": "no stars matched"}

    def test_the_quality_names_a_declared_field(self) -> None:
        with pytest.raises(ValidationError, match="quality names fields"):
            SkyQualityRecord(**BASE, n_stars_used=0, quality={"zero_point": "typo"})

    @pytest.mark.parametrize(
        "overrides", [{"cloud_fraction": 1.1}, {"transparency": -0.1}, {"n_stars_used": -1}]
    )
    def test_values_stay_in_their_physical_range(self, overrides: dict[str, Any]) -> None:
        values: dict[str, Any] = {"n_stars_used": 5}
        values.update(overrides)
        with pytest.raises(ValidationError):
            SkyQualityRecord(**BASE, **values)

    def test_the_two_brightness_fields_use_the_same_unit(self) -> None:
        record = SkyQualityRecord(
            **BASE, n_stars_used=9, sky_mag_arcsec2=20.1, sky_mag_arcsec2_v=20.3
        )
        assert (record.sky_mag_arcsec2, record.sky_mag_arcsec2_v) == (20.1, 20.3)


class TestPointingRecord:
    def make(self, **overrides: Any) -> PointingRecord:
        values: dict[str, Any] = {"n_matched": 12, "readout_mode": "bin2", "solver": "fake"}
        values.update(overrides)
        return PointingRecord(**BASE, **values)

    def test_the_documented_flag_codes(self) -> None:
        assert set(POINTING_FLAGS) == {
            "unsolved",
            "few_stars",
            "moved",
            "roll_undefined",
            "time_invalid",
        }

    def test_an_unsolved_frame_has_no_geometry(self) -> None:
        record = self.make(
            n_matched=0, flags=["unsolved"], quality={"center_ra_deg": "no solution"}
        )
        assert record.center_ra_deg is None
        assert record.attitude is None

    def test_the_attitude_has_nine_numbers(self) -> None:
        matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        assert self.make(attitude=matrix).attitude == matrix
        for bad in ([1.0] * 4, [1.0] * 10, []):
            with pytest.raises(ValidationError):
                self.make(attitude=bad)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"center_ra_deg": -1.0},
            {"center_ra_deg": 361.0},
            {"center_dec_deg": 90.5},
            {"plate_scale_arcsec_px": 0.0},
            {"offset_arcmin": -0.1},
            {"readout_mode": ""},
            {"solver": ""},
        ],
    )
    def test_values_stay_in_their_physical_range(self, overrides: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            self.make(**overrides)

    def test_the_plate_scale_comes_with_its_readout_mode(self) -> None:
        record = self.make(plate_scale_arcsec_px=3.82)
        assert (record.readout_mode, record.plate_scale_arcsec_px) == ("bin2", 3.82)

    def test_the_pole_position_is_optional_and_comes_after_the_flags(self) -> None:
        record = self.make()
        assert (record.pole_x_px, record.pole_y_px) == (None, None)
        names = [spec.name for spec in field_specs(PointingRecord)]
        assert names[-3:] == ["flags", "pole_x_px", "pole_y_px"]  # the declaration only grew

    @pytest.mark.parametrize("position", [(2100.5, 1399.5), (-35.5, 3000.25), (-1e6, 1e6)])
    def test_the_pole_may_lie_outside_the_frame(self, position: tuple[float, float]) -> None:
        record = self.make(pole_x_px=position[0], pole_y_px=position[1])
        assert (record.pole_x_px, record.pole_y_px) == position
        assert PointingRecord.from_row(record.to_row(), strict=True) == record

    def test_a_missing_pole_says_why_in_the_quality(self) -> None:
        reason = "the pole is not in front of the camera"
        record = self.make(quality={"pole_x_px": reason, "pole_y_px": reason})
        assert record.to_row()["pole_x_px"] is None
        assert record.to_row()["quality"] == {"pole_x_px": reason, "pole_y_px": reason}

    @pytest.mark.parametrize(
        "overrides", [{"pole_x_px": float("nan")}, {"pole_y_px": float("inf")}]
    )
    def test_the_pole_must_be_a_finite_number(self, overrides: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            self.make(**overrides)


class TestStarRecords:
    def test_a_star_list_keeps_its_rows_as_float32(self) -> None:
        data = floats(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        record = StarListRecord(
            **BASE, n_stars=2, columns=["x_px", "y_px", "flux_e"], data=data, catalog="gaia-dr3"
        )
        assert StarListRecord.retention_days == 365
        assert len(record.data) == 24
        assert StarListRecord.from_row(record.to_row()) == record

    def test_a_star_list_can_be_empty(self) -> None:
        record = StarListRecord(**BASE, n_stars=0, columns=["x_px"], data=b"")
        assert record.catalog is None

    @pytest.mark.parametrize(
        ("n_stars", "columns", "data"),
        [
            (1, ["x_px", "y_px"], floats(1.0)),  # too few bytes
            (1, ["x_px"], floats(1.0, 2.0)),  # too many bytes
            (2, ["x_px"], b"\x00" * 7),  # not a multiple of 4
            (1, ["x_px", "x_px"], floats(1.0, 2.0)),  # a repeated column
            (1, ["", "y_px"], floats(1.0, 2.0)),  # an empty column name
        ],
    )
    def test_the_data_must_match_the_rows_and_columns(
        self, n_stars: int, columns: list[str], data: bytes
    ) -> None:
        with pytest.raises(ValidationError):
            StarListRecord(**BASE, n_stars=n_stars, columns=columns, data=data)
        with pytest.raises(ValidationError):
            StarEpochRecord(
                **BASE, night="2026-10-01", n_stars=n_stars, n_frames=3, columns=columns, data=data
            )

    def test_a_star_epoch_is_one_summary_for_a_night(self) -> None:
        record = StarEpochRecord(
            **BASE,
            night="2026-10-01",
            n_stars=1,
            n_frames=240,
            columns=["dx_arcsec", "dy_arcsec", "mag"],
            data=floats(0.01, -0.02, 9.5),
        )
        assert StarEpochRecord.retention_days is None
        assert StarEpochRecord.from_row(record.to_row()) == record

    @pytest.mark.parametrize("night", ["2026-13-01", "2026-02-30", "26-10-01", "2026/10/01", ""])
    def test_a_night_is_a_calendar_date(self, night: str) -> None:
        with pytest.raises(ValidationError):
            StarEpochRecord(**BASE, night=night, n_stars=0, n_frames=0, columns=["mag"], data=b"")


class TestStarRows:
    def test_rows_survive_packing_and_reading(self) -> None:
        rows = np.array([[1.5, -2.25, 9.0], [0.125, np.nan, 11.5]])
        record = StarListRecord(
            **BASE, n_stars=2, columns=["x_px", "y_px", "mag"], data=pack_star_rows(rows)
        )
        back = star_rows(record)
        assert back.shape == (2, 3)
        assert back.dtype == np.dtype("<f4")
        assert np.array_equal(back, rows.astype("<f4"), equal_nan=True)
        assert not back.flags.writeable

    def test_the_packing_is_little_endian_float32_in_row_major_order(self) -> None:
        data = pack_star_rows([[1.0, 2.0], [3.0, 4.0]])
        assert data == struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
        assert pack_star_rows(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=">f8")) == data
        assert pack_star_rows(np.array([[1, 3], [2, 4]]).T) == data  # a transposed view

    def test_an_empty_list_has_no_bytes_and_no_rows(self) -> None:
        assert pack_star_rows(np.empty((0, 3))) == b""
        record = StarEpochRecord(
            **BASE, night="2026-10-01", n_stars=0, n_frames=0, columns=["a", "b", "c"], data=b""
        )
        assert star_rows(record).shape == (0, 3)

    @pytest.mark.parametrize("rows", [[1.0, 2.0], 3.0, [[[1.0]]]])
    def test_rows_must_be_two_dimensional(self, rows: object) -> None:
        with pytest.raises(ValueError, match="2-D"):
            pack_star_rows(rows)  # type: ignore[arg-type]

    def test_stars_need_columns(self) -> None:
        with pytest.raises(ValidationError, match="at least one column"):
            StarListRecord(**BASE, n_stars=2, columns=[], data=b"")


class TestReferenceRecord:
    def test_a_fixed_reading(self) -> None:
        record = ReferenceRecord(
            **BASE,
            instrument="sqm-le",
            source="fixed",
            value_mag_arcsec2=21.4,
            temperature_c=-3.5,
            altitude_deg=45.0,
            azimuth_deg=0.0,
        )
        assert record.note is None
        assert set(REFERENCE_SOURCES) == {"fixed", "manual"}

    def test_a_manual_reading_may_leave_the_pointing_out(self) -> None:
        record = ReferenceRecord(
            **BASE, instrument="sqm-l", source="manual", value_mag_arcsec2=21.0, note="outside"
        )
        assert record.altitude_deg is None

    @pytest.mark.parametrize(
        "overrides",
        [{"source": "handheld"}, {"altitude_deg": 91}, {"azimuth_deg": 361}, {"instrument": ""}],
    )
    def test_the_values_must_be_valid(self, overrides: dict[str, Any]) -> None:
        values: dict[str, Any] = {
            "instrument": "sqm-le",
            "source": "fixed",
            "value_mag_arcsec2": 21.4,
        }
        values.update(overrides)
        with pytest.raises(ValidationError):
            ReferenceRecord(**BASE, **values)
