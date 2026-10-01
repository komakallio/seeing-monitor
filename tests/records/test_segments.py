from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from seeingmon.records.seeing import FrameRecord, SeeingWindowRecord
from seeingmon.records.segments import (
    from_segment,
    segment_columns,
    segment_dtype,
    segment_header_fields,
    segment_layout,
    to_segment,
)
from tests.records.strategies import minimal_values, record_values

# The segment layout is a file format. Append a column at the end of FrameRecord and extend
# this list. Never reorder, remove, or retype a column, or old segment files stop loading.
FRAME_LAYOUT = (
    ("t_utc_ns", "<i8"),
    ("seq", "<u4"),
    ("t_err_us", "<u2"),
    ("cx_px", "<f4"),
    ("cy_px", "<f4"),
    ("width_x_px", "<f4"),
    ("width_y_px", "<f4"),
    ("peak_dn", "<u2"),
    ("flux_e", "<f4"),
    ("bg_dn", "<f4"),
    ("flags", "<u2"),
    ("dropped_before", "<u2"),
)


def make_frame(**overrides: Any) -> FrameRecord:
    values: dict[str, Any] = {
        "station_id": "station-a",
        "profile_id": "profile-a",
        "provenance": {"algo": "fast-1"},
        "stream_id": 3,
        "t_utc_ns": 1_800_000_000_000_000_000,
        "seq": 10,
        "t_err_us": 1500,
        "peak_dn": 30_000,
    }
    values.update(overrides)
    return FrameRecord(**values)


@st.composite
def segments(draw: st.DrawFn) -> list[FrameRecord]:
    """Records that share one header, as in one segment file."""
    header_names = [spec.name for spec in segment_header_fields("frame")]
    first = draw(record_values("frame"))
    header = {name: first[name] for name in header_names}
    rows = draw(st.lists(record_values("frame"), min_size=1, max_size=6))
    return [FrameRecord(**{**row, **header}) for row in rows]


class TestFrameLayout:
    def test_the_layout_is_the_declared_file_format(self) -> None:
        assert segment_layout("frame") == FRAME_LAYOUT
        assert [spec.name for spec in segment_columns("frame")] == [n for n, _ in FRAME_LAYOUT]

    def test_the_dtype_is_packed_and_little_endian(self) -> None:
        dtype = segment_dtype("frame")
        sizes = [np.dtype(code).itemsize for _, code in FRAME_LAYOUT]
        assert dtype.itemsize == sum(sizes)
        assert not dtype.isalignedstruct
        assert dtype.names == tuple(name for name, _ in FRAME_LAYOUT)
        for name, code in FRAME_LAYOUT:
            assert dtype[name].str == code

    def test_a_row_is_about_40_bytes(self) -> None:
        # The architecture budgets about 40 bytes for each frame.
        assert 36 <= segment_dtype("frame").itemsize <= 44

    def test_the_header_holds_the_fields_that_the_rows_do_not(self) -> None:
        assert [spec.name for spec in segment_header_fields("frame")] == [
            "station_id",
            "revision",
            "profile_id",
            "provenance",
            "quality",
            "stream_id",
        ]

    def test_a_class_works_as_well_as_a_name(self) -> None:
        assert segment_dtype(FrameRecord) == segment_dtype("frame")

    @pytest.mark.parametrize(
        "function", [segment_dtype, segment_layout, segment_columns, segment_header_fields]
    )
    def test_a_table_record_has_no_segment_layout(self, function: Any) -> None:
        with pytest.raises(ValueError, match="not a segment record"):
            function("seeing_window")


class TestPacking:
    def test_rows_hold_the_values(self) -> None:
        header, rows = to_segment(
            [
                make_frame(seq=1, cx_px=100.5, peak_dn=7, flags=3),
                make_frame(seq=2, cx_px=101.25, peak_dn=8, dropped_before=4),
            ]
        )
        assert header == {
            "station_id": "station-a",
            "revision": 0,
            "profile_id": "profile-a",
            "provenance": {"algo": "fast-1"},
            "quality": None,
            "stream_id": 3,
        }
        assert rows.dtype == segment_dtype("frame")
        assert rows["seq"].tolist() == [1, 2]
        assert rows["cx_px"].tolist() == [100.5, 101.25]
        assert rows["flags"].tolist() == [3, 0]
        assert rows["dropped_before"].tolist() == [0, 4]
        assert rows["t_utc_ns"].tolist() == [1_800_000_000_000_000_000] * 2

    def test_none_is_nan_in_a_row_and_nan_is_none_in_a_record(self) -> None:
        header, rows = to_segment([make_frame(cx_px=None, width_x_px=2.5)])
        assert np.isnan(rows["cx_px"][0])
        assert rows["width_x_px"][0] == 2.5
        (record,) = from_segment("frame", header, rows)
        assert record == make_frame(cx_px=None, width_x_px=2.5)
        assert isinstance(record, FrameRecord)
        assert record.cx_px is None

    def test_a_float_that_cannot_be_stored_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValidationError):
            make_frame(cx_px=1e39)

    def test_a_segment_needs_records_that_share_one_header(self) -> None:
        with pytest.raises(ValueError, match="at least one record"):
            to_segment([])
        with pytest.raises(ValueError, match="same header"):
            to_segment([make_frame(), make_frame(stream_id=4)])
        with pytest.raises(ValueError, match="same header"):
            to_segment([make_frame(), make_frame(provenance={"algo": "fast-2"})])

    def test_a_segment_holds_one_record_type(self) -> None:
        window = SeeingWindowRecord(**minimal_values("seeing_window"))
        with pytest.raises(ValueError, match="not a segment record"):
            to_segment([window])

    def test_rows_with_another_layout_are_rejected(self) -> None:
        header, _ = to_segment([make_frame()])
        other = np.zeros(1, dtype=[("seq", "<u4")])
        with pytest.raises(ValueError, match="layout"):
            from_segment("frame", header, other)

    def test_the_bytes_of_a_segment_are_the_file_payload(self) -> None:
        _, rows = to_segment([make_frame(seq=n) for n in range(5)])
        payload = rows.tobytes()
        assert len(payload) == 5 * segment_dtype("frame").itemsize
        restored = np.frombuffer(payload, dtype=segment_dtype("frame"))
        assert restored.tobytes() == payload
        assert restored["seq"].tolist() == [0, 1, 2, 3, 4]

    @given(segments())
    def test_records_survive_a_trip_through_a_segment(self, frames: list[FrameRecord]) -> None:
        header, rows = to_segment(frames)
        assert rows.shape == (len(frames),)
        assert from_segment("frame", header, rows) == frames
