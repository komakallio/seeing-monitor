"""Property tests of the row conversion for every declared record type."""

from __future__ import annotations

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from seeingmon.records.base import Record
from tests.records.strategies import ALL_RECORD_TYPES, records, type_id


@pytest.mark.parametrize("cls", ALL_RECORD_TYPES, ids=type_id)
@given(data=st.data())
def test_a_record_survives_a_trip_through_json(cls: type[Record], data: st.DataObject) -> None:
    record = data.draw(records(cls))
    row = json.loads(json.dumps(record.to_row(), allow_nan=False))
    assert cls.from_row(row, strict=True) == record
