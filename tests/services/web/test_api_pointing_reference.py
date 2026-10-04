"""The Pointing card gets its offset from the latest `pointing` record: with a reference or without.

The card shows `offset_arcmin` as its large value. When the value is null, the note under it is the
text that `quality` gives for the field. `seeingmon pointing set-reference` makes the reference
from a stored record, and the survey path then writes the offset and the ID of the reference into
every later record (`tests/survey/test_pointing_cli.py` and `tests/survey/test_pipeline.py` test
those two steps). This test follows the records through the store and the API to the card.
"""

from __future__ import annotations

import pytest

from seeingmon.store.db import Store
from seeingmon.survey import pointing as pt
from seeingmon.survey import reference as ref
from tests.services.web.client import TestClient
from tests.services.web.seed import STATION
from tests.survey.pointfx import MINUTE_NS, PROFILE, T0, made, mount, tilted

API = "/api/v1"


def test_the_offset_replaces_the_note_once_a_reference_exists(
    empty_client: TestClient, writer: Store
) -> None:
    rotation = mount()
    first = made(T0, rotation_tirs=rotation, station_id=STATION)
    writer.write(first.record)
    card = empty_client.get(f"{API}/pointing/latest").json()
    assert card["offset_arcmin"] is None
    assert card["reference_id"] is None
    assert card["quality"] == {"offset_arcmin": "no reference solution"}  # the note of the card

    # The reference that `seeingmon pointing set-reference` makes from that record.
    reference = pt.ReferenceSolution("reference-1", ref.solution_from_record(first.record, PROFILE))
    same = made(T0 + 3 * MINUTE_NS, rotation_tirs=rotation, reference=reference, station_id=STATION)
    writer.write(same.record)
    card = empty_client.get(f"{API}/pointing/latest").json()
    assert card["offset_arcmin"] == pytest.approx(0.0, abs=1e-6)
    assert card["reference_id"] == "reference-1"
    assert card["flags"] == []
    assert not card["quality"]  # no note: the card shows the value

    moved = made(
        T0 + 6 * MINUTE_NS,
        rotation_tirs=tilted(rotation, 2.0),
        reference=reference,
        station_id=STATION,
    )
    writer.write(moved.record)
    card = empty_client.get(f"{API}/pointing/latest").json()
    assert card["offset_arcmin"] == pytest.approx(2.0, rel=1e-6)
    assert card["reference_id"] == "reference-1"

    history = empty_client.get(
        f"{API}/pointing",
        params={
            "from": "2026-10-01T21:00:00Z",
            "to": "2026-10-01T23:00:00Z",
            "fields": "offset_arcmin,reference_id",
        },
    ).json()["items"]
    assert [item["offset_arcmin"] for item in history][1:] == [
        pytest.approx(0.0, abs=1e-6),
        pytest.approx(2.0, rel=1e-6),
    ]
    assert [item["reference_id"] for item in history] == [None, "reference-1", "reference-1"]


def test_a_move_beyond_the_limit_shows_the_moved_flag(
    empty_client: TestClient, writer: Store
) -> None:
    rotation = mount()
    first = made(T0, rotation_tirs=rotation, station_id=STATION)
    reference = pt.ReferenceSolution("reference-1", ref.solution_from_record(first.record, PROFILE))
    far = made(
        T0 + MINUTE_NS,
        rotation_tirs=tilted(rotation, 8.0),
        reference=reference,
        station_id=STATION,
    )
    writer.write(far.record)
    card = empty_client.get(f"{API}/pointing/latest").json()
    assert card["offset_arcmin"] == pytest.approx(8.0, rel=1e-6)
    assert card["flags"] == ["moved"]
