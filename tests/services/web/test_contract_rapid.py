"""The rapid focus part of the contract: the readings, the view, and where they travel."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from seeingmon.services.ipc.codec import CodecError
from seeingmon.services.web.contract import (
    MAX_RAPID_READINGS,
    MAX_STATE_BYTES,
    AlignmentState,
    PolarisState,
    RapidFocusView,
    RapidReadingsView,
    decode_alignment_state,
    pack_polaris_frame,
    unpack_polaris_frame,
)
from tests.services.web.helpers import alignment_state, polaris_state, tiny_png


def readings_example(count: int = 5, *, session: int = 1) -> dict[str, Any]:
    return {
        "session": session,
        "reset": True,
        "index": list(range(1, count + 1)),
        "t_utc_ms": [1_800_000_000_000 + 50 * n for n in range(count)],
        "fwhm_arcsec": [4.0 - 0.01 * n for n in range(count)],
        "peak_fraction": [0.31] * count,
        "n_frames": [4] * count,
        "spike": [False] * count,
        "saturated": [False] * count,
    }


def view_example(count: int = 5, **changes: Any) -> RapidFocusView:
    fields: dict[str, Any] = {
        "available": True,
        "active": True,
        "since_utc": "2026-10-05T21:00:00.000Z",
        "mode": "bin1",
        "exposure_us": 2000,
        "gain": 0,
        "roi": {"x": 4080, "y": 2758, "width": 128, "height": 128},
        "scale_arcsec_px": 1.91,
        "n_stars": 1,
        "fwhm_arcsec": 3.96,
        "best_fwhm_arcsec": 3.9,
        "peak_fraction": 0.31,
        "readings": readings_example(count),
    }
    fields.update(changes)
    return RapidFocusView.model_validate(fields)


def test_the_readings_survive_the_round_trip_through_json() -> None:
    readings = RapidReadingsView.model_validate(readings_example())
    assert RapidReadingsView.model_validate_json(readings.model_dump_json()) == readings
    assert readings.reset is True  # a message that says nothing is a whole history


def test_readings_in_lists_of_different_length_are_refused() -> None:
    value = readings_example()
    value["saturated"] = [False]
    with pytest.raises(ValidationError, match="differ in length"):
        RapidReadingsView.model_validate(value)


def test_the_history_holds_600_readings_and_no_more() -> None:
    assert MAX_RAPID_READINGS == 600
    assert len(RapidReadingsView.model_validate(readings_example(600)).index) == 600
    with pytest.raises(ValidationError):
        RapidReadingsView.model_validate(readings_example(601))


def test_a_view_that_nothing_runs_has_only_the_offer() -> None:
    view = RapidFocusView(
        available=False,
        reason="the stars are too wide for the rapid mode: 21 arcsec, the limit is 12",
        coarse_fwhm_arcsec=21.0,
        max_fwhm_arcsec=12.0,
    )
    assert (view.active, view.readings, view.n_stars, view.fwhm_arcsec) == (False, None, None, None)
    assert (view.spike, view.saturated, view.quality) == (False, False, {})
    assert RapidFocusView.model_validate_json(view.model_dump_json()) == view


def test_a_running_view_survives_the_round_trip_through_json() -> None:
    view = view_example()
    again = RapidFocusView.model_validate_json(view.model_dump_json())
    assert again == view
    assert isinstance(again.readings, RapidReadingsView)


def test_the_view_is_strict_about_its_types() -> None:
    with pytest.raises(ValidationError):
        RapidFocusView.model_validate({"available": "yes"})
    with pytest.raises(ValidationError):
        RapidFocusView.model_validate({"available": True, "located_by": "a guess"})
    with pytest.raises(ValidationError):
        RapidFocusView.model_validate({"available": True, "max_fwhm_arcsec": 0.0})


@pytest.mark.parametrize("located_by", ["current solution", "last solution", "brightest star"])
def test_the_view_says_what_located_polaris(located_by: str) -> None:
    view = RapidFocusView.model_validate({"available": True, "located_by": located_by})
    assert view.located_by == located_by


def test_the_alignment_state_carries_the_view_and_decodes_it() -> None:
    state = alignment_state().model_copy(update={"rapid_focus": view_example()})
    wire = json.loads(state.model_dump_json())
    assert wire["rapid_focus"]["readings"]["reset"] is True
    assert decode_alignment_state(wire) == state


def test_an_alignment_state_from_an_older_core_has_no_rapid_focus() -> None:
    assert decode_alignment_state({"active": True}).rapid_focus is None
    assert AlignmentState(active=False).rapid_focus is None
    with pytest.raises(CodecError):
        decode_alignment_state({"active": True, "rapid_focus": {"available": "yes"}})


def test_a_polaris_frame_of_the_mode_carries_the_view_in_its_state() -> None:
    state = polaris_state(live=False).model_copy(update={"rapid_focus": view_example(600)})
    payload = pack_polaris_frame(state, tiny_png(40, (128, 128)))
    frame = unpack_polaris_frame(payload)
    assert frame.state == state
    assert frame.state.rapid_focus is not None
    assert len(frame.state.rapid_focus.readings.index) == 600  # type: ignore[union-attr]


def test_the_whole_history_in_a_polaris_state_stays_well_inside_the_limit() -> None:
    """Core sends the whole history in each message, and `web` cuts it for each viewer."""
    state = polaris_state(live=False).model_copy(update={"rapid_focus": view_example(600)})
    size = len(state.model_dump_json())
    assert size < MAX_STATE_BYTES // 2
    assert 15_000 < size < 35_000  # about 47 bytes for each of 600 readings, and the rest


def test_a_state_without_the_mode_stays_as_small_as_before() -> None:
    assert PolarisState.model_validate_json(polaris_state().model_dump_json()).rapid_focus is None
    assert len(polaris_state().model_dump_json()) < 1500
