"""The sky view of the alignment state: the shape, the round trip, and the way it ages.

An older `core` leaves `sky` out, and a newer `core` may add fields to it. Both must decode,
like every other view of the contract.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from seeingmon.services.ipc.codec import CodecError
from seeingmon.services.web.contract import (
    MAX_STATE_BYTES,
    AlignmentState,
    SkyView,
    decode_alignment_state,
    pack_frame,
    unpack_frame,
)
from seeingmon.survey.geometry import ARCSEC_PER_RAD
from seeingmon.survey.skyview import build_sky_view
from seeingmon.survey.wcs_fit import CameraAttitude, pixel_center
from tests.services.web.helpers import alignment_state, tiny_jpeg
from tests.survey.synth import make_attitude

WIDTH, HEIGHT = 4144, 2822


def sky_view(distance_deg: float = 0.9, roll_deg: float = -65.0) -> SkyView:
    camera = CameraAttitude(
        rotation=make_attitude(distance_deg, 40.0, roll_deg),
        scale_rad_px=3.82 / ARCSEC_PER_RAD,
        parity=1,
        center_px=pixel_center(WIDTH, HEIGHT),
    )
    return SkyView.from_geometry(build_sky_view(camera, WIDTH, HEIGHT, 0.6265))


def wire(view: SkyView) -> dict[str, Any]:
    value: dict[str, Any] = json.loads(view.model_dump_json())
    return value


def test_an_alignment_state_has_no_sky_by_default() -> None:
    assert AlignmentState().sky is None
    assert alignment_state().sky is None


def test_a_state_with_a_sky_survives_the_round_trip_through_json() -> None:
    state = alignment_state(seq=4).model_copy(update={"sky": sky_view()})
    again = decode_alignment_state(json.loads(state.model_dump_json()))
    assert again == state
    assert again.sky is not None
    assert again.sky.pole.in_front
    assert again.sky.orbit is not None
    assert again.sky.orbit.fits


def test_a_state_from_an_older_core_decodes_without_a_sky() -> None:
    old = json.loads(alignment_state(seq=2).model_dump_json())
    old.pop("sky", None)
    assert "sky" not in old or old["sky"] is None
    assert decode_alignment_state(old).sky is None


def test_a_state_from_a_newer_core_with_more_fields_decodes() -> None:
    value = json.loads(alignment_state().model_copy(update={"sky": sky_view()}).model_dump_json())
    value["sky"]["horizon"] = {"altitude_deg": 41.0}
    value["sky"]["camera"]["distortion"] = [0.0, 0.1]
    value["sky"]["pole"]["refraction_arcmin"] = 1.5
    value["sky"]["orbit"]["margin_percent"] = 5.0
    state = decode_alignment_state(value)
    assert state.sky is not None
    assert state.sky.pole.in_front


def test_the_sky_has_the_documented_shape() -> None:
    view = wire(sky_view())
    assert set(view) == {"camera", "pole", "polaris_colatitude_deg", "orbit"}
    assert set(view["camera"]) == {
        "rotation",
        "scale_arcsec_px",
        "parity",
        "center_x_px",
        "center_y_px",
    }
    assert len(view["camera"]["rotation"]) == 9
    assert set(view["pole"]) == {
        "in_front",
        "x_px",
        "y_px",
        "inside_frame",
        "dx_px",
        "dy_px",
        "distance_px",
        "distance_arcmin",
        "roll_deg",
    }
    assert set(view["orbit"]) == {"fits", "margin_px", "margin_arcmin"}
    assert view["polaris_colatitude_deg"] == pytest.approx(0.6265)


def test_the_sky_takes_a_few_hundred_bytes_and_a_full_state_stays_far_below_the_limit() -> None:
    view = sky_view()
    assert len(view.model_dump_json()) < 700
    state = alignment_state(seq=1).model_copy(update={"sky": view})
    assert len(state.model_dump_json()) < MAX_STATE_BYTES // 8
    assert unpack_frame(pack_frame(state, tiny_jpeg())).state == state


def test_a_pole_behind_the_camera_has_nulls_and_still_decodes() -> None:
    camera = CameraAttitude(
        rotation=make_attitude(120.0, 0.0, 0.0),
        scale_rad_px=3.82 / ARCSEC_PER_RAD,
        parity=1,
        center_px=pixel_center(WIDTH, HEIGHT),
    )
    view = SkyView.from_geometry(build_sky_view(camera, WIDTH, HEIGHT, 0.6265))
    again = SkyView.model_validate(wire(view))
    assert not again.pole.in_front
    assert again.pole.x_px is None
    assert again.pole.dx_px is None
    assert again.orbit is not None
    assert not again.orbit.fits


def test_a_sky_without_a_colatitude_has_no_orbit() -> None:
    camera = CameraAttitude(
        rotation=make_attitude(0.5, 0.0, 0.0),
        scale_rad_px=3.82 / ARCSEC_PER_RAD,
        parity=-1,
        center_px=pixel_center(WIDTH, HEIGHT),
    )
    view = SkyView.from_geometry(build_sky_view(camera, WIDTH, HEIGHT, None))
    assert view.orbit is None
    assert view.polaris_colatitude_deg is None
    assert view.camera.parity == -1
    assert SkyView.model_validate(wire(view)) == view


def test_numbers_that_json_writes_without_a_fraction_still_decode() -> None:
    value = wire(sky_view())
    value["camera"]["center_x_px"] = 2071
    value["pole"]["dx_px"] = 12
    value["orbit"]["margin_px"] = 40
    view = SkyView.model_validate(value)
    assert view.camera.center_x_px == 2071.0
    assert view.pole.dx_px == 12.0


@pytest.mark.parametrize(
    "change",
    [
        lambda sky: sky["camera"].update(rotation=sky["camera"]["rotation"][:8]),
        lambda sky: sky["camera"].update(rotation=[*sky["camera"]["rotation"], 0.0]),
        lambda sky: sky["camera"].update(rotation="identity"),
        lambda sky: sky["camera"].update(parity=0),
        lambda sky: sky["camera"].update(parity=2),
        lambda sky: sky["camera"].update(scale_arcsec_px=0.0),
        lambda sky: sky["camera"].update(scale_arcsec_px="3.8"),
        lambda sky: sky["camera"].pop("center_x_px"),
        lambda sky: sky["pole"].update(in_front="yes"),
        lambda sky: sky["pole"].update(x_px=float("nan")),
        lambda sky: sky["pole"].update(distance_px=-1.0),
        lambda sky: sky["pole"].pop("in_front"),
        lambda sky: sky["orbit"].update(fits=1),
        lambda sky: sky["orbit"].update(margin_px=float("inf")),
        lambda sky: sky.update(polaris_colatitude_deg=95.0),
        lambda sky: sky.update(polaris_colatitude_deg=-1.0),
        lambda sky: sky.pop("camera"),
        lambda sky: sky.pop("pole"),
    ],
)
def test_a_malformed_sky_is_refused(change: Any) -> None:
    value = json.loads(alignment_state().model_copy(update={"sky": sky_view()}).model_dump_json())
    change(value["sky"])
    with pytest.raises(CodecError):
        decode_alignment_state(value)


def test_the_error_of_a_malformed_sky_names_the_fields_and_not_the_values() -> None:
    value = json.loads(alignment_state().model_copy(update={"sky": sky_view()}).model_dump_json())
    value["sky"]["camera"]["scale_arcsec_px"] = "secret-scale"
    with pytest.raises(CodecError) as raised:
        decode_alignment_state(value)
    assert "scale_arcsec_px" in str(raised.value)
    assert "secret-scale" not in str(raised.value)


def test_the_sky_is_frozen() -> None:
    view = sky_view()
    with pytest.raises(ValueError, match="frozen"):
        view.pole = view.pole  # type: ignore[misc]
