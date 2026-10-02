"""The sky view of the alignment state: the shape, the round trip, and the way it ages.

An older `core` leaves `sky` out, and a newer `core` may add fields to it. Both must decode,
like every other view of the contract.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest

from seeingmon.services.ipc.codec import CodecError
from seeingmon.services.web.contract import (
    MAX_STATE_BYTES,
    AlignmentState,
    ReticleView,
    SkyView,
    decode_alignment_state,
    pack_frame,
    unpack_frame,
)
from seeingmon.survey.geometry import ARCSEC_PER_RAD
from seeingmon.survey.skyview import build_sky_view, reticle_geometry, zenith_vector
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
    x, y, _ = camera.project(np.array([math.sin(0.0109), 0.0, math.cos(0.0109)]))
    geometry = build_sky_view(
        camera,
        WIDTH,
        HEIGHT,
        0.6265,
        polaris_xy=(float(x[0]), float(y[0])),
        zenith=zenith_vector(50.0, 10.0, 1.2),  # a synthetic site
    )
    return SkyView.from_geometry(geometry)


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
    assert set(view) == {
        "camera",
        "pole",
        "polaris_colatitude_deg",
        "orbit",
        "aim",
        "aim_ring",
        "altitude_arcmin",
        "azimuth_arcmin",
        "axes",
    }
    assert set(view["aim"]) == {"x_px", "y_px", "dx_px", "dy_px", "distance_arcmin"}
    assert set(view["aim_ring"]) == {"x_px", "y_px"}
    assert set(view["axes"]) == {"altitude_dx", "altitude_dy", "azimuth_dx", "azimuth_dy"}
    assert isinstance(view["altitude_arcmin"], float)
    assert isinstance(view["azimuth_arcmin"], float)
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


def test_the_sky_takes_about_a_kilobyte_and_a_full_state_stays_far_below_the_limit() -> None:
    view = sky_view()
    assert len(view.model_dump_json()) < 1300
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


def test_a_sky_from_a_core_without_the_aim_decodes_with_nulls() -> None:
    value = wire(sky_view())
    for name in ("aim", "aim_ring", "altitude_arcmin", "azimuth_arcmin", "axes"):
        value.pop(name)
    view = SkyView.model_validate(value)
    assert view.aim is None
    assert view.aim_ring is None
    assert view.altitude_arcmin is None
    assert view.axes is None
    assert view.pole.in_front


def test_a_sky_without_a_site_has_the_aim_and_the_ring_but_no_move() -> None:
    camera = CameraAttitude(
        rotation=make_attitude(0.5, 10.0, 20.0),
        scale_rad_px=3.82 / ARCSEC_PER_RAD,
        parity=1,
        center_px=pixel_center(WIDTH, HEIGHT),
    )
    view = SkyView.from_geometry(
        build_sky_view(camera, WIDTH, HEIGHT, 0.6265, polaris_xy=(900.0, 800.0))
    )
    assert view.aim is not None
    assert view.aim.distance_arcmin == pytest.approx(30.0, abs=0.05)
    assert view.aim_ring is not None
    assert view.altitude_arcmin is None
    assert view.azimuth_arcmin is None
    assert view.axes is None
    assert SkyView.model_validate(wire(view)) == view


def test_the_reticle_survives_the_round_trip_and_the_state_decodes_without_it() -> None:
    geometry = reticle_geometry(WIDTH, HEIGHT, 3.82, 0.6265)
    assert geometry is not None
    reticle = ReticleView(
        x_px=geometry.x_px,
        y_px=geometry.y_px,
        radius_px=geometry.radius_px,
        polaris_colatitude_deg=0.6265,
    )
    state = alignment_state(seq=2).model_copy(update={"reticle": reticle, "sky": sky_view()})
    again = decode_alignment_state(json.loads(state.model_dump_json()))
    assert again == state
    assert again.reticle is not None
    assert again.reticle.radius_px == pytest.approx(590.44, abs=0.01)
    older = json.loads(alignment_state(seq=2).model_dump_json())
    older.pop("reticle", None)
    assert decode_alignment_state(older).reticle is None


@pytest.mark.parametrize(
    "change",
    [
        lambda sky: sky["camera"].update(rotation=sky["camera"]["rotation"][:8]),
        lambda sky: sky["aim"].update(x_px="left"),
        lambda sky: sky["aim"].update(distance_arcmin=-1.0),
        lambda sky: sky["aim"].pop("y_px"),
        lambda sky: sky["aim_ring"].update(x_px=float("nan")),
        lambda sky: sky["axes"].pop("azimuth_dy"),
        lambda sky: sky.update(altitude_arcmin="up"),
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


@pytest.mark.parametrize(
    "change",
    [
        lambda reticle: reticle.update(radius_px=0.0),
        lambda reticle: reticle.update(radius_px=-5.0),
        lambda reticle: reticle.update(radius_px="large"),
        lambda reticle: reticle.update(x_px=float("inf")),
        lambda reticle: reticle.pop("y_px"),
        lambda reticle: reticle.update(polaris_colatitude_deg=95.0),
    ],
)
def test_a_malformed_reticle_is_refused(change: Any) -> None:
    value = json.loads(alignment_state().model_dump_json())
    value["reticle"] = {"x_px": 2071.5, "y_px": 1410.5, "radius_px": 590.4}
    change(value["reticle"])
    with pytest.raises(CodecError):
        decode_alignment_state(value)
