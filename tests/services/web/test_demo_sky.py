"""The sky view of the demo: a pole that drifts through the center, with Polaris on its orbit.

The demo has no camera, but its alignment frames carry the same `sky` view that `core` builds for
a real solution (`build_sky_view`), so the Align page shows the pole, the orbit, and the grid.
"""

from __future__ import annotations

import math
import sys

import pytest

from seeingmon.services.web.demo import (
    FRAME_CENTER_X,
    FRAME_CENTER_Y,
    FRAME_HEIGHT_PX,
    FRAME_PERIOD_S,
    FRAME_WIDTH_PX,
    PLATE_SCALE_ARCSEC_PX,
    POLE_DRIFT_PERIOD_S,
    SKY_COLATITUDE_DEG,
    StarField,
    demo_sky,
    pole_offset_px,
)

pytest.importorskip("erfa", reason="the camera model of the survey path needs the survey extra")

PX_PER_DEG = 3600.0 / PLATE_SCALE_ARCSEC_PX
TIGHT_FRACTION = 0.05  # the page paints the orbit amber below this share of the shorter side


@pytest.fixture(scope="module")
def field() -> StarField:
    return StarField(stars=20)  # the stars do not matter here, and fewer make a frame faster


def orbit_state(seq: int, field: StarField) -> str:
    sky = field.frame(seq).state.sky
    assert sky is not None
    assert sky.orbit is not None
    if not sky.orbit.fits:
        return "bad"
    shorter = min(FRAME_WIDTH_PX, FRAME_HEIGHT_PX)
    return "good" if sky.orbit.margin_px >= TIGHT_FRACTION * shorter else "tight"


def test_the_pole_starts_nine_tenths_of_a_degree_right_and_four_tenths_above_the_center() -> None:
    right, down = pole_offset_px(0.0)
    assert right == pytest.approx(0.9 * PX_PER_DEG, abs=0.5)
    assert down == pytest.approx(-0.4 * PX_PER_DEG, abs=0.5)


def test_the_first_frame_has_a_sky_with_the_pole_where_the_drift_puts_it(field: StarField) -> None:
    sky = field.frame(0).state.sky
    assert sky is not None
    assert sky.pole.in_front
    assert sky.pole.inside_frame
    assert sky.pole.dx_px == pytest.approx(0.9 * PX_PER_DEG, abs=1.0)
    assert sky.pole.dy_px == pytest.approx(-0.4 * PX_PER_DEG, abs=1.0)
    assert sky.pole.distance_arcmin == pytest.approx(math.hypot(0.9, 0.4) * 60.0, abs=0.2)
    assert sky.polaris_colatitude_deg == SKY_COLATITUDE_DEG
    assert sky.camera.scale_arcsec_px == pytest.approx(PLATE_SCALE_ARCSEC_PX)
    assert sky.camera.center_x_px == pytest.approx(FRAME_CENTER_X)
    assert sky.camera.center_y_px == pytest.approx(FRAME_CENTER_Y)


def test_the_pole_follows_the_drift_through_the_center_and_back(field: StarField) -> None:
    for seq in (0, 20, 75, 150, 233, 300):
        t = seq * FRAME_PERIOD_S
        right, down = pole_offset_px(t)
        sky = field.frame(seq).state.sky
        assert sky is not None
        assert sky.pole.dx_px == pytest.approx(right, abs=1.0), seq
        assert sky.pole.dy_px == pytest.approx(down, abs=1.0), seq
    # Half a period in, the pole is near the center, with only the wobble left.
    half = POLE_DRIFT_PERIOD_S / 2.0
    right, down = pole_offset_px(half)
    assert math.hypot(right, down) < 60.0
    assert pole_offset_px(0.0) != pole_offset_px(half)
    assert pole_offset_px(POLE_DRIFT_PERIOD_S)[0] == pytest.approx(pole_offset_px(0.0)[0], abs=45.0)


def test_polaris_lies_on_its_orbit_around_the_pole(field: StarField) -> None:
    for seq in (0, 40, 150, 260):
        state = field.frame(seq).state
        assert state.sky is not None
        assert state.solved is not None
        center_x, center_y = FRAME_CENTER_X, FRAME_CENTER_Y
        pole_x = center_x + (state.sky.pole.dx_px or 0.0)
        pole_y = center_y + (state.sky.pole.dy_px or 0.0)
        distance = math.hypot(state.solved.x_px - pole_x, state.solved.y_px - pole_y)
        assert distance == pytest.approx(SKY_COLATITUDE_DEG * PX_PER_DEG, rel=0.01), seq


def test_the_target_is_where_polaris_belongs_when_the_pole_sits_at_the_center(
    field: StarField,
) -> None:
    assert field.target_x == FRAME_CENTER_X
    assert field.target_y == pytest.approx(
        FRAME_CENTER_Y + SKY_COLATITUDE_DEG * PX_PER_DEG, abs=0.01
    )


def test_the_orbit_is_red_amber_and_green_in_turn(field: StarField) -> None:
    states = {orbit_state(seq, field) for seq in range(0, 320, 5)}
    assert states == {"bad", "tight", "good"}
    assert orbit_state(0, field) == "bad"  # the pole starts far out, so the circle leaves the frame
    assert orbit_state(round(POLE_DRIFT_PERIOD_S / 2 / FRAME_PERIOD_S), field) == "good"


def test_the_numbers_of_the_orbit_are_consistent_with_the_state(field: StarField) -> None:
    for seq in range(0, 320, 20):
        sky = field.frame(seq).state.sky
        assert sky is not None
        assert sky.orbit is not None
        assert sky.orbit.fits == (sky.orbit.margin_px >= 0.0)
        assert sky.orbit.margin_arcmin == pytest.approx(
            (sky.orbit.margin_px or 0.0) * PLATE_SCALE_ARCSEC_PX / 60.0, abs=0.05
        )


def test_the_sky_stays_small(field: StarField) -> None:
    sky = field.frame(10).state.sky
    assert sky is not None
    assert len(sky.model_dump_json()) < 700


def test_a_demo_without_the_survey_extra_shows_no_sky_and_everything_else(
    monkeypatch: pytest.MonkeyPatch, field: StarField
) -> None:
    monkeypatch.setitem(sys.modules, "seeingmon.survey.wcs_fit", None)  # an import now fails
    assert demo_sky(100.0, -50.0, 0.0) is None
    state = field.frame(4).state
    assert state.sky is None
    assert state.solved is not None
    assert state.offset is not None
    assert state.target is not None
