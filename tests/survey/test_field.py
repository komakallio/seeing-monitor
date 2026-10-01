"""The catalog stars of a field, in their apparent places."""

from __future__ import annotations

import numpy as np
import pytest

from seeingmon.survey import apparent
from seeingmon.survey.field import FIELD_MARGIN_DEG, catalog_field, half_diagonal_deg
from seeingmon.survey.geometry import ARCSEC_PER_RAD, angular_separation
from seeingmon.survey.wcs_fit import CameraAttitude
from tests.survey import synth

T = synth.NIGHT_UTC_NS
WIDTH, HEIGHT = 4144, 2822
SCALE = 3.82 / ARCSEC_PER_RAD


def attitude() -> CameraAttitude:
    epoch = apparent.epoch_from_utc_ns(T)
    tirs = synth.make_attitude(0.9, 40.0, 25.0)
    return CameraAttitude(
        tirs @ apparent.cirs_to_earth_fixed(epoch.era_rad),
        SCALE,
        1,
        ((WIDTH - 1) / 2.0, (HEIGHT - 1) / 2.0),
    )


def test_the_half_diagonal_follows_the_plate_scale() -> None:
    # 4144 x 2822 pixels at 3.82 arcsec per pixel: the diagonal is 5.3 degrees.
    assert half_diagonal_deg(attitude(), WIDTH, HEIGHT) == pytest.approx(
        0.5 * np.hypot(WIDTH, HEIGHT) * 3.82 / 3600.0, rel=1e-9
    )


def test_the_field_holds_the_stars_in_and_near_the_frame_and_no_others() -> None:
    catalog = synth.synthetic_catalog(cap_radius_deg=8.0, density_scale=1.0, seed=3)
    model = attitude()
    epoch = apparent.epoch_from_utc_ns(T)
    rows, vectors = catalog_field(catalog, model, epoch, width_px=WIDTH, height_px=HEIGHT)
    assert rows.size > 1000
    assert vectors.shape == (rows.size, 3)
    # Every star of the frame is in the field, with its apparent place.
    x, y, front = model.project(vectors)
    inside = front & (x >= 0) & (x < WIDTH) & (y >= 0) & (y < HEIGHT)
    assert inside.sum() > 800
    everywhere = apparent.apparent_vectors(
        catalog.ra_deg,
        catalog.dec_deg,
        catalog.pm_ra_mas_yr,
        catalog.pm_dec_mas_yr,
        catalog.parallax_mas,
        epoch,
        catalog_epoch_jyear=catalog.epoch_jyear,
    )
    ax, ay, afront = model.project(everywhere)
    in_frame_truth = np.flatnonzero(afront & (ax >= 0) & (ax < WIDTH) & (ay >= 0) & (ay < HEIGHT))
    assert set(in_frame_truth) <= set(rows)
    np.testing.assert_allclose(vectors, everywhere[rows], atol=1e-15)
    # The field reaches no farther than the half diagonal plus the margin.
    separation = np.degrees(angular_separation(vectors, model.boresight()))
    assert separation.max() < half_diagonal_deg(model, WIDTH, HEIGHT) + FIELD_MARGIN_DEG + 0.01


def test_a_magnitude_limit_and_an_empty_field() -> None:
    catalog = synth.synthetic_catalog(cap_radius_deg=8.0, density_scale=1.0, seed=3)
    model = attitude()
    epoch = apparent.epoch_from_utc_ns(T)
    bright, _ = catalog_field(
        catalog, model, epoch, width_px=WIDTH, height_px=HEIGHT, max_g_mag=10.0
    )
    everything, _ = catalog_field(catalog, model, epoch, width_px=WIDTH, height_px=HEIGHT)
    assert 0 < bright.size < everything.size
    assert np.all(catalog.g_mag[bright] < 10.0)
    # A camera that looks away from the cap sees nothing.
    away = CameraAttitude(
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
        SCALE,
        1,
        model.center_px,
    )
    rows, vectors = catalog_field(catalog, away, epoch, width_px=WIDTH, height_px=HEIGHT)
    assert rows.size == 0
    assert vectors.shape == (0, 3)
