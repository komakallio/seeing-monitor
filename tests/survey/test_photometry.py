"""Aperture photometry on images that the tests draw: flux, noise, trails, and what it refuses."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from seeingmon.survey import photometry as ph
from seeingmon.survey._scipy import erf
from seeingmon.survey.detect import Detections, detect_stars

E_PER_ADU = 0.88
READ_NOISE_E = 1.85
SATURATION_DN = 16383.0
OFFSET_DN = 120.0
FloatArray = npt.NDArray[np.float64]


def add_star(
    electrons: FloatArray,
    x: float,
    y: float,
    flux_e: float,
    sigma: float = 0.8,
    *,
    trail_px: float = 0.0,
    angle_rad: float = 0.0,
    n_sub: int = 40,
) -> None:
    """Add a Gaussian star that moves along a line during the exposure, in electrons."""
    height, width = electrons.shape
    reach = int(np.ceil(5.0 * sigma + trail_px / 2.0 + 2.0))
    cx, cy = round(x), round(y)
    px = np.arange(max(cx - reach, 0), min(cx + reach + 1, width))
    py = np.arange(max(cy - reach, 0), min(cy + reach + 1, height))
    along = (np.arange(n_sub) + 0.5) / n_sub - 0.5
    xs = x + along * trail_px * np.cos(angle_rad)
    ys = y + along * trail_px * np.sin(angle_rad)
    edges_x = np.concatenate([px - 0.5, [px[-1] + 0.5]])
    edges_y = np.concatenate([py - 0.5, [py[-1] + 0.5]])
    root2 = np.sqrt(2.0)
    phi_x = 0.5 * np.diff(erf((edges_x[None, :] - xs[:, None]) / (sigma * root2)), axis=1)
    phi_y = 0.5 * np.diff(erf((edges_y[None, :] - ys[:, None]) / (sigma * root2)), axis=1)
    stamp = (phi_y.T @ phi_x) * (flux_e / n_sub)
    electrons[py[0] : py[-1] + 1, px[0] : px[-1] + 1] += stamp


def to_counts(electrons: FloatArray, *, seed: int | None = None) -> npt.NDArray[np.float32]:
    """Electrons to counts: with a seed, add Poisson and read noise first."""
    if seed is not None:
        rng = np.random.default_rng(seed)
        electrons = rng.poisson(electrons) + rng.normal(0.0, READ_NOISE_E, electrons.shape)
    return np.asarray(electrons / E_PER_ADU + OFFSET_DN, dtype=np.float32)


def measure(
    data: npt.NDArray[np.float32],
    x: FloatArray,
    y: FloatArray,
    trail: float | FloatArray = 0.0,
    angle: float | FloatArray = 0.0,
    bad: npt.NDArray[np.bool_] | None = None,
) -> ph.AperturePhotometry:
    trails = np.broadcast_to(np.asarray(trail, dtype=np.float64), x.shape).copy()
    angles = np.broadcast_to(np.asarray(angle, dtype=np.float64), x.shape).copy()
    return ph.aperture_photometry(
        data,
        x,
        y,
        trails,
        angles,
        e_per_adu=E_PER_ADU,
        saturation_dn=SATURATION_DN,
        bad=bad,
    )


def matched(
    data: npt.NDArray[np.float32],
    detections: Detections,
    cat_row: npt.NDArray[np.intp],
    *,
    flat_at: FloatArray | None = None,
    origin_px: tuple[float, float] = (0.0, 0.0),
    min_snr: float = 0.0,
) -> ph.StarPhotometry:
    return ph.measure_matched_stars(
        data,
        detections,
        cat_row,
        exposure_s=30.0,
        e_per_adu=E_PER_ADU,
        saturation_dn=SATURATION_DN,
        flat_at=flat_at,
        origin_px=origin_px,
        min_snr=min_snr,
    )


def test_a_gaussian_star_gives_its_flux_and_the_sky_level_without_noise() -> None:
    sky_e = 130.0
    electrons = np.full((120, 160), sky_e)
    add_star(electrons, 80.3, 60.7, 50_000.0)
    result = measure(to_counts(electrons), np.array([80.3]), np.array([60.7]))
    assert result.ok[0]
    assert result.flux_e[0] == pytest.approx(50_000.0, rel=1e-3)
    assert result.background_dn[0] == pytest.approx(sky_e / E_PER_ADU + OFFSET_DN, rel=1e-3)
    assert result.noise_e_px[0] < 0.05  # no noise was added
    assert result.aperture_px2[0] == pytest.approx(np.pi * 5.0**2, rel=0.02)


def test_the_error_matches_the_scatter_of_many_noisy_stars() -> None:
    """A grid of equal stars with Poisson and read noise: the stated error is the real one."""
    sky_e = 130.0
    flux = 20_000.0
    rng = np.random.default_rng(1)
    columns, rows = np.meshgrid(np.arange(30.0, 480.0, 30.0), np.arange(30.0, 330.0, 30.0))
    x = columns.ravel() + rng.uniform(-0.5, 0.5, columns.size)
    y = rows.ravel() + rng.uniform(-0.5, 0.5, rows.size)
    electrons = np.full((360, 510), sky_e)
    for xi, yi in zip(x, y, strict=True):
        add_star(electrons, xi, yi, flux)
    result = measure(to_counts(electrons, seed=2), x, y)
    assert result.ok.all()
    measured = result.flux_e
    assert np.mean(measured) == pytest.approx(flux, rel=0.01)
    scatter = float(np.std(measured, ddof=1))
    predicted = float(np.mean(result.error_e))
    assert scatter == pytest.approx(predicted, rel=0.2)
    # Poisson noise of the star plus the noise of about 80 sky pixels.
    expected = np.sqrt(flux + 78.5 * (sky_e + READ_NOISE_E**2))
    assert predicted == pytest.approx(expected, rel=0.1)


def test_a_trail_needs_the_stadium_aperture() -> None:
    electrons = np.full((140, 180), 130.0)
    angle = 0.7
    add_star(electrons, 90.0, 70.0, 40_000.0, trail_px=10.0, angle_rad=angle)
    data = to_counts(electrons)
    x, y = np.array([90.0]), np.array([70.0])
    with_trail = measure(data, x, y, trail=10.0, angle=angle)
    without = measure(data, x, y, trail=0.0, angle=0.0)
    assert with_trail.flux_e[0] == pytest.approx(40_000.0, rel=2e-3)
    assert without.flux_e[0] < 0.95 * 40_000.0  # a round aperture cuts the trail
    assert with_trail.aperture_px2[0] > np.pi * 25.0 + 2.0 * 5.0 * 10.0 - 5.0


def test_the_background_ignores_a_neighbor_and_a_hot_pixel_in_the_ring() -> None:
    electrons = np.full((140, 180), 130.0)
    add_star(electrons, 90.0, 70.0, 30_000.0)
    add_star(electrons, 90.0 + 11.0, 70.0, 20_000.0)  # in the ring, 11 px away
    data = to_counts(electrons)
    data[70 + 12, 90] += 4000.0  # a hot pixel in the ring
    data[70 - 10, 90 + 1] = SATURATION_DN  # a saturated pixel in the ring
    result = measure(data, np.array([90.0]), np.array([70.0]))
    assert result.ok[0]
    assert result.background_dn[0] == pytest.approx(130.0 / E_PER_ADU + OFFSET_DN, rel=0.01)
    assert result.flux_e[0] == pytest.approx(30_000.0, rel=0.01)


def test_a_bad_pixel_in_the_aperture_spoils_the_star() -> None:
    electrons = np.full((140, 180), 130.0)
    add_star(electrons, 90.0, 70.0, 30_000.0)
    add_star(electrons, 40.0, 40.0, 30_000.0)
    add_star(electrons, 140.0, 100.0, 30_000.0)
    data = to_counts(electrons)
    data[70, 91] = SATURATION_DN  # saturated, in the first aperture
    bad = np.zeros(data.shape, dtype=bool)
    bad[41, 40] = True  # masked as hot, in the second aperture
    result = measure(data, np.array([90.0, 40.0, 140.0]), np.array([70.0, 40.0, 100.0]), bad=bad)
    assert list(result.ok) == [False, False, True]
    assert np.isnan(result.flux_e[0])
    assert np.isnan(result.flux_e[1])
    # A masked pixel in the ring only is fine.
    ring_bad = np.zeros(data.shape, dtype=bool)
    ring_bad[100 + 11, 140] = True
    again = measure(data, np.array([140.0]), np.array([100.0]), bad=ring_bad)
    assert again.ok[0]


def test_a_star_whose_aperture_leaves_the_frame_has_no_measurement() -> None:
    electrons = np.full((140, 180), 130.0)
    columns = np.array([3.0, 8.0, 90.0])
    rows = np.array([30.0, 100.0, 70.0])
    for column, row in zip(columns, rows, strict=True):
        add_star(electrons, column, row, 30_000.0)
    result = measure(to_counts(electrons), columns, rows)
    # The aperture of the first star reaches past the edge. The ring of the second is cut, but
    # more than the minimum remains, and its aperture is whole.
    assert list(result.ok) == [False, True, True]
    assert result.flux_e[1] == pytest.approx(30_000.0, rel=0.01)


def test_a_ring_with_too_few_pixels_gives_no_measurement() -> None:
    electrons = np.full((140, 180), 130.0)
    add_star(electrons, 90.0, 70.0, 30_000.0)
    data = to_counts(electrons)
    bad = np.zeros(data.shape, dtype=bool)
    bad[70 - 15 : 70 + 16, 90 - 15 : 90 + 16] = True
    bad[70 - 6 : 70 + 7, 90 - 6 : 90 + 7] = False  # the aperture stays clean
    result = measure(data, np.array([90.0]), np.array([70.0]), bad=bad)
    assert not result.ok[0]


def test_the_options_refuse_nonsense() -> None:
    with pytest.raises(ValueError, match="radii"):
        ph.PhotometryOptions(aperture_px=6.0, annulus_inner_px=5.0)
    with pytest.raises(ValueError, match="clipping"):
        ph.PhotometryOptions(min_annulus_px=2)


# --- Neighbors -----------------------------------------------------------------------------


def test_a_bright_neighbor_inside_the_aperture_makes_a_star_not_isolated() -> None:
    x = np.array([100.0, 106.0, 300.0, 100.0])
    y = np.array([100.0, 100.0, 100.0, 120.0])
    flux = np.array([10_000.0, 5_000.0, 10_000.0, 10_000.0])
    trail = np.zeros(4)
    alone = ph.isolated(x, y, flux, trail, x, y, flux)
    # Star 0 has star 1 (half its flux) 6 px away, and star 1 has star 0 (twice its flux).
    assert list(alone) == [False, False, True, True]
    faint = np.array([10_000.0, 100.0, 10_000.0, 10_000.0])  # 1% of the flux: below the limit
    assert ph.isolated(x, y, faint, trail, x, y, faint)[0]
    assert not ph.isolated(x, y, faint, trail, x, y, faint)[1]  # but star 1 sees a bright one


def test_a_trail_widens_the_neighborhood() -> None:
    x = np.array([100.0, 111.0])
    y = np.array([100.0, 100.0])
    flux = np.array([10_000.0, 10_000.0])
    assert ph.isolated(x, y, flux, np.zeros(2), x, y, flux).all()  # 11 px apart is clear
    assert not ph.isolated(x, y, flux, np.full(2, 6.0), x, y, flux).any()  # 5 + 3 + 3 reaches


def test_isolation_of_nothing_is_empty() -> None:
    empty = np.zeros(0)
    assert ph.isolated(empty, empty, empty, empty, empty, empty, empty).size == 0


# --- The matched stars of a detected frame -------------------------------------------------


def star_field(n_hot: int = 0) -> tuple[npt.NDArray[np.float32], FloatArray, FloatArray]:
    """A 400 x 300 frame with a grid of stars of rising flux, and the sky."""
    electrons = np.full((300, 400), 130.0)
    columns = np.array([60.0, 120.0, 180.0, 240.0, 300.0, 350.0])
    rows = np.array([60.0, 130.0, 200.0, 250.0])
    x, y = np.meshgrid(columns, rows)
    x, y = x.ravel(), y.ravel()
    fluxes = np.geomspace(3_000.0, 60_000.0, x.size)
    for xi, yi, flux in zip(x, y, fluxes, strict=True):
        add_star(electrons, xi, yi, flux)
    return to_counts(electrons, seed=3), x, y


def test_the_matched_stars_come_with_a_rate_an_error_and_a_signal_to_noise_ratio() -> None:
    data, x, y = star_field()
    detections = detect_stars(data, saturation_dn=SATURATION_DN, e_per_adu=E_PER_ADU)
    cat_row = np.arange(len(detections), dtype=np.intp)
    result = ph.measure_matched_stars(
        data,
        detections,
        cat_row,
        exposure_s=30.0,
        e_per_adu=E_PER_ADU,
        saturation_dn=SATURATION_DN,
    )
    assert len(result) == x.size
    fluxes = np.geomspace(3_000.0, 60_000.0, x.size)
    # Match each measurement to the star that was drawn at its position.
    truth_rate = (
        np.array(
            [
                fluxes[int(np.argmin(np.hypot(x - px, y - py)))]
                for px, py in zip(result.x, result.y, strict=True)
            ]
        )
        / 30.0
    )
    np.testing.assert_allclose(result.rate_e_per_s, truth_rate, rtol=0.05)
    assert np.all(result.snr > 20.0)
    np.testing.assert_allclose(result.mag_error, 2.5 / np.log(10.0) / result.snr)
    assert np.array_equal(result.cat_row, cat_row[result.index])


def test_unmatched_flagged_and_dim_stars_stay_out() -> None:
    data, _, _ = star_field()
    detections = detect_stars(data, saturation_dn=SATURATION_DN, e_per_adu=E_PER_ADU)
    cat_row = np.arange(len(detections), dtype=np.intp)
    cat_row[::3] = -1  # a third of the detections matched no catalog star
    result = matched(data, detections, cat_row)
    assert len(result) == int(np.sum(cat_row >= 0))
    assert not np.any(np.isin(result.index, np.flatnonzero(cat_row < 0)))
    bright = matched(data, detections, np.arange(len(detections)), min_snr=60.0)
    assert 0 < len(bright) < len(detections)
    assert np.all(bright.snr >= 60.0)


def test_a_flat_field_scales_the_rate_and_an_origin_shifts_the_positions() -> None:
    data, _, _ = star_field()
    detections = detect_stars(data, saturation_dn=SATURATION_DN, e_per_adu=E_PER_ADU)
    cat_row = np.arange(len(detections), dtype=np.intp)
    plain = matched(data, detections, cat_row)
    flat = np.full(len(detections), 0.8)
    dimmer = matched(data, detections, cat_row, flat_at=flat)
    np.testing.assert_allclose(dimmer.rate_e_per_s, plain.rate_e_per_s / 0.8)
    # The same frame as a crop of the sensor: detections carry sensor positions.
    moved = detections.shifted(1000.0, 500.0)
    shifted = matched(data, moved, cat_row, origin_px=(1000.0, 500.0))
    np.testing.assert_allclose(shifted.rate_e_per_s, plain.rate_e_per_s)
    np.testing.assert_allclose(shifted.x, plain.x + 1000.0)


def test_a_saturated_star_is_not_measured() -> None:
    electrons = np.full((300, 400), 130.0)
    add_star(electrons, 100.0, 100.0, 5_000_000.0)  # far above saturation
    add_star(electrons, 250.0, 150.0, 20_000.0)
    data = np.minimum(to_counts(electrons, seed=4), SATURATION_DN)
    detections = detect_stars(data, saturation_dn=SATURATION_DN, e_per_adu=E_PER_ADU)
    cat_row = np.arange(len(detections), dtype=np.intp)
    result = ph.measure_matched_stars(
        data,
        detections,
        cat_row,
        exposure_s=30.0,
        e_per_adu=E_PER_ADU,
        saturation_dn=SATURATION_DN,
    )
    assert len(result) == 1
    assert result.x[0] == pytest.approx(250.0, abs=0.3)
