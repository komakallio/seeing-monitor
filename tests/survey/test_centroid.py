"""Model fits of star stamps: the profile, the fit, and what it does for undersampled stars."""

from __future__ import annotations

import numpy as np
import pytest

from seeingmon.survey import centroid as c
from seeingmon.survey.geometry import FloatArray

HALF = 8


def brute_force_stamp(
    x0: float,
    y0: float,
    sigma: float,
    dx: float,
    dy: float,
    *,
    half: int = HALF,
    oversample: int = 30,
    n_sub: int = 100,
) -> FloatArray:
    """The unit-flux profile by brute-force integration: a fine grid over each pixel, and many
    positions along the trail. It shares no code with `stamp_model`."""
    pixels = np.arange(-half, half + 1)
    offsets = (np.arange(oversample) + 0.5) / oversample - 0.5
    fine = (pixels[:, None] + offsets[None, :]).ravel()
    image = np.zeros((pixels.size, pixels.size))
    for s in (np.arange(n_sub) + 0.5) / n_sub - 0.5:
        gx = np.exp(-0.5 * ((fine - (x0 + dx * s)) / sigma) ** 2)
        gy = np.exp(-0.5 * ((fine - (y0 + dy * s)) / sigma) ** 2)
        cells = np.outer(gy, gx).reshape(pixels.size, oversample, pixels.size, oversample)
        image += cells.mean(axis=(1, 3))
    return image / (n_sub * 2.0 * np.pi * sigma**2)


def model_stamp(
    x0: float, y0: float, sigma: float, dx: float, dy: float, n_sub: int = 24
) -> tuple[FloatArray, ...]:
    pixels = np.arange(-HALF, HALF + 1, dtype=np.float64)[None, :]
    edges = np.concatenate([pixels - 0.5, pixels[:, -1:] + 0.5], axis=1)
    one = np.array
    return c.stamp_model(
        edges, edges, one([x0]), one([y0]), one([sigma]), one([dx]), one([dy]), n_sub
    )


def test_the_profile_matches_brute_force_integration() -> None:
    for sigma, dx, dy in [(0.45, 0.0, 0.0), (0.45, 3.0, 1.5), (0.8, -4.0, 2.0)]:
        t = model_stamp(0.3, -0.2, sigma, dx, dy, n_sub=128)[0][0]
        expected = brute_force_stamp(0.3, -0.2, sigma, dx, dy)
        assert t.sum() == pytest.approx(1.0, abs=1e-6)
        # The brute force is itself accurate to about 3e-4 of the peak.
        assert np.max(np.abs(t - expected)) < 6e-4 * expected.max()


def test_the_trail_sum_converges_with_the_square_of_the_spacing() -> None:
    # The sum over sub-positions is a midpoint rule on a line with two ends, so its error falls
    # by 4 each time the spacing halves. `fit_stars` spaces the positions by about one width.
    reference = model_stamp(0.3, -0.2, 0.3, 5.0, 0.0, n_sub=512)[0][0]
    errors = [
        float(np.max(np.abs(model_stamp(0.3, -0.2, 0.3, 5.0, 0.0, n_sub=k)[0][0] - reference)))
        / float(reference.max())
        for k in (16, 32, 64)
    ]
    assert errors[0] < 5e-3
    assert errors[1] < errors[0] / 3.0
    assert errors[2] < errors[1] / 3.0


@pytest.mark.parametrize("trail", [(0.0, 0.0), (4.0, 2.0)])
def test_the_derivatives_match_finite_differences(trail: tuple[float, float]) -> None:
    x0, y0, sigma = 0.3, -0.2, 0.55
    n_sub = 24
    _, d_x0, d_y0, d_sigma = model_stamp(x0, y0, sigma, *trail, n_sub)
    step = 1e-4  # the profile is evaluated in single precision, so the step is not tiny

    def t(x: float, y: float, s: float) -> FloatArray:
        return np.asarray(model_stamp(x, y, s, *trail, n_sub)[0][0], dtype=np.float64)

    scale = float(np.max(np.abs(t(x0, y0, sigma))))
    for derivative, plus, minus in [
        (d_x0, t(x0 + step, y0, sigma), t(x0 - step, y0, sigma)),
        (d_y0, t(x0, y0 + step, sigma), t(x0, y0 - step, sigma)),
        (d_sigma, t(x0, y0, sigma + step), t(x0, y0, sigma - step)),
    ]:
        numeric = (plus - minus) / (2 * step)
        assert np.max(np.abs(numeric - derivative[0])) < 5e-3 * scale / 0.1


def render_frame(
    shape: tuple[int, int],
    x: FloatArray,
    y: FloatArray,
    flux: FloatArray,
    sigma: float,
    dx: FloatArray,
    dy: FloatArray,
) -> FloatArray:
    height, width = shape
    image = np.zeros(shape)
    for xs, ys, f, tdx, tdy in zip(x, y, flux, dx, dy, strict=True):
        ix, iy = round(float(xs)), round(float(ys))
        t = model_stamp(float(xs) - ix, float(ys) - iy, sigma, float(tdx), float(tdy), 32)[0][0]
        y0, y1 = max(iy - HALF, 0), min(iy + HALF + 1, height)
        x0, x1 = max(ix - HALF, 0), min(ix + HALF + 1, width)
        image[y0:y1, x0:x1] += (
            f * t[y0 - (iy - HALF) : y1 - (iy - HALF), x0 - (ix - HALF) : x1 - (ix - HALF)]
        )
    return image


def grid_stars(count: int, seed: int) -> tuple[FloatArray, FloatArray]:
    """Positions on a coarse grid, so the stars never overlap, with random sub-pixel phases."""
    rng = np.random.default_rng(seed)
    columns = int(np.ceil(np.sqrt(count)))
    gx, gy = np.meshgrid(np.arange(columns), np.arange(columns))
    x = 20.0 + 30.0 * gx.ravel()[:count] + rng.uniform(-0.5, 0.5, count)
    y = 20.0 + 30.0 * gy.ravel()[:count] + rng.uniform(-0.5, 0.5, count)
    return x, y


@pytest.mark.parametrize("sigma", [0.3, 0.45, 0.8])
def test_a_noise_free_star_is_recovered_without_bias_at_every_phase(sigma: float) -> None:
    x, y = grid_stars(100, seed=1)
    flux = np.full(100, 5000.0)
    zeros = np.zeros(100)
    image = render_frame((340, 340), x, y, flux, sigma, zeros, zeros)
    fit = c.fit_stars(
        image.astype(np.float32),
        x + 0.3,
        y - 0.25,
        flux * 0.5,
        np.full(100, 0.6),
        zeros,
        zeros,
        np.full(100, 1.0),
    )
    assert np.max(np.abs(fit.x - x)) < 2e-3
    assert np.max(np.abs(fit.y - y)) < 2e-3
    assert np.max(np.abs(fit.sigma - sigma)) < 0.01
    assert np.max(np.abs(fit.flux / flux - 1.0)) < 2e-3
    assert fit.converged.all()


def test_a_trailed_star_is_recovered_with_the_trail_model() -> None:
    x, y = grid_stars(81, seed=2)
    rng = np.random.default_rng(3)
    length = rng.uniform(1.0, 8.0, 81)
    angle = rng.uniform(-np.pi / 2, np.pi / 2, 81)
    dx, dy = length * np.cos(angle), length * np.sin(angle)
    flux = np.full(81, 8000.0)
    image = render_frame((300, 300), x, y, flux, 0.45, dx, dy)
    fit = c.fit_stars(
        image.astype(np.float32),
        x + 0.4,
        y + 0.2,
        flux * 0.6,
        np.full(81, 0.6),
        dx,
        dy,
        np.full(81, 1.0),
    )
    assert np.max(np.abs(fit.x - x)) < 3e-3
    assert np.max(np.abs(fit.y - y)) < 3e-3
    assert np.max(np.abs(fit.sigma - 0.45)) < 0.02  # the width excludes the trail


def test_the_plain_centroid_shifts_with_the_pixel_phase_and_the_fit_does_not() -> None:
    # A research notes row: a plain centroid of a star with FWHM 0.8 pixel (sigma 0.34) has a
    # pixel-phase bias of 0.065 pixel peak to peak. The fit has none.
    sigma = 0.34
    phases = np.linspace(-0.5, 0.5, 21)
    x = 20.0 + 30.0 * np.arange(21) + phases
    y = np.full(21, 20.0)
    flux = np.full(21, 20000.0)
    zeros = np.zeros(21)
    image = render_frame((50, 660), x, y, flux, sigma, zeros, zeros)
    plain = []
    for xs in x:
        ix = round(float(xs))
        window = image[18:23, ix - 2 : ix + 3]
        columns = np.arange(ix - 2, ix + 3)
        plain.append(float((window.sum(axis=0) * columns).sum() / window.sum()) - float(xs))
    fit = c.fit_stars(
        image.astype(np.float32), x, y, flux, np.full(21, 0.5), zeros, zeros, np.full(21, 1.0)
    )
    assert np.ptp(plain) > 0.04
    assert np.max(np.abs(fit.x - x)) < 2e-3


def test_the_reported_errors_match_the_scatter_in_a_noisy_frame() -> None:
    rng = np.random.default_rng(4)
    x, y = grid_stars(400, seed=5)
    flux = 10 ** rng.uniform(np.log10(500.0), np.log10(20000.0), 400)
    length = rng.uniform(0.0, 6.0, 400)
    angle = rng.uniform(-np.pi / 2, np.pi / 2, 400)
    dx, dy = length * np.cos(angle), length * np.sin(angle)
    clean = render_frame((620, 620), x, y, flux, 0.45, dx, dy)
    sky, read_noise, e_per_adu = 80.0, 3.0, 1.0
    noisy = (
        rng.poisson((clean + sky) * e_per_adu) / e_per_adu
        - sky
        + rng.normal(0, read_noise, clean.shape)
    ).astype(np.float32)
    noise = np.full(400, np.sqrt(sky / e_per_adu + read_noise**2))
    fit = c.fit_stars(
        noisy,
        x + rng.normal(0, 0.3, 400),
        y + rng.normal(0, 0.3, 400),
        flux * 0.7,
        np.full(400, 0.6),
        dx,
        dy,
        noise,
        e_per_adu=e_per_adu,
    )
    errors = np.concatenate([(fit.x - x) / fit.x_error, (fit.y - y) / fit.y_error])
    good = fit.converged
    assert good.mean() > 0.97
    # Tolerance: with 800 values the standard deviation of the pulls is 1 within a few percent,
    # and the heavy tails of faint stars keep the bound loose.
    assert 0.9 < np.std(errors) < 1.15
    assert abs(np.mean(errors)) < 0.1
    bright = flux > 5000.0
    assert np.std(fit.x[bright] - x[bright]) < 0.02  # a few hundredths of a pixel at high SNR
    assert np.median(fit.flux / flux) == pytest.approx(1.0, abs=0.01)
    assert 0.9 < np.mean(fit.chi2_reduced) < 1.15


def test_saturated_pixels_carry_no_weight() -> None:
    x, y = grid_stars(16, seed=6)
    flux = np.full(16, 60000.0)
    zeros = np.zeros(16)
    image = render_frame((140, 140), x, y, flux, 0.8, zeros, zeros)
    level = 0.4 * image.max()
    clipped = np.minimum(image, level).astype(np.float32)
    bad = image >= level
    assert bad.sum() > 16  # every star has saturated pixels
    args = (clipped, x + 0.2, y - 0.2, flux * 0.5, np.full(16, 0.7), zeros, zeros, np.full(16, 1.0))
    with_mask = c.fit_stars(*args, bad=bad)
    without_mask = c.fit_stars(*args)
    assert np.median(np.abs(with_mask.flux / flux - 1.0)) < 0.02
    assert np.median(without_mask.flux / flux) < 0.9  # the clipped core looks like less light
    assert np.max(np.abs(with_mask.x - x)) < 0.01
    assert np.all(with_mask.n_pixels < without_mask.n_pixels)


def test_a_star_at_the_frame_edge_fits_with_the_pixels_that_exist() -> None:
    zeros = np.zeros(4)
    x = np.array([0.4, 99.6, 50.0, 50.0])
    y = np.array([50.0, 50.0, 0.3, 99.7])
    flux = np.full(4, 8000.0)
    image = render_frame((100, 100), x, y, flux, 0.5, zeros, zeros)
    fit = c.fit_stars(
        image.astype(np.float32), x + 0.2, y - 0.2, flux, np.full(4, 0.6), zeros, zeros, np.ones(4)
    )
    assert np.max(np.abs(fit.x - x)) < 0.02
    assert np.max(np.abs(fit.y - y)) < 0.02
    assert np.all(fit.n_pixels < 15 * 15)


def test_an_empty_list_of_stars_gives_an_empty_fit() -> None:
    empty = np.array([])
    fit = c.fit_stars(
        np.zeros((10, 10), dtype=np.float32), empty, empty, empty, empty, empty, empty, empty
    )
    assert fit.x.shape == (0,)
    assert fit.converged.shape == (0,)
