"""The turbulence model: analytic constants, the spectrum it builds, and the image motion it makes.

The statistical tests draw many independent seeds. Each test computes the standard error of its
estimate from the spread of the per-seed means, states it next to the tolerance, and checks
that the error is small enough for the tolerance to mean something.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from seeingmon.drivers.sim.turbulence import (
    DEFAULT_LAYERS,
    GridSpec,
    Layer,
    PupilGrid,
    TurbulenceConfig,
    TurbulenceModel,
    g_tilt_coefficient,
    g_tilt_rms_arcsec,
    g_tilt_variance_rad2,
    outer_scale_ratio,
    r0_at_wavelength,
    r0_at_zenith_angle,
    seeing_fwhm_arcsec,
)

APERTURE_M = 0.050
LAMBDA_500NM = 500e-9

# The reductions of the one-axis variance by a finite outer scale (research notes).
OUTER_SCALE_RATIOS = {10.0: 0.740, 20.0: 0.793, 50.0: 0.848}


def reference_variance(r0_m: float, outer_scale_m: float = math.inf) -> float:
    """The one-axis G-tilt variance of the brief: 0.170 lambda^2 D^(-1/3) r0^(-5/3), in rad^2.

    A finite outer scale multiplies it by the ratio from the research notes.
    """
    variance = 0.170 * LAMBDA_500NM**2 * math.pow(APERTURE_M, -1 / 3) * math.pow(r0_m, -5 / 3)
    return variance * OUTER_SCALE_RATIOS.get(outer_scale_m, 1.0)


@pytest.fixture(scope="module")
def pupil() -> PupilGrid:
    return PupilGrid.circular(APERTURE_M, APERTURE_M / 46)


def single_layer(
    *,
    r0_m: float = 0.10,
    outer_scale_m: float = math.inf,
    screen_points: int = 256,
    zenith_angle_deg: float = 0.0,
    boiling: bool = True,
    seed: int = 1,
) -> TurbulenceConfig:
    """A config with one layer that blows at 10 m/s."""
    return TurbulenceConfig(
        r0_m=r0_m,
        layers=(Layer(1.0, 10.0, 30.0),),
        outer_scale_m=outer_scale_m,
        zenith_angle_deg=zenith_angle_deg,
        seed=seed,
        screen_points=screen_points,
        boiling=boiling,
    )


def tilt_ratio(
    base: TurbulenceConfig,
    seeds: range,
    pupil: PupilGrid,
    times: tuple[float, ...] = (0.0,),
) -> tuple[float, float]:
    """The simulated one-axis tilt variance over the reference, and its standard error.

    Every seed builds a model and reads both axes at each time. The standard error comes from
    the spread of the per-seed means, so it includes any correlation between the times.
    """
    reference = reference_variance(base.r0_m, base.outer_scale_m)
    time_array = np.asarray(times, dtype=np.float64)
    per_seed = np.empty(len(seeds))
    for index, seed in enumerate(seeds):
        model = TurbulenceModel(replace(base, seed=seed), APERTURE_M)
        tilt = model.tilt_series_rad(time_array, pupil)
        per_seed[index] = float(np.mean(tilt**2))
    mean = float(per_seed.mean())
    standard_error = float(per_seed.std(ddof=1) / math.sqrt(len(per_seed)))
    return mean / reference, standard_error / reference


# --- analytic results ---------------------------------------------------------------------


def test_kolmogorov_tilt_coefficient_matches_martin() -> None:
    # Martin 1987, Eq. 7: K = 0.170. The integral of the spectrum gives 0.1698.
    assert g_tilt_coefficient(APERTURE_M) == pytest.approx(0.170, rel=0.005)


def test_one_axis_motion_matches_the_research_notes() -> None:
    # "r0 of 5, 10, and 15 cm gives 0.85, 0.48, and 0.34 arcsec" (rms, one axis, D = 50 mm).
    for r0, expected in ((0.05, 0.85), (0.10, 0.48), (0.15, 0.34)):
        assert g_tilt_rms_arcsec(APERTURE_M, r0) == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize(("outer_scale", "ratio"), sorted(OUTER_SCALE_RATIOS.items()))
def test_outer_scale_ratio_matches_the_research_notes(outer_scale: float, ratio: float) -> None:
    assert outer_scale_ratio(APERTURE_M, outer_scale) == pytest.approx(ratio, abs=0.002)


def test_seeing_and_r0_scaling() -> None:
    # "Seeing is 1.011 arcsec at 500 nm" for r0 = 10 cm, and r0 (650 nm) is 0.137 m.
    assert seeing_fwhm_arcsec(0.10) == pytest.approx(1.011, abs=0.002)
    assert r0_at_wavelength(0.10, 650e-9) == pytest.approx(0.137, abs=0.001)
    assert r0_at_zenith_angle(0.10, 60.0) == pytest.approx(0.10 * 0.5 ** (3 / 5))
    assert seeing_fwhm_arcsec(0.10, outer_scale_m=20.0) < seeing_fwhm_arcsec(0.10)


def test_tilt_variance_is_achromatic() -> None:
    at_500 = g_tilt_variance_rad2(APERTURE_M, 0.10, wavelength_m=500e-9)
    at_650 = g_tilt_variance_rad2(APERTURE_M, r0_at_wavelength(0.10, 650e-9), wavelength_m=650e-9)
    assert at_650 == pytest.approx(at_500, rel=1e-9)


# --- the spectrum the model builds ----------------------------------------------------------


@pytest.mark.parametrize("outer_scale", [math.inf, 50.0, 20.0, 10.0])
@pytest.mark.parametrize("screen_points", [128, 256, 512])
def test_built_spectrum_reproduces_the_analytic_variance(
    outer_scale: float, screen_points: int
) -> None:
    """The spectral cells that the model uses integrate to the exact image-motion variance.

    This test has no sampling noise. A cell quadrature error of 0.4% at the smallest screen
    leaves most of the 5% budget for sampling and for the pupil grid.
    """
    config = single_layer(outer_scale_m=outer_scale, screen_points=screen_points)
    model = TurbulenceModel(config, APERTURE_M)
    analytic = g_tilt_variance_rad2(APERTURE_M, config.r0_m, outer_scale)
    assert model.expected_tilt_variance_rad2() == pytest.approx(analytic, rel=0.004)


def test_layers_share_the_variance() -> None:
    one = TurbulenceModel(single_layer(outer_scale_m=20.0), APERTURE_M)
    three = TurbulenceModel(TurbulenceConfig(layers=DEFAULT_LAYERS, outer_scale_m=20.0), APERTURE_M)
    assert three.expected_tilt_variance_rad2() == pytest.approx(
        one.expected_tilt_variance_rad2(), rel=1e-9
    )


def test_variance_follows_r0_and_zenith_angle() -> None:
    base = TurbulenceModel(single_layer(r0_m=0.10), APERTURE_M).expected_tilt_variance_rad2()
    weaker = TurbulenceModel(single_layer(r0_m=0.05), APERTURE_M).expected_tilt_variance_rad2()
    assert weaker / base == pytest.approx(2 ** (5 / 3), rel=1e-9)
    slanted = TurbulenceModel(single_layer(r0_m=0.10, zenith_angle_deg=60.0), APERTURE_M)
    assert slanted.expected_tilt_variance_rad2() / base == pytest.approx(1 / 0.5, rel=1e-9)
    assert slanted.r0_observed_m() == pytest.approx(0.10 * 0.5 ** (3 / 5))


# --- determinism and structure --------------------------------------------------------------


def test_same_seed_gives_the_same_phase(pupil: PupilGrid) -> None:
    config = TurbulenceConfig(seed=7)
    first = TurbulenceModel(config, APERTURE_M).phase(1.234, pupil.spec)
    second = TurbulenceModel(config, APERTURE_M).phase(1.234, pupil.spec)
    other = TurbulenceModel(replace(config, seed=8), APERTURE_M).phase(1.234, pupil.spec)
    assert np.array_equal(first, second)
    assert not np.allclose(first, other)


def test_evaluation_order_does_not_matter(pupil: PupilGrid) -> None:
    config = TurbulenceConfig(seed=3)
    forward = TurbulenceModel(config, APERTURE_M)
    backward = TurbulenceModel(config, APERTURE_M)
    times = [0.0, 0.4, 1.7, 9.3]
    in_order = [forward.phase(t, pupil.spec) for t in times]
    reversed_order = [backward.phase(t, pupil.spec) for t in reversed(times)][::-1]
    for a, b in zip(in_order, reversed_order, strict=True):
        assert np.array_equal(a, b)


def test_a_batch_equals_single_evaluations(pupil: PupilGrid) -> None:
    model = TurbulenceModel(TurbulenceConfig(seed=5), APERTURE_M)
    times = np.array([0.0, 0.011, 0.5, 0.511, 3.0])
    batch = model.phase_series(times, pupil.spec)
    for index, time in enumerate(times):
        assert np.allclose(batch[index], model.phase(float(time), pupil.spec), atol=1e-9)


def test_the_phase_moves_with_the_wind() -> None:
    """Frozen flow: the phase at `x` and `t + dt` is the phase at `x - v dt` and `t`.

    The shift stays under a centimeter, so the low-frequency sinusoids, which the model expands
    about the pupil centre, stay accurate.
    """
    layer = Layer(1.0, 8.0, 35.0)
    config = TurbulenceConfig(layers=(layer,), boiling=False, seed=2, screen_points=512)
    model = TurbulenceModel(config, APERTURE_M)
    vx, vy = layer.velocity_m_s
    spec = GridSpec(-0.02, -0.02, 0.001, 40, 40)
    dt = 0.00125
    later = model.phase(1.0 + dt, spec)
    shifted = GridSpec(spec.x0 - vx * dt, spec.y0 - vy * dt, spec.dx, spec.nx, spec.ny)
    earlier = model.phase(1.0, shifted)
    assert np.allclose(later, earlier, atol=2e-3)


def test_the_screen_refreshes_so_that_a_long_run_does_not_repeat(pupil: PupilGrid) -> None:
    """A periodic screen repeats after one crossing. The refreshed screen does not.

    The test compares the curvature of the phase, which the low-frequency sinusoids barely
    touch, one crossing apart and several crossings apart.
    """
    layer = Layer(1.0, 8.0, 0.0)
    period = 128 / 16 * APERTURE_M / layer.wind_speed_m_s  # the screen length over the wind speed
    correlations = {}
    for boiling in (False, True):
        config = TurbulenceConfig(
            layers=(layer,), boiling=boiling, seed=4, screen_points=128, outer_scale_m=20.0
        )
        model = TurbulenceModel(config, APERTURE_M)
        t = 0.3 * period
        a = np.diff(model.phase(t, pupil.spec), n=2, axis=1)
        b = np.diff(model.phase(t + 3 * period, pupil.spec), n=2, axis=1)
        correlations[boiling] = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    assert correlations[False] > 0.95
    assert abs(correlations[True]) < 0.3


def test_scheduled_r0_scales_the_phase(pupil: PupilGrid) -> None:
    base = TurbulenceConfig(seed=11)
    scheduled = replace(base, r0_schedule=((0.0, 0.10), (100.0, 0.05)))
    plain = TurbulenceModel(base, APERTURE_M)
    varying = TurbulenceModel(scheduled, APERTURE_M)
    assert varying.r0_zenith_m(50.0) == pytest.approx(0.075)
    for t in (0.0, 25.0, 100.0, 150.0):
        gain = (0.10 / varying.r0_zenith_m(t)) ** (5 / 6)
        assert np.allclose(varying.phase(t, pupil.spec), gain * plain.phase(t, pupil.spec))


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="r0_m"):
        TurbulenceConfig(r0_m=-1.0)
    with pytest.raises(ValueError, match="layer"):
        TurbulenceConfig(layers=())
    with pytest.raises(ValueError, match="screen_points"):
        TurbulenceConfig(screen_points=33)
    with pytest.raises(ValueError, match="r0_schedule"):
        TurbulenceConfig(r0_schedule=((5.0, 0.1), (1.0, 0.1)))
    with pytest.raises(ValueError, match="cn2"):
        Layer(cn2_fraction=0.0)


def test_pupil_tilt_weights_are_exact_for_a_linear_phase(pupil: PupilGrid) -> None:
    xs, ys = pupil.coordinates()
    slope = 120.0  # rad/m
    tilt_x, tilt_y = pupil.tilt_rad(slope * xs[None, :] + 0.0 * ys[:, None])
    assert tilt_x == pytest.approx(slope * LAMBDA_500NM / (2 * math.pi), rel=1e-9)
    assert tilt_y == pytest.approx(0.0, abs=1e-18)
    assert pupil.coverage.sum() * pupil.spec.dx**2 == pytest.approx(
        math.pi * (APERTURE_M / 2) ** 2, rel=1e-3
    )


# --- sampling tests: the done criteria ----------------------------------------------------


@pytest.mark.parametrize("r0", [0.05, 0.10, 0.15])
def test_kolmogorov_image_motion_matches_theory(r0: float, pupil: PupilGrid) -> None:
    """One-axis G-tilt variance within 5% of 0.170 lambda^2 D^(-1/3) r0^(-5/3), L0 = infinity.

    3,000 seeds with both axes give 6,000 samples. The standard error is about 1.8%, so the
    5% tolerance is about 2.8 sigma. A small 128-point screen (0.4 m) keeps the test fast and
    puts most of the variance into the sinusoid part. The slow test below uses the default screen.
    """
    base = single_layer(r0_m=r0, screen_points=128)
    ratio, standard_error = tilt_ratio(base, range(100, 3100), pupil)
    assert standard_error < 0.025
    assert ratio == pytest.approx(1.0, abs=0.05), (ratio, standard_error)


@pytest.mark.parametrize("outer_scale", sorted(OUTER_SCALE_RATIOS))
def test_outer_scale_reduction_matches_the_research_notes(
    outer_scale: float, pupil: PupilGrid
) -> None:
    """A finite outer scale lowers the variance by 0.740, 0.793, and 0.848 (within 5%).

    With 2,000 seeds and both axes, the standard error is about 2.2%, so 5% is about 2.3 sigma.
    """
    base = single_layer(outer_scale_m=outer_scale, screen_points=128)
    ratio, standard_error = tilt_ratio(base, range(300, 2300), pupil)
    assert standard_error < 0.03
    assert ratio == pytest.approx(1.0, abs=0.05), (ratio, standard_error)


@pytest.mark.parametrize(("xi", "expected"), [(1.0, 0.93), (2.0, 0.83), (4.0, 0.72)])
def test_exposure_averaging_matches_martin(xi: float, expected: float, pupil: PupilGrid) -> None:
    """Averaging over an exposure lowers the variance by 0.93, 0.83, and 0.72 for vT/D of 1, 2, 4.

    These are the single-layer numbers of Martin (1987) for Kolmogorov turbulence and a wind at
    45 degrees to the axes, which is the mean of the two axes. The test pairs the averaged and
    the instantaneous tilt of the same seed, so the standard error of the ratio is small. The
    tolerance is three standard errors plus 0.006 for the two-digit rounding of the notes.
    """
    wind = 10.0
    exposure = xi * APERTURE_M / wind
    centre = 0.123
    instantaneous: list[float] = []
    averaged: list[float] = []
    for seed in range(800):
        config = TurbulenceConfig(
            layers=(Layer(1.0, wind, 45.0),),
            outer_scale_m=math.inf,
            screen_points=128,
            boiling=False,
            seed=seed,
        )
        model = TurbulenceModel(config, APERTURE_M)
        instantaneous.extend(model.tilt_rad(centre, pupil))
        n_sub = model.suggest_substeps(exposure)
        averaged.extend(model.exposure_tilt_rad(centre - exposure / 2, exposure, pupil, n_sub))
    inst_sq = np.asarray(instantaneous) ** 2
    avg_sq = np.asarray(averaged) ** 2
    ratio = float(avg_sq.mean() / inst_sq.mean())
    # Delta method for the ratio of two correlated means.
    residual = avg_sq - ratio * inst_sq
    standard_error = float(residual.std(ddof=1) / math.sqrt(len(inst_sq)) / inst_sq.mean())
    assert standard_error < 0.02
    assert abs(ratio - expected) < 3 * standard_error + 0.006, (ratio, standard_error)


# --- the slow versions, with the default screen -------------------------------------------


@pytest.mark.slow
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("r0", [0.05, 0.10, 0.15])
def test_kolmogorov_image_motion_matches_theory_default_screen(r0: float, pupil: PupilGrid) -> None:
    """The same criterion with the default 256-point screen and two screens per seed.

    4,000 seeds x 2 times x 2 axes. The two times fall in two crossings of the screen, which are
    the real and the imaginary part of one FFT. The sinusoid part (about 45% of the variance) has
    one realization per seed, so the standard error is about 1.5%.
    """
    base = single_layer(r0_m=r0)
    ratio, standard_error = tilt_ratio(base, range(1000, 5000), pupil, (0.03, 0.11))
    assert standard_error < 0.02
    assert ratio == pytest.approx(1.0, abs=0.05), (ratio, standard_error)


@pytest.mark.slow
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("outer_scale", sorted(OUTER_SCALE_RATIOS))
def test_outer_scale_reduction_default_screen(outer_scale: float, pupil: PupilGrid) -> None:
    base = single_layer(outer_scale_m=outer_scale)
    ratio, standard_error = tilt_ratio(base, range(5000, 9000), pupil, (0.03, 0.11))
    assert standard_error < 0.02
    assert ratio == pytest.approx(1.0, abs=0.05), (ratio, standard_error)
