"""The seeing estimator against simulated and synthetic truth.

**Sampling error.** The variance of one axis from `N` independent samples has a relative error of
`sqrt(2 / N)`, and the seeing carries 0.6 times that (the research notes). The tilt of a 50 mm
aperture at 25 frames per second is nearly independent from frame to frame, so a 40 s window of
1,000 samples gives 4.5% in the variance and 2.7% in `r0` per axis, and 1.9% for the two axes
together. The tests below average two windows of each simulated series, which gives about 1.4% in
`r0`, so the 10% tolerance of the done criterion is seven standard errors.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from seeingmon.fastpath import models
from seeingmon.fastpath.estimator import (
    EstimatorSettings,
    MotionSeries,
    SeeingEstimate,
    estimate_seeing,
    structure_variance_px2,
)
from tests.fastpath.helpers import r0_scale, simulated_tilt_arcsec, synthetic_tilt_arcsec

FloatArray = npt.NDArray[np.float64]

PLATE = 1.910  # arcsec per pixel in bin1
RATE_HZ = 25.0
PERIOD_S = 1.0 / RATE_HZ
WINDOW = 1000  # samples: 40 s at 25 frames per second


def settings(**kwargs: Any) -> EstimatorSettings:
    values: dict[str, Any] = {
        "aperture_m": 0.05,
        "plate_scale_arcsec_per_px": PLATE,
        "exposure_s": 0.002,
        "centroid_gain_variance_ratio": 1.0,
    }
    values.update(kwargs)
    return EstimatorSettings(**values)


def sim_tilt(exposure_s: float, substeps: int) -> FloatArray:
    """80 s of the exposure-averaged tilt for r0 = 10 cm, from the wave-optics turbulence model."""
    return simulated_tilt_arcsec(
        exposure_s=exposure_s, substeps=substeps, screen_points=128, duration_s=80.0
    )


def make_series(
    tilt_arcsec: FloatArray,
    *,
    noise_px: float = 0.0,
    drift_arcsec: tuple[float, float] = (0.0, 0.0),
    seed: int = 0,
    known_noise: bool = True,
) -> MotionSeries:
    """The centroids of one window: the tilt, a linear drift over the window, and white noise."""
    rng = np.random.default_rng(seed)
    n = len(tilt_arcsec)
    ramp = np.linspace(-0.5, 0.5, n)
    columns = []
    for axis in range(2):
        arcsec = tilt_arcsec[:, axis] + drift_arcsec[axis] * ramp
        columns.append(4000.0 + arcsec / PLATE + rng.normal(0.0, noise_px, n))
    variance = np.full(n, noise_px**2 if known_noise else np.nan)
    return MotionSeries(PERIOD_S, columns[0], columns[1], variance, variance.copy())


def windows_of(tilt: FloatArray, size: int = WINDOW) -> list[FloatArray]:
    return [tilt[i : i + size] for i in range(0, len(tilt) - size + 1, size)]


def mean_r0_cm(estimates: list[SeeingEstimate]) -> float:
    values = [e.r0_m for e in estimates if e.r0_m is not None]
    assert len(values) == len(estimates)
    return 100.0 * float(np.mean(values))


def mean_structure_r0_cm(estimates: list[SeeingEstimate]) -> float:
    values = [e.r0_structure_m for e in estimates if e.r0_structure_m is not None]
    assert len(values) == len(estimates)
    return 100.0 * float(np.mean(values))


@pytest.fixture(scope="module")
def tilt_2ms() -> FloatArray:
    return sim_tilt(0.002, 3)


@pytest.fixture(scope="module")
def tilt_10ms() -> FloatArray:
    return sim_tilt(0.010, 6)


class TestRecoversR0FromSimulatedTurbulence:
    """The done criterion: r0 of 5, 10, and 15 cm within 10%.

    The tilt of the simulator scales as `r0^(-5/6)`, so one series gives all three. The tests add
    the centroid noise of Polaris (0.017 px) and a drift of 10 arcsec per window.
    """

    @pytest.mark.parametrize("r0_cm", [5.0, 10.0, 15.0])
    def test_recovers_r0_within_ten_percent(self, tilt_2ms: FloatArray, r0_cm: float) -> None:
        tilt = tilt_2ms * r0_scale(r0_cm / 100.0)
        estimates = [
            estimate_seeing(
                make_series(w, noise_px=0.017, drift_arcsec=(10.0, -6.0), seed=i), settings()
            )
            for i, w in enumerate(windows_of(tilt))
        ]
        recovered = mean_r0_cm(estimates)
        assert recovered == pytest.approx(r0_cm, rel=0.10)
        # In practice the error is a few percent: two windows of 1,000 samples give 1.4% in r0.
        assert recovered == pytest.approx(r0_cm, rel=0.06)

    @pytest.mark.parametrize("r0_cm", [5.0, 10.0, 15.0])
    def test_the_seeing_is_the_kolmogorov_value_at_500_nm(
        self, tilt_2ms: FloatArray, r0_cm: float
    ) -> None:
        tilt = tilt_2ms * r0_scale(r0_cm / 100.0)
        estimates = [estimate_seeing(make_series(w), settings()) for w in windows_of(tilt)]
        fwhm = float(np.mean([e.fwhm_arcsec for e in estimates if e.fwhm_arcsec is not None]))
        truth = 0.98 * 500e-9 / (r0_cm / 100.0) * models.ARCSEC_PER_RAD
        assert fwhm == pytest.approx(truth, rel=0.06)

    def test_the_structure_function_agrees_within_a_modest_margin(
        self, tilt_2ms: FloatArray
    ) -> None:
        """The cross-check on the same data.

        It leans on the correlation of the tilt at lags of 40 to 120 ms, which the small screens
        of this fast simulation reproduce only roughly, so the margin is 12%. The full-size screens
        of the `sim` driver agree to 3% (see the end-to-end test).
        """
        estimates = [estimate_seeing(make_series(w), settings()) for w in windows_of(tilt_2ms)]
        assert mean_structure_r0_cm(estimates) == pytest.approx(10.0, rel=0.12)

    def test_the_factors_are_stored_and_multiply_to_the_kolmogorov_variance(
        self, tilt_2ms: FloatArray
    ) -> None:
        estimate = estimate_seeing(make_series(windows_of(tilt_2ms)[0]), settings())
        factors = estimate.factors
        assert factors is not None
        assert factors.outer_scale == pytest.approx(1.0 / 0.7931, rel=1e-3)
        assert factors.exposure == pytest.approx(1.0 / 0.9804, rel=3e-3)
        assert factors.detrend == pytest.approx(1.0 / (1.0 - 0.0041), rel=0.1 / 100)  # 40 s window
        assert factors.centroid_gain == 1.0
        assert factors.zenith_r0 == 1.0
        assert estimate.kolmogorov_variance_arcsec2 is not None
        assert estimate.rms_x_arcsec is not None
        assert estimate.rms_y_arcsec is not None
        motion = 0.5 * (estimate.rms_x_arcsec**2 + estimate.rms_y_arcsec**2)
        chain = factors.detrend * factors.centroid_gain * factors.outer_scale * factors.exposure
        assert estimate.kolmogorov_variance_arcsec2 == pytest.approx(motion * chain, rel=1e-9)


class TestCorrections:
    def test_the_exposure_correction_matters_at_ten_milliseconds(
        self, tilt_10ms: FloatArray
    ) -> None:
        """At 10 ms and 10 m/s the exposure removes 21% of the variance.

        The estimator with the right wind recovers r0. One that ignores the exposure reads r0 too
        high by about 15%.
        """
        estimates = [
            estimate_seeing(make_series(w), settings(exposure_s=0.010))
            for w in windows_of(tilt_10ms)
        ]
        assert mean_r0_cm(estimates) == pytest.approx(10.0, rel=0.10)
        ignored = [
            estimate_seeing(make_series(w), settings(exposure_s=0.0)) for w in windows_of(tilt_10ms)
        ]
        assert mean_r0_cm(ignored) > 11.0

    def test_the_assumed_wind_speed_is_a_systematic_at_ten_milliseconds(
        self, tilt_10ms: FloatArray
    ) -> None:
        """A faster assumed wind needs a larger correction, so it gives a smaller r0."""
        window = windows_of(tilt_10ms)[0]
        r0 = {
            wind: estimate_seeing(
                make_series(window), settings(exposure_s=0.010, wind_ms=wind)
            ).r0_m
            for wind in (5.0, 10.0, 20.0)
        }
        assert None not in r0.values()
        assert r0[5.0] > r0[10.0] > r0[20.0]  # type: ignore[operator]

    def test_the_outer_scale_correction_follows_the_assumed_scale(
        self, tilt_2ms: FloatArray
    ) -> None:
        """The same data with an assumed L0 of 20 m and of 40 m: r0 changes by the ratio's power."""
        window = windows_of(tilt_2ms)[0]
        a = estimate_seeing(make_series(window), settings(outer_scale_m=20.0))
        b = estimate_seeing(make_series(window), settings(outer_scale_m=40.0))
        assert a.r0_m is not None
        assert b.r0_m is not None
        ratio = models.outer_scale_ratio(0.05, 40.0) / models.outer_scale_ratio(0.05, 20.0)
        # A larger assumed L0 means less suppression, so the same variance means weaker turbulence
        # and a larger r0. The exposure and detrend factors move a little with L0, so the match is
        # to 2%.
        assert b.r0_m > a.r0_m
        assert b.r0_m / a.r0_m == pytest.approx(ratio**0.6, rel=0.02)

    def test_zenith_angle_converts_r0_to_the_zenith(self, tilt_2ms: FloatArray) -> None:
        window = windows_of(tilt_2ms)[0]
        sight_line = estimate_seeing(make_series(window), settings())
        zenith = estimate_seeing(make_series(window), settings(zenith_angle_deg=60.0))
        assert sight_line.r0_m is not None
        assert zenith.r0_m is not None
        assert sight_line.fwhm_arcsec is not None
        assert zenith.fwhm_arcsec is not None
        assert zenith.r0_m / sight_line.r0_m == pytest.approx(1.0 / 0.5**0.6, rel=1e-9)
        assert zenith.fwhm_arcsec == pytest.approx(sight_line.fwhm_arcsec * 0.5**0.6, rel=1e-9)
        assert zenith.factors is not None
        assert zenith.factors.zenith_r0 == pytest.approx(0.5**0.6)

    def test_the_centroid_gain_ratio_divides_the_variance(self, tilt_2ms: FloatArray) -> None:
        window = windows_of(tilt_2ms)[0]
        plain = estimate_seeing(make_series(window), settings())
        gained = estimate_seeing(make_series(window), settings(centroid_gain_variance_ratio=1.02))
        assert plain.kolmogorov_variance_arcsec2
        assert gained.kolmogorov_variance_arcsec2
        assert gained.kolmogorov_variance_arcsec2 == pytest.approx(
            plain.kolmogorov_variance_arcsec2 / 1.02, rel=1e-9
        )


@pytest.fixture(scope="module")
def synthetic() -> FloatArray:
    """Eight windows of independent samples: 8,000 frames at 25 frames per second."""
    return synthetic_tilt_arcsec(0.10, rate_hz=RATE_HZ, samples=8000, seed=11)


class TestNoiseDriftAndOutliers:
    """The mechanics of the estimator, on synthetic series with exactly the model's statistics."""

    def test_returns_the_true_r0_from_a_series_with_the_model_statistics(
        self, synthetic: FloatArray
    ) -> None:
        """Eight windows of 40 s: the standard error of the mean is 1%, and the bound is 5%."""
        estimates = [estimate_seeing(make_series(w), settings()) for w in windows_of(synthetic)]
        assert mean_r0_cm(estimates) == pytest.approx(10.0, rel=0.05)

    def test_subtracting_the_modeled_noise_removes_its_bias(self, synthetic: FloatArray) -> None:
        """A centroid noise of 0.1 px adds 0.036 arcsec^2, which is 20% of the motion.

        The estimator subtracts it, and r0 stays within 6%. Without the subtraction r0 reads
        about 10% too low.
        """
        windows = windows_of(synthetic)
        with_noise = [
            estimate_seeing(make_series(w, noise_px=0.1, seed=i), settings())
            for i, w in enumerate(windows)
        ]
        unaware = [
            estimate_seeing(make_series(w, noise_px=0.1, seed=i, known_noise=False), settings())
            for i, w in enumerate(windows)
        ]
        assert mean_r0_cm(with_noise) == pytest.approx(10.0, rel=0.06)
        assert mean_r0_cm(unaware) < 0.93 * mean_r0_cm(with_noise)
        assert with_noise[0].centroid_noise_px == pytest.approx(0.1)

    def test_a_linear_drift_biases_the_plain_variance_and_not_the_estimators(
        self, synthetic: FloatArray
    ) -> None:
        """Polaris drifts about 10 arcsec in a minute.

        The plain variance of a window is off by a factor of 40, the quadratic detrend recovers
        r0, and the structure function does too.
        """
        drift = (10.0, -10.0)
        series = [make_series(w, drift_arcsec=drift) for w in windows_of(synthetic)]
        plain = float(np.mean([np.var(s.x_px) * PLATE**2 for s in series]))
        assert plain > 5.0 * float(np.var(synthetic[:, 0]))  # the drift adds 8 arcsec^2 to 0.18
        estimates = [estimate_seeing(s, settings()) for s in series]
        assert mean_r0_cm(estimates) == pytest.approx(10.0, rel=0.06)
        assert mean_structure_r0_cm(estimates) == pytest.approx(10.0, rel=0.06)

    def test_a_slow_vibration_biases_the_variance_but_barely_the_structure_function(
        self, synthetic: FloatArray
    ) -> None:
        """A 0.5 Hz sway of 0.5 arcsec amplitude adds 0.125 arcsec^2, which is 70% of the motion.

        The detrended variance reads it, so r0 falls by a quarter. The structure function at lags
        of 40 to 120 ms sees 4% of it, so its r0 changes by under 3%. The disagreement of the two
        estimators is the sign that something slow shakes the star.
        """
        t = np.arange(len(synthetic)) / RATE_HZ
        shaken = synthetic + (0.5 * np.sin(2.0 * math.pi * 0.5 * t))[:, None]
        calm = [estimate_seeing(make_series(w), settings()) for w in windows_of(synthetic)]
        windy = [estimate_seeing(make_series(w), settings()) for w in windows_of(shaken)]
        assert mean_r0_cm(windy) < 0.8 * mean_r0_cm(calm)
        assert mean_structure_r0_cm(windy) == pytest.approx(mean_structure_r0_cm(calm), rel=0.03)

    def test_outliers_are_dropped(self, synthetic: FloatArray) -> None:
        """Ten centroid jumps of 3 px (a cosmic ray in the aperture) leave r0 within 6%."""
        rng = np.random.default_rng(2)
        kept, naive = [], []
        for i, w in enumerate(windows_of(synthetic)):
            series = make_series(w, seed=i)
            x = series.x_px.copy()
            x[rng.choice(len(x), 10, replace=False)] += 3.0
            spoiled = MotionSeries(
                series.period_s, x, series.y_px, series.noise_var_x_px2, series.noise_var_y_px2
            )
            kept.append(estimate_seeing(spoiled, settings()))
            naive.append(estimate_seeing(spoiled, settings(outlier_sigma=0.0)))
        assert mean_r0_cm(kept) == pytest.approx(10.0, rel=0.06)
        assert mean_r0_cm(naive) < 9.0
        assert all(e.x is not None and e.x.outliers >= 8 for e in kept)

    def test_gaps_do_not_bias_r0(self, synthetic: FloatArray) -> None:
        """Seven percent of the frames missing, and six runs of five, leave r0 within 6%."""
        rng = np.random.default_rng(3)
        estimates = []
        for i, w in enumerate(windows_of(synthetic)):
            series = make_series(w, seed=i)
            x, y = series.x_px.copy(), series.y_px.copy()
            missing = rng.random(len(x)) < 0.07
            for start in rng.choice(len(x) - 5, 6, replace=False):
                missing[start : start + 5] = True
            x[missing] = np.nan
            y[missing] = np.nan
            gapped = MotionSeries(
                series.period_s, x, y, series.noise_var_x_px2, series.noise_var_y_px2
            )
            estimates.append(estimate_seeing(gapped, settings()))
        assert mean_r0_cm(estimates) == pytest.approx(10.0, rel=0.06)

    def test_a_window_with_too_few_frames_reports_why(self, synthetic: FloatArray) -> None:
        estimate = estimate_seeing(make_series(synthetic[:50]), settings())
        assert estimate.r0_m is None
        assert estimate.fwhm_arcsec is None
        assert estimate.quality["r0_cm"] == "too few usable frames"

    def test_a_still_star_reports_motion_below_the_noise(self) -> None:
        n = 1000
        steady = MotionSeries(
            PERIOD_S, np.full(n, 4000.0), np.full(n, 4000.0), np.full(n, 0.0004), np.full(n, 0.0004)
        )
        estimate = estimate_seeing(steady, settings())
        assert estimate.r0_m is None
        assert "below the centroid noise" in estimate.quality["seeing_fwhm_arcsec"]

    def test_without_a_zenith_angle_the_value_is_for_the_line_of_sight(
        self, synthetic: FloatArray
    ) -> None:
        window = windows_of(synthetic)[0]
        assert "line of sight" in estimate_seeing(make_series(window), settings()).quality["r0_cm"]
        zenith = estimate_seeing(make_series(window), settings(zenith_angle_deg=30.0))
        assert "r0_cm" not in zenith.quality


class TestStructureFunction:
    def test_the_noise_of_both_samples_is_subtracted(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(0.0, 0.05, 4000)  # white noise of 0.05 px
        lags = np.array([1, 2, 3], dtype=np.intp)
        kappa = np.zeros(3)
        assert structure_variance_px2(x, 0.0, lags, kappa) == pytest.approx(0.05**2, rel=0.1)
        assert structure_variance_px2(x, 0.05**2, lags, kappa) == pytest.approx(0.0, abs=3e-4)

    def test_pairs_with_a_missing_frame_are_skipped(self) -> None:
        x = np.arange(100, dtype=np.float64)
        x[10:20] = np.nan
        lags = np.array([5], dtype=np.intp)
        value = structure_variance_px2(x, 0.0, lags, np.array([0.0]))
        assert value == pytest.approx(25.0 / 2.0)  # every valid pair differs by exactly 5

    def test_too_few_pairs_give_none(self) -> None:
        lags = np.array([5], dtype=np.intp)
        assert structure_variance_px2(np.arange(10.0), 0.0, lags, np.array([0.0])) is None
