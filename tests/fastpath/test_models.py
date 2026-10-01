"""The theory of the estimator against the numbers of the research notes and the simulator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from seeingmon.fastpath import models
from tests.fastpath.helpers import simulated_tilt_arcsec

D = 0.05  # the aperture of the GS-250, in meters


class TestSeeingFormulas:
    @pytest.mark.parametrize(("r0_cm", "rms_arcsec"), [(5.0, 0.850), (10.0, 0.477), (15.0, 0.340)])
    def test_image_motion_of_a_50_mm_aperture(self, r0_cm: float, rms_arcsec: float) -> None:
        """The research notes give 0.85, 0.48, and 0.34 arcsec of one-axis motion (G-tilt)."""
        variance = models.tilt_variance_rad2(r0_cm / 100.0, D)
        assert math.sqrt(variance) * models.ARCSEC_PER_RAD == pytest.approx(rms_arcsec, abs=0.003)

    def test_seeing_of_a_10_cm_r0_is_one_arcsecond(self) -> None:
        """The notes give 1.011 arcsec at 500 nm for an r0 of 10 cm."""
        assert models.seeing_fwhm_arcsec(0.10) == pytest.approx(1.011, abs=0.001)

    def test_r0_inverts_the_variance(self) -> None:
        for r0 in (0.03, 0.10, 0.25):
            variance = models.tilt_variance_rad2(r0, D)
            assert models.r0_from_tilt_variance(variance, D) == pytest.approx(r0, rel=1e-12)
        with pytest.raises(ValueError, match="positive"):
            models.r0_from_tilt_variance(0.0, D)

    def test_the_zenith_factor(self) -> None:
        assert models.zenith_r0_factor(0.0) == 1.0
        assert models.zenith_r0_factor(60.0) == pytest.approx(0.5**0.6)
        with pytest.raises(ValueError, match="zenith angle"):
            models.zenith_r0_factor(89.5)


class TestOuterScale:
    @pytest.mark.parametrize(
        ("outer_scale", "ratio"), [(10.0, 0.740), (20.0, 0.793), (50.0, 0.848)]
    )
    def test_the_exact_ratios_of_the_research_notes(self, outer_scale: float, ratio: float) -> None:
        """The notes give 0.740, 0.793, and 0.848 for D = 50 mm (three digits)."""
        assert models.outer_scale_ratio(D, outer_scale) == pytest.approx(ratio, abs=0.001)

    def test_the_ratio_depends_on_d_over_l0_and_grows_with_l0(self) -> None:
        assert models.outer_scale_ratio(0.10, 40.0) == pytest.approx(
            models.outer_scale_ratio(0.05, 20.0), rel=1e-9
        )
        ratios = [models.outer_scale_ratio(D, l0) for l0 in (5.0, 10.0, 20.0, 100.0, 1000.0)]
        assert ratios == sorted(ratios)
        assert models.outer_scale_ratio(D, math.inf) == 1.0
        assert models.outer_scale_ratio(D, 1e6) == pytest.approx(1.0, abs=0.01)

    def test_matches_the_series_of_the_aristidi_formula_for_a_small_d_over_l0(self) -> None:
        """Aristidi et al. 2019: 1 - 1.525 (D / L0)^(1/3), accurate to 0.002 for these values."""
        for outer_scale in (10.0, 20.0, 50.0):
            series = 1.0 - 1.525 * (D / outer_scale) ** (1.0 / 3.0)
            assert models.outer_scale_ratio(D, outer_scale) == pytest.approx(series, abs=0.002)

    def test_rejects_a_non_positive_scale(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            models.outer_scale_ratio(D, 0.0)


class TestExposureAveraging:
    @pytest.mark.parametrize(("xi", "ratio"), [(1.0, 0.93), (2.0, 0.83), (4.0, 0.72)])
    def test_the_kolmogorov_ratios_of_the_research_notes(self, xi: float, ratio: float) -> None:
        """The notes give 0.93, 0.83, and 0.72 for v T / D of 1, 2, and 4 (a single layer)."""
        spectrum = models.tilt_spectrum(D, math.inf, 10.0)
        exposure = xi * D / 10.0
        assert spectrum.exposure_variance_ratio(exposure) == pytest.approx(ratio, abs=0.01)

    def test_a_finite_outer_scale_lowers_the_ratio_a_little(self) -> None:
        kolmogorov = models.tilt_spectrum(D, math.inf, 10.0).exposure_variance_ratio(0.01)
        von_karman = models.tilt_spectrum(D, 20.0, 10.0).exposure_variance_ratio(0.01)
        assert von_karman < kolmogorov
        assert von_karman == pytest.approx(0.79, abs=0.01)

    def test_two_milliseconds_costs_two_percent_of_the_variance_at_ten_meters_per_second(
        self,
    ) -> None:
        spectrum = models.tilt_spectrum(D, 20.0, 10.0)
        assert spectrum.exposure_variance_ratio(0.002) == pytest.approx(0.980, abs=0.003)
        # The wind speed matters little at 2 ms: 0.5% at 5 m/s and 6% at 20 m/s.
        slow = models.tilt_spectrum(D, 20.0, 5.0).exposure_variance_ratio(0.002)
        fast = models.tilt_spectrum(D, 20.0, 20.0).exposure_variance_ratio(0.002)
        assert slow == pytest.approx(0.995, abs=0.003)
        assert fast == pytest.approx(0.936, abs=0.005)

    def test_no_exposure_keeps_everything(self) -> None:
        assert models.tilt_spectrum(D, 20.0, 10.0).exposure_variance_ratio(0.0) == 1.0

    def test_agrees_with_the_turbulence_model_of_the_simulator(self) -> None:
        """The same turbulence realization at 2 ms and at 10 ms exposures.

        The variance ratio of the two series is 0.805 in the simulator and 0.807 in the model. The
        two series share the screens, so the ratio has little scatter, and 3% is a loose bound.
        """
        short = simulated_tilt_arcsec(
            exposure_s=0.002, duration_s=80.0, substeps=3, screen_points=128
        )
        long = simulated_tilt_arcsec(
            exposure_s=0.010, duration_s=80.0, substeps=6, screen_points=128
        )
        measured = long.var(axis=0).mean() / short.var(axis=0).mean()
        spectrum = models.tilt_spectrum(D, 20.0, 10.0)
        model = spectrum.exposure_variance_ratio(0.010) / spectrum.exposure_variance_ratio(0.002)
        assert measured == pytest.approx(model, rel=0.03)


class TestDetrending:
    @pytest.mark.parametrize("order", [0, 1, 2, 3])
    def test_the_response_integrates_to_the_number_of_removed_modes(self, order: int) -> None:
        """For a flat spectrum the fit removes `order + 1` of the `T f_s` samples' variance.

        The integral of the response over all frequencies is `(order + 1) / T`.
        """
        window = 60.0
        nu = np.geomspace(1e-5, 2e3, 200_000)
        response = models.detrend_response(nu, window, order)
        integral = 2.0 * np.trapezoid(response, nu)
        assert integral == pytest.approx((order + 1) / window, rel=0.01)
        assert response[0] == pytest.approx(order + 1, rel=1e-3) or response[0] > 0.9

    def test_the_fraction_of_the_variance_that_a_quadratic_fit_removes(self) -> None:
        """0.3% for L0 = 20 m, and 8% for Kolmogorov turbulence (the notes: 9% for a linear fit)."""
        assert models.tilt_spectrum(D, 20.0, 10.0).detrend_variance_fraction(60.0, 2, 0.002) == (
            pytest.approx(0.0028, abs=0.0005)
        )
        assert models.tilt_spectrum(D, math.inf, 10.0).detrend_variance_fraction(
            60.0, 2, 0.002
        ) == pytest.approx(0.08, abs=0.015)

    def test_a_longer_window_loses_less_and_a_higher_order_loses_more(self) -> None:
        spectrum = models.tilt_spectrum(D, 20.0, 10.0)
        short, long = (spectrum.detrend_variance_fraction(t, 2, 0.002) for t in (20.0, 60.0))
        assert short > long
        assert spectrum.detrend_variance_fraction(60.0, 3, 0.002) > long
        assert spectrum.detrend_variance_fraction(60.0, 1, 0.002) < long


class TestAutocorrelation:
    def test_decays_with_the_lag_and_starts_at_one(self) -> None:
        spectrum = models.tilt_spectrum(D, 20.0, 10.0)
        kappa = spectrum.autocorrelation(np.array([0.0, 0.0113, 0.04, 0.12, 0.5]), 0.002)
        assert kappa[0] == pytest.approx(1.0)
        assert list(kappa[1:]) == sorted(kappa[1:], reverse=True)
        assert kappa[2] == pytest.approx(0.195, abs=0.01)  # 8 aperture diameters apart

    def test_a_faster_wind_decorrelates_sooner(self) -> None:
        lags = np.array([0.04])
        slow = models.tilt_spectrum(D, 20.0, 5.0).autocorrelation(lags, 0.002)[0]
        fast = models.tilt_spectrum(D, 20.0, 20.0).autocorrelation(lags, 0.002)[0]
        assert slow > fast


class TestCentroidGain:
    def test_the_windowed_centroid_has_slightly_more_variance_than_the_g_tilt(self) -> None:
        """1.6% for a 16-pixel aperture in bin1 (radius 6.2 lambda / D), as the simulation shows."""
        radius = 8.0 * 1.910 / models.lambda_over_d_arcsec(600e-9, D)
        assert radius == pytest.approx(6.17, abs=0.02)
        assert models.windowed_centroid_variance_ratio(radius) == pytest.approx(1.016, abs=0.002)

    def test_decreases_with_the_aperture_and_clamps_a_tiny_one(self) -> None:
        ratios = [models.windowed_centroid_variance_ratio(r) for r in (3.0, 6.0, 12.0)]
        assert ratios == sorted(ratios, reverse=True)
        assert models.windowed_centroid_variance_ratio(
            0.1
        ) == models.windowed_centroid_variance_ratio(2.0)

    def test_spectrum_arguments_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            models.tilt_spectrum(D, 20.0, 0.0)
