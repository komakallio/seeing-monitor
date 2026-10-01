"""The scintillation index against an injected index, the Poisson floor, and trends."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

from seeingmon.fastpath.scintillation import scintillation_index

FloatArray = npt.NDArray[np.float64]

PERIOD_S = 0.0113
SAMPLES = 5310
MEAN_E = 14_000.0  # Polaris at 2 ms


def lognormal_flux(index: float, seed: int, samples: int = SAMPLES) -> FloatArray:
    """Flux with a mean of `MEAN_E` and a relative variance of `index` (a log-normal factor)."""
    rng = np.random.default_rng(seed)
    sigma_sq = math.log1p(index)
    factor = np.exp(math.sqrt(sigma_sq) * rng.standard_normal(samples) - 0.5 * sigma_sq)
    return np.asarray(MEAN_E * factor, dtype=np.float64)


def measured(
    flux: FloatArray, trend_s: float = 1.0, exclude: npt.NDArray[np.bool_] | None = None
) -> float:
    result = scintillation_index(
        flux, PERIOD_S, trend_s=trend_s, pixel_var_e2=2.65**2, area_px2=200.0, exclude=exclude
    )
    assert result is not None
    return result.index


class TestIndex:
    @pytest.mark.parametrize("index", [0.05, 0.15, 0.30])
    def test_recovers_the_injected_index(self, index: float) -> None:
        """Four windows of 5,310 frames: the standard error of the mean index is about 2%."""
        values = [measured(lognormal_flux(index, seed)) for seed in range(4)]
        assert float(np.mean(values)) == pytest.approx(index, rel=0.08)

    def test_the_poisson_floor_is_subtracted(self) -> None:
        """A star with no scintillation has only photon and pixel noise, so the index is zero."""
        rng = np.random.default_rng(1)
        flux = rng.poisson(MEAN_E, SAMPLES) + rng.normal(0.0, 2.65 * math.sqrt(200.0), SAMPLES)
        result = scintillation_index(
            flux.astype(np.float64), PERIOD_S, pixel_var_e2=2.65**2, area_px2=200.0
        )
        assert result is not None
        floor = (MEAN_E + 200.0 * 2.65**2) / MEAN_E**2
        assert result.floor == pytest.approx(floor, rel=0.01)
        assert result.raw_variance == pytest.approx(floor, rel=0.06)
        assert result.index < 0.1 * floor  # zero within the sampling error of the floor

    def test_the_index_is_clipped_at_zero(self) -> None:
        flux = np.full(SAMPLES, MEAN_E)
        assert measured(flux) == 0.0

    def test_noise_and_scintillation_add(self) -> None:
        rng = np.random.default_rng(2)
        flux = lognormal_flux(0.15, 3)
        noisy = flux + rng.normal(0.0, 0.05 * MEAN_E, SAMPLES)  # 0.0025 of extra variance
        assert measured(noisy, trend_s=1.0) == pytest.approx(0.15 + 0.0025, rel=0.08)
        # Declaring the noise (as the pixel variance) takes it out again.
        extra = scintillation_index(
            noisy, PERIOD_S, pixel_var_e2=2.65**2 + (0.05 * MEAN_E) ** 2 / 200.0, area_px2=200.0
        )
        assert extra is not None
        assert extra.index == pytest.approx(0.15, rel=0.08)


class TestTrends:
    def test_a_slow_transparency_change_does_not_count(self) -> None:
        """A 20% swing at 0.05 Hz (a 20 s period) would add 0.02 to the plain variance."""
        t = np.arange(SAMPLES) * PERIOD_S
        flux = lognormal_flux(0.15, 5) * (1.0 + 0.2 * np.sin(2.0 * math.pi * 0.05 * t))
        assert float(np.var(flux) / np.mean(flux) ** 2) > 0.16  # the plain variance is inflated
        assert measured(flux) == pytest.approx(0.15, rel=0.08)

    def test_a_drop_for_cloud_adds_little(self) -> None:
        """A thin cloud takes the flux down by 40% for 10 s of the 60 s window."""
        flux = lognormal_flux(0.15, 6)
        flux[1000:1900] *= 0.6
        assert measured(flux) == pytest.approx(0.15, rel=0.12)

    def test_the_trend_length_is_the_cutoff(self) -> None:
        """A 0.1 s moving average also removes some scintillation. The default 1 s removes 1%."""
        flux = lognormal_flux(0.15, 7)
        assert measured(flux, trend_s=0.1) < measured(flux, trend_s=1.0)

    def test_scintillation_correlated_over_a_few_frames_is_kept(self) -> None:
        """A flux that stays correlated for 5 frames has the same index at 1 s of trend."""
        rng = np.random.default_rng(8)
        n = SAMPLES + 4
        sigma_sq = math.log1p(0.15)
        base = rng.standard_normal(n)
        smooth = np.convolve(base, np.ones(5) / math.sqrt(5), mode="valid")
        flux = MEAN_E * np.exp(math.sqrt(sigma_sq) * smooth - 0.5 * sigma_sq)
        assert measured(flux) == pytest.approx(0.15, rel=0.1)


class TestGapsAndExclusions:
    def test_missing_frames_leave_the_index_alone(self) -> None:
        flux = lognormal_flux(0.15, 9)
        flux[::10] = np.nan
        flux[3000:3100] = np.nan
        assert measured(flux) == pytest.approx(0.15, rel=0.1)

    def test_excluded_frames_do_not_count(self) -> None:
        """Frames that the caller excludes (such as saturated ones) leave no trace in the index."""
        flux = lognormal_flux(0.15, 10)
        clean = measured(flux)
        spoiled = flux.copy()
        bad = np.zeros(SAMPLES, dtype=bool)
        bad[::200] = True
        spoiled[bad] *= 8.0  # a handful of frames that read far too bright
        assert measured(spoiled) > 1.5 * clean
        assert measured(spoiled, exclude=bad) == pytest.approx(clean, rel=0.05)

    def test_too_few_frames_give_none(self) -> None:
        flux = lognormal_flux(0.15, 11, samples=15)
        assert scintillation_index(flux, PERIOD_S) is None
        sparse = np.full(1000, np.nan)
        sparse[::100] = MEAN_E
        assert scintillation_index(sparse, PERIOD_S) is None

    def test_nonpositive_flux_is_ignored(self) -> None:
        flux = lognormal_flux(0.15, 12)
        flux[100:110] = -50.0
        flux[200:210] = 0.0
        assert measured(flux) == pytest.approx(0.15, rel=0.1)

    def test_reports_the_mean_flux_and_the_number_of_frames(self) -> None:
        flux = lognormal_flux(0.15, 13)
        result = scintillation_index(flux, PERIOD_S, pixel_var_e2=2.65**2, area_px2=200.0)
        assert result is not None
        assert result.mean_flux_e == pytest.approx(MEAN_E, rel=0.02)
        assert result.frames == SAMPLES
