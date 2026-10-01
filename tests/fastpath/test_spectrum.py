"""The Welch spectrum, its units, the degrees of freedom, and the vibration lines."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from seeingmon.fastpath.spectrum import (
    aliasing_expected,
    compute_spectrum,
    fill_short_gaps,
    log_bins,
    merge_lines,
    welch_dof,
)
from tests.fastpath.helpers import synthetic_tilt_arcsec

FloatArray = npt.NDArray[np.float64]

RATE_HZ = 88.5
PERIOD_S = 1.0 / RATE_HZ
SAMPLES = 5310  # a 60 s window of bin1 frames


@pytest.fixture(scope="module")
def turbulence() -> FloatArray:
    """Eight windows of turbulent image motion at the bin1 frame rate, in arcseconds."""
    return synthetic_tilt_arcsec(0.10, rate_hz=RATE_HZ, samples=8 * SAMPLES, seed=21)


def window(series: FloatArray, index: int = 0) -> FloatArray:
    return series[index * SAMPLES : (index + 1) * SAMPLES]


class TestUnits:
    def test_white_noise_has_a_flat_one_sided_spectrum_of_2_sigma_squared_dt(self) -> None:
        """White noise of variance 0.04 arcsec^2 has the level 2 sigma^2 dt = 9.0e-4 arcsec^2/Hz."""
        rng = np.random.default_rng(1)
        x = rng.normal(0.0, 0.2, SAMPLES)
        y = rng.normal(0.0, 0.2, SAMPLES)
        spectrum = compute_spectrum(x, y, PERIOD_S)
        assert spectrum is not None
        level = 2.0 * 0.04 * PERIOD_S
        assert float(np.mean(spectrum.psd_x[2:-2])) == pytest.approx(level, rel=0.03)
        assert float(np.mean(spectrum.psd_y[2:-2])) == pytest.approx(level, rel=0.03)
        assert spectrum.nyquist_hz == pytest.approx(RATE_HZ / 2.0)

    def test_the_spectrum_integrates_to_the_variance(self, turbulence: FloatArray) -> None:
        """Parseval: the integral of the spectrum is the variance of the detrended series."""
        x = window(turbulence)[:, 0]
        y = window(turbulence)[:, 1]
        spectrum = compute_spectrum(x - x.mean(), y - y.mean(), PERIOD_S)
        assert spectrum is not None
        df = float(spectrum.freq_hz[1] - spectrum.freq_hz[0])
        assert float(np.sum(spectrum.psd_x) * df) == pytest.approx(np.var(x), rel=0.1)
        assert float(np.sum(spectrum.psd_y) * df) == pytest.approx(np.var(y), rel=0.1)

    def test_the_bin_scatter_follows_the_degrees_of_freedom(self) -> None:
        """A bin of a Welch spectrum is chi-squared distributed: its scatter is sqrt(2 / dof)."""
        rng = np.random.default_rng(2)
        spectrum = compute_spectrum(rng.normal(size=SAMPLES), rng.normal(size=SAMPLES), PERIOD_S)
        assert spectrum is not None
        bins = np.concatenate([spectrum.psd_x[2:-2], spectrum.psd_y[2:-2]])
        scatter = float(np.std(bins) / np.mean(bins))
        assert scatter == pytest.approx(np.sqrt(2.0 / spectrum.dof), rel=0.12)

    def test_degrees_of_freedom_of_overlapped_and_independent_segments(self) -> None:
        length = 180
        window_ = np.hanning(length + 2)[1:-1]
        assert welch_dof(1, window_, 90) == 2.0
        # Independent segments (no overlap): twice the segment count.
        assert welch_dof(40, window_, length) == pytest.approx(80.0)
        # 50% overlap of Hann windows: 36 K^2 / (19 K - 1), the textbook value (77.8 for K = 41).
        assert welch_dof(41, window_, 90) == pytest.approx(36 * 41**2 / (19 * 41 - 1), rel=0.01)

    def test_a_window_of_60_seconds_gives_about_80_degrees_of_freedom(
        self, turbulence: FloatArray
    ) -> None:
        spectrum = compute_spectrum(window(turbulence)[:, 0], window(turbulence)[:, 1], PERIOD_S)
        assert spectrum is not None
        assert spectrum.segments == pytest.approx(58, abs=3)
        assert spectrum.dof == pytest.approx(1.89 * spectrum.segments, rel=0.03)

    def test_too_short_a_series_gives_no_spectrum(self) -> None:
        rng = np.random.default_rng(3)
        assert compute_spectrum(rng.normal(size=300), rng.normal(size=300), PERIOD_S) is None

    def test_mismatched_series_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            compute_spectrum(np.zeros(100), np.zeros(99), PERIOD_S)


class TestLines:
    @staticmethod
    def with_lines(
        series: FloatArray, lines: list[tuple[float, float]]
    ) -> tuple[FloatArray, FloatArray]:
        t = np.arange(len(series)) * PERIOD_S
        x, y = series[:, 0].copy(), series[:, 1].copy()
        for frequency, amplitude in lines:
            x += amplitude * np.sin(2.0 * np.pi * frequency * t + 0.7)
            y += 0.5 * amplitude * np.sin(2.0 * np.pi * frequency * t + 2.1)
        return x, y

    def test_finds_injected_vibration_lines_at_their_frequencies(
        self, turbulence: FloatArray
    ) -> None:
        """Lines at 7.3 Hz (0.5 arcsec) and 21.9 Hz (0.15 arcsec) in turbulence of 0.42 arcsec rms.

        The turbulence spectrum is highest at low frequencies, so a line at 7 Hz needs about 0.25
        arcsec to stand 5 times above it, and one at 22 Hz needs 0.1 arcsec. The frequency
        resolution of the 2 s segments is 0.5 Hz, and the parabola through the three highest bins
        locates a line to 0.1 Hz.
        """
        x, y = self.with_lines(window(turbulence), [(7.3, 0.5), (21.9, 0.15)])
        spectrum = compute_spectrum(x, y, PERIOD_S)
        assert spectrum is not None
        assert len(spectrum.lines_hz) == 2
        assert spectrum.lines_hz[0] == pytest.approx(7.3, abs=0.15)
        assert spectrum.lines_hz[1] == pytest.approx(21.9, abs=0.15)

    def test_a_line_in_one_axis_only_is_found(self, turbulence: FloatArray) -> None:
        t = np.arange(SAMPLES) * PERIOD_S
        x = window(turbulence)[:, 0] + 0.15 * np.sin(2.0 * np.pi * 33.0 * t)
        spectrum = compute_spectrum(x, window(turbulence)[:, 1], PERIOD_S)
        assert spectrum is not None
        assert len(spectrum.lines_hz) == 1
        assert spectrum.lines_hz[0] == pytest.approx(33.0, abs=0.15)

    def test_turbulence_alone_has_no_lines(self, turbulence: FloatArray) -> None:
        """Eight windows of turbulence at 5 times the local median: no false alarms."""
        for index in range(8):
            w = window(turbulence, index)
            spectrum = compute_spectrum(w[:, 0], w[:, 1], PERIOD_S)
            assert spectrum is not None
            assert spectrum.lines_hz == ()

    def test_a_line_above_the_nyquist_frequency_appears_at_its_folded_frequency(
        self, turbulence: FloatArray
    ) -> None:
        """A 70 Hz shake sampled at 88.5 frames per second shows at 88.5 - 70 = 18.5 Hz."""
        x, y = self.with_lines(window(turbulence), [(70.0, 0.15)])
        spectrum = compute_spectrum(x, y, PERIOD_S)
        assert spectrum is not None
        assert len(spectrum.lines_hz) == 1
        assert spectrum.lines_hz[0] == pytest.approx(RATE_HZ - 70.0, abs=0.2)

    def test_lines_below_the_minimum_frequency_are_not_reported(
        self, turbulence: FloatArray
    ) -> None:
        x, y = self.with_lines(window(turbulence), [(0.5, 0.5)])
        spectrum = compute_spectrum(x, y, PERIOD_S, min_line_hz=1.0)
        assert spectrum is not None
        assert spectrum.lines_hz == ()

    def test_a_weak_line_stays_below_the_threshold(self, turbulence: FloatArray) -> None:
        x, y = self.with_lines(window(turbulence), [(15.0, 0.02)])  # 0.0002 arcsec^2
        spectrum = compute_spectrum(x, y, PERIOD_S)
        assert spectrum is not None
        assert spectrum.lines_hz == ()
        loose = compute_spectrum(x, y, PERIOD_S, threshold=1.5)
        assert loose is not None
        assert len(loose.lines_hz) >= 1


class TestGaps:
    def test_short_gaps_are_interpolated_and_long_gaps_are_not(self) -> None:
        series = np.arange(20.0)
        series[5:7] = np.nan  # a gap of two
        series[10:15] = np.nan  # a gap of five
        filled, interpolated = fill_short_gaps(series, 3)
        assert filled[5:7].tolist() == pytest.approx([5.0, 6.0])
        assert np.isnan(filled[10:15]).all()
        assert interpolated.tolist() == [i in (5, 6) for i in range(20)]

    def test_leading_and_trailing_gaps_stay_empty(self) -> None:
        series = np.array([np.nan, np.nan, 1.0, 2.0, np.nan])
        filled, interpolated = fill_short_gaps(series, 3)
        assert np.isnan(filled[[0, 1, 4]]).all()
        assert not interpolated.any()

    def test_a_series_with_random_gaps_gives_the_same_spectrum_and_the_same_line(
        self, turbulence: FloatArray
    ) -> None:
        rng = np.random.default_rng(4)
        t = np.arange(SAMPLES) * PERIOD_S
        clean_x = window(turbulence)[:, 0] + 0.35 * np.sin(2.0 * np.pi * 12.0 * t)
        clean_y = window(turbulence)[:, 1]
        x, y = clean_x.copy(), clean_y.copy()
        missing = rng.random(SAMPLES) < 0.03  # isolated lost frames
        x[missing] = np.nan
        y[missing] = np.nan
        reference = compute_spectrum(clean_x, clean_y, PERIOD_S)
        gapped = compute_spectrum(x, y, PERIOD_S)
        assert reference is not None
        assert gapped is not None
        assert gapped.lines_hz[0] == pytest.approx(12.0, abs=0.15)
        low = slice(1, 40)  # below 20 Hz, where interpolation costs little
        assert float(np.mean(gapped.psd_x[low])) == pytest.approx(
            float(np.mean(reference.psd_x[low])), rel=0.05
        )

    def test_a_long_gap_costs_the_segments_that_it_touches(self, turbulence: FloatArray) -> None:
        x, y = window(turbulence)[:, 0].copy(), window(turbulence)[:, 1].copy()
        x[2000:2300] = np.nan
        y[2000:2300] = np.nan
        full = compute_spectrum(window(turbulence)[:, 0], window(turbulence)[:, 1], PERIOD_S)
        cut = compute_spectrum(x, y, PERIOD_S)
        assert full is not None
        assert cut is not None
        assert cut.segments < full.segments
        assert cut.segments >= full.segments - 6  # a 300-frame gap spoils about five segments

    def test_too_many_gaps_leave_no_spectrum(self) -> None:
        rng = np.random.default_rng(5)
        x = rng.normal(size=SAMPLES)
        x[::5] = np.nan  # a gap every fifth frame: 20% interpolated in every segment
        assert compute_spectrum(x, x.copy(), PERIOD_S) is None


class TestReducedBins:
    def test_the_bins_are_logarithmic_and_hold_the_mean_spectrum(
        self, turbulence: FloatArray
    ) -> None:
        spectrum = compute_spectrum(
            window(turbulence)[:, 0], window(turbulence)[:, 1], PERIOD_S, bins=24
        )
        assert spectrum is not None
        freq = spectrum.bin_freq_hz
        assert 12 <= len(freq) <= 24
        assert list(freq) == sorted(freq)
        assert freq[0] >= spectrum.freq_hz[0]
        assert freq[-1] <= spectrum.freq_hz[-1]
        assert len(spectrum.bin_psd_x) == len(spectrum.bin_psd_y) == len(freq)
        # The reduced spectrum keeps the variance to within the width of the bins.
        full = float(np.trapezoid(spectrum.psd_x, spectrum.freq_hz))
        reduced = float(np.trapezoid(spectrum.bin_psd_x, freq))
        assert reduced == pytest.approx(full, rel=0.35)

    def test_empty_bins_are_dropped(self) -> None:
        freq = np.arange(1.0, 11.0)  # ten frequencies for 24 bins
        psd = np.ones(10)
        kept_freq, kept_x, kept_y = log_bins(freq, psd, 2 * psd, 24)
        assert len(kept_freq) == 10
        assert list(kept_x) == [1.0] * 10
        assert list(kept_y) == [2.0] * 10

    def test_no_frequencies_give_no_bins(self) -> None:
        empty = np.empty(0)
        assert all(len(a) == 0 for a in log_bins(empty, empty, empty, 24))


class TestHelpers:
    def test_close_lines_merge_and_the_stronger_wins(self) -> None:
        assert merge_lines([(10.0, 6.0), (10.4, 9.0), (25.0, 7.0)], 0.75) == [10.4, 25.0]
        assert merge_lines([], 1.0) == []

    @pytest.mark.parametrize(
        ("nyquist", "expected"), [(44.25, True), (60.0, True), (100.0, False), (180.0, False)]
    )
    def test_aliasing_is_expected_when_nyquist_is_below_the_turbulence_corner(
        self, nyquist: float, expected: bool
    ) -> None:
        """The corner lies at 0.5 v / D = 100 Hz at most for D = 50 mm and v = 10 m/s."""
        assert aliasing_expected(nyquist, 0.05, 10.0) is expected
