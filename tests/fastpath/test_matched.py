"""The matched filters: the SNR that decides whether the star is in a frame.

The frames are of the reference profile in bin1 at gain 0 (a 12-bit ADC at 3.5 e- per count, with
2.65 e- of read noise), as the analyzer sets up the kernel. The filters are Gaussians of 1, 2, and
4 Airy FWHM of the mode (1.33, 2.67, and 5.33 px). Most stars are Gaussians of the Airy FWHM, whose
pixels hold the exact integral, so the first filter matches them. The bright sky is the daylight
sky of the detection estimate near the pole: 4,300 e- in a pixel at the exposure of 1.23 ms that
puts it at 0.3 of the full well, where the median frame holds 7,980 e- of Polaris
(`docs/research-notes.md`, "Polaris in a bright sky").
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import numpy.typing as npt
import pytest

from seeingmon.fastpath import FastPathAnalyzer, FastPathConfig
from seeingmon.fastpath.kernel import (
    FrameCalibration,
    KernelParams,
    Measurement,
    measure_frame,
    search_frame,
)
from seeingmon.fastpath.matched import PHASES, matched_filter, near, search
from seeingmon.profile import Profile
from tests.fastpath.helpers import box_integrated_gaussian, digitize

SHAPE = (128, 128)
ROI_X, ROI_Y = 100, 200
CENTER = (ROI_X + 63.5, ROI_Y + 63.5)  # the predicted place, in sensor pixels
READ_NOISE_E = 2.65
E_PER_ADU = 3.5
DAY_SKY_E = 4_300.0
DAY_POLARIS_E = 7_980.0
SEARCH_RADIUS_PX = 20.0  # the default `[scheduler.search] radius_px`
DETECT_SNR = 10.0  # the default `[scheduler.search] detect_snr`
BURST_FRAMES = 50  # the default `[scheduler.search] burst_frames`
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
PLATE_SCALE = 1.91  # arcsec per px in bin1

UInt16Frame = npt.NDArray[np.uint16]


@pytest.fixture(scope="module")
def setup(profile: Profile) -> tuple[KernelParams, FrameCalibration]:
    """The kernel of the analyzer for the fast mode of the reference profile."""
    adc_bits = profile.mode("bin1").adc_bits
    analyzer = FastPathAnalyzer(profile, FastPathConfig())
    return analyzer.kernel_setup("bin1", 0, 1230, adc_bits, 16)


def fwhm(params: KernelParams) -> float:
    """The FWHM of the first filter, the Airy FWHM of the mode."""
    return params.matched_fwhms_px[0]


def sigma_px(params: KernelParams) -> float:
    return fwhm(params) / FWHM_PER_SIGMA


def first_only(params: KernelParams) -> KernelParams:
    """The kernel with the first filter alone, as the search was before the bank."""
    return replace(params, matched_fwhms_px=params.matched_fwhms_px[:1])


def camera_frame(
    rng: np.random.Generator,
    sky_e: float,
    star: tuple[float, float, float, float] | None = None,
) -> UInt16Frame:
    """A frame of the camera: the sky, a star `(x, y, sigma, flux)` in ROI pixels, and noise.

    The photon noise is Gaussian, which holds within 1% at the counts here.
    """
    mean = np.full(SHAPE, sky_e)
    if star is not None:
        x, y, sigma, flux = star
        mean = mean + box_integrated_gaussian(SHAPE, x, y, sigma, flux)
    electrons = mean + rng.standard_normal(SHAPE) * np.sqrt(mean + READ_NOISE_E**2)
    counts = digitize(electrons, e_per_adu=E_PER_ADU, read_noise_e=0.0, adc_bits=12)
    return np.asarray(counts, dtype=np.uint16)


def matched_sums(sigma: float, dx: float, dy: float) -> tuple[float, float]:
    """`sum(w^2)` and `sum(w^3)` of a Gaussian filter at an offset, from an independent stamp."""
    weights = box_integrated_gaussian((15, 15), 7.0 + dx, 7.0 + dy, sigma)
    return float(np.sum(weights**2)), float(np.sum(weights**3))


class TestTheFilter:
    def test_the_bank_is_1_2_and_4_airy_widths(
        self, setup: tuple[KernelParams, FrameCalibration]
    ) -> None:
        params, _ = setup
        # The Airy FWHM of bin1 is 1.333 px.
        assert params.matched_fwhms_px == pytest.approx((1.333, 2.667, 5.334), abs=0.001)

    def test_the_filter_has_the_area_of_a_pixelated_gaussian(
        self, setup: tuple[KernelParams, FrameCalibration]
    ) -> None:
        """`1 / sum(w^2)` is about `4 pi (sigma^2 + 1/12)`: 5.08 px^2 for 1.33 px, 65 for 5.33."""
        params, _ = setup
        for width in params.matched_fwhms_px:
            sigma = width / FWHM_PER_SIGMA
            area = matched_filter(width).effective_area_px2
            # 1%: the pixel phases of the table against the continuous approximation.
            assert area == pytest.approx(4 * math.pi * (sigma**2 + 1 / 12), rel=0.01), width

    def test_the_snr_follows_the_formula(
        self, setup: tuple[KernelParams, FrameCalibration]
    ) -> None:
        """A star on a grid point of the filter: `S = F sum(w^2)` and the noise of the formula."""
        params, _ = setup
        sigma = sigma_px(params)
        x, y = 64.0 + 0.125, 63.0 - 0.125  # a grid point, a quarter of a pixel from its neighbors
        flux, level, pixel_var = 8_000.0, 1_000.0, 4_300.0
        image = level + box_integrated_gaussian(SHAPE, x, y, sigma, flux)
        best = near(image, 64, 63, matched_filter(fwhm(params)), level, 1.0, pixel_var)
        sum_w2, sum_w3 = matched_sums(sigma, 0.125, -0.125)
        # The image is the filter, so the photon noise `F sum(w^2 P)` is `F sum(w^3)`.
        expected = flux * sum_w2 / math.sqrt(pixel_var * sum_w2 + flux * sum_w3)
        assert best.snr == pytest.approx(expected, rel=1e-6)
        assert best.flux_e == pytest.approx(flux, rel=1e-6)
        # 0.02 px: the parabola through the grid points (see the next test).
        assert best.x == pytest.approx(x, abs=0.02)
        assert best.y == pytest.approx(y, abs=0.02)

    def test_the_photon_noise_of_a_filter_wider_than_the_star(
        self, setup: tuple[KernelParams, FrameCalibration]
    ) -> None:
        """The widest filter on a star of the Airy FWHM: the photon noise is `F sum(w^2 P)`.

        The pixels give it, whatever the image. The form that assumes a matched image,
        `(S / sum(w^2)) sum(w^3)`, would understate it, and the SNR of the dark sky would pass
        the best weighting of the pixels, which no filter can.
        """
        params, _ = setup
        wide = params.matched_fwhms_px[-1]
        x, y = 64.0 + 0.125, 63.0 - 0.125  # a grid point of the wide filter too
        flux, level, pixel_var = 100_000.0, 1_000.0, 10.0  # a dark sky: the star's photons rule
        star = box_integrated_gaussian(SHAPE, x, y, sigma_px(params), flux)
        best = near(level + star, 64, 63, matched_filter(wide), level, 1.0, pixel_var)
        weights = box_integrated_gaussian(SHAPE, x, y, wide / FWHM_PER_SIGMA)
        signal = float(np.sum(weights * star))
        sum_w2 = float(np.sum(weights**2))
        expected = signal / math.sqrt(pixel_var * sum_w2 + float(np.sum(weights**2 * star)))
        # 1e-4: the stamp of the filter holds all but about 1e-5 of its weight.
        assert best.snr == pytest.approx(expected, rel=1e-4)
        limit = math.sqrt(float(np.sum(star**2 / (pixel_var + star))))  # the best weighting
        assert best.snr < limit
        assumed = signal / math.sqrt(
            pixel_var * sum_w2 + signal / sum_w2 * float(np.sum(weights**3))
        )
        assert assumed > limit  # the matched form would claim more than the photons allow

    def test_the_position_between_the_grid_points(
        self, setup: tuple[KernelParams, FrameCalibration]
    ) -> None:
        """Without noise, the parabola through the grid puts the star within 0.02 px."""
        params, _ = setup
        matched = matched_filter(fwhm(params))
        worst = 0.0
        for fx in np.linspace(-0.5, 0.45, 9):
            for fy in np.linspace(-0.45, 0.5, 9):
                x, y = 64.0 + fx, 63.0 + fy
                image = 100.0 + box_integrated_gaussian(SHAPE, x, y, sigma_px(params), 1e4)
                best = search(image, 64.0, 63.0, SEARCH_RADIUS_PX, matched, 100.0, 1.0, 10.0)
                assert best is not None
                worst = max(worst, abs(best.x - x), abs(best.y - y))
        assert worst < 0.02  # a 16th of the grid's quarter pixel would be 0.016
        assert PHASES == 4

    def test_the_search_stays_within_its_radius(
        self, setup: tuple[KernelParams, FrameCalibration]
    ) -> None:
        params, _ = setup
        matched = matched_filter(fwhm(params))
        image = 100.0 + box_integrated_gaussian(SHAPE, 94.0, 64.0, sigma_px(params), 1e4)
        outside = search(image, 64.0, 64.0, SEARCH_RADIUS_PX, matched, 100.0, 1.0, 10.0)
        # 30 px away: the widened circle and the grid around its brightest pixel reach 28 px,
        # where the star leaves nothing but rounding.
        assert outside is None or outside.snr < 1e-6
        inside = search(image, 64.0, 64.0, 40.0, matched, 100.0, 1.0, 10.0)
        assert inside is not None
        assert inside.x == pytest.approx(94.0, abs=0.02)
        assert inside.snr > 50.0  # 10,000 e- with little sky: about 0.83 sqrt(F)
        assert search(image, -40.0, 64.0, SEARCH_RADIUS_PX, matched, 100.0, 1.0, 10.0) is None

    @pytest.mark.parametrize("beyond_px", [3.0, 5.0, 8.0])
    def test_a_wide_filter_puts_no_star_at_the_edge_of_the_circle(
        self, setup: tuple[KernelParams, FrameCalibration], beyond_px: float
    ) -> None:
        """A bright star just beyond the circle brightens the widest filter's image inside it.

        The search looks at the widened circle, finds the star where it is, and reports none.
        """
        params, _ = setup
        wide = matched_filter(params.matched_fwhms_px[-1])
        x = 64.0 + SEARCH_RADIUS_PX + beyond_px
        image = 100.0 + box_integrated_gaussian(SHAPE, x, 64.0, sigma_px(params), 1e5)
        assert search(image, 64.0, 64.0, SEARCH_RADIUS_PX, wide, 100.0, 1.0, 10.0) is None
        found = search(image, 64.0, 64.0, SEARCH_RADIUS_PX + 10.0, wide, 100.0, 1.0, 10.0)
        assert found is not None
        assert found.x == pytest.approx(x, abs=0.1)  # 0.1 px: the wide filter's parabola

    def test_a_star_at_the_frame_edge_is_found(
        self, setup: tuple[KernelParams, FrameCalibration]
    ) -> None:
        """Pixels outside the frame read as the sky, so the stamp of the filter may cross it."""
        params, _ = setup
        image = 100.0 + box_integrated_gaussian(SHAPE, 0.3, 126.7, sigma_px(params), 1e4)
        best = search(
            image, 10.0, 120.0, SEARCH_RADIUS_PX, matched_filter(fwhm(params)), 100.0, 1.0, 10.0
        )
        assert best is not None
        # 0.05 px: the part of the star beyond the edge is missing, which pulls the peak inward.
        assert best.x == pytest.approx(0.3, abs=0.05)
        assert best.y == pytest.approx(126.7, abs=0.05)

    def test_rejects_a_filter_of_an_impossible_size(self) -> None:
        with pytest.raises(ValueError, match="FWHM"):
            matched_filter(0.1)
        with pytest.raises(ValueError, match="matched_fwhms_px"):
            KernelParams(matched_fwhms_px=(1.3, 50.0))


class TestTheFallbackOfTheSearch:
    """Without a matched filter or an electron scale, a search frame is measured as `push` does.

    The kernel then starts at `at`, falls back to the brightest patch, and decides by the SNR of
    the centroid aperture, or by the contrast without an electron scale.
    """

    @staticmethod
    def same(first: Measurement, second: Measurement) -> None:
        np.testing.assert_equal(tuple(first), tuple(second))  # NaN equals NaN here

    def test_without_a_matched_filter(self, setup: tuple[KernelParams, FrameCalibration]) -> None:
        params, calibration = setup
        plain = replace(params, matched_fwhms_px=())
        rng = np.random.default_rng(25)
        data = camera_frame(rng, 1_000.0, (64.3, 63.8, sigma_px(params), 50_000.0))
        at = (ROI_X + 60.0, ROI_Y + 60.0)
        found = search_frame(data, ROI_X, ROI_Y, plain, calibration, at, SEARCH_RADIUS_PX)
        self.same(found, measure_frame(data, ROI_X, ROI_Y, plain, calibration, at))
        assert found.found
        assert math.isnan(found.matched_snr)
        assert found.width_x > 0  # the centroid loop ran, which the search alone does not

    def test_without_an_electron_scale(self, setup: tuple[KernelParams, FrameCalibration]) -> None:
        params, calibration = setup
        unscaled = FrameCalibration(calibration.full_scale_dn, calibration.saturation_dn)
        rng = np.random.default_rng(26)
        data = camera_frame(rng, 1_000.0, (64.3, 63.8, sigma_px(params), 50_000.0))
        found = search_frame(data, ROI_X, ROI_Y, params, unscaled, CENTER, SEARCH_RADIUS_PX)
        self.same(found, measure_frame(data, ROI_X, ROI_Y, params, unscaled, CENTER))
        assert found.found
        assert math.isnan(found.snr)
        assert math.isnan(found.matched_snr)


class TestDetectionInDaylight:
    """The matched filters find the star where the wide centroid aperture cannot."""

    def test_polaris_in_the_daylight_sky_of_the_estimate(
        self, setup: tuple[KernelParams, FrameCalibration]
    ) -> None:
        """The median frame of daylight: a matched SNR near 44, and about 8.7 in the aperture.

        The Gaussian star matches the filter, so it gives a little more than the 41 of the
        simulator's Airy image in the detection estimate.
        """
        params, calibration = setup
        rng = np.random.default_rng(21)
        sigma = sigma_px(params)
        matched, aperture = [], []
        for index in range(200):
            x, y = 64.0 + rng.uniform(-0.5, 0.5), 64.0 + rng.uniform(-0.5, 0.5)
            data = camera_frame(rng, DAY_SKY_E, (x, y, sigma, DAY_POLARIS_E))
            m = measure_frame(data, ROI_X, ROI_Y, params, calibration, (ROI_X + 64, ROI_Y + 64))
            assert m.found, index
            matched.append(m.matched_snr)
            aperture.append(m.snr)
        pixel_var = DAY_SKY_E + READ_NOISE_E**2 + E_PER_ADU**2 / 12
        sum_w2, sum_w3 = matched_sums(sigma, 0.125, 0.125)
        flux = DAY_POLARIS_E
        on_grid = flux * sum_w2 / math.sqrt(pixel_var * sum_w2 + flux * sum_w3)
        # 4%: the star falls between the grid points (about 1%), and the sky noise of the border
        # scatters by 3% from frame to frame.
        assert float(np.median(matched)) == pytest.approx(on_grid, rel=0.04)
        inside = 0.97 * flux  # the share of the star in the centroid aperture
        wide = inside / math.sqrt(inside + params.area_px2 * pixel_var)
        # The aperture's SNR sits up to 8% above the formula (4.7% with this seed). The border
        # reads the pixel variance about 3% low, which raises the SNR by 1.4%, and the aperture
        # recenters on the noise at an SNR of 8, which raises its sum by about 3%.
        assert 1.0 <= float(np.median(aperture)) / wide <= 1.08
        assert float(np.median(matched)) > 5 * float(np.median(aperture))

    def test_a_star_that_the_aperture_misses_is_found_by_the_search(
        self, setup: tuple[KernelParams, FrameCalibration]
    ) -> None:
        """A star of 2,800 e- in daylight: about 3 in the aperture, and 17 in the filter.

        The search finds it in every frame, near its place. The aperture alone, as the kernel
        decided before, loses it in most frames.
        """
        params, calibration = setup
        aperture_only = replace(params, matched_fwhms_px=())
        rng = np.random.default_rng(22)
        star = (64.3, 63.8, sigma_px(params), 2_800.0)
        place = (ROI_X + 64.3, ROI_Y + 63.8)
        snrs = []
        found_aperture = 0
        for _ in range(100):
            data = camera_frame(rng, DAY_SKY_E, star)
            m = search_frame(data, ROI_X, ROI_Y, params, calibration, CENTER, SEARCH_RADIUS_PX)
            assert m.found
            # 0.3 px: the position noise of a matched filter at an SNR of 17 is about 0.06 px.
            assert math.hypot(m.x - place[0], m.y - place[1]) < 0.3
            snrs.append(m.matched_snr)
            found_aperture += measure_frame(
                data, ROI_X, ROI_Y, aperture_only, calibration, place
            ).found
        # S / sqrt(844 + 140) for the first filter. The best of three noisy filters sits a little
        # higher, within the 8%.
        assert float(np.median(snrs)) == pytest.approx(17.5, rel=0.08)
        assert found_aperture <= 20  # 3 against a threshold of 6

    @pytest.mark.parametrize("sky_e", [1.0, 30.0, 1_000.0, DAY_SKY_E])
    def test_a_starless_sky_makes_no_star(
        self, setup: tuple[KernelParams, FrameCalibration], sky_e: float
    ) -> None:
        params, calibration = setup
        rng = np.random.default_rng(23)
        found = sum(
            measure_frame(camera_frame(rng, sky_e), ROI_X, ROI_Y, params, calibration).found
            for _ in range(200)
        )
        assert found == 0  # the threshold is 6, far above the noise peaks of a frame (about 4)


class TestAWiderImage:
    """A star whose image is wider than the diffraction limit: defocus, or the color of the optics.

    The owner's recordings show an image 13 times the area of the simulator's. A filter of the
    Airy FWHM then loses much of the SNR, and the bank keeps the filter that fits.
    """

    def test_the_bank_keeps_the_snr_that_the_airy_filter_loses(
        self, setup: tuple[KernelParams, FrameCalibration]
    ) -> None:
        """Polaris in daylight with an image of 10 arcsec FWHM (5.2 px, 3.9 Airy FWHM).

        The best weighting of a Gaussian image of `sigma` gives about
        `F / sqrt(v 4 pi (sigma^2 + 1/12))`, 15.3 here. The widest filter of the bank fits the image
        and reaches it within a few percent, and a burst detects. The filter of the Airy FWHM
        alone reaches about half (`2 s_f s_i / (s_f^2 + s_i^2)`), and a burst would not detect.
        """
        params, calibration = setup
        sigma = 10.0 / PLATE_SCALE / FWHM_PER_SIGMA
        rng = np.random.default_rng(27)
        star = (64.3, 63.8, sigma, DAY_POLARIS_E)
        place = (ROI_X + 64.3, ROI_Y + 63.8)
        bank, airy = [], []
        for _ in range(100):
            data = camera_frame(rng, DAY_SKY_E, star)
            m = search_frame(data, ROI_X, ROI_Y, params, calibration, CENTER, SEARCH_RADIUS_PX)
            assert m.found
            # 1 px: the position noise of a 5.2 px image at an SNR of 15 is about
            # `sqrt(2) sigma / SNR` = 0.2 px along each axis.
            assert math.hypot(m.x - place[0], m.y - place[1]) < 1.0
            bank.append(m.matched_snr)
            only = search_frame(
                data, ROI_X, ROI_Y, first_only(params), calibration, CENTER, SEARCH_RADIUS_PX
            )
            airy.append(only.matched_snr if only.found else 0.0)
        pixel_var = DAY_SKY_E + READ_NOISE_E**2 + E_PER_ADU**2 / 12
        best = DAY_POLARIS_E / math.sqrt(pixel_var * 4 * math.pi * (sigma**2 + 1 / 12))
        assert best == pytest.approx(15.3, abs=0.05)
        # 8%: the photon noise of the star (2%), the sky noise of the border, and the misfit.
        assert float(np.median(bank)) == pytest.approx(best, rel=0.08)
        assert float(np.median(bank)) >= DETECT_SNR
        assert float(np.median(airy)) < 0.6 * best
        assert float(np.median(airy)) < DETECT_SNR

    def test_a_defocused_bright_star_is_not_missing(
        self, setup: tuple[KernelParams, FrameCalibration]
    ) -> None:
        """A star of 3,000 e- in a dark sky, defocused to 8 px sigma, as in the rapid focus mode.

        The filter of the Airy FWHM sees an SNR of about 3, but the centroid aperture holds 39% of
        the star at an SNR of about 13, so the missing-star test of `push` keeps it, as it did when
        the aperture decided alone.
        """
        params, calibration = setup
        rng = np.random.default_rng(28)
        star = (64.0, 64.0, 8.0, 3_000.0)
        at = (ROI_X + 64.0, ROI_Y + 64.0)
        found, matched_below = 0, 0
        for _ in range(100):
            m = measure_frame(camera_frame(rng, 30.0, star), ROI_X, ROI_Y, params, calibration, at)
            found += m.found
            matched_below += m.matched_snr < params.min_snr
        assert found == 100
        assert matched_below >= 90  # the frames that the filter alone would lose

    def test_the_rapid_focus_keeps_a_wide_star_in_daylight(
        self, setup: tuple[KernelParams, FrameCalibration]
    ) -> None:
        """A star of 12 arcsec FWHM in daylight, the widest that rapid focus offers.

        The missing-star test keeps every frame that the aperture alone kept.
        """
        params, calibration = setup
        aperture_only = replace(params, matched_fwhms_px=())
        rng = np.random.default_rng(29)
        star = (64.2, 63.7, 12.0 / PLATE_SCALE / FWHM_PER_SIGMA, DAY_POLARIS_E)
        at = (ROI_X + 64.2, ROI_Y + 63.7)
        both, alone = 0, 0
        for _ in range(100):
            data = camera_frame(rng, DAY_SKY_E, star)
            both += measure_frame(data, ROI_X, ROI_Y, params, calibration, at).found
            alone += measure_frame(data, ROI_X, ROI_Y, aperture_only, calibration, at).found
        assert both >= alone
        assert alone >= 90  # the aperture's SNR of about 8 against the threshold of 6


class TestFalseDetections:
    """How often the burst rule detects Polaris in frames without it.

    A burst detects Polaris when the median SNR of its 50 frames reaches 10, where each frame
    reports the highest matched SNR of the three filters within 20 px of the prediction. On a
    frame without a star, that is the highest of about 69,000 correlated positions of noise: for
    each filter, the reported position lies within 20 px, on one of 16 positions within one of at
    most 1,428 pixels (the 1,257 pixels of the circle and the grid of `near` around them). Each is
    a unit Gaussian, and the star's photon noise only lowers the SNR, so the union bound
    `3 x 22,848 Q(t)` caps the chance that a frame reaches `t`.
    """

    @staticmethod
    def candidates() -> int:
        """The grid positions that the search can report for one filter, at the worst place of
        the center."""
        reach = math.ceil(SEARCH_RADIUS_PX) + 2
        span = range(-reach, reach + 1)
        counts = []
        for fx in range(PHASES):
            for fy in range(PHASES):
                cx, cy = fx / PHASES, fy / PHASES
                circle = [
                    (x, y)
                    for x in span
                    for y in span
                    if (x - cx) ** 2 + (y - cy) ** 2 <= SEARCH_RADIUS_PX**2
                ]
                near = {
                    (x + dx, y + dy) for x, y in circle for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                }
                counts.append(len(near))
        return max(counts) * PHASES**2

    def bound(self, threshold: float, filters: int) -> float:
        """The union bound on the chance that a starless frame reaches `threshold`."""
        return filters * self.candidates() * 0.5 * math.erfc(threshold / math.sqrt(2.0))

    def test_the_noise_peaks_of_starless_frames_stay_under_the_union_bound(
        self, setup: tuple[KernelParams, FrameCalibration]
    ) -> None:
        """2,000 frames of the daylight sky: the noise peaks follow a Gaussian tail.

        The best of the three filters in a frame has a median of about 3.4 and stayed below 5.7.
        Above 4.5 and 5, it falls under the union bound, so the tail of the statistic is no
        heavier than that of Gaussian noise, and the bound carries to 10. In about 10% of the
        frames, every filter peaks beyond the radius, and the frame counts as 0, as it does in
        the scheduler.
        """
        params, calibration = setup
        open_params = replace(params, min_snr=0.0)  # report the peak of every frame
        filters = len(params.matched_fwhms_px)
        rng = np.random.default_rng(24)
        peaks = np.asarray(
            [
                search_frame(
                    camera_frame(rng, DAY_SKY_E),
                    ROI_X,
                    ROI_Y,
                    open_params,
                    calibration,
                    CENTER,
                    SEARCH_RADIUS_PX,
                ).matched_snr
                for _ in range(2_000)
            ]
        )
        assert 0.05 < float(np.mean(np.isnan(peaks))) < 0.15  # no peak within the radius
        peaks = np.nan_to_num(peaks, nan=0.0)
        assert self.candidates() == 22_848  # 16 times 1,428 pixels
        assert filters == 3
        assert float(np.median(peaks)) == pytest.approx(3.45, abs=0.1)  # extreme-value statistics
        assert float(peaks.max()) < params.min_snr  # no starless frame reaches 6
        for threshold in (4.5, 5.0):
            # The counts carry their own noise: allow 3 sigma of a binomial count above the bound.
            count = int(np.sum(peaks >= threshold))
            limit = len(peaks) * self.bound(threshold, filters)
            assert count <= limit + 3 * math.sqrt(limit), threshold
        # A starless burst: the median of 50 frames sits near the median peak, far below 10.
        bursts = peaks.reshape(-1, BURST_FRAMES)
        assert float(np.median(bursts, axis=1).max()) < 4.0

    def test_the_bound_makes_a_false_detection_negligible_at_the_threshold(self) -> None:
        """At 10, a frame reaches the threshold with a chance below 6e-19, and a burst needs 25.

        A burst detects only when at least 25 of its 50 independent frames reach 10, so its
        chance is below `C(50, 25) p^25`, which underflows any float: about 10^-443. Two such
        bursts in a row start measure.
        """
        per_frame = self.bound(DETECT_SNR, 3)
        assert per_frame < 6e-19
        log10_burst = math.log10(math.comb(BURST_FRAMES, 25)) + 25 * math.log10(per_frame)
        assert log10_burst < -400
