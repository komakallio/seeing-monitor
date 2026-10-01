"""The per-frame kernel against analytic truth.

The stars are Gaussians whose pixels hold the exact integral, so the true position is known to
rounding. The tolerance sits next to each assertion.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

from seeingmon.fastpath.kernel import (
    FLAG_EDGE,
    FLAG_HOT_PIXEL,
    FLAG_NO_STAR,
    FLAG_SATURATED,
    PHASES,
    FrameCalibration,
    KernelParams,
    measure_frame,
    measure_stack,
)
from tests.fastpath.helpers import (
    POLARIS_ELECTRONS_2MS,
    box_integrated_gaussian,
    digitize,
)

SHAPE = (128, 128)
CALIBRATION = FrameCalibration.for_container(
    adc_bits=12, container_bits=16, e_per_dn=3.5 / 16, pixel_var_e2=2.65**2
)
PARAMS = KernelParams(aperture_diameter_px=16.0)


CALIBRATION16 = FrameCalibration.for_container(
    adc_bits=16, container_bits=16, e_per_dn=3.5, pixel_var_e2=2.65**2
)


def star16(
    x: float, y: float, flux: float = 300_000.0, sigma: float = 1.0
) -> npt.NDArray[np.uint8] | npt.NDArray[np.uint16]:
    """A noise-free frame of a 16-bit ADC, so that rounding adds under 1e-4 px of noise."""
    return digitize(box_integrated_gaussian(SHAPE, x, y, sigma, flux * 3.5), adc_bits=16)


def star(
    x: float, y: float, flux: float = 12_000.0, sigma: float = 1.0
) -> npt.NDArray[np.uint8] | npt.NDArray[np.uint16]:
    """A noise-free frame in 16-bit container counts. The flux is in ADC counts of a 12-bit ADC."""
    electrons = box_integrated_gaussian(SHAPE, x, y, sigma, flux * 3.5)
    return digitize(electrons)


class TestCentroid:
    def test_recovers_the_position_over_the_sub_pixel_phases(self) -> None:
        """A noise-free Gaussian star: the error stays under 0.0005 px at every phase.

        The aperture position is quantized to 1/16 px, and the aperture follows the star, so
        the error is the leakage of the aperture times half a step. A Gaussian of 1 px sigma
        leaks nothing, so the result is far below the 0.0006 px that a star with wings can cost.
        """
        worst = 0.0
        for fx in np.linspace(-0.5, 0.45, 11):
            for fy in np.linspace(-0.45, 0.5, 11):
                x, y = 64.0 + fx, 63.0 + fy
                m = measure_frame(star16(x, y), 0, 0, PARAMS, CALIBRATION16, guess=(63.5, 63.5))
                assert m.found
                worst = max(worst, abs(m.x - x), abs(m.y - y))
        assert worst < 5e-4

    def test_a_finer_phase_table_does_not_matter_for_a_converged_star(self) -> None:
        """The same star, found from two different first guesses, gives the same centroid."""
        data = star(70.3, 55.7)
        a = measure_frame(data, 0, 0, PARAMS, CALIBRATION, guess=(70.0, 56.0))
        b = measure_frame(data, 0, 0, PARAMS, CALIBRATION)  # the brightest-patch search
        assert abs(a.x - b.x) < 2e-3  # 0.002 px: both end within the quantization of the true spot
        assert abs(a.y - b.y) < 2e-3

    def test_reports_sensor_coordinates(self) -> None:
        """The ROI origin is added, so a ROI move never looks like motion."""
        data = star(64.25, 64.75)
        a = measure_frame(data, 1000, 2000, PARAMS, CALIBRATION)
        b = measure_frame(data, 4100, 2800, PARAMS, CALIBRATION)
        assert a.x == pytest.approx(1064.25, abs=1e-3)
        assert b.y == pytest.approx(2864.75, abs=1e-3)
        assert (b.x - a.x, b.y - a.y) == pytest.approx((3100.0, 800.0), abs=1e-9)

    def test_a_guess_in_sensor_coordinates_follows_a_roi_move(self) -> None:
        """A guess that stays at the same sky position works after the ROI moves."""
        first = measure_frame(star(64.0, 64.0), 1000, 2000, PARAMS, CALIBRATION)
        # The ROI moves by (-20, +10), so the star appears at (84, 54) in ROI pixels.
        moved = measure_frame(
            star(84.0, 54.0), 980, 2010, PARAMS, CALIBRATION, guess=(first.x, first.y)
        )
        assert moved.x == pytest.approx(first.x, abs=1e-3)
        assert moved.y == pytest.approx(first.y, abs=1e-3)

    def test_the_background_comes_from_the_border_median(self) -> None:
        data = star(64, 64)
        data[0:4, :] = 0  # the median ignores a dark band and a few hot pixels in the ring
        data[100, 2] = 60_000
        data[5, 126] = 65_000
        m = measure_frame(data, 0, 0, PARAMS, CALIBRATION, guess=(64.0, 64.0))
        assert m.bg_dn == pytest.approx(480.0)  # the black level of 30 ADU in 16-bit counts
        assert m.found

    @pytest.mark.parametrize(("sigma", "expected_width"), [(1.0, 1.0), (1.5, 1.5)])
    def test_second_moment_width_of_a_gaussian(self, sigma: float, expected_width: float) -> None:
        """The width is the second moment inside the aperture: sigma, within 1% for sigma <= 1.5.

        A box-integrated Gaussian adds the pixel's 1/12 to the variance, so the width is
        `sqrt(sigma^2 + 1/12)`, and the aperture's truncation changes it by 0.1% at most.
        """
        m = measure_frame(star(64.3, 64.6, sigma=sigma), 0, 0, PARAMS, CALIBRATION)
        expected = math.sqrt(expected_width**2 + 1.0 / 12.0)
        assert m.width_x == pytest.approx(expected, rel=0.01)
        assert m.width_y == pytest.approx(expected, rel=0.01)

    def test_flux_is_the_aperture_sum_minus_the_background(self) -> None:
        """The flux in container counts is the star's counts: 12,000 ADC counts x 16."""
        m = measure_frame(star(64.0, 64.0), 0, 0, PARAMS, CALIBRATION)
        assert m.flux_dn == pytest.approx(12_000.0 * 16.0, rel=2e-3)

    def test_the_peak_is_the_brightest_pixel_of_the_aperture_box(self) -> None:
        data = star(64.0, 64.0)
        m = measure_frame(data, 0, 0, PARAMS, CALIBRATION)
        assert m.peak_dn == float(data[60:68, 60:68].max())


class TestNoise:
    def test_polaris_like_flux_gives_a_centroid_noise_below_a_fiftieth_of_a_pixel(self) -> None:
        """Photon and read noise of Polaris at 2 ms (14,000 e-): under 0.02 px.

        A sample of 1,500 frames gives the standard deviation to 2%. The model comes within 15%,
        because the model's flux is the measured flux and the noise of the weights is ignored.
        """
        rng = np.random.default_rng(3)
        mean = box_integrated_gaussian(SHAPE, 64.3, 63.8, 1.0, POLARIS_ELECTRONS_2MS)
        xs, models = [], []
        for _ in range(1500):
            m = measure_frame(
                digitize(mean, rng=rng), 0, 0, PARAMS, CALIBRATION, guess=(64.0, 64.0)
            )
            xs.append(m.x)
            models.append(m.noise_var_x)
        measured = float(np.std(xs))
        assert measured < 0.02
        assert math.sqrt(float(np.mean(models))) == pytest.approx(measured, rel=0.15)

    def test_a_faint_star_is_noisier_and_the_model_follows(self) -> None:
        rng = np.random.default_rng(4)
        mean = box_integrated_gaussian(SHAPE, 64.3, 63.8, 1.0, 1_000.0)
        xs, models = [], []
        for _ in range(800):
            m = measure_frame(digitize(mean, rng=rng), 0, 0, PARAMS, CALIBRATION, guess=(64, 64))
            assert m.found
            xs.append(m.x)
            models.append(m.noise_var_x)
        assert float(np.std(xs)) > 0.02
        assert math.sqrt(float(np.mean(models))) == pytest.approx(float(np.std(xs)), rel=0.2)


class TestGainAtTruncation:
    """A fixed aperture leaks light and responds with a gain below 1; a recentered one does not."""

    @staticmethod
    def airy_like_star(dx: float) -> np.ndarray:
        """A star with diffraction wings, from the Gaussian mixture of the simulator, at `dx`."""
        from seeingmon.drivers.sim.optics import MixturePsf, PsfConfig
        from seeingmon.drivers.sim.params import SimParams

        params = SimParams.reference("bin1")
        mixture = MixturePsf(params, PsfConfig(mode="gaussian"))
        weights, sigmas = mixture.components(1.0)  # r0 = 1 m: nearly diffraction limited
        stamp = mixture.stamps(
            np.asarray([dx]), np.asarray([0.0]), weights, sigmas, size_px=SHAPE[0]
        )[0].astype(np.float64)
        return digitize(stamp * 12_000.0 * 3.5)

    def gain(self, iterations: int, diameter: float) -> float:
        """The response to a shift of 0.4 px, from the centroids at -0.2 and +0.2 px."""
        params = KernelParams(
            aperture_diameter_px=diameter, recenter_iterations=iterations, spike_ratio=None
        )
        centre = SHAPE[0] // 2
        guess = (float(centre), float(centre))  # the aperture starts at the pixel center
        low = measure_frame(self.airy_like_star(-0.2), 0, 0, params, CALIBRATION, guess=guess)
        high = measure_frame(self.airy_like_star(0.2), 0, 0, params, CALIBRATION, guess=guess)
        return (high.x - low.x) / 0.4

    @pytest.mark.parametrize(
        ("diameter", "low", "high"), [(11.0, 0.96, 0.99), (16.0, 0.975, 0.995)]
    )
    def test_a_fixed_aperture_has_a_gain_below_one(
        self, diameter: float, low: float, high: float
    ) -> None:
        """Without recentering, the aperture leaks: 1 - pi R^2 I(R) / F, about 0.976 at 11 px.

        The research notes give 0.976 for 11 pixels and 0.983 for 15 (the Airy wings).
        """
        assert low < self.gain(0, diameter) < high

    def test_a_recentered_aperture_has_a_gain_of_one(self) -> None:
        """The aperture follows the star, so the leak does not bias the shift: 1.000 +- 0.002."""
        assert self.gain(2, 16.0) == pytest.approx(1.0, abs=0.002)


class TestFlagsAndDepth:
    def test_saturation_flag_sets_at_98_percent_of_the_adc_full_scale(self) -> None:
        full_scale = 4095 * 16  # a 12-bit ADC in a 16-bit container
        assert CALIBRATION.full_scale_dn == full_scale
        assert CALIBRATION.saturation_dn == pytest.approx(0.98 * full_scale)
        below = star(64.0, 64.0, flux=8_000.0)  # a peak of 1,300 ADC counts, a third of the level
        assert not measure_frame(below, 0, 0, PARAMS, CALIBRATION).flags & FLAG_SATURATED
        saturated = np.minimum(star(64.0, 64.0, flux=200_000.0), full_scale)
        m = measure_frame(saturated, 0, 0, PARAMS, CALIBRATION)
        assert m.flags & FLAG_SATURATED
        assert m.found

    def test_the_level_follows_the_adc_depth_and_the_container(self) -> None:
        cases = [(12, 16, 65_520), (14, 16, 65_532), (16, 16, 65_535), (8, 8, 255), (14, 8, 255)]
        for adc_bits, container_bits, full_scale in cases:
            calibration = FrameCalibration.for_container(
                adc_bits=adc_bits, container_bits=container_bits
            )
            assert calibration.full_scale_dn == full_scale
            assert calibration.saturation_dn == pytest.approx(0.98 * full_scale)

    def test_a_star_near_the_roi_edge_carries_the_edge_flag(self) -> None:
        for x, y in ((8.0, 64.0), (64.0, 5.0), (120.0, 64.0), (64.0, 124.0)):
            m = measure_frame(star(x, y), 0, 0, PARAMS, CALIBRATION)
            assert m.found
            assert m.flags & FLAG_EDGE, (x, y)
        centered = measure_frame(star(64.0, 64.0), 0, 0, PARAMS, CALIBRATION)
        assert not centered.flags & FLAG_EDGE

    def test_a_blank_frame_has_no_star(self) -> None:
        rng = np.random.default_rng(1)
        data = digitize(np.zeros(SHAPE), rng=rng)
        m = measure_frame(data, 0, 0, PARAMS, CALIBRATION)
        assert not m.found
        assert m.flags & FLAG_NO_STAR
        assert math.isnan(m.x)
        assert math.isnan(m.flux_dn)
        assert m.bg_dn == pytest.approx(480.0, abs=16.0)

    def test_an_isolated_bright_pixel_carries_the_hot_pixel_flag(self) -> None:
        electrons = box_integrated_gaussian(SHAPE, 64.0, 64.0, 1.2, 8_000.0)
        electrons[70, 68] += 12_000.0  # a hot pixel inside the aperture, brighter than the star
        data = digitize(electrons)
        assert measure_frame(data, 0, 0, PARAMS, CALIBRATION, guess=(64, 64)).flags & FLAG_HOT_PIXEL
        assert not measure_frame(data, 0, 0, KernelParams(spike_ratio=None), CALIBRATION).flags & (
            FLAG_HOT_PIXEL
        )
        clean = digitize(box_integrated_gaussian(SHAPE, 64.0, 64.0, 1.2, 8_000.0))
        assert not measure_frame(clean, 0, 0, PARAMS, CALIBRATION).flags & FLAG_HOT_PIXEL

    def test_eight_bit_and_sixteen_bit_frames_agree(self) -> None:
        """The same star in an 8-bit and a 16-bit container gives the same centroid.

        The 8-bit frame holds the top 8 bits of the 12-bit ADC value, so its counts are coarser,
        and the centroid differs by the quantization, which is below 0.01 px for this star. The
        flux differs more: the background of the 8-bit frame is a whole number of counts, so the
        aperture sum can be off by up to half a count per pixel (100 counts of 750 here).
        """
        electrons = box_integrated_gaussian(SHAPE, 64.37, 63.61, 1.1, 42_000.0)
        wide = digitize(electrons, container_bits=16)
        narrow = digitize(electrons, container_bits=8)
        assert narrow.dtype == np.uint8
        assert wide.dtype == np.uint16
        calibration8 = FrameCalibration.for_container(adc_bits=12, container_bits=8)
        a = measure_frame(wide, 0, 0, PARAMS, CALIBRATION, guess=(64, 64))
        b = measure_frame(narrow, 0, 0, PARAMS, calibration8, guess=(64, 64))
        assert a.found
        assert b.found
        assert abs(a.x - b.x) < 0.01
        assert abs(a.y - b.y) < 0.01
        assert b.flux_dn * 256.0 == pytest.approx(
            a.flux_dn, rel=0.1
        )  # 16-bit counts are 256 x 8-bit

    def test_a_frame_smaller_than_the_aperture_box_still_works(self) -> None:
        for shape, edge in (((8, 8), True), ((16, 16), True), ((24, 32), False)):
            electrons = box_integrated_gaussian(shape, shape[1] / 2, shape[0] / 2, 1.0, 40_000.0)
            m = measure_frame(digitize(electrons), 0, 0, PARAMS, CALIBRATION)
            assert m.found
            assert bool(m.flags & FLAG_EDGE) == edge  # an edge flag when the aperture does not fit
            assert m.x == pytest.approx(shape[1] / 2, abs=0.05)


class TestStack:
    def test_the_stack_form_matches_the_single_frame_form(self) -> None:
        rng = np.random.default_rng(5)
        mean = [
            box_integrated_gaussian(
                SHAPE, 64 + 0.4 * math.sin(i / 3), 64 + 0.3 * math.cos(i / 5), 1.0, 14_000.0
            )
            for i in range(40)
        ]
        stack = np.asarray(np.stack([digitize(m, rng=rng) for m in mean]), dtype=np.uint16)
        result = measure_stack(stack, 500, 600, PARAMS, CALIBRATION)
        assert len(result) == 40
        guess: tuple[float, float] | None = None
        for index in range(40):
            single = measure_frame(stack[index], 500, 600, PARAMS, CALIBRATION, guess)
            assert tuple(result.row(index)) == pytest.approx(tuple(single), nan_ok=True, rel=1e-12)
            guess = (single.x, single.y)

    def test_a_lost_star_sends_the_next_frame_back_to_the_search(self) -> None:
        rng = np.random.default_rng(6)
        good = digitize(box_integrated_gaussian(SHAPE, 90.0, 40.0, 1.0, 14_000.0), rng=rng)
        blank = digitize(np.zeros(SHAPE), rng=rng)
        result = measure_stack(
            np.asarray(np.stack([good, blank, good]), dtype=np.uint16),
            0,
            0,
            PARAMS,
            CALIBRATION,
            guess=(64, 64),
        )
        assert list(result.found) == [True, False, True]
        assert result.x[2] == pytest.approx(90.0, abs=0.05)

    def test_rejects_a_two_dimensional_array_and_accepts_an_empty_stack(self) -> None:
        with pytest.raises(ValueError, match="frames, rows, columns"):
            measure_stack(star(64, 64), 0, 0, PARAMS, CALIBRATION)
        assert (
            len(measure_stack(np.empty((0, 128, 128), dtype=np.uint16), 0, 0, PARAMS, CALIBRATION))
            == 0
        )


class TestParameters:
    def test_the_aperture_box_and_the_weights(self) -> None:
        params = KernelParams(aperture_diameter_px=16.0)
        assert params.radius_px == 8.0
        assert params.box_px == 19
        assert params.area_px2 == pytest.approx(math.pi * 8.0**2, rel=0.01)
        assert params.second_moment_px4 == pytest.approx(math.pi * 8.0**4 / 4, rel=0.02)
        assert PHASES == 16

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"aperture_diameter_px": 2.0},
            {"recenter_iterations": -1},
            {"border_px": 0},
            {"min_snr": -1.0},
            {"spike_ratio": 1.5},
        ],
    )
    def test_rejects_invalid_settings(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError, match=r"aperture|recenter|border|min_snr|spike"):
            KernelParams(**kwargs)  # type: ignore[arg-type]

    def test_rejects_an_invalid_container(self) -> None:
        with pytest.raises(ValueError, match="container_bits"):
            FrameCalibration.for_container(adc_bits=12, container_bits=12)
        with pytest.raises(ValueError, match="saturation_fraction"):
            FrameCalibration.for_container(adc_bits=12, container_bits=16, saturation_fraction=0.0)
