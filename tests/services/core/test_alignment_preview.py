"""The live-view image: shrinking, the asinh stretch, the JPEG, the histogram, and saturation."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pytest

from seeingmon.services.core.alignment.preview import (
    block_mean,
    encode_jpeg,
    frame_saturation_dn,
    histogram_counts,
    make_preview,
    saturated_fraction,
    saturation_in_frame_units,
    shrink_factor,
    stretch_asinh,
)
from seeingmon.services.web.contract import JPEG_MAGIC

PIL = pytest.importorskip("PIL.Image", reason="the preview needs Pillow")


def sky_with_a_star(height: int = 96, width: int = 128, *, seed: int = 1) -> np.ndarray:
    """A noisy 16-bit sky around 1000 counts, with one bright star."""
    rng = np.random.default_rng(seed)
    data = rng.normal(1000.0, 10.0, (height, width))
    yy, xx = np.mgrid[0:height, 0:width]
    data += 30000.0 * np.exp(-((xx - 40) ** 2 + (yy - 30) ** 2) / (2 * 1.5**2))
    return np.clip(data, 0, 65535).astype(np.uint16)


class TestShrinking:
    @pytest.mark.parametrize(
        ("height", "width", "limit", "factor"),
        [
            (2822, 4144, 1_000_000, 4),  # the bin2 frame of the reference camera
            (480, 640, 1_000_000, 1),
            (1000, 1000, 250_000, 2),
            (1000, 1000, 249_999, 3),
            (3, 3, 1, 3),
        ],
    )
    def test_the_factor_is_the_smallest_that_fits(
        self, height: int, width: int, limit: int, factor: int
    ) -> None:
        k = shrink_factor(height, width, limit)
        assert k == factor
        assert (height // k) * (width // k) <= max(limit, 1) or k == min(height, width)

    def test_a_non_positive_input_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            shrink_factor(0, 10, 100)

    def test_the_block_mean_matches_a_plain_average(self) -> None:
        data = np.arange(7 * 9, dtype=np.uint16).reshape(7, 9)
        mean = block_mean(data, 2)
        assert mean.shape == (3, 4)  # the last row and column do not fill a block
        assert mean[1, 2] == pytest.approx(data[2:4, 4:6].mean())
        assert block_mean(data, 1).dtype == np.float32


class TestStretch:
    def test_the_sky_is_dark_gray_and_the_star_is_white(self) -> None:
        image = block_mean(sky_with_a_star(), 1)
        stretched = stretch_asinh(image)
        assert stretched.dtype == np.uint8
        assert 40 <= int(np.median(stretched)) <= 62  # the sky sits at about 20% gray
        assert stretched[30, 40] == 255

    def test_faint_structure_stays_visible(self) -> None:
        sky = np.full((64, 64), 500.0, dtype=np.float32)
        sky += np.random.default_rng(3).normal(0, 2.0, sky.shape).astype(np.float32)
        sky[20:23, 20:23] += 20.0  # a faint star, ten sigmas above the sky
        stretched = stretch_asinh(sky)
        assert stretched[21, 21] > 200
        assert stretched[21, 21] > int(stretched[40, 40]) + 40

    def test_the_curve_never_decreases_with_brightness(self) -> None:
        ramp = np.linspace(0, 5000, 1000, dtype=np.float32).reshape(10, 100)
        values = stretch_asinh(ramp).ravel()
        assert np.all(np.diff(values.astype(int)) >= 0)

    def test_a_flat_image_does_not_divide_by_zero(self) -> None:
        flat = np.full((16, 16), 700.0, dtype=np.float32)
        stretched = stretch_asinh(flat)
        assert int(stretched.min()) == int(stretched.max())


class TestJpeg:
    def test_the_bytes_are_a_jpeg_of_the_same_size(self) -> None:
        image = stretch_asinh(block_mean(sky_with_a_star(), 1))
        jpeg = encode_jpeg(image, 80)
        assert jpeg.startswith(JPEG_MAGIC)
        decoded = PIL.open(io.BytesIO(jpeg))
        assert decoded.size == (128, 96)
        assert decoded.mode == "L"

    def test_a_better_quality_makes_a_larger_file(self) -> None:
        image = stretch_asinh(block_mean(sky_with_a_star(), 1))
        assert len(encode_jpeg(image, 30)) < len(encode_jpeg(image, 90))

    def test_the_preview_is_shrunk_under_the_pixel_limit(self) -> None:
        data = sky_with_a_star(300, 400)
        preview = make_preview(data, max_pixels=10_000, quality=80)
        assert preview.factor == 4
        assert (preview.width_px, preview.height_px) == (100, 75)
        assert preview.jpeg.startswith(JPEG_MAGIC)

    def test_a_frame_under_the_limit_keeps_its_size(self) -> None:
        preview = make_preview(sky_with_a_star(), max_pixels=1_000_000, quality=80)
        assert (preview.factor, preview.width_px, preview.height_px) == (1, 128, 96)


class TestTheCalibrationHook:
    def test_without_a_calibration_the_image_goes_straight_to_the_stretch(self) -> None:
        data = sky_with_a_star(300, 400)
        preview = make_preview(data, max_pixels=10_000, quality=80)
        assert preview.jpeg == encode_jpeg(stretch_asinh(block_mean(data, 4)), 80)

    def test_the_calibration_gets_the_frame_the_shrunk_image_and_the_factor(self) -> None:
        data = sky_with_a_star(300, 400)
        seen: list[tuple[bool, tuple[int, ...], np.dtype[Any], int]] = []

        def spy(frame: Any, image: Any, factor: int) -> Any:
            seen.append((frame is data, image.shape, image.dtype, factor))
            return image

        preview = make_preview(data, max_pixels=10_000, quality=80, calibration=spy)
        assert seen == [(True, (75, 100), np.dtype(np.float32), 4)]
        assert preview.jpeg == make_preview(data, max_pixels=10_000, quality=80).jpeg

    def test_the_stretch_takes_the_image_that_the_calibration_returns(self) -> None:
        data = sky_with_a_star(300, 400)

        def invert(frame: Any, image: Any, factor: int) -> Any:
            return np.asarray(image.max() - image, dtype=np.float32)

        preview = make_preview(data, max_pixels=10_000, quality=80, calibration=invert)
        shrunk = block_mean(data, 4)
        assert preview.jpeg == encode_jpeg(stretch_asinh(shrunk.max() - shrunk), 80)
        assert preview.jpeg != make_preview(data, max_pixels=10_000, quality=80).jpeg
        assert (preview.width_px, preview.height_px, preview.factor) == (100, 75, 4)


class TestSaturation:
    @pytest.mark.parametrize(
        ("native", "adc_bits", "pixel_bits", "expected"),
        [
            (16383.0, 14, 16, 65532.0),  # a 16-bit container holds the ADC value in the high bits
            (4095.0, 12, 16, 65520.0),
            (16383.0, 14, 8, 255.0),  # the top eight bits: the limit of the data type
            (4095.0, 12, 8, 255.0),
        ],
    )
    def test_the_level_follows_the_container(
        self, native: float, adc_bits: int, pixel_bits: int, expected: float
    ) -> None:
        assert saturation_in_frame_units(native, adc_bits, pixel_bits) == pytest.approx(expected)

    def test_the_level_of_a_real_frame_follows_its_format(self) -> None:
        from seeingmon.frames import Frame, FrameFlag, Roi, TimeQuality

        data = np.zeros((4, 4), dtype=np.uint16)
        frame = Frame(
            data, 1, 0, 0, 0, 0, TimeQuality.EXACT, 0, 1000, 0, "bin2", Roi(0, 0, 4, 4), 14,
            None, FrameFlag.NONE,
        )  # fmt: skip
        assert frame_saturation_dn(frame, 16383.0) == pytest.approx(65532.0)

    def test_the_fraction_counts_pixels_at_or_above_the_level(self) -> None:
        data = np.zeros((10, 10), dtype=np.uint16)
        data[0, :5] = 65000
        assert saturated_fraction(data, 64000.0) == pytest.approx(0.05)
        assert saturated_fraction(data, 70000.0) == 0.0


class TestHistogram:
    def test_the_counts_cover_the_sampled_pixels(self) -> None:
        data = sky_with_a_star(100, 100)
        counts = histogram_counts(data, 32, 65532.0)
        assert len(counts) == 32
        assert sum(counts) == 50 * 50  # every second pixel in each direction
        assert counts[0] > 0.9 * sum(counts)  # the sky sits in the lowest bin

    def test_a_value_above_the_maximum_counts_in_the_last_bin(self) -> None:
        data = np.full((4, 4), 60000, dtype=np.uint16)
        counts = histogram_counts(data, 8, 30000.0)
        assert counts[-1] == 4
        assert sum(counts[:-1]) == 0

    def test_a_bad_request_is_refused(self) -> None:
        data = np.zeros((4, 4), dtype=np.uint16)
        with pytest.raises(ValueError, match="positive"):
            histogram_counts(data, 0, 100.0)
        with pytest.raises(ValueError, match="positive"):
            histogram_counts(data, 8, 0.0)
