"""Raw frame data in native counts, and the robust levels that calibration uses."""

from __future__ import annotations

import numpy as np
import pytest

from seeingmon.frames import Frame, PixelFormat, Roi, TimeQuality
from seeingmon.survey import rawdata


def frame_with(values: np.ndarray, *, adc_bits: int = 14) -> Frame:
    return Frame(
        data=values,
        stream_id=1,
        seq=0,
        t_arrival_ns=0,
        t_utc_ns=0,
        t_err_ns=0,
        t_quality=TimeQuality.EXACT,
        dropped_before=0,
        exposure_us=1000,
        gain=120,
        mode="bin2",
        roi=Roi(0, 0, values.shape[1], values.shape[0]),
        adc_bits=adc_bits,
    )


def test_a_16_bit_frame_gives_its_adc_value_from_the_high_bits() -> None:
    native = np.array([[0, 1, 120, 16383]], dtype=np.uint16)
    frame = frame_with((native << 2).astype(np.uint16))  # 14-bit values in the high bits
    assert frame.pixel_format is PixelFormat.RAW16
    np.testing.assert_array_equal(rawdata.native_u16(frame), native)
    np.testing.assert_array_equal(rawdata.native_counts(frame), native.astype(np.float32))
    assert rawdata.native_counts(frame).dtype == np.float32


def test_an_8_bit_frame_scales_up_to_the_adc_range() -> None:
    top = np.array([[0, 1, 200, 255]], dtype=np.uint8)
    frame = frame_with(top, adc_bits=14)
    np.testing.assert_array_equal(rawdata.native_u16(frame), top.astype(np.uint16) << 6)
    np.testing.assert_array_equal(rawdata.native_counts(frame), top.astype(np.float32) * 64.0)


def test_the_robust_level_ignores_outliers() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(100.0, 4.0, 200_000)
    values[:2000] = 5000.0
    median, sigma = rawdata.robust_level(values)
    assert median == pytest.approx(100.0, abs=0.1)
    assert sigma == pytest.approx(4.0, rel=0.03)


def test_a_large_input_gives_the_same_answer_every_time() -> None:
    rng = np.random.default_rng(2)
    values = rng.normal(100.0, 4.0, (1500, 1500)).astype(np.float32)
    first = rawdata.robust_level(values, max_samples=100_000)
    assert rawdata.robust_level(values, max_samples=100_000) == first
    assert rawdata.clipped_mean(values, max_samples=100_000) == rawdata.clipped_mean(
        values, max_samples=100_000
    )
    assert first[0] == pytest.approx(100.0, abs=0.1)


def test_the_clipped_mean_resolves_less_than_one_count() -> None:
    """Whole counts with a mean of 147.3: the median cannot say, and the clipped mean can."""
    rng = np.random.default_rng(3)
    counts = np.rint(rng.normal(147.3, 2.5, 400_000))
    counts[:500] += 900.0  # hot pixels and stars
    assert rawdata.clipped_mean(counts) == pytest.approx(147.3, abs=0.02)


def test_a_constant_image_and_empty_input_are_handled() -> None:
    assert rawdata.clipped_mean(np.full((10, 10), 7, dtype=np.uint16)) == 7.0
    assert rawdata.robust_level(np.full((10, 10), 7, dtype=np.uint16)) == (7.0, 0.0)
    with pytest.raises(ValueError, match="no values"):
        rawdata.robust_level(np.zeros(0))
    with pytest.raises(ValueError, match="no values"):
        rawdata.clipped_mean(np.zeros(0))
