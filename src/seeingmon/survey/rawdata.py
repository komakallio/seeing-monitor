"""Raw frame data in ADC counts, and the robust statistics that calibration needs.

A `Frame` holds the ADC value in the high bits of a 16-bit container, or the top 8 bits. The
survey steps work in native counts, the values that the ADC produced. The dark library works in
whole counts, so `native_u16` keeps them as unsigned integers.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from seeingmon.frames import Frame

_SAMPLE_SEED = 0x5EED


def native_counts(frame: Frame) -> npt.NDArray[np.float32]:
    """The frame in ADC counts: the high bits of a 16-bit container, or an 8-bit value scaled up."""
    scale = np.float32(2.0 ** (frame.adc_bits - frame.pixel_format.value))
    return np.asarray(frame.data.astype(np.float32) * scale, dtype=np.float32)


def native_u16(frame: Frame) -> npt.NDArray[np.uint16]:
    """The frame in whole ADC counts, as `uint16`.

    A 16-bit frame holds the ADC value in its high bits, so the shift is exact. An 8-bit frame
    has lost the low bits, and the function scales it up.
    """
    shift = frame.adc_bits - frame.pixel_format.value
    data = frame.data.astype(np.uint16)
    if shift >= 0:
        return np.asarray(data << np.uint16(shift), dtype=np.uint16)
    return np.asarray(data >> np.uint16(-shift), dtype=np.uint16)


def robust_level(values: npt.ArrayLike, *, max_samples: int = 1_000_000) -> tuple[float, float]:
    """The median of the values and a robust sigma (1.4826 times the median absolute deviation).

    A large input gives its statistics from `max_samples` values at fixed random positions, so
    the result is the same on every call, and no regular pattern of the sensor aliases with the
    sample.
    """
    flat = np.asarray(values).reshape(-1)
    if flat.size == 0:
        raise ValueError("no values")
    if flat.size > max_samples:
        rng = np.random.default_rng(_SAMPLE_SEED)
        flat = flat[rng.integers(0, flat.size, max_samples)]
    sample = flat.astype(np.float32, copy=False)
    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample - np.float32(median))))
    return median, 1.4826 * mad


def clipped_mean(
    values: npt.ArrayLike,
    *,
    sigma: float = 3.0,
    iterations: int = 4,
    max_samples: int = 1_000_000,
) -> float:
    """The mean of the values after clipping the outliers: a level that is exact below one count.

    The median of whole counts is exact only to half a count, which is a large error for a dark
    level of a few counts. The function clips at `sigma` robust sigmas around the median, repeats
    `iterations` times, and averages what remains. Like `robust_level`, it samples a large input
    at fixed random positions.
    """
    flat = np.asarray(values).reshape(-1)
    if flat.size == 0:
        raise ValueError("no values")
    if flat.size > max_samples:
        rng = np.random.default_rng(_SAMPLE_SEED)
        flat = flat[rng.integers(0, flat.size, max_samples)]
    sample = flat.astype(np.float64)
    center = float(np.median(sample))
    spread = 1.4826 * float(np.median(np.abs(sample - center)))
    kept = sample
    for _ in range(iterations):
        if spread <= 0.0:
            break
        kept = sample[np.abs(sample - center) <= sigma * spread]
        if kept.size == 0:
            return center
        center = float(kept.mean())
        spread = float(kept.std())
    return float(kept.mean()) if kept.size else center
