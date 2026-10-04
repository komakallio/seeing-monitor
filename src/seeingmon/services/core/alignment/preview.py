"""The live-view image: a small, stretched JPEG of an alignment frame, and the numbers beside it.

An alignment frame is a full bin2 frame (12 megapixels on the reference camera). The helper cannot
send that to a phone, so it makes a preview in five steps:

1. **Shrink.** Average blocks of `k` by `k` pixels, with the smallest `k` that brings the image
   under the pixel limit. The average keeps the noise down, and a star stays visible.
2. **Calibrate.** When the caller passes a calibration (`PreviewCalibration`, which
   `seeingmon.services.core.alignment.calibration` makes), it takes the dark level, the vignetting,
   and the dust shadows out of the shrunk image, so that the stretch shows the sky and not the
   optics. Without one, the image goes on as it is.
3. **Stretch.** Subtract the median, scale by the bright end of the sky and the stars, and apply
   `asinh`. The curve is linear for faint pixels and logarithmic for bright ones, so the faint stars
   and the core of Polaris show in one image.
4. **Encode.** Pillow writes a grayscale JPEG.
5. **Measure.** The histogram of the frame (with the counts to draw on a log axis) and the share of
   saturated pixels come from the full frame, not the preview, because one saturated pixel decides
   the exposure.

All functions take plain arrays and numbers, so a test needs no camera. Pillow loads when the first
JPEG is encoded.
"""

from __future__ import annotations

import io
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from seeingmon.frames import Frame, FrameData

# The stretch: pixels above the median by `WHITE_SIGMAS` robust sigmas of the sky (and at least
# `MIN_RANGE_DN`) map to white. The factor `ASINH_GAIN` sets where the linear part ends.
ASINH_GAIN = 40.0
MIN_RANGE_DN = 8.0
WHITE_PERCENTILE = 99.95
MAD_TO_SIGMA = 1.4826
HISTOGRAM_STRIDE = 2

# A step between the shrink and the stretch. `calibration(data, image, factor)` gets the frame (it
# must leave it as it is), the shrunk image of the frame (it may change that in place), and the
# shrink factor, and it returns the calibrated image. It must not raise: a preview never fails
# because of its calibration, so a step that cannot run returns the image as it got it.
PreviewCalibration = Callable[[FrameData, npt.NDArray[np.float32], int], npt.NDArray[np.float32]]


@dataclass(frozen=True, slots=True)
class Preview:
    """A JPEG and its size. `factor` is the shrink factor from the frame to the image."""

    jpeg: bytes
    width_px: int
    height_px: int
    factor: int


def shrink_factor(height: int, width: int, max_pixels: int) -> int:
    """The smallest block size `k` for which `(height // k) * (width // k)` fits `max_pixels`."""
    if height < 1 or width < 1 or max_pixels < 1:
        raise ValueError("the frame and the limit must be positive")
    k = max(1, math.ceil(math.sqrt(height * width / max_pixels)))
    while k > 1 and (height // k) * (width // k) < 1:
        k -= 1
    return k


def block_mean(data: FrameData, k: int) -> npt.NDArray[np.float32]:
    """Average blocks of `k` by `k` pixels. The edge that does not fill a block is dropped."""
    if k == 1:
        return data.astype(np.float32)
    height, width = data.shape
    rows, columns = height // k, width // k
    blocks = data[: rows * k, : columns * k].reshape(rows, k, columns, k)
    return np.asarray(
        blocks.sum(axis=(1, 3), dtype=np.uint32) / np.float32(k * k), dtype=np.float32
    )


def stretch_asinh(image: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    """Map an image to 8 bits with an `asinh` curve between the median and a bright percentile."""
    black = float(np.median(image))
    spread = MAD_TO_SIGMA * float(np.median(np.abs(image - black)))
    white = float(np.percentile(image, WHITE_PERCENTILE))
    span = max(white - black, MIN_RANGE_DN, 3.0 * spread)
    scaled = np.clip((image - np.float32(black)) / np.float32(span), 0.0, 1.0)
    curve = np.arcsinh(np.float32(ASINH_GAIN) * scaled) / np.float32(math.asinh(ASINH_GAIN))
    return np.asarray(np.clip(curve * 255.0 + 0.5, 0, 255), dtype=np.uint8)


def encode_jpeg(image: npt.NDArray[np.uint8], quality: int) -> bytes:
    """Encode a 2-D 8-bit image as a grayscale JPEG with Pillow."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def make_preview(
    data: FrameData,
    *,
    max_pixels: int,
    quality: int,
    calibration: PreviewCalibration | None = None,
) -> Preview:
    """The JPEG of a frame: shrunk under `max_pixels`, calibrated, stretched, and encoded.

    Without a `calibration`, the shrunk image goes straight to the stretch.
    """
    height, width = data.shape
    factor = shrink_factor(height, width, max_pixels)
    shrunk = block_mean(data, factor)
    if calibration is not None:
        shrunk = calibration(data, shrunk, factor)
    image = stretch_asinh(shrunk)
    return Preview(encode_jpeg(image, quality), image.shape[1], image.shape[0], factor)


def saturation_in_frame_units(native_dn: float, adc_bits: int, pixel_bits: int) -> float:
    """The saturation level in the units of the frame data.

    A 16-bit frame holds the ADC value in the high bits, and an 8-bit frame holds the top eight
    bits, so the native level moves by `2 ** (pixel_bits - adc_bits)`. The result never exceeds the
    largest value that the data type holds.
    """
    level = native_dn * 2.0 ** (pixel_bits - adc_bits)
    return min(level, float(2**pixel_bits - 1))


def frame_saturation_dn(frame: Frame, native_dn: float) -> float:
    """The saturation level of a frame in its own units (see `saturation_in_frame_units`)."""
    return saturation_in_frame_units(native_dn, frame.adc_bits, frame.pixel_format.value)


def histogram_counts(data: FrameData, bins: int, max_dn: float) -> list[int]:
    """Counts of pixels in `bins` equal bins from 0 to `max_dn`. The last bin holds the top.

    The count uses every second pixel in each direction, which is enough for a histogram and
    four times cheaper. Values above `max_dn` count in the last bin.
    """
    if bins < 1 or max_dn <= 0:
        raise ValueError("bins and max_dn must be positive")
    sample = data[::HISTOGRAM_STRIDE, ::HISTOGRAM_STRIDE]
    index = np.minimum(
        (sample.astype(np.float32) * np.float32(bins / max_dn)).astype(np.int32), bins - 1
    )
    return [int(count) for count in np.bincount(index.ravel(), minlength=bins)[:bins]]


def saturated_fraction(data: FrameData, level_dn: float) -> float:
    """The share of pixels at or above `level_dn`."""
    if data.size == 0:
        return 0.0
    return float(np.count_nonzero(data >= level_dn)) / float(data.size)
