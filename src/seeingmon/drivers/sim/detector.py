"""The sensor: photon noise, dark current, read noise, saturation, and the ADC.

`Detector.digitize` turns a *mean* image in electrons into camera data. The mean image holds the
expected signal of every pixel: the stars, the sky, the dark current, and the hot pixels. The
detector draws the photon count of each pixel from a Poisson distribution, clips the charge at
the full well, adds Gaussian read noise, converts to ADU with the electrons per ADU of the gain,
adds the black level (the offset), and rounds and clips to the ADC range. RAW16 data carries
the ADC value in the high bits (the low bits are zero), and RAW8 data carries the top 8 bits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from seeingmon.drivers.sim.params import SimParams
from seeingmon.frames import FrameData, PixelFormat, Roi

FloatArray = npt.NDArray[np.float64]

_CHUNK_PIXELS = 1_000_000
# The offset setting counts in 12-bit ADU. The black level scales up for deeper ADCs.
_OFFSET_REFERENCE_BITS = 12


@dataclass(frozen=True, slots=True)
class HotPixelConfig:
    """A fixed map of hot pixels: `density_per_mpix` pixels per megapixel of the full frame.

    Each hot pixel has a dark rate that is log-normal with a median of `median_rate_e_per_s` and
    a log-sigma of `spread`, in electrons per second at the reference temperature.
    """

    density_per_mpix: float = 0.0
    median_rate_e_per_s: float = 3.0
    spread: float = 1.0

    def __post_init__(self) -> None:
        if self.density_per_mpix < 0 or self.median_rate_e_per_s < 0 or self.spread < 0:
            raise ValueError("hot-pixel density, rate, and spread must not be negative")


class HotPixelMap:
    """The positions and rates of the hot pixels of one readout mode. The map never changes."""

    def __init__(self, params: SimParams, config: HotPixelConfig, seed: int) -> None:
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, 0x407, params.width, params.height])
        )
        count = round(config.density_per_mpix * params.width * params.height / 1e6)
        self.x = rng.integers(0, params.width, count)
        self.y = rng.integers(0, params.height, count)
        self.rate_e_per_s = config.median_rate_e_per_s * np.exp(
            config.spread * rng.standard_normal(count)
        )
        self._params = params

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def add_to(
        self, image: npt.NDArray[np.float32], roi: Roi, exposure_s: float, temperature_c: float
    ) -> None:
        """Add the hot-pixel electrons that fall inside the ROI to a mean image."""
        if len(self) == 0:
            return
        inside = (self.x >= roi.x) & (self.x < roi.x_end) & (self.y >= roi.y) & (self.y < roi.y_end)
        if not inside.any():
            return
        params = self._params
        # A hot pixel's rate follows the same temperature law as the dark current.
        factor = params.dark_rate_e_per_s(temperature_c) / params.dark_rate_e_per_s(
            params.dark_reference_c
        )
        np.add.at(
            image,
            (self.y[inside] - roi.y, self.x[inside] - roi.x),
            (self.rate_e_per_s[inside] * exposure_s * factor).astype(np.float32),
        )


# Above this mean, a Poisson count is drawn as a rounded Gaussian. The skewness is 0.1, and the
# draw is three times faster, which matters for the 12 million pixels of a survey frame.
_GAUSSIAN_PHOTON_THRESHOLD = 100.0


def _photon_counts(
    mean: npt.NDArray[np.float32], rng: np.random.Generator
) -> npt.NDArray[np.float32]:
    """Draw photon counts for a block of mean electron counts."""
    large = mean >= _GAUSSIAN_PHOTON_THRESHOLD
    if not large.any():
        return rng.poisson(mean).astype(np.float32)
    noise = rng.standard_normal(mean.shape, dtype=np.float32)
    counts = mean + np.sqrt(mean) * noise
    np.rint(counts, out=counts)
    np.maximum(counts, 0.0, out=counts)
    small = ~large
    if small.any():
        counts[small] = rng.poisson(mean[small])
    return np.asarray(counts, dtype=np.float32)


class Detector:
    """Converts mean electron images into ADC data for one readout mode."""

    def __init__(self, params: SimParams) -> None:
        self._params = params

    @property
    def params(self) -> SimParams:
        return self._params

    def black_level_adu(self, offset: int | None) -> float:
        """The ADC value of a pixel with no signal, from the offset setting."""
        setting = self._params.default_offset if offset is None else offset
        return setting * 2.0 ** (self._params.adc_bits - _OFFSET_REFERENCE_BITS)

    def digitize(
        self,
        mean_electrons: npt.NDArray[np.float32],
        *,
        gain: int,
        offset: int | None,
        pixel_format: PixelFormat,
        rng: np.random.Generator,
    ) -> FrameData:
        """Draw one frame from a mean image in electrons.

        `mean_electrons` has shape `(height, width)`. The generator supplies the photon and
        read noise, so a seeded generator makes the frame repeatable.
        """
        params = self._params
        sensor = params.sensor_at(gain)
        bias = self.black_level_adu(offset)
        full_scale = params.adc_full_scale
        height, width = mean_electrons.shape
        dtype = np.uint8 if pixel_format is PixelFormat.RAW8 else np.uint16
        out = np.empty((height, width), dtype=dtype)
        rows_per_chunk = max(1, _CHUNK_PIXELS // max(width, 1))
        for start in range(0, height, rows_per_chunk):
            block = np.maximum(mean_electrons[start : start + rows_per_chunk], 0.0)
            electrons = _photon_counts(block, rng)
            np.minimum(electrons, np.float32(sensor.full_well_e), out=electrons)
            electrons += rng.standard_normal(electrons.shape, dtype=np.float32) * np.float32(
                sensor.read_noise_e
            )
            adu = electrons * np.float32(1.0 / sensor.e_per_adu) + np.float32(bias)
            np.rint(adu, out=adu)
            np.clip(adu, 0, full_scale, out=adu)
            values = adu.astype(np.uint16)
            if pixel_format is PixelFormat.RAW8:
                out[start : start + rows_per_chunk] = values >> (params.adc_bits - 8)
            else:
                out[start : start + rows_per_chunk] = values << (16 - params.adc_bits)
        return out

    def expected_adu(self, electrons: float, gain: int, offset: int | None) -> float:
        """The ADC value that a pixel holding `electrons` reads (before noise and clipping)."""
        sensor = self._params.sensor_at(gain)
        return electrons / sensor.e_per_adu + self.black_level_adu(offset)

    def saturation_electrons(self, gain: int) -> float:
        """The charge at which a pixel clips: the full well or the ADC limit, whichever is lower."""
        return self._params.sensor_at(gain).full_well_e

    def noise_floor_adu(self, gain: int) -> float:
        """The rms read noise in ADU at a gain."""
        sensor = self._params.sensor_at(gain)
        return sensor.read_noise_e / sensor.e_per_adu
