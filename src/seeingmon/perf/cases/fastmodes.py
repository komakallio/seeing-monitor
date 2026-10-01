"""The fast-path modes that the `kernel` and `fastpath` cases time, and their frames.

Each mode is a readout mode of the reference profile with a ROI, a pixel container, and the frame
rate that the budget uses:

| Mode | What it is | Rate |
|---|---|---|
| `bin1_128x128_u16` | The planned fast mode: bin1, a 128 x 128 ROI, 16-bit container. | 98 fps |
| `bin2_64x64_u16` | The fastest mode: bin2, a 64 x 64 ROI, 16-bit container. | 360 fps |
| `bin2_320x240_u8` | The owner's recordings: bin2, 320 x 240, 8-bit container. | 98 fps |

The frames come from `seeingmon.fastpath.benchmark.star_frames`, which draws a Polaris-like star
with photon and read noise. The profile's frame period for the first mode is 11.3 ms (88 fps), and
the budget uses the 98 fps of the owner's recordings, which is the harder figure. A smoke run uses
a smaller ROI for the largest mode, so that drawing its frames takes a fraction of a second.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy.typing as npt

    from seeingmon.profile import Profile

REFERENCE_PROFILE = "asi294mm-gs250"


@dataclass(frozen=True, slots=True)
class FastMode:
    """One fast-path mode. `shape` is `(height, width)` of the ROI."""

    key: str
    mode: str  # the readout mode in the profile
    shape: tuple[int, int]
    smoke_shape: tuple[int, int]
    container_bits: int
    exposure_us: int
    gain: int
    rate_hz: float
    summary: str

    @property
    def period_ns(self) -> int:
        """The frame period in nanoseconds, rounded up so that a count of frames never falls
        short of the time that it stands for."""
        return math.ceil(1e9 / self.rate_hz)

    def shape_for(self, smoke: bool) -> tuple[int, int]:
        return self.smoke_shape if smoke else self.shape

    def adc_bits(self, profile: Profile) -> int:
        """The ADC depth of the frames: the profile's for a 16-bit container, 8 for an 8-bit one."""
        return profile.mode(self.mode).adc_bits if self.container_bits == 16 else 8


FAST_MODES = (
    FastMode(
        "bin1_128x128_u16",
        "bin1",
        (128, 128),
        (128, 128),
        16,
        2000,
        0,
        98.0,
        "bin1, 128 x 128, 16 bit",
    ),
    FastMode(
        "bin2_64x64_u16", "bin2", (64, 64), (64, 64), 16, 2000, 0, 360.0, "bin2, 64 x 64, 16 bit"
    ),
    FastMode(
        "bin2_320x240_u8",
        "bin2",
        (240, 320),
        (60, 80),
        8,
        10_000,
        0,
        98.0,
        "bin2, 320 x 240, 8 bit",
    ),
)


def pool_frames(profile: Profile, mode: FastMode, smoke: bool = False) -> list[npt.NDArray[Any]]:
    """A pool of 64 distinct frames of a Polaris-like star in the pixel format of the mode.

    A case cycles through the pool, so that the cache cannot hide the cost of fresh pixels.

    A 16-bit mode keeps the ADC value in the high bits, as the camera delivers it. An 8-bit mode
    keeps the top eight bits and uses a conversion gain that puts the star below saturation, as
    the owner's defocused recordings do.
    """
    import numpy as np

    from seeingmon.fastpath.benchmark import star_frames

    shape = mode.shape_for(smoke)
    read_noise = profile.read_noise_e(mode.mode, mode.gain)
    if mode.container_bits == 16:
        pool = star_frames(
            shape, profile.e_per_adu(mode.mode, mode.gain), read_noise, mode.adc_bits(profile)
        )
    else:
        wide = star_frames(shape, 23.0, read_noise, 8)  # about 120 counts at the peak
        pool = (wide >> 8).astype(np.uint8)
    return list(pool)
