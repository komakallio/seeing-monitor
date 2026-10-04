"""A synthetic sky for the preview calibration: a vignetted lens, a dust shadow, and hot pixels.

The sensitivity of the sensor falls with `cos^4` of the angle from the middle of the frame
(`CORNER_DEG` in the corners), and one dust shadow takes a part of it. A frame holds the bias, the
dark current, the sky times the sensitivity, and the hot pixels, as 14-bit counts in the high bits
of 16, as the driver delivers them. `write_flat` and `add_dark_set` describe the same sensor in the
files that the calibrator reads, so the calibrated image of a frame shows the sky and nothing of
the lens.

The sky is smooth and has no noise, so a test can ask for a flat image to within a fraction of a
percent. Nothing here depends on the real camera.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt

from seeingmon.clock import NS_PER_S
from seeingmon.frames import Frame
from seeingmon.survey.dark import DarkLibrary, DarkSet
from tests.scheduler.helpers import make_frame

FloatImage = npt.NDArray[np.float32]

SHAPE = (96, 128)  # (rows, columns): the sensor of the test
CORNER_DEG = 32.0  # the angle from the middle to the corner
DUST = (30.0, 44.0, 4.0, 0.25)  # row, column, sigma in pixels, and depth of the shadow
HOT = (60, 80)  # row and column of the hot pixel
HOT_EXCESS_DN = 3000.0  # native counts above the neighbors, in an exposure of SET_EXPOSURE_S
BIAS_DN = 130.0
DARK_RATE_DN_PER_S = 0.5  # at REFERENCE_C
DOUBLING_C = 6.0  # the prior of the dark model, which one set leaves in place
SKY_DN = 600.0  # native counts of the sky in the middle of the frame
SET_EXPOSURE_S = 30.0
REFERENCE_C = 20.0
SHIFT = 4  # 14 bits in the high bits of 16
SET_TIME_NS = 1_790_000_000 * NS_PER_S


def sensitivity(
    shape: tuple[int, int] = SHAPE,
    *,
    dust: tuple[float, float, float, float] | None = DUST,
    corner_deg: float = CORNER_DEG,
) -> FloatImage:
    """The relative sensitivity: `cos^4` of the angle from the middle, and the dust shadow."""
    height, width = shape
    rows, columns = np.mgrid[0:height, 0:width].astype(np.float64)
    r2 = (rows - (height - 1) / 2) ** 2 + (columns - (width - 1) / 2) ** 2
    focal2 = r2.max() / math.tan(math.radians(corner_deg)) ** 2
    value = 1.0 / (1.0 + r2 / focal2) ** 2  # cos^4 of the angle, whose tangent is r / f
    if dust is not None:
        row, column, sigma, depth = dust
        distance2 = (rows - row) ** 2 + (columns - column) ** 2
        value *= 1.0 - depth * np.exp(-distance2 / (2.0 * sigma**2))
    return np.asarray(value, dtype=np.float32)


def dark_level_dn(exposure_s: float, temperature_c: float = REFERENCE_C) -> float:
    """The bias plus the dark current of a dark frame, in native counts."""
    rate = DARK_RATE_DN_PER_S * math.pow(2.0, (temperature_c - REFERENCE_C) / DOUBLING_C)
    return BIAS_DN + rate * exposure_s


def frame_data(
    sens: FloatImage,
    *,
    exposure_s: float = SET_EXPOSURE_S,
    temperature_c: float = REFERENCE_C,
    sky_dn: float = SKY_DN,
    hot: Mapping[tuple[int, int], float] | None = None,
    dtype: type[np.uint8] | type[np.uint16] = np.uint16,
) -> npt.NDArray[np.uint8] | npt.NDArray[np.uint16]:
    """The pixels of a frame: the dark level, the sky through the sensitivity, and hot pixels.

    `hot` maps (row, column) to the excess of the pixel in native counts, as the frame has it.
    The default is the hot pixel `HOT` with the excess that follows its set: the exposure and the
    temperature scale it as they scale the dark current. An 8-bit frame holds the top 8 bits of
    the 14-bit value.
    """
    native = dark_level_dn(exposure_s, temperature_c) + sky_dn * sens.astype(np.float64)
    if hot is None:
        scale = exposure_s / SET_EXPOSURE_S
        scale *= math.pow(2.0, (temperature_c - REFERENCE_C) / DOUBLING_C)
        hot = {HOT: HOT_EXCESS_DN * scale}
    for (row, column), excess in hot.items():
        native[row, column] += excess
    if dtype is np.uint8:
        return np.asarray(np.clip(np.rint(native / 2 ** (14 - 8)), 0, 255), dtype=np.uint8)
    return np.asarray(np.clip(np.rint(native * SHIFT), 0, 65535), dtype=np.uint16)


def make_survey_frame(
    sens: FloatImage | None = None,
    *,
    exposure_s: float = SET_EXPOSURE_S,
    temperature_c: float | None = REFERENCE_C,
    mode: str = "bin2",
    gain: int = 120,
    seq: int = 0,
    t_utc_ns: int = SET_TIME_NS,
    sky_dn: float = SKY_DN,
    hot: Mapping[tuple[int, int], float] | None = None,
    dtype: type[np.uint8] | type[np.uint16] = np.uint16,
) -> Frame:
    """A frame of the sky through a sensor (see `frame_data`). A `None` temperature: no reading."""
    data = frame_data(
        sensitivity() if sens is None else sens,
        exposure_s=exposure_s,
        temperature_c=REFERENCE_C if temperature_c is None else temperature_c,
        sky_dn=sky_dn,
        hot=hot,
        dtype=dtype,
    )
    frame = make_frame(
        data,
        mode=mode,
        gain=gain,
        exposure_us=round(exposure_s * 1e6),
        adc_bits=14,
        t_utc_ns=t_utc_ns,
        seq=seq,
    )
    return dataclasses.replace(frame, temperature_c=temperature_c)


def window_of(frame: Frame, x: int, y: int, width: int, height: int) -> Frame:
    """The frame of a region of interest: the same pixels, and the offset on the sensor."""
    roi = dataclasses.replace(frame.roi, x=x, y=y, width=width, height=height)
    data = frame.data[y : y + height, x : x + width].copy()
    return dataclasses.replace(frame, data=data, roi=roi)


def write_flat(path: Path, sens: FloatImage) -> Path:
    """Write the sensitivity as the flat file that `[survey] flat_file` names (`.npy`)."""
    np.save(path, sens)
    return path


def add_dark_set(
    library: DarkLibrary,
    shape: tuple[int, int] = SHAPE,
    *,
    temperature_c: float = REFERENCE_C,
    hot: Mapping[tuple[int, int], float] | None = None,
    noise_dn: float = 2.0,
    gain: int = 120,
    seed: int = 1,
) -> DarkSet:
    """Record a dark set of the sensor: the master dark, with hot pixels, and the bias level.

    `hot` maps (row, column) to the excess of the pixel in the master dark. The default is `HOT`.
    """
    level = dark_level_dn(SET_EXPOSURE_S, temperature_c)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_dn, shape) if noise_dn else np.zeros(shape)
    master = np.rint(level + noise)
    for (row, column), excess in (hot or {HOT: HOT_EXCESS_DN}).items():
        master[row, column] += excess
    return library.add_set(
        master.astype(np.uint16),
        mode="bin2",
        gain=gain,
        exposure_s=SET_EXPOSURE_S,
        temperature_c=temperature_c,
        temperature_spread_c=0.3,
        t_utc_ns=SET_TIME_NS,
        n_frames=9,
        n_bias_frames=9,
        bias_dn=BIAS_DN,
        read_noise_dn=2.0,
        adc_bits=14,
        dark_dn=level,
    )


class ListedFlat:
    """A flat model (`seeingmon.survey.sky.FlatModel`) that wraps a sensitivity image.

    A test returns it from the flat provider of a calibrator, as a library-aware provider would.
    """

    def __init__(self, sens: FloatImage, version: str = "listed") -> None:
        self._image = sens
        self.version = version

    def at(self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.ones(np.shape(x), dtype=np.float64)

    def image(
        self, shape: tuple[int, int], origin: tuple[int, int] = (0, 0)
    ) -> npt.NDArray[np.float32] | None:
        x0, y0 = origin
        return self._image[y0 : y0 + shape[0], x0 : x0 + shape[1]]
