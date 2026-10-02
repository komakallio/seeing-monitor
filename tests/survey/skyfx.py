"""A synthetic night for the sky flat tests: survey frames of a rotating sky through a lens.

The camera is fixed to the ground, and the pole is at the middle of the frame. A frame at the time
`t` shows a sky that has turned about the middle by the roll `roll(t)`, the Earth rotation angle.
The model draws, in the units of a bin2 frame at gain 120 (counts of the 14-bit ADC):

- **A sky level of 400 electrons a pixel**, five times the real one. The sensor has 65 times fewer
  pixels than the real one, and the analysis bins by 2 where the real one bins by 4, so this level
  gives the noise of a binned pixel that the real sky gives. The sky has a smooth structure that
  is fixed on the sky (1.5% rms, which the rotation turns through the frame), and a gradient that
  is fixed to the ground, and so to the sensor (4.5% across the height).
- **Stars that turn about the pole:** 150 that the detector finds and 400 that stay below it, with
  a Gaussian profile of 1.1 pixels. The density is 20 times the real one per pixel and still low
  enough that the masks cover only a few percent of a frame.
- **Polaris:** a star at the radius of its orbit, saturated, with a halo that falls with the cube
  of the distance. A halo that the mask leaves makes a ring at the radius of the orbit.
- **The lens:** the whole sky, stars and halo included, goes through the flat of `flatfx` (a
  vignetting, a tilt, dust shadows, and a fixed pattern of the pixels).
- **The sensor:** photon and read noise, a bias pattern, a dark level, and hot pixels. A dark
  library holds the master dark (the pattern, the dark level, and the hot pixels).

The functions use the same Earth rotation angle as the command, so the roll of a frame is exact.
The site is an arbitrary one, and the times are a night in December when the Sun stays below -18
degrees for 11.75 hours and the Moon is down.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from seeingmon.clock import NS_PER_S, iso_to_utc_ns
from seeingmon.frames import FrameFlag
from seeingmon.profile import Profile
from seeingmon.scheduler.config import SiteConfig
from seeingmon.store.layout import DataLayout
from seeingmon.survey import framefile
from seeingmon.survey.apparent import earth_rotation_angle
from seeingmon.survey.dark import DarkLibrary
from tests.scheduler.helpers import make_frame
from tests.survey import flatfx as fx

TEST_SITE = SiteConfig(latitude_deg=52.0, longitude_deg=8.0)  # an arbitrary site, not a station
NIGHT_START_NS = iso_to_utc_ns("2026-12-09T17:45:00Z")  # the Sun is below -18 degrees to 05:15
STEP_NS = 30 * 60 * NS_PER_S
EXPOSURE_S = 30.0
TEMPERATURE_C = 10.0
SKY_E_PER_PX = 400.0
STRUCTURE_RMS = 0.015
STRUCTURE_SIGMA_PX = 3.0
FLUX_SCALE = SKY_E_PER_PX / 80.0  # the fluxes of the stars follow the sky level
GROUND_GRADIENT = (0.0, 0.045)  # a fraction across the width and the height, fixed to the sensor
STAR_SIGMA_PX = 1.1
BIAS_NATIVE = 133.9
DARK_LEVEL_NATIVE = 4.0

FloatImage = npt.NDArray[np.float32]
Pixels = npt.NDArray[np.uint16]


def night_times(count: int, *, start_ns: int = NIGHT_START_NS, step_ns: int = STEP_NS) -> list[int]:
    """The times of `count` frames, one every half hour from the start of the night."""
    return [start_ns + i * step_ns for i in range(count)]


def roll_rad(t_utc_ns: int) -> float:
    return float(earth_rotation_angle(t_utc_ns))


def smooth_field(size: int, sigma: float, seed: int) -> FloatImage:
    """A random field with unit rms and a correlation length of about `sigma` pixels."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((size, size))
    freq = np.fft.fftfreq(size)
    kernel = np.exp(-2.0 * (np.pi * sigma) ** 2 * (freq[:, None] ** 2 + freq[None, :] ** 2))
    field = np.fft.ifft2(np.fft.fft2(white) * kernel).real
    field -= field.mean()
    return np.asarray(field / field.std(), dtype=np.float32)


def bilerp(grid: FloatImage, x: FloatImage, y: FloatImage) -> FloatImage:
    """The grid at positions `(x, y)` in grid pixels, by bilinear interpolation."""
    height, width = grid.shape
    x0 = np.clip(np.floor(x).astype(np.intp), 0, width - 2)
    y0 = np.clip(np.floor(y).astype(np.intp), 0, height - 2)
    fx_ = np.clip(x - x0, 0.0, 1.0)
    fy_ = np.clip(y - y0, 0.0, 1.0)
    top = grid[y0, x0] * (1 - fx_) + grid[y0, x0 + 1] * fx_
    bottom = grid[y0 + 1, x0] * (1 - fx_) + grid[y0 + 1, x0 + 1] * fx_
    return np.asarray(top * (1 - fy_) + bottom * fy_, dtype=np.float32)


@dataclass(frozen=True)
class Halo:
    """The halo of Polaris: `amplitude` times the sky level at the core, falling as
    `(1 + (d / core_px)^2)^-1.5` with the distance `d`."""

    amplitude: float = 2.2
    core_px: float = 6.0


class NightSky:
    """A sky fixed to the pole and the lens, ground, and sensor around it. See the module text."""

    def __init__(
        self,
        truth: FloatImage,
        *,
        orbit_px: float,
        seed: int = 11,
        stars: int = 150,
        faint_stars: int = 400,
        halo: Halo | None = None,
        sky_e: float = SKY_E_PER_PX,
        structure_rms: float = STRUCTURE_RMS,
        ground_gradient: tuple[float, float] = GROUND_GRADIENT,
    ) -> None:
        self.truth = truth
        self.shape = (int(truth.shape[0]), int(truth.shape[1]))
        height, width = self.shape
        self.center = ((width - 1) / 2.0, (height - 1) / 2.0)
        self.orbit_px = orbit_px
        self.halo = halo or Halo()
        self.sky_adu = sky_e / fx.E_PER_ADU
        self.structure_rms = structure_rms
        reach = math.hypot(width, height) / 2.0 + 12.0
        size = int(2 * reach) + 8
        self._reach = reach
        self._grid = smooth_field(size, STRUCTURE_SIGMA_PX, seed)
        rng = np.random.default_rng(seed + 1)
        count = stars + faint_stars
        radius = reach * np.sqrt(rng.uniform(0.0, 1.0, count))
        angle = rng.uniform(0.0, 2.0 * np.pi, count)
        self._star_a = (radius * np.cos(angle)).astype(np.float64)
        self._star_b = (radius * np.sin(angle)).astype(np.float64)
        bright = rng.uniform(0.0, 1.0, stars) ** -1.0 * 600.0  # electrons, a power law
        faint = rng.uniform(40.0, 220.0, faint_stars)
        self._star_e = FLUX_SCALE * np.concatenate([np.minimum(bright, 4e5), faint])
        self._polaris_phase = 0.9
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        self._u = xx - np.float32(self.center[0])
        self._v = yy - np.float32(self.center[1])
        gx, gy = ground_gradient
        self._ground = np.asarray(
            1.0 + gx * self._u / width + gy * self._v / height, dtype=np.float32
        )
        pattern_rng = np.random.default_rng(seed + 2)
        self.bias = np.asarray(
            BIAS_NATIVE + 1.2 * pattern_rng.standard_normal(self.shape), dtype=np.float32
        )
        hot = pattern_rng.integers(0, height * width, 40)
        self.hot_flat = np.zeros(height * width, dtype=np.float32)
        self.hot_flat[hot] = pattern_rng.uniform(60.0, 400.0, 40)
        self.master_dark = (
            self.bias + np.float32(DARK_LEVEL_NATIVE) + self.hot_flat.reshape(self.shape)
        )

    def polaris_position(self, theta: float) -> tuple[float, float]:
        a = self.orbit_px * math.cos(self._polaris_phase)
        b = self.orbit_px * math.sin(self._polaris_phase)
        return (
            self.center[0] + a * math.cos(theta) - b * math.sin(theta),
            self.center[1] + a * math.sin(theta) + b * math.cos(theta),
        )

    def signal_adu(self, theta: float) -> FloatImage:
        """The light of the sky before the lens, in counts: sky, stars, and Polaris."""
        height, width = self.shape
        cos, sin = math.cos(theta), math.sin(theta)
        a = self._u * np.float32(cos) + self._v * np.float32(sin)
        b = -self._u * np.float32(sin) + self._v * np.float32(cos)
        grid_center = self._grid.shape[0] / 2.0
        structure = bilerp(self._grid, a + np.float32(grid_center), b + np.float32(grid_center))
        sky = (
            np.float32(self.sky_adu)
            * self._ground
            * (1.0 + np.float32(self.structure_rms) * structure)
        )
        stars = np.zeros(self.shape, dtype=np.float32)
        xs = self.center[0] + self._star_a * cos - self._star_b * sin
        ys = self.center[1] + self._star_a * sin + self._star_b * cos
        sigma = STAR_SIGMA_PX
        reach = 6
        gx = np.arange(-reach, reach + 1, dtype=np.float32)
        for x, y, e in zip(xs, ys, self._star_e, strict=True):
            ix, iy = round(float(x)), round(float(y))
            if ix < -reach or ix >= width + reach or iy < -reach or iy >= height + reach:
                continue
            x0, x1 = max(ix - reach, 0), min(ix + reach + 1, width)
            y0, y1 = max(iy - reach, 0), min(iy + reach + 1, height)
            if x0 >= x1 or y0 >= y1:
                continue
            px = np.exp(-((gx[x0 - ix + reach : x1 - ix + reach] + ix - x) ** 2) / (2 * sigma**2))
            py = np.exp(-((gx[y0 - iy + reach : y1 - iy + reach] + iy - y) ** 2) / (2 * sigma**2))
            stars[y0:y1, x0:x1] += np.float32(e / fx.E_PER_ADU / (2 * np.pi * sigma**2)) * (
                py[:, None] * px[None, :]
            )
        px_polaris, py_polaris = self.polaris_position(theta)
        distance = np.hypot(
            np.arange(width, dtype=np.float32)[None, :] - np.float32(px_polaris),
            np.arange(height, dtype=np.float32)[:, None] - np.float32(py_polaris),
        )
        core = np.float32(FLUX_SCALE * 4e7 / fx.E_PER_ADU / (2 * np.pi * 1.4**2)) * np.exp(
            -(distance**2) / np.float32(2 * 1.4**2)
        )
        halo = np.float32(self.halo.amplitude * self.sky_adu) / (
            1.0 + (distance / np.float32(self.halo.core_px)) ** 2
        ) ** np.float32(1.5)
        return np.asarray(sky + stars + core + halo, dtype=np.float32)

    def frame_pixels(self, t_utc_ns: int, rng: np.random.Generator) -> Pixels:
        """The pixels of a frame at a time, in the 16-bit container: 14-bit counts shifted by 2."""
        light = self.signal_adu(roll_rad(t_utc_ns)) * self.truth
        electrons = light * np.float32(fx.E_PER_ADU)
        noise = np.sqrt(electrons + np.float32(fx.READ_NOISE_E**2)) * rng.standard_normal(
            self.shape, dtype=np.float32
        )
        adc = (electrons + noise) / np.float32(fx.E_PER_ADU) + self.master_dark
        return fx.to_file_counts(adc, shift=fx.CONTAINER_SHIFT)


def make_night_library(
    directory: Path, sky: NightSky, *, temperature_c: float = TEMPERATURE_C
) -> None:
    """A dark library with the master dark of the model: its pattern, dark level, and hot pixels."""
    library = DarkLibrary(directory / "darks")
    master = np.rint(sky.master_dark).astype(np.uint16)
    library.add_set(
        master,
        mode="bin2",
        gain=120,
        exposure_s=EXPOSURE_S,
        temperature_c=temperature_c,
        temperature_spread_c=0.1,
        t_utc_ns=iso_to_utc_ns("2026-12-01T12:00:00Z"),
        n_frames=9,
        n_bias_frames=9,
        bias_dn=BIAS_NATIVE,
        read_noise_dn=2.1,
        adc_bits=14,
        dark_dn=BIAS_NATIVE + DARK_LEVEL_NATIVE,
    )


def write_frame(
    layout: DataLayout,
    pixels: Pixels,
    t_utc_ns: int,
    profile: Profile,
    *,
    cloud_fraction: float | None = 0.02,
    transparency: float | None = 0.99,
    reasons: Sequence[str] = ("every_10",),
    temperature_c: float | None = TEMPERATURE_C,
    flags: FrameFlag = FrameFlag.NONE,
    compress: bool = False,
    exposure_s: float = EXPOSURE_S,
    seq: int = 0,
) -> Path:
    """Write a frame as a survey FITS file under the layout, as `core` names it."""
    frame = make_frame(
        pixels,
        mode="bin2",
        gain=120,
        exposure_us=round(exposure_s * 1e6),
        adc_bits=14,
        t_utc_ns=t_utc_ns,
        seq=seq,
    )
    frame = dataclasses.replace(frame, flags=flags, temperature_c=temperature_c)
    cards = framefile.frame_cards(
        frame,
        profile=profile,
        station_id="st",
        reasons=reasons,
        cloud_fraction=cloud_fraction,
        transparency=transparency,
    )
    comments = framefile.frame_comments(frame)
    path = layout.survey_path(t_utc_ns)

    def write(handle: object) -> None:
        framefile.write_frame_fits(
            handle,  # type: ignore[arg-type]
            framefile.native_pixels(frame),
            cards,
            comments,
            compress=compress,
        )

    layout.write_atomic(path, write)
    return path


def write_night(
    layout: DataLayout,
    sky: NightSky,
    profile: Profile,
    times: Sequence[int],
    *,
    seed: int = 21,
    compress: bool = False,
    **options: object,
) -> list[Path]:
    """Write one frame for each time. The noise of each frame follows from `seed` and the time."""
    paths: list[Path] = []
    for index, t in enumerate(times):
        rng = np.random.default_rng([seed, t % 2**31])
        pixels = sky.frame_pixels(t, rng)
        paths.append(
            write_frame(layout, pixels, t, profile, compress=compress, seq=index, **options)  # type: ignore[arg-type]
        )
    return paths
