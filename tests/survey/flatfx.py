"""Synthetic panel frames for the flat tests: a lens, a light source, a sensor, and a dark library.

**Two lenses.** `OWNER_LENS` imitates the real lens as a bench test measured it: a vignetting of
0.43% at 0.5 degrees from the middle, 2.25% at 1.0, 4.05% at 1.5, 6.5% at 2.0, 9.3% at 2.5, and
10% in the corners, a tilt of the optics and the sensor of +0.56% across the width and -0.34%
across the height, one dust shadow of 3% (71 px across), two smaller ones (65 and 58 px), and two
edge artifacts of 1.3% about 13 px wide at the left edge. `SIMULATION_LENS` is the lens of the
simulation of the sky builder: 30% radial vignetting and a 1.5% tilt. The lens is a function of
the angle from the middle, so a smaller sensor with a larger plate scale shows the same lens:
`scale_down` is the factor, and the positions and sizes of the dust shadows (given in pixels of
the reference sensor) scale with it. The edge artifacts keep their size in pixels.

**A light source.** A source has a gradient of its own across the frame, a fraction across the
width and across the height. A phone screen turned by 180 degrees flips it.

**A sensor.** A frame holds a bias level with a fixed pattern, the light of the source through the
lens, photon noise, and read noise. It stores 14-bit counts in the high bits of 16 (the container
that the SDK delivers) or as native counts.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from seeingmon.clock import iso_to_utc_ns
from seeingmon.profile import Profile
from seeingmon.survey.dark import DarkLibrary
from tests.survey import synth

REFERENCE_SHAPE = (2822, 4144)  # (height, width) of the survey mode, in pixels
REFERENCE_SCALE_ARCSEC_PX = 3.82
E_PER_ADU = 0.88  # at gain 120 in bin2
READ_NOISE_E = 1.85
NATIVE_FULL_SCALE = 16383  # 14 bits
CONTAINER_SHIFT = 2

FloatImage = npt.NDArray[np.float32]
Pixels = npt.NDArray[np.uint16]


# --- The lens -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Spot:
    """A dust shadow or an edge artifact: its center, depth (a fraction), and diameter in pixels."""

    x: float
    y: float
    depth: float
    diameter: float


@dataclass(frozen=True)
class LensSpec:
    """A lens and a sensor: the loss of light by radius, the tilt, and the dust."""

    name: str
    radii_deg: tuple[float, ...]
    losses: tuple[float, ...]
    optics_tilt: tuple[float, float]
    shadows: tuple[Spot, ...]
    edge_artifacts: tuple[Spot, ...]
    prnu: float = 0.003

    def loss_at(self, radius_deg: npt.NDArray[np.float32]) -> FloatImage:
        return monotone_interp(self.radii_deg, self.losses, radius_deg)


OWNER_LENS = LensSpec(
    name="owner",
    radii_deg=(0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 2.66),
    losses=(0.0, 0.0011, 0.0043, 0.0225, 0.0405, 0.065, 0.093, 0.100),
    optics_tilt=(0.0056, -0.0034),
    shadows=(
        Spot(2489, 1994, 0.030, 71),
        Spot(4065, 1395, 0.016, 65),
        Spot(1462, 1795, 0.014, 58),
    ),
    edge_artifacts=(Spot(14, 700, 0.0135, 13), Spot(14, 2100, 0.013, 13)),
)

SIMULATION_LENS = LensSpec(
    name="simulation",
    radii_deg=(0.0, 1.0, 2.0, 2.66),
    losses=(0.0, 0.0424, 0.1696, 0.30),
    optics_tilt=(0.015, 0.0),
    shadows=(
        Spot(1200, 900, 0.040, 120),
        Spot(2900, 2000, 0.020, 90),
        Spot(3500, 700, 0.025, 100),
    ),
    edge_artifacts=(),
)

# The tilt that the owner's source adds to set A, as a fraction across the width and the height.
# Turned by 180 degrees, it adds the negative of this to set B.
OWNER_SOURCE_GRADIENT = (-0.0064, 0.0068)


def monotone_interp(
    x: Sequence[float], y: Sequence[float], query: npt.NDArray[np.float32]
) -> FloatImage:
    """A monotone cubic interpolation (Fritsch and Carlson) through the points, held at the ends."""
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    h = np.diff(xs)
    delta = np.diff(ys) / h
    slope = np.zeros_like(ys)
    for i in range(1, len(xs) - 1):
        if delta[i - 1] * delta[i] > 0:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            slope[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
    slope[-1] = delta[-1]
    q = np.clip(query.astype(np.float64), xs[0], xs[-1])
    index = np.clip(np.searchsorted(xs, q, side="right") - 1, 0, len(xs) - 2)
    t = (q - xs[index]) / h[index]
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    out = (
        h00 * ys[index]
        + h10 * h[index] * slope[index]
        + h01 * ys[index + 1]
        + h11 * h[index] * slope[index + 1]
    )
    return np.asarray(out, dtype=np.float32)


def _spot_profile(
    x: FloatImage,
    y: FloatImage,
    spot: Spot,
    *,
    size_scale: float,
    x_scale: float,
    y_scale: float,
) -> FloatImage:
    """The fraction of light that a spot removes at each pixel: a disk with a soft edge."""
    radius = spot.diameter / size_scale / 2.0
    distance = np.hypot(x - spot.x / x_scale, y - spot.y / y_scale)
    edge = max(0.12 * radius, 0.6)
    profile = 0.5 * (1.0 - np.tanh((distance - radius) / edge))
    return np.asarray(spot.depth * profile, dtype=np.float32)


def lens_flat(
    spec: LensSpec,
    shape: tuple[int, int],
    *,
    scale_down: float = 1.0,
    seed: int = 1,
    edge_artifact_x: float | None = None,
) -> FloatImage:
    """The flat of a lens on a sensor of `shape`, scaled to a median of 1.

    `scale_down` says how many times fewer pixels the sensor has in each direction than the
    reference one: a sensor of 518 x 352 pixels with `scale_down` 8 has pixels 8 times as large, so
    it shows the same field. The edge artifacts keep their size and their distance from the edge
    in pixels, and `edge_artifact_x` moves them to another distance, which a small sensor needs
    to keep them inside the margin that its test uses.
    """
    height, width = shape
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    center_x, center_y = (width - 1) / 2.0, (height - 1) / 2.0
    scale = REFERENCE_SCALE_ARCSEC_PX * scale_down / 3600.0
    radius_deg = np.hypot(x - center_x, y - center_y) * np.float32(scale)
    flat = 1.0 - spec.loss_at(radius_deg)
    tilt_x, tilt_y = spec.optics_tilt
    flat *= (
        1.0
        + np.float32(tilt_x) * (x - center_x) / width
        + np.float32(tilt_y) * (y - center_y) / height
    )
    for spot in spec.shadows:
        flat *= 1.0 - _spot_profile(
            x, y, spot, size_scale=scale_down, x_scale=scale_down, y_scale=scale_down
        )
    for spot in spec.edge_artifacts:  # the distance from the edge and the size keep their pixels
        placed = (
            spot
            if edge_artifact_x is None
            else Spot(edge_artifact_x, spot.y, spot.depth, spot.diameter)
        )
        flat *= 1.0 - _spot_profile(x, y, placed, size_scale=1.0, x_scale=1.0, y_scale=scale_down)
    rng = np.random.default_rng(seed)
    flat *= 1.0 + np.float32(spec.prnu) * rng.standard_normal((height, width), dtype=np.float32)
    flat /= np.float32(np.median(flat))
    return np.asarray(flat, dtype=np.float32)


def source_pattern(shape: tuple[int, int], gradient: tuple[float, float]) -> FloatImage:
    """The light of a source with a gradient: a fraction across the width and across the height."""
    height, width = shape
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    pattern = 1.0 + np.float32(gradient[0]) * (x - (width - 1) / 2.0) / width
    pattern += np.float32(gradient[1]) * (y - (height - 1) / 2.0) / height
    return np.asarray(pattern, dtype=np.float32)


# --- The sensor -----------------------------------------------------------------------------


def scaled_profile(width: int, height: int) -> Profile:
    """The reference profile on a smaller sensor of larger pixels, so that the field stays.

    The bin2 mode has `width` x `height` pixels, the pixels are `4144 / width` times as large as
    the reference ones, and the lens is the same, so the plate scale grows by that factor.
    """
    factor = REFERENCE_SHAPE[1] / width
    data = synth.reference_profile().model_dump(mode="python")
    for mode in data["readout_modes"]:
        binning = 2 // mode["sdk_bin"]
        mode["width_px"] = width * binning
        mode["height_px"] = height * binning
        mode["pixel_size_um"] = mode["pixel_size_um"] * factor
    return Profile.model_validate(data)


def bias_pattern(shape: tuple[int, int], *, level: float, seed: int = 7) -> FloatImage:
    """The bias in native counts: a level with a fixed pattern of 1.2 counts rms."""
    rng = np.random.default_rng(seed)
    return np.asarray(level + 1.2 * rng.standard_normal(shape, dtype=np.float32), dtype=np.float32)


def to_file_counts(adc: FloatImage, *, shift: int) -> Pixels:
    """Round native counts, clip them to 14 bits, and shift them into the file's scale."""
    native = np.clip(np.rint(adc), 0, NATIVE_FULL_SCALE).astype(np.uint16)
    return np.asarray(native << np.uint16(shift), dtype=np.uint16)


def panel_frame(
    truth: FloatImage,
    source: FloatImage,
    *,
    level: float,
    bias: FloatImage,
    rng: np.random.Generator,
    shift: int = CONTAINER_SHIFT,
) -> Pixels:
    """One frame of a panel: the lens, the source, photon and read noise, and the bias.

    `level` is the mean signal above the bias in native counts.
    """
    electrons = truth * source * np.float32(level * E_PER_ADU)
    noise = np.sqrt(electrons + np.float32(READ_NOISE_E**2)) * rng.standard_normal(
        truth.shape, dtype=np.float32
    )
    adc = (electrons + noise) / np.float32(E_PER_ADU) + bias
    return to_file_counts(adc, shift=shift)


# --- Frame sources --------------------------------------------------------------------------


class ArraySource:
    """A set of frames in memory, or made on demand by a function of the frame number."""

    def __init__(
        self,
        frames: Sequence[Pixels] | Callable[[int], Pixels],
        *,
        count: int | None = None,
        shape: tuple[int, int] | None = None,
        declared_bits: int | None = None,
        temperature_c: float | None = None,
        gain: int | None = None,
    ) -> None:
        self._frames = frames
        if callable(frames):
            if count is None or shape is None:
                raise ValueError("a function of the frame number needs a count and a shape")
            self._count, self._shape = count, shape
        else:
            self._count = len(frames)
            self._shape = (int(frames[0].shape[0]), int(frames[0].shape[1]))
        self._declared_bits = declared_bits
        self._temperature_c = temperature_c
        self._gain = gain
        self.reads = 0

    @property
    def count(self) -> int:
        return self._count

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def declared_bits(self) -> int | None:
        return self._declared_bits

    @property
    def temperature_c(self) -> float | None:
        return self._temperature_c

    @property
    def gain(self) -> int | None:
        return self._gain

    def frame(self, index: int) -> Any:
        self.reads += 1
        if callable(self._frames):
            return self._frames(index)
        return self._frames[index]

    def close(self) -> None:
        return None


def panel_set(
    truth: FloatImage,
    gradient: tuple[float, float],
    *,
    frames: int,
    seed: int,
    level: float = 7400.0,
    bias: FloatImage | None = None,
    bias_level: float = 133.9,
    flicker: dict[int, float] | None = None,
    shift: int = CONTAINER_SHIFT,
    temperature_c: float | None = None,
    gain: int | None = None,
    on_demand: bool = False,
) -> ArraySource:
    """A set of panel frames: the lens `truth` lit by a source with `gradient`.

    `flicker` maps a frame number to a relative change of the level, such as `{3: 0.15}`.
    `on_demand` makes each frame when the command asks for it, which keeps a full-size set out of
    memory. The same seed gives the same frames either way.
    """
    pattern = source_pattern(truth.shape, gradient)
    offset = bias if bias is not None else bias_pattern(truth.shape, level=bias_level)

    def make(index: int) -> Pixels:
        rng = np.random.default_rng([seed, index])
        scale = 1.0 + (flicker or {}).get(index, 0.0)
        return panel_frame(truth, pattern, level=level * scale, bias=offset, rng=rng, shift=shift)

    shape = (truth.shape[0], truth.shape[1])
    if on_demand:
        return ArraySource(make, count=frames, shape=shape, temperature_c=temperature_c, gain=gain)
    return ArraySource([make(i) for i in range(frames)], temperature_c=temperature_c, gain=gain)


def bias_set(
    shape: tuple[int, int],
    *,
    frames: int,
    seed: int,
    level: float = 133.9,
    light: float = 0.0,
    middle_extra: float = 0.0,
    shift: int = CONTAINER_SHIFT,
) -> ArraySource:
    """Bias frames. With `light` above zero, the lens was not covered: light adds a level, noise,
    and `middle_extra` counts at the middle of the frame."""
    pattern = bias_pattern(shape, level=level)
    height, width = shape
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    radius = np.hypot(x - (width - 1) / 2.0, y - (height - 1) / 2.0)
    corner = float(radius.max())
    glow = np.asarray(
        light + middle_extra * np.exp(-((radius / (0.25 * corner)) ** 2)), dtype=np.float32
    )
    frames_out: list[Pixels] = []
    for index in range(frames):
        rng = np.random.default_rng([seed, index])
        electrons = glow * np.float32(E_PER_ADU)
        noise = rng.standard_normal(shape, dtype=np.float32)
        adc = (
            pattern
            + glow
            + (np.sqrt(electrons + np.float32(READ_NOISE_E**2)) * noise / np.float32(E_PER_ADU))
        )
        frames_out.append(to_file_counts(adc, shift=shift))
    return ArraySource(frames_out)


# --- The dark library -----------------------------------------------------------------------


def make_library(
    directory: Path,
    *,
    biases: Sequence[tuple[float, float]] = ((20.2, 128.0), (30.0, 134.2)),
    gain: int = 120,
) -> DarkLibrary:
    """A dark library under `directory` with one set of bin2 for each `(temperature, bias)`.

    The master darks are small, because the flat builder reads only the bias level of the sets.
    """
    library = DarkLibrary(directory / "darks")
    stamp = iso_to_utc_ns("2026-10-01T12:00:00Z")
    for number, (temperature, bias) in enumerate(biases):
        master = np.full((24, 32), round(bias + 1.0), dtype=np.uint16)
        library.add_set(
            master,
            mode="bin2",
            gain=gain,
            exposure_s=30.0,
            temperature_c=temperature,
            temperature_spread_c=0.1,
            t_utc_ns=stamp + number * 3_600_000_000_000,
            n_frames=9,
            n_bias_frames=9,
            bias_dn=bias,
            read_noise_dn=2.1,
            adc_bits=14,
            dark_dn=bias + 1.0,
        )
    return library


def expected_vignetting_percent(spec: LensSpec, radii_deg: Sequence[float]) -> list[float]:
    """The loss of the lens at each radius, in percent and negative, as the report shows it."""
    radii = np.asarray(radii_deg, dtype=np.float32)
    return [-100.0 * float(value) for value in spec.loss_at(radii)]


def rms(values: npt.NDArray[Any]) -> float:
    return float(math.sqrt(float(np.mean(np.square(values.astype(np.float64))))))
